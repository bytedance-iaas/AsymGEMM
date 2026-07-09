#!/usr/bin/env python3
"""Offline INT8 weight converter for the AsymGEMM *unified* MoE backend.

The unified MoE path (``asym_gemm.unified_moe.Layer``) normally quantizes the
BF16 expert masters to per-channel INT8 at model-load time, via a per-expert
Python loop over every MoE layer. For a large MoE (e.g. 48 layers x 128 experts)
that conversion dominates startup. This script performs the *same* quantization
once, offline, and writes the result to disk so the server can load it directly
(see SGLANG_ASYMGEMM_UNIFIED_INT8_PATH in sglang).

Correctness: the per-projection quantization calls the very same
``quantize_per_channel_int8`` used by ``Layer.from_bf16`` (imported from
``asym_gemm.unified_moe.runtime``), so an offline-converted checkpoint is
byte-identical to what online quantization would have produced. The server-side
loader (``Layer.from_int8``) only rebuilds the small device SFB broadcasts.

Output layout (one file per MoE decoder layer, keyed by the decoder layer index,
which equals sglang's ``layer.layer_id``):

    <output>/layer_{L}.safetensors   # 6 tensors per layer:
        gate_int8   [G, inter, hidden]  int8      up_int8   [G, inter, hidden]  int8
        gate_scales [G, inter]          float32   up_scales [G, inter]          float32
        down_int8   [G, hidden, inter]  int8      down_scales [G, hidden]       float32
    <output>/asym_int8_manifest.json  # model + quant metadata, converted layers
    <output>/config.json              # copy of the source config (for reference)

The gate/up split follows sglang's silu_and_mul convention (w13 first half =
gate, second half = up), matching ``asym_gemm_unified.maybe_create_...``.

Example:
    python scripts/convert_int8_weights.py \
        --input-path /data00/models/Qwen3-Coder-30B-A3B-Instruct \
        --output     /data00/models/Qwen3-Coder-30B-A3B-Instruct-asymint8
"""

import argparse
import gc
import json
import os
import shutil
import time
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Single source of truth for the quantization contract — identical bytes to the
# online path (asym_gemm.unified_moe.Layer.from_bf16).
from asym_gemm.unified_moe.runtime import GRAN_K, quantize_per_channel_int8

MANIFEST_NAME = "asym_int8_manifest.json"
FORMAT_VERSION = 1

# FP8 storage dtypes we know how to dequantize (block-scaled, DeepSeek/Qwen style).
FP8_DTYPES = tuple(
    dt for dt in (getattr(torch, "float8_e4m3fn", None),
                  getattr(torch, "float8_e5m2", None)) if dt is not None
)


def dequant_fp8_blockwise(
    w_fp8: torch.Tensor, scale_inv: torch.Tensor, block: List[int]
) -> torch.Tensor:
    """Block-scaled FP8 -> BF16, mirroring KTransformers' ``weight_dequant``.

    The checkpoint stores an FP8 (e4m3) weight plus one FP32 scale per
    ``block = [bh, bw]`` tile in ``weight_scale_inv``; dequant is simply
    ``bf16 = fp8.to(fp32) * scale`` with the scale broadcast over its tile
    (``scale_inv`` is the multiply factor despite the "inv" name). Handles a
    ragged last block via ceil-shaped scale + crop.
    """
    bh, bw = block
    H, W = w_fp8.shape
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(bh, dim=0).repeat_interleave(bw, dim=1)[:H, :W]
    return (w_fp8.to(torch.float32) * s).to(torch.bfloat16)


def load_model_config(input_path: str) -> Dict:
    """Read the fields the unified layer needs from the HF config.json."""
    config_path = os.path.join(input_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {input_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    text_cfg = config.get("text_config", config)

    num_experts = text_cfg.get("num_experts", text_cfg.get("n_routed_experts"))
    hidden = text_cfg.get("hidden_size")
    inter = text_cfg.get("moe_intermediate_size", text_cfg.get("intermediate_size"))
    top_k = text_cfg.get("num_experts_per_tok", 2)
    missing = [k for k, v in {
        "num_experts": num_experts, "hidden_size": hidden,
        "moe_intermediate_size": inter,
    }.items() if v is None]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    if hidden % GRAN_K != 0 or inter % GRAN_K != 0:
        raise ValueError(
            f"hidden ({hidden}) and moe_intermediate_size ({inter}) must both "
            f"be multiples of {GRAN_K} for the SM90 INT8 kernel"
        )

    # Block-scaled FP8 checkpoints carry a quantization_config with the tile size.
    quant_config = config.get("quantization_config") or text_cfg.get("quantization_config")
    fp8_block = None
    if quant_config is not None:
        wbs = quant_config.get("weight_block_size")
        if wbs is not None:
            if not (isinstance(wbs, list) and len(wbs) == 2):
                raise ValueError(f"Invalid weight_block_size {wbs}; expected [bh, bw]")
            fp8_block = [int(wbs[0]), int(wbs[1])]

    return {
        "model_type": config.get("model_type"),
        "num_experts": int(num_experts),
        "num_experts_per_tok": int(top_k),
        "hidden_size": int(hidden),
        "moe_intermediate_size": int(inter),
        "quant_method": (quant_config or {}).get("quant_method"),
        "fp8_weight_block_size": fp8_block,
    }


class Int8ExpertConverter:
    """Walk an HF MoE checkpoint and emit per-layer INT8 expert weights."""

    def __init__(self, input_path: str, output_path: str, model_config: Dict,
                 device: str, input_type: str = "auto"):
        self.input_path = input_path
        self.output_path = output_path
        self.model_config = model_config
        self.device = device
        self.input_type = input_type            # 'auto' | 'bf16' | 'fp16' | 'fp8'
        self.fp8_block = model_config.get("fp8_weight_block_size")
        self.num_experts = model_config["num_experts"]
        self.hidden = model_config["hidden_size"]
        self.inter = model_config["moe_intermediate_size"]
        self.layout = "base"  # or "fused"

        self.tensor_file_map: Dict[str, str] = {}   # normalized key -> filename
        self.tensor_key_map: Dict[str, str] = {}    # normalized key -> original key
        self.file_handle_map: Dict[str, object] = {}
        self._load_input_files()
        self._resolve_input_type()

    def _resolve_input_type(self):
        """Pin down the source dtype (auto-detect from a sample expert weight)
        and validate the FP8 tile size when the source is block-scaled FP8."""
        if self.input_type == "auto":
            sample = next(
                (k for k in self.tensor_file_map
                 if ".mlp.experts." in k and k.endswith(".weight")
                 and "scale" not in k),
                None,
            )
            dt = self._load(sample).dtype if sample is not None else torch.bfloat16
            if dt in FP8_DTYPES:
                self.input_type = "fp8"
            elif dt == torch.float16:
                self.input_type = "fp16"
            else:
                self.input_type = "bf16"
            print(f"  auto-detected source dtype {dt} -> input_type={self.input_type}")

        if self.input_type == "fp8":
            if not FP8_DTYPES:
                raise RuntimeError("this torch build has no float8 dtypes")
            if self.fp8_block is None:
                self.fp8_block = [GRAN_K, GRAN_K]
                print(f"  WARNING: no weight_block_size in config; assuming {self.fp8_block}")
            print(f"  FP8 dequant enabled (block={self.fp8_block})")

    # -- input -------------------------------------------------------------
    def _load_input_files(self):
        print(f"Loading safetensors from {self.input_path}")
        found = False
        for root, _, files in os.walk(self.input_path):
            for file in sorted(files):
                if not file.endswith(".safetensors"):
                    continue
                found = True
                handle = safe_open(os.path.join(root, file), framework="pt")
                self.file_handle_map[file] = handle
                for key in handle.keys():
                    norm = key.replace("language_model.", "")
                    self.tensor_key_map[norm] = key
                    self.tensor_file_map[norm] = file
        if not found:
            raise FileNotFoundError(f"No safetensors files found in {self.input_path}")
        print(f"  indexed {len(self.tensor_file_map)} tensors")

    def _load(self, key: str) -> torch.Tensor:
        file = self.tensor_file_map[key]
        return self.file_handle_map[file].get_tensor(self.tensor_key_map.get(key, key))

    def _has(self, key: str) -> bool:
        return key in self.tensor_file_map

    # -- layer discovery ---------------------------------------------------
    def find_expert_layers(self) -> Dict[int, List[int]]:
        """Return {layer_idx: [expert_ids...]}. Detects base vs fused layout."""
        for key in self.tensor_file_map:
            if ".mlp.experts." in key and "gate_up" in key:
                self.layout = "fused"
                break

        if self.layout == "fused":
            layers = set()
            for key in self.tensor_file_map:
                if "model.layers." in key and ".mlp.experts." in key:
                    layers.add(int(key.split(".")[2]))
            print(f"Found {len(layers)} fused-MoE layers")
            return {L: [-1] for L in sorted(layers)}

        layers: Dict[int, set] = defaultdict(set)
        for key in self.tensor_file_map:
            if "model.layers." in key and ".mlp.experts." in key:
                parts = key.split(".")
                layers[int(parts[2])].add(int(parts[5]))
        result = {L: sorted(e) for L, e in layers.items()}
        print(f"Found {len(result)} MoE layers "
              f"({self.num_experts} experts each expected)")
        return result

    # -- per-layer bf16 gathering -----------------------------------------
    def _load_proj_bf16(self, base: str, proj: str) -> torch.Tensor:
        """Load one expert projection as BF16, dequantizing block-scaled FP8
        (weight + weight_scale_inv) when the source is FP8."""
        w = self._load(f"{base}.{proj}.weight")
        if self.input_type == "fp8" or w.dtype in FP8_DTYPES:
            sk = f"{base}.{proj}.weight_scale_inv"
            if not self._has(sk):
                raise KeyError(
                    f"FP8 weight {base}.{proj}.weight has no matching "
                    f"weight_scale_inv (needed for block-scaled dequant)"
                )
            return dequant_fp8_blockwise(w, self._load(sk), self.fp8_block or [GRAN_K, GRAN_K])
        return w.to(torch.bfloat16)

    def _gather_layer_bf16(self, layer_idx: int, expert_ids: List[int]):
        """Return (gate, up, down) bf16 tensors:
        gate/up [G, inter, hidden], down [G, hidden, inter]."""
        if self.layout == "fused":
            if self.input_type == "fp8":
                raise NotImplementedError(
                    "FP8 + fused gate_up layout is not supported yet (matches "
                    "KTransformers, whose fused path is bf16/fp16 only)"
                )
            prefix = f"model.layers.{layer_idx}.mlp.experts."
            projs = sorted({k.split(".")[5] for k in self.tensor_file_map
                            if k.startswith(prefix)})
            # Expect a down-like and a gate_up-like fused tensor.
            gate_up_name = next(p for p in projs if "gate_up" in p)
            down_name = next(p for p in projs if "down" in p)
            gate_up = self._load(f"{prefix}{gate_up_name}").to(torch.bfloat16)  # [E,H,2I]
            down_fused = self._load(f"{prefix}{down_name}").to(torch.bfloat16)  # [E,I,H]
            E, H, twoI = gate_up.shape
            assert twoI == 2 * self.inter, f"unexpected fused 2I={twoI}"
            gate_up_t = gate_up.transpose(1, 2).contiguous()  # [E, 2I, H]
            gate = gate_up_t[:, : self.inter, :].contiguous()
            up = gate_up_t[:, self.inter:, :].contiguous()
            down = down_fused.transpose(1, 2).contiguous()    # [E, H, I]
            return gate, up, down

        gate_list, up_list, down_list = [], [], []
        for e in expert_ids:
            base = f"model.layers.{layer_idx}.mlp.experts.{e}"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                if not self._has(f"{base}.{proj}.weight"):
                    raise KeyError(f"Missing {proj} for layer {layer_idx} expert {e}")
            gate_list.append(self._load_proj_bf16(base, "gate_proj"))
            up_list.append(self._load_proj_bf16(base, "up_proj"))
            down_list.append(self._load_proj_bf16(base, "down_proj"))
        gate = torch.stack(gate_list, dim=0).contiguous()  # [G, inter, hidden]
        up = torch.stack(up_list, dim=0).contiguous()      # [G, inter, hidden]
        down = torch.stack(down_list, dim=0).contiguous()  # [G, hidden, inter]
        return gate, up, down

    def _quant_projection(self, w: torch.Tensor):
        """Quantize [G, N, K] bf16 -> (int8 [G, N, K], fp32 scales [G, N]).

        Reshapes to [G*N, K] and calls the exact same per-channel quantizer the
        online path uses, so each output row's amax/127 scale (and thus every
        byte) is identical to Layer.from_bf16."""
        G, N, K = w.shape
        w2d = w.reshape(G * N, K)
        if self.device != "cpu":
            w2d = w2d.to(self.device)
        q, s = quantize_per_channel_int8(w2d)  # numpy int8 [G*N,K], fp32 [G*N]
        q_t = torch.from_numpy(q).reshape(G, N, K).contiguous()
        s_t = torch.from_numpy(s).reshape(G, N).contiguous()
        return q_t, s_t

    def convert_layer(self, layer_idx: int, expert_ids: List[int]) -> Dict[str, torch.Tensor]:
        gate, up, down = self._gather_layer_bf16(layer_idx, expert_ids)
        G = gate.shape[0]
        if self.layout == "base" and G != self.num_experts:
            print(f"  WARNING layer {layer_idx}: found {G} experts, "
                  f"config says {self.num_experts}")
        assert gate.shape == (G, self.inter, self.hidden), \
            f"gate {tuple(gate.shape)} != {(G, self.inter, self.hidden)}"
        assert down.shape == (G, self.hidden, self.inter), \
            f"down {tuple(down.shape)} != {(G, self.hidden, self.inter)}"

        gate_int8, gate_s = self._quant_projection(gate)
        up_int8, up_s = self._quant_projection(up)
        down_int8, down_s = self._quant_projection(down)
        del gate, up, down
        gc.collect()
        if self.device != "cpu":
            torch.cuda.empty_cache()
        return {
            "gate_int8": gate_int8, "gate_scales": gate_s,
            "up_int8": up_int8, "up_scales": up_s,
            "down_int8": down_int8, "down_scales": down_s,
        }

    # -- driver ------------------------------------------------------------
    def convert(self, resume_layer: int = 0):
        os.makedirs(self.output_path, exist_ok=True)
        expert_layers = self.find_expert_layers()
        if not expert_layers:
            raise RuntimeError("No MoE expert layers found in input!")

        converted: List[int] = []
        for i, (layer_idx, expert_ids) in enumerate(sorted(expert_layers.items())):
            if layer_idx < resume_layer:
                continue
            t0 = time.time()
            tensors = self.convert_layer(layer_idx, expert_ids)
            out_file = os.path.join(self.output_path, f"layer_{layer_idx}.safetensors")
            save_file(tensors, out_file)
            converted.append(layer_idx)
            print(f"  [{i + 1}/{len(expert_layers)}] layer {layer_idx}: "
                  f"wrote {os.path.basename(out_file)} in {time.time() - t0:.2f}s")
            del tensors
            gc.collect()

        self._write_manifest(converted)
        self._copy_config()
        print(f"Done. Converted {len(converted)} layers to {self.output_path}")

    def _write_manifest(self, converted_layers: List[int]):
        manifest = {
            "format_version": FORMAT_VERSION,
            "producer": "AsymGEMM/scripts/convert_int8_weights.py",
            "source_model": os.path.abspath(self.input_path),
            "layout": self.layout,
            # keyed by decoder layer index == sglang layer.layer_id
            "layer_file_pattern": "layer_{layer_id}.safetensors",
            "converted_layers": converted_layers,
            "source_input_type": self.input_type,
            "fp8_weight_block_size": self.fp8_block if self.input_type == "fp8" else None,
            "quant": {
                "scheme": "per_channel_symmetric_int8",
                "scale": "amax/127",
                "gran_k": GRAN_K,
                "weight_dtype": "int8",
                "scale_dtype": "float32",
                "gate_up_split": "sglang silu_and_mul (w13[:inter]=gate, w13[inter:]=up)",
            },
            "model": self.model_config,
        }
        path = os.path.join(self.output_path, MANIFEST_NAME)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  wrote manifest: {path}")

    def _copy_config(self):
        src = os.path.join(self.input_path, "config.json")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(self.output_path, "config.json"))

    def close(self):
        self.file_handle_map.clear()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert MoE expert weights to INT8 for the AsymGEMM unified "
                    "MoE backend. BF16/FP16 input is quantized directly (bit-exact "
                    "with the online path); block-scaled FP8 is dequantized to BF16 "
                    "first (like KTransformers), then quantized.")
    parser.add_argument("--input-path", "-i", required=True,
                        help="HF model directory (BF16/FP16 or block-scaled FP8)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for INT8 expert weights")
    parser.add_argument("--input-type", choices=["auto", "bf16", "fp16", "fp8"],
                        default="auto",
                        help="Source weight dtype (default: auto-detect from the "
                             "checkpoint). 'fp8' dequantizes block-scaled FP8 "
                             "(needs weight_scale_inv + quantization_config."
                             "weight_block_size) to BF16 before quantizing.")
    parser.add_argument("--device", default="cpu",
                        help="Device for the quantization math (default: cpu). For "
                             "BF16/FP16 input CPU is bit-exact with the online path "
                             "(sglang quantizes on CPU) and about the same speed; "
                             "'cuda' differs by ~1 ULP on a tiny fraction of values. "
                             "(FP8 input has no online path to match.)")
    parser.add_argument("--resume-layer", type=int, default=0,
                        help="Resume conversion at this layer index (default: 0)")
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Error: input path does not exist: {args.input_path}")
        return 1

    print("Loading model configuration...")
    model_config = load_model_config(args.input_path)
    print(f"  {model_config}")

    converter = Int8ExpertConverter(
        args.input_path, args.output, model_config, args.device, args.input_type)
    try:
        converter.convert(resume_layer=args.resume_layer)
    finally:
        converter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
