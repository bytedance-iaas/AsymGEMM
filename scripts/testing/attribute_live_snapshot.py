#!/usr/bin/env python
"""Attribute a torch CUDA memory snapshot's live (active_allocated) blocks.

Pair with the after-forward snapshot dumped by lf_trace
(env ASYM_GEMM_LF_PROFILE_AFTER_FORWARD_SNAPSHOT_PATH) to find exactly what holds the
forward->backward-boundary resident HBM -- including saved-activations and offload-staging
buffers that the live-module-output detail (`live_activation_detail_rows`) does NOT capture.

Requires the snapshot to have been recorded with python stacks
(ASYM_GEMM_LF_PROFILE_MEMORY_SNAPSHOT=1), else blocks have no frames to attribute.
"""
from __future__ import annotations

import argparse
import collections
import pickle
from pathlib import Path

_TORCH_MARKERS = (
    "site-packages/torch/",
    "/torch/cuda/",
    "/torch/_",
    "/torch/autograd/",
    "/torch/nn/modules/module.py",
    "CUDACachingAllocator",
    "c10/",
)

# Keyword -> component, checked in order (most specific first).
_COMPONENTS = [
    ("input_layernorm", "norms.input_layernorm"),
    ("post_attention_layernorm", "norms.post_attn_layernorm"),
    ("decoder_activation_offload", "layer_act.offload"),
    ("decoder_layer_glue_gc", "layer_gc"),
    ("attention_activation_offload", "attention.offload"),
    ("o_proj", "attention.o_proj"),
    ("q_proj", "attention.qkv"),
    ("k_proj", "attention.qkv"),
    ("v_proj", "attention.qkv"),
    ("self_attn", "attention"),
    ("down_base", "experts.down_base"),
    ("gate_up_base", "experts.gate_up_base"),
    ("activation_offload", "experts.offload_stage"),
    ("llama4_experts", "experts"),
    ("llama4_moe", "experts.moe"),
    ("shared_expert", "shared_experts"),
    ("shared_mlp", "shared_experts"),
    ("lm_head", "lm_head"),
    ("cross_entropy", "loss"),
    ("forward_loss", "loss"),
    ("compute_loss", "loss"),
    ("embed", "embed_tokens"),
    ("rms_norm", "norms"),
    ("norm", "norms"),
]


def _frames(blk: dict, seg: dict) -> list:
    return blk.get("frames") or seg.get("frames") or []


def _user_frame(frames: list) -> str:
    # Prefer a frame in model / training / asym code.
    for fr in frames:
        fn = str(fr.get("filename", ""))
        if any(s in fn for s in ("asym_gemm/", "transformers/models/", "llamafactory/")):
            return f"{Path(fn).name}:{fr.get('line', '?')}:{fr.get('name', '?')}"
    # Else first non-torch-internal frame.
    for fr in frames:
        fn = str(fr.get("filename", ""))
        if not any(m in fn for m in _TORCH_MARKERS):
            return f"{Path(fn).name}:{fr.get('line', '?')}:{fr.get('name', '?')}"
    if frames:
        fr = frames[0]
        return f"{Path(str(fr.get('filename', '?'))).name}:{fr.get('line', '?')}:{fr.get('name', '?')}"
    return "<no-frames>"


def _component(frames: list) -> str:
    text = " ".join(f"{fr.get('filename', '')}:{fr.get('name', '')}" for fr in frames).lower()
    for key, comp in _COMPONENTS:
        if key in text:
            return comp
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--min-mib", type=float, default=20.0)
    args = ap.parse_args()

    with args.snapshot.open("rb") as fh:
        snap = pickle.load(fh)

    by_frame: dict[str, int] = collections.defaultdict(int)
    by_comp: dict[str, int] = collections.defaultdict(int)
    total = 0
    nblk = 0
    no_frame_bytes = 0
    for seg in snap.get("segments", []):
        for blk in seg.get("blocks", []):
            if blk.get("state") != "active_allocated":
                continue
            sz = int(blk.get("requested_size") or blk.get("size") or 0)
            frames = _frames(blk, seg)
            total += sz
            nblk += 1
            if not frames:
                no_frame_bytes += sz
            by_frame[_user_frame(frames)] += sz
            by_comp[_component(frames)] += sz

    print(f"snapshot: {args.snapshot}")
    print(f"live active_allocated: {total / 2**20:,.0f} MiB across {nblk} blocks")
    if no_frame_bytes:
        print(f"  (no-frame blocks: {no_frame_bytes / 2**20:,.0f} MiB -- enable ASYM_GEMM_LF_PROFILE_MEMORY_SNAPSHOT=1)")
    print("\n=== by component (heuristic) ===")
    for comp, b in sorted(by_comp.items(), key=lambda x: -x[1]):
        if b / 2**20 < args.min_mib:
            continue
        print(f"  {b / 2**20:9,.0f} MiB  {comp}")
    print("\n=== by allocation frame (top) ===")
    for frame, b in sorted(by_frame.items(), key=lambda x: -x[1])[: args.top]:
        if b / 2**20 < args.min_mib:
            continue
        print(f"  {b / 2**20:9,.0f} MiB  {frame}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
