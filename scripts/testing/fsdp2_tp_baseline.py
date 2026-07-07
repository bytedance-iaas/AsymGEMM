#!/usr/bin/env python3
"""DEV pace car #2 (gb200_tp.md P-DEV): FSDP2(CPUOffloadPolicy) x TP-2 + LoRA on Qwen3-32B.

A GOOD-FAITH strong external baseline (not a strawman):
  - REAL tensor parallelism: q/k/v/gate/up column-sharded (attention runs on half the
    heads per rank), o/down row-sharded with ONE all-reduce per region output; Megatron
    f/g duality implemented with torch.distributed autograd Functions.
  - LoRA sharded to match (col layers: A replicated + B row-sliced -> local add, no comm;
    row layers: A col-sliced consuming the local input shard, scale-before-B, partial adds
    BEFORE the region all-reduce -> one comm covers base+LoRA). Replicated-piece LoRA grads
    (col-A, row-B) are SUM-all-reduced post-backward.
  - Frozen base shards live in pinned CPU DRAM via FSDP2 fully_shard(CPUOffloadPolicy) on
    each base Linear over a 1-rank mesh (pure per-module H2D gather/free staging - the
    "TP-Staged" posture). LoRA + optimizer stay on-GPU (adapter-scale).
  - Both ranks consume the SAME synthetic batch (TP semantics; b8 global). Loss is a real
    CE over trained logits; comparability rows use step_s / step_H (synthetic tokens make
    the loss value itself non-comparable to LF-dataset rows - recorded in the shim).

Launch (through the harness guards is not applicable - this is torchrun standalone, so it
carries its own oom_score_adj + assumes the caller watches host memory; host footprint is
~32 GB pinned per rank):
  torchrun --nproc_per_node 2 scripts/testing/fsdp2_tp_baseline.py \
      --model Qwen/Qwen3-32B --seq 20000 --batch 8 --steps 3 --warmup 1 \
      --out profiling_gb200tp_p0/fsdp2_tp_baseline.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def rank0(*args) -> None:
    if dist.get_rank() == 0:
        print("[fsdp2_tp]", *args, flush=True)


# ---- Megatron f/g duality over the tp group ----
class RegionEntry(torch.autograd.Function):
    """f-operator: fwd identity, bwd all-reduce(SUM) of the accumulated local dX."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return grad


class RegionReduce(torch.autograd.Function):
    """g-operator: fwd all-reduce(SUM) of partial outputs, bwd identity."""

    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


class ShardedLoraLinear(nn.Module):
    """One TP-sharded frozen Linear + sharded LoRA.

    kind == "col": weight shard [N/2, K]; A [r,K] replicated, B [N/2, r] sliced.
    kind == "row": weight shard [N, K/2]; A [r, K/2] sliced, B [N, r] replicated;
                   forward consumes the LOCAL K/2 input slice and returns a PARTIAL [M,N].
    """

    def __init__(self, weight_shard: torch.Tensor, kind: str, lora_rank: int, lora_alpha: float, dtype=torch.bfloat16):
        super().__init__()
        self.kind = kind
        self.base = nn.Linear(weight_shard.shape[1], weight_shard.shape[0], bias=False, dtype=dtype, device="meta")
        self.base.weight = nn.Parameter(weight_shard, requires_grad=False)
        self.scale = lora_alpha / lora_rank
        n_out, k_in = weight_shard.shape
        device = torch.device("cuda", torch.cuda.current_device())
        self.lora_a = nn.Parameter(torch.randn(lora_rank, k_in, dtype=dtype, device=device) * (1.0 / max(k_in, 1)) ** 0.5)
        self.lora_b = nn.Parameter(torch.zeros(n_out, lora_rank, dtype=dtype, device=device))
        # replicated-piece grads need a post-backward SUM all-reduce:
        self.replicated_lora_params = [self.lora_a] if kind == "col" else [self.lora_b]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        s = F.linear(x, self.lora_a) * self.scale
        return y + F.linear(s, self.lora_b)


class TpDecoderLayer(nn.Module):
    def __init__(self, cfg, layer_state: dict, tp_rank: int, tp_size: int, lora_rank: int, lora_alpha: float):
        super().__init__()
        hidden = cfg.hidden_size
        heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        inter = cfg.intermediate_size
        assert heads % tp_size == 0 and kv_heads % tp_size == 0 and inter % tp_size == 0
        self.heads_l = heads // tp_size
        self.kv_l = kv_heads // tp_size
        self.hd = hd
        self.tp_rank = tp_rank

        def col(name, n_total):
            w = layer_state[name]
            shard = w[tp_rank * (n_total // tp_size):(tp_rank + 1) * (n_total // tp_size)].contiguous()
            return ShardedLoraLinear(shard, "col", lora_rank, lora_alpha)

        def row(name, k_total):
            w = layer_state[name]
            shard = w[:, tp_rank * (k_total // tp_size):(tp_rank + 1) * (k_total // tp_size)].contiguous()
            return ShardedLoraLinear(shard, "row", lora_rank, lora_alpha)

        self.q = col("self_attn.q_proj.weight", heads * hd)
        self.k = col("self_attn.k_proj.weight", kv_heads * hd)
        self.v = col("self_attn.v_proj.weight", kv_heads * hd)
        self.o = row("self_attn.o_proj.weight", heads * hd)
        self.gate = col("mlp.gate_proj.weight", inter)
        self.up = col("mlp.up_proj.weight", inter)
        self.down = row("mlp.down_proj.weight", inter)
        eps = cfg.rms_norm_eps
        self.in_norm = nn.RMSNorm(hidden, eps=eps, dtype=torch.bfloat16, device="cuda")
        self.post_norm = nn.RMSNorm(hidden, eps=eps, dtype=torch.bfloat16, device="cuda")
        self.in_norm.weight = nn.Parameter(layer_state["input_layernorm.weight"].cuda(), requires_grad=False)
        self.post_norm.weight = nn.Parameter(layer_state["post_attention_layernorm.weight"].cuda(), requires_grad=False)
        self.q_norm = nn.RMSNorm(hd, eps=eps, dtype=torch.bfloat16, device="cuda")
        self.k_norm = nn.RMSNorm(hd, eps=eps, dtype=torch.bfloat16, device="cuda")
        self.q_norm.weight = nn.Parameter(layer_state["self_attn.q_norm.weight"].cuda(), requires_grad=False)
        self.k_norm.weight = nn.Parameter(layer_state["self_attn.k_norm.weight"].cuda(), requires_grad=False)

    def forward(self, x, cos, sin):
        b, s, h = x.shape
        n = RegionEntry.apply(self.in_norm(x))
        flat = n.view(b * s, h)
        q = self.q(flat).view(b, s, self.heads_l, self.hd)
        k = self.k(flat).view(b, s, self.kv_l, self.hd)
        v = self.v(flat).view(b, s, self.kv_l, self.hd)
        q, k = self.q_norm(q), self.k_norm(k)
        q = (q * cos.unsqueeze(2) + _rot(q) * sin.unsqueeze(2))
        k = (k * cos.unsqueeze(2) + _rot(k) * sin.unsqueeze(2))
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        a = a.transpose(1, 2).reshape(b * s, self.heads_l * self.hd)
        p = RegionReduce.apply(self.o(a))  # base partial + row-LoRA partial -> ONE all-reduce
        x = x + p.view(b, s, h)
        n2 = RegionEntry.apply(self.post_norm(x))
        flat2 = n2.view(b * s, h)
        act = F.silu(self.gate(flat2)) * self.up(flat2)
        q2 = RegionReduce.apply(self.down(act))
        return x + q2.view(b, s, h)


def _rot(t):
    half = t.shape[-1] // 2
    return torch.cat([-t[..., half:], t[..., :half]], dim=-1)


class TpModel(nn.Module):
    def __init__(self, cfg, state, tp_rank, tp_size, lora_rank, lora_alpha, checkpoint_layers=True):
        super().__init__()
        self.cfg = cfg
        self.checkpoint_layers = checkpoint_layers
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=torch.bfloat16, device="cuda")
        self.embed.weight = nn.Parameter(state["model.embed_tokens.weight"].cuda(), requires_grad=False)
        self.layers = nn.ModuleList()
        for i in range(cfg.num_hidden_layers):
            prefix = f"model.layers.{i}."
            layer_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            self.layers.append(TpDecoderLayer(cfg, layer_state, tp_rank, tp_size, lora_rank, lora_alpha))
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=torch.bfloat16, device="cuda")
        self.norm.weight = nn.Parameter(state["model.norm.weight"].cuda(), requires_grad=False)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, dtype=torch.bfloat16, device="meta")
        head_key = "lm_head.weight" if "lm_head.weight" in state else "model.embed_tokens.weight"
        self.lm_head.weight = nn.Parameter(state[head_key], requires_grad=False)
        rope_params = getattr(cfg, "rope_parameters", None) or {}
        theta = rope_params.get("rope_theta", getattr(cfg, "rope_theta", 1e6))
        inv = 1.0 / (theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32, device="cuda") / cfg.head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    save_on_cpu_ctx = None  # set by main() when --save-on-cpu

    def forward(self, input_ids, labels):
        b, s = input_ids.shape
        x = self.embed(input_ids)
        pos = torch.arange(s, device=x.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos().to(x.dtype)[None], emb.sin().to(x.dtype)[None]
        import contextlib

        saved_ctx = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if self.save_on_cpu_ctx
            else contextlib.nullcontext()
        )
        with saved_ctx:
            for layer in self.layers:
                if self.checkpoint_layers:
                    x = torch.utils.checkpoint.checkpoint(layer, x, cos, sin, use_reentrant=False)
                else:
                    x = layer(x, cos, sin)
        x = self.norm(x)
        # chunked CE to avoid a [b*s, V] logits blowup
        flat = x.view(b * s, -1)
        flat_labels = labels.view(-1)
        total = torch.zeros((), device=x.device, dtype=torch.float32)
        count = 0
        chunk = 8192
        for i in range(0, flat.shape[0], chunk):
            logits = self.lm_head(flat[i:i + chunk]).float()
            piece_labels = flat_labels[i + 1:i + chunk + 1]
            piece_logits = logits[: piece_labels.shape[0]]
            if piece_labels.numel():
                total = total + F.cross_entropy(piece_logits, piece_labels, reduction="sum")
                count += piece_labels.numel()
        return total / max(count, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--seq", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=8, help="GLOBAL batch (TP: same batch on both ranks)")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="fsdp2_tp_baseline.json")
    parser.add_argument("--no-cpu-offload", action="store_true")
    parser.add_argument("--save-on-cpu", action="store_true",
                        help="wrap the layer stack in torch.autograd.graph.save_on_cpu(pin_memory=True) — "
                             "the standard-PyTorch analog of activation offload; without it the "
                             "checkpoint inputs alone (64 x [M,H]) exceed HBM at s20000 b8")
    args = parser.parse_args()

    try:
        Path("/proc/self/oom_score_adj").write_text("1000")
    except OSError:
        pass

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(42)

    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    rank0(f"loading {args.model} state on CPU (bf16)...")
    t_load = time.perf_counter()
    hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True)
    state = hf.state_dict()
    rank0(f"state loaded in {time.perf_counter() - t_load:.1f}s")

    model = TpModel(cfg, state, rank, dist.get_world_size(), args.lora_rank, args.lora_alpha)
    model.save_on_cpu_ctx = bool(args.save_on_cpu)
    del hf, state

    # FSDP2 CPU offload for the frozen base shards + lm_head/embed (whole-unit staging)
    if not args.no_cpu_offload:
        from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy
        from torch.distributed.device_mesh import DeviceMesh

        # per-rank SOLO mesh (world-size-1 FSDP = pure CPU-offload staging machinery);
        # new_group must be called by ALL ranks for EACH group.
        solo_groups = [dist.new_group([r]) for r in range(dist.get_world_size())]
        solo_mesh = DeviceMesh.from_group(solo_groups[rank], "cuda")
        wrapped = 0
        for layer in model.layers:
            for module in (layer.q, layer.k, layer.v, layer.o, layer.gate, layer.up, layer.down):
                fully_shard(module.base, mesh=solo_mesh, offload_policy=CPUOffloadPolicy())
                wrapped += 1
        fully_shard(model.lm_head, mesh=solo_mesh, offload_policy=CPUOffloadPolicy())
        wrapped += 1
        rank0(f"fully_shard(CPUOffloadPolicy) on {wrapped} frozen units")

    lora_params = [p for p in model.parameters() if p.requires_grad]
    replicated = [p for layer in model.layers for m in (layer.q, layer.k, layer.v, layer.o, layer.gate, layer.up, layer.down) for p in m.replicated_lora_params]
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)
    rank0(f"trainable LoRA params: {sum(p.numel() for p in lora_params):,}")

    vocab = cfg.vocab_size
    gen = torch.Generator(device="cpu").manual_seed(1234)
    input_ids = torch.randint(0, vocab, (args.batch, args.seq), generator=gen).cuda()
    labels = input_ids.clone()

    records = []
    for step in range(args.warmup + args.steps):
        torch.cuda.reset_peak_memory_stats()
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = model(input_ids, labels)
        t_fwd = time.perf_counter()
        loss.backward()
        for p in replicated:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        t_bwd = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dist.barrier()
        t_end = time.perf_counter()
        rec = {
            "step": step,
            "is_warmup": step < args.warmup,
            "loss": float(loss.item()),
            "forward_milliseconds": (t_fwd - t0) * 1e3,
            "backward_milliseconds": (t_bwd - t_fwd) * 1e3,
            "optimizer_milliseconds": (t_end - t_bwd) * 1e3,
            "step_milliseconds": (t_end - t0) * 1e3,
            "peak_allocated_hbm_bytes": int(torch.cuda.max_memory_allocated()),
        }
        records.append(rec)
        rank0(f"step {step} loss={rec['loss']:.4f} step={rec['step_milliseconds']/1e3:.1f}s "
              f"(fwd {rec['forward_milliseconds']/1e3:.1f} bwd {rec['backward_milliseconds']/1e3:.1f} opt {rec['optimizer_milliseconds']/1e3:.1f}) "
              f"peakH={rec['peak_allocated_hbm_bytes']/2**30:.1f}GiB")

    rss_peak = 0
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            rss_peak = int(line.split()[1]) * 1024
    measured = [r for r in records if not r["is_warmup"]]
    shim = {
        "backend": "fsdp2_tp2_cpuoffload_lora",
        "note": "external DEV pace car; REAL head-split TP-2 + FSDP2 CPUOffload frozen base; synthetic tokens (loss value not comparable to LF-dataset rows)",
        "world_size": dist.get_world_size(),
        "rank": rank,
        "config": {"model": args.model, "seq_len": args.seq, "global_batch_size": args.batch,
                   "lora_rank": args.lora_rank, "cpu_offload": not args.no_cpu_offload},
        "step_samples": {"rows": records},
        "trainer": {"timing": {
            "measured_steps": len(measured),
            "measured_e2e_step_milliseconds": sum(r["step_milliseconds"] for r in measured),
        }, "losses": [{"measured_step": r["step"], "loss": r["loss"], "is_warmup": r["is_warmup"]} for r in records]},
        "memory": {"gpu": {"peak_allocated_hbm_bytes": max(r["peak_allocated_hbm_bytes"] for r in records)},
                    "process": {"rss_peak_bytes": rss_peak}},
    }
    out = Path(args.out)
    out = out.with_name(f"{out.stem}_rank{rank}{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(shim, indent=2, sort_keys=True) + "\n")
    print(f"[fsdp2_tp] rank{rank} wrote {out}")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
