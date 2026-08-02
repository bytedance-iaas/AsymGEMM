#!/usr/bin/env python3
"""M3 — per-module composition bench (motivation_v2_plots.md §2.4.2).

ONE transformer layer's modules of a Qwen3-32B-shaped dense model (d=5120,
inter=25600, 64 q heads / 8 kv heads, head_dim 128), driven standalone with
synthetic bf16 inputs the way the repo tests drive the wrappers, under the
SHIPPED per-layer checkpointing semantics (torch reentrant checkpoint: the
outer forward runs no-grad and saves only the module input; the backward
window recomputes with grad enabled and then differentiates — exactly where
the shipped offload/recompute machinery engages, cf. the keep-acts docstrings
"the wrapper runs inside the unsloth-GC recompute").

Three per-activation-type policies (fresh subprocess per policy so env pins,
the memoized placement policy, pinned pools and the CUDA context never leak
across policies):
  (a) recompute_all — pure checkpointing, no per-type offload:
        ASYMM_ATTN_ACT_KEEP_ACTS_HBM=1, ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM=1,
        no saved-tensor wrapper installed, qknorm/rope recompute off.
  (b) offload_all — every offloadable activation type to pinned CPU in the
        (recompute) forward, restaged in the backward:
        AttentionSavedTensorOffloadWrapper installed (default min_bytes 1MiB,
        bf16/fp16/fp32), per-projection U/S offload, MLP fine-grained offload
        of X/gate/up/act/S plus dgate/dup roundtrips; qknorm/rope OFF so the
        fp32 norm saves and the roped SDPA operands move as bytes.
  (c) composed — the shipped mix: ASYM_PLACEMENT_POLICY=1 on a dense-class
        model (P12 qknorm-recompute ON: the big fp32 norm chain is never
        saved, only the small bf16 norm parent is offloaded once; P12 rope
        ON: SDPA q/k operands saved as recompute recipes and rebuilt from the
        offloaded norm parent; P13 restage-prefetch ON: MLP dgate/dup kept
        on-GPU, gate/up prefetched), saved-tensor wrapper installed for the
        remaining classes (v, SDPA out, ...).

Metrics per (module, policy), forward segment and backward segment separately:
  * peak GPU memory: torch.cuda.max_memory_allocated, reset at each segment
    boundary (absolute; the segment-entry allocated baseline is recorded too).
  * link bytes: the runtime's own offload/restage copies, counted at the
    runtime chokepoints (ActivationOffloadManager.offload/stage/stage_rows/
    stage_begin/stage_concat_columns/record_cpu_ready, the attention
    saved-tensor wrapper's pack/unpack counters, qknorm _stage_fresh).
    In-place C2C kernel reads (streamed base weights, CPU-left U reads) are
    NOT copies and are reported separately/analytically in the json.
  * wall time: CUDA events around the segment + host sync (host-blocked gaps
    are included since the events bracket the whole segment), plus host wall.

Run (see HOW-TO in the task / container recipe):
  CUDA_VISIBLE_DEVICES=0 numactl --membind=0 .venv/bin/python \
      scripts/motivation_bench/bench_m3.py
Output: profiling_results/motivation/m3.json + a compact summary table.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "profiling_results/motivation/m3.json"
CHILD_DIR = Path(os.environ.get("ASYM_M3_CHILD_DIR", "/tmp/asym_m3"))

TOKENS_DEFAULT = 131072  # 128K rows, batch 1 (task: try 128K first)
HIDDEN = 5120
INTER = 25600
N_HEADS = 64
N_KV = 8
HEAD_DIM = 128
RANK = 64
ALPHA = 128.0
WARMUP = 1
REPS = 2  # measured reps (means reported over these)

POLICIES = ("recompute_all", "offload_all", "composed")

# env pins per policy — recorded verbatim in the json manifest.
COMMON_ENV = {
    # arm the shipped fine-grained dense MLP path (the dense LF stack's flag)
    "ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD": "1",
}
POLICY_ENV = {
    "recompute_all": {
        "ASYMM_ATTN_ACT_KEEP_ACTS_HBM": "1",
        "ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM": "1",
    },
    "offload_all": {},
    "composed": {
        "ASYM_PLACEMENT_POLICY": "1",
    },
}
INSTALL_WRAPPER = {"recompute_all": False, "offload_all": True, "composed": True}


# --------------------------------------------------------------------------
# child: one policy, both modules
# --------------------------------------------------------------------------

def _memdiag_peak_report(snapshot: dict, top_n: int = 18) -> list[str]:
    """Replay the CUDA memory-history event stream, find the instant of peak
    live bytes, and report the largest live blocks with their allocating frame
    (harness diagnosis for the M3 peak attribution; not part of the metrics)."""
    trace = snapshot["device_traces"][0]
    live: dict[int, tuple[int, tuple]] = {}
    cur = 0
    peak = -1
    peak_live: dict[int, tuple[int, tuple]] = {}
    for ev in trace:
        act = ev.get("action")
        addr = ev.get("addr")
        size = int(ev.get("size", 0))
        if act == "alloc":
            frames = tuple(
                f"{f.get('filename', '?').rsplit('/', 1)[-1]}:{f.get('line', 0)}:{f.get('name', '?')}"
                for f in (ev.get("frames") or [])
                if any(k in f.get("filename", "") for k in ("asym_gemm", "bench_m3", "transformers", "checkpoint", "functional", "_tensor", "qknorm"))
            )[:3]
            live[addr] = (size, frames)
            cur += size
            if cur > peak:
                peak = cur
                peak_live = dict(live)
        elif act in ("free_completed",):
            ent = live.pop(addr, None)
            if ent is not None:
                cur -= ent[0]
    lines = [f"peak live (traced allocs only) = {peak / 2**30:.2f} GiB across {len(peak_live)} blocks"]
    blocks = sorted(peak_live.values(), key=lambda t: -t[0])[:top_n]
    for size, frames in blocks:
        lines.append(f"  {size / 2**30:7.3f} GiB  {' <- '.join(frames) if frames else '(no tagged frame)'}")
    return lines


def _child(policy: str, tokens: int, out_path: str, memdiag: str | None = None) -> None:
    import gc

    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.checkpoint import checkpoint

    import asym_gemm  # noqa: F401 (extension must be importable)
    from asym_gemm.training import activation_offload as _ao
    from asym_gemm.training import pinned_ledger
    from asym_gemm.training import placement_policy
    from asym_gemm.training import qknorm_recompute as _qr
    from asym_gemm.training.activation_offload import ActivationOffloadManager
    from asym_gemm.training.attention_activation_offload import (
        AsymActivationOffloadLoRALinear,
        AttentionActivationOffloadContext,
        install_attention_saved_tensor_offload,
    )
    from asym_gemm.training.dense_mlp_finegrained import build_finegrained_dense_mlp
    from asym_gemm.training.frozen_linear import AsymExecutionStats
    from asym_gemm.training.host_weight import HostWeight
    from asym_gemm.training.qknorm_recompute import (
        install_qknorm_recompute,
        install_rope_recompute,
        qknorm_recompute_stats,
    )
    from transformers.models.qwen3 import modeling_qwen3 as _hf_qwen3
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda:0")

    # the fine-grained modules register this themselves at construction; do it
    # up front so every policy-consulting gate (rope P12 needs dense-class)
    # sees the model class the real Qwen3-32B run would have registered.
    placement_policy.register_model_class("dense")

    # ---------------- bench-side link-byte ledger (counts the runtime's own
    # explicit offload/restage copies at the runtime chokepoints) ------------
    LED = {"d2h": 0, "h2d": 0}
    BY_TAG: dict[str, list[int]] = {}

    def _led(kind: str, tag: str, nbytes: int) -> None:
        LED[kind] += int(nbytes)
        ent = BY_TAG.setdefault(tag, [0, 0])
        ent[0 if kind == "d2h" else 1] += int(nbytes)

    def install_ledger() -> None:
        M = ActivationOffloadManager
        orig_offload = M.offload

        def offload(self, tensor, tag):
            was_cuda = tensor.device.type == "cuda"
            h = orig_offload(self, tensor, tag)
            if was_cuda:
                _led("d2h", tag, h.nbytes)
            return h

        M.offload = offload

        orig_rcr = M.record_cpu_ready

        def record_cpu_ready(self, handle):
            orig_rcr(self, handle)
            _led("d2h", handle.tag, handle.nbytes)  # direct row-write D2H paths

        M.record_cpu_ready = record_cpu_ready

        orig_stage = M.stage

        def stage(self, handle, *, tag=None, mutable=True):
            out = orig_stage(self, handle, tag=tag, mutable=mutable)
            if handle.tensor.device.type == "cpu":
                _led("h2d", tag or handle.tag, handle.nbytes)
            return out

        M.stage = stage

        orig_rows = M.stage_rows

        def stage_rows(self, handle, start, end, *, tag=None):
            out = orig_rows(self, handle, start, end, tag=tag)
            if handle.tensor.device.type == "cpu":
                per_row = handle.nbytes // max(1, int(handle.tensor.shape[0]))
                _led("h2d", tag or handle.tag, (int(end) - int(start)) * per_row)
            return out

        M.stage_rows = stage_rows

        orig_begin = M.stage_begin

        def stage_begin(self, handle, *, tag=None):
            out = orig_begin(self, handle, tag=tag)
            if handle.tensor.device.type == "cpu":
                _led("h2d", f"prefetch.{tag or handle.tag}", handle.nbytes)
            return out

        M.stage_begin = stage_begin

        orig_cat = M.stage_concat_columns

        def stage_concat_columns(self, left, right, *, tag):
            out = orig_cat(self, left, right, tag=tag)
            _led("h2d", tag, left.nbytes + right.nbytes)
            return out

        M.stage_concat_columns = stage_concat_columns

        orig_sf = _qr._stage_fresh

        def _stage_fresh(handle, manager):
            _led("h2d", f"qknorm_stage.{handle.tag}", handle.nbytes)
            return orig_sf(handle, manager)

        _qr._stage_fresh = _stage_fresh

    install_ledger()

    wrappers: list = []  # attention saved-tensor wrappers (counter snapshots)

    def ledger_snap() -> dict:
        w_d2h = sum(w.offloaded_bytes for w in wrappers)
        w_h2d = sum(sum(w.stage_bytes_by_tag.values()) for w in wrappers)
        return {
            "d2h": LED["d2h"] + w_d2h,
            "h2d": LED["h2d"] + w_h2d,
            "by_tag": {k: list(v) for k, v in BY_TAG.items()},
            "wrapper_by_tag_d2h": {
                k: v for w in wrappers for k, v in w.offload_bytes_by_tag.items()
            },
            "wrapper_by_tag_h2d": {
                k: v for w in wrappers for k, v in w.stage_bytes_by_tag.items()
            },
        }

    def ledger_delta(a: dict, b: dict) -> dict:
        tags = {}
        for k, v in b["by_tag"].items():
            v0 = a["by_tag"].get(k, [0, 0])
            d = [v[0] - v0[0], v[1] - v0[1]]
            if d[0] or d[1]:
                tags[k] = d
        for side, key in (("wrapper_by_tag_d2h", 0), ("wrapper_by_tag_h2d", 1)):
            for k, v in b[side].items():
                d = v - a[side].get(k, 0)
                if d:
                    tags.setdefault(f"saved_wrapper.{k}", [0, 0])[key] += d
        return {
            "d2h_bytes": b["d2h"] - a["d2h"],
            "h2d_bytes": b["h2d"] - a["h2d"],
            "by_tag_bytes_d2h_h2d": tags,
        }

    # ---------------- modules -------------------------------------------------
    torch.manual_seed(20260726)

    def _host_weight(out_f: int, in_f: int) -> HostWeight:
        w = torch.empty(out_f, in_f, dtype=torch.bfloat16)
        w.normal_(0.0, 0.02)
        return HostWeight(w, pin_memory=True, clone=False)

    attn_stats = AsymExecutionStats()
    attn_ctx = AttentionActivationOffloadContext()

    def _attn_proj(out_f: int, in_f: int, role: str) -> AsymActivationOffloadLoRALinear:
        return AsymActivationOffloadLoRALinear.from_host_weight(
            _host_weight(out_f, in_f),
            rank=RANK,
            alpha=ALPHA,
            backend="asym",
            stats=attn_stats,
            device=dev,
            lora_dtype=torch.bfloat16,
            init_lora_weights="peft",
            lora_dropout=0.0,
            projection_role=role,
            attention_context=attn_ctx,
        )

    class BenchAttention(nn.Module):
        """Qwen3-style attention: shipped CPU-homed LoRA projections, q/k-norm,
        rope (the transformers function, resolved at call time so the shipped
        install_rope_recompute monkeypatch applies), SDPA with GQA."""

        def __init__(self) -> None:
            super().__init__()
            self.q_proj = _attn_proj(N_HEADS * HEAD_DIM, HIDDEN, "q_proj")
            self.k_proj = _attn_proj(N_KV * HEAD_DIM, HIDDEN, "k_proj")
            self.v_proj = _attn_proj(N_KV * HEAD_DIM, HIDDEN, "v_proj")
            self.o_proj = _attn_proj(HIDDEN, N_HEADS * HEAD_DIM, "o_proj")
            self.q_norm = Qwen3RMSNorm(HEAD_DIM, eps=1e-6)
            self.k_norm = Qwen3RMSNorm(HEAD_DIM, eps=1e-6)

        def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            b, s, _ = x.shape
            q = self.q_norm(self.q_proj(x).view(b, s, N_HEADS, HEAD_DIM)).transpose(1, 2)
            k = self.k_norm(self.k_proj(x).view(b, s, N_KV, HEAD_DIM)).transpose(1, 2)
            v = self.v_proj(x).view(b, s, N_KV, HEAD_DIM).transpose(1, 2)
            q, k = _hf_qwen3.apply_rotary_pos_emb(q, k, cos, sin)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            out = out.transpose(1, 2).contiguous().view(b, s, N_HEADS * HEAD_DIM)
            return self.o_proj(out)

    attn = BenchAttention().to(dev)
    with torch.no_grad():
        for m in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
            m.lora_a.normal_(0.0, 0.02)
            m.lora_b.normal_(0.0, 0.02)
        attn.q_norm.weight.data = attn.q_norm.weight.data.to(dev, torch.bfloat16)
        attn.k_norm.weight.data = attn.k_norm.weight.data.to(dev, torch.bfloat16)
    attn.q_norm.weight.requires_grad_(False)
    attn.k_norm.weight.requires_grad_(False)
    attn.train()

    # shipped wiring (integrations/lf.py _wrap_attention_saved_tensor_offload_modules):
    # saved-tensor wrapper on the attention parent + qknorm wrappers + rope patch.
    install_qknorm_recompute(attn)  # pure passthrough unless armed (env/policy)
    install_rope_recompute()  # self-gated per call on rope_enabled()
    if INSTALL_WRAPPER[policy]:
        wrappers.append(install_attention_saved_tensor_offload(attn))

    class _MLPSource(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(HIDDEN, INTER, bias=False, dtype=torch.bfloat16)
            self.up_proj = nn.Linear(HIDDEN, INTER, bias=False, dtype=torch.bfloat16)
            self.down_proj = nn.Linear(INTER, HIDDEN, bias=False, dtype=torch.bfloat16)
            self.act_fn = F.silu

    mlp_src = _MLPSource()
    with torch.no_grad():
        for lin in (mlp_src.gate_proj, mlp_src.up_proj, mlp_src.down_proj):
            lin.weight.normal_(0.0, 0.02)
    mlp_stats = AsymExecutionStats()
    mlp = build_finegrained_dense_mlp(
        mlp_src,
        backend="asym",
        precision="bf16",
        lora_rank=RANK,
        lora_alpha=ALPHA,
        lora_dropout=0.0,
        stats=mlp_stats,
        strict=True,
        profile_prefix="bench.layers.0.mlp",
    ).cuda()
    with torch.no_grad():
        for m in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            m.lora_a.normal_(0.0, 0.02)
            m.lora_b.normal_(0.0, 0.02)
    mlp.train()

    # rope tables (frozen, bf16, [1, S, HEAD_DIM] — the modeling layout)
    theta = 1e6
    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=dev) / HEAD_DIM))
    t = torch.arange(tokens, dtype=torch.float32, device=dev)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    rope_cos = emb.cos()[None].to(torch.bfloat16)
    rope_sin = emb.sin()[None].to(torch.bfloat16)

    torch.manual_seed(7)
    x_attn = torch.randn(1, tokens, HIDDEN, device=dev, dtype=torch.bfloat16).requires_grad_(True)
    x_mlp = torch.randn(1, tokens, HIDDEN, device=dev, dtype=torch.bfloat16).requires_grad_(True)

    def attn_fwd(x):
        return checkpoint(lambda t_: attn(t_, rope_cos, rope_sin), x, use_reentrant=True)

    def mlp_fwd(x):
        return checkpoint(lambda t_: mlp(t_), x, use_reentrant=True)

    # ---------------- region runner ------------------------------------------
    def _lora_params(module):
        return [p for n, p in module.named_parameters() if "lora_" in n]

    def run_region(mod_name: str, fwd_fn, x: torch.Tensor, module) -> dict:
        reps = []
        for rep in range(WARMUP + REPS):
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            base_fwd = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            snap0 = ledger_snap()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            e0.record()
            out = fwd_fn(x)
            loss = out.sum()
            e1.record()
            torch.cuda.synchronize()
            fwd_wall = (time.perf_counter() - t0) * 1e3
            fwd_ev = e0.elapsed_time(e1)
            fwd_peak = torch.cuda.max_memory_allocated()
            snap1 = ledger_snap()

            base_bwd = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            e2 = torch.cuda.Event(enable_timing=True)
            e3 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            e2.record()
            loss.backward()
            e3.record()
            torch.cuda.synchronize()
            bwd_wall = (time.perf_counter() - t0) * 1e3
            bwd_ev = e2.elapsed_time(e3)
            bwd_peak = torch.cuda.max_memory_allocated()
            snap2 = ledger_snap()

            if rep == 0:
                assert x.grad is not None and torch.isfinite(x.grad.float()).all(), (
                    f"{mod_name}/{policy}: non-finite input grads"
                )
            x.grad = None
            for p in _lora_params(module):
                p.grad = None
            del out, loss

            reps.append(
                {
                    "fwd": {
                        "ev_ms": fwd_ev,
                        "wall_ms": fwd_wall,
                        "peak_bytes": int(fwd_peak),
                        "entry_allocated_bytes": int(base_fwd),
                        "link": ledger_delta(snap0, snap1),
                    },
                    "bwd": {
                        "ev_ms": bwd_ev,
                        "wall_ms": bwd_wall,
                        "peak_bytes": int(bwd_peak),
                        "entry_allocated_bytes": int(base_bwd),
                        "link": ledger_delta(snap1, snap2),
                    },
                }
            )
            print(
                f"[{policy}/{mod_name}] rep{rep} fwd {fwd_ev:8.1f} ms  bwd {bwd_ev:8.1f} ms  "
                f"fwd_peak {fwd_peak / 2**30:6.2f} GiB  bwd_peak {bwd_peak / 2**30:6.2f} GiB  "
                f"link d2h {(reps[-1]['fwd']['link']['d2h_bytes'] + reps[-1]['bwd']['link']['d2h_bytes']) / 2**30:6.2f} "
                f"h2d {(reps[-1]['fwd']['link']['h2d_bytes'] + reps[-1]['bwd']['link']['h2d_bytes']) / 2**30:6.2f} GiB",
                flush=True,
            )
        measured = reps[WARMUP:]

        def _mean(path):
            vals = [r[path[0]][path[1]] for r in measured]
            return sum(vals) / len(vals)

        agg = {
            "fwd_ms_mean": _mean(("fwd", "ev_ms")),
            "bwd_ms_mean": _mean(("bwd", "ev_ms")),
            "fwd_wall_ms_mean": _mean(("fwd", "wall_ms")),
            "bwd_wall_ms_mean": _mean(("bwd", "wall_ms")),
            "fwd_peak_bytes_mean": _mean(("fwd", "peak_bytes")),
            "bwd_peak_bytes_mean": _mean(("bwd", "peak_bytes")),
            "peak_bytes_mean": max(_mean(("fwd", "peak_bytes")), _mean(("bwd", "peak_bytes"))),
            "link_d2h_bytes_mean": sum(
                r["fwd"]["link"]["d2h_bytes"] + r["bwd"]["link"]["d2h_bytes"] for r in measured
            )
            / len(measured),
            "link_h2d_bytes_mean": sum(
                r["fwd"]["link"]["h2d_bytes"] + r["bwd"]["link"]["h2d_bytes"] for r in measured
            )
            / len(measured),
        }
        agg["link_total_bytes_mean"] = agg["link_d2h_bytes_mean"] + agg["link_h2d_bytes_mean"]
        return {"reps": reps, "agg": agg}

    # ---------------- engagement / policy-fired verification ------------------
    def _engagement(mod_name: str) -> dict:
        eng: dict = {
            "placement_policy_enabled": placement_policy.enabled(),
            "model_class": placement_policy.model_class(),
            "qknorm": qknorm_recompute_stats(),
            "restage_gap_total_ms": _ao.restage_gap_stats()["total_exposed_ms"],
            "cpu_pool": _ao.activation_offload_cpu_pool_stats(),
            "pinned_ledger": {
                k: pinned_ledger.stats()[k]
                for k in ("live_bytes", "high_water_bytes", "total_live_bytes", "total_high_water_bytes", "denials")
            },
        }
        if mod_name == "attention":
            eng["attn_context"] = attn_ctx.snapshot()
            if wrappers:
                eng["saved_wrapper"] = wrappers[0].snapshot()
            eng["attn_exec_stats"] = {
                "asym_forward_calls": attn_stats.asym_forward_calls,
                "attn_act_lora_a_forward_calls": attn_stats.attn_act_lora_a_forward_calls,
                "attn_act_hbm_gemm_calls_by_tag": dict(attn_stats.attn_act_hbm_gemm_calls_by_tag),
            }
        else:
            eng["mlp_last_offload_stats"] = dict(mlp._last_activation_offload_stats)
            eng["mlp_exec_stats"] = {
                "forward_calls": mlp_stats.dense_mlp_finegrained_forward_calls,
                "backward_calls": mlp_stats.dense_mlp_finegrained_backward_calls,
                "gpu_silu_bwd_calls": mlp_stats.dense_mlp_finegrained_gpu_silu_bwd_calls,
                "cpu_silu_bwd_calls": mlp_stats.dense_mlp_finegrained_cpu_silu_bwd_calls,
            }
        return eng

    # analytic notes: link traffic that is NOT an offload/restage copy — the
    # streamed in-place C2C reads (weights identical across policies; CPU-left
    # U/act reads only where those operands are CPU-homed, i.e. not under
    # keep-acts). Bytes per fwd+bwd pass at `tokens` rows.
    row_b = 2  # bf16
    # 3 full weight reads per step: outer no-grad forward, recompute forward,
    # backward dx (identical in every policy — CPU-homed base weights).
    weight_reads = {
        "attention_weight_stream_bytes_per_step": 3
        * (N_HEADS * HEAD_DIM * HIDDEN + 2 * N_KV * HEAD_DIM * HIDDEN + HIDDEN * N_HEADS * HEAD_DIM)
        * row_b,
        "mlp_weight_stream_bytes_per_step": 3 * (3 * HIDDEN * INTER) * row_b,
    }
    inplace_reads = {
        "attention_cpu_left_read_bytes": (
            0
            if policy == "recompute_all"
            else (3 * tokens * HIDDEN + tokens * N_HEADS * HEAD_DIM) * row_b * 2
        ),  # q/k/v share one U handle but each LoRA-A fwd + dA reads it; o reads its own U
        "mlp_cpu_left_read_bytes": (
            0 if policy == "recompute_all" else (4 * tokens * HIDDEN + 2 * tokens * INTER) * row_b
        ),  # gate/up LoRA-A fwd + dA read X; down LoRA-A fwd + dA read act
    }

    if memdiag:
        # harness diagnosis mode: 1 warmup rep, then 1 rep under CUDA memory
        # history; prints the live-set attribution at the peak instant. No json.
        fn, x, module = (
            (attn_fwd, x_attn, attn) if memdiag == "attention" else (mlp_fwd, x_mlp, mlp)
        )
        out_t = fn(x)
        loss = out_t.sum()
        loss.backward()
        x.grad = None
        for p in _lora_params(module):
            p.grad = None
        del out_t, loss
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.memory._record_memory_history(max_entries=400000)
        out_t = fn(x)
        loss = out_t.sum()
        loss.backward()
        torch.cuda.synchronize()
        snap = torch.cuda.memory._snapshot()
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"[memdiag {policy}/{memdiag}] (baseline x/weights/rope allocated before tracing)")
        for line in _memdiag_peak_report(snap):
            print("  " + line, flush=True)
        return

    results = {}
    for mod_name, fn, x, module in (
        ("attention", attn_fwd, x_attn, attn),
        ("mlp", mlp_fwd, x_mlp, mlp),
    ):
        print(f"=== {policy} / {mod_name} ===", flush=True)
        results[mod_name] = run_region(mod_name, fn, x, module)
        results[mod_name]["engagement"] = _engagement(mod_name)

    # hard harness gates (correctness of policy forcing, not of the result)
    tol = 64 * 2**20
    if policy == "recompute_all":
        for mod_name in results:
            a = results[mod_name]["agg"]
            assert a["link_total_bytes_mean"] < tol, (
                f"recompute_all/{mod_name}: link bytes {a['link_total_bytes_mean']} — a type offloaded"
            )
    if policy == "offload_all":
        assert wrappers[0].offload_calls > 0, "offload_all: saved-tensor wrapper never packed"
        assert wrappers[0].recipe_packs == 0, "offload_all: rope recipes engaged unexpectedly"
        q = results["attention"]["engagement"]["qknorm"]
        assert q["norm_offloads"] == 0, "offload_all: qknorm recompute engaged unexpectedly"
    if policy == "composed":
        q = results["attention"]["engagement"]["qknorm"]
        assert q["norm_offloads"] > 0, "composed: qknorm recompute did NOT engage"
        assert wrappers[0].recipe_packs > 0, "composed: rope recipe path did NOT engage"

    out = {
        "policy": policy,
        "env": {k: os.environ.get(k) for k in sorted({**COMMON_ENV, **POLICY_ENV[policy]})},
        "saved_tensor_wrapper_installed": INSTALL_WRAPPER[policy],
        "modules": results,
        "analytic_c2c_inplace_reads_note": {
            **weight_reads,
            **inplace_reads,
            "note": "streamed kernel reads over C2C, not offload/restage copies; weights are "
            "identical across policies, U/act CPU-left reads exist only where those operands "
            "are CPU-homed (offload_all, composed).",
        },
    }
    Path(out_path).write_text(json.dumps(out, indent=1))
    print(f"[child {policy}] wrote {out_path}", flush=True)


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------

def _gpu_idle_check() -> None:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=30,
        )
        used_mb, util = (int(v) for v in res.stdout.strip().split(",")[:2])
        print(f"[m3] GPU0 pre-run: {used_mb} MiB used, {util}% util")
        if used_mb > 4096:
            raise RuntimeError(f"GPU0 not idle ({used_mb} MiB in use) — refuse heavy run")
    except FileNotFoundError:
        print("[m3] nvidia-smi unavailable; skipping idle check")


def _orchestrate(tokens: int) -> None:
    _gpu_idle_check()
    CHILD_DIR.mkdir(parents=True, exist_ok=True)
    merged = {
        "bench": "m3_per_module_composition",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spec": "agent/impls/s04-p1-dgx-02-c06/motivation_v2_plots.md M3",
        "shapes": {
            "tokens": tokens, "batch": 1, "hidden": HIDDEN, "intermediate": INTER,
            "n_heads": N_HEADS, "n_kv_heads": N_KV, "head_dim": HEAD_DIM,
            "lora_rank": RANK, "lora_alpha": ALPHA, "dtype": "bf16",
        },
        "protocol": {
            "checkpointing": "torch reentrant checkpoint per module (shipped per-layer GC "
            "semantics: outer forward no-grad saves the module input; the backward window "
            "recomputes with grad enabled — where the shipped offload/recompute machinery "
            "runs — then differentiates)",
            "warmup_reps": WARMUP, "measured_reps": REPS,
            "timing": "CUDA events bracketing each segment + torch.cuda.synchronize; host wall recorded too",
            "peak": "torch.cuda.max_memory_allocated reset at each segment boundary (absolute)",
            "link": "runtime offload/restage copies counted at the runtime chokepoints "
            "(ActivationOffloadManager offload/stage*/record_cpu_ready, saved-tensor wrapper "
            "pack/unpack counters, qknorm _stage_fresh); in-place C2C kernel reads reported "
            "analytically, not in this metric",
            "process_isolation": "one subprocess per policy (env pins, memoized placement "
            "policy, pinned pools, CUDA context all isolated)",
        },
        "policies": {},
    }
    base_env = {k: v for k, v in os.environ.items() if not k.startswith(("ASYM", "UNSLOTH"))}
    for policy in POLICIES:
        child_out = CHILD_DIR / f"m3_child_{policy}.json"
        env = dict(base_env)
        env.update(COMMON_ENV)
        env.update(POLICY_ENV[policy])
        cmd = [sys.executable, str(Path(__file__).resolve()), "--policy", policy,
               "--tokens", str(tokens), "--child-out", str(child_out)]
        print(f"[m3] running policy={policy}: {' '.join(cmd)}", flush=True)
        t0 = time.perf_counter()
        res = subprocess.run(cmd, env=env)
        if res.returncode != 0:
            raise RuntimeError(f"policy {policy} child failed (rc={res.returncode})")
        print(f"[m3] policy={policy} done in {time.perf_counter() - t0:.0f}s", flush=True)
        merged["policies"][policy] = json.loads(child_out.read_text())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(merged, indent=1))
    print(f"[m3] wrote {OUT}")
    _print_table(merged)


def _print_table(merged: dict) -> None:
    gib = 2**30
    print("\n== M3 per-module composition (means over measured reps) ==")
    hdr = (f"{'module':10s} {'policy':14s} {'fwd_ms':>9s} {'bwd_ms':>9s} "
           f"{'fwdPk_GiB':>10s} {'bwdPk_GiB':>10s} {'d2h_GiB':>8s} {'h2d_GiB':>8s} {'link_GiB':>9s}")
    print(hdr)
    for mod in ("attention", "mlp"):
        for policy in POLICIES:
            a = merged["policies"][policy]["modules"][mod]["agg"]
            print(
                f"{mod:10s} {policy:14s} {a['fwd_ms_mean']:9.1f} {a['bwd_ms_mean']:9.1f} "
                f"{a['fwd_peak_bytes_mean'] / gib:10.2f} {a['bwd_peak_bytes_mean'] / gib:10.2f} "
                f"{a['link_d2h_bytes_mean'] / gib:8.2f} {a['link_h2d_bytes_mean'] / gib:8.2f} "
                f"{a['link_total_bytes_mean'] / gib:9.2f}"
            )
    print("\n== expectation checks (spec M3) ==")
    for mod in ("attention", "mlp"):
        g = {p: merged["policies"][p]["modules"][mod]["agg"] for p in POLICIES}
        rc, off, co = g["recompute_all"], g["offload_all"], g["composed"]
        peak_note = (
            f"bwd peak: recompute {rc['bwd_peak_bytes_mean'] / gib:.1f} / offload "
            f"{off['bwd_peak_bytes_mean'] / gib:.1f} / composed {co['bwd_peak_bytes_mean'] / gib:.1f} GiB"
        )
        link_note = (
            f"link: recompute {rc['link_total_bytes_mean'] / gib:.2f} / offload "
            f"{off['link_total_bytes_mean'] / gib:.2f} / composed {co['link_total_bytes_mean'] / gib:.2f} GiB "
            f"(composed/offload = {co['link_total_bytes_mean'] / max(1, off['link_total_bytes_mean']):.2f})"
        )
        tot = {p: g[p]["fwd_ms_mean"] + g[p]["bwd_ms_mean"] for p in POLICIES}
        best = min(tot.values())
        time_note = " ".join(f"{p}={tot[p]:.0f}ms({tot[p] / best - 1:+.1%})" for p in POLICIES)
        print(f"[{mod}] {peak_note}\n[{mod}] {link_note}\n[{mod}] time: {time_note}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=POLICIES)
    ap.add_argument("--tokens", type=int, default=TOKENS_DEFAULT)
    ap.add_argument("--child-out", default=None)
    ap.add_argument("--memdiag", choices=("attention", "mlp"), default=None,
                    help="with --policy: one rep under CUDA memory history, print peak attribution")
    args = ap.parse_args()
    if args.policy:
        _child(args.policy, args.tokens,
               args.child_out or str(CHILD_DIR / f"m3_child_{args.policy}.json"), args.memdiag)
    else:
        _orchestrate(args.tokens)


if __name__ == "__main__":
    main()
