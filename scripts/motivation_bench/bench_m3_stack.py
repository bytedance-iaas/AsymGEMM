#!/usr/bin/env python3
"""M3-STACK — N-layer composition bench (fig 4 redesign, 2026-07-27).

Rhetoric this bench serves (motivation_v2.md §2.4.2, corrected 2026-07-27):
composed = recompute AND offload on the SAME activation tensor — each layer is
recomputed in its backward window, every wide save is spilled to pinned host
right after regeneration (saved-tensor pack), and restaged just-in-time in the
backward's reverse consumption order (R5 prefetch / lazy stage). The wins only
become visible across MULTIPLE layers:
  * vs offload_all (the traditional shape: save+offload the true forward's
    activations, park them until that layer's backward): composed's host
    residency is per-LAYER-transient, not per-MODEL — offload_all's host
    high-water climbs ~N x C while composed stays near-flat;
  * vs recompute_all (pure checkpointing): the regenerated working set never
    piles up in HBM — recompute_all pays the full one-layer rematerialized
    transient regardless of N;
  * time: with real neighbor layers the spills/restages overlap adjacent
    compute (the isolated single-layer M3 run denied composed its overlap
    context, which is why recompute_all looked fastest there).

Three policies (fresh subprocess per (policy, N) so env pins, the memoized
placement policy, pinned pools and the CUDA context never leak):
  (a) recompute_all — per-layer reentrant checkpoint, KEEP_ACTS pins, no
      saved-tensor wrapper: nothing leaves the GPU.
  (b) offload_all — NO checkpoint: the saved-tensor wrapper packs the TRUE
      forward's saves (D2H at forward time) and the fg-MLP offloads its
      forward activations the same way; every pack parks on pinned host until
      that layer's backward unpacks it (restage prefetch default-off -> lazy
      just-at-consumer stage). This is the across-all-layers baseline of the
      rhetoric ("traditional offload saves across all the layers").
  (c) composed — per-layer checkpoint + ASYM_PLACEMENT_POLICY=1 + wrapper:
      the shipped mix (P12 qknorm/rope recipes rebuilt from the offloaded
      norm parent, P13 guarded reverse-order prefetch, policy keeps); ALL
      spill happens inside each layer's backward window.

Choreography gates (hard asserts — they prove the mechanism, not the result):
  offload_all:   >80% of D2H bytes land in the FWD segment (parks at forward);
                 recipe/qknorm paths NOT engaged.
  composed:      <5% of D2H bytes land in the FWD segment (spills in-window);
                 qknorm + rope-recipe paths engaged.
  recompute_all: link ~ 0.

Metrics per (policy, N):
  * time: CUDA events bracketing the whole-stack fwd and bwd segments.
  * peak GPU memory: torch.cuda.max_memory_allocated per segment (absolute).
  * HOST: pinned_ledger.stats() verbatim at build-end / after every rep /
    run-end (the wrapper's per-save buffers are booked under family "saved",
    attention_activation_offload._empty_strided_cpu_like), plus VmRSS/VmHWM
    from /proc/self/status. Headline host metric = run-end
    total_high_water_bytes minus build-end total_live_bytes (weights pinning
    excluded).
  * link bytes: the runtime's own offload/restage copies at the runtime
    chokepoints (same hooks as bench_m3.py), split by fwd/bwd segment.

Run (in-container, GPU pair rule: 0/1 = NUMA node 0):
  CUDA_VISIBLE_DEVICES=1 numactl --membind=0 .venv/bin/python \
      scripts/motivation_bench/bench_m3_stack.py
Output: profiling_results/motivation/m3_stack.json + summary table.
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
OUT = REPO / "profiling_results/motivation/m3_stack.json"
# durable + host-visible (container /tmp is a private tmpfs: child jsons there
# die with the instance, and a killed cell loses every completed cell's host
# metrics). Existing child jsons are reused unless M3S_RESUME=0.
CHILD_DIR = Path(os.environ.get(
    "ASYM_M3S_CHILD_DIR", str(REPO / "profiling_results/motivation/m3s_children")))

TOKENS_DEFAULT = int(os.environ.get("M3S_TOKENS", "65536"))  # 64K rows, batch 1
LAYERS_DEFAULT = os.environ.get("M3S_LAYERS", "1,2,4,8")
HIDDEN = 5120
INTER = 25600
N_HEADS = 64
N_KV = 8
HEAD_DIM = 128
RANK = 64
ALPHA = 128.0
WARMUP = 1
REPS = 2  # measured reps (means over these)

POLICIES = ("recompute_all", "offload_all", "composed")

COMMON_ENV = {
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
CHECKPOINTED = {"recompute_all": True, "offload_all": False, "composed": True}


def _proc_status() -> dict:
    out = {}
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(("VmRSS", "VmHWM")):
                key, val = line.split(":", 1)
                out[key] = int(val.strip().split()[0]) * 1024
    return out


# --------------------------------------------------------------------------
# child: one (policy, n_layers)
# --------------------------------------------------------------------------

def _node0_free_bytes() -> int | None:
    try:
        with open("/sys/devices/system/node/node0/meminfo") as fh:
            for line in fh:
                if "MemFree" in line:
                    return int(line.split()[-2]) * 1024
    except OSError:
        return None
    return None


def _child(policy: str, tokens: int, n_layers: int, out_path: str) -> None:
    import gc

    # capacity guard (offload_all only): the uncheckpointed baseline parks
    # ~47 GiB of pinned saves per layer at 128K on the membind node. The first
    # N=8x128K attempt was SIGKILLed by the host OOM reaper (rc=-9, archived in
    # m3_stack_128k.log) — that kill IS the measured infeasibility; refuse the
    # re-attempt cleanly instead of re-rolling the OOM killer on a shared box.
    if policy == "offload_all" and os.environ.get("M3S_CAPACITY_GUARD", "1") == "1":
        per_layer = int(47 * 2**30 * (tokens / 131072))
        demand = n_layers * per_layer + 60 * 2**30
        free = _node0_free_bytes()
        # MemFree alone over-promises for page-locking (measured: N=8x128K was
        # SIGKILLed twice with MemFree > demand) — also enforce the measured
        # feasibility ceiling directly.
        max_layers = int(os.environ.get("M3S_OFFLOAD_MAX_LAYERS", "4")) * (131072 // tokens or 1)
        if n_layers > max_layers or (free is not None and demand > free):
            Path(out_path).write_text(json.dumps({
                "policy": policy, "n_layers": n_layers,
                "failed_capacity": {
                    "estimated_pinned_demand_bytes": demand,
                    "node0_free_bytes": free,
                    "note": "uncheckpointed offload parks ~47 GiB/layer of pinned saves "
                    "at 128K; demand exceeds the membind node's free memory. First "
                    "attempt was OOM-killed by the host (rc=-9).",
                },
            }, indent=1))
            print(f"[child {policy}/N={n_layers}] CAPACITY REFUSAL: "
                  f"~{demand / 2**30:.0f} GiB pinned demand vs {free / 2**30:.0f} GiB free on node 0",
                  flush=True)
            return

    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.checkpoint import checkpoint

    import asym_gemm  # noqa: F401
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
    placement_policy.register_model_class("dense")

    # ------- link-byte ledger at the runtime chokepoints (same as bench_m3) --
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
            _led("d2h", handle.tag, handle.nbytes)

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

    wrappers: list = []

    def ledger_snap() -> dict:
        w_d2h = sum(w.offloaded_bytes for w in wrappers)
        w_h2d = sum(sum(w.stage_bytes_by_tag.values()) for w in wrappers)
        # backward-born gradient spills (mlp.dact/dup/dgate roundtrips) can never
        # sit in the forward segment — split them out so the choreography gates
        # judge only the SAVE class.
        grad_d2h = sum(v[0] for t, v in BY_TAG.items() if t.startswith("mlp.d"))
        return {"d2h": LED["d2h"] + w_d2h, "h2d": LED["h2d"] + w_h2d, "grad_d2h": grad_d2h}

    # ------- model stack -----------------------------------------------------
    torch.manual_seed(20260727)

    def _host_weight(out_f: int, in_f: int) -> HostWeight:
        w = torch.empty(out_f, in_f, dtype=torch.bfloat16)
        w.normal_(0.0, 0.02)
        return HostWeight(w, pin_memory=True, clone=False)

    attn_stats = AsymExecutionStats()
    mlp_stats = AsymExecutionStats()

    class BenchAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            ctx = AttentionActivationOffloadContext()
            self._ctx = ctx

            def proj(out_f: int, in_f: int, role: str) -> AsymActivationOffloadLoRALinear:
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
                    attention_context=ctx,
                )

            self.q_proj = proj(N_HEADS * HEAD_DIM, HIDDEN, "q_proj")
            self.k_proj = proj(N_KV * HEAD_DIM, HIDDEN, "k_proj")
            self.v_proj = proj(N_KV * HEAD_DIM, HIDDEN, "v_proj")
            self.o_proj = proj(HIDDEN, N_HEADS * HEAD_DIM, "o_proj")
            self.q_norm = Qwen3RMSNorm(HEAD_DIM, eps=1e-6)
            self.k_norm = Qwen3RMSNorm(HEAD_DIM, eps=1e-6)

        def forward(self, x, cos, sin):
            b, s, _ = x.shape
            q = self.q_norm(self.q_proj(x).view(b, s, N_HEADS, HEAD_DIM)).transpose(1, 2)
            k = self.k_norm(self.k_proj(x).view(b, s, N_KV, HEAD_DIM)).transpose(1, 2)
            v = self.v_proj(x).view(b, s, N_KV, HEAD_DIM).transpose(1, 2)
            q, k = _hf_qwen3.apply_rotary_pos_emb(q, k, cos, sin)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            out = out.transpose(1, 2).contiguous().view(b, s, N_HEADS * HEAD_DIM)
            return self.o_proj(out)

    class _MLPSource(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(HIDDEN, INTER, bias=False, dtype=torch.bfloat16)
            self.up_proj = nn.Linear(HIDDEN, INTER, bias=False, dtype=torch.bfloat16)
            self.down_proj = nn.Linear(INTER, HIDDEN, bias=False, dtype=torch.bfloat16)
            self.act_fn = F.silu

    class BenchLayer(nn.Module):
        """Pre-norm decoder layer: norm -> attn -> +res -> norm -> mlp -> +res."""

        def __init__(self, idx: int) -> None:
            super().__init__()
            self.input_norm = Qwen3RMSNorm(HIDDEN, eps=1e-6)
            self.post_norm = Qwen3RMSNorm(HIDDEN, eps=1e-6)
            self.attn = BenchAttention()
            src = _MLPSource()
            with torch.no_grad():
                for lin in (src.gate_proj, src.up_proj, src.down_proj):
                    lin.weight.normal_(0.0, 0.02)
            self.mlp = build_finegrained_dense_mlp(
                src,
                backend="asym",
                precision="bf16",
                lora_rank=RANK,
                lora_alpha=ALPHA,
                lora_dropout=0.0,
                stats=mlp_stats,
                strict=True,
                profile_prefix=f"bench.layers.{idx}.mlp",
            )

        def forward(self, x, cos, sin):
            x = x + self.attn(self.input_norm(x), cos, sin)
            x = x + self.mlp(self.post_norm(x))
            return x

    layers: list[nn.Module] = []
    for i in range(n_layers):
        layer = BenchLayer(i).to(dev)
        with torch.no_grad():
            for m in (layer.attn.q_proj, layer.attn.k_proj, layer.attn.v_proj,
                      layer.attn.o_proj, layer.mlp.gate_proj, layer.mlp.up_proj,
                      layer.mlp.down_proj):
                m.lora_a.normal_(0.0, 0.02)
                m.lora_b.normal_(0.0, 0.02)
            for norm in (layer.input_norm, layer.post_norm, layer.attn.q_norm, layer.attn.k_norm):
                norm.weight.data = norm.weight.data.to(dev, torch.bfloat16)
                norm.weight.requires_grad_(False)
        layer.train()
        install_qknorm_recompute(layer.attn)
        if INSTALL_WRAPPER[policy]:
            # composed mirrors the shipped wiring (wrapper on the attention
            # module; the checkpoint boundary owns the layer-level saves).
            # offload_all is UNcheckpointed, so the wrapper must cover the WHOLE
            # layer — otherwise the norm/residual saves silently accumulate in
            # HBM and the baseline is denied its best shape.
            target = layer if policy == "offload_all" else layer.attn
            wrappers.append(install_attention_saved_tensor_offload(target))
        layers.append(layer)
        print(f"[build] layer {i + 1}/{n_layers} ready", flush=True)
    install_rope_recompute()  # global monkeypatch, self-gated per call

    theta = 1e6
    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=dev) / HEAD_DIM))
    t = torch.arange(tokens, dtype=torch.float32, device=dev)
    emb = torch.cat([torch.outer(t, inv_freq)] * 2, dim=-1)
    rope_cos = emb.cos()[None].to(torch.bfloat16)
    rope_sin = emb.sin()[None].to(torch.bfloat16)

    torch.manual_seed(7)
    x0 = torch.randn(1, tokens, HIDDEN, device=dev, dtype=torch.bfloat16).requires_grad_(True)

    def stack_forward(x):
        for layer in layers:
            if CHECKPOINTED[policy]:
                x = checkpoint(lambda t_, l=layer: l(t_, rope_cos, rope_sin), x, use_reentrant=True)
            else:
                x = layer(x, rope_cos, rope_sin)
        return x

    def _lora_params():
        for layer in layers:
            for name, p in layer.named_parameters():
                if "lora_" in name:
                    yield p

    # ------- measured region --------------------------------------------------
    build_ledger = pinned_ledger.stats()
    build_status = _proc_status()
    reps: list[dict] = []
    for rep in range(WARMUP + REPS):
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()
        snap0 = ledger_snap()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        e0.record()
        out = stack_forward(x0)
        loss = out.sum()
        e1.record()
        torch.cuda.synchronize()
        fwd_wall = (time.perf_counter() - t0) * 1e3
        fwd_ev = e0.elapsed_time(e1)
        fwd_peak = torch.cuda.max_memory_allocated()
        snap1 = ledger_snap()

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
            assert x0.grad is not None and torch.isfinite(x0.grad.float()).all(), (
                f"{policy}/N={n_layers}: non-finite input grads"
            )
        x0.grad = None
        for p in _lora_params():
            p.grad = None
        del out, loss

        reps.append(
            {
                "fwd": {"ev_ms": fwd_ev, "wall_ms": fwd_wall, "peak_bytes": int(fwd_peak),
                        "d2h_bytes": snap1["d2h"] - snap0["d2h"], "h2d_bytes": snap1["h2d"] - snap0["h2d"],
                        "grad_d2h_bytes": snap1["grad_d2h"] - snap0["grad_d2h"]},
                "bwd": {"ev_ms": bwd_ev, "wall_ms": bwd_wall, "peak_bytes": int(bwd_peak),
                        "d2h_bytes": snap2["d2h"] - snap1["d2h"], "h2d_bytes": snap2["h2d"] - snap1["h2d"],
                        "grad_d2h_bytes": snap2["grad_d2h"] - snap1["grad_d2h"]},
                "pinned_ledger": pinned_ledger.stats(),
                "proc_status": _proc_status(),
            }
        )
        r = reps[-1]
        print(
            f"[{policy}/N={n_layers}] rep{rep} fwd {fwd_ev:8.1f} ms  bwd {bwd_ev:8.1f} ms  "
            f"fwd_peak {fwd_peak / 2**30:6.2f} GiB  bwd_peak {bwd_peak / 2**30:6.2f} GiB  "
            f"d2h {(r['fwd']['d2h_bytes'] + r['bwd']['d2h_bytes']) / 2**30:6.2f} GiB "
            f"(fwd {r['fwd']['d2h_bytes'] / 2**30:5.2f})  "
            f"h2d {(r['fwd']['h2d_bytes'] + r['bwd']['h2d_bytes']) / 2**30:6.2f} GiB",
            flush=True,
        )

    measured = reps[WARMUP:]

    def _mean(seg: str, key: str) -> float:
        return sum(r[seg][key] for r in measured) / len(measured)

    end_ledger = pinned_ledger.stats()
    end_status = _proc_status()
    host_act_hw = int(end_ledger.get("total_high_water_bytes", 0)) - int(build_ledger.get("total_live_bytes", 0))

    agg = {
        "fwd_ms_mean": _mean("fwd", "ev_ms"),
        "bwd_ms_mean": _mean("bwd", "ev_ms"),
        "step_ms_mean": _mean("fwd", "ev_ms") + _mean("bwd", "ev_ms"),
        "fwd_peak_bytes_mean": _mean("fwd", "peak_bytes"),
        "bwd_peak_bytes_mean": _mean("bwd", "peak_bytes"),
        "step_peak_bytes_mean": max(_mean("fwd", "peak_bytes"), _mean("bwd", "peak_bytes")),
        "d2h_bytes_mean": _mean("fwd", "d2h_bytes") + _mean("bwd", "d2h_bytes"),
        "h2d_bytes_mean": _mean("fwd", "h2d_bytes") + _mean("bwd", "h2d_bytes"),
        "fwd_d2h_bytes_mean": _mean("fwd", "d2h_bytes"),
        "grad_d2h_bytes_mean": _mean("fwd", "grad_d2h_bytes") + _mean("bwd", "grad_d2h_bytes"),
        "host_act_high_water_bytes": host_act_hw,
        "vm_hwm_minus_build_rss_bytes": end_status["VmHWM"] - build_status["VmRSS"],
    }

    # ------- engagement + choreography gates ---------------------------------
    qk = qknorm_recompute_stats()
    eng = {
        "placement_policy_enabled": placement_policy.enabled(),
        "qknorm": qk,
        "wrapper": {
            "offload_calls": sum(w.offload_calls for w in wrappers),
            "offloaded_bytes": sum(w.offloaded_bytes for w in wrappers),
            "cpu_peak_bytes_live": sum(w.cpu_peak_bytes_live for w in wrappers),
            "recipe_packs": sum(w.recipe_packs for w in wrappers),
            "recipe_bytes_avoided": sum(w.recipe_bytes_avoided for w in wrappers),
            "unpack_calls": sum(w.unpack_calls for w in wrappers),
        },
        "restage_gap_total_ms": _ao.restage_gap_stats()["total_exposed_ms"],
        "cpu_pool": _ao.activation_offload_cpu_pool_stats(),
        "pinned_ledger_build": build_ledger,
        "pinned_ledger_end": end_ledger,
        "proc_status_build": build_status,
        "proc_status_end": end_status,
    }

    tol = 64 * 2**20 * n_layers
    total_d2h = agg["d2h_bytes_mean"]
    if policy == "recompute_all":
        assert total_d2h + agg["h2d_bytes_mean"] < tol, (
            f"recompute_all/N={n_layers}: link bytes {total_d2h + agg['h2d_bytes_mean']:.0f} — a type offloaded"
        )
    if policy == "offload_all":
        assert eng["wrapper"]["offload_calls"] > 0, "offload_all: wrapper never packed"
        assert eng["wrapper"]["recipe_packs"] == 0, "offload_all: rope recipes engaged unexpectedly"
        assert qk["norm_offloads"] == 0, "offload_all: qknorm recompute engaged unexpectedly"
        save_d2h = total_d2h - agg["grad_d2h_bytes_mean"]
        assert save_d2h > 0 and agg["fwd_d2h_bytes_mean"] > 0.95 * save_d2h, (
            f"offload_all/N={n_layers}: only {agg['fwd_d2h_bytes_mean'] / max(1, save_d2h):.0%} of "
            "save-class D2H in the fwd segment — baseline is not parking at forward time"
        )
    if policy == "composed":
        assert qk["norm_offloads"] > 0, "composed: qknorm recompute did NOT engage"
        assert eng["wrapper"]["recipe_packs"] > 0, "composed: rope recipe path did NOT engage"
        assert agg["fwd_d2h_bytes_mean"] < 0.05 * max(1, total_d2h), (
            f"composed/N={n_layers}: {agg['fwd_d2h_bytes_mean'] / max(1, total_d2h):.0%} of D2H in the "
            "fwd segment — spill is not confined to the backward window"
        )

    out_doc = {
        "policy": policy,
        "n_layers": n_layers,
        "checkpointed": CHECKPOINTED[policy],
        "env": {k: os.environ.get(k) for k in sorted({**COMMON_ENV, **POLICY_ENV[policy]})},
        "saved_tensor_wrapper_installed": INSTALL_WRAPPER[policy],
        "agg": agg,
        "reps": reps,
        "engagement": eng,
    }
    Path(out_path).write_text(json.dumps(out_doc, indent=1))
    print(f"[child {policy}/N={n_layers}] wrote {out_path}", flush=True)


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------

def _gpu_idle_check() -> None:
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits", "-i", dev],
            capture_output=True, text=True, timeout=30,
        )
        used_mb, util = (int(v) for v in res.stdout.strip().split(",")[:2])
        print(f"[m3s] GPU{dev} pre-run: {used_mb} MiB used, {util}% util")
        if used_mb > 4096:
            raise RuntimeError(f"GPU{dev} not idle ({used_mb} MiB in use) — refuse heavy run")
    except FileNotFoundError:
        print("[m3s] nvidia-smi unavailable; skipping idle check")


def _orchestrate(tokens: int, layer_counts: list[int]) -> None:
    _gpu_idle_check()
    CHILD_DIR.mkdir(parents=True, exist_ok=True)
    merged = {
        "bench": "m3_stack_composition",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spec": "fig-4 redesign 2026-07-27 (recompute+offload composed on the same tensor; "
        "N-layer host/HBM/time sweep)",
        "shapes": {
            "tokens": tokens, "batch": 1, "hidden": HIDDEN, "intermediate": INTER,
            "n_heads": N_HEADS, "n_kv_heads": N_KV, "head_dim": HEAD_DIM,
            "lora_rank": RANK, "lora_alpha": ALPHA, "dtype": "bf16",
        },
        "layer_counts": layer_counts,
        "protocol": {
            "checkpointing": "per-layer reentrant checkpoint for recompute_all/composed; "
            "offload_all runs UNcheckpointed (true-forward saves park on host until that "
            "layer's backward — the traditional across-all-layers offload shape)",
            "warmup_reps": WARMUP, "measured_reps": REPS,
            "timing": "CUDA events bracketing the whole-stack fwd and bwd segments",
            "peak": "torch.cuda.max_memory_allocated reset per segment (absolute)",
            "host": "pinned_ledger.stats() at build-end/per-rep/run-end (wrapper saves booked "
            "under family 'saved'); headline = end total_high_water - build total_live; "
            "VmRSS/VmHWM recorded",
            "process_isolation": "one subprocess per (policy, n_layers)",
        },
        "runs": {},
    }
    base_env = {k: v for k, v in os.environ.items() if not k.startswith(("ASYM", "UNSLOTH"))}
    resume = os.environ.get("M3S_RESUME", "1") == "1"
    for policy in POLICIES:
        for n in layer_counts:
            child_out = CHILD_DIR / f"m3s_{policy}_N{n}.json"
            key = f"{policy}_N{n}"
            if resume and child_out.exists():
                doc = json.loads(child_out.read_text())
                if "agg" in doc or "failed_capacity" in doc:
                    print(f"[m3s] {key}: reusing existing child json", flush=True)
                    merged["runs"][key] = doc
                    continue
            env = dict(base_env)
            env.update(COMMON_ENV)
            env.update(POLICY_ENV[policy])
            cmd = [sys.executable, str(Path(__file__).resolve()), "--policy", policy,
                   "--tokens", str(tokens), "--layers", str(n), "--child-out", str(child_out)]
            print(f"[m3s] running {policy} N={n}", flush=True)
            t0 = time.perf_counter()
            res = subprocess.run(cmd, env=env)
            if res.returncode != 0:
                print(f"[m3s] {key} FAILED rc={res.returncode} — recorded, continuing", flush=True)
                merged["runs"][key] = {"policy": policy, "n_layers": n,
                                       "failed": True, "rc": res.returncode}
                continue
            print(f"[m3s] {policy} N={n} done in {time.perf_counter() - t0:.0f}s", flush=True)
            merged["runs"][key] = json.loads(child_out.read_text())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(merged, indent=1))
    print(f"[m3s] wrote {OUT}")
    _print_table(merged, layer_counts)


def _print_table(merged: dict, layer_counts: list[int]) -> None:
    gib = 2**30
    print("\n== M3-STACK (means over measured reps) ==")
    print(f"{'policy':14s} {'N':>3s} {'fwd_ms':>9s} {'bwd_ms':>9s} {'ms/layer':>9s} "
          f"{'stepPk_GiB':>11s} {'host_GiB':>9s} {'d2h_GiB':>8s} {'fwd_d2h%':>9s} {'h2d_GiB':>8s}")
    for policy in POLICIES:
        for n in layer_counts:
            doc = merged["runs"][f"{policy}_N{n}"]
            if "agg" not in doc:
                why = ("host-capacity refusal" if "failed_capacity" in doc
                       else f"FAILED rc={doc.get('rc')}")
                print(f"{policy:14s} {n:3d} {'-- ' + why + ' --':>60s}")
                continue
            a = doc["agg"]
            fwd_pct = a["fwd_d2h_bytes_mean"] / max(1.0, a["d2h_bytes_mean"])
            print(
                f"{policy:14s} {n:3d} {a['fwd_ms_mean']:9.1f} {a['bwd_ms_mean']:9.1f} "
                f"{a['step_ms_mean'] / n:9.1f} {a['step_peak_bytes_mean'] / gib:11.2f} "
                f"{a['host_act_high_water_bytes'] / gib:9.2f} {a['d2h_bytes_mean'] / gib:8.2f} "
                f"{fwd_pct:9.0%} {a['h2d_bytes_mean'] / gib:8.2f}"
            )
    nmax = layer_counts[-1]
    print("\n== expectation checks ==")
    for metric, key, unit in (
        ("host high-water", "host_act_high_water_bytes", gib),
        ("step peak HBM", "step_peak_bytes_mean", gib),
        ("ms/layer", None, None),
    ):
        line = []
        for policy in POLICIES:
            done = [n for n in layer_counts if "agg" in merged["runs"].get(f"{policy}_N{n}", {})]
            if not done:
                line.append(f"{policy} (no data)")
                continue
            nlo, nhi = done[0], done[-1]
            a1 = merged["runs"][f"{policy}_N{nlo}"]["agg"]
            aN = merged["runs"][f"{policy}_N{nhi}"]["agg"]
            suffix = "" if nhi == nmax else f" (to N={nhi})"
            if key is None:
                v1, vN = a1["step_ms_mean"] / nlo, aN["step_ms_mean"] / nhi
                line.append(f"{policy} {v1:.0f}->{vN:.0f} ms/layer{suffix}")
            else:
                line.append(f"{policy} {a1[key] / unit:.1f}->{aN[key] / unit:.1f} GiB{suffix}")
        print(f"[{metric}] N={layer_counts[0]} -> N={nmax}: " + " | ".join(line))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=POLICIES)
    ap.add_argument("--tokens", type=int, default=TOKENS_DEFAULT)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--child-out", default=None)
    args = ap.parse_args()
    if args.policy:
        n = args.layers or 1
        _child(args.policy, args.tokens, n,
               args.child_out or str(CHILD_DIR / f"m3s_{args.policy}_N{n}.json"))
    else:
        counts = [int(v) for v in LAYERS_DEFAULT.split(",") if v.strip()]
        _orchestrate(args.tokens, sorted(counts))


if __name__ == "__main__":
    main()
