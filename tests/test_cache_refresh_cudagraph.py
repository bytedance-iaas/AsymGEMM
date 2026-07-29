"""CUDA-graph replay picks up Layer.update_gpu_cache mutations (Phase 1).

Decisive claim under test: capturable_decode_forward's hybrid CPU+cached-GPU
path (asym_gemm/unified_moe/capturable.py:171-230) reads `layer.cached_slot`
fresh every replay — not a capture-time constant — so mutating the VRAM
cache's contents in place between replays (never reassigning the tensor
objects) is picked up correctly by BOTH the CPU-bucket routing mask and the
GPU cached-bucket compute on the very next replay, with no recapture.

This is the first CUDA-graph capture/replay test in this repo:
capturable.py's own module docstring cites tests/phase0_cudagraph_decode.py
as the mechanism's validation harness, but that file does not exist in this
checkout (confirmed by search) — there is otherwise zero capture/replay
coverage here.

Designed to run two ways (AsymGEMM convention):
    python tests/test_cache_refresh_cudagraph.py
    pytest tests/test_cache_refresh_cudagraph.py -s

Internal skip when AMX or CUDA is not present, same idiom as
tests/test_unified_moe.py.
"""
from __future__ import annotations

import os
import sys

import torch

import asym_gemm


def _module_skip(msg: str) -> None:
    print(f"[SKIP] {msg}")
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg, allow_module_level=True)
    sys.exit(0)


if asym_gemm.unified_moe is None:
    _module_skip("asym_gemm.unified_moe sub-package not available "
                 "(CPU extension _cpu_C did not build).")

_caps = asym_gemm.unified_moe._C.caps()
if not _caps.get("has_amx_int8", False):
    _module_skip(f"AMX-INT8 not present on this host (caps={_caps}). "
                 "The unified_moe CPU bucket requires AMX.")

if not torch.cuda.is_available():
    _module_skip("No CUDA device — graph capture/replay tests cannot run.")

from asym_gemm.unified_moe import Layer
from asym_gemm.unified_moe.capturable import (
    capturable_decode_forward,
    capturable_decode_supported,
    init_capturable_decode,
)


def fp32_reference_expert(x_bf16_cpu, gate_bf16, up_bf16, down_bf16):
    x = x_bf16_cpu.float()
    gate = x @ gate_bf16.float().t()
    up   = x @ up_bf16.float().t()
    act  = torch.nn.functional.silu(gate) * up
    act_bf16 = act.to(torch.bfloat16).float()
    return act_bf16 @ down_bf16.float().t()


def scale_rel(actual: torch.Tensor, ref: torch.Tensor) -> float:
    a, r = actual.float(), ref.float()
    return float(((a - r).abs().max() / r.abs().max().clamp(min=1e-6)).item())


def test_cache_refresh_visible_across_graph_replay():
    """Capture once with expert 0 cached; evict it via update_gpu_cache
    without recapturing; replay again with identical inputs — output must
    still match the FP32 reference for expert 0, not silently drop to zero.
    """
    torch.manual_seed(20)
    G, H, I, top_k, T = 4, 256, 512, 1, 8
    gate = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.07
    up   = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.07
    down = torch.randn(G, H, I, dtype=torch.bfloat16) * 0.05

    prev = os.environ.get("ASYMGEMM_GPU_CACHED_EXPERTS")
    os.environ["ASYMGEMM_GPU_CACHED_EXPERTS"] = "2"  # cache_n=2 of 4
    try:
        layer = Layer.from_bf16(gate, up, down, top_k=top_k, cpu_threads=8)
    finally:
        if prev is None:
            os.environ.pop("ASYMGEMM_GPU_CACHED_EXPERTS", None)
        else:
            os.environ["ASYMGEMM_GPU_CACHED_EXPERTS"] = prev

    assert layer.cache_n == 2
    assert bool(layer._cached_mask_np[0]), "expert 0 must start cached (default 0..n-1)"

    x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    expert_ids = torch.zeros(T, top_k, dtype=torch.int64, device="cuda")  # all -> expert 0
    route_w    = torch.ones(T, top_k, dtype=torch.float32, device="cuda")

    y_ref = fp32_reference_expert(x.cpu(), gate[0], up[0], down[0]).to(torch.bfloat16)

    init_capturable_decode(layer, [T])
    assert capturable_decode_supported(layer, T)

    # Warm up the hybrid path eagerly (side-stream creation, host-fn launch,
    # grouped-GEMM JIT) — all lazy init must happen before capture.
    _ = capturable_decode_forward(layer, x, expert_ids, route_w)
    torch.cuda.synchronize()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        y_static = capturable_decode_forward(layer, x, expert_ids, route_w)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_static = capturable_decode_forward(layer, x, expert_ids, route_w)

    g.replay()
    torch.cuda.synchronize()
    y1 = y_static.clone().cpu()
    r1 = scale_rel(y1, y_ref)
    print(f"  [test_cache_refresh_visible_across_graph_replay] "
          f"replay #1 (expert 0 cached) scale_rel={r1:.3e}")
    assert r1 <= 4e-2, f"replay #1 scale_rel {r1:.3e} > 4e-2"

    torch.cuda.synchronize()
    changed = layer.update_gpu_cache([2, 3])  # evicts expert 0 (and 1)
    assert changed
    assert not bool(layer._cached_mask_np[0]), "expert 0 must be evicted"

    g.replay()
    torch.cuda.synchronize()
    y2 = y_static.clone().cpu()
    r2 = scale_rel(y2, y_ref)
    print(f"  [test_cache_refresh_visible_across_graph_replay] "
          f"replay #2 (expert 0 evicted, same graph) scale_rel={r2:.3e}")
    assert y2.float().abs().max().item() > 1e-3, (
        "output silently zeroed after eviction — the captured graph is not "
        "reading cached_slot's live contents on replay"
    )
    assert r2 <= 4e-2, (
        f"replay #2 scale_rel {r2:.3e} > 4e-2 — eviction did not correctly "
        "redirect expert 0's tokens to the CPU bucket"
    )
