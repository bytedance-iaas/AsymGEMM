"""asym_gemm.unified_moe — end-to-end performance benchmark.

Sweeps `(num_experts, hidden, inter, batch, m_cpu)` and reports forward
throughput as effective TFLOPS:

    FLOPs(forward)   = T · top_k · 6 · H · I        # gate + up + down
                       └─────────┘ └────────┘
                         tokens/   2·H·I per
                         expert    GEMM × 3 GEMMs
    TFLOPS(peak)     = FLOPs / min(wall_time) / 1e12

"Effective" because the wall-time covers everything the kernel actually
does: per-token activation quant, AMX/AMX-INT8 compute, SwiGLU,
H2D/D2H copies for the GPU bucket, weighted scatter-add. Per-call setup
that doesn't scale with (H, I) shows up as lower-than-peak numbers at
small T.

Run:
    PYTHONPATH=/workspace/AsymGEMM_Main/AsymGEMM \\
        python tests/bench_unified_moe.py
    PYTHONPATH=… python tests/bench_unified_moe.py --quick
    PYTHONPATH=… python tests/bench_unified_moe.py --cpu-threads 32 --iters 10

The `m_cpu` sweep is the routing knob: `m_cpu = 0` forces every expert to
the GPU bucket, `m_cpu = ∞` forces every expert to the CPU bucket, and
values in between let the dispatcher mix CPU + GPU per-expert.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

import asym_gemm

# -- Hardware/build gating ---------------------------------------------------

if asym_gemm.unified_moe is None:
    print("[SKIP] asym_gemm.unified_moe sub-package not available.")
    sys.exit(0)

_CAPS = asym_gemm.unified_moe._C.caps()
CPU_HAS_AMX = bool(_CAPS.get("has_amx_int8", False))
GPU_AVAILABLE = torch.cuda.is_available()

from asym_gemm.unified_moe import Layer  # noqa: E402

INF_M_CPU = 10**9


# -- Helpers -----------------------------------------------------------------

def unique_topk_routing(T: int, G: int, top_k: int, seed: int = 0) -> torch.Tensor:
    """[T, top_k] expert ids, no duplicates per row — real-router shape."""
    assert G >= top_k
    gen = torch.Generator().manual_seed(seed)
    out = torch.empty(T, top_k, dtype=torch.int64)
    for t in range(T):
        out[t] = torch.randperm(G, generator=gen)[:top_k]
    return out


@dataclass
class BenchResult:
    G: int
    H: int
    I: int
    T: int
    top_k: int
    m_cpu: int
    ms_mean: float
    ms_min: float
    tflops_peak: float          # using ms_min
    tflops_mean: float          # using ms_mean
    avg_m_e: float
    routing: str                # "all_cpu" / "all_gpu" / "mixed(c/g)"


def make_layer(G: int, H: int, I: int, top_k: int, cpu_threads: int) -> Layer:
    gate = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.07
    up   = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.07
    down = torch.randn(G, H, I, dtype=torch.bfloat16) * 0.05
    return Layer.from_bf16(gate, up, down, top_k=top_k, cpu_threads=cpu_threads)


def classify_routing(G: int, expert_ids: torch.Tensor, m_cpu: int,
                     T: int, top_k: int) -> tuple[str, float]:
    """Mirror Layer.forward's per-expert dispatch decision and report the
    resulting CPU/GPU bucket split + average per-expert token count."""
    counts = [0] * G
    ei = expert_ids.cpu().to(torch.int64).numpy()
    for t in range(T):
        for j in range(top_k):
            counts[int(ei[t, j])] += 1
    active = [c for c in counts if c > 0]
    avg_m_e = float(sum(active) / max(len(active), 1)) if active else 0.0
    cpu_n = sum(1 for c in active if c <= m_cpu)
    gpu_n = sum(1 for c in active if c >  m_cpu)
    if gpu_n == 0:
        return "all_cpu", avg_m_e
    if cpu_n == 0:
        return "all_gpu", avg_m_e
    return f"mixed({cpu_n}c/{gpu_n}g)", avg_m_e


def bench_one(G: int, H: int, I: int, T: int, top_k: int, m_cpu: int,
              cpu_threads: int, iters: int, warmup: int,
              seed: int = 0) -> BenchResult:
    layer = make_layer(G, H, I, top_k, cpu_threads)
    layer.set_m_cpu(m_cpu)

    torch.manual_seed(seed)
    expert_ids = unique_topk_routing(T, G, top_k, seed=seed)
    route_w    = torch.softmax(torch.randn(T, top_k), dim=-1).float()

    # Where the activations live drives the H2D path the GPU bucket sees.
    # We keep them on the GPU when CUDA is available — that's the realistic
    # production shape (router output already on device).
    if GPU_AVAILABLE:
        x_dev = torch.device("cuda")
        x = torch.randn(T, H, dtype=torch.bfloat16, device=x_dev)
        expert_ids = expert_ids.cuda()
        route_w    = route_w.cuda()
    else:
        x_dev = torch.device("cpu")
        x = torch.randn(T, H, dtype=torch.bfloat16)

    def one_call() -> None:
        layer.forward(x, expert_ids, route_w)

    # Warmup — first call triggers NVRTC JIT for the SM90 INT8 kernel
    # plus the AMX tile-config + per-thread pinning. Account generously.
    for _ in range(warmup):
        one_call()
    if GPU_AVAILABLE:
        torch.cuda.synchronize()

    runs: list[float] = []
    for _ in range(iters):
        if GPU_AVAILABLE:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_call()
        if GPU_AVAILABLE:
            torch.cuda.synchronize()
        runs.append((time.perf_counter() - t0) * 1000.0)

    ms_mean = float(np.mean(runs))
    ms_min  = float(np.min(runs))
    flops   = 6.0 * H * I * T * top_k                # gate + up + down
    tflops_peak = flops / (ms_min  * 1e-3) / 1e12
    tflops_mean = flops / (ms_mean * 1e-3) / 1e12

    routing, avg_m_e = classify_routing(G, expert_ids, m_cpu, T, top_k)
    return BenchResult(G, H, I, T, top_k, m_cpu,
                       ms_mean, ms_min, tflops_peak, tflops_mean,
                       avg_m_e, routing)


def print_markdown_table(results: list[BenchResult]) -> None:
    if not results:
        print("(no results)")
        return
    print(
        "| G  |   H  |   I  |   T  | top_k | m_cpu | avg m_e | routing       "
        "| time (ms) |  TFLOPS  |"
    )
    print(
        "|----|------|------|------|-------|-------|---------|---------------"
        "|-----------|----------|"
    )
    for r in results:
        m_cpu_s = "inf" if r.m_cpu >= INF_M_CPU else str(r.m_cpu)
        print(
            f"| {r.G:<2} | {r.H:<4} | {r.I:<4} | {r.T:<4} | {r.top_k:<5} "
            f"| {m_cpu_s:<5} | {r.avg_m_e:7.1f} | {r.routing:<13} "
            f"| {r.ms_mean:9.3f} | {r.tflops_peak:8.3f} |"
        )


# -- Config grid -------------------------------------------------------------

# (G, H, I, T, top_k)
QUICK_CONFIGS = [
    (4,  512, 1024,  16, 2),
    (4,  512, 1024,  64, 2),
    (8, 1024, 2048,  64, 2),
]
QUICK_M_CPU = [0, 8, INF_M_CPU]

FULL_CONFIGS = [
    # tiny — decode-shaped
    (4,  512, 1024,    8, 2),
    (4,  512, 1024,   64, 2),
    (4,  512, 1024,  256, 2),
    # mid — prefill-shaped
    (8, 1024, 2048,    8, 2),
    (8, 1024, 2048,   64, 2),
    (8, 1024, 2048,  256, 2),
    (8, 1024, 2048, 1024, 2),
    # large — production MoE shapes
    (32, 2048, 4096,  128, 2),
    (32, 2048, 4096,  512, 2),
    (32, 2048, 4096, 2048, 2),
]
FULL_M_CPU = [0, 1, 4, 16, 64, 256, INF_M_CPU]


# -- Entry point -------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="Minimal smoke sweep (~30 s).")
    parser.add_argument("--cpu-threads", type=int, default=16,
                        help="cpu_gemm worker threads (default 16).")
    parser.add_argument("--iters", type=int, default=5,
                        help="Timing iterations after warmup (default 5).")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup calls before timing (default 3).")
    args = parser.parse_args()

    if not CPU_HAS_AMX:
        print("[SKIP] AMX-INT8 not present on this host (CPU bucket disabled).")
        sys.exit(0)

    print(f"asym_gemm.unified_moe perf bench")
    print(f"  host       : AMX-INT8={CPU_HAS_AMX}, CUDA={GPU_AVAILABLE}, "
          f"device={(torch.cuda.get_device_name(0) if GPU_AVAILABLE else '-')}")
    print(f"  cpu_threads: {args.cpu_threads}")
    print(f"  iters      : {args.iters} (warmup {args.warmup})")
    if not GPU_AVAILABLE:
        print("  NOTE: no CUDA — m_cpu=0 / mixed configs will be skipped.")
    print()

    configs = QUICK_CONFIGS if args.quick else FULL_CONFIGS
    m_cpu_sweep = QUICK_M_CPU if args.quick else FULL_M_CPU

    results: list[BenchResult] = []
    for (G, H, I, T, top_k) in configs:
        print(f"--- (G={G}, H={H}, I={I}, T={T}, top_k={top_k}) ---")
        for m_cpu in m_cpu_sweep:
            # Skip GPU-bucket configs when CUDA is absent.
            avg_m_e = (T * top_k) / G  # uniform-routing estimate
            if not GPU_AVAILABLE and m_cpu < avg_m_e:
                continue
            try:
                r = bench_one(G, H, I, T, top_k, m_cpu, args.cpu_threads,
                              iters=args.iters, warmup=args.warmup,
                              seed=(G * 131 + T))
            except Exception as e:    # noqa: BLE001
                m_cpu_s = "inf" if m_cpu >= INF_M_CPU else str(m_cpu)
                print(f"  [m_cpu={m_cpu_s}] FAILED: {type(e).__name__}: {e}")
                continue
            results.append(r)
            m_cpu_s = "inf" if r.m_cpu >= INF_M_CPU else str(r.m_cpu)
            print(f"  [m_cpu={m_cpu_s:<4}] {r.routing:<13} "
                  f"avg_m_e={r.avg_m_e:5.1f}  "
                  f"time(min/mean)={r.ms_min:.3f}/{r.ms_mean:.3f} ms  "
                  f"peak={r.tflops_peak:6.3f} / mean={r.tflops_mean:6.3f} TFLOPS")
        print()

    print("\n=== Summary ===\n")
    print_markdown_table(results)


if __name__ == "__main__":
    main()
