"""Throughput (INT8 TOPS) benchmark for the SM90 int8 asym grouped GEMM.

Sweeps a range of experts (num_groups) and tokens-per-expert (M) on the masked
MoE path: a[G, M, K] @ b[G, N, K].mT -> d[G, M, N] (FP32 out).

Useful compute per call = G * 2 * M * N * K MACs-as-FLOPs (all tokens valid).
The dense INT8 tensor-core peak used for % utilization is picked from the
device name (H20 ~= 293 TOPS, H100/H200 SXM ~= 1979); override with
--peak-tops if the detection is wrong for your part.
"""
import argparse
import torch
import asym_gemm
from asym_gemm.testing import bench, get_arch_major
from asym_gemm.utils.math import per_channel_cast_to_int8, per_token_cast_to_int8

GRAN = 128

# Dense INT8 tensor-core peak (TOPS) by device, for % utilization.
_KNOWN_PEAKS = [
    ("H200", 1979.0),
    ("H100", 1979.0),
    ("H800", 1979.0),
    ("H20", 293.0),   # checked after H200 — substring order matters
]


def detect_peak_int8_tops() -> float:
    name = torch.cuda.get_device_name(0)
    for part, peak in _KNOWN_PEAKS:
        if part in name:
            return peak
    print(f"[WARN] unknown device '{name}' — assuming H100/H200-class INT8 peak; "
          "pass --peak-tops to correct the utilization figures.")
    return 1979.0


def make_inputs(num_groups, m, n, k, pinned_weights=False):
    kb = k // GRAN
    a = torch.randn(num_groups, m, k, device="cuda") / k ** 0.25
    b = torch.randn(num_groups, n, k, device="cuda") / k ** 0.25
    aq = torch.empty_like(a, dtype=torch.int8)
    bq = torch.empty_like(b, dtype=torch.int8)
    sfa = torch.empty(num_groups, m, kb, device="cuda")
    sfb = torch.empty(num_groups, n, kb, device="cuda")
    for g in range(num_groups):
        aq[g], sfa[g] = per_token_cast_to_int8(a[g])
        bq[g], sfb[g] = per_channel_cast_to_int8(b[g])
    # PCIe-resident experts: weights B in CUDA-pinned host memory (SFB stays on GPU).
    if pinned_weights:
        bq = bq.cpu().pin_memory()
    masked_m = torch.full((num_groups,), m, device="cuda", dtype=torch.int32)
    d = torch.empty(num_groups, m, n, device="cuda", dtype=torch.float32)
    return aq, sfa, bq, sfb, masked_m, d


def run(num_groups, m, n, k, num_tests=20, pinned_weights=False):
    aq, sfa, bq, sfb, masked_m, d = make_inputs(num_groups, m, n, k, pinned_weights)

    def fn():
        asym_gemm.m_grouped_int8_asym_gemm_nt_masked(
            (aq, sfa), (bq, sfb), d, masked_m, m, recipe=(1, GRAN, GRAN)
        )

    fn()  # JIT compile before timing
    torch.cuda.synchronize()
    secs = bench(fn, num_warmups=5, num_tests=num_tests)

    flops = num_groups * 2.0 * m * n * k          # useful MAC-FLOPs
    tops = flops / secs / 1e12
    # Bytes of expert weights streamed per call (the PCIe-bound term when pinned).
    weight_gib = num_groups * n * k / (1024 ** 3)  # int8 = 1 byte
    weight_bw = weight_gib / secs                  # GiB/s of B fetched
    # released after measuring so the sweep doesn't accumulate allocations
    del aq, sfa, bq, sfb, masked_m, d
    torch.cuda.empty_cache()
    return secs, tops, weight_bw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--experts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--tokens", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--pinned", action="store_true",
                    help="place expert weights B in CUDA-pinned host memory (PCIe-resident)")
    ap.add_argument("--peak-tops", type=float, default=None,
                    help="dense INT8 peak TOPS for %% utilization (default: by device name)")
    args = ap.parse_args()

    assert get_arch_major() == 9, "SM90 (Hopper) required"
    peak = args.peak_tops if args.peak_tops is not None else detect_peak_int8_tops()
    n, k, pw = args.n, args.k, args.pinned
    where = "pinned-CPU (PCIe)" if pw else "HBM"
    print(f"\nSM90 INT8 asym grouped GEMM throughput   (N={n}, K={k}, FP32 out, B in {where})")
    print(f"device {torch.cuda.get_device_name(0)}, peak INT8 ~= {peak:.0f} TOPS (dense)\n")
    header = "  G\\M  | " + " | ".join(f"{m:>6d}" for m in args.tokens)
    print(header)
    print("-" * len(header))
    errors = []
    for g in args.experts:
        cells = []
        for m in args.tokens:
            try:
                secs, tops, _ = run(g, m, n, k, pinned_weights=pw)
                cells.append(f"{tops:6.0f}")
            except Exception as e:  # OOM or unsupported shape — reported below
                errors.append((g, m, f"{type(e).__name__}: {e}"))
                cells.append("   err")
        print(f"  {g:>4d} | " + " | ".join(cells))
    print("\n(values are INT8 TOPS = G*2*M*N*K / time; divide by "
          f"{peak:.0f} for utilization)\n")
    if errors:
        print("Errored cells:")
        for g, m, msg in errors:
            print(f"   G={g:>3d} M={m:>5d}: {msg}")
        print()

    # A couple of detailed single-point reports (latency + utilization + weight BW).
    print("Detailed points:")
    for (g, m) in [(8, 512), (16, 1024), (32, 2048)]:
        if g in args.experts and m in args.tokens:
            secs, tops, wbw = run(g, m, n, k, pinned_weights=pw)
            print(f"   G={g:>3d} M={m:>5d} N={n} K={k}: "
                  f"{secs*1e6:8.1f} us   {tops:6.0f} TOPS   "
                  f"{100*tops/peak:4.1f}% of peak   "
                  f"B fetch {wbw:6.1f} GiB/s")


if __name__ == "__main__":
    main()
