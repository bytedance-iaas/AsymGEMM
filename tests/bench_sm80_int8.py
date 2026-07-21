# tests/bench_sm80_int8.py
"""
Performance benchmark for the SM80 INT8 asym MoE kernel
(m_grouped_int8_asym_gemm_sm80_contiguous).

Designed for the A100 estimation workflow (unified_kernel_sm80.md Phase 5):

  1. On the H100/H200 dev box:   python tests/bench_sm80_int8.py \
                                     --save bench_results/sm80_int8_h200.json
  2. On the A100 server:         python tests/bench_sm80_int8.py \
                                     --save bench_results/sm80_int8_a100.json \
                                     --baseline bench_results/sm80_int8_h200.json

The kernel's arch guard is >= 800, so the SAME code path runs on both boxes;
the baseline comparison turns the H100-recorded numbers into falsifiable A100
estimates (expected ratios: PCIe-bound ~1.0x, HBM-bound ~0.6x of H100 /
~0.42x of H200, tensor-pipe-bound 0.32-0.58x of H100).

What it measures:
  * copy-engine H2D bandwidth (pinned -> HBM) — the PCIe ceiling reference
  * streamed benches (B in pinned host): kernel-achieved PCIe GB/s — THE
    number that decides the asym side's viability on A100 (target: >= ~85%
    of the copy-engine ceiling)
  * HBM benches (B in HBM): INT8 TOPS + estimated HBM traffic GB/s
  * a small parity sanity check before timing (aborts on numerical failure)

Run:  python tests/bench_sm80_int8.py [--save out.json] [--baseline ref.json]
                                      [--quick]
"""
import argparse
import datetime
import json
import os
import statistics
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import asym_gemm  # noqa: E402

# Dense (no-sparsity) datasheet peaks for utilization context. Keyed by
# substring of torch.cuda.get_device_name(). HBM GB/s, INT8 tensor TOPS.
KNOWN_PEAKS = [
    ("H200", {"hbm_gbs": 4800.0, "int8_tops": 1979.0}),
    ("H100", {"hbm_gbs": 3350.0, "int8_tops": 1979.0}),
    ("A100", {"hbm_gbs": 2039.0, "int8_tops": 624.0}),   # 80GB; 40GB -> 1555
    ("A800", {"hbm_gbs": 2039.0, "int8_tops": 624.0}),
]


def device_peaks(name: str, total_mem_gb: float):
    for key, peaks in KNOWN_PEAKS:
        if key in name:
            p = dict(peaks)
            if key in ("A100", "A800") and total_mem_gb < 60:
                p["hbm_gbs"] = 1555.0   # 40 GB SKU
            return p
    return None


def cuda_time_ms(fn, warmup=3, iters=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return min(times), statistics.median(times)


def make_case(S, m_per_expert, N, K, b_pinned, seed=0):
    """Build one contiguous grouped-GEMM case (pair-style segment layout)."""
    torch.manual_seed(seed)
    dev = "cuda"
    kb = K // 128
    lens = [m_per_expert] * S
    starts = [i * m_per_expert for i in range(S)]
    ends = [s + m_per_expert for s in starts]
    M = S * m_per_expert

    offsets = torch.tensor([v for se in zip(starts, ends) for v in se],
                           dtype=torch.int32, device=dev)
    experts = torch.tensor(list(range(S)) + [-1], dtype=torch.int32, device=dev)

    a = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
    sfa = (torch.rand(M, kb, device=dev) * 0.9 + 0.1) * 0.01
    if b_pinned:
        b = torch.randint(-127, 128, (S, N, K), dtype=torch.int8).pin_memory()
        sfb = ((torch.rand(S, N, kb) * 0.9 + 0.1) * 0.01).pin_memory()
    else:
        b = torch.randint(-127, 128, (S, N, K), dtype=torch.int8, device=dev)
        sfb = (torch.rand(S, N, kb, device=dev) * 0.9 + 0.1) * 0.01
    d = torch.zeros(M, N, dtype=torch.float32, device=dev)

    def run():
        asym_gemm.m_grouped_int8_asym_gemm_sm80_contiguous(
            a, b, d, offsets, experts, S + 1, sfa, sfb)

    stats = {
        "flops": 2.0 * M * N * K,
        "w_bytes": float(S * N * K),                       # int8 weights
        # D partial-sum traffic: write M*N*4 per K-block + read M*N*4 per
        # K-block after the first; A read once; all in HBM.
        "hbm_bytes": float(M * K + M * N * 4 * (2 * (K // 128) - 1)),
        "M": M, "N": N, "K": K, "S": S,
    }
    return run, stats


def sanity_check():
    """Tiny parity check against a float64 reference; abort on failure."""
    torch.manual_seed(7)
    dev = "cuda"
    S, m, N, K = 2, 70, 128, 256
    kb = K // 128
    a = torch.randint(-127, 128, (S * m, K), dtype=torch.int8, device=dev)
    sfa = (torch.rand(S * m, kb, device=dev) + 0.1) * 0.01
    b = torch.randint(-127, 128, (S, N, K), dtype=torch.int8).pin_memory()
    sfb = ((torch.rand(S, N, kb) + 0.1) * 0.01).pin_memory()
    d = torch.zeros(S * m, N, dtype=torch.float32, device=dev)
    offsets = torch.tensor([0, m, m, 2 * m], dtype=torch.int32, device=dev)
    experts = torch.tensor([0, 1, -1], dtype=torch.int32, device=dev)
    asym_gemm.m_grouped_int8_asym_gemm_sm80_contiguous(
        a, b, d, offsets, experts, S + 1, sfa, sfb)
    torch.cuda.synchronize()

    bd, sd = b.cuda().to(torch.float64), sfb.cuda().to(torch.float64)
    ref = torch.zeros(S * m, N, dtype=torch.float64, device=dev)
    for s in range(S):
        rows = slice(s * m, (s + 1) * m)
        for k in range(kb):
            sl = slice(k * 128, (k + 1) * 128)
            ref[rows] += ((a[rows, sl].to(torch.float64) @ bd[s][:, sl].T)
                          * sfa[rows, k, None].to(torch.float64)
                          * sd[s, None, :, k])
    rel = ((d.to(torch.float64) - ref).abs().max() / ref.abs().max()).item()
    assert rel < 1e-4, f"parity sanity FAILED: rel={rel:.3e}"
    print(f"parity sanity: OK (rel={rel:.2e})")


def pcie_copy_baseline_gbs(nbytes=256 << 20, iters=8):
    src = torch.empty(nbytes, dtype=torch.uint8).pin_memory()
    dst = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    t_min, _ = cuda_time_ms(lambda: dst.copy_(src, non_blocking=True),
                            warmup=2, iters=iters)
    return nbytes / (t_min * 1e-3) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=str, default=None,
                    help="write results JSON to this path")
    ap.add_argument("--baseline", type=str, default=None,
                    help="compare against a previously saved results JSON")
    ap.add_argument("--quick", action="store_true",
                    help="fewer iterations / smaller shape set")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA device required"
    name = torch.cuda.get_device_name(0)
    cc = "%d.%d" % torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    peaks = device_peaks(name, total_gb)
    print(f"device: {name} (sm_{cc.replace('.', '')}, {total_gb:.0f} GB)")
    if peaks:
        print(f"datasheet peaks: HBM {peaks['hbm_gbs']:.0f} GB/s, "
              f"INT8 {peaks['int8_tops']:.0f} TOPS (dense)")

    sanity_check()

    pcie_gbs = pcie_copy_baseline_gbs()
    print(f"copy-engine H2D (pinned->HBM, 256 MB): {pcie_gbs:.1f} GB/s "
          f"<- PCIe ceiling reference\n")

    # (tag, S, m/expert, N, K) — K multiple of 128.
    shapes = [
        ("prefill_up",   8, 256, 4096, 2048),
        ("prefill_down", 8, 256, 2048, 4096),
        ("decode",       8,  16, 4096, 2048),
        ("deep_k",       4, 256, 4096, 7168),
    ]
    if args.quick:
        shapes = shapes[:2]
    iters = 6 if args.quick else 15

    results = {
        "device": name, "cc": cc, "total_mem_gb": round(total_gb, 1),
        "torch": torch.__version__,
        "date": datetime.date.today().isoformat(),
        "pcie_copy_h2d_gbs": round(pcie_gbs, 2),
        "benches": {},
    }

    header = (f"{'bench':<28} {'ms(min)':>9} {'ms(med)':>9} "
              f"{'TOPS':>7} {'PCIe GB/s':>10} {'HBM GB/s':>9}")
    print(header)
    print("-" * len(header))
    for b_pinned in (True, False):
        for tag, S, m, N, K in shapes:
            key = f"{tag}_{'pinned' if b_pinned else 'hbm'}"
            run, st = make_case(S, m, N, K, b_pinned)
            ms_min, ms_med = cuda_time_ms(run, warmup=3, iters=iters)
            tops = st["flops"] / (ms_min * 1e-3) / 1e12
            pcie = (st["w_bytes"] / (ms_min * 1e-3) / 1e9) if b_pinned else 0.0
            hbm = ((st["hbm_bytes"] + (0 if b_pinned else st["w_bytes"]))
                   / (ms_min * 1e-3) / 1e9)
            results["benches"][key] = {
                "ms_min": round(ms_min, 4), "ms_med": round(ms_med, 4),
                "tops": round(tops, 2),
                "pcie_gbs": round(pcie, 2), "hbm_gbs_est": round(hbm, 2),
                "M": st["M"], "N": N, "K": K, "S": S,
            }
            pcie_s = f"{pcie:10.1f}" if b_pinned else f"{'-':>10}"
            print(f"{key:<28} {ms_min:9.3f} {ms_med:9.3f} "
                  f"{tops:7.2f} {pcie_s} {hbm:9.1f}")

    if peaks:
        best_tops = max(v["tops"] for v in results["benches"].values())
        print(f"\nbest INT8 utilization: "
              f"{100 * best_tops / peaks['int8_tops']:.1f}% of dense peak")
    best_pcie = max(v["pcie_gbs"] for v in results["benches"].values())
    print(f"best kernel PCIe streaming: {best_pcie:.1f} GB/s "
          f"({100 * best_pcie / pcie_gbs:.0f}% of copy-engine ceiling; "
          f"unified_kernel_sm80.md gate: >= ~85%)")

    if args.baseline:
        with open(args.baseline) as f:
            base = json.load(f)
        print(f"\nvs baseline {base['device']} ({args.baseline}):")
        print(f"  copy-engine H2D ratio: "
              f"{pcie_gbs / base['pcie_copy_h2d_gbs']:.2f}x")
        print(f"{'bench':<28} {'this ms':>9} {'base ms':>9} {'speed':>7}")
        for key, v in results["benches"].items():
            bv = base.get("benches", {}).get(key)
            if bv:
                print(f"{key:<28} {v['ms_min']:9.3f} {bv['ms_min']:9.3f} "
                      f"{bv['ms_min'] / v['ms_min']:6.2f}x")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nsaved: {args.save}")


if __name__ == "__main__":
    main()
