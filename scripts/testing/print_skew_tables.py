#!/usr/bin/env python3
"""Assemble the two skew tables (fix_gb200_ep_v2.md GOALS) as markdown.

Table 1 (micro): profiling_both_skew/table1_micro.json from ep_balance_bench.
Table 2 (e2e):   run dirs under profiling_both_skew/ (and any extra roots passed
                 via --e2e-root) written by profile_lora_lf_test_both/_source.

Usage:
  python3 scripts/testing/print_skew_tables.py \
      [--micro profiling_both_skew/table1_micro.json] \
      [--e2e-root profiling_both_skew] [--warmup 1]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

# mode keys per row; "sep" is the pre-2026-07-10 spelling of "plan" (old jsons)
MICRO_ROWS = [(("owned",), "EP (owned split)"), (("sdp",), "DP (all banks/GPU)"),
              (("plan", "sep"), "ours (plan)"), (("queue",), "ours (queue)")]
MICRO_COLS = [
    ("zipf0.0", "uniform"),
    ("zipf0.5", "zipf 0.5"),
    ("zipf0.8", "zipf 0.8"),
    ("zipf1.0", "zipf 1.0"),
    ("zipf1.5", "zipf 1.5"),
    ("zipf2.0", "zipf 2.0"),
    ("median:", "real median"),
    ("worst:", "real worst"),
]
E2E_ROWS = [
    ("asym_ep2_cpuadamwds", "EP (owned)"),
    ("asym_sdp2_cpuadamwds", "ours, no queue"),
    ("asym_sqdp2_cpuadamwds", "ours, queue"),
]
E2E_COLS = [(None, "natural"), ("05", "zipf 0.5"), ("08", "zipf 0.8"), ("10", "zipf 1.0"), ("20", "zipf 2.0")]
E2E_TOKENS_PER_STEP = 320_000  # 20000 seq x 8 batch x 2 ranks


def micro_table(path: str) -> list[str]:
    data = json.load(open(path))
    cells: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for case in data["cases"]:
        name = case["case"]
        for prefix, _ in MICRO_COLS:
            if name.startswith(prefix.rstrip(":")) if prefix.endswith(":") else name.startswith(prefix + "|"):
                for keys, _ in MICRO_ROWS:
                    for mode in keys:
                        if mode in case:
                            cells[prefix][keys[0]].append(case[mode])
                            break
    lines = ["| system | " + " | ".join(label for _, label in MICRO_COLS) + " |"]
    lines.append("|" + "---|" * (len(MICRO_COLS) + 1))
    for keys, row_label in MICRO_ROWS:
        mode = keys[0]
        parts = [row_label]
        for prefix, _ in MICRO_COLS:
            trials = cells[prefix].get(mode)
            if not trials:
                parts.append("—")
                continue
            walls = [t["wall_s"] * 1e3 for t in trials]
            imbs = [t["imbalance"] for t in trials]
            gbs = [max(t["b_mb"]) / 1024.0 for t in trials]
            if len(trials) > 1:
                cell = f"{sum(walls)/len(walls):.1f}/{max(walls):.1f} ms · {100*sum(imbs)/len(imbs):.0f}% · {sum(gbs)/len(gbs):.2f} GB"
            else:
                cell = f"{walls[0]:.1f} ms · {100*imbs[0]:.0f}% · {gbs[0]:.2f} GB"
            parts.append(cell)
        lines.append("| " + " | ".join(parts) + " |")
    lines.append("")
    lines.append("zipf cells: mean/worst wall over 3 seeded expert-ID shuffles; % = mean GPU "
                 "imbalance; GB = mean per-GPU max weight bytes streamed.")
    return lines


def _steady(step_samples: str, warmup: int) -> tuple[float | None, float | None]:
    rows = json.load(open(step_samples))
    measured = [r for r in rows if int(r.get("raw_step", 0)) > warmup]
    if not measured:
        return None, None
    secs = [float(r["step_milliseconds"]) / 1e3 for r in measured]
    losses = [float(r["loss"]) for r in measured if r.get("loss") is not None]
    return sum(secs) / len(secs), (losses[-1] if losses else None)


def _find_run(root: str, backend: str, ztag: str | None) -> str | None:
    hits = []
    for p in glob.glob(os.path.join(root, "*", "qwen3-30b-a3b__gpus2__b8_s20000_*", f"{backend}*", "*", "step_samples.json")):
        leaf = p.split(os.sep)[-3]  # backend-level dir
        has_z = re.search(r"_zipf(\d+)__", leaf)
        if ztag is None and has_z:
            continue
        if ztag is not None and (not has_z or has_z.group(1) != ztag):
            continue
        hits.append(p)
    return max(hits, key=os.path.getmtime) if hits else None


def e2e_table(roots: list[str], warmup: int) -> list[str]:
    lines = ["| system | " + " | ".join(label for _, label in E2E_COLS) + " |"]
    lines.append("|" + "---|" * (len(E2E_COLS) + 1))
    nat_losses: list[tuple[str, float | None]] = []
    for backend, row_label in E2E_ROWS:
        parts = [row_label]
        for ztag, _ in E2E_COLS:
            path = None
            for root in roots:
                path = path or _find_run(root, backend, ztag)
            if not path:
                parts.append("—")
                continue
            steady, loss = _steady(path, warmup)
            if steady is None:
                parts.append("—")
                continue
            parts.append(f"{steady:.1f} s · {E2E_TOKENS_PER_STEP/steady:,.0f} tok/s")
            if ztag is None:
                nat_losses.append((row_label, loss))
        lines.append("| " + " | ".join(parts) + " |")
    if nat_losses:
        lines.append("")
        lines.append("natural-row final losses: " + ", ".join(
            f"{name} {loss:.4f}" if loss is not None else f"{name} n/a" for name, loss in nat_losses))
    lines.append("")
    lines.append("cells: steady step seconds (mean of steps 2-4 of a 1+4 run) · tokens/s = "
                 f"{E2E_TOKENS_PER_STEP:,}/step_seconds. z rows are timing-only (loss invalid by design).")
    return lines


MICRO_TABLES = [
    ("profiling_both_skew/table1_micro.json",
     "Table 1 — q3-30b-a3b micro, one expert GEMM (128E top-8, N=768 K=2048, 5.12M rows)"),
    ("profiling_both_skew/table1b_experts.json",
     "Table 1b — q3-30b-a3b micro, experts block: gate+up GEMMs, SiLU*mul, down GEMM"),
    ("profiling_both_skew/table1c_moe.json",
     "Table 1c — q3-30b-a3b micro, MoE block: router + gather + experts block + combine"),
    ("profiling_both_skew/table1_q3235b_gemm.json",
     "Table 1d — q3-235b-a22b micro, one expert GEMM (128E top-8, N=1536 K=4096, 3.84M rows)"),
    ("profiling_both_skew/table1_q3235b_experts.json",
     "Table 1e — q3-235b-a22b micro, experts block"),
    ("profiling_both_skew/table1_q3235b_moe.json",
     "Table 1f — q3-235b-a22b micro, MoE block"),
    ("profiling_both_skew/table1_q35122b_gemm.json",
     "Table 1g — q3.5-122b-a10b micro, one expert GEMM (256E top-8, N=1024 K=3072, 5.12M rows)"),
    ("profiling_both_skew/table1_q35122b_experts.json",
     "Table 1h — q3.5-122b-a10b micro, experts block"),
    ("profiling_both_skew/table1_q35122b_moe.json",
     "Table 1i — q3.5-122b-a10b micro, MoE block (+ shared expert N=1024, mode-flat)"),
    ("profiling_both_skew/table1_l4scout_gemm.json",
     "Table 1j — llama4-scout micro, fused gate_up GEMM (16E top-1, N=8192 K=5120, 1.28M rows)"),
    ("profiling_both_skew/table1_l4scout_experts.json",
     "Table 1k — llama4-scout micro, experts block (fused gate_up + SiLU*mul + down)"),
    ("profiling_both_skew/table1_l4scout_moe.json",
     "Table 1l — llama4-scout micro, MoE block (+ shared expert N=8192, mode-flat)"),
    ("profiling_both_skew/table1_q330b_layer.json",
     "Table 1m — q3-30b-a3b micro, WHOLE LAYER (attention seq20k + MoE block)"),
    ("profiling_both_skew/table1_q3235b_layer.json",
     "Table 1n — q3-235b-a22b micro, WHOLE LAYER"),
    ("profiling_both_skew/table1_q35122b_layer.json",
     "Table 1o — q3.5-122b-a10b micro, WHOLE LAYER"),
    ("profiling_both_skew/table1_l4scout_layer.json",
     "Table 1p — llama4-scout micro, WHOLE LAYER"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro", action="append", default=None,
                    help="micro json (repeatable; default: the three table1*.json)")
    ap.add_argument("--e2e-root", action="append", default=None)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()
    roots = args.e2e_root or ["profiling_both_skew"]
    micros = [(p, f"Table 1 — micro ({p})") for p in args.micro] if args.micro else MICRO_TABLES

    for path, title in micros:
        if not os.path.exists(path):
            continue
        print(f"## {title}\n")
        print("\n".join(micro_table(path)))
        print()
    print("## Table 2 — e2e (q3-30b-a3b 20000|8|1, 2 GPUs)\n")
    print("\n".join(e2e_table(roots, args.warmup)))


if __name__ == "__main__":
    main()
