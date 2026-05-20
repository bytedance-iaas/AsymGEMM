#!/usr/bin/env python3
"""Run Nsight Compute on AsymGEMM kernels for fundamental workloads."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.postprocess_ncu_asymgemm import summarize, markdown


DEFAULT_SECTIONS = [
    "SpeedOfLight",
    "ComputeWorkloadAnalysis",
    "MemoryWorkloadAnalysis",
    "MemoryWorkloadAnalysis_Tables",
    "SchedulerStats",
    "WarpStateStats",
    "Occupancy",
    "LaunchStats",
    "SpeedOfLight_HierarchicalTensorRooflineChart",
]


QUICK_SECTIONS = [
    "SpeedOfLight",
    "LaunchStats",
    "Occupancy",
]


DEFAULT_LAUNCH_SKIP = {
    "matrix_1b": 2,
    "mlp_1b": 4,
}


DEFAULT_LAUNCH_COUNT = {
    "matrix_1b": 2,
    "mlp_1b": 4,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=sorted(DEFAULT_LAUNCH_SKIP), required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--backend", default="asym_only")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=1)
    parser.add_argument("--launch-skip", type=int)
    parser.add_argument("--launch-count", type=int)
    parser.add_argument("--preset", choices=["quick", "paper"], default="paper")
    parser.add_argument("--section", action="append", dest="sections")
    parser.add_argument("--ncu-bin", default="ncu")
    parser.add_argument("--output-root", type=Path, default=Path("profiling"))
    parser.add_argument("--clear-jit-cache", action="store_true")
    parser.add_argument("--jit-cache-dir", type=Path)
    parser.add_argument("--extra-ncu-arg", action="append", default=[])
    args = parser.parse_args()

    out_dir = args.output_root / args.workload / "ncu"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = out_dir / "source_debug"
    raw_csv = out_dir / "raw.csv"
    report_prefix = out_dir / "report"
    output_json = out_dir / "profile.json"
    output_md = out_dir / "table.md"

    env = os.environ.copy()
    env["DG_JIT_WITH_LINEINFO"] = env.get("DG_JIT_WITH_LINEINFO", "1")
    if args.jit_cache_dir is not None:
        env["DG_JIT_CACHE_DIR"] = str(args.jit_cache_dir)
    if args.clear_jit_cache:
        cache_dir = Path(env.get("DG_JIT_CACHE_DIR", str(Path.home() / ".asym_gemm"))) / "cache"
        shutil.rmtree(cache_dir, ignore_errors=True)

    sections = args.sections or (QUICK_SECTIONS if args.preset == "quick" else DEFAULT_SECTIONS)
    cmd = [
        args.ncu_bin,
        "--target-processes",
        "all",
        "--kernel-name-base",
        "demangled",
        "-k",
        "regex:.*asym_gemm::sm90_bf16_asym_gemm_impl.*",
        "--launch-skip",
        str(args.launch_skip if args.launch_skip is not None else DEFAULT_LAUNCH_SKIP[args.workload]),
        "--launch-count",
        str(args.launch_count if args.launch_count is not None else DEFAULT_LAUNCH_COUNT[args.workload]),
        "--page",
        "raw",
        "--csv",
        "--log-file",
        str(raw_csv),
        "--export",
        str(report_prefix),
        "--force-overwrite",
    ]
    for section in sections:
        cmd += ["--section", section]
    cmd += args.extra_ncu_arg
    cmd += [
        sys.executable,
        str(ROOT / "scripts/profile_m4_steps.py"),
        "--workload",
        args.workload,
        "--device",
        args.device,
        "--backend",
        args.backend,
        "--warmup-steps",
        str(args.warmup_steps),
        "--measure-steps",
        str(args.measure_steps),
        "--timing-mode",
        "profile",
        "--output-dir",
        str(source_dir),
    ]

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    report = summarize(raw_csv, workload=args.workload)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {output_md}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
