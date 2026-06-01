#!/usr/bin/env python3
"""Run Nsight Compute on AsymGEMM kernels for LoRA-SFT workloads."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lora.postprocess_ncu_asymgemm import summarize, markdown

DEFAULT_LORA_BATCH_SIZE = 32
DEFAULT_LORA_SEQ_LEN = 64
DEFAULT_LORA_HIDDEN_DIM = 1024
DEFAULT_LORA_MLP_EXPANSION = 4
DEFAULT_DENSE_TARGET_MODE = "all"
DEFAULT_TARGET_MODULES = "all"
DEFAULT_OFFLOAD_MODULES = ""
DENSE_TARGET_MODES = ("mlp_only", "attention_only", "all")
LORA_DTYPE_CHOICES = ("bf16", "bfloat16", "fp16", "float16", "fp32", "float32")


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
    "mm_1b": 2,
    "mm_3b": 2,
    "mlp_1b": 4,
    "mlp_3b": 4,
    "dense_3b": 14,
    "dense_14b": 14,
    "moe-604m-a75m": 0,
    "moe-604m-a38m": 0,
}


DEFAULT_LAUNCH_COUNT = {
    "mm_1b": 2,
    "mm_3b": 2,
    "mlp_1b": 4,
    "mlp_3b": 4,
    "dense_3b": 14,
    "dense_14b": 14,
    "moe-604m-a75m": 32,
    "moe-604m-a38m": 32,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=sorted(DEFAULT_LAUNCH_SKIP), required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--backend", choices=["asym"], default="asym")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=1)
    parser.add_argument("--moe-mode", choices=["contiguous", "masked"], default="contiguous")
    parser.add_argument("--moe-route-pattern", choices=["balanced", "learned"], default="balanced")
    parser.add_argument("--hf-layer-index", type=int, default=0)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--hf-local-files-only", action="store_true")
    parser.add_argument("--profile-seed", type=int, default=1234)
    parser.add_argument("--target-preset", "--dense-target-mode", dest="dense_target_mode", choices=DENSE_TARGET_MODES, default=DEFAULT_DENSE_TARGET_MODE)
    parser.add_argument("--target-modules", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--offload-modules", default=DEFAULT_OFFLOAD_MODULES)
    parser.add_argument("--profile-layers", "--real-profile-layers", dest="profile_layers", type=int, default=1)
    parser.add_argument("--batch-size", "--real-batch-size", dest="batch_size", type=int, default=DEFAULT_LORA_BATCH_SIZE)
    parser.add_argument("--seq-len", "--real-seq-len", dest="seq_len", type=int, default=DEFAULT_LORA_SEQ_LEN)
    parser.add_argument("--hidden-dim", "--real-hidden-dim", dest="hidden_dim", type=int, default=DEFAULT_LORA_HIDDEN_DIM)
    parser.add_argument("--mlp-intermediate-dim", "--real-mlp-intermediate-dim", dest="mlp_intermediate_dim", type=int, default=0)
    parser.add_argument("--mlp-expansion", "--real-mlp-expansion", dest="mlp_expansion", type=int, default=DEFAULT_LORA_MLP_EXPANSION)
    parser.add_argument("--lora-rank", "--real-lora-rank", dest="lora_rank", type=int, default=64)
    parser.add_argument("--lora-alpha", "--real-lora-alpha", dest="lora_alpha", type=float, default=128.0)
    parser.add_argument("--lora-dtype", choices=LORA_DTYPE_CHOICES, default="bf16")
    parser.add_argument("--vocab-rows", "--real-vocab-rows", dest="vocab_rows", type=int, default=4096)
    parser.add_argument("--expert-recompute-policy", default="none")
    parser.add_argument("--launch-skip", type=int)
    parser.add_argument("--launch-count", type=int)
    parser.add_argument("--preset", choices=["quick", "paper"], default="paper")
    parser.add_argument("--section", action="append", dest="sections")
    parser.add_argument("--ncu-bin", default="ncu")
    parser.add_argument("--output-root", type=Path, default=Path("profiling"))
    parser.add_argument("--output-dir", type=Path, help="Exact output directory. Overrides --output-root/<workload>/ncu.")
    parser.add_argument("--clear-jit-cache", action="store_true")
    parser.add_argument("--jit-cache-dir", type=Path)
    parser.add_argument("--extra-ncu-arg", action="append", default=[])
    args = parser.parse_args()

    out_dir = args.output_dir if args.output_dir is not None else args.output_root / args.workload / "ncu"
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
        "regex:.*asym_gemm::sm(90|100)_bf16_asym_gemm_impl.*",
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
        str(ROOT / "scripts/lora/profile_lora_e2e.py"),
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
        "--moe-mode",
        args.moe_mode,
        "--moe-route-pattern",
        args.moe_route_pattern,
        "--hf-layer-index",
        str(args.hf_layer_index),
        "--profile-seed",
        str(args.profile_seed),
        "--dense-target-mode",
        args.dense_target_mode,
        "--target-modules",
        args.target_modules,
        "--offload-modules",
        args.offload_modules,
        "--profile-layers",
        str(args.profile_layers),
        "--batch-size",
        str(args.batch_size),
        "--seq-len",
        str(args.seq_len),
        "--hidden-dim",
        str(args.hidden_dim),
        "--mlp-intermediate-dim",
        str(args.mlp_intermediate_dim),
        "--mlp-expansion",
        str(args.mlp_expansion),
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dtype",
        args.lora_dtype,
        "--vocab-rows",
        str(args.vocab_rows),
        "--expert-recompute-policy",
        str(args.expert_recompute_policy),
        "--output-dir",
        str(source_dir),
    ]
    if args.hf_cache_dir:
        cmd.extend(["--hf-cache-dir", str(args.hf_cache_dir)])
    if args.hf_local_files_only:
        cmd.append("--hf-local-files-only")

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    report = summarize(raw_csv, workload=args.workload)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {output_md}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
