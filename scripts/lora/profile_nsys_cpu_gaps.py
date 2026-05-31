#!/usr/bin/env python3
"""Capture an Nsight Systems CPU-debug trace and summarize GPU no-kernel gaps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lora.postprocess_nsys_cpu_gaps import markdown, summarize_cpu_gaps  # noqa: E402


WORKLOADS = [
    "mlp",
    "dense",
    "moe",
    "dense_3b",
    "dense_14b",
    "moe-604m-a75m",
    "moe-604m-a38m",
    "mm_1b",
    "mm_3b",
    "mlp_1b",
    "mlp_3b",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--backend", default="asym")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp8", "fp4"])
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=1)
    parser.add_argument("--moe-mode", choices=["contiguous", "masked"], default="contiguous")
    parser.add_argument("--moe-route-pattern", choices=["balanced", "learned"], default="balanced")
    parser.add_argument("--hf-layer-index", type=int, default=0)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--hf-local-files-only", action="store_true")
    parser.add_argument("--profile-seed", type=int, default=1234)
    parser.add_argument("--profile-layers", "--real-profile-layers", dest="real_profile_layers", metavar="N", type=int, default=1)
    parser.add_argument("--batch-size", "--real-batch-size", dest="real_batch_size", metavar="N", type=int, default=1)
    parser.add_argument("--seq-len", "--real-seq-len", dest="real_seq_len", metavar="N", type=int, default=64)
    parser.add_argument("--tokens", "--real-tokens", dest="real_tokens", metavar="N", type=int, default=0)
    parser.add_argument("--lora-rank", "--real-lora-rank", dest="real_lora_rank", metavar="N", type=int, default=64)
    parser.add_argument("--lora-alpha", "--real-lora-alpha", dest="real_lora_alpha", metavar="FLOAT", type=float, default=128.0)
    parser.add_argument("--vocab-rows", "--real-vocab-rows", dest="real_vocab_rows", metavar="N", type=int, default=4096)
    parser.add_argument("--expert-recompute-threshold", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("profiling"))
    parser.add_argument("--output-dir", type=Path, help="Exact output directory. Overrides --output-root/<workload>/cpu_gaps.")
    parser.add_argument("--nsys-bin", default="nsys")
    parser.add_argument("--trace", default="cuda,nvtx")
    parser.add_argument("--sample", default="none")
    parser.add_argument(
        "--wait",
        choices=["primary", "all"],
        default="primary",
        help="Nsight wait mode. primary avoids waiting on re-parented helper processes after the target exits.",
    )
    parser.add_argument(
        "--cpuctxsw",
        choices=["process-tree", "system-wide", "none"],
        default="none",
        help="Nsight Systems context-switch scope. This Nsight install rejects --cpuctxsw=true.",
    )
    parser.add_argument("--extra-nsys-arg", action="append", default=[])
    parser.add_argument("--max-gap-rows", type=int, default=50)
    parser.add_argument("--max-stacks", type=int, default=20)
    parser.add_argument("--force-overwrite", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_dir = args.output_dir if args.output_dir is not None else args.output_root / args.workload / "cpu_gaps"
    source_dir = out_dir / "source_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    report_prefix = out_dir / "trace"
    nsys_rep = out_dir / "trace.nsys-rep"
    sqlite_path = out_dir / "trace.sqlite"
    output_json = out_dir / "profile.json"
    output_md = out_dir / "table.md"

    env = os.environ.copy()
    env["DG_JIT_WITH_LINEINFO"] = env.get("DG_JIT_WITH_LINEINFO", "1")
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        args.nsys_bin,
        "profile",
        f"--trace={args.trace}",
        f"--sample={args.sample}",
        f"--cpuctxsw={args.cpuctxsw}",
        f"--wait={args.wait}",
        f"--force-overwrite={'true' if args.force_overwrite else 'false'}",
        f"--output={report_prefix}",
        *args.extra_nsys_arg,
        sys.executable,
        str(ROOT / "scripts/lora/profile_lora_e2e.py"),
        "--workload",
        args.workload,
        "--device",
        args.device,
        "--backend",
        args.backend,
        "--precision",
        args.precision,
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
        "--profile-layers",
        str(args.real_profile_layers),
        "--batch-size",
        str(args.real_batch_size),
        "--seq-len",
        str(args.real_seq_len),
        "--tokens",
        str(args.real_tokens),
        "--lora-rank",
        str(args.real_lora_rank),
        "--lora-alpha",
        str(args.real_lora_alpha),
        "--vocab-rows",
        str(args.real_vocab_rows),
        "--expert-recompute-threshold",
        str(args.expert_recompute_threshold),
        "--output-dir",
        str(source_dir),
    ]
    if args.hf_cache_dir:
        cmd.extend(["--hf-cache-dir", str(args.hf_cache_dir)])
    if args.hf_local_files_only:
        cmd.append("--hf-local-files-only")

    command_txt = out_dir / "command.json"
    command_txt.write_text(json.dumps({"command": cmd, "cwd": str(ROOT)}, indent=2) + "\n", encoding="utf-8")
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    export_cmd = [
        args.nsys_bin,
        "export",
        "--type=sqlite",
        "--force-overwrite=true",
        f"--output={sqlite_path}",
        str(nsys_rep),
    ]
    print("Exporting:", " ".join(export_cmd), flush=True)
    subprocess.run(export_cmd, cwd=ROOT, env=env, check=True)

    report = summarize_cpu_gaps(sqlite_path, max_gap_rows=args.max_gap_rows, max_stacks=args.max_stacks)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {output_md}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
