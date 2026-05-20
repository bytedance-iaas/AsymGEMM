#!/usr/bin/env python3
"""Run LoRA-SFT profiling workloads for AsymGEMM and the Torch fallback path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCRIPT = ROOT / "scripts" / "profile_lora.py"
NSYS_POSTPROCESS_SCRIPT = ROOT / "scripts" / "postprocess_nsys_m4.py"
CPU_GAPS_SCRIPT = ROOT / "scripts" / "profile_nsys_cpu_gaps.py"
NCU_SCRIPT = ROOT / "scripts" / "profile_ncu_asymgemm.py"

WORKLOADS = ("mlp", "dense", "moe", "qwen3_14b", "qwen3_30b_a3b", "matrix_1b", "mlp_1b")
WORKLOAD_ALIASES = {
    "toy": ("mlp", "dense", "moe"),
    "qwen": ("qwen3_14b", "qwen3_30b_a3b"),
    "fundamental": ("matrix_1b", "mlp_1b"),
    "all": WORKLOADS,
}
BACKENDS = ("asym_only", "torch_only")
PROFILERS = ("source", "nsys", "cpu", "ncu")
PROFILER_ALIASES = {
    "all": PROFILERS,
    "none": ("source",),
    "profile": ("source",),
    "cpu_gaps": ("cpu",),
}
NCU_WORKLOADS = {"matrix_1b", "mlp_1b", "qwen3_14b", "qwen3_30b_a3b"}


def _split_tokens(values: Iterable[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        tokens.extend(part.strip() for part in value.split(",") if part.strip())
    return tokens


def _expand_workloads(values: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for value in _split_tokens(values):
        if value in WORKLOAD_ALIASES:
            expanded.extend(WORKLOAD_ALIASES[value])
        elif value in WORKLOADS:
            expanded.append(value)
        else:
            allowed = ", ".join((*WORKLOADS, *WORKLOAD_ALIASES))
            raise SystemExit(f"unknown workload {value!r}; allowed: {allowed}")
    return list(dict.fromkeys(expanded))


def _expand_backends(values: Iterable[str]) -> list[str]:
    backends = _split_tokens(values)
    bad = [backend for backend in backends if backend not in BACKENDS]
    if bad:
        raise SystemExit(f"unknown backend(s) {bad}; allowed: {', '.join(BACKENDS)}")
    return list(dict.fromkeys(backends))


def _expand_profilers(values: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for value in _split_tokens(values):
        if value in PROFILER_ALIASES:
            expanded.extend(PROFILER_ALIASES[value])
        elif value in PROFILERS:
            expanded.append(value)
        else:
            allowed = ", ".join((*PROFILERS, *PROFILER_ALIASES))
            raise SystemExit(f"unknown profiler {value!r}; allowed: {allowed}")
    return list(dict.fromkeys(expanded))


def _cuda_devices(values: str | None) -> list[str]:
    if not values:
        return []
    devices = _split_tokens([values])
    cleaned: list[str] = []
    for device in devices:
        cleaned.append(device.removeprefix("cuda:"))
    return cleaned


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_label(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() or ch == "-" else "_" for ch in value).strip("_-")


def _backend_label(backend: str) -> str:
    return backend.replace("_", "-")


def _result_stem(args: argparse.Namespace, backend: str, profiler: str) -> str:
    mode = args.mode
    if mode == "auto":
        mode = f"{_backend_label(backend)}_{profiler}"
    return "_".join(
        part
        for part in (
            _safe_label(args.precision),
            _safe_label(args.workflow),
            _safe_label(mode),
        )
        if part
    )


def _raw_output_dir(run_dir: Path, workload: str, args: argparse.Namespace, backend: str, profiler: str) -> Path:
    return run_dir / workload / _result_stem(args, backend, profiler)


def _load_profile(output_dir: Path) -> tuple[dict[str, Any], Path | None]:
    candidates = [output_dir / "profile.json", *sorted(output_dir.glob("*_profile.json"))]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")), path
    return {}, None


def _find_markdown(output_dir: Path) -> Path | None:
    candidates = [output_dir / "table.md", *sorted(output_dir.glob("*_profile.md"))]
    for path in candidates:
        if path.exists():
            return path
    return None


def _write_result_aliases(args: argparse.Namespace, run_dir: Path, row: dict[str, Any]) -> dict[str, str]:
    if row.get("status") != "ok":
        return {}
    workload = str(row["workload"])
    backend = str(row["backend"])
    profiler = str(row["profiler"])
    output_dir = Path(str(row["output_dir"]))
    workload_dir = run_dir / workload
    workload_dir.mkdir(parents=True, exist_ok=True)
    stem = _result_stem(args, backend, profiler)
    aliases: dict[str, str] = {}

    markdown_path = _find_markdown(output_dir)
    if markdown_path is not None:
        target = workload_dir / f"{stem}.md"
        shutil.copyfile(markdown_path, target)
        aliases["table_md"] = str(target)

    profile_json = row.get("profile_json")
    json_path = Path(str(profile_json)) if profile_json else output_dir / "profile.json"
    if json_path.exists():
        target = workload_dir / f"{stem}.json"
        shutil.copyfile(json_path, target)
        aliases["profile_json"] = str(target)

    return aliases


def _common_profile_args(args: argparse.Namespace, workload: str, backend: str, device: str, output_dir: Path) -> list[str]:
    return [
        "--workload",
        workload,
        "--device",
        device,
        "--backend",
        backend,
        "--warmup-steps",
        str(args.warmup_steps),
        "--measure-steps",
        str(args.measure_steps),
        "--moe-mode",
        args.moe_mode,
        "--profile-layers",
        str(args.profile_layers),
        "--batch-size",
        str(args.batch_size),
        "--seq-len",
        str(args.seq_len),
        "--tokens",
        str(args.tokens),
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--vocab-rows",
        str(args.vocab_rows),
        "--output-dir",
        str(output_dir),
    ]


def _summary_row(
    *,
    workload: str,
    backend: str,
    profiler: str,
    device: str,
    physical_cuda_device: str | None,
    output_dir: Path,
    returncode: int,
    profile: dict[str, Any],
) -> dict[str, Any]:
    step = profile.get("step", {}) if isinstance(profile, dict) else {}
    memory = profile.get("memory", {}) if isinstance(profile, dict) else {}
    gpu_memory = memory.get("gpu", {}) if isinstance(memory, dict) else {}
    cpu_memory = memory.get("cpu", {}) if isinstance(memory, dict) else {}
    host_w_bytes = cpu_memory.get("host_w_bytes")
    host_w_t_bytes = cpu_memory.get("host_w_t_bytes")
    if isinstance(host_w_bytes, int) and isinstance(host_w_t_bytes, int):
        fallback_hbm_saved = host_w_bytes + host_w_t_bytes
    else:
        fallback_hbm_saved = None
    return {
        "workload": workload,
        "backend": backend,
        "profiler": profiler,
        "device": device,
        "physical_cuda_device": physical_cuda_device,
        "status": "ok" if returncode == 0 else "failed",
        "returncode": returncode,
        "step_ms": step.get("total_milliseconds"),
        "peak_hbm_bytes": memory.get("peak_hbm_bytes") or gpu_memory.get("peak_hbm_bytes"),
        "expected_hbm_saved_bytes": memory.get("expected_hbm_saved_bytes") or fallback_hbm_saved,
        "pinned_cpu_bytes": memory.get("pinned_cpu_bytes") or cpu_memory.get("pinned_total_bytes"),
        "output_dir": str(output_dir),
    }


def _add_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_workload: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (str(row["workload"]), str(row.get("profiler", "source")))
        by_workload.setdefault(key, {})[str(row["backend"])] = row
    comparisons: list[dict[str, Any]] = []
    for (workload, profiler), items in by_workload.items():
        asym = items.get("asym_only")
        torch = items.get("torch_only")
        if not asym or not torch:
            continue
        asym_ms = asym.get("step_ms")
        torch_ms = torch.get("step_ms")
        if not isinstance(asym_ms, (int, float)) or not isinstance(torch_ms, (int, float)) or asym_ms <= 0:
            continue
        comparisons.append(
            {
                "workload": workload,
                "profiler": profiler,
                "asym_step_ms": asym_ms,
                "torch_step_ms": torch_ms,
                "asym_vs_torch_speedup": torch_ms / asym_ms,
                "asym_vs_torch_step_ms_delta": asym_ms - torch_ms,
            }
        )
    return comparisons


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LoRA-SFT Profiling Workflow",
        "",
        f"Precision: `{summary.get('precision', '-')}`  ",
        f"Workflow: `{summary.get('workflow', '-')}`",
        "",
        "`asym_only` measures the direct AsymGEMM host-weight path. `torch_only` keeps the same",
        "host-weight wrapper and uses the PyTorch fallback path; it is not a GPU-resident",
        "LLaMA-Factory baseline.",
        "",
        "## Runs",
        "",
        "| Workload | Backend | Profiler | Status | Device | Step ms | Peak HBM | HBM saved | Pinned CPU | Result table | Output |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary["runs"]:
        def fmt(value: Any) -> str:
            if isinstance(value, float):
                return f"{value:.3f}"
            if isinstance(value, int):
                return str(value)
            if value is None:
                return "-"
            return str(value)

        device = row["device"]
        if row.get("physical_cuda_device") is not None:
            device = f"CUDA_VISIBLE_DEVICES={row['physical_cuda_device']}:{device}"
        lines.append(
            "| {workload} | {backend} | {profiler} | {status} | {device} | {step_ms} | {peak_hbm_bytes} | "
            "{expected_hbm_saved_bytes} | {pinned_cpu_bytes} | {result_table} | `{output_dir}` |".format(
                workload=row["workload"],
                backend=row["backend"],
                profiler=row.get("profiler", "source"),
                status=row["status"],
                device=device,
                step_ms=fmt(row.get("step_ms")),
                peak_hbm_bytes=fmt(row.get("peak_hbm_bytes")),
                expected_hbm_saved_bytes=fmt(row.get("expected_hbm_saved_bytes")),
                pinned_cpu_bytes=fmt(row.get("pinned_cpu_bytes")),
                result_table=f"`{row['result_aliases']['table_md']}`" if row.get("result_aliases", {}).get("table_md") else "-",
                output_dir=row["output_dir"],
            )
        )
    lines += ["", "## Comparisons", ""]
    if summary["comparisons"]:
        lines += [
            "| Workload | Profiler | Asym step ms | Torch step ms | Torch/Asym speedup | Asym minus Torch ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in summary["comparisons"]:
            lines.append(
                "| {workload} | {profiler} | {asym_step_ms:.3f} | {torch_step_ms:.3f} | "
                "{asym_vs_torch_speedup:.3f} | {asym_vs_torch_step_ms_delta:.3f} |".format(**row)
            )
    else:
        lines.append("No paired `asym_only`/`torch_only` comparisons were available.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["qwen"],
        help=f"Workload list or aliases. Workloads: {', '.join(WORKLOADS)}. Aliases: {', '.join(WORKLOAD_ALIASES)}.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["asym_only", "torch_only"],
        help="Backend list. Only asym_only and torch_only are accepted for clean comparison.",
    )
    parser.add_argument(
        "--profilers",
        "--profile-modes",
        nargs="+",
        default=["source"],
        help=(
            "Profiler list or aliases. source=profile_lora.py, nsys=Nsight Systems truth table, "
            "cpu=Nsight CPU-gap debug, ncu=Nsight Compute kernel metrics. Alias: all."
        ),
    )
    parser.add_argument("--device", default="cuda:0", help="Device passed to profile_lora.py when --cuda-devices is not set.")
    parser.add_argument(
        "--cuda-devices",
        default="",
        help="Optional physical CUDA devices, e.g. 2,3,4. Each run gets CUDA_VISIBLE_DEVICES=<id> and profile_lora.py sees cuda:0.",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--profile-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=float, default=128.0)
    parser.add_argument("--vocab-rows", type=int, default=4096)
    parser.add_argument("--precision", default="bf16", help="Experiment precision label used in result table filenames.")
    parser.add_argument("--workflow", default="lora_sft", help="Experiment workflow label used in result table filenames.")
    parser.add_argument(
        "--mode",
        default="auto",
        help="Result filename mode label. Default auto uses <backend-label>_<profiler>, e.g. asym-only_nsys.",
    )
    parser.add_argument("--moe-mode", choices=["contiguous", "masked"], default="contiguous")
    parser.add_argument("--timing-mode", choices=["profile", "debug_sync"], default="profile")
    parser.add_argument("--nsys-bin", default="nsys")
    parser.add_argument("--ncu-bin", default="ncu")
    parser.add_argument("--ncu-preset", choices=["quick", "paper"], default="paper")
    parser.add_argument("--ncu-clear-jit-cache", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("profiling"))
    parser.add_argument("--run-name", default="", help="Optional subdirectory under --output-root. Default writes directly into --output-root.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workloads = _expand_workloads(args.workloads)
    backends = _expand_backends(args.backends)
    profilers = _expand_profilers(args.profilers)
    cuda_devices = _cuda_devices(args.cuda_devices)
    run_dir = args.output_root / args.run_name if args.run_name else args.output_root
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    task_index = 0
    for workload in workloads:
        for backend in backends:
            for profiler in profilers:
                physical_cuda_device = cuda_devices[task_index % len(cuda_devices)] if cuda_devices else None
                child_device = "cuda:0" if physical_cuda_device is not None else args.device
                output_dir = _raw_output_dir(run_dir, workload, args, backend, profiler)
                output_dir.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                if physical_cuda_device is not None:
                    env["CUDA_VISIBLE_DEVICES"] = physical_cuda_device

                commands_for_task: list[list[str]]
                skip_reason = ""
                if profiler == "source":
                    commands_for_task = [
                        [
                            sys.executable,
                            str(PROFILE_SCRIPT),
                            *_common_profile_args(args, workload, backend, child_device, output_dir),
                            "--timing-mode",
                            args.timing_mode,
                        ]
                    ]
                elif profiler == "nsys":
                    source_dir = output_dir / "source_debug"
                    report_prefix = output_dir / "trace"
                    sqlite_path = output_dir / "trace.sqlite"
                    commands_for_task = [
                        [
                            args.nsys_bin,
                            "profile",
                            "--trace=cuda,nvtx",
                            "--sample=none",
                            "--cpuctxsw=none",
                            "--resolve-symbols=false",
                            "--wait=primary",
                            "--force-overwrite=true",
                            f"--output={report_prefix}",
                            sys.executable,
                            str(PROFILE_SCRIPT),
                            *_common_profile_args(args, workload, backend, child_device, source_dir),
                            "--timing-mode",
                            "profile",
                        ],
                        [
                            args.nsys_bin,
                            "export",
                            "--type=sqlite",
                            "--force-overwrite=true",
                            f"--output={sqlite_path}",
                            str(output_dir / "trace.nsys-rep"),
                        ],
                        [
                            sys.executable,
                            str(NSYS_POSTPROCESS_SCRIPT),
                            str(sqlite_path),
                            "--output-json",
                            str(output_dir / "profile.json"),
                            "--output-md",
                            str(output_dir / "table.md"),
                        ],
                    ]
                elif profiler == "cpu":
                    commands_for_task = [
                        [
                            sys.executable,
                            str(CPU_GAPS_SCRIPT),
                            *_common_profile_args(args, workload, backend, child_device, output_dir),
                            "--nsys-bin",
                            args.nsys_bin,
                        ]
                    ]
                elif profiler == "ncu":
                    if backend != "asym_only":
                        skip_reason = "ncu only profiles AsymGEMM kernels; torch_only has no matching AsymGEMM kernel"
                        commands_for_task = []
                    elif workload not in NCU_WORKLOADS:
                        skip_reason = f"ncu wrapper supports {sorted(NCU_WORKLOADS)}, not {workload!r}"
                        commands_for_task = []
                    else:
                        ncu_cmd = [
                            sys.executable,
                            str(NCU_SCRIPT),
                            "--workload",
                            workload,
                            "--device",
                            child_device,
                            "--backend",
                            backend,
                            "--warmup-steps",
                            str(args.warmup_steps),
                            "--measure-steps",
                            str(args.measure_steps),
                            "--preset",
                            args.ncu_preset,
                            "--ncu-bin",
                            args.ncu_bin,
                            "--output-dir",
                            str(output_dir),
                            "--moe-mode",
                            args.moe_mode,
                            "--profile-layers",
                            str(args.profile_layers),
                            "--batch-size",
                            str(args.batch_size),
                            "--seq-len",
                            str(args.seq_len),
                            "--tokens",
                            str(args.tokens),
                            "--lora-rank",
                            str(args.lora_rank),
                            "--lora-alpha",
                            str(args.lora_alpha),
                            "--vocab-rows",
                            str(args.vocab_rows),
                        ]
                        if args.ncu_clear_jit_cache:
                            ncu_cmd.append("--clear-jit-cache")
                        commands_for_task = [ncu_cmd]
                else:
                    raise AssertionError(profiler)

                command_record = {
                    "workload": workload,
                    "backend": backend,
                    "profiler": profiler,
                    "device": child_device,
                    "physical_cuda_device": physical_cuda_device,
                    "output_dir": str(output_dir),
                    "commands": commands_for_task,
                    "command": commands_for_task[0] if commands_for_task else [],
                    "skip_reason": skip_reason,
                }
                commands.append(command_record)

                if skip_reason:
                    row = _summary_row(
                        workload=workload,
                        backend=backend,
                        profiler=profiler,
                        device=child_device,
                        physical_cuda_device=physical_cuda_device,
                        output_dir=output_dir,
                        returncode=0,
                        profile={},
                    )
                    row["status"] = "skipped"
                    row["skip_reason"] = skip_reason
                    rows.append(row)
                    task_index += 1
                    continue

                for cmd in commands_for_task:
                    print("Running:", " ".join(cmd), flush=True)
                    if physical_cuda_device is not None:
                        print(f"  CUDA_VISIBLE_DEVICES={physical_cuda_device}", flush=True)
                    if args.dry_run:
                        continue
                    result = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
                    if result.returncode != 0:
                        break
                else:
                    result = subprocess.CompletedProcess(commands_for_task[-1] if commands_for_task else [], 0)

                if args.dry_run:
                    row = _summary_row(
                        workload=workload,
                        backend=backend,
                        profiler=profiler,
                        device=child_device,
                        physical_cuda_device=physical_cuda_device,
                        output_dir=output_dir,
                        returncode=0,
                        profile={},
                    )
                else:
                    profile, profile_path = _load_profile(output_dir)
                    row = _summary_row(
                        workload=workload,
                        backend=backend,
                        profiler=profiler,
                        device=child_device,
                        physical_cuda_device=physical_cuda_device,
                        output_dir=output_dir,
                        returncode=result.returncode,
                        profile=profile,
                    )
                    row["profile_json"] = str(profile_path) if profile_path is not None else None
                    row["result_aliases"] = _write_result_aliases(args, run_dir, row)
                rows.append(row)
                if result.returncode != 0 and not args.continue_on_error:
                    break
                task_index += 1
            if rows and rows[-1]["returncode"] != 0 and not args.continue_on_error:
                break
        if rows and rows[-1]["returncode"] != 0 and not args.continue_on_error:
            break

    summary = {
        "run_dir": str(run_dir),
        "precision": args.precision,
        "workflow": args.workflow,
        "mode": args.mode,
        "backend_semantics": {
            "asym_only": "direct AsymGEMM host-weight path",
            "torch_only": "same host-weight wrapper with PyTorch fallback; not a GPU-resident LLaMA-Factory baseline",
        },
        "workloads": workloads,
        "backends": backends,
        "profilers": profilers,
        "commands": commands,
        "runs": rows,
        "comparisons": _add_comparisons(rows),
    }
    summary_stem = "_".join(part for part in (_safe_label(args.precision), _safe_label(args.workflow), "summary") if part)
    commands_path = run_dir / f"{_safe_label(args.precision)}_{_safe_label(args.workflow)}_commands.json"
    commands_path.write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    (run_dir / f"{summary_stem}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / f"{summary_stem}.md").write_text(_markdown(summary), encoding="utf-8")
    print(f"Wrote {run_dir / f'{summary_stem}.md'}")
    print(f"Wrote {run_dir / f'{summary_stem}.json'}")
    print(f"Wrote {commands_path}")

    failed = [row for row in rows if row["returncode"] != 0]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
