#!/usr/bin/env python3
"""Run LoRA-SFT profiling workloads for AsymGEMM and a GPU-resident Torch baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCRIPT = ROOT / "scripts" / "profile_lora.py"
NSYS_POSTPROCESS_SCRIPT = ROOT / "scripts" / "postprocess_nsys_lora.py"
CPU_GAPS_SCRIPT = ROOT / "scripts" / "profile_nsys_cpu_gaps.py"
NCU_SCRIPT = ROOT / "scripts" / "profile_ncu_asymgemm.py"
DEFAULT_LORA_BATCH_SIZE = 32
DEFAULT_LORA_SEQ_LEN = 64
DEFAULT_LORA_HIDDEN_DIM = 1024
DEFAULT_LORA_MLP_EXPANSION = 4
DEFAULT_DENSE_TARGET_MODE = "all"
DEFAULT_TARGET_MODULES = "all"
DEFAULT_OFFLOAD_MODULES = ""
DENSE_TARGET_MODES = ("mlp_only", "attention_only", "all")
LORA_DTYPE_CHOICES = ("bf16", "bfloat16", "fp16", "float16", "fp32", "float32")
WORKFLOW_LABEL = "lora-sft"

WORKLOADS = (
    "mlp_1b",
    "mlp_3b",
    "mm_1b",
    "mm_3b",
    "dense_3b",
    "dense_14b",
    "moe-604m-a75m",
    "moe-604m-a38m",
    "mlp",
    "dense",
    "moe",
)
WORKLOAD_ALIASES = {
    "toy": ("mlp", "dense", "moe"),
    "custom3b": ("dense_3b", "moe-604m-a75m"),
    "qwen": ("dense_14b", "moe-604m-a38m"),
    "all": WORKLOADS,
}
BACKENDS = ("asym", "torch", "kt")
KT_MOE_WORKLOADS = {"moe", "moe-604m-a75m", "moe-604m-a38m"}
PROFILERS = ("source", "nsys", "cpu", "ncu")
NCU_WORKLOADS = {"mm_1b", "mm_3b", "mlp_1b", "mlp_3b", "dense_3b", "dense_14b", "moe-604m-a75m", "moe-604m-a38m"}


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    profile_layers: int | None = None

    @property
    def label(self) -> str:
        if self.profile_layers is None:
            return self.name
        return f"{self.name}-l{self.profile_layers}"


def _split_tokens(values: Iterable[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        tokens.extend(part.strip() for part in value.split(",") if part.strip())
    return tokens


def _parse_workload_token(value: str) -> tuple[str, int | None]:
    if "|" not in value:
        return value, None
    name, layers_text = value.split("|", 1)
    name = name.strip()
    layers_text = layers_text.strip()
    if not name or not layers_text:
        raise SystemExit(f"invalid workload layer spec {value!r}; expected workload|layers")
    try:
        layers = int(layers_text)
    except ValueError as exc:
        raise SystemExit(f"invalid workload layer spec {value!r}; layers must be an integer") from exc
    if layers <= 0:
        raise SystemExit(f"invalid workload layer spec {value!r}; layers must be positive")
    return name, layers


def _expand_workloads(values: Iterable[str]) -> list[WorkloadSpec]:
    expanded: list[WorkloadSpec] = []
    for value in _split_tokens(values):
        name, layers = _parse_workload_token(value)
        if name in WORKLOAD_ALIASES:
            expanded.extend(WorkloadSpec(workload, layers) for workload in WORKLOAD_ALIASES[name])
        elif name in WORKLOADS:
            expanded.append(WorkloadSpec(name, layers))
        else:
            allowed = ", ".join((*WORKLOADS, *WORKLOAD_ALIASES))
            raise SystemExit(f"unknown workload {name!r}; allowed: {allowed}")
    return list({spec.label: spec for spec in expanded}.values())


def _expand_backends(values: Iterable[str]) -> list[str]:
    backends = _split_tokens(values)
    bad = [backend for backend in backends if backend not in BACKENDS]
    if bad:
        raise SystemExit(f"unknown backend(s) {bad}; allowed: {', '.join(BACKENDS)}")
    return list(dict.fromkeys(backends))


def _expand_profilers(values: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for value in _split_tokens(values):
        if value == "all":
            expanded.extend(PROFILERS)
        elif value in PROFILERS:
            expanded.append(value)
        else:
            allowed = ", ".join((*PROFILERS, "all"))
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


def _seq_lens(args: argparse.Namespace) -> list[int]:
    values = getattr(args, "seq_lens", None)
    if not values:
        return [int(args.seq_len)]
    seq_lens: list[int] = []
    for value in _split_tokens(values):
        try:
            seq_len = int(value)
        except ValueError as exc:
            raise SystemExit(f"invalid --seq-lens value {value!r}; entries must be integers") from exc
        if seq_len <= 0:
            raise SystemExit(f"invalid --seq-lens value {value!r}; entries must be positive")
        seq_lens.append(seq_len)
    return list(dict.fromkeys(seq_lens))


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _preflight_runtime(args: argparse.Namespace) -> None:
    if args.dry_run or args.collect_existing:
        return
    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "error: selected Python interpreter cannot import torch:\n"
            f"  python: {sys.executable}\n"
            f"  {exc.__class__.__name__}: {exc}\n"
            "hint: run the shell wrapper with --python-bin /path/to/python, or invoke this script with a Python that has torch installed."
        )


def _safe_label(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() or ch == "-" else "_" for ch in value).strip("_-")


def _backend_label(backend: str) -> str:
    return backend.replace("_", "-")


def _input_config_label(args: argparse.Namespace) -> str:
    batch = int(getattr(args, "batch_size", 0) or 0)
    seq = int(getattr(args, "seq_len", 0) or 0)
    recompute = "recomp" if bool(getattr(args, "activation_recompute", False)) else "norecomp"
    parts = []
    if batch > 0:
        parts.append(f"b{batch}")
    if seq > 0:
        parts.append(f"s{seq}")
    parts.append(recompute)
    return "_".join(parts)


def _result_stem(args: argparse.Namespace, backend: str, profiler: str) -> str:
    mode = args.mode
    if mode == "auto":
        mode = f"{_backend_label(backend)}_{profiler}"
    return "_".join(
        part
        for part in (
            _safe_label(args.precision),
            _safe_label(WORKFLOW_LABEL),
            _safe_label(_input_config_label(args)),
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


def _common_profile_args(
    args: argparse.Namespace,
    workload: str,
    backend: str,
    device: str,
    output_dir: Path,
    *,
    profile_layers: int,
) -> list[str]:
    profile_args = [
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
        "--dense-target-mode",
        args.dense_target_mode,
        "--target-modules",
        args.target_modules,
        "--offload-modules",
        args.offload_modules,
        "--profile-layers",
        str(profile_layers),
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
        "--precision",
        str(args.precision),
        "--kt-method",
        args.kt_method,
        "--kt-cpu-threads",
        str(args.kt_cpu_threads),
        "--kt-threadpool-count",
        str(args.kt_threadpool_count),
        "--kt-max-cache-depth",
        str(args.kt_max_cache_depth),
        "--output-dir",
        str(output_dir),
    ]
    if bool(getattr(args, "activation_recompute", False)):
        profile_args.append("--activation-recompute")
    return profile_args


def _numeric_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _numeric_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _profile_views(profile: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(profile, dict):
        return []
    views = [profile]
    for key in ("source_profile", "memory_profile"):
        nested = profile.get(key)
        if isinstance(nested, dict):
            views.append(nested)
    return views


def _profile_step_milliseconds(profile: dict[str, Any]) -> float | None:
    stages = profile.get("stages") if isinstance(profile, dict) else None
    if isinstance(stages, list):
        total = 0.0
        found = False
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            value = _numeric_float(stage.get("total_milliseconds"))
            if value is None:
                continue
            total += value
            found = True
        if found:
            return total

    for view in _profile_views(profile):
        step = view.get("step")
        if isinstance(step, dict):
            value = _numeric_float(step.get("total_milliseconds"))
            if value is not None:
                return value
    return None


def _memory_views(profile: dict[str, Any]) -> list[dict[str, Any]]:
    memories: list[dict[str, Any]] = []
    for view in _profile_views(profile):
        memory = view.get("memory")
        if isinstance(memory, dict):
            memories.append(memory)
    return memories


def _profile_memory_value(profile: dict[str, Any], top_key: str, section_key: str, nested_key: str) -> int | None:
    for memory in _memory_views(profile):
        top_value = _numeric_int(memory.get(top_key))
        if top_value is not None:
            return top_value
        section = memory.get(section_key)
        if isinstance(section, dict):
            nested_value = _numeric_int(section.get(nested_key))
            if nested_value is not None:
                return nested_value
    return None


def _profile_expected_hbm_saved_bytes(profile: dict[str, Any]) -> int | None:
    direct = _profile_memory_value(profile, "expected_hbm_saved_bytes", "gpu", "expected_hbm_saved_bytes")
    if direct is not None:
        return direct
    for memory in _memory_views(profile):
        cpu_memory = memory.get("cpu")
        if not isinstance(cpu_memory, dict):
            continue
        host_w_bytes = _numeric_int(cpu_memory.get("host_w_bytes"))
        if host_w_bytes is not None:
            return host_w_bytes
    return None


def _hbm_saved_percent(peak_hbm_bytes: Any, hbm_saved_bytes: Any) -> float | None:
    peak = _numeric_float(peak_hbm_bytes)
    saved = _numeric_float(hbm_saved_bytes)
    if peak is None or saved is None:
        return None
    denominator = peak + saved
    if denominator <= 0.0:
        return None
    return saved * 100.0 / denominator


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
    peak_hbm_bytes = _profile_memory_value(profile, "peak_hbm_bytes", "gpu", "peak_hbm_bytes")
    expected_hbm_saved_bytes = _profile_expected_hbm_saved_bytes(profile)
    return {
        "workload": workload,
        "backend": backend,
        "profiler": profiler,
        "device": device,
        "physical_cuda_device": physical_cuda_device,
        "status": "ok" if returncode == 0 else "failed",
        "returncode": returncode,
        "step_ms": _profile_step_milliseconds(profile),
        "peak_hbm_bytes": peak_hbm_bytes,
        "expected_hbm_saved_bytes": expected_hbm_saved_bytes,
        "hbm_saved_percent": _hbm_saved_percent(peak_hbm_bytes, expected_hbm_saved_bytes),
        "pinned_cpu_bytes": _profile_memory_value(profile, "pinned_cpu_bytes", "cpu", "pinned_total_bytes"),
        "output_dir": str(output_dir),
    }


def _add_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_workload: dict[tuple[str, str, int | None, bool], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        seq_len = row.get("seq_len")
        key = (
            str(row["workload"]),
            str(row.get("profiler", "source")),
            int(seq_len) if isinstance(seq_len, int) else None,
            bool(row.get("activation_recompute", False)),
        )
        by_workload.setdefault(key, {})[str(row["backend"])] = row
    comparisons: list[dict[str, Any]] = []
    for (workload, profiler, seq_len, activation_recompute), items in by_workload.items():
        asym = items.get("asym")
        torch = items.get("torch")
        kt = items.get("kt")
        if asym and torch:
            asym_ms = asym.get("step_ms")
            torch_ms = torch.get("step_ms")
            if isinstance(asym_ms, (int, float)) and isinstance(torch_ms, (int, float)) and asym_ms > 0:
                comparisons.append(
                    {
                        "workload": workload,
                        "profiler": profiler,
                        "seq_len": seq_len,
                        "activation_recompute": activation_recompute,
                        "comparison": "asym_vs_torch",
                        "candidate_step_ms": asym_ms,
                        "torch_step_ms": torch_ms,
                        "torch_over_candidate_speedup": torch_ms / asym_ms,
                        "candidate_minus_torch_ms": asym_ms - torch_ms,
                    }
                )
        if kt and torch:
            kt_ms = kt.get("step_ms")
            torch_ms = torch.get("step_ms")
            if isinstance(kt_ms, (int, float)) and isinstance(torch_ms, (int, float)) and kt_ms > 0:
                comparisons.append(
                    {
                        "workload": workload,
                        "profiler": profiler,
                        "seq_len": seq_len,
                        "activation_recompute": activation_recompute,
                        "comparison": "kt_vs_torch",
                        "candidate_step_ms": kt_ms,
                        "torch_step_ms": torch_ms,
                        "torch_over_candidate_speedup": torch_ms / kt_ms,
                        "candidate_minus_torch_ms": kt_ms - torch_ms,
                    }
                )
    return comparisons


def _fmt_table_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, int):
        return str(value)
    if value is None:
        return "-"
    return str(value)


def _fmt_bytes(value: Any) -> str:
    numeric = _numeric_float(value)
    if numeric is None:
        return "-"
    abs_value = abs(numeric)
    if abs_value >= 1024.0**3:
        return f"{numeric / (1024.0**3):.2f} GiB"
    return f"{numeric / (1024.0**2):.2f} MiB"


def _fmt_percent(value: Any) -> str:
    numeric = _numeric_float(value)
    return "-" if numeric is None else f"{numeric:.2f}%"


def _row_hbm_saved_percent(row: dict[str, Any]) -> float | None:
    direct = _numeric_float(row.get("hbm_saved_percent"))
    if direct is not None:
        return direct
    return _hbm_saved_percent(row.get("peak_hbm_bytes"), row.get("expected_hbm_saved_bytes"))


def _rank_sort_key(row: dict[str, Any], key: str) -> tuple[bool, float]:
    value = _numeric_float(row.get(key))
    return (value is not None, value if value is not None else float("-inf"))


def _result_table_link(row: dict[str, Any]) -> str:
    output_dir = Path(str(row.get("output_dir", "")))
    table_md = _find_markdown(output_dir)
    return f"`{table_md}`" if table_md is not None else "-"


def _ranked_rows(summary: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows = list(summary.get("runs", []))
    return sorted(rows, key=lambda row: _rank_sort_key(row, key), reverse=True)


def _latency_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LoRA-SFT Latency Rankings",
        "",
        f"Precision: `{summary.get('precision', '-')}`  ",
        f"Workflow: `{summary.get('workflow', '-')}`",
        "",
        "Sorted by `Step ms` from largest to smallest.",
        "",
        "| Rank | Workload | Backend | Profiler | Status | Device | Step ms | Peak HBM | Summary | Output |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for index, row in enumerate(_ranked_rows(summary, "step_ms"), start=1):
        device = row["device"]
        if row.get("physical_cuda_device") is not None:
            device = f"CUDA_VISIBLE_DEVICES={row['physical_cuda_device']}:{device}"
        lines.append(
            "| {rank} | {workload} | {backend} | {profiler} | {status} | {device} | {step_ms} | {peak_hbm} | {result_table} | `{output_dir}` |".format(
                rank=index,
                workload=row["workload"],
                backend=row["backend"],
                profiler=row.get("profiler", "source"),
                status=row["status"],
                device=device,
                step_ms=_fmt_table_value(row.get("step_ms")),
                peak_hbm=_fmt_bytes(row.get("peak_hbm_bytes")),
                result_table=_result_table_link(row),
                output_dir=row["output_dir"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _memory_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LoRA-SFT Memory Rankings",
        "",
        f"Precision: `{summary.get('precision', '-')}`  ",
        f"Workflow: `{summary.get('workflow', '-')}`",
        "",
        "Sorted by `Peak HBM` from largest to smallest.",
        "",
        "| Rank | Workload | Backend | Profiler | Status | Device | Peak HBM | GPU HBM saved | GPU HBM saved % | Pinned CPU | Step ms | Summary | Output |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for index, row in enumerate(_ranked_rows(summary, "peak_hbm_bytes"), start=1):
        device = row["device"]
        if row.get("physical_cuda_device") is not None:
            device = f"CUDA_VISIBLE_DEVICES={row['physical_cuda_device']}:{device}"
        lines.append(
            "| {rank} | {workload} | {backend} | {profiler} | {status} | {device} | {peak_hbm} | {hbm_saved} | {hbm_saved_percent} | {pinned_cpu} | {step_ms} | {result_table} | `{output_dir}` |".format(
                rank=index,
                workload=row["workload"],
                backend=row["backend"],
                profiler=row.get("profiler", "source"),
                status=row["status"],
                device=device,
                peak_hbm=_fmt_bytes(row.get("peak_hbm_bytes")),
                hbm_saved=_fmt_bytes(row.get("expected_hbm_saved_bytes")),
                hbm_saved_percent=_fmt_percent(_row_hbm_saved_percent(row)),
                pinned_cpu=_fmt_bytes(row.get("pinned_cpu_bytes")),
                step_ms=_fmt_table_value(row.get("step_ms")),
                result_table=_result_table_link(row),
                output_dir=row["output_dir"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LoRA-SFT Profiling Workflow",
        "",
        f"Precision: `{summary.get('precision', '-')}`  ",
        f"Workflow: `{summary.get('workflow', '-')}`",
        "",
        "`asym` measures the direct AsymGEMM host-weight path. `torch` measures",
        "a normal GPU-resident PyTorch LoRA baseline with frozen base weights stored as",
        "CUDA buffers. `kt` measures the KTransformers AMX SFT MoE path for MoE workloads.",
        "",
        "## Runs",
        "",
        "| Workload | Backend | Profiler | Status | Device | Step ms | Peak HBM | GPU HBM saved | GPU HBM saved % | Pinned CPU | Summary | Output |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary["runs"]:
        device = row["device"]
        if row.get("physical_cuda_device") is not None:
            device = f"CUDA_VISIBLE_DEVICES={row['physical_cuda_device']}:{device}"
        lines.append(
            "| {workload} | {backend} | {profiler} | {status} | {device} | {step_ms} | {peak_hbm_bytes} | "
            "{expected_hbm_saved_bytes} | {hbm_saved_percent} | {pinned_cpu_bytes} | {result_table} | `{output_dir}` |".format(
                workload=row["workload"],
                backend=row["backend"],
                profiler=row.get("profiler", "source"),
                status=row["status"],
                device=device,
                step_ms=_fmt_table_value(row.get("step_ms")),
                peak_hbm_bytes=_fmt_bytes(row.get("peak_hbm_bytes")),
                expected_hbm_saved_bytes=_fmt_bytes(row.get("expected_hbm_saved_bytes")),
                hbm_saved_percent=_fmt_percent(_row_hbm_saved_percent(row)),
                pinned_cpu_bytes=_fmt_bytes(row.get("pinned_cpu_bytes")),
                result_table=_result_table_link(row),
                output_dir=row["output_dir"],
            )
        )
    lines += ["", "## Comparisons", ""]
    if summary["comparisons"]:
        lines += [
            "| Workload | Profiler | Comparison | Candidate step ms | Torch step ms | Torch/Candidate speedup | Candidate minus Torch ms |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
        for row in summary["comparisons"]:
            lines.append(
                "| {workload} | {profiler} | {comparison} | {candidate_step_ms:.3f} | {torch_step_ms:.3f} | "
                "{torch_over_candidate_speedup:.3f} | {candidate_minus_torch_ms:.3f} |".format(**row)
            )
    else:
        lines.append("No paired comparisons against `torch` were available.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["qwen"],
        help=(
            f"Workload list or aliases. Use workload|layers for per-workload depth, e.g. moe-604m-a75m|2. "
            f"Workloads: {', '.join(WORKLOADS)}. Aliases: {', '.join(WORKLOAD_ALIASES)}."
        ),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["asym", "torch", "kt"],
        help="Backend list. Allowed values: asym, torch, kt.",
    )
    parser.add_argument(
        "--profilers",
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
    parser.add_argument("--batch-size", type=int, default=DEFAULT_LORA_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_LORA_SEQ_LEN)
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        default=None,
        help="Sequence length sweep. Accepts space- or comma-separated values and writes one standard result directory per length.",
    )
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_LORA_HIDDEN_DIM)
    parser.add_argument("--mlp-intermediate-dim", type=int, default=0)
    parser.add_argument("--mlp-expansion", type=int, default=DEFAULT_LORA_MLP_EXPANSION)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=float, default=128.0)
    parser.add_argument("--lora-dtype", choices=LORA_DTYPE_CHOICES, default="bf16")
    parser.add_argument("--vocab-rows", type=int, default=4096)
    parser.add_argument("--precision", default="bf16", help="Experiment precision label used in result table filenames.")
    parser.add_argument(
        "--mode",
        default="auto",
        help="Result filename mode label. Default auto uses <backend-label>_<profiler>, e.g. asym_nsys.",
    )
    parser.add_argument("--moe-mode", choices=["contiguous", "masked"], default="contiguous")
    parser.add_argument("--kt-method", default="AMXBF16_SFT", choices=["AMXBF16_SFT", "AMXINT8_SFT", "AMXINT4_SFT"])
    parser.add_argument("--kt-cpu-threads", type=int, default=1)
    parser.add_argument("--kt-threadpool-count", type=int, default=1)
    parser.add_argument("--kt-max-cache-depth", type=int, default=1)
    parser.add_argument(
        "--target-preset",
        "--dense-target-mode",
        dest="dense_target_mode",
        choices=DENSE_TARGET_MODES,
        default=DEFAULT_DENSE_TARGET_MODE,
        help="Dense workload target scope. Default adapts all known target projections.",
    )
    parser.add_argument("--target-modules", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--offload-modules", default=DEFAULT_OFFLOAD_MODULES)
    parser.add_argument(
        "--activation-recompute",
        action="store_true",
        help="Pass --activation-recompute to profile_lora.py for layer-level activation checkpointing.",
    )
    parser.add_argument("--nsys-bin", default="nsys")
    parser.add_argument("--ncu-bin", default="ncu")
    parser.add_argument("--ncu-preset", choices=["quick", "paper"], default="paper")
    parser.add_argument("--ncu-clear-jit-cache", action="store_true")
    parser.add_argument(
        "--skip-memory-attribution",
        action="store_true",
        help="For nsys runs, skip the separate source-only saved-activation memory attribution pass.",
    )
    parser.add_argument(
        "--memory-attribution-steps",
        type=int,
        default=1,
        help="Measured steps for the separate memory-only attribution pass. One step is enough for fixed-shape profiles.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("profiling"))
    parser.add_argument("--run-name", default="", help="Optional subdirectory under --output-root. Default writes directly into --output-root.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-existing", action="store_true", help="Build summaries from existing output directories without running profilers.")
    parser.add_argument("--skip-summary", action="store_true", help="Do not write run-level summary/commands files. Useful for parallel shell orchestration.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _preflight_runtime(args)
    workload_specs = _expand_workloads(args.workloads)
    backends = _expand_backends(args.backends)
    profilers = _expand_profilers(args.profilers)
    seq_lens = _seq_lens(args)
    cuda_devices = _cuda_devices(args.cuda_devices)
    run_dir = args.output_root / args.run_name if args.run_name else args.output_root
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    task_index = 0
    stop_requested = False
    for workload_spec in workload_specs:
        workload = workload_spec.name
        workload_label = workload_spec.label
        profile_layers = int(workload_spec.profile_layers or args.profile_layers)
        for seq_len in seq_lens:
            args.seq_len = seq_len
            for backend in backends:
                for profiler in profilers:
                    physical_cuda_device = cuda_devices[task_index % len(cuda_devices)] if cuda_devices else None
                    child_device = "cuda:0" if physical_cuda_device is not None else args.device
                    output_dir = _raw_output_dir(run_dir, workload_label, args, backend, profiler)
                    env = os.environ.copy()
                    env["PYTHONUNBUFFERED"] = "1"
                    if physical_cuda_device is not None:
                        env["CUDA_VISIBLE_DEVICES"] = physical_cuda_device

                    commands_for_task: list[list[str]]
                    skip_reason = ""
                    if backend == "kt" and workload not in KT_MOE_WORKLOADS:
                        skip_reason = "backend=kt is only implemented for MoE LoRA SFT workloads"
                        commands_for_task = []
                    elif profiler == "source":
                        commands_for_task = [
                            [
                                sys.executable,
                                str(PROFILE_SCRIPT),
                                *_common_profile_args(args, workload, backend, child_device, output_dir, profile_layers=profile_layers),
                            ]
                        ]
                    elif profiler == "nsys":
                        source_dir = output_dir / "source_debug"
                        memory_dir = output_dir / "memory_debug"
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
                                *_common_profile_args(args, workload, backend, child_device, source_dir, profile_layers=profile_layers),
                            ],
                            [
                                args.nsys_bin,
                                "export",
                                "--type=sqlite",
                                "--force-overwrite=true",
                                f"--output={sqlite_path}",
                                str(output_dir / "trace.nsys-rep"),
                            ],
                        ]
                        if not args.skip_memory_attribution:
                            commands_for_task.append(
                                [
                                    sys.executable,
                                    str(PROFILE_SCRIPT),
                                    *_common_profile_args(args, workload, backend, child_device, memory_dir, profile_layers=profile_layers),
                                    "--warmup-steps",
                                    "0",
                                    "--measure-steps",
                                    str(max(1, int(args.memory_attribution_steps))),
                                    "--memory-attribution",
                                ]
                            )
                        postprocess_cmd = [
                            sys.executable,
                            str(NSYS_POSTPROCESS_SCRIPT),
                            str(sqlite_path),
                            "--source-profile-dir",
                            str(source_dir),
                        ]
                        if not args.skip_memory_attribution:
                            postprocess_cmd.extend(["--memory-profile-dir", str(memory_dir)])
                        postprocess_cmd.extend(
                            [
                                "--output-json",
                                str(output_dir / "profile.json"),
                                "--output-md",
                                str(output_dir / "table.md"),
                            ]
                        )
                        commands_for_task.append(postprocess_cmd)
                    elif profiler == "cpu":
                        commands_for_task = [
                            [
                                sys.executable,
                                str(CPU_GAPS_SCRIPT),
                                *_common_profile_args(args, workload, backend, child_device, output_dir, profile_layers=profile_layers),
                                "--nsys-bin",
                                args.nsys_bin,
                            ]
                        ]
                    elif profiler == "ncu":
                        if backend != "asym":
                            skip_reason = f"ncu only profiles AsymGEMM kernels; backend={backend} has no matching AsymGEMM kernel"
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
                                "--dense-target-mode",
                                args.dense_target_mode,
                                "--target-modules",
                                args.target_modules,
                                "--offload-modules",
                                args.offload_modules,
                                "--profile-layers",
                                str(profile_layers),
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
                            ]
                            if args.ncu_clear_jit_cache:
                                ncu_cmd.append("--clear-jit-cache")
                            commands_for_task = [ncu_cmd]
                    else:
                        raise AssertionError(profiler)

                    command_record = {
                        "workload": workload_label,
                        "base_workload": workload,
                        "profile_layers": profile_layers,
                        "seq_len": seq_len,
                        "activation_recompute": bool(args.activation_recompute),
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
                            workload=workload_label,
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
                        row["base_workload"] = workload
                        row["profile_layers"] = profile_layers
                        row["seq_len"] = seq_len
                        row["activation_recompute"] = bool(args.activation_recompute)
                        rows.append(row)
                        task_index += 1
                        continue

                    if not args.dry_run and not args.collect_existing:
                        output_dir.mkdir(parents=True, exist_ok=True)

                    if args.collect_existing:
                        profile, profile_path = _load_profile(output_dir)
                        returncode = 0 if profile_path is not None else 1
                        row = _summary_row(
                            workload=workload_label,
                            backend=backend,
                            profiler=profiler,
                            device=child_device,
                            physical_cuda_device=physical_cuda_device,
                            output_dir=output_dir,
                            returncode=returncode,
                            profile=profile,
                        )
                        row["profile_json"] = str(profile_path) if profile_path is not None else None
                        row["base_workload"] = workload
                        row["profile_layers"] = profile_layers
                        row["seq_len"] = seq_len
                        row["activation_recompute"] = bool(args.activation_recompute)
                        rows.append(row)
                        task_index += 1
                        if returncode != 0 and not args.continue_on_error:
                            stop_requested = True
                            break
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
                            workload=workload_label,
                            backend=backend,
                            profiler=profiler,
                            device=child_device,
                            physical_cuda_device=physical_cuda_device,
                            output_dir=output_dir,
                            returncode=0,
                            profile={},
                        )
                        row["base_workload"] = workload
                        row["profile_layers"] = profile_layers
                    else:
                        profile, profile_path = _load_profile(output_dir)
                        row = _summary_row(
                            workload=workload_label,
                            backend=backend,
                            profiler=profiler,
                            device=child_device,
                            physical_cuda_device=physical_cuda_device,
                            output_dir=output_dir,
                            returncode=result.returncode,
                            profile=profile,
                        )
                        row["profile_json"] = str(profile_path) if profile_path is not None else None
                        row["base_workload"] = workload
                        row["profile_layers"] = profile_layers
                    row["seq_len"] = seq_len
                    row["activation_recompute"] = bool(args.activation_recompute)
                    rows.append(row)
                    task_index += 1
                    if result.returncode != 0 and not args.continue_on_error:
                        stop_requested = True
                        break
                if stop_requested:
                    break
            if stop_requested:
                break
        if stop_requested:
            break

    summary = {
        "run_dir": str(run_dir),
        "precision": args.precision,
        "workflow": WORKFLOW_LABEL,
        "mode": args.mode,
        "backend_semantics": {
            "asym": "direct AsymGEMM host-weight path",
            "torch": "normal GPU-resident PyTorch LoRA baseline with frozen base weights stored as CUDA buffers",
            "kt": "KTransformers AMX SFT MoE path for MoE workloads",
        },
        "workloads": [spec.label for spec in workload_specs],
        "backends": backends,
        "profilers": profilers,
        "seq_lens": seq_lens,
        "commands": commands,
        "runs": rows,
        "comparisons": _add_comparisons(rows),
    }
    if args.skip_summary:
        failed = [row for row in rows if row["returncode"] != 0]
        if failed:
            raise SystemExit(1)
        return

    summary_stem = "_".join(part for part in (_safe_label(args.precision), _safe_label(WORKFLOW_LABEL), "summary") if part)
    commands_path = run_dir / f"{_safe_label(args.precision)}_{_safe_label(WORKFLOW_LABEL)}_commands.json"
    latency_path = run_dir / "lat.md"
    memory_path = run_dir / "memory.md"
    commands_path.write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    (run_dir / f"{summary_stem}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / f"{summary_stem}.md").write_text(_markdown(summary), encoding="utf-8")
    latency_path.write_text(_latency_markdown(summary), encoding="utf-8")
    memory_path.write_text(_memory_markdown(summary), encoding="utf-8")
    print(f"Wrote {run_dir / f'{summary_stem}.md'}")
    print(f"Wrote {run_dir / f'{summary_stem}.json'}")
    print(f"Wrote {latency_path}")
    print(f"Wrote {memory_path}")
    print(f"Wrote {commands_path}")

    failed = [row for row in rows if row["returncode"] != 0]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
