#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Throughput (tokens/sec) is derived per run from step_samples.csv: tokens_per_step / step_seconds,
# computed over post-warmup (measured) steps only. Mirrors the c2c/interconnect combined-plot layout.
THROUGHPUT_COLOR = "#2ca02c"
THROUGHPUT_EFF_COLOR = "#1f77b4"
_META_FIELDS = [
    "workload",
    "backend",
    "router_mode",
    "asymm_expert_act_offload",
    "expact",
    "asymm_attn_act_offload",
    "attnact",
    "asymm_layer_act_offload",
    "layeract",
    "asymm_layer_gc",
    "layergc",
    "liger_loss",
    "profiler",
    "recompute",
    "expert_policy",
    "seq_len",
    "precision",
    "batch_size",
    "gradient_accumulation_steps",
    "lora_dropout",
    "lora_rank",
    "lora_alpha",
    "lora_target",
    "config",
]
SUMMARY_FIELDS = _META_FIELDS + [
    "run_label",
    "tokens_per_step",
    "measured_steps",
    "effective_tokens_per_second",
    "mean_tokens_per_second",
    "median_tokens_per_second",
    "min_tokens_per_second",
    "max_tokens_per_second",
    "measured_elapsed_seconds",
    "profile_json",
    "run_dir",
]
STEP_FIELDS = _META_FIELDS + [
    "run_label",
    "step",
    "tokens_per_second",
    "step_milliseconds",
    "profile_json",
    "run_dir",
]
INDEX_FIELDS = _META_FIELDS + [
    "run_label",
    "profile_json",
    "run_dir",
]
RUN_DIR_RE = re.compile(r"^b(?P<batch_size>[0-9]+)_s(?P<seq_len>[0-9]+)_ga(?P<grad_accum>[0-9]+)$")


@dataclass(frozen=True)
class RunRecord:
    run_dir: Path
    profile_path: Path
    metadata: dict[str, str]
    tokens_per_step: int
    by_step: dict[int, float]  # measured step -> tokens/sec
    measured_elapsed_seconds: float

    @property
    def label(self) -> str:
        parts = [
            self.metadata.get("workload", ""),
            self.metadata.get("backend", ""),
            f"router={self.metadata.get('router_mode', '')}" if self.metadata.get("router_mode") else "",
            self.metadata.get("expact", ""),
            self.metadata.get("attnact", ""),
            self.metadata.get("layeract", ""),
            self.metadata.get("layergc", ""),
            self.metadata.get("liger_loss", ""),
            self.metadata.get("recompute", ""),
            self.metadata.get("expert_policy", ""),
            f"s{self.metadata.get('seq_len', '')}" if self.metadata.get("seq_len") else "",
        ]
        return " ".join(part for part in parts if part)

    def effective_tokens_per_second(self) -> float:
        if self.measured_elapsed_seconds <= 0.0 or not self.by_step:
            return 0.0
        return (len(self.by_step) * self.tokens_per_step) / self.measured_elapsed_seconds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot combined LF training throughput (tokens/sec) artifacts.")
    parser.add_argument("--input-root", type=Path, action="append", default=[], help="Root to scan for profile.json files.")
    parser.add_argument("--run-dir", type=Path, action="append", default=[], help="Explicit run directory containing profile.json.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for combined throughput plots.")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--combined-only", action="store_true", help="Accepted for parity with other LF plotting scripts.")
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--backend", action="append", default=[])
    parser.add_argument("--profiler", action="append", default=[])
    parser.add_argument("--router-mode", action="append", default=[])
    parser.add_argument("--expact", action="append", default=[])
    parser.add_argument("--attnact", action="append", default=[])
    parser.add_argument("--layeract", action="append", default=[])
    parser.add_argument("--layergc", action="append", default=[])
    parser.add_argument("--liger-loss", action="append", default=[])
    parser.add_argument("--recompute", action="append", default=[])
    parser.add_argument("--workloads", nargs="+", default=[])
    parser.add_argument("--expert-recompute-policies", nargs="+", default=[])
    return parser.parse_args()


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_bool_config(value: Any, default: str = "false") -> str:
    text = str(value if value is not None else default).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return "true"
    if text in {"0", "false", "no", "n", "off"}:
        return "false"
    return default


def _expact_label(value: Any) -> str:
    return "expact1" if _normalize_bool_config(value) == "true" else "expact0"


def _attnact_label(value: Any) -> str:
    return "attnact1" if _normalize_bool_config(value) == "true" else "attnact0"


def _layeract_label(value: Any) -> str:
    return "layeract1" if _normalize_bool_config(value) == "true" else "layeract0"


def _layergc_label(value: Any) -> str:
    return "layergc1" if _normalize_bool_config(value) == "true" else "layergc0"


def _parse_expact_part(part: str) -> tuple[str, str] | None:
    value = part.strip().lower()
    if value in {"expact1", "expacttrue"}:
        return "true", "expact1"
    if value in {"expact0", "expactfalse"}:
        return "false", "expact0"
    return None


def _parse_attnact_part(part: str) -> tuple[str, str] | None:
    value = part.strip().lower()
    if value in {"attnact1", "attnacttrue"}:
        return "true", "attnact1"
    if value in {"attnact0", "attnactfalse"}:
        return "false", "attnact0"
    return None


def _parse_layeract_part(part: str) -> tuple[str, str] | None:
    value = part.strip().lower()
    if value in {"layeract1", "layeracttrue"}:
        return "true", "layeract1"
    if value in {"layeract0", "layeractfalse"}:
        return "false", "layeract0"
    return None


def _parse_layergc_part(part: str) -> tuple[str, str] | None:
    value = part.strip().lower()
    if value in {"layergc1", "layergctrue"}:
        return "true", "layergc1"
    if value in {"layergc0", "layergcfalse"}:
        return "false", "layergc0"
    return None


def _parse_liger_loss_part(part: str) -> str | None:
    value = part.strip().lower()
    if value in {"ligerloss0", "ligerloss1"}:
        return value
    return None


def _parse_job_dir_parts(job_dir_name: str) -> dict[str, str] | None:
    parts = job_dir_name.split("__")
    if len(parts) < 4:
        return None
    backend_part, profiler_part, recompute_part, policy_part = parts[:4]
    tail = parts[4:]
    router_part = "routerhf"
    expact_value, expact = "false", "expact0"
    attnact_value, attnact = "false", "attnact0"
    layeract_value, layeract = "false", "layeract0"
    layergc_value, layergc = "false", "layergc0"
    liger_loss = "ligerloss0"

    if tail:
        router_part = tail.pop(0)
        if not router_part.startswith("router"):
            return None

    for part in tail:
        parsed_expact = _parse_expact_part(part)
        if parsed_expact is not None:
            expact_value, expact = parsed_expact
            continue
        parsed_attnact = _parse_attnact_part(part)
        if parsed_attnact is not None:
            attnact_value, attnact = parsed_attnact
            continue
        parsed_layeract = _parse_layeract_part(part)
        if parsed_layeract is not None:
            layeract_value, layeract = parsed_layeract
            continue
        parsed_layergc = _parse_layergc_part(part)
        if parsed_layergc is not None:
            layergc_value, layergc = parsed_layergc
            continue
        parsed_liger_loss = _parse_liger_loss_part(part)
        if parsed_liger_loss is not None:
            liger_loss = parsed_liger_loss
            continue
        continue

    return {
        "backend": backend_part,
        "profiler": profiler_part,
        "recompute": recompute_part,
        "policy_part": policy_part,
        "router_part": router_part,
        "asymm_expert_act_offload": expact_value,
        "expact": expact,
        "asymm_attn_act_offload": attnact_value,
        "attnact": attnact,
        "asymm_layer_act_offload": layeract_value,
        "layeract": layeract,
        "asymm_layer_gc": layergc_value,
        "layergc": layergc,
        "liger_loss": liger_loss,
    }


def _filter_values(values: list[str]) -> set[str]:
    result: set[str] = set()
    for value in values:
        for part in str(value).replace(",", " ").split():
            if part:
                result.add(part.lower())
    return result


def _apply_workload_filters(args: argparse.Namespace) -> None:
    workload_tuples: set[tuple[str, str, str]] = set()
    for value in args.workloads:
        for workload in str(value).replace(",", " ").split():
            fields = workload.split("|")
            if len(fields) != 3 or not all(field.isdigit() and int(field) > 0 for field in fields):
                raise SystemExit(
                    "--workloads items must be seq_len|per_device_batch_size|gradient_accumulation_steps, "
                    f"got {workload!r}"
                )
            workload_tuples.add((fields[0], fields[1], fields[2]))
    args.workload_tuples = workload_tuples


def _seq_len_from_run_dir_name(name: str) -> str:
    match = RUN_DIR_RE.match(name)
    return match.group("seq_len") if match is not None else ""


def _find_profile_paths(input_roots: list[Path], run_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for run_dir in run_dirs:
        profile_path = run_dir / "profile.json"
        if profile_path.exists():
            paths.append(profile_path)
    for root in input_roots:
        if root.is_file() and root.name == "profile.json":
            paths.append(root)
            continue
        if not root.exists():
            continue
        paths.extend(sorted(root.rglob("profile.json")))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _source_config(run_dir: Path, profile: dict[str, Any]) -> dict[str, Any]:
    source_profile = profile.get("source_profile")
    if not isinstance(source_profile, dict) or not source_profile:
        source_profile = _safe_read_json(run_dir / "source_profile.json")
    config = source_profile.get("config") if isinstance(source_profile, dict) else {}
    return config if isinstance(config, dict) else {}


def _infer_metadata(profile_path: Path, profile: dict[str, Any]) -> dict[str, str] | None:
    run_dir = profile_path.parent
    job_root = run_dir.parent
    config_root = job_root.parent
    config = _source_config(run_dir, profile)
    run_dir_match = RUN_DIR_RE.match(run_dir.name)
    job_meta = _parse_job_dir_parts(job_root.name)
    if job_meta is None:
        return None
    backend_part = job_meta["backend"]
    profiler_part = job_meta["profiler"]
    recompute_part = job_meta["recompute"]
    policy_part = job_meta["policy_part"]
    router_part = job_meta["router_part"]
    expact_value = job_meta["asymm_expert_act_offload"]
    expact = job_meta["expact"]
    attnact_value = job_meta["asymm_attn_act_offload"]
    attnact = job_meta["attnact"]
    layeract_value = job_meta["asymm_layer_act_offload"]
    layeract = job_meta["layeract"]
    layergc_value = job_meta["asymm_layer_gc"]
    layergc = job_meta["layergc"]
    if not policy_part.startswith("pol") or not router_part.startswith("router"):
        return None
    if recompute_part not in {"norecomp", "recomp", "unsloth"}:
        return None

    expert_policy = str(config.get("expert_policy") or policy_part[len("pol") :] or "none")
    router_mode = str(config.get("router_mode") or router_part[len("router") :])

    seq_len = str(
        config.get("seq_len")
        or config.get("cutoff_len")
        or (run_dir_match.group("seq_len") if run_dir_match else "")
    )
    if not seq_len:
        seq_len = _seq_len_from_run_dir_name(run_dir.name)
    expact_value = _normalize_bool_config(config.get("asymm_expert_act_offload", expact_value))
    expact = _expact_label(expact_value)
    attnact_value = _normalize_bool_config(config.get("asymm_attn_act_offload", attnact_value))
    attnact = _attnact_label(attnact_value)
    layeract_value = _normalize_bool_config(config.get("asymm_layer_act_offload", layeract_value))
    layeract = _layeract_label(layeract_value)
    layergc_config_value = config.get("asymm_layer_gc", config.get("asym_layer_glue_gc_enabled", layergc_value))
    layergc_value = _normalize_bool_config(layergc_config_value)
    layergc = _layergc_label(layergc_value)
    liger_loss = str(config.get("liger_loss") or job_meta.get("liger_loss") or "ligerloss0").lower()
    if liger_loss not in {"ligerloss0", "ligerloss1"}:
        liger_loss = "ligerloss0"

    metadata = {
        "workload": str(config.get("workload") or config_root.name.split("__", 1)[0]),
        "backend": str(config.get("backend") or backend_part),
        "router_mode": router_mode,
        "asymm_expert_act_offload": expact_value,
        "expact": expact,
        "asymm_attn_act_offload": attnact_value,
        "attnact": attnact,
        "asymm_layer_act_offload": layeract_value,
        "layeract": layeract,
        "asymm_layer_gc": layergc_value,
        "layergc": layergc,
        "liger_loss": liger_loss,
        "profiler": str(profiler_part),
        "recompute": recompute_part,
        "expert_policy": expert_policy,
        "seq_len": seq_len,
        "precision": str(config.get("precision") or ""),
        "batch_size": str(config.get("batch_size") or (run_dir_match.group("batch_size") if run_dir_match else "")),
        "gradient_accumulation_steps": str(
            config.get("gradient_accumulation_steps") or (run_dir_match.group("grad_accum") if run_dir_match else "")
        ),
        "lora_dropout": str(config.get("lora_dropout") if config.get("lora_dropout") is not None else ""),
        "lora_rank": str(config.get("lora_rank") if config.get("lora_rank") is not None else ""),
        "lora_alpha": str(config.get("lora_alpha") if config.get("lora_alpha") is not None else ""),
        "lora_target": str(config.get("lora_target") or ""),
        "config": config_root.name,
    }
    return metadata


def _matches_filters(record: RunRecord, args: argparse.Namespace) -> bool:
    if getattr(args, "workload_tuples", set()):
        workload_tuple = (
            record.metadata.get("seq_len", ""),
            record.metadata.get("batch_size", ""),
            record.metadata.get("gradient_accumulation_steps", ""),
        )
        if workload_tuple not in args.workload_tuples:
            return False
    filters = {
        "workload": _filter_values(args.workload),
        "backend": _filter_values(args.backend),
        "profiler": _filter_values(args.profiler),
        "router_mode": _filter_values(args.router_mode),
        "expact": _filter_values(getattr(args, "expact", [])),
        "attnact": _filter_values(getattr(args, "attnact", [])),
        "layeract": _filter_values(getattr(args, "layeract", [])),
        "layergc": _filter_values(getattr(args, "layergc", [])),
        "liger_loss": _filter_values(getattr(args, "liger_loss", [])),
        "recompute": _filter_values(args.recompute),
        "expert_policy": _filter_values(args.expert_recompute_policies),
    }
    for key, allowed in filters.items():
        if not allowed:
            continue
        value = record.metadata.get(key, "").lower()
        if key == "workload":
            config_workload = record.metadata.get("config", "").split("__", 1)[0].lower()
            if value in allowed or config_workload in allowed:
                continue
        if value not in allowed:
            return False
    return True


def _tokens_per_step(metadata: dict[str, str]) -> int:
    try:
        b = int(metadata.get("batch_size") or 0)
        s = int(metadata.get("seq_len") or 0)
        g = int(metadata.get("gradient_accumulation_steps") or 0)
    except (TypeError, ValueError):
        return 0
    return b * s * g


def _read_step_throughput(run_dir: Path, tokens_per_step: int) -> tuple[dict[int, float], float]:
    path = run_dir / "step_samples.csv"
    if tokens_per_step <= 0 or not path.exists():
        return {}, 0.0
    by_step: dict[int, float] = {}
    total_ms = 0.0
    try:
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("is_warmup", "")).strip().lower() in {"true", "1", "yes"}:
                    continue
                ms = _to_float(row.get("step_milliseconds"))
                if ms <= 0.0:
                    continue
                step = 0
                for key in ("measured_step", "step", "raw_step"):
                    try:
                        candidate = int(float(row.get(key)))
                    except (TypeError, ValueError):
                        candidate = 0
                    if candidate > 0:
                        step = candidate
                        break
                if step <= 0:
                    step = len(by_step) + 1
                by_step[step] = tokens_per_step / (ms / 1000.0)
                total_ms += ms
    except Exception:
        return {}, 0.0
    return by_step, total_ms / 1000.0


def _load_runs(args: argparse.Namespace) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for profile_path in _find_profile_paths(args.input_root, args.run_dir):
        profile = _safe_read_json(profile_path)
        metadata = _infer_metadata(profile_path, profile)
        if metadata is None:
            continue
        tokens_per_step = _tokens_per_step(metadata)
        by_step, elapsed = _read_step_throughput(profile_path.parent, tokens_per_step)
        if not by_step:
            continue
        record = RunRecord(
            run_dir=profile_path.parent,
            profile_path=profile_path,
            metadata=metadata,
            tokens_per_step=tokens_per_step,
            by_step=by_step,
            measured_elapsed_seconds=elapsed,
        )
        if _matches_filters(record, args):
            runs.append(record)
    return sorted(
        runs,
        key=lambda run: (
            run.metadata.get("workload", ""),
            int(run.metadata.get("seq_len") or 0),
            run.metadata.get("backend", ""),
            run.metadata.get("router_mode", ""),
            run.metadata.get("liger_loss", ""),
            run.metadata.get("recompute", ""),
            run.metadata.get("expert_policy", ""),
            str(run.run_dir),
        ),
    )


def _prepare_output(path: Path, clean: bool) -> tuple[Path, Path]:
    if clean and path.exists():
        shutil.rmtree(path)
    plots_dir = path  # plots sit directly in the metric folder; data/ stays nested
    data_dir = path / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, data_dir


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if fieldnames:
            writer.writerows(rows)


def _summary_rows(runs: list[RunRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        values = [run.by_step[step] for step in sorted(run.by_step)]
        if not values:
            continue
        rows.append(
            {
                **run.metadata,
                "run_label": run.label,
                "tokens_per_step": run.tokens_per_step,
                "measured_steps": len(values),
                "effective_tokens_per_second": round(run.effective_tokens_per_second(), 2),
                "mean_tokens_per_second": round(statistics.fmean(values), 2),
                "median_tokens_per_second": round(statistics.median(values), 2),
                "min_tokens_per_second": round(min(values), 2),
                "max_tokens_per_second": round(max(values), 2),
                "measured_elapsed_seconds": round(run.measured_elapsed_seconds, 4),
                "profile_json": str(run.profile_path),
                "run_dir": str(run.run_dir),
            }
        )
    return rows


def _step_rows(runs: list[RunRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for step in sorted(run.by_step):
            tps = run.by_step[step]
            step_ms = (run.tokens_per_step / tps * 1000.0) if tps > 0 else 0.0
            rows.append(
                {
                    **run.metadata,
                    "run_label": run.label,
                    "step": step,
                    "tokens_per_second": round(tps, 2),
                    "step_milliseconds": round(step_ms, 4),
                    "profile_json": str(run.profile_path),
                    "run_dir": str(run.run_dir),
                }
            )
    return rows


def _plot_by_step(runs: list[RunRecord], plots_dir: Path) -> None:
    n_runs = len(runs)
    ncols = 2 if n_runs > 2 else 1
    nrows = math.ceil(n_runs / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.0 * ncols, 3.4 * nrows), sharey=True, squeeze=False, constrained_layout=True)
    for idx, run in enumerate(runs):
        ax = axes[idx // ncols][idx % ncols]
        steps = sorted(run.by_step)
        values = [run.by_step[step] for step in steps]
        ax.plot(steps, values, label="tokens/sec", color=THROUGHPUT_COLOR, linewidth=1.8, marker="o", markersize=3)
        eff = run.effective_tokens_per_second()
        if eff > 0.0:
            ax.axhline(eff, color=THROUGHPUT_EFF_COLOR, linestyle="--", linewidth=1.0, label=f"effective {eff:,.0f}")
        ax.set_title(run.label or run.run_dir.name, fontsize=9)
        ax.set_xlabel("Measured step")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, axis="y", alpha=0.25)
        if idx % ncols == 0:
            ax.set_ylabel("Throughput (tokens/sec)")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=7, loc="lower right", frameon=False)
    for idx in range(n_runs, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle("Throughput by Measured Step", fontsize=12)
    fig.savefig(plots_dir / "throughput_by_step.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_readme(out_dir: Path, *, runs: list[RunRecord], reason: str | None = None) -> None:
    lines = [
        "# LF Throughput Artifacts",
        "",
        "Training throughput in tokens/sec, derived per run from step_samples.csv:",
        "`tokens_per_sec = (batch_size x seq_len x grad_accum) / step_seconds`, over post-warmup (measured) steps only.",
        "Effective throughput = total measured tokens / total measured wall time (jitter-robust headline number).",
        "Use the `source` profiler runs for throughput; nsys runs carry profiling overhead.",
        "",
    ]
    if reason:
        lines.extend(["## Status", "", reason, ""])
    else:
        lines.extend(
            [
                "## Files",
                "",
                "- `throughput_by_step.png`: subplots per run; x-axis is measured step, y-axis tokens/sec, dashed line = effective.",
                "- `data/summary.csv`: one row per run (effective/mean/median/min/max tok/s).",
                "- `data/step_summary.csv`: one row per run and measured step.",
                "- `data/index.csv` / `data/index.json`: input run index.",
                "",
                f"Runs included: {len(runs)}.",
                "",
            ]
        )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_empty_outputs(out_dir: Path, clean: bool, reason: str) -> None:
    _plots_dir, data_dir = _prepare_output(out_dir, clean)
    _write_csv(data_dir / "summary.csv", [], SUMMARY_FIELDS)
    _write_csv(data_dir / "step_summary.csv", [], STEP_FIELDS)
    _write_csv(data_dir / "index.csv", [], INDEX_FIELDS)
    (data_dir / "index.json").write_text("[]\n", encoding="utf-8")
    _write_readme(out_dir, runs=[], reason=reason)


def _write_outputs(runs: list[RunRecord], out_dir: Path, clean: bool) -> None:
    plots_dir, data_dir = _prepare_output(out_dir, clean)
    summary_rows = _summary_rows(runs)
    step_rows = _step_rows(runs)
    index_rows = [
        {
            **run.metadata,
            "run_label": run.label,
            "profile_json": str(run.profile_path),
            "run_dir": str(run.run_dir),
        }
        for run in runs
    ]
    _write_csv(data_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    _write_csv(data_dir / "step_summary.csv", step_rows, STEP_FIELDS)
    _write_csv(data_dir / "index.csv", index_rows, INDEX_FIELDS)
    (data_dir / "index.json").write_text(json.dumps(index_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot_by_step(runs, plots_dir)
    _write_readme(out_dir, runs=runs)


def main() -> None:
    args = _parse_args()
    if not args.profiler:
        args.profiler = ["source"]
    _apply_workload_filters(args)
    runs = _load_runs(args)
    if not runs:
        reason = (
            "No runs with usable per-step timing (step_samples.csv) were found for the requested filters. "
            "Throughput is computed from `source` profiler runs."
        )
        _write_empty_outputs(args.output_dir, args.clean_output, reason)
        print(f"[plot_lf_throughput] no runs matched; wrote empty artifacts to {args.output_dir}")
        return
    _write_outputs(runs, args.output_dir, args.clean_output)
    print(f"[plot_lf_throughput] wrote throughput artifacts for {len(runs)} run(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
