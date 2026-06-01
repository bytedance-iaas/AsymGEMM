#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write LF profile artifacts and compare LF smoke losses.")
    parser.add_argument("--profile-json", type=Path, help="Profile JSON to convert into CSV artifacts.")
    parser.add_argument("--source-profile-json", type=Path, help="Source-profile JSON to convert into markdown artifacts.")
    parser.add_argument("--output-dir", type=Path, help="Directory for profile artifacts.")
    parser.add_argument("--baseline-dir", type=Path, help="Baseline LF run dir for loss comparison.")
    parser.add_argument("--candidate-dir", type=Path, help="Candidate LF run dir for loss comparison.")
    parser.add_argument("--min-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--first-step-rel-tol", type=float, default=0.02)
    parser.add_argument("--max-rel-tol", type=float, default=0.10)
    args = parser.parse_args()
    if not any((args.profile_json, args.source_profile_json, args.baseline_dir, args.candidate_dir)):
        parser.error("provide --profile-json, --source-profile-json, or --baseline-dir/--candidate-dir")
    if (args.profile_json or args.source_profile_json) and args.output_dir is None:
        parser.error("--output-dir is required for profile artifact output")
    if bool(args.baseline_dir) != bool(args.candidate_dir):
        parser.error("--baseline-dir and --candidate-dir must be provided together")
    return args


def _as_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        rows = value.get("rows", [])
    else:
        rows = value
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if fieldnames:
            writer.writerows(rows)


def _fmt_ms(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "-"


def _fmt_mib(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) / (1024.0 ** 2):.2f}"
    return "-"


def _stage_memory_row(profile: dict[str, Any], name: str) -> dict[str, Any]:
    rows = profile.get("stage_memory", {}).get("rows", [])
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and row.get("name") == name:
            return row
    return {}


def _source_summary_markdown(profile: dict[str, Any]) -> str:
    config = profile.get("config", {})
    warmup_steps = int(config.get("warmup_steps", 0) or 0)
    measure_steps = config.get("measure_steps", config.get("max_steps", "-"))
    memory = profile.get("memory", {})
    gpu = memory.get("gpu", {}) if isinstance(memory, dict) else {}
    trainer = profile.get("trainer", {})
    forward_ms = profile.get("forward", {}).get("total_milliseconds")
    backward_ms = profile.get("backward", {}).get("total_milliseconds")
    lines = [
        "# LF LoRA-SFT Source Profile",
        "",
        f"Workload: `{profile.get('workload', '-')}`  ",
        f"Backend: `{config.get('backend', '-')}`  ",
        f"Precision: `{config.get('precision', '-')}`  ",
        f"Seq len: `{config.get('seq_len', '-')}`",
        f"Steps: `{warmup_steps}` warmup + `{measure_steps}` measured",
        "",
        "| Stage | host ms | avg start MiB | avg end MiB | avg local peak MiB | samples |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, total_ms in (("step.forward", forward_ms), ("step.backward", backward_ms)):
        row = _stage_memory_row(profile, name)
        lines.append(
            "| {name} | {ms} | {start} | {end} | {peak} | {samples} |".format(
                name=name,
                ms=_fmt_ms(total_ms),
                start=_fmt_mib(row.get("avg_allocated_start_bytes")),
                end=_fmt_mib(row.get("avg_allocated_end_bytes")),
                peak=_fmt_mib(row.get("avg_local_peak_bytes")),
                samples=row.get("samples", "-"),
            )
        )
    lines += [
        "",
        f"Peak HBM: `{_fmt_mib(gpu.get('peak_hbm_bytes'))} MiB`",
        f"Trainer log: `{trainer.get('trainer_log', '')}`",
        "",
    ]
    losses = trainer.get("losses", [])
    if isinstance(losses, list) and losses:
        measured_losses = [row for row in losses if isinstance(row, dict) and not row.get("is_warmup")]
        lines += [
            "## Measured Losses",
            "",
            "| Step | Raw step | Loss |",
            "|---:|---:|---:|",
        ]
        for row in measured_losses:
            lines.append(f"| {row.get('measured_step', row.get('step', '-'))} | {row.get('raw_step', '-')} | {row.get('loss', '-')} |")
        lines.append("")
    return "\n".join(lines)


def _source_latency_markdown(profile: dict[str, Any]) -> str:
    step_ms = profile.get("step", {}).get("total_milliseconds")
    forward_ms = profile.get("forward", {}).get("total_milliseconds")
    backward_ms = profile.get("backward", {}).get("total_milliseconds")
    return "\n".join(
        [
            "# LF LoRA-SFT Latency",
            "",
            "| Metric | ms |",
            "|---|---:|",
            f"| step.forward + step.backward | {_fmt_ms(step_ms)} |",
            f"| step.forward | {_fmt_ms(forward_ms)} |",
            f"| step.backward | {_fmt_ms(backward_ms)} |",
            "",
        ]
    )


def _source_memory_markdown(profile: dict[str, Any]) -> str:
    memory = profile.get("memory", {})
    gpu = memory.get("gpu", {}) if isinstance(memory, dict) else {}
    memory_attribution = profile.get("memory_attribution", {})
    category_rows = memory_attribution.get("rows", []) if isinstance(memory_attribution, dict) else []
    saved_tensors = memory_attribution.get("saved_tensors", {}) if isinstance(memory_attribution, dict) else {}
    saved_rows = saved_tensors.get("by_owner", []) if isinstance(saved_tensors, dict) else []
    lines = [
        "# LF LoRA-SFT Memory",
        "",
        "| Metric | MiB |",
        "|---|---:|",
        f"| peak_hbm_bytes | {_fmt_mib(gpu.get('peak_hbm_bytes'))} |",
        f"| stage_local_peak_hbm_bytes | {_fmt_mib(gpu.get('stage_local_peak_hbm_bytes'))} |",
        "",
    ]
    if isinstance(category_rows, list) and category_rows:
        lines += [
            "## Tensor Categories",
            "",
            "| Category | Component | Device | MiB |",
            "|---|---|---|---:|",
        ]
        sorted_categories = sorted(
            (row for row in category_rows if isinstance(row, dict)),
            key=lambda row: float(row.get("bytes", 0) or 0),
            reverse=True,
        )
        for row in sorted_categories[:12]:
            lines.append(
                "| {category} | {component} | {device} | {mib} |".format(
                    category=row.get("category", "-"),
                    component=row.get("component", "-"),
                    device=row.get("device", "-"),
                    mib=_fmt_mib(row.get("bytes")),
                )
            )
        lines.append("")
    if isinstance(saved_tensors, dict) and saved_tensors.get("enabled"):
        lines += [
            "## Saved Activation Owners",
            "",
            "| Owner | MiB |",
            "|---|---:|",
        ]
        sorted_saved = sorted(
            (row for row in saved_rows if isinstance(row, dict)),
            key=lambda row: float(row.get("bytes", 0) or 0),
            reverse=True,
        )
        for row in sorted_saved[:20]:
            lines.append(f"| {row.get('owner', '-')} | {_fmt_mib(row.get('bytes'))} |")
        lines += [
            "",
            f"Unique saved tensors: `{saved_tensors.get('unique_tensors', '-')}`  ",
            f"Unique saved tensor bytes: `{_fmt_mib(saved_tensors.get('total_unique_bytes'))} MiB`  ",
            f"Reference-counted saved tensor bytes: `{_fmt_mib(saved_tensors.get('total_reference_bytes'))} MiB`",
            "",
        ]
    elif isinstance(memory_attribution, dict) and not memory_attribution.get("enabled", False):
        lines += ["Saved activation attribution was disabled for this run.", ""]
    return "\n".join(lines)


def _write_step_samples(profile: dict[str, Any], output_dir: Path) -> None:
    step_samples = profile.get("step_samples", {})
    rows = step_samples.get("rows", []) if isinstance(step_samples, dict) else []
    if not isinstance(rows, list) or not rows:
        return
    normalized_rows = [row for row in rows if isinstance(row, dict)]
    if not normalized_rows:
        return
    fieldnames = list(normalized_rows[0].keys())
    for row in normalized_rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    (output_dir / "step_samples.json").write_text(
        json.dumps(normalized_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "step_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


def _stage_value(stage: dict[str, Any], name: str) -> float:
    for row in _as_rows(stage.get("stage_breakdown", {})):
        if row.get("name") == name:
            try:
                return float(row.get("milliseconds", 0.0))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _host_api_value(stage: dict[str, Any], name: str) -> float:
    for row in _as_rows(stage.get("host_api_breakdown", {})):
        if row.get("name") == name:
            try:
                return float(row.get("milliseconds", 0.0))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _layer(name: str) -> str:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return match.group(1) if match else ""


def _module(name: str) -> str:
    if ".self_attn" in name or ".attention" in name:
        return "attention"
    if ".mlp.experts" in name or ".experts" in name:
        return "experts"
    if ".mlp" in name:
        return "mlp"
    if "lora" in name.lower():
        return "lora"
    if "optimizer" in name.lower():
        return "optimizer"
    if "router" in name.lower():
        return "router"
    return "other"


def _profile_memory(profile: dict[str, Any]) -> dict[str, Any]:
    for container in (profile, profile.get("source_profile", {}), profile.get("memory_profile", {})):
        if isinstance(container, dict) and isinstance(container.get("memory_attribution"), dict):
            return container["memory_attribution"]
    return {}


def _timing_by_stage(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(profile.get("stages"), list):
        for stage in profile["stages"]:
            if not isinstance(stage, dict):
                continue
            rows.append(
                {
                    "stage": stage.get("stage", ""),
                    "total_milliseconds": stage.get("total_milliseconds", 0.0),
                    "cuda_kernel_busy_milliseconds": _stage_value(stage, "cuda_kernel_busy_union"),
                    "cuda_memcpy_milliseconds": _stage_value(stage, "cuda_memcpy_union"),
                    "gpu_no_kernel_milliseconds": _stage_value(stage, "gpu_no_kernel_time"),
                    "cuda_runtime_api_milliseconds": _host_api_value(stage, "cuda_runtime_api_sum_overlaps_gpu_timeline"),
                    "cuda_sync_api_milliseconds": _host_api_value(stage, "cuda_synchronization_api_sum_overlaps_gpu_timeline"),
                }
            )
        return rows
    for row in _as_rows(profile.get("step", {})):
        rows.append(
            {
                "stage": row.get("name", ""),
                "host_milliseconds": row.get("milliseconds", 0.0),
                "samples": row.get("samples", ""),
            }
        )
    return rows


def _timing_by_op(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in profile.get("stages", []):
        if not isinstance(stage, dict):
            continue
        stage_name = stage.get("stage", "")
        for row in _as_rows(stage.get("operation_kernel_time", {})):
            name = str(row.get("name", ""))
            rows.append(
                {
                    "stage": stage_name,
                    "operation": name,
                    "layer": _layer(name),
                    "module": _module(name),
                    "gpu_kernel_milliseconds": row.get("milliseconds", 0.0),
                    "percent_of_stage": row.get("percent", 0.0),
                }
            )
    return rows


def _timing_by_layer(op_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str], float] = {}
    for row in op_rows:
        layer = str(row.get("layer", ""))
        if not layer:
            continue
        key = (str(row.get("stage", "")), layer)
        totals[key] = totals.get(key, 0.0) + float(row.get("gpu_kernel_milliseconds") or 0.0)
    return [
        {"stage": stage, "layer": layer, "gpu_kernel_milliseconds": value}
        for (stage, layer), value in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    ]


def _timing_by_module(op_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str], float] = {}
    for row in op_rows:
        key = (str(row.get("stage", "")), str(row.get("module", "other")))
        totals[key] = totals.get(key, 0.0) + float(row.get("gpu_kernel_milliseconds") or 0.0)
    return [
        {"stage": stage, "module": module, "gpu_kernel_milliseconds": value}
        for (stage, module), value in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    ]


def _kernel_by_op(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in profile.get("stages", []):
        if not isinstance(stage, dict):
            continue
        stage_name = stage.get("stage", "")
        for row in _as_rows(stage.get("operation_kernel_classes", {})):
            op = str(row.get("operation", ""))
            rows.append(
                {
                    "stage": stage_name,
                    "operation": op,
                    "layer": _layer(op),
                    "module": _module(op),
                    "kernel_class": row.get("kernel_class", ""),
                    "milliseconds": row.get("milliseconds", 0.0),
                    "percent_of_stage": row.get("percent", 0.0),
                }
            )
    return rows


def _unattributed(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in profile.get("stages", []):
        if not isinstance(stage, dict):
            continue
        stage_name = stage.get("stage", "")
        for row in _as_rows(stage.get("gpu_no_kernel_gap_attribution", {})):
            name = str(row.get("name", ""))
            if "unattributed" in name.lower() or "no-kernel" in name.lower():
                rows.append(
                    {
                        "stage": stage_name,
                        "name": name,
                        "milliseconds": row.get("milliseconds", 0.0),
                        "percent_of_stage": row.get("percent", 0.0),
                    }
                )
        for row in _as_rows(stage.get("operation_kernel_time", {})):
            name = str(row.get("name", ""))
            if "unattributed" in name.lower():
                rows.append(
                    {
                        "stage": stage_name,
                        "name": name,
                        "milliseconds": row.get("milliseconds", 0.0),
                        "percent_of_stage": row.get("percent", 0.0),
                    }
                )
    return rows


def _memory_by_category(profile: dict[str, Any]) -> list[dict[str, Any]]:
    memory = _profile_memory(profile)
    return _as_rows(memory.get("rows", []))


def _memory_by_module(profile: dict[str, Any]) -> list[dict[str, Any]]:
    memory = _profile_memory(profile)
    saved = memory.get("saved_tensors", {}) if isinstance(memory, dict) else {}
    rows = _as_rows(saved.get("by_owner", [])) if isinstance(saved, dict) else []
    if rows:
        return rows
    return _as_rows(profile.get("stage_memory", {}))


def _write_source_artifacts(source_profile_json: Path, output_dir: Path, profile_json: Path | None) -> None:
    profile = json.loads(source_profile_json.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    target_profile = profile_json or output_dir / "profile.json"
    target_profile.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = _source_summary_markdown(profile)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    (output_dir / "table.md").write_text(summary, encoding="utf-8")
    (output_dir / "lat.md").write_text(_source_latency_markdown(profile), encoding="utf-8")
    (output_dir / "memory.md").write_text(_source_memory_markdown(profile), encoding="utf-8")
    _write_step_samples(profile, output_dir)


def _write_profile_csv_artifacts(profile_json: Path, output_dir: Path) -> None:
    profile = json.loads(profile_json.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_rows = _timing_by_stage(profile)
    op_rows = _timing_by_op(profile)
    _write_csv(output_dir / "timing_by_stage.csv", stage_rows)
    _write_csv(output_dir / "timing_by_op.csv", op_rows)
    _write_csv(output_dir / "timing_by_layer.csv", _timing_by_layer(op_rows))
    _write_csv(output_dir / "timing_by_module.csv", _timing_by_module(op_rows))
    _write_csv(output_dir / "kernel_by_op.csv", _kernel_by_op(profile))
    _write_csv(output_dir / "memory_by_category.csv", _memory_by_category(profile))
    _write_csv(output_dir / "memory_by_module.csv", _memory_by_module(profile))
    _write_csv(output_dir / "unattributed_timing.csv", _unattributed(profile))


def _find_loss_log(run_dir: Path) -> Path:
    candidates = [run_dir / "trainer_log.jsonl", *sorted(run_dir.glob("loss_*.trainer_log.jsonl"))]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no trainer loss log found in {run_dir}")


def _read_losses(run_dir: Path, *, warmup_steps: int = 0) -> list[tuple[int, float]]:
    log_path = _find_loss_log(run_dir)
    losses: list[tuple[int, float]] = []
    for line_no, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if "loss" not in record:
            continue
        loss = float(record["loss"])
        if not math.isfinite(loss):
            raise ValueError(f"{log_path}:{line_no} has non-finite loss {loss}")
        step = int(record.get("current_steps", record.get("step", len(losses) + 1)))
        if step <= warmup_steps:
            continue
        step -= warmup_steps
        losses.append((step, loss))
    if not losses:
        raise ValueError(f"{log_path} has no loss records")
    return losses


def _rel_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def _compare_losses(args: argparse.Namespace) -> None:
    baseline = _read_losses(args.baseline_dir, warmup_steps=max(args.warmup_steps, 0))
    candidate = _read_losses(args.candidate_dir, warmup_steps=max(args.warmup_steps, 0))
    if len(baseline) < args.min_steps:
        raise SystemExit(f"baseline has {len(baseline)} loss records, expected at least {args.min_steps}")
    if len(candidate) < args.min_steps:
        raise SystemExit(f"candidate has {len(candidate)} loss records, expected at least {args.min_steps}")

    baseline = baseline[: args.min_steps]
    candidate = candidate[: args.min_steps]
    first_rel = _rel_diff(baseline[0][1], candidate[0][1])
    max_rel = max(_rel_diff(base_loss, cand_loss) for (_, base_loss), (_, cand_loss) in zip(baseline, candidate))

    print(f"baseline_first={baseline[0][1]:.6f}")
    print(f"candidate_first={candidate[0][1]:.6f}")
    print(f"first_step_rel_diff={first_rel:.6f}")
    print(f"max_{args.min_steps}_step_rel_diff={max_rel:.6f}")

    if first_rel > args.first_step_rel_tol:
        raise SystemExit(f"first-step relative diff {first_rel:.6f} exceeds {args.first_step_rel_tol:.6f}")
    if max_rel > args.max_rel_tol:
        raise SystemExit(f"max relative diff {max_rel:.6f} exceeds {args.max_rel_tol:.6f}")


def main() -> None:
    args = _parse_args()
    if args.source_profile_json:
        _write_source_artifacts(args.source_profile_json, args.output_dir, args.profile_json)
    if args.profile_json:
        _write_profile_csv_artifacts(args.profile_json, args.output_dir)
    if args.baseline_dir:
        _compare_losses(args)


if __name__ == "__main__":
    main()
