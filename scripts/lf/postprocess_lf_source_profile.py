#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write standard LF source-profile artifacts.")
    parser.add_argument("--source-profile-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


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


def _markdown(profile: dict[str, Any]) -> str:
    config = profile.get("config", {})
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
        lines += [
            "## Losses",
            "",
            "| Step | Loss |",
            "|---:|---:|",
        ]
        for row in losses:
            if not isinstance(row, dict):
                continue
            lines.append(f"| {row.get('step', '-')} | {row.get('loss', '-')} |")
        lines.append("")
    return "\n".join(lines)


def _latency_markdown(profile: dict[str, Any]) -> str:
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


def _memory_markdown(profile: dict[str, Any]) -> str:
    memory = profile.get("memory", {})
    gpu = memory.get("gpu", {}) if isinstance(memory, dict) else {}
    return "\n".join(
        [
            "# LF LoRA-SFT Memory",
            "",
            "| Metric | MiB |",
            "|---|---:|",
            f"| peak_hbm_bytes | {_fmt_mib(gpu.get('peak_hbm_bytes'))} |",
            f"| stage_local_peak_hbm_bytes | {_fmt_mib(gpu.get('stage_local_peak_hbm_bytes'))} |",
            "",
        ]
    )


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
    (output_dir / "step_samples.json").write_text(json.dumps(normalized_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "step_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


def main() -> None:
    args = _parse_args()
    profile = json.loads(args.source_profile_json.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "profile.json").write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "table.md").write_text(_markdown(profile), encoding="utf-8")
    (args.output_dir / "lat.md").write_text(_latency_markdown(profile), encoding="utf-8")
    (args.output_dir / "memory.md").write_text(_memory_markdown(profile), encoding="utf-8")
    _write_step_samples(profile, args.output_dir)


if __name__ == "__main__":
    main()
