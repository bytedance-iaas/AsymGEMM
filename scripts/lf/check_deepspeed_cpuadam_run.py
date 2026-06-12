#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


BASIC_OPTIMIZER_MARKER = "DeepSpeed Basic Optimizer = DeepSpeedCPUAdam"
CPUADAM_ENABLED_MARKER = "DeepSpeed CPUAdam runtime enabled = true"


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _source_profile(profile: dict[str, Any]) -> dict[str, Any]:
    nested = profile.get("source_profile")
    return nested if isinstance(nested, dict) else profile


def _read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _is_cpuadam_class(value: Any) -> bool:
    return str(value or "") == "DeepSpeedCPUAdam"


def _profile_cpuadam_summary(profile: dict[str, Any]) -> dict[str, Any]:
    source_profile = _source_profile(profile)
    cpuadam = source_profile.get("cpuadam", {})
    if not isinstance(cpuadam, dict):
        cpuadam = {}
    return cpuadam


def summarize(profile_json: Path | None, train_log: Path | None) -> dict[str, Any]:
    profile = _load_json(profile_json)
    source_profile = _source_profile(profile)
    cpuadam = _profile_cpuadam_summary(profile)
    config = source_profile.get("config", {})
    if not isinstance(config, dict):
        config = {}

    optimizer_class = cpuadam.get("optimizer_class")
    basic_optimizer_class = cpuadam.get("basic_optimizer_class")
    optimizer_wrapper_class = cpuadam.get("optimizer_wrapper_class")
    profile_marker = (
        cpuadam.get("runtime_verified") is True
        or _is_cpuadam_class(basic_optimizer_class)
        or _is_cpuadam_class(optimizer_class)
    )

    text = _read_text(train_log)
    log_marker = BASIC_OPTIMIZER_MARKER in text or CPUADAM_ENABLED_MARKER in text
    enabled = bool(profile_marker or log_marker)

    config_enabled = (
        cpuadam.get("config_optimizer_type") == "AdamW"
        and cpuadam.get("config_offload_optimizer_device") == "cpu"
    ) or config.get("backend") == "zero3_cpuadam"

    marker_source = None
    if profile_marker:
        marker_source = "profile"
    elif log_marker:
        marker_source = "train_log"

    return {
        "enabled": enabled,
        "config_enabled": bool(config_enabled),
        "profile_json": str(profile_json) if profile_json is not None else None,
        "train_log": str(train_log) if train_log is not None else None,
        "optimizer_class": optimizer_class,
        "optimizer_wrapper_class": optimizer_wrapper_class,
        "basic_optimizer_class": basic_optimizer_class,
        "marker_source": marker_source,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that a LF DeepSpeed run used DeepSpeedCPUAdam.")
    parser.add_argument("--profile-json", type=Path)
    parser.add_argument("--train-log", type=Path)
    parser.add_argument("--require-enabled", action="store_true")
    args = parser.parse_args()

    summary = summarize(args.profile_json, args.train_log)
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    if args.require_enabled and not summary["enabled"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
