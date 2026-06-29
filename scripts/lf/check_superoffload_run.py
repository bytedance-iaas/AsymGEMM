#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


FINAL_OPTIMIZER_MARKER = "DeepSpeed Final Optimizer = SuperOffloadOptimizer_Stage3"


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def summarize(profile_json: Path | None, train_log: Path | None) -> dict[str, Any]:
    profile = _load_json(profile_json)
    superoffload = profile.get("superoffload", {})
    if not isinstance(superoffload, dict):
        superoffload = {}

    optimizer_class = superoffload.get("optimizer_class")
    profile_marker = optimizer_class == "SuperOffloadOptimizer_Stage3"
    config_enabled = (
        superoffload.get("config_super_offload") is True
        or profile.get("config", {}).get("backend") in {"superoffload", "superoffload_mem", "superoffload_mem_opnvme", "superoffload_mem_panvme"}
    )
    log_marker = FINAL_OPTIMIZER_MARKER in _read_text(train_log)
    enabled = profile_marker or log_marker

    marker_source = None
    if profile_marker:
        marker_source = "profile"
    elif log_marker:
        marker_source = "train_log"

    return {
        "enabled": enabled,
        "config_enabled": config_enabled,
        "profile_json": str(profile_json) if profile_json is not None else None,
        "train_log": str(train_log) if train_log is not None else None,
        "optimizer_class": optimizer_class,
        "marker_source": marker_source,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that a LF run used SuperOffload.")
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
