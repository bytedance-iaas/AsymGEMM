#!/usr/bin/env python3
"""Validate staged KT ARM BF16 SFT optimization benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_command(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=False, text=True, capture_output=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key in ("results", "latency"):
        value = payload.get(key)
        if isinstance(value, list):
            items.extend(item for item in value if isinstance(item, dict))
    return items


def find_latency(payload: dict[str, Any], backend: str, qlen: int) -> dict[str, Any]:
    backend_l = backend.lower()
    aliases = {backend_l}
    if backend_l == "armbf16_sft":
        aliases.update({"kt_armbf16", "armbf16_sft", "armbf16_sft".lower()})
    for item in _items(payload):
        name = str(item.get("method", item.get("backend", ""))).lower()
        item_qlen = int(item.get("qlen", payload.get("config", {}).get("qlen", -1)))
        if name in aliases and item_qlen == qlen:
            return item
    raise AssertionError(f"latency item not found for backend={backend!r}, qlen={qlen}")


def assert_accuracy(payload: dict[str, Any], max_abs: float, rel_l2: float) -> None:
    for item in payload.get("accuracy_vs_torch", []):
        for key in ("output", "grad_input"):
            stats = item.get(key)
            if isinstance(stats, dict) and stats.get("max_abs", 0.0) > max_abs and stats.get("rel_l2", 0.0) > rel_l2:
                raise AssertionError(f"{item.get('backend')} {key} failed accuracy gate: {stats}")


def assert_speedup(baseline: dict[str, Any], candidate: dict[str, Any], min_speedup_pct: float) -> None:
    base = float(baseline["latency_ms_mean"])
    cand = float(candidate["latency_ms_mean"])
    speedup = (base - cand) / base * 100.0
    if speedup < min_speedup_pct:
        raise AssertionError(f"speedup {speedup:.2f}% is below required {min_speedup_pct:.2f}%")


def assert_regression(baseline: dict[str, Any], candidate: dict[str, Any], max_regression_pct: float) -> None:
    base = float(baseline["latency_ms_mean"])
    cand = float(candidate["latency_ms_mean"])
    regression = (cand - base) / base * 100.0
    if regression > max_regression_pct:
        raise AssertionError(f"regression {regression:.2f}% exceeds allowed {max_regression_pct:.2f}%")


def assert_kernel(payload: dict[str, Any], backend: str, expected_kernel: str) -> None:
    config_kernel = payload.get("base_projection_kernel") or payload.get("config", {}).get("base_projection_kernel")
    if config_kernel == expected_kernel:
        return
    backend_l = backend.lower()
    for item in _items(payload):
        name = str(item.get("method", item.get("backend", ""))).lower()
        if name == backend_l and item.get("base_projection_kernel") == expected_kernel:
            return
    raise AssertionError(f"base_projection_kernel={expected_kernel!r} not found for backend={backend!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--qlen", type=int, required=True)
    parser.add_argument("--min-speedup-pct", type=float)
    parser.add_argument("--max-regression-pct", type=float)
    parser.add_argument("--max-abs", type=float, default=0.0)
    parser.add_argument("--rel-l2", type=float, default=0.0)
    parser.add_argument("--require-kernel")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_payload = load_json(args.candidate_json)
    candidate_item = find_latency(candidate_payload, args.backend, args.qlen)
    if args.max_abs > 0.0 or args.rel_l2 > 0.0:
        assert_accuracy(candidate_payload, args.max_abs, args.rel_l2)
    if args.require_kernel:
        assert_kernel(candidate_payload, args.backend, args.require_kernel)
    if args.baseline_json is not None:
        baseline_payload = load_json(args.baseline_json)
        baseline_item = find_latency(baseline_payload, args.backend, args.qlen)
        if args.min_speedup_pct is not None:
            assert_speedup(baseline_item, candidate_item, args.min_speedup_pct)
        if args.max_regression_pct is not None:
            assert_regression(baseline_item, candidate_item, args.max_regression_pct)
    print(json.dumps({"stage": args.stage, "status": "pass", "candidate": candidate_item}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        raise
