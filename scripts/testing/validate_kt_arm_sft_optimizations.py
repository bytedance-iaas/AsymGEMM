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


def normalize_backend_name(name: str) -> str:
    normalized = name.lower()
    aliases = {
        "armbf16_sft": "kt_armbf16",
        "armbf16": "kt_armbf16",
        "kt_armbf16": "kt_armbf16",
        "torchbf16_sft": "kt_torchbf16",
        "torchbf16": "kt_torchbf16",
        "kt_torchbf16": "kt_torchbf16",
    }
    return aliases.get(normalized, normalized)


def find_latency(payload: dict[str, Any], backend: str, qlen: int) -> dict[str, Any]:
    backend_l = normalize_backend_name(backend)
    for item in _items(payload):
        name = normalize_backend_name(str(item.get("method", item.get("backend", ""))))
        item_qlen = int(item.get("qlen", payload.get("config", {}).get("qlen", -1)))
        if name == backend_l and item_qlen == qlen:
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


def parse_kv_lines(path: Path, prefix: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith(prefix):
                continue
            record: dict[str, str] = {}
            for token in line.split()[1:]:
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                record[key] = value
            records.append(record)
    return records


def assert_item_field(item: dict[str, Any], field: str, expected: str) -> None:
    actual = item.get(field)
    if str(actual) != expected:
        raise AssertionError(f"{field}={actual!r}, expected {expected!r}")


def assert_item_bool(item: dict[str, Any], field: str) -> None:
    if not bool(item.get(field, False)):
        raise AssertionError(f"{field} was not true in latency item: {item}")


def assert_route_metadata(item: dict[str, Any]) -> None:
    valid = int(item.get("valid_route_count", 0) or 0)
    padded = int(item.get("padded_route_count", 0) or 0)
    active = int(item.get("active_expert_count", 0) or 0)
    max_local = int(item.get("max_local_routes", 0) or 0)
    if valid <= 0 or padded < valid or active <= 0 or max_local <= 0:
        raise AssertionError(
            "route metadata is incomplete: "
            f"valid={valid} padded={padded} active={active} max_local={max_local}"
        )


def assert_route_skew_fields(item: dict[str, Any]) -> None:
    assert_route_metadata(item)
    hottest_value = item.get("last_hottest_expert", -1)
    hottest = -1 if hottest_value is None else int(hottest_value)
    skew = float(item.get("last_route_skew_ratio", 0.0) or 0.0)
    if hottest < 0 or skew < 1.0:
        raise AssertionError(f"route skew metadata is incomplete: hottest={hottest} skew={skew}")


def assert_route_skew_records(records: list[dict[str, str]]) -> None:
    if not records:
        raise AssertionError("profile records are required for route-skew checks")
    record = records[-1]
    active = int(float(record.get("active_experts", "0")))
    valid = int(float(record.get("total_valid_routes", record.get("valid_routes", "0"))))
    max_tokens = int(float(record.get("max_expert_tokens", record.get("max_local_routes", "0"))))
    hottest = int(float(record.get("hottest_expert", "-1")))
    skew = float(record.get("route_skew_ratio", "0"))
    if active < 1 or valid < active or max_tokens < 1 or hottest < 0 or skew < 1.0:
        raise AssertionError(
            "profile route-skew fields are incomplete: "
            f"active={active} valid={valid} max_tokens={max_tokens} hottest={hottest} skew={skew}"
        )


def _coerce_expected(value: str) -> Any:
    if value in {"0", "1"}:
        return value
    return value


def assert_profile_field(records: list[dict[str, str]], key: str, expected: str) -> None:
    if not records:
        raise AssertionError("profile records are required for profile-field checks")
    actual = records[-1].get(key)
    if actual != _coerce_expected(expected):
        raise AssertionError(f"profile field {key}={actual!r}, expected {expected!r}")


def assert_profile_numeric_min(records: list[dict[str, str]], key: str, expected_min: float) -> None:
    if not records:
        raise AssertionError("profile records are required for numeric checks")
    value = records[-1].get(key)
    if value is None:
        raise AssertionError(f"profile field {key!r} missing")
    if float(value) < expected_min:
        raise AssertionError(f"profile field {key}={value} is below {expected_min}")


def assert_pool_backed(records: list[dict[str, str]]) -> None:
    if not records:
        raise AssertionError("pool records are required for pool-backed check")
    if not any(record.get("pool_backed") == "1" for record in records):
        raise AssertionError("no pool log record had pool_backed=1")


def assert_pool_growth_bounded(records: list[dict[str, str]], event: str, max_growth_events: int) -> None:
    if not records:
        raise AssertionError("pool records are required for pool-growth check")
    matches = [record for record in records if record.get("event") == event]
    growth = 0
    last_requested = None
    for record in matches:
        requested = int(float(record.get("requested", "0")))
        if last_requested is None or requested > last_requested:
            growth += 1
            last_requested = requested
    if growth > max_growth_events:
        raise AssertionError(f"pool event {event!r} grew {growth} times, allowed {max_growth_events}")


def assert_nan_check_clean(records: list[dict[str, str]]) -> None:
    for record in records:
        if record.get("nan_count") not in {None, "0"} or record.get("inf_count") not in {None, "0"}:
            raise AssertionError(f"NaN/Inf profile check failed: {record}")
        if record.get("pool_overflow") == "1":
            raise AssertionError(f"pool overflow was recorded: {record}")


def assert_stderr_records(records: list[dict[str, str]], required_key: str | None = None) -> None:
    if not records:
        raise AssertionError("required ARM stderr records were not found")
    if required_key and not any(required_key in record for record in records):
        raise AssertionError(f"required key {required_key!r} missing from stderr records")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--candidate-json", type=Path)
    parser.add_argument("--backend")
    parser.add_argument("--qlen", type=int)
    parser.add_argument("--min-speedup-pct", type=float)
    parser.add_argument("--max-regression-pct", type=float)
    parser.add_argument("--max-abs", type=float, default=0.0)
    parser.add_argument("--rel-l2", type=float, default=0.0)
    parser.add_argument("--require-kernel")
    parser.add_argument("--require-forward-path")
    parser.add_argument("--require-task-dispatch")
    parser.add_argument("--require-warmup", action="store_true")
    parser.add_argument("--require-warmup-ran", action="store_true")
    parser.add_argument("--require-lora-warmup", action="store_true")
    parser.add_argument("--require-aligned-weights", action="store_true")
    parser.add_argument("--require-async-repack-enabled", choices=["0", "1"])
    parser.add_argument("--require-route-metadata", action="store_true")
    parser.add_argument("--require-route-skew-fields", action="store_true")
    parser.add_argument("--require-nan-check-clean", action="store_true")
    parser.add_argument("--require-pool-backed", action="store_true")
    parser.add_argument("--pool-event")
    parser.add_argument("--max-pool-growth-events", type=int)
    parser.add_argument("--require-profile-field", action="append", default=[])
    parser.add_argument("--require-profile-numeric-min", action="append", default=[])
    parser.add_argument("--require-profile-float-min", action="append", default=[])
    parser.add_argument("--stderr-log", type=Path)
    parser.add_argument("--profile-stderr", type=Path)
    parser.add_argument("--require-profile-stderr", action="store_true")
    parser.add_argument("--require-pool-stderr", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stderr_log = args.profile_stderr or args.stderr_log
    profile_records: list[dict[str, str]] = []
    pool_records: list[dict[str, str]] = []
    if stderr_log is not None:
        profile_records = parse_kv_lines(stderr_log, "KT_ARM_SFT_PROFILE")
        pool_records = parse_kv_lines(stderr_log, "KT_ARM_SFT_POOL")
    candidate_payload: dict[str, Any] = {}
    candidate_item: dict[str, Any] = {}
    if args.candidate_json is not None:
        if args.backend is None or args.qlen is None:
            raise AssertionError("--backend and --qlen are required with --candidate-json")
        candidate_payload = load_json(args.candidate_json)
        candidate_item = find_latency(candidate_payload, args.backend, args.qlen)
    elif args.baseline_json is not None or args.require_kernel or args.require_forward_path:
        raise AssertionError("--candidate-json is required for this validation mode")
    if (args.max_abs > 0.0 or args.rel_l2 > 0.0) and candidate_payload:
        assert_accuracy(candidate_payload, args.max_abs, args.rel_l2)
    if args.require_kernel:
        assert_kernel(candidate_payload, args.backend, args.require_kernel)
    if args.require_forward_path:
        if candidate_item:
            assert_item_field(candidate_item, "last_forward_path", args.require_forward_path)
        else:
            assert_profile_field(profile_records, "path", args.require_forward_path)
    if args.require_task_dispatch:
        if candidate_item:
            assert_item_field(candidate_item, "last_task_dispatch", args.require_task_dispatch)
        else:
            assert_profile_field(profile_records, "task_dispatch", args.require_task_dispatch)
    if args.require_warmup or args.require_warmup_ran:
        assert_item_bool(candidate_item, "warmup_ran")
        assert_item_bool(candidate_item, "base_warmup_ran")
    if args.require_lora_warmup:
        assert_item_bool(candidate_item, "lora_warmup_ran")
    if args.require_aligned_weights:
        assert_item_bool(candidate_item, "aligned_weights")
    if args.require_async_repack_enabled is not None:
        expected = args.require_async_repack_enabled == "1"
        actual = bool(candidate_item.get("async_backward_repack_enabled", False))
        if actual != expected:
            raise AssertionError(
                f"async_backward_repack_enabled={actual!r}, expected {expected!r}"
            )
    if args.require_route_metadata:
        assert_route_metadata(candidate_item)
    if args.require_route_skew_fields:
        if candidate_item:
            assert_route_skew_fields(candidate_item)
        else:
            assert_route_skew_records(profile_records)
    if args.require_nan_check_clean:
        assert_nan_check_clean(profile_records)
    if stderr_log is not None:
        if args.require_profile_stderr:
            assert_stderr_records(profile_records, "path")
        if args.require_pool_stderr:
            assert_stderr_records(pool_records, "requested")
    if args.require_pool_backed:
        if bool(candidate_item.get("pool_backed", False)):
            pass
        else:
            assert_pool_backed(pool_records)
    if args.pool_event and args.max_pool_growth_events is not None:
        assert_pool_growth_bounded(pool_records, args.pool_event, args.max_pool_growth_events)
    for field_spec in args.require_profile_field:
        key, expected = field_spec.split("=", 1)
        if str(candidate_item.get(key)) == expected:
            continue
        assert_profile_field(profile_records, key, expected)
    for field_spec in args.require_profile_numeric_min + args.require_profile_float_min:
        key, expected = field_spec.split("=", 1)
        value = candidate_item.get(key)
        if value is not None and float(value) >= float(expected):
            continue
        assert_profile_numeric_min(profile_records, key, float(expected))
    if args.baseline_json is not None:
        if args.backend is None or args.qlen is None or not candidate_item:
            raise AssertionError("--baseline-json requires --candidate-json, --backend, and --qlen")
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
