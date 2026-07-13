#!/usr/bin/env python
"""Compare a baseline vs an NVMe-candidate LF profile and gate on it.

Checks (per --target): per-step memory drop/drift, latency ratios, per-role NVMe read+write
presence, and paired loss identity. Prints a JSON verdict and SystemExit(2) on any failure.

Targets:
  no_change       candidate must NOT change memory (|drift| <= --max-memory-drift-gib) or loss.
  activation_cpu  activation spill: candidate per-step RSS DROPS by >= --min-memory-drop-{gib,pct}.
  base_weight_cpu base-weight paging: same drop expectation on the host-weight-heavy RSS metric.
  maxseq          capability probe: candidate simply trains; memory/ratio gates informational.

The gate is loss-first: pass --max-loss-delta 0 to require bit-identical training loss vs the
baseline (actnvme/panvme change only tensor residency, never math).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

GIB = 1024 ** 3


def _fail(message: str, *, payload: dict[str, Any] | None = None) -> None:
    out = {"ok": False, "error": message}
    if payload:
        out.update(payload)
    print(json.dumps(out, indent=2, sort_keys=True))
    raise SystemExit(2)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return value


def _first_existing(run_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        p = run_dir / name
        if p.is_file():
            return p
    return None


def _load_profile(run_dir: Path) -> tuple[dict[str, Any], Path]:
    path = _first_existing(run_dir, ("source_profile.json", "profile.json"))
    if path is None:
        raise FileNotFoundError(f"{run_dir} has no source_profile.json or profile.json")
    return _load_json(path), path


def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    try:
        r = float(v)
    except (TypeError, ValueError):
        return None
    return r if math.isfinite(r) else None


def _read_measured_rows(run_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    path = _first_existing(run_dir, ("step_samples.csv",))
    if path is None:
        raise FileNotFoundError(f"{run_dir} has no step_samples.csv")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    measured = [r for r in rows if not _truthy(r.get("is_warmup"))]
    if not measured:
        raise ValueError(f"{path} has no measured (non-warmup) rows")
    return measured, path


def _max_metric(rows: list[dict[str, Any]], col: str, source: Path) -> float:
    vals = [v for v in (_float(r.get(col)) for r in rows) if v is not None]
    if not vals:
        raise ValueError(f"{source} has no finite {col!r} values")
    return max(vals)


def _median(rows: list[dict[str, Any]], col: str, source: Path) -> float:
    vals = [v for v in (_float(r.get(col)) for r in rows) if v is not None]
    if not vals:
        raise ValueError(f"{source} has no finite {col!r} values")
    return float(statistics.median(vals))


def _losses(rows: list[dict[str, Any]], profile: dict[str, Any], source: Path) -> list[float]:
    if rows and "loss" in rows[0]:
        vals = [_float(r.get("loss")) for r in rows]
        if all(v is not None for v in vals):
            return [v for v in vals if v is not None]
    trainer = profile.get("trainer", {})
    tl = trainer.get("losses", []) if isinstance(trainer, dict) else []
    vals = [v for v in (_float(v) for v in tl) if v is not None]
    if not vals:
        raise ValueError(f"{source} has no finite per-step loss values")
    return vals


def _memory_col(metric: str) -> str:
    if not metric.startswith("step_samples."):
        raise ValueError(f"--memory-metric must be step_samples.<col>, got {metric!r}")
    return metric.split(".", 1)[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare NVMe-candidate vs baseline LF profiles.")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--target", required=True,
                   choices=("no_change", "activation_cpu", "base_weight_cpu", "maxseq"))
    p.add_argument("--memory-metric", default="step_samples.training_step_process_rss_peak_end_bytes")
    p.add_argument("--min-memory-drop-gib", type=float, default=None)
    p.add_argument("--min-memory-drop-pct", type=float, default=None)
    p.add_argument("--max-memory-drift-gib", type=float, default=2.0)
    p.add_argument("--max-step-ratio", type=float, default=None)
    p.add_argument("--max-forward-ratio", type=float, default=None)
    p.add_argument("--max-backward-ratio", type=float, default=None)
    p.add_argument("--expect-nvme-role", default=None,
                   help="assert asym_nvme.enabled AND role in roles AND bytes_written>0 AND bytes_read>0")
    p.add_argument("--max-loss-delta", type=float, default=None,
                   help="median |loss_i - baseline_loss_i| over measured steps must be <= this")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    failures: list[str] = []
    checks: dict[str, Any] = {}
    try:
        base_profile, _ = _load_profile(args.baseline)
        cand_profile, _ = _load_profile(args.candidate)
        base_rows, base_csv = _read_measured_rows(args.baseline)
        cand_rows, cand_csv = _read_measured_rows(args.candidate)

        # ---- preflight ----
        if len(base_rows) < 3 or len(cand_rows) < 3:
            raise ValueError(
                f"need >=3 measured steps (baseline={len(base_rows)}, candidate={len(cand_rows)})")
        if _first_existing(args.candidate, ("asym_nvme.csv",)) is None:
            raise FileNotFoundError(f"{args.candidate} has no asym_nvme.csv (postprocess did not emit it)")
        cand_cfg = cand_profile.get("config", {}) if isinstance(cand_profile.get("config"), dict) else {}
        cand_nvme = cand_profile.get("asym_nvme", {}) if isinstance(cand_profile.get("asym_nvme"), dict) else {}
        checks["candidate_config_asym_nvme_roles"] = cand_cfg.get("asym_nvme_roles", "")
        checks["candidate_asym_nvme_enabled"] = bool(cand_nvme.get("enabled"))
        checks["candidate_asym_nvme_bytes_written"] = cand_nvme.get("asym_nvme_bytes_written", {})
        checks["candidate_asym_nvme_bytes_read"] = cand_nvme.get("asym_nvme_bytes_read", {})

        # ---- memory (max over measured rows of the chosen step_samples column) ----
        col = _memory_col(args.memory_metric)
        base_mem = _max_metric(base_rows, col, base_csv)
        cand_mem = _max_metric(cand_rows, col, cand_csv)
        drop = base_mem - cand_mem
        checks["baseline_memory_gib"] = base_mem / GIB
        checks["candidate_memory_gib"] = cand_mem / GIB
        checks["memory_drop_gib"] = drop / GIB
        checks["memory_drop_pct"] = 100.0 * drop / base_mem if base_mem > 0 else 0.0

        if args.target == "no_change":
            if abs(drop) > args.max_memory_drift_gib * GIB:
                failures.append(
                    f"memory drift {drop / GIB:.2f} GiB exceeds +/-{args.max_memory_drift_gib} GiB")
        elif args.target in ("activation_cpu", "base_weight_cpu"):
            if args.min_memory_drop_gib is not None and drop < args.min_memory_drop_gib * GIB:
                failures.append(
                    f"memory drop {drop / GIB:.2f} GiB below {args.min_memory_drop_gib} GiB")
            if args.min_memory_drop_pct is not None and checks["memory_drop_pct"] < args.min_memory_drop_pct:
                failures.append(
                    f"memory drop {checks['memory_drop_pct']:.1f}% below {args.min_memory_drop_pct}%")
        # maxseq: memory is informational only.

        # ---- latency ratios (median over measured rows) ----
        for name, col_ms, thresh in (
            ("step", "step_milliseconds", args.max_step_ratio),
            ("forward", "forward_milliseconds", args.max_forward_ratio),
            ("backward", "backward_milliseconds", args.max_backward_ratio),
        ):
            if thresh is None:
                continue
            b = _median(base_rows, col_ms, base_csv)
            c = _median(cand_rows, col_ms, cand_csv)
            ratio = c / b if b > 0 else float("inf")
            checks[f"{name}_ratio"] = ratio
            if ratio > thresh:
                failures.append(f"{name} latency ratio {ratio:.3f} exceeds {thresh}")

        # ---- paired loss identity ----
        if args.max_loss_delta is not None:
            base_losses = _losses(base_rows, base_profile, base_csv)
            cand_losses = _losses(cand_rows, cand_profile, cand_csv)
            n = min(len(base_losses), len(cand_losses))
            if n == 0:
                raise ValueError("no paired losses to compare")
            deltas = [abs(cand_losses[i] - base_losses[i]) for i in range(n)]
            med = float(statistics.median(deltas))
            checks["median_loss_delta"] = med
            checks["max_loss_delta_observed"] = max(deltas)
            checks["paired_steps"] = n
            if med > args.max_loss_delta:
                failures.append(f"median |loss delta| {med:.6g} exceeds {args.max_loss_delta}")

        # ---- per-role NVMe read + write presence ----
        if args.expect_nvme_role is not None:
            role = args.expect_nvme_role
            if not cand_nvme.get("enabled"):
                failures.append("asym_nvme.enabled is not true on candidate")
            else:
                roles = cand_nvme.get("roles", [])
                if role not in roles:
                    failures.append(f"role {role!r} not in asym_nvme.roles {roles}")
                w = int((cand_nvme.get("asym_nvme_bytes_written", {}) or {}).get(role, 0) or 0)
                r = int((cand_nvme.get("asym_nvme_bytes_read", {}) or {}).get(role, 0) or 0)
                checks[f"nvme_bytes_written[{role}]"] = w
                checks[f"nvme_bytes_read[{role}]"] = r
                if w <= 0:
                    failures.append(f"asym_nvme bytes_written[{role}] is {w} (expected > 0)")
                if r <= 0:
                    failures.append(f"asym_nvme bytes_read[{role}] is {r} (expected > 0)")
    except Exception as exc:
        _fail(str(exc))

    result = {
        "ok": not failures,
        "failures": failures,
        "target": args.target,
        "checks": checks,
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
