#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LLaMA-Factory smoke trainer losses.")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--min-steps", type=int, default=10)
    parser.add_argument("--first-step-rel-tol", type=float, default=0.02)
    parser.add_argument("--max-rel-tol", type=float, default=0.10)
    return parser.parse_args()


def _find_log(run_dir: Path) -> Path:
    candidates = [run_dir / "trainer_log.jsonl", *sorted(run_dir.glob("loss_*.trainer_log.jsonl"))]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no trainer loss log found in {run_dir}")


def _read_losses(run_dir: Path) -> list[tuple[int, float]]:
    log_path = _find_log(run_dir)
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
        losses.append((step, loss))
    if not losses:
        raise ValueError(f"{log_path} has no loss records")
    return losses


def _rel_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def main() -> None:
    args = _parse_args()
    baseline = _read_losses(Path(args.baseline_dir))
    candidate = _read_losses(Path(args.candidate_dir))
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


if __name__ == "__main__":
    main()
