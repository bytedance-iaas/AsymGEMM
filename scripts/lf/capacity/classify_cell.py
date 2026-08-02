#!/usr/bin/env python3
"""Classify a capacity cell (rebuilt 2026-07-25): OK / G_OOM / C_OOM(load|steady) /
CRASH / UNKNOWN from trainer log + sampler trace + watchdog marker; append ledger row.

Usage: classify_cell.py CELL CAPDIR [--floor 25] [--t0 E] [--t1 E] [--hbm-mib N]
Reads  CAPDIR/logs/cell_CELL.log, CAPDIR/traces/trace_CELL.tsv,
       CAPDIR/logs/cell_CELL.killed (external watchdog marker, optional)
Writes CAPDIR/ledger.tsv row: cell verdict peak_unevict min_avail peak_hbm steps wall_s
"""
import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell")
    ap.add_argument("capdir")
    ap.add_argument("--floor", type=float, default=25.0)
    ap.add_argument("--t0", type=float, default=0.0, help="run start epoch (window the trace)")
    ap.add_argument("--t1", type=float, default=0.0, help="run end epoch")
    ap.add_argument("--hbm-mib", type=float, default=0.0, help="peak GPU memory.used (MiB) polled by the runner")
    a = ap.parse_args()

    log_path = os.path.join(a.capdir, "logs", f"cell_{a.cell}.log")
    trace_path = os.path.join(a.capdir, "traces", f"trace_{a.cell}.tsv")
    killed_path = os.path.join(a.capdir, "logs", f"cell_{a.cell}.killed")

    log = ""
    if os.path.exists(log_path):
        with open(log_path, errors="replace") as f:
            log = f.read()

    peak_unevict = -1.0
    min_avail = 1e9
    rows = 0
    if os.path.exists(trace_path):
        with open(trace_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 7:
                    continue
                try:
                    t = float(parts[0]); unevict = float(parts[3]); avail = float(parts[6])
                except ValueError:
                    continue
                if a.t0 and t < a.t0:
                    continue
                if a.t1 and t > a.t1:
                    continue
                rows += 1
                peak_unevict = max(peak_unevict, unevict)
                min_avail = min(min_avail, avail)
    if not rows:
        min_avail = -1.0

    peak_hbm = a.hbm_mib / 1024.0 if a.hbm_mib > 0 else -1.0

    # Verdict rules (2026-07-25, after the H-chain audit): a cell is OK ONLY on
    # positive evidence of a completed training run (train_runtime). The old
    # classifier keyed on driver rc=0 — which is 0 even when the training
    # command fails — and stamped load-phase footprints of CRASHED runs as OK
    # (every h=12800 64k/32k cell: GQA crash at the first attention call).
    # NOTE (D1 attrib, 2026-07-25): do NOT phase-classify from the teardown
    # "asym_forward_calls=" line of a killed run — it prints 0 spuriously.
    gpu_oom = ("CUDA out of memory" in log) or ("OutOfMemoryError" in log)
    trainer_watchdog = "[host-mem-watchdog]" in log
    ext_killed = os.path.exists(killed_path)
    reached_training = "***** Running training *****" in log
    train_ok = ("train_runtime" in log) and ("Training command failed" not in log)
    crashed = ("Training command failed" in log) or ("Traceback (most recent call last)" in log)

    if trainer_watchdog or ext_killed:
        verdict = "C_OOM(steady)" if reached_training else "C_OOM(load)"
    elif gpu_oom:
        verdict = "G_OOM"
    elif train_ok:
        verdict = "OK"
    elif crashed:
        verdict = "CRASH"
    else:
        verdict = "UNKNOWN"

    row = (f"{a.cell}\t{verdict}\t{peak_unevict:.1f}\t{min_avail:.1f}\t"
           f"{peak_hbm:.1f}\t-\t-")
    ledger = os.path.join(a.capdir, "ledger.tsv")
    if not os.path.exists(ledger):
        with open(ledger, "w") as f:
            f.write("cell\tverdict\tpeak_unevict_gib\tmin_avail_gib\tpeak_hbm_gib\tsteps\twall_s\n")
    with open(ledger, "a") as f:
        f.write(row + "\n")
    print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
