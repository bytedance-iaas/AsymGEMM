#!/bin/bash
# M6 tier-tradeoff: T1 at the LONG length (512000, b1) — the OOM-probe run.
# Spec (motivation_v2_plots.md M6): T1-at-long infeasibility must be ONE
# attempted run that OOMs, recorded. MAX_SAMPLES=512 = house choice for long
# rows (merge_cpu_modules.md S6). If this FITs, the long length moves to 1M.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

echo "=== M6 T1 long-probe (q32 T1 512k b1) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh q3-32b m6t1l "asym_cpuadamwds|T1|ligerloss1" 512000 1
echo "M6_T1L_EXIT=$?"
echo "=== M6 T1L DONE $(date -u +%H:%M:%S)"
