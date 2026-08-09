#!/bin/bash
# M6 tier-tradeoff: LONG group feasible tiers — q3-32b, seq 512000, b1.
# Run AFTER .m6_t1long.sh confirms T1 OOM at 512k. MAX_SAMPLES=512 (house
# choice for long rows, merge_cpu_modules.md S6).
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

echo "=== M6 T2 long (q32 T2 512k b1) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh q3-32b m6t2l "asym_cpuadamwds|T2|ligerloss1" 512000 1
echo "M6_T2L_EXIT=$?"

echo "=== M6 T3 long (q32 T3 512k b1) $(date -u +%H:%M:%S)"
MAX_SAMPLES=512 bash scripts/lf/tp_probe.sh q3-32b m6t3l "asym_cpuadamwds|T3|ligerloss1" 512000 1
echo "M6_T3L_EXIT=$?"

echo "=== M6 LONG DONE $(date -u +%H:%M:%S)"
