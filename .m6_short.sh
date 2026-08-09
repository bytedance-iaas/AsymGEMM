#!/bin/bash
# M6 tier-tradeoff (motivation_v2_plots.md): SHORT group — q3-32b, seq 32000, b1.
# T1/T2/T3 tier presets, tp_probe defaults (w1+m2, MAX_SAMPLES=1024, PROFILERS=source).
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

echo "=== M6 T1 short (q32 T1 32k b1) $(date -u +%H:%M:%S)"
bash scripts/lf/tp_probe.sh q3-32b m6t1s "asym_cpuadamwds|T1|ligerloss1" 32000 1
echo "M6_T1S_EXIT=$?"

echo "=== M6 T2 short (q32 T2 32k b1) $(date -u +%H:%M:%S)"
bash scripts/lf/tp_probe.sh q3-32b m6t2s "asym_cpuadamwds|T2|ligerloss1" 32000 1
echo "M6_T2S_EXIT=$?"

echo "=== M6 T3 short (q32 T3 32k b1) $(date -u +%H:%M:%S)"
bash scripts/lf/tp_probe.sh q3-32b m6t3s "asym_cpuadamwds|T3|ligerloss1" 32000 1
echo "M6_T3S_EXIT=$?"

echo "=== M6 SHORT DONE $(date -u +%H:%M:%S)"
