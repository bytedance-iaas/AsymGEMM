#!/bin/bash
# fig12 probe 6 — GLM-4.7-Flash naive-AsymGEMM (T1) vs AsymLoRA at long
# lengths {640k, 800k, 1.0M}. Uncharted territory (GLMTP stopped at 448k):
# B-T1 ladder walks up to its wall; A walks T2 -> T2B -> T3 per length
# (selector behavior). Datasets auto-build inline (serial, GPU idle during).
set -uo pipefail
export GPU="${GPU:-0}" HOSTFLOOR="${HOSTFLOOR:-500}" GPU_POOL="${GPU:-0}"
. /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/fig12_lib.sh
POL="none|false|false|false|false|false"     # T1: no attnact
POLA="none|false|true|false|false|false"     # A arms: attnact on
M=glm4.7-flash

a_walk() { local tag="$1" seq="$2" v
  for tier in T2 T2B T3; do
    v=$(ARM_ENV="" run_cell "${tag}${tier,,}" "$M" "asym_cpuadamwds|${tier}" "$seq" "1" "$POLA" 1)
    echo "PROBE6 A ${seq} ${tier} -> $v" >> "$S"
    [ "$v" = "TRAINED" ] && return 0
  done; return 1; }

echo "PROBE6 begin $(date +%H:%M)" >> "$S"
BWALL=""
for sk in 640000 800000 1000000; do
  v=$(ARM_ENV="" run_cell "gb${sk%000}" "$M" "asym_cpuadamwds|T1" "$sk" "1" "$POL" 1)
  echo "PROBE6 B-T1 ${sk} -> $v" >> "$S"
  if [ "$v" != "TRAINED" ]; then BWALL="$sk"; break; fi
done
[ -n "$BWALL" ] && echo "PROBE6 naive wall at ${BWALL}" >> "$S"
a_walk ga640 640000
a_walk ga800 800000
a_walk ga1000 1000000
echo "PROBE6-DONE $(date +%H:%M)" >> "$S"
