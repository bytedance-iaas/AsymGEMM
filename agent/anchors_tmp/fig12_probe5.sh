#!/bin/bash
# fig12 probe 5 (v2) — 30B ratio-length pair at 900k (same-day):
#   B = naive-AsymGEMM integration (T1: unsloth-GC ohbm0 + staged/streamed
#       base GEMMs, GPU LoRA legs, no swap schedule)
#   A = AsymLoRA at the scheduler's pick (T2 probe zone; walk to T2B on OOM)
# 1.1M/1.4M rows come from banked m6v2 runs (A: t2b1100b=381, t2b1400=320;
# B: m6v2t11100 = measured CUDA OOM) — no reruns needed.
set -uo pipefail
export GPU="${GPU:-0}" HOSTFLOOR="${HOSTFLOOR:-600}" GPU_POOL="${GPU:-0}"
. /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/fig12_lib.sh
POL="none|false|false|false|false|false"     # T1: no attnact
POLA="none|false|true|false|false|false"     # A arms: attnact on
echo "PROBE5 begin $(date +%H:%M)" >> "$S"
ARM_ENV="" run_cell nb900 q3-30b-a3b "asym_cpuadamwds|T1" 900000 "1" "$POL" 1
v=$(ARM_ENV="" run_cell na900 q3-30b-a3b "asym_cpuadamwds|T2" 900000 "1" "$POLA" 1)
if [ "$v" != "TRAINED" ]; then
  echo "PROBE5 A-T2@900k -> $v; falling to T2B" >> "$S"
  ARM_ENV="" run_cell na900b q3-30b-a3b "asym_cpuadamwds|T2B" 900000 "1" "$POLA" 1
fi
echo "PROBE5-DONE $(date +%H:%M)" >> "$S"
