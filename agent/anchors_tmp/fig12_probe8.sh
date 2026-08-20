#!/bin/bash
# fig11 right-panel candidate — Hunyuan-A13B (non-Qwen MoE) peak-HBM panel:
#   SO      = asym_cpuadamwds|recomp      (weight+grad offload, ker000)
#   SO+RC   = asym_cpuadamwds|unsloth-off (roots offloaded GC)
#   Ours    = asym_cpuadamwds|T3          (streaming tier, attnact)
# Adaptive: walk rc up to its wall, uo up to its wall, then T3 at
# {L1=rc last fit, L2=uo last fit, L3=first uo-OOM length} (walk down on OOM).
# W1+M1 fit/peak probes (peak_reserved from profile.json of TRAINED cells).
set -uo pipefail
export GPU="${GPU:-0}" HOSTFLOOR="${HOSTFLOOR:-500}" GPU_POOL="${GPU:-0}"
. /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/fig12_lib.sh
export MAX_STEPS=1 WARMUP_STEPS=1
POLN="none|false|false|false|false|false"
POLA="none|false|true|false|false|false"
M=hunyuan-a13b
OFF="ASYM_CPU_ADAMW_GRAD_OFFLOAD=true ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true"

echo "PROBE8 begin $(date +%H:%M)" >> "$S"
RCLAST=""; RCWALL=""
for sk in 128 192 256 320 384; do
  v=$(ARM_ENV="$OFF" run_cell "hyrc${sk}" "$M" "asym_cpuadamwds|recomp" "${sk}000" "1" "$POLN" 1)
  echo "PROBE8 rc ${sk}k -> $v" >> "$S"
  if [ "$v" = "TRAINED" ]; then RCLAST=$sk; else RCWALL=$sk; break; fi
done
UOLAST=""; UOWALL=""
for sk in 256 384 512 640 768; do
  v=$(ARM_ENV="$OFF" run_cell "hyuo${sk}" "$M" "asym_cpuadamwds|unsloth-off" "${sk}000" "1" "$POLN" 1)
  echo "PROBE8 uo ${sk}k -> $v" >> "$S"
  if [ "$v" = "TRAINED" ]; then UOLAST=$sk; else UOWALL=$sk; break; fi
done
echo "PROBE8 walls: rc last=${RCLAST:-?} wall=${RCWALL:-none<=384}; uo last=${UOLAST:-?} wall=${UOWALL:-none<=768}" >> "$S"
L1=${RCLAST:-128}; L2=${UOLAST:-512}; L3=${UOWALL:-768}
for L in $L1 $L2 $L3; do
  v=$(ARM_ENV="" run_cell "hyt3_${L}" "$M" "asym_cpuadamwds|T3" "${L}000" "1" "$POLA" 1)
  echo "PROBE8 T3 ${L}k -> $v" >> "$S"
  if [ "$v" != "TRAINED" ] && [ "$L" = "$L3" ]; then
    L3b=$(( (L3 + UOLAST) / 2 / 32 * 32 ))
    ARM_ENV="" run_cell "hyt3_${L3b}" "$M" "asym_cpuadamwds|T3" "${L3b}000" "1" "$POLA" 1
    echo "PROBE8 T3 walkdown ${L3b}k" >> "$S"
  fi
done
echo "PROBE8-DONE $(date +%H:%M)" >> "$S"
