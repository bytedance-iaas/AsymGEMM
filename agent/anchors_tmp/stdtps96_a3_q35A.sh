#!/bin/bash
# stdtps96_a3_q35A.sh — Agent 3 / Phase A (standardize_tps_96gb.md): Qwen3.5-
# 35B-A3B 2-RANK CEILING SEARCH under the 96G occupiers. Start at the prior
# (320K, sEP-T2 b1), step +-64K while it trains/fails, then bisect the failing
# gap to <=16K. Tier ladder at each failing rung: T2 -> T2B (185G 2r family;
# T1 is HBM-heavier and never used at 2r for this model). Every probe banks.
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU="0,1" HOSTFLOOR=300 OCC_PIDS="613592 613593"
. /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.c14.sh
export GPU_POOL="0,1" DDP_TIMEOUT=1500
python3 /workspace/AsymGEMM-SFT-38/.repair_dataset_info.py >> "$S" 2>&1 || true
echo "== STDTPS96-A3-Q35-PHASEA begin (2r ceiling; prior 320-384K sEP-T2) $(date '+%m-%d %H:%M')" >> "$S"
Q35=q3.5-35b-a3b; P="none|false|false|false|false|false"; BE=asym_sepplan2_cpuadamwds
declare -A RES

ladder() { local s=$1 v
  [ -n "${RES[$s]:-}" ] && { [ "${RES[$s]%%:*}" = "TRAINED" ]; return; }
  v=$(run_cell n35t2$((s/1000)) $Q35 "${BE}|T2" $s "1" "$P" 2)
  if [ "$v" = "TRAINED" ]; then RES[$s]="TRAINED:T2"; return 0; fi
  ooms "$v" || { echo "ABORT-Q35A infra $v @T2 s=$s" >> "$S"; exit 1; }
  v=$(run_cell n35t2b$((s/1000)) $Q35 "${BE}|T2B" $s "1" "$P" 2)
  if [ "$v" = "TRAINED" ]; then RES[$s]="TRAINED:T2B"; return 0; fi
  ooms "$v" || { echo "ABORT-Q35A infra $v @T2B s=$s" >> "$S"; exit 1; }
  RES[$s]="OOM"; return 1
}

STEP=64000; S0=320000
if ladder $S0; then
  lo=$S0; s=$((S0+STEP))
  while ladder $s; do lo=$s; s=$((s+STEP)); done
  hi=$s
else
  hi=$S0; s=$((S0-STEP))
  until ladder $s; do hi=$s; s=$((s-STEP)); if [ $s -lt 32000 ]; then echo "ABORT-Q35A floor reached" >> "$S"; exit 1; fi; done
  lo=$s
fi
while [ $((hi-lo)) -gt 16000 ]; do
  mid=$(( (lo + (hi-lo)/2) / 16000 * 16000 )); [ $mid -le $lo ] && mid=$((lo+16000))
  if ladder $mid; then lo=$mid; else hi=$mid; fi
done
echo "Q35-2R-CAP $lo bracket=($lo,$hi] tier=${RES[$lo]#TRAINED:} $(date '+%m-%d %H:%M')" >> "$S"
echo "$lo ${RES[$lo]#TRAINED:}" > "$LOGD/q35_cap96.txt"
echo "== STDTPS96-A3-Q35-PHASEA-DONE $(date '+%m-%d %H:%M')" >> "$S"
