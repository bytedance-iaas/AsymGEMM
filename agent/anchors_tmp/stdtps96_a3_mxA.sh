#!/bin/bash
# stdtps96_a3_mxA.sh — Agent 3 / Phase A: Mixtral-8x22B 2-RANK CEILING under
# the 96G occupiers. Prior ~112-144K (T1). sdp2 shared fabric (arena 285,
# mixtral precedent; host shm — the HBM occupier does not touch it). 2r tier
# ladder = T1 ONLY (T2/T3 2r host-dead per the 185G record). Start 128K b1,
# step +-32K, bisect to <=16K. FUSED local ckpt required (built 02:54).
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU="0,1" HOSTFLOOR=300 OCC_PIDS="613592 613593"
. /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.c14.sh
export GPU_POOL="0,1" DDP_TIMEOUT=1500 ASYM_ARENA_SHM_CAP_GB=285
python3 /workspace/AsymGEMM-SFT-38/.repair_dataset_info.py >> "$S" 2>&1 || true
echo "== STDTPS96-A3-MX-PHASEA begin (2r ceiling; prior 112-144K T1) $(date '+%m-%d %H:%M')" >> "$S"
MX=mixtral-8x22b; P="none|false|false|false|false|false"
declare -A RES

probe() { local s=$1 v
  [ -n "${RES[$s]:-}" ] && { [ "${RES[$s]}" = "TRAINED" ]; return; }
  v=$(run_cell nmxt1$((s/1000)) $MX "asym_sdp2_cpuadamwds|T1" $s "1" "$P" 2)
  if [ "$v" = "TRAINED" ]; then RES[$s]="TRAINED"; return 0; fi
  ooms "$v" || { echo "ABORT-MXA infra $v s=$s" >> "$S"; exit 1; }
  RES[$s]="OOM"; return 1
}

STEP=32000; S0=128000
if probe $S0; then
  lo=$S0; s=$((S0+STEP))
  while probe $s; do lo=$s; s=$((s+STEP)); done
  hi=$s
else
  hi=$S0; s=$((S0-STEP))
  until probe $s; do hi=$s; s=$((s-STEP)); if [ $s -lt 32000 ]; then echo "ABORT-MXA floor reached" >> "$S"; exit 1; fi; done
  lo=$s
fi
while [ $((hi-lo)) -gt 16000 ]; do
  mid=$(( (lo + (hi-lo)/2) / 16000 * 16000 )); [ $mid -le $lo ] && mid=$((lo+16000))
  if probe $mid; then lo=$mid; else hi=$mid; fi
done
echo "MX-2R-CAP $lo bracket=($lo,$hi] tier=T1 $(date '+%m-%d %H:%M')" >> "$S"
echo "$lo T1" > "$LOGD/mx_cap96.txt"
echo "== STDTPS96-A3-MX-PHASEA-DONE $(date '+%m-%d %H:%M')" >> "$S"
