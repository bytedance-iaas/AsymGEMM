#!/bin/bash
# stdtps96_a3_q35A2.sh — Phase A continuation for 35B after the T2B@512k FAIL
# (hang+watchdog, truncated rank0 traceback in wrapped_training_step).
# State seeded: T2 448k TRAINED (93.5G/98%), T2 512k GOOM -> T2 wall (448,512];
# T2B@512k FAIL x1. Plan: retry T2B@512k once; TRAINED -> walk UP on T2B
# (+64k) then bisect; FAIL/GOOM -> bisect (448,512] with T2->T2B ladder.
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU="0,1" HOSTFLOOR=300 OCC_PIDS="613592 613593"
. /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.c14.sh
export GPU_POOL="0,1" DDP_TIMEOUT=1500
echo "== STDTPS96-A3-Q35-PHASEA2 begin $(date '+%m-%d %H:%M')" >> "$S"
Q35=q3.5-35b-a3b; P="none|false|false|false|false|false"; BE=asym_sepplan2_cpuadamwds
CAPTIER=T2; lo=448000; hi=512000

probe_t2b() { local s=$1 v
  v=$(run_cell n35t2b$((s/1000))r $Q35 "${BE}|T2B" $s "1" "$P" 2)
  [ "$v" = "TRAINED" ] && return 0
  ooms "$v" && return 1
  echo "NOTE T2B@$s verdict $v (non-OOM) — treated as non-fitting for the cap, caveat in ledger" >> "$S"; return 1
}
probe_pair() { local s=$1 v   # T2 (inside its bracket) then T2B
  v=$(run_cell n35t2$((s/1000))r $Q35 "${BE}|T2" $s "1" "$P" 2)
  if [ "$v" = "TRAINED" ]; then CAPTIER=T2; return 0; fi
  ooms "$v" || echo "NOTE T2@$s verdict $v (non-OOM)" >> "$S"
  if probe_t2b $s; then CAPTIER=T2B; return 0; fi
  return 1
}

if probe_t2b 512000; then
  CAPTIER=T2B; lo=512000; s=576000
  while probe_t2b $s; do lo=$s; s=$((s+64000)); done
  hi=$s
  while [ $((hi-lo)) -gt 16000 ]; do
    mid=$(( (lo + (hi-lo)/2) / 16000 * 16000 )); [ $mid -le $lo ] && mid=$((lo+16000))
    if probe_t2b $mid; then lo=$mid; else hi=$mid; fi
  done
else
  while [ $((hi-lo)) -gt 16000 ]; do
    mid=$(( (lo + (hi-lo)/2) / 16000 * 16000 )); [ $mid -le $lo ] && mid=$((lo+16000))
    if probe_pair $mid; then lo=$mid; else hi=$mid; fi
  done
fi
echo "Q35-2R-CAP $lo bracket=($lo,$hi] tier=$CAPTIER $(date '+%m-%d %H:%M')" >> "$S"
echo "$lo $CAPTIER" > "$LOGD/q35_cap96.txt"
echo "== STDTPS96-A3-Q35-PHASEA2-DONE $(date '+%m-%d %H:%M')" >> "$S"
