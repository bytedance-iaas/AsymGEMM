#!/bin/bash
# stdtps96_a3_q35C1r.sh — Agent 3 / Phase C (1-RANK, after the 2r column):
# 35B 1r cells on the SAME rungs 192-512K under one 96G occupier (sim GPU =
# phys 0 via NVD="0"). First: the Phase-A 1r cap confirm @512K (ladder
# T1->T2->T2B; if nothing fits, the cap shrinks -> record anomaly, stop).
# Then baselines walk-up (96G walls), asym per-rung ladder last.
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU="0" HOSTFLOOR=300 OCC_PIDS="613592 613593"
. /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.c14.sh
export GPU_POOL="0"; unset DDP_TIMEOUT ASYM_ARENA_SHM_CAP_GB || true
echo "== STDTPS96-A3-Q35-PHASEC1R begin $(date '+%m-%d %H:%M')" >> "$S"
Q35=q3.5-35b-a3b; P="none|false|false|false|false|false"; BE=asym_cpuadamwds
RC="superoffload_mem|recomp"; UN="superoffload_mem|unsloth-ohbm0"
UO="superoffload_mem|unsloth-off-ohbm0"; Z3="zero3_offload_mem|recomp"; FD="fsdp2_offload|recomp"

aslad() { local pfx="$1" s="$2" blist="$3" v   # asym ladder T1->T2->T2B at seq s
  v=$(run_cell ${pfx}t1_$((s/1000)) $Q35 "${BE}|T1" $s "$blist" "$P" 1)
  [ "$v" = "TRAINED" ] && { echo T1; return 0; }
  ooms "$v" || { echo "NOTE ${pfx}T1@$s infra $v" >> "$S"; }
  v=$(run_cell ${pfx}t2_$((s/1000)) $Q35 "${BE}|T2" $s "$blist" "$P" 1)
  [ "$v" = "TRAINED" ] && { echo T2; return 0; }
  ooms "$v" || echo "NOTE ${pfx}T2@$s infra $v" >> "$S"
  v=$(run_cell ${pfx}t2b_$((s/1000)) $Q35 "${BE}|T2B" $s "$blist" "$P" 1)
  [ "$v" = "TRAINED" ] && { echo T2B; return 0; }
  echo NONE; return 1
}
walk_up() { local pfx="$1" tok="$2" v s; shift 2
  for s in "$@"; do
    v=$(run_cell ${pfx}$((s/1000)) $Q35 "$tok" $s "1" "$P" 1)
    [ "$v" = "TRAINED" ] || { ooms "$v" || echo "NOTE ${pfx}@$s infra $v" >> "$S"; echo "WALL ${pfx} at $s ($v) $(date +%H:%M)" >> "$S"; return; }
  done; echo "WALL ${pfx} none through $* $(date +%H:%M)" >> "$S"; }

# --- 1r cap confirm @512K (Phase A tail) ---
cap=$(aslad n35c1cap 512000 "1") || true
echo "Q35-1R-CAP-CONFIRM @512K -> ${cap} $(date +%H:%M)" >> "$S"
if [ "$cap" = "NONE" ]; then
  echo "ANOMALY: 1r cannot hold the 2r cap 512K — cap must shrink; STOPPING for re-derive" >> "$S"
  exit 1
fi

# --- baselines, walk-ups (96G walls pinned inside the grid) ---
walk_up n35c1rc "$RC" 192000 256000
walk_up n35c1z3 "$Z3" 192000 256000
walk_up n35c1un "$UN" 192000 256000 320000 384000 448000
walk_up n35c1fd "$FD" 192000 256000 320000 384000
walk_up n35c1uo "$UO" 192000 256000 320000 384000 448000

# --- asym per-rung (fastest fitting tier; 192K batch walk "2 1") ---
aslad n35c1 448000 "1" >/dev/null || true
aslad n35c1 384000 "1" >/dev/null || true
aslad n35c1 320000 "1" >/dev/null || true
aslad n35c1 256000 "1" >/dev/null || true
aslad n35c1 192000 "2 1" >/dev/null || true
echo "== STDTPS96-A3-Q35-PHASEC1R-DONE $(date '+%m-%d %H:%M')" >> "$S"
