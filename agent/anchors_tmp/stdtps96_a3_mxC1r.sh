#!/bin/bash
# stdtps96_a3_mxC1r.sh — Mixtral gap probe + 1-RANK column (rungs 64-144K).
# Order: un@144k 2r GAP (sole-asym decider at the cap) -> 1r cap confirm
# (T1@144k) -> baselines walks -> asym walks. fd = one 64k class probe
# (185G host-load-dead precedent). Serial, one 96G occupier per GPU.
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU="0,1" HOSTFLOOR=300 OCC_PIDS="613592 613593"
. /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.c14.sh
export GPU_POOL="0,1" DDP_TIMEOUT=1500 ASYM_ARENA_SHM_CAP_GB=285
echo "== STDTPS96-A3-MX-GAP+C1R begin $(date '+%m-%d %H:%M')" >> "$S"
MX=mixtral-8x22b; P="none|false|false|false|false|false"
RC="superoffload_mem|recomp"; UN="superoffload_mem|unsloth-ohbm0"
UO="superoffload_mem|unsloth-off-ohbm0"; Z3="zero3_offload_mem|recomp"; FD="fsdp2_offload|recomp"

run_cell nmx2unup144 $MX "$UN" 144000 "1" "$P" 2 >/dev/null    # THE gap probe (2r)

# ---- switch to 1r on sim GPU inside-0 (phys 0) ----
export GPU="0" GPU_POOL="0" CUDA_VISIBLE_DEVICES=0
unset DDP_TIMEOUT ASYM_ARENA_SHM_CAP_GB || true
AS="asym_cpuadamwds|T1"

v=$(run_cell nmx1t1_144 $MX "$AS" 144000 "1" "$P" 1)
if [ "$v" != "TRAINED" ]; then
  echo "ANOMALY: mixtral 1r cannot hold the 2r cap 144K ($v) — cap must shrink; STOPPING" >> "$S"
  exit 1
fi
walk_up() { local pfx="$1" tok="$2" bl="$3" v s; shift 3
  for s in "$@"; do
    v=$(run_cell ${pfx}$((s/1000)) $MX "$tok" $s "$bl" "$P" 1)
    [ "$v" = "TRAINED" ] || { ooms "$v" || echo "NOTE ${pfx}@$s infra $v" >> "$S"; echo "WALL ${pfx} at $s ($v) $(date +%H:%M)" >> "$S"; return; }
  done; echo "WALL ${pfx} none through $* $(date +%H:%M)" >> "$S"; }

walk_up nmx1rc "$RC" "2 1" 64000
walk_up nmx1rcu "$RC" "1" 80000
walk_up nmx1z3 "$Z3" "2 1" 64000
walk_up nmx1z3u "$Z3" "1" 80000
walk_up nmx1un "$UN" "2 1" 64000 80000 96000
walk_up nmx1unu "$UN" "1" 112000 128000 144000
run_cell nmx1fd64 $MX "$FD" 64000 "1" "$P" 1 >/dev/null   # class probe
walk_up nmx1uo "$UO" "2 1" 64000 80000 96000
walk_up nmx1uou "$UO" "1" 112000 128000 144000
run_cell nmx1t1_64  $MX "$AS" 64000  "3 2 1" "$P" 1 >/dev/null
run_cell nmx1t1_80  $MX "$AS" 80000  "2 1"   "$P" 1 >/dev/null
run_cell nmx1t1_96  $MX "$AS" 96000  "2 1"   "$P" 1 >/dev/null
run_cell nmx1t1_112 $MX "$AS" 112000 "2 1"   "$P" 1 >/dev/null
run_cell nmx1t1_128 $MX "$AS" 128000 "2 1"   "$P" 1 >/dev/null
echo "== STDTPS96-A3-MX-C1R-DONE $(date '+%m-%d %H:%M')" >> "$S"
