#!/bin/bash
# stdtps96_a3_mxC2r.sh — Mixtral Phase C (2-RANK): fill 64-144K. Batch walks
# seeded ~half the 185G bests (64k b4->walk "3 2 1"; 80/96k "2 1"; 112k
# "2 1"; 128k b2-upgrade probe over the banked b1). Baselines walk to their
# 96G walls; fsdp2 = ONE 64k probe (185G verdict: host load-dead,
# seq-independent — confirm class, then row-OOM); megatron banked OOM (185G
# wall (8k,16k] HBM => strictly worse at 96G, no runs).
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU="0,1" HOSTFLOOR=300 OCC_PIDS="613592 613593"
. /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.c14.sh
export GPU_POOL="0,1" DDP_TIMEOUT=1500 ASYM_ARENA_SHM_CAP_GB=285
echo "== STDTPS96-A3-MX-PHASEC2R begin $(date '+%m-%d %H:%M')" >> "$S"
MX=mixtral-8x22b; P="none|false|false|false|false|false"
RC="superoffload_mem|recomp"; UN="superoffload_mem|unsloth-ohbm0"
UO="superoffload_mem|unsloth-off-ohbm0"; Z3="zero3_offload_mem|recomp"; FD="fsdp2_offload|recomp"
AS="asym_sdp2_cpuadamwds|T1"

walk_up() { local pfx="$1" tok="$2" bl="$3" v s; shift 3
  for s in "$@"; do
    v=$(run_cell ${pfx}$((s/1000)) $MX "$tok" $s "$bl" "$P" 2)
    [ "$v" = "TRAINED" ] || { ooms "$v" || echo "NOTE ${pfx}@$s infra $v" >> "$S"; echo "WALL ${pfx} at $s ($v) $(date +%H:%M)" >> "$S"; return; }
  done; echo "WALL ${pfx} none through $* $(date +%H:%M)" >> "$S"; }

walk_up nmx2rc "$RC" "2 1" 64000
walk_up nmx2rcup "$RC" "1" 80000 96000 2>/dev/null || true
walk_up nmx2z3 "$Z3" "2 1" 64000
walk_up nmx2z3up "$Z3" "1" 80000 96000 2>/dev/null || true
walk_up nmx2un "$UN" "2 1" 64000 80000 96000
walk_up nmx2unup "$UN" "1" 112000 128000
run_cell nmx2fd64 $MX "$FD" 64000 "1" "$P" 2 >/dev/null   # class-confirm probe
walk_up nmx2uo "$UO" "2 1" 64000 80000 96000
walk_up nmx2uoup "$UO" "1" 112000 128000 144000
run_cell nmx2t1_64  $MX "$AS" 64000  "3 2 1" "$P" 2 >/dev/null
run_cell nmx2t1_80  $MX "$AS" 80000  "2 1"   "$P" 2 >/dev/null
run_cell nmx2t1_96  $MX "$AS" 96000  "2 1"   "$P" 2 >/dev/null
run_cell nmx2t1_112 $MX "$AS" 112000 "2 1"   "$P" 2 >/dev/null
run_cell nmx2t1_128b $MX "$AS" 128000 "2"    "$P" 2 >/dev/null   # b2-upgrade probe over banked b1
echo "== STDTPS96-A3-MX-PHASEC2R-DONE $(date '+%m-%d %H:%M')" >> "$S"
