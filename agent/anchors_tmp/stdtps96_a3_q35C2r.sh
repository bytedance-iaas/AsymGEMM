#!/bin/bash
# stdtps96_a3_q35C2r.sh — Agent 3 / Phase C (2-RANK first): fill the 35B
# 192-512K grid. Walk-ups pin walls inside the 96G world (185G walls do not
# transfer). Reused (resv<=92G rule, banked at fill time, no runs): uo@256k
# (1498, 46.3G) + uo@384k (1706, 74G). Asym 320-512K = Phase-A cells.
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU="0,1" HOSTFLOOR=300 OCC_PIDS="613592 613593"
. /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.c14.sh
export GPU_POOL="0,1" DDP_TIMEOUT=1500
echo "== STDTPS96-A3-Q35-PHASEC2R begin $(date '+%m-%d %H:%M')" >> "$S"
Q35=q3.5-35b-a3b; P="none|false|false|false|false|false"; BE=asym_sepplan2_cpuadamwds
RC="superoffload_mem|recomp"; UN="superoffload_mem|unsloth-ohbm0"
UO="superoffload_mem|unsloth-off-ohbm0"; Z3="zero3_offload_mem|recomp"; FD="fsdp2_offload|recomp"

walk_up() { local pfx="$1" tok="$2" v s; shift 2
  for s in "$@"; do
    v=$(run_cell ${pfx}$((s/1000)) $Q35 "$tok" $s "1" "$P" 2)
    [ "$v" = "TRAINED" ] || { ooms "$v" || echo "NOTE ${pfx}@$s infra $v" >> "$S"; echo "WALL ${pfx} at $s ($v) $(date +%H:%M)" >> "$S"; return; }
  done; echo "WALL ${pfx} none through $* $(date +%H:%M)" >> "$S"; }

walk_up n35c2rc "$RC" 192000 256000
walk_up n35c2z3 "$Z3" 192000 256000
walk_up n35c2un "$UN" 192000 256000 320000 384000
walk_up n35c2fd "$FD" 192000 256000 320000 384000
run_cell n35c2uo192 $Q35 "$UO" 192000 "1" "$P" 2 >/dev/null
walk_up n35c2uo "$UO" 320000 448000 512000
run_cell n35c2t2_256 $Q35 "${BE}|T2" 256000 "1" "$P" 2 >/dev/null
run_cell n35c2t2_192 $Q35 "${BE}|T2" 192000 "1" "$P" 2 >/dev/null
echo "== STDTPS96-A3-Q35-PHASEC2R-DONE $(date '+%m-%d %H:%M')" >> "$S"
