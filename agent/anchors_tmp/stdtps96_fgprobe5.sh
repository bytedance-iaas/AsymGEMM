#!/bin/bash
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
POL="none|false|false|false|false|false"
echo "=== FG-PROBE5 (GPU2 solo, NO occupier) BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
# deliberately kill the GPU2 occupier for this one probe, then restore
p=$(cat "$(_occ_pidfile 2)" 2>/dev/null); [ -n "$p" ] && kill -TERM "$p" 2>/dev/null; sleep 5; rm -f "$(_occ_pidfile 2)"
occupiers_alive() { return 0; }   # bypass for this probe only
guard() { return 0; }
v=$(ONE_RANK_GPU=2 ARM_ENV="PYTHONFAULTHANDLER=1" run_cell s96h1t2b064g2n hunyuan-a13b "asym_cpuadamwds|T2B" 64000 "2" "$POL" 1)
echo "FG-PROBE5 1r T2B@64k b2 GPU2 NO-occupier -> $v (TRAINED => occupier interplay; segv => GPU2 outright)" >> "$S"
echo "=== FG-PROBE5 DONE $(date '+%F %H:%M:%S') ===" >> "$S"
