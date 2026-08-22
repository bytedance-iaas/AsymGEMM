#!/bin/bash
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
echo "=== FG-PROBE6 (GPU0 solo, NO occupier — tree-regression test) BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
p=$(cat "$(_occ_pidfile 0)" 2>/dev/null); [ -n "$p" ] && kill -TERM "$p" 2>/dev/null; sleep 5; rm -f "$(_occ_pidfile 0)"
occupiers_alive() { return 0; }
guard() { return 0; }
POL="none|false|false|false|false|false"
v=$(ONE_RANK_GPU=0 ARM_ENV="PYTHONFAULTHANDLER=1" run_cell s96h1t2b064g0n hunyuan-a13b "asym_cpuadamwds|T2B" 64000 "2" "$POL" 1)
echo "FG-PROBE6 1r T2B@64k b2 GPU0 NO-occupier -> $v (segv => -46 TREE regression for hunyuan fg)" >> "$S"
echo "=== FG-PROBE6 DONE $(date '+%F %H:%M:%S') ===" >> "$S"
