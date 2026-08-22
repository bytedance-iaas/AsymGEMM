#!/bin/bash
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
POL="none|false|false|false|false|false"
start_occupiers || exit 1
echo "=== FG-PROBE4 (solo 1r fg on GPU2) BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
v=$(ONE_RANK_GPU=2 ARM_ENV="PYTHONFAULTHANDLER=1" run_cell s96h1t2b064g2 hunyuan-a13b "asym_cpuadamwds|T2B" 64000 "2" "$POL" 1)
echo "FG-PROBE4 1r T2B@64k b2 SOLO on GPU2 -> $v (segv => GPU2+cpu-left broken outright)" >> "$S"
if [ "$v" != "TRAINED" ]; then
  v2=$(ONE_RANK_GPU=0 ARM_ENV="PYTHONFAULTHANDLER=1" run_cell s96h1t2b064g0 hunyuan-a13b "asym_cpuadamwds|T2B" 64000 "2" "$POL" 1)
  echo "FG-PROBE4 control: same cell SOLO on GPU0 -> $v2" >> "$S"
fi
echo "=== FG-PROBE4 DONE $(date '+%F %H:%M:%S') ===" >> "$S"
