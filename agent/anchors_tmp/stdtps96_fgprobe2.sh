#!/bin/bash
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
POL="none|false|false|false|false|false"
start_occupiers || exit 1
echo "=== FG-PROBE2 (faulthandler) BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
v=$(ARM_ENV="ASYM_ARENA_SHM_CAP_GB=320 PYTHONFAULTHANDLER=1 CUDA_LAUNCH_BLOCKING=1" run_cell s96h2t2b064f hunyuan-a13b "asym_sdp2_cpuadamwds|T2B" 64000 "2" "$POL" 2)
echo "FG-PROBE2 T2B@64k b2 on 0,2 with faulthandler -> $v" >> "$S"
echo "=== FG-PROBE2 DONE $(date '+%F %H:%M:%S') ===" >> "$S"
