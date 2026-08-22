#!/bin/bash
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
POL="none|false|false|false|false|false"
start_occupiers || exit 1
echo "=== FG-CROSS-SOCKET PROBE BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
v=$(ARM_ENV="ASYM_ARENA_SHM_CAP_GB=320" run_cell s96h2t2b064 hunyuan-a13b "asym_sdp2_cpuadamwds|T2B" 64000 "2" "$POL" 2)
echo "FG-PROBE T2B@64k b2 on 0,2 (185G ran 59.3G on 0,1) -> $v" >> "$S"
echo "=== FG-CROSS-SOCKET PROBE DONE $(date '+%F %H:%M:%S') ===" >> "$S"
