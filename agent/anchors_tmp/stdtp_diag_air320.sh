#!/bin/bash
# stdtp_diag_air320.sh — DIAGNOSTIC, not a cell: re-probe the BANKED Air-2r
# T1@320k b1 (July GLMTP precedent: cap 240, TRAINED 989 tok/s @98%) under
# the July cap. TRAINED => fabric stack matches the banked era and the 448k
# bank growth is real; cap-exceeded => sdp2 fabric regression in the current
# tree (report, stop Air lane). W1+M1 (fit probe protocol).
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU="0,1" HOSTFLOOR=1200
. /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/stdtp_lib.sh
export GPU_POOL="0,1" DDP_TIMEOUT=1500 ASYM_ARENA_SHM_CAP_GB=240 MAX_STEPS=1
echo "== STDTP-DIAG-AIR320 begin (banked-cell reproduction probe) $(date '+%m-%d %H:%M')" >> "$S"
v=$(run_cell d2t1320 glm4.5-air "asym_sdp2_cpuadamwds|T1" 320000 "1" "none|false|false|false|false|false" 2)
echo "== STDTP-DIAG-AIR320 verdict: $v $(date '+%m-%d %H:%M')" >> "$S"
