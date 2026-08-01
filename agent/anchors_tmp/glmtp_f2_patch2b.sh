#!/bin/bash
# F2 patch-2b (replaces patch2): the 2-rank asym legs lost to the ep2 detour —
# sdp2 walks at 192k, 160k, 128k. Gated on F2-PATCH-DONE; emits F2-PATCH2-DONE
# (the marker the Air chain waits for).
set -uo pipefail
export GPU="0,1" HOSTFLOOR=1300 GPU_POOL="0,1" TORCHINDUCTOR_COMPILE_THREADS=1
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
for i in $(seq 1 2880); do grep -q "F2-PATCH-DONE" "$S" 2>/dev/null && break; sleep 30; done
grep -q "F2-PATCH-DONE" "$S" || { echo "F2PATCH2B-ABORT $(date +%H:%M)" >> "$S"; exit 1; }
echo "F2-PATCH2B begin $(date +%H:%M)" >> "$S"
run_cell f2s_t1192 glm4.7-flash "asym_sdp2_cpuadamwds|T1" 192000 "4 2 1" "none|false|false|false|false|false" 2
run_cell f2s_t1160 glm4.7-flash "asym_sdp2_cpuadamwds|T1" 160000 "2 1"   "none|false|false|false|false|false" 2
run_cell f2s_t1128 glm4.7-flash "asym_sdp2_cpuadamwds|T1" 128000 "3 2"   "none|false|false|false|false|false" 2
echo "F2-PATCH2-DONE $(date +%H:%M)" >> "$S"
