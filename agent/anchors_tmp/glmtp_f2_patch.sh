#!/bin/bash
# F2 patch: cells lost to the NCCL-timeout + guard-orphan incident.
# rc 2-rank b1 (with ddp_timeout=7200 the slow step now completes) and the
# un 2-rank b1 OOM confirmation. Gated on F1-REDO-DONE.
set -uo pipefail
export GPU="0,1" HOSTFLOOR=1300 GPU_POOL="0,1" TORCHINDUCTOR_COMPILE_THREADS=1
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
for i in $(seq 1 2880); do grep -q "F1-REDO-DONE" "$S" 2>/dev/null && break; sleep 30; done
grep -q "F1-REDO-DONE" "$S" || { echo "F2PATCH-ABORT $(date +%H:%M)" >> "$S"; exit 1; }
echo "F2-PATCH begin $(date +%H:%M)" >> "$S"
run_cell f2s_rc192 glm4.7-flash "superoffload_mem|recomp"  192000 "1" "none|false|false|false|false|false" 2
run_cell f2s_un192 glm4.7-flash "superoffload_mem|unsloth" 192000 "1" "none|false|false|false|false|false" 2
run_cell f2s_t1192 glm4.7-flash "asym_ep2_cpuadamwds|T1" 192000 "4 2 1" "none|false|false|false|false|false" 2
echo "F2-PATCH-DONE $(date +%H:%M)" >> "$S"
