#!/bin/bash
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
echo "=== FG-PROBE10 BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
occupiers_alive() { return 0; }
guard() { return 0; }
POL="none|false|false|false|false|false"
HYOFF="ASYM_OFFLOAD_MODULES=routed_experts,shared_experts,attention,norms,mlp_dense"
# (a) exact d2sdp64 replica: 2r sdp2 on GPUs 0,1 (EP2 family forces offloads false)
SIM_GPUS_SAVE="$SIM_GPUS"; SIM_GPUS="0,1"
v=$(ARM_ENV="$HYOFF ASYM_ARENA_SHM_CAP_GB=320" run_cell s96h2t2b064r01 hunyuan-a13b "asym_sdp2_cpuadamwds|T2B" 64000 "2" "$POL" 2)
echo "FG-PROBE10a d2sdp64 replica (2r sdp2 0,1) -> $v" >> "$S"
SIM_GPUS="$SIM_GPUS_SAVE"
# (b) the 1r case with faulthandler (bug-report frame)
v=$(ONE_RANK_GPU=0 ARM_ENV="$HYOFF ASYM_CPU_ADAMW_GRAD_OFFLOAD=false ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false PYTHONFAULTHANDLER=1" run_cell s96h1t2b064ff hunyuan-a13b "asym_cpuadamwds|T2B" 64000 "2" "$POL" 1)
echo "FG-PROBE10b 1r + faulthandler -> $v" >> "$S"
echo "=== FG-PROBE10 DONE $(date '+%F %H:%M:%S') ===" >> "$S"
