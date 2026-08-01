#!/bin/bash
# A2 v3: remaining Air 2-rank matrix. Adds ASYM_ARENA_SHM_CAP_GB=240 (Air
# banks ~200 GB > 160 default; /dev/shm 479G) + rc128 b1 (lost to FAIL-break).
# Fresh a2c_ tags where rerun needed; DDP_TIMEOUT=1500 kept.
set -uo pipefail
export GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" TORCHINDUCTOR_COMPILE_THREADS=1 DDP_TIMEOUT=1500
export ASYM_ARENA_SHM_CAP_GB=240
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
echo "A2C begin $(date +%H:%M)" >> "$S"
R() { run_cell "$1" glm4.5-air "$2" "$3" "$4" "none|false|false|false|false|false" 2; }
# 128k completions
R a2c_rc128 "superoffload_mem|recomp" 128000 "1"
R a2c_t1128 "asym_sdp2_cpuadamwds|T1" 128000 "3 2"
# 96k
R a2b_rc96 "superoffload_mem|recomp"            96000 "2 1"
R a2b_un96 "superoffload_mem|unsloth"           96000 "3 2"
R a2b_uo96 "superoffload_mem|unsloth-off-ohbm0" 96000 "2 1"
R a2b_t196 "asym_sdp2_cpuadamwds|T1"            96000 "4 3"
# 64k
R a2b_rc64 "superoffload_mem|recomp"            64000 "3 2"
R a2b_un64 "superoffload_mem|unsloth"           64000 "4 3"
R a2b_uo64 "superoffload_mem|unsloth-off-ohbm0" 64000 "4 2"
R a2b_t164 "asym_sdp2_cpuadamwds|T1"            64000 "6 4"
# 48k
R a2b_rc48 "superoffload_mem|recomp"            48000 "4 2"
R a2b_un48 "superoffload_mem|unsloth"           48000 "6 4"
R a2b_uo48 "superoffload_mem|unsloth-off-ohbm0" 48000 "4 2"
R a2b_t148 "asym_sdp2_cpuadamwds|T1"            48000 "8 6"
# 32k
R a2b_rc32 "superoffload_mem|recomp"            32000 "6 4"
R a2b_un32 "superoffload_mem|unsloth"           32000 "8 6"
R a2b_uo32 "superoffload_mem|unsloth-off-ohbm0" 32000 "8 4"
R a2b_t132 "asym_sdp2_cpuadamwds|T1"            32000 "12 8"
# 16k
R a2b_rc16 "superoffload_mem|recomp"            16000 "12 8"
R a2b_un16 "superoffload_mem|unsloth"           16000 "16 12"
R a2b_uo16 "superoffload_mem|unsloth-off-ohbm0" 16000 "16 8"
R a2b_t116 "asym_sdp2_cpuadamwds|T1"            16000 "16"
echo "A2-DONE GLMTP-ALL-DONE $(date +%H:%M)" >> "$S"
