#!/bin/bash
# A2 v2: Air 2-rank with wall-informed blists (1-rank fits + ZeRO-3 headroom
# ~= +1 batch step for rc/uns; uo host-bound so flat) and DDP_TIMEOUT=1500
# (caps the near-wall allgather-hang waste at 25 min; real slow steps are
# ~7-12 min at these rungs). Fresh a2b_ tags. Runs NOW (field cleared).
set -uo pipefail
export GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" TORCHINDUCTOR_COMPILE_THREADS=1 DDP_TIMEOUT=1500
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
echo "A2B begin $(date +%H:%M)" >> "$S"
R() { # R TAG SYSTOK SEQ BLIST
  run_cell "$1" glm4.5-air "$2" "$3" "$4" "none|false|false|false|false|false" 2
}
# 128k
R a2b_rc128 "superoffload_mem|recomp"            128000 "2 1"
R a2b_un128 "superoffload_mem|unsloth"           128000 "2 1"
R a2b_uo128 "superoffload_mem|unsloth-off-ohbm0" 128000 "2 1"
R a2b_t1128 "asym_sdp2_cpuadamwds|T1"            128000 "3 2"
# 96k
R a2b_rc96 "superoffload_mem|recomp"            96000 "2 1"
R a2b_un96 "superoffload_mem|unsloth"           96000 "3 2"
R a2b_uo96 "superoffload_mem|unsloth-off-ohbm0" 96000 "2"
R a2b_t196 "asym_sdp2_cpuadamwds|T1"            96000 "4 3"
# 64k
R a2b_rc64 "superoffload_mem|recomp"            64000 "3 2"
R a2b_un64 "superoffload_mem|unsloth"           64000 "4 3"
R a2b_uo64 "superoffload_mem|unsloth-off-ohbm0" 64000 "4"
R a2b_t164 "asym_sdp2_cpuadamwds|T1"            64000 "6 4"
# 48k
R a2b_rc48 "superoffload_mem|recomp"            48000 "4 2"
R a2b_un48 "superoffload_mem|unsloth"           48000 "6 4"
R a2b_uo48 "superoffload_mem|unsloth-off-ohbm0" 48000 "4"
R a2b_t148 "asym_sdp2_cpuadamwds|T1"            48000 "8 6"
# 32k
R a2b_rc32 "superoffload_mem|recomp"            32000 "6 4"
R a2b_un32 "superoffload_mem|unsloth"           32000 "8 6"
R a2b_uo32 "superoffload_mem|unsloth-off-ohbm0" 32000 "8"
R a2b_t132 "asym_sdp2_cpuadamwds|T1"            32000 "12 8"
# 16k
R a2b_rc16 "superoffload_mem|recomp"            16000 "12 8"
R a2b_un16 "superoffload_mem|unsloth"           16000 "16 12"
R a2b_uo16 "superoffload_mem|unsloth-off-ohbm0" 16000 "16"
R a2b_t116 "asym_sdp2_cpuadamwds|T1"            16000 "16"
echo "A2-DONE GLMTP-ALL-DONE $(date +%H:%M)" >> "$S"
