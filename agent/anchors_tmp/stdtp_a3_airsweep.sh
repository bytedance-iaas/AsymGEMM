#!/bin/bash
# stdtp_a3_airsweep.sh — (1) g0t3r448: Session B's flag-closure probe — the
# HISTORICAL raw-token GLM T3 (glmext_rev pattern, hand token = NO tier env)
# @448k 2r cap 450. Expected COOM (same fg fabric bank+fg). If TRAINED the
# gate REOPENS -> stop for replan. (2) The Air 160-320K sweep (the unique
# 6-rung 32K-step grid ending at the 2r max): 2r un224-probe/as224/as288,
# then 1r un224-probe/uo224/uo288-probe/as224/as288. NVD=1,3 -> inside 0,1;
# 1r block runs on inside-1 = phys GPU 3 (user assignment).
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU="0,1" HOSTFLOOR=1200
. /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/stdtp_lib.sh
export GPU_POOL="0,1" DDP_TIMEOUT=1500 ASYM_ARENA_SHM_CAP_GB=450
python3 /workspace/AsymGEMM-SFT-38/.repair_dataset_info.py >> "$S" 2>&1 || true
echo "== STDTP-A3-AIRSWEEP begin $(date '+%m-%d %H:%M')" >> "$S"
P="none|false|false|false|false|false"
RAWT3="asym_sdp2_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"

v=$(run_cell g0t3r448 glm4.5-air "$RAWT3" 448000 "1" "$P" 2)
if [ "$v" = "TRAINED" ]; then
  echo "G0-TRAINED — RAW-T3 CARRIES 448k: GATE REOPENS, STOPPING FOR REPLAN" >> "$S"
  exit 0
fi
echo "G0 raw-T3@448k verdict: $v (flag closed if OOM-class) $(date +%H:%M)" >> "$S"

# ---- 2r sweep cells (GPUs 0,1 = phys 1,3) ----
run_cell g2un224 glm4.5-air "superoffload_mem|unsloth" 224000 "1"   "$P" 2 >/dev/null
run_cell g2as224 glm4.5-air "asym_sdp2_cpuadamwds|T1"  224000 "2 1" "$P" 2 >/dev/null
run_cell g2as288 glm4.5-air "asym_sdp2_cpuadamwds|T1"  288000 "1"   "$P" 2 >/dev/null

# ---- 1r sweep cells on inside-GPU 1 (= phys 3) ----
export GPU="1" CUDA_VISIBLE_DEVICES=1 GPU_POOL=1 HOSTFLOOR=600
unset DDP_TIMEOUT ASYM_ARENA_SHM_CAP_GB || true
run_cell g1un224 glm4.5-air "superoffload_mem|unsloth"           224000 "1" "$P" 1 >/dev/null
run_cell g1uo224 glm4.5-air "superoffload_mem|unsloth-off-ohbm0" 224000 "1" "$P" 1 >/dev/null
run_cell g1uo288 glm4.5-air "superoffload_mem|unsloth-off-ohbm0" 288000 "1" "$P" 1 >/dev/null
run_cell g1as224 glm4.5-air "asym_cpuadamwds|T1"                 224000 "1" "$P" 1 >/dev/null
run_cell g1as288 glm4.5-air "asym_cpuadamwds|T1"                 288000 "1" "$P" 1 >/dev/null
echo "== STDTP-A3-AIRSWEEP-DONE $(date '+%m-%d %H:%M')" >> "$S"
