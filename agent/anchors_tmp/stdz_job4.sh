#!/bin/bash
# stdz_job4.sh — TP x-axis standardization, Job 4: Hunyuan-A13B (grid
# 160/192/224/256/288/320K). gpt-oss-20B is render-only (all 12 grid cells
# measured; MAIN_RUNGS already selects them) — no runs here.
# Launch with stdz_launch.sh on a 2-GPU set. Cell lists (standardize_tps.md):
#  HY-1r: new cols 160/224/288 -> asym T1 x3; uns@160/224 tok/s (fits, wall
#         (256,320]); uns@288 = the extra-sole PROBE; rc@160 tok/s (fits,
#         wall (192,256]); rc@224 probe (inside bracket).
#  HY-2r: new cols 160/224 -> asym sdp2-T1 x2; uns x2 (wall (256,288]);
#         rc@160 tok/s (fits 192), rc@224 probe (bracket (192,256]).
set -u
export GPU=0 HOSTFLOOR=500
source /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/fig12_lib.sh
S="$LOGD/stdz_status.log"
POL="none|false|false|false|false|false"
UNS="superoffload_mem|unsloth"
RC="superoffload_mem|recomp"
SDP="asym_sdp2_cpuadamwds"
echo "== stdz job4 (hunyuan) 1r start $(date +%F_%H:%M) ==" >> "$S"

ARM_ENV="" run_cell s1uhy_160 hunyuan-a13b "$UNS" 160000 "1" "$POL" 1
ARM_ENV="" run_cell s1ahy_160 hunyuan-a13b "asym_cpuadamwds|T1" 160000 "1" "$POL" 1
ARM_ENV="" run_cell s1rhy_160 hunyuan-a13b "$RC" 160000 "1" "$POL" 1
ARM_ENV="" run_cell s1uhy_224 hunyuan-a13b "$UNS" 224000 "1" "$POL" 1
ARM_ENV="" run_cell s1ahy_224 hunyuan-a13b "asym_cpuadamwds|T1" 224000 "1" "$POL" 1
ARM_ENV="" run_cell s1rhy_224 hunyuan-a13b "$RC" 224000 "1" "$POL" 1
ARM_ENV="" run_cell s1uhy_288 hunyuan-a13b "$UNS" 288000 "1" "$POL" 1   # extra-sole probe
ARM_ENV="" run_cell s1ahy_288 hunyuan-a13b "asym_cpuadamwds|T1" 288000 "1" "$POL" 1

echo "== stdz job4 2r start $(date +%F_%H:%M) ==" >> "$S"
export GPU="0,1" CUDA_VISIBLE_DEVICES="0,1" GPU_POOL="0,1" DDP_TIMEOUT=1500 HOSTFLOOR=1100
ARM_ENV="" run_cell s2uhy_160 hunyuan-a13b "$UNS" 160000 "1" "$POL" 2
ARM_ENV="" run_cell s2ahy_160 hunyuan-a13b "$SDP|T1" 160000 "1" "$POL" 2
ARM_ENV="" run_cell s2rhy_160 hunyuan-a13b "$RC" 160000 "1" "$POL" 2
ARM_ENV="" run_cell s2uhy_224 hunyuan-a13b "$UNS" 224000 "1" "$POL" 2
ARM_ENV="" run_cell s2ahy_224 hunyuan-a13b "$SDP|T1" 224000 "1" "$POL" 2
ARM_ENV="" run_cell s2rhy_224 hunyuan-a13b "$RC" 224000 "1" "$POL" 2
echo "== stdz job4 DONE $(date +%F_%H:%M) ==" >> "$S"
