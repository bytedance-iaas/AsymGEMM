#!/bin/bash
# stdz_job1_2r.sh — TP x-axis standardization, Job 1 rank-2 cells
# (standardize_tps.md): 30B grid 384-1024K (fills 512/768/896/1024),
# 122B grid 160-320K (fills 160/224). Runs INSIDE the container on a GPU
# PAIR (inside ids 0,1). 2r backend = asym_sdp2_cpuadamwds (the banked rows'
# backend; sepplan mirror ledger equates them within ~1%). 122B sEP arena 345
# per fix_plot_placeholders §5/§6; mid-curve cells plain (no grad-offload),
# CHAIN-V neighbor-consistent convention. zero3-2r stays derived-est for
# 122B (row convention — no measured cells in that row); rc/zero3-2r 30B
# dead ≥512K free (wall (384,392]); uns walls (640,660] 30B / (288,320] 122B.
set -u
export GPU="0,1" HOSTFLOOR=1200
source /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/fig12_lib.sh
export GPU_POOL="0,1" DDP_TIMEOUT=1500
S="$LOGD/stdz_status.log"
POL="none|false|false|false|false|false"
UNS="superoffload_mem|unsloth"
RC="superoffload_mem|recomp"
F2="fsdp2_offload|recomp"
SDP="asym_sdp2_cpuadamwds"
echo "== stdz job1 2r start $(date +%F_%H:%M) ==" >> "$S"

# ---- 122B rank-2 fills (arena 345)
ARM_ENV="" run_cell s2u122_160 q3.5-122b-a10b "$UNS" 160000 "2 1" "$POL" 2
ARM_ENV="ASYM_ARENA_SHM_CAP_GB=345" run_cell s2a122_160 q3.5-122b-a10b "$SDP|T1" 160000 "2 1" "$POL" 2
ARM_ENV="" run_cell s2u122_224 q3.5-122b-a10b "$UNS" 224000 "1" "$POL" 2
ARM_ENV="ASYM_ARENA_SHM_CAP_GB=345" run_cell s2a122_224 q3.5-122b-a10b "$SDP|T1" 224000 "1" "$POL" 2
ARM_ENV="" run_cell s2r122_160 q3.5-122b-a10b "$RC" 160000 "1" "$POL" 2
ARM_ENV="" run_cell s2r122_224 q3.5-122b-a10b "$RC" 224000 "1" "$POL" 2

# ---- 30B rank-2 fills
ARM_ENV="" run_cell s2u30_512 q3-30b-a3b "$UNS" 512000 "1" "$POL" 2
ARM_ENV="" run_cell s2a30_512 q3-30b-a3b "$SDP|T2" 512000 "1" "$POL" 2
ARM_ENV="" run_cell s2f30_512 q3-30b-a3b "$F2" 512000 "1" "$POL" 2
ARM_ENV="" run_cell s2a30_768 q3-30b-a3b "$SDP|T2" 768000 "1" "$POL" 2
ARM_ENV="" run_cell s2a30_896 q3-30b-a3b "$SDP|T2" 896000 "1" "$POL" 2
ARM_ENV="" run_cell s2a30_1024 q3-30b-a3b "$SDP|T2" 1024000 "1" "$POL" 2
echo "== stdz job1 2r DONE $(date +%F_%H:%M) ==" >> "$S"
