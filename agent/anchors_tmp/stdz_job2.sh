#!/bin/bash
# stdz_job2.sh — TP x-axis standardization, Job 2: Qwen3.5-35B-A3B (grid
# 256/384/512/640/768/896K) + Mixtral-8x22B (grid 128/160/192/224/256/288K).
# Launch with stdz_launch.sh on a 2-GPU set (inside ids 0,1); 1r section uses
# inside GPU 0, 2r section uses 0,1. Cell lists derived in standardize_tps.md
# from the DATA banks:
#  35B-1r: only asym@640/768 missing (baseline walls measured: rc/z3 (256,384],
#          fsdp2 (384,512], uns (512,576] -> all new columns OOM free).
#  35B-2r: only asym@768 missing (uns-2r wall (512,576]; others est/measured).
#  MX-1r : new cols 160/224/288 -> asym x3, uns@160/224 tok/s, uns@288 = THE
#          sole-rung GATE (bracket (256,320]), rc@160 probe (bracket (128,192]).
#  MX-2r : new cols 160/224 -> asym x2 (sdp2-T1), uns x2 (wall (256,288]),
#          rc@160 + zero3@160 probes (bracket (128,192]); 224 beyond-wall free.
set -u
export GPU=0 HOSTFLOOR=500
source /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/fig12_lib.sh
S="$LOGD/stdz_status.log"
POL="none|false|false|false|false|false"
UNS="superoffload_mem|unsloth"
RC="superoffload_mem|recomp"
Z3="zero3_offload_mem|recomp"
SDP="asym_sdp2_cpuadamwds"
echo "== stdz job2 (35B+mixtral) 1r start $(date +%F_%H:%M) ==" >> "$S"

# ---- 35B rank-1: NO RUNS — 640k/768k columns filled 2026-08-20 from the
# banked c18 depth-ladder cells (fix_qwen3.5_tp chain-3/4) by the peer
# session's STDTPS edit; only the 2r 768k cell below remains.

# ---- Mixtral rank-1
ARM_ENV="" run_cell s1umx_160 mixtral-8x22b "$UNS" 160000 "1" "$POL" 1
ARM_ENV="" run_cell s1amx_160 mixtral-8x22b "asym_cpuadamwds|T1" 160000 "1" "$POL" 1
ARM_ENV="" run_cell s1rmx_160 mixtral-8x22b "$RC" 160000 "1" "$POL" 1
ARM_ENV="" run_cell s1umx_224 mixtral-8x22b "$UNS" 224000 "1" "$POL" 1
ARM_ENV="" run_cell s1amx_224 mixtral-8x22b "asym_cpuadamwds|T1" 224000 "1" "$POL" 1
ARM_ENV="" run_cell s1umx_288 mixtral-8x22b "$UNS" 288000 "1" "$POL" 1   # SOLE-RUNG GATE
ARM_ENV="" run_cell s1amx_288 mixtral-8x22b "asym_cpuadamwds|T1" 288000 "1" "$POL" 1

# ---- rank-2 section (inside GPUs 0,1)
echo "== stdz job2 2r start $(date +%F_%H:%M) ==" >> "$S"
export GPU="0,1" CUDA_VISIBLE_DEVICES="0,1" GPU_POOL="0,1" DDP_TIMEOUT=1500 HOSTFLOOR=1100
ARM_ENV="" run_cell s2a35_768 q3.5-35b-a3b "$SDP|T2" 768000 "1" "$POL" 2
ARM_ENV="" run_cell s2umx_160 mixtral-8x22b "$UNS" 160000 "1" "$POL" 2
ARM_ENV="" run_cell s2amx_160 mixtral-8x22b "$SDP|T1" 160000 "1" "$POL" 2
ARM_ENV="" run_cell s2rmx_160 mixtral-8x22b "$RC" 160000 "1" "$POL" 2
ARM_ENV="" run_cell s2zmx_160 mixtral-8x22b "$Z3" 160000 "1" "$POL" 2
ARM_ENV="" run_cell s2umx_224 mixtral-8x22b "$UNS" 224000 "1" "$POL" 2
ARM_ENV="" run_cell s2amx_224 mixtral-8x22b "$SDP|T1" 224000 "1" "$POL" 2
echo "== stdz job2 DONE $(date +%F_%H:%M) ==" >> "$S"
