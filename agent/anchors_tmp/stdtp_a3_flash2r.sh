#!/bin/bash
# stdtp_a3_flash2r.sh — Agent 3 / phase 2 (standardize_tps.md): GLM-4.7-Flash
# 2-rank @384k. rc + fsdp2 walls bracket (320k,416k] -> 384k must be PROBED;
# uns/uo fit at 416k -> measured cells; asym T1 batch walk "2 1" (416k banked
# b2, 320k b2). zero3 stays derived-from-rc (est) per the row's banking rule.
# Baselines first, asym last (house order). Arena default (Flash 2r precedent).
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU="0,1" HOSTFLOOR=1200
. /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/stdtp_lib.sh
export GPU_POOL="0,1" DDP_TIMEOUT=1500
python3 /workspace/AsymGEMM-SFT-38/.repair_dataset_info.py >> "$S" 2>&1 || echo "repair-dataset-info FAILED" >> "$S"
echo "== STDTP-A3-FLASH2R begin (2r, phys GPUs 1,3 -> inside 0,1) $(date '+%m-%d %H:%M')" >> "$S"
P="none|false|false|false|false|false"

run_cell f2rc384 glm4.7-flash "superoffload_mem|recomp"            384000 "1" "$P" 2 >/dev/null
run_cell f2fd384 glm4.7-flash "fsdp2_offload|recomp"               384000 "1" "$P" 2 >/dev/null
run_cell f2un384 glm4.7-flash "superoffload_mem|unsloth"           384000 "1" "$P" 2 >/dev/null
run_cell f2uo384 glm4.7-flash "superoffload_mem|unsloth-off-ohbm0" 384000 "1" "$P" 2 >/dev/null
run_cell f2t1384 glm4.7-flash "asym_sdp2_cpuadamwds|T1"            384000 "2 1" "$P" 2 >/dev/null
echo "== STDTP-A3-FLASH2R-DONE $(date '+%m-%d %H:%M')" >> "$S"
