#!/bin/bash
# uo@512k 1r — the one cell my C-1r walk list missed (ended at 448k).
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU="0" HOSTFLOOR=300 OCC_PIDS="613592 613593"
. /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.c14.sh
export GPU_POOL="0"; unset DDP_TIMEOUT || true
run_cell n35c1uo512 q3.5-35b-a3b "superoffload_mem|unsloth-off-ohbm0" 512000 "1" "none|false|false|false|false|false" 1 >/dev/null
echo "== Q35-GAPCELL-DONE $(date '+%m-%d %H:%M')" >> "$S"
