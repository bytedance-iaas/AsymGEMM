#!/bin/bash
# gpt-oss-20b — CHAIN D3: single T2B ceiling-bracketing cell at 1.664M
# (1.536M trained @RSS 825G; host pool ~900G usable -> expect host wall).
set -uo pipefail
export GPU="${GPU:-0}" HOSTFLOOR="${HOSTFLOOR:-600}"
source /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c17.sh
S2="$LOGD/gptoss_status.log"
POL="none|false|false|false|false|false"
export CUDA_VISIBLE_DEVICES=0 GPU_POOL=0
export ASYM_OFFLOAD_MODULES=all
note() { echo "[$(date +%H:%M:%S)] $*" >> "$S2"; }
v=$(run_cell "d1a2b1664" "gpt-oss-20b" "asym_cpuadamwds|T2B" 1664000 "1" "$POL" 1)
note "CELL d1a2b1664 T2B s=1664000 -> $v"
/workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/.venv/bin/python \
  agent/anchors_tmp/gptoss_harvest.py d1a2b1664 1 2>/dev/null | tee -a "$S2"
if [ "$v" = "TRAINED" ]; then
  note "T2B ALIVE at 1.664M — ceiling deeper than planned grid; stopping 1r here (diminishing returns, 2r pending)"
else
  note "1R T2B CEILING BRACKETED (1536k,1664k] ($v)"
fi
note "CHAIN-D3 COMPLETE"
