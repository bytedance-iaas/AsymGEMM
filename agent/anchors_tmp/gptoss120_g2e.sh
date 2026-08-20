#!/bin/bash
# gpt-oss-120b campaign — CHAIN G1 (1-rank turning points), c14, serial GPU0.
# Mimics GPTOSS20B_CAMPAIGN chain structure on the c14g lib: per-system upward
# cascade to the first OOM (walls), then the asym tier ladder (T1 -> T2B ->
# T3-raw) to its deepest fit. FA4 auto (is_gptoss_model_name), liger loss-only
# both sides, OFFLOAD_MODULES=all, watchdog floor 35.
set -uo pipefail
export GPU="${GPU:-0,1}" HOSTFLOOR="${HOSTFLOOR:-500}"
source /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c14g.sh
S2="$LOGD/gptoss120_status.log"
POL="none|false|false|false|false|true"
MODEL=gpt-oss-120b
export CUDA_VISIBLE_DEVICES=0,1 GPU_POOL=0,1
export ASYM_OFFLOAD_MODULES=all
export ASYM_HOST_MEM_WATCHDOG_FLOOR_GIB=25 ASYM_HOST_MEM_WATCHDOG_FLOOR_GB=25 HOST_MEM_WATCHDOG_FLOOR_GB=25
RC="superoffload_mem|recomp"
UNS="superoffload_mem|unsloth-ohbm0"
UO="superoffload_mem|unsloth-off-ohbm0"
T1="asym_sepplanlink2_cpuadamwds|unsloth-ohbm2"
T2B="asym_sepplanlink2_cpuadamwds|T2B"
T3TOK="asym_sepplanlink2_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
T3ENV=(ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1
       ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 ASYMM_QWEN3_MOE_FG_DA_GPU=1
       ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1 ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1
       ASYM_CPU_OPS_THREADS=48 ASYM_PLACEMENT_POLICY=1)
note() { echo "[$(date +%m-%d_%H:%M)] $*" >> "$S2"; }

LADDER="768000"
blist_for() { case "$1" in 32000|64000) echo "2 1" ;; *) echo "1" ;; esac; }

# ---- smoke: integration proof (asym T1 @32k b1) ----
note "G2E crown probe: T1-ohbm2 + gradoff @768k"

# ---- baseline walls ----
# ---- asym T1-ohbm4 ladder (no fallthrough; host-targeted) ----
for seq in $LADDER; do
  v=$(run_cell "g2ia$((seq/1000))" "$MODEL" "$T1" "$seq" "1" "$POL" 2)
  note "CELL asym[T1o2] s=$seq -> $v"
  if [ "$v" = "GOOM" ] || [ "$v" = "COOM" ]; then note "asym T1o2 WALL at $seq"; break; fi
  [ "$v" = "FAIL" ] && { note "asym T1o2 FAIL at $seq"; break; }
done
note "G2E-DONE"
