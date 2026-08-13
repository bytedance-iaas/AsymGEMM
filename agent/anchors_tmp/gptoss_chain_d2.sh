#!/bin/bash
# gpt-oss-20b campaign — CHAIN D2: 1-rank CEILING EXTENSION past 1.02M.
# Asym-only (baselines all walled ≤768k): T1 + T2B dual-track, T3 after T2B
# walls. Runs until the asym wall brackets or rungs exhaust at 1.5M.
set -uo pipefail
export GPU="${GPU:-0}" HOSTFLOOR="${HOSTFLOOR:-600}"
source /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c17.sh
S2="$LOGD/gptoss_status.log"
POL="none|false|false|false|false|false"
MODEL=gpt-oss-20b
export CUDA_VISIBLE_DEVICES=0 GPU_POOL=0
export ASYM_OFFLOAD_MODULES=all
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
T3ENV=(ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1
       ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 ASYMM_QWEN3_MOE_FG_DA_GPU=1
       ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1 ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1
       ASYM_CPU_OPS_THREADS=48 ASYM_PLACEMENT_POLICY=1)
note() { echo "[$(date +%H:%M:%S)] $*" >> "$S2"; }
harv() { /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/.venv/bin/python \
           agent/anchors_tmp/gptoss_harvest.py "$1" "${2:-1}" 2>/dev/null | tee -a "$S2"; }
run_sys() { local v; v=$(run_cell "$1" "$MODEL" "$2" "$3" "$4" "$POL" 1)
  note "CELL $1 ${2%%|*} s=$3 -> $v"; harv "$1" >/dev/null; echo "$v"; }
run_t3() { local v; v=$( (export "${T3ENV[@]}"; run_cell "$1" "$MODEL" "$T3TOK" "$3" "$4" "$POL" 1) )
  note "CELL $1 T3 s=$3 -> $v"; harv "$1" >/dev/null; echo "$v"; }

dead_t1="${DEAD_T1:-0}"; dead_t2b="${DEAD_T2B:-0}"
note "CHAIN-D2 begin (ceiling extension; t1dead=$dead_t1 t2bdead=$dead_t2b)"
for seq in 1152000 1280000 1408000 1536000; do
  sk=$((seq/1000)); bl="1"
  note "RUNG ${sk}k begin (ext; t1dead=$dead_t1 t2bdead=$dead_t2b)"
  va=FAIL
  if [ "$dead_t1" = 0 ]; then
    v=$(run_sys "d1t1${sk}" "asym_cpuadamwds|T1" "$seq" "$bl")
    [ "$v" = "TRAINED" ] || { dead_t1=1; note "T1 WALL at ${sk}k"; }
    [ "$v" = "TRAINED" ] && va=TRAINED
  fi
  if [ "$dead_t2b" = 0 ]; then
    v=$(run_sys "d1a2b${sk}" "asym_cpuadamwds|T2B" "$seq" "$bl")
    [ "$v" = "TRAINED" ] || { dead_t2b=1; note "T2B WALL at ${sk}k"; }
    [ "$v" = "TRAINED" ] && va=TRAINED
  fi
  if [ "$dead_t2b" = 1 ]; then
    v=$(run_t3 "d1t3${sk}" x "$seq" "$bl")
    [ "$v" = "TRAINED" ] && va=TRAINED
    if [ "$v" != "TRAINED" ] && [ "$va" != "TRAINED" ]; then
      note "1R ASYM CEILING BRACKETED at ${sk}k — extension ends"
      break
    fi
  fi
  note "RUNG ${sk}k done"
done
note "CHAIN-D2 COMPLETE"
