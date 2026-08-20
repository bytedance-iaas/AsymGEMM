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
POL="none|false|false|false|false|false"
MODEL=gpt-oss-120b
export CUDA_VISIBLE_DEVICES=0,1 GPU_POOL=0,1
export ASYM_OFFLOAD_MODULES=all
export ASYM_HOST_MEM_WATCHDOG_FLOOR_GIB=35 ASYM_HOST_MEM_WATCHDOG_FLOOR_GB=35
RC="superoffload_mem|recomp"
UNS="superoffload_mem|unsloth-ohbm0"
UO="superoffload_mem|unsloth-off-ohbm0"
T1="asym_sepplanlink2_cpuadamwds|T1"
T2B="asym_sepplanlink2_cpuadamwds|T2B"
T3TOK="asym_sepplanlink2_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
T3ENV=(ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1
       ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 ASYMM_QWEN3_MOE_FG_DA_GPU=1
       ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1 ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1
       ASYM_CPU_OPS_THREADS=48 ASYM_PLACEMENT_POLICY=1)
note() { echo "[$(date +%m-%d_%H:%M)] $*" >> "$S2"; }

LADDER="32000 64000 128000 192000 256000 320000 384000 448000 512000 640000 768000 896000 1020000"
blist_for() { case "$1" in 32000|64000) echo "2 1" ;; *) echo "1" ;; esac; }

# ---- smoke: integration proof (asym T1 @32k b1) ----
note "G2B resume (un/uo/asym; patched verdict)"

# ---- baseline walls ----
for pair in "un:$UNS" "uo:$UO"; do
  name="${pair%%:*}"; tok="${pair#*:}"
  for seq in $LADDER; do
    v=$(run_cell "g220${name}$((seq/1000))" "$MODEL" "$tok" "$seq" "$(blist_for $seq)" "$POL" 2)
    note "CELL ${name} s=$seq -> $v"
    case "$v" in GOOM|COOM) note "${name} WALL at $seq"; break ;; FAIL) note "${name} FAIL at $seq (investigate)"; break ;; esac
  done
done

# ---- asym tier ladder ----
tier=T1; tok="$T1"
for seq in $LADDER; do
  if [ "$tier" = "T3" ]; then
    v=$( (export "${T3ENV[@]}"; run_cell "g220a$((seq/1000))" "$MODEL" "$T3TOK" "$seq" "1" "$POL" 2) )
  else
    v=$(run_cell "g220a$((seq/1000))" "$MODEL" "$tok" "$seq" "$(blist_for $seq)" "$POL" 2)
  fi
  note "CELL asym[$tier] s=$seq -> $v"
  if [ "$v" = "GOOM" ] || [ "$v" = "COOM" ]; then
    case "$tier" in
      T1)  tier=T2B; tok="$T2B"; note "tier -> T2B at $seq"
           v=$(run_cell "g220a$((seq/1000))b" "$MODEL" "$tok" "$seq" "1" "$POL" 2); note "CELL asym[T2B] s=$seq -> $v" ;;
      T2B) tier=T3; note "tier -> T3 at $seq"
           v=$( (export "${T3ENV[@]}"; run_cell "g220a$((seq/1000))c" "$MODEL" "$T3TOK" "$seq" "1" "$POL" 2) ); note "CELL asym[T3] s=$seq -> $v" ;;
      T3)  note "asym WALL (T3) at $seq"; break ;;
    esac
    if [ "$v" = "GOOM" ] || [ "$v" = "COOM" ]; then
      if [ "$tier" = "T2B" ]; then tier=T3; note "tier -> T3 at $seq (T2B also OOM)"
        v=$( (export "${T3ENV[@]}"; run_cell "g220a$((seq/1000))c" "$MODEL" "$T3TOK" "$seq" "1" "$POL" 2) ); note "CELL asym[T3] s=$seq -> $v"
        [ "$v" = "TRAINED" ] || { note "asym WALL at $seq"; break; }
      else
        note "asym WALL at $seq"; break
      fi
    fi
  fi
done
note "G2B-DONE"
