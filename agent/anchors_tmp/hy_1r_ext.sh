#!/bin/bash
# Hunyuan 1-rank EXTENSION — runs after HY_ALL_DONE (phase 3 finishes).
# Goal: complete the last-standing arc. (a) climb uns-off until its wall;
# (b) asym re-enters at 384k on deeper tiers (T2B, then T3 fallback per rung)
# and climbs until its wall. Panel = best tier per rung (house collapse).
set -uo pipefail
S2="/workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/hy_status.log"
until grep -q 'HY_ALL_DONE' "$S2" 2>/dev/null; do sleep 120; done

export GPU=0 HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c17.sh
export CUDA_VISIBLE_DEVICES=0 GPU_POOL=0
export ASYM_OFFLOAD_MODULES="routed_experts,shared_experts,attention,norms,mlp_dense"
POL="none|false|false|false|false|false"

cell() { # $1 tag $2 systok $3 seq
  local v
  v=$(run_cell "$1" hunyuan-a13b "$2" "$3" "1" "$POL" 1)
  echo "HY CELL $1 r1 ${2%%|*} s=$3 -> $v $(date +%H:%M:%S)" >> "$S2"
  echo "$v"
}

# (a) uns-off wall hunt
for s in 448000 512000 576000 640000; do
  v=$(cell "hyx_uo$((s/1000))" "superoffload_mem|unsloth-off-ohbm0" "$s")
  [ "$v" = "TRAINED" ] || { echo "HY r1 UO wall at $s ($v)" >> "$S2"; break; }
done

# (b) asym deep tiers: per rung try T2B, fall back to T3; stop when both fail
for s in 384000 448000 512000 576000 640000; do
  v=$(cell "hyx_a2b$((s/1000))" "asym_cpuadamwds|T2B" "$s")
  if [ "$v" != "TRAINED" ]; then
    v=$(cell "hyx_a3t$((s/1000))" "asym_cpuadamwds|T3" "$s")
    if [ "$v" != "TRAINED" ]; then
      echo "HY r1 ASYM-deep wall at $s (T2B+T3 both $v-class)" >> "$S2"
      break
    fi
  fi
done
echo "HY_1R_EXT_DONE $(date +%H:%M:%S)" >> "$S2"
