#!/bin/bash
# HY 1r resume #2 (rungs 224-256K): rc+un DEAD, T1 dead (wall (160K,192K]) ->
# ladder starts at T2B. Registry repaired for s224000.
set -uo pipefail
export GPU=0 HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
export CUDA_VISIBLE_DEVICES=0
unset GPU_POOL DDP_TIMEOUT ASYM_ARENA_SHM_CAP_GB || true
export ASYM_OFFLOAD_MODULES="routed_experts,shared_experts,attention,norms,mlp_dense"
POL="none|false|false|false|false|false"
UO="superoffload_mem|unsloth-off-ohbm0"
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker101-ceil0000-ohbm0"
TIERS=(T2B T3); ti=0
cell() { local v; shm_guard
  v=$(run_cell "$1" hunyuan-a13b "$2" "$3" "$4" "$POL" 1)
  echo "STDTPS96-A4 CELL $1 r1 ${2#*|} s=$3 -> $v $(date +%H:%M:%S)" >> "$S"; echo "$v"; }
asym() { local seq=$1 t tok v
  while [ $ti -lt ${#TIERS[@]} ]; do
    t=${TIERS[$ti]}
    tok="asym_cpuadamwds|$t"; [ "$t" = "T3" ] && tok="$T3TOK"
    v=$(cell "s96hyf1as_$((seq/1000))${t,,}c" "$tok" "$seq" "1")
    case "$v" in
      TRAINED) return 0;;
      GOOM|COOM) ti=$((ti+1));;
      *) echo "STDTPS96-A4 FAIL-STOP asym s=$seq v='$v'" >> "$S"; exit 1;;
    esac
  done
  echo "STDTPS96-A4 r1 ASYM DEAD at $seq" >> "$S"; exit 1; }
uocell() { local seq=$1 v
  v=$(cell "s96hyf1uo_$((seq/1000))c" "$UO" "$seq" "1")
  case "$v" in TRAINED|GOOM|COOM) ;; *) echo "STDTPS96-A4 FAIL-STOP uo s=$seq v='$v'" >> "$S"; exit 1;; esac
  echo "$v"; }
echo "STDTPS96-A4-HY1R-C BEGIN (224K) $(date '+%F %H:%M:%S')" >> "$S"
v=$(uocell 224000); asym 224000
if [ "$v" = "TRAINED" ]; then uocell 256000 >/dev/null; fi
asym 256000
echo "STDTPS96-A4-HY1R-C-DONE $(date '+%F %H:%M:%S')" >> "$S"
