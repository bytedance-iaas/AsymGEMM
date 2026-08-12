#!/bin/bash
# gpt-oss-20b campaign — CHAIN D: 1-rank tp-vs-seq ladder (turning points +
# ceiling). Per rung: rc -> un -> uo -> asym (T1, promote T2B -> T3 on OOM).
# Dead systems skip later rungs (wall bracketed = OOM with b=1). Ladder stops
# when asym T3 itself walls (the ceiling) or rungs exhaust.
set -uo pipefail
export GPU="${GPU:-0}" HOSTFLOOR="${HOSTFLOOR:-600}"
source /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c17.sh
S2="$LOGD/gptoss_status.log"
POL="none|false|false|false|false|false"
MODEL=gpt-oss-20b
export CUDA_VISIBLE_DEVICES=0 GPU_POOL=0
export ASYM_OFFLOAD_MODULES=all
RC="superoffload_mem|recomp"
UNS="superoffload_mem|unsloth-ohbm0"
UO="superoffload_mem|unsloth-off-ohbm0"
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
T3ENV=(ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1
       ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 ASYMM_QWEN3_MOE_FG_DA_GPU=1
       ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1 ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1
       ASYM_CPU_OPS_THREADS=48 ASYM_PLACEMENT_POLICY=1)
note() { echo "[$(date +%H:%M:%S)] $*" >> "$S2"; }
harv() { /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/.venv/bin/python \
           agent/anchors_tmp/gptoss_harvest.py "$1" "${2:-1}" 2>/dev/null | tee -a "$S2"; }

dead_rc=0; dead_un=0; dead_uo=0
asym_tier=T1   # promotes T1 -> T2B -> T3; ladder ends when T3 walls

blist_for() { # $1 seq -> batch walk list
  local s=$1
  if   [ "$s" -le 64000 ];  then echo "8 6 4 2 1"
  elif [ "$s" -le 128000 ]; then echo "6 4 2 1"
  elif [ "$s" -le 256000 ]; then echo "2 1"
  else echo "1"; fi
}

run_sys() { # $1 tag $2 systok $3 seq $4 blist -> verdict
  local v; v=$(run_cell "$1" "$MODEL" "$2" "$3" "$4" "$POL" 1)
  note "CELL $1 ${2%%|*} s=$3 -> $v"; harv "$1" >/dev/null; echo "$v"; }
run_t3() { local v; v=$( (export "${T3ENV[@]}"; run_cell "$1" "$MODEL" "$T3TOK" "$3" "$4" "$POL" 1) )
  note "CELL $1 T3 s=$3 -> $v"; harv "$1" >/dev/null; echo "$v"; }

note "CHAIN-D begin (1r ladder)"
for seq in 32000 64000 96000 128000 192000 256000 320000 384000 448000 512000 640000 768000 896000 1024000; do
  sk=$((seq/1000)); bl=$(blist_for "$seq")
  note "RUNG ${sk}k begin (asym_tier=$asym_tier)"
  if [ "$dead_rc" = 0 ]; then
    v=$(run_sys "d1rc${sk}" "$RC" "$seq" "$bl"); [ "$v" = "TRAINED" ] || dead_rc=1
  fi
  if [ "$dead_un" = 0 ]; then
    v=$(run_sys "d1un${sk}" "$UNS" "$seq" "$bl"); [ "$v" = "TRAINED" ] || dead_un=1
  fi
  if [ "$dead_uo" = 0 ]; then
    v=$(run_sys "d1uo${sk}" "$UO" "$seq" "$bl"); [ "$v" = "TRAINED" ] || dead_uo=1
  fi
  # asym with tier promotion inside the rung (T1 -> T2 -> T2B -> T3)
  va=FAIL
  if [ "$asym_tier" = "T1" ]; then
    va=$(run_sys "d1t1${sk}" "asym_cpuadamwds|T1" "$seq" "$bl")
    [ "$va" = "TRAINED" ] || { asym_tier=T2; note "PROMOTE asym T1->T2 at ${sk}k"; }
  fi
  if [ "$va" != "TRAINED" ] && [ "$asym_tier" = "T2" ]; then
    va=$(run_sys "d1t2${sk}" "asym_cpuadamwds|T2" "$seq" "$bl")
    [ "$va" = "TRAINED" ] || { asym_tier=T2B; note "PROMOTE asym T2->T2B at ${sk}k"; }
  fi
  if [ "$va" != "TRAINED" ] && [ "$asym_tier" = "T2B" ]; then
    va=$(run_sys "d1a2b${sk}" "asym_cpuadamwds|T2B" "$seq" "$bl")
    [ "$va" = "TRAINED" ] || { asym_tier=T3; note "PROMOTE asym T2B->T3 at ${sk}k"; }
  fi
  if [ "$va" != "TRAINED" ] && [ "$asym_tier" = "T3" ]; then
    va=$(run_t3 "d1t3${sk}" x "$seq" "$bl")
    if [ "$va" != "TRAINED" ]; then
      note "ASYM WALL at ${sk}k (T3 failed) — CEILING BRACKETED, ladder ends"
      break
    fi
  fi
  note "RUNG ${sk}k done"
done
note "CHAIN-D COMPLETE"
