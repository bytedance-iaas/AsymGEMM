#!/bin/bash
# stdtps96_lib.sh — run library for the GH200-96GB simulated campaign
# (standardize_tps_96gb.md). Fork of fig12_lib.sh with the OCCUPIER-AWARE
# guard: the HBM occupiers (hbm96_occupy.py, one per simulated GPU) are
# whitelisted residents — the guard waits for every OTHER compute app to
# drain and NEVER reaps occupier PIDs. Env: GPU (inside ids), HOSTFLOOR,
# OCC_PIDS (space-separated occupier PIDs, REQUIRED).
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512 DATASET_OVERWRITE=false OVERWRITE=false
export ASYM_ZERO_ROUTER_JITTER=1 TRUST_REMOTE_CODE=false
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
export NUMACTL_ENABLE=1 NUMACTL_MODE=membind NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1
export CUDA_VISIBLE_DEVICES=${GPU:?}
: "${OCC_PIDS:?set OCC_PIDS to the occupier pids before sourcing}"
LOGD=/workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp
S="$LOGD/stdtps96_status.log"
B=profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16

is_occ() { local q="$1" p; for p in $OCC_PIDS; do [ "$q" = "$p" ] && return 0; done; return 1; }

guard() { for i in $(seq 1 180); do
    n=0; for p in $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      p=${p//,/}; is_occ "$p" && continue
      [ -d "/proc/$p" ] && n=$((n+1)); done
    # occupier liveness: every OCC_PID must be alive or the budget is open
    local dead=0; for p in $OCC_PIDS; do [ -d "/proc/$p" ] || dead=1; done
    if [ "$dead" = 1 ]; then echo "GUARD-OCCUPIER-DEAD $(date +%H:%M)" >> "$S"; return 2; fi
    if [ "$n" -eq 0 ] && ls /dev/shm/asym_* >/dev/null 2>&1; then
      echo "GUARD-SHM-CLEAN $(date +%H:%M)" >> "$S"
      rm -f /dev/shm/asym_* 2>/dev/null || true
    fi
    a=$(free -g | awk 'NR==2{print $7}')
    if [ "$n" -eq 0 ] && [ "$a" -ge "${HOSTFLOOR:?}" ]; then return 0; fi
    if [ $((i % 9)) -eq 0 ]; then
      for p in $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        p=${p//,/}; is_occ "$p" && continue
        pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d " ")
        [ "$pp" = "1" ] && kill -9 "$p" 2>/dev/null && echo "GUARD-REAPED orphan $p $(date +%H:%M)" >> "$S"
      done
    fi
    sleep 20
  done; echo "GUARD-TIMEOUT gpu=$GPU $(date +%H:%M)" >> "$S"; return 1; }

verdict() { local dtag="$1" log="$2"
  local tsv="$B/${dtag}/jobs.tsv"
  grep -aqE "OutOfMemoryError|CUDA out of memory" "$log" 2>/dev/null && { echo GOOM; return; }
  grep -aq "dropped below floor" "$log" 2>/dev/null && { echo COOM; return; }
  if [ -f "$tsv" ] && awk -F'\t' 'NR>1 && $1=="ok"{f=1} END{exit !f}' "$tsv"; then echo TRAINED; return; fi
  echo FAIL
}

# run_cell TAG MODEL SYSTOKEN SEQ "B1 B2 ..." [POLICY] [RANKS]
run_cell() { local tag="$1" model="$2" systok="$3" seq="$4" blist="$5" policy="${6:-none|false|false|false|false|false}" ranks="${7:-1}" v=SKIP g b
  for b in $blist; do
    guard; g=$?
    if [ "$g" = 2 ]; then echo "CELL $tag $systok s=$seq b=$b -> INVALID-OCCUPIER-DEAD $(date +%H:%M)" >> "$S"; echo OCCDEAD; return 1; fi
    [ "$g" != 0 ] && return 1
    echo "START $tag $model $systok s=$seq b=$b r=$ranks arm='${ARM_ENV:-}' $(date +%H:%M)" >> "$S"
    ( set -a; for kv in ${ARM_ENV:-}; do export "$kv"; done; set +a
      RUN_NAME="${tag}_${model}" RUNS="${model}|${ranks} ; ${systok}|ligerloss1 ; ${seq}|${b}|1 ; ${policy}" \
        bash scripts/lf/profile_lora_lf_test_source.sh ) >> "$LOGD/r_${tag}_b${b}.log" 2>&1
    local dmodel=${model//./_}
    v=$(verdict "${tag}_${dmodel}__b${b}_s${seq}_ga1_drop000" "$LOGD/r_${tag}_b${b}.log")
    # occupier liveness at cell end: dead occupier invalidates the verdict
    for p in $OCC_PIDS; do [ -d "/proc/$p" ] || { echo "CELL $tag $systok s=$seq b=$b -> INVALID-OCCUPIER-DIED $(date +%H:%M)" >> "$S"; echo OCCDEAD; return 1; }; done
    echo "CELL $tag $systok s=$seq b=$b -> $v $(date +%H:%M)" >> "$S"
    [ "$v" = "TRAINED" ] && break
    [ "$v" = "FAIL" ] && break
  done; echo "$v"; }
