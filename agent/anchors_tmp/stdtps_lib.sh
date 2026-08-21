#!/bin/bash
# stdtps_lib.sh — c11/SFT-39 port of tpfig_lib_c17.sh (sourced) for the
# TP-figure x-axis standardization campaign (agent/impls/s04-p1-dgx-02-c06/
# standardize_tps.md). IDENTICAL measurement protocol; only workspace root,
# log dir, and node tag differ. Env expected: GPU (index or "0,1"),
# HOSTFLOOR (GB available required to start).
cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512 DATASET_OVERWRITE=false OVERWRITE=false
export ASYM_ZERO_ROUTER_JITTER=1 TRUST_REMOTE_CODE=false
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
export CUDA_VISIBLE_DEVICES=${GPU:?}
LOGD=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp
S="$LOGD/stdtps_status.log"
B=profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16

guard() { for i in $(seq 1 180); do
    n=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    a=$(free -g | awk 'NR==2{print $7}')
    if [ "$n" -eq 0 ] && [ "$a" -ge "${HOSTFLOOR:?}" ]; then return 0; fi
    if [ $((i % 9)) -eq 0 ]; then
      # reap ORPHANED gpu holders (ppid=1) only — never a live run's children
      for p in $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d " ")
        [ "$pp" = "1" ] && kill -9 "$p" 2>/dev/null && echo "GUARD-REAPED orphan $p gpu=$GPU $(date +%H:%M)" >> "$S"
      done
    fi
    sleep 20
  done; echo "GUARD-TIMEOUT gpu=$GPU $(date +%H:%M)" >> "$S"; return 1; }

# shm_guard — 2-rank fabric cells only: stale asym_* segments in /dev/shm
# poison verdicts (HY_CAMPAIGN ops lesson). Wait until /dev/shm < 5G used.
shm_guard() { local used
  for i in $(seq 1 30); do
    used=$(df -BG /dev/shm | awk 'NR==2{gsub("G","",$3); print $3}')
    [ "${used:-0}" -lt 5 ] && return 0
    rm -rf /dev/shm/asym_* 2>/dev/null || true
    sleep 10
  done
  echo "STDTPS shm_guard timeout used=${used}G $(date +%H:%M)" >> "$S"; }

verdict() { local dtag="$1" log="$2"
  local tsv="$B/${dtag}/jobs.tsv"
  grep -aqE "OutOfMemoryError|CUDA out of memory" "$log" 2>/dev/null && { echo GOOM; return; }
  grep -aq "dropped below floor" "$log" 2>/dev/null && { echo COOM; return; }
  if [ -f "$tsv" ] && awk -F'\t' 'NR>1 && $1=="ok"{f=1} END{exit !f}' "$tsv"; then echo TRAINED; return; fi
  echo FAIL
}

# run_cell TAG MODEL SYSTOKEN SEQ "B1 B2 ..." [POLICY] [RANKS]  -> walks batches, first TRAINED wins
run_cell() { local tag="$1" model="$2" systok="$3" seq="$4" blist="$5" policy="${6:-none|false|false|false|false|false}" ranks="${7:-1}" v=SKIP b
  for b in $blist; do
    guard || return 1
    echo "START $tag $model $systok s=$seq b=$b r=$ranks $(date +%H:%M)" >> "$S"
    RUN_NAME="${tag}-${ASYM_HOSTTAG:-${HOSTNAME##*-}}_${model}" RUNS="${model}|${ranks} ; ${systok}|ligerloss1 ; ${seq}|${b}|1 ; ${policy}" \
      bash scripts/lf/profile_lora_lf_test_source.sh >> "$LOGD/r_${tag}_b${b}.log" 2>&1
    local dmodel=${model//./_}
    v=$(verdict "${tag}-${ASYM_HOSTTAG:-${HOSTNAME##*-}}_${dmodel}__b${b}_s${seq}_ga1_drop000" "$LOGD/r_${tag}_b${b}.log")
    echo "CELL $tag $systok s=$seq b=$b -> $v $(date +%H:%M)" >> "$S"
    [ "$v" = "TRAINED" ] && break
    [ "$v" = "FAIL" ] && break
  done; echo "$v"; }
