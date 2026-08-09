#!/bin/bash
# Shared library for the tp-vs-seq figure campaign chains (sourced).
# c18 COPY of the SFT-39 template (run_glms.md §4): paths -> SFT-46 tree,
# RUN_NAME/dirname suffix -c18_, NUMACTL pinning added (c18 house protocol;
# GPU HBM = NUMA nodes 2/10/18/26 — never bind those, CPU = 0,1).
# Env expected: GPU (index or "0,1"), HOSTFLOOR (GB available required to start).
cd /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512 DATASET_OVERWRITE=false OVERWRITE=false
export ASYM_ZERO_ROUTER_JITTER=1 TRUST_REMOTE_CODE=false
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
export NUMACTL_ENABLE=1 NUMACTL_MODE=membind NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1
export CUDA_VISIBLE_DEVICES=${GPU:?}
LOGD=/workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp
S="$LOGD/tpfig_status.log"
B=profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16

guard() { for i in $(seq 1 180); do
    # count only LIVE holders — the driver keeps ghost entries for dead pids
    n=0; for p in $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      p=${p//,/}; [ -d "/proc/$p" ] && n=$((n+1)); done
    # stale fabric arenas from SIGBUS/killed teardowns hold tmpfs pages that
    # count against MemAvailable (2026-08-05: a full 479G shm pinned avail
    # below the floor and wedged this guard for 40 min) — clean when no
    # trainer is alive to own them.
    if [ "$n" -eq 0 ] && ls /dev/shm/asym_* >/dev/null 2>&1; then
      echo "GUARD-SHM-CLEAN $(ls /dev/shm | head -3 | tr '\n' ' ') $(date +%H:%M)" >> "$S"
      rm -f /dev/shm/asym_* 2>/dev/null || true
    fi
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

verdict() { local dtag="$1" log="$2"
  local tsv="$B/${dtag}/jobs.tsv"
  grep -aqE "OutOfMemoryError|CUDA out of memory" "$log" 2>/dev/null && { echo GOOM; return; }
  grep -aq "dropped below floor" "$log" 2>/dev/null && { echo COOM; return; }
  if [ -f "$tsv" ] && awk -F'\t' 'NR>1 && $1=="ok"{f=1} END{exit !f}' "$tsv"; then echo TRAINED; return; fi
  echo FAIL
}

# run_cell TAG MODEL SYSTOKEN SEQ "B1 B2 ..." [POLICY] [RANKS]  -> walks batches, first TRAINED wins
# POLICY = the 4th RUNS field: policy|expact|attnact|layeract|layergc|sdparecomp
# RANKS: 1 (default) or 2 (GPU_POOL must span the GPUs; guard covers $GPU list)
run_cell() { local tag="$1" model="$2" systok="$3" seq="$4" blist="$5" policy="${6:-none|false|false|false|false|false}" ranks="${7:-1}" v=SKIP b
  for b in $blist; do
    guard || return 1
    echo "START $tag $model $systok s=$seq b=$b r=$ranks $(date +%H:%M)" >> "$S"
    RUN_NAME="${tag}-c18_${model}" RUNS="${model}|${ranks} ; ${systok}|ligerloss1 ; ${seq}|${b}|1 ; ${policy}" \
      bash scripts/lf/profile_lora_lf_test_source.sh >> "$LOGD/r_${tag}_b${b}.log" 2>&1
    local dmodel=${model//./_}
    v=$(verdict "${tag}-c18_${dmodel}__b${b}_s${seq}_ga1_drop000" "$LOGD/r_${tag}_b${b}.log")
    echo "CELL $tag $systok s=$seq b=$b -> $v $(date +%H:%M)" >> "$S"
    [ "$v" = "TRAINED" ] && break
    [ "$v" = "FAIL" ] && break
  done; echo "$v"; }
