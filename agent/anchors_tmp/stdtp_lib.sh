#!/bin/bash
# stdtp_lib.sh — TP-figure x-axis standardization campaign (standardize_tps.md),
# machine s04-p1-dgx-02-c14, SFT-38 tree. Port of tpfig_lib_c14g.sh: paths ->
# SFT-38, RUN_NAME suffix -c14s_. Container note: asym_sft_38's rootfs on c14
# is a broken shell (no /root, no /bin/sh) — chains run in the known-good
# asym_sft_46 rootfs with /home/kevinni/AsymGEMM-SFT-38 mounted at
# /workspace/AsymGEMM-SFT-38 (same image family; venv lives in the tree).
# GPU numbering: the host wrapper restricts NVIDIA_VISIBLE_DEVICES, so inside
# the container GPUs are always 0[,1] (phys 3 for 1r; phys 1,3 for 2r — user
# assignment 2026-08-20).
# Env expected: GPU (inside index or "0,1"), HOSTFLOOR (GB avail required).
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512 DATASET_OVERWRITE=false OVERWRITE=false
export ASYM_ZERO_ROUTER_JITTER=1 TRUST_REMOTE_CODE=false
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
# All models + the smoltalk corpus are in the node cache; offline mode
# skips hub HEAD checks that were 429-stalling runs (override per-cell
# with HF_HUB_OFFLINE=0 if a build ever needs a fresh hub file).
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export NUMACTL_ENABLE=1 NUMACTL_MODE=membind NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1
export CUDA_VISIBLE_DEVICES=${GPU:?}
LOGD=/workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp
S="$LOGD/stdtp_status.log"
B=profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16

guard() { for i in $(seq 1 180); do
    # count only LIVE holders — the driver keeps ghost entries for dead pids
    n=0; for p in $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      p=${p//,/}; [ -d "/proc/$p" ] && n=$((n+1)); done
    # stale fabric arenas hold tmpfs pages against MemAvailable — clean when
    # no trainer is alive to own them.
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
  # completeness-checker quirk: jobs.tsv says failed but the run finished —
  # trust step_samples with a completed measured step.
  local d="$B/${dtag}"
  if ls "$d"/*/b*_s*/step_samples.json >/dev/null 2>&1; then
    if python3 - "$d" <<'PY'
import glob, json, sys
for f in glob.glob(sys.argv[1] + "/*/b*_s*/step_samples.json"):
    ss = json.load(open(f))
    if any(not x.get("is_warmup") for x in ss):
        sys.exit(0)
sys.exit(1)
PY
    then echo TRAINED; return; fi
  fi
  echo FAIL
}

# run_cell TAG MODEL SYSTOKEN SEQ "B1 B2 ..." [POLICY] [RANKS] -> walks batches, first TRAINED wins
run_cell() { local tag="$1" model="$2" systok="$3" seq="$4" blist="$5" policy="${6:-none|false|false|false|false|false}" ranks="${7:-1}" v=SKIP b
  for b in $blist; do
    guard || return 1
    echo "START $tag $model $systok s=$seq b=$b r=$ranks $(date +%H:%M)" >> "$S"
    : > "$LOGD/r_std_${tag}_b${b}.log"   # fresh log per attempt — stale-log verdict trap (TPFIG lesson #2)
    RUN_NAME="${tag}-c14s_${model}" RUNS="${model}|${ranks} ; ${systok}|ligerloss1 ; ${seq}|${b}|1 ; ${policy}" \
      bash scripts/lf/profile_lora_lf_test_source.sh >> "$LOGD/r_std_${tag}_b${b}.log" 2>&1
    local dmodel=${model//./_}
    v=$(verdict "${tag}-c14s_${dmodel}__b${b}_s${seq}_ga1_drop000" "$LOGD/r_std_${tag}_b${b}.log")
    echo "CELL $tag $systok s=$seq b=$b -> $v $(date +%H:%M)" >> "$S"
    [ "$v" = "TRAINED" ] && break
    [ "$v" = "FAIL" ] && break
  done; echo "$v"; }


# ooms V -> true only for real OOM verdicts (tier/batch ladders walk on these;
# FAIL = infra problem -> abort the phase, never mislabel)
ooms() { [ "$1" = "GOOM" ] || [ "$1" = "COOM" ]; }

# harvest TAG MODEL SEQ B RANKS -> "lat tp hbm hbm% rss spread" from the run dir
harvest() { local tag="$1" model="$2" seq="$3" b="$4" ranks="${5:-1}"
  local dmodel=${model//./_}
  python3 scripts/lf/parse_fill_cell.py "$B/${tag}-c14s_${dmodel}__b${b}_s${seq}_ga1_drop000" "$ranks" "$seq" "$b" 2>/dev/null | tail -1; }
