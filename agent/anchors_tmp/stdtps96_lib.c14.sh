#!/bin/bash
# stdtps96_lib.sh — GH200-96GB sim campaign lib (standardize_tps_96gb.md),
# c14 / SFT-38 tree in the asym_sft_46 rootfs. Port of stdtp_lib.sh with the
# occupier-aware guard: OCC_PIDS (space-sep host pids) are whitelisted GPU
# holders — preflight = "no compute apps except occupiers", reaper never
# touches them, and every cell checks occupier liveness before AND after
# (an occupier death mid-cell invalidates the cell — rule "budget was open").
# Env expected: GPU (inside index or "0,1"), HOSTFLOOR, OCC_PIDS.
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512 DATASET_OVERWRITE=false OVERWRITE=false
export ASYM_ZERO_ROUTER_JITTER=1 TRUST_REMOTE_CODE=false
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export NUMACTL_ENABLE=1 NUMACTL_MODE=membind NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1
export CUDA_VISIBLE_DEVICES=${GPU:?}
LOGD=/workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp
S="$LOGD/stdtps96_status.log"
B=profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16

occ_ok() { local p ok=1
  for p in ${OCC_PIDS:?}; do [ -d "/proc/$p" ] || { echo "OCC-DEAD $p $(date +%H:%M)" >> "$S"; ok=0; }; done
  [ "$ok" = 1 ]; }

is_occ() { local q; for q in ${OCC_PIDS:?}; do [ "$1" = "$q" ] && return 0; done; return 1; }

guard() { for i in $(seq 1 180); do
    n=0; for p in $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      p=${p//,/}; [ -d "/proc/$p" ] && ! is_occ "$p" && n=$((n+1)); done
    if [ "$n" -eq 0 ] && ls /dev/shm/asym_* >/dev/null 2>&1; then
      echo "GUARD-SHM-CLEAN $(ls /dev/shm | head -3 | tr '\n' ' ') $(date +%H:%M)" >> "$S"
      rm -f /dev/shm/asym_* 2>/dev/null || true
    fi
    a=$(free -g | awk 'NR==2{print $7}')
    if [ "$n" -eq 0 ] && [ "$a" -ge "${HOSTFLOOR:?}" ]; then return 0; fi
    if [ $((i % 9)) -eq 0 ]; then
      for p in $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        p=${p//,/}; is_occ "$p" && continue
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

ooms() { [ "$1" = "GOOM" ] || [ "$1" = "COOM" ]; }

# run_cell TAG MODEL SYSTOKEN SEQ "B1 B2 ..." [POLICY] [RANKS] -> walks batches, first TRAINED wins.
# OCC-INVALID: occupier died during the attempt -> verdict OCCFAIL (cell must be rerun).
run_cell() { local tag="$1" model="$2" systok="$3" seq="$4" blist="$5" policy="${6:-none|false|false|false|false|false}" ranks="${7:-1}" v=SKIP b
  for b in $blist; do
    guard || return 1
    occ_ok || { echo "CELL $tag $systok s=$seq b=$b -> OCCFAIL(pre) $(date +%H:%M)" >> "$S"; echo OCCFAIL; return 1; }
    echo "START $tag $model $systok s=$seq b=$b r=$ranks $(date +%H:%M)" >> "$S"
    : > "$LOGD/r_96_${tag}_b${b}.log"
    RUN_NAME="${tag}-96c14_${model}" RUNS="${model}|${ranks} ; ${systok}|ligerloss1 ; ${seq}|${b}|1 ; ${policy}" \
      bash scripts/lf/profile_lora_lf_test_source.sh >> "$LOGD/r_96_${tag}_b${b}.log" 2>&1
    local dmodel=${model//./_}
    v=$(verdict "${tag}-96c14_${dmodel}__b${b}_s${seq}_ga1_drop000" "$LOGD/r_96_${tag}_b${b}.log")
    occ_ok || { echo "CELL $tag $systok s=$seq b=$b -> OCCFAIL(post, raw=$v) $(date +%H:%M)" >> "$S"; echo OCCFAIL; return 1; }
    echo "CELL $tag $systok s=$seq b=$b -> $v $(date +%H:%M)" >> "$S"
    [ "$v" = "TRAINED" ] && break
    [ "$v" = "FAIL" ] && break
  done; echo "$v"; }

harvest() { local tag="$1" model="$2" seq="$3" b="$4" ranks="${5:-1}"
  local dmodel=${model//./_}
  python3 scripts/lf/parse_fill_cell.py "$B/${tag}-96c14_${dmodel}__b${b}_s${seq}_ga1_drop000" "$ranks" "$seq" "$b" 2>/dev/null | tail -1; }
