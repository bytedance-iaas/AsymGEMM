#!/bin/bash
# stdtps96_lib_c17.sh — Session E (c17) run library, METHOD V2 PEAK-AUDIT
# (standardize_tps_96gb.md "METHOD V2"): cells run WITHOUT the occupier on
# the full 185G card; the 96G verdict comes from the EXACT post-run peak
# reserved HBM. Port of c11's stdtps96_lib2.sh; only workspace root, log
# dir, and node tag differ. TERMINATION LADDER (Kevin: NEVER lead with -9)
# is in _ladder_kill and in the cell timeout (INT-first).
# Verdicts: TRAINED_FIT96 (peak<=92.5G) | TRAINED_EDGE96 (<=95.6G, label)
#           | OVER96 (peak>95.6G — the 96G wall, inferred, run trained)
#           | GOOM/COOM (real 185G crash) | FAIL.
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512 DATASET_OVERWRITE=false OVERWRITE=false
export ASYM_ZERO_ROUTER_JITTER=1 TRUST_REMOTE_CODE=false
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
# Expired shared HF token 401s hub calls (D's 03:1x flag); all campaign
# models are public -> force anonymous hub access.
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN=""
export CUDA_VISIBLE_DEVICES=${GPU:?}
LOGD=/workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp
S="$LOGD/stdtps96_status.log"
B=profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16
PA_FIT_GIB=92.5
PA_BUDGET_GIB=95.6

# Kevin's rule: INT (KeyboardInterrupt -> full python unwind, torch/NCCL
# destructors) -> ~10s -> TERM (launcher-orchestrated shutdown) -> ~15s ->
# -9 ONLY if both were ignored (hard teardown of NVLink-mapped pinned
# memory is the suspected driver-wedge class).
_ladder_kill() { local p=$1
  kill -INT "$p" 2>/dev/null
  for w in 1 2; do sleep 5; [ -d "/proc/$p" ] || return 0; done
  kill -TERM "$p" 2>/dev/null
  for w in 1 2 3; do sleep 5; [ -d "/proc/$p" ] || return 0; done
  kill -9 "$p" 2>/dev/null; }

guard() { for i in $(seq 1 180); do
    n=0; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      p=${p//,/}; [ -d "/proc/$p" ] && n=$((n+1)); done
    if [ "$n" -eq 0 ] && ls /dev/shm/asym_* >/dev/null 2>&1; then
      rm -rf /dev/shm/asym_* 2>/dev/null || true
      echo "GUARD-SHM-CLEAN $(date +%H:%M)" >> "$S"
    fi
    a=$(free -g | awk 'NR==2{print $7}')
    if [ "$n" -eq 0 ] && [ "$a" -ge "${HOSTFLOOR:?}" ]; then return 0; fi
    if [ $((i % 9)) -eq 0 ]; then
      for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        p=${p//,/}
        pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d " ")
        [ "$pp" = "1" ] && _ladder_kill "$p" && echo "GUARD-REAPED orphan $p (ladder) $(date +%H:%M)" >> "$S"
      done
    fi
    sleep 20
  done; echo "GUARD-TIMEOUT gpu=$GPU $(date +%H:%M)" >> "$S"; return 1; }

shm_guard() { local used
  for i in $(seq 1 30); do
    used=$(df -BG /dev/shm | awk 'NR==2{gsub("G","",$3); print $3}')
    [ "${used:-0}" -lt 5 ] && return 0
    rm -rf /dev/shm/asym_* 2>/dev/null || true
    sleep 10
  done
  echo "STDTPS96 shm_guard timeout used=${used}G $(date +%H:%M)" >> "$S"; }

# peak (GiB, max over ranks) from every profile.json under the run dir
_peak_gib() { local dir="$1"
  /usr/bin/env python3 - "$dir" <<'PYEOF2'
import glob, json, sys
peaks = []
for pj in glob.glob(sys.argv[1] + "/**/profile.json", recursive=True):
    try:
        m = json.load(open(pj)).get("memory", {})
        peaks.append(m.get("peak_reserved_hbm_bytes", 0) / 2**30)
    except Exception:
        pass
print(f"{max(peaks):.2f}" if peaks else "NA")
PYEOF2
}

verdict_pa() { local dtag="$1" log="$2"
  local tsv="$B/${dtag}/jobs.tsv"
  grep -aqE "OutOfMemoryError|CUDA out of memory" "$log" 2>/dev/null && { echo GOOM; return; }
  grep -aq "dropped below floor" "$log" 2>/dev/null && { echo COOM; return; }
  if [ -f "$tsv" ] && awk -F'\t' 'NR>1 && $1=="ok"{f=1} END{exit !f}' "$tsv"; then
    local pk; pk=$(_peak_gib "$B/${dtag}")
    [ "$pk" = "NA" ] && { echo FAIL; return; }
    awk -v p="$pk" -v f="$PA_FIT_GIB" -v b="$PA_BUDGET_GIB" 'BEGIN{
      if (p <= f) print "TRAINED_FIT96 peak=" p;
      else if (p <= b) print "TRAINED_EDGE96 peak=" p;
      else print "OVER96 peak=" p; }'
    return
  fi
  echo FAIL
}

# run_cell_pa TAG MODEL SYSTOKEN SEQ "B1 B2 ..." [POLICY] [RANKS]
# batch walk: OVER96 (and GOOM) walk DOWN like an OOM; first FIT/EDGE wins.
# Walltime watchdog: CELL_TIMEOUT_MIN (default 150) with INT-first teardown
# (c14 step-0-hang precedent); a timed-out run has no OOM marker -> FAIL
# -> chain fail-stop for investigation, never an OOM verdict.
run_cell_pa() { local tag="$1" model="$2" systok="$3" seq="$4" blist="$5" policy="${6:-none|false|false|false|false|false}" ranks="${7:-1}" v=SKIP b
  for b in $blist; do
    guard || { echo GUARDFAIL; return 1; }
    echo "START $tag $model $systok s=$seq b=$b r=$ranks PA $(date +%H:%M)" >> "$S"
    RUN_NAME="${tag}-g96c17_${model}" RUNS="${model}|${ranks} ; ${systok}|ligerloss1 ; ${seq}|${b}|1 ; ${policy}" \
      timeout --signal=INT --kill-after=120s "$(( ${CELL_TIMEOUT_MIN:-150} * 60 ))s" \
      bash scripts/lf/profile_lora_lf_test_source.sh >> "$LOGD/r_${tag}_b${b}.log" 2>&1
    local dmodel=${model//./_}
    v=$(verdict_pa "${tag}-g96c17_${dmodel}__b${b}_s${seq}_ga1_drop000" "$LOGD/r_${tag}_b${b}.log")
    echo "CELL $tag $systok s=$seq b=$b -> $v $(date +%H:%M)" >> "$S"
    case "$v" in
      TRAINED_FIT96*|TRAINED_EDGE96*) break;;
      FAIL) break;;
    esac   # GOOM/COOM/OVER96 -> next (smaller) batch
  done; echo "$v"; }
