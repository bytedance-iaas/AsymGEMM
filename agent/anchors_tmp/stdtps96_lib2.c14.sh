#!/bin/bash
# stdtps96_lib2.sh — METHOD V2 (peak-audit) lib for c14 (standardize_tps_96gb
# OPS LESSONS adopted 2026-08-21 23:5x): cells run on the FULL 185G card (no
# occupier), verdict from the allocator-tracked post-run peak reserved:
#   <= 92.5 GiB -> TRAINED_FIT96 · <= 95.6 -> TRAINED_EDGE96 · else OVER96
# (treated as the 96G G-OOM in walks/ladders — no violent teardown).
# Termination ladder everywhere (NEVER lead with -9). Infra verdicts
# FAIL-STOP (never demote). OCC_PIDS optional ("" for V2 mode).
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512 DATASET_OVERWRITE=false OVERWRITE=false
export ASYM_ZERO_ROUTER_JITTER=1 TRUST_REMOTE_CODE=false
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN=""   # shared token EXPIRED (D's flag)
export ALLOW_CROSS_SUPERCHIP=1                        # per-socket sim pair (D's flag)
export NUMACTL_ENABLE=1 NUMACTL_MODE=membind NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1
export CUDA_VISIBLE_DEVICES=${GPU:?}
LOGD=/workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp
S="$LOGD/stdtps96_status.log"
B=profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16
PA_FIT_GIB=${PA_FIT_GIB:-92.5}
PA_BUDGET_GIB=${PA_BUDGET_GIB:-95.6}
OCC_PIDS=${OCC_PIDS:-}

is_occ() { local q; for q in $OCC_PIDS; do [ "$1" = "$q" ] && return 0; done; return 1; }

# termination ladder: INT -> 10s -> TERM -> 15s -> 9 (Kevin's rule)
ladder_kill() { local p="$1" why="${2:-}"
  [ -d "/proc/$p" ] || return 0
  kill -INT "$p" 2>/dev/null; local w
  for w in 1 2; do sleep 5; [ -d "/proc/$p" ] || { echo "LKILL $p clean-INT $why $(date +%H:%M)" >> "$S"; return 0; }; done
  kill -TERM "$p" 2>/dev/null
  for w in 1 2 3; do sleep 5; [ -d "/proc/$p" ] || { echo "LKILL $p clean-TERM $why $(date +%H:%M)" >> "$S"; return 0; }; done
  kill -9 "$p" 2>/dev/null; echo "LKILL $p HARD (INT+TERM ignored) $why $(date +%H:%M)" >> "$S"; }

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
        [ "$pp" = "1" ] && ladder_kill "$p" "orphan-gpu$GPU"
      done
    fi
    sleep 20
  done; echo "GUARD-TIMEOUT gpu=$GPU $(date +%H:%M)" >> "$S"; return 1; }

_peak_gib() { python3 - "$1" <<'PY'
import glob, json, sys
pks = []
for f in glob.glob(sys.argv[1] + "/**/profile.json", recursive=True):
    try:
        m = json.load(open(f)).get("memory", {})
        v = m.get("peak_reserved_hbm_bytes")
        if v: pks.append(v / 2**30)
    except Exception: pass
print(f"{max(pks):.1f}" if pks else "NA")
PY
}

verdict_pa() { local dtag="$1" log="$2"
  local tsv="$B/${dtag}/jobs.tsv"
  grep -aqE "OutOfMemoryError|CUDA out of memory" "$log" 2>/dev/null && { echo GOOM; return; }
  grep -aq "dropped below floor" "$log" 2>/dev/null && { echo COOM; return; }
  if [ -f "$tsv" ] && awk -F'\t' 'NR>1 && $1=="ok"{f=1} END{exit !f}' "$tsv"; then
    local pk; pk=$(_peak_gib "$B/${dtag}")
    [ "$pk" = "NA" ] && { echo FAIL; return; }
    awk -v p="$pk" -v f="$PA_FIT_GIB" -v b="$PA_BUDGET_GIB" 'BEGIN{
      if (p+0 <= f+0) print "TRAINED_FIT96 peak=" p;
      else if (p+0 <= b+0) print "TRAINED_EDGE96 peak=" p;
      else print "OVER96 peak=" p; }'
    return
  fi
  echo FAIL
}

# run_cell_pa: batch walk — OVER96/GOOM/COOM step down; first FIT/EDGE wins;
# FAIL breaks (caller must FAIL-STOP, never demote).
run_cell_pa() { local tag="$1" model="$2" systok="$3" seq="$4" blist="$5" policy="${6:-none|false|false|false|false|false}" ranks="${7:-1}" v=SKIP b
  for b in $blist; do
    guard || { echo GUARDFAIL; return 1; }
    echo "START $tag $model $systok s=$seq b=$b r=$ranks PA $(date +%H:%M)" >> "$S"
    : > "$LOGD/r_96_${tag}_b${b}.log"
    RUN_NAME="${tag}-96c14_${model}" RUNS="${model}|${ranks} ; ${systok}|ligerloss1 ; ${seq}|${b}|1 ; ${policy}" \
      bash scripts/lf/profile_lora_lf_test_source.sh >> "$LOGD/r_96_${tag}_b${b}.log" 2>&1
    local dmodel=${model//./_}
    v=$(verdict_pa "${tag}-96c14_${dmodel}__b${b}_s${seq}_ga1_drop000" "$LOGD/r_96_${tag}_b${b}.log")
    echo "CELL $tag $systok s=$seq b=$b -> $v $(date +%H:%M)" >> "$S"
    case "$v" in TRAINED_FIT96*|TRAINED_EDGE96*|FAIL) break;; esac
  done; echo "$v"; }

harvest() { local tag="$1" model="$2" seq="$3" b="$4" ranks="${5:-1}"
  local dmodel=${model//./_}
  python3 scripts/lf/parse_fill_cell.py "$B/${tag}-96c14_${dmodel}__b${b}_s${seq}_ga1_drop000" "$ranks" "$seq" "$b" 2>/dev/null | tail -1; }
