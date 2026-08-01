#!/bin/bash
# tp_probe_fill.sh — placeholder-fill campaign probe (2026-07-28). Rank-aware variant of
# tp_probe.sh (which stays rank-1 canonical): adds <gpus> arg, artifacts-FIRST verdicts
# (the q3.5 T2/T3 dirty-teardown prints "Training command failed" after full training —
# rescue must run before the hardfail greps), freshness-bounded evidence (mtime >= probe
# start so stale dirs can't vouch), and a one-shot DATASET_OVERWRITE=true retry on the
# missing-registration hardfail (standing user rule 2026-07-22).
# Usage (inside container):
#   bash scripts/lf/tp_probe_fill.sh <model> <tag> <config-string> <seq> <gpus> <b1> [b2 ...]
# Batches tried in order: first FIT wins; OOM -> next; other failure -> HARDFAIL.
# Exit: 0 fit, 1 all-OOM, 2 hardfail.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 3
export SFT_ROOT=${SFT_ROOT:-$(cd ../.. && pwd)}
export PROFILERS=${PROFILERS:-source} MAX_STEPS=${MAX_STEPS:-2} WARMUP_STEPS=${WARMUP_STEPS:-1}
export MAX_SAMPLES=${MAX_SAMPLES:-1024} DATASET_OVERWRITE=${DATASET_OVERWRITE:-false} OVERWRITE=${OVERWRITE:-false}
T="none|false|false|false|false|false"

model="$1"; tag="$2"; cfg="$3"; seq="$4"; gpus="$5"; shift 5
[ $# -ge 1 ] || { echo "usage: tp_probe_fill.sh <model> <tag> <config> <seq> <gpus> <b...>"; exit 3; }
# FILL_POOL overrides GPU selection (2026-07-29 contention redo: singles on GPU0,
# pairs on 0,1 — driver requires same-superchip pairs, 0,3 is rejected).
pool="${FILL_POOL:-0}"; [ "${gpus}" = "2" ] && pool="${FILL_POOL:-0,1}"

python3 /workspace/AsymGEMM-SFT/.repair_dataset_info.py >/dev/null 2>&1 || true

batches=("$@")
i=0
ds_retried=0
while [ $i -lt ${#batches[@]} ]; do
  b="${batches[$i]}"
  start_ts=$(date +%s)
  echo "########## RUN ${model} r${gpus} ${tag} seq=${seq} b=${b} $(date -u +%H:%M:%S)"
  out=$(RUN_NAME="${tag}_${model}" GPU_POOL="${pool}" \
    RUNS="${model}|${gpus} ; ${cfg} ; ${seq}|${b}|1 ; ${T}" \
    bash scripts/lf/profile_lora_lf_test_source.sh 2>&1)
  echo "$out" | tail -2
  tagl=$(printf '%s_%s' "${tag}" "${model}" | tr 'A-Z.' 'a-z_')
  tsv=$(ls -t profiling_results/profiling*/asym_long_sft_smoke__lora__lf__bf16/"${tagl}"__b"${b}"_s"${seq}"_*/jobs.tsv 2>/dev/null | head -1)
  fresh=""
  if [ -n "${tsv}" ] && [ "$(stat -c %Y "${tsv}")" -ge "${start_ts}" ]; then fresh=1; fi
  # 1) positive FIT evidence (this run's jobs.tsv only)
  if [ -n "${fresh}" ] && awk -F'\t' 'NR>1 && $1=="ok"{f=1} END{exit !f}' "${tsv}"; then
    echo "@@@ FIT ${model} r${gpus} ${tag} ${seq} b=${b}"
    echo "RUNDIR=$(dirname "${tsv}")"
    exit 0
  fi
  # 2) artifacts-complete rescue BEFORE any failure grep (teardown-only false-fails)
  if [ -n "${fresh}" ]; then
    ss=$(ls -t "$(dirname "${tsv}")"/*/b"${b}"_s"${seq}"_*/step_samples.json 2>/dev/null | head -1)
    if [ -n "${ss}" ] && [ "$(stat -c %Y "${ss}")" -ge "${start_ts}" ] && python3 -c "
import json, sys
rows = json.load(open('${ss}'))
rows = rows if isinstance(rows, list) else rows.get('steps', [])
meas = [r for r in rows if str(r.get('is_warmup','')).lower() in ('false','0','')]
sys.exit(0 if len(meas) >= int('${MAX_STEPS}') else 1)
" 2>/dev/null; then
      echo "@@@ FIT ${model} r${gpus} ${tag} ${seq} b=${b} (artifacts-complete; teardown-only failure)"
      echo "RUNDIR=$(dirname "${tsv}")"
      exit 0
    fi
  fi
  # 3) OOM -> next batch
  if echo "$out" | grep -qiE "CUDA out of memory|OutOfMemoryError|HOST_OOM_EVIDENCE=true|host memory watchdog|failed with status 143"; then
    echo "@@@ OOM ${model} r${gpus} ${tag} ${seq} b=${b} -> next (GPU or HOST oom)"
    i=$((i+1)); continue
  fi
  # 4) dataset-registration self-heal: retry SAME batch once with DATASET_OVERWRITE=true
  if [ "${ds_retried}" = "0" ] && echo "$out" | grep -qiE "validation_ok=False|missing registration|is not registered"; then
    echo "@@@ DATASET-RETRY ${model} ${tag} ${seq} b=${b} (DATASET_OVERWRITE=true, same batch)"
    export DATASET_OVERWRITE=true; ds_retried=1; continue
  fi
  # 5) hardfail
  echo "@@@ HARDFAIL ${model} r${gpus} ${tag} ${seq} b=${b}"
  echo "$out" | grep -iE "Error|validation_ok|missing registration" | tail -4
  exit 2
done
echo "@@@ ALL-OOM ${model} r${gpus} ${tag} ${seq}"
exit 1
