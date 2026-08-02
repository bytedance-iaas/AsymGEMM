#!/bin/bash
# tp_probe.sh — reusable throughput-gap probe executor (strategy lives in the agent/prompt).
# Usage (inside container):
#   bash scripts/lf/tp_probe.sh <model> <tag> <config-string> <seq> <b1> [b2 ...]
#   e.g. bash scripts/lf/tp_probe.sh llama3.3-70b tputrc "superoffload_mem|recomp|ligerloss1" 80000 1
#        bash scripts/lf/tp_probe.sh q3-32b tput "superoffload_mem|unsloth-ohbm0|ligerloss1" 128000 2 1
# Batches are tried IN ORDER (descending recommended): first FIT wins; OOM -> next; other
# failure -> HARDFAIL (aborts, prints cause). Exit: 0 fit, 1 all-OOM, 2 hardfail.
# Env overrides: PROFILERS MAX_STEPS WARMUP_STEPS MAX_SAMPLES DATASET_OVERWRITE OVERWRITE + any
# ASYM*/ASYMM* latency-mode vars (forwarded by the driver).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 3
export SFT_ROOT=${SFT_ROOT:-$(cd ../.. && pwd)}
# w1+m2 protocol (2026-07-19): 1 warmup + 2 measured steps. Post-warmup steps are stable
# (<~1% spread), so 2 samples suffice; saves ~2 step-times per run at long seq.
export PROFILERS=${PROFILERS:-source} MAX_STEPS=${MAX_STEPS:-2} WARMUP_STEPS=${WARMUP_STEPS:-1}
export MAX_SAMPLES=${MAX_SAMPLES:-1024} DATASET_OVERWRITE=${DATASET_OVERWRITE:-false} OVERWRITE=${OVERWRITE:-false}
T="none|false|false|false|false|false"

model="$1"; tag="$2"; cfg="$3"; seq="$4"; shift 4
[ $# -ge 1 ] || { echo "usage: tp_probe.sh <model> <tag> <config> <seq> <b...>"; exit 3; }

# self-heal dataset registrations (LF git syncs wipe them -> validation_ok=False hardfails)
python3 "${SFT_ROOT}/.repair_dataset_info.py" >/dev/null 2>&1 || true

for b in "$@"; do
  echo "########## RUN ${model} ${tag} seq=${seq} b=${b} $(date -u +%H:%M:%S)"
  out=$(RUN_NAME="${tag}_${model}" RUNS="${model}|1 ; ${cfg} ; ${seq}|${b}|1 ; ${T}" \
    bash scripts/lf/profile_lora_lf_test_source.sh 2>&1)
  echo "$out" | tail -2
  if echo "$out" | grep -qiE "CUDA out of memory|OutOfMemoryError|HOST_OOM_EVIDENCE=true|host memory watchdog|failed with status 143"; then
    echo "@@@ OOM ${model} ${tag} ${seq} b=${b} -> next (GPU or HOST oom)"; continue
  fi
  if echo "$out" | grep -qiE "profiling job\(s\) failed|Training command failed|Traceback|^error:|error: RUNS"; then
    echo "@@@ HARDFAIL ${model} ${tag} ${seq} b=${b}"
    echo "$out" | grep -iE "Error|validation_ok|missing registration" | tail -4
    exit 2
  fi
  # FIT needs POSITIVE evidence, not just the absence of failure patterns: the driver
  # exits 0 even on failed jobs (CONTINUE_ON_ERROR), and an early arg error (e.g. unknown
  # model shorthand) matches nothing above. Trust this run's own jobs.tsv only.
  tagl=$(printf '%s_%s' "${tag}" "${model}" | tr 'A-Z.' 'a-z_')
  tsv=$(ls -t profiling_results/profiling*/asym_long_sft_smoke__lora__lf__bf16/"${tagl}"__b"${b}"_s"${seq}"_*/jobs.tsv 2>/dev/null | head -1)
  if [ -n "${tsv}" ] && awk -F'\t' 'NR>1 && $1=="ok"{f=1} END{exit !f}' "${tsv}"; then
    echo "@@@ FIT ${model} ${tag} ${seq} b=${b}"
    exit 0
  fi
  # Fallback: the driver marks a job failed:N on ANY nonzero trainer exit — including
  # dirty TEARDOWN after training fully completed (seen 2026-07-20: q3.5 T2/T3 runs
  # with complete artifacts marked failed). Trust the artifacts: a step_samples.json
  # containing >= MAX_STEPS measured rows next to this jobs.tsv means the run trained.
  if [ -n "${tsv}" ]; then
    ss=$(ls -t "$(dirname "${tsv}")"/*/b"${b}"_s"${seq}"_*/step_samples.json 2>/dev/null | head -1)
    if [ -n "${ss}" ] && python3 -c "
import json, sys
rows = json.load(open('${ss}'))
rows = rows if isinstance(rows, list) else rows.get('steps', [])
meas = [r for r in rows if str(r.get('is_warmup','')).lower() in ('false','0','')]
sys.exit(0 if len(meas) >= int('${MAX_STEPS}') else 1)
" 2>/dev/null; then
      echo "@@@ FIT ${model} ${tag} ${seq} b=${b} (artifacts-complete; jobs.tsv says failed = teardown-only)"
      exit 0
    fi
  fi
  echo "@@@ HARDFAIL ${model} ${tag} ${seq} b=${b} (no ok row in jobs.tsv: ${tsv:-missing})"
  echo "$out" | grep -iE "error" | tail -3
  exit 2
done
echo "@@@ ALL-OOM ${model} ${tag} ${seq}"
exit 1
