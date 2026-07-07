#!/usr/bin/env bash
# D1 contention probe (gb200_dp.md Stage D1): two INDEPENDENT |1 asym jobs side-by-side on
# GPU0/GPU1 (same model/row/seed). Deliberately NOT DP — no sync — so any degradation vs a solo
# |1 run is pure shared-Grace {DRAM/capacity/cores} contention. Artifacts are tagged dp2_probe
# and are NEVER presented as a DP training row.
# Children launch through scripts/lf/profile_lora_lf_test_source.sh (HC3: watchdog +
# oom_score_adj guards stay wired). HC5: per-rank pinned-pool cap predeclared below.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASYM_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="q3-32b"
ROW="20000|4|1"
BACKEND="asym_cpuadamwds"
RECOMP="recomp-off-full-fg-ker000"
LOSS="ligerloss1"
OUTPUT_ROOT="profiling_gb200dp_d1"
MAX_STEPS="${MAX_STEPS:-3}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
# HC5 predeclaration: per-rank pinned CPU activation pool cap (bytes). 96 GiB/rank.
POOL_CAP_BYTES="${POOL_CAP_BYTES:-$((96 * 1024 * 1024 * 1024))}"

usage() {
  cat <<USAGE
Usage: run_dp2_pair.sh [--model M] [--row seq|b|ga] [--backend B] [--recomp R] [--loss L] [--output-root DIR]
Launches two independent |1 rows concurrently on GPUs ${GPU_A} and ${GPU_B}, waits for both,
then writes DIR/dp2_probe_merged.json via aggregate_dp_ranks.py.
USAGE
}

while (($#)); do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --row) ROW="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --recomp) RECOMP="$2"; shift 2 ;;
    --loss) LOSS="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

RUNS_ROW="${MODEL}|1 ; ${BACKEND}|${RECOMP}|${LOSS} ; ${ROW} ; none|false|false|false|false|false"
mkdir -p "${OUTPUT_ROOT}"

# PREPARE_DATASETS=false skips the per-model dataset-name derivation, so pass the ALREADY
# PREPARED suffixed dataset explicitly (built by the solo reference run; also avoids the
# two children racing on a concurrent dataset build).
model_tag() {
  case "$1" in
    q3-32b) echo "qwen3-32b" ;;
    q2.5-72b) echo "qwen2_5-72b-instruct" ;;
    llama3.3-70b) echo "llama-3_3-70b-instruct" ;;
    q3-30b-a3b) echo "qwen3-30b-a3b" ;;
    *) echo "" ;;
  esac
}
SEQ_LEN="${ROW%%|*}"
TAG="$(model_tag "${MODEL}")"
[[ -n "${TAG}" ]] || { echo "[dp2_pair] no dataset tag mapping for model '${MODEL}'" >&2; exit 2; }
DATASET_NAME="asym_long_sft_smoke__${TAG}__s${SEQ_LEN}"
DATASET_JSONL="${ASYM_DIR}/../LlamaFactory/data/${DATASET_NAME}.jsonl"
[[ -f "${DATASET_JSONL}" ]] || { echo "[dp2_pair] prepared dataset missing: ${DATASET_JSONL} (run the solo reference first)" >&2; exit 2; }

run_rank() {
  # runs in the FOREGROUND of its caller; the caller backgrounds it so $! is a direct
  # child of the main shell (command-substituted launches are un-waitable — reparented).
  local gpu="$1" tag="$2"
  local root="${OUTPUT_ROOT}/dp2_probe_rank${tag}"
  mkdir -p "${root}"
  RUNS="${RUNS_ROW}" \
  OUTPUT_ROOT="${root}" \
  RUNS_LOG="${root}/runs.log" \
  MAX_STEPS="${MAX_STEPS}" WARMUP_STEPS="${WARMUP_STEPS}" \
  PLOT=false RUN_POST=false PREPARE_DATASETS=false \
  DATASET="${DATASET_NAME}" \
  ASYM_EXPACT_CPU_POOL_MAX_BYTES="${POOL_CAP_BYTES}" \
    bash "${SCRIPT_DIR}/profile_lora_lf_test_source.sh" --gpus "${gpu}" --overwrite false \
    > "${root}/driver.log" 2>&1
}

echo "[dp2_pair] row: ${RUNS_ROW}"
echo "[dp2_pair] dataset: ${DATASET_NAME}"
echo "[dp2_pair] pool cap per rank: ${POOL_CAP_BYTES} bytes"
run_rank "${GPU_A}" 0 &
pid_a=$!
run_rank "${GPU_B}" 1 &
pid_b=$!
echo "[dp2_pair] rank0 pid=${pid_a} (gpu ${GPU_A}), rank1 pid=${pid_b} (gpu ${GPU_B})"

status=0
wait "${pid_a}" || { echo "[dp2_pair] rank0 FAILED" >&2; status=1; }
wait "${pid_b}" || { echo "[dp2_pair] rank1 FAILED" >&2; status=1; }

"${ASYM_DIR}/.venv/bin/python" "${SCRIPT_DIR}/aggregate_dp_ranks.py" \
  --rank-dir "${OUTPUT_ROOT}/dp2_probe_rank0" \
  --rank-dir "${OUTPUT_ROOT}/dp2_probe_rank1" \
  --label "dp2_probe ${MODEL} ${ROW} ${BACKEND}" \
  --out "${OUTPUT_ROOT}/dp2_probe_merged.json"

echo "[dp2_pair] merged -> ${OUTPUT_ROOT}/dp2_probe_merged.json (status ${status})"
exit "${status}"
