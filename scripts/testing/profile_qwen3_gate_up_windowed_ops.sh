#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-1}"
CASES="${CASES:-tiny}"
MODES="${MODES:-cache_first_window}"
P_VALUES="${P_VALUES:-}"
Q_VALUES="${Q_VALUES:-}"
BM_VALUES="${BM_VALUES:-}"
BK_VALUES="${BK_VALUES:-}"
G_WORK_VALUES="${G_WORK_VALUES:-}"
WITH_DOWN_LORA_A="${WITH_DOWN_LORA_A:-1}"
R_DOWN="${R_DOWN:-8}"
DOWN_DROPOUT_P="${DOWN_DROPOUT_P:-0.0}"
WARMUP_ITERS="${WARMUP_ITERS:-3}"
LATENCY_ITERS="${LATENCY_ITERS:-10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-profiling_results/profiling/qwen3_gate_up_windowed_bwd/op_profile}"

IFS=',' read -r -a CASE_LIST <<< "${CASES}"
IFS=',' read -r -a MODE_LIST <<< "${MODES}"

for case_name in "${CASE_LIST[@]}"; do
  for mode in "${MODE_LIST[@]}"; do
    args=(
      scripts/lf/validate_qwen3_gate_up_windowed_bwd.py
      --stage "op_profile"
      --op "native_e2e"
      --case "${case_name}"
      --device "cuda:0"
      --mode "${mode}"
      --warmup-iters "${WARMUP_ITERS}"
      --latency-iters "${LATENCY_ITERS}"
      --output-dir "${OUTPUT_ROOT}/${case_name}/${mode}"
    )
    [[ -n "${P_VALUES}" ]] && args+=(--p "${P_VALUES%%,*}")
    [[ -n "${Q_VALUES}" ]] && args+=(--q "${Q_VALUES%%,*}")
    [[ -n "${BM_VALUES}" ]] && args+=(--bm "${BM_VALUES%%,*}")
    [[ -n "${BK_VALUES}" ]] && args+=(--bk "${BK_VALUES%%,*}")
    [[ -n "${G_WORK_VALUES}" ]] && args+=(--g-work "${G_WORK_VALUES%%,*}")
    if [[ "${WITH_DOWN_LORA_A}" == "1" || "${WITH_DOWN_LORA_A}" == "true" ]]; then
      args+=(--with-down-lora-a --r-down "${R_DOWN}" --down-dropout-p "${DOWN_DROPOUT_P}")
    fi

    CUDA_VISIBLE_DEVICES="${GPU}" python "${args[@]}"
  done
done
