#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

# =============================================================================
# User Parameters
# =============================================================================
ROOT=${ROOT:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM}
LF_DIR=${LF_DIR:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory}
CONDA_EXE=${CONDA_EXE:-conda}
NSYS_BIN=${NSYS_BIN:-nsys}

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen3-30B-A3B}
BACKEND=${BACKEND:-asym}              # hf | asym_torch | asym
GPU_ID=${GPU_ID:-0}
REQUIRE_SM100=${REQUIRE_SM100:-1}

DATASET=${DATASET:-asym_long_sft_smoke}
TEMPLATE=${TEMPLATE:-qwen3_nothink}
CUTOFF_LEN=${CUTOFF_LEN:-4096}
MAX_SAMPLES=${MAX_SAMPLES:-64}

MAX_STEPS=${MAX_STEPS:-10}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}

ASYM_PRECISION=${ASYM_PRECISION:-bf16}
ASYM_OFFLOAD_MODULES=${ASYM_OFFLOAD_MODULES:-routed_experts}
ASYM_STRICT=${ASYM_STRICT:-true}

PROFILE=${PROFILE:-0}
PROFILE_PROFILER=${PROFILE_PROFILER:-source} # source | nsys
PROFILE_MEMORY=${PROFILE_MEMORY:-1}
PROFILE_SOURCE_JSON=${PROFILE_SOURCE_JSON:-}
PROFILE_NSYS_PREFIX=${PROFILE_NSYS_PREFIX:-}
PROFILE_WORKLOAD_LABEL=${PROFILE_WORKLOAD_LABEL:-}
PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-}
PROFILE_EXPERT_POLICY=${PROFILE_EXPERT_POLICY:-none}

# =============================================================================
# Derived Parameters
# =============================================================================
ASYM_DIR=${ASYM_DIR:-${ROOT}}
ENV_DIR=${ENV_DIR:-${LF_DIR}/.venv}
ENV_PYTHON=${ENV_PYTHON:-${ENV_DIR}/bin/python}
MODEL_TAG=$(basename "${MODEL_NAME_OR_PATH}" | tr '/:' '__')
RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
DEFAULT_RUN_ID="${RUN_TS}_${MODEL_TAG}_${BACKEND}_${ASYM_PRECISION}_ctx${CUTOFF_LEN}_bs${PER_DEVICE_TRAIN_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}_r${LORA_RANK}_a${LORA_ALPHA}_steps${MAX_STEPS}_offload${ASYM_OFFLOAD_MODULES}"
RUN_ID=${RUN_ID:-${DEFAULT_RUN_ID}}
OUT_DIR=${OUT_DIR:-${LF_DIR}/saves/asymgemm_smoke/${RUN_ID}}
LOG_FILE=${LOG_FILE:-${OUT_DIR}/train_${RUN_ID}.log}
LOSS_LOG_COPY=${LOSS_LOG_COPY:-${OUT_DIR}/loss_${RUN_ID}.trainer_log.jsonl}
DATASET_FILE="${LF_DIR}/data/${DATASET}.jsonl"
PROFILE_LAUNCHER=${PROFILE_LAUNCHER:-${ASYM_DIR}/scripts/lf/run_lf_profiled_train.py}

# =============================================================================
# Main Logic
# =============================================================================
if [[ "${BACKEND}" != "hf" && "${BACKEND}" != "asym_torch" && "${BACKEND}" != "asym" ]]; then
  echo "BACKEND must be one of: hf, asym_torch, asym" >&2
  exit 2
fi

if [[ "${PROFILE}" != "0" && "${PROFILE}" != "1" ]]; then
  echo "PROFILE must be 0 or 1" >&2
  exit 2
fi

if [[ "${PROFILE_PROFILER}" != "source" && "${PROFILE_PROFILER}" != "nsys" ]]; then
  echo "PROFILE_PROFILER must be one of: source, nsys" >&2
  exit 2
fi

if [[ ! -f "${DATASET_FILE}" ]]; then
  echo "Missing dataset ${DATASET_FILE}." >&2
  echo "Build it with:" >&2
  echo "  python ${ASYM_DIR}/scripts/lf/build_lf_long_sft_smoke_dataset.py --lf-dir ${LF_DIR} --model-name-or-path ${MODEL_NAME_OR_PATH} --cutoff-len ${CUTOFF_LEN} --num-samples ${MAX_SAMPLES}" >&2
  exit 2
fi

if [[ ! -d "${ENV_DIR}" ]]; then
  echo "Missing env ${ENV_DIR}. Run ${ASYM_DIR}/scripts/lf/bootstrap_lf_asym_env.sh first." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

if [[ "${PROFILE}" == "1" && ! -f "${PROFILE_LAUNCHER}" ]]; then
  echo "Missing profile launcher ${PROFILE_LAUNCHER}" >&2
  exit 2
fi

if [[ "${PROFILE}" == "1" && ! -x "${ENV_PYTHON}" ]]; then
  echo "Missing environment Python ${ENV_PYTHON}" >&2
  exit 2
fi

if [[ "${PROFILE}" == "1" && -z "${PROFILE_SOURCE_JSON}" ]]; then
  PROFILE_SOURCE_JSON="${OUT_DIR}/lf_source_profile.json"
fi

if [[ "${PROFILE}" == "1" && "${PROFILE_PROFILER}" == "nsys" && -z "${PROFILE_NSYS_PREFIX}" ]]; then
  PROFILE_NSYS_PREFIX="${OUT_DIR}/lf_trace"
fi

PY_CHECK='import torch, sys
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
major, minor = torch.cuda.get_device_capability(0)
print(f"CUDA device: {torch.cuda.get_device_name(0)} capability=sm_{major}{minor}")
'
CUDA_VISIBLE_DEVICES=${GPU_ID} "${CONDA_EXE}" run -p "${ENV_DIR}" python - <<PY
${PY_CHECK}
if "${BACKEND}" == "asym" and "${REQUIRE_SM100}" == "1":
    import torch
    major, minor = torch.cuda.get_device_capability(0)
    if major < 10:
        raise SystemExit(f"BACKEND=asym requires SM100-class GPU when REQUIRE_SM100=1, got sm_{major}{minor}")
PY

if [[ "${PROFILE}" == "1" ]]; then
  TRAIN_ENTRYPOINT=("${ENV_PYTHON}" "${PROFILE_LAUNCHER}")
  CMD_PREFIX=()
else
  TRAIN_ENTRYPOINT=(llamafactory-cli train)
  CMD_PREFIX=("${CONDA_EXE}" run -p "${ENV_DIR}")
fi

CMD=(
  "${CMD_PREFIX[@]}" "${TRAIN_ENTRYPOINT[@]}"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --trust_remote_code true
  --stage sft
  --do_train true
  --finetuning_type lora
  --lora_rank "${LORA_RANK}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --lora_target all
  --dataset "${DATASET}"
  --dataset_dir "${LF_DIR}/data"
  --template "${TEMPLATE}"
  --cutoff_len "${CUTOFF_LEN}"
  --max_samples "${MAX_SAMPLES}"
  --overwrite_cache true
  --preprocessing_num_workers 4
  --dataloader_num_workers 2
  --output_dir "${OUT_DIR}"
  --logging_steps 1
  --logging_first_step true
  --save_strategy no
  --eval_strategy no
  --report_to none
  --plot_loss false
  --overwrite_output_dir true
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --max_steps "${MAX_STEPS}"
  --lr_scheduler_type constant
  --warmup_steps 0
  --bf16 true
  --pure_bf16 true
)

case "${GRADIENT_CHECKPOINTING,,}" in
  1|true|yes|y|on) CMD+=(--gradient_checkpointing true --disable_gradient_checkpointing false) ;;
  0|false|no|n|off) CMD+=(--gradient_checkpointing false --disable_gradient_checkpointing true) ;;
  *) echo "GRADIENT_CHECKPOINTING must be true or false" >&2; exit 2 ;;
esac

if [[ "${BACKEND}" == "asym_torch" ]]; then
  CMD+=(--use_asym_gemm true --asym_backend torch --asym_precision "${ASYM_PRECISION}")
  CMD+=(--asym_offload_modules "${ASYM_OFFLOAD_MODULES}" --asym_strict "${ASYM_STRICT}")
elif [[ "${BACKEND}" == "asym" ]]; then
  CMD+=(--use_asym_gemm true --asym_backend asym --asym_precision "${ASYM_PRECISION}")
  CMD+=(--asym_offload_modules "${ASYM_OFFLOAD_MODULES}" --asym_strict "${ASYM_STRICT}")
fi

echo "RUN_ID=${RUN_ID}" | tee "${LOG_FILE}"
echo "OUT_DIR=${OUT_DIR}" | tee -a "${LOG_FILE}"
echo "PROFILE=${PROFILE}" | tee -a "${LOG_FILE}"
if [[ "${PROFILE}" == "1" ]]; then
  echo "PROFILE_PROFILER=${PROFILE_PROFILER}" | tee -a "${LOG_FILE}"
  echo "PROFILE_SOURCE_JSON=${PROFILE_SOURCE_JSON}" | tee -a "${LOG_FILE}"
  [[ -n "${PROFILE_NSYS_PREFIX}" ]] && echo "PROFILE_NSYS_PREFIX=${PROFILE_NSYS_PREFIX}" | tee -a "${LOG_FILE}"
fi

RUN_ENV=(
  CUDA_VISIBLE_DEVICES="${GPU_ID}"
  PYTHONPATH="${ASYM_DIR}:${LF_DIR}/src:${PYTHONPATH:-}"
)
if [[ "${BACKEND}" == "hf" ]]; then
  :
else
  RUN_ENV+=(USE_ASYM_GEMM=1)
fi

if [[ "${PROFILE}" == "1" ]]; then
  RUN_ENV+=(
    ASYM_GEMM_LF_PROFILE_SOURCE_JSON="${PROFILE_SOURCE_JSON}"
    ASYM_GEMM_LF_PROFILE_MEMORY="${PROFILE_MEMORY}"
    ASYM_GEMM_LF_CONFIG_WORKLOAD="${PROFILE_WORKLOAD_LABEL:-${MODEL_TAG}}"
    ASYM_GEMM_LF_CONFIG_BACKEND="${PROFILE_BACKEND_LABEL:-${BACKEND/asym_torch/torch}}"
    ASYM_GEMM_LF_CONFIG_PRECISION="${ASYM_PRECISION}"
    ASYM_GEMM_LF_CONFIG_SEQ_LEN="${CUTOFF_LEN}"
    ASYM_GEMM_LF_CONFIG_ACTIVATION_RECOMPUTE="${GRADIENT_CHECKPOINTING}"
    ASYM_GEMM_LF_CONFIG_EXPERT_POLICY="${PROFILE_EXPERT_POLICY}"
  )
fi

if [[ "${PROFILE}" == "1" && "${PROFILE_PROFILER}" == "nsys" ]]; then
  env "${RUN_ENV[@]}" \
    "${NSYS_BIN}" profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --resolve-symbols=false \
    --wait=primary \
    --force-overwrite=true \
    --output="${PROFILE_NSYS_PREFIX}" \
    "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
else
  env "${RUN_ENV[@]}" "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
fi

if [[ -f "${OUT_DIR}/trainer_log.jsonl" ]]; then
  cp "${OUT_DIR}/trainer_log.jsonl" "${LOSS_LOG_COPY}"
  echo "Copied loss log to ${LOSS_LOG_COPY}" | tee -a "${LOG_FILE}"
else
  echo "trainer_log.jsonl was not found in ${OUT_DIR}" | tee -a "${LOG_FILE}"
  exit 1
fi

if [[ "${PROFILE}" == "1" && -f "${PROFILE_SOURCE_JSON}" ]]; then
  echo "Wrote LF source profile to ${PROFILE_SOURCE_JSON}" | tee -a "${LOG_FILE}"
fi
