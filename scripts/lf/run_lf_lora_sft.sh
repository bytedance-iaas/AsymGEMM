#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

# User parameters.
ROOT=${ROOT:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM}
LF_DIR=${LF_DIR:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory}
ASYM_DIR=${ASYM_DIR:-${ROOT}}
ENV_DIR=${ENV_DIR:-${LF_DIR}/.venv}
CONDA_EXE=${CONDA_EXE:-conda}

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

ASYM_PRECISION=${ASYM_PRECISION:-bf16}
ASYM_OFFLOAD_MODULES=${ASYM_OFFLOAD_MODULES:-routed_experts}
ASYM_STRICT=${ASYM_STRICT:-true}

# Derived parameters.
MODEL_TAG=$(basename "${MODEL_NAME_OR_PATH}" | tr '/:' '__')
RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ID="${RUN_TS}_${MODEL_TAG}_${BACKEND}_${ASYM_PRECISION}_ctx${CUTOFF_LEN}_bs${PER_DEVICE_TRAIN_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}_r${LORA_RANK}_a${LORA_ALPHA}_steps${MAX_STEPS}_offload${ASYM_OFFLOAD_MODULES}"
OUT_DIR="${LF_DIR}/saves/asymgemm_smoke/${RUN_ID}"
LOG_FILE="${OUT_DIR}/train_${RUN_ID}.log"
LOSS_LOG_COPY="${OUT_DIR}/loss_${RUN_ID}.trainer_log.jsonl"
DATASET_FILE="${LF_DIR}/data/${DATASET}.jsonl"

if [[ "${BACKEND}" != "hf" && "${BACKEND}" != "asym_torch" && "${BACKEND}" != "asym" ]]; then
  echo "BACKEND must be one of: hf, asym_torch, asym" >&2
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

CMD=(
  "${CONDA_EXE}" run -p "${ENV_DIR}" llamafactory-cli train
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

if [[ "${BACKEND}" == "asym_torch" ]]; then
  CMD+=(--use_asym_gemm true --asym_backend torch --asym_precision "${ASYM_PRECISION}")
  CMD+=(--asym_offload_modules "${ASYM_OFFLOAD_MODULES}" --asym_strict "${ASYM_STRICT}")
elif [[ "${BACKEND}" == "asym" ]]; then
  CMD+=(--use_asym_gemm true --asym_backend asym --asym_precision "${ASYM_PRECISION}")
  CMD+=(--asym_offload_modules "${ASYM_OFFLOAD_MODULES}" --asym_strict "${ASYM_STRICT}")
fi

echo "RUN_ID=${RUN_ID}" | tee "${LOG_FILE}"
echo "OUT_DIR=${OUT_DIR}" | tee -a "${LOG_FILE}"

if [[ "${BACKEND}" == "hf" ]]; then
  env CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${ASYM_DIR}:${LF_DIR}/src:${PYTHONPATH:-}" \
    "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
else
  env CUDA_VISIBLE_DEVICES="${GPU_ID}" USE_ASYM_GEMM=1 PYTHONPATH="${ASYM_DIR}:${LF_DIR}/src:${PYTHONPATH:-}" \
    "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
fi

if [[ -f "${OUT_DIR}/trainer_log.jsonl" ]]; then
  cp "${OUT_DIR}/trainer_log.jsonl" "${LOSS_LOG_COPY}"
  echo "Copied loss log to ${LOSS_LOG_COPY}" | tee -a "${LOG_FILE}"
else
  echo "trainer_log.jsonl was not found in ${OUT_DIR}" | tee -a "${LOG_FILE}"
  exit 1
fi
