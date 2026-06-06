#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

# =============================================================================
# User Parameters
# =============================================================================
ROOT=${ROOT:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM}
LF_DIR=${LF_DIR:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory}
KT_KERNEL_DIR=${KT_KERNEL_DIR:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/ktransformers/kt-kernel}
CONDA_EXE=${CONDA_EXE:-conda}
NSYS_BIN=${NSYS_BIN:-nsys}

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen3-30B-A3B}
BACKEND=${BACKEND:-asym}              # torch | asym | kt_torchbf16 | kt_armbf16
GPU_ID=${GPU_ID:-0}
NUM_GPUS=${NUM_GPUS:-1}
REQUIRE_SM100=${REQUIRE_SM100:-1}
TORCH_DISTRIBUTED_BACKEND=${TORCH_DISTRIBUTED_BACKEND:-deepspeed} # deepspeed | fsdp2 | ddp
TORCH_FSDP_CONFIG=${TORCH_FSDP_CONFIG:-${LF_DIR}/examples/accelerate/fsdp2_config.yaml}
TORCH_DEEPSPEED_CONFIG=${TORCH_DEEPSPEED_CONFIG:-${LF_DIR}/examples/deepspeed/ds_z3_config.json}

DATASET=${DATASET:-asym_long_sft_smoke}
TEMPLATE=${TEMPLATE:-auto}
CUTOFF_LEN=${CUTOFF_LEN:-4096}
MAX_SAMPLES=${MAX_SAMPLES:-64}

MAX_STEPS=${MAX_STEPS:-10}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
SEED=${SEED:-42}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}

ASYM_PRECISION=${ASYM_PRECISION:-bf16}
KT_PRECISION=${KT_PRECISION:-${ASYM_PRECISION}}
ASYM_OFFLOAD_MODULES=${ASYM_OFFLOAD_MODULES:-routed_experts}
ASYM_EXPERT_RECOMPUTE_POLICY=${ASYM_EXPERT_RECOMPUTE_POLICY:-none}
ASYM_STRICT=${ASYM_STRICT:-true}
CHECK_ASYM_CALLS=${CHECK_ASYM_CALLS:-1}
TORCH_USE_ASYM_GEMM_LORA=${TORCH_USE_ASYM_GEMM_LORA:-false}
CHECK_KT_CALLS=${CHECK_KT_CALLS:-1}
KT_NUM_THREADS=${KT_NUM_THREADS:-}
KT_THREADPOOL_COUNT=${KT_THREADPOOL_COUNT:-}
KT_MAX_CACHE_DEPTH=${KT_MAX_CACHE_DEPTH:-2}
KT_SHARE_BACKWARD_BB=${KT_SHARE_BACKWARD_BB:-}
KT_SHARE_CACHE_POOL=${KT_SHARE_CACHE_POOL:-}
KT_TP_ENABLED=${KT_TP_ENABLED:-false}
KT_NUM_GPU_EXPERTS=${KT_NUM_GPU_EXPERTS:-}
KT_WEIGHT_PATH=${KT_WEIGHT_PATH:-}
KT_EXPERT_CHECKPOINT_PATH=${KT_EXPERT_CHECKPOINT_PATH:-}
KT_USE_LORA_EXPERTS=${KT_USE_LORA_EXPERTS:-}
KT_LORA_EXPERT_NUM=${KT_LORA_EXPERT_NUM:-}
KT_LORA_EXPERT_INTERMEDIATE_SIZE=${KT_LORA_EXPERT_INTERMEDIATE_SIZE:-}
KT_TORCHBF16_SFT_DEVICE=${KT_TORCHBF16_SFT_DEVICE:-cuda}
KT_ARM_OMP_NUM_THREADS=${KT_ARM_OMP_NUM_THREADS:-64}
KT_ARM_OMP_PROC_BIND=${KT_ARM_OMP_PROC_BIND:-close}
KT_ARM_OMP_PLACES=${KT_ARM_OMP_PLACES:-cores}

PROFILE=${PROFILE:-0}
PROFILE_PROFILER=${PROFILE_PROFILER:-source} # source | nsys
PROFILE_MEMORY=${PROFILE_MEMORY:-1}
PROFILE_LEVEL=${PROFILE_LEVEL:-op}           # stage | module | op | deep
PROFILE_LAYERS=${PROFILE_LAYERS:-all}
PROFILE_MEMORY_ATTRIBUTION=${PROFILE_MEMORY_ATTRIBUTION:-auto}
PROFILE_MEMORY_BREAKDOWN=${PROFILE_MEMORY_BREAKDOWN:-auto}
PROFILE_MEMORY_BREAKDOWN_INTERVAL=${PROFILE_MEMORY_BREAKDOWN_INTERVAL:-1}
PROFILE_MEMORY_BREAKDOWN_STEPS=${PROFILE_MEMORY_BREAKDOWN_STEPS:-}
PROFILE_MEMORY_BREAKDOWN_MODULES=${PROFILE_MEMORY_BREAKDOWN_MODULES:-attention,mlp,experts,lora,embedding,loss}
PROFILE_MEMORY_BREAKDOWN_OUTPUT=${PROFILE_MEMORY_BREAKDOWN_OUTPUT:-memory_breakdown}
PROFILE_SYNC=${PROFILE_SYNC:-0}
PROFILE_MODULE_FILTER=${PROFILE_MODULE_FILTER:-attention,mlp,experts,lora,optimizer,kt}
PROFILE_SOURCE_JSON=${PROFILE_SOURCE_JSON:-}
PROFILE_NSYS_PREFIX=${PROFILE_NSYS_PREFIX:-}
PROFILE_NSYS_SQLITE=${PROFILE_NSYS_SQLITE:-}
PROFILE_NSYS_CAPTURE_RANGE=${PROFILE_NSYS_CAPTURE_RANGE:-cudaProfilerApi} # cudaProfilerApi | none
PROFILE_NSYS_GPU_METRICS_DEVICES=${PROFILE_NSYS_GPU_METRICS_DEVICES:-${GPU_ID}}
PROFILE_JSON=${PROFILE_JSON:-}
PROFILE_SUMMARY_MD=${PROFILE_SUMMARY_MD:-}
PROFILE_OUTPUT_DIR=${PROFILE_OUTPUT_DIR:-}
PROFILE_WORKLOAD_LABEL=${PROFILE_WORKLOAD_LABEL:-}
PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-}
PROFILE_EXPERT_POLICY=${PROFILE_EXPERT_POLICY:-${ASYM_EXPERT_RECOMPUTE_POLICY}}

# =============================================================================
# Derived Parameters
# =============================================================================
ASYM_DIR=${ASYM_DIR:-${ROOT}}
KT_TOOLS_DIR=${KT_TOOLS_DIR:-${ASYM_DIR}}
ENV_DIR=${ENV_DIR:-${LF_DIR}/.venv}
ENV_PYTHON=${ENV_PYTHON:-${ENV_DIR}/bin/python}
TORCHRUN_BIN=${TORCHRUN_BIN:-${ENV_DIR}/bin/torchrun}
ACCELERATE_BIN=${ACCELERATE_BIN:-${ENV_DIR}/bin/accelerate}
unset KT_BACKEND                      # Not user-facing; derive the KT enum only from BACKEND.
KT_BACKEND_INTERNAL=""
case "${BACKEND,,}" in
  torch) BACKEND=torch ;;
  asym) BACKEND=asym ;;
  kt_torchbf16)
    BACKEND=kt_torchbf16
    KT_BACKEND_INTERNAL=TORCHBF16
    ;;
  kt_armbf16)
    BACKEND=kt_armbf16
    KT_BACKEND_INTERNAL=ARMBF16
    ;;
  *) echo "BACKEND must be one of: torch, asym, kt_torchbf16, kt_armbf16; got '${BACKEND}'" >&2; exit 2 ;;
esac

if [[ "${BACKEND}" == "torch" || "${BACKEND}" == kt_* ]]; then
  PROFILE_EXPERT_POLICY=none
fi

if [[ "${BACKEND}" == kt_* && "${NUM_GPUS}" != "1" ]]; then
  echo "KT SFT profiling is single-process/single-GPU for this script; got NUM_GPUS=${NUM_GPUS}" >&2
  exit 2
fi

if [[ ! "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer, got '${NUM_GPUS}'" >&2
  exit 2
fi

case "${TORCH_DISTRIBUTED_BACKEND,,}" in
  fsdp2) TORCH_DISTRIBUTED_BACKEND=fsdp2 ;;
  deepspeed|ds|zero3|z3) TORCH_DISTRIBUTED_BACKEND=deepspeed ;;
  ddp) TORCH_DISTRIBUTED_BACKEND=ddp ;;
  *) echo "TORCH_DISTRIBUTED_BACKEND must be one of: fsdp2, deepspeed, ddp; got '${TORCH_DISTRIBUTED_BACKEND}'" >&2; exit 2 ;;
esac

infer_template() {
  local model="$1"
  local lower="${model,,}"
  local base="${lower##*/}"
  case "${base}" in
    gemma-4-*|gemma4-*) printf 'gemma4\n' ;;
    llama-4-*) printf 'llama4\n' ;;
    qwen3-*|qwen3-next-*) printf 'qwen3_nothink\n' ;;
    *) printf 'qwen3_nothink\n' ;;
  esac
}

if [[ "${TEMPLATE}" == "auto" ]]; then
  TEMPLATE="$(infer_template "${MODEL_NAME_OR_PATH}")"
fi

MODEL_TAG=$(basename "${MODEL_NAME_OR_PATH}" | tr '/:' '__')
EXPERT_POLICY_TAG=$(printf '%s' "${ASYM_EXPERT_RECOMPUTE_POLICY}" | tr '/:' '__' | tr -c '[:alnum:]_-' '_')
RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
PROFILE_TAG="prof${PROFILE}_${PROFILE_PROFILER}_${PROFILE_LEVEL}"
if [[ "${BACKEND}" == kt_* ]]; then
  KT_BACKEND_TAG="${KT_BACKEND_INTERNAL:-none}"
  DEFAULT_RUN_ID="${RUN_TS}_${MODEL_TAG}_${BACKEND}_${KT_BACKEND_TAG}_${KT_PRECISION}_ctx${CUTOFF_LEN}_bs${PER_DEVICE_TRAIN_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}_r${LORA_RANK}_a${LORA_ALPHA}_steps${MAX_STEPS}_${PROFILE_TAG}"
else
  DEFAULT_RUN_ID="${RUN_TS}_${MODEL_TAG}_${BACKEND}_${ASYM_PRECISION}_ctx${CUTOFF_LEN}_bs${PER_DEVICE_TRAIN_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}_r${LORA_RANK}_a${LORA_ALPHA}_steps${MAX_STEPS}_offload${ASYM_OFFLOAD_MODULES}_pol${EXPERT_POLICY_TAG}_${PROFILE_TAG}"
fi
RUN_ID=${RUN_ID:-${DEFAULT_RUN_ID}}
if [[ "${BACKEND}" == kt_* ]]; then
  OUT_DIR=${OUT_DIR:-${LF_DIR}/saves/kt_smoke/${RUN_ID}}
else
  OUT_DIR=${OUT_DIR:-${LF_DIR}/saves/asymgemm_smoke/${RUN_ID}}
fi
LOG_FILE=${LOG_FILE:-${OUT_DIR}/train_${RUN_ID}.log}
LOSS_LOG_COPY=${LOSS_LOG_COPY:-${OUT_DIR}/loss_${RUN_ID}.trainer_log.jsonl}
DATASET_FILE="${LF_DIR}/data/${DATASET}.jsonl"
PROFILE_LAUNCHER=${PROFILE_LAUNCHER:-${ASYM_DIR}/scripts/lf/run_lf_profiled_train.py}
PROFILE_NSYS_POSTPROCESS_SCRIPT=${PROFILE_NSYS_POSTPROCESS_SCRIPT:-${ASYM_DIR}/scripts/lora/postprocess_nsys_lora.py}
PROFILE_POSTPROCESS_SCRIPT=${PROFILE_POSTPROCESS_SCRIPT:-${ASYM_DIR}/scripts/lf/postprocess_lf_profile_artifacts.py}

# =============================================================================
# Main Logic
# =============================================================================
bool_string() {
  local name="$1"
  local value="${2,,}"
  case "${value}" in
    1|true|yes|y|on) printf 'true\n' ;;
    0|false|no|n|off) printf 'false\n' ;;
    *) echo "${name} must be true or false, got '${2}'" >&2; exit 2 ;;
  esac
}

bool_01() {
  case "$(bool_string "$1" "$2")" in
    true) printf '1\n' ;;
    false) printf '0\n' ;;
  esac
}

optional_bool_string() {
  [[ -z "$2" ]] && return 0
  bool_string "$1" "$2"
}

positive_int_value() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "${name} must be a positive integer, got '${value}'" >&2; exit 2; }
}

nonnegative_int_value() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || { echo "${name} must be a non-negative integer, got '${value}'" >&2; exit 2; }
}

is_torch_distributed_run() {
  [[ "${BACKEND}" == "torch" && "${NUM_GPUS}" -gt 1 ]]
}

assert_deepspeed_scope() {
  local arg
  [[ "${BACKEND}" == "torch" ]] && return 0
  for arg in "${CMD_ARGS[@]}"; do
    if [[ "${arg}" == "--deepspeed" ]]; then
      echo "internal error: --deepspeed was added for BACKEND=${BACKEND}; DeepSpeed is restricted to BACKEND=torch" >&2
      exit 2
    fi
  done
}

if [[ "${PROFILE}" != "0" && "${PROFILE}" != "1" ]]; then
  echo "PROFILE must be 0 or 1" >&2
  exit 2
fi

if [[ "${CHECK_ASYM_CALLS}" != "0" && "${CHECK_ASYM_CALLS}" != "1" ]]; then
  echo "CHECK_ASYM_CALLS must be 0 or 1" >&2
  exit 2
fi

if [[ "${PROFILE_PROFILER}" != "source" && "${PROFILE_PROFILER}" != "nsys" ]]; then
  echo "PROFILE_PROFILER must be one of: source, nsys" >&2
  exit 2
fi

case "${PROFILE_LEVEL}" in
  stage|module|op|deep) ;;
  *) echo "PROFILE_LEVEL must be one of: stage, module, op, deep" >&2; exit 2 ;;
esac

case "${PROFILE_MEMORY_ATTRIBUTION,,}" in
  auto|0|1|true|false|yes|no|y|n|on|off) ;;
  *) echo "PROFILE_MEMORY_ATTRIBUTION must be auto, true, or false" >&2; exit 2 ;;
esac

case "${PROFILE_MEMORY_ATTRIBUTION,,}" in
  auto)
    if [[ "${PROFILE_PROFILER}" == "source" ]]; then
      PROFILE_MEMORY_ATTRIBUTION=true
    else
      PROFILE_MEMORY_ATTRIBUTION=false
    fi
    ;;
  1|true|yes|y|on) PROFILE_MEMORY_ATTRIBUTION=true ;;
  0|false|no|n|off) PROFILE_MEMORY_ATTRIBUTION=false ;;
esac

case "${PROFILE_MEMORY_BREAKDOWN,,}" in
  auto|0|1|true|false|yes|no|y|n|on|off) ;;
  *) echo "PROFILE_MEMORY_BREAKDOWN must be auto, true, or false" >&2; exit 2 ;;
esac

case "${PROFILE_MEMORY_BREAKDOWN,,}" in
  auto)
    if [[ "${PROFILE_PROFILER}" == "source" ]]; then
      PROFILE_MEMORY_BREAKDOWN=true
    else
      PROFILE_MEMORY_BREAKDOWN=false
    fi
    ;;
  1|true|yes|y|on) PROFILE_MEMORY_BREAKDOWN=true ;;
  0|false|no|n|off) PROFILE_MEMORY_BREAKDOWN=false ;;
esac

if [[ ! "${PROFILE_MEMORY_BREAKDOWN_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PROFILE_MEMORY_BREAKDOWN_INTERVAL must be a positive integer, got '${PROFILE_MEMORY_BREAKDOWN_INTERVAL}'" >&2
  exit 2
fi

case "${PROFILE_SYNC}" in
  0|1|true|false|yes|no|on|off) ;;
  *) echo "PROFILE_SYNC must be true or false" >&2; exit 2 ;;
esac
TORCH_USE_ASYM_GEMM_LORA="$(bool_string TORCH_USE_ASYM_GEMM_LORA "${TORCH_USE_ASYM_GEMM_LORA}")"
if [[ "${BACKEND}" == kt_* ]]; then
  CHECK_KT_CALLS="$(bool_01 CHECK_KT_CALLS "${CHECK_KT_CALLS}")"
  KT_TP_ENABLED="$(bool_string KT_TP_ENABLED "${KT_TP_ENABLED}")"
  KT_SHARE_BACKWARD_BB="$(optional_bool_string KT_SHARE_BACKWARD_BB "${KT_SHARE_BACKWARD_BB}")"
  KT_SHARE_CACHE_POOL="$(optional_bool_string KT_SHARE_CACHE_POOL "${KT_SHARE_CACHE_POOL}")"
  KT_USE_LORA_EXPERTS="$(optional_bool_string KT_USE_LORA_EXPERTS "${KT_USE_LORA_EXPERTS}")"
  [[ -z "${KT_NUM_THREADS}" ]] || positive_int_value KT_NUM_THREADS "${KT_NUM_THREADS}"
  [[ -z "${KT_THREADPOOL_COUNT}" ]] || positive_int_value KT_THREADPOOL_COUNT "${KT_THREADPOOL_COUNT}"
  positive_int_value KT_MAX_CACHE_DEPTH "${KT_MAX_CACHE_DEPTH}"
  positive_int_value KT_ARM_OMP_NUM_THREADS "${KT_ARM_OMP_NUM_THREADS}"
  [[ -z "${KT_NUM_GPU_EXPERTS}" ]] || nonnegative_int_value KT_NUM_GPU_EXPERTS "${KT_NUM_GPU_EXPERTS}"
  [[ -z "${KT_LORA_EXPERT_NUM}" ]] || positive_int_value KT_LORA_EXPERT_NUM "${KT_LORA_EXPERT_NUM}"
  [[ -z "${KT_LORA_EXPERT_INTERMEDIATE_SIZE}" ]] || positive_int_value KT_LORA_EXPERT_INTERMEDIATE_SIZE "${KT_LORA_EXPERT_INTERMEDIATE_SIZE}"
else
  CHECK_KT_CALLS=0
fi

if [[ ! -f "${DATASET_FILE}" ]]; then
  echo "Missing dataset ${DATASET_FILE}." >&2
  echo "Build it with:" >&2
  echo "  python ${ASYM_DIR}/scripts/lf/build_lf_sft_eval_pair.py --lf-dir ${LF_DIR} --asym-dir ${ASYM_DIR} --model-name-or-path ${MODEL_NAME_OR_PATH} --train-name ${DATASET} --eval-name ${DATASET}_eval --train-rows ${MAX_SAMPLES} --eval-rows 128 --cutoff-len ${CUTOFF_LEN}" >&2
  echo "If the dataset already exists and you only need stats/validation:" >&2
  echo "  python ${ASYM_DIR}/scripts/lf/build_lf_sft_eval_pair.py --lf-dir ${LF_DIR} --asym-dir ${ASYM_DIR} --model-name-or-path ${MODEL_NAME_OR_PATH} --train-name ${DATASET} --eval-name ${DATASET}_eval" >&2
  echo "To force rewriting an existing pair, add --overwrite." >&2
  exit 2
fi

if [[ ! -d "${ENV_DIR}" ]]; then
  echo "Missing env ${ENV_DIR}. Run ${ASYM_DIR}/scripts/lf/bootstrap_lf_asym_env.sh first." >&2
  exit 2
fi

if [[ "${PROFILE}" == "1" && ! -f "${PROFILE_LAUNCHER}" ]]; then
  echo "Missing profile launcher ${PROFILE_LAUNCHER}" >&2
  exit 2
fi

if [[ "${PROFILE}" == "1" && ! -x "${ENV_PYTHON}" ]]; then
  echo "Missing environment Python ${ENV_PYTHON}" >&2
  exit 2
fi

if [[ "${PROFILE}" == "1" ]] && is_torch_distributed_run && [[ "${TORCH_DISTRIBUTED_BACKEND}" == "ddp" && ! -x "${TORCHRUN_BIN}" ]]; then
  echo "Missing torchrun executable ${TORCHRUN_BIN}" >&2
  exit 2
fi

if is_torch_distributed_run && [[ "${TORCH_DISTRIBUTED_BACKEND}" == "fsdp2" ]]; then
  if [[ ! -x "${ACCELERATE_BIN}" ]]; then
    echo "Missing accelerate executable ${ACCELERATE_BIN}" >&2
    exit 2
  fi
  if [[ ! -f "${TORCH_FSDP_CONFIG}" ]]; then
    echo "Missing FSDP2 accelerate config ${TORCH_FSDP_CONFIG}" >&2
    exit 2
  fi
fi
if is_torch_distributed_run && [[ "${TORCH_DISTRIBUTED_BACKEND}" == "deepspeed" ]]; then
  if [[ ! -x "${ACCELERATE_BIN}" ]]; then
    echo "Missing accelerate executable ${ACCELERATE_BIN}" >&2
    exit 2
  fi
  if [[ ! -f "${TORCH_DEEPSPEED_CONFIG}" ]]; then
    echo "Missing DeepSpeed config ${TORCH_DEEPSPEED_CONFIG}" >&2
    exit 2
  fi
fi

if [[ "${PROFILE}" == "1" && -z "${PROFILE_SOURCE_JSON}" ]]; then
  PROFILE_SOURCE_JSON="${OUT_DIR}/source_profile.json"
fi

if [[ "${PROFILE}" == "1" && "${PROFILE_PROFILER}" == "nsys" && -z "${PROFILE_NSYS_PREFIX}" ]]; then
  PROFILE_NSYS_PREFIX="${OUT_DIR}/lf_trace"
fi
if [[ "${PROFILE}" == "1" && "${PROFILE_PROFILER}" == "nsys" && -z "${PROFILE_NSYS_SQLITE}" ]]; then
  PROFILE_NSYS_SQLITE="${PROFILE_NSYS_PREFIX}.sqlite"
fi
if [[ "${PROFILE}" == "1" && -z "${PROFILE_JSON}" ]]; then
  PROFILE_JSON="${OUT_DIR}/profile.json"
fi
if [[ "${PROFILE}" == "1" && -z "${PROFILE_OUTPUT_DIR}" ]]; then
  PROFILE_OUTPUT_DIR="$(dirname "${PROFILE_JSON}")"
fi
if [[ "${PROFILE}" == "1" && -z "${PROFILE_SUMMARY_MD}" ]]; then
  PROFILE_SUMMARY_MD="${OUT_DIR}/summary.md"
fi

mkdir -p "${OUT_DIR}" "$(dirname "${LOG_FILE}")"
: > "${LOG_FILE}"

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

if [[ "${BACKEND}" == kt_* ]]; then
  if [[ ! -d "${KT_KERNEL_DIR}" ]]; then
    echo "Missing integrated kt-kernel source: ${KT_KERNEL_DIR}" >&2
    exit 2
  fi
  env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTHONPATH="${KT_TOOLS_DIR}:${KT_KERNEL_DIR}:${LF_DIR}/src:${PYTHONPATH:-}" \
    "${ENV_PYTHON}" - <<'PY' | tee -a "${LOG_FILE}"
import kt_kernel
import torch

print("kt_kernel_variant", getattr(kt_kernel, "__cpu_variant__", "unknown"))
print("kt_kernel_file", getattr(kt_kernel, "__file__", "unknown"))
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY
fi

CMD_ARGS=(
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
  --seed "${SEED}"
  --bf16 true
)

if [[ "${BACKEND}" == "torch" && "${NUM_GPUS}" -gt 1 && "${TORCH_DISTRIBUTED_BACKEND}" == "deepspeed" ]]; then
  CMD_ARGS+=(--pure_bf16 false)
else
  CMD_ARGS+=(--pure_bf16 true)
fi

case "${GRADIENT_CHECKPOINTING,,}" in
  1|true|yes|y|on) CMD_ARGS+=(--gradient_checkpointing true --disable_gradient_checkpointing false) ;;
  0|false|no|n|off) CMD_ARGS+=(--gradient_checkpointing false --disable_gradient_checkpointing true) ;;
  *) echo "GRADIENT_CHECKPOINTING must be true or false" >&2; exit 2 ;;
esac

if [[ "${BACKEND}" == "torch" && "${TORCH_USE_ASYM_GEMM_LORA}" == "true" ]]; then
  CMD_ARGS+=(--use_asym_gemm true --asym_backend torch --asym_precision "${ASYM_PRECISION}")
  CMD_ARGS+=(--asym_offload_modules "${ASYM_OFFLOAD_MODULES}" --asym_strict "${ASYM_STRICT}")
  CMD_ARGS+=(--asym_expert_recompute_policy "${ASYM_EXPERT_RECOMPUTE_POLICY}")
elif [[ "${BACKEND}" == "asym" ]]; then
  CMD_ARGS+=(--use_asym_gemm true --asym_backend asym --asym_precision "${ASYM_PRECISION}")
  CMD_ARGS+=(--asym_offload_modules "${ASYM_OFFLOAD_MODULES}" --asym_strict "${ASYM_STRICT}")
  CMD_ARGS+=(--asym_expert_recompute_policy "${ASYM_EXPERT_RECOMPUTE_POLICY}")
elif [[ "${BACKEND}" == kt_* ]]; then
  case "${LORA_DROPOUT}" in
    0|0.0|0.00) ;;
    *) echo "KT SFT profiling requires LORA_DROPOUT=0.0 in the first ARM path; got ${LORA_DROPOUT}" >&2; exit 2 ;;
  esac
  CMD_ARGS+=(--use_kt true --kt_backend "${KT_BACKEND_INTERNAL}")
  [[ -z "${KT_NUM_THREADS}" ]] || CMD_ARGS+=(--kt_num_threads "${KT_NUM_THREADS}")
  [[ -z "${KT_THREADPOOL_COUNT}" ]] || CMD_ARGS+=(--kt_threadpool_count "${KT_THREADPOOL_COUNT}")
  [[ -z "${KT_MAX_CACHE_DEPTH}" ]] || CMD_ARGS+=(--kt_max_cache_depth "${KT_MAX_CACHE_DEPTH}")
  CMD_ARGS+=(--kt_tp_enabled "${KT_TP_ENABLED}")
  [[ -z "${KT_NUM_GPU_EXPERTS}" ]] || CMD_ARGS+=(--kt_num_gpu_experts "${KT_NUM_GPU_EXPERTS}")
  [[ -z "${KT_WEIGHT_PATH}" ]] || CMD_ARGS+=(--kt_weight_path "${KT_WEIGHT_PATH}")
  [[ -z "${KT_EXPERT_CHECKPOINT_PATH}" ]] || CMD_ARGS+=(--kt_expert_checkpoint_path "${KT_EXPERT_CHECKPOINT_PATH}")
  [[ -z "${KT_USE_LORA_EXPERTS}" ]] || CMD_ARGS+=(--kt_use_lora_experts "${KT_USE_LORA_EXPERTS}")
  [[ -z "${KT_LORA_EXPERT_NUM}" ]] || CMD_ARGS+=(--kt_lora_expert_num "${KT_LORA_EXPERT_NUM}")
  [[ -z "${KT_LORA_EXPERT_INTERMEDIATE_SIZE}" ]] || CMD_ARGS+=(--kt_lora_expert_intermediate_size "${KT_LORA_EXPERT_INTERMEDIATE_SIZE}")
fi

echo "RUN_ID=${RUN_ID}" | tee -a "${LOG_FILE}"
echo "OUT_DIR=${OUT_DIR}" | tee -a "${LOG_FILE}"
echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}" | tee -a "${LOG_FILE}"
echo "TEMPLATE=${TEMPLATE}" | tee -a "${LOG_FILE}"
echo "BACKEND=${BACKEND}" | tee -a "${LOG_FILE}"
echo "GPU_ID=${GPU_ID}" | tee -a "${LOG_FILE}"
echo "NUM_GPUS=${NUM_GPUS}" | tee -a "${LOG_FILE}"
echo "SEED=${SEED}" | tee -a "${LOG_FILE}"
if [[ "${BACKEND}" == kt_* ]]; then
  echo "KT_BACKEND=${KT_BACKEND_INTERNAL}" | tee -a "${LOG_FILE}"
  echo "KT_KERNEL_DIR=${KT_KERNEL_DIR}" | tee -a "${LOG_FILE}"
  echo "CHECK_KT_CALLS=${CHECK_KT_CALLS}" | tee -a "${LOG_FILE}"
  [[ "${BACKEND}" == "kt_torchbf16" ]] && echo "KT_TORCHBF16_SFT_DEVICE=${KT_TORCHBF16_SFT_DEVICE}" | tee -a "${LOG_FILE}"
  if [[ "${BACKEND}" == "kt_armbf16" ]]; then
    echo "KT_ARM_OMP_NUM_THREADS=${KT_ARM_OMP_NUM_THREADS}" | tee -a "${LOG_FILE}"
    echo "KT_ARM_OMP_PROC_BIND=${KT_ARM_OMP_PROC_BIND}" | tee -a "${LOG_FILE}"
    echo "KT_ARM_OMP_PLACES=${KT_ARM_OMP_PLACES}" | tee -a "${LOG_FILE}"
  fi
  [[ -n "${KT_NUM_THREADS}" ]] && echo "KT_NUM_THREADS=${KT_NUM_THREADS}" | tee -a "${LOG_FILE}"
  [[ -n "${KT_THREADPOOL_COUNT}" ]] && echo "KT_THREADPOOL_COUNT=${KT_THREADPOOL_COUNT}" | tee -a "${LOG_FILE}"
  [[ -n "${KT_MAX_CACHE_DEPTH}" ]] && echo "KT_MAX_CACHE_DEPTH=${KT_MAX_CACHE_DEPTH}" | tee -a "${LOG_FILE}"
  echo "KT_TP_ENABLED=${KT_TP_ENABLED}" | tee -a "${LOG_FILE}"
  [[ -n "${KT_SHARE_BACKWARD_BB}" ]] && echo "KT_SHARE_BACKWARD_BB=${KT_SHARE_BACKWARD_BB}" | tee -a "${LOG_FILE}"
  [[ -n "${KT_SHARE_CACHE_POOL}" ]] && echo "KT_SHARE_CACHE_POOL=${KT_SHARE_CACHE_POOL}" | tee -a "${LOG_FILE}"
  [[ -n "${KT_NUM_GPU_EXPERTS}" ]] && echo "KT_NUM_GPU_EXPERTS=${KT_NUM_GPU_EXPERTS}" | tee -a "${LOG_FILE}"
  [[ -n "${KT_WEIGHT_PATH}" ]] && echo "KT_WEIGHT_PATH=${KT_WEIGHT_PATH}" | tee -a "${LOG_FILE}"
  [[ -n "${KT_EXPERT_CHECKPOINT_PATH}" ]] && echo "KT_EXPERT_CHECKPOINT_PATH=${KT_EXPERT_CHECKPOINT_PATH}" | tee -a "${LOG_FILE}"
  [[ -n "${KT_USE_LORA_EXPERTS}" ]] && echo "KT_USE_LORA_EXPERTS=${KT_USE_LORA_EXPERTS}" | tee -a "${LOG_FILE}"
  [[ -n "${KT_LORA_EXPERT_NUM}" ]] && echo "KT_LORA_EXPERT_NUM=${KT_LORA_EXPERT_NUM}" | tee -a "${LOG_FILE}"
  [[ -n "${KT_LORA_EXPERT_INTERMEDIATE_SIZE}" ]] && echo "KT_LORA_EXPERT_INTERMEDIATE_SIZE=${KT_LORA_EXPERT_INTERMEDIATE_SIZE}" | tee -a "${LOG_FILE}"
fi
if is_torch_distributed_run; then
  echo "TORCH_DISTRIBUTED_BACKEND=${TORCH_DISTRIBUTED_BACKEND}" | tee -a "${LOG_FILE}"
  [[ "${TORCH_DISTRIBUTED_BACKEND}" == "fsdp2" ]] && echo "TORCH_FSDP_CONFIG=${TORCH_FSDP_CONFIG}" | tee -a "${LOG_FILE}"
  [[ "${TORCH_DISTRIBUTED_BACKEND}" == "deepspeed" ]] && echo "TORCH_DEEPSPEED_CONFIG=${TORCH_DEEPSPEED_CONFIG}" | tee -a "${LOG_FILE}"
fi
echo "ASYM_EXPERT_RECOMPUTE_POLICY=${ASYM_EXPERT_RECOMPUTE_POLICY}" | tee -a "${LOG_FILE}"
echo "TORCH_USE_ASYM_GEMM_LORA=${TORCH_USE_ASYM_GEMM_LORA}" | tee -a "${LOG_FILE}"
echo "PROFILE=${PROFILE}" | tee -a "${LOG_FILE}"
if [[ "${PROFILE}" == "1" ]]; then
  echo "PROFILE_PROFILER=${PROFILE_PROFILER}" | tee -a "${LOG_FILE}"
  echo "PROFILE_LEVEL=${PROFILE_LEVEL}" | tee -a "${LOG_FILE}"
  echo "PROFILE_LAYERS=${PROFILE_LAYERS}" | tee -a "${LOG_FILE}"
  echo "PROFILE_MEMORY_ATTRIBUTION=${PROFILE_MEMORY_ATTRIBUTION}" | tee -a "${LOG_FILE}"
  echo "PROFILE_MEMORY_BREAKDOWN=${PROFILE_MEMORY_BREAKDOWN}" | tee -a "${LOG_FILE}"
  echo "PROFILE_MEMORY_BREAKDOWN_INTERVAL=${PROFILE_MEMORY_BREAKDOWN_INTERVAL}" | tee -a "${LOG_FILE}"
  [[ -n "${PROFILE_MEMORY_BREAKDOWN_STEPS}" ]] && echo "PROFILE_MEMORY_BREAKDOWN_STEPS=${PROFILE_MEMORY_BREAKDOWN_STEPS}" | tee -a "${LOG_FILE}"
  echo "PROFILE_MEMORY_BREAKDOWN_MODULES=${PROFILE_MEMORY_BREAKDOWN_MODULES}" | tee -a "${LOG_FILE}"
  echo "PROFILE_MEMORY_BREAKDOWN_OUTPUT=${PROFILE_MEMORY_BREAKDOWN_OUTPUT}" | tee -a "${LOG_FILE}"
  echo "PROFILE_SYNC=${PROFILE_SYNC}" | tee -a "${LOG_FILE}"
  echo "PROFILE_MODULE_FILTER=${PROFILE_MODULE_FILTER}" | tee -a "${LOG_FILE}"
  echo "PROFILE_SOURCE_JSON=${PROFILE_SOURCE_JSON}" | tee -a "${LOG_FILE}"
  echo "PROFILE_JSON=${PROFILE_JSON}" | tee -a "${LOG_FILE}"
  echo "PROFILE_OUTPUT_DIR=${PROFILE_OUTPUT_DIR}" | tee -a "${LOG_FILE}"
  echo "PROFILE_SUMMARY_MD=${PROFILE_SUMMARY_MD}" | tee -a "${LOG_FILE}"
  [[ -n "${PROFILE_NSYS_PREFIX}" ]] && echo "PROFILE_NSYS_PREFIX=${PROFILE_NSYS_PREFIX}" | tee -a "${LOG_FILE}"
  [[ -n "${PROFILE_NSYS_SQLITE}" ]] && echo "PROFILE_NSYS_SQLITE=${PROFILE_NSYS_SQLITE}" | tee -a "${LOG_FILE}"
  if [[ "${PROFILE_PROFILER}" == "nsys" ]]; then
    echo "PROFILE_NSYS_CAPTURE_RANGE=${PROFILE_NSYS_CAPTURE_RANGE}" | tee -a "${LOG_FILE}"
    echo "PROFILE_NSYS_GPU_METRICS_DEVICES=${PROFILE_NSYS_GPU_METRICS_DEVICES}" | tee -a "${LOG_FILE}"
  fi
fi

RUN_PYTHONPATH="${ASYM_DIR}:${LF_DIR}/src:${PYTHONPATH:-}"
if [[ "${BACKEND}" == kt_* ]]; then
  RUN_PYTHONPATH="${KT_TOOLS_DIR}:${ASYM_DIR}:${KT_KERNEL_DIR}:${LF_DIR}/src:${PYTHONPATH:-}"
fi

RUN_ENV=(
  CUDA_VISIBLE_DEVICES="${GPU_ID}"
  PATH="${ENV_DIR}/bin:${PATH}"
  PYTHONPATH="${RUN_PYTHONPATH}"
)
if [[ "${BACKEND}" == kt_* ]]; then
  RUN_ENV+=(
    USE_KT=1
    ACCELERATE_KT_BACKEND="${KT_BACKEND_INTERNAL}"
    ACCELERATE_KT_TP_ENABLED="${KT_TP_ENABLED}"
  )
  if [[ "${BACKEND}" == "kt_torchbf16" ]]; then
    RUN_ENV+=(
      KT_TORCHBF16_SFT_DEVICE="${KT_TORCHBF16_SFT_DEVICE}"
      ASYM_GEMM_LF_CONFIG_KT_TORCHBF16_SFT_DEVICE="${KT_TORCHBF16_SFT_DEVICE}"
    )
  fi
  if [[ "${BACKEND}" == "kt_armbf16" ]]; then
    RUN_ENV+=(
      OMP_NUM_THREADS="${KT_ARM_OMP_NUM_THREADS}"
      OMP_PROC_BIND="${KT_ARM_OMP_PROC_BIND}"
      OMP_PLACES="${KT_ARM_OMP_PLACES}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_OMP_NUM_THREADS="${KT_ARM_OMP_NUM_THREADS}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_OMP_PROC_BIND="${KT_ARM_OMP_PROC_BIND}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_OMP_PLACES="${KT_ARM_OMP_PLACES}"
    )
  fi
  [[ -z "${KT_NUM_THREADS}" ]] || RUN_ENV+=(ACCELERATE_KT_NUM_THREADS="${KT_NUM_THREADS}")
  [[ -z "${KT_THREADPOOL_COUNT}" ]] || RUN_ENV+=(ACCELERATE_KT_THREADPOOL_COUNT="${KT_THREADPOOL_COUNT}")
  [[ -z "${KT_MAX_CACHE_DEPTH}" ]] || RUN_ENV+=(ACCELERATE_KT_MAX_CACHE_DEPTH="${KT_MAX_CACHE_DEPTH}")
  [[ -z "${KT_SHARE_BACKWARD_BB}" ]] || RUN_ENV+=(ACCELERATE_KT_SHARE_BACKWARD_BB="${KT_SHARE_BACKWARD_BB}")
  [[ -z "${KT_SHARE_CACHE_POOL}" ]] || RUN_ENV+=(ACCELERATE_KT_SHARE_CACHE_POOL="${KT_SHARE_CACHE_POOL}")
  [[ -z "${KT_NUM_GPU_EXPERTS}" ]] || RUN_ENV+=(ACCELERATE_KT_NUM_GPU_EXPERTS="${KT_NUM_GPU_EXPERTS}")
  [[ -z "${KT_WEIGHT_PATH}" ]] || RUN_ENV+=(ACCELERATE_KT_WEIGHT_PATH="${KT_WEIGHT_PATH}")
  [[ -z "${KT_EXPERT_CHECKPOINT_PATH}" ]] || RUN_ENV+=(ACCELERATE_KT_EXPERT_CHECKPOINT_PATH="${KT_EXPERT_CHECKPOINT_PATH}")
  [[ -z "${KT_USE_LORA_EXPERTS}" ]] || RUN_ENV+=(ACCELERATE_KT_USE_LORA_EXPERTS="${KT_USE_LORA_EXPERTS}")
  [[ -z "${KT_LORA_EXPERT_NUM}" ]] || RUN_ENV+=(ACCELERATE_KT_LORA_EXPERT_NUM="${KT_LORA_EXPERT_NUM}")
  [[ -z "${KT_LORA_EXPERT_INTERMEDIATE_SIZE}" ]] || RUN_ENV+=(ACCELERATE_KT_LORA_EXPERT_INTERMEDIATE_SIZE="${KT_LORA_EXPERT_INTERMEDIATE_SIZE}")
elif [[ "${BACKEND}" == "torch" && "${TORCH_USE_ASYM_GEMM_LORA}" != "true" ]]; then
  :
else
  RUN_ENV+=(USE_ASYM_GEMM=1 ASYM_GEMM_LF_LOG_RUNTIME_STATS=1)
fi

if is_torch_distributed_run; then
  RUN_ENV+=(
    FORCE_TORCHRUN=1
    NNODES="${NNODES:-1}"
    NODE_RANK="${NODE_RANK:-0}"
    NPROC_PER_NODE="${NUM_GPUS}"
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    MASTER_PORT="${MASTER_PORT:-29500}"
  )
fi

if is_torch_distributed_run && [[ "${TORCH_DISTRIBUTED_BACKEND}" == "deepspeed" ]]; then
  CMD_ARGS+=(--deepspeed "${TORCH_DEEPSPEED_CONFIG}")
fi
assert_deepspeed_scope

if [[ "${PROFILE}" == "1" ]]; then
  profile_precision="${ASYM_PRECISION}"
  [[ "${BACKEND}" == kt_* ]] && profile_precision="${KT_PRECISION}"
  RUN_ENV+=(
      ASYM_GEMM_LF_PROFILE_SOURCE_JSON="${PROFILE_SOURCE_JSON}"
      ASYM_GEMM_LF_PROFILE_MEMORY="${PROFILE_MEMORY}"
      ASYM_GEMM_LF_PROFILE_LEVEL="${PROFILE_LEVEL}"
      ASYM_GEMM_LF_PROFILE_LAYERS="${PROFILE_LAYERS}"
      ASYM_GEMM_LF_PROFILE_MEMORY_ATTRIBUTION="${PROFILE_MEMORY_ATTRIBUTION}"
      ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN="${PROFILE_MEMORY_BREAKDOWN}"
      ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN_INTERVAL="${PROFILE_MEMORY_BREAKDOWN_INTERVAL}"
      ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN_STEPS="${PROFILE_MEMORY_BREAKDOWN_STEPS}"
      ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN_MODULES="${PROFILE_MEMORY_BREAKDOWN_MODULES}"
      ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN_OUTPUT="${PROFILE_MEMORY_BREAKDOWN_OUTPUT}"
      ASYM_GEMM_LF_PROFILE_SYNC="${PROFILE_SYNC}"
      ASYM_GEMM_LF_PROFILE_MODULE_FILTER="${PROFILE_MODULE_FILTER}"
	    ASYM_GEMM_LF_CONFIG_WORKLOAD="${PROFILE_WORKLOAD_LABEL:-${MODEL_TAG}}"
	    ASYM_GEMM_LF_CONFIG_BACKEND="${PROFILE_BACKEND_LABEL:-${BACKEND}}"
	    ASYM_GEMM_LF_CONFIG_PRECISION="${profile_precision}"
	    ASYM_GEMM_LF_CONFIG_SEQ_LEN="${CUTOFF_LEN}"
	    ASYM_GEMM_LF_CONFIG_ACTIVATION_RECOMPUTE="${GRADIENT_CHECKPOINTING}"
	    ASYM_GEMM_LF_CONFIG_EXPERT_POLICY="${PROFILE_EXPERT_POLICY}"
	    ASYM_GEMM_LF_CONFIG_PROFILE_LEVEL="${PROFILE_LEVEL}"
	    ASYM_GEMM_LF_CONFIG_WARMUP_STEPS="${PROFILE_WARMUP_STEPS:-0}"
	    ASYM_GEMM_LF_CONFIG_MEASURE_STEPS="${PROFILE_MEASURE_STEPS:-${MAX_STEPS}}"
	    ASYM_GEMM_LF_CONFIG_TOTAL_STEPS="${PROFILE_TOTAL_STEPS:-${MAX_STEPS}}"
	  )
  if [[ "${BACKEND}" == kt_* ]]; then
    RUN_ENV+=(
	    ASYM_GEMM_LF_CONFIG_KT_BACKEND="${KT_BACKEND_INTERNAL:-}"
	    ASYM_GEMM_LF_CONFIG_KT_KERNEL_DIR="${KT_KERNEL_DIR}"
	    ASYM_GEMM_LF_CONFIG_KT_NUM_THREADS="${KT_NUM_THREADS}"
	    ASYM_GEMM_LF_CONFIG_KT_THREADPOOL_COUNT="${KT_THREADPOOL_COUNT}"
	    ASYM_GEMM_LF_CONFIG_KT_MAX_CACHE_DEPTH="${KT_MAX_CACHE_DEPTH}"
	    ASYM_GEMM_LF_CONFIG_KT_TP_ENABLED="${KT_TP_ENABLED}"
	    ASYM_GEMM_LF_CONFIG_KT_SHARE_BACKWARD_BB="${KT_SHARE_BACKWARD_BB}"
	    ASYM_GEMM_LF_CONFIG_KT_SHARE_CACHE_POOL="${KT_SHARE_CACHE_POOL}"
	    ASYM_GEMM_LF_CONFIG_KT_NUM_GPU_EXPERTS="${KT_NUM_GPU_EXPERTS}"
	    ASYM_GEMM_LF_CONFIG_KT_WEIGHT_PATH="${KT_WEIGHT_PATH}"
	    ASYM_GEMM_LF_CONFIG_KT_EXPERT_CHECKPOINT_PATH="${KT_EXPERT_CHECKPOINT_PATH}"
	    ASYM_GEMM_LF_CONFIG_KT_USE_LORA_EXPERTS="${KT_USE_LORA_EXPERTS}"
	    ASYM_GEMM_LF_CONFIG_KT_LORA_EXPERT_NUM="${KT_LORA_EXPERT_NUM}"
	    ASYM_GEMM_LF_CONFIG_KT_LORA_EXPERT_INTERMEDIATE_SIZE="${KT_LORA_EXPERT_INTERMEDIATE_SIZE}"
	  )
  fi
  if [[ "${PROFILE_PROFILER}" == "nsys" && "${PROFILE_NSYS_CAPTURE_RANGE}" == "cudaProfilerApi" ]]; then
    RUN_ENV+=(ASYM_GEMM_LF_NSYS_CAPTURE_RANGE=1)
  fi
fi

if [[ "${PROFILE}" == "1" ]]; then
  LAUNCH_CMD=("${ENV_PYTHON}" "${PROFILE_LAUNCHER}" "${CMD_ARGS[@]}")
else
  LAUNCH_CMD=("${CONDA_EXE}" run -p "${ENV_DIR}" llamafactory-cli train "${CMD_ARGS[@]}")
fi

if is_torch_distributed_run && [[ "${TORCH_DISTRIBUTED_BACKEND}" == "fsdp2" ]]; then
  if [[ "${PROFILE}" == "1" ]]; then
    LAUNCH_CMD=(
      "${ACCELERATE_BIN}" launch
      --config_file "${TORCH_FSDP_CONFIG}"
      --num_processes "${NUM_GPUS}"
      --main_process_port "${MASTER_PORT:-29500}"
      "${PROFILE_LAUNCHER}"
      "${CMD_ARGS[@]}"
    )
  else
    LAUNCH_CMD=(
      "${ACCELERATE_BIN}" launch
      --config_file "${TORCH_FSDP_CONFIG}"
      --num_processes "${NUM_GPUS}"
      --main_process_port "${MASTER_PORT:-29500}"
      "${LF_DIR}/src/train.py"
      "${CMD_ARGS[@]}"
    )
  fi
elif is_torch_distributed_run && [[ "${TORCH_DISTRIBUTED_BACKEND}" == "deepspeed" ]]; then
  if [[ "${PROFILE}" == "1" ]]; then
    LAUNCH_CMD=(
      "${ACCELERATE_BIN}" launch
      --num_processes "${NUM_GPUS}"
      --main_process_port "${MASTER_PORT:-29500}"
      "${PROFILE_LAUNCHER}"
      "${CMD_ARGS[@]}"
    )
  else
    LAUNCH_CMD=(
      "${ACCELERATE_BIN}" launch
      --num_processes "${NUM_GPUS}"
      --main_process_port "${MASTER_PORT:-29500}"
      "${LF_DIR}/src/train.py"
      "${CMD_ARGS[@]}"
    )
  fi
elif [[ "${PROFILE}" == "1" ]] && is_torch_distributed_run; then
  LAUNCH_CMD=(
    "${TORCHRUN_BIN}"
    --nnodes "${NNODES:-1}"
    --node_rank "${NODE_RANK:-0}"
    --nproc_per_node "${NUM_GPUS}"
    --master_addr "${MASTER_ADDR:-127.0.0.1}"
    --master_port "${MASTER_PORT:-29500}"
    "${PROFILE_LAUNCHER}"
    "${CMD_ARGS[@]}"
  )
fi

if [[ "${PROFILE}" == "1" && "${PROFILE_PROFILER}" == "nsys" ]]; then
  NSYS_CMD=(
    "${NSYS_BIN}"
    profile
    --trace=cuda,nvtx
    --sample=none
    --cpuctxsw=none
    --resolve-symbols=false
    --wait=primary
    --force-overwrite=true
    --gpu-metrics-devices="${PROFILE_NSYS_GPU_METRICS_DEVICES}"
    --gpu-metrics-set=gb10x
    --gpu-metrics-frequency=100
    "--output=${PROFILE_NSYS_PREFIX}"
  )
  if [[ "${PROFILE_NSYS_CAPTURE_RANGE}" == "cudaProfilerApi" ]]; then
    NSYS_CMD+=(--capture-range=cudaProfilerApi --capture-range-end=stop)
  fi
  env "${RUN_ENV[@]}" "${NSYS_CMD[@]}" "${LAUNCH_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
else
  env "${RUN_ENV[@]}" "${LAUNCH_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
fi

if [[ "${BACKEND}" == "asym" && "${CHECK_ASYM_CALLS}" == "1" ]]; then
  ASYM_CALL_CHECK_OUTPUT=$(python3 - "${LOG_FILE}" <<'PY'
import re
import sys

log_file = sys.argv[1]
text = open(log_file, encoding="utf-8").read()
matches = list(
    re.finditer(
        r"AsymGEMM LoRA-SFT runtime: .*?asym_forward_calls=(\d+), asym_dx_calls=(\d+)",
        text,
    )
)
if not matches:
    raise SystemExit(f"Missing AsymGEMM runtime call report in {log_file}")
forward_calls = int(matches[-1].group(1))
dx_calls = int(matches[-1].group(2))
if forward_calls <= 0 or dx_calls <= 0:
    raise SystemExit(
        f"Expected positive AsymGEMM forward and dx calls, got "
        f"asym_forward_calls={forward_calls} asym_dx_calls={dx_calls}"
    )
print(f"Verified AsymGEMM runtime calls: asym_forward_calls={forward_calls} asym_dx_calls={dx_calls}")
PY
  )
  echo "${ASYM_CALL_CHECK_OUTPUT}" | tee -a "${LOG_FILE}"
fi

if [[ "${BACKEND}" == kt_* && "${CHECK_KT_CALLS}" == "1" ]]; then
  if [[ -f "${PROFILE_SOURCE_JSON}" ]]; then
    KT_CALL_CHECK_OUTPUT=$("${ENV_PYTHON}" - "${PROFILE_SOURCE_JSON}" "${KT_BACKEND_INTERNAL}" <<'PY'
import json
import sys

profile = json.load(open(sys.argv[1], encoding="utf-8"))
kt_backend = sys.argv[2]
kt = profile.get("kt", {})
wrappers = int(kt.get("wrapper_count", 0) or 0)
fw = int(kt.get("total_forward_calls", 0) or 0)
bw = int(kt.get("total_backward_calls", 0) or 0)
methods = sorted({str(row.get("method", "")) for row in kt.get("rows", []) if isinstance(row, dict)})
if wrappers <= 0:
    raise SystemExit(f"Expected positive KT wrapper_count in {sys.argv[1]}, got {wrappers}")
if fw <= 0 or bw <= 0:
    raise SystemExit(f"Expected positive KT forward/backward calls in {sys.argv[1]}, got fw={fw} bw={bw}")
print(f"Verified KT source counters: backend={kt_backend} wrappers={wrappers} fw={fw} bw={bw} methods={methods}")
PY
    )
  else
    KT_CALL_CHECK_OUTPUT=$("${ENV_PYTHON}" - "${LOG_FILE}" "${KT_BACKEND_INTERNAL}" <<'PY'
import re
import sys

log_file = sys.argv[1]
kt_backend = sys.argv[2]
text = open(log_file, encoding="utf-8").read()

backend_seen = kt_backend in text or f"kt_backend {kt_backend}" in text or f"kt_backend={kt_backend}" in text
wrapper_seen = re.search(
    r"(_kt_wrappers|KT wrappers?|KTransformers.*wrapp|kt_adapt_peft_lora|KTMoELayerWrapper)",
    text,
    flags=re.IGNORECASE,
) is not None
if not backend_seen:
    raise SystemExit(f"Missing KT backend evidence for {kt_backend} in {log_file}")
if not wrapper_seen:
    raise SystemExit(
        f"Missing KT wrapper/adaptation evidence in {log_file}. "
        "The LF KT integration should log wrapper count or kt_adapt_peft_lora evidence."
    )

counter_matches = re.findall(
    r"(?:kt|KT)[A-Za-z0-9_ .:-]*(forward|backward)[A-Za-z0-9_ .:-]*calls?=(\d+)",
    text,
)
bad = [(name, int(value)) for name, value in counter_matches if int(value) <= 0]
if bad:
    raise SystemExit(f"KT runtime counters were present but non-positive: {bad}")
if counter_matches:
    print(f"Verified KT runtime counters: {counter_matches[-4:]}")
else:
    print("Verified KT wrapper/backend evidence; no KT runtime call counters were found yet.")
PY
    )
  fi
  echo "${KT_CALL_CHECK_OUTPUT}" | tee -a "${LOG_FILE}"
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

if [[ "${PROFILE}" == "1" && "${PROFILE_PROFILER}" == "source" ]]; then
  if [[ ! -f "${PROFILE_POSTPROCESS_SCRIPT}" ]]; then
    echo "Missing profile postprocess script ${PROFILE_POSTPROCESS_SCRIPT}" >&2
    exit 2
  fi
  "${CONDA_EXE}" run -p "${ENV_DIR}" python "${PROFILE_POSTPROCESS_SCRIPT}" \
    --source-profile-json "${PROFILE_SOURCE_JSON}" \
    --profile-json "${PROFILE_JSON}" \
    --output-dir "${PROFILE_OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"
  echo "Wrote source profile artifacts under ${PROFILE_OUTPUT_DIR}" | tee -a "${LOG_FILE}"
fi

if [[ "${PROFILE}" == "1" && "${PROFILE_PROFILER}" == "nsys" ]]; then
  if [[ ! -f "${PROFILE_NSYS_PREFIX}.nsys-rep" ]]; then
    echo "Missing Nsight report ${PROFILE_NSYS_PREFIX}.nsys-rep" >&2
    exit 1
  fi
  "${NSYS_BIN}" export \
    --type=sqlite \
    --force-overwrite=true \
    --output="${PROFILE_NSYS_SQLITE}" \
    "${PROFILE_NSYS_PREFIX}.nsys-rep" 2>&1 | tee -a "${LOG_FILE}"
  if [[ ! -f "${PROFILE_NSYS_POSTPROCESS_SCRIPT}" ]]; then
    echo "Missing Nsight postprocess script ${PROFILE_NSYS_POSTPROCESS_SCRIPT}" >&2
    exit 2
  fi
  "${CONDA_EXE}" run -p "${ENV_DIR}" python "${PROFILE_NSYS_POSTPROCESS_SCRIPT}" \
    "${PROFILE_NSYS_SQLITE}" \
    --source-profile-json "${PROFILE_SOURCE_JSON}" \
    --output-json "${PROFILE_JSON}" \
    --output-md "${PROFILE_SUMMARY_MD}" 2>&1 | tee -a "${LOG_FILE}"
  "${CONDA_EXE}" run -p "${ENV_DIR}" python "${PROFILE_POSTPROCESS_SCRIPT}" \
    --profile-json "${PROFILE_JSON}" \
    --output-dir "${PROFILE_OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"
  echo "Wrote Nsight profile artifacts to ${PROFILE_JSON} and ${PROFILE_SUMMARY_MD}" | tee -a "${LOG_FILE}"
fi
