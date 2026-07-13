#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

if (($# > 0)); then
  echo "run_lf_lora_sft.sh does not accept command-line arguments; set environment variables or use profile_lora_lf_test_source.sh." >&2
  echo "Unexpected arguments: $*" >&2
  exit 2
fi

# =============================================================================
# User Parameters
# =============================================================================

# Paths and tools
# Repo root = AsymGEMM-SFT (../.. from the AsymGEMM dir you run in). Override with SFT_ROOT=...
SFT_ROOT=${SFT_ROOT:-$(cd ../.. && pwd)}
ROOT=${ROOT:-${SFT_ROOT}/third_party/AsymGEMM}
_LF_DIR_ENV_SET=false
_ENV_DIR_ENV_SET=false
_FLASH_ATTN_ENV_SET=false
[[ -n "${LF_DIR+x}" ]] && _LF_DIR_ENV_SET=true
[[ -n "${ENV_DIR+x}" ]] && _ENV_DIR_ENV_SET=true
[[ -n "${FLASH_ATTN+x}" ]] && _FLASH_ATTN_ENV_SET=true
LF_DIR=${LF_DIR:-${SFT_ROOT}/third_party/LlamaFactory}
LF_FA4_DIR=${LF_FA4_DIR:-${SFT_ROOT}/third_party/LlamaFactory}
LF_DS_CONFIG_DIR=${LF_DS_CONFIG_DIR:-${SFT_ROOT}/third_party/LlamaFactory/examples/deepspeed}
KT_KERNEL_DIR=${KT_KERNEL_DIR:-${SFT_ROOT}/third_party/ktransformers/kt-kernel}
DEEPSPEED_DIR=${DEEPSPEED_DIR:-${SFT_ROOT}/third_party/deepspeed}
CONDA_EXE=${CONDA_EXE:-conda}
NSYS_BIN=${NSYS_BIN:-nsys}
DIST_LAUNCHER=${DIST_LAUNCHER:-torchrun} # torchrun | accelerate | deepspeed

# DeepSpeed NVMe offload (async_io) needs libaio, provided by a shared sidecar conda env at
# ${ROOT}/.aioenv so every container sees it like .venv. Guarded + harmless for non-NVMe backends.
# CFLAGS/LDFLAGS let DeepSpeed JIT-compile async_io; LD_LIBRARY_PATH lets the built op load libaio.so.1.
AIO_HOME=${AIO_HOME:-${ROOT}/.aioenv}
if [[ -d "${AIO_HOME}/lib" ]]; then
  # g++ honors CPATH/LIBRARY_PATH directly; the DeepSpeed/ninja JIT build does NOT pass CFLAGS/LDFLAGS
  # into its compile line, so CPATH/LIBRARY_PATH are what actually let it find libaio.h and -laio.
  # LD_LIBRARY_PATH lets the compiled async_io op load libaio.so.1 at runtime. (CFLAGS/LDFLAGS kept
  # too for DeepSpeed's is_compatible() probe; harmless otherwise.)
  export CPATH="${AIO_HOME}/include${CPATH:+:${CPATH}}"
  export LIBRARY_PATH="${AIO_HOME}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${AIO_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export CFLAGS="-I${AIO_HOME}/include ${CFLAGS:-}"
  export LDFLAGS="-L${AIO_HOME}/lib ${LDFLAGS:-}"
fi

# Workload and placement
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen3-30B-A3B}
BACKEND=${BACKEND:-asym}              # torch | zero2 | zero3 | zero3_offload | zero3_offload_mem | zero3_offload_opnvme | zero3_offload_panvme | zero3_offload_mem_opnvme | zero3_offload_mem_panvme | zero3_cpuadam | superoffload | superoffload_mem | superoffload_mem_opnvme | superoffload_mem_panvme | asym_torch | asym | kt_torchbf16 | kt_armbf16
ASYM_QWEN35_FA4_AUTO=${ASYM_QWEN35_FA4_AUTO:-1}
FLASH_ATTN=${FLASH_ATTN:-auto}
GPU_ID=${GPU_ID:-0}
NUM_GPUS=${NUM_GPUS:-1}
NUMACTL_ENABLE=${NUMACTL_ENABLE:-1}
NUMACTL_BIN=${NUMACTL_BIN:-numactl}
NUMACTL_MEMBIND=${NUMACTL_MEMBIND:-0,1}
NUMACTL_CPUNODEBIND=${NUMACTL_CPUNODEBIND:-0,1}
# Lever-1 toggle: membind (default = today's behavior) | interleave (spread pinned pages 50/50 across nodes)
NUMACTL_MODE=${NUMACTL_MODE:-membind}
REQUIRE_SM100=${REQUIRE_SM100:-1}

# Dataset
DATASET=${DATASET:-asym_long_sft_smoke}
TEMPLATE=${TEMPLATE:-auto}
CUTOFF_LEN=${CUTOFF_LEN:-4096}
MAX_SAMPLES=${MAX_SAMPLES:-64}

# Training
MAX_STEPS=${MAX_STEPS:-10}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
FINETUNING_TYPE=${FINETUNING_TYPE:-lora} # lora | full
LF_QWEN_MOE_EXPERT_LORA_IMPL=${LF_QWEN_MOE_EXPERT_LORA_IMPL:-split-target-parameters}
SEED=${SEED:-42}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-}
PREPROCESSING_NUM_WORKERS=${PREPROCESSING_NUM_WORKERS:-}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-}
TRAINING_BF16=${TRAINING_BF16:-}
ENABLE_LIGER_KERNEL=${ENABLE_LIGER_KERNEL:-false}
USE_UNSLOTH_GC=${USE_UNSLOTH_GC:-false}
UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=${UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU:-false}
# Keep every Nth outer Unsloth checkpoint root in HBM instead of CPU RAM
# (ohbm<N> tag; profile drivers set it per run from a -ohbm<N> recompute suffix).
# Keep at 0 for fixed-seq apples-to-apples comparisons; nonzero is for
# max-seq ceiling hunting.
UNSLOTH_GC_OUTER_HBM_EVERY_N=${UNSLOTH_GC_OUTER_HBM_EVERY_N:-0}

# AsymGEMM
ASYM_PRECISION=${ASYM_PRECISION:-bf16}
ASYM_OFFLOAD_MODULES=${ASYM_OFFLOAD_MODULES:-routed_experts}
ASYMM_EXPERT_ACT_OFFLOAD=${ASYMM_EXPERT_ACT_OFFLOAD:-false}
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD-cpu}
ASYMM_EXPERT_SILU_BWD_GPU=${ASYMM_EXPERT_SILU_BWD_GPU:-1}
ASYMM_DENSE_MLP_SURGICAL_OFFLOAD=${ASYMM_DENSE_MLP_SURGICAL_OFFLOAD:-0}
ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD=${ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD:-0}
ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT=${ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT:-0}
ASYMM_DENSE_MLP_FINEGRAINED_NOGRAD_CPU_OFFLOAD=${ASYMM_DENSE_MLP_FINEGRAINED_NOGRAD_CPU_OFFLOAD:-0}
ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD=${ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD:-0}
ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=${ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS:-0}
ASYMM_ATTN_ACT_OFFLOAD=${ASYMM_ATTN_ACT_OFFLOAD:-false}
ASYMM_LAYER_ACT_OFFLOAD=${ASYMM_LAYER_ACT_OFFLOAD:-false}
ASYMM_LAYER_GC=${ASYMM_LAYER_GC:-false}
ASYMM_ATTN_SDPA_RECOMPUTE=${ASYMM_ATTN_SDPA_RECOMPUTE:-false}
ASYM_OFFLOAD_ACT_RECOMPUTE=${ASYM_OFFLOAD_ACT_RECOMPUTE:-0}
ASYM_OFFLOAD_X_UNPACKED=${ASYM_OFFLOAD_X_UNPACKED:-0}
ASYM_EXPERT_RECOMPUTE_POLICY=${ASYM_EXPERT_RECOMPUTE_POLICY:-none}
ASYMM_MLP_RECOMPUTE_CHUNK=${ASYMM_MLP_RECOMPUTE_CHUNK:-0}
ASYM_ROUTER_MODE=${ASYM_ROUTER_MODE:-whole}
ASYM_STRICT=${ASYM_STRICT:-true}
USE_ASYM_CPU_ADAMW=${USE_ASYM_CPU_ADAMW:-false}
ASYM_CPU_ADAMW_BACKEND=${ASYM_CPU_ADAMW_BACKEND:-deepspeed}
ASYM_CPU_ADAMW_PIN_MEMORY=${ASYM_CPU_ADAMW_PIN_MEMORY:-true}
ASYM_CPU_ADAMW_FP32_MASTER=${ASYM_CPU_ADAMW_FP32_MASTER:-true}
ASYM_CPU_ADAMW_GRAD_OFFLOAD=${ASYM_CPU_ADAMW_GRAD_OFFLOAD:-false}
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=${ASYM_CPU_ADAMW_WEIGHT_OFFLOAD:-false}
CHECK_ASYM_CALLS=${CHECK_ASYM_CALLS:-1}
CHECK_TRAINABLE_SURFACE=${CHECK_TRAINABLE_SURFACE:-0}  # opt-in; set 1 to guard MoE expert-LoRA runs

# KT
KT_PRECISION=${KT_PRECISION:-${ASYM_PRECISION}}
CHECK_KT_CALLS=${CHECK_KT_CALLS:-1}
KT_NUM_THREADS=${KT_NUM_THREADS:-}
KT_THREADPOOL_COUNT=${KT_THREADPOOL_COUNT:-}
KT_MAX_CACHE_DEPTH=${KT_MAX_CACHE_DEPTH:-2}
KT_SHARE_BACKWARD_BB=${KT_SHARE_BACKWARD_BB:-}
KT_TP_ENABLED=${KT_TP_ENABLED:-false}
KT_NUM_GPU_EXPERTS=${KT_NUM_GPU_EXPERTS:-}
KT_WEIGHT_PATH=${KT_WEIGHT_PATH:-}
KT_EXPERT_CHECKPOINT_PATH=${KT_EXPERT_CHECKPOINT_PATH:-}
KT_USE_LORA_EXPERTS=${KT_USE_LORA_EXPERTS:-}
KT_LORA_EXPERT_NUM=${KT_LORA_EXPERT_NUM:-}
KT_LORA_EXPERT_INTERMEDIATE_SIZE=${KT_LORA_EXPERT_INTERMEDIATE_SIZE:-}
KT_REQUIRE_STARTUP=${KT_REQUIRE_STARTUP:-}
KT_REQUIRE_FUSED_LORA_STARTUP=${KT_REQUIRE_FUSED_LORA_STARTUP:-}
KT_TORCHBF16_SFT_DEVICE=${KT_TORCHBF16_SFT_DEVICE:-cuda}
KT_ARM_OMP_NUM_THREADS=${KT_ARM_OMP_NUM_THREADS:-64}
KT_ARM_OMP_PROC_BIND=${KT_ARM_OMP_PROC_BIND:-close}
KT_ARM_OMP_PLACES=${KT_ARM_OMP_PLACES:-cores}
KT_SFT_PROGRESS=${KT_SFT_PROGRESS:-}
KT_ARM_SFT_PROFILE=${KT_ARM_SFT_PROFILE:-}
KT_ARM_SFT_POOL_LOG=${KT_ARM_SFT_POOL_LOG:-}
KT_ARM_SFT_TOP_K=${KT_ARM_SFT_TOP_K:-8}
KT_ARM_SFT_TOKEN_CHUNK_SIZE=${KT_ARM_SFT_TOKEN_CHUNK_SIZE:-}
KT_ARM_SFT_MAX_ROUTE_RANK_WORK=${KT_ARM_SFT_MAX_ROUTE_RANK_WORK:-}
KT_ARM_SFT_DEFAULT_MAX_ROUTE_RANK_WORK=${KT_ARM_SFT_DEFAULT_MAX_ROUTE_RANK_WORK:-1048576}
KT_ARM_SFT_BACKWARD_THREADS=${KT_ARM_SFT_BACKWARD_THREADS:-}
KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES=${KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES:-}
KT_ARM_SFT_DEFAULT_MAX_BACKWARD_SCRATCH_BYTES=${KT_ARM_SFT_DEFAULT_MAX_BACKWARD_SCRATCH_BYTES:-34359738368}
KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK=${KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK:-0}
KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH=${KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH:-0}
KT_ARM_ALLOW_NSYS_WITHOUT_SOURCE_OK=${KT_ARM_ALLOW_NSYS_WITHOUT_SOURCE_OK:-0}
KT_ARM_ALLOW_RAW_NSYS_WITHOUT_SOURCE_OK=${KT_ARM_ALLOW_RAW_NSYS_WITHOUT_SOURCE_OK:-0}
KT_ARM_SOURCE_OK_PROFILE_JSON=${KT_ARM_SOURCE_OK_PROFILE_JSON:-}
KT_ARM_FIRST_STEP_TIMEOUT_SECONDS=${KT_ARM_FIRST_STEP_TIMEOUT_SECONDS:-0}

# SuperOffload
CHECK_SUPEROFFLOAD=${CHECK_SUPEROFFLOAD:-1}

# DeepSpeed CPUAdam baseline
CHECK_CPUADAM=${CHECK_CPUADAM:-1}

# Profiling
PROFILE=${PROFILE:-0}
PROFILE_PROFILER=${PROFILE_PROFILER:-source} # source | nsys
PROFILE_MEMORY=${PROFILE_MEMORY:-1}
PROFILE_LEVEL=${PROFILE_LEVEL:-op}           # stage | module | op | deep
PROFILE_LAYERS=${PROFILE_LAYERS:-all}
# Single knob: PROFILE_MEMORY_BREAKDOWN gates BOTH the per-module breakdown and the saved-tensor/param
# attribution; both ASYM_GEMM_LF_PROFILE_MEMORY_* toggles are set from it below. Default true.
PROFILE_MEMORY_BREAKDOWN=${PROFILE_MEMORY_BREAKDOWN:-true}
PROFILE_MEMORY_BREAKDOWN_INTERVAL=${PROFILE_MEMORY_BREAKDOWN_INTERVAL:-1}
PROFILE_MEMORY_BREAKDOWN_STEPS=${PROFILE_MEMORY_BREAKDOWN_STEPS:-}
PROFILE_MEMORY_BREAKDOWN_MODULES=${PROFILE_MEMORY_BREAKDOWN_MODULES:-attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss}
PROFILE_MEMORY_BREAKDOWN_OUTPUT=${PROFILE_MEMORY_BREAKDOWN_OUTPUT:-memory_breakdown}
PROFILE_LIVE_ACTIVATION_DETAILS=${PROFILE_LIVE_ACTIVATION_DETAILS:-${ASYM_GEMM_LF_PROFILE_LIVE_ACTIVATION_DETAILS:-false}}
PROFILE_LIVE_ACTIVATION_TOPK=${PROFILE_LIVE_ACTIVATION_TOPK:-${ASYM_GEMM_LF_PROFILE_LIVE_ACTIVATION_TOPK:-100}}
PROFILE_MEMORY_SNAPSHOT=${PROFILE_MEMORY_SNAPSHOT:-false}
PROFILE_MEMORY_SNAPSHOT_PATH=${PROFILE_MEMORY_SNAPSHOT_PATH:-}
PROFILE_EXTERNAL_MEMORY=${PROFILE_EXTERNAL_MEMORY:-false}
PROFILE_SYNC=${PROFILE_SYNC:-false}
PROFILE_MODULE_FILTER=${PROFILE_MODULE_FILTER:-attention,router,mlp,experts,lora,optimizer,kt}
PROFILE_SOURCE_JSON=${PROFILE_SOURCE_JSON:-}
PROFILE_NSYS_PREFIX=${PROFILE_NSYS_PREFIX:-}
PROFILE_NSYS_SQLITE=${PROFILE_NSYS_SQLITE:-}
PROFILE_NSYS_CAPTURE_RANGE=${PROFILE_NSYS_CAPTURE_RANGE:-cudaProfilerApi} # cudaProfilerApi | none
PROFILE_NSYS_GPU_METRICS_DEVICES=${PROFILE_NSYS_GPU_METRICS_DEVICES:-${GPU_ID}}
# NVMe-offload I/O monitoring. osrt traces the libaio reader threads so the param swap-in cost is
# attributable; override NSYS_TRACE=cuda,nvtx to drop it. RECORD_IO samples /sys/block/<dev>/stat
# once a second so an agent reads real NVMe bandwidth without parsing the multi-100MB nsys sqlite.
NSYS_TRACE=${NSYS_TRACE:-cuda,nvtx,osrt}
RECORD_IO=${RECORD_IO:-1}
NVME_OFFLOAD_IO_DEVICE=${NVME_OFFLOAD_IO_DEVICE:-md0}
OFFLOAD_SUMMARY_SCRIPT=${OFFLOAD_SUMMARY_SCRIPT:-${ROOT}/scripts/lf/summarize_nvme_offload.py}
PROFILE_JSON=${PROFILE_JSON:-}
PROFILE_SUMMARY_MD=${PROFILE_SUMMARY_MD:-}
PROFILE_HEARTBEAT_JSON=${PROFILE_HEARTBEAT_JSON:-}
PROFILE_PARTIAL_INTERVAL_SECONDS=${PROFILE_PARTIAL_INTERVAL_SECONDS:-5}
PROFILE_OUTPUT_DIR=${PROFILE_OUTPUT_DIR:-}
PROFILE_WORKLOAD_LABEL=${PROFILE_WORKLOAD_LABEL:-}
PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-}
PROFILE_EXPERT_POLICY=${PROFILE_EXPERT_POLICY:-${ASYM_EXPERT_RECOMPUTE_POLICY}}
INTERRUPT_GRACE_SECONDS=${INTERRUPT_GRACE_SECONDS:-2}

# Host-OOM guards (GB200 nvidia-imex wedge mitigation):
# - TRAIN_OOM_SCORE_ADJ marks the training tree as the kernel OOM killer's preferred
#   victim, so a host OOM can never take system daemons (nvidia-imex/persistenced)
#   down with it. Raising the score needs no privileges. Empty string disables.
# - HOST_MEM_WATCHDOG interrupts training gracefully (SIGSTOP+SIGINT+SIGCONT) before
#   the kernel OOM killer fires: it polls available memory on the CPU NUMA nodes and
#   triggers below HOST_MEM_WATCHDOG_FLOOR_GB. false or FLOOR_GB=0 disables.
TRAIN_OOM_SCORE_ADJ=${TRAIN_OOM_SCORE_ADJ:-1000}
HOST_MEM_WATCHDOG=${HOST_MEM_WATCHDOG:-true}
# exact per-model floor (GB); unlisted model without an exported floor = hard error
declare -A WATCHDOG_FLOOR_GB_BY_MODEL=(
  ["Qwen/Qwen3-30B-A3B"]=35
  ["Qwen/Qwen3-32B"]=35
  ["Qwen/Qwen3.5-27B"]=35
  ["Qwen/Qwen3.5-35B-A3B"]=35
  ["Qwen/Qwen2.5-32B-Instruct"]=35
  ["Qwen/Qwen2.5-72B-Instruct"]=50
  ["meta-llama/Llama-3.3-70B-Instruct"]=50
  ["meta-llama/Llama-4-Scout-17B-16E"]=50
  ["Qwen/Qwen3.5-122B-A10B"]=50
  ["Qwen/Qwen3-235B-A22B"]=60
)
if [[ -z "${HOST_MEM_WATCHDOG_FLOOR_GB:-}" ]]; then
  if [[ "${HOST_MEM_WATCHDOG,,}" == "true" ]]; then
    HOST_MEM_WATCHDOG_FLOOR_GB="${WATCHDOG_FLOOR_GB_BY_MODEL[${MODEL_NAME_OR_PATH}]:-}"
    if [[ -z "${HOST_MEM_WATCHDOG_FLOOR_GB}" ]]; then
      echo "error: no watchdog floor mapped for model '${MODEL_NAME_OR_PATH}'; add it to" \
           "WATCHDOG_FLOOR_GB_BY_MODEL in run_lf_lora_sft.sh or export HOST_MEM_WATCHDOG_FLOOR_GB" >&2
      exit 2
    fi
  else
    HOST_MEM_WATCHDOG_FLOOR_GB=0  # watchdog disabled: floor unused
  fi
fi
HOST_MEM_WATCHDOG_POLL_SECONDS=${HOST_MEM_WATCHDOG_POLL_SECONDS:-0.05}
# kill fast: the watchdog line already recorded the C-OOM; a long grace lets the
# kernel OOM killer win the race and wedge the GPU driver
HOST_MEM_WATCHDOG_KILL_GRACE_SECONDS=${HOST_MEM_WATCHDOG_KILL_GRACE_SECONDS:-1}

# =============================================================================
# Derived Parameters
# =============================================================================
ASYM_DIR=${ASYM_DIR:-${ROOT}}
KT_TOOLS_DIR=${KT_TOOLS_DIR:-${ASYM_DIR}}
KT_REPO_DIR=${KT_REPO_DIR:-$(dirname "${KT_KERNEL_DIR}")}
KT_GGUF_PY_DIR=${KT_GGUF_PY_DIR:-${KT_REPO_DIR}/third_party/llama.cpp/gguf-py}
KT_GGUF_PYTHONPATH_ENTRY=""
if [[ -f "${KT_GGUF_PY_DIR}/gguf/gguf_reader.py" ]]; then
  KT_GGUF_PYTHONPATH_ENTRY="${KT_GGUF_PY_DIR}:"
fi

is_qwen35_model_name() {
  case "${1,,}" in
    *qwen3.5*|*qwen3_5*) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ "${ASYM_QWEN35_FA4_AUTO}" == "1" ]] && is_qwen35_model_name "${MODEL_NAME_OR_PATH}"; then
  [[ "${_LF_DIR_ENV_SET}" == "true" ]] || LF_DIR="${LF_FA4_DIR}"
  [[ "${_ENV_DIR_ENV_SET}" == "true" ]] || ENV_DIR="${ASYM_DIR}/.venv-fa4"
  [[ "${_FLASH_ATTN_ENV_SET}" == "true" ]] || FLASH_ATTN=fa4
fi

# Persist FA4 cute JIT kernels across runs (flash-attn-4 defaults this off; default dir is gitignored).
if [[ "${FLASH_ATTN}" == "fa4" ]]; then
  export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED="${FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED:-1}"
  export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR="${FLASH_ATTENTION_CUTE_DSL_CACHE_DIR:-${ROOT}/.cache/fa4_cute_dsl}"
  mkdir -p "${FLASH_ATTENTION_CUTE_DSL_CACHE_DIR}" 2>/dev/null || true
fi

KT_RUN_PYTHONPATH="${KT_TOOLS_DIR}:${ASYM_DIR}:${KT_KERNEL_DIR}:${KT_GGUF_PYTHONPATH_ENTRY}${LF_DIR}/src:${PYTHONPATH:-}"
ENV_DIR=${ENV_DIR:-${ASYM_DIR}/.venv}
ENV_PYTHON=${ENV_PYTHON:-${ENV_DIR}/bin/python}
LF_CLI_BIN=${LF_CLI_BIN:-${ENV_DIR}/bin/llamafactory-cli}
TORCHRUN_BIN=${TORCHRUN_BIN:-${ENV_DIR}/bin/torchrun}
ACCELERATE_BIN=${ACCELERATE_BIN:-${ENV_DIR}/bin/accelerate}
DEEPSPEED_BIN=${DEEPSPEED_BIN:-${ENV_DIR}/bin/deepspeed}
CHECK_SUPEROFFLOAD_SCRIPT=${CHECK_SUPEROFFLOAD_SCRIPT:-${ASYM_DIR}/scripts/lf/check_superoffload_run.py}
CHECK_CPUADAM_SCRIPT=${CHECK_CPUADAM_SCRIPT:-${ASYM_DIR}/scripts/lf/check_deepspeed_cpuadam_run.py}
unset KT_BACKEND                      # Not user-facing; derive the KT enum only from BACKEND.
KT_BACKEND_INTERNAL=""
ZERO_BACKEND_LABEL=""
TORCH_DEEPSPEED_CONFIG=""
CPUADAM_ALIAS_SELECTED=0
TORCHRUN_CMD=()
ACCELERATE_CMD=()
DEEPSPEED_CMD=()

zero_deepspeed_config() {
  local _name
  case "${1}" in
    zero2) _name=ds_z2_config.json ;;
    zero3) _name=ds_z3_config.json ;;
    zero3_offload) _name=ds_z3_offload_config.json ;;
    zero3_offload_mem) _name=ds_z3_offload_mem_config.json ;;
    zero3_offload_mem_nocpuadamw) _name=ds_z3_offload_mem_nocpuadamw_config.json ;;
    zero3_offload_opnvme) _name=ds_z3_offload_opnvme_config.json ;;
    zero3_offload_panvme) _name=ds_z3_offload_panvme_config.json ;;
    zero3_offload_mem_opnvme) _name=ds_z3_offload_mem_opnvme_config.json ;;
    zero3_offload_mem_panvme) _name=ds_z3_offload_mem_panvme_config.json ;;
    zero3_cpuadam) _name=ds_z3_cpuadam_config.json ;;
    superoffload) _name=ds_z3_superoffload_config.json ;;
    superoffload_mem) _name=ds_z3_superoffload_mem_config.json ;;
    superoffload_mem_nocpuadamw) _name=ds_z3_superoffload_mem_nocpuadamw_config.json ;;
    superoffload_mem_opnvme) _name=ds_z3_superoffload_mem_opnvme_config.json ;;
    superoffload_mem_panvme) _name=ds_z3_superoffload_mem_panvme_config.json ;;
    *) return 1 ;;
  esac
  if [[ -f "${LF_DIR}/examples/deepspeed/${_name}" ]]; then
    printf '%s\n' "${LF_DIR}/examples/deepspeed/${_name}"
  else
    printf '%s\n' "${LF_DS_CONFIG_DIR}/${_name}"
  fi
}

case "${BACKEND,,}" in
  torch)
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-torch}
    BACKEND=torch
    ;;
  zero2)
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-zero2}
    ZERO_BACKEND_LABEL=zero2
    BACKEND=torch
    TORCH_DEEPSPEED_CONFIG="$(zero_deepspeed_config zero2)"
    ;;
  zero3)
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-zero3}
    ZERO_BACKEND_LABEL=zero3
    BACKEND=torch
    TORCH_DEEPSPEED_CONFIG="$(zero_deepspeed_config zero3)"
    ;;
  zero3_offload)
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-zero3_offload}
    ZERO_BACKEND_LABEL=zero3_offload
    BACKEND=torch
    TORCH_DEEPSPEED_CONFIG="$(zero_deepspeed_config zero3_offload)"
    ;;
  zero3_offload_mem)
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-zero3_offload_mem}
    ZERO_BACKEND_LABEL=zero3_offload_mem
    BACKEND=torch
    TORCH_DEEPSPEED_CONFIG="$(zero_deepspeed_config zero3_offload_mem)"
    ;;
  zero3_offload_mem_nocpuadamw)
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-zero3_offload_mem_nocpuadamw}
    ZERO_BACKEND_LABEL=zero3_offload_mem_nocpuadamw
    BACKEND=torch
    TORCH_DEEPSPEED_CONFIG="$(zero_deepspeed_config zero3_offload_mem_nocpuadamw)"
    ;;
  zero3_offload_opnvme)
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-zero3_offload_opnvme}
    ZERO_BACKEND_LABEL=zero3_offload_opnvme
    BACKEND=torch
    TORCH_DEEPSPEED_CONFIG="$(zero_deepspeed_config zero3_offload_opnvme)"
    ;;
  zero3_offload_panvme)
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-zero3_offload_panvme}
    ZERO_BACKEND_LABEL=zero3_offload_panvme
    BACKEND=torch
    TORCH_DEEPSPEED_CONFIG="$(zero_deepspeed_config zero3_offload_panvme)"
    ;;
  zero3_offload_mem_opnvme)
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-zero3_offload_mem_opnvme}
    ZERO_BACKEND_LABEL=zero3_offload_mem_opnvme
    BACKEND=torch
    TORCH_DEEPSPEED_CONFIG="$(zero_deepspeed_config zero3_offload_mem_opnvme)"
    ;;
  zero3_offload_mem_panvme)
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-zero3_offload_mem_panvme}
    ZERO_BACKEND_LABEL=zero3_offload_mem_panvme
    BACKEND=torch
    TORCH_DEEPSPEED_CONFIG="$(zero_deepspeed_config zero3_offload_mem_panvme)"
    ;;
  zero3_cpuadam)
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-zero3_cpuadam}
    ZERO_BACKEND_LABEL=zero3_cpuadam
    BACKEND=torch
    TORCH_DEEPSPEED_CONFIG="$(zero_deepspeed_config zero3_cpuadam)"
    ;;
  superoffload)
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-superoffload}
    ZERO_BACKEND_LABEL=superoffload
    BACKEND=torch
    TORCH_DEEPSPEED_CONFIG="$(zero_deepspeed_config superoffload)"
    ;;
  superoffload_mem)
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-superoffload_mem}
    ZERO_BACKEND_LABEL=superoffload_mem
    BACKEND=torch
    TORCH_DEEPSPEED_CONFIG="$(zero_deepspeed_config superoffload_mem)"
    ;;
  superoffload_mem_nocpuadamw)
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-superoffload_mem_nocpuadamw}
    ZERO_BACKEND_LABEL=superoffload_mem_nocpuadamw
    BACKEND=torch
    TORCH_DEEPSPEED_CONFIG="$(zero_deepspeed_config superoffload_mem_nocpuadamw)"
    ;;
  superoffload_mem_opnvme)
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-superoffload_mem_opnvme}
    ZERO_BACKEND_LABEL=superoffload_mem_opnvme
    BACKEND=torch
    TORCH_DEEPSPEED_CONFIG="$(zero_deepspeed_config superoffload_mem_opnvme)"
    ;;
  superoffload_mem_panvme)
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-superoffload_mem_panvme}
    ZERO_BACKEND_LABEL=superoffload_mem_panvme
    BACKEND=torch
    TORCH_DEEPSPEED_CONFIG="$(zero_deepspeed_config superoffload_mem_panvme)"
    ;;
  asym_stp|asym_stp_cpuadamwds|tp2_resident_cpuadamwds|tp2_offstage_cpuadamwds)
    # GB200 sTP family (gb200_tp.md): single-process TP-2 over a same-superchip GPU pair.
    # BACKEND stays asym so is_torch_run remains false => ONE process drives both GPUs.
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-${BACKEND,,}}
    STP_BACKEND_RAW="${BACKEND,,}"
    USE_ASYM_CPU_ADAMW=true
    ASYM_CPU_ADAMW_BACKEND=deepspeed
    CPUADAM_ALIAS_SELECTED=1
    ASYM_STP=1
    # Post-accumulate hooks (grad D2H + weight release) fire BEFORE the sTP mirror merge;
    # LoRA banks stay GPU-resident (adapter-scale).
    ASYM_CPU_ADAMW_GRAD_OFFLOAD=false
    ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false
    BACKEND=asym
    ;;
  asym_dp2_cpuadamwds)
    # GB200 real-DP pair (gb200_dp.md D2 Route A): torchrun 2-proc + HF DDP over the LoRA params.
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-asym_dp2_cpuadamwds}
    USE_ASYM_CPU_ADAMW=true
    ASYM_CPU_ADAMW_BACKEND=deepspeed
    CPUADAM_ALIAS_SELECTED=1
    ASYM_DP=1
    # Route A ordering hazard: DDP's bucketed all-reduce completes at backward END; the
    # per-param post-accumulate grad-offload hooks would D2H PRE-reduction grads. Hook-based
    # offload stays OFF; CPUAdamW's step-time path reads param.grad after DDP finalization.
    ASYM_CPU_ADAMW_GRAD_OFFLOAD=false
    ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false
    BACKEND=asym
    ;;
  asym_ep2_cpuadamwds|asym_sdp2_cpuadamwds|asym_sep2_cpuadamwds|asym_sqdp2_cpuadamwds|asym_sqep2_cpuadamwds|asym_sqeq2_cpuadamwds)
    # GB200 rank-per-GPU EP family (fix_gb200_ep.md; NAMING EPOCH 2026-07-08 — the
    # backend NAME selects the EP mode and flips the flags automatically):
    #   asym_ep2_cpuadamwds   = VANILLA EP  (owned expert slices, allgather dispatch +
    #                           reduce-scatter combine; private per-rank pins, no queue)
    #   asym_sep2_cpuadamwds  = sEP plain   (ownerless shared /dev/shm fabric, each rank
    #                           runs ALL experts for its own shard; no queue)
    #   asym_sqep2_cpuadamwds = sEP + QUEUE (the final system: ownerless fabric + queued
    #                           launches; asym_sqeq2_* accepted as an alias spelling)
    # Common shape: torchrun 2 ranks, each rank the |1 asym stack VERBATIM on its own
    # DistributedSampler shard; NO DDP wrapper — run_lf_profiled_train.py bypasses
    # Accelerator.prepare_model and does ONE manual grad allreduce per step.
    # NAMING EPOCH 3 (2026-07-09): canonical names asym_sdp2/asym_sqdp2 — shared-bank
    # streaming DP (honest taxonomy; sep2/sqep2/sqeq2 remain accepted aliases).
    case "${BACKEND,,}" in
      asym_sqeq2_cpuadamwds) BACKEND=asym_sqep2_cpuadamwds ;;
    esac
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-${BACKEND,,}}
    USE_ASYM_CPU_ADAMW=true
    ASYM_CPU_ADAMW_BACKEND=deepspeed
    CPUADAM_ALIAS_SELECTED=1
    ASYM_DP=1          # reuse the torchrun launch path + the LF parser distributed unlock
    ASYM_EP2=1
    case "${BACKEND,,}" in
      asym_ep2_cpuadamwds)
        ASYM_EP_VANILLA=1; ASYM_ARENA_SHM=0; ASYM_EP_QUEUED=0
        echo "[runlf] backend asym_ep2 -> VANILLA EP (owned+dispatch): ASYM_EP_VANILLA=1 ASYM_ARENA_SHM=0 ASYM_EP_QUEUED=0" ;;
      asym_sdp2_cpuadamwds)
        ASYM_EP_VANILLA=0; ASYM_ARENA_SHM=1; ASYM_EP_QUEUED=0
        echo "[runlf] backend asym_sdp2 -> shared-bank streaming DP (no queue): ASYM_ARENA_SHM=1 ASYM_EP_QUEUED=0" ;;
      asym_sqdp2_cpuadamwds)
        ASYM_EP_VANILLA=0; ASYM_ARENA_SHM=1; ASYM_EP_QUEUED=1
        echo "[runlf] backend asym_sqdp2 -> shared-bank streaming DP + queue: ASYM_ARENA_SHM=1 ASYM_EP_QUEUED=1" ;;
      asym_sep2_cpuadamwds)
        ASYM_EP_VANILLA=0; ASYM_ARENA_SHM=1; ASYM_EP_QUEUED=0; ASYM_EP_SEP=1
        echo "[runlf] backend asym_sep2 -> TRUE sEP (S6 union work sharing): ASYM_EP_SEP=1 ASYM_ARENA_SHM=1" ;;
      asym_sqep2_cpuadamwds)
        ASYM_EP_VANILLA=0; ASYM_ARENA_SHM=1; ASYM_EP_QUEUED=1; ASYM_EP_SEP=1
        echo "[runlf] backend asym_sqep2 -> TRUE sEP + queued unarmed launches: ASYM_EP_SEP=1 ASYM_EP_QUEUED=1" ;;
    esac
    # same hook-offload hazard as dp2: grads must be reduced BEFORE any D2H copy; the
    # CPUAdamW step-time path reads param.grad after the manual allreduce.
    ASYM_CPU_ADAMW_GRAD_OFFLOAD=false
    ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false
    BACKEND=asym
    ;;
  asym_cpuadamwtorch)
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-asym_cpuadamwtorch}
    USE_ASYM_CPU_ADAMW=true
    ASYM_CPU_ADAMW_BACKEND=torch
    CPUADAM_ALIAS_SELECTED=1
    BACKEND=asym
    ;;
  asym_cpuadamwds)
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-asym_cpuadamwds}
    USE_ASYM_CPU_ADAMW=true
    ASYM_CPU_ADAMW_BACKEND=deepspeed
    CPUADAM_ALIAS_SELECTED=1
    BACKEND=asym
    ;;
  asym_cpuadamwds_panvme|asym_cpuadamwds_actnvme|asym_cpuadamwds_bothnvme)
    # asym_cpuadamwds + one/both NVMe roles (activation spill and/or base-weight paging).
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-${BACKEND,,}}
    USE_ASYM_CPU_ADAMW=true
    ASYM_CPU_ADAMW_BACKEND=deepspeed
    CPUADAM_ALIAS_SELECTED=1
    case "${BACKEND,,}" in
      *_panvme)   ASYM_NVME_ROLES="base_weight" ;;
      *_actnvme)  ASYM_NVME_ROLES="activation" ;;
      *_bothnvme) ASYM_NVME_ROLES="base_weight,activation" ;;
    esac
    export ASYM_NVME_ROLES
    export ASYM_NVME_PATH="${ASYM_NVME_PATH:-/scratch_local/asym_nvme}"
    # Stage 3: enable the LF checkpointing.py boundary-offload hook when activation spill is on.
    if [[ ",${ASYM_NVME_ROLES}," == *",activation,"* ]]; then export ASYM_UNSLOTH_GC_NVME=1; fi
    BACKEND=asym
    ;;
  asym_torch) BACKEND=asym_torch ;;
  asym) BACKEND=asym ;;
  kt_torchbf16)
    BACKEND=kt_torchbf16
    KT_BACKEND_INTERNAL=TORCHBF16
    ;;
  kt_armbf16)
    BACKEND=kt_armbf16
    KT_BACKEND_INTERNAL=ARMBF16
    ;;
  *) echo "BACKEND must be one of: torch, zero2, zero3, zero3_offload, zero3_offload_mem, zero3_offload_mem_nocpuadamw, zero3_offload_opnvme, zero3_offload_panvme, zero3_offload_mem_opnvme, zero3_offload_mem_panvme, zero3_cpuadam, superoffload, superoffload_mem, superoffload_mem_nocpuadamw, superoffload_mem_opnvme, superoffload_mem_panvme, asym_cpuadamwtorch, asym_cpuadamwds, asym_torch, asym, kt_torchbf16, kt_armbf16; got '${BACKEND}'" >&2; exit 2 ;;
esac

case "${DIST_LAUNCHER,,}" in
  torchrun) DIST_LAUNCHER=torchrun ;;
  accelerate|accelerate_launch) DIST_LAUNCHER=accelerate ;;
  deepspeed|ds) DIST_LAUNCHER=deepspeed ;;
  *) echo "DIST_LAUNCHER must be torchrun, accelerate, or deepspeed, got '${DIST_LAUNCHER}'" >&2; exit 2 ;;
esac

if [[ "${BACKEND}" == "torch" || "${BACKEND}" == kt_* ]]; then
  PROFILE_EXPERT_POLICY=none
fi

if [[ "${BACKEND}" == kt_* && "${NUM_GPUS}" != "1" ]]; then
  echo "KT SFT profiling is single-process/single-GPU for this script; got NUM_GPUS=${NUM_GPUS}" >&2
  exit 2
fi

# ---- sEP guards (gb200_ep.md E0/HC-EP4) ----
if [[ -n "${ASYM_EP_MODE:-}" && "${ASYM_EP_E2E_LANDED:-0}" != "1" ]]; then
  echo "ASYM_EP_MODE='${ASYM_EP_MODE}' locked until sEP e2e stages land (gb200_ep.md)" >&2
  exit 2
fi
if [[ -n "${ASYM_EP_SKEW_HOT:-}" && "${ASYM_EP_SKEW_ACK:-0}" != "1" ]]; then
  echo "ASYM_EP_SKEW_HOT is loss-INVALID (HC-EP1); set ASYM_EP_SKEW_ACK=1" >&2
  exit 2
fi

# ---- GB200 |2 guards (gb200_tp.md HC1-HC3/I0; gb200_dp.md HC4) ----
if [[ "${ASYM_STP:-0}" == "1" && "${ASYM_DP:-0}" == "1" ]]; then
  echo "ASYM_STP and ASYM_DP are mutually exclusive (gb200_dp.md HC4)" >&2
  exit 2
fi
if [[ "${ASYM_STP:-0}" == "1" || "${ASYM_DP:-0}" == "1" ]]; then
  if [[ "${NUM_GPUS}" != "2" ]]; then
    echo "|2 backends (ASYM_STP/ASYM_DP) require NUM_GPUS=2, got ${NUM_GPUS}" >&2
    exit 2
  fi
  if [[ "${TRAIN_OOM_SCORE_ADJ}" != "1000" ]]; then
    echo "HC2: TRAIN_OOM_SCORE_ADJ must stay 1000 under |2 backends, got '${TRAIN_OOM_SCORE_ADJ}'" >&2
    exit 2
  fi
  case "${HOST_MEM_WATCHDOG,,}" in
    1|true|yes|y|on) ;;
    *) echo "HC2: HOST_MEM_WATCHDOG must be enabled under |2 backends, got '${HOST_MEM_WATCHDOG}'" >&2; exit 2 ;;
  esac
  case "${GPU_ID}" in
    0,1|2,3) ;;
    *)
      if [[ "${ALLOW_CROSS_SUPERCHIP:-0}" != "1" ]]; then
        echo "|2 rows must use a same-superchip pair (GPU_ID=0,1 or 2,3); got '${GPU_ID}'." \
             "Set ALLOW_CROSS_SUPERCHIP=1 only for the contention study." >&2
        exit 2
      fi
      ;;
  esac
fi
if [[ "${ASYM_STP:-0}" == "1" ]]; then
  ASYM_STP_TP_SIZE="${ASYM_STP_TP_SIZE:-2}"
  _stp_mode_for_backend() {
    case "${1}" in
      asym_stp*) echo stream ;;
      tp2_resident*) echo resident ;;
      tp2_offstage*) echo stage ;;
      *) echo stream ;;
    esac
  }
  _stp_derived_mode="$(_stp_mode_for_backend "${STP_BACKEND_RAW:-asym_stp_cpuadamwds}")"
  ASYM_STP_WEIGHT_MODE="${ASYM_STP_WEIGHT_MODE:-${_stp_derived_mode}}"
  if [[ "${ASYM_STP_WEIGHT_MODE}" != "${_stp_derived_mode}" ]]; then
    echo "ASYM_STP_WEIGHT_MODE='${ASYM_STP_WEIGHT_MODE}' mismatches backend '${STP_BACKEND_RAW:-}' (derived '${_stp_derived_mode}')" >&2
    exit 2
  fi
  case "${STP_BACKEND_RAW:-}" in
    tp2_*)
      # Rung purity: arena/coord CLAIMS are ablated on asym_stp only; tp2_* pins both off.
      ASYM_STP_SHARED_ARENA=0
      ASYM_STP_COORD=0
      ;;
    *)
      ASYM_STP_SHARED_ARENA="${ASYM_STP_SHARED_ARENA:-1}"
      ASYM_STP_COORD="${ASYM_STP_COORD:-1}"
      ;;
  esac
  case "${LORA_DROPOUT}" in
    0|0.0|0.00) ;;
    *) echo "ASYM_STP requires LORA_DROPOUT=0.00 (x0==x1 bit-identity; gb200_tp.md I0)" >&2; exit 2 ;;
  esac
  case "${MODEL_NAME_OR_PATH}" in
    *A3B*|*a3b*|*A22B*|*a22b*|*Llama-4*|*llama-4*|*16E*)
      if [[ "${ASYM_STP_MOE:-0}" != "1" ]]; then
        echo "ASYM_STP on MoE model '${MODEL_NAME_OR_PATH}' is locked until Stage I7 lands (needs ASYM_STP_MOE=1)" >&2
        exit 2
      fi
      ;;
  esac
  if [[ "${MODEL_NAME_OR_PATH}" == *Llama-4* && "${ASYM_ROUTER_MODE:-whole}" != "whole" ]]; then
    echo "ASYM_STP on llama4 requires ASYM_ROUTER_MODE=whole (router stays detached)" >&2
    exit 2
  fi
  export ASYM_STP ASYM_STP_TP_SIZE ASYM_STP_WEIGHT_MODE ASYM_STP_SHARED_ARENA ASYM_STP_COORD
  export ASYM_STP_MOE="${ASYM_STP_MOE:-0}"
fi
if [[ "${ASYM_DP:-0}" == "1" ]]; then
  export ASYM_DP
fi
if [[ "${ASYM_EP2:-0}" == "1" ]]; then
  # rank-per-GPU sEP (fix_gb200_ep.md S1): the trainer reads these to bypass the DDP
  # wrap, run the manual grad allreduce, and build/seal the shared /dev/shm fabric.
  export ASYM_EP2
  export ASYM_ARENA_SHM="${ASYM_ARENA_SHM:-1}"
  export ASYM_ARENA_SHM_CAP_GB="${ASYM_ARENA_SHM_CAP_GB:-160}"
  [[ -z "${ASYM_EP_QUEUED:-}" ]] || export ASYM_EP_QUEUED  # S2a queued base GEMMs
  [[ -z "${ASYM_EP_VANILLA:-}" ]] || export ASYM_EP_VANILLA  # S5b vanilla-EP baseline rung
  [[ -z "${ASYM_EP_SEP:-}" ]] || export ASYM_EP_SEP          # S6 true-sEP union sharing
  [[ -z "${ASYM_EP_SEP_SLOT_ROWS:-}" ]] || export ASYM_EP_SEP_SLOT_ROWS
  [[ -z "${ASYM_EP_SEP_MAX_MPE:-}" ]] || export ASYM_EP_SEP_MAX_MPE
fi

if [[ "${BACKEND}" == "kt_armbf16" && -z "${KT_NUM_THREADS}" ]]; then
  KT_NUM_THREADS="${KT_ARM_OMP_NUM_THREADS}"
fi

if [[ -z "${PREPROCESSING_NUM_WORKERS}" ]]; then
  if [[ "${BACKEND}" == "kt_armbf16" ]]; then
    PREPROCESSING_NUM_WORKERS=1
  else
    PREPROCESSING_NUM_WORKERS=4
  fi
fi
if [[ -z "${DATALOADER_NUM_WORKERS}" ]]; then
  if [[ "${BACKEND}" == "kt_armbf16" ]]; then
    DATALOADER_NUM_WORKERS=0
  else
    DATALOADER_NUM_WORKERS=2
  fi
fi
if [[ "${BACKEND}" == "kt_armbf16" && -z "${MAX_GRAD_NORM}" ]]; then
  MAX_GRAD_NORM=0
fi
if [[ "${BACKEND}" == "kt_armbf16" && "${PROFILE}" == "1" ]]; then
  KT_SFT_PROGRESS=${KT_SFT_PROGRESS:-1}
  KT_ARM_SFT_PROFILE=${KT_ARM_SFT_PROFILE:-1}
  KT_ARM_SFT_POOL_LOG=${KT_ARM_SFT_POOL_LOG:-1}
fi

if [[ ! "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer, got '${NUM_GPUS}'" >&2
  exit 2
fi

RUN_BACKEND_LABEL="${PROFILE_BACKEND_LABEL:-${BACKEND}}"

infer_template() {
  local model="$1"
  local lower="${model,,}"
  local base="${lower##*/}"
  case "${base}" in
    gemma-4-*|gemma4-*) printf 'gemma4\n' ;;
    llama-4-*|llama4-*) printf 'llama4\n' ;;
    llama-3*|llama3-*|meta-llama-3*) printf 'llama3\n' ;;
    llama-2*|llama2-*) printf 'llama2\n' ;;
    qwen2-vl-*|qwen2.5-vl-*|qvq-*) printf 'qwen2_vl\n' ;;
    qwen2-audio-*) printf 'qwen2_audio\n' ;;
    qwen2.5-omni-*) printf 'qwen2_omni\n' ;;
    qwen3-omni-*-thinking*) printf 'qwen3_omni\n' ;;
    qwen3-omni-*) printf 'qwen3_omni_nothink\n' ;;
    qwen3-vl-*-thinking*) printf 'qwen3_vl\n' ;;
    qwen3-vl-*) printf 'qwen3_vl_nothink\n' ;;
    qwen3.5-*) printf 'qwen3_5\n' ;;
    qwen3-*-instruct*|qwen3-next-*-instruct*) printf 'qwen3_nothink\n' ;;
    qwen3-*|qwen3-next-*) printf 'qwen3\n' ;;
    qwen2.5-*|qwen2-*|qwq-*) printf 'qwen\n' ;;
    *) echo "error: TEMPLATE=auto cannot infer a safe template for '${model}'. Set TEMPLATE explicitly." >&2; return 2 ;;
  esac
}

is_llama4_model() {
  case "${1,,}" in
    *llama-4*|*llama4*) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ "${TEMPLATE}" == "auto" ]]; then
  TEMPLATE="$(infer_template "${MODEL_NAME_OR_PATH}")"
fi
if [[ -z "${TRAINING_BF16}" ]]; then
  if is_llama4_model "${MODEL_NAME_OR_PATH}" && [[ "${BACKEND}" == "asym" || "${BACKEND}" == "asym_torch" ]]; then
    TRAINING_BF16=false
  else
    TRAINING_BF16=true
  fi
fi
case "${TRAINING_BF16,,}" in
  1|true|yes|y|on) TRAINING_BF16=true ;;
  0|false|no|n|off) TRAINING_BF16=false ;;
  *) echo "TRAINING_BF16 must be true or false, got '${TRAINING_BF16}'" >&2; exit 2 ;;
esac
case "${ENABLE_LIGER_KERNEL,,}" in
  1|true|yes|y|on) ENABLE_LIGER_KERNEL=true ;;
  0|false|no|n|off) ENABLE_LIGER_KERNEL=false ;;
  *) echo "ENABLE_LIGER_KERNEL must be true or false, got '${ENABLE_LIGER_KERNEL}'" >&2; exit 2 ;;
esac
case "${USE_UNSLOTH_GC,,}" in
  1|true|yes|y|on) USE_UNSLOTH_GC=true ;;
  0|false|no|n|off) USE_UNSLOTH_GC=false ;;
  *) echo "USE_UNSLOTH_GC must be true or false, got '${USE_UNSLOTH_GC}'" >&2; exit 2 ;;
esac
case "${UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU,,}" in
  1|true|yes|y|on) UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=true ;;
  0|false|no|n|off) UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=false ;;
  *) echo "UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU must be true or false, got '${UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU}'" >&2; exit 2 ;;
esac
if ! [[ "${UNSLOTH_GC_OUTER_HBM_EVERY_N}" =~ ^[0-9]+$ ]]; then
  echo "UNSLOTH_GC_OUTER_HBM_EVERY_N must be a non-negative integer, got '${UNSLOTH_GC_OUTER_HBM_EVERY_N}'" >&2
  exit 2
fi
# Unsloth GC only installs when gradient checkpointing is enabled (LF prepare_model_for_training).
if [[ "${USE_UNSLOTH_GC}" == "true" ]]; then
  case "${GRADIENT_CHECKPOINTING,,}" in
    1|true|yes|y|on) : ;;
    *) echo "USE_UNSLOTH_GC=true requires GRADIENT_CHECKPOINTING=true" >&2; exit 2 ;;
  esac
fi
if [[ "${UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU}" == "true" && "${USE_UNSLOTH_GC}" != "true" ]]; then
  echo "UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=true requires USE_UNSLOTH_GC=true" >&2
  exit 2
fi
if [[ "${UNSLOTH_GC_OUTER_HBM_EVERY_N}" != "0" && "${USE_UNSLOTH_GC}" != "true" ]]; then
  echo "UNSLOTH_GC_OUTER_HBM_EVERY_N>0 requires USE_UNSLOTH_GC=true" >&2
  exit 2
fi
if [[ "${ENABLE_LIGER_KERNEL}" == "true" ]]; then
  LIGER_LOSS_TAG=ligerloss1
else
  LIGER_LOSS_TAG=ligerloss0
fi

MODEL_TAG=$(basename "${MODEL_NAME_OR_PATH}" | tr '/:' '__')
case "${FINETUNING_TYPE,,}" in
  lora|full) FINETUNING_TYPE="${FINETUNING_TYPE,,}" ;;
  *) echo "FINETUNING_TYPE must be lora or full; got '${FINETUNING_TYPE}'" >&2; exit 2 ;;
esac
case "${LF_QWEN_MOE_EXPERT_LORA_IMPL,,}" in
  peft-target-parameters|split-target-parameters|off) LF_QWEN_MOE_EXPERT_LORA_IMPL="${LF_QWEN_MOE_EXPERT_LORA_IMPL,,}" ;;
  *) echo "LF_QWEN_MOE_EXPERT_LORA_IMPL must be peft-target-parameters, split-target-parameters, or off; got '${LF_QWEN_MOE_EXPERT_LORA_IMPL}'" >&2; exit 2 ;;
esac
QWEN_EXPERT_LORA_TAG=$(printf '%s' "${LF_QWEN_MOE_EXPERT_LORA_IMPL}" | tr '/:' '__' | tr -c '[:alnum:]_-' '_')
EXPERT_POLICY_TAG=$(printf '%s' "${ASYM_EXPERT_RECOMPUTE_POLICY}" | tr '/:' '__' | tr -c '[:alnum:]_-' '_')
case "${ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD,,}" in
  1|true|yes|y|on) ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD=1; QWEN3_MOE_FINEGRAINED_TAG=moefg1 ;;
  0|false|no|n|off) ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD=0; QWEN3_MOE_FINEGRAINED_TAG=moefg0 ;;
  *) echo "ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD must be true or false, got '${ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD}'" >&2; exit 2 ;;
esac
if [[ ! "${ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS}" =~ ^[0-9]+$ ]]; then
  echo "ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS must be a nonnegative integer, got '${ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS}'" >&2
  exit 2
fi
QWEN3_MOE_DOWN_SCATTER_TAG="dscatter${ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS}"
case "${ASYMM_EXPERT_ACT_OFFLOAD,,}" in
  1|true|yes|y|on) ASYMM_EXPERT_ACT_OFFLOAD=true; EXP_ACT_OFFLOAD_TAG=expact1 ;;
  0|false|no|n|off) ASYMM_EXPERT_ACT_OFFLOAD=false; EXP_ACT_OFFLOAD_TAG=expact0 ;;
  *) echo "ASYMM_EXPERT_ACT_OFFLOAD must be true or false, got '${ASYMM_EXPERT_ACT_OFFLOAD}'" >&2; exit 2 ;;
esac
case "${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}" in
  cpu) ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu; EXP_ACT_LORA_A_FWD_TAG=loraafwdcpu ;;
  hbm) ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=hbm; EXP_ACT_LORA_A_FWD_TAG=loraafwdhbm ;;
  *) echo "ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD must be cpu or hbm, got '${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}'" >&2; exit 2 ;;
esac
case "${ASYMM_ATTN_ACT_OFFLOAD,,}" in
  1|true|yes|y|on) ASYMM_ATTN_ACT_OFFLOAD=true; ATTN_ACT_OFFLOAD_TAG=attnact1 ;;
  0|false|no|n|off) ASYMM_ATTN_ACT_OFFLOAD=false; ATTN_ACT_OFFLOAD_TAG=attnact0 ;;
  *) echo "ASYMM_ATTN_ACT_OFFLOAD must be true or false, got '${ASYMM_ATTN_ACT_OFFLOAD}'" >&2; exit 2 ;;
esac
case "${ASYMM_LAYER_ACT_OFFLOAD,,}" in
  1|true|yes|y|on) ASYMM_LAYER_ACT_OFFLOAD=true; LAYER_ACT_OFFLOAD_TAG=layeract1 ;;
  0|false|no|n|off) ASYMM_LAYER_ACT_OFFLOAD=false; LAYER_ACT_OFFLOAD_TAG=layeract0 ;;
  *) echo "ASYMM_LAYER_ACT_OFFLOAD must be true or false, got '${ASYMM_LAYER_ACT_OFFLOAD}'" >&2; exit 2 ;;
esac
case "${ASYMM_LAYER_GC,,}" in
  1|true|yes|y|on) ASYMM_LAYER_GC=true; LAYER_GC_TAG=layergc1 ;;
  0|false|no|n|off) ASYMM_LAYER_GC=false; LAYER_GC_TAG=layergc0 ;;
  *) echo "ASYMM_LAYER_GC must be true or false, got '${ASYMM_LAYER_GC}'" >&2; exit 2 ;;
esac
ATTN_GC_ENABLED=false
if [[ "${ASYM_EXPERT_RECOMPUTE_POLICY}" == "gc-attn-exp" ]]; then
  ATTN_GC_ENABLED=true
fi
LAYER_GC_ENABLED=false
if [[ "${ASYM_EXPERT_RECOMPUTE_POLICY}" == "gc-layer" ]]; then
  LAYER_GC_ENABLED=true
fi
RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
PROFILE_TAG="prof${PROFILE}_${PROFILE_PROFILER}_${PROFILE_LEVEL}"
uses_asym_deepspeed_cpuadam=false
if [[ "${BACKEND}" == "asym" && "${USE_ASYM_CPU_ADAMW}" == "true" && "${ASYM_CPU_ADAMW_BACKEND}" == "deepspeed" ]]; then
  uses_asym_deepspeed_cpuadam=true
fi

OUTER_HBM_RUN_TAG=""
if [[ "${UNSLOTH_GC_OUTER_HBM_EVERY_N:-0}" != "0" ]]; then
  OUTER_HBM_RUN_TAG="_ohbm${UNSLOTH_GC_OUTER_HBM_EVERY_N}"
fi
if [[ "${BACKEND}" == kt_* ]]; then
  KT_BACKEND_TAG="${KT_BACKEND_INTERNAL:-none}"
  DEFAULT_RUN_ID="${RUN_TS}_${MODEL_TAG}_${BACKEND}_${KT_BACKEND_TAG}_${KT_PRECISION}_ctx${CUTOFF_LEN}_bs${PER_DEVICE_TRAIN_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}_r${LORA_RANK}_a${LORA_ALPHA}_steps${MAX_STEPS}_${EXP_ACT_LORA_A_FWD_TAG}_qwenexpert${QWEN_EXPERT_LORA_TAG}_${QWEN3_MOE_FINEGRAINED_TAG}_${QWEN3_MOE_DOWN_SCATTER_TAG}_${LIGER_LOSS_TAG}_${PROFILE_TAG}"
else
  DEFAULT_RUN_ID="${RUN_TS}_${MODEL_TAG}_${RUN_BACKEND_LABEL}_${ASYM_PRECISION}_ctx${CUTOFF_LEN}_bs${PER_DEVICE_TRAIN_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}_r${LORA_RANK}_a${LORA_ALPHA}_steps${MAX_STEPS}_offload${ASYM_OFFLOAD_MODULES}_pol${EXPERT_POLICY_TAG}_router${ASYM_ROUTER_MODE}_${EXP_ACT_OFFLOAD_TAG}_${ATTN_ACT_OFFLOAD_TAG}_${EXP_ACT_LORA_A_FWD_TAG}_qwenexpert${QWEN_EXPERT_LORA_TAG}_${QWEN3_MOE_FINEGRAINED_TAG}_${QWEN3_MOE_DOWN_SCATTER_TAG}_${LIGER_LOSS_TAG}${OUTER_HBM_RUN_TAG}_${PROFILE_TAG}"
fi
RUN_ID=${RUN_ID:-${DEFAULT_RUN_ID}}
if [[ "${BACKEND}" == kt_* ]]; then
  OUT_DIR=${OUT_DIR:-${LF_DIR}/saves/kt_smoke/${RUN_ID}}
else
  OUT_DIR=${OUT_DIR:-${LF_DIR}/saves/asymgemm_smoke/${RUN_ID}}
fi
if [[ "${BACKEND}" == kt_* && -z "${TRITON_CACHE_DIR:-}" ]]; then
  TRITON_CACHE_DIR="${TMPDIR:-/tmp}/asymgemm_triton_cache/${RUN_ID}"
fi
LOG_FILE=${LOG_FILE:-${OUT_DIR}/train_${RUN_ID}.log}
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

profile_memory_flag() {
  local name="$1"
  local value="${2,,}"
  local profiler="$3"
  case "${value}" in
    auto)
      if [[ "${profiler}" == "source" ]]; then
        printf 'true\n'
      else
        printf 'false\n'
      fi
      ;;
    1|true|yes|y|on) printf 'true\n' ;;
    0|false|no|n|off) printf 'false\n' ;;
    *) echo "${name} must be auto, true, or false" >&2; exit 2 ;;
  esac
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

find_free_port() {
  python3 - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

is_torch_run() {
  # ASYM_DP=1 (asym_dp2, gb200_dp.md D2) keeps BACKEND=asym for the LF surgery but launches
  # torchrun 2-proc so HF DDP shards data + all-reduces the LoRA grads.
  [[ "${BACKEND}" == "torch" || "${ASYM_DP:-0}" == "1" ]]
}

is_zero_backend_run() {
  [[ "${BACKEND}" == "torch" && -n "${ZERO_BACKEND_LABEL}" ]]
}

is_superoffload_zero_run() {
  case "${ZERO_BACKEND_LABEL}" in
    superoffload|superoffload_mem|superoffload_mem_opnvme|superoffload_mem_panvme) return 0 ;;
    *) return 1 ;;
  esac
}

is_cpuadam_zero_run() {
  [[ "${ZERO_BACKEND_LABEL}" == "zero3_cpuadam" ]]
}

is_plain_torch_run() {
  [[ "${BACKEND}" == "torch" && -z "${ZERO_BACKEND_LABEL}" ]]
}

assert_deepspeed_scope() {
  local arg
  for arg in "${CMD_ARGS[@]}"; do
    if [[ "${arg}" == "--deepspeed" ]]; then
      if ! is_zero_backend_run; then
        echo "internal error: --deepspeed was added for BACKEND=${RUN_BACKEND_LABEL}; DeepSpeed is restricted to zero-policy runs" >&2
        exit 2
      fi
      return 0
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

# Resolve the single memory knob (true/false only; the old 'auto' is gone). No per-backend (CPUAdamW)
# override, so every backend gets identical memory instrumentation. Both ASYM_GEMM_LF_PROFILE_MEMORY_*
# toggles are derived from it below.
if [[ "${PROFILE_MEMORY_BREAKDOWN,,}" == "auto" ]]; then
  echo "PROFILE_MEMORY_BREAKDOWN must be true or false (the removed 'auto' is no longer supported)" >&2
  exit 2
fi
PROFILE_MEMORY_BREAKDOWN="$(profile_memory_flag PROFILE_MEMORY_BREAKDOWN "${PROFILE_MEMORY_BREAKDOWN}" "${PROFILE_PROFILER}")"
PROFILE_MEMORY_SNAPSHOT="$(profile_memory_flag PROFILE_MEMORY_SNAPSHOT "${PROFILE_MEMORY_SNAPSHOT}" "${PROFILE_PROFILER}")"
PROFILE_EXTERNAL_MEMORY="$(profile_memory_flag PROFILE_EXTERNAL_MEMORY "${PROFILE_EXTERNAL_MEMORY}" "${PROFILE_PROFILER}")"

validate_kt_arm_source_ok_profile() {
  local profile_json="$1"
  [[ -f "${profile_json}" ]] || {
    echo "KT_ARM_SOURCE_OK_PROFILE_JSON does not exist: ${profile_json}" >&2
    return 1
  }
  case "$(basename "${profile_json}")" in
    partial_profile.json|source_profile.partial.json)
      echo "KT_ARM_SOURCE_OK_PROFILE_JSON must point at a completed source artifact, not ${profile_json}" >&2
      return 1
      ;;
  esac
  python3 - "${profile_json}" "${MODEL_NAME_OR_PATH}" "${CUTOFF_LEN}" "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
	    "${LORA_RANK}" "${LORA_DROPOUT}" "${KT_ARM_SFT_TOP_K}" "${KT_ARM_SFT_TOKEN_CHUNK_SIZE}" \
	    "${KT_ARM_SFT_MAX_ROUTE_RANK_WORK}" "${KT_ARM_SFT_DEFAULT_MAX_ROUTE_RANK_WORK}" \
	    "${KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK}" "${GRADIENT_CHECKPOINTING}" "${KT_MAX_CACHE_DEPTH}" <<'PY'
import json
import math
import sys

(
    profile_path,
    expected_model,
    expected_seq,
    expected_batch,
    expected_rank,
    expected_dropout,
    expected_top_k,
    expected_token_chunk,
    expected_limit,
    expected_default_limit,
    allow_unvalidated,
    expected_recompute,
    expected_cache_depth,
) = sys.argv[1:14]
profile = json.load(open(profile_path, encoding="utf-8"))
source_profile = profile.get("source_profile", {})
source_profile = source_profile if isinstance(source_profile, dict) and source_profile else profile
if profile.get("partial") is True or source_profile.get("partial") is True:
    raise SystemExit("source-ok profile is partial")
heartbeat = source_profile.get("heartbeat", {})
latest = heartbeat.get("latest", {}) if isinstance(heartbeat, dict) else {}
stage = latest.get("stage") if isinstance(latest, dict) else None
if stage != "source_profile_written":
    raise SystemExit(f"source-ok profile heartbeat is not final: {stage or 'missing'}")
config = source_profile.get("config", {})
if not isinstance(config, dict):
    raise SystemExit("source-ok profile missing config")
if str(config.get("backend") or "") != "kt_armbf16":
    raise SystemExit("source-ok profile backend is not kt_armbf16")
kt_backend = str(config.get("kt_backend") or "").upper()
if kt_backend != "ARMBF16":
    raise SystemExit(f"source-ok profile kt_backend is not ARMBF16: {kt_backend or '<missing>'}")
if str(config.get("model_name_or_path") or "") != expected_model:
    raise SystemExit("source-ok profile model does not match this run")
try:
    if int(config.get("seq_len")) != int(expected_seq):
        raise SystemExit("source-ok profile seq_len does not match this run")
except (TypeError, ValueError):
    raise SystemExit("source-ok profile seq_len missing or invalid")
def int_value(container, key):
    try:
        return int(container.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0
def optional_int_value(container, key):
    if not isinstance(container, dict):
        return None
    value = container.get(key)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
def _optional_int(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
def kt_lora_health_passed(health):
    if not isinstance(health, dict) or not health:
        return False, "missing KT fused LoRA update health"
    rows = health.get("rows", [])
    row_dicts = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    updated_from_rows = sum(
        int(
            bool(row.get("nonzero_grad_changed_after_step"))
            or (bool(row.get("grad_nonzero_before_step")) and bool(row.get("param_changed_after_step")))
        )
        for row in row_dicts
    )
    unchanged_from_rows = sum(
        int(bool(row.get("grad_nonzero_before_step")) and not bool(row.get("param_changed_after_step")))
        for row in row_dicts
    )
    updated_grad_tensors = int_value(health, "updated_grad_tensors") or updated_from_rows
    grad_nonzero_unchanged_tensors = int_value(health, "grad_nonzero_unchanged_tensors") or unchanged_from_rows
    compared_tensors = int_value(health, "compared_tensors") or len(row_dicts)
    grad_nonzero_tensors = int_value(health, "grad_nonzero_tensors")
    sampled_tensors = int_value(health, "sampled_tensors") or len(row_dicts)
    total_fused_tensors = int_value(health, "total_fused_tensors")
    after_sampled_tensors = _optional_int(health.get("after_sampled_tensors"))
    after_total_fused_tensors = _optional_int(health.get("after_total_fused_tensors"))
    missing_after_tensors = int_value(health, "missing_after_tensors")
    unexpected_after_tensors = int_value(health, "unexpected_after_tensors")
    exhaustive = bool(health.get("exhaustive", total_fused_tensors > 0 and sampled_tensors == total_fused_tensors))
    input_passed = health.get("passed") if "passed" in health else None
    if health.get("available") is not True:
        return False, "KT fused LoRA update health unavailable"
    missing_snapshot_fields = [
        key
        for key in (
            "after_sampled_tensors",
            "after_total_fused_tensors",
            "missing_after_tensors",
            "unexpected_after_tensors",
        )
        if key not in health or health.get(key) in (None, "")
    ]
    if missing_snapshot_fields:
        return False, "KT fused LoRA update health missing after-step snapshot fields: " + ", ".join(missing_snapshot_fields)
    if after_sampled_tensors is None or after_total_fused_tensors is None:
        return False, "KT fused LoRA update health has invalid after-step snapshot counts"
    if missing_after_tensors > 0 or unexpected_after_tensors > 0:
        return False, "sampled fused LoRA tensor set changed between before/after optimizer snapshots"
    if sampled_tensors != after_sampled_tensors or total_fused_tensors != after_total_fused_tensors:
        return False, "fused LoRA tensor counts changed between before/after optimizer snapshots"
    if exhaustive and compared_tensors != sampled_tensors:
        return False, f"exhaustive fused LoRA health compared {compared_tensors} of {sampled_tensors} tensors"
    if compared_tensors <= 0:
        return False, "no sampled fused LoRA tensors were comparable"
    if grad_nonzero_tensors <= 0:
        return False, "no sampled fused LoRA tensors had nonzero gradients before optimizer step"
    if grad_nonzero_unchanged_tensors > 0:
        return False, "one or more sampled nonzero-gradient fused LoRA tensors did not change after optimizer step"
    if updated_grad_tensors <= 0:
        return False, "no sampled nonzero-gradient fused LoRA tensors changed after optimizer step"
    if input_passed is False:
        return False, str(health.get("reason") or "input KT fused LoRA update health failed")
    return True, "all sampled nonzero-gradient fused LoRA tensors changed after optimizer step"
def optimizer_process_memory_passed(optimizer_memory):
    if not isinstance(optimizer_memory, dict) or not optimizer_memory:
        return False, "missing optimizer_memory"
    for key in ("process_memory_at_start", "process_memory_before_step", "process_memory_after_step"):
        snapshot = optimizer_memory.get(key)
        if not isinstance(snapshot, dict) or not snapshot:
            return False, f"missing {key}"
        if _optional_int(snapshot.get("rss_bytes")) is None:
            return False, f"{key}.rss_bytes missing or invalid"
    for key in ("process_rss_pre_step_overhead_delta_bytes", "process_rss_delta_bytes"):
        if key not in optimizer_memory or _optional_int(optimizer_memory.get(key)) is None:
            return False, f"{key} missing or invalid"
    return True, "optimizer process RSS snapshots are present"
def require_int_config(key, expected):
    try:
        actual = int(config.get(key))
    except (TypeError, ValueError):
        raise SystemExit(f"source-ok profile {key} missing or invalid")
    if actual != int(expected):
        raise SystemExit(f"source-ok profile {key} mismatch: expected {expected}, got {actual}")
require_int_config("per_device_train_batch_size", expected_batch)
require_int_config("lora_rank", expected_rank)
require_int_config("kt_max_cache_depth", expected_cache_depth)
actual_recompute = str(config.get("activation_recompute")).lower()
if actual_recompute not in {"true", "false"}:
    raise SystemExit("source-ok profile activation_recompute missing or invalid")
if actual_recompute != str(expected_recompute).lower():
    raise SystemExit(
        f"source-ok profile activation_recompute mismatch: expected {expected_recompute}, got {actual_recompute}"
    )
try:
    actual_dropout = float(config.get("lora_dropout"))
    wanted_dropout = float(expected_dropout)
except (TypeError, ValueError):
    raise SystemExit("source-ok profile lora_dropout missing or invalid")
if not math.isclose(actual_dropout, wanted_dropout, rel_tol=0.0, abs_tol=1e-9):
    raise SystemExit(f"source-ok profile lora_dropout mismatch: expected {wanted_dropout}, got {actual_dropout}")
logical_qlen = int(expected_seq) * int(expected_batch)
token_chunk = int(expected_token_chunk) if str(expected_token_chunk).strip() else None
effective_route_qlen = min(logical_qlen, token_chunk) if token_chunk is not None else logical_qlen
token_chunks = (logical_qlen + token_chunk - 1) // token_chunk if token_chunk is not None and token_chunk < logical_qlen else 1
route_rank_work = effective_route_qlen * int(expected_top_k) * int(expected_rank)
normalized_limit = None
if str(expected_limit).strip():
    normalized_limit = int(expected_limit)
elif str(allow_unvalidated).strip() != "1":
    normalized_limit = int(expected_default_limit)
require_int_config("kt_arm_sft_top_k", expected_top_k)
if token_chunk is None:
    if str(config.get("kt_arm_sft_token_chunk_size") or "").strip():
        raise SystemExit("source-ok profile token chunk size mismatch: expected unset")
else:
    require_int_config("kt_arm_sft_token_chunk_size", token_chunk)
require_int_config("kt_arm_effective_route_qlen", effective_route_qlen)
require_int_config("kt_arm_token_chunks", token_chunks)
require_int_config("kt_arm_route_rank_work", route_rank_work)
if normalized_limit is not None:
    require_int_config("kt_arm_sft_max_route_rank_work", normalized_limit)
lora_target = str(config.get("lora_target") or "").lower()
if lora_target not in {"all", "all-linear", "all_linear"}:
    raise SystemExit("source-ok profile lora_target is not all")
kt = source_profile.get("kt", {})
if not isinstance(kt, dict):
    raise SystemExit("source-ok profile missing kt counters")
methods = {str(row.get("method") or "") for row in kt.get("rows", []) if isinstance(row, dict)}
if "ARMBF16_SFT" not in methods:
    raise SystemExit("source-ok profile has no ARMBF16_SFT KT row method")
if int_value(kt, "wrapper_count") <= 0:
    raise SystemExit("source-ok profile has no KT wrappers")
if int_value(kt, "total_forward_calls") <= 0 or int_value(kt, "total_backward_calls") <= 0:
    raise SystemExit("source-ok profile has no completed KT forward/backward step")
preflight = source_profile.get("optimizer_memory_preflight", {})
if not isinstance(preflight, dict) or preflight.get("available") is not True:
    raise SystemExit("source-ok profile missing optimizer_memory_preflight")
optimizer_memory = source_profile.get("optimizer_memory", {})
memory_ok, memory_reason = optimizer_process_memory_passed(optimizer_memory)
if not memory_ok:
    raise SystemExit(f"source-ok profile missing optimizer process memory evidence: {memory_reason}")
surface = source_profile.get("trainable_surface", {})
if not isinstance(surface, dict) or not surface.get("surface"):
    lora_for_surface = source_profile.get("lora", {})
    if not isinstance(lora_for_surface, dict) or lora_for_surface.get("available") is False:
        raise SystemExit("source-ok profile missing trainable_surface")
    surface_counter_keys = (
        "peft_lora_parameters",
        "peft_expert_lora_parameters",
        "kt_peft_expert_lora_parameters",
        "kt_expert_lora_parameters",
        "qwen_moe_expert_lora_parameters",
        "kt_fused_expert_lora_parameters",
    )
    if not any(optional_int_value(lora_for_surface, key) is not None for key in surface_counter_keys):
        raise SystemExit("source-ok profile missing trainable_surface")
    peft_lora = optional_int_value(lora_for_surface, "peft_lora_parameters")
    peft_expert = optional_int_value(lora_for_surface, "peft_expert_lora_parameters")
    if peft_expert is None:
        peft_expert = optional_int_value(lora_for_surface, "kt_peft_expert_lora_parameters") or 0
    kt_expert = optional_int_value(lora_for_surface, "kt_expert_lora_parameters") or 0
    qwen_moe_expert = optional_int_value(lora_for_surface, "qwen_moe_expert_lora_parameters") or 0
    expert_lora = max(kt_expert, peft_expert or 0, qwen_moe_expert)
    non_expert_peft = 0 if peft_lora is None else max(0, peft_lora - (peft_expert or 0))
    if expert_lora > 0 and non_expert_peft > 0:
        derived_surface = "attention+expert LoRA"
    elif expert_lora > 0:
        derived_surface = "expert LoRA"
    elif non_expert_peft > 0:
        derived_surface = "attention-only LoRA"
    else:
        derived_surface = "no trainable LoRA detected"
    surface = {"surface": derived_surface}
lora = source_profile.get("lora", {})
fused_lora_params = int_value(lora, "kt_fused_expert_lora_parameters") if isinstance(lora, dict) else 0
expert_lora_params = 0
if isinstance(lora, dict):
    expert_lora_params = max(
        int_value(lora, "kt_expert_lora_parameters") or 0,
        int_value(lora, "peft_expert_lora_parameters") or 0,
        int_value(lora, "qwen_moe_expert_lora_parameters") or 0,
        fused_lora_params or 0,
    )
if "qwen3" in expected_model.lower() and expert_lora_params <= 0:
    raise SystemExit("source-ok Qwen3 profile has no captured expert LoRA params")
if fused_lora_params > 0:
    health = optimizer_memory.get("kt_lora_update_health", {}) if isinstance(optimizer_memory, dict) else {}
    health_ok, health_reason = kt_lora_health_passed(health)
    if not health_ok:
        raise SystemExit(f"source-ok profile KT fused LoRA update health failed: {health_reason}")
PY
}

case "${ASYM_ROUTER_MODE,,}" in
  hf) ASYM_ROUTER_MODE=hf ;;
  whole) ASYM_ROUTER_MODE=whole ;;
  *) echo "ASYM_ROUTER_MODE must be hf or whole, got '${ASYM_ROUTER_MODE}'" >&2; exit 2 ;;
esac

if [[ ! "${PROFILE_MEMORY_BREAKDOWN_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PROFILE_MEMORY_BREAKDOWN_INTERVAL must be a positive integer, got '${PROFILE_MEMORY_BREAKDOWN_INTERVAL}'" >&2
  exit 2
fi

case "${PROFILE_SYNC}" in
  0|1|true|false|yes|no|on|off) ;;
  *) echo "PROFILE_SYNC must be true or false" >&2; exit 2 ;;
esac
CHECK_SUPEROFFLOAD="$(bool_01 CHECK_SUPEROFFLOAD "${CHECK_SUPEROFFLOAD}")"
CHECK_CPUADAM="$(bool_01 CHECK_CPUADAM "${CHECK_CPUADAM}")"
USE_ASYM_CPU_ADAMW="$(bool_string USE_ASYM_CPU_ADAMW "${USE_ASYM_CPU_ADAMW}")"
ASYM_CPU_ADAMW_PIN_MEMORY="$(bool_string ASYM_CPU_ADAMW_PIN_MEMORY "${ASYM_CPU_ADAMW_PIN_MEMORY}")"
ASYM_CPU_ADAMW_FP32_MASTER="$(bool_string ASYM_CPU_ADAMW_FP32_MASTER "${ASYM_CPU_ADAMW_FP32_MASTER}")"
ASYM_CPU_ADAMW_GRAD_OFFLOAD="$(bool_string ASYM_CPU_ADAMW_GRAD_OFFLOAD "${ASYM_CPU_ADAMW_GRAD_OFFLOAD}")"
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD="$(bool_string ASYM_CPU_ADAMW_WEIGHT_OFFLOAD "${ASYM_CPU_ADAMW_WEIGHT_OFFLOAD}")"
case "${ASYM_CPU_ADAMW_BACKEND,,}" in
  torch) ASYM_CPU_ADAMW_BACKEND=torch ;;
  deepspeed|ds) ASYM_CPU_ADAMW_BACKEND=deepspeed ;;
  *) echo "ASYM_CPU_ADAMW_BACKEND must be torch or deepspeed, got '${ASYM_CPU_ADAMW_BACKEND}'" >&2; exit 2 ;;
esac
if [[ "${USE_ASYM_CPU_ADAMW}" == "true" && "${CPUADAM_ALIAS_SELECTED}" != "1" ]]; then
  echo "Use BACKEND=asym_cpuadamwtorch or BACKEND=asym_cpuadamwds for AsymGEMM CPU AdamW; direct USE_ASYM_CPU_ADAMW=true is not allowed for BACKEND=${RUN_BACKEND_LABEL}." >&2
  exit 2
fi
if [[ "${BACKEND}" == kt_* ]]; then
  if [[ "${BACKEND}" == "kt_armbf16" ]]; then
    [[ -n "${KT_REQUIRE_STARTUP}" ]] || KT_REQUIRE_STARTUP=1
    if [[ -z "${KT_REQUIRE_FUSED_LORA_STARTUP}" ]]; then
      if [[ "${MODEL_NAME_OR_PATH,,}" == *"qwen3"* ]]; then
        KT_REQUIRE_FUSED_LORA_STARTUP=1
      else
        KT_REQUIRE_FUSED_LORA_STARTUP=0
      fi
    fi
  else
    [[ -n "${KT_REQUIRE_STARTUP}" ]] || KT_REQUIRE_STARTUP=0
    [[ -n "${KT_REQUIRE_FUSED_LORA_STARTUP}" ]] || KT_REQUIRE_FUSED_LORA_STARTUP=0
  fi
  CHECK_KT_CALLS="$(bool_01 CHECK_KT_CALLS "${CHECK_KT_CALLS}")"
  KT_REQUIRE_STARTUP="$(bool_01 KT_REQUIRE_STARTUP "${KT_REQUIRE_STARTUP}")"
  KT_REQUIRE_FUSED_LORA_STARTUP="$(bool_01 KT_REQUIRE_FUSED_LORA_STARTUP "${KT_REQUIRE_FUSED_LORA_STARTUP}")"
  KT_TP_ENABLED="$(bool_string KT_TP_ENABLED "${KT_TP_ENABLED}")"
  KT_SHARE_BACKWARD_BB="$(optional_bool_string KT_SHARE_BACKWARD_BB "${KT_SHARE_BACKWARD_BB}")"
  KT_USE_LORA_EXPERTS="$(optional_bool_string KT_USE_LORA_EXPERTS "${KT_USE_LORA_EXPERTS}")"
  [[ -z "${KT_NUM_THREADS}" ]] || positive_int_value KT_NUM_THREADS "${KT_NUM_THREADS}"
  [[ -z "${KT_THREADPOOL_COUNT}" ]] || positive_int_value KT_THREADPOOL_COUNT "${KT_THREADPOOL_COUNT}"
  positive_int_value KT_MAX_CACHE_DEPTH "${KT_MAX_CACHE_DEPTH}"
  positive_int_value KT_ARM_OMP_NUM_THREADS "${KT_ARM_OMP_NUM_THREADS}"
  positive_int_value KT_ARM_SFT_TOP_K "${KT_ARM_SFT_TOP_K}"
  [[ -z "${KT_ARM_SFT_TOKEN_CHUNK_SIZE}" ]] || positive_int_value KT_ARM_SFT_TOKEN_CHUNK_SIZE "${KT_ARM_SFT_TOKEN_CHUNK_SIZE}"
  positive_int_value KT_ARM_SFT_DEFAULT_MAX_ROUTE_RANK_WORK "${KT_ARM_SFT_DEFAULT_MAX_ROUTE_RANK_WORK}"
  [[ -z "${KT_ARM_SFT_MAX_ROUTE_RANK_WORK}" ]] || positive_int_value KT_ARM_SFT_MAX_ROUTE_RANK_WORK "${KT_ARM_SFT_MAX_ROUTE_RANK_WORK}"
  [[ -z "${KT_ARM_SFT_BACKWARD_THREADS}" ]] || positive_int_value KT_ARM_SFT_BACKWARD_THREADS "${KT_ARM_SFT_BACKWARD_THREADS}"
  positive_int_value KT_ARM_SFT_DEFAULT_MAX_BACKWARD_SCRATCH_BYTES "${KT_ARM_SFT_DEFAULT_MAX_BACKWARD_SCRATCH_BYTES}"
  [[ -z "${KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES}" ]] || positive_int_value KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES "${KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES}"
  nonnegative_int_value KT_ARM_FIRST_STEP_TIMEOUT_SECONDS "${KT_ARM_FIRST_STEP_TIMEOUT_SECONDS}"
  KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK="$(bool_01 KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK "${KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK}")"
  KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH="$(bool_01 KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH "${KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH}")"
  KT_ARM_ALLOW_NSYS_WITHOUT_SOURCE_OK="$(bool_01 KT_ARM_ALLOW_NSYS_WITHOUT_SOURCE_OK "${KT_ARM_ALLOW_NSYS_WITHOUT_SOURCE_OK}")"
  KT_ARM_ALLOW_RAW_NSYS_WITHOUT_SOURCE_OK="$(bool_01 KT_ARM_ALLOW_RAW_NSYS_WITHOUT_SOURCE_OK "${KT_ARM_ALLOW_RAW_NSYS_WITHOUT_SOURCE_OK}")"
  [[ -z "${KT_NUM_GPU_EXPERTS}" ]] || nonnegative_int_value KT_NUM_GPU_EXPERTS "${KT_NUM_GPU_EXPERTS}"
  [[ -z "${KT_LORA_EXPERT_NUM}" ]] || positive_int_value KT_LORA_EXPERT_NUM "${KT_LORA_EXPERT_NUM}"
  [[ -z "${KT_LORA_EXPERT_INTERMEDIATE_SIZE}" ]] || positive_int_value KT_LORA_EXPERT_INTERMEDIATE_SIZE "${KT_LORA_EXPERT_INTERMEDIATE_SIZE}"
  if [[ "${BACKEND}" == "kt_armbf16" ]]; then
    if [[ "${KT_TP_ENABLED}" != "false" ]]; then
      echo "BACKEND=kt_armbf16 does not support KT_TP_ENABLED=true" >&2
      exit 2
    fi
    if [[ -n "${KT_THREADPOOL_COUNT}" && "${KT_THREADPOOL_COUNT}" != "1" ]]; then
      echo "BACKEND=kt_armbf16 requires KT_THREADPOOL_COUNT empty or 1, got ${KT_THREADPOOL_COUNT}" >&2
      exit 2
    fi
    if [[ -n "${KT_NUM_GPU_EXPERTS}" && "${KT_NUM_GPU_EXPERTS}" != "0" ]]; then
      echo "BACKEND=kt_armbf16 does not support GPU experts; set KT_NUM_GPU_EXPERTS=0 or leave it empty" >&2
      exit 2
    fi
    if [[ "${GRADIENT_ACCUMULATION_STEPS}" != "1" ]]; then
      echo "BACKEND=kt_armbf16 requires GRADIENT_ACCUMULATION_STEPS=1 until native expert-LoRA gradient accumulation is fixed" >&2
      exit 2
    fi
    if [[ "${PROFILE}" == "1" && "${PROFILE_PROFILER}" == "nsys" ]]; then
      if [[ -n "${KT_ARM_SOURCE_OK_PROFILE_JSON}" ]]; then
        if ! validate_kt_arm_source_ok_profile "${KT_ARM_SOURCE_OK_PROFILE_JSON}"; then
          echo "BACKEND=kt_armbf16 PROFILE_PROFILER=nsys requires a matching completed source profile" >&2
          exit 2
        fi
      elif [[ "${KT_ARM_ALLOW_NSYS_WITHOUT_SOURCE_OK}" != "1" || "${KT_ARM_ALLOW_RAW_NSYS_WITHOUT_SOURCE_OK}" != "1" ]]; then
        echo "BACKEND=kt_armbf16 requires PROFILE_PROFILER=source before nsys; set KT_ARM_SOURCE_OK_PROFILE_JSON to a completed same-shape source profile, or set both KT_ARM_ALLOW_NSYS_WITHOUT_SOURCE_OK=1 and KT_ARM_ALLOW_RAW_NSYS_WITHOUT_SOURCE_OK=1 only for raw validation." >&2
        exit 2
      fi
    fi
  fi
else
  CHECK_KT_CALLS=0
fi

nonnegative_int_value PREPROCESSING_NUM_WORKERS "${PREPROCESSING_NUM_WORKERS}"
nonnegative_int_value DATALOADER_NUM_WORKERS "${DATALOADER_NUM_WORKERS}"
if [[ -n "${MAX_GRAD_NORM}" ]]; then
  python3 - "${MAX_GRAD_NORM}" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError as exc:
    raise SystemExit(f"MAX_GRAD_NORM must be a non-negative float, got {sys.argv[1]!r}") from exc
if not math.isfinite(value) or value < 0:
    raise SystemExit(f"MAX_GRAD_NORM must be a non-negative finite float, got {sys.argv[1]!r}")
PY
fi
PROFILE_LOGICAL_QLEN=$((PER_DEVICE_TRAIN_BATCH_SIZE * CUTOFF_LEN))
PROFILE_GLOBAL_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))
if [[ "${ASYM_STP:-0}" == "1" ]]; then
  # sTP is model-parallel: both GPUs process the SAME batch, so the global batch does NOT
  # multiply by NUM_GPUS (gb200_tp.md I0 step 3).
  PROFILE_GLOBAL_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
fi
KT_ARM_LOGICAL_QLEN=""
KT_ARM_EFFECTIVE_ROUTE_QLEN=""
KT_ARM_TOKEN_CHUNKS=""
KT_ARM_ROUTE_RANK_WORK=""
if [[ "${BACKEND}" == "kt_armbf16" ]]; then
  if [[ -z "${KT_ARM_SFT_MAX_ROUTE_RANK_WORK}" && "${KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK}" != "1" ]]; then
    KT_ARM_SFT_MAX_ROUTE_RANK_WORK="${KT_ARM_SFT_DEFAULT_MAX_ROUTE_RANK_WORK}"
  fi
  if [[ -z "${KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES}" && "${KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH}" != "1" ]]; then
    KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES="${KT_ARM_SFT_DEFAULT_MAX_BACKWARD_SCRATCH_BYTES}"
  fi
  KT_ARM_LOGICAL_QLEN="${PROFILE_LOGICAL_QLEN}"
  KT_ARM_EFFECTIVE_ROUTE_QLEN="${KT_ARM_LOGICAL_QLEN}"
  KT_ARM_TOKEN_CHUNKS=1
  if [[ -n "${KT_ARM_SFT_TOKEN_CHUNK_SIZE}" ]]; then
    positive_int_value KT_ARM_SFT_TOKEN_CHUNK_SIZE "${KT_ARM_SFT_TOKEN_CHUNK_SIZE}"
    if [[ "${KT_ARM_SFT_TOKEN_CHUNK_SIZE}" -lt "${KT_ARM_LOGICAL_QLEN}" ]]; then
      KT_ARM_EFFECTIVE_ROUTE_QLEN="${KT_ARM_SFT_TOKEN_CHUNK_SIZE}"
      KT_ARM_TOKEN_CHUNKS=$(((KT_ARM_LOGICAL_QLEN + KT_ARM_SFT_TOKEN_CHUNK_SIZE - 1) / KT_ARM_SFT_TOKEN_CHUNK_SIZE))
    fi
  fi
  KT_ARM_ROUTE_RANK_WORK=$((KT_ARM_EFFECTIVE_ROUTE_QLEN * KT_ARM_SFT_TOP_K * LORA_RANK))
  if [[ -n "${KT_ARM_SFT_MAX_ROUTE_RANK_WORK}" && "${KT_ARM_ROUTE_RANK_WORK}" -gt "${KT_ARM_SFT_MAX_ROUTE_RANK_WORK}" ]]; then
    echo "BACKEND=kt_armbf16 route-rank work ${KT_ARM_ROUTE_RANK_WORK} exceeds KT_ARM_SFT_MAX_ROUTE_RANK_WORK=${KT_ARM_SFT_MAX_ROUTE_RANK_WORK}" >&2
    echo "Reduce PER_DEVICE_TRAIN_BATCH_SIZE, CUTOFF_LEN, LORA_RANK, set KT_ARM_SFT_TOKEN_CHUNK_SIZE, raise the explicit limit, or set KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK=1 only for validation." >&2
    exit 2
  fi
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
  if [[ "${FLASH_ATTN}" == "fa4" ]]; then
    echo "Missing env ${ENV_DIR}. Run ${ASYM_DIR}/scripts/lf/bootstrap_lf_venv_fa4.sh first." >&2
  else
    echo "Missing env ${ENV_DIR}. Run ${ASYM_DIR}/scripts/lf/bootstrap_lf_venv.sh first." >&2
  fi
  exit 2
fi

if [[ "${PROFILE}" == "1" && ! -f "${PROFILE_LAUNCHER}" ]]; then
  echo "Missing profile launcher ${PROFILE_LAUNCHER}" >&2
  exit 2
fi

if [[ ! -x "${ENV_PYTHON}" ]]; then
  echo "Missing environment Python ${ENV_PYTHON}" >&2
  exit 2
fi

case "${FLASH_ATTN}" in
  auto|disabled|sdpa|fa2|fa3|fa4) ;;
  *) echo "FLASH_ATTN must be one of auto, disabled, sdpa, fa2, fa3, fa4; got '${FLASH_ATTN}'" >&2; exit 2 ;;
esac
if [[ "${FLASH_ATTN}" == "fa4" ]]; then
  if [[ ! -d "${LF_DIR}" ]]; then
    echo "FLASH_ATTN=fa4 requires a LlamaFactory checkout at LF_DIR=${LF_DIR}" >&2
    exit 2
  fi
  if ! PYTHONPATH="${LF_DIR}/src:${PYTHONPATH:-}" "${ENV_PYTHON}" - <<'PY' >/dev/null; then
import importlib.metadata as md
from transformers.utils import is_flash_attn_4_available
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
assert is_flash_attn_4_available()
assert md.version("flash-attn-4")
assert flash_attn_func is not None and flash_attn_varlen_func is not None
PY
    echo "FLASH_ATTN=fa4 validation failed for ENV_DIR=${ENV_DIR} LF_DIR=${LF_DIR}." >&2
    echo "Run ${ASYM_DIR}/scripts/lf/bootstrap_lf_venv_fa4.sh or set ASYM_QWEN35_FA4_AUTO=0 to opt out." >&2
    exit 2
  fi
fi
if [[ "${PROFILE}" != "1" ]] && ! is_torch_run && [[ ! -x "${LF_CLI_BIN}" ]]; then
  echo "Missing LlamaFactory CLI ${LF_CLI_BIN}" >&2
  exit 2
fi

if is_torch_run && [[ "${DIST_LAUNCHER}" == "torchrun" ]]; then
  if [[ -x "${TORCHRUN_BIN}" ]]; then
    TORCHRUN_CMD=("${TORCHRUN_BIN}")
  elif [[ -x "${ENV_PYTHON}" ]]; then
    TORCHRUN_CMD=("${ENV_PYTHON}" -m torch.distributed.run)
  else
    echo "Missing torchrun executable ${TORCHRUN_BIN} and environment Python ${ENV_PYTHON}" >&2
    exit 2
  fi
fi
if is_torch_run && [[ "${DIST_LAUNCHER}" == "accelerate" ]]; then
  if [[ -x "${ACCELERATE_BIN}" ]]; then
    ACCELERATE_CMD=("${ACCELERATE_BIN}")
  elif command -v accelerate >/dev/null 2>&1; then
    ACCELERATE_CMD=(accelerate)
  else
    echo "Missing accelerate executable ${ACCELERATE_BIN}" >&2
    exit 2
  fi
fi
if is_torch_run && [[ "${DIST_LAUNCHER}" == "deepspeed" ]]; then
  if [[ -x "${DEEPSPEED_BIN}" ]]; then
    DEEPSPEED_CMD=("${DEEPSPEED_BIN}")
  elif command -v deepspeed >/dev/null 2>&1; then
    DEEPSPEED_CMD=(deepspeed)
  else
    echo "Missing deepspeed executable ${DEEPSPEED_BIN}" >&2
    exit 2
  fi
fi

if is_zero_backend_run; then
  if [[ ! -f "${TORCH_DEEPSPEED_CONFIG}" ]]; then
    echo "Missing DeepSpeed config ${TORCH_DEEPSPEED_CONFIG}" >&2
    exit 2
  fi
  if is_superoffload_zero_run; then
    if [[ ! -f "${DEEPSPEED_DIR}/deepspeed/runtime/superoffload/superoffload_stage3.py" ]]; then
      echo "Missing local SuperOffload DeepSpeed tree at ${DEEPSPEED_DIR}" >&2
      exit 2
    fi
    if [[ ! -f "${CHECK_SUPEROFFLOAD_SCRIPT}" ]]; then
      echo "Missing SuperOffload checker ${CHECK_SUPEROFFLOAD_SCRIPT}" >&2
      exit 2
    fi
  fi
  if is_cpuadam_zero_run; then
    if [[ ! -f "${CHECK_CPUADAM_SCRIPT}" ]]; then
      echo "Missing DeepSpeed CPUAdam checker ${CHECK_CPUADAM_SCRIPT}" >&2
      exit 2
    fi
  fi
fi

MASTER_PORT_VALUE="${MASTER_PORT:-}"
if is_torch_run && [[ -z "${MASTER_PORT_VALUE}" ]]; then
  MASTER_PORT_VALUE="$(find_free_port)"
fi
MASTER_PORT_VALUE="${MASTER_PORT_VALUE:-29500}"

if [[ "${PROFILE}" == "1" && -z "${PROFILE_SOURCE_JSON}" ]]; then
  PROFILE_SOURCE_JSON="${OUT_DIR}/source_profile.json"
fi
if [[ "${PROFILE}" == "1" && -z "${PROFILE_HEARTBEAT_JSON}" ]]; then
  PROFILE_HEARTBEAT_JSON="${OUT_DIR}/heartbeat.jsonl"
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
if [[ "${PROFILE}" == "1" ]]; then
  mkdir -p "$(dirname "${PROFILE_SOURCE_JSON}")" "$(dirname "${PROFILE_JSON}")" "${PROFILE_OUTPUT_DIR}"
  rm -f \
    "${PROFILE_SOURCE_JSON}" \
    "$(dirname "${PROFILE_SOURCE_JSON}")/source_profile.partial.json" \
    "${PROFILE_JSON}" \
    "$(dirname "${PROFILE_JSON}")/partial_profile.json" \
    "${PROFILE_OUTPUT_DIR}/profile.json" \
    "${PROFILE_OUTPUT_DIR}/partial_profile.json" \
    "${PROFILE_HEARTBEAT_JSON}" \
    "${PROFILE_HEARTBEAT_JSON%.*}.latest.json" \
    2>/dev/null || true
fi
if [[ -n "${TRITON_CACHE_DIR:-}" ]]; then
  mkdir -p "${TRITON_CACHE_DIR}"
fi
: > "${LOG_FILE}"

managed_interrupted=false
managed_interrupt_exit_status=130
managed_child_pid=""
managed_child_pid_file=""
managed_wait_pid=""
host_oom_cgroup_kind=""
host_oom_cgroup_path=""
host_oom_failcnt_before=""
host_oom_oom_before=""
host_oom_oom_kill_before=""
host_oom_under_oom_before=""

cgroup_memory_dir() {
  local rel path
  if [[ -r "/sys/fs/cgroup/memory.events" ]]; then
    printf '%s\n' "/sys/fs/cgroup"
    return 0
  fi
  rel="$(awk -F: '$2 ~ /(^|,)memory(,|$)/ {print $3; exit}' /proc/self/cgroup 2>/dev/null || true)"
  if [[ -n "${rel}" ]]; then
    if [[ "${rel}" == "/" ]]; then
      path="/sys/fs/cgroup/memory"
    else
      path="/sys/fs/cgroup/memory/${rel#/}"
    fi
    if [[ -r "${path}/memory.failcnt" || -r "${path}/memory.oom_control" ]]; then
      printf '%s\n' "${path}"
      return 0
    fi
  fi
  if [[ -r "/sys/fs/cgroup/memory/memory.failcnt" || -r "/sys/fs/cgroup/memory/memory.oom_control" ]]; then
    printf '%s\n' "/sys/fs/cgroup/memory"
    return 0
  fi
  return 1
}

cgroup_kv() {
  local file="$1" key="$2"
  awk -v key="${key}" '$1 == key {print $2; found=1; exit} END {if (!found) exit 1}' "${file}" 2>/dev/null || true
}

cgroup_scalar() {
  local file="$1"
  if [[ -r "${file}" ]]; then
    tr -d '[:space:]' < "${file}" 2>/dev/null || true
  fi
}

detect_host_oom_monitor() {
  host_oom_cgroup_path="$(cgroup_memory_dir || true)"
  [[ -n "${host_oom_cgroup_path}" ]] || return 0
  if [[ -r "${host_oom_cgroup_path}/memory.events" ]]; then
    host_oom_cgroup_kind="v2"
    host_oom_oom_before="$(cgroup_kv "${host_oom_cgroup_path}/memory.events" oom)"
    host_oom_oom_kill_before="$(cgroup_kv "${host_oom_cgroup_path}/memory.events" oom_kill)"
  else
    host_oom_cgroup_kind="v1"
    host_oom_failcnt_before="$(cgroup_scalar "${host_oom_cgroup_path}/memory.failcnt")"
    host_oom_oom_kill_before="$(cgroup_kv "${host_oom_cgroup_path}/memory.oom_control" oom_kill)"
    host_oom_under_oom_before="$(cgroup_kv "${host_oom_cgroup_path}/memory.oom_control" under_oom)"
  fi
}

int_delta() {
  local before="$1" after="$2"
  if [[ "${before}" =~ ^[0-9]+$ && "${after}" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$((after - before))"
  else
    printf 'NA\n'
  fi
}

positive_delta() {
  [[ "$1" =~ ^-?[0-9]+$ ]] && (( $1 > 0 ))
}

signal_label_from_status() {
  local status="$1" sig
  if [[ "${status}" =~ ^[0-9]+$ && "${status}" -ge 128 ]]; then
    sig=$((status - 128))
    printf 'SIG%s\n' "$(kill -l "${sig}" 2>/dev/null || printf '%s' "${sig}")"
  else
    printf 'none\n'
  fi
}

append_host_oom_diagnostics() {
  local status="$1"
  local signal_label evidence=false reason="no cgroup memory kill counter increase observed"
  local oom_after="" oom_kill_after="" failcnt_after="" under_oom_after=""
  local oom_delta="NA" oom_kill_delta="NA" failcnt_delta="NA"
  local watchdog_fired=false
  if [[ -n "${LOG_FILE:-}" && -f "${LOG_FILE}.host_mem_watchdog_fired" ]]; then
    watchdog_fired=true
  fi
  signal_label="$(signal_label_from_status "${status}")"
  {
    printf '===== host memory OOM diagnostics =====\n'
    printf 'exit_status=%s\n' "${status}"
    printf 'exit_signal=%s\n' "${signal_label}"
    printf 'HOST_MEM_WATCHDOG_FIRED=%s\n' "${watchdog_fired}"
    if [[ -z "${host_oom_cgroup_path}" ]]; then
      printf 'cgroup_memory_monitor=unavailable\n'
      if [[ "${watchdog_fired}" == "true" ]]; then
        printf 'HOST_OOM_EVIDENCE=true\n'
        printf 'HOST_OOM_EVIDENCE_REASON=host memory watchdog interrupted the run before kernel OOM (soft host OOM)\n'
      else
        printf 'HOST_OOM_EVIDENCE=false\n'
        printf 'HOST_OOM_EVIDENCE_REASON=no readable cgroup memory counters\n'
      fi
    else
      printf 'cgroup_memory_monitor=%s\n' "${host_oom_cgroup_kind}"
      printf 'cgroup_memory_path=%s\n' "${host_oom_cgroup_path}"
      if [[ "${host_oom_cgroup_kind}" == "v2" ]]; then
        oom_after="$(cgroup_kv "${host_oom_cgroup_path}/memory.events" oom)"
        oom_kill_after="$(cgroup_kv "${host_oom_cgroup_path}/memory.events" oom_kill)"
        oom_delta="$(int_delta "${host_oom_oom_before}" "${oom_after}")"
        oom_kill_delta="$(int_delta "${host_oom_oom_kill_before}" "${oom_kill_after}")"
        printf 'memory.events.oom.before=%s after=%s delta=%s\n' "${host_oom_oom_before:-NA}" "${oom_after:-NA}" "${oom_delta}"
        printf 'memory.events.oom_kill.before=%s after=%s delta=%s\n' "${host_oom_oom_kill_before:-NA}" "${oom_kill_after:-NA}" "${oom_kill_delta}"
        if positive_delta "${oom_kill_delta}"; then
          evidence=true
          reason="cgroup v2 memory.events oom_kill increased during run"
        fi
      else
        failcnt_after="$(cgroup_scalar "${host_oom_cgroup_path}/memory.failcnt")"
        oom_kill_after="$(cgroup_kv "${host_oom_cgroup_path}/memory.oom_control" oom_kill)"
        under_oom_after="$(cgroup_kv "${host_oom_cgroup_path}/memory.oom_control" under_oom)"
        failcnt_delta="$(int_delta "${host_oom_failcnt_before}" "${failcnt_after}")"
        oom_kill_delta="$(int_delta "${host_oom_oom_kill_before}" "${oom_kill_after}")"
        printf 'memory.failcnt.before=%s after=%s delta=%s\n' "${host_oom_failcnt_before:-NA}" "${failcnt_after:-NA}" "${failcnt_delta}"
        printf 'memory.oom_control.oom_kill.before=%s after=%s delta=%s\n' "${host_oom_oom_kill_before:-NA}" "${oom_kill_after:-NA}" "${oom_kill_delta}"
        printf 'memory.oom_control.under_oom.before=%s after=%s\n' "${host_oom_under_oom_before:-NA}" "${under_oom_after:-NA}"
        if positive_delta "${oom_kill_delta}"; then
          evidence=true
          reason="cgroup v1 memory.oom_control oom_kill increased during run"
        fi
      fi
      if [[ "${watchdog_fired}" == "true" ]]; then
        if [[ "${evidence}" == "true" ]]; then
          reason="host memory watchdog fired and ${reason}"
        else
          reason="host memory watchdog interrupted the run before kernel OOM (soft host OOM)"
        fi
        evidence=true
      fi
      printf 'HOST_OOM_EVIDENCE=%s\n' "${evidence}"
      printf 'HOST_OOM_EVIDENCE_REASON=%s\n' "${reason}"
    fi
    printf '===== end host memory OOM diagnostics =====\n'
  } | tee -a "${LOG_FILE}" >/dev/null
}

detect_host_oom_monitor

managed_process_alive() {
  local target_pid
  for target_pid in "$@"; do
    [[ "${target_pid}" =~ ^[0-9]+$ ]] || continue
    if kill -0 "-${target_pid}" 2>/dev/null || kill -0 "${target_pid}" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

heartbeat_latest_json_path() {
  [[ -n "${PROFILE_HEARTBEAT_JSON}" ]] || return 1
  printf '%s.latest.json\n' "${PROFILE_HEARTBEAT_JSON%.*}"
}

heartbeat_latest_stage() {
  local latest_path
  latest_path="$(heartbeat_latest_json_path)" || return 1
  [[ -f "${latest_path}" ]] || return 1
  python3 - "${latest_path}" <<'PY'
import json
import sys

try:
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(1)
print(payload.get("stage", ""))
PY
}

first_step_watchdog_enabled() {
  [[ "${BACKEND}" == "kt_armbf16" ]] || return 1
  [[ "${PROFILE}" == "1" ]] || return 1
  [[ "${KT_ARM_FIRST_STEP_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] || return 1
  [[ "${KT_ARM_FIRST_STEP_TIMEOUT_SECONDS}" != "0" ]] || return 1
}

is_first_step_started_stage() {
  case "$1" in
    trainer_start|dataloader_batch_fetch_start|dataloader_batch_fetch_end|training_step_start|model_forward_enter|model_forward_exit|training_step_end|grad_clip_start|grad_clip_end|optimizer_step_start|optimizer_step_end|kt_lora_pointer_refresh_start|kt_sft_*) return 0 ;;
    *) return 1 ;;
  esac
}

is_first_step_completed_stage() {
  case "$1" in
    kt_lora_pointer_refresh_end|trainer_end|source_profile_written) return 0 ;;
    *) return 1 ;;
  esac
}

signal_managed_targets() {
  local signal_name="$1"
  shift
  local target_pid
  for target_pid in "$@"; do
    [[ "${target_pid}" =~ ^[0-9]+$ ]] || continue
    kill "-${signal_name}" "-${target_pid}" 2>/dev/null || true
    kill "-${signal_name}" "${target_pid}" 2>/dev/null || true
  done
}

managed_child_targets() {
  local file_pid="" target_pid
  if [[ -n "${managed_child_pid_file:-}" && -s "${managed_child_pid_file}" ]]; then
    IFS= read -r file_pid < "${managed_child_pid_file}" || true
  fi
  for target_pid in "${file_pid}" "${managed_child_pid:-}" "${managed_wait_pid:-}"; do
    [[ "${target_pid}" =~ ^[0-9]+$ ]] && printf '%s\n' "${target_pid}"
  done | awk 'NF && !seen[$0]++'
}

kill_managed_child() {
  local -a target_pids=()
  mapfile -t target_pids < <(managed_child_targets)
  rm -f "${managed_child_pid_file:-}" 2>/dev/null || true

  ((${#target_pids[@]} > 0)) || return 0
  signal_managed_targets INT "${target_pids[@]}"

  if [[ "${INTERRUPT_GRACE_SECONDS}" != "0" ]]; then
    sleep "${INTERRUPT_GRACE_SECONDS}" || true
  fi
  if managed_process_alive "${target_pids[@]}"; then
    signal_managed_targets TERM "${target_pids[@]}"
  fi

  if [[ "${INTERRUPT_GRACE_SECONDS}" != "0" ]]; then
    sleep "${INTERRUPT_GRACE_SECONDS}" || true
  fi
  if managed_process_alive "${target_pids[@]}"; then
    signal_managed_targets KILL "${target_pids[@]}"
  fi
}

cleanup_managed_child() {
  if [[ -n "${managed_child_pid:-}" || -n "${managed_wait_pid:-}" || -n "${managed_child_pid_file:-}" ]]; then
    kill_managed_child
  fi
}

host_mem_watchdog_enabled() {
  case "${HOST_MEM_WATCHDOG,,}" in
    1|true|yes|on) ;;
    *) return 1 ;;
  esac
  [[ "${HOST_MEM_WATCHDOG_FLOOR_GB}" =~ ^[0-9]+$ ]] || return 1
  (( HOST_MEM_WATCHDOG_FLOOR_GB > 0 )) || return 1
}

# Available memory (kB) summed over CPU NUMA nodes only. Training is membind to the
# CPU nodes, so the GPU HBM NUMA nodes (memory-only on GB200) must not count: global
# MemAvailable includes free HBM the trainer cannot allocate, which would let the
# kernel OOM killer fire before the watchdog. Falls back to global MemAvailable.
host_mem_watchdog_avail_kb() {
  local node cpus free_kb file_kb shmem_kb total=0 seen=0
  for node in /sys/devices/system/node/node*; do
    [[ -r "${node}/cpulist" && -r "${node}/meminfo" ]] || continue
    cpus="$(tr -d '[:space:]' < "${node}/cpulist" 2>/dev/null || true)"
    [[ -n "${cpus}" ]] || continue
    free_kb="$(awk '$3 == "MemFree:" {s += $4} END {print s+0}' "${node}/meminfo" 2>/dev/null || echo 0)"
    file_kb="$(awk '$3 == "FilePages:" {s += $4} END {print s+0}' "${node}/meminfo" 2>/dev/null || echo 0)"
    shmem_kb="$(awk '$3 == "Shmem:" {s += $4} END {print s+0}' "${node}/meminfo" 2>/dev/null || echo 0)"
    if (( file_kb > shmem_kb )); then
      free_kb=$(( free_kb + file_kb - shmem_kb ))
    fi
    total=$(( total + free_kb ))
    seen=1
  done
  if (( seen == 0 )); then
    awk '/^MemAvailable:/ {print $2; exit}' /proc/meminfo 2>/dev/null || echo ""
    return 0
  fi
  printf '%s\n' "${total}"
}

# Poll available CPU-node memory while the managed child runs; on breach, freeze the
# process group (stops allocation instantly), queue SIGINT, resume so the normal
# graceful-interrupt path runs, and escalate through kill_managed_child only if the
# run does not exit within HOST_MEM_WATCHDOG_KILL_GRACE_SECONDS.
host_mem_watchdog_loop() {
  local floor_kb=$(( HOST_MEM_WATCHDOG_FLOOR_GB * 1024 * 1024 ))
  local avail_kb deadline
  local -a targets=()
  while :; do
    mapfile -t targets < <(managed_child_targets)
    ((${#targets[@]} > 0)) || return 0
    managed_process_alive "${targets[@]}" || return 0
    avail_kb="$(host_mem_watchdog_avail_kb 2>/dev/null || true)"
    if [[ "${avail_kb}" =~ ^[0-9]+$ ]] && (( avail_kb < floor_kb )); then
      printf '[host-mem-watchdog] CPU-node available memory %s GiB dropped below floor %s GiB; interrupting training before the kernel OOM killer fires (soft host OOM).\n' \
        "$(( avail_kb / 1024 / 1024 ))" "${HOST_MEM_WATCHDOG_FLOOR_GB}" >&2
      if [[ -n "${LOG_FILE:-}" ]]; then
        printf 'fired_at=%s avail_kb=%s floor_gb=%s\n' "$(date -Is 2>/dev/null || true)" "${avail_kb}" "${HOST_MEM_WATCHDOG_FLOOR_GB}" \
          > "${LOG_FILE}.host_mem_watchdog_fired" 2>/dev/null || true
      fi
      signal_managed_targets STOP "${targets[@]}"
      signal_managed_targets INT "${targets[@]}"
      signal_managed_targets CONT "${targets[@]}"
      deadline=$(( SECONDS + HOST_MEM_WATCHDOG_KILL_GRACE_SECONDS ))
      while (( SECONDS < deadline )); do
        managed_process_alive "${targets[@]}" || return 0
        sleep 1
      done
      if managed_process_alive "${targets[@]}"; then
        echo "[host-mem-watchdog] training did not exit within ${HOST_MEM_WATCHDOG_KILL_GRACE_SECONDS}s of soft-OOM interrupt; escalating." >&2
        kill_managed_child
      fi
      return 0
    fi
    sleep "${HOST_MEM_WATCHDOG_POLL_SECONDS}"
  done
}

run_managed_command() {
  local status=0 wait_pid="" child_pid="" pid_file="" attempt
  local watchdog_started_at=0 watchdog_status=0 watchdog_stage=""
  managed_child_pid=""
  managed_wait_pid=""
  managed_child_pid_file=""
  if command -v setsid >/dev/null 2>&1 && setsid --help 2>&1 | grep -q -- '--wait' && setsid --help 2>&1 | grep -q -- '--fork'; then
    pid_file="$(mktemp "${TMPDIR:-/tmp}/run_lf_lora_sft_child.XXXXXX")"
    managed_child_pid_file="${pid_file}"
    # Wrapper: record the session pid, mark the tree as the OOM killer's preferred
    # victim, and restore default SIGINT/SIGQUIT (ignored in &-launched trees, which
    # would silently disable KeyboardInterrupt in the trainer) before exec.
    setsid --fork --wait bash -c 'pid_file="$1"; oom_adj="$2"; shift 2; echo "$$" > "${pid_file}"; if [[ -n "${oom_adj}" ]]; then { echo "${oom_adj}" > /proc/self/oom_score_adj; } 2>/dev/null || true; fi; if env --default-signal=SIGINT,SIGQUIT true 2>/dev/null; then exec env --default-signal=SIGINT,SIGQUIT "$@"; fi; exec "$@"' _ "${pid_file}" "${TRAIN_OOM_SCORE_ADJ}" "$@" &
    wait_pid=$!
    managed_wait_pid="${wait_pid}"
    managed_child_pid="${wait_pid}"
    for ((attempt = 0; attempt < 100; attempt++)); do
      if [[ -s "${pid_file}" ]]; then
        IFS= read -r child_pid < "${pid_file}" || true
        if [[ -n "${child_pid}" ]]; then
          managed_child_pid="${child_pid}"
          break
        fi
      fi
      kill -0 "${wait_pid}" 2>/dev/null || break
      sleep 0.02
    done
  else
    bash -c 'oom_adj="$1"; shift; if [[ -n "${oom_adj}" ]]; then { echo "${oom_adj}" > /proc/self/oom_score_adj; } 2>/dev/null || true; fi; if env --default-signal=SIGINT,SIGQUIT true 2>/dev/null; then exec env --default-signal=SIGINT,SIGQUIT "$@"; fi; exec "$@"' _ "${TRAIN_OOM_SCORE_ADJ}" "$@" &
    wait_pid=$!
    managed_wait_pid="${wait_pid}"
    managed_child_pid="${wait_pid}"
  fi

  local host_mem_watchdog_pid=""
  if [[ -n "${LOG_FILE:-}" ]]; then
    rm -f "${LOG_FILE}.host_mem_watchdog_fired" 2>/dev/null || true
  fi
  if host_mem_watchdog_enabled; then
    host_mem_watchdog_loop &
    host_mem_watchdog_pid=$!
  fi

  if first_step_watchdog_enabled; then
    while kill -0 "${wait_pid}" 2>/dev/null; do
      local wait_state
      wait_state="$(ps -o stat= -p "${wait_pid}" 2>/dev/null || true)"
      [[ "${wait_state}" == *Z* ]] && break
      watchdog_stage="$(heartbeat_latest_stage 2>/dev/null || true)"
      if is_first_step_completed_stage "${watchdog_stage}"; then
        watchdog_started_at=0
      elif is_first_step_started_stage "${watchdog_stage}"; then
        if [[ "${watchdog_started_at}" == "0" ]]; then
          watchdog_started_at="$(date +%s)"
        elif (( $(date +%s) - watchdog_started_at >= KT_ARM_FIRST_STEP_TIMEOUT_SECONDS )); then
          echo "BACKEND=kt_armbf16 did not complete the first optimizer step within KT_ARM_FIRST_STEP_TIMEOUT_SECONDS=${KT_ARM_FIRST_STEP_TIMEOUT_SECONDS}; latest heartbeat stage=${watchdog_stage}" >&2
          kill_managed_child
          watchdog_status=124
          break
        fi
      fi
      sleep 1
    done
  fi
  wait "${wait_pid}" || status=$?
  if [[ -n "${host_mem_watchdog_pid}" ]]; then
    kill "${host_mem_watchdog_pid}" 2>/dev/null || true
    wait "${host_mem_watchdog_pid}" 2>/dev/null || true
  fi
  if [[ "${watchdog_status}" != "0" ]]; then
    status="${watchdog_status}"
  fi
  if [[ "${managed_interrupted}" == "true" ]]; then
    kill_managed_child
    wait "${wait_pid}" 2>/dev/null || true
    managed_child_pid=""
    managed_child_pid_file=""
    managed_wait_pid=""
    exit "${managed_interrupt_exit_status}"
  fi
  if [[ "${status}" == "130" || "${status}" == "143" ]]; then
    kill_managed_child
    wait "${wait_pid}" 2>/dev/null || true
  fi
  [[ -z "${pid_file}" ]] || rm -f "${pid_file}" 2>/dev/null || true
  managed_child_pid=""
  managed_child_pid_file=""
  managed_wait_pid=""
  return "${status}"
}

run_logged_command() {
  run_managed_command "$@" > >(tee -a "${LOG_FILE}") 2>&1
}

postprocess_source_profile_if_available() {
  local source_json="${PROFILE_SOURCE_JSON}"
  local profile_json="${PROFILE_JSON}"
  local partial_source=0
  local force_partial="${1:-0}"
  if [[ "${force_partial}" == "1" && -f "${source_json}" ]]; then
    profile_json="$(dirname "${PROFILE_JSON}")/partial_profile.json"
    partial_source=1
    echo "Training failed; postprocessing source profile as partial artifact ${profile_json}" | tee -a "${LOG_FILE}"
  elif [[ ! -f "${source_json}" ]]; then
    local partial_json
    partial_json="$(dirname "${PROFILE_SOURCE_JSON}")/source_profile.partial.json"
    if [[ -f "${partial_json}" ]]; then
      source_json="${partial_json}"
      profile_json="$(dirname "${PROFILE_JSON}")/partial_profile.json"
      partial_source=1
      echo "Final source profile is missing; postprocessing partial profile ${partial_json}" | tee -a "${LOG_FILE}"
    else
      echo "No source profile artifact found at ${PROFILE_SOURCE_JSON} or ${partial_json}" | tee -a "${LOG_FILE}"
      return 1
    fi
  fi
  if [[ ! -f "${PROFILE_POSTPROCESS_SCRIPT}" ]]; then
    echo "Missing profile postprocess script ${PROFILE_POSTPROCESS_SCRIPT}" >&2
    return 2
  fi
  local postprocess_status=0
  "${ENV_PYTHON}" "${PROFILE_POSTPROCESS_SCRIPT}" \
    --source-profile-json "${source_json}" \
    --profile-json "${profile_json}" \
    --output-dir "${PROFILE_OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}" || postprocess_status=$?
  if [[ "${postprocess_status}" != "0" ]]; then
    return "${postprocess_status}"
  fi
  if [[ "${partial_source}" != "1" ]]; then
    local output_profile_json="${PROFILE_OUTPUT_DIR}/profile.json"
    if [[ "${profile_json}" != "${output_profile_json}" ]]; then
      cp "${profile_json}" "${output_profile_json}"
      echo "Copied canonical source profile artifact to ${output_profile_json}" | tee -a "${LOG_FILE}"
    fi
  fi
  if [[ "${partial_source}" == "1" ]]; then
    local output_partial_json="${PROFILE_OUTPUT_DIR}/partial_profile.json"
    if [[ "${profile_json}" != "${output_partial_json}" ]]; then
      cp "${profile_json}" "${output_partial_json}"
      echo "Copied partial source profile artifact to ${output_partial_json}" | tee -a "${LOG_FILE}"
    fi
    echo "Partial source profile artifacts were written to ${profile_json}; canonical profile.json was not created." | tee -a "${LOG_FILE}"
  fi
}

source_profile_final_json_proven() {
  local source_json="$1"
  [[ -f "${source_json}" ]] || return 1
  local latest_path=""
  latest_path="$(heartbeat_latest_json_path 2>/dev/null || true)"
  python3 - "${source_json}" "${latest_path}" <<'PY'
import json
import os
import sys

source_path, latest_path = sys.argv[1:3]
try:
    profile = json.load(open(source_path, encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"failed to parse source profile {source_path}: {exc}")

source_profile = profile.get("source_profile", {})
source_profile = source_profile if isinstance(source_profile, dict) and source_profile else profile
if profile.get("partial") is True or source_profile.get("partial") is True:
    raise SystemExit("source profile is marked partial")

def embedded_stage(payload):
    heartbeat = payload.get("heartbeat", {})
    latest = heartbeat.get("latest", {}) if isinstance(heartbeat, dict) else {}
    if isinstance(latest, dict) and latest.get("stage"):
        return str(latest.get("stage"))
    latest = payload.get("heartbeat_latest", {})
    if isinstance(latest, dict) and latest.get("stage"):
        return str(latest.get("stage"))
    if isinstance(latest, str) and latest:
        return latest
    return ""

stage = embedded_stage(source_profile)
if stage != "source_profile_written":
    raise SystemExit(f"source profile heartbeat is not final: {stage or 'missing'}")

if latest_path and os.path.exists(latest_path):
    try:
        latest_payload = json.load(open(latest_path, encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"failed to parse latest heartbeat {latest_path}: {exc}")
    latest_stage = str(latest_payload.get("stage") or "")
    if latest_stage != "source_profile_written":
        raise SystemExit(f"latest heartbeat is not final: {latest_stage or 'missing'}")
PY
}

source_profile_completion_proven() {
  [[ "${PROFILE}" == "1" ]] || return 1
  [[ "${PROFILE_PROFILER}" == "source" ]] || return 1
  source_profile_final_json_proven "${PROFILE_SOURCE_JSON}" || return $?
  if [[ "${BACKEND}" == "kt_armbf16" ]]; then
    validate_kt_arm_source_ok_profile "${PROFILE_SOURCE_JSON}" || return $?
  fi
}

check_trainable_surface_if_requested() {
  [[ "${CHECK_TRAINABLE_SURFACE}" == "1" ]] || return 0
  [[ "${PROFILE}" == "1" ]] || return 0
  [[ -f "${PROFILE_JSON}" ]] || return 0
  "${ENV_PYTHON}" - "${PROFILE_JSON}" <<'PY'
import json
import sys

profile_path = sys.argv[1]
profile = json.load(open(profile_path, encoding="utf-8"))
source_profile = profile.get("source_profile", {})
if isinstance(source_profile, dict) and source_profile:
    profile = source_profile
config = profile.get("config", {}) if isinstance(profile.get("config"), dict) else {}
model_name = str(config.get("model_name_or_path") or "")
lora_target = str(config.get("lora_target") or "").lower()
if "qwen3" not in model_name.lower() or lora_target not in {"all", "all-linear", "all_linear"}:
    raise SystemExit(0)

surface = profile.get("trainable_surface", {})
lora = profile.get("lora", {}) if isinstance(profile.get("lora"), dict) else {}
expert = surface.get("expert_lora_parameters")
if expert is None:
    expert = (
        lora.get("peft_expert_lora_parameters", 0)
        or lora.get("kt_expert_lora_parameters", 0)
        or lora.get("qwen_moe_expert_lora_parameters", 0)
    )
try:
    expert = int(expert or 0)
except (TypeError, ValueError):
    expert = 0
if expert <= 0:
    available = lora.get("available", "unknown")
    reason = lora.get("reason", "")
    raise SystemExit(
        "Qwen3 lora_target=all profile has no captured expert LoRA parameters "
        f"(available={available}, reason={reason!r}). This is not comparable to KT expert-LoRA runs. "
        "Set CHECK_TRAINABLE_SURFACE=0 only for historical artifact collection."
    )
print(f"Verified trainable surface: expert_lora_parameters={expert}")
PY
}

log_kv() {
  printf '%s=%s\n' "$1" "$2" | tee -a "${LOG_FILE}"
}

log_kv_if_set() {
  [[ -z "$2" ]] || log_kv "$1" "$2"
  return 0
}

handle_managed_interrupt() {
  local signal_name="${1:-INT}"
  case "${signal_name}" in
    TERM) managed_interrupt_exit_status=143 ;;
    *) managed_interrupt_exit_status=130 ;;
  esac
  managed_interrupted=true
  echo "Interrupted; stopping LF training command." >&2
  trap - INT TERM
  kill_managed_child
  exit "${managed_interrupt_exit_status}"
}

trap 'handle_managed_interrupt INT' INT
trap 'handle_managed_interrupt TERM' TERM
trap 'cleanup_managed_child' EXIT

PY_CHECK='import torch, sys
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
major, minor = torch.cuda.get_device_capability(0)
print(f"CUDA device: {torch.cuda.get_device_name(0)} capability=sm_{major}{minor}")
'
CUDA_VISIBLE_DEVICES=${GPU_ID} NVIDIA_VISIBLE_DEVICES=${GPU_ID} "${ENV_PYTHON}" - <<PY
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
  if [[ ! -f "${KT_GGUF_PY_DIR}/gguf/gguf_reader.py" ]]; then
    if ! PYTHONPATH="${KT_RUN_PYTHONPATH}" "${ENV_PYTHON}" - <<'PY' >/dev/null 2>&1; then
from gguf.gguf_reader import GGUFReader
PY
      echo "Missing KT gguf dependency; set KT_GGUF_PY_DIR to vendored gguf-py or install the gguf package." >&2
      exit 2
    fi
  fi
  env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    NVIDIA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTHONPATH="${KT_RUN_PYTHONPATH}" \
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
  --finetuning_type "${FINETUNING_TYPE}"
  --dataset "${DATASET}"
  --dataset_dir "${LF_DIR}/data"
  --template "${TEMPLATE}"
  --cutoff_len "${CUTOFF_LEN}"
  --max_samples "${MAX_SAMPLES}"
  --overwrite_cache true
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
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
  --bf16 "${TRAINING_BF16}"
  --enable_liger_kernel "${ENABLE_LIGER_KERNEL}"
  --use_unsloth_gc "${USE_UNSLOTH_GC}"
)
if [[ "${FLASH_ATTN}" != "auto" ]]; then
  CMD_ARGS+=(--flash_attn "${FLASH_ATTN}")
fi
if [[ "${FINETUNING_TYPE}" == "lora" ]]; then
  _lora_target_arg="${LORA_TARGET:-all}"
  CMD_ARGS+=(
    --lora_rank "${LORA_RANK}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${LORA_DROPOUT}"
    --lora_target "${_lora_target_arg//+/,}"
  )
fi
[[ -z "${MAX_GRAD_NORM}" ]] || CMD_ARGS+=(--max_grad_norm "${MAX_GRAD_NORM}")

if [[ "${ASYM_DP:-0}" == "1" ]]; then
  # Dense DP rows: no unused params; buffers identical across ranks (frozen base is not a
  # Parameter, DDP only sees the LoRA adapters). MoE rows (gb200_dp.md D2.5) must run with
  # ASYM_DP_FIND_UNUSED=true: an expert LoRA slice can receive zero routes in a step and
  # the DDP reducer hangs waiting on its grad otherwise.
  case "${ASYM_DP_FIND_UNUSED:-false}" in
    true|false) ;;
    *) echo "error: ASYM_DP_FIND_UNUSED must be true or false (got '${ASYM_DP_FIND_UNUSED}')" >&2; exit 2 ;;
  esac
  CMD_ARGS+=(--ddp_find_unused_parameters "${ASYM_DP_FIND_UNUSED:-false}" --ddp_broadcast_buffers false)
fi

if is_zero_backend_run; then
  CMD_ARGS+=(--pure_bf16 false)
else
  CMD_ARGS+=(--pure_bf16 true)
fi

case "${GRADIENT_CHECKPOINTING,,}" in
  1|true|yes|y|on) CMD_ARGS+=(--gradient_checkpointing true --disable_gradient_checkpointing false) ;;
  0|false|no|n|off) CMD_ARGS+=(--gradient_checkpointing false --disable_gradient_checkpointing true) ;;
  *) echo "GRADIENT_CHECKPOINTING must be true or false" >&2; exit 2 ;;
esac

if [[ "${BACKEND}" == "asym_torch" ]]; then
  CMD_ARGS+=(--use_asym_gemm true --asym_backend torch --asym_precision "${ASYM_PRECISION}")
  CMD_ARGS+=(--asym_offload_modules "${ASYM_OFFLOAD_MODULES}" --asym_strict "${ASYM_STRICT}")
  CMD_ARGS+=(--asym_expert_recompute_policy "${ASYM_EXPERT_RECOMPUTE_POLICY}")
  CMD_ARGS+=(--asym_router_mode "${ASYM_ROUTER_MODE}")
elif [[ "${BACKEND}" == "asym" ]]; then
  CMD_ARGS+=(--use_asym_gemm true --asym_backend asym --asym_precision "${ASYM_PRECISION}")
  CMD_ARGS+=(--asym_offload_modules "${ASYM_OFFLOAD_MODULES}" --asym_strict "${ASYM_STRICT}")
  CMD_ARGS+=(--asym_expert_recompute_policy "${ASYM_EXPERT_RECOMPUTE_POLICY}")
  CMD_ARGS+=(--asym_router_mode "${ASYM_ROUTER_MODE}")
elif [[ "${BACKEND}" == kt_* ]]; then
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

if [[ "${BACKEND}" == "asym" || "${BACKEND}" == "asym_torch" ]]; then
  CMD_ARGS+=(--use_asym_cpu_adamw "${USE_ASYM_CPU_ADAMW}")
  CMD_ARGS+=(--asym_cpu_adamw_backend "${ASYM_CPU_ADAMW_BACKEND}")
  CMD_ARGS+=(--asym_cpu_adamw_pin_memory "${ASYM_CPU_ADAMW_PIN_MEMORY}")
  CMD_ARGS+=(--asym_cpu_adamw_fp32_master "${ASYM_CPU_ADAMW_FP32_MASTER}")
  CMD_ARGS+=(--asym_cpu_adamw_grad_offload "${ASYM_CPU_ADAMW_GRAD_OFFLOAD}")
  CMD_ARGS+=(--asym_cpu_adamw_weight_offload "${ASYM_CPU_ADAMW_WEIGHT_OFFLOAD}")
fi

log_kv RUN_ID "${RUN_ID}"
log_kv OUT_DIR "${OUT_DIR}"
log_kv MODEL_NAME_OR_PATH "${MODEL_NAME_OR_PATH}"
log_kv TEMPLATE "${TEMPLATE}"
log_kv BACKEND "${BACKEND}"
log_kv LF_DIR "${LF_DIR}"
log_kv ENV_DIR "${ENV_DIR}"
log_kv FLASH_ATTN "${FLASH_ATTN}"
log_kv_if_set PROFILE_BACKEND_LABEL "${PROFILE_BACKEND_LABEL}"
log_kv GPU_ID "${GPU_ID}"
log_kv NUM_GPUS "${NUM_GPUS}"
if [[ "${ASYM_STP:-0}" == "1" ]]; then
  log_kv ASYM_STP "${ASYM_STP}"
  log_kv ASYM_STP_TP_SIZE "${ASYM_STP_TP_SIZE}"
  log_kv ASYM_STP_WEIGHT_MODE "${ASYM_STP_WEIGHT_MODE}"
  log_kv ASYM_STP_SHARED_ARENA "${ASYM_STP_SHARED_ARENA}"
  log_kv ASYM_STP_COORD "${ASYM_STP_COORD}"
  log_kv ASYM_STP_MOE "${ASYM_STP_MOE:-0}"
  log_kv TRAIN_OOM_SCORE_ADJ "${TRAIN_OOM_SCORE_ADJ}"
  log_kv HOST_MEM_WATCHDOG "${HOST_MEM_WATCHDOG}"
  log_kv PROFILE_GLOBAL_BATCH_SIZE_STP "$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
fi
if [[ "${ASYM_DP:-0}" == "1" ]]; then
  log_kv ASYM_DP "${ASYM_DP}"
  log_kv TRAIN_OOM_SCORE_ADJ "${TRAIN_OOM_SCORE_ADJ}"
  log_kv HOST_MEM_WATCHDOG "${HOST_MEM_WATCHDOG}"
fi
log_kv NUMACTL_ENABLE "${NUMACTL_ENABLE}"
if [[ "${NUMACTL_ENABLE}" == "1" ]]; then
  log_kv NUMACTL_MEMBIND "${NUMACTL_MEMBIND}"
  log_kv NUMACTL_CPUNODEBIND "${NUMACTL_CPUNODEBIND}"
fi
is_torch_run && log_kv DIST_LAUNCHER "${DIST_LAUNCHER}"
log_kv SEED "${SEED}"
log_kv TRAINING_BF16 "${TRAINING_BF16}"
log_kv ENABLE_LIGER_KERNEL "${ENABLE_LIGER_KERNEL}"
log_kv LIGER_LOSS_TAG "${LIGER_LOSS_TAG}"
log_kv USE_UNSLOTH_GC "${USE_UNSLOTH_GC}"
log_kv UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU "${UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU}"
log_kv UNSLOTH_GC_OUTER_HBM_EVERY_N "${UNSLOTH_GC_OUTER_HBM_EVERY_N}"
log_kv ASYMM_EXPERT_SILU_BWD_GPU "${ASYMM_EXPERT_SILU_BWD_GPU}"
log_kv ASYMM_DENSE_MLP_SURGICAL_OFFLOAD "${ASYMM_DENSE_MLP_SURGICAL_OFFLOAD}"
log_kv ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD "${ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD}"
log_kv ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT "${ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT}"
log_kv ASYMM_DENSE_MLP_FINEGRAINED_NOGRAD_CPU_OFFLOAD "${ASYMM_DENSE_MLP_FINEGRAINED_NOGRAD_CPU_OFFLOAD}"
log_kv ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD "${ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD}"
log_kv ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS "${ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS}"
log_kv ASYMM_MLP_RECOMPUTE_CHUNK "${ASYMM_MLP_RECOMPUTE_CHUNK}"
log_kv PREPROCESSING_NUM_WORKERS "${PREPROCESSING_NUM_WORKERS}"
log_kv DATALOADER_NUM_WORKERS "${DATALOADER_NUM_WORKERS}"
log_kv_if_set MAX_GRAD_NORM "${MAX_GRAD_NORM}"
if [[ "${BACKEND}" == kt_* ]]; then
  log_kv KT_BACKEND "${KT_BACKEND_INTERNAL}"
  log_kv KT_KERNEL_DIR "${KT_KERNEL_DIR}"
  log_kv KT_REPO_DIR "${KT_REPO_DIR}"
  log_kv KT_GGUF_PY_DIR "${KT_GGUF_PY_DIR}"
  log_kv CHECK_KT_CALLS "${CHECK_KT_CALLS}"
  log_kv KT_REQUIRE_STARTUP "${KT_REQUIRE_STARTUP}"
  log_kv KT_REQUIRE_FUSED_LORA_STARTUP "${KT_REQUIRE_FUSED_LORA_STARTUP}"
  [[ "${BACKEND}" == "kt_torchbf16" ]] && log_kv KT_TORCHBF16_SFT_DEVICE "${KT_TORCHBF16_SFT_DEVICE}"
  if [[ "${BACKEND}" == "kt_armbf16" ]]; then
    log_kv KT_ARM_OMP_NUM_THREADS "${KT_ARM_OMP_NUM_THREADS}"
    log_kv KT_ARM_OMP_PROC_BIND "${KT_ARM_OMP_PROC_BIND}"
    log_kv KT_ARM_OMP_PLACES "${KT_ARM_OMP_PLACES}"
    log_kv KT_SFT_PROGRESS "${KT_SFT_PROGRESS}"
    log_kv KT_ARM_SFT_PROFILE "${KT_ARM_SFT_PROFILE}"
    log_kv KT_ARM_SFT_POOL_LOG "${KT_ARM_SFT_POOL_LOG}"
    log_kv KT_ARM_SFT_TOP_K "${KT_ARM_SFT_TOP_K}"
    log_kv_if_set KT_ARM_SFT_TOKEN_CHUNK_SIZE "${KT_ARM_SFT_TOKEN_CHUNK_SIZE}"
    log_kv KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK "${KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK}"
    log_kv KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH "${KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH}"
    log_kv KT_ARM_ALLOW_NSYS_WITHOUT_SOURCE_OK "${KT_ARM_ALLOW_NSYS_WITHOUT_SOURCE_OK}"
    log_kv KT_ARM_ALLOW_RAW_NSYS_WITHOUT_SOURCE_OK "${KT_ARM_ALLOW_RAW_NSYS_WITHOUT_SOURCE_OK}"
    log_kv_if_set KT_ARM_SOURCE_OK_PROFILE_JSON "${KT_ARM_SOURCE_OK_PROFILE_JSON}"
    log_kv KT_ARM_LOGICAL_QLEN "${KT_ARM_LOGICAL_QLEN}"
    log_kv KT_ARM_EFFECTIVE_ROUTE_QLEN "${KT_ARM_EFFECTIVE_ROUTE_QLEN}"
    log_kv KT_ARM_TOKEN_CHUNKS "${KT_ARM_TOKEN_CHUNKS}"
    log_kv KT_ARM_ROUTE_RANK_WORK "${KT_ARM_ROUTE_RANK_WORK}"
    log_kv_if_set KT_ARM_SFT_MAX_ROUTE_RANK_WORK "${KT_ARM_SFT_MAX_ROUTE_RANK_WORK}"
    log_kv_if_set KT_ARM_SFT_BACKWARD_THREADS "${KT_ARM_SFT_BACKWARD_THREADS}"
    log_kv_if_set KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES "${KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES}"
  fi
  log_kv_if_set KT_NUM_THREADS "${KT_NUM_THREADS}"
  log_kv_if_set KT_THREADPOOL_COUNT "${KT_THREADPOOL_COUNT}"
  log_kv_if_set KT_MAX_CACHE_DEPTH "${KT_MAX_CACHE_DEPTH}"
  log_kv KT_TP_ENABLED "${KT_TP_ENABLED}"
  log_kv_if_set KT_SHARE_BACKWARD_BB "${KT_SHARE_BACKWARD_BB}"
  log_kv_if_set KT_NUM_GPU_EXPERTS "${KT_NUM_GPU_EXPERTS}"
  log_kv_if_set KT_WEIGHT_PATH "${KT_WEIGHT_PATH}"
  log_kv_if_set KT_EXPERT_CHECKPOINT_PATH "${KT_EXPERT_CHECKPOINT_PATH}"
  log_kv_if_set KT_USE_LORA_EXPERTS "${KT_USE_LORA_EXPERTS}"
  log_kv_if_set KT_LORA_EXPERT_NUM "${KT_LORA_EXPERT_NUM}"
  log_kv_if_set KT_LORA_EXPERT_INTERMEDIATE_SIZE "${KT_LORA_EXPERT_INTERMEDIATE_SIZE}"
fi
if is_zero_backend_run; then
  log_kv ZERO_BACKEND_LABEL "${ZERO_BACKEND_LABEL}"
  log_kv TORCH_DEEPSPEED_CONFIG "${TORCH_DEEPSPEED_CONFIG}"
  case "${DIST_LAUNCHER}" in
    torchrun) log_kv TORCHRUN_CMD "${TORCHRUN_CMD[*]}" ;;
    accelerate) log_kv ACCELERATE_CMD "${ACCELERATE_CMD[*]}" ;;
    deepspeed) log_kv DEEPSPEED_CMD "${DEEPSPEED_CMD[*]}" ;;
  esac
  log_kv MASTER_PORT "${MASTER_PORT_VALUE}"
  if is_superoffload_zero_run; then
    log_kv DEEPSPEED_DIR "${DEEPSPEED_DIR}"
    log_kv CHECK_SUPEROFFLOAD "${CHECK_SUPEROFFLOAD}"
  fi
  if is_cpuadam_zero_run; then
    log_kv CHECK_CPUADAM "${CHECK_CPUADAM}"
  fi
elif is_plain_torch_run; then
  case "${DIST_LAUNCHER}" in
    torchrun) log_kv TORCHRUN_CMD "${TORCHRUN_CMD[*]}" ;;
    accelerate) log_kv ACCELERATE_CMD "${ACCELERATE_CMD[*]}" ;;
    deepspeed) log_kv DEEPSPEED_CMD "${DEEPSPEED_CMD[*]}" ;;
  esac
  log_kv MASTER_PORT "${MASTER_PORT_VALUE}"
fi
log_kv ASYM_EXPERT_RECOMPUTE_POLICY "${ASYM_EXPERT_RECOMPUTE_POLICY}"
log_kv ASYM_ROUTER_MODE "${ASYM_ROUTER_MODE}"
log_kv ASYMM_EXPERT_ACT_OFFLOAD "${ASYMM_EXPERT_ACT_OFFLOAD}"
log_kv ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD "${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}"
log_kv ASYMM_ATTN_ACT_OFFLOAD "${ASYMM_ATTN_ACT_OFFLOAD}"
log_kv ASYMM_LAYER_ACT_OFFLOAD "${ASYMM_LAYER_ACT_OFFLOAD}"
log_kv ASYMM_LAYER_GC "${ASYMM_LAYER_GC}"
log_kv ASYM_OFFLOAD_ACT_RECOMPUTE "${ASYM_OFFLOAD_ACT_RECOMPUTE}"
log_kv ASYM_OFFLOAD_X_UNPACKED "${ASYM_OFFLOAD_X_UNPACKED}"
log_kv ATTN_GC_ENABLED "${ATTN_GC_ENABLED}"
log_kv LAYER_GC_ENABLED "${LAYER_GC_ENABLED}"
if [[ "${BACKEND}" == "asym" || "${BACKEND}" == "asym_torch" ]]; then
  log_kv USE_ASYM_CPU_ADAMW "${USE_ASYM_CPU_ADAMW}"
  log_kv ASYM_CPU_ADAMW_BACKEND "${ASYM_CPU_ADAMW_BACKEND}"
  log_kv ASYM_CPU_ADAMW_PIN_MEMORY "${ASYM_CPU_ADAMW_PIN_MEMORY}"
  log_kv ASYM_CPU_ADAMW_FP32_MASTER "${ASYM_CPU_ADAMW_FP32_MASTER}"
  log_kv ASYM_CPU_ADAMW_GRAD_OFFLOAD "${ASYM_CPU_ADAMW_GRAD_OFFLOAD}"
  log_kv ASYM_CPU_ADAMW_WEIGHT_OFFLOAD "${ASYM_CPU_ADAMW_WEIGHT_OFFLOAD}"
  if [[ "${uses_asym_deepspeed_cpuadam}" == "true" ]]; then
    log_kv DEEPSPEED_DIR "${DEEPSPEED_DIR}"
  fi
fi
log_kv PROFILE "${PROFILE}"
if [[ "${PROFILE}" == "1" ]]; then
  log_kv PROFILE_PROFILER "${PROFILE_PROFILER}"
  log_kv PROFILE_LEVEL "${PROFILE_LEVEL}"
  log_kv PROFILE_LAYERS "${PROFILE_LAYERS}"
  log_kv PROFILE_MEMORY_BREAKDOWN "${PROFILE_MEMORY_BREAKDOWN}"
  log_kv PROFILE_MEMORY_BREAKDOWN_INTERVAL "${PROFILE_MEMORY_BREAKDOWN_INTERVAL}"
  log_kv_if_set PROFILE_MEMORY_BREAKDOWN_STEPS "${PROFILE_MEMORY_BREAKDOWN_STEPS}"
  log_kv PROFILE_MEMORY_BREAKDOWN_MODULES "${PROFILE_MEMORY_BREAKDOWN_MODULES}"
  log_kv PROFILE_MEMORY_BREAKDOWN_OUTPUT "${PROFILE_MEMORY_BREAKDOWN_OUTPUT}"
  log_kv PROFILE_LIVE_ACTIVATION_DETAILS "${PROFILE_LIVE_ACTIVATION_DETAILS}"
  log_kv PROFILE_LIVE_ACTIVATION_TOPK "${PROFILE_LIVE_ACTIVATION_TOPK}"
  log_kv PROFILE_MEMORY_SNAPSHOT "${PROFILE_MEMORY_SNAPSHOT}"
  log_kv_if_set PROFILE_MEMORY_SNAPSHOT_PATH "${PROFILE_MEMORY_SNAPSHOT_PATH}"
  log_kv PROFILE_EXTERNAL_MEMORY "${PROFILE_EXTERNAL_MEMORY}"
  log_kv PROFILE_SYNC "${PROFILE_SYNC}"
  log_kv PROFILE_MODULE_FILTER "${PROFILE_MODULE_FILTER}"
  log_kv PROFILE_SOURCE_JSON "${PROFILE_SOURCE_JSON}"
  log_kv PROFILE_HEARTBEAT_JSON "${PROFILE_HEARTBEAT_JSON}"
  log_kv PROFILE_PARTIAL_INTERVAL_SECONDS "${PROFILE_PARTIAL_INTERVAL_SECONDS}"
  log_kv PROFILE_JSON "${PROFILE_JSON}"
  log_kv PROFILE_OUTPUT_DIR "${PROFILE_OUTPUT_DIR}"
  log_kv PROFILE_SUMMARY_MD "${PROFILE_SUMMARY_MD}"
  log_kv PER_DEVICE_TRAIN_BATCH_SIZE "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  log_kv GLOBAL_BATCH_SIZE "${PROFILE_GLOBAL_BATCH_SIZE}"
  log_kv LOGICAL_QLEN "${PROFILE_LOGICAL_QLEN}"
  log_kv_if_set PROFILE_NSYS_PREFIX "${PROFILE_NSYS_PREFIX}"
  log_kv_if_set PROFILE_NSYS_SQLITE "${PROFILE_NSYS_SQLITE}"
  if [[ "${PROFILE_PROFILER}" == "nsys" ]]; then
    log_kv PROFILE_NSYS_CAPTURE_RANGE "${PROFILE_NSYS_CAPTURE_RANGE}"
    log_kv PROFILE_NSYS_GPU_METRICS_DEVICES "${PROFILE_NSYS_GPU_METRICS_DEVICES}"
  fi
fi

RUN_PYTHONPATH="${ASYM_DIR}:${LF_DIR}/src:${PYTHONPATH:-}"
if is_superoffload_zero_run || [[ "${uses_asym_deepspeed_cpuadam}" == "true" ]]; then
  RUN_PYTHONPATH="${DEEPSPEED_DIR}:${ASYM_DIR}:${LF_DIR}/src:${PYTHONPATH:-}"
elif [[ "${BACKEND}" == kt_* ]]; then
  RUN_PYTHONPATH="${KT_RUN_PYTHONPATH}"
fi

RUN_ENV=(
  PATH="${ENV_DIR}/bin:${PATH}"
  PYTHONPATH="${RUN_PYTHONPATH}"
  ASYMM_EXPERT_ACT_OFFLOAD="${ASYMM_EXPERT_ACT_OFFLOAD}"
  ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD="${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}"
  ASYMM_EXPERT_SILU_BWD_GPU="${ASYMM_EXPERT_SILU_BWD_GPU}"
  ASYMM_DENSE_MLP_SURGICAL_OFFLOAD="${ASYMM_DENSE_MLP_SURGICAL_OFFLOAD}"
  ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD="${ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD}"
  ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT="${ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT}"
  ASYMM_DENSE_MLP_FINEGRAINED_NOGRAD_CPU_OFFLOAD="${ASYMM_DENSE_MLP_FINEGRAINED_NOGRAD_CPU_OFFLOAD}"
  ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD="${ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD}"
  ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS="${ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS}"
  ASYMM_ATTN_ACT_OFFLOAD="${ASYMM_ATTN_ACT_OFFLOAD}"
  ASYMM_LAYER_ACT_OFFLOAD="${ASYMM_LAYER_ACT_OFFLOAD}"
  ASYMM_LAYER_GC="${ASYMM_LAYER_GC}"
  ASYMM_LAYER_GC_RECOMPUTE="${ASYMM_LAYER_GC_RECOMPUTE:-false}"
  ASYMM_MLP_RECOMPUTE_CHUNK="${ASYMM_MLP_RECOMPUTE_CHUNK}"
  ASYM_OFFLOAD_ACT_RECOMPUTE="${ASYM_OFFLOAD_ACT_RECOMPUTE}"
  ASYM_OFFLOAD_X_UNPACKED="${ASYM_OFFLOAD_X_UNPACKED}"
  ASYM_CPU_ADAMW_GRAD_OFFLOAD="${ASYM_CPU_ADAMW_GRAD_OFFLOAD}"
  ASYM_CPU_ADAMW_WEIGHT_OFFLOAD="${ASYM_CPU_ADAMW_WEIGHT_OFFLOAD}"
  LF_QWEN_MOE_EXPERT_LORA_IMPL="${LF_QWEN_MOE_EXPERT_LORA_IMPL}"
  ENABLE_LIGER_KERNEL="${ENABLE_LIGER_KERNEL}"
  USE_UNSLOTH_GC="${USE_UNSLOTH_GC}"
  UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU="${UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU}"
  UNSLOTH_GC_OUTER_HBM_EVERY_N="${UNSLOTH_GC_OUTER_HBM_EVERY_N}"
  ASYM_GEMM_LF_CONFIG_LIGER_LOSS="${LIGER_LOSS_TAG}"
  ASYM_GEMM_LF_CONFIG_RECOMP_OFF_STAGE="${ASYM_GEMM_LF_CONFIG_RECOMP_OFF_STAGE:-}"
  ASYM_GEMM_LF_CONFIG_USE_UNSLOTH_GC="${USE_UNSLOTH_GC}"
  ASYM_GEMM_LF_CONFIG_UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU="${UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU}"
  ASYM_GEMM_LF_CONFIG_UNSLOTH_GC_OUTER_HBM_EVERY_N="${UNSLOTH_GC_OUTER_HBM_EVERY_N}"
  ASYM_GEMM_LF_CONFIG_ASYMM_EXPERT_SILU_BWD_GPU="${ASYMM_EXPERT_SILU_BWD_GPU}"
  ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_SURGICAL_OFFLOAD="${ASYMM_DENSE_MLP_SURGICAL_OFFLOAD}"
  ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD="${ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD}"
  ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT="${ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT}"
  ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_FINEGRAINED_NOGRAD_CPU_OFFLOAD="${ASYMM_DENSE_MLP_FINEGRAINED_NOGRAD_CPU_OFFLOAD}"
  ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD="${ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD}"
  ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS="${ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS}"
  ASYM_GEMM_LF_CONFIG_ASYMM_MLP_RECOMPUTE_CHUNK="${ASYMM_MLP_RECOMPUTE_CHUNK}"
  ASYM_GEMM_LF_CONFIG_QWEN_EXPERT_LORA_IMPL="${LF_QWEN_MOE_EXPERT_LORA_IMPL}"
)
if [[ -n "${TRITON_CACHE_DIR:-}" ]]; then
  RUN_ENV+=(TRITON_CACHE_DIR="${TRITON_CACHE_DIR}")
fi
ENV_CMD=(env)
if ! { is_torch_run && [[ "${DIST_LAUNCHER}" == "deepspeed" ]]; }; then
  RUN_ENV=(CUDA_VISIBLE_DEVICES="${GPU_ID}" NVIDIA_VISIBLE_DEVICES="${GPU_ID}" "${RUN_ENV[@]}")
else
  ENV_CMD+=( -u CUDA_VISIBLE_DEVICES -u NVIDIA_VISIBLE_DEVICES )
fi
if [[ "${BACKEND}" == kt_* ]]; then
  RUN_ENV+=(
    USE_KT=1
    ACCELERATE_KT_BACKEND="${KT_BACKEND_INTERNAL}"
    ACCELERATE_KT_TP_ENABLED="${KT_TP_ENABLED}"
    ASYM_GEMM_LF_REQUIRE_KT_STARTUP="${KT_REQUIRE_STARTUP}"
    ASYM_GEMM_LF_REQUIRE_KT_FUSED_LORA_STARTUP="${KT_REQUIRE_FUSED_LORA_STARTUP}"
    ASYM_GEMM_LF_CONFIG_KT_REQUIRE_STARTUP="${KT_REQUIRE_STARTUP}"
    ASYM_GEMM_LF_CONFIG_KT_REQUIRE_FUSED_LORA_STARTUP="${KT_REQUIRE_FUSED_LORA_STARTUP}"
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
      KT_SFT_PROGRESS="${KT_SFT_PROGRESS}"
      KT_ARM_SFT_PROFILE="${KT_ARM_SFT_PROFILE}"
      KT_ARM_SFT_POOL_LOG="${KT_ARM_SFT_POOL_LOG}"
      KT_ARM_SFT_TOP_K="${KT_ARM_SFT_TOP_K}"
      KT_ARM_SFT_MAX_ROUTE_RANK_WORK="${KT_ARM_SFT_MAX_ROUTE_RANK_WORK}"
      KT_ARM_SFT_DEFAULT_MAX_ROUTE_RANK_WORK="${KT_ARM_SFT_DEFAULT_MAX_ROUTE_RANK_WORK}"
      KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK="${KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_OMP_NUM_THREADS="${KT_ARM_OMP_NUM_THREADS}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_OMP_PROC_BIND="${KT_ARM_OMP_PROC_BIND}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_OMP_PLACES="${KT_ARM_OMP_PLACES}"
      ASYM_GEMM_LF_CONFIG_KT_SFT_PROGRESS="${KT_SFT_PROGRESS}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_SFT_PROFILE="${KT_ARM_SFT_PROFILE}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_SFT_POOL_LOG="${KT_ARM_SFT_POOL_LOG}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_SFT_TOP_K="${KT_ARM_SFT_TOP_K}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_LOGICAL_QLEN="${KT_ARM_LOGICAL_QLEN}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_EFFECTIVE_ROUTE_QLEN="${KT_ARM_EFFECTIVE_ROUTE_QLEN}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_TOKEN_CHUNKS="${KT_ARM_TOKEN_CHUNKS}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_ROUTE_RANK_WORK="${KT_ARM_ROUTE_RANK_WORK}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_SFT_TOKEN_CHUNK_SIZE="${KT_ARM_SFT_TOKEN_CHUNK_SIZE}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_SFT_MAX_ROUTE_RANK_WORK="${KT_ARM_SFT_MAX_ROUTE_RANK_WORK}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_SFT_BACKWARD_THREADS="${KT_ARM_SFT_BACKWARD_THREADS}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES="${KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK="${KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH="${KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_ALLOW_NSYS_WITHOUT_SOURCE_OK="${KT_ARM_ALLOW_NSYS_WITHOUT_SOURCE_OK}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_ALLOW_RAW_NSYS_WITHOUT_SOURCE_OK="${KT_ARM_ALLOW_RAW_NSYS_WITHOUT_SOURCE_OK}"
      ASYM_GEMM_LF_CONFIG_KT_ARM_SOURCE_OK_PROFILE_JSON="${KT_ARM_SOURCE_OK_PROFILE_JSON}"
    )
    [[ -z "${KT_ARM_SFT_TOKEN_CHUNK_SIZE}" ]] || RUN_ENV+=(KT_ARM_SFT_TOKEN_CHUNK_SIZE="${KT_ARM_SFT_TOKEN_CHUNK_SIZE}")
    [[ -z "${KT_ARM_SFT_BACKWARD_THREADS}" ]] || RUN_ENV+=(KT_ARM_SFT_BACKWARD_THREADS="${KT_ARM_SFT_BACKWARD_THREADS}")
    [[ -z "${KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES}" ]] || RUN_ENV+=(KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES="${KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES}")
  fi
  [[ -z "${KT_NUM_THREADS}" ]] || RUN_ENV+=(ACCELERATE_KT_NUM_THREADS="${KT_NUM_THREADS}")
  [[ -z "${KT_THREADPOOL_COUNT}" ]] || RUN_ENV+=(ACCELERATE_KT_THREADPOOL_COUNT="${KT_THREADPOOL_COUNT}")
  [[ -z "${KT_MAX_CACHE_DEPTH}" ]] || RUN_ENV+=(ACCELERATE_KT_MAX_CACHE_DEPTH="${KT_MAX_CACHE_DEPTH}")
  [[ -z "${KT_SHARE_BACKWARD_BB}" ]] || RUN_ENV+=(ACCELERATE_KT_SHARE_BACKWARD_BB="${KT_SHARE_BACKWARD_BB}")
  [[ -z "${KT_NUM_GPU_EXPERTS}" ]] || RUN_ENV+=(ACCELERATE_KT_NUM_GPU_EXPERTS="${KT_NUM_GPU_EXPERTS}")
  [[ -z "${KT_WEIGHT_PATH}" ]] || RUN_ENV+=(ACCELERATE_KT_WEIGHT_PATH="${KT_WEIGHT_PATH}")
  [[ -z "${KT_EXPERT_CHECKPOINT_PATH}" ]] || RUN_ENV+=(ACCELERATE_KT_EXPERT_CHECKPOINT_PATH="${KT_EXPERT_CHECKPOINT_PATH}")
  [[ -z "${KT_USE_LORA_EXPERTS}" ]] || RUN_ENV+=(ACCELERATE_KT_USE_LORA_EXPERTS="${KT_USE_LORA_EXPERTS}")
  [[ -z "${KT_LORA_EXPERT_NUM}" ]] || RUN_ENV+=(ACCELERATE_KT_LORA_EXPERT_NUM="${KT_LORA_EXPERT_NUM}")
  [[ -z "${KT_LORA_EXPERT_INTERMEDIATE_SIZE}" ]] || RUN_ENV+=(ACCELERATE_KT_LORA_EXPERT_INTERMEDIATE_SIZE="${KT_LORA_EXPERT_INTERMEDIATE_SIZE}")
elif [[ "${BACKEND}" == "torch" ]]; then
  :
else
  RUN_ENV+=(USE_ASYM_GEMM=1 ASYM_GEMM_LF_LOG_RUNTIME_STATS=1)
fi

if [[ "${ASYM_STP:-0}" == "1" ]]; then
  RUN_ENV+=(
    ASYM_STP=1
    ASYM_STP_TP_SIZE="${ASYM_STP_TP_SIZE}"
    ASYM_STP_WEIGHT_MODE="${ASYM_STP_WEIGHT_MODE}"
    ASYM_STP_SHARED_ARENA="${ASYM_STP_SHARED_ARENA}"
    ASYM_STP_COORD="${ASYM_STP_COORD}"
    ASYM_STP_MOE="${ASYM_STP_MOE:-0}"
    ASYM_GEMM_LF_CONFIG_ASYM_STP=1
    ASYM_GEMM_LF_CONFIG_ASYM_STP_TP_SIZE="${ASYM_STP_TP_SIZE}"
    ASYM_GEMM_LF_CONFIG_ASYM_STP_WEIGHT_MODE="${ASYM_STP_WEIGHT_MODE}"
    ASYM_GEMM_LF_CONFIG_ASYM_STP_SHARED_ARENA="${ASYM_STP_SHARED_ARENA}"
    ASYM_GEMM_LF_CONFIG_ASYM_STP_COORD="${ASYM_STP_COORD}"
    ASYM_GEMM_LF_CONFIG_TRAIN_OOM_SCORE_ADJ="${TRAIN_OOM_SCORE_ADJ}"
    ASYM_GEMM_LF_CONFIG_HOST_MEM_WATCHDOG="${HOST_MEM_WATCHDOG}"
  )
fi
if [[ "${ASYM_DP:-0}" == "1" ]]; then
  RUN_ENV+=(
    ASYM_DP=1
    ASYM_GEMM_LF_CONFIG_ASYM_DP=1
    ASYM_GEMM_LF_CONFIG_TRAIN_OOM_SCORE_ADJ="${TRAIN_OOM_SCORE_ADJ}"
    ASYM_GEMM_LF_CONFIG_HOST_MEM_WATCHDOG="${HOST_MEM_WATCHDOG}"
  )
fi

if is_torch_run; then
  RUN_ENV+=(
    FORCE_TORCHRUN=1
    NNODES="${NNODES:-1}"
    NODE_RANK="${NODE_RANK:-0}"
    NPROC_PER_NODE="${NUM_GPUS}"
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    MASTER_PORT="${MASTER_PORT_VALUE}"
  )
fi

if is_zero_backend_run || [[ "${uses_asym_deepspeed_cpuadam}" == "true" ]]; then
  RUN_ENV+=(
    ASYM_GEMM_LF_CONFIG_DEEPSPEED_DIR="${DEEPSPEED_DIR}"
  )
fi

if is_zero_backend_run; then
  RUN_ENV+=(
    ASYM_GEMM_LF_CONFIG_DEEPSPEED_CONFIG="${TORCH_DEEPSPEED_CONFIG}"
  )
fi

if is_superoffload_zero_run; then
  RUN_ENV+=(
    ASYM_GEMM_LF_CONFIG_SUPEROFFLOAD_CONFIG="${TORCH_DEEPSPEED_CONFIG}"
  )
fi

if is_cpuadam_zero_run; then
  RUN_ENV+=(
    ASYM_GEMM_LF_CONFIG_CPUADAM_CONFIG="${TORCH_DEEPSPEED_CONFIG}"
  )
fi

if is_zero_backend_run; then
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
    ASYM_GEMM_LF_PROFILE_MEMORY_ATTRIBUTION="${PROFILE_MEMORY_BREAKDOWN}"
    ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN="${PROFILE_MEMORY_BREAKDOWN}"
    ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN_INTERVAL="${PROFILE_MEMORY_BREAKDOWN_INTERVAL}"
    ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN_STEPS="${PROFILE_MEMORY_BREAKDOWN_STEPS}"
    ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN_MODULES="${PROFILE_MEMORY_BREAKDOWN_MODULES}"
    ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN_OUTPUT="${PROFILE_MEMORY_BREAKDOWN_OUTPUT}"
    ASYM_GEMM_LF_PROFILE_LIVE_ACTIVATION_DETAILS="${PROFILE_LIVE_ACTIVATION_DETAILS}"
    ASYM_GEMM_LF_PROFILE_LIVE_ACTIVATION_TOPK="${PROFILE_LIVE_ACTIVATION_TOPK}"
    ASYM_GEMM_LF_PROFILE_MEMORY_SNAPSHOT="${PROFILE_MEMORY_SNAPSHOT}"
    ASYM_GEMM_LF_PROFILE_MEMORY_SNAPSHOT_PATH="${PROFILE_MEMORY_SNAPSHOT_PATH}"
    ASYM_GEMM_LF_PROFILE_EXTERNAL_MEMORY="${PROFILE_EXTERNAL_MEMORY}"
    ASYM_GEMM_LF_PROFILE_SYNC="${PROFILE_SYNC}"
    ASYM_GEMM_LF_PROFILE_MODULE_FILTER="${PROFILE_MODULE_FILTER}"
    ASYM_GEMM_LF_HEARTBEAT_JSON="${PROFILE_HEARTBEAT_JSON}"
    ASYM_GEMM_LF_PROFILE_PARTIAL_INTERVAL_SECONDS="${PROFILE_PARTIAL_INTERVAL_SECONDS}"
    ASYM_GEMM_LF_CONFIG_WORKLOAD="${PROFILE_WORKLOAD_LABEL:-${MODEL_TAG}}"
    ASYM_GEMM_LF_CONFIG_BACKEND="${PROFILE_BACKEND_LABEL:-${BACKEND}}"
    ASYM_GEMM_LF_CONFIG_DIST_LAUNCHER="${DIST_LAUNCHER}"
    ASYM_GEMM_LF_CONFIG_ASYM_OFFLOAD_MODULES="${ASYM_OFFLOAD_MODULES}"
    ASYM_GEMM_LF_CONFIG_ASYM_STRICT="${ASYM_STRICT}"
    ASYM_GEMM_LF_CONFIG_RECOMP_OFF_STAGE="${ASYM_GEMM_LF_CONFIG_RECOMP_OFF_STAGE:-}"
    ASYM_GEMM_LF_CONFIG_USE_UNSLOTH_GC="${USE_UNSLOTH_GC}"
    ASYM_GEMM_LF_CONFIG_UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU="${UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU}"
    ASYM_GEMM_LF_CONFIG_UNSLOTH_GC_OUTER_HBM_EVERY_N="${UNSLOTH_GC_OUTER_HBM_EVERY_N}"
    ASYM_GEMM_LF_CONFIG_ASYMM_EXPERT_ACT_OFFLOAD="${ASYMM_EXPERT_ACT_OFFLOAD}"
    ASYM_GEMM_LF_CONFIG_ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD="${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}"
    ASYM_GEMM_LF_CONFIG_ASYMM_EXPERT_SILU_BWD_GPU="${ASYMM_EXPERT_SILU_BWD_GPU}"
    ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_SURGICAL_OFFLOAD="${ASYMM_DENSE_MLP_SURGICAL_OFFLOAD}"
    ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD="${ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD}"
    ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT="${ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT}"
    ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_FINEGRAINED_NOGRAD_CPU_OFFLOAD="${ASYMM_DENSE_MLP_FINEGRAINED_NOGRAD_CPU_OFFLOAD}"
    ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_ACT_OFFLOAD="${ASYMM_ATTN_ACT_OFFLOAD}"
    ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_ACT_OFFLOAD="${ASYMM_LAYER_ACT_OFFLOAD}"
    ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_GC="${ASYMM_LAYER_GC}"
    ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_SDPA_RECOMPUTE="${ASYMM_ATTN_SDPA_RECOMPUTE}"
    ASYM_GEMM_LF_CONFIG_ASYM_OFFLOAD_ACT_RECOMPUTE="${ASYM_OFFLOAD_ACT_RECOMPUTE}"
    ASYM_GEMM_LF_CONFIG_ASYM_OFFLOAD_X_UNPACKED="${ASYM_OFFLOAD_X_UNPACKED}"
    ASYM_GEMM_LF_CONFIG_ASYMM_MLP_RECOMPUTE_CHUNK="${ASYMM_MLP_RECOMPUTE_CHUNK}"
    ASYM_GEMM_LF_CONFIG_ATTN_GC_ENABLED="${ATTN_GC_ENABLED}"
    ASYM_GEMM_LF_CONFIG_LAYER_GC_ENABLED="${LAYER_GC_ENABLED}"
    ASYM_GEMM_LF_CONFIG_ROUTER_MODE="${ASYM_ROUTER_MODE}"
    ASYM_GEMM_LF_CONFIG_PRECISION="${profile_precision}"
    ASYM_GEMM_LF_CONFIG_TRAINING_BF16="${TRAINING_BF16}"
    ASYM_GEMM_LF_CONFIG_SEQ_LEN="${CUTOFF_LEN}"
    ASYM_GEMM_LF_CONFIG_PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}"
    ASYM_GEMM_LF_CONFIG_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}"
    ASYM_GEMM_LF_CONFIG_GLOBAL_BATCH_SIZE="${PROFILE_GLOBAL_BATCH_SIZE}"
    ASYM_GEMM_LF_CONFIG_GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}"
    ASYM_GEMM_LF_CONFIG_LOGICAL_QLEN="${PROFILE_LOGICAL_QLEN}"
    ASYM_GEMM_LF_CONFIG_ACTIVATION_RECOMPUTE="${GRADIENT_CHECKPOINTING}"
    ASYM_GEMM_LF_CONFIG_EXPERT_POLICY="${PROFILE_EXPERT_POLICY}"
    ASYM_GEMM_LF_CONFIG_PROFILE_LEVEL="${PROFILE_LEVEL}"
    ASYM_GEMM_LF_CONFIG_WARMUP_STEPS="${PROFILE_WARMUP_STEPS:-0}"
    ASYM_GEMM_LF_CONFIG_MEASURE_STEPS="${PROFILE_MEASURE_STEPS:-${MAX_STEPS}}"
    ASYM_GEMM_LF_CONFIG_TOTAL_STEPS="${PROFILE_TOTAL_STEPS:-${MAX_STEPS}}"
    ASYM_GEMM_LF_CONFIG_PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS}"
    ASYM_GEMM_LF_CONFIG_DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS}"
    ASYM_GEMM_LF_CONFIG_MAX_GRAD_NORM="${MAX_GRAD_NORM}"
    ASYM_GEMM_LF_CONFIG_USE_ASYM_CPU_ADAMW="${USE_ASYM_CPU_ADAMW}"
    ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_BACKEND="${ASYM_CPU_ADAMW_BACKEND}"
    ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_PIN_MEMORY="${ASYM_CPU_ADAMW_PIN_MEMORY}"
    ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_FP32_MASTER="${ASYM_CPU_ADAMW_FP32_MASTER}"
    ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_GRAD_OFFLOAD="${ASYM_CPU_ADAMW_GRAD_OFFLOAD}"
    ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_WEIGHT_OFFLOAD="${ASYM_CPU_ADAMW_WEIGHT_OFFLOAD}"
  )
  if [[ -n "${ASYM_NVME_ROLES:-}" ]]; then
    # raw ASYM_NVME_* for the store; ASYM_GEMM_LF_CONFIG_* mirror for the profile report.
    RUN_ENV+=(
      ASYM_NVME_ROLES="${ASYM_NVME_ROLES}"
      ASYM_NVME_PATH="${ASYM_NVME_PATH:-/scratch_local/asym_nvme}"
      ASYM_NVME_SYNC="${ASYM_NVME_SYNC:-1}"
      ASYM_NVME_ACT_CPU_BUDGET_BYTES="${ASYM_NVME_ACT_CPU_BUDGET_BYTES:-auto}"
      ASYM_UNSLOTH_GC_NVME="${ASYM_UNSLOTH_GC_NVME:-}"
      ASYM_GEMM_LF_CONFIG_ASYM_NVME_ROLES="${ASYM_NVME_ROLES}"
      ASYM_GEMM_LF_CONFIG_ASYM_NVME_PATH="${ASYM_NVME_PATH:-/scratch_local/asym_nvme}"
      ASYM_GEMM_LF_CONFIG_ASYM_NVME_SYNC="${ASYM_NVME_SYNC:-1}"
      ASYM_GEMM_LF_CONFIG_ASYM_NVME_ACT_CPU_BUDGET_BYTES="${ASYM_NVME_ACT_CPU_BUDGET_BYTES:-auto}"
    )
  fi
  if [[ "${BACKEND}" == kt_* ]]; then
    RUN_ENV+=(
      ASYM_GEMM_LF_CONFIG_KT_BACKEND="${KT_BACKEND_INTERNAL:-}"
      ASYM_GEMM_LF_CONFIG_KT_KERNEL_DIR="${KT_KERNEL_DIR}"
      ASYM_GEMM_LF_CONFIG_KT_REPO_DIR="${KT_REPO_DIR}"
      ASYM_GEMM_LF_CONFIG_KT_GGUF_PY_DIR="${KT_GGUF_PY_DIR}"
      ASYM_GEMM_LF_CONFIG_KT_NUM_THREADS="${KT_NUM_THREADS}"
      ASYM_GEMM_LF_CONFIG_KT_THREADPOOL_COUNT="${KT_THREADPOOL_COUNT}"
      ASYM_GEMM_LF_CONFIG_KT_MAX_CACHE_DEPTH="${KT_MAX_CACHE_DEPTH}"
      ASYM_GEMM_LF_CONFIG_KT_TP_ENABLED="${KT_TP_ENABLED}"
      ASYM_GEMM_LF_CONFIG_KT_SHARE_BACKWARD_BB="${KT_SHARE_BACKWARD_BB}"
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

NUMACTL_CMD=()
if [[ "${NUMACTL_ENABLE}" == "1" ]]; then
  if ! command -v "${NUMACTL_BIN}" >/dev/null 2>&1; then
    echo "NUMACTL_ENABLE=1 but numactl binary not found: ${NUMACTL_BIN}" >&2
    exit 2
  fi
  if [[ "${NUMACTL_MODE}" == "interleave" ]]; then
    NUMACTL_CMD=(
      "${NUMACTL_BIN}"
      "--interleave=${NUMACTL_MEMBIND}"
      "--cpunodebind=${NUMACTL_CPUNODEBIND}"
    )
  else
    NUMACTL_CMD=(
      "${NUMACTL_BIN}"
      "--membind=${NUMACTL_MEMBIND}"
      "--cpunodebind=${NUMACTL_CPUNODEBIND}"
    )
  fi
  echo "NUMACTL_MODE=${NUMACTL_MODE} -> ${NUMACTL_CMD[*]}" >&2
fi

if [[ "${PROFILE}" == "1" ]]; then
  LAUNCH_CMD=("${ENV_PYTHON}" "${PROFILE_LAUNCHER}" "${CMD_ARGS[@]}")
else
  LAUNCH_CMD=("${LF_CLI_BIN}" train "${CMD_ARGS[@]}")
fi

launch_entry="${LF_DIR}/src/train.py"
[[ "${PROFILE}" == "1" ]] && launch_entry="${PROFILE_LAUNCHER}"
if is_torch_run; then
  if [[ "${DIST_LAUNCHER}" == "deepspeed" ]]; then
    if [[ "${NNODES:-1}" != "1" ]]; then
      echo "DIST_LAUNCHER=deepspeed currently supports single-node launches only; got NNODES=${NNODES:-1}" >&2
      exit 2
    fi
    LAUNCH_CMD=(
      "${DEEPSPEED_CMD[@]}"
      --include "localhost:${GPU_ID}"
      --master_addr "${MASTER_ADDR:-127.0.0.1}"
      --master_port "${MASTER_PORT_VALUE}"
      "${launch_entry}"
      "${CMD_ARGS[@]}"
    )
  elif [[ "${DIST_LAUNCHER}" == "accelerate" ]]; then
    ACCELERATE_LAUNCH_ARGS=(
      --num_processes "${NUM_GPUS}"
      --num_machines "${NNODES:-1}"
      --machine_rank "${NODE_RANK:-0}"
      --main_process_ip "${MASTER_ADDR:-127.0.0.1}"
      --main_process_port "${MASTER_PORT_VALUE}"
    )
    if ((NUM_GPUS > 1)); then
      ACCELERATE_LAUNCH_ARGS=(--multi_gpu "${ACCELERATE_LAUNCH_ARGS[@]}")
    fi
    LAUNCH_CMD=(
      "${ACCELERATE_CMD[@]}" launch
      "${ACCELERATE_LAUNCH_ARGS[@]}"
      "${launch_entry}"
      "${CMD_ARGS[@]}"
    )
  elif [[ "${PROFILE}" == "1" ]]; then
    LAUNCH_CMD=(
      "${TORCHRUN_CMD[@]}"
      --nnodes "${NNODES:-1}"
      --node_rank "${NODE_RANK:-0}"
      --nproc_per_node "${NUM_GPUS}"
      --master_addr "${MASTER_ADDR:-127.0.0.1}"
      --master_port "${MASTER_PORT_VALUE}"
      "${PROFILE_LAUNCHER}"
      "${CMD_ARGS[@]}"
    )
  else
    LAUNCH_CMD=(
      "${TORCHRUN_CMD[@]}"
      --nnodes "${NNODES:-1}"
      --node_rank "${NODE_RANK:-0}"
      --nproc_per_node "${NUM_GPUS}"
      --master_addr "${MASTER_ADDR:-127.0.0.1}"
      --master_port "${MASTER_PORT_VALUE}"
      "${LF_DIR}/src/train.py"
      "${CMD_ARGS[@]}"
    )
  fi
fi

# --- NVMe offload I/O sampler -------------------------------------------------
# Records a per-second /sys/block/<dev>/stat timeseries (sectors read/written) for the whole run.
# summarize_nvme_offload.py turns it into offload_io.json so an agent reads the real NVMe offload
# bandwidth without parsing the nsys sqlite. Cheap (1 read/sec), best-effort.
IO_SAMPLER_PID=""
IO_SAMPLES_CSV=""
start_io_sampler() {
  IO_SAMPLER_PID=""
  [[ "${RECORD_IO}" == "1" && "${PROFILE}" == "1" ]] || return 0
  local statf="/sys/block/${NVME_OFFLOAD_IO_DEVICE}/stat"
  [[ -r "${statf}" ]] || { echo "RECORD_IO: ${statf} not readable; skipping io sampler" | tee -a "${LOG_FILE}"; return 0; }
  [[ -n "${PROFILE_OUTPUT_DIR}" ]] || return 0
  mkdir -p "${PROFILE_OUTPUT_DIR}"
  IO_SAMPLES_CSV="${PROFILE_OUTPUT_DIR}/io_samples.csv"
  echo "epoch_s,sectors_read,sectors_written" > "${IO_SAMPLES_CSV}"
  ( while :; do
      read -r _r1 _r2 _sread _r4 _r5 _r6 _swrite _rest < "${statf}" || break
      printf '%s,%s,%s\n' "$(date +%s.%N)" "${_sread}" "${_swrite}" >> "${IO_SAMPLES_CSV}"
      sleep 1
    done ) &
  IO_SAMPLER_PID=$!
  echo "RECORD_IO: sampling ${statf} -> ${IO_SAMPLES_CSV} (pid ${IO_SAMPLER_PID})" | tee -a "${LOG_FILE}"
}
stop_io_sampler() {
  [[ -n "${IO_SAMPLER_PID}" ]] || return 0
  kill "${IO_SAMPLER_PID}" 2>/dev/null || true
  wait "${IO_SAMPLER_PID}" 2>/dev/null || true
  IO_SAMPLER_PID=""
}

TRAIN_STATUS=0
SOURCE_PROFILE_POSTPROCESSED=0
start_io_sampler
if [[ "${PROFILE}" == "1" && "${PROFILE_PROFILER}" == "nsys" ]]; then
  NSYS_CMD=(
    "${NSYS_BIN}"
    profile
    --trace=${NSYS_TRACE}
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
  set +e
  run_logged_command "${ENV_CMD[@]}" "${RUN_ENV[@]}" "${NUMACTL_CMD[@]}" "${NSYS_CMD[@]}" "${LAUNCH_CMD[@]}"
  TRAIN_STATUS=$?
  set -e
else
  set +e
  run_logged_command "${ENV_CMD[@]}" "${RUN_ENV[@]}" "${NUMACTL_CMD[@]}" "${LAUNCH_CMD[@]}"
  TRAIN_STATUS=$?
  set -e
fi
stop_io_sampler

if [[ "${TRAIN_STATUS}" != "0" ]] && source_profile_completion_proven; then
  echo "Training launcher command returned status ${TRAIN_STATUS}, but final source_profile_written heartbeat and ${PROFILE_SOURCE_JSON} exist; accepting the completed source-profile run." | tee -a "${LOG_FILE}"
  TRAIN_STATUS=0
fi

if [[ "${PROFILE}" == "1" && "${PROFILE_PROFILER}" == "source" ]]; then
  POSTPROCESS_STATUS=0
  if [[ "${TRAIN_STATUS}" == "0" ]]; then
    SOURCE_PROFILE_COMPLETION_OUTPUT=""
    if ! SOURCE_PROFILE_COMPLETION_OUTPUT="$(source_profile_completion_proven 2>&1)"; then
      echo "Training completed but ${PROFILE_SOURCE_JSON} does not prove a non-partial source_profile_written completion; refusing to accept it as complete." | tee -a "${LOG_FILE}"
      if [[ -n "${SOURCE_PROFILE_COMPLETION_OUTPUT}" ]]; then
        echo "${SOURCE_PROFILE_COMPLETION_OUTPUT}" | tee -a "${LOG_FILE}"
      fi
      exit 1
    fi
    postprocess_source_profile_if_available || POSTPROCESS_STATUS=$?
  else
    postprocess_source_profile_if_available 1 || POSTPROCESS_STATUS=$?
  fi
  if [[ "${POSTPROCESS_STATUS}" == "0" ]]; then
    SOURCE_PROFILE_POSTPROCESSED=1
    echo "Wrote source profile artifacts under ${PROFILE_OUTPUT_DIR}" | tee -a "${LOG_FILE}"
    if [[ "${TRAIN_STATUS}" == "0" && ! -f "${PROFILE_SOURCE_JSON}" ]]; then
      echo "Training completed but final source profile is missing at ${PROFILE_SOURCE_JSON}; refusing to accept partial profile as complete." | tee -a "${LOG_FILE}"
      exit 1
    fi
    if [[ "${TRAIN_STATUS}" == "0" ]]; then
      check_trainable_surface_if_requested 2>&1 | tee -a "${LOG_FILE}"
    fi
  elif [[ "${TRAIN_STATUS}" == "0" ]]; then
    exit "${POSTPROCESS_STATUS}"
  fi
fi

if [[ "${TRAIN_STATUS}" != "0" ]]; then
  if [[ "${PROFILE}" == "1" && "${PROFILE_PROFILER}" != "source" ]]; then
    postprocess_source_profile_if_available || true
  fi
  append_host_oom_diagnostics "${TRAIN_STATUS}"
  echo "Training command failed with status ${TRAIN_STATUS}" | tee -a "${LOG_FILE}"
  exit "${TRAIN_STATUS}"
fi

if is_superoffload_zero_run && [[ "${CHECK_SUPEROFFLOAD}" == "1" ]]; then
  SUPER_OFFLOAD_CHECK_ARGS=(--train-log "${LOG_FILE}" --require-enabled)
  if [[ -n "${PROFILE_SOURCE_JSON}" && -f "${PROFILE_SOURCE_JSON}" ]]; then
    SUPER_OFFLOAD_CHECK_ARGS=(--profile-json "${PROFILE_SOURCE_JSON}" "${SUPER_OFFLOAD_CHECK_ARGS[@]}")
  fi
  SUPER_OFFLOAD_CHECK_OUTPUT=$("${ENV_PYTHON}" "${CHECK_SUPEROFFLOAD_SCRIPT}" "${SUPER_OFFLOAD_CHECK_ARGS[@]}") || {
      echo "backend=superoffload completed without a SuperOffload runtime marker; inspect ${LOG_FILE}" | tee -a "${LOG_FILE}"
      exit 1
    }
  echo "${SUPER_OFFLOAD_CHECK_OUTPUT}" | tee -a "${LOG_FILE}"
fi

if is_cpuadam_zero_run && [[ "${CHECK_CPUADAM}" == "1" ]]; then
  CPUADAM_CHECK_ARGS=(--train-log "${LOG_FILE}" --require-enabled)
  if [[ -n "${PROFILE_SOURCE_JSON}" && -f "${PROFILE_SOURCE_JSON}" ]]; then
    CPUADAM_CHECK_ARGS=(--profile-json "${PROFILE_SOURCE_JSON}" "${CPUADAM_CHECK_ARGS[@]}")
  fi
  CPUADAM_CHECK_OUTPUT=$("${ENV_PYTHON}" "${CHECK_CPUADAM_SCRIPT}" "${CPUADAM_CHECK_ARGS[@]}") || {
      echo "backend=zero3_cpuadam completed without a DeepSpeedCPUAdam runtime marker; inspect ${LOG_FILE}" | tee -a "${LOG_FILE}"
      exit 1
    }
  echo "${CPUADAM_CHECK_OUTPUT}" | tee -a "${LOG_FILE}"
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
lora = profile.get("lora", {})
optimizer_memory = profile.get("optimizer_memory", {})
def int_value(container, key):
    try:
        return int(container.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0
def optional_int(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
def kt_lora_health_passed(health):
    if not isinstance(health, dict) or not health:
        return False, "missing KT fused LoRA update health"
    rows = health.get("rows", [])
    row_dicts = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    updated_from_rows = sum(
        int(
            bool(row.get("nonzero_grad_changed_after_step"))
            or (bool(row.get("grad_nonzero_before_step")) and bool(row.get("param_changed_after_step")))
        )
        for row in row_dicts
    )
    unchanged_from_rows = sum(
        int(bool(row.get("grad_nonzero_before_step")) and not bool(row.get("param_changed_after_step")))
        for row in row_dicts
    )
    updated_grad_tensors = int_value(health, "updated_grad_tensors") or updated_from_rows
    grad_nonzero_unchanged_tensors = int_value(health, "grad_nonzero_unchanged_tensors") or unchanged_from_rows
    compared_tensors = int_value(health, "compared_tensors") or len(row_dicts)
    grad_nonzero_tensors = int_value(health, "grad_nonzero_tensors")
    sampled_tensors = int_value(health, "sampled_tensors") or len(row_dicts)
    total_fused_tensors = int_value(health, "total_fused_tensors")
    after_sampled_tensors = optional_int(health.get("after_sampled_tensors"))
    after_total_fused_tensors = optional_int(health.get("after_total_fused_tensors"))
    missing_after_tensors = int_value(health, "missing_after_tensors")
    unexpected_after_tensors = int_value(health, "unexpected_after_tensors")
    exhaustive = bool(health.get("exhaustive", total_fused_tensors > 0 and sampled_tensors == total_fused_tensors))
    input_passed = health.get("passed") if "passed" in health else None
    if health.get("available") is not True:
        return False, "KT fused LoRA update health unavailable"
    missing_snapshot_fields = [
        key
        for key in (
            "after_sampled_tensors",
            "after_total_fused_tensors",
            "missing_after_tensors",
            "unexpected_after_tensors",
        )
        if key not in health or health.get(key) in (None, "")
    ]
    if missing_snapshot_fields:
        return False, "KT fused LoRA update health missing after-step snapshot fields: " + ", ".join(missing_snapshot_fields)
    if after_sampled_tensors is None or after_total_fused_tensors is None:
        return False, "KT fused LoRA update health has invalid after-step snapshot counts"
    if missing_after_tensors > 0 or unexpected_after_tensors > 0:
        return False, "sampled fused LoRA tensor set changed between before/after optimizer snapshots"
    if sampled_tensors != after_sampled_tensors or total_fused_tensors != after_total_fused_tensors:
        return False, "fused LoRA tensor counts changed between before/after optimizer snapshots"
    if exhaustive and compared_tensors != sampled_tensors:
        return False, f"exhaustive fused LoRA health compared {compared_tensors} of {sampled_tensors} tensors"
    if compared_tensors <= 0:
        return False, "no sampled fused LoRA tensors were comparable"
    if grad_nonzero_tensors <= 0:
        return False, "no sampled fused LoRA tensors had nonzero gradients before optimizer step"
    if grad_nonzero_unchanged_tensors > 0:
        return False, "one or more sampled nonzero-gradient fused LoRA tensors did not change after optimizer step"
    if updated_grad_tensors <= 0:
        return False, "no sampled nonzero-gradient fused LoRA tensors changed after optimizer step"
    if input_passed is False:
        return False, str(health.get("reason") or "input KT fused LoRA update health failed")
    return True, "all sampled nonzero-gradient fused LoRA tensors changed after optimizer step"
def optimizer_process_memory_passed(optimizer_memory):
    if not isinstance(optimizer_memory, dict) or not optimizer_memory:
        return False, "missing optimizer_memory"
    for key in ("process_memory_at_start", "process_memory_before_step", "process_memory_after_step"):
        snapshot = optimizer_memory.get(key)
        if not isinstance(snapshot, dict) or not snapshot:
            return False, f"missing {key}"
        if optional_int(snapshot.get("rss_bytes")) is None:
            return False, f"{key}.rss_bytes missing or invalid"
    for key in ("process_rss_pre_step_overhead_delta_bytes", "process_rss_delta_bytes"):
        if key not in optimizer_memory or optional_int(optimizer_memory.get(key)) is None:
            return False, f"{key} missing or invalid"
    return True, "optimizer process RSS snapshots are present"
wrappers = int(kt.get("wrapper_count", 0) or 0)
fw = int(kt.get("total_forward_calls", 0) or 0)
bw = int(kt.get("total_backward_calls", 0) or 0)
methods = sorted({str(row.get("method", "")) for row in kt.get("rows", []) if isinstance(row, dict)})
fused_lora_params = int(lora.get("kt_fused_expert_lora_parameters", 0) or 0) if isinstance(lora, dict) else 0
if wrappers <= 0:
    raise SystemExit(f"Expected positive KT wrapper_count in {sys.argv[1]}, got {wrappers}")
if fw <= 0 or bw <= 0:
    raise SystemExit(f"Expected positive KT forward/backward calls in {sys.argv[1]}, got fw={fw} bw={bw}")
if fused_lora_params > 0:
    health = {}
    if isinstance(optimizer_memory, dict):
        health = optimizer_memory.get("kt_lora_update_health", {}) or {}
    if not isinstance(health, dict):
        raise SystemExit(
            f"KT fused expert LoRA params are present ({fused_lora_params}) but optimizer update health is missing"
        )
    memory_ok, memory_reason = optimizer_process_memory_passed(optimizer_memory)
    if not memory_ok:
        raise SystemExit(f"KT fused expert LoRA optimizer process memory evidence missing: {memory_reason}")
    health_ok, health_reason = kt_lora_health_passed(health)
    if not health_ok:
        raise SystemExit(f"KT fused expert LoRA optimizer update health failed: {health_reason}; {health}")
print(
    f"Verified KT source counters: backend={kt_backend} wrappers={wrappers} fw={fw} bw={bw} "
    f"methods={methods} fused_lora_params={fused_lora_params}"
)
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

if [[ "${PROFILE}" == "1" && -f "${PROFILE_SOURCE_JSON}" ]]; then
  echo "Wrote LF source profile to ${PROFILE_SOURCE_JSON}" | tee -a "${LOG_FILE}"
fi

if [[ "${PROFILE}" == "1" && "${PROFILE_PROFILER}" == "source" && "${SOURCE_PROFILE_POSTPROCESSED}" != "1" ]]; then
  postprocess_source_profile_if_available
  echo "Wrote source profile artifacts under ${PROFILE_OUTPUT_DIR}" | tee -a "${LOG_FILE}"
  check_trainable_surface_if_requested 2>&1 | tee -a "${LOG_FILE}"
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
  # Compact, agent-readable CSV summaries so downstream agents skip the multi-100MB sqlite.
  "${NSYS_BIN}" stats --format csv \
    --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
    --report cuda_gpu_mem_size_sum --report osrt_sum \
    --output "${PROFILE_OUTPUT_DIR}/nsys_stats" \
    "${PROFILE_NSYS_SQLITE}" 2>&1 | tee -a "${LOG_FILE}" \
    || echo "nsys stats failed (non-fatal)" | tee -a "${LOG_FILE}"
  if [[ ! -f "${PROFILE_NSYS_POSTPROCESS_SCRIPT}" ]]; then
    echo "Missing Nsight postprocess script ${PROFILE_NSYS_POSTPROCESS_SCRIPT}" >&2
    exit 2
  fi
  "${ENV_PYTHON}" "${PROFILE_NSYS_POSTPROCESS_SCRIPT}" \
    "${PROFILE_NSYS_SQLITE}" \
    --source-profile-json "${PROFILE_SOURCE_JSON}" \
    --output-json "${PROFILE_JSON}" \
    --output-md "${PROFILE_SUMMARY_MD}" 2>&1 | tee -a "${LOG_FILE}"
  "${ENV_PYTHON}" "${PROFILE_POSTPROCESS_SCRIPT}" \
    --profile-json "${PROFILE_JSON}" \
    --output-dir "${PROFILE_OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"
  # NVMe-offload cost summary (NVTX swap-in wait + GPU compute/H2D + md0 bandwidth) -> flat JSON.
  if [[ -f "${OFFLOAD_SUMMARY_SCRIPT}" ]]; then
    "${ENV_PYTHON}" "${OFFLOAD_SUMMARY_SCRIPT}" \
      "${PROFILE_NSYS_SQLITE}" \
      --io-samples "${IO_SAMPLES_CSV:-}" \
      --profile-json "${PROFILE_JSON}" \
      --output-json "${PROFILE_OUTPUT_DIR}/offload_io.json" 2>&1 | tee -a "${LOG_FILE}" \
      || echo "offload summary failed (non-fatal)" | tee -a "${LOG_FILE}"
  fi
  check_trainable_surface_if_requested 2>&1 | tee -a "${LOG_FILE}"
  echo "Wrote Nsight profile artifacts to ${PROFILE_JSON} and ${PROFILE_SUMMARY_MD}" | tee -a "${LOG_FILE}"
fi
