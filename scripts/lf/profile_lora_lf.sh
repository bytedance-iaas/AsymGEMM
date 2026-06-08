#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# User Parameters
# =============================================================================

# Paths and tools
ROOT=${ROOT:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM}
LF_DIR=${LF_DIR:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory}
KT_KERNEL_DIR=${KT_KERNEL_DIR:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/ktransformers/kt-kernel}
DEEPSPEED_DIR=${DEEPSPEED_DIR:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/deepspeed}
CONDA_EXE=${CONDA_EXE:-conda}
NSYS_BIN=${NSYS_BIN:-nsys}
DIST_LAUNCHER=${DIST_LAUNCHER:-torchrun}

# Sweep axes
GPU_POOL=${GPU_POOL:-3}
# MODEL_SPECS entries are model|num_gpus. Recompute belongs only in BACKEND_SPECS.
MODEL_SPECS=${MODEL_SPECS:-"Qwen/Qwen3-30B-A3B|1"}
# EXPERT_POLICIES=${EXPERT_POLICIES-"none,tok-le0,tok-le512,tok-le512-act"}
EXPERT_POLICIES=${EXPERT_POLICIES-"none"}
# Primary backend sweep axis. Each entry is backend|recompute.  Torch is plain
# distributed launch. Zero backends add the matching DeepSpeed config.
# BACKEND_SPECS=${BACKEND_SPECS:-"zero2|norecomp,zero2|recomp,zero3_offload|norecomp,zero3_offload|recomp"}
BACKEND_SPECS=${BACKEND_SPECS:-"zero3_offload|recomp"}
ROUTER_MODES=${ROUTER_MODES:-whole}
# PROFILERS=${PROFILERS:-nsys,source}
PROFILERS=${PROFILERS:-nsys}
PRECISION=${PRECISION:-bf16}
SEQ_LENS=${SEQ_LENS:-4096}
# LORA_DROPOUT=${LORA_DROPOUT:-0.00,0.10}
LORA_DROPOUT=${LORA_DROPOUT:-0.00}

# Dataset
DATASET=${DATASET:-asym_long_sft_smoke}
PREPARE_DATASETS=${PREPARE_DATASETS:-true}
DATASET_MIN_TOKENS=${DATASET_MIN_TOKENS:-auto}
DATASET_EVAL_ROWS=${DATASET_EVAL_ROWS:-128}
DATASET_OVERWRITE=${DATASET_OVERWRITE:-false}
TEMPLATE=${TEMPLATE:-auto}
MAX_SAMPLES=${MAX_SAMPLES:-128}

# Training
MAX_STEPS=${MAX_STEPS:-10}
WARMUP_STEPS=${WARMUP_STEPS:-5}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-16}
SEED=${SEED:-42}

# Backend checks and AsymGEMM options
ASYM_OFFLOAD_MODULES=${ASYM_OFFLOAD_MODULES:-routed_experts}
ASYM_STRICT=${ASYM_STRICT:-true}
REQUIRE_SM100=${REQUIRE_SM100:-1}

# Output and profiling
OUTPUT_ROOT=${OUTPUT_ROOT:-}
PROFILE_LEVEL=${PROFILE_LEVEL:-op}
PROFILE_LAYERS=${PROFILE_LAYERS:-all}
PROFILE_MEMORY_ATTRIBUTION=${PROFILE_MEMORY_ATTRIBUTION:-auto}
PROFILE_MEMORY_BREAKDOWN=${PROFILE_MEMORY_BREAKDOWN:-auto}
PROFILE_MEMORY_BREAKDOWN_INTERVAL=${PROFILE_MEMORY_BREAKDOWN_INTERVAL:-1}
PROFILE_MEMORY_BREAKDOWN_STEPS=${PROFILE_MEMORY_BREAKDOWN_STEPS:-}
PROFILE_MEMORY_BREAKDOWN_MODULES=${PROFILE_MEMORY_BREAKDOWN_MODULES:-attention,router,mlp,experts,lora,embedding,loss}
PROFILE_SYNC=${PROFILE_SYNC:-0}
PROFILE_MODULE_FILTER=${PROFILE_MODULE_FILTER:-attention,router,mlp,experts,lora,optimizer,kt}

# KT backend
KT_NUM_THREADS=${KT_NUM_THREADS:-}
KT_THREADPOOL_COUNT=${KT_THREADPOOL_COUNT:-}
KT_MAX_CACHE_DEPTH=${KT_MAX_CACHE_DEPTH:-2}
KT_TP_ENABLED=${KT_TP_ENABLED:-false}
KT_TORCHBF16_SFT_DEVICE=${KT_TORCHBF16_SFT_DEVICE:-cuda}
KT_ARM_OMP_NUM_THREADS=${KT_ARM_OMP_NUM_THREADS:-64}
KT_ARM_OMP_PROC_BIND=${KT_ARM_OMP_PROC_BIND:-close}
KT_ARM_OMP_PLACES=${KT_ARM_OMP_PLACES:-cores}
KT_SHARE_BACKWARD_BB=${KT_SHARE_BACKWARD_BB:-}
KT_SHARE_CACHE_POOL=${KT_SHARE_CACHE_POOL:-}
KT_NUM_GPU_EXPERTS=${KT_NUM_GPU_EXPERTS:-}
KT_WEIGHT_PATH=${KT_WEIGHT_PATH:-}
KT_EXPERT_CHECKPOINT_PATH=${KT_EXPERT_CHECKPOINT_PATH:-}
KT_USE_LORA_EXPERTS=${KT_USE_LORA_EXPERTS:-}
KT_LORA_EXPERT_NUM=${KT_LORA_EXPERT_NUM:-}
KT_LORA_EXPERT_INTERMEDIATE_SIZE=${KT_LORA_EXPERT_INTERMEDIATE_SIZE:-}
CHECK_KT_CALLS=${CHECK_KT_CALLS:-1}

# SuperOffload backend
CHECK_SUPEROFFLOAD=${CHECK_SUPEROFFLOAD:-1}

# Plotting
PLOT=${PLOT:-true}
PLOT_MEMORY_BREAKDOWN=${PLOT_MEMORY_BREAKDOWN:-true}
MEMORY_BREAKDOWN_PLOT_Y_SCALE=${MEMORY_BREAKDOWN_PLOT_Y_SCALE:-shared}
PLOT_OUTPUT_DIR=${PLOT_OUTPUT_DIR:-}

# Execution
OVERWRITE=${OVERWRITE:-false}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-true}
DRY_RUN=${DRY_RUN:-false}
COLLECT_EXISTING=${COLLECT_EXISTING:-false}
INTERRUPT_GRACE_SECONDS=${INTERRUPT_GRACE_SECONDS:-2}
RUN_NAME=${RUN_NAME:-}

# =============================================================================
# Derived Parameters
# =============================================================================
ASYM_DIR=${ASYM_DIR:-${ROOT}}
KT_TOOLS_DIR=${KT_TOOLS_DIR:-${ASYM_DIR}}
KT_REPO_DIR_ENV_SET=${KT_REPO_DIR+x}
KT_REPO_DIR=${KT_REPO_DIR:-$(dirname "${KT_KERNEL_DIR}")}
ENV_DIR=${ENV_DIR:-${LF_DIR}/.venv}
ENV_PYTHON=${ENV_PYTHON:-${ENV_DIR}/bin/python}
RUN_LF_SCRIPT="${ASYM_DIR}/scripts/lf/run_lf_lora_sft.sh"
BUILD_DATASET_SCRIPT="${ASYM_DIR}/scripts/lf/build_lf_sft_eval_pair.py"
PROFILE_POSTPROCESS_SCRIPT="${ASYM_DIR}/scripts/lf/postprocess_lf_profile_artifacts.py"
PLOT_SCRIPT="${ASYM_DIR}/scripts/plotting/plot_activation_recompute_sweep.py"
MEMORY_PLOT_SCRIPT="${ASYM_DIR}/scripts/plotting/plot_lf_memory_breakdown.py"
INTERCONNECT_PLOT_SCRIPT="${ASYM_DIR}/scripts/plotting/plot_lf_interconnect_ctc.py"

# =============================================================================
# Main Logic
# =============================================================================
usage() {
  cat <<USAGE
Usage:
  scripts/lf/profile_lora_lf.sh [options]

Defaults:
  --gpus ${GPU_POOL}
  --dist-launcher ${DIST_LAUNCHER}
  --backend-specs ${BACKEND_SPECS}
  --profilers ${PROFILERS}
  --seq-lens ${SEQ_LENS}
  --output-root ${OUTPUT_ROOT}

Options:
  List values must be comma-separated with no spaces.

  Sweep:
  --gpus LIST                    Physical GPU pool, e.g. 0,1.
  --dist-launcher torchrun|accelerate
                                 Launcher for torch/zero/SuperOffload jobs. Default ${DIST_LAUNCHER}.
  --models LIST                  Model specs. Each item is model_name_or_path|num_gpus.
                                 Example: meta-llama/Llama-4-Scout-17B-16E|1,meta-llama/Llama-4-Maverick-17B-128E|4
  --backend-specs LIST           Backend/recompute specs, e.g. 'torch|recomp,asym|norecomp,zero3_offload|recomp'.
                                 Use canonical recompute labels: norecomp or recomp. Use both to expand to both modes.
  --router-modes LIST            AsymGEMM router modes: hf, whole. Default ${ROUTER_MODES}.
  --profilers LIST               source and/or nsys.
  --seq-lens LIST                LF cutoff lengths. Accepts positive integers, e.g. 4096,8192.
  --expert-policies LIST         AsymGEMM expert policies: none, tok-le0, tok-le0-act, tok-leN, tok-geN, tokA-B, and -act variants.

  Dataset:
  --dataset NAME
  --prepare-datasets true|false  Build/audit model+length-specific LF datasets before training.
  --dataset-min-tokens N|auto    Minimum source tokens for generated/audited rows. auto uses the seq length.
  --dataset-eval-rows N
  --dataset-overwrite true|false Rewrite existing generated dataset files.
  --template NAME
  --max-samples N

  Training:
  --max-steps N                  Measured steps kept in plots/summaries.
  --warmup-steps N               Extra initial steps to run but exclude from plots/summaries.
  --batch-size N
  --gradient-accumulation-steps N
  --learning-rate VALUE
  --lora-rank N
  --lora-alpha VALUE
  --lora-dropout LIST           LoRA dropout probabilities in fixed 0.xx format, e.g. 0.00,0.10.
                                 KT supports nonzero dropout for validated kt_torchbf16 and kt_armbf16 SFT backends.
  --seed N
  --precision NAME

  Profiling:
  --profile-level stage|module|op|deep
  --profile-layers all|first,last|0,1,2|every4
  --profile-memory-attribution auto|true|false
  --profile-memory-breakdown auto|true|false
  --profile-memory-breakdown-interval N
  --profile-memory-breakdown-steps LIST
  --profile-memory-breakdown-modules LIST
  --profile-sync true|false
  --profile-module-filter LIST

  KT:
  --kt-kernel-dir DIR            Integrated kt-kernel source tree.
  --kt-tools-dir DIR             Helper source tree to put on PYTHONPATH for KT jobs. Defaults to ROOT.
  --kt-repo-dir DIR              KT artifact root owner. Defaults to dirname(--kt-kernel-dir).
  --kt-num-threads N
  --kt-threadpool-count N
  --kt-max-cache-depth N
  --kt-tp-enabled true|false
  --kt-torchbf16-sft-device DEV  cpu, cuda, cuda:N, or another torch device string.
  --kt-arm-omp-num-threads N
  --kt-arm-omp-proc-bind VALUE
  --kt-arm-omp-places VALUE
  --kt-share-backward-bb true|false
  --kt-share-cache-pool true|false
  --kt-num-gpu-experts N
  --kt-weight-path PATH
  --kt-expert-checkpoint-path PATH
  --kt-use-lora-experts true|false
  --kt-lora-expert-num N
  --kt-lora-expert-intermediate-size N
  --check-kt-calls true|false

  SuperOffload:
  --deepspeed-dir DIR            Local DeepSpeed/SuperOffload source tree to put before site-packages.
  --check-superoffload true|false

  Outputs and plotting:
  --output-root DIR              Default config layout: <root>/<dataset>__lora__lf__<precision>/<model>__gpus<model_gpus>__b<batch>_s<seq>_w<warmup>_s<steps>_r<rank>_a<alpha>_drop0xx
                                 Per-run dirs add <backend>__<profiler>__<recompute>__pol<policy>__router<mode>/s<seq>.
  --run-name NAME                Optional config directory under <dataset>__lora__lf__<precision>.
  --plot true|false
  --plot-memory-breakdown true|false
  --memory-breakdown-plot-y-scale shared|per-plot|global
  --plot-output-dir DIR

  Execution:
  --overwrite true|false
  --continue-on-error true|false
  --collect-existing             Skip training and regenerate plots from existing profile.json files.
  --dry-run                      Print commands without running training or plotting.
  -h, --help
USAGE
}

die() {
  echo "error: $*" >&2
  exit 2
}

need_value() {
  local opt="$1"
  local value="${2-}"
  [[ -n "${value}" && "${value}" != --* ]] || die "${opt} requires a value"
}

collect_values() {
  local opt="$1"
  local -n out_ref="$2"
  shift 2
  out_ref=()
  while (($#)); do
    [[ "$1" == --* ]] && break
    out_ref+=("$1")
    shift
  done
  ((${#out_ref[@]} > 0)) || die "${opt} requires at least one value"
  REMAINING=("$@")
}

tokens() (
  set -f
  local value part
  local -a parts
  for value in "$@"; do
    read -r -a parts <<< "${value//,/ }"
    for part in "${parts[@]}"; do
      [[ -n "${part}" ]] && printf '%s\n' "${part}"
    done
  done
)

require_comma_list() {
  local name="$1"
  local value="$2"
  if [[ -n "${value}" && "${value}" =~ [[:space:]] ]]; then
    die "${name} must be comma-separated with no spaces, got '${value}'"
  fi
}

dedupe() {
  awk '!seen[$0]++'
}

bool_value() {
  case "${1,,}" in
    1|true|yes|y|on) printf 'true\n' ;;
    0|false|no|n|off) printf 'false\n' ;;
    *) die "expected true or false, got '${1}'" ;;
  esac
}

optional_bool_value() {
  [[ -z "$1" ]] && return 0
  bool_value "$1"
}

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "${ROOT}/$1" ;;
  esac
}

safe_label() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]_-' '_' | sed -e 's/^[_-]*//' -e 's/[_-]*$//'
}

lora_dropout_label() {
  local value="$1"
  [[ "${value}" =~ ^0\.[0-9][0-9]$ ]] || die "LORA_DROPOUT must use fixed 0.xx format, e.g. 0.10; got '${value}'"
  printf 'drop0%s\n' "${value#*.}"
}

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

nonnegative_int() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer, got '${value}'"
}

positive_int() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer, got '${value}'"
}

parse_model_spec() {
  local spec="$1"
  local -a fields

  IFS='|' read -r -a fields <<< "${spec}"
  ((${#fields[@]} <= 2)) || die "model spec must be model|num_gpus, got '${spec}'"

  parsed_model_name="${fields[0]}"
  if ((${#fields[@]} >= 2)); then
    parsed_model_gpu_count="${fields[1]}"
  else
    parsed_model_gpu_count=1
  fi

  [[ -n "${parsed_model_name}" ]] || die "empty model name in model spec '${spec}'"
  [[ -n "${parsed_model_gpu_count}" ]] || die "empty model GPU count in model spec '${spec}'"
  positive_int "model GPU count for ${parsed_model_name}" "${parsed_model_gpu_count}"
}

gpu_slice() {
  local count="$1"
  local -a selected=()
  local index
  ((count <= ${#gpus[@]})) || die "model requests ${count} GPU(s), but GPU pool only has ${#gpus[@]}: ${gpus[*]}"
  for ((index = 0; index < count; index++)); do
    selected+=("${gpus[${index}]}")
  done
  local IFS=,
  printf '%s\n' "${selected[*]}"
}

backend_gpu_count() {
  local backend="$1"
  local model_gpu_count="$2"
  case "${backend}" in
    asym|asym_torch) printf '1\n' ;;
    torch|zero2|zero3|zero3_offload|superoffload) printf '%s\n' "${model_gpu_count}" ;;
    kt_torchbf16|kt_armbf16) printf '1\n' ;;
    *) die "internal backend label must be torch, asym, asym_torch, zero2, zero3, zero3_offload, superoffload, kt_torchbf16, or kt_armbf16, got '${backend}'" ;;
  esac
}

zero_deepspeed_config() {
  case "${1}" in
    zero2) printf '%s\n' "${LF_DIR}/examples/deepspeed/ds_z2_config.json" ;;
    zero3) printf '%s\n' "${LF_DIR}/examples/deepspeed/ds_z3_config.json" ;;
    zero3_offload) printf '%s\n' "${LF_DIR}/examples/deepspeed/ds_z3_offload_config.json" ;;
    superoffload) printf '%s\n' "${LF_DIR}/examples/deepspeed/ds_z3_superoffload_config.json" ;;
    *) return 1 ;;
  esac
}

is_zero_backend() {
  case "${1}" in
    zero2|zero3|zero3_offload|superoffload) return 0 ;;
    *) return 1 ;;
  esac
}

is_policy_independent_backend() {
  case "${1}" in
    torch|zero2|zero3|zero3_offload|superoffload|kt_*) return 0 ;;
    *) return 1 ;;
  esac
}

recompute_label() {
  case "${1,,}" in
    norecomp|recomp) printf '%s\n' "${1,,}" ;;
    norecompute|no_recompute|no-recompute) printf 'norecomp\n' ;;
    recompute) printf 'recomp\n' ;;
    *) die "expected recompute mode norecomp/recomp or norecompute/recompute; got '${1}'" ;;
  esac
}

normalize_expert_policy() {
  local raw="$1"
  case "${raw}" in
    none|tok-le0|tok-le0-act)
      printf '%s\n' "${raw}"
      return
      ;;
  esac
  if [[ "${raw}" =~ ^tok-le[1-9][0-9]*(-act)?$ || "${raw}" =~ ^tok-ge[1-9][0-9]*(-act)?$ || "${raw}" =~ ^tok[1-9][0-9]*-[1-9][0-9]*(-act)?$ ]]; then
    printf '%s\n' "${raw}"
    return
  fi
  die "invalid expert policy '${1}'; expected none, tok-le0, tok-le0-act, tok-leN, tok-geN, tokA-B, or -act variants"
}

backend_label() {
  case "${1,,}" in
    torch) printf 'torch\n' ;;
    asym) printf 'asym\n' ;;
    asym_torch) printf 'asym_torch\n' ;;
    zero2) printf 'zero2\n' ;;
    zero3) printf 'zero3\n' ;;
    zero3_offload) printf 'zero3_offload\n' ;;
    superoffload) printf 'superoffload\n' ;;
    kt_torchbf16) printf 'kt_torchbf16\n' ;;
    kt_armbf16) printf 'kt_armbf16\n' ;;
    *) die "backend must be torch, asym, asym_torch, zero2, zero3, zero3_offload, superoffload, kt_torchbf16, or kt_armbf16, got '${1}'" ;;
  esac
}

router_mode_label() {
  case "${1,,}" in
    hf|whole) printf '%s\n' "${1,,}" ;;
    *) die "router mode must be hf or whole, got '${1}'" ;;
  esac
}

append_backend_spec() {
  local raw="$1"
  local backend_part recompute_part backend recompute_token recompute_mode
  local -a recompute_tokens

  [[ "${raw}" == *"|"* ]] || die "backend spec must be backend|recompute, got '${raw}'"
  backend_part="${raw%%|*}"
  recompute_part="${raw#*|}"

  [[ -n "${backend_part}" ]] || die "empty backend in backend spec '${raw}'"
  [[ -n "${recompute_part}" ]] || die "empty recompute mode in backend spec '${raw}'"
  case "${backend_part,,}" in
    torch) backend=torch ;;
    asym) backend=asym ;;
    asym_torch) backend=asym_torch ;;
    zero2) backend=zero2 ;;
    zero3) backend=zero3 ;;
    zero3_offload) backend=zero3_offload ;;
    superoffload) backend=superoffload ;;
    kt_torchbf16) backend=kt_torchbf16 ;;
    kt_armbf16) backend=kt_armbf16 ;;
    *) die "backend must be torch, asym, asym_torch, zero2, zero3, zero3_offload, superoffload, kt_torchbf16, or kt_armbf16, got '${backend_part}'" ;;
  esac

  mapfile -t recompute_tokens < <(tokens "${recompute_part}")
  ((${#recompute_tokens[@]} > 0)) || die "empty recompute mode in backend spec '${raw}'"
  for recompute_token in "${recompute_tokens[@]}"; do
    if [[ "${recompute_token,,}" == "both" ]]; then
      backend_specs_raw+=("${backend}|norecomp" "${backend}|recomp")
      continue
    fi
    recompute_mode="$(recompute_label "${recompute_token}")"
    backend_specs_raw+=("${backend}|${recompute_mode}")
  done
}

profiler_label() {
  case "${1,,}" in
    source|nsys) printf '%s\n' "${1,,}" ;;
    *) die "profiler must be source or nsys, got '${1}'" ;;
  esac
}

dist_launcher_label() {
  case "${1,,}" in
    torchrun) printf 'torchrun\n' ;;
    accelerate|accelerate_launch) printf 'accelerate\n' ;;
    *) die "dist launcher must be torchrun or accelerate, got '${1}'" ;;
  esac
}

profile_memory_flag_for_profiler() {
  local option="$1"
  local value="$2"
  local profiler="$3"
  case "${value,,}" in
    auto)
      if [[ "${profiler}" == "source" ]]; then
        printf 'true\n'
      else
        printf 'false\n'
      fi
      ;;
    1|true|yes|y|on) printf 'true\n' ;;
    0|false|no|n|off) printf 'false\n' ;;
    *) die "${option} must be auto, true, or false; got '${value}'" ;;
  esac
}

config_root_path() {
  local seq_len="$1"
  local config_label step_label dropout_label
  step_label="w${WARMUP_STEPS}_s${MAX_STEPS}"
  dropout_label="${lora_dropout_label_value}"
  if [[ -n "${run_name}" ]]; then
    if ((${#seq_lens[@]} > 1)); then
      config_label="$(safe_label "${run_name}__s${seq_len}_${dropout_label}")"
    else
      config_label="$(safe_label "${run_name}__${dropout_label}")"
    fi
  else
    config_label="$(safe_label "${workload_label}__gpus${current_model_gpu_count}__b${batch_size}_s${seq_len}_${step_label}_r${LORA_RANK}_a${LORA_ALPHA}_${dropout_label}")"
  fi
  printf '%s/%s\n' "${precision_root}" "${config_label}"
}

job_root_path() {
  local config_root="$1"
  local backend="$2"
  local profiler="$3"
  local recompute="$4"
  local expert_policy="$5"
  local router_mode="$6"
  printf '%s/%s\n' "${config_root}" "$(safe_label "${backend}__${profiler}__${recompute}__pol${expert_policy}__router${router_mode}")"
}

plot_workload_from_config_root() {
  local config_name workload
  config_name="$(basename "$1")"
  workload="${config_name%%__*}"
  if [[ "${config_name}" =~ (^|__)gpus([1-9][0-9]*)(__|$) ]]; then
    printf '%s gpus%s\n' "${workload}" "${BASH_REMATCH[2]}"
  else
    printf '%s\n' "${workload}"
  fi
}

plot_workload_base_from_config_root() {
  local config_name
  config_name="$(basename "$1")"
  printf '%s\n' "${config_name%%__*}"
}

print_command() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

find_free_port() {
  python3 - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

ensure_jobs_tsv() {
  local config_root="$1"
  mkdir -p "${config_root}"
  if [[ ! -e "${config_root}/jobs.tsv" ]]; then
    printf 'status\tgpu\tseq_len\trecompute\texpert_policy\trouter_mode\tbackend\tprofiler\tjob_dir\tprofile_json\tlog\n' > "${config_root}/jobs.tsv"
  fi
}

append_job_record() {
  local config_root="$1"
  local status="$2"
  shift 2
  ensure_jobs_tsv "${config_root}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${status}" "$@" >> "${config_root}/jobs.tsv"
}

plot_cmd_base() {
  local -n _cmd_ref="$1"
  local input_root="$2"
  local output_dir="$3"
  local combined_output_dir="$4"
  shift 4
  (($# > 0)) || die "plot_cmd_base requires at least one sequence length"
  _cmd_ref=(
    "${CONDA_EXE}" run -p "${ENV_DIR}" python "${PLOT_SCRIPT}"
    --input-root "${input_root}"
    --output-dir "${output_dir}"
    --combined-output-dir "${combined_output_dir}"
    --precision "${PRECISION}"
    --clean-output
    --batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --seq-lens "$@"
  )
}

memory_combined_plot_cmd_base() {
  local -n _cmd_ref="$1"
  local input_root="$2"
  local output_dir="$3"
  shift 3
  (($# > 0)) || die "memory_combined_plot_cmd_base requires at least one sequence length"
  _cmd_ref=(
    "${CONDA_EXE}" run -p "${ENV_DIR}" python "${MEMORY_PLOT_SCRIPT}"
    --input-root "${input_root}"
    --output-dir "${output_dir}"
    --clean-output
    --combined-only
    --y-scale "${MEMORY_BREAKDOWN_PLOT_Y_SCALE}"
    --seq-lens "$@"
    --expert-recompute-policies "${expert_policies[@]}"
  )
}

interconnect_combined_plot_cmd_base() {
  local -n _cmd_ref="$1"
  local input_root="$2"
  local output_dir="$3"
  shift 3
  (($# > 0)) || die "interconnect_combined_plot_cmd_base requires at least one sequence length"
  _cmd_ref=(
    "${CONDA_EXE}" run -p "${ENV_DIR}" python "${INTERCONNECT_PLOT_SCRIPT}"
    --input-root "${input_root}"
    --output-dir "${output_dir}"
    --clean-output
    --combined-only
    --seq-lens "$@"
    --expert-recompute-policies "${expert_policies[@]}"
  )
}

append_backend_filters() {
  local -n _cmd_ref="$1"
  local backend
  for backend in "${backends[@]}"; do _cmd_ref+=(--backend "${backend}"); done
}

append_plot_profiler_filters() {
  local -n _cmd_ref="$1"
  local profiler
  for profiler in "${plot_profilers[@]}"; do _cmd_ref+=(--profiler "${profiler}"); done
}

append_recompute_filters() {
  local -n _cmd_ref="$1"
  local recompute
  for recompute in "${recompute_modes[@]}"; do _cmd_ref+=(--recompute "${recompute}"); done
}

append_router_mode_filters() {
  local -n _cmd_ref="$1"
  local router_mode
  for router_mode in "${plot_router_modes[@]}"; do _cmd_ref+=(--router-mode "${router_mode}"); done
}

append_fixed_profiler_filter() {
  local -n _cmd_ref="$1"
  local profiler="$2"
  _cmd_ref+=(--profiler "${profiler}")
}

append_sweep_plot_filters() {
  append_backend_filters "$1"
  append_plot_profiler_filters "$1"
  append_recompute_filters "$1"
  append_router_mode_filters "$1"
}

memory_plot_filters() {
  append_backend_filters "$1"
  append_fixed_profiler_filter "$1" source
  append_router_mode_filters "$1"
}

interconnect_plot_filters() {
  append_backend_filters "$1"
  append_fixed_profiler_filter "$1" nsys
  append_recompute_filters "$1"
  append_router_mode_filters "$1"
}

profiler_selected_for_plots() {
  local profiler="$1"
  local selected
  for selected in "${plot_profilers[@]}"; do
    [[ "${selected}" == "${profiler}" ]] && return 0
  done
  return 1
}

dataset_name_for_seq() {
  local seq_len="$1"
  safe_label "${DATASET}__${workload_label}__s${seq_len}"
}

dataset_min_tokens_for_seq() {
  local seq_len="$1"
  if [[ "${DATASET_MIN_TOKENS}" == "auto" ]]; then
    printf '%s\n' "${seq_len}"
  else
    printf '%s\n' "${DATASET_MIN_TOKENS}"
  fi
}

prepare_dataset_for_seq() {
  local seq_len="$1"
  local dataset_name="$2"
  local min_tokens
  min_tokens="$(dataset_min_tokens_for_seq "${seq_len}")"

  local -a tools_dir_arg=(--asym-dir "${ASYM_DIR}")
  if [[ "${selected_has_kt:-false}" == "true" && "${selected_has_asym:-false}" != "true" ]]; then
    tools_dir_arg=(--asym-dir "${KT_TOOLS_DIR}")
  fi

  local -a dataset_cmd=(
    "${ENV_PYTHON}" "${BUILD_DATASET_SCRIPT}"
    --lf-dir "${LF_DIR}"
    "${tools_dir_arg[@]}"
    --model-name-or-path "${current_model_name}"
    --template "${TEMPLATE}"
    --train-name "${dataset_name}"
    --eval-name "${dataset_name}__eval"
    --train-rows "${MAX_SAMPLES}"
    --eval-rows "${DATASET_EVAL_ROWS}"
    --cutoff-len "${seq_len}"
    --min-tokens "${min_tokens}"
    --precision "${PRECISION}"
    --batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --lora-rank "${LORA_RANK}"
    --lora-alpha "${LORA_ALPHA}"
    --skip-lf-preprocess-check
  )
  if [[ "${DATASET_OVERWRITE}" == "true" ]]; then
    dataset_cmd+=(--overwrite)
  fi

  echo "Preparing LF dataset=${dataset_name} model=${current_model_name} seq=${seq_len} min_tokens=${min_tokens}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    print_command "${dataset_cmd[@]}"
    return 0
  fi
  run_tracked_command "${dataset_cmd[@]}"
}

gpu_spec="${GPU_POOL}"
model_spec="${MODEL_SPECS}"
backend_specs_spec="${BACKEND_SPECS}"
router_mode_spec="${ROUTER_MODES}"
profiler_spec="${PROFILERS}"
seq_spec="${SEQ_LENS}"
expert_policy_spec="${EXPERT_POLICIES}"
lora_dropout_spec="${LORA_DROPOUT}"
output_root="${OUTPUT_ROOT}"
run_name="${RUN_NAME}"
batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}"
template_spec="${TEMPLATE}"
kt_repo_dir_user_set=false
[[ -n "${KT_REPO_DIR_ENV_SET}" ]] && kt_repo_dir_user_set=true

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --gpus) need_value "$1" "${2-}"; gpu_spec="$2"; shift 2 ;;
    --gpus=*) gpu_spec="${1#*=}"; shift ;;
    --dist-launcher) need_value "$1" "${2-}"; DIST_LAUNCHER="$(dist_launcher_label "$2")"; shift 2 ;;
    --dist-launcher=*) DIST_LAUNCHER="$(dist_launcher_label "${1#*=}")"; shift ;;
    --models|--model-specs) collect_values "$1" vals "${@:2}"; model_spec="${vals[*]}"; set -- "${REMAINING[@]}" ;;
    --models=*|--model-specs=*) model_spec="${1#*=}"; shift ;;
    --backend-specs) collect_values "$1" vals "${@:2}"; backend_specs_spec="${vals[*]}"; set -- "${REMAINING[@]}" ;;
    --backend-specs=*) backend_specs_spec="${1#*=}"; shift ;;
    --router-modes) need_value "$1" "${2-}"; router_mode_spec="$2"; ROUTER_MODES="$2"; shift 2 ;;
    --router-modes=*) router_mode_spec="${1#*=}"; ROUTER_MODES="${1#*=}"; shift ;;
    --profilers) need_value "$1" "${2-}"; profiler_spec="$2"; shift 2 ;;
    --profilers=*) profiler_spec="${1#*=}"; shift ;;
    --seq-lens) need_value "$1" "${2-}"; seq_spec="$2"; shift 2 ;;
    --seq-lens=*) seq_spec="${1#*=}"; shift ;;
    --expert-policies) need_value "$1" "${2-}"; expert_policy_spec="$2"; shift 2 ;;
    --expert-policies=*) expert_policy_spec="${1#*=}"; shift ;;
    --dataset) need_value "$1" "${2-}"; DATASET="$2"; shift 2 ;;
    --dataset=*) DATASET="${1#*=}"; shift ;;
    --prepare-datasets) need_value "$1" "${2-}"; PREPARE_DATASETS="$(bool_value "$2")"; shift 2 ;;
    --prepare-datasets=*) PREPARE_DATASETS="$(bool_value "${1#*=}")"; shift ;;
    --dataset-min-tokens) need_value "$1" "${2-}"; DATASET_MIN_TOKENS="$2"; shift 2 ;;
    --dataset-min-tokens=*) DATASET_MIN_TOKENS="${1#*=}"; shift ;;
    --dataset-eval-rows) need_value "$1" "${2-}"; DATASET_EVAL_ROWS="$2"; shift 2 ;;
    --dataset-eval-rows=*) DATASET_EVAL_ROWS="${1#*=}"; shift ;;
    --dataset-overwrite) need_value "$1" "${2-}"; DATASET_OVERWRITE="$(bool_value "$2")"; shift 2 ;;
    --dataset-overwrite=*) DATASET_OVERWRITE="$(bool_value "${1#*=}")"; shift ;;
    --template) need_value "$1" "${2-}"; template_spec="$2"; TEMPLATE="$2"; shift 2 ;;
    --template=*) template_spec="${1#*=}"; TEMPLATE="${1#*=}"; shift ;;
    --max-samples) need_value "$1" "${2-}"; MAX_SAMPLES="$2"; shift 2 ;;
    --max-samples=*) MAX_SAMPLES="${1#*=}"; shift ;;
    --max-steps) need_value "$1" "${2-}"; MAX_STEPS="$2"; shift 2 ;;
    --max-steps=*) MAX_STEPS="${1#*=}"; shift ;;
    --warmup-steps) need_value "$1" "${2-}"; WARMUP_STEPS="$2"; shift 2 ;;
    --warmup-steps=*) WARMUP_STEPS="${1#*=}"; shift ;;
    --batch-size) need_value "$1" "${2-}"; batch_size="$2"; PER_DEVICE_TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --batch-size=*) batch_size="${1#*=}"; PER_DEVICE_TRAIN_BATCH_SIZE="${1#*=}"; shift ;;
    --gradient-accumulation-steps) need_value "$1" "${2-}"; GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --gradient-accumulation-steps=*) GRADIENT_ACCUMULATION_STEPS="${1#*=}"; shift ;;
    --learning-rate) need_value "$1" "${2-}"; LEARNING_RATE="$2"; shift 2 ;;
    --learning-rate=*) LEARNING_RATE="${1#*=}"; shift ;;
    --lora-rank) need_value "$1" "${2-}"; LORA_RANK="$2"; shift 2 ;;
    --lora-rank=*) LORA_RANK="${1#*=}"; shift ;;
    --lora-alpha) need_value "$1" "${2-}"; LORA_ALPHA="$2"; shift 2 ;;
    --lora-alpha=*) LORA_ALPHA="${1#*=}"; shift ;;
    --seed) need_value "$1" "${2-}"; SEED="$2"; shift 2 ;;
    --seed=*) SEED="${1#*=}"; shift ;;
    --lora-dropout) collect_values "$1" vals "${@:2}"; lora_dropout_spec="${vals[*]}"; LORA_DROPOUT="${lora_dropout_spec}"; set -- "${REMAINING[@]}" ;;
    --lora-dropout=*) lora_dropout_spec="${1#*=}"; LORA_DROPOUT="${lora_dropout_spec}"; shift ;;
    --precision) need_value "$1" "${2-}"; PRECISION="$2"; shift 2 ;;
    --precision=*) PRECISION="${1#*=}"; shift ;;
    --profile-level) need_value "$1" "${2-}"; PROFILE_LEVEL="$2"; shift 2 ;;
    --profile-level=*) PROFILE_LEVEL="${1#*=}"; shift ;;
    --profile-layers) need_value "$1" "${2-}"; PROFILE_LAYERS="$2"; shift 2 ;;
    --profile-layers=*) PROFILE_LAYERS="${1#*=}"; shift ;;
    --profile-memory-attribution) need_value "$1" "${2-}"; PROFILE_MEMORY_ATTRIBUTION="$2"; shift 2 ;;
    --profile-memory-attribution=*) PROFILE_MEMORY_ATTRIBUTION="${1#*=}"; shift ;;
    --profile-memory-breakdown) need_value "$1" "${2-}"; PROFILE_MEMORY_BREAKDOWN="$2"; shift 2 ;;
    --profile-memory-breakdown=*) PROFILE_MEMORY_BREAKDOWN="${1#*=}"; shift ;;
    --profile-memory-breakdown-interval) need_value "$1" "${2-}"; PROFILE_MEMORY_BREAKDOWN_INTERVAL="$2"; shift 2 ;;
    --profile-memory-breakdown-interval=*) PROFILE_MEMORY_BREAKDOWN_INTERVAL="${1#*=}"; shift ;;
    --profile-memory-breakdown-steps) need_value "$1" "${2-}"; PROFILE_MEMORY_BREAKDOWN_STEPS="$2"; shift 2 ;;
    --profile-memory-breakdown-steps=*) PROFILE_MEMORY_BREAKDOWN_STEPS="${1#*=}"; shift ;;
    --profile-memory-breakdown-modules) need_value "$1" "${2-}"; PROFILE_MEMORY_BREAKDOWN_MODULES="$2"; shift 2 ;;
    --profile-memory-breakdown-modules=*) PROFILE_MEMORY_BREAKDOWN_MODULES="${1#*=}"; shift ;;
    --profile-sync) need_value "$1" "${2-}"; PROFILE_SYNC="$2"; shift 2 ;;
    --profile-sync=*) PROFILE_SYNC="${1#*=}"; shift ;;
    --profile-module-filter) need_value "$1" "${2-}"; PROFILE_MODULE_FILTER="$2"; shift 2 ;;
    --profile-module-filter=*) PROFILE_MODULE_FILTER="${1#*=}"; shift ;;
    --kt-kernel-dir) need_value "$1" "${2-}"; KT_KERNEL_DIR="$2"; shift 2 ;;
    --kt-kernel-dir=*) KT_KERNEL_DIR="${1#*=}"; shift ;;
    --kt-tools-dir) need_value "$1" "${2-}"; KT_TOOLS_DIR="$2"; shift 2 ;;
    --kt-tools-dir=*) KT_TOOLS_DIR="${1#*=}"; shift ;;
    --kt-repo-dir) need_value "$1" "${2-}"; KT_REPO_DIR="$2"; kt_repo_dir_user_set=true; shift 2 ;;
    --kt-repo-dir=*) KT_REPO_DIR="${1#*=}"; kt_repo_dir_user_set=true; shift ;;
    --kt-num-threads) need_value "$1" "${2-}"; KT_NUM_THREADS="$2"; shift 2 ;;
    --kt-num-threads=*) KT_NUM_THREADS="${1#*=}"; shift ;;
    --kt-threadpool-count) need_value "$1" "${2-}"; KT_THREADPOOL_COUNT="$2"; shift 2 ;;
    --kt-threadpool-count=*) KT_THREADPOOL_COUNT="${1#*=}"; shift ;;
    --kt-max-cache-depth) need_value "$1" "${2-}"; KT_MAX_CACHE_DEPTH="$2"; shift 2 ;;
    --kt-max-cache-depth=*) KT_MAX_CACHE_DEPTH="${1#*=}"; shift ;;
    --kt-tp-enabled) need_value "$1" "${2-}"; KT_TP_ENABLED="$(bool_value "$2")"; shift 2 ;;
    --kt-tp-enabled=*) KT_TP_ENABLED="$(bool_value "${1#*=}")"; shift ;;
    --kt-torchbf16-sft-device) need_value "$1" "${2-}"; KT_TORCHBF16_SFT_DEVICE="$2"; shift 2 ;;
    --kt-torchbf16-sft-device=*) KT_TORCHBF16_SFT_DEVICE="${1#*=}"; shift ;;
    --kt-arm-omp-num-threads) need_value "$1" "${2-}"; KT_ARM_OMP_NUM_THREADS="$2"; shift 2 ;;
    --kt-arm-omp-num-threads=*) KT_ARM_OMP_NUM_THREADS="${1#*=}"; shift ;;
    --kt-arm-omp-proc-bind) need_value "$1" "${2-}"; KT_ARM_OMP_PROC_BIND="$2"; shift 2 ;;
    --kt-arm-omp-proc-bind=*) KT_ARM_OMP_PROC_BIND="${1#*=}"; shift ;;
    --kt-arm-omp-places) need_value "$1" "${2-}"; KT_ARM_OMP_PLACES="$2"; shift 2 ;;
    --kt-arm-omp-places=*) KT_ARM_OMP_PLACES="${1#*=}"; shift ;;
    --kt-share-backward-bb) need_value "$1" "${2-}"; KT_SHARE_BACKWARD_BB="$(bool_value "$2")"; shift 2 ;;
    --kt-share-backward-bb=*) KT_SHARE_BACKWARD_BB="$(bool_value "${1#*=}")"; shift ;;
    --kt-share-cache-pool) need_value "$1" "${2-}"; KT_SHARE_CACHE_POOL="$(bool_value "$2")"; shift 2 ;;
    --kt-share-cache-pool=*) KT_SHARE_CACHE_POOL="$(bool_value "${1#*=}")"; shift ;;
    --kt-num-gpu-experts) need_value "$1" "${2-}"; KT_NUM_GPU_EXPERTS="$2"; shift 2 ;;
    --kt-num-gpu-experts=*) KT_NUM_GPU_EXPERTS="${1#*=}"; shift ;;
    --kt-weight-path) need_value "$1" "${2-}"; KT_WEIGHT_PATH="$2"; shift 2 ;;
    --kt-weight-path=*) KT_WEIGHT_PATH="${1#*=}"; shift ;;
    --kt-expert-checkpoint-path) need_value "$1" "${2-}"; KT_EXPERT_CHECKPOINT_PATH="$2"; shift 2 ;;
    --kt-expert-checkpoint-path=*) KT_EXPERT_CHECKPOINT_PATH="${1#*=}"; shift ;;
    --kt-use-lora-experts) need_value "$1" "${2-}"; KT_USE_LORA_EXPERTS="$(bool_value "$2")"; shift 2 ;;
    --kt-use-lora-experts=*) KT_USE_LORA_EXPERTS="$(bool_value "${1#*=}")"; shift ;;
    --kt-lora-expert-num) need_value "$1" "${2-}"; KT_LORA_EXPERT_NUM="$2"; shift 2 ;;
    --kt-lora-expert-num=*) KT_LORA_EXPERT_NUM="${1#*=}"; shift ;;
    --kt-lora-expert-intermediate-size) need_value "$1" "${2-}"; KT_LORA_EXPERT_INTERMEDIATE_SIZE="$2"; shift 2 ;;
    --kt-lora-expert-intermediate-size=*) KT_LORA_EXPERT_INTERMEDIATE_SIZE="${1#*=}"; shift ;;
    --check-kt-calls) need_value "$1" "${2-}"; CHECK_KT_CALLS="$(bool_value "$2")"; shift 2 ;;
    --check-kt-calls=*) CHECK_KT_CALLS="$(bool_value "${1#*=}")"; shift ;;
    --deepspeed-dir) need_value "$1" "${2-}"; DEEPSPEED_DIR="$2"; shift 2 ;;
    --deepspeed-dir=*) DEEPSPEED_DIR="${1#*=}"; shift ;;
    --check-superoffload) need_value "$1" "${2-}"; CHECK_SUPEROFFLOAD="$(bool_value "$2")"; shift 2 ;;
    --check-superoffload=*) CHECK_SUPEROFFLOAD="$(bool_value "${1#*=}")"; shift ;;
    --output-root) need_value "$1" "${2-}"; output_root="$2"; shift 2 ;;
    --output-root=*) output_root="${1#*=}"; shift ;;
    --run-name) need_value "$1" "${2-}"; run_name="$2"; shift 2 ;;
    --run-name=*) run_name="${1#*=}"; shift ;;
    --plot) need_value "$1" "${2-}"; PLOT="$(bool_value "$2")"; shift 2 ;;
    --plot=*) PLOT="$(bool_value "${1#*=}")"; shift ;;
    --plot-memory-breakdown) need_value "$1" "${2-}"; PLOT_MEMORY_BREAKDOWN="$(bool_value "$2")"; shift 2 ;;
    --plot-memory-breakdown=*) PLOT_MEMORY_BREAKDOWN="$(bool_value "${1#*=}")"; shift ;;
    --memory-breakdown-plot-y-scale) need_value "$1" "${2-}"; MEMORY_BREAKDOWN_PLOT_Y_SCALE="$2"; shift 2 ;;
    --memory-breakdown-plot-y-scale=*) MEMORY_BREAKDOWN_PLOT_Y_SCALE="${1#*=}"; shift ;;
    --plot-output-dir) need_value "$1" "${2-}"; PLOT_OUTPUT_DIR="$2"; shift 2 ;;
    --plot-output-dir=*) PLOT_OUTPUT_DIR="${1#*=}"; shift ;;
    --overwrite) need_value "$1" "${2-}"; OVERWRITE="$(bool_value "$2")"; shift 2 ;;
    --overwrite=*) OVERWRITE="$(bool_value "${1#*=}")"; shift ;;
    --continue-on-error) need_value "$1" "${2-}"; CONTINUE_ON_ERROR="$(bool_value "$2")"; shift 2 ;;
    --continue-on-error=*) CONTINUE_ON_ERROR="$(bool_value "${1#*=}")"; shift ;;
    --collect-existing) COLLECT_EXISTING=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) die "unknown option: $1" ;;
  esac
done

DIST_LAUNCHER="$(dist_launcher_label "${DIST_LAUNCHER}")"

require_comma_list "--gpus/GPU_POOL" "${gpu_spec}"
require_comma_list "--models/MODEL_SPECS" "${model_spec}"
require_comma_list "--backend-specs/BACKEND_SPECS" "${backend_specs_spec}"
require_comma_list "--router-modes/ROUTER_MODES" "${router_mode_spec}"
require_comma_list "--profilers/PROFILERS" "${profiler_spec}"
require_comma_list "--seq-lens/SEQ_LENS" "${seq_spec}"
require_comma_list "--expert-policies/EXPERT_POLICIES" "${expert_policy_spec}"
require_comma_list "--lora-dropout/LORA_DROPOUT" "${lora_dropout_spec}"

nonnegative_int "--max-steps" "${MAX_STEPS}"
nonnegative_int "--warmup-steps" "${WARMUP_STEPS}"
nonnegative_int "INTERRUPT_GRACE_SECONDS" "${INTERRUPT_GRACE_SECONDS}"
positive_int "--profile-memory-breakdown-interval" "${PROFILE_MEMORY_BREAKDOWN_INTERVAL}"
case "${MEMORY_BREAKDOWN_PLOT_Y_SCALE}" in
  shared|per-plot|global) ;;
  *) die "--memory-breakdown-plot-y-scale must be shared, per-plot, or global; got '${MEMORY_BREAKDOWN_PLOT_Y_SCALE}'" ;;
esac
mapfile -t lora_dropouts < <(tokens "${lora_dropout_spec}" | dedupe)
((${#lora_dropouts[@]})) || die "LoRA dropout list is empty"
for value in "${lora_dropouts[@]}"; do
  lora_dropout_label "${value}" >/dev/null
done
LORA_DROPOUT="${lora_dropouts[0]}"
lora_dropout_label_value="$(lora_dropout_label "${LORA_DROPOUT}")"
TOTAL_STEPS=$((MAX_STEPS + WARMUP_STEPS))
PREPARE_DATASETS=$(bool_value "${PREPARE_DATASETS}")
DATASET_OVERWRITE=$(bool_value "${DATASET_OVERWRITE}")
PLOT=$(bool_value "${PLOT}")
PLOT_MEMORY_BREAKDOWN=$(bool_value "${PLOT_MEMORY_BREAKDOWN}")
OVERWRITE=$(bool_value "${OVERWRITE}")
CONTINUE_ON_ERROR=$(bool_value "${CONTINUE_ON_ERROR}")
DRY_RUN=$(bool_value "${DRY_RUN}")
COLLECT_EXISTING=$(bool_value "${COLLECT_EXISTING}")
CHECK_SUPEROFFLOAD=$(bool_value "${CHECK_SUPEROFFLOAD}")
DATASET_MIN_TOKENS="${DATASET_MIN_TOKENS,,}"
positive_int "--dataset-eval-rows" "${DATASET_EVAL_ROWS}"
if [[ "${DATASET_MIN_TOKENS}" != "auto" ]]; then
  positive_int "--dataset-min-tokens" "${DATASET_MIN_TOKENS}"
fi

mapfile -t gpus < <(tokens "${gpu_spec}" | sed 's/^cuda://' | dedupe)
mapfile -t model_specs < <(tokens "${model_spec}" | dedupe)
backend_specs_raw=()
mapfile -t raw_backend_spec_tokens < <(tokens "${backend_specs_spec}")
((${#raw_backend_spec_tokens[@]} > 0)) || die "BACKEND_SPECS must include at least one backend|recompute spec"
for value in "${raw_backend_spec_tokens[@]}"; do
  append_backend_spec "${value}"
done
if ((${#backend_specs_raw[@]})); then
  mapfile -t backend_specs < <(printf '%s\n' "${backend_specs_raw[@]}" | dedupe)
else
  backend_specs=()
fi
mapfile -t backends < <(printf '%s\n' "${backend_specs[@]}" | cut -d '|' -f1 | dedupe)
mapfile -t backend_recompute_modes < <(printf '%s\n' "${backend_specs[@]}" | cut -d '|' -f2 | dedupe)
recompute_modes=("${backend_recompute_modes[@]}")
mapfile -t router_modes < <(tokens "${router_mode_spec}" | while read -r value; do router_mode_label "${value}"; done | dedupe)
router_hf_selected=false
router_whole_selected=false
for router_mode in "${router_modes[@]}"; do
  [[ "${router_mode}" == "hf" ]] && router_hf_selected=true
  [[ "${router_mode}" == "whole" ]] && router_whole_selected=true
done
selected_has_asym=false
selected_has_kt=false
selected_has_zero=false
selected_has_superoffload=false
selected_has_non_asym=false
for backend in "${backends[@]}"; do
  case "${backend}" in
    asym|asym_torch) selected_has_asym=true ;;
    zero2|zero3|zero3_offload) selected_has_zero=true ;;
    superoffload) selected_has_zero=true; selected_has_superoffload=true ;;
    kt_*) selected_has_kt=true ;;
  esac
  case "${backend}" in
    asym|asym_torch) ;;
    *) selected_has_non_asym=true ;;
  esac
done
plot_router_modes=("${router_modes[@]}")
if [[ "${router_whole_selected}" == "true" && "${selected_has_non_asym}" == "true" ]]; then
  plot_router_modes+=("hf")
fi
mapfile -t plot_router_modes < <(printf '%s\n' "${plot_router_modes[@]}" | dedupe)
if [[ "${selected_has_kt}" == "true" ]]; then
  KT_TP_ENABLED=$(bool_value "${KT_TP_ENABLED}")
  CHECK_KT_CALLS=$(bool_value "${CHECK_KT_CALLS}")
  KT_SHARE_BACKWARD_BB="$(optional_bool_value "${KT_SHARE_BACKWARD_BB}")"
  KT_SHARE_CACHE_POOL="$(optional_bool_value "${KT_SHARE_CACHE_POOL}")"
  KT_USE_LORA_EXPERTS="$(optional_bool_value "${KT_USE_LORA_EXPERTS}")"
  [[ -z "${KT_NUM_THREADS}" ]] || positive_int "--kt-num-threads" "${KT_NUM_THREADS}"
  [[ -z "${KT_THREADPOOL_COUNT}" ]] || positive_int "--kt-threadpool-count" "${KT_THREADPOOL_COUNT}"
  positive_int "--kt-max-cache-depth" "${KT_MAX_CACHE_DEPTH}"
  positive_int "--kt-arm-omp-num-threads" "${KT_ARM_OMP_NUM_THREADS}"
  [[ -z "${KT_NUM_GPU_EXPERTS}" ]] || nonnegative_int "--kt-num-gpu-experts" "${KT_NUM_GPU_EXPERTS}"
  [[ -z "${KT_LORA_EXPERT_NUM}" ]] || positive_int "--kt-lora-expert-num" "${KT_LORA_EXPERT_NUM}"
  [[ -z "${KT_LORA_EXPERT_INTERMEDIATE_SIZE}" ]] || positive_int "--kt-lora-expert-intermediate-size" "${KT_LORA_EXPERT_INTERMEDIATE_SIZE}"
  if [[ "${kt_repo_dir_user_set}" != "true" ]]; then
    KT_REPO_DIR="$(dirname "${KT_KERNEL_DIR}")"
  fi
fi
if [[ -z "${output_root}" ]]; then
  if [[ "${selected_has_kt}" == "true" ]]; then
    output_root="${KT_REPO_DIR}/profiling_kt"
  else
    output_root="${ASYM_DIR}/profiling"
  fi
fi
mapfile -t profilers < <(tokens "${profiler_spec}" | while read -r value; do profiler_label "${value}"; done | dedupe)
if printf '%s\n' "${profilers[@]}" | grep -qx 'nsys'; then
  plot_profilers=(nsys)
else
  plot_profilers=()
  for profiler in "${profilers[@]}"; do
    if [[ "$(profile_memory_flag_for_profiler "--profile-memory-breakdown" "${PROFILE_MEMORY_BREAKDOWN}" "${profiler}")" != "true" ]]; then
      plot_profilers+=("${profiler}")
    fi
  done
fi
mapfile -t seq_lens < <(tokens "${seq_spec}" | dedupe)
mapfile -t raw_expert_policies < <(tokens "${expert_policy_spec}" | dedupe)
((${#raw_expert_policies[@]} > 0)) || die "expert policies must include at least one explicit policy"
expert_policies=()
for value in "${raw_expert_policies[@]}"; do
  expert_policies+=("$(normalize_expert_policy "${value}")")
done
mapfile -t expert_policies < <(printf '%s\n' "${expert_policies[@]}" | dedupe)

((${#gpus[@]})) || die "GPU pool is empty"
((${#model_specs[@]})) || die "model spec list is empty"
((${#backend_specs[@]})) || die "backend spec list is empty"
((${#backends[@]})) || die "backend list is empty"
((${#router_modes[@]})) || die "router mode list is empty"
((${#profilers[@]})) || die "profiler list is empty"
((${#seq_lens[@]})) || die "sequence length list is empty"
for seq_len in "${seq_lens[@]}"; do
  positive_int "--seq-lens item" "${seq_len}"
done
((${#expert_policies[@]})) || die "expert policy list is empty"
[[ -f "${RUN_LF_SCRIPT}" ]] || die "missing ${RUN_LF_SCRIPT}"
if [[ "${selected_has_kt}" == "true" ]]; then
  [[ -d "${KT_KERNEL_DIR}" ]] || die "missing integrated kt-kernel dir ${KT_KERNEL_DIR}"
fi
[[ -f "${BUILD_DATASET_SCRIPT}" ]] || die "missing ${BUILD_DATASET_SCRIPT}"
[[ -f "${PROFILE_POSTPROCESS_SCRIPT}" ]] || die "missing ${PROFILE_POSTPROCESS_SCRIPT}"
[[ -f "${PLOT_SCRIPT}" ]] || die "missing ${PLOT_SCRIPT}"
if [[ "${PLOT_MEMORY_BREAKDOWN}" == "true" ]]; then
  [[ -f "${MEMORY_PLOT_SCRIPT}" ]] || die "missing ${MEMORY_PLOT_SCRIPT}"
fi
if [[ "${PLOT}" == "true" ]] && printf '%s\n' "${profilers[@]}" | grep -qx 'nsys'; then
  [[ -f "${INTERCONNECT_PLOT_SCRIPT}" ]] || die "missing ${INTERCONNECT_PLOT_SCRIPT}"
fi
if [[ "${PREPARE_DATASETS}" == "true" && "${DRY_RUN}" != "true" && "${COLLECT_EXISTING}" != "true" ]]; then
  [[ -x "${ENV_PYTHON}" ]] || die "missing executable LF Python at ${ENV_PYTHON}"
fi
if [[ "${selected_has_zero}" == "true" ]]; then
  for backend in "${backends[@]}"; do
    if is_zero_backend "${backend}"; then
      [[ -f "$(zero_deepspeed_config "${backend}")" ]] || die "missing DeepSpeed config for ${backend}: $(zero_deepspeed_config "${backend}")"
    fi
  done
fi
base_output_root="$(abs_path "${output_root}")"
if [[ "${selected_has_superoffload}" == "true" ]]; then
  [[ -f "${DEEPSPEED_DIR}/deepspeed/runtime/superoffload/superoffload_stage3.py" ]] || die "missing local SuperOffload DeepSpeed tree at ${DEEPSPEED_DIR}"
fi
precision_label="$(safe_label "${PRECISION}")"
dataset_root_label="$(safe_label "${DATASET}")"
[[ -n "${dataset_root_label}" ]] || die "--dataset must not be empty"
precision_root="${base_output_root}/${dataset_root_label}__lora__lf__${precision_label}"
if [[ "${DRY_RUN}" != "true" && "${COLLECT_EXISTING}" != "true" ]]; then
  mkdir -p "${precision_root}"
fi
echo "Output precision root: ${precision_root}"

declare -A plot_roots=()
declare -A memory_plot_roots=()
declare -A interconnect_plot_roots=()
failures=0
interrupted=false
interrupt_exit_status=130
current_child_pid=""
current_child_pid_file=""
current_wait_pid=""

child_process_alive() {
  local target_pid
  for target_pid in "$@"; do
    [[ "${target_pid}" =~ ^[0-9]+$ ]] || continue
    if kill -0 "-${target_pid}" 2>/dev/null || kill -0 "${target_pid}" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

signal_child_targets() {
  local signal_name="$1"
  shift
  local target_pid
  for target_pid in "$@"; do
    [[ "${target_pid}" =~ ^[0-9]+$ ]] || continue
    kill "-${signal_name}" "-${target_pid}" 2>/dev/null || true
    kill "-${signal_name}" "${target_pid}" 2>/dev/null || true
  done
}

current_child_targets() {
  local file_pid="" target_pid
  if [[ -n "${current_child_pid_file:-}" && -s "${current_child_pid_file}" ]]; then
    IFS= read -r file_pid < "${current_child_pid_file}" || true
  fi
  for target_pid in "${file_pid}" "${current_child_pid:-}" "${current_wait_pid:-}"; do
    [[ "${target_pid}" =~ ^[0-9]+$ ]] && printf '%s\n' "${target_pid}"
  done | awk 'NF && !seen[$0]++'
}

kill_current_child() {
  local -a target_pids=()
  mapfile -t target_pids < <(current_child_targets)
  rm -f "${current_child_pid_file:-}" 2>/dev/null || true

  ((${#target_pids[@]} > 0)) || return 0
  signal_child_targets INT "${target_pids[@]}"

  if [[ "${INTERRUPT_GRACE_SECONDS}" != "0" ]]; then
    sleep "${INTERRUPT_GRACE_SECONDS}" || true
  fi
  if child_process_alive "${target_pids[@]}"; then
    signal_child_targets TERM "${target_pids[@]}"
  fi

  if [[ "${INTERRUPT_GRACE_SECONDS}" != "0" ]]; then
    sleep "${INTERRUPT_GRACE_SECONDS}" || true
  fi
  if child_process_alive "${target_pids[@]}"; then
    signal_child_targets KILL "${target_pids[@]}"
  fi
}

cleanup_on_exit() {
  if [[ -n "${current_child_pid:-}" || -n "${current_wait_pid:-}" || -n "${current_child_pid_file:-}" ]]; then
    kill_current_child
  fi
}

run_tracked_command() {
  local status=0 wait_pid="" child_pid="" pid_file="" attempt
  current_child_pid=""
  current_wait_pid=""
  if command -v setsid >/dev/null 2>&1 && setsid --help 2>&1 | grep -q -- '--wait' && setsid --help 2>&1 | grep -q -- '--fork'; then
    pid_file="$(mktemp "${TMPDIR:-/tmp}/profile_lora_lf_child.XXXXXX")"
    current_child_pid_file="${pid_file}"
    setsid --fork --wait bash -c 'pid_file="$1"; shift; echo "$$" > "${pid_file}"; exec "$@"' _ "${pid_file}" "$@" &
    wait_pid=$!
    current_wait_pid="${wait_pid}"
    current_child_pid="${wait_pid}"
    for ((attempt = 0; attempt < 100; attempt++)); do
      if [[ -s "${pid_file}" ]]; then
        IFS= read -r child_pid < "${pid_file}" || true
        if [[ -n "${child_pid}" ]]; then
          current_child_pid="${child_pid}"
          break
        fi
      fi
      kill -0 "${wait_pid}" 2>/dev/null || break
      sleep 0.02
    done
  else
    "$@" &
    wait_pid=$!
    current_wait_pid="${wait_pid}"
    current_child_pid="${wait_pid}"
  fi
  wait "${wait_pid}" || status=$?
  if [[ "${interrupted}" == "true" ]]; then
    kill_current_child
    wait "${wait_pid}" 2>/dev/null || true
    current_child_pid=""
    current_child_pid_file=""
    current_wait_pid=""
    echo "Interrupted command; exiting without scheduling more jobs." >&2
    exit "${interrupt_exit_status}"
  fi
  if [[ "${status}" == "130" || "${status}" == "143" ]]; then
    kill_current_child
    wait "${wait_pid}" 2>/dev/null || true
    current_child_pid=""
    current_child_pid_file=""
    current_wait_pid=""
    echo "Interrupted command; exiting without scheduling more jobs." >&2
    exit "${status}"
  fi
  [[ -z "${pid_file}" ]] || rm -f "${pid_file}" 2>/dev/null || true
  current_child_pid=""
  current_child_pid_file=""
  current_wait_pid=""
  return "${status}"
}

run_tracked_command_logged() {
  local log_file="$1"
  shift
  run_tracked_command "$@" > >(tee -a "${log_file}") 2>&1
}

handle_interrupt() {
  local signal="${1:-INT}"
  case "${signal}" in
    TERM) interrupt_exit_status=143 ;;
    *) interrupt_exit_status=130 ;;
  esac
  interrupted=true
  echo "Interrupted; stopping LF profiling sweep." >&2
  trap - INT TERM
  kill_current_child
  exit "${interrupt_exit_status}"
}

trap 'handle_interrupt INT' INT
trap 'handle_interrupt TERM' TERM
trap 'cleanup_on_exit' EXIT

run_job() {
  local backend="$1"
  local profiler="$2"
  local recompute="$3"
  local seq_len="$4"
  local gpu="$5"
  local gpu_count="$6"
  local expert_policy="$7"
  local router_mode="$8"
  local dataset_name="$9"
  local gradient_checkpointing=false
  [[ "${recompute}" == "recomp" ]] && gradient_checkpointing=true

  local config_root job_root seq_root source_profile lf_out log_file run_id profile_json
  config_root="$(config_root_path "${seq_len}")"
  job_root="$(job_root_path "${config_root}" "${backend}" "${profiler}" "${recompute}" "${expert_policy}" "${router_mode}")"
  seq_root="${job_root}/s${seq_len}"
  source_profile="${seq_root}/source_profile.json"
  lf_out="${seq_root}/lf_run"
  log_file="${seq_root}/train.log"
  run_id="lf_${backend}_${profiler}_${recompute}_pol${expert_policy}_router${router_mode}_s${seq_len}_${lora_dropout_label_value}"
  profile_json="${seq_root}/profile.json"
  local profile_memory_attribution profile_memory_breakdown
  local master_port
  profile_memory_attribution="$(profile_memory_flag_for_profiler "--profile-memory-attribution" "${PROFILE_MEMORY_ATTRIBUTION}" "${profiler}")"
  profile_memory_breakdown="$(profile_memory_flag_for_profiler "--profile-memory-breakdown" "${PROFILE_MEMORY_BREAKDOWN}" "${profiler}")"
  master_port="${MASTER_PORT:-}"
  if [[ -z "${master_port}" ]]; then
    master_port="$(find_free_port)"
  fi

  plot_roots["${config_root}"]="${seq_len}"
  if [[ "${profile_memory_breakdown}" == "true" ]]; then
    memory_plot_roots["${config_root}"]="${seq_len}"
  fi
  if [[ "${profiler}" == "nsys" ]]; then
    interconnect_plot_roots["${config_root}"]="${seq_len}"
  fi
  if [[ "${DRY_RUN}" != "true" && -e "${profile_json}" && "${OVERWRITE}" != "true" && "${COLLECT_EXISTING}" != "true" ]]; then
    echo "Skipping existing: ${profile_json}"
    append_job_record "${config_root}" skipped \
      "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${router_mode}" "${backend}" "${profiler}" "${seq_root}" "${profile_json}" "${log_file}"
    return 0
  fi

  if [[ "${DRY_RUN}" != "true" && "${COLLECT_EXISTING}" == "true" ]]; then
    if [[ -e "${profile_json}" ]]; then
      echo "Found existing: ${profile_json}"
      return 0
    fi
    echo "Missing existing profile: ${profile_json}" >&2
    return 1
  fi

  local -a run_env=(
    ROOT="${ROOT}"
    LF_DIR="${LF_DIR}"
    ASYM_DIR="${ASYM_DIR}"
    ENV_DIR="${ENV_DIR}"
    CONDA_EXE="${CONDA_EXE}"
    NSYS_BIN="${NSYS_BIN}"
    DIST_LAUNCHER="${DIST_LAUNCHER}"
    DEEPSPEED_DIR="${DEEPSPEED_DIR}"
    CHECK_SUPEROFFLOAD="${CHECK_SUPEROFFLOAD}"
    MODEL_NAME_OR_PATH="${current_model_name}"
    BACKEND="${backend}"
    GPU_ID="${gpu}"
    NUM_GPUS="${gpu_count}"
    REQUIRE_SM100="${REQUIRE_SM100}"
    DATASET="${dataset_name}"
    TEMPLATE="${TEMPLATE}"
    CUTOFF_LEN="${seq_len}"
    MAX_SAMPLES="${MAX_SAMPLES}"
    MAX_STEPS="${TOTAL_STEPS}"
    PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}"
    LEARNING_RATE="${LEARNING_RATE}"
    LORA_RANK="${LORA_RANK}"
    LORA_ALPHA="${LORA_ALPHA}"
    LORA_DROPOUT="${LORA_DROPOUT}"
    SEED="${SEED}"
    GRADIENT_CHECKPOINTING="${gradient_checkpointing}"
    ASYM_PRECISION="${PRECISION}"
    ASYM_OFFLOAD_MODULES="${ASYM_OFFLOAD_MODULES}"
    ASYM_EXPERT_RECOMPUTE_POLICY="${expert_policy}"
    ASYM_ROUTER_MODE="${router_mode}"
    ASYM_STRICT="${ASYM_STRICT}"
    PROFILE=1
    PROFILE_PROFILER="${profiler}"
    PROFILE_LEVEL="${PROFILE_LEVEL}"
    PROFILE_LAYERS="${PROFILE_LAYERS}"
    PROFILE_MEMORY_ATTRIBUTION="${profile_memory_attribution}"
    PROFILE_MEMORY_BREAKDOWN="${profile_memory_breakdown}"
    PROFILE_MEMORY_BREAKDOWN_INTERVAL="${PROFILE_MEMORY_BREAKDOWN_INTERVAL}"
    PROFILE_MEMORY_BREAKDOWN_STEPS="${PROFILE_MEMORY_BREAKDOWN_STEPS}"
    PROFILE_MEMORY_BREAKDOWN_MODULES="${PROFILE_MEMORY_BREAKDOWN_MODULES}"
    PROFILE_SYNC="${PROFILE_SYNC}"
    PROFILE_MODULE_FILTER="${PROFILE_MODULE_FILTER}"
    PROFILE_SOURCE_JSON="${source_profile}"
    PROFILE_NSYS_PREFIX="${seq_root}/trace"
    PROFILE_NSYS_SQLITE="${seq_root}/trace.sqlite"
    PROFILE_NSYS_GPU_METRICS_DEVICES="${gpu}"
    PROFILE_JSON="${profile_json}"
    PROFILE_OUTPUT_DIR="${seq_root}"
    PROFILE_SUMMARY_MD="${seq_root}/summary.md"
    PROFILE_WORKLOAD_LABEL="${workload_label}"
    PROFILE_BACKEND_LABEL="${backend}"
    PROFILE_EXPERT_POLICY="${expert_policy}"
    INTERRUPT_GRACE_SECONDS="${INTERRUPT_GRACE_SECONDS}"
    PROFILE_WARMUP_STEPS="${WARMUP_STEPS}"
    PROFILE_MEASURE_STEPS="${MAX_STEPS}"
    PROFILE_TOTAL_STEPS="${TOTAL_STEPS}"
    MASTER_PORT="${master_port}"
    ASYM_GEMM_LF_CONFIG_WARMUP_STEPS="${WARMUP_STEPS}"
    ASYM_GEMM_LF_CONFIG_MEASURE_STEPS="${MAX_STEPS}"
    ASYM_GEMM_LF_CONFIG_TOTAL_STEPS="${TOTAL_STEPS}"
    ASYM_GEMM_LF_CONFIG_DEEPSPEED_DIR="${DEEPSPEED_DIR}"
    OUT_DIR="${lf_out}"
    LOG_FILE="${log_file}"
    RUN_ID="${run_id}"
  )
  if [[ "${backend}" == kt_* ]]; then
    run_env+=(
      KT_TOOLS_DIR="${KT_TOOLS_DIR}"
      KT_KERNEL_DIR="${KT_KERNEL_DIR}"
      KT_PRECISION="${PRECISION}"
      KT_NUM_THREADS="${KT_NUM_THREADS}"
      KT_THREADPOOL_COUNT="${KT_THREADPOOL_COUNT}"
      KT_MAX_CACHE_DEPTH="${KT_MAX_CACHE_DEPTH}"
      KT_TP_ENABLED="${KT_TP_ENABLED}"
      KT_SHARE_BACKWARD_BB="${KT_SHARE_BACKWARD_BB}"
      KT_SHARE_CACHE_POOL="${KT_SHARE_CACHE_POOL}"
      KT_NUM_GPU_EXPERTS="${KT_NUM_GPU_EXPERTS}"
      KT_WEIGHT_PATH="${KT_WEIGHT_PATH}"
      KT_EXPERT_CHECKPOINT_PATH="${KT_EXPERT_CHECKPOINT_PATH}"
      KT_USE_LORA_EXPERTS="${KT_USE_LORA_EXPERTS}"
      KT_LORA_EXPERT_NUM="${KT_LORA_EXPERT_NUM}"
      KT_LORA_EXPERT_INTERMEDIATE_SIZE="${KT_LORA_EXPERT_INTERMEDIATE_SIZE}"
      KT_TORCHBF16_SFT_DEVICE="${KT_TORCHBF16_SFT_DEVICE}"
      KT_ARM_OMP_NUM_THREADS="${KT_ARM_OMP_NUM_THREADS}"
      KT_ARM_OMP_PROC_BIND="${KT_ARM_OMP_PROC_BIND}"
      KT_ARM_OMP_PLACES="${KT_ARM_OMP_PLACES}"
      CHECK_KT_CALLS="${CHECK_KT_CALLS}"
    )
  fi

  local -a run_cmd=(env "${run_env[@]}" "${RUN_LF_SCRIPT}")

  echo "Running backend=${backend} profiler=${profiler} recompute=${recompute} expert_policy=${expert_policy} router_mode=${router_mode} seq=${seq_len} lora_dropout=${LORA_DROPOUT} gpu=${gpu} num_gpus=${gpu_count}"
  echo "  dir=${seq_root}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    print_command "${run_cmd[@]}"
    mkdir -p "${seq_root}"
    ensure_jobs_tsv "${config_root}"
    {
      print_command "${run_cmd[@]}"
    } > "${seq_root}/command.txt"
    append_job_record "${config_root}" dry-run \
      "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${router_mode}" "${backend}" "${profiler}" "${seq_root}" "${profile_json}" "${log_file}"
    return 0
  fi

  mkdir -p "${seq_root}"
  ensure_jobs_tsv "${config_root}"
  {
    print_command "${run_cmd[@]}"
  } > "${seq_root}/command.txt"

  local status=0
  run_tracked_command "${run_cmd[@]}" || status=$?
  if [[ "${interrupted}" == "true" ]]; then
    echo "Interrupted run; exiting without scheduling more jobs." >&2
    exit "${interrupt_exit_status}"
  fi
  if [[ "${status}" == "130" || "${status}" == "143" ]]; then
    echo "Interrupted run; exiting without scheduling more jobs." >&2
    exit "${status}"
  fi
  if ((status == 0)); then
    if [[ ! -f "${profile_json}" ]]; then
      echo "Missing expected profile artifact: ${profile_json}" >&2
      status=1
    fi
  fi

  if ((status == 0)); then
    append_job_record "${config_root}" ok \
      "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${router_mode}" "${backend}" "${profiler}" "${seq_root}" "${profile_json}" "${log_file}"
    if profiler_selected_for_plots "${profiler}"; then
      plot_single_run "${config_root}" "${seq_len}" "${backend}" "${profiler}" "${recompute}" "${expert_policy}" "${router_mode}" "${seq_root}"
      plot_running_combined "${config_root}" "${seq_len}" "${seq_root}"
    fi
    if [[ "${profile_memory_breakdown}" == "true" ]]; then
      plot_memory_single_run "${seq_root}"
      plot_memory_running_combined "${config_root}" "${seq_len}" "${seq_root}"
    fi
  else
    append_job_record "${config_root}" "failed:${status}" \
      "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${router_mode}" "${backend}" "${profiler}" "${seq_root}" "${profile_json}" "${log_file}"
  fi
  return "${status}"
}

plot_config_root() {
  local config_root="$1"
  local seq_len="$2"
  local plot_root
  [[ "${PLOT}" == "true" ]] || return 0
  ((${#plot_profilers[@]})) || return 0

  plot_root="${config_root}/combined"
  [[ -n "${PLOT_OUTPUT_DIR}" ]] && plot_root="$(abs_path "${PLOT_OUTPUT_DIR}")/$(basename "${config_root}")/combined"
  local -a plot_cmd
  plot_cmd_base plot_cmd "${config_root}" "${plot_root}" "${plot_root}" "${seq_len}"
  plot_cmd+=(--expert-recompute-policies "${expert_policies[@]}")
  append_sweep_plot_filters plot_cmd
  echo "Writing LF config combined plots: ${plot_root}"
  run_tracked_command "${plot_cmd[@]}"
}

plot_running_combined() {
  local config_root="$1"
  local seq_len="$2"
  local seq_root="$3"
  local plot_root
  [[ "${PLOT}" == "true" ]] || return 0
  ((${#plot_profilers[@]})) || return 0

  plot_root="${seq_root}/plots/_combined"
  local -a plot_cmd
  plot_cmd_base plot_cmd "${config_root}" "${plot_root}" "${plot_root}" "${seq_len}"
  plot_cmd+=(--combined-only --expert-recompute-policies "${expert_policies[@]}")
  append_sweep_plot_filters plot_cmd
  echo "Writing LF running combined plots: ${plot_root}"
  if ! run_tracked_command "${plot_cmd[@]}"; then
    echo "warning: failed to write running combined plots for ${seq_root}" >&2
  fi
}

plot_single_run() {
  local config_root="$1"
  local seq_len="$2"
  local backend="$3"
  local profiler="$4"
  local recompute="$5"
  local expert_policy="$6"
  local router_mode="$7"
  local seq_root="$8"
  local plot_root
  [[ "${PLOT}" == "true" ]] || return 0
  ((${#plot_profilers[@]})) || return 0

  plot_root="${seq_root}/plots"
  local -a plot_cmd
  plot_cmd_base plot_cmd "${config_root}" "${plot_root}" "${plot_root}/combined" "${seq_len}"
  plot_cmd+=(
    --skip-combined
    --expert-recompute-policies "${expert_policy}"
    --backend "${backend}"
    --profiler "${profiler}"
    --recompute "${recompute}"
    --router-mode "${router_mode}"
  )
  echo "Writing LF per-run plots: ${plot_root}"
  if ! run_tracked_command "${plot_cmd[@]}"; then
    echo "warning: failed to write per-run plots for ${seq_root}" >&2
  fi
}

plot_memory_single_run() {
  local seq_root="$1"
  local plot_root
  [[ "${PLOT}" == "true" && "${PLOT_MEMORY_BREAKDOWN}" == "true" ]] || return 0

  plot_root="${seq_root}/memory_plots"
  local -a plot_cmd=(
    "${CONDA_EXE}" run -p "${ENV_DIR}" python "${MEMORY_PLOT_SCRIPT}"
    --run-dir "${seq_root}"
    --output-dir "${plot_root}"
    --clean-output
    --y-scale "${MEMORY_BREAKDOWN_PLOT_Y_SCALE}"
  )
  echo "Writing LF source memory plots: ${plot_root}"
  if ! run_tracked_command "${plot_cmd[@]}"; then
    echo "warning: failed to write source memory plots for ${seq_root}" >&2
  fi
}

plot_memory_running_combined() {
  local config_root="$1"
  local seq_len="$2"
  local seq_root="$3"
  local plot_root
  [[ "${PLOT}" == "true" && "${PLOT_MEMORY_BREAKDOWN}" == "true" ]] || return 0

  plot_root="${seq_root}/memory_plots/_combined"
  local -a plot_cmd
  memory_combined_plot_cmd_base plot_cmd "${config_root}" "${plot_root}" "${seq_len}"
  memory_plot_filters plot_cmd
  echo "Writing LF running source-memory combined plots: ${plot_root}"
  if ! run_tracked_command "${plot_cmd[@]}"; then
    echo "warning: failed to write running source-memory combined plots for ${seq_root}" >&2
  fi
}

plot_memory_config_root() {
  local config_root="$1"
  local seq_len="$2"
  local plot_root
  [[ "${PLOT}" == "true" && "${PLOT_MEMORY_BREAKDOWN}" == "true" ]] || return 0

  plot_root="${config_root}/memory_combined"
  [[ -n "${PLOT_OUTPUT_DIR}" ]] && plot_root="$(abs_path "${PLOT_OUTPUT_DIR}")/$(basename "${config_root}")/memory_combined"
  local -a plot_cmd
  memory_combined_plot_cmd_base plot_cmd "${config_root}" "${plot_root}" "${seq_len}"
  memory_plot_filters plot_cmd
  echo "Writing LF source-memory combined plots: ${plot_root}"
  if ! run_tracked_command "${plot_cmd[@]}"; then
    echo "warning: failed to write source-memory combined plots for ${config_root}" >&2
  fi
}

write_missing_combined_readme() {
  local output_dir="$1"
  local title="$2"
  local reason="$3"
  mkdir -p "${output_dir}"
  cat > "${output_dir}/README.md" <<EOF
# ${title}

No plots were generated here.

${reason}
EOF
}

write_config_artifact_readme() {
  local config_root="$1"
  mkdir -p "${config_root}"
  cat > "${config_root}/ARTIFACTS.md" <<EOF
# LF Profiling Artifacts

This config root is organized as follows:

- \`combined/\`: config-level LF timing and allocator-summary plots from \`profile.json\`.
- \`memory_combined/\`: config-level source-memory breakdown plots. If no source-memory rows were collected, this folder contains a README explaining why.
- \`c2c_combined/\`: config-level C2C/CTC saturation plots from Nsight GPU metrics. If old traces lack GPU metrics, this folder contains a README explaining why.
- \`<backend>__<profiler>__<recompute>__pol<policy>__router<mode>/s<seq>/\`: per-run artifacts.

If \`PLOT_OUTPUT_DIR\` is set, combined plot folders are written under that external plot output root instead of this config root.

Per-run nsys folders contain \`profile.json\`, markdown summaries, \`plots/\` for per-run LF plots, and \`interconnect_ctc_*.png/csv\` when C2C GPU metrics are available.
EOF
}

write_precision_artifact_readme() {
  local precision_root="$1"
  mkdir -p "${precision_root}"
  cat > "${precision_root}/ARTIFACTS.md" <<EOF
# LF Profiling Precision Root

This precision root is organized as follows:

- \`combined/\`: global LF timing and allocator-summary plots across config roots.
- \`memory_combined/\`: global source-memory breakdown plots across config roots. If no source-memory rows were collected, this folder contains a README explaining why.
- \`c2c_combined/\`: global C2C/CTC saturation plots across config roots. If old traces lack Nsight GPU metrics, this folder contains a README explaining why.
- \`<config_root>/\`: one workload/configuration root. Each config root has its own \`combined/\`, \`memory_combined/\`, \`c2c_combined/\`, and per-run backend/profiler folders.

If \`PLOT_OUTPUT_DIR\` is set, global combined plot folders are written under that external plot output root instead of this precision root.

Fresh nsys runs collect C2C GPU metrics at 100 Hz. Existing traces created without Nsight \`GPU_METRICS\` tables cannot be converted into C2C saturation plots.
EOF
}

plot_interconnect_config_root() {
  local config_root="$1"
  local seq_len="$2"
  local plot_root
  [[ "${PLOT}" == "true" ]] || return 0

  plot_root="${config_root}/c2c_combined"
  [[ -n "${PLOT_OUTPUT_DIR}" ]] && plot_root="$(abs_path "${PLOT_OUTPUT_DIR}")/$(basename "${config_root}")/c2c_combined"
  local -a plot_cmd
  interconnect_combined_plot_cmd_base plot_cmd "${config_root}" "${plot_root}" "${seq_len}"
  interconnect_plot_filters plot_cmd
  echo "Writing LF C2C combined plots: ${plot_root}"
  if ! run_tracked_command "${plot_cmd[@]}"; then
    echo "warning: failed to write C2C combined plots for ${config_root}" >&2
  fi
}

collect_timing_workloads_from_roots() {
  local roots_name="$1"
  local workloads_name="$2"
  local workload_bases_name="$3"
  local -n roots_ref="${roots_name}"
  local -n workloads_ref="${workloads_name}"
  local -n workload_bases_ref="${workload_bases_name}"
  local config_root

  workloads_ref=()
  workload_bases_ref=()
  for config_root in "${!roots_ref[@]}"; do
    workloads_ref["$(plot_workload_from_config_root "${config_root}")"]=1
    workload_bases_ref["$(plot_workload_base_from_config_root "${config_root}")"]=1
  done
}

collect_workload_bases_from_roots() {
  local roots_name="$1"
  local workload_bases_name="$2"
  local -n roots_ref="${roots_name}"
  local -n workload_bases_ref="${workload_bases_name}"
  local config_root

  workload_bases_ref=()
  for config_root in "${!roots_ref[@]}"; do
    workload_bases_ref["$(plot_workload_base_from_config_root "${config_root}")"]=1
  done
}

timing_precision_combined_cmd() {
  local cmd_name="$1"
  local -n _cmd_ref="${cmd_name}"
  local output_dir="$2"
  shift 2

  plot_cmd_base "${cmd_name}" "${precision_root}" "${output_dir}" "${output_dir}" "${seq_lens[@]}"
  _cmd_ref+=(--combined-only --expert-recompute-policies "${expert_policies[@]}")
  _cmd_ref+=("$@")
}

memory_precision_combined_cmd() {
  local cmd_name="$1"
  local -n _cmd_ref="${cmd_name}"
  local output_dir="$2"
  shift 2

  memory_combined_plot_cmd_base "${cmd_name}" "${precision_root}" "${output_dir}" "${seq_lens[@]}"
  _cmd_ref+=("$@")
}

interconnect_precision_combined_cmd() {
  local cmd_name="$1"
  local -n _cmd_ref="${cmd_name}"
  local output_dir="$2"
  shift 2

  interconnect_combined_plot_cmd_base "${cmd_name}" "${precision_root}" "${output_dir}" "${seq_lens[@]}"
  _cmd_ref+=("$@")
}

run_precision_combined_plot() {
  local plot_root="$1"
  local build_func="$2"
  local filter_func="$3"
  local label="$4"
  local warn_on_failure="$5"
  shift 5

  local -a plot_cmd
  "${build_func}" plot_cmd "${plot_root}" "$@"
  "${filter_func}" plot_cmd
  echo "Writing combined LF ${label} plots: ${plot_root}"
  if [[ "${warn_on_failure}" == "true" ]]; then
    if ! run_tracked_command "${plot_cmd[@]}"; then
      echo "warning: failed to write combined LF ${label} plots" >&2
    fi
  else
    run_tracked_command "${plot_cmd[@]}"
  fi
}

run_model_split_precision_combined_plots() {
  local workload_bases_name="$1"
  local output_root="$2"
  local build_func="$3"
  local filter_func="$4"
  local label="$5"
  local warn_on_failure="$6"
  local -n workload_bases_ref="${workload_bases_name}"
  local workload_base plot_root

  for workload_base in "${!workload_bases_ref[@]}"; do
    plot_root="${output_root}/$(safe_label "${workload_base}")"
    local -a plot_cmd
    "${build_func}" plot_cmd "${plot_root}" --workload "${workload_base}"
    "${filter_func}" plot_cmd
    echo "Writing model-split combined LF ${label} plots: ${plot_root}"
    if [[ "${warn_on_failure}" == "true" ]]; then
      if ! run_tracked_command "${plot_cmd[@]}"; then
        echo "warning: failed to write model-split combined LF ${label} plots for ${workload_base}" >&2
      fi
    else
      run_tracked_command "${plot_cmd[@]}"
    fi
  done
}

plot_timing_precision_combined() {
  local combined_plot_root workload
  local -a workload_filters=()

  if ((${#plot_profilers[@]})); then
    combined_plot_root="${precision_root}/combined"
    [[ -n "${PLOT_OUTPUT_DIR}" ]] && combined_plot_root="$(abs_path "${PLOT_OUTPUT_DIR}")/combined"
    declare -A combined_workloads=()
    declare -A combined_workload_bases=()
    collect_timing_workloads_from_roots plot_roots combined_workloads combined_workload_bases

    for workload in "${!combined_workloads[@]}"; do
      workload_filters+=(--workload "${workload}")
    done
    run_precision_combined_plot "${combined_plot_root}" timing_precision_combined_cmd append_sweep_plot_filters profile false "${workload_filters[@]}"
    run_model_split_precision_combined_plots combined_workload_bases "${combined_plot_root}" timing_precision_combined_cmd append_sweep_plot_filters profile false
  elif [[ -z "${PLOT_OUTPUT_DIR}" ]]; then
    combined_plot_root="${precision_root}/combined"
    write_missing_combined_readme \
      "${combined_plot_root}" \
      "LF Global Combined Artifacts" \
      "No global LF timing plots were generated. This can happen when plotting is disabled or no matching profile rows are available."
  fi
}

plot_memory_precision_combined() {
  local combined_memory_plot_root
  [[ "${PLOT_MEMORY_BREAKDOWN}" == "true" ]] || return 0

  combined_memory_plot_root="${precision_root}/memory_combined"
  [[ -n "${PLOT_OUTPUT_DIR}" ]] && combined_memory_plot_root="$(abs_path "${PLOT_OUTPUT_DIR}")/memory_combined"
  if [[ "${#memory_plot_roots[@]}" -gt 0 ]]; then
    declare -A memory_combined_workload_bases=()
    collect_workload_bases_from_roots memory_plot_roots memory_combined_workload_bases
    run_precision_combined_plot "${combined_memory_plot_root}" memory_precision_combined_cmd memory_plot_filters source-memory false
    run_model_split_precision_combined_plots memory_combined_workload_bases "${combined_memory_plot_root}" memory_precision_combined_cmd memory_plot_filters source-memory false
  else
    write_missing_combined_readme \
      "${combined_memory_plot_root}" \
      "LF Source Memory Combined Artifacts" \
      "No source-memory breakdown rows were collected in this sweep. Include the source profiler in PROFILERS, or set PROFILE_MEMORY_BREAKDOWN=true for a run where source-memory hook overhead is acceptable."
  fi
}

plot_interconnect_precision_combined() {
  local combined_interconnect_plot_root

  combined_interconnect_plot_root="${precision_root}/c2c_combined"
  [[ -n "${PLOT_OUTPUT_DIR}" ]] && combined_interconnect_plot_root="$(abs_path "${PLOT_OUTPUT_DIR}")/c2c_combined"
  if [[ "${#interconnect_plot_roots[@]}" -gt 0 ]]; then
    declare -A interconnect_combined_workload_bases=()
    collect_workload_bases_from_roots interconnect_plot_roots interconnect_combined_workload_bases
    run_precision_combined_plot "${combined_interconnect_plot_root}" interconnect_precision_combined_cmd interconnect_plot_filters C2C true
    run_model_split_precision_combined_plots interconnect_combined_workload_bases "${combined_interconnect_plot_root}" interconnect_precision_combined_cmd interconnect_plot_filters C2C true
  else
    write_missing_combined_readme \
      "${combined_interconnect_plot_root}" \
      "LF C2C / CTC Combined Artifacts" \
      "No nsys profiler runs were selected in this sweep, so no Nsight C2C/CTC GPU metric samples can be summarized."
  fi
}

for model_spec_entry in "${model_specs[@]}"; do
  parse_model_spec "${model_spec_entry}"
  current_model_name="${parsed_model_name}"
  current_model_gpu_count="${parsed_model_gpu_count}"
  current_model_tag=$(basename "${current_model_name}" | tr '/:' '__')
  workload_label="$(safe_label "${current_model_tag}")"
  TEMPLATE="${template_spec}"
  if [[ "${TEMPLATE}" == "auto" ]]; then
    TEMPLATE="$(infer_template "${current_model_name}")"
  fi
  echo "Using template: ${TEMPLATE} for model ${current_model_name} requesting ${current_model_gpu_count} GPU(s), recompute from backend specs"

  for seq_len in "${seq_lens[@]}"; do
    current_dataset="${DATASET}"
    if [[ "${PREPARE_DATASETS}" == "true" ]]; then
      current_dataset="$(dataset_name_for_seq "${seq_len}")"
      if [[ "${COLLECT_EXISTING}" != "true" ]]; then
        if ! prepare_dataset_for_seq "${seq_len}" "${current_dataset}"; then
          failures=$((failures + 1))
          if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
            exit 1
          fi
          continue
        fi
      fi
    fi
    for lora_dropout in "${lora_dropouts[@]}"; do
      LORA_DROPOUT="${lora_dropout}"
      lora_dropout_label_value="$(lora_dropout_label "${LORA_DROPOUT}")"
      config_root="$(config_root_path "${seq_len}")"
      for router_mode in "${router_modes[@]}"; do
        for expert_policy in "${expert_policies[@]}"; do
          for backend_recompute in "${backend_specs[@]}"; do
            backend="${backend_recompute%%|*}"
            recompute="${backend_recompute##*|}"
            for profiler in "${profilers[@]}"; do
              job_router_mode="${router_mode}"
              if [[ "${router_mode}" == "whole" && "${backend}" != "asym" && "${backend}" != "asym_torch" ]]; then
                if [[ "${router_hf_selected}" != "true" ]]; then
                  job_router_mode=hf
                else
                  echo "Skipping backend=${backend} router_mode=${router_mode}; owned routing requires an AsymGEMM backend."
                  continue
                fi
              fi
              if is_policy_independent_backend "${backend}" && [[ "${expert_policy}" != "none" ]]; then
                echo "Skipping backend=${backend} expert_policy=${expert_policy}; torch/zero/SuperOffload/KT backends are policy-independent."
                continue
              fi
              gpu_count="$(backend_gpu_count "${backend}" "${current_model_gpu_count}")"
              gpu="$(gpu_slice "${gpu_count}")"
              if ! run_job "${backend}" "${profiler}" "${recompute}" "${seq_len}" "${gpu}" "${gpu_count}" "${expert_policy}" "${job_router_mode}" "${current_dataset}"; then
                failures=$((failures + 1))
                if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
                  exit 1
                fi
              fi
            done
          done
        done
      done
      if [[ "${DRY_RUN}" != "true" ]]; then
        plot_config_root "${config_root}" "${seq_len}"
        if [[ -z "${PLOT_OUTPUT_DIR}" && ! -d "${config_root}/combined" ]]; then
          write_missing_combined_readme \
            "${config_root}/combined" \
            "LF Config Combined Artifacts" \
            "No config-level LF timing plots were generated for this config. This can happen when plotting is disabled or no matching profile rows are available."
        fi
        if [[ -n "${memory_plot_roots[${config_root}]+set}" ]]; then
          plot_memory_config_root "${config_root}" "${seq_len}"
        else
          write_missing_combined_readme \
            "${config_root}/memory_combined" \
            "LF Source Memory Combined Artifacts" \
            "No source-memory breakdown rows were collected for this config. Include the source profiler in PROFILERS, or set PROFILE_MEMORY_BREAKDOWN=true for a run where source-memory hook overhead is acceptable."
        fi
        if [[ -n "${interconnect_plot_roots[${config_root}]+set}" ]]; then
          plot_interconnect_config_root "${config_root}" "${seq_len}"
        else
          write_missing_combined_readme \
            "${config_root}/c2c_combined" \
            "LF C2C / CTC Combined Artifacts" \
            "No nsys profiler run was selected for this config, so no Nsight C2C/CTC GPU metric samples can be summarized."
        fi
        write_config_artifact_readme "${config_root}"
      fi
    done
  done
done

if ((failures > 0)); then
  echo "${failures} profiling job(s) failed" >&2
  if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
    exit 1
  fi
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "Dry run completed; skipping plots."
  exit 0
fi

if [[ "${PLOT}" == "true" ]]; then
  plot_timing_precision_combined
  plot_memory_precision_combined
  plot_interconnect_precision_combined
fi

if [[ "${DRY_RUN}" != "true" ]]; then
  write_precision_artifact_readme "${precision_root}"
fi

echo "LF profiling completed. Results: ${precision_root}"
