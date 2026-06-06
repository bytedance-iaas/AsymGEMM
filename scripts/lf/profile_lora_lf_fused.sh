#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# User Parameters
# =============================================================================
ROOT=${ROOT:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM}
LF_DIR=${LF_DIR:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory}
CONDA_EXE=${CONDA_EXE:-conda}
NSYS_BIN=${NSYS_BIN:-nsys}

GPU_POOL=${GPU_POOL:-0,1}
MODEL_SPECS=${MODEL_SPECS:-"Qwen/Qwen3-30B-A3B|1,meta-llama/Llama-4-Scout-17B-16E|2,meta-llama/Llama-4-Scout-17B-16E|1"}
# MODEL_SPECS=${MODEL_SPECS:-"meta-llama/Llama-4-Scout-17B-16E|1"}

EXPERT_POLICIES=${EXPERT_POLICIES-"none,tok-le0,tok-le512,tok-le1024,tok-le512-act,tok-le1024-act"}

# Primary backend sweep axis. Each entry is backend|recompute; recompute aliases
# such as recompute/norecompute normalize to recomp/norecomp internally.
if [[ -z "${BACKEND_SPECS+x}" ]]; then
  if [[ -n "${BACKENDS+x}" || -n "${RECOMPUTE+x}" ]]; then
    BACKEND_SPECS=
  else
    BACKEND_SPECS="asym|norecompute,torch|norecompute,torch|recompute"
  fi
fi
BACKENDS=${BACKENDS:-asym,torch}  # legacy; used only when BACKEND_SPECS is empty or --backends/--recompute is passed
RECOMPUTE=${RECOMPUTE:-norecomp}  # legacy; use BACKEND_SPECS for new sweeps
# BACKEND_SPECS=${BACKEND_SPECS:-"asym|norecompute,torch|recompute"}
PROFILERS=${PROFILERS:-nsys}
PRECISION=${PRECISION:-bf16} 

DATASET=${DATASET:-asym_long_sft_smoke}
PREPARE_DATASETS=${PREPARE_DATASETS:-true}
DATASET_MIN_TOKENS=${DATASET_MIN_TOKENS:-auto}
DATASET_EVAL_ROWS=${DATASET_EVAL_ROWS:-128}
DATASET_OVERWRITE=${DATASET_OVERWRITE:-false}
TEMPLATE=${TEMPLATE:-auto}
SEQ_LENS=${SEQ_LENS:-4096}
MAX_SAMPLES=${MAX_SAMPLES:-128}
MAX_STEPS=${MAX_STEPS:-10}
WARMUP_STEPS=${WARMUP_STEPS:-5}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.00,0.10}

ASYM_OFFLOAD_MODULES=${ASYM_OFFLOAD_MODULES:-routed_experts}
ASYM_STRICT=${ASYM_STRICT:-true}
TORCH_USE_ASYM_GEMM_LORA=${TORCH_USE_ASYM_GEMM_LORA:-true}
REQUIRE_SM100=${REQUIRE_SM100:-1}
TORCH_DISTRIBUTED_BACKEND=${TORCH_DISTRIBUTED_BACKEND:-deepspeed}
TORCH_FSDP_CONFIG=${TORCH_FSDP_CONFIG:-${LF_DIR}/examples/accelerate/fsdp2_config.yaml}
# TORCH_DEEPSPEED_CONFIG=${TORCH_DEEPSPEED_CONFIG:-${LF_DIR}/examples/deepspeed/ds_z3_offload_config.json}
TORCH_DEEPSPEED_CONFIG=${TORCH_DEEPSPEED_CONFIG:-${LF_DIR}/examples/deepspeed/ds_z3_config.json}

# Optional output/profile controls
OUTPUT_ROOT=${OUTPUT_ROOT:-profiling}
PROFILE_LEVEL=${PROFILE_LEVEL:-op}
PROFILE_LAYERS=${PROFILE_LAYERS:-all}
PROFILE_MEMORY_ATTRIBUTION=${PROFILE_MEMORY_ATTRIBUTION:-auto}
PROFILE_MEMORY_BREAKDOWN=${PROFILE_MEMORY_BREAKDOWN:-auto}
PROFILE_MEMORY_BREAKDOWN_INTERVAL=${PROFILE_MEMORY_BREAKDOWN_INTERVAL:-1}
PROFILE_MEMORY_BREAKDOWN_STEPS=${PROFILE_MEMORY_BREAKDOWN_STEPS:-}
PROFILE_MEMORY_BREAKDOWN_MODULES=${PROFILE_MEMORY_BREAKDOWN_MODULES:-attention,mlp,experts,lora,embedding,loss}
PROFILE_SYNC=${PROFILE_SYNC:-0}
PROFILE_MODULE_FILTER=${PROFILE_MODULE_FILTER:-attention,mlp,experts,lora,optimizer}

# Optional loss-comparison controls
COMPARE_LOSSES=${COMPARE_LOSSES:-true}
COMPARE_BASELINE_BACKEND=${COMPARE_BASELINE_BACKEND:-torch}
COMPARE_CANDIDATE_BACKEND=${COMPARE_CANDIDATE_BACKEND:-asym}
COMPARE_FIRST_STEP_REL_TOL=${COMPARE_FIRST_STEP_REL_TOL:-0.02}
COMPARE_MAX_REL_TOL=${COMPARE_MAX_REL_TOL:-0.10}

# Optional plotting controls
PLOT=${PLOT:-true}
PLOT_MEMORY_BREAKDOWN=${PLOT_MEMORY_BREAKDOWN:-true}
MEMORY_BREAKDOWN_PLOT_Y_SCALE=${MEMORY_BREAKDOWN_PLOT_Y_SCALE:-shared}

# Optional execution controls
OVERWRITE=${OVERWRITE:-false}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-true}
DRY_RUN=${DRY_RUN:-false}
COLLECT_EXISTING=${COLLECT_EXISTING:-false}
INTERRUPT_GRACE_SECONDS=${INTERRUPT_GRACE_SECONDS:-2}

# Empty optional user parameters
RUN_NAME=${RUN_NAME:-}
COMPARE_MIN_STEPS=${COMPARE_MIN_STEPS:-}
PLOT_OUTPUT_DIR=${PLOT_OUTPUT_DIR:-}

# =============================================================================
# Derived Parameters
# =============================================================================
ASYM_DIR=${ASYM_DIR:-${ROOT}}
ENV_DIR=${ENV_DIR:-${LF_DIR}/.venv}
ENV_PYTHON=${ENV_PYTHON:-${ENV_DIR}/bin/python}
RUN_LF_SCRIPT="${ASYM_DIR}/scripts/lf/run_lf_lora_sft.sh"
BUILD_DATASET_SCRIPT="${ASYM_DIR}/scripts/lf/build_lf_sft_eval_pair.py"
PROFILE_POSTPROCESS_SCRIPT="${ASYM_DIR}/scripts/lf/postprocess_lf_profile_artifacts.py"
PLOT_SCRIPT="${ASYM_DIR}/scripts/plotting/plot_activation_recompute_sweep.py"
MEMORY_PLOT_SCRIPT="${ASYM_DIR}/scripts/plotting/plot_lf_memory_breakdown.py"

# =============================================================================
# Main Logic
# =============================================================================
usage() {
  cat <<USAGE
Usage:
  scripts/lf/profile_lora_lf.sh [options]

Defaults:
  --gpus ${GPU_POOL}
  --backend-specs ${BACKEND_SPECS:-"<legacy: ${BACKENDS}|${RECOMPUTE}>"}
  --profilers ${PROFILERS}
  --seq-lens ${SEQ_LENS}
  --output-root ${OUTPUT_ROOT}

Options:
  List values must be comma-separated with no spaces.

  --gpus LIST                    Physical GPU pool, e.g. 0,1.
  --models LIST                  Model specs. Each item is model_name_or_path|num_gpus.
                                 Example: meta-llama/Llama-4-Scout-17B-16E|1,meta-llama/Llama-4-Maverick-17B-128E|4
  --backend-specs LIST           Backend/recompute specs, e.g. 'asym|norecompute,torch|recompute'.
                                 Accepts recompute/norecompute or recomp/norecomp.
  --backends LIST                Legacy: asym and/or torch. Combined with --recompute.
  --profilers LIST               source and/or nsys.
  --seq-lens LIST                LF cutoff lengths. Accepts positive integers, e.g. 4096,8192.
  --recompute norecomp|recomp|both  Legacy: expands every backend to the selected recompute mode(s).
  --expert-policies LIST         AsymGEMM expert policies: none, tok-le0, tok-le0-act, tok-leN, tok-geN, tokA-B, and -act variants.
  --dataset NAME
  --prepare-datasets true|false  Build/audit model+length-specific LF datasets before training.
  --dataset-min-tokens N|auto    Minimum source tokens for generated/audited rows. auto uses the seq length.
  --dataset-eval-rows N
  --dataset-overwrite true|false Rewrite existing generated dataset files.
  --template NAME
  --max-samples N
  --max-steps N                 Measured steps kept in plots/summaries.
  --warmup-steps N              Extra initial steps to run but exclude from plots/summaries.
  --batch-size N
  --gradient-accumulation-steps N
  --learning-rate VALUE
  --lora-rank N
  --lora-alpha VALUE
  --lora-dropout LIST           LoRA dropout probabilities in fixed 0.xx format, e.g. 0.00,0.10.
  --precision NAME
  --profile-level stage|module|op|deep
  --profile-layers all|first,last|0,1,2|every4
  --profile-memory-attribution auto|true|false
  --profile-memory-breakdown auto|true|false
  --profile-memory-breakdown-interval N
  --profile-memory-breakdown-steps LIST
  --profile-memory-breakdown-modules LIST
  --profile-sync true|false
  --profile-module-filter LIST
  --torch-use-asym-gemm-lora true|false
                                 For BACKEND=torch, attach packed-expert LoRA through the AsymGEMM torch backend.
                                 Default true so torch/asym train the same LF LoRA target=all modules.
  --torch-distributed-backend fsdp2|deepspeed|ddp
  --torch-fsdp-config PATH       Accelerate config for torch FSDP2. Defaults to LF's examples/accelerate/fsdp2_config.yaml.
  --torch-deepspeed-config PATH  DeepSpeed config for torch backend. Defaults to LF's examples/deepspeed/ds_z3_config.json.
  --compare-losses true|false
  --compare-baseline-backend torch|asym
  --compare-candidate-backend torch|asym
  --compare-min-steps N
  --compare-first-step-rel-tol VALUE
  --compare-max-rel-tol VALUE
  --output-root DIR              Default layout: <root>/<dataset>__lora__lf__<precision>/<model>__gpus<model_gpus>__b<batch>_s<seq>_w<warmup>_s<steps>_r<rank>_a<alpha>_drop0xx
  --run-name NAME                Optional config directory under <dataset>__lora__lf__<precision>.
  --plot true|false
  --plot-memory-breakdown true|false
  --memory-breakdown-plot-y-scale shared|per-plot|global
  --plot-output-dir DIR
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
  parsed_model_name="${spec%|*}"
  parsed_model_gpu_count="${spec##*|}"
  if [[ "${parsed_model_name}" == "${spec}" ]]; then
    parsed_model_gpu_count=1
  fi
  [[ -n "${parsed_model_name}" ]] || die "empty model name in model spec '${spec}'"
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
    asym) printf '1\n' ;;
    torch) printf '%s\n' "${model_gpu_count}" ;;
    *) die "internal backend label must be asym or torch, got '${backend}'" ;;
  esac
}

recompute_label() {
  case "${1,,}" in
    norecomp|norecompute|no-recompute|no_recompute|false|0|off) printf 'norecomp\n' ;;
    recomp|recompute|true|1|on) printf 'recomp\n' ;;
    both) printf 'both\n' ;;
    *) die "expected recompute mode norecomp/recomp, norecompute/recompute, or both; got '${1}'" ;;
  esac
}

recompute_values() {
  case "$(recompute_label "$1")" in
    norecomp) printf 'norecomp\n' ;;
    recomp) printf 'recomp\n' ;;
    both) printf 'norecomp\nrecomp\n' ;;
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
    asym) printf 'asym\n' ;;
    torch|asym_torch) printf 'torch\n' ;;
    *) die "backend must be asym or torch, got '${1}'" ;;
  esac
}

expand_backend_spec() {
  local raw="$1"
  local backend_part recompute_part backend recompute_token recompute_mode
  local -a recompute_tokens recompute_modes_for_spec

  if [[ "${raw}" == *"|"* ]]; then
    backend_part="${raw%%|*}"
    recompute_part="${raw#*|}"
  else
    backend_part="${raw}"
    recompute_part="norecomp"
  fi

  [[ -n "${backend_part}" ]] || die "empty backend in backend spec '${raw}'"
  [[ -n "${recompute_part}" ]] || die "empty recompute mode in backend spec '${raw}'"
  backend="$(backend_label "${backend_part}")"

  mapfile -t recompute_tokens < <(tokens "${recompute_part//\//,}")
  ((${#recompute_tokens[@]} > 0)) || die "empty recompute mode in backend spec '${raw}'"
  for recompute_token in "${recompute_tokens[@]}"; do
    mapfile -t recompute_modes_for_spec < <(recompute_values "${recompute_token}")
    for recompute_mode in "${recompute_modes_for_spec[@]}"; do
      printf '%s|%s\n' "${backend}" "${recompute_mode}"
    done
  done
}

expand_legacy_backend_specs() {
  local backend_spec="$1"
  local recompute_spec="$2"
  local backend recompute
  local -a legacy_backends legacy_recompute_modes

  mapfile -t legacy_backends < <(tokens "${backend_spec}" | while read -r backend; do backend_label "${backend}"; done | dedupe)
  mapfile -t legacy_recompute_modes < <(recompute_values "${recompute_spec}")
  for backend in "${legacy_backends[@]}"; do
    for recompute in "${legacy_recompute_modes[@]}"; do
      printf '%s|%s\n' "${backend}" "${recompute}"
    done
  done
}

profiler_label() {
  case "${1,,}" in
    source|nsys) printf '%s\n' "${1,,}" ;;
    *) die "profiler must be source or nsys, got '${1}'" ;;
  esac
}

memory_attribution_for_profiler() {
  local profiler="$1"
  case "${PROFILE_MEMORY_ATTRIBUTION,,}" in
    auto)
      if [[ "${profiler}" == "source" ]]; then
        printf 'true\n'
      else
        printf 'false\n'
      fi
      ;;
    1|true|yes|y|on) printf 'true\n' ;;
    0|false|no|n|off) printf 'false\n' ;;
    *) die "--profile-memory-attribution must be auto, true, or false; got '${PROFILE_MEMORY_ATTRIBUTION}'" ;;
  esac
}

memory_breakdown_for_profiler() {
  local profiler="$1"
  case "${PROFILE_MEMORY_BREAKDOWN,,}" in
    auto)
      if [[ "${profiler}" == "source" ]]; then
        printf 'true\n'
      else
        printf 'false\n'
      fi
      ;;
    1|true|yes|y|on) printf 'true\n' ;;
    0|false|no|n|off) printf 'false\n' ;;
    *) die "--profile-memory-breakdown must be auto, true, or false; got '${PROFILE_MEMORY_BREAKDOWN}'" ;;
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
  printf '%s/%s\n' "${config_root}" "$(safe_label "${backend}__${profiler}__${recompute}__pol${expert_policy}")"
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

ensure_jobs_tsv() {
  local config_root="$1"
  mkdir -p "${config_root}"
  if [[ ! -e "${config_root}/jobs.tsv" ]]; then
    printf 'status\tgpu\tseq_len\trecompute\texpert_policy\tbackend\tprofiler\tjob_dir\tprofile_json\tlog\n' > "${config_root}/jobs.tsv"
  fi
}

append_job_record() {
  local config_root="$1"
  local status="$2"
  shift 2
  ensure_jobs_tsv "${config_root}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${status}" "$@" >> "${config_root}/jobs.tsv"
}

plot_cmd_base() {
  local -n cmd_ref="$1"
  local input_root="$2"
  local output_dir="$3"
  local combined_output_dir="$4"
  local seq_len="$5"
  cmd_ref=(
    "${CONDA_EXE}" run -p "${ENV_DIR}" python "${PLOT_SCRIPT}"
    --input-root "${input_root}"
    --output-dir "${output_dir}"
    --combined-output-dir "${combined_output_dir}"
    --precision "${PRECISION}"
    --clean-output
    --batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --seq-lens "${seq_len}"
  )
}

append_sweep_plot_filters() {
  local -n cmd_ref="$1"
  local backend profiler recompute
  for backend in "${backends[@]}"; do cmd_ref+=(--backend "${backend}"); done
  for profiler in "${plot_profilers[@]}"; do cmd_ref+=(--profiler "${profiler}"); done
  for recompute in "${recompute_modes[@]}"; do cmd_ref+=(--recompute "${recompute}"); done
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

  local -a dataset_cmd=(
    "${ENV_PYTHON}" "${BUILD_DATASET_SCRIPT}"
    --lf-dir "${LF_DIR}"
    --asym-dir "${ASYM_DIR}"
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
backend_spec="${BACKENDS}"
profiler_spec="${PROFILERS}"
seq_spec="${SEQ_LENS}"
recompute_spec="${RECOMPUTE}"
expert_policy_spec="${EXPERT_POLICIES}"
lora_dropout_spec="${LORA_DROPOUT}"
output_root="${OUTPUT_ROOT}"
run_name="${RUN_NAME}"
batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}"
template_spec="${TEMPLATE}"
backend_specs_cli_set=false
legacy_backend_axes_cli_set=false

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --gpus) need_value "$1" "${2-}"; gpu_spec="$2"; shift 2 ;;
    --gpus=*) gpu_spec="${1#*=}"; shift ;;
    --models|--model-specs) collect_values "$1" vals "${@:2}"; model_spec="${vals[*]}"; set -- "${REMAINING[@]}" ;;
    --models=*|--model-specs=*) model_spec="${1#*=}"; shift ;;
    --backend-specs|--backend-spec) collect_values "$1" vals "${@:2}"; backend_specs_spec="${vals[*]}"; backend_specs_cli_set=true; set -- "${REMAINING[@]}" ;;
    --backend-specs=*|--backend-spec=*) backend_specs_spec="${1#*=}"; backend_specs_cli_set=true; shift ;;
    --backends) need_value "$1" "${2-}"; backend_spec="$2"; backend_specs_spec=""; legacy_backend_axes_cli_set=true; shift 2 ;;
    --backends=*) backend_spec="${1#*=}"; backend_specs_spec=""; legacy_backend_axes_cli_set=true; shift ;;
    --profilers) need_value "$1" "${2-}"; profiler_spec="$2"; shift 2 ;;
    --profilers=*) profiler_spec="${1#*=}"; shift ;;
    --seq-lens) need_value "$1" "${2-}"; seq_spec="$2"; shift 2 ;;
    --seq-lens=*) seq_spec="${1#*=}"; shift ;;
    --recompute) need_value "$1" "${2-}"; recompute_spec="$2"; backend_specs_spec=""; legacy_backend_axes_cli_set=true; shift 2 ;;
    --recompute=*) recompute_spec="${1#*=}"; backend_specs_spec=""; legacy_backend_axes_cli_set=true; shift ;;
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
    --torch-use-asym-gemm-lora) need_value "$1" "${2-}"; TORCH_USE_ASYM_GEMM_LORA="$(bool_value "$2")"; shift 2 ;;
    --torch-use-asym-gemm-lora=*) TORCH_USE_ASYM_GEMM_LORA="$(bool_value "${1#*=}")"; shift ;;
    --torch-distributed-backend) need_value "$1" "${2-}"; TORCH_DISTRIBUTED_BACKEND="$2"; shift 2 ;;
    --torch-distributed-backend=*) TORCH_DISTRIBUTED_BACKEND="${1#*=}"; shift ;;
    --torch-fsdp-config) need_value "$1" "${2-}"; TORCH_FSDP_CONFIG="$2"; shift 2 ;;
    --torch-fsdp-config=*) TORCH_FSDP_CONFIG="${1#*=}"; shift ;;
    --torch-deepspeed-config) need_value "$1" "${2-}"; TORCH_DEEPSPEED_CONFIG="$2"; shift 2 ;;
    --torch-deepspeed-config=*) TORCH_DEEPSPEED_CONFIG="${1#*=}"; shift ;;
    --compare-losses) need_value "$1" "${2-}"; COMPARE_LOSSES="$(bool_value "$2")"; shift 2 ;;
    --compare-losses=*) COMPARE_LOSSES="$(bool_value "${1#*=}")"; shift ;;
    --compare-baseline-backend) need_value "$1" "${2-}"; COMPARE_BASELINE_BACKEND="$2"; shift 2 ;;
    --compare-baseline-backend=*) COMPARE_BASELINE_BACKEND="${1#*=}"; shift ;;
    --compare-candidate-backend) need_value "$1" "${2-}"; COMPARE_CANDIDATE_BACKEND="$2"; shift 2 ;;
    --compare-candidate-backend=*) COMPARE_CANDIDATE_BACKEND="${1#*=}"; shift ;;
    --compare-min-steps) need_value "$1" "${2-}"; COMPARE_MIN_STEPS="$2"; shift 2 ;;
    --compare-min-steps=*) COMPARE_MIN_STEPS="${1#*=}"; shift ;;
    --compare-first-step-rel-tol) need_value "$1" "${2-}"; COMPARE_FIRST_STEP_REL_TOL="$2"; shift 2 ;;
    --compare-first-step-rel-tol=*) COMPARE_FIRST_STEP_REL_TOL="${1#*=}"; shift ;;
    --compare-max-rel-tol) need_value "$1" "${2-}"; COMPARE_MAX_REL_TOL="$2"; shift 2 ;;
    --compare-max-rel-tol=*) COMPARE_MAX_REL_TOL="${1#*=}"; shift ;;
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

if [[ "${backend_specs_cli_set}" == "true" && "${legacy_backend_axes_cli_set}" == "true" ]]; then
  die "--backend-specs cannot be combined with --backends or --recompute"
fi

require_comma_list "--gpus/GPU_POOL" "${gpu_spec}"
require_comma_list "--models/MODEL_SPECS" "${model_spec}"
require_comma_list "--backend-specs/BACKEND_SPECS" "${backend_specs_spec}"
require_comma_list "--backends/BACKENDS" "${backend_spec}"
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
if [[ -z "${COMPARE_MIN_STEPS}" ]]; then
  COMPARE_MIN_STEPS="${MAX_STEPS}"
fi
nonnegative_int "--compare-min-steps" "${COMPARE_MIN_STEPS}"
PREPARE_DATASETS=$(bool_value "${PREPARE_DATASETS}")
DATASET_OVERWRITE=$(bool_value "${DATASET_OVERWRITE}")
TORCH_USE_ASYM_GEMM_LORA=$(bool_value "${TORCH_USE_ASYM_GEMM_LORA}")
DATASET_MIN_TOKENS="${DATASET_MIN_TOKENS,,}"
positive_int "--dataset-eval-rows" "${DATASET_EVAL_ROWS}"
if [[ "${DATASET_MIN_TOKENS}" != "auto" ]]; then
  positive_int "--dataset-min-tokens" "${DATASET_MIN_TOKENS}"
fi

mapfile -t gpus < <(tokens "${gpu_spec}" | sed 's/^cuda://' | dedupe)
mapfile -t model_specs < <(tokens "${model_spec}" | dedupe)
if [[ -n "${backend_specs_spec}" ]]; then
  mapfile -t backend_specs < <(tokens "${backend_specs_spec}" | while read -r value; do expand_backend_spec "${value}"; done | dedupe)
else
  mapfile -t backend_specs < <(expand_legacy_backend_specs "${backend_spec}" "${recompute_spec}" | dedupe)
fi
mapfile -t backends < <(printf '%s\n' "${backend_specs[@]}" | cut -d '|' -f1 | dedupe)
mapfile -t recompute_modes < <(printf '%s\n' "${backend_specs[@]}" | cut -d '|' -f2 | dedupe)
mapfile -t profilers < <(tokens "${profiler_spec}" | while read -r value; do profiler_label "${value}"; done | dedupe)
if printf '%s\n' "${profilers[@]}" | grep -qx 'nsys'; then
  plot_profilers=(nsys)
else
  plot_profilers=()
  for profiler in "${profilers[@]}"; do
    if [[ "$(memory_breakdown_for_profiler "${profiler}")" != "true" ]]; then
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

case "${TORCH_DISTRIBUTED_BACKEND,,}" in
  fsdp2) TORCH_DISTRIBUTED_BACKEND=fsdp2 ;;
  deepspeed|ds|zero3|z3) TORCH_DISTRIBUTED_BACKEND=deepspeed ;;
  ddp) TORCH_DISTRIBUTED_BACKEND=ddp ;;
  *) die "--torch-distributed-backend must be fsdp2, deepspeed, or ddp, got '${TORCH_DISTRIBUTED_BACKEND}'" ;;
esac

((${#gpus[@]})) || die "GPU pool is empty"
((${#model_specs[@]})) || die "model spec list is empty"
((${#backend_specs[@]})) || die "backend spec list is empty"
((${#backends[@]})) || die "backend list is empty"
((${#profilers[@]})) || die "profiler list is empty"
((${#seq_lens[@]})) || die "sequence length list is empty"
for seq_len in "${seq_lens[@]}"; do
  positive_int "--seq-lens item" "${seq_len}"
done
((${#expert_policies[@]})) || die "expert policy list is empty"
[[ -f "${RUN_LF_SCRIPT}" ]] || die "missing ${RUN_LF_SCRIPT}"
[[ -f "${BUILD_DATASET_SCRIPT}" ]] || die "missing ${BUILD_DATASET_SCRIPT}"
[[ -f "${PROFILE_POSTPROCESS_SCRIPT}" ]] || die "missing ${PROFILE_POSTPROCESS_SCRIPT}"
[[ -f "${PLOT_SCRIPT}" ]] || die "missing ${PLOT_SCRIPT}"
if [[ "${PLOT_MEMORY_BREAKDOWN}" == "true" ]]; then
  [[ -f "${MEMORY_PLOT_SCRIPT}" ]] || die "missing ${MEMORY_PLOT_SCRIPT}"
fi
if [[ "${PREPARE_DATASETS}" == "true" && "${DRY_RUN}" != "true" && "${COLLECT_EXISTING}" != "true" ]]; then
  [[ -x "${ENV_PYTHON}" ]] || die "missing executable LF Python at ${ENV_PYTHON}"
fi
if [[ "${TORCH_DISTRIBUTED_BACKEND}" == "fsdp2" ]]; then
  [[ -f "${TORCH_FSDP_CONFIG}" ]] || die "missing torch FSDP2 accelerate config ${TORCH_FSDP_CONFIG}"
fi
if [[ "${TORCH_DISTRIBUTED_BACKEND}" == "deepspeed" ]]; then
  [[ -f "${TORCH_DEEPSPEED_CONFIG}" ]] || die "missing torch DeepSpeed config ${TORCH_DEEPSPEED_CONFIG}"
fi
COMPARE_LOSSES=$(bool_value "${COMPARE_LOSSES}")
PLOT_MEMORY_BREAKDOWN=$(bool_value "${PLOT_MEMORY_BREAKDOWN}")
compare_baseline_backend="$(backend_label "${COMPARE_BASELINE_BACKEND}")"
compare_candidate_backend="$(backend_label "${COMPARE_CANDIDATE_BACKEND}")"
[[ "${compare_baseline_backend}" != "${compare_candidate_backend}" ]] || die "compare backends must differ"

base_output_root="$(abs_path "${output_root}")"
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
declare -A run_dirs=()
declare -A compare_groups=()
declare -A compare_group_config_roots=()
declare -A compare_group_labels=()
declare -a compare_group_keys=()
failures=0
interrupted=false
interrupt_exit_status=130
current_child_pid=""
current_child_pid_file=""
current_wait_pid=""

child_process_alive() {
  local pid="$1"
  kill -0 "-${pid}" 2>/dev/null || kill -0 "${pid}" 2>/dev/null
}

kill_current_child() {
  local base_pid="${current_child_pid:-}"
  local wait_pid="${current_wait_pid:-}"
  local file_pid="" target_pid
  if [[ -n "${current_child_pid_file:-}" && -s "${current_child_pid_file}" ]]; then
    IFS= read -r file_pid < "${current_child_pid_file}" || true
  fi
  rm -f "${current_child_pid_file:-}" 2>/dev/null || true

  [[ -n "${base_pid}" || -n "${file_pid}" || -n "${wait_pid}" ]] || return 0

  for target_pid in "${base_pid}" "${file_pid}" "${wait_pid}"; do
    [[ -n "${target_pid}" ]] || continue
    kill -INT "-${target_pid}" 2>/dev/null || true
    kill -INT "${target_pid}" 2>/dev/null || true
  done

  if [[ "${INTERRUPT_GRACE_SECONDS}" != "0" ]]; then
    sleep "${INTERRUPT_GRACE_SECONDS}" || true
  fi

  for target_pid in "${base_pid}" "${file_pid}" "${wait_pid}"; do
    [[ -n "${target_pid}" ]] || continue
    child_process_alive "${target_pid}" || continue
    kill -TERM "-${target_pid}" 2>/dev/null || true
    kill -TERM "${target_pid}" 2>/dev/null || true
  done

  if [[ "${INTERRUPT_GRACE_SECONDS}" != "0" ]]; then
    sleep "${INTERRUPT_GRACE_SECONDS}" || true
  fi

  for target_pid in "${base_pid}" "${file_pid}" "${wait_pid}"; do
    [[ -n "${target_pid}" ]] || continue
    child_process_alive "${target_pid}" || continue
    kill -KILL "-${target_pid}" 2>/dev/null || true
    kill -KILL "${target_pid}" 2>/dev/null || true
  done
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
  if command -v setsid >/dev/null 2>&1 && setsid --help 2>&1 | grep -q -- '--wait'; then
    pid_file="$(mktemp "${TMPDIR:-/tmp}/profile_lora_lf_child.XXXXXX")"
    current_child_pid_file="${pid_file}"
    setsid --wait bash -c 'pid_file="$1"; shift; echo "$$" > "${pid_file}"; exec "$@"' _ "${pid_file}" "$@" &
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
  run_tracked_command bash -o pipefail -c 'log_file="$1"; shift; "$@" 2>&1 | tee -a "${log_file}"' _ "${log_file}" "$@"
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
  local dataset_name="$8"
  local gradient_checkpointing=false
  [[ "${recompute}" == "recomp" ]] && gradient_checkpointing=true

  local config_root job_root seq_root source_profile lf_out log_file run_id profile_json
  config_root="$(config_root_path "${seq_len}")"
  job_root="$(job_root_path "${config_root}" "${backend}" "${profiler}" "${recompute}" "${expert_policy}")"
  seq_root="${job_root}/s${seq_len}"
  source_profile="${seq_root}/source_profile.json"
  lf_out="${seq_root}/lf_run"
  log_file="${seq_root}/train.log"
  run_id="lf_${backend}_${profiler}_${recompute}_pol${expert_policy}_s${seq_len}_${lora_dropout_label_value}"
  profile_json="${seq_root}/profile.json"
  local group_key="${config_root}|${profiler}|${recompute}|${expert_policy}|${seq_len}"
  local profile_memory_attribution profile_memory_breakdown
  profile_memory_attribution="$(memory_attribution_for_profiler "${profiler}")"
  profile_memory_breakdown="$(memory_breakdown_for_profiler "${profiler}")"

  plot_roots["${config_root}"]="${seq_len}"
  if [[ "${profile_memory_breakdown}" == "true" ]]; then
    memory_plot_roots["${config_root}"]="${seq_len}"
  fi
  if [[ -z "${compare_groups[${group_key}]+set}" ]]; then
    compare_groups["${group_key}"]=1
    compare_group_keys+=("${group_key}")
    compare_group_config_roots["${group_key}"]="${config_root}"
    compare_group_labels["${group_key}"]="dropout=${LORA_DROPOUT} profiler=${profiler} recompute=${recompute} expert_policy=${expert_policy} seq_len=${seq_len}"
  fi

  if [[ "${DRY_RUN}" != "true" && -e "${profile_json}" && "${OVERWRITE}" != "true" && "${COLLECT_EXISTING}" != "true" ]]; then
    echo "Skipping existing: ${profile_json}"
    run_dirs["${group_key}|${backend}"]="${lf_out}"
    append_job_record "${config_root}" skipped \
      "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${backend}" "${profiler}" "${seq_root}" "${profile_json}" "${log_file}"
    return 0
  fi

  if [[ "${DRY_RUN}" != "true" && "${COLLECT_EXISTING}" == "true" ]]; then
    if [[ -e "${profile_json}" ]]; then
      echo "Found existing: ${profile_json}"
      run_dirs["${group_key}|${backend}"]="${lf_out}"
      return 0
    fi
    echo "Missing existing profile: ${profile_json}" >&2
    return 1
  fi

  local -a run_env=(
    ROOT="${ROOT}"
    LF_DIR="${LF_DIR}"
    ASYM_DIR="${ASYM_DIR}"
    BACKENDS=
    ENV_DIR="${ENV_DIR}"
    CONDA_EXE="${CONDA_EXE}"
    NSYS_BIN="${NSYS_BIN}"
    MODEL_NAME_OR_PATH="${current_model_name}"
    BACKEND="${backend}"
    GPU_ID="${gpu}"
    NUM_GPUS="${gpu_count}"
    REQUIRE_SM100="${REQUIRE_SM100}"
    TORCH_DISTRIBUTED_BACKEND="${TORCH_DISTRIBUTED_BACKEND}"
    TORCH_FSDP_CONFIG="${TORCH_FSDP_CONFIG}"
    TORCH_DEEPSPEED_CONFIG="${TORCH_DEEPSPEED_CONFIG}"
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
    GRADIENT_CHECKPOINTING="${gradient_checkpointing}"
    ASYM_PRECISION="${PRECISION}"
    ASYM_OFFLOAD_MODULES="${ASYM_OFFLOAD_MODULES}"
    ASYM_EXPERT_RECOMPUTE_POLICY="${expert_policy}"
    ASYM_STRICT="${ASYM_STRICT}"
    TORCH_USE_ASYM_GEMM_LORA="${TORCH_USE_ASYM_GEMM_LORA}"
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
    PROFILE_JSON="${profile_json}"
    PROFILE_OUTPUT_DIR="${seq_root}"
    PROFILE_SUMMARY_MD="${seq_root}/summary.md"
    PROFILE_WORKLOAD_LABEL="${workload_label}"
    PROFILE_BACKEND_LABEL="${backend}"
    PROFILE_EXPERT_POLICY="${expert_policy}"
    PROFILE_WARMUP_STEPS="${WARMUP_STEPS}"
    PROFILE_MEASURE_STEPS="${MAX_STEPS}"
    PROFILE_TOTAL_STEPS="${TOTAL_STEPS}"
    ASYM_GEMM_LF_CONFIG_WARMUP_STEPS="${WARMUP_STEPS}"
    ASYM_GEMM_LF_CONFIG_MEASURE_STEPS="${MAX_STEPS}"
    ASYM_GEMM_LF_CONFIG_TOTAL_STEPS="${TOTAL_STEPS}"
    OUT_DIR="${lf_out}"
    LOG_FILE="${log_file}"
    LOSS_LOG_COPY="${seq_root}/loss.trainer_log.jsonl"
    RUN_ID="${run_id}"
  )

  local -a run_cmd=(env "${run_env[@]}" "${RUN_LF_SCRIPT}")

  echo "Running backend=${backend} profiler=${profiler} recompute=${recompute} expert_policy=${expert_policy} seq=${seq_len} lora_dropout=${LORA_DROPOUT} gpu=${gpu} num_gpus=${gpu_count}"
  echo "  dir=${seq_root}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    print_command "${run_cmd[@]}"
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
    run_dirs["${group_key}|${backend}"]="${lf_out}"
    append_job_record "${config_root}" ok \
      "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${backend}" "${profiler}" "${seq_root}" "${profile_json}" "${log_file}"
    if profiler_selected_for_plots "${profiler}"; then
      plot_single_run "${config_root}" "${seq_len}" "${backend}" "${profiler}" "${recompute}" "${expert_policy}" "${seq_root}"
      plot_running_combined "${config_root}" "${seq_len}" "${seq_root}"
    fi
    if [[ "${profile_memory_breakdown}" == "true" ]]; then
      plot_memory_single_run "${seq_root}"
      plot_memory_running_combined "${config_root}" "${seq_len}" "${seq_root}"
    fi
  else
    append_job_record "${config_root}" "failed:${status}" \
      "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${backend}" "${profiler}" "${seq_root}" "${profile_json}" "${log_file}"
  fi
  return "${status}"
}

baseline_selected=false
candidate_selected=false
for backend in "${backends[@]}"; do
  [[ "${backend}" == "${compare_baseline_backend}" ]] && baseline_selected=true
  [[ "${backend}" == "${compare_candidate_backend}" ]] && candidate_selected=true
done

compare_config_root() {
  local target_config_root="$1"
  local group_key baseline_dir candidate_dir config_root compare_dir compare_tsv compare_log status
  local group_tail group_expert_policy

  [[ "${COMPARE_LOSSES}" == "true" ]] || return 0
  if [[ "${baseline_selected}" != "true" || "${candidate_selected}" != "true" ]]; then
    echo "Skipping loss comparison because selected backends do not include both ${compare_baseline_backend} and ${compare_candidate_backend}."
    return 0
  fi

  for group_key in "${compare_group_keys[@]}"; do
    config_root="${compare_group_config_roots[${group_key}]}"
    [[ "${config_root}" == "${target_config_root}" ]] || continue
    group_tail="${group_key%|*}"
    group_expert_policy="${group_tail##*|}"
    if [[ "${group_expert_policy}" != "none" && ( "${compare_baseline_backend}" == torch* || "${compare_candidate_backend}" == torch* ) ]]; then
      echo "Skipping loss comparison for ${compare_group_labels[${group_key}]}; torch baseline is policy-independent."
      continue
    fi
    baseline_dir="${run_dirs[${group_key}|${compare_baseline_backend}]-}"
    candidate_dir="${run_dirs[${group_key}|${compare_candidate_backend}]-}"
    compare_dir="${config_root}/comparisons"
    compare_tsv="${compare_dir}/loss_compare.tsv"
    compare_log="${compare_dir}/$(safe_label "${compare_group_labels[${group_key}]} ${compare_baseline_backend} vs ${compare_candidate_backend}").log"
    mkdir -p "${compare_dir}"
    if [[ ! -e "${compare_tsv}" ]]; then
      printf 'status\tbaseline_backend\tcandidate_backend\tbaseline_dir\tcandidate_dir\tlog\tlabel\n' > "${compare_tsv}"
    fi

    if [[ -z "${baseline_dir}" || -z "${candidate_dir}" ]]; then
      echo "Missing loss comparison input for ${compare_group_labels[${group_key}]}" | tee "${compare_log}"
      printf 'missing\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${compare_baseline_backend}" "${compare_candidate_backend}" "${baseline_dir}" "${candidate_dir}" "${compare_log}" "${compare_group_labels[${group_key}]}" \
        >> "${compare_tsv}"
      failures=$((failures + 1))
      if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
        exit 1
      fi
      continue
    fi

    echo "Comparing losses: ${compare_group_labels[${group_key}]} ${compare_baseline_backend} vs ${compare_candidate_backend}" | tee "${compare_log}"
    local -a compare_cmd=(
      "${CONDA_EXE}" run -p "${ENV_DIR}" python "${PROFILE_POSTPROCESS_SCRIPT}"
      --baseline-dir "${baseline_dir}"
      --candidate-dir "${candidate_dir}"
      --min-steps "${COMPARE_MIN_STEPS}"
      --warmup-steps "${WARMUP_STEPS}"
      --first-step-rel-tol "${COMPARE_FIRST_STEP_REL_TOL}"
      --max-rel-tol "${COMPARE_MAX_REL_TOL}"
    )
    if run_tracked_command_logged "${compare_log}" "${compare_cmd[@]}"; then
      printf 'ok\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${compare_baseline_backend}" "${compare_candidate_backend}" "${baseline_dir}" "${candidate_dir}" "${compare_log}" "${compare_group_labels[${group_key}]}" \
        >> "${compare_tsv}"
    else
      status=$?
      printf 'failed:%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${status}" "${compare_baseline_backend}" "${compare_candidate_backend}" "${baseline_dir}" "${candidate_dir}" "${compare_log}" "${compare_group_labels[${group_key}]}" \
        >> "${compare_tsv}"
      failures=$((failures + 1))
      if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
        exit 1
      fi
    fi
  done
}

plot_config_root() {
  local config_root="$1"
  local seq_len="$2"
  local plot_root
  [[ "${PLOT}" == "true" ]] || return 0
  ((${#plot_profilers[@]})) || return 0

  plot_root="${config_root}/plots"
  [[ -n "${PLOT_OUTPUT_DIR}" ]] && plot_root="$(abs_path "${PLOT_OUTPUT_DIR}")/$(basename "${config_root}")"
  local -a plot_cmd
  plot_cmd_base plot_cmd "${config_root}" "${plot_root}" "${plot_root}" "${seq_len}"
  plot_cmd+=(--expert-recompute-policies "${expert_policies[@]}")
  append_sweep_plot_filters plot_cmd
  echo "Writing LF profile plots: ${plot_root}"
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
  local seq_root="$7"
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
  )
  echo "Writing LF per-run plots: ${plot_root}"
  if ! run_tracked_command "${plot_cmd[@]}"; then
    echo "warning: failed to write per-run plots for ${seq_root}" >&2
  fi
}

memory_plot_filters() {
  local -n cmd_ref="$1"
  local backend
  for backend in "${backends[@]}"; do cmd_ref+=(--backend "${backend}"); done
  cmd_ref+=(--profiler source)
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
  local -a plot_cmd=(
    "${CONDA_EXE}" run -p "${ENV_DIR}" python "${MEMORY_PLOT_SCRIPT}"
    --input-root "${config_root}"
    --output-dir "${plot_root}"
    --clean-output
    --combined-only
    --y-scale "${MEMORY_BREAKDOWN_PLOT_Y_SCALE}"
    --seq-lens "${seq_len}"
    --expert-recompute-policies "${expert_policies[@]}"
  )
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
  local -a plot_cmd=(
    "${CONDA_EXE}" run -p "${ENV_DIR}" python "${MEMORY_PLOT_SCRIPT}"
    --input-root "${config_root}"
    --output-dir "${plot_root}"
    --clean-output
    --combined-only
    --y-scale "${MEMORY_BREAKDOWN_PLOT_Y_SCALE}"
    --seq-lens "${seq_len}"
    --expert-recompute-policies "${expert_policies[@]}"
  )
  memory_plot_filters plot_cmd
  echo "Writing LF source-memory combined plots: ${plot_root}"
  if ! run_tracked_command "${plot_cmd[@]}"; then
    echo "warning: failed to write source-memory combined plots for ${config_root}" >&2
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
  echo "Using template: ${TEMPLATE} for model ${current_model_name} requesting ${current_model_gpu_count} GPU(s)"

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
      for expert_policy in "${expert_policies[@]}"; do
        for backend_recompute in "${backend_specs[@]}"; do
          backend="${backend_recompute%%|*}"
          recompute="${backend_recompute##*|}"
          for profiler in "${profilers[@]}"; do
            if [[ "${recompute}" == "recomp" && "${expert_policy}" != "none" ]]; then
              echo "Skipping backend=${backend} expert_policy=${expert_policy} recompute=recomp; LF gradient checkpointing is only swept for expert_policy=none."
              continue
            fi
            if [[ "${backend}" == torch* && "${expert_policy}" != "none" ]]; then
              echo "Skipping backend=${backend} expert_policy=${expert_policy}; torch baseline is policy-independent."
              continue
            fi
            gpu_count="$(backend_gpu_count "${backend}" "${current_model_gpu_count}")"
            gpu="$(gpu_slice "${gpu_count}")"
            if ! run_job "${backend}" "${profiler}" "${recompute}" "${seq_len}" "${gpu}" "${gpu_count}" "${expert_policy}" "${current_dataset}"; then
              failures=$((failures + 1))
              if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
                exit 1
              fi
            fi
          done
        done
      done
      if [[ "${DRY_RUN}" != "true" ]]; then
        compare_config_root "${config_root}"
        plot_config_root "${config_root}" "${seq_len}"
        if [[ -n "${memory_plot_roots[${config_root}]+set}" ]]; then
          plot_memory_config_root "${config_root}" "${seq_len}"
        fi
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
  if ((${#plot_profilers[@]})); then
  combined_plot_root="${precision_root}/combined"
  [[ -n "${PLOT_OUTPUT_DIR}" ]] && combined_plot_root="$(abs_path "${PLOT_OUTPUT_DIR}")/combined"
  declare -A combined_workloads=()
  declare -A combined_workload_bases=()
  for config_root in "${!plot_roots[@]}"; do
    combined_workloads["$(plot_workload_from_config_root "${config_root}")"]=1
    combined_workload_bases["$(plot_workload_base_from_config_root "${config_root}")"]=1
  done
  combined_plot_cmd=(
    "${CONDA_EXE}" run -p "${ENV_DIR}" python "${PLOT_SCRIPT}"
    --input-root "${precision_root}"
    --output-dir "${combined_plot_root}"
    --combined-output-dir "${combined_plot_root}"
    --precision "${PRECISION}"
    --clean-output
    --combined-only
    --batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --seq-lens "${seq_lens[@]}"
    --expert-recompute-policies "${expert_policies[@]}"
  )
  for workload in "${!combined_workloads[@]}"; do combined_plot_cmd+=(--workload "${workload}"); done
  for backend in "${backends[@]}"; do combined_plot_cmd+=(--backend "${backend}"); done
  for profiler in "${plot_profilers[@]}"; do combined_plot_cmd+=(--profiler "${profiler}"); done
  for recompute in "${recompute_modes[@]}"; do combined_plot_cmd+=(--recompute "${recompute}"); done
  echo "Writing combined LF profile plots: ${combined_plot_root}"
  run_tracked_command "${combined_plot_cmd[@]}"

  for workload_base in "${!combined_workload_bases[@]}"; do
    model_combined_plot_root="${combined_plot_root}/$(safe_label "${workload_base}")"
    model_combined_plot_cmd=(
      "${CONDA_EXE}" run -p "${ENV_DIR}" python "${PLOT_SCRIPT}"
      --input-root "${precision_root}"
      --output-dir "${model_combined_plot_root}"
      --combined-output-dir "${model_combined_plot_root}"
      --precision "${PRECISION}"
      --clean-output
      --combined-only
      --batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
      --seq-lens "${seq_lens[@]}"
      --expert-recompute-policies "${expert_policies[@]}"
      --workload "${workload_base}"
    )
    for backend in "${backends[@]}"; do model_combined_plot_cmd+=(--backend "${backend}"); done
    for profiler in "${plot_profilers[@]}"; do model_combined_plot_cmd+=(--profiler "${profiler}"); done
    for recompute in "${recompute_modes[@]}"; do model_combined_plot_cmd+=(--recompute "${recompute}"); done
    echo "Writing model-split combined LF profile plots: ${model_combined_plot_root}"
    run_tracked_command "${model_combined_plot_cmd[@]}"
  done
  fi

  if [[ "${PLOT_MEMORY_BREAKDOWN}" == "true" && "${#memory_plot_roots[@]}" -gt 0 ]]; then
    combined_memory_plot_root="${precision_root}/memory_combined"
    [[ -n "${PLOT_OUTPUT_DIR}" ]] && combined_memory_plot_root="$(abs_path "${PLOT_OUTPUT_DIR}")/memory_combined"
    declare -A memory_combined_workload_bases=()
    for config_root in "${!memory_plot_roots[@]}"; do
      memory_combined_workload_bases["$(plot_workload_base_from_config_root "${config_root}")"]=1
    done
    combined_memory_plot_cmd=(
      "${CONDA_EXE}" run -p "${ENV_DIR}" python "${MEMORY_PLOT_SCRIPT}"
      --input-root "${precision_root}"
      --output-dir "${combined_memory_plot_root}"
      --clean-output
      --combined-only
      --y-scale "${MEMORY_BREAKDOWN_PLOT_Y_SCALE}"
      --seq-lens "${seq_lens[@]}"
      --expert-recompute-policies "${expert_policies[@]}"
    )
    memory_plot_filters combined_memory_plot_cmd
    echo "Writing combined LF source-memory plots: ${combined_memory_plot_root}"
    run_tracked_command "${combined_memory_plot_cmd[@]}"

    for workload_base in "${!memory_combined_workload_bases[@]}"; do
      model_memory_plot_root="${combined_memory_plot_root}/$(safe_label "${workload_base}")"
      model_memory_plot_cmd=(
        "${CONDA_EXE}" run -p "${ENV_DIR}" python "${MEMORY_PLOT_SCRIPT}"
        --input-root "${precision_root}"
        --output-dir "${model_memory_plot_root}"
        --clean-output
        --combined-only
        --y-scale "${MEMORY_BREAKDOWN_PLOT_Y_SCALE}"
        --seq-lens "${seq_lens[@]}"
        --expert-recompute-policies "${expert_policies[@]}"
        --workload "${workload_base}"
      )
      memory_plot_filters model_memory_plot_cmd
      echo "Writing model-split combined LF source-memory plots: ${model_memory_plot_root}"
      run_tracked_command "${model_memory_plot_cmd[@]}"
    done
  fi
fi

echo "LF profiling completed. Results: ${precision_root}"
