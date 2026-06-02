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
# MODEL_SPECS=${MODEL_SPECS:-meta-llama/Llama-4-Scout-17B-16E|2}
# MODEL_SPECS=${MODEL_SPECS:-Qwen/Qwen3-30B-A3B|1}
# MODEL_SPECS=${MODEL_SPECS:-Qwen/Qwen3-235B-A22B|2}
# MODEL_SPECS=${MODEL_SPECS:-google/gemma-4-26B-A4B|1}
MODEL_SPECS=${MODEL_SPECS:-"Qwen/Qwen3-30B-A3B|1 meta-llama/Llama-4-Scout-17B-16E|2"}
# MODEL_SPECS=${MODEL_SPECS:-"meta-llama/Llama-4-Scout-17B-16E|2 google/gemma-4-26B-A4B|1"}

# BACKENDS=${BACKENDS:-asym,torch}
BACKENDS=${BACKENDS:-torch}
PROFILERS=${PROFILERS:-nsys}
PRECISION=${PRECISION:-bf16}

DATASET=${DATASET:-asym_long_sft_smoke}
TEMPLATE=${TEMPLATE:-auto}
SEQ_LENS=${SEQ_LENS:-4096}
MAX_SAMPLES=${MAX_SAMPLES:-128}
MAX_STEPS=${MAX_STEPS:-10}
WARMUP_STEPS=${WARMUP_STEPS:-5}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
RECOMPUTE=${RECOMPUTE:-norecomp}
# EXPERT_POLICIES=${EXPERT_POLICIES-"tok-le0 tok-le256 tok-le512 tok-le1024 tok-le2048"}
# EXPERT_POLICIES=${EXPERT_POLICIES-"tok-le0 tok-le512 tok-le1024 tok-le2048 tok-le0-act tok-le512-act tok-le1024-act tok-le2048-act tok-le256 tok-le256-act"}
EXPERT_POLICIES=${EXPERT_POLICIES-"none"}


ASYM_OFFLOAD_MODULES=${ASYM_OFFLOAD_MODULES:-routed_experts}
ASYM_STRICT=${ASYM_STRICT:-true}
REQUIRE_SM100=${REQUIRE_SM100:-1}
TORCH_DISTRIBUTED_BACKEND=${TORCH_DISTRIBUTED_BACKEND:-deepspeed}
TORCH_FSDP_CONFIG=${TORCH_FSDP_CONFIG:-${LF_DIR}/examples/accelerate/fsdp2_config.yaml}
TORCH_DEEPSPEED_CONFIG=${TORCH_DEEPSPEED_CONFIG:-${LF_DIR}/examples/deepspeed/ds_z3_offload_config.json}

# Optional output/profile controls
OUTPUT_ROOT=${OUTPUT_ROOT:-profiling}
PROFILE_LEVEL=${PROFILE_LEVEL:-op}
PROFILE_LAYERS=${PROFILE_LAYERS:-all}
PROFILE_MEMORY_ATTRIBUTION=${PROFILE_MEMORY_ATTRIBUTION:-0}
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

# Optional execution controls
OVERWRITE=${OVERWRITE:-false}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-true}
DRY_RUN=${DRY_RUN:-false}
COLLECT_EXISTING=${COLLECT_EXISTING:-false}

# Empty optional user parameters
RUN_NAME=${RUN_NAME:-}
COMPARE_MIN_STEPS=${COMPARE_MIN_STEPS:-}
PLOT_OUTPUT_DIR=${PLOT_OUTPUT_DIR:-}

# =============================================================================
# Derived Parameters
# =============================================================================
ASYM_DIR=${ASYM_DIR:-${ROOT}}
ENV_DIR=${ENV_DIR:-${LF_DIR}/.venv}
RUN_LF_SCRIPT="${ASYM_DIR}/scripts/lf/run_lf_lora_sft.sh"
PROFILE_POSTPROCESS_SCRIPT="${ASYM_DIR}/scripts/lf/postprocess_lf_profile_artifacts.py"
PLOT_SCRIPT="${ASYM_DIR}/scripts/plotting/plot_activation_recompute_sweep.py"

# =============================================================================
# Main Logic
# =============================================================================
usage() {
  cat <<USAGE
Usage:
  scripts/lf/profile_lora_lf.sh [options]

Defaults:
  --gpus ${GPU_POOL}
  --backends ${BACKENDS}
  --profilers ${PROFILERS}
  --seq-lens ${SEQ_LENS}
  --recompute ${RECOMPUTE}
  --output-root ${OUTPUT_ROOT}

Options:
  --gpus LIST                    Physical GPU pool. Accepts 0,1 or "0 1".
  --models LIST                  Model specs. Each item is model_name_or_path|num_gpus.
                                 Accepts comma/space-separated values, for example:
                                 meta-llama/Llama-4-Scout-17B-16E|1 meta-llama/Llama-4-Maverick-17B-128E|4
  --backends LIST                asym and/or torch. torch uses pure LF/HF torch; asym uses AsymGEMM.
  --profilers LIST               source and/or nsys.
  --seq-lens LIST                LF cutoff lengths. Accepts 2048,4096 or "2048 4096".
  --recompute norecomp|recomp|both
  --expert-policies LIST         AsymGEMM expert policies: none, tok-le0, tok-le0-act, tok-leN, tok-geN, tokA-B, and -act variants.
  --dataset NAME
  --template NAME
  --max-samples N
  --max-steps N                 Measured steps kept in plots/summaries.
  --warmup-steps N              Extra initial steps to run but exclude from plots/summaries.
  --batch-size N
  --gradient-accumulation-steps N
  --learning-rate VALUE
  --lora-rank N
  --lora-alpha VALUE
  --lora-dropout VALUE
  --precision NAME
  --profile-level stage|module|op|deep
  --profile-layers all|first,last|0,1,2|every4
  --profile-memory-attribution true|false
  --profile-sync true|false
  --profile-module-filter LIST
  --torch-distributed-backend fsdp2|deepspeed|ddp
  --torch-fsdp-config PATH       Accelerate config for torch FSDP2. Defaults to LF's examples/accelerate/fsdp2_config.yaml.
  --torch-deepspeed-config PATH  DeepSpeed config for torch backend. Defaults to LF's examples/deepspeed/ds_z3_config.json.
  --compare-losses true|false
  --compare-baseline-backend torch|asym
  --compare-candidate-backend torch|asym
  --compare-min-steps N
  --compare-first-step-rel-tol VALUE
  --compare-max-rel-tol VALUE
  --output-root DIR              Default layout: <root>/lora_lf_<precision>/<model>__b<batch>_s<seq>_r<rank>_a<alpha>
  --run-name NAME                Optional config directory under lora_lf_<precision>.
  --plot true|false
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

recompute_values() {
  case "${1,,}" in
    norecomp) printf 'norecomp\n' ;;
    recomp) printf 'recomp\n' ;;
    both) printf 'norecomp\nrecomp\n' ;;
    *) die "expected recompute mode norecomp, recomp, or both; got '${1}'" ;;
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

lf_backend() {
  case "${1}" in
    asym) printf 'asym\n' ;;
    torch) printf 'torch\n' ;;
    *) die "internal backend label must be asym or torch, got '${1}'" ;;
  esac
}

profiler_label() {
  case "${1,,}" in
    source|nsys) printf '%s\n' "${1,,}" ;;
    *) die "profiler must be source or nsys, got '${1}'" ;;
  esac
}

config_root_path() {
  local seq_len="$1"
  local config_label step_label
  step_label="w${WARMUP_STEPS}_s${MAX_STEPS}"
  if [[ -n "${run_name}" ]]; then
    if ((${#seq_lens[@]} > 1)); then
      config_label="$(safe_label "${run_name}__s${seq_len}")"
    else
      config_label="$(safe_label "${run_name}")"
    fi
  else
    config_label="$(safe_label "${workload_label}__b${batch_size}_s${seq_len}_${step_label}_r${LORA_RANK}_a${LORA_ALPHA}")"
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
  local config_name
  config_name="$(basename "$1")"
  printf '%s\n' "${config_name%%__*}"
}

print_command() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

gpu_spec="${GPU_POOL}"
model_spec="${MODEL_SPECS}"
backend_spec="${BACKENDS}"
profiler_spec="${PROFILERS}"
seq_spec="${SEQ_LENS}"
recompute_spec="${RECOMPUTE}"
expert_policy_spec="${EXPERT_POLICIES}"
output_root="${OUTPUT_ROOT}"
run_name="${RUN_NAME}"
batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}"
template_spec="${TEMPLATE}"

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --gpus) need_value "$1" "${2-}"; gpu_spec="$2"; shift 2 ;;
    --gpus=*) gpu_spec="${1#*=}"; shift ;;
    --models|--model-specs) collect_values "$1" vals "${@:2}"; model_spec="${vals[*]}"; set -- "${REMAINING[@]}" ;;
    --models=*|--model-specs=*) model_spec="${1#*=}"; shift ;;
    --backends) need_value "$1" "${2-}"; backend_spec="$2"; shift 2 ;;
    --backends=*) backend_spec="${1#*=}"; shift ;;
    --profilers) need_value "$1" "${2-}"; profiler_spec="$2"; shift 2 ;;
    --profilers=*) profiler_spec="${1#*=}"; shift ;;
    --seq-lens) need_value "$1" "${2-}"; seq_spec="$2"; shift 2 ;;
    --seq-lens=*) seq_spec="${1#*=}"; shift ;;
    --recompute) need_value "$1" "${2-}"; recompute_spec="$2"; shift 2 ;;
    --recompute=*) recompute_spec="${1#*=}"; shift ;;
    --expert-policies) need_value "$1" "${2-}"; expert_policy_spec="$2"; shift 2 ;;
    --expert-policies=*) expert_policy_spec="${1#*=}"; shift ;;
    --dataset) need_value "$1" "${2-}"; DATASET="$2"; shift 2 ;;
    --dataset=*) DATASET="${1#*=}"; shift ;;
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
    --lora-dropout) need_value "$1" "${2-}"; LORA_DROPOUT="$2"; shift 2 ;;
    --lora-dropout=*) LORA_DROPOUT="${1#*=}"; shift ;;
    --precision) need_value "$1" "${2-}"; PRECISION="$2"; shift 2 ;;
    --precision=*) PRECISION="${1#*=}"; shift ;;
    --profile-level) need_value "$1" "${2-}"; PROFILE_LEVEL="$2"; shift 2 ;;
    --profile-level=*) PROFILE_LEVEL="${1#*=}"; shift ;;
    --profile-layers) need_value "$1" "${2-}"; PROFILE_LAYERS="$2"; shift 2 ;;
    --profile-layers=*) PROFILE_LAYERS="${1#*=}"; shift ;;
    --profile-memory-attribution) need_value "$1" "${2-}"; PROFILE_MEMORY_ATTRIBUTION="$2"; shift 2 ;;
    --profile-memory-attribution=*) PROFILE_MEMORY_ATTRIBUTION="${1#*=}"; shift ;;
    --profile-sync) need_value "$1" "${2-}"; PROFILE_SYNC="$2"; shift 2 ;;
    --profile-sync=*) PROFILE_SYNC="${1#*=}"; shift ;;
    --profile-module-filter) need_value "$1" "${2-}"; PROFILE_MODULE_FILTER="$2"; shift 2 ;;
    --profile-module-filter=*) PROFILE_MODULE_FILTER="${1#*=}"; shift ;;
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

nonnegative_int "--max-steps" "${MAX_STEPS}"
nonnegative_int "--warmup-steps" "${WARMUP_STEPS}"
TOTAL_STEPS=$((MAX_STEPS + WARMUP_STEPS))
if [[ -z "${COMPARE_MIN_STEPS}" ]]; then
  COMPARE_MIN_STEPS="${MAX_STEPS}"
fi
nonnegative_int "--compare-min-steps" "${COMPARE_MIN_STEPS}"

mapfile -t gpus < <(tokens "${gpu_spec}" | sed 's/^cuda://' | dedupe)
mapfile -t model_specs < <(tokens "${model_spec}" | dedupe)
mapfile -t backends < <(tokens "${backend_spec}" | while read -r value; do backend_label "${value}"; done | dedupe)
mapfile -t profilers < <(tokens "${profiler_spec}" | while read -r value; do profiler_label "${value}"; done | dedupe)
mapfile -t seq_lens < <(tokens "${seq_spec}" | dedupe)
mapfile -t recompute_modes < <(recompute_values "${recompute_spec}")
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
((${#backends[@]})) || die "backend list is empty"
((${#profilers[@]})) || die "profiler list is empty"
((${#seq_lens[@]})) || die "sequence length list is empty"
((${#expert_policies[@]})) || die "expert policy list is empty"
for recompute in "${recompute_modes[@]}"; do
  [[ "${recompute}" == "recomp" ]] || continue
  for expert_policy in "${expert_policies[@]}"; do
    [[ "${expert_policy}" == "none" ]] && continue
    die "expert policy '${expert_policy}' conflicts with --recompute ${recompute_spec}; use --recompute norecomp when sweeping expert policies"
  done
done
[[ -f "${RUN_LF_SCRIPT}" ]] || die "missing ${RUN_LF_SCRIPT}"
[[ -f "${PROFILE_POSTPROCESS_SCRIPT}" ]] || die "missing ${PROFILE_POSTPROCESS_SCRIPT}"
[[ -f "${PLOT_SCRIPT}" ]] || die "missing ${PLOT_SCRIPT}"
if [[ "${TORCH_DISTRIBUTED_BACKEND}" == "fsdp2" ]]; then
  [[ -f "${TORCH_FSDP_CONFIG}" ]] || die "missing torch FSDP2 accelerate config ${TORCH_FSDP_CONFIG}"
fi
if [[ "${TORCH_DISTRIBUTED_BACKEND}" == "deepspeed" ]]; then
  [[ -f "${TORCH_DEEPSPEED_CONFIG}" ]] || die "missing torch DeepSpeed config ${TORCH_DEEPSPEED_CONFIG}"
fi
COMPARE_LOSSES=$(bool_value "${COMPARE_LOSSES}")
compare_baseline_backend="$(backend_label "${COMPARE_BASELINE_BACKEND}")"
compare_candidate_backend="$(backend_label "${COMPARE_CANDIDATE_BACKEND}")"
[[ "${compare_baseline_backend}" != "${compare_candidate_backend}" ]] || die "compare backends must differ"

base_output_root="$(abs_path "${output_root}")"
precision_label="$(safe_label "${PRECISION}")"
precision_root="${base_output_root}/lora_lf_${precision_label}"
mkdir -p "${precision_root}"
echo "Output precision root: ${precision_root}"

declare -A plot_roots=()
declare -A run_dirs=()
declare -A compare_groups=()
declare -A compare_group_config_roots=()
declare -A compare_group_labels=()
declare -a compare_group_keys=()
failures=0
interrupted=false
current_child_pid=""

handle_interrupt() {
  interrupted=true
  echo "Interrupted; stopping LF profiling sweep." >&2
  if [[ -n "${current_child_pid}" ]]; then
    kill -INT "-${current_child_pid}" 2>/dev/null || true
    kill -INT "${current_child_pid}" 2>/dev/null || true
  fi
}

trap handle_interrupt INT TERM

run_job() {
  local backend="$1"
  local profiler="$2"
  local recompute="$3"
  local seq_len="$4"
  local gpu="$5"
  local gpu_count="$6"
  local expert_policy="$7"
  local gradient_checkpointing=false
  [[ "${recompute}" == "recomp" ]] && gradient_checkpointing=true

  local config_root job_root seq_root source_profile lf_out log_file run_id profile_json
  config_root="$(config_root_path "${seq_len}")"
  job_root="$(job_root_path "${config_root}" "${backend}" "${profiler}" "${recompute}" "${expert_policy}")"
  seq_root="${job_root}/s${seq_len}"
  source_profile="${seq_root}/source_profile.json"
  lf_out="${seq_root}/lf_run"
  log_file="${seq_root}/train.log"
  run_id="lf_${backend}_${profiler}_${recompute}_pol${expert_policy}_s${seq_len}"
  profile_json="${seq_root}/profile.json"
  local group_key="${config_root}|${profiler}|${recompute}|${expert_policy}|${seq_len}"

  plot_roots["${config_root}"]="${seq_len}"
  if [[ -z "${compare_groups[${group_key}]+set}" ]]; then
    compare_groups["${group_key}"]=1
    compare_group_keys+=("${group_key}")
    compare_group_config_roots["${group_key}"]="${config_root}"
    compare_group_labels["${group_key}"]="profiler=${profiler} recompute=${recompute} expert_policy=${expert_policy} seq_len=${seq_len}"
  fi

  if [[ -e "${profile_json}" && "${OVERWRITE}" != "true" && "${COLLECT_EXISTING}" != "true" ]]; then
    echo "Skipping existing: ${profile_json}"
    run_dirs["${group_key}|${backend}"]="${lf_out}"
    mkdir -p "${config_root}"
    if [[ ! -e "${config_root}/jobs.tsv" ]]; then
      printf 'status\tgpu\tseq_len\trecompute\texpert_policy\tbackend\tprofiler\tjob_dir\tprofile_json\tlog\n' > "${config_root}/jobs.tsv"
    fi
    printf 'skipped\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${backend}" "${profiler}" "${seq_root}" "${profile_json}" "${log_file}" \
      >> "${config_root}/jobs.tsv"
    return 0
  fi

  if [[ "${COLLECT_EXISTING}" == "true" ]]; then
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
    BACKEND="$(lf_backend "${backend}")"
    GPU_ID="${gpu}"
    NUM_GPUS="${gpu_count}"
    REQUIRE_SM100="${REQUIRE_SM100}"
    TORCH_DISTRIBUTED_BACKEND="${TORCH_DISTRIBUTED_BACKEND}"
    TORCH_FSDP_CONFIG="${TORCH_FSDP_CONFIG}"
    TORCH_DEEPSPEED_CONFIG="${TORCH_DEEPSPEED_CONFIG}"
    DATASET="${DATASET}"
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
    PROFILE=1
    PROFILE_PROFILER="${profiler}"
    PROFILE_LEVEL="${PROFILE_LEVEL}"
    PROFILE_LAYERS="${PROFILE_LAYERS}"
    PROFILE_MEMORY_ATTRIBUTION="${PROFILE_MEMORY_ATTRIBUTION}"
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

  echo "Running backend=${backend} profiler=${profiler} recompute=${recompute} expert_policy=${expert_policy} seq=${seq_len} gpu=${gpu} num_gpus=${gpu_count}"
  echo "  dir=${seq_root}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    print_command "${run_cmd[@]}"
    return 0
  fi

  mkdir -p "${seq_root}"
  if [[ ! -e "${config_root}/jobs.tsv" ]]; then
    printf 'status\tgpu\tseq_len\trecompute\texpert_policy\tbackend\tprofiler\tjob_dir\tprofile_json\tlog\n' > "${config_root}/jobs.tsv"
  fi
  {
    print_command "${run_cmd[@]}"
  } > "${seq_root}/command.txt"

  local status=0
  current_child_pid=""
  if command -v setsid >/dev/null 2>&1; then
    setsid "${run_cmd[@]}" &
  else
    "${run_cmd[@]}" &
  fi
  current_child_pid=$!
  wait "${current_child_pid}" || status=$?
  current_child_pid=""
  if [[ "${interrupted}" == "true" || "${status}" == "130" || "${status}" == "143" ]]; then
    echo "Interrupted run; exiting without scheduling more jobs." >&2
    exit 130
  fi
  if ((status == 0)); then
    if [[ ! -f "${profile_json}" ]]; then
      echo "Missing expected profile artifact: ${profile_json}" >&2
      status=1
    fi
  fi

  if ((status == 0)); then
    run_dirs["${group_key}|${backend}"]="${lf_out}"
    printf 'ok\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${backend}" "${profiler}" "${seq_root}" "${profile_json}" "${log_file}" \
      >> "${config_root}/jobs.tsv"
    plot_single_run "${config_root}" "${seq_len}" "${backend}" "${profiler}" "${recompute}" "${expert_policy}" "${seq_root}"
    plot_running_combined "${config_root}" "${seq_len}" "${seq_root}"
  else
    printf 'failed:%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${status}" "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${backend}" "${profiler}" "${seq_root}" "${profile_json}" "${log_file}" \
      >> "${config_root}/jobs.tsv"
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

  [[ "${COMPARE_LOSSES}" == "true" ]] || return 0
  if [[ "${baseline_selected}" != "true" || "${candidate_selected}" != "true" ]]; then
    echo "Skipping loss comparison because selected backends do not include both ${compare_baseline_backend} and ${compare_candidate_backend}."
    return 0
  fi

  for group_key in "${compare_group_keys[@]}"; do
    config_root="${compare_group_config_roots[${group_key}]}"
    [[ "${config_root}" == "${target_config_root}" ]] || continue
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
    if "${CONDA_EXE}" run -p "${ENV_DIR}" python "${PROFILE_POSTPROCESS_SCRIPT}" \
      --baseline-dir "${baseline_dir}" \
      --candidate-dir "${candidate_dir}" \
      --min-steps "${COMPARE_MIN_STEPS}" \
      --warmup-steps "${WARMUP_STEPS}" \
      --first-step-rel-tol "${COMPARE_FIRST_STEP_REL_TOL}" \
      --max-rel-tol "${COMPARE_MAX_REL_TOL}" 2>&1 | tee -a "${compare_log}"; then
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

  plot_root="${config_root}/plots"
  [[ -n "${PLOT_OUTPUT_DIR}" ]] && plot_root="$(abs_path "${PLOT_OUTPUT_DIR}")/$(basename "${config_root}")"
  local -a plot_cmd=(
    "${CONDA_EXE}" run -p "${ENV_DIR}" python "${PLOT_SCRIPT}"
    --input-root "${config_root}"
    --output-dir "${plot_root}"
    --combined-output-dir "${plot_root}"
    --precision "${PRECISION}"
    --clean-output
    --batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --seq-lens "${seq_len}"
    --expert-recompute-policies "${expert_policies[@]}"
  )
  for backend in "${backends[@]}"; do plot_cmd+=(--backend "${backend}"); done
  for profiler in "${profilers[@]}"; do plot_cmd+=(--profiler "${profiler}"); done
  for recompute in "${recompute_modes[@]}"; do plot_cmd+=(--recompute "${recompute}"); done
  echo "Writing LF profile plots: ${plot_root}"
  "${plot_cmd[@]}"
}

plot_running_combined() {
  local config_root="$1"
  local seq_len="$2"
  local seq_root="$3"
  local plot_root
  [[ "${PLOT}" == "true" ]] || return 0

  plot_root="${seq_root}/plots/_combined"
  local -a plot_cmd=(
    "${CONDA_EXE}" run -p "${ENV_DIR}" python "${PLOT_SCRIPT}"
    --input-root "${config_root}"
    --output-dir "${plot_root}"
    --combined-output-dir "${plot_root}"
    --precision "${PRECISION}"
    --clean-output
    --combined-only
    --batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --seq-lens "${seq_len}"
    --expert-recompute-policies "${expert_policies[@]}"
  )
  for backend in "${backends[@]}"; do plot_cmd+=(--backend "${backend}"); done
  for profiler in "${profilers[@]}"; do plot_cmd+=(--profiler "${profiler}"); done
  for recompute in "${recompute_modes[@]}"; do plot_cmd+=(--recompute "${recompute}"); done
  echo "Writing LF running combined plots: ${plot_root}"
  if ! "${plot_cmd[@]}"; then
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

  plot_root="${seq_root}/plots"
  local -a plot_cmd=(
    "${CONDA_EXE}" run -p "${ENV_DIR}" python "${PLOT_SCRIPT}"
    --input-root "${config_root}"
    --output-dir "${plot_root}"
    --combined-output-dir "${plot_root}/combined"
    --precision "${PRECISION}"
    --clean-output
    --skip-combined
    --batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --seq-lens "${seq_len}"
    --expert-recompute-policies "${expert_policy}"
    --backend "${backend}"
    --profiler "${profiler}"
    --recompute "${recompute}"
  )
  echo "Writing LF per-run plots: ${plot_root}"
  if ! "${plot_cmd[@]}"; then
    echo "warning: failed to write per-run plots for ${seq_root}" >&2
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
    config_root="$(config_root_path "${seq_len}")"
    for expert_policy in "${expert_policies[@]}"; do
      for recompute in "${recompute_modes[@]}"; do
        for backend in "${backends[@]}"; do
          for profiler in "${profilers[@]}"; do
            gpu_count="$(backend_gpu_count "${backend}" "${current_model_gpu_count}")"
            gpu="$(gpu_slice "${gpu_count}")"
            if ! run_job "${backend}" "${profiler}" "${recompute}" "${seq_len}" "${gpu}" "${gpu_count}" "${expert_policy}"; then
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
      compare_config_root "${config_root}"
      plot_config_root "${config_root}" "${seq_len}"
    fi
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
  combined_plot_root="${precision_root}/combined"
  [[ -n "${PLOT_OUTPUT_DIR}" ]] && combined_plot_root="$(abs_path "${PLOT_OUTPUT_DIR}")/combined"
  declare -A combined_workloads=()
  for config_root in "${!plot_roots[@]}"; do
    combined_workloads["$(plot_workload_from_config_root "${config_root}")"]=1
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
  for profiler in "${profilers[@]}"; do combined_plot_cmd+=(--profiler "${profiler}"); done
  for recompute in "${recompute_modes[@]}"; do combined_plot_cmd+=(--recompute "${recompute}"); done
  echo "Writing combined LF profile plots: ${combined_plot_root}"
  "${combined_plot_cmd[@]}"
fi

echo "LF profiling completed. Results: ${precision_root}"
