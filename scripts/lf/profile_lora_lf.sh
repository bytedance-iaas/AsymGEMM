#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# User Parameters
# =============================================================================
ROOT=${ROOT:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM}
LF_DIR=${LF_DIR:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory}
CONDA_EXE=${CONDA_EXE:-conda}
NSYS_BIN=${NSYS_BIN:-nsys}

GPU_POOL="0"
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen3-30B-A3B}
BACKENDS="asym,torch"
PROFILERS="nsys"
OUTPUT_ROOT="profiling"
RUN_NAME=""
PRECISION="bf16"

DATASET=${DATASET:-asym_long_sft_smoke}
TEMPLATE=${TEMPLATE:-qwen3_nothink}
SEQ_LENS="4096"
MAX_SAMPLES=${MAX_SAMPLES:-64}
MAX_STEPS=${MAX_STEPS:-10}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
RECOMPUTE="norecomp"
EXPERT_POLICIES=${EXPERT_POLICIES:-none}

ASYM_OFFLOAD_MODULES=${ASYM_OFFLOAD_MODULES:-routed_experts}
ASYM_STRICT=${ASYM_STRICT:-true}
REQUIRE_SM100=${REQUIRE_SM100:-1}

PLOT=true
PLOT_OUTPUT_DIR=""
OVERWRITE=false
CONTINUE_ON_ERROR=true
DRY_RUN=false
COLLECT_EXISTING=false

# =============================================================================
# Derived Parameters
# =============================================================================
ASYM_DIR=${ASYM_DIR:-${ROOT}}
ENV_DIR=${ENV_DIR:-${LF_DIR}/.venv}
RUN_LF_SCRIPT="${ASYM_DIR}/scripts/lf/run_lf_lora_sft.sh"
SOURCE_POSTPROCESS_SCRIPT="${ASYM_DIR}/scripts/lf/postprocess_lf_source_profile.py"
NSYS_POSTPROCESS_SCRIPT="${ASYM_DIR}/scripts/lora/postprocess_nsys_lora.py"
PLOT_SCRIPT="${ASYM_DIR}/scripts/plotting/plot_activation_recompute_sweep.py"
MODEL_TAG=$(basename "${MODEL_NAME_OR_PATH}" | tr '/:' '__')

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
  --backends LIST                asym and/or torch. torch maps to run_lf_lora_sft.sh BACKEND=asym_torch.
  --profilers LIST               source and/or nsys.
  --seq-lens LIST                LF cutoff lengths. Accepts 2048,4096 or "2048 4096".
  --recompute norecomp|recomp|both
  --expert-policies LIST         AsymGEMM expert policies: none, splitN, tokN-ckpt, tokN-act-ckpt.
  --model-name-or-path NAME
  --dataset NAME
  --template NAME
  --max-samples N
  --max-steps N
  --batch-size N
  --gradient-accumulation-steps N
  --learning-rate VALUE
  --lora-rank N
  --lora-alpha VALUE
  --lora-dropout VALUE
  --precision NAME
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

recompute_values() {
  case "${1,,}" in
    norecomp) printf 'norecomp\n' ;;
    recomp) printf 'recomp\n' ;;
    both) printf 'norecomp\nrecomp\n' ;;
    *) die "expected recompute mode norecomp, recomp, or both; got '${1}'" ;;
  esac
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
    torch) printf 'asym_torch\n' ;;
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
  local config_label
  if [[ -n "${run_name}" ]]; then
    if ((${#seq_lens[@]} > 1)); then
      config_label="$(safe_label "${run_name}__s${seq_len}")"
    else
      config_label="$(safe_label "${run_name}")"
    fi
  else
    config_label="$(safe_label "${workload_label}__b${batch_size}_s${seq_len}_r${LORA_RANK}_a${LORA_ALPHA}")"
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
backend_spec="${BACKENDS}"
profiler_spec="${PROFILERS}"
seq_spec="${SEQ_LENS}"
recompute_spec="${RECOMPUTE}"
expert_policy_spec="${EXPERT_POLICIES}"
output_root="${OUTPUT_ROOT}"
run_name="${RUN_NAME}"
batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}"

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --gpus) need_value "$1" "${2-}"; gpu_spec="$2"; shift 2 ;;
    --gpus=*) gpu_spec="${1#*=}"; shift ;;
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
    --model-name-or-path) need_value "$1" "${2-}"; MODEL_NAME_OR_PATH="$2"; MODEL_TAG=$(basename "${MODEL_NAME_OR_PATH}" | tr '/:' '__'); shift 2 ;;
    --model-name-or-path=*) MODEL_NAME_OR_PATH="${1#*=}"; MODEL_TAG=$(basename "${MODEL_NAME_OR_PATH}" | tr '/:' '__'); shift ;;
    --dataset) need_value "$1" "${2-}"; DATASET="$2"; shift 2 ;;
    --dataset=*) DATASET="${1#*=}"; shift ;;
    --template) need_value "$1" "${2-}"; TEMPLATE="$2"; shift 2 ;;
    --template=*) TEMPLATE="${1#*=}"; shift ;;
    --max-samples) need_value "$1" "${2-}"; MAX_SAMPLES="$2"; shift 2 ;;
    --max-samples=*) MAX_SAMPLES="${1#*=}"; shift ;;
    --max-steps) need_value "$1" "${2-}"; MAX_STEPS="$2"; shift 2 ;;
    --max-steps=*) MAX_STEPS="${1#*=}"; shift ;;
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

mapfile -t gpus < <(tokens "${gpu_spec}" | sed 's/^cuda://' | dedupe)
mapfile -t backends < <(tokens "${backend_spec}" | while read -r value; do backend_label "${value}"; done | dedupe)
mapfile -t profilers < <(tokens "${profiler_spec}" | while read -r value; do profiler_label "${value}"; done | dedupe)
mapfile -t seq_lens < <(tokens "${seq_spec}" | dedupe)
mapfile -t recompute_modes < <(recompute_values "${recompute_spec}")
mapfile -t expert_policies < <(tokens "${expert_policy_spec}" | dedupe)

((${#gpus[@]})) || die "GPU pool is empty"
((${#backends[@]})) || die "backend list is empty"
((${#profilers[@]})) || die "profiler list is empty"
((${#seq_lens[@]})) || die "sequence length list is empty"
((${#expert_policies[@]})) || die "expert policy list is empty"
[[ -f "${RUN_LF_SCRIPT}" ]] || die "missing ${RUN_LF_SCRIPT}"
[[ -f "${SOURCE_POSTPROCESS_SCRIPT}" ]] || die "missing ${SOURCE_POSTPROCESS_SCRIPT}"
[[ -f "${NSYS_POSTPROCESS_SCRIPT}" ]] || die "missing ${NSYS_POSTPROCESS_SCRIPT}"
[[ -f "${PLOT_SCRIPT}" ]] || die "missing ${PLOT_SCRIPT}"

base_output_root="$(abs_path "${output_root}")"
precision_label="$(safe_label "${PRECISION}")"
precision_root="${base_output_root}/lora_lf_${precision_label}"
workload_label="$(safe_label "${MODEL_TAG}")"
mkdir -p "${precision_root}"
echo "Output precision root: ${precision_root}"

declare -A plot_roots=()
failures=0
job_index=0

run_job() {
  local backend="$1"
  local profiler="$2"
  local recompute="$3"
  local seq_len="$4"
  local gpu="$5"
  local expert_policy="$6"
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

  plot_roots["${config_root}"]="${seq_len}"

  if [[ -e "${profile_json}" && "${OVERWRITE}" != "true" && "${COLLECT_EXISTING}" != "true" ]]; then
    echo "Skipping existing: ${profile_json}"
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
    MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}"
    BACKEND="$(lf_backend "${backend}")"
    GPU_ID="${gpu}"
    REQUIRE_SM100="${REQUIRE_SM100}"
    DATASET="${DATASET}"
    TEMPLATE="${TEMPLATE}"
    CUTOFF_LEN="${seq_len}"
    MAX_SAMPLES="${MAX_SAMPLES}"
    MAX_STEPS="${MAX_STEPS}"
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
    PROFILE_SOURCE_JSON="${source_profile}"
    PROFILE_NSYS_PREFIX="${seq_root}/trace"
    PROFILE_WORKLOAD_LABEL="${workload_label}"
    PROFILE_BACKEND_LABEL="${backend}"
    PROFILE_EXPERT_POLICY="${expert_policy}"
    OUT_DIR="${lf_out}"
    LOG_FILE="${log_file}"
    LOSS_LOG_COPY="${seq_root}/loss.trainer_log.jsonl"
    RUN_ID="${run_id}"
  )

  local -a run_cmd=(env "${run_env[@]}" "${RUN_LF_SCRIPT}")

  echo "Running backend=${backend} profiler=${profiler} recompute=${recompute} expert_policy=${expert_policy} seq=${seq_len} gpu=${gpu}"
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
  "${run_cmd[@]}" || status=$?
  if ((status == 0)); then
    if [[ "${profiler}" == "source" ]]; then
      "${CONDA_EXE}" run -p "${ENV_DIR}" python "${SOURCE_POSTPROCESS_SCRIPT}" \
        --source-profile-json "${source_profile}" \
        --output-dir "${seq_root}" || status=$?
    else
      "${NSYS_BIN}" export \
        --type=sqlite \
        --force-overwrite=true \
        --output="${seq_root}/trace.sqlite" \
        "${seq_root}/trace.nsys-rep" || status=$?
      if ((status == 0)); then
        "${CONDA_EXE}" run -p "${ENV_DIR}" python "${NSYS_POSTPROCESS_SCRIPT}" \
          "${seq_root}/trace.sqlite" \
          --source-profile-json "${source_profile}" \
          --output-json "${profile_json}" \
          --output-md "${seq_root}/table.md" || status=$?
      fi
    fi
  fi

  if ((status == 0)); then
    printf 'ok\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${backend}" "${profiler}" "${seq_root}" "${profile_json}" "${log_file}" \
      >> "${config_root}/jobs.tsv"
  else
    printf 'failed:%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${status}" "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${backend}" "${profiler}" "${seq_root}" "${profile_json}" "${log_file}" \
      >> "${config_root}/jobs.tsv"
  fi
  return "${status}"
}

for expert_policy in "${expert_policies[@]}"; do
  for recompute in "${recompute_modes[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
      for backend in "${backends[@]}"; do
        for profiler in "${profilers[@]}"; do
          gpu="${gpus[$((job_index % ${#gpus[@]}))]}"
          job_index=$((job_index + 1))
          if ! run_job "${backend}" "${profiler}" "${recompute}" "${seq_len}" "${gpu}" "${expert_policy}"; then
            failures=$((failures + 1))
            if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
              exit 1
            fi
          fi
        done
      done
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
  for config_root in "${!plot_roots[@]}"; do
    seq_len="${plot_roots[$config_root]}"
    plot_root="${config_root}/plots"
    [[ -n "${PLOT_OUTPUT_DIR}" ]] && plot_root="$(abs_path "${PLOT_OUTPUT_DIR}")/$(basename "${config_root}")"
    plot_cmd=(
      "${CONDA_EXE}" run -p "${ENV_DIR}" python "${PLOT_SCRIPT}"
      --input-root "${config_root}"
      --output-dir "${plot_root}"
      --precision "${PRECISION}"
      --clean-output
      --skip-combined
      --batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
      --seq-lens "${seq_len}"
      --expert-recompute-policies "${expert_policies[@]}"
    )
    for backend in "${backends[@]}"; do plot_cmd+=(--backend "${backend}"); done
    for profiler in "${profilers[@]}"; do plot_cmd+=(--profiler "${profiler}"); done
    for recompute in "${recompute_modes[@]}"; do plot_cmd+=(--recompute "${recompute}"); done
    echo "Writing LF profile plots: ${plot_root}"
    "${plot_cmd[@]}"
  done

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
