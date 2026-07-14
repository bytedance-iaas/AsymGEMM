#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# User Parameters
# =============================================================================
# AsymGEMM dir = the dir you run in. Override with ROOT=...
ROOT=${ROOT:-$(pwd)}

DEVICE="cuda:0"
BACKENDS="torch,asym"
OPERATIONS="full_lora"
BATCH_SIZES="1"
SEQ_LENS="4096"
# Same default tensor sizes as the former scripts/lora/profile_transpose.sh (removed), expressed as
# LoRA feature pairs IN|OUT with tokens=M=2048:
#   gate/up: X[M,H] @ W[I,H].T -> Y[M,I]
#   down:    X[M,I] @ W[H,I].T -> Y[M,H]
FEATURE_DIMS="4096|4096"
RANK=16
SCALE=16
DROPOUT_P=0.1
DTYPE="bf16"
PRECISION="bf16"
ASYM_BF16_OUTPUT_DTYPE="bf16"
WARMUP=200
ITERS=200
BACKWARD="both"
CUDA_GRAPH=true

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
OUTPUT_ROOT="${ROOT}/profiling_results/profiling"
OUTPUT_DIR=""
RUN_NAME=""
SAVE_RESULTS=true
PLOT=true
PLOT_OUTPUT_ROOT=""
PLOT_OUTPUT_DIR=""
CLEAN_PLOTS=true
OVERWRITE=false
VERBOSE=false

# =============================================================================
# Derived Parameters
# =============================================================================
PROFILE_SCRIPT="${ROOT}/scripts/lora/profile_lora_ops.py"
PLOT_SCRIPT="${ROOT}/scripts/plotting/plot_lora_operator.py"

# =============================================================================
# Main Logic
# =============================================================================
usage() {
  cat <<USAGE
Usage:
  scripts/lora/profile_lora_ops.sh [options] [extra profile_lora_ops.py args]

Options:
  --device NAME
  --backends LIST
  --operation LIST      xw_sb and/or full_lora; comma or quoted whitespace separated
  --batch-sizes LIST
  --seq-lens LIST
  --feature-dims LIST   in|out feature pairs, e.g. '4096|4096,4096|11008,11008|4096'
  --rank N
  --scale FLOAT
  --dropout-p FLOAT
  --dtype bf16|fp16|fp32
  --precision NAME      precision label; common values: bf16, fp8, fp4
  --asym-bf16-output-dtype bf16|fp32
                       BF16 AsymGEMM temporary/output buffer dtype. Default: bf16.
  --warmup N
  --iters N
  --backward true|false|both
  --cuda-graph true|false
  --output-root DIR     base output root; default: third_party/AsymGEMM/profiling_results/profiling
                       default layout: <root>/lora_ops_<precision>/<operation>__b<batch>_s<seq>_r<rank>
  --output-dir DIR      exact config root; overrides the default config directory
  --run-name NAME       optional config directory under lora_ops_<precision>
  --plot true|false     default: true
  --plot-output-root DIR optional base plot root; default: <config>/plots
  --plot-output-dir DIR exact plot root; overrides the default plots directory
  --clean-plots true|false
  --overwrite true|false
  --no-save             only print the terminal table; disables plots
  --verbose             print each profile_lora_ops.py command before running it
  -h, --help

Default MoE-shaped LoRA tensors:
  tokens = batch_size * seq_len = 1 * 2048 = 2048
  gate/up: 2048|768 and 4096|1536
  down:    768|2048 and 1536|4096
  These correspond to the former profile_transpose.sh shapes:
    2048|2048|768 2048|4096|1536 2048|768|2048 2048|1536|4096
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

bool_value() {
  case "${1,,}" in
    1|true|yes|y|on) printf 'true\n' ;;
    0|false|no|n|off) printf 'false\n' ;;
    *) die "expected true or false, got '${1}'" ;;
  esac
}

backward_values() {
  case "${1,,}" in
    both|all) printf 'false\ntrue\n' ;;
    1|true|yes|y|on) printf 'true\n' ;;
    0|false|no|n|off) printf 'false\n' ;;
    *) die "expected true, false, or both, got '${1}'" ;;
  esac
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

feature_pairs() {
  local pair in_features out_features
  [[ -n "${FEATURE_DIMS}" ]] || die "--feature-dims requires at least one IN|OUT pair"
  for pair in $(tokens "${FEATURE_DIMS}"); do
    [[ "${pair}" == *"|"* ]] || die "--feature-dims entries must be IN|OUT pairs, got '${pair}'"
    in_features="${pair%%|*}"
    out_features="${pair#*|}"
    [[ -n "${in_features}" && -n "${out_features}" ]] || die "--feature-dims entries must be IN|OUT pairs, got '${pair}'"
    [[ "${in_features}" =~ ^[0-9]+$ && "${out_features}" =~ ^[0-9]+$ ]] || die "--feature-dims entries must use positive integer IN|OUT pairs, got '${pair}'"
    [[ "${in_features}" -gt 0 && "${out_features}" -gt 0 ]] || die "--feature-dims entries must be positive, got '${pair}'"
    printf '%s %s\n' "${in_features}" "${out_features}"
  done
}

resolve_dir() {
  local path="$1"
  mkdir -p "${path}"
  (cd "${path}" && pwd)
}

safe_label() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]_.=-' '_' | sed -e 's/^_*//' -e 's/_*$//'
}

csv_header() {
  printf 'operation,backend,pass,device,tokens,batch_size,seq_len,in_features,out_features,rank,scale,dtype,precision,asym_bf16_output_dtype,dropout_p,warmup,iters,cuda_graph,median_ms,mean_ms,min_ms,max_ms,peak_hbm_gib,peak_hbm_bytes\n'
}

print_table_header() {
  printf '%-7s %-10s %-7s %-8s %8s %11s %6s %-13s %10s %10s %11s\n' \
    "status" "operation" "backend" "pass" "tokens" "shape" "rank" "dtype/prec" "median_ms" "mean_ms" "peak_GiB"
  printf '%-7s %-10s %-7s %-8s %8s %11s %6s %-13s %10s %10s %11s\n' \
    "------" "---------" "-------" "----" "------" "-----" "----" "----------" "---------" "-------" "--------"
}

print_table_separator() {
  printf '%-7s %-10s %-7s %-8s %8s %11s %6s %-13s %10s %10s %11s\n' \
    "----" "----------" "-------" "----" "------" "-----------" "----" "-------------" "----------" "----------" "-----------"
}

maybe_print_group_separator() {
  local group_key="$1"
  if [[ -n "${current_group_key}" && "${group_key}" != "${current_group_key}" ]]; then
    print_table_separator
  fi
  current_group_key="${group_key}"
}

pass_label() {
  case "$1" in
    forward) printf 'fw\n' ;;
    backward) printf 'bw\n' ;;
    *) safe_label "$1"; printf '\n' ;;
  esac
}

default_config_label() {
  local operation="$1"
  local batch_size="$2"
  local seq_len="$3"
  printf '%s__b%s_s%s_r%s\n' \
    "$(safe_label "${operation}")" \
    "$(safe_label "${batch_size}")" \
    "$(safe_label "${seq_len}")" \
    "$(safe_label "${RANK}")"
}

config_root_path() {
  local operation="$1"
  local batch_size="$2"
  local seq_len="$3"
  local config_label
  if [[ -n "${CONFIG_ROOT_OVERRIDE}" ]]; then
    printf '%s\n' "${CONFIG_ROOT_OVERRIDE}"
    return
  fi
  if [[ -n "${RUN_NAME}" ]]; then
    config_label="$(safe_label "${RUN_NAME}")"
  else
    config_label="$(default_config_label "${operation}" "${batch_size}" "${seq_len}")"
  fi
  printf '%s/%s\n' "${PRECISION_ROOT}" "${config_label}"
}

result_csv_path() {
  local config_root="$1"
  local backend="$2"
  local pass="$3"
  local in_features="$4"
  local out_features="$5"
  local leaf
  leaf="${in_features}x${out_features}__$(safe_label "${backend}")__$(pass_label "${pass}")"
  printf '%s/%s/result.csv\n' "${config_root}" "${leaf}"
}

result_log_path() {
  local result_csv="$1"
  printf '%s/log.txt\n' "$(dirname "${result_csv}")"
}

write_operator_log() {
  local log_file="$1"
  local mode="$2"
  local status="$3"
  local result_csv="$4"
  local row="${5-}"
  shift 5 || true
  mkdir -p "$(dirname "${log_file}")"
  if [[ "${mode}" == ">" ]]; then
    {
      printf 'time=%s\n' "$(date -Is)"
      printf 'status=%s\n' "${status}"
      printf 'result_csv=%s\n' "${result_csv}"
      if (($#)); then
        printf 'command='
        printf ' %q' "$@"
        printf '\n'
      fi
      [[ -n "${row}" ]] && printf 'row=%s\n' "${row}"
      printf '\n'
    } > "${log_file}"
  else
    {
      printf 'time=%s\n' "$(date -Is)"
      printf 'status=%s\n' "${status}"
      printf 'result_csv=%s\n' "${result_csv}"
      if (($#)); then
        printf 'command='
        printf ' %q' "$@"
        printf '\n'
      fi
      [[ -n "${row}" ]] && printf 'row=%s\n' "${row}"
      printf '\n'
    } >> "${log_file}"
  fi
}

print_row() {
  local status="$1"
  local row="$2"
  local row_operation row_backend row_pass row_device row_tokens row_batch_size row_seq_len row_in_features row_out_features row_rank row_scale row_dtype row_precision row_asym_bf16_output_dtype row_dropout_p row_warmup row_iters row_cuda_graph row_median_ms row_mean_ms row_min_ms row_max_ms row_peak_hbm_gib row_peak_hbm_bytes
  IFS=',' read -r row_operation row_backend row_pass row_device row_tokens row_batch_size row_seq_len row_in_features row_out_features row_rank row_scale row_dtype row_precision row_asym_bf16_output_dtype row_dropout_p row_warmup row_iters row_cuda_graph row_median_ms row_mean_ms row_min_ms row_max_ms row_peak_hbm_gib row_peak_hbm_bytes <<< "${row}"
  printf '%-7s %-10s %-7s %-8s %8s %11s %6s %-13s %10.4f %10.4f %11.4f\n' \
    "${status}" "${row_operation}" "${row_backend}" "${row_pass}" "${row_tokens}" \
    "${row_in_features}x${row_out_features}" "${row_rank}" "${row_dtype}/${row_precision}" \
    "${row_median_ms}" "${row_mean_ms}" "${row_peak_hbm_gib}"
}

extra_args=()
while (($#)); do
  case "$1" in
    --device) need_value "$1" "${2-}"; DEVICE="$2"; shift 2 ;;
    --device=*) DEVICE="${1#*=}"; shift ;;
    --backends) need_value "$1" "${2-}"; BACKENDS="$2"; shift 2 ;;
    --backends=*) BACKENDS="${1#*=}"; shift ;;
    --operation) need_value "$1" "${2-}"; OPERATIONS="$2"; shift 2 ;;
    --operation=*) OPERATIONS="${1#*=}"; shift ;;
    --batch-sizes) need_value "$1" "${2-}"; BATCH_SIZES="$2"; shift 2 ;;
    --batch-sizes=*) BATCH_SIZES="${1#*=}"; shift ;;
    --batch-size|--batch-size=*) die "use --batch-sizes for wrapper sweeps" ;;
    --seq-lens) need_value "$1" "${2-}"; SEQ_LENS="$2"; shift 2 ;;
    --seq-lens=*) SEQ_LENS="${1#*=}"; shift ;;
    --feature-dims) need_value "$1" "${2-}"; FEATURE_DIMS="$2"; shift 2 ;;
    --feature-dims=*) FEATURE_DIMS="${1#*=}"; shift ;;
    --in-features|--in-features=*|--out-features|--out-features=*) die "use --feature-dims 'IN|OUT[,IN|OUT...]' instead of separate in/out feature flags" ;;
    --rank) need_value "$1" "${2-}"; RANK="$2"; shift 2 ;;
    --rank=*) RANK="${1#*=}"; shift ;;
    --scale) need_value "$1" "${2-}"; SCALE="$2"; shift 2 ;;
    --scale=*) SCALE="${1#*=}"; shift ;;
    --dropout-p) need_value "$1" "${2-}"; DROPOUT_P="$2"; shift 2 ;;
    --dropout-p=*) DROPOUT_P="${1#*=}"; shift ;;
    --dtype) need_value "$1" "${2-}"; DTYPE="$2"; shift 2 ;;
    --dtype=*) DTYPE="${1#*=}"; shift ;;
    --precision) need_value "$1" "${2-}"; PRECISION="$2"; shift 2 ;;
    --precision=*) PRECISION="${1#*=}"; shift ;;
    --asym-bf16-output-dtype) need_value "$1" "${2-}"; ASYM_BF16_OUTPUT_DTYPE="$2"; shift 2 ;;
    --asym-bf16-output-dtype=*) ASYM_BF16_OUTPUT_DTYPE="${1#*=}"; shift ;;
    --warmup) need_value "$1" "${2-}"; WARMUP="$2"; shift 2 ;;
    --warmup=*) WARMUP="${1#*=}"; shift ;;
    --iters) need_value "$1" "${2-}"; ITERS="$2"; shift 2 ;;
    --iters=*) ITERS="${1#*=}"; shift ;;
    --backward) need_value "$1" "${2-}"; BACKWARD="$2"; backward_values "${BACKWARD}" >/dev/null; shift 2 ;;
    --backward=*) BACKWARD="${1#*=}"; backward_values "${BACKWARD}" >/dev/null; shift ;;
    --cuda-graph) need_value "$1" "${2-}"; CUDA_GRAPH="$(bool_value "$2")"; shift 2 ;;
    --cuda-graph=*) CUDA_GRAPH="$(bool_value "${1#*=}")"; shift ;;
    --output-root) need_value "$1" "${2-}"; OUTPUT_ROOT="$2"; shift 2 ;;
    --output-root=*) OUTPUT_ROOT="${1#*=}"; shift ;;
    --output-dir) need_value "$1" "${2-}"; OUTPUT_DIR="$2"; shift 2 ;;
    --output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
    --run-name) need_value "$1" "${2-}"; RUN_NAME="$2"; shift 2 ;;
    --run-name=*) RUN_NAME="${1#*=}"; shift ;;
    --plot) need_value "$1" "${2-}"; PLOT="$(bool_value "$2")"; shift 2 ;;
    --plot=*) PLOT="$(bool_value "${1#*=}")"; shift ;;
    --no-plot) PLOT=false; shift ;;
    --plot-output-root) need_value "$1" "${2-}"; PLOT_OUTPUT_ROOT="$2"; shift 2 ;;
    --plot-output-root=*) PLOT_OUTPUT_ROOT="${1#*=}"; shift ;;
    --plot-output-dir) need_value "$1" "${2-}"; PLOT_OUTPUT_DIR="$2"; shift 2 ;;
    --plot-output-dir=*) PLOT_OUTPUT_DIR="${1#*=}"; shift ;;
    --clean-plots) need_value "$1" "${2-}"; CLEAN_PLOTS="$(bool_value "$2")"; shift 2 ;;
    --clean-plots=*) CLEAN_PLOTS="$(bool_value "${1#*=}")"; shift ;;
    --overwrite) need_value "$1" "${2-}"; OVERWRITE="$(bool_value "$2")"; shift 2 ;;
    --overwrite=*) OVERWRITE="$(bool_value "${1#*=}")"; shift ;;
    --no-save) SAVE_RESULTS=false; PLOT=false; shift ;;
    --save) SAVE_RESULTS=true; shift ;;
    --verbose) VERBOSE=true; shift ;;
    --verbose=*) VERBOSE="$(bool_value "${1#*=}")"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) extra_args+=("$1"); shift ;;
  esac
done

graph_label="eager"
if [[ "${CUDA_GRAPH}" == "true" ]]; then
  graph_label="cudagraph"
fi
PRECISION_LABEL="$(safe_label "${PRECISION}")"
PRECISION_ROOT=""
COMBINED_ROOT=""
CONFIG_ROOT_OVERRIDE=""
if [[ "${SAVE_RESULTS}" == "true" ]]; then
  if [[ -n "${OUTPUT_DIR}" ]]; then
    CONFIG_ROOT_OVERRIDE="$(resolve_dir "${OUTPUT_DIR}")"
    PRECISION_ROOT="$(dirname "${CONFIG_ROOT_OVERRIDE}")"
  else
    OUTPUT_ROOT="$(resolve_dir "${OUTPUT_ROOT}")"
    PRECISION_ROOT="$(resolve_dir "${OUTPUT_ROOT}/lora_ops_${PRECISION_LABEL}")"
  fi
  COMBINED_ROOT="$(resolve_dir "${PRECISION_ROOT}/combined")"
  printf 'LoRA operator profile\nPrecision root: %s\n\n' "${PRECISION_ROOT}"
else
  printf 'LoRA operator profile\nData: not saved (--no-save)\n\n'
fi
print_table_header

run_count=0
skip_count=0
current_group_key=""
declare -A plot_configs=()
declare -a plot_csvs=()

while read -r in_features out_features; do
  for operation in $(tokens "${OPERATIONS}"); do
    for backend in $(tokens "${BACKENDS}"); do
      for seq_len in $(tokens "${SEQ_LENS}"); do
        for batch_size in $(tokens "${BATCH_SIZES}"); do
          for backward in $(backward_values "${BACKWARD}"); do
            pass="forward"
            if [[ "${backward}" == "true" ]]; then
              pass="backward"
            fi
            group_key="${operation}|${batch_size}|${seq_len}|${in_features}|${out_features}|${RANK}|${DTYPE}|${PRECISION}|${ASYM_BF16_OUTPUT_DTYPE}|${SCALE}|${DROPOUT_P}|${graph_label}"
            result_csv=""
            if [[ "${SAVE_RESULTS}" == "true" ]]; then
              config_root="$(config_root_path "${operation}" "${batch_size}" "${seq_len}")"
              plot_configs["${config_root}"]="${operation}"$'\t'"${batch_size}"$'\t'"${seq_len}"
              result_csv="$(result_csv_path "${config_root}" "${backend}" "${pass}" "${in_features}" "${out_features}")"
              plot_csvs+=("${result_csv}")
              if [[ -e "${result_csv}" && "${OVERWRITE}" != "true" ]]; then
                header="$(head -n 1 "${result_csv}")"
                if [[ "${header}" != *"asym_bf16_output_dtype"* ]]; then
                  printf 'Stale result schema, rerunning: %s\n' "${result_csv}" >&2
                else
                  row="$(tail -n +2 "${result_csv}" | tail -n 1)"
                  [[ -n "${row}" ]] || die "existing result has no data row: ${result_csv}"
                  row_output_dtype="$(printf '%s\n' "${row}" | cut -d, -f14)"
                  if [[ "${row_output_dtype,,}" != "${ASYM_BF16_OUTPUT_DTYPE,,}" ]]; then
                    printf 'Stale output dtype %s, rerunning: %s\n' "${row_output_dtype}" "${result_csv}" >&2
                  else
                    write_operator_log "$(result_log_path "${result_csv}")" ">>" "skip" "${result_csv}" "${row}"
                    maybe_print_group_separator "${group_key}"
                    print_row "skip" "${row}"
                    skip_count=$((skip_count + 1))
                    continue
                  fi
                fi
              fi
            fi

            cmd=(
              "${PYTHON_BIN}" "${PROFILE_SCRIPT}"
              --operation "${operation}"
              --backend "${backend}"
              --device "${DEVICE}"
              --batch-size "${batch_size}"
              --seq-len "${seq_len}"
              --in-features "${in_features}"
              --out-features "${out_features}"
              --rank "${RANK}"
              --scale "${SCALE}"
              --dropout-p "${DROPOUT_P}"
              --dtype "${DTYPE}"
              --precision "${PRECISION}"
              --asym-bf16-output-dtype "${ASYM_BF16_OUTPUT_DTYPE}"
              --warmup "${WARMUP}"
              --iters "${ITERS}"
            )
            if [[ "${backward}" == "true" ]]; then
              cmd+=(--backward)
            fi
            if [[ "${CUDA_GRAPH}" == "true" ]]; then
              cmd+=(--cuda-graph)
            fi
            cmd+=("${extra_args[@]}" --output-format csv)
            if [[ "${VERBOSE}" == "true" ]]; then
              printf '+'
              printf ' %q' "${cmd[@]}"
              printf '\n'
            fi
            log_file=""
            if [[ "${SAVE_RESULTS}" == "true" ]]; then
              log_file="$(result_log_path "${result_csv}")"
              write_operator_log "${log_file}" ">" "run" "${result_csv}" "" "${cmd[@]}"
            fi
            set +e
            output="$("${cmd[@]}")"
            status=$?
            set -e
            if ((status != 0)); then
              if [[ -n "${log_file}" ]]; then
                {
                  printf 'exit_status=%s\n' "${status}"
                  [[ -n "${output}" ]] && printf 'output=%s\n' "${output}"
                } >> "${log_file}"
              fi
              [[ -n "${output}" ]] && printf '%s\n' "${output}" >&2
              exit "${status}"
            fi
            row="$(printf '%s\n' "${output}" | tail -n 1)"
            [[ -n "${row}" ]] || die "profile_lora_ops.py produced no CSV output"
            if [[ "${SAVE_RESULTS}" == "true" ]]; then
              mkdir -p "$(dirname "${result_csv}")"
              csv_header > "${result_csv}"
              printf '%s\n' "${row}" >> "${result_csv}"
              {
                printf 'exit_status=0\n'
                printf 'row=%s\n' "${row}"
              } >> "${log_file}"
            fi
            maybe_print_group_separator "${group_key}"
            print_row "run" "${row}"
            run_count=$((run_count + 1))
          done
        done
      done
    done
  done
done < <(feature_pairs)

printf '\nCompleted %d run(s); skipped %d existing result(s).\n' "${run_count}" "${skip_count}"

if [[ "${SAVE_RESULTS}" == "true" && "${PLOT}" == "true" ]]; then
  for config_root in "${!plot_configs[@]}"; do
    IFS=$'\t' read -r cfg_operation cfg_batch_size cfg_seq_len <<< "${plot_configs[${config_root}]}"
    if [[ -n "${PLOT_OUTPUT_DIR}" ]]; then
      PLOT_ROOT="$(resolve_dir "${PLOT_OUTPUT_DIR}")"
    elif [[ -n "${PLOT_OUTPUT_ROOT}" ]]; then
      PLOT_OUTPUT_ROOT="$(resolve_dir "${PLOT_OUTPUT_ROOT}")"
      PLOT_ROOT="$(resolve_dir "${PLOT_OUTPUT_ROOT}/$(basename "${config_root}")/plots")"
    else
      PLOT_ROOT="$(resolve_dir "${config_root}/plots")"
    fi
    plot_cmd=(
      "${PYTHON_BIN}" "${PLOT_SCRIPT}"
      --input-root "${config_root}"
      --output-dir "${PLOT_ROOT}"
      --skip-combined
      --flat-output
      --operation "${cfg_operation}"
      --batch-size "${cfg_batch_size}"
      --seq-lens "${cfg_seq_len}"
      --rank "${RANK}"
      --dtype "${DTYPE}"
      --precision "${PRECISION}"
      --dropout-p "${DROPOUT_P}"
      --scale "${SCALE}"
      --cuda-graph "${CUDA_GRAPH}"
      --feature-dims "${FEATURE_DIMS}"
    )
    for backend in $(tokens "${BACKENDS}"); do plot_cmd+=(--backend "${backend}"); done
    for backward in $(backward_values "${BACKWARD}"); do
      if [[ "${backward}" == "true" ]]; then
        plot_cmd+=(--pass backward)
      else
        plot_cmd+=(--pass forward)
      fi
    done
    if [[ "${CLEAN_PLOTS}" == "true" ]]; then
      plot_cmd+=(--clean-output)
    fi
    "${plot_cmd[@]}"
    printf 'Plots: %s\n' "${PLOT_ROOT}"
  done

  if ((${#plot_csvs[@]})); then
    combined_cmd=(
      "${PYTHON_BIN}" "${PLOT_SCRIPT}"
      --combined-output-dir "${COMBINED_ROOT}"
      --combined-only
      --rank "${RANK}"
      --dtype "${DTYPE}"
      --precision "${PRECISION}"
      --dropout-p "${DROPOUT_P}"
      --scale "${SCALE}"
      --cuda-graph "${CUDA_GRAPH}"
      --feature-dims "${FEATURE_DIMS}"
    )
    for csv_path in "${plot_csvs[@]}"; do combined_cmd+=(--input-csv "${csv_path}"); done
    for operation in $(tokens "${OPERATIONS}"); do combined_cmd+=(--operation "${operation}"); done
    for batch_size in $(tokens "${BATCH_SIZES}"); do combined_cmd+=(--batch-size "${batch_size}"); done
    for seq_len in $(tokens "${SEQ_LENS}"); do combined_cmd+=(--seq-lens "${seq_len}"); done
    for backend in $(tokens "${BACKENDS}"); do combined_cmd+=(--backend "${backend}"); done
    for backward in $(backward_values "${BACKWARD}"); do
      if [[ "${backward}" == "true" ]]; then
        combined_cmd+=(--pass backward)
      else
        combined_cmd+=(--pass forward)
      fi
    done
    if [[ "${CLEAN_PLOTS}" == "true" ]]; then
      combined_cmd+=(--clean-output)
    fi
    "${combined_cmd[@]}"
    printf 'Combined: %s\n' "${COMBINED_ROOT}"
  fi
fi
