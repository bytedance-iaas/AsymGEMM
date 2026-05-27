#!/usr/bin/env bash
set -Eeuo pipefail

DEVICE="cuda:0"
BACKENDS="torch,asym"
OPERATIONS="full_lora"
# BATCH_SIZES="8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32"
BATCH_SIZES="8"
SEQ_LENS="1024"
# MoE expert projections from Qwen/Qwen3-30B-A3B:
# gate/up: hidden -> expert intermediate, down: expert intermediate -> hidden.
FEATURE_DIMS="2048|768,768|2048"
RANK=16
SCALE=16
DROPOUT_P=0.1
DTYPE="bf16"
PRECISION="bf16"
WARMUP=200
ITERS=200
BACKWARD="both"
CUDA_GRAPH=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
OUTPUT_ROOT="${ROOT}/profiling/lora_operator"
OUTPUT_DIR=""
RUN_NAME=""
SAVE_RESULTS=true
PLOT=true
PLOT_OUTPUT_ROOT="${ROOT}/profiling/lora_operator_plots"
PLOT_OUTPUT_DIR=""
CLEAN_PLOTS=true
OVERWRITE=false
VERBOSE=false

usage() {
  cat <<USAGE
Usage:
  scripts/profile_lora.sh [options] [extra profile_lora.py args]

Options:
  --device NAME
  --backends LIST
  --operation LIST      xw_sb and/or full_lora; comma or quoted whitespace separated
  --operations LIST     alias for --operation
  --batch-size LIST     alias for --batch-sizes
  --batch-sizes LIST
  --seq-lens LIST
  --feature-dims LIST   in|out feature pairs, e.g. '4096|4096,4096|11008,11008|4096'
  --rank N
  --scale FLOAT
  --dropout-p FLOAT
  --dtype bf16|fp16|fp32
  --precision bf16|fp8|fp4
  --warmup N
  --iters N
  --backward true|false|both
  --cuda-graph true|false
  --output-root DIR     data root; default: third_party/AsymGEMM/profiling/lora_operator
  --output-dir DIR      exact data root; overrides --output-root/--run-name
  --run-name NAME       optional subdirectory under --output-root and --plot-output-root
  --plot true|false     default: true
  --plot-output-root DIR
  --plot-output-dir DIR exact plot root; overrides --plot-output-root[/<run-name>]
  --clean-plots true|false
  --overwrite true|false
  --no-save             only print the terminal table; disables plots
  --verbose             print each profile_lora.py command before running it
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
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:].=-' '_' | sed -e 's/^_*//' -e 's/_*$//'
}

csv_header() {
  printf 'operation,backend,pass,device,tokens,batch_size,seq_len,in_features,out_features,rank,scale,dtype,precision,dropout_p,warmup,iters,cuda_graph,median_ms,mean_ms,min_ms,max_ms,peak_hbm_gib,peak_hbm_bytes\n'
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

result_csv_path() {
  local data_root="$1"
  local operation="$2"
  local backend="$3"
  local pass="$4"
  local batch_size="$5"
  local seq_len="$6"
  local in_features="$7"
  local out_features="$8"
  local graph_label="$9"
  local stem
  stem="b${batch_size}_s${seq_len}_in${in_features}_out${out_features}_r${RANK}_${DTYPE}_${PRECISION}_scale${SCALE}_drop${DROPOUT_P}_${backend}_${pass}_${graph_label}"
  printf '%s/%s/%s/result.csv\n' "${data_root}" "$(safe_label "${operation}")" "$(safe_label "${stem}")"
}

print_row() {
  local status="$1"
  local row="$2"
  local row_operation row_backend row_pass row_device row_tokens row_batch_size row_seq_len row_in_features row_out_features row_rank row_scale row_dtype row_precision row_dropout_p row_warmup row_iters row_cuda_graph row_median_ms row_mean_ms row_min_ms row_max_ms row_peak_hbm_gib row_peak_hbm_bytes
  IFS=',' read -r row_operation row_backend row_pass row_device row_tokens row_batch_size row_seq_len row_in_features row_out_features row_rank row_scale row_dtype row_precision row_dropout_p row_warmup row_iters row_cuda_graph row_median_ms row_mean_ms row_min_ms row_max_ms row_peak_hbm_gib row_peak_hbm_bytes <<< "${row}"
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
    --operation|--operations) need_value "$1" "${2-}"; OPERATIONS="$2"; shift 2 ;;
    --operation=*|--operations=*) OPERATIONS="${1#*=}"; shift ;;
    --batch-size|--batch-sizes) need_value "$1" "${2-}"; BATCH_SIZES="$2"; shift 2 ;;
    --batch-size=*|--batch-sizes=*) BATCH_SIZES="${1#*=}"; shift ;;
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

DATA_ROOT=""
if [[ "${SAVE_RESULTS}" == "true" ]]; then
  if [[ -n "${OUTPUT_DIR}" ]]; then
    DATA_ROOT="$(resolve_dir "${OUTPUT_DIR}")"
  else
    OUTPUT_ROOT="$(resolve_dir "${OUTPUT_ROOT}")"
    DATA_ROOT="${OUTPUT_ROOT}"
    if [[ -n "${RUN_NAME}" ]]; then
      DATA_ROOT="$(resolve_dir "${OUTPUT_ROOT}/${RUN_NAME}")"
    fi
  fi
  printf 'LoRA operator profile\nData root: %s\n\n' "${DATA_ROOT}"
else
  printf 'LoRA operator profile\nData: not saved (--no-save)\n\n'
fi
print_table_header

run_count=0
skip_count=0
current_group_key=""
graph_label="eager"
if [[ "${CUDA_GRAPH}" == "true" ]]; then
  graph_label="cudagraph"
fi

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
            group_key="${operation}|${batch_size}|${seq_len}|${in_features}|${out_features}|${RANK}|${DTYPE}|${PRECISION}|${SCALE}|${DROPOUT_P}|${graph_label}"
            result_csv=""
            if [[ "${SAVE_RESULTS}" == "true" ]]; then
              result_csv="$(result_csv_path "${DATA_ROOT}" "${operation}" "${backend}" "${pass}" "${batch_size}" "${seq_len}" "${in_features}" "${out_features}" "${graph_label}")"
              if [[ -e "${result_csv}" && "${OVERWRITE}" != "true" ]]; then
                row="$(tail -n +2 "${result_csv}" | tail -n 1)"
                [[ -n "${row}" ]] || die "existing result has no data row: ${result_csv}"
                maybe_print_group_separator "${group_key}"
                print_row "skip" "${row}"
                skip_count=$((skip_count + 1))
                continue
              fi
            fi

            cmd=(
              "${PYTHON_BIN}" "${ROOT}/scripts/profile_lora.py"
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
            set +e
            output="$("${cmd[@]}")"
            status=$?
            set -e
            if ((status != 0)); then
              [[ -n "${output}" ]] && printf '%s\n' "${output}" >&2
              exit "${status}"
            fi
            row="$(printf '%s\n' "${output}" | tail -n 1)"
            [[ -n "${row}" ]] || die "profile_lora.py produced no CSV output"
            if [[ "${SAVE_RESULTS}" == "true" ]]; then
              mkdir -p "$(dirname "${result_csv}")"
              csv_header > "${result_csv}"
              printf '%s\n' "${row}" >> "${result_csv}"
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
  if [[ -n "${PLOT_OUTPUT_DIR}" ]]; then
    PLOT_ROOT="$(resolve_dir "${PLOT_OUTPUT_DIR}")"
  else
    PLOT_OUTPUT_ROOT="$(resolve_dir "${PLOT_OUTPUT_ROOT}")"
    PLOT_ROOT="${PLOT_OUTPUT_ROOT}"
    if [[ -n "${RUN_NAME}" ]]; then
      PLOT_ROOT="$(resolve_dir "${PLOT_OUTPUT_ROOT}/${RUN_NAME}")"
    fi
  fi
  plot_cmd=(
    "${PYTHON_BIN}" "${ROOT}/scripts/plotting/plot_lora_operator.py"
    --input-root "${DATA_ROOT}"
    --output-dir "${PLOT_ROOT}"
    --rank "${RANK}"
    --dtype "${DTYPE}"
    --precision "${PRECISION}"
    --dropout-p "${DROPOUT_P}"
    --scale "${SCALE}"
    --cuda-graph "${CUDA_GRAPH}"
  )
  for operation in $(tokens "${OPERATIONS}"); do plot_cmd+=(--operation "${operation}"); done
  for batch_size in $(tokens "${BATCH_SIZES}"); do plot_cmd+=(--batch-size "${batch_size}"); done
  if [[ "${CLEAN_PLOTS}" == "true" ]]; then
    plot_cmd+=(--clean-output)
  fi
  "${plot_cmd[@]}"
  printf 'Plots: %s\n' "${PLOT_ROOT}"
fi
