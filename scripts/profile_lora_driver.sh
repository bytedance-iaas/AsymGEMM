#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# User-Specified Parameters
# =============================================================================
# Edit this section for the default run. CLI flags override these values.
# List defaults use the same comma-separated format accepted by CLI flags.

GPU_POOL="0,1,2,3"
# Per-workload layers can be set as "workload|layers", e.g. "moe-604m-a75m|2".
# Dense workload names are total-model labels; MoE workload names are per-layer routed-expert total/active labels.
# WORKLOADS="mlp_3b,mm_3b,dense_3b,moe-604m-a75m,mlp,dense,moe,dense_14b,moe-604m-a38m"
# WORKLOADS="dense_3b,moe-604m-a75m|2"
# WORKLOADS="mlp_3b,mm_3b,dense_3b,moe-604m-a75m|4"
WORKLOADS="moe-604m-a38m|2"
# WORKLOADS="mlp,dense,moe"
BACKENDS="asym,torch"
# BACKENDS="torch"
# PROFILERS="nsys,cpu"
PROFILERS="nsys"

JOBS_PER_GPU=1
OUTPUT_ROOT="profiling"
RUN_NAME=""
PRECISION="bf16"
ATTENTION_IMPL="sdpa"
MODE="auto"

WARMUP_STEPS=10
MEASURE_STEPS=20
PROFILE_LAYERS=2
BATCH_SIZE=8
SEQ_LENS="64,128,256,512,640,768,896,1024,2048,3072,4096,6144,8192,10240,16384,20480"
# SEQ_LENS="256"
EXPERT_RECOMPUTE_THRESHOLDS="0"
# EXPERT_RECOMPUTE_THRESHOLDS="0,16,32,64,128"
HIDDEN_DIM=1024
MLP_INTERMEDIATE_DIM=0
MLP_EXPANSION=4
LORA_RANK=64
LORA_ALPHA=128
VOCAB_ROWS=4096
MOE_MODE="contiguous"
DENSE_TARGET_MODE="mlp_only"

KT_METHOD="AMXBF16_SFT"
KT_CPU_THREADS=1
KT_THREADPOOL_COUNT=1
KT_MAX_CACHE_DEPTH=1

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"

PLOT=true
PLOT_OUTPUT_DIR=""
RECOMPUTE="both"
OVERWRITE=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY_DRIVER="${ROOT}/scripts/profile_lora_driver.py"
PLOT_RECOMPUTE_SCRIPT="${ROOT}/scripts/plotting/plot_activation_recompute_sweep.py"

usage() {
  cat <<USAGE
Usage:
  scripts/profile_lora_driver.sh [options]

Defaults:
  --gpus ${GPU_POOL}
  --workloads ${WORKLOADS}
  --backends ${BACKENDS}
  --profilers ${PROFILERS}
  --seq-lens ${SEQ_LENS}
  --expert-recompute-thresholds ${EXPERT_RECOMPUTE_THRESHOLDS}
  --recompute ${RECOMPUTE}
  --overwrite ${OVERWRITE}

Shell options:
  --gpus LIST                         Physical GPU pool. Accepts 0,1 or "0 1".
  --workloads LIST                    Workloads or aliases. Use workload|layers.
  --backends LIST                     asym, torch, kt, or all.
  --profilers LIST                    source, nsys, cpu, ncu, or all.
  --seq-lens LIST                     Sequence lengths. Accepts 64,128 or "64 128".
  --expert-recompute-thresholds LIST   MoE expert token thresholds. 0 disables fine-grained recompute.
  --jobs-per-gpu N                    Concurrent Python driver jobs per GPU.
  --python-bin PATH                   Python interpreter. Default: PYTHON or python3.
  --output-root PATH                  Output root.
  --run-name NAME                     Optional subdirectory under output root.
  --precision NAME                    Result precision label.
  --attention-impl NAME               Attention implementation: sdpa, fa2, fa3, or fa4.
  --mode NAME                         Result filename mode.
  --plot true|false                   Write recompute-vs-seq plots after profiling.
  --plot-output-dir PATH              Plot output directory.
  --recompute norecomp|recomp|both     Run without recompute, with recompute, or both.
  --overwrite true|false              Re-run completed result dirs. Default false skips them.
  -h, --help                          Show this help.

Unknown options are passed through to scripts/profile_lora_driver.py, so common
driver flags such as --dry-run, --target-modules, and --skip-memory-attribution
still work here.
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

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "${ROOT}/$1" ;;
  esac
}

safe_label() {
  printf '%s' "$1" | tr -cs '[:alnum:]_-' '_' | sed -e 's/^[_-]*//' -e 's/[_-]*$//'
}

bool_value() {
  case "${1,,}" in
    1|true|yes|y|on) printf '1\n' ;;
    0|false|no|n|off) printf '0\n' ;;
    *) die "expected true or false, got '${1}'" ;;
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

expand_workload() {
  local item="$1"
  local base="${item}"
  local suffix=""
  if [[ "${item}" == *"|"* ]]; then
    base="${item%%|*}"
    suffix="|${item#*|}"
  fi
  case "${base}" in
    toy) printf '%s\n' "mlp${suffix}" "dense${suffix}" "moe${suffix}" ;;
    custom3b) printf '%s\n' "dense_3b${suffix}" "moe-604m-a75m${suffix}" ;;
    qwen) printf '%s\n' "dense_14b${suffix}" "moe-604m-a38m${suffix}" ;;
    all)
      printf '%s\n' \
        "mlp_1b${suffix}" "mlp_3b${suffix}" "mm_1b${suffix}" "mm_3b${suffix}" \
        "dense_3b${suffix}" "dense_14b${suffix}" \
        "moe-604m-a75m${suffix}" "moe-604m-a38m${suffix}" \
        "mlp${suffix}" "dense${suffix}" "moe${suffix}"
      ;;
    *) printf '%s\n' "${item}" ;;
  esac
}

workload_label() {
  if [[ "$1" == *"|"* ]]; then
    printf '%s-l%s\n' "${1%%|*}" "${1#*|}"
  else
    printf '%s\n' "$1"
  fi
}

expand_alias() {
  local kind="$1"
  local value="$2"
  case "${kind}:${value}" in
    backend:all) printf '%s\n' asym torch kt ;;
    backend:*) printf '%s\n' "${value}" ;;
    profiler:all) printf '%s\n' source nsys cpu ncu ;;
    profiler:*) printf '%s\n' "${value}" ;;
  esac
}

parse_gpu_list() {
  local value
  while read -r value; do
    value="${value#cuda:}"
    printf '%s\n' "${value}"
  done < <(tokens "$@" | dedupe)
}

parse_seq_lens() {
  tokens "$@" | dedupe
}

parse_expert_recompute_thresholds() {
  local value
  while read -r value; do
    [[ "${value}" =~ ^[0-9]+$ ]] || die "expert recompute thresholds must be non-negative integers; got '${value}'"
    printf '%s\n' "${value}"
  done < <(tokens "$@" | dedupe)
}

collect_values() {
  local opt="$1"
  shift
  local -n out="$1"
  shift
  out=()
  while (($#)) && [[ "$1" != --* ]]; do
    out+=("$1")
    shift
  done
  ((${#out[@]} > 0)) || die "${opt} requires at least one value"
  REMAINING=("$@")
}

preflight_python() {
  local report
  if ! PYTHON_BIN="$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)' 2>&1)"; then
    die "python interpreter failed: ${PYTHON_BIN}"
  fi
  PYTHON_BIN="${PYTHON_BIN%%$'\n'*}"
  if ((dry_run)); then
    echo "Using Python: ${PYTHON_BIN}"
    return
  fi
  if ! report="$("${PYTHON_BIN}" - <<'PY' 2>&1
import sys
try:
    import torch
except Exception as exc:
    print(f"{exc.__class__.__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)
print(f"{sys.executable} torch={getattr(torch, '__version__', 'unknown')}")
PY
)"; then
    echo "${report}" >&2
    die "${PYTHON_BIN} cannot import torch; pass --python-bin from the profiling environment"
  fi
  echo "Using Python: ${report}"
}

kill_tree() {
  local pid="$1"
  local child
  while read -r child; do
    [[ -n "${child}" ]] && kill_tree "${child}"
  done < <(pgrep -P "${pid}" 2>/dev/null || true)
  kill -TERM "${pid}" 2>/dev/null || true
}

kill_tree_force() {
  local pid="$1"
  local child
  while read -r child; do
    [[ -n "${child}" ]] && kill_tree_force "${child}"
  done < <(pgrep -P "${pid}" 2>/dev/null || true)
  kill -KILL "${pid}" 2>/dev/null || true
}

gpu_spec="${GPU_POOL}"
workload_spec="${WORKLOADS}"
backend_spec="${BACKENDS}"
profiler_spec="${PROFILERS}"
seq_spec="${SEQ_LENS}"
expert_recompute_threshold_spec="${EXPERT_RECOMPUTE_THRESHOLDS}"
batch_size="${BATCH_SIZE}"
jobs_per_gpu="${JOBS_PER_GPU}"
output_root="${OUTPUT_ROOT}"
run_name="${RUN_NAME}"
precision="${PRECISION}"
attention_impl="${ATTENTION_IMPL}"
mode="${MODE}"
plot="$(bool_value "${PLOT}")"
plot_output_dir="${PLOT_OUTPUT_DIR}"
recompute_spec="${RECOMPUTE}"
overwrite="$(bool_value "${OVERWRITE}")"
dry_run=0
pass_args=()

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --gpus=*) gpu_spec="${1#*=}"; shift ;;
    --gpu-pool=*|--cuda-devices=*) die "use --gpus; this wrapper keeps one option per setting" ;;
    --workloads=*) workload_spec="${1#*=}"; shift ;;
    --backends=*) backend_spec="${1#*=}"; shift ;;
    --profilers=*) profiler_spec="${1#*=}"; shift ;;
    --seq-len=*) die "use --seq-lens; this wrapper keeps one option per setting" ;;
    --seq-lens=*) seq_spec="${1#*=}"; shift ;;
    --expert-recompute-threshold=*) die "use --expert-recompute-thresholds; this wrapper keeps one option per setting" ;;
    --expert-recompute-thresholds=*) expert_recompute_threshold_spec="${1#*=}"; shift ;;
    --batch-size=*) batch_size="${1#*=}"; shift ;;
    --jobs-per-gpu=*) jobs_per_gpu="${1#*=}"; shift ;;
    --python-bin=*) PYTHON_BIN="${1#*=}"; shift ;;
    --output-root=*) output_root="${1#*=}"; shift ;;
    --run-name=*) run_name="${1#*=}"; shift ;;
    --precision=*) precision="${1#*=}"; shift ;;
    --attention-impl=*) attention_impl="${1#*=}"; shift ;;
    --mode=*) mode="${1#*=}"; shift ;;
    --plot=*) die "use '--plot true' or '--plot false' instead of --plot=..." ;;
    --plot-output-dir=*) plot_output_dir="${1#*=}"; shift ;;
    --recompute=*) recompute_spec="${1#*=}"; shift ;;
    --overwrite=*) die "use '--overwrite true' or '--overwrite false' instead of --overwrite=..." ;;
    --gpus) collect_values "$1" vals "${@:2}"; gpu_spec="${vals[*]}"; set -- "${REMAINING[@]}" ;;
    --gpu-pool|--cuda-devices) die "use --gpus; this wrapper keeps one option per setting" ;;
    --workloads) collect_values "$1" vals "${@:2}"; workload_spec="${vals[*]}"; set -- "${REMAINING[@]}" ;;
    --backends) collect_values "$1" vals "${@:2}"; backend_spec="${vals[*]}"; set -- "${REMAINING[@]}" ;;
    --profilers) collect_values "$1" vals "${@:2}"; profiler_spec="${vals[*]}"; set -- "${REMAINING[@]}" ;;
    --seq-len) die "use --seq-lens; this wrapper keeps one option per setting" ;;
    --seq-lens) collect_values "$1" vals "${@:2}"; seq_spec="${vals[*]}"; set -- "${REMAINING[@]}" ;;
    --expert-recompute-threshold) die "use --expert-recompute-thresholds; this wrapper keeps one option per setting" ;;
    --expert-recompute-thresholds) collect_values "$1" vals "${@:2}"; expert_recompute_threshold_spec="${vals[*]}"; set -- "${REMAINING[@]}" ;;
    --batch-size) need_value "$1" "${2-}"; batch_size="$2"; shift 2 ;;
    --jobs-per-gpu) need_value "$1" "${2-}"; jobs_per_gpu="$2"; shift 2 ;;
    --python-bin) need_value "$1" "${2-}"; PYTHON_BIN="$2"; shift 2 ;;
    --output-root) need_value "$1" "${2-}"; output_root="$2"; shift 2 ;;
    --run-name) need_value "$1" "${2-}"; run_name="$2"; shift 2 ;;
    --precision) need_value "$1" "${2-}"; precision="$2"; shift 2 ;;
    --attention-impl) need_value "$1" "${2-}"; attention_impl="$2"; shift 2 ;;
    --mode) need_value "$1" "${2-}"; mode="$2"; shift 2 ;;
    --plot) need_value "$1" "${2-}"; plot="$(bool_value "$2")"; shift 2 ;;
    --plot-output-dir) need_value "$1" "${2-}"; plot_output_dir="$2"; shift 2 ;;
    --recompute) need_value "$1" "${2-}"; recompute_spec="$2"; shift 2 ;;
    --overwrite) need_value "$1" "${2-}"; overwrite="$(bool_value "$2")"; shift 2 ;;
    --activation-recompute) die "use --recompute recomp; this wrapper keeps one option per setting" ;;
    --) shift; pass_args+=("$@"); break ;;
    *) pass_args+=("$1"); [[ "$1" == "--dry-run" ]] && dry_run=1; shift ;;
  esac
done

for arg in "${pass_args[@]}"; do
  [[ "${arg}" == "--dry-run" ]] && dry_run=1
done

[[ -n "${PYTHON_BIN}" ]] || die "--python-bin cannot be empty"
[[ -n "${output_root}" ]] || die "--output-root cannot be empty"
[[ -n "${precision}" ]] || die "--precision cannot be empty"
[[ -n "${attention_impl}" ]] || die "--attention-impl cannot be empty"
[[ -n "${mode}" ]] || die "--mode cannot be empty"
[[ "${jobs_per_gpu}" =~ ^[0-9]+$ && "${jobs_per_gpu}" -gt 0 ]] || die "--jobs-per-gpu must be a positive integer"

mapfile -t gpus < <(parse_gpu_list "${gpu_spec}")
mapfile -t seq_lens < <(parse_seq_lens "${seq_spec}")
mapfile -t expert_recompute_thresholds < <(parse_expert_recompute_thresholds "${expert_recompute_threshold_spec}")
mapfile -t workloads < <(tokens "${workload_spec}" | while read -r value; do expand_workload "${value}"; done | dedupe)
mapfile -t plot_workloads < <(printf '%s\n' "${workloads[@]}" | while read -r value; do workload_label "${value}"; done | dedupe)
mapfile -t backends < <(tokens "${backend_spec}" | while read -r value; do expand_alias backend "${value}"; done | dedupe)
mapfile -t profilers < <(tokens "${profiler_spec}" | while read -r value; do expand_alias profiler "${value}"; done | dedupe)
mapfile -t recompute_modes < <(recompute_values "${recompute_spec}")

((${#gpus[@]})) || die "GPU pool is empty"
((${#seq_lens[@]})) || die "sequence length list is empty"
((${#expert_recompute_thresholds[@]} > 0)) || die "expert recompute threshold list is empty"
((${#workloads[@]} && ${#backends[@]} && ${#profilers[@]} && ${#recompute_modes[@]})) || die "workloads/backends/profilers/recompute expanded to an empty list"

preflight_python

driver_args=(
  --warmup-steps "${WARMUP_STEPS}"
  --measure-steps "${MEASURE_STEPS}"
  --profile-layers "${PROFILE_LAYERS}"
  --batch-size "${batch_size}"
  --hidden-dim "${HIDDEN_DIM}"
  --mlp-intermediate-dim "${MLP_INTERMEDIATE_DIM}"
  --mlp-expansion "${MLP_EXPANSION}"
  --lora-rank "${LORA_RANK}"
  --lora-alpha "${LORA_ALPHA}"
  --vocab-rows "${VOCAB_ROWS}"
  --attention-impl "${attention_impl}"
  --moe-mode "${MOE_MODE}"
  --dense-target-mode "${DENSE_TARGET_MODE}"
  --kt-method "${KT_METHOD}"
  --kt-cpu-threads "${KT_CPU_THREADS}"
  --kt-threadpool-count "${KT_THREADPOOL_COUNT}"
  --kt-max-cache-depth "${KT_MAX_CACHE_DEPTH}"
  --seq-lens "${seq_lens[@]}"
)
if ((overwrite)); then
  driver_args+=(--overwrite)
else
  driver_args+=(--skip-existing)
fi

run_root="$(abs_path "${output_root}")"
[[ -n "${run_name}" ]] && run_root="${run_root}/${run_name}"
log_dir="${run_root}/driver_logs"
mkdir -p "${log_dir}"
manifest="${log_dir}/$(safe_label "${precision}_lora-sft").tsv"
printf 'status\tpid\tgpu\trecompute\texpert_threshold\tworkload\tbackend\tlog\n' > "${manifest}"

declare -a all_pids=() active_pids=()
declare -A pid_gpu=() pid_recompute=() pid_expert_threshold=() pid_workload=() pid_backend=() pid_log=()
failures=0

cleanup_jobs() {
  ((${#all_pids[@]})) || return
  echo "Stopping background profiling jobs..." >&2
  local pid
  for pid in "${all_pids[@]}"; do kill_tree "${pid}"; done
  sleep 2
  for pid in "${all_pids[@]}"; do kill_tree_force "${pid}"; done
}

trap 'trap - INT TERM EXIT; cleanup_jobs; exit 130' INT TERM
trap 'status=$?; trap - EXIT; if ((status != 0)); then cleanup_jobs; fi; exit "${status}"' EXIT

finish_one() {
  local pid="${active_pids[0]}"
  local status=0
  if wait "${pid}"; then
    echo "Finished pid=${pid} gpu=${pid_gpu[$pid]} recompute=${pid_recompute[$pid]} expert_threshold=${pid_expert_threshold[$pid]} workload=${pid_workload[$pid]} backend=${pid_backend[$pid]}"
  else
    status=$?
    failures=$((failures + 1))
    echo "FAILED pid=${pid} status=${status} gpu=${pid_gpu[$pid]} recompute=${pid_recompute[$pid]} expert_threshold=${pid_expert_threshold[$pid]} workload=${pid_workload[$pid]} backend=${pid_backend[$pid]}; log=${pid_log[$pid]}" >&2
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${status}" "${pid}" "${pid_gpu[$pid]}" "${pid_recompute[$pid]}" "${pid_expert_threshold[$pid]}" "${pid_workload[$pid]}" "${pid_backend[$pid]}" "${pid_log[$pid]}" >> "${manifest}"
  active_pids=("${active_pids[@]:1}")
}

launch_job() {
  local recompute="$1"
  local expert_threshold="$2"
  local workload="$3"
  local backend="$4"
  local gpu="$5"
  local threshold_label=""
  ((expert_threshold > 0)) && threshold_label="_expertthr${expert_threshold}"
  local log_file="${log_dir}/$(safe_label "${precision}_lora-sft_${recompute}${threshold_label}_${workload}_${backend}_gpu${gpu}").log"
  local cmd=(
    "${PYTHON_BIN}" "${PY_DRIVER}"
    --workloads "${workload}"
    --backends "${backend}"
    --profilers "${profilers[@]}"
    --cuda-devices "${gpu}"
    --output-root "${output_root}"
    --precision "${precision}"
    --mode "${mode}"
    --expert-recompute-threshold "${expert_threshold}"
    --skip-summary
    "${driver_args[@]}"
    "${pass_args[@]}"
  )
  [[ -n "${run_name}" ]] && cmd+=(--run-name "${run_name}")
  [[ "${recompute}" == "recomp" ]] && cmd+=(--activation-recompute)

  echo "Launching gpu=${gpu} recompute=${recompute} expert_threshold=${expert_threshold} workload=${workload} backend=${backend}"
  echo "  log=${log_file}"
  (cd "${ROOT}" && exec "${cmd[@]}") > "${log_file}" 2>&1 &

  local pid=$!
  all_pids+=("${pid}")
  active_pids+=("${pid}")
  pid_gpu[$pid]="${gpu}"
  pid_recompute[$pid]="${recompute}"
  pid_expert_threshold[$pid]="${expert_threshold}"
  pid_workload[$pid]="${workload}"
  pid_backend[$pid]="${backend}"
  pid_log[$pid]="${log_file}"
}

max_parallel=$(( ${#gpus[@]} * jobs_per_gpu ))
job_index=0
for expert_threshold in "${expert_recompute_thresholds[@]}"; do
  for recompute in "${recompute_modes[@]}"; do
    for workload in "${workloads[@]}"; do
      for backend in "${backends[@]}"; do
        while ((${#active_pids[@]} >= max_parallel)); do finish_one; done
        launch_job "${recompute}" "${expert_threshold}" "${workload}" "${backend}" "${gpus[$((job_index % ${#gpus[@]}))]}"
        job_index=$((job_index + 1))
      done
    done
  done
done
while ((${#active_pids[@]})); do finish_one; done

echo "Job manifest: ${manifest}"
if ((failures > 0)); then
  echo "${failures} profiling job(s) failed" >&2
  exit 1
fi

if ((dry_run)); then
  echo "Dry run completed; skipping aggregate summary collection."
  exit 0
fi

echo "Writing aggregate summary..."
for expert_threshold in "${expert_recompute_thresholds[@]}"; do
  for recompute in "${recompute_modes[@]}"; do
    summary_cmd=(
      "${PYTHON_BIN}" "${PY_DRIVER}"
      --workloads "${workloads[@]}"
      --backends "${backends[@]}"
      --profilers "${profilers[@]}"
      --output-root "${output_root}"
      --precision "${precision}"
      --mode "${mode}"
      --expert-recompute-threshold "${expert_threshold}"
      --collect-existing
      "${driver_args[@]}"
      "${pass_args[@]}"
    )
    [[ -n "${run_name}" ]] && summary_cmd+=(--run-name "${run_name}")
    [[ "${recompute}" == "recomp" ]] && summary_cmd+=(--activation-recompute)
    echo "  recompute=${recompute} expert_threshold=${expert_threshold}"
    (cd "${ROOT}" && "${summary_cmd[@]}")
  done
done

if ((plot)); then
  plot_cmd=("${PYTHON_BIN}" "${PLOT_RECOMPUTE_SCRIPT}" --input-root "${run_root}" --precision "${precision}" --clean-output)
  for workload in "${plot_workloads[@]}"; do plot_cmd+=(--workload "${workload}"); done
  for backend in "${backends[@]}"; do plot_cmd+=(--backend "${backend}"); done
  for profiler in "${profilers[@]}"; do plot_cmd+=(--profiler "${profiler}"); done
  for recompute in "${recompute_modes[@]}"; do plot_cmd+=(--recompute "${recompute}"); done
  plot_cmd+=(--expert-recompute-thresholds "${expert_recompute_thresholds[@]}")
  plot_cmd+=(--batch-size "${batch_size}" --seq-lens "${seq_lens[@]}")
  [[ -n "${plot_output_dir}" ]] && plot_cmd+=(--output-dir "$(abs_path "${plot_output_dir}")")
  echo "Writing activation recompute sweep plots..."
  (cd "${ROOT}" && "${plot_cmd[@]}")
fi

echo "All profiling jobs completed."
