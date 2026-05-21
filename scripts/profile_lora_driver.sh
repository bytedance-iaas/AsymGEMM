#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# User-Specified Parameters
# =============================================================================
# Edit this section for the default run. CLI flags override these values.

USER_GPU_POOL=(0 1 2 3)
# USER_WORKLOADS=(mlp_3b mm_3b dense_3b moe_3b mlp dense moe qwen3_14b qwen3_30b_a3b)
# USER_WORKLOADS=(dense_3b moe_3b)
# USER_WORKLOADS=(mlp_3b mm_3b dense_3b moe_3b)
USER_WORKLOADS=(qwen3_14b qwen3_30b_a3b)
# USER_WORKLOADS=(mlp dense moe)
USER_BACKENDS=(asym_only torch_only)
# USER_BACKENDS=(torch_only)
# USER_PROFILERS=(nsys cpu)
USER_PROFILERS=(nsys)

USER_JOBS_PER_GPU=1
USER_OUTPUT_ROOT="profiling"
USER_RUN_NAME=""
USER_PRECISION="bf16"
USER_WORKFLOW="lora_sft"
USER_MODE="auto"

USER_WARMUP_STEPS=5
USER_MEASURE_STEPS=20
USER_PROFILE_LAYERS=1
USER_BATCH_SIZE=32
USER_SEQ_LEN=64
USER_HIDDEN_DIM=1024
USER_MLP_INTERMEDIATE_DIM=0
USER_MLP_EXPANSION=4
USER_LORA_RANK=64
USER_LORA_ALPHA=128
USER_VOCAB_ROWS=4096
USER_MOE_MODE="contiguous"
USER_DENSE_TARGET_MODE="mlp_only"

USER_NSYS_BIN="nsys"
USER_NCU_BIN="ncu"
USER_NCU_PRESET="paper"
USER_NCU_CLEAR_JIT_CACHE=0
USER_PYTHON_BIN="python3"

# =============================================================================
# Derived Parameters
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY_DRIVER="${ROOT}/scripts/profile_lora_driver.py"

DRIVER_ARGS=(
  --warmup-steps "${USER_WARMUP_STEPS}"
  --measure-steps "${USER_MEASURE_STEPS}"
  --profile-layers "${USER_PROFILE_LAYERS}"
  --batch-size "${USER_BATCH_SIZE}"
  --seq-len "${USER_SEQ_LEN}"
  --hidden-dim "${USER_HIDDEN_DIM}"
  --mlp-intermediate-dim "${USER_MLP_INTERMEDIATE_DIM}"
  --mlp-expansion "${USER_MLP_EXPANSION}"
  --lora-rank "${USER_LORA_RANK}"
  --lora-alpha "${USER_LORA_ALPHA}"
  --vocab-rows "${USER_VOCAB_ROWS}"
  --moe-mode "${USER_MOE_MODE}"
  --dense-target-mode "${USER_DENSE_TARGET_MODE}"
  --nsys-bin "${USER_NSYS_BIN}"
  --ncu-bin "${USER_NCU_BIN}"
  --ncu-preset "${USER_NCU_PRESET}"
)
if ((USER_NCU_CLEAR_JIT_CACHE)); then
  DRIVER_ARGS+=(--ncu-clear-jit-cache)
fi

# =============================================================================
# Core Logic
# =============================================================================

usage() {
  cat <<USAGE
Usage:
  scripts/profile_lora_driver.sh --gpus 2,3,4,5,6,7 [options]

Required:
  --gpus, --gpu-pool, --cuda-devices  Physical GPU pool. Comma or space separated.
                                      Not required if USER_GPU_POOL is set
                                      at the top of this script.

Shell-only options:
  --jobs-per-gpu N                    Concurrent Python driver jobs per GPU. Default: ${USER_JOBS_PER_GPU}.
  -h, --help                          Show this help.

Default run matrix:
  --workloads ${USER_WORKLOADS[*]}
  --backends ${USER_BACKENDS[*]}
  --profilers ${USER_PROFILERS[*]}

Common options:
  Defaults are defined at the top of this script and can be overridden with
  flags such as --precision, --workflow, --mode, and --output-root.

The shell launches one background Python driver job per workload/backend pair,
assigns each job one GPU from the pool, waits for all jobs, and traps
INT/TERM/ERR to terminate the whole background process tree.
The Python driver is an internal worker; use this shell script as the profiling
entrypoint for the standard workflow.
USAGE
}

split_values() {
  local value part
  for value in "$@"; do
    IFS=',' read -r -a _parts <<< "${value}"
    for part in "${_parts[@]}"; do
      part="${part#"${part%%[![:space:]]*}"}"
      part="${part%"${part##*[![:space:]]}"}"
      [[ -n "${part}" ]] && printf '%s\n' "${part#cuda:}"
    done
  done
}

dedupe_lines() {
  awk '!seen[$0]++'
}

expand_workloads() {
  local item
  for item in "$@"; do
    case "${item}" in
      toy) printf '%s\n' mlp dense moe ;;
      custom3b) printf '%s\n' dense_3b moe_3b ;;
      qwen) printf '%s\n' qwen3_14b qwen3_30b_a3b ;;
      all) printf '%s\n' mlp_1b mlp_3b mm_1b mm_3b dense_3b moe_3b mlp dense moe qwen3_14b qwen3_30b_a3b ;;
      *) printf '%s\n' "${item}" ;;
    esac
  done
}

expand_backends() {
  local item
  for item in "$@"; do
    case "${item}" in
      all) printf '%s\n' asym_only torch_only ;;
      *) printf '%s\n' "${item}" ;;
    esac
  done
}

expand_profilers() {
  local item
  for item in "$@"; do
    case "${item}" in
      all) printf '%s\n' source nsys cpu ncu ;;
      *) printf '%s\n' "${item}" ;;
    esac
  done
}

safe_label() {
  printf '%s' "$1" | tr -cs '[:alnum:]_-' '_' | sed -e 's/^[_-]*//' -e 's/[_-]*$//'
}

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "${ROOT}/$1" ;;
  esac
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

declare -a gpu_values=("${USER_GPU_POOL[@]}")
declare -a workload_values=("${USER_WORKLOADS[@]}")
declare -a backend_values=("${USER_BACKENDS[@]}")
declare -a profiler_values=("${USER_PROFILERS[@]}")
declare -a pass_args=()
jobs_per_gpu="${USER_JOBS_PER_GPU}"
output_root="${USER_OUTPUT_ROOT}"
run_name="${USER_RUN_NAME}"
precision="${USER_PRECISION}"
workflow="${USER_WORKFLOW}"
mode="${USER_MODE}"

while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --gpus|--gpu-pool|--cuda-devices)
      gpu_values=()
      shift
      while (($#)) && [[ "$1" != --* ]]; do
        gpu_values+=("$1")
        shift
      done
      ;;
    --workloads)
      workload_values=()
      shift
      while (($#)) && [[ "$1" != --* ]]; do
        workload_values+=("$1")
        shift
      done
      ;;
    --backends)
      backend_values=()
      shift
      while (($#)) && [[ "$1" != --* ]]; do
        backend_values+=("$1")
        shift
      done
      ;;
    --profilers)
      profiler_values=()
      shift
      while (($#)) && [[ "$1" != --* ]]; do
        profiler_values+=("$1")
        shift
      done
      ;;
    --jobs-per-gpu)
      jobs_per_gpu="$2"
      shift 2
      ;;
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --run-name)
      run_name="$2"
      shift 2
      ;;
    --precision)
      precision="$2"
      shift 2
      ;;
    --workflow)
      workflow="$2"
      shift 2
      ;;
    --mode)
      mode="$2"
      shift 2
      ;;
    --)
      shift
      pass_args+=("$@")
      break
      ;;
    *)
      pass_args+=("$1")
      shift
      ;;
  esac
done

if ((${#gpu_values[@]} == 0)); then
  echo "error: specify a GPU pool with --gpus 2,3 or --gpu-pool 2 3" >&2
  exit 2
fi

if ! [[ "${jobs_per_gpu}" =~ ^[0-9]+$ ]] || ((jobs_per_gpu < 1)); then
  echo "error: --jobs-per-gpu must be a positive integer" >&2
  exit 2
fi

mapfile -t gpus < <(split_values "${gpu_values[@]}" | dedupe_lines)
if ((${#gpus[@]} == 0)); then
  echo "error: empty GPU pool" >&2
  exit 2
fi

mapfile -t workloads < <(split_values "${workload_values[@]}" | xargs -r -n1 printf '%s\n' | while read -r x; do expand_workloads "$x"; done | dedupe_lines)
mapfile -t backends < <(split_values "${backend_values[@]}" | xargs -r -n1 printf '%s\n' | while read -r x; do expand_backends "$x"; done | dedupe_lines)
mapfile -t profilers < <(split_values "${profiler_values[@]}" | xargs -r -n1 printf '%s\n' | while read -r x; do expand_profilers "$x"; done | dedupe_lines)

if ((${#workloads[@]} == 0 || ${#backends[@]} == 0 || ${#profilers[@]} == 0)); then
  echo "error: workloads/backends/profilers expanded to empty lists" >&2
  exit 2
fi

run_root="$(abs_path "${output_root}")"
[[ -n "${run_name}" ]] && run_root="${run_root}/${run_name}"
log_dir="${run_root}/driver_logs"
mkdir -p "${log_dir}"
manifest="${log_dir}/$(safe_label "${precision}_${workflow}")_jobs.tsv"
printf 'status\tpid\tgpu\tworkload\tbackend\tlog\n' > "${manifest}"

declare -a all_pids=()
declare -a active_pids=()
declare -a active_gpus=()
declare -a active_workloads=()
declare -a active_backends=()
declare -a active_logs=()
failures=0

cleanup_jobs() {
  local pid
  if ((${#all_pids[@]} == 0)); then
    return
  fi
  echo "Stopping background profiling jobs..." >&2
  for pid in "${all_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill_tree "${pid}"
    fi
  done
  sleep 2
  for pid in "${all_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill_tree_force "${pid}"
    fi
  done
}

on_signal() {
  trap - INT TERM EXIT
  cleanup_jobs
  exit 130
}

on_exit() {
  local status=$?
  trap - EXIT
  if ((status != 0)); then
    cleanup_jobs
  fi
}

trap on_signal INT TERM
trap on_exit EXIT

pop_active_front() {
  active_pids=("${active_pids[@]:1}")
  active_gpus=("${active_gpus[@]:1}")
  active_workloads=("${active_workloads[@]:1}")
  active_backends=("${active_backends[@]:1}")
  active_logs=("${active_logs[@]:1}")
}

wait_one() {
  local pid="${active_pids[0]}"
  local gpu="${active_gpus[0]}"
  local workload="${active_workloads[0]}"
  local backend="${active_backends[0]}"
  local log_file="${active_logs[0]}"
  local status=0
  if wait "${pid}"; then
    status=0
    echo "Finished pid=${pid} gpu=${gpu} workload=${workload} backend=${backend}"
  else
    status=$?
    failures=$((failures + 1))
    echo "FAILED pid=${pid} status=${status} gpu=${gpu} workload=${workload} backend=${backend}; log=${log_file}" >&2
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${status}" "${pid}" "${gpu}" "${workload}" "${backend}" "${log_file}" >> "${manifest}"
  pop_active_front
}

wait_for_slot() {
  local max_parallel=$(( ${#gpus[@]} * jobs_per_gpu ))
  while ((${#active_pids[@]} >= max_parallel)); do
    wait_one
  done
}

launch_job() {
  local workload="$1"
  local backend="$2"
  local gpu="$3"
  local label
  label="$(safe_label "${precision}_${workflow}_${workload}_${backend}_gpu${gpu}")"
  local log_file="${log_dir}/${label}.log"
  local cmd=(
    "${USER_PYTHON_BIN}" "${PY_DRIVER}"
    --workloads "${workload}"
    --backends "${backend}"
    --profilers "${profilers[@]}"
    --cuda-devices "${gpu}"
    --output-root "${output_root}"
    --precision "${precision}"
    --workflow "${workflow}"
    --mode "${mode}"
    --skip-summary
    "${DRIVER_ARGS[@]}"
    "${pass_args[@]}"
  )
  if [[ -n "${run_name}" ]]; then
    cmd+=(--run-name "${run_name}")
  fi

  echo "Launching gpu=${gpu} workload=${workload} backend=${backend}"
  echo "  log=${log_file}"
  (
    cd "${ROOT}"
    exec "${cmd[@]}"
  ) > "${log_file}" 2>&1 &

  local pid=$!
  all_pids+=("${pid}")
  active_pids+=("${pid}")
  active_gpus+=("${gpu}")
  active_workloads+=("${workload}")
  active_backends+=("${backend}")
  active_logs+=("${log_file}")
}

job_index=0
for workload in "${workloads[@]}"; do
  for backend in "${backends[@]}"; do
    wait_for_slot
    gpu="${gpus[$((job_index % ${#gpus[@]}))]}"
    launch_job "${workload}" "${backend}" "${gpu}"
    job_index=$((job_index + 1))
  done
done

while ((${#active_pids[@]} > 0)); do
  wait_one
done

echo "Job manifest: ${manifest}"
if ((failures > 0)); then
  echo "${failures} profiling job(s) failed" >&2
  exit 1
fi

summary_cmd=(
  "${USER_PYTHON_BIN}" "${PY_DRIVER}"
  --workloads "${workloads[@]}"
  --backends "${backends[@]}"
  --profilers "${profilers[@]}"
  --output-root "${output_root}"
  --precision "${precision}"
  --workflow "${workflow}"
  --mode "${mode}"
  --collect-existing
  "${DRIVER_ARGS[@]}"
  "${pass_args[@]}"
)
if [[ -n "${run_name}" ]]; then
  summary_cmd+=(--run-name "${run_name}")
fi

echo "Writing aggregate summary..."
(
  cd "${ROOT}"
  "${summary_cmd[@]}"
)

echo "All profiling jobs completed."
