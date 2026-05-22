#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# User-Specified Parameters
# =============================================================================
# Edit this section for the default run. CLI flags override these values.

USER_GPU_POOL=(0 1 2 3)
# Per-workload layers can be set as "workload|layers", for example "moe-604m-a75m|2".
# Dense workload names are total-model labels; MoE workload names are per-layer routed-expert total/active labels.
# USER_WORKLOADS=(mlp_3b mm_3b dense_3b moe-604m-a75m mlp dense moe dense_14b moe-604m-a38m)
# USER_WORKLOADS=(dense_3b "moe-604m-a75m|2")
# USER_WORKLOADS=(mlp_3b mm_3b dense_3b "moe-604m-a75m|4")
USER_WORKLOADS=("moe-604m-a38m|1")
# USER_WORKLOADS=(mlp dense moe)
USER_BACKENDS=(asym torch)
# USER_BACKENDS=(torch)
# USER_PROFILERS=(nsys cpu)
USER_PROFILERS=(nsys)

USER_JOBS_PER_GPU=1
USER_OUTPUT_ROOT="profiling"
USER_RUN_NAME=""
USER_PRECISION="bf16"
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

USER_KT_METHOD="AMXBF16_SFT"
USER_KT_CPU_THREADS=1
USER_KT_THREADPOOL_COUNT=1
USER_KT_MAX_CACHE_DEPTH=1

USER_NSYS_BIN="nsys"
USER_NCU_BIN="ncu"
USER_NCU_PRESET="paper"
USER_NCU_CLEAR_JIT_CACHE=0
USER_PYTHON_BIN="${USER_PYTHON_BIN:-${PYTHON:-python3}}"

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
  --kt-method "${USER_KT_METHOD}"
  --kt-cpu-threads "${USER_KT_CPU_THREADS}"
  --kt-threadpool-count "${USER_KT_THREADPOOL_COUNT}"
  --kt-max-cache-depth "${USER_KT_MAX_CACHE_DEPTH}"
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
  --python-bin PATH                   Python interpreter used for the driver and profiled child process.
                                      Default: USER_PYTHON_BIN, PYTHON, then python3.
  -h, --help                          Show this help.

Default run matrix:
  --workloads ${USER_WORKLOADS[*]}
  --backends ${USER_BACKENDS[*]}
  --profilers ${USER_PROFILERS[*]}

Common options:
  Defaults are defined at the top of this script and can be overridden with
  flags such as --precision, --mode, and --output-root.

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
  done
}

expand_backends() {
  local item
  for item in "$@"; do
    case "${item}" in
      all) printf '%s\n' asym torch kt ;;
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

preflight_python() {
  local python_bin="$1"
  local python_executable
  local python_report

  if ! python_executable="$("${python_bin}" -c 'import sys; print(sys.executable)' 2>&1)"; then
    echo "error: python interpreter failed: ${python_bin}" >&2
    echo "${python_executable}" >&2
    echo "hint: pass --python-bin /path/to/python or set PYTHON=/path/to/python." >&2
    exit 2
  fi
  USER_PYTHON_BIN="${python_executable%%$'\n'*}"

  if ((dry_run_requested)); then
    echo "Using Python: ${USER_PYTHON_BIN}"
    return
  fi

  if ! python_report="$("${USER_PYTHON_BIN}" - <<'PY' 2>&1
import sys

try:
    import torch
except Exception as exc:
    print(f"{exc.__class__.__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)

print(f"{sys.executable} torch={getattr(torch, '__version__', 'unknown')}")
PY
)"; then
    echo "error: ${USER_PYTHON_BIN} cannot import torch; profiling would fail inside nsys." >&2
    echo "${python_report}" >&2
    echo "hint: pass --python-bin /path/to/python from the environment where torch is installed." >&2
    exit 2
  fi
  echo "Using Python: ${python_report}"
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
mode="${USER_MODE}"
dry_run_requested=0

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
    --python-bin)
      USER_PYTHON_BIN="$2"
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

for arg in "${pass_args[@]}"; do
  if [[ "${arg}" == "--dry-run" ]]; then
    dry_run_requested=1
    break
  fi
done

if ((${#gpu_values[@]} == 0)); then
  echo "error: specify a GPU pool with --gpus 2,3 or --gpu-pool 2 3" >&2
  exit 2
fi

if ! [[ "${jobs_per_gpu}" =~ ^[0-9]+$ ]] || ((jobs_per_gpu < 1)); then
  echo "error: --jobs-per-gpu must be a positive integer" >&2
  exit 2
fi

preflight_python "${USER_PYTHON_BIN}"

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
manifest="${log_dir}/$(safe_label "${precision}_lora_sft").tsv"
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
  label="$(safe_label "${precision}_lora_sft_${workload}_${backend}_gpu${gpu}")"
  local log_file="${log_dir}/${label}.log"
  local cmd=(
    "${USER_PYTHON_BIN}" "${PY_DRIVER}"
    --workloads "${workload}"
    --backends "${backend}"
    --profilers "${profilers[@]}"
    --cuda-devices "${gpu}"
    --output-root "${output_root}"
    --precision "${precision}"
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

if ((dry_run_requested)); then
  echo "Dry run completed; skipping aggregate summary collection."
  exit 0
fi

summary_cmd=(
  "${USER_PYTHON_BIN}" "${PY_DRIVER}"
  --workloads "${workloads[@]}"
  --backends "${backends[@]}"
  --profilers "${profilers[@]}"
  --output-root "${output_root}"
  --precision "${precision}"
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
