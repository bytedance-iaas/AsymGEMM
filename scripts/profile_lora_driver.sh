#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY_DRIVER="${ROOT}/scripts/profile_lora_driver.py"

usage() {
  cat <<'USAGE'
Usage:
  scripts/profile_lora_driver.sh --gpus 2,3,4,5,6,7 [driver args...]

Required:
  --gpus, --gpu-pool, --cuda-devices  Physical GPU pool. Comma or space separated.

Shell-only options:
  --jobs-per-gpu N                    Concurrent Python driver jobs per GPU. Default: 1.
  -h, --help                          Show this help.

Common forwarded options:
  --workloads qwen3_14b qwen3_30b_a3b
  --backends asym_only torch_only
  --profilers source nsys cpu ncu
  --precision bf16 --workflow lora_sft --mode auto
  --output-root profiling

The shell launches one background Python driver job per workload/backend pair,
assigns each job one GPU from the pool, waits for all jobs, and traps
INT/TERM/ERR to terminate the whole background process tree.
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
      qwen) printf '%s\n' qwen3_14b qwen3_30b_a3b ;;
      fundamental) printf '%s\n' matrix_1b mlp_1b ;;
      all) printf '%s\n' mlp dense moe qwen3_14b qwen3_30b_a3b matrix_1b mlp_1b ;;
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

declare -a gpu_values=()
declare -a workload_values=()
declare -a backend_values=()
declare -a pass_args=()
jobs_per_gpu=1
output_root="profiling"
run_name=""
precision="bf16"
workflow="lora_sft"

while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --gpus|--gpu-pool|--cuda-devices)
      shift
      while (($#)) && [[ "$1" != --* ]]; do
        gpu_values+=("$1")
        shift
      done
      ;;
    --workloads)
      shift
      while (($#)) && [[ "$1" != --* ]]; do
        workload_values+=("$1")
        shift
      done
      ;;
    --backends)
      shift
      while (($#)) && [[ "$1" != --* ]]; do
        backend_values+=("$1")
        shift
      done
      ;;
    --jobs-per-gpu)
      jobs_per_gpu="$2"
      shift 2
      ;;
    --output-root)
      output_root="$2"
      pass_args+=("$1" "$2")
      shift 2
      ;;
    --run-name)
      run_name="$2"
      pass_args+=("$1" "$2")
      shift 2
      ;;
    --precision)
      precision="$2"
      pass_args+=("$1" "$2")
      shift 2
      ;;
    --workflow)
      workflow="$2"
      pass_args+=("$1" "$2")
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

if ((${#workload_values[@]} == 0)); then
  workload_values=(qwen)
fi
if ((${#backend_values[@]} == 0)); then
  backend_values=(asym_only torch_only)
fi

mapfile -t workloads < <(split_values "${workload_values[@]}" | xargs -r -n1 printf '%s\n' | while read -r x; do expand_workloads "$x"; done | dedupe_lines)
mapfile -t backends < <(split_values "${backend_values[@]}" | xargs -r -n1 printf '%s\n' | while read -r x; do expand_backends "$x"; done | dedupe_lines)

if ((${#workloads[@]} == 0 || ${#backends[@]} == 0)); then
  echo "error: workloads/backends expanded to empty lists" >&2
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
    python "${PY_DRIVER}"
    --workloads "${workload}"
    --backends "${backend}"
    --cuda-devices "${gpu}"
    --skip-summary
    "${pass_args[@]}"
  )

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

echo "All profiling jobs completed."
