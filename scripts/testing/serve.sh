#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-start}"
GPU_POOL="${GPU_POOL:-${GPUS:-0,1}}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
BASE_PORT="${BASE_PORT:-30000}"
HOST="${HOST:-0.0.0.0}"
HEALTH_HOST="${HEALTH_HOST:-127.0.0.1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.95}"
SERVED_MODEL_PREFIX="${SERVED_MODEL_PREFIX:-gpu-test}"
RUN_DIR="${RUN_DIR:-/tmp/sglang-gpu-pool}"
PID_DIR="${PID_DIR:-${RUN_DIR}/pids}"
LOAD_PID_DIR="${LOAD_PID_DIR:-${RUN_DIR}/load-pids}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
LOAD_LOG_DIR="${LOAD_LOG_DIR:-${RUN_DIR}/load-logs}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-600}"
LOAD_ENABLE="${LOAD_ENABLE:-1}"
LOAD_WORKERS_PER_SERVER="${LOAD_WORKERS_PER_SERVER:-32}"
LOAD_MAX_NEW_TOKENS="${LOAD_MAX_NEW_TOKENS:-128}"
LOAD_TIMEOUT_SEC="${LOAD_TIMEOUT_SEC:-120}"
LOAD_SLEEP_SEC="${LOAD_SLEEP_SEC:-0}"
LOAD_PROMPT="${LOAD_PROMPT:-This is a sustained GPU load test. Continue with detailed technical reasoning about matrix multiplication, attention, kernels, memory bandwidth, scheduling, and throughput.}"
SITE_PACKAGES="${SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages:/usr/lib/python3/dist-packages:/usr/lib/python3.12/dist-packages}"
SGLANG_SOURCE="${SGLANG_SOURCE:-/sgl-workspace/sglang/python}"
SGL_KERNEL_SOURCE="${SGL_KERNEL_SOURCE:-/sgl-workspace/sglang/sgl-kernel/python/sgl_kernel}"
SGL_KERNEL_WHEEL="${SGL_KERNEL_WHEEL:-/usr/local/lib/python3.12/dist-packages/sgl_kernel}"
SGL_KERNEL_OVERLAY="${SGL_KERNEL_OVERLAY:-${RUN_DIR}/sgl-kernel-overlay}"
SGLANG_PYTHONPATH="${SGLANG_SOURCE}:${SGL_KERNEL_OVERLAY}:${SITE_PACKAGES}${EXTRA_PYTHONPATH:+:${EXTRA_PYTHONPATH}}"

mkdir -p "${PID_DIR}" "${LOAD_PID_DIR}" "${LOG_DIR}" "${LOAD_LOG_DIR}"

read -r -a GPU_IDS <<< "${GPU_POOL//,/ }"

prepare_kernel_overlay() {
  if [[ -f "${SGL_KERNEL_OVERLAY}/sgl_kernel/__init__.py" ]] &&
    compgen -G "${SGL_KERNEL_OVERLAY}/sgl_kernel/sm100/common_ops*.so" >/dev/null; then
    return 0
  fi

  rm -rf "${SGL_KERNEL_OVERLAY}"
  mkdir -p "${SGL_KERNEL_OVERLAY}"
  cp -r --no-preserve=mode,ownership,timestamps \
    "${SGL_KERNEL_SOURCE}" \
    "${SGL_KERNEL_OVERLAY}/"

  if compgen -G "${SGL_KERNEL_WHEEL}/*.so" >/dev/null; then
    cp --no-preserve=mode,ownership,timestamps \
      "${SGL_KERNEL_WHEEL}"/*.so \
      "${SGL_KERNEL_OVERLAY}/sgl_kernel/"
  fi

  local arch
  for arch in sm90 sm100; do
    if compgen -G "${SGL_KERNEL_WHEEL}/${arch}/*.so" >/dev/null; then
      mkdir -p "${SGL_KERNEL_OVERLAY}/sgl_kernel/${arch}"
      cp --no-preserve=mode,ownership,timestamps \
        "${SGL_KERNEL_WHEEL}/${arch}"/*.so \
        "${SGL_KERNEL_OVERLAY}/sgl_kernel/${arch}/"
    fi
  done
}

gpu_port() {
  local wanted_gpu="$1"
  local idx=0
  local gpu
  for gpu in "${GPU_IDS[@]}"; do
    if [[ "${gpu}" == "${wanted_gpu}" ]]; then
      echo $((BASE_PORT + idx))
      return 0
    fi
    idx=$((idx + 1))
  done
  return 1
}

kill_group_from_file() {
  local label="$1"
  local pid_file="$2"

  if [[ ! -s "${pid_file}" ]]; then
    echo "${label}: no pid file"
    return 0
  fi

  local pid
  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "${label}: stopping process group ${pid}"
    kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
  else
    echo "${label}: pid ${pid} is not running"
  fi
  rm -f "${pid_file}"
}

stop_gpu() {
  local gpu="$1"
  kill_group_from_file "GPU ${gpu} load" "${LOAD_PID_DIR}/gpu_${gpu}.pid"
  kill_group_from_file "GPU ${gpu} server" "${PID_DIR}/gpu_${gpu}.pid"
}

stop_all() {
  local gpu
  for gpu in "${GPU_IDS[@]}"; do
    stop_gpu "${gpu}"
  done
}

start_server() {
  local gpu="$1"
  local port="$2"
  local pid_file="${PID_DIR}/gpu_${gpu}.pid"
  local log_file="${LOG_DIR}/gpu_${gpu}.log"
  local served_model_name="${SERVED_MODEL_PREFIX}-${gpu}"

  if [[ -s "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    echo "GPU ${gpu}: server already running with pid $(cat "${pid_file}") on port ${port}"
    return 0
  fi

  rm -f "${pid_file}"
  echo "GPU ${gpu}: starting SGLang server on port ${port}"
  setsid env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${SGLANG_PYTHONPATH}" \
    python3 -S -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --load-format dummy \
      --mem-fraction-static "${MEM_FRACTION_STATIC}" \
      --host "${HOST}" \
      --port "${port}" \
      --served-model-name "${served_model_name}" \
      >"${log_file}" 2>&1 &

  echo "$!" > "${pid_file}"
  echo "GPU ${gpu}: server pid $(cat "${pid_file}"), log ${log_file}"
}

wait_for_server() {
  local gpu="$1"
  local port="$2"
  local pid_file="${PID_DIR}/gpu_${gpu}.pid"
  local log_file="${LOG_DIR}/gpu_${gpu}.log"
  local url="http://${HEALTH_HOST}:${port}/generate"
  local start_ts
  start_ts="$(date +%s)"

  echo "GPU ${gpu}: waiting for ${url}"
  while true; do
    if curl -fsS --max-time 30 \
      -H "Content-Type: application/json" \
      -d '{"text":"ready","sampling_params":{"temperature":0,"max_new_tokens":1}}' \
      "${url}" >/dev/null 2>&1; then
      echo "GPU ${gpu}: server is ready"
      return 0
    fi

    if [[ -s "${pid_file}" ]] && ! kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
      echo "GPU ${gpu}: server exited while starting; last log lines:"
      tail -80 "${log_file}" || true
      return 1
    fi

    if (( "$(date +%s)" - start_ts > READY_TIMEOUT_SEC )); then
      echo "GPU ${gpu}: timed out waiting for readiness; last log lines:"
      tail -80 "${log_file}" || true
      return 1
    fi

    sleep 2
  done
}

start_load() {
  local gpu="$1"
  local port="$2"
  local url="http://${HEALTH_HOST}:${port}"
  local pid_file="${LOAD_PID_DIR}/gpu_${gpu}.pid"
  local log_file="${LOAD_LOG_DIR}/gpu_${gpu}.log"

  if [[ -s "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    echo "GPU ${gpu}: load loop already running with pid $(cat "${pid_file}")"
    return 0
  fi

  rm -f "${pid_file}"
  echo "GPU ${gpu}: starting ${LOAD_WORKERS_PER_SERVER} request loops against ${url}/generate"
  setsid bash -c '
    set -uo pipefail
    url="$1"
    workers="$2"
    max_new_tokens="$3"
    timeout_sec="$4"
    sleep_sec="$5"
    prompt="$6"

    echo "load supervisor $$ targeting ${url}/generate with ${workers} workers"
    for ((worker = 0; worker < workers; worker++)); do
      (
        count=0
        while true; do
          curl -sS --max-time "${timeout_sec}" \
            -H "Content-Type: application/json" \
            -d "{\"text\":\"${prompt}\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":${max_new_tokens},\"ignore_eos\":true}}" \
            "${url}/generate" >/dev/null || true
          count=$((count + 1))
          if (( count % 100 == 0 )); then
            echo "worker ${worker}: ${count} requests sent"
          fi
          if [[ "${sleep_sec}" != "0" ]]; then
            sleep "${sleep_sec}"
          fi
        done
      ) &
    done
    wait
  ' _ "${url}" "${LOAD_WORKERS_PER_SERVER}" "${LOAD_MAX_NEW_TOKENS}" "${LOAD_TIMEOUT_SEC}" "${LOAD_SLEEP_SEC}" "${LOAD_PROMPT}" \
    >"${log_file}" 2>&1 &

  echo "$!" > "${pid_file}"
  echo "GPU ${gpu}: load pid $(cat "${pid_file}"), log ${log_file}"
}

start_all() {
  local idx=0
  local gpu

  prepare_kernel_overlay

  for gpu in "${GPU_IDS[@]}"; do
    start_server "${gpu}" "$((BASE_PORT + idx))"
    idx=$((idx + 1))
  done

  idx=0
  for gpu in "${GPU_IDS[@]}"; do
    wait_for_server "${gpu}" "$((BASE_PORT + idx))"
    idx=$((idx + 1))
  done

  if [[ "${LOAD_ENABLE}" == "1" ]]; then
    idx=0
    for gpu in "${GPU_IDS[@]}"; do
      start_load "${gpu}" "$((BASE_PORT + idx))"
      idx=$((idx + 1))
    done
  fi

  status
}

status() {
  local idx=0
  local gpu
  for gpu in "${GPU_IDS[@]}"; do
    local port=$((BASE_PORT + idx))
    local server_pid_file="${PID_DIR}/gpu_${gpu}.pid"
    local load_pid_file="${LOAD_PID_DIR}/gpu_${gpu}.pid"
    local server_pid="-"
    local load_pid="-"

    if [[ -s "${server_pid_file}" ]] && kill -0 "$(cat "${server_pid_file}")" 2>/dev/null; then
      server_pid="$(cat "${server_pid_file}")"
    fi
    if [[ -s "${load_pid_file}" ]] && kill -0 "$(cat "${load_pid_file}")" 2>/dev/null; then
      load_pid="$(cat "${load_pid_file}")"
    fi

    echo "GPU ${gpu}: port=${port} server_pid=${server_pid} load_pid=${load_pid}"
    idx=$((idx + 1))
  done

  echo
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
}

case "${ACTION}" in
  start)
    start_all
    ;;
  stop)
    stop_all
    ;;
  stop-gpu)
    if [[ $# -ne 2 ]]; then
      echo "usage: $0 stop-gpu <gpu-id>" >&2
      exit 2
    fi
    stop_gpu "$2"
    ;;
  status)
    status
    ;;
  *)
    echo "usage: $0 [start|stop|stop-gpu <gpu-id>|status]" >&2
    exit 2
    ;;
esac
