#!/usr/bin/env bash
# Capacity fit-probe cell runner (rebuilt 2026-07-25). One cell per invocation,
# serial protocol per agent/impls/model_capacity.md §2: sync-guard, node0+1
# sampler, availability watchdog (kill = the C_OOM verdict), timeout guard,
# nvidia-smi HBM polling, classification appended to the ledger.
#
# Usage: run_capacity_cell.sh CELL MODEL_DIR SPEC SEQ BATCH [FLOOR_GIB]
#   e.g. run_capacity_cell.sh X1_218b_128k /path/d5e-dense-215b-112x12288 \
#          'asym_cpuadamwds|unsloth-ohbm0|ligerloss1' 128000 1 25
# Env: CAPDIR (default profiling_results/capacity_push_c17), TIMEOUT_S (6000),
#      TEMPLATE (qwen3), GPU_ID (0); any ASYM_*/UNSLOTH_* capacity-mode envs are
#      inherited by the trainer and dumped into the log header for provenance.
# Note OVERWRITE=true is forced: the driver dedupes by run-dir signature and the
# capacity env gates are NOT part of the signature (a repeat spec would SKIP as
# existing-complete — trap #17).
set -u
CELL="$1"; MODEL="$2"; SPEC="$3"; SEQ="$4"; BATCH="$5"; FLOOR="${6:-25}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CAPDIR="${CAPDIR:-${ROOT}/profiling_results/capacity_push_c17}"
TIMEOUT_S="${TIMEOUT_S:-6000}"
mkdir -p "${CAPDIR}/logs" "${CAPDIR}/traces" "${CAPDIR}/runs"
LOG="${CAPDIR}/logs/cell_${CELL}.log"
TRACE="${CAPDIR}/traces/trace_${CELL}.tsv"
KILLED="${CAPDIR}/logs/cell_${CELL}.killed"
rm -f "${KILLED}"

avail_gib() {
  local memfree=0 file=0 shmem=0
  for n in 0 1; do
    # node meminfo rows are "Node N <Key>: <val> kB" — key is field 3
    while read -r _ _ key val _; do
      case "$key" in
        MemFree:) memfree=$((memfree+val));;
        FilePages:) file=$((file+val));;
        Shmem:) shmem=$((shmem+val));;
      esac
    done < "/sys/devices/system/node/node$n/meminfo"
  done
  echo $(( (memfree + file - shmem) / 1048576 ))
}

sync
sleep 3
T0=$(date +%s)
{
  echo "[cell ${CELL}] $(date -Iseconds) model=${MODEL} spec=${SPEC} seq=${SEQ}x${BATCH} floor=${FLOOR}"
  echo "[cell ${CELL}] capacity-env: $(env | grep -E '^(ASYM_|UNSLOTH_GC_)' | tr '\n' ' ')"
} >> "${LOG}"

setsid bash "${ROOT}/scripts/lf/capacity/mem_sampler.sh" "${TRACE}" &
SAMPLER_PID=$!

setsid env OUTPUT_ROOT="${CAPDIR}/runs" TEMPLATE="${TEMPLATE:-qwen3}" \
  HOST_MEM_WATCHDOG_FLOOR_GB="${FLOOR}" OVERWRITE=true \
  RUNS="${MODEL}|1 ; ${SPEC} ; ${SEQ}|${BATCH}|1 ; none|false|false|false|false|false" \
  bash "${ROOT}/scripts/lf/profile_lora_lf_test_source.sh" >> "${LOG}" 2>&1 &
DRIVER_PID=$!

PEAK_HBM_MIB=0
while kill -0 "${DRIVER_PID}" 2>/dev/null; do
  NOW=$(date +%s)
  AVAIL=$(avail_gib)
  HBM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_ID:-0}" 2>/dev/null | head -1)
  case "${HBM}" in (*[!0-9]*|'') ;; (*) [ "${HBM}" -gt "${PEAK_HBM_MIB}" ] && PEAK_HBM_MIB=${HBM};; esac
  if [ "${AVAIL}" -lt "${FLOOR}" ]; then
    echo "watchdog: avail ${AVAIL} < floor ${FLOOR} at $(date -Iseconds)" > "${KILLED}"
    kill -TERM -- "-${DRIVER_PID}" 2>/dev/null
    sleep 8
    kill -KILL -- "-${DRIVER_PID}" 2>/dev/null
    break
  fi
  if [ $((NOW - T0)) -gt "${TIMEOUT_S}" ]; then
    echo "watchdog: timeout ${TIMEOUT_S}s" > "${KILLED}.timeout"
    kill -TERM -- "-${DRIVER_PID}" 2>/dev/null
    sleep 8
    kill -KILL -- "-${DRIVER_PID}" 2>/dev/null
    break
  fi
  sleep 2
done
wait "${DRIVER_PID}" 2>/dev/null
T1=$(date +%s)

kill "${SAMPLER_PID}" 2>/dev/null
pkill -f "mem_sampler.sh ${TRACE}" 2>/dev/null

ROW=$("${ROOT}/.venv/bin/python" "${ROOT}/scripts/lf/capacity/classify_cell.py" "${CELL}" "${CAPDIR}" --floor "${FLOOR}" --t0 "${T0}" --t1 "${T1}" --hbm-mib "${PEAK_HBM_MIB}")
WALL=$((T1 - T0))
sed -i "s|^${CELL}\t\(.*\)\t-\t-$|${CELL}\t\1\t-\t${WALL}|" "${CAPDIR}/ledger.tsv" 2>/dev/null
echo "[cell ${CELL}] done wall=${WALL}s ${ROW}" | tee -a "${LOG}"
sleep 20
