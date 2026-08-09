#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# NeMo / Megatron-Bridge LoRA-SFT profiling driver — the `nemo` backend ONLY.
#
# Sibling of profile_lora_lf_test_source.sh with the SAME measurement protocol
# (w1+m2 latency law, effective tok/s over post-warmup steps, one run at a
# time, host-mem watchdog, OOM verdicts) but launching Megatron-Bridge
# (Megatron-LM with LoRA support) out of .venv-nemo instead of the LF stack.
# Runs INSIDE the asym_sft container. Bootstrap: scripts/lf/bootstrap_nemo_venv.sh.
#
# RUNS row (house shape, backend must be `nemo`):
#   model|ranks ; nemo|<recompute>|ligerloss1 ; seq|batch|ga ; none|false|false|false|false|false
# recompute tokens:
#   recomp          full uniform recompute (NeMo's strongest recompute; the
#                   analogue of the baselines' `recomp` cells)
#   norecomp        no recompute, no offload
#   actoff          fine-grained activation offloading, every offloadable
#                   module, no recompute (NeMo's strongest activation offload;
#                   mutually exclusive with recompute upstream)
#   selrecomp       selective recompute (core_attn,moe_act,layernorm,mlp)
#   selrecomp-actoff selective recompute + offload of the non-recomputed set
# ligerloss field: accepted for interface parity; nemo always trains with TE
# fused cross entropy (the ligerloss analogue), so only `ligerloss1` is valid.
#
# Example:
#   RUNS='q3-30b-a3b|2 ; nemo|recomp|ligerloss1 ; 384000|1|1 ; none|false|false|false|false|false' \
#     bash scripts/lf/profile_lora_nemo.sh
# =============================================================================

SFT_ROOT=${SFT_ROOT:-$(cd ../.. && pwd)}
ROOT=${ROOT:-${SFT_ROOT}/third_party/AsymGEMM}
ENV_DIR=${ENV_DIR:-${ROOT}/.venv-nemo}
# FA4 sibling env (bootstrap_nemo_venv_fa4.sh). REQUIRED for the 256-dim-head
# attention families: TE 2.16's cuDNN fused kernels support head_dim<=128 for
# training (DeepSeek-MLA (192,128) special-cased), so qwen3.5 (full-attn
# layers: head_dim 256 + attn_output_gate) and glm4.7-flash (MLA qk 192+64=256,
# v 256) fall back to UnfusedDotProductAttention WITHOUT FlashAttention —
# materializing the [H,S,S] fp32 score tensor (61-76 GiB @32k = the measured
# 08-02 walls). TE detects flash-attn-4 (v4_is_installed) and its FA gate
# covers head_dim<=256 on sm100, so these models route to .venv-nemo-fa4.
ENV_DIR_FA4=${ENV_DIR_FA4:-${ROOT}/.venv-nemo-fa4}
ENV_PYTHON=${ENV_PYTHON:-${ENV_DIR}/bin/python}
NEMO_RUNNER=${NEMO_RUNNER:-${ROOT}/scripts/lf/run_nemo_lora_sft.py}

nemo_env_for_model() {
  # $1 = model key/path -> venv dir. fa4 for the unfused-fallback families.
  case "${1,,}" in
    *q3.5*|*qwen3.5*|*qwen3_5*|*glm4.7*|*glm4_7*|*glm-4.7*) printf '%s\n' "${ENV_DIR_FA4}" ;;
    *) printf '%s\n' "${ENV_DIR}" ;;
  esac
}

GPU_POOL=${GPU_POOL:-0,1}
DIST_LAUNCHER=${DIST_LAUNCHER:-torchrun}

MAX_STEPS=${MAX_STEPS:-2}          # measured steps (house latency law: 1w+2m)
WARMUP_STEPS=${WARMUP_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
LORA_DROPOUT=${LORA_DROPOUT:-0.00}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-16}
SEED=${SEED:-42}
PRECISION=${PRECISION:-bf16}

NEMO_EP=${NEMO_EP:-}               # default: = ranks (normal EP over all ranks)
NEMO_TP=${NEMO_TP:-1}
NEMO_LOAD_WEIGHTS=${NEMO_LOAD_WEIGHTS:-true}
NEMO_MOE_DISPATCHER=${NEMO_MOE_DISPATCHER:-alltoall}

OUTPUT_ROOT=${OUTPUT_ROOT:-$(pwd)/profiling_results/profiling_nemo}
RUN_NAME=${RUN_NAME:-}
OVERWRITE=${OVERWRITE:-false}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-true}
DRY_RUN=${DRY_RUN:-false}
RUN_TIMEOUT_SECONDS=${RUN_TIMEOUT_SECONDS:-5400}
HOST_MEM_WATCHDOG_FLOOR_GB=${HOST_MEM_WATCHDOG_FLOOR_GB:-50}
PREFLIGHT_SETTLE_SECONDS=${PREFLIGHT_SETTLE_SECONDS:-30}
_RUNS_LOG="${RUNS_LOG:-${ROOT}/scripts/lf/runs.log}"

export HF_HOME=${HF_HOME:-/scratch_local/user_data/shutian/kevin/cache/huggingface}
export TOKENIZERS_PARALLELISM=false
EXPANDABLE_SEG=${EXPANDABLE_SEG:-true}

declare -A M=(
  [q3-30b-a3b]="Qwen/Qwen3-30B-A3B"
  [q3-235b-a22b]="Qwen/Qwen3-235B-A22B"
  [q3.5-35b-a3b]="Qwen/Qwen3.5-35B-A3B"
  [q3.5-122b-a10b]="Qwen/Qwen3.5-122B-A10B"
  [q3-32b]="Qwen/Qwen3-32B"
  [mixtral-8x22b]="mistralai/Mixtral-8x22B-v0.1"
  [glm4.5-air]="zai-org/GLM-4.5-Air"
  [glm4.7-flash]="zai-org/GLM-4.7-Flash"
)

die() { echo "error: $*" >&2; exit 2; }

# ── RUNS parsing (house shape) ──────────────────────────────────────────────
if [[ -z "${RUNS+x}" ]]; then
  die "RUNS is required, e.g. RUNS='q3-30b-a3b|2 ; nemo|recomp|ligerloss1 ; 16000|1|1 ; none|false|false|false|false|false'"
fi
_runs_env_lines="${RUNS//||/$'\n'}"
RUN_ROWS=()
while IFS= read -r _run; do
  _run="${_run#"${_run%%[![:space:]]*}"}"; _run="${_run%"${_run##*[![:space:]]}"}"
  [[ -n "${_run}" ]] && RUN_ROWS+=("${_run}")
done <<< "${_runs_env_lines}"
(( ${#RUN_ROWS[@]} > 0 )) || die "RUNS is empty"

recompute_to_env() {
  # $1 = recompute token -> sets NEMO_RECOMPUTE_MODE / NEMO_ACT_OFFLOAD_MODE
  case "$1" in
    recomp)            NEMO_RECOMPUTE_MODE=full NEMO_ACT_OFFLOAD_MODE=0 ;;
    norecomp)          NEMO_RECOMPUTE_MODE=none NEMO_ACT_OFFLOAD_MODE=0 ;;
    actoff)            NEMO_RECOMPUTE_MODE=none NEMO_ACT_OFFLOAD_MODE=1 ;;
    selrecomp)         NEMO_RECOMPUTE_MODE=sel  NEMO_ACT_OFFLOAD_MODE=0 ;;
    selrecomp-actoff)  NEMO_RECOMPUTE_MODE=sel  NEMO_ACT_OFFLOAD_MODE=1 ;;
    *) die "unknown nemo recompute token '$1' (recomp|norecomp|actoff|selrecomp|selrecomp-actoff)" ;;
  esac
}

resolve_model_path() {
  # $1 = HF repo id -> local snapshot dir (cache only, no network)
  "${ENV_PYTHON}" - "$1" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], local_files_only=True))
PY
}

cpu_pool_available_gb() {
  # CPU pool = NUMA nodes that own CPUs (project_rules.md §1: nodes 0+1; the
  # HBM-as-NUMA nodes have empty cpulists). Available = MemFree + reclaimable
  # page cache (FilePages - Shmem) — raw MemFree false-fires the watchdog the
  # moment a 61 GB safetensors read lands in page cache (house
  # host_mem_watchdog_avail_kb method, run_lf_lora_sft.sh).
  local node cpus free_kb file_kb shmem_kb total=0 seen=0
  for node in /sys/devices/system/node/node*; do
    [[ -r "${node}/cpulist" && -r "${node}/meminfo" ]] || continue
    cpus="$(tr -d '[:space:]' < "${node}/cpulist" 2>/dev/null || true)"
    [[ -n "${cpus}" ]] || continue
    free_kb="$(awk '$3 == "MemFree:" {s += $4} END {print s+0}' "${node}/meminfo" 2>/dev/null || echo 0)"
    file_kb="$(awk '$3 == "FilePages:" {s += $4} END {print s+0}' "${node}/meminfo" 2>/dev/null || echo 0)"
    shmem_kb="$(awk '$3 == "Shmem:" {s += $4} END {print s+0}' "${node}/meminfo" 2>/dev/null || echo 0)"
    if (( file_kb > shmem_kb )); then
      free_kb=$(( free_kb + file_kb - shmem_kb ))
    fi
    total=$(( total + free_kb ))
    seen=1
  done
  if (( seen == 0 )); then
    free -g | awk 'NR==2{print $7}'
    return 0
  fi
  echo $(( total / 1024 / 1024 ))
}

gpu_list_csv() { echo "${GPU_POOL}"; }

preflight_guard() {
  # Project rule: GPU compute list must be EMPTY before launching; host above floor.
  local i tries=0
  while :; do
    local busy=0
    for i in ${GPU_POOL//,/ }; do
      local n
      n=$(nvidia-smi -i "${i}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l)
      (( n > 0 )) && busy=1
    done
    local avail
    avail=$(cpu_pool_available_gb)
    if (( busy == 0 )) && (( avail >= HOST_MEM_WATCHDOG_FLOOR_GB )); then
      return 0
    fi
    tries=$((tries+1))
    if (( tries > 60 )); then
      echo "[driver] preflight guard timeout (busy=${busy} avail=${avail}G floor=${HOST_MEM_WATCHDOG_FLOOR_GB}G)" >&2
      return 1
    fi
    sleep 20
  done
}

write_config_json() {
  local dir="$1"
  "${ENV_PYTHON}" - "${dir}" <<'PY' 2>/dev/null || true
import json, os, sys
cfg = {k: v for k, v in os.environ.items() if k.startswith(("NEMO_", "LORA_", "WARMUP", "MAX_STEPS", "SEED", "LEARNING_RATE"))}
cfg["_backend"] = "nemo (Megatron-Bridge)"
cfg["_venv"] = os.environ.get("ENV_DIR", "")
json.dump(cfg, open(os.path.join(sys.argv[1], "config.json"), "w"), indent=1, sort_keys=True)
PY
}

harvest_run() {
  # $1 run_dir  $2 ranks $3 batch $4 seq $5 ga $6 warmup $7 measured
  "${ENV_PYTHON}" - "$@" <<'PY'
import csv, json, os, re, sys

run_dir, ranks, batch, seq, ga, warmup, measured = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]))
log_path = os.path.join(run_dir, "train.log")
text = open(log_path, errors="replace").read()

iters = []  # (iteration, elapsed_ms)
for m in re.finditer(r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\): ([0-9.]+)", text):
    iters.append((int(m.group(1)), float(m.group(2))))

with open(os.path.join(run_dir, "step_samples.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["step_index", "step_milliseconds", "is_warmup"])
    for it, ms in iters:
        w.writerow([it, f"{ms:.1f}", str(it <= warmup).lower()])

meas = [ms for it, ms in iters if it > warmup]
eff = None
if len(meas) == measured and sum(meas) > 0:
    tokens_per_iter = batch * ranks * ga * seq  # GLOBAL tokens (dp = ranks, tp=1)
    eff = tokens_per_iter * len(meas) / (sum(meas) / 1000.0)

peaks = {}
for m in re.finditer(r"NEMO_PEAK_MEM rank=(\d+) max_allocated_bytes=(\d+) max_reserved_bytes=(\d+)", text):
    peaks[int(m.group(1))] = {"alloc": int(m.group(2)), "resv": int(m.group(3))}

goom = bool(re.search(r"CUDA out of memory|OutOfMemoryError|CUDNN_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation", text))
coom = "dropped below floor" in text
done = "NEMO_RUN_DONE" in text
if done and eff is not None:
    verdict = "TRAINED"
elif goom:
    verdict = "GOOM"
elif coom:
    verdict = "COOM"
else:
    verdict = "FAIL"

peak_resv = max((v["resv"] for v in peaks.values()), default=0)
smi_path = os.path.join(run_dir, "gpu_mem_peak.txt")
smi_peak_mib = 0
if os.path.exists(smi_path):
    try:
        smi_peak_mib = int(open(smi_path).read().strip() or 0)
    except ValueError:
        smi_peak_mib = 0

prof = {
    "verdict": verdict,
    "effective_tokens_per_second_global": round(eff) if eff else None,
    "measured_step_ms": meas,
    "warmup_step_ms": [ms for it, ms in iters if it <= warmup],
    "tokens_per_iter_global": batch * ranks * ga * seq,
    "ranks": ranks,
    "memory": {
        "peak_reserved_hbm_bytes_max_rank": peak_resv,
        "per_rank": peaks,
        "nvidia_smi_peak_used_mib_max_gpu": smi_peak_mib,
    },
}
json.dump(prof, open(os.path.join(run_dir, "profile.json"), "w"), indent=1)
print(f"VERDICT={verdict} eff_tps={round(eff) if eff else '-'} "
      f"peak_resv_gib={peak_resv/2**30:.1f} smi_peak_mib={smi_peak_mib}")
PY
}

ensure_jobs_tsv() {
  local config_root="$1"
  if [[ ! -e "${config_root}/jobs.tsv" ]]; then
    printf 'status\tgpus\tseq_len\tbatch_size\tgrad_accum\trecompute\tbackend\teff_tps\tjob_dir\tlog\n' > "${config_root}/jobs.tsv"
  fi
}

# ── main loop (strictly serial, per project rules) ──────────────────────────
overall_rc=0
for _run in "${RUN_ROWS[@]}"; do
  IFS=';' read -r _m _b _w _p _extra <<< "${_run}"
  _m="${_m// /}"; _b="${_b// /}"; _w="${_w// /}"; _p="${_p// /}"
  [[ -n "${_m}" && -n "${_b}" && -n "${_w}" && -z "${_extra:-}" ]] || die "bad RUNS row '${_run}'"

  model_key="${_m%%|*}"; ranks="${_m#*|}"
  [[ "${ranks}" =~ ^[0-9]+$ ]] || die "model field needs |ranks: '${_m}'"
  model_id="${M[${model_key}]:-${model_key}}"

  backend="${_b%%|*}"
  [[ "${backend}" == "nemo" ]] || die "this driver only supports the 'nemo' backend, got '${backend}'"
  rest="${_b#*|}"; recompute_tok="${rest%%|*}"
  liger_f="ligerloss1"; [[ "${rest}" == *"|"* ]] && liger_f="${rest#*|}"
  [[ "${liger_f}" == "ligerloss1" ]] || die "nemo always uses TE fused CE; only ligerloss1 is valid"
  recompute_to_env "${recompute_tok}"

  IFS='|' read -r seq_len batch ga <<< "${_w}"
  [[ "${seq_len}" =~ ^[0-9]+$ && "${batch}" =~ ^[0-9]+$ && "${ga}" =~ ^[0-9]+$ ]] || die "bad workload '${_w}'"
  [[ "${_p}" == "none|false|false|false|false|false" || -z "${_p}" ]] || die "nemo driver supports only policy none|false|false|false|false|false"

  ep="${NEMO_EP:-${ranks}}"
  job_env_dir="$(nemo_env_for_model "${model_key}")"
  ENV_PYTHON="${job_env_dir}/bin/python"
  if [[ "${job_env_dir}" != "${ENV_DIR}" ]]; then
    echo "[driver] ${model_key}: fa4 venv (${job_env_dir}) — TE unfused-fallback family needs FlashAttention-4"
  fi

  # ── output layout (house shape) ───────────────────────────────────────────
  drop_tag="drop$(printf '%03d' "$(awk -v d="${LORA_DROPOUT}" 'BEGIN{printf "%d", d*100}')")"
  model_tag="${model_key//./_}"
  config_dir="${OUTPUT_ROOT}/mock__lora__nemo__${PRECISION}/${RUN_NAME:+${RUN_NAME}_}${model_tag}__gpus${ranks}__b${batch}_s${seq_len}_ga${ga}_w${WARMUP_STEPS}_s${MAX_STEPS}_r${LORA_RANK}_a${LORA_ALPHA}_${drop_tag}"
  run_dir="${config_dir}/nemo__source__${recompute_tok}__polnone__routerwhole/b${batch}_s${seq_len}_ga${ga}"
  if [[ -e "${run_dir}/profile.json" && "${OVERWRITE}" != "true" ]]; then
    echo "[driver] SKIP existing ${run_dir} (OVERWRITE=false)"
    continue
  fi
  mkdir -p "${run_dir}"

  model_path="$(resolve_model_path "${model_id}")" || die "model ${model_id} not in local HF cache"

  echo "[driver] RUN ${model_key} ranks=${ranks} ep=${ep} ${recompute_tok} s=${seq_len} b=${batch} ga=${ga} -> ${run_dir}"
  if [[ "${DRY_RUN}" == "true" ]]; then continue; fi

  preflight_guard || { echo "[driver] preflight failed, skipping row" >&2; overall_rc=1; continue; }

  # env for the runner
  export NEMO_MODEL_PATH="${model_path}"
  export NEMO_SEQ_LEN="${seq_len}"
  export NEMO_MICRO_BATCH="${batch}"
  export NEMO_GRAD_ACCUM="${ga}"
  export NEMO_WARMUP_STEPS="${WARMUP_STEPS}"
  export NEMO_MEASURED_STEPS="${MAX_STEPS}"
  export NEMO_EP="${ep}"
  export NEMO_TP="${NEMO_TP}"
  export NEMO_RECOMPUTE="${NEMO_RECOMPUTE_MODE}"
  export NEMO_ACT_OFFLOAD="${NEMO_ACT_OFFLOAD_MODE}"
  export NEMO_LORA_RANK="${LORA_RANK}"
  export NEMO_LORA_ALPHA="${LORA_ALPHA}"
  export NEMO_LORA_DROPOUT="${LORA_DROPOUT}"
  export NEMO_SEED="${SEED}"
  export NEMO_LR="${LEARNING_RATE}"
  export NEMO_OUT_DIR="${run_dir}"
  export NEMO_LOAD_WEIGHTS="${NEMO_LOAD_WEIGHTS}"
  export NEMO_MOE_DISPATCHER="${NEMO_MOE_DISPATCHER}"
  export CUDA_VISIBLE_DEVICES="$(gpu_list_csv)"
  if [[ "${EXPANDABLE_SEG}" == "true" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  fi

  write_config_json "${run_dir}"
  {
    echo "torchrun --nproc_per_node=${ranks} ${NEMO_RUNNER}"
    echo "venv=${job_env_dir}"
    env | grep -E '^(NEMO_|CUDA_VISIBLE|PYTORCH_CUDA)' | sort
  } > "${run_dir}/command.txt"

  master_port=$((29500 + RANDOM % 1000))
  log="${run_dir}/train.log"
  : > "${log}"
  : > "${run_dir}/gpu_mem_peak.txt"

  set +e
  timeout --signal=KILL "${RUN_TIMEOUT_SECONDS}" \
    "${job_env_dir}/bin/torchrun" --nproc_per_node="${ranks}" \
      --master_addr=127.0.0.1 --master_port="${master_port}" \
      "${NEMO_RUNNER}" >> "${log}" 2>&1 &
  train_pid=$!

  # watchdog: host floor + nvidia-smi peak sampling, house phrasing for COOM
  (
    peak=0
    while kill -0 "${train_pid}" 2>/dev/null; do
      avail=$(cpu_pool_available_gb)
      if (( avail < HOST_MEM_WATCHDOG_FLOOR_GB )); then
        echo "[host-mem-watchdog] CPU-node available memory ${avail} GiB dropped below floor ${HOST_MEM_WATCHDOG_FLOOR_GB} GiB; interrupting training before the kernel OOM killer fires (soft host OOM)." >> "${log}"
        pkill -KILL -P "${train_pid}" 2>/dev/null
        kill -KILL "${train_pid}" 2>/dev/null
        break
      fi
      for g in ${GPU_POOL//,/ }; do
        u=$(nvidia-smi -i "${g}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        [[ "${u:-0}" =~ ^[0-9]+$ ]] && (( u > peak )) && peak=${u}
      done
      echo "${peak}" > "${run_dir}/gpu_mem_peak.txt"
      sleep 5
    done
  ) &
  watchdog_pid=$!

  wait "${train_pid}"; train_rc=$?
  kill "${watchdog_pid}" 2>/dev/null; wait "${watchdog_pid}" 2>/dev/null
  set -e

  # reap any straggler ranks bound to our GPUs (explicit pids only, no pkill -f)
  for g in ${GPU_POOL//,/ }; do
    for p in $(nvidia-smi -i "${g}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      kill -9 "${p}" 2>/dev/null || true
    done
  done
  sleep "${PREFLIGHT_SETTLE_SECONDS}"

  harvest_line="$(harvest_run "${run_dir}" "${ranks}" "${batch}" "${seq_len}" "${ga}" "${WARMUP_STEPS}" "${MAX_STEPS}")" || harvest_line="VERDICT=FAIL eff_tps=-"
  echo "[driver] ${harvest_line} (rc=${train_rc})"

  verdict="${harvest_line#VERDICT=}"; verdict="${verdict%% *}"
  eff_tps="$(echo "${harvest_line}" | sed -n 's/.*eff_tps=\([0-9-]*\).*/\1/p')"
  status="failed:${train_rc}"
  [[ "${verdict}" == "TRAINED" ]] && status="ok"
  ensure_jobs_tsv "${config_dir}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${status}" "${GPU_POOL}" "${seq_len}" "${batch}" "${ga}" "${recompute_tok}" \
    "nemo" "${eff_tps}" "${run_dir}" "${log}" >> "${config_dir}/jobs.tsv"
  printf '%s\t%s\n' "$(date '+%F %T')" "nemo ${model_key}|${ranks} ${recompute_tok} s=${seq_len} b=${batch} ga=${ga} -> ${verdict} eff=${eff_tps}" >> "${_RUNS_LOG}" 2>/dev/null || true

  if [[ "${verdict}" != "TRAINED" && "${CONTINUE_ON_ERROR}" != "true" ]]; then
    overall_rc=1
    break
  fi
done
exit "${overall_rc}"
