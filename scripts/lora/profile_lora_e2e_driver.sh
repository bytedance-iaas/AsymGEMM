#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# User-Specified Parameters
# =============================================================================
# Edit this section for the default run. CLI flags override these values.
# List defaults use the same comma-separated format accepted by CLI flags.

ROOT=${ROOT:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM}

GPU_POOL="0,1,3"
# Per-workload layers can be set as "workload|layers", e.g. "moe-604m-a75m|2".
# Use "workload|all" to profile the full configured/HF model depth.
# Dense workload names are total-model labels; MoE workload names are per-layer routed-expert total/active labels.
# WORKLOADS="mlp_3b,mm_3b,dense_3b,moe-604m-a75m,mlp,dense,moe,dense_14b,moe-604m-a38m"
# WORKLOADS="dense_3b,moe-604m-a75m|2"
# WORKLOADS="mlp_3b,mm_3b,dense_3b,moe-604m-a75m|4"
WORKLOADS="Qwen/Qwen3-30B-A3B|4,Qwen/Qwen3-235B-A22B-Instruct-2507|4"
# WORKLOADS="Qwen/Qwen3-30B-A3B|all"
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
PROFILE_LAYERS=1
BATCH_SIZE=8
# SEQ_LENS="64,128,256,512,640,768,896,1024,2048,3072,4096,6144,8192,10240,16384,20480"
SEQ_LENS="2048,4096"
# Expert recompute policy specs:
#   none, splitXX, tokXX-ckpt, tokXX-act-ckpt
# splitXX measures split/merge overhead only.
# tokXX-ckpt checkpoints the full selected expert body.
# tokXX-act-ckpt saves gate/up and checkpoints activation/down only.
EXPERT_RECOMPUTE_POLICIES="none,split0,tok0-ckpt,tok0-act-ckpt,split512,tok512-ckpt,tok512-act-ckpt"
MLP_INTERMEDIATE_DIM=0
MLP_EXPANSION=4
LORA_RANK=64
LORA_ALPHA=128
VOCAB_ROWS=4096
MOE_MODE="contiguous"
MOE_ROUTE_PATTERN="learned"
HF_LAYER_INDEX=0
HF_CACHE_DIR=""
HF_LOCAL_FILES_ONLY=false
PROFILE_SEED=1234
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
CONTINUE_ON_ERROR=true

PY_DRIVER="${ROOT}/scripts/lora/profile_lora_e2e_driver.py"
PLOT_RECOMPUTE_SCRIPT="${ROOT}/scripts/plotting/plot_activation_recompute_sweep.py"

usage() {
  cat <<USAGE
Usage:
  scripts/lora/profile_lora_e2e_driver.sh [options]

Defaults:
  --gpus ${GPU_POOL}
  --workloads ${WORKLOADS}
  --backends ${BACKENDS}
  --profilers ${PROFILERS}
  --seq-lens ${SEQ_LENS}
  --expert-recompute-policies ${EXPERT_RECOMPUTE_POLICIES}
  --moe-route-pattern ${MOE_ROUTE_PATTERN}
  --hf-layer-index ${HF_LAYER_INDEX}
  --profile-seed ${PROFILE_SEED}
  --recompute ${RECOMPUTE}
  --overwrite ${OVERWRITE}
  --continue-on-error ${CONTINUE_ON_ERROR}

Shell options:
  --gpus LIST                         Physical GPU pool. Accepts 0,1 or "0 1".
  --workloads LIST                    Workloads, aliases, or HF model ids. Use workload|layers.
  --backends LIST                     asym, torch, kt, or all.
  --profilers LIST                    source, nsys, cpu, ncu, or all.
  --seq-lens LIST                     Sequence lengths. Accepts 64,128 or "64 128".
  --expert-recompute-policies LIST     MoE expert policies: none,split128,tok128-ckpt,tok128-act-ckpt.
  --jobs-per-gpu N                    Concurrent Python driver jobs per GPU.
  --python-bin PATH                   Python interpreter. Default: PYTHON or python3.
  --output-root PATH                  Base output root. Default: profiling.
                                      Default layout: <root>/lora_e2e_<precision>/<workload>__b<batch>_s<seqs>_r<rank>_a<alpha>
  --run-name NAME                     Optional config directory under lora_e2e_<precision>.
  --precision NAME                    Result precision label and output precision directory suffix.
  --attention-impl NAME               Attention implementation: sdpa, fa2, fa3, or fa4.
  --mode NAME                         Result filename mode.
  --moe-route-pattern balanced|learned MoE routing for profiles.
  --hf-layer-index N                  Qwen decoder layer used by Qwen MoE workloads.
  --hf-cache-dir PATH                 Optional Hugging Face cache directory.
  --hf-local-files-only true|false     Do not download missing Hugging Face files.
  --profile-seed N                    Seed for generated profiling batches and learned routing.
  --plot true|false                   Write per-config plots and a top-level combined plot directory after profiling.
  --plot-output-dir PATH              Plot output directory.
  --recompute norecomp|recomp|both     Run without recompute, with recompute, or both.
                                      Active expert policies run checkpoint expert recompute; layer recompute is only used for policy none.
  --overwrite true|false              Re-run completed result dirs. Default false skips them.
  --continue-on-error true|false      Keep sweeping if a point OOMs or fails. Default true records failed rows.
  -h, --help                          Show this help.

Unknown options are passed through to scripts/lora/profile_lora_e2e_driver.py, so common
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
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]_-' '_' | sed -e 's/^[_-]*//' -e 's/[_-]*$//'
}

workload_label() {
  local value="$1"
  local base="${value}"
  local layers=""
  local label
  if [[ "${value}" == *"|"* ]]; then
    base="${value%%|*}"
    layers="${value#*|}"
  fi
  base="${base%/}"
  base="${base##*/}"
  label="$(safe_label "${base}")"
  if [[ -n "${layers}" && "${layers}" != "all" ]]; then
    label="${label}-l$(safe_label "${layers}")"
  fi
  printf '%s\n' "${label}"
}

config_root_path() {
  local workload="$1"
  local seq_len="$2"
  local config_label
  if [[ -n "${run_name}" ]]; then
    if ((${#seq_lens[@]} > 1)); then
      config_label="$(safe_label "${run_name}__s${seq_len}")"
    else
      config_label="$(safe_label "${run_name}")"
    fi
  else
    config_label="$(safe_label "$(workload_label "${workload}")__b${batch_size}_s${seq_len}_r${LORA_RANK}_a${LORA_ALPHA}")"
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
    qwen) printf '%s\n' "dense_14b${suffix}" "Qwen/Qwen3-30B-A3B${suffix}" ;;
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

normalize_util_label() {
  local suffix="$1"
  local digits
  case "${suffix}" in
    1|1.0|1.00|1.000) printf '%s\n' "util100"; return ;;
    0|0.0|0.00|0.000) printf '%s\n' "util000"; return ;;
  esac
  if [[ "${suffix}" =~ ^0\.([0-9]+)$ ]]; then
    digits="${BASH_REMATCH[1]}"
    [[ "${#digits}" -eq 1 ]] && digits="${digits}0"
    printf 'util%03d\n' "$((10#${digits:0:3}))"
  elif [[ "${suffix}" =~ ^0([0-9]+)$ ]]; then
    digits="${BASH_REMATCH[1]}"
    printf 'util%03d\n' "$((10#${digits}))"
  elif [[ "${suffix}" =~ ^[0-9]+$ ]]; then
    [[ "${suffix}" == "1" ]] && { printf '%s\n' "util100"; return; }
    printf 'util%03d\n' "$((10#${suffix}))"
  else
    die "invalid util policy threshold '${suffix}'"
  fi
}

normalize_expert_recompute_policy() {
  local raw="${1,,}"
  raw="${raw//_/-}"
  case "${raw}" in
    ""|none|off|false|0) printf '%s\n' "none"; return ;;
  esac
  if [[ "${raw}" =~ ^split([0-9]+)$ ]]; then
    printf 'split%d\n' "$((10#${BASH_REMATCH[1]}))"
    return
  fi
  if [[ "${raw}" =~ ^tok([0-9]+)-ckpt$ ]]; then
    printf 'tok%d-ckpt\n' "$((10#${BASH_REMATCH[1]}))"
    return
  fi
  if [[ "${raw}" =~ ^tok([0-9]+)-act-ckpt$ ]]; then
    printf 'tok%d-act-ckpt\n' "$((10#${BASH_REMATCH[1]}))"
    return
  fi
  die "invalid expert recompute policy '${1}'; expected none, splitXX, tokXX-ckpt, or tokXX-act-ckpt"
}

parse_expert_recompute_policies() {
  local value
  while read -r value; do
    normalize_expert_recompute_policy "${value}"
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
expert_recompute_policy_spec="${EXPERT_RECOMPUTE_POLICIES}"
batch_size="${BATCH_SIZE}"
jobs_per_gpu="${JOBS_PER_GPU}"
output_root="${OUTPUT_ROOT}"
run_name="${RUN_NAME}"
precision="${PRECISION}"
attention_impl="${ATTENTION_IMPL}"
mode="${MODE}"
moe_route_pattern="${MOE_ROUTE_PATTERN}"
hf_layer_index="${HF_LAYER_INDEX}"
hf_cache_dir="${HF_CACHE_DIR}"
hf_local_files_only="$(bool_value "${HF_LOCAL_FILES_ONLY}")"
profile_seed="${PROFILE_SEED}"
plot="$(bool_value "${PLOT}")"
plot_output_dir="${PLOT_OUTPUT_DIR}"
recompute_spec="${RECOMPUTE}"
overwrite="$(bool_value "${OVERWRITE}")"
continue_on_error="$(bool_value "${CONTINUE_ON_ERROR}")"
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
    --expert-recompute-policy=*) die "use --expert-recompute-policies; this wrapper keeps one option per setting" ;;
    --expert-recompute-policies=*) expert_recompute_policy_spec="${1#*=}"; shift ;;
    --expert-recompute-threshold=*|--expert-recompute-thresholds=*) die "legacy threshold flags were removed; use --expert-recompute-policies none,split128,tok128-ckpt,tok128-act-ckpt" ;;
    --batch-size=*) batch_size="${1#*=}"; shift ;;
    --jobs-per-gpu=*) jobs_per_gpu="${1#*=}"; shift ;;
    --python-bin=*) PYTHON_BIN="${1#*=}"; shift ;;
    --output-root=*) output_root="${1#*=}"; shift ;;
    --run-name=*) run_name="${1#*=}"; shift ;;
    --precision=*) precision="${1#*=}"; shift ;;
    --attention-impl=*) attention_impl="${1#*=}"; shift ;;
    --mode=*) mode="${1#*=}"; shift ;;
    --moe-route-pattern=*) moe_route_pattern="${1#*=}"; shift ;;
    --hf-layer-index=*) hf_layer_index="${1#*=}"; shift ;;
    --hf-cache-dir=*) hf_cache_dir="${1#*=}"; shift ;;
    --hf-local-files-only=*) die "use '--hf-local-files-only true' or '--hf-local-files-only false' instead of --hf-local-files-only=..." ;;
    --profile-seed=*) profile_seed="${1#*=}"; shift ;;
    --plot=*) die "use '--plot true' or '--plot false' instead of --plot=..." ;;
    --plot-output-dir=*) plot_output_dir="${1#*=}"; shift ;;
    --recompute=*) recompute_spec="${1#*=}"; shift ;;
    --overwrite=*) die "use '--overwrite true' or '--overwrite false' instead of --overwrite=..." ;;
    --continue-on-error=*) die "use '--continue-on-error true' or '--continue-on-error false' instead of --continue-on-error=..." ;;
    --gpus) collect_values "$1" vals "${@:2}"; gpu_spec="${vals[*]}"; set -- "${REMAINING[@]}" ;;
    --gpu-pool|--cuda-devices) die "use --gpus; this wrapper keeps one option per setting" ;;
    --workloads) collect_values "$1" vals "${@:2}"; workload_spec="${vals[*]}"; set -- "${REMAINING[@]}" ;;
    --backends) collect_values "$1" vals "${@:2}"; backend_spec="${vals[*]}"; set -- "${REMAINING[@]}" ;;
    --profilers) collect_values "$1" vals "${@:2}"; profiler_spec="${vals[*]}"; set -- "${REMAINING[@]}" ;;
    --seq-len) die "use --seq-lens; this wrapper keeps one option per setting" ;;
    --seq-lens) collect_values "$1" vals "${@:2}"; seq_spec="${vals[*]}"; set -- "${REMAINING[@]}" ;;
    --expert-recompute-policy) die "use --expert-recompute-policies; this wrapper keeps one option per setting" ;;
    --expert-recompute-policies) collect_values "$1" vals "${@:2}"; expert_recompute_policy_spec="${vals[*]}"; set -- "${REMAINING[@]}" ;;
    --expert-recompute-threshold|--expert-recompute-thresholds) die "legacy threshold flags were removed; use --expert-recompute-policies none,split128,tok128-ckpt,tok128-act-ckpt" ;;
    --batch-size) need_value "$1" "${2-}"; batch_size="$2"; shift 2 ;;
    --jobs-per-gpu) need_value "$1" "${2-}"; jobs_per_gpu="$2"; shift 2 ;;
    --python-bin) need_value "$1" "${2-}"; PYTHON_BIN="$2"; shift 2 ;;
    --output-root) need_value "$1" "${2-}"; output_root="$2"; shift 2 ;;
    --run-name) need_value "$1" "${2-}"; run_name="$2"; shift 2 ;;
    --precision) need_value "$1" "${2-}"; precision="$2"; shift 2 ;;
    --attention-impl) need_value "$1" "${2-}"; attention_impl="$2"; shift 2 ;;
    --mode) need_value "$1" "${2-}"; mode="$2"; shift 2 ;;
    --moe-route-pattern) need_value "$1" "${2-}"; moe_route_pattern="$2"; shift 2 ;;
    --hf-layer-index) need_value "$1" "${2-}"; hf_layer_index="$2"; shift 2 ;;
    --hf-cache-dir) need_value "$1" "${2-}"; hf_cache_dir="$2"; shift 2 ;;
    --hf-local-files-only) need_value "$1" "${2-}"; hf_local_files_only="$(bool_value "$2")"; shift 2 ;;
    --profile-seed) need_value "$1" "${2-}"; profile_seed="$2"; shift 2 ;;
    --plot) need_value "$1" "${2-}"; plot="$(bool_value "$2")"; shift 2 ;;
    --plot-output-dir) need_value "$1" "${2-}"; plot_output_dir="$2"; shift 2 ;;
    --recompute) need_value "$1" "${2-}"; recompute_spec="$2"; shift 2 ;;
    --overwrite) need_value "$1" "${2-}"; overwrite="$(bool_value "$2")"; shift 2 ;;
    --continue-on-error) need_value "$1" "${2-}"; continue_on_error="$(bool_value "$2")"; shift 2 ;;
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
case "${moe_route_pattern}" in balanced|learned) ;; *) die "--moe-route-pattern must be balanced or learned" ;; esac
[[ "${hf_layer_index}" =~ ^[0-9]+$ ]] || die "--hf-layer-index must be a non-negative integer"
[[ "${profile_seed}" =~ ^-?[0-9]+$ ]] || die "--profile-seed must be an integer"
[[ "${jobs_per_gpu}" =~ ^[0-9]+$ && "${jobs_per_gpu}" -gt 0 ]] || die "--jobs-per-gpu must be a positive integer"

mapfile -t gpus < <(parse_gpu_list "${gpu_spec}")
mapfile -t seq_lens < <(parse_seq_lens "${seq_spec}")
mapfile -t expert_recompute_policies < <(parse_expert_recompute_policies "${expert_recompute_policy_spec}" | dedupe)
mapfile -t workloads < <(tokens "${workload_spec}" | while read -r value; do expand_workload "${value}"; done | dedupe)
mapfile -t plot_workloads < <(printf '%s\n' "${workloads[@]}" | while read -r value; do workload_label "${value}"; done | dedupe)
mapfile -t backends < <(tokens "${backend_spec}" | while read -r value; do expand_alias backend "${value}"; done | dedupe)
mapfile -t profilers < <(tokens "${profiler_spec}" | while read -r value; do expand_alias profiler "${value}"; done | dedupe)
mapfile -t recompute_modes < <(recompute_values "${recompute_spec}")

((${#gpus[@]})) || die "GPU pool is empty"
((${#seq_lens[@]})) || die "sequence length list is empty"
((${#expert_recompute_policies[@]} > 0)) || die "expert recompute policy list is empty"
((${#workloads[@]} && ${#backends[@]} && ${#profilers[@]} && ${#recompute_modes[@]})) || die "workloads/backends/profilers/recompute expanded to an empty list"

preflight_python

driver_args=(
  --warmup-steps "${WARMUP_STEPS}"
  --measure-steps "${MEASURE_STEPS}"
  --profile-layers "${PROFILE_LAYERS}"
  --batch-size "${batch_size}"
  --mlp-intermediate-dim "${MLP_INTERMEDIATE_DIM}"
  --mlp-expansion "${MLP_EXPANSION}"
  --lora-rank "${LORA_RANK}"
  --lora-alpha "${LORA_ALPHA}"
  --vocab-rows "${VOCAB_ROWS}"
  --attention-impl "${attention_impl}"
  --moe-mode "${MOE_MODE}"
  --moe-route-pattern "${moe_route_pattern}"
  --hf-layer-index "${hf_layer_index}"
  --profile-seed "${profile_seed}"
  --dense-target-mode "${DENSE_TARGET_MODE}"
  --kt-method "${KT_METHOD}"
  --kt-cpu-threads "${KT_CPU_THREADS}"
  --kt-threadpool-count "${KT_THREADPOOL_COUNT}"
  --kt-max-cache-depth "${KT_MAX_CACHE_DEPTH}"
)
[[ -n "${hf_cache_dir}" ]] && driver_args+=(--hf-cache-dir "${hf_cache_dir}")
((hf_local_files_only)) && driver_args+=(--hf-local-files-only)
if ((overwrite)); then
  driver_args+=(--overwrite)
else
  driver_args+=(--skip-existing)
fi
((continue_on_error)) && driver_args+=(--continue-on-error)

base_output_root="$(abs_path "${output_root}")"
precision_label="$(safe_label "${precision}")"
precision_root="${base_output_root}/lora_e2e_${precision_label}"
mkdir -p "${precision_root}"
echo "Output precision root: ${precision_root}"

declare -a all_pids=() active_pids=() available_gpus=()
declare -A pid_gpu=() pid_recompute=() pid_expert_policy=() pid_workload=() pid_backend=() pid_profiler=() pid_seq_len=() pid_log=() pid_config_root=() pid_job_root=()
declare -A plot_roots=() plot_seq_lens=()
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
  local pid=""
  local status=0
  if wait -n -p pid "${active_pids[@]}"; then
    status=0
  else
    status=$?
  fi
  [[ -n "${pid}" ]] || pid="${active_pids[0]}"
  if ((status == 0)); then
    echo "Finished pid=${pid} gpu=${pid_gpu[$pid]} seq=${pid_seq_len[$pid]} recompute=${pid_recompute[$pid]} expert_policy=${pid_expert_policy[$pid]} workload=${pid_workload[$pid]} backend=${pid_backend[$pid]} profiler=${pid_profiler[$pid]}"
  else
    failures=$((failures + 1))
    echo "FAILED pid=${pid} status=${status} gpu=${pid_gpu[$pid]} seq=${pid_seq_len[$pid]} recompute=${pid_recompute[$pid]} expert_policy=${pid_expert_policy[$pid]} workload=${pid_workload[$pid]} backend=${pid_backend[$pid]} profiler=${pid_profiler[$pid]}; log=${pid_log[$pid]}" >&2
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${status}" "${pid}" "${pid_gpu[$pid]}" "${pid_seq_len[$pid]}" "${pid_recompute[$pid]}" "${pid_expert_policy[$pid]}" \
    "${pid_workload[$pid]}" "${pid_backend[$pid]}" "${pid_profiler[$pid]}" "${pid_job_root[$pid]}" "${pid_log[$pid]}" \
    >> "${pid_config_root[$pid]}/jobs.tsv"
  local -a remaining=()
  local active_pid
  for active_pid in "${active_pids[@]}"; do
    [[ "${active_pid}" != "${pid}" ]] && remaining+=("${active_pid}")
  done
  active_pids=("${remaining[@]}")
  available_gpus+=("${pid_gpu[$pid]}")
}

launch_job() {
  local recompute="$1"
  local expert_policy="$2"
  local workload="$3"
  local seq_len="$4"
  local backend="$5"
  local profiler="$6"
  local gpu="$7"
  local config_root
  local job_root
  config_root="$(config_root_path "${workload}" "${seq_len}")"
  job_root="$(job_root_path "${config_root}" "${backend}" "${profiler}" "${recompute}" "${expert_policy}")"
  mkdir -p "${job_root}"
  if [[ ! -e "${config_root}/jobs.tsv" ]]; then
    printf 'status\tpid\tgpu\tseq_len\trecompute\texpert_policy\tworkload\tbackend\tprofiler\tjob_dir\tlog\n' > "${config_root}/jobs.tsv"
  fi
  plot_roots["${config_root}"]="1"
  plot_seq_lens["${config_root}"]="${seq_len}"
  local log_file="${job_root}/job.log"
  local command_file="${job_root}/command.txt"
  local cmd=(
    "${PYTHON_BIN}" "${PY_DRIVER}"
    --workloads "${workload}"
    --backends "${backend}"
    --profilers "${profiler}"
    --cuda-devices "${gpu}"
    --output-root "${job_root}"
    --precision "${precision}"
    --mode "${mode}"
    --expert-recompute-policy "${expert_policy}"
    --skip-summary
    --flat-output
    "${driver_args[@]}"
    --seq-lens "${seq_len}"
    "${pass_args[@]}"
  )
  [[ "${recompute}" == "recomp" ]] && cmd+=(--activation-recompute)

  {
    printf 'cd %q\n' "${ROOT}"
    printf 'CUDA_VISIBLE_DEVICES=%q' "${gpu}"
    printf ' %q' "${cmd[@]}"
    printf '\n'
  } > "${command_file}"

  echo "Launching gpu=${gpu} seq=${seq_len} recompute=${recompute} expert_policy=${expert_policy} workload=${workload} backend=${backend} profiler=${profiler}"
  echo "  job=${job_root}"
  echo "  log=${log_file}"
  (cd "${ROOT}" && exec "${cmd[@]}") > "${log_file}" 2>&1 &

  local pid=$!
  all_pids+=("${pid}")
  active_pids+=("${pid}")
  pid_gpu[$pid]="${gpu}"
  pid_seq_len[$pid]="${seq_len}"
  pid_recompute[$pid]="${recompute}"
  pid_expert_policy[$pid]="${expert_policy}"
  pid_workload[$pid]="${workload}"
  pid_backend[$pid]="${backend}"
  pid_profiler[$pid]="${profiler}"
  pid_log[$pid]="${log_file}"
  pid_config_root[$pid]="${config_root}"
  pid_job_root[$pid]="${job_root}"
}

for gpu in "${gpus[@]}"; do
  for ((slot = 0; slot < jobs_per_gpu; slot++)); do
    available_gpus+=("${gpu}")
  done
done
for expert_policy in "${expert_recompute_policies[@]}"; do
  for recompute in "${recompute_modes[@]}"; do
    if [[ "${expert_policy}" != "none" && "${recompute}" == "recomp" ]]; then
      continue
    fi
    for workload in "${workloads[@]}"; do
      for seq_len in "${seq_lens[@]}"; do
        for backend in "${backends[@]}"; do
          for profiler in "${profilers[@]}"; do
            while ((${#available_gpus[@]} == 0)); do finish_one; done
            run_gpu="${available_gpus[0]}"
            available_gpus=("${available_gpus[@]:1}")
            launch_job "${recompute}" "${expert_policy}" "${workload}" "${seq_len}" "${backend}" "${profiler}" "${run_gpu}"
          done
        done
      done
    done
  done
done
while ((${#active_pids[@]})); do finish_one; done

if ((failures > 0)); then
  echo "${failures} profiling job(s) failed" >&2
  exit 1
fi

if ((dry_run)); then
  echo "Dry run completed; skipping plots."
  exit 0
fi

if ((plot)); then
  for config_root in "${!plot_roots[@]}"; do
    plot_root="${config_root}/plots"
    [[ -n "${plot_output_dir}" ]] && plot_root="$(abs_path "${plot_output_dir}")/$(basename "${config_root}")"
    plot_seq_len="${plot_seq_lens[$config_root]}"
    plot_cmd=(
      "${PYTHON_BIN}" "${PLOT_RECOMPUTE_SCRIPT}"
      --input-root "${config_root}"
      --output-dir "${plot_root}"
      --precision "${precision}"
      --clean-output
      --skip-combined
    )
    for backend in "${backends[@]}"; do plot_cmd+=(--backend "${backend}"); done
    for profiler in "${profilers[@]}"; do plot_cmd+=(--profiler "${profiler}"); done
    for recompute in "${recompute_modes[@]}"; do plot_cmd+=(--recompute "${recompute}"); done
    plot_cmd+=(--expert-recompute-policies "${expert_recompute_policies[@]}")
    plot_cmd+=(--batch-size "${batch_size}" --seq-lens "${plot_seq_len}")
    echo "Writing activation recompute sweep plots: ${plot_root}"
    (cd "${ROOT}" && "${plot_cmd[@]}")
  done

  combined_plot_root="${precision_root}/combined"
  [[ -n "${plot_output_dir}" ]] && combined_plot_root="$(abs_path "${plot_output_dir}")/combined"
  combined_plot_cmd=(
    "${PYTHON_BIN}" "${PLOT_RECOMPUTE_SCRIPT}"
    --input-root "${precision_root}"
    --output-dir "${combined_plot_root}"
    --combined-output-dir "${combined_plot_root}"
    --precision "${precision}"
    --clean-output
    --combined-only
  )
  for backend in "${backends[@]}"; do combined_plot_cmd+=(--backend "${backend}"); done
  for profiler in "${profilers[@]}"; do combined_plot_cmd+=(--profiler "${profiler}"); done
  for recompute in "${recompute_modes[@]}"; do combined_plot_cmd+=(--recompute "${recompute}"); done
  combined_plot_cmd+=(--expert-recompute-policies "${expert_recompute_policies[@]}")
  combined_plot_cmd+=(--batch-size "${batch_size}" --seq-lens "${seq_lens[@]}")
  if [[ -z "${run_name}" ]]; then
    for workload in "${plot_workloads[@]}"; do combined_plot_cmd+=(--workload "${workload}"); done
  fi
  echo "Writing combined activation recompute sweep plots: ${combined_plot_root}"
  (cd "${ROOT}" && "${combined_plot_cmd[@]}")
fi

echo "All profiling jobs completed."
