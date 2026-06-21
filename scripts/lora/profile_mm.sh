#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# User Parameters
# =============================================================================
# AsymGEMM dir = the dir you run in. Override with ROOT=...
ROOT=${ROOT:-$(pwd)}
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"

MODE="${MODE:-timing}"
DEVICE="${DEVICE:-cuda:3}"
PRECISION="${PRECISION:-bf16}"
SHAPES="${SHAPES:-8192|8192|8192 4096|4096|4096}"

SEED="${SEED:-1234}"
SCALE="${SCALE:-0.1}"
ATOL="${ATOL:-0.5}"
RTOL="${RTOL:-0.05}"
SLOW_THRESHOLD="${SLOW_THRESHOLD:-1.25}"
COMPILED_DIMS="${COMPILED_DIMS:-mnk}"
JSON_OUTPUT="${JSON_OUTPUT:-}"
BF16_BLOCK_M="${BF16_BLOCK_M:-${DG_BF16_BLOCK_M:-128}}"
BF16_BLOCK_N="${BF16_BLOCK_N:-${DG_BF16_BLOCK_N:-128}}"
BF16_BLOCK_K="${BF16_BLOCK_K:-${DG_BF16_BLOCK_K:-128}}"
BF16_TRANSPOSE_BLOCK_M="${BF16_TRANSPOSE_BLOCK_M:-${DG_BF16_TRANSPOSE_BLOCK_M:-128}}"
BF16_TRANSPOSE_BLOCK_N="${BF16_TRANSPOSE_BLOCK_N:-${DG_BF16_TRANSPOSE_BLOCK_N:-128}}"
BF16_TRANSPOSE_BLOCK_K="${BF16_TRANSPOSE_BLOCK_K:-${DG_BF16_TRANSPOSE_BLOCK_K:-128}}"
NCU_BIN="${NCU_BIN:-ncu}"
NCU_OUTPUT_DIR="${NCU_OUTPUT_DIR:-profiling/mm_ncu}"
NCU_KERNEL_REGEX="${NCU_KERNEL_REGEX:-regex:.*sm(90|100).*(bf16|fp8|fp4).*asym_gemm.*impl.*}"
NCU_SET="${NCU_SET:-full}"
NCU_REPLAY_MODE="${NCU_REPLAY_MODE:-kernel}"
NCU_CLOCK_CONTROL="${NCU_CLOCK_CONTROL:-none}"
NCU_SECTIONS="${NCU_SECTIONS:-MemoryWorkloadAnalysis_Chart SpeedOfLight MemoryWorkloadAnalysis MemoryWorkloadAnalysis_Tables LaunchStats Occupancy SchedulerStats}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-0}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-}"

# =============================================================================
# Derived Parameters
# =============================================================================
PROFILE_SCRIPT="${ROOT}/scripts/lora/profile_mm.py"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# =============================================================================
# Main Logic
# =============================================================================
usage() {
  cat <<USAGE
Usage:
  scripts/lora/profile_mm.sh [options] [extra profile_mm.py args]

Compares:
  torch_nontranspose: X[M,K] @ W[N,K].T
  asym_nontranspose:  X[M,K] @ W[N,K].T
  torch_transpose:    G[M,N] @ W[N,K]
  asym_transpose:     G[M,N] @ W[N,K] via AsymGEMM transpose_b=True
  asym_stored_transpose:
                     G[M,N] @ WT[K,N].T with a second CPU-pinned WT=W.T.contiguous()

Shape format:
  M|K|N
  nontranspose MM always means: X[M,K] @ W[N,K].T -> Y[M,N]
  transposed MM always means:  G[M,N] @ W[N,K]   -> dX[M,K]

Default shapes:
  ${SHAPES}
  2048|2048|768 and 2048|4096|1536 are gate/up-style shapes.
  2048|768|2048 and 2048|1536|4096 are down-style shapes.

Options:
  --mode timing|ncu
                      timing: regular timing mode.
                      ncu: Nsight Compute mode; auto uses warmup=0,iters=1,repeats=1.
  --device DEV
  --precision bf16|fp8
  --shapes LIST        whitespace/comma separated list like '2048|2048|768 2048|4096|1536'
  --shape M|K|N        append one shape; may be repeated
  --seed N
  --scale FLOAT
  --atol FLOAT
  --rtol FLOAT
  --slow-threshold FLOAT
  --compiled-dims STR
  --json-output PATH
  --bf16-block-m N
  --bf16-block-n N
  --bf16-block-k N
  --bf16-transpose-block-m N
  --bf16-transpose-block-n N
  --bf16-transpose-block-k N
  --ncu-bin PATH
  --ncu-output-dir DIR
  --ncu-kernel-regex REGEX
  --ncu-set NAME      NCU section set, default: ${NCU_SET}; set NCU_SET="" to disable.
  --ncu-replay-mode MODE
                      NCU replay mode, default: ${NCU_REPLAY_MODE}; set NCU_REPLAY_MODE="" to disable.
  --ncu-clock-control MODE
                      NCU clock control, default: ${NCU_CLOCK_CONTROL}; set NCU_CLOCK_CONTROL="" to disable.
  --ncu-section NAME   repeatable; default: ${NCU_SECTIONS}
  --ncu-launch-skip N
  --ncu-launch-count N
  --extra-ncu-arg ARG  repeatable, appended before the Python command
  -h, --help
USAGE
}

need_value() {
  local opt="$1"
  local value="${2-}"
  [[ -n "${value}" && "${value}" != --* ]] || {
    echo "error: ${opt} requires a value" >&2
    exit 2
  }
}

need_any_value() {
  local opt="$1"
  local value="${2-}"
  [[ -n "${value}" ]] || {
    echo "error: ${opt} requires a value" >&2
    exit 2
  }
}

EXTRA_ARGS=()
SHAPE_ARGS=()
NCU_SECTION_ARGS=()
NCU_EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) need_value "$1" "${2-}"; MODE="$2"; shift 2 ;;
    --device) need_value "$1" "${2-}"; DEVICE="$2"; shift 2 ;;
    --precision) need_value "$1" "${2-}"; PRECISION="$2"; shift 2 ;;
    --shapes) need_value "$1" "${2-}"; SHAPES="$2"; shift 2 ;;
    --shape) need_value "$1" "${2-}"; SHAPE_ARGS+=("$2"); shift 2 ;;
    --m|--k|--n)
      echo "error: ${1} is managed by profile_mm.sh; use --shape M|K|N" >&2
      exit 2
      ;;
    --warmup|--iters|--repeats)
      echo "error: ${1} is managed by --mode" >&2
      exit 2
      ;;
    --seed) need_value "$1" "${2-}"; SEED="$2"; shift 2 ;;
    --scale) need_value "$1" "${2-}"; SCALE="$2"; shift 2 ;;
    --atol) need_value "$1" "${2-}"; ATOL="$2"; shift 2 ;;
    --rtol) need_value "$1" "${2-}"; RTOL="$2"; shift 2 ;;
    --slow-threshold) need_value "$1" "${2-}"; SLOW_THRESHOLD="$2"; shift 2 ;;
    --compiled-dims) need_value "$1" "${2-}"; COMPILED_DIMS="$2"; shift 2 ;;
    --json-output) need_value "$1" "${2-}"; JSON_OUTPUT="$2"; shift 2 ;;
    --bf16-block-m) need_value "$1" "${2-}"; BF16_BLOCK_M="$2"; shift 2 ;;
    --bf16-block-n) need_value "$1" "${2-}"; BF16_BLOCK_N="$2"; shift 2 ;;
    --bf16-block-k) need_value "$1" "${2-}"; BF16_BLOCK_K="$2"; shift 2 ;;
    --bf16-transpose-block-m) need_value "$1" "${2-}"; BF16_TRANSPOSE_BLOCK_M="$2"; shift 2 ;;
    --bf16-transpose-block-n) need_value "$1" "${2-}"; BF16_TRANSPOSE_BLOCK_N="$2"; shift 2 ;;
    --bf16-transpose-block-k) need_value "$1" "${2-}"; BF16_TRANSPOSE_BLOCK_K="$2"; shift 2 ;;
    --ncu-bin) need_value "$1" "${2-}"; NCU_BIN="$2"; shift 2 ;;
    --ncu-output-dir) need_value "$1" "${2-}"; NCU_OUTPUT_DIR="$2"; shift 2 ;;
    --ncu-kernel-regex) need_value "$1" "${2-}"; NCU_KERNEL_REGEX="$2"; shift 2 ;;
    --ncu-set) need_value "$1" "${2-}"; NCU_SET="$2"; shift 2 ;;
    --ncu-replay-mode) need_value "$1" "${2-}"; NCU_REPLAY_MODE="$2"; shift 2 ;;
    --ncu-clock-control) need_value "$1" "${2-}"; NCU_CLOCK_CONTROL="$2"; shift 2 ;;
    --ncu-section) need_value "$1" "${2-}"; NCU_SECTION_ARGS+=("$2"); shift 2 ;;
    --ncu-launch-skip) need_value "$1" "${2-}"; NCU_LAUNCH_SKIP="$2"; shift 2 ;;
    --ncu-launch-count) need_value "$1" "${2-}"; NCU_LAUNCH_COUNT="$2"; shift 2 ;;
    --extra-ncu-arg) need_any_value "$1" "${2-}"; NCU_EXTRA_ARGS+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

for extra_arg in "${EXTRA_ARGS[@]}"; do
  case "${extra_arg}" in
    --m|--m=*|--k|--k=*|--n|--n=*)
      echo "error: ${extra_arg%%=*} is managed by profile_mm.sh; use --shape M|K|N" >&2
      exit 2
      ;;
    --warmup|--warmup=*|--iters|--iters=*|--repeats|--repeats=*)
      echo "error: ${extra_arg%%=*} is managed by --mode" >&2
      exit 2
      ;;
  esac
done

case "${MODE}" in
  timing)
    MODE="timing"
    WARMUP=10
    ITERS=100
    REPEATS=1
    ;;
  ncu)
    MODE="ncu"
    WARMUP=0
    ITERS=1
    REPEATS=1
    ;;
  *)
    echo "error: --mode must be one of: timing, ncu" >&2
    exit 2
    ;;
esac

if [[ "${MODE}" == "ncu" ]]; then
  command -v "${NCU_BIN}" >/dev/null 2>&1 || {
    echo "error: ncu mode requested but '${NCU_BIN}' was not found" >&2
    exit 2
  }
fi

NCU_SECTIONS_FINAL=()
if [[ "${#NCU_SECTION_ARGS[@]}" -gt 0 ]]; then
  NCU_SECTIONS_FINAL=("${NCU_SECTION_ARGS[@]}")
else
  read -r -a NCU_SECTIONS_FINAL <<< "${NCU_SECTIONS}"
fi

export DG_BF16_BLOCK_M="${BF16_BLOCK_M}"
export DG_BF16_BLOCK_N="${BF16_BLOCK_N}"
export DG_BF16_BLOCK_K="${BF16_BLOCK_K}"
export DG_BF16_TRANSPOSE_BLOCK_M="${BF16_TRANSPOSE_BLOCK_M}"
export DG_BF16_TRANSPOSE_BLOCK_N="${BF16_TRANSPOSE_BLOCK_N}"
export DG_BF16_TRANSPOSE_BLOCK_K="${BF16_TRANSPOSE_BLOCK_K}"

shape_entries=()
if [[ "${#SHAPE_ARGS[@]}" -gt 0 ]]; then
  shape_entries=("${SHAPE_ARGS[@]}")
else
  # Accept whitespace and comma separated lists.
  read -r -a shape_entries <<< "${SHAPES//,/ }"
fi

validate_shape() {
  local shape="$1"
  [[ "${shape}" =~ ^[0-9]+\|[0-9]+\|[0-9]+$ ]] || {
    echo "error: invalid shape '${shape}', expected M|K|N" >&2
    exit 2
  }
}

json_output_for_shape() {
  local output="$1"
  local m="$2"
  local k="$3"
  local n="$4"
  local count="$5"
  if [[ -z "${output}" ]]; then
    return 0
  fi
  if [[ "${count}" -le 1 ]]; then
    printf '%s\n' "${output}"
  elif [[ "${output}" == *.* ]]; then
    printf '%s.m%s_k%s_n%s.%s\n' "${output%.*}" "${m}" "${k}" "${n}" "${output##*.}"
  else
    printf '%s.m%s_k%s_n%s.json\n' "${output}" "${m}" "${k}" "${n}"
  fi
}

print_shape_header() {
  local index="$1"
  local count="$2"
  local m="$3"
  local k="$4"
  local n="$5"

  printf '\n'
  printf '%*s\n' 80 '' | tr ' ' '='
  printf 'Profile shape %s/%s: M=%s K=%s N=%s\n' "${index}" "${count}" "${m}" "${k}" "${n}"
  printf '  mode: %s\n' "${MODE}"
  printf '  benchmark config: warmup=%s iters=%s repeats=%s\n' "${WARMUP}" "${ITERS}" "${REPEATS}"

  if (( k > n )); then
    printf '  MoE projection: gate/up-style, H=%s I=%s\n' "${k}" "${n}"
    printf '  forward MM:     X[M,H]  @ W_gate/up[I,H].T -> Y[M,I]\n'
    printf '  backward dX:    dY[M,I] @ W_gate/up[I,H]   -> dX[M,H]\n'
  elif (( k < n )); then
    printf '  MoE projection: down-style, H=%s I=%s\n' "${n}" "${k}"
    printf '  forward MM:     X[M,I]  @ W_down[H,I].T -> Y[M,H]\n'
    printf '  backward dX:    dY[M,H] @ W_down[H,I]   -> dX[M,I]\n'
  else
    printf '  MoE projection: square/unknown, K=N=%s\n' "${k}"
  fi

  printf '  generic MM:     X[%s,%s] @ W[%s,%s].T -> [%s,%s]\n' "${m}" "${k}" "${n}" "${k}" "${m}" "${n}"
  printf '  generic dX MM:  G[%s,%s] @ W[%s,%s]   -> [%s,%s]\n' "${m}" "${n}" "${n}" "${k}" "${m}" "${k}"
  printf '%*s\n' 80 '' | tr ' ' '='
}

ncu_report_prefix_for_shape() {
  local output_dir="$1"
  local precision="$2"
  local m="$3"
  local k="$4"
  local n="$5"
  mkdir -p "${output_dir}"
  printf '%s/profile_mm_%s_m%s_k%s_n%s\n' "${output_dir%/}" "${precision}" "${m}" "${k}" "${n}"
}

run_count=0
total_count="${#shape_entries[@]}"
[[ "${total_count}" -gt 0 ]] || {
  echo "error: no shapes provided" >&2
  exit 2
}

for shape in "${shape_entries[@]}"; do
  [[ -n "${shape}" ]] || continue
  validate_shape "${shape}"
  IFS='|' read -r shape_m shape_k shape_n <<< "${shape}"
  run_count=$((run_count + 1))

  print_shape_header "${run_count}" "${total_count}" "${shape_m}" "${shape_k}" "${shape_n}"

  cmd=(
    "${PYTHON_BIN}" "${PROFILE_SCRIPT}"
    --device "${DEVICE}"
    --precision "${PRECISION}"
    --m "${shape_m}"
    --k "${shape_k}"
    --n "${shape_n}"
    --warmup "${WARMUP}"
    --iters "${ITERS}"
    --repeats "${REPEATS}"
    --seed "${SEED}"
    --scale "${SCALE}"
    --atol "${ATOL}"
    --rtol "${RTOL}"
    --slow-threshold "${SLOW_THRESHOLD}"
    --compiled-dims "${COMPILED_DIMS}"
  )

  shape_json_output="$(json_output_for_shape "${JSON_OUTPUT}" "${shape_m}" "${shape_k}" "${shape_n}" "${total_count}")"
  if [[ -n "${shape_json_output}" ]]; then
    cmd+=(--json-output "${shape_json_output}")
  fi

  cmd+=("${EXTRA_ARGS[@]}")
  if [[ "${MODE}" == "ncu" ]]; then
    report_prefix="$(ncu_report_prefix_for_shape "${NCU_OUTPUT_DIR}" "${PRECISION}" "${shape_m}" "${shape_k}" "${shape_n}")"
    ncu_cmd=(
      "${NCU_BIN}"
      --target-processes all
      --kernel-name-base demangled
      -k "${NCU_KERNEL_REGEX}"
      --page raw
      --csv
      --log-file "${report_prefix}.csv"
      --export "${report_prefix}"
      --force-overwrite
    )
    if [[ -n "${NCU_SET}" ]]; then
      ncu_cmd+=(--set "${NCU_SET}")
    fi
    if [[ -n "${NCU_REPLAY_MODE}" ]]; then
      ncu_cmd+=(--replay-mode "${NCU_REPLAY_MODE}")
    fi
    if [[ -n "${NCU_CLOCK_CONTROL}" ]]; then
      ncu_cmd+=(--clock-control "${NCU_CLOCK_CONTROL}")
    fi
    if [[ -n "${NCU_LAUNCH_SKIP}" ]]; then
      ncu_cmd+=(--launch-skip "${NCU_LAUNCH_SKIP}")
    fi
    if [[ -n "${NCU_LAUNCH_COUNT}" ]]; then
      ncu_cmd+=(--launch-count "${NCU_LAUNCH_COUNT}")
    fi
    for section in "${NCU_SECTIONS_FINAL[@]}"; do
      ncu_cmd+=(--section "${section}")
    done
    ncu_cmd+=("${NCU_EXTRA_ARGS[@]}")
    ncu_cmd+=("${cmd[@]}")

    printf 'NCU mode auto config: warmup=%s iters=%s repeats=%s\n' "${WARMUP}" "${ITERS}" "${REPEATS}"
    printf 'NCU report: %s.ncu-rep\n' "${report_prefix}"
    printf 'NCU raw CSV: %s.csv\n' "${report_prefix}"
    "${ncu_cmd[@]}"
  else
    "${cmd[@]}"
  fi
done

exit 0

# BF16 SM100 block-size catalog, May 29 2026
#
# Best tested forward/nontranspose env:
#   DG_BF16_BLOCK_M=128
#   DG_BF16_BLOCK_N=32
#   DG_BF16_BLOCK_K=256
#
# Best tested true-transpose dX env:
#   DG_BF16_TRANSPOSE_BLOCK_M=128
#   DG_BF16_TRANSPOSE_BLOCK_N=64
#   DG_BF16_TRANSPOSE_BLOCK_K=256
#
# Confirmed with repeats=10,warmup=20,iters=50.  All rows below passed
# correctness against Torch.
#
# H=2048, I=768:
#   gate/up forward, W[I,H]=[768,2048]:
#     K=2048,N=768 nontranspose -> ~0.296 ms, BM=128,BN=32,BK=256
#   gate/up backward dX:
#     K=2048,N=768 true transpose -> ~0.159 ms, BM=128,BN=64,BK=256
#   down forward, W[H,I]=[2048,768]:
#     K=768,N=2048 nontranspose -> ~0.127 ms, BM=128,BN=32,BK=256
#   down backward dX:
#     K=768,N=2048 true transpose -> ~0.390 ms, BM=128,BN=64,BK=256
#
# H=4096, I=1536:
#   gate/up forward, W[I,H]=[1536,4096]:
#     K=4096,N=1536 nontranspose -> ~0.727 ms, BM=128,BN=32,BK=256
#   gate/up backward dX:
#     K=4096,N=1536 true transpose -> ~0.321 ms, BM=128,BN=64,BK=256
#   down forward, W[H,I]=[4096,1536]:
#     K=1536,N=4096 nontranspose -> ~0.310 ms, BM=128,BN=32,BK=256
#   down backward dX:
#     K=1536,N=4096 true transpose -> ~0.801 ms, BM=128,BN=64,BK=256
#
# Why the nontranspose config wins:
#   BN=32 exposes more output-column CTAs than BN=64:
#     I=768:  12 -> 24 tiles
#     I=1536: 24 -> 48 tiles
#     H=2048: 32 -> 64 tiles
#     H=4096: 64 -> 128 tiles
#   BM=128 keeps enough work per CTA and satisfies the fixed 128-row UMMA layout.
#   BK=256 keeps shared memory legal.  BK=512 with BM=128,BN=32 is illegal, and
#   BM=64,BN=32,BK=512 fails the UMMA shared-memory check.
