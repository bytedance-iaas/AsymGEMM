#!/usr/bin/env bash
# Dial ladder (scheduler_v2.md §3): memory→latency rungs @ q3-30b-a3b 120k b8.
# Steady protocol: w1+m4; per-run timeout + one retry (rare router-bincount hang,
# see memory_mode.md 2026-07-13).
set -uo pipefail
cd "$(dirname "$0")/../.."
export SFT_ROOT=${SFT_ROOT:-/workspace/AsymGEMM-SFT}
export PROFILERS=source MAX_STEPS=4 WARMUP_STEPS=1
TIMEOUT_S=${TIMEOUT_S:-5400}

R101='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker101-ceil0000-ohbm0|ligerloss1 ; 120000|8|1 ; none|false|false|false|false|false'
R000='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1 ; 120000|8|1 ; none|false|false|false|false|false'
R000_OHBM='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm8|ligerloss1 ; 120000|8|1 ; none|false|false|false|false|false'
RSO='q3-30b-a3b|1 ; superoffload_mem|unsloth-off-ohbm0|ligerloss1 ; 120000|8|1 ; none|false|false|false|false|false'

rung_ok() {
  # the driver exits 0 even on failed jobs (CONTINUE_ON_ERROR); trust jobs.tsv only
  local name="$1"
  local tsv
  tsv=$(ls profiling/asym_long_sft_smoke__lora__lf__bf16/"${name}"__*/jobs.tsv 2>/dev/null | head -1)
  [[ -n "${tsv}" ]] && awk -F'\t' 'NR>1 && $1=="ok"{found=1} END{exit !found}' "${tsv}"
}

run_rung() {
  local name="$1"; shift
  local runs="$1"; shift
  # remaining args: KEY=VAL env pins
  local attempt
  for attempt in 1 2; do
    echo "=== RUNG ${name} attempt ${attempt} $(date -u +%H:%M:%S)"
    env "$@" RUN_NAME="${name}" RUNS="${runs}" OVERWRITE=true \
        timeout --kill-after=60 "${TIMEOUT_S}" bash scripts/lf/profile_lora_lf_test_both.sh
    if rung_ok "${name}"; then
      echo "=== RUNG ${name} OK"
      return 0
    fi
    echo "=== RUNG ${name} attempt ${attempt} FAILED/TIMED OUT; cleaning up"
    pkill -9 -f run_lf_profiled_train 2>/dev/null
    sleep 20
  done
  echo "=== RUNG ${name} GAVE UP"
  return 1
}

# Explicit env on every rung — never rely on defaults for dial measurements.
BASE=(ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1)
if [[ "${SKIP_L0:-0}" != "1" ]]; then
run_rung dial_l0_memmode  "$R101" "${BASE[@]}"
fi
run_rung dial_l1_staged   "$R101" "${BASE[@]}" ASYM_GEMM_DISPATCH=staged
run_rung dial_l2_ker000   "$R000" "${BASE[@]}" ASYM_GEMM_DISPATCH=staged
run_rung dial_l3_keepacts "$R000" "${BASE[@]}" ASYM_GEMM_DISPATCH=staged ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1
run_rung dial_l4_nochunk  "$R000" ASYMM_FG_ELEMENTWISE_CHUNK_MB=0 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYM_GEMM_DISPATCH=staged ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1
run_rung dial_l5_ohbm8    "$R000_OHBM" "${BASE[@]}" ASYM_GEMM_DISPATCH=staged ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1
run_rung dial_ref_so      "$RSO"
echo DIAL_LADDER_DONE
