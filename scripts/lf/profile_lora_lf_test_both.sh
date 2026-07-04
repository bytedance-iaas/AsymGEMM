#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# User Parameters
# =============================================================================

# Paths and tools
# Repo root = AsymGEMM-SFT; override with SFT_ROOT=
SFT_ROOT=${SFT_ROOT:-$(cd ../.. && pwd)}
ROOT=${ROOT:-${SFT_ROOT}/third_party/AsymGEMM}
_LF_DIR_ENV_SET=false
_ENV_DIR_ENV_SET=false
_FLASH_ATTN_ENV_SET=false
[[ -n "${LF_DIR+x}" ]] && _LF_DIR_ENV_SET=true
[[ -n "${ENV_DIR+x}" ]] && _ENV_DIR_ENV_SET=true
[[ -n "${FLASH_ATTN+x}" ]] && _FLASH_ATTN_ENV_SET=true
LF_DIR=${LF_DIR:-${SFT_ROOT}/third_party/LlamaFactory}
# FA4 support is ported into the canonical LlamaFactory (2026-07-03); the qwen3.5
# auto-switch keeps the canonical LF (loss-only liger + save-on-cpu machinery) and
# only swaps the venv (flash-attn-4 package) + FLASH_ATTN=fa4.
LF_FA4_DIR=${LF_FA4_DIR:-${SFT_ROOT}/third_party/LlamaFactory}
LF_DS_CONFIG_DIR=${LF_DS_CONFIG_DIR:-${SFT_ROOT}/third_party/LlamaFactory/examples/deepspeed}
KT_KERNEL_DIR=${KT_KERNEL_DIR:-${SFT_ROOT}/third_party/ktransformers/kt-kernel}
DEEPSPEED_DIR=${DEEPSPEED_DIR:-${SFT_ROOT}/third_party/deepspeed}
CONDA_EXE=${CONDA_EXE:-conda}
NSYS_BIN=${NSYS_BIN:-nsys}
# DIST_LAUNCHER=${DIST_LAUNCHER:-accelerate}
DIST_LAUNCHER=${DIST_LAUNCHER:-torchrun}
# DIST_LAUNCHER=${DIST_LAUNCHER:-deepspeed}
RUN_POST=${RUN_POST:-false}

# Run list
GPU_POOL=${GPU_POOL:-3}

# RUNS: model ; backend|recompute|liger ; seq|batch|grad_accum ; policy|expact|attnact|layeract|layergc|sdparecomp
# Models use the M shorthand. To override the default list from the environment, pass:
#   RUNS='q3-30b-a3b|1 ; superoffload_mem|unsloth|ligerloss1 ; 4092|8|1 ; none|false|false|false|false|false || q3-30b-a3b|1 ; asym_cpuadamwds|norecompute|ligerloss1 ; 4092|8|1 ; none|true|true|false|true|true'
#   backend  : asym_cpuadamwds | zero3_offload | zero3_offload_mem | zero3_offload_opnvme | zero3_offload_panvme | zero3_offload_mem_opnvme | zero3_offload_mem_panvme | superoffload | superoffload_mem | superoffload_mem_opnvme | superoffload_mem_panvme
#   recompute: recomp | norecomp | unsloth | unsloth-off | recomp-off-full-fg | recomp-off-full-fg-kerXYZ
#              recomp-off-full-fg-kerXYZ pins the effective routed-kernel bits in the run spec.
#              Dense/non-routed runs use ker000; Qwen3-30B-A3B MoE auto-default is ker101 unless route env vars override it.
#              Unsloth-GC modes accept a trailing -ohbm<N> suffix (e.g. unsloth-off-ohbm4,
#              recomp-off-full-fg-ker101-ohbm6): keep every Nth outer checkpoint root in HBM
#              (UNSLOTH_GC_OUTER_HBM_EVERY_N). Omitted = ohbm0 = all roots offloaded to CPU.
#   liger    : ligerloss0 | ligerloss1
#   policy   : none|false|false|false|false|false (off)  |  none|true|true|false|true|true (offload+gc)
declare -A M=(
  # MoE                                          (key = family-version + total size + active size)
  [q3-30b-a3b]="Qwen/Qwen3-30B-A3B"                  # layers: 48
  [q3-235b-a22b]="Qwen/Qwen3-235B-A22B"              # layers: 94
  [q3.5-35b-a3b]="Qwen/Qwen3.5-35B-A3B"              # layers: 40
  [q3.5-122b-a10b]="Qwen/Qwen3.5-122B-A10B"          # layers: 48
  [llama4-scout]="meta-llama/Llama-4-Scout-17B-16E"  # layers: 48
  # dense
  [q3-32b]="Qwen/Qwen3-32B"                          # layers: 64
  [q3.5-27b]="Qwen/Qwen3.5-27B"                      # layers: 64
  [q2.5-32b]="Qwen/Qwen2.5-32B-Instruct"             # layers: 64
  [q2.5-72b]="Qwen/Qwen2.5-72B-Instruct"             # layers: 80
  [llama3.3-70b]="meta-llama/Llama-3.3-70B-Instruct" # layers: 80
)
declare -A _M_REV=()
for _k in "${!M[@]}"; do _M_REV["${M[$_k]%%|*}"]="${_k}"; done
_RUNS_ENV_SET=false
_RUNS_ENV_SPEC=""
if [[ -n "${RUNS+x}" ]]; then
  _RUNS_ENV_SET=true
  _RUNS_ENV_SPEC="${RUNS}"
  unset RUNS
fi
_RUNS_LOG="${RUNS_LOG:-${ROOT}/scripts/lf/runs.log}"
if [[ "${_RUNS_ENV_SET}" == "true" ]]; then
  RUNS=()
  _runs_env_lines="${_RUNS_ENV_SPEC//||/$'\n'}"
  while IFS= read -r _run; do
    _run="${_run#"${_run%%[![:space:]]*}"}"
    _run="${_run%"${_run##*[![:space:]]}"}"
    [[ -n "${_run}" ]] && RUNS+=("${_run}")
  done <<< "${_runs_env_lines}"
else
  RUNS=(
    "q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 60000|8|1 ; none|false|false|false|false|false" # C-OOM 53k
    # "q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 55000|8|1 ; none|false|false|false|false|false" # C-OOM 53k
    # "q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 52000|8|1 ; none|false|false|false|false|false" # C-OOM 53k

    # "llama3.3-70b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 32000|8|1 ; none|false|false|false|false|false" # C-OOM 33k
    # "llama3.3-70b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 34000|8|1 ; none|false|false|false|false|false" # C-OOM 33k

    # "llama4-scout|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 14500|8|1 ; none|false|false|false|false|false" # G-OOM 15k
    # "q2.5-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 52000|8|1 ; none|false|false|false|false|false" # C-OOM 53k
    # "q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 131000|8|1 ; none|false|false|false|false|false" # C-OOM 132k
    # "q2.5-72b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 32000|8|1 ; none|false|false|false|false|false" # C-OOM 33k
  )
  # RUNS=(
  #   # "q3-30b-a3b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 131000|8|1 ; none|false|false|false|false|false" # C-OOM 132k
  #   # "q3-32b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 52000|8|1 ; none|false|false|false|false|false" # C-OOM 53k
  #   # "llama3.3-70b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 32000|8|1 ; none|false|false|false|false|false" # C-OOM 33k
  #   # "llama4-scout|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 14500|8|1 ; none|false|false|false|false|false" # G-OOM 15k
  #   # "q2.5-32b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 52000|8|1 ; none|false|false|false|false|false" # C-OOM 53k
  #   # "q2.5-72b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 32000|8|1 ; none|false|false|false|false|false" # C-OOM 33k

  #   # "q3-30b-a3b|1 ; superoffload_mem|unsloth|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false" # G-OOM 81k
  #   # "q3-32b|1 ; superoffload_mem|unsloth|ligerloss1 ; 49000|8|1 ; none|false|false|false|false|false" # G-OOM 50k
  #   # "llama3.3-70b|1 ; superoffload_mem|unsloth|ligerloss1 ; 45000|8|1 ; none|false|false|false|false|false" # G-OOM 46k
  #   # "llama4-scout|1 ; superoffload_mem|unsloth|ligerloss1 ; 9500|8|1 ; none|false|false|false|false|false" # G-OOM 10k
  #   # "q2.5-32b|1 ; superoffload_mem|unsloth|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false" # G-OOM 51k
  #   # "q2.5-72b|1 ; superoffload_mem|unsloth|ligerloss1 ; 40000|8|1 ; none|false|false|false|false|false" # G-OOM 41k
  #   # "q3.5-35b-a3b|1 ; superoffload_mem|unsloth|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false" # G-OOM 81k

  #   # "q3-30b-a3b|1 ; superoffload_mem|recomp|ligerloss1 ; 45000|8|1 ; none|false|false|false|false|false" # G-OOM 46k
  #   # "q3-32b|1 ; superoffload_mem|recomp|ligerloss1 ; 20000|8|1 ; none|false|false|false|false|false" # G-OOM 21k
  #   # "llama3.3-70b|1 ; superoffload_mem|recomp|ligerloss1 ; 12000|8|1 ; none|false|false|false|false|false" # G-OOM 13k
  #   # "llama4-scout|1 ; superoffload_mem|recomp|ligerloss1 ; 8000|8|1 ; none|false|false|false|false|false" # G-OOM 9k
  #   # "q2.5-32b|1 ; superoffload_mem|recomp|ligerloss1 ; 21000|8|1 ; none|false|false|false|false|false" # G-OOM 22k
  #   # "q2.5-72b|1 ; superoffload_mem|recomp|ligerloss1 ; 12000|8|1 ; none|false|false|false|false|false" # G-OOM 13k
  # )
  # RUNS=(
  #   # "q3-30b-a3b|1 ; superoffload_mem|unsloth|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false" # G-OOM 81k
  #   # "q3-30b-a3b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false" # G-OOM 81k
  #   # "q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false" # G-OOM 81k

  #   # "q3-32b|1 ; superoffload_mem|unsloth|ligerloss1 ; 49000|8|1 ; none|false|false|false|false|false" # G-OOM 51k
  #   # "q3-32b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 49000|8|1 ; none|false|false|false|false|false" # G-OOM 81k
  #   # "q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 49000|8|1 ; none|false|false|false|false|false" # G-OOM 81k

  #   # "llama3.3-70b|1 ; superoffload_mem|unsloth|ligerloss1 ; 32000|8|1 ; none|false|false|false|false|false" # G-OOM 46k
  #   # "llama3.3-70b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 32000|8|1 ; none|false|false|false|false|false" # G-OOM 46k
  #   # "llama3.3-70b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 32000|8|1 ; none|false|false|false|false|false" # G-OOM 46k

  #   # "llama4-scout|1 ; superoffload_mem|unsloth|ligerloss1 ; 9500|8|1 ; none|false|false|false|false|false" # G-OOM 10k
  #   # "llama4-scout|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 9500|8|1 ; none|false|false|false|false|false" # G-OOM 10k
  #   # "llama4-scout|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 9500|8|1 ; none|false|false|false|false|false" # G-OOM 10k

  #   # "q2.5-32b|1 ; superoffload_mem|unsloth|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false" # G-OOM 51k
  #   # "q2.5-32b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false" # G-OOM 51k
  #   # "q2.5-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false" # G-OOM 51k

  #   # "q2.5-72b|1 ; superoffload_mem|unsloth|ligerloss1 ; 32000|8|1 ; none|false|false|false|false|false" # G-OOM 41k
  #   # "q2.5-72b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 32000|8|1 ; none|false|false|false|false|false" # G-OOM 41k
  #   # "q2.5-72b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 32000|8|1 ; none|false|false|false|false|false" # G-OOM 41k
    
  #   # "q3.5-35b-a3b|1 ; superoffload_mem|unsloth|ligerloss1 ; 70000|8|1 ; none|false|false|false|false|false" # fla illegal memory 75k
  #   # "q3.5-35b-a3b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 70000|8|1 ; none|false|false|false|false|false" # fla illegal memory 75k
  #   # "q3.5-35b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 70000|8|1 ; none|false|false|false|false|false" # fla illegal memory 75k
  # )
fi

# Iterate and parse the rows into scheduler metadata. Each RUNS item remains one scheduled run.
_run_models=(); _run_backends=(); _run_workloads=(); _run_policies=(); _run_specs=()
(( ${#RUNS[@]} > 0 )) || { echo "error: RUNS is empty" >&2; exit 2; }
for _run in "${RUNS[@]}"; do
  IFS=';' read -r _m _b _w _p _extra <<< "${_run}"
  _m="${_m// /}"; _b="${_b// /}"; _w="${_w// /}"; _p="${_p// /}"
  [[ -n "${_m}" && -n "${_b}" && -n "${_w}" && -n "${_p}" && -z "${_extra:-}" ]] || {
    echo "error: RUNS item must be model ; backend|recompute|ligerloss ; seq|batch|grad_accum ; policy|expact|attnact|layeract|layergc|sdparecomp, got '${_run}'" >&2
    exit 2
  }
  _m_key="${_m%%|*}"
  if [[ "${_m}" == *"|"* && -n "${M[$_m_key]:-}" ]]; then
    # shorthand|N: resolve the shorthand's path, then apply the GPU count N from the RUNS row.
    _m_spec="${M[$_m_key]%%|*}|${_m#*|}"
  elif [[ -n "${M[$_m]:-}" ]]; then
    echo "error: RUNS model shorthand '${_m}' must include GPU count, e.g. '${_m}|1'" >&2
    exit 2
  elif [[ "${_m}" == */* ]]; then
    _m_spec="${_m}"
  else
    echo "error: RUNS unknown model shorthand '${_m}' (add it to M or use model/path|num_gpus)" >&2
    exit 2
  fi
  _run_models+=("${_m_spec}"); _run_backends+=("${_b}"); _run_workloads+=("${_w}"); _run_policies+=("${_p}")
  _run_specs+=("${_m_spec};${_b};${_w};${_p}")
done
RUN_MODEL_SPECS="$(IFS=,; echo "${_run_models[*]}")"
RUN_BACKENDS="$(IFS=,; echo "${_run_backends[*]}")"
RUN_WORKLOADS="$(IFS=,; echo "${_run_workloads[*]}")"
RUN_EXP_ACT_POLICIES="$(IFS=,; echo "${_run_policies[*]}")"

ROUTER_MODES=${ROUTER_MODES:-whole}
PROFILERS=${PROFILERS:-both}
# PROFILERS=${PROFILERS:-source}
PRECISION=${PRECISION:-bf16}
LF_EXPERT_LORA_IMPLS=${LF_EXPERT_LORA_IMPLS:-split-target-parameters}

# Training
# MAX_STEPS=${MAX_STEPS:-3}
# WARMUP_STEPS=${WARMUP_STEPS:-1}
# MAX_STEPS=${MAX_STEPS:-6}
# WARMUP_STEPS=${WARMUP_STEPS:-6}
# MAX_STEPS=${MAX_STEPS:-7}
# WARMUP_STEPS=${WARMUP_STEPS:-3}
MAX_STEPS=${MAX_STEPS:-1}
WARMUP_STEPS=${WARMUP_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
# LORA_PARAMS is the canonical LoRA knob: sweep tuples, each "dropout|rank|alpha[|target]".
# Older LORA_DROPOUT/LORA_RANK/LORA_ALPHA knobs still seed the default tuple.
# rank moves memory/throughput; alpha is a scalar (no profiling effect); keep target=all for MoE expert-LoRA.
# Multi-module target uses '+' (commas separate tuples), e.g. "0.00|64|128|q_proj+k_proj" -> q_proj,k_proj.
_LORA_PARAMS_ENV_SET=false
[[ -n "${LORA_PARAMS+x}" ]] && _LORA_PARAMS_ENV_SET=true
LORA_DROPOUT=${LORA_DROPOUT:-0.00}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_PARAMS=${LORA_PARAMS:-"${LORA_DROPOUT}|${LORA_RANK}|${LORA_ALPHA}|all"}
# LORA_PARAMS=${LORA_PARAMS:-"0.00|64|128|all,0.00|16|32|all"}
SEED=${SEED:-42}

# Recompute-offload module controls
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD-hbm}
ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD=${ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD:-0}
ASYMM_DENSE_MLP_FINEGRAINED_NOGRAD_CPU_OFFLOAD=${ASYMM_DENSE_MLP_FINEGRAINED_NOGRAD_CPU_OFFLOAD:-0}

# Qwen3 MoE fine-grained/routed kernels. The three route bits are intentionally
# left unset here so Qwen3-30B-A3B + asym* + recomp-off-full-fg can auto-enable them.
ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD=${ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD:-0}
ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=${ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS:-0}
ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM=${ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM:-0}
ASYMM_QWEN3_MOE_ROUTE_LORA=${ASYMM_QWEN3_MOE_ROUTE_LORA:-0}
ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=${ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE:-fp32}
ASYMM_QWEN3_MOE_ROUTE_KERNEL_DEBUG=${ASYMM_QWEN3_MOE_ROUTE_KERNEL_DEBUG:-0}

# Outer checkpoint placement: keep every Nth outer Unsloth checkpoint root in HBM
# instead of CPU RAM. Sweep-wide default; a per-run -ohbm<N> recompute suffix in
# RUNS overrides it. Keep 0 for fixed-seq apples-to-apples comparisons; nonzero
# is for max-seq ceiling hunting (runs gain an __ohbm<N> dir suffix).
UNSLOTH_GC_OUTER_HBM_EVERY_N=${UNSLOTH_GC_OUTER_HBM_EVERY_N:-0}

# CUDA allocator
EXPANDABLE_SEG=${EXPANDABLE_SEG:-true}

# Kernel implementation toggles (1=on, 0=off)
ASYMM_EXPERT_SILU_BWD_GPU=${ASYMM_EXPERT_SILU_BWD_GPU:-1}
ASYMM_MLP_RECOMPUTE_CHUNK=${ASYMM_MLP_RECOMPUTE_CHUNK:-0}
DG_BF16_CPU_LEFT_COMPACT_GRID=${DG_BF16_CPU_LEFT_COMPACT_GRID:-0}
ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE=${ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE:-0}

# AsymGEMM backend/offload controls
# ASYM_OFFLOAD_MODULES=${ASYM_OFFLOAD_MODULES:-routed_experts}
ASYM_OFFLOAD_MODULES=${ASYM_OFFLOAD_MODULES:-all}
ASYM_STRICT=${ASYM_STRICT:-true}
REQUIRE_SM100=${REQUIRE_SM100:-1}
# Single boolean values; multi-value sweeps would make one RUNS row expand to multiple launches.
ASYM_CPU_ADAMW_GRAD_OFFLOAD=${ASYM_CPU_ADAMW_GRAD_OFFLOAD:-true}
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=${ASYM_CPU_ADAMW_WEIGHT_OFFLOAD:-true}

# Optimizer
USE_ASYM_CPU_ADAMW=${USE_ASYM_CPU_ADAMW:-true}
ASYM_CPU_ADAMW_BACKEND=${ASYM_CPU_ADAMW_BACKEND:-deepspeed}
ASYM_CPU_ADAMW_PIN_MEMORY=${ASYM_CPU_ADAMW_PIN_MEMORY:-true}
ASYM_CPU_ADAMW_FP32_MASTER=${ASYM_CPU_ADAMW_FP32_MASTER:-true}

# Execution
OVERWRITE=${OVERWRITE:-true}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-true}
DRY_RUN=${DRY_RUN:-false}
COLLECT_EXISTING=${COLLECT_EXISTING:-false}
INTERRUPT_GRACE_SECONDS=${INTERRUPT_GRACE_SECONDS:-2}
RUN_NAME=${RUN_NAME:-}

# Dataset
DATASET_OVERWRITE=${DATASET_OVERWRITE:-true}
DATASET=${DATASET:-asym_long_sft_smoke}
PREPARE_DATASETS=${PREPARE_DATASETS:-true}
DATASET_MIN_TOKENS=${DATASET_MIN_TOKENS:-auto}
DATASET_EVAL_ROWS=${DATASET_EVAL_ROWS:-1}
TEMPLATE=${TEMPLATE:-auto}
MAX_SAMPLES=${MAX_SAMPLES:-256}

# Expert activation-backfetch toggles (default off)
ASYM_OFFLOAD_ACT_RECOMPUTE=${ASYM_OFFLOAD_ACT_RECOMPUTE:-0}
ASYM_OFFLOAD_X_UNPACKED=${ASYM_OFFLOAD_X_UNPACKED:-0}

# Output and profiling
OUTPUT_ROOT=${OUTPUT_ROOT:-}
PROFILE_LEVEL=${PROFILE_LEVEL:-op}
PROFILE_LAYERS=${PROFILE_LAYERS:-all}
PROFILE_MEMORY_BREAKDOWN=${PROFILE_MEMORY_BREAKDOWN:-true}
PROFILE_MEMORY_BREAKDOWN_INTERVAL=${PROFILE_MEMORY_BREAKDOWN_INTERVAL:-1}
PROFILE_MEMORY_BREAKDOWN_STEPS=${PROFILE_MEMORY_BREAKDOWN_STEPS:-}
PROFILE_MEMORY_BREAKDOWN_MODULES=${PROFILE_MEMORY_BREAKDOWN_MODULES:-attention,linear_attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss}
PROFILE_LIVE_ACTIVATION_DETAILS=${PROFILE_LIVE_ACTIVATION_DETAILS:-${ASYM_GEMM_LF_PROFILE_LIVE_ACTIVATION_DETAILS:-true}}
PROFILE_LIVE_ACTIVATION_TOPK=${PROFILE_LIVE_ACTIVATION_TOPK:-${ASYM_GEMM_LF_PROFILE_LIVE_ACTIVATION_TOPK:-100}}
PROFILE_MEMORY_SNAPSHOT=${PROFILE_MEMORY_SNAPSHOT:-false}
PROFILE_MEMORY_SNAPSHOT_PATH=${PROFILE_MEMORY_SNAPSHOT_PATH:-}
PROFILE_EXTERNAL_MEMORY=${PROFILE_EXTERNAL_MEMORY:-false}
PROFILE_SYNC=${PROFILE_SYNC:-false}
PROFILE_MODULE_FILTER=${PROFILE_MODULE_FILTER:-attention,linear_attention,router,mlp,experts,lora,optimizer,kt}


# KT backend
KT_NUM_THREADS=${KT_NUM_THREADS:-}
KT_THREADPOOL_COUNT=${KT_THREADPOOL_COUNT:-}
KT_MAX_CACHE_DEPTH=${KT_MAX_CACHE_DEPTH:-2}
KT_TP_ENABLED=${KT_TP_ENABLED:-false}
KT_TORCHBF16_SFT_DEVICE=${KT_TORCHBF16_SFT_DEVICE:-cuda}
KT_ARM_OMP_NUM_THREADS=${KT_ARM_OMP_NUM_THREADS:-64}
KT_ARM_OMP_PROC_BIND=${KT_ARM_OMP_PROC_BIND:-close}
KT_ARM_OMP_PLACES=${KT_ARM_OMP_PLACES:-cores}
KT_ARM_SFT_TOP_K=${KT_ARM_SFT_TOP_K:-8}
KT_ARM_SFT_TOKEN_CHUNK_SIZE=${KT_ARM_SFT_TOKEN_CHUNK_SIZE:-}
KT_ARM_SFT_MAX_ROUTE_RANK_WORK=${KT_ARM_SFT_MAX_ROUTE_RANK_WORK:-}
KT_ARM_SFT_DEFAULT_MAX_ROUTE_RANK_WORK=${KT_ARM_SFT_DEFAULT_MAX_ROUTE_RANK_WORK:-1048576}
KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK=${KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK:-0}
KT_SHARE_BACKWARD_BB=${KT_SHARE_BACKWARD_BB:-}
KT_NUM_GPU_EXPERTS=${KT_NUM_GPU_EXPERTS:-}
KT_WEIGHT_PATH=${KT_WEIGHT_PATH:-}
KT_EXPERT_CHECKPOINT_PATH=${KT_EXPERT_CHECKPOINT_PATH:-}
KT_USE_LORA_EXPERTS=${KT_USE_LORA_EXPERTS:-}
KT_LORA_EXPERT_NUM=${KT_LORA_EXPERT_NUM:-}
KT_LORA_EXPERT_INTERMEDIATE_SIZE=${KT_LORA_EXPERT_INTERMEDIATE_SIZE:-}
KT_ARM_ALLOW_NSYS_WITHOUT_SOURCE_OK=${KT_ARM_ALLOW_NSYS_WITHOUT_SOURCE_OK:-0}
KT_ARM_FIRST_STEP_TIMEOUT_SECONDS=${KT_ARM_FIRST_STEP_TIMEOUT_SECONDS:-0}
CHECK_KT_CALLS=${CHECK_KT_CALLS:-1}

# DeepSpeed/SuperOffload backends
CHECK_SUPEROFFLOAD=${CHECK_SUPEROFFLOAD:-1}
CHECK_CPUADAM=${CHECK_CPUADAM:-1}

# Plotting
PLOT=${PLOT:-true}
PLOT_MEMORY_BREAKDOWN=${PLOT_MEMORY_BREAKDOWN:-true}
MEMORY_BREAKDOWN_PLOT_Y_SCALE=${MEMORY_BREAKDOWN_PLOT_Y_SCALE:-shared}
PLOT_OUTPUT_DIR=${PLOT_OUTPUT_DIR:-}


# =============================================================================
# Derived Parameters
# =============================================================================
ASYM_DIR=${ASYM_DIR:-${ROOT}}
ASYM_QWEN35_FA4_AUTO=${ASYM_QWEN35_FA4_AUTO:-1}
FLASH_ATTN=${FLASH_ATTN:-auto}
FA4_ENV_DIR=${FA4_ENV_DIR:-${ASYM_DIR}/.venv-fa4}
KT_TOOLS_DIR=${KT_TOOLS_DIR:-${ASYM_DIR}}
KT_REPO_DIR_ENV_SET=${KT_REPO_DIR+x}
KT_REPO_DIR=${KT_REPO_DIR:-$(dirname "${KT_KERNEL_DIR}")}
KT_GGUF_PY_DIR_ENV_SET=${KT_GGUF_PY_DIR+x}
KT_GGUF_PY_DIR=${KT_GGUF_PY_DIR:-${KT_REPO_DIR}/third_party/llama.cpp/gguf-py}
ENV_DIR=${ENV_DIR:-${ASYM_DIR}/.venv}
ENV_PYTHON=${ENV_PYTHON:-${ENV_DIR}/bin/python}
RUN_LF_SCRIPT="${ASYM_DIR}/scripts/lf/run_lf_lora_sft.sh"
BUILD_DATASET_SCRIPT="${ASYM_DIR}/scripts/lf/build_lf_sft_eval_pair.py"
PROFILE_NSYS_POSTPROCESS_SCRIPT="${ASYM_DIR}/scripts/lora/postprocess_nsys_lora.py"
PROFILE_POSTPROCESS_SCRIPT="${ASYM_DIR}/scripts/lf/postprocess_lf_profile_artifacts.py"
MEMORY_SCHEMA_VALIDATOR="${ASYM_DIR}/scripts/lf/validate_lf_memory_capacity_schema.py"
PLOT_SCRIPT="${ASYM_DIR}/scripts/plotting/plot_activation_recompute_sweep.py"
MEMORY_PLOT_SCRIPT="${ASYM_DIR}/scripts/plotting/plot_lf_memory_breakdown.py"
INTERCONNECT_PLOT_SCRIPT="${ASYM_DIR}/scripts/plotting/plot_lf_interconnect_ctc.py"
THROUGHPUT_PLOT_SCRIPT="${ASYM_DIR}/scripts/plotting/plot_lf_throughput.py"
UTILIZATION_PLOT_SCRIPT="${ASYM_DIR}/scripts/plotting/plot_lf_utilization.py"

# =============================================================================
# Main Logic
# =============================================================================
usage() {
  cat <<USAGE
Usage:
  scripts/lf/profile_lora_lf.sh [options]

Defaults:
  --gpus ${GPU_POOL}
  --dist-launcher ${DIST_LAUNCHER}
  --profilers ${PROFILERS}
  RUNS rows: ${#RUNS[@]}
  --output-root ${OUTPUT_ROOT}

Options:
  RUNS is the job list. Each row is:
    model ; backend|recompute|ligerloss ; seq|batch|grad_accum ; policy|expact|attnact|layeract|layergc|sdparecomp
  Unsloth-GC recompute modes take an optional trailing -ohbm<N> suffix (keep every Nth
  outer checkpoint root in HBM; omitted = ohbm0 = all roots to CPU), e.g. unsloth-off-ohbm4.
  Override it from the environment with rows separated by '||':
    RUNS='q3-30b-a3b|1 ; superoffload_mem|unsloth|ligerloss1 ; 4092|8|1 ; none|false|false|false|false|false || q3-30b-a3b|1 ; asym_cpuadamwds|norecompute|ligerloss1 ; 4092|8|1 ; none|true|true|false|true|true'

  Sweep:
  --gpus LIST                    Physical GPU pool, e.g. 0,1.
  --dist-launcher torchrun|accelerate|deepspeed
                                 Launcher for torch/zero/SuperOffload/zero3_cpuadam jobs. Default ${DIST_LAUNCHER}.
  --router-modes NAME            AsymGEMM router mode: hf or whole. Default ${ROUTER_MODES}.
  --profilers NAME               source, nsys, or both. both runs Nsight once and materializes sibling source artifacts.

  Dataset:
  --dataset NAME
  --prepare-datasets true|false  Build/audit model+length-specific LF datasets before training.
  --dataset-min-tokens N|auto    Minimum source tokens for generated/audited rows. auto uses the seq length.
  --dataset-eval-rows N
  --dataset-overwrite true|false Rewrite existing generated dataset files.
  --template NAME
  --max-samples N

  Training:
  --max-steps N                  Measured steps kept in plots/summaries.
  --warmup-steps N               Extra initial steps to run but exclude from plots/summaries. Use 5+ for stable timing; 0/1 is allowed for smoke tests.
  --learning-rate VALUE
  --lora-params LIST            LoRA sweep tuples, each dropout|rank|alpha[|target] (target default all; multi-module via '+').
  --lf-expert-lora-impls NAME   Qwen fused expert LoRA implementation: peft-target-parameters, split-target-parameters, off.
                                 Use split-target-parameters for the corrected PEFT-compatible LF/ZeRO expert LoRA path.
  --seed N
  --precision NAME

  Profiling:
  --profile-level stage|module|op|deep
  --profile-layers all|first,last|0,1,2|every4
  --profile-memory-breakdown true|false   Gate per-module memory breakdown AND saved-tensor/param attribution.
  --profile-memory-breakdown-interval N
  --profile-memory-breakdown-steps LIST
  --profile-memory-breakdown-modules LIST
  --profile-memory-snapshot auto|true|false
  --profile-memory-snapshot-path PATH
  --profile-external-memory auto|true|false
  --profile-sync true|false
  --profile-module-filter LIST
  --expandable-seg true|false   Set PYTORCH_CUDA_ALLOC_CONF expandable_segments for training jobs.
                                 Default ${EXPANDABLE_SEG}.
  --use-asym-cpu-adamw true|false
                                 Low-level forwarding control; prefer setting the backend in RUNS to asym_cpuadamwtorch or asym_cpuadamwds.
  --asym-cpu-adamw-backend torch|deepspeed
  --asym-cpu-adamw-pin-memory true|false
  --asym-cpu-adamw-fp32-master true|false
  --asym-cpu-adamw-grad-offload true|false
                                 Grad offload mode for Asym CPUAdamW backends.
  --asym-cpu-adamw-weight-offload true|false
                                 Weight offload mode for Asym CPUAdamW backends; requires grad offload.
  KT:
  --kt-kernel-dir DIR            Integrated kt-kernel source tree.
  --kt-tools-dir DIR             Helper source tree to put on PYTHONPATH for KT jobs. Defaults to ROOT.
  --kt-repo-dir DIR              KT artifact root owner. Defaults to dirname(--kt-kernel-dir).
  --kt-gguf-py-dir DIR           Vendored llama.cpp gguf-py tree. Defaults to --kt-repo-dir/third_party/llama.cpp/gguf-py.
  --kt-num-threads N
  --kt-threadpool-count N
  --kt-max-cache-depth N
  --kt-tp-enabled true|false
  --kt-torchbf16-sft-device DEV  cpu, cuda, cuda:N, or another torch device string.
  --kt-arm-omp-num-threads N
  --kt-arm-omp-proc-bind VALUE
  --kt-arm-omp-places VALUE
  --kt-arm-sft-token-chunk-size N
  --kt-share-backward-bb true|false
  --kt-num-gpu-experts N
  --kt-weight-path PATH
  --kt-expert-checkpoint-path PATH
  --kt-use-lora-experts true|false
  --kt-lora-expert-num N
  --kt-lora-expert-intermediate-size N
  --check-kt-calls true|false

  DeepSpeed/SuperOffload:
  --deepspeed-dir DIR            Local DeepSpeed/SuperOffload source tree to put before site-packages.
  --check-superoffload true|false
  --check-cpuadam true|false

  Outputs and plotting:
  --output-root DIR              Default config layout: <root>/<dataset>__lora__lf__<precision>/<model>__gpus<model_gpus>__b<batch>_s<seq>_ga<grad_accum>_w<warmup>_s<steps>_r<rank>_a<alpha>_drop0xx
                                 Per-run dirs add <backend>__<profiler>__<recompute>__pol<policy>__router<mode>/b<batch>_s<seq>_ga<grad_accum>.
  --run-name NAME                Optional config directory under <dataset>__lora__lf__<precision>.
  --plot true|false
  --plot-memory-breakdown true|false
  --memory-breakdown-plot-y-scale shared|per-plot|global
  --plot-output-dir DIR
  --run-post true|false          Run scripts/lf/test_profiling.sh after this sweep. Default ${RUN_POST}.

  Execution:
  --overwrite true|false
  --continue-on-error true|false
  --collect-existing             Skip training and regenerate plots from existing profile.json files.
  --dry-run                      Print commands without running training or plotting.
  -h, --help

  Note:
  numactl --membind=0,1 --cpunodebind=0,1 ...
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

collect_values() {
  local opt="$1"
  local -n out_ref="$2"
  shift 2
  out_ref=()
  while (($#)); do
    [[ "$1" == --* ]] && break
    out_ref+=("$1")
    shift
  done
  ((${#out_ref[@]} > 0)) || die "${opt} requires at least one value"
  REMAINING=("$@")
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

require_comma_list() {
  local name="$1"
  local value="$2"
  if [[ -n "${value}" && "${value}" =~ [[:space:]] ]]; then
    die "${name} must be comma-separated with no spaces, got '${value}'"
  fi
}

dedupe() {
  awk '!seen[$0]++'
}

bool_value() {
  case "${1,,}" in
    1|true|yes|y|on) printf 'true\n' ;;
    0|false|no|n|off) printf 'false\n' ;;
    *) die "expected true or false, got '${1}'" ;;
  esac
}

bool_tag() {
  local prefix="$1"
  case "$(bool_value "$2")" in
    true) printf '%s1\n' "${prefix}" ;;
    false) printf '%s0\n' "${prefix}" ;;
  esac
}

truthy_digit() {
  case "${1,,}" in
    1|true|yes|y|on) printf '1\n' ;;
    *) printf '0\n' ;;
  esac
}

normalize_expact_lora_a_fwd() {
  case "${1}" in
    cpu) printf 'cpu\n' ;;
    hbm) printf 'hbm\n' ;;
    *) die "ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD must be cpu or hbm, got '${1}'" ;;
  esac
}

expact_lora_a_fwd_tag() {
  case "$(normalize_expact_lora_a_fwd "$1")" in
    cpu) printf 'loraafwdcpu\n' ;;
    hbm) printf 'loraafwdhbm\n' ;;
  esac
}

expact_tag() {
  bool_tag expact "$1"
}

attnact_tag() {
  bool_tag attnact "$1"
}

layeract_tag() {
  bool_tag layeract "$1"
}

layergc_tag() {
  bool_tag layergc "$1"
}

sdparecomp_tag() {
  bool_tag sdparecomp "$1"
}

# Lever-2 toggles (AsymGEMM-only). Encoded in the run dir so each setting gets its own folder.
actrecomp_tag() {
  printf 'actrecomp%s\n' "$(truthy_digit "$1")"
}

xunpack_tag() {
  printf 'xunpack%s\n' "$(truthy_digit "$1")"
}

moefg_tag() {
  bool_tag moefg "$1"
}

dscatter_tag() {
  local value="${1:-0}"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS must be a nonnegative integer, got '${value}'"
  printf 'dscatter%s\n' "${value}"
}

qwen3_route_bool() {
  case "$(bool_value "$1")" in
    true) printf '1\n' ;;
    false) printf '0\n' ;;
  esac
}

qwen3_route_effective_flag() {
  local name="$1"
  local raw
  if [[ -n "${!name+x}" && -n "${!name}" ]]; then
    raw="${!name}"
  else
    raw="${ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM:-0}"
  fi
  qwen3_route_bool "${raw}"
}

qwen3_route_tag() {
  local fwd="$1" gather="$2" dx="$3" lora="$4" accum="$5"
  [[ -n "${accum}" ]] || accum=fp32
  printf 'route%s%s%s_lora%s_acc%s\n' \
    "$(qwen3_route_bool "${fwd}")" \
    "$(qwen3_route_bool "${gather}")" \
    "$(qwen3_route_bool "${dx}")" \
    "$(qwen3_route_bool "${lora}")" \
    "$(safe_label "${accum}")"
}

qwen3_route_kernel_code() {
  local fwd="$1" gather="$2" dx="$3"
  printf '%s%s%s\n' \
    "$(qwen3_route_bool "${fwd}")" \
    "$(qwen3_route_bool "${gather}")" \
    "$(qwen3_route_bool "${dx}")"
}

recompute_run_label() {
  local recompute="$1" kernel_code="${2:-000}"
  case "${recompute}" in
    recomp-off-full-fg) printf '%s-ker%s\n' "${recompute}" "${kernel_code}" ;;
    *) printf '%s\n' "${recompute}" ;;
  esac
}

qwen3_route_any_enabled() {
  local value
  for value in "$@"; do
    [[ "$(qwen3_route_bool "${value}")" == "1" ]] && return 0
  done
  return 1
}

is_qwen3_moe_routed_model() {
  case "$1" in
    *"Qwen3-30B-A3B"*|*"Qwen3.5-35B-A3B"*|*"Qwen3.5-122B-A10B"*) return 0 ;;
    *) return 1 ;;
  esac
}

is_known_dense_recompute_model() {
  case "$1" in
    *"Qwen3-32B"*|*"Qwen3.5-27B"*|*"Qwen2.5-32B"*|*"Qwen2.5-72B"*|*"Llama-3.3-70B"*) return 0 ;;
    *) return 1 ;;
  esac
}

validate_recompute_kernel_for_model() {
  local model="$1" kernel_code="$2" lora_flag="${3:-0}"
  if is_known_dense_recompute_model "${model}" &&
    [[ "${kernel_code}" != "000" || "$(qwen3_route_bool "${lora_flag}")" != "0" ]]; then
    die "dense model '${model}' cannot use Qwen3 MoE routed kernels; use recomp-off-full-fg-ker000 with route000_lora0 (got kernel_code=${kernel_code}, lora=$(qwen3_route_bool "${lora_flag}"))"
  fi
  if [[ "${kernel_code}" != "000" ]] && ! is_qwen3_moe_routed_model "${model}"; then
    die "recomp-off-full-fg-ker${kernel_code} is only supported for Qwen3/Qwen3.5 routed MoE; dense/non-routed model '${model}' must use recomp-off-full-fg-ker000"
  fi
}

qwen3_moe_routed_auto_default() {
  local backend="$1" recomp_stage="$2" model="$3" expert_act="$4"
  [[ "${backend}" == asym* ]] || return 1
  [[ "${recomp_stage}" == "full-fg" ]] || return 1
  [[ "${expert_act}" == "false" ]] || return 1
  is_qwen3_moe_routed_model "${model}"
}

is_recomp_off_recompute() {
  case "${1,,}" in
    recomp-off|recomp-off-base|recomp-off-attn|recomp-off-dense|recomp-off-full|recomp-off-dense-fg|recomp-off-full-fg) return 0 ;;
    *) return 1 ;;
  esac
}

recomp_off_stage_label() {
  case "${1,,}" in
    recomp-off|recomp-off-full) printf 'full\n' ;;
    recomp-off-base) printf 'base\n' ;;
    recomp-off-attn) printf 'attn\n' ;;
    recomp-off-dense) printf 'dense\n' ;;
    recomp-off-dense-fg) printf 'dense-fg\n' ;;
    recomp-off-full-fg) printf 'full-fg\n' ;;
    *) return 1 ;;
  esac
}

parse_exp_act_policy_tuple() {
  local raw="$1"
  local policy_part expact_part attnact_part layeract_part layergc_part sdparecomp_part policy expact attnact layeract layergc sdparecomp
  local -a fields
  IFS='|' read -r -a fields <<< "${raw}"
  (( ${#fields[@]} == 4 || ${#fields[@]} == 5 || ${#fields[@]} == 6 )) ||
    die "ASYMM_EXP_ACT_POLICIES item must be policy|expert_act|attn_act|layer_act[|layer_gc[|sdpa_recompute]], got '${raw}'"
  policy_part="${fields[0]}"
  expact_part="${fields[1]}"
  attnact_part="${fields[2]}"
  layeract_part="${fields[3]}"
  layergc_part="${fields[4]:-false}"
  sdparecomp_part="${fields[5]:-false}"
  [[ -n "${policy_part}" && -n "${expact_part}" && -n "${attnact_part}" && -n "${layeract_part}" && -n "${layergc_part}" && -n "${sdparecomp_part}" ]] ||
    die "empty policy, activation-offload, layer-GC, or sdpa-recompute value in ASYMM_EXP_ACT_POLICIES item '${raw}'"
  policy="$(normalize_expert_policy "${policy_part}")"
  expact="$(bool_value "${expact_part}")"
  attnact="$(bool_value "${attnact_part}")"
  layeract="$(bool_value "${layeract_part}")"
  layergc="$(bool_value "${layergc_part}")"
  sdparecomp="$(bool_value "${sdparecomp_part}")"
  if [[ "${layeract}" == "true" && "${layergc}" == "true" ]]; then
    die "RUNS policy item '${raw}' is unsupported: ASYMM_LAYER_ACT_OFFLOAD and ASYMM_LAYER_GC are mutually exclusive"
  fi
  if [[ "${layergc}" == "true" && "${policy}" != "none" ]]; then
    die "RUNS policy item '${raw}' is unsupported: ASYMM_LAYER_GC requires policy none"
  fi
  if [[ ( "${expact}" == "true" || "${attnact}" == "true" || "${layeract}" == "true" ) && "${policy}" != "none" ]]; then
    die "RUNS policy item '${raw}' is unsupported: activation offload is compared without GC/recompute"
  fi
  printf '%s|%s|%s|%s|%s|%s\n' "${policy}" "${expact}" "${attnact}" "${layeract}" "${layergc}" "${sdparecomp}"
}

optional_bool_value() {
  [[ -z "$1" ]] && return 0
  bool_value "$1"
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

lora_dropout_label() {
  local value="$1"
  [[ "${value}" =~ ^0\.[0-9][0-9]$ ]] || die "LORA_DROPOUT must use fixed 0.xx format, e.g. 0.10; got '${value}'"
  printf 'drop0%s\n' "${value#*.}"
}

infer_template() {
  local model="$1"
  local lower="${model,,}"
  local base="${lower##*/}"
  case "${base}" in
    gemma-4-*|gemma4-*) printf 'gemma4\n' ;;
    llama-4-*|llama4-*) printf 'llama4\n' ;;
    llama-3*|llama3-*|meta-llama-3*) printf 'llama3\n' ;;
    llama-2*|llama2-*) printf 'llama2\n' ;;
    qwen2-vl-*|qwen2.5-vl-*|qvq-*) printf 'qwen2_vl\n' ;;
    qwen2-audio-*) printf 'qwen2_audio\n' ;;
    qwen2.5-omni-*) printf 'qwen2_omni\n' ;;
    qwen3-omni-*-thinking*) printf 'qwen3_omni\n' ;;
    qwen3-omni-*) printf 'qwen3_omni_nothink\n' ;;
    qwen3-vl-*-thinking*) printf 'qwen3_vl\n' ;;
    qwen3-vl-*) printf 'qwen3_vl_nothink\n' ;;
    qwen3.5-*) printf 'qwen3_5\n' ;;
    qwen3-*-instruct*|qwen3-next-*-instruct*) printf 'qwen3_nothink\n' ;;
    qwen3-*|qwen3-next-*) printf 'qwen3\n' ;;
    qwen2.5-*|qwen2-*|qwq-*) printf 'qwen\n' ;;
    *) echo "error: TEMPLATE=auto cannot infer a safe template for '${model}'. Set TEMPLATE or --template explicitly." >&2; return 2 ;;
  esac
}

is_qwen35_model_name() {
  case "${1,,}" in
    *qwen3.5*|*qwen3_5*) return 0 ;;
    *) return 1 ;;
  esac
}

flash_attn_tag() {
  case "${1}" in
    auto|"") printf 'attnauto\n' ;;
    disabled) printf 'attndisabled\n' ;;
    sdpa) printf 'attnsdpa\n' ;;
    fa2) printf 'attnfa2\n' ;;
    fa3) printf 'attnfa3\n' ;;
    fa4) printf 'attnfa4\n' ;;
    *) printf 'attn%s\n' "$(safe_label "${1}")" ;;
  esac
}

resolve_current_runtime_for_model() {
  local model="$1"
  CURRENT_LF_DIR="${LF_DIR}"
  CURRENT_ENV_DIR="${ENV_DIR}"
  CURRENT_ENV_PYTHON="${ENV_PYTHON}"
  CURRENT_FLASH_ATTN="${FLASH_ATTN}"
  if [[ "${ASYM_QWEN35_FA4_AUTO}" == "1" ]] && is_qwen35_model_name "${model}"; then
    [[ "${_LF_DIR_ENV_SET}" == "true" ]] || CURRENT_LF_DIR="${LF_FA4_DIR}"
    [[ "${_ENV_DIR_ENV_SET}" == "true" ]] || CURRENT_ENV_DIR="${FA4_ENV_DIR}"
    [[ "${_FLASH_ATTN_ENV_SET}" == "true" ]] || CURRENT_FLASH_ATTN=fa4
  fi
  CURRENT_ENV_PYTHON="${CURRENT_ENV_DIR}/bin/python"
  CURRENT_FLASH_ATTN_LABEL="$(flash_attn_tag "${CURRENT_FLASH_ATTN}")"
}

validate_current_fa4_runtime() {
  local model="$1"
  [[ "${CURRENT_FLASH_ATTN}" == "fa4" ]] || return 0
  [[ "${DRY_RUN}" == "true" || "${COLLECT_EXISTING}" == "true" ]] && return 0
  [[ -d "${CURRENT_LF_DIR}" ]] || die "qwen3.5 FA4 path missing LF_DIR=${CURRENT_LF_DIR}"
  [[ -x "${CURRENT_ENV_PYTHON}" ]] || die "qwen3.5 FA4 path missing Python ${CURRENT_ENV_PYTHON}; run ${ASYM_DIR}/scripts/lf/bootstrap_lf_venv_fa4.sh"
  PYTHONPATH="${CURRENT_LF_DIR}/src:${PYTHONPATH:-}" "${CURRENT_ENV_PYTHON}" - "${model}" <<'PY' >/dev/null
import importlib.metadata as md
import sys
from transformers.utils import is_flash_attn_4_available
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

model = sys.argv[1]
if not is_flash_attn_4_available():
    raise SystemExit(f"FlashAttention-4 unavailable for {model}")
md.version("flash-attn-4")
assert flash_attn_func is not None and flash_attn_varlen_func is not None
PY
}

nonnegative_int() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer, got '${value}'"
}

positive_int() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer, got '${value}'"
}

parse_workload_tuple() {
  local raw="$1"
  local seq_len batch grad_accum
  local -a fields

  IFS='|' read -r -a fields <<< "${raw}"
  ((${#fields[@]} == 3)) || die "workload item must be seq_len|per_device_batch_size|gradient_accumulation_steps, got '${raw}'"

  seq_len="${fields[0]}"
  batch="${fields[1]}"
  grad_accum="${fields[2]}"
  [[ -n "${seq_len}" && -n "${batch}" && -n "${grad_accum}" ]] || die "empty value in workload item '${raw}'"
  positive_int "workload seq_len in '${raw}'" "${seq_len}"
  positive_int "workload per-device batch size in '${raw}'" "${batch}"
  positive_int "workload gradient accumulation steps in '${raw}'" "${grad_accum}"

  printf '%s|%s|%s\n' "${seq_len}" "${batch}" "${grad_accum}"
}

parse_model_spec() {
  local spec="$1"
  local -a fields

  IFS='|' read -r -a fields <<< "${spec}"
  ((${#fields[@]} <= 2)) || die "model spec must be model|num_gpus, got '${spec}'"

  parsed_model_name="${fields[0]}"
  if ((${#fields[@]} >= 2)); then
    parsed_model_gpu_count="${fields[1]}"
  else
    parsed_model_gpu_count=1
  fi

  [[ -n "${parsed_model_name}" ]] || die "empty model name in model spec '${spec}'"
  [[ -n "${parsed_model_gpu_count}" ]] || die "empty model GPU count in model spec '${spec}'"
  positive_int "model GPU count for ${parsed_model_name}" "${parsed_model_gpu_count}"
}

gpu_slice() {
  local count="$1"
  local -a selected=()
  local index
  ((count <= ${#gpus[@]})) || die "model requests ${count} GPU(s), but GPU pool only has ${#gpus[@]}: ${gpus[*]}"
  for ((index = 0; index < count; index++)); do
    selected+=("${gpus[${index}]}")
  done
  local IFS=,
  printf '%s\n' "${selected[*]}"
}

backend_gpu_count() {
  local backend="$1"
  local model_gpu_count="$2"
  case "${backend}" in
    asym|asym_torch|asym_cpuadamwtorch|asym_cpuadamwds) printf '1\n' ;;
    torch|zero2|zero3|zero3_offload|zero3_offload_mem|zero3_offload_mem_nocpuadamw|zero3_offload_opnvme|zero3_offload_panvme|zero3_offload_mem_opnvme|zero3_offload_mem_panvme|zero3_cpuadam|superoffload|superoffload_mem|superoffload_mem_nocpuadamw|superoffload_mem_opnvme|superoffload_mem_panvme) printf '%s\n' "${model_gpu_count}" ;;
    kt_torchbf16|kt_armbf16) printf '1\n' ;;
    *) die "internal backend label must be torch, asym, asym_torch, asym_cpuadamwtorch, asym_cpuadamwds, zero2, zero3, zero3_offload, zero3_offload_mem, zero3_offload_mem_nocpuadamw, zero3_offload_opnvme, zero3_offload_panvme, zero3_offload_mem_opnvme, zero3_offload_mem_panvme, zero3_cpuadam, superoffload, superoffload_mem, superoffload_mem_nocpuadamw, superoffload_mem_opnvme, superoffload_mem_panvme, kt_torchbf16, or kt_armbf16, got '${backend}'" ;;
  esac
}

zero_deepspeed_config() {
  local _name
  case "${1}" in
    zero2) _name=ds_z2_config.json ;;
    zero3) _name=ds_z3_config.json ;;
    zero3_offload) _name=ds_z3_offload_config.json ;;
    zero3_offload_mem) _name=ds_z3_offload_mem_config.json ;;
    zero3_offload_mem_nocpuadamw) _name=ds_z3_offload_mem_nocpuadamw_config.json ;;
    zero3_offload_opnvme) _name=ds_z3_offload_opnvme_config.json ;;
    zero3_offload_panvme) _name=ds_z3_offload_panvme_config.json ;;
    zero3_offload_mem_opnvme) _name=ds_z3_offload_mem_opnvme_config.json ;;
    zero3_offload_mem_panvme) _name=ds_z3_offload_mem_panvme_config.json ;;
    zero3_cpuadam) _name=ds_z3_cpuadam_config.json ;;
    superoffload) _name=ds_z3_superoffload_config.json ;;
    superoffload_mem) _name=ds_z3_superoffload_mem_config.json ;;
    superoffload_mem_nocpuadamw) _name=ds_z3_superoffload_mem_nocpuadamw_config.json ;;
    superoffload_mem_opnvme) _name=ds_z3_superoffload_mem_opnvme_config.json ;;
    superoffload_mem_panvme) _name=ds_z3_superoffload_mem_panvme_config.json ;;
    *) return 1 ;;
  esac
  if [[ -f "${LF_DIR}/examples/deepspeed/${_name}" ]]; then
    printf '%s\n' "${LF_DIR}/examples/deepspeed/${_name}"
  else
    printf '%s\n' "${LF_DS_CONFIG_DIR}/${_name}"
  fi
}

is_zero_backend() {
  case "${1}" in
    zero2|zero3|zero3_offload|zero3_offload_mem|zero3_offload_mem_nocpuadamw|zero3_offload_opnvme|zero3_offload_panvme|zero3_offload_mem_opnvme|zero3_offload_mem_panvme|zero3_cpuadam|superoffload|superoffload_mem|superoffload_mem_nocpuadamw|superoffload_mem_opnvme|superoffload_mem_panvme) return 0 ;;
    *) return 1 ;;
  esac
}

is_policy_independent_backend() {
  case "${1}" in
    torch|zero2|zero3|zero3_offload|zero3_offload_mem|zero3_offload_mem_nocpuadamw|zero3_offload_opnvme|zero3_offload_panvme|zero3_offload_mem_opnvme|zero3_offload_mem_panvme|zero3_cpuadam|superoffload|superoffload_mem|superoffload_mem_nocpuadamw|superoffload_mem_opnvme|superoffload_mem_panvme|kt_*) return 0 ;;
    *) return 1 ;;
  esac
}

# Collapse AsymGEMM-only policy axes to inert values when they're no-ops: policy-independent backends or
# recompute=recomp. Mutates the caller's dynamically-scoped vars, so callers must declare them `local` first.
canonicalize_policy_axis_for_inert_run() {
  local backend="${1}"
  local recompute="${2:-}"
  # Generic activation-offload baselines (off-layer / any expert|attn|layer-GC offload flag) are NOT inert on
  # non-asym backends -- they drive the backend-agnostic generic_offload hook. Keep their policy axes intact.
  if [[ "${expert_policy}" == "off-layer" || "${ASYMM_EXPERT_ACT_OFFLOAD}" == "true" || "${ASYMM_ATTN_ACT_OFFLOAD}" == "true" || "${ASYMM_LAYER_GC}" == "true" ]]; then
    return 0
  fi
  { is_policy_independent_backend "${backend}" || [[ "${recompute}" == "recomp" ]]; } || return 0
  expert_policy=none
  ASYMM_EXPERT_ACT_OFFLOAD=false; expact_label="$(expact_tag false)"
  ASYMM_ATTN_ACT_OFFLOAD=false; attnact_label="$(attnact_tag false)"
  ASYMM_LAYER_ACT_OFFLOAD=false; layeract_label="$(layeract_tag false)"
  ASYMM_LAYER_GC=false; layergc_label="$(layergc_tag false)"
  ASYMM_ATTN_SDPA_RECOMPUTE=false; sdparecomp_label="$(sdparecomp_tag false)"
  ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=hbm; expact_lora_a_fwd_label="$(expact_lora_a_fwd_tag hbm)"
  ASYM_OFFLOAD_ACT_RECOMPUTE=0; actrecomp_label="$(actrecomp_tag 0)"
  ASYM_OFFLOAD_X_UNPACKED=0; xunpack_label="$(xunpack_tag 0)"
  ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD=0; moefg_label="$(moefg_tag 0)"
  ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0; dscatter_label="$(dscatter_tag 0)"
}

recompute_kernel_code_label() {
  local norm="${1,,}"
  norm="${norm//_/-}"
  case "${norm}" in
    recomp-off-full-fg-ker[01][01][01]|recompoff-full-fg-ker[01][01][01])
      printf '%s\n' "${norm: -3}"
      ;;
    recomp-off-full-fg-ker*|recompoff-full-fg-ker*)
      die "expected recomp-off-full-fg-kerXYZ with X/Y/Z in {0,1}, got '${1}'"
      ;;
  esac
}

# Trailing -ohbm<N> recompute suffix -> UNSLOTH_GC_OUTER_HBM_EVERY_N for the run.
recompute_outer_hbm_n() {
  local norm="${1,,}"
  norm="${norm//_/-}"
  case "${norm}" in
    *-ohbm*)
      local n="${norm##*-ohbm}"
      [[ "${n}" =~ ^[0-9]+$ ]] || die "expected trailing -ohbm<N> with a nonnegative integer N (after any -kerXYZ), got '${1}'"
      printf '%s\n' "${n}"
      ;;
    *) printf '0\n' ;;
  esac
}

recompute_token_without_ohbm() {
  local norm="${1,,}"
  norm="${norm//_/-}"
  printf '%s\n' "${norm%-ohbm*}"
}

recompute_label() {
  local kernel_code
  kernel_code="$(recompute_kernel_code_label "$1")"
  if [[ -n "${kernel_code}" ]]; then
    printf 'recomp-off-full-fg\n'
    return
  fi
  case "${1,,}" in
    norecomp|recomp|unsloth|unsloth-off) printf '%s\n' "${1,,}" ;;
    unslothgc|unsloth_gc|unsloth-gc) printf 'unsloth\n' ;;
    unslothoff|unsloth_off|unsloth-savecpu|unsloth_savecpu|unsloth-recomp-off|unsloth_recomp_off) printf 'unsloth-off\n' ;;
    norecompute|no_recompute|no-recompute) printf 'norecomp\n' ;;
    recompute) printf 'recomp\n' ;;
    recomp-off|recompoff|recomp_off) printf 'recomp-off\n' ;;
    recomp-off-base|recompoff-base|recomp_off_base) printf 'recomp-off-base\n' ;;
    recomp-off-attn|recompoff-attn|recomp_off_attn) printf 'recomp-off-attn\n' ;;
    recomp-off-dense|recompoff-dense|recomp_off_dense) printf 'recomp-off-dense\n' ;;
    recomp-off-full|recompoff-full|recomp_off_full) printf 'recomp-off-full\n' ;;
    recomp-off-dense-fg|recompoff-dense-fg|recomp_off_dense_fg) printf 'recomp-off-dense-fg\n' ;;
    recomp-off-full-fg|recompoff-full-fg|recomp_off_full_fg) printf 'recomp-off-full-fg\n' ;;
    *) die "expected recompute mode norecomp/recomp/unsloth/unsloth-off/recomp-off[-base|-attn|-dense|-full|-dense-fg|-full-fg] or norecompute/recompute; got '${1}'" ;;
  esac
}

liger_loss_label() {
  case "${1,,}" in
    ligerloss0|ligerloss1) printf '%s\n' "${1,,}" ;;
    *) die "expected Liger-loss mode ligerloss0 or ligerloss1; got '${1}'" ;;
  esac
}

normalize_expert_policy() {
  local raw="$1"
  case "${raw}" in
    none|gc-exp|gc-attn-exp|gc-layer|off-layer|tok-le0|tok-le0-act)
      printf '%s\n' "${raw}"
      return
      ;;
  esac
  if [[ "${raw}" =~ ^tok-le[1-9][0-9]*(-act)?$ || "${raw}" =~ ^tok-ge[1-9][0-9]*(-act)?$ || "${raw}" =~ ^tok[1-9][0-9]*-[1-9][0-9]*(-act)?$ ]]; then
    printf '%s\n' "${raw}"
    return
  fi
  die "invalid expert policy '${1}'; expected none, gc-exp, gc-attn-exp, gc-layer, off-layer, tok-le0, tok-le0-act, tok-leN, tok-geN, tokA-B, or -act variants"
}

cpuadam_backend_for_label() {
  case "${1}" in
    asym_cpuadamwtorch) printf 'torch\n' ;;
    asym_cpuadamwds) printf 'deepspeed\n' ;;
    *) return 1 ;;
  esac
}

router_mode_label() {
  case "${1,,}" in
    hf|whole) printf '%s\n' "${1,,}" ;;
    *) die "router mode must be hf or whole, got '${1}'" ;;
  esac
}

append_backend_spec() {
  local raw="$1"
  local backend_part recompute_part liger_loss_part backend recompute_token recompute_mode recompute_kernel_code liger_loss
  local recompute_token_base recompute_outer_hbm_every_n norm_kernel_code norm_outer_hbm
  local pipe_chars
  local -a recompute_tokens

  pipe_chars="${raw//[^|]/}"
  case "${#pipe_chars}" in
    1)
      IFS='|' read -r backend_part recompute_part <<< "${raw}"
      liger_loss_part="ligerloss0"
      ;;
    2)
      IFS='|' read -r backend_part recompute_part liger_loss_part <<< "${raw}"
      [[ -n "${liger_loss_part}" ]] || die "empty Liger-loss mode in backend spec '${raw}'"
      ;;
    3|4)
      # Already-normalized spec (backend|mode|liger|ker[|ohbm]) looping back
      # through the axis expansion; reassemble the recompute token and reparse.
      IFS='|' read -r backend_part recompute_part liger_loss_part norm_kernel_code norm_outer_hbm <<< "${raw}"
      [[ -n "${liger_loss_part}" ]] || die "empty Liger-loss mode in backend spec '${raw}'"
      [[ -n "${norm_kernel_code}" ]] && recompute_part="${recompute_part}-ker${norm_kernel_code}"
      if [[ -n "${norm_outer_hbm:-}" && "${norm_outer_hbm}" != "0" ]]; then
        recompute_part="${recompute_part}-ohbm${norm_outer_hbm}"
      fi
      ;;
    *)
      die "backend spec must be backend|recompute or backend|recompute|ligerloss0/ligerloss1, got '${raw}'"
      ;;
  esac

  [[ -n "${backend_part}" ]] || die "empty backend in backend spec '${raw}'"
  [[ -n "${recompute_part}" ]] || die "empty recompute mode in backend spec '${raw}'"
  case "${backend_part,,}" in
    torch) backend=torch ;;
    asym) backend=asym ;;
    asym_torch) backend=asym_torch ;;
    asym_cpuadamwtorch) backend=asym_cpuadamwtorch ;;
    asym_cpuadamwds) backend=asym_cpuadamwds ;;
    zero2) backend=zero2 ;;
    zero3) backend=zero3 ;;
    zero3_offload) backend=zero3_offload ;;
    zero3_offload_mem) backend=zero3_offload_mem ;;
    zero3_offload_mem_nocpuadamw) backend=zero3_offload_mem_nocpuadamw ;;
    zero3_offload_opnvme) backend=zero3_offload_opnvme ;;
    zero3_offload_panvme) backend=zero3_offload_panvme ;;
    zero3_offload_mem_opnvme) backend=zero3_offload_mem_opnvme ;;
    zero3_offload_mem_panvme) backend=zero3_offload_mem_panvme ;;
    zero3_cpuadam) backend=zero3_cpuadam ;;
    superoffload) backend=superoffload ;;
    superoffload_mem) backend=superoffload_mem ;;
    superoffload_mem_nocpuadamw) backend=superoffload_mem_nocpuadamw ;;
    superoffload_mem_opnvme) backend=superoffload_mem_opnvme ;;
    superoffload_mem_panvme) backend=superoffload_mem_panvme ;;
    kt_torchbf16) backend=kt_torchbf16 ;;
    kt_armbf16) backend=kt_armbf16 ;;
    *) die "backend must be torch, asym, asym_torch, asym_cpuadamwtorch, asym_cpuadamwds, zero2, zero3, zero3_offload, zero3_offload_mem, zero3_offload_mem_nocpuadamw, zero3_offload_opnvme, zero3_offload_panvme, zero3_offload_mem_opnvme, zero3_offload_mem_panvme, zero3_cpuadam, superoffload, superoffload_mem, superoffload_mem_nocpuadamw, superoffload_mem_opnvme, superoffload_mem_panvme, kt_torchbf16, or kt_armbf16, got '${backend_part}'" ;;
  esac
  liger_loss="$(liger_loss_label "${liger_loss_part}")"

  mapfile -t recompute_tokens < <(tokens "${recompute_part}")
  ((${#recompute_tokens[@]} > 0)) || die "empty recompute mode in backend spec '${raw}'"
  for recompute_token in "${recompute_tokens[@]}"; do
    if [[ "${recompute_token,,}" == "both" ]]; then
      backend_specs_raw+=("${backend}|norecomp|${liger_loss}" "${backend}|recomp|${liger_loss}")
      continue
    fi
    recompute_outer_hbm_every_n="$(recompute_outer_hbm_n "${recompute_token}")"
    recompute_token_base="$(recompute_token_without_ohbm "${recompute_token}")"
    recompute_mode="$(recompute_label "${recompute_token_base}")"
    recompute_kernel_code="$(recompute_kernel_code_label "${recompute_token_base}")"
    if (( recompute_outer_hbm_every_n > 0 )); then
      if [[ "${recompute_mode}" != "unsloth" && "${recompute_mode}" != "unsloth-off" ]] &&
        ! is_recomp_off_recompute "${recompute_mode}"; then
        die "-ohbm<N> requires an Unsloth-GC recompute mode (unsloth, unsloth-off, or recomp-off-*); got '${recompute_token}'"
      fi
      backend_specs_raw+=("${backend}|${recompute_mode}|${liger_loss}|${recompute_kernel_code}|${recompute_outer_hbm_every_n}")
    elif [[ -n "${recompute_kernel_code}" ]]; then
      backend_specs_raw+=("${backend}|${recompute_mode}|${liger_loss}|${recompute_kernel_code}")
    else
      backend_specs_raw+=("${backend}|${recompute_mode}|${liger_loss}")
    fi
  done
}

backend_entries_for_run_spec() {
  local raw="$1"
  local -a backend_specs_raw=()
  append_backend_spec "${raw}"
  printf '%s\n' "${backend_specs_raw[@]}"
}

normalize_run_spec_entry() {
  local raw="$1"
  local model_part backend_part workload_part policy_part extra
  local normalized_model parsed_workload parsed_policy
  local -a backend_entries

  IFS=';' read -r model_part backend_part workload_part policy_part extra <<< "${raw}"
  model_part="${model_part// /}"
  backend_part="${backend_part// /}"
  workload_part="${workload_part// /}"
  policy_part="${policy_part// /}"
  [[ -n "${model_part}" && -n "${backend_part}" && -n "${workload_part}" && -n "${policy_part}" && -z "${extra:-}" ]] ||
    die "RUNS item must be model ; backend|recompute[|ligerloss] ; seq|batch|grad_accum ; policy|expact|attnact|layeract|layergc|sdparecomp, got '${raw}'"

  if [[ "${model_part}" == *"|"* && -n "${M[${model_part%%|*}]:-}" ]]; then
    # shorthand|N: resolve the shorthand's path, then apply the GPU count N from the RUNS row.
    model_part="${M[${model_part%%|*}]%%|*}|${model_part#*|}"
  elif [[ -n "${M[$model_part]:-}" ]]; then
    die "RUNS model shorthand '${model_part}' must include GPU count, e.g. '${model_part}|1'"
  elif [[ "${model_part}" != */* ]]; then
    die "RUNS unknown model shorthand '${model_part%%|*}' (add it to M or use model/path|num_gpus)"
  fi
  parse_model_spec "${model_part}"
  normalized_model="${parsed_model_name}|${parsed_model_gpu_count}"
  parsed_workload="$(parse_workload_tuple "${workload_part}")"
  parsed_policy="$(parse_exp_act_policy_tuple "${policy_part}")"

  mapfile -t backend_entries < <(backend_entries_for_run_spec "${backend_part}")
  ((${#backend_entries[@]} == 1)) ||
    die "RUNS backend spec must resolve to exactly one job; got ${#backend_entries[@]} from '${backend_part}'"
  printf '%s;%s;%s;%s\n' "${normalized_model}" "${backend_entries[0]}" "${parsed_workload}" "${parsed_policy}"
}

profiler_label() {
  case "${1,,}" in
    source|nsys|both) printf '%s\n' "${1,,}" ;;
    *) die "profiler must be source, nsys, or both, got '${1}'" ;;
  esac
}

dist_launcher_label() {
  case "${1,,}" in
    torchrun) printf 'torchrun\n' ;;
    accelerate|accelerate_launch) printf 'accelerate\n' ;;
    deepspeed|ds) printf 'deepspeed\n' ;;
    *) die "dist launcher must be torchrun, accelerate, or deepspeed, got '${1}'" ;;
  esac
}

kt_gguf_available() {
  [[ -f "${KT_GGUF_PY_DIR}/gguf/gguf_reader.py" ]] && return 0
  [[ -x "${ENV_PYTHON}" ]] || return 1
  PYTHONPATH="${KT_KERNEL_DIR}:${LF_DIR}/src:${PYTHONPATH:-}" "${ENV_PYTHON}" - <<'PY' >/dev/null 2>&1
from gguf.gguf_reader import GGUFReader
PY
}

allocator_conf_with_expandable_seg() {
  local enabled="$1"
  local existing="${PYTORCH_CUDA_ALLOC_CONF:-}"
  local enabled_value item key output="" sep=""
  local -a items=()

  case "$(bool_value "${enabled}")" in
    true) enabled_value=True ;;
    false) enabled_value=False ;;
  esac

  IFS=, read -r -a items <<< "${existing}"
  for item in "${items[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    [[ -n "${item}" ]] || continue
    key="${item%%:*}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    [[ "${key}" == "expandable_segments" ]] && continue
    output+="${sep}${item}"
    sep=","
  done
  output+="${sep}expandable_segments:${enabled_value}"
  printf '%s\n' "${output}"
}

existing_memory_breakdown_valid() {
  local seq_root="$1"
  local summary_json
  for summary_json in \
    "${seq_root}/memory_breakdown_summary.json" \
    "${seq_root}/memory_breakdown/memory_breakdown_summary.json"; do
    [[ -f "${summary_json}" ]] || continue
    "${ENV_PYTHON}" "${MEMORY_SCHEMA_VALIDATOR}" \
      --memory-breakdown-summary "${summary_json}" >/dev/null 2>&1 && return 0
  done
  return 1
}

job_profile_complete() {
  local profile_json="$1"
  local seq_root="$2"
  local expected_backend="$3"
  local expected_seq_len="$4"
  local expected_model_name="$5"
  local expected_recompute="$6"
  local expected_offload_modules="$7"
  local expected_expact="$8"
  local expected_attnact="$9"
  local expected_layeract="${10}"
  local expected_layergc="${11}"
  local expected_lf_expert_lora_impl="${12}"
  local expected_expact_lora_a_fwd="${13}"
  local expected_grad_offload="${14}"
  local require_memory_breakdown="${15}"
  local expected_liger_loss="${16:-}"
  local expected_unsloth_recompute_save_on_cpu="${17:-}"
  if [[ -z "${expected_unsloth_recompute_save_on_cpu}" ]]; then
    if [[ "${expected_recompute}" == "unsloth-off" ]] || is_recomp_off_recompute "${expected_recompute}"; then
      expected_unsloth_recompute_save_on_cpu=true
    fi
  fi

  existing_profile_complete \
    "${profile_json}" \
    "${expected_backend}" \
    "${expected_seq_len}" \
    "${expected_model_name}" \
    "all" \
    "${expected_recompute}" \
    "${expected_offload_modules}" \
    "${expected_expact}" \
    "${expected_attnact}" \
    "${expected_layeract}" \
    "${expected_layergc}" \
	    "${expected_lf_expert_lora_impl}" \
	    "${expected_expact_lora_a_fwd}" \
	    "${expected_grad_offload}" \
	    "${expected_liger_loss}" \
	    "${expected_unsloth_recompute_save_on_cpu}" || return 1
  [[ "${require_memory_breakdown}" != "true" ]] || existing_memory_breakdown_valid "${seq_root}"
}

existing_profile_complete() {
  local profile_json="$1"
  local expected_backend="${2:-}"
  local expected_seq_len="${3:-}"
  local expected_model_name="${4:-}"
  local expected_lora_target="${5:-}"
  local expected_recompute="${6:-}"
  local expected_offload_modules="${7:-}"
  local expected_expact="${8:-}"
  local expected_attnact="${9:-}"
  local expected_layeract="${10:-}"
  local expected_layergc="${11:-}"
  local expected_lf_expert_lora_impl="${12:-}"
  local expected_expact_lora_a_fwd="${13:-}"
  local expected_grad_offload="${14:-}"
  local expected_liger_loss="${15:-}"
  local expected_unsloth_recompute_save_on_cpu="${16:-}"
  local current_profile_sync="${PROFILE_SYNC:-}"
  local current_outer_hbm_every_n="${UNSLOTH_GC_OUTER_HBM_EVERY_N:-0}"
  local current_batch="${PER_DEVICE_TRAIN_BATCH_SIZE:-}"
  local current_grad_accum="${GRADIENT_ACCUMULATION_STEPS:-}"
  local current_lora_rank="${LORA_RANK:-}"
  local current_lora_dropout="${LORA_DROPOUT:-}"
  local current_cache_depth="${KT_MAX_CACHE_DEPTH:-}"
  local current_top_k="${KT_ARM_SFT_TOP_K:-}"
  local current_token_chunk_size="${KT_ARM_SFT_TOKEN_CHUNK_SIZE:-}"
  local current_route_rank_limit="${KT_ARM_SFT_MAX_ROUTE_RANK_WORK:-}"
  local current_default_route_rank_limit="${KT_ARM_SFT_DEFAULT_MAX_ROUTE_RANK_WORK:-}"
  local current_allow_unvalidated_route_rank="${KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK:-0}"
  if [[ -z "${expected_unsloth_recompute_save_on_cpu}" ]]; then
    if [[ "${expected_recompute}" == "unsloth-off" ]] || is_recomp_off_recompute "${expected_recompute}"; then
      expected_unsloth_recompute_save_on_cpu=true
    fi
  fi
  [[ -f "${profile_json}" ]] || return 1
  "${ENV_PYTHON}" - "${profile_json}" "${expected_backend}" "${expected_seq_len}" "${expected_model_name}" \
    "${expected_lora_target}" "${expected_recompute}" "${current_batch}" "${current_grad_accum}" "${current_lora_rank}" \
    "${current_lora_dropout}" "${current_cache_depth}" "${current_top_k}" "${current_token_chunk_size}" \
    "${current_route_rank_limit}" "${current_default_route_rank_limit}" \
    "${current_allow_unvalidated_route_rank}" "${expected_offload_modules}" "${expected_expact}" "${expected_attnact}" "${expected_layeract}" "${expected_layergc}" \
    "${expected_lf_expert_lora_impl}" "${expected_expact_lora_a_fwd}" "${expected_grad_offload}" "${expected_liger_loss}" \
    "${expected_unsloth_recompute_save_on_cpu}" "${current_profile_sync}" "${current_outer_hbm_every_n}" <<'PY' >/dev/null 2>&1
import json
import math
import sys

profile = json.load(open(sys.argv[1], encoding="utf-8"))
expected_backend = sys.argv[2]
expected_seq_len = sys.argv[3]
expected_model_name = sys.argv[4]
expected_lora_target = sys.argv[5]
expected_recompute = sys.argv[6]
expected_batch = sys.argv[7]
expected_grad_accum = sys.argv[8]
expected_rank = sys.argv[9]
expected_dropout = sys.argv[10]
expected_cache_depth = sys.argv[11]
expected_top_k = sys.argv[12]
expected_token_chunk = sys.argv[13]
expected_limit = sys.argv[14]
expected_default_limit = sys.argv[15]
allow_unvalidated = sys.argv[16]
expected_offload_modules = sys.argv[17]
expected_expact = sys.argv[18]
expected_attnact = sys.argv[19]
expected_layeract = sys.argv[20]
expected_layergc = sys.argv[21] if len(sys.argv) > 21 else ""
expected_lf_expert_lora_impl = sys.argv[22] if len(sys.argv) > 22 else ""
expected_expact_lora_a_fwd = sys.argv[23] if len(sys.argv) > 23 else ""
expected_grad_offload = sys.argv[24] if len(sys.argv) > 24 else ""
expected_liger_loss = sys.argv[25] if len(sys.argv) > 25 else ""
expected_unsloth_recompute_save_on_cpu = sys.argv[26] if len(sys.argv) > 26 else ""
expected_profile_sync = sys.argv[27] if len(sys.argv) > 27 else ""
expected_outer_hbm_every_n = sys.argv[28] if len(sys.argv) > 28 else ""
source_profile = profile.get("source_profile", {})
source_profile = source_profile if isinstance(source_profile, dict) and source_profile else profile
if profile.get("partial") is True:
    raise SystemExit("partial profile")
if source_profile.get("partial") is True:
    raise SystemExit("partial nested source profile")
config = source_profile.get("config", {})
config = config if isinstance(config, dict) else {}
backend = str(config.get("backend") or "") if isinstance(config, dict) else ""
def heartbeat_stage(payload):
    heartbeat = payload.get("heartbeat", {})
    latest = heartbeat.get("latest", {}) if isinstance(heartbeat, dict) else {}
    return latest.get("stage") if isinstance(latest, dict) else None
stage = heartbeat_stage(profile)
if stage is not None and stage not in {"source_profile_written", "trainer_end", "kt_lora_pointer_refresh_end"}:
    raise SystemExit(f"incomplete heartbeat stage: {stage}")
nested_stage = heartbeat_stage(source_profile)
if source_profile is not profile:
    if nested_stage != "source_profile_written":
        raise SystemExit(f"incomplete nested source heartbeat stage: {nested_stage or '<missing>'}")
elif backend.startswith("kt_"):
    if nested_stage != "source_profile_written":
        raise SystemExit(f"incomplete KT source heartbeat stage: {nested_stage or '<missing>'}")
elif nested_stage is not None and nested_stage not in {"source_profile_written", "trainer_end", "kt_lora_pointer_refresh_end"}:
    raise SystemExit(f"incomplete source heartbeat stage: {nested_stage}")
if expected_backend and backend != expected_backend:
    raise SystemExit(f"profile backend mismatch: expected {expected_backend}, got {backend or '<missing>'}")
if expected_seq_len:
    try:
        actual_seq_len = int(config.get("seq_len"))
        wanted_seq_len = int(expected_seq_len)
    except (TypeError, ValueError):
        raise SystemExit("profile seq_len missing or invalid")
    if actual_seq_len != wanted_seq_len:
        raise SystemExit(f"profile seq_len mismatch: expected {wanted_seq_len}, got {actual_seq_len}")
if expected_model_name:
    model_name = str(config.get("model_name_or_path") or "")
    if model_name != expected_model_name:
        raise SystemExit(f"profile model mismatch: expected {expected_model_name}, got {model_name or '<missing>'}")
if expected_lora_target:
    lora_target = str(config.get("lora_target") or "")
    if lora_target != expected_lora_target:
        raise SystemExit(f"profile lora_target mismatch: expected {expected_lora_target}, got {lora_target or '<missing>'}")
if expected_liger_loss:
    actual_liger_loss = str(config.get("liger_loss") or "")
    if actual_liger_loss != expected_liger_loss:
        raise SystemExit(
            f"profile liger_loss mismatch: expected {expected_liger_loss}, got {actual_liger_loss or '<missing>'}"
        )
def require_int_config_any(keys, expected, label):
    if not str(expected).strip():
        return
    wanted = int(expected)
    for key in keys:
        value = config.get(key)
        if value in (None, ""):
            continue
        try:
            actual = int(value)
        except (TypeError, ValueError):
            raise SystemExit(f"profile {key} missing or invalid")
        if actual != wanted:
            raise SystemExit(f"profile {label} mismatch: expected {wanted}, got {actual}")
        return
    raise SystemExit(f"profile {label} missing or invalid")
require_int_config_any(("per_device_train_batch_size", "batch_size"), expected_batch, "batch_size")
require_int_config_any(("gradient_accumulation_steps",), expected_grad_accum, "gradient_accumulation_steps")
if expected_offload_modules and backend in {"asym", "asym_torch", "asym_cpuadamwtorch", "asym_cpuadamwds"}:
    def normalize_selector(value):
        return ",".join(
            sorted(
                part.strip().lower().replace("-", "_")
                for part in str(value or "").split(",")
                if part.strip()
            )
        )
    actual_offload = str(config.get("asym_offload_modules") or "")
    if normalize_selector(actual_offload) != normalize_selector(expected_offload_modules):
        raise SystemExit(
            "profile asym_offload_modules mismatch: "
            f"expected {expected_offload_modules}, got {actual_offload or '<missing>'}"
        )
def normalize_bool(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return "true"
    if text in {"0", "false", "no", "n", "off"}:
        return "false"
    return ""
def recomp_off_stage(label):
    return {
        "recomp-off": "full",
        "recomp-off-base": "base",
        "recomp-off-attn": "attn",
        "recomp-off-dense": "dense",
        "recomp-off-full": "full",
        "recomp-off-dense-fg": "dense-fg",
        "recomp-off-full-fg": "full-fg",
    }.get(str(label or "").strip().lower())
expected_recomp_off_stage = recomp_off_stage(expected_recompute)
if expected_unsloth_recompute_save_on_cpu:
    actual_unsloth_off = normalize_bool(config.get("unsloth_gc_recompute_save_on_cpu"))
    wanted_unsloth_off = normalize_bool(expected_unsloth_recompute_save_on_cpu)
    if not actual_unsloth_off:
        raise SystemExit("profile unsloth_gc_recompute_save_on_cpu missing or invalid")
    if actual_unsloth_off != wanted_unsloth_off:
        raise SystemExit(
            "profile unsloth_gc_recompute_save_on_cpu mismatch: "
            f"expected {wanted_unsloth_off}, got {actual_unsloth_off}"
        )
if expected_outer_hbm_every_n != "":
    raw_outer_hbm = config.get("unsloth_gc_outer_hbm_every_n")
    try:
        # Profiles predating the knob lack the key; treat missing as 0.
        actual_outer_hbm = int(raw_outer_hbm) if raw_outer_hbm not in (None, "") else 0
    except (TypeError, ValueError):
        actual_outer_hbm = None
    if actual_outer_hbm is None or actual_outer_hbm != int(expected_outer_hbm_every_n):
        raise SystemExit(
            "profile unsloth_gc_outer_hbm_every_n mismatch: "
            f"expected {expected_outer_hbm_every_n}, got {raw_outer_hbm!r}"
        )
if expected_recomp_off_stage:
    actual_stage = str(config.get("recomp_off_stage") or "").strip()
    if actual_stage != expected_recomp_off_stage:
        raise SystemExit(
            f"profile recomp_off_stage mismatch: expected {expected_recomp_off_stage}, got {actual_stage or '<missing>'}"
        )
    actual_use_unsloth_gc = normalize_bool(config.get("use_unsloth_gc"))
    if actual_use_unsloth_gc != "true":
        raise SystemExit(f"profile use_unsloth_gc mismatch for {expected_recompute}: got {actual_use_unsloth_gc or '<missing>'}")
    actual_recompute = normalize_bool(config.get("activation_recompute"))
    if actual_recompute != "true":
        raise SystemExit(
            f"profile activation_recompute mismatch for {expected_recompute}: got {actual_recompute or '<missing>'}"
        )
    actual_silu_gpu = normalize_bool(config.get("asymm_expert_silu_bwd_gpu"))
    if actual_silu_gpu != "false":
        raise SystemExit(
            "profile asymm_expert_silu_bwd_gpu mismatch for recomp-off stage: "
            f"expected false, got {actual_silu_gpu or '<missing>'}"
        )
    expected_dense_surgical = "true" if expected_recomp_off_stage in {"dense", "full"} else "false"
    expected_dense_finegrained = "true" if expected_recomp_off_stage in {"dense-fg", "full-fg"} else "false"
    actual_dense_surgical = normalize_bool(config.get("asymm_dense_mlp_surgical_offload"))
    actual_dense_finegrained = normalize_bool(config.get("asymm_dense_mlp_finegrained_offload"))
    if actual_dense_surgical != expected_dense_surgical:
        raise SystemExit(
            "profile asymm_dense_mlp_surgical_offload mismatch: "
            f"expected {expected_dense_surgical}, got {actual_dense_surgical or '<missing>'}"
        )
    if actual_dense_finegrained != expected_dense_finegrained:
        raise SystemExit(
            "profile asymm_dense_mlp_finegrained_offload mismatch: "
            f"expected {expected_dense_finegrained}, got {actual_dense_finegrained or '<missing>'}"
        )
    qwen3_moe_target = "Qwen3-30B-A3B" in str(expected_model_name)
    expected_moefg = "true" if expected_recomp_off_stage == "full-fg" and qwen3_moe_target else "false"
    actual_moefg = normalize_bool(config.get("asymm_qwen3_moe_finegrained_offload", "false"))
    if actual_moefg != expected_moefg:
        raise SystemExit(
            "profile asymm_qwen3_moe_finegrained_offload mismatch: "
            f"expected {expected_moefg}, got {actual_moefg or '<missing>'}"
        )
    if normalize_bool(config.get("asym_offload_act_recompute")) != "false":
        raise SystemExit("profile asym_offload_act_recompute must be false for recomp-off stages")
    if normalize_bool(config.get("asym_offload_x_unpacked")) != "false":
        raise SystemExit("profile asym_offload_x_unpacked must be false for recomp-off stages")
    if str(config.get("expert_recompute_policy_spec") or config.get("expert_policy") or "").strip() not in {"", "none"}:
        raise SystemExit("profile expert recompute policy must be none for recomp-off stages")
    if str(config.get("asymm_mlp_recompute_chunk") or "0").strip() not in {"", "0"}:
        raise SystemExit("profile asymm_mlp_recompute_chunk must be 0 for recomp-off stages")
if expected_backend.endswith("_nocpuadamw"):
    offload_param = str(config.get("deepspeed_offload_param_device") or "").strip().lower()
    offload_optimizer = str(config.get("deepspeed_offload_optimizer_device") or "").strip().lower()
    super_offload = normalize_bool(config.get("deepspeed_super_offload"))
    if offload_param != "cpu":
        raise SystemExit(
            f"profile {expected_backend} must prove offload_param.device=cpu, got {offload_param or '<missing>'}"
        )
    if offload_optimizer not in {"", "none", "false"}:
        raise SystemExit(
            f"profile {expected_backend} must prove offload_optimizer disabled, got {offload_optimizer}"
        )
    if super_offload == "true":
        raise SystemExit(f"profile {expected_backend} must not enable super_offload")
if expected_profile_sync:
    actual_profile_sync = normalize_bool(config.get("profile_sync"))
    wanted_profile_sync = normalize_bool(expected_profile_sync)
    if not actual_profile_sync:
        raise SystemExit("profile profile_sync missing or invalid")
    if actual_profile_sync != wanted_profile_sync:
        raise SystemExit(
            "profile profile_sync mismatch: "
            f"expected {wanted_profile_sync}, got {actual_profile_sync}"
        )
if expected_grad_offload and backend in {"asym_cpuadamwtorch", "asym_cpuadamwds"}:
    actual_grad_offload = normalize_bool(config.get("asym_cpu_adamw_grad_offload"))
    wanted_grad_offload = normalize_bool(expected_grad_offload)
    if not actual_grad_offload:
        raise SystemExit("profile asym_cpu_adamw_grad_offload missing or invalid")
    if actual_grad_offload != wanted_grad_offload:
        raise SystemExit(
            "profile asym_cpu_adamw_grad_offload mismatch: "
            f"expected {wanted_grad_offload}, got {actual_grad_offload}"
        )
if expected_expact:
    actual_expact = normalize_bool(config.get("asymm_expert_act_offload"))
    wanted_expact = normalize_bool(expected_expact)
    if not actual_expact:
        raise SystemExit("profile asymm_expert_act_offload missing or invalid")
    if actual_expact != wanted_expact:
        raise SystemExit(
            "profile asymm_expert_act_offload mismatch: "
            f"expected {wanted_expact}, got {actual_expact}"
        )
if expected_attnact:
    actual_attnact = normalize_bool(config.get("asymm_attn_act_offload"))
    wanted_attnact = normalize_bool(expected_attnact)
    if not actual_attnact:
        raise SystemExit("profile asymm_attn_act_offload missing or invalid")
    if actual_attnact != wanted_attnact:
        raise SystemExit(
            "profile asymm_attn_act_offload mismatch: "
            f"expected {wanted_attnact}, got {actual_attnact}"
        )
if expected_layeract:
    actual_layeract = normalize_bool(config.get("asymm_layer_act_offload"))
    wanted_layeract = normalize_bool(expected_layeract)
    if not actual_layeract:
        raise SystemExit("profile asymm_layer_act_offload missing or invalid")
    if actual_layeract != wanted_layeract:
        raise SystemExit(
            "profile asymm_layer_act_offload mismatch: "
            f"expected {wanted_layeract}, got {actual_layeract}"
        )
if expected_layergc:
    wanted_layergc = normalize_bool(expected_layergc)
    actual_layergc_value = config.get("asymm_layer_gc", config.get("asym_layer_glue_gc_enabled"))
    actual_layergc = "false" if actual_layergc_value is None and wanted_layergc == "false" else normalize_bool(actual_layergc_value)
    if not actual_layergc:
        raise SystemExit("profile asymm_layer_gc missing or invalid")
    if actual_layergc != wanted_layergc:
        raise SystemExit(
            "profile asymm_layer_gc mismatch: "
            f"expected {wanted_layergc}, got {actual_layergc}"
        )
if expected_lf_expert_lora_impl:
    actual_lf_expert_lora_impl = str(config.get("qwen_moe_expert_lora_impl") or "")
    if actual_lf_expert_lora_impl != expected_lf_expert_lora_impl:
        raise SystemExit(
            "profile qwen_moe_expert_lora_impl mismatch: "
            f"expected {expected_lf_expert_lora_impl}, got {actual_lf_expert_lora_impl or '<missing>'}"
        )
def normalize_lora_a_fwd(value):
    text = str(value or "")
    if text in {"cpu", "hbm"}:
        return text
    return text
if expected_expact_lora_a_fwd:
    actual_expact_lora_a_fwd = normalize_lora_a_fwd(config.get("asymm_expert_act_offload_lora_a_fwd"))
    wanted_expact_lora_a_fwd = normalize_lora_a_fwd(expected_expact_lora_a_fwd)
    if not actual_expact_lora_a_fwd:
        raise SystemExit("profile asymm_expert_act_offload_lora_a_fwd missing or invalid")
    if actual_expact_lora_a_fwd != wanted_expact_lora_a_fwd:
        raise SystemExit(
            "profile asymm_expert_act_offload_lora_a_fwd mismatch: "
            f"expected {wanted_expact_lora_a_fwd}, got {actual_expact_lora_a_fwd}"
        )
if backend == "kt_armbf16" and expected_seq_len and expected_batch and expected_rank:
    def require_int_config(key, expected):
        try:
            actual = int(config.get(key))
        except (TypeError, ValueError):
            raise SystemExit(f"profile {key} missing or invalid")
        if actual != int(expected):
            raise SystemExit(f"profile {key} mismatch: expected {expected}, got {actual}")
    require_int_config("per_device_train_batch_size", expected_batch)
    require_int_config("lora_rank", expected_rank)
    if str(expected_cache_depth).strip():
        require_int_config("kt_max_cache_depth", expected_cache_depth)
    if expected_recompute:
        if expected_recompute in ("recomp", "unsloth", "unsloth-off"):
            wanted_recompute = "true"
        elif expected_recompute == "norecomp":
            wanted_recompute = "false"
        else:
            raise SystemExit(f"unknown expected recompute label: {expected_recompute}")
        actual_recompute = str(config.get("activation_recompute")).lower()
        if actual_recompute not in {"true", "false"}:
            raise SystemExit("profile activation_recompute missing or invalid")
        if actual_recompute != wanted_recompute:
            raise SystemExit(
                f"profile activation_recompute mismatch: expected {wanted_recompute}, got {actual_recompute}"
            )
    try:
        actual_dropout = float(config.get("lora_dropout"))
        wanted_dropout = float(expected_dropout)
    except (TypeError, ValueError):
        raise SystemExit("profile lora_dropout missing or invalid")
    if not math.isclose(actual_dropout, wanted_dropout, rel_tol=0.0, abs_tol=1e-9):
        raise SystemExit(f"profile lora_dropout mismatch: expected {wanted_dropout}, got {actual_dropout}")
    logical_qlen = int(expected_seq_len) * int(expected_batch)
    token_chunk = int(expected_token_chunk) if str(expected_token_chunk).strip() else None
    effective_route_qlen = min(logical_qlen, token_chunk) if token_chunk is not None else logical_qlen
    token_chunks = (
        (logical_qlen + token_chunk - 1) // token_chunk
        if token_chunk is not None and token_chunk < logical_qlen
        else 1
    )
    route_rank_work = effective_route_qlen * int(expected_top_k) * int(expected_rank)
    normalized_limit = None
    if str(expected_limit).strip():
        normalized_limit = int(expected_limit)
    elif str(allow_unvalidated).strip() != "1":
        normalized_limit = int(expected_default_limit)
    require_int_config("kt_arm_sft_top_k", expected_top_k)
    if token_chunk is None:
        if str(config.get("kt_arm_sft_token_chunk_size") or "").strip():
            raise SystemExit("profile token chunk size mismatch: expected unset")
    else:
        require_int_config("kt_arm_sft_token_chunk_size", token_chunk)
    require_int_config("kt_arm_effective_route_qlen", effective_route_qlen)
    require_int_config("kt_arm_token_chunks", token_chunks)
    require_int_config("kt_arm_route_rank_work", route_rank_work)
    if normalized_limit is not None:
        require_int_config("kt_arm_sft_max_route_rank_work", normalized_limit)
if backend.startswith("kt_"):
    kt = source_profile.get("kt", {})
    if not isinstance(kt, dict):
        raise SystemExit("KT profile missing kt counters")

    def int_value(container, key):
        try:
            return int(container.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0

    def optional_int(value):
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def kt_lora_health_passed(health):
        if not isinstance(health, dict) or not health:
            return False, "missing KT fused LoRA update health"
        rows = health.get("rows", [])
        row_dicts = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
        updated_from_rows = sum(
            int(
                bool(row.get("nonzero_grad_changed_after_step"))
                or (bool(row.get("grad_nonzero_before_step")) and bool(row.get("param_changed_after_step")))
            )
            for row in row_dicts
        )
        unchanged_from_rows = sum(
            int(bool(row.get("grad_nonzero_before_step")) and not bool(row.get("param_changed_after_step")))
            for row in row_dicts
        )
        updated_grad_tensors = int_value(health, "updated_grad_tensors") or updated_from_rows
        grad_nonzero_unchanged_tensors = int_value(health, "grad_nonzero_unchanged_tensors") or unchanged_from_rows
        compared_tensors = int_value(health, "compared_tensors") or len(row_dicts)
        grad_nonzero_tensors = int_value(health, "grad_nonzero_tensors")
        sampled_tensors = int_value(health, "sampled_tensors") or len(row_dicts)
        total_fused_tensors = int_value(health, "total_fused_tensors")
        after_sampled_tensors = optional_int(health.get("after_sampled_tensors"))
        after_total_fused_tensors = optional_int(health.get("after_total_fused_tensors"))
        missing_after_tensors = int_value(health, "missing_after_tensors")
        unexpected_after_tensors = int_value(health, "unexpected_after_tensors")
        exhaustive = bool(health.get("exhaustive", total_fused_tensors > 0 and sampled_tensors == total_fused_tensors))
        input_passed = health.get("passed") if "passed" in health else None
        if health.get("available") is not True:
            return False, "KT fused LoRA update health unavailable"
        missing_snapshot_fields = [
            key
            for key in (
                "after_sampled_tensors",
                "after_total_fused_tensors",
                "missing_after_tensors",
                "unexpected_after_tensors",
            )
            if key not in health or health.get(key) in (None, "")
        ]
        if missing_snapshot_fields:
            return False, "KT fused LoRA update health missing after-step snapshot fields: " + ", ".join(missing_snapshot_fields)
        if after_sampled_tensors is None or after_total_fused_tensors is None:
            return False, "KT fused LoRA update health has invalid after-step snapshot counts"
        if missing_after_tensors > 0 or unexpected_after_tensors > 0:
            return False, "sampled fused LoRA tensor set changed between before/after optimizer snapshots"
        if sampled_tensors != after_sampled_tensors or total_fused_tensors != after_total_fused_tensors:
            return False, "fused LoRA tensor counts changed between before/after optimizer snapshots"
        if exhaustive and compared_tensors != sampled_tensors:
            return False, f"exhaustive fused LoRA health compared {compared_tensors} of {sampled_tensors} tensors"
        if compared_tensors <= 0:
            return False, "no sampled fused LoRA tensors were comparable"
        if grad_nonzero_tensors <= 0:
            return False, "no sampled fused LoRA tensors had nonzero gradients before optimizer step"
        if grad_nonzero_unchanged_tensors > 0:
            return False, "one or more sampled nonzero-gradient fused LoRA tensors did not change after optimizer step"
        if updated_grad_tensors <= 0:
            return False, "no sampled nonzero-gradient fused LoRA tensors changed after optimizer step"
        if input_passed is False:
            return False, str(health.get("reason") or "input KT fused LoRA update health failed")
        return True, "all sampled nonzero-gradient fused LoRA tensors changed after optimizer step"

    def optimizer_process_memory_passed(optimizer_memory):
        if not isinstance(optimizer_memory, dict) or not optimizer_memory:
            return False, "missing optimizer_memory"
        for key in ("process_memory_at_start", "process_memory_before_step", "process_memory_after_step"):
            snapshot = optimizer_memory.get(key)
            if not isinstance(snapshot, dict) or not snapshot:
                return False, f"missing {key}"
            if optional_int(snapshot.get("rss_bytes")) is None:
                return False, f"{key}.rss_bytes missing or invalid"
        for key in ("process_rss_pre_step_overhead_delta_bytes", "process_rss_delta_bytes"):
            if key not in optimizer_memory or optional_int(optimizer_memory.get(key)) is None:
                return False, f"{key} missing or invalid"
        return True, "optimizer process RSS snapshots are present"

    if int_value(kt, "wrapper_count") <= 0:
        raise SystemExit("KT profile has no KT wrappers")
    if int_value(kt, "total_forward_calls") <= 0:
        raise SystemExit("KT profile has no KT forward calls")
    if int_value(kt, "total_backward_calls") <= 0:
        raise SystemExit("KT profile has no KT backward calls")
    if backend == "kt_armbf16":
        kt_backend = str(config.get("kt_backend") or "").upper()
        if kt_backend != "ARMBF16":
            raise SystemExit(f"KT ARM profile kt_backend is not ARMBF16: {kt_backend or '<missing>'}")
        methods = {str(row.get("method") or "") for row in kt.get("rows", []) if isinstance(row, dict)}
        if "ARMBF16_SFT" not in methods:
            raise SystemExit("KT ARM profile has no ARMBF16_SFT KT row method")
    preflight = source_profile.get("optimizer_memory_preflight", {})
    if not isinstance(preflight, dict) or preflight.get("available") is not True:
        raise SystemExit("KT profile missing optimizer_memory_preflight")
    optimizer_memory = source_profile.get("optimizer_memory", {})
    if backend == "kt_armbf16":
        memory_ok, memory_reason = optimizer_process_memory_passed(optimizer_memory)
        if not memory_ok:
            raise SystemExit(f"KT ARM profile missing optimizer process memory evidence: {memory_reason}")
    lora = source_profile.get("lora", {})
    fused_lora_params = 0
    if isinstance(lora, dict):
        try:
            fused_lora_params = int(lora.get("kt_fused_expert_lora_parameters", 0) or 0)
        except (TypeError, ValueError):
            fused_lora_params = 0
    model_name = str(config.get("model_name_or_path") or "")
    lora_target = str(config.get("lora_target") or "")
    if "qwen3" in model_name.lower() and lora_target in {"all", "all-linear", "all_linear"} and fused_lora_params <= 0:
        raise SystemExit("Qwen3 KT lora_target=all profile has no fused expert LoRA params")
    if fused_lora_params > 0:
        health = optimizer_memory.get("kt_lora_update_health", {}) if isinstance(optimizer_memory, dict) else {}
        health_ok, health_reason = kt_lora_health_passed(health)
        if not health_ok:
            raise SystemExit(f"KT fused LoRA update health missing or failed: {health_reason}")
    surface = source_profile.get("trainable_surface", {})
    if not isinstance(surface, dict) or not surface.get("surface"):
        raise SystemExit("KT profile missing trainable_surface")
PY
}

kt_arm_route_rank_limit() {
  if [[ -n "${KT_ARM_SFT_MAX_ROUTE_RANK_WORK}" ]]; then
    positive_int "KT_ARM_SFT_MAX_ROUTE_RANK_WORK" "${KT_ARM_SFT_MAX_ROUTE_RANK_WORK}"
    printf '%s\n' "${KT_ARM_SFT_MAX_ROUTE_RANK_WORK}"
    return
  fi
  if [[ "$(bool_value "${KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK}")" == "true" ]]; then
    printf '\n'
    return
  fi
  positive_int "KT_ARM_SFT_DEFAULT_MAX_ROUTE_RANK_WORK" "${KT_ARM_SFT_DEFAULT_MAX_ROUTE_RANK_WORK}"
  printf '%s\n' "${KT_ARM_SFT_DEFAULT_MAX_ROUTE_RANK_WORK}"
}

check_kt_arm_route_rank_for_sweep() {
  local seq_len="$1"
  local logical_qlen effective_route_qlen route_rank_work limit
  positive_int "KT_ARM_SFT_TOP_K" "${KT_ARM_SFT_TOP_K}"
  logical_qlen=$((PER_DEVICE_TRAIN_BATCH_SIZE * seq_len))
  effective_route_qlen="${logical_qlen}"
  if [[ -n "${KT_ARM_SFT_TOKEN_CHUNK_SIZE}" ]]; then
    positive_int "KT_ARM_SFT_TOKEN_CHUNK_SIZE" "${KT_ARM_SFT_TOKEN_CHUNK_SIZE}"
    if [[ "${KT_ARM_SFT_TOKEN_CHUNK_SIZE}" -lt "${logical_qlen}" ]]; then
      effective_route_qlen="${KT_ARM_SFT_TOKEN_CHUNK_SIZE}"
    fi
  fi
  route_rank_work=$((effective_route_qlen * KT_ARM_SFT_TOP_K * LORA_RANK))
  limit="$(kt_arm_route_rank_limit)"
  if [[ -n "${limit}" && "${route_rank_work}" -gt "${limit}" ]]; then
    die "BACKEND=kt_armbf16 route-rank work ${route_rank_work} exceeds KT_ARM_SFT_MAX_ROUTE_RANK_WORK=${limit}; reduce batch/seq/rank, set KT_ARM_SFT_TOKEN_CHUNK_SIZE, raise the explicit limit, or set KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK=1 only for validation"
  fi
}

config_root_path() {
  local seq_len="$1"
  local config_label step_label dropout_label
  step_label="w${WARMUP_STEPS}_s${MAX_STEPS}"
  dropout_label="${lora_dropout_label_value}"
  if [[ -n "${run_name}" ]]; then
    config_label="$(safe_label "${run_name}__b${batch_size}_s${seq_len}_ga${GRADIENT_ACCUMULATION_STEPS}_${dropout_label}")"
  else
    local lora_target_frag=""
    [[ "${LORA_TARGET:-all}" != "all" ]] && lora_target_frag="_tgt${LORA_TARGET}"
    config_label="$(safe_label "${workload_label}__gpus${current_model_gpu_count}__b${batch_size}_s${seq_len}_ga${GRADIENT_ACCUMULATION_STEPS}_${step_label}_r${LORA_RANK}_a${LORA_ALPHA}_${dropout_label}${lora_target_frag}")"
  fi
  printf '%s/%s\n' "${precision_root}" "${config_label}"
}

job_root_path() {
  local config_root="$1"
  local backend="$2"
  local profiler="$3"
  local recompute="$4"
  local expert_policy="$5"
  local router_mode="$6"
  local liger_loss="${7:-ligerloss0}"
  local grad_offload="${8:-false}"
  local weight_offload="${9:-false}"
  local grad_offload_suffix=""
  local outer_hbm_suffix=""
  local flash_attn_suffix=""
  if [[ "${flashattn_label:-attnauto}" != "attnauto" ]]; then
    flash_attn_suffix="__${flashattn_label}"
  fi
  local path_label="${backend}__${profiler}__${recompute}__pol${expert_policy}__router${router_mode}__${expact_label}__${attnact_label}__${layeract_label}__${layergc_label}__${sdparecomp_label}${flash_attn_suffix}"
  if cpuadam_backend_for_label "${backend}" >/dev/null; then
    grad_offload_suffix="__gradoff${grad_offload}__weightoff${weight_offload}"
  fi
  if [[ "${UNSLOTH_GC_OUTER_HBM_EVERY_N:-0}" != "0" ]]; then
    outer_hbm_suffix="__ohbm${UNSLOTH_GC_OUTER_HBM_EVERY_N}"
  fi
  path_label="${path_label}__${expact_lora_a_fwd_label}__${actrecomp_label}__${xunpack_label}__${moefg_label}__${dscatter_label}__${q3rt_label}__${liger_loss}${grad_offload_suffix}${outer_hbm_suffix}"
  printf '%s/%s\n' "${config_root}" "$(safe_label "${path_label}")"
}

workload_run_dir_name() {
  local seq_len="$1"
  printf 'b%s_s%s_ga%s\n' "${PER_DEVICE_TRAIN_BATCH_SIZE}" "${seq_len}" "${GRADIENT_ACCUMULATION_STEPS}"
}

current_workload_tuple() {
  local seq_len="$1"
  printf '%s|%s|%s\n' "${seq_len}" "${PER_DEVICE_TRAIN_BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}"
}

kt_arm_matching_source_profile_complete() {
  local config_root="$1"
  local backend="$2"
  local recompute="$3"
  local expert_policy="$4"
  local router_mode="$5"
  local liger_loss="$6"
  local seq_len="$7"
  local model_name="$8"
  kt_arm_resolve_matching_source_profile_json "${config_root}" "${backend}" "${recompute}" "${expert_policy}" "${router_mode}" "${liger_loss}" "${seq_len}" "${model_name}" >/dev/null
}

kt_arm_resolve_matching_source_profile_json() {
  local config_root="$1"
  local backend="$2"
  local recompute="$3"
  local expert_policy="$4"
  local router_mode="$5"
  local liger_loss="$6"
  local seq_len="$7"
  local model_name="$8"
  # Canonicalize the policy axes so the matched source path and expected values agree with run_job.
  local source_profile_json
  local expact_label="${expact_label}" attnact_label="${attnact_label}" layeract_label="${layeract_label}" layergc_label="${layergc_label}" sdparecomp_label="${sdparecomp_label}" expact_lora_a_fwd_label="${expact_lora_a_fwd_label}" actrecomp_label="${actrecomp_label}" xunpack_label="${xunpack_label}" moefg_label="${moefg_label}" dscatter_label="${dscatter_label}"
  local ASYMM_EXPERT_ACT_OFFLOAD="${ASYMM_EXPERT_ACT_OFFLOAD}" ASYMM_ATTN_ACT_OFFLOAD="${ASYMM_ATTN_ACT_OFFLOAD}" ASYMM_LAYER_ACT_OFFLOAD="${ASYMM_LAYER_ACT_OFFLOAD}" ASYMM_LAYER_GC="${ASYMM_LAYER_GC}" ASYMM_ATTN_SDPA_RECOMPUTE="${ASYMM_ATTN_SDPA_RECOMPUTE}" ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD="${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}"
  local ASYM_OFFLOAD_ACT_RECOMPUTE="${ASYM_OFFLOAD_ACT_RECOMPUTE}" ASYM_OFFLOAD_X_UNPACKED="${ASYM_OFFLOAD_X_UNPACKED}"
  canonicalize_policy_axis_for_inert_run "${backend}" "${recompute}"
  while IFS= read -r source_profile_json; do
    existing_profile_complete "${source_profile_json}" "${backend}" "${seq_len}" "${model_name}" "all" "${recompute}" "${ASYM_OFFLOAD_MODULES}" "${ASYMM_EXPERT_ACT_OFFLOAD}" "${ASYMM_ATTN_ACT_OFFLOAD}" "${ASYMM_LAYER_ACT_OFFLOAD}" "${ASYMM_LAYER_GC}" "${lf_expert_lora_impl:-split-target-parameters}" "${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}" "" "${liger_loss}" || continue
    printf '%s\n' "${source_profile_json}"
    return 0
  done < <(kt_arm_matching_source_profile_json_candidates "${config_root}" "${backend}" "${recompute}" "${expert_policy}" "${router_mode}" "${liger_loss}" "${seq_len}")
  return 1
}

kt_arm_matching_source_profile_json_candidates() {
  local config_root="$1"
  local backend="$2"
  local recompute="$3"
  local expert_policy="$4"
  local router_mode="$5"
  local liger_loss="$6"
  local seq_len="$7"
  local source_job_root run_dir_name legacy_path_label
  source_job_root="$(job_root_path "${config_root}" "${backend}" "source" "${recompute}" "${expert_policy}" "${router_mode}" "${liger_loss}")"
  run_dir_name="$(workload_run_dir_name "${seq_len}")"
  printf '%s/profile.json\n' "${source_job_root}/${run_dir_name}"
  if [[ "${sdparecomp_label:-}" == "sdparecomp0" ]]; then
    legacy_path_label="${backend}__source__${recompute}__pol${expert_policy}__router${router_mode}__${expact_label}__${attnact_label}__${layeract_label}__${layergc_label}"
    legacy_path_label="${legacy_path_label}__${expact_lora_a_fwd_label}__${actrecomp_label}__${xunpack_label}__${liger_loss}"
    printf '%s/profile.json\n' "${config_root}/$(safe_label "${legacy_path_label}")/${run_dir_name}"
  fi
}

plot_workload_from_config_root() {
  local config_name workload
  config_name="$(basename "$1")"
  workload="${config_name%%__*}"
  if [[ "${config_name}" =~ (^|__)gpus([1-9][0-9]*)(__|$) ]]; then
    printf '%s gpus%s\n' "${workload}" "${BASH_REMATCH[2]}"
  else
    printf '%s\n' "${workload}"
  fi
}

print_command() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

find_free_port() {
  python3 - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

ensure_jobs_tsv() {
  local config_root="$1"
  mkdir -p "${config_root}"
  if [[ ! -e "${config_root}/jobs.tsv" ]]; then
    printf 'status\tgpu\tseq_len\tbatch_size\tgradient_accumulation_steps\trecompute\texpert_policy\trouter_mode\tbackend\tprofiler\tliger_loss\tgrad_offload\tjob_dir\tprofile_json\tlog\tqwen_expert_lora_impl\texpert_lora_a_fwd\n' > "${config_root}/jobs.tsv"
  fi
}

append_job_record() {
  local config_root="$1"
  local status="$2"
  local gpu="$3"
  local seq_len="$4"
  shift 4
  ensure_jobs_tsv "${config_root}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${status}" "${gpu}" "${seq_len}" "${PER_DEVICE_TRAIN_BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}" "$@" "${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}" >> "${config_root}/jobs.tsv"
}

append_runs_log_entry() {
  local tag="$1"
  local payload="$2"
  local reason="${3:-}"
  local extra_tag="${4:-}"
  local line
  line="[$(date '+%Y-%m-%d %H:%M:%S')] [${tag}]"
  [[ -n "${extra_tag}" ]] && line+=" [${extra_tag}]"
  line+=" ${payload}"
  [[ -n "${reason}" ]] && line+=" ; ${reason}"
  echo "${line}" | tee -a "${_RUNS_LOG}"
}

plot_cmd_base() {
  local -n _cmd_ref="$1"
  local input_root="$2"
  local output_dir="$3"
  local combined_output_dir="$4"
  shift 4
  (($# > 0)) || die "plot_cmd_base requires at least one workload"
  _cmd_ref=(
    "${ENV_PYTHON}" "${PLOT_SCRIPT}"
    --input-root "${input_root}"
    --output-dir "${output_dir}"
    --combined-output-dir "${combined_output_dir}"
    --precision "${PRECISION}"
    --clean-output
  )
  _cmd_ref+=(--workloads "$@")
}

config_combined_metric_dir() {
  local config_root="$1"
  local metric="$2"
  if [[ -n "${PLOT_OUTPUT_DIR}" ]]; then
    printf '%s/%s/combined/%s\n' "$(abs_path "${PLOT_OUTPUT_DIR}")" "$(basename "${config_root}")" "${metric}"
  else
    printf '%s/combined/%s\n' "${config_root}" "${metric}"
  fi
}

precision_combined_metric_dir() {
  local metric="$1"
  if [[ -n "${PLOT_OUTPUT_DIR}" ]]; then
    printf '%s/combined/%s\n' "$(abs_path "${PLOT_OUTPUT_DIR}")" "${metric}"
  else
    printf '%s/combined/%s\n' "${precision_root}" "${metric}"
  fi
}

run_metric_dir() {
  local seq_root="$1"
  local metric="$2"
  printf '%s/metrics/%s\n' "${seq_root}" "${metric}"
}

cleanup_legacy_run_artifacts() {
  local seq_root="$1"
  rm -rf "${seq_root}/plots" "${seq_root}/memory_plots"
  rm -f \
    "${seq_root}/utilization.png" \
    "${seq_root}/utilization.csv" \
    "${seq_root}/utilization_timeseries.csv" \
    "${seq_root}/utilization_summary.csv" \
    "${seq_root}/interconnect_ctc_timeline.png" \
    "${seq_root}/interconnect_ctc_by_step.png" \
    "${seq_root}/interconnect_ctc_by_phase.png" \
    "${seq_root}/interconnect_ctc_timeseries.csv" \
    "${seq_root}/interconnect_ctc_step_summary.csv" \
    "${seq_root}/interconnect_metric_summary.csv"
}

cleanup_legacy_combined_artifacts() {
  local root="$1"
  rm -rf \
    "${root}/memory_combined" \
    "${root}/c2c_combined" \
    "${root}/utilization_combined" \
    "${root}/throughput_combined"
  if [[ -d "${root}/combined" ]]; then
    find "${root}/combined" -mindepth 1 -maxdepth 1 -type f -delete
    find "${root}/combined" -mindepth 1 -maxdepth 1 -type d \
      ! -name timing \
      ! -name memory \
      ! -name utilization \
      ! -name interconnect \
      ! -name throughput \
      -exec rm -rf {} +
  fi
}

memory_metric_plot_cmd_base() {
  local -n _cmd_ref="$1"
  local input_root="$2"
  local output_dir="$3"
  shift 3
  (($# > 0)) || die "memory_metric_plot_cmd_base requires at least one workload"
  _cmd_ref=(
    "${ENV_PYTHON}" "${MEMORY_PLOT_SCRIPT}"
    --input-root "${input_root}"
    --output-dir "${output_dir}"
    --clean-output
    --combined-only
    --y-scale "${MEMORY_BREAKDOWN_PLOT_Y_SCALE}"
    --workloads "$@"
  )
}

interconnect_metric_plot_cmd_base() {
  local -n _cmd_ref="$1"
  local input_root="$2"
  local output_dir="$3"
  shift 3
  (($# > 0)) || die "interconnect_metric_plot_cmd_base requires at least one workload"
  _cmd_ref=(
    "${ENV_PYTHON}" "${INTERCONNECT_PLOT_SCRIPT}"
    --input-root "${input_root}"
    --output-dir "${output_dir}"
    --clean-output
    --combined-only
    --workloads "$@"
  )
}

throughput_metric_plot_cmd_base() {
  local -n _cmd_ref="$1"
  local input_root="$2"
  local output_dir="$3"
  shift 3
  (($# > 0)) || die "throughput_metric_plot_cmd_base requires at least one workload"
  _cmd_ref=(
    "${ENV_PYTHON}" "${THROUGHPUT_PLOT_SCRIPT}"
    --input-root "${input_root}"
    --output-dir "${output_dir}"
    --clean-output
    --combined-only
    --workloads "$@"
  )
}

utilization_metric_plot_cmd_base() {
  local -n _cmd_ref="$1"
  local input_root="$2"
  local output_dir="$3"
  shift 3
  (($# > 0)) || die "utilization_metric_plot_cmd_base requires at least one workload"
  _cmd_ref=(
    "${ENV_PYTHON}" "${UTILIZATION_PLOT_SCRIPT}"
    --input-root "${input_root}"
    --output-dir "${output_dir}"
    --clean-output
    --combined-only
    --workloads "$@"
  )
}

append_plot_profiler_filters() {
  local -n _cmd_ref="$1"
  local profiler
  for profiler in "${plot_profilers[@]}"; do _cmd_ref+=(--profiler "${profiler}"); done
}

append_fixed_profiler_filter() {
  local -n _cmd_ref="$1"
  local profiler="$2"
  _cmd_ref+=(--profiler "${profiler}")
}

# Combined plots include every config present on disk; only the profiler dedup filter is applied.
append_sweep_plot_filters() {
  append_plot_profiler_filters "$1"
}

# Memory combined: source profiler only (dedups the nsys/source pair); no config narrowing.
memory_plot_filters() {
  append_fixed_profiler_filter "$1" source
}

# C2C combined: nsys profiler only (C2C metrics exist only in nsys runs); no config narrowing.
interconnect_plot_filters() {
  append_fixed_profiler_filter "$1" nsys
}

# Utilization combined: nsys profiler only so GPU and CPU are available together.
utilization_plot_filters() {
  append_fixed_profiler_filter "$1" nsys
}

# Throughput combined: source profiler only (per-step timing without nsys overhead); no config narrowing.
throughput_plot_filters() {
  append_fixed_profiler_filter "$1" source
}

profiler_selected_for_plots() {
  local profiler="$1"
  local selected
  for selected in "${plot_profilers[@]}"; do
    [[ "${selected}" == "${profiler}" ]] && return 0
  done
  return 1
}

dataset_name_for_seq() {
  local seq_len="$1"
  safe_label "${DATASET}__${workload_label}__s${seq_len}"
}

dataset_min_tokens_for_seq() {
  local seq_len="$1"
  if [[ "${DATASET_MIN_TOKENS}" == "auto" ]]; then
    printf '%s\n' "${seq_len}"
  else
    printf '%s\n' "${DATASET_MIN_TOKENS}"
  fi
}

prepare_dataset_for_seq() {
  local seq_len="$1"
  local dataset_name="$2"
  local min_tokens
  local dataset_lf_dir="${CURRENT_LF_DIR:-${LF_DIR}}"
  local dataset_env_python="${CURRENT_ENV_PYTHON:-${ENV_PYTHON}}"
  min_tokens="$(dataset_min_tokens_for_seq "${seq_len}")"

  local -a tools_dir_arg=(--asym-dir "${ASYM_DIR}")
  if [[ "${selected_has_kt:-false}" == "true" && "${selected_has_asym:-false}" != "true" ]]; then
    tools_dir_arg=(--asym-dir "${KT_TOOLS_DIR}")
  fi

  local -a dataset_cmd=(
    "${dataset_env_python}" "${BUILD_DATASET_SCRIPT}"
    --lf-dir "${dataset_lf_dir}"
    "${tools_dir_arg[@]}"
    --model-name-or-path "${current_model_name}"
    --template "${TEMPLATE}"
    --train-name "${dataset_name}"
    --eval-name "${dataset_name}__eval"
    --train-rows "${MAX_SAMPLES}"
    --eval-rows "${DATASET_EVAL_ROWS}"
    --cutoff-len "${seq_len}"
    --min-tokens "${min_tokens}"
    --precision "${PRECISION}"
    --batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --lora-rank "${LORA_RANK}"
    --lora-alpha "${LORA_ALPHA}"
    --skip-lf-preprocess-check
  )
  if [[ "${DATASET_OVERWRITE}" == "true" ]]; then
    dataset_cmd+=(--overwrite)
  fi

  echo "Preparing LF dataset=${dataset_name} model=${current_model_name} seq=${seq_len} min_tokens=${min_tokens} lf_dir=${dataset_lf_dir}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    print_command "${dataset_cmd[@]}"
    return 0
  fi
  run_tracked_command "${dataset_cmd[@]}"
}

gpu_spec="${GPU_POOL}"
if [[ "${_RUNS_ENV_SET}" == "true" ]]; then
  model_spec="${RUN_MODEL_SPECS}"
  backend_specs_spec="${RUN_BACKENDS}"
  workload_spec="${RUN_WORKLOADS}"
  exp_act_policy_spec="${RUN_EXP_ACT_POLICIES}"
  axis_run_override=false
else
  model_spec="${MODEL_SPECS:-${RUN_MODEL_SPECS}}"
  backend_specs_spec="${BACKEND_SPECS:-${RUN_BACKENDS}}"
  workload_spec="${WORKLOADS:-${RUN_WORKLOADS}}"
  exp_act_policy_spec="${ASYMM_EXP_ACT_POLICIES:-${RUN_EXP_ACT_POLICIES}}"
  axis_run_override=false
  if [[ -n "${MODEL_SPECS+x}" || -n "${BACKEND_SPECS+x}" || -n "${WORKLOADS+x}" || -n "${ASYMM_EXP_ACT_POLICIES+x}" ]]; then
    axis_run_override=true
  fi
fi
router_mode_spec="${ROUTER_MODES}"
profiler_spec="${PROFILERS}"
lora_params_spec="${LORA_PARAMS}"
lf_expert_lora_impl_spec="${LF_EXPERT_LORA_IMPLS}"
output_root="${OUTPUT_ROOT}"
run_name="${RUN_NAME}"
template_spec="${TEMPLATE}"
kt_repo_dir_user_set=false
[[ -n "${KT_REPO_DIR_ENV_SET}" ]] && kt_repo_dir_user_set=true
kt_gguf_py_dir_user_set=false
[[ -n "${KT_GGUF_PY_DIR_ENV_SET}" ]] && kt_gguf_py_dir_user_set=true

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --gpus) need_value "$1" "${2-}"; gpu_spec="$2"; shift 2 ;;
    --gpus=*) gpu_spec="${1#*=}"; shift ;;
    --dist-launcher) need_value "$1" "${2-}"; DIST_LAUNCHER="$(dist_launcher_label "$2")"; shift 2 ;;
    --dist-launcher=*) DIST_LAUNCHER="$(dist_launcher_label "${1#*=}")"; shift ;;
    --models|--model-specs) collect_values "$1" vals "${@:2}"; model_spec="${vals[*]}"; axis_run_override=true; set -- "${REMAINING[@]}" ;;
    --models=*|--model-specs=*) model_spec="${1#*=}"; axis_run_override=true; shift ;;
    --backend-specs) collect_values "$1" vals "${@:2}"; backend_specs_spec="${vals[*]}"; axis_run_override=true; set -- "${REMAINING[@]}" ;;
    --backend-specs=*) backend_specs_spec="${1#*=}"; axis_run_override=true; shift ;;
    --router-modes) need_value "$1" "${2-}"; router_mode_spec="$2"; ROUTER_MODES="$2"; shift 2 ;;
    --router-modes=*) router_mode_spec="${1#*=}"; ROUTER_MODES="${1#*=}"; shift ;;
    --profilers) need_value "$1" "${2-}"; profiler_spec="$2"; shift 2 ;;
    --profilers=*) profiler_spec="${1#*=}"; shift ;;
    --workloads) collect_values "$1" vals "${@:2}"; workload_spec="${vals[*]}"; WORKLOADS="${workload_spec}"; axis_run_override=true; set -- "${REMAINING[@]}" ;;
    --workloads=*) workload_spec="${1#*=}"; WORKLOADS="${workload_spec}"; axis_run_override=true; shift ;;
    --asymm-exp-act-policies) need_value "$1" "${2-}"; exp_act_policy_spec="$2"; ASYMM_EXP_ACT_POLICIES="$2"; axis_run_override=true; shift 2 ;;
    --asymm-exp-act-policies=*) exp_act_policy_spec="${1#*=}"; ASYMM_EXP_ACT_POLICIES="${1#*=}"; axis_run_override=true; shift ;;
    --dataset) need_value "$1" "${2-}"; DATASET="$2"; shift 2 ;;
    --dataset=*) DATASET="${1#*=}"; shift ;;
    --prepare-datasets) need_value "$1" "${2-}"; PREPARE_DATASETS="$(bool_value "$2")"; shift 2 ;;
    --prepare-datasets=*) PREPARE_DATASETS="$(bool_value "${1#*=}")"; shift ;;
    --dataset-min-tokens) need_value "$1" "${2-}"; DATASET_MIN_TOKENS="$2"; shift 2 ;;
    --dataset-min-tokens=*) DATASET_MIN_TOKENS="${1#*=}"; shift ;;
    --dataset-eval-rows) need_value "$1" "${2-}"; DATASET_EVAL_ROWS="$2"; shift 2 ;;
    --dataset-eval-rows=*) DATASET_EVAL_ROWS="${1#*=}"; shift ;;
    --dataset-overwrite) need_value "$1" "${2-}"; DATASET_OVERWRITE="$(bool_value "$2")"; shift 2 ;;
    --dataset-overwrite=*) DATASET_OVERWRITE="$(bool_value "${1#*=}")"; shift ;;
    --template) need_value "$1" "${2-}"; template_spec="$2"; TEMPLATE="$2"; shift 2 ;;
    --template=*) template_spec="${1#*=}"; TEMPLATE="${1#*=}"; shift ;;
    --max-samples) need_value "$1" "${2-}"; MAX_SAMPLES="$2"; shift 2 ;;
    --max-samples=*) MAX_SAMPLES="${1#*=}"; shift ;;
    --max-steps) need_value "$1" "${2-}"; MAX_STEPS="$2"; shift 2 ;;
    --max-steps=*) MAX_STEPS="${1#*=}"; shift ;;
    --warmup-steps) need_value "$1" "${2-}"; WARMUP_STEPS="$2"; shift 2 ;;
    --warmup-steps=*) WARMUP_STEPS="${1#*=}"; shift ;;
    --learning-rate) need_value "$1" "${2-}"; LEARNING_RATE="$2"; shift 2 ;;
    --learning-rate=*) LEARNING_RATE="${1#*=}"; shift ;;
    --seed) need_value "$1" "${2-}"; SEED="$2"; shift 2 ;;
    --seed=*) SEED="${1#*=}"; shift ;;
    --lora-rank) need_value "$1" "${2-}"; LORA_RANK="$2"; [[ "${_LORA_PARAMS_ENV_SET}" == "true" ]] || { LORA_PARAMS="${LORA_DROPOUT}|${LORA_RANK}|${LORA_ALPHA}|all"; lora_params_spec="${LORA_PARAMS}"; }; shift 2 ;;
    --lora-rank=*) LORA_RANK="${1#*=}"; [[ "${_LORA_PARAMS_ENV_SET}" == "true" ]] || { LORA_PARAMS="${LORA_DROPOUT}|${LORA_RANK}|${LORA_ALPHA}|all"; lora_params_spec="${LORA_PARAMS}"; }; shift ;;
    --lora-alpha) need_value "$1" "${2-}"; LORA_ALPHA="$2"; [[ "${_LORA_PARAMS_ENV_SET}" == "true" ]] || { LORA_PARAMS="${LORA_DROPOUT}|${LORA_RANK}|${LORA_ALPHA}|all"; lora_params_spec="${LORA_PARAMS}"; }; shift 2 ;;
    --lora-alpha=*) LORA_ALPHA="${1#*=}"; [[ "${_LORA_PARAMS_ENV_SET}" == "true" ]] || { LORA_PARAMS="${LORA_DROPOUT}|${LORA_RANK}|${LORA_ALPHA}|all"; lora_params_spec="${LORA_PARAMS}"; }; shift ;;
    --lora-dropout) collect_values "$1" vals "${@:2}"; LORA_DROPOUT="${vals[*]}"; [[ "${_LORA_PARAMS_ENV_SET}" == "true" ]] || { LORA_PARAMS="${LORA_DROPOUT}|${LORA_RANK}|${LORA_ALPHA}|all"; lora_params_spec="${LORA_PARAMS}"; }; set -- "${REMAINING[@]}" ;;
    --lora-dropout=*) LORA_DROPOUT="${1#*=}"; [[ "${_LORA_PARAMS_ENV_SET}" == "true" ]] || { LORA_PARAMS="${LORA_DROPOUT}|${LORA_RANK}|${LORA_ALPHA}|all"; lora_params_spec="${LORA_PARAMS}"; }; shift ;;
    --lora-params) collect_values "$1" vals "${@:2}"; lora_params_spec="${vals[*]}"; LORA_PARAMS="${lora_params_spec}"; set -- "${REMAINING[@]}" ;;
    --lora-params=*) lora_params_spec="${1#*=}"; LORA_PARAMS="${lora_params_spec}"; shift ;;
    --lf-expert-lora-impls) need_value "$1" "${2-}"; lf_expert_lora_impl_spec="$2"; LF_EXPERT_LORA_IMPLS="$2"; shift 2 ;;
    --lf-expert-lora-impls=*) lf_expert_lora_impl_spec="${1#*=}"; LF_EXPERT_LORA_IMPLS="${1#*=}"; shift ;;
    --precision) need_value "$1" "${2-}"; PRECISION="$2"; shift 2 ;;
    --precision=*) PRECISION="${1#*=}"; shift ;;
    --profile-level) need_value "$1" "${2-}"; PROFILE_LEVEL="$2"; shift 2 ;;
    --profile-level=*) PROFILE_LEVEL="${1#*=}"; shift ;;
    --profile-layers) need_value "$1" "${2-}"; PROFILE_LAYERS="$2"; shift 2 ;;
    --profile-layers=*) PROFILE_LAYERS="${1#*=}"; shift ;;
    --profile-memory-breakdown) need_value "$1" "${2-}"; PROFILE_MEMORY_BREAKDOWN="$2"; shift 2 ;;
    --profile-memory-breakdown=*) PROFILE_MEMORY_BREAKDOWN="${1#*=}"; shift ;;
    --profile-memory-breakdown-interval) need_value "$1" "${2-}"; PROFILE_MEMORY_BREAKDOWN_INTERVAL="$2"; shift 2 ;;
    --profile-memory-breakdown-interval=*) PROFILE_MEMORY_BREAKDOWN_INTERVAL="${1#*=}"; shift ;;
    --profile-memory-breakdown-steps) need_value "$1" "${2-}"; PROFILE_MEMORY_BREAKDOWN_STEPS="$2"; shift 2 ;;
    --profile-memory-breakdown-steps=*) PROFILE_MEMORY_BREAKDOWN_STEPS="${1#*=}"; shift ;;
    --profile-memory-breakdown-modules) need_value "$1" "${2-}"; PROFILE_MEMORY_BREAKDOWN_MODULES="$2"; shift 2 ;;
    --profile-memory-breakdown-modules=*) PROFILE_MEMORY_BREAKDOWN_MODULES="${1#*=}"; shift ;;
    --profile-memory-snapshot) need_value "$1" "${2-}"; PROFILE_MEMORY_SNAPSHOT="$2"; shift 2 ;;
    --profile-memory-snapshot=*) PROFILE_MEMORY_SNAPSHOT="${1#*=}"; shift ;;
    --profile-memory-snapshot-path) need_value "$1" "${2-}"; PROFILE_MEMORY_SNAPSHOT_PATH="$2"; shift 2 ;;
    --profile-memory-snapshot-path=*) PROFILE_MEMORY_SNAPSHOT_PATH="${1#*=}"; shift ;;
    --profile-external-memory) need_value "$1" "${2-}"; PROFILE_EXTERNAL_MEMORY="$2"; shift 2 ;;
    --profile-external-memory=*) PROFILE_EXTERNAL_MEMORY="${1#*=}"; shift ;;
    --profile-sync) need_value "$1" "${2-}"; PROFILE_SYNC="$2"; shift 2 ;;
    --profile-sync=*) PROFILE_SYNC="${1#*=}"; shift ;;
    --profile-module-filter) need_value "$1" "${2-}"; PROFILE_MODULE_FILTER="$2"; shift 2 ;;
    --profile-module-filter=*) PROFILE_MODULE_FILTER="${1#*=}"; shift ;;
    --expandable-seg|--expandable-segments) need_value "$1" "${2-}"; EXPANDABLE_SEG="$(bool_value "$2")"; shift 2 ;;
    --expandable-seg=*|--expandable-segments=*) EXPANDABLE_SEG="$(bool_value "${1#*=}")"; shift ;;
    --use-asym-cpu-adamw) need_value "$1" "${2-}"; USE_ASYM_CPU_ADAMW="$(bool_value "$2")"; shift 2 ;;
    --use-asym-cpu-adamw=*) USE_ASYM_CPU_ADAMW="$(bool_value "${1#*=}")"; shift ;;
    --asym-cpu-adamw-backend) need_value "$1" "${2-}"; ASYM_CPU_ADAMW_BACKEND="$2"; shift 2 ;;
    --asym-cpu-adamw-backend=*) ASYM_CPU_ADAMW_BACKEND="${1#*=}"; shift ;;
    --asym-cpu-adamw-pin-memory) need_value "$1" "${2-}"; ASYM_CPU_ADAMW_PIN_MEMORY="$(bool_value "$2")"; shift 2 ;;
    --asym-cpu-adamw-pin-memory=*) ASYM_CPU_ADAMW_PIN_MEMORY="$(bool_value "${1#*=}")"; shift ;;
    --asym-cpu-adamw-fp32-master) need_value "$1" "${2-}"; ASYM_CPU_ADAMW_FP32_MASTER="$(bool_value "$2")"; shift 2 ;;
    --asym-cpu-adamw-fp32-master=*) ASYM_CPU_ADAMW_FP32_MASTER="$(bool_value "${1#*=}")"; shift ;;
    --asym-cpu-adamw-grad-offload) need_value "$1" "${2-}"; ASYM_CPU_ADAMW_GRAD_OFFLOAD="$2"; shift 2 ;;
    --asym-cpu-adamw-grad-offload=*) ASYM_CPU_ADAMW_GRAD_OFFLOAD="${1#*=}"; shift ;;
    --asym-cpu-adamw-weight-offload) need_value "$1" "${2-}"; ASYM_CPU_ADAMW_WEIGHT_OFFLOAD="$2"; shift 2 ;;
    --asym-cpu-adamw-weight-offload=*) ASYM_CPU_ADAMW_WEIGHT_OFFLOAD="${1#*=}"; shift ;;
    --kt-kernel-dir) need_value "$1" "${2-}"; KT_KERNEL_DIR="$2"; shift 2 ;;
    --kt-kernel-dir=*) KT_KERNEL_DIR="${1#*=}"; shift ;;
    --kt-tools-dir) need_value "$1" "${2-}"; KT_TOOLS_DIR="$2"; shift 2 ;;
    --kt-tools-dir=*) KT_TOOLS_DIR="${1#*=}"; shift ;;
    --kt-repo-dir) need_value "$1" "${2-}"; KT_REPO_DIR="$2"; kt_repo_dir_user_set=true; shift 2 ;;
    --kt-repo-dir=*) KT_REPO_DIR="${1#*=}"; kt_repo_dir_user_set=true; shift ;;
    --kt-gguf-py-dir) need_value "$1" "${2-}"; KT_GGUF_PY_DIR="$2"; kt_gguf_py_dir_user_set=true; shift 2 ;;
    --kt-gguf-py-dir=*) KT_GGUF_PY_DIR="${1#*=}"; kt_gguf_py_dir_user_set=true; shift ;;
    --kt-num-threads) need_value "$1" "${2-}"; KT_NUM_THREADS="$2"; shift 2 ;;
    --kt-num-threads=*) KT_NUM_THREADS="${1#*=}"; shift ;;
    --kt-threadpool-count) need_value "$1" "${2-}"; KT_THREADPOOL_COUNT="$2"; shift 2 ;;
    --kt-threadpool-count=*) KT_THREADPOOL_COUNT="${1#*=}"; shift ;;
    --kt-max-cache-depth) need_value "$1" "${2-}"; KT_MAX_CACHE_DEPTH="$2"; shift 2 ;;
    --kt-max-cache-depth=*) KT_MAX_CACHE_DEPTH="${1#*=}"; shift ;;
    --kt-tp-enabled) need_value "$1" "${2-}"; KT_TP_ENABLED="$(bool_value "$2")"; shift 2 ;;
    --kt-tp-enabled=*) KT_TP_ENABLED="$(bool_value "${1#*=}")"; shift ;;
    --kt-torchbf16-sft-device) need_value "$1" "${2-}"; KT_TORCHBF16_SFT_DEVICE="$2"; shift 2 ;;
    --kt-torchbf16-sft-device=*) KT_TORCHBF16_SFT_DEVICE="${1#*=}"; shift ;;
    --kt-arm-omp-num-threads) need_value "$1" "${2-}"; KT_ARM_OMP_NUM_THREADS="$2"; shift 2 ;;
    --kt-arm-omp-num-threads=*) KT_ARM_OMP_NUM_THREADS="${1#*=}"; shift ;;
    --kt-arm-omp-proc-bind) need_value "$1" "${2-}"; KT_ARM_OMP_PROC_BIND="$2"; shift 2 ;;
    --kt-arm-omp-proc-bind=*) KT_ARM_OMP_PROC_BIND="${1#*=}"; shift ;;
    --kt-arm-omp-places) need_value "$1" "${2-}"; KT_ARM_OMP_PLACES="$2"; shift 2 ;;
    --kt-arm-omp-places=*) KT_ARM_OMP_PLACES="${1#*=}"; shift ;;
    --kt-arm-sft-token-chunk-size) need_value "$1" "${2-}"; KT_ARM_SFT_TOKEN_CHUNK_SIZE="$2"; shift 2 ;;
    --kt-arm-sft-token-chunk-size=*) KT_ARM_SFT_TOKEN_CHUNK_SIZE="${1#*=}"; shift ;;
    --kt-share-backward-bb) need_value "$1" "${2-}"; KT_SHARE_BACKWARD_BB="$(bool_value "$2")"; shift 2 ;;
    --kt-share-backward-bb=*) KT_SHARE_BACKWARD_BB="$(bool_value "${1#*=}")"; shift ;;
    --kt-num-gpu-experts) need_value "$1" "${2-}"; KT_NUM_GPU_EXPERTS="$2"; shift 2 ;;
    --kt-num-gpu-experts=*) KT_NUM_GPU_EXPERTS="${1#*=}"; shift ;;
    --kt-weight-path) need_value "$1" "${2-}"; KT_WEIGHT_PATH="$2"; shift 2 ;;
    --kt-weight-path=*) KT_WEIGHT_PATH="${1#*=}"; shift ;;
    --kt-expert-checkpoint-path) need_value "$1" "${2-}"; KT_EXPERT_CHECKPOINT_PATH="$2"; shift 2 ;;
    --kt-expert-checkpoint-path=*) KT_EXPERT_CHECKPOINT_PATH="${1#*=}"; shift ;;
    --kt-use-lora-experts) need_value "$1" "${2-}"; KT_USE_LORA_EXPERTS="$(bool_value "$2")"; shift 2 ;;
    --kt-use-lora-experts=*) KT_USE_LORA_EXPERTS="$(bool_value "${1#*=}")"; shift ;;
    --kt-lora-expert-num) need_value "$1" "${2-}"; KT_LORA_EXPERT_NUM="$2"; shift 2 ;;
    --kt-lora-expert-num=*) KT_LORA_EXPERT_NUM="${1#*=}"; shift ;;
    --kt-lora-expert-intermediate-size) need_value "$1" "${2-}"; KT_LORA_EXPERT_INTERMEDIATE_SIZE="$2"; shift 2 ;;
    --kt-lora-expert-intermediate-size=*) KT_LORA_EXPERT_INTERMEDIATE_SIZE="${1#*=}"; shift ;;
    --check-kt-calls) need_value "$1" "${2-}"; CHECK_KT_CALLS="$(bool_value "$2")"; shift 2 ;;
    --check-kt-calls=*) CHECK_KT_CALLS="$(bool_value "${1#*=}")"; shift ;;
    --deepspeed-dir) need_value "$1" "${2-}"; DEEPSPEED_DIR="$2"; shift 2 ;;
    --deepspeed-dir=*) DEEPSPEED_DIR="${1#*=}"; shift ;;
    --check-superoffload) need_value "$1" "${2-}"; CHECK_SUPEROFFLOAD="$(bool_value "$2")"; shift 2 ;;
    --check-superoffload=*) CHECK_SUPEROFFLOAD="$(bool_value "${1#*=}")"; shift ;;
    --check-cpuadam) need_value "$1" "${2-}"; CHECK_CPUADAM="$(bool_value "$2")"; shift 2 ;;
    --check-cpuadam=*) CHECK_CPUADAM="$(bool_value "${1#*=}")"; shift ;;
    --output-root) need_value "$1" "${2-}"; output_root="$2"; shift 2 ;;
    --output-root=*) output_root="${1#*=}"; shift ;;
    --run-name) need_value "$1" "${2-}"; run_name="$2"; shift 2 ;;
    --run-name=*) run_name="${1#*=}"; shift ;;
    --plot) need_value "$1" "${2-}"; PLOT="$(bool_value "$2")"; shift 2 ;;
    --plot=*) PLOT="$(bool_value "${1#*=}")"; shift ;;
    --plot-memory-breakdown) need_value "$1" "${2-}"; PLOT_MEMORY_BREAKDOWN="$(bool_value "$2")"; shift 2 ;;
    --plot-memory-breakdown=*) PLOT_MEMORY_BREAKDOWN="$(bool_value "${1#*=}")"; shift ;;
    --memory-breakdown-plot-y-scale) need_value "$1" "${2-}"; MEMORY_BREAKDOWN_PLOT_Y_SCALE="$2"; shift 2 ;;
    --memory-breakdown-plot-y-scale=*) MEMORY_BREAKDOWN_PLOT_Y_SCALE="${1#*=}"; shift ;;
    --plot-output-dir) need_value "$1" "${2-}"; PLOT_OUTPUT_DIR="$2"; shift 2 ;;
    --plot-output-dir=*) PLOT_OUTPUT_DIR="${1#*=}"; shift ;;
    --run-post) need_value "$1" "${2-}"; RUN_POST="$(bool_value "$2")"; shift 2 ;;
    --run-post=*) RUN_POST="$(bool_value "${1#*=}")"; shift ;;
    --overwrite) need_value "$1" "${2-}"; OVERWRITE="$(bool_value "$2")"; shift 2 ;;
    --overwrite=*) OVERWRITE="$(bool_value "${1#*=}")"; shift ;;
    --continue-on-error) need_value "$1" "${2-}"; CONTINUE_ON_ERROR="$(bool_value "$2")"; shift 2 ;;
    --continue-on-error=*) CONTINUE_ON_ERROR="$(bool_value "${1#*=}")"; shift ;;
    --collect-existing) COLLECT_EXISTING=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) die "unknown option: $1" ;;
  esac
done

DIST_LAUNCHER="$(dist_launcher_label "${DIST_LAUNCHER}")"

if [[ "${_RUNS_ENV_SET}" == "true" && "${axis_run_override}" == "true" ]]; then
  die "RUNS is an explicit schedule; do not combine it with --models/--backend-specs/--workloads/--asymm-exp-act-policies"
fi

if [[ "${_RUNS_ENV_SET}" != "true" && "${axis_run_override}" == "true" ]]; then
  _run_models=(); _run_backends=(); _run_workloads=(); _run_policies=(); _run_specs=()
  mapfile -t _axis_models < <(tokens "${model_spec}" | dedupe)
  mapfile -t _axis_backend_tokens < <(tokens "${backend_specs_spec}" | dedupe)
  mapfile -t _axis_workloads < <(tokens "${workload_spec}" | dedupe)
  mapfile -t _axis_policies < <(tokens "${exp_act_policy_spec}" | dedupe)
  ((${#_axis_models[@]})) || die "model spec list is empty"
  ((${#_axis_backend_tokens[@]})) || die "backend spec list is empty"
  ((${#_axis_workloads[@]})) || die "workload list is empty"
  ((${#_axis_policies[@]})) || die "expert/attention activation policy tuple list is empty"
  for _axis_model in "${_axis_models[@]}"; do
    for _axis_backend_token in "${_axis_backend_tokens[@]}"; do
      mapfile -t _axis_backend_entries < <(backend_entries_for_run_spec "${_axis_backend_token}")
      for _axis_backend in "${_axis_backend_entries[@]}"; do
        for _axis_workload in "${_axis_workloads[@]}"; do
          for _axis_policy in "${_axis_policies[@]}"; do
            _run_models+=("${_axis_model}")
            _run_backends+=("${_axis_backend}")
            _run_workloads+=("${_axis_workload}")
            _run_policies+=("${_axis_policy}")
            _run_specs+=("${_axis_model};${_axis_backend};${_axis_workload};${_axis_policy}")
          done
        done
      done
    done
  done
fi

require_comma_list "--gpus/GPU_POOL" "${gpu_spec}"
require_comma_list "--models/MODEL_SPECS or RUNS-derived model list" "${model_spec}"
require_comma_list "--backend-specs/BACKEND_SPECS or RUNS-derived backend list" "${backend_specs_spec}"
require_comma_list "--router-modes/ROUTER_MODES" "${router_mode_spec}"
require_comma_list "--profilers/PROFILERS" "${profiler_spec}"
require_comma_list "--workloads/WORKLOADS or RUNS-derived workload list" "${workload_spec}"
require_comma_list "--asymm-exp-act-policies/ASYMM_EXP_ACT_POLICIES or RUNS-derived policy list" "${exp_act_policy_spec}"
require_comma_list "--lora-params/LORA_PARAMS" "${lora_params_spec}"
require_comma_list "--lf-expert-lora-impls/LF_EXPERT_LORA_IMPLS" "${lf_expert_lora_impl_spec}"

nonnegative_int "--max-steps" "${MAX_STEPS}"
nonnegative_int "--warmup-steps" "${WARMUP_STEPS}"
if ((WARMUP_STEPS < 5)) && [[ "$(bool_value "${DRY_RUN}")" != "true" && "$(bool_value "${COLLECT_EXISTING}")" != "true" ]]; then
  echo "warning: --warmup-steps ${WARMUP_STEPS} is below the recommended 5; timing may be noisier, but the run is allowed." >&2
fi
nonnegative_int "INTERRUPT_GRACE_SECONDS" "${INTERRUPT_GRACE_SECONDS}"
positive_int "--profile-memory-breakdown-interval" "${PROFILE_MEMORY_BREAKDOWN_INTERVAL}"
# Normalize the single memory knob once, at top level, so invalid values (including the removed
# 'auto') hard-fail here instead of being silently swallowed inside run_job's command substitution.
PROFILE_MEMORY_BREAKDOWN="$(bool_value "${PROFILE_MEMORY_BREAKDOWN}")"
case "${MEMORY_BREAKDOWN_PLOT_Y_SCALE}" in
  shared|per-plot|global) ;;
  *) die "--memory-breakdown-plot-y-scale must be shared, per-plot, or global; got '${MEMORY_BREAKDOWN_PLOT_Y_SCALE}'" ;;
esac
# LORA_PARAMS entries are dropout|rank|alpha[|target]; parsed into aligned sweep arrays.
lora_dropouts=(); lora_ranks=(); lora_alphas=(); lora_targets=()
while IFS= read -r _lp_tuple; do
  IFS='|' read -r _lp_d _lp_r _lp_a _lp_t _lp_extra <<< "${_lp_tuple}"
  _lp_t="${_lp_t:-all}"
  [[ -n "${_lp_d}" && -n "${_lp_r}" && -n "${_lp_a}" && -z "${_lp_extra:-}" ]] || die "LORA_PARAMS entry must be dropout|rank|alpha[|target], got '${_lp_tuple}'"
  lora_dropout_label "${_lp_d}" >/dev/null
  positive_int "LoRA rank" "${_lp_r}"
  positive_int "LoRA alpha" "${_lp_a}"
  lora_dropouts+=("${_lp_d}"); lora_ranks+=("${_lp_r}"); lora_alphas+=("${_lp_a}"); lora_targets+=("${_lp_t}")
done < <(tokens "${LORA_PARAMS}" | dedupe)
((${#lora_dropouts[@]})) || die "LORA_PARAMS list is empty"
lf_expert_lora_impls=()
while IFS= read -r value; do
  lf_expert_lora_impls+=("${value,,}")
done < <(tokens "${lf_expert_lora_impl_spec}")
# The multi-impl sweep was retired: exactly one implementation is allowed now.
if ((${#lf_expert_lora_impls[@]} > 1)); then
  die "LF_EXPERT_LORA_IMPLS must be a single implementation; the multi-impl sweep was retired"
fi
((${#lf_expert_lora_impls[@]})) || die "LF expert LoRA implementation list is empty"
case "${lf_expert_lora_impls[0]}" in
  peft-target-parameters|split-target-parameters|off) ;;
  *) die "LF expert LoRA implementation must be peft-target-parameters, split-target-parameters, or off; got '${lf_expert_lora_impls[0]}'" ;;
esac
for value in "${lf_expert_lora_impls[@]}"; do
  if [[ "${value}" == "peft-target-parameters" || "${value}" == "split-target-parameters" ]]; then
    for dropout in "${lora_dropouts[@]}"; do
      [[ "${dropout}" == "0.00" ]] || die "LF_EXPERT_LORA_IMPLS=${value} requires LORA_DROPOUT=0.00, got '${dropout}'"
    done
  fi
done
LF_EXPERT_LORA_IMPLS="$(IFS=,; printf '%s' "${lf_expert_lora_impls[*]}")"
LORA_DROPOUT="${lora_dropouts[0]}"
LORA_RANK="${lora_ranks[0]}"
LORA_ALPHA="${lora_alphas[0]}"
LORA_TARGET="${lora_targets[0]}"
lora_dropout_label_value="$(lora_dropout_label "${LORA_DROPOUT}")"
TOTAL_STEPS=$((MAX_STEPS + WARMUP_STEPS))
PREPARE_DATASETS=$(bool_value "${PREPARE_DATASETS}")
DATASET_OVERWRITE=$(bool_value "${DATASET_OVERWRITE}")
PLOT=$(bool_value "${PLOT}")
PLOT_MEMORY_BREAKDOWN=$(bool_value "${PLOT_MEMORY_BREAKDOWN}")
OVERWRITE=$(bool_value "${OVERWRITE}")
CONTINUE_ON_ERROR=$(bool_value "${CONTINUE_ON_ERROR}")
DRY_RUN=$(bool_value "${DRY_RUN}")
COLLECT_EXISTING=$(bool_value "${COLLECT_EXISTING}")
CHECK_SUPEROFFLOAD=$(bool_value "${CHECK_SUPEROFFLOAD}")
CHECK_CPUADAM=$(bool_value "${CHECK_CPUADAM}")
RUN_POST=$(bool_value "${RUN_POST}")
EXPANDABLE_SEG=$(bool_value "${EXPANDABLE_SEG}")
USE_ASYM_CPU_ADAMW=$(bool_value "${USE_ASYM_CPU_ADAMW}")
ASYM_CPU_ADAMW_PIN_MEMORY=$(bool_value "${ASYM_CPU_ADAMW_PIN_MEMORY}")
ASYM_CPU_ADAMW_FP32_MASTER=$(bool_value "${ASYM_CPU_ADAMW_FP32_MASTER}")
case "${ASYM_CPU_ADAMW_BACKEND,,}" in
  torch) ASYM_CPU_ADAMW_BACKEND=torch ;;
  deepspeed|ds) ASYM_CPU_ADAMW_BACKEND=deepspeed ;;
  *) die "--asym-cpu-adamw-backend must be torch or deepspeed, got '${ASYM_CPU_ADAMW_BACKEND}'" ;;
esac
PYTORCH_CUDA_ALLOC_CONF_EFFECTIVE="$(allocator_conf_with_expandable_seg "${EXPANDABLE_SEG}")"
DATASET_MIN_TOKENS="${DATASET_MIN_TOKENS,,}"
positive_int "--dataset-eval-rows" "${DATASET_EVAL_ROWS}"
if [[ "${DATASET_MIN_TOKENS}" != "auto" ]]; then
  positive_int "--dataset-min-tokens" "${DATASET_MIN_TOKENS}"
fi

mapfile -t gpus < <(tokens "${gpu_spec}" | sed 's/^cuda://' | dedupe)
mapfile -t model_specs < <(tokens "${model_spec}" | dedupe)
backend_specs_raw=()
mapfile -t raw_backend_spec_tokens < <(tokens "${backend_specs_spec}")
((${#raw_backend_spec_tokens[@]} > 0)) || die "RUNS must include at least one backend|recompute[|ligerloss] spec"
for value in "${raw_backend_spec_tokens[@]}"; do
  append_backend_spec "${value}"
done
if ((${#backend_specs_raw[@]})); then
  mapfile -t backend_specs < <(printf '%s\n' "${backend_specs_raw[@]}" | dedupe)
else
  backend_specs=()
fi
mapfile -t backends < <(printf '%s\n' "${backend_specs[@]}" | cut -d '|' -f1 | dedupe)
mapfile -t backend_recompute_modes < <(printf '%s\n' "${backend_specs[@]}" | cut -d '|' -f2 | dedupe)
mapfile -t liger_loss_modes < <(printf '%s\n' "${backend_specs[@]}" | cut -d '|' -f3 | dedupe)
recompute_modes=("${backend_recompute_modes[@]}")
mapfile -t router_modes < <(tokens "${router_mode_spec}" | while read -r value; do router_mode_label "${value}"; done | dedupe)
((${#router_modes[@]} == 1)) || die "ROUTER_MODES must be a single value so each RUNS row maps to one launch"
router_hf_selected=false
router_whole_selected=false
for router_mode in "${router_modes[@]}"; do
  [[ "${router_mode}" == "hf" ]] && router_hf_selected=true
  [[ "${router_mode}" == "whole" ]] && router_whole_selected=true
done
selected_has_asym=false
selected_has_kt=false
selected_has_zero=false
selected_has_superoffload=false
selected_has_non_asym=false
for backend in "${backends[@]}"; do
  case "${backend}" in
    asym|asym_torch|asym_cpuadamwtorch|asym_cpuadamwds) selected_has_asym=true ;;
    zero2|zero3|zero3_offload|zero3_offload_mem|zero3_offload_mem_nocpuadamw|zero3_offload_opnvme|zero3_offload_panvme|zero3_offload_mem_opnvme|zero3_offload_mem_panvme|zero3_cpuadam) selected_has_zero=true ;;
    superoffload|superoffload_mem|superoffload_mem_opnvme|superoffload_mem_panvme) selected_has_zero=true; selected_has_superoffload=true ;;
    superoffload_mem_nocpuadamw) selected_has_zero=true ;;
    kt_*) selected_has_kt=true ;;
  esac
  case "${backend}" in
    asym|asym_torch|asym_cpuadamwtorch|asym_cpuadamwds) ;;
    *) selected_has_non_asym=true ;;
  esac
done
plot_router_modes=("${router_modes[@]}")
if [[ "${router_whole_selected}" == "true" && "${selected_has_non_asym}" == "true" ]]; then
  plot_router_modes+=("hf")
fi
mapfile -t plot_router_modes < <(printf '%s\n' "${plot_router_modes[@]}" | dedupe)
if [[ "${selected_has_kt}" == "true" ]]; then
  KT_TP_ENABLED=$(bool_value "${KT_TP_ENABLED}")
  CHECK_KT_CALLS=$(bool_value "${CHECK_KT_CALLS}")
  KT_SHARE_BACKWARD_BB="$(optional_bool_value "${KT_SHARE_BACKWARD_BB}")"
  KT_USE_LORA_EXPERTS="$(optional_bool_value "${KT_USE_LORA_EXPERTS}")"
  [[ -z "${KT_NUM_THREADS}" ]] || positive_int "--kt-num-threads" "${KT_NUM_THREADS}"
  [[ -z "${KT_THREADPOOL_COUNT}" ]] || positive_int "--kt-threadpool-count" "${KT_THREADPOOL_COUNT}"
  positive_int "--kt-max-cache-depth" "${KT_MAX_CACHE_DEPTH}"
  positive_int "--kt-arm-omp-num-threads" "${KT_ARM_OMP_NUM_THREADS}"
  [[ -z "${KT_ARM_SFT_TOKEN_CHUNK_SIZE}" ]] || positive_int "--kt-arm-sft-token-chunk-size" "${KT_ARM_SFT_TOKEN_CHUNK_SIZE}"
  [[ -z "${KT_NUM_GPU_EXPERTS}" ]] || nonnegative_int "--kt-num-gpu-experts" "${KT_NUM_GPU_EXPERTS}"
  [[ -z "${KT_LORA_EXPERT_NUM}" ]] || positive_int "--kt-lora-expert-num" "${KT_LORA_EXPERT_NUM}"
  [[ -z "${KT_LORA_EXPERT_INTERMEDIATE_SIZE}" ]] || positive_int "--kt-lora-expert-intermediate-size" "${KT_LORA_EXPERT_INTERMEDIATE_SIZE}"
  if [[ "${kt_repo_dir_user_set}" != "true" ]]; then
    KT_REPO_DIR="$(dirname "${KT_KERNEL_DIR}")"
  fi
  if [[ "${kt_gguf_py_dir_user_set}" != "true" ]]; then
    KT_GGUF_PY_DIR="${KT_REPO_DIR}/third_party/llama.cpp/gguf-py"
  fi
fi
mapfile -t profilers < <(tokens "${profiler_spec}" | while read -r value; do profiler_label "${value}"; done | dedupe)
((${#profilers[@]} == 1)) || die "PROFILERS must be a single value so each RUNS row maps to one launch; use both for one nsys run plus source artifacts"
both_profiler_selected=false
nsys_artifacts_selected=false
if printf '%s\n' "${profilers[@]}" | grep -qx 'both'; then
  both_profiler_selected=true
  nsys_artifacts_selected=true
fi
if printf '%s\n' "${profilers[@]}" | grep -qx 'nsys'; then
  nsys_artifacts_selected=true
fi
if [[ "${both_profiler_selected}" == "true" && "${#profilers[@]}" != "1" ]]; then
  die "PROFILERS=both must be used alone; use PROFILERS=source,nsys for two separate runs or PROFILERS=both for one nsys run plus materialized source artifacts"
fi
if [[ -z "${output_root}" ]]; then
  if [[ "${both_profiler_selected}" == "true" ]]; then
    output_root="${ASYM_DIR}/profiling_both"
  else
    output_root="${ASYM_DIR}/profiling"
  fi
fi
mapfile -t asym_cpu_adamw_grad_offload_modes < <(
  tokens "${ASYM_CPU_ADAMW_GRAD_OFFLOAD}" | while read -r value; do bool_value "${value}"; done | dedupe
)
((${#asym_cpu_adamw_grad_offload_modes[@]})) || die "ASYM_CPU_ADAMW_GRAD_OFFLOAD must include at least one boolean value"
mapfile -t asym_cpu_adamw_weight_offload_modes < <(
  tokens "${ASYM_CPU_ADAMW_WEIGHT_OFFLOAD}" | while read -r value; do bool_value "${value}"; done | dedupe
)
((${#asym_cpu_adamw_weight_offload_modes[@]})) || die "ASYM_CPU_ADAMW_WEIGHT_OFFLOAD must include at least one boolean value"
if [[ "${asym_cpu_adamw_weight_offload_modes[0]}" == "true" && "${asym_cpu_adamw_grad_offload_modes[0]}" != "true" ]]; then
  die "ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true requires ASYM_CPU_ADAMW_GRAD_OFFLOAD=true so each CPUAdamW RUNS row maps to one launch"
fi
if [[ "${both_profiler_selected}" == "true" ]]; then
  plot_profilers=(nsys source)
elif [[ "${nsys_artifacts_selected}" == "true" ]]; then
  plot_profilers=(nsys)
else
  plot_profilers=()
  for profiler in "${profilers[@]}"; do
    if [[ "$(bool_value "${PROFILE_MEMORY_BREAKDOWN}")" != "true" ]]; then
      plot_profilers+=("${profiler}")
    fi
  done
fi
workloads=()
while IFS= read -r workload_value; do
  [[ -n "${workload_value}" ]] || continue
  parsed_workload="$(parse_workload_tuple "${workload_value}")" || exit 1
  workloads+=("${parsed_workload}")
done <<< "$(tokens "${workload_spec}")"
((${#workloads[@]})) || die "workload list is empty"
mapfile -t workloads < <(printf '%s\n' "${workloads[@]}" | dedupe)
mapfile -t seq_lens < <(printf '%s\n' "${workloads[@]}" | cut -d '|' -f1 | dedupe)
mapfile -t workload_batch_sizes < <(printf '%s\n' "${workloads[@]}" | cut -d '|' -f2 | dedupe)
mapfile -t workload_grad_accum_steps < <(printf '%s\n' "${workloads[@]}" | cut -d '|' -f3 | dedupe)
IFS='|' read -r _first_workload_seq PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS <<< "${workloads[0]}"
batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}"
exp_act_policy_pairs=()
mapfile -t raw_exp_act_policy_pairs < <(tokens "${exp_act_policy_spec}" | dedupe)
((${#raw_exp_act_policy_pairs[@]} > 0)) || die "RUNS must include at least one policy|expert_act|attn_act|layer_act|layer_gc tuple"
for value in "${raw_exp_act_policy_pairs[@]}"; do
  exp_act_policy_pairs+=("$(parse_exp_act_policy_tuple "${value}")")
done
mapfile -t exp_act_policy_pairs < <(printf '%s\n' "${exp_act_policy_pairs[@]}" | dedupe)
((${#exp_act_policy_pairs[@]})) || die "expert/attention activation policy tuple list is empty"
mapfile -t expert_policies < <(printf '%s\n' "${exp_act_policy_pairs[@]}" | cut -d '|' -f1 | dedupe)
mapfile -t expact_values < <(printf '%s\n' "${exp_act_policy_pairs[@]}" | cut -d '|' -f2 | dedupe)
mapfile -t attnact_values < <(printf '%s\n' "${exp_act_policy_pairs[@]}" | cut -d '|' -f3 | dedupe)
mapfile -t layeract_values < <(printf '%s\n' "${exp_act_policy_pairs[@]}" | cut -d '|' -f4 | dedupe)
mapfile -t layergc_values < <(printf '%s\n' "${exp_act_policy_pairs[@]}" | cut -d '|' -f5 | dedupe)
mapfile -t sdparecomp_values < <(printf '%s\n' "${exp_act_policy_pairs[@]}" | cut -d '|' -f6 | dedupe)
plot_expact_values=("${expact_values[@]}")
plot_attnact_values=("${attnact_values[@]}")
plot_layeract_values=("${layeract_values[@]}")
plot_layergc_values=("${layergc_values[@]}")
if [[ "${selected_has_non_asym}" == "true" ]]; then
  # Policy-independent baselines are canonicalized to no activation offload when they run.
  # Include that canonical label in plot filters so those baselines are not filtered out.
  plot_expact_values+=(false)
  plot_attnact_values+=(false)
  plot_layeract_values+=(false)
  plot_layergc_values+=(false)
fi
mapfile -t plot_expact_values < <(printf '%s\n' "${plot_expact_values[@]}" | dedupe)
mapfile -t plot_attnact_values < <(printf '%s\n' "${plot_attnact_values[@]}" | dedupe)
mapfile -t plot_layeract_values < <(printf '%s\n' "${plot_layeract_values[@]}" | dedupe)
mapfile -t plot_layergc_values < <(printf '%s\n' "${plot_layergc_values[@]}" | dedupe)
ASYMM_EXPERT_ACT_OFFLOAD="${expact_values[0]}"
expact_label="$(expact_tag "${ASYMM_EXPERT_ACT_OFFLOAD}")"
actrecomp_label="$(actrecomp_tag "${ASYM_OFFLOAD_ACT_RECOMPUTE}")"
xunpack_label="$(xunpack_tag "${ASYM_OFFLOAD_X_UNPACKED}")"
moefg_label="$(moefg_tag "${ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD}")"
dscatter_label="$(dscatter_tag "${ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS}")"
q3rt_fwd_flag="$(qwen3_route_effective_flag ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER)"
q3rt_gather_flag="$(qwen3_route_effective_flag ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER)"
q3rt_dx_flag="$(qwen3_route_effective_flag ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER)"
q3rt_lora_flag="$(qwen3_route_bool "${ASYMM_QWEN3_MOE_ROUTE_LORA}")"
q3rt_label="$(qwen3_route_tag "${q3rt_fwd_flag}" "${q3rt_gather_flag}" "${q3rt_dx_flag}" "${q3rt_lora_flag}" "${ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE}")"
ASYMM_ATTN_ACT_OFFLOAD="${attnact_values[0]}"
attnact_label="$(attnact_tag "${ASYMM_ATTN_ACT_OFFLOAD}")"
ASYMM_LAYER_ACT_OFFLOAD="${layeract_values[0]}"
layeract_label="$(layeract_tag "${ASYMM_LAYER_ACT_OFFLOAD}")"
ASYMM_LAYER_GC="${layergc_values[0]}"
layergc_label="$(layergc_tag "${ASYMM_LAYER_GC}")"
ASYMM_ATTN_SDPA_RECOMPUTE="${sdparecomp_values[0]:-false}"
sdparecomp_label="$(sdparecomp_tag "${ASYMM_ATTN_SDPA_RECOMPUTE}")"
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD="$(normalize_expact_lora_a_fwd "${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}")"
expact_lora_a_fwd_label="$(expact_lora_a_fwd_tag "${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}")"

paired_run_specs=()
for run_spec_value in "${_run_specs[@]}"; do
  [[ -n "${run_spec_value}" ]] || continue
  normalized_run_spec="$(normalize_run_spec_entry "${run_spec_value}")" || exit $?
  [[ -n "${normalized_run_spec}" ]] || die "RUNS item produced an empty normalized run: '${run_spec_value}'"
  paired_run_specs+=("${normalized_run_spec}")
done
((${#paired_run_specs[@]})) || die "RUNS schedule is empty"
echo "Using RUNS schedule: ${#paired_run_specs[@]} row(s)"

((${#gpus[@]})) || die "GPU pool is empty"
((${#model_specs[@]})) || die "model spec list is empty"
((${#backend_specs[@]})) || die "backend spec list is empty"
((${#backends[@]})) || die "backend list is empty"
((${#router_modes[@]})) || die "router mode list is empty"
((${#profilers[@]})) || die "profiler list is empty"
((${#workloads[@]})) || die "workload list is empty"
((${#seq_lens[@]})) || die "sequence length list is empty"
((${#workload_batch_sizes[@]})) || die "workload batch-size list is empty"
((${#workload_grad_accum_steps[@]})) || die "workload gradient-accumulation list is empty"
((${#expert_policies[@]})) || die "expert policy list is empty"
[[ -f "${RUN_LF_SCRIPT}" ]] || die "missing ${RUN_LF_SCRIPT}"
if [[ "${selected_has_kt}" == "true" ]]; then
  [[ -d "${KT_KERNEL_DIR}" ]] || die "missing integrated kt-kernel dir ${KT_KERNEL_DIR}"
  kt_gguf_available || die "missing KT gguf dependency; set --kt-gguf-py-dir to vendored gguf-py or install the gguf package"
fi
[[ -f "${BUILD_DATASET_SCRIPT}" ]] || die "missing ${BUILD_DATASET_SCRIPT}"
[[ -f "${PROFILE_POSTPROCESS_SCRIPT}" ]] || die "missing ${PROFILE_POSTPROCESS_SCRIPT}"
[[ -f "${MEMORY_SCHEMA_VALIDATOR}" ]] || die "missing ${MEMORY_SCHEMA_VALIDATOR}"
[[ -f "${PLOT_SCRIPT}" ]] || die "missing ${PLOT_SCRIPT}"
if [[ "${PLOT_MEMORY_BREAKDOWN}" == "true" ]]; then
  [[ -f "${MEMORY_PLOT_SCRIPT}" ]] || die "missing ${MEMORY_PLOT_SCRIPT}"
fi
if [[ "${PLOT}" == "true" && "${nsys_artifacts_selected}" == "true" ]]; then
  [[ -f "${INTERCONNECT_PLOT_SCRIPT}" ]] || die "missing ${INTERCONNECT_PLOT_SCRIPT}"
  [[ -f "${UTILIZATION_PLOT_SCRIPT}" ]] || die "missing ${UTILIZATION_PLOT_SCRIPT}"
  [[ -f "${THROUGHPUT_PLOT_SCRIPT}" ]] || die "missing ${THROUGHPUT_PLOT_SCRIPT}"
fi
if [[ "${DRY_RUN}" != "true" && ( "${PLOT}" == "true" || ( "${PREPARE_DATASETS}" == "true" && "${COLLECT_EXISTING}" != "true" ) ) ]]; then
  [[ -x "${ENV_PYTHON}" ]] || die "missing executable LF Python at ${ENV_PYTHON}"
fi
if [[ "${selected_has_zero}" == "true" ]]; then
  for backend in "${backends[@]}"; do
    if is_zero_backend "${backend}"; then
      [[ -f "$(zero_deepspeed_config "${backend}")" ]] || die "missing DeepSpeed config for ${backend}: $(zero_deepspeed_config "${backend}")"
    fi
  done
fi
base_output_root="$(abs_path "${output_root}")"
if [[ "${selected_has_superoffload}" == "true" ]]; then
  [[ -f "${DEEPSPEED_DIR}/deepspeed/runtime/superoffload/superoffload_stage3.py" ]] || die "missing local SuperOffload DeepSpeed tree at ${DEEPSPEED_DIR}"
fi
precision_label="$(safe_label "${PRECISION}")"
dataset_root_label="$(safe_label "${DATASET}")"
[[ -n "${dataset_root_label}" ]] || die "--dataset must not be empty"
precision_root="${base_output_root}/${dataset_root_label}__lora__lf__${precision_label}"
if [[ "${DRY_RUN}" != "true" && "${COLLECT_EXISTING}" != "true" ]]; then
  mkdir -p "${precision_root}"
fi
echo "Output precision root: ${precision_root}"
mkdir -p "$(dirname "${_RUNS_LOG}")"
if [[ -s "${_RUNS_LOG}" ]]; then
  printf '\n' >> "${_RUNS_LOG}"
fi
{
  printf '===== profile_lora_lf_test_both.sh invocation: %s =====\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  printf 'runs=%s lora_configs=%s profilers=%s router_modes=%s\n' "${#paired_run_specs[@]}" "${#lora_dropouts[@]}" "${PROFILERS}" "${ROUTER_MODES}"
} >> "${_RUNS_LOG}"
echo "Run log: ${_RUNS_LOG} (append mode)"

declare -A plot_roots=()
declare -A memory_plot_roots=()
declare -A interconnect_plot_roots=()
declare -A utilization_plot_roots=()
declare -A throughput_plot_roots=()
failures=0
interrupted=false
interrupt_exit_status=130
current_child_pid=""
current_child_pid_file=""
current_wait_pid=""

child_process_alive() {
  local target_pid
  for target_pid in "$@"; do
    [[ "${target_pid}" =~ ^[0-9]+$ ]] || continue
    if kill -0 "-${target_pid}" 2>/dev/null || kill -0 "${target_pid}" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

signal_child_targets() {
  local signal_name="$1"
  shift
  local target_pid
  for target_pid in "$@"; do
    [[ "${target_pid}" =~ ^[0-9]+$ ]] || continue
    kill "-${signal_name}" "-${target_pid}" 2>/dev/null || true
    kill "-${signal_name}" "${target_pid}" 2>/dev/null || true
  done
}

current_child_targets() {
  local file_pid="" target_pid
  if [[ -n "${current_child_pid_file:-}" && -s "${current_child_pid_file}" ]]; then
    IFS= read -r file_pid < "${current_child_pid_file}" || true
  fi
  for target_pid in "${file_pid}" "${current_child_pid:-}" "${current_wait_pid:-}"; do
    [[ "${target_pid}" =~ ^[0-9]+$ ]] && printf '%s\n' "${target_pid}"
  done | awk 'NF && !seen[$0]++'
}

kill_current_child() {
  local -a target_pids=()
  mapfile -t target_pids < <(current_child_targets)
  rm -f "${current_child_pid_file:-}" 2>/dev/null || true

  ((${#target_pids[@]} > 0)) || return 0
  signal_child_targets INT "${target_pids[@]}"

  if [[ "${INTERRUPT_GRACE_SECONDS}" != "0" ]]; then
    sleep "${INTERRUPT_GRACE_SECONDS}" || true
  fi
  if child_process_alive "${target_pids[@]}"; then
    signal_child_targets TERM "${target_pids[@]}"
  fi

  if [[ "${INTERRUPT_GRACE_SECONDS}" != "0" ]]; then
    sleep "${INTERRUPT_GRACE_SECONDS}" || true
  fi
  if child_process_alive "${target_pids[@]}"; then
    signal_child_targets KILL "${target_pids[@]}"
  fi
}

cleanup_on_exit() {
  if [[ -n "${current_child_pid:-}" || -n "${current_wait_pid:-}" || -n "${current_child_pid_file:-}" ]]; then
    kill_current_child
  fi
}

run_tracked_command() {
  local status=0 wait_pid="" child_pid="" pid_file="" attempt
  current_child_pid=""
  current_wait_pid=""
  if command -v setsid >/dev/null 2>&1 && setsid --help 2>&1 | grep -q -- '--wait' && setsid --help 2>&1 | grep -q -- '--fork'; then
    pid_file="$(mktemp "${TMPDIR:-/tmp}/profile_lora_lf_child.XXXXXX")"
    current_child_pid_file="${pid_file}"
    setsid --fork --wait bash -c 'pid_file="$1"; shift; echo "$$" > "${pid_file}"; exec "$@"' _ "${pid_file}" "$@" &
    wait_pid=$!
    current_wait_pid="${wait_pid}"
    current_child_pid="${wait_pid}"
    for ((attempt = 0; attempt < 100; attempt++)); do
      if [[ -s "${pid_file}" ]]; then
        IFS= read -r child_pid < "${pid_file}" || true
        if [[ -n "${child_pid}" ]]; then
          current_child_pid="${child_pid}"
          break
        fi
      fi
      kill -0 "${wait_pid}" 2>/dev/null || break
      sleep 0.02
    done
  else
    "$@" &
    wait_pid=$!
    current_wait_pid="${wait_pid}"
    current_child_pid="${wait_pid}"
  fi
  wait "${wait_pid}" || status=$?
  if [[ "${interrupted}" == "true" ]]; then
    kill_current_child
    wait "${wait_pid}" 2>/dev/null || true
    current_child_pid=""
    current_child_pid_file=""
    current_wait_pid=""
    echo "Interrupted command; exiting without scheduling more jobs." >&2
    exit "${interrupt_exit_status}"
  fi
  if [[ "${status}" == "130" || "${status}" == "143" ]]; then
    kill_current_child
    wait "${wait_pid}" 2>/dev/null || true
    current_child_pid=""
    current_child_pid_file=""
    current_wait_pid=""
    echo "Interrupted command; exiting without scheduling more jobs." >&2
    exit "${status}"
  fi
  [[ -z "${pid_file}" ]] || rm -f "${pid_file}" 2>/dev/null || true
  current_child_pid=""
  current_child_pid_file=""
  current_wait_pid=""
  return "${status}"
}

run_tracked_command_logged() {
  local log_file="$1"
  shift
  run_tracked_command "$@" > >(tee -a "${log_file}") 2>&1
}

materialize_source_artifacts_from_nsys() {
  local nsys_source_profile="$1"
  local source_seq_root="$2"
  local source_profile_json="$3"
  local source_output_profile_json="$4"
  local source_log_file="$5"
  local nsys_seq_root="$6"
  local -a postprocess_cmd=(
    "${ENV_PYTHON}" "${PROFILE_POSTPROCESS_SCRIPT}"
    --source-profile-json "${source_profile_json}"
    --profile-json "${source_output_profile_json}"
    --output-dir "${source_seq_root}"
  )

  if [[ ! -f "${nsys_source_profile}" ]]; then
    echo "Cannot materialize source artifacts; missing nsys source profile: ${nsys_source_profile}" >&2
    return 1
  fi

  mkdir -p "${source_seq_root}"
  rm -f "${source_seq_root}/PENDING.md"  # drop the placeholder now that real artifacts land here
  cp "${nsys_source_profile}" "${source_profile_json}"
  for memory_artifact in memory_breakdown.jsonl memory_breakdown_summary.json; do
    if [[ -f "${nsys_seq_root}/${memory_artifact}" ]]; then
      cp "${nsys_seq_root}/${memory_artifact}" "${source_seq_root}/${memory_artifact}"
    fi
  done
  if [[ -d "${nsys_seq_root}/memory_breakdown" ]]; then
    mkdir -p "${source_seq_root}/memory_breakdown"
    for memory_artifact in memory_breakdown.jsonl memory_breakdown_summary.json; do
      if [[ -f "${nsys_seq_root}/memory_breakdown/${memory_artifact}" ]]; then
        cp "${nsys_seq_root}/memory_breakdown/${memory_artifact}" "${source_seq_root}/memory_breakdown/${memory_artifact}"
      fi
    done
  fi
  {
    echo "Materialized source artifacts from nsys run:"
    echo "  nsys_dir=${nsys_seq_root}"
    echo "  nsys_source_profile=${nsys_source_profile}"
    print_command "${postprocess_cmd[@]}"
  } > "${source_seq_root}/command.txt"
  : > "${source_log_file}"
  echo "Materializing source artifacts from ${nsys_source_profile} into ${source_seq_root}"
  run_tracked_command_logged "${source_log_file}" "${postprocess_cmd[@]}"
}

postprocess_existing_nsys_artifacts() {
  local seq_root="$1"
  local source_profile_json="$2"
  local profile_json="$3"
  local sqlite_path="${seq_root}/trace.sqlite"
  local summary_md="${seq_root}/summary.md"
  local -a postprocess_cmd=(
    "${ENV_PYTHON}" "${PROFILE_NSYS_POSTPROCESS_SCRIPT}"
    "${sqlite_path}"
    --output-json "${profile_json}"
    --output-md "${summary_md}"
  )

  if [[ ! -f "${sqlite_path}" ]]; then
    echo "warning: cannot refresh nsys metric artifacts; missing sqlite trace: ${sqlite_path}" >&2
    return 0
  fi
  if [[ ! -f "${PROFILE_NSYS_POSTPROCESS_SCRIPT}" ]]; then
    echo "Missing Nsight postprocess script ${PROFILE_NSYS_POSTPROCESS_SCRIPT}" >&2
    return 2
  fi
  if [[ -f "${source_profile_json}" ]]; then
    postprocess_cmd+=(--source-profile-json "${source_profile_json}")
  fi

  echo "Refreshing LF nsys metric artifacts: ${seq_root}"
  run_tracked_command "${postprocess_cmd[@]}"
}

handle_interrupt() {
  local signal="${1:-INT}"
  case "${signal}" in
    TERM) interrupt_exit_status=143 ;;
    *) interrupt_exit_status=130 ;;
  esac
  interrupted=true
  echo "Interrupted; stopping LF profiling sweep." >&2
  trap - INT TERM
  kill_current_child
  exit "${interrupt_exit_status}"
}

trap 'handle_interrupt INT' INT
trap 'handle_interrupt TERM' TERM
trap 'cleanup_on_exit' EXIT

run_job() {
  local backend="$1"
  local profiler="$2"
  local run_profiler="$2"
  local materialize_source_from_nsys=false
  local recompute="$3"
  local liger_loss="$4"
  local seq_len="$5"
  local gpu="$6"
  local gpu_count="$7"
  local expert_policy="$8"
  local router_mode="$9"
  local dataset_name="${10}"
  local lf_expert_lora_impl="${11}"
  local grad_offload="${12:-false}"
  local weight_offload="${13:-false}"
  local requested_recompute_kernel_code="${14:-}"
  local requested_outer_hbm_every_n="${15:-0}"
  # Canonicalize the policy axes for inert runs; local shadows feed run_id, the folder, the env, and the check.
  local expact_label="${expact_label}" attnact_label="${attnact_label}" layeract_label="${layeract_label}" layergc_label="${layergc_label}" sdparecomp_label="${sdparecomp_label}" expact_lora_a_fwd_label="${expact_lora_a_fwd_label}" actrecomp_label="${actrecomp_label}" xunpack_label="${xunpack_label}" moefg_label="${moefg_label}" dscatter_label="${dscatter_label}" q3rt_label="${q3rt_label}"
  local ASYMM_EXPERT_ACT_OFFLOAD="${ASYMM_EXPERT_ACT_OFFLOAD}" ASYMM_ATTN_ACT_OFFLOAD="${ASYMM_ATTN_ACT_OFFLOAD}" ASYMM_LAYER_ACT_OFFLOAD="${ASYMM_LAYER_ACT_OFFLOAD}" ASYMM_LAYER_GC="${ASYMM_LAYER_GC}" ASYMM_ATTN_SDPA_RECOMPUTE="${ASYMM_ATTN_SDPA_RECOMPUTE}" ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD="${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}"
  local ASYM_OFFLOAD_ACT_RECOMPUTE="${ASYM_OFFLOAD_ACT_RECOMPUTE}" ASYM_OFFLOAD_X_UNPACKED="${ASYM_OFFLOAD_X_UNPACKED}"
  local ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD="${ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD}"
  local ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS="${ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS}"
  local ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM="${ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM:-0}"
  local ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER="${ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER-}"
  local ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER="${ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER-}"
  local ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER="${ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER-}"
  local ASYMM_QWEN3_MOE_ROUTE_LORA="${ASYMM_QWEN3_MOE_ROUTE_LORA:-0}"
  local ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE="${ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE:-fp32}"
  local ASYMM_QWEN3_MOE_ROUTE_KERNEL_DEBUG="${ASYMM_QWEN3_MOE_ROUTE_KERNEL_DEBUG:-0}"
  local q3rt_fwd_flag q3rt_gather_flag q3rt_dx_flag q3rt_lora_flag q3rt_kernel_code recompute_artifact_label recompute_plot_filter
  local ASYMM_EXPERT_SILU_BWD_GPU="${ASYMM_EXPERT_SILU_BWD_GPU:-1}"
  local ASYMM_MLP_RECOMPUTE_CHUNK="${ASYMM_MLP_RECOMPUTE_CHUNK:-0}"
  # Per-run -ohbm<N> wins over the sweep-wide env; the dynamic scope feeds
  # job_root_path, the launch env, and the profile completeness checks.
  local UNSLOTH_GC_OUTER_HBM_EVERY_N="${UNSLOTH_GC_OUTER_HBM_EVERY_N:-0}"
  if [[ "${requested_outer_hbm_every_n:-0}" != "0" ]]; then
    UNSLOTH_GC_OUTER_HBM_EVERY_N="${requested_outer_hbm_every_n}"
  fi
  # Non-Unsloth-GC modes ignore a sweep-wide env value: run_lf rejects
  # UNSLOTH_GC_OUTER_HBM_EVERY_N>0 with USE_UNSLOTH_GC=false, and the dir
  # suffix would misdescribe the run.
  case "${recompute}" in
    unsloth|unsloth-off) ;;
    *) is_recomp_off_recompute "${recompute}" || UNSLOTH_GC_OUTER_HBM_EVERY_N=0 ;;
  esac
  local job_lf_dir="${CURRENT_LF_DIR:-${LF_DIR}}"
  local job_env_dir="${CURRENT_ENV_DIR:-${ENV_DIR}}"
  local job_flash_attn="${CURRENT_FLASH_ATTN:-${FLASH_ATTN}}"
  local flashattn_label="${CURRENT_FLASH_ATTN_LABEL:-$(flash_attn_tag "${job_flash_attn}")}"
  local requested_policy_tuple="${expert_policy}|${ASYMM_EXPERT_ACT_OFFLOAD}|${ASYMM_ATTN_ACT_OFFLOAD}|${ASYMM_LAYER_ACT_OFFLOAD}|${ASYMM_LAYER_GC}|${ASYMM_ATTN_SDPA_RECOMPUTE}"
  canonicalize_policy_axis_for_inert_run "${backend}" "${recompute}"
  local run_log_extra_tag=""
  local run_log_inert_reason=""
  if [[ "${profiler}" == "both" ]]; then
    run_profiler=nsys
    materialize_source_from_nsys=true
  fi
  local gradient_checkpointing=false
  local use_unsloth_gc=false
  local unsloth_recompute_save_on_cpu=false
  local attention_gc_enabled=false
  local layer_gc_enabled=false
  local dense_mlp_surgical="${ASYMM_DENSE_MLP_SURGICAL_OFFLOAD:-0}"
  local dense_mlp_finegrained="${ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD:-0}"
  local requested_qwen3_moe_down_scatter_block_experts="${ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS}"
  local recomp_off_stage=""
  [[ "${recompute}" == "recomp" ]] && gradient_checkpointing=true
  [[ "${recompute}" == "unsloth" || "${recompute}" == "unsloth-off" ]] && { gradient_checkpointing=true; use_unsloth_gc=true; }
  if [[ "${recompute}" == "unsloth-off" ]]; then
    unsloth_recompute_save_on_cpu=true
    dense_mlp_surgical=0
    dense_mlp_finegrained=0
  fi
  if is_recomp_off_recompute "${recompute}"; then
    recomp_off_stage="$(recomp_off_stage_label "${recompute}")"
    gradient_checkpointing=true
    use_unsloth_gc=true
    unsloth_recompute_save_on_cpu=true
    expert_policy=none
    ASYM_OFFLOAD_ACT_RECOMPUTE=0; actrecomp_label="$(actrecomp_tag 0)"
    ASYM_OFFLOAD_X_UNPACKED=0; xunpack_label="$(xunpack_tag 0)"
    ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD=0; moefg_label="$(moefg_tag 0)"
    ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0; dscatter_label="$(dscatter_tag 0)"
    ASYMM_LAYER_ACT_OFFLOAD=false; layeract_label="$(layeract_tag false)"
    if [[ "$(bool_value "${ASYMM_RECOMP_OFF_ALLOW_LAYER_GC:-false}")" != "true" ]]; then
      ASYMM_LAYER_GC=false; layergc_label="$(layergc_tag false)"
    fi
    if [[ "$(bool_value "${ASYMM_RECOMP_OFF_ALLOW_SDPA_RECOMPUTE:-false}")" != "true" ]]; then
      ASYMM_ATTN_SDPA_RECOMPUTE=false; sdparecomp_label="$(sdparecomp_tag false)"
    fi
    ASYMM_EXPERT_SILU_BWD_GPU=0
    ASYMM_MLP_RECOMPUTE_CHUNK=0
    dense_mlp_surgical=0
    dense_mlp_finegrained=0
    case "${recomp_off_stage}" in
      base)
        ASYMM_EXPERT_ACT_OFFLOAD=false; expact_label="$(expact_tag false)"
        ASYMM_ATTN_ACT_OFFLOAD=false; attnact_label="$(attnact_tag false)"
        ;;
      attn)
        ASYMM_EXPERT_ACT_OFFLOAD=false; expact_label="$(expact_tag false)"
        ASYMM_ATTN_ACT_OFFLOAD=true; attnact_label="$(attnact_tag true)"
        ;;
      dense)
        dense_mlp_surgical=1
        ASYMM_EXPERT_ACT_OFFLOAD=true; expact_label="$(expact_tag true)"
        ASYMM_ATTN_ACT_OFFLOAD=false; attnact_label="$(attnact_tag false)"
        ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu; expact_lora_a_fwd_label="$(expact_lora_a_fwd_tag cpu)"
        ;;
      full)
        dense_mlp_surgical=1
        ASYMM_EXPERT_ACT_OFFLOAD=true; expact_label="$(expact_tag true)"
        ASYMM_ATTN_ACT_OFFLOAD=true; attnact_label="$(attnact_tag true)"
        ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu; expact_lora_a_fwd_label="$(expact_lora_a_fwd_tag cpu)"
        ;;
      dense-fg)
        dense_mlp_finegrained=1
        ASYMM_EXPERT_ACT_OFFLOAD=false; expact_label="$(expact_tag false)"
        ASYMM_ATTN_ACT_OFFLOAD=false; attnact_label="$(attnact_tag false)"
        ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu; expact_lora_a_fwd_label="$(expact_lora_a_fwd_tag cpu)"
        ;;
      full-fg)
        dense_mlp_finegrained=1
        ASYMM_EXPERT_ACT_OFFLOAD=false; expact_label="$(expact_tag false)"
        ASYMM_ATTN_ACT_OFFLOAD=true; attnact_label="$(attnact_tag true)"
        ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu; expact_lora_a_fwd_label="$(expact_lora_a_fwd_tag cpu)"
        if is_qwen3_moe_routed_model "${current_model_name}"; then
          ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD=1; moefg_label="$(moefg_tag 1)"
          ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS="${requested_qwen3_moe_down_scatter_block_experts}"
          dscatter_label="$(dscatter_tag "${ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS}")"
          # fix_qwen3 v1: GPU LoRA-A recompute-forward + wide pinned pool (192 GiB)
          ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU="${ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU:-1}"
          ASYM_EXPACT_CPU_POOL_MAX_BYTES="${ASYM_EXPACT_CPU_POOL_MAX_BYTES:-206158430208}"
          # fix_qwen3 v2: compact-X GPU dA + dgate/dup kept in HBM
          ASYMM_QWEN3_MOE_FG_DA_GPU="${ASYMM_QWEN3_MOE_FG_DA_GPU:-1}"
          ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM="${ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM:-1}"
        fi
        ;;
    esac
  fi
  if [[ -n "${requested_recompute_kernel_code}" ]]; then
    ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER="${requested_recompute_kernel_code:0:1}"
    ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER="${requested_recompute_kernel_code:1:1}"
    ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER="${requested_recompute_kernel_code:2:1}"
    ASYMM_QWEN3_MOE_ROUTE_LORA="${ASYMM_QWEN3_MOE_ROUTE_LORA:-0}"
    ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE="${ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE:-fp32}"
  elif qwen3_moe_routed_auto_default "${backend}" "${recomp_off_stage}" "${current_model_name}" "${ASYMM_EXPERT_ACT_OFFLOAD}"; then
    ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER="${ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER:-1}"
    ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER="${ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER:-0}"
    ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER="${ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER:-1}"
    ASYMM_QWEN3_MOE_ROUTE_LORA="${ASYMM_QWEN3_MOE_ROUTE_LORA:-0}"
    ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE="${ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE:-fp32}"
    ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0; dscatter_label="$(dscatter_tag 0)"
  fi
  q3rt_fwd_flag="$(qwen3_route_effective_flag ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER)"
  q3rt_gather_flag="$(qwen3_route_effective_flag ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER)"
  q3rt_dx_flag="$(qwen3_route_effective_flag ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER)"
  q3rt_lora_flag="$(qwen3_route_bool "${ASYMM_QWEN3_MOE_ROUTE_LORA}")"
  q3rt_label="$(qwen3_route_tag "${q3rt_fwd_flag}" "${q3rt_gather_flag}" "${q3rt_dx_flag}" "${q3rt_lora_flag}" "${ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE}")"
  q3rt_kernel_code="$(qwen3_route_kernel_code "${q3rt_fwd_flag}" "${q3rt_gather_flag}" "${q3rt_dx_flag}")"
  validate_recompute_kernel_for_model "${current_model_name}" "${q3rt_kernel_code}" "${q3rt_lora_flag}"
  recompute_artifact_label="$(recompute_run_label "${recompute}" "${q3rt_kernel_code}")"
  # Full recompute token for per-run plot filters (matches the plotters'
  # folded metadata axis, e.g. recomp-off-full-fg-ker101-ohbm6); base tokens
  # would drop ker/ohbm runs from their own per-run plots.
  recompute_plot_filter="${recompute_artifact_label}"
  if [[ "${UNSLOTH_GC_OUTER_HBM_EVERY_N:-0}" != "0" ]]; then
    recompute_plot_filter="${recompute_plot_filter}-ohbm${UNSLOTH_GC_OUTER_HBM_EVERY_N}"
  fi
  if qwen3_route_any_enabled "${q3rt_fwd_flag}" "${q3rt_gather_flag}" "${q3rt_dx_flag}" "${q3rt_lora_flag}"; then
    [[ "${ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE,,}" == "fp32" ]] || die "Qwen3 routed kernels currently require ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32, got '${ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE}'"
    [[ "${backend}" == asym* ]] || die "Qwen3 routed kernels are only valid for asym* backends, got backend='${backend}'"
    [[ "${recomp_off_stage}" == "full-fg" ]] || die "Qwen3 routed kernels require recomp-off-full-fg, got recompute='${recompute}'"
    is_qwen3_moe_routed_model "${current_model_name}" || die "Qwen3 routed kernels are scoped to Qwen3/Qwen3.5 routed MoE, got model='${current_model_name}'"
    [[ "$(qwen3_route_bool "${ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD}")" == "1" ]] || die "Qwen3 routed kernels require ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD=1"
    [[ "${ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS}" == "0" ]] || die "Qwen3 routed kernels must not use ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=${ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS}"
  fi
  [[ "${expert_policy}" == "gc-attn-exp" ]] && attention_gc_enabled=true
  [[ "${expert_policy}" == "gc-layer" ]] && layer_gc_enabled=true
  if ! cpuadam_backend_for_label "${backend}" >/dev/null; then
    grad_offload=false
    weight_offload=false
  fi
  local effective_policy_tuple="${expert_policy}|${ASYMM_EXPERT_ACT_OFFLOAD}|${ASYMM_ATTN_ACT_OFFLOAD}|${ASYMM_LAYER_ACT_OFFLOAD}|${ASYMM_LAYER_GC}|${ASYMM_ATTN_SDPA_RECOMPUTE}"
  if [[ "${effective_policy_tuple}" != "${requested_policy_tuple}" ]] && ! is_recomp_off_recompute "${recompute}"; then
    run_log_extra_tag="INERT"
    run_log_inert_reason=" requested_policy=${requested_policy_tuple}"
  fi
  local enable_liger_kernel
  case "${liger_loss}" in
    ligerloss0) enable_liger_kernel=false ;;
    ligerloss1) enable_liger_kernel=true ;;
    *) die "internal error: invalid liger_loss '${liger_loss}'" ;;
  esac

  local config_root job_root seq_root source_profile lf_out log_file run_id profile_json run_dir_name
  local source_materialized_job_root="" source_materialized_seq_root="" source_materialized_profile_json="" source_materialized_source_profile="" source_materialized_log_file=""
  local kt_arm_source_ok_profile_json=""
  config_root="$(config_root_path "${seq_len}")"
  job_root="$(job_root_path "${config_root}" "${backend}" "${run_profiler}" "${recompute_artifact_label}" "${expert_policy}" "${router_mode}" "${liger_loss}" "${grad_offload}" "${weight_offload}")"
  run_dir_name="$(workload_run_dir_name "${seq_len}")"
  seq_root="${job_root}/${run_dir_name}"
  source_profile="${seq_root}/source_profile.json"
  lf_out="${seq_root}/lf_run"
  log_file="${seq_root}/train.log"
  if [[ "${materialize_source_from_nsys}" == "true" ]]; then
    source_materialized_job_root="$(job_root_path "${config_root}" "${backend}" "source" "${recompute_artifact_label}" "${expert_policy}" "${router_mode}" "${liger_loss}" "${grad_offload}" "${weight_offload}")"
    source_materialized_seq_root="${source_materialized_job_root}/${run_dir_name}"
    source_materialized_source_profile="${source_materialized_seq_root}/source_profile.json"
    source_materialized_profile_json="${source_materialized_seq_root}/profile.json"
    source_materialized_log_file="${source_materialized_seq_root}/train.log"
    # Create the source sibling up front so both __nsys__ and __source__ folders are visible from
    # the start; it holds only a PENDING note until materialized from the nsys run after it finishes.
    mkdir -p "${source_materialized_seq_root}"
    printf '%s\n' "Source artifacts are materialized from the sibling nsys run after it completes." > "${source_materialized_seq_root}/PENDING.md"
  fi
  local grad_offload_run_label=""
  if cpuadam_backend_for_label "${backend}" >/dev/null; then
    grad_offload_run_label="_gradoff${grad_offload}_weightoff${weight_offload}"
  fi
  run_id="lf_${backend}_${run_profiler}_${recompute_artifact_label}_pol${expert_policy}_router${router_mode}_${expact_label}_${attnact_label}_${layeract_label}_${layergc_label}_${sdparecomp_label}_${flashattn_label}_${expact_lora_a_fwd_label}_${actrecomp_label}_${xunpack_label}_${moefg_label}_${dscatter_label}_${q3rt_label}_${liger_loss}${grad_offload_run_label}_b${PER_DEVICE_TRAIN_BATCH_SIZE}_s${seq_len}_ga${GRADIENT_ACCUMULATION_STEPS}_${lora_dropout_label_value}"
  profile_json="${seq_root}/profile.json"
  local run_log_payload
  run_log_payload="${_M_REV[${current_model_name}]:-${current_model_name}} ; ${backend}|${recompute}|${liger_loss} ; ${seq_len}|${PER_DEVICE_TRAIN_BATCH_SIZE}|${GRADIENT_ACCUMULATION_STEPS} ; ${effective_policy_tuple} ; lora=${LORA_DROPOUT}|${LORA_RANK}|${LORA_ALPHA}|${LORA_TARGET} ; flash_attn=${job_flash_attn}"
  local profile_memory_breakdown deepspeed_dir_for_profile
  local profile_backend_label job_use_asym_cpu_adamw job_asym_cpu_adamw_backend cpuadam_backend
  local master_port
  profile_backend_label="${backend}"
  job_use_asym_cpu_adamw=false
  job_asym_cpu_adamw_backend="${ASYM_CPU_ADAMW_BACKEND}"
  if cpuadam_backend="$(cpuadam_backend_for_label "${backend}")"; then
    job_use_asym_cpu_adamw=true
    job_asym_cpu_adamw_backend="${cpuadam_backend}"
  fi
  # Single knob: PROFILE_MEMORY_BREAKDOWN gates the per-module breakdown AND the attribution.
  # true/false only (no auto), and no per-backend override so all backends behave identically.
  profile_memory_breakdown="$(bool_value "${PROFILE_MEMORY_BREAKDOWN}")"
  deepspeed_dir_for_profile=""
  if [[ "${backend}" == zero* || "${backend}" == superoffload* ]]; then
    deepspeed_dir_for_profile="${DEEPSPEED_DIR}"
  fi
  if [[ "${job_use_asym_cpu_adamw}" == "true" && "${job_asym_cpu_adamw_backend}" == "deepspeed" ]]; then
    deepspeed_dir_for_profile="${DEEPSPEED_DIR}"
  fi
  master_port="${MASTER_PORT:-}"
  if [[ -z "${master_port}" ]]; then
    master_port="$(find_free_port)"
  fi

  plot_roots["${config_root}"]="${seq_len}"
  if [[ "${profile_memory_breakdown}" == "true" ]]; then
    memory_plot_roots["${config_root}"]="${seq_len}"
  fi
  if [[ "${run_profiler}" == "nsys" ]]; then
    interconnect_plot_roots["${config_root}"]="${seq_len}"
    utilization_plot_roots["${config_root}"]="${seq_len}"
  fi
  if [[ "${run_profiler}" == "source" || "${materialize_source_from_nsys}" == "true" ]]; then
    throughput_plot_roots["${config_root}"]="${seq_len}"
  fi
  if [[ "${backend}" == "kt_armbf16" ]]; then
    check_kt_arm_route_rank_for_sweep "${seq_len}"
    if [[ "${run_profiler}" == "nsys" ]]; then
      kt_arm_source_ok_profile_json="$(kt_arm_resolve_matching_source_profile_json "${config_root}" "${backend}" "${recompute}" "${expert_policy}" "${router_mode}" "${liger_loss}" "${seq_len}" "${current_model_name}")"
    fi
  fi
  if [[ "${DRY_RUN}" != "true" && -e "${profile_json}" && "${OVERWRITE}" != "true" && "${COLLECT_EXISTING}" != "true" ]]; then
    if job_profile_complete "${profile_json}" "${seq_root}" "${backend}" "${seq_len}" "${current_model_name}" "${recompute}" "${ASYM_OFFLOAD_MODULES}" "${ASYMM_EXPERT_ACT_OFFLOAD}" "${ASYMM_ATTN_ACT_OFFLOAD}" "${ASYMM_LAYER_ACT_OFFLOAD}" "${ASYMM_LAYER_GC}" "${lf_expert_lora_impl}" "${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}" "${grad_offload}" "${profile_memory_breakdown}" "${liger_loss}" && {
      [[ "${materialize_source_from_nsys}" != "true" ]] ||
        job_profile_complete "${source_materialized_profile_json}" "${source_materialized_seq_root}" "${backend}" "${seq_len}" "${current_model_name}" "${recompute}" "${ASYM_OFFLOAD_MODULES}" "${ASYMM_EXPERT_ACT_OFFLOAD}" "${ASYMM_ATTN_ACT_OFFLOAD}" "${ASYMM_LAYER_ACT_OFFLOAD}" "${ASYMM_LAYER_GC}" "${lf_expert_lora_impl}" "${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}" "${grad_offload}" "${profile_memory_breakdown}" "${liger_loss}"
    }; then
      echo "Skipping existing: ${profile_json}"
      append_runs_log_entry "SKIP" "${run_log_payload}" "reason=existing-complete profile_json=${profile_json}${run_log_inert_reason}" "${run_log_extra_tag}"
      append_job_record "${config_root}" skipped \
        "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${router_mode}" "${backend}" "${run_profiler}" "${liger_loss}" "${grad_offload}" "${seq_root}" "${profile_json}" "${log_file}" "${lf_expert_lora_impl}"
      if [[ "${materialize_source_from_nsys}" == "true" ]]; then
        append_job_record "${config_root}" skipped \
          "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${router_mode}" "${backend}" source "${liger_loss}" "${grad_offload}" "${source_materialized_seq_root}" "${source_materialized_profile_json}" "${source_materialized_log_file}" "${lf_expert_lora_impl}"
      fi
      return 0
    fi
    if [[ "${materialize_source_from_nsys}" == "true" ]] &&
      job_profile_complete "${profile_json}" "${seq_root}" "${backend}" "${seq_len}" "${current_model_name}" "${recompute}" "${ASYM_OFFLOAD_MODULES}" "${ASYMM_EXPERT_ACT_OFFLOAD}" "${ASYMM_ATTN_ACT_OFFLOAD}" "${ASYMM_LAYER_ACT_OFFLOAD}" "${ASYMM_LAYER_GC}" "${lf_expert_lora_impl}" "${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}" "${grad_offload}" "${profile_memory_breakdown}" "${liger_loss}"; then
      echo "Existing nsys profile is complete; materializing missing/stale source artifacts: ${source_materialized_profile_json}" >&2
      append_runs_log_entry "SKIP" "${run_log_payload}" "reason=existing-nsys-complete materializing_source_profile=${source_materialized_profile_json}${run_log_inert_reason}" "${run_log_extra_tag}"
      postprocess_existing_nsys_artifacts "${seq_root}" "${source_profile}" "${profile_json}" || return $?
      materialize_source_artifacts_from_nsys \
        "${source_profile}" \
        "${source_materialized_seq_root}" \
        "${source_materialized_source_profile}" \
        "${source_materialized_profile_json}" \
        "${source_materialized_log_file}" \
        "${seq_root}" || return $?
      if ! job_profile_complete "${source_materialized_profile_json}" "${source_materialized_seq_root}" "${backend}" "${seq_len}" "${current_model_name}" "${recompute}" "${ASYM_OFFLOAD_MODULES}" "${ASYMM_EXPERT_ACT_OFFLOAD}" "${ASYMM_ATTN_ACT_OFFLOAD}" "${ASYMM_LAYER_ACT_OFFLOAD}" "${ASYMM_LAYER_GC}" "${lf_expert_lora_impl}" "${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}" "${grad_offload}" "${profile_memory_breakdown}" "${liger_loss}"; then
        echo "Materialized source profile is incomplete or partial: ${source_materialized_profile_json}" >&2
        return 1
      fi
      append_job_record "${config_root}" skipped \
        "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${router_mode}" "${backend}" "${run_profiler}" "${liger_loss}" "${grad_offload}" "${seq_root}" "${profile_json}" "${log_file}" "${lf_expert_lora_impl}"
      append_job_record "${config_root}" ok \
        "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${router_mode}" "${backend}" source "${liger_loss}" "${grad_offload}" "${source_materialized_seq_root}" "${source_materialized_profile_json}" "${source_materialized_log_file}" "${lf_expert_lora_impl}"
      if profiler_selected_for_plots source; then
        plot_single_run "${config_root}" "${seq_len}" "${backend}" source "${recompute_plot_filter}" "${expert_policy}" "${router_mode}" "${liger_loss}" "${source_materialized_seq_root}"
      fi
      if [[ "${profile_memory_breakdown}" == "true" ]]; then
        plot_memory_single_run "${source_materialized_seq_root}"
      fi
      plot_throughput_single_run "${source_materialized_seq_root}"
      return 0
    else
      echo "Existing profile is incomplete, partial, or has missing/stale schema-v2 source-memory breakdown; rerunning: ${profile_json}" >&2
    fi
  fi

  if [[ "${DRY_RUN}" != "true" && "${COLLECT_EXISTING}" == "true" ]]; then
    if [[ -e "${profile_json}" ]]; then
      if ! job_profile_complete "${profile_json}" "${seq_root}" "${backend}" "${seq_len}" "${current_model_name}" "${recompute}" "${ASYM_OFFLOAD_MODULES}" "${ASYMM_EXPERT_ACT_OFFLOAD}" "${ASYMM_ATTN_ACT_OFFLOAD}" "${ASYMM_LAYER_ACT_OFFLOAD}" "${ASYMM_LAYER_GC}" "${lf_expert_lora_impl}" "${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}" "${grad_offload}" "${profile_memory_breakdown}" "${liger_loss}"; then
        echo "Existing profile is incomplete or partial: ${profile_json}" >&2
        return 1
      fi
      if [[ "${materialize_source_from_nsys}" == "true" ]] &&
        ! job_profile_complete "${source_materialized_profile_json}" "${source_materialized_seq_root}" "${backend}" "${seq_len}" "${current_model_name}" "${recompute}" "${ASYM_OFFLOAD_MODULES}" "${ASYMM_EXPERT_ACT_OFFLOAD}" "${ASYMM_ATTN_ACT_OFFLOAD}" "${ASYMM_LAYER_ACT_OFFLOAD}" "${ASYMM_LAYER_GC}" "${lf_expert_lora_impl}" "${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}" "${grad_offload}" "${profile_memory_breakdown}" "${liger_loss}"; then
        materialize_source_artifacts_from_nsys \
          "${source_profile}" \
          "${source_materialized_seq_root}" \
          "${source_materialized_source_profile}" \
          "${source_materialized_profile_json}" \
          "${source_materialized_log_file}" \
          "${seq_root}" || return $?
        job_profile_complete "${source_materialized_profile_json}" "${source_materialized_seq_root}" "${backend}" "${seq_len}" "${current_model_name}" "${recompute}" "${ASYM_OFFLOAD_MODULES}" "${ASYMM_EXPERT_ACT_OFFLOAD}" "${ASYMM_ATTN_ACT_OFFLOAD}" "${ASYMM_LAYER_ACT_OFFLOAD}" "${ASYMM_LAYER_GC}" "${lf_expert_lora_impl}" "${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}" "${grad_offload}" "${profile_memory_breakdown}" "${liger_loss}" || return 1
      fi
      echo "Found existing: ${profile_json}"
      append_runs_log_entry "SKIP" "${run_log_payload}" "reason=collect-existing-found profile_json=${profile_json}${run_log_inert_reason}" "${run_log_extra_tag}"
      if [[ "${run_profiler}" == "nsys" ]]; then
        postprocess_existing_nsys_artifacts "${seq_root}" "${source_profile}" "${profile_json}" || return $?
      fi
      if profiler_selected_for_plots "${run_profiler}"; then
        plot_single_run "${config_root}" "${seq_len}" "${backend}" "${run_profiler}" "${recompute_plot_filter}" "${expert_policy}" "${router_mode}" "${liger_loss}" "${seq_root}"
      fi
      if [[ "${materialize_source_from_nsys}" == "true" ]] && profiler_selected_for_plots source; then
        plot_single_run "${config_root}" "${seq_len}" "${backend}" source "${recompute_plot_filter}" "${expert_policy}" "${router_mode}" "${liger_loss}" "${source_materialized_seq_root}"
      fi
      if [[ "${profile_memory_breakdown}" == "true" ]]; then
        if [[ "${materialize_source_from_nsys}" == "true" ]]; then
          plot_memory_single_run "${source_materialized_seq_root}"
          plot_throughput_single_run "${source_materialized_seq_root}"
        elif [[ "${run_profiler}" == "source" ]]; then
          plot_memory_single_run "${seq_root}"
          plot_throughput_single_run "${seq_root}"
        fi
      elif [[ "${materialize_source_from_nsys}" == "true" ]]; then
        plot_throughput_single_run "${source_materialized_seq_root}"
      elif [[ "${run_profiler}" == "source" ]]; then
        plot_throughput_single_run "${seq_root}"
      fi
      return 0
    fi
    echo "Missing existing profile: ${profile_json}" >&2
    append_runs_log_entry "SKIP" "${run_log_payload}" "reason=collect-existing-missing profile_json=${profile_json}${run_log_inert_reason}" "${run_log_extra_tag}"
    return 1
  fi

  local -a run_env=(
    ROOT="${ROOT}"
    NUMACTL_MODE="${NUMACTL_MODE:-membind}"
    ASYM_OFFLOAD_ACT_RECOMPUTE="${ASYM_OFFLOAD_ACT_RECOMPUTE:-0}"
    ASYM_OFFLOAD_X_UNPACKED="${ASYM_OFFLOAD_X_UNPACKED:-0}"
	    ASYMM_EXPERT_SILU_BWD_GPU="${ASYMM_EXPERT_SILU_BWD_GPU}"
	    ASYMM_MLP_RECOMPUTE_CHUNK="${ASYMM_MLP_RECOMPUTE_CHUNK}"
	    ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT="${ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT:-0}"
	    ASYMM_DENSE_MLP_FINEGRAINED_NOGRAD_CPU_OFFLOAD="${ASYMM_DENSE_MLP_FINEGRAINED_NOGRAD_CPU_OFFLOAD:-0}"
	    DG_BF16_CPU_LEFT_COMPACT_GRID="${DG_BF16_CPU_LEFT_COMPACT_GRID:-0}"
    ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE="${ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE:-0}"
    ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU="${ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU:-}"
    ASYMM_QWEN3_MOE_FG_DA_GPU="${ASYMM_QWEN3_MOE_FG_DA_GPU:-}"
    ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM="${ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM:-}"
    ASYM_EXPACT_CPU_POOL_MAX_BYTES="${ASYM_EXPACT_CPU_POOL_MAX_BYTES:-}"
    LF_DIR="${job_lf_dir}"
    ASYM_DIR="${ASYM_DIR}"
    ENV_DIR="${job_env_dir}"
    CONDA_EXE="${CONDA_EXE}"
    NSYS_BIN="${NSYS_BIN}"
    EXPANDABLE_SEG="${EXPANDABLE_SEG}"
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_EFFECTIVE}"
    DIST_LAUNCHER="${DIST_LAUNCHER}"
    DEEPSPEED_DIR="${DEEPSPEED_DIR}"
    FLASH_ATTN="${job_flash_attn}"
    CHECK_SUPEROFFLOAD="${CHECK_SUPEROFFLOAD}"
    CHECK_CPUADAM="${CHECK_CPUADAM}"
    MODEL_NAME_OR_PATH="${current_model_name}"
    BACKEND="${profile_backend_label}"
    USE_ASYM_CPU_ADAMW="${job_use_asym_cpu_adamw}"
    ASYM_CPU_ADAMW_BACKEND="${job_asym_cpu_adamw_backend}"
    ASYM_CPU_ADAMW_PIN_MEMORY="${ASYM_CPU_ADAMW_PIN_MEMORY}"
    ASYM_CPU_ADAMW_FP32_MASTER="${ASYM_CPU_ADAMW_FP32_MASTER}"
    ASYM_CPU_ADAMW_GRAD_OFFLOAD="${grad_offload}"
    ASYM_CPU_ADAMW_WEIGHT_OFFLOAD="${weight_offload}"
    GPU_ID="${gpu}"
    NUM_GPUS="${gpu_count}"
    REQUIRE_SM100="${REQUIRE_SM100}"
    DATASET="${dataset_name}"
    TEMPLATE="${TEMPLATE}"
    CUTOFF_LEN="${seq_len}"
    MAX_SAMPLES="${MAX_SAMPLES}"
    MAX_STEPS="${TOTAL_STEPS}"
    PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}"
    LEARNING_RATE="${LEARNING_RATE}"
    LORA_RANK="${LORA_RANK}"
    LORA_ALPHA="${LORA_ALPHA}"
    LORA_DROPOUT="${LORA_DROPOUT}"
    LORA_TARGET="${LORA_TARGET}"
    LF_QWEN_MOE_EXPERT_LORA_IMPL="${lf_expert_lora_impl}"
    ENABLE_LIGER_KERNEL="${enable_liger_kernel}"
    SEED="${SEED}"
    GRADIENT_CHECKPOINTING="${gradient_checkpointing}"
    USE_UNSLOTH_GC="${use_unsloth_gc}"
    UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU="${unsloth_recompute_save_on_cpu}"
    UNSLOTH_GC_OUTER_HBM_EVERY_N="${UNSLOTH_GC_OUTER_HBM_EVERY_N}"
    ASYM_PRECISION="${PRECISION}"
    ASYM_OFFLOAD_MODULES="${ASYM_OFFLOAD_MODULES}"
    ASYMM_EXPERT_ACT_OFFLOAD="${ASYMM_EXPERT_ACT_OFFLOAD}"
    ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD="${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}"
    ASYMM_DENSE_MLP_SURGICAL_OFFLOAD="${dense_mlp_surgical}"
    ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD="${dense_mlp_finegrained}"
    ASYMM_DENSE_MLP_FINEGRAINED_NOGRAD_CPU_OFFLOAD="${ASYMM_DENSE_MLP_FINEGRAINED_NOGRAD_CPU_OFFLOAD:-0}"
    ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD="${ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD}"
    ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS="${ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS}"
    ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM="${ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM}"
    ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER="${q3rt_fwd_flag}"
    ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER="${q3rt_gather_flag}"
    ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER="${q3rt_dx_flag}"
    ASYMM_QWEN3_MOE_ROUTE_LORA="${q3rt_lora_flag}"
    ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE="${ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE}"
    ASYMM_QWEN3_MOE_ROUTE_KERNEL_DEBUG="${ASYMM_QWEN3_MOE_ROUTE_KERNEL_DEBUG}"
    ASYMM_ATTN_ACT_OFFLOAD="${ASYMM_ATTN_ACT_OFFLOAD}"
    ASYMM_LAYER_ACT_OFFLOAD="${ASYMM_LAYER_ACT_OFFLOAD}"
    ASYMM_LAYER_GC="${ASYMM_LAYER_GC}"
    ASYMM_ATTN_SDPA_RECOMPUTE="${ASYMM_ATTN_SDPA_RECOMPUTE}"
    ASYM_EXPERT_RECOMPUTE_POLICY="${expert_policy}"
    ASYM_ROUTER_MODE="${router_mode}"
    ASYM_STRICT="${ASYM_STRICT}"
    PROFILE=1
    PROFILE_PROFILER="${run_profiler}"
    PROFILE_LEVEL="${PROFILE_LEVEL}"
    PROFILE_LAYERS="${PROFILE_LAYERS}"
    PROFILE_MEMORY_BREAKDOWN="${profile_memory_breakdown}"
    PROFILE_MEMORY_BREAKDOWN_INTERVAL="${PROFILE_MEMORY_BREAKDOWN_INTERVAL}"
    PROFILE_MEMORY_BREAKDOWN_STEPS="${PROFILE_MEMORY_BREAKDOWN_STEPS}"
    PROFILE_MEMORY_BREAKDOWN_MODULES="${PROFILE_MEMORY_BREAKDOWN_MODULES}"
    PROFILE_LIVE_ACTIVATION_DETAILS="${PROFILE_LIVE_ACTIVATION_DETAILS}"
    PROFILE_LIVE_ACTIVATION_TOPK="${PROFILE_LIVE_ACTIVATION_TOPK}"
    PROFILE_MEMORY_SNAPSHOT="${PROFILE_MEMORY_SNAPSHOT}"
    PROFILE_MEMORY_SNAPSHOT_PATH="${PROFILE_MEMORY_SNAPSHOT_PATH}"
    PROFILE_EXTERNAL_MEMORY="${PROFILE_EXTERNAL_MEMORY}"
    PROFILE_SYNC="${PROFILE_SYNC}"
    PROFILE_MODULE_FILTER="${PROFILE_MODULE_FILTER}"
    PROFILE_SOURCE_JSON="${source_profile}"
    PROFILE_NSYS_PREFIX="${seq_root}/trace"
    PROFILE_NSYS_SQLITE="${seq_root}/trace.sqlite"
    PROFILE_NSYS_GPU_METRICS_DEVICES="${gpu}"
    PROFILE_JSON="${profile_json}"
    PROFILE_OUTPUT_DIR="${seq_root}"
    PROFILE_SUMMARY_MD="${seq_root}/summary.md"
    PROFILE_WORKLOAD_LABEL="${workload_label}"
    PROFILE_BACKEND_LABEL="${profile_backend_label}"
    PROFILE_EXPERT_POLICY="${expert_policy}"
    INTERRUPT_GRACE_SECONDS="${INTERRUPT_GRACE_SECONDS}"
    PROFILE_WARMUP_STEPS="${WARMUP_STEPS}"
    PROFILE_MEASURE_STEPS="${MAX_STEPS}"
    PROFILE_TOTAL_STEPS="${TOTAL_STEPS}"
    MASTER_PORT="${master_port}"
    ASYM_GEMM_LF_CONFIG_ASYM_STRICT="${ASYM_STRICT}"
    ASYM_GEMM_LF_CONFIG_LIGER_LOSS="${liger_loss}"
    ASYM_GEMM_LF_CONFIG_RECOMP_LABEL="${recompute_artifact_label}"
    ASYM_GEMM_LF_CONFIG_RECOMP_OFF_STAGE="${recomp_off_stage}"
    ASYM_GEMM_LF_CONFIG_USE_UNSLOTH_GC="${use_unsloth_gc}"
    ASYM_GEMM_LF_CONFIG_UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU="${unsloth_recompute_save_on_cpu}"
    ASYM_GEMM_LF_CONFIG_UNSLOTH_GC_OUTER_HBM_EVERY_N="${UNSLOTH_GC_OUTER_HBM_EVERY_N}"
    ASYM_GEMM_LF_CONFIG_ASYMM_EXPERT_SILU_BWD_GPU="${ASYMM_EXPERT_SILU_BWD_GPU}"
    ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_GRAD_OFFLOAD="${grad_offload}"
    ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_WEIGHT_OFFLOAD="${weight_offload}"
    ASYM_GEMM_LF_CONFIG_ASYMM_EXPERT_ACT_OFFLOAD="${ASYMM_EXPERT_ACT_OFFLOAD}"
    ASYM_GEMM_LF_CONFIG_ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD="${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}"
	    ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_SURGICAL_OFFLOAD="${dense_mlp_surgical}"
	    ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD="${dense_mlp_finegrained}"
	    ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT="${ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT:-0}"
	    ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_FINEGRAINED_NOGRAD_CPU_OFFLOAD="${ASYMM_DENSE_MLP_FINEGRAINED_NOGRAD_CPU_OFFLOAD:-0}"
	    ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD="${ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD}"
	    ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU="${ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU:-}"
	    ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_FG_DA_GPU="${ASYMM_QWEN3_MOE_FG_DA_GPU:-}"
	    ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM="${ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM:-}"
	    ASYM_GEMM_LF_CONFIG_ASYM_EXPACT_CPU_POOL_MAX_BYTES="${ASYM_EXPACT_CPU_POOL_MAX_BYTES:-}"
	    ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS="${ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS}"
	    ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM="${ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM}"
	    ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER="${q3rt_fwd_flag}"
	    ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER="${q3rt_gather_flag}"
	    ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER="${q3rt_dx_flag}"
	    ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_ROUTE_LORA="${q3rt_lora_flag}"
	    ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE="${ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE}"
	    ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_ROUTE_KERNEL_CODE="${q3rt_kernel_code}"
	    ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_ROUTE_KERNEL_DEBUG="${ASYMM_QWEN3_MOE_ROUTE_KERNEL_DEBUG}"
	    ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_ACT_OFFLOAD="${ASYMM_ATTN_ACT_OFFLOAD}"
    ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_ACT_OFFLOAD="${ASYMM_LAYER_ACT_OFFLOAD}"
    ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_GC="${ASYMM_LAYER_GC}"
    ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_SDPA_RECOMPUTE="${ASYMM_ATTN_SDPA_RECOMPUTE}"
    ASYM_GEMM_LF_CONFIG_ASYM_OFFLOAD_ACT_RECOMPUTE="${ASYM_OFFLOAD_ACT_RECOMPUTE}"
    ASYM_GEMM_LF_CONFIG_ASYM_OFFLOAD_X_UNPACKED="${ASYM_OFFLOAD_X_UNPACKED}"
    ASYM_GEMM_LF_CONFIG_ASYMM_MLP_RECOMPUTE_CHUNK="${ASYMM_MLP_RECOMPUTE_CHUNK}"
    ASYM_GEMM_LF_CONFIG_QWEN_EXPERT_LORA_IMPL="${lf_expert_lora_impl}"
    ASYM_GEMM_LF_CONFIG_SEQ_LEN="${seq_len}"
    ASYM_GEMM_LF_CONFIG_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}"
    ASYM_GEMM_LF_CONFIG_GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}"
    ASYM_GEMM_LF_CONFIG_ATTN_GC_ENABLED="${attention_gc_enabled}"
    ASYM_GEMM_LF_CONFIG_LAYER_GC_ENABLED="${layer_gc_enabled}"
    ASYM_GEMM_LF_CONFIG_WARMUP_STEPS="${WARMUP_STEPS}"
    ASYM_GEMM_LF_CONFIG_MEASURE_STEPS="${MAX_STEPS}"
    ASYM_GEMM_LF_CONFIG_TOTAL_STEPS="${TOTAL_STEPS}"
    ASYM_GEMM_LF_CONFIG_DEEPSPEED_DIR="${deepspeed_dir_for_profile}"
    ASYM_GEMM_LF_CONFIG_FLASH_ATTN="${job_flash_attn}"
    OUT_DIR="${lf_out}"
    LOG_FILE="${log_file}"
    RUN_ID="${run_id}"
  )
  if [[ "${backend}" == kt_* ]]; then
    local kt_num_threads_for_job="${KT_NUM_THREADS}"
    if [[ "${backend}" == "kt_armbf16" && -z "${kt_num_threads_for_job}" ]]; then
      kt_num_threads_for_job="${KT_ARM_OMP_NUM_THREADS}"
    fi
    run_env+=(
      KT_TOOLS_DIR="${KT_TOOLS_DIR}"
      KT_KERNEL_DIR="${KT_KERNEL_DIR}"
      KT_REPO_DIR="${KT_REPO_DIR}"
      KT_GGUF_PY_DIR="${KT_GGUF_PY_DIR}"
      KT_PRECISION="${PRECISION}"
      KT_NUM_THREADS="${kt_num_threads_for_job}"
      KT_THREADPOOL_COUNT="${KT_THREADPOOL_COUNT}"
      KT_MAX_CACHE_DEPTH="${KT_MAX_CACHE_DEPTH}"
      KT_TP_ENABLED="${KT_TP_ENABLED}"
      KT_SHARE_BACKWARD_BB="${KT_SHARE_BACKWARD_BB}"
      KT_NUM_GPU_EXPERTS="${KT_NUM_GPU_EXPERTS}"
      KT_WEIGHT_PATH="${KT_WEIGHT_PATH}"
      KT_EXPERT_CHECKPOINT_PATH="${KT_EXPERT_CHECKPOINT_PATH}"
      KT_USE_LORA_EXPERTS="${KT_USE_LORA_EXPERTS}"
      KT_LORA_EXPERT_NUM="${KT_LORA_EXPERT_NUM}"
      KT_LORA_EXPERT_INTERMEDIATE_SIZE="${KT_LORA_EXPERT_INTERMEDIATE_SIZE}"
      KT_TORCHBF16_SFT_DEVICE="${KT_TORCHBF16_SFT_DEVICE}"
      KT_ARM_OMP_NUM_THREADS="${KT_ARM_OMP_NUM_THREADS}"
      KT_ARM_OMP_PROC_BIND="${KT_ARM_OMP_PROC_BIND}"
      KT_ARM_OMP_PLACES="${KT_ARM_OMP_PLACES}"
      KT_ARM_SFT_TOP_K="${KT_ARM_SFT_TOP_K}"
      KT_ARM_SFT_TOKEN_CHUNK_SIZE="${KT_ARM_SFT_TOKEN_CHUNK_SIZE}"
      KT_ARM_SFT_MAX_ROUTE_RANK_WORK="${KT_ARM_SFT_MAX_ROUTE_RANK_WORK}"
      KT_ARM_SFT_DEFAULT_MAX_ROUTE_RANK_WORK="${KT_ARM_SFT_DEFAULT_MAX_ROUTE_RANK_WORK}"
      KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK="${KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK}"
      KT_ARM_ALLOW_NSYS_WITHOUT_SOURCE_OK="${KT_ARM_ALLOW_NSYS_WITHOUT_SOURCE_OK}"
      KT_ARM_SOURCE_OK_PROFILE_JSON="${kt_arm_source_ok_profile_json}"
      KT_ARM_FIRST_STEP_TIMEOUT_SECONDS="${KT_ARM_FIRST_STEP_TIMEOUT_SECONDS}"
      CHECK_KT_CALLS="${CHECK_KT_CALLS}"
    )
  fi

  local -a run_cmd=(env)
  run_cmd+=("${run_env[@]}" "${RUN_LF_SCRIPT}")

  echo "Running backend=${backend} profiler=${profiler} run_profiler=${run_profiler} recompute=${recompute} recompute_label=${recompute_artifact_label} liger_loss=${liger_loss} expert_policy=${expert_policy} router_mode=${router_mode} grad_offload=${grad_offload} qwen_expert_lora_impl=${lf_expert_lora_impl} flash_attn=${job_flash_attn} lora_a_fwd=${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD} ${expact_label} ${attnact_label} ${layeract_label} ${layergc_label} ${flashattn_label} ${expact_lora_a_fwd_label} ${q3rt_label} seq=${seq_len} batch=${PER_DEVICE_TRAIN_BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS} lora_dropout=${LORA_DROPOUT} gpu=${gpu} num_gpus=${gpu_count}"
  echo "  dir=${seq_root}"
  if [[ "${materialize_source_from_nsys}" == "true" ]]; then
    echo "  source_dir=${source_materialized_seq_root}"
  fi
  if [[ "${DRY_RUN}" == "true" ]]; then
    append_runs_log_entry "RUN" "${run_log_payload}" "reason=dry-run${run_log_inert_reason}" "${run_log_extra_tag}"
    print_command "${run_cmd[@]}"
    mkdir -p "${seq_root}"
    ensure_jobs_tsv "${config_root}"
    {
      print_command "${run_cmd[@]}"
    } > "${seq_root}/command.txt"
    append_job_record "${config_root}" dry-run \
      "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${router_mode}" "${backend}" "${run_profiler}" "${liger_loss}" "${grad_offload}" "${seq_root}" "${profile_json}" "${log_file}" "${lf_expert_lora_impl}"
    if [[ "${materialize_source_from_nsys}" == "true" ]]; then
      local -a source_postprocess_cmd=(
        "${ENV_PYTHON}" "${PROFILE_POSTPROCESS_SCRIPT}"
        --source-profile-json "${source_materialized_source_profile}"
        --profile-json "${source_materialized_profile_json}"
        --output-dir "${source_materialized_seq_root}"
      )
      mkdir -p "${source_materialized_seq_root}"
      {
        echo "Dry-run source materialization from nsys source profile:"
        echo "  nsys_source_profile=${source_profile}"
        print_command cp "${source_profile}" "${source_materialized_source_profile}"
        print_command "${source_postprocess_cmd[@]}"
      } > "${source_materialized_seq_root}/command.txt"
      append_job_record "${config_root}" dry-run \
        "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${router_mode}" "${backend}" source "${liger_loss}" "${grad_offload}" "${source_materialized_seq_root}" "${source_materialized_profile_json}" "${source_materialized_log_file}" "${lf_expert_lora_impl}"
    fi
    return 0
  fi

  mkdir -p "${seq_root}"
  ensure_jobs_tsv "${config_root}"
  {
    print_command "${run_cmd[@]}"
  } > "${seq_root}/command.txt"

  local status=0
  append_runs_log_entry "RUN" "${run_log_payload}" "${run_log_inert_reason# }" "${run_log_extra_tag}"
  run_tracked_command "${run_cmd[@]}" || status=$?
  if [[ "${interrupted}" == "true" ]]; then
    echo "Interrupted run; exiting without scheduling more jobs." >&2
    exit "${interrupt_exit_status}"
  fi
  if [[ "${status}" == "130" || "${status}" == "143" ]]; then
    echo "Interrupted run; exiting without scheduling more jobs." >&2
    exit "${status}"
  fi
  if ((status == 0)); then
    if [[ ! -f "${profile_json}" ]]; then
      echo "Missing expected profile artifact: ${profile_json}" >&2
      status=1
    elif ! job_profile_complete "${profile_json}" "${seq_root}" "${backend}" "${seq_len}" "${current_model_name}" "${recompute}" "${ASYM_OFFLOAD_MODULES}" "${ASYMM_EXPERT_ACT_OFFLOAD}" "${ASYMM_ATTN_ACT_OFFLOAD}" "${ASYMM_LAYER_ACT_OFFLOAD}" "${ASYMM_LAYER_GC}" "${lf_expert_lora_impl}" "${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}" "${grad_offload}" "${profile_memory_breakdown}" "${liger_loss}"; then
      echo "Expected completed profile artifact but found incomplete/partial profile: ${profile_json}" >&2
      status=1
    fi
  fi
  if ((status == 0)) && [[ "${materialize_source_from_nsys}" == "true" ]]; then
    materialize_source_artifacts_from_nsys \
      "${source_profile}" \
      "${source_materialized_seq_root}" \
      "${source_materialized_source_profile}" \
      "${source_materialized_profile_json}" \
      "${source_materialized_log_file}" \
      "${seq_root}" || status=$?
    if ((status == 0)) &&
      ! job_profile_complete "${source_materialized_profile_json}" "${source_materialized_seq_root}" "${backend}" "${seq_len}" "${current_model_name}" "${recompute}" "${ASYM_OFFLOAD_MODULES}" "${ASYMM_EXPERT_ACT_OFFLOAD}" "${ASYMM_ATTN_ACT_OFFLOAD}" "${ASYMM_LAYER_ACT_OFFLOAD}" "${ASYMM_LAYER_GC}" "${lf_expert_lora_impl}" "${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}" "${grad_offload}" "${profile_memory_breakdown}" "${liger_loss}"; then
      echo "Expected completed materialized source profile but found incomplete/partial profile: ${source_materialized_profile_json}" >&2
      status=1
    fi
  fi

  if ((status == 0)); then
    append_job_record "${config_root}" ok \
      "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${router_mode}" "${backend}" "${run_profiler}" "${liger_loss}" "${grad_offload}" "${seq_root}" "${profile_json}" "${log_file}" "${lf_expert_lora_impl}"
    if [[ "${materialize_source_from_nsys}" == "true" ]]; then
      append_job_record "${config_root}" ok \
        "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${router_mode}" "${backend}" source "${liger_loss}" "${grad_offload}" "${source_materialized_seq_root}" "${source_materialized_profile_json}" "${source_materialized_log_file}" "${lf_expert_lora_impl}"
    fi
    if profiler_selected_for_plots "${run_profiler}"; then
      plot_single_run "${config_root}" "${seq_len}" "${backend}" "${run_profiler}" "${recompute_plot_filter}" "${expert_policy}" "${router_mode}" "${liger_loss}" "${seq_root}"
    fi
    if [[ "${materialize_source_from_nsys}" == "true" ]] && profiler_selected_for_plots source; then
      plot_single_run "${config_root}" "${seq_len}" "${backend}" source "${recompute_plot_filter}" "${expert_policy}" "${router_mode}" "${liger_loss}" "${source_materialized_seq_root}"
    fi
    if [[ "${profile_memory_breakdown}" == "true" ]]; then
      if [[ "${materialize_source_from_nsys}" == "true" ]]; then
        plot_memory_single_run "${source_materialized_seq_root}"
        plot_throughput_single_run "${source_materialized_seq_root}"
      else
        plot_memory_single_run "${seq_root}"
        plot_throughput_single_run "${seq_root}"
      fi
    elif [[ "${materialize_source_from_nsys}" == "true" ]]; then
      plot_throughput_single_run "${source_materialized_seq_root}"
    elif [[ "${run_profiler}" == "source" ]]; then
      plot_throughput_single_run "${seq_root}"
    fi
  else
    append_job_record "${config_root}" "failed:${status}" \
      "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${router_mode}" "${backend}" "${run_profiler}" "${liger_loss}" "${grad_offload}" "${seq_root}" "${profile_json}" "${log_file}" "${lf_expert_lora_impl}"
    if [[ "${materialize_source_from_nsys}" == "true" ]]; then
      append_job_record "${config_root}" "failed:${status}" \
        "${gpu}" "${seq_len}" "${recompute}" "${expert_policy}" "${router_mode}" "${backend}" source "${liger_loss}" "${grad_offload}" "${source_materialized_seq_root}" "${source_materialized_profile_json}" "${source_materialized_log_file}" "${lf_expert_lora_impl}"
    fi
  fi
  return "${status}"
}

plot_config_root() {
  local config_root="$1"
  local seq_len="$2"
  local plot_root
  [[ "${PLOT}" == "true" ]] || return 0
  ((${#plot_profilers[@]})) || return 0

  cleanup_legacy_combined_artifacts "${config_root}"
  plot_root="$(config_combined_metric_dir "${config_root}" timing)"
  local -a plot_cmd
  plot_cmd_base plot_cmd "${config_root}" "${plot_root}" "${plot_root}" "$(current_workload_tuple "${seq_len}")"
  # Cross-config comparison only; per-run timing plots live in each leaf's metrics/timing dir.
  plot_cmd+=(--combined-only)
  append_sweep_plot_filters plot_cmd
  echo "Writing LF config combined plots: ${plot_root}"
  run_tracked_command "${plot_cmd[@]}"
}

plot_single_run() {
  local config_root="$1"
  local seq_len="$2"
  local backend="$3"
  local profiler="$4"
  local recompute="$5"
  local expert_policy="$6"
  local router_mode="$7"
  local liger_loss="$8"
  local seq_root="$9"
  local plot_root
  [[ "${PLOT}" == "true" ]] || return 0
  ((${#plot_profilers[@]})) || return 0

  cleanup_legacy_run_artifacts "${seq_root}"
  plot_root="$(run_metric_dir "${seq_root}" timing)"
  local -a plot_cmd
  plot_cmd_base plot_cmd "${config_root}" "${plot_root}" "${plot_root}/combined" "$(current_workload_tuple "${seq_len}")"
  plot_cmd+=(
    --skip-combined
    --expert-recompute-policies "${expert_policy}"
    --backend "${backend}"
    --profiler "${profiler}"
    --recompute "${recompute}"
    --liger-loss "${liger_loss}"
    --router-mode "${router_mode}"
    --expact "${expact_label}"
    --attnact "${attnact_label}"
    --layeract "${layeract_label}"
    --layergc "${layergc_label}"
  )
  echo "Writing LF per-run plots: ${plot_root}"
  if ! run_tracked_command "${plot_cmd[@]}"; then
    echo "warning: failed to write per-run plots for ${seq_root}" >&2
  fi
}

plot_memory_single_run() {
  local seq_root="$1"
  local plot_root
  [[ "${PLOT}" == "true" && "${PLOT_MEMORY_BREAKDOWN}" == "true" ]] || return 0

  cleanup_legacy_run_artifacts "${seq_root}"
  plot_root="$(run_metric_dir "${seq_root}" memory)"
  local -a plot_cmd=(
    "${ENV_PYTHON}" "${MEMORY_PLOT_SCRIPT}"
    --run-dir "${seq_root}"
    --output-dir "${plot_root}"
    --clean-output
    --y-scale "${MEMORY_BREAKDOWN_PLOT_Y_SCALE}"
  )
  echo "Writing LF source memory plots: ${plot_root}"
  if ! run_tracked_command "${plot_cmd[@]}"; then
    echo "warning: failed to write source memory plots for ${seq_root}" >&2
  fi
}

plot_throughput_single_run() {
  local seq_root="$1"
  local plot_root
  [[ "${PLOT}" == "true" ]] || return 0

  cleanup_legacy_run_artifacts "${seq_root}"
  plot_root="$(run_metric_dir "${seq_root}" throughput)"
  local -a plot_cmd=(
    "${ENV_PYTHON}" "${THROUGHPUT_PLOT_SCRIPT}"
    --run-dir "${seq_root}"
    --output-dir "${plot_root}"
    --clean-output
    --combined-only
  )
  echo "Writing LF source throughput plots: ${plot_root}"
  if ! run_tracked_command "${plot_cmd[@]}"; then
    echo "warning: failed to write source throughput plots for ${seq_root}" >&2
  fi
}

plot_memory_config_root() {
  local config_root="$1"
  local seq_len="$2"
  local plot_root
  [[ "${PLOT}" == "true" && "${PLOT_MEMORY_BREAKDOWN}" == "true" ]] || return 0

  cleanup_legacy_combined_artifacts "${config_root}"
  plot_root="$(config_combined_metric_dir "${config_root}" memory)"
  local -a plot_cmd
  memory_metric_plot_cmd_base plot_cmd "${config_root}" "${plot_root}" "$(current_workload_tuple "${seq_len}")"
  memory_plot_filters plot_cmd
  echo "Writing LF source-memory combined plots: ${plot_root}"
  if ! run_tracked_command "${plot_cmd[@]}"; then
    echo "warning: failed to write source-memory combined plots for ${config_root}" >&2
  fi
}

write_missing_combined_readme() {
  local output_dir="$1"
  local title="$2"
  local reason="$3"
  mkdir -p "${output_dir}/data"
  cat > "${output_dir}/README.md" <<EOF
# ${title}

No plots were generated here.

${reason}
EOF
}

write_config_artifact_readme() {
  local config_root="$1"
  mkdir -p "${config_root}"
  cat > "${config_root}/ARTIFACTS.md" <<EOF
# LF Profiling Artifacts

This config root is organized as follows:

- \`combined/timing/\`: config-level LF timing and allocator-summary plots from \`profile.json\`.
- \`combined/memory/\`: config-level source-memory breakdown plots. If no source-memory rows were collected, this folder contains a README explaining why.
- \`combined/interconnect/\`: config-level C2C/CTC saturation plots. If old traces lack GPU metrics, this folder contains a README explaining why.
- \`combined/utilization/\`: config-level GPU/CPU utilization summaries from nsys runs.
- \`combined/throughput/\`: config-level training throughput (tokens/sec) line plots from source-profiler runs.
- \`<backend>__<profiler>__<recompute>__pol<policy>__router<mode>__...__ligerloss0/b<batch>_s<seq>_ga<grad_accum>/\`: per-run artifacts.

If \`PLOT_OUTPUT_DIR\` is set, combined plot folders are written under that external plot output root instead of this config root.

Per-run metric plots live directly under \`metrics/<metric>/\`, with tabular data under \`metrics/<metric>/data/\`. Source and nsys runs remain separate folders and only contain metrics that profiler can produce.
EOF
}

write_precision_artifact_readme() {
  local precision_root="$1"
  mkdir -p "${precision_root}"
  cat > "${precision_root}/ARTIFACTS.md" <<EOF
# LF Profiling Precision Root

This precision root is organized as follows:

- \`combined/timing/\`: global LF timing and allocator-summary plots across config roots.
- \`combined/memory/\`: global source-memory breakdown plots across config roots.
- \`combined/interconnect/\`: global C2C/CTC saturation plots across config roots.
- \`combined/utilization/\`: global GPU/CPU utilization summaries from nsys runs.
- \`combined/throughput/\`: global training throughput (tokens/sec) line plots across config roots.
- \`<config_root>/\`: one workload/configuration root. Each config root has its own \`combined/<metric>/\` folders and per-run backend/profiler folders.

If \`PLOT_OUTPUT_DIR\` is set, global combined plot folders are written under that external plot output root instead of this precision root.

Fresh nsys runs collect C2C GPU metrics at 100 Hz. Existing traces created without Nsight \`GPU_METRICS\` tables cannot be converted into C2C saturation plots.
EOF
}

plot_interconnect_config_root() {
  local config_root="$1"
  local seq_len="$2"
  local plot_root
  [[ "${PLOT}" == "true" ]] || return 0

  cleanup_legacy_combined_artifacts "${config_root}"
  plot_root="$(config_combined_metric_dir "${config_root}" interconnect)"
  local -a plot_cmd
  interconnect_metric_plot_cmd_base plot_cmd "${config_root}" "${plot_root}" "$(current_workload_tuple "${seq_len}")"
  interconnect_plot_filters plot_cmd
  echo "Writing LF C2C combined plots: ${plot_root}"
  if ! run_tracked_command "${plot_cmd[@]}"; then
    echo "warning: failed to write C2C combined plots for ${config_root}" >&2
  fi
}

plot_utilization_config_root() {
  local config_root="$1"
  local seq_len="$2"
  local plot_root
  [[ "${PLOT}" == "true" ]] || return 0

  cleanup_legacy_combined_artifacts "${config_root}"
  plot_root="$(config_combined_metric_dir "${config_root}" utilization)"
  local -a plot_cmd
  utilization_metric_plot_cmd_base plot_cmd "${config_root}" "${plot_root}" "$(current_workload_tuple "${seq_len}")"
  utilization_plot_filters plot_cmd
  echo "Writing LF utilization combined plots: ${plot_root}"
  if ! run_tracked_command "${plot_cmd[@]}"; then
    echo "warning: failed to write utilization combined plots for ${config_root}" >&2
  fi
}

plot_throughput_config_root() {
  local config_root="$1"
  local seq_len="$2"
  local plot_root
  [[ "${PLOT}" == "true" ]] || return 0

  cleanup_legacy_combined_artifacts "${config_root}"
  plot_root="$(config_combined_metric_dir "${config_root}" throughput)"
  local -a plot_cmd
  throughput_metric_plot_cmd_base plot_cmd "${config_root}" "${plot_root}" "$(current_workload_tuple "${seq_len}")"
  throughput_plot_filters plot_cmd
  echo "Writing LF throughput combined plots: ${plot_root}"
  if ! run_tracked_command "${plot_cmd[@]}"; then
    echo "warning: failed to write throughput combined plots for ${config_root}" >&2
  fi
}

collect_timing_workloads_from_roots() {
  local roots_name="$1"
  local workloads_name="$2"
  local -n roots_ref="${roots_name}"
  local -n workloads_ref="${workloads_name}"
  local config_root

  workloads_ref=()
  for config_root in "${!roots_ref[@]}"; do
    workloads_ref["$(plot_workload_from_config_root "${config_root}")"]=1
  done
}

timing_precision_combined_cmd() {
  local cmd_name="$1"
  local -n _cmd_ref="${cmd_name}"
  local output_dir="$2"
  shift 2

  plot_cmd_base "${cmd_name}" "${precision_root}" "${output_dir}" "${output_dir}" "${workloads[@]}"
  _cmd_ref+=(--combined-only)
  _cmd_ref+=("$@")
}

memory_precision_combined_cmd() {
  local cmd_name="$1"
  local -n _cmd_ref="${cmd_name}"
  local output_dir="$2"
  shift 2

  memory_metric_plot_cmd_base "${cmd_name}" "${precision_root}" "${output_dir}" "${workloads[@]}"
  _cmd_ref+=("$@")
}

interconnect_precision_combined_cmd() {
  local cmd_name="$1"
  local -n _cmd_ref="${cmd_name}"
  local output_dir="$2"
  shift 2

  interconnect_metric_plot_cmd_base "${cmd_name}" "${precision_root}" "${output_dir}" "${workloads[@]}"
  _cmd_ref+=("$@")
}

utilization_precision_combined_cmd() {
  local cmd_name="$1"
  local -n _cmd_ref="${cmd_name}"
  local output_dir="$2"
  shift 2

  utilization_metric_plot_cmd_base "${cmd_name}" "${precision_root}" "${output_dir}" "${workloads[@]}"
  _cmd_ref+=("$@")
}

throughput_precision_combined_cmd() {
  local cmd_name="$1"
  local -n _cmd_ref="${cmd_name}"
  local output_dir="$2"
  shift 2

  throughput_metric_plot_cmd_base "${cmd_name}" "${precision_root}" "${output_dir}" "${workloads[@]}"
  _cmd_ref+=("$@")
}

run_precision_combined_plot() {
  local plot_root="$1"
  local build_func="$2"
  local filter_func="$3"
  local label="$4"
  local warn_on_failure="$5"
  shift 5

  local -a plot_cmd
  "${build_func}" plot_cmd "${plot_root}" "$@"
  "${filter_func}" plot_cmd
  echo "Writing combined LF ${label} plots: ${plot_root}"
  if [[ "${warn_on_failure}" == "true" ]]; then
    if ! run_tracked_command "${plot_cmd[@]}"; then
      echo "warning: failed to write combined LF ${label} plots" >&2
    fi
  else
    run_tracked_command "${plot_cmd[@]}"
  fi
}

cleanup_precision_metric_extra_dirs() {
  local output_root="$1"

  # Strict layout: precision combined metric dirs contain only plots/, data/, and README.md.
  if [[ -d "${output_root}" ]]; then
    find "${output_root}" -mindepth 1 -maxdepth 1 -type d ! -name plots ! -name data -exec rm -rf -- {} +
  fi
}

plot_timing_precision_combined() {
  local combined_plot_root workload
  local -a workload_filters=()

  if ((${#plot_profilers[@]})); then
    cleanup_legacy_combined_artifacts "${precision_root}"
    combined_plot_root="$(precision_combined_metric_dir timing)"
    declare -A combined_workloads=()
    collect_timing_workloads_from_roots plot_roots combined_workloads

    for workload in "${!combined_workloads[@]}"; do
      workload_filters+=(--workload "${workload}")
    done
    run_precision_combined_plot "${combined_plot_root}" timing_precision_combined_cmd append_sweep_plot_filters profile false "${workload_filters[@]}"
    cleanup_precision_metric_extra_dirs "${combined_plot_root}"
  elif [[ -z "${PLOT_OUTPUT_DIR}" ]]; then
    combined_plot_root="$(precision_combined_metric_dir timing)"
    write_missing_combined_readme \
      "${combined_plot_root}" \
      "LF Global Combined Artifacts" \
      "No global LF timing plots were generated. This can happen when plotting is disabled or no matching profile rows are available."
  fi
}

plot_memory_precision_combined() {
  local combined_memory_plot_root
  [[ "${PLOT_MEMORY_BREAKDOWN}" == "true" ]] || return 0

  cleanup_legacy_combined_artifacts "${precision_root}"
  combined_memory_plot_root="$(precision_combined_metric_dir memory)"
  if [[ "${#memory_plot_roots[@]}" -gt 0 ]]; then
    run_precision_combined_plot "${combined_memory_plot_root}" memory_precision_combined_cmd memory_plot_filters source-memory false
    cleanup_precision_metric_extra_dirs "${combined_memory_plot_root}"
  else
    write_missing_combined_readme \
      "${combined_memory_plot_root}" \
      "LF Source Memory Combined Artifacts" \
      "No source-memory breakdown rows were collected in this sweep. Include the source profiler in PROFILERS, or set PROFILE_MEMORY_BREAKDOWN=true for a run where source-memory hook overhead is acceptable."
  fi
}

plot_interconnect_precision_combined() {
  local combined_interconnect_plot_root

  cleanup_legacy_combined_artifacts "${precision_root}"
  combined_interconnect_plot_root="$(precision_combined_metric_dir interconnect)"
  if [[ "${#interconnect_plot_roots[@]}" -gt 0 ]]; then
    run_precision_combined_plot "${combined_interconnect_plot_root}" interconnect_precision_combined_cmd interconnect_plot_filters C2C true
    cleanup_precision_metric_extra_dirs "${combined_interconnect_plot_root}"
  else
    write_missing_combined_readme \
      "${combined_interconnect_plot_root}" \
      "LF C2C / CTC Combined Artifacts" \
      "No nsys profiler runs were selected in this sweep, so no Nsight C2C/CTC GPU metric samples can be summarized."
  fi
}

plot_utilization_precision_combined() {
  local combined_utilization_plot_root

  cleanup_legacy_combined_artifacts "${precision_root}"
  combined_utilization_plot_root="$(precision_combined_metric_dir utilization)"
  if [[ "${#utilization_plot_roots[@]}" -gt 0 ]]; then
    run_precision_combined_plot "${combined_utilization_plot_root}" utilization_precision_combined_cmd utilization_plot_filters utilization true
    cleanup_precision_metric_extra_dirs "${combined_utilization_plot_root}"
  else
    write_missing_combined_readme \
      "${combined_utilization_plot_root}" \
      "LF Utilization Combined Artifacts" \
      "No nsys profiler runs were selected in this sweep, so no Nsight GPU utilization samples can be summarized."
  fi
}

plot_throughput_precision_combined() {
  local combined_throughput_plot_root
  [[ "${PLOT}" == "true" ]] || return 0

  cleanup_legacy_combined_artifacts "${precision_root}"
  combined_throughput_plot_root="$(precision_combined_metric_dir throughput)"
  if [[ "${#throughput_plot_roots[@]}" -gt 0 ]]; then
    run_precision_combined_plot "${combined_throughput_plot_root}" throughput_precision_combined_cmd throughput_plot_filters throughput false
    cleanup_precision_metric_extra_dirs "${combined_throughput_plot_root}"
  else
    write_missing_combined_readme \
      "${combined_throughput_plot_root}" \
      "LF Throughput Combined Artifacts" \
      "No source profiler runs were selected in this sweep, so no per-step timing is available to compute tokens/sec."
  fi
}

plot_config_artifacts_for_root() {
  local config_root="$1"
  local seq_len="$2"
  [[ "${DRY_RUN}" != "true" ]] || return 0

  plot_config_root "${config_root}" "${seq_len}"
  if [[ -z "${PLOT_OUTPUT_DIR}" && ! -d "$(config_combined_metric_dir "${config_root}" timing)" ]]; then
    write_missing_combined_readme \
      "$(config_combined_metric_dir "${config_root}" timing)" \
      "LF Config Combined Artifacts" \
      "No config-level LF timing plots were generated for this config. This can happen when plotting is disabled or no matching profile rows are available."
  fi
  if [[ -n "${memory_plot_roots[${config_root}]+set}" ]]; then
    plot_memory_config_root "${config_root}" "${seq_len}"
  else
    write_missing_combined_readme \
      "$(config_combined_metric_dir "${config_root}" memory)" \
      "LF Source Memory Combined Artifacts" \
      "No source-memory breakdown rows were collected for this config. Include the source profiler in PROFILERS, or set PROFILE_MEMORY_BREAKDOWN=true for a run where source-memory hook overhead is acceptable."
  fi
  if [[ -n "${interconnect_plot_roots[${config_root}]+set}" ]]; then
    plot_interconnect_config_root "${config_root}" "${seq_len}"
  else
    write_missing_combined_readme \
      "$(config_combined_metric_dir "${config_root}" interconnect)" \
      "LF C2C / CTC Combined Artifacts" \
      "No nsys profiler run was selected for this config, so no Nsight C2C/CTC GPU metric samples can be summarized."
  fi
  if [[ -n "${utilization_plot_roots[${config_root}]+set}" ]]; then
    plot_utilization_config_root "${config_root}" "${seq_len}"
  else
    write_missing_combined_readme \
      "$(config_combined_metric_dir "${config_root}" utilization)" \
      "LF Utilization Combined Artifacts" \
      "No nsys profiler run was selected for this config, so no Nsight GPU utilization samples can be summarized."
  fi
  if [[ -n "${throughput_plot_roots[${config_root}]+set}" ]]; then
    plot_throughput_config_root "${config_root}" "${seq_len}"
  else
    write_missing_combined_readme \
      "$(config_combined_metric_dir "${config_root}" throughput)" \
      "LF Throughput Combined Artifacts" \
      "No source profiler run was selected for this config, so no per-step timing is available to compute tokens/sec."
  fi
  write_config_artifact_readme "${config_root}"
}

declare -A prepared_dataset_keys=()
for _lp_idx in "${!lora_dropouts[@]}"; do
    LORA_DROPOUT="${lora_dropouts[$_lp_idx]}"
    LORA_RANK="${lora_ranks[$_lp_idx]}"
    LORA_ALPHA="${lora_alphas[$_lp_idx]}"
    LORA_TARGET="${lora_targets[$_lp_idx]}"
    lora_dropout_label_value="$(lora_dropout_label "${LORA_DROPOUT}")"

    unset lora_config_roots
    declare -A lora_config_roots=()
    for paired_run_spec in "${paired_run_specs[@]}"; do
      IFS=';' read -r model_spec_entry backend_recompute workload exp_act_policy_pair paired_run_extra <<< "${paired_run_spec}"
      [[ -n "${model_spec_entry}" && -n "${backend_recompute}" && -n "${workload}" && -n "${exp_act_policy_pair}" && -z "${paired_run_extra:-}" ]] ||
        die "internal error: malformed paired RUNS entry '${paired_run_spec}'"

      parse_model_spec "${model_spec_entry}"
      current_model_name="${parsed_model_name}"
      current_model_gpu_count="${parsed_model_gpu_count}"
      current_model_tag=$(basename "${current_model_name}" | tr '/:' '__')
      resolve_current_runtime_for_model "${current_model_name}"
      validate_current_fa4_runtime "${current_model_name}"
      workload_label="$(safe_label "${current_model_tag}")"
      TEMPLATE="${template_spec}"
      if [[ "${TEMPLATE}" == "auto" ]]; then
        TEMPLATE="$(infer_template "${current_model_name}")"
      fi
      echo "Using template: ${TEMPLATE} for model ${current_model_name} requesting ${current_model_gpu_count} GPU(s), recompute from RUNS, flash_attn=${CURRENT_FLASH_ATTN}, lf_dir=${CURRENT_LF_DIR}, env_dir=${CURRENT_ENV_DIR}"

      IFS='|' read -r seq_len PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS <<< "${workload}"
      batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}"
      current_dataset="${DATASET}"
      if [[ "${PREPARE_DATASETS}" == "true" ]]; then
        current_dataset="$(dataset_name_for_seq "${seq_len}")"
        if [[ "${COLLECT_EXISTING}" != "true" ]]; then
          dataset_key="${current_model_name}|${seq_len}|${current_dataset}"
          if [[ -z "${prepared_dataset_keys[${dataset_key}]+set}" ]]; then
            if ! prepare_dataset_for_seq "${seq_len}" "${current_dataset}"; then
              failures=$((failures + 1))
              if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
                exit 1
              fi
              continue
            fi
            prepared_dataset_keys["${dataset_key}"]=1
          fi
        fi
      fi

      config_root="$(config_root_path "${seq_len}")"
      lora_config_roots["${config_root}"]="${seq_len}"
      for lf_expert_lora_impl in "${lf_expert_lora_impls[@]}"; do
        expert_policy="${exp_act_policy_pair%%|*}"
        policy_tail="${exp_act_policy_pair#*|}"
        ASYMM_EXPERT_ACT_OFFLOAD="${policy_tail%%|*}"
        policy_tail="${policy_tail#*|}"
        ASYMM_ATTN_ACT_OFFLOAD="${policy_tail%%|*}"
        policy_tail="${policy_tail#*|}"
        ASYMM_LAYER_ACT_OFFLOAD="${policy_tail%%|*}"
        policy_tail="${policy_tail#*|}"
        ASYMM_LAYER_GC="${policy_tail%%|*}"
        ASYMM_ATTN_SDPA_RECOMPUTE="${policy_tail#*|}"
        expact_label="$(expact_tag "${ASYMM_EXPERT_ACT_OFFLOAD}")"
        attnact_label="$(attnact_tag "${ASYMM_ATTN_ACT_OFFLOAD}")"
        layeract_label="$(layeract_tag "${ASYMM_LAYER_ACT_OFFLOAD}")"
        layergc_label="$(layergc_tag "${ASYMM_LAYER_GC}")"
        sdparecomp_label="$(sdparecomp_tag "${ASYMM_ATTN_SDPA_RECOMPUTE}")"

        IFS='|' read -r backend recompute liger_loss recompute_kernel_code recompute_outer_hbm_n backend_spec_extra <<< "${backend_recompute}"
        [[ -n "${backend}" && -n "${recompute}" && -n "${liger_loss}" && -z "${backend_spec_extra:-}" ]] || die "internal error: malformed normalized backend spec '${backend_recompute}'"
        recompute_outer_hbm_n="${recompute_outer_hbm_n:-0}"
        if { is_policy_independent_backend "${backend}" || [[ "${recompute}" == "recomp" ]]; } && [[ "${exp_act_policy_pair}" != "${exp_act_policy_pairs[0]}" ]]; then
          if [[ "${expert_policy}" != "off-layer" && "${ASYMM_EXPERT_ACT_OFFLOAD}" != "true" && "${ASYMM_ATTN_ACT_OFFLOAD}" != "true" && "${ASYMM_LAYER_GC}" != "true" ]]; then
            echo "Skipping backend=${backend} recompute=${recompute} policy=${exp_act_policy_pair}; inert policy axes run once (canonicalized to none|false|false|false|false|false)."
            continue
          fi
        fi
        if [[ "${recompute}" == "recomp" ]]; then
          if [[ "${expert_policy}" != "none" ]]; then
            die "selective GC policy '${expert_policy}' is not supported with global recompute; set backend recompute=norecomp, unsloth-off, or a recomp-off-* stage"
          fi
          if [[ "${ASYMM_EXPERT_ACT_OFFLOAD}" == "true" || "${ASYMM_ATTN_ACT_OFFLOAD}" == "true" || "${ASYMM_LAYER_ACT_OFFLOAD}" == "true" || "${ASYMM_LAYER_GC}" == "true" ]]; then
            die "activation offload tuples are not supported with global recompute; set backend recompute=norecomp, unsloth-off, or a recomp-off-* stage"
          fi
        fi
        for router_mode in "${router_modes[@]}"; do
          for profiler in "${profilers[@]}"; do
            profiler_runs_nsys=false
            [[ "${profiler}" == "nsys" || "${profiler}" == "both" ]] && profiler_runs_nsys=true
            if [[ "${backend}" == "kt_armbf16" && "${profiler_runs_nsys}" == "true" && "${KT_ARM_ALLOW_NSYS_WITHOUT_SOURCE_OK}" != "1" ]]; then
              echo "Skipping backend=kt_armbf16 profiler=${profiler}; run profiler=source first and set KT_ARM_ALLOW_NSYS_WITHOUT_SOURCE_OK=1 only after one source step completes."
              continue
            fi
            job_router_mode="${router_mode}"
            if [[ "${router_mode}" == "whole" && "${backend}" != "asym" && "${backend}" != "asym_torch" && "${backend}" != "asym_cpuadamwtorch" && "${backend}" != "asym_cpuadamwds" ]]; then
              if [[ "${router_hf_selected}" != "true" ]]; then
                job_router_mode=hf
              else
                die "backend=${backend} router_mode=${router_mode} is invalid; owned routing requires an AsymGEMM backend"
              fi
            fi
            if [[ "${backend}" == "kt_armbf16" && "${profiler_runs_nsys}" == "true" ]]; then
              if ! kt_arm_matching_source_profile_complete "${config_root}" "${backend}" "${recompute}" "${expert_policy}" "${job_router_mode}" "${liger_loss}" "${seq_len}" "${current_model_name}"; then
                echo "Skipping backend=kt_armbf16 profiler=${profiler}; matching source profile is missing, incomplete, or stale for seq=${seq_len} recompute=${recompute} liger_loss=${liger_loss} expert_policy=${expert_policy} router_mode=${job_router_mode} qwen_expert_lora_impl=${lf_expert_lora_impl} ${expact_label} ${attnact_label} ${layeract_label} ${layergc_label}."
                continue
              fi
            fi
            gpu_count="$(backend_gpu_count "${backend}" "${current_model_gpu_count}")"
            gpu="$(gpu_slice "${gpu_count}")"
            if cpuadam_backend_for_label "${backend}" >/dev/null; then
              grad_offload_modes_for_job=("${asym_cpu_adamw_grad_offload_modes[@]}")
              weight_offload_modes_for_job=("${asym_cpu_adamw_weight_offload_modes[@]}")
            else
              grad_offload_modes_for_job=(false)
              weight_offload_modes_for_job=(false)
            fi
            for grad_offload in "${grad_offload_modes_for_job[@]}"; do
              for weight_offload in "${weight_offload_modes_for_job[@]}"; do
                if [[ "${weight_offload}" == "true" && "${grad_offload}" != "true" ]]; then
                  die "internal error: weight offload requires grad offload"
                fi
                if ! run_job "${backend}" "${profiler}" "${recompute}" "${liger_loss}" "${seq_len}" "${gpu}" "${gpu_count}" "${expert_policy}" "${job_router_mode}" "${current_dataset}" "${lf_expert_lora_impl}" "${grad_offload}" "${weight_offload}" "${recompute_kernel_code:-}" "${recompute_outer_hbm_n:-0}"; then
                  failures=$((failures + 1))
                  if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
                    exit 1
                  fi
                fi
              done
            done
          done
        done
      done
    done

    if [[ "${DRY_RUN}" != "true" ]]; then
      for config_root in "${!lora_config_roots[@]}"; do
        plot_config_artifacts_for_root "${config_root}" "${lora_config_roots[${config_root}]}"
      done
    fi
  done
if ((failures > 0)); then
  echo "${failures} profiling job(s) failed" >&2
  if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
    exit 1
  fi
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "Dry run completed; skipping plots."
  exit 0
fi

if [[ "${PLOT}" == "true" ]]; then
  plot_timing_precision_combined
  plot_memory_precision_combined
  plot_interconnect_precision_combined
  plot_utilization_precision_combined
  plot_throughput_precision_combined
fi

if [[ "${DRY_RUN}" != "true" ]]; then
  write_precision_artifact_readme "${precision_root}"
fi

echo "LF profiling completed. Results: ${precision_root}"


if [[ "${RUN_POST}" == "true" ]]; then
  bash scripts/lf/test_profiling.sh
fi
