#!/bin/bash
# T3-outlives-T2B campaign, round 10 (INSIDE container): the cpu-act window fix.
# ATTRIBUTION VERDICT (round 9, .figtmp/attrib352.log): the dial-invariant
# early-backward death is the moe CPU-act live window — gate/up/act host
# buffers [2S,16384] bf16 = 21.5 GiB EACH at 352K, ~65 GiB/layer, logical
# bytes (immune to exact-pinning). Tier dial: ASYMM_QWEN3_MOE_FG_CPU_ACT_MAX_
# ROWS=500000 -> at >=250K tokens (2S rows > cap) the engine's GPU act path
# runs instead (documented threshold; CPU act was only a ~2% win here).
# T3''' = fix2 infra + ohbm14 + skip-bwd + direct-reuse + threads64 + the cap.
# T2B stays fix2-infra preset. Pair @384K; ascend to 400K (ohbm28) if both fit.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

guard() {
  sleep 30
  local apps
  for _ in $(seq 1 21); do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    [ -z "$apps" ] && return 0
    echo "!!! GUARD: GPU busy before $1 (pids: $apps) — waiting"
    sleep 30
  done
  echo "!!! GUARD FAIL before $1 — aborting chain"
  exit 9
}

t3run() { # $1 = seq, $2 = ohbm
  guard "T3_$1"
  echo "=== T3'''(capact,ohbm$2) @$1 (w1+m1) $(date -u +%H:%M:%S)"
  ASYM_EXACT_PINNED=1 ASYM_EXACT_PINNED_ROOTS=1 ASYM_EXACT_PINNED_SAVED=1 \
  ASYM_ATTN_SAVED_TENSOR_OFFLOAD_SKIP_IN_BACKWARD=1 \
  ASYM_MEM_ATTRIB_LOG=/workspace/AsymGEMM-SFT/third_party/AsymGEMM/.figtmp/attrib_r10.log \
  ASYMM_QWEN3_MOE_FG_CPU_ACT_MAX_ROWS=500000 \
  ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 \
  ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 \
  ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 \
  ASYMM_QWEN3_MOE_FG_DA_GPU=1 \
  ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1 \
  ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 \
  ASYM_CPU_OPS_THREADS=64 \
  ASYM_PLACEMENT_POLICY=1 \
  ASYMM_MOE_FG_DIRECT_REUSE=1 \
  WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
    bash scripts/lf/tp_probe.sh mixtral-8x22b "m6mxf5t3$(( $1 / 1000 ))" "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm$2|ligerloss1" "$1" 1
}

t2brun() { # $1 = seq
  guard "T2B_$1"
  echo "=== T2B(fix2) @$1 capacity attempt (w1+m1) $(date -u +%H:%M:%S)"
  ASYM_EXACT_PINNED=1 ASYM_EXACT_PINNED_ROOTS=1 ASYM_EXACT_PINNED_SAVED=1 \
  WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
    bash scripts/lf/tp_probe.sh mixtral-8x22b "m6mxf5t2b$(( $1 / 1000 ))" "asym_cpuadamwds|T2B|ligerloss1" "$1" 1
}

set +e
if t3run 384000 14; then
  if t2brun 384000; then
    echo "PAIR@384: BOTH FIT -> ascend 400K"
    if t3run 400000 28; then
      if t2brun 400000; then echo "PROBE10_STATE=both_fit_400"; exit 3
      else echo "PROBE10_WINNER=400000"; exit 0; fi
    else echo "PROBE10_STATE=window_(384,400)_t3wall"; exit 3; fi
  else
    echo "PROBE10_WINNER=384000"; exit 0
  fi
else
  echo "PROBE10_STATE=t3_dead_384_capact"
  exit 3
fi
