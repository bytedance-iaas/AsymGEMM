#!/bin/bash
# FIG-1b v3 chain (INSIDE container): new layout — per model 2 bars = short/
# long seq under SuperOffload (superoffload_mem|unsloth), r16 adapters,
# CPU segments estimated (weights bf16 + adapter state fp32), GPU measured
# from memory_breakdown_summary. One run per bar; q30b long bar reuses the
# archived tput som@640K cell. All w1+m1 memory probes.
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

guard V1
echo "=== V1: som q30b 192K b1 r16 (w1+m1) $(date -u +%H:%M:%S)"
LORA_RANK=16 WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh q3-30b-a3b f1vsomm "superoffload_mem|unsloth-ohbm0|ligerloss1" 192000 1
echo "F1V_V1_EXIT=$?"

guard V2
echo "=== V2: som mixtral 128K b1 r16 (w1+m1) $(date -u +%H:%M:%S)"
LORA_RANK=16 WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh mixtral-8x22b f1vsomx1 "superoffload_mem|unsloth-ohbm0|ligerloss1" 128000 1
echo "F1V_V2_EXIT=$?"

guard V3
echo "=== V3: som mixtral 256K b1 r16 (w1+m1; if OOM retry 224K) $(date -u +%H:%M:%S)"
LORA_RANK=16 WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh mixtral-8x22b f1vsomx2 "superoffload_mem|unsloth-ohbm0|ligerloss1" 256000 1
rc=$?
echo "F1V_V3_EXIT=$rc"
if [ "$rc" = "1" ]; then
  guard V3B
  echo "=== V3B: som mixtral 224K b1 r16 fallback (w1+m1) $(date -u +%H:%M:%S)"
  LORA_RANK=16 WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
    bash scripts/lf/tp_probe.sh mixtral-8x22b f1vsomx2b "superoffload_mem|unsloth-ohbm0|ligerloss1" 224000 1
  echo "F1V_V3B_EXIT=$?"
fi

echo "=== FIG1B V3 CHAIN DONE $(date -u +%H:%M:%S)"
