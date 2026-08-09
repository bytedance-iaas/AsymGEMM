#!/bin/bash
# Round 11 (INSIDE container): forensic death run @384K — every exact-pinned
# registration prints (shape + call site) + wrapper pack shapes. ~12 min.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
sleep 30
for _ in $(seq 1 21); do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  [ -z "$apps" ] && break
  echo "!!! GUARD: GPU busy (pids: $apps) — waiting"; sleep 30
done
echo "=== R11: forensic T3'''@384K (w1+m1, DEBUG) $(date -u +%H:%M:%S)"
ASYM_EXACT_PINNED=1 ASYM_EXACT_PINNED_ROOTS=1 ASYM_EXACT_PINNED_SAVED=1 \
ASYM_EXACT_PINNED_DEBUG=1 \
ASYM_ATTN_SAVED_TENSOR_DEDUP_DEBUG=1 \
ASYM_ATTN_SAVED_TENSOR_OFFLOAD_SKIP_IN_BACKWARD=1 \
ASYM_MEM_ATTRIB_LOG=/workspace/AsymGEMM-SFT/third_party/AsymGEMM/.figtmp/attrib_r11.log \
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
  bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxf6t3384 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm14|ligerloss1" 384000 1
echo "M6MX11_EXIT=$?"
echo "=== ROUND 11 DONE $(date -u +%H:%M:%S)"
