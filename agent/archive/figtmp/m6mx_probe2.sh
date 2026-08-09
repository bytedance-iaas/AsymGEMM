#!/bin/bash
# T3-outlives-T2B campaign, probe round 2 (INSIDE container).
# P-C m6mxt3p384a: T3' @384K b1 — the deeper-T3 recipe:
#   token ohbm14 (keep every 14th outer GC boundary in HBM: 4 of 56, ~17.6 GiB
#   HBM at 384K, same off host), pool-cache trim 77->24 GB
#   (ASYM_EXPACT_CPU_POOL_MAX_BYTES), guarded moe-fg direct-reuse byte-diet.
#   Everything else = the mixtral T3 run-form (ker000 + moe|T3 env, chunk 1024
#   confirmed optimal by m6mxp1 null result).
# Run w1+m2 (driver defaults) so a FIT is directly the plotted timed cell.
# Context: T2B preset @384K = HOST abort (m6mxt2b384, watchdog 49<50 GiB,
# HOST_OOM_EVIDENCE=true) — this run fitting = the sole-survivor separation.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

sleep 30
for _ in $(seq 1 21); do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  [ -z "$apps" ] && break
  echo "!!! GUARD: GPU busy (pids: $apps) — waiting"
  sleep 30
done

echo "=== P-C: T3' mixtral 384K b1 (w1+m2, ohbm14+pool24+reuse) $(date -u +%H:%M:%S)"
ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 \
ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 \
ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 \
ASYMM_QWEN3_MOE_FG_DA_GPU=1 \
ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1 \
ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 \
ASYM_CPU_OPS_THREADS=48 \
ASYM_PLACEMENT_POLICY=1 \
ASYM_EXPACT_CPU_POOL_MAX_BYTES=24000000000 \
ASYMM_MOE_FG_DIRECT_REUSE=1 \
MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxt3p384a "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm14|ligerloss1" 384000 1
echo "M6MXP_C_EXIT=$?"
echo "=== PROBE ROUND 2 DONE $(date -u +%H:%M:%S)"
