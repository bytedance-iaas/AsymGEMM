#!/bin/bash
# CPU-merge queue 7: final re-validations with both gate fixes in tree
# (record_stream drop 23a2efc + dense-T2 qknorm regime gate 03a43e7).
#  Q7a: M2 with the qknorm regime gate (expect ~977, PASS bar 971.2)
#  Q7b: conditional M8 bisect — ONLY if Q6d M8f failed; qknorm-off at deep-shed
#       (if it recovers, extend the P12 gate with a MoE deep-shed clause per data)
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
POL="ASYM_PLACEMENT_POLICY=1 ASYM_CPU_OPS_THREADS=48"
T2D="ASYM_GEMM_DISPATCH=staged ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM=1 ASYM_SAVED_TENSOR_ASYNC_UNPACK=1 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024"
PINS="ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 ASYMM_QWEN3_MOE_FG_DA_GPU=1 ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1"

echo "=== Q7a M2 with qknorm regime gate (expect ~977 >= 971.2) $(date -u +%H:%M:%S)"
env $POL $T2D bash scripts/lf/tp_probe.sh q3-32b cpum2g "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 128000 2
echo "Q7A_EXIT=$?"

if [ "${RUN_M8_BISECT:-0}" = "1" ]; then
  echo "=== Q7b M8 qknorm-OFF deep-shed bisect $(date -u +%H:%M:%S)"
  env $POL ASYM_GEMM_DISPATCH=staged $PINS ASYM_POLICY_QKNORM=0 MAX_SAMPLES=1024 bash scripts/lf/tp_probe.sh q3-30b-a3b cpub8nq "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 800000 1
  echo "Q7B_EXIT=$?"
fi
echo "=== CPU-Q7 DONE $(date -u +%H:%M:%S)"
