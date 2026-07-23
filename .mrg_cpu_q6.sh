#!/bin/bash
# CPU-merge queue 6: M2 adjudication controls (same-night).
#  Q6a: M2 FLAGS-OFF tonight — the decisive drift control (C4b precedent):
#       policy-on tonight = 960/956/955; if flags-off tonight ≈ same band,
#       the policy is exonerated and M2 passes by same-day-control rule.
#  Q6b: M2 qknorm-OFF — attribution completeness on the last engaged feature.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
POL="ASYM_PLACEMENT_POLICY=1 ASYM_CPU_OPS_THREADS=48"
T2D="ASYM_GEMM_DISPATCH=staged ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM=1 ASYM_SAVED_TENSOR_ASYNC_UNPACK=1 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024"

echo "=== Q6a M2 FLAGS-OFF drift control (policy-on tonight: 960/956/955; flags-off yesterday: 973) $(date -u +%H:%M:%S)"
env $T2D bash scripts/lf/tp_probe.sh q3-32b cpum2ctl "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 128000 2
echo "Q6A_EXIT=$?"

echo "=== Q6b M2 qknorm-OFF attribution $(date -u +%H:%M:%S)"
env $POL $T2D ASYM_POLICY_QKNORM=0 bash scripts/lf/tp_probe.sh q3-32b cpum2nk "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 128000 2
echo "Q6B_EXIT=$?"
KA120="ASYM_GEMM_DISPATCH=staged ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1"
echo "=== Q6c M7 re-validate after record_stream fix (expect peak ~165.7, tok/s >= 2723) $(date -u +%H:%M:%S)"
env $POL $KA120 MAX_SAMPLES=1024 bash scripts/lf/tp_probe.sh q3-30b-a3b cpum7f "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 120000 8
echo "Q6C_EXIT=$?"
echo "=== CPU-Q6 DONE $(date -u +%H:%M:%S)"
