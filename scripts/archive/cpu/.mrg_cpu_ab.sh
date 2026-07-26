#!/bin/bash
# Same-day A/B: PRE-CPU-MERGE lib (main_kevin) vs MERGED lib, flags-off,
# q32 T2 128k b2 (the S3a config). Decides merge-vs-day for the S3a delta.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
ENV_T2="ASYM_GEMM_DISPATCH=staged ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM=1 ASYM_SAVED_TENSOR_ASYNC_UNPACK=1 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024"
CFG="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1"

echo "=== AB-M MERGED lib (flags-off) $(date -u +%H:%M:%S)"
env $ENV_T2 bash scripts/lf/tp_probe.sh q3-32b cpuabm "$CFG" 128000 2
echo "ABM_EXIT=$?"

echo "=== AB-P swap to PRE-CPU-MERGE lib $(date -u +%H:%M:%S)"
mkdir -p .mrg_cpu_merged
for f in activation_offload attention_activation_offload dense_mlp_finegrained qwen3_moe qwen3_moe_finegrained cpu_adam gc_boundary_offload; do
  cp "asym_gemm/training/$f.py" ".mrg_cpu_merged/$f.py"
  cp ".mrg_cpu_premerge/$f.py" "asym_gemm/training/$f.py"
done
cp asym_gemm/integrations/lf.py .mrg_cpu_merged/lf.py && cp .mrg_cpu_premerge/lf.py asym_gemm/integrations/lf.py
cp asym_gemm/__init__.py .mrg_cpu_merged/__init__.py && cp .mrg_cpu_premerge/__init__.py asym_gemm/__init__.py
echo "swapped to pre-cpu-merge"
env $ENV_T2 bash scripts/lf/tp_probe.sh q3-32b cpuabp "$CFG" 128000 2
echo "ABP_EXIT=$?"

echo "=== AB-R restore MERGED lib $(date -u +%H:%M:%S)"
for f in activation_offload attention_activation_offload dense_mlp_finegrained qwen3_moe qwen3_moe_finegrained cpu_adam gc_boundary_offload; do
  cp ".mrg_cpu_merged/$f.py" "asym_gemm/training/$f.py"
done
cp .mrg_cpu_merged/lf.py asym_gemm/integrations/lf.py
cp .mrg_cpu_merged/__init__.py asym_gemm/__init__.py
python3 -c "from asym_gemm.training import placement_policy, cpu_ops; print('restore verified: merged lib back')"
echo "=== CPU-AB DONE $(date -u +%H:%M:%S)"
