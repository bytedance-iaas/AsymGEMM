#!/bin/bash
# A/B: q32 T3 128k b2 (offload-heavy state), PRE-MERGE lib vs MERGED lib on the
# SAME machine. Decides merge-vs-machine attribution for the −2/−3% seen on the
# two maximally-offloading rows. Swap is in-place + restored; runs are serial.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
CFG="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1"

echo "=== AB-1 MERGED lib, q32 T3 128k b2 $(date -u +%H:%M:%S)"
MAX_SAMPLES=1024 bash scripts/lf/tp_probe.sh q3-32b mrgabm "${CFG}" 128000 2
echo "ABM_EXIT=$?"

echo "=== AB-2 swap to PRE-MERGE lib $(date -u +%H:%M:%S)"
mkdir -p .mrg_merged_files
for f in activation_offload attention_activation_offload decoder_activation_offload dense_mlp_finegrained frozen_linear qwen3_moe_finegrained; do
  cp "asym_gemm/training/$f.py" ".mrg_merged_files/$f.py"
  cp ".mrg_premerge_files/$f.py" "asym_gemm/training/$f.py"
done
echo "swapped to pre-merge"
MAX_SAMPLES=1024 bash scripts/lf/tp_probe.sh q3-32b mrgabp "${CFG}" 128000 2
echo "ABP_EXIT=$?"

echo "=== AB-3 restore MERGED lib $(date -u +%H:%M:%S)"
for f in activation_offload attention_activation_offload decoder_activation_offload dense_mlp_finegrained frozen_linear qwen3_moe_finegrained; do
  cp ".mrg_merged_files/$f.py" "asym_gemm/training/$f.py"
done
python3 -c "import asym_gemm.training.qwen3_moe_finegrained as q; assert hasattr(q, '_keep_stage_noclone_enabled'); print('restore verified: merged lib back in place')"
echo "=== AB DONE $(date -u +%H:%M:%S)"
