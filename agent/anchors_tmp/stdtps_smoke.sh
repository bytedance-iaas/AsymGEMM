#!/bin/bash
# stdtps_smoke.sh — in-container preflight for the stdtps campaign (no GPU run).
cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM || { echo "SMOKE FAIL cd"; exit 1; }
echo "== host/gpu =="; hostname; nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
echo "== venv torch + _C =="
.venv/bin/python - <<'EOF'
import torch, importlib
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(), "ndev", torch.cuda.device_count())
import asym_gemm._C as C
print("_C ok:", bool(C))
import matplotlib; print("matplotlib", matplotlib.__version__)
import huggingface_hub; print("hf_hub", huggingface_hub.__version__)
import transformers; print("transformers", transformers.__version__)
EOF
rc=$?
echo "== fa4 venv =="
.venv-fa4/bin/python -c "import torch; print('fa4 torch', torch.__version__)" 2>&1 | tail -1
echo "== paths =="
ls -d /workspace/env/figures scripts/lf/profile_lora_lf_test_source.sh scripts/lf/parse_fill_cell.py 2>&1
ls /scratch_local/user_data/shutian/kevin/cache/huggingface/hub/models--tencent--Hunyuan-A13B-Instruct/snapshots/*/config.json 2>&1 | head -1
echo "SMOKE rc=$rc"
