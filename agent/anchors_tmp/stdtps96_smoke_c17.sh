#!/bin/bash
# Session E (c17) prereq-1 smoke: container + venv + asym_gemm._C + CUDA
# contexts on the sim pair + weights visibility. NO training.
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
echo "== inside container: $(hostname) CVD=$CUDA_VISIBLE_DEVICES NVD=$NVIDIA_VISIBLE_DEVICES"
echo "== venv python/torch/asym_gemm._C =="
.venv/bin/python - <<'PY'
import sys
print("python", sys.version.split()[0])
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(), "ndev", torch.cuda.device_count())
import asym_gemm._C as C
print("asym_gemm._C OK:", C.__name__)
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"dev{i} {torch.cuda.get_device_name(i)} free={free/2**30:.1f}G total={total/2**30:.1f}G")
PY
echo "== weights on node cache =="
ls /scratch_local/user_data/shutian/kevin/cache/fused/gpt-oss-20b-bf16/config.json \
   /scratch_local/user_data/shutian/kevin/cache/huggingface/hub/models--tencent--Hunyuan-A13B-Instruct \
   /scratch_local/user_data/shutian/kevin/cache/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B \
   /scratch_local/user_data/shutian/kevin/cache/huggingface/hub/models--zai-org--GLM-4.5-Air \
   /scratch_local/user_data/shutian/kevin/cache/huggingface/hub/models--zai-org--GLM-4.7-Flash >/dev/null && echo WEIGHTS-OK || echo WEIGHTS-MISSING
echo "== cross-superchip gate present =="
grep -n "ALLOW_CROSS_SUPERCHIP" scripts/lf/run_lf_lora_sft.sh | head -2
echo "== hub anonymous check (expired-token gotcha) =="
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN="" .venv/bin/python - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("tencent/Hunyuan-A13B-Instruct", trust_remote_code=False)
print("hunyuan tokenizer OK, vocab", tok.vocab_size)
PY
echo "SMOKE-DONE rc=$?"
