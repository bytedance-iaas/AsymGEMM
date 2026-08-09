import torch
from transformers.models.jamba.configuration_jamba import JambaConfig
from transformers.models.jamba.modeling_jamba import JambaSparseMoeBlock
import sys
sys.path.insert(0, "/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM")
from asym_gemm.training.jamba_moe import is_jamba_moe_block, wrap_jamba_moe_block

torch.manual_seed(0)
cfg = JambaConfig(hidden_size=64, intermediate_size=128, num_experts=8, num_experts_per_tok=2)
blk = JambaSparseMoeBlock(cfg).to(torch.bfloat16).eval()
for p in blk.parameters():
    torch.nn.init.normal_(p, std=0.02)
assert is_jamba_moe_block(blk), "detector rejects a genuine JambaSparseMoeBlock"

x = torch.randn(2, 16, 64, dtype=torch.bfloat16)
with torch.no_grad():
    ref = blk(x)

w = wrap_jamba_moe_block(blk, backend="torch", precision="bf16", offload=False,
                         lora_rank=8, lora_alpha=16.0, lora_dropout=0.0)
w = w.eval()
with torch.no_grad():
    out = w(x)
diff = (out - ref).abs().max().item()
print("max|diff| torch-backend vs HF:", diff)
assert diff < 1e-2, f"parity FAIL {diff}"

# negative controls: qwen3 detector must NOT capture jamba; jamba detector must not capture mixtral
from asym_gemm.training.qwen3_moe import is_qwen3_moe_block
from asym_gemm.training.mixtral_moe import is_mixtral_moe_block
print("qwen3-detector-on-jamba:", is_qwen3_moe_block(blk), "(wrapped now, expect False)")
print("mixtral-detector-on-jamba:", is_mixtral_moe_block(blk))
print("UNIT-PARITY-OK")
