"""Probe the new MixtralBridge: provider fields from config, then full download."""
import os

from huggingface_hub import snapshot_download

p = snapshot_download("mistralai/Mixtral-8x22B-v0.1", allow_patterns=["*.json"], local_files_only=True)

import torch  # noqa: F401
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained(p, trust_remote_code=False)
prov = bridge.to_megatron_provider(load_weights=False)
print("MIXTRAL_BRIDGE_REGISTERED_OK", flush=True)
for f in ("num_layers", "hidden_size", "ffn_hidden_size", "moe_ffn_hidden_size",
          "num_attention_heads", "num_query_groups", "kv_channels", "vocab_size",
          "num_moe_experts", "moe_router_topk", "moe_router_pre_softmax",
          "rotary_base", "qk_layernorm", "add_qkv_bias", "gated_linear_unit",
          "share_embeddings_and_output_weights"):
    print(f"  {f} = {getattr(prov, f, '<missing>')}", flush=True)

print("downloading full weights...", flush=True)
full = snapshot_download("mistralai/Mixtral-8x22B-v0.1", max_workers=8)
print("MIXTRAL_DL_DONE", full, flush=True)
