"""Config-only AutoBridge probe: is Llama-4-Scout runnable in Megatron-Bridge?"""
import json
import os

from huggingface_hub import snapshot_download

p = snapshot_download("meta-llama/Llama-4-Scout-17B-16E", allow_patterns=["*.json"])
c = json.load(open(os.path.join(p, "config.json")))
print("architectures:", c.get("architectures"), flush=True)

import torch  # noqa: F401
from megatron.bridge import AutoBridge

try:
    AutoBridge.from_hf_pretrained(p, trust_remote_code=False)
    print("LLAMA4_BRIDGE_OK (unexpected)", flush=True)
except Exception as exc:
    print("LLAMA4_BRIDGE_UNSUPPORTED:", type(exc).__name__, str(exc)[:400], flush=True)
