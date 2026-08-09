"""One-time mixtral checkpoint fusion (2026-08-06): load once on CPU (the
load applies transformers-5.x's per-expert->fused WeightConverters) and
re-save in the NEW fused format. Future loads then match no converter
patterns -> mmap-lazy, zero conversion transient, any rank count.
Weights are bit-identical (stack/concat of the same tensors, dtype unchanged).
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = "mistralai/Mixtral-8x22B-v0.1"
DST = "/scratch_local/user_data/shutian/kevin/cache/mixtral-8x22b-v0_1-fused"

t0 = time.time()
print(f"[fuse] loading {SRC} on CPU (conversion happens here)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.bfloat16)
print(f"[fuse] loaded in {time.time()-t0:.0f}s; saving fused -> {DST}", flush=True)
t1 = time.time()
model.save_pretrained(DST, safe_serialization=True, max_shard_size="10GB")
AutoTokenizer.from_pretrained(SRC).save_pretrained(DST)
print(f"[fuse] saved in {time.time()-t1:.0f}s; total {time.time()-t0:.0f}s", flush=True)
