"""Build the DEQUANTIZED bf16 local gpt-oss-120b checkpoint on THIS node (c14),
mirroring the 20b precedent (GPTOSS20B_CAMPAIGN.md): transformers dequants the
MXFP4 experts on load (venv lacks the `kernels` pkg — the warning path IS the
dequant we want); tf-5.6 save_pretrained MANGLES dequantized expert weights via
shared-tensor detection, so shards are written MANUALLY (mx_fuse_local.py
pattern). Config is stripped of quantization_config; tokenizer files copied
verbatim from the snapshot.
"""
import json, os, shutil, glob

os.environ.setdefault("HF_HOME", "/scratch_local/user_data/shutian/kevin/cache/huggingface")
import torch
from safetensors.torch import save_file
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

SRC = "openai/gpt-oss-120b"
DST = "/scratch_local/user_data/shutian/kevin/cache/fused/gpt-oss-120b-bf16"
os.makedirs(DST, exist_ok=True)

print("loading (dequants MXFP4 on the fly)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.bfloat16, device_map=None)
sd = model.state_dict()
print(f"state dict: {len(sd)} tensors, {sum(t.numel()*t.element_size() for t in sd.values())/2**30:.1f} GiB", flush=True)

# sanity: per-layer expert tensors must be DISTINCT storages (the 20b mangle class)
ptrs = {}
for k, t in sd.items():
    if "experts" in k and ("gate_up_proj" in k or "down_proj" in k) and "bias" not in k:
        ptrs.setdefault(t.data_ptr(), []).append(k)
dups = {p: ks for p, ks in ptrs.items() if len(ks) > 1}
assert not dups, f"shared expert storages detected: {list(dups.values())[:2]}"
print(f"expert tensors distinct: {len(ptrs)} storages ok", flush=True)

SHARD_LIMIT = 9 * 2**30
new_map, shard_id, cur, cur_bytes = {}, 1, {}, 0
def flush():
    global shard_id, cur, cur_bytes
    if not cur:
        return
    name = f"model-{shard_id:05d}.safetensors"
    save_file({k: v.contiguous() for k, v in cur.items()}, os.path.join(DST, name),
              metadata={"format": "pt"})
    for k in cur:
        new_map[k] = name
    print(f"  wrote {name} ({cur_bytes/2**30:.1f} GiB, {len(cur)} tensors)", flush=True)
    shard_id += 1; cur, cur_bytes = {}, 0

for k, t in sd.items():
    b = t.numel() * t.element_size()
    if cur_bytes + b > SHARD_LIMIT and cur:
        flush()
    cur[k] = t; cur_bytes += b
flush()

json.dump({"metadata": {"total_size": sum(t.numel()*t.element_size() for t in sd.values())},
           "weight_map": new_map},
          open(os.path.join(DST, "model.safetensors.index.json"), "w"), indent=1)

snap = snapshot_download(SRC, allow_patterns=["*.json", "tokenizer*", "*.jinja"])
cfg = json.load(open(os.path.join(snap, "config.json")))
cfg.pop("quantization_config", None)
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=1)
for f in glob.glob(os.path.join(snap, "*")):
    base = os.path.basename(f)
    if base in ("config.json", "model.safetensors.index.json") or base.endswith(".safetensors"):
        continue
    if os.path.isfile(f):
        shutil.copy2(f, os.path.join(DST, base))
        print("  copied", base, flush=True)
print("DONE:", DST, flush=True)
