"""Rebuild the FUSED Mixtral-8x22B checkpoint on THIS node (c17), 2026-08-09.
Identical math to agent/archive/figtmp/mx_fuse_ckpt2.py (the 08-06 rev2 tool),
but dst = the driver's M-map path directly and HF offline (local snapshot).
Needed because the fused copy lived on the c12/c14 scratch (node-local) and
mixtral 2-rank loads without it blow the host pool (per-rank WeightConverter
transient, figure provenance note)."""
import json, os, re, glob, shutil

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

src_dir = snapshot_download("mistralai/Mixtral-8x22B-v0.1", allow_patterns=["*.safetensors*", "*.json", "tokenizer*", "*.model"])
dst = "/scratch_local/user_data/shutian/kevin/cache/fused/Mixtral-8x22B-v0.1"
os.makedirs(dst, exist_ok=True)

idx = json.load(open(os.path.join(src_dir, "model.safetensors.index.json")))
wmap = idx["weight_map"]
handles = {}
def get(key):
    f = wmap[key]
    if f not in handles:
        handles[f] = safe_open(os.path.join(src_dir, f), framework="pt", device="cpu")
    return handles[f].get_tensor(key)

new_map, shard_id, cur, cur_bytes = {}, 1, {}, 0
SHARD_LIMIT = 9 * 2**30
def flush():
    global shard_id, cur, cur_bytes
    if not cur:
        return
    name = f"model-fused-{shard_id:05d}.safetensors"
    save_file(cur, os.path.join(dst, name), metadata={"format": "pt"})
    for k in cur:
        new_map[k] = name
    shard_id += 1; cur = {}; cur_bytes = 0
def put(key, tensor):
    global cur_bytes
    cur[key] = tensor.contiguous()
    cur_bytes += tensor.numel() * tensor.element_size()
    if cur_bytes >= SHARD_LIMIT:
        flush()

done_expert = set()
for key in sorted(wmap):
    m = re.match(r"model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight", key)
    if m:
        L = int(m.group(1))
        if L in done_expert:
            continue
        w1 = torch.stack([get(f"model.layers.{L}.block_sparse_moe.experts.{e}.w1.weight") for e in range(8)], dim=0)
        w3 = torch.stack([get(f"model.layers.{L}.block_sparse_moe.experts.{e}.w3.weight") for e in range(8)], dim=0)
        put(f"model.layers.{L}.mlp.experts.gate_up_proj", torch.cat([w1, w3], dim=1))
        del w1, w3
        w2 = torch.stack([get(f"model.layers.{L}.block_sparse_moe.experts.{e}.w2.weight") for e in range(8)], dim=0)
        put(f"model.layers.{L}.mlp.experts.down_proj", w2)
        del w2
        done_expert.add(L)
        if L % 8 == 0:
            print(f"[fuse-local] layer {L} done", flush=True)
    else:
        put(key.replace(".block_sparse_moe.", ".mlp."), get(key))
flush()
json.dump({"metadata": {"total_size": sum(os.path.getsize(os.path.join(dst, f)) for f in set(new_map.values()))},
           "weight_map": new_map}, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=1)
for f in glob.glob(os.path.join(src_dir, "*.json")) + glob.glob(os.path.join(src_dir, "tokenizer*")) + glob.glob(os.path.join(src_dir, "*.model")):
    b = os.path.basename(f)
    if b != "model.safetensors.index.json" and not os.path.exists(os.path.join(dst, b)):
        shutil.copy(f, os.path.join(dst, b))
print(f"[fuse-local] DONE: {len(new_map)} keys, {shard_id-1} shards -> {dst}", flush=True)
