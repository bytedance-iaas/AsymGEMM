"""Manual streaming fusion of the Mixtral-8x22B checkpoint (2026-08-06 rev2).
save_pretrained writes hub-format (old keys), so apply the registry's
conversions ourselves: per layer L:
  stack_e w1[L,e] (dim0) ++ stack_e w3[L,e] -> concat(dim1) -> mlp.experts.gate_up_proj
  stack_e w2[L,e] (dim0)                                  -> mlp.experts.down_proj
  rename .block_sparse_moe. -> .mlp.  (all remaining keys)
Weights bit-identical; output loads pattern-free (no WeightConverter match)
-> mmap-lazy, zero conversion transient for the deepspeed-path runs.
"""
import json, os, re, glob
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from huggingface_hub import snapshot_download
src_dir = snapshot_download("mistralai/Mixtral-8x22B-v0.1", allow_patterns=["*.safetensors*", "*.json", "tokenizer*", "*.model"])
dst = "/scratch_local/user_data/shutian/kevin/cache/mixtral-8x22b-v0_1-fused"
os.makedirs(dst, exist_ok=True)

idx = json.load(open(os.path.join(src_dir, "model.safetensors.index.json")))
wmap = idx["weight_map"]
handles = {}
def get(key):
    f = wmap[key]
    if f not in handles:
        handles[f] = safe_open(os.path.join(src_dir, f), framework="pt", device="cpu")
    return handles[f].get_tensor(key)

layers = sorted({int(m.group(1)) for k in wmap for m in [re.match(r"model\.layers\.(\d+)\.", k)] if m})
new_map, shard_id, cur, cur_bytes = {}, 1, {}, 0
SHARD_LIMIT = 9 * 2**30
def flush():
    global shard_id, cur, cur_bytes
    if not cur: return
    name = f"model-fused-{shard_id:05d}.safetensors"
    save_file(cur, os.path.join(dst, name), metadata={"format": "pt"})
    for k in cur: new_map[k] = name
    shard_id += 1; cur = {}; cur_bytes = 0
def put(key, tensor):
    global cur_bytes
    cur[key] = tensor.contiguous()
    cur_bytes += tensor.numel() * tensor.element_size()
    if cur_bytes >= SHARD_LIMIT: flush()

done_expert = set()
for key in sorted(wmap):
    m = re.match(r"model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight", key)
    if m:
        L = int(m.group(1))
        if L in done_expert: continue
        w1 = torch.stack([get(f"model.layers.{L}.block_sparse_moe.experts.{e}.w1.weight") for e in range(8)], dim=0)
        w3 = torch.stack([get(f"model.layers.{L}.block_sparse_moe.experts.{e}.w3.weight") for e in range(8)], dim=0)
        put(f"model.layers.{L}.mlp.experts.gate_up_proj", torch.cat([w1, w3], dim=1))
        del w1, w3
        w2 = torch.stack([get(f"model.layers.{L}.block_sparse_moe.experts.{e}.w2.weight") for e in range(8)], dim=0)
        put(f"model.layers.{L}.mlp.experts.down_proj", w2)
        del w2
        done_expert.add(L)
        if L % 8 == 0: print(f"[fuse2] layer {L} done", flush=True)
    else:
        put(key.replace(".block_sparse_moe.", ".mlp."), get(key))
flush()
json.dump({"metadata": {"total_size": sum(os.path.getsize(os.path.join(dst, f)) for f in set(new_map.values()))},
           "weight_map": new_map}, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=1)
for f in glob.glob(os.path.join(src_dir, "*.json")) + glob.glob(os.path.join(src_dir, "tokenizer*")) + glob.glob(os.path.join(src_dir, "*.model")):
    b = os.path.basename(f)
    if b != "model.safetensors.index.json" and not os.path.exists(os.path.join(dst, b)):
        import shutil; shutil.copy(f, os.path.join(dst, b))
old_shards = glob.glob(os.path.join(dst, "model-000*-of-*.safetensors"))
for f in old_shards: os.remove(f)
print(f"[fuse2] DONE: {len(new_map)} keys, {shard_id-1} shards; removed {len(old_shards)} stale no-op shards", flush=True)
