#!/bin/bash
cd /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface HF_HUB_OFFLINE=1
.venv/bin/python - <<'PY'
import json, os, glob, torch
from safetensors import safe_open
from huggingface_hub import snapshot_download
dst = "/scratch_local/user_data/shutian/kevin/cache/fused/Mixtral-8x22B-v0.1"
idx = json.load(open(f"{dst}/model.safetensors.index.json"))
wm = idx["weight_map"]; files = sorted(set(wm.values()))
missing = [f for f in files if not os.path.exists(f"{dst}/{f}")]
tot = sum(os.path.getsize(f"{dst}/{f}") for f in files if os.path.exists(f"{dst}/{f}"))
print(f"keys={len(wm)} shards={len(files)} missing={missing} total={tot/2**30:.1f} GiB index_total={idx['metadata']['total_size']/2**30:.1f} GiB")
print("aux files:", sorted(os.path.basename(p) for p in glob.glob(f"{dst}/*") if not p.endswith(".safetensors")))
# every key readable + shape sanity for layer 0 and 55
nkeys = 0
for f in files:
    with safe_open(f"{dst}/{f}", framework="pt", device="cpu") as h:
        nkeys += len(list(h.keys()))
print("keys readable:", nkeys)
src = snapshot_download("mistralai/Mixtral-8x22B-v0.1", allow_patterns=["*.json"])
sidx = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
def sget(k):
    f = sidx[k]
    with safe_open(f"{src}/{f}", framework="pt", device="cpu") as h:
        return h.get_tensor(k)
def dget(k):
    with safe_open(f"{dst}/{wm[k]}", framework="pt", device="cpu") as h:
        return h.get_tensor(k)
for L in (0, 27, 55):
    gu = dget(f"model.layers.{L}.mlp.experts.gate_up_proj"); dn = dget(f"model.layers.{L}.mlp.experts.down_proj")
    w1 = sget(f"model.layers.{L}.block_sparse_moe.experts.3.w1.weight"); w3 = sget(f"model.layers.{L}.block_sparse_moe.experts.3.w3.weight"); w2 = sget(f"model.layers.{L}.block_sparse_moe.experts.3.w2.weight")
    ok = torch.equal(gu[3], torch.cat([w1, w3], 0)) and torch.equal(dn[3], w2)
    print(f"layer {L}: gate_up {tuple(gu.shape)} down {tuple(dn.shape)} expert3 bit-identical={ok}")
q = dget("model.layers.0.self_attn.q_proj.weight"); print("q_proj L0 equal:", torch.equal(q, sget("model.layers.0.self_attn.q_proj.weight")))
print("lm_head equal:", torch.equal(dget("lm_head.weight"), sget("lm_head.weight")))
PY
