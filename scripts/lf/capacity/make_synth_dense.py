#!/usr/bin/env python3
"""Synthetic dense Qwen3 checkpoint builder (rebuilt 2026-07-25; the original
did not survive the tree wipe). Emits bf16 ZEROS checkpoints as SPARSE
safetensors shards (header written, payload ftruncate'd holes): the loader path
is identical to a dense build (same mmap/copy/pin behavior, file cache is
evictable either way) but builds are ~instant and cost ~0 disk. Zeros-build
equivalence to the historic random builds was the R13/H-chain protocol already;
V1 (2026-07-25) reproduced the campaign's R3 anchor at +0.5 GiB.

Config template = verbatim field set recovered from the campaign cell logs
(transformers 5.6.0 Qwen3Config). GQA trap #4: heads = hidden/128 MUST be
divisible by kv=8 — h=12800 (100 heads) crashes at the first SDPA call and
produced the fake §8b OKs; this builder refuses such shapes.

Usage:
  make_synth_dense.py --out DIR --layers L --hidden H [--inter I]
True param count printed; index total_size = 2*params bytes.
"""

import argparse
import glob
import json
import os
import shutil
import struct
import sys

VOCAB = 151936
HEAD_DIM = 128
KV_HEADS = 8
DONOR_GLOB = "/scratch_local/user_data/shutian/kevin/cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/*"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json")
SHARD_BYTES_TARGET = 24 * (1 << 30)

# (h -> intermediate) pairs recovered from campaign cell logs
KNOWN_INTER = {9216: 32256, 12288: 43008, 12800: 44800, 8192: 28672, 10240: 35840, 11264: 39424, 13312: 46592, 1024: 3072}


def tensor_list(layers: int, hidden: int, inter: int) -> list[tuple[str, tuple[int, ...]]]:
    heads = hidden // HEAD_DIM
    q_out = heads * HEAD_DIM
    kv_out = KV_HEADS * HEAD_DIM
    out: list[tuple[str, tuple[int, ...]]] = [("model.embed_tokens.weight", (VOCAB, hidden))]
    for i in range(layers):
        p = f"model.layers.{i}."
        out += [
            (p + "input_layernorm.weight", (hidden,)),
            (p + "mlp.down_proj.weight", (hidden, inter)),
            (p + "mlp.gate_proj.weight", (inter, hidden)),
            (p + "mlp.up_proj.weight", (inter, hidden)),
            (p + "post_attention_layernorm.weight", (hidden,)),
            (p + "self_attn.k_norm.weight", (HEAD_DIM,)),
            (p + "self_attn.k_proj.weight", (kv_out, hidden)),
            (p + "self_attn.o_proj.weight", (hidden, q_out)),
            (p + "self_attn.q_norm.weight", (HEAD_DIM,)),
            (p + "self_attn.q_proj.weight", (q_out, hidden)),
            (p + "self_attn.v_proj.weight", (kv_out, hidden)),
        ]
    out += [("model.norm.weight", (hidden,)), ("lm_head.weight", (VOCAB, hidden))]
    return out


def numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def write_sparse_shard(path: str, entries: list[tuple[str, tuple[int, ...]]]) -> int:
    header: dict = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, shape in entries:
        nbytes = numel(shape) * 2
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    hj = json.dumps(header, separators=(",", ":")).encode()
    pad = (-(8 + len(hj))) % 8
    hj += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.truncate(8 + len(hj) + offset)  # zero payload as a hole
    return offset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--inter", type=int, default=0)
    ap.add_argument("--max-pos", type=int, default=262144)
    a = ap.parse_args()

    inter = a.inter or KNOWN_INTER.get(a.hidden, 0)
    if not inter:
        sys.exit(f"no known intermediate_size for hidden={a.hidden}; pass --inter")
    if a.hidden % HEAD_DIM:
        sys.exit("hidden must be divisible by 128")
    heads_check = a.hidden // HEAD_DIM
    if heads_check % KV_HEADS:
        # trap #4, re-armed 2026-07-25: h=12800 (100 heads, 100%8!=0) crashed at
        # the first SDPA call in EVERY H/I/J-chain cell; the old classifier
        # stamped those crashed loads OK. Never build an indivisible shape again.
        sys.exit(f"INVALID GQA: heads={heads_check} not divisible by kv={KV_HEADS} "
                 f"(this exact shape produced the fake H-chain OKs)")

    tensors = tensor_list(a.layers, a.hidden, inter)
    total_params = sum(numel(s) for _, s in tensors)
    os.makedirs(a.out, exist_ok=True)

    # shard packing (greedy, big tensors in listed order)
    shards: list[list[tuple[str, tuple[int, ...]]]] = [[]]
    acc = 0
    for name, shape in tensors:
        nb = numel(shape) * 2
        if acc and acc + nb > SHARD_BYTES_TARGET:
            shards.append([])
            acc = 0
        shards[-1].append((name, shape))
        acc += nb
    n = len(shards)
    weight_map = {}
    total_bytes = 0
    for i, entries in enumerate(shards, 1):
        fn = f"model-{i:05d}-of-{n:05d}.safetensors"
        total_bytes += write_sparse_shard(os.path.join(a.out, fn), entries)
        for name, _ in entries:
            weight_map[name] = fn
    with open(os.path.join(a.out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, f, indent=2)

    heads = a.hidden // HEAD_DIM
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": HEAD_DIM,
        "hidden_act": "silu",
        "hidden_size": a.hidden,
        "initializer_range": 0.02,
        "intermediate_size": inter,
        "layer_types": ["full_attention"] * a.layers,
        "max_position_embeddings": a.max_pos,
        "max_window_layers": min(64, a.layers),
        "model_type": "qwen3",
        "num_attention_heads": heads,
        "num_hidden_layers": a.layers,
        "num_key_value_heads": KV_HEADS,
        "pad_token_id": None,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
        "sliding_window": None,
        "tie_word_embeddings": False,
        "transformers_version": "5.6.0",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": VOCAB,
    }
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    donors = sorted(glob.glob(DONOR_GLOB))
    if not donors:
        sys.exit("donor snapshot not found for tokenizer files")
    for fn in TOKENIZER_FILES:
        src = os.path.join(donors[-1], fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(a.out, fn))

    print(f"built {a.out}: {a.layers}L x {a.hidden} (I={inter}, heads={heads}/kv{KV_HEADS}) "
          f"true_params={total_params/1e9:.3f}B total_size={total_bytes} shards={n} (sparse zeros)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
