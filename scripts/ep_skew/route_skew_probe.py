#!/usr/bin/env python
"""route_skew_probe.py — dataset scout for real MoE routing skew (surface_ep_skew.md).

Finds which real datasets naturally skew a checkpoint's routing, per sample.
Vanilla transformers, bf16, forward-only, no LM head. Router capture is a
forward-pre-hook on each MoE layer's `experts` module: every family in scope
(qwen3_moe, qwen3_5_moe, glm4_moe, glm4_moe_lite, hunyuan_v1_moe) calls
`experts(hidden_2d, selected_experts, routing_weights)`, so the captured
indices are the model's ACTUAL routing (post sigmoid/bias/group logic), not a
naive top-k over raw logits.

One invocation = one model x N datasets (model stays resident; dataset swaps
are free). Outputs per cell:
  profiling_results/ep_skew/route_skew_<model>_<dataset>.json
plus gzip'd token-level top-k for the median/P95 samples, and a manifest
entry (idempotent restarts: done cells are skipped unless --overwrite).

Metrics per (sample, layer): hot-GPU share under a static 2-way expert
partition (default contiguous E/2|E/2), top-expert share, Zipf z fit.
Report per dataset: median + P95 over samples of max-over-layers hot share,
per-layer medians, and the domain-mean histogram (per-doc mean) that feeds
the 1M-length prediction (§1.5) and cluster fallback.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

import torch

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

MODELS = {
    # key -> (hf repo, expected model_type, notes)
    "qwen3-30b": ("Qwen/Qwen3-30B-A3B", "qwen3_moe"),
    "qwen3.5-122b": ("Qwen/Qwen3.5-122B-A10B", "qwen3_5_moe"),
    "glm4.5-air": ("zai-org/GLM-4.5-Air", "glm4_moe"),
    "glm4.7-flash": ("zai-org/GLM-4.7-Flash", "glm4_moe_lite"),
    "hunyuan-a13b": ("tencent/Hunyuan-A13B-Instruct", "hunyuan_v1_moe"),
}

# Per-dataset fixed seeds: every model sees the IDENTICAL document order.
DATASETS = {
    # key -> (hf id, config, split, seed, kind)
    "dapo": ("BytedTsinghua-SIA/DAPO-Math-17k", None, "train", 1017, "chat"),
    "megamath": ("IFM/MegaMath", None, "train", 1002, "text"),  # "Megatron-Math" (Du et al. 25) not on HF; MegaMath (web-pro subset, see DATA_FILES) is the closest public math corpus — substitution recorded in every output spec.
    "codeforces": ("open-r1/codeforces", None, "train", 1003, "codeforces"),
    "swebench": ("princeton-nlp/SWE-bench", None, "train", 1004, "swebench"),
    "gpqa": ("Idavidrein/gpqa", "gpqa_main", "train", 1005, "qa"),  # gated
    "openscience": ("nvidia/OpenScience", "OS-Q3-235B-4", "train", 1006, "qa"),  # config required (multi-config repo); Qwen3-235B-generated subset
    "longbench": ("THUDM/LongBench", None, None, 1007, "longbench"),
    "sft_mix": ("HuggingFaceTB/smoltalk", "longalign", "train", 1008, "chat"),
}

# Repos with mixed per-subdir schemas cannot stream repo-wide (CastError):
# pin one uniform-schema subdir per dataset key.
DATA_FILES = {
    "megamath": "megamath-web-pro/*.parquet",
}

HUB_CACHE = os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
)


def snapshot_sha(repo: str) -> str:
    d = os.path.join(HUB_CACHE, "models--" + repo.replace("/", "--"), "snapshots")
    try:
        snaps = sorted(os.listdir(d))
        return snaps[-1] if snaps else "unknown"
    except OSError:
        return "unknown"


def git_hash(repo_dir: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir, text=True
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Dataset adapters -> deterministic iterator of (doc_id, text)
# ---------------------------------------------------------------------------

def _join_fields(row, fields, sep="\n\n"):
    parts = []
    for f in fields:
        v = row.get(f)
        if v is None or v == "None" or v == "":
            continue
        parts.append(str(v))
    return sep.join(parts)


def _render_examples(examples):
    if not examples:
        return ""
    out = []
    for i, ex in enumerate(examples):
        if isinstance(ex, dict):
            out.append(
                f"Example {i + 1}:\nInput:\n{ex.get('input', '')}\nOutput:\n{ex.get('output', '')}"
            )
    return "\n".join(out)


def _chat_to_text(messages, tokenizer):
    """Render a message list the way training/serving feeds the model."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return "\n\n".join(str(m.get("content", "")) for m in messages)


def doc_iterator(ds_key: str, tokenizer, streaming_buffer: int = 5000):
    """Yield (doc_id, text). Deterministic given the per-dataset seed."""
    import datasets as hfds

    repo, config, split, seed, kind = DATASETS[ds_key]

    if kind == "longbench":
        # Script-based repo; read the raw data.zip directly.
        from huggingface_hub import hf_hub_download
        import zipfile

        zp = hf_hub_download(repo, "data.zip", repo_type="dataset")
        rows = []
        with zipfile.ZipFile(zp) as z:
            names = [n for n in z.namelist() if n.endswith(".jsonl") and "_e" not in n]
            for n in sorted(names):
                with z.open(n) as f:
                    for j, line in enumerate(f):
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        rows.append((f"{n}[{j}]", r))
        rng = random.Random(seed)
        rng.shuffle(rows)
        for doc_id, r in rows:
            txt = _join_fields(r, ["context", "input"])
            if txt:
                yield doc_id, txt
        return

    # 2026-08-12: streaming re-resolves via the hub API every run and 429s
    # under concurrent runners — download once into the shared cache instead
    # (codeforces/openscience/megamath are all disk-cheap vs 12T free).
    stream_big = False
    load_kw = {"split": split, "streaming": stream_big}
    if ds_key in DATA_FILES:
        load_kw["data_files"] = DATA_FILES[ds_key]
    ds = None
    for attempt in range(4):  # hub 429s under concurrent runners
        try:
            ds = hfds.load_dataset(repo, config, **load_kw)
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(20 * (attempt + 1))
    if stream_big:
        ds = ds.shuffle(seed=seed, buffer_size=streaming_buffer)
        rows = enumerate(iter(ds))
    else:
        idx = list(range(len(ds)))
        random.Random(seed).shuffle(idx)
        rows = ((i, ds[i]) for i in idx)

    for i, row in rows:
        if kind == "chat":
            msgs = row.get("messages") or row.get("prompt") or []
            doc_id = str(
                (row.get("extra_info") or {}).get("index", i)
                if ds_key == "dapo"
                else i
            )
            txt = _chat_to_text(msgs, tokenizer) if msgs else ""
            # DAPO rows carry the ground truth separately; append it so the
            # pack holds problem + answer text (LLEP packs problem+response).
            if ds_key == "dapo":
                gt = (row.get("reward_model") or {}).get("ground_truth")
                if gt:
                    txt = txt + f"\nAnswer: {gt}"
        elif kind == "codeforces":
            doc_id = str(row.get("id", i))
            txt = _join_fields(
                row, ["title", "description", "input_format", "output_format"]
            )
            ex = _render_examples(row.get("examples"))
            if ex:
                txt += "\n\n" + ex
            txt = _join_fields({"a": txt, "note": row.get("note"), "ed": row.get("editorial")}, ["a", "note", "ed"])
        elif kind == "swebench":
            doc_id = str(row.get("instance_id", i))
            txt = _join_fields(row, ["problem_statement", "patch"])
        elif kind == "qa":
            txt = _join_fields(
                row,
                [
                    "Question", "question", "problem", "input",
                    "Correct Answer", "answer", "output", "response", "solution",
                    "text", "content",
                ],
            )
            # content-hash id: streamed rows have no stable row index, and
            # pack rebuilding must find docs again in any iteration order
            doc_id = "h" + hashlib.sha1(txt.encode()).hexdigest()[:16]
        else:  # generic text
            txt = str(row.get("text") or row.get("content") or "")
            doc_id = "h" + hashlib.sha1(txt.encode()).hexdigest()[:16]
        if txt and len(txt.strip()) > 64:
            yield doc_id, txt


# ---------------------------------------------------------------------------
# Packing (the launch unit)
# ---------------------------------------------------------------------------

@dataclass
class Pack:
    tokens: list = field(default_factory=list)
    spans: list = field(default_factory=list)  # (doc_id, start, end, truncated)


def build_packs(ds_key, tokenizer, seq_len, n_samples, max_docs=200000):
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.pad_token_id or 0
    packs, cur = [], Pack()
    n_docs = 0
    for doc_id, text in doc_iterator(ds_key, tokenizer):
        n_docs += 1
        if n_docs > max_docs:
            break
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(eos)
        pos = 0
        while pos < len(ids):
            room = seq_len - len(cur.tokens)
            take = min(room, len(ids) - pos)
            start = len(cur.tokens)
            cur.tokens.extend(ids[pos : pos + take])
            truncated = (pos + take) < len(ids)
            cur.spans.append((doc_id, start, start + take, truncated))
            pos += take
            if len(cur.tokens) == seq_len:
                packs.append(cur)
                cur = Pack()
                if len(packs) >= n_samples:
                    return packs, n_docs
            # A doc longer than one pack keeps spilling into the next pack.
    if len(packs) >= min(100, max(1, n_samples // 2)):
        print(
            f"[pack] {ds_key} exhausted at {len(packs)}/{n_samples} full packs "
            f"({n_docs} docs) — proceeding with what exists",
            flush=True,
        )
        return packs, n_docs
    raise RuntimeError(
        f"dataset {ds_key} exhausted after {n_docs} docs with only "
        f"{len(packs)}/{n_samples} full {seq_len}-token packs"
    )


# ---------------------------------------------------------------------------
# Model loading + hook capture
# ---------------------------------------------------------------------------

EXPERT_CLASS_SUFFIXES = ("Experts", "NaiveMoe")


def load_model(repo, max_memory=None):
    # AutoModel first: it maps to the backbone/composite class whose
    # checkpoint keys match by construction (incl. the VL-wrapped
    # Qwen3.5-122B), and it skips the LM head entirely (memory note in
    # surface_ep_skew.md). AutoModelForCausalLM is the fallback.
    # max_memory: device_map="auto" reserves no activation headroom and
    # will happily pack a GPU to the brim (measured: 151 GiB weights on
    # GPU0 for the 122B) — callers pass e.g. {0: "115GiB", 1: "170GiB"}.
    from transformers import AutoModelForCausalLM, AutoModel

    last_err = None
    # local_files_only first: the hub API 429s under concurrent
    # unauthenticated runners, and cached models need no network at all.
    for local in (True, False):
        for loader in (AutoModel, AutoModelForCausalLM):
            for attn in (None, "sdpa", "eager"):
                kw = dict(dtype=torch.bfloat16, device_map="auto",
                          local_files_only=local)
                if max_memory:
                    kw["max_memory"] = max_memory
                if attn:
                    kw["attn_implementation"] = attn
                try:
                    m = loader.from_pretrained(repo, **kw)
                    m.eval()
                    return m, f"{loader.__name__}/attn={attn or 'auto'}/local={local}"
                except Exception as e:  # noqa: BLE001
                    last_err = e
    raise RuntimeError(f"could not load {repo}: {last_err}")


def resolve_backbone(model):
    """Descend to the text backbone (skips LM head and vision wrappers)."""
    m, path = model, "model"
    for _ in range(4):
        if hasattr(m, "language_model") and isinstance(
            m.language_model, torch.nn.Module
        ):
            m, path = m.language_model, path + ".language_model"
        elif hasattr(m, "model") and isinstance(m.model, torch.nn.Module):
            m, path = m.model, path + ".model"
        else:
            break
    return m, path


def wrap_experts_row_sliced(model, rows_per_slice):
    """Slice each experts-module call over token rows.

    At 1M tokens the HF experts modules materialize fp32 buffers of
    [tokens*top_k, hidden] (~64 GiB measured on qwen3-30b) in one shot.
    Output rows depend only on their own input rows, so computing in
    slices is exact while capping the transient at rows_per_slice scale.
    Capture is untouched: forward_pre_hooks fire at __call__ with the
    FULL tensors; the wrapper slices inside forward and calls the
    original directly (no per-slice hook fire).
    """
    n = 0
    for _, mod in model.named_modules():
        if type(mod).__name__.endswith(EXPERT_CLASS_SUFFIXES):
            orig = mod.forward

            def sliced(hidden, sel, w, _orig=orig, _n=rows_per_slice, **kw):
                rows = hidden.shape[0]
                if rows <= _n:
                    return _orig(hidden, sel, w, **kw)
                outs = [
                    _orig(hidden[i : i + _n], sel[i : i + _n], w[i : i + _n], **kw)
                    for i in range(0, rows, _n)
                ]
                return torch.cat(outs, dim=0)

            mod.forward = sliced
            n += 1
    return n


MOE_BLOCK_CLASSES = {
    "Qwen3MoeSparseMoeBlock",
    "Qwen3_5MoeSparseMoeBlock",
    "Glm4MoeMoE",
    "Glm4MoeLiteMoE",
    "HunYuanMoEV1Moe",
}


def wrap_moe_block_seq_sliced(model, tokens_per_slice):
    """Slice each whole MoE BLOCK over sequence tokens (batch==1 only).

    The 122B hybrid OOMs on a 24 GiB fp32 [seq, hidden] buffer created
    INSIDE the sparse block (router/shared-expert path) before the experts
    call, so experts-level slicing cannot help. Every path in these blocks
    is per-token, so slicing the block input over dim 1 is exact. The
    experts pre-hook then fires once per slice — RouterCapture concatenates
    slices in order (valid because batch==1 keeps row order = token order).
    """
    n = 0
    for _, mod in model.named_modules():
        if type(mod).__name__ in MOE_BLOCK_CLASSES:
            orig = mod.forward

            def sliced(hidden, *a, _orig=orig, _n=tokens_per_slice, **kw):
                if hidden.dim() != 3 or hidden.shape[0] != 1 or hidden.shape[1] <= _n:
                    return _orig(hidden, *a, **kw)
                outs = [
                    _orig(hidden[:, i : i + _n], *a, **kw)
                    for i in range(0, hidden.shape[1], _n)
                ]
                return torch.cat(outs, dim=1)

            mod.forward = sliced
            n += 1
    return n


class RouterCapture:
    def __init__(self, model):
        self.layer_of = {}
        self.captured = {}
        self.handles = []
        pat = re.compile(r"layers\.(\d+)\.")
        for name, mod in model.named_modules():
            cls = type(mod).__name__
            if cls.endswith(EXPERT_CLASS_SUFFIXES):
                m = pat.search(name)
                if m is None:
                    continue
                li = int(m.group(1))
                self.layer_of[id(mod)] = li
                self.handles.append(
                    mod.register_forward_pre_hook(self._hook, with_kwargs=True)
                )
        self.n_moe_layers = len(self.layer_of)

    def _hook(self, mod, args, kwargs):
        sel = None
        for a in list(args) + list(kwargs.values()):
            if (
                isinstance(a, torch.Tensor)
                and not a.is_floating_point()
                and a.dim() == 2
            ):
                sel = a
                break
        if sel is not None:
            # MoE-block slicing (wrap_moe_block_seq_sliced) makes the experts
            # module fire once per slice — collect and concat in order.
            self.captured.setdefault(self.layer_of[id(mod)], []).append(
                sel.detach().to("cpu", torch.int32)
            )
        return None

    def take(self):
        out = {
            li: (parts[0] if len(parts) == 1 else torch.cat(parts, dim=0))
            for li, parts in self.captured.items()
        }
        self.captured = {}
        return out

    def remove(self):
        for h in self.handles:
            h.remove()


def forward_backbone(backbone, model, input_ids):
    try:
        backbone(input_ids=input_ids, use_cache=False)
        return "backbone"
    except TypeError:
        pass
    try:
        model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
        return "logits_to_keep"
    except TypeError:
        model(input_ids=input_ids, use_cache=False)
        return "full"


def forward_backbone_chunked(backbone, model, input_ids, chunk):
    """Chunked prefill with the model cache carrying cross-chunk state.

    The 122B hybrid's per-layer live transients at 1M tokens (~100 GiB on
    one shard GPU) don't fit any static split; prefilling in chunks caps
    transients at chunk scale while the cache holds KV for the few
    full-attention layers plus the tiny linear-attention recurrent state.
    Causal identity: routing counts equal the single-shot forward.
    RouterCapture concatenates the per-chunk experts fires in order
    (batch==1 keeps row order == token order)."""
    assert input_ids.shape[0] == 1, "chunked prefill requires batch==1"
    tgt = backbone
    try:
        past = None
        for i in range(0, input_ids.shape[1], chunk):
            out = tgt(
                input_ids=input_ids[:, i : i + chunk],
                use_cache=True,
                past_key_values=past,
            )
            past = getattr(out, "past_key_values", None)
            if past is None and i + chunk < input_ids.shape[1]:
                raise RuntimeError("backbone returned no cache — cannot chunk")
        return f"backbone_chunked_{chunk}"
    except TypeError:
        past = None
        for i in range(0, input_ids.shape[1], chunk):
            out = model(
                input_ids=input_ids[:, i : i + chunk],
                use_cache=True,
                past_key_values=past,
                logits_to_keep=1,
            )
            past = getattr(out, "past_key_values", None)
            if past is None and i + chunk < input_ids.shape[1]:
                raise RuntimeError("model returned no cache — cannot chunk")
        return f"model_chunked_{chunk}"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def zipf_fit(counts_sorted_desc):
    import numpy as np

    c = np.asarray(counts_sorted_desc, dtype=np.float64)
    tot = c.sum()
    if tot <= 0:
        return 0.0, 0.0
    p = c / tot
    r = np.arange(1, len(c) + 1, dtype=np.float64)
    best_z, best_sse = 0.0, float("inf")
    for z in np.arange(0.0, 2.51, 0.05):
        q = r ** (-z)
        q /= q.sum()
        sse = float(((p - q) ** 2).sum())
        if sse < best_sse:
            best_z, best_sse = float(z), sse
    return best_z, best_sse


def cell_metrics(sample_layer_counts, partition_a):
    """sample_layer_counts: [S][L][E] torch int64. partition_a: expert ids on
    GPU0 — either one flat list (same split every layer) or a per-layer list
    of lists (real EP places each layer's experts independently; the
    build_placed_sets.py partitions are per-layer)."""
    import numpy as np

    arr = sample_layer_counts.numpy()  # S,L,E
    S, L, E = arr.shape
    if partition_a and isinstance(partition_a[0], (list, tuple)):
        if len(partition_a) != L:
            raise ValueError(
                f"per-layer partition has {len(partition_a)} layers, model captured {L}"
            )
        mask_a = np.zeros((L, E), dtype=bool)
        for l, ids in enumerate(partition_a):
            mask_a[l, list(ids)] = True
        tot = arr.sum(-1)
        tot_safe = np.maximum(tot, 1)
        share_a = (arr * mask_a[None]).sum(-1) / tot_safe
        return _metrics_from_shares(arr, share_a, tot_safe)
    mask_a = np.zeros(E, dtype=bool)
    mask_a[list(partition_a)] = True
    tot = arr.sum(-1)  # S,L
    tot_safe = np.maximum(tot, 1)
    share_a = arr[..., mask_a].sum(-1) / tot_safe
    return _metrics_from_shares(arr, share_a, tot_safe)


def _metrics_from_shares(arr, share_a, tot_safe):
    import numpy as np

    S = arr.shape[0]
    hot = np.maximum(share_a, 1.0 - share_a)  # S,L
    top_share = arr.max(-1) / tot_safe
    max_hot = hot.max(1)  # S
    layer_avg = hot.mean(1)  # S — per-sample hot share averaged over ALL MoE layers
    med_layer_hot = np.median(hot, axis=0)  # L
    # Zipf on the per-sample layer with max hot share
    zs = []
    for s in range(S):
        li = int(hot[s].argmax())
        z, _ = zipf_fit(np.sort(arr[s, li])[::-1])
        zs.append(z)
    return {
        "hot_share": hot,
        "max_hot_per_sample": max_hot,
        "layer_avg_hot_per_sample": layer_avg,
        "mean_layer_avg_hot": float(layer_avg.mean()),
        "median_max_hot": float(np.median(max_hot)),
        "p95_max_hot": float(np.percentile(max_hot, 95)),
        "mean_max_hot": float(max_hot.mean()),
        "median_top_expert_share": float(np.median(top_share.max(1))),
        "per_layer_median_hot": med_layer_hot.tolist(),
        "zipf_z_at_max_layer": zs,
        "median_zipf_z": float(np.median(zs)),
    }


# ---------------------------------------------------------------------------
# Main per-cell run
# ---------------------------------------------------------------------------

def build_packs_from_file(pack_file, tokenizer, seq_len):
    """Rebuild curated packs from a curate_packs.py JSON: exact doc lists,
    refetched deterministically; short packs repeat their own docs to fill."""
    spec = json.load(open(pack_file))
    needed = {}  # ds -> set(doc_id)
    for p in spec["packs"]:
        for ds, did in p["docs"]:
            needed.setdefault(ds, set()).add(str(did))
    texts = {}  # (ds, did) -> text
    for ds, want in needed.items():
        got = 0
        for did, txt in doc_iterator(ds, tokenizer):
            if str(did) in want and (ds, str(did)) not in texts:
                texts[(ds, str(did))] = txt
                got += 1
                if got == len(want):
                    break
        print(f"[pack-file] {ds}: found {got}/{len(want)} docs", flush=True)
    eos = tokenizer.eos_token_id or 0
    packs = []
    for p in spec["packs"]:
        cur = Pack()
        doc_cycle = [(ds, str(did)) for ds, did in p["docs"] if (ds, str(did)) in texts]
        ci = 0
        while len(cur.tokens) < seq_len and doc_cycle:
            ds, did = doc_cycle[ci % len(doc_cycle)]
            ids = tokenizer.encode(texts[(ds, did)], add_special_tokens=False)
            ids.append(eos)
            room = seq_len - len(cur.tokens)
            take = min(room, len(ids))
            start = len(cur.tokens)
            cur.tokens.extend(ids[:take])
            cur.spans.append((f"{ds}:{did}", start, start + take, take < len(ids)))
            ci += 1
        if len(cur.tokens) == seq_len:
            packs.append(cur)
        else:
            print(f"[pack-file] WARNING: pack underfilled ({len(cur.tokens)}) — skipped", flush=True)
    return packs, sum(len(v) for v in needed.values()), spec


def run_cell(model, backbone, cap, tokenizer, model_key, ds_key, args, spec_base, out_dir,
             packs_override=None):
    import numpy as np

    t0 = time.time()
    if packs_override is not None:
        if isinstance(packs_override, tuple):
            packs, n_docs_scanned = packs_override
        else:  # bare pack list from build_packs_from_file
            packs, n_docs_scanned = packs_override, len(packs_override)
    else:
        packs, n_docs_scanned = build_packs(
            ds_key, tokenizer, args.seq_len, args.num_samples
        )
    t_pack = time.time() - t0
    S, T, B = len(packs), args.seq_len, args.batch_size

    # Window mode: forward each long pack as independent W-token windows and
    # aggregate counts back per source pack. Routing then reflects W-token
    # context (recorded in spec) — the bounded-memory proxy for models whose
    # full-context transients fit no static split (122B hybrid at 1M).
    win_of, n_src_packs = None, None
    if getattr(args, "window_tokens", 0) and packs and len(packs[0].tokens) > args.window_tokens:
        W = args.window_tokens
        wpacks, win_of = [], []
        for pi, p in enumerate(packs):
            for i in range(0, len(p.tokens) - W + 1, W):
                wpacks.append(Pack(tokens=p.tokens[i : i + W], spans=[]))
                win_of.append(pi)
        n_src_packs = len(packs)
        packs = wpacks
        S, T = len(packs), W

    dev = next(backbone.parameters()).device
    doc_counts = {}  # (doc_id) -> L,E accumulated over its spans
    state = {"fwd_path": None, "layer_ids": None}

    def _forward_all(bsz):
        tok_topk = {}
        with torch.inference_mode():
            for b0 in range(0, S, bsz):
                batch = packs[b0 : b0 + bsz]
                ids = torch.tensor(
                    [p.tokens for p in batch], dtype=torch.long, device=dev
                )
                if args.prefill_chunk_tokens > 0 and ids.shape[0] == 1:
                    state["fwd_path"] = forward_backbone_chunked(
                        backbone, model, ids, args.prefill_chunk_tokens
                    )
                else:
                    state["fwd_path"] = forward_backbone(backbone, model, ids)
                got = cap.take()
                if not got:
                    raise RuntimeError(
                        "no router capture fired — hook targets missing"
                    )
                if state["layer_ids"] is None:
                    state["layer_ids"] = sorted(got.keys())
                for j, _ in enumerate(batch):
                    s = b0 + j
                    per_layer = []
                    for li in state["layer_ids"]:
                        sel = got[li]  # [B*T, k] int32
                        k = sel.shape[1]
                        per_layer.append(sel.view(len(batch), T, k)[j])
                    tok_topk[s] = torch.stack(per_layer)  # L,T,k
                del got, ids
        return tok_topk

    t1 = time.time()
    tok_topk = None
    while True:
        try:
            tok_topk = _forward_all(B)
            break
        except torch.OutOfMemoryError:
            cap.take()  # flush partial captures from the aborted batch
            torch.cuda.empty_cache()
            if B <= 1:
                raise
            B //= 2
            print(f"[cell] OOM — retrying with batch_size={B}", flush=True)
    fwd_path = state["fwd_path"]
    layer_ids = state["layer_ids"]
    n_layers = len(layer_ids)
    # Determine E from config
    E = infer_num_experts(model)
    L = n_layers
    sample_counts = torch.zeros((S, L, E), dtype=torch.int64)
    for s in range(S):
        sl = tok_topk[s]  # L,T,k
        for li in range(L):
            sample_counts[s, li] = torch.bincount(
                sl[li].reshape(-1).to(torch.int64), minlength=E
            )
    # per-document counts
    doc_meta = []
    for s, p in enumerate(packs):
        for (doc_id, st, en, trunc) in p.spans:
            key = doc_id
            c = torch.zeros((L, E), dtype=torch.int64)
            sl = tok_topk[s]  # L,T,k
            for li in range(L):
                c[li] = torch.bincount(
                    sl[li][st:en].reshape(-1).to(torch.int64), minlength=E
                )
            if key in doc_counts:
                doc_counts[key] += c
            else:
                doc_counts[key] = c
            doc_meta.append(
                {"doc": doc_id, "sample": s, "start": st, "end": en, "truncated": trunc}
            )
    if win_of is not None:
        agg = torch.zeros((n_src_packs, L, E), dtype=torch.int64)
        for w, pi in enumerate(win_of):
            agg[pi] += sample_counts[w]
        sample_counts = agg
        S = n_src_packs
    t_fwd = time.time() - t1

    partition_a = list(range(E // 2)) if args.partition == "contiguous" else json.load(open(args.partition))
    if isinstance(partition_a, dict):
        partition_a = partition_a["layers"]  # build_placed_sets.py per-layer format
    met = cell_metrics(sample_counts, partition_a)

    # token-level gzip for median + P95 samples (by max hot share)
    order = np.argsort(met["max_hot_per_sample"])
    keep = {int(order[len(order) // 2]): "median", int(order[max(0, int(round(0.95 * (len(order) - 1))))]): "p95"}
    os.makedirs(out_dir, exist_ok=True)
    gz_paths = {}
    if win_of is None:  # window mode: tok_topk keys are windows, skip gz
        for s, tag in keep.items():
            gz = os.path.join(out_dir, f"route_skew_{model_key}_{ds_key}_topk_{tag}_s{s}.npz.gz")
            with gzip.open(gz, "wb") as f:
                torch.save(tok_topk[s].to(torch.int16), f)
            gz_paths[tag] = os.path.basename(gz)

    dom_mean = (
        torch.stack(list(doc_counts.values())).to(torch.float64)
        if doc_counts
        else torch.zeros((L, E), dtype=torch.float64).unsqueeze(0)
    )
    dom_mean = (dom_mean / dom_mean.sum(-1, keepdim=True).clamp(min=1)).mean(0)  # L,E

    out = {
        "spec": {
            **spec_base,
            "dataset_key": ds_key,
            "dataset_id": DATASETS.get(ds_key, (None,) * 5)[0],
            "dataset_config": DATASETS.get(ds_key, (None,) * 5)[1],
            "dataset_split": DATASETS.get(ds_key, (None,) * 5)[2],
            "dataset_seed": DATASETS.get(ds_key, (None,) * 5)[3],
            "seq_len": T,
            "window_tokens": getattr(args, "window_tokens", 0) or None,
            "pack_tokens_total": args.seq_len if win_of is not None else None,
            "batch_size": B,
            "num_samples": S,
            "docs_scanned": n_docs_scanned,
            "forward_path": fwd_path,
            "n_moe_layers": L,
            "moe_layer_ids": layer_ids,
            "num_routed_experts": E,
            "partition": args.partition,
            "special_tokens": "eos-separator only, no BOS",
            "note_megamath": "MegaMath (IFM/MegaMath) substitutes LLEP's 'Megatron-Math' (not on HF)" if ds_key == "megamath" else None,
        },
        "summary": {
            "mean_layer_avg_hot_gpu_share": met["mean_layer_avg_hot"],
            "layer_avg_hot_per_sample": [round(float(x), 4) for x in met["layer_avg_hot_per_sample"]],
            "median_max_hot_gpu_share": met["median_max_hot"],
            "p95_max_hot_gpu_share": met["p95_max_hot"],
            "mean_max_hot_gpu_share": met["mean_max_hot"],
            "median_top_expert_share": met["median_top_expert_share"],
            "median_zipf_z": met["median_zipf_z"],
            "per_layer_median_hot": met["per_layer_median_hot"],
            "timing_s": {"pack": round(t_pack, 1), "forward": round(t_fwd, 1)},
        },
        "samples": {
            "max_hot_per_sample": met["max_hot_per_sample"].tolist(),
            "zipf_z_at_max_layer": met["zipf_z_at_max_layer"],
            "counts": sample_counts.tolist(),
        },
        "docs": {
            "spans": doc_meta,
            "counts_gz": f"route_skew_{model_key}_{ds_key}_docs.json.gz",
            "domain_mean_hist": dom_mean.tolist(),
        },
        "packs": [[sp[0] for sp in p.spans] for p in packs],
        "topk_gz": gz_paths,
    }
    # Per-document per-layer counts can be ~100 MB raw on short-doc
    # datasets — sidecar gzip keeps the main JSON readable.
    with gzip.open(os.path.join(out_dir, out["docs"]["counts_gz"]), "wt") as f:
        json.dump({k: v.tolist() for k, v in doc_counts.items()}, f)
    path = os.path.join(out_dir, f"route_skew_{model_key}_{ds_key}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    return path, met, {"pack_s": t_pack, "fwd_s": t_fwd, "docs": len(doc_counts)}


def infer_num_experts(model):
    cfg = model.config
    for c in (getattr(cfg, "text_config", None), cfg):
        if c is None:
            continue
        for k in ("num_experts", "n_routed_experts"):
            v = getattr(c, k, None)
            if isinstance(v, int):
                return v
            if isinstance(v, list):
                return int(v[0])
    raise RuntimeError("cannot infer expert count from config")


def update_manifest(out_dir, cell, entry):
    """flock'd read-modify-write — two replicas share one manifest."""
    import fcntl

    mpath = os.path.join(out_dir, "manifest.json")
    lock = open(mpath + ".lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        man = {}
        if os.path.exists(mpath):
            try:
                man = json.load(open(mpath))
            except Exception:
                man = {}
        man[cell] = entry
        tmp = mpath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(man, f, indent=1, sort_keys=True)
        os.replace(tmp, mpath)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--datasets", required=True, help="comma list of dataset keys")
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-samples", type=int, default=104)
    ap.add_argument("--partition", default="contiguous")
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--pack-file", default=None,
                    help="curate_packs.py JSON: run its exact packs as one cell; "
                    "--datasets then supplies the cell LABEL (e.g. curated_mathmix)")
    ap.add_argument(
        "--window-tokens", type=int, default=0,
        help="verify long packs as independent W-token windows (counts "
        "aggregated per source pack; routing context = W, recorded in spec); "
        "bounded-memory proxy when full-context transients fit no split",
    )
    ap.add_argument(
        "--max-packs", type=int, default=0,
        help="pack-file mode: cap number of packs (0 = all)",
    )
    ap.add_argument(
        "--prefill-chunk-tokens", type=int, default=0,
        help="chunked prefill with cache (batch==1): caps per-layer live "
        "transients at chunk scale — required for the 122B hybrid at 1M "
        "whose deltanet live-set (~100 GiB/GPU) fits no static split; 0 = off",
    )
    ap.add_argument(
        "--moe-block-tokens-per-slice", type=int, default=0,
        help="slice whole MoE blocks over sequence tokens (batch==1 only; "
        "exact; kills the fp32 [seq,hidden] router/shared transients that "
        "OOM hybrids at 1M); takes precedence over --moe-rows-per-slice",
    )
    ap.add_argument(
        "--moe-rows-per-slice", type=int, default=0,
        help="slice experts-module calls over token rows (exact; caps the "
        "fp32 [rows,hidden] transient that OOMs at 1M tokens); 0 = off",
    )
    ap.add_argument(
        "--max-memory",
        default=None,
        help="per-visible-GPU GiB caps, comma list (e.g. '115,170') — leaves "
        "activation headroom that device_map=auto refuses to reserve",
    )
    args = ap.parse_args()

    repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = args.out or os.path.join(repo_dir, "profiling_results", "ep_skew")
    os.makedirs(out_dir, exist_ok=True)

    ds_keys = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not args.pack_file:
        for d in ds_keys:
            if d not in DATASETS:
                sys.exit(f"unknown dataset key {d}")
    elif len(ds_keys) != 1:
        sys.exit("--pack-file mode takes exactly one label in --datasets")

    # Skip cells already in the manifest (idempotent restart) before loading.
    mpath = os.path.join(out_dir, "manifest.json")
    man = {}
    if os.path.exists(mpath) and not args.overwrite:
        try:
            man = json.load(open(mpath))
        except Exception:
            man = {}
    todo = []
    for d in ds_keys:
        cell = f"{args.model}|{d}"
        if man.get(cell, {}).get("status") == "done":
            print(f"[skip] {cell} already done", flush=True)
        else:
            todo.append(d)
    if not todo:
        print("[probe] nothing to do", flush=True)
        return

    repo, expect_type = MODELS[args.model]
    print(f"[probe] loading {repo} ({expect_type}) ...", flush=True)
    t0 = time.time()
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    except Exception:  # hub API 429s under concurrent unauthenticated runners
        tokenizer = AutoTokenizer.from_pretrained(repo)
    max_memory = None
    if args.max_memory:
        max_memory = {
            i: f"{v.strip()}GiB"
            for i, v in enumerate(args.max_memory.split(","))
        }
    model, load_path = load_model(repo, max_memory=max_memory)
    backbone, bb_path = resolve_backbone(model)
    cap = RouterCapture(model)
    if args.moe_block_tokens_per_slice > 0:
        nb = wrap_moe_block_seq_sliced(model, args.moe_block_tokens_per_slice)
        print(f"[probe] MoE-block seq-slicing at {args.moe_block_tokens_per_slice} tokens on {nb} blocks", flush=True)
    elif args.moe_rows_per_slice > 0:
        nw = wrap_experts_row_sliced(model, args.moe_rows_per_slice)
        print(f"[probe] experts row-slicing at {args.moe_rows_per_slice} rows on {nw} modules", flush=True)
    print(
        f"[probe] loaded in {time.time() - t0:.0f}s via {load_path}; backbone={bb_path}; "
        f"moe_layers={cap.n_moe_layers}",
        flush=True,
    )
    if cap.n_moe_layers == 0:
        sys.exit("no expert modules found — hook suffixes need extending")

    spec_base = {
        "model_key": args.model,
        "model_id": repo,
        "model_revision": snapshot_sha(repo),
        "model_type_expected": expect_type,
        "dtype": "bfloat16",
        "loader": load_path,
        "backbone": bb_path,
        "capture": "forward_pre_hook on experts module (actual selected indices)",
        "probe_git": git_hash(repo_dir),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device_map": "auto",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
    }

    for d in todo:
        cell = f"{args.model}|{d}"
        print(f"[cell] {cell} starting", flush=True)
        try:
            packs_override = None
            if args.pack_file:
                res = build_packs_from_file(
                    args.pack_file, tokenizer, args.seq_len
                )
                # co-edited helper has returned a list, a 2-tuple and a
                # 3-tuple (packs, n_docs, spec) at different times — accept all
                if isinstance(res, tuple):
                    plist = res[0]
                    n_found = res[1] if len(res) > 1 else len(res[0])
                else:
                    plist, n_found = res, len(res)
                if args.max_packs > 0:
                    plist = plist[: args.max_packs]
                packs_override = (plist, n_found)
                spec_base["pack_file"] = os.path.basename(args.pack_file)
            path, met, extra = run_cell(
                model, backbone, cap, tokenizer, args.model, d, args, spec_base, out_dir,
                packs_override=packs_override,
            )
            update_manifest(
                out_dir,
                cell,
                {
                    "status": "done",
                    "json": os.path.basename(path),
                    "median_max_hot": round(met["median_max_hot"], 4),
                    "p95_max_hot": round(met["p95_max_hot"], 4),
                    "median_zipf_z": round(met["median_zipf_z"], 3),
                    "ts": spec_base["timestamp"],
                },
            )
            print(
                f"[cell] {cell} DONE median_max_hot={met['median_max_hot']:.4f} "
                f"p95={met['p95_max_hot']:.4f} z_med={met['median_zipf_z']:.2f} "
                f"docs={extra['docs']} pack={extra['pack_s']:.0f}s fwd={extra['fwd_s']:.0f}s",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            update_manifest(
                out_dir, cell, {"status": "failed", "error": f"{type(e).__name__}: {e}"[:300]}
            )
            print(f"[cell] {cell} FAILED {type(e).__name__}: {e}", flush=True)

    cap.remove()
    print("[probe] all requested cells processed", flush=True)


if __name__ == "__main__":
    main()
