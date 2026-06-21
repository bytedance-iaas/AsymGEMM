# fix_spda.md — SDPA-only activation recompute for Llama-4 attention (staged)

## Summary
- **Goal:** free the **~6.7 GiB forward-resident SDPA output `O`** that activation *offload* cannot
  reach, by **recomputing only the SDPA call** in backward. Compose with `attn_act` (keep offloading
  q/k/v) — do **not** replace it.
- **Lever, not requirement:** the shared-MLP silu fix already puts both target cells under the
  28,094 MiB bar (`layer_act` = 26,925). This is optional headroom → peak ~26.9 → ~20 GiB.
- **Target cells:** `layer_act` (`none|T|T|T|F`) and `layer_gc` (`none|T|T|F|T`), both have `attn_act` ON.
- **Toggle = the 6th policy field.** `ASYMM_EXP_ACT_POLICIES` grows from 5 to 6 fields:
  `expert_policy|exp_act|attn_act|layer_act|layer_gc|sdpa_recompute`, e.g.
  `none|true|true|true|false|true` or `none|true|false|false|false|true`. Default 6th = `false` → exact
  no-op (5-field policies stay valid). The field drives env
  `ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_SDPA_RECOMPUTE`, which the module already reads.

Execute the stages **in order**. Each stage has a **Validation Gate** — do not proceed until it is ✅.

---

## Background (grounded, condensed)

**Why offload can't, recompute can.** `attn_act` offloads attention saved tensors via Python
`saved_tensors_hooks` (`attention_activation_offload.py:240`). The fused SDPA kernel saves `O` inside
its **C++ autograd node**, which the Python hooks do not intercept — proven: snapshot replay puts
6.7 GiB at `sdpa_attention.py:92` (the `O` alloc), and `_should_offload`
(`attention_activation_offload.py:243`) *would accept* `O` (bf16, 160 MiB/layer, non-leaf,
requires_grad), yet it is never offloaded. Recompute removes the kernel's save entirely (no_grad
forward builds no node), regenerating `O` per-layer in backward.

**The crux (why resident `O` actually drops).** `O` (alloc at `sdpa:92`) has two savers:
1. SDPA's C++ backward node — **not** offloadable (pins HBM).
2. `o_proj`'s saved input — **offloadable** by `attn_act`, but blocked while (1) holds it.
Checkpointing the SDPA call deletes saver (1); `attn_act` then offloads (2) → HBM `O` released.

**The seam (verified).** `transformers/models/llama4/modeling_llama4.py`,
`Llama4Attention.forward`:
```
# 394  attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, ...)  # PER-CALL
# 397  attn_output, attn_weights = attention_interface(self, q, k, v, attention_mask, dropout=.., scaling=.., **kw)
# 409  attn_output = self.o_proj(attn_output)   # OUTSIDE the interface
```
`ALL_ATTENTION_FUNCTIONS` is a registry (`.register`/`.get`/`.get_interface`) → swap without forking
transformers. **Two call sites:** 397 (text) and **824 (vision)** → scope to the **text** config only.
`attention_dropout = 0.0` (verified) → recompute is deterministic.

**Existing `attention_checkpoint.py`** = the *whole-attention* wrapper (recomputes projections too;
stacking it on `attn_act` is what regressed +8 GiB). Not used here; left as-is.

---

## Stage 0 — Lock baseline + attribution tool

**Changes:** none (tooling + measurement only).

Create the reusable peak-attribution script `scripts/testing/peak_attrib.py`:
```python
#!/usr/bin/env python
"""Replay a torch CUDA memory snapshot to its peak live-set and attribute by allocation frame."""
import pickle, sys, collections
from pathlib import Path
_TORCH = ("site-packages/torch/", "/torch/cuda/", "/torch/_", "/torch/autograd/",
          "/torch/nn/modules/module.py", "c10/")
def uf(fr):
    fr = fr or []
    for f in fr:
        fn = str(f.get("filename", ""))
        if any(s in fn for s in ("asym_gemm/", "transformers/models/", "llamafactory/")):
            return f"{Path(fn).name}:{f.get('line','?')}:{f.get('name','?')}"
    for f in fr:
        fn = str(f.get("filename", ""))
        if not any(m in fn for m in _TORCH):
            return f"{Path(fn).name}:{f.get('line','?')}:{f.get('name','?')}"
    return fr[0].get("name", "?") if fr else "<none>"
def main(path):
    snap = pickle.load(open(path, "rb"))
    best = None
    for dev in snap.get("device_traces") or []:
        live = {}; tot = 0; peak = 0; pl = None
        for ev in dev:
            a = ev.get("action"); addr = ev.get("addr"); sz = int(ev.get("size") or 0)
            if a == "alloc":
                live[addr] = (sz, ev.get("frames")); tot += sz
                if tot > peak: peak = tot; pl = dict(live)
            elif a in ("free_completed", "free_requested"):
                if addr in live: tot -= live[addr][0]; del live[addr]
        if pl and (best is None or peak > best[0]): best = (peak, pl)
    peak, live = best
    byf = collections.defaultdict(int)
    for _, (sz, fr) in live.items(): byf[uf(fr)] += sz
    print(f"PEAK {peak/2**20:,.0f} MiB across {len(live)} blocks")
    for f, b in sorted(byf.items(), key=lambda x: -x[1])[:12]:
        if b/2**20 >= 40: print(f"  {b/2**20:9,.0f} MiB  {f}")
if __name__ == "__main__":
    main(sys.argv[1])
```

The validation is a **per-cell A/B that flips only the 6th field** (everything else fixed):

| cell | OFF (baseline) | ON |
|---|---|---|
| layer_act | `none\|true\|true\|true\|false\|false` | `none\|true\|true\|true\|false\|true` |
| layer_gc  | `none\|true\|true\|false\|true\|false` | `none\|true\|true\|false\|true\|true` |

The OFF form (`…|false`) is byte-identical to the existing 5-field cell, so it must reproduce the
current peaks (layer_act ≈ **26,925 MiB**) — that doubles as a sanity check that the 6-field parser
changed nothing.

**Capture the baselines with the real timing harness** `scripts/lf/profile_lora_lf_test.sh`
(`PROFILERS=both`, real workload). Efficiency is a first-class metric — sdpa_recompute trades latency
for memory, so **memory-only is not acceptance**.

Pull timing **and** memory with the new metrics tools (one table per model, all configs at once — these
replace hand-rolled JSON parsing):
```
scripts/lf/show_status.sh  <profiling_dir>     # confirm each leaf is OK (not OOM/FAILED/RUNNING/INCOMPLETE)
scripts/lf/show_metrics.sh <profiling_dir>     # forward/backward/optimizer/step (s) + forward/backward/step (GiB)
```
Plus, per leaf, the two things `show_metrics` does not cover:
```
.venv/bin/python scripts/testing/peak_attrib.py <leaf>/memory_snapshot.pickle   # sdpa:92 frame MiB
grep -m1 loss <leaf>/train.log                                                  # correctness ref L0
```

**Validation Gate 0 ✅** when recorded for both OFF cells. **Real anchors** from `show_metrics.sh` on the
current `layer_act` run: forward **7.396 s**, backward **59.320 s**, optimizer **1.974 s**, step
**72.139 s**, step memory **26.294 GiB**; `peak_attrib.py` `sdpa:92` ≈ **6,733 MiB**; loss `L0`.
Notes: `show_metrics` "step (GiB)" is the step peak-allocated (= the ~26,925 MiB number; the fwd/bwd GiB
columns currently mirror it — use **step (GiB)** for the memory comparison). For a production-latency
read unaffected by per-range sync, optionally capture a `PROFILE_SYNC=false` pair; the ON/OFF *delta* is
valid either way since the sync overhead is constant.

---

## Stage 1 — Core module + isolated correctness

**Change 1.1 — new file `asym_gemm/training/sdpa_recompute.py`:**
```python
from __future__ import annotations
import os
import torch
from torch.utils.checkpoint import checkpoint
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

_INTERFACE_NAME = "asym_sdpa_recompute"
_FALSEY = {"", "0", "false", "no", "off"}

def _enabled() -> bool:
    # accept the direct flag OR the harness-forwarded ASYM_GEMM_LF_CONFIG_* mirror
    for name in ("ASYMM_ATTN_SDPA_RECOMPUTE", "ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_SDPA_RECOMPUTE"):
        raw = os.environ.get(name)
        if raw is not None and raw.strip().lower() not in _FALSEY:
            return True
    return False

def _make_checkpointed(base_fn):
    def _checkpointed(module, query, key, value, attention_mask=None, **kwargs):
        # eval / no-grad: passthrough, zero overhead
        if not (module.training and torch.is_grad_enabled()):
            return base_fn(module, query, key, value, attention_mask, **kwargs)
        # checkpoint ONLY the SDPA math; q/k/v are the recompute inputs (offloaded by attn_act);
        # non-tensor args ride the closure. SDPA's attn_weights is None -> keep output a tensor.
        def _run(q, k, v):
            out, _ = base_fn(module, q, k, v, attention_mask, **kwargs)
            return out
        attn_output = checkpoint(_run, query, key, value, use_reentrant=False)
        return attn_output, None
    return _checkpointed

def install_sdpa_recompute(model) -> bool:
    """Idempotent, self-gated, text-scoped. Returns True iff the interface is now active."""
    if not _enabled():
        return False
    try:
        cfg = getattr(model, "config", None)
        text_cfg = getattr(cfg, "text_config", cfg)
        base_impl = getattr(text_cfg, "_attn_implementation", "sdpa") or "sdpa"
        if base_impl == _INTERFACE_NAME:                 # already installed
            return True
        base_fn = ALL_ATTENTION_FUNCTIONS.get(base_impl) or ALL_ATTENTION_FUNCTIONS.get("sdpa")
        if base_fn is None:
            return False
        ALL_ATTENTION_FUNCTIONS.register(_INTERFACE_NAME, _make_checkpointed(base_fn))
        text_cfg._attn_implementation = _INTERFACE_NAME  # plain attr; get_interface reads it
        print(f"[asym] sdpa_recompute installed (base={base_impl}) on text attention", flush=True)
        return True
    except Exception as exc:                             # never break the run
        print(f"[asym] sdpa_recompute install skipped: {exc!r}", flush=True)
        return False

__all__ = ["install_sdpa_recompute"]
```

**Change 1.2 — new test `scripts/testing/test_sdpa_recompute.py`** (proves determinism + grad identity,
the entire correctness claim, in isolation):
```python
import torch, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def main():
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    B, H, S, D = 4, 8, 512, 128
    mk = lambda: torch.randn(B, H, S, D, device=dev, dtype=dt, requires_grad=True)
    q, k, v = mk(), mk(), mk()
    sdpa = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
    o1 = sdpa(q, k, v); o1.float().pow(2).sum().backward()
    g1 = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v): t.grad = None
    o2 = checkpoint(sdpa, q, k, v, use_reentrant=False); o2.float().pow(2).sum().backward()
    g2 = [t.grad.clone() for t in (q, k, v)]
    assert torch.equal(o1, o2), f"output differs: {(o1-o2).abs().max()}"
    gdiff = max((a - b).abs().max().item() for a, b in zip(g1, g2))
    assert gdiff == 0.0, f"grad differs: {gdiff}"
    print(f"OK  output exact-equal, grad max-diff={gdiff}")

if __name__ == "__main__":
    main()
```

**Validation Gate 1 ✅:**
```
.venv/bin/python scripts/testing/test_sdpa_recompute.py        # -> prints "OK  output exact-equal, grad max-diff=0.0"
.venv/bin/python -c "import asym_gemm.training.sdpa_recompute"  # imports clean
```
Do not proceed unless output is **exact-equal** and grad diff is **0.0** (deterministic recompute).

---

## Stage 2 — Gated install into the model build + no-op safety

**Change 2.1 — `asym_gemm/integrations/lf.py`, add import** (next to the other attention-offload
imports, ~lines 20-25):
```python
from ..training.sdpa_recompute import install_sdpa_recompute
```

**Change 2.2 — `asym_gemm/integrations/lf.py`, call it once** in the attn_act wrap function
`_wrap_attention_saved_tensor_offload_modules` (its first param is `model`). Anchor: the loop that ends
with `install_attention_saved_tensor_offload(module)` at **lf.py:1520**. Insert **after the for-loop,
before the `if strict and not wrapped:` check (~line 1523)**:
```python
        install_attention_saved_tensor_offload(module)
        wrapped.append(name)

    # SDPA-only recompute pairs with attn_act; self-gated on ASYMM_ATTN_SDPA_RECOMPUTE, text-scoped.
    install_sdpa_recompute(model)            # <-- ADD (model is the fn's first param; idempotent no-op when flag off)

    if strict and not wrapped:
        raise RuntimeError(...)
```
(Model-level config install — does **not** need the per-module `_is_text_attention_module_name`
check; scoping is via `text_config`.)

**Validation Gate 2 ✅** — a tiny driver (`scripts/testing/check_sdpa_install.py`) that loads the model
through the asym LF path with `attn_act` ON, and runs one fwd+bwd:

| flag | assert |
|---|---|
| `ASYMM_ATTN_SDPA_RECOMPUTE=0` | `install_sdpa_recompute(model) is False`; `text_cfg._attn_implementation == "sdpa"` (unchanged); fwd+bwd runs; loss == baseline `L0` |
| `ASYMM_ATTN_SDPA_RECOMPUTE=1` | returns `True`; `text_cfg._attn_implementation == "asym_sdpa_recompute"`; **`vision_config._attn_implementation == "sdpa"`** (untouched); fwd+bwd runs, no exception |

Do not proceed unless flag-OFF is byte-identical to today **and** flag-ON installs text-only + steps cleanly.

---

## Stage 3 — Add SDPA-recompute as the 6th policy field

The toggle is the **6th field** of `ASYMM_EXP_ACT_POLICIES`
(`expert_policy|exp_act|attn_act|layer_act|layer_gc|sdpa_recompute`). It is parsed **exactly like the
5th field (`ASYMM_LAYER_GC`)** and ends up as `ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_SDPA_RECOMPUTE`, which the
module's `_enabled()` already reads. **No Python change** — harness parser/forwarder only.

It is **orthogonal** (valid with any combination, e.g. `none|true|false|false|false|true`); the memory
win is largest with `attn_act=true`. The `(expact||attnact||layeract) && policy!=none` guard
(`profile_lora_lf.sh:444`) stays as-is — sdpa_recompute is a recompute, not an offload, so it adds no
new constraint.

**Rule: clone every `layergc` / `ASYMM_LAYER_GC` site, substituting `sdparecomp` / `ASYMM_ATTN_SDPA_RECOMPUTE`.**
Enumerate them with:
```
grep -niE 'layergc|ASYMM_LAYER_GC|LAYER_GC' scripts/lf/profile_lora_lf.sh scripts/lf/run_lf_lora_sft.sh
```

**Change 3.1 — `scripts/lf/profile_lora_lf.sh`** (parser, tags, labels, forward):

| site (anchor) | layergc reference | add the sibling |
|---|---|---|
| tag fn (~L399) | `layergc_tag(){ … layergc0/1 }` | `sdparecomp_tag(){ … sdparecomp0/1 }` |
| parser decl (L423) | `… layeract_part layergc_part … layeract layergc` | append `sdparecomp_part … sdparecomp` |
| field count (L426) | `(( …==4 \|\| …==5 ))` | `(( …==4 \|\| …==5 \|\| ${#fields[@]}==6 ))` |
| extract (L431) | `layergc_part="${fields[4]:-false}"` | `sdparecomp_part="${fields[5]:-false}"` |
| non-empty (L432) | `… -n "${layergc_part}"` | add `&& -n "${sdparecomp_part}"` |
| normalize (L437) | `layergc="$(bool_value "${layergc_part}")"` | `sdparecomp="$(bool_value "${sdparecomp_part}")"` |
| re-emit (L447) | `printf '%s\|%s\|%s\|%s\|%s\n' … "${layergc}"` | make it 6: `printf '%s\|%s\|%s\|%s\|%s\|%s\n' … "${layergc}" "${sdparecomp}"` |
| defaults (L598) | `ASYMM_LAYER_GC=false; layergc_label="$(layergc_tag false)"` | `ASYMM_ATTN_SDPA_RECOMPUTE=false; sdparecomp_label="$(sdparecomp_tag false)"` |
| values array (~L2077-78) | `ASYMM_LAYER_GC="${layergc_values[0]}"; layergc_label=…` | build `sdparecomp_values` like `layergc_values`, then `ASYMM_ATTN_SDPA_RECOMPUTE="${sdparecomp_values[0]}"; sdparecomp_label="$(sdparecomp_tag "$ASYMM_ATTN_SDPA_RECOMPUTE")"` |
| kt path (~L3165) | `ASYMM_LAYER_GC="${policy_tail#*\|}"; layergc_label=…` | parse the 6th field → `ASYMM_ATTN_SDPA_RECOMPUTE` + `sdparecomp_label` |
| path label (~L1313) | `…__${layeract_label}__${layergc_label}` | append `__${sdparecomp_label}` |
| run_id (~L2388) | `…_${layeract_label}_${layergc_label}_…` | add `_${sdparecomp_label}` |
| subshell locals (~L1355, ~L2342) | `…layergc_label… ASYMM_LAYER_GC="${ASYMM_LAYER_GC}"` | add `sdparecomp_label` and `ASYMM_ATTN_SDPA_RECOMPUTE="${ASYMM_ATTN_SDPA_RECOMPUTE}"` |
| child env (~L2546) | `ASYMM_LAYER_GC="${ASYMM_LAYER_GC}"` | `ASYMM_ATTN_SDPA_RECOMPUTE="${ASYMM_ATTN_SDPA_RECOMPUTE}"` |
| config forward (~L2588) | `ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_GC="${ASYMM_LAYER_GC}"` | `ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_SDPA_RECOMPUTE="${ASYMM_ATTN_SDPA_RECOMPUTE}"` |

(Optional: mirror the `--layergc` plot-arg at L1541/1557/2832 to surface the 6th field in plots — not
required for the runs.)

**Change 3.2 — `scripts/lf/run_lf_lora_sft.sh`:**

| site (anchor) | layergc reference | add the sibling |
|---|---|---|
| default (L68) | `ASYMM_LAYER_GC=${ASYMM_LAYER_GC:-false}` | `ASYMM_ATTN_SDPA_RECOMPUTE=${ASYMM_ATTN_SDPA_RECOMPUTE:-false}` |
| validate+tag (L392-395) | `case "${ASYMM_LAYER_GC,,}" … LAYER_GC_TAG=layergc1/0 … exit 2` | same case for `ASYMM_ATTN_SDPA_RECOMPUTE` → `SDPARECOMP_TAG=sdparecomp1/0` |
| config forward (next to `ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_GC=…`) | `ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_GC="${ASYMM_LAYER_GC}"` | `ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_SDPA_RECOMPUTE="${ASYMM_ATTN_SDPA_RECOMPUTE}"` |

**Validation Gate 3 ✅** — run a cell with the 6th field ON:
```
NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 ENABLE_LIGER_KERNEL=false GPU_POOL=3 \
MODEL_SPECS='meta-llama/Llama-4-Scout-17B-16E|1' WORKLOADS='4096|4|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|true|false|true' \
PROFILE_MEMORY_BREAKDOWN=true PROFILE_MEMORY_SNAPSHOT=true PROFILE_SYNC=true OVERWRITE=true \
bash scripts/lf/profile_lora_lf.sh

ls -d <profiling>/...__layeract1__layergc0__sdparecomp1__...   # dir tag carries the 6th field
grep -c ASYMM_ATTN_SDPA_RECOMPUTE <run>/b4_s4096_ga1/command.txt   # >= 1
grep "sdpa_recompute installed"  <run>/b4_s4096_ga1/train.log     # present
```
Proceed only when the **dir tag shows `sdparecomp1`**, the flag is in `command.txt`, **and** the install
log line appears (proves the field flowed policy → env → subprocess → interface).

---

## Stage 4 — E2E acceptance: memory **and** efficiency (both cells)

**Changes:** none — measurement only. Run each cell's A/B pair (OFF + ON) in **one** sweep with
`scripts/lf/profile_lora_lf_test.sh` (`PROFILERS=both`). test.sh now defaults to **2 models × 3
workloads × 7 policies, `MAX_STEPS=10`, `PROFILE_SYNC=false`, `PROFILE_MEMORY_SNAPSHOT=false`** — override
to scope the A/B and enable the snapshot:
```
# layer_act A/B — the comma list runs OFF then ON in one sweep
ASYMM_EXP_ACT_POLICIES='none|true|true|true|false|false,none|true|true|true|false|true' \
MODEL_SPECS='meta-llama/Llama-4-Scout-17B-16E|1' WORKLOADS='4096|4|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0' \
PROFILE_MEMORY_SNAPSHOT=true PROFILE_SYNC=true MAX_STEPS=3 WARMUP_STEPS=1 OVERWRITE=true GPU_POOL=3 \
NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 ENABLE_LIGER_KERNEL=false \
bash scripts/lf/profile_lora_lf_test.sh

# layer_gc A/B
ASYMM_EXP_ACT_POLICIES='none|true|true|false|true|false,none|true|true|false|true|true' ... bash scripts/lf/profile_lora_lf_test.sh
```

Compare OFF vs ON directly with the metrics tools (both rows print side by side — memory **and** timing):
```
scripts/lf/show_status.sh  <profiling_dir>     # both leaves OK (not OOM/FAILED/RUNNING)
scripts/lf/show_metrics.sh <profiling_dir>     # forward/backward/optimizer/step (s) + step (GiB), OFF vs ON
.venv/bin/python scripts/testing/peak_attrib.py <on_leaf>/memory_snapshot.pickle   # sdpa:92 -> ~0
grep -m1 loss <off_leaf>/train.log ; grep -m1 loss <on_leaf>/train.log            # correctness
```
(OFF and ON may share a `Config` label in `show_metrics` until `sdparecomp` is added to its label map —
distinguish them by the `sdparecomp0`/`sdparecomp1` tag in the leaf path.)

**Validation Gate 4 ✅ (hard acceptance), per cell.** A run is accepted only if **all three** of memory,
efficiency, and correctness pass — *memory-only is not acceptance*.

*MEMORY*
1. `peak_attrib.py` `sdpa:92` frame drops **~6.7 GiB → ~0**.
2. `show_metrics` **step (GiB)** drops ~6–7 GiB (26.3 → **~20 GiB**); stays **< 27.4** (the recomp
   baseline = the 28,094 MiB bar).

*EFFICIENCY (`show_metrics` seconds columns, real workload)*
3. **forward (s)**: ON within ~±5% of OFF (recompute lands in backward, not forward).
4. **backward (s)**: ON > OFF by a **bounded** delta (per-layer SDPA recompute + q/k/v restage), and the
   increase is **localized to attention** — `timing_by_module.csv` `stage=step.backward` attention kernel
   time rises while non-attention rows stay ≈ flat. (OFF backward ≈ **59.3 s**. `timing_by_module.csv` is
   the **nsys** output — populated by `PROFILERS=both` (test.sh default); it is empty in source-only mode.)
5. **step (s)**: record absolute + % increase and state the tradeoff explicitly —
   **"freed ≈X GiB for +Y s (+Z%) step time"**. **Reject as a blow-up** if step time rises
   disproportionately (e.g. >~25%) or the delta is not attributable to attention — that is the
   checkpoint×offload re-staging pathology (the old whole-attention +8 GiB regression).

*CORRECTNESS & SANITY*
6. Step loss within fp-noise of OFF `L0` (deterministic recompute ⇒ ≈ exact).
7. Model loads + steps; vision path intact (multimodal).

Report the per-cell A/B table — `{peak, sdpa:92, fwd_ms, bwd_ms, step_ms, e2e_ms, loss}` for OFF vs ON
with deltas. If any check fails, stop and diagnose against the matching stage gate before iterating.

---

## Rollback
`ASYMM_ATTN_SDPA_RECOMPUTE` unset/false (default) → `install_sdpa_recompute` returns False, registers
nothing, leaves `_attn_implementation == "sdpa"`. Behavior is byte-identical to the attn_act-only path.
To fully remove: delete `sdpa_recompute.py`, the import + call in `lf.py`, and the mirrored harness lines.

---

## File-change index (quick reference)
| Stage | File | Change |
|---|---|---|
| 0 | `scripts/testing/peak_attrib.py` | new — snapshot→peak attribution tool |
| 1 | `asym_gemm/training/sdpa_recompute.py` | new — interface + `install_sdpa_recompute` |
| 1 | `scripts/testing/test_sdpa_recompute.py` | new — isolated determinism/grad test |
| 2 | `asym_gemm/integrations/lf.py` | import (~L20) + `install_sdpa_recompute(model)` after L1520 |
| 2 | `scripts/testing/check_sdpa_install.py` | new — flag on/off install assertions |
| 3 | `scripts/lf/profile_lora_lf.sh` | add 6th policy field: mirror every `layergc`/`ASYMM_LAYER_GC` site (parser L423-447, tag L399, defaults L598, values L2090, labels L1313/L2401, forward L2559/L2601) |
| 3 | `scripts/lf/run_lf_lora_sft.sh` | mirror `ASYMM_LAYER_GC` at L68, L392-395, and the `ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_GC` forward |
| 0/4 | `scripts/lf/profile_lora_lf_test.sh` | reuse as-is — `PROFILERS=both` real-workload harness; override its broad defaults (2 models × 3 workloads × 7 policies, `MAX_STEPS=10`, snapshot/sync off) to scope the A/B |
| 0/4 | `scripts/lf/show_metrics.sh` + `show_status.sh` | reuse as-is — metric/status tables (timing **+** step GiB per config); primary A/B comparison tool |
| 4 | — | measurement only — `show_metrics`/`show_status` (fwd/bwd/opt/step s + step GiB) + `peak_attrib.py` (sdpa:92) + loss |
