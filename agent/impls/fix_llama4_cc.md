# Fix: Llama-4-Scout activation-offload (norecomp) must beat recompute on HBM

## The hard goal

For `meta-llama/Llama-4-Scout-17B-16E`, workload `4096,4,1`, make the **offload (norecomp)** configs

- `asym_cpuadamwds,norecomp | none,true,true,false,true`  (exp_act + attn_act + layer_gc)
- `asym_cpuadamwds,norecomp | none,true,true,true,false`   (exp_act + attn_act + layer_act)

peak **less HBM** than both recompute baselines:

- `asym_cpuadamwds,recomp | none,false,false,false,false`  → **28,094 MiB** (binding target)
- `zero3_offload,recomp   | none,false,false,false,false`  → 50,716 MiB (already beaten)

Current norecomp = **47,626 MiB**. Must cut ~20 GiB. The thesis: at equal config, offloading activations to CPU must reclaim more HBM than recomputing them — same forward math, just park on CPU instead of throwing away.

---

## Corrected root cause — it is a FORWARD-side leak, not the backward

> Supersedes the earlier draft that blamed the expert *backward* and prescribed token-chunking (a misread of a peak residual-attribution). Measured phase timeline (`memory_breakdown.jsonl`, step 2):

| run | **after_forward resident** | peak | backward transient added |
|---|---:|---:|---:|
| norecomp (layergc) | **34,597 MiB** | 47,626 | +13,029 |
| norecomp (layeract) | 34,605 MiB | 47,634 | +13,029 |
| recomp | **15,231 MiB** | 28,094 | +12,863 |

1. **Backward transient is identical (~13.0 GiB) in all three.** The backward is not the problem; **no grouped-GEMM/token-chunk rewrite is needed.**
2. **The entire ~19.4 GiB gap is after-forward resident.** Forward offload leaves 34.6 GiB on the GPU; recompute discards to 15.2. `peak ≈ after_forward + 13`, so winning after_forward wins the peak.

So the bug is the one you named: we "offload instead of throw away," but the forward keeps the GPU copy resident anyway (offload-AND-keep) and/or never offloads the residual/norm stream. The offload machinery *is* firing hard (293 GB copied to CPU cumulatively, 9.4 GiB live), so this is a **coverage + release** problem, not a failure to copy.

### What the 19.4 GiB is
- **~8 GiB confirmed**: per-layer residual/norm hidden, `[16384×5120] bf16 = 166 MiB × ~48 layers` (`norms live_activation`). Recompute discards these; offload keeps them resident. Offloading them also pushes *below* recompute (recompute is forced to keep its layer-boundary residuals on GPU; we can offload ours).
- **~11 GiB to enumerate (Stage 0)**: other forward activations not released (candidates: attention SDPA/o-proj path, a coverage gap, or GPU sources not freed after copy). Expert intermediates are NOT it — the expert Function `del`s gate/up/act (`llama4_experts.py:336`).

### Ruled out by data
- **Shared MLP** — resident only under *recomp*; norecomp offloads it.
- **Expert backward** — transient identical to recompute.
- **layer_act/layer_gc** — `layeract1` wrapped 48 layers, no fallback, yet after_forward unchanged; `layergc1` ran with `layer_gc_wrapped=0` (plumbing bug). The layer saved-tensor offload is a measured no-op (Stage 1 resolves why).

---

## Qwen3 is an existence proof, not a gold standard
Qwen3 satisfies the thesis (58.0 < 63.8) — its forward offload releases, so its after_forward is low — but wins by only ~6 GiB, both numbers dominated by a common ~41 GiB loss/logits term; its model-activation footprint is tiny. Don't chase its thin margin or copy its code (expert `intermediate=768` ≈ 11× smaller than Scout's 8192). It only proves a correctly-releasing forward offload exists; Llama-4's isn't releasing.

---

## Staged implementation plan

**Hard invariant (never violate):** **Move tensors, never reshape math.** Stages relocate activations only; they do NOT touch the grouped-GEMM/expert kernels. No per-expert Python loops, no GEMM splitting, no extra small kernels, no degraded kernel-launch patterns.

**Release is the goal; latency is secondary.** The essential effect is that an activation copied to CPU is **freed from HBM**. With latency de-prioritized, the simplest correct form is fine: a **blocking D2H copy frees the source immediately** (`cpu.copy_(src); del src`). A dedicated copy stream + `src.record_stream(copy_stream)` (torchtune pattern, pinned/double-buffered) is an *optional* upgrade to overlap the transfer **only if** a stage shows a real slowdown — not required up front.

**Accept/reject — measured e2e only (toy profiling not accepted):** ACCEPT a stage iff `after_forward` drops **meaningfully** (≥ ~2 GiB, not trivial) AND fwd/bwd latency **does not pathologically blow up** (a modest rise is acceptable). REJECT only if memory is flat/trivial or latency explodes; then revert and move on.

### Stage 0 — Diagnostic: name the 34.6 GiB (GATE, no code change)
**Scope:** profiling only. Turn "~11 GiB other" into a per-module punch-list.
```
MODEL_SPECS="meta-llama/Llama-4-Scout-17B-16E|1" \
ASYMM_EXP_ACT_POLICIES="none|true|true|true|false" MAX_STEPS=2 WARMUP_STEPS=1 \
ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN=1 \
ASYM_GEMM_LF_PROFILE_LIVE_ACTIVATION_DETAILS=1 ASYM_GEMM_LF_PROFILE_LIVE_ACTIVATION_TOPK=128 \
bash scripts/lf/profile_lora_lf.sh        # seq=4096 batch=4 cell (per jobs.tsv)
```
**Read:** `memory_breakdown_summary.json → live_activation_detail_rows` + the `after_forward` rows; group by `module_name`, collapse layer indices. **Gate:** proceed only once 34.6 GiB is attributed to named modules (residual/norm vs attention vs other). **Watch:** if rows are empty, confirm `ASYM_GEMM_LF_PROFILE_LIVE_ACTIVATION_DETAILS` is plumbed (`lf_trace.py:119`).

### Stage 1 — Micro-test: does our offload free the GPU source? (isolated, <1 min; picks Stage 2's shape)
**Why:** layeract copied 293 GB to CPU yet freed ~0 HBM. Prove in isolation whether the inline compute-stream offload releases the source.
```python
import torch
from asym_gemm.training.activation_offload import ActivationOffloadManager
x = torch.randn(16384, 5120, device="cuda", dtype=torch.bfloat16, requires_grad=True)
base = torch.cuda.memory_allocated()
m = ActivationOffloadManager(pin_memory=True); h = m.offload(x, "probe"); del x
torch.cuda.synchronize()
print("still held MiB:", (torch.cuda.memory_allocated()-base)/2**20)  # ~160 => NOT freed
```
**Decision:** still-held ≈ 160 MiB → the source isn't released → fix = side-stream + `record_stream` (Stage 2 Change A). ~0 → the 19 GiB is pure *coverage* (tensors never handed to `offload`) → Stage 2 is coverage-only. (Isolated kernel/alloc test — acceptable without e2e.)

### Stage 2 — Release the offloaded GPU source (fixes the OFFLOAD cell `none,T,T,T,F` + the common exp/attn of both cells)
`none,T,T,T,F` (layer_act = **offload** the glue) and `none,T,T,F,T` (layer_gc = **recompute** the glue) are *different, mutually-exclusive* mechanisms (`profile_lora_lf.sh:439`) → different fixes. Stage 2 fixes the offload path; Stage 3 fixes the recompute path. Both cells share exp_act+attn_act via `ActivationOffloadManager`, so Stage 2's release helps both; the layer_act glue funnels through `DecoderSavedTensorOffloadWrapper._pack`, so Stage 2 *fully* fixes the offload cell.
**Scope (exact) — make the D2H copy free its source at all three offload copy sites (same one-pattern change):**
- `asym_gemm/training/activation_offload.py` → `ActivationOffloadManager.offload`/`.stage` — exp_act+attn_act (both cells).
- `asym_gemm/training/decoder_activation_offload.py` → `DecoderSavedTensorOffloadWrapper._pack`/`_unpack` — the layer_act glue.
- `asym_gemm/training/attention_activation_offload.py` → its `_pack`.

**Change A — overlap + release (infra, in `offload`):**
```python
def _copy_stream(self, device):
    s = self._streams.get(device.index)
    if s is None: s = torch.cuda.Stream(device=device); self._streams[device.index] = s
    return s

def offload(self, tensor, tag):
    if tensor.device.type == "cpu": return self.adopt_cpu(tensor, tag, original_device=tensor.device)
    handle = self.empty_cpu(tuple(tensor.shape), tensor.dtype, tensor.device, tag)  # pinned dst
    src = tensor.detach(); cs = self._copy_stream(src.device)
    cs.wait_stream(torch.cuda.current_stream(src.device))   # copy waits for producer
    with torch.cuda.stream(cs):
        handle.tensor.copy_(src, non_blocking=True)         # D2H overlaps later compute
        src.record_stream(cs)                               # allocator frees src only after copy
    ev = torch.cuda.Event(); ev.record(cs); self._pending_cpu_ready_events[id_(handle)] = ev
    return handle  # stage(): prefetch H2D on cs, event-sync before use; public API unchanged
```
Pure how-not-what: same tensors offloaded, copy now overlapped and source provably released. Cannot change numerics.

**Change B — only if Stage 0/1 shows the residual is a *live* tensor the `_pack` hooks never see** (so Change A′ can't catch it). Offload it explicitly via a shared helper called from **both** the layer_act path and `_manual_forward` (layer_gc), so both configs benefit:
```python
residual = hidden_states
normed   = self._checkpoint_norm(layer.input_layernorm, hidden_states)
mixer    = _call_self_attention(layer, normed, values)
hidden_states = _add_offloaded_residual(residual, mixer, self.saved_tensor_offload.manager)
# ... identical for the post_attention residual + feed_forward ...
```
```python
class _AddOffloadedResidual(torch.autograd.Function):   # one elementwise add + one async D2H; no GEMM, no loop
    @staticmethod
    def forward(ctx, residual, other, manager):
        ctx.h = manager.offload(residual, "residual"); ctx.manager = manager  # side-stream copy, residual GPU released after add
        return residual + other                                               # add consumes residual; then freed
    @staticmethod
    def backward(ctx, g):
        res = ctx.manager.stage(ctx.h);  ctx.manager.release_cpu(ctx.h)        # H2D prefetch only if a grad needs it
        return g, g, None
```
**Efficiency:** ≤ 2 extra `[tok,hidden]` D2H/layer, fully overlapped on the copy stream; ~16 GiB/step D2H, trivial vs the existing 293 GB; one copy per residual (no launch-count growth).

**Validation (e2e via `scripts/lf/profile_lora_lf.sh` — the OFFLOAD cell):**
```
MODEL_SPECS="meta-llama/Llama-4-Scout-17B-16E|1" BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYMM_EXP_ACT_POLICIES="none|true|true|true|false" MAX_STEPS=5 WARMUP_STEPS=1 \
ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN=1 bash scripts/lf/profile_lora_lf.sh   # seq=4096 batch=4 cell
```
vs the current run, compare `after_forward` (`memory_breakdown.jsonl`), `actual_peak_allocated_hbm_bytes` (`memory_breakdown_summary.json`), and `fwd_ms`/`bwd_ms` (timing table). **ACCEPT** iff after_forward drops meaningfully (target peak `< 28,094` MiB) and latency not pathologically worse.
**Risks / watch:** (1) `record_stream` grad parity (only if using the side-stream variant) — verify vs main on a 2-layer numeric check + `tests/training`; the blocking-copy form has no stream risk; (2) `stage()` in backward must not serialize the critical path — prefetch one layer ahead if it does; (3) if Stage 0 shows the residual is a *live* tensor the `_pack` hooks never see, the release alone won't free it → use Change B (below).

### Stage 3 — Engage `layer_gc` so the RECOMPUTE cell `none,T,T,F,T` works
**Scope:** the plumbing that should install `DecoderLayerGlueGCWrapper` on the 48 decoder layers but reports `layer_glue_gc_wrapped=0`. Trace `ASYMM_LAYER_GC` → `_layer_glue_gc_enabled()` and the install gate in `asym_gemm/integrations/lf.py` + the env wiring in `scripts/lf/run_lf_lora_sft.sh` / `run_lf_profiled_train.py`; make it actually wrap. With Stage 2's release already applied to the shared manager/`_pack`, the glue is then recomputed (norms) + offloaded-and-freed (the rest).
**Validation (e2e — the RECOMPUTE cell):**
```
MODEL_SPECS="meta-llama/Llama-4-Scout-17B-16E|1" BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYMM_EXP_ACT_POLICIES="none|true|true|false|true" MAX_STEPS=5 WARMUP_STEPS=1 \
ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN=1 bash scripts/lf/profile_lora_lf.sh
```
**ACCEPT** iff (a) `train.log` now shows `layer_glue_gc_wrapped=48`, (b) `after_forward`/peak drop (target peak `< 28,094`), (c) latency rises only modestly (norm recompute is cheap), not pathologically. **Watch:** layer_gc requires policy `none` and is mutually exclusive with layer_act (`profile_lora_lf.sh:439`).

### Stage 4 — Close any remainder (only if a cell is still > 28 after Stage 2/3)
**Scope:** the largest still-resident owner named by Stage 0 (e.g. an attention SDPA/o-proj activation, or the residual via Change B). Apply the same Stage-2 offload-and-release to it — no new mechanism. **Validation (both cells together):**
```
MODEL_SPECS="meta-llama/Llama-4-Scout-17B-16E|1" BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYMM_EXP_ACT_POLICIES="none|true|true|true|false,none|true|true|false|true" MAX_STEPS=5 WARMUP_STEPS=1 \
ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN=1 bash scripts/lf/profile_lora_lf.sh
```
**ACCEPT** iff **both** cells peak `< 28,094` MiB with latency not pathologically worse. **Watch:** if SDPA is flash/fused its saved activations are already minimal — confirm it's a real owner via Stage 0 before touching it.

## Exit criterion
Goal met when both norecomp cells report `actual_peak_allocated_hbm_bytes < 29,459,163,648` (asym recomp) **at `ligerloss0`** — no Liger, matching the recompute baselines, so the comparison is fair — with fwd/bwd latency not pathologically worse, via the `profile_lora_lf.sh` e2e runs above (not toy profiling). Track `after_forward` as the leading indicator at every stage.
