# Fix CPU memory: reclaim ~90 GiB of wasted host RAM (lossless)

## Why
The qwen3-30b-a3b LoRA-SFT smoke OOM-kills in the first forward. Real CPU ceiling is **~958 GiB** (2 Grace NUMA nodes 0+1, `numactl --membind=0,1`; the "1.65 TiB" `free` reports includes GPU HBM). The workload needs ~0.8 TB, so it dies. Two host-RAM wastes are **lossless** to remove (no numerics change, no recompute, no GEMM change):
- **A. ~60 GiB** — every frozen base weight is held twice (original HF tensor + pinned `HostWeight` copy); the original is never freed.
- **B. ~32 GiB** — the expert activation-offload warm pinned pool (`_CPU_BUFFER_POOL`) is capped at 32 GiB and only trims on insert.

Target: drop preflight resident from **~213 GiB → ~150 GiB** and forward peak by up to **32 GiB**, with **no** forward/backward latency regression.

## How to validate (mostly just a CPU-RAM check — NOT the heavy e2e timing profile)
These reclaims do not touch the compute path, so we do **not** need the full e2e timing profiling that kernel/compute changes require. Split by what each stage changes:
- **Stage 1 = load-time memory only** → verify **CPU RSS** at model-load. Forward/backward kernels are byte-identical, so latency is unchanged *by construction* — no timing run, no completed training.
- **Stage 2 = per-step alloc pattern** → also needs a short forward/backward timing check (it changes how often `cudaHostAlloc` fires).

Keep a change only if it **meaningfully** cuts memory (not trivially) **without** blowing up latency where latency can actually move.

### Measuring CPU RSS (Stage 1, the main signal)
Baseline already exists: the OOM run recorded `rss_peak_bytes = 213,376,696,320` (~213 GiB) in its `process_memory.csv` and `model_loaded` heartbeat. Post-change number — either:
- **Re-use the existing launch** (it may still OOM later in forward; preflight RSS is captured *before* that): run it, read `process_memory.csv:rss_bytes` and the `model_loaded` line in `lf_run/heartbeat.jsonl`. Expect **~150 GiB**.
- **Or a standalone load-and-measure** (no training): build the AsymGEMM-wrapped model exactly as the loader does (CPU-first, `ASYM_OFFLOAD_MODULES=all`), `gc.collect()`, then read `VmRSS`/`VmHWM` from `/proc/self/status`. Expect ~150 GiB.

Always on an **idle** node — `numactl -H` (node 0/1 free) first; co-tenants corrupt the comparison.

---

## Stage 1 — Free the duplicated frozen base weights (~60 GiB)

### Scope (exact)
- `asym_gemm/training/qwen3_moe.py` → `class AsymQwen3Experts.__init__` (lines ~1919–1989) — **primary** (the ~58 GiB of packed experts).
- `asym_gemm/integrations/lf.py` → `apply_lf_asym_lora` (def ~993; expert install loop ~1155–1167) — add one `gc.collect()` after the loop.
- `asym_gemm/integrations/lf.py` → `_wrap_lf_linear_leaf` (~563) — **optional 1b** (~3 GiB non-expert Linears; trivial alone, keep only because it's the same mechanism at ~zero cost).

### Root cause (verified in code)
`AsymQwen3Experts.__init__`: `gate_up = source.gate_up_proj.detach()` (aliases source storage) → `AsymGroupedFrozenLinear(gate_up.to(bf16), clone=False, pin_memory=True)`. With `clone=False`, `HostWeight` allocates a **new** buffer only via `pin_memory()` (`host_weight.py:222`, which always copies — [PyTorch #21076]). The result is an independent pinned copy, but **`source.gate_up_proj` / `source.down_proj` are never released**, and `apply_lf_asym_lora` keeps the old module in `expert_replacements` and never calls `gc`. → two CPU copies of all 30B frozen weights.

### Code change — 1a (primary), in `AsymQwen3Experts.__init__`, immediately AFTER the pinned-assert (~line 1981)
Emptying the source `Parameter` drops its reference to the big storage; the local `gate_up`/`down` (still used at `_resolve_device(gate_up)` ~line 1987) free at function return → per-layer transient only, never whole-model 2×.

```python
# existing (1963–1981): self.gate_up_base / self.down_base built with clone=False + pinned; assert pinned
# NEW — release the duplicated source base weights (the pinned HostWeight copies are independent):
gate_up_pinned = self.gate_up_base.host_weight.weight.is_pinned()
down_pinned    = self.down_base.host_weight.weight.is_pinned()
if self.offload and backend == "asym" and gate_up_pinned and down_pinned:   # SAFETY GATE
    for _attr in ("gate_up_proj", "down_proj"):
        _src = getattr(source, _attr, None)
        if isinstance(_src, torch.nn.Parameter):
            _src.data = torch.empty(0, dtype=_src.dtype, device=_src.device)  # free storage, keep a valid Parameter
        elif _src is not None:
            try:
                setattr(source, _attr, None)
            except Exception:
                pass
# Do NOT free in the torch/non-offload branch (TorchGroupedFrozenLinear may alias the source).
```

### Code change — 1a insurance, end of `apply_lf_asym_lora` (before `return model, report`, ~1405)
```python
import gc
gc.collect()   # drop the orphaned old modules retained in expert_replacements / any reference cycles
```

### Code change — 1b (optional, ~3 GiB), in `_wrap_lf_linear_leaf` after `adopt_host_weight(...)` builds the pinned `host_weight`
```python
if getattr(host_weight, "is_pinned", False) and isinstance(module.weight, torch.nn.Parameter):
    module.weight.data = torch.empty(0, dtype=module.weight.dtype, device=module.weight.device)
```

### Efficiency / latency / kernel reasoning
Load-time weight lifecycle only. No change to the grouped GEMM, expert dispatch, dtype, or any per-step op → forward/backward kernels and launch pattern are **identical by construction**. No per-expert loops, no GEMM splitting. Forward/backward Δ = 0 % → **no timing run needed**.

### Ambiguities resolved
- `pin_memory()` always copies, so the source is safe to free once pinned — [PyTorch #21076], [#32167].
- `AsymQwen3Experts.__init__` does **not** store `source` on `self` (read 1919–1989) → after construction the source is reachable only via the old module; emptying its two packed params releases the storage.
- Non-strict / pin-failed path leaves `HostWeight` aliasing the source → the `is_pinned()` gate prevents an unsafe free.

### Risks / watch items
- **Source read after free** (a base-`state_dict` save / correctness probe). LoRA-SFT saves only adapter params, so it shouldn't happen — watch any path reading `*.experts.gate_up_proj`.
- **Other families** (`AsymQwen35Experts`, Llama4, packed `wrap_qwen3_experts`) have their own classes — Stage 1a fixes Qwen3 only (our workload). Replicate before using those models.
- **The 213→150 drop IS the hypothesis test.** If RSS falls < 40 GiB, the duplicate wasn't the cause → REJECT and re-examine the 213 GiB composition (CUDA ctx / dataloader / DeepSpeed).

### Validation (CPU-side only)
1. **Correctness (isolated, allowed):** build one `AsymQwen3Experts` from a small fake source, forward a fixed input, run the source-free, forward again → `assert torch.equal(out_before, out_after)`.
2. **Memory (the acceptance signal):** get post-change RSS via "Measuring CPU RSS" above. **ACCEPT** if RSS at `model_loaded` drops ≥ 40 GiB (~213 → ~150 GiB); **REJECT** if < 10 GiB.
3. **No timing run** (compute path unchanged). Optional belt-and-suspenders: step-0 loss matches a pre-change run (bit-for-bit; lossless).

**✅ Result (executed — `agent/impls/_val_stage1.py`, real 128-expert/H2048/I768 dims, venv torch 2.12+cu130):**
`pinned=True,True` · `faithful copy=True,True` (forward weights bit-exact) · `source numel=0,0` (freed) · 8 layers: RSS grew **10.97 GiB vs one-copy 9.00 GiB (1.22×, not 2×=18 GiB)**. The duplicate is gone on the real expert dims; ×48 layers ⇒ ~58 GiB reclaimed.
Regression: `pytest tests/training/test_lf_qwen3_asym_backend.py` → **108 passed** (incl. SM100 forward+backward+checkpoint+anomaly vs reference) — no break, "source-read-after-free" risk did not trigger. **PASS.**
Still to confirm via one e2e (node idle: GPU0 free, ~794 GiB on Grace nodes 0/1): full-model preflight RSS (~213 → ~150 GiB) and the Stage 2 latency sweep.

---

## Stage 2 — Cap the warm pinned activation pool (~32 GiB, forward-time)

### Scope (exact)
- `asym_gemm/training/activation_offload.py` → `_DEFAULT_CPU_POOL_MAX_BYTES` (line 13), `_cpu_pool_max_bytes()` (reads env `ASYM_EXPACT_CPU_POOL_MAX_BYTES`), `clear_activation_offload_cpu_pool()` (line 67). **No code change** — drive via env.
- Only the **expert** offload path uses this pool (decoder/attention saved-tensor paths alloc per-tensor, unaffected).

### Change
```bash
ASYM_EXPACT_CPU_POOL_MAX_BYTES=8589934592   # 8 GiB (sweep 32→8→0, find the knee)
```

### Efficiency / latency / kernel reasoning
The pool amortizes `cudaHostAlloc` (slow). Shrinking it trades a smaller live pinned cache for more alloc churn **on the expert offload path only** — the one place latency can move, so this stage *does* need a timing check. No GEMM/compute change.

### Risks / watch items
- **Latency knee:** too small → repeated `cudaHostAlloc` in forward → fwd-ms regression. If even `0` shows no regression, the pool was pure waste → set it low permanently.

### Validation (needs a short forward/backward run — the only stage that does)
Short real-model run (not the heavy profile): `scripts/lf/profile_lora_lf.sh` with `MAX_STEPS=5 PROFILE_WARMUP_STEPS=2 CUTOFF_LEN=6144 PER_DEVICE_TRAIN_BATCH_SIZE=8 MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B` and the same offload flags as the OOM run, for `ASYM_EXPACT_CPU_POOL_MAX_BYTES ∈ {32 GiB(base), 8 GiB, 0}`, on an idle node. Record `rss_peak_bytes`, `activation_offload_cpu_pool_stats()['cpu_pool_cached_bytes']`, and `summary.md` `step.forward`/`step.backward` ms.
**ACCEPT** the smallest cap with fwd/bwd regression ≤ 3 % and `rss_peak` drop ≥ 20 GiB; else REJECT (keep 32 GiB).

---

## Stage 3 — In-place pinning (deferred, optional)
Avoid the second copy entirely: instead of `pin_memory()` (always copies), pin the existing storage in place with `torch.cuda.cudart().cudaHostRegister(ptr, nbytes, 0)` (no stable `pin_memory_()` yet — [PyTorch #32167]). Removes the duplicate **and** the transient 2× during load.
- **Risk (don't ship before Stage 1 lands):** source pages must stay alive, aligned, and unmigrated for all of training; on the 64 KB-page Grace kernel under `numactl --membind` this is fragile. Follow-up only if the (already small, ~1 layer) load-time peak ever matters.

## Sources
`pin_memory()` always copies; in-place via `cudaHostRegister`: [PyTorch #21076](https://github.com/pytorch/pytorch/issues/21076), [PyTorch #32167](https://github.com/pytorch/pytorch/issues/32167), [CUDA semantics](https://docs.pytorch.org/docs/2.11/notes/cuda.html).
