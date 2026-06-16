# KT ARM BF16 SFT Fix Plan v12 — Remove dead allocations (memory)

**KT remains native ARM BF16 CPU code. DeepSpeed is not used.**

The deep re-audit also found dead allocations the earlier memory loop (v8-v10)
missed. v12 reclaims them. Acceptance by steady-state working RSS
(`process_memory_after_step.rss_bytes`), not rss_peak.

## Findings (verified: allocated but never dereferenced for compute)

- **`cache_pool_` — ~3.16 GiB.** `ensure_cache_pool` grows it on every cache-save;
  with `share_cache_pool=false` (default) it is per-instance, 67.5 MiB x 48 layers.
  The actual checkpoint cache lives in `cache_stack_` std::vectors; the pool
  pointer is only stored, accounted, and freed, never read. **Safe, biggest win.**
- **`backward_pool_` / `pools.bwd_work` — ~0.38 GiB.** `ensure_backward_buffers`
  grows it; never read (real backward buffers are the per-shard grads + scratch).
- **`merged_output_fp32` — 128 MiB + a per-forward zero-fill.** Allocated in
  `ensure_forward_buffers` but never read (`merge_routes_to_output` writes the
  bf16 output directly).

## Change (v12)

- Dropped the `ensure_cache_pool(...)` call (cache still saved via `cache_stack_`).
- Dropped the `ensure_backward_buffers(...)` allocation call.
- `merged_output_fp32.assign(...)` -> `.clear()` (leave empty).

All three are pure dead-allocation removals — no compute reads them, so zero
correctness/latency risk (a tiny latency *win* from one fewer 128 MiB memset per
forward). Expected steady RSS: ~94.5 -> ~91 GiB (~-3.5 GiB).

Risk: none (dead memory). Validate: reference + dropout tests, then short LF
profile; confirm steady RSS drops ~3.5 GiB with no latency/correctness regression.

## Results log

### Short 3-step e2e profile (bundled with v11)

steady RSS: v10 94.27 -> v11+v12 **94.38 GiB** (no change). HBM unchanged; losses
match.

**Outcome: NOT a memory win (RSS-neutral). Kept only as harmless dead-code/virtual
cleanup.** Key learning: `cache_pool_` and `bwd_work` were `grow_pool`/`malloc`'d
but **never written** (truly dead), so they were virtual-address-space only and
were never resident — Linux demand-paging means untouched pages are not in RSS.
Removing them frees virtual memory, not steady RSS. (Contrast v8/v9/v10, which
removed `.assign()`'d / written buffers that WERE committed, and did cut RSS.)

Implication for the acceptance rule: only removing/​shrinking *written* (committed)
buffers reduces steady RSS; eliminating untouched allocations does not. The dead
pools are removed for hygiene but claim zero memory benefit.
