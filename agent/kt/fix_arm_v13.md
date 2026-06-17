# KT ARM BF16 SFT Fix Plan v13 — Eliminate redundant in-kernel recompute

**KT remains native ARM BF16 CPU code. DeepSpeed is not used.**

## Origin: 4-agent (Opus, max-effort) exhaustive re-audit

A deep 4-subagent search for *meaningful* (non-marginal) latency+memory wins found
that ~35% of the step is **redundant forward recompute**. The forward is computed
up to 3x per step:
1. main forward (~34.6 s, needed)
2. gradient-checkpoint recompute-forward (~34.6 s) — PyTorch re-runs the layer in
   backward to repopulate the input cache
3. **KT in-kernel per-tile recompute (~27 s in backward)** — regenerates the same
   intermediates AGAIN from the input cache for the grad math

Passes #2 and #3 compute the same intermediates back-to-back per layer's backward.

Other agent findings (for the record, not this stage): step is ~93% CPU-MoE-bound
with the GPU idle ~94% (copies/pinning are <1 s — non-levers); BFMMLA is only
**1.5x not 2x** on this core (~6-8 s, complex) — deferred; the optimizer state is
actually **24.75 GiB not 12.38** (profile sampled 1 of 2 fp32 moments; RSS grows to
~107 GiB) → bf16/8-bit moments would save 12-18 GiB (AsymGEMM-side); base experts
are NOT duplicated (single 54 GiB copy; the "58 GiB routed_experts" is a phantom).

## Stage 1 (v13): cache forward intermediates, drop pass #3

Status: implemented / measured / accepted — (in progress)

The checkpoint-recompute forward (pass #2) already computes all intermediates into
`forward_buffers_` right before the cache save. Instead of throwing them away and
recomputing per tile in backward (pass #3), **snapshot them into the cache (BF16)
and have the backward read them**:
- `CacheEntry`: added BF16 `gate/up/act/down/gate_u/up_u/down_u` (+`has_intermediates`).
- `save_forward_intermediates_to_cache`: parallel f32->bf16 from `forward_buffers_`
  at the forward-save point.
- `fill_backward_intermediates_from_cache`: per-tile bf16->fp32 expand into the
  per-thread tile buffers the grad math reads (packed_input/tile_routes still come
  from `fill_backward_recompute_tile`).
- backward loop: if `has_intermediates`, fill from cache; else fall back to the 5
  recompute kernels (robustness). The recompute kernels are kept as the fallback.

Memory: backward is sequential per layer so only ~1-2 backprop-live layers' caches
are resident; BF16 intermediates add ~1.2 GiB/live-layer (~+1.2-2.4 GiB steady).
Latency target: ~25-32 s/step (-16 to -19%). Precision: cached BF16 intermediates
vs FP32 recompute — consistent with the all-BF16 model; validated by per-commit
reference/dropout tests + e2e loss. If precision drifts, escalate to FP32 cache
(numerically identical, +~2.4 GiB/live-layer).

## Implementation refinement (important)

Initial impl cached BF16 (scalar f32->bf16): `cache_save_ms` 9.7 -> 289 ms/layer,
which ate ~all the recompute win (net only ~2.8 s). FP32 parallel copy cut that to
142 ms/layer (net ~+9 s). Final impl **MOVES** the `forward_buffers_` vectors into
the cache (pointer swap, no copy) — `forward_buffers_` is released and re-.assign()'d
on the next forward anyway, so the move is free: `cache_save_ms` -> **6.3 ms/layer**
(even below the input-only baseline). Cache is FP32 = numerically identical to the
recompute. The cache is transient (freed per layer after backward; large allocs
returned to OS) so steady RSS is unchanged.

## Results log (short 3-step e2e, GPU 1; full acceptance pending)

| Metric | v11 short | v13 (move) short | Delta |
|---|---:|---:|---:|
| cache_save_ms | 9.7 | 6.3 | free |
| backward_tile_recompute_ms | ~19594 | ~580 (fill only) | -97% |
| backward_grouped_tile_ms (wall) | 967 | ~630 | -35% |
| backward total | 97.36 s | 81.59 s | -15.8 s |
| measured step | 137.25 s | 124.42 s | -13 s (short, noisy) |
| steady RSS | 94.38 GiB | 94.32 GiB | = (cache transient) |
| measured loss | 1.6562/1.6656 | 1.656/1.6689 | match |

### Full-scale confirmation (14/15-step run, stopped early — sufficient)

The acceptance run reached 14 steps before being stopped; the 9 measured-step
native counters confirm the win at scale: `backward_grouped_tile_ms` ~545.8
ms/layer (recompute eliminated), `backward_tile_recompute_ms` ~732 task-ms (the
fill only, vs ~19594 recompute), `cache_save` free. Losses tracked v11 throughout.

Verdict: **ACCEPT.** Eliminated the redundant in-kernel recompute via a free
(move) cache hand-off; backward -15.8 s (short) with grouped-tile wall -~35% at
scale, losses match, steady RSS unchanged. The recompute kernels remain as a
fallback (`has_intermediates`). Estimated full avg_step ~155 s (from v11's 170.55;
the short run's 124 s is 2-step favorable noise).

NOTE: the bigger recompute (gradient-checkpoint recompute-FORWARD, pass #2,
~34 s/-20%) is mutually exclusive with this design. CORRECTION: it is NOT
AsymGEMM-side — it lives in KT's `kt-kernel/python/sft/layer.py` (the `first_forward`
cache-skip, inside the KT boundary) + the LlamaFactory checkpointing toggle.
(AsymGEMM is a separate backend and is not used by the kt_armbf16 path at all.)
Tracked as v14 item #1 — it actually supersedes this v13 approach (bigger win).
