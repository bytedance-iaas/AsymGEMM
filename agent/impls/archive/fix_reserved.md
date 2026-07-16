# Fix the reserved≫allocated gap (qwen3.5 chunked runs): phase-boundary release

Owner doc for the "peak reserved is weirdly higher than peak allocated" anomaly
observed in the T3 chunked-delta-net scoreboard (`fix_qwen3.5.md` §10b). Written
2026-07-14. Companion artifacts live under
`/workspace/qwen35_local/profiling_fix_qwen35_{chunk,final,e1,e2}` (local disk;
NFS was full when they were produced).

## 1. The anomaly (measured, RULES protocol, chunk=16000 on BOTH backends)

| run @80000×8 | peak allocated | peak reserved | gap |
|---|---:|---:|---:|
| A asym tuned+chunk | 78.9 GiB | 91.7 GiB | **+12.8 GiB (16%)** |
| B superoffload+chunk | 100.4 GiB | 103.0 GiB | +2.6 GiB (2.6%) |

| run @45000×8 | peak allocated | peak reserved | gap |
|---|---:|---:|---:|
| A asym tuned+chunk | 44.7 GiB | 50.7 GiB | +6.0 GiB |
| A asym tuned (no chunk) | 58.5 GiB | 58.9 GiB | +0.4 GiB |
| B superoffload+chunk | 59.4 GiB | 72.4 GiB | **+13.0 GiB** |
| B superoffload (no chunk) | 58.5 GiB | 60.5 GiB | +2.0 GiB |

Two facts fall out immediately:

1. The gap is **introduced by the chunk loop** (unchunked runs sit at +0.4…+2.6),
   and it can hit EITHER backend (B pays +13 at 45k, A pays +12.8 at 80k) — it is
   an allocator-behavior artifact of cycling many similar-but-not-identical
   transient shapes, not an asym-specific leak.
2. `garbage_collection_threshold:0.6` was measured a **byte-identical no-op**
   (A80f/B80f vs A80c/B80c) — that knob only drives the non-expandable block
   allocator's `release_cached_blocks` path and does nothing for
   expandable-segment pools.

## 2. Why reserved inflates: allocator mechanics

With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (sweep default,
`EXPANDABLE_SEG=true`):

- The allocator keeps **per-(device, stream) pools**. A block freed on stream S
  returns to S's pool and is preferentially reused by S; cross-stream reuse
  requires event-sync bookkeeping (`record_stream`) and is conservative.
- An expandable segment is a large **virtual-address reservation** whose
  physical pages are `cuMemMap`ped on demand as the pool grows. Freed blocks
  keep their pages **mapped in the pool** for reuse; pages are only unmapped by
  an explicit release (`torch.cuda.empty_cache()` → release of free expandable
  pages) or allocator teardown.
- `reserved_bytes` counts **mapped pages**, `allocated_bytes` counts live
  blocks. `max_memory_reserved()` is the high-water of mapped pages — this is
  what `memory.md whole_process_peak_reserved_hbm_bytes` reports
  (`scripts/lf/run_lf_profiled_train.py:183,3017,3061`).

Consequence: if phase X (the chunked linattn recompute, cycling per-chunk
in-proj/conv/fla-state transients) grows some pools to their max, and later
phase Y (fg MoE backward: the ~47 GiB full-R expert transient; FA4 attention
backward; optimizer drain) grows *other* pools — or the same pools with
poorly-tiling shapes — then

```text
peak_reserved ≈ Σ_pools max_over_time(pool_mapped)   (a UNION across phases)
peak_allocated = max_over_time(Σ live blocks)        (a true simultaneous max)
```

The union exceeds the simultaneous max exactly when phases with large disjoint
working sets alternate — which is what chunking creates (that is why the gap
appears WITH chunking, and whichever backend's phase shapes tile worse pays it:
B at 45k, A at 80k). Asym's extra side streams (H2D restage, D2H offload, fg
engine) multiply the pool count and worsen the union in its runs. Classic
(non-expandable) segments have the SAME retention behavior plus real
fragmentation, so `EXPANDABLE_SEG=false` is not expected to help (E3 below
measures it anyway rather than arguing).

## 3. What "phase-boundary release" is

Insert `torch.cuda.empty_cache()` at a small number of **phase boundaries**
inside each training step. Under expandable segments this unmaps the physical
pages of all *free* blocks in every pool (virtual reservations stay — no VA
churn), so the next phase re-grows from ~allocated instead of from the previous
phase's high-water. The recorded `max_memory_reserved` then tracks "max over
phases of (live + that phase's transient)" instead of the union — i.e., it
converges toward peak_allocated + one phase's slack.

Properties:

- **Numerics-neutral**: only free blocks are touched; live tensors never move.
- **Async-safe**: blocks with pending stream uses (event refs from
  `record_stream`/cross-stream frees) are NOT free and are left alone;
  `empty_cache` issues no device-wide synchronize itself.
- **Honest metric**: it lowers the *true* mapped-page high-water — the number
  the card must physically back — not a stats-reset trick. Real OOM headroom
  improves by the same amount.
- **Cost**: `cuMemUnmap` + later `cuMemMap`/`cuMemSetAccess` on re-growth,
  order ~ms per GiB per boundary. At 2 boundaries/step cycling ~10–15 GiB on
  300–430 s steps: ≲0.1 s/step (<0.05%). Gated by a measured lat check anyway.
- **Fairness**: applied at harness level, identically for every backend —
  superoffload benefits too (it pays +13 GiB at 45k).

## 4. Diagnosis first (confirm the union theory before shipping the fix)

- **D1 — segment-trace attribution** (cheap): rerun A@80k 2-step with
  `PROFILE_MEMORY_SNAPSHOT=true`, then extend
  `scratchpad/analyze_snapshot.py` to replay `segment_map`/`segment_unmap`
  events (they carry stream + size + frames): reconstruct per-stream
  mapped-bytes timelines, find the reserved-peak moment, print per-stream
  mapped vs allocated at that moment plus the frames that last grew each pool.
  PASS = the reserved−allocated gap decomposes into ≥2 pools whose growth
  frames belong to different phases (chunk loop / fg backward / attention
  backward / offload streams).
- **D2 — empirical A/B** (one 2-step run each): A@80k with P1+P2 release vs
  without. Expected: reserved 91.7 → ≤ ~83, allocated unchanged, loss identical.
- **E3 — expandable-off control** (answers "should I just set
  EXPANDABLE_SEG=false?" with data): A@80k, `EXPANDABLE_SEG=false`, 2-step.
  Expected similar-or-worse reserved + frag-OOM risk near ceilings; if it is
  actually better AND stable, prefer it for scoreboards and stop here.

## 5. Concrete implementation plan

### 5.1 Knob

`ASYM_EMPTY_CACHE_PHASES` — comma list from
`{forward_end, backward_end, step_end, layer_every_<N>}`; unset/empty = off
(default; zero behavior change). Parsed once per process. Harness-level so
every backend gets it (scoreboard fairness). The value must be quoted in any
scoreboard row (command.txt records it, like the other tuned-env knobs).

### 5.2 Hook points (all already exist — no new plumbing)

1. **`forward_end` / `backward_end`** — `asym_gemm/profiling/lf_trace.py`
   already brackets the phases with root-module hooks
   (`register_full_backward_pre_hook` ≈ forward→backward boundary, :876;
   `register_full_backward_hook` ≈ backward end, :877; same pair at :2067).
   Add inside those callbacks:
   `if phase in _EMPTY_CACHE_PHASES: torch.cuda.empty_cache()`.
   The source profiler runs in every scoreboard row for both backends, so
   coverage is symmetric and there is no cost when the env is unset.
2. **`step_end`** — `scripts/lf/run_lf_profiled_train.py:1847`
   (`wrapped_training_step` epilogue) or the `wrapped_optimizer_step` epilogue
   (:1866); one call per step.
3. **`layer_every_<N>`** (only if D1 shows intra-backward union across layer
   phases): piggyback the per-layer backward hooks lf_trace owns (:2067).
   40×/step remap churn — do NOT start here.
4. **Chunk-loop boundary** (only if D1 blames the chunk pool specifically):
   in the chunked delta-net forward
   (`/workspace/qwen35_local/pypatch/sitecustomize.py`, to be upstreamed to
   `LlamaFactory/src/llamafactory/model/model_utils/qwen35_delta_chunk.py`):
   optional `QWEN35_DELTA_CHUNK_EMPTY_CACHE=1` → one `empty_cache()` after the
   per-layer chunk loop (NOT per chunk). Model-level ⇒ also symmetric.

Start with `forward_end,backward_end`; escalate only on D1 evidence.

### 5.3 Code sketch (lf_trace.py, inside the existing root bracket callbacks)

```python
_EMPTY_CACHE_PHASES = frozenset(
    p.strip() for p in os.environ.get("ASYM_EMPTY_CACHE_PHASES", "").split(",") if p.strip()
)

def _maybe_release(phase: str) -> None:
    if phase in _EMPTY_CACHE_PHASES and torch.cuda.is_initialized():
        torch.cuda.empty_cache()

# root backward_pre hook  -> _maybe_release("forward_end")
# root backward_post hook -> _maybe_release("backward_end")
```

### 5.4 Validation gate (RULES protocol)

1. A@80k and B@80k with `ASYM_EMPTY_CACHE_PHASES=forward_end,backward_end`,
   chunk=16000, 4 measured steps. PASS iff loss bands unchanged; lat within
   +2% of the no-release rows (A 428–433 s, B ~360 s); **A reserved ≤ ~83 GiB**
   (≈ alloc 78.9 + one phase's slack); B reserved ≤ 103.0.
2. Same check at 45k (expect A ~46–48, B ~61–63 — the fair fix helps the
   baseline MORE at 45k; fine: the scoreboard then compares true footprints).
3. Non-regression: env unset ⇒ code path inert (review-level check; the diff
   must keep the release strictly behind the env gate).
4. Every quoted row reports BOTH reserved and allocated; any row with
   gap > 4 GiB gets flagged "pool-union suspect".

### 5.5 Rollback / risks

- Kill switch: unset the env. No persistent state.
- Remap page-faults at the next phase's first kernels → small lat blip;
  covered by the +2% gate.
- empty_cache × in-flight side-stream frees: event-pending blocks are skipped
  by design; loss/lat gates watch for surprises.
- Not applicable here: CUDA graphs (unused), NCCL pools (single GPU),
  cudaMallocAsync backend (not enabled).

## 6. Alternatives considered (and why not first)

| option | verdict |
|---|---|
| `garbage_collection_threshold` | measured byte-identical no-op under expandable segments (A80f/B80f) |
| `EXPANDABLE_SEG=false` | same pool retention + real fragmentation + near-ceiling frag-OOM risk; kept only as the E3 control measurement |
| fewer asym side streams | invasive redesign of restage/offload paths; only if D1 blames stream sprawl specifically |
| preallocated chunk-buffer ring in the chunked forward | real code; shape-general reuse is awkward (last chunk differs); only if D1 blames the chunk pool alone |
| `torch.cuda.MemPool` shared pools across phases | experimental API, high blast radius |
| score on allocated instead | RULES.md defines G = peak reserved; reserved is what OOMs the card |

## 7. Results (fill in)

| date | run | phases | alloc (GiB) | reserved (GiB) | verdict |
|---|---|---|---:|---:|---|
| 07-14 | A@80k chunk16k baseline | off | 78.9 | 91.7 | the anomaly |
| 07-14 | B@80k chunk16k baseline | off | 100.4 | 103.0 | small gap |
| 07-14 | **E3 A@80k `EXPANDABLE_SEG=false`** | — | 78.9 | **172.3** | **catastrophic fragmentation — expandable segments SAVE ~80 GiB; keep true, question answered** |
| 07-14 | D2 A@80k | fwd_end,bwd_end | 78.9 | 93.3* | −0.6 GiB only: the union forms INSIDE backward (per-layer sub-phases), boundary release too coarse (*2-step probe; baseline probe 93.9 ≡ 4-step 91.7 within warmup variance) |
| 07-14 | D3 A@80k | `ASYM_EMPTY_CACHE_EVERY_GC_LAYERS=4` | 79.0 | 93.3 | flat again, and steps ~1–2% slower (444/438 vs ~433 s) |
| — | E1/E2 companions (fix_qwen3.5) | dgrads-cpu / chunk8000 | 78.9/79.0 | 93.9/93.9 | both flat ⇒ 80k alloc floor is fg-expert+attention-phase bound |

Implementation landed (default-off, both backends):
- `ASYM_EMPTY_CACHE_PHASES` — `asym_gemm/profiling/lf_trace.py` (forward_end in
  `compute_loss_with_profile`, backward_end in `backward_with_profile`).
- `ASYM_EMPTY_CACHE_EVERY_GC_LAYERS=N` — `LlamaFactory/.../checkpointing.py`
  (`_maybe_release_cached_blocks_per_gc_layer` after each checkpointed layer's
  backward; global counter, every Nth layer).

## 8. VERDICT (2026-07-14, measured)

1. **Keep `EXPANDABLE_SEG=true`** — turning it off left allocated identical and
   blew reserved to 172.3 GiB (E3): the expandable allocator is SAVING ~80 GiB
   on this workload, not causing the gap.
2. **The residual +12.8 GiB reserved-over-alloc is not reclaimable free pages.**
   Boundary releases (D2) and per-4-layer releases (D3) each recovered only
   ~0.6 GiB (D3 also cost ~1–2% step time). Therefore the gap at the peak
   instant consists of (a) intra-segment fragmentation — free fragments sharing
   2 MiB pages with live blocks, which `cuMemUnmap` cannot release — and
   (b) freed-but-event-pending blocks on asym's side streams, which
   `empty_cache` correctly skips. Neither responds to release cadence.
3. **Accept 91.7 GiB as the honest A@80k reserved.** Quote reserved AND
   allocated in scoreboards (78.9 alloc); the gap is an allocator-shape
   property of chunked workloads, symmetric in kind (B pays +13 GiB at 45k).
4. The two knobs stay in-tree, default-off, for future workloads whose gap IS
   idle-pool-shaped: `ASYM_EMPTY_CACHE_PHASES` (lf_trace.py),
   `ASYM_EMPTY_CACHE_EVERY_GC_LAYERS` (LF checkpointing.py).
5. Further reduction would need shape-stable preallocated chunk buffers or
   fewer side streams (§6) — not warranted at the current goal margins.

Changelog:
- 2026-07-14: doc created. Anomaly quantified (§1), mechanism analyzed (§2),
  fix specified (§3–5). Pending: E1/E2 (expert-transient knobs, running now),
  E3 control, D1 trace attribution, D2 implementation A/B.
- 2026-07-14 (later): E3/D2/D3 measured; §7 table + §8 verdict written. D1
  trace attribution skipped as moot (D2/D3 empirically bounded the reclaimable
  fraction at ~0.6 GiB). Investigation closed.
