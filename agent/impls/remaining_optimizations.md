# remaining_optimizations — the forward-looking ledger (2026-07-19)

One place for every known-but-unbuilt optimization. Summary table first; self-contained
detail below. Provenance: fix_asym.md (§2/§5 ledgers), scheduler_v2.md (§3/§8/§9),
agent/handoffs/prompt.md (scheduler design + validation log), c14 campaign tables
(agent/impls/s04-p1-dgx-02-c14/).

## Summary table

| # | item | what it buys (measured/est) | class | effort | blocked on |
|---|------|------------------------------|-------|--------|-----------|
| 1 | C1b: fused MoE routing on the staged path | kills Set A's +36 GiB routing buffer while keeping cuBLAS speed → collapses the engine tradeoff | engineering (vLLM/Megatron-class kernel) | kernel-days | — |
| 2 | Graded per-layer keep-acts (keep last k layers) | turns Set B's one ~50 GiB step into a smooth dial; greedy-in-k provably exact (concave frontier) | mostly engineering | medium | — |
| 3 | GPU buffer pool (Megatron GPUTensorPool pattern) | reclaims measured 57–62 GiB reserved-vs-alloc fragmentation at high B → +2–3 batch | engineering | medium | — |
| 4 | H2D prefetch (issue-ahead) + D2H overlap residual | ~5–8 us/tok of remaining copy-wait (side-streams alone measured null without true prefetch) | engineering | medium | — |
| 5 | Re-profile asym-mem on post-fix source | honest constants for the leanest rung; the ~1.5M-token ceiling claim rests on a pre-fix anchor | measurement | 2–3 runs | GPU time |
| 6 | 1.1M balanced-mode validation run | confirms the scheduler's first flag-drop decision on hardware (pred 436 tok/s, ~158–175 GiB) | measurement | 1 run (~3 h) | user go |
| 7 | dX-side index-op reuse (sibling of shipped F-A) | ~3–5 us/tok (twin-trace index-bucket residue) | engineering | small | — |
| 7b | Port keep-acts to linear-attention module (q3.5 hybrid models) | 122b @32k×8: kills much of the 1165 GiB/step CPU round-trip → closes most of asym's −7.5% vs uns AND relieves RSS 903 (host is asym's binding wall at 122b) | engineering (mirror of shipped attn-KA) | small-medium | — |
| 8 | Host-RAM dial integration (ohbm as scheduler asset) | host-wall coverage; the arbitrage knob is already priced in scheduler_v2 §1/§3 | engineering | small | — |

Research vs engineering, honestly: items 1–4 and 7–8 are adoption of known techniques
(DeepSpeed ZeRO-Offload/Infinity, Megatron fine-grained offload + tensor pools,
vLLM/SGLang fused dispatch). The project's defensible research is elsewhere:
(a) the asym GEMM primitive — training-path dense GEMM computed DIRECTLY against
CPU-resident weights over C2C, no staging; (b) the scheduler formulation — measured
affine cost model + single knapsack over batch+residency assets, modes as emergent
labels, baseline fallback by argmax, predict→probe→refit loop, with machine-checked
nested/monotone behavior (scripts/lf/asym_scheduler.py --selftest); (c) the empirical
frontier map — per-config walls, batch-flat knee law, edge-penalty regimes, the 640k
parity crossover and sole-coverage window to 800k+ (≈2.3× the strongest baseline's wall).

## Details

### 1. C1b — fused routing on the staged path
Today the fast engine (ASYM_GEMM_DISPATCH=staged + ker000) materializes the
route-space tensor — one row per (token × chosen expert), ~36 GiB @120k×8 — because
the fused route kernels (ker101) exist only inside the slow streamed engine. C1b =
write the standard fused-dispatch kernel (gather/scatter inside the grouped GEMM,
vLLM/Megatron-style) over torch grouped_mm / cuBLAS. Result: staged speed AND ker101
leanness — the engine tradeoff (tuning Set A) collapses; the scheduler's ker000 rung
memory price drops to ~0 and the latency window extends right. Scope note: asym-path
only — superoffload's expert compute is the stock HF path and is untouched by this.

### 2. Graded keep-acts
Set B currently keeps ALL layers' recompute-saved tensors in GPU memory or none.
Backward consumes layers last-to-first, so the LAST layers' saves are needed soonest —
keeping only the last k layers gets most of the latency win at a fraction of the
memory. The frontier in k is concave (scheduler_v2 §2.1) so greedy is exact. Needs a
per-layer gate + count env in the fg wrapper. Turns the biggest ladder step into a
near-continuous dial; the scheduler then emits k instead of on/off.

### 3. GPU buffer pool
Measured: reserved−allocated gap grows to 57–62 GiB at 208k b4–b5 (torch caching
allocator fragmentation under many large, varied allocations/frees). Megatron's
GPUTensorPool pre-allocates shape-keyed recycled buffers; adopting the pattern for
the offload/staging temporaries collapses the gap → +2–3 samples of batch at fixed
seq, and near-100% utilization becomes safe (removes the edge-penalty band too).

### 4. Copy prefetch/overlap
The shipped async-pack/unpack flags put copies on side streams but measured NULL at
current anchors because nothing issues AHEAD of use: backward still requests each
tensor at consume time (activation_offload.stage()'s own comment documents why its
side-stream is structurally null). True prefetch = at layer ℓ's backward start, issue
layer ℓ−1's restage on the side channel; symmetrical for outbound stores. Sized
~5–8 us/tok from the twin-trace memcpy residue (+8.3 H2D, +5.3 D2H @208k b2).

### 5. asym-mem re-profile
All memory-mode numbers (100.3 GiB @120k×8 ladder rung, the 174k|8 = 170.9 GiB
capacity anchor, the ~1.5M b1 ceiling extrapolation) predate the phase-2 fixes.
2–3 runs on current source (480k b1, 800k b1, one big-batch point) refit the
mem-family constants; until then the scheduler's staged-only/bare rungs carry
pre-fix uncertainty.

### 6. 1.1M balanced validation
The interrupted run (tputschedb). Exact recipe in agent/handoffs/prompt.md v2
validation log: staged+ker000+pins, NO keep-acts flags, NO GC override,
MAX_SAMPLES=512 MAX_STEPS=3, seq 1100000 b1. Gates: fits (~158–175 GiB), tok/s
436±10%; an OOM is itself a valid bracket (the allocator then drops ker000 → re-emit
and rerun).

### 7. dX index reuse
F-A (shipped: ASYMM_QWEN3_MOE_FG_REUSE_PACKED_X) reuses forward's packed-X for the
dA weight-grad gather. The twin-trace index bucket still carries dX-side
scatter/gather work vs sup's 6.2 us/tok baseline. Same pattern as F-A: keep the
route-space intermediate for the dX scatter instead of re-deriving indices. Small,
contained in qwen3_moe_finegrained.py backward.

### 8. Host-RAM dial
The scheduler currently prices only HBM. The class-3 knobs (ohbm root placement,
async grad staging — scheduler_v2 §1/§3 D4) arbitrage HBM↔host-RAM and matter when
RSS approaches the host watchdog (measured 400–900 GB class at ultra-long). Add DRAM
as a second endowment with the same water-fill; reserved_dram analog of reserved_hbm.
The math already exists in scheduler_v2 §0 (the β_D term); this is plumbing.
