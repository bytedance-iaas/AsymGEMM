# AsymLoRA kernels — rhetoric, optimization targets, and new-kernel pipeline

(2026-07-27, Kevin + session handoff. Audience: the agent that keeps
developing/optimizing the LoRA kernels. Read this WHOLE doc before touching
anything; the point is not speed for its own sake — every unit of work here
exists to sharpen ONE paper claim. This doc is self-contained: the goals
are §0, the claim they serve is §1, the shapes are §3, the inventory and
task details are §4, the protocol is §5.)

## 0. THE THREE GOALS (Kevin 2026-07-27 — do these, in this order)

We NEED structurally different kernels — NOT trivial adaptations of the
AsymGEMM inference kernels. Concretely:

1. **Make the CURRENT kernels' (K1/K2) structural-novelty claim airtight —
   by proof and systematization, NOT by rewriting.** Verdict (2026-07-27,
   settled): K1/K2 already flipped every structural axis the LoRA workload
   forces (SMEM ownership, grid keying, retirement policy; K2's
   stream-axis reduction has no upstream counterpart) — they sit at the
   structural endpoint their dataflows dictate, and forcing further
   divergence would be artificial. Goal 1 therefore =
   (a) PROVE the structure is necessary: the direct-use bar (§4.3);
   (b) SYSTEMATIZE it: the K1/K2 family unification (§4.1b-1) — same
       instantiations, parity by construction, expressed as one designed
       space with upstream as a single point;
   (c) the ONE remaining structural axis, conditionally: the grouping
       axis via the boundary-crossing continuous stream (§4.1b-2),
       measure-first.
   Do NOT rewrite K1/K2's core dataflows chasing extra difference.
   Efficiency parity is acceptable throughout; regressions are not.
2. **Salvage the pair-LoRA kernels and make them MORE efficient.** Kevin
   recalls the pair kernel LOSING BIG ON TIMING in earlier runs — that is
   why it is env-gated off despite halving link bytes by construction.
   Diagnose why it loses (candidate causes: SMEM pressure / occupancy from
   the doubled descriptor+output loop, halved pipeline depth, per-half
   epilogue serialization, mainloop stalls between halves), fix it, and
   get it to WIN at link-bound cells — the byte arithmetic says ~2× is the
   ceiling on the gate/up legs, so a correct implementation should show a
   real speedup, not a loss. Then flip the default (gate C4: claim only
   with the measured number).
3. **Formulate and build NEW LoRA kernels that emphasize the contribution**
   (§4.2 pipeline: qkv shared-stream triple first, stream-once-serve-many
   as the abstraction, rank-scalable dA).

**Standard workload for ALL three goals** (measure everything on this,
plus smaller points only as sanity): **Qwen3-30B-A3B** (d=2048, E=128
experts, top-8, expert intermediate 768), **seq 480K × batch 2** (960K
pre-routing rows; routed rows = 960K × 8 = 7.68M, ragged per-expert
segments; routed X ≈ 31.5 GB bf16 pinned), bf16, T3 streamed recipe
(activations CPU-resident/pinned), attention legs at the same 960K rows.
In-container, GPU pairing + w1+m2 per §5. If a kernel cannot run this
workload, that itself is a finding — record it, do not shrink the workload
silently.

## 1. The rhetoric this work serves (the only success criterion)

**AsymLoRA ships NEW LoRA-specific streaming kernels that are structurally
different from the AsymGEMM inference kernels — not trivial adaptations.**
The argument, in the form the paper will state it:

- The upstream inference kernel is an engine built around one bet, placed
  three ways: *the streamed operand will be re-read by many output tiles*.
  The multi-buffered SMEM pipeline (cache the streamed tile), the
  output-keyed launch grid (many output tiles per streamed tile), and
  retire-as-you-go accumulation (the reduction axis is resident) all
  amortize stream bytes across output tiles.
- LoRA voids the bet **arithmetically**: reuse of a streamed tile = number
  of output row-tiles = ceil(r/tile_m) = **1** at r=64. Direct use still
  RUNS — a GEMM is a GEMM — but every mechanism then amortizes a reuse of
  one. Runs-but-inefficient is prong 1 (measurable, §4.3). The adapter
  gradient is prong 2: its reduction runs OVER the streamed axis, which no
  operand mapping of the inference kernel expresses at all — the only
  fallback degenerates to staging, which re-buys the offloaded bytes at the
  backward peak (categorical, no measurement needed).
- Our kernels therefore (i) replace each voided mechanism with its inverse
  — fetch-once slot instead of the reuse pipeline, stream-partitioned grid
  instead of the output-keyed grid, register-stationary output instead of
  early retirement — and (ii) where the workload allows it, **manufacture
  the reuse LoRA lacks**: one activation stream serving several adapters
  (pair today, triple next, §4.2). Inference kernels exploit reuse that
  exists; AsymLoRA kernels first survive without it, then create it.

Every optimization or new kernel must be justifiable in one sentence of
that register. If a change is generic GEMM tuning that upstream could ship
tomorrow, it does not serve the paper — do it only if it is also free.

## 2. Scope guard (hard)

- **LoRA-specific kernels ONLY**: the adapter contractions and their
  gradients, and shared-stream variants across adapters.
- **OUT of scope**: general asymm GEMMs (frozen weight streams — upstream's
  territory), grouped_mm compositions ("not our kernel"), and ALL
  MoE/routing fusion kernels (scatter/gather epilogues etc.) — Kevin
  2026-07-27: fusion is well-trodden engineering, it was relocated out of
  the motivation (M2b → §3.2) and must not absorb kernel-agent time.

## 3. The dataflows and shapes (why the smallness is the design driver)

Dense site (Qwen3-32B: d=5120, r=64, N = batch×seq rows, bf16; MoE form in
brackets: Qwen3-30B-A3B d=2048, E=128 experts, top-8 ⇒ R = tokens×8 ragged
rows, per-expert A_e):

| | inputs | output | contraction axis |
|---|---|---|---|
| **K1 fwd** `S = X·Aᵀ` | X [N×d] CPU-resident, streamed once (2.7 GB @256K) · A [r×d] GPU-resident, **0.66 MB** [per-expert 256 KB] | S [N×r] — long, narrow | d (shared width) |
| **K2 grad** `dA = dSᵀ·X` | dS [N×r] GPU (long, narrow) · X [N×d] CPU-resident, streamed | dA [r×d] — **entirely small, 0.66 MB** [E separate dA_e] | **N (the streamed axis!)** |

Consequences (the three structural commitments, paper §2.3): K1's output
keeps N → parallelism must come from row segments and SMEM ownership must
invert (the ~4000×-smaller factor is the only reused operand). K2's output
has no long axis left → nothing may retire until the stream ends → the
[r × k-tile] accumulators live in registers for a segment's whole stream,
written once. Ragged per-expert segments at rank 64 are why an expert-keyed
grid collapses (~1 block/expert).

## 4. Kernel inventory

### 4.1 Existing — optimization targets (keep them winning, keep them ours)

Code: `csrc/exp_act_offload/exp_act_offload_kernels.cu`; launchers
`asym_gemm/training/cpu_left.py`, `exp_act_offload_lora.py`, `lora.py`;
harnesses `tests/training/test_cpu_left_lora.py`,
`tests/training/test_profile_lora_backends.py`,
`scripts/motivation_bench/bench_m2a.py`.

- **`grouped_expert_lora_cpu_left`** (K1). Adoption 🟡 (attention LoRA-A fwd
  at dense T2/T3 + MoE T2B/T3; MLP fwd at dense T3). Baseline numbers
  (M2a, 2026-07-26): link-bound at ~210 GB/s, 1.598→25.388 ms across
  32K→512K rows, rel-err 4.7e-3, ≈3% faster than staged at equal link
  time with 0.7 MB held vs GBs. Optimization angles that stay on-message:
  sustained link saturation at SMALL N and at high routing skew (ragged
  tail segments); SMEM footprint of the fetch-once slot (frees occupancy);
  never re-introduce a reuse-shaped pipeline for X.
- **`sm100_grouped_lora_a_grad_bf16_cpu_right`** (K2). Adoption 🟡 (dense
  T3 MLP dA; MoE T2B/T3 down-proj dA). The register-stationary invariant
  is the claim — angles: register pressure vs occupancy at the shipped r;
  segment-parallel partials + cross-segment reduction only if it preserves
  "no mid-stream retirement" semantics per segment.
- **`grouped_expert_lora_pair_cpu_left` + pair-grad twin** — ✅ GOAL 2
  DONE (2026-07-27, this host c12; numbers in motivation_v2_plots.md
  "PAIR salvage A/B" entry; JSONs
  `profiling_results/motivation/pair_gateup_{default,compact}_grid.json`;
  harness `scripts/motivation_bench/bench_pair_gateup.py`).
  VERDICT: the recalled LOSS DOES NOT REPRODUCE on the current tree.
  - **fwd pair**: 224.7 ms vs 450.9 (2×single) = **2.007×** at default
    grid; **149.7 vs 299.3 = 2.000× at full link saturation (210.2 GB/s
    = raw-copy ceiling 211.2)** with DG_BF16_CPU_LEFT_COMPACT_GRID=1
    (the default grid's 140 GB/s is sentinel-block scheduling waste,
    equal in both arms — a K1 angle, not a pair defect). Outputs
    BIT-IDENTICAL to the single kernel. No kernel change was needed.
  - **pair-grad twin**: the shipped backward already called it
    (qwen3_moe.py / llama4_experts.py — the old "env-gated off
    everywhere" note was fwd-only). It DID have the defect family the
    doc predicted: serialized stage→compute loop + RANK_MAX=128 dead
    accumulator slots; the pair's doubled FMA/staging sat on the
    critical path (77.2→59.4 GB/s per pass). Fixed in
    exp_act_offload_kernels.cu v14+v15 (compile-time PAIR + exact
    rank-64 instantiation, register-prefetch of the next X/dS chunk
    under the current chunk's compute, f32 TRANSPOSED smem + float4
    row-vectorized inner loop, __launch_bounds__(512,2), zero spills):
    grad_single2 814.7→301.4 ms (**per-pass now LINK-SATURATED, 208.7
    GB/s**) and grad_pair 529.9→**162.7 ms = 1.853×** (193.4 GB/s
    per-pass; residual vs 2.0× = the pair compute still ~8% longer than
    one chunk's link time — Little's-law bound, recorded honestly).
    Outputs bit-identical to single-grad throughout.
  - **DEFAULT FLIPPED** (fwd): `grouped_lora_a_pair_forward_cpu_left`
    now takes the native pair by default
    (ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE=0 opts out; falls back to
    2×single when the binding is absent); lf profile scripts' `:-0`
    pins → `:-1`. Recorded in A.1 + toconfirm.
- (`asym_bf16_cpu_right_matmul` at the attention-dA shed-tier site is the
  non-grouped inverted-residency cousin — maintenance only, not a target.)

### 4.1b GOAL-1 structural candidates for K1/K2 (novelty-first; parity OK)

1. **Unify K1/K2 into one explicit "training-dataflow asymmetric GEMM"
   family.** Restructure so both are instantiations of one template over
   (streamed axis, reduction axis, output residency): K1 = stream ×
   shared-width reduce, K2 = stream × stream-axis reduce; upstream is one
   point in this space (stream-reused × resident-reduce × retire-early).
   Bytes unchanged; hazard = refactor only (same instantiations ⇒ parity
   by construction, verify with M2a + a K2 micro); kill: none. Converts
   "we modified a kernel" into "we defined the design space training
   needs" — the strongest possible §3.2 sentence.
2. **Boundary-crossing continuous stream (K1 grouped form; secondary,
   measure-first).** This is a rank-64 consequence, which is why it
   qualifies: a segment's work is ∝ r, so at rank 64 each per-expert
   pipeline drain/refill has almost nothing to amortize against — a
   problem upstream never faces because its grouped outputs are wide.
   Fix: stream X as one continuous row-space, descriptor/expert switching
   MID-STREAM — "the stream is primary, grouping is metadata" (inverts
   upstream's group-primary structure). MEASURE FIRST: link utilization at
   z=2.0 on the standard workload; ≥95% ⇒ no bubbles ⇒ KILL. Hazard:
   in-flight descriptor switching complexity.

VERIFY BEFORE ANY CROSS-KERNEL SHARING CLAIM: the attention-site
choreography of who reads X from where (base GEMM via
`asym_bf16_cpu_right_matmul` vs LoRA-A fwd vs dA — GPU-staged vs
host-in-place) — pin it down in `attention_activation_offload.py` /
`cpu_left.py` and record it here; it constrains which stream-sharing is
legal (and whether the triple's 3-reads count is fwd-only or fwd+dA).

### 4.2-PRE Novelty slate v2 (2026-07-29, Kevin directive: novelty over
engineering; LoRA = a new design space of asym kernels). The unifying claim:
upstream's engine assumes reuse EXISTS; AsymLoRA kernels first SURVIVE
without it (K1/K2 inversions), then MANUFACTURE it along three axes no
inference kernel expresses:
  axis 1 — across ADAPTERS: one stream serves k weight sets (pair k=2 ✅
           measured 2.0×; in-kernel triple k=3 = N1 below);
  axis 2 — across DIRECTIONS: one stream serves forward AND gradient
           dataflows simultaneously (N2 — the strongest new structure);
  axis 3 — across RANKS: the output-stationary invariant survives rank
           growth via an accumulator hierarchy (N3).
Plus N0: when the reduction axis IS the streamed axis (K2), parallelism
must come from SPLITTING THE STREAM — hierarchical stream-end retirement
(per-substream register residency + one merge), a grid upstream never
needs because its reduction axis is resident.

- **N0 — K2 stream-split grid.** BYTES: none saved; fixes K2's collapse at
  few-segment sites (measured 54 GB/s single-group vs 208 upstream-form —
  the reason attention dA still uses the upstream kernel). Adaptive split
  S = f(groups, k_tiles): S=1 reproduces today's kernel exactly at the
  128-expert cell (parity by construction). HAZARD: partials workspace
  [S,r,K] fp32 + merge kernel; reduction-order change (fp32 partial sums —
  tolerance-tested). KILL: if split kernel < 1.5× vs upstream-form at the
  attention shape, K2 stays grouped-only.
- **N1 — in-kernel qkv triple (k=3 outputs).** kPairOutput → kNumOutputs;
  slack record says resource-free (+0 SMEM, ~+4 regs). BYTES: same as the
  shipped cat-trick — value is STRUCTURAL (the fetch-once slot provably
  serves 3 consumers in-kernel; cat is a Python arrangement). KILL: any
  regression vs cat at link saturation.
- **N2 — dual-dataflow stream (one X pass → S AND dA).** At GC-recompute
  sites the SAME X is streamed twice in one backward window: once to
  recompute S=X·Aᵀ, once for dA=dSᵀ·X — and dS is computable BEFORE either
  (dS = g·B needs no S). One kernel, one X pass, grid (k-slices ×
  row-splits): each CTA reads its X block once, emits S-partials
  (reduce-add over k-slices) and dA-partials (stream-end merge) — a stream
  feeding a retire-per-tile dataflow and a stream-end-stationary dataflow
  SIMULTANEOUSLY. No inference kernel has a second consumer, let alone one
  per direction. BYTES: halves X reads at every recompute-window LoRA-A
  site (attention qkv today; ×3 with N1 = 6 consumers of one stream).
  HAZARD: fp32 S-partial traffic = N×r×4×2 bytes vs saved N×d×2 (d=2048,
  r=64 ⇒ save 4×); co-occurrence requires dS-first ordering (verified:
  attention bwd computes dS from grad_out before dA today). KILL: if
  measured win at the attention site < 1.3× vs two-pass, or dS-first
  reordering breaks a numeric test.
- **N3 — rank-tier accumulator hierarchy.** r sweep to the register cliff
  (64-reg boundary measured at r=64+prefetch), then SMEM-tier accumulators
  preserving stream-end retirement; turns K2 into a rank-family. KILL:
  none (worst case = a measured cliff table for the paper).

### 4.2 New kernels that serve the novelty (proposal pipeline, ranked)

PROPOSAL RULE (Kevin 2026-07-27 — no nonsense): every new-kernel proposal
must carry, IN WRITING before any code: (a) a byte/roofline argument for
why it wins (which bytes disappear, at which bound), (b) a feasibility
hazard analysis (registers / SMEM / occupancy / pipeline depth — the
resources the change pressures), and (c) a kill criterion (the measurement
outcome that retires the idea). A proposal without all three is not a
proposal.

1. **q/k/v shared-stream triple** (NEW; GATED ON GOAL 2). Byte argument:
   q/k/v LoRA-A all contract the SAME X, and the shipped code reads the
   host copy THREE times (bench_m3.py analytic note: "q/k/v share one U
   handle but each LoRA-A fwd + dA reads it"); the legs are link-bound
   (M2a), so on the standard workload X ≈ 3.93 GB/layer/pass → fwd triple
   removes ~7.9 GB, grad twin another ~7.9 GB ⇒ ~15.7 GB/layer ≈ 74 ms of
   link time; leg-level 2–3×, e2e single digits — state honestly. Shapes
   are symmetric (three A [64×d], three S [N×64]) so one fetch-once slot
   can feed three adapter pipelines. HAZARD: three rank-64 fp32
   accumulator sets alive across the whole d-loop ≈ 3× K1's register
   pressure → smaller row-tiles / occupancy risk — plausibly the SAME
   failure mode that made the pair lose. Therefore the pair salvage (goal
   2) is the gating experiment: pair loss fixable ⇒ triple follows; pair
   loss fundamental (multi-accumulator pressure starves the mainloop) ⇒
   KILL the triple too and record why. Do not start the triple before the
   pair verdict. NOTE pair-wins is NECESSARY, NOT SUFFICIENT: the X slot
   is shared (does not grow with k) but accumulators/descriptors do, so
   the pair verdict must record HOW MUCH SLACK remained (registers/
   occupancy margin at link saturation) — visible headroom ⇒ triple go;
   barely-scraping ⇒ stop at pair.
   **PAIR VERDICT SLACK RECORD (2026-07-27, ncu at link saturation):**
   - fwd kernel: SMEM-bound at 1 CTA/SM BY DESIGN (213.11 KB/block,
     IDENTICAL single vs pair — the B/CD slots are recycled, pairing
     adds only descriptors); regs 48 (single) → 52 (pair) of a 4-5-block
     register limit. **A third adapter costs +0 SMEM and ~+4 regs ⇒ fwd
     triple is resource-FREE — GO** (it saturates the link at 2.000×
     with visible headroom on every non-binding resource).
   - grad kernel: register-bound at EXACTLY the cliff — 64 regs/thread =
     the 2-blocks/SM boundary (theoretical occupancy 50%, achieved
     49.4%; SMEM limit would allow 5 blocks). A third dS/accumulator set
     (+8 accums +2 prefetch regs) breaks 64 ⇒ 1 block/SM ⇒ loses the
     cross-block link/compute overlap. **grad triple is NOT free**:
     needs RY rebalance or the SMEM-tier accumulator fallback (§4.2-3)
     first — i.e., fwd-triple-only is the safe first cut, with the grad
     staying pairwise (q/k dA + v dA = 2 X reads, still saves 1 of 3).
2. **Stream-once-serve-many as the unifying abstraction**: k adapters
   share one streamed operand (pair k=2, triple k=3). If both measure out,
   the paper names the abstraction once and the kernels become its
   instances — structurally unreachable from the inference kernel, whose
   pipeline serves exactly one consumer by construction.
3. **Rank-scalable output-stationary dA**: at what r does the register
   accumulator break, and what is the graceful tier (registers → SMEM) that
   preserves no-mid-stream-retirement? Turns K2 from a point design into a
   family across ranks — useful against "only works at r=64" reviews.
4. Considered, deprioritized: A→B epilogue fusion (adapter-delta into Y) —
   fusion-flavored engineering, small bytes; only touch if a measurement
   shows something structural.

### 4.3 The direct-use baseline bar (measurement task, prong-1 evidence)

Build the runs-but-slow bar with NO strawman: `asym_bf16_cpu_right_matmul`
IS the upstream inference form in our tree — compute Sᵀ = A·Xᵀ by operand
swap (A as the 64-row resident "batch", X as the streamed "weight"), dense
site first, and the expert-keyed grouped direct form if expressible.
Deliverable: measured table (direct-use vs `grouped_expert_lora_cpu_left`,
M2a shapes + one grouped cell) for §2.3/§2.4.1. Expect the grouped case to
collapse hardest; the dense case may only degrade — report whatever is
true, the prose adapts (EXPECTATION rule).

## 5. Protocol + rules (house, binding)

- In-container only (this host: `asym_sft_40` = `/workspace/AsymGEMM-SFT`),
  GPU pairing rule: GPUs 0/1 = NUMA node 0, 2/3 = node 1 — membind must
  match the GPU; never straddle. Orphan-process sweep + idle GPU before
  every run. w1+m2; CUDA events + device sync (micro protocol in
  motivation_v2_plots.md header).
- Used-only rule (toconfirm.md): nothing enters paper prose before a
  CONFIRMED in-container measurement; ❌ kernels appear only as
  status-marked capability sentences. Numbers land in
  `motivation_v2_plots.md` (new dated entries) and the A.1 audit
  (GOAL/IMPL register verbatim).
- Kevin commits the repo; leave work uncommitted. Overleaf prose is not
  this agent's surface — record numbers + GOAL/IMPL lines and hand off.

### 4.2-PRE-v3 (2026-08-01) — the stream-algebra slate (novelty round 3)

FRAME (paper register): a LoRA-SFT step is a SYSTEM OF STREAMS — activations
outbound (offload), activations inbound (LoRA legs), frozen weights inbound
(fwd + dx), gradients inbound (dS/dB/dx consumers). AsymGEMM optimizes ONE
point: weight-stream × single contraction × retire-early. AsymLoRA v3 =
completing the algebra along axes it cannot express. All are structurally
new vs our own K1/K2 family too. Acceptance per Kevin: real-model,
near-capacity, memory OR latency, never losing.

- **P1 — weight-stationary dual-orientation stream.** In GC-recompute
  backward windows the SAME frozen W is fetched twice back-to-back:
  recompute-Y = X·Wᵀ then dX = dY·W — two OPPOSITE-orientation contractions.
  One W stream serves both: grid keyed by W TILES (a third grid-keying —
  neither output-keyed (upstream) nor stream-row-keyed (K1/K2)); each W tile
  feeds an NT consumer (Y) and an NN consumer (dX) before eviction. BYTES:
  halves W link/stage traffic in every recompute window (T1/unsloth tiers,
  ohbm0: q32 ~35 GB/step, llama ~141 GB/step halved). HAZARD: two output
  pipelines + orientation-dual SMEM layout (swizzle must serve both
  majors); e2e ceiling ~1% latency at compute-heavy cells (state honestly);
  removes the staged-dispatch W buffer (GPU headroom) as the memory angle.
  KILL: wired A/B loses at any cell, or the dual-major SMEM forces a rate
  below the single-orientation stream.
- **P2 — gradient-stream multi-consumer kernel (mixed retirement).** At
  cpu-source sites the output-gradient dY is HOST-resident and is consumed
  THREE ways: dS = dY·B (retire-per-tile, long output), dB = dYᵀ·S
  (stream-end stationary, tiny output), base-dx = dY·W (retire-per-tile).
  One dY stream feeds consumers with DIFFERENT retirement policies in one
  pass — stream-once-serve-many generalized across RETIREMENT CLASSES (our
  pair/triple/dual all shared one class). A dead atomic prototype
  (`sm100_grouped_lora_b_backward_bf16_cpu_source`, "not on best paths")
  proves the site; the novel kernel replaces it with the structured form.
  BYTES: dY re-reads at that site 3→1. HAZARD: register budget = dB
  accumulators + dS tile buffers coexist; verify dY host-residency per
  tier before building (grad_out_cpu path). KILL: site not host-resident
  in shipped tiers, or <parity vs split kernels.
- **P3 — offload-through-compute (outbound-stream math).** Emit S (and the
  host copy) in ONE pass while X leaves the GPU: compute on the OUTBOUND
  direction — with P1/P2, both link directions carry computed bytes, never
  dead ones. BYTES: the last inbound X read (post-triple) → 0. HAZARD:
  D2H write + GEMM read contention on the same tile; needs store-through
  tile pipeline. KILL: slower than triple+offload overlapped, or memory
  delta nil.

4.2-PRE-v3 STATUS (2026-08-01): P2 KILLED by its own site check — dY is not
host-resident on any shipped tier (grouped_lora_b_backward_cpu_source has
no caller; dS/dB run as GPU GEMMs, no stream exists to share). Recorded per
the kill rule. LIVE: P1 (weight-stationary dual-orientation stream — the
new grid-keying axis; build at the unsloth/T1 recompute window, A/B on both
Qwen cells per the e2e-only rule) and P3 (offload-through-compute). P1 first.
