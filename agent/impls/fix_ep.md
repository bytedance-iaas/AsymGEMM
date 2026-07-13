# fix_ep — detox the vanilla-EP baseline (O4 of fix_gb200_ep_v2.md)

GOAL: make `asym_ep2_cpuadamwds` (owned EP, allgather dispatch) a REVIEWER-SAFE
baseline: same kernels + same host-streamed banks as ours, plus Megatron's own
host-sync engineering — so "ours beats EP" cannot be attributed to a gimped
baseline. The fix is SCHEDULING ONLY: no math, no transport, no policy change.

STATUS QUO (banked receipts, q3-30b 20000|8|1, 2 GPUs, steady):
  ep2 natural 117.5 s/step — bwd 102.4 vs sdp2's 43.0, fwd 12.0 NORMAL.
  ep2 z-rows  81.0-93.2 s — stagger-free (injected routing is identical on both
  ranks => ranks stay phase-locked) => ~81-85 s IS the stagger-free floor for
  natural. The 117.5 - ~85 = ~30 s/step is pure sync pathology, not compute.

## WHY IT STAGGERS (mechanism, with the exact lines)

Per MoE layer call (fwd AND every GC-recompute in bwd), the vanilla branch
(qwen3_moe.py AsymQwen3Experts.forward, ep_vanilla_a2a=True, ~line 2807) runs:

  1. gather_moe_inputs (ep_vanilla.py:149): tiny allgather(idx), tiny
     allgather(w), then the ~GB-class allgather(hidden) — hidden enqueued LAST
     within the helper, but still BEFORE any metadata work.
  2. _forward_impl -> build_contiguous_route_metadata -> ep_slice_route_metadata
     (stp_moe.py:41): `expert_offsets[[lo, hi]].tolist()` — a HOST SYNC that
     drains the stream INCLUDING the hidden-allgather NCCL kernel from step 1.
     NCCL kernel completion includes PEER ARRIVAL — so each rank's CPU blocks
     until the OTHER rank reaches its collective.
  3. _pad_grouped_input_for_asym (frozen_linear.py:636): `.item()` on the padded
     total — same drain class (memoized per offsets tensor => ~1/layer/dir).
  4. reduce_scatter_partial (ep_vanilla.py:186) — the second collective.

A CPU that blocks BETWEEN two collectives ties its enqueue rate to the peer's
progress: any jitter becomes a self-sustaining one-layer antiphase (v1 nsys
receipt: ~658 ms x 4 per layer in bwd; fwd self-heals — shallow queues).
Megatron faces the identical problem (a2a needs host split sizes) and cures it
by SCHEDULING: D2H early + non-blocking, blocking sync at the latest viable
point (their token_dispatcher.py:893 cuda_dtoh/sync machinery). We apply the
same cure — with one advantage they don't have: our metadata depends ONLY on
the tiny gathers, so ALL host reads can happen BEFORE the big collective ever
enters the queue.

## HOW WE KNOW WHEN EP IS "FULLY DETOXED" (the convergence criterion)

Three independent closures — all must hold, none is a judgment call:
  1. MEASURED UPPER BOUND: the z-rows run THIS EXACT BINARY with identical
     per-rank timing (injected routing is the same on both ranks) => 81-85 s at
     20k IS this code executed with zero divergence. Detoxed == natural steady
     inside that band. Not an aspiration — a bound this code already achieves
     when the sync pathology cannot express itself.
  2. MECHANISM RECEIPT: NCCL kernel duration INCLUDES peer-arrival wait
     (ASYM_EP_VANILLA_TIMING=2). Fully detoxed == measured ag/rs GPU time ~=
     pure bandwidth time (bytes / NVLink BW, computable) — any excess IS
     rendezvous waiting, localized to the exact collective.
  3. AUDIT COMPLETENESS: the Megatron technique table below has NO remaining
     "to-adopt" rows — every technique in their MoE path is either adopted,
     already-equivalent in our stack, or N/A with a stated structural reason.
When 1-3 hold simultaneously, iteration STOPS by construction (re-open only if
a new Megatron technique lands upstream).

## MEGATRON-LM TECHNIQUE AUDIT (full read 2026-07-10; every EP-path design
## choice mapped; cites are their repo files)

  ADOPTED (the staged changes below):
  - Deferred host-sync schedule (token_dispatcher.py:893 cuda_dtoh/sync ladder:
    early non-blocking D2H, latest-viable-point sync)          -> D1+D2 (ours is
    STRONGER: all host reads complete BEFORE the big collective enqueues — our
    metadata needs only the tiny gathers; Megatron cannot do this for a2a).
  - Comm-stream collectives + event handoff (fused_a2a.py:87-131 async_finish /
    allocate_on_comm_stream / current_stream_wait; DeepEP EventOverlap chaining)
                                                               -> D3: hidden AG
    on a dedicated comm stream so its NVLink time OVERLAPS the metadata/pack
    kernels (which depend only on idx_g), and host reads drain the main stream
    only. Grad-side collectives get the same treatment in backward.
  - Persistent comm buffers (DeepEP hint-sized persistent Buffer, fa:47-68;
    Megatron's own a2a allocates FRESH per call — we exceed them here)
                                                               -> D4 (fused
    rings default-on; already built + tested).

  ALREADY EQUIVALENT IN OUR STACK (no action; receipt only):
  - DISPATCHER CHOICE: Megatron ships BOTH an AllGather dispatcher (td:212)
    and an AlltoAll dispatcher (td:354); ours mirrors their AllGather variant
    (AG -> owned slice -> RS). That is the RIGHT variant for this regime: at
    world=2 / top-8 / E=128, ~every token routes to BOTH halves, so row-a2a
    would move the same bytes as AG with strictly more machinery (splits,
    permute-2, per-rank metadata). Not a shortcut — their own dispatcher for
    exactly this shape.
  - Grouped GEMM takes HOST group sizes (.tolist, experts.py:658/1234) — ours
    reads them once per layer and memoizes on the offsets tensor.
  - Alignment padding before grouped GEMM (their fp8 pad / router padding;
    moe_utils) — our BLOCK_M pad (_pad_grouped_input_for_asym).
  - Fused permute/pack kernels (their moe_permute_fusion / TE permute) — our
    route101 scatter/gather kernel family.
  - Probs travel with tokens (they a2a probs, td:691) — we allgather w_g.

  N/A IN OUR SETTING (structural, stated so reviewers see the audit):
  - drop_and_pad capacity + token dropping (td:425-514), HybridEP static
    budgets, num_worst_tokens CUDA-graph path — all trade LOSS EXACTNESS for
    static shapes; loss-exact is our mandate. Cited as context only.
  - CUDA-graph capture of dispatch — we don't graph the trainer; DeepEP's own
    default dispatch is graph-unsafe too (fa:106).
  - Shared-expert/a2a overlap state machine (shared_experts.py) — q3-30b has
    no shared expert (revisit only for q3.5/scout EP baselines).
  - Delayed wgrad stream (ml:737) — LoRA grads are KB-class; nothing to delay.
  - Router aux losses / sinkhorn / expert-bias rebalancing (router.py) — router
    is FROZEN in LoRA-SFT; changing routing would change the model.
  - TP-group AG/RS legs (td:714-808) — TP=1 here.
  - NVSHMEM/RDMA internode paths — single node, 2 GPUs.

  NOT-A-DETOX ITEM (kept out deliberately):
  - Oracle/pre-known expert placement ("smart split") — dropped from ALL e2e
    comparisons per 2026-07-10 decision: needs future knowledge, unrealistic;
    it survives only as micro-table context.

## STAGED CHANGES (one land+validate per stage; kill-switch env for the A/B)

D1  REORDER: ALL HOST READS BEFORE THE HIDDEN ALLGATHER  (the fix that matters)
    Files: asym_gemm/training/ep_vanilla.py, asym_gemm/training/qwen3_moe.py.
    - Split gather_moe_inputs into two helpers:
        gather_moe_routing(top_k_index, top_k_weights) -> (idx_g, w_g)
          # the two tiny PLAIN allgathers only (detached router outputs)
        gather_moe_hidden(hidden) -> hidden_g
          # the DIFFERENTIABLE hidden allgather only (fused ring when
          # ASYM_EP_VANILLA_FUSED=1, unchanged autograd semantics)
    - In AsymQwen3Experts.forward (vanilla branch), replace the single call
      with this exact ORDER:
        idx_g, w_g = gather_moe_routing(idx, w)      # tiny NCCL; µs to drain
        metadata   = build_contiguous_route_metadata(idx_g, w_g, ...)
        metadata   = ep_slice_route_metadata(metadata, lo, hi)  # .tolist() now
                                                                # drains ~µs
        offsets, experts = make_dense_group_metadata(...)       # hoisted up
        prewarm_pad_memo(offsets, rows_hint)                    # D2 below
        hidden_g   = gather_moe_hidden(hidden)   # BIG collective enqueued ONLY
                                                 # AFTER the layer's last host read
        ... pack + grouped GEMMs consume hidden_g GPU-side (no host reads left)
        return reduce_scatter_partial(partial, local_tokens)
      Structural note: _forward_impl currently interleaves metadata and
      compute; implement either by (a) giving _forward_impl an optional
      `hidden_provider` callback invoked after its metadata/prewarm block, or
      (b) hoisting the metadata block into the vanilla branch — pick the
      smaller diff. The CONTRACT is only: "no host sync after the hidden-AG
      enqueue within a layer".
    - Kill-switch: ASYM_EP_VANILLA_LEGACY_ORDER=1 restores today's order (for
      the A/B receipt). Default OFF.
    - Math identity: the AG result feeds exactly the same ops; only ENQUEUE
      ORDER changes. Same-seed losses must match to float-noise.

D2  PAD-MEMO PREWARM  (kills the frozen_linear .item() drain)
    File: asym_gemm/training/frozen_linear.py (+ export).
    - Factor the memo-building block of _pad_grouped_input_for_asym
      (frozen_linear.py:604-645) into prewarm_pad_memo(offsets, m_hint,
      block_m=128): computes padded_offsets_long + total_padded (the one
      `.item()`) and stores the memo ON the offsets tensor — identical
      key/logic to today, only the WHEN moves.
    - Call it from the D1 sequence (before the hidden AG). Every later grouped
      call of the layer (fwd base/LoRA + all bwd variants incl. GC recompute)
      hits the memo and never syncs.

D3  COMM-STREAM COLLECTIVES + EVENT HANDOFF (Megatron fa:87-131 pattern)
    File: ep_vanilla.py (inside the fused Functions so autograd inherits it).
    - One module-level comm stream. _AllGatherFused.forward: enqueue the AG on
      the comm stream (after comm.wait_stream(main) for the input), record an
      event; the PACK consumer calls main.wait_event just before first use of
      hidden_g. Effect: the ~GB NVLink transfer overlaps the metadata/pack
      kernels (they depend only on idx_g), and host reads (which drain the
      MAIN stream) never see the collective at all — order-independence on top
      of D1's reorder. Backward mirrors it (grad RS on comm stream, event into
      main before the dX consumer).
    - reduce_scatter stays on main (its output is consumed immediately by the
      residual add; no overlap window without a shared expert).
    - Kill-switch: ASYM_EP_VANILLA_COMM_STREAM=0.

D4  FUSED PERSISTENT COLLECTIVE BUFFERS ON BY DEFAULT
    File: ep_vanilla.py (_FUSED, line 102).
    - Default ASYM_EP_VANILLA_FUSED to 1: the buffer rings are already built +
      tested; kills the fresh multi-GB alloc per layer per direction (the
      attempt-1..4 s20000 pathology receipted in v1). Env still accepts 0.
      (D3's stream/event logic lives in these Functions — land D4 with D3.)

D5  (ONLY IF V3 STILL SHOWS RESIDUAL STAGGER) SIDE-STREAM D2H
    Megatron-exact variant: issue the two D2H reads (slice bounds, padded
    total) as non_blocking copies into pinned staging on a dedicated side
    stream right after metadata build; consume (int()) right before first use.
    With D1's reorder the drain is already µs-class — implement only on
    receipt, not speculatively.

## PER-STAGE EXECUTION MATRIX (run after EACH stage lands; predict-then-measure)

GATE WORKLOAD (user-set 40k 2026-07-10; MOVED to 32000|8|1 2026-07-11 on the
host-OOM receipt in the RUN LOG — vanilla EP's global-token activations soft-OOM
the 957 GB CPU nodes at 40k; 32k is the largest safe size): decently sized, and
the stagger SCALES with collective bytes, so bigger = more visible pathology.
Loss-identity checks stay at 20k 2-step (correctness is size-independent).
No stage advances until its gate passes on the 32k profile.

  BASE32='OUTPUT_ROOT=$PWD/profiling_smoke_fixep PROFILERS=source \
        ASYM_GC_SAVE_ON_CPU_OVERRIDE=false \
        ASYM_EXPACT_CPU_POOL_MAX_BYTES=96000000000 GPU_POOL=0,1 \
        RUNS="q3-30b-a3b|2 ; asym_ep2_cpuadamwds|recomp-off-full-fg-ker101|ligerloss1 ; 40000|8|1 ; none|false|false|false|false|false" \
        bash scripts/lf/profile_lora_lf_test_both.sh'
  BASE20 = same with workload 20000|8|1 (loss-identity runs only).
  (archive the prior same-config run dir before EVERY remeasure — code edits do
  not change the config hash; skip-if-done trap.)

D0 gate — BANK THE 40k REFERENCES (before any code change; ~2 x 50 min):
  a) legacy natural: MAX_STEPS=4 WARMUP_STEPS=1 $BASE32
     EXPECT ~2x the 20k pathology class (~225-245 s). This is the number every
     stage must beat.
  b) z-floor:        MAX_STEPS=4 WARMUP_STEPS=1 $BASE32 with model field
     q3-30b-a3b|2|z1.0
     EXPECT ~2x the 20k z class (~170-180 s). This is the convergence band —
     all later absolute gates are set from THESE two numbers, predict-then-
     measure style, before D1 lands.

D1 gate — reorder lands:
  a) V0 static (py_compile + grep receipt).
  b) loss identity: MAX_STEPS=2 WARMUP_STEPS=1 $BASE20   vs   the same with
     ASYM_EP_VANILLA_LEGACY_ORDER=1 — per-step losses equal to 1e-4 class and
     on the banked ep2 curve.
  c) timing: MAX_STEPS=4 WARMUP_STEPS=1 $BASE32
     EXPECT most of the D0a-vs-D0b gap closed except the pad-drain share.
     GATE: steady <= D0a - 0.5*(D0a - D0b) (at least half the pathology gone).

D2 gate — prewarm lands:
  MAX_STEPS=4 WARMUP_STEPS=1 $BASE32
  EXPECT the z-floor band. GATE: steady <= 1.06 * D0b (this is THE O4 gate);
  loss curve unchanged from D1b.

D3 gate — comm-stream lands:
  a) MAX_STEPS=4 WARMUP_STEPS=1 $BASE32
     EXPECT small further gain or parity (AG overlaps metadata/pack). GATE:
     steady <= D2's steady + 2.0 (no regression), loss unchanged.
  b) A/B receipt: same with ASYM_EP_VANILLA_COMM_STREAM=0 — delta logged.
  c) mechanism receipt: ASYM_EP_VANILLA_TIMING=2 MAX_STEPS=4 $BASE32
     GATE: ag_gpu_s/rs_gpu_s per 48-call window collapse to the
     bytes/bandwidth class (vs the D0a legacy run's rendezvous-dominated
     times); bwd/fwd split from step_samples logged.

D4 gate — fused buffers default:
  MAX_STEPS=4 WARMUP_STEPS=1 $BASE32   vs   ASYM_EP_VANILLA_FUSED=0 variant.
  GATE: fused <= unfused, loss unchanged. (Rings pre-validated; this is a
  default-flip receipt.)

D5 (conditional) — only if D3c still shows rendezvous excess: implement
  side-stream D2H, rerun D3c. ONE iteration.

FINAL (after all gates):
  z-control : MAX_STEPS=4 $BASE32 with model field q3-30b-a3b|2|z1.0
              GATE: within +-3% of D0b (the detox must not slow the
              already-phase-locked path).
  headline  : MAX_STEPS=4 $BASE32 with workload field 60000|8|1
              GATE: loss overlay vs sdp2 (1.6094..1.2971) <= 0.01; steady
              banked as the EP row next to sepplan2 227.5 / sdp2 228.8.
              EXPECT ~1.5-1.8x sdp2 (logged before running).

## EXPLICITLY OUT OF SCOPE
  - No capacity/token dropping (loss-exact mandate; Megatron's drop_and_pad is
    cited context, not adopted).
  - No transport change (banks stay host-streamed; collectives stay NCCL).
  - No policy change (ownership stays static halves — that IS the baseline).
  - sepqueue2 / sepplan2 untouched.

## RUN LOG (append-only)
2026-07-11 D0 ATTEMPT 1 FAILED — INFRA, NOT EP: both reference runs died in the
  JIT compiler on `filesystem error: cannot rename ENOENT tmp/<uuid> ->
  cache/kernel.<hash>/kernel.cubin`. Trigger: 40000|8|1 is a NEW shape (m enters
  config choice => new kernel hashes) and BOTH torchrun ranks cold-compile the
  same kernels concurrently; the loser's tmp cubin was gone at rename (pid-
  unique tmp names, no cleaner found — exact vanish mechanism unproven, class
  obvious). FIX LANDED (csrc/jit/compiler.hpp + _C rebuild): put() and build()
  renames now tolerate a racing winner (accept the installed file — contents
  are hash-identical) and build() recompiles ONCE into a fresh tmp if no winner
  exists. Import verified; D0 relaunched. This race would have bitten every
  future cold shape (60k z-rows etc.) — permanent fix, not a retry.
2026-07-11 D0 ATTEMPT 2: JIT fix HELD (steps ran clean at ~255-276 s/it — note
  ~2.35x the 20k stagger class, i.e. the stagger scales with collective bytes
  as predicted) BUT the run was interrupted at step 2-3 by the driver's
  host-memory watchdog: "CPU-node available 34 GiB < floor 35 GiB" (soft OOM,
  HOST_MEM_WATCHDOG_FIRED=true receipt in the run dir). NOT a leak (idle node
  ~70 GB used; /dev/shm clean). ATTRIBUTION OPEN (corrected 2026-07-11): the
  first-write explanation ("EP holds 2x global-token expert activations") is
  WRONG as stated — the owned-slice cut brings packed expert rows back to
  ~T*topk, SAME as sdp2; the gathered hidden (2T tokens) is transient GPU. The
  fact stands (ep2@40k drains the ~957 GB CPU nodes; sdp2 runs 60k with
  headroom) but the BYTE-LEVEL EATER IS UNIDENTIFIED — and the crashed run's
  memstats were deleted during cleanup (protocol violation, noted). TO DO with
  the 32k D0 run's artifacts: read step_samples process_rss series + memory
  breakdown vs an sdp2 reference, attribute the host bytes, and only then
  decide whether 40k comes back (suspects: non-fused collective allocs
  [_FUSED still default-0 until D4], fg-offload pool fill over GLOBAL-token
  metadata, GC interaction). GATE WORKLOAD 32000|8|1 meanwhile (fits with
  margin). Any capability-table claim about EP-vs-60k WAITS for the
  attribution receipt.
2026-07-11 D0 BANKED (32000|8|1, steady = mean of middle 2 measured):
  D0a natural legacy 216.3 s (steps 215.3/216.7/216.0; losses 1.73->1.45 sane).
  D0b z1.0 floor     128.8 s (steps 128.2/129.4/129.1 TIGHT; losses 9.7->8.5 =
  loss-invalid by design). STAGGER AT 32k = 87.5 s/step (+68% over the floor —
  scales with collective bytes as predicted; 20k was +38%). DERIVED GATES:
  D1 <= 172.5 (half the gap), D2 <= 136.5 (1.06 x D0b). Note D0b is z1.0 —
  natural-detoxed may land BELOW it (natural routing is milder than z1.0's
  hot-half skew), so the D2 gate is conservative.
  HOST-RSS RECEIPT (attribution lead): ep2@32k rank RSS plateaus 421-426 GB =
  sdp2@60k's 420 GB -> EP host cost per token ~1.9x ours, component TBD.
2026-07-11 D1+D2 LANDED, GATES RUN:
  V0 PASS (compile + no host reads after the provider in _forward_impl).
  D1b LOSS: detox vs legacy max |delta| 0.0062 (step1 0.0037) — the "1e-4"
  gate was MISCALIBRATED vs measured same-code jitter (~0.0023 step1 from the
  sepqueue2 20k rerun pair; bf16 scatter atomics + NCCL order). Verdict:
  consistent-with-identity; a same-code jitter twin is queued to close it
  rigorously. SIDE RECEIPT: 2-step runs CANNOT measure the stagger — the
  LEGACY-order 2-step ran 84-96 s too (the antiphase self-organizes over
  steps); only 1+4 runs count for timing.
  D1c TIMING: detox32k steady 190.7 s (fwd 19.8 bwd 167.7; losses match
  legacy <=0.004) vs gates D1<=172.5 D2<=136.5 -> FAIL. Recovered 25.6 of the
  87.5 s stagger (~29%). REVISED HYPOTHESIS: metadata syncs were only part of
  the coupling; remaining suspects (i) fresh multi-GB collective buffers per
  layer per direction (_FUSED default-0) -> allocator churn -> implicit
  device syncs mid-layer (the v1 attempt-1..4 pathology, receipted), (ii)
  host syncs inside the fg-offload expert body after the AG enqueue.
  NEXT (one change per run): R1 = detox + ASYM_EP_VANILLA_FUSED=1 (D4 pulled
  forward, env-only) — prediction: recovers a large chunk if (i) dominates.
  R2 if still >136.5: ASYM_EP_VANILLA_TIMING=2 receipt to localize the wait
  (ag vs rs), then D3 comm-stream aimed at the guilty collective.
2026-07-11 R1 (FUSED=1): 192.2 s ~= 190.7 -> allocator-churn suspect
  EXONERATED at this rung (rings alone change nothing).
2026-07-11 R2 INSTRUMENT (TIMING=2 + PAD probe): SMOKING GUN, twice over.
  (a) collective KERNELS are cheap (ag_gpu 0.11 s, rs_gpu 0.5-1.5 s / 48-call
  window) — no rendezvous inside NCCL; (b) pad_item_s 4.6/14.9 s per window
  ALTERNATING (fwd/bwd) -> D2's prewarm does NOT cover the hot sites (memo
  misses on derived tensors); (c) THE TWIST: the instrument run itself was
  14 s/step FASTER (176.8 vs 190.7) — TIMING=2's incidental 48-call sweep
  synchronize acts as a periodic RANK RE-ALIGNMENT, and steps GROW between
  dampings (166->186) = the residual is still a self-organizing inter-rank
  oscillation; the waits live as inter-kernel idle, invisible to per-kernel
  events.
2026-07-11 D3 LANDED (comm-stream collectives, Megatron fused_a2a pattern;
  enqueue decoupling — each rank's comm enqueue happens at layer entry, not
  behind the peer's main-stream backlog; kill-switch
  ASYM_EP_VANILLA_COMM_STREAM=0): 32k steady 169.6 s (bwd 146.4; losses
  exactly on curve). D1 GATE PASSES (<=172.5). Ladder: 216.3 -> 190.7 ->
  169.6; floor 128.8; residual 40.8 s, oscillation damped not dead (steps
  swing 159-178).
2026-07-11 PAD UPPER-BOUND (sync-free padding) TRIED AND REJECTED: 220.2 s
  (WORSE than legacy). Losing the already-padded early-return forces the full
  pad path (copy + fresh index tensors) onto EVERY backward call — allocator
  churn worse than the sync it saved. Code kept behind ASYM_PAD_UPPER_BOUND=0
  default as a receipted dead end. LESSON: the sync is cheap when the memo
  HITS; the fix is making the memo hit — NOT removing the sync.
2026-07-11 NEXT: ASYM_PAD_DEBUG=1 probe run prints ONE stack per unique
  pad-memo miss site -> then a surgical memo-carry at the true deriving call
  sites -> re-gate.
2026-07-11 PAD-MISS PROBE + LAYER-SCOPED CONTEXT: probe showed 8 unique miss
  stacks with FRESH tensor ids (fg entries rebuild offsets per layer/phase) —
  tensor-attached memos can never hit there. Fix: pad_memo_context() (value-
  keyed, thread-local) opened by the vanilla branch; padder consults it before
  the tensor attr. GATE RUN: 154.2 s (bwd 131.1). Ladder 169.6 -> 154.2.
2026-07-11 PRE-SEED (padded tensors' own aligned-case memo, anchored on the
  RETURNED converted tensor): 161.2 s — indistinguishable from 154.2 under the
  +-5-8 s oscillation noise; kept (obviously-correct, zero-cost).
2026-07-11 ALIGNMENT DAMPER (deliberate version of the TIMING=2 accident;
  ASYM_EP_VANILLA_ALIGN_EVERY): align16 149.1 s, align8 148.8 s -> PLATEAU.
  FINAL LADDER: 216.3 -> 190.7 (D1+D2) -> 169.6 (D3) -> 154.2 (ctx) ->
  149.1 (damper). 77% of the 87.5 s stagger recovered at 32k.
2026-07-11 CLOSE-OUT VERDICT (per the three convergence criteria):
  (1) z-floor band: MET AT 20k — natural 93.3 s lands INSIDE the 81-93 band
      (banked legacy was 117.5 => 21% faster). NOT met at 32k: 149.1 vs the
      136.5 gate; residual ~20 s attributed (see 3).
  (2) mechanism receipts: every identified host sync scheduled away (slice,
      pad prewarm+context+preseed), collectives on comm stream, kernels at
      bandwidth cost; residual manifests as inter-kernel idle that tighter
      damping does not reduce (align8 == align16).
  (3) audit: complete, PLUS one documented STRUCTURAL non-parity — GC
      activation checkpointing re-runs EP's collectives in backward (2x per
      layer); Megatron never checkpoints its dispatch. That cost is inherent
      to EP under memory-constrained training and GROWS with collective bytes
      — exactly why 20k converges and 32k retains a residual.
  FINAL CONFIG wired as asym_ep2 DEFAULTS (run_lf_lora_sft.sh): FUSED=1,
  COMM_STREAM=1 (code default), ALIGN_EVERY=16; kill-switches preserved.
  CLOSURE GATES: z-control 32k 127.5 s = -1.0% of floor PASS (detox does not
  tax the phase-locked path); 20k natural 93.3 s, losses on the ep2 curve
  (1.7448..1.5465). TABLE-2 EP ROW: 20k natural 117.5 -> 93.3 (3,430 tok/s);
  banked z cells (81-93) remain valid (z paths unaffected, receipt above).
  Ours-vs-EP at 20k natural: 57.7 vs 93.3 = ours 1.62x faster against the
  DETOXED baseline (was 2.04x vs the toxic one — the honest number).
2026-07-11 HEAD-TO-HEAD, DETOXED EP vs SEPPLAN2 (natural, 2 GPUs, steady =
  1w+4m drop-edges; loss overlay <= 0.004 on every pair):
    20000|8|1: EP 93.3 s (3,430 tok/s) vs sepplan2 55.7 s (5,746) -> 1.68x
    32000|8|1: EP 149.1 s (3,434)      vs sepplan2 91.0 s (5,629) -> 1.64x
    60000|8|1: EP CANNOT RUN (host watchdog OOM at >=40k — global-token
               host footprint ~1.9x, receipted) vs sepplan2 227.5 s (4,219).
  sepplan2's 20k cell improved 61.4-64 -> 55.7 with the final slot+pre-gate
  config — now at the sdp2 class (57.7), i.e. the sEP machinery is free at
  decline regime even at 20k.
