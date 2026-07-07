# GB200 Angle v2 — honest baselines, real GB200-specificity, and a design that survives the
# "why can't the baseline just add your trick?" test

Living doc; supersedes `archive/gb200_angle.md` (archived; holds the detailed SuperOffload
mechanics receipts, the FP4-in-codebase findings, and the killed-angles graveyard — mine it when
writing related work).
`gb200_tp.md` stays the implementation substrate — its stages are unchanged; anything adopted here
lands as new stages/knobs on top. Iterate this doc until convergence.
Status: v3.1 — D2 NOVELTY CHECK COMPLETE (2026-07-05): **PARTIALLY-NOVEL, core mechanism SURVIVES
in narrow form** — no prior instance of a cross-GPU work-stolen grouped GEMM via a shared queue in
COHERENT HOST memory (system-scope atomics, 2 GPUs, 1 logical grid) for MoE TRAINING with ownerless
host-streamed experts. Two claimed EFFECTS are taken and demoted to corollaries (zero-a2a: Janus
"move experts not tokens"; no-token-drop: MegaBlocks dropless). Infra prior art to cite UP FRONT:
ICS'25 device-side multi-GPU queues (NVSHMEM, HPC task graphs), Atos (ICPP'22), Chen IPDPS'10,
cuBLASXt/BLASX (host-ORCHESTRATED dynamic multi-GPU tiling from host memory, PCIe-era, dense,
inference). CONCURRENT-WORK WINDOW IS CLOSING: MegaTrain (2604.05091, 1-GPU C2C training stream),
C2CServe (2605.19481, GH200 serving), FEPLB (2604.19654, within-step NVLink rebalancing but
OWNED+resident experts, host scheduler), SonicMoE (2512.14080, Blackwell MoE kernels 1-GPU),
Perseus (2605.00686, multi-node megakernels) — none combine the pillars; move fast.
Doc-side convergence REACHED. Remaining: empirical gates + A4 (fold D2/D3 into gb200_tp.md).
Self-verdict on v2.1 (kept below as substrate/honesty analysis): does not clear the bar alone.

---

# v3 THE STORY (manager-level; supersedes v2's bottom line)

## Why SOTA fails on GB200 + why trivial add-ons cannot fix it

```text
resident TP/EP (Megatron/NeMo): experts must be HBM-resident -> 235B-class MoE (Qwen3-235B-A22B,
  ~470 GB bf16) cannot fit the pair's 372 GB at all; dense 70B OOMs the long-seq frontier.
  Add-on "stage/offload weights": staging reads the ENTIRE expert bank per layer regardless of
  routing, keeps static expert OWNERSHIP (imbalance stays), and no staged-TP-LoRA-offload system
  exists — building one ~= building our runtime (we do, as baseline B3).
DP/ZeRO-offload (SuperOffload, SOTA superchip trainer): 1:1-symmetric design; on 2:1 both ranks
  burst on ONE Grace in the same direction simultaneously. For MoE it is structurally catastrophic:
  ZeRO-3 ALL-GATHERS THE FULL EXPERT BANK TO EVERY RANK EVERY LAYER (~full-model bytes x 2 ranks x
  fwd+bwd per step) over the shared link; we read only routed tiles, once.
  Add-ons: shm-one-copy fixes capacity ONLY (works only because our base is FROZEN — full-FT
  cannot share); phase tricks cannot split b=1 sequences, cannot fix the MoE gather, cannot use
  the idle NVLink.
MoE trainers (Megatron-EP/DeepSpeed-MoE/Tutel; ES-MoE closest): HBM-resident OWNED experts + token
  all-to-all + aux-loss/capacity-factor (dropped tokens). Static EP-2 under skewed fine-tuning
  routing idles one GPU. Add-on "add streaming to Megatron-EP": needs tile-streaming kernels +
  UN-owned experts + tokens replicated on both GPUs + fine-grained cross-GPU coordination = the
  whole system.
```

## Our design (D1-D3), each pinned to a GB200-only leg

```text
D1 SUBSTRATE (dense): one frozen arena in Grace; disjoint-TP tile-streaming; zero weight-HBM.
   (= gb200_tp.md I0-I6. HONEST: ties a well-built staged variant on dense throughput — dense
   carries capacity/frontier rows, not the novelty.)
D2 HEADLINE — SKEW-ADAPTIVE COOPERATIVE MoE EXECUTION: experts OWNERLESS in the shared arena; both
   GPUs hold ALL tokens (replicated residual — affordable only because NVLink is 4x the C2C); each
   MoE layer = ONE logical grouped-GEMM work queue in CACHE-COHERENT Grace memory; persistent
   kernels on BOTH GPUs self-schedule (expert, token-block) items via system-scope atomics — hot
   experts split across GPUs, cold experts atomic. => perfect load balance under ANY routing skew,
   zero all-to-all, no aux-loss, no token drops, only routed tiles ever read from host.
   Structurally unavailable to baselines: GH200 has no second GPU on the Grace; PCIe has no
   coherent host atomics; multi-process NCCL stacks have no shared queue and their experts are
   owned + resident. FEASIBILITY (code-checked): asymScheduler.cuh already enumerates
   (expert, m-block, n-block) work — converting blockIdx enumeration to a persistent loop popping
   an atomicAdd_system counter in pinned host memory is contained kernel surgery; both GPUs use
   the SAME full-bank TMA descriptor over the one arena (no E/2 slicing, no local-id remap —
   SIMPLER than static EP-2); outputs scatter into per-GPU y_i (items disjoint) -> existing
   AllReduce2 combine; LoRA dB partials allreduce2 (tiny).
   NOVELTY VERDICT (checked 2026-07-05): core mechanism SURVIVES in this NARROW wording:
     "kernel-level, ownerless, coherent-host-queue work stealing across a superchip for MoE
     TRAINING" — cite ICS'25/Atos/IPDPS'10 (cross-GPU device-side queues, other domains) and
     cuBLASXt (host-orchestrated dynamic multi-GPU tiles) UP FRONT. DEMOTED TO COROLLARIES (taken
     as effects): zero-a2a (Janus already moves experts-not-tokens) and no-token-drop/aux-loss-free
     (MegaBlocks dropless; DeepSeek loss-free bias is the algorithmic sibling — claim only the
     SYSTEMS half: "2-GPU throughput invariant to routing skew at kernel granularity"; note the
     aux loss also shapes specialization, so we do not claim training-quality equivalence blindly).
   D2 ATTACK LEDGER (answer in-paper):
     ATTACK "it's CUTLASS's atomic tile scheduler with the counter in pinned host memory — a
     50-line delta; cuBLASXt did dynamic multi-GPU host-operand tiling in 2014": PARTIALLY LANDS
     on the naive form. The research content is exactly what the naive form lacks: (a) system-
     atomic CONTENTION over C2C — our own kernel history moved asymScheduler to per-expert CTA
     ownership precisely to eliminate atomic contention; re-introducing a shared cross-GPU queue
     PROFITABLY (hierarchical/chunked items, GPU-affinity hints, steal-only-on-imbalance) is the
     contribution; (b) hot-expert splits duplicate weight-tile fetches (2x C2C for that expert) —
     queue GRANULARITY/AFFINITY co-design decides when splitting pays (K-outer amortization);
     (c) grad determinism under nondeterministic work splits (fixed reduction trees per expert);
     (d) LoRA-grad reduction placement. EVAL BAR the reviewers will set (predeclare): must beat
     (i) static E/2 split + per-GPU AsymGEMM (our own I7 rung), (ii) scheduler-level token
     rebalancing (FEPLB/ES-MoE style, host-decided), (iii) host-orchestrated dynamic dispatch
     (cuBLASXt-style) — at real fine-tuning skew AND at synthetic worst-case skew.
D3 PLACEMENT: freed HBM (~2x150 GB) pools as a DEDUP'D checkpoint tier (bit-identical residuals),
   Grace = spill; RELAY-RESTAGE for replicated re-reads (one C2C read + NVLink forward instead of
   two C2C reads — halves DRAM/C2C bytes for all replicated H2D); one duplex/phase scheduler for
   the shared link. (v2's M3/M3'/M4, kept with their qualified claims + attack ledger below.)
FLAGSHIP: Qwen3-235B-A22B LoRA on one GB200 NODE — HONEST ARITHMETIC: bf16 arena ~470 GB does NOT
   leave room on a single 480 GB Grace for pinned act pools + optimizer (~100-200 GB); the flagship
   runs with the arena spanning the node's two Graces (960 GB, HC1; hot layers pair-local) OR with
   an NVFP4 arena (~132 GB incl. scales — quality-gated, the AMP). Either way the point stands:
   resident (>372 GB pair HBM) and two-copy DP (2x470 GB) are impossible on the node; one-copy
   streamed is the only design that runs it. Plus the existing matrix (q3-30b-a3b, llama4-scout
   carry D2's skew/balance results).
```

## Why GB200-specific (one line per leg)

```text
LEG1 2 GPUs on ONE COHERENT Grace  -> shared arena + shared atomic work queue        (D1, D2)
LEG2 NVLink = 4x per-GPU C2C       -> token replication, relay-restage, HBM pooling ~free (D2, D3)
LEG3 one 500 GB/s DRAM behind both -> forces dedup/disjointness/phase-scheduling     (D3; the
     penalty that makes every symmetric port collapse — the motivation figure)
```

---

---

## 0. Verdict up front (what changed after re-thinking)

1. The headline baseline CANNOT be SuperOffload alone — that comparison is confounded (TP vs DP).
   The strongest nontrivial baseline is **one-copy offloaded (staged) TP-2** — which NO framework
   ships; we must BUILD it ourselves as a rung (`tp2_offstage`), and we must be honest that at the
   dense long-seq flagship it will nearly TIE pure tile-streaming on step_s (T3 below).
2. Therefore "tile-streaming beats staging" and "one copy saves memory" are NOT the paper. The paper
   is the COMPOSITION the GB200 topology forces and enables: zero-weight-residency streaming (M1)
   creates ~150+ GB/GPU of HBM slack, the 2:1 topology makes Grace the choke and leaves a 900 GB/s
   peer NVLink idle, so the slack becomes a **pair-pooled, deduplicated, NVLink-served activation
   tier** (M3) + a **cross-GPU duplex DMA schedule** (M4) that no shipping baseline can retrofit
   without becoming our system.
3. GB200-specificity is stated as a resource-shift law, not a feature list:
   **GB200 halves every per-GPU Grace resource (BW, capacity, cores) and grows the per-GPU HBM and
   adds an on-package peer link; a correct GB200 design must move load OFF the halved resources ONTO
   the grown ones. Every existing system does the opposite or leaves the grown ones idle.**

---

## 1. Q1 — the baseline question (kill the TP-vs-DP confound)

The reviewer failure mode: "you compared your TP system against a DP system; the win is just
TP-vs-DP (or batch-shape), not your mechanisms." Fix: a LADDER where each rung isolates one thing.

```text
B0  asym_dp2          two |1 processes, 1 Grace     SHIPS (trivially)   isolates: naive-port collapse (motivation)
B1  superoffload_mem  ZeRO-3 DP offload, b4/GPU     SHIPS (SOTA offload) external validity row (NOT the headline)
B2  tp2_resident      TP-2, weights in HBM          SHIPS (Megatron/Automodel semantics)  isolates: residency wall
B3  tp2_offstage      TP-2, ONE host copy, per-layer double-buffered staging to HBM
                      DOES NOT SHIP — WE BUILD IT   **the strongest nontrivial baseline** — isolates M1
B4  asym_stp          TP-2, ONE copy, in-kernel tile streaming (M1)      substrate
B5  B4 + M3 + M4      + peer-HBM checkpoint cache + duplex scheduling    the system
```

Rules that keep it clean:
- Headline comparisons: B5 vs B3 (mechanism win, same parallelism, same copy count) and B5 vs
  B1/B2 (external validity vs what actually ships). B0 is the motivation figure only.
- Batch-shape fairness: TP rows b8-global, DP rows b4/GPU (equal global batch); PLUS b=1 max-seq
  rows where DP structurally cannot split a sequence — that is the frontier comparison, stated as
  such, not hidden inside throughput tables.
- HONESTY LINE (predeclare): we EXPECT B4 ~= B3 on step_s at the dense flagship (T3). The streamed
  rung is kept because it is strictly simpler + frees the last GBs and wins at small-M (MoE tails,
  b1); the THROUGHPUT wins must come from M3/M4, and each carries an ablation gate (>=10% or drop).

Why "one-copy staged TP" doesn't ship (receipts in gb200_angle.md): SuperOffload = ZeRO-3 DP, TP=1,
1 GPU:1 Grace, 64 MB bucket-prefetch-to-HBM, full-FT, MoE-agnostic. Megatron/Automodel TP =
HBM-resident weights, no streaming (Bridge cpu-offload = TE bulk swap PP=1-only; Automodel = FSDP2
whole-unit offload = DP-shaped). ES-MoE = PCIe whole-expert-prefetch DP. Nobody composes
{TP shards} x {one pinned host copy} x {offloaded activations} x {LoRA}. Building B3 takes ~90% of
our runtime (single-process dual-device, shared arena, TP autograd with LoRA layouts) — which is
precisely why it is an ABLATION of our system, not prior work. Present it that way.

The frozen-base nuance that defends one-copy (use it): a FULL-FT system CANNOT share one weight
copy across workers — each rank UPDATES its shard every step (ZeRO-3's per-rank ownership is
intrinsic; SuperOffload's 12-Psi optimizer offload is its whole point). One shared read-only arena
is a **frozen-base/LoRA-specific design point**, not an oversight in prior work — and a plain
shm+cudaHostRegister retrofit onto DP fixes only CAPACITY (still 2x stream traffic, still no b1
split, still idle NVLink). Say this before the reviewer does.

---

## 2. The truth table (adversarial arithmetic — what is and isn't real)

```text
T1 ACTIVATIONS DOMINATE. Measured (fix_qwen3.md): 9.93 TB/step activation D2H+H2D at q3-30b s80000
   vs ~0.12 TB/step weight reads. Dense 70B s25000: est 2-4 TB acts vs 0.28 TB weights (2 reads).
   => weight-copy-count and staged-vs-streamed are SECOND-ORDER for bandwidth. Any honest step_s
   win must come from the ACTIVATION path (M3/M4) or overlap quality — not weight strategy.
T2 ONE-COPY is a CAPACITY killer for DP, not a bandwidth story: 2x70B copies = 280 GB of the ONE
   480 GB Grace; + act pools + optimizer -> soft host OOM (the 35 GB watchdog). Real, measurable,
   but capacity-retrofittable via shm (see above) — so never the headline.
T3 STREAMED ~= STAGED at large M (the uncomfortable one). Staging hides when T_compute >= T_copy:
   per layer, T_copy = S/C2C ~ 0.9 GB / 225 GB/s ~ 4 ms; T_compute at M=200k ~ 200 ms => hidden
   50x over. Streaming reads the SAME bytes over the SAME C2C direction (TMA vs copy-engine changes
   the agent, not the bytes). Streamed's real deltas vs staged: ~2 GB slack, no HBM round-trip
   (HBM BW is free anyway), tile-granular access. Tile granularity matters ONLY when per-GEMM M is
   small: M < ~5000 rows (then T_copy > T_compute) — MoE expert tails, b=1 short rows, many-expert
   models (128-expert top-1 at M=200k -> ~1.5k rows/expert). Predeclare the tie; claim the small-M
   regime and the simplicity.
T4 THE 2:1 RESOURCE SHIFT (the GB200-specific law). Per GPU, GH200 -> GB200:
     C2C/dir   450 -> ~225 (shared)   [halved]     Grace DRAM  ~500 -> ~250 (shared)  [halved]
     Grace GB  480 -> 240 (shared)    [halved]     Grace cores 72 -> 36 (shared)      [halved]
     HBM       96/144 -> 186          [GROWN]      on-package peer NVLink  none -> 900/dir  [NEW]
   Aggregate C2C demand potential (2 GPUs x 2 dirs x ~225) ~ 900 GB/s vs ~500 GB/s DRAM => Grace
   CAN be ~1.8x oversubscribed — but ONLY in bursts: measured |1 AVERAGE demand is tens of GB/s
   (3 TB / 90 s ~ 33 GB/s). So the collapse is CAPACITY + CORES + BURST-COLLISIONS + stall
   coupling, NOT sustained-BW starvation. Q4 (the dp2 row) MEASURES the real number; do not
   fabricate a 2x. If dp2 scaling is 1.2x, THAT is the motivation number.
T5 NVLINK IS IDLE in every offload baseline (DP: no peer traffic at all; staged/streamed TP: only
   the O(M*H) exchanges ~13 GB/layer-step-scale vs 900 GB/s) = the pair's single largest unused
   resource.
T6 THE COUPLING that defeats the trivial-extension attack: M3 needs (a) HBM slack and (b) pair
   dedup. Slack exists because weights never touch HBM (M1: 0 GB; staged B3: ~2 GB buffers — also
   fine; resident B2: 70 GB — dead; DP-offload B1: has slack BUT its ranks hold DIFFERENT
   micro-batches, so NOTHING dedups — a peer cache under DP degenerates to two private mirrors at
   half capacity in a multi-process world with no shared namespace). Pair-pooled dedup additionally
   requires BIT-IDENTICAL replicated residuals across the pair — a TP-with-replicated-residual
   invariant (our I5 dedup invariant), impossible under DP (different data) and dropped by
   Megatron-SP (shards the residual). That is the lock.
```

## 3. "Why porting the GH200 solution to GB200 fails" — the paper's Figure 1 / Table 1

Figure 1 (imagined; per-design stacked resource bars; red = saturated/OOM, grey = idle):

```text
              per-GPU HALVED resources                     per-GPU GROWN resources
              C2C(~225) DRAM(~250) GraceGB(240) cores(36)  HBM(186GB)        peer NVLink(900)
GH200 |1      [ 50% ]  [ 40% ]   [ 90%  ]    [ 70% ]      [ 55% ]           [   n/a    ]  <- balanced by design
GB200 ports:
 DP-of-|1     [100%!]  [ 90%!]   [ OOM!! ]   [100%!]      [ 55% ]           [   IDLE   ]  <- halved rails saturate
 SuperOffload [ 80% ]  [ 70% ]   [OOM 70B+]  [ 90% ]      [ 60% ]           [   IDLE   ]
 resident TP2 [ 10% ]  [ 20% ]   [ 30%  ]    [ 40% ]      [ OOM @frontier ] [ exch only]  <- grown rail is the wall
 staged TP2   [ 60% ]  [ 50% ]   [ 55%  ]    [ 70% ]      [ 60% ]           [ exch only]  <- B3: good! NVLink idle
 OURS (B5)    [ 45% ]  [ 35% ]   [ 55%  ]    [ 60% ]      [ 95% = cache ]   [ 60% cache]  <- load on the grown rails
(percentages illustrative until Q4 + I-stage runs fill them; the SHAPE is the argument)
```

Table 1 (design space; every baseline has a red cell our composition avoids):

```text
                       weights   host     act traffic      b1 seq   70B@25k    NVLink    2:1
                       in HBM    copies   to Grace         split?   on 1 SC?   used?     aware?
DP-of-|1 (B0)          0         2 (OOM)  1x               no       host-OOM   no        no
SuperOffload (B1)      transit   1/rank*  1x               no       b4 only    no        no (1:1 design)
resident TP2 (B2)      ~70 GB    0        1x               yes      OOM@front  exch      no
staged TP2 (B3, ours)  ~2 GB     1        1x               yes      yes        exch      partially
sTP (B4, ours)         0         1        1x               yes      yes        exch      partially
+M3/M4 (B5, ours)      0         1        1x MINUS cache   yes      yes        CACHE     yes
* ZeRO-3 shards are per-rank OWNED because full-FT updates them; frozen-LoRA is what makes 1 copy possible.
```

The one-sentence design law: **GB200 halves every per-GPU Grace resource while growing HBM and
adding a 900 GB/s peer link; our system is the load-shift that topology forces — weights stop
occupying HBM (streamed tiles), the freed slack becomes a pair-pooled deduplicated checkpoint cache
served over NVLink with Grace as spill, and only the cold tail ever touches the halved C2C/DRAM,
under one global two-GPU DMA schedule.**

---

## 4. The design — mechanisms, each with its anti-trivial-extension defense

```text
M1 IN-KERNEL DISJOINT WEIGHT STREAMING (substrate; = gb200_tp.md I1-I4). Zero weight residency;
   TP-2 shards streamed from ONE pinned arena. NOT claimed as a throughput win over B3 (T3!);
   claimed as (a) the slack-maximizer enabling M3, (b) the small-M/MoE-tail winner, (c) strictly
   simpler than staging (no slab lifecycle). "Add streaming to SuperOffload"? -> needs cpu-right
   tile kernels + single-process TP + LoRA-aware autograd = becomes this system.
M2 ONE FROZEN ARENA (= I2/I5). Full-FT systems CANNOT share a copy (per-rank updates are
   intrinsic); frozen-LoRA-specific design point. shm-retrofit fixes DP's capacity only (T2) —
   we SAY so, and still win on b1-frontier + scaling + M3/M4.
M3 PAIR-POOLED DEDUPLICATED NVLINK CHECKPOINT CACHE (the candidate headline mechanism).
   Fwd: saved tensors/checkpoints written ONCE per pair (bit-identity invariant) into the pair's
   pooled HBM slack (~2 x 70-110 GB measured slack at P1-class rows) via NVLink write-through;
   cold tail spills to Grace. Bwd: LIFO consumption — top-of-stack hits the cache at 900 GB/s
   (private, idle link) instead of ~190 GB/s (shared, contended C2C); the freed H2D lane carries
   ONLY weight re-reads; as cache frees, prefetch deeper checkpoints Grace->cache.
   GATE: >=10% step_s at P1 or DROP (evidence discipline).
   A3 FEASIBILITY ARITHMETIC (done — tempers the plain-cache form): under the |1-style
   recompute-OFF full-fg regime the saved set is ~30 GB/layer x 80 ~ 2.4 TB/step vs a pool of only
   ~190-270 GB -> traffic cut ~ pool/saved ~ 8-12% — BORDERLINE vs the gate. The plain cache is a
   smoothing/prefetch buffer, not a headline. => the headline form is M3' below.
   A1 NOVELTY VERDICT (done): PARTIALLY-NOVEL — composition only. Components are ALL known:
   dedup-across-MP-ranks + CPU spill = ZeRO-R partition_activations+cpu_checkpointing (2019,
   shipped in DeepSpeed); peer-GPU HBM as a training swap tier over NVLink = HUVM (ATC'22) + BPipe
   (ICML'23, parks activations in peers); pooled spare NVLink HBM over a contended host link =
   Aqua/Harvest (inference). Claimable ONLY as: "first training system serving checkpoints from a
   pooled TP-dedup'd cache spanning a GB200 pair's freed HBM with LIFO spill/prefetch — the
   composition, whose CAPACITY is created by zero-weight-residency." Never claim dedup / peer-tier
   / host-spill individually.
M3' RECOMPUTE-PLACEMENT INVERSION (the sharper, GB200-quantified form of M3).
   With per-layer checkpointing the saved set collapses to the residual checkpoints:
   80 x [M,H] bf16 ~ 264 GB at P1 — which FITS the pair's pooled dedup'd slack almost exactly once
   weights are streamed (0 HBM residency). Then activation traffic to Grace ~ ZERO; backward
   recomputes intermediates from HBM-resident checkpoints; the halved shared link carries ONLY
   weight tiles. Everyone else fails this exact config: unsloth-DP b4 = 140 GB resident weights +
   132 GB checkpoints > 186 GB -> OOM; resident TP2 = 70 GB weights + replicated checkpoints ->
   OOM; offload-recompute baselines round-trip the 264 GB x2 through the halved shared link; DP
   cannot dedup (different micro-batches).
   CLAIM — QUALIFIED FORM ONLY (A1 verdict: the unconditional form is CONTRADICTED by public data.
   Poolside measured C2C activation offload BEATING selective recompute by 6-13% ON GB200 — but
   with RESIDENT weights, an otherwise-IDLE C2C link, and MLP-slice-only offload volume.
   POET/Capuchin own the abstract "link bandwidth decides recompute-vs-offload" framework — ours is
   an instance, not a new axis. CITE BOTH AND DELIMIT, or a reviewer does it for us):
     "For STREAMED-WEIGHT fine-tuning at long seq on GB200 — where (i) two GPUs share ~500 GB/s of
     Grace bandwidth and (ii) the C2C link is already consumed by base-weight streaming — the
     offload-everything strategy that is optimal at 1:1 (GH200; and on GB200 with resident weights
     per Poolside) FLIPS: full-layer recompute with checkpoints pinned in pair-pooled HBM
     dominates. We MEASURE the bandwidth-ratio boundary of the flip."
   The flip-boundary figure (step_s vs seq for both configs, crossing point, vs link load) is the
   result EITHER WAY — decisive whichever side wins at P1; overclaiming the unconditional
   inversion is not publishable. Mostly-existing knobs (recompute modes +
   UNSLOTH_GC_OUTER_HBM_EVERY_N) + the pair-pool layer.
   GATE: recomp-on+pooled-HBM beats recomp-off+offload at P1 (step_H in budget) OR the measured
   flip frontier is reported as the result.
   Defenses + ATTACK LEDGER (A1, answer these IN the paper):
     ATTACK 1 "just enable Megatron-SP / ZeRO-R partition_activations+cpu_checkpointing" (the
       strongest): SP-sharding the checkpoints 2x gives 132 GB/GPU — inside slack with NO pooling.
       PARTIALLY LANDS for dense. Our answers: (a) SP shards the residual -> kills the
       bit-identity invariant that no-a2a MoE (I7 identical-topk) and dedup depend on — the MoE
       models are where the paper's capacity story lives; (b) pooling handles asymmetric slack +
       graceful host spill/prefetch (P_a+cpu is static all-or-nothing placement); (c) concede
       capacity-parity for dense TP-2 and, if cheap, RUN the SP-shard variant as a rung.
     ATTACK 2 "retarget saved_tensors_hooks at cuda:1" (~20 lines): lands on the mechanism in
       isolation, NOT the system — naive retargeting double-stores (both ranks push), steals peer
       slack uncoordinated, and contends with TP exchanges on NVLink; the single-namespace
       admission/spill policy is exactly what the one-liner lacks. Say so explicitly.
     ATTACK 3 "Poolside refutes the inversion": lands on the unconditional wording only — their
       link is idle + resident weights; ours is loaded by weight streaming. The delimitation above
       + our contention measurement is the answer; make that measurement the centerpiece.
     vs B1/DP — different micro-batches, nothing dedups, no shared namespace across processes.
     vs B2 — no slack (70 GB weights). vs B3 — OUR rung; B3+M3 is an ablation row we RUN.
M4 GLOBAL DUPLEX DMA SCHEDULE (the 2:1 scheduler). One scheduler owns BOTH GPUs' {d2h, h2d, p2p}
   lanes against the ONE DRAM: phase rules (fwd: D2H=offload, H2D=weights; bwd: restage=NVLink-
   first, H2D=weights+prefetch), a shared token bucket capping concurrent Grace-touching bytes, and
   PHASE-OFFSET execution — deliberately skew the two GPUs' per-layer transfer bursts within the
   slack before each region exchange so DRAM bursts interleave instead of collide. GB200-specific
   by construction (no second GPU exists on a GH200 Grace). Meaningless outside a single-process
   dual-device runtime (DP has no cross-rank DMA scheduler, no phase control). GATE: >=10% or fold
   into "coord" and report honestly.
M5 MoE OWNERLESS-EXPERT STREAMING + EQUAL-COUNT SLICES (kept, honest version). Experts live
   ownerless in the arena; each GPU streams what its balanced slice routes to; zero a2a; skew-
   immunity by construction => the load-balance aux loss / capacity-factor token-drop become
   UNNECESSARY for frozen+LoRA (C7 corollary). Honesty: at b8 flagship all experts are hit (no
   traffic win); the wins are imbalance-robustness, aux-loss-free quality, the small-M tile
   advantage (T3), and experts-beyond-HBM capacity. Prior-art fences per gb200_angle.md.
AMP (demoted, quality-gated): NVFP4 streamed arena — Blackwell-only native-FP4 compute, ~3.5x fewer
   streamed bytes; SFT quality UNPROVEN -> small probe before any claim. NVL72 = scale projection.
```

## 5. Claims table (each claim names the row and the gate that must back it)

```text
CLAIM                                                        BACKED BY            GATE
GB200 naive ports collapse (capacity/cores/bursts)           B0 vs |1 (Q4)        measured, whatever it is
frontier: largest model+seq LoRA on ONE superchip            B5 vs B1/B2 at b1    >=1.5x mem pace car (P5)
throughput: beats every SHIPPING system at flagship          B5 vs B1 (+B2)       step_s < b4 pace cars (P1-P4)
mechanism: M3' inversion (recomp-on+pooled-HBM vs recomp-off) B5 config A/B       beats P1 or report the flip-frontier
mechanism: M3 plain cache delta (if kept at all)             B5 vs B4; B3+M3 row  >=10% step_s or DROP (arith says ~8-12%)
mechanism: M4 schedule delta                                 on/off ablation      >=10% or fold into coord
streamed-vs-staged honesty                                   B4 vs B3             report the tie; claim small-M/MoE
MoE: aux-loss-free + skew-robust + beyond-HBM experts        I7 rows + C7         parity + imbalance sweep
scaling: 1 -> 2 GPUs                                         B5 vs |1 vs B0       >=1.6x where B0 <=1.2x
```

## 6. What we do NOT claim (reviewer-bait list)

```text
- "tile-streaming beats staging" at dense large-M (it does not; T3 — we predeclare the tie).
- "one copy" as THE novelty (capacity-retrofittable via shm; frozen-specific anyway).
- "first to stream weights from Grace" (SuperOffload/MegaTrain), "first no-a2a MoE" (Hecate/Janus),
  "first host experts in training" (ES-MoE), FP4/quantization as novelty (QLoRA/Hopper-int4).
- any raw C2C-bandwidth superiority of tile streaming (SuperOffload Fig 7 kills it).
- "checkpoint dedup across TP ranks" as novel (ZeRO-R partition_activations + cpu_checkpointing,
  2019); "peer-GPU HBM as a training tier" as novel (HUVM ATC'22; BPipe ICML'23); "pooled spare
  NVLink HBM over a contended host link" as novel (Aqua/Harvest, inference).
- the UNCONDITIONAL "GB200 inverts recompute-vs-offload" (Poolside's public GB200 measurement shows
  offload winning with resident weights + idle link; only the streamed-weight qualified form holds).
- "recompute-vs-offload depends on bandwidth" as a new insight (POET/Capuchin/SPPO/MEMO/ProTrain).

PRIOR-ART RECEIPTS added by A1 (cite all): ZeRO-R (arXiv:1910.02054, DeepSpeed activation
checkpointing docs), HUVM/memHarvester (ATC'22), BPipe (ICML'23), Aqua (arXiv:2407.21255), Harvest
(arXiv:2602.00328), Poolside "C2C Activation Offloading on Grace Blackwell" (poolside.ai blog),
POET (ICML'22), Capuchin (ASPLOS'20), Stronghold (SC'22), ZeRO-Infinity, TransformerEngine CPU
offload, Megatron fine-grained act offload, Unsloth offloaded GC, SSDTrain/TBA, MLP-Offload (SC'25),
SPPO (2503.10377), MEMO (2407.12117), ProTrain (2406.08334), PipeOffload (2503.01328), Megatron-SP
(2205.05198), NeMo/Megatron-Bridge perf guides (vendor: GB200 offload-friendly guidance).
```

## 7. Convergence checklist (iterate until all closed)

```text
[x] A1 PRIOR-ART CHECK — DONE (2026-07-05): M3 = PARTIALLY-NOVEL (composition only; components all
      known: ZeRO-R P_a dedup+cpu-spill, HUVM/BPipe peer tiers, Aqua/Harvest pooling). M3' =
      PARTIALLY-NOVEL and the unconditional form REFUTED by Poolside's GB200 datapoint -> survives
      only as the QUALIFIED streamed-weight flip-boundary claim (wording fixed above). Attack
      ledger (SP/P_a; saved_tensors_hooks-to-cuda:1; Poolside) recorded in M3' — answer in-paper.
[ ] A2 Q4 MEASUREMENT: B0 (dp2) + |1 reference at P1 workload -> the real collapse number (sizes
      T4's claim). Cheap: I0's run_dp2_pair.sh, no new code.
[x] A3 M3 FEASIBILITY — DONE (2026-07-05, arithmetic in M3/M3' above): plain cache ~8-12% traffic
      cut at P1 (borderline) -> demoted to smoothing; the RECOMPUTE-INVERSION form (M3') is the
      quantified headline candidate: 264 GB checkpoint set ~= pair pooled slack -> ~zero Grace act
      traffic, a config every baseline OOMs on or round-trips. Verify empirically at I-stage.
[ ] A4 fold the surviving story into gb200_tp.md: positioning rewrite to the v3 spine; NEW stage
      for D2 (upgrade of I7: shared-queue cooperative kernel; static E/2 split becomes the
      ABLATION rung); stage(s) for D3 (M3' pooled-checkpoint config + relay-restage + M4 knobs);
      the B3+M3 ablation row. Stages I0-I6 untouched as substrate.
[ ] A7 D2 EMPIRICAL GATES (predeclared): beats (i) static-split I7, (ii) FEPLB-style host token
      rebalancing, (iii) host-orchestrated dispatch — at real fine-tuning skew AND synthetic
      worst-case skew; atomic-contention overhead < 2% of layer time (hierarchical queue);
      grad parity vs static split (determinism fix verified).
[~] A5 trivial-extension pass: A1's attack ledger covers M3/M3' (the load-bearing ones); rerun a
      final whole-composition pass ONCE Q4/A2 numbers exist (attacks may shift with the data).
[ ] A6 DENSE-vs-MoE split decision: for DENSE TP-2 concede SP-shard capacity-parity (Attack 1) and
      optionally run the SP-shard variant as a rung; keep replicated-residual + pooled dedup as THE
      design for MoE (no-a2a needs it) — the MoE models carry the capacity story.
```

## 8. Bottom line (v2.0)

Best baseline: **one-copy staged TP-2 (B3) — nothing ships it; we build it and treat it as the
strongest baseline/ablation, with SuperOffload + resident-TP as the shipping-SOTA external rows.**
The GB200 story: **the superchip halves every per-GPU Grace resource and grows HBM + adds an idle
900 GB/s peer link; our system is the load-shift that topology forces — streamed weights (zero HBM
residency) free exactly the slack that lets the pair pin its DEDUPLICATED checkpoint set in pooled
HBM (M3': the 2:1 topology INVERTS the recompute-vs-offload optimum, and only a streamed-weight
system can act on it), with NVLink as the pool fabric, Grace as spill, one dual-GPU duplex/
phase-offset DMA schedule (M4) — plus ownerless, aux-loss-free MoE (M5).** Every mechanism carries
an ablation gate and an explicit anti-retrofit defense; the known ties (streamed~staged at large M;
one-copy capacity retrofit; plain-cache ~8-12%) are predeclared instead of being discovered by
reviewers.
