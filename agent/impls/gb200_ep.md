# GB200 sEP: Streamed Expert Parallelism (cooperative work-splitting) — Balanced BY CONSTRUCTION (`ASYM_EP_MODE=sep`)

Companion to `agent/impls/gb200_tp.md` (the sTP substrate this track RIDES ON — I4 replicated
residual + shared arena + I7 static EP-2 are PREREQUISITES). Paper-story D2 HEADLINE:
"skew-adaptive cooperative MoE execution" — this doc is its implementation plan.
Style/discipline mirror `gb200_tp.md`/`gb200_dp.md`: staged, gated, one experiment at a time,
fresh artifacts, never advance on inconclusive, predeclared softening rules.

## Goal (ULTRA CLEAR — read before any change)

```text
WHAT (the headline, sEP-v2): EP that KEEPS EP's invariants — each GPU processes ITS OWN
    sequences (sharded batches, global = 2xb) — but replaces expert OWNERSHIP with ownerless
    work-splitting over a shared coherent host arena:
      dense parts: pure per-device DP (zero comm; single process drives both GPUs);
      MoE phase: both ranks' packed expert inputs land in ONE shared pinned host pool (the
        D2H the fine-grained design ALREADY pays for dA); a shared work list of
        (expert, token-block) tiles in cache-coherent host memory is consumed by persistent
        kernels on BOTH GPUs (front/back counters, chunked atomic pops);
      AFFINITY ORDERING: each GPU's queue section starts with its OWN tokens' items ->
        foreign-token fetches + output returns (NVLink P2P into the owner's y buffer) occur
        ONLY for STOLEN items. Comm is PROPORTIONAL TO THE IMBALANCE REPAIRED, zero when
        balanced — vs classic EP's unconditional all-to-all.
    Routing is UNTOUCHED (pure infra); balance floor = 1/2 by construction (assignment
    schemes floor at the hottest expert's share).
TWO DEPLOYMENT SHAPES OF ONE MECHANISM (shapes-per-metric; neither replaces the other):
    sEP-v1 (sTP-shaped, REPLICATED tokens) = the LONG-SEQ shape: one sequence in flight,
      per-device activations ~halved, host acts 1x -> carries the MoE FRONTIER rows
      (P4 s80000, b1 ladders). Also the mechanism-isolation rung (zero steal traffic).
    sEP-v2 (DP-shaped, SHARDED batches) = the EP-INVARIANT throughput shape: different
      sequences per GPU, global 2xb -> carries the balance/tokens-per-sec rows. Its cost is
      structural: 2x total activations, per-sequence max seq ~= |1's — NEVER present v2 on a
      max-seq row (that is v1's job); never present v1 as the EP headline (that is v2's).
WHY ONLY US / ONLY HERE: (a) weights STREAM per tile from the shared arena — any GPU can
    compute any expert with zero migration/replication (EPLB/FasterMoE must copy hot experts
    into HBM); (b) the expert inputs are ALREADY host-resident in this design (x_cpu for dA)
    — the shared token pool is a layout change, not new traffic; (c) GB200 C2C coherence
    gives system-scope atomics on host memory to both GPUs + CPU (PCIe cannot; GH200 has one
    GPU per Grace; multi-process NCCL stacks have no shared queue and owned resident experts).
DATA-LAYOUT CORRECTION (user 2026-07-07 — critical for baseline validity):
  REAL/deployed EP (Megatron, DeepSpeed) is laid on top of DP: each GPU holds a DIFFERENT
  batch shard, experts are OWNED, tokens ALL-TO-ALL to their expert's owner (token on GPU0
  needing GPU1's expert is shipped to GPU1 and back). So the REPRESENTATIVE vanilla-EP
  baseline MUST be sharded-batch + owned + a2a — NOT the replicated-batch shape.
  The replicated-batch static-EP we first built (same batch both GPUs, owned E/2, no a2a) is
  numerically correct BUT is a SUBSTRATE-COMPOSITION ABLATION (it drops onto the sTP
  replicated residual), not the real deployment shape. Demoted accordingly below.

MODES (ASYM_EP_MODE), grouped by DATA LAYOUT:
  REPLICATED-batch shapes (sTP substrate; ABLATION lane — same tokens on both GPUs):
    static_rep = per-device E/2, owned (was 'static')     -> substrate ablation ONLY
    sep1       = queue over replicated tokens              -> mechanism-isolation ablation
  SHARDED-batch shapes (real EP layout; THE JUDGMENT LANE — different batch per GPU):
    static     = owned E/2 + all-to-all dispatch           -> THE REAL vanilla-EP BASELINE
    hostsplit  = per-step exact-optimal host split (sharded)-> strongest scheduler baseline
    sep        = ownerless queue + shared host token pool   -> THE HEADLINE (sEP-v2)
paper names: EP-Static (sharded, the real baseline), EP-HostSplit, AsymLoRA-sEP;
  static_rep/sep1 reported ONLY as substrate/mechanism ablations, never as the EP baseline.
KEY DIFFERENCE the headline turns on: real vanilla EP pays UNCONDITIONAL all-to-all (every
  token to its expert-owner, every step) AND straggles under skew (owner of hot experts is
  overloaded); sEP-v2 has NO owner and NO a2a — a tile's tokens are pulled from the shared
  host pool by whichever GPU grabs it, so cross-GPU traffic is PROPORTIONAL TO IMBALANCE
  REPAIRED (zero when balanced) and no device can straggle.
pair 0,1 only. EXACT configs: next section.
```

## STATUS (2026-07-06 — nothing built; read interlocks)

```text
IN PROGRESS (2026-07-06 session): E0 VALIDATED (guards/tags/skew-knob/histograms);
E3 KERNEL + PROBE VALIDATED at prod-scale M (queue imb <=4% at all alpha; static/queue
4.24x/6.94x/8.58x; balanced overhead 0.969; bitwise-exact; EG3 per-launch tie with
hostsplit hit as predeclared); G-E1.2 natural skew MEASURED on real data (|1): static-E/2
device share mean 0.530 / worst layer-step 0.631, temporally volatile; I7-CORE block-level
parity PASS (stp_moe.py slicing over shared pinned banks). REMAINING: queue-kernel e2e wiring
(sep1), E4 sweeps, E5 sEP-v2, scout (model not cached). Original prerequisites below.
HARD PREREQUISITES (in order):
  1. tp.md I5 (dedup) — RECOMMENDED first: MoE rows inherit the same duplicated [M,H]-class
     host traffic; skew wins would be diluted by a fixable constant.
  2. tp.md I7 (static EP-2 + DispatchFn/combine + ASYM_STP_MOE unlock) — REQUIRED: sEP swaps
     ONLY the grouped-GEMM launch inside I7's structure; Dispatch/combine/LoRA plumbing is I7's.
  3. This doc's E0-E6 (E5 = the v2 dual-batch/dense-DP + shared-pool stage; the queue
     kernel from E3/v1 is REUSED unchanged — v2 is plumbing above it, not new kernel work).
INTERLOCK: I7's "EP fallback: a starved device may stream ANY expert's shard" risk-note is
  SUBSUMED by this track — do not build that fallback inside I7; land I7 static-only.
```

## Dev Workloads & Baselines-To-Beat (EXACT configs — copy-paste into RUNS)

```text
DEV RULES: ONE primary MoE model q3-30b-a3b (48L, H=2048, I=768, E=128 top8, ker101 — the
LARGE-E averaging case) + ONE secondary llama4-scout (48L, H=5120, E=16, shared expert, ker000
— the LOW-E skew-prone case; exercises the shared-expert path). Modest dev seq (s20000 / parity
s2048); the P4 target row (80000) and 235B flagship are PAPER PHASE. Host RSS under the
watchdog floor at ALL times (HC2). Timing-only skew rows are LOSS-INVALID and labeled so.

# ============ OURS (the ladder; ONE knob apart; b8 = global 8, sTP shape) ============
ASYM_STP_MOE=1 ASYM_EP_MODE=sep \
  q3-30b-a3b|2 ; asym_stp_cpuadamwds|recomp-off-full-fg-ker101|ligerloss1 ; 20000|8|1 ; none|false|false|false|false|false
ASYM_STP_MOE=1 ASYM_EP_MODE=hostsplit  (same row)
ASYM_STP_MOE=1 ASYM_EP_MODE=static     (same row)          # = tp.md I7 rung, the attribution row
# parity/loss-gate row (grad dumps step 2, logits s2048-only):
ASYM_STP_MOE=1 ASYM_EP_MODE=sep \
  q3-30b-a3b|2 ; asym_stp_cpuadamwds|recomp-off-full-fg-ker101|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false
# scout parity + natural-skew row (low E; shared expert):
ASYM_STP_MOE=1 ASYM_EP_MODE=sep \
  llama4-scout|2 ; asym_stp_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 9500|8|1 ; none|false|false|false|false|false

# ============ THE HEADLINE ROW (sEP-v2: sharded batches, b4/GPU = global 8) ============
ASYM_STP_MOE=1 ASYM_EP_MODE=sep \
  q3-30b-a3b|2 ; asym_stp_cpuadamwds|recomp-off-full-fg-ker101|ligerloss1 ; 20000|4|1 ; none|false|false|false|false|false
# (per-device batch 4, DIFFERENT sequences per device, global 8 — the EP-invariant row;
#  compare against static/hostsplit at the SAME shape and the sep1 ablation at sTP shape)

# ============ SYNTHETIC-SKEW TIMING ROWS (loss-invalid; timing/balance evidence ONLY) ======
# ASYM_EP_SKEW_HOT=<alpha> forces a fraction alpha of routed slots onto expert 0 (router
# override AFTER detach; forward values garbage, kernel work REAL). Sweep per mode:
ASYM_EP_SKEW_HOT=0.25|0.50|0.75  x  ASYM_EP_MODE=static|hostsplit|sep   (q3-30b-a3b 20000|8|1)

# ============ SYSTEM-LEVEL BASELINE (shipping SOTA; NO EP mechanism at all) ============
# superoffload/zero3 run MoE models with experts as plain offloaded weights on EVERY rank:
q3-30b-a3b|2 ; superoffload_mem|unsloth-off|ligerloss1 ; 20000|4|1 ; none|false|false|false|false|false

# ============ |1 REFERENCE (scaling frame) ============
q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker101|ligerloss1 ; 20000|8|1 ; none|false|false|false|false|false

# BEAT relation (dev) = EG gates below. PAPER PHASE: P4 (80000|8|1), scout 9500 full matrix,
# b1/boundary frontier rows, Qwen3-235B-A22B flagship (arena spans both Graces per the paper-story doc).
```

## THE BALANCE GATE (headline; with predeclared expectation math + softening rules)

```text
EG1 BY-CONSTRUCTION BALANCE: at synthetic hot-expert skew alpha in {0.25, 0.50, 0.75}:
      per-device expert-GEMM busy-time imbalance |b0-b1|/max(b0,b1):
      sep <= 5%   REGARDLESS of alpha;
      static shows the assignment floor (predicted below);
      hostsplit <= 5% (PREDECLARED TIE on balance — see EG3 for how sEP must beat it).
EG2 SKEW SPEEDUP vs static (the attribution number): expert-GEMM segment wall ratio
      static/sep ~ (1 + alpha') where alpha' = hot share ABOVE the balanced half
      (dev0 load (1+alpha)/2 vs sep 1/2). At alpha=0.5 => ~1.5x on the MoE segment; scale by
      the MoE fraction of step_s for the e2e number. If natural-skew (real data) device
      imbalance < ~10% on q3-30b-a3b (E=128 CLT averaging makes this LIKELY — predeclared),
      the e2e natural-skew win is small: REPORT IT HONESTLY and lead with (a) worst-case
      robustness (synthetic sweep + scout E=16) and (b) the low-E models where skew is structural.
EG3 vs HOSTSPLIT (the honest fight — hostsplit also reaches the balance floor since it may
      split within an expert): sEP must win on AT LEAST ONE of (predeclare, measure all):
      (i) zero per-step replan cost on the critical path (hostsplit re-plans + re-launches
          per layer per step; measure at SMALL per-layer M where launch/replan dominates);
      (ii) robustness to runtime variance (tile time is NOT proportional to tokens: k-dim
          tails, C2C jitter, allocator/stream noise) — measure busy-time spread;
      (iii) launch count / CPU-side overhead (1 persistent launch vs 2xG grouped launches).
      IF sEP ties hostsplit everywhere: the paper claim NARROWS to "balanced-by-construction
      WITHOUT host orchestration" and hostsplit ships as the recommended mode — both rungs are
      OURS, the by-construction story survives either way (this is the safety of owning the
      whole ladder).
EG-V2 EP-INVARIANT GATE (headline rows are v2-only): sharded batches proven (per-rank
      sample ids disjoint, union == the DP reference set — the dp.md G-D2.1 shard receipt);
      steal-traffic accounting: foreign-token H2D + P2P output-return bytes ~ stolen-item
      fraction (ZERO at balanced routing) — THE anti-a2a receipt vs classic EP's
      unconditional token movement; dense segments show ZERO cross-device comm.
EG4 NO-REGRESSION: at BALANCED routing, sep step_s <= 1.02x static (the queue must be free
      when not needed); atomic/queue overhead < 2% of MoE-layer time (paper-story bar).
EG5 CORRECTNESS: route bits identical across modes; loss band vs static; adapter-grad parity
      under the I4 measured-envelope method (fixed 1e-2 is unsatisfiable at depth — see tp.md
      2026-07-06 envelope entry); zero dropped tokens by construction (assert).
STATIC-FLOOR MATH (predeclared): with hot share alpha on ONE expert owned by dev0 and the
      remaining (1-alpha) spread evenly: dev0 ~ alpha + (1-alpha)/2 = (1+alpha)/2, dev1 ~
      (1-alpha)/2 -> imbalance floor = alpha/(1+alpha) ... 2x wall at alpha->1. NO ownership
      assignment can beat max_e share_e as its floor; work-splitting's floor is 1/2. THIS
      inequality is the "by construction" sentence of the paper.
```

## HARD CONSTRAINTS (inherit gb200_tp.md HC1-HC3 + EP-specific)

```text
HC1-HC3 inherited verbatim (both-node membind; oom_score_adj + watchdog; launcher-only).
HC-EP1 SKEW ROWS ARE TIMING-ONLY: ASYM_EP_SKEW_HOT != 0 forces routing AFTER detach; forward
       values are garbage. Artifacts carry a `skewXX` tag + `loss_invalid=true` in
       profile.json.config; such rows may NEVER appear in a loss/quality table.
HC-EP2 the persistent kernel must NOT cluster-launch (num_multicast==1 asserted — the static
       LaunchAttrHandle in handle.hpp:156 is thread-shared; tp.md I1 note) and must NOT
       require cooperative launch (items are independent; no grid-wide barrier -> plain
       launch, grid = #SMs, pop-until-empty).
HC-EP3 queue counters live in PINNED HOST memory allocated ONCE at setup (tiny); system-scope
       atomics only (atomicAdd_system / atomicSub_system); NO device-mapped writes into
       another GPU's HBM (keep NVLink for the existing exchanges only).
HC-EP4 ASYM_EP_MODE != static requires ASYM_STP=1 + ASYM_STP_MOE=1 + an MoE target model;
       sep/hostsplit die at the launcher until their stage lands (mislabeled-artifact guard,
       same pattern as the I0 arena/coord locks).
```

## Why This Design Is Correct & Efficient (derived from the landed substrate)

```text
CORRECT — every ingredient is already proven in-tree:
- TOKENS: under I4 sTP the residual is replicated and BIT-IDENTICAL on both devices, so the
  packed expert input (pack_tokens_contiguous, moe.py:758) built on each device is IDENTICAL.
  Dynamic (expert, m-block) assignment therefore needs ZERO token movement: each device
  already holds every operand row it could ever be assigned.
- OUTPUTS ARE DISJOINT: each work item writes its own m-block rows of its expert's output.
  Each device scatter-adds ONLY its computed items into its local y_i; the EXISTING I7
  combine (AllReduce2Fn at scatter_contiguous, moe.py:804) sums the two partials -> the
  union is complete and each token counted once. Forward is EXACTLY the static result
  regardless of which GPU computed which tile (same tiles, same math, disjoint writes).
- WEIGHTS: both devices stream tiles from the SAME full [E,N,K] pinned bank with the SAME
  full-bank TMA descriptor — NO E/2 slicing, NO local-id remap (SIMPLER than static EP-2).
  Dual-lane reads of one pinned buffer are proven legal + fast (I1 probe: 174.7 GB/s/lane
  concurrently on the same buffer).
- LoRA/grads: dB/dA accumulate per device over the items it executed (partials), then the
  tiny allreduce2 (I7's existing pattern) sums them. Accumulation ORDER varies with pop
  order -> grads judged under the I4 measured-envelope method (EG5), with an optional
  determinism mode (fixed per-expert slot reduction) if a gate demands it.
EFFICIENT — each choice kills a specific cost:
- FRONT/BACK COUNTER PAIR (the whole scheduling design): items sorted by (expert, m-block);
  GPU0 pops from the FRONT (atomicAdd on head), GPU1 from the BACK (head/tail cross =>
  done). Cold experts are consumed by ONE device (full weight-tile reuse, zero duplicate
  C2C fetch); ONLY the expert at the meeting point splits — and under single-hot-expert
  skew the meeting expert IS the hot one, split exactly as needed. Affinity + minimal
  duplicate-fetch fall out of the data structure; no heuristics. (Answers the paper-story attack
  (b): duplicate weight fetches are bounded to the split expert's K-loop.)
- CHUNKED POPS: each pop claims CHUNK consecutive m-blocks of one expert (amortizes the
  ~C2C-latency system atomic over ~10-50us of tile compute; answers attack (a) contention —
  target < 2% of layer time, EG4). Hierarchical refinement (per-GPU batch counter) only if
  the measured contention demands it — do not build preemptively.
- PERSISTENT LOOP: converts asymScheduler.cuh's blockIdx enumeration (expert_id=blockIdx.y,
  n_idx=blockIdx.x*BLOCK_N+shape_n*expert_id, m-range from offsets pairs — :56-58, :88-91)
  into a pop-loop over the SAME (expert, m-block, n-block) tuple space. One launch per
  device per MoE GEMM phase vs hostsplit's per-step re-plan + re-launch.
- ZERO token all-to-all and zero dropped tokens — systems effects of replication +
  work-splitting; routing itself is NEVER modified (infra-only scope).
```

## Hardware Facts (queue-relevant; re-verify in E3)

```text
C2C coherence: system-scope atomics on pinned host memory are architecturally supported on
  GB200 (CPU+both GPUs coherent). MEASURE in the E3 probe: atomic round-trip latency from
  each GPU (expect ~0.5-2us), sustained pop throughput 2 GPUs hammering one counter, and
  the chunk size where contention < 2% of tile time. No public microbench — ours is the number.
Streaming: both lanes reading ONE pinned bank: 174.7 GB/s/lane measured (I1). Split-expert
  duplicate fetches ride the same lanes; bound = the split expert's weight bytes x1 extra.
NVLink: 778 GB/s/dir measured — the combine/allreduce2 path, unchanged from I4/I7.
```

## Verified Code Map (the real anchors; checked against the working tree 2026-07-06)

```text
SCHEDULER (the surgery site): asym_gemm/include/asym_gemm/common/asymScheduler.cuh
  :54-58  offsets = per-expert (start,end) M pairs; m-block range derived per expert
  :61,:71 expert_id / n_idx fields
  :88-91  MGroupedMasked branch: expert_id = blockIdx.y; n_idx = blockIdx.x*BLOCK_N +
          shape_n*expert_id — the (expert, m, n) tuple space the persistent loop re-enumerates.
GROUPED KERNEL LAUNCH: gemm.hpp:506 m_grouped_bf16_asym_gemm_nt_contiguous(a, b=[G,N,K], d,
  offsets_i32[2G], experts_i32[G+1, -1 sentinel], list_size, ...); python wrappers
  frozen_linear.py:707 (_asym_bf16_nt, single-group) / :732 (_asym_grouped_bf16_nt, launch :753).
KERNEL RUNTIME: kernel_runtime.hpp LaunchRuntime::launch uses the CURRENT device's stream —
  per-device persistent launches ride the existing FIX A runtime-API path (handle.hpp:52-108);
  per-device kernel handles materialize on demand (I1-proven).
HOST-SIDE DRAIN TO KILL FIRST: validate_group_plan does offsets/experts .to(CPU) per grouped
  call (exp_act_offload_kernels.cu:89-90; call sites :365/:437/:512) — under sep the item
  list is BUILT host-side anyway, so pass host metadata down and skip the device round-trip.
MoE PRODUCTION PATHS (I7's insertion points, reused unchanged):
  qwen3_moe.py:2556 _forward_qwen3_moe_finegrained_offload (carries offsets/experts/
    token_indices); :3097-3107 AsymQwen3MoeBlock.forward (out = self.experts(...) at :3106);
    router detach :3085-3086; :2509-2543 _ensure_qwen3_moe_finegrained_bases (fused [E,2I,H]
    -> split [E,I,H] copies; the I2 arena-awareness fix applies BEFORE any EP mode).
  llama4_moe.py:292-303 (shared_expert :300 — rides the DENSE sTP path per I7's rule;
    forward_input_scaled :302; combine :303); llama4_experts.py:117 _rebuild_packed_x_cpu.
  moe.py: pack_tokens_contiguous :758 (per-device identical pack under replication),
    scatter_contiguous :804 / _ScatterContiguousRouterNoGrad :771 (the combine site).
sTP SUBSTRATE (landed): stp_runtime.py (streams, allreduce2/allreduce2_out,
  bcast01_from_host), stp_functions.py (AllReduce2Fn/TPRegionFn/Bcast01Fn/Join01Fn; I7 adds
  DispatchFn), stp_wrap.py (StpDecoderLayer two-branch; MoE branches slot into the same shape).
HARNESS: ASYM_EP_MODE knob + tags land next to the ASYM_STP_* derivation in
  profile_lora_lf_test_source.sh (the stp_tag block) + run_lf_lora_sft.sh guards (HC-EP4);
  expert_token_distribution + per-step max/mean tokens-per-device already planned by I7's
  gate — E1 extends with busy-time counters.
```

## Contribution -> Evidence Map

```text
C-EP1 by-construction balance (assignment floor vs split floor): EG1 synthetic sweep,
      static-vs-sep busy-time curves vs alpha.                              [E3, E4]
C-EP2 skew-invariant throughput: EG2 ratios at alpha sweep + scout natural skew. [E4]
C-EP3 the queue is FREE when balanced: EG4 <=2% overhead row.                 [E3]
C-EP4 minimal duplicate weight traffic: split-expert extra C2C bytes measured ~= that
      expert's bank bytes (front/back design receipt).                       [E3, E4]
C-EP5 routing-untouched receipt (pure infra): route bits identical across modes, zero
      dropped tokens asserted — we balance compute PLACEMENT, never token ASSIGNMENT. [E4]
C-EP6 ownerless-streaming uniqueness: baselines' mechanism-absence receipts (owned resident
      experts, no coherent host atomics, no shared queue) recorded file:line. [E5]
```

## Baselines (reviewer-safe-but-beatable — the extensive reasoning, honestly)

```text
PRINCIPLE (from the dp/tp panels): equal GPUs, each system's best mechanism, run-as-is where
it exists, mechanism-correct-but-modest where we must build it, cited-with-receipt where no
LoRA artifact exists. For MEMORY/BALANCE claims a mechanism-correct baseline cannot be
attacked as a strawman (balance behavior is decided by the MECHANISM, not kernel tuning).

TIER 1 (run-as-is, exists today): superoffload_mem / zero3_offload_mem on the MoE models.
  These do NO expert parallelism (every rank offloads ALL experts; MoE = big dense-ish
  offload). They anchor the system-level table (step_s/step_H/RSS) and the "shipping SOTA
  has no EP mechanism for this regime" line. BEATABLE: same axes as the dense panel PLUS
  they pay full expert-bank host bytes per rank (2x)... and they are NOT balance-relevant
  (no EP) — never present them as an EP baseline, only as the system row.
TIER 2 (in-stack rungs, OURS, one knob apart — the attribution ladder):
  EP-Static (SHARDED — the REAL vanilla-EP baseline, per the data-layout correction above):
    different batch per GPU + owned E/2 + all-to-all token dispatch to expert-owners. This
    is what Megatron/DeepSpeed actually deploy; it is THE load-bearing comparison for the
    headline. BEATABLE ON TWO AXES: (1) unconditional a2a every step (sEP-v2 pays steal
    traffic only ~ imbalance, zero when balanced); (2) owner-of-hot-experts straggles under
    skew (EG2 math). Same kernels/offload/LoRA, only assignment+dispatch differ =>
    attribution airtight. BUILD alongside v2 (a2a dispatch = the one extra mechanism vs the
    replicated ablation; DispatchFn already exists in the sTP substrate).
  EP-Static-rep + sep1 (REPLICATED ablations, NOT the EP baseline): same-batch-both-GPUs
    variants used to isolate the substrate (static_rep) and the queue mechanism (sep1) from
    the sharded-batch plumbing. Reported as ablations only; the paper's "vanilla EP" number
    is ALWAYS the sharded EP-Static above.
  EP-HostSplit (E2): the STRONGEST scheduler-class baseline (FEPLB/ES-MoE-style, upgraded:
    per-step EXACT optimal split — counts are known host-side before launch, and it may
    split within an expert via duplicated expert ids in the metadata, which the grouped
    kernel already accepts). WHY BUILD THE STRONGEST VERSION OURSELVES: a reviewer's first
    attack on the queue is "the host already knows the counts — just split on the host";
    pre-empting it with the exact-optimal version (not a weakened one) makes EG3 honest.
    PREDECLARED RISK: hostsplit may TIE sep on balance AND on step_s at large M — the
    softening rule in EG3 keeps the paper safe (both rungs are ours; the claim narrows,
    never breaks).
TIER 3 (external EP trainers — cited with receipts, per dp.md's Ulysses discipline):
  DeepSpeed-MoE / Tutel / Megatron-EP: multi-process a2a EP with OWNED, HBM-RESIDENT
  experts; no LoRA-SFT path for our targets in-stack (record file:line receipts in E5
  BEFORE any table row, same rule as the Ulysses receipt). Their MECHANISM is absent for
  host-streamed experts (nothing to run even in principle: their expert placement IS the
  ownership we dissolve). If a reviewer demands a measured a2a-EP row: the honest scope is
  a resident-experts run on a model whose banks fit HBM — a paper-phase decision, not dev.
SCOPE EXCLUSION (one line, related-work only): aux-loss / capacity-factor balancing is a
  MODELING technique (changes token assignment); this track is INFRA (compute PLACEMENT
  only; router frozen/detached — tp.md invariant). Not a baseline, not a claim.
WHY THIS SET IS SAFE: every mechanism class is represented (none "conveniently missing"),
  the strongest in-class versions are OURS (can't strawman ourselves), external absences
  carry receipts, and every predicted tie has a predeclared narrowing rule instead of a
  post-hoc excuse.
```

## Evidence Discipline

Same as `gb200_tp.md`. One experiment at a time; new `OUTPUT_ROOT` per stage
(`profiling_gb200ep_e<N>`); pre-run declaration {model, pair, mode, skew, workload, artifact
tag, comparison row, likely failure}; after: command.txt (ASYM_EP_* echoed), per-device
step_H, busy-time counters, expert histograms, loss band (non-skew rows only), watchdog
sentinels. Skew rows carry `loss_invalid=true`. Labels: `validated | blocked_by_stage_bug |
inconclusive_wrong_config | inconclusive_partial_profile | inconclusive_stale_artifact |
inconclusive_unexpected_path`. Never advance on inconclusive.

## Stage E0 — Harness Knobs + Balance Instrumentation (no kernel change)

**Objective.** `ASYM_EP_MODE` plumbed + guarded; per-device busy-time + expert-histogram
evidence emitted; synthetic-skew injection knob (timing-only) behind loud guards.

**Files & functions:**

```text
scripts/lf/profile_lora_lf_test_source.sh   stp-tag block: append _ep<mode>[_skewNN] to the
    artifact tag; die on sep/hostsplit pre-E2/E3 (HC-EP4 locks, mirrors arena/coord locks).
scripts/lf/run_lf_lora_sft.sh               guards: ASYM_EP_MODE != static requires ASYM_STP=1
    + ASYM_STP_MOE=1 + MoE model; ASYM_EP_SKEW_HOT requires ASYM_EP_SKEW_ACK=1 and stamps
    loss_invalid into the config env.
asym_gemm/training/qwen3_moe.py / llama4_moe.py   skew injection at the DETACHED top-k site
    (:3085-3086 / :287-289): override a fraction of top_k_index slots to expert 0 AFTER
    detach; counters for per-expert tokens per layer per step (extend the existing
    expert_token_distribution emission).
scripts/lf/run_lf_profiled_train.py         emit per-device MoE busy-time (CUDA events around
    the expert-GEMM segment per branch) + histograms into profile.json (ep_balance block).
```

**Validation gate.** Dry-runs show the tag + guards die correctly (positive + negative, I0
pattern); a static-mode s2048 run emits ep_balance with sane histograms; skew knob without ACK
dies; with ACK the histogram shows the forced alpha.

## Stage E1 — Static EP-2 Baseline Rows (consumes tp.md I7; no new mechanism)

**Objective.** Land the attribution rung's numbers: parity (I7's own gates) + natural-skew
measurement + the static floor under synthetic skew.

**Runs.** q3-30b-a3b parity s2048 (I7 command) then: 20000|8|1 static natural-skew row;
skew sweep alpha={0.25,0.50,0.75} static; scout 9500|8|1 static. All with E0 instrumentation.

**Validation gate.**

```text
G-E1.1 I7 parity gates pass (route bits identical both devices; adapter-grad envelope; loss band).
G-E1.2 natural-skew device imbalance MEASURED on real data for both models (predeclared
       expectation: q3-30b-a3b small [E=128 averaging], scout larger [E=16]) — this number
       decides how much of the paper leans on synthetic vs natural rows (EG2 honesty).
G-E1.3 static floor curve matches the (1+alpha)/2 prediction within ~10% (validates the
       instrumentation before it judges sEP).
```

## Stage E2 — EP-HostSplit (the strongest scheduler rung)

**Objective.** Per-step exact-optimal host split: greedy LPT over (expert, m-chunk) items
using the CURRENT step's counts (offsets are host-known pre-launch); hot experts split
within-expert by duplicating the expert id with sub-ranges in the metadata (the grouped
kernel accepts duplicate ids — n_idx depends only on expert_id). Per-device
make_dense_group_metadata over its assigned items; NO kernel change.

**Validation gate.**

```text
G-E2.1 balance <= 5% at all alphas (it should match sEP's floor — that is the point).
G-E2.2 replan+launch overhead measured per layer (the number EG3(i) compares against).
G-E2.3 parity vs static (same tiles, different grouping): loss band + envelope grads.
```

**Risks / watch:** metadata rebuild per step per layer touches the validate_group_plan drain
(cu:89-90) — build host-side and pass down (kills the drain for ALL modes; do it here so
static/sep inherit); duplicated-id groups change dB accumulation grouping — verify the
LoRA-grad path sums sub-groups of one expert before the allreduce2.

## Stage E3 — The Cooperative Queue Kernel (the headline)

**Objective.** Persistent variant of the grouped kernel: grid = #SMs, each CTA loops
{pop chunk from the shared front/back counters (atomicAdd_system on pinned host mem) ->
derive (expert, m-block, n-block) from the item index -> existing tile pipeline}. Both
devices launch the SAME kernel over the SAME full bank + identical packed X; outputs go to
per-device y_i slots; existing combine sums.

**Isolated probe first (I1 pattern — allowed, small buffers only):**
`scripts/testing/ep_queue_probe.py`: atomic RTT per GPU, dual-GPU pop throughput vs chunk
size, a 2-GPU queued grouped-GEMM vs the static launch on synthetic banks — numerics
identical (disjoint writes), busy-balance under forced skew, contention curve.

**Validation gate.**

```text
G-E3.1 probe: numerics bit-match static per tile; balance <=5% at alpha=0.75; atomic
       overhead <2% of layer time at the chosen chunk; duplicate-fetch bytes ~= split
       expert's bank bytes only (front/back receipt).
G-E3.2 e2e parity row (q3-30b-a3b s2048 sep): I7-style gates under the envelope method.
G-E3.3 EG4 no-regression row at natural routing.
FAIL SIGNATURES: one device exits early (termination race: head/tail cross check must
       claim-then-verify); n_idx mismatch (item->tuple math vs :88-91); silent partial
       combine (a device skipped items -> loss band catches only if large — assert
       items_done[dev0]+items_done[dev1]==N_items EVERY layer).
```

### E3.5 — sep1-minimal e2e wiring (design decision, 2026-07-07)

```text
DECISION: sep1's e2e form queues the BASE grouped GEMMs ONLY (fwd gate/up/down + bwd dX —
the host-streamed bank GEMMs where imbalance actually bites); the LoRA grouped GEMMs and
their grads STAY on the static E/2 layout. Why this is exact and sufficient:
  - base and LoRA contributions are ADDITIVE (separate GEMMs) — any split of who computes
    the base tiles is numerically invisible (disjoint outputs, zero-init d, partials sum
    via the existing combine);
  - LoRA compute ~ r/K ~= 64/2048 ~= 3% of expert work -> statically-split LoRA costs
    <= 3% x skew, negligible vs the >=1.2x base-GEMM win; LoRA stays disjoint per branch
    (no replication, no grad allreduce changes).
IMPLEMENTATION SKETCH: branch keeps its E/2 slice for LoRA; base fwd/dX calls swap to
  (full bank ref + FULL pre-slice metadata + ep_queued kernel, side=branch, SHARED pinned
  counters reset per launch-pair; d zero-initialized so unclaimed rows contribute zero).
  Thread metadata_full alongside the sliced metadata through the fg path for base calls.
VERIFIED CONSTRAINTS (2026-07-07 scouting):
  - fg FWD base calls (_base_forward -> AsymGroupedFrozenLinear.forward -> _asym_grouped_
    bf16_nt frozen_linear.py:732/:753) use the PLAIN contiguous grouped kernel — the
    ep_queued variant applies DIRECTLY (add ep_queue/ep_side kwargs + zero-init d).
  - bwd dX uses the ROUTED kernel family (down_dx_gather_left -> qwen3_moe_bf16_down_dx_
    gather_left_) — needs the SAME EP_QUEUED codegen treatment on the routed runtime class
    (mechanical mirror of SM100BF16EpQueuedAsymGemmRuntime; n_blk fix already in the shared
    kernel .cuh).
  - THE REAL RESTRUCTURE: base-queued needs FULL-metadata packing while LoRA keeps sliced
    packing — the fg chain interleaves base+LoRA per stage over ONE packed layout, so sep1
    must pack FULL rows once (identical on both branches) and give the LORA stages sliced
    ROW-RANGE views into the full pack (expert-sorted rows make branch d's rows one
    contiguous range [off_full[lo_d], off_full[hi_d]) — a VIEW, no repack). Outputs: LoRA
    delta scatters into the full-row buffer slice; base outputs queued-disjoint; act/mul
    chain runs on full rows per branch (2x elementwise vs static — negligible) OR sliced
    (needs range views only). This is the v1 wiring plan.
GATES: unchanged (EG1/EG2/EG4/EG5); plus assert claimed==total_items per launch pair.
```

## Stage E4 — Skew Sweeps + Beat Rows (EG1-EG5 evaluated here)

**Runs.** The full mode x alpha grid (dev rows above), scout natural-skew rows, balanced-
routing no-regression rows, 5-step loss overlays static-vs-sep, grad-envelope dumps.
**Gate:** every EG line gets a number + verdict (incl. the EG3 narrowing decision if hit);
Decision Log entries with artifact paths; nothing advances on inconclusive.

## Stage E5 — sEP-v2: Sharded Batches + Shared Token Pool (the headline stage)

**Objective.** Single process, both GPUs, EP invariants restored: split each global batch
into two per-device micro-batches; dense/attention run per-device with ZERO cross-device
comm (DP semantics inside one process); the MoE phase pools both ranks' packed expert
inputs in ONE shared pinned host buffer and runs the E3 queue kernel over the UNION of
(expert, token-block) items with AFFINITY ORDERING (own-token items first per GPU).

**Key mechanics (all on landed substrate):**

```text
dataloader: one loader; global batch 2xb split per device (Trainer still sees n_gpu=1 —
    the sTP patches; PROFILE_GLOBAL_BATCH_SIZE = 2*b under ASYM_EP_MODE=sep).
dense LoRA grads: per-device partials + allreduce2 at step end (DP semantics; tiny).
token pool: the EXISTING x_cpu offload (dA path) writes into a shared layout keyed by
    (rank, expert, block) instead of per-manager buffers — layout change, not new traffic.
steal path: foreign-token X tiles stream host->stealing GPU (H2D, C2C); output tiles
    return via NVLink P2P write into the owner's y_i slot; bwd dX mirrors it. ALL steal
    traffic accounted per step (EG-V2 receipt: ~ stolen fraction; zero when balanced).
queue: E3 kernel UNCHANGED; item list = union over both micro-batches; front/back counters
    with per-GPU sections ordered own-tokens-first (affinity = the front/back trick applied
    per rank-section).
```

**Validation gate.**

```text
G-E5.1 shard receipt: per-device sample ids disjoint; union == the asym_dp2 reference set;
       dense segments show zero cross-device comm (nsys).
G-E5.2 parity vs asym_dp2 at the same row (same global batch, same shards): loss overlay +
       adapter-grad envelope — sEP-v2's dense semantics ARE dp2's; only the expert phase
       differs.
G-E5.3 steal accounting: balanced routing -> ~0 foreign bytes; synthetic alpha sweep ->
       foreign bytes ~ (alpha-0.5)+ fraction; balance <= 5% at all alphas.
G-E5.4 beats static-EP at the SAME sharded shape on the skew sweep (EG2 ratios); beats/ties
       hostsplit per EG3's predeclared rules.
```

**Risks / watch:** shared-pool layout vs the existing per-manager CPU pool (device-keyed
handles already exist — extend keying by rank-section); P2P output returns must ride the
p2p streams with producer events (I1 discipline); bwd steal symmetry (dX of stolen items
returns to the token owner); watchdog floors with the pooled layout (HC5-style audit).

## Stage E6 — Paper Rows + Receipts + Freeze

```text
P4 row (q3-30b-a3b 80000|8|1 ker101) per mode; scout 9500 matrix; Tier-3 mechanism-absence
receipts (file:line, BEFORE any table statement); zero-drop asserts in every reported row;
flagship feasibility note (235B arena spans both Graces — paper-story arithmetic) if pursued.
DONE criteria: EG1-EG5 all verdicted; C-EP1..6 each backed by a named artifact; the ladder
table (static/hostsplit/sep, one knob apart) + system rows published with the honesty
lines (natural-vs-synthetic skew, EG3 narrowing if applicable) in the table notes.
```

## Reporting Format

```text
fwd_s bwd_s opt_s step_s  step_H(g0/g1)  RAM  moe_seg_s  busy0/busy1(%)  imb(%)  dup_fetch_GB
  + per-layer expert histogram artifact; skew rows flagged loss_invalid; labels exactly as
  generated (asym_stp_cpuadamwds | ...-ker101 | _epsep_skew50 fragment).
```

## Decision Log (append-only; date + decision + evidence path)

```text
2026-07-07 DATA-LAYOUT CORRECTION (user): the representative vanilla-EP baseline is
  SHARDED-batch (different batch/GPU + owned experts + all-to-all), matching real
  Megatron/DeepSpeed deployment AND matching sEP-v2's layout. The replicated-batch
  static-EP first built is demoted to a substrate-composition ablation (static_rep); sep1
  stays a mechanism-isolation ablation. ACTION: build the sharded EP-Static baseline
  alongside v2 (E5) — a2a dispatch is the one extra mechanism (DispatchFn exists). The
  paper's 'vanilla EP' row = sharded EP-Static, never the replicated one. Modes + TIER 2
  updated above.
2026-07-06 E3-KERNEL LANDED + PROBE PASS (G-E3.1 at probe level, incl. probe-level G-E1.3/G-E2.2).
  Kernel: NOT a persistent in-CTA loop — each CTA CLAIMS one (segment, n-block) item at KERNEL
  ENTRY (before any barrier/TMEM init; no-ticket CTAs exit at ~atomic cost) via 3 pinned-host
  counters ([claimed, head_taken, tail_taken]; linearizable: front [0..head) and back
  (N-tail..N-1] cannot overlap since claims cap at N). Scheduler gains an explicit-ids ctor +
  n_blk field; epilogue store and transpose-B path were reading blockIdx.x directly (LATENT BUG
  for any non-blockIdx scheduling) — now scheduler.n_blk. Files: asymScheduler.cuh,
  sm100_bf16_asym_gemm.cuh (ASYM_BF16_EP_QUEUED variant, qwen3-routed macro pattern),
  sm100_bf16_asym_gemm.hpp (SM100BF16EpQueuedAsymGemmRuntime + launcher, DG_EP_QUEUE_GRID_PCT
  default 75), gemm.hpp + __init__.py (m_grouped_bf16_asym_gemm_nt_contiguous_ep_queued).
2026-07-06 MEASURED PHYSICS (scripts/testing/ep_queue_probe.py, q3-30b-a3b geometry E=128
  N=768 K=2048, membind=0,1 — REQUIRED: first-touch NUMA moved static busy 1.33->3.38ms):
  (a) per-segment host-B re-fetch ~25us (TMA sysmem reads do NOT L2-cache across CTAs) ->
      fine-chunking ALL experts taxes balanced routing ~13900/chunk_rows %, M-INDEPENDENT;
  (b) vanilla per-expert enumeration SERIALIZES the hot expert's m-loop (only n-parallel
      CTAs) -> static's measured skew wall is far WORSE than the (1+alpha)/2 token floor
      (57.99ms at alpha=0.75 vs 3.3ms balanced at M=1.28M);
  (c) granularity co-design: whole-expert segments for average experts + hot-expert (>2x avg,
      cap 8) chunks at EP_HOT_CHUNK_ROWS=8192 (wall-minimizing sweep 2048/4096/8192/16384/32768;
      16384 lowers the alpha=0.75 wall further but breaks the <=5% balance gate at alpha=0.5).
2026-07-06 G-E3.1 VERDICT (validated; profiling_gb200ep_e3/ep_queue_probe_final.json,
  M=1.28M ~= per-layer routed rows of the s20000|8 ker101 dev row):
  numerics: queue union BITWISE == static reference; zero double-compute; counters exact.
  balance: queue imb 2.1/2.2/4.0% at alpha=0.25/0.5/0.75 (static floor: 82/91/94%).
  speedup: static/queue = 4.24x/6.94x/8.58x (exceeds (1+alpha) because of (b) above —
    report the token floor as the CONSERVATIVE bound, the measured ratio as vanilla-EP reality).
  EG4: balanced overhead 0.969 (<=1.02 PASS; queue is free when balanced).
  EG3 (predeclared tie hit): hostsplit ties/edges queue per-launch (queue/hs wall 0.95-1.0;
    hostsplit replan 0.13-0.21ms host-side per launch) -> per the narrowing rule the queue's
    e2e case rests on aggregate replan cost (~300 grouped launches/step x ~0.15ms vs zero)
    + no per-launch metadata rebuild; DECIDE AT E4/e2e, both rungs are ours.
  DEV-PROCESS NOTE: an sglang server (180 GiB) is resident on GPU 3 — pair 0,1 clean; all
    probe runs under numactl --membind=0,1 --cpunodebind=0,1 (HC1).
2026-07-06 G-E1.2 NATURAL-SKEW MEASURED (validated; |1 q3-30b-a3b s2048|8 ker101, real
  smoke data, ASYM_EP_STATS=1 -> profiling_gb200ep_e1/ep_balance_natural_s2048.json):
  hottest single expert 2.4% mean / 5.5% max share (E=128 top-8; mean expert = 0.78%);
  static-E/2 device share: mean 0.530 (=6% imbalance), worst layer mean 0.567, worst
  layer-step 0.631 (slow device +26%); per-layer max >> mean = temporally VOLATILE skew
  (matches ViBE micro-batch hot-expert turnover). CONFIRMS the predeclared EG2 honesty
  split: natural-skew e2e wins are single-digit % on this model; synthetic sweep + low-E
  models (scout) carry the worst-case headline. Layers 14/26/38/42/46 are systematically
  the most skewed.
2026-07-06 I7-CORE BLOCK-LEVEL PARITY PASS (validated; scripts/testing/stp_moe_block_parity.py,
  tiny E=8 top4 two-device block, plain AND fg/ker101 env): static EP-2 mechanism proven —
  asym_gemm/training/stp_moe.py: ep_slice_route_metadata (expert-sorted flat metadata ->
  CONTIGUOUS local slice, local 0-based ids by offset rebase, zero wasted compute;
  EpSlicedRouteMetadata overrides num_routes), slice_experts_for_ep (dim-0 pinned-bank VIEWS
  shared with |1 — the ownerless arena — + per-device LoRA copies), build_ep_branch_block
  (frozen router replica per device). Hook: ep_expert_range in AsymQwen3Experts.forward;
  EMPTY local range returns a zeros partial (real fix — crash otherwise; possible at low E).
  Gates: route bits identical across branches; y0+y1 vs full block: band 4.9e-4/7.3e-4
  (~1-2 ulp at max|y| 0.14, non-vacuous: vacuity guard added after a degenerate all-zeros
  'pass'); branch LoRA grads EXACTLY equal full-block grads sliced; dX sum band 5.7% of
  mean|dx|. TEST LESSONS: tiny random routers collapse top-k onto experts 0..3 (near-tied
  bf16 logits, stable sort) — spread router explicitly; default-init tiny models underflow
  bf16 to vacuous zeros — scale weights + guard signal.
2026-07-06 G-E1.2b SCOUT NATURAL SKEW MEASURED (validated; |1 llama4-scout s2048|8 ker000,
  real data, first run after auto-download; profiling_gb200ep_e1b/ep_balance_scout_s2048.json):
  static-E/2 device share MEAN 0.608 across 48 layers (busier GPU +22% on the expert
  segment EVERY step on average); WORST layer-step 0.872 (1.74x the balanced time);
  PERSISTENTLY skewed layers exist (layer 40 mean 0.783; layers 0/20/43 all >0.72 mean);
  hottest single expert 16.8% mean / 33.1% max (uniform would be 6.25%). E=16 top-1
  structural skew CONFIRMED on real data with no synthetic forcing — scout carries the
  natural-gains claim as predeclared; expert-segment speedup available to sep1 vs static:
  ~1.22x average, ~1.74x worst-layer, BEFORE MoE-fraction scaling (nsys rows next).
2026-07-06 PRE-FIX MEASUREMENT SET COMPLETE (validated; e1c/e1d artifacts):
  q3 s20000 (10x volume): worst dev-share 0.588 (s2048's 0.631 was partly noise) BUT
    persistent per-layer skew is REAL: layers 29/14/23 hold 0.57-0.58 MEAN -> 1.15-1.18x
    those layers every step. q3 fwd expert fraction (nsys NVTX) = 36.9% of step (bwd
    expert pass roughly doubles it -> expert work ~55-65% of step).
  scout s9500 (target seq): skew PERSISTS at volume — mean dev-share 0.586 (vs 0.608 at
    s2048), worst layer-step 0.813; layer 0 mean 0.787 (a persistent 1.57x layer).
    scout fwd expert fraction = 23.7% (H=5120 dense + top-1 dilute it; ~40-50% incl bwd).
  E2E GAIN BOUNDS (sep1 vs static, natural data): q3 ~3%; scout ~6-8% average e2e with
    persistent worst layers contributing most (segment-level 1.17x avg / 1.45-1.57x on
    skewed layers). Scout carries the natural-gains claim; both models' numbers now fully
    grounded BEFORE the wiring build.
2026-07-07 I7-STATIC E2E LANDED (validated; the first two-GPU MoE run through the real
  trainer): q3-30b-a3b|2 asym_stp_cpuadamwds recomp-off-full-fg-ker101 s2048|8, ASYM_STP_MOE=1
  ASYM_EP_MODE=static. LOSS PARITY vs |1 reference: deltas 0.0040/0.0014/0.0049/0.0088
  across 4 steps (I4 bf16 envelope class). Artifacts: profiling_gb200ep_e2e.
  SHAKE-OUT FIXES (all landed): (a) router-mode whole->hf downgrade allow-list was missing
  the |2 asym family in BOTH driver copies (profile_lora_lf_test_source.sh AND _both.sh
  ~:4529) -> MoE blocks were never asymized under sTP; (b) build_ep_branch_block must SHARE
  AsymQwen3MoeBlock gates that are HostWeight routers (no Parameters; deepcopy UN-PINS the
  host weight -> weight_not_pinned at the first router GEMM) and deepcopy-replicate only
  param-carrying routers; (c) build_stp_full_tp branches on AsymQwen3MoeBlock (MoE) vs
  AsymFinegrainedDenseMLP (dense) with Qwen3MoeAttention for branch1 and moe_ep param_map
  entries; expert LoRA slices are DISJOINT per branch (no mirrors). GPU pool env is
  GPU_POOL=0,1 (not GPUS).
2026-07-07 STATIC s20000 TIMING ROW (validated as BASELINE; interpretation gated on I5):
  step_ms [262344, 231134, 207056]; fwd 12.0s / bwd 245.6s / opt 2.2s; zero kernel
  fallbacks. The bwd dominates exactly as the STATUS interlock PREDICTED: pre-I5 duplicated
  per-branch [M,H] offload traffic (dense track measured the same wall at 209s bwd on
  q3-32b) + MoE bwd x_cpu re-staging. CONSEQUENCE: ladder comparisons (static vs sep1 vs
  hostsplit) remain VALID (shared bwd substrate; MoE-segment delta rides on top) but e2e
  percentages are DILUTED until tp.md I5 lands — land I5 BEFORE quoting e2e ladder numbers
  (the doc's build order, now measured). |1 reference at the same row: 15-20s/step.
2026-07-07 THE 13x WAS MOSTLY MEASUREMENT ARTIFACT (validated; diag runs): the parity run's
  50-95s bwd at s2048 was COLD-JIT (first-ever compiles of the branch kernel variants,
  disk-cached after) + the E0 histogram collector's per-router .cpu() syncs. With stats OFF
  + warm JIT: static |2 s2048 bwd = 13.7-18.4s vs |1's 9-14s -> TRUE substrate gap ~1.3-1.5x
  (the real I5-dedup/serialization work), NOT 13x. nsys decomposition receipts: GPUs 7% busy,
  62s idle gaps, copies only 7s busy (584 GB moved — bandwidth NOT the wall). DISCIPLINE
  RULE ADOPTED: never quote first-run step times for a new kernel-variant config; verify JIT
  cache warm (rerun) before recording any timing row. The s20000 245s-bwd row must be RERUN
  warm before use.
2026-07-07 WARM s20000 STATIC (steady-state rule: drop warmup + last step): steps
  [158.0, 184.5]s; fwd 9.6-10.5s HEALTHY; bwd 118-169s. VERDICT REFINED: s2048's blowup was
  JIT-cold (warm bwd 14-18s ~ 1.4x |1), but s20000's bwd is REAL and TOKEN-LINEAR (14s at
  16k tokens -> ~140s at 160k = the duplicated per-branch [M,H] offload, 200-400 GB/step) —
  tp.md I5 DEDUP IS NOW THE UNAVOIDABLE CRITICAL PATH to the e2e goal (bwd target: ~1x |1
  via shared host copies + dual-lane split). MEASUREMENT RULE ADOPTED (user): timing rows
  exclude the warmup AND the final step; MAX_STEPS=4 so 3 steady middle steps remain.
2026-07-07 I5-LITE MECHANISM (design, next build): dedup the REPLICATED [M,H] saved
  tensors (attention input h, mlp input g, residuals — bit-identical across branches by
  Bcast01Fn/TPRegionFn construction) via a per-layer REPLICA REGISTRY + saved-tensor pack
  hooks: StpDecoderLayer.forward tags branch tensors with a replica key (layer, slot);
  branch0's pack hook offloads ONCE and registers the host handle; branch1's hook looks up
  the key and returns a restager that H2Ds from the SAME host buffer to dev1 (dual-lane
  reads of one pinned buffer proven at 174.7 GB/s/lane). Halves the duplicated D2H AND host
  RSS for those tensors. Disjoint per-branch tensors (q/k/v shards, packed expert rows)
  stay as-is. Target: s20000 bwd 118-169s -> ~|1's (~10-13s) + margin.
2026-07-07 SUBSTRATE FIX ROUNDS (static s20000 bwd, steady-state): 245s(cold-JIT) ->
  122-145(warm) -> 97-104 [R1: bucketed pinned pool — offload buffers keyed by exact shape
  never reused under EP's variable routed counts; two-tier granule 8k/64k in
  activation_offload._alloc_cpu] -> 72-86 [R2: async event-based unpack in
  attention_activation_offload (was non_blocking=False + event.synchronize per restage);
  side-stream + compute waits event] -> ~same [R3: same treatment in
  ActivationOffloadManager.stage/wait_cpu_ready] -> 58-65 [R4: memoized pad/group metadata
  on offsets tensors: _pad_grouped_input_for_asym, _pad_cpu_left_grouped_input_for_asym
  (+pool for its pinned buffer), _expert_blocks, prepare_grouped_lora_metadata — each call
  was paying .item()/sync-D2H/mask-size syncs] -> 49-67 [R5: memoized the SOURCE producers
  (_pad_route_metadata_for_asym, _group_metadata_for_kernel, build_contiguous_route_metadata)
  so downstream memos key on stable per-layer objects]. Loss stable at 4 decimals across ALL
  rounds; block parity re-verified after each. METHODOLOGY: stack-census (py-spy burst) ->
  fix top frame -> steady-state rerun; Megatron-LM deep-read receipts guide the discipline
  (three streams, zero host sync, prefetch-1, uncapped pool; their EP pads to capacity —
  our bucketing gets reuse without padded FLOPs). REMAINING gap vs |1 (~15-20s step): fresh
  nsys decomposition in flight; named levers left: backward prefetch-1, fp32 SDPA save
  volume, fwd d2h side-streaming.
2026-07-06 DEVIATION (scope, honest): probe-level static/hostsplit rows stand in for
  G-E1.3/G-E2.2 until I7 lands the e2e substrate; E1/E2 e2e gates remain OPEN.
```

## Stage Dependency Summary (build order)

```text
tp.md I5 (recommended) -> tp.md I7 static EP-2 (required)
E0 harness knobs + balance instrumentation      (no kernel change; unblocks all gates)
E1 static rows: parity + natural skew + floor    (the attribution rung's numbers)
E2 hostsplit rung (+ host-side metadata build    (strongest scheduler baseline; kills the
   that also removes the validate_group_plan      per-call device->host drain for all modes)
   drain)
E3 queue kernel + probe (= sep1/v1 ablation)     (the mechanism, isolated)
E4 skew sweeps + EG verdicts on v1               (mechanism evidence)
E5 sEP-v2: sharded batches + shared token pool   (THE HEADLINE — EP invariants kept;
                                                  parity anchor = asym_dp2)
E6 paper rows + receipts + freeze                (feeds the paper-story D2 claims)
```
