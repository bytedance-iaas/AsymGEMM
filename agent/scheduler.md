# AsymGEMM Scheduler — formulation + latency-recovery plan (v2)

**Goal**: formalize the scheduler behind the flagship
`asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm8|ligerloss1`
(lowest peak HBM; +25–35% maxB/max-seq ceilings over `superoffload_mem|unsloth-off`) so that
(1) the paper has ONE tractable high-level motif whose implementation details are
instantiations, and (2) the same formulation yields the **memory↔latency dial** that buys back
the ~0.62× (MoE) / 2.7–3.7× (dense) throughput deficit without giving up the ceiling story.

v2 changes (2026-07-09, after artifact re-verification):
- **NVMe removed from the v1 scope** (never exercised in any artifact — all 171 dirs are
  `ceil0000`; keeps the system clean to reason about). Homes are {HBM, DRAM}. NVMe returns
  later as a third home with unchanged structure.
- **Timing protocol: steady-state only** — drop the warmup step, the FIRST measured step, and
  the LAST measured step; report the mean of the middle steps (needs ≥5 total). Several
  existing anchors are n=1–2 → re-anchor before quoting deltas.
- **The grad-hook "smoking gun" is demoted after a controlled A/B** (§0.2) — the sync-hook
  wall time is real but mostly *shadowed* by GPU execution; the verified dominant tax is
  **GEMM-engine + fg kernel-work**, not transfers, not the optimizer, not the hooks.

---

> **2026-07-09 UPDATE — Phase D executed (see `agent/impls/fix_throughput.md` Phase D
> ledger + D6 close-out): flag-gated A/Bs landed. @32k×8 q3-30b the stack
> `ASYM_GEMM_DISPATCH=staged` (+18.4 s solo, 0 HBM) + `ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1`
> (+14.9 s solo, +1.7 GiB in-stack) + async GC unpack (null @32k, held for long-seq)
> = **109.45 → 76.45 s (−30.2%), asym backward now faster than superoffload's**, gap
> 1.64×→1.148×. U1/U2/U3 of §6 are now measured; the §0 numbers below are the pre-D
> baseline kept for context.**

## 0. Ground truth (verified in artifacts, healthy-margin runs)

### 0.1 Standings (q3-30b-a3b b8; `lat.md`/`summary.md`, 4-sample means)

```text
                              step_s   fwd_s   bwd_s   peak-alloc-HBM   tok/s
s32k  asym full-fg ker101      161.0    18.9   137.3     31.7 GiB       1,590
s32k  asym + async-hook fix    159.9    19.0   135.8     (same)         1,601   <- A/B: −1.1 s only
s32k  so  unsloth-off           85.6     8.1    76.1     39.9 GiB       2,991
s80k  asym full-fg ker101      354.6    54.1   294.6     80.6 GiB       1,805
s80k  so  unsloth-off          260.1    26.7   231.1     96.7 GiB       2,461
s80k  so  unsloth (roots HBM)  186.9    26.7   158.1    181.2 GiB       3,424
s131k asym ker101 (at-ceiling regime, host-pressure-inflated)  810.8 / 131.8 GiB
s131k so  unsloth-off                                          554.2 / 152.3 GiB
ceilings: asym 173k vs so 131k (b8); llama70b asym 1038.4 s vs so-off 280.2 s @25k
```

Note what the memory win costs today: at s32k the flagship touches **31.7 of 186 GiB** —
~154 GiB idle — while paying streamed-engine prices on every GEMM. The whole q3-30b frozen
bank (~60 GB bf16: experts ~58 + attention ~2) fits in that headroom twice over.

### 0.2 Where the time actually goes @ s32k (nsys semantic decomposition, per step)

Backward (asym 92.1 s vs so 56.9 s in the nsys window):

```text
                                  asym            so             delta
CUDA memcpy                       24.2 s (26%)    24.2 s (43%)   0      <- traffic is a WASH
host/runtime top-level gap        22.3 s (24%)    14.2 s (25%)   +8 s
GEMM/MLP kernel work              ~30 s (33%)     ~5.0 s ( 9%)   +25 s  <- THE gap
  (other-CUDA asym GEMMs 13.7 + MLP wrapper 12.5 + base dX 3.8 | so: wrapper 3.6 + cuBLAS 1.4)
attention flash bwd                5.85 s          5.83 s        0      <- attention is NOT it
elementwise/copy/index             7.8 s           5.5 s         +2.3 s
```

Forward (19.0 vs 8.1 s): attention projections asym 7.4 vs so 4.2 (SDPA fprop equal ⇒
projections ~4.5×), MLP chain 9.9 vs 2.5 s (~4×), memcpy 0.47 vs 0.58 s (!). Forward has no
grads, no optimizer, almost no copies — **the forward gap is pure kernel time.**

Findings that survive verification:

- **F-A (dominant): engine + fg kernel-work tax.** Streamed asym GEMMs run well below native
  (attention projections ~4.5×; fix_throughput F1 measured ≈⅓ cuBLAS on llama). On top:
  fg-path extra kernel work — token-space LoRA scatter/fill (down-proj LoRA alone = 2.27 s/step
  fwd, ≈ the base GEMM itself; `ROUTE_LORA` fusion bit exists, default 0), padded M=0 expert-
  group launches, 12.5 vs 3.6 s of MLP wrapper-scope kernels. ~70% of the mid-seq gap.
- **F-B: the sync grad hooks are real wall time but SHADOWED.** `hook_grad_copy_ms` = 30.4 s/
  step @32k, 82.8 @80k, 269 @131k — but the controlled A/B (`asyncfix__*` run, async D2H
  staging engaged, hook ms 30.4→24.6) moved step time only **161.0→159.9 s**. The host blocks
  inside the hook while the GPU keeps draining its queue, so most of that wall time overlaps
  execution. The async fix (uncommitted, `ASYM_CPU_ADAMW_ASYNC_GRAD_OFFLOAD=1` default, drain
  once per step) is correct and worth keeping — but it is ~1 s, not ~30/270 s. Lesson encoded
  in §5: *never attribute from wall-time counters alone; demand a controlled A/B or a
  timeline subtraction.*
- **F-C: transfers are a wash at mid-seq** (24 s both — asym moves acts, so moves ZeRO weight
  gathers + save_on_cpu round trips) and both are ~fully overlapped. Traffic becomes a
  first-order problem only near the host wall (131k+: asym idle 63%, attention-boundary fetch
  gaps ~80 s, thrash cliff q3-32b 65k: 1441 vs 714 s @64k). Optimizer: ≤2 s/step everywhere.
- **F-D: attention SDPA is identical** across backends (5.8 s bwd @32k) — the differentiator
  is everything around it.

### 0.3 The structural observation the numbers point at

In `recomp-off-full-fg`, the outer forward runs the **no-grad pure-GPU path** (no MoE offload
— hence fwd memcpy ≈ 0.5 s); all saving happens inside the *backward recompute*: the fg
Function offloads X/gate/up/act/S to CPU and stages them back **within the same layer's
backward window** (production µs–ms before consumption). At healthy margins this within-layer
round trip buys almost no peak-HBM (peak ≈ live GEMM working set, F4/F6 of
`finegrained_offload_failures.md`) and costs traffic + sync points; it is only justified when
one layer's saved set genuinely doesn't fit (near-ceiling: @173k a layer's X+gate+up+act ≈
90+ GB). ⇒ the offload aggressiveness must be a *function of headroom*, not a constant.

---

## 1. Formulation

> **LoRA-SFT is the scheduling of a fixed training DAG onto a compute×memory lattice: every
> operator gets a compute engine, every tensor a storage home, every backward-needed
> activation a liveness action, and every transfer a time slot — subject to HBM/DRAM budgets —
> and a single budget scalar sweeps the schedule from memory-mode to latency-mode.**

```text
σ = (p, e, h, λ, τ)
p(v) ∈ {GPU, CPU}                       compute placement of op v
e(v) ∈ {native-HBM, asym-R, asym-L, asym-fused}   GEMM engine = which operand may be DRAM-resident
h(t) ∈ {HBM, DRAM(pinned)}              storage home (NVMe deferred)
λ(t) ∈ {retain, offload, recompute, derive}       liveness of fwd→bwd saved tensors
τ    = stream assignment, phase-direction budgets, prefetch distance k, sync discipline
```

Feasible iff `peak_HBM(σ) ≤ B_H` and `peak_DRAM(σ) ≤ B_D − margin_thrash`.
Objectives: latency mode (min T s.t. budgets), memory mode (min peak_HBM — the ceiling
regime), capacity×throughput (max s·B/T).

Cost model — five meters plus a residual, with the measured @32k values attached:

```text
T_phase ≈ max{ W_gemm/(η(e)·R_gpu),   [asym: η≈⅓ native — currently the binding meter]
               W_cpu/R_cpu,           [adam 0.5 s, silu on GPU now — slack]
               bytes_D2H/β_c2c, bytes_H2D/β_c2c,   [~24 s bwd, overlapped — slack at mid-seq]
               bytes_dram/β_dram }    [Grace DRAM ~500 GB/s shared — F5 history]
        + T_serial(τ)                 [host gap: 22.3 s bwd @32k; explodes near ceiling]
```

Decisions move load between meters (λ=offload→C2C bytes; λ=recompute→GPU FLOPs + W restream;
λ=retain→HBM budget; λ=derive→keep the [M,r]/compact byproduct; p=CPU→CPU+DRAM meters;
e=native↔asym → η vs HBM residency of one operand) or shrink the residual (τ). **Placement
moves meters; timing drains the residual. Balance the meters, drain the residual, spend
leftover budget on the highest-Δt/Δbyte upgrades.** The measured meter table says: today the
binding meter at healthy margin is **η(e)** — so the dial's first purchases are engine swaps,
not traffic cuts.

Baselines as degenerate corners: `unsloth-off` = {λ=recompute globally + boundary offload,
e=native, weights re-gathered every layer (43% memcpy bwd)}; flagship = {λ=offload/derive
everything, e=asym everywhere, h=DRAM}. Neither is optimal anywhere in between.

---

## 2. The decision lattice

### 2.1 Level-0 laws (fixed, never searched)

1. **GEMMs run on GPU; the engine follows operand residency** — and residency follows the
   budget: DRAM-resident panels stream through asym kernels; HBM-cached panels take native
   kernels. (v2: engine choice is *per site per workload*, not a wrap-time constant — §0.2.)
2. **Memory-bound elementwise follows its data, subject to the DRAM meter** (silu is on GPU
   today because the CPU path was Grace-BW-bound under contention — F5).
3. **Frozen base weights are data**: one read-only pinned copy; never wgrad; never optimizer
   state; HBM holds *caches* of it, never owned copies (the anti-ZeRO-gather law).
4. **Trainable surface is tiny and CPU-owned**: eager per-param grad D2H (async staging +
   one drain), CPU AdamW, bf16 homes. Settled; only its sync discipline was ever a problem.
5. **Backward consumes saved tensors at known GEMM boundaries in LIFO order** → every fetch
   and every cache decision is schedulable ahead of need.

### 2.2 Level-1: per-tensor-class liveness λ (the memory axis)

```text
class            size       recompute cost                   derive option           flagship today
boundary root    [M,H]      whole-layer re-fwd (unsloth GC)  —                       offload; 1/N in HBM (ohbm)
attn U (qkv/o)   [M,H|K]    re-projection (W restream)       shared U across q,k,v   offload, consumed CPU-left
attn S_*         [M,r]      trivial                          ≡ derive                offload (tiny)
sdpa internals   [M,hd,…]   re-SDPA from q/k/v               —                       recompute
MoE X            [R,H]      re-gather (index_select)         compact [M,H]+idx       derive (FG_DA_GPU=1)
MoE gate,up      [R,I]      re-GEMM (frozen-W restream!)     save-gate/recomp-up     offload (in-bwd round trip)
MoE act          [R,I]      elementwise from gate,up         —                       offload (GPU silu on staged)
MoE S_g/S_u/S_d  [R,r]      trivial                          ≡ derive                offload (tiny)
MoE dgrads       [R,I](bwd) —                                —                       retain-HBM (KEEP_DGRADS_HBM=1)
logits           [M,V]      liger chunked loss               —                       ligerloss1
```

Class cost heterogeneity (act recomputes W-free; gate/up don't; X derives at 1/topk bytes;
S at r/I) is why per-class λ (a fractional knapsack) dominates global GC/offload (0/1
knapsacks). Memory-mode boundary condition: with everything offloaded, peak-HBM ≈ live GEMM
working set — beyond that only *chunking* (block-experts, row chunks) lowers the floor. So the
dial's far end hands over from λ to chunk granularity.

### 2.3 Level-2: per-layer tiers (the interpolation axis)

```text
KEEP : λ=retain all classes (roots + acts in HBM)      BAL : retain small + recompute cheap + offload big
MEM  : full-fg offload of every class (flagship)
```

`ohbm N` = the existing 1-D instance (every Nth root KEEP; measured winners llama@3,
q3-32b@8, q3-30b@0; the s80k unsloth row = +84 GiB of KEEP-roots → 1.39×). Generalize to a
tier vector with the position rule: **early layers MEM (longest fwd→bwd residence, most
overlap slack), late layers KEEP/BAL (consumed almost immediately in backward)** — LIFO makes
this orientation provably right.

### 2.4 Level-3: engine + timing knobs (the latency axis)

- **W-panel cache** (new, the biggest lever): h(W_panel) ∈ {HBM-cached, DRAM-streamed} per
  panel; cached sites dispatch to native kernels. Persistent across steps (weights frozen) ⇒
  zero steady-state traffic — unlike ZeRO's 53.6 TB/step re-gather.
- `ker XYZ` route-fused kernels (auto-101 on q3-30b, 000 on dense, *wrong* on qwen3.5 —
  per-model profiled choice); `ROUTE_LORA` bit (default 0 — the 2.27 s/step token-space LoRA
  scatter is the candidate); skip-empty-group launches (M=0 padding).
- LoRA-A fwd engine (cpu-left vs hbm), dA engine (FG_DA_GPU), silu placement, dgrads home,
  chunk sizes, windowed gate/up-recompute backward.
- τ: async grad staging (in tree, validated ≈ free — keep), prefetch distance k (0 everywhere
  today; `weight_offload.py:95` stream still unused), async unpack for decoder/linear-attn
  (blocking `synchronize()` today), per-phase direction budgets.

---

## 2.5 THE TRADEOFF LEDGER — every memory↔latency decision axis, with measured prices

The scheduler's raw material. Each axis: what it moves, which budget it charges, the
measured price tag (q3-30b 80k×8 ohbm0 anchor unless noted), and its mode assignment.
(2026-07-10; sources: Phase D/D11 ledger in `agent/impls/fix_throughput.md`, paper.md
historical tables, status.md, D8/D9/D10 probes.)

```text
#  axis (knob)                          latency effect            memory effect            mode rule
── ─────────────────────────────────── ───────────────────────── ──────────────────────── ─────────────
A1 GEMM engine per site                 −46 s @80k / −18 @32k     +0 HBM measured, even    latency: staged
   ASYM_GEMM_DISPATCH=asym|staged       (native vs ⅓-peak stream) at 3 GiB margin (D8)     memory: either (0-cost!)
A2 Route-fused kernels (ker XYZ)        under staged: ker000      same HBM; ker101 avoids  follow dispatch:
                                        −28 s @80k; under asym:   [R,H] intermediates      asym→101, staged→000
                                        ker101 wins               under asym dispatch      (fp32-accum caveat A13)
A3 Within-layer act round trip          −15 s @32k (bwd only)     +transient ≈ 3·[R,I]+    latency: on below s*
   ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM     (skip D2H→H2D + waits)    [M,H]: 10 GB@32k,        (fit-check from linear
                                                                  ~29@80k, ~73@131k        model); memory: off
A4 GC-root share (ohbm N; share=1/N)    unsloth vs unsloth-off:   ±4.9 GB/layer-class      THE host↔HBM dial;
                                        −87 s @80k (so rows)      HBM↔host; 60k q3-32b:    per-model winner from
                                                                  ohbm8=host-OOM,          ceiling search
                                                                  ohbm4=179 GiB HBM
A5 Whole-layer recompute (unsloth GC)   +~1 fwd pass in bwd       −inner acts per layer    both modes today;
   + save_on_cpu of recompute saves     (bwd/fwd 4–8×)            (the ceiling enabler)    finer λ = future
A6 Per-class liveness λ                 small each                class bytes each:        memory: off/derive;
   (X_UNPACKED/FG_DA_GPU compact X;     (FG_DA_GPU also           X [R,H]→[M,H] = 1/topk;  latency: derive+keep
   ACT_RECOMPUTE; KEEP_DGRADS_HBM;      changes dA engine)        act one [R,I]            per headroom
   S_* derive)
A7 Optimizer/master/grad placement      GPU adam −7.4 s/step;     GPU adam +~40 GB HBM     latency: cpuadamw +
   (asym_cpuadamwds vs GPU adam;        C0 async staging −5 s     (opt states); C0 staging async staging;
   ASYM_CPU_ADAMW_ASYNC_GRAD_OFFLOAD)   class                     +7 GB pinned host        memory: + ASYNC=0
                                                                  (broke the 60k probe!)   (the D9 fix)
A8 Weight residency per module class    routed_experts vs all:    ±22.9 GiB persistent     memory: all;
   (ASYM_OFFLOAD_MODULES)               −35% step (llama4 hist)   HBM (llama4 hist)        latency@short-s: subset
A9 Transfer sync discipline             0 @32k (shadowed);        0 (pinned↔pageable       latency: on (free
   (ASYM_SAVED_TENSOR_ASYNC_UNPACK:     candidate at long s       neutral, D8)             insurance); memory: off
   pinned roots + side-stream restage)  (147 ms/layer fetches)
A10 Host budgets & thrash guard         near-wall thrash: 2×      host caps: pool bytes,   both: margin ≥ 2×
   (EXPACT_CPU_POOL_MAX_BYTES,          step (q3-32b 65k)         watchdog floor 35 GB     watchdog floor rule
   HOST_MEM_WATCHDOG_FLOOR_GB)
A11 Batch size B                        ~flat tok/s past knee     linear act bytes in B    B = B_knee(s) always;
                                        (≤10–15% fixed-cost       (maxB ceiling = capacity maxB is a CAPACITY
                                        amortization)             metric, not a lever)     metric only
A12 Chunking (block-experts,            +launch overhead          caps live transients     memory extreme only
   MLP_RECOMPUTE_CHUNK, liger loss)     (usually small)           below working-set floor; (past full offload);
                                                                  liger: [M,V] never built liger: always on
A13 Accum precision (ker101 fp32        ker000 faster under       same bytes               verify loss curves
   scatter vs ker000 bf16 index_add)    staged (−28 s @80k)                                before promoting ker000
A14 LoRA-A/dA compute placement         hbm vs cpu-left:          ~0 (transient)           latency: GPU;
   (LORA_A_FWD, FG_DA_GPU)              −8.3 s hist @b4_4k        X compact home           memory: cpu-left ok
```

Composition rules (measured; load-bearing for any solver):
1. **Peaks don't add across time**: A3's +9.9 GiB solo became +1.7 GiB in the stack —
   A1 removed the asym padded-operand copies from the same peak window. Cost a schedule
   by simulating the live-set timeline, never by summing per-knob price tags.
2. **Wall-time counters ≠ critical path**: grad hooks, the LoRA scatter bin, and the
   pageable root fetch all had large counters and null A/Bs at 32k — the GPU queue
   shadows host blocking. Only one-flag A/Bs (or timeline subtraction) price an axis.
3. **HBM and host are SEPARATE walls** (q3-32b 60k: host-OOM at ohbm8 AND HBM-wedged at
   ohbm4): carry two budgets; A4/A7/A10 move load between them, A1/A9 are ~free on
   both, A3/A8/A11 charge HBM only, A5/A12 buy memory with FLOPs/overhead.
4. **Time-additivity holds when mechanisms are disjoint** (A1+A3: −33.0 measured vs
   −33.3 predicted @32k) but must be re-verified per anchor (D11 re-measures at 80k).
5. **Anchor discipline**: price latency axes at the memory-differentiating anchor
   (q3-30b 80k×8 ohbm0; q3-32b 49k×8) — never at small workloads where backends
   converge — and every latency flag must hold the asym memory class (< so-off's peak)
   at that anchor or it gets a threshold rule (A3's s*).

## 3. Search space, explicitly

```text
σ = ( λ_c per ~9 classes  ·  π_ℓ per layer {KEEP,BAL,MEM}  ·  chunk
      W-cache set C ⊆ panels (GiB-bounded)  ·  ker/loraA/dA/silu/dgrads bits
      k_prefetch, stream map, sync discipline )
```

Collapse: laws fix p; dominance prunes λ (never offload what derive covers; small tensors
always retain/derive; recompute W-heavy classes only behind a panel cache); tiers and the
W-cache are monotone in the HBM budget ⇒ the search is **one budget scalar + per-class table +
per-model engine bits**. Existing run-name grammar already encodes the coordinates
(`<backend>|<recomp>-ker<XYZ>-ohbm<N>|ligerloss<b>`); new axes extend it (`-hbmcache<GiB>`,
`-tier<K.B.M>`, `-pf<k>`) so ceiling-search ledger fingerprints stay per-operating-point.
(`-ceil` / NVMe tokens: deferred with NVMe.)

Knob→coordinate map: `recomp-off-{base,attn,dense,full}[-fg]` = staged λ presets; `ohbm` =
π_ℓ on roots; `cpuadamwds(+gradoff/weightoff)` = law-4 placement; `LORA_A_FWD/FG_DA_GPU/
KEEP_DGRADS_HBM/SILU_BWD_GPU/X_UNPACKED/ACT_RECOMPUTE` = per-class λ/p flips;
`DOWN_SCATTER_BLOCK_EXPERTS/MLP_RECOMPUTE_CHUNK` = chunk; `ligerloss` = λ_logits.
Baselines are rows of the same table — that IS the "comprehensive formulation" artifact.

---

## 4. Inputs are (model, s[, ranks]); batch is DERIVED — and retention beats max-B

The scheduler's inputs are only the workload the user actually chooses: **model, sequence
length s, (#ranks)**. B (per-device micro-batch) and ga are outputs:

- Training semantics are carried by the *global* batch = B·ga·ranks — ga is free in memory,
  so B is purely a system-efficiency variable.
- Step time is empirically ~linear in tokens at fixed s (`ceiling_table.py` fit
  `t = t0 + c_g·B·s·(1+k·s)` with t0 ≈ 3–13 s ≪ step): **tok/s is ~flat in B** once fixed
  per-step costs amortize. Raising B multiplies work and time together; it does not touch
  c_g. Sanity: so-off @131k b8 measured 554 s ⇒ 1.9k tok/s ≈ its fitted flat curve.
- Retention/caching *reduces c_g itself*. Measured: +84.5 GiB of KEEP-roots (unsloth-off →
  unsloth, same B, same s) = 260.1→186.9 s = **1.39×**; the same GiB spent on B ≈ 1.0× (flat).
  And the biggest c_g lever (W-cache → native engine) attacks the verified ~70%-of-gap
  kernel tax: at s32k the *entire* q3-30b frozen bank (~60 GB) fits in the 154 GiB headroom.

**Rule: raise B only to the knee, then spend every remaining GiB on c_g reduction.**
`B_knee(s)` = smallest B such that (i) fixed per-step overhead (t0 + per-layer orchestration
gaps + hook bubbles) ≤ ~5% of step, and (ii) per-expert rows R/E = topk·B·s/E clear the
grouped-GEMM efficiency floor (~4–8k rows; also amortizes the 128-row padding and K-outer
weight-fetch reuse). At b8, q3-30b s≥32k satisfies both (R/E = 16k @32k) — i.e. **b8 is
already past the knee; maxB-chasing above it is wasted headroom**. Exception: at the extreme
ceiling B is forced to 1–2 anyway and the question disappears. maxB stays meaningful for the
*capacity* story (ceiling tables), not for throughput.

So "keep more on HBM or max out batch?" → **keep more on HBM (weights first, then late-layer
activations), run at B_knee.** Priority of HBM spending, by measured Δt/ΔGiB @mid-seq:

```text
1. W-panel cache (engine swap)      ~60 GB (q3-30b, all)   attacks the ~70% kernel tax; persistent, 0 traffic
2. KEEP-tier late layers (roots+acts)  per-layer GiB       measured 1.39× per 84 GiB on the roots axis alone
3. skip within-layer round trips (gate/up/act stay HBM)    frees C2C + sync points (§0.3)
4. more B beyond knee                                       ~flat; only t0 amortization
```

---

## 5. Cost model, calibration, and the measurement protocol

Closed-form per (model, s): class bytes/layer (U=[M,K] shared over qkv; gate/up/act=[R,I],
R=topk·M; boundary=[M,H]; S=[M,r]); live-set simulation of the fg sequence for peak-HBM.
Calibrated once per (model, hw): η(e) per GEMM-site class (asym vs native, per M,N,K shape —
attention projections measured ~4.5× @32k), β_c2c ≈ 187–206 GB/s achieved, per-layer
orchestration gap, fence bubble. Seed ratios (per 17 GiB tensor): CPU-stage ≈ 38 ms vs
recompute ≈ 700 ms (`finegrained_offload.md:98`).

**Measurement protocol (all future numbers)**: ≥5 total steps; drop warmup, drop the first
measured step, drop the last measured step; report the middle-step mean. `PROFILERS=source`
for timing; strictly one experiment per node; healthy-margin criterion (never quote at-the-
wall points as throughput — the q3-32b 65k thrash row and the 131k anchors are the cautionary
tales). **Attribution rule (F-B lesson): a wall-time counter is a hypothesis, not a finding —
confirm with a controlled A/B (one knob) or a timeline subtraction before ranking work.**

Solver: (1) analytic seed — laws + dominance fill λ; budget scalar fills the W-cache then the
tier vector greedily by calibrated Δt/ΔGiB (fractional across panels/layers ⇒ greedy ≈
optimal); (2) one steady-state probe run re-ranks with measured marginals (contention
coupling); (3) offline certification on the named-config grid (existing ceiling_table
protocol). The controller replaces hand-picked configs with derived ones.

---

## 6. The dial + re-ranked upgrade list (verified expected wins)

```text
given (model, s):  B := B_knee(s);  headroom H := B_H − peakHBM(σ_MEM, B_knee)
  H < 0  → escalate memory end: chunking, deeper MEM tiers (ceiling regime)
  H ≥ 0  → apply τ fixes (free), then spend H:  W-cache → KEEP/BAL tiers → (B stays at knee)
```

```text
U1 W-PANEL CACHE + HYBRID DISPATCH (engine; the verified #1)
   Cache frozen panels in headroom, dispatch those sites to native kernels; stream the rest.
   Expected @32k q3-30b (60 GB cache, 154 GiB free): fwd 19→~9 s, bwd GEMM work 30→~5-8 s
   ⇒ step ~161→~95-105 s ≈ so-off parity at HALF its 32k gap, still ≤ so-off HBM. Dense:
   this is most of the 3.7× (llama 140 GB vs 134 GiB headroom @25k ⇒ ~95% cacheable).
   The cache fraction shrinks automatically as s grows — the dial in one number.
U2 FG KERNEL-WORK DIET (engine/fusion; second verified tax)
   ROUTE_LORA=1 path (kill the 2.27 s/step token-space LoRA scatter), skip M=0 padded group
   launches, trim MLP wrapper-scope kernels (12.5 vs 3.6 s bwd @32k), per-model ker bits.
U3 τ DISCIPLINE (free; mostly implemented or small)
   Keep the async grad staging (validated ≈ −1 s; also removes 25-270 s of host wall = frees
   the CPU thread), async decoder/lin-attn unpack, k-ahead restage prefetch on the existing
   side stream + LoRA slab prefetch. Main payoff at LONG seq where fetch waits surface
   (attention-boundary gaps ~80 s @131k) — sized by timeline subtraction, not assumed.
U4 TIER/LIVENESS RE-ASSIGNMENT (λ, π_ℓ)
   Late-layer KEEP/BAL first (measured axis: 1.39×/84 GiB on roots); skip within-layer
   round trips at healthy margin (§0.3); attn U/S retain next. Matters most 64k→ceiling,
   and it is what makes room-accounting for U1 exact.
U5 PHASE/DIRECTION ARBITRATION (τ, later; single-GPU sibling of gb200 M4)
```

Modes: memory = fit-minimum (flagship+chunking; ceilings unchanged), latency = capacity
(full W-cache + KEEP ≈ native-engine system *without* ZeRO re-gather — should beat
so-unsloth's 186.9 s at similar HBM), auto = fit-first-spend-rest. Frontier calibration
points @s80k already measured: (80.6, 1805) → (96.7, 2461) → (181.2, 3424) tok/s; the
controller's job is to dominate that curve from the left.

---

## 7. Roadmap (each gate at steady-state protocol, ceilings re-searched after)

```text
P0 Validate+land async grad staging (done in tree; A/B'd −1.1 s @32k, frees host thread).
   Gate: hook_grad_copy_ms→enqueue-scale; no step regression; ceilings unchanged.
P1 U1 static W-cache: per-layer panels (dense) / whole-bank+expert-heat (MoE), token
   `-hbmcache<GiB>`. Gate: q3-30b @32k ≤ 110 s (so-off 85.6, from 161); llama @25k ≤ 1.3×
   so-off; identical footprint at cache=0.
P2 U2 fg kernel diet: ROUTE_LORA=1 A/B, skip-empty-groups, wrapper-scope trim.
   Gate: ≥10 s/step @32k q3-30b combined, loss-parity.
P3 U4 tier vector (generalize ohbm; late-layer KEEP first) + skip in-bwd round trips at
   margin. Gate: monotone (tok/s, HBM) frontier over ≥4 budget points; MEM endpoint
   footprint identical to flagship.
P4 U3 prefetch-k + async unpack, sized from a 131k timeline subtraction.
   Gate: attention-boundary gaps gone; long-seq stall% −15 pts; else fold honestly.
P5 Controller (analytic seed + probe): emits the config token from (model, s).
   Gate: auto within 5% of best grid point per (model, s).
```

---

## 8. Paper presentation

1. Motif figure: the lattice (ops × engines, tensors × homes, λ arrows, τ clock); caption =
   §1 boldface.
2. Laws paragraph (§2.1) + the meter/residual equation with measured @32k values — the
   motivation table IS the cost model instantiated.
3. Coordinate table (§3): unsloth / unsloth-off / superoffload / flagship / ours-auto as rows.
4. The dial (§6): frontier plot (tok/s vs peak-HBM; three measured points + our curve;
   "the budget scalar is the mode"); capacity plot (ceilings, so = x beyond 131k).
5. Honesty ledger: recompute-vs-offload bandwidth dependence = POET/Capuchin's axis (we claim
   the streamed-weight instance + measured flip boundary); Poolside's resident-weight GB200
   datapoint delimits; η_asym kernel work is orthogonal (U1 makes the system robust to it);
   dense superoffload rows train attention-only LoRA (align surfaces before headline dense
   comparisons); hook-counter mis-attribution documented as the A/B-methodology example.

Open v2 questions: (a) DRAM-side budget governor without NVMe (margin_thrash enforcement =
watchdog + B_D-aware tier demotion); (b) expert-heat plumbing for U1-MoE (ASYM_EP_STATS);
(c) does U1 need TMA-visible HBM copies or plain tensors + dispatch switch (implementation
detail, not design); (d) 2-GPU composition unchanged (gb200_story D2/D3; NVLink = 6th meter).
