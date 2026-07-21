# Scheduler v2 — the memory↔latency dial, formulated to convergence

2026-07-13 (v2.2 after two critique passes). **This is the single working scheduler
doc.** It absorbs the retired `scheduler.md` (v1) and `scheduler_knob.md`; v1's laws
and meter model survive inside §2, its A1–A14 axis ledger is condensed into §3's D1–D6
+ the class-1 pins. Full A1–A14 detail + the frozen per-model metric snapshot live in
`agent/impls/archive/scheduler_knob.md` (record only). Impl+probe evidence:
`agent/impls/memory_mode.md`. Jobs of this doc:
(A) the formulation — one objective, one admission rule, all knobs classified;
(B) the transferable knob table (memory→latency, gradual and clean);
(C) the measured dial ladder + per-model records (§3b, §7) and operating procedure.

---

## 0. The problem, minimally stated

One training step of LoRA-SFT (frozen base, tiny trainable surface) is a FIXED dag.
Scheduling freedom is only: where tensors live, which engine multiplies them, what is
saved vs recomputed vs rematerialized, and when bytes move. Two hard budgets: HBM
capacity B_H and host DRAM B_D (separate walls, measured independently bindable).
Objective: steady-state step time T. The "mode" is nothing but the relative price of
HBM vs time.

```text
P(B_H, B_D):  min_x T(x)   s.t.  M(x) ≤ B_H − m_H,   D(x) ≤ B_D − m_D,   x ∈ F
J(x; β_H, β_D) = T(x) + β_H·M(x) + β_D·D(x)          (scalarized; β = shadow prices)
```

memory mode = lexicographic min M then T (β_H → ∞) · latency mode = min T s.t. fit
(β_H → 0+) · every intermediate point = one finite β_H. Sweeping β_H traces the lower
convex hull of the achievable (M, T) set — **the dial IS β_H**.

---

## 1. Knob taxonomy — the first structural theorem of the dial

Every knob k has a measured price vector (ΔT_k, ΔM_k, ΔD_k) (conditional on the
already-admitted set; see §4). Knobs partition into four classes, and ONLY class-2
knobs belong on the dial:

```text
class 1  UNCONDITIONAL OPTIMA (dominant: ΔT ≤ 0 AND ΔM ≤ 0, or strictly better at any β)
         → in the base config of BOTH modes; never a tradeoff.
class 2  FRONTIER KNOBS (ΔT < 0, ΔM > 0): buy time with HBM; admitted in ρ order.
class 3  WALL ARBITRAGE (ΔM and ΔD opposite signs): move load between the two walls;
         admitted by two-price rule, independent of the time dial.
class 4  REJECTED (ΔT ≥ 0 and ΔM ≥ 0 in the binding window, or guard-violating).
```

Class-1 membership is an empirical claim per knob and must be re-verified when the
binding window moves (a class-1 knob can become class-4 at another anchor — chunk
granularity almost did). Current classification (evidence: memory_mode.md):

```text
class 1: fg chunking g=1024 (swept optimum: reserved −13 GiB, ΔT≈0)
         staged down-dx (removes grad_2d AND the 5.5 s/call gather kernel)
         lean chunked RMSNorm; liger loss; CPU AdamW + eager grad offload;
         B := B_knee(s); ker-follows-dispatch; in-place LoRA delta/dx adds
class 2: ASYM_GEMM_DISPATCH=staged (margin guard: ≥ max panel bytes)
         ker000-under-staged (ΔM≈0 ⇒ ρ≈∞ — degenerate class 2, effectively free)
         keep-acts κ ∈ [0,1] (graded, per-layer from the TOP of the stack; §3)
         W-panel cache c ≥ 0 GiB (future; lowest ρ, admitted last)
class 3: ohbm r ∈ [0,1] (roots HBM↔DRAM: ΔM = +r·L·root, ΔD = −same, ΔT < 0)
         async grad staging (ΔD = +7 GB host, ΔT ≈ −1 s: on unless host-bound)
class 4: keep-dgrads-off; attention-LoRA chunk; sdparecomp; chunk 512/2048;
         ker111 (dominated by staged down-dx); B past the knee; GPU AdamW
```

---

## 2. Cost structure

### 2.1 Peak memory: window-max, never additive

```text
M(x) = max_w  Σ_{t live in w} bytes_t(x),   w ranges over per-layer backward windows
       {attn-bwd(ℓ), silu/down(ℓ), gate/up(ℓ), recompute-fwd(ℓ)} ∪ {loss}
```

LIFO backward (law) fixes liveness intervals, so M(x) is computable by a live-set
simulation without running. The **binding window** w*(x) = argmax. A knob's effective
memory price is its bytes in w* only: ΔM_k^{w*}. Evidence: three probes removed
10–24 GiB from non-binding windows and moved the peak ~0 (kd0, attn-chunk, sdparecomp);
today w* = attention-backward (FA-bwd set ≈ 33 GiB @120k + projection transients).

Two corollaries:
- **w* migrates as knobs are admitted** (keep-acts at large κ re-binds w* to the kept
  windows) — recompute w* after every admission; a knob's ρ is not a constant.
- **Retention orientation**: retaining layer ℓ's saved set keeps it live in every
  window between fwd(ℓ) and bwd(ℓ) — nearly ALL windows for early ℓ (full peak
  charge), almost none for the last layers. Hence κ retains from the top of the stack
  downward, marginal cost is non-decreasing ⇒ the κ-frontier is concave ⇒ greedy
  along κ is exact.

### 2.2 Time: meters and residual (v1's model, kept for off-anchor prediction)

```text
T(x) ≈ Σ_w max{ W_gemm(w)/(η(e)·R_gpu),  bytes_D2H(w)/β_c2c,  bytes_H2D(w)/β_c2c,
                W_cpu(w)/R_cpu,  bytes_dram(w)/β_dram }  +  T_serial(τ, sync points)
```

Knobs act on identified meters: engine bits raise η (asym ≈ ⅓ native streamed; staged
≈ native + panel H2D; resident = native); keep-acts deletes D2H+H2D bytes AND their
sync points from T_serial; ohbm deletes root-fetch bytes; chunking bounds transient
bytes at ~zero meter cost (launch overhead shadowed for g ≥ 512 MB). The 2 s/call
host tax of asym-dx / CPU-right calls lives in T_serial — hence the law: exactly one
of each per layer, block only GPU-only work.

### 2.3 Price scaling laws (why ONE ladder serves all seqs)

With tokens N = B·s and routed rows R = topk·N, every measured price is affine in N
to first order: ΔT_k ≈ c_k·N + d_k, ΔM_k ≈ a_k·N + b_k. Therefore the exchange rate

```text
ρ_k(s) = −ΔT_k/ΔM_k^{w*}  →  c_k/a_k   (s-independent for large N)
```

**The admission ORDER is stable across workloads; only the budget line moves.** The
seq-threshold rule (s*) for any priced knob is the crossing point
`Σ_{admitted} ΔM(s) = B_H − m_H − M_base(s)`: at longer s the same ladder simply
truncates earlier. This is the precise sense in which the dial "transfers gradually
and cleanly": one ordered knob list, one budget-crossing rule.

### 2.4 Floors (the frontier's endpoints, analytic and falsifiable)

```text
M_floor(s) = max-window live set with ALL class-1 on and κ=r=c=0
           ≈ FA-bwd set (q,k,v,out,dout,dq,dk,dv + lse) + projection transients
           + chunk-bounded fg transients   [measured @120k b8: 97 alloc / 104 reserved]
T_floor(s) = roofline with every panel resident: Σ FLOPs/(η_native·R_gpu) + FA fwd+bwd
           + irreducible C2C (grad D2H of the trainable surface only)
```

The dial interpolates between (M_floor, T(M_floor)) and (M(T_floor), T_floor). Any
config beating a floor falsifies the model → update the window set or the meter list.
Distance-to-T_floor is the honest "how much latency is left on the table" certificate.

---

## 3. THE TRANSFERABLE KNOB TABLE (memory mode → latency mode)

Base config (both modes; class 1 — not choices): chunk=1024, staged down-dx, lean
RMSNorm, liger, CPU AdamW + grad offload, B=B_knee, ker-follows-dispatch.

The dial: admit top-to-bottom (ρ order); stop when the fit-check fails or β says stop.

```text
#   knob (graded?)                        ΔM^{w*}                ΔT                    ρ = s saved / GiB     stops when
D1  ASYM_GEMM_DISPATCH=staged             +max panel ~0.4–2 GiB  −46 s @80k (v1)       ≫ 10 s/GiB            margin < panel
D2  ker000 token (with D1)                ~0                     −28 s @80k (v1)       ~∞                    dispatch=asym (then ker101)
D3  keep-acts κ: last ⌈κL⌉ layers         +κ·Σ_ℓ saved-set(ℓ,s)  −κ·L·(roundtrip+sync) MoE ≈ 5, dense ≈ 15   Σ ΔM crosses headroom (s* rule)
    (MoE: FG_KEEP_ACTS_HBM; dense twin)   (linear in s)          (linear in s)         (v1 anchors)
D4  ohbm r: every-Nth root in HBM         +r·L·[B·s,H]           −r·(root fetches)     arbitrage: also       host wall no longer binding /
    (class 3: also relieves host)         (3.7 GiB/root @120k)                         −ΔD host              HBM margin gone
D5  W-panel cache c GiB (P3, unbuilt;     +c exactly, persistent −(staged H2D/call)    ≈0.5 s/GiB (est.)     leftover headroom = 0
    subsumes v1-A8 module residency)
D6  GC-off tier: last k layers skip       +full inner act set    −(recompute fwd,      ≈0.1–0.2 s/GiB        never at long s; short-s
    unsloth GC entirely (unbuilt)         ≈10–20 GiB/layer @120k ≈1.9 s/layer @120k)   (est.) — LAST         only, after D5
```

Class-1 pins not shown above (already optimal, never dialed): FG_LORA_A_FWD_GPU=1,
FG_DA_GPU=1, KEEP_DGRADS_HBM=1, silu-bwd on GPU, S-tensors derive, compact-X.

Reverse direction (latency→memory) is the same table bottom-to-top: shed D5, D4, D3…
each shed returns its exact ΔM. Nothing else changes — that is the "clean" property:
class-1 knobs never flip, class-4 never enter, and the admission order never reorders
(§2.3 scale invariance).

## 3b. Measured dial ladder — 30B-A3B @120k×8, §5 protocol (CONFIRMED)

MEASURED 2026-07-13 (steady = middle-2 of 4; loss parity on every rung, spread ≤0.007):

| Rung | config delta | reserved GiB | steady s/it | alloc | CPU RSS | marginal price → ρ |
|---|---|---|---|---|---|---|
| L0 MEMORY MODE | base, ker101 | **100.3** | 483.6 | 97.3 | 572 | — |
| L1 | +D1 staged dispatch | 107.6 | 416.8 | 97.4 | 572 | −66.8 s / +7.3 GiB → **ρ=9.2** |
| L2 | +D2 ker000 | 143.6 | 384.3 | 113.9 | 572 | −32.5 s / +36.0 GiB → ρ=0.90 |
| L3 LATENCY MODE | +D3 keep-acts (κ=1) | 180.0 | **352.5** | 118.0 | 517 | −31.8 s / +36.4 GiB → ρ=0.87 |
| ~~L4~~ | ~~control: chunk→0~~ | 173.4 | 382.1 | 123.9 | 637 | STRUCK: +29.6 s AND +5.9 alloc vs L3 — chunking is a LATENCY win too (class 1 confirmed, stronger than predicted) |
| ~~L5~~ | ~~+D4 ohbm8 on L3~~ | 182.7 | 355.7 | 131.0 | **493** | STRUCK from speed dial at this anchor (+3.2 s, +2.7 GiB); pure class-3 arbitrage: CPU −24 GB — use when HOST-bound |
| REF | superoffload_mem\|unsloth-off | 140.7 | 459.3 | 139.8 | 617 | **dominated by L1 on BOTH axes** (−33 GiB, −42.5 s) |

Readings:
- The frontier (reserved, steady): (100.3, 483.6) → (107.6, 416.8) → (143.6, 384.3) →
  (180.0, 352.5). ρ is monotone decreasing (9.2 → 0.90 → 0.87) — the admission order
  was correct, and the near-tie of L2/L3 means their relative order is free.
- **Goal check, measured**: latency mode (L3) is **23% faster than superoffload** at
  +28% memory (fits: 180 < 185); L1 is faster AND 33 GiB smaller (strict domination);
  memory mode (L0) is 40 GiB smaller at +5% time. The whole dial sits left/below the
  superoffload point.
- Price-table corrections vs v1 anchors: D2 (ker000-under-staged) is NOT ~free at the
  new memory floor — the route kernels were carrying a real memory role (fwd_scatter +
  gateup_dx_scatter avoid [R,H] route-space tensors), so ker000 costs +36 GiB here;
  its v1 "~0 HBM" price was measured on the pre-chunking base where those tensors
  were shadowed by bigger transients. Window-max model behaving exactly as specified.
- D4 (ohbm) has ~zero ΔT at 120k (root fetches fully overlapped at this seq) — its
  v1 −87 s price came from the pre-async-unpack era. Reclassified: pure host↔HBM
  arbitrage until a host-bound anchor shows otherwise.

Banked anchors (2026-07-12, 4-step averages): memory mode 30B@120k 97.2/109.0/536.6 vs
so 139.8/140.7/~494 · 32B@52k 88.0/103.0/557.7 vs so 115.2/126.5/418 · 30B@160k
127.6/138.5/779 vs so infeasible.

---

### 3c. Blessed presets (from the measured ladder)

```text
MEMORY MODE   = L0: recomp-off-full-fg-ker101 + defaults              100.3 GiB / 483.6 s
BALANCED      = L1: + ASYM_GEMM_DISPATCH=staged                       107.6 GiB / 416.8 s   <- default recommendation:
                                                                                              strictly dominates superoffload
LATENCY MODE  = L3: + ker000 token + ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1   180.0 GiB / 352.5 s
HOST-BOUND    = any + ohbm N (class-3: −CPU RSS, ~0 ΔT @120k)
```

Open probes (cheap, optional): keep-acts directly on L1 (order-swap of the near-tied
L2/L3 rungs); graded κ between L1 and L3 for mid-budget points; dense-model ladder.

## 4. The solver (complete procedure)

```text
given (model, s, B_H, B_D, β_user or target M*):
1  B := B_knee(s);  x ← class-1 set                     # memory mode stops here
2  simulate live-set → M(x), w*(x);  H := B_H − m_H − M(x)
3  while H > 0 and best ρ > β_user:
       k* := argmax_k ρ_k computed against CURRENT w*(x)   # graded knobs: next unit
       admit k*;  re-simulate M, w*;  update H
       (class-3 knobs admit on the two-price rule β_D·|ΔD| − ΔT > β_H·ΔM)
4  ONE steady-state probe (§5): correct ΔT (interaction error) and M (simulation
   error); if any admitted knob's corrected ρ < β_user, shed it; at most one repeat.
5  loss-parity gate; emit config token.
```

Why greedy is enough (and what it assumes): the κ-axis is concave (§2.1), binary knob
prices are measured as CONDITIONAL marginals in the exact admission order (the ladder
= the chain of conditional prices), and time savings show diminishing returns across
mechanism-overlapping knobs (staged dispatch ⊃ staged down-dx measured). We do not
claim global optimality over arbitrary knob sets — we claim: (i) exactness along the
graded axes, (ii) frontier-consistency at the measured rungs, (iii) any violation is
caught by step 4's probe and corrected. For ≤ ~6 frontier knobs this closes the gap
to exhaustive search at 1/50th the run cost.

## 5. Measurement protocol (unchanged, canonical)

WARMUP_STEPS=1 MAX_STEPS=4, PROFILERS=source, one experiment per node; steady s/it =
mean of the middle 2 measured steps from heartbeat `training_step_start` deltas;
memory = peak reserved (alloc alongside); loss parity gates every quoted rung; a
wall-time counter is a hypothesis until a one-knob A/B confirms it. Ladder runner with
timeout+retry: `scripts/lf/run_dial_ladder.sh` (rare router-bincount hang, 1/~15 runs);
the runner trusts only `jobs.tsv` `ok` status — the driver exits 0 on failed jobs
(CONTINUE_ON_ERROR), which produced a false-OK rung on 2026-07-13.

Composition bug found by the ladder (fixed 2026-07-13): fg blocked paths passed
expert-SUBSET metadata with `dense_experts=True`; the asym kernel tolerates subsets
(indexes weights by expert id) but the staged/torch grouped path validates
groups==num_experts → D1 crashed on the new memory-mode base. Fix: `dense=False` on
all 9 blocked `_base_forward`/`_base_dx` sites; verified exact-equal vs dense
full-width. Frame lesson: knob COMPOSITIONS cross engine boundaries that solo probes
never exercised — the ladder is the integration test of the feasibility set F, not
just its pricing instrument.

---

## 6. Convergence review (v2.2) — what would still improve this, and why it doesn't block

Checked and incorporated: knob taxonomy (v2.1); window-max peak + binding-window
migration; graded-κ concavity from LIFO; affine scaling laws ⇒ order stability ⇒ the
s* rule as budget crossing; two-wall pricing with arbitrage class; meters retained for
off-anchor prediction; analytic floors as falsifiable endpoints; conditional-marginal
pricing = ladder semantics; probe-correct-shed loop; explicit greedy-optimality scope.

Known irreducible gaps (accepted, not formulation defects):
1. Prices are measured at anchors; the affine laws are first-order. Remedy is data
   (second anchor per knob), not more math.
2. w*-simulation fidelity: fragmentation (reserved−alloc) is empirical g-dependent,
   not derivable from the live-set model — carried as a measured surcharge per g.
3. Interactions beyond the admission chain (a knob admitted out of order) are
   unpriced by construction; the procedure never takes those paths.
4. T_floor needs the resident-roofline microbench per model (bench exists for the
   down site; extend when W-cache lands).

v2.2→v2.3 pass: added D6 (per-layer GC-off KEEP tier — the one v1 axis, A5, that was
held fixed rather than dialed; its estimated ρ ranks it last, so its absence never
changed an admission decision, but the table is now complete over all 14 v1 axes);
merged v1-A8 into D5; listed the class-1 pins explicitly. Scalarization caveat made
explicit: β-sweep reaches the convex hull; the budget-greedy (step 3 of §4) also
reaches hull-interior points — the procedure is the budget version, β is only the
stop-early preference. Multi-GPU stays out of scope (v1 gb200 notes unchanged).

Verdict: **the formulation is converged.** Audit trail: all v1 axes A1–A14 are now
either class-1 pins, frontier rows D1–D6, class-3 arbitrage, constraints, or
rejected-with-evidence; the admission rule + window-max peak model explains every
measured null and every measured win to date; remaining improvements are strictly new
measurement (anchor refresh, floors) or new mechanism (D5/D6/graded-κ builds), not a
better mathematical frame. If a future probe contradicts a prediction (a config beats
a floor, or an out-of-order admission wins), §2.4/§4 say exactly which assumption to
re-open — the formulation is falsifiable, which is the strongest convergence claim a
measurement-driven frame can make.

---

## 7. Per-model measured dials + capacity (records)

Absorbed from the retired `scheduler_knob.md`; the frozen full-detail snapshot (incl.
the A1–A14 ledger) is `agent/impls/archive/scheduler_knob.md`. Protocol §5 unless noted.

### 7a. q3-30b-a3b (MoE) @80000|8|1 ohbm0 — the dial

```text
dial point                                    s/step   tok/s   peak HBM   what the step trades
LATENCY MODE (staged+ker000+keep-acts+async)  175.7    3,642   84.7 GiB   —
 - keep-acts (ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=0)  200.0   3,200   80.0    +24.3 s buys −4.7 GiB
 - ker000 → ker101                            227.6    2,813   80.1       +27.6 s buys ~0 (follows dispatch)
 - staged → asym engine                       273.5    2,340   80.1       +45.9 s buys ~0.4 GB transient
MEMORY MODE (+ASYM_CPU_ADAMW_ASYNC_GRAD_OFFLOAD=0)  ~278  ~2,300  80.1     −7 GB HOST (pinned staging off)
references:  superoffload_mem|unsloth-off 228.1/94.4 · unsloth (hist) 186.9/176.9
```
Entire asym dial sits LEFT of both baselines in memory; fast end beats both in speed.

### 7b. q3-32b (dense) @49000|8|1 ohbm0 — the dial

```text
DENSE LATENCY MODE (staged + keep-acts)       277.6    132.2 GiB  —
 - keep-acts (ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM=0)  836.7   95.5   +559 s buys −36.7 GiB (!!)
 - staged → asym engine                       964.5     95.5     +128 s buys ~0
references:  superoffload-off 373.1/108.5 · unsloth 221.0/177.3
```
Dense keep-acts lever is 20× the MoE one (tensors [392k,25600] = 20 GB each). Latency
mode 1.34× faster than so-off at +24 GiB. At 49k keep-acts EXCEEDS so-off memory (near
dense s*); at smaller s the +37 GiB shrinks and stays under. Loss parity Δ ≤ 0.0013.

### 7c. llama3.3-70b (dense) @32000|8|1 ohbm0 — post-merge (2026-07-14), w1+m1

| mode | alloc | reserved | s/it | loss | vs superoffload |
|---|---|---|---|---|---|
| memory (keep-acts off)  | 67.0  | 82.7  | 476 | 1.207 | −23 GiB HBM; +39% time |
| latency (staged+keep-acts) | 111.0 | 135.2 | 199 | 1.206 | 1.72× faster; +21 GiB |
| superoffload (control)  | 90.3  | 101.5 | 343 | 1.206 | baseline |

Both goals hold on merged main_kevin: memory mode beats so on HBM (67 vs 90), latency
mode beats so on speed (199 vs 343 = 1.72×); smooth (+44 GiB buys −277 s), loss parity.

### 7d. Post-merge re-verification of §3b (2026-07-14, w1+m4)

The §3b 120k ladder reproduces on merged main_kevin: memory 104.6 alloc / 483.8 s,
latency 126.7 / 352.6 s, superoffload 150.1 / 474.1 s — s/it matches the banked ladder
exactly; a UNIFORM ~+8–10 GiB HBM / +40 GB RSS offset appears across ALL three configs
INCLUDING superoffload (independent path) ⇒ system-wide measurement/env baseline drift
vs the 2026-07-12 bank, NOT a regression. Relationship + ordering unchanged.

### 7e. Sequence-capacity ceilings (post-merge, w1+m1) — of ~185 GiB GB200

| run | alloc | reserved | CPU RSS | s/it | verdict |
|---|---|---|---|---|---|
| q3-32b @65k ker000 ohbm8      | 154.2 | 166.8 | 882 GB | 1281 | FITS (record C-OOM ~66k) |
| q3-30b-a3b @174k ker101 ohbm16 | 153.6 | 170.9 | 825 GB | 878  | FITS (record 172k max-OK / G-OOM 188k) |

---

## 8. THE REGIME SCHEDULER (v3, 2026-07-18) — formal decision process over the dial

§4's greedy knob solver optimizes WITHIN a mode. This section adds the layer the owner
asked for: a FORMAL top-level decision process that picks the OPERATING REGIME per
workload, such that the composed system's throughput ≥ max(all baselines) at every
(model, s, B) — falling back to a baseline-EQUIVALENT configuration when the workload
fits (not an if/else between two systems: one system, one config space, three regions).

### 8.0 Inputs, models, constants (all measured on c12/c14, GB200 185.0 GiB)
```
workload  w = (model, s, B_req | token-budget)
memory model    resv_c(s,B) = base_c + k_c·(B·s)          # fitted per (model, config c);
                validated ±2% on 200+ runs; UNDER-predicts near the wall — treat any
                prediction within 8% of M_phys as "probe before trusting" (frontier rule).
throughput model  t_c(s) us/tok = attn(s) + tax_c          # attn(s) shared by ALL configs
                (proof: rc ≡ uns per-token at 128k/160k: 1101≈1110, 941=942); saturation
                requires B·s ≥ N*_c (knee); below knee MFU falls (the deficit windows).
SAFE = 0.92·M_phys = 170 GiB (thrash guard: >97% + irregular allocs = churn; recomp-like
                regular patterns tolerate 98% — regime-1 configs are irregular ⇒ 0.92).
```

### 8.1 The three regimes (single config space, three regions)
```
R1 FAST  (baseline-equivalent, memory pressure LOW) — REVISED 2026-07-18 after S0':
    config: `asym_cpuadamwds|unsloth-ohbm<N>` — the driver's EXISTING unsloth-GC recompute
            mode on the asym backend (+ ASYM_GEMM_DISPATCH=staged for native GEMMs).
    ⇒ compute shape ≡ superoffload-unsloth (GC recompute, ~336 GB/step boundary traffic
      only) BUT base weights stream from host (staged per-GEMM at measured parity)
      ⇒ ~65 GB less HBM than sup at equal (s,B) ⇒ same per-token speed, MORE batch
      headroom. R1 is a SUPERSET of the baseline, not an imitation.
    KEY INSIGHT (S0' root cause): the old "latency mode" (recomp-off-*) is NOT the fast
      mode at long seq — recompute-OFF saves+offloads ~2.2 TB/step of attention tensors
      (serial tax +286 us/tok bwd) vs recompute's +166. recomp-off is an R2/R3 tool
      (it frees recompute FLOPs when traffic can hide); unsloth-GC is the R1 spine.
R2 GAP   (pressure MODERATE: s long, B undersaturated at R1):
    config: KA + staged + selected fg offload classes admitted by §4's greedy dial —
            EXACTLY enough offload to fit B_target = min(B_sat(s), B fitting at SAFE).
    ⇒ trades copy-engine time for batch headroom; wins when the batch gain outruns the
      offload tax (requires fix_asym S-mem to keep the tax ≤ the deficit it recovers).
R3 ULTRA (pressure HIGH: even B=1 needs full offload):
    config: memory mode (full fg offload, ohbm per ceiling ladder, C0 off) —
            the capacity flagship; objective flips from tok/s to feasibility.
```

### 8.2 The decision function (formal)
```
SCHEDULE(w):
1  B_sat := ceil(N*_R1 / s)                                  # smallest saturated batch
2  if resv_R1(s, max(B_req, B_sat)) ≤ SAFE:      return (R1, that B)     # fallback zone
3  B1_max := max B: resv_R1(s,B) ≤ SAFE
   B2_max := max B: resv_R2*(s,B) ≤ SAFE          # R2* = R2 with dial-admitted knobs
   if B2_max > B1_max and t_R2(s)·(1+underSat(B1_max,s)) > t_R2 gain check:
       # admit R2 iff modeled tok/s(R2, B2_max) > tok/s(R1, B1_max); both from §8.0 models
       return (R2, B2_max, knobs from §4 dial run to fit B2_max)
   else: return (R1, B1_max)                      # R1 at its B_max still the best
4  if resv_R1(s,1) > SAFE and resv_R2*(s,1) > SAFE: return (R3, B=1..)   # capacity zone
5  Hysteresis: regime switches require the modeled gain > 5% (measurement noise 0.3%,
   model error ~2%); ties break toward the LOWER regime (simpler config).
6  One steady-state probe (§5) validates the choice; a probe miss > 8% demotes the
   model constants for that (model, config) and re-schedules — the same self-correction
   loop as §4 step 4.
```

### 8.3 Measured regime boundaries (today's constants)
| model | R1 zone (fallback) | R2 zone (gap) | R3 zone (ultra) |
|---|---|---|---|
| q3-32b | s ≤ ~96k (B_max≥3 sat.) | 96k → ~390k (uns wall) | > ~390k (asym-only to ~490k) |
| llama-70B | s ≤ ~96-112k | 112k → ~326k | > ~326k |
| q3-30b MoE | s ≤ ~64-128k | 128k → ~660k | > ~660k (asym-only) |

### 8.3b MEASURED R1 VALIDATION (2026-07-18, q3-32b) — the fallback is real
| point | R1 (asym|unsloth+staged) | sup unsloth | verdict |
|---|---|---|---|
| q3-32b 128k b2 | 1104 tok/s @116.0 GiB | 1110 @127.9 | parity, -12 GiB |
| q3-32b 160k b2 | 938 @144.8 | 942 @159.1 | parity, -14 GiB |
| q3-32b 192k b2 | 812 @175.3 | 816 (b1-capped) @96.4 | parity |
| llama 128k b2 | 786 @128.7 | 792 @147.7 | parity, -19 GiB |
| llama 192k b1 | **603 @96.3** | 601 @110.2 | **+0.3% (B<=0.92 rule), -14 GiB** |
| llama 192k b2 | 577 @182.7 (97.7%) | — (sup b2 OOM) | edge-tax -4% — rule's counterexample |
| q3-32b 384k b1 (R2/KA) | 426 @140.6 | 424 @98% edge | parity AT sup's wall |
CONVERGENCE LAW (3 independent pairs): at long seq, per-token converges across batch AND
config for every recompute-shaped system — the deficit is shared attention/regime physics.
⇒ R2's "bigger batch beats sup" premise is DEAD on dense; R2's real role is narrower:
fit the batch that R1 cannot (capacity), at parity per-token where possible. The composed
scheduler therefore delivers: parity everywhere sup lives + sole coverage beyond sup's
walls + memory superiority (HBM -12 GiB in R1; host-RAM story per config). Strict tok/s
wins on dense require shared-cost kernel work (C1a), out of scheduler scope.

### 8.4 Success criterion + current status (honest)
Target: composed(SCHEDULE) ≥ max(sup-unsloth, sup-recomp, memory-mode) tok/s ∀ (s,B).
- R1 zone: satisfied BY CONSTRUCTION once the R1 config lands (baseline-equivalent
  fallback; the fg-emission-OFF config = the asym-as-unsloth shape). IMPLEMENTATION GAP:
  R1 as a single config token does not exist yet — it is ohbm=FULL + KA + staged + the
  fg-emission kill; the S-mem-a work in fix_asym.md builds exactly that kill switch.
- R2 zone: TODAY asym trails ohbm0 by -8..-15% mid-window; S0 attribution says the gap
  is 120 us/tok of copy traffic (fix_asym S-mem a/b/c) + 46 kernels — recoverable ≥ the
  145 needed. At the far end of R2 the criterion ALREADY holds (384k parity 426 vs 424;
  640k MoE parity 732 vs 731).
- R3 zone: satisfied (asym runs where nothing else does; 424→426 handoff is seamless).
Convergence loop: land one fix_asym item → re-probe the R2 head-to-heads → move the
measured R1/R2 boundary left/right accordingly → re-emit this table. The scheduler is
DONE when the R2 row shows ≥ baselines at 128k/160k (dense) and 128k/192k (llama).

---

## 9. THE UNIFIED φ SCHEDULER (v4, 2026-07-19) — one continuous knob, measured frontier

### 9.0 The knob
φ ∈ [0,1] = fraction of offloadable bytes moved off HBM, admitted in MEASURED-PRICE order
(cheapest first). No user preference exists: HBM is always saturated up to SAFE = 0.92·M_phys
(above it the allocator-churn tax is measured: llama 192k b2 @97.7% = −4%). The scheduler
COMPUTES φ per workload:

    φ*(model, s, B) = clamp( (M₀(s,B) − SAFE·M_phys) / ΔM_offloadable(s,B), 0, 1 )
    B* = max{ B : M(φ*; s,B) ≤ SAFE·M_phys }     (batch = capacity-only; convergence law)

"Latency-focused vs memory-focused" is the computed φ*, not a setting.

### 9.1 THE LADDER — measured end-to-end at ONE fixed workload (q3-32b, 128k, b2)
Monotonicity test (2026-07-19, runs tputL1-L4 + prior R1/sup):
| φ rung (cumulative) | config token | resv GiB | us/tok | marginal ρ (us/tok/GiB) |
|---|---|---|---|---|
| 0. weights resident (ref) | superoffload|unsloth-ohbm0 | 127.9 | 901.0 | — |
| 1. + weights off | asym|unsloth-ohbm0 + staged (**R1**) | 116.0 | 906.1 | 0.4 (≈free) |
| 1b. roots graded check | asym|unsloth-ohbm8 (⅛ roots kept) | 133.1 | 904.9 | ≈0 (free, graded) |
| 2. + MLP fg-managed in HBM | recomp-off-full-fg + KEEP_ACTS (**R2/KA**) | 93.6 | 1044.1 | 6.2 |
| 3. + attention/recompute-tensors off | asym|unsloth-off-ohbm0 | 86.3 | 1158.7 | 15.7 |
| 4. + everything off (**R3**) | recomp-off-full-fg defaults | 54.8 | 1843.0 | 21.7 |

VERDICTS: M strictly ↓ (127.9→54.8), marginal price strictly ↑ (0.4 → 6.2 → 15.7 → 21.7) ⇒ Θ(φ) convex
⇒ "smallest φ that fits = fastest" is optimal, no oscillation, deterministic config emission.
MEASUREMENT-FORCED CORRECTION to the naive byte-class story: the frontier order is
weights → roots → **KA-MLP-management** → attention-off → full-MLP-offload — KA precedes
attention (KA both SAVES memory vs pure recompute −22 GiB AND costs less than attn-off).
Rung 3 (uns-off) is NOT dominated (leaner than KA, slower) — a true rung, kept.

### 9.2 Segment slopes (per-token memory k, GiB per 1k tok per sample, q3-32b)
R1 0.47-0.51 · KA 0.34 · memory ~0.31(dense)/0.17(MoE) — these + base_c give closed-form
regime boundaries: attn-onset s* where 0.47s+base>SAFE·M (≈350k dense b1), KA wall ≈480k,
memory wall ≈915k (L4 slope 0.175 measured; MoE ~1M). All match the measured record (384k ran KA; walls est).

### 9.3 Emission map (φ → env)
φ ≤ w-frac → R1 flags; + root-frac → UNSLOTH_GC_OUTER_HBM_EVERY_N grading; crossing KA
segment → recomp-off-full-fg + KEEP_ACTS_HBM=1 (+staged, dx-staged, AU); crossing attn →
save_on_cpu path; φ→1 → memory defaults.
GRANULARITY CORRECTION (owner-caught, 2026-07-19): the expensive segments (KA acts,
attention saved-tensors, MLP acts) are WITHIN-LAYER transients — produced in layer ℓ's
backward-recompute, consumed in the same layer's backward, so only ~one layer's transient
is live at once. Peak = window-max ⇒ per-layer %-offload does NOT grade the peak (20% vs
80% offloaded = same peak, more traffic = DOMINATED — never emitted). These segments are
inherently BINARY in peak; the earlier "offload first ⌈frac·L⌉ layers" idea is retracted.
True graded axes: roots (cross-layer liveness — ohbm-N measured proportional, ohbm8 =
+17 GiB), batch B, and chunk size g bounding the offloaded side's window (class-1 pinned
g=1024). ⇒ the realizable frontier = ~5 discrete rungs + graded roots + B; the φ equation
picks the cheapest rung that fits — stepping, not sliding, between expensive rungs, and
that discreteness is forced by window-max memory accounting (§2.1), not implementation.

### 9.4 ULTRA-LONG VALIDATION (2026-07-19) — the scheduler's own picks, predicted then run
| ctx|B | φ* selection (computed) | predicted resv | measured | err | result |
|---|---|---|---|---|---|
| 448k|1 | KA rung (R1 198✗ → KA 159✓) | 159 GiB | 164.2 (89%) | +3.3% | **380 tok/s, fits** |
| 576k|1 | memory rung (KA 203✗ → mem 111✓) | 111 GiB | **111.2 (60%)** | +0.2% | **245 tok/s, fits** |
sup-unsloth at both: DNF (wall ~390k). so-recomp: DNF (wall ~176k). END-TO-END: the φ
equation selected the regime, predicted the footprint within tolerance (8%), and both
ultra-long runs completed — dense q3-32b now measured to 576k ctx (3.3× sup's ceiling,
memory-rung wall ~915k est). System design VALIDATED: convex ladder (§9.1) + correct
regime selection + exact capacity prediction + ultra-long coverage.


## 10′. MERGE TOMBSTONE + RECORD MAP (grafted 2026-07-21; merge_scheduler.md is the ruling doc)

TOMBSTONED as runtime decision machinery (kept offline only, behind
`asym_scheduler.py --predict`): 42's β-dial (§9 v3.1 of the c14 fork) and the
water-fill allocator. WHY NOT: (a) τ fits are machine/day-flavored and were
q3-30b-only — Kevin ruled against µs-as-budget and for hardware-agnostic byte
thresholding (2026-07-20); (b) BEND/edge-penalty as decision input is
measured-WORSE (would have rejected llama T2 448k, a recorded FIT); (c) the
probe rule + byte lines reproduced every recorded decision (asym_scheduler.py
--replay). The merged runtime rule = feasibility-first over rung-prefix tiers
with the HOST term and knee-capped batch; see asym_scheduler.py + 
agent/impls/merge_scheduler.md §2/§3.

The c14 fork's §10 record map is preserved below verbatim (its prompt.md is
archived as agent/handoffs/prompt_v2_c14.md).

## 10. DESIGN RECORD MAP (2026-07-20) — this doc + three satellites = the full record

THIS FILE = the formulation of record (§0–7 dial, §8 families/fallback, §9 unified
formula + constants). The remainder lives in:
- **agent/handoffs/prompt.md** — interface evolution + FINAL FORM: v1 budget → **v2
  ALLOCATION/WATER-FILL (current: no user knob; batch+residency compete per
  marginal Δtok/s per GiB; modes = emergent labels)** → v3 hardware-agnosticity
  (mode selection = byte thresholding from model arch + device capacity; timing =
  optional 2-probe calibration of the fallback boundary only) + GPU validation log.
- **scripts/lf/asym_scheduler.py** — executable spec: fitted constants, water-fill,
  --sweep (regime table generator), --selftest (5 machine-checked cleanness
  properties: nested shedding, reverse-ρ shed order, monotone tok/s, reserved-HBM
  nesting, safety monotonicity).
- **agent/impls/s04-p1-dgx-02-c14/system_summary.md** — tier definitions (T1/T2/T3
  flag sets + measured prices), defensibility notes (what to claim vs downplay);
  **test_throughpout_v2.md** — the crossover-point evidence (P1–P5 + 1.6M max-seq)
  behind every constant.
Paper skeleton = §9 formulas + prompt.md v2/v3 framing + system_summary tiers +
the P1–P5/walls tables as the evaluation.
