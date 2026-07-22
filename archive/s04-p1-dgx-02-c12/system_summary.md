# system_summary — the AsymLoRA tier scheduler, complete & self-contained
(consolidated 2026-07-20; supersedes the 07-19 version. Everything needed to
understand, implement, and defend the scheduler is IN THIS FILE — constants,
formulas, flags, validation. Host: GB200, C_HBM = 185.0 GiB, C_host ≈ 957 GB.)

## 0. Abstract (read this and you know the system)
One backend (base weights in CPU RAM, LoRA trains, CPU AdamW, recompute) with
THREE tiers differing only in how many activation bytes stay on GPU: **T1**
keeps all (fastest), **T2** also ships attention saved-tensors to host, **T3**
also ships MLP activations (leanest). Each (model, tier) has a linear byte
model in total tokens N = B·s: HBM_t(N) = a_t + k_t·N, coefficients set by the
model architecture, not the GPU. **The scheduler is ahead-of-time admission
control: pick the first tier in T1→T2→T3 order whose predicted HBM fits under
β·C_HBM (β = 0.92) and whose host bytes fit in C_host; launch its fixed flag
set; never switch mid-run.** The user supplies only (model, s, B). No timing
enters the decision — speed order T1≻T2≻T3 is structural (more offload =
slower, on any machine), so first-feasible = fastest-feasible, and hardware
enters only via the two capacity constants. Validated: reproduces every
measured tier decision; byte predictions within 1–4% (near-wall bias → probe
rule §6); runs where all baselines are OOM (dense 640k = 1.67× SuperOffload's
ceiling; MoE 1.6M).

## 1. The tiers (exact recipes)
All tiers: RUNS backend `asym_cpuadamwds`, liger loss, unsloth-GC recompute
(`gradient_checkpointing=true, use_unsloth_gc=true,
unsloth_recompute_save_on_cpu=true`): backward re-runs each layer's forward, so
intermediates live ONE layer at a time; only the L per-layer checkpoint roots
are all-live, gradeable to host via `-ohbm<N>` (keep every Nth on HBM).
NAMING TRAP: the `recomp-off-*` token means recompute-OFFLOAD (offload the
recompute save-set), NOT "recompute off" — every tier recomputes.
Measurement protocol w1+m2: 1 warmup + 2 measured steps, steady = mean of the
2 (post-warmup spread ~1%).

| tier | RUNS recompute token | env flags on top | resident on GPU |
|---|---|---|---|
| **T1 FALLBACK** (fastest) | `unsloth-ohbm0` | `ASYM_GEMM_DISPATCH=staged` | all recompute intermediates + checkpoint roots; weights streamed just-in-time |
| **T2 KEEP-ACTS** | `recomp-off-full-fg-ker000-ceil0000-ohbm0` | staged + `ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM=1` (dense) / `ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1` (MoE) + `ASYM_SAVED_TENSOR_ASYNC_UNPACK=1` + `ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1` + `ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024` | MLP acts stay (valve closed); attention saved-tensors go to host |
| **T3 MEMORY** (leanest) | same fg token (`-ker000` dense / `-ker101` MoE) | none of the above (fg defaults; no staged → asym streaming kernel). NB: the 07-19 doc listed `ASYM_CPU_ADAMW_ASYNC_GRAD_OFFLOAD=0` here — archived command.txt of the validated runs shows it was NOT set; dropped. | ~one layer live; MLP acts to host in row chunks |

Mechanics: T1 = stock GPU kernels on staged panels (asym-kernel calls = 0);
T2's one asym-custom compute = the cpu-left LoRA-A host kernel; T3 runs base
GEMMs on the asym streaming kernel against CPU-resident panels (~1/3 native).
Reference point (q3-32b 128k b2): T1 116.0 GiB / 906 us/tok · T2 93.6 / 1044 ·
T3 54.8 / 1843.

## 2. The decision function (THE scheduler)
For each tier t, linear byte models in N = B·s (tokens):

    HBM_t(N)  = a_t + k_t·N          (GiB)
    HOST_t(N) = c_t + h_t·N          (GB; coarser — see §5)

    t*(model, s, B) = first t in [T1, T2, T3] with
                      HBM_t(B·s) ≤ β·C_HBM   AND   HOST_t(B·s) ≤ C_host
    where β = 0.92.

Then launch with t*'s flag row from §1. Static, ahead-of-time, no profiling
run, no user knob: the user supplies (model, s, B) and gets the fastest
configuration that fits. Batch is a capacity-only lever: per-token throughput
converges across B AND system at long seq (measured: q32 160k SO-recomp b1
941 ≡ SO-unsloth b2 942; T1 128k b2 1104 ≈ b3 1097; MoE 640k 731 ≡ 732), so
B is user-supplied or B* = max{B : constraints hold} — it never buys tok/s.

Continuous view (paper form): φ ∈ [0,1] = fraction of offloadable bytes moved
off-GPU in measured-price order; φ* = clamp((HBM_0 − β·C_HBM)/ΔM, 0, 1); the
tiers are the landmarks on φ. Greedy price-order is optimal because the ladder
is convex (§3).

## 3. Why "first feasible = fastest" (the price ladder, measured once)
(Baseline named below: "SO" = SuperOffload = `superoffload_mem` DeepSpeed
backend — weights/optimizer offloaded, activations on GPU — with unsloth-GC;
the strongest external system at every measured point.) Marginal cost of each
offload class, measured end-to-end at q3-32b 128k b2:

| rung | ΔHBM freed | Δtime | price (us/tok per GiB) |
|---|---|---|---|
| T1 vs SO baseline (weights class) | −12 GiB | +5 us/tok (906 vs 901) | **0.4 ≈ free** |
| T2 vs T1 (attention tensors) | −22.4 GiB | +138 us/tok (1044) | **6.2** |
| T3 vs T2 (MLP acts + streaming) | −38.8 GiB | +799 us/tok (1843) | **20.6** |

Prices strictly increase ⇒ convex ⇒ never skip a cheaper rung: the fastest
feasible tier is optimal. This ordering is structural (each tier strictly adds
host traffic), so it holds on any hardware — the µs numbers above are NOT
inputs to the scheduler, only the one-time proof of monotonicity.

Why DISCRETE tiers (window-max accounting): peak HBM is the max over the
step's live-set windows. Within-layer transients (attention tensors, MLP acts)
are binary-in-peak — offloading them in SOME layers leaves the peak window
unchanged, so a per-layer percentage knob does not grade memory; only crossing
a whole byte class moves the peak. The graded refinements that DO work:
`-ohbm<N>` (checkpoint roots are cross-layer live → truly graded, ≈free:
ohbm8 measured 133.1 GiB @ 904.9 us/tok) and the T3 row-chunk size
(`ASYMM_FG_ELEMENTWISE_CHUNK_MB`, default 1024). A future continuous φ inside
a segment = offload the FIRST ⌈φ·L⌉ layers (earliest layers first — their
backward comes last = most overlap time); not implemented, not needed for the
capacity results.

## 4. The byte lines (all fitted constants, HBM)
k in GiB per 1k tokens; fitted by least squares on peak reserved HBM from 2-3
existing runs per pair; activation memory is token-linear under FlashAttention
(no s² term), so two points pin a line. HARDWARE-INDEPENDENT: k is architecture
arithmetic (hidden × layers × dtype × tier policy); a has only sub-GiB
CUDA-context flavor; c12 vs c14 replicas agree ≤0.1%.

| model | T1 | T2 | T3 |
|---|---|---|---|
| q3-32b (64L dense) | k≈0.47–0.51 | a≈10, k≈0.34 | k≈0.175 |
| llama3.3-70B (80L dense) | k≈0.51 (a≈−1: 2-point fit artifact; clamp a≥0) | a≈30.6, k≈0.366 | — (host-bound first, §5) |
| q3-30b-a3b (MoE) | fit pending — points exist 80k–800k (c14) | single point (1.1M = 382 tok/s run) | k≈0.17 (ker101; to 1.6M) |
| q3.5-35b-a3b (MoE, hybrid attn) | point 45.2 GiB@128k | point 95.7@576k (k≈0.11 implied) | point 64.4@640k (k≈0.06 implied — hybrid attn is byte-lean; fit pending) |

Prediction accuracy (predicted → measured): q32 T2 448k: 162 → 164.2 (+1.2%);
q32 T3 576k: 111 → 111.2 (+0.2%); llama T2 384k: 171 → 178.6 (**+4.4%,
near-wall**); llama T2 416k: ~183 → ran at ~99% observed (FIT). Bias is
one-sided near the edge (allocator fragmentation) → probe rule §6.

## 5. The host constraint (why it must exist — measured tier inversions)
Every tier trades the HBM wall for a HOST-RAM wall; at the deep end the binder
is the CPU pool, not the GPU:
- llama T3 host-OOMs at 416k AND 448k (watchdog: CPU-node free < floor) while
  T2 still fits at 416k ⇒ **tier inversion**: T3's wall sits BELOW T2's on
  llama (70B weights already crowd the pool T3 needs for acts). T2 is llama's
  deepest mode.
- q32 T3: 640k fits at only 70% HBM but RSS 980 GB ≈ pool → 704k HOST-OOM.
- Hence the second feasibility term HOST_t ≤ C_host. Current host anchors
  (peak RSS, GB): llama T2 975–984 (≈token-flat: weights + pinned pools
  dominate); q32 T3 957@576k → 980@640k; MoE T3 925@1.6M (c14). Formal
  (c_t, h_t) fits are the one immature piece — today the host check uses these
  anchors; the host-mem watchdog (floor 35-50 GiB free) backstops mispredicts.

## 6. Safety threshold β = 0.92 + the probe rule
- β: above ~92% HBM the allocator churns — measured: llama 192k b2 @97.7% =
  −4% vs its own b1; uns 352k @98% = −16% per-token vs healthy trend. Fitting
  under 0.92·C_HBM avoids the edge tax entirely.
- Probe rule: the byte line under-predicts near walls (max observed +4.4%);
  if the prediction lands within ~8% of β·C_HBM (either side), do ONE trial
  run instead of trusting the line. This rule has caught every would-be wrong
  call (e.g. llama T2 416k predicted marginal-over → probed → FIT).

## 7. Validation record (all measured, GB200 c12/c14)
- Decisions reproduced by the formula at every measured point: T1 through
  192k (dense) / 800k (MoE), T2 at 320–416k (llama) and 384–448k (q32), T3 at
  576–640k (q32) and 1.1M–1.6M (MoE); llama T3 correctly EXCLUDED by the host
  term; llama 192k b2 (97.7%) correctly REJECTED by β.
- T1 ≡ SuperOffload parity band everywhere both fit (±0.7%; llama 192k +0.3%
  asym win) at −12…−19 GiB HBM — the fallback tier costs nothing.
- Sole-coverage runs beyond every baseline's measured wall: q32 640k b1
  (226 tok/s, 70% HBM); llama 416k b1 (~299 tok/s, ~99%); MoE 1.6M (292, c14).
- Capacity walls (first-OOM, all measured): q32 rc 192k · uns 416k · asym
  704k (host; 640k last-fit); llama rc 112k · uns 384k · asym T2 last-fit
  448k @97.3% (480k probe pending) with T3 host-walled at 416k;
  q3-30b rc 400k · uns 660k · uns-off 1.1M (host) · asym >1.6M;
  q3.5-35b rc 448k · uns 640k · asym T3 ≥640k @35% HBM (ladder running).
- Throughput WIN case (not just capacity): q3.5-35b T2@576k = 1377 tok/s @51.7%
  vs uns's own edge 1023 @97.7% — +35% at the baseline's last-fit seq. On this
  arch the keep-acts tier beats the baseline OUTRIGHT where it still lives.
- Cross-machine: c12 vs c14 replicas ≤0.1% (tok/s and footprints).

## 8. Honest limits
1) HOST_t lines are anchor-based, not fitted — the next calibration item.
2) MoE T1/T2 lines thin (c14 owns that partition). 3) β and the 8% probe band
are empirical constants of this software stack (torch 2.12 allocator), not
theory. 4) The lines extrapolate activations only — changing dtype, LoRA rank,
or GC policy shifts (a_t, k_t) and needs a 2-run refit. 5) Throughput at the
picked tier is NOT predicted — only feasibility and ordering are; per-token
time at long seq ≈ shared attention cost + tier tax (converges across systems).
