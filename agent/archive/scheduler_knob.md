# The Mode Dial — measured latency↔memory interpolation (ARCHIVED metrics record)

> ARCHIVED SNAPSHOT (metrics only). The live/working doc is `agent/scheduler_v2.md`
> (formulation + records). This file is a frozen record of the measured per-model
> dials and the A1–A14 axis-price ledger; not maintained going forward.

2026-07-10. Steady protocol (1 warmup + 4 measured, middle-2 steady) except where noted.
Run ledger: `agent/impls/fix_throughput.md` (Phase D, D8–D13).

## Headline: latency mode DOES use more peak HBM — keep-acts is the only flag paying it

```text
                      memory mode   latency mode    delta      vs baselines
q3-30b-a3b @80k×8     80.1 GiB      84.7 GiB        +4.6       still UNDER so-off (94.4); unsloth 176.9
q3-32b     @49k×8     95.5 GiB      132.2 GiB       +36.7      ABOVE so-off (108.5), under unsloth (177.3)
speed for that delta  273.5→175.7 s (1.56×)  |  964.5→277.6 s (3.5×)
```

- The delta = the kept per-layer activation set (consumed within the same layer's
  backward). Thin on MoE experts, 20 GB/tensor on dense — and it grows with seq, so
  near the ceiling the fit-check fails and the config degrades to memory-mode
  footprint automatically (that IS the s* rule).
- Everything else in latency mode (staged dispatch, ker choice, async unpack/staging)
  is memory-free on HBM (0 bytes measured down to a 3 GiB margin); async grad staging
  costs +7 GB HOST only — hence its kill-switch in memory mode.
- Baseline bracket: MoE latency mode dominates so-off on BOTH axes; dense latency mode
  beats so-off on speed (1.34×) while spending +24 GiB — a chosen point, not a loss:
  dial back keep-acts and you're at 95.5 GiB / 836.7 s.

## q3-30b-a3b @ 80000|8|1, ohbm0 — the dial, latency → memory (remove one flag per row)

```text
dial point                                    s/step   tok/s   peak HBM   what the step trades
LATENCY MODE (staged+ker000+keep-acts+async)  175.7    3,642   84.7 GiB   —
 - keep-acts  (ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=0)
                                              200.0    3,200   80.0       +24.3 s buys −4.7 GiB HBM
 - ker000 → ker101                            227.6    2,813   80.1       +27.6 s buys ~0 (kernel pick, follows dispatch)
 - staged → asym engine (ASYM_GEMM_DISPATCH)  273.5    2,340   80.1       +45.9 s buys ~0.4 GB transient (zero-transient safety)
MEMORY MODE (+ASYM_CPU_ADAMW_ASYNC_GRAD_OFFLOAD=0)
                                              ~278     ~2,300  80.1       −7 GB HOST (pinned staging off)
references @ same workload:
superoffload_mem|unsloth-off-ohbm0            228.1    2,806   94.4
superoffload_mem|unsloth-ohbm0 (hist)         186.9    3,424   176.9
```

The entire asym dial sits LEFT of both baselines in memory; its fast end beats both in
speed. Loss parity held on every arm.

## Knob semantics (what actually interpolates)

- **keep-acts** is the only flag that trades HBM for time (+4.7 GiB ⇒ −24 s @80k).
  Transient ~linear in s (10 GB@32k → 29 GB-class @80k naive, +4.7 measured in-stack —
  peaks don't add across time). Auto-off above s* (fit-check vs headroom).
- **staged** and **ker-follows-dispatch** are ~free speed (0 HBM measured down to a
  3 GiB margin, D8) — on in BOTH modes unless margin is sub-GB (asym engine = the
  zero-transient fallback). ker rule: asym dispatch→101, staged→000 (fp32-accum caveat).
- **async flags** (grad staging, GC unpack): latency-side hygiene; ~0 time @32–80k
  (shadowed), the grad staging costs +7 GB pinned HOST ⇒ off in memory mode (the D9 fix).
- **ohbm N** is the separate HOST↔HBM wall balancer (roots ~5 GB/layer): q3-32b @65k:
  ohbm0 = 125.3 GiB HBM / 892 GB host (host-bound) vs ohbm8 = 163.8 / 886 (HBM-bound);
  68k@ohbm8 passes (161.0 / 598). Pick N to equalize distance-to-wall on both budgets.

## Dense (q3-32b @49000|8|1, ohbm0) — the dial (port landed 2026-07-10)

```text
dial point                                    s/step   peak HBM   what the step trades
DENSE LATENCY MODE (staged + keep-acts)       277.6    132.2 GiB  —
 - keep-acts (ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM=0)
                                              836.7     95.5      +559 s buys −36.7 GiB (!!)
 - staged → asym engine                       964.5     95.5      +128 s buys ~0
references:
superoffload-off / unsloth                    373.1 / 221.0   108.5 / 177.3
```

The dense keep-acts lever is 20× the MoE one (+559 s vs +24 s — its tensors are
[392k,25600] = 20 GB each; nsys showed the round trips = 69% of backward). Dense
latency mode is 1.34× FASTER than so-off at +24 GiB (and 45 GiB leaner than unsloth
while 56 s slower). Unlike MoE@80k, dense keep-acts at 49k EXCEEDS so-off's memory —
i.e. 49k is near dense s*: at smaller s the +37 GiB shrinks and stays under; near the
ceiling it must turn off. Loss parity exact across all arms (Δ ≤ 0.0013).
Implementation: `ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM` (dense_mlp_finegrained.py — the
MoE _HBMKeepManager shim + is_cuda reroutes in _cpu_left_lora_a/_cpu_right_lora_a_grad;
subsumes all four MoE levers on dense; guard vs CPU_ACT; bit-exact toy parity).

## llama3.3-70b (dense) dial — post-merge verification (2026-07-14, main_kevin)

@32000|8|1 ohbm0, PROFILERS=source w1+m1. Confirms the dense dial generalizes to
llama3.3-70b and both goals vs superoffload hold on merged main_kevin.

| mode | alloc HBM | reserved | s/it | loss | note |
|---|---|---|---|---|---|
| memory (dense-fg, keep-acts off)        | 67.0  | 82.7  | 476 | 1.207 | −23 GiB vs so; +39% time |
| latency (staged + ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM=1) | 111.0 | 135.2 | 199 | 1.206 | 1.72× faster than so; +21 GiB |
| superoffload_mem\|unsloth-off (control) | 90.3  | 101.5 | 343 | 1.206 | baseline (record: G-95 / 359 s) |

- GOAL 1 met: memory mode beats superoffload on HBM (67 vs 90 GiB alloc).
- GOAL 2 met: latency mode beats superoffload on speed (199 vs 343 s = 1.72x).
- Smooth transition: memory->latency trades +44 GiB HBM for -277 s/step (monotone).
- Loss parity across all three (1.206-1.207); zero torch fallback on asym modes.

## Sequence-capacity ceilings (post-merge, 2026-07-14, PROFILERS=source w1+m1)

Near-ceiling runs that FIT on merged main_kevin (of ~185 GiB GB200):

| run | alloc | reserved | CPU RSS | s/it | verdict |
|---|---|---|---|---|---|
| q3-32b @65k ker000 ohbm8      | 154.2 | 166.8 GiB | 882 GB | 1281 | FITS (record C-OOM ~66k) |
| q3-30b-a3b @174k ker101 ohbm16 | 153.6 | 170.9 GiB | 825 GB | 878  | FITS (record 172k max-OK / G-OOM 188k) |

## A1–A14 axis-price ledger (from the retired scheduler.md §2.5)

The scheduler's raw material. Each axis: what it moves, which budget it charges, the
measured price tag (q3-30b 80k×8 ohbm0 anchor unless noted), and its mode assignment.
(2026-07-10; sources: Phase D/D11 ledger in `agent/impls/fix_throughput.md`, paper.md
historical tables, D8/D9/D10 probes.) In `scheduler_v2.md` these are condensed into the
D1–D6 frontier table + the class-1 pins; this is the full 14-axis detail.

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
