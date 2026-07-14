# The Mode Dial — measured latency↔memory interpolation (knob record)

2026-07-10. Steady protocol (1 warmup + 4 measured, middle-2 steady) except where noted.
Full axis catalog + composition rules: `agent/scheduler.md` §2.5. Run ledger:
`agent/impls/fix_throughput.md` (Phase D, D8–D13).

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
