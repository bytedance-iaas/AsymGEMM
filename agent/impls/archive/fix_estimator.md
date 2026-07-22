# fix_estimator — throughput (+capacity) estimator for the AsymGEMM backend

Running doc: goal, plan, progress, artifacts/metrics/logs.

## Goal
A trustworthy **throughput-vs-sequence-length curve** for the AsymGEMM training
backend, per config, so we understand its throughput behavior without re-running a
seq×batch grid. **Throughput is the PRIMARY (hard) target; capacity/max-seq is
secondary (easy).** Order: fix **q3-32b|1** first, then **q3-30b-a3b|1**, then others.

## Core idea (why this is tractable AND robust)
- We only ever care about **max-HBM (near-saturation) throughput**, never low-memory.
- Throughput = `tok/s = N / (t0 + c_g·N·(1+k·s))`, `N=B·s`. It **rises with batch then
  saturates at the knee `N* ≈ t0/c_g`**; past the knee tok/s is **batch-independent**
  → depends only on seq. (More batch past the knee = more HBM, ~0 tok/s gain; can DIP
  at OOM due to fragmentation.)
- So the **batch axis collapses** — for each seq, ONE past-knee point gives the
  saturated throughput. The **seq axis does NOT collapse** (attention term grows with
  s and extrapolates badly — round 1 was +70% off). ⇒ **measure the seq axis directly,
  model only the batch/capacity axis.**

### The knee is per (model × backend × mode/config)
`N*=t0/c_g`: `c_g` set by arch + engine (asym vs staged); `t0` set by the offload
strategy (CPU-Adam, async staging, ohbm, keep-acts, how much weight offloaded — the
SuperOffload lever: more offload → bigger t0 → knee pushes out). ⇒ calibrate the knee
per config. Same order of magnitude across a model's modes.

### SuperOffload reconciliation (why "bigger batch → more throughput" is also true)
That claim is the BELOW-knee regime: a less HBM-efficient system has small `N_max<N*`,
so it sits on the rising curve → offloading raises `N_max` → higher tok/s. AsymGEMM
offloads aggressively → huge `N_max` (round-1: q3-30b `N_max~1.4M` vs `N*~63k`, ~22×
past) → already saturated → can't buy more tok/s from batch. Model both regimes with
the FULL curve `tok/s = N_max/(t0 + c_g·N_max·(1+k·s))` (not just the asymptote).

## Plan (per config)
1. **Knee calibration — 2 runs.** Batch sweep at one seq (`{s}|1` + `{s}|8`) → fit
   `t0, c_g` → `N*`. Rule: past-knee batch at seq s = `max(1, round(2·N*/s))`.
2. **Direct throughput sweep — 1 run per seq.** For seqs of interest
   (~8k,16k,32k,65k,100k,174k), run one w1+m1 job at the **per-seq past-knee batch**
   (small N, cheap; batch DECREASES with seq — e.g. q3-30b: 8k|16,16k|8,32k|4,65k|2,
   100k|2,174k|1; q3-32b: mostly |1–2). Record **measured tok/s**. Interpolate between
   seqs; never far-extrapolate.
3. **Capacity (modeled, rides along).** Fit alloc `M=base+slope·N` from the same runs
   → `N_max` → `B_max(s)=N_max/s`. Use alloc (well-behaved) + ~10% margin off reserved.
4. **Validate** vs the already-measured ceiling runs (true max-HBM points):
   q3-32b@65k|8 = 166.8 GiB / 1281 s ; q3-30b@174k|8 = 170.9 GiB / 878 s.

Protocol: `PROFILERS=source`, `MAX_STEPS=2 WARMUP_STEPS=1`. Metric = measured s/it
(measured_elapsed/measured_steps), peak alloc/reserved HBM, CPU RSS, from profile.json;
τ(s) = s_it/N. Configs: q3-32b `asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm8`,
q3-30b `asym_cpuadamwds|recomp-off-full-fg-ker101-ceil0000-ohbm16`.

## Decision log
- Shortcut far-extrapolation of throughput: REJECTED (round-1 q3-30b +70%; attention
  term doesn't extrapolate). Full seq×batch grid: unnecessary (batch collapses past
  knee). Chosen: measure seq axis directly (1 past-knee run/seq), model capacity only.

## Progress
### Round 1 (2026-07-14) — 3 short anchors/model, analytic-k. Memory good, throughput bad.
Anchors (s/it, alloc, reserved GiB, rss GB):
```
q3-32b     s= 8192 B=1 N=  8192  s/it= 15.8  A= 3.0 R= 4.7 rss=160
q3-32b     s= 8192 B=8 N= 65536  s/it= 90.4  A=19.6 R=22.6 rss=244
q3-32b     s=16000 B=8 N=128000  s/it=174.9  A=38.1 R=44.4 rss=339
q3-30b-a3b s=12288 B=1 N= 12288  s/it= 13.1  A= 8.5 R=13.5 rss=278
q3-30b-a3b s=12288 B=8 N= 98304  s/it= 35.5  A=16.7 R=20.0 rss=314
q3-30b-a3b s=24000 B=8 N=192000  s/it= 67.5  A=26.0 R=28.3 rss=355
```
Predict ceiling vs measured (err): q3-32b@65k s/it 983 vs 1281 (-23%), alloc -1%,
reserved +4%, rss +5%. q3-30b@174k s/it 1493 vs 878 (+70%), alloc -7%, reserved -26%,
rss +5%. ⇒ alloc+rss extrapolate well; reserved unreliable long-seq (use alloc+margin);
throughput must be MEASURED, not extrapolated. Fit knees: q3-32b N*~6.3e3, q3-30b ~6.3e4.

### Round 2 (next): direct throughput sweep per plan, q3-32b first. [pending]
