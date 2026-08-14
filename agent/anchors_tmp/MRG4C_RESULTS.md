# mrg4c post-merge regression — ALL PASS (2026-08-14, c17, asym_sft_45, serial GPU0)

Merged tree @9ccb419 (mrg4b union + FULL-nvlink flavor flip + rebuilt _C).
House harvest: post-warmup effective tok/s from step_samples.csv (w1+m2);
resv = peak reserved HBM, rss = process peak (rank0_memstats.json).

| cell | eff tok/s | anchor | delta | resv | RSS | loss | anchor provenance |
|---|---|---|---|---|---|---|---|
| q3-30b-a3b asym T2 @320k b1 | 1369.7 | 1370 | -0.0% | 68.0G | 387G | 4.385 | c17 mrg4 chain 08-09 (same node) |
| glm4.7-flash asym trueT3 @256k b1 | 564.1 | 564 | +0.0% | 42.3G | 320G | 2.275 | c17 mrg4 chain 08-09 (same node) |
| q3.5-35b-a3b asym T2 @896k b1 | 1486.3 | 1492 | -0.4% | 172.3G | 459G | 0.735 | c18 fig-row campaign 07-24 (cross-node) |
| glm4.5-air asym T1 @320k b1 | 509.6 | 510 | -0.1% | 168.8G | 804G | 6.023 | c18 glmext 08-05 (cross-node; arena 240) |

Notes:
- q35b chain verdict said FAIL — the KNOWN q3.5-35b harness quirk (hybrid
  GatedDeltaNet auto-disables part of attn-act offload -> profile
  completeness validator rejects profile.json; jobs.tsv failed:1 while the
  run is complete). Training ran to completion (33m51s); banked from
  step_samples per the 08-12 protocol in
  agent/impls/archive/fix_dynamic_ep_39tree.md. Harness fix still queued.
- air 320k = beyond-ctx RoPE cell; 91%-HBM near-ceiling reproduced
  cross-node within -0.1%.
- 1r cells never enable ASYM_EP_SEP -> the sepplanlink flavor flip is inert
  here (its correctness gate = the PR5 bitwise trio, banked in 54acb99).
  What these cells DO exercise: rebuilt _C (csrc assert widening),
  frozen_linear pre-gate-first + dispatch gating (disabled path), lf.py
  rotary component classify, driver/case merges, dataset registry union.
Verdict: the 08-14 4-way merge introduces NO regression in memory, latency,
or throughput on any of the four requested MoEs at near-ceiling.
