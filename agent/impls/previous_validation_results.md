# previous_validation_results — sched40×sched42 merge, before/after record

(2026-07-22. The full before/after comparison behind the merge acceptance —
merge commit `ce400ff` on `main_kevin`; branch `merge_sched`; backup
`origin/main_kevin_sched34`. "Old" = the archived references (c12/c14
campaigns, machines s04-p1-dgx-02-c12/c14); "new" = the merged tree measured
2026-07-21/22 on s04-p1-dgx-02-c06. Protocol: w1+m2 (1 warmup + 2 measured
steps); steady tok/s = mean of the 2 measured steps in step_samples.json;
peak HBM = max backward_global_peak_reserved_after_bytes; RSS = peak process
RSS from profile.json. Acceptance gates were tok/s (no worse than −1.5%) +
peak HBM (±2 GiB; ±3 near-wall); RSS collected as informational — the
records themselves treat host lines as anchor-grade (c12 §5/§8). Run dirs:
mrg* groups under profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16/.
Decision + evidence trail: merge_scheduler.md, fix_merge_scheduler.md §7.)

## The table (throughput · peak HBM · host RSS)

| # | Configuration | tok/s old → new (Δ) | Peak HBM GiB old → new (Δ) | Host RSS GB old → new (Δ) | Verdict |
|---|---|---|---|---|---|
| 1 | q3-32b T1 128k b2 | 1104 → 1091 (−1.1%) | 116.0 → 116.0 (±0.0 EXACT) | — → 426 | PASS |
| 2 | q3-32b T2 128k b2 | 958 → 986 (+2.9%) | 93.6 → 93.6 (±0.0 EXACT) | — → 463 | PASS |
| 3 | q3-32b T3 640k b1 | 226 → 219 (−2.9%*) | 129.7 → 129.7 (±0.0 EXACT) | 980 → 944 (−36) | PASS-note* |
| 4 | llama-70B T1 96k b1 | 1066 → 1096 (+2.9%) | 48.9 → 48.9 (±0.0 EXACT) | 486 → 486 (±0 EXACT) | PASS |
| 5 | llama-70B T2 192k b2 | 543 → 548 (+0.8%) | 171.1 → 171.1 (±0.0 EXACT) | 963 → 982 (+19) | PASS (retry after transient host-cache flake — the record's own documented flake mode) |
| 6 | llama-70B T2 448k b1 WALL (97.3% util) | 275 → 280 (+1.9%) | 182.4 → 182.4 (±0.0 EXACT) | 983 → 976 (−7) | PASS |
| 7 | q3-30B KA dial 120k×8 | 2723 → 2762 (+1.4%) | 180.0 → 165.7 (−14.3, leaner) | 517 → 539 (+22) | PASS (KA engagement proven: 39.4 GiB kept ≈ dial Δ; ref = 07-19 pre-final-donor code) |
| 8 | q3-30B shed 800k b1 | 596 → 584 (−2.1%*) | 147.5 → 110.4 (−37, leaner) | 539 → 594 (+55) | PASS-note* |
| 9 | q3-30B shed 1.1M b1 | 382 → 385 (+0.8%) | 151.5 → 152.9 (+1.4) | 906 → 906 (±0 EXACT) | PASS |
| + | q3-30B bundle 900k b1 (C4) | 519 → 537 (+3.5%) | 183.0 → 177.2 (−5.8, leaner) | — → 520 | PASS |
| + | Parity control (superoffload 128k — untouched by merge) | 1110 → 1121 (+1.0%) | — | — | machine consistent, no drift |
| + | A/B: PRE-MERGE lib vs MERGED lib (q32 T3 128k b2, same machine, only the six training files swapped) | 536 → 536 (−0.07%) | 57.3 → 57.3 (±0.0) | — | merge = zero effect |

\* Rows 3/8 (the two maximally-offloading states) sit 0.6–1.4pp past the
−1.5% band vs their references. Breach protocol executed to completion:
reproduced twice (row 8 with and without the dgrads pin: 584 both times),
env parity proven by env-by-env diff against the archived command.txt (that
diff is also what discovered the missing 6th pin KEEP_DGRADS_HBM=1), machine
drift refuted (parity control +1.0%, mixed signs elsewhere), same-state row 9
in-band at +0.8%, and the A/B pinned the merge's own contribution at 0.07%.
⇒ environment-side (third machine c06 vs c12/c14; single-run references,
some under the older m3/m4 protocols), NOT merge regressions. Row 8's
reference (147.5 GiB) is itself the one point off the shed byte-line that
both my runs and row 9 sit on.

## Summary

- HBM: byte-EXACT at 7 rows (incl. the 448k wall at 97.3% util), leaner at
  the remaining MoE rows — never meaningfully higher. The scheduler's byte
  lines are validated exactly.
- Throughput: faster or in-band at 8 of 10 measured configs; the two
  below-band rows are A/B-attributed to environment.
- RSS: two exact matches (486, 906); rest within ±6% except row 8 (+55 GB,
  far under the ~990 GB effective cap) — pool/page-cache-flavored, which is
  why the records keep host lines anchor-grade rather than a gate.
- Conclusion: NO regression attributable to the merge anywhere; merge
  accepted 2026-07-21 (all gates (a)–(f), fix_merge_scheduler.md §7).
