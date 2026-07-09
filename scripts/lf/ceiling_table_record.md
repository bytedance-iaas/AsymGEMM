# Ceiling Table — Record (measured + estimated)

Generated 2026-07-09 05:21. Extends ceiling_table.md with **estimated** `superoffload_mem|unsloth` rows (no artifacts yet).

Estimation basis for `(est)` rows: ceiling = manual G-OOM annotation minus 3k (the .sh prior convention: G-OOM 46k/50k/81k -> 43k/47k/78k), ohbm0 (HBM-bound); `c_g = 1.30 x` same-model unsloth-off c_g (recompute adds ~1 extra forward ~= +33% FLOPs, partly offset by less offload traffic); same k, t0. `e` = estimated cell, `a` = measured anchor.

## Capacity + latency + throughput

| model | backend | config | 8k | 12k | 16k | 32k | 48k | 64k | 128k |
|---|---|---|---|---|---|---|---|---|---|
| llama3.3-70b | asym_cpuadamwds | recomp-off-full-fg-ker000-ceil0000-ohbm3 | 34 / . (.) | 22 / . (.) | 17 / . (.) | 8 / . (.) | 5 / . (.) | 4 / . (.) | 2 / . (.) |
| llama3.3-70b | superoffload_mem | unsloth-off-ohbm0 | 32 / 297s (862) | 21 / 302s (834) | 16 / 317s (807) | 8 / 358s (716)a | 5 / 374s (642) | 4 / 439s (583) | 2 / 601s (426) |
| llama3.3-70b | superoffload_mem | unsloth-ohbm0 (est) | 43 / 516s (667)e | 28 / 521s (644)e | 21 / 539s (624)e | 10 / 579s (553)e | 7 / 677s (496)e | 5 / 711s (450)e | 2 / 780s (328)e |
| q3-30b-a3b | asym_cpuadamwds | recomp-off-full-fg-ker101-ceil0000-ohbm0 | 173 / 293s (4,724) | 115 / 319s (4,327) | 86 / 345s (3,992) | 43 / 451s (3,049) | 28 / 545s (2,465) | 21 / 649s (2,069) | 10 / 1,016s (1,260) |
| q3-30b-a3b | superoffload_mem | unsloth-off-ohbm0 | 131 / 138s (7,612) | 87 / 150s (6,964) | 65 / 162s (6,417) | 32 / 210s (4,884) | 21 / 256s (3,942) | 16 / 310s (3,306) | 8 / 510s (2,008) |
| q3-30b-a3b | superoffload_mem | unsloth-ohbm0 (est) | 78 / 107s (5,815)e | 52 / 117s (5,323)e | 39 / 127s (4,908)e | 19 / 163s (3,739)e | 13 / 206s (3,023)e | 9 / 227s (2,533)e | 4 / 333s (1,539)e |
| q3-32b | asym_cpuadamwds | recomp-off-full-fg-ker000-ceil0000-ohbm8 | 65 / 1,038s (501) | 43 / 1,068s (483) | 32 / 1,097s (467) | 16 / 1,246s (411) | 10 / 1,309s (367) | 8 / 1,545s (331) | 4 / 2,142s (239) |
| q3-32b | superoffload_mem | unsloth-off-ohbm4 | 53 / 315s (1,347) | 35 / 323s (1,300) | 26 / 331s (1,255) | 13 / 377s (1,105) | 8 / 390s (986) | 6 / 431s (890) | 3 / 598s (642) |
| q3-32b | superoffload_mem | unsloth-ohbm0 (est) | 47 / 362s (1,038)e | 31 / 372s (1,001)e | 23 / 381s (967)e | 11 / 414s (850)e | 7 / 443s (759)e | 5 / 467s (685)e | 2 / 519s (493)e |

## Throughput (tok/s) at fixed sequence length

| model | backend | config | 8k | 12k | 16k | 32k | 48k | 64k | 128k |
|---|---|---|---|---|---|---|---|---|---|
| llama3.3-70b | asym_cpuadamwds | recomp-off-full-fg-ker000-ceil0000-ohbm3 | . | . | . | . | . | . | . |
| llama3.3-70b | superoffload_mem | unsloth-off-ohbm0 | 862 | 834 | 807 | 716a | 642 | 583 | 426 |
| llama3.3-70b | superoffload_mem | unsloth-ohbm0 (est) | 667e | 644e | 624e | 553e | 496e | 450e | 328e |
| q3-30b-a3b | asym_cpuadamwds | recomp-off-full-fg-ker101-ceil0000-ohbm0 | 4,724 | 4,327 | 3,992 | 3,049 | 2,465 | 2,069 | 1,260 |
| q3-30b-a3b | superoffload_mem | unsloth-off-ohbm0 | 7,612 | 6,964 | 6,417 | 4,884 | 3,942 | 3,306 | 2,008 |
| q3-30b-a3b | superoffload_mem | unsloth-ohbm0 (est) | 5,815e | 5,323e | 4,908e | 3,739e | 3,023e | 2,533e | 1,539e |
| q3-32b | asym_cpuadamwds | recomp-off-full-fg-ker000-ceil0000-ohbm8 | 501 | 483 | 467 | 411 | 367 | 331 | 239 |
| q3-32b | superoffload_mem | unsloth-off-ohbm4 | 1,347 | 1,300 | 1,255 | 1,105 | 986 | 890 | 642 |
| q3-32b | superoffload_mem | unsloth-ohbm0 (est) | 1,038e | 1,001e | 967e | 850e | 759e | 685e | 493e |

Confidence: measured rows = artifact-fitted; `(est)` rows are best guesses — one 4-step anchor run per unsloth config (~20-30 min each) upgrades them to measured.
