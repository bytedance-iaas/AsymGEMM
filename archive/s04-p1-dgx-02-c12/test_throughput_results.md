# test_throughput_results — PURE METRICS

RECONSTRUCTED 2026-07-17 from session record after commit 01c56d8 deleted this file
(working-tree state was never committed; do not delete without committing first).

HOST: s04-p1-dgx-02-c12 | GPU: GB200 (189471 MiB = 185.0 GiB HBM) | container asym_sft_40, venv torch 2.12.
(Cross-machine variance: other hosts record under agent/impls/<hostname>/ — see agent/impls/throughput_prompt.md.)
RAW ARTIFACTS (this host): profiling_results/profiling_tp_s04-p1-dgx-02-c12/asym_long_sft_smoke__lora__lf__bf16/
(tput*/tputrc*/tputasl* run dirs archived there post-campaign; in-flight runs live under profiling_results/profiling/... until moved.)

Auto-generated throughput tables. Metric defs & protocol: `test_throughput.md`.
Units: resv/%HBM in GiB (physical HBM=185.0 GiB); RSS in GB (host RAM).
MFU vs GB200 bf16 peak 2250 TFLOPS. `MAX`=peak tok/s at that seq; `thrash`=>=95% HBM + step blowup; `OOM`=out of memory.
`▼N%` = saturation deficit at B_max = 1 - MFU(B_max)/plateau (gap point: fits but under the knee).
Measured: steady s/it (mid of w1+m4), PROFILERS=source. N_active: q3-32b 32.8e9, q3-30b-a3b 3.34e9, llama3.3-70b 70.6e9.

## q3-32b — superoffload_mem | unsloth-ohbm0
| seq | B | s/it | resv GiB | %HBM | RSS GB | tok/s | MFU% | flag |
|---|---|---|---|---|---|---|---|---|
| 8000 | 24 | 58.0 | 96.1 | 52% | 227 | 3312 | 31.3 |  |
| 8000 | 32 | 75.4 | 128.0 | 69% | 364 | 3397 | 32.1 |  |
| 8000 | 36 | 84.2 | 143.1 | 77% | 364 | 3421 | 32.3 |  |
| 8000 | 40 | 93.2 | 161.0 | 87% | 364 | 3434 | 32.4 |  |
| 8000 | 44 | 102.1 | 174.4 | 94% | 364 | 3447 | 32.6 | **MAX** |
| 12000 | 24 | 88.5 | 143.1 | 77% | 364 | 3255 | 31.9 |  |
| 12000 | 30 | 110.0 | 178.4 | 96% | 364 | 3274 | 32.1 | **MAX** |
| 12000 | 32 | 167.2 | 181.2 | 98% | 364 | 2297 | 22.5 | thrash |
| 12000 | 40 | OOM | 171 | 92% | 637 | — | — | OOM |
| 16000 | 16 | 82.4 | 128.0 | 69% | 364 | 3107 | 31.5 |  |
| 16000 | 20 | 101.9 | 161.0 | 87% | 364 | 3140 | 31.9 | **MAX** |
| 16000 | 24 | 162.6 | 181.2 | 98% | 364 | 2362 | 24.0 | thrash |
| 20000 | 16 | 106.5 | 161.0 | 87% | 364 | 3005 | 31.5 | **MAX** |
| 20000 | 20 | 177.9 | 181.4 | 98% | 364 | 2249 | 23.6 | thrash |
| 24000 | 8 | 67.8 | 96.2 | 52% | 227 | 2830 | 30.7 |  |
| 24000 | 12 | 100.4 | 143.1 | 77% | 364 | 2870 | 31.1 |  |
| 24000 | 14 | 116.3 | 168.9 | 91% | 364 | 2889 | 31.3 | **MAX** |
| 24000 | 16 | 180.8 | 181.4 | 98% | 364 | 2124 | 23.0 | thrash |
| 28000 | 8 | 82.4 | 112.1 | 61% | 364 | 2718 | 30.4 |  |
| 28000 | 10 | 101.7 | 139.7 | 76% | 364 | 2754 | 30.8 |  |
| 28000 | 12 | 121.2 | 168.9 | 91% | 364 | 2772 | 31.0 | **MAX** |
| 30000 | 8 | 89.8 | 120.3 | 65% | 364 | 2672 | 30.4 |  |
| 30000 | 10 | 110.9 | 149.0 | 81% | 364 | 2706 | 30.8 |  |
| 30000 | 12 | 132.6 | 178.4 | 96% | 364 | 2714 | 30.9 | **MAX** |
| 32000 | 8 | 95.6 | 128.0 | 69% | 364 | 2679 | 30.9 |  |
| 32000 | 10 | 118.5 | 161.0 | 87% | 364 | 2700 | 31.2 | **MAX** (b12 unprobed) |
| 36000 | 6 | 86.2 | 108.2 | 58% | 364 | 2507 | 29.8 |  |
| 36000 | 8 | 113.4 | 143.1 | 77% | 364 | 2540 | 30.2 |  |
| 36000 | 10 | 140.8 | 178.4 | 96% | 364 | 2556 | 30.4 | **MAX** |
| 40000 | 6 | 99.1 | 120.0 | 65% | 364 | 2422 | 29.6 |  |
| 40000 | 8 | 130.7 | 161.0 | 87% | 364 | 2449 | 30.0 | **MAX** |
| 45000 | 6 | 115.4 | 134.9 | 73% | 364 | 2339 | 29.7 |  |
| 45000 | 8 | 152.8 | 178.4 | 96% | 364 | 2356 | 29.9 | **MAX** (▼4%) |
| 56000 | 5 | 131.0 | 139.7 | 76% | 364 | 2138 | 29.2 |  |
| 56000 | 6 | 156.4 | 168.9 | 91% | 364 | 2149 | 29.3 | **MAX** (▼6%) |
| 96000 | 3 | 171.5 | 145.2 | 78% | 364 | 1680 | 28.8 | **B_max, gap ▼8%** (b4 ~193 OOM est.) |
| 128000 | 2 | 230.7 | 127.9 | 69% | 364 | 1110 | 22.1 | **B_max, gap ▼29%** — seq-driven overhead confirmed on dense-32B |
| 160000 | 2 | 339.7 | 159.1 | 86% | 364 | 942 | 21.4 | **gap ▼32%** — deepens; b2 wall ~180k est. |
| 320000 | 1 | 602.6 | 158.8 | 86% | 364 | 531 | 19.5 | ▼37% |
| 384000 | 1 | 905.6 | 181.4 | 98% | 364 | 424 | 17.9 | **last-fitting edge** (98% HBM, ▼42%; b1 wall just above — capF) |

## q3-32b — superoffload_mem | recomp
| seq | B | s/it | resv GiB | %HBM | RSS GB | tok/s | MFU% | flag |
|---|---|---|---|---|---|---|---|---|
| 8000 | 12 | 31.8 | 104.9 | 57% | 142 | 3022 | 28.5 |  |
| 8000 | 16 | 39.7 | 139.2 | 75% | 142 | 3222 | 30.4 |  |
| 8000 | 20 | 48.4 | 173.7 | 94% | 142 | 3303 | 31.2 | **MAX** |
| 12000 | 8 | 33.0 | 104.9 | 57% | 142 | 2908 | 28.5 |  |
| 12000 | 12 | 46.0 | 156.6 | 85% | 142 | 3128 | 30.6 |  |
| 12000 | 14 | 52.6 | 181.4 | 98% | 142 | 3197 | 31.3 | **MAX** |
| 12000 | 16 | OOM | 180 | 97% | 142 | — | — | OOM |
| 16000 | 8 | 42.8 | 139.3 | 75% | 142 | 2993 | 30.4 |  |
| 16000 | 10 | 52.4 | 173.5 | 94% | 142 | 3055 | 31.0 | **MAX** |
| 20000 | 8 | 55.1 | 173.5 | 94% | 142 | 2905 | 30.5 | **MAX** |
| 24000 | 4 | 35.7 | 105.0 | 57% | 142 | 2685 | 29.1 |  |
| 24000 | 6 | 51.1 | 156.3 | 84% | 142 | 2819 | 30.6 | **MAX** |
| 24000 | 8 | OOM | — | — | 142 | — | — | OOM |
| 28000 | 4 | 42.6 | 122.1 | 66% | 142 | 2630 | 29.4 |  |
| 28000 | 6 | 61.7 | 180.8 | 98% | 142 | 2725 | 30.5 | **MAX** |
| 30000 | 4 | 46.1 | 130.7 | 71% | 142 | 2601 | 29.6 | **MAX** |
| 30000 | 6 | OOM | — | — | 142 | — | — | OOM |
| 32000 | 4 | 48.7 | 139.3 | 75% | 142 | 2631 | 30.4 | **MAX** (b6 unprobed) |
| 36000 | 3 | 44.3 | 117.8 | 64% | 142 | 2440 | 29.0 |  |
| 36000 | 4 | 57.2 | 156.6 | 85% | 142 | 2519 | 30.0 | **MAX** |
| 40000 | 3 | 50.6 | 130.7 | 71% | 142 | 2372 | 29.0 |  |
| 40000 | 4 | 65.7 | 173.7 | 94% | 142 | 2436 | 29.8 | **MAX** |
| 45000 | 2 | 41.0 | 98.4 | 53% | 142 | 2195 | 27.8 |  |
| 45000 | 3 | 58.5 | 146.7 | 79% | 142 | 2309 | 29.3 | **MAX** (▼6%) |
| 50000 | 2 | 46.6 | 109.3 | 59% | 143 | 2146 | 28.1 |  |
| 50000 | 3 | 67.4 | 163.3 | 88% | 142 | 2226 | 29.2 | **MAX** (▼7%; tok/s rising b2→b3 = under-knee) |
| 56000 | 2 | 53.8 | 122.2 | 66% | 142 | 2081 | 28.4 | ▼9% (b3 ~183 edge, untested) |
| 80000 | 2 | 86.7 | 174.3 | 94% | 142 | 1846 | 29.1 | **B_max, gap ▼7%** (dense rc deficit saturates 7-9%) |
| 96000 | 1 | 58.5 | 104.8 | 57% | 142 | 1642 | 28.1 | **gap ▼10%** (b1; b2 ~206 OOM est.) |
| 128000 | 1 | 116.3 | 139.3 | 75% | 142 | 1101 | 21.9 | ▼27% — MIRRORS unsloth's 22.1% ⇒ the 96→128k step-down is SHARED (attention/regime), not offload |
| 160000 | 1 | 170.1 | 173.4 | 94% | 142 | 941 | 21.4 | ▼29% — EXACTLY unsloth's 942/21.4: dense rc≡uns per-token at long seq |
| 192000 | 1 | OOM | — | — | 142 | — | — | **b1 WALL pinned ∈ (160k, 192k)** (~173k est) |

## q3-30b-a3b — superoffload_mem | unsloth-ohbm0
| seq | B | s/it | resv GiB | %HBM | RSS GB | tok/s | MFU% | flag |
|---|---|---|---|---|---|---|---|---|
| 16000 | 24 | 42.8 | 113.6 | 61% | 278 | 8969 | 11.8 |  |
| 16000 | 32 | 55.3 | 149.9 | 81% | 278 | 9261 | 12.1 |  |
| 16000 | 36 | 61.8 | 167.8 | 91% | 381 | 9320 | 12.2 | **MAX** |
| 16000 | 40 | 98.8 | 181.4 | 98% | 381 | 6479 | 8.5 | thrash |
| 16000 | 48 | OOM | 174 | 94% | 354 | — | — | OOM |
| 24000 | 16 | 47.2 | 113.6 | 61% | 278 | 8136 | 12.4 |  |
| 24000 | 20 | 57.9 | 140.9 | 76% | 278 | 8287 | 12.6 |  |
| 24000 | 24 | 68.5 | 167.8 | 91% | 381 | 8411 | 12.8 | **MAX** |
| 24000 | 26 | 84.9 | 180.6 | 98% | 381 | 7349 | 11.2 | thrash |
| 24000 | 28 | OOM | 175 | 95% | 355 | — | — | OOM |
| 24000 | 32 | OOM | 174 | 94% | 354 | — | — | OOM |
| 32000 | 12 | 49.9 | 113.6 | 61% | 278 | 7700 | 13.3 |  |
| 32000 | 16 | 65.0 | 149.9 | 81% | 278 | 7877 | 13.6 | **MAX** |
| 32000 | 20 | 109.3 | 181.4 | 98% | 381 | 5856 | 10.1 | thrash |
| 40000 | 8 | 46.6 | 94.8 | 51% | 278 | 6873 | 13.3 |  |
| 40000 | 12 | 67.5 | 140.7 | 76% | 278 | 7114 | 13.8 | **MAX** |
| 40000 | 16 | 117.3 | 181.3 | 98% | 382 | 5455 | 10.6 | thrash |
| 45000 | 8 | 54.4 | 106.6 | 58% | 278 | 6619 | 13.7 |  |
| 45000 | 12 | 79.2 | 157.4 | 85% | 382 | 6817 | 14.1 | **MAX** |
| 45000 | 16 | OOM | — | — | — | — | — | OOM |
| 50000 | 8 | 63.3 | 117.9 | 64% | 278 | 6321 | 13.9 |  |
| 50000 | 10 | 77.7 | 146.5 | 79% | 278 | 6433 | 14.2 |  |
| 50000 | 12 | 92.3 | 174.8 | 94% | 382 | 6499 | 14.3 | **MAX** |
| 56000 | 8 | 73.8 | 131.4 | 71% | 279 | 6073 | 14.3 |  |
| 56000 | 10 | 91.0 | 163.2 | 88% | 381 | 6157 | 14.5 | **MAX** (no deficit — plateau holds) |
| 64000 | 6 | 66.3 | 113.6 | 61% | 278 | 5795 | 14.9 |  |
| 64000 | 8 | 87.0 | 150.0 | 81% | 278 | 5883 | 15.1 | **MAX** (no deficit) |
| 64000 | 10 | 138.1 | 181.3 | 98% | 381 | 4634 | 11.9 | thrash |
| 128000 | 4 | 166.5 | 149.9 | 81% | 278 | 3074 | 13.1 | **B_max, gap ▼13%** vs 15.1 — MoE gap FOUND |

## q3-30b-a3b — superoffload_mem | recomp
| seq | B | s/it | resv GiB | %HBM | RSS GB | tok/s | MFU% | flag |
|---|---|---|---|---|---|---|---|---|
| 8000 | 24 | 22.8 | 90.9 | 49% | 204 | 8405 | 9.2 |  |
| 8000 | 32 | 27.9 | 120.7 | 65% | 204 | 9177 | 10.1 |  |
| 8000 | 40 | 33.2 | 150.5 | 81% | 204 | 9638 | 10.6 |  |
| 8000 | 48 | 38.6 | 181.0 | 98% | 204 | 9953 | 11.0 | **MAX** |
| 8000 | 56 | OOM | 180 | 97% | 204 | — | — | OOM |
| 16000 | 12 | 24.5 | 90.9 | 49% | 204 | 7846 | 10.3 |  |
| 16000 | 16 | 30.3 | 120.7 | 65% | 204 | 8459 | 11.1 |  |
| 16000 | 20 | 36.2 | 150.5 | 81% | 204 | 8839 | 11.6 |  |
| 16000 | 24 | 42.2 | 181.0 | 98% | 204 | 9103 | 11.9 | **MAX** |
| 16000 | 28 | OOM | 180 | 97% | 204 | — | — | OOM |
| 24000 | 8 | 26.7 | 90.9 | 49% | 204 | 7192 | 10.9 |  |
| 24000 | 12 | 36.7 | 135.6 | 73% | 204 | 7848 | 11.9 |  |
| 24000 | 15 | 43.9 | 169.7 | 92% | 204 | 8201 | 12.5 |  |
| 24000 | 16 | 46.6 | 181.0 | 98% | 204 | 8244 | 12.5 | **MAX** |
| 32000 | 10 | 41.9 | 150.5 | 81% | 204 | 7645 | 13.2 |  |
| 32000 | 12 | 49.2 | 181.0 | 98% | 204 | 7804 | 13.5 | **MAX** |
| 32000 | 14 | OOM | — | — | 204 | — | — | OOM |
| 40000 | 8 | 45.9 | 150.5 | 81% | 204 | 6973 | 13.5 | **MAX** |
| 40000 | 12 | OOM | — | — | 204 | — | — | OOM |
| 45000 | 8 | 54.1 | 169.7 | 92% | 204 | 6655 | 13.8 | **MAX** |
| 45000 | 12 | OOM | — | — | 204 | — | — | OOM (unsloth fits b12; recomp caps b8) |
| 50000 | 6 | 47.7 | 141.2 | 76% | 204 | 6292 | 13.9 | **MAX** (no deficit) |
| 50000 | 8 | OOM | — | — | 204 | — | — | OOM |
| 56000 | 4 | 38.9 | 106.5 | 58% | 204 | 5763 | 13.6 |  |
| 56000 | 6 | 55.3 | 158.5 | 86% | 204 | 6072 | 14.3 | **MAX** (no deficit) |
| 64000 | 4 | 45.0 | 120.7 | 65% | 204 | 5694 | 14.6 |  |
| 64000 | 6 | 64.9 | 181.0 | 98% | 204 | 5919 | 15.2 | **MAX** (no deficit) |
| 128000 | 2 | 85.5 | 119.5 | 65% | 204 | 2995 | 12.7 | **gap ▼16%** vs 15.2 (b3 ~179 edge, untested) |

## llama3.3-70b — superoffload_mem | unsloth-ohbm0
| seq | B | s/it | resv GiB | %HBM | RSS GB | tok/s | MFU% | flag |
|---|---|---|---|---|---|---|---|---|
| 8000 | 8 | 37.1 | 37.1 | 20% | 300 | 1727 | 34.9 |  |
| 8000 | 12 | 51.6 | 55.5 | 30% | 350 | 1860 | 37.6 |  |
| 8000 | 16 | 67.0 | 73.2 | 40% | 350 | 1912 | 38.7 |  |
| 8000 | 24 | 97.0 | 111.1 | 60% | 522 | 1980 | 40.0 |  |
| 8000 | 32 | 130.9 | 147.7 | 80% | 522 | 1956 | 39.6 |  |
| 8000 | 40 | 160.1 | 181.0 | 98% | 865 | 1998 | 40.4 | **MAX** |
| 8000 | 48 | OOM | 181 | 98% | 862 | — | — | OOM |
| 16000 | 6 | 54.4 | 55.5 | 30% | 350 | 1763 | 38.1 |  |
| 16000 | 8 | 70.9 | 73.2 | 40% | 350 | 1806 | 39.1 |  |
| 16000 | 12 | 103.3 | 111.1 | 60% | 522 | 1858 | 40.2 |  |
| 16000 | 16 | 138.6 | 147.7 | 80% | 522 | 1847 | 39.9 |  |
| 16000 | 20 | 170.5 | 181.0 | 98% | 865 | 1877 | 40.6 | **MAX** |
| 16000 | 24 | OOM | 181 | 98% | 863 | — | — | OOM |
| 24000 | 10 | 138.9 | 138.7 | 75% | 522 | 1728 | 39.8 |  |
| 24000 | 13 | 177.3 | 180.1 | 97% | 865 | 1759 | 40.5 | **MAX** |
| 32000 | 4 | 79.4 | 73.2 | 40% | 350 | 1613 | 39.4 |  |
| 32000 | 6 | 116.8 | 111.1 | 60% | 522 | 1644 | 40.2 |  |
| 32000 | 8 | 156.4 | 147.7 | 80% | 522 | 1637 | 40.0 |  |
| 32000 | 10 | 192.5 | 181.0 | 98% | 865 | 1663 | 40.6 | **MAX** |
| 32000 | 12 | OOM | 181 | 98% | 863 | — | — | OOM |
| 40000 | 6 | 157.0 | 138.7 | 75% | 522 | 1529 | 39.5 |  |
| 40000 | 8 | 204.9 | 181.0 | 98% | 865 | 1562 | 40.3 | **MAX** |
| 48000 | 6 | 196.1 | 165.9 | 90% | 866 | 1469 | 40.0 | **MAX** |
| 48000 | 7 | 253.0 | 181.4 | 98% | 865 | 1328 | 36.1 | thrash |
| 48000 | 8 | OOM | 181 | 98% | 863 | — | — | OOM |
| 64000 | 4 | 192.1 | 147.7 | 80% | 522 | 1333 | 40.0 |  |
| 64000 | 5 | 237.7 | 181.0 | 98% | 865 | 1346 | 40.4 | **MAX** |
| 64000 | 6 | OOM | 181 | 98% | 863 | — | — | OOM |
| 96000 | 2 | 168.0 | 110.2 | 60% | 522 | 1143 | 40.7 | **MAX** (saturated) |
| 96000 | 3 | 253.7 | 165.9 | 90% | 865 | 1135 | 40.4 |  |
| 112000 | 2 | 260.2 | 129.6 | 70% | 522 | 861 | 33.1 | **gap ▼17%** — onset ∈ (96k,112k] |
| 128000 | 2 | 323.3 | 147.7 | 80% | 522 | 792 | 32.6 | **gap ▼18%**; MFU falls off 40% plateau |
| 144000 | 2 | 393.2 | 165.9 | 90% | 865 | 732 | 32.2 | **gap ▼20%** (B_max, still fits) |
| 192000 | 1 | 319.3 | 110.2 | 60% | 522 | 601 | 31.5 | **gap ▼21%** (b1 frontier) |
| 272000 | 1 | 592.7 | 156.8 | 85% | 865 | 459 | 30.5 |  |
| 320000 | 1 | 795.1 | 180.5 | 98% | 865 | 402 | 30.1 | **last-fitting edge** (98% HBM; MFU back UP to 30 — amortized; wall just above) |

## llama3.3-70b — superoffload_mem | recomp
| seq | B | s/it | resv GiB | %HBM | RSS GB | tok/s | MFU% | flag |
|---|---|---|---|---|---|---|---|---|
| 8000 | 4 | 24.5 | 57.1 | 31% | 300 | 1305 | 26.4 |  |
| 8000 | 8 | 36.5 | 113.3 | 61% | 300 | 1755 | 35.5 |  |
| 8000 | 12 | 50.6 | 169.9 | 92% | 300 | 1896 | 38.3 | **MAX** |
| 8000 | 13 | 54.9 | 181.3 | 98% | 300 | 1895 | 38.3 | thrash |
| 8000 | 16 | OOM | 180 | 98% | 300 | — | — | OOM |
| 16000 | 4 | 38.2 | 113.3 | 61% | 300 | 1674 | 36.2 |  |
| 16000 | 6 | 53.3 | 169.9 | 92% | 300 | 1802 | 39.0 | **MAX** |
| 16000 | 7 | OOM | 180 | 97% | 300 | — | — | OOM |
| 16000 | 8 | OOM | 180 | 98% | 300 | — | — | OOM |
| 24000 | 3 | 44.2 | 127.1 | 69% | 300 | 1627 | 37.5 |  |
| 24000 | 4 | 56.3 | 169.9 | 92% | 300 | 1706 | 39.3 | **MAX** |
| 32000 | 2 | 42.3 | 113.3 | 61% | 300 | 1513 | 36.9 |  |
| 32000 | 3 | 59.6 | 169.9 | 92% | 300 | 1610 | 39.3 | **MAX** |
| 32000 | 4 | OOM | 180 | 98% | 300 | — | — | OOM |
| 32000 | 6 | OOM | 181 | 98% | 300 | — | — | OOM |
| 48000 | 2 | 65.2 | 169.9 | 92% | 300 | 1472 | 40.1 | **MAX** |
| 52000 | 2 | 74.1 | 180.9 | 98% | 300 | 1404 | 39.2 | **MAX** (edge; still saturated ▼2%) |
| 56000 | 2 | OOM | — | — | 300 | — | — | OOM — b2 wall pinned (52k,56k) |
| 56000 | 1 | 43.8 | 99.5 | 54% | 300 | 1277 | 36.5 | **gap ▼9%** — b1 pocket (56k tok/step < knee) |
| 64000 | 2 | OOM | — | — | 300 | — | — | OOM |
| 80000 | 1 | 66.2 | 141.9 | 77% | 300 | 1208 | 39.6 | saturated again (▼1%) — b1 pocket CLOSES by 80k |
| 96000 | 1 | 84.8 | 169.7 | 92% | 300 | 1132 | 40.3 | saturated; b1 wall ~just past 96k. STORY CLOSED: gap = narrow b1 pocket 56-70k only |

## Peak tok/s summary (batch). ▼ = gap point (fits, under plateau).
| model | config | 8k | 16k | 20k | 24k | 28k | 30k | 32k | 36k | 40k | 45k | 48k | 50k | 56k | 64k | 80k | 96k | 112k | 128k | 144k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q3-32b | unsloth-ohbm0 | 3447(b44) | 3140(b20) | 3005(b16) | 2889(b14) | 2772(b12) | 2714(b12) | 2700(b10) | 2556(b10) | 2449(b8) | 2356(b8) | — | — | 2149(b6)▼6% | — | — | 1680(b3)▼8% | — | — | — |
| q3-32b | recomp | 3303(b20) | 3055(b10) | 2904(b8) | 2819(b6) | 2725(b6) | 2601(b4) | 2631(b4) | 2519(b4) | 2436(b4) | 2309(b3) | — | 2226(b3)▼7% | 2081(b2)▼9% | — | 1846(b2)▼7% | — | — | — | — |
| q3-30b-a3b | unsloth-ohbm0 | — | 9319(b36) | — | 8411(b24) | — | — | 7877(b16) | — | 7114(b12) | 6817(b12) | — | 6499(b12) | 6157(b10) | 5883(b8) | — | — | — | 3074(b4)▼13% | — |
| q3-30b-a3b | recomp | 9952(b48) | 9103(b24) | — | 8244(b16) | — | — | 7804(b12) | — | 6973(b8) | 6655(b8) | — | 6292(b6) | 6072(b6) | 5919(b6) | — | — | — | 2995(b2)▼16% | — |
| llama3.3-70b | unsloth-ohbm0 | 1998(b40) | 1876(b20) | — | 1759(b13) | — | — | 1662(b10) | — | 1562(b8) | — | 1468(b6) | — | — | 1346(b5) | — | 1143(b2) | 861(b2)▼17% | 792(b2)▼18% | 732(b2)▼20% |
| llama3.3-70b | recomp | 1895(b12) | 1801(b6) | — | 1706(b4) | — | — | 1609(b3) | — | — | — | 1472(b2) | — | 1277(b1)▼9% | — | 1208(b1) | 1132(b1) | — | — | — |

**Capacity (CORRECTS the old "unsloth OOM past 24k" claim — FALSE, never measured):**
unsloth-ohbm0 offloads activations to host → leaner HBM → fits higher batch AND longer seq than recomp
at every point measured (cost: host RSS 364–865 GB vs recomp 142–300 GB). recomp fails by OOM.
b8 seq-ceilings (user-verified): q3-32b 49k|8 (G-OOM 50k), q3-30b 80k|8 (G-OOM 81k), llama 45k|8 (G-OOM 46k).

**GAP WINDOWS — one block per (model × config), full per-seq series (steady-state saturated
peaks + deficit rows), schema: seq | B_max | resv GiB | %HBM | RSS GB | tok/s | MFU% | deficit | note.
NO cross-config ranking; NEVER drop the memory columns.**

q3-32b | unsloth-ohbm0:
| seq | B_max | resv | %HBM | RSS | tok/s | MFU% | deficit | note |
|---|---|---|---|---|---|---|---|---|
| 8k | 44 | 174.4 | 94% | 364 | 3447 | 32.6 | 0 | saturated |
| 12k | 30 | 178.4 | 96% | 364 | 3274 | 32.1 | 0 | |
| 16k | 20 | 161.0 | 87% | 364 | 3140 | 31.9 | 0 | |
| 20k | 16 | 161.0 | 87% | 364 | 3005 | 31.5 | 0 | |
| 24k | 14 | 168.9 | 91% | 364 | 2889 | 31.3 | 0 | |
| 28k | 12 | 168.9 | 91% | 364 | 2772 | 31.0 | 0 | |
| 30k | 12 | 178.4 | 96% | 364 | 2714 | 30.9 | 0 | |
| 32k | 10 | 161.0 | 87% | 364 | 2700 | 31.2 | 0 | |
| 36k | 10 | 178.4 | 96% | 364 | 2556 | 30.4 | 0 | |
| 40k | 8 | 161.0 | 87% | 364 | 2449 | 30.0 | 0 | |
| 45k | 8 | 178.4 | 96% | 364 | 2356 | 29.9 | ▼4% | window edge |
| 56k | 6 | 168.9 | 91% | 364 | 2149 | 29.3 | ▼6% | |
| 96k | 3 | 145.2 | 78% | 364 | 1680 | 28.8 | ▼8% | |
| 128k | 2 | 127.9 | 69% | 364 | 1110 | 22.1 | **▼29%** | would-be saturated ~1560 |
| 160k | 2 | 159.1 | 86% | 364 | 942 | 21.4 | **▼32%** | b2 wall ~180k est. |

q3-32b | recomp:
| seq | B_max | resv | %HBM | RSS | tok/s | MFU% | deficit | note |
|---|---|---|---|---|---|---|---|---|
| 8k | 20 | 173.7 | 94% | 142 | 3303 | 31.2 | 0 | saturated |
| 12k | 14 | 181.4 | 98% | 142 | 3197 | 31.3 | 0 | |
| 16k | 10 | 173.5 | 94% | 142 | 3055 | 31.0 | 0 | |
| 20k | 8 | 173.5 | 94% | 142 | 2905 | 30.5 | 0 | |
| 24k | 6 | 156.3 | 84% | 142 | 2819 | 30.6 | 0 | |
| 28k | 6 | 180.8 | 98% | 142 | 2725 | 30.5 | 0 | |
| 30k | 4 | 130.7 | 71% | 142 | 2601 | 29.6 | 0 | |
| 32k | 4 | 139.3 | 75% | 142 | 2631 | 30.4 | 0 | |
| 36k | 4 | 156.6 | 85% | 142 | 2519 | 30.0 | 0 | |
| 40k | 4 | 173.7 | 94% | 142 | 2436 | 29.8 | 0 | |
| 45k | 3 | 146.7 | 79% | 142 | 2309 | 29.3 | ▼6% | window edge |
| 50k | 3 | 163.3 | 88% | 142 | 2226 | 29.2 | ▼7% | rising b2→b3 = under-knee |
| 56k | 2 | 122.2 | 66% | 142 | 2081 | 28.4 | ▼9% | |
| 80k | 2 | 174.3 | 94% | 142 | 1846 | 29.1 | ▼7% | deficit saturates 7-9% (batch-knee) |
| 96k | 1 | 104.8 | 57% | 142 | 1642 | 28.1 | ▼10% | b1 frontier |

q3-30b-a3b | unsloth-ohbm0:
| seq | B_max | resv | %HBM | RSS | tok/s | MFU% | deficit | note |
|---|---|---|---|---|---|---|---|---|
| 16k | 36 | 167.8 | 91% | 381 | 9320 | 12.2 | 0 | saturated (MFU rises w/ seq) |
| 24k | 24 | 167.8 | 91% | 381 | 8411 | 12.8 | 0 | |
| 32k | 16 | 149.9 | 81% | 278 | 7877 | 13.6 | 0 | |
| 40k | 12 | 140.7 | 76% | 278 | 7114 | 13.8 | 0 | |
| 45k | 12 | 157.4 | 85% | 382 | 6817 | 14.1 | 0 | |
| 50k | 12 | 174.8 | 94% | 382 | 6499 | 14.3 | 0 | |
| 56k | 10 | 163.2 | 88% | 381 | 6157 | 14.5 | 0 | |
| 64k | 8 | 150.0 | 81% | 278 | 5883 | 15.1 | 0 | |
| 128k | 4 | 149.9 | 81% | 278 | 3074 | 13.1 | **▼13%** | window opens; deeper = c14 |

q3-30b-a3b | recomp:
| seq | B_max | resv | %HBM | RSS | tok/s | MFU% | deficit | note |
|---|---|---|---|---|---|---|---|---|
| 8k | 48 | 181.0 | 98% | 204 | 9953 | 11.0 | 0 | saturated |
| 16k | 24 | 181.0 | 98% | 204 | 9103 | 11.9 | 0 | |
| 24k | 16 | 181.0 | 98% | 204 | 8244 | 12.5 | 0 | |
| 32k | 12 | 181.0 | 98% | 204 | 7804 | 13.5 | 0 | |
| 40k | 8 | 150.5 | 81% | 204 | 6973 | 13.5 | 0 | |
| 45k | 8 | 169.7 | 92% | 204 | 6655 | 13.8 | 0 | |
| 50k | 6 | 141.2 | 76% | 204 | 6292 | 13.9 | 0 | |
| 56k | 6 | 158.5 | 86% | 204 | 6072 | 14.3 | 0 | |
| 64k | 6 | 181.0 | 98% | 204 | 5919 | 15.2 | 0 | |
| 128k | 2 | 119.5 | 65% | 204 | 2995 | 12.7 | **▼16%** | window opens; deeper = c14 |

llama3.3-70b | unsloth-ohbm0:
| seq | B_max | resv | %HBM | RSS | tok/s | MFU% | deficit | note |
|---|---|---|---|---|---|---|---|---|
| 8k | 40 | 181.0 | 98% | 865 | 1998 | 40.4 | 0 | saturated |
| 16k | 20 | 181.0 | 98% | 865 | 1877 | 40.6 | 0 | |
| 24k | 13 | 180.1 | 97% | 865 | 1759 | 40.5 | 0 | |
| 32k | 10 | 181.0 | 98% | 865 | 1663 | 40.6 | 0 | |
| 40k | 8 | 181.0 | 98% | 865 | 1562 | 40.3 | 0 | |
| 48k | 6 | 165.9 | 90% | 866 | 1469 | 40.0 | 0 | |
| 64k | 5 | 181.0 | 98% | 865 | 1346 | 40.4 | 0 | |
| 96k | 2 | 110.2 | 60% | 522 | 1143 | 40.7 | 0 | last saturated point |
| 112k | 2 | 129.6 | 70% | 522 | 861 | 33.1 | **▼17%** | onset ∈ (96k,112k] |
| 128k | 2 | 147.7 | 80% | 522 | 792 | 32.6 | **▼18%** | |
| 144k | 2 | 165.9 | 90% | 865 | 732 | 32.2 | **▼20%** | still fits @90% |
| 192k | 1 | 110.2 | 60% | 522 | 601 | 31.5 | **▼21%** | b1 frontier; fits to ~300k est. |

llama3.3-70b | recomp:
| seq | B_max | resv | %HBM | RSS | tok/s | MFU% | deficit | note |
|---|---|---|---|---|---|---|---|---|
| 8k | 12 | 169.9 | 92% | 300 | 1896 | 38.3 | 0 | saturated |
| 16k | 6 | 169.9 | 92% | 300 | 1802 | 39.0 | 0 | |
| 24k | 4 | 169.9 | 92% | 300 | 1706 | 39.3 | 0 | |
| 32k | 3 | 169.9 | 92% | 300 | 1610 | 39.3 | 0 | |
| 48k | 2 | 169.9 | 92% | 300 | 1472 | 40.1 | 0 | |
| 52k | 2 | 180.9 | 98% | 300 | 1404 | 39.2 | ▼2% | b2 wall edge |
| 56k | 1 | 99.5 | 54% | 300 | 1277 | 36.5 | ▼9% | narrow b1 pocket (~56-70k) |
| 80k | 1 | 141.9 | 77% | 300 | 1208 | 39.6 | ▼1% | pocket closed |
| 96k | 1 | 169.7 | 92% | 300 | 1132 | 40.3 | 0 | last fit |
| 112k | 1 | OOM | — | 300 | — | — | — | **b1 WALL pinned ∈ (96k, 112k)** |

## PHASE B — asym head-to-head at the gap seqs (RUNNING 2026-07-17)
Goal: beat superoffload tok/s at the gap seqs with larger batch. Asym configs:
LAT-KA = staged + dense keep-acts (short-seq latency mode; HBM transient ~linear in s);
LAT-LEAN = staged + ASYM_SAVED_TENSOR_ASYNC_UNPACK=1, NO keep-acts (canonical long-seq latency);
MEM = default memory mode. Prior asym refs: q3-32b LAT-KA 24k b8=2084@50%, b12=2107@70% (plateau ~23% MFU).

q3-32b @128k (superoffload uns b2 = 1110 tok/s, 22.1% MFU — the bar):
| system | B | s/it | resv | %HBM | RSS | tok/s | MFU% | verdict |
|---|---|---|---|---|---|---|---|---|
| superoffload uns | 2 | 230.7 | 127.9 | 69% | 364 | 1110 | 22.1 | bar |
| asym LAT-KA (no AU) | 3 | 401.6 | 136.3 | 74% | 484 | 956 | 19.1 | -14% (17% under own plateau; no async-unpack, no ohbm) |
| asym LAT-LEAN (no KA) | 4 | 679.8 | 114.2 | 62% | 956 | 753 | 15.0 | -32% — dropping keep-acts costs ~280us/tok (act restore path) |
| asym KA+AU+ohbm16 | 3 | 401.3 | 151.0 | 82% | 467 | 957 | 19.1 | -14% — IDENTICAL to plain KA: AU + hot weights buy NOTHING |

q3-32b @160k (superoffload uns b2 = 942 tok/s, 21.4% MFU — the bar):
| system | B | s/it | resv | %HBM | RSS | tok/s | MFU% | verdict |
|---|---|---|---|---|---|---|---|---|
| superoffload uns | 2 | 339.7 | 159.1 | 86% | 364 | 942 | 21.4 | bar |
| asym KA+AU | 3 | 560.0 | 175.1 | 95% | 777 | 857 | 19.5 | -9% (per-token gap narrowing with seq: +16%@128k -> +10%@160k) |

llama @128k (superoffload uns b2 = 792 tok/s, 32.6% MFU — the bar):
| system | B | s/it | resv | %HBM | RSS | tok/s | MFU% | verdict |
|---|---|---|---|---|---|---|---|---|
| superoffload uns | 2 | 323.3 | 147.7 | 80% | 522 | 792 | 32.6 | bar |
| asym KA+AU+ohbm16 | 3 | 566.9 | 182.6 | 99% | 919 | 677 | 27.9 | -15% (99% HBM — ohbm16 ate act headroom; clean ~29-30% est.) |

CONCLUSION (dense, flag-level): asym latency plateau sits just UNDER superoffload's deficit
floor on both dense models (q3-32b ~19.5% vs floor 21.4; llama ~28-30% vs floor 31.5) ->
straight MFU-vs-MFU loses on dense. Streaming fully hidden (ohbm null on q3-32b), AU null ->
residual gap is CODE (C2 fg wrapper diet + C1 dX/dW structure).
BATCH-QUANTIZATION EDGE tested on llama @192k (sup b1 = 601 tok/s, 31.5% — the bar):
| system | B | s/it | resv | %HBM | RSS | tok/s | MFU% | verdict |
|---|---|---|---|---|---|---|---|---|
| superoffload uns | 1 | 319.3 | 110.2 | 60% | 522 | 601 | 31.5 | bar |
| asym KA+AU | 2 | 693.9 | 171.1 | 92% | 963 | 553 | 29.0 | -8% — sup's b1 amortizes fine at 192k tok/step; edge insufficient on llama |
q3-32b @192k (sup b1 = 816 tok/s, 20.8% — b1 recovers, no collapse):
| system | B | s/it | resv | %HBM | RSS | tok/s | MFU% | verdict |
|---|---|---|---|---|---|---|---|---|
| superoffload uns | 1 | 235.4 | 96.4 | 52% | 227 | 816 | 20.8 | bar |
| asym KA+AU | 2 | 525.2 | 140.7 | 76% | 484 | 731 | 18.7 | -10% |

## PHASE B VERDICT vs unsloth-ohbm0 (flag-level, 2026-07-17) — FINAL
| point | sup ohbm0 | asym best (config) | margin |
|---|---|---|---|
| q3-32b 128k | 1110 (b2) | 957 (KA+AU+ohbm16 b3) | -14% |
| q3-32b 160k | 942 (b2) | 857 (KA+AU b3) | -9% |
| q3-32b 192k | 816 (b1) | 731 (KA+AU b2) | -10% |
| llama 128k | 792 (b2) | 677 (KA+AU+ohbm16 b3, 99%) | -15% |
| llama 192k | 601 (b1) | 553 (KA+AU b2) | -8% |
Asym latency (current code) loses tok/s to unsloth-ohbm0 everywhere measured: superoffload's
b1 re-amortizes at long seq (no collapse), and asym's ~2.0-2.1 vs 1.9 plateau-MFU-equivalent
deficit is code-level (C1 dX/dW + C2 fg wrapper; flags exhausted: KA essential, AU null,
ohbm null, LEAN regression). Paths to a win: (1) C1/C2 fixes (~100-143us/tok recovers all
five points), (2) vs unsloth-OFF bars (capP4b), (3) capacity/ceiling story (proven, separate).

## PHASE B vs unsloth-OFF (the fix_throughput defensible target) — WINS
q3-32b @128k: **asym WINS +34%** (and uses 200GB less host RAM; so-off b4 host-watchdog-killed
-> so-off batch headroom is HOST-capped):
| system | B | s/it | resv | %HBM | RSS | tok/s | MFU% | verdict |
|---|---|---|---|---|---|---|---|---|
| asym KA+AU | 3 | 401.6 | 136.3 | 74% | 484 | 957 | 19.1 | **WIN +34%** |
| superoffload unsloth-OFF | 3 | 539.3 | 117.0 | 63% | 690 | 712 | 14.2 | |
| (ref) superoffload ohbm0 | 2 | 230.7 | 127.9 | 69% | 364 | 1110 | 22.1 | ohbm0 gap = C1/C2 work |
q3-32b @160k: **asym WINS +34%**:
| system | B | s/it | resv | %HBM | RSS | tok/s | MFU% | verdict |
|---|---|---|---|---|---|---|---|---|
| asym KA+AU | 3 | 560.0 | 175.1 | 95% | 777 | 857 | 19.5 | **WIN +34%** |
| superoffload unsloth-OFF | 2 | 500.5 | 98.0 | 53% | 605 | 639 | 14.5 | |
llama @128k: **asym WINS +39%**:
| system | B | s/it | resv | %HBM | RSS | tok/s | MFU% | verdict |
|---|---|---|---|---|---|---|---|---|
| asym KA+AU+ohbm16 | 3 | 566.9 | 182.6 | 99% | 919 | 677 | 27.9 | **WIN +39%** |
| superoffload unsloth-OFF | 2 | 524.6 | 96.6 | 52% | 705 | 488 | 20.1 | |
llama @192k: **asym WINS +35%**:
| system | B | s/it | resv | %HBM | RSS | tok/s | MFU% | verdict |
|---|---|---|---|---|---|---|---|---|
| asym KA+AU | 2 | 693.9 | 171.1 | 92% | 963 | 553 | 29.0 | **WIN +35%** |
| superoffload unsloth-OFF | 1 | 469.7 | 73.6 | 40% | 705 | 409 | 21.4 | |

## q3-30b FRONTIER (c14 handoff data, 2026-07-18 — agent/handoffs screenshot; c12 replica of the
## 2 capacity runs in flight: uns 640k|1, rc 392k|1)
| run | RSS GB | resv GiB (%HBM) | s/it | tok/s |
|---|---|---|---|---|
| q3-30b **asym** 640k\|1 ✅ | 537 | 111.4 (**60%**) | 873.8 [1365 us/tok] | **732 (+0.1% vs uns 731 — PARITY)** |
| q3-30b **uns** 640k\|1 ✅ | 382 | 181.5 (**98%**) — last fit | 875.4 [1368 us/tok] | 731 |
| q3-30b **uns** 660k\|1 ✅ | — | **OOM — WALL** | — | DNF |
| q3-30b **rc** 392k\|1 ✅ | 204 | 181.4 (**98%**) — last fit | 331.0 [844 us/tok] | 1184 |
Same pattern as c12's dense/llama frontiers: asym parity at sup's edge (60% vs 98% HBM),
MoE walls: uns 660k, rc ~400k+, asym extends beyond.
c12 REPLICAS (cross-machine variance check):
| run | RSS GB | resv GiB (%HBM) | s/it | tok/s |
|---|---|---|---|---|
| q3-30b **uns** 640k\|1 ✅ c12 | 382 | 181.4 (98%) — last fit | 876.2 [1369 us/tok] | **730 (−0.1% vs c14 — REPRODUCED)** |
| q3-30b **rc** 392k\|1 ✅ c12 | 204 | 181.4 (98%) — last fit | 331.3 [845 us/tok] | **1183 (−0.1% vs c14 1184 — REPRODUCED)** |
Both replicas ≤0.1% — cross-machine variance nil; capacity edges are machine-independent.

## CAPACITY FRONTIER HEAD-TO-HEAD @384k (capG, 2026-07-18) — FIRST PARITY POINT
| system | B | s/it | resv | %HBM | RSS | tok/s | MFU% | verdict |
|---|---|---|---|---|---|---|---|---|
| sup unsloth | 1 | 905.6 | 181.4 | **98%** | 364 | 424 | 17.9 | last-fitting edge (wall ~390k) |
| asym KA | 1 | 900.5 | 140.6 | **76%** | 484 | **426** | 18.0 | **parity (+0.5%, in-noise) with 40 GiB headroom** |
At sup's absolute frontier, asym matches its tok/s while healthy — asym's frontier extends
further (b1 @76% → est. wall ~490k). recomp at 384k: DNF (wall 160-192k).

llama @320k (sup's edge seq):
| system | B | s/it | resv | %HBM | RSS | tok/s | MFU% | verdict |
|---|---|---|---|---|---|---|---|---|
| sup unsloth | 1 | 795.1 | 180.5 | **98%** | 865 | 402 | 30.1 | last-fitting edge |
| asym KA | 1 | 836.2 | 147.7 | **80%** | 984 | 383 | 28.6 | -5% at sup's edge, 33 GiB headroom; asym frontier extends (~390k est) |
recomp at 320k: DNF (wall 96-112k). llama parity NOT reached (dense was +0.5%); llama's
residual bwd tax larger — same fix_asym lever.

## PHASE B-2: THE R1 CONFIG (asym|unsloth-GC recompute + staged GEMMs) — 2026-07-18
Root cause of the old -8..-15%: `recomp-off` latency mode = recompute OFF = save+offload
~2.2 TB/step of attention tensors (serial +286 us/tok bwd tax) vs sup's recompute (+166).
R1 = the driver's existing `asym_cpuadamwds|unsloth-ohbm0` + ASYM_GEMM_DISPATCH=staged:
sup's compute shape, asym's host weights.
| @128k | B | s/it | resv | %HBM | RSS | tok/s | us/tok | verdict |
|---|---|---|---|---|---|---|---|---|
| sup unsloth | 2 | 230.7 | 127.9 | 69% | 364 | 1110 | 901.0 | bar |
| **asym R1** | 2 | 232.0 | **116.0** | 63% | 427 | 1104 | 906.1 | **PARITY (+0.6%), −12 GiB** |
| asym R1 | 4 | OOM | — | — | — | — | — | b4 wall (slope 0.51 GiB/1k tok) |
| asym R1 | 3 | 350.0 | 181.2 | 98% | 428 | 1097 | 911.5 | edge; batch is a non-lever at 128k (per-token-limited) |
⇒ 128k verdict: PARITY BAND (1104/1097 vs 1110). The structural R1 win = sup's batch-quantized
seqs: 192k sup b1=816 vs R1 b2 (384k tok = 181 GiB, fits) — fix6 measuring.
| (old asym best: recomp-off ladder) | 3 | 401.6 | 136.3 | 74% | 484 | 957 | 1045.8 | superseded by R1 |
Fix-ladder receipts: noclone NULL (+0.3%); attn-off b3 NEGATIVE (98% churn); attn-off b2
+2.2% (978); R1 +15% (1104) — the mode was the lever, not the copies.

## PHASE B FINAL SCOREBOARD (2026-07-17, all measured)
| point | vs unsloth-OFF | vs recomp | vs unsloth-ohbm0 |
|---|---|---|---|
| q3-32b 128k | **WIN +34%** (957 vs 712) | loses (rc b1 ~1500 est.) | -14% (1110) |
| q3-32b 160k | **WIN +34%** (857 vs 639) | loses (rc b1 to ~160k wall) | -9% (942) |
| q3-32b 192k | untested (so-off host-capped) | rc DNF, but ohbm0 alive | -10% (816) |
| llama 128k | **WIN +39%** (677 vs 488) | **WIN by DNF** (rc wall ~96k) | -15% (792) |
| llama 192k | **WIN +35%** (553 vs 409) | **WIN by DNF** | -8% (601) |
Standing claims: asym latency beats unsloth-OFF at every measured gap seq (+34-39%) with the
LARGER batch; beats recomp by DNF on llama >=112k (recomp cannot run at all); trails only
unsloth-ohbm0 by -8..-15% = the C1b/C2 backward code tax (fix_asym.md). Capacity story separate
and intact. so-off is HOST-RAM-capped at long seq (b4 128k host-watchdog-killed @690GB RSS).

FINDINGS (2026-07-17): fwd is at PARITY with superoffload per-token (168 vs 166 us/tok);
the entire gap is backward (+143 us/tok for KA-b3). Keep-acts is ESSENTIAL at long seq
(LEAN regression). Win window math: asym dense plateau ~23% MFU (24k b12 ref) vs superoffload
realized 22.1%@128k / 21.4%@160k / lower @192k(b1) -> win requires asym standing at its
plateau via KA + ASYM_SAVED_TENSOR_ASYNC_UNPACK (C3 stalls) + ohbm hot weights (panel
streaming). 96k UNWINNABLE (sup 28.8% > asym plateau) — dropped.
capP3 ladder: q3-32b 160k KA+AU b3 (bar 942, need >21.4%) -> 128k KA+AU+ohbm16 (bar 1110)
-> llama 128k KA+AU+ohbm16 (bar 792, llama asym plateau unknown) -> llama 192k KA+AU b2
(bar 601) -> sup 192k b1 (new bar) -> q3-32b 192k KA+AU b2. MEM arm dropped for tok/s
(memory-mode ~10% MFU: 65k|8 anchor 406 tok/s — capacity story only).

## PHASE W — llama3.3-70b so-unsloth capacity WALL (tputW1, 2026-07-19, c12)
Goal (user directive): pin so-unsloth's smallest-OOM seq for llama; asym intentionally NOT run.
Protocol: w1+m4 (legacy — chain launched pre-w1+m2 switch), MAX_SAMPLES=512, PROFILERS=source.

| run | verdict | metrics |
|---|---|---|
| 352000 b1 uns-ohbm0 | FIT (deep edge) | 336 tok/s · 2979 us/tok · step 1048.5s (m3: 1048.8/1051.9/1044.8) · resv 181.3 GiB = **98.0%** · RSS 806 GiB |
| 384000 b1 uns-ohbm0 | **OOM** | CUDA OOM: tried 20.51 GiB, 5.88 GiB free of 184; HOST_OOM_EVIDENCE=false |

- **WALL PINNED: last-fit 352k @98.0% · smallest-OOM 384k.** The old "wall ~326k" estimate
  (token-linear fit) under-predicted by ≥8% — capG rule confirmed again: probe, never predict.
- Edge tax deepens with depth: 320k = 402 tok/s (2488 us/tok) → 352k = 336 (2979 us/tok);
  +20% per-token for +10% tokens (attention share predicts ~+7%) — allocator churn at 98%
  worsens as the free pool shrinks.
- First production use of the FAST dataset builder (2026-07-19 fix): s352000 + s384000 n512
  built in ~90s each (was ~6 min), byte-identity A/B-proven (concat/sample/audit paths).
- asym at 352k/384k: deliberately not run (so-unsloth wall first, per user). llama s384000
  dataset now on disk for any future asym probe.


## PHASE W6/W7 + X — llama stretch/est-kill + Qwen3.5-35B-A3B campaign (2026-07-20, offline mode)
Context: HF auth token vanished mid-day (env reorg) → all probes now run
HF_HUB_OFFLINE=1 (models/datasets cached). Driver false-fails: (a) 401 in
post-process killed jobs.tsv rows of a COMPLETED run (llama T2 416k); (b) dirty
teardown marks completed runs failed:1 (q35 T2/T3) — tp_probe gained an
artifacts-complete fallback (step_samples rows >= MAX_STEPS ⇒ FIT).

| run | verdict | metrics |
|---|---|---|
| llama T2 416k b1 | **FIT** (recovered) | ~299 tok/s (3 uniform steps 1390s; profiler artifacts lost to 401) · ~99% observed |
| llama T2 448k b1 | **FIT — new llama deepest** | 275 tok/s · lat 1628.2s · **180.1 GiB = 97.3%** · RSS 983 GB |
| llama T1 96k b1 | FIT | 1066 tok/s · 48.9 GiB = 26.5% · RSS 486 GB (b2 batch-matched cell in flight) |
| llama T2 352k b1 | HOST-OOM (flake — 384/416/448k all fit; T2 lives at RSS≈pool, zero margin) | re-probe in flight |
| q35 rc 256k b1 | FIT | 848 tok/s · 121.3 GiB = 65.5% · RSS 190 GB |
| q35 rc 384k b1 | FIT (edge) | 1002 tok/s · 178.3 GiB = 96.4% · RSS 190 GB |
| q35 rc 448k/512k b1 | **OOM — wall (384k, 448k]** | |
| q35 uns 512k b1 | FIT | 1067 tok/s · 178.4 GiB = 96.4% · RSS 275 GB |
| q35 uns 576k b1 | FIT (edge, last fit) | 1023 tok/s · 180.8 GiB = 97.7% · RSS 362 GB |
| q35 uns 640k b1 | **OOM — wall (576k, 640k]** | |
| q35 T1 128k b1 | FIT | 609 tok/s · 45.2 GiB = 24.5% · RSS 224 GB |
| q35 T2 576k b1 | **FIT — +35% OVER uns AT UNS'S EDGE** | **1377 tok/s · 95.7 GiB = 51.7%** · RSS 524 GB |
| q35 T3 640k b1 | **FIT — sole coverage at uns's OOM seq** | **1142 tok/s · 64.4 GiB = 34.8%** · RSS 554 GB |

- Qwen3.5 arch fully supported: T3 ran the asym streaming kernel (asym_forward_calls=36810).
- v4 in flight: q35 T3 704k→+64k ladder · q35 uns/rc 128k cells · llama T2 480k wall ·
  llama 352k re-probe · llama T1 96k b2.
