# test_throughput_results — PURE METRICS (c14) — q3-30b-a3b

HOST: s04-p1-dgx-02-c14 | GPU: GB200 (189471 MiB = 185.0 GiB HBM, same as reference c12) | driver 580.105.08 | venv torch 2.12.0+cu130 (py 3.12.3) | host RAM 1693 GB
ISOLATION: shared NFS checkout (10.78.200.27:/data/home/kevinni/AsymGEMM-SFT-39 → /workspace/AsymGEMM-SFT-39); session runs inside the CUDA container on c14 directly (no enroot hop; venv path-baked to this checkout). RUN_NAME host-tagged; live root verified quiet at start; model partition: c14 owns q3-30b-a3b only. HF cache node-local. tp_probe.sh / .repair_dataset_info.py absent here → prompt launcher-template semantics used. All asym runs use the 2026-07-17 rebuilt `_C`.
RAW ARTIFACTS: profiling_results/profiling_tp_s04-p1-dgx-02-c14/asym_long_sft_smoke__lora__lf__bf16/ (archived post-campaign; in-flight under profiling_results/profiling/...).
Units: resv in GiB (phys HBM 185.0); RSS in GB. tok/s = B·seq/s_it. MFU vs 2250 TFLOPS bf16. Steady s/it = mid of w1+m4, PROFILERS=source. N_active 3.34e9, L=48, h=2048.
Flags: **MAX** = peak tok/s at that seq | edge = 92–98% HBM, <1.5× blowup | thrash = ≥95% + ≥1.5× blowup | OOM.

## CONFIG LEGEND (exact `backend|recompute|liger` strings; recompute flags verified in each run's config.json)
| short name | full config string | key flags | tag |
|---|---|---|---|
| so-recomp | superoffload_mem\|recomp\|ligerloss1 | use_unsloth_gc=false | tputrc-c14 |
| so-unsloth | superoffload_mem\|unsloth-ohbm0\|ligerloss1 | unsloth_gc=true, save_on_cpu=false | tput-c14 |
| so-unsloth-OFF | superoffload_mem\|unsloth-off-ohbm0\|ligerloss1 | unsloth_gc=true, save_on_cpu=true | tputuo-c14 (NOT YET RUN) |
| asym-lat | asym_cpuadamwds\|recomp-off-full-fg-ker000-ceil0000-ohbm0\|ligerloss1 + staged-dispatch env stack¹ | route000_lora0 | tputasl-c14 |
| asym-lat+RL | asym-lat + ROUTE_LORA=1 + KEEP_DGRADS_HBM=1 | route000_lora1 | tputasl2/3-c14 |
| asym-mem | asym_cpuadamwds\|recomp-off-full-fg-ker101-ceil0000-ohbm16\|ligerloss1 | default env | tputasm-c14 (NOT YET RUN) |

¹ asym-lat env stack: ASYM_GEMM_DISPATCH=staged, ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1, ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1, ASYMM_QWEN3_MOE_FG_DA_GPU=1, ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0, ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024, ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1.

## RAW — q3-30b-a3b × so-recomp (superoffload_mem|recomp|ligerloss1)
| seq | B | s/it | resv GiB | %HBM | RSS GB | tok/s | MFU% | flag |
|---|---|---|---|---|---|---|---|---|
| 128000 | 2 | 85.8 | 119.5 | 65 | 204 | 2985 | 12.7 | **MAX**·confirm |
| 160000 | 2 | 124.1 | 151.0 | 82 | 204 | 2578 | 13.1 | **MAX** |
| 180000 | 2 | 153.2 | 169.7 | 92 | 204 | 2349 | 13.2 | **MAX** |
| 208000 | 1 | 101.2 | 98.7 | 53 | 204 | 2056 | 13.0 | **MAX** |
| 256000 | 1 | 146.7 | 119.9 | 65 | 204 | 1745 | 13.3 | **MAX** |
| 320000 | 1 | 222.3 | 151.0 | 82 | 204 | 1440 | 13.4 | **MAX** |
| 360000 | 1 | 279.9 | 169.7 | 92 | 204 | 1286 | 13.3 | **MAX**²ᵇ |
| 384000 | 1 | 315.3 | 181.0 | 98 | 204 | 1218 | 13.3 | **MAX**·edge²ᶜ |
| 392000 | 1 | 331.0 | 181.4 | 98 | 204 | 1184 | 13.2 | **MAX**·edge²ᵈ |
| 400000 | 1 | — | — | — | — | — | — | OOM² |

² OOM: needed 12.21 GiB more with 169.7 allocated → absolute wall; bisection in progress (384k in flight, 392k queued), model 392k.
²ᵇ 360k = saturation point at the 92% healthy limit (pred 170/92% exact; slope 0.4675 GiB/1k confirmed); deepest healthy recomp point.
²ᶜ 384k = 98% edge fit, ZERO pressure penalty (healthy-linear 316.8, measured 315.3) — recomp rides its edge free, like so-unsloth.
²ᵈ 392k = 98% fit (+0.9% penalty); linear pred 184.7 but ~3.3 GiB near-wall allocator bend carried it. **recomp b1 WALL = (392k, 400k] — 8k granularity, bisection complete.** Deepest recomp fit = 392k.

## RAW — q3-30b-a3b × so-unsloth (superoffload_mem|unsloth-ohbm0|ligerloss1)
| seq | B | s/it | resv GiB | %HBM | RSS GB | tok/s | MFU% | flag |
|---|---|---|---|---|---|---|---|---|
| 128000 | 4 | 167.6 | 149.9 | 81 | 279 | 3055 | 13.0 | **MAX**·confirm |
| 160000 | 3 | 184.6 | 140.9 | 76 | 279 | 2601 | 13.2 | **MAX** |
| 180000 | 3 | 229.1 | 157.4 | 85 | 382 | 2357 | 13.2 | **MAX** |
| 208000 | 2 | 198.2 | 122.5 | 66 | 279 | 2099 | 13.3 | **MAX** |
| 256000 | 2 | 291.6 | 149.3 | 81 | 278 | 1756 | 13.3 | **MAX** |
| 320000 | 2 | 472.2 | 181.3 | 98 | 382 | 1355 | 12.6 | edge³ |
| 320000 | 1 | 222.8 | 94.8 | 51 | 278 | 1436 | 13.3 | **MAX** |
| 480000 | 1 | 484.8 | 141.1 | 76 | 279 | 990 | 13.3 | **MAX** |
| 600000 | 1 | 749.9 | 175.1 | 95 | 382 | 800 | 13.3 | **MAX**·edge⁴ |
| 640000 | 1 | 875.4 | 181.5 | 98 | 382 | 731 | 12.9 | **MAX**·edge⁴ᵇ |
| 660000 | 1 | — | — | — | — | — | — | OOM⁴ᶜ |

³ b2@320k edge: +5.9% s/it over healthy-linear; healthy b1 out-throughputs it (1436 > 1355) — batch scaling inverts at the ceiling. Capacity-model first miss (pred 186.4 OOM, actual 181.3).
⁴ 95% HBM but s/it = healthy-linear (748): zero pressure penalty at 600k.
⁴ᵇ 640k: 98% edge, +3.0% over healthy-linear (850) — deepest so-unsloth fit.
⁴ᶜ 660k OOM: single 20.14 GiB alloc failed with 18.83 free (162.5 allocated). **so-unsloth b1 WALL = (640k, 660k]** — 20k granularity; the failing alloc is a ~20 GiB monolith, so the workload's own memory granularity exceeds 10k-seq steps → 650k bisect skipped as below-physics (would likely OOM on the same chunk).

## RAW — q3-30b-a3b × asym-lat (asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0 + staged stack¹)
| seq | B | s/it | resv GiB | %HBM | RSS GB | tok/s | MFU% | flag |
|---|---|---|---|---|---|---|---|---|
| 208000 | 2 | 220.9 | 78.1 | 42 | 407 | 1883 | 11.9 | **MAX** |
| 208000 | 5 | 557.9 | 180.1 | 97 | 539 | 1864 | 11.8 | edge⁵ |
| 320000 | 2 | 479.2 | 117.5 | 63 | 537 | 1336 | 12.4 | **MAX** |
| 480000 | 1 | 506.5 | 90.2 | 49 | 408 | 948 | 12.8 | **MAX** |
| 600000 | 1 | 915.4 | 110.3 | 60 | 546 | 655 | 10.9 | anomaly⁵ᵇ |
| 640000 | 1 | 873.8 | 111.4 | 60 | 537 | 732 | 12.9 | **MAX·PARITY**⁵ᶜ |
| 800000 | 1 | 1340.0 | 147.5 | 80 | 539 | 597 | 13.1 | **MAX·exclusive**⁵ᵈ |

⁵ b5 edge: alloc 118.1 vs resv 180.1 = 62 GiB allocator fragmentation; s/it +1% over linear → batch amortization nullified. tok/s batch-flat: 1883(b2)/1867(b4·RL)/1864(b5).
⁵ᵇ 600k b1: +9.4% over asym's own linear (exp ~837), MFU 10.9. RE-CLASSIFIED as ONE-OFF ANOMALY: the 640k run (⁵ᶜ) came in FASTER in absolute s/it (873.8 < 915.4) with MFU back on trend (12.9) — a seq-scaling law would forbid that. CPU pool exonerated (0 fallbacks). Row kept as measured; rerun would likely land ~790 s/it / ~760 tok/s.
⁵ᶜ **640k b1 = PARITY POINT: asym 732 tok/s @60% HBM vs so-unsloth 731 @98% edge (+0.1%, in-noise) — asym matches superoffload-unsloth's tok/s at superoffload's absolute frontier seq with 70 GiB headroom.** Mirrors c12's q3-32b parity @384k (426 vs 424).
⁵ᵈ **800k b1 = EXCLUSIVE-FIT ceiling sanity: healthy 80% HBM, MFU 13.1 on-trend (−0.9% vs own-linear), at +21% past so-unsloth's wall (660k) and 2x recomp's (400k).** No seq-scaling regression through 800k (600k row confirmed one-off). Asym b1 wall unprobed; slope extrapolates ~980k.

## RAW — q3-30b-a3b × asym-lat+RL (…+ROUTE_LORA=1+KEEP_DGRADS_HBM=1) — 208k lever A/Bs
| seq | B | s/it | resv GiB | %HBM | RSS GB | tok/s | MFU% | flag |
|---|---|---|---|---|---|---|---|---|
| 208000 | 4 | 445.7 | 153.2 | 83 | 537 | 1867 | 11.8 | levers≈0⁶ |
| 208000 | 4 | 484.1 | 152.8 | 83 | 606 | 1719 | 10.9 | regression⁷ |

⁶ Flags verified engaged (route000_lora1 tag); no measurable gain vs plain stack.
⁷ +CHUNK_MB=0 +expandable_segments: −8% tok/s; both reverted for later runs.
**208k verdict (5 asym configs): best asym 1883 vs so-unsloth 2099 → no tok/s beat at 208k. Cause: attention = 86% of FLOPs, GEMM-saturated at b1–b2, tok/s batch-flat; ~11% per-token tax not amortizable. Asym wins memory only (78.1 vs 122.5 GiB at b2).**

## RAW — q3-30b-a3b × asym-mem (ker101-ceil0000-ohbm16) — NOT YET RUN
| seq | B | s/it | resv GiB | %HBM | RSS GB | tok/s | MFU% | flag |
|---|---|---|---|---|---|---|---|---|

## GAP WINDOW — q3-30b-a3b × so-recomp
| seq | B_max | resv GiB | %HBM | RSS GB | tok/s | MFU% | deficit | note |
|---|---|---|---|---|---|---|---|---|
| 64000 | 6 | 181.0 | 98 | 204 | 5919 | 15.2 | 0% | c12 lead-in |
| 128000 | 2 | 119.5 | 65 | 204 | 2985 | 12.7 | 16.4% | c14 measured (anchor resv bit-match) |
| 160000 | 2 | 151.0 | 82 | 204 | 2578 | 13.1 | 13.8% | measured |
| 180000 | 2 | 169.7 | 92 | 204 | 2349 | 13.2 | 13.2% | measured |
| 208000 | 1 | 98.7 | 53 | 204 | 2056 | 13.0 | 14.5% | measured |
| 256000 | 1 | 119.9 | 65 | 204 | 1745 | 13.3 | 12.5% | measured |
| 320000 | 1 | 151.0 | 82 | 204 | 1440 | 13.4 | 11.8% | measured |
| 360000 | 1 | 169.7 | 92 | 204 | 1286 | 13.3 | 12.5% | measured — saturation |
| 400000 | 1 | — | — | — | — | — | — | OOM wall |
Frontier CLOSED: onset (80k,128k] · b2→b1 (180k,208k] · wall (320k,400k].

## GAP WINDOW — q3-30b-a3b × so-unsloth
| seq | B_max | resv GiB | %HBM | RSS GB | tok/s | MFU% | deficit | note |
|---|---|---|---|---|---|---|---|---|
| 64000 | 8 | 150.0 | 81 | 278 | 5883 | 15.1 | 0% | c12 lead-in |
| 128000 | 4 | 149.9 | 81 | 279 | 3055 | 13.0 | 13.9% | c14 measured (anchor resv bit-match) |
| 160000 | 3 | 140.9 | 76 | 279 | 2601 | 13.2 | 12.6% | measured |
| 180000 | 3 | 157.4 | 85 | 382 | 2357 | 13.2 | 12.6% | measured |
| 208000 | 2 | 122.5 | 66 | 279 | 2099 | 13.3 | 11.9% | measured |
| 256000 | 2 | 149.3 | 81 | 278 | 1756 | 13.3 | 11.9% | measured |
| 320000 | 1 | 94.8 | 51 | 278 | 1436 | 13.3 | 11.9% | measured⁹ |
| 480000 | 1 | 141.1 | 76 | 279 | 990 | 13.3 | 11.9% | measured |
| 600000 | 1 | 175.1 | 95 | 382 | 800 | 13.3 | 11.9% | edge |
Onset (80k,128k] · b3→b2 (180k,208k] · b2→b1-healthy (256k,320k] · wall pred 640–660k (mapping).

⁸ c12 anchor rows: resv from prompt anchors; tok/s+MFU derived from anchored deficit; s/it derived; RSS unknown.
⁹ b1 is MAX at 320k because b2 only edge-fits (98%) and is slower (1355 < 1436).

## HEAD-TO-HEAD — asym-lat vs so-unsloth, equal seq (goal-1/2 core)
| seq | asym-lat tok/s | asym-lat %HBM | so-unsloth tok/s | so-unsloth %HBM | asym-lat vs so-unsloth tok/s |
|---|---|---|---|---|---|
| 208000 | 1883 (b2) | 42 | 2099 (b2) | 66 | **−10.3%** |
| 320000 | 1336 (b2) | 63 | 1436 (b1) | 51 | **−7.0%** |
| 480000 | 948 (b1) | 49 | 990 (b1) | 76 | **−4.2%** |
| 600000 | 655 (b1)⚠anomaly | 60 | 800 (b1) | 95 | (−18.1%)⚠ |
| 640000 | 732 (b1) | 60 | 731 (b1) | 98 | **+0.1% — PARITY** |
Tax dilution: −10.3 → −7.0 → −4.2% (208k→480k) → **+0.1% @640k = PARITY at so-unsloth's absolute frontier**, asym at 60% vs 98% HBM. The 600k asym row is a re-classified one-off anomaly (⁵ᵇ; 640k ran faster in absolute s/it). Beyond 660k so-unsloth cannot run at all — asym continues (800k probe in flight).
vs so-recomp: asym-lat −8.4% @208k, −7.2% @320k, then so-recomp DEAD ≥400k → asym wins by existence.

## Peak tok/s summary — tok/s(B_max). † = c12 ref/anchor-derived. — = not run. ⚠ = open regression.
| config | 64k | 128k | 160k | 208k | 256k | 320k | 360k | 392k | 480k | 600k | 640k | b1 WALL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| so-recomp | 5919(b6)† | 2985(b2) | 2578(b2) | 2056(b1) | 1745(b1) | 1440(b1) | 1286(b1) | 1184(b1) | — | — | — | **(392k, 400k]** |
| so-unsloth | 5883(b8)† | 3055(b4) | 2601(b3) | 2099(b2) | 1756(b2) | 1436(b1) | — | — | 990(b1) | 800(b1) | 731(b1) | **(640k, 660k]** |
| asym-lat | — | — | — | 1883(b2) | — | 1336(b2) | — | — | 948(b1) | 655(b1)⚠ | — | unprobed (≥660k) |
(180k in raw tables: rc 2349(b2), uns 2357(b3). asym paused; asym-mem and unsloth-off not run — out of scope per user.)

## CAMPAIGN STATUS: COMPLETE incl. asym capacity phase (2026-07-18 ~01:50 PT)
- Superoffload frontiers CLOSED with saturation rows + walls at ≤20k granularity (rc 8k, uns 20k — uns bisect stopped at the 20-GiB alloc-granularity floor).
- Variance vs c12: 4/4 checks pass, |Δ| ≤ 1.2%, peak-resv bit-identical → hosts directly comparable.
- All 30 run dirs archived to profiling_results/profiling_tp_s04-p1-dgx-02-c14/; live root clean.
- All runs PROFILERS=source (every run dir carries the __source__ tag).
- OPEN ITEMS (asym, paused per user): 600k seq-scaling regression (+9.4% over own-linear, pool exonerated); asym-mem capacity rows; GPU-side buffer pool idea (would recover ~60 GiB fragmentation → +2-3 batch).
