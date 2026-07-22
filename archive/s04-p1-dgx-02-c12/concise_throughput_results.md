# concise_throughput_results — max tok/s per ctx + thrash points + smallest-OOM, per model

Cell = tok/s · batch · %HBM (resv/185 GiB). Host GB200; PROFILERS=source, w1+m4
(runs from 2026-07-19 onward: w1+m2 — post-warmup steps stable within ~1%, so comparable).
Backends: **asym** (best config per point: R1 = asym|unsloth-ohbm0+staged; R2 = keep-acts
latency; both recompute — see fix_asym.md mechanism note) · **so-unsloth** (unsloth-ohbm0)
· **so-recomp**. "edge" = last fitting point at 97-98% HBM (runs, degraded by alloc
pressure). "thrash" = one batch past B_max: fits but collapses. OOM = smallest seq/batch
that failed. Full per-run record: test_throughput_results.md.

## q3-32b (dense 32B)
| ctx | asym | so-unsloth | so-recomp |
|---|---|---|---|
| 128k | 1104 · b2 · 63% (R1) | 1110 · b2 · 69% | 1101 · b1 · 75% |
| 160k | 938 · b2 · 78% (R1) | 942 · b2 · 86% | 941 · b1 · **94%** |
| 192k | 812 · b2 · 95% (R1) | 816 · b1 · 52% | **OOM** (b1) |
| 384k | 426 · b1 · **76%** (R2) | 424 · b1 · **98% edge** | — |
| batch-thrash example | (not probed for R1) | 24k: b14=2889 → b16=**2124** (98%) | rarely thrashes — OOMs instead |
| seq capacity (any batch) | **T3 640k fits @70% (226 tok/s, RSS 980 GB); HOST-OOM at 704k — wall pinned 07-20** | 384k = last fit (edge); **OOM at 416k — pinned 07-20** | 160k = last fit @94%; **OOM at 192k** |

## llama3.3-70B (dense)
| ctx | asym | so-unsloth | so-recomp |
|---|---|---|---|
| 128k | 786 · b2 · 70% (R1) | 792 · b2 · 80% | **OOM** (b1) |
| 192k | **603** · b1 · 52% (R1) | 601 · b1 · 60% | — |
| 320k | 383 · b1 · 80% (R2; R1 untried) | 402 · b1 · **98% edge** | — |
| 352k | 355 est (one host-OOM flake; re-probe in flight) | 336 · b1 · **98.0% edge** | — |
| 384k | **326 · b1 · 96.6%** (T2; RSS 975 GB) | **OOM** (wall) | — |
| 416k | ~299 · b1 · ~99% (T2; steady from 3-step total) | OOM | — |
| 448k | **275 · b1 · 97.3%** (T2; RSS 983 GB) | OOM | — |
| batch-thrash example | 192k: b2@**97.7%** = 577 (−4% vs own b1 — edge tax measured) | 48k: b6=1469 → b7=**1328** (98%) | 8k: b13=1895 (98%, flat) |
| seq capacity (any batch) | **T2 448k = last fit @97.3% (275 tok/s); 480k probe in flight. T3 host-walled at 416k (tier inversion)** | **352k = last fit @98.0%; OOM at 384k** (wall pinned 07-19) | 96k = last fit @92%; **OOM at 112k** |

## q3-30b-a3b (MoE)
| ctx | asym | so-unsloth | so-recomp |
|---|---|---|---|
| 128k | not measured in R1 (hole — c14) | 3074 · b4 · 81% | 2995 · b2 · 65% |
| 640k | **732 · b1 · 60%** (c14, memory-mode ker101) | 731 · b1 · **98% edge** | **OOM** (b1) |
| batch-thrash example | (not probed) | 64k: b8=5883 → b10=**4634** (98%) | 64k: b6=5919 @98% (flat — no thrash) |
| seq capacity (any batch) | **640k fits @60%; wall est ~1M (not probed)** | 640k = last fit (edge); **OOM at 660k** | 392k = last fit @98%; OOM just above (~400k, not probed) |

## Cross-model reads (all measured)
- Speed where all fit: three-way TIE (±1%) — per-token converges across backend AND batch
  at long ctx (shared attention cost); asym runs it at the lowest %HBM.
- so-unsloth degrades before it dies (edge points at 98% lose 5-40% MFU); so-recomp dies
  before it degrades (flat to the wall, then OOM); asym at the same seqs sits at 52-80%.
- Batch is a CAPACITY lever, not a throughput lever; pushing past ~92% HBM costs tok/s
  (llama 192k b2: −4%). Scheduler rule: largest B with resv ≤ 0.92·185 GiB.
- asym-only territory (all MEASURED): q32 416–640k · llama 384–448k · q3-30b >660k
  · q3.5-35b ≥640k (T3 1142 tok/s @35% at uns's OOM seq; depth ladder in flight).
- so-unsloth edge tax deepens with depth: llama 320k 402 tok/s (2488 us/tok) → 352k 336
  (2979): +20% per-token for +10% tokens (attention predicts ~+7%).

## q3.5-35b-a3b (MoE, tputX campaign 2026-07-20, b1 throughout)
| ctx | asym | so-unsloth | so-recomp |
|---|---|---|---|
| 128k | 609 · b1 · 24.5% (T1) | (in flight) | (in flight) |
| 256k | — | — | 848 · b1 · 65.5% |
| 384k | — | — | 1002 · b1 · **96.4% edge** |
| 448k | — | — | **OOM** (wall 384–448k) |
| 512k | — | 1067 · b1 · 96.4% | OOM |
| 576k | **1377 · b1 · 51.7%** (T2 — **+35% over uns at uns's own edge**, half the HBM) | 1023 · b1 · **97.7% edge — last fit** | OOM |
| 640k | **1142 · b1 · 34.8%** (T3 — sole coverage; RSS 554 GB) | **OOM** (wall 576–640k) | OOM |
| seq capacity (any batch) | **T3 ≥640k @35%; +64k ladder in flight (huge headroom)** | 576k = last fit (edge); OOM at 640k | 384k = last fit (edge); OOM at 448k |

- fg/asym kernels fully engage on the Qwen3.5 arch (T3 asym_forward_calls=36810).
- MoE MFU rises with seq at b1 (rc 848@256k → 1002@384k) — per-token falls with depth.
- Driver marks these runs failed on dirty teardown despite complete artifacts —
  tp_probe now trusts step_samples (artifacts-complete fallback, 07-20).
