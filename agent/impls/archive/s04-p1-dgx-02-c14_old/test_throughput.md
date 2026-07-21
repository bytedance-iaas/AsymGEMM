# test_throughput (c14) — deep-end gap hunt, q3-30b-a3b only

HOST: s04-p1-dgx-02-c14 | GPU: GB200 (189471 MiB = 185.0 GiB HBM) | driver 580.105.08 | venv torch 2.12.0+cu130 | host RAM 1693 GB
Reference (READ-ONLY): agent/impls/s04-p1-dgx-02-c12/test_throughput{,_results}.md — c12 owns llama3.3-70b + q3-32b + the q3-30b probes through 128k (grandfathered).
This host per HOST PARTITION (2026-07-17): **q3-30b-a3b ONLY**, superoffload deep-end 160k+, asym Phase B, ~2 variance spot-checks per config.
Protocol, metric definitions, R0-R6 rules: agent/impls/throughput_prompt.md + c12's test_throughput.md (identical protocol; PROFILERS=source MAX_STEPS=4 WARMUP_STEPS=1 MAX_SAMPLES=1024 DATASET_OVERWRITE=false OVERWRITE=false).
Isolation: shared NFS checkout — RUN_NAME host-tagged `tput-c14_*`/`tputrc-c14_*`; archive per campaign to profiling_results/profiling_tp_s04-p1-dgx-02-c14/; serial, GPU 0.

## MISSION
Find seqs where superoffload's capacity-capped B_max (<=4) sits below the throughput knee
(MFU(B_max) meaningfully under plateau ~15%): saturation deficit = 1 - MFU(B_max)/MFU_plateau.
At those seqs asym (leaner HBM/sample) fits 2-4x the batch and wins tok/s outright.

## ANCHOR-DERIVED PREDICTIONS (from c12 + prompt anchors; HBM = base + per_sample*B, per_sample ∝ seq)
- unsloth-ohbm0: 64k b8=150.0 GiB & b6=113.6 → per_sample(64k)=18.2, base≈4.4 → checks out vs 128k b4=149.9 anchor.
  - 160k: per_sample≈45.5 → b3≈141 GiB (76%) FIT expected; b4≈186 GiB > 185 phys → predicted OOM. Probe: b3.
  - 208k: per_sample≈59.2 → b2≈123 GiB (66%); b3≈182 (98%) thrash-zone. Probe: b2 (maybe b3 to bracket).
  - 256k: per_sample≈72.8 → b2≈150 GiB (81%). Probe: b2.
- recomp: 64k b4=120.7 & b6=181.0 → per_sample(64k)=30.15, base≈0.1 → checks out vs 128k b2=119.5 anchor.
  - 160k: per_sample≈75.4 → b2≈151 GiB (81%) FIT expected; b3 OOM. Probe: b2.
  - 208k: per_sample≈98 → b1≈98 GiB (53%); b2≈196 OOM. Probe: b1 (b2 only as OOM bracket if needed).
- asym Phase B (config from repo default RUNS: asym_cpuadamwds|recomp-off-full-fg-ker101-ceil0000-ohbm16|ligerloss1):
  anchor 174k|8 = 170.9 GiB → 160k b8 comfortably fits; run at confirmed gap seqs to show 2-4x batch + tok/s win.
  MUST rebuild first: bash scripts/lf/rebuild_asymgemm.sh (asym source changed 2026-07-17).

## STOP RULE
B_max hits 1-2 with deficit mapped, or even b1 OOMs. Every gap point must fit <=92% HBM, no thrash.

## PROBE LOG (chronological; every run gets a line — fits, OOMs, thrash, anomalies)
(started 2026-07-16 ~22:4x PT)
- R1 unsloth-ohbm0 160k b3 → FIT 140.9 GiB (76%), 184.6 s/it, 2601 tok/s, 13.2% MFU. Prediction (141 GiB/76%) dead-on; linear per-sample model validated at 160k. Deficit vs 15.1% plateau = 12.6% → **GAP @160k b3**. Deficit ≈ flat vs 128k b4 (13%) because tokens/step similar (480k vs 512k); expect deepening at B_max=2. s160000 dataset built fresh (~1024 samples); build warning "164244 > 131072" benign (tokenizer max-len notice during build).
- R2 recomp 160k b2 → FIT 151.0 GiB (82%), 124.1 s/it, 2578 tok/s, 13.1% MFU. Prediction (151/81%) dead-on again. Deficit vs 15.2% plateau = 13.8% → **GAP @160k b2**. Note: recomp deficit @160k b2 (13.8%) < @128k b2 (16%) — MFU drifts up with seq at fixed B as attention FLOP share grows; knee depth varies non-monotonically with seq at the SAME B_max. tok/s parity with unsloth b3 (2578 vs 2601) at 6 pp less HBM.
- R3 unsloth 208k b2 launched (pred ~123 GiB / 66%; b3 pred ~182 GiB / 98% = thrash zone, healthy-fit rule excludes it). s208000 dataset builds fresh.

- R3 unsloth 208k b2 → FIT 122.5 GiB (66%), 198.2 s/it, 2099 tok/s, 13.3% MFU. Prediction (123/66%) exact. Deficit vs 15.1% = 11.9%. **Gap NOT deepening with seq for q3-30b unsloth** (128k 13% → 160k 12.6% → 208k 11.9%): attention FLOP share rises with seq and props MFU at small B; contrast llama (17→20% growing). Plateau-reference caveat: using 64k-saturated 15.1%; the counterfactual b8 plateau at 160k+ is unobservable on superoffload (capacity-capped) — deficits vs a seq-matched plateau would be LARGER.

QUEUE (user asked to densify/squeeze the deep end — 180k added, 256k extension):
- R4 recomp 208k b1 → FIT 98.7 GiB (53%), 101.2 s/it, 2056 tok/s, 13.0% MFU. Pred (98/53%) exact. Deficit 14.5%. **recomp hits B_max=1 at 208k** (b2 pred 196 GiB OOM) → recomp stop-rule met; recomp deficits hover 14-16% across 128k-208k. At 53% HBM with B_max=1, capacity is wasted BELOW the knee — prime asym territory (asym b4-b8 at 208k pending Phase-B capacity check).
- R5 unsloth 180k b3 → FIT 157.4 GiB (85%), 229.1 s/it, 2357 tok/s, 13.2% MFU. Pred (158/85%) exact. Deficit 12.6% — unsloth deficit dead-flat 12-13% across 160-208k. B_max=3 holds thru 180k. RSS 382 GB (vs 279 at b3/160k — CPU offload pool scales with tokens).
- R6 recomp 180k b2 → FIT 169.7 GiB (92%) — prediction 169.7 EXACT to the decimal. 153.2 s/it (on token-scaled trend, zero thrash inflation), 2349 tok/s, 13.2% MFU, deficit 13.2%. B_max=2 holds at the healthy limit; recomp b2→b1 transition pinned to (180k, 208k], healthy-b2 ceiling ≈183-185k by the model.
- R7 unsloth 256k b2 → FIT 149.3 GiB (81%), 291.6 s/it, 1756 tok/s, 13.3% MFU. Pred (150/81%) exact (7/7). Deficit 11.9% — flat through 256k. s256000 dataset build ≈25 min.
- R8 recomp 256k b1 → FIT 119.9 GiB (65%), 146.7 s/it, 1745 tok/s, 13.3% MFU, deficit 12.5%. Pred (121/65%) exact — 8/8. Superoffload core sweep 160-256k COMPLETE.
- PHASE B start (user core goal: BEAT superoffload at 208k/256k; fallback: deeper b1-only regime + advocacy). R9 = rebuild_asymgemm (2026-07-17 staged/keep-acts source) && asym LATENCY mode (ker000-ohbm0 + staged + KEEP_ACTS_HBM[+LORA_A_FWD_GPU,DA_GPU,SCATTER_BLOCK=0] + chunk1024 + dx-staged) 208k b2 — direct head-to-head vs unsloth 208k b2 = 2099 tok/s. Beat targets: 2099 @208k, 1756 @256k.
- R9 asym-latency 208k b2 → FIT 78.1 GiB (42%), 220.9 s/it, 1883 tok/s, 11.9% MFU. Rebuild clean; "trainable parameters = 0" is normal asym-cpuadamwds logging (matches Jul-10 ceiling train.logs). EQUAL-BATCH head-to-head: asym −10.3% tok/s vs unsloth (1883 vs 2099) but at 42% vs 66% HBM — 44.4 GiB leaner, 107 GiB free. Massive improvement over pre-fix asym (~2x per-token deficit → now −10%). Latency per-sample ≈25.6 GiB (incl. kept acts) → B_pred=5 (~84%).
- R10 asym-latency 208k b5 launched — the batch-leverage attempt. Beat math: b5=1.04M tok/step; wins if s/it < 495.5 (i.e. any sublinearity vs 5/2×220.9=552 linear); fixed weight-stream+adam amortization should deliver. b6 pred ≈181 GiB (98%) — edge, not attempted first.
- PHASE-SPLIT ANALYSIS @208k b2 (from profile.json): asym-lat fwd 36.4 / bwd 184.1 / opt 1.3 vs unsloth fwd 31.3 / bwd 166.6 / opt ~0. Residual asym deficit is UNIFORM ~1.10-1.16x (pre-fix was 1.5-1.8x — the staged/keep-acts fix removed the stall). Visible batch-independent cost is small → b5 honest range 1990-2150 tok/s, right at the 2099 bar → backup levers staged:
  L1. ASYMM_QWEN3_MOE_ROUTE_LORA=1 — DEFAULT IS 0 (checked source); D1 receipt: 2.27 s/step fwd + 2-3x bwd of token-space LoRA fill/scatter at 256k tok/step → ≈30-36 s/step at b5. Biggest lever; loss-parity gate noted (numerics-touching; timing probes OK, flag it in results).
  L2. ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1 — default off; cuts bwd D2H round trips; costs HBM (fine at b4 70%, tight at b5 84%).
  L3. ASYMM_FG_ELEMENTWISE_CHUNK_MB=0 (dial-l4 rung) — removes chunking overhead.
  (ASYM_CPU_ADAMW_ASYNC_GRAD_OFFLOAD already defaults 1 = on.)
  Decision tree: b5 wins → lock + replicate 256k b4 (+ optional L1 for margin). b5 short → R11 = b5+L1 (one flag per A/B, repo discipline). b5 thrash/OOM → b4+L1+L2. Then 256k.
- R10 asym-latency 208k b5 → 180.1 GiB resv (97%) vs alloc 118.1 — 62 GiB allocator fragmentation gap (b2 gap was 27); s/it 557.9 = +1% OVER linear (552) → amortization nullified by near-ceiling reserve pressure; 1864 tok/s < b2's 1883. FINDING: asym-latency per-sample = 34.0 GiB RESERVED (not 25.6 alloc-based); healthy max = b4 (pred 146 GiB / 79%). The reserve wall, not compute, is b5's blocker → PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is the potential b5-rescue (alloc-only footprint would put b5 at ~125-135 GiB / 70%) — try after b4 verdict if more margin wanted.
- R11 launched: 208k b4 + L1(ROUTE_LORA=1) + L2(KEEP_DGRADS_HBM=1), tag tputasl2. Projection: linear b4 ≈422 s/it (1972 tok/s) − L1 ≈24-29s − L2 ≈10-20s → ~375-390 s/it ≈ 2130-2220 tok/s vs bar 2099. Two flags stacked deliberately (win-first, ablate-if-needed); both validated trap-free against source.
- R11 result: 445.7 s/it, 153.2 GiB (83%), 1867 tok/s, 11.8% MFU. Flags VERIFIED engaged (config tag `route000_lora1_accfp32`, snapshot route_lora:true keep_dgrads_hbm:1) — levers yielded ≈0 at this scale, and s/it is +5.6% OVER b2-linear.
- ⚠ EMERGING FINDING (208k tok/s-beat likely unwinnable by batch): asym-latency tok/s is BATCH-FLAT (b2 1883 / b4 1867 / b5 1864) because at 208k the workload is attention-FLOP-dominated (86% of FLOPs) and already GEMM-saturated at b1-b2 — there is no amortizable fixed cost worth >2% of the step. The unsloth "deficit" vs the 64k plateau is likewise an artifact: its 13.3% MFU at B_max is seq-matched-saturated. Batch leverage cannot beat a ~11% PER-TOKEN tax (fwd 1.16x, bwd 1.10x) — mirrors the prompt's llama-recomp CLOSED verdict, but for a different reason (batch-saturation, not OOM-death).
- R12 launched (final 208k attempt): b4 + full stack + L3(CHUNK_MB=0) + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (targets the 57 GiB reserve-fragmentation gap + its +5.6% pressure slowdown). Best-case ≈1990-2050 — closes 208k either way.
- GOAL-2 EXECUTION PLAN (pre-authorized fallback; starts after R12 regardless unless it wins):
  1. b1-collapse map: recomp 320k b1 (fit ~82%) + 400k b1 (expect OOM — death bracket, wall pred 390k); unsloth 320k b1, 480k b1, 600k b1 (thrash/wall pred ~590-635k).
  2. asym at the SAME seqs, healthy batch (expandable-segments alloc-based sizing): 320k b3 (~61% alloc), 480k b2, 600k b2 — near-parity tok/s at 2-3x batch where rivals strain/die; asym is the ONLY healthy system past recomp's 390k wall (vs recomp) and ~600k (vs unsloth).
  3. asym MEMORY-mode capacity crown at 320k (b6-b8, per-sample ≈2.4x leaner than unsloth).
  4. Advocacy assembly: capacity-per-GiB + max-seq exclusivity + near-parity per-token cost + the honest tok/s physics at batch-saturated seqs.
- R12 asym 208k b4 full-stack (chunk0+expandable): 484.1 s/it, 1719 tok/s — REGRESSION −8% (chunk0 inflates temporaries: alloc 96→108; expandable did not shrink resv 153→153). 208k CLOSED: best asym 1883 (b2) vs unsloth 2099. Reverted chunk to 1024 for all later runs.
- R13 recomp 320k b1 → FIT 151.0 (82%), 222.3 s/it, 1440 tok/s, 13.4% MFU (deficit 11.8%). Pred exact (9/9). NB: concurrent nice-19 dataset builds (480k/600k) did NOT contaminate: MFU rose vs 208k. b1 wall model → 392k; 400k = OOM bracket (dataset queued behind 480k/600k builds).
- R14 launched: unsloth 320k DESCENDING b2 (pred 186.4 > 185 phys → expected OOM = empirical b2-wall bracket {256k fit, 320k OOM}) then b1 (measurement; pred 95.4 GiB / 52%). Known cosmetic bug: the loop's OOM-grep reads the whole accumulated log, so the b1 marker line may mislabel; ground truth = per-dir profile.json + train.log (parser).
- R14 result: unsloth 320k b2 did NOT OOM — edge-fit 181.3 GiB (98%), 472.2 s/it (+5.9% over healthy-linear 446, sub-thrash), 1355 tok/s, 12.6% MFU. First capacity-model miss (pred 186.4): allocator bends ~5 GiB near the wall. Healthy-b2 ceiling stays ≈290-300k (92% line).
- R15 unsloth 320k b1 → FIT 94.8 (51%), 222.8 s/it, 1436 tok/s, 13.3% MFU. **KEY EXHIBIT: healthy b1 OUT-THROUGHPUTS edge b2 (1436 vs 1355)** — near-wall pressure inverts batch scaling; unsloth at 320k either idles half its HBM or loses tok/s.
- R16 asym-LATENCY-stack 320k b2 → FIT 117.5 GiB (63%), 479.2 s/it, 1336 tok/s, 12.4% MFU. Relative per-token tax vs superoffload·unsloth-ohbm0-healthy-linear = 7.4% (was 11% @208k) — tax DILUTES with seq. Runs 2× the batch of superoffload·recomp b1 @320k, both healthy.
- R17 superoffload·recomp 400k b1 → **OOM** (12.21 GiB short, 169.7 GiB allocated of 184). recomp ABSOLUTE WALL pinned to (320k, 400k], model 392k. First config fully closed.
- R18 superoffload·unsloth-ohbm0 480k b1 → FIT 141.1 GiB (76%), 484.8 s/it, 990 tok/s, 13.3% MFU. Pred exact.
- R19 asym-LATENCY-stack 480k b1 → FIT 90.2 GiB (49%), 506.5 s/it, 948 tok/s, 12.8% MFU. vs unsloth-ohbm0 b1 990 = **−4.2% tok/s** at 49% vs 76% HBM. Tax trend −10.3%(208k)→−7.0%(320k)→−4.2%(480k): converging to a crossover ~600–640k.
- R20 superoffload·unsloth-ohbm0 600k b1 → FIT 175.1 GiB (95% edge), 749.9 s/it (= healthy-linear 748: ZERO pressure penalty at 95%), 800 tok/s, 13.3% MFU. Edge is free at 600k → crossover vs asym-lat pushes to ~640k+; wall pred 640-660k.
- unsloth-OFF cost model (from prompt anchor 131k·b8=532s → 5.08e-4 s/tok): unsloth-off pays ~57% more per token than so-unsloth for its leaner HBM → asym-lat (−4% vs so-unsloth @480k) should BEAT unsloth-off on tok/s at every deep seq by ~40-50%; unsloth-off's edge is capacity only (b1 wall very deep, possibly host-RSS-limited). To MEASURE: unsloth-off 600k b1 + 320k desc.
- R21 launched: asym-LATENCY 600k b1 (pred ~118 GiB / 64%, ~783 s/it → ~766 tok/s vs so-unsloth's 800 = −4%). Then R22 unsloth-off 600k b1 head-to-head.

## ⚠ CONFIG-CLARITY AUDIT (2026-07-17, at user request — "recomp / unsloth / unsloth-off are very different and very critical")
Ground-truth verified from run-dir config tags + config.json (`use_unsloth_gc`, `unsloth_gc_recompute_save_on_cpu`); source parsing at profile_lora_lf_test_source.sh:1120-1132 / :1433 / :1491:
- **recomp** = use_unsloth_gc=false, save_on_cpu=false (plain gradient-checkpoint recompute). My `tputrc-c14` runs. ✔
- **unsloth-ohbm0** = use_unsloth_gc=true, save_on_cpu=**false** (unsloth GC, recompute-saved acts kept in HBM; b8 ceiling 80k). My `tput-c14` runs. ✔
- **unsloth-off-ohbm0** = use_unsloth_gc=true, save_on_cpu=**true** (acts OFFLOADED to CPU → far leaner HBM; b8 ceiling ~131k, then C-OOM = HOST-RAM limited). **NOT run on c14 yet.**
Finding: the c14 superoffload sweep covers recomp + unsloth-ohbm0 only (the two configs in throughput_prompt.md §PROTOCOL). unsloth-off is superoffload's leanest config and the fix_throughput.md advocacy baseline; the "asym trains where superoffload can't / asym wins capacity" claim is only honest if checked against unsloth-off (which walls MUCH deeper than the two configs tested). → unsloth-off frontier PROMOTED to next priority after R20. All tables rewritten with full `backend | recompute` config strings + a CONFIG LEGEND; bare "unsloth" eliminated.

## REVISED QUEUE (unsloth-off promoted per user)
1. superoffload·unsloth-ohbm0 600k b1 (R20, running) → 640k b1 (its edge/OOM wall).
2. **superoffload·unsloth-off-ohbm0 frontier** — 208k, 320k, 480k, 600k, 640k descending-B. The real advocacy yardstick. Note: unsloth-off may be HOST-RAM (RSS) limited at high batch, not HBM — watch RSS vs 1693 GB.
3. asym-LATENCY 600k / 640k b1 (crossover + exclusive-fit vs whichever superoffload config goes deepest).
4. asym-MEMORY stack (ker101-ohbm16): 320k big-batch capacity crown + 480k/600k exclusive-fit.
5. Variance spot-checks vs c12; archive; advocacy synthesis.
## ASYM PRE-FIX BASELINE (parsed from Jul-10 ceiling__ artifacts in live root; config ker101-ceil0000 [pre-ohbm naming], asym source BEFORE the 2026-07-17 fix — NOT Phase-B numbers)
| seq | B | s/it | resv GiB | %HBM | tok/s | MFU% | note |
|---|---|---|---|---|---|---|---|
| 128000 | 8 | 617.7 | 153.0 | 83% | 1658 | 7.0 | pre-fix |
| 132000 | 8 | 659.9 | 157.7 | 85% | 1600 | 7.0 | pre-fix |
| 140000 | 8 | 719.5 | 167.3 | 90% | 1557 | 7.1 | pre-fix |
| 156000 | 8 | — | — | — | — | — | NOPROF (0 measured steps — killed/bracket, not OOM-labeled) |
Implications: (1) pre-fix asym per-token throughput is ~2x BELOW superoffload at the same seq — the Phase-B throughput-win claim rides on the 2026-07-17 asym fix; rebuild is mandatory before any Phase-B run and these baselines give the before/after delta. (2) capacity slope 128→140k b8 ≈ 1.19 GiB/1k seq extrapolates to ~208 GiB @174k, inconsistent with the 174k|8=170.9 ohbm16 anchor → memory curve is config-sensitive/nonlinear; Phase-B capacity predictions must be refit from the FIRST rebuilt-asym run (start 160k b8, anchored safe), not from this curve.

## PROMPT v2 (2026-07-17) COMPLIANCE NOTES
- Re-read mid-campaign at user request: SEARCH PROTOCOL is now v2 "B1/B2-FIRST" — map the WHOLE b1/b2 frontier to b1's absolute OOM wall (win window [onset, wall]), large seq steps (q3-30b 40-80k), deficit-flat points are NOT stop signals. Queue extended accordingly (below).
- Table format: GAP WINDOW blocks per config (schema `| seq | B_max | resv GiB | %HBM | RSS GB | tok/s | MFU% | deficit | note |`, saturated lead-in rows included, memory columns never dropped, no cross-config ranking) — applied in test_throughput_results.md.
- `scripts/lf/tp_probe.sh` and `/workspace/AsymGEMM-SFT/.repair_dataset_info.py` referenced by v2 do NOT exist in this checkout — used the (also-in-prompt) launcher-template semantics instead: descending-B probes, first-fit-is-measurement, OOM-grep on logs, DATASET_OVERWRITE=false self-consistency. No dataset_info corruption observed (every build clean so far).

EXTENDED FRONTIER QUEUE (v2):
- unsloth: 256k b2 (in flight) → 288k b2 (pred ~168 GiB / 91% — pins healthy-b2 wall; pred-b2-wall 291k) → 320k b1 (pred ~95 / 52%) → 480k b1 (pred ~141 / 76%) → 620k b1 (pred ~181 / 98% edge) / 640k OOM bracket = b1 wall.
- recomp: 256k b1 (pred ~121 / 65%) → 320k b1 (pred ~151 / 82%) → 360k b1 (pred ~170 / 92% edge = healthy wall) → 400k b1 (pred ~189 → OOM bracket).
- Practicality guard: dataset build cost grows ~linearly with seq (s256000 build ≈ 25 min); 480k+ builds are 45-60+ min each — will report if wall-mapping cost balloons.

## VARIANCE TABLE vs c12 reference (deliverable 2 schema) — COMPLETE, all |Δ| ≤ 1.2%
| model | config | seq | B | tok/s here | tok/s c12 | Δ% | verdict-match? |
|---|---|---|---|---|---|---|---|
| q3-30b-a3b | superoffload_mem\|recomp | 45000 | 8 | 6735 | 6655 | +1.2% | yes (MAX, 92%) |
| q3-30b-a3b | superoffload_mem\|recomp | 64000 | 6 | 5894 | 5919 | −0.4% | yes (MAX, 98%) |
| q3-30b-a3b | superoffload_mem\|unsloth-ohbm0 | 45000 | 12 | 6768 | 6817 | −0.7% | yes (MAX; resv 157.4 = c12 exact) |
| q3-30b-a3b | superoffload_mem\|unsloth-ohbm0 | 64000 | 8 | 5862 | 5883 | −0.4% | yes (MAX; resv 150.0 = c12 exact) |
Cross-machine variance c14 vs c12: negligible (|Δ| ≤ 1.2%, well inside the 5% gate; peak-resv values bit-identical). c14 deep-end numbers are directly comparable to c12's reference tables.

## CROSS-MODEL CAPACITY CONFIRMS (q3-32b, run on c14 at user request — host-tagged tput*-c14_q3-32b; c12's model, deepest fitting point per config)
| model | config | seq | B | tok/s c14 | tok/s c12 | Δ% | resv c14/c12 | RSS c14/c12 |
|---|---|---|---|---|---|---|---|---|
| q3-32b | superoffload_mem\|unsloth-ohbm0 | 96000 | 3 | 1674 | 1680 | −0.4% | 145.2 / 145.2 (bit-match) | 364 / 364 |
| q3-32b | superoffload_mem\|recomp | 80000 | 2 | 1839 | 1846 | −0.4% | 174.3 / 174.3 (bit-match) | 143 / 142 |
c12's q3-32b capacity numbers CONFIRMED on c14 to the decimal — capacity frontiers are machine-independent on this cluster.

## GAP WINDOW blocks (final deliverable format)
Live in test_throughput_results.md (one block per config, v2 schema).
