# EP routing-skew screen — c12 ledger (surface_ep_skew.md campaign)

User directive 2026-08-11: run the three-agent screening split from
agent/impls/surface_ep_skew.md on this machine, in order agent 2 -> 3 -> 1:
- Agent B (=2): M2 GLM-4.7-Flash x {D1,D3,D8; D2,D4; D5,D6,D7,(D9)} + M3.2 Hunyuan-A13B x {D1,D3,D8}
- Agent C (=3): M3 Qwen3.5-122B x {D1,D3,D8} + M3.1 GLM-4.5-Air x {D1,D3,D8} (2-GPU shards)
- Agent A (=1): M1 Qwen3-30B-A3B x {D1,D3,D8; D2,D4; D5,D6,D7,(D9)}  (M4 gpt-oss stays deferred)
Context doc: agent/impls/fix_dynamic_ep.md (Dynamic EP = the Balancer this evidence motivates).

Probe: scripts/ep_skew/route_skew_probe.py (new). Capture = forward pre-hook on each
MoE block's `.experts` (actual routed indices, uniform across all 5 archs in this venv).
16k packs, per-dataset seeds, >=100 samples/cell, per-sample+per-doc counts, hot-GPU
share under contiguous E/2 partition, Zipf z at worst layer. LM head skipped.
Outputs: profiling_results/ep_skew/route_skew_<model>_<dataset>.json + __docs.npz +
__replay_{median,p95}.npz + logs/ + manifest.json (idempotent skip).

Container: asym_sft_40 one-shots (enroot store /scratch_local/user_data/kevinni/enroot),
mounts: AsymGEMM-SFT, env, shutian cache. HF_HOME=/scratch_local/user_data/shutian/kevin/
cache/huggingface (models cached there: GLM-4.7-Flash 59G, Qwen3-30B 57G, Qwen3.5-122B 234G).

Dataset ids resolved 2026-08-11 (scripts/ep_skew/dataset_ids.json):
D1 BytedTsinghua-SIA/DAPO-Math-17k · D2 SUBSTITUTE nvidia/Nemotron-Math-v2 ("Megatron-Math
Du et al 2025" has no public HF release) · D3 open-r1/codeforces · D4 princeton-nlp/SWE-bench
(test; issue+patch) · D5 Idavidrein/gpqa (gated=auto, token present, verify) · D6
nvidia/OpenScience · D7 zai-org/LongBench (THUDM renamed; data.zip direct) · D8 local
asym_long_sft_smoke__llama-3_3-70b-instruct__s16000__n1024.jsonl · D9 fineweb-2 cmn_Hani
(optional, only if P0 weak).

REGISTRY (final, per route_skew_probe.py — supersedes the id notes above):
models: glm4.7-flash · glm4.5-air · hunyuan-a13b · qwen3-30b · qwen3.5-122b.
datasets: dapo(D1) · megamath(D2; IFM/MegaMath megamath-web-pro/*.parquet — LLEP's
"Megatron-Math" has no public release) · codeforces(D3) · swebench(D4;
problem_statement+patch, train split) · gpqa(D5; gated OK via env token; only ~3
full 16k packs exist — runs with --num-samples 3, thin-tail caveat) · openscience
(D6; config OS-Q3-235B-4 required) · longbench(D7; data.zip direct, tasks mixed+
shuffled) · sft_mix(D8; smoltalk/longalign as the balanced public control) ·
(D9 dropped from registry — only if P0 weak). Chat kinds render via each model's
chat template; capture = pre-hook on `experts` modules (actual selected indices);
16384-token packs, 104 samples, B=8 (OOM auto-halve), contiguous E/2 partition.

## Log (append-only; c12)
- [08-11] Campaign start: probe (route_skew_probe.py, Kevin-revised) + driver written.
  Smoke: glm4.7-flash x dapo 4-pack run PASSED — median_max_hot 0.666, z~0.83 (46 MoE
  layers captured). Stage check: 6/8 adapters OK; fixed megamath (DATA_FILES pin) +
  openscience (config). Hunyuan-A13B downloaded (150G, 51s!).
- [08-11] WAVE B1 LAUNCH: glm4.7-flash x {dapo,megamath | codeforces,openscience |
  sft_mix,swebench | longbench} on GPUs 0-3 (4 replicas). gpqa follows at n=3.
  GLM-4.5-Air download started in background (206G, ~1 min).
  GOTCHA fixed in driver.sh: enroot nvidia hook FILTERS NVIDIA_VISIBLE_DEVICES and
  renumbers inside ids 0..k-1 — passing host idx as CUDA_VISIBLE_DEVICES gives 0
  visible devices -> silent CPU fallback (3 replicas killed+relaunched).
- [08-11] WAVE B1 DONE (glm4.7-flash, 104x16k packs, median/p95 max-layer hot-GPU
  share, zipf z at worst layer):
    dapo        0.7011 / 0.7102  z=0.95  (math wins)
    codeforces  0.6709 / 0.6858  z=0.95
    openscience 0.6599 / 0.6892  z=0.60
    swebench    0.6554 / 0.7592  z=0.65
    longbench   0.6394 / 0.8650  z=0.60  (mixed ctrl; heavy p95 tail)
    megamath    0.6374 / 0.8435  z=0.70  (web-pro; weaker than problem-style math)
    sft_mix     0.6354 / 0.7511  z=0.65  (balanced ctrl)
  => domain-pure math/code packs add ~+3.5..6.6pp over the ~63.5% inherent floor;
  fwd ~70-110s/cell. gpqa capacity = 3 full packs only (447 docs) -> n=3 rerun.
- [08-11] glm4.7-flash|gpqa n=3 DONE: 0.6509/0.6933 z=0.75 (thin-sample caveat).
- [08-11] WAVE B2 DONE (hunyuan-a13b, 32 MoE layers):
    sft_mix     0.5465 / 0.5733  z=0.30
    dapo        0.5404 / 0.5497  z=0.30
    codeforces  0.5361 / 0.5457  z=0.35
  => Hunyuan ~BALANCED on all packs (control >= math >= code; no domain signal) —
  negative result a la Mixtral: aux-loss recipes don't skew on domain-pure packs;
  GLM's loss-free router does. AGENT B (=2) COMPLETE.
- [08-11] AGENT C LAUNCH: qwen3.5-122b (GPUs 0-1, max-mem 115,170) + glm4.5-air
  (GPUs 2-3, same caps) x {dapo,codeforces,sft_mix}, simultaneous 2-GPU shards.
- [08-11] AGENT C + AGENT A cells found DONE in manifest — completed concurrently
  by the parallel session (incl. the 122B max-memory OOM fix + rerun). My relaunch
  no-op'd via the flock'd manifest (idempotency held). qwen3-30b|gpqa n=3 rerun by
  this session: 0.6670/0.6685 z=0.75.
- [08-11] GRID COMPLETE: 25/25 cells (5 models; gpqa thin n=3 x2 noted). Node
  quiesced, 547M banked in profiling_results/ep_skew (counts, per-doc sidecars,
  replay topk gz). 122B replay inputs already extracted to ep_skew/replay/ by the
  parallel session (ep_hist_real_qwen3.5-122b_{dapo,codeforces}_{median,p95}.json).
- [08-11] FINAL TABLE (median/p95 max-layer hot-GPU share; z at worst layer;
  1M_pred = domain-mean-hist hot share = natural-long-pack prediction):
    glm4.7-flash : dapo .7011/.7102 z.95 pred.7019 | codeforces .6709/.6858 z.95
      pred.6727 | openscience .6599 | swebench .6554 | gpqa(n3) .6509 | longbench
      .6394 (p95 .865!) | megamath .6374 | sft_mix .6354 pred.5743
    qwen3.5-122b : codeforces .6578/.6737 z.70 pred.6676 | dapo .6343 | sft_mix .6143
    qwen3-30b    : gpqa(n3) .6670 | longbench .6547 | swebench .6508 pred.6277 |
      openscience .6412 | sft_mix .6342 pred.5597 | dapo .6290 | codeforces .6128 |
      megamath .5999
    glm4.5-air   : codeforces .6107/.6217 z.73 pred.6160 | dapo .5855 | sft_mix .5761
    hunyuan-a13b : all ~.54 z.30 (BALANCED — negative; aux-loss recipe)
  VERDICTS: decision gate (>~55%) PASSED for all Balancer-claim models (M1/M2/M3)
  + Air; natural skew exists, curated fallback NOT needed. Winners: flash-dapo,
  122b-codeforces, 30b-swebench (robust n=104; its domain deltas are small — 30B
  skew is mostly inherent, LLEP-style). Real-z anchors: 0.5-0.95 (flash math .95
  ~= paper's z=1.0 point; 122b code .70). Controls at .54-.64 = inherent
  post-training imbalance replicated on our checkpoints. Stage-1.5: domain-pure
  skew SURVIVES doc-averaging (pred ~= per-pack median) while controls collapse
  -> natural 1M packs viable; no curation for winners. AGENTS 2, 3, 1 COMPLETE.
- [08-12] NEW MISSION (Kevin, prompt.md): curate 2-3 dataset sets per model whose
  1M-token b=1 packs sustain >=65/35 hot-GPU share AVERAGED OVER ALL MoE LAYERS
  (not worst-layer) for >=6 packs/set, EP=2. Analytic packing from per-doc router
  traces + real-forward verification. Order: flash -> qwen3-30b -> qwen3.5-122b.
  Plan: deep screens (n=200-500/dataset -> profiling_results/ep_skew_deep/, own
  manifest) -> greedy signed-signature pack construction (curate_packs.py,
  objective = mean over layers of |token-weighted signed half-deviation|) ->
  probe --pack-file real 1M forwards. Known bar: natural launch-avg ~0.55,
  kmeans-curated ~0.57-0.63 on small pools; pushing via bigger pools + direct
  greedy + cross-dataset mixes.
- [08-12] c11 CLAIM (parallel session, same mission): qwen3-30b lane — deep traces
  n=512 x {dapo,codeforces,swebench,openscience,megamath} -> profiling_results/
  ep_skew_deep/ (unified dir), then curate_packs.py sets + probe --pack-file 1M
  verification on this node. c12 keeps flash lane; 122b = whoever's first (check
  this ledger + ep_skew_deep/manifest.json before spending GPU). Probe patched:
  tokenizer/model local_files_only-first + dataset-load retry (4x, 429 storm from
  concurrent unauthenticated runners measured), content-hash doc ids for qa/text
  kinds (stable rebuild keys for streamed corpora).
- [08-12] c11 PIVOTAL ANALYSIS — doc-mix curation at the CONTIGUOUS split cannot
  reach 0.65 layer-avg: perfect-alignment ceilings from real per-doc traces are
  0.554-0.594 (30b/flash/122b; per-doc mean|lean| p50 ~0.05-0.08). BUT real EP
  fixes placement PER LAYER, and a fixed per-layer placement calibrated on HALF
  the stage-1 samples, evaluated on the HELD-OUT half (16k packs, layer-AVG hot):
    qwen3-30b: dapo .898 · codeforces .898 · swebench .909 · openscience .933 · sft_mix .703
    glm4.7-flash: dapo .821 · codeforces .763 · sft_mix .639
    qwen3.5-122b: dapo .848 · codeforces .820 · sft_mix .656
  (contiguous same-eval ~.53-.56 everywhere; p10 ~= median -> transfer robust).
  => the 65/35 target falls out of PLACEMENT x domain-pure data, not doc mixing.
  Plan: per-(model,set) partition JSON from deep-trace calibration + held-out-doc
  1M packs + probe --pack-file verification with PER-LAYER partition support
  (probe patch incoming, backward-compat). EPLB-style variant (balance placement
  on sft_mix, eval on domains) being computed as the defensible framing check.
- [08-12] FLASH deep screens r1 done (n=200-500): dapo .7011 (FULL 17k corpus,
  8.2M tok) | codeforces .6720 p95 .939 (5826 docs) | openscience .6657 p95 .918 |
  swebench .6588 | megamath .6402 | longbench .6398 | sft_mix .6336. Pool 36k docs
  40.9M tok. Greedy disjoint 1M packs (curate_packs.py): 0.554-0.611 avg — DILUTION
  kills it. Aligned-cluster ceiling (sign power-iteration): 0.6447 @50k unique.
  -> v2 (curate_v2.py): aligned cluster + alignment-weighted REPETITION to 1M
  (house pipeline already concat/repeats; some layers structurally ~0.50).
  Expanded screens launched --overwrite: swebench n1500 gpu0, openscience n1500
  gpu1, megamath n1000 gpu2, codeforces n500 gpu3. Next: rerun v2 on bigger pool;
  if >=0.65 pred -> verify via probe --pack-file (NEW, additive) at seq 1M B=1
  n=6/set; metric = summary.mean_layer_avg_hot_gpu_share (NEW). Then 30B, 122B.
- [08-12] Expanded screens DONE: swebench n1500 .7543(!) 2532 docs | openscience
  n1500 .6660 12287 | megamath n1000 .6220 18637 | codeforces FULL .6711 9548.
  curate_v2 run on ~60k-doc pool — results in packs2_*.json; verify next via
  probe --pack-file seq 1M B=1 (3 GPUs, one set each), metric mean_layer_avg_hot.
- [08-12] CURATE v2 (flash, expanded pool): codemix pred 0.6512 >=0.65 TARGET MET
  (44 uniq docs ~30k tok, weighted repeats); allmix 0.6512 (SAME cluster — must
  exclude codemix docs for a distinct 3rd set); mathmix 0.6474 (short by 0.3pp —
  try deeper dapo/openscience tails or per-layer seed refinement). min_l ~0.50
  layers are structural. NEXT: (1) verify codemix packs REAL: driver.sh
  glm4.7-flash <gpu> curated_codemix --pack-file .../packs2_glm4.7-flash_codemix.json
  --seq-len 1000000 --batch-size 1 --out .../ep_skew_deep — check
  summary.mean_layer_avg_hot_gpu_share >= 0.65 in the cell JSON; (2) allmix rerun
  with codemix-cluster docs EXCLUDED (edit curate_v2 pool filter) for set #3;
  (3) push mathmix over 0.65; then qwen3-30b (top_k=8), then qwen3.5-122b.
- [08-12] SET STATUS (flash, analytic): codemix packs2 0.6512 MET | mathmix
  packs4 0.6518 MET (frac 0.45) | allmix2-excluded 0.6466 short — needs deeper
  megamath n3000 + openscience n4000 screens (unbounded corpora) once GPUs free,
  then re-curate. swemix capped 0.6411 (its .754 was single-layer). VERIFY RUNS
  LIVE: codemix packs2 (GPUs 0,1) + mathmix packs3 0.6491-pred (GPUs 2,3;
  doubles as predictor calibration; rerun packs4 after). OOM fix: 2-GPU shard +
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (now default in run_cell.sh).
- [08-12] c11 qwen3-30b 1M VERIFICATION (real forwards, b=1, seq=1048576, 2-GPU
  shard, experts row-sliced at 256k rows — the HF experts module allocates a
  64 GiB fp32 [rows,hidden] transient at 1M unsliced; slicing is exact):
    science (openscience): 6/6 packs layer-avg 0.753-0.760 MEAN 0.757 ✓ (max-layer .989)
    math (dapo):           6/6 packs layer-avg 0.821-0.826 MEAN 0.824 ✓ (max-layer .928)
  Analytic->1M attenuation: science .933->.757, math .897->.824 — margin holds
  every pack >=0.65. code + mathmix verifying now. Note: deep traces done for
  all 5 datasets (dapo 17k docs / codeforces 9.5k / swebench 2.5k /
  openscience 4.2k / megamath 8.8k).
- [08-12] c11 QWEN3-30B COMPLETE — 4 verified sets (target 2-3), each 6x ~1M b=1
  packs, MEASURED layer-avg hot (>=0.65 bar): code(codeforces+swebench) 0.857 ·
  math(dapo) 0.824 · mathmix(dapo+megamath, full-1M distinct docs) 0.810 ·
  science(openscience) 0.757. Max-layer per pack 0.93-0.99. Artifacts:
  profiling_results/ep_skew_deep/placed/{packs,partition,route_skew}_qwen3-30b_*.
  122B lane started: deep n=256 dapo DONE (13.9k docs), codeforces + openscience
  tracing on both GPU pairs.
- [08-12] 1M VERIFICATION (flash, 6 packs/set, 2-GPU shard, ~2h/set):
  codemix MEASURED layer-avg 0.5875 (pred 0.6512), med max-layer 0.7589
  mathmix MEASURED layer-avg 0.5812 (pred 0.6491), med max-layer 0.7800
  Time-weighted (MoE-time fallback): 0.5936 / 0.5884. layers>=0.65: 7 & 4 of 46.
  => doc-signature ADDITIVITY BREAKS at 1M (-6pp): long-context attention
  homogenizes hidden states, softening routing vs 16k-measured signatures.
  Curated still beats natural 16k layer-avg (0.553) by +3pp. Worst-layer skew
  at 1M is STRONG and verified (0.76-0.78 sustained over 6 packs).
  ITERATION 2 LIVE: natural-1M screens (dapo,codeforces | swebench,openscience;
  6x1M packs each, out=ep_skew_1m) to harvest IN-REGIME per-doc signatures ->
  re-curate from those -> re-verify. If iter-2 plateaus ~0.60, that is the
  measured curation ceiling at EP=2/1M for flash (report honestly to Kevin).
- [08-12] c11 -> c12 FLASH HANDOFF NOTE: your codemix/mathmix 1M plateau (0.58-0.59
  measured) is the CONTIGUOUS-split ceiling from the 08-12 pivotal entry (perfect-
  align bound 0.594 for flash) — additivity break makes it worse, iter-2 in-regime
  signatures cannot beat the bound. The PLACEMENT construction sidesteps it:
  per-layer top-E/2 partition from calibration-half traces (your held-out eval:
  flash dapo .821 / codeforces .763). Placed sets NOW BUILT from your deep pools:
  packs_glm4.7-flash_{math .8207, code .7260, science .8315}.json + partition_*.json
  in ep_skew_deep/placed/ (calibration/pack docs disjoint via hash parity).
  Expected 1M landing after observed attenuation (30b -.07..-.17, 122b ~-.02 at
  131k-window): math/science comfortably >=0.65, code borderline (fallback:
  frac-tightened selection). Verify with probe --pack-file + --partition (per-layer
  dict format supported) — c11 will run these if GPUs free first; manifest dedupes.
  Also: 122b MEASURED (131k-window aggregate on 1M packs, spec-recorded): math
  0.8223 ✓ code 0.8085 ✓ (science leg re-running after the stale-capture fix
  e231319; true-1M single-shot infeasible in vanilla HF for the hybrid — deltanet
  live-set ~100GiB/GPU; window mode is the bounded-memory proxy). NOTE for this
  node (c11, 64k-page ARM kernel): expandable_segments correlated with
  12-16GiB-alloc failures at 40+GiB free — dropped here; your node may differ.
- [08-12] NATURAL-1M baselines (flash, 6x1M each): dapo layer-avg .5886 (med-max
  .8681!) | codeforces .5787 | openscience .5725 | swebench .5630. ITERATION-2
  curation from IN-REGIME (1M) signatures: codemix1m pred .6612, mathmix1m pred
  .6507 (3-doc clusters, frac .45; disjoint sets). VERIFY LIVE on both GPU pairs
  (out=ep_skew_1m, cells curated_codemix1m/curated_mathmix1m). If measured >=.65:
  2 sets banked for flash -> build 3rd (allmix1m needs exclusion rerun) -> move
  to qwen3-30b deep screens. If short: gap = self-context vs natural-mix-context
  signature drift; next lever = signatures FROM curated-pack verifications
  (iterate once more) or accept measured ceiling.
- [08-12] ITER-2 MEASURED (in-regime signatures): codemix1m 0.5651 (pred .6612),
  mathmix1m 0.5823 (pred .6507) — NO better than iter-1; predictions overshoot
  in both regimes. 8 measured 1M configs span 0.563-0.589; natural dapo best
  (0.5886, med-max 0.8681). VERDICT: ~0.59 = data-side layer-avg ceiling for
  flash @1M/EP=2; 0.65 all-layer avg unreachable by packing (balanced early
  layers + long-context homogenization). Doc updated with verdict + savings
  columns (f_moe/f_e2e placeholders; measurement task handed off in
  measure_moe_time_fractions.md). AWAITING Kevin decision: accept 0.588 avg +
  0.72-0.87 worst-layer as the stress deliverable (=> ~15% expert-GEMM, ~5%
  e2e est.) or redefine target before starting 30B/122B grind (their previews
  cap LOWER: 30b ~0.56, 122b ~0.54 at 16k).
- [08-12] QWEN3-30B natural-1M DONE (6x1M each): swebench gemm-avg .5782
  med-max .7106 | dapo .5742/.7559 | codeforces .5691/.7405 | openscience
  .5585/.7374. 1M RAISES 30B skew vs 16k (.54->.58) — same direction as flash;
  same ~0.58 plateau. No curation round (flash proved it mispredicts).
  QWEN3.5-122B natural-1M launched (codeforces,dapo | openscience,sft_mix;
  2-GPU shards, max-mem 115,170) — the last model stage.
- [08-12] c11 FLASH placed 1M VERIFIED (true 1M single-shot, 2-GPU shard, experts
  row-sliced; fwd ~8.7ks/set): math gemm-avg 0.7933 (packs 0.7929-0.7939, med-max
  0.9715, z1.15) ✓ · science 0.6682 (packs 0.661-0.678, ALL >=0.65, med-max 0.9023)
  ✓ — science attenuated -0.16 from pred vs math -0.03. code (pred 0.726) running.
  122b science also VERIFIED: 0.7656 (w131k). Report rows in skewness_results_39.md
  (gemm-avg terminology per Kevin).
- [08-12] c11 CAMPAIGN COMPLETE — flash code VERIFIED 0.7396 gemm-avg (packs
  0.728-0.755, med-max 0.929, landed +0.01 ABOVE pred). Final tally, all true-1M
  b=1 measured (122b via w131k proxy), every one of 60 packs >=0.65:
    qwen3-30b   : code .857 · math .824 · mathmix .810 · science .757  (4 sets)
    qwen3.5-122b: math .822 · code .809 · science .766                 (3 sets)
    glm4.7-flash: math .793 · code .740 · science .668                 (3 sets)
  Full report: agent/impls/s04-p1-dgx-02-c06/skewness_results_39.md (gemm-avg
  terms). Artifacts: ep_skew_deep/placed/{packs,partition,route_skew}_*.json —
  packs rebuildable from doc ids; partitions are the per-layer placements the
  6-step e2e runs should install.
- [08-13] c11 FIG13 (fig:ablation-balancer) REFRESHED WITH REAL MEASUREMENTS —
  ep_balance_bench scope=experts timed EVERY MoE layer's grouped launch on the
  actual routed counts of one held-out curated pack per (model x math/code/
  science), at each model's banked 2-rank anchor (30b 384k b1 m=6.14M · 122b
  320k b1 m=5.12M · flash 320k b2 m=5.12M). Arms: owned@placed partition
  (Static EP, hot share .67-.86) · plan (DSEP) · owned@hindsight-balanced
  (Oracle) · owned@contig (banked reference). RESULTS (sweep-sum walls):
  DSEP recovers 25.1-38.3% of expert-GEMM time vs static — matches the
  analytic (h-0.5)/h at each pack's measured gemm-avg to 0-3pp (flash EXACT:
  25.1/25.1, 32.4/32.2, 37.2/37.0; 30b/122b ~3pp under = plan chunking
  overhead). DSEP ≈ oracle everywhere and BEATS oracle 4.7% on flash-code
  (E=64 + z~1: hindsight split cannot balance; per-launch cut can). e2e row:
  at ceiling lengths attention dominates (expert sweep ~2% of step measured)
  -> +0.2..1.3% e2e; caption/body updated to the honest composition (F=2
  fwd+dgrad, solo streaming = conservative). THEORY CHECK PASSED. Artifacts:
  fig13/{fig13_*,walls_*,fig13_data.json}; scripts fig13_hists.py,
  fig13_aggregate.py, fig13_bench_driver.sh; plot_balancer_e2e.py rewritten
  (3-model measured). Overleaf pushed 832ecee (figure+caption+body).
  BENCH BUGFIX: ep_balance_bench parent tag was clobbered to "nat" by the
  alphas loop -> concurrent parents collided on /dev/shm/asym_epbench_nat;
  loop var renamed (pid-unique shm restored).
- [08-13] c11 128k b1 MoE-SHARE MEASUREMENT (Kevin's question "is expert time
  really ~2% at long ctx"). Real 2-step 2-rank anchor cells run on this node
  (asym_sdp2, b1, 128000): 30b T2 step 48.62s = 5265 tok/s · 122b T1 265.4s =
  964 tok/s · flash T1 109.1s = 2347 tok/s. Walls re-measured at the b1 launch
  scale (m=2.05M/2.05M/1.02M; walls128_*): mechanism savings 22-38% still
  tracks (h-0.5)/h (compressed ~3pp by small-m launch floor). MEASURED expert-
  GEMM share of step at 128k b1 (F=2): 30b 4.4% · flash 1.7% · 122b 1.4% ->
  DSEP e2e worth +2.4% / +0.9% / +0.8% at 128k b1. Physics: attention is still
  10-30x expert FLOPs at 128k (crossover ~5-9k tok), and 122b's step is
  dominated by its fixed per-step weight-streaming floor at b1 token counts.
  Earlier 320-384k row confirmed: share ~2%, e2e +0.2-1.3%. fig13_data_128k.json
  banked; figure kept at anchor lengths (measured style, no est borders).
- [08-14] c11 SHARE-VS-LENGTH SWEEP (Kevin: 48/64/80/100k, b1) — 12 real 2-step
  anchor cells + walls at each true launch size (wallsS*_math_placed).
  Expert-GEMM share of step / DSEP e2e gain (F=2, math domain):
    qwen3-30b   : 48k 7.9%/3.6% · 64k 7.6%/3.7% · 80k 7.0%/3.6% · 100k 6.2%/3.3% · 128k 4.4%/2.4%
    glm4.7-flash: 48k 3.5%/1.4% · 64k 2.9%/1.4% · 80k 2.5%/1.2% · 100k 2.1%/1.0% · 128k 1.7%/0.9%
    qwen3.5-122b: 48k 0.7%/0.2% · 64k 0.9%/0.3% · 80k 1.0%/0.4% · 100k 1.1%/0.6% · 128k 1.4%/0.8%
  READING: (1) 30b/flash shares RISE as seq shrinks (attention quadratic) but
  plateau near 8%/3.5% — the asym per-step weight-streaming floor + attention
  (crossover ~5-9k) cap it; (2) 122b is floor-FLAT: step ~250s from 48k to
  128k (fixed ~230GB/step weight streaming; share INVERTS — shrinks at shorter
  seq; tok/s 384@48k vs 964@128k — b1 short-seq is the wrong operating point
  for it); (3) the "MoE ~40% of step" regime needs ~4-8k seq AND deep batch
  (streaming amortized) — pretraining-shaped, outside this paper's long-ctx
  b1 cells. DSEP e2e gain peaks ~3.6-3.7% at 30b 48-64k b1.
  Data: fig13/share_vs_length.json; cells fs{48,64,80,100}{q30b,gflash,q122b}.
- [08-15] c11 NOTE: peer artifact migration (08-14) froze profiling_results into
  asymlora/history/sft and re-pointed the repo path at the empty live root —
  ep_skew{,_deep,_1m}+motivation hardlinked back into live (zero-copy) so all
  campaign scripts/paths work; new writes land in live as real files.
- [08-15] c11 SHORT-SEQ DEEP-BATCH (tokens/step fixed at 256k -> walls128 reuse,
  same launch size): measured share / DSEP e2e gain:
    qwen3-30b   : 32k b4 9.8%/5.3% · 16k b8 11.2%/6.1% · 8k b16 11.9%/6.5%
    glm4.7-flash: 32k b4 5.0%/2.5% · 16k b8 7.5%/3.8% · 8k b16 10.0%/5.1%
    qwen3.5-122b: 16k b8 1.4%/0.8% (streaming-floor-bound at every shape)
  -> "obvious" gains land: 30b +6.5%, flash +5.1%; 4k b32 rung running.
  gpt-oss-20b lane opened: ckpt downloaded (24L/32E/top4/ctx128k, MXFP4->bf16),
  probe registry entry added (GptOssExperts hook-compatible).
- [08-15] c11 GPT-OSS-20B LANE COMPLETE (same drill, one day): deep traces
  n=512 x {dapo .850, megamath .887, codeforces .893, openscience .612
  med-max contig — the LLEP inherent-imbalance model confirmed on our probe};
  placed sets at ctx-max 128k packs: math/code/science pred .867/.867/.877,
  VERIFIED .868/.866/.877 (16k-window aggregate — hf eager-sinks attention
  cannot single-shot 128k; ZERO attenuation). fig13 hists + walls (geom
  32,2880,2880 topk4, m=256k union) + vanilla-PEFT LoRA reference step
  (4k b8 GC, 9.51s, eager attention — gpt-oss NOT in the LF stack; F=3
  fwd+recompute+dgrad documented): mechanism 38.7-41.0% (theory 42.5%),
  share 10.3-10.5%, E2E GAIN +6.6/+7.0/+7.2% (math/code/science) — DSEP
  BEATS the hindsight oracle on ALL 3 domains (E=32 concentrated routing).
  Probe fixes en route: num_local_experts in infer, eager-OOM lessons.
  CAMPAIGN GOAL MET — obvious e2e gains: gpt-oss +7.2%, 30b +6.5-6.7%,
  flash +5.1-6.0% (short-seq deep-batch), 122b ~1% (streaming-floor, honest).
