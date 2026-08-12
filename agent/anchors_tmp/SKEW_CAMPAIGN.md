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
