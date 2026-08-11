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
