# GLM throughput-panel campaign (4 subplots: Flash 1r/2r, Air 1r/2r)
(2026-07-30 user directive: house-quality tp-vs-seq panels for BOTH GLMs at
1 rank AND 2 ranks (streaming-EP/DP, GPU_POOL=0,1), >6 seq rungs each,
batches saturated-not-meticulous (tokens/step >= ~256k target), turning
points + walls visible, asym advertised honestly. ORDER: Flash 1r -> Flash
2r -> ONLY THEN Air 1r -> Air 2r. Fair-comparison rule applies: verdict
configs only (NO qchunk; plain T1/T2/T3-raw + rc/uns/uns-off baselines).
Don't stop until all 4 panels render with supporting numbers.)

## Rungs & batch walk-lists (same list all systems; walk down on OOM;
## asym = T1 first, tier-promote on full-list OOM)
FLASH (ctx 202k, all in-context):
  32k:[12 8 6 4]  64k:[6 4 3 2]  96k:[4 3 2]  128k:[3 2 1]
  160k:[2 1]  192k:[4 2 1]  (+ existing: uns-off 192k b2-b5, T3 192k b2/b5,
  dev 8k pair — harvest, don't rerun)
AIR (ctx 131k):
  16k:[16 12 8]  32k:[12 8 6 4]  48k:[8 6 4]  64k:[6 4 3 2]  96k:[4 3 2]
  128k:[3 2 1]  (+ existing: uns-off 128k b2, T3 128k b2/b3, dev 8k pair)

## Phases & lanes
F1 Flash 1r: lane A GPU0 [192k, 96k, 32k], lane B GPU1 [160k, 128k, 64k];
   per rung serial: rc -> uns -> uns_off -> asym(T1->T2->T3).
   Lane host floor 500 (Flash cells 250-770 GB; worst pair ~1500/1690).
F2 Flash 2r: both GPUs, serial; same rungs; per-rank batch = F1 best-fit
   list head; ranks=2 via RUNS "model|2" + GPU_POOL=0,1.
A1/A2 Air: same structure after F2 (Air heavier: lane floor 600, the two
   192k... n/a; heaviest pairs ~870+890 — floor keeps them apart if needed).

## Tags
f1<sys><seq> / f2<sys><seq> / a1<sys><seq> / a2<sys><seq>, sys in
{rc,un,uo,t1,t2,t3}; e.g. f1rc32, f2t1160. Status log: tpfig_status.log.
Markers: F1-LANEA-DONE / F1-LANEB-DONE / F1-DONE / F2-DONE / A1-* / A2-* /
GLMTP-ALL-DONE.

## Plot integration
1r: DATA["glm4.7-flash"], DATA["glm4.5-air"] in plot_tp_vs_seq.py (+ COMBINED
grows or stays — the 4 GLM panels may become their own figure; decide at
render: plot_tp_vs_seq.py single-model outputs + tp2r variants from
plot_tp_vs_seq_2r.py). Cells = eff tok/s (step_samples, w1+m2). Walls red
OOM; est cells only where house-legal.

## EXT phase (2026-08-04, run_baselines session) — TURNING POINTS
User: GLM panels must show the walls + asym-last-standing like every other
model; the in-context cap above was wrong for the paper (qwen/llama panels
run far beyond native ctx). glmext.sh (anchors_tmp) extends: Flash 1r
256/320/384/448k, Flash 2r 256/320/416/512k, Air 1r+2r 160/192/256/320k;
systems rc/un/uo + fsdp2 (Flash only — Air fsdp2 is load-phase host-dead
at 16k, seq-independent) + asym T1→T2→T3. Tags x1/x2/y1/y2 + sys + sk.
Same lib/verdicts/status log; SOLO serial. Banking after: extend seqs +
all series in both DATA dicts, LEAN_DROP re-pick ~6 rendered cols.

## State — CAMPAIGN COMPLETE 2026-07-31 16:4x (~330 cells over ~35 h)
   (superseded on seq range by the EXT phase above)
- [x] F1 + solo redo (lane contention lesson: throughput cells must run
  SOLO; the parallel-lane originals were off by up to -44%)
- [x] F2 + patches (NCCL: ddp_timeout 7200 for slow steps, 1500 for A2
  hang-caps; asym 2-rank backend = asym_sdp2_cpuadamwds — ep2-vanilla's
  expert-slice swap is qwen3-block-only; ASYM_ARENA_SHM_CAP_GB=240 for Air's
  ~200 GB banks)
- [x] A1 serial + b1 patch  - [x] A2 v3 + final patch + retry round
- [x] 4 panels rendered & verified  - [x] tp_glm_combined.pdf (2x2 deliverable)
- [x] overleaf figures synced (5 PDFs)  - [x] docs + memory updated
Data quality: EVERY plotted cell measured solo/serial; zero est cells.
RESULTS (best-batch eff tok/s; 2r = global):
- Flash 1r: asym T1 leads 4/6 rungs (2147/1186/964/812 @64/128/160/192k),
  ties 32k/96k; uns RED-OOM @192k; rc/uns b1-crawl >=128k vs T1 b2-b3.
- Flash 2r (sdp2): asym leads 5/6 (7400...1934); 192k asym 1526 b4/rank
  (97% HBM) vs rc/uns 1560 b1 (-2% tok/s at +4x batch; both first-fit).
- Air 1r: asym T1 leads ALL SIX (3352/2803/2411/2095/1692/1087) with top
  batch everywhere.
- Air 2r (sdp2): asym leads ALL SIX by +13-45% (6844/5646/4730/4156/3302/
  2162), DP scaling 97-102% vs measured 1r; rc/uns replicate ~480 GB host
  per rank vs ONE shared fabric bank; uns-OFF host-DEAD 96k/128k (+48k
  measured x2, non-monotone vs its 64k fit — host-transient class).
