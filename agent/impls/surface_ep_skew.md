# Surfacing real EP skew to motivate Dynamic EP

2026-08-11. Context, findings, and options for replacing/anchoring the
synthetic routing-skew axis in the Dynamic EP (Balancer) evidence.

## What we are working on

AsymLoRA's Balancer re-derives a balanced expert-work cut per grouped
launch across the two GB200 GPUs (shared list, collective-free). Its
current evidence in the paper:

- **Fig 5 (motivation, measured):** Qwen3.5-122B expert-GEMM microbench,
  EP vs sDP vs our cut — but skew is **synthetic** Zipf z ∈ {0.5, 1.0,
  1.5, 2.0} (3 seeds; walls EP 29.3/32.1/37.7/44.9 ms vs ours
  29.0/29.4/29.8/36.1 ms).
- **Fig 12 right panel (ablation, estimated):** e2e tok/s derived from
  those walls (anchor 1540 tok/s @ 2xGB200 320k, expert share ~0.5) —
  EP 1532/1462/1339/1209 vs ours 1540/1529/1519/1372 (↑5/13/13%).

Gap: no **measured** e2e on/off pair, and no **real-data** skew — a
reviewer from the EP-balancing community will flag both. Goal: obtain
routing skew from a real-world dataset on a model we already run, then
measure (or at least anchor) the EP-vs-Dynamic-EP delta on it.

## Why this is nontrivial

Skew = balancing recipe x batch domain-purity, not the dataset alone.

- Old EP papers (FasterMoE/SmartMoE/FlexMoE/NetMoE/MegaBlocks) train
  MoEs **from scratch** (Wikipedia / The Pile / unnamed), where young
  routers are inherently imbalanced — not reproducible on a mature
  pretrained checkpoint like ours. Several name no dataset at all
  (NetMoE) or replay collected router traces (SmartMoE `moe_trace/`).
- Mixtral-class checkpoints (per-batch aux loss) barely skew on any
  topic (their own paper) — avoid.
- Qwen3 MoE lineage uses **global-batch** load-balancing loss (Qwen
  "Demons in the Detail", ACL'25): balance holds only corpus-wide, so a
  domain-pure batch (code/math/non-English) routes to concentrated
  experts. GLM's loss-free (DeepSeek-style) recipe skews similarly —
  that is why DeepSeek built EPLB.

## What the 2026 EP papers actually use (verified)

| Paper | Model | Data | Skew source |
|---|---|---|---|
| LLEP (arXiv 2601.17111, Salesforce) | gpt-oss-20b/120b (real ckpt) | Megatron-Math (public) | **Inherent** — E11 ~20% vs 3% balanced; GPU0 30-35% vs 12.5% under 8-way EP; "imbalanced inherently even on in-domain data"; post-training setting |
| UltraEP (2606.04101) | Qwen3-235B, GLM4.7-358B, DS-V3 | Serving: Codeforces, SWE-bench, DAPO-Math-17K, GPQA, LongBench — record real routing trace, replay across balancers. Training: in-house corpus | Real traces on public data |
| TAOT (2608.03676) | **Qwen3-30B-A3B (our model)** | Pile-test ("representative routing traffic") | **Synthetic injected** 10-70% imbalance, following LLEP protocol |
| LAER-MoE (2602.11686) | Mixtral-arch 8x7B/22B | Wikitext, C4 | Real traces from own training |
| FEPLB (2604.19654) | GLM-5 variant | Internal traces (7k iters) | Real, private |

Field consensus: math/code/reasoning datasets + router-trace replay;
synthetic controlled injection remains accepted (TAOT, on our exact
checkpoint, Aug 2026).

**Confirmed natural-skew pairs (checkpoint x dataset, verified):**
- gpt-oss-20b/120b x math (Megatron-Math): E11 ~20% vs ~3% balanced,
  hot GPU 30-35% vs 12.5% under 8-way EP (LLEP) — the only pair with
  public checkpoint + public dataset + printed magnitudes.
- Qwen3-235B x science/coding/mixed serving traffic: imbalance ratio
  ~1.5-2.5 across domains at EP=64 (UltraEP Fig 4) — family-level;
  ratios not attributed to specific datasets.
- Mixtral x The Pile: confirmed WEAK topic skew (negative pair).
- NOT confirmed anywhere: our exact checkpoints (Qwen3-30B-A3B,
  GLM-4.7-Flash, Qwen3.5-122B) on any named dataset; GLM4.7-358B has
  no individual imbalance number in UltraEP; LAER-MoE's Mixtral
  imbalance is from their own training (init unspecified); TAOT
  injects its imbalance. Hence the probe must screen ALL our
  Balancer-claim models — Qwen3-30B-A3B, GLM-4.7-Flash, Qwen3.5-122B
  (the EP-microbench model) — optionally with gpt-oss-20b as the
  literature-anchored reference.

## Options

**A. gpt-oss-20b as skew-demo model (zero risk).** Inherent ~2.6x GPU
imbalance on math already published (LLEP), post-training regime like
ours. Cost: adds a model to the paper for one panel.

**B. Our models + the 2026 papers' datasets (preferred).**
The complete explicit dataset list across the recent papers (verbatim
sources below): Codeforces, SWE-bench, DAPO-Math-17K, GPQA,
OpenScience, LongBench (UltraEP serving); Megatron-Math with
self-generated responses, AIME'25 as the accuracy metric (LLEP SFT);
Pile-test (TAOT); Wikitext, C4 (LAER-MoE). Every training-side result
uses private in-house corpora. Run Qwen3-30B-A3B / GLM-4.7-Flash on
the math/code subset of that list — same domains, same model families
as UltraEP. Long-context caveat: no EP paper exceeds "tens of
thousands of tokens"; packing domain-pure long sequences (repo-level
code, full papers, same-domain concatenation) is OUR addition, though
standard LM packing practice. Needs one ~30-min probe: a few forward
steps, dump per-layer router histograms, pick the skewest domain; the
histogram doubles as the "real skew ~ z=…" anchor for Figs 5/12.

Verbatim quotes worth citing:
- UltraEP: "We construct queries from realistic reasoning workloads:
  (1) STEM, including coding (Codeforces, SWE-bench), mathematics
  (DAPO-Math-17K), science (GPQA, OpenScience), and (2) Mixed, with
  additional multi-task, long-context queries from LongBench";
  training runs use "a 200B-token subset from our in-house corpus".
- LLEP: throughput "on samples from the Megatron-Math dataset (Du et
  al., 2025), where the responses were generated by gpt-oss-120b
  itself"; imbalance measured "across batches of a math dataset";
  e2e = full-param SFT of gpt-oss-20b scored by AIME'25 vs wall-clock.
- LLEP motivation sentence that matches our setting exactly: "when an
  MoE model is further fine-tuned or evaluated on a specific domain,
  such as mathematics, experts specialized for that domain are
  activated more frequently. This leads to imbalanced routing" — and
  "during post-training or inference, parameter-altering load
  balancing, like auxiliary losses, is discouraged ... Standard EP,
  as such, is not designed to handle this phenomenon."

**C. Keep the synthetic z-sweep, cite precedent (zero work).** TAOT +
LLEP justify controlled injection as the sensitivity axis. Weakest
alone; fine as the backbone.

**D. Trace replay (methodology stamp).** Record the real router trace
from a few SFT steps on B's data; replay it through static EP vs
Dynamic EP in the microbench harness — the UltraEP/SmartMoE/FEPLB
protocol. Turns B's probe into "real routing trace" evidence and gives
the measured EP-vs-ours delta at real skew.

## Screening protocol (find the skew-makers, then compare throughput)

1. **Router screen** (minutes/dataset; forward-only, no labels;
   single GPU for small models, sharded for the big ones).
   Candidates: DAPO-Math-17K, Megatron-Math, Codeforces, SWE-bench,
   GPQA, OpenScience, LongBench + controls (our mixed SFT data, one
   non-English shard). Pack 8-32k batches, record per-layer top-k
   expert counts (via `output_router_logits`; hooks only as
   fallback, see probe design). Rank by **GPU-level
   imbalance under our static 2-way expert partition** (hot-GPU share
   vs 50%, max over layers) — per-expert entropy alone can be
   neutralized by a lucky static split. Fit sorted counts to Zipf for
   the "real skew ~ z=…" anchor. Prior: math >= code > science MCQ >
   LongBench (mixed = balanced control).
   **Granularity: per sample, never dataset-aggregate.** The Balancer
   acts per grouped launch = one micro-batch x one layer; at batch 1
   that is one packed sequence x layer. Compute skew per (sample,
   layer) and report the distribution over samples (median + worst
   hot-GPU share). Summing counts across a dataset cancels rotating
   skew — the exact case where the dynamic cut wins — and is the
   reason domain-pure packing matters: the launch only ever sees one
   sequence.
2. **Trace replay into the existing microbench** (small patch: the
   harness synthesizes Zipf counts internally today — add a
   counts-from-file input). Feed the winners' real per-layer routed
   counts into the harness behind
   ep_owned_fair_q35122b_gemm.json in place of synthetic Zipf counts
   -> measured EP / sDP / Dynamic-EP walls on real routing (UltraEP's
   record-and-replay protocol). Adds a "real routing" group beside the
   z-sweep in Figs 5/12.
1.5 **Long-length prediction from doc signatures (the 1M question).**
   A 1M pack holds ~700 short docs, so its launch histogram
   converges to the DOMAIN-MEAN distribution — per-doc quirks
   average out; only the mean survives. Therefore, from the 16k
   screen's per-doc records, compute per cell:
   (a) **domain-mean hot-GPU share** (mean histogram over docs) —
   the analytic prediction of a natural 1M pack's skew, no 1M
   forwards needed. Strong mean -> natural long packs work.
   (b) **cluster structure + per-cluster token mass** — what a
   CONSTRUCTED 1M pack could reach, and whether any cluster holds
   >=1M tokens to build one. Near-balanced mean -> curated
   signature-packing is the ONLY route at 1M (go straight to the
   sequel, skip waiting for a "natural" winner).
   Then verify only the chosen route with a handful of real
   target-length forwards (full-attn models: a 1M prefill is
   minutes with FA, no grads; the hybrids are cheap).
   **Long-document lever:** dilution scales with the number of docs
   summed into one pack — 1M of 2k math problems = 700 summands;
   1M of repo-level code / books = ~10. Long-homogeneous-doc domains
   (repos, books, legal/biomed) are structurally better 1M
   candidates than short-problem sets, independent of per-doc skew
   strength.
3. **One e2e pair** for the winner only: a few SFT steps, balancer
   on vs off, 2-rank, measured tok/s.

Screen short, measure long: for stages 2-3 pack the winning domain to
campaign lengths via same-domain concatenation (standard LM packing;
our addition). Keep per-layer resolution — skew varies by layer; the
max-layer number is the honest headline.

## Search grid (independent model x dataset cells)

Models: M1 Qwen3-30B-A3B, M2 GLM-4.7-Flash, M3 Qwen3.5-122B-A10B
(EP-microbench model), M3.1 GLM-4.5-Air, M3.2 Hunyuan-A13B (both in
our eval suite / Fig 9 panels), M4 gpt-oss-20b (optional anchor).
Datasets: D1 DAPO-Math-17K, D2 Megatron-Math, D3 Codeforces,
D4 SWE-bench, D5 GPQA, D6 OpenScience, D7 LongBench (mixed control),
D8 our SFT mix (balanced control), D9 non-English shard (optional).

- **P0 (must, 9 cells):** {M1,M2,M3} x {D1, D3, D8} — one math, one
  code, one control per claim-model; picks each model's winner and
  its real-z anchor.
- **P1 (widen, 10):** {M1,M2} x {D2, D4}; {M3.1, M3.2} x
  {D1, D3, D8} (secondary eval MoEs, same math/code/control triplet
  as P0).
- **Deferred tail:** M4 x D2 (replicates LLEP's proven gpt-oss x
  math pair -> validates the probe) — gpt-oss is NOT integrated in
  our stack; runs last, only if still wanted.
- **P2 (breadth, 8):** {M1,M2} x {D5, D6, D7, D9}; D7 expected
  balanced, D9 only if P0 skew is weak.

Cells are fully independent (one probe invocation per pair) and
parallelize across GPUs/sessions. Sample budget: >=100 packed
samples per cell, all models (sharded-model cells just take longer —
still minutes). P0 alone decides the trace-replay input and the e2e
on/off candidate.

**Download location: `/scratch_local/user_data/kevinni` on EACH
machine** (each agent runs on a different machine; scratch is
per-machine local). Point the HF cache there before any pull —
`export HF_HOME=/scratch_local/user_data/kevinni/hf` (covers hub
checkpoints + datasets) — never into /home.

**Download budget.** The screen consumes only ~1.6M tokens
(~10-15 MB of text) per dataset — adapters should STREAM big
corpora (HF streaming, seeded shuffle buffer, doc-ids recorded),
never mirror them. Published sizes: DAPO-Math-17K tens of MB, GPQA
<5 MB (gated), LongBench ~100-200 MB, SWE-bench ~100 MB packing
issue+patch text only (never clone repo snapshots), Codeforces
hundreds of MB-GB, OpenScience a few GB, Megatron-Math unknown
(open item). Checkpoints dominate: Agent A ~310-340 GB
(61 + 234 + gpt-oss), Agent B ~400+ GB (GLM-4.7-Flash + 212 + 160).

**4-GPU execution plan.** Download + tokenize all datasets once up
front (shared HF cache; CPU/network work, overlaps GPU runs). Then
parallelize BY MODEL (loading dominates setup — pin one model per
GPU and sweep all its datasets; never reload per cell).
**Iteration order rule: exhaust the resident model first.** Dataset
swaps are free (pre-tokenized shards); model swaps cost a full
checkpoint load. One WAVE per model: replicate the model on all 4
GPUs, split the datasets (or sample shards) across replicas, finish
the model's whole cell list (P0 then P1 then P2), then move all GPUs
to the next checkpoint. Priority ordering applies WITHIN the
resident model's queue only — never interleave models by priority
(e.g., Agent B completes ALL GLM-4.7-Flash cells before Hunyuan
loads). bf16 fit: M1 ~61 GB, M2, M3.2 (~160 GB), M4
= single-GPU -> 4 replicas; M3 (~234 GB) and M3.1 (~212 GB) don't
fit one GPU -> those waves run 2 sharded replicas x 2 GPUs each,
datasets split between the two replicas. **Sharding may span up to
all 4 GPUs** (we have 4/machine): if a 2-GPU shard runs tight
(weights + activations + router logits headroom), fall back to a
3- or 4-way `device_map` shard — that wave then runs 1 replica
instead of 2. Sharding is placement only; router results are
identical at any shard width.

**Three-agent split (three machines): split BY MODEL.**
Checkpoints are the big downloads (60-234 GB); datasets are tiny —
model-split means each machine pulls only its own checkpoints, cells
are disjoint (no cross-machine locking), and per-dataset seeds make
packs identical across machines, so JSONs merge trivially.
- Agent A: M1 Qwen3-30B-A3B (9 cells) — single-GPU, 4-wide
  replicas (~61 GB downloads).
- **M4 gpt-oss-20b: DEFERRED to the very end** — not integrated in
  our stack (loader/format support unverified); run its single
  calibration cell only after everything else lands, and only if
  probe validation is still wanted.
- Agent B: M2 GLM-4.7-Flash (9) + M3.2 Hunyuan-A13B (3) —
  single-GPU models, 4-wide replicas (~200+ GB).
- Agent C (sharded specialist): M3 Qwen3.5-122B (3) + M3.1
  GLM-4.5-Air (3) — runs BOTH simultaneously as two 2-GPU-sharded
  replicas (122B on GPUs 0-1, Air on GPUs 2-3; ~446 GB downloads);
  slow forwards, few cells — parallelism absorbs it.
Each agent runs the 4-GPU wave plan and orchestration rules above
independently.

**Orchestration: shared queue + backfill (agent-monitored).** Cells
have uneven durations, so never assign them statically:
- Per wave, all replicas pull datasets from one shared per-model
  queue (claim-file or manifest lock). A replica that finishes early
  immediately claims the next unclaimed dataset of the RESIDENT
  model — natural backfill within the wave.
- When the wave's queue empties while stragglers still run, idle
  GPUs start LOADING the next wave's checkpoint (load overlaps the
  stragglers' compute — cross-wave backfill). CPU-side tokenization
  of upcoming datasets also runs during GPU waves.
- The driving agent launches replicas as background jobs, polls
  their logs, reassigns on completion, and appends finished cells to
  a manifest JSON (cell -> output path, status) so any restart is
  idempotent: done cells are skipped, crashed cells re-queued.

**What runs vs what is recorded.** The gate at layer L consumes the
post-attention hidden state built by all layers below (including
their experts' outputs) — routers cannot run in isolation, and the
forward must traverse every decoder layer in order. We RUN the full
stack and RECORD only each gate's top-k output; forward-only (no
labels/loss/backward) is the entire saving. One real cut: skip the
LM head — HF CausalLM materializes [tokens x vocab] logits (~40 GB
bf16 at B=8, T=16k, V~150k); call the backbone `model.model(...)`
(router logits still returned) or pass `logits_to_keep=1`.

## Probe design (route_skew_probe.py, planned)

`scripts/ep_skew/route_skew_probe.py` — a **dataset scout**, not a
motivation microbenchmark: it finds which real datasets naturally
skew our checkpoint's routing, per sample. Vanilla transformers — no
vLLM/SGLang (engines fuse the MoE path and hide top-k indices; we do
prefill-only forwards where they add nothing but complexity). Its
outputs feed the Fig 5/12 real-routing group and the trace replay.

- **Model/loading:** bf16, `torch.inference_mode()`. Attention
  backend = whatever our training venv already uses per model (FA4
  Blackwell build for Qwen3.5 hybrids' full-attn layers — their
  linear-attn layers use their own custom kernels regardless; FA4 or
  SDPA fallback for the standard-attn MoEs). Backend affects speed
  only — routing counts are identical under any correct attention.
  No hooks needed: HF MoE models return per-layer router logits via
  `model(input_ids, output_router_logits=True)` (aux-loss machinery)
  — tuple over MoE layers, each `[B*T, E]`.
- **Sample budget: >=100 packed samples per cell**, randomly drawn:
  shuffle documents with a fixed PER-DATASET seed (recorded in the
  output spec), then pack in shuffled order — never take the dataset
  head (ordering bias can hide skewed content). Per-dataset seeding
  means every model is measured on the IDENTICAL packs, so
  cross-model differences attribute to routing, not sampling. One 16k pack = ~786k routing decisions (16k tokens x 48
  layers), so per-sample histograms are tight with tens; 100 exists
  for the tail — P95/worst hot-GPU share then rests on ~5 samples
  instead of 2-3. Cost ~13 B=8 batches -> minutes per cell even
  sharded. Replay uses the median/P95 samples only.
- **Data adapters:** one tiny `name -> iterator[str]` per candidate
  (DAPO-Math-17K, Megatron-Math, Codeforces, SWE-bench, GPQA,
  LongBench; controls: our SFT mix shard, one non-English shard).
- **Packing = the launch unit:** greedily concatenate same-dataset
  docs with EOS to target T_screen (default 16k; wide sweep may use
  8k). One packed sequence = one "sample", mirroring batch-1 training.
- **Batching:** stack B=8 packed sequences `[B, T]` (pad + attention
  mask; pads masked out of counts). Per layer: logits
  `.view(B, T, E)` -> `topk(k=num_experts_per_tok)` -> per-sample
  `bincount(E)` -> `counts[sample][layer][E]`.
- **Metrics per (sample, layer):** hot-GPU share under the static
  2-way expert partition (same expert->GPU assignment as the
  microbench "owned" mode; partition = CLI arg), top-expert share,
  Zipf z fit (grid z in [0, 2.5], LSE on sorted normalized counts).
  Per dataset report: median and P95 over samples of max-over-layers
  hot share, plus per-layer medians.
- **Output / trace recording (built for later reproduction):**
  `profiling_results/ep_skew/route_skew_<model>_<dataset>.json` with
  (a) the trace: per-sample per-layer expert counts (~25 KB/sample)
  — sufficient for replay, since grouped-GEMM walls depend only on
  per-expert row counts — PLUS per-DOCUMENT per-layer counts (sliced
  by each doc's token span inside the pack, same logits, no extra
  compute): the document-level expert signatures that let the
  fallback's cluster-and-repack run entirely from recorded JSONs,
  no GPU re-run; (b) gzip'd token-level top-k indices for
  the median/P95 replay candidates only (~6 MB/sample), enabling
  any finer-grained reconstruction without a GPU; (c) the repro
  spec: model id + revision, dtype, attention backend, dataset name
  + HF revision, per-dataset seed, packing params, and the ORDERED
  doc-id list of every pack (packs rebuildable byte-for-byte even
  if shuffle code changes). The manifest adds probe git hash,
  timestamp, shard config. Replay: take a recorded sample's
  counts[layer][expert], scale to bench m_total, feed the
  ep_owned_fair harness in place of Zipf counts.
- **Cost:** one B=8, T=16k forward = ~128k tokens = seconds; 100
  samples/dataset = ~13 batches = minutes; a full 8-dataset sweep
  well under an hour per single-GPU model. Router-logit residency
  ~1.6 GB/batch, freed per batch.

## Open items (verify at implementation time)

- **`output_router_logits` support per model.** Confirmed pattern for
  HF Qwen3-MoE; GLM-4.7-Flash / GLM-4.5-Air / Hunyuan-A13B /
  Qwen3.5 hybrids / gpt-oss may use remote code that does not expose
  it — fallback is a forward hook on each gate module (works
  everywhere; probe should feature-detect and record which path ran).
- **Pinned HF dataset ids + licenses.** DAPO-Math-17K, Codeforces,
  SWE-bench (adapter must choose which text fields to pack — issue
  text? patches?), OpenScience, LongBench, and especially
  Megatron-Math (Du et al. 2025 — locate the release) need exact ids
  + revisions; GPQA is gated (license click-through) — resolve
  access before the sweep. D9 language unpicked.
- **Model sizes to confirm before wave planning:** GLM-4.7-Flash
  param count (single-GPU assumed), Hunyuan-A13B ~160 GB, GLM-4.5-Air
  ~212 GB, Qwen3.5-122B ~234 GB. Layer/expert counts differ per model
  (the 48x128 arithmetic in this doc is M1-specific) — probe reads
  them from config.
- **2-way partition for non-microbench models.** "Same as microbench
  owned mode" only defined for M3; default for the rest = contiguous
  halves E/2|E/2 (CLI-overridable).
- **Replay loader patch** in the ep_owned_fair harness (accept
  counts-from-file; today it synthesizes Zipf internally).
- **e2e on/off switch.** Where the Balancer toggle lives in the
  training stack, and whether the R2 campaign ran balancer-on —
  Kevin to confirm.

## Sequel steps: if no naturally-skewed dataset is found (last resort)

Decision gate: after the P0/P1 sweep, if the best cell's MEDIAN
max-over-layers hot-GPU share is barely above balanced (say < ~55%
vs the 50% ideal), no dataset skews naturally at pack level ->
**mine individual skewed samples and curate a skew dataset**:

1. **Mine per-document — from the recorded traces, no re-run.** The
   probe records per-DOCUMENT per-layer counts (sliced by doc spans
   at recording time), so document-level skew and expert signatures
   come straight from the stage-1 JSONs. A doc-granularity re-probe
   is only needed for datasets never screened.
2. **Cluster by expert signature, then pack.** Packing skewed docs
   naively can CANCEL (two skewed docs heating different experts sum
   to balanced). Cluster documents by similarity of their per-layer
   expert-count vectors (e.g., cosine on the flattened count
   profile / top-expert overlap), and build each curated pack from
   ONE cluster so the pack concentrates on the same experts.
3. **Verify curated packs** with a quick re-probe (packs are the
   launch unit — the pack-level histogram is the number that
   matters), then feed the winners to trace replay and the e2e pair
   exactly as in stages 2-3.
   **Pool sizing:** the winning cluster must hold enough tokens for
   stages 2-3 (~2M curated tokens for a few 320k steps x 2 arms;
   100 packs/cell ~ 800-3000 docs). If the cluster is thin, expand
   the stage-1 screen (more samples — linear cost) before curating.
4. **Presentation: disclose curation.** Frame as a real-data
   stress/worst-case workload — "the most skew-inducing samples
   found across public reasoning corpora, clustered by routing
   signature" — real samples, real routing, selection disclosed.
   Never present the curated set as a representative dataset.

## Log (append-only)

- [2026-08-11] **Stage-1 screen + stage-2 replay COMPLETE** (runners: c11
  this entry = Agent C then A backfill; a peer runner on another node
  covered Agent B incl. gated GPQA — shared NFS manifest merged cleanly).
  **Probe built**: `scripts/ep_skew/route_skew_probe.py` (vanilla
  transformers 5.3, bf16, backbone-only forward, capture = forward-pre-hook
  on each MoE layer's `experts` module -> the model's ACTUAL selected
  indices post sigmoid/bias/group logic; 16k packs, eos-sep, per-dataset
  seeds; per-sample + per-doc counts; JSON + flock'd manifest; OOM ->
  batch-halving; `--max-memory` because device_map=auto reserves no
  activation headroom). Summaries: `scripts/ep_skew/summarize_skew.py`;
  replay converter: `scripts/ep_skew/trace_to_hist.py`. Runner:
  `agent/anchors_tmp/skew_in39.sh` (NOTE: enroot renumbers
  NVIDIA_VISIBLE_DEVICES to 0..n-1 inside — CVD must use renumbered ids).
  Dataset pins: dapo=BytedTsinghua-SIA/DAPO-Math-17k;
  codeforces=open-r1/codeforces; sft_mix=smoltalk:longalign (the LF SFT
  source); megamath=IFM/MegaMath megamath-web-pro/* ("Megatron-Math" (Du
  et al.'25) is NOT on HF — substitution recorded in every spec);
  swebench=princeton-nlp/SWE-bench train; openscience=nvidia/OpenScience
  OS-Q3-235B-4; longbench=THUDM/LongBench data.zip; gpqa=Idavidrein/gpqa
  (gated — ran only on the token-bearing peer node).
  **Screen results** (104 x 16k packs/cell; median | P95 of max-over-layers
  hot-GPU share under contiguous E/2; z = median Zipf fit at max layer):
  ```text
  model          dataset       medHot  p95Hot  domHot     z  topE%
  glm4.7-flash   dapo          0.7011  0.7102  0.7019  0.95  19.6   <- strongest cell
  glm4.7-flash   codeforces    0.6709  0.6858  0.6727  0.95  21.6
  glm4.7-flash   openscience   0.6599  0.6892  0.6630  0.60  14.5
  glm4.7-flash   swebench      0.6554  0.7592  0.6590  0.65  15.1
  glm4.7-flash   gpqa          0.6509  0.6933  0.6692  0.75  16.7   (S=3 only — GPQA too small for 16k packs; indicative)
  glm4.7-flash   longbench     0.6394  0.8650  0.5795  0.60  13.5
  glm4.7-flash   megamath      0.6374  0.8435  0.5945  0.70  15.0
  glm4.7-flash   sft_mix       0.6354  0.7511  0.5743  0.65  13.0
  qwen3.5-122b   codeforces    0.6578  0.6737  0.6676  0.70  11.4   <- EP-microbench model winner
  qwen3.5-122b   dapo          0.6343  0.6444  0.6326  0.75   8.5
  qwen3.5-122b   sft_mix       0.6143  0.6639  0.5730  0.65   6.5
  qwen3-30b      longbench     0.6547  0.7241  0.5971  0.70   9.6
  qwen3-30b      swebench      0.6508  0.6968  0.6277  0.75  11.0
  qwen3-30b      openscience   0.6412  0.6732  0.6374  0.70   9.5
  qwen3-30b      sft_mix       0.6342  0.6937  0.5597  0.65   8.4
  qwen3-30b      dapo          0.6290  0.6377  0.6364  0.65  10.3
  qwen3-30b      codeforces    0.6128  0.6345  0.6066  0.75  11.8
  qwen3-30b      megamath      0.5999  0.7416  0.5789  0.55   8.8
  glm4.5-air     codeforces    0.6107  0.6217  0.6160  0.73  10.6
  glm4.5-air     dapo          0.5855  0.5918  0.5868  0.50   8.4
  glm4.5-air     sft_mix       0.5761  0.6072  0.5221  0.50   6.5
  hunyuan-a13b   sft_mix       0.5465  0.5733  0.5399  0.30   3.5   <- negative case (balances)
  hunyuan-a13b   dapo          0.5404  0.5497  0.5416  0.30   3.8
  hunyuan-a13b   codeforces    0.5361  0.5457  0.5274  0.35   3.7
  ```
  Verdicts: (1) EVERY Balancer-claim model clears the >=55% natural-skew
  gate on real public data — curated-skew fallback NOT needed. (2) domHot
  (mean histogram over docs = §1.5a analytic 1M-pack prediction) stays
  ~= per-pack median for math/code (e.g. flash dapo 0.702, 122b
  codeforces 0.668) -> natural long packs KEEP the skew; controls collapse
  toward balance (sft_mix domHot 0.52-0.57) as expected. (3) Long-doc
  purity lever measured: longbench/longalign at ~1.7 docs/pack run hot on
  every model regardless of domain — pack purity is a co-driver of skew
  alongside domain (disclose when presenting). (4) hunyuan-a13b is the
  honest counter-row: per-batch aux-loss recipe balances (z~=0.3,
  topE ~3.6% ~= 2.3/64 uniform-ish) — mirrors the Mixtral prior. (5) Real-z
  anchors for Figs 5/12: flash-class ~z0.95, Qwen-class ~z0.7, air ~z0.7,
  vs the paper's synthetic z in {0.5,1.0,1.5,2.0}.
  **Stage-2 replay MEASURED** (UltraEP record-and-replay: real per-layer
  counts -> `ep_balance_bench.py` via `--hist`, q35-122b geom 256,1024,3072
  topk8 shared1024, m=5.12M, scope=gemm, natural alpha, repo .venv build
  — container-installed asym_gemm lacks the new `transpose_b` kernel arg):
  ```text
  sample(worst layer)        owned   sdp   plan  queue  ownedImb
  codeforces median L26       39.7  41.7   31.5   31.4     0.410
  codeforces P95    L26       41.0  41.9   32.8   31.7     0.443
  dapo       median L27       38.5  40.4   31.2   32.1     0.397
  dapo       P95    L27       37.2  40.9   31.3   32.8     0.394
  (median layers: owned 31.5-33.3, plan 29.1-31.7 -> +4..8%)
  ```
  -> at REAL code/math routing the static-EP worst-layer wall is
  **+19-26% over our planned cut** (39.7 vs 31.5 ms etc.), landing between
  synthetic z=1.5 (37.9) and z=2.0 (45.1) walls even though the marginal
  histogram fits z~=0.7: synthetic Zipf sprays hot experts uniformly over
  the halves, real hot experts CLUSTER on one half — the synthetic axis
  UNDERSELLS real skew at matched z. This is the measured "real routing"
  group for Figs 5/12.
  **Artifacts** (gitignored, on shared FS):
  `profiling_results/ep_skew/route_skew_<model>_<dataset>.json` (+docs.gz,
  +topk gz for median/P95), `manifest.json`,
  `profiling_results/ep_skew/replay/ep_real_replay_q35122b_*.json`.
  **Open**: qwen3-30b|gpqa needs a token-bearing env (manifest has the
  failed claim; peer can fill); GPQA cells generally too small for >=100
  16k packs (only ~3) — treat as indicative or drop; e2e on/off pair
  (stage 3) still pending Kevin's R2 balancer-on answer — natural
  candidate: 122b x codeforces packs @ ~320k, balancer on/off; consider a
  short-doc control config of smoltalk to decouple the purity lever from
  domain in the control row.

## Recommendation

C (keep the z-sweep as the sensitivity axis) + B (run the search
grid: P0 first — it alone decides winners and real-z anchors per
model) + D (replay the winners' traces for measured EP-vs-Dynamic-EP
points), with the length-scaling check before stages 2-3 and the
curated-skew fallback only if the decision gate trips. Mention A
(gpt-oss) only if a reviewer demands a literature-anchored model.
Separately, the ablation still wants one measured e2e on/off pair
(2-rank Qwen3.5-122B @ ~320k) to replace the estimated anchor in
Fig 12's right panel — confirm whether the R2 campaign ran
balancer-on.

## §RESULTS LOG (append-only)

- [2026-08-11, c12] SCREEN COMPLETE — stage-1 grid done for the full three-agent
  split (order run: B -> C -> A; C/A partially executed by the parallel session,
  merged via the flock'd manifest). Probe: scripts/ep_skew/route_skew_probe.py
  (capture = pre-hook on `experts` modules -> ACTUAL selected indices; 16384-token
  packs, 104 samples x B=8, contiguous E/2 partition; chat kinds via each model's
  chat template). Outputs: profiling_results/ep_skew/ (per-cell JSON + per-doc
  sidecars + median/P95 token-level topk gz + manifest.json); summarizer:
  scripts/ep_skew/summarize.py. Dataset substitutions: megamath = IFM/MegaMath
  megamath-web-pro (LLEP's "Megatron-Math" has no public release); sft_mix =
  smoltalk/longalign (public balanced control); gpqa capacity = 3 full 16k packs
  (n=3, thin-tail caveat). Hunyuan tokenizer/model native in venv transformers.
- KEY NUMBERS (median max-over-layers hot-GPU share / P95; Zipf z at worst layer;
  "1M pred" = hot share of the domain-mean histogram, §1.5):
  - GLM-4.7-Flash x DAPO-Math: 0.7011/0.7102, z=0.95, 1M pred 0.7019  <- flagship
    (codeforces 0.6709 z=0.95; controls ~0.635-0.639)
  - Qwen3.5-122B x Codeforces: 0.6578/0.6737, z=0.70, 1M pred 0.6676
    (dapo 0.6343; sft_mix 0.6143) <- the EP-microbench model's real-z anchor
  - Qwen3-30B: swebench 0.6508 (robust winner), longbench 0.6547, control 0.6342 —
    domain deltas small; skew mostly INHERENT (LLEP-style) but mean-persistent
  - GLM-4.5-Air x Codeforces: 0.6107/0.6217, z=0.73, 1M pred 0.6160
  - Hunyuan-A13B: ~0.54 everywhere, z=0.30 — BALANCED (negative; aux-loss recipe,
    mirrors Mixtral's known weak topic skew)
- READINGS: (1) decision gate (<~55% -> curated fallback) PASSES for all
  Balancer-claim models — natural real-data skew exists, no curation needed;
  (2) real skew z in [0.5, 0.95] — the paper's synthetic z sweep brackets it
  (z=1.0 point ~= flash math; 1.5/2.0 = stress tail); (3) controls at 0.54-0.64
  = inherent post-training routing imbalance on OUR checkpoints, per-pack
  granularity; (4) §1.5 verdict: domain-pure skew survives doc-averaging (1M pred
  ~= per-pack median for math/code cells; controls collapse toward 0.5) ->
  natural long packs are viable for stages 2-3.
- NEXT (decision-gated, per Recommendation): trace replay of the winners into the
  ep_owned_fair harness (counts-from-file loader; 122B inputs already extracted
  to profiling_results/ep_skew/replay/ep_hist_real_*.json) -> measured EP/sDP/
  Dynamic-EP walls at real routing for Figs 5/12; then ONE e2e balancer on/off
  pair (2r Qwen3.5-122B ~320k, codeforces packs) — pending Kevin's R2
  balancer-on confirmation (open item).
