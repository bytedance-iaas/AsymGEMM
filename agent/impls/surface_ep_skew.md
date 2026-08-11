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
