Check the current throughput results for qwen3 32b, qwen3 30b and llama3.3 we all ahve some asym config from scheduler (profiling to select some more approriate config). teh thropugp resilts and artifacts are all stored here and their metris docs too. HWOEVER for the plots for qwen3.5 the rests are very weird. 
scripts/figures/out/tp_complete_q3_5-122b-a10b.png -  whye a smalelr sq elgnths ooms asym* but a longer seq passed ...???
scripts/figures/out/tp_complete_q3_5-35b-a3b.png also why is thiws soo many palceholders ..??? 

Ok so the goal is that siiarl to the current 3 models, we stil need asym* to beat the otehr basleine son qwne3.5 35b or 122b (lets srart with eh qw.35 35b) its mmeory foorprit shuld beeven smaller than qwn3 30b due to ienar attnetion sooo we can tet much longer squnecs forn the start.
Stil the goal is that asym shold flal abck to baslines when the seq legth is not that long and for ultra long sqiunce we wil do t3 (but if the ris need o be adjuet opr new modes creted to fit qwen3.5 seroes better thats fine too. bus tjsut ercored eveurnig jsut incase). This doc wil serve as the potnof records.
Keep thsi goal in moidn dont stop unitl this goal has beebn met. We need tosee the asym can adapt to diffet tiers (maube thc urrent tiers work maube not i am ont sure) adn mach baseline trhrougput on not soo long sequences but beat them at capacity again
Also all teh conten needs otbe writ after this prompt. keep this prompt here for record

---

# qwen3.5 throughput campaign — POINT OF RECORD (started 2026-07-23, node s04-p1-dgx-02-c18)

Tree: `main_kevin` @ d22440d (merged scheduler tree; tier presets = `backend|T1/T2/T2B/T3`
from scripts/lf/tier_recipes.sh). Prior q3.5 evidence: c12 record
(`agent/impls/s04-p1-dgx-02-c12/concise_throughput_results.md` §q3.5-35b, tputX campaign
2026-07-20) + c14 record (`agent/impls/s04-p1-dgx-02-c14/test_throughpout_v2.md` §q3.5-122b).
Plot data bank: `scripts/figures/plot_tp_vs_seq.py` DATA dict.

## §0 ANSWERS to the two plot questions (from the records — both are real, neither is a plot bug)

**Q1 — 122b: why does asym OOM at 320k but pass at 384k/448k?** Measured, reproduced
host-side anomaly, NOT a plot error: asym T1 @320k b1 was host-watchdog-killed 4 times
(avail 46–49 GiB vs the 50 floor; n512 ×3 from dirty+clean starts, n256 ×1) while 384k
AND 448k fit healthily at RSS 848–849 GB (c14 record, addendum 2026-07-20 ~22:00). The
288–320k band systematically exceeds the host floor on a ~957 GB node; mechanism
unresolved — pinned-pool shape quantization suspected, same host-pool family as the
missing linear-attn keep-acts port (remaining_optimizations #7b). The cell is recorded
as measured HOST-DNF (red OOM in the plot); the deep-end story rests on the measured
384k/448k points. Non-monotone fit-vs-seq is therefore REAL on this model/node class.

**Q2 — 35b: why so many placeholders?** The tputX campaign (c12, 2026-07-20) was cut
short: only 7 cells were ever MEASURED — rc 848@256k · rc 1002@384k (96% edge) + wall
(384k,448k] · uns 1067@512k · uns 1023@576k (98% edge) + wall (576k,640k] · asym T1
609@128k · asym T2 1377@576k · asym T3 1142@640k. Everything else in the plot ("est"
black-border bars; hatch = tier, NOT placeholder) was banked as estimates: uns/rc 128k
were "in flight" and never landed; uns 256–448k est via rc-parity; asym 256–512k est by
interpolation. The plot honestly marks these, but the density of est cells is exactly
the campaign debt. WORSE: the one measured short-seq asym cell (T1 @128k = 609 @24.5%
HBM) sits ~27% BELOW the est baseline band (~830–860) — the fall-back-to-baseline
property fails on qwen3.5 as measured, unlike q3-30b/q3-32b/llama where T1 tracked
baselines within ~1%.

## §1 GOAL (restating the prompt as gates) + gap list

Per the standing model template (parity where baselines live / beat at capacity / sole
coverage beyond), on q3.5-35b-a3b first:
- **G1 PARITY**: asym within ~2% of the best baseline at not-so-long seq (128k, 256k),
  measured on THIS node. Today: measured 609 vs est ~840 → FAILING, mechanism unknown.
- **G2 REAL CELLS**: replace every "est" cell in the 35b plot row with a measurement
  (uns/rc @128k, uns @256k/384k/448k, asym @256k/384k/448k/512k) — all same-node c18.
- **G3 WALLS re-verified on c18**: rc (384k,448k], uns (576k,640k] were c12 verdicts;
  re-bracket cheaply here (one fit + one OOM each side).
- **G4 CAPACITY**: extend asym past 640k (T3 was at 34.8% HBM, RSS 554 GB → huge
  headroom; linear attention ⇒ leaner act slope than q3-30b which reached 1.6M).
  Don't stop until asym's 35b wall is bracketed.
- **G5 TIER ADAPTATION**: current T1 (unsloth-ohbm0+staged) fails G1 → find/build the
  tier that restores short-seq parity (candidates below), record any new mode/recipe.
  Scheduler tiers may need a qwen3.5-specific recipe row — that is sanctioned.

## §2 architecture facts (config.json, Qwen/Qwen3.5-35B-A3B)

40 layers = **30 linear_attention (gated delta-net) + 10 full_attention** (interval 4);
hidden 2048; full attn: 16 heads / 2 KV / head_dim 256; linear attn: 16 key heads ×128,
32 value heads ×128, conv kernel 4; MoE 256 experts top-8, moe_intermediate 512 +
shared expert 512; vocab 248320; native max_position 262144 (throughput probes beyond
256k are still valid capacity/latency measurements — same treatment as the banked c12
campaign). Weights ~70 GB bf16. fa4 venv + FLASH_ATTN=fa4 auto-switch in the driver;
QWEN35_DELTA_CHUNK_SIZE chunks delta-net to dodge the fla ≥75k illegal-memory fault.
KEY CODE FACT: `linear_attention_activation_offload.py` has NO keep-acts flag —
`ASYMM_ATTN_ACT_KEEP_ACTS_HBM` exists only for full attention (10/40 layers), so under
T2 the 30 delta-net layers still round-trip their saved tensors over C2C every step
(remaining_optimizations #7b names this the top q3.5 lever: on 122b it costs −7.5% TP
@32k AND most of the host-RSS bloat; on 35b it is the prime suspect for 609-vs-840).

## §3 parity-gap hypotheses (to kill/confirm by profile, in order)

- **H0 the est baseline is wrong, not asym** (checked FIRST, costs one run): the 128k
  "~840" band was extrapolated from rc's measured 848@256k, but at b1 on a 256-expert
  MoE the per-token cost RISES as seq shrinks (per-step fixed costs — optimizer,
  streaming, launches — amortize over fewer tokens; measured per-token: rc 1179 us/tok
  @256k → 998 @384k, still falling at the edge). Extrapolating flat/upward to 128k was
  optimistic; measured uns@128k may land ~600–750, i.e. T1 609 could already be parity.
- **H1 linear-attn saved-tensor round-trip** (no keep-acts port, 30/40 layers). Fix =
  port keep-acts gate to linear_attention_activation_offload.py (#7b, small-medium).
  NOTE T1 runs unsloth-GC recompute where the offload hooks are guarded out of the
  recompute window (_in_backward_graph_task) — so if T1 is the slow one, H1 alone
  cannot be the whole story; T2's gap (if any) is the H1 readout.
- **H2 delta-net recompute cost under unsloth-GC**: T1 recomputes the WHOLE layer incl.
  the delta-net scan; if the fla backward-through-recompute is disproportionately
  expensive at 128k, T1 (recompute tier) is structurally wrong for qwen3.5 short-seq —
  the adaptation is "T2 (keep-acts, no recompute) IS the short-seq tier" (HBM is at
  24.5% — plenty to spend).
- **H3 chunked delta path overhead** (QWEN35_DELTA_CHUNK_SIZE default vs 0/unchunked at
  seqs < the fault length... chunking exists to dodge a ≥75k fla fault, so 128k NEEDS
  chunks; measure chunk-size sensitivity only if H1/H2 don't close the gap).
- **H4 MoE fg path at small token counts** (256 tiny experts, b1 128k): staged dispatch
  fixed costs amortize with tokens; compare per-token fg cost vs q3-30b's banked curve.

## §4 protocol (c18, one run at a time — ABSOLUTE)

- w1+m2 (PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1), MAX_SAMPLES=1024 (<900k) / 512
  (≥900k); steady = mean of the 2 measured steps; b1 ladder for comparability with the
  banked cells; batch probes only at the parity end (both sides get best batch).
- Same-node discipline: every A/B verdict is c18-internal (c12 numbers guide
  expectations only — cross-node drift is not controlled).
- GPU-EMPTY GUARD before every launch (`nvidia-smi --query-compute-apps` empty, else
  abort loudly); host guard: MemAvailable sane + no stale /dev/shm asym_fabric arenas
  (single-GPU runs don't use the arena, but the hygiene rule stands). Kills: `kill -9
  <exact GPU PID>` only, never pkill; verify GPU empty after.
- Datasets: fast builder (build_lf_sft_eval_pair.py) n1024/n512; if an entry is missing
  but the file exists → rerun with DATASET_OVERWRITE=true (user rule 2026-07-22).
- Runs tagged `q35p_*` (probe) / `q35l_*` (ladder), archived under
  profiling_tp_s04-p1-dgx-02-c18/; every result lands in §6 of THIS doc as it arrives.

## §5 run queue (serial; updated as verdicts land)

| # | run | why |
|---|---|---|
| Q1 | uns @128k b1 | baseline anchor for G1 (est 840 → measure) |
| Q2 | asym T1 @128k b1 | reproduce the 609 on c18 (same-node pair with Q1) |
| Q3 | asym T2 @128k b1 | H2 readout: keep-acts tier at short seq (HBM is empty) |
| Q4 | rc @128k b1 | completes the 128k group |
| Q5 | profile diff Q1-vs-Q2/Q3 artifacts | pick H1–H4; decide the #7b port |
| Q6+ | (a) fix per §3 → re-measure 128k; (b) mid cells 256k/384k/448k/512k (uns+asym); (c) wall re-brackets; (d) T3 depth ladder 704k→ | fill G1–G4 |

## §6 PROGRESS LOG (newest at bottom; every run and verdict, no exceptions)

- [2026-07-23 03:0x] Campaign opened on c18 (4 GPUs idle, ~957 GB usable host RAM —
  `free` shows 1693 GB but that is fabric-inflated). Tree d22440d clean. Prior-record
  audit done (→ §0 answers). Qwen3.5-35B-A3B weights NOT in this node's HF cache —
  download started in background (~70 GB, scratch has 13 TB free). Datasets present
  only to s60000 for 35b → 128k+ builds queued behind the smoke.
- [09:2x] Weights DONE (67 GB, snapshot 59d61f3c). .venv-fa4 verified on c18 (torch
  2.12+cu130, 4 GPUs, fa4-cute OK, fla 0.5.0). c12 detail table recovered
  (test_throughput_results.md §W6/W7+X): T1@128k = 609 · 45.2 GiB (24.5%) · RSS 224;
  T2@576k = 1377 · 95.7 (51.7%) · RSS 524; T3@640k = 1142 · 64.4 (34.8%) · RSS 554;
  the never-landed v4 queue (T3 704k+64k ladder, uns/rc 128k, …) is exactly the
  placeholder debt. Also inherited hazard: c12 ran HF_HUB_OFFLINE=1 after a 401
  post-process flake — if c18 hits 401s, rerun with HF_HUB_OFFLINE=1. Added H0 (§3).
- [09:29] **Q1 LAUNCHED** (guarded; GPU empty, no stale arenas): uns @128k b1, tag
  q35p1uns, w1+m2, n1024 dataset auto-build in-run.
- [09:34] Q1 HARDFAIL in ~4 min: `numactl: execution of .venv-fa4/bin/torchrun: No
  such file or directory` → the RELOCATED-VENV TRAP, fa4 flavor: `.venv-fa4` was
  copied from the SFT-38 worktree. Fixed BOTH layers on c18: (a) 72 stale shebangs in
  .venv-fa4/bin/* + pyvenv.cfg rewritten SFT-38→SFT-46; (b) **editable installs
  (asym_gemm, deepspeed, liger_kernel, llamafactory, ktransformers .pth/finder files)
  still pointed at SFT-38 — and /workspace/AsymGEMM-SFT-38 EXISTS, so before the fix
  every fa4 run would have SILENTLY imported the WRONG TREE'S CODE** (no crash — the
  worst failure mode). All imports verified → SFT-46 paths; main .venv audited clean.
- [09:4x] **Q1 RELAUNCHED** (guarded) after the venv repair.
- [10:5x] **Q1 DONE: uns @128k b1 = 242.5 s/it · 528 tok/s · 50.6 GiB (27%) · RSS 211
  GB** (spread 0.1%, clean FIT). **H0 CONFIRMED**: the plot's est band (~830–860) at
  128k was extrapolated far too optimistically — measured uns per-token cost is 1895
  us/tok @128k vs 937 @512k (fixed per-step costs dominate at b1 short-seq on this
  arch). The c12-banked asym T1 609 would be **+15% ABOVE** this baseline, not −27%
  below — the "parity failure" was an artifact of the placeholder. Same-node verdict
  pending Q2. Chain Q2 (asym T1) → Q3 (asym T2) → Q4 (rc) @128k b1 LAUNCHED.
- [11:2x] **Q2 DONE: asym T1 @128k b1 = 208.1 s/it · 615 tok/s · 45.1 GiB (24%) · RSS
  224 GB** (spread 2.9%). Reproduces c12's 609 within +1% (cross-node consistency).
  **SAME-NODE VERDICT: asym T1 BEATS uns 528 by +16.5% at 128k** — G1 is not merely
  parity, it's a beat; the placeholder est had inverted the story's sign.
- [11:2x] Q3 (asym T2 @128k b1) chain-ABORTED as HARDFAIL — but the trainer actually
  COMPLETED (3/3 steps, losses 0.75/0.74/0.66, full artifacts): the KNOWN q3.5 T2/T3
  dirty-teardown false-fail (c12 W6/W7+X), which also prints "Training command
  failed", and tp_probe's hardfail grep ran BEFORE its own artifacts-complete rescue.
  Parsed from artifacts: **asym T2 @128k b1 = 214.5 s/it · 597 tok/s · 36.8 GiB (20%)
  · RSS 268 GB** (spread 2.1%). T2 within −3% of T1 at short seq (keep-acts does NOT
  collapse here; runtime counters confirm attn-KA 10 + linear-saved-offload 30 + fg 40
  wrapped, staged dispatch). Tier picture at 128k: T1 615 > T2 597 > uns 528 > (rc
  pending). FIX SHIPPED to scripts/lf/tp_probe.sh: artifacts-complete rescue moved
  BEFORE the hardfail greps, and all this-run evidence (jobs.tsv/step_samples) is now
  freshness-bounded (mtime >= probe start) so stale dirs can't vouch for a new run.
  Also noted: hardfail's `grep -iE error` tail can surface DATASET SAMPLE text (code
  conversations) — cosmetic, diagnose from train.log, not the probe tail.
- [11:3x] Chain LAUNCHED: Q4 rc @128k b1 → Q5 uns @128k b4(→b2) → Q6 asym T1 @128k
  b4(→b2). Rationale for batch probes: at b1 both sides sit ≤27% HBM; the parity-end
  claim is only honest at each side's best batch (other models' banked cells are
  best-over-batch; 35b was b1-throughout).
- [12:5x] Q4–Q6 DONE, all clean:
  **Q4 rc @128k b1 = 241.5 s/it · 530 tok/s · 65.8 GiB (36%) · RSS 189** — rc ≈ uns
  (528/530: the convergence law holds on q3.5 too).
  **Q5 uns @128k b4 = 279.7 s/it · 1831 tok/s · 176.5 GiB (95% edge) · RSS 275** (b4
  = uns's max batch, first-fit descending).
  **Q6 asym T1 @128k b4 = 265.6 s/it · 1928 tok/s · 147.8 GiB (80%) · RSS 291** —
  **+5.3% over uns's best batch at 15pp lower HBM.**
  ⇒ 128k GROUP COMPLETE, G1 SETTLED AS A BEAT: b1 asym +16.5%, best-batch asym +5.3%,
  and asym never sits at the allocator edge (80% vs 95%). rc-b2 cell = later polish.
- [12:5x] **CHAIN-3 LAUNCHED** (mid cells + walls, b1, adjacent same-day pairs):
  uns/T1 @256k → uns/T1 @384k → uns/T1 @448k → uns/T1 @512k → uns/T2 @576k →
  uns(expect OOM)/T2/T3 @640k. Tier plan: T1 until GOOM (never probed past 128k —
  linear attn may carry it far), shed to T2 at the edge; T3 explicitly at 640k for
  the sole-coverage anchor + T2@640k to see if the plot's 640k cell upgrades. ~7 h.
- [17:0x] **CHAIN-3 DONE — 11 fits + 2 uns OOMs. The c18 mid-band table (b1, w1+m2,
  cell = lat · TP · HBM(%) · RSS):**
  | seq | uns | asym T1 | asym T2 | asym T3 |
  |---|---|---|---|---|
  | 256k | 309.1 · 828 · 93.8 (51%) · 232 | 239.3 · **1070** · 79.3 (43%) · 246 | | |
  | 384k | 394.8 · 973 · 138.5 (75%) · 275 | 283.3 · **1355** · 112.5 (61%) · 290 | | |
  | 448k | 451.8 · 992 · 156.0 (84%) · 275 | 312.7 · **1433** · 130.9 (71%) · 291 | | |
  | 512k | 491.4 · 1042 · 181.2 (**98% edge**, spread 7%) · 275 | 348.5 · **1469** · 147.7 (80%) · 291 | | |
  | 576k | **GPU-OOM** | — | 387.6 · **1486** · 118.5 (64%) · 466 | |
  | 640k | **OOM** | — | 419.0 · **1528** · 132.6 (72%) · 466 | 572.0 · 1119 · 64.2 (35%) · 554 |
  READINGS: (1) asym beat GROWS with seq: +29/+39/+44/+41% over uns at 256–512k —
  far bigger than the c12-era placeholders implied. (2) **uns wall on c18 =
  (512k, 576k]** — one rung LEFT of c12's (576k,640k]; c12's 576k fit was a 97.7%
  edge cell, exactly the fragile-edge class; uns@512k on c18 already sits at 98%
  with 7% step spread (edge tax visible). (3) T2@640k = 1528 with TP still RISING
  in seq (linear-attn flatness + amortized fixed costs) — the 640k plot cell
  upgrades from T3 1142(c12) to T2 1528(c18); T3@640k re-anchored 1119 (c12 1142,
  −2%, consistent). (4) T1 never hit its wall through 512k (80%); its slope ≈0.267
  GiB/1k → hand-off to T2 expected (576k,640k]. (5) c18/merged-tree asym runs are
  systematically FASTER than the c12-era record (T2@576k 1486 vs 1377) — same-node
  re-measurement of every plot cell is vindicated.
- [17:0x] **CHAIN-4 (depth ladder) LAUNCHED**: T1@576k (shed point) → T2@768k →
  T2@896k → T2B@1024k(n512) → T2B@1152k(n512) → T3@1280k(n512) → T3@1408k(n512).
  T2B added to the 35b tier map (staged/ker000, no keep-acts — should outrun T3
  wherever it fits, q3-30b R2A precedent). Host guard added to the chain (abort if
  MemAvailable <1200 GB fabric-scale at launch). Predictions (probe-not-predict
  rule: these only set expectations): T2 slope ~0.18 GiB/1k → T2 wall ~(832k,896k];
  T3@640k at 35% HBM → HBM never binds T3 in this ladder; host RSS 554@640k with
  ~907 GB budget → host wall probably past 1.4M. ~6 h.
- [17:3x] ENV AUDIT (user question): .venv-fa4 DOES use FA4 (flash_attn.cute loads,
  FLASH_ATTN=fa4 in runs, attnfa4 labels) for the 10 full-attn layers; **causal_conv1d
  is NOT installed** — every run (today's and the c12 campaign's) hits fla's "Falling
  back to torch implementation" for the delta-net conv, affecting asym AND baselines
  identically (model-side code). Deliberately NOT installing mid-campaign (same-env
  pair law; chain-4 in flight). causal-conv1d lever: DROPPED by user (2026-07-23,
  "dont need to que that") — expected benefit is small (conv ≪1–2% of step FLOPs);
  recorded here only so the fallback warning in train.logs is understood.
- [21:0x] CHAIN-4 L1–L5 ALL FIT (cell = lat · TP · HBM(%) · RSS): **T1@576k =
  373.2 · 1543 · 161.6 (87%) · 377** (T1 outruns T2's 1486 at 576k — T1 hand-off is
  ≥640k, later than predicted); T2@768k = 512.9 · 1498 · 152.5 (82%) · 468; T2@896k
  = 600.4 · 1492 · 175.6 (**95% edge**) · 469; T2B@1024k = 803.5 · 1274 · 137.9
  (75%) · 557; **T2B@1152k = 894.4 · 1288 · 150.6 (81%) · RSS 880 GB** — host is
  nearly full (~907 effective); 1288 tok/s at 1152k is still ABOVE uns's best cell
  anywhere (1042). L6 T3@1280k in flight — host verdict imminent.
- [21:2x] USER RE-SCOPE (2026-07-23): **uns-OFF (superoffload_mem|unsloth-off-ohbm0)
  joins the q3.5-35b matrix** — its capacity wall is the REAL bar for asym's max-seq
  (it is the deepest baseline; on q3-30b it out-lived uns by ~400k). This supersedes,
  for qwen3.5, the earlier "no unsloth-off outside q3-30b" ruling (same precedent as
  the 2026-07-19 q3-30b re-scope). PRIORITY: uns-OFF capacity walker runs BEFORE all
  queued polish items; queued items 1–5 targets may RESET based on its wall (asym
  sole-coverage must start beyond uns-OFF's wall, and P3-band beat is vs uns-OFF
  tok/s where it is the last-alive baseline). Chain-5 walker prepared: uns-OFF @576k
  → 896k → 1152k → 1280k → 1408k → 1536k, stop at first OOM (bracket recorded);
  launches immediately after chain-4's last rung (hands-off — no interruption).
- [23:45] **CHAIN-4 COMPLETE — ALL 7 RUNGS FIT.** Deep rungs: **T3@1.28M = 1213.7 ·
  1055 tok/s · 118.7 (64%) · RSS 883** · **T3@1.41M = 1387.7 · 1015 tok/s · 129.4
  (70%) · RSS 884** (both rescued by the new artifacts-complete-first tp_probe logic —
  teardown-only failures, exactly the known q3.5 T2/T3 quirk). RSS PLATEAU ~880-884
  across 1.15M→1.41M (act-pool reuse; same sublinear host behavior the 30b stretch
  showed). **asym max measured fit so far = 1.41M tokens = 2.75× the uns wall — a
  FLOOR, not a wall: HBM 70%, host flat → ladder top reached, EXTEND flagged.** No
  asym wall found yet; extension target resets after uns-OFF's wall lands (user
  priority). 576k skipped in the walker per user ("576 for sure fits") — walker =
  896k → 1.15M → 1.28M → 1.41M → 1.54M. **CHAIN-5 (uns-OFF walker) LAUNCHED.**
- [00:48] **CHAIN-5 DONE: uns-OFF @896k = 799.5 · 1121 tok/s · 142.0 (77%) · RSS 784
  — FIT; @1152k = HOST C-OOM (watchdog 34 < 35 GiB floor) → uns-OFF wall ∈ (896k,
  1152k], HOST-bound** (act-offload pools + streamed weights; same wall class as
  q3-30b's uns-OFF ~1.05M). Beat check at 896k where uns-OFF is the last-alive
  baseline: **asym T2 1492 vs uns-OFF 1121 = +33%** (and T2B@1.15M 1288 > uns-OFF's
  best anywhere). TARGET RESET (user's motivation for this detour): asym's measured
  1.41M floor already clears uns-OFF's wall by ≥1.22×; sole-coverage span starts just
  past the uns-OFF wall. Refinement probe uns-OFF @1024k LAUNCHED (n512) to tighten
  the bracket to 128k grade; asym extension rung 1.54M queued after it.
- [02:0x] **uns-OFF @1024k = 924.9 · 1107 tok/s · 162.8 (88%) · RSS 801 — FIT** ⇒
  **final uns-OFF wall = (1024k, 1152k], HOST-bound.** Beat at its deepest fit:
  asym T2B@1024k 1274 vs uns-OFF 1107 = **+15%** (75% vs 88% HBM). CHAIN-6 LAUNCHED:
  asym T3 @1.536M (→1.664M if fit; either way brackets or extends the asym floor) +
  rc polish cells (256k/384k/448k-wall/128k-b2) — the last measurement block before
  plot rebuild + consolidation.
- [06:00] **CHAIN-6 DONE.** (a) **asym T3 @1.536M = 1565.6 · 981 · 140.7 (76%) · RSS
  886 FIT**; **@1.664M = 1738.9 · 957 · 158.3 (86%) · RSS 887 FIT** ⇒ **asym floor =
  1.664M tokens** (no wall found; HBM headroom to ~1.79M; host FLAT 884-887 across
  1.15M→1.66M — the pool-reuse plateau). Capacity: **3.25× uns (512k), ≥1.44×
  uns-OFF (≤1152k)** — exceeds the "smaller footprint than q3-30b" hope (30b rank-1
  topped at 1.6M; 35b ≥1.66M with a bigger model). (b) rc cells: **rc@256k = 844
  (68%)** (c12 848 — consistent); **rc@384k = OOM on c18** (c12's 96.4%-edge fit did
  not reproduce — same fragile-edge class as uns@576k) ⇒ **rc wall c18 = (256k,
  384k]**; rc@448k OOM confirms; **rc@128k b2 = 1024 (68%)** = rc best-batch (b4
  structurally OOM). (c) CHAIN-7 LAUNCHED: shallow uns-OFF cells @128k/384k/576k b1 —
  MEASURED instead of estimated so the rebuilt plot carries zero placeholder values
  in the 35b row (only proper beyond-wall OOM* markers).
- [07:0x] CHAIN-7 DONE — uns-OFF shallow cells: **@128k = 234.2 · 547 · 30.1 (16%) ·
  RSS 262** (above uns's 528 — offload tax hidden at short seq, HBM tiny);
  **@384k = 369.1 · 1040 · 75.0 (41%) · RSS 476** (ABOVE uns 973: act offload
  relieves allocator pressure — uns-OFF is the STRONGEST baseline at mid seq on this
  model, vindicating the user's re-scope); **@576k = 514.5 · 1120 · 108.9 (59%) ·
  RSS 741**. Δ vs best-alive baseline updates: +12% @128k · +30% @384k · +38% @576k
  (asym still leads every column). USER SCOPE DECISION (2026-07-23): after uns-OFF
  @512k + @256k, 35b measurement STOPS (minimal-but-full matrix) — 448k/640k plot
  columns are dropped from the figure grid (cells preserved here); final grid =
  128k·256k·384k·512k·576k·896k·1.02M·1.15M·1.41M·1.66M, 100% measured. Then plot
  rebuild + consolidation, then q3.5-122b (weights download starts during rebuild).
  CHAIN-8 (uns-OFF @512k → @256k) LAUNCHED.
- [08:04] CHAIN-8 DONE: **uns-OFF @512k = 457.9 · 1118 · 97.5 (53%) · RSS 493** ·
  **@256k = 292.2 · 876 · 46.3 (25%) · RSS 339** — uns-OFF is the top baseline at
  EVERY shared mid column. **35B MEASUREMENT CAMPAIGN CLOSED** (user scope decision).
  122b weights download started. Plot rebuild next.

## §7 FINAL 35B TABLE (c18, 1 GPU, b1, w1+m2; cell = lat s/it · TP tok/s · HBM GiB (%) · RSS GB)

| seq | so-recomp | so-unsloth | so-unsloth-OFF | asym (tier) | Δ vs best-alive |
|---|---|---|---|---|---|
| 128k | 241.5 · 530 · 65.8 (36%) · 189 | 242.5 · 528 · 50.6 (27%) · 211 | 234.2 · 547 · 30.1 (16%) · 262 | **208.1 · 615 · 45.1 (24%) · 224 (T1)** | **+12%** |
| 128k best-batch | b2: 250.1 · 1024 · 125.9 (68%) | b4: 279.7 · 1831 · 176.5 (95%) | — | **b4: 265.6 · 1928 · 147.8 (80%) (T1)** | **+5%** |
| 256k | 303.2 · 844 · 125.8 (68%) · 189 | 309.1 · 828 · 93.8 (51%) · 232 | 292.2 · 876 · 46.3 (25%) · 339 | **239.3 · 1070 · 79.3 (43%) · 246 (T1)** | **+22%** |
| 384k | **G-OOM → wall (256k,384k]** | 394.8 · 973 · 138.5 (75%) · 275 | 369.1 · 1040 · 75.0 (41%) · 476 | **283.3 · 1355 · 112.5 (61%) · 290 (T1)** | **+30%** |
| 448k | OOM (measured) | 451.8 · 992 · 156.0 (84%) · 275 | — | **312.7 · 1433 · 130.9 (71%) · 291 (T1)** | +44% vs uns |
| 512k | OOM* | 491.4 · 1042 · 181.2 (98% edge) · 275 | 457.9 · 1118 · 97.5 (53%) · 493 | **348.5 · 1469 · 147.7 (80%) · 291 (T1)** | **+31%** |
| 576k | OOM* | **G-OOM → wall (512k,576k]** | 514.5 · 1120 · 108.9 (59%) · 741 | **373.2 · 1543 · 161.6 (87%) · 377 (T1)** | **+38%** |
| 640k | OOM* | OOM (measured) | — | **419.0 · 1528 · 132.6 (72%) · 466 (T2)** · T3: 572.0 · 1119 · 64.2 (35%) · 554 | — |
| 768k | OOM* | OOM* | — | 512.9 · 1498 · 152.5 (82%) · 468 (T2) | — |
| 896k | OOM* | OOM* | 799.5 · 1121 · 142.0 (77%) · 784 | **600.4 · 1492 · 175.6 (95%) · 469 (T2)** | **+33%** |
| 1.02M | OOM* | OOM* | 924.9 · 1107 · 162.8 (88%) · 801 | **803.5 · 1274 · 137.9 (75%) · 557 (T2B)** | **+15%** |
| 1.15M | OOM* | OOM* | **HOST C-OOM → wall (1.02M,1.15M]** | **894.4 · 1288 · 150.6 (81%) · 880 (T2B)** | **sole** |
| 1.28M | OOM* | OOM* | OOM* | 1213.7 · 1055 · 118.7 (64%) · 883 (T3) | sole |
| 1.41M | OOM* | OOM* | OOM* | 1387.7 · 1015 · 129.4 (70%) · 884 (T3) | sole |
| 1.54M | OOM* | OOM* | OOM* | 1565.6 · 981 · 140.7 (76%) · 886 (T3) | sole |
| **1.66M** | OOM* | OOM* | OOM* | **1738.9 · 957 · 158.3 (86%) · 887 (T3) — FLOOR, no asym wall found** | **sole** |

VERDICTS (all measured, zero placeholders): **G1 parity→beat** ✅ (+12% b1 / +5%
best-batch at 128k; asym never at the allocator edge). **G2 real cells** ✅ (every
plot cell same-node c18). **G3 walls** ✅ rc (256k,384k] · uns (512k,576k] · uns-OFF
(1.02M,1.15M] host — both c12 "fits" at 96-98% edges did NOT reproduce (fragile-edge
class). **G4 capacity** ✅ asym ≥1.66M = 3.25× uns, 1.44× uns-OFF; host flat 884-887
(pool plateau), HBM 86% at top. **G5 tier adaptation** ✅ existing tiers suffice —
T1 to 576k, T2 to 896k, T2B to 1.15M, T3 beyond; no new mode needed; T2B earns a
plot hatch. Linear-attn keep-acts port (#7b) NOT needed for 35b (recorded as a 122b
lever). tok/s RISES with seq for asym T1 (615→1543): linear-attn flatness +
amortized fixed costs — the signature qwen3.5 result.

FIGURES REGENERATED (2026-07-24): tp_vs_seq_q3_5-35b-a3b + tp_complete_* rebuilt
from the c18 cells — 10-column grid (448k/640k dropped per user minimal-scope), NO
est borders anywhere, T2B hatch "\\" added to the tier encoding, walls red-marked at
their measured columns. plot_tp_vs_seq.py DATA comment carries the off-grid cells.

PAPER INTEGRATION (2026-07-24, user-approved lean six): lean grid = 128k · 384k ·
576k · 1.02M · 1.15M · 1.66M (turning points: closest race → rc wall → uns wall +
asym peak → uns-OFF last fit → uns-OFF wall → capacity crown). Combined 2×2 panel:
4th slot = Qwen3.5-35B-A3B (replaces the llama duplicate placeholder). Overleaf
updated (figures/tp_combined.pdf + tp_vs_seq_q3_5-35b-a3b.pdf; main_results.tex
caption "batch size 8"→"best fitting configuration per point", prose → four models
+ the q3.5 rising-TP/1.66M story) and PUSHED to the Overleaf remote as 0aac871
(rebased over an intervening web edit af8ef4a, no conflicts).

================================================================================
## PHASE 122B (2026-07-24): q3.5-122b-a10b on c18
================================================================================
Weights downloaded (234 GB, snapshot dc4d348). Inherited c14 state (2026-07-20):
32k×8: uns 909 (83%) vs asym T1 841 (61%) = −7.5% (GAP; #7b lever) · rc GPU-OOM ·
uns-OFF HOST-OOM. b1: uns 665@288k/640@320k both 97-98% edge, wall (320k,352k];
asym T1 874@384k (85%) / 897@448k (98% = T1 HBM edge); asym@320k = 4× host-DNF
ANOMALY (288-320k band trips floor 50; 384k/448k fit at RSS 848); 480k T1 G-OOM
pred + T2 HOST-OOM (tier inversion: keep-acts OFF adds host on 234-GB-weight
models). PLAN: A) re-anchor on c18 + anomaly reproduction (chain-9 LAUNCHED:
uns/T1 @32k×8 → uns@320k → T1@384k → T1@320k ×2 (anomaly) → uns@352k (wall) →
T1@448k; ~5 h). B) build #7b port (linear-attn keep-acts flag, mirror of
ASYMM_ATTN_ACT_KEEP_ACTS_HBM) + loss-parity smoke — targets BOTH the −7.5% @32k×8
AND the host-pool family (320k anomaly, T2@480k host-OOM). C) A/B the port at
32k×8 and probe T2+linKA @480k+ for capacity extension. D) consolidate + plot.
- [08:5x 07-24] **USER HALT + PREEMPT**: chain-9 stopped mid-rung-a1 (TaskStop; GPU
  verified empty, no orphans, arenas clean; a1 had banked nothing — it will simply
  re-run when 122b resumes). Priority now = **35b 128k/384k SATURATION SWEEP**
  (user: the two lean-plot short columns must carry max-batch ≤0.92·HBM cells — the
  b1 ladder under-utilizes there, which is also why TP rises toward 576k).
  CHAIN-10 LAUNCHED (first-fit descending): 128k: uns-OFF b8→4→2 · asym T1 b6→5 ·
  asym T2 b8→6→4 · uns b3 (≤92% candidate vs its edge-taxed b4 1831 @95%) · rc b3
  (expect OOM ⇒ b2 confirmed max) · 384k: uns-OFF b3→2 · asym T1 b2 (expect OOM ⇒
  b1 confirmed max) · asym T2 b2 (leaner acts — real chance to beat T1 b1 1355) ·
  uns b2 (expect OOM ⇒ 973 confirmed max). After the sweep: adopt best measured
  ≤92% cell per point (edge cells recorded alongside), regen figures, re-push
  Overleaf, THEN resume 122b phase A (chain-9 from a1).
- [12:03] **CHAIN-10 DONE — saturation sweep results** (lat · TP · HBM(%) · RSS):
  128k: uns-OFF **b8 = 506.9 · 2020 · 162.6 (88%) · 797** (b9 pred 98% ⇒ b8 is its
  ≤92% max) · asym T1 b6 G-OOM, **b5 = 283.9 · 2254 · 176.4 (95% — over-line) ·
  378** ⇒ T1's ≤92% cell stays b4 1928 (80%) · asym **T2 b8 = 412.2 · 2484 · 180.0
  (97% — over-line) · 500** (fastest 128k cell measured; first-fit descending never
  probed its ≤92% batch — b7 pred ~86%, ~2300 via lat=F+kb fit) · uns **b3 = 265.7
  · 1445 · 133.3 (72%) · 275** = uns's ≤92% cell (b4 1831 = 95% over-line, recorded
  as edge) · rc b3 G-OOM ⇒ rc b2 1024 (68%) confirmed max.
  384k: uns-OFF b3 G-OOM, **b2 = 528.7 · 1453 · 123.1 (67%) · 770** · asym T1 b2
  G-OOM ⇒ b1 1355 confirmed T1-max · asym **T2 b2 = 426.4 · 1801 · 169.9 (92% —
  exactly at the line, PASSES) · 516** ⇒ **new 384k asym cell: +24% over uns-OFF
  1453, +33% over own T1 b1** · uns b2 G-OOM ⇒ b1 973 confirmed max.
  CONVENTION SET (user 2026-07-24): plot cells = LARGEST BATCH WITH RESERVED ≤
  0.92·185 GiB per system per point (edge cells >92% preserved in this doc, not in
  the figure). DECISIVE RUN LAUNCHED: asym T2 @128k b7→b6 (pred b7 ~86% / ~2300 —
  would top uns-OFF 2020 within-rule; fallback = T1 b4 1928 → 128k column would
  read asym −4.6% vs uns-OFF).
- [12:2x] **RULE RELAXED (user): cells = MAX MEASURED THROUGHPUT over batch, near-
  full HBM allowed** (supersedes the strict ≤92% convention above; the b7 run in
  flight became a bonus curve point). ADOPTED CELLS — 128k: rc 1024 (b2) · uns 1831
  (b4, 95%) · uns-OFF 2020 (b8, 88%) · **asym T2 b8 2484 (97%) = +23% over best
  baseline**; 384k: uns 973 · uns-OFF 1453 (b2) · **asym T2 b2 1801 (92%) = +24%**.
  Figures regenerated (lean + complete + combined); Overleaf push 6e805e2. Note
  scope: 256k/512k (complete-variant-only columns) remain b1-ladder cells — their
  saturation was not requested; recorded as a known limitation of the complete
  figure, NOT the paper figure.
- [13:1x] Bonus point landed: asym T2 @128k **b7 = 366.2 · 2447 · 178.5 (96%) ·
  500** — confirms b8 2484 as the max-TP cell (b7 −1.5%); T2's batch curve at 128k
  is flat-topped 2447→2484 at 96-97%. **35b SATURATION QUESTION FULLY CLOSED.**
  **122b PHASE A RESUMED** (chain-9 relaunched from rung a1, all 8 rungs).
- [01:5x 07-25 UTC] TIMELINE CORRECTION + a1/a2 BANKED: artifact mtimes show the
  ORIGINAL chain-9 completed BOTH a1 (08:07→08:29Z) and a2 (→08:51Z) before the
  08:5x halt (the "mid-a1" note above was a stale read; GPU was empty at kill =
  between-runs, chain-10 uncontaminated). The resumed chain banked both via the
  driver's OVERWRITE=false skip-on-existing (36-s "runs", fresh jobs.tsv, real
  same-day artifacts). **RESULTS @32k×b8 (c18 pair): uns = 299.2 · 856 · 158.9
  (86%) · 660 · asym T1 = 296.2 · 864 · 136.2 (74%) · 695 ⇒ asym +0.9% ≈ PARITY —
  the c14 −7.5% gap is GONE on the merged tree** (c14: 909 vs 841; both sides
  moved, paired same-node verdict is what counts). #7b port demoted from
  "parity-blocker" to "host-relief lever" — decision after the 320k anomaly
  verdict. a3 (uns @320k b1) now running live.
- [04:00 07-25] **PHASE A COMPLETE (a3–a8).** c18 b1 results: a3 uns @320k =
  **GPU-OOM** (c14's 640 @98% edge does NOT reproduce — third fragile-edge kill;
  uns c18 wall ≤320k, 288k bracket probing) · a4 asym T1 @384k = **442.9 · 867 ·
  180.6 (98% T1 edge) · RSS 798** (TP matches c14 874 within −0.8%) · **a5/a6 asym
  T1 @320k = 823 (88%) / 826 (89%), RSS 696 — BOTH FIT: the c14 4×-host-DNF
  320k ANOMALY DOES NOT REPRODUCE on c18** (cell upgrades to measured; the
  288–320k host-floor trap was c14-node-specific and/or pre-merge — mechanism
  moot for the record here) · a7 uns @352k = OOM ✓ · a8 asym T1 @448k = **GPU-OOM**
  (c14's 897 @98% edge also does not reproduce) ⇒ **asym T1 c18 wall ∈ (384k,
  448k]**. Emerging c18 verdicts: 32k×8 parity (+0.9%); asym sole coverage
  already AT 320k (uns dead, asym 823-826 healthy); capacity ≥384k vs uns ≤320k.
  CHAIN-11 LAUNCHED: uns @288k (wall bracket) → asym T1 @416k (tighten T1 wall)
  → asym T2 @448k (tier-inversion re-check — c14's T2-adds-host verdict deserves
  a c18 test now that the host anomaly is gone).
- [05:27] **CHAIN-11 DONE — TIER INVERSION OVERTURNED ON c18.** b1: uns @288k =
  **444.2 · 648 · 178.1 (96%) · 659 FIT** ⇒ **uns c18 wall = (288k, 320k]** (one
  rung left of c14's (320k,352k], fragile-edge pattern again) · b2: asym T1 @416k
  = GPU-OOM ⇒ **T1 wall = (384k, 416k]** · b3: **asym T2 @448k = 520.2 · 861 ·
  171.3 (93%) · RSS 846 FIT** — c14's "keep-acts OFF adds host → T2 infeasible →
  T1 is the deepest tier" verdict does NOT hold on c18: T2 extends PAST T1's wall
  with host at 846 GB (comfortably under floor) and TP flat vs T1@384k (861 vs
  867). #7b port verdict: **NOT NEEDED on c18** — parity holds at 32k×8 without
  it, and at the deep end it would RAISE HBM (keeps 30 linear layers' saves
  resident) exactly where T2 sits at 93%; stays in remaining_optimizations as a
  c14-class-host-node lever only. CHAIN-12 (T2 walker 480k → 512k → 560k, stop at
  first fail) LAUNCHED — c14's T2@480k HOST-OOM gets its c18 re-test as rung 1.
- [07:47] **CHAIN-12: ALL THREE FIT — no T2 wall found.** T2 @480k = 563.6 · 852 ·
  177.8 (96%) · 849 (c14's HOST-OOM point: REFUTED) · @512k = 596.4 · 858 · 178.8
  (97%) · 849 · @560k = 679.3 · 824 · 181.1 (98%) · 894. T2 TP ~flat 852-861
  across 448-512k (linear-attn flatness again); both pools near-full at 560k
  (98% HBM, RSS 894 vs ~907 effective). CHAIN-13 LAUNCHED: asym T1 @288k (the
  head-to-head at uns's last-fit — c14 never measured it, anomaly is gone here)
  → T2 @608k (wall bracket; expect G/C-OOM) → T3 @640k (leaner-HBM tier's reach
  past T2). Consolidation + plot row after.
- [08:0x] STANDING PRINCIPLE (user, 2026-07-25 — applies to ALL remaining work):
  **CAPACITY FIRST, THROUGHPUT SECOND.** Per model/config: (1) pin the capacity
  map (walls, sole-coverage span, deepest fit) with b1 ladder runs; (2) only then
  spend GPU time on throughput-squeeze runs (batch sweeps, tier head-to-heads,
  saturated cells) at the columns that made the map. The 35b sequence (b1 ladder →
  chain-10 saturation sweep) is the template; 122b follows it now (capacity chains
  9–13 first, squeeze sweep after).
- [09:31] **CHAIN-13 DONE.** d1: **asym T1 @288k = 363.3 · 793 · 148.2 (80%) · 695
  FIT — head-to-head at uns's last fit: 793 vs 648 = +22% at 15pp lower HBM.**
  d2: **T2 @608k = 768.7 · 791 · 181.0 (98%) · RSS 896 FIT** — capacity ≥608k =
  2.11× uns's 288k. d3: T3 @640k = **HOST C-OOM** (watchdog ×2, no GPU OOM) —
  T3's act-offload ADDS host on this 234-GB-weights model; **T2 (keep-acts) is
  the deepest 122b tier on c18** — the c14 inversion claim was right about
  direction (offload→host cost) but attached to the wrong tier boundary (T2 vs
  T1 there; T3 vs T2 here). CHAIN-14: T2 @640k probe launched (expect GPU-OOM at
  ~188 GiB pred ⇒ would pin asym wall = (608k, 640k]).
- [10:1x] CHAIN-14: **T2 @640k = 836.1 · 765 · 181.2 (98%) · RSS 896 — FIT** (the
  predicted GPU-OOM did not materialize: T2 ceiling-hugs the allocator like
  q3-30b sEP-T2 did, HBM flat 181.0→181.2 across 608→640k, RSS flat 896).
  Capacity ≥640k = 2.22× uns. Wall STILL not found → CHAIN-15 walker: T2 @672k →
  704k → 736k, stop at first fail.
- [11:50] **CHAIN-15: WALL FOUND — asym 122b wall = (672k, 704k], HOST-bound**
  (T2 @672k = 888.0 · 757 · 181.2 (98%) · 895 FIT; @704k = HOST C-OOM watchdog×2).
  CHAIN-16 launched: uns-OFF @32k×8 (c14 declared it host-dead there — c18
  re-check) + uns-OFF @288k b1 (b1 host load is lighter; does the deepest baseline
  live ANYWHERE on 122b?) — the last capacity checkboxes before consolidation.

## §8 FINAL 122B TABLE (c18, 1 GPU, w1+m2; cell = lat s/it · TP tok/s · HBM GiB (%) · RSS GB)

| seq | so-recomp | so-unsloth | so-unsloth-OFF | asym (tier) | verdict |
|---|---|---|---|---|---|
| 32k×b8 | G-OOM (c14, ~181 needed) | 299.2 · 856 · 158.9 (86%) · 660 | c18 re-check in flight (c14: HOST-OOM) | **296.2 · 864 · 136.2 (74%) · 695 (T1)** | **parity +0.9%** |
| 288k b1 | OOM* | 444.2 · 648 · 178.1 (96%) · 659 — LAST FIT | re-check in flight | **363.3 · 793 · 148.2 (80%) · 695 (T1)** | **+22% at uns's edge** |
| 320k b1 | OOM* | **G-OOM → wall (288k, 320k]** | — | **389.1/387.5 · 823/826 · 88-89% · 696 (T1)** — c14 anomaly ABSENT | **sole coverage begins** |
| 352k b1 | OOM* | OOM ✓ | — | — | |
| 384k b1 | OOM* | OOM | — | 442.9 · 867 · 180.6 (98% edge) · 798 (T1) | T1's edge |
| 416k b1 | OOM* | OOM | — | T1 G-OOM → **T1 wall (384k, 416k]** | tier shed → T2 |
| 448k b1 | OOM* | OOM | — | 520.2 · 861 · 171.3 (93%) · 846 (T2) | c14 tier-inversion refuted |
| 480k b1 | OOM* | OOM | — | 563.6 · 852 · 177.8 (96%) · 849 (T2) | c14 T2-host-OOM refuted |
| 512k b1 | OOM* | OOM | — | 596.4 · 858 · 178.8 (97%) · 849 (T2) | |
| 560k b1 | OOM* | OOM | — | 679.3 · 824 · 181.1 (98%) · 894 (T2) | ceiling-hug regime |
| 608k b1 | OOM* | OOM | — | 768.7 · 791 · 181.0 (98%) · 896 (T2) | |
| 640k b1 | OOM* | OOM | — | 836.1 · 765 · 181.2 (98%) · 896 (T2); T3 = HOST C-OOM | T3 not viable deep (offload adds host) |
| **672k b1** | OOM* | OOM | — | **888.0 · 757 · 181.2 (98%) · 895 (T2) — LAST FIT** | **capacity crown 2.33× uns** |
| 704k b1 | OOM* | OOM | — | **T2 HOST C-OOM → asym wall (672k, 704k]** | host-bound wall |

VERDICTS (c18, all measured): parity ✅ (+0.9% @32k×8) · beat ✅ (+22% at uns's last
fit) · sole coverage 320k→672k ✅ · capacity 2.33× ✅ · tier adaptation ✅ (T1→384k,
T2 384k→672k; T3 counterproductive at depth on this host-heavy model — recorded).
c14 non-reproductions on c18 (all fragile 96-98% edge cells or node-specific host
effects): uns@320k fit, asym T1@448k fit, the 320k 4×-host-DNF anomaly, T2@480k
host-OOM, tier-inversion-at-T2. The c14 record remains valid FOR c14's node-day;
the c18 table is the campaign record going forward.
- [12:5x] CHAIN-16 VERDICTS: **uns-OFF @32k×8 = HOST C-OOM (c14 reproduces) AND
  @288k b1 = HOST C-OOM ⇒ uns-OFF is host-dead at every probed 122b point on the
  ~957 GB node class** — §8 row updated from "in flight" to measured. (Ops note:
  g1's dying trainer lingered on GPU ~seconds; guard correctly blocked g2 → g2
  relaunched clean. Guard works as designed.)
- [12:5x] **SQUEEZE/COMPLETION CHAIN-17 LAUNCHED** (capacity map is closed; per
  the capacity-first principle these are the throughput/completeness runs):
  (1) rc @128k b1 — rc's all-OOM row is a b8-cluster verdict; b1 short-seq never
  probed on 122b (rc roots ≈ 25 GiB @128k → may fit) (2) rc @288k b1 (3) uns
  @128k b2→b1 (4) asym T1 @128k b3→b2→b1 (5) asym T1 @32k b12→b10 (batch push;
  74% @b8) (6) uns @32k b10→b9 (batch push; 86% @b8). Adds a saturated 128k
  column + max-batch 32k cells. ~4 h.
- [14:21] **CHAIN-17 DONE — squeeze results** (lat · TP · HBM(%) · RSS):
  **rc @128k b1 = 310.5 · 412 · 116.9 (63%) · 576 FIT** — rc LIVES on 122b at b1
  (the all-OOM row was a 32k×b8-cluster verdict, not a model property); rc @288k
  b1 = G-OOM ⇒ **rc wall (128k, 288k]** (coarse; fine-grain low-value).
  **uns @128k b2 = 339.2 · 755 · 171.1 (92%) · 659** (first fit at b2).
  **asym T1 @128k b3 = 368.4 · 1042 · 180.3 (97%) · 799 = +38% over uns.**
  **asym T1 @32k b12 = 332.5 · 1155 · 179.8 (97%) · 795** (batch push 864@b8 →
  1155@b12). **uns @32k b10 = G-OOM → b9 = 299.6 · 961 · 175.3 (95%) · 659** ⇒
  32k best-batch verdict: **asym 1155 vs uns 961 = +20%** (matched-b8 parity
  +0.9% stands as the protocol-matched cell). CHAIN-18: rc @32k b4→b2 (last
  unmeasured fit-able cell) — then figures + close-out.
- [14:5x] CHAIN-18: **rc @32k b4 = 269.6 · 475 · 116.4 (63%) · 576 FIT** (b4 =
  131k total tokens ≈ rc@128k b1's memory — physics consistent; b5/b6 unprobed,
  ~edge by token count). **ALL 122b CELLS MEASURED.**

## §9 122b CAMPAIGN CLOSED (2026-07-25) — figure + final saturated cells

Figures regenerated from §8 + squeeze cells: tp_complete_q3_5-122b-a10b (11
columns: 32k·128k·288k·320k·384k·448k·512k·560k·608k·640k·672k, ZERO est
borders) and tp_vs_seq_q3_5-122b-a10b (lean six: 32k·128k·288k·320k·512k·672k).
Saturated short-seq cells (max-TP over batch): 32k = rc 475 (b4) · uns 961 (b9)
· uns-OFF HOST-OOM · **asym T1 1155 (b12) = +20%**; 128k = rc 412 (b1) · uns 755
(b2) · **asym T1 1042 (b3) = +38%**. Matched-b8 parity cell kept in §8 (+0.9%).
FINAL VERDICT vs prompt goals, 122b: parity✅(+0.9% matched / +20% best-batch) ·
beat✅(+22% to +38% everywhere shared) · sole coverage 320k→672k✅ · capacity
2.33× uns✅ · tiers adapt✅ (T1→T2 shed at 384k; T3 recorded-not-viable deep).
The old c14-era plot row (with the 320k red-OOM oddity the prompt asked about)
is fully superseded — the anomaly does not exist on c18 and every cell in the
new row is a same-node measurement.

**CAMPAIGN COMPLETE FOR BOTH MODELS (35b §7, 122b §8/§9).** Paper: 35b panel is
in tp_combined (Overleaf d52fd19). The 122b figure is repo/env-side only — the
2×2 combined is full; adding 122b means a 5th panel or an appendix figure
(user decision pending).

================================================================================
## PHASE R2-Q35 (2026-07-25, user): RANK-2 (2-GPU) on q3.5-35b + q3.5-122b
================================================================================
GOAL (user): rank-2 turning-point tables/plots for both qwen3.5 models, same
template as c14's R2A — ending with **asym as the ONLY backend running long
sequences at rank 2**. Capacity-first (standing principle). Setup: GPUs 0+1,
b1/rank ga1, w1+m2; asym = sEP (asym_sepplan2_cpuadamwds — MoE canonical:
ASYM_EP2=1, 160 GB shm arena, halved expert weights/rank) with T2→T2B sheds;
baselines = DP2 (|2) rc / uns / uns-OFF. Global TP = ranks·seq·b/lat (parse
shows per-invocation seq·b/lat — DOUBLE it for the record). tp_probe.sh gained
GPUS env (GPUS=2 → RUNS "model|2", GPU_POOL=0,1). Stale-arena guard mandatory
(c14 fiasco). Expectations from c14 R2A physics: DP does NOT shard activations
→ baseline walls stay put or move LEFT (NCCL buffers); uns-OFF's host machinery
duplicates per rank → its wall shrinks vs 1-rank; sEP shares weights via arena
→ asym's host cost grows sub-2×; NCCL-unhandled-CUDA errors count as G-OOM.
CHAIN-19 (35b R2) LAUNCHED: sep2-T2 smoke @64k (abort-on-fail) → uns @512k/576k
→ rc @256k/384k → uns-OFF @768k→512k (host-shrink probe) → sEP-T2 @640k/896k →
sEP-T2B @1.15M. ~7-8 h. 122b R2 chain follows.
- [02:59 07-26] **CHAIN-19 (35b R2) DONE** (cell = per-rank lat · GLOBAL TP (2×) ·
  HBM/rank (%) · RSS/rank): smoke sep2 @64k FIT (sEP runs qwen3.5 ✓). Baselines:
  **rc @256k = 316.2 · 1620 · 125.8 (68%) · 108 FIT; @384k G-OOM → rc 2R wall
  (256k,384k]** (= 1-rank) · **uns @512k = 541.5 · 1892 · 178.3 (96%) · 192 FIT;
  @576k G-OOM → uns 2R wall (512k,576k]** (= 1-rank) · **uns-OFF @768k HOST-C-OOM,
  @512k = 551.7 · 1856 · 98.1 (53%) · 416 FIT → host wall SHRANK into (512k,768k]**
  (1-rank was (1.02M,1.15M] — per-rank machinery duplicates, c14-R2A physics ✓).
  asym sEP: **T2 @640k = 455.3 · 2812 · 153.0 (83%) · 440** (DP scaling 92% vs
  2×1-rank-1528) · **T2 @896k = 678.9 · 2640 · 179.5 (97%) · 442** · T2B @1.15M
  HOST-C-OOM → asym 2R wall ∈ (896k, 1.15M]. CHAIN-20 (completion): sEP-T2 @512k
  (h2h at uns's edge) → uns-OFF @640k (wall tighten) → sEP-T2B @1024k (asym wall
  tighten).
- [03:40] CHAIN-20 partial, then EXTERNALLY STOPPED (not by the agent; GPU
  verified empty, no orphans, no stale arenas — state clean). Banked before the
  stop: **sEP-T2 @512k = 376.0 · 2724 global · 131.5 (71%)/rank · RSS 330/rank —
  h2h at uns's 2R edge: asym 2724 vs uns 1892 = +44% at 29pp lower HBM** ·
  uns-OFF @640k = HOST-C-OOM ⇒ **uns-OFF 2R wall = (512k, 640k]**. NOT run:
  r2c3 sEP-T2B @1024k (asym 2R wall tighten; current bracket (896k, 1.15M]).
  RESUME LIST when cleared: (1) r2c3 → (2) 122b R2 chain (smoke → uns 288k/320k
  → rc 128k/288k → sEP-T2 448k/672k/up) → (3) tp2r figure rows for both models.
- [resume 07-26] CHAIN-21 LAUNCHED (probe wrapper = scratchpad tp_probe2.sh —
  rank-aware + artifacts-first verdicts; repo tp_probe.sh left at its rank-1
  canonical form): r2c3 sEP-T2B @1024k (35b R2 wall tighten) → 122b R2: sep2
  smoke @64k (abort-on-fail; arena-cap suspect if it dies → retry
  ASYM_ARENA_SHM_CAP_GB=280) → uns @288k/@320k → rc @128k/@288k → sEP-T2
  @448k/@672k/@800k(n512). ~6-7 h. Then tp2r figure rows for both models.
- [10:0x 07-26] CHAIN-21 verdicts: **r2c3 sEP-T2B @1024k (35b R2) = HOST C-OOM
  (watchdog ×2; ran after auto dataset-rebuild) ⇒ 35b R2 asym wall = (896k,
  1024k] — 35b RANK-2 CAPACITY MAP COMPLETE** (rc (256k,384k] · uns (512k,576k]
  · uns-OFF (512k,640k] · asym 896k = ≥1.56× best baseline, sole ≥640k).
  q2r1 sep2 smoke on 122b = HARDFAIL: `shared fabric cap exceeded (used=169.8 GB,
  cap=160 GiB) — raise ASYM_ARENA_SHM_CAP_GB` = the pre-suspected arena ceiling
  (122b expert bank > q3-30b's; the 160 default was tuned for 30b). CHAIN-22
  relaunched the 122b sequence with **ASYM_ARENA_SHM_CAP_GB=280** (node budget
  fine: 280 shared + 2×~250/rank ≪ 957). Auto dataset-rebuild retry kept.
- [12:4x 07-26] **CHAIN-22 DONE — 122b R2 baselines mapped, asym ladder HOST-dead
  at depth.** Results (per-rank lat · GLOBAL TP · HBM/rank · RSS/rank): sep2
  smoke @64k FIT (arena=400 loads; actual bank use ~310 GB; NB the fabric FILE is
  created at CAP size — 400 GiB in a 479-GiB /dev/shm; killed runs LEAK it, one
  381-GB-resident stale arena found+cleared after the chain) · **uns @288k =
  488.6 · 1178 · 178.8 (97%) · 412 FIT; @320k G-OOM → uns R2 wall (288k,320k]
  (= rank-1)** · **rc @128k = 340.8 · 752 · 117.4 (63%) · 412 FIT; @288k G-OOM →
  rc R2 wall (128k,288k] (= rank-1)** · asym sEP-T2: **@448k, @672k, @800k ALL
  HOST-C-OOM** — arena ~310 + 2× per-rank pools ≫ 957 at depth; rank-1's 846-GB
  single-process budget does not halve under sEP because the expert bank is only
  ~1/3 of 122b's host footprint (act/optimizer pools duplicate per rank).
  CHAIN-23 (descending walker) LAUNCHED: sEP-T2 @288k → @192k → @128k, stop at
  first FIT — brackets the asym R2 wall from below and gives the h2h cell.
  If even 128k fails: record the llama-R2A-class characterized limitation
  (rank-2 fixed host costs invert the rank-1 lead on host-heavy 122b).
- [13:21] CHAIN-23: sEP-T2 @288k/@192k/@128k ALL HOST-C-OOM (only the 64k smoke
  ever fit, node ≈950 there) ⇒ **T2 is structurally out at rank 2 on 122b** —
  its per-rank CPU pools (linear-attn saved offload + fg pools) duplicate on top
  of the ~310 GB arena. This is the llama-R2A TIER-INVERSION signature (c14:
  "at rank 2 the host constraint flips the tier order — T1 above T2"). CHAIN-24
  LAUNCHED: sEP-T1 ladder @128k → 288k → 320k → 352k (T1 = recompute = host-lean;
  linear-attn offload hooks are guarded out of recompute). Decision tree: T1@128k
  fails → 122b R2 closed as characterized limitation (asym ≤64k); T1 reaches
  ≥320k → sole coverage past uns's (288k,320k] wall stands at rank 2 too.
- [14:5x 07-26] **CHAIN-24 DONE — 122b R2 CLOSED.** sEP-T1: @128k = 269.6 · 950
  global · 116.1 (63%) · RSS 494/rank FIT · @288k = 373.4 · 1542 global · 180.9
  (98%) · 543 FIT · @320k G-OOM ⇒ **asym T1 R2 wall = (288k,320k] — TIES uns's
  wall; +31% at the shared 288k edge** (1542 vs 1178). The llama-R2A inversion
  confirmed on 122b: host flips the tier order at rank 2 (T1 host-lean wins;
  T2/T2B/T3 all C-OOM ≤128k under arena+duplicated pools).

## §10 RANK-2 FINAL TABLES (c18 GPUs 0+1, b1/rank ga1, w1+m2; TP = GLOBAL tok/s)

q3.5-35b-a3b (asym = sEP asym_sepplan2, arena 160):
| seq | rc | uns | uns-OFF | asym sEP | verdict |
|---|---|---|---|---|---|
| 256k | 1620 · 68% · 108 | — | — | — | rc last fit |
| 384k | **G-OOM → (256k,384k]** | — | — | — | rc dead (= rank-1) |
| 512k | OOM* | 1892 · 96% · 192 | 1856 · 53% · 416 | **2724 · 71% · 330 (T2) = +44%** | all-alive h2h |
| 576k | OOM* | **G-OOM → (512k,576k]** | — | — | uns dead (= rank-1) |
| 640k | OOM* | OOM | **C-OOM → (512k,640k]** (rank-1 was 1.02M!) | **2812 · 83% · 440 (T2)** | uns-OFF host wall SHRINKS |
| 896k | OOM* | OOM | OOM | **2640 · 97% · 442 (T2)** | sole; last fit |
| 1.02M | OOM* | OOM | OOM | **T2B HOST-C-OOM → asym wall (896k,1024k]** | |
⇒ capacity 896k = 1.6-1.75× every baseline · DP scaling 92% @640k · sole ≥640k.

q3.5-122b-a10b (asym = sEP, arena 400 — bank needs ~310 GB):
| seq | rc | uns | asym sEP-T1 | verdict |
|---|---|---|---|---|
| 128k | 752 · 63% · 412 | (est 1374) | 950 · 63% · 494 (T1) | all alive |
| 288k | **G-OOM → (128k,288k]** | 1178 · 97% · 412 — last fit | **1542 · 98% · 543 (T1) = +31%** | h2h at the edge |
| 320k | OOM* | **G-OOM → (288k,320k]** | **G-OOM → (288k,320k]** | **CAPACITY TIE, asym +31% faster** |
uns-OFF: host-dead at every probed point (rank-1 measured; rank-2 strictly worse).
T2/T2B/T3 rank-2: HOST-C-OOM at 128k-800k (arena ~310 + duplicated per-rank pools
≫ 957) — llama-R2A tier-inversion class, root-caused. Squeeze gap (optional):
asym b2/rank @128k unprobed (HBM 63% → room; would decide the 128k-column sign
vs uns's est). Lever for capacity>320k on bigger-host nodes: #7b linear-attn
keep-acts port (cuts T2's per-rank pools).

**RANK-2 CAMPAIGN CLOSED (both models).** Figures: tp2r_* regenerated (35b = 4th
combined panel; 122b standalone). Overleaf: tp2r_combined.pdf + §Scaling prose
updated & pushed (13eb194). Paper now carries measured rank-1 AND rank-2 qwen3.5
results end-to-end.

---
ADDENDUM (2026-07-30, redo campaign — fix_plot_placeholders.md §5): the §8
"uns-OFF host-dead at every probed 122b point" verdict is SUPERSEDED for
small-batch short-seq: uns-OFF FITS at 32k b4 (403 tok/s, RSS 703) and 128k b1
(377, RSS 710) on c18; the ×b8-cluster and ≥288k HOST-C-OOMs reconfirmed
physical (3rd strike, NUMA-pinned clean env). rc@32k saturates at b6 682 (b4
475 was 31% under). 35b 384k asym T2 = 1810 (b3 plateau). 122b RANK-2: the
(288k,320k] capacity tie is a characterized node-class limit — T2/T2B/T3/
off-recompute all host-dead (arena ~341 GiB + per-rank pools ≫ 957), T1
G-OOM at 320k, delta-chunk + ohbm levers exhausted (ohbm0 = already
max-offloaded; ohbmN keeps roots). Figures + Overleaf synced (318a660).

---
ADDENDUM 2 (2026-07-30, capacity push — fix_plot_placeholders.md §6): the §10
122b R2 "CAPACITY TIE at (288k,320k]" verdict is SUPERSEDED. Root cause of the
tie: the EP2 driver family hard-forced grad/weight offload OFF, leaving ~18 GiB
trainable expert-LoRA weights + ~18 GiB grads resident per rank (rank-1 runs
gradofftrue). Port shipped (run_lf_profiled_train.py + LF parser +
run_lf_lora_sft.sh, opt-in ASYM_EP2_GRAD_OFFLOAD): backward-time sync D2H into
the CPU optimizer's pinned fp32 flat buffer, cross-rank allreduce OVER that
buffer via a chunked GPU stage (DDP-mean semantics preserved; zeros for
absent grads; smoke loss-parity PASS). Winning 320k stack: sEP-T1 ohbm16 +
grad-offload(sync) + weight-offload OFF + arena 345 + watchdog floor 40.
**122b R2 FINAL: asym 1665@128k (+15%) · 1542@288k (+31%) · 1565@320k SOLE
(86% HBM, RSS 562/rank) · wall (320k,352k] host ×2-measured** = capacity
1.11× past uns's wall, sole coverage begins 320k. Lore: ASYNC grad offload
defeats the HBM shed (keepalives hold every CUDA grad until drain — sync is
the capacity mode); weight-offload's pinned copies are net-negative on
host-bound walls; watchdog floor is per-model config (c14 ran 35).

ADDENDUM 2b (2026-07-30 ceiling squeeze): the addendum-2 wall (320k,352k] is
tightened — fine-rung 336k FITS (448.7 · 1498 global · 90% HBM · RSS 558/rank,
same grad-offload stack, floor 35). **122b R2 FINAL: sole coverage 320k AND
336k; ceiling 336k = 1.17× past uns's 288k last fit; wall (336k,352k] with
352k proven over-budget conclusively** (5 attempts: ohbm16/8 × floors 40/35/30
all HOST-C-OOM with the dip bottoming ~1 GiB under every floor; ohbm4 flips to
G-OOM — pincered both ways). Overleaf d9135d9.
