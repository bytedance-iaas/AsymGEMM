# GLM turning-point campaign — handoff (2026-08-04)

PROMPT FOR THE RUNNER AGENT: You are extending the GLM throughput panels
(GLM-4.7-Flash, GLM-4.5-Air; 1 rank AND 2 ranks) beyond native context until
every baseline OOMs and asym is the last system standing. Run everything
inside YOUR machine's enroot container from YOUR machine's own repo tree —
do NOT ssh to any other node, do NOT touch another machine's tree; the only
shared artifacts are the HF weight cache and THIS doc (shared disk). Another
session (c14) is running the same ladder forward (X1→X2→Y1→Y2) — CLAIM your
phase in §Log below before running, work the REVERSE order (Y2→Y1→X2→X1),
and skip any phase already claimed or complete. Don't stop until every
phase you claimed has all walls measured.

## §1 WHY (context)
- The paper's tp-vs-seq panels tell one story per model: baselines die as seq
  grows, AsymLoRA survives deepest. Qwen/Llama panels run FAR beyond native
  ctx (llama 448k on 128k ctx; q3-30b at 1.4M). The original GLM campaign
  (agent/anchors_tmp/GLMTP_CAMPAIGN.md) capped at native ctx — Flash 192k
  (ctx 202k), Air 128k (ctx 131k) — so almost nothing walls and the panels
  show a flat tie. User directive: extend like every other model.
- Beyond-ctx is LEGAL for these two: both configs verified full-attention
  RoPE, NO sliding_window / windowed layers (unlike Phi-3.5 which switches
  attention regime — never extrapolate that one). Beyond-ctx = same compute,
  longer positions. If a run fails for a non-memory reason, that's a bug to
  fix, not a wall.
- Sizes: Air ≈ 106B MoE (heavy: ~200 GB asym banks, rc/uns ~480 GB host per
  rank at 2r) · Flash ≈ 30B-class MoE (light).

## §2 GOAL (what "done" means, per model × rank)
1. Each baseline series has a MEASURED first-OOM wall: rc (SuperOffload),
   un (+Unsloth-GC), uo (+Unsloth-GC-Offload), and for FLASH also fd
   (FSDP2-Offload). Air fd needs NO runs: load-phase host-OOM at 16k
   (fp32 masters), seq-independent — banked OOM everywhere.
2. asym has ≥1 (ideally 2) TRAINED rungs beyond the last baseline wall
   (tier-promote T1→T2→T3 before declaring an asym wall).
3. Every cell verdict is real (trained value or measured OOM) — zero OOM*.
   Metrics per TRAINED cell: eff tok/s (step_samples, w1+m2 house metric;
   2r = global), batch, peak HBM, peak RSS.

## §3 LADDERS (extension rungs; in-ctx rungs are DONE — harvest, don't rerun)
Batch lists walk DOWN on OOM; first TRAINED wins. asym starts T1; on
full-list OOM promote T2 (top-2 batches), then T3-raw.
- X1 Flash 1r: 256k [2 1] · 320k [2 1] · 384k [1] · 448k [1]
- X2 Flash 2r: 256k [rc/un/uo/fd 2 1; asym 3 2] · 320k [1; asym 2 1] ·
  416k [1; asym 2 1] · 512k [1]
- Y1 Air 1r: 160k [2 1] · 192k [2 1] · 256k [1] · 320k [1]
- Y2 Air 2r: 160k [rc 1; un 2 1; uo 1; asym 3 2] · 192k [2 1] · 256k [1;
  asym 2 1] · 320k [1]
- If ANY baseline still fits at the ladder cap, extend that model/rank in
  +64–96k steps until it walls. Walls already measured in the original
  campaign: Flash-1r un@192k; Air-2r uo@96k/128k (host-transient class).

## §4 HOW (exact mechanics — mirror of the original campaign)
- Container: your machine's asym enroot image, your repo mounted at
  /workspace/<your-tree>/third_party/AsymGEMM. NEVER run on the host.
  HF cache (weights present for both GLMs):
  HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
- Templates to COPY into your tree (read them from the 39 tree, read-only):
  agent/anchors_tmp/tpfig_lib.sh (run_cell/guard/verdict — NOTE the guard
  must count only /proc-live pids; ghost nvidia-smi entries are a known
  trap) and agent/anchors_tmp/glmext.sh (implements the exact ladder above).
- One cell = profile_lora_lf_test_source.sh with
  RUNS="<model>|<ranks> ; <systok>|ligerloss1 ; <seq>|<b>|1 ; none|false|false|false|false|false"
  models: glm4.7-flash / glm4.5-air. System tokens:
  rc superoffload_mem|recomp · un superoffload_mem|unsloth ·
  uo superoffload_mem|unsloth-off-ohbm0 · fd fsdp2_offload|recomp ·
  asym 1r asym_cpuadamwds|T1 (then T2; T3 =
  asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0) ·
  asym 2r backend asym_sdp2_cpuadamwds (ep2-vanilla swap is qwen3-only).
- Env: PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512
  ASYM_ZERO_ROUTER_JITTER=1, numactl membind+cpunodebind 0,1.
  1r: CUDA_VISIBLE_DEVICES=<gpu>, host floor 500 (Flash) / 600 (Air).
  2r: CUDA_VISIBLE_DEVICES=0,1 GPU_POOL=0,1 DDP_TIMEOUT=1500, floor 1200.
  Air (any rank): ASYM_ARENA_SHM_CAP_GB=240 (banks ~200 GB > 160 default).
- Verdicts from the run log + jobs.tsv: CUDA OOM → GOOM · "dropped below
  floor" → COOM(host) · jobs.tsv ok-row → TRAINED · else FAIL (debug it;
  FAIL is never a wall).
- DISCIPLINE: strictly SOLO+serial on your node (parallel lanes once
  measured up to −44% low); one batch attempt at a time; fresh tag per
  attempt; PREFIX your tags (e.g. bx2rc256) so they never collide with
  c14's x1/x2/y1/y2 tags; never edit a script while a run is live.

## §5 REPORTING / BANKING
- Append every cell to §Log here: phase, tag, system, seq, b, verdict,
  eff tok/s (+HBM/RSS for TRAINED). This doc is the shared ledger.
- Do NOT edit scripts/figures/plot_tp_vs_seq*.py, do NOT push Overleaf —
  the c14 session banks centrally after both runners finish (the figure
  scripts are hardlinked across trees; double-banking corrupts them).
  zero3 columns for new rungs are derived-where-rc-fits (banking-time rule).

## §Log (append-only; claims + results)
- [2026-08-04 23:1x] c14 session: CLAIM X1→X2→Y1→Y2 (forward, in progress;
  X1 256k rc b2 GOOM already — first beyond-ctx verdict, mechanism proven).
  Back-runner: claim from Y2 backward; skip anything already claimed or
  logged complete below.
- [2026-08-06 12:5x] c14: FLASH 1r PROGRESS BANK pushed to Overleaf (d954614).
  Walls measured: rc (320k,384k] · fd (384k,448k] · un (576k,640k] (192k
  stale-OOM re-probed -> 795 banked) · uo ALIVE thru 896k (172 tok/s, RSS 772
  GiB). Lean six (interim): 128k/256k/320k/384k/448k/640k — 640k column = un
  dead, uo 235 vs asym T1 248. Stage 3 in flight: uo 1024k/1152k (host-wall
  bracket) + asym deep trio 1024k/1152k/1280k (T2-start). Final pick swaps in
  the uo-wall + asym-sole columns. 512k/576k asym deliberately unmeasured
  (never-rendered; targeting trim). Air row untouched here (other runner's
  complete cascade rides along in the re-render).
- [2026-08-07 12:5x] c14: CLAIM solo-column round E (do not fork parallel
  fixes — d-round's ohbm5 GOOM'd 12:42; HBM model under-counts ~10 GiB w/
  4.7-GB pinned roots, notch-dialing at 1152k is razor-edge). Plan E:
  (1) uo@1088k floor-25 probe (64k house rung between uo's 1024k fit and
  1152k wall; its 1152k death was EARLY-warmup -> likely dead at 1088k too);
  (2) if uo COOMs -> T3-RAW@1088k (ohbm0 — verdict-config-pure, projected
  avail bottom ~69 GiB, no HBM risk) = the solo column, no ohbm debate;
  (3) if uo fits 1088k -> bank its crawl + fall back to T3-ohbm8@1152k
  (6 roots ≈ 28 GB, corrected peak ≈ 166 — 18 GiB HBM margin).
- [2026-08-07 13:0x] c14: NOTE a foreign-launched cell x1euo1088 (uo@1088k,
  wall-bracket tightening) is LIVE on c14 GPU0 — not from this session's
  chains. Cooperative and useful (narrows uo's wall to (1088k,1152k] or
  moves the last-stand column to 1088k); this session's ohbm6 solo-column
  chain stood down (guard would have 60-min-timed-out) and relaunches after
  it completes. Whoever launched it: please claim cells in this ledger first
  — c14 runs are this session's per the machine-ownership rule.
- [2026-08-07 14:4x] c14 COORDINATION: round E is CLAIMED and LIVE — the
  x1et3r1088 solo cell (T3-raw @1088k, floor 25) has held GPU0 since 13:22
  and runs ~6h. The 12:52/13:26 "GLM1RE (ohbm6)" duplicate launches guard-
  blocked and one emitted a spurious X1E2-DONE at its guard-timeout (14:26)
  — markers in tpfig_status.log from those instances are VOID. Other runner:
  please do NOT launch further Flash-1r cells until this claim closes with
  an E-SOLO-1088 verdict line + a §Log close-out here. uo wall TIGHTENED:
  (1024k,1088k] COOM at matched floor 25 (x1euo1088).
- [2026-08-07 16:4x] c14: round E closed — E-SOLO-1088 = COOM (T3-raw died
  16:38, 3h16m in, mid-step-2 — a ~20-30 GiB host shortfall, vs uo's warmup
  death; the separation window is real but needs root relief). CLAIM round F:
  T3-ohbm8@1088k (6 roots ≈ 27 GB HBM, corrected peak ≤ 180); fallback on
  COOM = 1056k pair (uo probe + T3-raw — both projected to separate there).
  Stand-down request stands until F closes here.
- [2026-08-07 23:5x] c14: FLASH 1r COMPLETE + PUSHED (6f5b1a9). Round F
  closed the cascade: T3-ohbm8@1088k TRAINED 142 tok/s @181.7 GiB RSS 886
  (uo COOM at matched floor 25 -> ASYM-ALONE column real). Final lean six
  128k/384k/448k/640k/1024k/1.09M — one baseline class dies per column;
  visual check vs sibling panels PASSES. Asym deeper walls (1152k T2 G-OOM
  / T3-raw C-OOM) banked as record, render-excluded. Option A (asym ceiling
  walk past 1.09M, T3-ohbm8/ohbm6 at 1152k+) queues next per user standing
  order; 2r phases and any Air follow-ups remain parked behind it.
- [2026-08-08 ~11:0x] c14: OPTION A CLOSED + PUSHED (787a5a1). T3-ohbm8
  @1152k TRAINED — crown 1.15M, 134 tok/s @181.8 GiB RSS 880. Banked as
  complete-variant-only (lean six unchanged, one kill per column). FLASH 1r
  CAMPAIGN FULLY COMPLETE: every baseline wall measured (rc/fd/un/uo),
  matched-floor fairness pairs at the deep end, asym sole at 1.09M and
  crowned at 1.15M. asym wall beyond 1.15M unprobed (record note). PARKED
  awaiting user: Flash 2r (glmext2.sh X2 base+ext ready), Air follow-ups,
  any further Option-A rungs (1280k T3-ohbm8).
- [2026-08-08 ~10:0x] c14: OPTION A CLOSED — x1gt3o8_1152 TRAINED 134 tok/s
  @181.8 GiB RSS 880 (06:53). Crown banked as ("T3",134) @1.15M by the
  co-runner (values agree with c14 extraction exactly) and pushed (787a5a1);
  lean keeps 1.09M as the asym-alone closer, 1.15M = record/complete. Flash
  1r is COMPLETE end-to-end: 6-column cascade, all walls measured, fairness
  floors matched, crown certified. c14 stands down from Flash-1r; Air/2r
  remain parked pending Kevin's direction.
