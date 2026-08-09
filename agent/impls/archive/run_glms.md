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
- [2026-08-04 ~21:5xZ] c18 session (back-runner): CLAIM Y1→Y2→X1→X2
  (user order: Air 1r → Air 2r → Flash 1r → Flash 2r; c14 is Flash-first
  forward, so coverage still converges). Tags b-prefixed (by1*/by2*/bx1*/
  bx2*), runner agent/anchors_tmp/glmext_rev.sh in the SFT-46 tree, GPUs
  0/1, solo+serial. Rung-skip rule: if a c14 forward asym cell (y2t1160-
  style) is already logged for a phase+seq, c18 skips that rung. One
  aborted 6-min by2rc160 attempt (pre-reorder) was killed, artifacts
  ignorable. Cell results appended below as they land.

- [08-05 06:53Z] c18 by1 by1rc160 superoffload_mem|recomp s=160000 b=1 -> GOOM | -
- [08-05 07:10Z] c18 by1 by1un160 superoffload_mem|unsloth s=160000 b=1 -> TRAINED | 178.4	897	146.3	79	637	0.1
- [08-05 07:25Z] c18 by1 by1uo160 superoffload_mem|unsloth-off-ohbm0 s=160000 b=1 -> TRAINED | 240.2	666	85.1	46	760	0.1
- [08-05 07:51Z] c18 by1 by1t1160 asym_cpuadamwds|T1 s=160000 b=2 -> TRAINED | 346.6	923	167.3	90	817	0.3
- [08-05 07:56Z] c18 by1 by1rc192 superoffload_mem|recomp s=192000 b=1 -> GOOM | -
- [08-05 08:12Z] c18 by1 by1un192 superoffload_mem|unsloth s=192000 b=1 -> TRAINED | 243.4	789	171.0	92	637	0.2
- [08-05 08:32Z] c18 by1 by1uo192 superoffload_mem|unsloth-off-ohbm0 s=192000 b=1 -> TRAINED | 308.2	623	99.7	54	809	0.3
- [08-05 09:04Z] c18 by1 by1t1192 asym_cpuadamwds|T1 s=192000 b=1 -> TRAINED | 242.5	792	106.9	58	778	0.4
- [08-05 09:10Z] c18 by1 by1rc256 superoffload_mem|recomp s=256000 b=1 -> GOOM | -
- [08-05 09:15Z] c18 by1 by1un256 superoffload_mem|unsloth s=256000 b=1 -> GOOM | -
- [08-05 09:44Z] c18 by1 by1uo256 superoffload_mem|unsloth-off-ohbm0 s=256000 b=1 -> TRAINED | 509.0	503	131.0	71	821	0.1
- [08-05 10:16Z] c18 by1 by1t1256 asym_cpuadamwds|T1 s=256000 b=1 -> TRAINED | 412.4	621	145.7	79	778	0.2
- [08-05 10:22Z] c18 by1 by1rc320 superoffload_mem|recomp s=320000 b=1 -> GOOM | -
- [08-05 10:27Z] c18 by1 by1un320 superoffload_mem|unsloth s=320000 b=1 -> GOOM | -
- [08-05 10:43Z] c18 by1 by1uo320 superoffload_mem|unsloth-off-ohbm0 s=320000 b=1 -> COOM | -
- [08-05 11:28Z] c18 by1 by1t1320 asym_cpuadamwds|T1 s=320000 b=1 -> TRAINED | 628.0	510	168.8	91	878	0.3
- [08-05 11:28Z] c18: PHASE-COMPLETE by1 (Air 1r)
- [08-05 11:32Z] c18 by2 by2rc160 superoffload_mem|recomp s=160000 b=1 -> GOOM | -
- [08-05 11:50Z] c18 by2 by2un160 superoffload_mem|unsloth s=160000 b=1 -> TRAINED | 190.7	1678	146.9	79	424	0.1
- [08-05 11:54Z] c18 by2 by2uo160 superoffload_mem|unsloth-off-ohbm0 s=160000 b=1 -> COOM | -
- [08-05 12:30Z] c18 by2 by2t1160 asym_sdp2_cpuadamwds|T1 s=160000 b=2 -> TRAINED | 379.7	1686	181.7	98	605	4.2
- [08-05 12:35Z] c18 by2 by2rc192 superoffload_mem|recomp s=192000 b=1 -> GOOM | -
- [08-05 12:52Z] c18 by2 by2un192 superoffload_mem|unsloth s=192000 b=1 -> TRAINED | 255.4	1503	171.0	92	424	0.1
- [08-05 12:56Z] c18 by2 by2uo192 superoffload_mem|unsloth-off-ohbm0 s=192000 b=1 -> COOM | -
- [08-05 13:24Z] c18 by2 by2t1192 asym_sdp2_cpuadamwds|T1 s=192000 b=1 -> TRAINED | 244.2	1573	130.2	70	636	0.2
- [08-05 13:55Z] c18 by2 by2rc256 superoffload_mem|recomp s=256000 b=1 -> FAIL | -
- [08-05 14:04Z] c18 by2 by2un256 superoffload_mem|unsloth s=256000 b=1 -> GOOM | -
- [08-05 14:09Z] c18 by2 by2uo256 superoffload_mem|unsloth-off-ohbm0 s=256000 b=1 -> COOM | -
- [08-05 14:44Z] c18 by2 by2t1256 asym_sdp2_cpuadamwds|T1 s=256000 b=1 -> TRAINED | 415.3	1233	160.1	87	637	0.4
- [08-05 15:16Z] c18 by2 by2rc320 superoffload_mem|recomp s=320000 b=1 -> FAIL | -
- [08-05 15:24Z] c18 by2 by2un320 superoffload_mem|unsloth s=320000 b=1 -> COOM | -
- [08-05 15:29Z] c18 by2 by2uo320 superoffload_mem|unsloth-off-ohbm0 s=320000 b=1 -> COOM | -
- [08-05 16:09Z] c18 by2 by2t1320 asym_sdp2_cpuadamwds|T1 s=320000 b=1 -> TRAINED | 647.1	989	182.0	98	609	1.5
- [08-05 16:09Z] c18: PHASE-COMPLETE by2 (Air 2r)
- [08-05 16:50Z] c18 by2 by2rc256r superoffload_mem|recomp s=256000 b=1 -> FAIL | - (RANKSYNC fix validation; replaces FAIL)
- [08-05 17:41Z] c18 by2 by2rc256f superoffload_mem|recomp s=256000 b=1 -> FAIL | - (report-path fix validation; replaces FAIL)
- [08-05 17:47Z] c18 by2 by2rc320f superoffload_mem|recomp s=320000 b=1 -> COOM | - (report-path fix validation; replaces FAIL)
- [08-05 17:47Z] c18 by2 by2rc256g superoffload_mem|recomp s=256000 b=1 -> COOM | - (fatal-exit fix validation; replaces FAIL)
- [08-05 17:5xZ] c18 CORRECTION: by2rc320f COOM and by2rc256g COOM above are
  INVALID — two rcredo instances raced on GPUs 0/1 (operator error; the f-
  instance continued past its 256f FAIL while the g-instance launched), so
  both "dropped below floor" verdicts are dual-Air-load host contention, not
  cell truth. Superseded by the SOLO by2rc256h/by2rc320h cells below. The
  rc256 FAIL class itself is DIAGNOSED+FIXED: one rank's routing-dependent
  31.25 GiB grouped-GEMM alloc CUDA-OOMs; the dying rank previously hung the
  peer's ZeRO-3 allgather until the 25-min NCCL watchdog (report() model-walk
  on the failure path was a second aggravator, also fixed). Harness now
  prints the fatal error and hard-exits on multi-rank failure
  (run_lf_profiled_train.py), so the peer is reaped in seconds.
- [08-05 18:20Z] c18 by2 by2rc256h superoffload_mem|recomp s=256000 b=1 -> FAIL | - (fatal-exit fix validation, SOLO; replaces FAIL)
- [08-05 18:54Z] c18 by2 by2rc320h superoffload_mem|recomp s=320000 b=1 -> FAIL | - (fatal-exit fix validation, SOLO; replaces FAIL)
- [08-05 19:05Z] c18 by2 by2rc256i superoffload_mem|recomp s=256000 b=1 -> GOOM | - (fatal-exit fix validation, SOLO; replaces FAIL)
- [08-05 19:13Z] c18 by2 by2rc320i superoffload_mem|recomp s=320000 b=1 -> GOOM | - (fatal-exit fix validation, SOLO; replaces FAIL)
- [08-05 19:37Z] c18 by1ext by1t1384 asym_cpuadamwds|T1 s=384000 b=1 -> GOOM | -
- [08-05 20:31Z] c18 by1ext by1t2384 asym_cpuadamwds|T2 s=384000 b=1 -> TRAINED | 918.5	418	154.9	84	913	0.1
- [08-05 20:32Z] c18 by1 s=160000: SKIP — rung already complete in ledger
- [08-05 20:32Z] c18 by1 s=192000: SKIP — rung already complete in ledger
- [08-05 20:32Z] c18 by1 s=256000: SKIP — rung already complete in ledger
- [08-05 20:32Z] c18 by1 s=320000: SKIP — rung already complete in ledger
- [08-05 20:32Z] c18: PHASE-COMPLETE by1 (Air 1r)
- [08-05 20:32Z] c18 by2 s=160000: SKIP — rung already complete in ledger
- [08-05 20:32Z] c18 by2 s=192000: SKIP — rung already complete in ledger
- [08-05 20:32Z] c18 by2 s=256000: SKIP — rung already complete in ledger
- [08-05 20:32Z] c18 by2 s=320000: SKIP — rung already complete in ledger
- [08-05 20:32Z] c18: PHASE-COMPLETE by2 (Air 2r)
- [08-05 21:02Z] c18 by2ext by2t1384 asym_sdp2_cpuadamwds|T1 s=384000 b=1 -> GOOM | -
- [08-05 21:08Z] c18 by2ext by2t2384 asym_sdp2_cpuadamwds|T2 s=384000 b=1 -> FAIL | -
- [08-05 21:16Z] c18 by2ext by2t2384b asym_sdp2_cpuadamwds|T2 s=384000 b=1 -> FAIL | -
- [08-05 21:36Z] c18 by2ext by2t2384c asym_sdp2_cpuadamwds|T2 s=384000 b=1 -> COOM | -
- [08-05 21:41Z] c18 by2ext by2t3384c asym_sdp2_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0 s=384000 b=1 -> FAIL | -
- [08-05 21:4xZ] c18: by2t3384c FAIL reclassified = HOST-FABRIC-C-OOM
  (SIGBUS, exitcode -7: T3's 384k-2r banks exhaust the 479G /dev/shm tmpfs;
  T2 at cap 420 GiB was an honest host COOM). 384k-2r walled for EVERY
  asym tier -> 2r asym wall (320k,384k]; 6th 2r rung moves to 352k
  (air_ext6b: T1->T2 ladder). 1r 448k (T2->T3) queued in the same chain.
- [08-05 22:04Z] c18 by2ext by2t1352 asym_sdp2_cpuadamwds|T1 s=352000 b=1 -> GOOM | -
- [08-05 22:12Z] c18 by2ext by2t2352 asym_sdp2_cpuadamwds|T2 s=352000 b=1 -> FAIL | -
- [08-05 23:03Z] c18 by2ext by2t3352 asym_sdp2_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0 s=352000 b=1 -> FAIL | -
- [08-05 23:0xZ] c18: 352k-2r walled at ALL tiers (by2t1352 G-OOM · by2t2352
  SIGBUS · by2t3352 SIGBUS — same shm-fabric exhaustion class as 384k) ->
  2r asym wall tightens to (320k,352k]. Ceiling pinned within 10% of the
  320k trained rung; fallback 336k probe queued after the 1r 448k block.
- [08-06 00:16Z] c18 by1ext by1t2448 asym_cpuadamwds|T2 s=448000 b=1 -> TRAINED | 1173.6	382	180.5	98	917	9.7
- [08-06 00:46Z] c18 by2ext by2t1336 asym_sdp2_cpuadamwds|T1 s=336000 b=1 -> GOOM | -
- [08-06 00:46Z] c18: by2t1336 G-OOM -> AIR 2r CEILING MEASURED (320k,336k]
  (T1; T2/T3 shm-fabric-dead across 336k-384k). 320k trained rung = within
  5% of the absolute 2r cap. AIR COMPLETE both ranks: 1r rungs 160k/192k/
  256k/320k/384k/448k (sole from 320k, T2 carries 384k+448k @98% HBM);
  2r rungs 160k/192k/256k/320k (sole from 256k) + measured ceiling bracket.
- [08-06 01:21Z] c18 bx2 bx2rc256 superoffload_mem|recomp s=256000 b=1 -> TRAINED | 424.9	1205	128.1	69	72	0.0
- [08-06 02:06Z] c18 bx2 bx2un256 superoffload_mem|unsloth s=256000 b=2 -> TRAINED | 837.7	1222	165.3	89	173	0.2
- [08-06 02:58Z] c18 bx2 bx2uo256 superoffload_mem|unsloth-off-ohbm0 s=256000 b=2 -> TRAINED | 987.6	1037	68.5	37	424	0.2
- [08-06 03:29Z] c18 bx2 bx2fd256 fsdp2_offload|recomp s=256000 b=1 -> TRAINED | 418.7	1223	132.7	72	166	0.3
- [08-06 04:41Z] c18 bx2 bx2t1256 asym_sdp2_cpuadamwds|T1 s=256000 b=3 -> TRAINED | 1333.9	1152	182.0	98	368	1.7
- [08-06 05:19Z] c18 bx2 bx2rc320 superoffload_mem|recomp s=320000 b=1 -> TRAINED | 657.7	973	159.8	86	72	0.1
- [08-06 05:55Z] c18 bx2 bx2un320 superoffload_mem|unsloth s=320000 b=1 -> TRAINED | 657.6	973	105.5	57	173	0.2
- [08-06 06:36Z] c18 bx2 bx2uo320 superoffload_mem|unsloth-off-ohbm0 s=320000 b=1 -> TRAINED | 750.0	853	48.0	26	313	0.2
- [08-06 07:13Z] c18 bx2 bx2fd320 fsdp2_offload|recomp s=320000 b=1 -> TRAINED | 652.0	982	164.3	89	166	0.3
- [08-06 08:24Z] c18 bx2 bx2t1320 asym_sdp2_cpuadamwds|T1 s=320000 b=2 -> TRAINED | 1300.7	984	157.6	85	368	0.2
- [08-06 08:33Z] c18 bx2 bx2rc416 superoffload_mem|recomp s=416000 b=1 -> GOOM | -
- [08-06 09:35Z] c18 bx2 bx2un416 superoffload_mem|unsloth s=416000 b=1 -> TRAINED | 1103.7	754	135.5	73	173	0.2
- [08-06 10:40Z] c18 bx2 bx2uo416 superoffload_mem|unsloth-off-ohbm0 s=416000 b=1 -> TRAINED | 1227.1	678	62.7	34	313	0.3
- [08-06 10:48Z] c18 bx2 bx2fd416 fsdp2_offload|recomp s=416000 b=1 -> GOOM | -
- [08-06 12:51Z] c18 bx2 bx2t1416 asym_sdp2_cpuadamwds|T1 s=416000 b=2 -> TRAINED | 2306.6	721	182.0	98	369	1.2
- [08-06 14:20Z] c18 bx2 bx2un512 superoffload_mem|unsloth s=512000 b=1 -> TRAINED | 1668.0	614	166.6	90	174	0.2
- [08-06 15:56Z] c18 bx2 bx2uo512 superoffload_mem|unsloth-off-ohbm0 s=512000 b=1 -> TRAINED | 1822.4	562	75.5	41	425	0.1
- [08-06 17:27Z] c18 bx2 bx2t1512 asym_sdp2_cpuadamwds|T1 s=512000 b=1 -> TRAINED | 1660.9	617	128.0	69	264	0.2
- [08-06 19:21Z] c18 bx2 bx2un576 superoffload_mem|unsloth s=576000 b=1 -> TRAINED | 2159.8	533	179.9	97	274	1.1
- [08-06 19:32Z] c18 bx2 bx2uo576 superoffload_mem|unsloth-off-ohbm0 s=576000 b=1 -> COOM | -
- [08-06 21:25Z] c18 bx2 bx2t1576 asym_sdp2_cpuadamwds|T1 s=576000 b=1 -> TRAINED | 2103.5	548	146.9	79	367	0.2
- [08-06 21:42Z] c18 bx2 bx2un640 superoffload_mem|unsloth s=640000 b=1 -> GOOM | -
- [08-07 00:03Z] c18 bx2 bx2t1640 asym_sdp2_cpuadamwds|T1 s=640000 b=1 -> TRAINED | 2598.0	493	158.8	86	367	0.2
- [08-07 02:52Z] c18 bx2 bx2t1704 asym_sdp2_cpuadamwds|T1 s=704000 b=1 -> TRAINED | 3143.9	448	173.2	94	368	0.2
- [08-07 03:24Z] c18 bx1 bx1rc256 superoffload_mem|recomp s=256000 b=1 -> TRAINED | 417.8	613	126.4	68	124	0.1
- [08-07 04:12Z] c18 bx1 bx1uo256 superoffload_mem|unsloth-off-ohbm0 s=256000 b=2 -> TRAINED | 885.9	578	68.5	37	476	0.1
- [08-07 04:42Z] c18 bx1 bx1fd256 fsdp2_offload|recomp s=256000 b=1 -> TRAINED | 415.8	616	129.1	70	260	0.3
- [08-07 05:29Z] c18 bx1 bx1t1256 asym_cpuadamwds|T1 s=256000 b=2 -> TRAINED | 829.2	617	122.1	66	291	0.1
- [08-07 08:58Z] c18 bx2deep bx2t1d768 asym_sdp2_cpuadamwds|T1 s=768000 b=1 -> TRAINED | 3797.2	405	182.0	98	368	1.2
- [08-07 12:55Z] c18 bx2deep bx2t1d832 asym_sdp2_cpuadamwds|T1 s=832000 b=1 -> TRAINED | 4479.2	371	182.1	98	369	0.4
- [08-07 14:33Z] c18 bx2deep bx2t1d896 asym_sdp2_cpuadamwds|T1 s=896000 b=1 -> GOOM | -
- [08-07 19:07Z] c18 bx2deep bx2t2d896 asym_sdp2_cpuadamwds|T2 s=896000 b=1 -> TRAINED | 5271.7	340	180.7	98	510	0.2
- [08-08 00:27Z] c18 bx2deep bx2t2d960 asym_sdp2_cpuadamwds|T2 s=960000 b=1 -> TRAINED | 6129.0	313	182.0	98	494	0.2
- [08-08 06:30Z] c18 bx2deep bx2t2d1024 asym_sdp2_cpuadamwds|T2 s=1024000 b=1 -> TRAINED | 6965.4	294	181.8	98	494	0.4
- [08-08 07:10Z] c18 bx2deep bx2t2d1088 asym_sdp2_cpuadamwds|T2 s=1088000 b=1 -> COOM | -
- [08-08 07:46Z] c18 bx2deep bx2t3d1088 asym_sdp2_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0 s=1088000 b=1 -> COOM | -
- [08-08 07:46Z] c18 bx2deep: ASYM 2r CEILING — all tiers walled at 1088k; last trained rung stands as the crown.
