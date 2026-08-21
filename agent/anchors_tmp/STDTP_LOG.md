# STDTP — TP-figure x-axis standardization campaign log (c14, 2026-08-20)
Doc: agent/impls/s04-p1-dgx-02-c06/standardize_tps.md. All four agents' jobs run
SERIALLY on s04-p1-dgx-02-c14 (user 2026-08-20), order 3 -> 4 -> 1 -> 2.
GPUs (user): 2-rank = phys 1+3, 1-rank = phys 3 (containers see 0,1 / 0 via
NVIDIA_VISIBLE_DEVICES). Tree: AsymGEMM-SFT-38 mounted into the asym_sft_46
rootfs (asym_sft_38's rootfs on c14 is a broken shell — no /root, no /bin/sh).
GPU0 avoided: 3.5 GB leaked by dead pids (needs GPU reset). Protocol: house
w1+m2, PROFILERS=source, MAX_SAMPLES=512, NUMACTL membind 0,1, serial+solo,
tags <cell>-c14s_. Status: stdtp_status.log; run logs r_std_<tag>_b<b>.log.

## Decisions / deviations
- 30B uns-1r@768K probe (doc table): SKIPPED by default — DATA records a
  MEASURED uns wall (640k,660k] (c14 tput campaign), so 768K is OOM by
  monotonicity (doc rule 2). The doc's (640,800] bracket ignored the 660k OOM.
- Flash 1r uo@512k/768k: TRAINED in x1e (status log) but artifacts+values lost
  -> banked as est (house t(S)=a+bS fit over 6 measured rungs 256k-1024k);
  uo is MAIN_DROP'd, so the main figure is unaffected.
- Air 2r gate starts at T2@448k (2r T1@320k banked at 98% HBM -> T1@448k
  physically excluded); T1@384k probe pins the 2r T1 wall that makes the
  448k tier label ladder-legal by monotonicity.
- Mixtral fused checkpoint absent on c14 scratch -> regenerate locally before
  Agent 2 (driver M-map hard-points at it).

## Cell log
- [08-20 02:5x] CONCURRENCY NOTE: scripts/figures is a SYMLINK to the shared
  figures dir (one copy for all worktrees/machines). A second session
  (".bak-stdtps" backups, 02:35-02:42) implemented the shared MAIN_RUNGS
  render piece (both files; Air pending its gate; 30B/gpt-oss label "1.02M")
  and banked 35B-1r 640k/768k reuse cells (Agent 2 material). Discipline here:
  re-read fresh before every shared-file edit, surgical edits only, verify +
  log after; before starting Agent 2 (last), re-check what is already banked.

## Per-agent run plans (gap analysis vs DATA, 2026-08-20 02:5x)
Main-series = recomp/unsloth/zero3/fsdp2/megatron/asym; uo = unsloth_off
(MAIN-dropped, still banked for lean/complete). "probe" = rung inside a
measured wall bracket. Reuse-verbatim everywhere a measured cell exists.

### Agent 3 — GLM-4.5-Air + GLM-4.7-Flash [IN PROGRESS]
- Air 1r: complete (all 6 rungs measured). Air 2r: only asym@384k/448k missing
  (all baselines beyond measured walls -> OOM monotone). = the cap gate,
  chain stdtp_a3_air.sh RUNNING (a2t2448 -> a2t1384 -> a2t2384 [+T2B/T3 fb]).
- Flash 2r@384k (chain stdtp_a3_flash2r.sh): rc probe b1 (wall (320,416]),
  fd probe b1 (same), un b1, uo b1, asym sdp2-T1 "2 1". zero3 stays
  derived-est per row rule; megatron OOM.
- Flash 1r (chain stdtp_a3_flash1r.sh): asym@512k T1 b1; asym@768k T1->T2;
  asym@896k from 768k tier. un@512k reuse 310 (banked comment). uo@512k/768k
  est (house fit; x1e artifacts lost). rc/fd/zero3/mega OOM monotone;
  896k uo=172 (banked comment).
- Banking: add Air 384k/448k cols (2r) + Flash 512k/768k/896k (1r), 384k (2r);
  add "glm4.5-air" to MAIN_RUNGS (both files) per gate outcome.

### Agent 4 — Hunyuan-A13B + gpt-oss-20B [chains written]
- gpt-oss-20B: render-only ("1.02M"=1024000 both ranks, labels already match).
- Hy 1r (stdtp_a4_hy1r.sh): 160k rc/un/uo b1 + asym "2 1"; 224k rc PROBE
  (wall (192,256]) + un/uo/asym b1; 288k un PROBE (doc sole-rung decider,
  wall (256,320]) + uo/asym b1. rc@288k OOM monotone after 224k resolves
  (256k GOOM measured).
- Hy 2r (stdtp_a4_hy2r.sh): 160k rc/un b1 + uo PROBE (wall (128,192]) +
  asym sdp2 "2 1"; 224k rc PROBE + un b1 + asym b1; uo@224k OOM monotone
  (192k COOM measured). Tokens: un = unsloth-ohbm0 (hy convention);
  ASYM_OFFLOAD_MODULES=routed_experts,shared_experts,attention,norms,mlp_dense;
  floor 1100; T1 default arena.

### Agent 1 — Qwen3-30B-A3B + Qwen3.5-122B-A10B [plan]
- 30B 1r missing 384/512/768/896/1024k ("1.02M" label): rc@384k b1 (wall
  (392,400] -> fits), zero3@384k b1 (rc-class), fsdp2@384k PROBE (bracket
  (320,480]), uns@384k+512k b1 (wall (640,660] measured -> 768k+ OOM
  monotone; doc's 768k probe SKIPPED per rule 2 — flagged to user), uo@384k/
  512k/768k/896k/1024k (fits thru 800k banked; 1.1M host-OOM -> 896k/1024k
  PROBES), asym T2 b1 @384/512/768k, T2@896k PROBE (T2 800k fit, T2B 1.1M),
  ladder to T2B; @1024k from 896k tier. Backend/env from the c14 tput +
  chainZ/fill ledgers (test_throughpout_v2.md, fix_plot_placeholders.md §3).
- 30B 2r missing 512/768/896/1024k: uns@512k b1 (wall (640,660]); uo@512k
  PROBE (wall (384,640]); fsdp2@512k PROBE (bracket (384,640]); asym sEP-T2
  b1 @512/768/896/1024k (720-1.04M banked around them). rc/zero3 OOM monotone.
  Backend = R2A sEP (exact token from fix_plot_placeholders.md — verify).
- 122B 1r missing 160/192/224/256k: rc walk-up PROBES from 160k (bracket
  (128,288]; 2r wall (192,256] suggests death ~224-256k); uns b1 x4 (fits
  288k); uo walk-up PROBES from 160k (bracket (128,288]); zero3 b1 where rc
  fits (measured-row convention); asym T1 "3 2 1"@160k then per-neighbor
  seeds. fsdp2/mega OOM.
- 122B 2r missing 160/224k: rc@160k b1 + rc@224k PROBE (wall (192,256]);
  uns "2 1"@160k, b1@224k; asym sEP-T1 "2 1"@160k, b1@224k; uo OOM (128k
  C-OOM measured). sEP arena 345-400, floors 35-40 — replicate §5/§6/§7.
  NOTE: 122B 2r rc/zero3/asym 128k/192k DATA cells are (v,"est") while the
  comment cites measured 752/1035 — resolve from ledger before banking.

### Agent 2 — Qwen3.5-35B-A3B + Mixtral-8x22B [LAST — recheck first]
Another session already banked 35B-1r 640k/768k reuse cells (02:42). Before
running: re-diff DATA vs the doc grids; remaining known gaps: 35B 2r@768k
asym sEP-T2 b1 (uns/uo/rc/z3/fd OOM monotone); mixtral 1r 160/224/288k
(rc@160k PROBE (128,192]; un 160/224 b1 + 288k PROBE (256,320] = doc decider;
uo@160k PROBE; asym T1 b1 x2 + 288k T1 PROBE -> T2 fb) + 2r 160/224k
(rc/uo/zero3@160k PROBES (128,192]; un "2 1"/b1; asym sdp2-T1 "2 1"/b1;
fsdp2 derived row). Mixtral REQUIRES fused ckpt regen on c14 first
(driver M-map: /scratch_local/.../fused/Mixtral-8x22B-v0.1; find the
fusion script from the c12 campaign or rebuild from mx2f provenance);
fabric shm rm between runs; floor 35; arena cap 285.
- [08-20 02:5x] Air gate attempt 2: datasets OK (copies fixed the symlink FAIL),
  model loaded, then FATAL at trainer start: shared fabric cap exceeded
  (bank used 256.7GB > cap 240GiB) — 448k Air-2r bank outgrew the 320k-era
  cap. Chain aborted on infra-FAIL as designed. Fixes: ASYM_ARENA_SHM_CAP_GB
  240->320 (hy-T2B precedent), per-attempt run-log truncation in run_cell
  (stale-log verdict trap). Relaunching attempt 3.
- [08-20 02:5x] BONUS: killing the orphaned SFT-39 dataloaders also freed the
  phantom GPU memory — GPU0 3520->2 MiB, GPU1 771->2 MiB. GPU3 keeps ~770 MiB.
- [08-20 03:1x] Fabric-cap explanation found (no regression): shared_fabric.py
  unchanged since 07-08; fg-family tiers (T2/T2B/T3) fabric = bank + fg bases
  (~2x expert bytes; HY precedent). Air fg fabric ≈ 410-427G vs T1's ~214G.
  July Air-2r cells were all T1 under cap 240. Gate chain cap -> 450
  (shm 479G ceiling). d2t1320 diagnostic (T1@320k cap 240, W1+M1) running as
  the banked-era sanity; gate relaunch after it. If T2/T2B/T3 all HOST-COOM
  at 448k AND 384k, that IS the gate answer -> doc's stop-and-report branch.
- [08-20 03:38] d2t1320 DIAGNOSTIC TRAINED (W1+M1, cap 240, T1@320k 2r):
  lat 645.3 s/step, 992 GLOBAL tok/s vs banked 989 (+0.3%), 181.8 GiB = 98%
  HBM (banked: 98%), RSS 582 GB/rank (incl. the shared 214G fabric mapping).
  Current tree reproduces the July GLMTP Air-2r operating point exactly ->
  stack validated, fg-fabric explanation stands, campaign comparability OK.
  Gate attempt 4 (T2@448k, cap 450) launched 03:38.
- [08-20 04:35] Air 2r gate ladder so far: 448k T2/T2B/T3 ALL HOST-C-OOM
  (arena 450, watchdog floor 50); 384k T1 G-OOM (pins 2r T1 wall (320k,384k],
  = the 1r wall; legalizes 448k T1 OOM by monotonicity); 384k T2 HOST-C-OOM.
  T2B/T3 @384k pending. If they COOM: gate outcome = "neither fits" -> doc's
  stop-and-report. Draft report: Air 2r max = 320k (T1, banked 989/992);
  fg tiers structurally host-infeasible at 2r on this 106B model (fabric
  bank+fg ~410-427G + 2x per-rank pools >> 957G pool). RECOMMENDATION for
  coordinator: 32K-step grid 160,192,224,256,288,320K (matches the 122B +
  Hunyuan row family; needs ~8-9 new cells: 1r uns@224k probe, uo@224k,
  uo@288k probe, asym@224k/288k; 2r uns@224k probe, asym@224k/288k — ~3-4h
  on c14, claimable after Flash). Banking on chain end: 384k+448k as
  asym-OOM RECORD columns (render-filtered) with the full ladder comment;
  glm4.5-air NOT added to MAIN_RUNGS until the coordinator picks the grid.
- [08-20 04:56] AIR GATE CLOSED: "neither fits". 448k T2/T2B/T3 COOM + 384k
  T1 GOOM (wall pin) / T2/T2B/T3 COOM. AIR 2R MAX = 320k (T1). Banked 384k/
  448k as record-only OOM columns in the 2r file (12-col rows, py_compile ok);
  STDTPS comment carries the full ladder + mechanism. Doc Air row updated
  with outcome + 8-cell 32K-step grid recommendation (160-320K) — awaiting
  coordinator. Air NOT in MAIN_RUNGS yet. Flash-2r chain live (f2rc384 04:57).
- [08-20 09:24] FLASH-2R 384k COMPLETE + BANKED (19-col rows, compiled):
  rc GOOM (wall tightens (320k,384k]) | fsdp2 813 b1 98% RSS387 | uns 812 b1
  68% | uo 777 b1 31% | asym T1 794 b2 98% (batch walk 2->first-fit) |
  zero3 OOM* (rc-class). Spreads 0.2-0.6%. "384k" added to flash-2r
  LEAN_DROP (lean unchanged). Flash-1r chain live 09:25 (f1t1512 on phys
  GPU3): asym 512k -> 768k (T1->T2 ladder) -> 896k; then bank 1r columns
  incl. un@512k=310 reuse + uo@896k=172 reuse + uo est 512k/768k.
- [08-20 09:4x] COORDINATION: Session B (c11) completed Agent 4 (adopted;
  my h1*/h2* chains dropped, never launched) and owns Flash-1r asym cells
  (512k TRAINED c11 07:40, 768k in flight) — my 09:25 duplicate chain was
  MY protocol miss; killed 09:3x, partial dir removed, B banks st3fl1*.
  Answered B's flag(a) in the doc: the FAILs they saw were attempt-1
  symlink breakage; attempts 3-4 show TIER presets accepted + host-watchdog
  COOMs (real). Launched stdtp_a3_airsweep.sh: g0t3r448 raw-T3 flag-closure
  probe, then the 8-cell 160-320K sweep (claimed in doc; unique 6-rung
  32K-step solution).
- [08-20 12:48] AIR COMPLETE: sweep harvested (2r asym 1379@224k b1 [b2 GOOM],
  1111@288k; 1r uo 548@224k, asym 687@224k, 554@288k; un@224k GOOM both ranks
  -> wall (192k,224k]; uo-1r@288k COOM -> wall (256k,288k]; g0 raw-T3@448k
  COOM). Banked both files (+224k/288k cols; rows 14-long; LEAN_DROP += 224k,
  288k; MAIN_RUNGS glm4.5-air = 160-320k). Rendered tp_main_combined +
  tp2r_main_combined: Air axes identical both ranks, walls visible, asym sole
  224-320k. Flash-2r complete. Remaining for Job 3: Flash-1r column (B's 3
  asym cells) + my reuse/est uo/un cells + LEAN_DROP labels + final render.
- [08-20 17:1x] JOB 3 COMPLETE. B banked the Flash-1r column at 17:02 with
  my reuse/est pieces included (un512=310, uo896=172, uo est 296/200; asym
  c11 cells 310/207/170, T1 wall (768k,896k]). Final re-render: 1r has zero
  MAIN_RUNGS warnings; 2r only 30B pending (c12). Flash axes identical
  256-896k both ranks; Air 160-320k both ranks. Doc rows + LIVE CLAIMS
  finalized; c14 GPUs offered for 30B-2r gap cells.
- [08-20 17:3x] OVERLEAF SYNC (Kevin asked): commit 4d922b8 pushed to the
  Overleaf git remote — tp_main_combined + tp2r_main_combined (tex-referenced)
  + 40 refreshed tp PDFs + new tp_glm_main_combined.pdf. 30B-2r panel partial
  until c18 banks; re-push planned on the 30B-2r watcher.
- [08-20 22:0x] CAMPAIGN COMPLETE (all lanes): 30B-2r banked by c18 21:59;
  final render zero warnings; 8/8 MoE panels identical 1r/2r axes; Overleaf
  re-pushed (main). Open for coordinator: gpt-oss-120B 9th panel (out of
  scope) axes differ per rank; tex prose/captions.
