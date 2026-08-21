# STDTPS campaign — TP-figure x-axis standardization, ALL FOUR agents serial on c11
(2026-08-20, Kevin: run Agent 4 -> 3 -> 2 -> 1 on THIS box, GPUs 0+1 only,
"don't stop until all goals achieved". Handoff/spec + approved grids:
`agent/impls/s04-p1-dgx-02-c06/standardize_tps.md`. This ledger = live state.)

## Box facts (c11 = SFT-39 tree, container asym_sft_39)
- Launcher: `agent/anchors_tmp/stdtps_launch.sh <in-container-script> <gpus>`
  (non-interactive enroot start, same mounts as asym39_enroot_run).
- Lib: `stdtps_lib.sh` (= tpfig_lib_c17 port; status -> `stdtps_status.log`,
  run logs `r_<tag>_b<b>.log`, run dirs tagged `-c11`). Harvest:
  `scripts/lf/parse_fill_cell.py <dtag-dir> <ranks> <seq> <b>`.
- GPUs: 0 (empty) + 1 (~9.4 GB held by another user -> ~180 GB free). 2r runs
  use 0,1; any 2r OOM at the ~180 GB edge = flag INCONCLUSIVE, don't bank as
  a wall without noting the deficit. GPUs 2/3 off-limits (other users).
- Serial rule: ONE measuring run at a time, in-container only (memory rules).
- Weights cached here: hunyuan/GLM-Air/GLM-Flash/30B/35B/122B/gpt-oss-20b.
  Mixtral ABSENT -> `stdtps_mx_dl.sh` downloading (bg); then build fused copy
  with `mx_fuse_local.py` BETWEEN GPU cells (2r loads need fused format).
- Plot scripts live at `~/env/figures/plot_tp_vs_seq{,_2r}.py` (symlinked from
  scripts/figures; inside container = /workspace/env/figures). Render in
  container: `cd /workspace/env/figures && /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/.venv/bin/python plot_tp_vs_seq.py`.
- Out of scope: git commits, Overleaf/tex, dense figure, gpt-oss-120b panel.

## Render-path change (shared, implement once)
`MAIN_RUNGS[model] -> [6 labels]` in BOTH scripts; variant=="main" renders
exactly those columns (fallback to LEAN_DROP when model absent); all NEW seq
labels also go into LEAN_DROP so lean/complete stay unchanged. Non-fatal
warning when a MAIN_RUNGS label is missing (final render must print none).
Labels: 1024K -> "1.02M" (house style; Flash-1r banked "1024k" label kept).

## Cell-gap worklist (from DATA audit 2026-08-20; reuse everything else)
Legend: run=new measurement, probe=in-bracket OOM probe, est=interpolated
(main-dropped series only: unsloth_off; megatron disabled; derived zero3/
fsdp2 rows stay derived).
- A4 hunyuan 1r: 160k rc/un/T1(b2,1) · 224k rc-probe/un/T1 · 288k un-PROBE/T1
- A4 hunyuan 2r: 160k rc/un/T1(b2,1) · 224k rc-probe/un/T1  [sdp2, GPUs 0,1]
- A4 gpt-oss-20b: render-only (both ranks fully banked)
- A3 Air GATE first: asym-2r@448k then @384k (T1->T2->T2B->T3).
  448 fits -> grid 128-448k (1r all banked; 2r adds only these 2 cells).
  only 384 -> grid 64-384k (64k step; everything banked + the 384k gate cell).
  neither -> STOP Air, report. Air 1r: zero runs either way.
- A3 flash 1r: 512k T1 · 768k T1->T2 · 896k T1->T2 (slow: ~3-4.5h/run);
  un@512k=310 + uo@896k=172 reused from banked comment cells; uo@512k/768k est.
- A3 flash 2r: 384k rc-probe/fd-probe/un/T1(b2,1); zero3 est-if-rc-fits else OOM*.
- A2 35b 1r: ZERO runs — add 640k/768k columns from banked c18 ladder cells
  (asym T2 1528@640k, T2 1498@768k; uns OOM (wall (512k,576k]); uo est ~1120).
- A2 35b 2r: 768k T2 run. ALSO: 256k column est->measured swap (fix_plot_
  placeholders §3: rc 1620 uns 1584 uo 1498 asym-T2 2005 fsdp2 2090) — do it,
  note in table (rule 1 reuse-measured), zero3@256k stays est.
- A2 mixtral 1r [after dl; fused ckpt ok, bit-identical]: 160k rc-probe/un/T1 ·
  224k un/T1 · 288k un-PROBE/T1->T2 (T1 bracket (256k,320k])
- A2 mixtral 2r [fused REQUIRED]: 160k rc-probe/z3-probe/un(b2,1)/T1(b2,1) ·
  224k un/T1; shm_guard between fabric cells; floor 35 precedent.
- A1 30b 1r: 384k rc/un/z3/fd-probe/T2 · 512k un/T2 · 768k un-probe*/T2 ·
  896k T2->T2B · 1.02M T2->T2B  (*DATA comment says c14 uns wall (640k,660k]
  measured -> 768k OOM by monotonicity; probe anyway per handoff, expect OOM)
- A1 30b 2r [sEP asym_sepplan2-class backend, see fix_qwen3.5_tp R2]: 512k
  un/z3-probe/fd-probe/T2 · 768k T2 · 896k T2 · 1.02M T2
- A1 122b 1r [FA4 auto (qwen3.5); arena precedent 400/345]: 160k/192k/224k/
  256k: rc walk-up probes (bracket (128k,288k]) + z3 same, un x4, T1 x4
  (batch seeds: 128k asym b3, un b2; walk down)
- A1 122b 2r: 160k rc/un/z3/T1 · 224k rc-probe/z3-probe/un/T1
- 2r backends per model: hunyuan+mixtral+GLM = asym_sdp2_cpuadamwds;
  30b/35b/122b = sEP (R2 campaign backend; check fix_qwen3.5_tp PHASE R2);
  baselines same tokens as 1r + "model|2".

## Multi-session coordination (discovered 02:45-02:5x)
FOUR sessions run this campaign with ROTATED serial orders (Kevin's design
reading: 4-way split w/ serial fallback; "the very first tasks for YOU"):
- B = THIS session, c11/-39, order 4,3,2,1 (owns Agent 4)
- A = c12, AsymGEMM-SFT-tree ledgers (stdz_*), order 1,2,3,4 (owns Agent 1;
  job1 live: 122b 1r since 02:32; runs land in -39/profiling_results)
- c14/-38 (stdtp_*, STDTP_LOG.md), order 3,4,1,2 (owns Agent 3; Air gate
  live but its 448k ladder = FAILs, NOT OOMs — flagged: T2B/moe-T3 are
  config-rejected on GLM; needs raw-T3 re-ladder)
- -46 tree (stdtps46_*), node unconfirmed, prepping mixtral (owns Agent 2?);
  its 89s DL + 100s FUSE "rc=0" is not physical — flagged, verify.
PROTOCOL (posted in the doc's LIVE CLAIMS): re-read claims + owner status
log BEFORE each phase launch; skip verifiably-measured cells (artifacts,
not log lines) and credit; bank only models you measured; sweep gaps when
reaching an already-done job. MY BOUNDARY RULE: after A4 done+banked,
re-assess A3 (c14 may own it), then A2, then A1 — measure only the holes.

## FINAL (17:1x) — Session B STOPPED per Kevin
- Agent 4 DONE (hunyuan both ranks measured+banked; gpt-oss verified).
- Agent 3 Flash-1r cells DONE+BANKED by B (512k T1 310 / 768k T1 207 /
  896k T2 170; T1 wall (768k,896k]); Air + Flash-2r by c14.
- Agent 2 DONE (35b banks by B; mixtral + 35b-2r@768k by -46).
- Agent 1: 122b both ranks + 30b-1r banked by c12/-46; ONLY 30b-2r
  (512k/768k/896k/1.02M) outstanding = c12 stdz_job1_2r.sh.
- Render 17:1x: 1r main = 8/8 panels 6/6 (zero warnings); 2r main = 7/8.
- No B jobs running (monitor stopped, no GPU procs). Remaining for the
  coordinator: final 2r render after 30b-2r lands; Notes/Decision rows
  for 30B/122B/35B/Mixtral/Air/Flash (lane owners); git/overleaf.

## State (04:4x refresh)
- [x] scaffolding + smoke; MAIN_RUNGS in both scripts; gpt-oss 6/6 verified
- [x] mixtral HF snapshot on c11 (262G) — NOTE: Agent 2 now owned by the
  -46 lane (its node had weights); c11 copy = backup only
- [x] A4 hunyuan 1r MEASURED+BANKED (st4hy1*): 160k rc 1317/un 1322/T1 1311
  b1 (b2 1039 @98.7% edge-taxed, recorded) - 224k rc GOOM (wall (192k,224k])
  /un 1043/T1 1031 - 288k un GOOM (SOLE rung; handoff probe resolved)/T1 844.
  Main panel verified 6/6, lean unchanged.
- [x] 35b bank-only work (1r 640k/768k cols; 2r 256k est->measured)
- [>] A4 hunyuan 2r chain RUNNING (st4hy2*: rc/un/uo-probe/T1-b2/T1-b1 @160k
  + rc-probe/un/T1 @224k; GPUs 0+1; GPU1 has ~9.4G foreign -> edge GOOMs
  get the deficit note)
- [ ] A4: bank 2r, render check, fill hunyuan+gptoss table rows
- [~] A3 = c14 lane (gate re-laddered honestly after my flag: 448k T2B/T3
  COOM -> grid B likely; T1@384k running). SWEEP if they stall/mis-bank.
- [~] A2 = -46 lane (mixtral 1r live: rc@160k GOOM, un running). Watch for
  35b-2r@768k — sweep it if their scope misses it.
- [~] A1 = c12/A lane (122b 1r: 160k/192k done, 224k running; then 30b).
  My staged worklist stays as their cross-check.
- [ ] final: both mains re-render, per-panel verify (identical axes, walls
  visible, sole-asym rungs per table), fill Notes/Decision in handoff table
