# STDTPS96 campaign — GH200-96GB sim, ALL FOUR lanes serial on c11
(2026-08-21, Kevin: run Agent 1 -> 2 -> 3 -> 4 [load-balanced split:
30B | Air+Flash | 35B+MX | 122B+HY+gptoss], don't stop until every model
has 6-rung turning-point panels at 1r AND 2r under the 96G budget.
Spec: agent/impls/s04-p1-dgx-02-c06/standardize_tps_96gb.md. This ledger
= node record; status -> stdtps96_status.log.)

## Node facts (c11)
- Simulated pair (per-socket rule): **phys GPU 0 (socket 0) + phys GPU 3
  (socket 1)** — GPU2 skipped (27.9G foreign vs GPU3's 17.1G). Containers
  see them as inside-indices 0,1 (launcher gpus arg "0,3"); 1r cells =
  phys 0 (inside 0). CVD GOTCHA: inside a container, ALWAYS use inside
  indices (the occ3 crash 02:52 was CVD=3 filtering a 1-GPU container).
- Occupiers UP 02:5x: pids 662682 (phys0, 89.4G) + 664565 (phys3, 71.0G
  auto-sized over the 17.1G foreign resident) -> free = 97,898 / 97,894
  MiB = the 96-GB GH200 budget on both. pids file:
  stdtps96_occupier.pids (guard whitelists + aborts if one dies).
- Host budget: measured 1693.8 GiB total /4 x 2 = **909.4 GB** node
  budget, BOTH rank modes (doc rule 7; flag HOST>909G on total RSS).
- Lib: stdtps96_lib.sh (occupier-aware guard; run dirs tagged -g96c11).
  Launcher: stdtps_launch.sh (same). Renderers CREATED (empty DATA):
  ~/env/figures/plot_tp_vs_seq{,_2r}_96gb.py -> out/tp96_*, tp2r96_*.
- Mixtral FUSED ckpt built on c11 02:5x (262G, 29 shards) — A3 2r ready.
- gpt-oss dequant-bf16 fused copy still ABSENT on c11 — build before A4
  (adapt gptoss120_dequant.py or port the 20b tool; M-map hard-points it).
- 122B 2r = WEIGHT-OFFLOAD sEP variant per Kevin's doc edit (find the
  dial: 185G comment "weight-offload variant's pinned copies" — locate
  its env/token before A4).

## Pending prep (next GPU-idle gap)
- Build gpt-oss-20b bf16 on c11: adapt agent/anchors_tmp/gptoss120_dequant.py
  (SRC=openai/gpt-oss-20b, DST=cache/fused/gpt-oss-20b-bf16, ~39G) — CPU-only
  but fuse-class heavy: run ONLY while no cell is measuring.
- 122B weight-offload 2r dial: research the token/env before any 122B cell.
- D-lane conditional sweep claim posted 16:0x (see spec doc LIVE CLAIMS).

## State
- [x] setup: renderers, lib, occupiers, fuse, claims, monitor
- [x] **A1 (30B) COMPLETE 21:0x**: cap 768K/T3; grid 128-768K; 2r + 1r
  columns measured+banked+rendered; sole-asym 384-768K both ranks (main
  baselines); walls rank-invariant; incidents (shm leak 05:01, static-
  occupier hole 16:5x) fixed+audited. Occupiers now DYNAMIC grow-only.
- [>] A4 SWEEP (D-lane dead 16h): gpt-oss bf16 dequant (CPU) THEN HY 2r
  cap search s96hy* RUNNING (bg b26lxn2s1; start 256K, T1->T2B->T3(ker101),
  +-32K walk, arena 320 for T2B/T3)
- [old] A1 v2 note (superseded): (stdtps96_a1_capsearch2.sh;
  v1 hit the 05:01 shm-leak guard deadlock — 410G asym_* fabric leak from
  the dead 896K runs; lib guard now auto-cleans; v2 resumes at T3@896K
  clean and re-runs the two contamination-suspect COOMs (896K T3, 768K
  T2). Trusted so far: 896K T2 GOOM + T2B GOOM (GPU-side).)
- [ ] A1 Phase B grid -> table row; Phase C 2r cells then 1r; Phase D
- [ ] A2 (Air, Flash): cap searches (priors 128-144K / 448-512K), grids,
  cells; Air likely in-ctx grid; sdp2 backend, T3-raw token, Air arena 240
- [ ] A3 (35B, MX): caps (320-384K / 112-144K), grids, cells; MX 16K-step
  likely; MX 2r = fused ckpt + shm_guard + arena 285 precedent
- [ ] A4 (122B weight-offload variant probe first, HY, gptoss): caps,
  grids, cells; build gptoss fused-bf16 first
- [ ] final: tp96_main_combined + tp2r96_main_combined render, all 8
  panels 6/6 shared axes, sole-asym tails verified, table rows filled
