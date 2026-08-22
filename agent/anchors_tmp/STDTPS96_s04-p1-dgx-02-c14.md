# STDTPS96 — GH200-96GB sim campaign, c14 lane (Session C14, 2026-08-21)
Doc: agent/impls/s04-p1-dgx-02-c06/standardize_tps_96gb.md. Kevin's order for
this session: Agent 3 (35B+Mixtral) -> 4 (122B+Hy+gptoss) -> 1 (30B) -> 2
(GLMs); first lane owned, later lanes only after re-reading LIVE CLAIMS.
Sim pair: phys GPU 0 (socket/NUMA 0) + phys GPU 2 (socket/NUMA 1); NVD
restriction -> inside indices 0,1. Occupiers: hbm96_occupy.py per GPU
(free -> 95.6 GiB), PIDs in the status-log header; guard whitelists them.
Tags `*-96c14`; run logs r_96_<tag>_b<b>.log; status stdtps96_status.log.
Banking target: scripts/figures/plot_tp_vs_seq_96gb.py + _2r_96gb.py ONLY.
Reuse rule: 185G cell reused iff recorded resv <= 92 GiB. Host flag:
HOST>900G on total RSS (2r = sum of ranks).

## Cell log / decisions
- [08-21] Phase 0: claim posted (after Session D's — D owns first-agent
  renderer/occupier creation; adopted). Occupiers up: 613592 (GPU0) /
  613593 (GPU2), free 95.60 GiB each. Mixtral fused build launched (CPU).
- [08-21 05:1x] Q35 Phase A: T2 320k TRAINED 2105 (85.7G=90%), 384k 2227
  (93.5G=98%, rss 342/rank), 448k 2082 (93.5G=98%, 347/rank) — resv pins at
  98% while act spill flexes to host. T2 512k GOOM -> T2 wall (448,512].
  T2B@512k FAIL (hang+watchdog ~22min into step 0; rank0 traceback truncated
  by teardown in wrapped_training_step) — Phase A2 launched: T2B retry then
  walk/bisect.
- [08-21 06:28] Q35 PHASE A DONE: 2R CAP = 512K (T2B), bracket (512,528].
  Phase-A cells banked-as-grid: T2@320/384/448 (2105/2227/2082), T2B@512
  (retry). T2B@528/544/576 HOST-C-OOM. Grid: 64K step, 192-512K. Doc row
  filled. Phase C-2r: measure rc/z3@192(+256 probe), un walk 192->till GOOM,
  fd walk 192->till GOOM, uo@192 (256=1498/46G, 384=1706/74G reused: <=92G
  rule), asym T2@192/256. Then C-1r same rungs + 1r cap confirm @512K.
- [08-21 17:1x] Q35 COMPLETE BOTH RANKS + BANKED into the _96gb scripts
  (MAIN_RUNGS rows added; compile OK). 1r: rc717/z3 724/un 721-852/fd 853/
  uo 736-1131 (all six)/asym T1 857-1256 -> T2 1354/1426/1334(99% cap).
  Walls identical to 2r. All audits pass (resv<=94.6G, host<=520+513G no
  flags). Doc row filled. Mixtral Phase A launched 17:09 (T1@128k first).
- [08-21 21:4x] MX 2r column banked (cap 144k T1; un/uo walls (128,144];
  rc/z3 (64,80]; fd+mega all-OOM; 144k SOLE across all series, asym 906).
  FIXUP QUEUE: un/uo/asym@64k b1 upgrade probes (b2 first-fits were
  96-98% edge-taxed, slower than their 80k b1 neighbors). HOST-flag ruling:
  VmHWM includes shared-fabric + mmap file-backed pages -> no flag from it
  alone; min-avail sampler to be added to later chains. 1r chain running
  (cap confirm T1@144k first).
- [08-22 00:5x] c14 WEDGED (see status log + LIVE CLAIMS). Agent-3 residual
  when hardware returns: V2 chain stdtps96_a3_mxV2.sh from the top (5 asym-1r
  rungs + 3 b1-upgrades; occupiers NOT needed — peak-audit). All banked data
  safe on disk. Then the claimed Agent-2 remainder per the plan.
