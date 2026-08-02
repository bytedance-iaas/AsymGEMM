# TP-vs-seq figure campaign: mixtral-8x22b + phi3.5-moe single-rank panels
(2026-07-28, user directive: add 2 single-rank panels matching the house
combined throughput figure; "don't stop until the goal is achieved".)

## Deliverable
Two new DATA entries in /home/kevinni/env/figures/plot_tp_vs_seq.py
("mixtral-8x22b", "phi3.5-moe"), regenerated tp_vs_seq_* PDFs + tp_combined
(grid grows to 3x2), copied into agent/archive/overleaf/.../figures/.
House encoding: cells = EFFECTIVE tok/s from step_samples.csv (post-warmup,
w1+m2), best over batch; baselines recomp/unsloth/unsloth_off (superoffload_mem
backend tokens `recomp`/`unsloth`/`unsloth-off`), asym cells (tier, v[, "est"])
hatched; (v,"est") black border; "OOM" red measured wall; "OOM*" black est wall.
FSDP2/ZeRO3 rows derive automatically from recomp.

## Measured cells already banked (effective tok/s)
- mixtral 8k: uns 463, asym-T1 732 (dev pair, 1 measured step)
- mixtral 64k: T1 1823 (b2), T2 1657 (b3), T3-current 821 (b2), uns_off 1003 (b2)
  [T3 moefg0-era: 745/781 — superseded by 821 current-code]
- phi 8k: uns 1067, asym-T1 1647 (dev pair)
- phi 128k: uns_off 2873 (b3), T3 2285 (b4; b3 2273, b5 2268)
- Memory slopes GiB/1k tok (GB200 ~183 budget): MX T1 .775 T2 .573 T3 .459
  (walls ~236k/319k/399k); PHI T3 .197. MX uns_off host wall: 192k tokens
  COOM ×3 (measured at 64k·b3). PHI uns_off GPU wall: 512k tokens GOOM
  (measured at 128k·b4).

## Panel rungs (lean = all for now; LEAN_DROP decided at plot time)
MIXTRAL: 8k, 64k, 128k, 192k, 256k, 320k, 384k
  Story: race @8k/64k -> recomp+uns_off die ~128-192k -> uns dies ~256k ->
  T1 OOM @256k -> T2 carries 256k, edge 320k -> T3 320k/384k crown.
PHI: 8k, 128k, 256k, 384k, 512k, 640k, 768k
  Story: race @8k/128k -> recomp dies ~256k -> uns dies ~384-512k ->
  uns_off dies @512k (GPU) -> T1->T2 promotion ~384-512k -> T3 640k/768k crown.

## Run chains (scripts in this dir; status -> tpfig_status.log)
- mx_chain.sh   GPU0: uns 64k[b2,b1], recomp 64k[b1], uns 128k, recomp 128k,
  T1 128k, T1 192k, uns 192k, T1 256k(exp OOM), uns 256k(exp OOM), T2 256k,
  T2 320k(edge), T3 320k, T3 384k, recomp 192k(cond).
- phi_chain.sh  GPU1: uns 128k[b3,b2,b1], recomp 128k, T1 128k, T1 256k,
  uns 256k, recomp 256k(cond), T2 256k, T1 384k(exp OOM), T2 384k, uns 384k,
  uns 512k(cond), unsoff 256k, unsoff 384k, unsoff 512k(exp GOOM), T2 512k,
  T3 512k.
- solo_chain.sh after BOTH (host-heavy, serial): MX unsoff 128k, MX unsoff
  192k(exp COOM), PHI T3 640k, PHI T3 768k(crown).
Batch-walk: try batches high->low, first TRAINED wins the cell. Guards: own-GPU
idle + host `available` >= floor (mx 1000 / phi-light 600 / solo 900) + shm clean.

## Harvest -> est-cell method (house style)
Fit per-system per-token step-time t(S) = a + b*S from measured rungs; est
cells = tokens/t(S) with "est"; walls beyond probes = "OOM*". Then DATA block
comments must name walls + which cells are est (mirror q3-32b/llama entries).

## v2 RESTRUCTURE (03:55): parallel chains ABANDONED -> serial_chain.sh
Lesson 1: mixtral superoffload cells need ~870 GB host; concurrent with any
phi cell the node COOMs (mxu064-b2 COOM at 03:50 was CONTENTION, not a wall —
discarded, cell redone serially). One run at a time, GPU0, HOSTFLOOR=1300.
Lesson 2 (verdict bug): jobs.tsv `failed:1` must NOT count as TRAINED on
current code (that acceptance was for the pre-fix gate era) — phu128-b3's
"TRAINED" was a real GPU-OOM. Fixed: OOM-grep first, then `ok`-only.
phu128-b3 = genuine GOOM (banked: uns@128k b3 wall); walk resumes at b2.

## FINDINGS (04:39–09:36 blocks, all cells harvested -> tpfig_cells.json)
- **PHI 256k UNIVERSAL WALL**: every system (incl. all asym tiers) GOOMs at
  256k·b1 on the IDENTICAL alloc "Tried to allocate 61.04 GiB" = the 256k²
  sliding-window mask (window=131k; past it masking_utils materializes [S,S]).
  Also explains the 192k throughput collapse (~4000 -> ~1300 tok/s all
  systems: masked-SDPA path). Phi's seq ladder ends at 256k for EVERYONE;
  phi's asym dominance = batch capacity at 128k (T3 b5 640k-tok) + post-window
  rungs where the batch-independent mask leaves asym headroom (patch2 probes).
- **Stale-log verdict lesson #2**: run_cell appends to r_TAG_bB.log; a re-run
  of the same tag+batch inherits the old attempt's COOM/OOM lines. mxu064-b2
  "COOM 04:45" was stale (parallel-era line); its serial artifact is a clean
  3-step TRAIN (1668 tok/s, jobs.tsv ok). Corrected in harvest. New probe tags
  use fresh names (mxu064p etc.).
- Mixtral measured walls at b1: recomp (128k,192k] GOOM; uns_off (128k,192k]
  COOM; uns >=256k fits (GPU slope .617/1k -> 320k probe expect GOOM);
  T1 >=256k fits (168 GiB); T2 320k fits AT EDGE (173.8); T3 320k fits
  (140.3) / 384k COOM (suspect dataset-build transient; 352k probe decides).
- Phi measured walls at b1: recomp (128k,192k] GOOM; everything else fits
  192k, dies 256k (mask). 128k batch walls: uns b3 GOOM (best b2 4052);
  uns_off b4 GOOM-era (best b3 2873); recomp b1 3921; T1 b1 4021 (b3 probing);
  T3 b5 2268 banked.

## State
- [x] plan written  - [x] serial chain v2 done (phi 04:39, mx 08:34, solo 08:45)
- [x] patch chain done (09:36: phi T3 256/384 GOOM = mask wall; 192k rung full)
- [x] harvest v1 (41 cells)  - [>] probe_chain running (bkv7uw0vu, ~2h:
  pht1128 b3/b2, mxu320, mxt1320, mxt2352, mxt3352, 64k best-batch x5, mxo128p b2)
- [x] probe chain done 12:33  - [x] patch2 done 13:37 (ALL RUNS COMPLETE)
- [x] DATA extended (both models; every cell measured, zero est; lean = all
  rungs: mixtral 64k/128k/192k/256k/320k/352k, phi 128k/192k/224k/256k)
- [x] figures regenerated (tp_combined now 3x2 + 2 standalone panels)
- [x] overleaf synced (5 PDFs + tex comment + "six models"; NOT pushed)
- [x] recorded in model_integration.md (campaign section: headlines, the phi
  mask wall, the mixtral host-pool blowup fix path, 4 ops lessons)
CAMPAIGN COMPLETE 2026-07-28 13:4x.
