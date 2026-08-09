# merge_validation — Phase 3-ext handoff (X1–X8), 2026-08-02

## Context (30 sec)
`main_kevin@5fdc327` = validated union merge of AsymGEMM-SFT{-39,-46,,-38} (code + LF-side + docs; trail: `agent/impls/merge_progress.md`). Already validated 5/5 near-capacity, NO regression: q3-30b (T2 120k·b8, T2B 1.1M·b1), q3.5-35b (T2 896k·b1), glm4.5-air (T3 128k·b3), glm4.7-flash (T3 192k·b5). **Goal now: 4 more models × 2 near-capacity cells vs recorded numbers — throughput (tok/s), latency (s/it), peak HBM, RSS. A front-runner session is ALREADY processing X1→X2→…→X8. YOU work the OPPOSITE direction: X8→X7→X6→… (table below is already in YOUR order). Before each cell, check `agent/anchors_tmp/tpfig_status.log`: if that X-id already has a `CELL … ->` verdict or a CLAIM/START from the other runner, SKIP it — meet in the middle, zero duplicated cells.**

## The cells (exact commands + refs)
Run inside container `asym_sft_42` (host: `asym42_enroot_run`; repo at `/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM`). All via:
```bash
export GPU=0 HOSTFLOOR=1300
. agent/anchors_tmp/tpfig_lib.sh          # sets MAX_SAMPLES=512 w1+m2, jitter0, HF_HOME, guard, run_cell
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
```
| id | run_cell (USE A FRESH TAG per attempt: e.g. bx8a, bx8b…) | reference (tok/s · peak GiB · RSS · other) |
|---|---|---|
| X8 | `run_cell <tag> q3.5-122b-a10b "asym_cpuadamwds|T2" 480000 "1"` | 852 · 177.8 (96% EDGE) · 849 · 563.6 s/it (c18; fragile-edge caveat on record) |
| X7 | `run_cell <tag> q3.5-122b-a10b "asym_cpuadamwds|T2" 448000 "1"` | 861 · 171.3 (93%) · 846 · 520.2 s/it (c18 → cross-node band) |
| X6 | `run_cell <tag> mixtral-8x22b "$T3TOK" 64000 "2"` | 58.7 · 2534 compute · 908 (anchor at3b2, c14) |
| X5 | `run_cell <tag> mixtral-8x22b "asym_cpuadamwds|T2" 320000 "1"` | 670 compute / 685 wall · 173.8 · 882 (c14 artifacts mxt2320) |
| X4 | `run_cell <tag> glm4.7-flash "asym_cpuadamwds|T1" 192000 "2"` | 812 compute / 845 wall · 93.5 · 273 (c14 artifacts f1s_t1192) |
| X3 | `run_cell <tag> glm4.7-flash "$T3TOK" 192000 "5"` | 158.8 · 723 · losses 1.322/1.226/1.235 (c14; replication V3: 157.9·719) |
| X2 | `run_cell <tag> llama3.3-70b "asym_cpuadamwds|T2" 448000 "1"` | 275–280 · **182.4 exact-line (97% wall)** · 976–983 |
| X1 | `MAX_SAMPLES=1024 run_cell <tag> llama3.3-70b "asym_cpuadamwds|T2" 192000 "2"` | 545–548 · **171.1 exact-line** · 963–982 |
Datasets all staged+registered (if "missing registration": `python3 /home/kevinni/AsymGEMM-SFT-39/.repair_dataset_info.py`). 122b ≥512k datasets don't exist on this host — 480k is the deepest cell.
**Front-runner's tags in the status log map to X-ids as**: x1llt2=X1 · x2llwall=X2 · x3f47t3=X3 · x4f47t1=X4 · x5mxt2=X5 · x6mxt3=X6 · x7q122=X7 · x8q122=X8 (its per-cell completion markers are `MRG-X<i>-DONE`). Treat a START with any of these tags as that X-id being claimed.

## If you are on ANOTHER MACHINE (not s04-p1-dgx-02-c14)
Repo, datasets, .venv, HF cache and the status log are shared NFS — reusable as-is. The enroot container is NODE-LOCAL: create your own instance of `asym_sft_42` from `/home/kevinni/qian-sglang-backup.sqsh` with the same mounts (pattern: the mixtral-anchors ops note, second-instance recipe in model_integration.md). SOLO rule then applies per-node (you and c14 can run concurrently). CAVEAT: X1–X6 refs are c14 measurements — on another node grade with the cross-node band protocol (like X7/X8) and say so in the verdict.

## Grading (the records' own protocol)
Extract: `.venv/bin/python agent/anchors_tmp/mrg_metrics.py "<tag>-c14_<model-slug>__b<b>_s<seq>_ga1_drop000" <tokens_per_step>` (tokens = seq×batch; model-slug = shorthand with `.`→`_`). Latency = mean_step_s. Losses: grep `"loss"` in the run's trainer_log.jsonl.
- tok/s: PASS if ≥ ref−1.5% (same-node refs); c18 refs (X7/X8): band judgment, note environment.
- peak HBM: ±2 GiB (±3 near-wall). If reserved is high but ref is an "exact-line", also check ALLOCATED (`training_step_global_peak_allocated_after_bytes`) — reserved deltas from prefetch/pool cache are the record's documented non-capacity class.
- RSS: informational (anchor-grade, not a gate). Losses (X3): ≤1%/step.
- `failed:1` in jobs.tsv with COMPLETE artifacts (all steps in step_samples.json + profile.json) = the recorded teardown flake ⇒ TRAINED (q3.5 esp.).

## HARD ops rules (from the records — violations invalidate numbers)
1. **SOLO node**: never two cells at once, even on different GPUs (host contention skewed tok/s up to −44%). Before starting: no compute apps on ANY GPU AND the last START in `agent/anchors_tmp/tpfig_status.log` has a matching `CELL … ->` verdict. Coordinate through that log (it is on shared NFS, visible from every node): APPEND `CLAIM X<i> <who> <time>` BEFORE starting a cell, and skip any X-id that already has a CLAIM, START, or `CELL … ->` verdict. Front-runner ascends X1→; you descend X8→; the first side to reach a claimed cell stops — done when all 8 have verdicts. Duplicate cells are wasted GPU-hours, not errors.
2. Guard timeout is 60 min — after a long cell, re-source and retry rather than assuming failure.
3. Fresh tag per attempt (stale-verdict pitfall). Never edit driver/runner scripts while any run is live.
4. Build/tests ONLY with `.venv/bin/python` (system python has a different torch — breaks fg kernels).
5. On OOM: run_cell walks batches down; a G-OOM at the listed batch for X2/X8 edge cells = breach protocol (reproduce once with a fresh tag, then env-parity check) before calling regression.

## If a cell breaches band
Reproduce once (fresh tag). Then A/B the first suspects in order: `ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE=0` (merged default flip), placement-policy off (`ASYM_PLACEMENT_POLICY=0` — but then compare vs the PRE-CPU-stack ref column), and inspect the profile's `placement_policy` block for which rules fired vs CPU-matrix expectations. Record verdicts + evidence in merge_progress.md (Phase 3-ext table).
