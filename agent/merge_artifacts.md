# Artifact merge plan — 4-way trees → central store (PROPOSED 2026-08-14, nothing moved yet)

Companion to the mrg4b source merge (main_kevin @14ea670). Source/docs are
fully merged and regression-validated; this plan covers the RUN ARTIFACTS
(~460G) still sitting per-tree. Model: the proven `agent/impls/<machine> ->
env/outputs/<machine>` symlink pattern, extended.

## §1 GOAL
- One repo (SFT-38) used by every machine; artifacts OUT of the repo tree,
  centrally managed under /home/kevinni/env/outputs/, visible from the repo
  via stable symlinks.
- All four trees' historical artifacts preserved and browsable from 38.
- ZERO breakage: writers and readers hardcode
  `${ASYM_DIR}/profiling_results/profiling/...` (profile_lora_lf_test_source.sh
  :3021/:3023, every harvest/plot/verdict script) — those paths must keep
  resolving unchanged.

## §2 INVENTORY (audited 08-14; all same filesystem dev=61 → mv = instant rename)
Principle (Kevin 08-14): store ACTUAL RUN RESULTS only — no checkpoints,
no test junk, no scratch. Everything below is the complete top-level
census of all four trees; every entry is KEEP (migrate), DELETE, or
EXCLUDED (rebuildable).

KEEP — ~456G total (outputs/ + results/ REINSTATED per Kevin 08-14 second
pass — "these are needed actually"; the earlier ckpt-deletion call is
RESCINDED since the ckpts live inside outputs/ and deletion is
irreversible — re-decide later if wanted, disk is the only cost):
| tree | roots | size |
|---|---|---|
| 38  | profiling_results | 13G |
| 39  | profiling_results | 94G |
| 46  | profiling_results (21G) + profiling (144M) + profiling_fixcpu (2.8G) + profiling_both_ceiling (7.4G) + profiling_both_ceiling_s04-p1-dgx-02-c18 (27G) + profiling_source_ceiling_s04-p1-dgx-02-c18 (895M) — all campaign profiling outputs, same class as profiling_results | ~59G |
| SFT | profiling_results (119G) + outputs (171G, incl. July qwen35 ckpts) + results (3.4M) | ~290G |

DELETE at migration time (test/scratch junk only):
- SFT `test_profiling_direct/` (4.4M), `test_profiling/` (16K),
  `.figtmp/` (140K)
- 39 `test_profiling_venv/` (12K)

EXCLUDED — rebuildable, verified non-artifacts: `.venv*`/`.aioenv`/`build/`
(environments), `.cache`/`.pytest_cache`/`.ruff_cache` (tool caches),
`stubs/_C.pyi` (regenerated per _C build). `Screenshot 2026-08-10*.png`
byte-identical in all four trees (38 already has it).

Staying put: `datasets/` dirs + LF `data/*.jsonl` (tiny,
registry-referenced), `agent/anchors_tmp/` ledgers+logs (small, tag-named,
belong with the repo), node-local /scratch_local caches.

## §3 KEY CONSTRAINT (why the naive per-machine symlink fails)
A static symlink cannot dispatch by hostname: the repo is ONE shared NFS
path seen identically by all machines, so `profiling_results ->
<this-machine>` is unexpressible. Per-machine WRITE dirs would need writer +
~6 reader base-path changes (Phase 2, optional). The zero-change design:
ONE shared live root + machine identity in the RUN TAG (the lib already
stamps `-c17`; generalize to `-$(hostname -s)` — 1 line in
tpfig_lib_c17.sh, readers unaffected since they glob the tag they created).

## §4 TARGET LAYOUT (rev 2, Kevin 08-14: store named `asymlora`; the
per-machine write-up dirs MOVE INSIDE it under docs/ — nothing floats at
env/outputs/ top level anymore)
```
/home/kevinni/env/outputs/
  asymlora/
    INDEX.md                          # provenance + migration record
    docs/                             # per-machine WRITE-UPS (moved from
      s04-p1-dgx-02-c06/              #   env/outputs/<machine>; reports/
      s04-p1-dgx-02-c12/              #   summaries/figures, not runs)
      s04-p1-dgx-02-c14/
    history/                          # frozen post-migration (read-only by convention)
      sft38/profiling_results/...
      sft39/profiling_results/...
      sft46/{profiling_results,profiling,profiling_fixcpu,
             profiling_both_ceiling*,profiling_source_ceiling*}/...
      sft/{profiling_results,outputs,results}/...
    live/
      profiling_results/...           # single shared write root, all machines
```
In repo 38 (three clean entries — history and docs are each ONE symlink to
the central dir, mirroring the store 1:1):
```
profiling_results -> ../../../env/outputs/asymlora/live/profiling_results
runs/
  live     -> ../../../../env/outputs/asymlora/live
  history  -> ../../../../env/outputs/asymlora/history
  docs     -> ../../../../env/outputs/asymlora/docs
```
Browse paths: runs/history/sft39/profiling_results/..., runs/live/...,
runs/docs/s04-p1-dgx-02-c14/...
Because the machine dirs move, the EXISTING agent/impls/<machine> links in
ALL FOUR trees must be re-pointed at the new location (same relative
depth): agent/impls/s04-p1-dgx-02-cNN ->
../../../../../env/outputs/asymlora/docs/s04-p1-dgx-02-cNN.
(agent/impls/throughput_prompt.md -> ../../../../../env/agent/... is
untouched — it points at env/agent, not env/outputs.)

RULE — RELATIVE LINKS ONLY, NEVER ABSOLUTE: /home/kevinni/... does not
exist inside enroot (tree mounts at /workspace/...); an absolute link
dangles there. A relative link climbs to the common layer
(/home/kevinni <-> /workspace) and resolves in BOTH worlds — the proven
agent/impls pattern. Up-count = link's own depth inside the tree + 3
(AsymGEMM -> third_party -> <tree> -> common root).

EXPLICIT SYMLINK TABLE (every link, exact target):
# repo 38 (link depth 0 -> 3 ups; inside runs/ -> 4; inside agent/impls/ -> 5)
profiling_results -> ../../../env/outputs/asymlora/live/profiling_results
runs/live         -> ../../../../env/outputs/asymlora/live
runs/history      -> ../../../../env/outputs/asymlora/history
runs/docs         -> ../../../../env/outputs/asymlora/docs
# agent/impls machine links RE-POINTED in ALL FOUR trees (dirs moved into docs/)
agent/impls/s04-p1-dgx-02-c06 -> ../../../../../env/outputs/asymlora/docs/s04-p1-dgx-02-c06
agent/impls/s04-p1-dgx-02-c12 -> ../../../../../env/outputs/asymlora/docs/s04-p1-dgx-02-c12
agent/impls/s04-p1-dgx-02-c14 -> ../../../../../env/outputs/asymlora/docs/s04-p1-dgx-02-c14
# sibling symlink-backs (each old root, depth 0 -> 3 ups)
39:  profiling_results -> ../../../env/outputs/asymlora/history/sft39/profiling_results
46:  profiling_results -> ../../../env/outputs/asymlora/history/sft46/profiling_results
46:  profiling         -> ../../../env/outputs/asymlora/history/sft46/profiling
46:  profiling_fixcpu  -> ../../../env/outputs/asymlora/history/sft46/profiling_fixcpu
46:  profiling_both_ceiling -> ../../../env/outputs/asymlora/history/sft46/profiling_both_ceiling
46:  profiling_both_ceiling_s04-p1-dgx-02-c18 -> ../../../env/outputs/asymlora/history/sft46/profiling_both_ceiling_s04-p1-dgx-02-c18
46:  profiling_source_ceiling_s04-p1-dgx-02-c18 -> ../../../env/outputs/asymlora/history/sft46/profiling_source_ceiling_s04-p1-dgx-02-c18
SFT: profiling_results -> ../../../env/outputs/asymlora/history/sft/profiling_results
SFT: outputs           -> ../../../env/outputs/asymlora/history/sft/outputs
SFT: results           -> ../../../env/outputs/asymlora/history/sft/results
38:  (own history browse only via runs/history/sft38 — live root replaces the old dir)

Post-create verification: `readlink` each link must start with ../ (no
absolute paths), then resolve each THROUGH enroot (ls inside asym_sft_45)
before declaring done.

History is namespaced by SOURCE TREE, not machine — honest provenance:
historical runs mix machines, and the machine already lives in each run
dir's tag (`-c17_`, `a1rc128-c14_`, `..._s04-p1-dgx-02-c18`). Per-machine
namespacing applies to NEW runs via the generalized tag.

## §5 NON-BREAKING MECHANICS
For every root: `mv <root> <central>/ && ln -s <relative-central> <root>`.
Same-FS rename is atomic+instant; open FDs survive; new path lookups
resolve through the symlink — even a LIVE run keeps writing correctly.
Same trick applied inside the sibling trees (their old paths become
symlinks into history/), so any legacy script pointed at a sibling tree
still works.

## §6 MIGRATION ORDER + LIVE-SESSION CAUTIONS
1. Create env/outputs/asymlora skeleton + INDEX.md; move the three
   env/outputs/<machine> dirs into asymlora/docs/ and re-point the
   agent/impls/<machine> links in ALL FOUR trees (leave a symlink-back at
   env/outputs/<machine> too, in case anything else references it).
2. Migrate 39 + 46 (idle) → history/, symlink-back in their trees.
3. Migrate 38's 13G → history/sft38; create live/ + repo symlinks. Do this
   with no cell running (rename window is ms, but don't race a writer).
4. SFT LAST — a live session is registering datasets/running smokes there
   (three registry growths observed 08-14). Symlink-back makes the move
   safe even mid-run, but prefer a quiet window. Re-run the LF
   dataset_info union sweep after it quiets (self-healing entries).
   At this step: migrate SFT profiling_results/ + outputs/ + results/ to
   history/sft/, and execute the §2 DELETE list (test/scratch junk only:
   SFT test_profiling_direct/, test_profiling/, .figtmp/; 39
   test_profiling_venv/) — Kevin 08-14.
5. Generalize the lib tag `-c17` → `-$(hostname -s)`.
6. Verify: one smoke cell writes through the new symlink; harvest globs
   still resolve; `runs/` browse links list all history.

## §7 PHASE 2 (OPTIONAL, LATER)
True per-machine write dirs (`runs/<machine>/profiling_results/...`):
writer path + base-path constants in tpfig_lib, mrg4/gptoss/fig harvests
(~6 files, one commit). Only if tag-level separation ever proves
insufficient — skip until then.

## §8 OPEN DECISIONS (Kevin)
- [ ] Green-light Phase 1 (§4-§6)?
- [x] SFT outputs/ + results/: KEEP (Kevin 08-14 second pass; earlier
      ckpt-delete rescinded — re-decide later if wanted). Test/scratch
      junk (SFT test_*/.figtmp, 39 test_profiling_venv): DELETE at §6
      step 4.
- [ ] Seed live/ fresh (recommended) or with 38's current 13G?
- [ ] Freeze history/ read-only (chmod -w) after migration?

## §Log
- [2026-08-14] Plan drafted post-mrg4c (regression ALL PASS). Nothing
  moved; awaiting the §8 calls.
- [2026-08-14] Kevin: qwen35 ckpts (171G) NOT needed → §2/§6 updated to
  DELETE at SFT-migration time; footprint to migrate drops to ~290G.
- [2026-08-14] Kevin: "actual run results only" — §2 rewritten as
  KEEP/DELETE/EXCLUDED census; SFT results/test_*/.figtmp + 39
  test_profiling_venv moved to DELETE alongside outputs/; migrate
  footprint ~285G (profiling* roots only).
- [2026-08-14] Kevin reversal: SFT outputs/ + results/ ARE needed ->
  reinstated to KEEP (migrate to history/sft/); ckpt-deletion rescinded
  (irreversibility rule); DELETE list now test/scratch junk only;
  migrate footprint back to ~456G.
- [2026-08-14] Kevin: group histories -> runs/ now has exactly three
  entries (live, history = ONE symlink to central history/, machines/);
  runs/history/<tree>/... mirrors the store 1:1.
- [2026-08-14] Kevin: runs/machines/ renamed runs/docs/ — those dirs hold
  per-machine WRITE-UPS (reports/summaries/figures), not runs.
- [2026-08-14] Kevin: store renamed asym_artifacts -> asymlora; the
  env/outputs/<machine> write-up dirs move INSIDE it as asymlora/docs/
  (nothing floats at env/outputs top level); runs/docs becomes ONE
  symlink; agent/impls/<machine> links re-pointed in all four trees
  (+ symlink-backs at the old env/outputs/<machine> paths).
