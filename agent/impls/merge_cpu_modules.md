<!-- I was working on repos on different machines. 
This repo ~/AsymGEMM-SFT/third_party/AsymGEMM is on merging various scheduler versions and is now complete and pushed to main_kevin.
During this time i was working on developing/optimizing/testing arm cpu kernels on the same repo (on a verision BEFORE the scheduler was developed): ~/AsymGEMM-SFT-46/third_party/AsymGEMM
So this other repo's code can be outdated in some aspects BUT its cpu related modules should be up to date and decently complete.
Therefore I wanna merge that repo's useful parts. mostly should be cpu kernels and ops and codepaths. it is probal very outdated in schuedeling and stuff..
So lets dive deep into the 2 repos. Explore extensivel the other repo ~/AsymGEMM-SFT-46/third_party/AsymGEMM abd reaons baout every nontrivial conflict. Read /home/kevinni/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/impls/cpu_compute.md to undersand waht it did for cpu computes. This shoud be a decently compreheslve doc that continas the an overview of the cpu modules and some metrics on its evaluations on microbenchmarking and e2e results on small and large workloads. BUT keep this midn that that rei was buulton the pre-scheduler state of this repo soo its code might be outpudate ins ome aspects.
Reason and judge very critically on how to o this merge efficiently and correctly onto this current repo status. Thnk critical we need optimizations that can help our current verison with differen tiers of scheduling.
Some aspects wil be to 1. decide what are the useful cpu features or features that repo has? among the useful ones, are they stil relevant / applicable / adoptable wiht some modifications. 
After u complete the merging we need to test that everying still correct and makes sense
STILL we DONT run exps outside containers at all. we do asym34_enroot_run this will start the right container and inside that container u cd to third_party/AsymGEMM and then run the validations.
We first validate that each merged/adopted new cpu component is correct, and that our scheduler adopt these cpu's in the most reasonabel possible way
And then similar as before lets do 3x3 vlaidation runs for e2e confirmation. list them first sequ=etnila and then execute them.
I beleive  agent/impls/previous_validation_results.md has the repvious runs; results to show that the scheduler merge (befo rehte cpu compute merge) was successful and correct (is that true?)
Dont stop until all the cpu modiels vlaidated and e2e reuslts confirmed to be the same at least if not slightly better after using these cpu modules. This is the goal and should be explicitly targeteed.

Aagain use /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/agent/impls/merge_cpu_modules.md as the record for designs of merign and the needed staged impleatojns/modicaitions needed.
This p[romt iself is in the doc jsut preserve it at the top as a comment as right now dont erase/dele this prompt. After tjeprmptl,
write the full implemation olna for mergin here before u start changing anything.
Explicitly specify the staged design include valiadtions for each stage.
Here in the doc aslo los the 9 validations runs top prove that htese cpu modueks at least des tno make our repo worse.

make the implementation plan into a staged, implementation-focused plan.

For each stage:
0. One the top of the doc make sure we have clear goals section with explicitly specified configs to run for comparison, behavior/rules for the agent (involves keep fixing and building and dont stop), and caveats to avoid
1. Explicitly explain the intended code changes clearly and include concrete pseudocode for all the needed implementations. It needs to be ultra clear and informative for a third person to follow and make the implementations.
Make sure the imoplementatino is complete and thorough. Reason about it like the actual code and amke sure it is efficient and correct.
2. Scope the stage to the exact files, functions, and classes that need to be modified. For each stage we need concrete explicit validations to prove the effectiness of this stage before moving on. 
3. Check for any ambiguity or uncertainty. Do the neeed online search extensivelt and do the needed code exploration to refine the implementation to be more concrete and correct.
4. Check for risks, unclear assumptions, missing edge cases, or uncertain behavior.
Resolve uncertainty by doing focused code exploration, small tests, related-work/code searches, or online documentation searches when needed.
5. If a risk cannot be resolved now, list it under that stage as a specific item to watch later.
Add explicit validation steps before moving to the next stage. These should include the exact command-line arguments, scripts, commands, or tests needed to confirm that the stage works. We cant rely on small toy profilng to accept changes (unkess it is reallt jsut a kenrle implementaitno that is totallt isolaated) Normaltl we need to run the e2e lora profiling to make sure the change is actually meaningful.
6. Reason about memory efficiency, lantecy, kernel launch efficiency.  NEVER break computations into small GEMMs. Don't use dumb loops to loop over experts. Dont use damn operations/inefficient kernel launching patters. 
7. Keep the plan concise and practical. Do not add extra sections unless they directly help implementation. 
8. Whenver u use subagentseahc needs to be fable5 with max effort reasoning effort.
After the doc has been finished, read it and execut it fully. Dont ask me to check over jsut go ahead. again dont stop until the goals bave been met. 
 -->

# merge_cpu_modules — CPU-kernel/ops merge (AsymGEMM-SFT-46 → this tree)

(2026-07-22. DONOR = `~/AsymGEMM-SFT-46/third_party/AsymGEMM`, the cpu_compute
campaign built on base `ead5d01` — a STRICT ANCESTOR of our `main_kevin`
(`d22440d`) ⇒ true git 3-way merge. Donor work snapshotted non-destructively
as `origin/cpu_compute_snapshot` (6b748fc, scratch-index commit; -46 working
tree untouched). Work branch: `merge_cpu` off main_kevin. Donor evidence:
`cpu_compute.md` (master table + verdicts), `placement.md` (P1–P15),
`fix_cpu_compute.md` — all three arrive WITH the merge. Full conflict
composition spec: `merge_cpu_modules_compspec.md` (appendix, applied in S1).
CONFIRMED: `previous_validation_results.md` is the accepted scheduler-merge
record; its "new" column = THIS machine's measured baseline = the reference
this campaign must meet-or-beat.)

## S0. Goals · references · rules · caveats

GOAL: land the donor's CPU stack (6 new modules + 8 ARM-SVE `_C` kernels +
hooks in 9 existing files), all default-OFF; rebuild `_C`; prove each module
correct in-container; adopt into the scheduler as ONE production flag
(`ASYM_PLACEMENT_POLICY=1` — the policy self-gates per-op from workload
numbers, so it composes with every tier); then the 9-row e2e matrix vs
`previous_validation_results.md` with the stack ON — **same-or-better is the
explicit target** (norm-recompute alone measured −3~4% step time at every
regime on the donor).

WHAT WE ADOPT (donor verdicts, re-judged for our tiers):
| Feature | Donor verdict | Our adoption |
|---|---|---|
| Placement-policy module | centerpiece, 1 flag | ADOPT — the only flag recipes carry; per-op gates self-adapt (P1 rows/bytes, P3 proj-rows, P12 always, P13 G-guard) |
| qknorm recompute (P12) | universal win −2.95/−3.57/−4.16%, C −16/−64/−32 | ADOPT everywhere (default-on under policy) |
| Restage prefetch (P13) | 32k −2.9%, dense −11.9%, 128k self-guarded neutral | ADOPT (G-guard auto-adapts per tier residency) |
| Expert wgrad deposit (P2) | MoE −2.6%/−0.4% | ADOPT MoE tiers |
| Attn wgrad deposit (P3) | 32k-class only (auto rows-gate) | ADOPT (self-gating) |
| SwiGLU fwd CPU (P1) | 32k-class MoE only (auto) | ADOPT (self-gating; cold on our blocked path — see S1 note) |
| Boundary pin+prefetch (P4), dup-copy removal (P5), fused widen (P6), wgrad 96T (P14) | never-hurts | ADOPT under policy (NB P4 engages only on async root paths — dense T2 / NVMe substrate; standard-tier roots are stock pageable) |
| Rope recompute | memory feature (≥524k tokens / dense) | ADOPT (tokens-gated) |
| SwiGLU bwd CPU, LoRA-B deposit, byte-diet (P15), save-dedup | rejected/dormant by donor | LAND CODE, stays off (P9/P15 permanent-False) |
| Batch scaling | measured-negative | not adopted (consistent with our knee cap) |

REFERENCE SET = the 9 rows of `previous_validation_results.md`, baselines =
its "new" column (measured on THIS machine, w1+m2, mrg* run dirs):
R1 q32-T1-128k-b2 1091/116.0 · R2 q32-T2-128k-b2 986/93.6 · R3 q32-T3-640k
219/129.7 · R4 llama-T1-96k 1096/48.9 · R5 llama-T2-192k-b2 548/171.1 ·
R6 llama-T2-448k 280/182.4 · R7 moe-KAdial-120k×8 2762(347.6 s/it)/165.7 ·
R8 moe-shed-800k 584/110.4 · R9 moe-shed-1.1M 385/152.9.
ACCEPT per row: tok/s ≥ baseline×0.985 AND peak HBM ≤ baseline+2 GiB
(target: tok/s ≥ baseline; C may drop). Protocols: matrix rows = w1+m2
(comparable to baseline); donor-replication A/Bs (S4) = donor protocol
w1+m4 steady=middle-2 (comparable to donor numbers). All runs in-container
(`asym34_enroot_run` → /workspace/AsymGEMM-SFT/third_party/AsymGEMM); serial,
one GPU; verdicts from jobs.tsv/artifacts never driver exit codes.

AGENT RULES: keep fixing until each gate passes; never end a stage half-done;
on failure read log tail + jobs.tsv + step_samples + config.json BEFORE code
changes; one variable per A/B; never fix a regression by disabling a measured
feature; e2e LoRA profiling is the only perf evidence (unit tests prove
wiring/parity only); every measured number → §7 ledger with run dirs; NO
driver/lib edits while any run is in flight; subagents = fable5 max effort.

CAVEATS:
- Defaults-off invariance is THE S1/S3 acceptance: policy unset ⇒ byte-identical
  behavior to current main_kevin (every donor branch env-gated; verified).
- OUR trunk survives unchanged: blocked fg fwd/bwd paths, KA managers,
  `mutable=` hints, `wait_cpu_ready_host` (get-not-pop race fix — ALL host
  reads route through it, donor's pop-based wait becomes a delegating alias),
  fused-addmm/reuse-packed-x/dgrads/async-pack, tier recipes/pins.
- KA precedence: keep-acts-resident beats every CPU-compute branch (deposits/
  cpu-act/prefetch need offloaded pinned handles); enforced via `not
  keep_acts_hbm` gates + `hasattr(manager, ...)` guards + F-0 no-op methods
  on `_HBMKeepManager`.
- Efficiency: donor's full-width `_lora_b_forward` deltas + `grouped_expert_
  lora` full-width dx are REJECTED (our chunked/blocked versions stay); no
  per-expert loops; chunking only via existing `fg_chunk_rows`/K-4 side-stream
  mechanisms.
- `_C` rebuild is REQUIRED for the 8 `cpu_*` symbols (all getattr-guarded —
  pre-rebuild the stack self-disables, post-rebuild `cpu_ops_sve_compiled()`
  must be True or kernels are stubs). Rebuild: `MAX_JOBS=8 .venv/bin/python
  setup.py build_ext --inplace` in-container (setup.py gained `-fopenmp` +
  `-march=armv8.2-a+fp16+dotprod+sve+bf16` — auto-merged).
- Known harness quirk: save_dedup + pinned_ledger suites trip an assert in
  ONE process — run suites as separate pytest processes.
- Host pool ≈957 GB budget (free's 1.69 TB is fabric-inflated); watchdog
  SIGSTOPs <35 GB free; CPU microbench only with GPU idle.

## S1 — Land the 3-way merge (resolve 5 files per the composition spec)

State: `git merge --no-commit cpu_compute_snapshot` is in progress on
`merge_cpu`; auto-merged: 6 new modules, csrc (setup.py flags included),
tests, cpu_adam (+ledger/fused-widen/deposit buffers), gc_boundary (R5
one-ahead), integrations/lf (qknorm/rope installs — passthrough until armed),
`__init__` (8 symbol re-exports), run_lf_* (telemetry + hostmem CSV),
postprocess. Conflicts (24 hunks/5 files) — resolve EXACTLY per
`merge_cpu_modules_compspec.md`; the load-bearing rules:
1. activation_offload: keep BOTH API families; donor `host_wait_cpu_ready`
   body → delegate to our `wait_cpu_ready_host`; `stage()` keeps OUR
   `mutable=` signature; keep-both `record_cpu_ready` + `stage_begin/commit`.
2. attention: pinned-ledger reserve + our `_PIN_FALLBACK_CALLS` compose;
   forward wrapper = our skip_in_backward guard FIRST then donor dedup/
   region-prefetch try/finally; dA = KA branch → deposit branch → legacy
   (deposit unreachable under KA — u_handle None); finally = our None-guards
   + `deposited_u` deferral.
3. dense fg: donor defs after ours; fused/async cpu-act branches ahead of
   OUR chunk/legacy chain with fallback host-waits = OURS; `deposit_ctx=`
   kwargs grafted; `mutable=False` on all our stage sites; +2 out-of-hunk
   KA/hasattr guards (:433 async cond, :600 R5 init).
4. qwen3_moe `_cpu_silu_mul`: donor fused branch first, fallback = our
   wait_cpu_ready_host pair.
5. qwen3_moe_finegrained: imports union; defs = ours + F-0 no-ops
   (`take_cpu_ready_event`/`host_wait_cpu_ready` on `_HBMKeepManager`) +
   donor's 17 defs; forward: BLOCKED LOOP UNTOUCHED (donor fwd features
   measured full-width-only ⇒ naturally cold on blocked path — a v2 note,
   not v1), full-width branch gets donor grafts with `not keep_acts_hbm`
   gates, 4-way act chain async→fused→our-chunk→legacy(+direct-reuse);
   backward: our blocked path + `allow_deposit=True` graft (G-D1), our
   full-width down path (donor's full-width dx REJECTED) + KA-else deposit
   graft, 5-way silu-bwd chain (K-5 CPU task → R5 commit → mech-4 →
   our-chunk → legacy), :1762 hasattr guard.
Commit the merge on `merge_cpu` when V1 passes.

V1 gates (no GPU): `ast.parse`/py_compile all touched files; per-file grep
checklists from the spec (one `def stage(` w/ mutable; `deposited_u` ×3;
`mutable=False` ×11 dense; `allow_deposit=True` ×2 fg; no `grouped_expert_
lora(` in full-width down-dx; `record_cpu_ready` only blocked/chunk paths);
in-container import of all modules; defaults-off unit sanity (each new
`_*_enabled()` False with clean env).

## S2 — `_C` rebuild + kernel/unit validation (in-container)

1. `MAX_JOBS=8 .venv/bin/python setup.py build_ext --inplace` (repo root).
2. V2 gates: `python -c "import asym_gemm; print(asym_gemm.cpu_ops_sve_compiled())"`
   → True; all 8 symbols present; unit suites EACH IN ITS OWN pytest process:
   test_cpu_ops (≤1-2 ulp parity), test_cpu_worker, test_placement_policy
   (13/13 dry-run), test_qknorm_recompute (9/9 bitwise), test_pinned_ledger,
   test_save_dedup (9/9), test_restage_prefetch, test_moe_direct_reuse
   (19/19) — all green; `ASYM_CPU_OPS_THREADS=48 .venv/bin/python
   tests/bench_modules.py --final` (GPU idle) → SwiGLU fwd ≈4.2× class,
   bwd ≈5.4×, widen 13.3×, rmsnorm ≈7× (sanity vs donor table, ±30%).

## S3 — Flags-off e2e invariance (the do-no-harm gate)

Two rows re-run with NO new env (policy unset): R2 (q32 T2 128k b2, dense +
KA path) and R8 (moe shed 800k b1, pins+dgrads path) — exact commands from
previous_validation_results run dirs. ACCEPT: in-band vs baseline (tok/s
≥×0.985, HBM ±2 GiB — expect ≈identical; this proves the merged code with
everything off is behaviorally our current tree, e2e).

## S4 — Module-level + donor-replication validation

1. SMOKE parity: donor protocol (8k b4, MAX_STEPS=6, same seed) policy ON vs
   OFF → loss curves within the 0.67-1.0% rerun envelope, no drift.
2. Donor-regime A/Bs on the merged tree (donor protocol w1+m4, steady =
   middle-2; ± `ASYM_PLACEMENT_POLICY=1 ASYM_CPU_OPS_THREADS=48`):
   a. MoE 32k b8 (donor cfg ker101-ohbm0): expect ON ≈ −10~12% step time
      (donor: 97.4→85.6 s class) + engagement markers (P1/P2/P3 traces).
   b. MoE 128k b8: expect −3~4% (norm-recompute dominant; P1/P3 auto-off
      traces prove the gates).
   c. Dense 32B 32k b8 (donor cfg ker000-ohbm8): expect −10~14% (prefetch +
      norm-recompute; P8 kill-switch traced).
   Verify per-run `placement_policy.json` sidecar decisions match P10
   acceptance sets; `cpu_worker` job_ms + deposit retention in profile.
3. Scheduler-tier probes (our recipes, w1+m2, ± policy flag): T2-MoE bundle
   @640k b1 and dense T2 @128k b2 — expect ≥0% (no harm where donor never
   measured; norm-recompute should still win) + traces showing correct
   auto-gating under KA/bundle env.
V4 accept: all above + no host-OOM (watchdog margin ≥35 GB free at peak;
hostmem CSV recorded).

## S5 — Scheduler adoption (recipes carry the ONE flag)

`asym_scheduler.py`: add to EVERY asym tier env (dense+moe T1/T2/T2B/T3)
`ASYM_PLACEMENT_POLICY=1` + `ASYM_CPU_OPS_THREADS=48` (new `_CPU_STACK` dict
merged into each TierLine env; NOT into `_MOE_PINS` — pins = measured-history
class). Rationale: the policy self-gates per-op from rows/bytes/tokens/model-
class + G-guard, so tier recipes need no per-feature flags; T1's surface is
minimal (boundary/qknorm) and safe. Regenerate `tier_recipes.sh`; docs note
in scheduler_v2.md §10′ addendum.
V5 gates: `--selftest` 5/5 + `--replay` 18/18 (env-only change — must be
untouched); TIER_DRY_RUN expansions show the 2 new vars (T2-moe: 11 env);
config.json captures them (ASYM* prefix ✓).

## S6 — THE 9-ROW MATRIX (listed first, then executed serially)

Each row = previous_validation_results config VERBATIM + `ASYM_PLACEMENT_
POLICY=1 ASYM_CPU_OPS_THREADS=48`; w1+m2; MAX_SAMPLES per row as before
(512: R3/R4/R6/R9; 1024: R5/R7/R8; tp_probe defaults R1/R2).
| # | run | baseline (tok/s · GiB) | expectation |
|---|---|---|---|
| M1 | q32 T1 128k b2 | 1091 · 116.0 | ≈ or − small (policy near-inert at T1; qknorm via lf install) |
| M2 | q32 T2 128k b2 | 986 · 93.6 | ≥ (norm-recompute + prefetch) |
| M3 | q32 T3 640k b1 | 219 · 129.7 | ≥ (norm-recompute; rope tokens-gate ON) |
| M4 | llama T1 96k b1 | 1096 · 48.9 | ≈ |
| M5 | llama T2 192k b2 | 548 · 171.1 | ≥ (dense P12+P13; P8 kills deposits) |
| M6 | llama T2 448k b1 | 280 · 182.4 | ≥, watch G-guard (97% util ⇒ prefetch self-offs) |
| M7 | moe KA-dial 120k×8 | 2762 · 165.7 | ≥ (P2 deposit + P12; P1/P3 rows-gates per traces) |
| M8 | moe shed 800k b1 | 584 · 110.4 | ≥ (P2 −0.4% + P12 −3.6% class) |
| M9 | moe shed 1.1M b1 | 385 · 152.9 | ≥ (same class) |
ACCEPT: 9/9 tok/s ≥ baseline×0.985 with TARGET ≥ baseline; HBM ≤ +2 GiB;
RSS informational (expect −10~60 GB at C-bound rows). Breach protocol =
same as the scheduler merge (rerun once → trace/config diff → one-variable
bisect: policy off vs per-feature manual flags → if a feature hurts a row,
the POLICY gate gets the fix (threshold/guard), never a silent recipe fork).
Results table → §7 ledger + appended to previous_validation_results.md.

## S7 — Close-out

Fold results into this doc's ledger + cpu_compute.md addendum (post-merge
numbers on the tiered tree); commit merge_cpu; leave main_kevin merge +
push to Kevin's explicit call (or his gbackup habit). Update memory notes.

## §7 STATUS LEDGER (append-only)

- [S0 2026-07-22] Plan written. Snapshot 6b748fc pushed; merge_cpu branch
  holds the in-progress 3-way merge (24 hunks/5 files enumerated); module
  map + composition spec produced by max-effort forks (appendix
  merge_cpu_modules_compspec.md); previous_validation_results.md confirmed
  as the baseline record.
- [S1 DONE 2026-07-22, commit 65cb647] All 24 hunks resolved per spec
  (3 files by coordinator, dense+moe fg by the spec's author fork — 2
  documented deviations, both corrections: `del packed` anchor collision
  fixed, carried_* hoisted to shared scope to avoid NameError on the
  blocked path). V1 PASS: 0 markers repo-wide; py_compile 16 files;
  in-container import green; ALL defaults-off asserted (donor + sched
  flags); spec greps exact (mutable=False ×11, allow_deposit ×2,
  record_cpu_ready 6=ours' 6, deposited_u wiring).
- [S2 UNITS 2026-07-22] pytest installed into container venv (was absent —
  donor ran suites elsewhere); ALL 8 SUITES GREEN, 59 tests: cpu_ops 6,
  cpu_worker 7, placement_policy 13/13, qknorm 9/9 bitwise, pinned_ledger 9,
  save_dedup 9, restage_prefetch 3, moe_direct_reuse 3.
- [S3 2026-07-22] Flags-off invariance: S3b MoE-shed-800k PASS (582 vs 584
  = −0.4%, peak 111.4 vs 110.4 — clean). S3a dense-T2-128k: 945 vs 986
  (−4.2%) with peak EXACT 93.6; telemetry proves zero donor features
  engaged (norm_offloads 0, worker idle, deposits idle; ledger counting
  only). Not uniform-day (S3b in-band) ⇒ decisive same-day lib A/B
  launched (.mrg_cpu_ab.sh: merged vs pre-cpu-merge main_kevin files,
  flags-off, same config). NB the 986 baseline was itself +2.9% above its
  c12 record — 945 is −1.4% vs the record.
- [S3 CLOSED 2026-07-22 — INVARIANCE PROVEN] Lib A/B (merged vs pre-cpu-
  merge main_kevin files, flags-off, same config/hour): 973 vs 974 tok/s =
  −0.11%, peak IDENTICAL 93.6. The S3a 945 sample was row noise (same
  merged code measured 973 1 h later; ±3% intra-day drift on the dense
  128k row under sustained load). With S3b (−0.4%) both invariance paths
  green. Merged lib restored + verified.
- [S4.1 SMOKE PASS 2026-07-22] loss parity ON vs OFF (moe 8k b4, 7 steps,
  fixed seed): max |Δ| = 0.48% (envelope 0.67-1.0%), no drift trend.
  Engagement verified in the ON profile: P1 cpu_act True ×336, P2 deposit
  True ×336, P3 attn deposit True ×1344, qknorm norm_offloads=672 —
  the full stack is live and lossless at SMOKE scale.
- [S4.2a MoE-32k A/B 2026-07-22] donor cfg (ker101, w1+m4 mid-2): OFF 90.95
  → ON 88.47 s/step = −2.72%, peak 32.8→32.4 GiB. Engagement: P2 ×240,
  P3 ×960 (960 worker dA jobs), qknorm 480 offloads, prefetch ×240. Donor's
  −10.7% does NOT fully transfer, measured-explained: (a) our OFF baseline
  is already 6.6% faster than the donor's OFF at the same config (the sched
  merge recovered overlapping gains); (b) P1 cpu-act is cold on our engine
  at this config (rule absent from traces — the S1 "cold path" note; v2
  item). Verdict: clean composed win, correct self-gating, no regression.
- [S4.2c dense-32k A/B 2026-07-22] donor cfg (ker000-ohbm8, w1+m4 mid-2):
  OFF 325.21 → ON 304.82 s/step = −6.27%. P8 correctly kills all dense
  CPU-compute (False decisions), qknorm 640 offloads, prefetch ×320, rope
  armed (dense rule). Peak 71.9→96.3 GiB = prefetch SPENDING free HBM by
  design (guard floor 16 GiB; 89 free here) — wall rows M5/M6 must show
  the guard self-disabling (watch item). Donor −13.8% partial-transfers
  for the same two reasons as MoE (our OFF already −14% vs donor OFF).
  S4 COMPLETE: SMOKE + both A/Bs positive, every gate behaved per design.
- [S5 DONE 2026-07-22, commit f24b1b4] _CPU_STACK
  (ASYM_PLACEMENT_POLICY=1 + ASYM_CPU_OPS_THREADS=48) merged into all 7
  family|TIER envs incl. synthetic moe|T1 (NOT into _MOE_PINS);
  tier_recipes.sh regenerated 7/7; selftest 5/5; replay 18/18;
  TIER_DRY_RUN expansion shows both vars. S6 matrix launched
  (.mrg_cpu_q3.sh, order M1→M2→M4→M5→M7→M6→M3→M8→M9 fail-fast).
- [S4.2c dense-32k A/B + S4 CLOSE 2026-07-22] OFF 325.2 → ON 304.8 s/step
  = −6.27% (donor class −10~14%; partially pre-recovered by sched merge).
  P8 correct (dense CPU-compute all-False); win = qknorm (640 offloads) +
  restage prefetch (×320). Peak 71.9→96.3 GiB: prefetch pipeline depth
  under 113 GB free — G-guard behavior at walls is exactly what M6 gates.
  S4 CLOSED: SMOKE parity + both donor-regime A/Bs green, self-gating
  correct on every trace.
- [S5 DONE 2026-07-22] _CPU_STACK dict merged into ALL TierLine envs (incl.
  q3.5 placeholders now inheriting full T2B/T3 envs); tier_recipes.sh
  regenerated — 7/7 TIER_ENV entries carry the pair; selftest 5/5, replay
  18/18. c06 system_summary S5-tense flipped to present.
- [S6 LAUNCH 2026-07-22] .mrg_cpu_q3.sh (prior-session draft) CORRECTED
  before launch: llama model key llama3_3→llama3.3 (dot form the probe
  script expects) and M7 reverted to the R7 dial env VERBATIM (mrgs1c3:
  staged+FG_KEEP_ACTS+chunk+dx only — the drafted T2-bundle env would have
  changed the comparison vs baseline 2762/165.7). Order fail-fast:
  M1,M2,M4,M5,M7,M6,M3,M8,M9.
- [S6 M1 WRECK EXPLAINED 2026-07-22 — LAUNCH COLLISION, policy exonerated]
  M1 (T1 128k) OOM had TWO trainer banners in one log: the pre-fix queue's
  in-container python survived the host-side kill (host kill ≠ container
  reap), finished loading, and trained M1's exact policy-on config to
  116.92 GiB ≈ baseline 116.0 — live proof M1 is memory-clean under the
  policy. The relaunched M1 then OOM'd against that squatter (12.21 GiB
  layer transient vs 1.08 free; foreign pid held 116.92). Sidecar confirms
  only prep-gates consulted at T1 (P5/P12/P15; P1/P2/P3/P13 absent = no
  fg surface, as designed). NO gate change, NO recipe fork; M1 re-runs
  clean after M9. PROTOCOL LESSON (binding): before any queue relaunch,
  verify GPU idle (nvidia-smi used < ~5 GiB) or pkill the driver inside
  the container — killing the host chain never reaps container pythons.
- [S6 INCIDENT 2026-07-22 ~23:13-23:50 UTC] M1 phantom-OOM root-caused:
  the PRIOR session had already launched .mrg_cpu_q3.sh; its container
  processes survived the session teardown (orphan-marked but alive) and my
  fresh launch ran CONCURRENTLY — two M1 jobs raced the GPU (116.9+66.0
  = 182.9/184 GiB in the OOM trace) and interleaved writes into the same
  log (torn: OOM/ALL-OOM lines vs HARDFAIL). NOT a policy memory
  regression: at T1-dense the policy allocates no GPU bytes (qknorm
  reduces; prefetch/deposits structurally idle). Cleanup: killed the
  orphan M1 python (119.7 GiB) + stale chains; verified exactly one queue
  instance + sole GPU occupant remains; M2 restarted fresh and healthy.
  M1 = only casualty (HARDFAIL artifact) — rerun SOLO after M9 drains.
  LESSON (adds to the container-pkill lesson): background container
  launches survive Claude session death — on session restart, ps/nvidia-smi
  sweep for orphaned queue instances BEFORE relaunching a queue script.
- [S6 M2 2026-07-22] 969 tok/s vs base 986 (−1.7%; gate 971 → 0.2pp short),
  peak BYTE-EXACT 93.6. Attribution: same-day flags-off control (cpuabm)
  ran this row at 973 ⇒ policy cost −0.4%, day-drift −1.3% (the ±3%
  dense-row drift documented in S3). Marked RETRY (with M1) after M9;
  not a policy-gate defect signature (memory exact + control-attributed).
- [S6 M4 PASS 2026-07-22] llama T1 96k b1 + policy: 1096 tok/s vs base
  1096 (−0.0%), peak 48.9 BYTE-EXACT. The clean-GPU T1+policy control —
  empirically seals the M1 collision verdict (T1+policy fully transparent).
- [S6 M5 PASS 2026-07-22] llama T2 192k b2 + policy: 545 vs 548 (−0.5%),
  peak 171.1 BYTE-EXACT — first near-wall row: free HBM 13.9 < 16 GiB ⇒
  prefetch guard held OFF by design (byte-exact peak is the proof).
- [S6 progress 2026-07-22/23] M4 PASS (1096=base, 48.9 byte-exact — T1
  policy provably harmless). M5 PASS (−0.5%, 171.1 byte-exact — prefetch
  guard self-limited at the 13.9 GB-headroom row). M2 969 = −1.7% (0.2pp
  past band, memory EXACT, engagement correct) → Q4b rerun + conditional
  rope/prefetch bisects. M7 BREACH −3.3% / +11.0 GiB (176.7): trace-
  attributed to restage prefetch — 19.3 GB headroom > the 16 GB floor →
  armed, held ~11 GB of stages, ate the KA margin. GATE FIX (one variable,
  policy-side): ASYM_PREFETCH_MIN_FREE_GB default 16→32 (all measured wins
  have ≥90 GB free — unaffected; near-wall rows now never arm). M7 rerun =
  Q4e. Deposits/P1 correctly cold at M7 (P3 False ×576; qknorm host-only).
- [S2 BUILD 2026-07-22] _C rebuilt in-container with -fopenmp + SVE-BF16
  march: cpu_ops_sve_compiled()=True, all 8 symbols present. Bench --final
  sanity vs donor table: swiglu fwd 44.3 ms@32k (donor 44.8 ✓), bwd 60.1
  (58.9 ✓), widen 82.9→5.2 (13.3×→15.8× ✓), deposits/norm/restage rows all
  same class. NOTE: venv had no pytest (donor ran suites elsewhere) —
  installed pytest into container venv, suites re-running.
