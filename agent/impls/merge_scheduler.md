# merge_scheduler — sched40 vs sched42: differences + the merge plan
(IMPLEMENTATION: `agent/impls/fix_merge_scheduler.md` — the staged build
plan with per-stage gates. This doc stays the decision/evidence record.)
(2026-07-20. Repos: THIS tree `AsymGEMM-SFT/third_party/AsymGEMM` = backup
`origin/main_kevin_sched40` (187cea7, 16:02) — the c12/dense-lineage session;
`AsymGEMM-SFT-39/third_party/AsymGEMM` = backup `origin/main_kevin_sched42`
(0829487, 16:07) — the c14/MoE-lineage session. Diff: 66 files, +3753/−1408.
NB: both LIVE trees have drifted past their backups — 40's tree has the
post-16:02 tp_probe artifacts-complete fix + today's W6/W7+X phase results;
-39's tree shows 60 dirty files vs its backup. Merge from live trees.)

RE-VERIFIED 2026-07-20 (2nd session; full live-tree diff + hunk-level code
audit): every checkable claim held EXCEPT the corrections noted below —
backup diffstat exact (66/+3753/−1408);
the six training files are the only source divergence; zero csrc; noclone
absent in 42; §1A matches asym_scheduler.py line-by-line (rungs/knee/anchors/
pins/selftest); ceiling (mean-of-2 vs ≥3-only), builder (340-line refactor),
ladder (MAX_STEPS 2v4) deltas as ruled. CORRECTED by the audit: 42 has FIVE
new flags, not four — the 1st pass missed `ASYM_SAVED_TENSOR_ASYNC_PACK` and
misattributed its two files to a "GC save-on-cpu override" (§1B); tp_probe +
driver bullets fixed in §1C; §1D symlink correction; newly found items: §1E.
Later passes re-sourced the cross-family 64k evidence (§2d′/§4 — the "+29%"
was unsourced), re-sourced §2c's drift examples, scoped the class split to
dense (§3 step 4), and fixed +116/−52, the 1.6M location (§1D), and the
prompt.md citation (§1E).
All five flags default-OFF and every 42 hunk is env-gated (flags-off paths
audited equivalent). SCOPE RULING (Kevin, 2026-07-20): the merge is
CODE-ONLY — datasets/, profiling_results/, archive/* run artifacts are OUT
of scope (leave in -39); doc-file reconciliation stays (§1E).
ADDED 2026-07-21 (Kevin): §3 step 7 — `backend|TIER` recipe/preset layer
(backend never auto-selected) + the config-label naming gap it must fix;
validation gate renumbered to step 8 with new item (e).

## 0. Outcome summary (what the merged tree IS, plain terms)
- SOURCE: 40's tree + 42's FIVE default-OFF features (attn keep-acts, fused
  LoRA addmm, MoE packed-X reuse, panel cache, saved-tensor async-pack D2H);
  40's noclone + `mutable=` kept in source, unwired. Every 42 hunk is
  env-gated (audited) ⇒ library behavior identical to 40 unless a recipe
  opts in. No kernel/csrc changes anywhere.
- SCHEDULING: 40's decision system (3 tiers T1/T2/T3, bytes-only feasibility,
  GPU AND host caps, probe near walls, 4-model tables) carrying 42's content:
  per-rung slope decomposition (+sum-check), knee batch B*=min(max-fit,
  ⌈N*/s⌉), >800k superlinear keep-acts patch, 5-property selftest. 42's
  τ-fits / water-fill / cross-family demoted to offline `--predict`.
- SCRIPTS: all 40's tooling unchanged; ONE new CLI (asym_scheduler.py,
  refactored per §3 step 4); ONE driver line swapped (:3844 → 42's attn-KA
  forwarding); a `backend|TIER` preset layer in the driver (§3 step 7)
  replaces hand-composed flag sets — backend never auto-picked, names stay
  full-fidelity.
- PERF: dense unchanged (no-regression gate (c)); MoE recipes = the
  archived c14 configurations verbatim (6 base pins incl. KEEP_DGRADS_HBM; T2 adds the KA bundle
  incl. attn-KA); parked upside behind named A/Bs: panel-cache,
  fused-addmm, reuse-packed-x (NONE measured in any c14 run — archive
  audit 2026-07-21), async-pack.

## 1. Difference inventory

### Meaningful
**A. Scheduler design (the core divergence — two different formulations)**
- 42 adds `scripts/lf/asym_scheduler.py` (271 lines): a WATER-FILL ALLOCATOR.
  HBM = endowment; buys units (batch +1, or residency "rungs") by highest
  marginal Δtok/s per GiB, using fitted per-family throughput lines
  τ(s)=a+b·s and per-rung Δτ/Δmem. Picks across FAMILIES (asym / sup-unsloth /
  sup-recomp). 4 asym rungs: staged(−70us/tok,+2GiB) → ker000(−34, .0375/ktok)
  → keep-acts(−33, .052/ktok) → panel-cache(−3,+6GiB). Extras: short-seq
  ANCHOR table (measured big-batch regime, fits invalid <160k), knee law
  (N*≈400k tokens: batch return collapses above), superlinear keep-acts slope
  past 800k (measured 900k calibration), edge-penalty {92%:0,95%:1%,98%:4%},
  BEND=4 GiB near-wall over-prediction allowance, safety dial (2/5/8%
  headroom), and a 5-property SELF-TEST (nested shedding, monotone tok/s,
  reserved-sweep nestedness, analytic-boundary consistency, safety
  monotonicity). Constants: q3-30b-a3b ONLY.
- 40's design (agent/impls/s04-p1-dgx-02-c12/system_summary.md, consolidated):
  FEASIBILITY THRESHOLDING. t* = first tier T1→T2→T3 with HBM_t(B·s) ≤
  0.92·C_HBM AND HOST_t ≤ C_host; byte lines only, NO timing inputs (τ-ladder
  used once, offline, to prove tier ordering); probe rule near walls (don't
  trust the line within 8%); multi-model constants (q32, llama, q3-30b,
  q3.5-35b); hardware enters only via (C_HBM, C_host); HOST constraint proven
  required by measured tier inversions (llama T3 host-walls at 416k while T2
  runs 448k). No scheduler *code* — formulas + hand application.
- Both docs evolved separately: 42's scheduler_v2.md = §8 v3 "regimes as
  outputs" + §9 v3.1 "one scalar dial β over families AND modes" (β = tok/s
  per GiB shadow price; the code then dropped the user knob → water-fill);
  40's scheduler_v2.md = §9 φ-scheduler + the consolidated system_summary.
  42 also adds agent/impls/remaining_optimizations.md (94 lines).

**B. asym_gemm source (42 has NEW perf features; 40 has small unique bits)**
- 42-only features (FIVE, all default-OFF; hunk-audit 2026-07-20 confirms
  every 42 delta is env-gated, flags-off paths equivalent):
  1. attention keep-acts (`ASYMM_ATTN_ACT_KEEP_ACTS_HBM`,
     attention_activation_offload +116/−52 — keeps U/S on GPU AND flips the
     LoRA-A fwd from the CPU-left kernel to a GPU GEMM on the kept source);
  2. FUSED LoRA ADDMM (`ASYMM_FUSED_LORA_ADDMM`, dense_mlp + attention —
     numerics-touching when ON per its own docstring: alpha in fp32 accum);
  3. packed-X reuse (`ASYMM_QWEN3_MOE_FG_REUSE_PACKED_X`, qwen3_moe);
  4. weight PANEL CACHE (`ASYM_W_PANEL_CACHE_GB`, frozen_linear — LRU keyed
     by data_ptr, passthrough when unset/0);
  5. saved-tensor ASYNC-PACK D2H (`ASYM_SAVED_TENSOR_ASYNC_PACK`,
     activation_offload + decoder_activation_offload — side-stream D2H,
     fix_asym S-mem(c); MISSED by the 1st-pass inventory, which wrongly
     attributed these two files to a "GC save-on-cpu override". That env
     (`ASYM_GC_SAVE_ON_CPU_OVERRIDE`) is a PRE-EXISTING shared DRIVER knob
     (both trees, fix_gb200_ep F1) that 42's keep-acts rung merely sets —
     no new training-file code. Do not confuse ASYNC_PACK (42-new) with
     ASYNC_UNPACK (pre-existing both trees).)
- 40-only: `ASYMM_FG_KEEP_STAGE_NOCLONE` + `mutable=` plumbing (A/B'd NULL —
  kept unwired per §2d′), stage()-site read-only annotations. 42 does NOT
  have these (0 hits) → the training files are a REAL two-sided merge zone.
- SCOPE CHECK (verified by full-tree diff): these six `asym_gemm/training/*.py`
  files are the ONLY genuine source-code divergence. Zero CUDA/C++/kernel
  diffs, zero csrc, zero integrations/profiling changes — the compiled engine
  is identical lineage on both sides. Everything else is scripts/docs/figures.
- MEASUREMENT-EMBEDDING — CORRECTED BY ARCHIVE AUDIT (2026-07-21, supersedes
  the earlier claim): 42's scheduler CODE pins `ASYMM_FUSED_LORA_ADDMM=1` +
  `ASYMM_QWEN3_MOE_FG_REUSE_PACKED_X=1` (ASYM_PINS) — but grep over EVERY
  archived c14 command.txt (tputsched 900k, tputasl 800k, tputschedb 1.1M,
  tputasm 1.4/1.6M, + all others) shows ZERO runs with either flag, and none
  with panel-cache. The ASYM_PINS were aspirational (for future emissions),
  never measured. ⇒ the c14 record embeds exactly SIX base pins
  (LORA_A_FWD_GPU=1, DA_GPU=1, DOWN_SCATTER=0, CHUNK_MB=1024, DX_STAGED=1,
  KEEP_DGRADS_HBM=1 — the sixth found 2026-07-21 via the C4b breach diff;
  it pre-exists in both trees so no inventory tracked it; set in ALL deep
  c14 runs, NOT in the 120k dial runs),
  and the 900k bundle adds MoE-KA + attn-KA + GC-save-hbm. RULING (revised):
  ALL FIVE new source features are default-off EVERYWHERE — none rides any
  recipe; attn-KA appears in the T2-MoE recipe ONLY (it IS in the measured
  900k bundle). fused-addmm/reuse-packed-x/panel-cache/async-pack: unwired,
  each pending its named promotion test (§2d′). This SIMPLIFIES the merge:
  no feature is load-bearing for any measured constant.
  Dense cross-check from 42's OWN record (fix_asym ledger, q3-32b 128k
  tputask ladder): fg + attn-KA + GC-save-HBM peaked at 1058 tok/s vs T1
  1104 / sup-unsloth 1110 ⇒ the new features do NOT change any dense pick
  (T1 still wins short-seq dense); the dense quarantine costs nothing.

**C. Tooling**
- tp_probe.sh: 40-ONLY (verified: -39 has NO tp_probe.sh at all, never in its
  git history — the "pre-hardening" bugs that bit twice today lived in 42's
  old ceiling/ladder verdict flow, not a probe file). Nothing to take from
  42. NB tp_probe.sh is UNTRACKED in 40 — `git add` it on the merge branch.
- ceiling_search.py: 40 has the w1+m2 mean-of-2 steady fallback; 42 reverted
  to middle-drop-only (returns NULL with 2 measured steps — wrong under the
  new protocol). TAKE 40.
- build_lf_sft_eval_pair.py: 42 TOOK 40's parallel rewrite and refactored it
  (chunk→chunk_rows, split `_concat_split_fast`/`_concat_split_legacy`).
  RULING (revised): KEEP 40's — it carries the byte-identity A/B proof; 42's
  refactor has none, and its fast/legacy split doubles the surface that must
  stay byte-identical, for no demonstrated benefit. Revisit only if 42's
  passes the harness AND shows a concrete win.
- driver (profile_lora_lf_test_source.sh): ONE line each way at :3844 and it
  is LOAD-BEARING, not cosmetic — 40 forwards ASYMM_FG_KEEP_STAGE_NOCLONE
  into the LF config; 42 forwards ASYMM_ATTN_ACT_KEEP_ACTS_HBM. TAKE 42's
  line (without it attn keep-acts never reaches the training process and the
  T2-MoE recipe cannot reproduce c14); dropping 40's line IS the noclone
  unwiring (§2d′). run_dial_ladder.sh: MAX_STEPS 2 (40) vs 4 (42) — KEEP
  40's (matches the w1+m2 mean-of-2 protocol; 42's 4-step belongs to the
  retired middle-drop). profile_lora_lf_test_both.sh: identical — no action.

**D. Results records** — disjoint truths, union them: 42 has the c14 MoE
record (`s04-p1-dgx-02-c14_old/*`, test_throughpout_v2 = the crossover
campaign; the 1.6M headline lives in the LIVE shared copy — the _old
snapshot predates it) and DELETED the c12_old docs; 40 has the c12 dense/llama
record + today's W6/W7+X phases (test_throughput_results.md). No conflict —
different dirs.
CORRECTION (verified): the LIVE c12/c14 dirs are symlinks in BOTH trees to
the SAME physical `/home/kevinni/env/outputs/{c12,c14}` — the live records
are already unified, nothing to union there; only the per-tree archive
snapshots differ (40: `impls/archive/s04-p1-dgx-02-c12_old`, -39:
`.../c14_old` → copy -39's c14_old over).

### Trivial
Figure archive moves (scripts/archive/figures/** = old figure infra + rendered
PDFs/PNGs), screenshot adds/removals, overleaf pointer file. Cosmetic; take
42's archive layout (matches the cleanup direction).

### E. Uninventoried divergences (2026-07-20 re-verification) + rulings
- `agent/impls/fix_asym.md` DIVERGED (240 diff lines): 40's is NEWER and
  RETRACTS 42's U-offload theory (nsys volume analysis — 13 TB/step D2D =
  `_HBMKeepManager.stage()` clone churn; "'never read back' was wrong"; this
  is the measurement behind 40's noclone flag). 42-unique: "§5a PHASE
  CLOSE-OUT" + post-fork STATUS LEDGER entries (async-pack/unpack NULLs,
  the q3-32b dense tputask ladder, fused-addmm C2 validation). RULING: 40's
  body + graft ALL of 42's unique record sections (§5a + its ledger tail).
  S2/S0' sections exist in BOTH copies, so 42's code comments citing them
  stay resolvable either way.
- `agent/impls/remaining_optimizations.md` (42-only, 94 lines): carry over
  as-is (optimization backlog record).
- `agent/handoffs/prompt.md` fully diverged (80 vs 233 lines): -39's IS the
  "prompt.md v2" formulation doc that asym_scheduler.py's own header cites
  as its source. Do NOT clobber — preserve -39's copy alongside (rename,
  e.g. prompt_v2_c14.md).
- `stubs/_C.pyi`: 40 strict superset (einsum/fp8_einsum/transform_sf stubs;
  csrc byte-identical ⇒ stub drift only). KEEP 40's.
- `agent/reports/`: -39 appends a "two scheduling levers" close to
  midterm_memory.md and adds midterm.md + figures/. Union -39's in.
- OUT OF SCOPE (Kevin: code-only): -39-only raw artifacts —
  profiling_results/profiling_both (149 files, c14 raw evidence), datasets
  (incl. fastverify — zero script refs), archive/profiling_ceiling_*. Leave
  them in -39.
- Noise: `_C.so` binary + venv editable metadata (rebuild artifacts),
  .claude settings, caches — ignore.

## 2. Design analysis (meticulous)
| dimension | 40 threshold | 42 water-fill | verdict |
|---|---|---|---|
| decision inputs | bytes + 2 capacities | fitted τ (us/tok) + Δτ per rung | **40** — τ fits are machine/day-flavored; Kevin ruled AGAINST µs-as-budget and FOR hardware-agnostic thresholding (2026-07-20). 42's τ is also q3-30b-only. |
| tok/s prediction | explicitly NOT predicted | predicted (validated −3% @900k) | keep 42's predictor as an OFFLINE/validation tool, not the decision |
| granularity | 3 coarse tiers | 4 rungs w/ per-rung Δmem | **42** — richer AND compatible: tiers = rung prefixes; a tier's byte-line = base + Σ admitted-rung slopes. Unify. (Prefix mapping holds only AFTER the §2b class split — as-shipped, 42's bundled keep-acts skips T2.) |
| batch | manual / "capacity-only" (valid long-seq only) | knee law + short-seq anchors | **42's insight in 40's form**: B* = min(max feasible B, ceil(N*/s)) — the knee as a hardware-agnostic THRESHOLD constant (N*≈400k tok for q3-30b; per-model). Anchors stay as the honest short-seq regime table. |
| near-wall | probe rule (measure) | BEND + edge-penalty (model) | **40 primary** (probe-don't-predict is battle-proven: llama T2 416k); keep 42's edge table as the prior that TRIGGERS probing. |
| host RAM | second constraint (proven by tier inversions) | ABSENT (c14 never host-OOM'd) | **40** — mandatory; extend 42's rungs with per-rung host slopes. |
| cross-family scheduling | asym-only (T1 ≡ baseline parity) | picks among asym/sup-uns/sup-rc | **40** — parity makes it moot, and it needs per-machine baseline τ fits. Drop (see §4 for the one exception). |
| validation harness | measured-record consistency | 5-property SELF-TEST | **42** — port the self-test to the merged rule (nestedness/monotonicity survive the reformulation). |
| model coverage | 4 models | 1 model | **40's** constant tables. |
| ultra-long physics | linear lines | superlinear keep-acts >800k (measured) | **42** — fold the piecewise slope into the byte lines. |

**VERDICT: 40 is the better SYSTEM (converge on it); 42 contributes data,
decomposition, one constant, one correction, one test harness, and quarantined
features — not its architecture.** 42's water-fill + τ-prediction survives
only as the offline `--predict` reporter, never the runtime decision.

### 2b. The keep-acts difference (the subtle trap — read before merging)
- 40's T2 splits the two activation byte classes: attention saved-tensors →
  host, MLP acts kept (GC-boundary saves travel with the attention class at
  T2). Justified by the 3× price gap (attention 6.2 vs MLP 20.6 us/tok/GiB)
  and by results (llama 448k, q35 +35% both ran T2).
- 42's "keep-acts" rung BUNDLES them: its env sets all three —
  `ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1` + `ASYMM_ATTN_ACT_KEEP_ACTS_HBM=1`
  + `ASYM_GC_SAVE_ON_CPU_OVERRIDE=false` — keeping MLP acts, attention
  tensors AND GC-boundary saves on GPU — so 42 EMITS no middle memory
  position (its scheduler_v2 §3b dial ladder did measure the split state
  once, 120k×8 — see §3 step 4); it is coarse exactly where the price ladder
  says
  granularity pays most. (Both sides' keep flags are global/all-layers — the difference is
  byte-CLASS coverage, not layer coverage.)
- Composing the flags the other way (MLP shed, attention kept) would be a 4th
  mode — REJECTED: it offloads the expensive class while keeping the cheap
  one; price-dominated under convexity, never optimal.
⇒ MERGE RULE: exactly 40's THREE modes, tokens and flags unchanged. 42's
extra KEEP-ACTS flags (attn-KA + GC-save override) appear in ONE place only:
the T2-MoE recipe, to reproduce c14's measured keep-acts configuration (the
FIVE archive-verified base pins ride all MoE asym recipes per §1B;
fused-addmm/reuse-packed-x ride NOTHING — archive audit 2026-07-21). No new
modes. No re-bundling.

### 2c. The line fits (who predicts better)
Same linear form both sides; different content. 40: memory-only, per
(model, tier), 4 models, but a single lumped line that drifts at extremes
(+4.4% under @llama-384k, +1.2% under @q32-448k (162→164.2, c12 §4) — the
probe rule is what saves it; two earlier drift figures here were unsourced
and are removed). 42: memory decomposed PER BYTE CLASS with a built-in
falsification check (rung slopes must sum to the tier slope — they do, ±bend)
plus the measured piecewise correction >800k — scientifically stronger, but
one model only; its throughput τ-fits are the weak part (machine-flavored;
42's own code bypasses them at short seq via anchors).
⇒ MERGED FIT = 42's decomposed FORM × 40's multi-model DATA, with per-fit
validity ranges and the piecewise patch; τ-fits offline-only. Strictly more
accurate and more defensible than either alone, at identical math complexity.

### 2d′. Effect classification (no-effect ≠ worse-effect — ruled per item)
| item | measured sign | context measured | ruling |
|---|---|---|---|
| 42 verdict tooling (pre-hardening ceiling/ladder flow; 42 never had a tp_probe) | **WORSE** (fake-FIT, null-steady — bit twice) | today, live | drop |
| BEND/edge-penalty as decision input | **WORSE** (would have rejected llama T2 448k = measured FIT) | c12 walls | decision: drop; offline predictor: keep |
| water-fill + τ-fits | NOT worse (−3% pred @900k) — architecturally idle (deep end converges; short end uses anchors) | c14 | demote to offline `--predict`, don't delete |
| noclone (40's) | **NULL in tested context** (q32 128k KA) | one point | keep code UNWIRED — concretely: `_keep_stage_noclone_enabled()` + `mutable=` annotations stay in source; the driver LF-forwarding line is dropped (:3844 resolves to 42's attn-KA line, §1C); no recipe emits it |
| async-pack D2H (42's, missed by 1st pass) | **NULL** (tputask5: 1056 vs 1058 tok/s, +0.8 us/tok; its unpack twin also NULL) | q3-32b 128k dense, GC-save-HBM b2 (42's fix_asym ledger) | keep code UNWIRED — 42's own ruling ("harmless, flag-gated, may matter under future prefetch"); neither pinned nor a rung ⇒ NOT embedded in c14 numbers |
| panel-cache (42's) | **SMALL POSITIVE** (−3 us/tok ≈ 0.2%) | c14 120k×8 MoE | quarantine + PROMOTION TEST: at SHORT seq the staging tax is proportionally larger (llama 8k: staging ~0.16s est. vs ~6-8s step ⇒ 2-3% hypothesis) and headroom is plentiful — A/B dense short-seq T1 ± cache before judging |
| fused-addmm, reuse-packed-x (42's) | **UNMEASURED on MoE** (archive audit 2026-07-21: in NO archived c14 run — the "embedded in the bundle" claim was false; 42's ASYM_PINS never ran). Dense: fused-addmm has one positive ledger point (C2 step-1, q3-32b 128k) | c14 archives + 42's ledger | default-off EVERYWHERE, in no recipe; promotion test = 2×2 A/B at MoE T2 (and a dense confirm) |
| cross-family emission | dense: NULL (parity ±0.7%) · MoE short-seq: at 64k recomp 5919 MEASURED (c12 lead-in row) vs asym ~4200 ANCHOR ESTIMATE (asym never RUN at 64k; the earlier "+29%" figure was unsourced), and measured 80k already flips to asym T1 +6% (3642 vs 3424) | c12 + c14 | scoped: optional anchors-based fallback for MoE <80k only — window at most (64k, 80k), needs a real asym 64k run first (pending Kevin — §4) |

Rule going forward: "drop" is reserved for measured-WORSE or
architecturally-idle machinery; measured-NULL keeps its code (unwired) with
the context noted; unmeasured features get a named promotion test, not a
verdict.

### 2d. Complexity rule (nothing ships without measured value)
Dropped under this rule beyond the table above: the proposed "phase-2
headroom-spender" code (its only unmeasured payload is panel-cache at
−3 us/tok ≈ 0.2%; ohbm-N grading already exists) — kept as a doc note only.
Adopted survivors each carry value per unit complexity: decomposed slopes
(adds the sum-check), knee constant (ONE number reproduces the measured
B_max table + the batch-never-helps-deep law), >800k patch (fixes a measured
13 GiB miss), self-test (~zero cost).

## 3. Concrete merge steps (order matters)
1. Branch `merge_sched` off 40's LIVE tree (has the newest tooling + docs).
2. Source: checkout 42's six `asym_gemm/training/*.py`, then re-apply 40's
   `mutable=` sites + `_keep_stage_noclone_enabled()` on top (mechanical;
   stays UNWIRED — the drop happens at the driver/recipe layer, §2d′).
   All FIVE new flags default-OFF, every 42 hunk env-gated (audited) ⇒ dense
   behavior unchanged. Recipe wiring per §1B nuance: MoE recipes keep 42's
   pins (fused-addmm, reuse-packed-x — embedded in c14's measurements);
   dense recipes get NONE of the five until A/B'd; panel-cache + async-pack
   wired nowhere (async-pack already NULL at its one measured point).
3. Tooling: keep 40's tp_probe (UNTRACKED — `git add` it), ceiling_search,
   run_dial_ladder (MAX_STEPS=2), builder (42's refactor rejected per the
   §1C revised ruling), stubs/_C.pyi. Driver: resolve the :3844 one-liner to
   42's `ASYMM_ATTN_ACT_KEEP_ACTS_HBM` forwarding (this drops noclone
   wiring).
4. Scheduler: land 42's `asym_scheduler.py`, then refactor its `schedule()`
   to the merged rule — feasibility-first over rung-prefixes with the HOST
   term and knee-threshold batch; SPLIT the keep-acts rung into its two byte
   classes (§2b) so T2 is a rung prefix — measured-exact for DENSE (c12 T2);
   MoE's split evidence is ONE dial point (scheduler_v2 §3b 120k×8 measured
   MoE-KA-only — source of the rung's −33 and the PRE-SECANT 0.038 slope)
   with NO deep-seq split LINE, and the 0.052 deep secant that replaced
   0.038 was calibrated on the full-bundle latency
   emission (tputsched 900k, prompt.md v2) — so the MoE T2 line/recipe stays
   the c14 bundle as-measured until a split line exists; keep the ANCHOR
   table as short-seq truth;
   keep `--sweep`/`--selftest`; τ-prediction moves behind `--predict`
   (offline). Port the 5 self-test properties.
5. Constants: one table per (model, rung-prefix) merging 40's byte lines with
   42's per-rung slopes; add per-rung HOST slopes/anchors (§2 host row); mark
   q3.5 pending-fit. Knee N* is measured for q3-30b ONLY — apply the B* cap
   where measured; dense models keep max-feasible-B (current behavior) until
   a dense N* exists. Do NOT silently change dense batch decisions.
6. Docs: c12+c14 live records already unified (shared symlink targets, §1D
   correction) — just copy -39's c14_old archive over; apply §1E doc rulings
   (fix_asym graft, prompt.md preserve, reports union, remaining_optimizations
   carry); merged formal spec = 40's
   system_summary §0–§8 + a new §9 "rung refinement + knee batch + self-test"
   from 42; tombstone the β-dial and water-fill-as-decision in scheduler_v2
   (why-not: µs inputs, Kevin's ruling, cross-machine portability); carry
   42's scheduler_v2 §10 "design record map" forward.
7. Recipe/preset layer (Kevin, 2026-07-21). RUNS accepts `backend|TIER`
   (e.g. `asym_cpuadamwds|T2`) alongside raw tokens (both keep working).
   The driver expands TIER → full recompute token + env flags from the SAME
   (model-family, tier) recipe table the scheduler owns (single source of
   truth — scheduler emits, driver expands, no drift). Backend is NEVER
   auto-selected (optimizer placement stays user-entered; scheduler
   auto-mode prints its default `asym_cpuadamwds` explicitly). Expansion
   happens BEFORE dir naming so artifacts keep full-fidelity names.
   NAMING GAP (verified 2026-07-21): the config label today ends at
   `__gradoff*__weightoff*` — env-only feature flags (both KEEP_ACTS, the
   attn-KA/GC bundle, the pins) are forwarded (driver :3842-3844) but
   appear in NO label component, so runs differing only in those flags
   collide into one config dir (historically dodged via separate campaign
   dirs). FIX (Kevin 2026-07-21 — DESCRIPTIVE components, no hash-as-
   identity): (i) append the recipe flags as short per-flag label
   components in the existing dial style (`__mlpka{0,1}__attnka{0,1}__
   gcsave{cpu,hbm}__fadd{0,1}__xreuse{0,1}__pcache{N}`) — the recipe-
   relevant flag set is ~6 dials, so names stay readable AND fully
   self-describing; the pre-existing `_h<sha1>` (safe_label) remains ONLY
   the >255-char NAME_MAX overflow guard, never the identity; (ii) dump
   `config.json` into each run dir — the complete resolved env dict +
   provenance (tier_requested, expanded token, recipe-table version).
   command.txt stays (c12 used it for config archaeology); profile-JSON
   field checks stay. Name = the dials that matter; config.json = the
   full-fidelity record.
8. Validation gate before declaring merged: (a) selftest passes; (b) the
   merged rule reproduces BOTH measured records (c12 dense/llama + c14 MoE
   crossover incl. 1.6M); (c) one A/B run per model class with 42's source
   features OFF confirms no regression vs 40 baselines; (d) rung features ON
   at their intended tiers re-validate the T2/T3 reference points; (e) two
   runs differing only in recipe flags land in DISTINCT config dirs AND
   each run dir carries a complete, accurate config.json (step-7
   naming+manifest fix effective); (f) the 3×3 post-merge no-regression
   matrix (3 models × 3 recorded points, refs = in-tree
   `archive/s04-p1-dgx-02-c1{2,4}` snapshots) passes 9/9 — see
   fix_merge_scheduler.md S7 for rows/bands/breach protocol.

## 4. Open questions for Kevin
- Only if 42's builder is ever revisited per §1C (currently rejected): its
  `_concat_split_fast/legacy` split — keep both paths or fold?
- panel-cache (+6 GiB for −3 us/tok) breaks strict price-monotonicity at some
  seqs — admit as an optional 4th rung only when headroom is free?
- Cross-family emission: parity kills it for dense, and the MoE short-seq
  case is WEAKER than first written — at 64k recomp 5919 is MEASURED (c12
  lead-in row) but the asym ~4200 side is only an ANCHOR ESTIMATE (asym
  never run at 64k; the earlier "+29%" was unsourced), and measured 80k
  already flips to asym T1 +6% (3642 vs 3424) ⇒ candidate fallback window
  at most (64k, 80k). Families do differ per-model elsewhere
  (q35 rc 384k=1002 at its edge). Run a real asym 64k point first, then:
  re-admit families in that sliver, or keep the scheduler asym-pure and
  document the exception?
