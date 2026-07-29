# fix_cpu_compute — next-builds implementation plan (from the closed op-coverage campaign)

Evidence base: `cpu_compute.md` (STATUS/matrix/log). Policy spec: `placement.md`.
Standing gates for EVERY item: default-off flag → unit/parity → [SMOKE] loss+grad-norm
parity (8k×4, 6 steps, same seed, ON vs OFF) → same-day e2e A/B on 30B@32k (ref 90.69 s)
AND 30B@128k (ref 613.5 s) → only then default-on + update `placement.md` + the
cpu_compute.md STATUS tables. Lossless only. One experiment per node. Record every
number immediately (leaves overwrite). Metrics per P11 (decision log, worker-job ms,
retained-bytes high-water, ENGAGED markers).

STATUS legend: ⬜ todo · 🔄 in progress · ✅ done · ❌ failed/abandoned (with reason).

---

## Item 1 — Single placement-policy module  ⬜
**Goal:** all CPU↔GPU placement decisions made by ONE runtime object implementing
`placement.md` P1–P8, replacing scattered env flags + the manual 32k/128k config split.
**Build:**
- `asym_gemm/training/placement_policy.py`: `PolicyInputs(model_class, context_len,
  routed_rows, act_bytes, proj_rows, host_budget_state)` → `PolicyDecision` per rule id;
  thresholds from `placement.md` (env-overridable, defaults = spec).
- Call sites: replace direct env checks in `qwen3_moe_finegrained.py` (P1 gates),
  `exp_act_offload_lora.py`/deposit paths (P2), `attention_activation_offload.py` (P3 —
  NEW automatic rows-gate, closing the manual split), dense paths (P8 kill-switch).
- P11 tracing: one `placement_policy` block in `runtime_counters.json` + one train.log
  line per decision kind.
**Acceptance:** decisions reproduce the P10 production sets EXACTLY at 30B@32k, 30B@128k,
32B@32k (assert in a dry-run harness + verify via ENGAGED markers in the e2e runs); e2e
numbers within noise of the references (no regression); SMOKE clean.

## Item 2 — Lossless dedup of normalization-layer fp32 saves  ⬜
**Goal:** the q/k-norm (+rope-adjacent) inputs are saved TWICE in fp32 through
`save_on_cpu` (~400 GB/step class at 32k). Deduplicate the saves — fp32 kept, NO recast
(the bf16-recast variant changes backward numerics → stays outside the lossless claim,
flag-gated only).
**Build:** a `saved_tensors_hooks` wrapper around the attention/norm region that
content-dedups identical packed tensors (pack returns a shared handle + refcount;
unpack returns the shared fp32 tensor). Files: `attention_activation_offload.py` (or the
LF checkpointing wrapper where the saves originate).
**Expected:** +2–6% @32k, more @128k; cuts host RAM (the binding limit at long context
and on dense). **Gates:** unit (pack/unpack identity + refcount), SMOKE (must be
bit-identical — pure dedup), both e2e A/Bs, C (host RSS) must DROP.

## Item 3 — Integrate the CPU RMSNorm kernel via module wrapping  ⬜
**Goal:** use the shipped 7× `cpu_rmsnorm_bf16` (≤1 ulp) to CPU-recompute q/k-norm
outputs in backward instead of saving them, WITHOUT saved-tensor-list surgery (that
approach was assessed and REJECTED — anonymous packed-list matching risks silent wrong
gradients).
**Build:** wrap the norm modules in a custom `torch.autograd.Function` that saves only
its input reference (already CPU-resident via the attention offload) and recomputes the
output on the CPU worker during backward, inside the attention-backward window.
Flag: `ASYMM_ATTN_QKNORM_CPU_RECOMPUTE`. Placement gated by the policy module (window
rule analogous to P3).
**Gates:** unit parity vs the HF module (fp32 math, bf16 out); SMOKE hard-gate; both e2e
A/Bs. Interaction check with Item 2 (dedup may already remove most of the win — measure
Item 2 FIRST, then decide if Item 3 still pays; if not, record ❌-superseded).

## Item 4 — Total pinned-memory accounting + cap  ⬜
**Goal:** the dense-32B OOMs are node-level page-locked growth invisible to process RSS.
Build (a) attribution: sample `/proc/meminfo` Mlocked/Unevictable in the host watchdog
log; tag every `cudaHostAlloc`/`pin_memory` site with a tag-family counter (boundary
pool, act pool, deposit slots, adam buffers); (b) enforcement: per-tag-family byte caps +
one global cap; exceeding ⇒ allocation falls back to unpinned or blocks (never OOMs).
**Acceptance:** synthetic stress test shows the cap holds; then ONE dense-32B re-gate run
of P2-deposits under the cap — outcome (win/neutral/bounded-slow/still-infeasible)
recorded as the final dense cell and `placement.md` P8 updated accordingly.

## Item 5 — (OPTIONAL, last) BFMMLA weight-gradient kernel + weight repacking  ⬜
Kernel-quality only (~1.5× micro at 20%→~30% of peak); e2e ≈ nil (wgrad is off the
critical path). Build only if kernel-roofline optics matter for the paper. Micro gate
≥1.3× or abandon.

---

## REMAINING SCOPE (2026-07-16 — the complete, closed list; nothing else is open)

The executor works these in order. Each row is DONE only when its gate column is fully
green and all three docs (this log, placement.md, cpu_compute.md STATUS) are updated.

| # | Item | What exactly | Gates to pass | Success looks like |
|---|---|---|---|---|
| R1 | **Item 7: batch-scaling measurement** (runs FIRST — zero code) | 30B@32k same-day pair: b8 (ref, adopted stack + policy ON) vs b16 (same flags); b24 only if b16 leaves G-headroom ≥40 GiB and C ≤700 GB | both arms 4 measured steps; flags verified; snapshot immediately | tokens/sec table b8 vs b16 (vs b24) with C/G — quantifies memory→throughput; if tok/s does NOT improve, that's a valid recorded verdict |
| R2 | **Item 6: norm/rope recompute-instead-of-save** (`ASYMM_QKNORM_RECOMPUTE`, default off, via placement policy) | custom autograd Function wrapping q/k-norm (+rope variant): save bf16 input ONCE via pinned pool; backward re-derives fp32 output exactly (CPU `cpu_rmsnorm_bf16` when input CPU-resident; GPU upcast fallback); NO saved-tensor-list surgery | (a) unit bit-parity: recomputed output == saved-forward output, EXACT; (b) SMOKE 8k×4×6 same-seed ON/OFF; (c) **headline: same-day e2e A/B 30B@128k** (class = 1.6 TB/step there; expected ~3–8%; C must DROP); (d) 30B@32k A/B (expected small +) | 128k step-time drop with C reduction, lossless; if neutral/negative → honest ❌ with the measured traffic-vs-exposure explanation |
| R3 | Dense-32B spot-check of R2 (cheap, after R2 passes) | one 32B@32k pair with `ASYMM_QKNORM_RECOMPUTE` ON vs OFF (norm-recompute is NOT a deposit — retention-safe by design, so P8 does not block it; confirm via the pinned/retention counters) | one same-day pair + counters flat | first CPU-compute feature validated on the dense model, or an honest cell |
| R4 | Docs closure | fold R1–R3 into: this execution log (final table), placement.md (new rule id for R2 with threshold/scope; batch guidance note), cpu_compute.md STATUS (+ADOPTION list) | — | all three docs agree; no ⬜ without a reason |

| R5 | **Long-context fetch-gap kill (the >5% candidate)** — measure, then prefetch | (a) MEASURE first: quantify GPU idle gaps at attention/layer boundaries in TODAY's tree at 30B@128k (repo record: ~80 s/step gaps @131k, 22.3 s/step bwd host-gap @32k — `agent/scheduler.md:51,:80,:315` — sized BEFORE our fixes; may be partly gone). Cheapest instrumentation: wall-time counters around the unpack/restage waits in `attention_activation_offload.py` (+ existing worker-ms counters), else one nsys arm; (b) if gaps ≥3% of step: implement **one-layer-ahead prefetch** — at each layer-backward boundary, enqueue the NEXT layer's saved-tensor H2D restages on the side stream (consumption order is deterministic); pre-allocate double-buffered stage slots; flag `ASYMM_ATTN_RESTAGE_PREFETCH` default off | (a) gap numbers logged; (b) unit: prefetch returns identical tensors (event-ordered); SMOKE; same-day 128k A/B headline + 32k pair | if today's gaps ≥5%: this is the only in-scope item that can clear the user's >5% bar at the heavy workload; if gaps are already <3%, record the honest closure ("earlier fixes ate them") |


**Explicitly OUT of scope (decided, do not reopen without user):** microbatch/grad-accum
pipelining (user veto) · int8/lossy (losslessness) · SVE Adam (GraceAdam) · dense-32B
deposits (retention-bound; unblock spec recorded) · BFMMLA repack (e2e ≈ nil; optional
paper-optics only) · two-socket kernels (parked: revisit only if R2 leaves the 128k
CPU-act/attention-deposit flips still marginal) · K-7-as-originally-designed
(superseded by R2's corrected design).

---

## Item 6 — Norm/rope recompute-instead-of-save (corrected item-3 design)  ⬜
**Goal:** collect the proven-non-dedupable intra-layer save class (~400 GB/step @32k,
~1.6 TB/step @128k): during the layer recompute, the q/k-norm fp32 outputs (and rope
operands) are saved+restored through `save_on_cpu`. Replace with: save the **bf16 input
once** (pooled pinned offload), recompute the fp32 output **exactly** at backward-use —
same bf16 input + same math ⇒ bit-identical (fp32-upcast recompute). CPU worker runs the
recompute with the existing `cpu_rmsnorm_bf16` (7×, ≤1 ulp) inside the attention-backward
window when inputs are CPU-resident; GPU recompute fallback otherwise (traffic still
collapses ~4–8× for the class). Needs a rope-recompute variant for the rope pair.
**Flag:** `ASYMM_QKNORM_RECOMPUTE` (default OFF), gated via placement policy.
**Gates:** unit bit-parity (recompute == saved-forward output, exact) → SMOKE →
**headline e2e = same-day A/B at 30B@128k** (where the class is 1.6 TB/step; expected
~3–8%) + 30B@32k. C (host RSS) must drop.
**Design caution (from the failed first attempt):** the attention wrapper offloads U/S,
NOT the norm inputs — this design must add the bf16-input save itself; wrap the norm
modules in a custom autograd Function (NO saved-tensor-list surgery).

## Item 7 — Batch-scaling measurement (memory→throughput conversion)  ⬜
**Goal:** quantify the throughput value of the freed HBM: 30B@32k G≈50/186 GiB under the
adopted stack. Same-day pair at 32k: b8 (ref) vs **b16** (and b24 if b16 fits with room),
adopted stack ON in both arms. Metric = tokens/sec (tokens/step ÷ steady). Expected:
step time grows <2× for 2× tokens (streaming amortization + kernel efficiency) →
tok/s up. Host check: C ~367→~550–600 GB (fits 957 pool); HBM ~50→~90 (fits).
NOT applicable at 128k (G≈180/186 — no headroom; that regime's lever is the ceiling
itself). Dense 32B excluded (b16 host-OOMs).
**Record:** tok/s table b8-vs-b16(-vs-b24) + C/G per arm → cpu_compute.md STATUS.

## Execution log
*(append: date · item · what happened · numbers · gate verdicts — keep cpu_compute.md
STATUS and placement.md in sync after every gate)*

### 2026-07-16 — Item 1 build + unit/dry-run gates (resumed after agent restart)
- Found `placement_policy.py` written by the prior session but UNWIRED (no call sites,
  no tests, empty log). Extended it: model-class registry (moe/dense, dense-sticky per
  P8), full rule surface P1–P9 (feature-level + per-call), P11 tracing (dedup per
  (rule,decision), one train.log line each, sidecar `placement_policy.json` next to the
  source profile, atexit export). **Deadlock found+fixed**: `stats()` acquired
  `_STATE_LOCK` then called `thresholds()`/`enabled()` (re-acquire) → RLock (the exact
  cpu_worker-BG lesson from 2026-07-14; py-spy'd, first test run hung).
- Wired call sites (policy consulted only when `ASYM_PLACEMENT_POLICY=1`; env flags
  untouched otherwise): `qwen3_moe_finegrained.py` (P1 feature/async/chunked helpers,
  NEW `_cpu_act_fits()` at both rows/bytes guard sites, P2 deposit helper, P9
  silu-bwd/dB helpers, K-2b arm-B disabled under policy, `register_model_class("moe")`
  at Function.forward), `attention_activation_offload.py` (P3 feature helper + NEW
  automatic per-call rows gate in `_try_deposit_attn_lora_a_grad` — closes the manual
  32k/128k split; P7 retained-bytes high-water counter + `deposit_retention_stats()`),
  `dense_mlp_finegrained.py` (P8 kill-switch on all three dense CPU-compute gates +
  `register_model_class("dense")` at init/forward), `cpu_worker.py`
  (enabled/bg via policy), `cpu_ops.py` (fused/split kernel gate via policy),
  `cpu_adam.py` (P6 fused-widen via policy), LF `checkpointing.py` (P4 boundary-pinned
  via policy, guarded import). P11 plumbing: `placement_policy` block (incl.
  deposit-retention + save-dedup counters) in the source profile
  (`run_lf_profiled_train.py`) and surfaced into `runtime_counters.json`
  (`postprocess_lf_profile_artifacts.py`).
- **Unit + dry-run acceptance harness: PASS 13/13** (`tests/test_placement_policy.py`):
  P10 production sets reproduced EXACTLY at 30B@32k (P1+P2+P3+P4/5/6 ON), 30B@128k
  (P1 auto-off rows, P3 auto-off rows, P2+P4/5/6 ON), 32B@32k (dense: P8 kills
  P1/P2/P3; P4/P6 ON); threshold boundaries; env-ignored-under-policy; sidecar export.
- Item 1 remaining gates: [SMOKE] policy-ON vs manual-flags → same-day e2e A/B 30B@32k
  + 30B@128k (policy-ON must land within noise of the manual production references).

### 2026-07-16 — Item 2 build + unit gate
- `asym_gemm/training/save_dedup.py`: `DedupSaveOnCpu(saved_tensors_hooks)` — drop-in
  for `save_on_cpu(pin_memory=True)` at BOTH GC recompute sites
  (`gc_boundary_offload.py` backward + LF `checkpointing.py` non-pinned path). Dedup
  key = (tensor OBJECT via WeakTensorKeyDictionary, `_version`) — same object + same
  version ⇒ bit-identical content; NO data_ptr matching (rejected as wrong-grad risk);
  map scoped to the recompute region. Pack replicates torch-2.12 semantics exactly
  (pinned empty + `copy_(non_blocking)`); duplicate saves return the SAME shared pack
  (refcount++, skips alloc+D2H). Unpack: each consumer gets its OWN `.to(device)` copy
  (H2D deliberately NOT deduped — a consumer mutating its unpacked tensor can never
  corrupt a sibling; lossless-by-construction). fp32 stays fp32 (no recast). Flag
  `ASYMM_SAVE_ON_CPU_DEDUP=1` default OFF; policy rule `P5.save_on_cpu_dedup` returns
  False until the gates pass. ENGAGED marker + hit/miss/bytes counters in the profile.
- **Unit gate: PASS 6/6** (`tests/test_save_dedup.py`): pack/unpack identity, refcount,
  version guard, weak-map lifetime, region-exit clear, and **bit-identical gradients
  vs stock save_on_cpu on the exact q/k-norm fp32 double-save pattern** (3 dedup hits
  on the fp32 upcast class as predicted). cpu_worker regression suite still green.

### 2026-07-16 — Item 4 build + stress gate
- `asym_gemm/training/pinned_ledger.py`: live/high-water/denials per tag FAMILY +
  global, release-on-GC via `weakref.finalize` (survives pool caching correctly: a
  pooled pinned buffer stays booked while alive), caps `ASYM_PINNED_CAP_TOTAL_GB` +
  `ASYM_PINNED_CAP_GB_<FAMILY>` (default 0 = unlimited ⇒ default-off enforcement).
  Denial ⇒ UNPINNED fallback (non_blocking degrades to sync; never OOMs, never
  blocks); one WARN line per family. OS-truth alongside in stats():
  `torch.cuda.host_memory_stats()` + /proc/meminfo Mlocked/Unevictable.
- Chokepoints wired: `activation_offload._alloc_cpu` (tag threaded through; families
  moe/attn/mlp/gc from handle tags; books the BUCKETED bytes = true page-locked size),
  `_DsSlots.acquire` (deposit), `cpu_adam._pin_if_requested` (adam),
  `save_dedup` pack (save_on_cpu family; stock torch save_on_cpu path stays untracked —
  documented limitation, production dedup path routes through the ledger).
- Host watchdog (`run_lf_lora_sft.sh`): `<train.log>.hostmem.csv` sampling
  (epoch,avail,Mlocked,Unevictable; `HOST_MEM_WATCHDOG_SAMPLE_SECONDS` default 15 s)
  + Mlocked/Unevictable recorded at trip time in the fired-marker file.
- Profile: `placement_policy.pinned_ledger` block in source profile →
  runtime_counters.json (P11 pinned-pool high-water requirement).
- **Stress gate: PASS 8/8** (`tests/test_pinned_ledger.py`): cap holds exactly (50×8 MiB
  requests under a 64 MiB global cap ⇒ exactly 8 pinned + 42 denials, high-water ≤ cap,
  live returns to 0 on GC), family attribution from tags, bucketed-bytes booking,
  unpinned fallbacks stay bit-correct (DsSlots copy + save pack round-trip).
  First run FAILED 5/7 — my test ignored the pool's dim-0 row bucketing (min 8192
  rows ⇒ a [1024,4096] "8 MiB" request actually pins 64 MiB); the LEDGER was right,
  the test was wrong (fixed with 1-D shapes + an explicit bucketed-booking test).
- Remaining item-4 gate: ONE dense-32B re-gate of P2-deposits under the cap (same-day
  pair `b32_off` vs `b32_deposit_cap`, queued in the batch chain after the 30B ladders).

### 2026-07-16 — Items 1+2 [SMOKE] gates (8k×b4, 6 steps, seed 42) — PASS
- Four arms: OFF-manual (full 32k production flag set), OFF-manual RERUN (pure
  nondeterminism envelope), ON-policy (`ASYM_PLACEMENT_POLICY=1` only), ON-policy+dedup.
  Leaves under `profiling_fixcpu/smoke*/` (own OUTPUT_ROOT per arm — no overwrite).
- **Envelope (OFF vs OFF′, identical flags+seed): max rel Δloss 0.67%**; step-0 loss
  spread across ALL four arms ≈1.0% — step-0 is a pure-forward quantity that neither
  the policy nor dedup can mechanically affect ⇒ the spread is route-scatter forward
  nondeterminism (matches the Stage-3 "atomic-add envelope" finding).
- **Item 1 SMOKE PASS**: policy-vs-manual max Δ 0.87% (vs OFF′ only 0.59%), no drift;
  decisions/markers EXACT: P1=True(rows=256000,nbytes=393216000), P2=True→Stage-3
  ENGAGED, P3=True(proj_rows=32000)→K-2 ENGAGED, P4/P6=True, P9=False; decision counts
  match the graph exactly (P1 336=48×7, P3 1344=4proj×48×7, P2 672). Flag-reach proven
  via /proc/<pid>/environ (policy arm carries NO manual feature flags).
- **Item 2 SMOKE PASS (outer wrapper)**: dedup-vs-policy max Δ 0.58%, no drift; outer
  `save_on_cpu` dedup ENGAGED with hits=672 ⇒ exactly 2 hits/layer/step of a
  [B·S]-fp32 (128 KB @8k) rstd-class tensor — i.e. the OUTER GC region only carries
  crumbs of the norm class.
- **KEY DISCOVERY (evidence-driven scope correction)**: the REAL ~400 GB/step fp32
  double-save class lives in `attention_activation_offload`'s own
  `saved_tensors_hooks` pack (`AttentionSavedTensorOffloadWrapper._pack`), NOT in the
  outer save_on_cpu: smoke leaf `offload_bytes_by_tag` shows
  `saved.float32.4x8000x32x128` = 352.3 GB/7 steps = 524 MB × **2 saves/layer/step**
  (the q-norm fp32 upcast; k-norm 4×8000×4×128 ×2; rsqrt 4×8000×32×1 ×2) — scaling ×8
  to 32k reproduces the census "~404 GB/step" exactly. **Item 2 completed at the true
  site**: same (object,_version) dedup added to `_pack` (shared handle; unpack stays
  per-consumer staged copies — same lossless-by-construction argument), same flag
  (`ASYMM_SAVE_ON_CPU_DEDUP` / policy P5 class), map scoped per run(); ENGAGED marker
  `attention saved-tensor dedup ENGAGED`; `dedup_hits/dedup_bytes` in the wrapper
  snapshot. Also item-4: `_empty_strided_cpu_like` (the per-save FRESH pinned alloc —
  prime suspect class for the dense-32B node-level pinned growth) now books through
  the pinned ledger (family `saved`).
- Timing note: the attention-pack dedup landed BEFORE the e32k batch started (all
  three e32k arms run identical code; compile+unit-verified pre-import), but AFTER
  the 8k smoke — supplementary gates queued post-chain: wrapper-level unit tests
  (bit-identity, refcount, version guard — added to tests/test_save_dedup.py) + a
  canonical 8k SMOKE pair for the attention dedup; e32k arm2-vs-arm3 loss parity
  serves as e2e-scale parity evidence meanwhile.

### 2026-07-16 — e2e 30B@32k×b8 ladder (same-day A/B chain) — item 1 PASS, item 2 measured
- Arms (each its own OUTPUT_ROOT under `profiling_fixcpu/e32k/`; 1 warmup + 4 measured;
  steady = middle-2 mean; flag-reach verified via /proc environ per arm):
  | arm | steady | steps | C | G |
  |---|---|---|---|---|
  | OFF-manual (full production set) | **90.73 s** | 90.77/90.70/90.77/90.93 | 367.3 | 52.7 |
  | ON-policy (`ASYM_PLACEMENT_POLICY=1` only) | **90.81 s** | — | 367.5 | 53.1 |
  | ON-policy + dedup | **90.79 s** | — | 367.2 | 52.4 |
- **Item 1 e2e @32k: PASS** — policy reproduces the manual production set within noise
  (+0.08 s = +0.09%; the reference 90.69 s reproduced by both arms same-day), C/G flat,
  decision counts exact (P1 True×240 = 48×5, P2×480, P3 True×960 = 4proj×48×5,
  P4/P6 True, P9 False, P8 never fired), ENGAGED markers all present, losses vs manual
  ≤0.38% (within the 0.67–1.0% rerun envelope), no drift.
- **Item 2 e2e @32k: TIME-NEUTRAL as measured (−0.02 s vs policy arm), C flat** —
  losses vs policy arm ≤0.16% (tight; e2e parity evidence). Engagement analysis:
  outer dedup hits=480 (decoder-LN rstd class, ~0.5 GB/step); attention dedup
  ENGAGED but hits=480 = ONLY the q/k variance class ([B,S,H,1] fp32, 8.8 GB total,
  first-hit tag `saved.float32.8x32000x32x1`); the BIG fp32 upcast pairs
  (`saved.float32.8x32000x32x128`, 4.2 GB ×2/layer/step — 4,547 of 4,565 GB total
  saved.float32 traffic) did NOT dedup: production presents them as DIFFERENT Python
  objects, while an isolated reproduction of the exact HF-norm sequence dedups the
  same pair perfectly (unique_ids=1 → hit; verified on idle GPU1). ⇒ production graph
  presents the two 4.2 GB packs differently (different wrappers or genuinely distinct
  tensors). **Fix attempt #1 armed**: inert pack-identity debug mode
  (`ASYM_ATTN_SAVED_TENSOR_DEDUP_DEBUG=1`, logs shape/id/version/data_ptr/storage_ptr
  for first 8 packs per shape) + smoke3 batch will classify the pair; decision after:
  either extend the key safely (weakref-anchored storage identity + shared version
  counter) or record the honest verdict (distinct-content saves ⇒ not deduplicable;
  neutral-shipped-off). Note: e128k arm-1 imported the module pre-debug-edit; arms
  2/3 import the same code plus the inert debug branch — no behavioral delta.

### 2026-07-16 — e2e 30B@128k×b8 ladder (same-day A/B chain), arms 1–2
- | arm | steady | steps | C | G |
  |---|---|---|---|---|
  | OFF-manual (128k production set: K-1+RS-2+K-9+K-8+bg, NO attn deposit, NO chunked) | **628.00 s** | 618.3/621.2/634.8/627.4 | 695.2 | 182.6 |
  | ON-policy (`ASYM_PLACEMENT_POLICY=1` only) | **616.54 s** | 615.5/615.9/617.2/620.1 | 695.1 | 180.6 |
- Cross-day note: manual 628.0 vs yesterday's 613.5 reference = day-drift class (±10 s
  documented); today's ladder self-anchors (same-day A/B only).
- **Item 1 e2e @128k: PASS** — the policy AUTOMATED the manual 32k/128k split exactly:
  P1.moe_cpu_act=False×480 (rows 8.192M > 4.2M), P3.attn_wgrad_deposit=False×960
  (proj rows 1.024M > 256k), P2=True×480 (deposits ENGAGED), P4/P5(class)/P6=True,
  NO K-2 attention marker (forbidden-marker check passed). Time: policy ≤ manual
  (616.5 vs 628.0; fastest-step 615.5 vs 618.3) — no regression; the 11.5 s delta is
  attributed to the manual arm's within-run variance (step spread 16.5 s vs 4.5 s;
  its step-3 outlier 634.8 s), NOT claimed as a policy win. Losses ≤0.40%, no drift;
  flag-reach verified (policy arm carries no manual feature flags).
- **Arm 3 — ON-policy + dedup @128k: 617.24 s** (steps 615.9–620.9 class), C-695.1
  (flat), G-182.9. vs policy arm: **+0.7 s ≈ TIME-NEUTRAL** (within the 4.5 s step
  spread); losses ≤0.59% vs policy arm, no drift. Dedup engagement mirrors 32k: outer
  hits=480 (1.97 GB, decoder-LN rstd class at 128k), attention wrapper on the
  variance class only — the big fp32 pairs still present as distinct objects.
- **Item 2 e2e verdict so far (both lengths): TIME-NEUTRAL, C-NEUTRAL, lossless** —
  the mechanism is correct (bit-identical unit gates; parity ≤0.6% e2e) but only
  collects the small duplicate classes as-built; the ~400 GB/step fp32 class does not
  dedup in production. Root-cause classification pends the smoke3 debug pair
  (post-chain); outcome = either fix attempt #2 (safe key extension) or an honest
  ❌/neutral verdict ("the census double-save class is two DISTINCT fp32 intermediates,
  not duplicates — only recompute (item 3 class) can collect it").

### 2026-07-16 — Item 4 dense-32B re-gate, arm 1 (b32_off = P10 dense reference) + RAM-pool correction
- Session cut killed my ladder wrappers but the whole driver tree survived detached;
  the ORIGINAL run_ladder also survived and launched the cap arm — my duplicate
  resume-chain was killed before reaching the GPU (verified via /proc environ
  OUTPUT_ROOT; one-experiment-per-node preserved; duplicate leaf removed).
- **b32_off: 384.70 s steady** (steps 384.4–386.5, tight), **C-560.5, G-109.9**,
  losses recorded; FORBID checks clean (no policy lines, no deposits, no dedup).
  vs the 352.3 s Jul-15 reference: +9% — cross-day + the K-1/K-8-ON dense set (P10
  says P4/P6 ON; the 352.3 anchor was flags-OFF) ⇒ today's pair self-anchors.
- **Pinned attribution (first-ever per-family numbers, OFF arm)**: ledger high-water
  total 245.6 GB — gc(boundary pool) **150.3 GB** (64 dense-layer boundaries ×2.6 GB),
  mlp 56.5 GB, saved(attn packs) 28.5 GB, adam 3.2 GB, o/q/k/v tiny. **Attribution
  finding: /proc/meminfo Mlocked=185 MB while 245.6 GB is cudaHostAlloc-pinned —
  page-locking via the CUDA driver is INVISIBLE to Mlocked/Unevictable on this
  kernel**, which explains the old "growth outside process-visible allocations"
  diagnosis and makes the ledger the only reliable tracker of our pinned bytes.
- **RAM-pool correction applied (agent/project_rules.md)**: host CPU pool = **~957 GB**
  (NUMA 0+1); `free`'s ~1.69 TB is fabric-inflated by GPU HBM NUMA nodes. Recalibrated:
  baseline C-560.5 = 58.6% of pool (headroom ≈ 397 GB − OS); the historical 32B OOM
  math now closes exactly (560 + measured ~513 GB backward growth = 1,073 > 957 →
  watchdog at 33–35 GB avail). Prior log lines quoting ~1.6 TB "avail" are void as
  headroom evidence.
- Cap arm (b32_deposit_cap) RUNNING with caps: total 350 GB, save_on_cpu 140, mlp 120,
  attn 60, **gc 20 (set before knowing gc's real 150 GB footprint — the arm will deny
  most boundary pins ⇒ heavy unpinned-fallback exercise; slowdown will be attributed
  to BOTH deposit backpressure and gc-pin denial, honestly)**; deposits+dedup ON.
- **Cap arm ATTEMPT 1: CRASHED (finding, not OOM)** — after ~4 min of setup + first
  forward: `RuntimeError: down: CPU-left LoRA-A source must be pinned CPU memory`
  (`exp_act_offload_lora._check_cpu_left_inputs`). Mechanism: the mlp-family cap
  denied a pin → `_alloc_cpu` returned an UNPINNED act-pool buffer → the CPU-left C2C
  GEMM kernel (which reads page-locked host memory directly from the GPU) refused it.
  **Lesson: the unpinned fallback is safe ONLY for copy/staging consumers (saved
  packs, gc-boundary H2D, deposit slots, adam staging — all verified in the stress
  suite + the attn pack has an explicit unpinned sync-copy branch); act-pool families
  feeding cpu-left kernels need a CONSUMER-SIDE fallback (route to the GPU LoRA-A
  path when the source is unpinned) — spec'd as follow-up, not built.** Watchdog did
  NOT fire, no OOM evidence (cgroup counters flat) — the caps ENGAGED (2 DENIAL lines)
  and failed CLOSED, not open. Evidence kept: dedup ENGAGED on 32B (fp32
  [8,32000,64,1] variance class), K-3 deposits ENGAGED, C-578 peak during setup.
  Runner caveat noted: the driver postprocesses partial profiles and exits rc=0, so
  the ladder printed PASS on markers — the STEADY=nan/STEPS=[] row is the tell.
- **ATTEMPT 2 queued (`b32c`)**: same arm with caps restricted to the SAFE families
  (total 350 GB + save_on_cpu 140 + gc 20 + adam 60 + deposit 8; NO mlp/attn per-family
  caps — they stay bounded by the total + their consumers keep pinned memory).

### 2026-07-16 — smoke3 (dedup debug/classification) + smoke4 (dense P8) + item-2 attempt #2
- **smoke3 PASS (canonical 8k SMOKE for the attention dedup)**: dedup-vs-ref losses
  ≤0.62% (envelope 0.67–1.0%), no drift; time 14.26 vs 14.27 s.
- **CLASSIFICATION (from `[attn-dedup-debug]` pack identities, mechanical)**:
  the big fp32 pairs (q [B,S,32,128], k [B,S,4,128]) are **SAME-STORAGE ALIASES** —
  different Python wrappers, identical (storage ptr, offset, sizes, strides), both
  version 0 ⇒ bit-identical content, SAFELY dedupable; the variance pairs are
  same-object (already deduped); the bf16 [B,H,S,D] pairs are **DISTINCT tensors**
  (different storages — rope/SDPA operands; NOT deduplicable, recompute-only class).
- **Item-2 attempt #2 BUILT**: weakref-ANCHORED alias map added to BOTH packs
  (attention wrapper + outer DedupSaveOnCpu): key = (storage ptr, offset, dtype,
  sizes, strides); a hit requires the FIRST wrapper alive (anchor ⇒ storage cannot
  have been freed/reused — no data_ptr-reuse hazard) AND both wrappers' version
  counters == packed version (view-lineage shares the counter ⇒ interleaved in-place
  writes refuse). **Unit gates 9/9 PASS** incl. new alias test (same-storage
  different-wrapper hit + mutation refusal). Gates queued (chain5): smoke5 pair +
  e32k2 pair + e128k2 pair — the 128k pair is the money measurement (the alias class
  is ~0.9 TB/step of D2H + pinned churn at 128k).
- **smoke4 PASS — item-1 dense column verified in a REAL 32B run (policy-ON)**:
  `model_class registered: dense`, P8.dense_cpu_compute=False traces fired, ZERO
  CPU-compute engagement (no Stage-3/K-3/K-2 markers), P4/P6 on; steady 43.9 s/step,
  C-191.5, G-14.8 at 8k×b4.
- **smoke5 (attempt-#2 canonical SMOKE): PASS** — dedup-vs-ref ≤0.45%, no drift;
  time 14.34 vs 14.28 s. BUT the leaf showed the ALIAS pairs did NOT hit (attention
  hits=336 = variance class only, 1.4 GB): **the weakref anchor was the bug** —
  production saved-wrappers are EPHEMERAL (autograd re-wraps per save; that is WHY the
  pair had different ids), so the first wrapper is dead before the duplicate arrives
  and the anchored entry self-invalidated. **Attempt #2b**: STRONG anchors in a FIFO
  capped at 4 entries (pop-on-hit, clear-at-region-exit) — memory-neutral by
  construction (the duplicate arrives within the same norm while the tensor is alive
  anyway; the last ≤4 packs of a region are alive at region end regardless). Unit
  suite updated for the ephemeral-wrapper case + the new anchor-lifetime contract:
  9/9 PASS. e32k2/e128k2 dedup arms measure 2b.
- **Attempt-#2/#2b e2e results (same-code pairs)**: @32k ref 90.80 s / dedup 90.78 s
  (TIME-NEUTRAL, C flat; variant A dtype-blind N=4 anchors cost **G +9.6 GiB** →
  fixed with fp32-only + N=2 anchoring, G verified flat at 128k). @128k evening pair:
  ref 636.5 s (steps 625–645!) / dedup 662.8 s (steps 636–672, rising) — **+4.1%
  DRIFT-CONFOUNDED**: the node degraded monotonically at 128k across the day
  (morning policy 616.5 → afternoon 628.0 → evening 636.5/662.8) while 32k stayed at
  90.7–90.8 all day; the tight MORNING pair (616.5 vs 617.2, ±2 s steps) is the
  quotable 128k measurement: **NEUTRAL**. G held at 182.9 (fp32-only anchor fix ✓);
  losses ≤0.4% every pair, no drift.
- **Engagement truth (both e2e arms + leaf counters)**: alias pairs STILL did not
  hit (hits remain the variance-only 480; 35.4 GB @128k). The once-per-class miss
  diag was consumed by each class's FIRST pack (design error in the diagnostic) —
  the informative near-miss line for the DUPLICATE was suppressed. Notably the
  supposedly-impossible case remains: same storage+offset+shape with equal-shape
  non-square dims implies equal strides, so the key "should" match — the remaining
  suspects are a differing storage at the dup (fresh materialization at save #2) or
  a key component changing between packs. **Attempt #2c = widened diagnosis (first
  3 misses per class, near-miss prints both key tuples) + one 8k debug smoke
  (smoke6, queued after the item-4 b32d re-gate). Per the 3-attempt rule this is the
  final iteration: if the key component is fixable trivially, fix + regate at 8k;
  otherwise item 2 ships as VARIANCE-CLASS-ONLY (lossless, time/C-neutral,
  ENGAGED-verified) with the full diagnosis recorded for the paper.**
- **Item 4 FINAL (attempt 3, b32d): rc=143 soft host-OOM at 0 steps — RETENTION,
  not pinning, is the dense blocker.** Watchdog fired at 35.8 GB avail (floor 35;
  957 GB real pool); only TWO cap denials (gc once, mlp once at total 364.2 GB —
  the 350+ GB pinned cap HELD); after the mlp denial the same growth continued in
  UNPINNED memory (hostmem.csv: ~1 GB/s drain to the floor; Mlocked flat at 185 MB
  throughout). The consumer-side fallback worked mechanically (no pinned-check
  crash; run survived denials deep into the first backward). CONCLUSION: the
  deposit-deferred mlp act-handle RETENTION (~13.4 GB/layer held) exhausts the node
  regardless of memory kind; a pinned cap relocates, not reduces, the footprint.
  **Item-4 deliverables that STAND**: ledger + per-family attribution (mlp 56.5→290 GB
  under deposits — first quantification of the dense retention), cap enforcement
  (fails closed), unpinned fallback (unit-exact + e2e-exercised), Mlocked-invisibility
  finding, watchdog hostmem.csv. **Dense-deposits cell: INFEASIBLE-AS-BUILT
  (retention-bound)**; root fix spec'd = dense deposit_ctx retention budget
  (P7-mirror), placement.md P8 updated with the revised unblock condition.: if attempt-#2 gates pass, item 3
  is **❌-superseded** (its target — the norm's duplicate fp32 save — is collected by
  alias dedup; the remaining large pairs are DISTINCT bf16 rope/SDPA operands outside
  the qk-norm-recompute design; a rope-recompute variant would be a NEW item). If
  attempt-#2 fails its gates, item 3 becomes the only collector of the class but is
  budget-infeasible today → defer-with-reason. Final verdict after chain5.

### 2026-07-16 — smoke6 (attempt #2c): DEFINITIVE classification → items 2 and 3 CLOSED
- **The big fp32 pairs are DISTINCT TENSORS.** smoke6's widened diagnostics show the
  duplicate's near-miss as `same-shape-diff-storage` (identical strides+offset,
  DIFFERENT storage ptrs) for both q [4,8000,32,128] and k [4,8000,4,128] classes.
  smoke3's earlier "SAME-STORAGE-ALIAS" label was an artifact: its 8-per-class global
  debug cap captured CROSS-STEP pack pairs whose storages matched via caching-allocator
  block reuse (same slot each step, fresh wrappers, both version 0). **No identity key
  can losslessly dedup two distinct allocations; the class is collectible only by
  RECOMPUTE.** Fix attempts #2/#2b/#2c exhausted per the 3-attempt rule.
- **ITEM 2 FINAL: SHIPPED, VARIANCE-CLASS-ONLY, NEUTRAL.** Lossless dedup
  (object-identity + sound alias key) collects the rstd/variance duplicate classes;
  time/C-neutral at both lengths (32k: 90.78 vs 90.80; 128k morning pair: 617.2 vs
  616.5; evening pair drift-confounded and not quoted), losses within envelope in
  every pair, 9/9 bit-identity unit gates. The alias-anchor code is SOUND (a live
  strong anchor pins the storage, so sptr matching cannot alias recycled blocks)
  but INERT in production; fp32-only N=2 anchoring is memory-neutral (G flat @128k;
  the dtype-blind variant's +9.6 GiB G @32k was found and fixed). Flag stays
  default-off; `P5.save_on_cpu_dedup` stays False (no measured win → no default-on).
- **ITEM 3 FINAL: DEFER-WITH-REASON, PRIORITY RAISED.** The ~400 GB/step @32k
  (~1.6 TB/step @128k) fp32 class is proven collectible only by recompute — item 3's
  family is the only path. Not built today because (a) the design premise "norm input
  already CPU-resident via the attention offload" is FALSE (the wrapper offloads U/S,
  not the norm input — verified in code), (b) the shipped CPU kernel is forward-only
  and ≤1 ulp (not bit-lossless; the bit-exact variant = save bf16 input once +
  recompute the exact fp32 upcast in backward, needing saved-tensor wiring in the
  attention region), (c) the SMOKE + same-day two-length e2e protocol cannot complete
  in the remaining day. Concrete next design recorded: single bf16 save (2.1 GB vs
  2×4.2 GB fp32 @32k, ×4 @128k) + exact-upcast recompute ⇒ bit-lossless by
  construction, collects BOTH fp32 copies.

### 🏁 2026-07-16 — EXECUTION CLOSED: final per-item verdict table
| item | shipped | gates | key numbers (same-day pairs) | verdict |
|---|---|---|---|---|
| 1 placement policy | `placement_policy.py` + all call sites + P11 tracing (`ASYM_PLACEMENT_POLICY=1`) | unit/dry-run 13/13 · SMOKE ≤0.87% (env 0.67–1.0%) · e2e 32k+128k · dense smoke (P8) | 32k: policy **90.81** vs manual **90.73** (ref 90.69 reproduced); 128k: policy **616.5** vs manual **628.0** (P1/P3 auto-off = split automated); decisions exact (P1×240/P2×480/P3×960) | **✅ SHIPPED+GATED — production-equivalent from one flag** |
| 2 norm-save dedup | `save_dedup.py` + attention-pack dedup + sound alias anchors (default-off) | 9/9 bit-identity units · SMOKE pairs clean · e2e 32k+128k pairs | 32k: 90.78 vs 90.80; 128k (morning, tight): 617.2 vs 616.5; G flat after fp32-only fix (dtype-blind cost +9.6 GiB, caught+fixed); hits = variance classes only | **✅ shipped, NEUTRAL (variance-only); big fp32 class = DISTINCT tensors (smoke6 proof) ⇒ not dedupable** |
| 3 CPU rmsnorm recompute | not built (kernel exists, fwd-only) | — | class size now proven recompute-only: ~400 GB/step @32k, ~1.6 TB/step @128k | **DEFER-WITH-REASON, priority raised; bit-exact design recorded (save bf16 once + exact upcast recompute)** |
| 4 pinned ledger + caps | `pinned_ledger.py` + 5 chokepoints + watchdog Mlocked CSV + consumer fallbacks | stress 8/8 · 3-attempt dense re-gate (crash-closed ×2, retention-OOM ×1) | dense mlp retention 56.5→290 GB quantified; cap held 364<376 GB while node OOM'd via UNPINNED retention (~1 GB/s); Mlocked 185 MB vs 364 GB booked | **✅ ledger/caps/fallbacks SHIPPED; dense deposits INFEASIBLE-AS-BUILT (retention-bound); P8 unblock = retention budget** |

Cross-cutting findings recorded: 957 GB real CPU pool arithmetic closes the historical
32B OOM exactly; 128k runs drift monotonically across a heavy day (616→663) while 32k
holds ±0.1 s — same-day means adjacent-pair at 128k; cudaHostAlloc is invisible to
Mlocked/Unevictable; caching-allocator block reuse can masquerade as tensor aliasing
in identity-based diagnostics (guard with strong anchors or per-run maps).

### 2026-07-16/17 — R1 (item 7) batch-scaling ladder @30B·32k (adjacent same-chain arms)
- Arms = adopted production stack via the gated policy (`ASYM_PLACEMENT_POLICY=1
  ASYM_CPU_OPS_THREADS=48`, NO manual feature flags — flag-reach proven per arm via
  /proc environ), own OUTPUT_ROOT per arm under `profiling_fixcpu/i7/`; b8 → b16 →
  b24 (b24 auto-gated on b16's measured C/G: RUN at hbm_room=82.6 GiB, c16=475,
  c24_proj=582.5).
- | arm | steady | steps | C | G | tokens/step | tok/s |
  |---|---|---|---|---|---|---|
  | b8 (ref) | **90.63 s** | 90.70/90.57/90.69/90.76 | 367.5 | 54.0 | 256,000 | **2824.7** |
  | b16 | **181.26 s** | 186.29/179.96/182.56/180.87 | 475.0 | 103.4 | 512,000 | **2824.7** |
- **b8→b16: EXACTLY throughput-neutral** (step time ×2.0002 for ×2 tokens; tok/s
  Δ +0.00%). The freed HBM does NOT convert to throughput at 32k in this regime —
  the policy auto-flips P3 OFF at b16 (proj_rows 512k > 262,144; K-2's −1.1% win
  is forfeited) and P1 stays ON at the window edge (rows 4,096,000 ≤ 4,194,304;
  bytes 6.29 ≤ 6.4 GB), i.e. the b16 arm runs a weaker placement set by the
  policy's own measured rules. Decisions verified: P1 True×240, P2 True×480,
  P3 False×960 (b16) vs True×960 (b8); markers exact (Stage-3 ENGAGED both; K-2
  ENGAGED b8 only). Host check: C 367.5→475.0 (fits 957 pool easily; below the
  ~550–600 estimate), G 54.0→103.4 (fits 186).
- **b24 (auto-gate RUN): 310.23 s, C-693.7, G-136.7**, 768,000 tok/step →
  **2475.6 tok/s = −12.4% vs b8** (REGRESSION). Policy flips BOTH winning
  placements off: P1 False×480 (rows 6,144,000 > 4,194,304) AND P3 False×960 —
  the large-batch arm runs the weakest stack by the policy's own measured rules.
  C-693.7 fits the 957 pool (matches the 582.5 linear projection loosely; the
  extra ~110 GB is the CPU-act path going back on-GPU when P1 auto-offs).
- **R1 VERDICT: batch scaling does NOT convert freed HBM to throughput at
  30B·32k.** tok/s = 2824.7 (b8) → 2824.7 (b16, exactly neutral) → 2475.6 (b24,
  −12.4%). Root cause is structural, not contention: the policy's rows-gates
  (P3 ≤262,144; P1 ≤4.194M) disable the CPU placements as batch grows, so bigger
  batches monotonically LOSE the placement wins (b8: P1+P3 on; b16: P1 on, P3 off;
  b24: both off). HBM headroom is genuinely freed (G 54→103→137 of 186) but the
  throughput lever it would feed (larger batch) simultaneously trips the gates that
  earned the per-step win. Honest recorded outcome per the R1 spec ("if tok/s does
  NOT improve, that's a valid recorded verdict"). Implication for the paper: the
  freed HBM's value is the SEQ-CEILING (128k regime, G≈180/186), not batch at 32k;
  and IF batch scaling is wanted, the P1/P3 thresholds would need re-measuring at
  b16/b24 shapes (the crossover rows were calibrated at b8) — a separate study.
  C/G all fit the 957 GB pool with room; no host-RAM pressure at any batch.

### 2026-07-17 — R2 (item 6) build + unit + SMOKE gates
- **BUILD**: `asym_gemm/training/qknorm_recompute.py` — custom autograd Function on the
  q/k-norm modules (installed per attention parent in lf.py's saved-tensor-offload walk):
  forward computes via the module's ORIGINAL forward under no_grad (nothing saved),
  offloads the bf16 input ONCE through the POOLED pinned offload
  (`ActivationOffloadManager`, tags `qknorm.{q,k}_norm.x`, async D2H + ready event);
  backward restages on the side stream (FRESH transient stage — no persistent cache;
  G@128k runs 180/186) and rebuilds the exact fp32 chain on the same device under
  enable_grad, `torch.autograd.grad` through the local graph ⇒ **bit-identical
  gradients by construction** (trainable-weight case links the real Parameter leaf —
  grads returned, never accumulated). NO saved-tensor-list surgery. Flag
  `ASYMM_QKNORM_RECOMPUTE` default OFF; policy rule `P12.qknorm_recompute` (False until
  gated; env arms even under policy — save_dedup precedent).
  **Evidence-driven scope corrections (measured, smoke5 tag census):** (a) the "rope
  pair" of the item text is SDPA's saved rope OUTPUTS (q_embed/attn_out bf16 [B,H,S,D]);
  the rope muls themselves save NOTHING operand-shaped (cos/sin frozen ⇒ rope backward
  is linear with constant coefficients) — the rope-recompute variant is BUILT +
  parity-gated (`ASYMM_ROPE_RECOMPUTE`) but EVIDENCE-OFF (it can only add saved bytes in
  this graph; a unit test documents the premise). (b) the CPU worker path: the shipped
  `cpu_rmsnorm_bf16` is FORWARD-only and ≤1 ulp; the norm BACKWARD needs the exact fp32
  chain on-device and any nonzero ulp breaks bit-parity ⇒ `ASYMM_QKNORM_RECOMPUTE_CPU`
  is accepted but resolves to the GPU recompute with a one-line notice (a CPU path
  needs an rmsnorm-BACKWARD kernel — recorded as the K-7 follow-up).
- **R5 instrumentation landed in the same batch** (both 128k arms carry it):
  GPU-EXPOSED restage-wait counters via CUDA timing-event pairs (compute-stream arrival
  mark vs side-stream copy-done; per scheduler.md's "never attribute from wall-time
  alone") at ALL restage sites: attention `_unpack`, `manager.stage`, qknorm
  `_stage_fresh` (+ host-ms for sync branches). Profile block
  `placement_policy.restage_gap` (+ `qknorm_recompute`) → runtime_counters.
- **Unit gates: PASS 9/9** (`tests/test_qknorm_recompute.py`): recomputed output +
  grad_x + grad_w EXACT (`torch.equal`) vs direct autograd for both the
  production-replica double-upcast norm and the stock HF norm (baseline determinism
  guarded); fp32 [B,S,H,D] saves ELIMINATED inside a pack-hooks region when ON (and
  exactly one bf16 offload); pool release/reuse clean; rope wrapper bit-parity; rope
  saves-nothing-today premise; CPU kernel ≤1 ulp contract re-verified; policy gate-arm.
  Placement-policy + save_dedup suites still green (22/22).
- **SMOKE (8k×b4, 6 steps, seed): PASS** — ref 14.44 s / ON 14.06 s, C 269.0→266.4
  (drops), G 12.6 flat; losses per-step max |Δ| = 0.60% (envelope 0.67–1.0%), no drift;
  markers exact (ENGAGED in ON only; P12=False traces in ref; rope never engaged).
- **SMOKE-leaf mechanism proof (offload_bytes_by_tag):** ref arm carries the fp32 class
  `saved.float32.4x8000x32x128` 328.1 GiB + k 41.0 + rsqrt 2.6 = 371.7 GiB/run; ON arm:
  **fp32 saved tags NONE** — replaced by `qknorm.q_norm.x` 82.0 GiB + `qknorm.k_norm.x`
  10.3 GiB = 92.3 GiB/run at exactly 672 offloads = 672 recomputes (once per norm per
  layer per pass) ⇒ **the class collapsed 4.03×** with bit-parity preserved. R5
  counters: ref total exposed restage wait ≈ 732 ms/step (~5.1% of step @8k; #1
  contributor = the fp32 unpack class at 238 ms/step) → ON ≈ 529 ms/step (the recompute
  removes most of the fp32-class gap). 128k headline pair RUNNING (adjacent arms).

### 2026-07-17 — R2 (item 6) e2e A/B gates: **PASS BOTH LENGTHS — headline delivered**
- **30B@128k×b8 (HEADLINE, adjacent same-day pair):** ref **618.30 s** (steps
  616.26–618.85, C-695.2, G-180.2) vs recompute-ON **596.23 s** (steps 595.60–597.08,
  C-631.5, G-180.2) ⇒ **−22.07 s = −3.57%** (inside the predicted 3–8% band),
  **C DROPS −63.7 GB**, G flat; losses per-step ≤0.61% (envelope 0.67–1.0%), no drift;
  step spreads tight and non-overlapping (±0.7 s each) — clean signal, not drift.
- **30B@32k×b8:** ref **90.62 s** (C-367.4, G-54.1; reproduces the 90.63/90.69
  reference) vs ON **87.95 s** (C-351.4, G-52.6) ⇒ **−2.67 s = −2.95%**, C −16.0 GB,
  G −1.5 GiB; losses ≤0.20%, no drift. **NEW 30B@32k best = 87.95 s** (prior 90.69).
- Markers/decisions exact in all arms (ENGAGED in ON arms only; P12=False traces in
  refs; P1/P3 auto-off @128k; rope never engaged; flag-reach via /proc environ).
- **R5 gap attribution @128k (ref arm; counters live in both arms):** total exposed
  restage wait **60.13 s/step = 9.7% of step**. Top classes: fp32 norm unpack
  12.71+1.60 s/step (⇒ removed by R2), moe stage family ~36 s/step
  (gate_for_silu_bwd_dup 7.19 + up_for_silu_bwd_dgate 7.15 + gate_for_silu_bwd_dgate
  7.13 + gate/up/act_for_act+down_base 3×4.82), bf16 SDPA unpack 6.43+0.79 s/step,
  X_for_dA 0.97, S_*_for_dB ~1.5. All copies share ONE side H2D stream and the compute
  stream waits at stage()-time. Structural G-bind noted: the double gate-stage exists
  because holding the gate stage across the up window costs +11.7 GiB at G-180/186
  (drop_cache is deliberate) — prefetch variants must fit the G ceiling. ≥3% bar
  CLEARED ⇒ R5 build phase is GO (variant selection pending 32k gap numbers + G
  headroom).
- **ITEM 6 VERDICT: ✅ SHIPPED + GATED WIN — lossless (bit-identical mechanism),
  −3.57% @128k with C −63.7 GB, −2.95% @32k with C −16 GB.** Default-on decision
  (policy P12 → True) queued for the docs-closure step (all gates green).

### 2026-07-17 — R3 (dense-32B spot-check of item 6): **PASS, −4.16% — first CPU-side win on dense**
- Same-day adjacent pair @32B·32k×b8, policy-ON both arms ([C32] row, ohbm8): ref
  **385.39 s** (C-560.4, G-109.9) vs recompute-ON **369.37 s** (C-528.6, G-107.7) ⇒
  **−16.02 s = −4.16%, C −31.8 GB, G −2.2 GiB**. Markers exact (ENGAGED ON-only; P8
  kill-switch traces in both; zero CPU-compute engagement); flag-reach verified.
- Counters: qknorm 640 offloads (2 norms × 64 layers × 5 passes) through TWO pooled
  pinned allocs (pool reuse perfect); 'saved'-family pinned allocs 3840 → 1280 (the
  fp32 class gone); deposit retention 0/0 both arms — **norm-recompute is retention-
  safe on dense as designed (P8 untouched; P12 is a save-traffic transform)**.
- **Dense is even more restage-bound than MoE**: exposed restage wait ref 65.5 →
  ON 55.2 s/step (14–17% of step!), dominated by the mlp stage family (8 tags ×
  5.65–6.72 s/step ≈ 50 s/step: gate/up/act forward restages, silu-bwd 3-stage,
  dgate/dup offload→restage roundtrips) — WITH ~78 GiB G headroom (G-108/186), i.e.
  the G-for-time trades that are forced at 128k are all AFFORDABLE on dense. This
  attribution is the R5/P1 dense build target.

### 2026-07-18 — P1 (R5 restage prefetch/reuse) BUILD (mandate round, priority 1)
- Core (`activation_offload.py`): SECOND dedicated H2D stream for prefetch copies
  (the counters measured the single shared side stream as a serializer: 60 ms/event
  for 23 ms-of-bytes SDPA unpacks @128k) + `stage_begin`/`stage_commit` split with
  per-tensor events (begin issues the copy WITHOUT the compute-stream wait; each
  consumer waits its OWN event at use time) + G guard `prefetch_free_ok`
  (`ASYM_PREFETCH_MIN_FREE_GB`, default 16 — prefetch holds stage memory
  earlier/longer, legal only with HBM headroom: live @32k G~53 and dense G~108,
  auto-no-op @128k G~180). Flag `ASYMM_ATTN_RESTAGE_PREFETCH` (alias
  `ASYM_RESTAGE_PREFETCH`), default OFF, policy rule P13 (False until gated).
- moe fg (`qwen3_moe_finegrained.py`): early-issue gate/up restages at backward
  entry (hidden under the down blocks) + single-gate-stage reuse (out-of-place silu
  keeps gate intact — eliminates the third H2D; bitwise-identical dgrads, unit-gated).
- dense fg (`dense_mlp_finegrained.py`): same two mechanisms PLUS dgate/dup kept
  ON-GPU (skips their offload→restage roundtrips — the mlp.dgate/mlp.dup classes,
  ~13.4 s/step); guard sized 4× gate bytes.
- attention wrapper: region-exit prefetch of all packed saves on the prefetch
  stream in REVERSE pack order (last packed ≈ first consumed), per-tensor events;
  documented: one-layer-ahead is IMPOSSIBLE for this class by construction (saves
  are born during the layer's backward-recompute) — region exit is the earliest
  legal issue point; the win channel is the second stream + earliest issue.
- gc boundary (`gc_boundary_offload.py`): TRUE one-layer-ahead — boundary tensors
  exist from forward; LIFO registry mirrors consumption; at each layer's backward
  the next boundary's H2D is enqueued on the prefetch stream (≤1 ahead =
  natural double-buffering).
- **Unit gates: 34/34 PASS** (4 new in `tests/test_restage_prefetch.py`:
  begin/commit identity under interleaved compute; single-gate-stage order bitwise
  == legacy 3-stage; flag/guard behavior — plus all prior suites green).
- Gate chain (chain11) RUNNING: SMOKE → **128k headline pair** → 32k pair →
  dense-32B pair (the largest measured target: ~50 s/step exposed with headroom).

### 2026-07-18 — P1 (R5 prefetch) gate chain VERDICTS (chain11, same-day adjacent pairs)
| pair | ref | ON | Δ | C | G | verdict |
|---|---|---|---|---|---|---|
| SMOKE 8k×b4 (6 st) | 14.15 | 13.67 | −3.4% | flat | flat | PASS, losses in envelope |
| **30B@128k×b8** | 602.47 (C-631.7, G-180.2) | 602.09 (C-631.6, G-176.3) | **−0.38 s ≈ NEUTRAL** | flat | −3.9 | **honest G-bound cell**: guard no-ops (gap 49.64→49.60 s/step — nothing engaged in steady state); prefetch cannot buy bandwidth at G≈180/186 |
| **30B@32k×b8** | 88.11 (C-351.6, G-54.2) | **85.57** (C-351.5, G-57.7) | **−2.54 s = −2.88%** | flat | +3.5 (held stages, by design) | **PASS — NEW 32k BEST 85.57 s** (cumulative −5.6% vs the 90.69 pre-item-6 best) |
| 32B@32k×b8 dense | 373.74 (C-528.5, G-107.7) | 369.54 (C-528.7, G-121.7) | −4.2 s = −1.12% | flat | +14.0 | PASS but **PARTIAL engagement** (see below) |
- **@32k mechanism=effect closure**: exposed gap 4.20 → **1.82 s/step** — the moe
  silu-bwd stage class (3×0.71) fully collected by early-issue+single-gate-stage AND
  the SDPA-unpack class (0.95+0.12) fully collected by the region prefetch on the
  second stream; Δexposure −2.38 s/step ≈ the −2.54 s e2e delta. Losses ≤0.31% all
  pairs; dense step-0 loss BIT-IDENTICAL across arms (prefetch never touches values).
- **Dense partial-engagement diagnosis**: gap only 58.47→54.26 s/step and legacy
  silu-bwd tags persist in the ON arm — the guard demand (extra=4×gate bytes ≈48.8
  GiB + 16 min-free ⇒ needs free ≥64.8 GiB) FLAPPED against free ≈64–78 GiB during
  backward transients (G peak 121.7): some layers prefetched (G +14, −4.2 s/step),
  many ran legacy. Fix queued: demand 2× (stages only; the kept dgrads replace
  same-size offload buffers) + `ASYM_PREFETCH_MIN_FREE_GB=8` re-gate arm (b32_r5b).
- **P1 VERDICT: ✅ shipped+gated; WIN @32k (−2.9%, new best 85.57), WIN-partial on
  dense (−1.1%, guard tuning queued), honest NEUTRAL @128k (G-bound — the 49.6
  s/step residual there is bandwidth/regime cost, not schedulable latency).**

### 2026-07-20 — P2 IMPLEMENTATION SPEC (rope/SDPA-operand recompute — next build, precise)
Evidence: rope backward is INPUT-FREE with frozen cos/sin (grad_q = g⊙cos +
cat((g⊙sin)[D/2:], −(g⊙sin)[:D/2]) — reproduces autograd's cat/neg/slice chain
bitwise); the collectible class is SDPA's saved q_embed/k_embed (bf16 [B,H,S,D],
6.5 s/step exposed + 0.46 TB/step D2H @128k). Build:
1. `_RopeRecomputeFunction`: forward computes q_embed/k_embed under no_grad via orig;
   backward uses the exact manual chain above (unit gate: bitwise vs autograd);
   NO input offload of its own.
2. Composition: the qknorm Function attaches `(handle, module, shape)` to its OUTPUT;
   the rope wrapper reads it off its inputs and attaches to ITS outputs a recompute
   recipe `(x_handle retained, norm module, transpose meta, cos, sin, unsqueeze_dim)`.
3. The attention wrapper `_pack`: a tensor carrying a recipe attribute (explicit
   object-attribute chaining — NOT anonymous matching) is packed as a recipe handle
   (no bytes copied, x_handle refcount++); `_unpack` stages x (shared staged cache on
   the handle so the later norm backward reuses ONE H2D), recomputes norm→rope on
   GPU (bit-identical), returns q_embed. Removes q_embed/k_embed D2H+pinned+H2D.
4. Ownership: refcounted release-once helper on the x handle (consumers: recipe
   unpack, norm backward); release order SDPA-bwd → rope-bwd → norm-bwd is layer-
   internal and sequential.
5. Gates: unit bitwise (rope manual grad; recipe unpack == saved tensor) → SMOKE →
   same-day 128k headline pair (+32k) — chain13 after chain12.

### 2026-07-20 — chain12 VERDICTS: dense re-gate UNLOCKED (−11.9%) + P4 repeat settled
- **b32_r5b (dense 32B@32k, guard-tuned re-gate, same-day adjacent):** ref **373.39 s**
  (C-528.5, G-107.7) vs prefetch-ON **328.97 s** (C-512.7, G-121.7) ⇒ **−44.42 s =
  −11.9%** — the 2×-demand guard fix (+min-free 8) unlocked full engagement of the
  dense mechanisms (early-issue + single-gate-stage + dgate/dup kept on-GPU). C also
  −15.8 GB. **NEW dense best 328.97 s** — cumulative vs the 385.4 same-day ref class:
  **−14.6% (item 6 + P1)**. P13's default-on case now has all three cells: 32k −2.9%,
  dense −11.9%, 128k honest-neutral (G-bound).
- **e128k_rep (P4 expert-grad repeat @128k on the item-6 stack):** deposits-ON 603.34
  (C-631.6) vs deposits-OFF 605.87 (C-629.7) ⇒ **−2.53 s = −0.42%** — REPRODUCES the
  original −0.4% RS-2 cell (🟡 noise question settled: consistent sign and magnitude
  across stacks/days; P2.moe_wgrad_deposit stays ON at 128k).

### 2026-07-22 — byte-diet round CLOSED: ❌ guard-starved at 128k (default-OFF)
| pair | ref | ON | Δt | ΔC | ΔG | mechs engaged | verdict |
|---|---|---|---|---|---|---|---|
| 128k b8 | 600.71 | 601.91 | +1.2s (+0.2%) | 614 flat | 175→179 (+4) | 7 of 960 (up 2, act 2, gate 1, silu 2; guard_denied 953) | ❌ neutral at +4G |
| 32k b8 | 85.52 | 85.63 | +0.11s | flat | flat | 0 attempts (CPU-act + P13 own the paths) | ✓ truly inert |
| smoke 8k ×2 pairs | 13.78/14.26 | 13.69/13.91 | noise | — | — | bd: 0, bd2: engaged | ✓ trajectory in envelope |

**Root cause the projection missed (counters, not hypothesis):** the removed copies were
NOT already-overlapped — the ON arm's restage census still shows the full exposed waits
(total_exposed 237.14 s vs ref 237.13 s per 5 steps; gate_for_act 23.97 s vs 24.08, act_for_
down_base 23.88 vs 24.08, silu-bwd pair 26.2+38.4 vs 26.4+38.1). The mechanisms simply
never ran: the per-mechanism G-guard (min-free 8 GiB + hold 12.6–25.2 GiB) was denied
953/960 times. At G-peak 176/184 GiB there is no persistent ≥20 GiB free window anywhere
in the 128k recompute/backward timeline; the 7 engagements were step-boundary transients.
+4 ΔG = those holds raising reserved peak; +1.2 s = pair noise. A looser guard is rejected:
free dips to ~0.2 GiB mid-backward at 128k (observed directly), so engagement = OOM risk.
Verdict: **default-OFF permanently at 128k-class; P15 stays False.** Mechanisms are sound
(19 bitwise unit tests; engaged calls corrupted nothing) and DORMANT, not dead — any
config with real headroom (2-GPU sharding, smaller model) auto-engages via the guard.

**Parity-gate recalibration (clean data):** two CLEAN runs of the same ref arm differ at
step 1 by 1.5e-3 (1.3402387 vs 1.3417232) — route101-accfp32 atomic scatter makes
cross-process bitwise parity unachievable on this workload. Correctness gate = unit
bitwise suite + clean-pair trajectory envelope. (The contaminated run-2 numbers are void.)

**FINAL 128k RESIDUAL STATEMENT (closes the byte-lever campaign):** after everything
shipped (norm/qknorm recompute, P2 wgrad deposit, P13 prefetch, K-2/K-4/K-9, and this
round's ❌), the 128k step = 600.7 s carries ≈47.4 s/step (7.9%) of exposed H2D restage
spread over ~20 tags, largest single tag 7.7 s/step (gate_for_silu_bwd_dup). No
hold-based scheduling lever can engage at G≈176/184. The only remaining lever classes
are out of byte-scheduling scope: (1) quantized (fp8/int8) staging — halves the bytes,
changes numerics, needs a quality gate; (2) 2-GPU sharding — halves resident bytes so
existing dormant mechanisms engage; (3) hardware H2D bandwidth. Three consecutive
neutral/negative 128k levers (rope-fusion, per-socket pools, byte-diet) ⇒ the regime is
**measured-saturated under the current 1-GPU bf16 architecture**.

### 2026-07-21 — byte-diet round: execution notes (mid-flight)
- Implemented all four mechanisms + P15 + per-mech counters/ENGAGED marker; units 19/19
  (3 new in tests/test_moe_direct_reuse.py, bitwise op-sequence identity proven).
- **BUG CAUGHT BY SMOKE + fixed**: mech-3's first cut did `F.silu(gate, inplace)` on the
  D2H SOURCE — `manager.offload` copies on a side stream, so in-place mutation races the
  in-flight copy and corrupts the CPU gate bytes backward host-reads. Fix: candidate-keep
  at the gate block, COMMIT only after the act-path decision (keeps CPU-act regimes
  allocation-identical to flag-off), and out-of-place silu (demand 2x in the guard).
  Rule now recorded: NEVER write in-place to any tensor that is an in-flight offload
  source; direct-reuse must be read-only or out-of-place.
- **Ops lesson (zombie kill)**: stopping a ladder with pkill on wrapper cmdlines left the
  128k TRAINER alive (0 GiB at kill time -> 173.5 GiB later), which OOM'd smoke_bd2_ref
  and contaminated the smoke_bd rerun (2x slower, both arms). Discipline: kill the PIDs
  from `nvidia-smi --query-compute-apps=pid` and re-check until empty.
- Parity-gate recalibration pending clean data: route101-accfp32 scatter uses atomics, so
  cross-process bitwise loss parity may be unachievable on this workload; the correctness
  anchor is the unit bitwise suite + clean-pair trajectory envelope. To be confirmed by
  the clean chain14 rerun (launched 19:13:57).

### 2026-07-21 — ACCEPTED ROUND: "128k moe-backward byte-diet" — EXECUTABLE SPEC (handoff)
Refined from the code (qwen3_moe_finegrained.py fg forward ~lines 960-980, GPU act
path used at 128k): gate/up/act are all GPU-BORN and offloaded, then RESTAGED within
the same region — the roundtrips are lossless, so DIRECT-REUSE of the GPU tensor is
bit-identical BY CONSTRUCTION. One flag `ASYMM_MOE_FG_DIRECT_REUSE` (default off,
policy P15, per-mechanism G-guard via prefetch_free_ok), FOUR sub-mechanisms ranked
by hold-window risk (each +12.6 GiB @128k during its window):
1. **up-direct** (hold across ~5 lines, no GEMM between offload and the mul):
   skip `manager.stage(up_cpu, "moe.up_for_act")`; use the GPU `up` tensor in
   `gate_stage.mul_(up)`. Kills 562.5 GiB/step ≈ −4.8 s/step exposed.
2. **act-direct** (hold across the down_lora block): skip
   `stage(act_cpu, "moe.act_for_down_base")` (and the K-9 carried-stage variant);
   pass `act_gpu` straight to the down base GEMM. act D2H STAYS (the dA deposit
   host-reads act_cpu). Kills 562.5 GiB/step ≈ −4.8 s/step.
3. **gate-direct** (hold across the up GEMM — riskiest window, coincides with the
   up GEMM's own peak): kills gate_for_act, 562.5 GiB/step ≈ −4.8 s/step.
4. **silu-bwd single-gate-stage at 128k**: P13's mechanism with a 128k-sized guard
   (hold ONE stage, not two: demand 12.6 not 25.2) — kills gate_for_silu_bwd_dgate
   ≈ −5.3 s/step.
G-budget: current 128k G = 176.3 (P2-ON) ⇒ ~9 GiB slack + transients. Mechanisms
engage independently under the guard — at minimum (1)+(4-partial) fit; measure G
after each. Full diet target −19 s/step; realistic first gate −9..−14 s/step
(−1.5..−2.3%). Gates: unit (direct-use bitwise == restage path — trivial by
construction but assert), SMOKE, same-day 128k headline pair, 32k no-regression
pair, then policy encoding (P15 ON at 128k-class iff guard; inert at 32k where the
roundtrips are already cheap/absorbed).
**Handoff state: spec final; no code written for this round yet (session context
boundary reached after the three prior rounds closed). All prior rounds' code,
gates, and docs are complete and green (56 tests; one filed cross-suite ordering
issue). Next session: implement mechanisms 1→4 in the listed order, one guard, one
ladder.**

### 2026-07-21 — mandate round N+1: per-socket pools ❌, BFMMLA closed, 128k residual reassessed
- **(1) Per-socket act-pool placement: ❌ CLOSED by micro NEGATIVE (no e2e needed).**
  Pinned H2D from socket-0 memory = **211.4 GB/s** vs from socket-1 = **125.3 GB/s
  (−41%)**; D2H 193.3 vs 124.5 (numactl --membind subprocess, 2 GiB, 5-run medians).
  The GPU reaches socket-1 pages across the Grace–Grace link at ~60% of local
  bandwidth — every restage moved to socket 1 pays 41% MORE transfer time, and the
  single-socket CPU kernels (P14: streaming kernels must stay socket-local) would
  add cross-socket reads on top. The premise (usable extra bandwidth for the
  restage class) is defeated at the fabric level. Recorded; do not re-open without
  a data-placement design that keeps GPU-facing pools socket-0-resident.
- **(2) BFMMLA repack-first: CLOSED without build (measured-value argument).**
  chain13a proved kernel speed does not move e2e for this class: +45% wgrad
  (1.96→2.85 TF/s) ⇒ e2e 85.50 vs 85.58 (neutral; the job is hidden in the
  deposit window). A further ≥1.3× micro would be equally e2e-nil; the item was
  paper-optics-only by its own spec and the campaign's impact rules rank it last.
  The 2.85 TF/s @96T stands as the kernel-quality number; the KleidiAI repack
  recipe stays recorded if roofline optics are ever wanted.
- **(3) 128k residual reassessment (current default stack, e128k_p2_ref leaf):**
  total exposed restage wait **47.65 s/step = 7.9%** of 600.3 s. Attribution:
  moe stage family **34.4 s/step** (gate×3 17.9 + gate/up/act-for-fwd 14.5 +
  X_for_dA 1.1; 3,375 GiB/step at ~84 GB/s effective vs 211 clean = 2.5×
  contention with concurrent D2H + TMA on socket-0 LPDDR), bf16 SDPA unpack 6.4
  (collected by P2-ON at 128k per the adoption gate), qknorm restage 3.2.
  **Next-round proposal (data-driven): "128k moe-backward byte-diet"** — reorder
  to stage gate+up ONCE per layer backward and (a) recompute act on-GPU for
  down_base (kills act D2H+H2D, 1,125 GiB/step) and (b) reuse the stages for
  silu-bwd (kills the 2nd/3rd gate stages, 1,125 GiB/step) — total −2,250
  GiB/step ≈ −19 s/step exposure IF the two held stages (2×12.6 GiB) fit: G is
  176.3 post-P2 (−4 GiB earned) ⇒ needs ~9 GiB more headroom — pair it with an
  ohbm/act-budget trade or accept partial (one held stage = half the diet).
  Second candidate: widen the P2 tokens-gate study (does rope-ON at 64k-class pay
  C with time-neutrality?).

### 2026-07-21 — KNOWN TEST-ORDERING ISSUE (filed, not a production regression)
- `pytest tests/test_save_dedup.py tests/test_pinned_ledger.py` (that ORDER, one
  process) fails `test_ds_slots_denial_still_functional_cuda` on a content-equality
  assert of the unpinned DENIAL-FALLBACK slot; every suite passes alone and in the
  other pairings tried (restage+ledger, qknorm+ledger, placement+ledger, and the
  5-suite combo without save_dedup). One fix attempt (clearing the module-global
  `_DS_SLOTS` cache in the ledger test reset) did NOT cure it — reverted. Suspected
  mechanism: cross-suite global state from the dedup suite (pinned-pool buffer or
  stream/event state) aliasing the fallback slot's copy. Affects the test HARNESS
  combination only (production never runs both lifecycles in one process). Next
  session: bisect which dedup test arms it; until then run the ledger suite in its
  own process (as its own docstring already instructs).

### 2026-07-21 — P2 gate VERDICTS (chain13b, same-day adjacent pairs): **TIME-NEUTRAL, HOST-MEMORY WIN — a MEMORY feature**
| pair | ref | rope-ON | Δt | ΔC | ΔG |
|---|---|---|---|---|---|
| SMOKE 8k | 13.69 | 13.94 | noise | ≈ | ≈ |
| 30B@128k | 600.32 (C-631.7) | 602.48 (C-613.7) | +0.36% (noise) | **−18.0 GB** | −1.0 |
| 30B@32k | 85.60 (C-351.6) | 86.88 (C-347.0) | +1.5% (noise edge) | −4.6 GB | −2.1 |
| 32B dense | 330.08 (C-512.5) | 332.38 (C-503.5) | +0.7% (noise) | **−9.0 GB** | −1.5 |
- All 8 arms PASS markers; losses in envelope. **Framing (per the honest read): P2's
  D2H savings were evidently already-overlapped copies** (consistent with the prefetch
  round having absorbed the exposed waits) — the collected class pays in HOST MEMORY,
  not time. At 32k the +1.5% sits at the noise edge with a plausible mechanism (the
  recipe rebuild = norm+rope recompute inside the SDPA-backward path where the class
  is small and C is not binding) — left OFF at 32k without spending a repeat pair.
- **ADOPTION ENCODED (policy P12.rope_recompute): ON where C is the binding
  resource** — dense-class (any length) + MoE per-call tokens ≥
  `ASYM_POLICY_ROPE_MIN_TOKENS` (default 524,288 ⇒ ON at 128k-class, OFF at 32k).
  Env `ASYMM_ROPE_RECOMPUTE` force-arms. Suites 25/25 after the encoding.
- Remaining queue handoff (specs in this log): per-socket act-pool placement
  experiment (the other lever on the 36 s/step moe-stage residual — requires NUMA-
  aware pool allocation + an e2e pair), BFMMLA repack-first micro (bar ≥1.3× vs
  2.85 TF/s or close the item), then the 128k residual reassessment.

### 2026-07-21 — P2 (rope/SDPA-operand recompute) BUILT — with a major design simplification
- **Simplification found during build: NO custom rope Function is needed.** Rope's
  backward is input-free (frozen cos/sin) and rope saves nothing — so the autograd
  graph is left completely UNTOUCHED (gradients bitwise-identical trivially). The
  wrapper only ATTACHES RECOMPUTE RECIPES to the rope outputs (q_embed/k_embed —
  exactly the bf16 [B,H,S,D] tensors SDPA saves): the attention pack stores the
  recipe instead of copying bytes (−D2H, −pinned pool), and unpack rebuilds the
  tensor on-GPU from the norm wrapper's already-offloaded bf16 x (stage → exact
  norm forward → transpose → q*cos + rotate_half(q)*sin — same device, same op
  chain ⇒ bit-identical). Explicit object-attribute chaining (`_asym_qknorm_src`
  on the norm output → `_asym_rope_recipe` on the rope outputs) — never anonymous
  packed-list matching. `_SharedXHandle` refcounts the x handle across its two
  consumer classes (norm backward + recipes).
- Expected @128k: −0.46 TB/step of D2H (q_embed 8.6 + k_embed 1.07 GiB per layer)
  + pinned-pool relief, in exchange for GPU recompute (~0.1 s/layer) + a second
  H2D of x per recipe (v1: no staged-x sharing — recorded as v2 if G allows).
- Old evidence-off input-offload rope Function REMOVED (superseded).
- **Unit gates: 34/34 PASS** incl. the new recipe test (rebuild bit-identical to
  the saved tensor; graph untouched; shared-handle refcount drains to 0).
- chain13b RUNNING: SMOKE → 128k headline → 32k → dense, full default stack
  (`ASYM_PLACEMENT_POLICY=1` alone = item6+prefetch+wgrad96) ± `ASYMM_ROPE_RECOMPUTE=1`.

### 2026-07-21 — kernel campaign continued + DEFAULT-ON CLOSURE
- **(1) wgrad-threads e2e ride (chain13a, same-day pair @30B·32k, full stack ±
  `ASYM_CPU_OPS_THREADS_WGRAD=96`): ref 85.50 vs ON 85.58 ⇒ e2e-NEUTRAL** (±0.1
  noise; deposits off-critical-path as predicted — the +45% micro is hidden-tail
  headroom, valuable for longer-context windows). **KEPT: default-on under policy**
  (cpu_ops.wgrad_threads → 96 when `ASYM_PLACEMENT_POLICY=1`; env override wins).
- **(2) silu FDIV-kill: ❌ NEGATIVE, reverted per protocol.** FRECPE+2-Newton vs
  FDIV: fwd 53.9→63.9 ms @2.1M (+19%), 211.6→254.7 @8.4M (+20%) — the dependent
  5-op estimate chain loses to the well-hidden FDIV in this bandwidth-bound kernel;
  AND parity broke (max 917 bf16 ulp, 0.076% >1 ulp, saturated-negative tail).
  Variant removed; negative result annotated at the kernel site; parity re-green.
- **(3) BFMMLA repack-first + (4) P2 rope/SDPA recompute: DEFERRED to next round
  with full specs recorded** (BFMMLA: research digest has the KleidiAI 8×12/2×4
  repack pattern + kt-kernel discipline; bar ≥1.3× vs the NEW 2.85 TF/s baseline.
  P2: complete implementation spec in the 2026-07-20 entry; gates = chain13b).
- **(5) DEFAULT-ON CLOSURE APPLIED (2026-07-21): P12.qknorm_recompute=True,
  P13.restage_prefetch=True, wgrad-threads=96 — all under `ASYM_PLACEMENT_POLICY=1`
  alone.** Gate evidence in each rule's docstring; the gate-arm unit test updated to
  the new default; **all suites 47/47 green**. placement.md P12/P13/P14 + cpu_compute
  ADOPT list updated. New production defaults (policy-only flags): 30B@32k ≈85.6 s,
  30B@128k ≈596 s, dense 32B ≈329 s — all lossless.

### 2026-07-21 — KERNEL CAMPAIGN round 1 (per-component; clean-window 5-run medians; kernel | variant | before | after | verdict)
- **Baseline sweep (threads × socket-binding, zero code changes; production shapes):**
  | kernel | binding/nt | before (prod 48T) | best measured | verdict |
  |---|---|---|---|---|
  | swiglu fwd [8.4M,768] | socket0 nt=72 | 219 GB/s | **287 GB/s (+31%)** | 57% STREAM; matters only if P3 re-enables CPU-act @128k (act auto-off there today) — recorded as enabler |
  | swiglu fwd [2.1M,768] | 48T | 210–228 GB/s | ≈flat across nt | already saturated at production shape |
  | swiglu bwd | 48T | 268 GB/s | 48T best (higher nt degrades) | keep 48 |
  | **wgrad g/u [2.1M,64]×2048** | threads spread ×96, data socket-0 (production layout) | 1.96 TF/s | **2.85 TF/s (+45%)**; 3.24 @144T w/ interleaved data | **KEEP — env `ASYM_CPU_OPS_THREADS_WGRAD` wired at all 3 deposit sites (default-off)**; rides next e2e |
  | rmsnorm [8.2M,128] | 48T | 279 GB/s | 48T best pre-variant | see hoist below |
  | widen+sqsum | 48T | 310 GB/s = 62% STREAM | 48T best | below the 70% stop-bar; one attempt allowed later |
- **Two-socket question (P3 premise) ANSWERED with data:** streaming kernels DO NOT
  scale across sockets (interleave 238–243 vs socket-local 287 GB/s — NUMA-remote
  caps; matches the research: near-2× needs per-socket first-touch, never interleave);
  the COMPUTE-bound wgrad DOES scale (2.85 TF/s threads-spread/data-home; 3.24 with
  interleaved operands = 1.65× vs prod-48T ≥ the 1.6× bar). ⇒ P3's placement-flip
  case holds for the WGRAD half only; the CPU-act flip would need per-socket data
  placement surgery (pool-level, recorded, not built).
- **rmsnorm variant 1 (csrc): hoist the per-8-lane scalar even/odd weight
  deinterleave out of the row loop** — before 15.0 ms / 279 GB/s → after **11.8 ms /
  355 GB/s (−21%, 71% of STREAM)** at [8.2M,128] 48T; [32.8M,128]: 59.8 ms @72T.
  Bit-identical by construction (same values, same lanes); parity 6/6 + ulp-contract
  green post-rebuild (`MAX_JOBS=8 build_ext --inplace`). **KEEP.** (Kernel is
  currently an enabler, not in the default production path — no SMOKE surface.)
- **Research digest (subagent, kt-kernel + Neoverse-V2 SWOG + KleidiAI/ACL + NUMA)**
  — queued variants, ranked: silu FDIV-kill (svrecpe+Newton; V2 FDIV is V02-only —
  likely the real 45%-STREAM limiter; bf16 output rounding should absorb the est
  error — parity-gate it), FEXPA exp, prefetch-distance sweep 256B–1KB + dual-level
  L1STRM/L2KEEP (SWOG gives no numbers; HW prefetchers do most), NT-stores SKIP
  (Grace auto write-allocate-evasion makes STNT1 moot — research-closed, no test
  needed), wgrad BFDOT with vector-ZIP pre-paired dS (replace scalar u32 packing at
  cpu_ops.cpp:389-396), BFMMLA 2×4-tile repack (KleidiAI 8×12 pattern; repack-first
  this time), kt-kernel tricks: dual pre-transposed weight copies, 8-row M-unroll,
  per-thread accumulators + tree reduce, padded tails. kt-kernel has NO prefetch/NT/
  BFMMLA — our prefetch already exceeds it.

### 2026-07-20 — MODULE MICROBENCH — FINAL (mandate): definitive per-module table landed
- Full 9-row table (per-invocation ms at production shapes @32k AND @128k + per-step
  aggregates + production-arm WHY) written to cpu_compute.md as "MODULE MICROBENCH —
  FINAL"; rerunnable via `tests/bench_modules.py --final`. Key numbers: SwiGLU
  fwd/bwd ours-CPU 4.2×/5.4× vs PyTorch-CPU (44.8/58.9 ms @2.1M); adapter-grad CPU
  61→189 ms (attn) and 334/117→1338/463 ms (expert g-u/down) vs GPU-remote 2–8×
  worse and copy+GPU 5–42 ms isolated-best (production stays CPU-deposit for
  window/link reasons — K-2b + R5 counters); norm recompute (R2) ~6× cheaper than
  the replaced fp32 roundtrip at BOTH scales (1.29 vs 4.48 s/step @32k; 5.2 vs 18.0
  @128k — matches the −22 s e2e); boundary+restage prefetched-ahead ≈ 0 exposed vs
  4.9–61 ms waited (e2e-verified @32k: 4.20→1.82 s/step); widen+sqsum fused 13.3×;
  dedup shared-handle ~free vs 0.24 ms dup pack (variance class only ⇒ e2e-neutral,
  as shipped). Boundary-128k pageable cell was an async mis-read (noted in the doc);
  all other cells clean-window measured this morning.
