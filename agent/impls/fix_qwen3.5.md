# Fix Qwen3.5: restore the asym memory win at `q3.5-35b-a3b | 80000|8|1`

Owner doc for the qwen3.5 memory-parity problem. Prior art (read before re-deriving
anything): `agent/impls/archive/fix_finegrained_qwen3.5_moe.md` (fg/ker101 bring-up,
FA4 port, S² mask root cause, fla >70k fault), `archive/fix_qwen35.md` (leaf+grad
offload-skip fix), `archive/fix_qwen35_v2.md` (old-generation gap analysis),
`agent/math/linear_attn_math.md`. Rules: `agent/RULES.md` (latency protocol, C/G
reporting) — those override anything here.

## 1. GOAL

On every qwen3 model driven by `scripts/lf/profile_lora_lf_test_both.sh`
(q3-30b-a3b, q3-32b, llama3.3-70b, q2.5-32b/72b, …) the asym config

```text
A (target):   asym_cpuadamwds | recomp-off-full-fg(-ker101-ceil0000-ohbm0) | ligerloss1
B (baseline): superoffload_mem | unsloth-off-ohbm0                         | ligerloss1
```

gives A a MUCH lower peak HBM than B at matched workload (e.g. q3-30b max-seq
173k+ for A vs 131k for B; q3.5 s45000 on the 2026-07-03 SDPA stack: A 90.3 GiB
vs B-family 160.2 GiB, −43.6%). On qwen3.5 (`q3.5-35b-a3b`, FA4 stack) the last
observations showed A ≈ B (~106 GiB both at s80000 diagnostics, 2026-07-03).
The code has moved since (SFT-38 merge, ceiling-search split, etc.) and **this
checkout has ZERO q3.5 artifacts** (verified 2026-07-13) — the current situation
is unknown and must be re-established first.

**Target workload:** `q3.5-35b-a3b | 80000|8|1 | ligerloss1`, policy
`none|false|false|false|false|false`, 1 GPU (GB200 SM100, 184 GiB card).

**Success criteria (all required):**

1. **G1 (memory win):** A's peak reserved HBM (`G`) ≤ **0.80×** B's `G` at
   `80000|8|1` — or, if B G-OOMs there, A completes with a clean audit (§7) and
   the B crash ceiling is reported as the baseline row. Parity (within ~5%) =
   goal NOT met → keep iterating (§6).
2. **G2 (validity):** the A run passes the config audit (§7): label
   `recomp-off-full-fg-ker101-ceil0000-ohbm0`, `moefg` wrapped 40/40, wrapper
   counts 40/30/10/40, zero fallbacks, loss in band, no fla fault/NaN. A win
   built on a hollow or numerics-broken run does not count (that was the
   2026-07-02 failure mode).
3. **G3 (non-regression):** qwen3-30b spot band unchanged — s20000 loss
   1.775 ± 0.05; s80000 ker101 `step_H` 80,521 MiB ± 3% (grad_norm there is the
   Known Systemic Issue; record, don't gate). Required after ANY shared-code
   edit (`qwen3_moe*.py`, `activation_offload*.py`, wrappers, LF fork).
4. **G4 (report):** results land in the §8 table (RULES.md format: steady `lat`
   from 4 measured steps dropping first/last, `C` = peak host RSS GiB,
   `G` = peak reserved HBM GiB), plus a one-line summary per run in the form
   `<lat>s, C-<ram>, G-<hbm>, <next-boundary> [DONE|IP]`.

**Iteration mandate:** if after any run the goal is not met, do NOT stop and do
NOT just report the table. Mine the run's artifacts (§7 list), form a specific
hypothesis about where A's HBM goes that B's doesn't (or why A's offload isn't
engaging), check the code paths involved, fix, re-run the gates, and append the
new row + verdict to §8. Repeat until G1–G4 hold or a blocker is proven and
documented with evidence (e.g. an un-offloadable kernel-workspace floor — then
quantify it and state the best achievable margin).

## 2. Validation scripts (the only entry points)

```bash
# Scoreboard / profiling driver (auto-switches qwen3.5 to the FA4 runtime):
scripts/lf/profile_lora_lf_test_both.sh        # RUNS env, PROFILERS=source for timing rows
# Env (re)build:
scripts/lf/bootstrap_lf_venv_fa4.sh            # .venv-fa4; INSTALL_CAUSAL_CONV1D=1 default
# Probes / gates:
scripts/testing/fla_gdn_longseq_repro.py       # fla chunk_gated_delta_rule >70k fault repro
scripts/testing/qwen35_fg_numeric_probe.py     # fg/ker101 numerics vs fp32 (add --zero-b, --qwen3 --tokens 655360)
scripts/lf/validate_lf_memory_capacity_schema.py
# Reporting:
scripts/lf/show_metrics.py <OUTPUT_ROOT>       # step_H / RAM / loss table
scripts/testing/collect_qwen35_scoreboard.py   # qwen3.5 scoreboard collector
```

Runtime resolution: `resolve_current_runtime_for_model` in the both-driver
auto-sets `ENV_DIR=.venv-fa4`, `FLASH_ATTN=fa4`, canonical `LF_DIR` for any
qwen3.5 model when `ASYM_QWEN35_FA4_AUTO=1` (default). Do NOT pass an explicit
`ENV_DIR`/`FLASH_ATTN`/`LF_DIR` for scoreboard rows — explicit values disable
the auto-switch guards. Artifact paths must NOT be trusted to prove the runtime;
check `command.txt` (env_dir, `FLASH_ATTN=fa4`) and no `attnsdpa` label leak.

## 3. Phase 0 — environment gate (blocking; verified state as of 2026-07-13)

Facts measured in this checkout on 2026-07-13:

- `.venv-fa4`: torch 2.12.0+cu130, transformers 5.6.0, flash-attn-4 4.0.0b16,
  fla/fla-core 0.5.0 — `import causal_conv1d` initially FAILED (not installed);
  **FIXED same day**: `causal_conv1d==1.6.2.post1` installed, and
  `is_causal_conv1d_available / is_flash_linear_attention_available /
  is_flash_attn_4_available` now all print True. Historical q3.5 runs made
  before 2026-07-13 ran WITHOUT causal_conv1d in this venv.
- canonical `.venv` always HAD causal_conv1d. Pre-fix qwen3.5 runs (which
  auto-switch to `.venv-fa4`) silently took the transformers **torch fallback conv path**
  (`modeling_qwen3_5.py` imports `causal_conv1d_fn` only
  `if is_causal_conv1d_available()`, else `torch_causal_conv1d_*`). This is both
  a perf and a memory suspect (fallback saves fatter conv activations) — and a
  possible contributor to the parity. fla 0.5.0 also ships its own
  `fla.modules.conv.cuda` fast path; which one fires must be recorded per run.

Gate (run before ANY scoreboard row):

```bash
# 1. Fix the venv — DONE 2026-07-13 (causal_conv1d==1.6.2.post1 installed; if the
#    env is ever recreated, the bootstrap installs it by default):
#    TORCH_CUDA_ARCH_LIST=10.0 .venv-fa4/bin/pip install --no-build-isolation causal_conv1d==1.6.2.post1
#    or: RECREATE_ENV=0 bash scripts/lf/bootstrap_lf_venv_fa4.sh
# 2. Verify the triple (re-check at the start of every session):
.venv-fa4/bin/python - <<'PY'
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available
from transformers.utils import is_flash_attn_4_available
print("causal_conv1d", is_causal_conv1d_available())
print("fla", is_flash_linear_attention_available())
print("fa4", is_flash_attn_4_available())
PY
# all three must print True
# 3. fla long-seq probe (expect: clean ≤70000; RECORD the exact fault boundary):
CUDA_VISIBLE_DEVICES=<free> .venv-fa4/bin/python scripts/testing/fla_gdn_longseq_repro.py
```

## 4. Phase 1 — re-establish the current situation (the re-profile)

Known blocker to plan around: **fla `chunk_gated_delta_rule` illegal memory
access at S≥75000 (B=8, qwen3.5-35B head shapes) on fla 0.5.0**; in-model it can
silently corrupt (loss→0, NaN grads) instead of faulting. It hits ALL backends
equally. **RE-CONFIRMED 2026-07-13** on the completed `.venv-fa4` (causal_conv1d
installed): probe clean at S=2048/32768/60000/65600/70000, CRASH (Triton illegal
memory access) at S=75000 — boundary unchanged from 2026-07-03; causal_conv1d
does not touch this kernel (Phase-2 route 2 answered: no). Therefore the ladder has numerics-valid anchors below the fault line
and the target row above it:

| rung | seq | purpose |
|---|---|---|
| L1 | 45000 | numerics-valid anchor; direct comparison to the 2026-07-03 result (A 90.3 vs B 160.2 GiB on SDPA — FA4 numbers WILL differ, that's the point) |
| L2 | 70000 | highest known-clean fla seq; primary comparison if 80000 stays blocked |
| L3 | 80000 | THE TARGET; valid only after Phase 2 clears the fla fault (else record blocked state + memory numbers labeled `numerics-invalid`) |

One run at a time on the whole node (asym rows take 380–800 GiB host RSS; wrap
heavy rows in `numactl --membind=0,1 --cpunodebind=0,1`). Per RULES.md:
`MAX_STEPS=4 WARMUP_STEPS=1`, drop first+last measured, mean of middle 2 = `lat`.

```bash
cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
# Row A (target) — sub in 45000 / 70000 / 80000:
GPU_POOL=<free> PROFILERS=source MAX_STEPS=4 WARMUP_STEPS=1 OVERWRITE=true PLOT=false \
OUTPUT_ROOT=$PWD/profiling_results/profiling_fix_qwen35_repro \
RUNS='q3.5-35b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 45000|8|1 ; none|false|false|false|false|false' \
bash scripts/lf/profile_lora_lf_test_both.sh
# Row B (baseline):
GPU_POOL=<free> PROFILERS=source MAX_STEPS=4 WARMUP_STEPS=1 OVERWRITE=true PLOT=false \
OUTPUT_ROOT=$PWD/profiling_results/profiling_fix_qwen35_repro \
RUNS='q3.5-35b-a3b|1 ; superoffload_mem|unsloth-off-ohbm0|ligerloss1 ; 45000|8|1 ; none|false|false|false|false|false' \
bash scripts/lf/profile_lora_lf_test_both.sh
```

Notes: put the UNSUFFIXED `recomp-off-full-fg` in RUNS — the harness auto-default
(`qwen3_moe_routed_auto_default`) resolves ker101 + the fg env set
(`ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD=1`, `FG_LORA_A_FWD_GPU=1`, `FG_DA_GPU=1`,
`FG_KEEP_DGRADS_HBM=1`, 192 GiB pinned pool) for this model. Never pin `-ker000`
on a scoreboard row. Smoke first if anything looks off: same rows at `2048|8|1`,
expected loss 1.6–1.9, A peak ≈ 3.4 GiB class.

## 5. Phase 2 — unblock s80000 (only path to the literal target)

The fla fault is upstream-shaped; candidate routes, in preference order:

1. **fla upgrade**: ANSWERED 2026-07-13 for the current latest — fla/fla-core
   0.5.1 (newest on PyPI) still crashes the probe at S=75000; env rolled back to
   the 0.5.0 pin. Re-check PyPI for >0.5.1 at the start of each session; if a
   new release lands, upgrade, re-probe at S=75k/80k/81k, and bump the pins in
   `bootstrap_lf_venv_fa4.sh` only on a clean probe.
2. **causal-conv1d present** (Phase 0) — ANSWERED 2026-07-13: installed and
   re-probed; the S=75000 fault persists (different op, as expected).
3. **Sequence-chunked delta rule with differentiable state carry** — proven
   bf16-exact and clean at S=80072 in the 2026-07-03 diagnostics, but was
   **REVERTED BY USER POLICY** (execution-path trick baselines didn't carry).
   Re-proposing it is allowed ONLY as a model-level change that applies
   identically to A and B rows (both backends run the same mixer), with explicit
   user sign-off recorded here before any scoreboard use.
4. Upstream report/patch (attach `fla_gdn_longseq_repro.py`).

Until one lands: L2 (70000) is the honest headline row; the 80000 A/B memory
numbers may be recorded but must carry `numerics-invalid (fla >70k)` in §8.

## 6. Phase 3 — diagnose the parity, fix, iterate

If (as expected) A ≈ B on G at L1/L2, the win-restoring work starts. Mining
order per run leaf (`<config>/b8_s*/`):

1. `memory_actual_peak_breakdown.csv` — component attribution AT the peak.
   2026-07-03 s45000 reference: `linear_attention` workspace 37.3 GiB was A's
   top term (delta-net backward), i.e. the S-scaling floor NEITHER backend
   offloads. If it still dominates both A and B peaks, that IS the parity
   mechanism — the fix must attack it (see H1).
2. `memory_live_activation_details.csv` — top live tensors at peak (beware the
   known allocator storage-reuse mislabeling; confirm with a snapshot).
3. `memory_breakdown.jsonl` / `PROFILE_MEMORY_SNAPSHOT=true` re-run of the
   single interesting rung for allocator-level truth.
4. `train.log` runtime counter line — wrapper counts (audit §7) + offload
   byte counters (`total_d2h_offloaded_bytes`, skipped bytes): is A's offload
   machinery even engaging on the FA4 stack?
5. `command.txt` — env actually applied (fg env set present? pool size right?).

Ranked hypotheses (update in place as evidence lands):

| # | hypothesis | discriminating evidence | fix direction |
|---|---|---|---|
| H1 | fla delta-net bwd workspace (S-scaling, un-offloaded) floors BOTH backends → parity | breakdown: `linear_attention` temp ≈ top term in A AND B; grows ~linearly 45k→70k | chunk-level recompute or saved-state offload for the GDN mixer (old doc's L2 lever; qwen3.5-only wrapper, needs its own non-regression pass) |
| H2 | A's saved-tensor/offload wrappers not firing (or partially) under FA4 | counter line: wrapped counts < 40/30/10/40, or d2h bytes ≪ s45000-era; skipped-bytes high | wrapper predicate fix (cf. the archived leaf+grad skip bug — same smell) |
| H3 | torch-fallback conv path (missing causal_conv1d) inflates saved conv activations in A and B | Phase 0 install → re-run L1; Δ peak | keep causal_conv1d pinned in the venv (Phase 0) |
| H4 | B improved rather than A regressed (FA4 lean forward + unsloth-off save-on-cpu already strips what A's offload used to win on) | compare B's L1 G vs its own SDPA-era 160.2 GiB; decompose B's saved-act term | then the win must come from H1-class levers, not wrapper repair |
| H5 | reserved-vs-allocated gap (fragmentation/pool) masks a real allocated win | reserved−allocated ≫ 10 GiB in A only | allocator conf / pool sizing; report both numbers |

Loop: hypothesis → artifact/code evidence → smallest fix → Phase-1 rung re-run
(+ smoke) → G3 non-regression if shared code moved → §8 row + verdict. Do not
advance on an inconclusive run (label it per §7 and re-run correctly).

## 7. Run audit (every scoreboard row; inconclusive ⇒ fix and re-run, never quote)

Required clean-audit facts per run: artifact label matches
`recomp-off-full-fg-ker101-ceil0000-ohbm0` (A) / `unsloth-off-ohbm0` (B);
`qwen35_moes_wrapped=40`, `qwen3_moe_finegrained_offload_wrapped=40`,
`linear_attention_saved_tensor_offload_wrapped=30`,
`attention_saved_tensor_offload_wrapped=10`, `attention_act_offload_wrapped=40`
(A rows); `reference_fallback_count=0`; runtime = canonical LF + `.venv-fa4` +
`FLASH_ATTN=fa4` (from `command.txt`, not the path); loss band 0.9–1.2 at
s≥20000 (1.6–1.9 smoke); grad_norm finite (NaN/0-loss = fla signature ⇒
numerics-invalid); `jobs.tsv` status is NOT authoritative either way (known
checker caveat — verdicts come from train.log + memory.md + counters).
Labels: `validated | blocked_by_fla | inconclusive_wrong_config |
inconclusive_wrong_runtime | inconclusive_partial_profile | numerics-invalid`.

## 8. RESULTS (fill in; the deliverable table — RULES.md metrics)

Summary-comment form per run: `<lat>s, C-<ram>, G-<hbm>, <next-boundary> [DONE|IP]`.

| date | seq×B | row | config label | lat (s) | C (GiB) | G (GiB) | loss | grad_norm | audit | verdict vs GOAL |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| 2026-07-03 (hist, SDPA) | 45000×8 | A | recomp-off-full-fg-ker101 | — | — | 88.2 | 0.948 | 0.205 | validated | reference: −43.6% vs unsloth |
| 2026-07-03 (hist, SDPA) | 45000×8 | B' | unsloth (not -off) | — | — | 156.5 | 0.952 | 0.284 | validated | reference baseline |
| 2026-07-03 (hist, diag) | 80000×8 | A≈B | both ~106 GiB | — | — | ~103.5 | invalid | NaN | numerics-invalid | the parity this doc attacks |
| 2026-07-13 | 2048×8 (smoke) | A | ker101, FA4 | — | — | 19.1 | 1.688 | 0.22–0.25 | validated | pipeline gate only |
| 2026-07-13 | 2048×8 (smoke) | B | unsloth-off-ohbm0 | — | — | 19.2 | 1.681 | 0.28–0.41 | validated | pipeline gate only |
| 2026-07-13 | 45000×8 | A | ker101-ceil0000-ohbm0 | 329.1 | 434.3 | **71.4** | 0.893–0.952 | 0.12–0.23 | validated | **FAIL G1: A/B=1.18** |
| 2026-07-13 | 45000×8 | B | unsloth-off-ohbm0 | 277.3 | 404.5 | **60.5** | 0.837–0.950 | 0.12–0.29 | validated | baseline (FA4-lean) |
| 2026-07-13 | 70000×8 | A | ker101-ceil0000-ohbm0 | 405.6 | 639.4 | **106.3** | 0.821–0.888 | 0.12–0.21 | validated | linattn ws 77.0 GiB = 72.5% of peak |
| — | 70000×8 | B | SKIPPED (user: no more baseline spend) | | | ~92 est | | | | scaling estimate from 45k, not measured |
| — | 80000×8 | A/B | NOT RUN (user: stop at 70k; fla >70k broken) | | | | | | | |
| 2026-07-14 | 45000×8 | A-tuned | ker101 + `ASYM_CPU_ADAMW_ASYNC_GRAD_OFFLOAD=0` + both `*_SKIP_IN_BACKWARD=1` | 308.3 | 402.1 | **58.9** | 0.892–0.952 | 0.13–0.20 | validated | **beats B (60.5): −2.6%; gap fully closed** |
| 2026-07-14 | 70000×8 | A-tuned | same tuned env | 404.4 | 591.3 | **93.5** | 0.824–0.893 | 0.12–0.23 | validated | −12.8 GiB vs baseline A; ≈ B scaling est (~92) |
| 2026-07-14 | 70000×8 | B | unsloth-off-ohbm0 (user later requested the row) | 323.7 | 632.3 | **93.0** | 0.781–0.892 | ok | validated | tuned-A parity at 70k confirmed (93.5 vs 93.0) |
| **q3.5-122b-a10b** (16000×8) | | B | unsloth-off-ohbm0 | 302.4 | 644.2 | 47.9 | 1.00–1.14 | ok | validated | asym rows HOST-OOM at 16k (>1.12 TB vs node's 1.17; B needs 644 GiB) |
| **q3.5-122b-a10b** (8000×8) | | B | unsloth-off-ohbm0 | 267.5 | 583 | **36.8** (29.5 alloc) | 1.07–1.58 | ok | validated | |
| **q3.5-122b-a10b** (8000×8) | | A-baseline | ker101 (pool 64 GiB) | 278.6 | 791 | **44.0** (39.0 alloc) | same band | ok | validated | SAME issue at 122B: +19.6% reserved, backlog ≈ +18 GiB alloc |
| **q3.5-122b-a10b** (8000×8) | | A-tuned | + syncgrad + skip-flags | 268.2 | 764 | **36.7** (**20.7 alloc**) | same band | ok | validated | reserved parity; **allocated 30% BELOW B** — win grows with model size |

(Historical MiB values converted: 90300.8 MiB = 88.2 GiB, 160220.0 = 156.5,
~106000 ≈ 103.5. New rows: G = peak reserved from memory.md/profile.json; C =
`/proc/self/status` VmHWM as surfaced in the run summary.)

## 9. Bookkeeping

- Fresh `OUTPUT_ROOT` per phase (`profiling_results/profiling_fix_qwen35_repro`, `_unblock`,
  `_fix<N>`); never mix with skew/epstats trees.
- Before deleting/overwriting nothing; `OVERWRITE=true` only inside this doc's
  own output roots.
- Check `pgrep -f run_lf_lora_sft` before launching (driver lock degraded).
- Every code change: cross-model matrix from
  `archive/fix_finegrained_qwen3.5_moe.md` §Cross-Model Non-Regression
  (fg numeric probe both shape sets, the two pytest files, qwen3-30b spot bands).
- Update THIS doc in place: §6 hypothesis table verdicts, §8 rows, and a dated
  changelog line per iteration at the bottom.

---
## 9b. T3 FINAL SCOREBOARD (2026-07-14 — chunked delta-net, model-level, BOTH backends)

Config: `QWEN35_DELTA_CHUNK_SIZE=16000` (model-level patch, identical math both
backends; upstreamed to `LlamaFactory/model_utils/qwen35_delta_chunk.py` +
patcher; `QWEN35_DELTA_CHUNK_CHECKPOINT` auto-off under ZeRO-3) + tuned asym env
(`ASYM_CPU_ADAMW_ASYNC_GRAD_OFFLOAD=0`, both `*_SKIP_IN_BACKWARD=1`).
All rows RULES protocol, numerics VALID (first-ever valid 80k rows — chunked fla
calls stay under the ≥75k fault length). Artifacts:
`/workspace/qwen35_local/profiling_fix_qwen35_{chunk,final}`.

| workload | row | lat (s) | C (GiB) | G alloc (GiB) | G reserved (GiB) | loss band |
|---|---|---:|---:|---:|---:|---|
| 45000×8 | **A tuned+chunk** | 306–308 | 383 | **44.7** | **49.5–50.7** | 0.89–0.95 ✓ |
| 45000×8 | B superoffload+chunk | 261.8 | 432 | 59.4 | 72.4 | 0.83–0.95 ✓ |
| 80000×8 | **A tuned+chunk** | 428–433 | 554 | **78.9** | **91.7** | 0.80–0.93 ✓ |
| 80000×8 | B superoffload+chunk | 359.7–360.5 | 685 | 100.4 | 103.0 | 0.77–0.93 ✓ |

**Savings: 45k = 32% reserved (25% alloc); 80k = 11.0% reserved / 21.4% alloc.**
User-relaxed goal (>10% saving) MET at both workloads incl. the literal
`80000|8|1` target. The strict 0.80× reserved bar is met at 45k (0.68) and
missed at 80k (0.89) solely due to the +12.8 GiB reserved-over-alloc allocator
gap — root-caused and closed as irreducible in `fix_reserved.md` §8 (intra-
segment fragmentation + event-pending side-stream blocks; boundary and
per-layer releases each recover only ~0.6 GiB; `EXPANDABLE_SEG=false` explodes
reserved to 172.3 GiB — keep true).

Exhausted-lever notes (all measured null at 80k): `FG_KEEP_DGRADS_HBM=0`
byte-identical; chunk 8000 vs 16000 flat (peak is fg-expert/attention-phase
bound: A routed-ws 47.2 GiB vs B 62.3 at the peak instant); allocator
`garbage_collection_threshold` no-op. Next real lever if ever needed: chunking
the fg engine's full-R token buffers (code work, `qwen3_moe_finegrained.py`).

## 10. Phase-1/3 findings (2026-07-13 — diagnosis complete, pre-optimization)

1. **A currently LOSES to B, not parity**: 45k reserved 71.4 vs 60.5 GiB
   (A/B = 1.18; goal ≤ 0.80) and A is ~19% slower. H4 confirmed: B collapsed
   from 156.5 GiB (SDPA era) to 60.5 GiB under the FA4 runtime — the lean FA4
   forward + unsloth-off save-on-cpu erased the offload edge A used to have.
2. **H1 confirmed — the fla delta-net backward workspace is the peak for BOTH
   rows and it scales ~linearly with S**: `temporary_workspace/linear_attention`
   = A 53.0 GiB (74.1%) vs B 33.2 GiB (54.8%) at 45k; A 77.0 GiB (72.5%) at 70k.
3. **A's expert-side machinery works as designed** (B carries ~15 GiB of
   routed/shared-expert workspace+acts at 45k that A offloads away) — it is
   simply swamped by item 2.
4. **A's delta-net residency is ~1.6× B's at the same seq** (53.0 vs 33.2).
   Suspect: asym-path GC-recompute + saved-tensor wrapper interplay keeps
   recomputed forward intermediates alive concurrently with fla's backward
   workspace. This 20 GiB gap is the first fix target; closing it alone gets
   A≈77 GiB @70k (parity-ish, not the 0.80 goal).
   **CORRECTED by control C1 (same day):** A@45k with the linattn wrapper
   neutered (`ASYM_LINEAR_ATTENTION_SAVED_TENSOR_OFFLOAD_MIN_BYTES=1e15`) got
   WORSE on memory (alloc 69.6→77.3 GiB, reserved 73.1→103.0 with 25.7 GiB
   fragmentation) and 10% FASTER (329→295 s), loss identical. So the wrapper
   is net-positive for memory; the 53-vs-33 component split was partly an
   inferred-workspace attribution artifact. The REAL A-vs-B gap is ~11.1 GiB
   allocated / 12.6 GiB reserved, source still to be decomposed (memsnap run,
   `profiling_results/profiling_fix_qwen35_memsnap`). New flags
   `ASYM_LINEAR_ATTENTION_SAVED_TENSOR_OFFLOAD_SKIP_IN_BACKWARD` /
   `ASYM_ATTN_SAVED_TENSOR_OFFLOAD_SKIP_IN_BACKWARD` (default off) skip the
   recompute-window offload round trip = a SPEED lever (~+10% tok/s for
   ~+4-30 GiB peak), not the memory fix.
5. **The winning lever is chunking the delta-net backward** (old L2 lever):
   the 77 GiB @70k floor breaks into per-chunk workspace; also the natural
   route to unblock >70k numerics (fla faults ≥75k, 0.5.0 AND 0.5.1). Needs
   user sign-off (2026-07-03 policy reverted a global seq-chunked delta rule;
   an asym-wrapper-scoped recompute strategy is arguably a legit product
   feature like unsloth-GC, but the call is the user's).
6. Environment exonerated: causal_conv1d now installed; smoke/45k/70k all
   loss-in-band with clean audits (wrappers 40/30/10/40, zero fallbacks).

### §10b Control-run resolution (2026-07-13/14) — the gap is CLOSED at 45k

Controls at A@45k (2-step, memory-focused; RULES-grade confirm below):

| control | change | alloc / reserved (MiB) | verdict |
|---|---|---|---|
| C1 | linattn wrapper neutered via `MIN_BYTES=1e15` | 79,147 / 105,486 | WORSE — inner hook returns raw tensor, so LF's outer save_on_cpu is bypassed; NOT equivalent to skipping the hook |
| C2 | `ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=0` | 69,623 / 73,746 | NULL — the 8.5 GiB blocks are not this term |
| C3 | `*_SKIP_IN_BACKWARD=1` flags (LF save_on_cpu takes over linattn+attn saves) | 69,648 / 71,686 | alloc flat; reserved −1.5 GiB; removes wrapper round-trip |
| **C2b** | **`ASYM_CPU_ADAMW_ASYNC_GRAD_OFFLOAD=0`** | **59,879 / 61,786** | **THE GAP: async grad-offload staging backlog (~35 queued per-layer expert-LoRA grad tensors ≈ 8.5–9.5 GiB)** |

RULES-grade tuned-A confirm (C2b + C3): 45k → 58.9 GiB reserved (**below** B's
60.5), lat 329→308 s; 70k → 93.5 GiB (−12.8 vs baseline A), lat flat. Loss
bands unchanged everywhere. So qwen3.5 A-vs-B parity is restored with env-only
changes; the remaining headroom to G1 (≤0.80×B) is the shared ~30 GiB
delta-net recompute floor → chunked delta-net (T3) remains the only route,
and it also makes A's 80k rows numerics-valid (fla called at sub-fault chunk
lengths). PENDING follow-up (code): bounded-backlog async grad offload
(cap staging bytes, drain above threshold) to recover the qwen3-30b throughput
benefit while keeping this memory win — needs G3 non-regression when done.
NOTE: the tuned env deltas are NOT encoded in run-dir labels; every quoted
tuned row must state them explicitly.

## 11. Deep-reduction iteration log (2026-07-15, user mandate: reduce peak HBM until exhausted)

- **H1 (host)**: 5/3 pinned-home bug root-caused (fused gate_up kept alongside split
  gate/up copies from `_ensure_qwen3_moe_finegrained_bases`; affects qwen3-30b too).
  Fix `ASYMM_QWEN3_MOE_FG_RELEASE_FUSED_HOME=1` (default off): census-validated
  −41 GiB @35B (165.09→124.05 pinned); −144 GiB projected @122B.
- **H3a (null)**: existing `dscatter` expert-blocked backward + ker000: alloc byte-identical
  (78.9) and 3× slower → peak not in the blocked loops.
- **Phase-peak discovery**: fwd_peak 77.1 vs bwd_peak 78.9 — BOTH phases at the cap;
  any single-phase fix is invisible in the max.
- **H3b (new code)**: `ASYMM_QWEN3_MOE_FG_FWD_BLOCK_EXPERTS` — row-blocked
  pack/gate/up/act in both fg forwards + row-blocked silu backward (with dscatter →
  zero full-R GPU tensors in the fg path). Parity: numeric asserts green with flags on;
  flags-off suite 171/171. Result @80k: **fwd_peak 77.1→47.0 GiB**, reserved
  91.7→83.8, losses in band; bwd_peak unchanged 78.9 (=global peak).
- **Attribution correction**: bwd peak 80,821 MiB is BYTE-IDENTICAL across
  ker101/dscatter/H3b ⇒ the peak event was never the expert backward; the
  "routed_experts 48.4 GiB (51%)" row was the inferred-residual heuristic
  mislabeling. True owner is in the linattn/attention backward stack — allocator
  snapshot at the H3b config running to name it exactly.
- **bwd-peak decomposition (memsnap @H3b config)**: 70.86 GiB of the 78.6 peak is
  11 blocks allocated inside ONE GC layer's `torch.autograd.backward`
  (`LF checkpointing.py:137`) — the layer's save_on_cpu-restaged saves + gradient
  working set (a full-attention layer's q/k/v/out/lse/dq/dk/dv at 80k, hd=256).
  Everything else: liger chunk 2.44, checkpoint roots ~5, asym restage 0.23 (!).
  The expert engine is fully exonerated; asym-side HBM machinery is now ~zero at peak.
- NEXT LEVERS for the 70.9 GiB GC-layer term (all model-level → fair to B):
  **L3 sub-layer checkpointing** (split each decoder layer's GC into attn-block +
  mlp-block checkpoints, halving the per-layer backward set; cheapest, LF-side,
  env-gated like the delta chunking), L1 q-block-chunked FA4 backward
  (kernel-level, big), L2 save_on_cpu restage windowing (autograd-internal, risky).
### L1 STATUS 2026-07-15: implemented v1, parity probe FAILED on alignment
`qwen35_attn_chunk.py` + patcher hook exist (env `QWEN35_ATTN_BWD_CHUNK_Q`,
recompute-based chunked backward). Probe (`scratchpad/probe_attn_chunk.py`,
B2 Hq16 Hk2 S4096 D256, chunk 1024): out exact, but dq rel 0.80 / dk 0.88 —
the (q_chunk, k[:s1]) unequal-length causal call is NOT bottom-right aligned
in the HF fa4 wrapper path (each chunk row t saw keys ≤ t instead of ≤ s0+t).
FIX OPTIONS (next iteration): (a) determine alignment with a forward-only
probe (compare chunk out vs full-out slice); if a bottom-right/causal-offset
flag exists in the cute interface kwargs (window_size / cu_seqlens), use it;
(b) exact fallback: re-forward the FULL PREFIX q[:, :, :s1] vs k[:s1] (equal
lengths, standard causal ✓) and take torch.autograd.grad of out rows [s0:s1]
only — no double count, exact; peak saving shrinks (last chunk re-saves the
full prefix graph ≈ ~25 GiB vs 70.9 — still a ~45 GiB win at 80k).

### L1 original spec (chunked FA4 attention backward)

Where: new `LlamaFactory/src/llamafactory/model/model_utils/qwen35_attn_chunk.py`
(sibling of `qwen35_delta_chunk.py`), patched from `patcher.py`, env
`QWEN35_ATTN_BWD_CHUNK_Q` (query-tokens per chunk, 0=off, default off).
Mechanism: wrap the 10 full-attention modules' attention call with a custom
autograd Function that, in backward, loops q-chunks: for each chunk call
`flash_attn.cute` bwd (or fwd+`torch.autograd.grad` on a re-run chunk fwd with
k,v full) producing dq_chunk (written into dq) and accumulating dk/dv (fp32
accum [8,80k,2,256] ≈ 2.5 GiB — kv heads are only 2). Working set per moment:
k,v (9.8) + q_chunk/do_chunk/dq_chunk (3×4.9/Nc) + dk/dv accum (5) ≈ 17 GiB at
Nc=4 vs ~45–60 today → projected bwd peak 78.9 → ~45–50. Numerics: exact
(chunked dq; dk/dv accumulation order changes only fp32 sums). Model-level ⇒
applied to BOTH backends (fairness), same policy as delta chunking. Validate:
toy parity vs unchunked flash bwd, then 80k probe.

- OPEN: implement+probe L1 per spec above; ker000+blocks speed (934–1032 s vs
  ker101 428 s) to be recovered later (block tuning / route-compat blocking);
  122B re-run with release_fused_home + fwd-blocks (host −144 GiB expected →
  24k+ attempt); 35B@120k stress row; promote H1/H3b envs into the qwen3.5
  harness defaults after RULES-grade confirms; bogus stats-tag asserts in
  test_asym_qwen3_experts_sm100_moe_finegrained_matches_torch_backend when
  block envs forced globally (numerics green — tags key on unblocked path).

Changelog:
- 2026-07-13: doc created; env audit found `.venv-fa4` missing `causal_conv1d`
  (canonical `.venv` has it); no q3.5 artifacts exist in this checkout.
- 2026-07-13: `causal_conv1d==1.6.2.post1` installed into `.venv-fa4`; the
  causal_conv1d/fla/fa4 availability triple verified all-True. Phase 0 step 1
  closed.
- 2026-07-13: fla long-seq probe run on GPU 3 (Phase 0 step 3): clean ≤70000,
  illegal memory access at S=75000 → the 80000|8|1 target is still fla-blocked;
  Phase 2 (fla upgrade / chunked-scan with sign-off / upstream) is mandatory
  before the literal target row; L2=70000 is the current honest headline rung.
  Phase 0 is now fully green otherwise.
- 2026-07-13: Phase-2 route 1 probed: fla/fla-core 0.5.1 (latest on PyPI) still
  faults at S=75000; rolled back to the 0.5.0 pin. Remaining unblock routes for
  s80000: chunked-scan with user sign-off (route 3) or upstream fix (route 4).
- 2026-07-13 (evening): Phase 1 executed (smoke + 45k A/B + 70k A; artifacts in
  `profiling_results/profiling_fix_qwen35_repro/`). NFS filled to 100% mid-run
  (ENOSPC killed the first 45k B attempt; rerun landed clean after rerouting
  writes; user later freed the volume and moved all outputs under
  `profiling_results/`). s70000/s80000 datasets are LOCAL files symlinked into
  `LlamaFactory/data/` (`/workspace/qwen35_local/build_l2_l3_datasets.sh`).
  Per user: no further superoffload baseline runs (45k baseline stands), stop
  the ladder at 70k. §10 has the diagnosis; awaiting user go-ahead on the fix
  direction before any optimization.
- 2026-07-14: control series C1/C2/C3/C2b run (§10b) — root cause of the A-vs-B
  gap = async grad-offload staging backlog; tuned-A (async-grad-offload off +
  SKIP_IN_BACKWARD flags) confirmed at RULES protocol: 45k 58.9 GiB reserved
  (beats B 60.5), 70k 93.5 GiB, loss bands unchanged, lat 329→308 s @45k.
  New default-off flags added in `linear_attention_activation_offload.py` /
  `attention_activation_offload.py` (`*_SKIP_IN_BACKWARD`). Artifacts:
  `profiling_results/profiling_fix_qwen35_{ctl,ctl2_dgradscpu,ctl2b_syncgrad,ctl3_skipbwd,memsnap,tunedA}`.
  OPEN: T3 chunked delta-net (goal-level win + 80k unblock) awaiting sign-off;
  bounded-backlog async grad offload as the code-proper alternative to the
  sync-mode env; pytest gates for the flag edits before merge.
- 2026-07-14 (T3, user-approved): chunked delta-net implemented MODEL-LEVEL
  (both backends; `QWEN35_DELTA_CHUNK_SIZE`, per-chunk non-reentrant checkpoint
  with ZeRO-3 autodetect off-switch, conv-tail + recurrent-state carries).
  Kernel parity 0.5% bf16; loss bands identical to unchunked; **first
  numerics-valid 80000|8|1 rows ever, both backends** (fla ≥75k fault
  sidestepped). Final scoreboard §9b: 45k A 49.5 vs B 72.4 (−32%); 80k A 91.7
  vs B 103.0 (−11% reserved, −21% alloc). User relaxed goal to >10% saving →
  MET at both workloads. Reserved≫alloc anomaly investigated and closed in
  `fix_reserved.md` (keep EXPANDABLE_SEG=true — false explodes to 172.3 GiB;
  release knobs `ASYM_EMPTY_CACHE_PHASES` / `ASYM_EMPTY_CACHE_EVERY_GC_LAYERS`
  added default-off, measured ~0.6 GiB only). Patch upstreamed from the
  local sitecustomize into `LlamaFactory/model_utils/qwen35_delta_chunk.py`
  + `patcher.py`. 122B addendum: baseline gap reproduces and tuned config wins
  (8k: alloc 20.7 vs 29.5, −30%); host RAM is asym's binding constraint ≥16k
  on this node (>1.12 TB vs B 644 GiB).
- 2026-07-14 (122B chunked battery, harness-default verified at 122B — 36
  modules patched per row; 8k rows byte-match unchunked = default provably
  inert at seq≤chunk): **16k HOST UNLOCK** — A@16k now completes (two prior
  host-OOMs were baseline-env; the tuned env's SKIP_IN_BACKWARD flags remove
  the recompute-window pinned save-on-cpu copies → C 801 GiB < 1.17 TB).
  16k pair at A's active chunk (8000): A 36.7 res / 31.2 alloc / 330.7 s vs
  B best-config 47.9 / 43.5 / 302.4 → **−23.4% reserved / −28.2% alloc**
  (vs matched-chunk B 53.7/49.2 → −32/−37; chunking hurts B at 16k because
  ZeRO-3 auto-disables its per-chunk checkpoint — quote the conservative
  best-config comparison). Strict 0.80×-reserved bar met at 122B (0.766).
  Artifacts: `profiling_results/profiling_fix_qwen35_122b_chunked{,_c8}`.
- 2026-07-14 (user decision): chunk=16000 is now the HARNESS DEFAULT for
  qwen3.5 rows in `profile_lora_lf_test_both.sh` (auto-set alongside the FA4
  runtime switch; `QWEN35_DELTA_CHUNK_DEFAULT` to change, explicit
  `QWEN35_DELTA_CHUNK_SIZE` — incl. 0 for stock — overrides; other model
  families unaffected, verified by dry-run). Reported qwen3.5 results need NOT
  quote the chunk config anymore — it is part of the canonical qwen3.5 runtime,
  like FA4. Chunk-size sensitivity on record: 8000 vs 16000 measured flat on
  both peak memory and latency at 80k.

### H4 stress battery (2026-07-15, clean config: tuned + chunk + ker101 + fused-home fix)
| row | lat s | tok/s | C GiB | G res | G alloc | verdict |
|---|---:|---:|---:|---:|---:|---|
| A35@120k|8 | 565.3 | 1698 | 547 | 131.5 | 118.3 | VALID (first 120k qwen3.5 row; 8 delta chunks) |
| B35@120k|8 | 460.1 | 2087 | 720 | 153.1 | 147.1 | VALID; 83% of card -> B ceiling ~130k vs A ~165k+ |
| A122@32k|8 | — | — | >1.12TB | — | — | HOST ceiling (watchdog; even with fused-home fix) |
| B122@32k|8 | — | — | >1.12TB | — | — | HOST ceiling (symmetric; node-bound, not system-bound) |
35B@120k: A −14% reserved, −24% host, B +19% faster. 122B: 16k stays the max
validated pair on this 1.17TB node (host-bound for BOTH systems at 32k).

- 2026-07-15 (defaults promoted, user directive "clear fixes default on"):
  `ASYMM_QWEN3_MOE_FG_RELEASE_FUSED_HOME` default ON;
  `*_SKIP_IN_BACKWARD` default AUTO (= active iff UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU).
  Latency-neutral by construction (init-time free; strictly-fewer copies).
  G3 certification (qwen3-30b s20000, new defaults): loss 1.76 ✓ band 1.775±0.05,
  host expert home 92,160 → **55,296 MiB = exact bf16 size, dedup GONE**,
  HBM 31.9 GiB sane. pytest 171/171. Sync-grad-offload NOT defaulted (real
  tradeoff; bounded-backlog remains the designed fix).
