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
| | 45000×8 | A | | | | | | | | |
| | 45000×8 | B | | | | | | | | |
| | 70000×8 | A | | | | | | | | |
| | 70000×8 | B | | | | | | | | |
| | 80000×8 | A | | | | | | | | |
| | 80000×8 | B | | | | | | | | |

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
