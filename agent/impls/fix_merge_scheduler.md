# fix_merge_scheduler — staged implementation plan for the sched40×sched42 merge

(Companion to `merge_scheduler.md` = the DECISION record (read its §0–§3 first);
this doc = HOW to build it, stage by stage, each with a hard gate. Trees:
BASE = `/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM` (40, live, the merge
target); DONOR = `/home/kevinni/AsymGEMM-SFT-39/third_party/AsymGEMM` (42,
live, read-only). All GPU validation runs INSIDE the container:
`asym34_enroot_run` (mounts AsymGEMM-SFT at /workspace/AsymGEMM-SFT), then
`cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM`. 2026-07-21.)

## 0. Goals · reference configs · agent rules · caveats

GOAL: one tree = BASE + 42's five default-OFF source features + the merged
scheduler (3 tiers, bytes-only feasibility, GPU AND host caps, knee batch,
`--predict` offline) + `backend|TIER` presets with full-fidelity artifact
naming + `config.json` manifests. Dense behavior identical to BASE (measured);
c14 MoE recipes reproducible in this tree (measured).

REFERENCE SET (fixed; every perf gate compares to these; accept = within
±1.5% tok/s (NOISE) and ±2 GiB HBM of the cited record). Command literals
are UNWRAPPED inside the fences — copy-paste ONLY from the fences.
- C1 dense-T1 — ref c12 system_summary §1: 906 us/tok (=1104 tok/s), 116.0 GiB
```
ASYM_GEMM_DISPATCH=staged bash scripts/lf/tp_probe.sh q3-32b mrgc1 "asym_cpuadamwds|unsloth-ohbm0|ligerloss1" 128000 2
```
- C2 dense-T2 — ref c12 §1: 1044 us/tok, 93.6 GiB
```
ASYM_GEMM_DISPATCH=staged ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM=1 ASYM_SAVED_TENSOR_ASYNC_UNPACK=1 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 bash scripts/lf/tp_probe.sh q3-32b mrgc2 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 128000 2
```
- C3 moe-dial-KA (= `dial_l3_keepacts`: MoE-KA ONLY + BASE pins) — ref
  scheduler_v2 §3b L3 row (−31.8 s/it vs L2, 180.0 GiB @120k×8; absolute
  s/it in the row)
```
ASYM_GEMM_DISPATCH=staged ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 bash scripts/lf/tp_probe.sh q3-30b-a3b mrgc3 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 120000 8
```
- C4 moe-bundle RECORD REPRODUCTION — the tputsched 900k b1 latency
  emission AS-MEASURED (panel-cache ON because the record had it; this is
  a record replay, NOT a recipe — the panel-cache quarantine stands).
  Ref (prompt.md v2 / 42's calibration): 519 tok/s, 183.0 GiB. NB ~99%
  util — the record's own operating point; expect long steps.
```
ASYM_GEMM_DISPATCH=staged ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1 ASYMM_ATTN_ACT_KEEP_ACTS_HBM=1 ASYM_GC_SAVE_ON_CPU_OVERRIDE=false ASYM_W_PANEL_CACHE_GB=6 ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 ASYMM_QWEN3_MOE_FG_DA_GPU=1 ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYMM_FUSED_LORA_ADDMM=1 ASYMM_QWEN3_MOE_FG_REUSE_PACKED_X=1 bash scripts/lf/tp_probe.sh q3-30b-a3b mrgc4 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 900000 1
```
- C4b moe deep SHED state — ref c14 test_throughpout_v2 P3 row (800k b1):
  597 tok/s, 147.5 GiB. Staged + pins, NO keep-acts flags, NO GC override —
  read the row's exact env from its archived command.txt before running.
- C5 parity — ref 1110 tok/s (DONOR fix_asym ledger :18; after S6, the
  merged fix_asym)
```
bash scripts/lf/tp_probe.sh q3-32b mrgc5 "superoffload_mem|unsloth-ohbm0|ligerloss1" 128000 2
```
Measurement protocol: w1+m2 (`PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1` —
tp_probe defaults); steady us/tok = mean of the 2 measured steps in
`step_samples.json`; HBM peak from the run's memory summary; verdicts from
jobs.tsv ok-row or artifacts-complete fallback (tp_probe.sh logic — NEVER the
driver exit code).

AGENT RULES (behavior while executing this plan):
- Keep fixing and building until the stage gate passes. Do not stop at the
  first failure; do not end the session with a stage half-done. On failure:
  read the run's log tail, jobs.tsv, step_samples.json, command.txt/config
  snapshot BEFORE changing code.
- One variable per A/B. Never "fix" a regression by turning a measured
  feature off — find the cause.
- e2e LoRA profiling (tp_probe.sh / the drivers) is the ONLY accepted perf
  evidence. Toy/unit runs prove wiring, never performance.
- Record every measured number + verdict in the STATUS LEDGER (§7 here) as
  you go; cite run dirs.
- Near a memory wall (≥92%−8% util predicted), probe (tp_probe.sh with B and
  B−1), don't trust lines.

CAVEATS (do-NOT list):
- Do NOT enable any of the five 42 features in ANY dense recipe (measured
  absent from every c12 dense number; fused-addmm is numerics-touching).
- Do NOT invent a MoE "split" T2 line/recipe (one dial point, no line —
  merge_scheduler.md §2b + §3 step 4). MoE T2 = c14 bundle as-measured.
- Backend is NEVER auto-selected (asym_cpuadamwds vs asym stays user input).
- Do NOT touch csrc/kernels/`_C` — zero divergence there by verified fact.
- Do NOT introduce per-expert Python loops, small-GEMM decompositions, or
  extra full-width elementwise sweeps. Port 42's patterns verbatim: fused
  `addmm_` single-kernel epilogue; `packed_rows[row_start:row_end]` slice
  reuse (no `index_select` re-gather when kept); panel-cache LRU hit returns
  the cached tensor (zero copies); grouped expert GEMMs stay grouped;
  chunking via `fg_chunk_rows` (≥1 GiB chunks → few launches).
- Do NOT rename/reorder existing dir-name components — only APPEND (S4).
- jobs.tsv/artifacts are truth; the driver exits 0 on failed jobs.

## S1 — Source port (six files) + driver forwarding line

Scope (exact):
- Copy DONOR→BASE (brings the five flag-gated features; every DONOR hunk
  verified env-gated):
```
D=/home/kevinni/AsymGEMM-SFT-39/third_party/AsymGEMM
for f in activation_offload attention_activation_offload decoder_activation_offload dense_mlp_finegrained frozen_linear qwen3_moe_finegrained; do cp "$D/asym_gemm/training/$f.py" asym_gemm/training/; done
```
- Re-apply BASE-only noclone/mutable on top (kept UNWIRED):
  1. `qwen3_moe_finegrained.py`: re-add `_keep_stage_noclone_enabled()`
     (module fn, reads `ASYMM_FG_KEEP_STAGE_NOCLONE`, default off); in
     `_HBMKeepManager`: `__init__` += `self.stage_clone_bytes = 0;
     self.stage_noclone_bytes = 0`; signature
     `stage(self, handle, *, tag, mutable: bool = True)` with
     `if mutable or not _keep_stage_noclone_enabled(): clone-path else
     return handle.tensor`; snapshot dict += the two byte counters.
  2. `dense_mlp_finegrained.py`: re-add `mutable=False` at the audited
     read-only stage() call sites (11 sites — tags: mlp.up_for_act ×2
     (grad + nograd paths), mlp.act_for_down_base ×2 (grad + nograd),
     mlp.S_down_for_dB, mlp.up_for_silu_bwd_dgate,
     mlp.gate_for_silu_bwd_dgate, mlp.dgate, mlp.S_gate_for_dB, mlp.dup,
     mlp.S_up_for_dB; capture exact lines via `diff` BEFORE overwriting).
     NB dense imports `_HBMKeepManager` FROM qwen3_moe_finegrained
     (dense :253-255) — item 1's single class edit covers dense too.
  3. `activation_offload.py`: CPU manager `stage(..., mutable: bool = True)`
     accepts-and-ignores the hint (inherently clone-safe) so call sites
     compile against either manager.
- Driver edits (definite, verified 2026-07-21):
  In `profile_lora_lf_test_source.sh`: REPLACE the :3844 noclone forwarding
  line with:
```
ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_ACT_KEEP_ACTS_HBM="${ASYMM_ATTN_ACT_KEEP_ACTS_HBM:-}"
```
  (this IS the noclone unwiring). In `profile_lora_lf_test_both.sh`: it has
  NEITHER line (verified, both trees) — ADD the same line after its :3843
  (`...DENSE_MLP_FG_KEEP_ACTS_HBM` entry); without it a `_both`-launched
  bundle run silently drops attention keep-acts. (`_both`'s command.txt
  writers sit at :3182/:3980/:3996/:4007 — ±1 vs `_source`; matters in S4.)
- `git add` the untracked keepers: `scripts/lf/tp_probe.sh`,
  `agent/impls/merge_scheduler.md`, this file.

Efficiency invariants to preserve while porting (read the diffs, don't
re-implement): fused path = ONE `dest.addmm_(lhs, rhs_t, alpha=scale)` per
(chunk×)call — never matmul→mul→cast→add_ chains; packed-X reuse = slice
view of the kept route-space tensor; panel cache = data_ptr-keyed LRU with
byte-cap eviction, passthrough when cap=0; async-pack = side-stream D2H with
producer-event + `record_stream` keepalive (port as-is, stays unwired).

Validation gates (in-container, from third_party/AsymGEMM):
- S1-V1 static (run from the fence):
```
python3 -c "import asym_gemm.training.qwen3_moe_finegrained, asym_gemm.training.dense_mlp_finegrained, asym_gemm.training.attention_activation_offload, asym_gemm.training.activation_offload, asym_gemm.training.decoder_activation_offload, asym_gemm.training.frozen_linear"
grep -c KEEP_STAGE_NOCLONE asym_gemm/training/qwen3_moe_finegrained.py   # >= 2
```
  plus: all five feature flags grep-present; defaults-off asserted via
  `python3 -c` on each `_..._enabled()` with a clean env.
- S1-V2 dense e2e A/B (flags off ⇒ must reproduce BASE): run C1 then C2
  exactly as fenced in §0 (tags mrgs1c1/mrgs1c2) → FIT + within band.
- S1-V3 MoE dial point: run C3 (§0 fence) → within band of scheduler_v2
  §3b L3 (180.0 GiB; s/it per row).
- S1-V4 bundle delivery proof at a FEASIBLE point: the bundle line gives
  ≈188 GiB at 120k×8 (> 0.92·185 = 170.2 — would false-HARDFAIL), so run
  the C4 env at 120k b4 (≈97 GiB) — once via tp_probe (`_source`) and once
  via a run_dial_ladder-style `_both` invocation. Accept: the run's
  profile/summary JSON under the config dir contains
  `attn_act_hbm_gemm` counts > 0 (emitted via integrations/lf.py); if no
  artifact carries it, fall back to a 3-line in-container python check
  (construct the wrapper with the env set, assert the keep path taken).
  This proves `ASYMM_ATTN_ACT_KEEP_ACTS_HBM` survives the driver→trainer
  boundary under BOTH drivers; if not, the LF_CONFIG line is missing in
  that driver — add, re-run.
- S1-V5 record reproduction: run C4 (900k b1, §0 fence) → 519 tok/s /
  183.0 GiB within band.
Risks/watch: (a) env delivery through the `_both` driver (V4 decides);
(b) machine-state drift vs c12/c14 records — if C1/C5 both shift by the
same factor, re-baseline the band on today's C1/C5 and note it in the
ledger; (c) `.venv` editable installs point at BASE paths — do NOT copy
DONOR venv metadata.

## S2 — Scheduler refactor (`scripts/lf/asym_scheduler.py`)

Scope: this one file (land DONOR's copy first — `cp` from DONOR — then
refactor; keep `--sweep`, `--selftest`, dataclass style).

Shape of the refactor (pseudocode; keep 42's file layout):
```python
TIERS = ("T1","T2","T3")            # memory-DESCENDING preference order
@dataclass(frozen=True) class TierLine:
    token: str                       # recompute token (dense/moe variant)
    env: dict                        # full recipe env (per §0 configs)
    base: float; m: float            # GiB intercept + GiB/ktok (B*s)
    m2: float|None; knee2_k: float|None   # >800k piecewise secant (MoE KA)
    host_c: float; host_h: float     # HOST_t(N) = c + h·N   [GB]
    valid_k: tuple                   # fit validity range (else: anchor/probe)
@dataclass(frozen=True) class ModelTab:
    name: str; tiers: dict[str, TierLine]
    knee_tokens_k: float|None        # 400.0 for q3-30b-a3b; None = unmeasured
C_HBM = 185.0; BETA = 0.92; C_HOST = 957.0   # c12 header constants

def mem(line, tok_k):                # piecewise-linear, closed form
    if line.m2 and tok_k > line.knee2_k:
        return line.base + line.m*line.knee2_k + line.m2*(tok_k - line.knee2_k)
    return line.base + line.m*tok_k
def host(line, tok_k): return line.host_c + line.host_h*tok_k
def max_B(line, s_k):                # closed form — NO loops
    B_hbm  = invert_mem(line, BETA*C_HBM, s_k)   # floor((cap−base)/(m·s_k)),
                                                 # piecewise-aware past knee2_k
    B_host = floor((C_HOST - line.host_c) / (line.host_h * s_k)) if line.host_h else INF
    return max(0, min(B_hbm, B_host))
def schedule(model, s):
    s_k = s/1000
    if s_k <= ANCHOR_MAX_K and model has anchors: return anchor_row(s_k)   # measured truth
    for tier in TIERS:               # first feasible = fastest (structural)
        B = max_B(model.tiers[tier], s_k)
        if model.knee_tokens_k: B = min(B, ceil(model.knee_tokens_k/s_k))
        if B >= 1: return Plan(tier, B, env=model.tiers[tier].env,
                               probe=near_wall(mem, BETA*C_HBM))   # ≥(β−0.08) ⇒ PROBE
    raise infeasible
# emit(): prints token, B, env lines, predicted GiB/host, and — when
# probe=True — the exact tp_probe.sh command for B and B−1.
# --predict: 42's τ/water-fill machinery moved verbatim behind this flag.
# --selftest: 5 properties re-targeted at tiers (nested shed along s,
#   monotone mem, reserved-sweep nestedness, analytic T2→T3 boundary vs
#   schedule(), safety monotonicity).
```
MoE T2 = the bundle TierLine (base 4.3+2.0 staged, m 0.100+0.0375+0.052
w/ >800k secant — 42's constants, provenance per merge_scheduler.md §3
step 4); dense T2/T1/T3 lines from c12 §4. No τ in any decision path.

Validation gates (offline, no GPU):
- S2-V1 `python3 scripts/lf/asym_scheduler.py --selftest` → PASS (5/5).
- S2-V2 spot decisions: `python3 scripts/lf/asym_scheduler.py q3-30b-a3b
  800000` → T2-bundle emission; `... q3-30b-a3b 1200000` → T3; `... q3-32b
  128000` → anchor/T1; `... llama3.3-70b 416000` → NOT T3 (host screen).
Risks/watch: closed-form max_B must round-trip the >800k piecewise segment
correctly (unit-check mem(max_B) ≤ cap < mem(max_B+1) inside selftest).

## S3 — Constants transcription + `--replay` gate

Scope: `asym_scheduler.py` MODEL tables + a REPLAY list.
- Transcribe per-(model, tier) byte lines from c12 system_summary §4 (q3-32b,
  llama3.3-70b, q3-30b; q3.5-35b marked `pending-fit`), host lines from c12
  §5 anchors (llama T2 975–984 flat; q32 T3 957@576k→980@640k; MoE T3
  925@1.6M), MoE rung slopes from 42 (§3b + 900k secant). Knee 400k =
  q3-30b ONLY (`knee_tokens_k=None` for dense — do NOT cap dense batch).
- REPLAY = a literal table of every measured record decision (c12 §8 list:
  dense T1→192k, MoE T1→800k, llama T2 320–416k, q32 T2 384–448k, q32 T3
  576–640k, MoE T3 1.1M–1.6M, llama T3 EXCLUDED by host) asserted against
  `schedule()`.

Validation: S3-V1 `python3 scripts/lf/asym_scheduler.py --replay` exit 0,
prints per-point PASS incl. the llama host-exclusion and 1.6M. S3-V2
`--sweep` table eyeball: no tier flapping, anchors verbatim at ≤128k.
Risks/watch: if a replay point fails only inside the near-wall band, that is
the probe rule working — mark the point `PROBE` in the replay table, not a
constant tweak.

## S4 — `backend|TIER` presets + naming components + config.json (BOTH drivers)

Scope: `scripts/lf/profile_lora_lf_test_source.sh` AND `scripts/lf/profile_
lora_lf_test_both.sh` (twin edits; they are siblings): the backend/recompute
parser (case at ~:1176–1256), the RUNS item parser (~:1318), the config-label
tail (`grad_offload_suffix`, ~:2068), the LF_CONFIG export block
(~:3812–3856), the four `command.txt` writer sites (:3182, :3981, :3997,
:4008 in `_source`). Plus `asym_scheduler.py --emit-recipes`.

Changes (pseudocode):
```bash
# 1. recipe table: generated file, single source of truth
#    asym_scheduler.py --emit-recipes > scripts/lf/tier_recipes.sh  (commit it)
#    declare -A TIER_TOKEN=( [q3moe|T2]="recomp-off-full-fg-ker000-ceil0000-ohbm0" ... )
#    declare -A TIER_ENV=(   [q3moe|T2]="ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1 ..." ... )
# 2. expansion, inside the recompute parser BEFORE tokenization:
if [[ "${recompute_part^^}" =~ ^T[123]$ ]]; then
  fam=$(model_family "${model}")            # dense|moe by model table
  recompute_part="${TIER_TOKEN[${fam}|${recompute_part^^}]}"
  for kv in ${TIER_ENV[...]}: export only-if-unset "${kv}"   # user env overrides recipe (A/B hook)
  tier_requested="T?"                        # provenance for config.json
fi
# 3. label tail — APPEND (never reorder) after __gradoff*__weightoff*:
label+="__mlpka${ka}__attnka${aka}__gcsave${sv}__fadd${fa}__xreuse${xr}__pcache${pc}"
#    values resolved from the final env (defaults 0/cpu/0); safe_label()
#    still guards >255 chars — unchanged.
# 4. config.json — at each command.txt writer, same dir:
python3 - <<'PY' > "${seq_root}/config.json"
import json, os, sys
cfg = {k: v for k, v in os.environ.items()
       if k.startswith(("ASYM", "UNSLOTH"))}
cfg["_tier_requested"]  = os.environ.get("TIER_REQUESTED", "")
cfg["_recompute_token"] = os.environ.get("RECOMPUTE_TOKEN_RESOLVED", "")
cfg["_recipes_file"]    = "tier_recipes.sh@<git-hash>"
json.dump(cfg, sys.stdout, indent=1, sort_keys=True)
PY
```
Backend is passed through UNTOUCHED (`asym_cpuadamwds|T2` — tier expansion
never rewrites `backend_part`). Raw tokens keep working (regex only matches
bare T1/T2/T3).

Validation gates:
- S4-V1 dry: a RUNS line `q3-30b-a3b|1 ; asym_cpuadamwds|T2|ligerloss1 ;
  24000|8|1 ; none|false|...` echoes the expanded token+env (add a
  `DRY_RUN=1` early-exit print if not present) and matches the recipe table.
- S4-V2 e2e distinctness (short but real): two runs, same (model,b,s)=
  (q3-30b-a3b, 8, 24000), one with `|T2|`, one raw ker000 token without KA →
  DISTINCT config dirs (label tails differ at `mlpka`/`attnka`), each with
  `config.json` whose diff shows exactly the recipe flags.
- S4-V3 compat: one historical raw-token RUNS line → runs; name = old name
  + the new appended components (expected — note in ledger); cached-run
  lookup for pre-merge dirs will MISS (accepted; OVERWRITE semantics).
- S4-V4 twin check: `diff <(sed -n '<label+LF_CONFIG blocks>' _source)
  <(same from _both)` → identical.
Risks/watch: (a) old-dir reuse is lost by the rename — acceptable, but if a
campaign later needs it, write the historical-dir renamer (safe_label
comment says one existed); (b) `only-if-unset` export ordering vs the
driver's own env defaulting (~:3300s) — expansion must run BEFORE the
driver computes `unsloth_recompute_save_on_cpu` etc. (:3326–3329 reads
`ASYM_GC_SAVE_ON_CPU_OVERRIDE`); (c) config.json must dump the FINAL env at
launch time (same scope as command.txt), not the parse-time env.

## S5 — Full merge gate (= merge_scheduler.md §3 step 8)

All in-container; all e2e; record in ledger:
- (a) S2-V1 selftest PASS.
- (b) S3-V1 replay PASS (both records incl. 1.6M + llama host exclusion).
- (c) no-regression: C1, C2, C5 (dense, features OFF) within band.
- (d) recipes ON: C3, C4 (record reproduction incl. panel-cache
  as-measured — a replay, not a recipe), C4b within band (re-validates MoE
  at its measured points).
- (e) S4-V2 distinct-dirs + config.json accuracy.
- (f) the S7 3×3 no-regression matrix: 9/9 rows within band (the final
  merge-success proof; see S7 for the rows, bands, and breach protocol).
Only after (a)–(f): declare merged; update merge_scheduler.md header with
the gate ledger line; commit on `merge_sched` branch (never on main_kevin).

## S7 — Post-merge 3×3 no-regression matrix (THE merge-success proof; runs AFTER S1–S6)

Goal (Kevin 2026-07-21): NO noticeable regression on asym* runs vs the
recorded metrics. Reference records = the in-tree snapshots
`archive/s04-p1-dgx-02-c12/` + `archive/s04-p1-dgx-02-c14/` (verified
byte-identical to the live `/home/kevinni/env/outputs/` copies that
`agent/impls/s04-p1-dgx-02-c1{2,4}` symlink to — cite the archive/ paths;
they are frozen in-tree). READ each reference row (tok/s + GiB + exact env
via its archived command.txt) from those files BEFORE running — do not
trust numbers quoted anywhere else, including this doc.

The 9 runs (3 models × 3 recorded operating points; all asym_cpuadamwds;
rows 1/2/7 are literally C1/C2/C3 — reuse the S5 results if fresh):
| # | model | point | recipe | reference row |
|---|---|---|---|---|
| 1 | q3-32b | T1 128k b2 | C1 | c12 system_summary §1 (906 us/tok, 116.0 GiB) |
| 2 | q3-32b | T2 128k b2 | C2 | c12 §1 (1044, 93.6) |
| 3 | q3-32b | T3 576k b1 | fg defaults, no staged | c12 records (111.2 GiB; tok/s from test_throughput_results.md) |
| 4 | llama3.3-70b | T1 192k b1 | C1-style | c12 (parity point, +0.3% vs SO) |
| 5 | llama3.3-70b | T2 384k b1 | C2-style (llama) | c12 (178.6 GiB near-wall row) |
| 6 | llama3.3-70b | T2 416k b1 | C2-style — WALL PROBE | c12 (FIT @~99% — llama's terminal point; T3 EXCLUDED for llama BY DESIGN, host inversion — do NOT run llama T3) |
| 7 | q3-30b-a3b | KA dial 120k b8 | C3 | scheduler_v2 §3b L3 |
| 8 | q3-30b-a3b | 800k b1 shed | C4b | c14 test_throughpout_v2 P3 (597 tok/s, 147.5 GiB) |
| 9 | q3-30b-a3b | T3 1.1M b1 | shed/streamed per archived command.txt | c14 crossover row @1.1M |

Protocol: w1+m2 via tp_probe.sh, one run each (rows 6 and 9 are long —
budget hours); metrics = steady us/tok (mean of 2 measured steps in
step_samples.json) + HBM peak (memory summary) + host RSS where the record
has it. ACCEPT per row: tok/s not worse than ref by >1.5% (NOISE) AND HBM
within ±2 GiB (±3 GiB for near-wall rows 5/6). PASS = 9/9.
ON BREACH — flag it, do NOT paper over: (i) rerun the row once (rule out
one-off); (ii) diff the run's config.json against the archived command.txt
env-by-env (the #1 suspect is a recipe/env delta); (iii) if config-identical
and still slow, bisect one variable at a time (feature flags OFF first —
they default off, so any delta implicates the port); (iv) if ALL rows shift
by a similar factor, suspect machine/driver drift — re-run C5 (superoffload
parity, untouched by the merge) and re-baseline only if C5 shifted equally;
(v) record every breach + diagnosis in the ledger. A real, unexplained
asym* regression = the merge is NOT accepted — stop and report.

## S6 — Docs reconciliation (mechanical; code-only scope per Kevin)

fix_asym.md: BASE body + graft DONOR's `## 5a` + DONOR's post-fork STATUS
LEDGER tail. Preserve DONOR's `agent/handoffs/prompt.md` as
`agent/handoffs/prompt_v2_c14.md`. Union `agent/reports/` (midterm_memory
tail + midterm.md + figures/). Copy `agent/impls/remaining_optimizations.md`
and `agent/impls/archive/s04-p1-dgx-02-c14_old/`. scheduler_v2.md: keep
BASE's, append DONOR's §10 record map + tombstone β-dial/water-fill.
Validation: grep-checks (`§5a` present exactly once; prompt_v2_c14.md
exists; c14_old present; §10 present).

## 7. STATUS LEDGER (append-only; every gate writes a line)

- [S0 2026-07-21] Plan written; reference set C1–C5 fixed; bands defined.
- [S0 2026-07-21] S7 added (Kevin): 3×3 post-merge no-regression matrix,
  gate (f). Reference paths verified: in-tree `archive/s04-p1-dgx-02-c1{2,4}`
  = byte-identical snapshots of the live env/outputs records.
- [S0 2026-07-21 review-pass-1] 6 fixes applied: C4 re-anchored to the
  REAL record (900k b1 latency emission, 519 tok/s / 183.0 GiB — the
  "576k row" never existed); C4b (800k shed, 597/147.5) added; `_both`
  driver needs an ADD (has neither forwarding line); mutable sites = 11
  (up_for_act ×2); S1-V4 moved to 120k b4 (b8 infeasible ≈188 GiB) with a
  concrete artifact check; all command literals moved into unwrapped
  fences.
