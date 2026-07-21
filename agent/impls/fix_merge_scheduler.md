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
- C4 moe-bundle RECORD REPRODUCTION — the tputsched-c14 900k b1 run
  AS-ARCHIVED (env verified from its command.txt 2026-07-21: KA bundle +
  the 5 base pins; NO panel-cache, NO fused-addmm, NO reuse-packed-x —
  prompt.md's "incl. panel-cache" claim contradicted the archive; archive
  wins). Ref (recomputed from its step_samples.json): 519 tok/s,
  183.0 GiB. NB ~99% util — the record's own operating point; long steps.
```
ASYM_GEMM_DISPATCH=staged ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1 ASYMM_ATTN_ACT_KEEP_ACTS_HBM=1 ASYM_GC_SAVE_ON_CPU_OVERRIDE=false ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 ASYMM_QWEN3_MOE_FG_DA_GPU=1 ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 bash scripts/lf/tp_probe.sh q3-30b-a3b mrgc4 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 900000 1
```
- C4b moe deep SHED state — ref c14 P3 (800k b1), archive-verified
  (tputasl-c14 step_samples: 596 tok/s, 147.5 GiB, RSS 539). Env from its
  command.txt: staged + ker000 + the 5 base pins ONLY (no KA, no GC
  override, no fused/reuse/panel):
```
ASYM_GEMM_DISPATCH=staged ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 ASYMM_QWEN3_MOE_FG_DA_GPU=1 ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 bash scripts/lf/tp_probe.sh q3-30b-a3b mrgc4b "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 800000 1
```
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

The 9 runs — chosen for COVERAGE, not convenience: per model the 3 rows
span different tiers/states, short vs deep seq, and healthy vs near-wall vs
host-heavy pressure; every row has a complete recorded metric row (tok/s +
HBM; RSS where the cell carries it). All asym_cpuadamwds. Rows 1/2 = C1/C2
(reuse S5 results if fresh); C3/C4/C4b still run under S5(d) regardless.
| # | model | point | state covered | reference row (in archive/) |
|---|---|---|---|---|
| 1 | q3-32b | T1 128k b2 | short, healthy (63%) | c12 system_summary §1: 906 us/tok = 1104 tok/s, 116.0 GiB |
| 2 | q3-32b | T2 128k b2 | short, mid-memory | c12 §1: 1044 us/tok, 93.6 GiB |
| 3 | q3-32b | T3 640k b1 | deep, host-heavy (RSS 980 ≈ pool) | tputw4 archives (verified): 226 tok/s · 129.7 GiB. SUBSTITUTED for 576k (no complete tok/s row there) |
| 4 | llama3.3-70b | T1 96k b1 | dense-70B T1, short/healthy | tputw7 archives (verified): 1066 tok/s · 48.9 GiB · RSS 486. SUBSTITUTED for 192k parity point (no complete T1-192k row) |
| 5 | llama3.3-70b | T2 192k b2 (KA+AU) | mid T2, near-wall 92% | tputask0 archives (verified): 543 tok/s · 171.1 GiB · RSS 963. SUBSTITUTED for 384k (measured GiB-only, no tok/s) |
| 6 | llama3.3-70b | T2 448k b1 | WALL — llama's deepest FIT | tputw6 archives (verified): 275 tok/s · 182.4 GiB peak-resv (record text: 180.1 = 97.3%) · RSS 983. (T3 EXCLUDED for llama BY DESIGN — host inversion; do NOT run llama T3) |
| 7 | q3-30b-a3b | KA dial 120k b8 (= C3) | short-ish, big-batch, KA state | scheduler_v2 §3b L3 (180.0 GiB; s/it in row). SUBSTITUTED for the 80k P1 cell: no archived asym 80k/64k b8 run exists in either tree (audit 2026-07-21) — P1's env is unreproducible verbatim |
| 8 | q3-30b-a3b | 800k b1 shed = C4b | deep, sole-survivor crossover | c14 P3 / tputasl-c14 archives: 596-597 tok/s · 147.5 GiB · RSS 539. Env verified: staged + ker000 + 5 base pins, NO KA flags |
| 9 | q3-30b-a3b | 1.1M b1 shed | ultra-deep + heaviest RSS | c14 P4 / tputschedb-c14: 382 tok/s · 151.5 GiB · RSS 906. Env verified: same shed state as row 8 (c14's "T2-BAL" label) |
Row env comes from the row's archived command.txt, verbatim. If a chosen
row's archive entry turns out to lack a complete metric triple, substitute
the nearest seq that has one and note it in the ledger.

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

- [S1-V2a C1 PASS 2026-07-21] dense-T1 q3-32b 128k b2 on the merged tree:
  1091 tok/s (ref 1104, −1.1% = in-band) · peak 116.0 GiB (ref 116.0,
  EXACT). Run: mrgs1c1 (jobs.tsv ok-row FIT).
- [S1-V2b C2 PASS 2026-07-21] dense-T2 q3-32b 128k b2: 986 tok/s (ref 958,
  +2.9% — faster than record) · peak 93.6 GiB (ref 93.6, EXACT). Run:
  mrgs1c2 FIT. Both dense tiers byte-exact on memory ⇒ port is
  behavior-identical with features off.
- [S1-V3 C3 PASS-with-note 2026-07-21] MoE KA dial 120k b8: 347.6 s/it (ref
  §3b L3 352.5 — in-band, faster) · peak resv 165.7 GiB vs ref 180.0
  (−14.3) · alloc 109.5 vs 118.0 (−8.5). DIAGNOSED per breach protocol: KA
  ENGAGED beyond doubt (fg_keep_acts_hbm true; hbm_kept_bytes_peak 39.4 GiB
  ≈ the dial's +36.4 Δ; clone counters live; −37 s/it vs the L2 no-KA rung
  = the KA speed signature). The dial's raw artifacts no longer exist in
  either tree (only the §3b table survives, dated 07-19 = BEFORE donor's
  final post-dial fixes, which are what we ported) ⇒ lower-memory delta
  attributed to donor code evolution, not a port defect. Decisive
  final-code memory gate = C4 (archive-verified 183.0 GiB @900k from the
  donor's FINAL code). Note: keep_stage_clone_bytes visible in profile ⇒
  the re-applied noclone counters flow end-to-end.
- [NAMING FIX-2 2026-07-21] C1's dir showed the flaw: appending all six
  recipe components pushed the ~250-char label past NAME_MAX → safe_label
  truncate+hash fired → OPAQUE tail (the exact outcome Kevin rejected).
  REVISED RULE: append ONLY NON-DEFAULT components (all-defaults ⇒ empty
  suffix ⇒ historical names byte-identical, pre-merge cached-run lookup
  works again; any flag delta ⇒ ≥1 non-default component ⇒ distinct dirs
  still guaranteed; hash remains the rare >255 fallback). Applied AFTER the
  S1 queue drained (no mid-flight driver edits); S1-queue run dirs carry
  the old suffix+hash — harmless (verdicts/metrics are name-agnostic).
- [DATASETS 2026-07-21] copied 3 MoE deep dataset dirs from -39 (800k n1024,
  900k n512, 1.1M n512 — bit-identical inputs for record reproduction);
  MAX_SAMPLES per replication run pinned to the archived run's value
  (rows 3/4/6 + C4 + row 9 = 512; rows 5/7/8 = 1024).
- [ARCHIVE AUDIT 2026-07-21 — MAJOR CORRECTION] grep over EVERY archived c14
  command.txt: fused-addmm, reuse-packed-x, panel-cache appear in ZERO c14
  runs — 42's ASYM_PINS were never measured; prompt.md's "900k incl.
  panel-cache" is contradicted by tputsched-c14's own command.txt. Verified
  measured envs: 900k bundle = staged+ker000+MoE-KA+attn-KA+GC-save-hbm+5
  base pins (519 tok/s / 183.0 GiB from its step_samples); 800k + 1.1M =
  shed (staged+ker000+5 pins, no KA) 596/147.5/539 + 382/151.5/906; 1.6M T3
  = ker101 no-staged + 5 pins, 287-292 tok/s / 156.1 GiB. Actions: MoE pins
  in recipes reduced to the 5 measured; T2 recipe = archived bundle env;
  C4/C4b fences corrected; MoE lines re-anchored on the archived points
  (T2 m=0.1963, T2B 0.132, T3 0.095); merge_scheduler.md §0/§1B/§2b/§2d′
  corrected. Selftest 5/5 + replay 18/18 re-PASS after re-anchoring. Also:
  no archived asym 80k/64k b8 run exists (P1 cell unreproducible verbatim)
  → S7 row 7 = the 120k×8 dial point; the 800k shed point (147.5) is OFF
  the shed line fit through 1.1M (allocator regime) — noted, probe covers.

- [S0 2026-07-21] Plan written; reference set C1–C5 fixed; bands defined.
- [S0 2026-07-21] S7 added (Kevin): 3×3 post-merge no-regression matrix,
  gate (f). Reference paths verified: in-tree `archive/s04-p1-dgx-02-c1{2,4}`
  = byte-identical snapshots of the live env/outputs records.
- [S0 2026-07-21] S7 matrix re-picked for coverage (Kevin): rows span
  tiers/states × short/deep × healthy/wall/host-heavy; llama wall row moved
  416k→448k (deepest FIT, complete metrics 275·180.1·983); MoE rows =
  80k big-batch / 800k crossover / 1.1M ultra-deep (complete c14 cells)
  — KA-dial dropped from matrix (still runs as C3 in S5(d)). Verified the
  archive records DO carry all three metrics: c12 tables are
  `seq|B|s/it|resv GiB|%HBM|RSS GB|tok/s|MFU%`; c14 cells are
  `lat s/step · tok/s · HBM GiB (%) · RSS GB` (a few cells omit RSS —
  substitution rule covers).
- [S1 CODE 2026-07-21] Six files ported (3 byte-identical to DONOR; qwen3_moe
  + activation_offload + dense carry ONLY the noclone/mutable re-apply — diff
  vs DONOR verified minimal); driver :3844 swapped in _source, attn-KA line
  ADDED to _both (:3844); keepers git-added; branch merge_sched.
- [S1-V1 PASS 2026-07-21] in-container imports OK; six flags default-off
  asserted; noclone unit semantics exact (clone default / clone flag-off /
  noclone only flag+mutable=False; counters 48/16).
- [S2+S3 PASS 2026-07-21] merged asym_scheduler.py written: SELFTEST 5/5;
  REPLAY 18/18 (all c12+c14 recorded decisions incl. llama host-exclusion,
  704k host-OOM, probe resolutions at llama 320-416k + moe 900k). Constants:
  c12 §4 lines; MoE ladder = rung-prefix states T2-bundle 6.3+0.1895 (900k
  record 0.1-GiB-exact incl. panel +6), T2B 6.3+0.1375 (1.1M), T3 4.3+0.100
  (1.6M); host caps calibrated C_HOST_EFF=990 (FIT 980-983 vs OOM ~1003);
  MoE T1 = anchor-zone only (c12 §4 fit-pending). NOTE: c12 §4's MoE T3
  k≈0.17 contradicts 42's measured 0.100 (0.17 would make 1.6M infeasible,
  which RAN) — took 0.100, discrepancy logged.
- [S4 CODE+OFFLINE-GATES 2026-07-21] preset layer inserted in BOTH drivers
  at the RUNS pre-parse loop (everything downstream sees a raw token; one
  tier|family per invocation enforced); recipe table generated
  (tier_recipes.sh); 6-component label suffix appended at the single
  config-dir label site; config.json at all 4 command.txt writers.
  S4-V1 PASS (T2 moe → 11 env; user-env-wins verified with FUSED=0; dense
  T2 → 5 env; T2B ✓; raw tokens pass through untouched); S4-V4 twin PASS
  (preset blocks byte-identical in both drivers); bash -n both OK.
- [S6 DONE 2026-07-21] fix_asym: §5a grafted + 42's 17 post-fork ledger
  entries appended under a marked section; prompt_v2_c14.md preserved;
  remaining_optimizations + c14_old archive + midterm.md + figures +
  midterm_memory tail copied; scheduler_v2: tombstone §10′ + DONOR §10
  record map appended. All S6 grep-checks pass.
- [PROCESS NOTE 2026-07-21] first C1 launch was killed and relaunched: the
  driver was edited while C1's bash was mid-flight (bash reads scripts
  lazily — offset corruption risk). Rule for the rest of the session: NO
  driver/scheduler edits while any run is in flight. All validation runs
  use the final scripts.
- [S0 2026-07-21 review-pass-1] 6 fixes applied: C4 re-anchored to the
  REAL record (900k b1 latency emission, 519 tok/s / 183.0 GiB — the
  "576k row" never existed); C4b (800k shed, 597/147.5) added; `_both`
  driver needs an ADD (has neither forwarding line); mutable sites = 11
  (up_for_act ×2); S1-V4 moved to 120k b4 (b8 infeasible ≈188 GiB) with a
  concrete artifact check; all command literals moved into unwrapped
  fences.
