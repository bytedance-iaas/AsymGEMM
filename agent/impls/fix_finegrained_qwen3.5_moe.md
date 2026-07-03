# Fix Fine-Grained Offload: Qwen3.5-35B-A3B MoE `ker101`

## Goal

Train Qwen3.5-35B-A3B LoRA-SFT with the fine-grained recompute-offload family and
compare it against the matching SuperOffload baselines at the real target workload:

```text
[q3.5-35b-a3b]="Qwen/Qwen3.5-35B-A3B"
workload: 80000|8|1        (intermediate proof point: 45000|8|1 — DONE, see below)
loss: ligerloss1
policy tuple: none|false|false|false|false|false
target artifact label: recomp-off-full-fg-ker101
route tag: route101_lora0_accfp32
```

The target IS a Qwen3 routed-kernel MoE. Unlike the llama4 and dense qwen3.5-27b
plans, the routed-kernel bits must resolve ON via the harness auto-default:

```text
ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD=1        (moefg1)
ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=1
ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=0
ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=1
route kernel code = 101
ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1          (fix_qwen3 v1)
ASYM_EXPACT_CPU_POOL_MAX_BYTES=206158430208  (192 GiB pinned pool)
ASYMM_QWEN3_MOE_FG_DA_GPU=1                  (fix_qwen3 v2)
ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1         (fix_qwen3 v2)
```

This resolution is owned by `is_qwen3_moe_routed_model`
(`scripts/lf/profile_lora_lf_test_source.sh:608-613`, extended to
`Qwen3.5-35B-A3B` and `Qwen3.5-122B-A10B` in commit `0f313c9`) plus the `full-fg`
stage branch (`:3089-3116`) and `qwen3_moe_routed_auto_default` (`:633-639`).
Put the unsuffixed `recomp-off-full-fg` in `RUNS`; the auto-default produces the
`-ker101` label for this model. Do not pin `-ker000` for the scoreboard rows.

## Current State (2026-07-03, verified — do not re-derive)

1. Every `q3.5-35b-a3b` full-fg artifact timestamped before 2026-07-02T09:11Z is
   HOLLOW: it ran `moefg0`/`ker000` because (a) the matcher only knew
   `Qwen3-30B-A3B` and (b) `apply_lf_asym_lora` did not propagate
   `_qwen3_moe_finegrained_enabled` for `qwen35_whole` blocks. Both were fixed in
   `0f313c9` (`asym_gemm/integrations/lf.py:1898-1900`). Runtime proof of the
   hole: `qwen3_moe_finegrained_offload_wrapped=0` in
   `profiling_q35_35b_a3b_s80000_20260702T003237Z/.../train.log` vs `=48` in the
   qwen3-30b runs. Treat the whole `profiling_q35_35b_a3b_*_2026070{1,2}T0*Z`
   sweep as context only, never as a scoreboard.
2. All three s80000 rows in that sweep FAILED with forward OOM at 178.3 / 183.3 /
   181.0 GiB (`failed:1` in jobs.tsv, phase `forward_exception`). The apparent
   "marginal difference" between backends at s80000 is three crash ceilings
   against the 184 GiB card, not a comparison.
3. The fixed config is numerically healthy and already measured on the canonical
   runtime (see Stage 2/3 below):

```text
Model: qwen3_5-35b-a3b   LoRA r64/a16/d0.00   b8 ga1   ligerloss1   canonical stack (SDPA)
Workload   Backend           Config                     step_H(MiB)  train_loss  grad_norm  status
---------  ----------------  -------------------------  -----------  ----------  ---------  ------
s2048.b8   asym_cpuadamwds   recomp-off-full-fg-ker101       3392.5*      1.762      0.210   ok
s45000.b8  superoffload_mem  unsloth                       160220.0       0.952      0.284   ok
s45000.b8  superoffload_mem  unsloth-off                   G-OOM (tried 484.35 GiB in SDPA)  fails >= s25000
s45000.b8  asym_cpuadamwds   recomp-off-full-fg-ker101      90300.8       0.948      0.205   ok   (-43.6% vs unsloth)
s80000.b8  superoffload_mem  unsloth                       G-OOM 178.3 GiB (fwd)
s80000.b8  superoffload_mem  unsloth-off                   G-OOM 183.3 GiB (fwd)
s80000.b8  asym_cpuadamwds   recomp-off-full-fg-ker101     G-OOM ~177 GiB (fwd, layer 39/40)  ← OPEN, Stage 4
* fa4-stack smoke value at the same shape; canonical s2048 smoke peak not re-measured.
```

   Artifacts: `profiling_fix_qwen35_moefg_sdpa_s2048_20260703T053226Z`,
   `profiling_fix_qwen35_moefg1_s45000_20260703T053819Z`,
   `profiling_fix_qwen35_moefg1_s80000_20260703T054955Z`.
4. The fg engine and the ker101 kernels are numerically exonerated for this model:
   `scripts/testing/qwen35_fg_numeric_probe.py` matches an fp32 reference at
   qwen3.5 shapes (E=256, H=2048, I=512, top_k=8) AND at s80000-scale row counts
   (`--qwen3 --tokens 655360`, R=5.24M), forward and backward, for
   plain / fg000 / fg101 / fg101+v2 / fg000+v2 (rel_fro <= ~0.8%, zero NaN).
5. The FA4 runtime (`LlamaFactory-fa4` + `.venv-fa4` + `FLASH_ATTN=fa4`) breaks
   asym-path numerics for this model: loss 13.78–14.10, grad_norm=nan
   (`outputs/fa4_qwen35_asymcpuadamwds_recompofffullfg_ker101_{smoke,s80000}_*`).
   The same config on the canonical runtime is healthy. FA4 + superoffload is
   healthy (1.677), so the break is fa4-stack × asym-wrapper interplay. That fix
   belongs to the FA4 workstream — this plan only PINS the canonical runtime.
6. The single remaining blocker for the s80000 row is a canonical-stack FORWARD
   retention pathology (Stage 4). It exists with and without moefg (identical
   ~177 GiB OOM at the same SDPA site), i.e. it predates and is independent of
   the fg fix. The fa4-stack forward is lean and completed s80000 at
   116,093.8 MiB — proof the fg config itself fits the card at s80000.

## Required Baselines

The apples-to-apples scoreboard for `asym_cpuadamwds` is the CPUAdamW /
CPU-optimizer-offload family, at the exact target workload:

```text
q3.5-35b-a3b|1 ; superoffload_mem|unsloth|ligerloss1     ; 80000|8|1 ; none|false|false|false|false|false
q3.5-35b-a3b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false
q3.5-35b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false
```

Baseline freshness and envelope rules:

1. Baseline ceilings are already established under the current harness:
   `unsloth-off` completes only <= s20000 (179,350 MiB at s20000; G-OOM from
   s25000 — at s45000 it dies trying to allocate 484.35 GiB inside SDPA).
   `unsloth` completes <= s45000 (160,220 MiB; G-OOM at s50000/60000/70000/80000).
   Both baseline s80000 rows are therefore expected `G-OOM`; report the crash
   ceilings as ceilings.
2. "Baseline OOM + target completes" only counts as a win if the target's config
   audit is clean (label `recomp-off-full-fg-ker101`, `moefg1` wrapped=40, loss
   in band). Never claim success from baseline OOM with a hollow or
   numerics-broken target — that is exactly the failure mode of the 2026-07-02
   sweep.
3. The valid sub-s80000 comparison already exists and stays in the report:
   s45000 vs `unsloth` (−43.6%). The `unsloth-off` comparison only exists
   <= s20000; produce it for 35B only if a reviewer requires that row.
4. If `asym|recomp-off-full-fg` (no CPUAdamW) is ever evaluated, compare it only
   against `superoffload_mem_nocpuadamw|unsloth[-off]`.

## Why This Is A Separate MoE Plan

Neither the Qwen3-30B-A3B plan (`fix_qwen3.md`) nor the dense qwen3.5-27b plan
(`fix_finegrained_qwen35.md`) covers this shape. Verified architecture facts
(HF config cached in run artifacts, `lf_run/config.json`):

```text
model_type qwen3_5_moe; 40 decoder layers, hybrid attention:
  30 linear_attention (gated delta-net, fla chunked kernels)
  10 full_attention (interval 4), head_dim 256
hidden_size 2048
routed experts: 256 experts, top_k 8, moe_intermediate_size 512
  gate_up_proj [E,2I,H] = [256,1024,2048]   (same gate-then-up packing as Qwen3)
  down_proj    [E,H,I]  = [256,2048,512]
shared expert per layer: shared_expert_intermediate_size 512
  + sigmoid shared_expert_gate Linear(H,1)
```

Consequences:

1. The expert engine is the SHARED `AsymQwen3Experts`
   (`asym_gemm/training/qwen3_moe.py:2042`); `AsymQwen35MoeBlock`
   (`asym_gemm/training/qwen35_moe.py:125`) calls it with the identical
   whole-router convention (`qwen35_moe.py:279` vs `qwen3_moe.py:3106`). There is
   no qwen3-vs-qwen35 branch anywhere in
   `qwen3_moe_finegrained.py`/`qwen3_moe_routed_gemm.py` — which is why any
   engine edit made "for qwen3.5" is automatically a qwen3-30b and llama4 edit
   too (see Scope Guardrails).
2. The ker101 kernels are shape-general (no hard-coded E/top_k/H/I;
   `csrc/qwen3/qwen3_moe_routed_gemm.cpp`, `sm100_bf16_asym_gemm.hpp:400-437`).
   qwen3.5's I=512 exercises the `block_k=64` transpose branch (qwen3-30b's
   I=768 uses 256) — covered by the numeric probe; keep the probe in every gate.
3. Expected wrapper counts differ from qwen3-30b (48/48/192):

```text
qwen3_moe_finegrained_offload_wrapped = 40
linear_attention_saved_tensor_offload_wrapped = 30
attention_saved_tensor_offload_wrapped = 10
attention_act_offload_wrapped = 40
qwen35_moes_wrapped = 40, qwen3_moes_wrapped = 0
```

4. The shared expert (`AsymQwen35SharedMLP`) is NOT covered by the moefg flag
   (`lf.py` sets the flag on `wrapped.experts` only) and its own offload path is
   gated on `ASYMM_EXPERT_ACT_OFFLOAD`, which `full-fg` forces to false. This is
   accepted for now: under unsloth GC the shared-expert saves exist one layer at
   a time (~2–4 GiB transient at s80000), ranked below the delta-net backward
   workspace. See Optional Levers.

## Scope Guardrails (hard rules — other models' paths must not move)

1. Harness gating stays keyed on `is_qwen3_moe_routed_model` and the `full-fg`
   stage branch. Any new behavior for this plan must be (a) inside that model
   match, or (b) behind a NEW default-off env flag. No unconditional edits to
   `run_job`'s recompute canonicalization.
2. `asym_gemm/training/qwen3_moe.py`, `qwen3_moe_finegrained.py`,
   `qwen3_moe_routed_gemm.py`, `activation_offload.py`,
   `linear_attention_activation_offload.py`, `llama4_shared_mlp.py` are SHARED
   engines. An edit there is only acceptable with the full Cross-Model
   Non-Regression Matrix (below) green in the same iteration.
3. Do not touch `_shared_mlp_activation_offload_enabled()` semantics
   (llama4 + qwen3.5 shared) as part of this plan.
4. Do not relax `validate_recompute_kernel_for_model`
   (`profile_lora_lf_test_source.sh:622-631`): dense/non-routed models must keep
   dying on `ker != 000`.
5. `Qwen3.5-122B-A10B` is matcher-included. Its shapes are unverified here; the
   engine guards (`qwen3_moe_finegrained.py:1629-1683` + kernel `DG_HOST_ASSERT`)
   fail loudly, not silently. Leave that behavior as is.
6. The pre-existing test failure
   `test_attention_activation_offload_excludes_vision_attention`
   (tests/training/test_lf_qwen3_asym_backend.py, broken at HEAD `1af5451`,
   fallout of the `_is_stateless_module` edit in `0f313c9`) must be FIXED or
   explicitly waived before this plan's code changes merge — do not let this
   plan's diffs hide behind it.

## Runtime Pin (UPDATED 2026-07-03 — unified FA4 runtime)

`resolve_current_runtime_for_model` (`profile_lora_lf_test_source.sh:758-772`)
auto-switches ANY qwen3.5 model to the LlamaFactory-fa4 FORK runtime unless each
of `FLASH_ATTN`, `LF_DIR`, `ENV_DIR` is explicitly set (three independent
`_*_ENV_SET` guards). Do NOT rely on the auto-switch: the fork's liger wiring
full-patches liger (no loss-only machinery) and collides with the asym staged
lm_head → loss ~14 (proof: fork + ligerloss0 = 1.77 healthy). The fork also
lacks `UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU`.

The validated UNIFIED runtime (canonical LF with FA4 support ported 2026-07-03,
plus the fa4 venv for the flash-attn-4 package):

```bash
FLASH_ATTN=fa4 \
LF_DIR=/workspace/AsymGEMM-SFT/third_party/LlamaFactory \
ENV_DIR=/workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv-fa4
```

Validated s2048: loss 1.76 / grad_norm 0.215 / moefg 40/40 / zero fallbacks —
parity with the canonical-SDPA smoke (1.762/0.210).

CAVEAT (2026-07-03, later the same day): at s80000 the unified-FA4 runtime
completes the step at 106,082 MiB (memory goal met) but with loss 0 /
grad_norm nan — the per-param dump (`ASYM_CPU_ADAMW_PER_PARAM_NORM_DEBUG=1`)
shows NaN in the linear_attn LoRA banks. Healthy at s2048 on the same stack ⇒
scale-triggered (s80000 crosses seq 2^16; fa4 is beta). Boundary bisect at
s45000-unified pending. Until resolved, the numerics-valid s80000 path is:

```bash
# canonical SDPA + all-ones-mask drop (LF collator, added 2026-07-03):
FLASH_ATTN=auto LF_DIR=.../LlamaFactory ENV_DIR=.../AsymGEMM/.venv \
LF_DROP_ALL_ONES_ATTENTION_MASK=1
```

REVERTED BY POLICY (2026-07-03, user decision): the diagnostic collator gates
(`LF_DROP_ALL_ONES_ATTENTION_MASK`, `LF_SKIP_DUMMY_MULTIMODAL`) and the
sequence-chunked delta rule (`ASYMM_QWEN35_DELTA_CHUNK_SEQ`, lf.py + patcher)
are NOT in the tree — scoreboards must not use execution-path tricks other
model families / baselines do not carry. The DIAGNOSTIC findings they produced
remain valid and are recorded here for whoever fixes this upstream:

1. The S² term is jointly caused by (a) the LF dummy multimodal image block
   (`data/collator.py` "avoid process hanging in zero3/fsdp case" — its M-RoPE
   position jumps trip packed-sequence detection) and (b) real right-pad tails
   (padding mask defeats `_ignore_causal_mask_sdpa`). Removing both restored
   the SDPA is_causal/enable_gqa fast path with EXACT loss parity (s2048:
   1.762/0.2124 == reference) and let s80000 asym complete at 105,948 MiB and
   even superoffload_mem|unsloth-off complete at 105,721 MiB.
2. fla `chunk_gated_delta_rule` faults/corrupts above ~70k tokens/row at
   qwen3.5-35B head shapes (probe: clean at S=70000, illegal access at
   S=75000, B=8, Hv=32, Dk=Dv=128); in-model it silently produces loss 0 /
   grad NaN on BOTH sdpa and fa4 stacks. A sequence-chunked scan with
   differentiable state carry (fla returns dh0) is bf16-exact (out_rel 0.46%,
   dq_rel 0.08% at S=40000) and clean at S=80072 — this is the upstream-worthy
   fix shape. UNTIL an approved fix exists, s80000 rows are NOT numerics-valid
   on any stack; the apples-to-apples headline block is s45000.

Dry-run proof (must show canonical `lf_dir`, no fa4 attn label, and
`recompute_label=recomp-off-full-fg-ker101`):

```bash
RUNS='q3.5-35b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false' \
FLASH_ATTN=auto LF_DIR=/workspace/AsymGEMM-SFT/third_party/LlamaFactory \
ENV_DIR=/workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv \
DRY_RUN=true PREPARE_DATASETS=false PLOT=false RUN_POST=false \
OUTPUT_ROOT=profiling_fix_qwen35_moe_dryrun RUNS_LOG=profiling_fix_qwen35_moe_dryrun/runs.log \
GPU_POOL=0 PROFILERS=source MAX_STEPS=1 WARMUP_STEPS=0 \
bash scripts/lf/profile_lora_lf_test_source.sh
```

Pass criteria:

```text
echo contains: recompute_label=recomp-off-full-fg-ker101
artifact path contains: __recomp-off-full-fg-ker101__ and __moefg1__ and route101_lora0_accfp32
artifact path does NOT contain: attnfa4
env line contains: ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD=1, ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1,
  ASYMM_QWEN3_MOE_FG_DA_GPU=1, ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1,
  ASYM_EXPACT_CPU_POOL_MAX_BYTES=206158430208, LF_DIR=<canonical>, ENV_DIR=<canonical .venv>
grad/weight offload resolve true (asym_cpuadamwds family)
```

## Evidence Discipline

Run experiments one at a time (heavy qwen3.5 offload runs take 380–800 GiB host
RSS; never overlap two of them). Use a fresh `OUTPUT_ROOT` per stage.

Before each run, write down:

```text
expected model: Qwen/Qwen3.5-35B-A3B
expected MoE status: qwen35_moes_wrapped=40, qwen3_moes_wrapped=0
expected backend:
expected recompute input: recomp-off-full-fg
expected artifact recompute label: recomp-off-full-fg-ker101
expected route kernel code: 101
expected runtime: canonical (no attnfa4, LlamaFactory, .venv)
expected CPUAdamW/optimizer-offload family:
expected fg wrapper counts: 40 / 30 / 10 / 40 (moefg / linattn / attn-saved / attn-act)
expected loss band: 0.9–1.2 (s>=20000 concat data) or 1.6–1.9 (s2048 smoke)
expected grad_norm band: < 5 (see Known Systemic Issue for the >=1e10 signature)
expected comparison baseline:
expected likely failure mode:
```

After each run, inspect:

```text
command.txt                      (env actually applied; canonical triplet present?)
train.log                        ('AsymGEMM LoRA-SFT runtime:' counter line; loss; grad_norm)
profile.json.config              (recomp_label, recomp_off_stage, route code, moefg)
memory.md / memory_breakdown_summary.json
memory_actual_peak_breakdown.csv (component attribution at actual peak)
memory_live_activation_details.csv (top live tensors at peak — retention evidence)
jobs.tsv status                  (see checker caveat below)
```

KNOWN CHECKER CAVEAT: the harness completeness check
(`profile_lora_lf_test_source.sh:3577`, `job_profile_complete`) intermittently
marks fully healthy runs `failed:1` with "Expected completed profile artifact
but found incomplete/partial profile" even though both sub-checks pass when
re-run post-hoc. Until that is root-caused, a run's verdict comes from
train.log loss/grad_norm + memory.md + the counter line — NOT from the jobs.tsv
status alone. (The inverse also holds: `ok` does not certify numerics — the
qwen3-30b s80000 runs are `ok` with grad_norm ~5e11.)

Treat the result as inconclusive if:

```text
artifact path label and profile config disagree
recomp_label is not recomp-off-full-fg-ker101
artifact path contains attnfa4 (FA4 runtime leaked in)
qwen3_moe_finegrained_offload_wrapped != 40
linear_attention_saved_tensor_offload_wrapped != 30 or attention_saved_tensor_offload_wrapped != 10
reference_fallback_count != 0 or fallback_reasons != none
UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=false, ASYMM_EXPERT_SILU_BWD_GPU=1, or ASYMM_MLP_RECOMPUTE_CHUNK != 0
UNSLOTH_GC_OUTER_HBM_EVERY_N != 0 on a scoreboard row (diagnostic-only knob)
loss ~13–14 (the FA4-stack / stale-delta-net-bug signature)
grad_norm nan, or >= 1e10 without the per-bank debug dump attached
stale artifact was reused (check run timestamps against 0f313c9)
```

Conclusion labels:

```text
validated
blocked_by_stage_bug
inconclusive_wrong_config
inconclusive_wrong_runtime (fa4 leak)
inconclusive_partial_profile
inconclusive_stale_artifact
inconclusive_unexpected_path
```

Do not advance to the next stage on an inconclusive result.

## Stage 1: Numeric Gate (no model load, minutes) — DONE; keep as regression gate

```bash
# qwen3.5 shapes, small
CUDA_VISIBLE_DEVICES=<free> .venv/bin/python scripts/testing/qwen35_fg_numeric_probe.py
# qwen3.5 shapes, step-1 condition (LoRA-B zero)
CUDA_VISIBLE_DEVICES=<free> .venv/bin/python scripts/testing/qwen35_fg_numeric_probe.py --zero-b
# qwen3-30b shapes at s80000-scale rows (cross-model + at-scale gate)
CUDA_VISIBLE_DEVICES=<free> .venv/bin/python scripts/testing/qwen35_fg_numeric_probe.py --qwen3 --tokens 655360
```

Pass criteria:

```text
verdict line: "all forward paths within 5% of fp32 reference"
every grad row vs plain: rel_fro < 0.05, nan=0
per-bank grad norms finite and O(0.01–10)
```

Unit gate:

```bash
.venv/bin/python -m pytest tests/training/test_lf_qwen35_asym_backend.py tests/training/test_lf_qwen3_asym_backend.py -q
```

Pass = everything green except the pre-existing
`test_attention_activation_offload_excludes_vision_attention` (Scope Guardrail 6).

## Stage 2: Small Config Gate s2048 — DONE 2026-07-03

Canonical runtime, one row:

```bash
RUNS='q3.5-35b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false' \
FLASH_ATTN=auto LF_DIR=/workspace/AsymGEMM-SFT/third_party/LlamaFactory \
ENV_DIR=/workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv \
GPU_POOL=<free> PROFILERS=source MAX_STEPS=1 WARMUP_STEPS=0 PLOT=false \
OUTPUT_ROOT=profiling_fix_qwen35_moe_s2048 bash scripts/lf/profile_lora_lf_test_source.sh
```

Validated result (`profiling_fix_qwen35_moefg_sdpa_s2048_20260703T053226Z`):
loss 1.762, grad_norm 0.210, moefg wrapped 40/40, ker101, zero fallbacks.

## Stage 3: s45000 Memory Gate — DONE 2026-07-03

Validated result (`profiling_fix_qwen35_moefg1_s45000_20260703T053819Z`):

```text
Workload   Backend           Config                     step_H(MiB)  act@peak  temp@peak  loss    grad_norm
s45000.b8  superoffload_mem  unsloth                       160220.0  103705.8    56514.2  0.9521  0.2842
s45000.b8  asym_cpuadamwds   recomp-off-full-fg-ker101      90300.8    2817.0    87481.9  0.948   0.2054
```

−43.6% peak HBM vs the strongest completing baseline, loss/grad parity. Peak
composition (memory_actual_peak_breakdown.csv, after_backward):
linear_attention workspace 37,329.7 MiB > norms-attributed 21,964.0 >
routed_experts 16,762.9 > reserved-unallocated 12,265.2 > attention 10,060.3.
The delta-net backward is now the dominant HBM term, not the MoE.

## Stage 4: s80000 Blocker — canonical forward retention (OPEN)

Evidence (do not re-derive):

```text
- canonical s80000 (moefg1, ker101): OOM in FORWARD at layer 39/40, SDPA alloc of
  9.78 GiB on top of 176.76 GiB allocated
  (profiling_fix_qwen35_moefg1_s80000_20260703T054955Z).
- the OLD hollow run (moefg0) OOMs at the SAME site with the SAME ~177 GiB ⇒ the
  pathology is independent of moefg and predates the fg fix.
- memory_live_activation_details.csv at the exception shows LAYER 1 tensors still
  alive at layer 40: layers.1.linear_attn.norm [20498432,128] 4.89 GiB,
  layers.1.linear_attn.in_proj_z [8,80072,4096] 4.89 GiB,
  layers.1.input_layernorm [8,80072,2048] 2.44 GiB (each listed twice).
  A checkpointed forward must not keep layer-1 activations alive at layer 40.
- the FA4-stack run of the SAME config completed s80000 with peak 116,093.8 MiB
  (act@peak 22,600.9, temp 93,491.0) — numerics broken there, but it proves the
  fg config's true step footprint fits the card. Forward leanness differs by
  runtime stack, not by fg configuration.
- both venvs run transformers 5.6.0 with identical modeling_qwen3_5_moe GC
  markers; both LF forks log "Gradient checkpointing enabled"; the LF forks'
  checkpointing.py differs only in env helpers + UNSLOTH_GC_OUTER_HBM_EVERY_N +
  the UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU save_on_cpu wrapper in backward
  (canonical has them; fa4 does not).
- implication: at <= s45000 the same retention exists but hides under the
  backward peak; a lean-forward canonical stack should land near the fa4 shape
  (~65 GiB at s45000), i.e. the current −43.6% understates the mechanism.
```

2026-07-03 UPDATE — the 2x2 isolation matrix is COMPLETE; suspects S1–S4 below are
superseded by these measurements:

```text
venv        LF fork    FLASH_ATTN   s80000 forward
canonical   canonical  auto (sdpa)  OOM 176.8 GiB   (profiling_fix_qwen35_moefg1_s80000_*)
canonical   fa4        auto (sdpa)  OOM 176.8 GiB   (profiling_q35_forkswap_s80000_*)
fa4         fa4        auto (sdpa)  OOM 176.8 GiB   (profiling_q35_fa4venv_sdpa_s80000_*)
fa4         fa4        fa4          FITS, fwd peak 81.5 GiB (outputs/fa4_..._ker101_s80000_*)
⇒ the ONLY differentiator is FLASH_ATTN=fa4 vs sdpa. LF fork and venv exonerated
  (pip-freeze core packages identical; fla fast path active in both venvs).
- decomposition from memory_breakdown.jsonl: canonical s45000 forward = 23.8 GiB
  retained at end-of-forward + ~46 GiB transient band (peak 69.9); fa4 s80000 =
  15.1 GiB retained + 66.4 transient (peak 81.5). SDPA-mode retains ~2.8x more
  per token AND has fatter transients (repeat_kv GQA expansion measured at
  +14.7 GiB/full-attn layer at s80000 vs +4.9 native-GQA; fp32 z-gate silu
  ~9.8 GiB/linear-attn layer).
- NOT asym-specific: superoffload_mem|unsloth s45000 forward peak = 72.6 GiB ≈
  asym's 69.9, and both baselines OOM'd IN FORWARD at s80000. This is a
  qwen3.5+SDPA-stack property; FA4-mode forward is what makes s80000 feasible
  for ANY backend on this model.
- ROOT CAUSE (memory-snapshot replay to the forward peak,
  profiling_q35_s45000_memsnap_*/.../memory_snapshot.pickle): NOT retention.
  The canonical s45000 forward peak (69.86 GiB) decomposes as
  sdpa_attention_forward 38.5 GiB (3 blocks) + masking_utils.and_mask
  15.14 GiB (ONE materialized [8,1,45072,45072] BOOL mask = 8*45072^2 bytes)
  + repeat_kv 5.5x2 GiB (GQA expansion, forced because a real mask disables
  enable_gqa/is_causal). The batch's padded tail makes the 2D mask non-all-ones,
  so transformers materializes the full S^2 mask and SDPA runs in its fat mode.
  At s80000 the mask alone is 47.8 GiB and the SDPA blocks ~66 GiB on top of the
  ~40 GiB fg/GC baseline by layer 39 → the observed ~177 GiB forward OOM.
  FLASH_ATTN=fa4 handles padding via varlen (no S^2 mask, native GQA) → lean.
  The earlier "layer-1 tensors alive at layer 40" live-details reading was
  allocator storage-reuse mislabeling.
- FIX (2026-07-03): FA4 support ported from the LlamaFactory-fa4 fork into
  canonical LlamaFactory (extras/constants.py AttentionFunction.FA4,
  model_utils/attention.py dispatch, data/collator.py Literal + neat-packing
  guard; ~12 lines, additive, sdpa/fa2 untouched). Target runtime for s80000:
  LF_DIR=canonical + ENV_DIR=.venv-fa4 + FLASH_ATTN=fa4 — canonical recomp-off
  machinery (save-on-cpu etc., which the fa4 fork LACKS) plus the lean fa4
  forward. The LlamaFactory-fa4 fork is retired for these scoreboards.
```

Historical suspects (for reference; S1–S4 largely falsified by the matrix —
the wrapper/GC/fla-version arms were equal across stacks):

```text
S1  linear-attention saved-tensor offload wrapper not offloading (or offloading
    while something else pins the GPU storage) during the grad-enabled pass on
    canonical. Discriminator: dump the wrapper's skipped_tensors/skipped_bytes
    stats and diff canonical vs fa4 at s45000; also A/B ASYMM_ATTN_ACT_OFFLOAD=false
    to see whether retention follows the attention wrappers.
S2  unsloth GC first pass not actually detaching for the hybrid layers on the
    canonical LF (grad-enabled first forward ⇒ per-layer saves accumulate).
    Discriminator: env-gated pack-hook counter inside layer 1 vs layer 30 during
    pass 1 on both stacks (temporary instrumentation, remove after).
S3  LF get_custom_gradient_checkpointing_func requires
    any(param.requires_grad) on the layer; with weight-offloaded 0-numel LoRA
    banks the emptiness/order could skip checkpointing for SOME layers on one
    stack. Discriminator: count UnslothGradientCheckpointing.apply calls per
    pass (expect 40).
S4  D2H offload backlog (async copies pinning sources). Retention is
    layer-1-at-layer-40, i.e. 39 layers deep — a backlog would be a few layers
    at most; deprioritized but cheap to check via copy-event timestamps.
```

Rules for the fix:

```text
- if the fix lands in the canonical LF fork's checkpointing/unsloth path, it must
  be env-gated or qwen3.5-conditional and leave qwen3-30b / dense resolution
  byte-identical (fix_qwen3.md and dense-doc scoreboards must not move).
- if the fix lands in an asym_gemm wrapper, the Cross-Model Non-Regression
  Matrix gates it (shared code).
- UNSLOTH_GC_OUTER_HBM_EVERY_N is a diagnostic knob only; scoreboard rows keep 0.
- alternatively, if the FA4 workstream fixes asym numerics on its stack first,
  the scoreboard may run there — but then the fa4 runtime must pass Stage 1's
  probe AND Stage 2/3 loss bands on that stack before any s80000 claim.
```

## Stage 5: Final s80000 Scoreboard

Only run after Stage 4 is validated (canonical forward peak at s80000 stays
under ~130 GiB before backward starts, or an approved alternate runtime).

Run each row separately, sequentially (host RSS):

```bash
RUNS='q3.5-35b-a3b|1 ; superoffload_mem|unsloth|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false' \
FLASH_ATTN=auto LF_DIR=/workspace/AsymGEMM-SFT/third_party/LlamaFactory \
ENV_DIR=/workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv \
OUTPUT_ROOT=profiling_fix_qwen35_moe_s80000 GPU_POOL=<free> PROFILERS=source \
MAX_STEPS=1 WARMUP_STEPS=0 PLOT=false bash scripts/lf/profile_lora_lf_test_source.sh

RUNS='q3.5-35b-a3b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false' \
# ...same env... 
bash scripts/lf/profile_lora_lf_test_source.sh

RUNS='q3.5-35b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false' \
# ...same env, plus:
ASYM_CPU_ADAMW_PER_PARAM_NORM_DEBUG=1 \
bash scripts/lf/profile_lora_lf_test_source.sh
```

Success criteria:

```text
target completes with loss in band (0.9–1.2) and moefg wrapped 40/40, ker101
target grad_norm sane OR the per-bank debug dump attached and the anomaly filed
  under the Known Systemic Issue (do not silently accept clipped 1e10+)
target peak HBM reported with full breakdown; expected order ~110–125 GiB
baselines reported as G-OOM ceilings (expected; Required Baselines rule 2)
target CPU RSS within host budget (run sequentially)
```

## Known Systemic Issue: long-seq grad_norm explosion (tracked, not a gate)

```text
signature: loss healthy, grad_norm finite but 1e10–1e12, clipping engages
           (clip_coef ~2e-12), 1-step runs look "ok".
affected:  qwen3-30b s80000 ALL fg variants (2.9e10–7.6e11, incl. pre-v1 plain fg)
           AND superoffload_mem|unsloth baseline (qwen2.5-72b s40000: 7.0e10,
           path=default — no AsymGEMM code in the loop).
healthy:   qwen3-30b <= s50000 (0.49–0.68), qwen2.5-32b s50000 (0.11),
           qwen3.5-35b <= s45000 (0.205).
exonerated: the fg expert engine (Stage 1 probe at R=5.24M matches fp32).
tooling:   ASYM_CPU_ADAMW_PER_PARAM_NORM_DEBUG=1 prints top-16 per-bank norms at
           clip time (asym_gemm/training/cpu_adam.py, asym_cpu_adamw_clip_grad_norm_).
handling:  memory scoreboards stand (peaks are measured independently of the
           clip), but no multi-step/convergence claim at these workloads until
           root-caused. Baselines are equally affected — never frame it as an
           asym-only defect.
```

## Optional Memory Levers (post-scoreboard only, each gated separately)

```text
L1  ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=0 — trades ~2x[R,I] HBM
    (~10 GiB at s80000: R=5.12M, I=512) for a CPU round-trip. Free knob,
    A/B at s45000 first (report step_s and step_H together).
L2  delta-net backward workspace (37.3 GiB at s45000, scaling with S) — requires
    fla-side chunk-level recompute or saved-state offload; qwen3.5-only by
    construction but touches the linear-attention wrapper; needs its own plan.
L3  shared-expert activation offload under fg — deferred: one-layer-at-a-time
    transient (~2–4 GiB at s80000), and the enabling predicate is shared with
    llama4 (Scope Guardrail 3).
```

## Cross-Model Non-Regression Matrix (required for ANY shared-code edit)

```text
1. Stage 1 probe, both shape sets (default and --qwen3 --tokens 655360): clean.
2. pytest tests/training/test_lf_qwen35_asym_backend.py
         tests/training/test_lf_qwen3_asym_backend.py: green modulo the one
         pre-existing vision failure (Guardrail 6).
3. qwen3-30b spot runs (canonical stack):
   s20000 ctl band: loss 1.775 ± 0.05, grad_norm ~0.49
   s80000 ker101 band: step_H 80,521 MiB ± ~3%, loss 1.689 ± 0.05
   (grad_norm there is the Known Systemic Issue; record, don't gate)
4. dense qwen3.5-27b s30000 gate from fix_finegrained_qwen35.md: step_H
   71,228.31 MiB, loss band 1.10–1.15, grad_norm ~0.17, dense counters fire
   (dense_mlp_finegrained_offload_wrapped=64), moefg stays 0.
   (The historical loss~13 was the zero-centered `Qwen3_5RMSNorm` keying bug,
   fixed 2026-07-03 in `asym_gemm/training/offload.py:402` — see the norm
   caveat in fix_finegrained_qwen35.md. Validated:
   profiling_dense27b_fixed_s30000_* → loss 1.143 at 71,228.31 MiB, −38.7% vs
   superoffload|unsloth 116,170.70 MiB.)
5. llama4-scout: no run required if no shared file changed semantics for
   non-matched models; otherwise the fix_finegrained_llama4.md Stage-1 gate.
```

## Reporting Format

Plain-text table, generated artifact labels, these metrics only:

```text
fwd_s  bwd_s  opt_s  step_s  fwd_H  bwd_H  step_H  RAM  loss  grad_norm
```

```text
asym_cpuadamwds    recomp-off-full-fg-ker101
superoffload_mem   unsloth
superoffload_mem   unsloth-off
```

Do not report a run as final unless the artifact audit proves it is
`q3.5-35b-a3b`, `ligerloss1`, `80000|8|1`, label `recomp-off-full-fg-ker101`,
`moefg1` wrapped 40/40, canonical runtime (no `attnfa4` in the path), loss in
band, and the jobs.tsv-status caveat has been applied.
