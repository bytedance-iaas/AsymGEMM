# Fix Fine-Grained Offload: Llama-4-Scout MoE `ker000`

## Goal

Train Llama-4-Scout LoRA-SFT with the fine-grained recompute-offload family and compare
it against the matching SuperOffload baselines at the real target workload:

```text
[llama4-scout]="meta-llama/Llama-4-Scout-17B-16E"
workload: 9500|8|1
loss: ligerloss1
policy tuple: none|false|false|false|false|false
target artifact label: recomp-off-full-fg-ker000
```

The target is a MoE model, but it is NOT the Qwen3-30B-A3B routed-kernel MoE. All Qwen3
routed-kernel bits must stay off:

```text
ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=0
ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=0
ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=0
route kernel code = 000
```

This is already enforced by the harness: `is_qwen3_moe_routed_model` matches only the
substring `Qwen3-30B-A3B` (`scripts/lf/profile_lora_lf_test_source.sh:608-613`), and
`validate_recompute_kernel_for_model` dies on any `ker != 000` for every other model
(`profile_lora_lf_test_source.sh:628-629`). Llama-4-Scout therefore can only resolve to
`recomp-off-full-fg-ker000`. Do not try to relax that guard as part of this plan.

Put `recomp-off-full-fg-ker000` directly in the `RUNS` recompute field. The unsuffixed
`recomp-off-full-fg` canonicalizes to the same thing for llama4-scout, but the explicit
suffix keeps intent and artifacts from drifting:

```text
llama4-scout|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 9500|8|1 ; none|false|false|false|false|false
```

The script must generate the `-ker000` label in artifact paths, `RUN_ID`, echo output,
and `profile.json.config.recomp_label`, with route tag `route000_lora0_accfp32`.

## Required Baselines

The apples-to-apples scoreboard for `asym_cpuadamwds` is the CPUAdamW /
CPU-optimizer-offload family, at the exact target workload:

```text
llama4-scout|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 9500|8|1 ; none|false|false|false|false|false
llama4-scout|1 ; superoffload_mem|unsloth|ligerloss1 ; 9500|8|1 ; none|false|false|false|false|false
llama4-scout|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 9500|8|1 ; none|false|false|false|false|false
```

Baseline freshness rules:

1. A completed `superoffload_mem|unsloth|ligerloss1` llama4-scout artifact exists at
   s9500.b8 (`profiling_both/.../llama-4-scout-17b-16e__gpus1__b8_s9500_ga1_w1_s3_r64_a128_drop000/...`):
   step_H 177018.35 MiB (~172.9 GiB), peak reserved 184508 MiB (at the HBM edge; the
   s10000 attempt failed G-OOM), RSS peak ~539 GiB. It was produced with `r64/a128`
   under an older harness. Treat it as context only. The scoreboard rows must all be
   produced fresh under the current harness defaults (`r64/a16/d0.00`) so LoRA config,
   liger wiring, and profile fields are identical across rows.
2. `superoffload_mem|unsloth-off` has NEVER been run for llama4-scout at any sequence
   length (only `unsloth` and `recomp` exist). It must be produced, not assumed.
3. Prior sweep ceilings for llama4-scout (commented RUNS table,
   `profile_lora_lf_test_source.sh:87-102`): `unsloth` fits 9500 (G-OOM 10k),
   `unsloth-off` fits ~14500 (G-OOM 15k), `recomp` fits 8000 (G-OOM 9k). So s9500 is the
   top of the `unsloth` envelope and both baselines complete there.

If the no-CPUAdamW backend `asym|recomp-off-full-fg-ker000` is ever tested, compare it
only against `superoffload_mem_nocpuadamw|unsloth[-off]`, never against the CPUAdamW
rows (see the cross-check section).

The required reported table must include at least:

```text
Model: llama4-scout    LoRA: r64/a16/d0.00
Workload  Backend           Config                     fwd_s  bwd_s  opt_s  step_s  fwd_H  bwd_H  step_H    RAM
--------  ----------------  -------------------------  ---------------------------  ------------------------  -----
s9500.b8  superoffload_mem  unsloth        [lg+ sd-]
s9500.b8  superoffload_mem  unsloth-off    [lg+ sd-]
s9500.b8  asym_cpuadamwds   recomp-off-full-fg-ker000  [lg+ sd-]
```

## Why This Is A Separate MoE Plan

This plan is neither the dense `ker000` plan (qwen3-32b / qwen2.5-32b / llama3.3-70b /
qwen3.5-27b) nor the Qwen3-30B-A3B routed-kernel plan. Llama-4-Scout has a third shape.

Verified architecture facts (vendored
`transformers/src/transformers/models/llama4/configuration_llama4.py:138-166`,
`modeling_llama4.py:63-104,163-165,419-423`):

```text
48 decoder layers; interleave_moe_layer_step=1 -> EVERY layer is MoE
   (no standalone dense decoder MLP anywhere; contrast Maverick which interleaves)
per layer: 16 routed experts as fused 3D banks
   gate_up_proj [E,H,2I] = [16,5120,16384]   # "in_out" layout, opposite of Qwen3 [E,2I,H]
   down_proj    [E,I,H]  = [16,8192,5120]
top-1 sigmoid router (num_experts_per_tok=1); the score multiplies the expert INPUT,
   not the output; routed rows R == tokens M (76,000 at s9500.b8), unlike qwen3-30b
   top-8 where R == 8M
always-on ungated shared expert per layer: Llama4TextMLP gate/up/down with I=16384,
   added to the routed sum
attention: qk-norm, NoPE layers, chunked attention (8192), attn temperature tuning —
   all downstream of the q/k/v/o projections, so orthogonal to projection wrappers
multimodal checkpoint (Llama4ForConditionalGeneration); vision tower frozen and
   excluded by the vision path markers
hidden_act = silu; hidden 5120 and intermediate 8192 are both %64
```

### What `recomp-off-full-fg-ker000` resolves to for llama4-scout today

The harness parses the target row cleanly today. The stage resolver gives llama4-scout
the dense-like `full-fg` bundle (`profile_lora_lf_test_source.sh:3036-3056,3084-3094`):

```text
USE_UNSLOTH_GC=true
UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=true
ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD=1     # env set, but wraps 0 modules on Scout (below)
ASYMM_DENSE_MLP_SURGICAL_OFFLOAD=0
ASYMM_EXPERT_ACT_OFFLOAD=false
ASYMM_ATTN_ACT_OFFLOAD=true
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu
ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD=0     # re-enabled only for Qwen3-30B-A3B (:3089-3091)
route bits 0/0/0; ASYMM_EXPERT_SILU_BWD_GPU=0; ASYMM_MLP_RECOMPUTE_CHUNK=0
ASYMM_LAYER_ACT_OFFLOAD=false; ASYMM_LAYER_GC=false; ASYMM_ATTN_SDPA_RECOMPUTE=false
```

So the Phase-A composition actually measured is:

```text
outer Unsloth GC (original forward saves only the CPU layer input)
+ outer save_on_cpu / unsloth-off recompute saved tensors (all 48 layers)
+ AsymGEMM CPU-resident frozen/base weights:
    attention q/k/v/o     -> AsymActivationOffloadLoRALinear (CPU-left LoRA-A, CPU U/S handles)
    routed expert banks   -> AsymLlama4Experts on packed in_out banks (plain expert body)
    shared expert         -> AsymLlama4SharedMLP with CPU-adopted gate/up/down base
                             weights (adopt_host_weight, llama4_shared_mlp.py:525-549);
                             plain forward, because its act-offload Function is gated by
                             ASYMM_EXPERT_ACT_OFFLOAD which full-fg forces false
    router                -> AsymLlama4Router, no-grad "whole" mode, excluded from LoRA
+ Asym CPUAdamW (DeepSpeed cpu_adam) with grad_offload=true, weight_offload=true
+ no MoE fine-grained function, no dense fine-grained wrapper, no route kernels
```

Verified mechanics of the Phase-A routed-expert path (this is what the memory artifacts
will show, so know it before reading them):

- During backward recompute the experts run the plain grouped body:
  `gate_up = self.gate_up_base(x, ...)` builds a fused route-space `[M,2I]` tensor and
  then `chunk(2, dim=-1)` (`qwen3_moe.py:2312-2313`), followed by separate gate/up LoRA
  adds and `silu(gate)*up` (`qwen3_moe.py:2347-2357`). At s9500.b8 that fused transient
  is `[76000,16384]` bf16 ~= 2.4 GiB and is an ALLOWED live transient. The recompute
  saved tensors themselves are covered by outer `save_on_cpu`.
- Base weights are staged host->device per grouped GEMM by `AsymGroupedFrozenLinear`
  (generic `asym_forward_calls` / `asym_dx_calls` counters fire); no full-layer ZeRO-3
  style gather.
- Because no custom expert Function runs, expert act-offload counters
  (`expact_lora_a_forward_*`) and all `qwen3_moe_finegrained_*` /
  `qwen3_moe_routed_*` counters must stay zero.
- Shared-expert wrapping requires `selection.shared_experts` and all three leaves
  LoRA-targeted (`lf.py:1771-1797`); both hold under the profile defaults
  (`ASYM_OFFLOAD_MODULES=all`, `profile_lora_lf_test_source.sh:197`; `lora_target=all`).
  If the setup report shows the shared expert unwrapped, the offload-module selection
  was wrong — audit `ASYM_OFFLOAD_MODULES` before blaming the model code.

Three code facts make this mechanically different from every existing plan, and they are
the reason Phase A must be measured before any implementation work:

1. The dense fine-grained path wraps ZERO modules on Scout. The installer gate is
   `backend == "asym" and not expert_prefixes and env` (`asym_gemm/integrations/lf.py:2005-2012`),
   and every wrapped Llama4 MoE appends to `expert_prefixes` (`lf.py:1830`). Scout is
   all-MoE, so there is no standalone dense MLP anyway. The dense plans' proof
   (`dense_mlp_finegrained_offload_wrapped > 0`) is IMPOSSIBLE here and its absence is
   correct, not a bug.
2. The MoE fine-grained path is not wired for llama4. The engine-level machinery exists
   and is scaling-agnostic — the base `AsymQwen3Experts.forward_input_scaled` already
   dispatches to `_forward_qwen3_moe_finegrained_offload` with `input_weighted=True`
   (`asym_gemm/training/qwen3_moe.py:2866-2885`) — but:
   - the installer sets `_qwen3_moe_finegrained_enabled` only in the Qwen3 branches
     (`lf.py:1924-1926`, `lf.py:1983-1985`); the llama4 branch (`lf.py:1928-1957`) never
     sets it;
   - `AsymLlama4Experts.forward_input_scaled` OVERRIDES the base method without the
     fine-grained branch (`asym_gemm/training/llama4_experts.py:935-990`: only
     act-offload / expert-gc / expert-recompute / plain body);
   - the base-split helper `_ensure_qwen3_moe_finegrained_bases` slices the fused bank
     as `fused[:, :I, :]` / `fused[:, I:, :]`, valid only for Qwen3 `[E,2I,H]`
     (`qwen3_moe.py:2492-2526`); llama4 banks are `in_out` `[E,H,2I]`
     (`llama4_experts.py:808-832`).
   The harness is self-consistent about this: it forces
   `ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD=0` for llama4, and the post-run validator
   expects `moefg=false` for any model that is not Qwen3-30B-A3B
   (`profile_lora_lf_test_source.sh:1436-1443`).
3. Route kernels stay off and the label stays `ker000`. The routed kernels assume the
   Qwen3 layout and are scoped to Qwen3-30B-A3B by both `_check_supported_base`
   (`asym_gemm/training/qwen3_moe_routed_gemm.py:105-111`) and the script guard. Note
   that llama4 top-1 routing makes route-space `[R,H]` tensors 8x smaller relative to
   qwen3-30b top-8, so the pressure those kernels exist to remove is much weaker here.

Precedent that a ker000 MoE target is legitimate: `qwen3_5-35b-a3b` (MoE) completed
`asym_cpuadamwds|recomp-off-full-fg-ker000` with `moefg=0` at s45000.b8
(step_H 112133.00 MiB, RSS 395095.56 MiB, `profiling_q35_35b_a3b_*`). Phase A for
llama4-scout is the same composition; the routed experts rely on CPU-resident banks plus
outer `unsloth-off` coverage of the recompute graph.

Therefore: Phase A measures exactly what the label resolves to today, with NO model-code
changes. Phase B (wiring llama4 MoE fine-grained) exists at the end of this document and
is entered only if Phase A artifacts prove the routed experts are the remaining blocker.

### Route kernels (`ker101`/`ker111`): quantified — why Scout stays `ker000`

The Qwen3-30B-A3B routed kernels exist to stop route-space `[R,H]` tensors from ever
being written to HBM (design: `agent/impls/fused_grouped_scatter.md`; kernels:
`asym_gemm/training/qwen3_moe_routed_gemm.py`):

```text
bit 1  FWD_SCATTER        down-base forward + scatter-add   avoids down_out    [R,H]
bit 2  DOWN_DX_GATHER     down-base dX with fused gather    avoids grad_routes [R,H]
bit 3  GATEUP_DX_SCATTER  gate/up-base dX + scatter-add     avoids dx_routes   [R,H]
```

They work, and they are validated — for Qwen3-30B-A3B:

- staged bring-up route=000 -> 100 -> 110 -> 111 with per-kernel acceptance = "the
  exact `[R,H]` owner disappears from the memory decomposition"
  (`fused_grouped_scatter.md` Stages 1-6);
- the harness auto-default for q3-30b full-fg is `ker101`
  (`profile_lora_lf_test_source.sh:3103-3106`). The middle bit's kernel passed its own
  gate but stays off by default: its `[R,H]` owner (down backward `grad_routes`) is not
  at/near the peak in the accepted schedule, and the design doc's own rule is that
  end-to-end peak reduction comes only from owners live at the peak. It remains
  re-enableable per run via env if the peak owner ever shifts;
- artifact evidence at s80000.b8: non-routed 112.9 GiB vs routed 73.9 GiB step_H
  (-39.1 GiB, -34.6%).

Why the win is that big on qwen3-30b and cannot be on Scout — the route-space
multiplier:

```text
R = M x top_k;    one [R,H] bf16 tensor = M x top_k x H x 2 bytes

qwen3-30b-a3b: top_k=8, H=2048  -> 32 KiB of route-space [R,H] per token
  s50000.b8 (M=400k): 12.2 GiB per tensor
  s80000.b8 (M=640k): 19.5 GiB per tensor
  (the observed -39.1 GiB at s80000 is consistent with exactly two [R,H]
   owners live at the peak: 2 x 19.5 GiB)

llama4-scout:  top_k=1, H=5120  -> 10 KiB of route-space [R,H] per token
  s9500.b8  (M=76k):  0.72 GiB per tensor
  s14500.b8 (M=116k): 1.11 GiB per tensor
```

The point is structural, not a tuning gap: Scout is top-1, so `R == M` and route space
IS token space. The 8x row multiplication that creates qwen3-30b's 12-20 GiB route
tensors does not exist on any Llama4 model (Maverick is also `num_experts_per_tok=1`).
A Scout `[R,H]` is the same size class as one residual-stream `[M,H]` activation, and
it is SMALLER than the tensors the fine-grained schedule already manages with CPU
handles: `[R,I]` gate/up/act are 1.16 GiB each at s9500, the shared-expert act
`[M,16384]` is 2.32 GiB. Removing all three `[R,H]` owners would save at most ~2.2 GiB
even if they were simultaneously live (they are not — they occur at different schedule
points), against a target peak around 100 GiB: roughly 1%, where the same kernels
bought qwen3-30b 35%.

Cost side, if someone tried anyway:

- the kernels are SM100 kernels compiled for the qwen3 `out_in` operand layout with
  `compiled_dims='nk'` (`_check_supported_base`, `qwen3_moe_routed_gemm.py:105-111`);
  llama4's split bases are `in_out`, so this is new kernel variants — "real kernel
  work" per the route-kernel doc's own summary, not a flag flip;
- the harness/validator scoping (`ker != 000` dies for non-q3-30b) would need the same
  symmetric extension as the moefg opt-in;
- for top-1 the weighted scatter-ADD epilogue is not even needed in weighted form:
  llama4 folds the router score into the packed INPUT, and each token receives exactly
  one routed contribution, so the scatter degenerates to a permutation write and the
  fp32-accumulation requirement (`ROUTE_ACCUM_DTYPE=fp32`) loses its main numeric
  rationale;
- the traffic argument is equally marginal: `index_select`/`index_add` over 0.72 GiB
  per layer per direction is noise against a fine-grained step already dominated by
  CPU staging (the q3-32b full-fg row is ~1.5x slower than unsloth-off end to end).

Decision rule: llama4-scout stays `ker000` in Phase A AND Phase B. Revisit only if a
Phase-B memory decomposition shows a route-space `[R,H]` tensor as a top peak owner —
by the arithmetic above that requires Scout workloads near s40000.b8 (M=320k, ~3 GiB
per tensor) or beyond, which is past both baselines' ceilings and outside this plan's
scope.

### Config facts the final target must prove

```text
config.recomp_label = recomp-off-full-fg-ker000
config.recomp_off_stage = full-fg
config.use_unsloth_gc = true
config.unsloth_gc_recompute_save_on_cpu = true
config.asymm_dense_mlp_finegrained_offload = 1
config.asymm_dense_mlp_surgical_offload = 0
config.asymm_expert_act_offload = false
config.asymm_attn_act_offload = true
config.asymm_expert_act_offload_lora_a_fwd = cpu
config.asymm_expert_silu_bwd_gpu = 0
config.asymm_mlp_recompute_chunk = 0
config.unsloth_gc_outer_hbm_every_n = 0
config.asymm_qwen3_moe_finegrained_offload = 0        # Phase A
config.asymm_qwen3_moe_route_kernel_code = 000
config.asym_offload_modules = all                      # else shared expert stays unwrapped
grad_offload = true, weight_offload = true            # asym_cpuadamwds family default
training_bf16 = false                                  # llama4+asym auto-guard; record it
```

### Setup report facts the target must prove (MoE wrapping, not dense wrapping)

```text
llama4_moes_wrapped = 48
qwen3_moes_wrapped = 0
qwen35_moes_wrapped = 0
dense_mlp_finegrained_offload_enabled = false   # installer gate `not expert_prefixes`
dense_mlp_finegrained_offload_wrapped = 0       # env stays 1; this asymmetry is expected
dense_mlp_act_offload_wrapped = 0
qwen3_moe_finegrained_offload_wrapped = 0       # Phase A
attention_act_offload_wrapped > 0
router mode = whole (router no-grad; router weights excluded from LoRA)
```

Do not apply the qwen35 dense checklist verbatim: there,
`dense_mlp_finegrained_offload_wrapped > 0` is the required proof; here it must be 0 and
`llama4_moes_wrapped=48` is the proof. An auditor who flags `wrapped=0` as a failure has
misread the model topology.

## Known Llama4-Specific Facts And Risks

1. `TRAINING_BF16` auto-guard. `run_lf_lora_sft.sh:486-492` auto-sets
   `TRAINING_BF16=false` for llama4 + asym backends (`asym_cpuadamwds` normalizes to
   `BACKEND=asym` at `:382-387` before the guard); baselines (`BACKEND=torch`) get
   `TRAINING_BF16=true`. This is pre-existing llama4-port behavior, shared by every
   prior llama4 asym artifact. Verified mechanism: the flag feeds only the trainer arg
   `--bf16` (`run_lf_lora_sft.sh:1902`). With `bf16=false` LF leaves
   `model_args.compute_dtype` unset (`LlamaFactory/src/llamafactory/hparams/parser.py:781-784`)
   and the patcher then infers it from the checkpoint `torch_dtype`
   (`model/patcher.py:315-319`) — Scout is a bf16 checkpoint, so the model, expert
   banks, and packed activations stay bf16. Prior llama4 asym runs exercised the
   bf16-only expert act-offload kernels successfully, which corroborates this. The
   flag's real effect is trainer-side (no bf16 AMP flag, TrainingArguments.bf16=False).
   Do not silently flip it. Requirements:
   - every row's profile must record `training_bf16`;
   - the Stage-1 target artifact must still confirm the actual parameter/activation
     dtypes from the memory-breakdown dtype attribution (do not rely on the inference
     above alone);
   - if any dtype inflation (not placement) turns out to dominate the target's peak or
     RSS, escalate the guard as its own decision item with a parity test, in a separate
     run — never bundled into a scoreboard row.
2. CPU RSS budget. The box has ~958 GiB usable CPU RAM across the two Grace nodes. The
   only completed llama4 asym ligerloss1 artifact (s4096.b8, old-style label,
   `profiling_both_c/...`) already peaked at RSS ~904353 MiB (~883 GiB). Phase A adds
   `unsloth-off` CPU saved tensors and CPUAdamW state on top of the CPU-resident base
   weights (~218 GB of frozen banks at bf16, more if host copies are wider under
   `training_bf16=false`). Run rows STRICTLY sequentially, never in parallel with any
   other heavy-offload job, and audit `rss_peak` at every stage before moving up in
   sequence length.
3. Trainable surface. LoRA r64 over 48 layers of fused expert banks yields ~2.19B
   trainable params (~2.09B in expert LoRA banks; the old s9500 baseline's
   `optimizer_memory_preflight` reported ~35 GB for fp32 moments+master and flagged the
   large-surface warning). Keep `grad_offload=true, weight_offload=true` (the family
   default used by every other model's full-fg row; artifact suffix
   `gradofftrue__weightofftrue`). Weight-offload single-owner registration for the
   packed llama4 banks is already covered by
   `test_lf_qwen3_asym_backend.py::...llama4_replaces_source_experts_with_single_cpu_owner`.
4. Liger is real for llama4 on the main stack, and only there. The fa4 stack switch is
   qwen3.5-only (`run_lf_lora_sft.sh:222-234`), so llama4 runs use
   `third_party/LlamaFactory` + `.venv`: baselines apply LF native loss-only liger
   (`llama4`/`llama4_text` in `_LOSS_ONLY_SUPPORTED_MODEL_TYPES`,
   `LlamaFactory/src/llamafactory/model/model_utils/liger_kernel.py:30-41`, backed by
   the vendored fork's `apply_liger_kernel_to_llama4`,
   `Liger-Kernel/src/liger_kernel/transformers/monkey_patch.py:432`); the asym row goes
   through `install_asym_liger_llama4_loss_bridge`
   (`asym_gemm/integrations/liger_loss.py:560-585`, dispatched at `:973-989`). Every
   row's audit must prove liger actually applied (train.log apply message / bridge
   install log). A row where `ligerloss1` silently no-ops is inconclusive — at s9500.b8
   the unfused CE logits alone would add tens of GiB and poison the comparison.
5. Vision tower. The checkpoint is multimodal; the text-only profile must keep vision
   handling IDENTICAL across all rows (frozen tower; asym wrappers already skip vision
   paths via `_VISION_PATH_MARKERS`, `lf.py:145-150`). Record
   `ASYM_GEMM_LF_DROP_FROZEN_VISION` if it is set; do not introduce it for one family
   only.
6. Expected path labels per family (from existing artifacts): target
   `...__routerwhole__...__ligerloss1__gradofftrue__weightofftrue`, baselines
   `...__routerhf__...__ligerloss1`. GPU count is 1 for all rows (`llama4-scout|1`;
   `asym_cpuadamwds` forces 1 GPU regardless).

## Stage 0: Label And Guard Proof

The `llama4-scout` alias already exists in both wrappers
(`profile_lora_lf_test_source.sh:48`, `profile_lora_lf_test_both.sh:48` — the files are
byte-twins). Verify, do not re-add. Do not run any GPU job until both dry-run proofs
below pass.

Positive dry-run proof:

```bash
RUNS='llama4-scout|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 128|1|1 ; none|false|false|false|false|false' \
DRY_RUN=true PREPARE_DATASETS=false PLOT=false RUN_POST=false \
OUTPUT_ROOT=profiling_fix_llama4_dryrun RUNS_LOG=profiling_fix_llama4_dryrun/runs.log \
GPU_POOL=0 PROFILERS=source MAX_STEPS=1 WARMUP_STEPS=1 \
bash scripts/lf/profile_lora_lf_test_source.sh
```

Pass criteria:

```text
model resolves to MODEL_NAME_OR_PATH=meta-llama/Llama-4-Scout-17B-16E, TEMPLATE=llama4
echo contains: recompute_label=recomp-off-full-fg-ker000
artifact path contains: __recomp-off-full-fg-ker000__ and __routerwhole__
route tag contains: route000_lora0_accfp32
RUN_ID contains: recomp-off-full-fg-ker000 ... moefg0 ... gradofftrue__weightofftrue
command.txt contains: ASYM_GEMM_LF_CONFIG_RECOMP_LABEL=recomp-off-full-fg-ker000
command.txt contains: ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_ROUTE_KERNEL_CODE=000
command.txt contains: ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD=0
command.txt contains: UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=true, ASYMM_EXPERT_SILU_BWD_GPU=0
```

Negative guard proof:

```bash
RUNS='llama4-scout|1 ; asym_cpuadamwds|recomp-off-full-fg-ker101|ligerloss1 ; 128|1|1 ; none|false|false|false|false|false' \
DRY_RUN=true PREPARE_DATASETS=false PLOT=false RUN_POST=false \
OUTPUT_ROOT=profiling_fix_llama4_dryrun_bad RUNS_LOG=profiling_fix_llama4_dryrun_bad/runs.log \
GPU_POOL=0 PROFILERS=source MAX_STEPS=1 WARMUP_STEPS=1 \
bash scripts/lf/profile_lora_lf_test_source.sh
```

Pass criteria for the negative guard:

```text
script exits nonzero before writing a valid training command
error states recomp-off-full-fg-ker101 is only supported for Qwen3-30B-A3B routed MoE
and that llama4-scout must use recomp-off-full-fg-ker000
```

Do not continue if the positive dry run shows any route bit set, `moefg1`, a dense-model
guard firing, or the negative guard accepting `ker101`.

## Evidence Discipline

Run experiments ONE AT A TIME, strictly sequentially — this model's offload runs are CPU
RAM heavy (risk item 2) and concurrent jobs corrupt both RSS numbers. Use a new
`OUTPUT_ROOT` per stage so artifacts are never overwritten. For validation gates set
`PROFILE_MEMORY_BREAKDOWN=true PROFILE_MEMORY_SNAPSHOT=true PROFILE_SYNC=true`.

Before each run, write down:

```text
expected model: meta-llama/Llama-4-Scout-17B-16E (template llama4, 1 GPU)
expected MoE status: all-MoE, llama4_moes_wrapped=48
expected backend:
expected recompute input:
expected artifact recompute label:
expected CPUAdamW/optimizer-offload family:
expected route kernel code: 000
expected moefg: 0 (Phase A) / 1 (Phase B only)
expected dense fine-grained wrapped: 0
expected attention wrapper count: > 0 (target rows)
expected training_bf16: false (target) / true (baselines)
expected liger: applied and logged
expected comparison baseline:
expected likely failure mode (G-OOM location / RSS blowup / wrong path):
```

After each run, inspect:

```text
command.txt
train.log (liger apply lines, setup report line with llama4_moes_wrapped=...)
profile.json.config
profile.json partial/completed fields
source_profile.json step_samples
memory_breakdown_summary.json + live activation details
runtime counters (module stats)
rss_peak
artifact path labels
```

Treat the result as inconclusive if:

```text
profile is partial and does not identify a stage bug
path label and profile config disagree
recomp_label is not recomp-off-full-fg-ker000
UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=false
ASYMM_EXPERT_SILU_BWD_GPU=1
ASYMM_MLP_RECOMPUTE_CHUNK != 0
UNSLOTH_GC_OUTER_HBM_EVERY_N != 0
any Qwen3 MoE routed bit is 1, or any qwen3_moe_routed_* counter is nonzero
asymm_qwen3_moe_finegrained_offload=1 in Phase A, or qwen3_moe_finegrained_* counters fire
expact_lora_a_forward_* counters fire (expert act-offload must stay off under full-fg)
llama4_moes_wrapped != 48, or qwen3/qwen35_moes_wrapped != 0
shared expert not wrapped (check ASYM_OFFLOAD_MODULES=all before blaming model code)
dense_mlp_finegrained_offload_wrapped > 0
attention_act_offload_wrapped == 0 on a target row
reference_fallback_count != 0
training_bf16 missing from the profile config
liger not proven applied on any ligerloss1 row
LoRA config differs across rows (all rows must be r64/a16/d0.00)
wrong CPUAdamW/SuperOffload family was compared
stale artifact was reused (e.g. the old r64/a128 s9500 baseline)
```

Conclusion labels:

```text
validated
blocked_by_stage_bug
inconclusive_wrong_config
inconclusive_partial_profile
inconclusive_stale_artifact
inconclusive_unexpected_path
```

Do not advance to the next stage on an inconclusive result.

## Stage 1: Small Config Gate

Purpose: prove the llama4 `ker000` config is real end-to-end — wrapping counts, no-grad
original forward, liger bridge, CPUAdamW family — before any memory conclusion.

Run each row separately:

```bash
RUNS='llama4-scout|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_llama4_s2048 MAX_STEPS=1 WARMUP_STEPS=0 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='llama4-scout|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_llama4_s2048 MAX_STEPS=1 WARMUP_STEPS=0 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false
```

Pass criteria:

```text
both rows complete a backward step with finite loss
target artifact label is recomp-off-full-fg-ker000, route tag route000_lora0_accfp32
target setup report: llama4_moes_wrapped=48, attention_act_offload_wrapped>0,
  dense_mlp_finegrained_offload_wrapped=0, qwen3_moe_finegrained_offload_wrapped=0
target counters: attn_act_lora_a_forward_calls>0, attn_act_lora_a_grad_calls>0,
  asym_forward_calls>0, asym_dx_calls>0 (CPU-resident base weights actually used),
  expact_lora_a_forward_*=0 (no expert act-offload path),
  qwen3_moe_finegrained_*=0, qwen3_moe_routed_*=0, reference_fallback_count=0
target config: training_bf16=false recorded; grad/weight offload true
liger proven applied on both rows
baseline is superoffload_mem (param+optimizer offload on), not a nocpuadamw variant
no vision-tower modules wrapped or resident on GPU beyond the frozen expectation
```

## Stage 2: Memory-Shape Gate At s8192

Purpose: make sure the composition scales before the real target. s8192 sits just under
the old `recomp` family ceiling (G-OOM 9k) and has prior llama4 artifacts for context.
This is not the final scoreboard.

Run each row separately:

```bash
RUNS='llama4-scout|1 ; superoffload_mem|unsloth|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_llama4_s8192 MAX_STEPS=1 WARMUP_STEPS=0 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='llama4-scout|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_llama4_s8192 MAX_STEPS=1 WARMUP_STEPS=0 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='llama4-scout|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_llama4_s8192 MAX_STEPS=1 WARMUP_STEPS=0 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false
```

Expected memory shape:

```text
superoffload_mem|unsloth keeps recompute intermediates in HBM during layer backward
superoffload_mem|unsloth-off moves recompute saved tensors to CPU
asym_cpuadamwds|recomp-off-full-fg-ker000 should sit at or below unsloth-off on
  saved-activation HBM, with base weights CPU-resident and read in place by AsymGEMM
  kernels instead of being gathered per layer
target RSS must stay within the host budget with clear headroom for s9500
```

If target HBM is higher than `unsloth-off`, run the memory decomposition below before
any conclusion. Identify whether the peak is:

```text
routed-expert route-space live operands (packed X [M,H], fused gate_up [M,2I] from the
  plain expert body at qwen3_moe.py:2312-2313, act [M,I]) during recompute/backward —
  a TRANSIENT [M,2I] here is expected Phase-A behavior; a SAVED [M,2I] activation that
  survives to the peak indicates outer save_on_cpu is not covering the recompute graph
shared-expert live operands ([M,16384] gate/up/act per layer)
attention projections / SDPA / chunked-attention buffers
router tensors (logits/scores; should be tiny at [M,16])
LoRA transient: expert LoRA bank gather under weight offload
liger CE chunk / lm_head
CPUAdamW grad-offload transfer buffers
save_on_cpu staging / allocator reserve
wrong no-grad forward path (internal tensors saved in the original forward)
dtype inflation from training_bf16=false (compare dtype attribution across rows)
```

## Stage 3: Final s9500 Scoreboard

Only run this after Stage 2 is validated.

Run each row separately:

```bash
RUNS='llama4-scout|1 ; superoffload_mem|unsloth|ligerloss1 ; 9500|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_llama4_s9500 MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='llama4-scout|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 9500|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_llama4_s9500 MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='llama4-scout|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 9500|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_llama4_s9500 MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false
```

Success criteria:

```text
target completes with finite loss
target artifact label is recomp-off-full-fg-ker000
target peak HBM is below superoffload_mem|unsloth-off at s9500.b8.ga1
target is also reported against superoffload_mem|unsloth
target CPU RSS is within host budget
setup report and counters match the Phase-A expectations exactly
  (llama4_moes_wrapped=48; MoE-fg, dense-fg, route counters all zero;
   attention counters nonzero; reference_fallback_count=0)
liger proven applied on all three rows
memory snapshot / live activation details identify no hidden wrong-path peak
```

If the target fails but a baseline also fails, report both failures explicitly. Do not
claim success from "baseline OOM" unless the target completed and the config audit is
clean.

## Stage 4 (Optional): Max-Seq Extension

Only after the s9500 scoreboard is validated. The research framing is a strictly longer
real sequence than the best existing-system baseline, so extend upward:

```text
s14500.b8: superoffload_mem|unsloth-off (its known ceiling) vs the target
beyond s14500: target alone, stepping up until G-OOM or RSS budget,
  each step run and audited individually
```

Any "longer than baseline" claim must name which baseline family it beats
(`superoffload_mem|unsloth` tops out ~9500; `superoffload_mem|unsloth-off` ~14500) and
show the corresponding completed/failed artifacts on both sides.

## Optional No-CPUAdamW Cross-Check

Only use this if evaluating `asym|recomp-off-full-fg-ker000` without CPUAdamW:

```bash
RUNS='llama4-scout|1 ; superoffload_mem_nocpuadamw|unsloth|ligerloss1 ; 9500|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_llama4_s9500_nocpuadamw MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='llama4-scout|1 ; superoffload_mem_nocpuadamw|unsloth-off|ligerloss1 ; 9500|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_llama4_s9500_nocpuadamw MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='llama4-scout|1 ; asym|recomp-off-full-fg-ker000|ligerloss1 ; 9500|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_llama4_s9500_nocpuadamw MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false
```

Do not compare `asym|...` against `superoffload_mem|...` as the only scoreboard. That
mixes optimizer-offload families.

## Memory Decomposition If The Target Is Higher Than Baseline

Produce a concise table for every failed or surprising stage:

```text
Workload  Backend           Config                     step_H  RAM   act_H  saved_GPU  saved_CPU  top_peak_owner
--------  ----------------  -------------------------  ------  ----  -----  ---------  ---------  --------------
s8192.b8  superoffload_mem  unsloth-off                ...
s8192.b8  asym_cpuadamwds   recomp-off-full-fg-ker000  ...
```

Then inspect the detailed live tensors and answer:

```text
which module owns the peak?
is it routed experts, shared expert, attention, LoRA, liger/lm_head, optimizer,
  or allocator reserve?
which exact tensor shape dominates ([M,2I]=[76000,16384]? [M,16384] shared?
  [M,H]=[76000,5120]?)
is the tensor a live GEMM operand or a saved activation?
is it covered by outer save_on_cpu, or was a raw tensor stored on ctx outside
  ActivationOffloadManager?
did the no-grad original forward accidentally save internal tensors?
is the gap dtype (training_bf16=false) rather than placement?
is the gap the expert LoRA bank gather (weight offload) rather than activations?
```

Only after this audit should Phase B (or any other implementation change) be proposed.
If the peak owner is attention, liger, optimizer, or dtype, Phase B is the WRONG fix —
address the responsible component instead.

## Phase B: Llama4 MoE Fine-Grained Wiring (Only If Phase A Evidence Demands It)

Entry condition — all of:

```text
Phase A target completes but does not beat superoffload_mem|unsloth-off at s8192/s9500
AND the decomposition attributes the gap to routed-expert recompute saved tensors or
route-space live operands (e.g. the plain expert body's fused gate_up [M,2I] staging)
AND no config/counter audit issue explains the number instead
```

The machinery to reuse is `_Qwen3MoeFinegrainedFunction`
(`asym_gemm/training/qwen3_moe_finegrained.py:344`; entry `:1251`), which already
supports input-scaled routing (`input_weighted=True`) end to end — the same semantic
`AsymLlama4Moe` uses today. NO new autograd Function, kernels, counters, or env
surface are written; the work is wiring plus one layout fix.

Enablement contract (fixed, not open for reinterpretation):

```text
REUSED verbatim (no llama4-renamed duplicates):
  env    ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD (+ ASYM_GEMM_LF_CONFIG_ mirror)
  attr   _qwen3_moe_finegrained_enabled on the experts engine
  counters qwen3_moe_finegrained_* (Function-level, engine-shared)
  report qwen3_moe_finegrained_offload_wrapped

NEW (harness-only bring-up gate):
  ASYMM_LLAMA4_MOE_FINEGRAINED=1   # default 0/absent
  Consumed ONLY by the two profile wrappers to decide whether the full-fg arm sets
  moefg=1 for llama4-scout. Default-off means Phase-A rows stay byte-reproducible
  AFTER the Phase-B code lands. Flipping llama4-scout to an auto-default (like
  q3-30b's ker101 auto-default) is a separate post-B3 change, never bundled.

RESERVED (not implemented until a Phase-B decomposition demands it; item 7):
  ASYMM_LLAMA4_SHARED_MLP_FINEGRAINED_OFFLOAD=1   # default 0
```

The wiring items:

1. Installer flag. In the llama4 wrap branch (`lf.py:1928-1957`), after
   `wrap_llama4_moe(...)` returns, mirror the qwen3_whole block (`lf.py:1924-1926`)
   exactly:

   ```python
   if qwen3_moe_finegrained_enabled and offload_experts:
       wrapped.experts._qwen3_moe_finegrained_enabled = True
       report.qwen3_moe_finegrained_offload_wrapped += 1
   ```

   `wrapped.experts` is the `AsymLlama4Experts` instance (`llama4_moe.py:302` proves
   the attribute). Reuse `qwen3_moe_finegrained_offload_wrapped` — do NOT add a
   llama4-named report field; the counters, validator, and config plumbing already
   flow this family, and a parallel field would fragment the audit surface. Set the
   flag for both `llama4_whole` and `llama4_hf` wrap kinds; hf-mode misuse is caught
   at runtime by the detached-router raise (item 2), and the harness default is
   routerwhole.
2. Dispatch. Add BOTH fine-grained branches to `AsymLlama4Experts.forward_input_scaled`
   (`llama4_experts.py:935-990`), copying the base engine's input-scaled pattern
   (`qwen3_moe.py:2866-2885`). Exact placement, because it is a correctness trap:

   - The branches take the UNPACKED flat `hidden_states` plus route metadata — the
     Function packs internally and applies the router weights itself via
     `input_weighted=True` (`qwen3_moe_finegrained.py:379-385`).
   - The llama4 override currently packs and pre-scales BEFORE building dense groups
     (`llama4_experts.py:954-963`: `packed = packed * route_scale`). Inserting the
     branches after that line would apply the router weights TWICE. Therefore: hoist
     `make_dense_group_metadata` above the branches (base-class order,
     `qwen3_moe.py:2858-2865`) and place the branches BEFORE `pack_tokens_contiguous`;
     the existing pack/pre-scale lines remain only for the non-fg fallback paths.

   ```python
   self.gather_lora_weights()
   try:
       input_dtype = hidden_states.dtype
       metadata = build_contiguous_route_metadata(top_k_index, input_weights, num_experts=self.num_experts)
       offsets, experts = make_dense_group_metadata(          # hoisted above packing
           metadata.expert_offsets, num_groups=self.num_experts, device=hidden_states.device)
       if self._uses_qwen3_moe_finegrained_offload():          # qwen3_moe.py:2478-2483
           return self._forward_qwen3_moe_finegrained_offload(
               hidden_states, offsets, experts,
               metadata.token_indices, metadata.routing_weights,
               input_weighted=True, output_weighted=False,
           ).to(dtype=input_dtype)
       if self._uses_qwen3_moe_finegrained_nograd_forward():   # qwen3_moe.py:2485-2490
           return self._forward_qwen3_moe_finegrained_nograd(
               hidden_states, offsets, experts,
               metadata.token_indices, metadata.routing_weights,
               input_weighted=True, output_weighted=False,
           ).to(dtype=input_dtype)
       packed = pack_tokens_contiguous(hidden_states, metadata)   # fallback paths only
       packed = packed * route_scale...
       ...existing act-offload / gc / recompute / plain-body branches unchanged...
   finally:
       if self._asym_weight_offload_release_after_forward():
           self.release_lora_weights()
   ```

   Both `_forward_qwen3_moe_finegrained_*` methods and both `_uses_*` gates are
   inherited from `AsymQwen3Experts` — no llama4 copies. The grad branch fires only in
   the backward recompute forward (training + grad enabled = the unsloth-off boundary);
   the nograd branch serves the original no-grad Unsloth forward so both forwards use
   the same split base objects.
   Router-detach requirement: the fine-grained entry hard-raises unless routing weights
   are detached (`qwen3_moe.py:2552-2553`). The llama4 "whole" wrap detaches the router
   (`llama4_moe.py:204,280-290`) and `ROUTER_MODES` defaults to `whole`
   (`profile_lora_lf_test_source.sh:139`) — Phase B is therefore routerwhole-only; a
   `llama4_hf`-mode run keeps router grads and must be rejected, not worked around.
3. Layout-aware base split (the one hard blocker), with a COMMITTED release decision.
   `_ensure_qwen3_moe_finegrained_bases` (`qwen3_moe.py:2492-2526`) already propagates
   `weight_layout`, `precision`, `compiled_dims`, and pinning into the split
   `AsymGroupedFrozenLinear` bases (`:2502-2523`); the fix is a layout-conditional
   slice axis. Do NOT transpose or copy the banks into the Qwen3 layout:

   ```python
   fused = self.gate_up_base.host_weight.weight
   if self.gate_up_base.weight_layout == "in_out":      # llama4 [E, H, 2I]
       gate_weight = fused[..., : self.intermediate_dim].contiguous()
       up_weight = fused[..., self.intermediate_dim :].contiguous()
   else:                                                 # qwen3 [E, 2I, H] — unchanged
       gate_weight = fused[:, : self.intermediate_dim, :].contiguous()
       up_weight = fused[:, self.intermediate_dim :, :].contiguous()
   ```

   Host-RSS consequence and the decision (fixed here, not left to Stage B1): for Qwen3
   `out_in`, the gate slice is a contiguous prefix (zero-copy view under `clone=False`)
   and only the up half is copied — qwen3 behavior stays byte-identical and the fused
   bank CANNOT be freed there (the gate base aliases its storage). For llama4 `in_out`,
   BOTH last-dim slices are non-contiguous, so `.contiguous()` copies BOTH halves
   (~2.5 GiB/layer, ~120 GiB pinned across 48 layers). Keeping the fused bank alongside
   would double the gate/up host bytes and does not fit the host budget next to
   CPUAdamW state and unsloth-off saved tensors. Therefore, COMMITTED: on the `in_out`
   branch, after both split bases are built, RELEASE the fused pinned host storage
   (drop `gate_up_base.host_weight`'s big tensor; net-neutral RSS, transient overhead =
   one layer's bank since the split happens per engine). Required follow-through, both
   specified:
   - eligibility: `qwen3_moe_finegrained_unsupported_reasons` checks the fused bank at
     `qwen3_moe_finegrained.py:1458,1490-1493`; amend so that when split bases already
     exist it validates THEM (isinstance / precision=bf16 / pinned) instead of the
     released fused bank. Qwen3 path unaffected (fused never released there).
   - any later path that still needs `gate_up_base` (eval-mode plain body, act-offload,
     expert-gc — all unreachable in these training-profile runs, which never run an
     eval expert forward) must raise a clear
     "gate_up_base host released for llama4 fine-grained mode" error, not compute
     garbage. Fail-loud, covered by a unit test (item 9). The fg forward, backward, and
     nograd paths never touch the fused bank (`qwen3_moe_finegrained.py:387,631,1300`).
4. Eligibility audit. The dispatch is fail-loud: `_check_qwen3_moe_finegrained_supported`
   raises `NotImplementedError` with the joined reasons (`qwen3_moe.py:2533-2537`), so a
   mis-wired Phase B cannot silently fall back — the s2048 gate catches it with the
   reason string in train.log. The verified gate list
   (`qwen3_moe_finegrained.py:1452-1506`): backend `asym` + expert base CPU offload;
   `AsymGroupedFrozenLinear` fused gate/up and down bases; expert recompute policy
   disabled; `lora_dropout=0`; SiLU; CUDA bf16 contiguous packed input; LoRA dtype ==
   packed dtype; hidden/intermediate %64 (5120/8192: pass); all six LoRA banks on
   device, bf16, contiguous; bases `precision=bf16` with pinned hosts; CPU kernels
   present. Scout is expected to pass every numeric requirement (model and banks stay
   bf16 even under `TRAINING_BF16=false`, see the facts section); still run the
   preflight on the real model at Stage B1 and record the reasons list if non-empty.
5. Route kernels stay OFF and the label stays `recomp-off-full-fg-ker000` — in Phase B
   too. See the quantified route-kernel section above: on Scout each avoided `[R,H]`
   is ~0.72 GiB (~1% of peak) vs 12-20 GiB (~35% of peak) on qwen3-30b, the kernels
   would need new SM100 variants for the `in_out` layout, and top-1 removes the
   weighted-accumulate rationale. Only a Phase-B decomposition showing a route-space
   `[R,H]` top peak owner reopens this.
6. Harness symmetry — exact diff, applied identically to BOTH byte-twin wrappers
   (`profile_lora_lf_test_source.sh`, `profile_lora_lf_test_both.sh`):
   - add `is_llama4_scout_model()` matching the literal substring `Llama-4-Scout`
     (same style as `is_qwen3_moe_routed_model`, `:608-613`);
   - extend the full-fg arm (`:3084-3094`) — today `:3089` keys on
     `is_qwen3_moe_routed_model` only — with the explicit opt-in:

     ```bash
     elif is_llama4_scout_model "${current_model_name}" \
         && [[ "$(bool_value "${ASYMM_LLAMA4_MOE_FINEGRAINED:-false}")" == "true" ]]; then
       ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD=1
       # route bits and ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS stay 0
     fi
     ```

   - forward `ASYMM_LLAMA4_MOE_FINEGRAINED` into `RUN_ENV` and as
     `ASYM_GEMM_LF_CONFIG_ASYMM_LLAMA4_MOE_FINEGRAINED` so `profile.json.config`
     records the opt-in;
   - post-run validator: extend the expectation at `:1436-1443` from
     `qwen3_moe_target = "Qwen3-30B-A3B" in model` to
     `expected_moefg = full-fg AND (qwen3_moe_target OR ("Llama-4-Scout" in model AND
     llama4 opt-in env true))`;
   - ker guards UNCHANGED: llama4-scout remains `ker000`-only; the route-bit guard
     (`:3119-3126`) still rejects any route bit for llama4.
   The `RUN_ID` `moefg` label then flips to `moefg1` via the existing `moefg_tag`,
   which is what distinguishes Phase-B from Phase-A artifacts; the recompute label does
   not change.
7. Shared expert stays plain in the first Phase-B run — one module family at a time. If
   a later decomposition shows the shared-expert `[M,16384]` recompute saves own the
   peak, add the reserved opt-in `ASYMM_LLAMA4_SHARED_MLP_FINEGRAINED_OFFLOAD=1`
   (default 0; named in the enablement contract above) that reuses the dense
   fine-grained Function (`build_finegrained_dense_mlp` / `_FinegrainedDenseMLPFunction`
   on the shared MLP's `gate_proj/up_proj/down_proj`, installed from the llama4 wrap
   branch since the generic dense installer is gated off by `expert_prefixes`), and
   reuses the existing `dense_mlp_finegrained_*` counters and report fields — in that
   configuration `dense_mlp_finegrained_offload_wrapped = 48` becomes expected instead
   of 0. Never bundle it into the run that first validates MoE fine-grained.
8. Weight-offload interplay is already engine-correct once the branches exist —
   verified, not assumed: the fine-grained Function re-gathers the LoRA banks in both
   forward (`qwen3_moe_finegrained.py:365`) and backward (`qwen3_moe_finegrained.py:608`),
   and `_asym_weight_offload_release_after_forward` returns true when fine-grained is
   active (`qwen3_moe.py:2464-2476`), so the llama4 override's release-in-finally is the
   right discipline. The q3-30b-a3b full-fg runs with `gradofftrue__weightofftrue` are
   the working precedent. Still cover it in the unit tests (next item) because the
   llama4 override is the only call site that wraps the branch in its own gather/release.
9. Tests before profiles. Mirror `tests/training/test_dense_mlp_finegrained.py` and the
   Qwen3 fine-grained tests with a small fake llama4 packed MoE (`in_out` banks, top-1
   sigmoid input scaling, block-level shared-expert add). Required checks, each one
   pinned to a failure mode identified above:
   - forward parity fine-grained ON vs OFF within bf16 tolerance, including nonuniform
     router scores — this is also the guard against the double-weighting trap in
     item 2 (weights applied once, inside the Function);
   - split-base correctness for `in_out`: gate/up split bases reproduce the fused
     `gate_up_base` matmul exactly on the same inputs (slice-axis proof);
   - LoRA A/B grads match; frozen banks stay grad-free;
   - CPU handles released after backward; the no-grad original forward saves nothing
     and routes through the fg-nograd branch (plain-body counters stay zero);
   - weight offload gather/release still works through the llama4 override's
     try/finally;
   - eligibility preflight returns an empty reasons list for the fake llama4 engine
     and a NON-empty list when a requirement is deliberately broken (e.g. fp32 bank);
   - after the item-3 fused-host release, any path that still needs `gate_up_base`
     raises the explicit "released" error instead of computing garbage, and the
     eligibility gate validates the split bases without touching the released fused
     bank;
   - qwen3 regression: `out_in` split behavior byte-identical (gate half still a
     zero-copy view; no release attempted).
10. Counter gates for Phase B (the Function-level counters are engine-shared, so the
    existing names fire for llama4 unchanged):

```text
qwen3_moe_finegrained_forward_calls > 0
qwen3_moe_finegrained_backward_calls > 0
qwen3_moe_finegrained_gate_base_calls > 0
qwen3_moe_finegrained_up_base_calls > 0
qwen3_moe_finegrained_down_base_calls > 0
qwen3_moe_finegrained_stage_concat_columns_calls == 0
qwen3_moe_routed_* == 0                      # route bits stay 000
qwen3_moe_finegrained_offload_wrapped == llama4_moes_wrapped == 48
dense_mlp_finegrained_offload_wrapped == 0   # unless the separate shared-MLP opt-in ran
```

### Phase B ladder

Phase B re-enters the same gate discipline with its own stages. Do not skip rungs, and
do not reuse Phase-A artifacts as Phase-B evidence (the `moefg` label in the RUN_ID and
path is what separates them: `moefg0` = Phase A, `moefg1` = Phase B).

```text
B0: unit tests (item 9) green; no GPU profiling before this.
B1: s2048 config gate — same commands as Stage 1 with ASYMM_LLAMA4_MOE_FINEGRAINED=1
    exported (the item-6 opt-in; without it the row must still resolve moefg0 —
    verify both). Pass: label recomp-off-full-fg-ker000 with moefg1,
    config.asymm_llama4_moe_finegrained=true recorded,
    qwen3_moe_finegrained_offload_wrapped=48, fine-grained counters fire, route
    counters zero, eligibility preflight silent, and RSS ~net-neutral vs the Phase-A
    s2048 row (item-3 fused-host release working; a ~+120 GiB RSS jump means it is
    not).
B2: s8192 memory-shape gate — three rows as Stage 2. Pass requires Phase B to beat the
    PHASE A artifact at s8192 (that is the whole justification for the wiring) and to
    close on or beat superoffload_mem|unsloth-off.
B3: s9500 scoreboard — same rows and success criteria as Stage 3, with
    expected moefg: 1 and the Phase-B counter gates.
```

If B2 does not beat Phase A, stop and decompose before B3: the likely owners are the
shared expert (item 7 becomes the next lever) or attention/liger/optimizer, none of
which Phase B touches.

## Reporting Format

The final response/table must be plain text and include these metrics only:

```text
fwd_s  bwd_s  opt_s  step_s  fwd_H  bwd_H  step_H  RAM
```

Use the generated artifact labels in the backend/config columns:

```text
asym_cpuadamwds    recomp-off-full-fg-ker000
superoffload_mem   unsloth
superoffload_mem   unsloth-off
```

Do not report a run as final unless the artifact audit proves it is `llama4-scout`
(all-MoE, `llama4_moes_wrapped=48`), `ligerloss1` actually applied, `9500|8|1`,
`recomp-off-full-fg-ker000` with route tag `route000_lora0_accfp32`, identical LoRA
`r64/a16/d0.00` across all rows, and the correct CPUAdamW family for the backend.
