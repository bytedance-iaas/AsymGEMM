# Fine-Grained MoE Offload Validation Log

This log is the companion artifact for `agent/impls/fix_finegrained_moe.md`.
Update it before and after every stage. Do not use undocumented runs as evidence.

## Global Rules

- Run validation jobs serially. Do not run experiments in parallel while debugging this
  path.
- Keep `UNSLOTH_GC_OUTER_HBM_EVERY_N=0` for all target and baseline comparisons.
- Keep the external policy tuple `none|false|false|false|false|false`.
- For the target, old `ASYMM_EXPERT_ACT_OFFLOAD` must be false.
- For the target, the new Qwen3 MoE fine-grained flag/counters must prove the path.
- Compare `asym_cpuadamwds` primarily against `superoffload_mem`, not
  `*_nocpuadamw`.

## Scoreboard Template

```text
Model: qwen3-30b-a3b    LoRA: r64/a16/d0.00    CPUAdamW family: yes
Workload   Backend            Recompute          Config                  fwd_s  bwd_s  opt_s  step_s  fwd_H  bwd_H  step_H    RAM  Status
---------  -----------------  -----------------  ----------------------  -----  -----  -----  ------  -----  -----  ------  -----  --------
s80000.b8  superoffload_mem   unsloth            none + ligerloss1        29.6  130.3    0.0   162.4   91.9  176.9   176.9  359.9  complete
s80000.b8  superoffload_mem   unsloth-off        none + ligerloss1        33.2  240.7    0.0   274.0   91.9   94.4    94.4  588.4  complete
s80000.b8  asym_cpuadamwds    recomp-off-full-fg moefg1 + ligerloss1      65.6  977.6    4.0  1043.3   86.0  112.9   112.9  642.0  complete; misses unsloth-off HBM
```

Known complete `superoffload_mem|unsloth` artifact:

```text
profiling_q3_30b_a3b_s80000/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b8_s80000_ga1_w0_s1_r64_a16_drop000/superoffload_mem__source__unsloth__polnone__routerhf__expact0__attnact0__layeract0__layergc0__sdparecomp0__loraafwdhbm__actrecomp0__xunpack0__ligerloss1/b8_s80000_ga1/source_profile.json
```

Known `superoffload_mem|unsloth` peak decomposition:

```text
saved_activations/routed_experts      69.75 GiB
temporary_workspace/routed_experts    55.78 GiB
saved_activations/norms               20.84 GiB
saved_activations/attention           13.99 GiB
saved_activations/router               2.81 GiB
live_activation/embed_tokens           2.44 GiB
live_activation/router                 0.16 GiB
actual peak allocated                 176.95 GiB
actual peak reserved                  180.69 GiB
```

## Stage 0: Config Truth And Path Ownership

stage: 0

expected run labels:

- `asym_cpuadamwds|recomp-off-full-fg|ligerloss1`

expected config:

- `recomp_off_stage=full-fg`
- `use_unsloth_gc=true`
- `unsloth_gc_recompute_save_on_cpu=true`
- `unsloth_gc_outer_hbm_every_n=0`
- `asymm_qwen3_moe_finegrained_offload=true`
- `asymm_expert_act_offload=false`
- `asymm_attn_act_offload=true`
- CPUAdamW grad/weight offload enabled by `asym_cpuadamwds`

expected path labels:

- includes `moefg1`
- includes `expact0`
- includes `attnact1`
- does not hide MoE-fg state only inside `recomp-off-full-fg`

expected wrappers/counters:

- setup report has `qwen3_moe_finegrained_offload_enabled=true`
- setup report has `qwen3_moe_finegrained_offload_wrapped > 0`
- old expact counters are zero until a diagnostic run explicitly enables old expact
- dense fine-grained counters are zero for MoE

observed:

- implemented `ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD` env/config/profile plumbing.
- implemented `moefg1/moefg0` artifact/run-id tags in both source and both profile scripts.
- implemented LF setup-report fields:
  `qwen3_moe_finegrained_offload_enabled` and
  `qwen3_moe_finegrained_offload_wrapped`.
- implemented MoE dispatch only on `AsymQwen3Experts` with routed-expert Asym offload;
  dense full-fg dispatch remains gated by `not expert_prefixes`.
- implemented a new `qwen3_moe_finegrained.py` custom Function separate from old
  `_ActivationOffloadQwen3ExpertFunction`.
- new path is gated by `torch.is_grad_enabled()`, so original checkpoint no-grad
  forward uses the normal expert body.

conclusion:

- Stage 0 code/config ownership is implemented. Runtime artifact truth still needs
  Stage 4 smoke profiles.

## Stage 1: s80000 Baselines

stage: 1

run labels:

- `superoffload_mem|unsloth|ligerloss1`
- `superoffload_mem|unsloth-off|ligerloss1`

commands:

```bash
RUNS='q3-30b-a3b|1 ; superoffload_mem|unsloth|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false' \
  scripts/lf/profile_lora_lf_test_source.sh --gpus 3 --overwrite true --plot false --max-steps 1 --warmup-steps 0 --output-root profiling_fix_fgm

RUNS='q3-30b-a3b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false' \
  scripts/lf/profile_lora_lf_test_source.sh --gpus 3 --overwrite true --plot false --max-steps 1 --warmup-steps 0 --output-root profiling_fix_fgm
```

expected:

- both run serially;
- both use `superoffload_mem`;
- both use `ligerloss1`;
- both use the same model, LoRA, batch, sequence, and grad accumulation;
- unsloth-off has `UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=true`;
- no Asym counters fire.

observed:

- `superoffload_mem|unsloth`: complete; metrics recorded in scoreboard.
- `superoffload_mem|unsloth-off`: complete:

```text
artifact:
profiling_fix_fgm/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b8_s80000_ga1_w0_s1_r64_a16_drop000/superoffload_mem__source__unsloth-off__polnone__routerhf__expact0__attnact0__layeract0__layergc0__sdparecomp0__loraafwdhbm__actrecomp0__xunpack0__moefg0__ligerloss1/b8_s80000_ga1

config:
backend=superoffload_mem
use_unsloth_gc=true
unsloth_gc_recompute_save_on_cpu=true
unsloth_gc_outer_hbm_every_n=0
asymm_qwen3_moe_finegrained_offload=0
asymm_expert_act_offload=false
asymm_attn_act_offload=false

metrics:
fwd_s=33.2  bwd_s=240.7  opt_s=0.0  step_s=274.0
fwd_H=91.9  bwd_H=94.4   step_H=94.4  RAM=588.4

peak:
temporary_workspace/routed_experts=59414 MiB
temporary_workspace/norms=17572 MiB
saved_activations/routed_experts=7500 MiB
temporary_workspace/attention=6706 MiB
reserved_unallocated=4180 MiB
temporary_workspace/embed_tokens=2768 MiB
saved_activations/embed_tokens=2500 MiB
saved_activations/router=205 MiB
```

conclusion:

- Stage 1 baseline matrix is complete. `superoffload_mem|unsloth-off` is much lower
  HBM than `superoffload_mem|unsloth` at s80000, so the final target must be measured
  directly before any success claim.

## Stage 2: Current Asym MoE Control

stage: 2

run labels:

- `asym_cpuadamwds|recomp-off-base|ligerloss1`
- `asym_cpuadamwds|recomp-off-attn|ligerloss1`
- `asym_cpuadamwds|recomp-off-full-fg|ligerloss1`

workload:

```text
s8192.b8 first; larger only after config truth passes
```

expected:

- base/attn prove outer Unsloth GC and CPUAdamW offload;
- current `full-fg` is not accepted as MoE-fg unless new MoE counters fire;
- old expact path stays off;
- dense fine-grained counters stay zero for MoE.

observed:

- `python -m py_compile` passed for:
  - `asym_gemm/training/qwen3_moe_finegrained.py`
  - `asym_gemm/training/qwen3_moe.py`
  - `asym_gemm/integrations/lf.py`
  - `asym_gemm/training/frozen_linear.py`
  - `scripts/lf/run_lf_profiled_train.py`
- `bash -n` passed for:
  - `scripts/lf/run_lf_lora_sft.sh`
  - `scripts/lf/profile_lora_lf_test_source.sh`
  - `scripts/lf/profile_lora_lf_test_both.sh`
- `python -m pytest tests/training/test_lf_qwen3_asym_backend.py -q -k
  'qwen3_moe_finegrained or marks_qwen3_moe_finegrained or moe_finegrained'`
  result: `2 passed, 1 skipped`.
- skipped test is the SM100 parity test because this Python environment reports the
  CUDA extension unavailable.
- `python -m pytest tests/training/test_dense_mlp_finegrained.py -q` result:
  `4 passed`.
- shell syntax re-check after adding `moefg` to run ids:
  `bash -n scripts/lf/run_lf_lora_sft.sh scripts/lf/profile_lora_lf_test_source.sh
  scripts/lf/profile_lora_lf_test_both.sh` passed.

conclusion:

- Stage 2 syntax/control tests pass and dense fine-grained tests still pass.
- Full SM100 parity remains to be exercised by LF/profile environment or a Python
  environment with the CUDA extension loaded.

## Stage 3: New MoE Fine-Grained Unit/Parity Tests

stage: 3

test cases:

- balanced static routes,
- skewed static routes,
- repeated experts,
- empty expert groups,
- learned router if the new wrapper owns routing,
- bf16 CUDA,
- LoRA r64/a16/drop0.00,
- CPUAdamW weight gather/release if touched.

expected:

- outputs finite;
- gradients finite;
- LoRA gradients close to existing Qwen3 MoE reference;
- no fused `gate_up [R,2I]` saved for backward;
- no fused `grad_gate_up [R,2I]`;
- `stage_concat_columns` count is zero.

observed:

- Added focused Python tests for LF flag ownership and grad-enabled dispatch.
- `python -m pytest tests/training/test_lf_qwen3_asym_backend.py -q -k
  'qwen3_moe_finegrained or marks_qwen3_moe_finegrained or moe_finegrained'`
  result: `2 passed, 1 skipped, 134 deselected`.
- The skipped case is the SM100 CUDA parity test because this Python environment
  reports the CUDA extension unavailable. LF smoke is therefore the runtime proof for
  the compiled path.

conclusion:

- Stage 3 Python ownership/dispatch tests pass. CUDA parity is not available from this
  Python environment and must be covered by LF runtime counters.

## Stage 4: Small LF Smoke

stage: 4

run labels:

- `asym_cpuadamwds|recomp-off-full-fg|ligerloss1`

workloads:

```text
s2048.b8
s8192.b8
```

expected:

- complete one measured step;
- new MoE fine-grained wrapper count > 0;
- `qwen3_moe_finegrained_forward_calls > 0`;
- `qwen3_moe_finegrained_backward_calls > 0`;
- `qwen3_moe_finegrained_stage_concat_columns_calls == 0`;
- old expact counters == 0;
- dense fine-grained counters == 0.

observed:

- `s2048.b8` complete:

```text
artifact:
profiling_fix_fgm/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b8_s2048_ga1_w0_s1_r64_a16_drop000/asym_cpuadamwds__source__recomp-off-full-fg__polnone__routerwhole__expact0__attnact1__layeract0__layergc0__sdparecomp0__loraafwdcpu__actrecomp0__xunpack0__moefg1__ligerloss1__gradofftrue__weightofftrue/b8_s2048_ga1

config:
backend=asym_cpuadamwds
recomp_off_stage=full-fg
use_unsloth_gc=true
unsloth_gc_recompute_save_on_cpu=true
unsloth_gc_outer_hbm_every_n=0
asymm_qwen3_moe_finegrained_offload=1
asymm_expert_act_offload=false
asymm_attn_act_offload=true
asym_cpu_adamw_grad_offload=True
asym_cpu_adamw_weight_offload=True

metrics:
fwd_s=56.7  bwd_s=84.3  opt_s=3.6  step_s=141.0
fwd_H=3.2   bwd_H=3.8   step_H=3.8  RAM=258.5

counters:
qwen3_moe_finegrained_forward_calls=48
qwen3_moe_finegrained_backward_calls=48
qwen3_moe_finegrained_gate_base_calls=96
qwen3_moe_finegrained_up_base_calls=96
qwen3_moe_finegrained_down_base_calls=96
qwen3_moe_finegrained_stage_concat_columns_calls=0
qwen3_moe_finegrained_gpu_silu_bwd_calls=48
qwen3_moe_finegrained_lora_a_forward_calls=144
qwen3_moe_finegrained_lora_a_grad_calls=144
qwen3_moe_finegrained_lora_b_backward_calls=144
qwen3_moe_finegrained_fused_gate_up_hbm_bytes=0
qwen3_moe_finegrained_saved_cpu_bytes=1375731712
qwen3_moe_finegrained_stage_hbm_peak_bytes=402653184
dense_mlp_finegrained_forward_calls=0
dense_mlp_finegrained_backward_calls=0
old expact lora_a/lora_b counters=0
reference_fallback_count=0

peak:
actual_peak_phase=after_backward
actual_peak_allocated_hbm=3.75 GiB
actual_peak_reserved_hbm=12.58 GiB
reserved_unallocated=8.83 GiB
saved_activations/routed_experts=512 MiB
temporary_workspace/routed_experts=2423 MiB
saved_activations/norms=64 MiB
saved_activations/embed_tokens=64 MiB
```
- `s8192.b8` complete:

```text
artifact:
profiling_fix_fgm/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b8_s8192_ga1_w0_s1_r64_a16_drop000/asym_cpuadamwds__source__recomp-off-full-fg__polnone__routerwhole__expact0__attnact1__layeract0__layergc0__sdparecomp0__loraafwdcpu__actrecomp0__xunpack0__moefg1__ligerloss1__gradofftrue__weightofftrue/b8_s8192_ga1

metrics:
fwd_s=28.0  bwd_s=77.0  opt_s=3.8  step_s=105.1
fwd_H=12.2  bwd_H=13.9  step_H=13.9  RAM=271.0

counters:
qwen3_moe_finegrained_forward_calls=48
qwen3_moe_finegrained_backward_calls=48
qwen3_moe_finegrained_gate_base_calls=96
qwen3_moe_finegrained_up_base_calls=96
qwen3_moe_finegrained_down_base_calls=96
qwen3_moe_finegrained_stage_concat_columns_calls=0
qwen3_moe_finegrained_fused_gate_up_hbm_bytes=0
qwen3_moe_finegrained_saved_cpu_bytes=5502926848
qwen3_moe_finegrained_stage_hbm_peak_bytes=1610612736
dense_mlp_finegrained_forward_calls=0
dense_mlp_finegrained_backward_calls=0
old expact lora_a/lora_b counters=0
reference_fallback_count=0

peak:
temporary_workspace/routed_experts=8686 MiB
saved_activations/routed_experts=2048 MiB
temporary_workspace/norms=1927 MiB
temporary_workspace/attention=841 MiB
saved_activations/norms=256 MiB
saved_activations/embed_tokens=256 MiB
reserved_unallocated=3123 MiB
```

conclusion:

- `s2048.b8` and `s8192.b8` prove the new MoE-fg path dispatches in LF, avoids
  `stage_concat_columns`, avoids old expact counters, and leaves dense-fg counters at
  zero. `s8192.b8` already shows routed-expert temporary workspace as the top HBM
  owner; Stage 5 must verify that scaling at meaningful sequence lengths before any
  final memory claim.

## Stage 5: Memory-Shape Gate

stage: 5

run labels:

- `asym_cpuadamwds|recomp-off-full-fg|ligerloss1`

workloads:

```text
s30000.b8
s45000.b8
```

expected:

- routed-expert saved activations and temporary workspace move in the expected
  direction versus baselines;
- if peak is still live gate/up/act overlap, memory details identify exact shapes and
  owners before any sequential-staging change;
- no conclusion from peak number alone.

observed:

- `s30000.b8` target complete:

```text
artifact:
profiling_fix_fgm/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b8_s30000_ga1_w0_s1_r64_a16_drop000/asym_cpuadamwds__source__recomp-off-full-fg__polnone__routerwhole__expact0__attnact1__layeract0__layergc0__sdparecomp0__loraafwdcpu__actrecomp0__xunpack0__moefg1__ligerloss1__gradofftrue__weightofftrue/b8_s30000_ga1

metrics:
fwd_s=78.3  bwd_s=239.1  opt_s=4.4  step_s=317.5
fwd_H=44.3  bwd_H=49.9    step_H=49.9  RAM=351.7

counters:
qwen3_moe_finegrained_forward_calls=48
qwen3_moe_finegrained_backward_calls=48
qwen3_moe_finegrained_stage_concat_columns_calls=0
qwen3_moe_finegrained_fused_gate_up_hbm_bytes=0
qwen3_moe_finegrained_saved_cpu_bytes=20152320000
qwen3_moe_finegrained_stage_hbm_peak_bytes=5898240000
dense_mlp_finegrained_forward_calls=0
dense_mlp_finegrained_backward_calls=0
old expact lora_a/lora_b counters=0
reference_fallback_count=0

peak:
temporary_workspace/routed_experts=30927 MiB
saved_activations/routed_experts=7500 MiB
temporary_workspace/norms=6953 MiB
temporary_workspace/attention=3062 MiB
saved_activations/norms=938 MiB
saved_activations/embed_tokens=938 MiB
reserved_unallocated=11144 MiB
```
- `s45000.b8` target complete:

```text
artifact:
profiling_fix_fgm/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b8_s45000_ga1_w0_s1_r64_a16_drop000/asym_cpuadamwds__source__recomp-off-full-fg__polnone__routerwhole__expact0__attnact1__layeract0__layergc0__sdparecomp0__loraafwdcpu__actrecomp0__xunpack0__moefg1__ligerloss1__gradofftrue__weightofftrue/b8_s45000_ga1

metrics:
fwd_s=78.9  bwd_s=528.9  opt_s=4.4  step_s=607.9
fwd_H=66.3  bwd_H=74.6    step_H=74.6  RAM=459.8

counters:
qwen3_moe_finegrained_forward_calls=48
qwen3_moe_finegrained_backward_calls=48
qwen3_moe_finegrained_stage_concat_columns_calls=0
qwen3_moe_finegrained_fused_gate_up_hbm_bytes=0
qwen3_moe_finegrained_saved_cpu_bytes=30228480000
qwen3_moe_finegrained_stage_hbm_peak_bytes=8847360000
dense_mlp_finegrained_forward_calls=0
dense_mlp_finegrained_backward_calls=0
old expact lora_a/lora_b counters=0
reference_fallback_count=0

peak:
temporary_workspace/routed_experts=46222 MiB
saved_activations/routed_experts=11250 MiB
temporary_workspace/norms=10411 MiB
temporary_workspace/attention=4590 MiB
saved_activations/norms=1406 MiB
saved_activations/embed_tokens=1406 MiB
reserved_unallocated=16973 MiB
```

conclusion:

- `s30000.b8` and `s45000.b8` pass config/counter gates. The peak scales predictably
  and is still dominated by routed-expert temporary workspace plus routed-expert saved
  activations, not fused gate/up tensors. Do not add sequential gate/up staging based
  on current evidence. Proceed to the missing s80000 `superoffload_mem|unsloth-off`
  baseline and the final s80000 target.

current-code v6 update:

- Later lifetime fixes changed the target path after the first s80000 run:
  - route pack/scatter moved fully inside the custom Function,
  - `grad_2d`, `d_s_down`, and `down_lora_dx` are explicitly released before later
    backward phases,
  - the original no-grad checkpoint forward uses the low-memory MoE-fg no-grad path.
- `profiling_fix_fgm_v6` is the current-code evidence root for these fixes.
- `s8192.b8` current-code target:

```text
artifact:
profiling_fix_fgm_v6/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b8_s8192_ga1_w0_s1_r64_a16_drop000/asym_cpuadamwds__source__recomp-off-full-fg__polnone__routerwhole__expact0__attnact1__layeract0__layergc0__sdparecomp0__loraafwdcpu__actrecomp0__xunpack0__moefg1__ligerloss1__gradofftrue__weightofftrue/b8_s8192_ga1

metrics:
fwd_s=7.8  bwd_s=45.0  opt_s=4.0  step_s=52.9
fwd_H=9.0  bwd_H=11.9   step_H=11.9  RAM=270.7

counters:
qwen3_moe_finegrained_nograd_forward_calls=48
qwen3_moe_finegrained_forward_calls=48
qwen3_moe_finegrained_backward_calls=48
qwen3_moe_finegrained_stage_concat_columns_calls=0
qwen3_moe_finegrained_fused_gate_up_hbm_bytes=0
dense_mlp_finegrained_forward_calls=0
old expact lora_a/lora_b counters=0
reference_fallback_count=0

actual peak:
phase=after_backward
allocated_H=11.9 GiB
temporary_workspace_H=9.4 GiB
live_activation_H=2.5 GiB
live routed tensor=model.layers.46.mlp.experts.down_base [524288,2048] = 2.0 GiB
```

- `s30000.b8` current-code target:

```text
artifact:
profiling_fix_fgm_v6/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b8_s30000_ga1_w0_s1_r64_a16_drop000/asym_cpuadamwds__source__recomp-off-full-fg__polnone__routerwhole__expact0__attnact1__layeract0__layergc0__sdparecomp0__loraafwdcpu__actrecomp0__xunpack0__moefg1__ligerloss1__gradofftrue__weightofftrue/b8_s30000_ga1

metrics:
fwd_s=21.6  bwd_s=166.3  opt_s=4.7  step_s=188.1
fwd_H=32.4  bwd_H=42.6   step_H=42.6  RAM=352.1

counters:
qwen3_moe_finegrained_nograd_forward_calls=48
qwen3_moe_finegrained_forward_calls=48
qwen3_moe_finegrained_backward_calls=48
qwen3_moe_finegrained_stage_concat_columns_calls=0
qwen3_moe_finegrained_fused_gate_up_hbm_bytes=0
qwen3_moe_finegrained_saved_cpu_bytes=20152320000
qwen3_moe_finegrained_stage_hbm_peak_bytes=5898240000
dense_mlp_finegrained_forward_calls=0
old expact lora_a/lora_b counters=0
reference_fallback_count=0

actual peak:
phase=after_backward
allocated_H=42.6 GiB
temporary_workspace_H=33.4 GiB
live_activation_H=9.2 GiB
live routed tensor=model.layers.46.mlp.experts.down_base [1920000,2048] = 7.3 GiB
```

- The current-code blocker is now sharper than the original Stage 5 result: the no-grad
  forward peak was reduced, but the grouped `down_base` still materializes one routed
  `[R,H]` output before scatter. At s80000 this tensor alone scales to
  `[5120000,2048]` = 20.0 GiB decimal, while `superoffload_mem|unsloth-off` has a live
  routed `[R,I]` act tensor of 7.3 GiB at its peak.

## Stage 6: Final s80000 Target

stage: 6

run label:

- `asym_cpuadamwds|recomp-off-full-fg|ligerloss1`

command:

```bash
RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false' \
  scripts/lf/profile_lora_lf_test_source.sh --gpus 3 --overwrite true --plot false --max-steps 1 --warmup-steps 0 --output-root profiling_fix_fgm_v6
```

expected:

- complete or produce an audited capacity/failure point;
- new MoE-fg counters prove dispatch;
- lower `step_H` than `superoffload_mem|unsloth-off`, or a precise audited reason why
  not;
- meaningful routed-expert activation/workspace reduction versus
  `superoffload_mem|unsloth`;
- final table includes `fwd_s`, `bwd_s`, `opt_s`, `step_s`, `fwd_H`, `bwd_H`,
  `step_H`, `RAM`.
- because the current-code s30000 live peak is still a route-expanded `down_base`
  `[R,H]` output, the expected s80000 result should improve versus the old target
  artifact but may still miss `superoffload_mem|unsloth-off` unless allocator behavior
  or temporary workspace scales better than linearly. If it misses, the evidence must
  identify whether the gap is the `[R,H]` down output, other routed temporary workspace,
  or stale/wrong config.

observed:

- pre-v6 target run completed one measured step and wrote complete source profile artifacts:

```text
artifact:
profiling_fix_fgm/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b8_s80000_ga1_w0_s1_r64_a16_drop000/asym_cpuadamwds__source__recomp-off-full-fg__polnone__routerwhole__expact0__attnact1__layeract0__layergc0__sdparecomp0__loraafwdcpu__actrecomp0__xunpack0__moefg1__ligerloss1__gradofftrue__weightofftrue/b8_s80000_ga1

metrics:
fwd_s=106.9  bwd_s=1046.8  opt_s=4.3  step_s=1153.8
fwd_H=117.7  bwd_H=132.4    step_H=132.4  RAM=635.2

config/path checks:
backend=asym_cpuadamwds
recomp_off_stage=full-fg
use_unsloth_gc=true
unsloth_gc_recompute_save_on_cpu=true
unsloth_gc_outer_hbm_every_n=0
asymm_qwen3_moe_finegrained_offload=1
asymm_expert_act_offload=false
asymm_attn_act_offload=true
asym_cpu_adamw_grad_offload=True
asym_cpu_adamw_weight_offload=True

counters:
qwen3_moe_finegrained_forward_calls=48
qwen3_moe_finegrained_backward_calls=48
qwen3_moe_finegrained_stage_concat_columns_calls=0
qwen3_moe_finegrained_fused_gate_up_hbm_bytes=0
qwen3_moe_finegrained_saved_cpu_bytes=53739520000
qwen3_moe_finegrained_stage_hbm_peak_bytes=15728640000
dense_mlp_finegrained_forward_calls=0
dense_mlp_finegrained_backward_calls=0
old expact lora_a/lora_b counters=0
reference_fallback_count=0

peak:
temporary_workspace/routed_experts=81914 MiB
reserved_unallocated=30138 MiB
saved_activations/routed_experts=20000 MiB
temporary_workspace/norms=18478 MiB
temporary_workspace/attention=8156 MiB
saved_activations/norms=2500 MiB
saved_activations/embed_tokens=2500 MiB
temporary_workspace/embed_tokens=2039 MiB
```

comparison against s80000 baselines:

```text
Workload   Backend            Recompute          routed temp  routed saved  norms temp  attn temp  step_H
---------  -----------------  -----------------  -----------  ------------  ----------  ---------  ------
s80000.b8  superoffload_mem   unsloth               55.8 GiB      69.8 GiB          -          -   176.9
s80000.b8  superoffload_mem   unsloth-off           58.0 GiB       7.3 GiB    17.2 GiB    6.5 GiB   94.4
s80000.b8  asym_cpuadamwds    recomp-off-full-fg    80.0 GiB      19.5 GiB    18.0 GiB    8.0 GiB  132.4
```

notes:

- The training step completed, but the wrapper returned a post-artifact Bash syntax
  error after writing `source_profile.json`. Current `bash -n` on the wrapper passes,
  so the completed source-profile artifacts are still usable as evidence. Watch this
  wrapper on the next run, but do not discard the s80000 metrics.
- Forward alone already reached 117.7 GiB, above the 94.4 GiB unsloth-off step peak.
- Backward spent a long time in CPU-side autograd work after forward completed:
  `bwd_s=1046.8`.

conclusion:

- Stage 6 is complete, but the target does not achieve the memory goal against
  `superoffload_mem|unsloth-off`: 132.4 GiB versus 94.4 GiB step peak.
- The implementation does beat `superoffload_mem|unsloth` on HBM, 132.4 GiB versus
  176.9 GiB, because it removes most of the unsloth saved routed-expert activations.
- The blocker versus `unsloth-off` is not fused gate/up materialization: the new path
  reports `stage_concat_columns=0` and `fused_gate_up_hbm_bytes=0`. The blocker is the
  routed-expert live workspace plus saved routed activations under the MoE-fg path:
  about 80.0 GiB routed temporary workspace and 19.5 GiB routed saved activations at
  the actual peak.
- Do not claim `recomp-off-full-fg` is successful for Qwen3 MoE at s80000 yet. The
  next design loop must reduce routed-expert workspace/saved activations or explain
  why `superoffload_mem|unsloth-off` has a smaller routed workspace for the same
  workload.

current-code v6 observed:

- target rerun completed one measured step and wrote complete source profile artifacts:

```text
artifact:
profiling_fix_fgm_v6/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b8_s80000_ga1_w0_s1_r64_a16_drop000/asym_cpuadamwds__source__recomp-off-full-fg__polnone__routerwhole__expact0__attnact1__layeract0__layergc0__sdparecomp0__loraafwdcpu__actrecomp0__xunpack0__moefg1__ligerloss1__gradofftrue__weightofftrue/b8_s80000_ga1

metrics:
fwd_s=65.6  bwd_s=977.6  opt_s=4.0  step_s=1043.3
fwd_H=86.0  bwd_H=112.9  step_H=112.9  RAM=642.0

config/path checks:
backend=asym_cpuadamwds
recomp_off_stage=full-fg
use_unsloth_gc=true
unsloth_gc_recompute_save_on_cpu=true
unsloth_gc_outer_hbm_every_n=0
asymm_qwen3_moe_finegrained_offload=1
asymm_expert_act_offload=false
asymm_attn_act_offload=true
asym_cpu_adamw_grad_offload=True
asym_cpu_adamw_weight_offload=True

counters:
qwen3_moe_finegrained_nograd_forward_calls=48
qwen3_moe_finegrained_forward_calls=48
qwen3_moe_finegrained_backward_calls=48
qwen3_moe_finegrained_gate_base_calls=144
qwen3_moe_finegrained_up_base_calls=144
qwen3_moe_finegrained_down_base_calls=144
qwen3_moe_finegrained_stage_concat_columns_calls=0
qwen3_moe_finegrained_fused_gate_up_hbm_bytes=0
qwen3_moe_finegrained_saved_cpu_bytes=53739520000
qwen3_moe_finegrained_stage_hbm_peak_bytes=15728640000
dense_mlp_finegrained_forward_calls=0
dense_mlp_finegrained_backward_calls=0
old expact lora_a/lora_b counters=0
reference_fallback_count=0

actual peak:
phase=after_backward
allocated_H=112.9 GiB
reserved_H=137.3 GiB
temporary_workspace_H=88.5 GiB
live_activation_H=24.4 GiB
reserved_unallocated=24.4 GiB
routed temporary_workspace=61.6 GiB
routed live/saved=19.5 GiB
live routed tensor=model.layers.46.mlp.experts.down_base [5120000,2048] = 19.5 GiB
```

final s80000 table:

```text
Model: qwen3-30b-a3b    LoRA: r64/a16/d0.00
Workload   Backend            Recompute          Config               fwd_s  bwd_s  opt_s  step_s  fwd_H  bwd_H  step_H    RAM
---------  -----------------  -----------------  -------------------  -----  -----  -----  ------  -----  -----  ------  -----
s80000.b8  superoffload_mem   unsloth            none + ligerloss1     29.6  130.3    0.1   160.0   91.9  176.9   176.9  360.0
s80000.b8  superoffload_mem   unsloth-off        none + ligerloss1     33.2  240.7    0.1   274.0   91.9   94.4    94.4  588.5
s80000.b8  asym_cpuadamwds    recomp-off-full-fg moefg1 + ligerloss1   65.6  977.6    4.0  1043.3   86.0  112.9   112.9  642.0
```

final memory decomposition at actual peak:

```text
Workload   Backend            Recompute          routed_tmp  routed_live/saved  norms_tmp  attn_tmp  live_total  temp_total  reserved
---------  -----------------  -----------------  ----------  -----------------  ---------  --------  ----------  ----------  --------
s80000.b8  superoffload_mem   unsloth                  55.8               69.7        7.3       2.8         2.6        66.9       3.7
s80000.b8  superoffload_mem   unsloth-off              58.0                7.3       17.2       6.5        10.0        84.4       4.1
s80000.b8  asym_cpuadamwds    recomp-off-full-fg        61.6              19.5       17.3       7.7        24.4        88.5      24.4
```

current-code conclusion:

- The current implementation beats `superoffload_mem|unsloth` on HBM:
  `112.9 GiB` versus `176.9 GiB`.
- The current implementation does not beat `superoffload_mem|unsloth-off`:
  `112.9 GiB` versus `94.4 GiB`.
- The first implementation was improved substantially by the v6 lifetime fixes:
  `132.4 GiB -> 112.9 GiB` and `117.7 GiB forward peak -> 86.0 GiB forward peak`.
- The remaining blocker is not gate/up fusion: `stage_concat_columns=0` and
  `fused_gate_up_hbm_bytes=0`.
- The audited blocker is the grouped Asym down projection materializing a full routed
  `[R,H]` output before scatter. At s80000 this is
  `model.layers.46.mlp.experts.down_base [5120000,2048] = 19.5 GiB`, whereas the
  `superoffload_mem|unsloth-off` peak's live routed tensor is
  `[5120000,768] = 7.3 GiB`.
- The current Python/AsymGroupedFrozenLinear API always returns the full `[R,N]` output.
  Without route chunking, eliminating this tensor requires a new fused grouped
  down-projection + route-weight + scatter/index-add output path in the kernel/API.
  More gate/up staging cannot solve this final gap.
