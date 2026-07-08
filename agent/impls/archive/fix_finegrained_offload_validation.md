# Fine-Grained Offload Validation Log

## Stage 0-1 Small Gate: s2048 Config Truth

stage: 0-1
run labels:
- `asym|recomp-off-base|ligerloss1`
- `asym|recomp-off-attn|ligerloss1`
- `asym|recomp-off-dense|ligerloss1`
- `asym|recomp-off-full|ligerloss1`
- `asym|recomp-off-dense-fg|ligerloss1`
- `asym|recomp-off-full-fg|ligerloss1`
- `superoffload_mem_nocpuadamw|unsloth-off|ligerloss1`
- `zero3_offload_mem_nocpuadamw|unsloth-off|ligerloss1`

backend family: no CPUAdamW comparison family.

CPUAdamW/optimizer-offload family:
- `asym`: no CPUAdamW, no Asym CPU optimizer.
- `*_nocpuadamw`: ZeRO param offload on CPU, optimizer offload absent/disabled.

expected wrappers:
- `recomp-off-base`: no attention activation wrapper and no dense/current wrapper.
- `recomp-off-attn`: attention activation wrapper only.
- `recomp-off-dense`: current dense E=1 surgical wrapper only.
- `recomp-off-full`: attention plus current dense E=1 surgical wrapper.
- `*-fg`: fine-grained dense flag set, current dense surgical wrapper off; actual model dispatch may still be absent until Stage 6.

expected zero counters:
- base: attention and dense activation counters zero.
- attn: dense activation counters zero.
- dense: attention counters zero.
- nocpuadamw baselines: Asym counters absent/zero.

expected nonzero counters:
- attn/full should have attention counters only if LoRA target/call path reaches the attention wrapper.
- dense/full should have current dense CPU-left LoRA-A counters if the dense wrapper dispatches.

expected peak owner:
- s2048 is primarily a config/path validation, not a memory conclusion.
- Any partial/OOM must still prove exact config and failure point.

expected comparison baseline:
- `asym|recomp-off-*` compares only to `superoffload_mem_nocpuadamw` and `zero3_offload_mem_nocpuadamw`.

expected failure mode:
- wrong resolved flag, stale artifact rejected, missing DeepSpeed label/config, or missing Stage 6 fine-grained implementation for `*-fg`.

artifact root:
`profiling_fix_fgo/asym_long_sft_smoke__lora__lf__bf16/qwen3-32b__gpus1__b8_s2048_ga1_w1_s3_r64_a16_drop000`

audit:
- all eight expected `profile.json`, `source_profile.json`, `memory_breakdown_summary.json`, `summary.md`, `command.txt`, and `train.log` artifacts exist.
- every artifact is complete, not partial.
- every `asym|recomp-off-*` artifact has `config.use_unsloth_gc=true`, `config.unsloth_gc_recompute_save_on_cpu=true`, `config.activation_recompute=true`, `config.asymm_expert_silu_bwd_gpu=0`, `config.asym_offload_act_recompute=0`, `config.asym_offload_x_unpacked=0`, `config.asymm_mlp_recompute_chunk=0`, and the expected `config.recomp_off_stage`.
- `superoffload_mem_nocpuadamw|unsloth-off` and `zero3_offload_mem_nocpuadamw|unsloth-off` both prove `offload_param.device=cpu`, no optimizer-offload device, and no `super_offload` optimizer path.
- setup wrapper counts match the stage labels:
  - base: `dense_mlp_act_offload_wrapped=0`, `attention_act_offload_wrapped=0`, `attention_saved_tensor_offload_wrapped=0`.
  - attn: `dense_mlp_act_offload_wrapped=0`, `attention_act_offload_wrapped=256`, `attention_saved_tensor_offload_wrapped=64`.
  - dense: `dense_mlp_act_offload_wrapped=64`, `attention_act_offload_wrapped=0`, `attention_saved_tensor_offload_wrapped=0`.
  - full: `dense_mlp_act_offload_wrapped=64`, `attention_act_offload_wrapped=256`, `attention_saved_tensor_offload_wrapped=64`.
  - dense-fg/full-fg: `ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD=1` is recorded, but `dense_mlp_act_offload_wrapped=0` and no fine-grained dense counters fire. This is plumbing only, not an implemented Stage-6 path.
- runtime counters match the stage labels:
  - base: `attn_act_lora_a_forward_calls=0`, `expact_lora_a_forward_cpu_left_grouped_calls=0`, `reference_fallback_count=0`.
  - attn: `attn_act_lora_a_forward_calls=1024`, `attn_act_lora_a_grad_calls=1024`, dense current counters zero.
  - dense: `expact_lora_a_forward_cpu_left_grouped_calls=768`, attention counters zero.
  - full: attention counters and dense current counters both nonzero.
  - dense-fg/full-fg: no dense fine-grained counters; treat as missing Stage-6 implementation.
- s2048 peak allocated HBM:
  - base: 8.58 GiB, reserved 9.12 GiB.
  - attn: 8.59 GiB, reserved 9.09 GiB.
  - dense: 9.68 GiB, reserved 10.13 GiB.
  - full: 9.68 GiB, reserved 10.22 GiB.
  - dense-fg: 8.58 GiB, reserved 9.12 GiB.
  - full-fg: 8.59 GiB, reserved 9.09 GiB.
  - superoffload_mem_nocpuadamw|unsloth-off: 12.28 GiB, reserved 13.33 GiB.
  - zero3_offload_mem_nocpuadamw|unsloth-off: 12.28 GiB, reserved 13.33 GiB.
- memory snapshot peak phase is `after_backward` for all eight artifacts, so these are completed-backward artifacts, not forward-only partials.
- current dense/full peak frame is distinct from base/attention: `asym_gemm/training/qwen3_moe.py:2681:forward` dominates about 4.69 GiB in the dense-current artifacts. The largest live activation detail includes `[16384, 25600]` BF16, about 0.78 GiB at s2048.
- base/attention/fg-plumbing peaks are dominated by the outer checkpoint backward/runtime/optimizer accounting, with live activations around 1.10 GiB.

conclusion: validated

next action:
Proceed to the s8192 memory-shape gate. Do not claim final superiority from s2048. Do not treat `*-fg` as implemented; Stage 6 still requires a dense-specific fine-grained MLP path if Stage 4/5 at larger sequence lengths prove the current dense wrapper is the blocker.

## Stage 2-5 Memory-Shape Gate: s8192

stage: 2-5
run labels:
- `asym|recomp-off-base|ligerloss1`
- `asym|recomp-off-attn|ligerloss1`
- `asym|recomp-off-dense|ligerloss1`
- `asym|recomp-off-full|ligerloss1`
- `superoffload_mem_nocpuadamw|unsloth-off|ligerloss1`
- `zero3_offload_mem_nocpuadamw|unsloth-off|ligerloss1`

run-length note:
Use `--max-steps 1 --warmup-steps 0` for this gate. The purpose is memory shape and
path validation at a larger token count, not stable timing. Treat runtime only as a
rough signal.

backend family: no CPUAdamW comparison family.

CPUAdamW/optimizer-offload family:
- `asym`: no CPUAdamW, no Asym CPU optimizer.
- `*_nocpuadamw`: ZeRO param offload on CPU, optimizer offload absent/disabled.

expected wrappers:
- base: no attention activation wrapper and no dense/current wrapper.
- attn: attention activation wrapper only.
- dense: current dense E=1 surgical wrapper only.
- full: attention plus current dense E=1 surgical wrapper.

expected zero counters:
- base: attention and dense activation counters zero.
- attn: dense activation counters zero.
- dense: attention counters zero.
- nocpuadamw baselines: Asym counters absent/zero.

expected nonzero counters:
- attn/full: attention LoRA-A forward and grad counters.
- dense/full: current dense CPU-left LoRA-A grouped counters.

expected peak owner:
- base/attn should show outer `save_on_cpu` behavior and Asym CPU-resident base weights.
- dense/full may expose the current E=1 dense wrapper, especially fused `gate_up [M,2I]`,
  `stage_concat_columns`, or qwen3_moe dense forward/backward frames.
- baselines may peak in ZeRO param gather or optimizer setup even with optimizer offload
  disabled.

expected comparison baseline:
- compare `asym|recomp-off-*` only to `superoffload_mem_nocpuadamw|unsloth-off` and
  `zero3_offload_mem_nocpuadamw|unsloth-off` at the same sequence length and step count.

expected failure mode:
- OOM in dense/full current wrapper if the fused `[M,2I]` path scales badly.
- OOM or partial in baseline ZeRO gather/setup.
- wrong resolved config or stale artifact should be treated as inconclusive.

partial current-dense observation:
- `asym|recomp-off-dense|ligerloss1` at s8192 reached `model_forward_exit` and then
  stayed in the backward side for more than 16 minutes with GPU utilization at 0% and
  one `pt_autograd_0` CPU thread saturated.
- The command/config were correct for Stage 4:
  `recomp_off_stage=dense`, `UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=true`,
  `ASYMM_DENSE_MLP_SURGICAL_OFFLOAD=1`, `ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD=0`,
  attention off, GPU SiLU off, and no chunking.
- This is not a memory win/fail artifact because the run was interrupted and only
  `partial_profile.json` exists. Treat it as `blocked_by_stage_bug` for the current
  dense surgical implementation: the old E=1 dense wrapper enters a full dense
  CPU/autograd backward path and is not a practical larger-sequence validation path.

conclusion: blocked_by_stage_bug

next action:
Proceed to Stage 6 implementation and validate `recomp-off-dense-fg`/`recomp-off-full-fg`
at s2048 before attempting s8192.

## Stage 6 Small Gate: s2048 Fine-Grained Dense

stage: 6
run labels:
- `asym|recomp-off-dense-fg|ligerloss1`
- `asym|recomp-off-full-fg|ligerloss1`

backend family: no CPUAdamW comparison family.

CPUAdamW/optimizer-offload family:
- `asym`: no CPUAdamW, no Asym CPU optimizer.

expected wrappers:
- dense-fg: fine-grained dense MLP wrapper installed; current E=1 dense surgical
  wrapper not installed; attention off.
- full-fg: fine-grained dense MLP wrapper installed; current E=1 dense surgical
  wrapper not installed; attention wrappers installed.

expected zero counters:
- `dense_mlp_finegrained_stage_concat_columns_calls=0`.
- current dense surgical `stage_concat_columns` path must not appear.
- dense-fg attention counters zero.

expected nonzero counters:
- `dense_mlp_finegrained_forward_calls > 0`.
- `dense_mlp_finegrained_backward_calls > 0`.
- gate/up/down fine-grained base counters > 0.
- full-fg attention counters > 0.

expected peak owner:
- no fused `gate_up [M,2I]` and no `grad_gate_up [M,2I]` stage.
- The first implementation uses fine-grained GPU SiLU backward without the old fused
  `_silu_backward_gpu`/`grad_gate_up` allocation, so `dense_mlp_finegrained_gpu_silu_bwd_calls`
  should be nonzero while `dense_mlp_finegrained_stage_concat_columns_calls` remains zero.

expected comparison baseline:
- s2048 is correctness/path validation only. Compare memory shape against prior s2048
  base/attn/dense artifacts, not final scoreboard claims.

expected failure mode:
- import/wrapper install bug,
- missing fine-grained counters,
- LoRA grad mismatch from the new custom Function,
- native CPU-left or CPU-right helper availability issue,
- memory peak still showing the old current dense wrapper path.

observed dense-fg after Stage 6 implementation:
- run:
  `profiling_fix_fgo/asym_long_sft_smoke__lora__lf__bf16/qwen3-32b__gpus1__b8_s2048_ga1_w0_s1_r64_a16_drop000/asym__source__recomp-off-dense-fg__polnone__routerwhole__expact0__attnact0__layeract0__layergc0__sdparecomp0__loraafwdcpu__actrecomp0__xunpack0__ligerloss1/b8_s2048_ga1`
- status: completed one measured step, finite loss `1.90249`, train runtime
  `27.08s`.
- config truth:
  - `backend=asym`.
  - `recomp_off_stage=dense-fg`.
  - `use_unsloth_gc=true`.
  - `asymm_dense_mlp_finegrained_offload=1`.
  - `asymm_dense_mlp_surgical_offload=0`.
  - `asymm_attn_act_offload=false`.
  - `asymm_expert_act_offload=false`.
  - `use_asym_cpu_adamw=false`.
- wrapper truth from `train.log`:
  - `dense_mlp_finegrained_offload_wrapped=64`.
  - `dense_mlp_act_offload_wrapped=0`.
  - `attention_act_offload_wrapped=0`.
- runtime counters:
  - `dense_mlp_finegrained_forward_calls=64`.
  - `dense_mlp_finegrained_backward_calls=64`.
  - `dense_mlp_finegrained_gate_base_calls=128`.
  - `dense_mlp_finegrained_up_base_calls=128`.
  - `dense_mlp_finegrained_down_base_calls=128`.
  - `dense_mlp_finegrained_stage_concat_columns_calls=0`.
  - `dense_mlp_finegrained_gpu_silu_bwd_calls=64`.
  - `dense_mlp_finegrained_cpu_silu_bwd_calls=0`.
  - `attn_act_lora_a_forward_calls=0`.
  - `attn_act_lora_a_grad_calls=0`.
  - `reference_fallback_count=0`.
- memory:
  - measured/actual peak allocated HBM: `6143.77 MiB`.
  - actual peak reserved HBM: `6824.00 MiB`.
  - whole-process peak reserved HBM: `7638.00 MiB`.
  - actual peak phase: `after_backward`.
  - saved-for-backward activations at peak: `0.00 MiB`.
  - live-output activations at peak: `2882.00 MiB`.
  - temporary/workspace at peak: `1213.77 MiB`.
- conclusion:
  - Stage 6 dense-fg path is mechanically valid at s2048.
  - It uses the intended modular wrapper and avoids the old dense surgical wrapper.
  - It avoids `stage_concat_columns`; the nonzero GPU SiLU counter is the new
    fine-grained Stage 6a schedule, not the old fused `[M,2I]` path.
  - Do not make final memory claims from s2048. Next isolated gate is
    `recomp-off-full-fg` at s2048, then larger-sequence dense/full-fg runs only
    if the small gate remains clean.

observed full-fg after Stage 6 implementation:
- run:
  `profiling_fix_fgo/asym_long_sft_smoke__lora__lf__bf16/qwen3-32b__gpus1__b8_s2048_ga1_w0_s1_r64_a16_drop000/asym__source__recomp-off-full-fg__polnone__routerwhole__expact0__attnact1__layeract0__layergc0__sdparecomp0__loraafwdcpu__actrecomp0__xunpack0__ligerloss1/b8_s2048_ga1`
- status: completed one measured step, finite loss `1.90249`, train runtime
  `27.73s`.
- config truth:
  - `backend=asym`.
  - `recomp_off_stage=full-fg`.
  - `use_unsloth_gc=true`.
  - `asymm_dense_mlp_finegrained_offload=1`.
  - `asymm_dense_mlp_surgical_offload=0`.
  - `asymm_attn_act_offload=true`.
  - `asymm_expert_act_offload=false`.
  - `use_asym_cpu_adamw=false`.
- wrapper truth from `train.log`:
  - `dense_mlp_finegrained_offload_wrapped=64`.
  - `dense_mlp_act_offload_wrapped=0`.
  - `attention_act_offload_wrapped=256`.
  - `attention_saved_tensor_offload_wrapped=64`.
- runtime counters:
  - `dense_mlp_finegrained_forward_calls=64`.
  - `dense_mlp_finegrained_backward_calls=64`.
  - `dense_mlp_finegrained_gate_base_calls=128`.
  - `dense_mlp_finegrained_up_base_calls=128`.
  - `dense_mlp_finegrained_down_base_calls=128`.
  - `dense_mlp_finegrained_stage_concat_columns_calls=0`.
  - `dense_mlp_finegrained_gpu_silu_bwd_calls=64`.
  - `dense_mlp_finegrained_cpu_silu_bwd_calls=0`.
  - `attn_act_hbm_calls=1024`.
  - `attn_act_lora_a_forward_calls=256`.
  - `attn_act_lora_a_grad_calls=256`.
  - `attn_act_stage_low_rank_calls=256`.
  - `reference_fallback_count=0`.
- memory:
  - measured/actual peak allocated HBM: `6149.77 MiB`.
  - actual peak reserved HBM: `7436.00 MiB`.
  - whole-process peak reserved HBM: `8258.00 MiB`.
  - actual peak phase: `after_backward`.
  - saved-for-backward activations at peak: `0.00 MiB`.
  - live-output activations at peak: `2882.00 MiB`.
  - temporary/workspace at peak: `1219.77 MiB`.
- conclusion:
  - Stage 6 full-fg path is mechanically valid at s2048.
  - Dense and attention modules are separately toggled and both fire only in the
    intended composition.
  - The small gate passes. The next gate should be a single isolated s8192
    `dense-fg` run before any `full-fg` or larger-sequence run.

## Stage 6 Medium Gate: s8192 Fine-Grained Dense

stage: 6
run label:
- `asym|recomp-off-dense-fg|ligerloss1`

observed dense-fg:
- run:
  `profiling_fix_fgo/asym_long_sft_smoke__lora__lf__bf16/qwen3-32b__gpus1__b8_s8192_ga1_w0_s1_r64_a16_drop000/asym__source__recomp-off-dense-fg__polnone__routerwhole__expact0__attnact0__layeract0__layergc0__sdparecomp0__loraafwdcpu__actrecomp0__xunpack0__ligerloss1/b8_s8192_ga1`
- status: completed one measured step, finite loss `1.66641`, train runtime
  `98.63s`.
- config truth:
  - `backend=asym`.
  - `recomp_off_stage=dense-fg`.
  - `use_unsloth_gc=true`.
  - `asymm_dense_mlp_finegrained_offload=1`.
  - `asymm_dense_mlp_surgical_offload=0`.
  - `asymm_attn_act_offload=false`.
  - `asymm_expert_act_offload=false`.
  - `use_asym_cpu_adamw=false`.
- wrapper truth from `train.log`:
  - `dense_mlp_finegrained_offload_wrapped=64`.
  - `dense_mlp_act_offload_wrapped=0`.
  - `attention_act_offload_wrapped=0`.
- runtime counters:
  - `dense_mlp_finegrained_forward_calls=64`.
  - `dense_mlp_finegrained_backward_calls=64`.
  - `dense_mlp_finegrained_gate_base_calls=128`.
  - `dense_mlp_finegrained_up_base_calls=128`.
  - `dense_mlp_finegrained_down_base_calls=128`.
  - `dense_mlp_finegrained_stage_concat_columns_calls=0`.
  - `dense_mlp_finegrained_gpu_silu_bwd_calls=64`.
  - `dense_mlp_finegrained_cpu_silu_bwd_calls=0`.
  - `attn_act_lora_a_forward_calls=0`.
  - `attn_act_lora_a_grad_calls=0`.
  - `reference_fallback_count=0`.
- memory:
  - measured/actual peak allocated HBM: `18243.94 MiB`.
  - actual peak reserved HBM: `21264.00 MiB`.
  - whole-process peak reserved HBM: `22080.00 MiB`.
  - actual peak phase: `after_backward`.
  - saved-for-backward activations at peak: `0.00 MiB`.
  - live-output activations at peak: `11528.00 MiB`.
  - temporary/workspace at peak: `4667.94 MiB`.
- direct local comparison artifacts:
  - s8192 base: `20665.32 MiB` allocated, `23280.00 MiB` reserved.
  - s8192 attn: `20681.32 MiB` allocated, `23898.00 MiB` reserved.
  - s8192 dense-fg: `18243.94 MiB` allocated, `22080.00 MiB`
    whole-process reserved.
- conclusion:
  - The medium dense-fg gate passes and shows a real HBM reduction versus the
    local base/attention artifacts.
  - Next isolated gate is s8192 `full-fg`; do not run it concurrently with any
    other experiment.

observed full-fg:
- run:
  `profiling_fix_fgo/asym_long_sft_smoke__lora__lf__bf16/qwen3-32b__gpus1__b8_s8192_ga1_w0_s1_r64_a16_drop000/asym__source__recomp-off-full-fg__polnone__routerwhole__expact0__attnact1__layeract0__layergc0__sdparecomp0__loraafwdcpu__actrecomp0__xunpack0__ligerloss1/b8_s8192_ga1`
- status: completed one measured step, finite loss `1.66641`, train runtime
  `99.81s`.
- config truth:
  - `backend=asym`.
  - `recomp_off_stage=full-fg`.
  - `use_unsloth_gc=true`.
  - `unsloth_gc_recompute_save_on_cpu=true`.
  - `asymm_dense_mlp_finegrained_offload=1`.
  - `asymm_dense_mlp_surgical_offload=0`.
  - `asymm_attn_act_offload=true`.
  - `asymm_expert_act_offload=false`.
  - `asymm_expert_silu_bwd_gpu=0`.
  - `use_asym_cpu_adamw=false`.
- wrapper truth from `train.log`:
  - `dense_mlp_finegrained_offload_wrapped=64`.
  - `dense_mlp_act_offload_wrapped=0`.
  - `attention_act_offload_wrapped=256`.
  - `attention_saved_tensor_offload_wrapped=64`.
- runtime counters:
  - `dense_mlp_finegrained_forward_calls=64`.
  - `dense_mlp_finegrained_backward_calls=64`.
  - `dense_mlp_finegrained_gate_base_calls=128`.
  - `dense_mlp_finegrained_up_base_calls=128`.
  - `dense_mlp_finegrained_down_base_calls=128`.
  - `dense_mlp_finegrained_stage_concat_columns_calls=0`.
  - `dense_mlp_finegrained_gpu_silu_bwd_calls=64`.
  - `dense_mlp_finegrained_cpu_silu_bwd_calls=0`.
  - `attn_act_hbm_calls=1024`.
  - `attn_act_lora_a_forward_calls=256`.
  - `attn_act_lora_a_grad_calls=256`.
  - `attn_act_stage_low_rank_calls=256`.
  - `reference_fallback_count=0`.
- memory:
  - measured/actual peak allocated HBM: `18267.94 MiB`.
  - actual peak reserved HBM: `21202.00 MiB`.
  - whole-process peak reserved HBM: `22018.00 MiB`.
  - actual peak phase: `after_backward`.
  - saved-for-backward activations at peak: `0.00 MiB`.
  - live-output activations at peak: `11528.00 MiB`.
  - temporary/workspace at peak: `4691.94 MiB`.

## Stage 1/6 Combined Baseline Matrix: s8192 no CPUAdamW

stage: 1 and 6
run labels:
- `superoffload_mem_nocpuadamw|unsloth|ligerloss1`
- `superoffload_mem_nocpuadamw|unsloth-off|ligerloss1`
- `zero3_offload_mem_nocpuadamw|unsloth|ligerloss1`
- `zero3_offload_mem_nocpuadamw|unsloth-off|ligerloss1`
- `asym|recomp-off-base|ligerloss1`
- `asym|recomp-off-attn|ligerloss1`
- `asym|recomp-off-dense-fg|ligerloss1`
- `asym|recomp-off-full-fg|ligerloss1`

run-length note:
All rows used `--max-steps 1 --warmup-steps 0`. Treat timings as gate timings, not
stable throughput.

config audit:
- all four `*_nocpuadamw` baselines have `offload_param.device=cpu`,
  `offload_optimizer` absent/disabled, `super_offload=false`, and no CPU optimizer
  state.
- `unsloth` baseline rows have `use_unsloth_gc=true` and
  `unsloth_gc_recompute_save_on_cpu=false`, which is expected for the plain baseline.
- `unsloth-off` baseline rows have `use_unsloth_gc=true` and
  `unsloth_gc_recompute_save_on_cpu=true`.
- all four Asym rows have `use_unsloth_gc=true`,
  `unsloth_gc_recompute_save_on_cpu=true`, no CPUAdamW, and the expected
  `recomp_off_stage`.

comparison table:

Important: `top_step_H` is the top-level allocator peak recorded in
`profile.json.memory.peak_allocated_hbm_bytes`. For the s8192 baseline rows it disagrees
with the memory-breakdown peak and hides the expected `unsloth` versus `unsloth-off`
activation difference. Use `breakdown_H`, `act_H`, and `saved_H` for this diagnostic.

```text
Model: qwen3-32b    LoRA: r64/a16/d0.00    CPUAdam: no
Workload   Backend                         Config                    fwd_s  bwd_s  step_s  top_step_H  breakdown_H  act_H  saved_H  reserved_H    RAM
---------  ------------------------------  ------------------------  --------------------  -------------------------------  ----------------
s8192.b8   superoffload_mem_nocpuadamw     unsloth                     9.9   35.7   45.8       18.1         32.9   22.6     22.0        36.0  139.7
s8192.b8   superoffload_mem_nocpuadamw     unsloth-off                10.0   59.9   70.1       18.1         21.2    3.8      0.0        24.0  187.9
s8192.b8   zero3_offload_mem_nocpuadamw    unsloth                     9.8   36.1   46.1       18.1         32.9   22.6     22.0        36.0  139.8
s8192.b8   zero3_offload_mem_nocpuadamw    unsloth-off                 9.9   60.2   70.3       18.1         21.2    3.8      0.0        24.0  187.8
s8192.b8   asym                            recomp-off-base            22.2   73.4   95.7       20.2         20.2    4.4      0.0        22.7  348.2
s8192.b8   asym                            recomp-off-attn            14.1   70.2   84.5       20.2         20.2    4.4      0.0        23.3  229.6
s8192.b8   asym                            recomp-off-dense-fg        14.0   82.6   96.8       17.8         17.8   11.3      0.0        21.6  249.1
s8192.b8   asym                            recomp-off-full-fg         14.1   83.7   98.0       17.8         17.8   11.3      0.0        21.5  231.6
```

interpretation:
- The s8192 gate supports the expected `unsloth-off` mechanism for the baselines:
  saved HBM activations drop from about `22.0 GiB` to `0.0 GiB`, while RAM rises.
- The previous top-level `step_H=18.1 GiB` baseline number is not a valid activation
  comparison because it misses the memory-breakdown peak that contains the saved
  activations.
- `dense-fg` and `full-fg` remove the old dense-current regression and keep saved HBM
  activations at `0.0 GiB`, but their live activations/workspace still need larger-seq
  validation.
- `base` and `attn` alone do not beat the no-CPUAdamW `unsloth-off` baselines on
  breakdown HBM. This keeps the dense MLP path as the meaningful Stage-6 axis.
- `full-fg` is below `unsloth-off` on breakdown HBM at s8192 (`17.8 GiB` versus
  `21.2 GiB`), but the Asym rows are slower and use higher host RAM. Do not claim final
  success from this gate.
- The next rung is a single s16384 `asym|recomp-off-full-fg` run, followed by matching
  no-CPUAdamW baselines only if the Asym row completes and its counters/config remain
  clean.

next action:
Run s16384 `asym|recomp-off-full-fg|ligerloss1` first, serialized. Expected shape:
allocated HBM should scale materially below `base/attn`; if it is not lower than the
matching no-CPUAdamW `unsloth-off` baselines once those baselines are run, inspect the
peak frame before changing design.

## Stage 6 Forensic Gate: s16384 no CPUAdamW

stage: 6
run labels:
- `superoffload_mem_nocpuadamw|unsloth|ligerloss1`
- `superoffload_mem_nocpuadamw|unsloth-off|ligerloss1`
- `asym|recomp-off-full-fg|ligerloss1`

run-length note:
All rows used `--max-steps 1 --warmup-steps 0`. This remains a gate run, not a stable
throughput claim.

expected shape before running:
- `unsloth` should have large HBM saved activations.
- `unsloth-off` should remove HBM saved activations and increase host RAM.
- `asym|recomp-off-full-fg` should keep saved HBM activations at zero and avoid
  `stage_concat_columns`.
- If `asym|recomp-off-full-fg` is not lower than `unsloth-off` on breakdown HBM, inspect
  the peak owner before changing implementation.

observed comparison:

```text
Model: qwen3-32b    LoRA: r64/a16/d0.00    CPUAdam: no
Workload   Backend                         Config                    fwd_s  bwd_s  step_s  top_H  br_H  act_H  saved_H  live_H  temp_H  res_H    RAM
---------  ------------------------------  ------------------------  --------------------  ------------------------------------------------
s16384.b8  superoffload_mem_nocpuadamw     unsloth                    17.8   50.7    68.7   33.1  62.5   43.6     42.3     1.2    14.9   68.8  204.4
s16384.b8  superoffload_mem_nocpuadamw     unsloth-off                18.0  100.8   119.0   33.1  39.3    7.5      0.0     7.5    27.8   42.5  298.2
s16384.b8  asym                            recomp-off-full-fg         27.2  137.1   164.5   33.6  33.6   22.5      0.0    22.5     9.1   39.8  332.0
```

config/counter audit:
- `superoffload_mem_nocpuadamw|unsloth` has `use_unsloth_gc=true`,
  `unsloth_gc_recompute_save_on_cpu=false`, `offload_param.device=cpu`,
  `super_offload=false`, and no CPU optimizer state.
- `superoffload_mem_nocpuadamw|unsloth-off` has `use_unsloth_gc=true`,
  `unsloth_gc_recompute_save_on_cpu=true`, `offload_param.device=cpu`,
  `super_offload=false`, and no CPU optimizer state.
- `asym|recomp-off-full-fg` has `use_unsloth_gc=true`,
  `unsloth_gc_recompute_save_on_cpu=true`, `recomp_off_stage=full-fg`,
  `asymm_dense_mlp_finegrained_offload=1`,
  `asymm_dense_mlp_surgical_offload=0`, `asymm_attn_act_offload=true`,
  and `use_asym_cpu_adamw=false`.
- `asym|recomp-off-full-fg` counters:
  `dense_mlp_finegrained_forward_calls=64`,
  `dense_mlp_finegrained_backward_calls=64`,
  `dense_mlp_finegrained_gate_base_calls=128`,
  `dense_mlp_finegrained_up_base_calls=128`,
  `dense_mlp_finegrained_down_base_calls=128`,
  `dense_mlp_finegrained_stage_concat_columns_calls=0`,
  `dense_mlp_finegrained_gpu_silu_bwd_calls=64`,
  `dense_mlp_finegrained_cpu_silu_bwd_calls=0`,
  `attn_act_lora_a_forward_calls=256`,
  `attn_act_lora_a_grad_calls=256`,
  `reference_fallback_count=0`.

interpretation:
- The s16384 gate validates the basic implementation direction.
- `unsloth-off` removes saved HBM activations as intended: `42.3 GiB` saved HBM in
  `unsloth` becomes `0.0 GiB` in `unsloth-off`.
- `asym|recomp-off-full-fg` also has `0.0 GiB` saved HBM activations and is lower than
  no-CPUAdamW `unsloth-off` on breakdown HBM: `33.6 GiB` versus `39.3 GiB`.
- The remaining Asym peak is live activations/workspace, dominated by LoRA live outputs
  and not by saved activations or the old fused dense path.
- This is enough to advance to s30000. It is not a final claim; s30000 is the first real
  bottleneck workload.

next action:
Run s30000 `asym|recomp-off-full-fg|ligerloss1` first, serialized. If it completes and
the same counters/config remain clean, run matching s30000 baselines. Do not jump to
s50000 until s30000 artifacts are complete and understood.

## Stage 6 First Real Bottleneck Gate: s30000 no CPUAdamW

stage: 6
run labels:
- `superoffload_mem_nocpuadamw|unsloth|ligerloss1`
- `superoffload_mem_nocpuadamw|unsloth-off|ligerloss1`
- `asym|recomp-off-full-fg|ligerloss1`

run-length note:
The row used `--max-steps 1 --warmup-steps 0`. This is still a single-step gate, but
`s30000.b8` is large enough that activation placement should be meaningful.

expected shape before running:
- `asym|recomp-off-full-fg` should complete before any baseline rerun is attempted.
- It must preserve the same clean config/counters as s16384:
  `use_unsloth_gc=true`, `unsloth_gc_recompute_save_on_cpu=true`,
  `recomp_off_stage=full-fg`, fine-grained dense on, old dense surgical off,
  attention activation offload on, no CPUAdamW, no fallback, and no
  `stage_concat_columns`.
- HBM saved activations should remain `0.0 GiB`. If the row is worse than the matching
  `unsloth-off` baseline after that baseline is run, inspect the live/temp owners before
  changing implementation.

observed comparison:

```text
Model: qwen3-32b    LoRA: r64/a16/d0.00    CPUAdam: no
Workload   Backend                         Config                    fwd_s  bwd_s  opt_s  step_s  top_H  br_H  act_H  saved_H  live_H  temp_H  res_H    RAM
---------  ------------------------------  ------------------------  ---------------------------  ------------------------------------------------
s30000.b8  superoffload_mem_nocpuadamw     unsloth                    35.1   87.3    0.1   122.7   58.0 111.7   80.0     77.7     2.3    27.7  123.1  332.2
s30000.b8  superoffload_mem_nocpuadamw     unsloth-off                34.7  220.3    0.1   255.2   58.0  69.5   13.7      0.0    13.7    51.7   75.9  520.3
s30000.b8  asym                            recomp-off-full-fg         60.9  319.7    0.8   380.8   59.8  59.8   41.2      0.0    41.2    16.6   71.5  535.9
```

config/counter audit:
- `superoffload_mem_nocpuadamw|unsloth` has `use_unsloth_gc=true`,
  `unsloth_gc_recompute_save_on_cpu=false`, `offload_param.device=cpu`,
  `super_offload=false`, and no CPU optimizer state.
- `superoffload_mem_nocpuadamw|unsloth-off` has `use_unsloth_gc=true`,
  `unsloth_gc_recompute_save_on_cpu=true`, `offload_param.device=cpu`,
  `super_offload=false`, and no CPU optimizer state.
- `asym|recomp-off-full-fg` has `use_unsloth_gc=true`,
  `unsloth_gc_recompute_save_on_cpu=true`, `recomp_off_stage=full-fg`,
  `asymm_dense_mlp_finegrained_offload=1`,
  `asymm_dense_mlp_surgical_offload=0`, `asymm_attn_act_offload=true`,
  `asymm_expert_act_offload=false`, and `use_asym_cpu_adamw=false`.
- Fine-grained dense counters are clean:
  `dense_mlp_finegrained_forward_calls=64`,
  `dense_mlp_finegrained_backward_calls=64`,
  `dense_mlp_finegrained_gate_base_calls=128`,
  `dense_mlp_finegrained_up_base_calls=128`,
  `dense_mlp_finegrained_down_base_calls=128`,
  `dense_mlp_finegrained_stage_concat_columns_calls=0`,
  `dense_mlp_finegrained_gpu_silu_bwd_calls=64`,
  `dense_mlp_finegrained_cpu_silu_bwd_calls=0`.
- Attention counters are clean:
  `attn_act_lora_a_forward_calls=256`,
  `attn_act_lora_a_grad_calls=256`,
  `attn_act_stage_low_rank_calls=256`.
- `reference_fallback_count=0`.

interpretation so far:
- The first real bottleneck comparison validates the activation-placement diagnosis.
- Plain `unsloth` keeps large saved activations in HBM: `77.7 GiB` saved HBM.
- `unsloth-off` removes those saved HBM activations (`0.0 GiB`) but shifts the peak to
  temporary/workspace (`51.7 GiB`).
- `asym|recomp-off-full-fg` also keeps saved HBM activations at `0.0 GiB` and is lower
  than no-CPUAdamW `unsloth-off` on breakdown HBM: `59.8 GiB` versus `69.5 GiB`.
- The remaining Asym peak is live activation-heavy (`41.2 GiB`), while the baseline
  offload peak is temp/workspace-heavy. This is an implementation/design issue to
  inspect, not evidence of saved-activation buildup.
- Timings remain unfavorable for Asym in this gate. Do not mix memory and throughput
  conclusions.

next action:
Run the actual `superoffload_mem` CPUAdam family at s30000 next, serialized:
`superoffload_mem|unsloth|ligerloss1` and `superoffload_mem|unsloth-off|ligerloss1`.
Use those rows only for comparison to an Asym CPUAdam-family row, not the no-CPUAdamW
`asym` row above.

## Stage 7 Actual Baseline Family: s30000 CPUAdam/SuperOffload

stage: 7
run labels:
- `superoffload_mem|unsloth|ligerloss1`
- `superoffload_mem|unsloth-off|ligerloss1`

run-length note:
Both rows used `--max-steps 1 --warmup-steps 0`. Treat timings as single-step gate
timings only.

observed comparison:

```text
Model: qwen3-32b    LoRA: r64/a16/d0.00    CPUAdam: SuperOffload/DeepSpeedCPUAdam
Workload   Backend                         Config                    fwd_s  bwd_s  opt_s  step_s  top_H  br_H  act_H  saved_H  live_H  temp_H  res_H    RAM
---------  ------------------------------  ------------------------  ---------------------------  ------------------------------------------------
s30000.b8  superoffload_mem                unsloth                    35.1   67.4    0.1   102.7  108.7 108.7   79.9     77.6     2.3    28.8  120.0  340.0
s30000.b8  superoffload_mem                unsloth-off                35.2  201.2    0.1   236.6   66.5  66.5   13.7      0.0    13.7    52.7   72.9  528.8
```

config audit:
- Both rows use `ds_z3_superoffload_mem_config.json`.
- Both rows verify `SuperOffloadOptimizer_Stage3`, `DeepSpeedCPUAdam`, and
  `DeepSpeed SuperOffload runtime enabled = true`.
- `superoffload_mem|unsloth` has `use_unsloth_gc=true` and
  `unsloth_gc_recompute_save_on_cpu=false`.
- `superoffload_mem|unsloth-off` has `use_unsloth_gc=true` and
  `unsloth_gc_recompute_save_on_cpu=true`.

interpretation:
- This is the actual family for the named external baseline.
- At s30000, `unsloth-off` removes saved HBM activations (`77.6 GiB` to `0.0 GiB`) and
  reduces breakdown HBM from `108.7 GiB` to `66.5 GiB`, with a large backward-time and
  host-RAM cost.
- The no-CPUAdamW `asym|recomp-off-full-fg` row from Stage 6 is lower than this
  `superoffload_mem|unsloth-off` row on breakdown HBM (`59.8 GiB` versus `66.5 GiB`),
  but it is not an optimizer-family apples-to-apples result. Do not claim final victory
  until an Asym CPUAdam-family row is added or the comparison is explicitly scoped to
  activation-only/no-CPUAdam behavior.

next action:
Before running s50000, inspect whether matching s50000 artifacts already exist. If not,
run the same rows serially, starting from `superoffload_mem|unsloth|ligerloss1` because
that is the named baseline to beat.

## Stage 7 Fix: Dense Fine-Grained LoRA Weight-Offload Ownership

stage: 7
issue:
- First `asym_cpuadamwds|recomp-off-full-fg|ligerloss1` gate at s8192 failed before
  producing a full profile:
  `RuntimeError: gate: CPU-left LoRA-A expects source [M,K] and weight [E,r,K]`.
- The resolved config was otherwise correct: `backend=asym_cpuadamwds`,
  `use_unsloth_gc=true`, `unsloth_gc_recompute_save_on_cpu=true`,
  `asymm_dense_mlp_finegrained_offload=1`, `asymm_dense_mlp_surgical_offload=0`,
  `asymm_attn_act_offload=true`, `use_asym_cpu_adamw=true`,
  `asym_cpu_adamw_grad_offload=true`, and `asym_cpu_adamw_weight_offload=true`.
- Root cause: the LoRA weight-offload installer registered the dense MLP child
  `AsymLoRALinear` modules independently, so the parent
  `AsymFinegrainedDenseMLP` did not own a coordinator group. The custom dense MLP
  Function then read released 0-size child LoRA placeholders instead of gathered
  `[r,K]` LoRA-A tensors.

implementation:
- `AsymFinegrainedDenseMLP` now exposes `_lora_weight_banks()` for its six dense MLP
  LoRA banks.
- `install_lora_weight_offload()` now registers each `AsymFinegrainedDenseMLP` parent
  before scanning generic child LoRA modules, marks `gate/up/down` children as
  parent-owned, and installs parent gather/release hooks.
- Dense fine-grained LoRA-A helpers now fail with explicit shape diagnostics if a
  non-`[r,K]` or non-`[1,r,K]` LoRA-A reaches the CPU-left/CPU-right path.
- Regression test added:
  `tests/training/test_dense_mlp_finegrained.py::test_finegrained_dense_mlp_weight_offload_registers_parent_group`.

validation:
- `pytest -q tests/training/test_dense_mlp_finegrained.py` passed: `2 passed`.
- s8192 retry completed:
  `asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 8192|8|1`.
- s8192 retry config/counters:
  - installed LoRA weight offload on `896` banks across `320` groups:
    `mlp_dense:64`, `attention:256`.
  - CPUAdamW grad hooks offloaded `896` grads / `536870912` elements.
  - `dense_mlp_finegrained_forward_calls=64`.
  - `dense_mlp_finegrained_backward_calls=64`.
  - `dense_mlp_finegrained_stage_concat_columns_calls=0`.
  - `dense_mlp_finegrained_gpu_silu_bwd_calls=64`.
  - `reference_fallback_count=0`.
- s8192 retry memory:
  - fwd/bwd/step HBM peak: `15.1/15.9/15.9 GiB`.
  - breakdown HBM: `15.9 GiB`.
  - saved HBM activations: `0.0 GiB`.
  - activation HBM at peak: `8.1 GiB`.
  - temp/workspace HBM at peak: `7.7 GiB`.

conclusion:
- The CPUAdamW/LoRA-weight-offload failure was not a fine-grained activation schedule
  failure. It was an ownership bug in trainable LoRA weight staging for the new dense
  MLP parent wrapper.
- After the fix, the CPUAdamW family can be compared apples-to-apples with
  `superoffload_mem`.

## Stage 7 Actual Baseline Family: s30000 with Asym CPUAdamWDS

stage: 7
run labels:
- `superoffload_mem|unsloth|ligerloss1`
- `superoffload_mem|unsloth-off|ligerloss1`
- `asym_cpuadamwds|recomp-off-full-fg|ligerloss1`

run-length note:
All rows used `--max-steps 1 --warmup-steps 0`. Treat timings as single-step gate
timings only.

observed comparison:

```text
Model: qwen3-32b    LoRA: r64/a16/d0.00    CPUAdam: SuperOffload/Asym CPUAdamWDS
Workload   Backend               Config              fwd_s  bwd_s  opt_s  step_s  fwd_H  bwd_H  step_H  br_H  act_H  saved_H  live_H  temp_H    RAM
---------- --------------------- ------------------- ------ ------ ----- ------- ------ ------ ------- ----- ------ -------- ------ ------ ------
s30000.b8  superoffload_mem      unsloth               35.1   67.4   0.1   102.6   55.0  108.7   108.7 108.7   79.9     77.6    2.3   28.8  340.0
s30000.b8  superoffload_mem      unsloth-off           35.2  201.2   0.1   236.4   55.0   66.5    66.5  66.5   13.7      0.0   13.7   52.7  528.8
s30000.b8  asym_cpuadamwds       recomp-off-full-fg    52.7  310.9   2.5   366.1   55.0   57.9    57.9  57.9   29.8      0.0   29.8   28.1  545.3
```

config/counter audit for `asym_cpuadamwds|recomp-off-full-fg`:
- `use_unsloth_gc=true`.
- `unsloth_gc_recompute_save_on_cpu=true`.
- `recomp_off_stage=full-fg`.
- `asymm_dense_mlp_finegrained_offload=1`.
- `asymm_dense_mlp_surgical_offload=0`.
- `asymm_attn_act_offload=true`.
- `use_asym_cpu_adamw=true`.
- `asym_cpu_adamw_grad_offload=true`.
- `asym_cpu_adamw_weight_offload=true`.
- `asymm_mlp_recompute_chunk=0`.
- LoRA weight offload groups: `mlp_dense:64`, `attention:256`.
- CPUAdamW grad hooks offloaded `896` grads / `536870912` elements.
- Runtime counters: `asym_forward_calls=896`, `asym_dx_calls=704`,
  `torch_forward_calls=0`, `torch_dx_calls=0`.
- Dense counters: `forward=64`, `backward=64`, `stage_concat_columns=0`,
  `gpu_silu_bwd=64`.
- `reference_fallback_count=0`.

interpretation:
- At s30000, the fixed `asym_cpuadamwds|recomp-off-full-fg` row is lower than the
  actual SuperOffload `unsloth-off` row on breakdown HBM: `57.9 GiB` versus `66.5 GiB`.
- It is much lower than the named plain baseline `superoffload_mem|unsloth`:
  `57.9 GiB` versus `108.7 GiB`.
- It preserves the intended saved-activation behavior: `saved_H=0.0 GiB`.
- It is slower and uses slightly more host RAM than `unsloth-off`; this stage proves
  memory direction, not throughput superiority.

## Stage 7 Actual Baseline Family: s50000 with Asym CPUAdamWDS

stage: 7
run labels:
- `superoffload_mem|unsloth|ligerloss1`
- `superoffload_mem|unsloth-off|ligerloss1`
- `asym_cpuadamwds|recomp-off-full-fg|ligerloss1`

run-length note:
All rows used `--max-steps 1 --warmup-steps 0`. These are fresh current-harness rows
with `r64/a16/d0.00`, not the older historical s50000 artifacts.

observed comparison:

```text
Model: qwen3-32b    LoRA: r64/a16/d0.00    CPUAdam: SuperOffload/Asym CPUAdamWDS
Workload   Backend               Config              fwd_s  bwd_s  opt_s  step_s  fwd_H  bwd_H  step_H  br_H  act_H  saved_H  live_H  temp_H    RAM
---------- --------------------- ------------------- ------ ------ ----- ------- ------ ------ ------- ----- ------ -------- ------ ------ ------
s50000.b8  superoffload_mem      unsloth               57.0  219.2   0.1   276.2   91.6  180.9   180.9 180.9  147.6    143.8    3.8   33.3  340.4
s50000.b8  superoffload_mem      unsloth-off           56.7  368.3   0.1   425.1   91.6  110.7   110.7 110.7   22.9      0.0   22.9   87.9  644.4
s50000.b8  asym_cpuadamwds       recomp-off-full-fg   101.0  561.6   2.5   665.2   91.7   96.4    96.4  96.4   49.6      0.0   49.6   46.8  657.7
```

config/counter audit for `asym_cpuadamwds|recomp-off-full-fg`:
- `use_unsloth_gc=true`.
- `unsloth_gc_recompute_save_on_cpu=true`.
- `recomp_off_stage=full-fg`.
- `asymm_dense_mlp_finegrained_offload=1`.
- `asymm_dense_mlp_surgical_offload=0`.
- `asymm_attn_act_offload=true`.
- `use_asym_cpu_adamw=true`.
- `asym_cpu_adamw_grad_offload=true`.
- `asym_cpu_adamw_weight_offload=true`.
- `asymm_mlp_recompute_chunk=0`.
- LoRA weight offload groups: `mlp_dense:64`, `attention:256`.
- CPUAdamW grad hooks offloaded `896` grads / `536870912` elements.
- Runtime counters: `asym_forward_calls=896`, `asym_dx_calls=704`,
  `torch_forward_calls=0`, `torch_dx_calls=0`.
- Dense counters: `forward=64`, `backward=64`, `stage_concat_columns=0`,
  `gpu_silu_bwd=64`.
- `reference_fallback_count=0`.

interpretation:
- The fixed path achieves the intended real-workload memory result at s50000:
  `asym_cpuadamwds|recomp-off-full-fg` is below the actual SuperOffload
  `unsloth-off` row on breakdown HBM: `96.4 GiB` versus `110.7 GiB`.
- It is far below the named plain baseline `superoffload_mem|unsloth`:
  `96.4 GiB` versus `180.9 GiB`.
- The plain baseline is near the allocator cap at s50000 and has
  `saved_H=143.8 GiB`; the fixed path has `saved_H=0.0 GiB`.
- The fixed path's remaining peak is live activation/workspace (`49.6 GiB` live,
  `46.8 GiB` temp), not saved activation buildup and not the old fused dense
  `stage_concat_columns` path.
- Runtime remains substantially worse: `665.2s` step for fixed Asym versus
  `425.1s` for `unsloth-off` and `276.2s` for plain `unsloth`.
- Host RAM is comparable to `unsloth-off` at s50000 (`657.7 GiB` versus `644.4 GiB`).

current conclusion:
- The original core concern is resolved for memory: after fixing dense LoRA weight
  ownership, `recompute-offload-back on demand + AsymGEMM` is lower HBM than
  `superoffload_mem|unsloth-off` at both s30000 and s50000.
- The design does not currently beat SuperOffload on runtime. Further work should focus
  on reducing fine-grained staging overhead and live activation/workspace peaks without
  reintroducing chunked MLP or the old fused `[M,2I]` dense path.
