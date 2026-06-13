# Expert Activation Offload V2 Staged Implementation Plan

Goal: reduce peak HBM during Qwen3 MoE LoRA SFT by offloading almost all
forward routed expert activations to pinned CPU memory, then consuming those
activations in backward through grouped CPU-source AsymGEMM/native kernels.

The fair claim is narrow:

```text
With model, optimizer, LoRA config, precision, routing, batch, sequence length,
profiler, and global recompute mode held fixed, replacing expert activation
recompute with expert activation offload plus grouped CPU-source AsymGEMM
fetchback reduces peak HBM.
```

Do not use unrelated memory optimizers to prove this claim. LoRAFusion,
alternate fused MoE stacks, different loss kernels, different checkpointing, or
optimizer changes are useful design references only if both sides of the
comparison get the same change. The main comparison remains:

```text
BACKEND_SPECS=asym_cpuadamwds|norecomp
ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false
```

This is the active
`/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh`
workflow. `none|true` is the target implementation. `gc-exp|false` is the
expert checkpoint/recompute baseline in this workflow. `none|false` is the
no-expact/no-expert-recompute control that shows whether the workload fits
without either expert memory strategy. Global transformer recompute remains a
useful lower bound, not the primary comparison.

Priority order:

1. Preserve and increase the HBM reduction of `none|true`.
2. Improve `none|true` latency whenever the change does not materially increase
   peak HBM or only increases it within a clearly justified, roughly flat
   budget.
3. Reject latency optimizations that erase the memory win, even if they make
   the path faster.

All LF profiling and acceptance validation must use the workflow script:

```bash
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Use direct unit tests and `scripts/testing/profile_qwen3_activation_offload.py`
only for fast kernel or wrapper checks. They do not replace the LF workflow
comparison. The canonical LF profile command shape is:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
OUTPUT_ROOT="$PWD/outputs/<run_name>" \
GPU_POOL=<gpu_id_or_pool> \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false \
ASYM_OFFLOAD_MODULES=all \
SEQ_LENS=<seq_len> \
PER_DEVICE_TRAIN_BATCH_SIZE=<batch_size> \
MAX_STEPS=<measured_steps> \
WARMUP_STEPS=<warmup_steps> \
PLOT=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus <gpu_id_or_pool>
```

For the current canonical `b4_s6144` validation, use `SEQ_LENS=6144`,
`PER_DEVICE_TRAIN_BATCH_SIZE=4`, and the same policy axis above. For final
acceptance, keep this script and only vary batch/sequence/profiler settings
deliberately.

Peak-memory attribution is part of the implementation loop. After each stage,
inspect the source profile, memory breakdown rows, stage tags, saved activation
attribution, live activation attribution, and allocator peak-growth attribution.
If the current attribution cannot explain the new peak or cannot distinguish
expert-block memory from loss/cross-entropy or other non-expert peaks, improve
the attribution first before guessing at an optimization. It is acceptable to
add stage-local tags, counters, saved-tensor labels, allocator snapshots, or
profile JSON fields when needed to surface the real issue.

## Non-Negotiable Invariants

- No Python loop or host loop that launches one GEMM per expert.
- No small-GEMM decomposition of expert work. Normal training uses grouped
  kernels/GEMMs.
- No full HBM staging of `X_cpu`, `act_cpu`, `dgate_cpu`, or `dup_cpu`.
- No full HBM concat solely to satisfy a convenient grouped_mm API.
- No full CUDA FP32 accumulator shaped like a full LoRA-A grad, LoRA-B grad, or
  full low-rank `dS`.
- `ctx` for activation offload stores CPU handles, metadata, weights, scalar
  config, and trainable LoRA tensors only. It must not keep CUDA activation
  views or staged CUDA tensors alive across forward.
- Every CUDA intermediate is released at its last use before the next
  high-memory region.
- D2H copies that feed CPU-side math have explicit completion before CPU reads.
- H2D/CPU-source kernel reads are stream ordered with their producer copies.
- Async prefetch/staging must be bounded. Async itself is not free: it can raise
  peak HBM if a staged buffer overlaps a still-live producer or consumer. The
  design must count in-flight bytes and cap concurrent staged/prefetch buffers.
- LoRA trainable parameters stay CUDA `nn.Parameter`s with normal `.grad`
  tensors. `AsymCPUAdamW` owns CPU master/state, but backward must still return
  CUDA gradients for the LoRA parameters.

Allowed CUDA materialization:

- final expert output `[M,H]`
- final input gradient `[M,H]` when the caller needs `dX`
- low-rank `[M,r]` tensors when they are true LoRA-A outputs
- final LoRA gradients with true parameter shapes
- tile-sized scratch whose size is independent of full activation width

## Current Blocking Issues

The current `ASYMM_EXPERT_ACT_OFFLOAD=true` path runs, but it is not the target
algorithm.

- `asym_gemm/training/qwen3_moe.py::_ActivationOffloadQwen3ExpertFunction.forward`
  restages `act_cpu` through `manager.stage(..., tag="act_for_down_base")` before
  down base.
- `asym_gemm/training/qwen3_moe.py::_ActivationOffloadQwen3ExpertFunction.backward`
  restages and concatenates `dgate`/`dup` through
  `manager.stage_concat_columns(..., tag="dgate_up_for_gate_up_base")`.
- The expact backward still uses `_grouped_lora_weight_grads_torch`,
  `_grouped_lora_cuda_view`, and `stage_low_rank_from_cpu` for LoRA-B regions.
- `csrc/exp_act_offload/exp_act_offload_kernels.cu` allocates full FP32
  `grad_acc`, `grad_gate_acc`, `grad_up_acc`, `dS_acc`, and `grad_b_acc`.
- `ActivationOffloadManager.offload` uses nonblocking D2H into pinned CPU
  buffers, but CPU-side consumers such as `_activation_offload_cpu_silu_mul`
  need an explicit completion contract.
- `grouped_lora_a_pair_forward_cpu_left` calls the single CPU-left LoRA-A path
  twice, so gate/up do not yet share one CPU source tile.
- Existing tests assert the old stage tags
  `act_for_down_base` and `dgate_up_for_gate_up_base`; those assertions should
  flip once Stage 1 lands.

## Related Code Constraints

Local code checked for design constraints:

- Megatron-LM keeps MoE work grouped by `tokens_per_expert` and has fine-grained
  activation offload/checkpoint lifetimes. Map that lesson to grouped offsets
  and explicit release points here, not to a new execution stack.
- DeepSpeed activation checkpointing and ZeRO offload use explicit buffer
  ownership/reset points. Map that to `ActivationOffloadManager` live-byte and
  cache accounting.
- DeepSpeed FPDT uses chunked `load_to_gpu`, `offload`, stream waits, and
  double-buffer style scheduling. Use that as the async model only after HBM is
  counted and bounded.
- LlamaFactory is the harness. Preserve its labels and CPU Adam integration.
- LoRAFusion is not the fair-comparison mechanism. It only motivates paired
  LoRA scheduling and avoiding repeated LoRA passes.

## Preserved Evidence And Baseline Context

These earlier findings remain important implementation context:

- Current canonical workflow, Qwen3-30B-A3B `b4_s6144`, dropout `0.00`,
  backend `asym_cpuadamwds|norecomp`, router `whole`:
  - `gc-exp|false`, implementation `torch checkpoint`:
    - peak allocated `167.462 GiB`
    - peak reserved `181.883 GiB`
    - average step `3.649 s`
    - average forward `1.464 s`
    - average backward `2.135 s`
  - `none|true`, implementation `activation offload`:
    - peak allocated `143.368 GiB`
    - peak reserved `158.055 GiB`
    - average step `42.842 s`
    - average forward `9.323 s`
    - average backward `33.430 s`
  - interpretation: current `none|true` already saves about `24.094 GiB`
    allocated HBM versus `gc-exp|false`, but it is about `11.7x` slower by
    average step time. The staged plan must preserve the memory win first while
    removing the avoidable latency from full staging, CPU idle regions, full
    FP32 workspaces, and repeated CPU-source passes.

- Qwen3 `4x8192`, recompute on:
  - `expact0`: peak allocated `73.546 GB`, measured step `14.708 s`
  - `expact1`: peak allocated `73.546 GB`, measured step `107.291 s`
- Qwen3 `4x8192`, recompute off:
  - `expact0`: OOM in first forward, peak allocated about `196.158 GB`
  - `expact1`: OOM in first forward, peak allocated about `195.853 GB`
- Qwen3 `2x8192`, recompute off:
  - `expact0`: OOM in first forward/loss, peak allocated `193.732 GB`; the
    failing allocation was cross-entropy trying to allocate about `9.27 GiB`
  - `expact1`: completed; peak allocated `135.628 GB`, peak reserved
    `141.503 GB`, measured step `44.950 s`, forward `10.578 s`,
    backward `32.939 s`

Interpretation to preserve:

- current expact can reduce enough activation HBM to make smaller no-recompute
  runs fit
- current expact does not yet beat the matched recompute path because it
  recreates wide HBM stages and full FP32 workspaces
- the `4x8192` no-recompute failure is still dominated by allocations outside
  the intended CPU-source schedule
- loss/cross-entropy peaks can hide the expert result, so expert-block peaks
  and loss peaks must be reported separately

Keep the three baseline categories separate:

- canonical workflow comparison:
  `BACKEND_SPECS=asym_cpuadamwds|norecomp` and
  `ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false`
- global recompute lower bound:
  `BACKEND_SPECS=asym_cpuadamwds|recomp` and
  `ASYMM_EXP_ACT_POLICIES=none|false`
- optional global-plus-expert recompute sanity check:
  `BACKEND_SPECS=asym_cpuadamwds|recomp` and
  `ASYMM_EXP_ACT_POLICIES=gc-exp|false`

Older thresholded token-policy examples are not the default workflow for this
plan. Use them only as explicit extra experiments if we need to compare custom
thresholded recompute against `gc-exp`.

## Stage 0: Measurement, Lifetime, And CPU Adam Guardrails

### Scope

- `asym_gemm/training/activation_offload.py`
  - `CPUActivationHandle`
  - `ActivationOffloadStats`
  - `ActivationOffloadManager.empty_cpu`
  - `ActivationOffloadManager.offload`
  - `ActivationOffloadManager.stage`
  - `ActivationOffloadManager.stage_concat_columns`
  - `ActivationOffloadManager.release_stage`
  - `ActivationOffloadManager.release_cpu`
  - `ActivationOffloadManager.snapshot`
- `asym_gemm/training/frozen_linear.py`
  - `AsymExecutionStats`
- `asym_gemm/training/qwen3_moe.py`
  - `_ActivationOffloadQwen3ExpertFunction.forward`
  - `_ActivationOffloadQwen3ExpertFunction.backward`
  - `_activation_offload_cpu_silu_mul`
  - `_activation_offload_cpu_silu_backward`
  - `AsymQwen3Experts._last_activation_offload_stats`
- `asym_gemm/integrations/lf.py`
  - `LFAsymReport.runtime_log_string`
- `scripts/lf/run_lf_profiled_train.py`
  - source profile JSON assembly near `SourceProfileRecorder.build_profile`
  - add model/runtime extraction for Asym execution stats if needed
- Tests:
  - `tests/training/test_lf_qwen3_asym_backend.py`
  - `tests/training/test_asym_cpu_adamw.py`
  - `tests/lf/test_asym_cpu_adamw_lf_integration.py`
  - `tests/lf/test_asym_cpu_adamw_args.py`

### Code Changes

1. Extend `CPUActivationHandle` with copy-order metadata:
   - `producer_stream` or `producer_event`
   - `ready_for_cpu_read`
   - `copy_direction`
   - `requires_cpu_sync_before_read`

2. Split offload APIs by consumer:
   - `offload(..., cpu_read=False)` for CUDA CPU-source kernel consumers
   - `mark_ready_for_cpu_read(handle)` or `wait_for_cpu_read(handle)` before
     CPU elementwise code

   First implementation can conservatively synchronize before CPU reads. Later
   stages can replace this with event polling or stream/event waits if timing
   requires it.

3. Make live memory accounting explicit:
   - live CPU bytes by tag
   - peak live CPU bytes by tag
   - live CUDA staged bytes by tag
   - peak live staged bytes by tag
   - stage-cache live/cached bytes
   - D2H bytes by tag
   - H2D staged bytes by tag
   - CPU-read wait count and waited bytes by tag
   - in-flight async source/stage bytes if async is enabled

4. Extend `AsymExecutionStats` with expact counters:
   - `expact_base_cpu_source_forward_calls`
   - `expact_base_cpu_source_dx_calls`
   - `expact_lora_a_pair_forward_grouped_calls`
   - `expact_lora_b_pair_backward_grouped_calls`
   - `expact_small_gemm_fallback_count`
   - `expact_cpu_source_kernel_bytes`
   - `expact_stage_full_activation_bytes`

5. Export these counters:
   - `LFAsymReport.runtime_log_string` prints the new fields.
   - `run_lf_profiled_train.py` writes an `asym_execution_stats` object in the
     source profile JSON.
   - `scripts/testing/profile_qwen3_activation_offload.py` already writes
     `stats`; keep that path compatible with new fields.

6. Improve peak-memory attribution when existing output is insufficient:
   - add expact phase rows to the source profile if peak ownership is unclear
   - label saved tensors and live activations from expert offload separately
     from loss, attention, router, and dense MLP
   - expose allocator peak-growth owner, current live CPU bytes, current live
     CUDA stage bytes, and in-flight async bytes at each expact phase
   - include enough profile JSON fields to tell whether the peak comes from
     expert offload, loss/cross-entropy, optimizer, padding/unpadding, or
     another component

7. Tighten lifetime in current expact without changing math:
   - Add `del` after last use of wide CUDA tensors.
   - Always `release_stage(..., drop_cache=True)` before entering the next
     high-memory region.
   - Release CPU handles once their last dependent gradient is produced.
   - Snapshot before and after each expact block.
   - Add `prof_range` names around every wide region and CPU wait.

8. Add CPU Adam assertions to small integration tests:
   - LoRA compute params remain CUDA.
   - CPU masters and optimizer state remain CPU.
   - `last_step_grad_param_count == last_step_copyback_param_count` after an
     expact backward + optimizer step when all LoRA params receive gradients.

### Risks And Watch Items

- Pinned D2H plus CPU read ordering is easy to get wrong. Stage 0 should prefer
  a conservative synchronization over silent races.
- A global sync before CPU SiLU will hurt timing, but it should not change HBM
  correctness. Later stages can reduce sync scope.
- Async copies can increase peak HBM if staged buffers overlap too aggressively.
  The plan does not allow async prefetch until the live-byte counters prove it
  is bounded.
- The current source profile may not have a direct handle to `LFAsymReport`.
  If so, implement model introspection in `run_lf_profiled_train.py` to collect
  `module.stats.as_dict()` from wrapped Asym modules.

### Validation Before Stage 1

Static audit:

```bash
rg -n "ctx\\.|manager.stage|stage_concat_columns|release_stage|release_cpu|offload\\(|wait_for_cpu|ready_for_cpu" \
  asym_gemm/training/qwen3_moe.py \
  asym_gemm/training/activation_offload.py
```

Unit correctness:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend \
  tests/training/test_asym_cpu_adamw.py \
  tests/lf/test_asym_cpu_adamw_lf_integration.py \
  tests/lf/test_asym_cpu_adamw_args.py
```

Small profile:

```bash
mkdir -p outputs
PYTHONPATH="$PWD" .venv/bin/python scripts/testing/profile_qwen3_activation_offload.py \
  --tokens 1024 \
  --top-k 2 \
  --num-experts 8 \
  --hidden-dim 4096 \
  --intermediate-dim 11008 \
  --rank 8 \
  --warmup 1 \
  --iters 2 \
  --output-json outputs/expact_v2_stage0_small.json
```

LF dry-run validation of labels and CPU Adam flags:

```bash
DRY_RUN=true \
OUTPUT_ROOT="$PWD/outputs/expact_v2_stage0_dryrun" \
GPU_POOL=0 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false \
ASYM_OFFLOAD_MODULES=all \
SEQ_LENS=8192 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PLOT=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus 0
```

Stage 0 passes only if:

- no CPU-side activation math reads an offloaded handle before explicit
  completion
- source profile or small profile exposes expact stats and AsymGEMM/GEMM counts
- CPU Adam summaries report `all_masters_on_cpu=true` and
  `all_cuda_params_on_cuda=true`
- peak HBM is not worse than the previous expact profile except for explicitly
  tagged measurement overhead

## Stage 1: Remove Full HBM Base Activation Restaging

### Scope

- `asym_gemm/training/qwen3_moe.py`
  - `_ActivationOffloadQwen3ExpertFunction.forward`
  - `_ActivationOffloadQwen3ExpertFunction.backward`
- New or extended Python wrapper:
  - prefer new `asym_gemm/training/exp_act_offload_base.py`
  - update `asym_gemm/training/__init__.py` if public imports are needed
- Existing abstractions:
  - `asym_gemm/training/host_weight.py::HostWeight`
  - `asym_gemm/training/frozen_linear.py::AsymGroupedFrozenLinear`
- Native binding:
  - `csrc/apis/exp_act_offload.hpp`
  - `csrc/exp_act_offload/exp_act_offload_kernels.cu`
  - `csrc/python_api.cpp` only if a new namespace/register path is added
  - `setup.py` only if a new `.cu` file is added instead of extending
    `exp_act_offload_kernels.cu`
- Tests:
  - new `tests/training/test_exp_act_offload_base_cpu_source.py`
  - update `tests/training/test_lf_qwen3_asym_backend.py`

### Code Changes

1. Add grouped CPU-source base forward:

```python
output = grouped_down_base_cpu_source(
    act_cpu.tensor,
    layer.down_base.host_weight.tensor,
    offsets,
    experts,
    metadata=lora_metadata,
    stats=layer.stats,
)
```

Contract:

```text
act_cpu: pinned CPU BF16 [M,I]
down_base_weight: pinned CPU BF16 HostWeight [E,H,I]
offsets/experts: grouped route metadata
return: CUDA BF16 [M,H]
```

2. Add grouped CPU-source gate/up base dX:

```python
grad_packed = grouped_gate_up_base_dx_cpu_source(
    grad_gate_cpu.tensor,
    grad_up_cpu.tensor,
    layer.gate_up_base.host_weight.tensor,
    offsets,
    experts,
    metadata=lora_metadata,
    stats=layer.stats,
)
```

Contract:

```text
dgate_cpu: pinned CPU BF16 [M,I]
dup_cpu: pinned CPU BF16 [M,I]
gate_up_base_weight: pinned CPU BF16 HostWeight [E,2I,H]
offsets/experts: grouped route metadata
return: CUDA BF16 [M,H]
```

3. Native scheduling requirements:
   - one grouped kernel per projection, not one launch per expert
   - consume CPU activation tiles directly
   - consume CPU `HostWeight` tiles directly or via bounded tile scratch
   - write only the final `[M,H]` output/grad to HBM
   - no full CUDA `[M,I]` or `[M,2I]` source tensor
   - increment `expact_base_cpu_source_forward_calls` and
     `expact_base_cpu_source_dx_calls`

4. Remove from expact:
   - `manager.stage(act_cpu, tag="act_for_down_base")`
   - `manager.stage_concat_columns(..., tag="dgate_up_for_gate_up_base")`
   - `manager.release_stage(grad_gate_up, ...)`

5. Update the Qwen3 unit test:
   - assert `stage_peak_by_tag` does not contain `act_for_down_base`
   - assert `stage_peak_by_tag` does not contain `dgate_up_for_gate_up_base`
   - assert new base CPU-source counters are positive

### Risks And Watch Items

- This is the biggest uncertainty: the kernel has CPU activations and CPU base
  weights. If both are fetched over PCIe/NVLink host mapping, bandwidth may
  dominate. Correctness and HBM reduction still come first.
- `HostWeight.metadata.can_map_host_memory` may be false or unknown on some
  systems. The kernel must fail closed rather than silently staging full HBM.
- Route metadata uses cumulative offsets with a sentinel expert entry. Tests
  must cover empty groups, repeated experts, dense experts, and uneven row
  counts.
- If native code needs a new `.cu` file, update `setup.py`; otherwise editable
  builds will not compile it.

### Validation Before Stage 2

Native correctness:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_exp_act_offload_base_cpu_source.py
```

End-to-end unit:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend
```

Static no-restage audit:

```bash
rg -n "act_for_down_base|dgate_up_for_gate_up_base|stage_concat_columns|manager.stage\\(" \
  asym_gemm/training/qwen3_moe.py \
  asym_gemm/training/activation_offload.py
```

Small profile:

```bash
mkdir -p outputs
PYTHONPATH="$PWD" .venv/bin/python scripts/testing/profile_qwen3_activation_offload.py \
  --tokens 1024 \
  --top-k 2 \
  --num-experts 8 \
  --hidden-dim 4096 \
  --intermediate-dim 11008 \
  --rank 8 \
  --warmup 1 \
  --iters 2 \
  --output-json outputs/expact_v2_stage1_small.json
```

Stage 1 passes only if:

- both old full-stage tags are absent
- base CPU-source counters are positive
- HBM peak drops relative to Stage 0
- trace shows grouped base CPU-source launches, not one launch per expert

## Stage 2: Replace LoRA-B Backward With Tiled CPU-Source Kernels

### Scope

- `asym_gemm/training/qwen3_moe.py`
  - `_ActivationOffloadQwen3ExpertFunction.backward`
  - remove expact dependence on `_grouped_lora_weight_grads_torch`
  - remove expact dependence on `_grouped_lora_cuda_view`
- `asym_gemm/training/exp_act_offload_lora.py`
  - `grouped_lora_b_backward_cpu_source`
  - add paired wrapper for gate/up
  - `require_expert_activation_offload_kernels`
- `asym_gemm/training/frozen_linear.py`
  - `AsymExecutionStats`
- Native:
  - `csrc/apis/exp_act_offload.hpp`
  - `csrc/exp_act_offload/exp_act_offload_kernels.cu`
- Tests:
  - update `tests/training/test_exp_act_offload_native.py`
  - new `tests/training/test_exp_act_offload_lora_b_backward_tiled.py`

### Code Changes

1. Replace the current native LoRA-B implementation with a tiled version behind
   a stable Python wrapper if possible:

```python
dS_down, grad_down_lora_B = grouped_lora_b_backward_cpu_source(
    grad_output_cpu_or_cuda,
    ctx.down_low_rank_cpu.tensor,
    down_lora_B,
    offsets,
    experts,
    scale=layer.lora_scale,
    stats=layer.stats,
    tag="down",
)
```

2. Add a paired gate/up wrapper:

```python
dS_gate, dS_up, grad_gate_lora_B, grad_up_lora_B = (
    grouped_lora_b_pair_backward_cpu_source(
        grad_gate_cpu.tensor,
        grad_up_cpu.tensor,
        ctx.gate_low_rank_cpu.tensor,
        ctx.up_low_rank_cpu.tensor,
        gate_lora_B,
        up_lora_B,
        offsets,
        experts,
        scale=layer.lora_scale,
        stats=layer.stats,
    )
)
```

3. Native scheduling requirements:
   - pair gate/up so route metadata and CPU gradient tiles are loaded once
   - use CPU `grad_gate_cpu` and `grad_up_cpu` directly
   - write final `dS_*` and `grad_*_lora_B` without full FP32 `dS_acc` or
     `grad_b_acc`
   - tile accumulators are allowed; full-output FP32 workspaces are not
   - low-rank CPU input is preferred; low-rank CUDA staging is allowed only if
     Stage 0 counters show it does not move peak HBM
   - increment single and paired LoRA-B counters

4. Keep CPU Adam compatible:
   - returned gradients for `gate_lora_B`, `up_lora_B`, and `down_lora_B` must
     be normal CUDA tensors with the same dtype/shape contract as current
     backward
   - do not write directly to `AsymCPUAdamW` CPU masters
   - do not move LoRA params to CPU to make the kernel simpler

### Risks And Watch Items

- Numerical tolerance may change because tiled reductions reorder
  accumulation. Define tolerances in tests against the current torch reference.
- Down LoRA-B consumes `grad_output`, which already exists in HBM. It may not
  need CPU offload. Gate/up must not use full HBM gradients.
- If `dS` is produced in BF16 for memory, LoRA-A grad accuracy must be checked.
  If FP32 `dS` is required, it must be final-sized by mathematical necessity
  and counted as such.

### Validation Before Stage 3

Native correctness:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_exp_act_offload_lora_b_backward_tiled.py \
  tests/training/test_exp_act_offload_native.py::test_grouped_lora_b_backward_cpu_source_matches_reference
```

Qwen expact correctness:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend
```

Static audit:

```bash
rg -n "_grouped_lora_weight_grads_torch|_grouped_lora_cuda_view|dS_acc|grad_b_acc|stage_low_rank" \
  asym_gemm/training/qwen3_moe.py \
  asym_gemm/training/exp_act_offload_lora.py \
  csrc/exp_act_offload
```

CPU Adam one-step checks:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_asym_cpu_adamw.py \
  tests/lf/test_asym_cpu_adamw_lf_integration.py
```

Stage 2 passes only if:

- expact backward no longer calls `_grouped_lora_weight_grads_torch` for LoRA-B
- expact backward no longer stages full gate/up gradients for LoRA-B
- no full FP32 `dS_acc` or `grad_b_acc` allocation remains in the CPU-source
  LoRA-B path
- CPU Adam sees normal LoRA gradients and updates all LoRA params that had grads

## Stage 3: Tile LoRA-A Gradient Reductions

### Scope

- `asym_gemm/training/exp_act_offload_lora.py`
  - `grouped_lora_a_grad_cpu_right`
  - `grouped_lora_a_pair_grad_cpu_right`
- Native:
  - `csrc/apis/exp_act_offload.hpp`
  - `csrc/exp_act_offload/exp_act_offload_kernels.cu`
- `asym_gemm/training/qwen3_moe.py`
  - no callsite changes if wrapper names stay stable
- Tests:
  - update `tests/training/test_exp_act_offload_native.py`
  - new `tests/training/test_exp_act_offload_lora_a_grad_tiled.py`

### Code Changes

1. Replace full-accumulator native LoRA-A grad internals:

```text
sm100_grouped_lora_a_grad_bf16_cpu_right
sm100_grouped_lora_a_pair_grad_bf16_cpu_right
```

The wrapper names can stay the same, but implementation must not allocate:

```text
grad_acc
grad_gate_acc
grad_up_acc
```

2. Tiled contract:

```text
dS_cuda: [M,r]
X_cpu: pinned CPU BF16 [M,K]
grad_A_cuda: [E,r,K]
offsets/experts: grouped route metadata
```

3. Pair policy:
   - one paired gate/up kernel consumes `dS_gate`, `dS_up`, and `X_cpu`
   - the same `X_cpu` tile is reused for both gate and up reductions
   - output two final gradient tiles
   - increment paired LoRA-A grad counters separately from single calls

4. Reduction policy:
   - accumulate in registers/shared memory or tile scratch
   - write the final tile to `grad_A_cuda`
   - no one-kernel-per-expert fallback in normal path

### Risks And Watch Items

- Atomic accumulation may be nondeterministic. Tests should use tolerant
  numerical comparison and fixed seeds.
- Rank and hidden dimensions may not always be multiples of the ideal tile.
  Keep existing fail-closed alignment checks unless the kernel supports tails.
- Full final `grad_A` is allowed because it is the real parameter gradient. Full
  extra FP32 grad workspaces are not allowed.

### Validation Before Stage 4

Native correctness:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_exp_act_offload_lora_a_grad_tiled.py \
  tests/training/test_exp_act_offload_native.py::test_grouped_lora_a_grad_cpu_right_matches_reference \
  tests/training/test_exp_act_offload_native.py::test_grouped_lora_a_pair_grad_cpu_right_matches_reference
```

Static audit:

```bash
rg -n "grad_acc|grad_gate_acc|grad_up_acc|torch::zeros" \
  csrc/exp_act_offload
```

Small profile:

```bash
mkdir -p outputs
PYTHONPATH="$PWD" .venv/bin/python scripts/testing/profile_qwen3_activation_offload.py \
  --tokens 1024 \
  --top-k 2 \
  --num-experts 8 \
  --hidden-dim 4096 \
  --intermediate-dim 11008 \
  --rank 8 \
  --warmup 1 \
  --iters 2 \
  --output-json outputs/expact_v2_stage3_small.json
```

Stage 3 passes only if:

- no full FP32 LoRA-A accumulator remains in the expact native path
- grouped kernel counts remain bounded and do not scale with expert count
- peak HBM and backward timing improve or the profile identifies another
  dominant bottleneck

## Stage 4: True Paired Gate/Up Scheduling

### Scope

- `asym_gemm/training/exp_act_offload_lora.py`
  - `grouped_lora_a_pair_forward_cpu_left`
  - paired backward wrappers added in Stage 2 and Stage 3
- `asym_gemm/training/qwen3_moe.py`
  - `_ActivationOffloadQwen3ExpertFunction.forward`
  - `_ActivationOffloadQwen3ExpertFunction.backward`
- `asym_gemm/training/lora.py`
  - `grouped_expert_lora_pair`
  - metadata reuse and `torch.cat` accounting
- Native:
  - `csrc/apis/exp_act_offload.hpp`
  - `csrc/exp_act_offload/exp_act_offload_kernels.cu`
- Tests:
  - update `tests/training/test_cpu_left_lora.py`
  - new `tests/training/test_exp_act_offload_gate_up_paired.py`

### Code Changes

1. Replace the current pair-forward wrapper that calls the single CPU-left
   kernel twice with a real paired CPU-left path:

```python
gate_low_rank, up_low_rank = grouped_lora_a_pair_forward_cpu_left(
    x_cpu.tensor,
    gate_lora_A,
    up_lora_A,
    offsets,
    experts,
    metadata=lora_metadata,
    stats=layer.stats,
    tag="gate_up",
)
```

2. Native forward schedule:
   - load one `X_cpu` tile
   - compute gate and up low-rank tiles before advancing the source tile
   - one grouped paired call, not two independent source passes

3. Backward schedule:
   - use paired LoRA-B from Stage 2
   - use paired LoRA-A grad from Stage 3
   - if `need_grad_packed`, compute gate/up LoRA `dX` through grouped calls and
     add them into the single final `grad_packed`
   - do not recreate `stage_concat_columns`

4. Track allocations:
   - count paired calls distinctly from two singles
   - record CPU source bytes loaded for paired gate/up
   - tag any `torch.cat` in `grouped_expert_lora_pair`

### Risks And Watch Items

- Pairing is mostly a timing and bandwidth fix, but it can reduce HBM by
  shortening overlapping low-rank/delta lifetimes. Do not make correctness
  harder unless Stage 2/3 profiles show repeated CPU source reads are material.
- `grouped_expert_lora_pair` currently uses concatenation for CUDA low-rank
  inputs and weights. That may be acceptable for low-rank tensors, but it must
  be counted and revisited if it moves peak.

### Validation Before Stage 5

Correctness:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_exp_act_offload_gate_up_paired.py \
  tests/training/test_cpu_left_lora.py \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend
```

Static audit:

```bash
rg -n "grouped_lora_a_pair_forward_cpu_left|stage_concat_columns|torch.cat" \
  asym_gemm/training/exp_act_offload_lora.py \
  asym_gemm/training/qwen3_moe.py \
  asym_gemm/training/lora.py
```

Stage 4 passes only if:

- stats show one paired gate/up LoRA-A forward call, not two single calls
- no per-expert GEMM launch pattern appears in source/NSYS stats
- paired CPU source bytes are lower than two independent passes
- Qwen3 expact remains numerically matched to the torch backend

## Stage 5: Hidden Materialization And Workspace Cleanup

### Scope

- `asym_gemm/training/activation_offload.py`
  - add optional bounded workspace/pool policy
  - global `_CPU_BUFFER_POOL`
- `asym_gemm/training/frozen_linear.py`
  - `_pad_grouped_input_for_asym`
  - `_unpad_grouped_output`
  - `_dispatch_grouped_nt`
- `asym_gemm/training/cpu_left.py`
  - `_pad_cpu_left_grouped_input_for_asym`
  - `_unpad_grouped_output`
  - `grouped_expert_lora_cpu_left`
- `asym_gemm/training/lora.py`
  - `prepare_grouped_lora_metadata`
  - `grouped_expert_lora_pair`
- `asym_gemm/training/qwen3_moe.py`
  - wide tensor lifetimes in expact forward/backward
- `asym_gemm/profiling/lf_trace.py`
  - expose allocation tags if source profile lacks enough detail
- Tests:
  - update focused allocation-count tests in `tests/training/test_cpu_left_lora.py`
  - update `tests/training/test_cpu_resident_frozen_base.py`
  - add expact allocation regression checks in
    `tests/training/test_lf_qwen3_asym_backend.py`

### Code Changes

1. Add bounded workspace ownership if profiling shows pool/cached buffers are
   moving peak:

```text
ActivationOffloadWorkspace
  get_cuda_scratch(tag, shape_or_tile, dtype)
  get_cpu_scratch(tag, shape_or_tile, dtype, pinned=True)
  release_cuda_scratch(tag)
  release_cpu_scratch(tag)
  snapshot_live_bytes()
```

2. Tag and shorten materializations:
   - `_unpad_grouped_output` padded and unpadded overlap
   - CPU-left padded source buffers
   - `torch.cat` in `grouped_expert_lora_pair`
   - wide `.contiguous()` on routed tensors
   - active expert `index_select`
   - LoRA deltas `gate_delta`, `up_delta`, `down_delta`

3. Cleanup rules:
   - scratch is tile-sized or low-rank-sized unless explicitly justified
   - scratch has profile-visible tags
   - CPU pinned cache has a max cached-byte policy or explicit clear hook
   - final output allocation is allowed
   - duplicate padded/unpadded output overlap is allowed only for the shortest
     possible scope

4. If forward peak is still high after Stage 1-4, add add-into-output variants:
   - LoRA-B delta directly into base output where safe
   - gate/up delta addition immediately followed by CPU offload and delete
   - do not start direct-to-CPU base gate/up output unless profiles show this is
     now the remaining dominant peak

### Risks And Watch Items

- Some padding/unpadding is required by existing AsymGEMM kernels. Removing it
  without changing kernel contracts can break correctness.
- CPU pinned memory pressure can become the next bottleneck. Track CPU live and
  cached bytes separately from HBM.
- Async double buffering should be enabled only after counters prove max
  in-flight HBM remains below the synchronous path.

### Validation Before Stage 6

Static audit:

```bash
rg -n "act_for_down_base|dgate_up_for_gate_up_base|grad_acc|grad_gate_acc|grad_up_acc|dS_acc|grad_b_acc|stage_concat_columns|_grouped_lora_weight_grads_torch|torch.cat|contiguous\\(|index_select|_unpad_grouped_output" \
  asym_gemm/training/qwen3_moe.py \
  asym_gemm/training/activation_offload.py \
  asym_gemm/training/exp_act_offload_lora.py \
  asym_gemm/training/cpu_left.py \
  asym_gemm/training/frozen_linear.py \
  asym_gemm/training/lora.py \
  csrc/exp_act_offload
```

Unit allocation regressions:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_cpu_left_lora.py \
  tests/training/test_cpu_resident_frozen_base.py \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend
```

Memory-breakdown profile:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v2_stage5_cleanup_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false \
ASYM_OFFLOAD_MODULES=all \
SEQ_LENS=8192 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_MEMORY_BREAKDOWN_MODULES=experts,lora,loss \
PLOT=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus 3
```

Stage 5 passes only if:

- no untagged full-width materialization remains in expact hot regions
- `max_stage_bytes_live` is bounded and does not include full activation stages
- source profile distinguishes expert-block peak from loss/cross-entropy peak
- grouped call counts remain bounded

## Stage 6: Final Fair Comparison And CPU Adam Acceptance

### Scope

- `/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh`
  - only if output naming or counter propagation is missing
- `scripts/lf/run_lf_lora_sft.sh`
  - only if `asym_cpuadamwds` profile logs omit needed CPU Adam markers
- `scripts/lf/run_lf_profiled_train.py`
  - ensure final profile includes `asym_execution_stats`,
    `activation_offload_stats`, and `asym_cpu_adamw`
- `scripts/lf/postprocess_lf_profile_artifacts.py`
  - only if CSV/plot artifacts need the new counters
- `scripts/lf/check_deepspeed_cpuadam_run.py`
  - no change expected; use manually when DeepSpeedCPUAdam markers are present

### Code Changes

1. Ensure the final profile JSON reports:
   - peak allocated/reserved HBM
   - expert-block peak separate from loss peak
   - `activation_offload_stats`
   - `asym_execution_stats`
   - CPU pinned live/cached bytes
   - D2H/H2D bytes by tag
   - CPU-read wait counts and bytes
   - grouped CPU-source kernel counts
   - base AsymGEMM forward/dX counts
   - LoRA-A/LoRA-B grouped counts
   - fallback reasons and small-GEMM fallback count
   - `asym_cpu_adamw` summary

2. CPU Adam acceptance:
   - `asym_cpu_adamw.enabled=true`
   - `all_masters_on_cpu=true`
   - `all_cuda_params_on_cuda=true`
   - `last_step_grad_param_count > 0`
   - `last_step_grad_param_count == last_step_copyback_param_count`
   - `backend=deepspeed` for the primary `asym_cpuadamwds` run
   - if the profile/log contains a DeepSpeedCPUAdam runtime marker, this manual
     checker must pass:

```bash
.venv/bin/python scripts/lf/check_deepspeed_cpuadam_run.py \
  --profile-json "$PROFILE_SOURCE_JSON" \
  --train-log "$TRAIN_LOG" \
  --require-enabled
```

### Risks And Watch Items

- `check_deepspeed_cpuadam_run.py` is primarily wired for `zero3_cpuadam`.
  For `asym_cpuadamwds`, the stronger required signal is the
  `asym_cpu_adamw` source-profile summary from `AsymCPUAdamW`.
- If `4x8192` still OOMs after Stage 5, record whether the failure is inside
  expert blocks or later loss/cross-entropy. The claim is not satisfied by a
  loss-side OOM explanation alone, but the next fix depends on where the peak
  moved.
- If async is introduced, compare sync vs async with identical kernels. Async
  must not increase peak HBM through excessive prefetch overlap.

### Final Validation

Canonical workflow comparison, batch 2:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v2_final_b2_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false \
ASYM_OFFLOAD_MODULES=all \
SEQ_LENS=8192 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PLOT=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus 3
```

Canonical workflow comparison, batch 4:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v2_final_b4_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false \
ASYM_OFFLOAD_MODULES=all \
SEQ_LENS=8192 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PLOT=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus 3
```

Global recompute lower bound:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v2_final_global_recompute_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|recomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES=none|false \
ASYM_OFFLOAD_MODULES=all \
SEQ_LENS=8192 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PLOT=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus 3
```

Optional NSYS kernel-count run after the source profile passes:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v2_final_nsys_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=nsys,source \
ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false \
ASYM_OFFLOAD_MODULES=all \
SEQ_LENS=8192 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PLOT=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus 3
```

Final acceptance requires:

- `polnone__expact1` peak allocated HBM is below `polgc-exp__expact0` in the
  canonical workflow comparison
- `2x8192` no-recompute fits with clear HBM margin
- `4x8192` no-recompute fits, or fails later for a specifically identified
  non-expert peak after expert-block HBM is below recompute
- no full activation stage tags remain
- no full FP32 expact accumulator workspaces remain
- no one-launch-per-expert GEMM pattern appears in source/NSYS traces
- CPU Adam summary proves CPU masters/state and CUDA LoRA compute params are in
  the intended places
- timing regression is explained by useful transfer/compute work, not long
  CPU-idle or GPU-idle gaps caused by ungrouped scheduling
