# Expert Activation Offload V2

This plan records the next implementation work for
`ASYMM_EXPERT_ACT_OFFLOAD=true` after the first end-to-end profiles. It focuses
only on the expert-activation memory issues that still exist inside the current
Qwen3 path:

- remove full wide HBM staging
- replace full FP32 LoRA-A gradient workspaces with tiled grouped reductions
- fuse gate/up backward so shared CPU source tiles are reused
- fuse LoRA-B backward / `dS` production where possible
- clean up allocator, padding, and hidden materialization hotspots

The goal is peak-HBM reduction without turning routed expert work into many
small GEMMs. Normal training must use grouped kernels or bounded grouped
schedules. Reference loops may exist only in tests.

Current target files:

- `asym_gemm/training/qwen3_moe.py`
- `asym_gemm/training/activation_offload.py`
- `asym_gemm/training/exp_act_offload_lora.py`
- `asym_gemm/training/frozen_linear.py`
- `csrc/exp_act_offload/exp_act_offload_kernels.cu`
- native binding registration files for `asym_gemm`
- focused tests under `tests/` or `scripts/`

Current profile evidence:

- Qwen3 `4x8192`, recompute on:
  - `expact0`: peak allocated `73.546 GB`, measured step `14.708 s`
  - `expact1`: peak allocated `73.546 GB`, measured step `107.291 s`
- Qwen3 `4x8192`, recompute off:
  - `expact0`: OOM in first forward, peak allocated about `196.158 GB`
  - `expact1`: OOM in first forward, peak allocated about `195.853 GB`
- Qwen3 `2x8192`, recompute off:
  - `expact0`: OOM in first forward/loss, peak allocated `193.732 GB`;
    the failing allocation was cross-entropy trying to allocate about `9.27 GiB`
  - `expact1`: completed; peak allocated `135.628 GB`, peak reserved
    `141.503 GB`, measured step `44.950 s`, forward `10.578 s`,
    backward `32.939 s`

This means current activation offload is not yet the intended design. It
offloads enough tensors to make smaller no-recompute cases fit, but it then
recreates large HBM buffers and expensive workspaces. The next design must keep
that memory benefit while replacing the slow/staging-heavy backward with grouped
CPU-source kernels.

## Required Invariants

Normal `ASYMM_EXPERT_ACT_OFFLOAD=true` must satisfy these invariants:

- no Python loop or C++ host loop that launches one GEMM per expert
- no full HBM staging of `X_cpu` for LoRA-A
- no full HBM staging of `act_cpu` for LoRA-A
- no extra full HBM concat solely to make LoRA-B or base `dX` convenient
- no full CUDA FP32 accumulator tensor shaped like a LoRA-A gradient
- no hidden wide `.contiguous()`, `torch.cat`, `index_select`, or unpad output
  allocation in hot expert activation-offload kernels
- CPU source tiles loaded by a kernel must be reused across the relevant output
  tiles before the next CPU tile is loaded
- every routed matrix operation is one grouped call or one native grouped
  kernel launch per projection/pair, not one launch per expert group

Allowed materialization:

- low-rank `[M,r]` tensors when they are the real output of LoRA-A
- final gradients with their true shapes
- bounded scratch buffers whose size is tied to tile/block shape, not full
  activation shape
- temporary wide HBM only if it is proven unavoidable for an existing base
  kernel stage and is removed in the corresponding stage below

## Stage 1: Remove Wide Activation Staging

Most promising memory fix. Current code stages:

- `act_cpu` back to HBM as `act_for_down_base` in
  `_ActivationOffloadQwen3ExpertFunction.forward`
- `[dgate_cpu, dup_cpu]` as `dgate_up_for_gate_up_base` in backward

These recreate the exact wide tensors activation offload is meant to avoid.

### Implement

Add native CPU-left grouped base calls for the wide base paths:

```text
grouped_down_base_cpu_left(
    act_cpu: pinned CPU BF16 [M,I],
    down_base_weight: CPU-resident expert base [E,H,I] or existing HostWeight,
    offsets,
    experts,
) -> CUDA BF16 [M,H]

grouped_gate_up_base_dx_cpu_source(
    dgate_cpu: pinned CPU BF16 [M,I],
    dup_cpu: pinned CPU BF16 [M,I],
    gate_up_base_weight: CPU-resident expert base [E,2I,H] or existing HostWeight,
    offsets,
    experts,
) -> CUDA BF16 [M,H]
```

Implementation policy:

- put Python wrappers in `asym_gemm/training/exp_act_offload_lora.py` or a new
  `asym_gemm/training/exp_act_offload_base.py`
- bind native kernels from `csrc/exp_act_offload/exp_act_offload_kernels.cu` or
  a new `csrc/exp_act_offload/exp_act_offload_base_kernels.cu`
- reuse normalized grouped offsets from existing AsymGEMM metadata helpers
- tile around the CPU operand: load one CPU source tile, reuse it across the
  output tile work, then advance
- do not call `manager.stage(...)` for `act_for_down_base`
- do not call `manager.stage_concat_columns(...)` for `dgate_up_for_gate_up_base`
  after this stage lands

Qwen wiring:

```python
# forward down base
output = grouped_down_base_cpu_left(
    act_cpu.tensor,
    layer.down_base.host_weight,
    offsets,
    experts,
    metadata=lora_metadata,
)

# backward gate/up base dX
grad_packed = grouped_gate_up_base_dx_cpu_source(
    grad_gate_cpu.tensor,
    grad_up_cpu.tensor,
    layer.gate_up_base.host_weight,
    offsets,
    experts,
    metadata=lora_metadata,
)
```

Remove or make unreachable in the expact path:

- `manager.stage(act_cpu, tag="act_for_down_base")`
- `manager.stage_concat_columns(grad_gate_cpu, grad_up_cpu, tag="dgate_up_for_gate_up_base")`

### Validate

Kernel correctness:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/test_exp_act_offload_base_cpu_source.py
```

Kernel profiling:

```bash
ncu --target-processes all --set full \
  -o /tmp/expact_base_cpu_source_ncu \
  .venv/bin/python scripts/profile_exp_act_offload_base_cpu_source.py \
  --model qwen3 --batch 2 --seq 8192 --dtype bf16
```

End-to-end:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v2_stage1_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|recomp" \
PROFILERS=source \
ASYMM_EXPERT_ACT_OFFLOAD=true,false \
ASYM_OFFLOAD_MODULES=all \
SEQ_LENS=8192 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PLOT=false \
scripts/lf/profile_lora_lf.sh --gpus 3
```

Check:

- no `act_for_down_base` stage tag in expact stats
- no `dgate_up_for_gate_up_base` stage tag in expact stats
- peak HBM moves in the expected direction
- timing is not dominated by idle CPU/offload time

## Stage 2: Tiled Grouped LoRA-A Grad Reductions

Current native LoRA-A grad kernels allocate full FP32 CUDA workspaces:

- `grad_acc`
- `grad_gate_acc`
- `grad_up_acc`

Those workspaces are shaped like the final LoRA-A gradients and are live during
backward. Replace them with tiled reductions.

### Implement

Replace:

```text
sm100_grouped_lora_a_grad_bf16_cpu_right
sm100_grouped_lora_a_pair_grad_bf16_cpu_right
```

with tiled implementations:

```text
sm100_grouped_lora_a_grad_bf16_cpu_right_tiled
sm100_grouped_lora_a_pair_grad_bf16_cpu_right_tiled
```

Kernel contract:

```text
dS_cuda      [M,r] BF16
X_cpu        [M,K] pinned BF16
grad_A_cuda  [E,r,K] BF16 or FP32 destination
offsets/expert metadata
```

Tiling policy:

- CTA covers `(group, r_tile, k_tile)`
- loop over rows belonging to that group
- load `X_cpu[row, k_tile]` once and reuse across all `r_tile` accumulators
- accumulate in registers/shared memory
- write final tile directly to `grad_A_cuda`
- no full `torch::zeros(... float32)` accumulator tensor

Gate/up pair policy:

- single pair kernel consumes `dS_gate`, `dS_up`, and `X_cpu`
- the same `X_cpu` tile is reused for both gate and up reductions
- output two final gradient tiles

Python wiring:

- keep function names `grouped_lora_a_grad_cpu_right` and
  `grouped_lora_a_pair_grad_cpu_right`
- update the native symbol they call
- remove allocation of full FP32 temporary tensors from the native wrappers

### Validate

Correctness:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/test_exp_act_offload_lora_a_grad_tiled.py
```

NCU:

```bash
ncu --target-processes all --set full \
  -o /tmp/expact_lora_a_grad_tiled_ncu \
  .venv/bin/python scripts/profile_exp_act_lora_a_grad_tiled.py \
  --m 16384 --hidden 2048 --intermediate 768 --experts 128 --topk 8 --rank 64
```

End-to-end:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v2_stage2_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|recomp" \
PROFILERS=source \
ASYMM_EXPERT_ACT_OFFLOAD=true,false \
ASYM_OFFLOAD_MODULES=all \
SEQ_LENS=8192 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PLOT=false \
scripts/lf/profile_lora_lf.sh --gpus 3
```

Check:

- no full FP32 `grad_acc` allocations remain in C++ source
- NCU shows one grouped kernel per LoRA-A projection/pair, not many tiny GEMMs
- peak HBM and backward time improve relative to the previous stage

## Stage 3: Fused Gate/Up Backward

Gate and up share `X_cpu`, route metadata, and the same token rows. Backward
must exploit that instead of creating independent wide stages or independent
passes over CPU source data.

### Implement

Add:

```text
grouped_gate_up_lora_backward_cpu_source(
    grad_gate_cpu,
    grad_up_cpu,
    S_gate_cpu,
    S_up_cpu,
    gate_lora_A,
    gate_lora_B,
    up_lora_A,
    up_lora_B,
    X_cpu,
    offsets,
    experts,
) -> dS_gate_cuda, dS_up_cuda, dB_gate_cuda, dB_up_cuda,
     dX_lora_gate_cuda, dX_lora_up_cuda, dA_gate_cuda, dA_up_cuda
```

This may be split into two native kernels if needed, but the schedule must stay
grouped and must not stage wide `dgate/dup` just to feed PyTorch helpers.

Required behavior:

- consume `grad_gate_cpu` and `grad_up_cpu` as CPU operands
- consume `X_cpu` once for paired `dA_gate/dA_up`
- reuse grouped route metadata
- produce final CUDA outputs directly
- no `stage_concat_columns`
- no separate pass that reloads `X_cpu` for gate and then reloads it for up

Qwen wiring:

```python
(
    dS_gate,
    dS_up,
    grad_gate_lora_B,
    grad_up_lora_B,
    grad_gate_lora_x,
    grad_up_lora_x,
    grad_gate_lora_A,
    grad_up_lora_A,
) = grouped_gate_up_lora_backward_cpu_source(...)
```

### Validate

Correctness:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/test_exp_act_offload_gate_up_backward_fused.py
```

NCU:

```bash
ncu --target-processes all --set full \
  -o /tmp/expact_gate_up_backward_fused_ncu \
  .venv/bin/python scripts/profile_exp_act_gate_up_backward_fused.py \
  --m 16384 --hidden 2048 --intermediate 768 --experts 128 --topk 8 --rank 64
```

End-to-end:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v2_stage3_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|recomp" \
PROFILERS=source \
ASYMM_EXPERT_ACT_OFFLOAD=true,false \
ASYM_OFFLOAD_MODULES=all \
SEQ_LENS=8192 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PLOT=false \
scripts/lf/profile_lora_lf.sh --gpus 3
```

Check:

- one fused gate/up backward schedule is visible
- `stage_concat_columns` is not called by expact
- no per-expert LoRA-B or LoRA-A loops appear in traces

## Stage 4: Fused LoRA-B Backward And dS

Current expact uses `_grouped_lora_weight_grads_torch` and CUDA low-rank stages
for parts of LoRA-B backward. This still creates unnecessary intermediate
pressure and keeps PyTorch helper behavior in the hot path.

### Implement

Add native grouped kernels:

```text
grouped_lora_b_backward_cpu_source_tiled
grouped_lora_b_pair_backward_cpu_source_tiled
```

Contracts:

```text
single:
  dY_or_dproj_cpu [M,N] pinned BF16
  S_cuda or S_cpu [M,r]
  B_cuda          [E,N,r]
  -> dS_cuda [M,r], dB_cuda [E,N,r]

pair:
  dgate_cpu, dup_cpu, S_gate_cpu, S_up_cpu, B_gate, B_up
  -> dS_gate, dS_up, dB_gate, dB_up
```

If `S_*` is kept on CPU for memory reasons, implement CPU-source handling for
both operands with a bounded CUDA tile accumulator. If `S_*` remains CUDA, keep
it low-rank and ensure no wide tensor is staged to support it.

Scheduling policy:

- down LoRA-B may use HBM `dY` directly because `dY` already exists as the
  upstream gradient
- gate/up LoRA-B should consume `dgate_cpu` and `dup_cpu` without full HBM
  staging
- pair gate/up when it reduces CPU reloads or launch count
- write final `dB` directly

### Validate

Correctness:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/test_exp_act_offload_lora_b_backward_tiled.py
```

NCU:

```bash
ncu --target-processes all --set full \
  -o /tmp/expact_lora_b_backward_tiled_ncu \
  .venv/bin/python scripts/profile_exp_act_lora_b_backward_tiled.py \
  --m 16384 --hidden 2048 --intermediate 768 --experts 128 --topk 8 --rank 64
```

End-to-end:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v2_stage4_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|recomp" \
PROFILERS=source \
ASYMM_EXPERT_ACT_OFFLOAD=true,false \
ASYM_OFFLOAD_MODULES=all \
SEQ_LENS=8192 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PLOT=false \
scripts/lf/profile_lora_lf.sh --gpus 3
```

Check:

- `_grouped_lora_weight_grads_torch` is no longer called from
  `_ActivationOffloadQwen3ExpertFunction.backward`
- no wide CPU-to-HBM stage is introduced for LoRA-B
- kernel count remains grouped and stable

## Stage 5: Allocator And Workspace Cleanup

This is lower priority than removing wide staging and full grad workspaces, but
it is still required before considering the path finished.

### Implement

Audit and remove hot-path materializations:

- `frozen_linear.py::_unpad_grouped_output`
- padding/unpadding in grouped CPU-left wrappers
- `torch.cat` in route metadata or paired LoRA paths when it creates wide HBM
  tensors
- `.contiguous()` calls on wide expert tensors inside activation-offload
  forward/backward
- `index_select` and scatter temporaries that duplicate routed activations

Add bounded scratch ownership:

```text
ActivationOffloadWorkspace
  get_cuda_scratch(tag, shape_or_tile, dtype)
  release_cuda_scratch(tag)
  record_peak_live_bytes()
```

Rules:

- scratch buffers are reused across layers where lifetime permits
- scratch sizes are tile-sized or low-rank-sized unless explicitly justified
- every scratch allocation has a profile-visible tag
- final output allocation is allowed; duplicate padded and unpadded outputs live
  together only where unavoidable and only for the shortest possible scope

### Validate

Static audit:

```bash
rg -n "act_for_down_base|dgate_up_for_gate_up_base|grad_acc|grad_gate_acc|grad_up_acc|stage_concat_columns|_grouped_lora_weight_grads_torch|torch.cat|contiguous\\(" \
  asym_gemm/training/qwen3_moe.py \
  asym_gemm/training/activation_offload.py \
  asym_gemm/training/exp_act_offload_lora.py \
  asym_gemm/training/frozen_linear.py \
  csrc/exp_act_offload
```

Memory profile:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v2_stage5_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|recomp" \
PROFILERS=source \
ASYMM_EXPERT_ACT_OFFLOAD=true,false \
ASYM_OFFLOAD_MODULES=all \
SEQ_LENS=8192 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PROFILE_MEMORY_BREAKDOWN=true \
PLOT=false \
scripts/lf/profile_lora_lf.sh --gpus 3
```

NCU spot check:

```bash
ncu --target-processes all --set full \
  -o /tmp/expact_v2_stage5_ncu \
  .venv/bin/python scripts/profile_exp_act_end_to_end_kernel_mix.py \
  --batch 2 --seq 8192 --rank 64
```

Check:

- no duplicate wide padded/unpadded output is live longer than the kernel call
- scratch tags show bounded peak live bytes
- grouped kernel mix has no many-small-GEMM pattern
- end-to-end peak HBM improves without the backward becoming CPU-idle

## Final Acceptance

Run both `2x8192` and `4x8192` comparisons after all stages:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v2_final_b2_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|recomp" \
PROFILERS=source \
ASYMM_EXPERT_ACT_OFFLOAD=true,false \
ASYM_OFFLOAD_MODULES=all \
SEQ_LENS=8192 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PLOT=false \
scripts/lf/profile_lora_lf.sh --gpus 3
```

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v2_final_b4_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|recomp" \
PROFILERS=source \
ASYMM_EXPERT_ACT_OFFLOAD=true,false \
ASYM_OFFLOAD_MODULES=all \
SEQ_LENS=8192 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PLOT=false \
scripts/lf/profile_lora_lf.sh --gpus 3
```

Report:

- peak allocated HBM
- peak reserved HBM
- forward/backward/measured step time
- AsymGEMM runtime call counts
- expact stage tags and max live stage bytes
- NCU kernel mix for the new grouped kernels

The implementation is not complete until `expact1` reduces peak HBM versus
`expact0` on the same workload and does not show the current long CPU-idle
backward behavior.
