# Expert Activation Offload V4: Remaining Safe Math-Lifetime Targets

This document starts from the accepted V3 post-S4 state. V2 is historical and
must not be reintroduced. V3 keeps the full measurement record and the
post-S4/S8 attribution verdict. V4 is narrower: it lists only the remaining
math-lifetime/kernel candidates that have not already been proven bad.

Core goal: reduce peak HBM for the `none|true` expert activation-offload path,
or improve its latency while keeping peak HBM flat or lower. The mechanism must
remain expert activation offload plus grouped CPU-source AsymGEMM/native kernels.
Do not use unrelated fused stacks, optimizer changes, different recompute modes,
per-expert loops, or small-GEMM decomposition to make the comparison look good.

Accepted comparison workflow:

```text
scripts/lf/profile_lora_lf.sh --gpus 0
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1"
BACKEND_SPECS="asym_cpuadamwds|norecomp"
ASYMM_EXP_ACT_POLICIES="none|true|false,gc-exp|false|false,none|false|false"
SEQ_LENS=4096
PER_DEVICE_TRAIN_BATCH_SIZE=4
DATASET=asym_long_sft_smoke
PREPARE_DATASETS=true
DATASET_MIN_TOKENS=4096
DATASET_OVERWRITE=true
LORA_DROPOUT=0.00
PROFILERS=source
PROFILE_MEMORY_ATTRIBUTION=false
PROFILE_MEMORY_BREAKDOWN=false
PROFILE_MEMORY_SNAPSHOT=false
WARMUP_STEPS=5
MAX_STEPS=10
RUN_POST=false
```

Accepted post-S4 baseline:

| policy | implementation | peak allocated HBM | peak reserved HBM | avg step | avg forward | avg backward |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `none|true` | activation offload target | `102.312 GiB` | `107.469 GiB` | `45.173 s` | `10.036 s` | `35.137 s` |
| `gc-exp|false` | expert checkpoint baseline | `126.312 GiB` | `131.414 GiB` | `3.953 s` | `1.570 s` | `2.383 s` |
| `none|false` | no expact/no expert recompute | `170.503 GiB` | `183.055 GiB` | `3.129 s` | `1.562 s` | `1.567 s` |

Current accepted call counts for `none|true`: `asym_forward_calls=5055`,
`asym_dx_calls=4290`, `torch_forward_calls=0`, `torch_dx_calls=0`,
`reference_fallback_count=0`.

The 1M-entry snapshot replay matched the source peak and showed that exact
routed-expert live blocks at the global peak are small. The earlier
`routed_experts:temporary_workspace:inferred_peak_workspace=5.795 GiB` row is
an inferred residual bucket, not exact expert ownership. Do not chase that row
as if it were a real expert tensor.

## Acceptance Rules

- Same peak HBM with worse latency is a regression.
- Tiny HBM savings with a large latency increase are rejected.
- Latency-only changes are accepted only if `none|true` peak HBM stays flat or
  lower and AsymGEMM/GEMM counts do not blow up.
- Memory-changing work must compare against the accepted post-S4 baseline above.
- Unit tests and microbenchmarks are not acceptance. They only allow an LF run.
- Every accepted run must record peak allocated/reserved HBM, forward/backward
  time, step time, AsymGEMM/GEMM counts, fallback count, and activation-offload
  stage/cpu-live counters.
- For the b4s4096 Qwen3-30B target, treat allocator-noise-sized changes
  (`<1 GiB` or `<1%`) as trivial. Prefer multi-GiB peak-HBM reductions; a
  smaller reduction needs explicit tensor-attribution evidence and must not
  materially worsen latency.

## Do Not Retry

These are already ruled out as standalone changes:

- CPU-source LoRA-B backward by itself while full `grad_gate_up` still exists.
  V2 showed this can keep peak HBM unchanged and worsen latency.
- Re-land the V2 paired LoRA-A wrapper/call-count change. Only a real native
  paired kernel with CPU-tile reuse is eligible.
- Offload full `dY` to CPU only to compute LoRA-B grads. This adds a wide D2H
  path and has no current peak evidence.
- Chase the inferred `5.795 GiB` expert-workspace residual without exact
  snapshot proof.
- Unbounded async prefetch/staging. Async may improve latency, but only with a
  hard cap on staged HBM.
- Add Python expert loops or split the expert path into per-expert small GEMMs.

CPU handle cleanup is not a V4 HBM optimization. It is already mostly working:
source profile reports `total_cpu_live_bytes=0` after cleanup. Keep debug-only
lifetime checks if useful, but run acceptance profiles with checks disabled.

## AsymGEMM-Specific Scope

The main body only tracks novel AsymGEMM-specific work that can support the
memory claim:

```text
Forward:
  offload routed expert activations/state to CPU at math last-use points

Backward:
  consume the CPU-resident activations through grouped CPU-source
  AsymGEMM/native kernels, avoiding full HBM fetchback whenever that tensor is
  a meaningful peak-HBM owner

Comparison:
  same model, optimizer, LoRA config, routing, sequence/batch, profiler, and
  global recompute setting; compare `none|true` against `gc-exp|false`
```

The real AsymGEMM-specific implementation work is CPU-resident expert-state
lifetime plus grouped CPU-source consumption. Generic LoRA fusion, generic
saved-tensor CPU offload, generic activation checkpointing, and HBM-resident
LoRA epilogue fusion are moved to the overlap notes at the end of this
document. They can inform implementation, but they are not the central claim.

## Real Remaining Kernel Work

Only the kernels below are currently eligible as novel V4 implementation work.
Each one must preserve the grouped-kernel design: no per-expert Python loops, no
small-GEMM decomposition, and no generic HBM reload path that hides the actual
CPU-source cost.

- K1 / V4-S1: CPU-source gate/up base `dX`, removing the full HBM
  `grad_gate_up [M,2I]` stage.
- K2 / V4-S2: true native paired gate/up LoRA-A forward, reusing one `X_cpu`
  tile for both gate and up.
- K3 / V4-S3: conditional optimization of the existing grouped LoRA-A grad
  CPU-right reductions, only if profiling proves latency ownership.
- K4 / V4-S4: conditional integrated gate/up backward CPU-source schedule, only
  if K1 evidence shows the separate schedule is insufficient.

The detailed file scope, pseudocode, risks, and validation gates are in the
implementation stages below.

## Implementation Stages

The stages below are the execution order. Do not skip the validation gate at the
end of a stage. Unit tests and standalone NCU runs are necessary for correctness
and kernel diagnosis, but the keep/reject decision normally requires the full
LF sweep.

Each stage is written for direct implementation with the same structure:
scope, concrete implementation/pseudocode, resolved facts, risks or unresolved
watch items, exact validation, and an explicit keep/reject gate.

Common full-sweep command:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
OUTPUT_ROOT="$PWD/outputs/<stage-name>_b4s4096_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=0 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYMM_EXP_ACT_POLICIES="none|true|false,gc-exp|false|false,none|false|false" \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
DATASET_MIN_TOKENS=4096 \
DATASET_OVERWRITE=true \
PROFILERS=source \
PROFILE_MEMORY_ATTRIBUTION=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_SNAPSHOT=false \
WARMUP_STEPS=5 \
MAX_STEPS=10 \
RUN_POST=false \
PLOT=false \
scripts/lf/profile_lora_lf.sh --gpus 0
```

For a stage-specific validation command, copy the full block above exactly and
replace only `OUTPUT_ROOT`; add the stage-specific env vars listed in that
stage immediately before the block.

Use the exact stage names below for `<stage-name>`:

```text
expact_v4_s1_gate_up_base_dx_cpu_source
expact_v4_s2_native_pair_forward
expact_v4_s3_lora_a_grad_reduce
expact_v4_s4_integrated_gate_up_backward
```

Keep a change only if `none|true` peak HBM drops meaningfully without a latency
blowup, or if it is a latency-only stage with flat/lower peak HBM. Reject same
memory with worse latency. Reject trivial memory savings with material latency
increase.

### V4-S0: Evidence And Instrumentation Gate

Why needed: V2 failed because a local tensor lifetime changed while the global
peak stayed elsewhere. S0 verifies exact peak ownership and counter reporting
before changing kernels.

Scope: files/functions/classes:

- `scripts/lf/run_lf_profiled_train.py`
  - `_start_memory_snapshot_recording`
  - `_dump_memory_snapshot`
  - `_activation_offload_counters_from_model`
- `scripts/testing/analyze_cuda_memory_snapshot.py`
  - `analyze_snapshot`
  - CLI `main`
- `asym_gemm/profiling/lf_trace.py`
  - memory-breakdown labels and residual wording
- `asym_gemm/training/frozen_linear.py`
  - `AsymExecutionStats`

Implementation and pseudocode:

1. Do not add new runtime hooks unless existing fields are missing. Current
   code already captures snapshots and activation-offload counters.
2. When adding later kernel counters, add explicit fields to
   `AsymExecutionStats` so source profile records them through `as_dict()`.
3. Keep inferred residual rows labeled as inferred; do not attribute them to
   experts without snapshot evidence.

Pseudocode for later counter additions:

```python
# asym_gemm/training/frozen_linear.py
class AsymExecutionStats:
    expact_gate_up_base_dx_cpu_source_calls: int = 0
    expact_lora_a_pair_forward_native_calls: int = 0
    expact_lora_a_grad_cpu_right_native_calls: int = 0
```

Resolved facts from code exploration:

- The current activation-offload backward stages full `grad_gate_up` through
  `ActivationOffloadManager.stage_concat_columns(...)`.
- Existing `grouped_lora_a_pair_forward_cpu_left(...)` is a wrapper over two
  CPU-left calls, not a physical paired kernel.
- Existing LoRA-A grad kernels are custom grouped reductions from CPU source;
  they are not normal AsymGEMM left/right calls.
- `layer.gate_up_base.host_weight.weight` is a pinned CPU `HostWeight`, so K1
  is not "CPU grad source plus CUDA weight"; it is CPU grad source plus
  CPU-resident base weight with CUDA `grad_packed` output.

Risks / unresolved watch items:

- Memory snapshot trace truncation can hide the true peak. Use one measured
  step and `ASYM_GEMM_LF_PROFILE_MEMORY_SNAPSHOT_MAX_ENTRIES=1000000`.
- Snapshot attribution may include C++ frames without useful Python labels.
  Use allocation sizes and stage tags as cross-checks.

Validation before next stage:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
OUTPUT_ROOT="$PWD/outputs/expact_v4_s0_snapshot_b4s4096_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=0 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYMM_EXP_ACT_POLICIES="none|true|false" \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
DATASET_MIN_TOKENS=4096 \
DATASET_OVERWRITE=true \
PROFILERS=source \
PROFILE_MEMORY_ATTRIBUTION=true \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_MEMORY_BREAKDOWN_INTERVAL=1 \
PROFILE_MEMORY_SNAPSHOT=true \
ASYM_GEMM_LF_PROFILE_MEMORY_SNAPSHOT_MAX_ENTRIES=1000000 \
WARMUP_STEPS=5 \
MAX_STEPS=1 \
RUN_POST=false \
PLOT=false \
scripts/lf/profile_lora_lf.sh --gpus 0

PYTHONPATH="$PWD" .venv/bin/python scripts/testing/analyze_cuda_memory_snapshot.py \
  <path-to-memory_snapshot.pickle> --top 40
```

Keep/reject gate: proceed only if the snapshot/counters support a V4 target, or
the next stage is strictly latency-only with flat/lower HBM expected.

### V4-S1: CPU-Source Gate/Up Base `dX` And No Full `grad_gate_up`

Why needed: this is the main memory-stage candidate. The current code creates
`grad_gate_up [M,2I]` in HBM for base `dX`; moving only LoRA-B to CPU-source
failed in V2 because this full stage remained. S1 must remove or strictly bound
that full stage.

Scope: files/functions/classes:

- `asym_gemm/training/qwen3_moe.py`
  - `_ActivationOffloadQwen3ExpertFunction.backward`
  - `_grouped_base_dx` remains fallback only
  - add/read `ASYMM_EXPACT_GATE_UP_BASE_DX_CPU_SOURCE`
- `asym_gemm/training/activation_offload.py`
  - `ActivationOffloadManager.stage`
  - `ActivationOffloadManager.stage_concat_columns` must not be used on the
    accepted S1 path
- `asym_gemm/training/exp_act_offload_lora.py`
  - add `GATE_UP_BASE_DX_CPU_SOURCE`
  - add `grouped_gate_up_base_dx_cpu_source`
  - add `_check_pinned_cpu_bf16_3d`
  - update `require_expert_activation_offload_kernels` if the S1 path is enabled
- `asym_gemm/training/frozen_linear.py`
  - add `expact_gate_up_base_dx_cpu_source_calls`
- `csrc/exp_act_offload/exp_act_offload_kernels.cu`
  - add `sm100_grouped_gate_up_base_dx_bf16_cpu_source`
- `csrc/apis/exp_act_offload.hpp`
  - add prototype and `m.def`
- `asym_gemm/__init__.py`
  - expose the new symbol through `_maybe_import_from_C`
- `setup.py`
  - no source-list change if implemented in existing
    `csrc/exp_act_offload/exp_act_offload_kernels.cu`
- `tests/training/test_exp_act_offload_native.py`
  - add direct native wrapper correctness test
- `tests/training/test_lf_qwen3_asym_backend.py`
  - update activation-offload correctness test to assert the accepted S1 path
    has no `dgate_up_for_gate_up_base` full stage when the native path is on
  - assert new bounded tags such as `dgate_for_gate_lora` and
    `dup_for_up_lora` are present only for LoRA-B half staging
- `scripts/testing/profile_expact_kernel_microbench.py`
  - new isolated K1/K2 microbenchmark for correctness timing and NCU

Shared microbench script pseudocode:

```python
# scripts/testing/profile_expact_kernel_microbench.py
def make_grouped_metadata(rows, groups, experts, device):
    # Build contiguous int32 pair offsets [start0,end0,start1,end1,...]
    # and experts with sentinel. Include uneven rows and repeated experts.
    return offsets_i32, experts_i32

def make_inputs(args):
    # Allocate CUDA BF16 references, then CPU pinned copies for CPU-source args.
    # Use shapes matching Qwen3-style [M,H], [M,I], [E,2I,H], [E,r,H].
    return tensors

def reference(kernel_name, tensors):
    # Use grouped torch math only for correctness, not performance.
    # Loop over route groups is allowed here because this is a validator.
    return expected_outputs

def run_kernel(kernel_name, tensors, iters):
    # Call the exact Python wrapper used by qwen3_moe.py.
    # Synchronize before/after timing. Return outputs and elapsed ms.
    return outputs, ms

def main():
    args = parse_args()
    tensors = make_inputs(args)
    outputs, ms = run_kernel(args.kernel, tensors, args.iters)
    if args.check:
        assert_close(outputs, reference(args.kernel, tensors))
    print(json.dumps({"kernel": args.kernel, "ms": ms, "shapes": ...}))
```

This script is not an acceptance benchmark. It exists only to validate native
math, support NCU, and prevent full LF runs on obviously broken kernels.

Python wrapper pseudocode:

```python
GATE_UP_BASE_DX_CPU_SOURCE = "sm100_grouped_gate_up_base_dx_bf16_cpu_source"

def _check_pinned_cpu_bf16_3d(source_cpu, tag):
    if source_cpu.device.type != "cpu":
        raise RuntimeError(f"{tag}: expected CPU tensor")
    if source_cpu.dtype != torch.bfloat16 or source_cpu.dim() != 3:
        raise RuntimeError(f"{tag}: expected BF16 CPU tensor [E,N,K]")
    if not source_cpu.is_contiguous():
        raise RuntimeError(f"{tag}: expected contiguous CPU tensor")
    if torch.cuda.is_available() and not source_cpu.is_pinned():
        raise RuntimeError(f"{tag}: expected pinned CPU tensor")

def grouped_gate_up_base_dx_cpu_source(
    dgate_cpu, dup_cpu, gate_up_weight_cpu, offsets, experts,
    *, input_dtype, stats, tag,
):
    native = _native_symbol(GATE_UP_BASE_DX_CPU_SOURCE)
    _check_pinned_cpu_bf16_2d(dgate_cpu, f"{tag}.dgate")
    _check_pinned_cpu_bf16_2d(dup_cpu, f"{tag}.dup")
    _check_pinned_cpu_bf16_3d(gate_up_weight_cpu, f"{tag}.weight")
    if dgate_cpu.shape != dup_cpu.shape:
        raise RuntimeError(f"{tag}: dgate/dup shape mismatch")
    if offsets.device.type != "cuda" or experts.device.type != "cuda":
        raise RuntimeError(f"{tag}: grouped metadata must be CUDA")

    M = int(dgate_cpu.shape[0])
    I = int(dgate_cpu.shape[1])
    E = int(gate_up_weight_cpu.shape[0])
    H = int(gate_up_weight_cpu.shape[2])
    if tuple(gate_up_weight_cpu.shape) != (E, 2 * I, H):
        raise RuntimeError(f"{tag}: expected gate_up weight [E,2I,H]")

    offsets_i32, experts_i32, list_size = _group_metadata_tensors(
        offsets, experts, device=offsets.device
    )
    grad_packed = torch.empty(
        (M, H),
        device=offsets.device,
        dtype=input_dtype,
    )
    native(dgate_cpu, dup_cpu, gate_up_weight_cpu, grad_packed,
           offsets_i32, experts_i32, list_size)
    if stats is not None:
        stats.expact_gate_up_base_dx_cpu_source_calls += 1
    return grad_packed
```

Autograd rewrite pseudocode:

```python
# current code creates full grad_gate_up; S1 path must not.

use_s1 = os.environ.get("ASYMM_EXPACT_GATE_UP_BASE_DX_CPU_SOURCE", "0") == "1"
if not use_s1:
    # Keep the current stage_concat_columns path unchanged.
    ...
    return current_backward_result

with prof_range("backward.mlp.activation_offload.gate_lora"):
    gate_low_rank = stage_low_rank_from_cpu(
        ctx.gate_low_rank_cpu, manager, tag="S_gate_for_dB", stats=layer.stats
    )
    grad_gate_stage = manager.stage(grad_gate_cpu, tag="dgate_for_gate_lora")
    dS_gate = _grouped_lora_cuda_view(
        grad_gate_stage, gate_lora_B.transpose(-1, -2), metadata=lora_metadata
    ).mul_(layer.lora_scale)
    grad_gate_lora_B = _grouped_lora_weight_grads_torch(
        grad_gate_stage, gate_low_rank, offsets, experts,
        int(gate_lora_B.shape[0]), out_dtype=gate_lora_B.dtype,
        metadata=lora_metadata, stats=layer.stats,
    ).mul_(layer.lora_scale)
    layer.stats.expact_lora_b_backward_grouped_calls += 1
    manager.release_stage(gate_low_rank, drop_cache=True)
    if need_grad_packed:
        grad_gate_lora_x = grouped_expert_lora(
            dS_gate, gate_lora_A.transpose(-1, -2), offsets, experts,
            metadata=lora_metadata,
        )
    else:
        grad_gate_lora_x = None
    manager.release_stage(grad_gate_stage, drop_cache=True)
    manager.release_cpu(ctx.gate_low_rank_cpu)

with prof_range("backward.mlp.activation_offload.up_lora"):
    up_low_rank = stage_low_rank_from_cpu(
        ctx.up_low_rank_cpu, manager, tag="S_up_for_dB", stats=layer.stats
    )
    grad_up_stage = manager.stage(grad_up_cpu, tag="dup_for_up_lora")
    dS_up = _grouped_lora_cuda_view(
        grad_up_stage, up_lora_B.transpose(-1, -2), metadata=lora_metadata
    ).mul_(layer.lora_scale)
    grad_up_lora_B = _grouped_lora_weight_grads_torch(
        grad_up_stage, up_low_rank, offsets, experts,
        int(up_lora_B.shape[0]), out_dtype=up_lora_B.dtype,
        metadata=lora_metadata, stats=layer.stats,
    ).mul_(layer.lora_scale)
    layer.stats.expact_lora_b_backward_grouped_calls += 1
    manager.release_stage(up_low_rank, drop_cache=True)
    if need_grad_packed:
        grad_up_lora_x = grouped_expert_lora(
            dS_up, up_lora_A.transpose(-1, -2), offsets, experts,
            metadata=lora_metadata,
        )
    else:
        grad_up_lora_x = None
    manager.release_stage(grad_up_stage, drop_cache=True)
    manager.release_cpu(ctx.up_low_rank_cpu)

with prof_range("backward.mlp.activation_offload.gate_up_lora_a_grad"):
    grad_gate_lora_A, grad_up_lora_A = grouped_lora_a_pair_grad_cpu_right(
        dS_gate, dS_up, ctx.x_cpu.tensor, offsets, experts, ...
    )
    manager.release_cpu(ctx.x_cpu)

if need_grad_packed:
    with prof_range("backward.mlp.activation_offload.gate_up_base_dx_cpu_source"):
        grad_packed = grouped_gate_up_base_dx_cpu_source(
            grad_gate_cpu.tensor,
            grad_up_cpu.tensor,
            layer.gate_up_base.host_weight.weight,
            offsets,
            experts,
            input_dtype=ctx.input_dtype,
            stats=layer.stats,
            tag="gate_up_base_dx",
        )
    if grad_gate_lora_x is not None:
        grad_packed.add_(grad_gate_lora_x.to(dtype=grad_packed.dtype))
    if grad_up_lora_x is not None:
        grad_packed.add_(grad_up_lora_x.to(dtype=grad_packed.dtype))
else:
    grad_packed = None

manager.release_cpu(grad_gate_cpu)
manager.release_cpu(grad_up_cpu)
```

Native kernel pseudocode:

```cpp
check_cpu_bf16_2d(dgate_cpu, "dgate_cpu");
check_cpu_bf16_2d(dup_cpu, "dup_cpu");
check_cpu_bf16_3d(gate_up_weight_cpu, "gate_up_weight_cpu");
check_cuda_bf16_or_output_dtype_2d(grad_packed, "grad_packed");
check_offsets_experts(offsets, experts, list_size);
validate shapes:
  dgate_cpu [M,I], dup_cpu [M,I]
  gate_up_weight_cpu [E,2I,H]
  grad_packed [M,H]

// One grouped grid over route groups and output-H tiles.
// dgate/dup and W are all pinned CPU operands; grad_packed is CUDA output.
for group g in parallel:
  expert = experts[g]
  rows = [offsets[2*g], offsets[2*g+1])
  for row_tile, h_tile:
    for row in row_tile:
      for h in h_tile:
        acc = 0.0f
        for i_tile in 0..I:
          // Efficient implementation must load CPU gradient and CPU weight
          // tiles into SMEM and reuse them across h_tile before evicting.
          acc += bf16(dgate_cpu[row, i]) * bf16(W[expert, i, h])
          acc += bf16(dup_cpu[row, i])   * bf16(W[expert, I + i, h])
        grad_packed[row, h] = cast_to_output_dtype(acc)
```

Implementation constraints:

- The accepted path must not call `stage_concat_columns` for gate/up backward.
- Do not use CPU-source LoRA-B as the main change; that repeats V2 unless the
  full base `dX` stage is gone.
- If the native S1 symbol is requested and missing, fail closed. Do not silently
  fall back during acceptance profiling.
- BF16 SM100 is the first target. Other precisions can keep the old path until
  separately justified.

Risks / unresolved watch items:

- Half-stage LoRA-B work may still overlap with `grad_packed` or LoRA `dX`
  temporaries. Validate with snapshot, not only stage counters.
- `grad_packed` created earlier can increase overlap. Keep base `dX` after
  releasing the half stages unless profiling proves another ordering is better.
- Weight layout for `gate_up_base.host_weight.weight` must be confirmed as
  pinned CPU `[E, 2I, H]` and gate rows before up rows.
- K1 has two CPU-source operands (`dgate/dup` and base weight). This is a real
  new kernel requirement, not a direct reuse of existing CPU-left or CPU-right
  AsymGEMM. If CPU traffic makes latency explode and peak HBM does not drop
  meaningfully, reject the stage.
- Native kernel must handle empty groups, repeated experts, nonuniform group
  lengths, and `need_grad_packed=False`.

Validation before next stage:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
.venv/bin/python -m pip install -e . --no-build-isolation
PYTHONPATH="$PWD" .venv/bin/python -m pytest \
  tests/training/test_exp_act_offload_native.py \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend \
  -q

PYTHONPATH="$PWD" .venv/bin/python scripts/testing/profile_expact_kernel_microbench.py \
  --kernel gate_up_base_dx_cpu_source --rows 32768 --hidden 2048 \
  --intermediate 6144 --experts 128 --groups 128 --check

ncu --set full --target-processes all -o outputs/ncu_gate_up_base_dx_cpu_source \
  .venv/bin/python scripts/testing/profile_expact_kernel_microbench.py \
  --kernel gate_up_base_dx_cpu_source --rows 32768 --hidden 2048 \
  --intermediate 6144 --experts 128 --groups 128 --iters 20
```

The correctness test must include these assertions when the S1 native path is
enabled:

```python
stats = asym_backend._last_activation_offload_stats
assert stats["stage_peak_by_tag"].get("dgate_up_for_gate_up_base", 0) == 0
assert stats["stage_peak_by_tag"].get("dgate_for_gate_lora", 0) > 0
assert stats["stage_peak_by_tag"].get("dup_for_up_lora", 0) > 0
assert asym_backend.stats.expact_gate_up_base_dx_cpu_source_calls > 0
```

Then run S0 snapshot with the new path on and the common full LF sweep. Use the
common full-sweep command exactly, with only these changes:

```text
OUTPUT_ROOT="$PWD/outputs/expact_v4_s1_gate_up_base_dx_cpu_source_b4s4096_$(date -u +%Y%m%dT%H%M%SZ)"
ASYMM_EXPACT_GATE_UP_BASE_DX_CPU_SOURCE=1
ASYMM_EXPACT_REQUIRE_GATE_UP_BASE_DX_CPU_SOURCE=1
```

Keep/reject gate: keep only if `none|true` peak HBM is meaningfully below
`102.312 GiB` without a backward-time blowup, or if HBM is flat/lower and
latency improves. Reject if memory is unchanged and latency is worse; reject if
the HBM reduction is trivial and latency materially worsens.

### V4-S2: True Paired Gate/Up LoRA-A Forward CPU-Tile Reuse

Why needed: after S1, latency remains the main weakness. The current
`grouped_lora_a_pair_forward_cpu_left` function calls two CPU-left kernels. S2
is eligible only as latency work with flat/lower HBM.

Scope: files/functions/classes:

- `asym_gemm/training/exp_act_offload_lora.py`
  - add `import os`
  - `grouped_lora_a_pair_forward_cpu_left`
  - add `LORA_A_PAIR_FORWARD_CPU_LEFT`
- `asym_gemm/training/qwen3_moe.py`
  - `_ActivationOffloadQwen3ExpertFunction.forward`
- `asym_gemm/training/frozen_linear.py`
  - add `expact_lora_a_pair_forward_native_calls`
- `csrc/exp_act_offload/exp_act_offload_kernels.cu`
  - add `sm100_grouped_lora_a_pair_forward_bf16_cpu_left`
- `csrc/apis/exp_act_offload.hpp`
  - add prototype and `m.def`
- `asym_gemm/__init__.py`
  - expose symbol
- `tests/training/test_cpu_left_lora.py`
  - update/add wrapper test for native paired path and fallback path
- `tests/training/test_lf_qwen3_asym_backend.py`
  - preserve full activation-offload correctness test

Wrapper pseudocode:

```python
LORA_A_PAIR_FORWARD_CPU_LEFT = "sm100_grouped_lora_a_pair_forward_bf16_cpu_left"

def grouped_lora_a_pair_forward_cpu_left(source_cpu, lora_a_gate, lora_a_up,
                                         offsets, experts, *, metadata, stats, tag):
    reason = _missing_symbol(LORA_A_PAIR_FORWARD_CPU_LEFT)
    require_native = os.environ.get("ASYMM_EXPACT_REQUIRE_NATIVE_PAIR_FORWARD", "0") == "1"
    if reason is None:
        native = _native_symbol(LORA_A_PAIR_FORWARD_CPU_LEFT)
        _check_cpu_left_inputs(source_cpu, lora_a_gate, f"{tag}.gate")
        _check_cpu_left_inputs(source_cpu, lora_a_up, f"{tag}.up")
        gate = torch.empty((M, rank), device=lora_a_gate.device, dtype=lora_a_gate.dtype)
        up = torch.empty_like(gate)
        offsets_i32, experts_i32, list_size = _group_metadata_tensors(...)
        native(source_cpu, lora_a_gate, lora_a_up, gate, up,
               offsets_i32, experts_i32, list_size)
        stats.expact_lora_a_pair_forward_native_calls += 1
        return gate, up

    if require_native:
        raise RuntimeError(f"{tag}: required native paired CPU-left kernel unavailable: {reason}")

    # Development fallback only, not an accepted S2 result.
    gate = grouped_lora_a_forward_cpu_left(...)
    up = grouped_lora_a_forward_cpu_left(...)
    return gate, up
```

Native kernel pseudocode:

```cpp
for group g in parallel:
  expert = experts[g]
  for row_tile, rank_tile:
    acc_gate[rows, rank_tile] = 0
    acc_up[rows, rank_tile] = 0
    for h_tile in 0..H:
      load X_cpu[row_tile, h_tile] once into SMEM
      load A_gate[expert, rank_tile, h_tile]
      load A_up[expert, rank_tile, h_tile]
      acc_gate += X_tile * A_gate_tile
      acc_up   += X_tile * A_up_tile
    write S_gate[row_tile, rank_tile]
    write S_up[row_tile, rank_tile]
```

Risks / unresolved watch items:

- If the implementation is just two launches behind one Python function, it is
  not S2 and must be rejected.
- CPU tile reuse must be visible in NCU; otherwise it is only call-count
  cleanup.
- This should not change peak HBM. Any peak increase rejects the stage even if
  forward time improves.

Validation before next stage:

```bash
.venv/bin/python -m pip install -e . --no-build-isolation
PYTHONPATH="$PWD" .venv/bin/python -m pytest \
  tests/training/test_cpu_left_lora.py::test_expact_lora_a_forward_wrappers_use_real_cpu_left \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend \
  -q

PYTHONPATH="$PWD" .venv/bin/python scripts/testing/profile_expact_kernel_microbench.py \
  --kernel lora_a_pair_forward_cpu_left --rows 32768 --hidden 2048 \
  --rank 8 --experts 128 --groups 128 --check

ncu --set full --target-processes all -o outputs/ncu_lora_a_pair_forward_cpu_left \
  .venv/bin/python scripts/testing/profile_expact_kernel_microbench.py \
  --kernel lora_a_pair_forward_cpu_left --rows 32768 --hidden 2048 \
  --rank 8 --experts 128 --groups 128 --iters 50
```

Then run the common full LF sweep with native-pair fail-closed. Use the common
full-sweep command exactly, with only these changes:

```text
OUTPUT_ROOT="$PWD/outputs/expact_v4_s2_native_pair_forward_b4s4096_$(date -u +%Y%m%dT%H%M%SZ)"
ASYMM_EXPACT_REQUIRE_NATIVE_PAIR_FORWARD=1
```

Keep/reject gate: keep only if `none|true` latency improves and peak HBM is
flat/lower. Reject any peak-HBM increase. Reject if this is only a wrapper/call
count cleanup without NCU evidence of CPU tile reuse.

### V4-S3: Conditional LoRA-A Grad CPU-Right Reduction Optimization

Why needed: K3 is not expected to reduce peak HBM. It is only worth doing if
S1/S2 full profiles or NCU show the existing LoRA-A grad kernels materially
own latency.

Scope: files/functions/classes:

- `csrc/exp_act_offload/exp_act_offload_kernels.cu`
  - `lora_a_grad_kernel`
  - `sm100_grouped_lora_a_grad_bf16_cpu_right`
  - `sm100_grouped_lora_a_pair_grad_bf16_cpu_right`
- `asym_gemm/training/exp_act_offload_lora.py`
  - `grouped_lora_a_grad_cpu_right`
  - `grouped_lora_a_pair_grad_cpu_right`
- `tests/training/test_exp_act_offload_native.py`
  - existing single/pair LoRA-A grad tests plus repeated-expert/uneven-route case
- `scripts/testing/profile_expact_kernel_microbench.py`
  - add `--kernel lora_a_grad_cpu_right` and `--kernel lora_a_pair_grad_cpu_right`

Kernel rewrite pseudocode:

```cpp
for group g, rank_tile, k_tile:
  expert = experts[g]
  acc0[rank_tile, k_tile] = 0
  acc1[rank_tile, k_tile] = 0
  for row_tile in group rows:
    load X_cpu[row_tile, k_tile] into SMEM once
    load dS_gate[row_tile, rank_tile]
    optionally load dS_up[row_tile, rank_tile]
    acc0 += dS_gate * X_tile
    acc1 += dS_up   * X_tile
  // repeated experts can appear across route groups, so final update still
  // needs group-level atomics or a separate expert reduction.
  atomicAdd grad_gate_acc[expert, rank_tile, k_tile] by acc0
  atomicAdd grad_up_acc[expert, rank_tile, k_tile] by acc1
cast FP32 accumulators to BF16 output
```

Risks / unresolved watch items:

- Repeated expert groups require correct accumulation. Removing atomics without
  a replacement reduction is wrong.
- FP32 accumulator memory can itself become visible. Do not add larger
  temporary buffers than the current implementation without snapshot proof.
- K3 is rejected if it keeps HBM flat but worsens latency, or if latency improves
  only in a toy microbench and not in the LF sweep.

Validation before next stage:

```bash
.venv/bin/python -m pip install -e . --no-build-isolation
PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/training/test_exp_act_offload_native.py -q

PYTHONPATH="$PWD" .venv/bin/python scripts/testing/profile_expact_kernel_microbench.py \
  --kernel lora_a_pair_grad_cpu_right --rows 32768 --hidden 2048 \
  --rank 8 --experts 128 --groups 128 --check

ncu --set full --target-processes all -o outputs/ncu_lora_a_pair_grad_cpu_right \
  .venv/bin/python scripts/testing/profile_expact_kernel_microbench.py \
  --kernel lora_a_pair_grad_cpu_right --rows 32768 --hidden 2048 \
  --rank 8 --experts 128 --groups 128 --iters 20
```

Then run the common full LF sweep exactly, changing only:

```text
OUTPUT_ROOT="$PWD/outputs/expact_v4_s3_lora_a_grad_reduce_b4s4096_$(date -u +%Y%m%dT%H%M%SZ)"
```

Keep/reject gate: keep only if the full `none|true` LF profile improves latency
with peak HBM flat/lower. Reject if the win exists only in the microbench or NCU
run.

### V4-S4: Conditional Integrated Gate/Up Backward CPU-Source Schedule

Why needed: only if S1 cannot remove enough peak HBM or profiling proves the
separate gate/up backward consumers reread CPU state inefficiently. S4 is the
only place where CPU-source LoRA-B can reappear, and only as part of an
integrated schedule that also removes the full base `dX` stage.

Scope: files/functions/classes:

- `asym_gemm/training/qwen3_moe.py`
  - `_ActivationOffloadQwen3ExpertFunction.backward`
- `asym_gemm/training/exp_act_offload_lora.py`
  - new integrated wrapper, for example
    `grouped_gate_up_backward_cpu_source_integrated`
- `csrc/exp_act_offload/exp_act_offload_kernels.cu`
  - new integrated kernel or a small set of grouped CPU-source kernels sharing
    one schedule
- `csrc/apis/exp_act_offload.hpp`
  - prototype and binding
- `asym_gemm/training/frozen_linear.py`
  - integrated-call counter
- `tests/training/test_exp_act_offload_native.py`
  - direct integrated-kernel correctness test
- `tests/training/test_lf_qwen3_asym_backend.py`
  - e2e correctness and stage-counters assertions

Integrated pseudocode:

```python
# High-level backward after activation backward produced grad_gate_cpu/grad_up_cpu.
outputs = grouped_gate_up_backward_cpu_source_integrated(
    dgate_cpu=grad_gate_cpu.tensor,
    dup_cpu=grad_up_cpu.tensor,
    x_cpu=ctx.x_cpu.tensor,
    S_gate_cpu=ctx.gate_low_rank_cpu.tensor,
    S_up_cpu=ctx.up_low_rank_cpu.tensor,
    gate_up_weight=layer.gate_up_base.host_weight.weight,
    gate_lora_A=gate_lora_A,
    up_lora_A=up_lora_A,
    gate_lora_B=gate_lora_B,
    up_lora_B=up_lora_B,
    offsets=offsets,
    experts=experts,
    scale=layer.lora_scale,
)
grad_packed = outputs.grad_packed
grad_gate_lora_A = outputs.grad_gate_lora_A
grad_up_lora_A = outputs.grad_up_lora_A
grad_gate_lora_B = outputs.grad_gate_lora_B
grad_up_lora_B = outputs.grad_up_lora_B
```

Native scheduling pseudocode:

```cpp
for group g in parallel:
  expert = experts[g]
  for row/output tiles:
    load dgate_cpu and dup_cpu tiles once
    compute base grad_packed contribution using W_gate_up
    compute dS_gate/dS_up using LoRA-B if integrated LoRA-B is enabled
    reduce LoRA-B grads against S_gate/S_up
    reduce LoRA-A grads against X_cpu
    add LoRA dX contribution to grad_packed without materializing full
    grad_gate_up [M,2I]
```

Risks / unresolved watch items:

- This is high-risk and may become a hard-to-debug fused backward. Do not start
  it unless S1 evidence says it is necessary.
- It can accidentally repeat the V2 failure if it only moves LoRA-B but still
  stages full `grad_gate_up`.
- Multiple reductions into parameter gradients must preserve repeated-expert
  correctness and FP32 accumulation quality.

Validation:

```bash
.venv/bin/python -m pip install -e . --no-build-isolation
PYTHONPATH="$PWD" .venv/bin/python -m pytest \
  tests/training/test_exp_act_offload_native.py \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend \
  -q

ncu --set full --target-processes all -o outputs/ncu_gate_up_backward_integrated \
  .venv/bin/python scripts/testing/profile_expact_kernel_microbench.py \
  --kernel gate_up_backward_integrated --rows 32768 --hidden 2048 \
  --intermediate 6144 --rank 8 --experts 128 --groups 128 --iters 20
```

Then run S0 snapshot and the common full LF sweep exactly, changing only:

```text
OUTPUT_ROOT="$PWD/outputs/expact_v4_s4_integrated_gate_up_backward_b4s4096_$(date -u +%Y%m%dT%H%M%SZ)"
```

Keep/reject gate: keep only if peak HBM drops meaningfully without blowing up
latency. Otherwise revert.

## Stop Conditions

Stop V4 work when one of these is true:

- V4-S0 shows the remaining candidate tensors are not peak-live and V4-S2 has no
  proven latency path.
- A candidate keeps memory the same and worsens latency.
- A candidate saves only tiny HBM while materially worsening latency.
- The implementation needs per-expert loops, small GEMM decomposition, or a
  comparison change outside the accepted LF workflow.

If all V4 stages are skipped or rejected, the honest conclusion is that the
remaining global peak is outside exact routed expert activation-offload tensors.
Further HBM reduction would need a fair change applied to both expact and
checkpoint baselines, such as non-expert activation/loss/norm memory work.

## Overlap Implementation Policy

The sections below are deliberately outside the main V4 implementation body.
They record prior-art overlap and possible supporting ideas so we do not
recreate or accidentally claim someone else's contribution.

Do not implement these overlapping candidates by default. Implement one only if
it satisfies at least one of these conditions:

- it directly unlocks K1, K2, K3, or K4 above;
- it is required to validate an AsymGEMM-specific CPU-source optimization;
- it improves latency while keeping the accepted `none|true` HBM comparison
  flat/lower and is clearly recorded as borrowed/non-core engineering.

Otherwise, skip it. A normal HBM-resident fused LoRA kernel, generic activation
offload hook, generic reload scheduler, or optimizer-side change is not a V4
contribution.

## Appendix: Overlap With LoRAFusion

LoRAFusion, local path `/workspace/AsymGEMM-SFT/third_party/lorafusion`, is
prior art for fused LoRA linear kernels, not for this memory claim. Its single
LoRA path computes and saves `S = dropout(X) A.T`, then fuses:

```text
forward:  Y = X W.T + alpha * S B.T
backward: dB, dS = fused(dY, S, B)
          dX     = fused(dY W + dS A)
```

Overlapping ideas:

- low-rank `S` as the saved LoRA state
- add/beta epilogues that avoid materializing `down_delta` or LoRA `dX`
- fused `dY W + dS A` structure as inspiration for appendix candidate LF-2
- NCU/kernel traffic analysis and per-kernel tuning discipline

Overlapping candidate LF-1, not a core AsymGEMM stage: remove
`down_delta [M,H]` materialization.

- Current expact forward materializes `down_delta`, keeps it alive through down
  base, then does `output.add_(down_delta)`.
- The LoRAFusion-like version is `Y_down += LoRA_B(S_down)` with an add/beta
  epilogue.
- Files would be `asym_gemm/training/qwen3_moe.py`,
  `asym_gemm/training/lora.py`, and a native grouped LoRA add-into-output
  binding if needed.
- This must not be implemented as `tmp = grouped_expert_lora(...);
  output.add_(tmp)`, because that is the current lifetime problem.
- This is only acceptable if it lowers peak HBM or improves latency with flat
  HBM in the full LF workflow. It is not a new AsymGEMM claim by itself.

Overlapping candidate LF-2, not a core AsymGEMM stage: add LoRA `dX` directly
into `grad_packed`.

- Current expact backward can materialize `grad_gate_lora_x [M,H]` and
  `grad_up_lora_x [M,H]`, then add them to `grad_packed`.
- The LoRAFusion-like version is `dX += dS_gate @ A_gate` and
  `dX += dS_up @ A_up` through grouped add-into-output epilogues.
- Files would be `asym_gemm/training/qwen3_moe.py`,
  `asym_gemm/training/lora.py`, and a native grouped LoRA add-into-output
  binding if needed.
- Risk: computing `grad_packed` earlier can make it overlap with LoRA backward
  work. If peak HBM does not improve, revert.
- This is only acceptable if it lowers peak HBM or improves latency with flat
  HBM in the full LF workflow. It is not a new AsymGEMM claim by itself.

How to use this overlap:

- It is valid prior art and implementation inspiration.
- Do not implement LF-1 or LF-2 unless it unlocks a later AsymGEMM-specific
  CPU-source kernel or produces accepted latency improvement with flat/lower
  HBM in the full LF workflow.
- Do not claim normal HBM-resident LoRAFusion-style fusion as our contribution.
- If a V4 change only fuses HBM LoRA operands, it is latency/local-memory
  engineering, not the central AsymGEMM activation-offload claim.
- The claim becomes AsymGEMM-specific only when the fused schedule consumes
  CPU-resident expert activations/low-rank state through grouped CPU-source
  kernels or preserves the fair expact-vs-recompute memory comparison.

## Appendix: Overlap With Megatron-LM

Megatron Core fine-grained activation offloading is prior art for module-level
activation offload. The local implementation is in
`/workspace/AsymGEMM-SFT/third_party/megatron-lm/megatron/core/pipeline_parallel/fine_grained_activation_offload.py`.
It uses saved-tensor hooks, D2H/H2D streams, group start/commit markers,
`forced_released_tensors`, inflight caps, and module choices such as
`expert_fc1` and `moe_act`.

Overlapping ideas:

- explicit group start/commit lifetime boundaries
- bounded async offload/reload queues
- forced release after offload when PyTorch GC is not enough
- offload summaries by module/group
- skipping offload for the last immediately-needed group to avoid blocking

Key difference:

- Megatron reloads saved CPU tensors back into full GPU tensors before backward
  consumers use them.
- Our core path must consume CPU-resident expert activations directly through
  grouped CPU-source AsymGEMM/native kernels where possible.
- If we only implement Megatron-style generic offload/reload, we are recreating
  prior art and should not claim a new AsymGEMM memory mechanism.

Implementation rule:

- Do not implement Megatron-style generic offload/reload as a V4 target.
- Borrow only lifetime-boundary or bounded-queue mechanics if they are needed
  to make K1/K2/K3/K4 correct or to cap staged HBM for an AsymGEMM-specific
  CPU-source kernel.
- If the result reloads the full tensor to HBM before the real consumer, it is
  overlap/prior art, not a core V4 optimization.

## Appendix: Generic Prior Art

PyTorch `saved_tensors_hooks` / `save_on_cpu` and DeepSpeed activation
checkpointing/CPU checkpointing are generic activation-memory mechanisms. They
are useful references for hook semantics, lifetime, and correctness, but they
do not replace the AsymGEMM claim. A fair result cannot be claimed from merely
wrapping the expert block in generic `save_on_cpu` or DeepSpeed checkpointing.

If an optimization comes from LoRAFusion/Megatron/PyTorch/DeepSpeed, record it
as borrowed prior art and only use it if both the target and baseline get the
same unrelated improvement, or if it is strictly internal latency work that
keeps the accepted memory comparison unchanged.

Implementation rule:

- Do not implement generic hooks/checkpointing/CPU optimizer changes as V4
  work.
- Use them only as references for correctness or if they are necessary support
  code for a CPU-source AsymGEMM kernel above.
- If adopted, label the result as supporting infrastructure, not as the paper's
  novel mechanism.
