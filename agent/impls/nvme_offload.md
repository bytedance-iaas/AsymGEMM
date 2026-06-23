# AsymGEMM NVMe Offload Implementation Plan

## Implementation Goal

Add NVMe support for AsymGEMM LoRA-SFT model parameters, LoRA weight homes, and optimizer state by reusing as much of vendored DeepSpeed's proven low-level NVMe machinery as is practical, while avoiding DeepSpeed/ZeRO ownership assumptions in the local single-GPU path.

The intended reuse is the performance substrate: AIO/GDS builders, AIO handles, alignment/min-size constants, pinned swap-buffer utilities, queue-depth/block-size/pipeline knobs, and reusable buffer-pool patterns. The local implementation should not copy DeepSpeed's distributed ZeRO state machine into AsymGEMM, and should not directly depend on classes that require `ds_id`, `ds_tensor`, partition state, process groups, ZeRO checkpoint ownership, or initialized DeepSpeed comm.

The design must keep a stable placement/materialization boundary so the local backend can be replaced later by a DeepSpeed-owned backend if multi-GPU/ZeRO support becomes necessary. In that future mode, DeepSpeed would own parameter residency, partitioning, optimizer state, checkpointing, and `param.data`; AsymGEMM would keep its compute policies/kernels and request materialized tensors or views through the same boundary. New local NVMe code should therefore improve single-GPU capacity/performance now without creating contract or ownership conflicts that make a later DeepSpeed backend hard to swap in.

## Design Contract

Target the local single-GPU AsymGEMM LoRA-SFT path first. This is not the DeepSpeed/ZeRO integration. It is a local AsymGEMM-owned implementation that mimics the useful DeepSpeed NVMe optimizations behind a replaceable placement backend. Keep the interface clean enough that a future DeepSpeed/ZeRO backend can own parameters and optimizer states, but do not implement that backend in the first NVMe work.

The runtime boundary is:

```text
AsymGEMM compute/policy
  -> placement/materialization interface
      -> LocalAsymNVMeBackend now
      -> DeepSpeedZeROBackend later, if multi-GPU becomes a real target
```

AsymGEMM kernels must still know the concrete tensor they consume. The abstraction hides previous residency, not required execution layout:

```text
Kernel/policy asks:  materialize LoRA group X on cuda, contiguous, dtype bf16
Backend does:        CPU cache lookup, NVMe pread, H2D staging, wait, stats
Kernel consumes:     a real CUDA tensor/view
```

Rules:

- One tensor role has one lifecycle owner in a run.
- Local mode can use current `.data` placeholder swapping, CPU homes, NVMe homes, CUDA staging slabs, and `AsymCPUAdamW`.
- Future DeepSpeed mode must not stack ZeRO ownership with the local `.data` placeholder/`AsymCPUAdamW` owner on the same trainable params. DeepSpeed would own ZeRO flat buffers, partition state, optimizer state, checkpointing, and `param.data`; AsymGEMM would only request materialized tensors/views.
- NVMe is block storage. No kernel computes from NVMe. Every use is `NVMe -> CPU pinned or GPU buffer -> wait -> compute`.
- Preserve current efficient compute shape: one layer/group H2D slab and the existing grouped GEMM. Never split experts into small GEMMs and never add per-expert I/O/kernel-launch loops.

DeepSpeed behavior to mirror at the low level:

- Use AIO/GDS-style handles, not `torch.save` as a performance path.
- Use pinned/page-aligned reusable transfer buffers.
- Do not swap tensors smaller than a meaningful threshold. DeepSpeed uses `MIN_AIO_BYTES = 1 MiB` and also raises the minimum to at least `aio.block_size`.
- Use block-size, queue-depth, single-submit, overlap-events, and intra-op-parallelism knobs.
- For parameter NVMe, keep a bounded CPU cache (`max_in_cpu` in DeepSpeed terms) and stage missing partitions before use.
- For optimizer NVMe, process large tiles and optionally pipeline read of the next tile and write of the previous tile.
- For prefetch, use known execution order/queues and bounded swap buffers. Do not build an adaptive hotness cache first; DeepSpeed's parameter path is primarily trace/order-prefetch plus reuse-distance release, not LFU/LRU hotness prediction from disk.

NVMe target roles for this plan:

- `lora_weights`: trainable adapter parameter homes. Use NVMe as backing storage, CPU as the staging/cache tier, and HBM only for the compute slab.
- `optimizer_state`: FP32 master weights plus Adam moments. These are persistent, large, and tile-friendly, so they are the strongest NVMe target.
- `base_weight`: optional frozen/base parameters. This is useful only when host memory is the blocker and the CPU cache can prevent blocking reads on hot layers.
- `gradient`: not a default NVMe target. Gradients are transient and backward-critical; keep the first implementation as a CPU flat grad buffer. Add gradient NVMe only as a later capacity mode, and only with large coalesced tile writes/reads during optimizer step, never per-parameter disk writes from autograd hooks.

Do not include activation NVMe in the main implementation path. Activation tensors are backward-critical and usually better handled by recompute/checkpointing or existing CPU activation offload. Revisit activation NVMe only if full e2e profiles prove CPU activation residency is the dominant remaining bottleneck after parameter and optimizer-state NVMe.

DeepSpeed pieces to reuse directly where practical:

```text
AsyncIOBuilder            required local AIO performance path
swap_in_tensors           use for full-file batched reads
swap_out_tensors          use for full-file batched writes
MIN_AIO_BYTES             use exact minimum threshold
AIO_ALIGNED_BYTES         use exact alignment base
SwapBuffer / SwapBufferPool
get_sized_buffer(s)
AsyncTensorSwapper        only later, and only behind a DeepSpeed-comm-safe wrapper
GDSBuilder                later optional GPU-direct experiment
DeepSpeed AIO knobs       same names/defaults where possible
DeepSpeedCPUAdam          only where it fits resident/tiled CPU tensors
```

Use `SwapBufferManager` directly only if it is safe in local single-process runs. In the vendored DeepSpeed code it calls `deepspeed.comm.get_rank()` during construction, which asserts unless DeepSpeed comm is initialized. If that blocks local AsymGEMM, implement a tiny `LocalSwapBufferManager` compatibility wrapper with the same `allocate`/`allocate_all`/`free` shape, but still use DeepSpeed's `SwapBufferPool`, `get_sized_buffer(s)`, constants, AIO handle, and swap functions. Do not reimplement the actual AIO submission path.

DeepSpeed compatibility rules:

- Direct-safe in local mode after import/build checks: `AsyncIOBuilder`, AIO handle `async_pread`/`async_pwrite`/`wait`, `swap_in_tensors`, `swap_out_tensors`, `MIN_AIO_BYTES`, `AIO_ALIGNED_BYTES`, `SwapBuffer`, `SwapBufferPool`, `get_sized_buffer(s)`.
- Guarded-safe only: `SwapBufferManager` and `AsyncTensorSwapper`, because they call `deepspeed.comm.get_rank()` in logging/stat paths. Use them only if DeepSpeed comm is initialized or a local wrapper suppresses the comm-dependent behavior.
- Not safe to reuse directly in local AsymGEMM mode: `OptimizerSwapper`, `PartitionedOptimizerSwapper`, `PipelinedOptimizerSwapper`, `AsyncPartitionedParameterSwapper`, and ZeRO coordinators. They assume `ds_id`, `ds_tensor`, ZeRO partition status, distributed rank state, and ZeRO checkpoint ownership.
- `DeepSpeedCPUAdam` is direct-safe only for resident CPU tensors. Do not mutate its internal state tensors into NVMe-backed placeholders. A paged optimizer must either use a separate tiled implementation or a proven resident-tile adapter.
- DeepSpeed `swap_in_tensors`/`swap_out_tensors` hardcode file offset `0`. Use them for whole-file tensor swaps. For optimizer tiles, call the AIO handle directly with explicit `file_offset` and validate offset reads/writes with DeepSpeed AIO unit tests before Stage 3.
- `GDSBuilder` is deferred. It is CUDA/GDS-specific and must not be required for the first local NVMe implementation.

DeepSpeed pieces not to reuse directly in local mode:

```text
ZeRO parameter coordinator
AsyncPartitionedParameterSwapper as-is
PartitionedOptimizerSwapper as-is
ZeRO optimizer/checkpoint ownership
```

Those high-level classes assume `ds_tensor`, `ds_id`, distributed partition state, all-gather, ZeRO release state, and ZeRO checkpoint ownership. Local AsymGEMM must not mix those owners with its current LoRA `.data` placeholder swapping and `AsymCPUAdamW` CPU-master ownership.

## Acceptance Gate

Each stage has an explicit correctness and latency gate. Memory-saving stages require full LF LoRA profiling. Stage 1 is isolated infrastructure, so its profile gate is a no-NVMe regression profile plus AIO/unit tests; it is not accepted from unit tests alone if the training scripts or imports changed.

Default acceptance for a memory-saving stage:

```text
candidate memory target decreases meaningfully
AND candidate forward/backward/e2e timing does not regress by more than 5%
AND correctness/profile artifacts are complete
```

Meaningful memory drop:

- HBM-target stages: peak allocated or reserved HBM drops by at least `max(5%, 1 GiB)`.
- Host-target stages: peak process RSS or pinned/resident host summary drops by at least `max(10%, 2 GiB)`.
- Optimizer-state paging target: host optimizer/master/state residency drops by at least `max(20%, 4 GiB)` on the Qwen3 profile.
- A change that leaves the target memory the same and increases latency is rejected.
- A change that saves only trivial memory is rejected even if latency is unchanged.
- LoRA NVMe homes are a host-memory/capacity feature on top of existing `asym_cpu_adamw_weight_offload=true`; they should not be counted as an HBM win unless the e2e memory report proves it.

Standard e2e baseline command shape:

```bash
OUTPUT_ROOT=profiling_nvme/stageN_baseline \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
WORKLOADS='4096|4|1' \
WARMUP_STEPS=5 \
MAX_STEPS=10 \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_EXTERNAL_MEMORY=true \
PROFILE_MEMORY_SNAPSHOT=false \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
ASYM_NVME_ENABLE=false \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
scripts/lf/profile_lora_lf.sh --gpus 0 --overwrite true
```

Use the same model/workload/steps for candidate runs and change only the stage-specific NVMe knobs.

Universal profile artifact gate:

- Required source artifacts after every LF profile: `source_profile.json`, `lat.md`, `memory.md`, `step_samples.csv`, `step_samples.json`, and `asym_cpu_adamw.csv` when `USE_ASYM_CPU_ADAMW=true`.
- Required with `PROFILE_MEMORY_BREAKDOWN=true`: `memory_breakdown.csv`, `memory_breakdown.md`, and `memory_breakdown_summary.json` or `memory_breakdown/memory_breakdown_summary.json`.
- Required for every NVMe candidate: top-level `asym_nvme` in `source_profile.json` and postprocessed `asym_nvme.csv`.
- Required timing fields: `trainer.timing.measured_e2e_step_milliseconds`, measured rows in `step_samples.rows`, and per-step `forward_milliseconds`, `backward_milliseconds`, `forward_backward_milliseconds`, and `step_milliseconds` after postprocess augmentation.
- Required memory fields: `memory.gpu.peak_allocated_hbm_bytes`, `memory.gpu.peak_reserved_hbm_bytes`, process RSS from `memory.process`, and the role-specific resident/logical/NVMe byte fields added by that stage.
- Required correctness fields: finite measured losses in `trainer.losses` or `step_samples.rows`, no startup validation errors, `trainer.timing.measured_steps >= 5`, and `config` values matching the requested NVMe enable/path/roles/cache/tile knobs.
- The comparison tool must fail if any required file or field is missing. A candidate is not accepted from console timing alone.

## Stage 0: Additive Contracts, Config, Profile Schema, Acceptance Tooling

Purpose: add all interface boundaries and reporting before changing runtime behavior. This prevents untracked NVMe behavior and gives every later stage a clear pass/fail gate.

Stage 0 is intentionally split into two implementation checkpoints:

- `Stage 0A`: additive contracts/no-op refactor. Add protocols, read-only iterators, summary aliases, and the placement interface. Existing `AsymCPUAdamW`, weight offload, grad clipping, state dicts, and trainer discovery must behave exactly as before.
- `Stage 0B`: config/profile/compare plumbing. Add CLI/env flags, profile schema, postprocess artifacts, and comparison tooling. `ASYM_NVME_ENABLE=true` may validate and report config, but it must not change tensor residency until later stages implement a role.

Do not start Stage 1 disk substrate until both Stage 0A and Stage 0B pass their validations.

Files to modify, grouped by checkpoint:

Stage 0A additive contracts/no-op refactor:

- add `asym_gemm/training/placement.py`
  - `TensorRole`
  - `TensorRef`
  - `MaterializedTensor`
  - `PrefetchHandle`
  - `PlacementBackend`
- add `asym_gemm/training/optimizer_contracts.py`
  - `LoRAParamRecord`
  - `WeightHomeCoordinator`
  - `AsymCPUOptimizerLike`
- `asym_gemm/training/cpu_adam.py`
  - `AsymCPUAdamW`
  - add `iter_lora_param_records`
  - keep existing public methods unchanged
- `asym_gemm/training/weight_offload.py`
  - `LoRAWeightOffloadCoordinator`
  - make current methods explicitly satisfy `WeightHomeCoordinator`
- add or extend tests:
  - new `tests/training/test_placement_backend_contract.py`
  - new `tests/training/test_asym_cpu_optimizer_contract.py`
  - new `tests/training/test_weight_home_coordinator_contract.py`
  - existing `tests/training/test_asym_cpu_adamw.py`
  - existing `tests/training/test_lora_weight_offload_generic.py`

Stage 0B config/profile/compare plumbing:

- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/hparams/finetuning_args.py`
  - class `FinetuningArguments`
- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/hparams/parser.py`
  - `_verify_asym_cpu_adamw_args`
  - main AsymGEMM validation block
- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/train/trainer_utils.py`
  - `_create_asym_cpu_adamw_optimizer`
- `scripts/lf/run_lf_lora_sft.sh`
- `scripts/lf/profile_lora_lf.sh`
- `scripts/lf/run_lf_profiled_train.py`
  - `_config_from_args`
  - add `_asym_nvme_summary_from_trace`
  - final source report dict
- `scripts/lf/postprocess_lf_profile_artifacts.py`
  - add NVMe CSV/markdown rows next to `_asym_cpu_adamw_rows`
- add `scripts/lf/compare_nvme_profiles.py`
- add or extend tests:
  - `tests/lf/test_asym_cpu_adamw_args.py`
  - `tests/test_lf_memory_breakdown.py`
  - new `tests/lf/test_nvme_profile_compare.py`

Contract/interface issues found in the current code and addressed by this stage:

- `AsymCPUAdamW` owns CPU masters, grad hooks, copyback, summary fields, and state dict format `asym_cpu_adamw_v1`. Stage 0 must not move those tensors or change the state dict.
- LlamaFactory's SFT trainer discovers the optimizer through `asym_cpu_adamw_grad_offload_enabled()` and then calls `asym_cpu_adamw_clip_grad_norm_()`. Any paged optimizer added later must keep that surface.
- `_create_asym_cpu_adamw_optimizer()` must keep the current construction order: create `AsymCPUAdamW`, then install `LoRAWeightOffloadCoordinator`, then attach the coordinator. Reordering loses full CUDA LoRA weights before CPU masters/homes are captured.
- Stage 0A must not switch optimizer classes, allocate NVMe buffers, build AIO handles, or change LoRA `.data` placeholder behavior. The only allowed `AsymCPUAdamW` changes are additive protocol methods and additive summary fields.
- `run_lf_profiled_train.py` currently reports `asym_cpu_adamw` from optimizer wrappers only. Add `asym_nvme` reporting there rather than relying on logs.
- `postprocess_lf_profile_artifacts.py` currently writes `asym_cpu_adamw.csv` but no NVMe artifact. Add `asym_nvme.csv` and make missing NVMe rows a candidate failure.
- `profile_lora_lf.sh` completion checks currently key on existing config fields. Add NVMe config to run labels and completion validation so `ASYM_NVME_ENABLE=false` and `true` cannot reuse the same source profile.
- DeepSpeed's `swap_in_tensors`/`swap_out_tensors` are whole-file helpers with `file_offset=0`; Stage 3 must use direct AIO handle calls for optimizer tiles.
- After Stage 0B, later stages should not need LF CLI/parser/run-label/postprocess refactors. Role implementations should report through `asym_nvme_summary()` and the generic Stage 0B `stats_by_role` schema. If a later stage discovers a missing required metric, treat that as a Stage 0B contract bug, fix the schema first, and rerun the Stage 0B no-change profile before continuing.

Implementation order inside Stage 0:

```text
0A.1 optimizer and weight-home contracts
0A.2 existing AsymCPUAdamW protocol conformance, no behavior change
0A.3 placement/materialization contract
0A.4 no-NVMe contract regression tests/profile
0B.1 LlamaFactory dataclass/parser flags
0B.2 profile scripts and completion checks
0B.3 source profile + postprocess NVMe artifacts
0B.4 compare tool
0B.5 dry-run and no-NVMe profile validation
```

Code changes:

1. Add the CPU optimizer and weight-home contracts before changing any runtime behavior. This is a compatibility refactor, not an NVMe implementation. The current `AsymCPUAdamW` and `LoRAWeightOffloadCoordinator` must keep all existing behavior, state dicts, summary keys, and module hooks.

```python
# asym_gemm/training/optimizer_contracts.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Protocol

import torch

@dataclass(frozen=True)
class LoRAParamRecord:
    name: str
    aliases: tuple[str, ...]
    cuda_param: torch.nn.Parameter
    shape: tuple[int, ...]
    numel: int
    dtype: torch.dtype
    flat_offset: int | None = None

class WeightHomeCoordinator(Protocol):
    def is_registered(self, param: torch.nn.Parameter) -> bool: ...
    def gather_group(self, module: Any) -> None: ...
    def release(self, param: torch.nn.Parameter) -> None: ...
    def release_group(self, module: Any) -> None: ...
    def refresh_home_from_master(self, param: torch.nn.Parameter, master_fp32: torch.Tensor) -> None: ...
    def summary(self) -> dict[str, Any]: ...

class AsymCPUOptimizerLike(Protocol):
    def attach_weight_offload_coordinator(self, coordinator: WeightHomeCoordinator) -> None: ...
    def iter_lora_param_records(self) -> Iterator[LoRAParamRecord]: ...
    def asym_cpu_adamw_grad_offload_enabled(self) -> bool: ...
    def asym_cpu_adamw_grad_buffers(self) -> list[torch.Tensor]: ...
    def asym_cpu_adamw_clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        chunk_elements: int = 8_388_608,
    ) -> tuple[torch.Tensor, dict[str, Any]]: ...
    def asym_cpu_adamw_summary(self) -> dict[str, Any]: ...
```

2. Add a read-only record iterator to the existing optimizer. Do not move masters, gradients, state, hooks, clipping, or checkpoint logic in this stage:

```python
# asym_gemm/training/cpu_adam.py
from .optimizer_contracts import LoRAParamRecord, WeightHomeCoordinator

class AsymCPUAdamW(torch.optim.Optimizer):
    def attach_weight_offload_coordinator(self, coordinator: WeightHomeCoordinator) -> None:
        self._coordinator = coordinator
        self.weight_offload = True

    def iter_lora_param_records(self):
        for mapping in self._mappings:
            yield LoRAParamRecord(
                name=mapping.name,
                aliases=tuple(mapping.aliases),
                cuda_param=mapping.cuda_param,
                shape=tuple(mapping.cuda_param.shape),
                numel=int(mapping.cuda_param.numel() or mapping.cpu_param.numel()),
                dtype=mapping.model_dtype,
                flat_offset=None,
            )
```

The `numel` fallback is required because weight-offloaded CUDA params may be 0-size placeholders while the CPU master still holds the logical tensor.

Add only additive summary aliases needed by later comparison tooling; keep all existing keys unchanged:

```python
def asym_cpu_adamw_summary(self) -> dict[str, Any]:
    ...
    optimizer_state_resident_bytes = master_bytes + optimizer_state_cpu_bytes + grad_buffer_bytes
    return {
        ...,
        "cpu_master_bytes": int(master_bytes),
        "optimizer_state_cpu_bytes": int(optimizer_state_cpu_bytes),
        "grad_offload_buffer_bytes": int(grad_buffer_bytes),
        # Comparable Stage 3 memory target for non-paged and paged optimizers.
        "optimizer_state_resident_bytes": int(optimizer_state_resident_bytes),
        "optimizer_paged": False,
    }
```

3. Make the weight coordinator satisfy the protocol without changing module call sites:

```python
# asym_gemm/training/weight_offload.py
from .optimizer_contracts import WeightHomeCoordinator

class LoRAWeightOffloadCoordinator:
    # Existing methods already match the protocol:
    # is_registered, gather_group, release, release_group,
    # refresh_home_from_master, summary.
    pass
```

4. Keep current construction order. The optimizer must still be created before installing weight offload so CPU masters are captured from full CUDA LoRA params:

```python
optimizer = AsymCPUAdamW(...)
coordinator = LoRAWeightOffloadCoordinator(...)
install_lora_weight_offload(model, coordinator)
optimizer.attach_weight_offload_coordinator(coordinator)
```

Do not switch to `PagedAsymCPUAdamW` in this stage. Later Stage 3 will add it as a separate class implementing `AsymCPUOptimizerLike`. The existing `AsymCPUAdamW` remains the default unless `ASYM_NVME_ROLES` includes `optimizer_state`.

5. Add the replaceable placement contract now, before wiring any runtime role to NVMe:

```python
# asym_gemm/training/placement.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch

TensorRole = Literal[
    "lora_weights",
    "optimizer_state",
    "base_weight",
    "gradient",
]

@dataclass(frozen=True)
class TensorRef:
    stable_id: str
    role: TensorRole
    shape: tuple[int, ...]
    dtype: torch.dtype
    logical_layout: str = "contiguous"
    required_alignment: int = 1
    mutable: bool = False

@dataclass
class MaterializedTensor:
    ref: TensorRef
    tensor: torch.Tensor
    device: torch.device
    layout: str
    owner_token: Any | None = None
    must_release: bool = False

@dataclass
class PrefetchHandle:
    ref: TensorRef
    target_device: str
    target_layout: str
    token: Any

class PlacementBackend(Protocol):
    def has_role(self, role: TensorRole) -> bool: ...

    def materialize(
        self,
        ref: TensorRef,
        *,
        target_device: Literal["cpu", "cpu_pinned", "cuda"],
        target_layout: str = "contiguous",
        stream: torch.cuda.Stream | None = None,
    ) -> MaterializedTensor: ...

    def prefetch(
        self,
        ref: TensorRef,
        *,
        target_device: Literal["cpu", "cpu_pinned", "cuda"],
        target_layout: str = "contiguous",
        stream: torch.cuda.Stream | None = None,
    ) -> PrefetchHandle | None: ...

    def wait(self, handle: PrefetchHandle) -> MaterializedTensor | None: ...
    def release(self, materialized: MaterializedTensor) -> None: ...
    def mark_dirty(self, ref: TensorRef) -> None: ...
    def flush(self, ref: TensorRef | None = None) -> None: ...
    def stats(self) -> dict[str, Any]: ...
```

Local helper methods such as `register_tensor_ref`, `write_cpu`, or `release_cpu_cache` may exist on `LocalAsymNVMeBackend`, but AsymGEMM compute/model code should call the protocol-shaped methods wherever possible. This keeps `DeepSpeedZeROBackend` replaceable later.

Stage 0A validation:

```bash
.venv/bin/python -m pytest \
  tests/training/test_asym_cpu_optimizer_contract.py \
  tests/training/test_weight_home_coordinator_contract.py \
  tests/training/test_placement_backend_contract.py \
  tests/training/test_asym_cpu_adamw.py \
  tests/training/test_lora_weight_offload_generic.py \
  -q
```

Before applying Stage 0A, capture a pre-contract no-NVMe profile with `OUTPUT_ROOT=profiling_nvme/stage0_pre_contract_no_nvme` and the same command shape below. After Stage 0A, run the no-NVMe LF profile:

```bash
OUTPUT_ROOT=profiling_nvme/stage0a_contract_no_nvme \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
WORKLOADS='4096|4|1' \
WARMUP_STEPS=5 MAX_STEPS=10 \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_EXTERNAL_MEMORY=true \
PROFILE_MEMORY_SNAPSHOT=false \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
ASYM_NVME_ENABLE=false \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
scripts/lf/profile_lora_lf.sh --gpus 0 --overwrite true
```

Compare pre/post Stage 0A as a no-change gate:

```bash
BASE=$(find profiling_nvme/stage0_pre_contract_no_nvme -path '*/source_profile.json' -print -quit)
CAND=$(find profiling_nvme/stage0a_contract_no_nvme -path '*/source_profile.json' -print -quit)
.venv/bin/python scripts/lf/compare_nvme_profiles.py \
  --baseline "$BASE" \
  --candidate "$CAND" \
  --target no_change \
  --memory-metric memory.gpu.peak_allocated_hbm_bytes \
  --extra-memory-metric memory.gpu.peak_reserved_hbm_bytes \
  --extra-memory-metric memory.process.rss_peak_bytes \
  --max-memory-drift-bytes $((512*1024*1024)) \
  --timing-metric trainer.timing.measured_e2e_step_milliseconds \
  --extra-timing-metric step_samples.forward_milliseconds \
  --extra-timing-metric step_samples.backward_milliseconds \
  --max-latency-regression-pct 2 \
  --expect-nvme-enabled false
```

Stage 0A acceptance is stricter than memory-saving stages:

- Existing tests pass.
- State dict format remains `asym_cpu_adamw_v1`.
- LlamaFactory `_asym_cpu_adamw_grad_offload_optimizer` still finds the optimizer through wrappers.
- Grad clipping/norm path still uses CPU grad buffers and reports the same summary shape.
- Existing `asym_cpu_adamw_summary()` keys remain present.
- Additive summary aliases such as `optimizer_state_resident_bytes` and `optimizer_paged=false` are present and do not remove or rename existing fields.
- Existing weight-offload gather/release hooks still use one group slab and 0-size CUDA placeholders at rest.
- No meaningful memory or latency change versus the pre-refactor no-NVMe profile. This checkpoint is rejected if it changes behavior.

6. Add LF dataclass args:

```python
class FinetuningArguments(...):
    asym_nvme_enable: bool = field(default=False)
    asym_nvme_path: str | None = field(default=None)
    asym_nvme_roles: str = field(default="")
    asym_nvme_cpu_cache_bytes: int = field(default=0)
    asym_nvme_min_swappable_bytes: int = field(default=1048576)
    asym_nvme_transfer_buffer_bytes: int = field(default=134217728)
    asym_nvme_transfer_buffer_count: int = field(default=5)
    asym_nvme_prefetch_depth: int = field(default=1)
    asym_nvme_aio_block_size: int = field(default=1048576)
    asym_nvme_aio_queue_depth: int = field(default=8)
    asym_nvme_aio_intra_op_parallelism: int = field(default=1)
    asym_nvme_aio_single_submit: bool = field(default=False)
    asym_nvme_aio_overlap_events: bool = field(default=True)
    asym_nvme_require_aio: bool = field(default=True)
    asym_nvme_optimizer_tile_bytes: int = field(default=268435456)
    asym_nvme_optimizer_pipeline_read: bool = field(default=False)
    asym_nvme_optimizer_pipeline_write: bool = field(default=False)
```

Allowed initial roles:

```text
lora_weights
optimizer_state
base_weight
```

Reserve `gradient` for a later explicit capacity mode. Do not accept `activation_spill` as an implementation role in this plan.

7. Add parser validation:

```python
def _parse_nvme_roles(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}

def _verify_asym_nvme_args(model_args, training_args, finetuning_args):
    if not finetuning_args.asym_nvme_enable:
        return
    roles = _parse_nvme_roles(finetuning_args.asym_nvme_roles)
    allowed = {"lora_weights", "optimizer_state", "base_weight"}
    unknown = roles - allowed
    if unknown:
        raise ValueError(f"unknown ASYM NVMe roles: {sorted(unknown)}")
    if not roles:
        raise ValueError("`asym_nvme_enable=true` requires at least one role.")
    if not finetuning_args.asym_nvme_path:
        raise ValueError("`asym_nvme_path` is required when ASYM NVMe is enabled.")
    if training_args.parallel_mode != ParallelMode.NOT_PARALLEL:
        raise ValueError("local AsymGEMM NVMe backend requires single-process single-device training.")
    if training_args.deepspeed is not None or is_deepspeed_zero3_enabled():
        raise ValueError("local AsymGEMM NVMe backend cannot be combined with DeepSpeed/ZeRO ownership.")
    if {"lora_weights", "optimizer_state"} & roles:
        if not finetuning_args.use_asym_cpu_adamw:
            raise ValueError("LoRA/optimizer NVMe roles require `use_asym_cpu_adamw=true`.")
        if not finetuning_args.asym_cpu_adamw_grad_offload:
            raise ValueError("LoRA/optimizer NVMe roles require grad offload for the current owner model.")
    if "lora_weights" in roles and not finetuning_args.asym_cpu_adamw_weight_offload:
        raise ValueError("`lora_weights` NVMe role requires `asym_cpu_adamw_weight_offload=true`.")
    if finetuning_args.asym_nvme_min_swappable_bytes < 1048576:
        raise ValueError("ASYM NVMe minimum swappable size must be >= 1 MiB.")
```

Call this from the same AsymGEMM validation path that already calls `_verify_asym_cpu_adamw_args`.

8. Add script plumbing in both `scripts/lf/run_lf_lora_sft.sh` and `scripts/lf/profile_lora_lf.sh`. `profile_lora_lf.sh` does not forward arbitrary unknown flags; it must parse these NVMe knobs, include them in run-directory labels, pass them through the training-job environment, and include them in `job_profile_complete` validation.

```bash
ASYM_NVME_ENABLE=${ASYM_NVME_ENABLE:-false}
ASYM_NVME_PATH=${ASYM_NVME_PATH:-}
ASYM_NVME_ROLES=${ASYM_NVME_ROLES:-}
ASYM_NVME_CPU_CACHE_BYTES=${ASYM_NVME_CPU_CACHE_BYTES:-0}
ASYM_NVME_MIN_SWAPPABLE_BYTES=${ASYM_NVME_MIN_SWAPPABLE_BYTES:-1048576}
ASYM_NVME_TRANSFER_BUFFER_BYTES=${ASYM_NVME_TRANSFER_BUFFER_BYTES:-134217728}
ASYM_NVME_TRANSFER_BUFFER_COUNT=${ASYM_NVME_TRANSFER_BUFFER_COUNT:-5}
ASYM_NVME_PREFETCH_DEPTH=${ASYM_NVME_PREFETCH_DEPTH:-1}
ASYM_NVME_AIO_BLOCK_SIZE=${ASYM_NVME_AIO_BLOCK_SIZE:-1048576}
ASYM_NVME_AIO_QUEUE_DEPTH=${ASYM_NVME_AIO_QUEUE_DEPTH:-8}
ASYM_NVME_AIO_INTRA_OP_PARALLELISM=${ASYM_NVME_AIO_INTRA_OP_PARALLELISM:-1}
ASYM_NVME_AIO_SINGLE_SUBMIT=${ASYM_NVME_AIO_SINGLE_SUBMIT:-false}
ASYM_NVME_AIO_OVERLAP_EVENTS=${ASYM_NVME_AIO_OVERLAP_EVENTS:-true}
ASYM_NVME_REQUIRE_AIO=${ASYM_NVME_REQUIRE_AIO:-true}
ASYM_NVME_OPTIMIZER_TILE_BYTES=${ASYM_NVME_OPTIMIZER_TILE_BYTES:-268435456}
ASYM_NVME_OPTIMIZER_PIPELINE_READ=${ASYM_NVME_OPTIMIZER_PIPELINE_READ:-false}
ASYM_NVME_OPTIMIZER_PIPELINE_WRITE=${ASYM_NVME_OPTIMIZER_PIPELINE_WRITE:-false}
...

# In run_lf_lora_sft.sh, use the existing bool_string helper.
ASYM_NVME_ENABLE="$(bool_string ASYM_NVME_ENABLE "${ASYM_NVME_ENABLE}")"
ASYM_NVME_AIO_SINGLE_SUBMIT="$(bool_string ASYM_NVME_AIO_SINGLE_SUBMIT "${ASYM_NVME_AIO_SINGLE_SUBMIT}")"
ASYM_NVME_AIO_OVERLAP_EVENTS="$(bool_string ASYM_NVME_AIO_OVERLAP_EVENTS "${ASYM_NVME_AIO_OVERLAP_EVENTS}")"
ASYM_NVME_REQUIRE_AIO="$(bool_string ASYM_NVME_REQUIRE_AIO "${ASYM_NVME_REQUIRE_AIO}")"
ASYM_NVME_OPTIMIZER_PIPELINE_READ="$(bool_string ASYM_NVME_OPTIMIZER_PIPELINE_READ "${ASYM_NVME_OPTIMIZER_PIPELINE_READ}")"
ASYM_NVME_OPTIMIZER_PIPELINE_WRITE="$(bool_string ASYM_NVME_OPTIMIZER_PIPELINE_WRITE "${ASYM_NVME_OPTIMIZER_PIPELINE_WRITE}")"

# In profile_lora_lf.sh, use the existing bool_value helper instead:
ASYM_NVME_ENABLE="$(bool_value "${ASYM_NVME_ENABLE}")"
ASYM_NVME_AIO_SINGLE_SUBMIT="$(bool_value "${ASYM_NVME_AIO_SINGLE_SUBMIT}")"
ASYM_NVME_AIO_OVERLAP_EVENTS="$(bool_value "${ASYM_NVME_AIO_OVERLAP_EVENTS}")"
ASYM_NVME_REQUIRE_AIO="$(bool_value "${ASYM_NVME_REQUIRE_AIO}")"
ASYM_NVME_OPTIMIZER_PIPELINE_READ="$(bool_value "${ASYM_NVME_OPTIMIZER_PIPELINE_READ}")"
ASYM_NVME_OPTIMIZER_PIPELINE_WRITE="$(bool_value "${ASYM_NVME_OPTIMIZER_PIPELINE_WRITE}")"

# Only run_lf_lora_sft.sh appends LlamaFactory CLI args.
CMD_ARGS+=(--asym_nvme_enable "${ASYM_NVME_ENABLE}")
CMD_ARGS+=(--asym_nvme_path "${ASYM_NVME_PATH}")
CMD_ARGS+=(--asym_nvme_roles "${ASYM_NVME_ROLES}")
CMD_ARGS+=(--asym_nvme_cpu_cache_bytes "${ASYM_NVME_CPU_CACHE_BYTES}")
CMD_ARGS+=(--asym_nvme_min_swappable_bytes "${ASYM_NVME_MIN_SWAPPABLE_BYTES}")
CMD_ARGS+=(--asym_nvme_transfer_buffer_bytes "${ASYM_NVME_TRANSFER_BUFFER_BYTES}")
CMD_ARGS+=(--asym_nvme_transfer_buffer_count "${ASYM_NVME_TRANSFER_BUFFER_COUNT}")
CMD_ARGS+=(--asym_nvme_prefetch_depth "${ASYM_NVME_PREFETCH_DEPTH}")
CMD_ARGS+=(--asym_nvme_aio_block_size "${ASYM_NVME_AIO_BLOCK_SIZE}")
CMD_ARGS+=(--asym_nvme_aio_queue_depth "${ASYM_NVME_AIO_QUEUE_DEPTH}")
CMD_ARGS+=(--asym_nvme_aio_intra_op_parallelism "${ASYM_NVME_AIO_INTRA_OP_PARALLELISM}")
CMD_ARGS+=(--asym_nvme_aio_single_submit "${ASYM_NVME_AIO_SINGLE_SUBMIT}")
CMD_ARGS+=(--asym_nvme_aio_overlap_events "${ASYM_NVME_AIO_OVERLAP_EVENTS}")
CMD_ARGS+=(--asym_nvme_require_aio "${ASYM_NVME_REQUIRE_AIO}")
CMD_ARGS+=(--asym_nvme_optimizer_tile_bytes "${ASYM_NVME_OPTIMIZER_TILE_BYTES}")
CMD_ARGS+=(--asym_nvme_optimizer_pipeline_read "${ASYM_NVME_OPTIMIZER_PIPELINE_READ}")
CMD_ARGS+=(--asym_nvme_optimizer_pipeline_write "${ASYM_NVME_OPTIMIZER_PIPELINE_WRITE}")
...

ASYM_GEMM_LF_CONFIG_ASYM_NVME_ENABLE="${ASYM_NVME_ENABLE}"
ASYM_GEMM_LF_CONFIG_ASYM_NVME_PATH="${ASYM_NVME_PATH}"
ASYM_GEMM_LF_CONFIG_ASYM_NVME_ROLES="${ASYM_NVME_ROLES}"
ASYM_GEMM_LF_CONFIG_ASYM_NVME_CPU_CACHE_BYTES="${ASYM_NVME_CPU_CACHE_BYTES}"
ASYM_GEMM_LF_CONFIG_ASYM_NVME_OPTIMIZER_TILE_BYTES="${ASYM_NVME_OPTIMIZER_TILE_BYTES}"
```

`profile_lora_lf.sh` should not append `CMD_ARGS`; it should parse/sanitize the same environment knobs, add them to the child training-job environment, add them to `ASYM_GEMM_LF_CONFIG_*`, and validate them in completion checks.

Add the same keys to `run_lf_profiled_train.py::_config_from_args()` so `source_profile.json.config` records the exact candidate settings. Extend `profile_lora_lf.sh`'s existing `job_profile_complete` Python validation to compare `asym_nvme_enable`, `asym_nvme_roles`, cache bytes, prefetch depth, and optimizer tile/pipeline flags against the requested environment.

Concrete `profile_lora_lf.sh` completion-check changes:

```bash
job_profile_complete() {
  ...
  local expected_nvme_enable="${17:-}"
  local expected_nvme_roles="${18:-}"
  local expected_nvme_cpu_cache_bytes="${19:-}"
  local expected_nvme_prefetch_depth="${20:-}"
  local expected_nvme_optimizer_tile_bytes="${21:-}"
  local expected_nvme_optimizer_pipeline_read="${22:-}"
  local expected_nvme_optimizer_pipeline_write="${23:-}"

  existing_profile_complete \
    ... \
    "${expected_liger_loss}" \
    "${expected_nvme_enable}" \
    "${expected_nvme_roles}" \
    "${expected_nvme_cpu_cache_bytes}" \
    "${expected_nvme_prefetch_depth}" \
    "${expected_nvme_optimizer_tile_bytes}" \
    "${expected_nvme_optimizer_pipeline_read}" \
    "${expected_nvme_optimizer_pipeline_write}" || return 1
}
```

Inside the embedded Python validator:

```python
expected_nvme_enable = sys.argv[26] if len(sys.argv) > 26 else ""
expected_nvme_roles = sys.argv[27] if len(sys.argv) > 27 else ""
expected_nvme_cpu_cache_bytes = sys.argv[28] if len(sys.argv) > 28 else ""
expected_nvme_prefetch_depth = sys.argv[29] if len(sys.argv) > 29 else ""
expected_nvme_optimizer_tile_bytes = sys.argv[30] if len(sys.argv) > 30 else ""
expected_nvme_optimizer_pipeline_read = sys.argv[31] if len(sys.argv) > 31 else ""
expected_nvme_optimizer_pipeline_write = sys.argv[32] if len(sys.argv) > 32 else ""

def normalize_roles(value):
    return ",".join(sorted(part.strip() for part in str(value or "").split(",") if part.strip()))

if expected_nvme_enable:
    actual = normalize_bool(config.get("asym_nvme_enable", False))
    wanted = normalize_bool(expected_nvme_enable)
    if actual != wanted:
        raise SystemExit(f"profile asym_nvme_enable mismatch: expected {wanted}, got {actual}")
if normalize_bool(expected_nvme_enable) == "true":
    if normalize_roles(config.get("asym_nvme_roles")) != normalize_roles(expected_nvme_roles):
        raise SystemExit("profile asym_nvme_roles mismatch")
    require_int_config_any(("asym_nvme_cpu_cache_bytes",), expected_nvme_cpu_cache_bytes, "asym_nvme_cpu_cache_bytes")
    require_int_config_any(("asym_nvme_prefetch_depth",), expected_nvme_prefetch_depth, "asym_nvme_prefetch_depth")
    require_int_config_any(("asym_nvme_optimizer_tile_bytes",), expected_nvme_optimizer_tile_bytes, "asym_nvme_optimizer_tile_bytes")
    for key, expected in (
        ("asym_nvme_optimizer_pipeline_read", expected_nvme_optimizer_pipeline_read),
        ("asym_nvme_optimizer_pipeline_write", expected_nvme_optimizer_pipeline_write),
    ):
        if expected:
            actual = normalize_bool(config.get(key))
            wanted = normalize_bool(expected)
            if actual != wanted:
                raise SystemExit(f"profile {key} mismatch: expected {wanted}, got {actual}")
```

Update every existing `job_profile_complete ... "${liger_loss}"` call in `run_job()` to append:

```bash
"${ASYM_NVME_ENABLE}" \
"${ASYM_NVME_ROLES}" \
"${ASYM_NVME_CPU_CACHE_BYTES}" \
"${ASYM_NVME_PREFETCH_DEPTH}" \
"${ASYM_NVME_OPTIMIZER_TILE_BYTES}" \
"${ASYM_NVME_OPTIMIZER_PIPELINE_READ}" \
"${ASYM_NVME_OPTIMIZER_PIPELINE_WRITE}"
```

Also pass the same environment into the subshell that launches `run_lf_lora_sft.sh`, alongside the existing `ASYM_GEMM_LF_CONFIG_*` values, so `_config_from_args()` records the requested candidate settings in `source_profile.json.config`.

In `profile_lora_lf.sh`, add NVMe fields to run labels and completion checks so baseline and candidate profiles cannot be confused:

```bash
nvme_tag="nvmeoff"
if [[ "$(bool_value "${ASYM_NVME_ENABLE}")" == "true" ]]; then
  nvme_tag="nvme_${ASYM_NVME_ROLES//,/+}"
fi
run_dir_name="${run_dir_name}_${nvme_tag}"
```

9. Add source-profile summary hook:

```python
def _asym_nvme_summary_from_trace(trace_handle: Any | None) -> dict[str, Any]:
    summaries = []
    optimizer = getattr(trace_handle, "optimizer", None) or getattr(trace_handle, "prepared_optimizer", None)
    for candidate in _walk_optimizer_wrappers(optimizer):
        fn = getattr(candidate, "asym_nvme_summary", None)
        if callable(fn):
            summaries.append(fn())
    model = getattr(trace_handle, "model", None)
    fn = getattr(model, "asym_nvme_summary", None)
    if callable(fn):
        summaries.append(fn())
    return merge_nvme_summaries(summaries) if summaries else {"enabled": False}
```

Add `"asym_nvme": _asym_nvme_summary_from_trace(trace_handle)` to the final report.

The summary schema must be stable from Stage 0 onward:

```python
{
    "enabled": bool,
    "backend": "local" | "deepspeed_zero" | "none",
    "roles": ["lora_weights", "optimizer_state"],
    "path": "/local_nvme/...",
    "cpu_cache_budget_bytes": int,
    "effective_min_swappable_bytes": int,
    "aio": {
        "available": bool,
        "block_size": int,
        "queue_depth": int,
        "single_submit": bool,
        "overlap_events": bool,
        "intra_op_parallelism": int,
        "using_local_swap_buffer_manager": bool,
    },
    "stats_by_role": {
        "lora_weights": {
            "logical_bytes": int,
            "stored_bytes": int,
            "resident_cpu_bytes": int,
            "cache_budget_bytes": int,
            "cache_peak_bytes": int,
            "cache_hits": int,
            "cache_misses": int,
            "bytes_read": int,
            "bytes_written": int,
            "read_ops": int,
            "write_ops": int,
            "wait_read_ms": float,
            "wait_write_ms": float,
        },
    },
}
```

For disabled runs, emit exactly `{"enabled": False, "backend": "none", "roles": [], "stats_by_role": {}}` so no-NVMe comparisons can assert `--expect-nvme-enabled false`.

Postprocess rows:

```python
def _asym_nvme_rows(profile):
    nvme = profile.get("asym_nvme", {})
    if not isinstance(nvme, dict):
        return []
    rows = []
    base = {k: v for k, v in nvme.items() if not isinstance(v, (dict, list))}
    for role, stats in sorted((nvme.get("stats_by_role") or {}).items()):
        row = dict(base)
        row["role"] = role
        if isinstance(stats, dict):
            row.update(stats)
        rows.append(row)
    return rows or [base]
```

Write these rows to `asym_nvme.csv`. Add a short NVMe block to `memory.md` or `summary.md` that lists roles, resident CPU bytes, NVMe stored bytes, read/write bytes, and wait time.

10. Add comparison tool:

```python
def main():
    baseline = load_profile(args.baseline)
    candidate = load_profile(args.candidate)
    target = args.target  # hbm, host, optimizer, lora_host, no_change

    base_mem = extract_metric(baseline, args.memory_metric)
    cand_mem = extract_metric(candidate, args.memory_metric)
    base_ms = extract_timing(baseline, args.timing_metric)
    cand_ms = extract_timing(candidate, args.timing_metric)

    mem_drop = base_mem - cand_mem
    mem_drop_pct = mem_drop / max(base_mem, 1) * 100.0
    latency_regression_pct = (cand_ms - base_ms) / max(base_ms, 1e-9) * 100.0

    assert candidate_is_complete(
        candidate,
        required_files=[
            "source_profile.json",
            "lat.md",
            "memory.md",
            "step_samples.csv",
            "step_samples.json",
        ],
        required_profile_fields=[
            "trainer.timing.measured_e2e_step_milliseconds",
            "memory.gpu.peak_allocated_hbm_bytes",
            "memory.gpu.peak_reserved_hbm_bytes",
            "step_samples.rows",
        ],
    )
    assert measured_steps(candidate) >= args.min_measured_steps
    assert measured_losses_are_finite(candidate)
    assert config_matches_requested_nvme(candidate, args)
    if args.expect_nvme_enabled is not None:
        assert bool(candidate.get("asym_nvme", {}).get("enabled")) is args.expect_nvme_enabled
    if target != "no_change":
        assert mem_drop >= args.min_memory_drop_bytes
        assert mem_drop_pct >= args.min_memory_drop_pct
    else:
        assert abs(mem_drop) <= args.max_memory_drift_bytes
    for metric in args.extra_memory_metric:
        assert abs(extract_metric(candidate, metric) - extract_metric(baseline, metric)) <= args.max_memory_drift_bytes
    assert latency_regression_pct <= args.max_latency_regression_pct
    for metric in args.extra_timing_metric:
        assert timing_regression_pct(baseline, candidate, metric) <= args.max_latency_regression_pct
    if args.require_nvme_role:
        assert args.require_nvme_role in candidate["asym_nvme"]["roles"]
```

Comparison metric resolution:

- Dotted JSON paths such as `memory.gpu.peak_allocated_hbm_bytes` and `asym_cpu_adamw.optimizer_state_resident_bytes` read from `source_profile.json`.
- `step_samples.<field>` reads measured, non-warmup rows from postprocessed `step_samples.csv` if present, otherwise `source_profile.json.step_samples.rows`; compare the median value.
- Required artifact checks infer the output directory as `Path(candidate_source_profile).parent`.
- `target=no_change` does not require memory drop; it enforces memory drift and latency regression bounds only.

Stage 0B validation:

```bash
bash -n scripts/lf/run_lf_lora_sft.sh scripts/lf/profile_lora_lf.sh

.venv/bin/python -m pytest \
  tests/lf/test_asym_cpu_adamw_args.py \
  tests/test_lf_memory_breakdown.py \
  tests/lf/test_nvme_profile_compare.py \
  -q

ASYM_NVME_ENABLE=true \
ASYM_NVME_PATH=/tmp/asym_nvme_dryrun \
ASYM_NVME_ROLES=lora_weights \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
WORKLOADS='4096|4|1' \
scripts/lf/profile_lora_lf.sh --dry-run --gpus 0
```

Run the no-NVMe profile again after Stage 0B plumbing. This proves that config/profile changes still do not affect runtime behavior:

```bash
OUTPUT_ROOT=profiling_nvme/stage0b_plumbing_no_nvme \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
WORKLOADS='4096|4|1' \
WARMUP_STEPS=5 MAX_STEPS=10 \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_EXTERNAL_MEMORY=true \
PROFILE_MEMORY_SNAPSHOT=false \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
ASYM_NVME_ENABLE=false \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
scripts/lf/profile_lora_lf.sh --gpus 0 --overwrite true
```

Compare Stage 0B against the accepted Stage 0A profile:

```bash
BASE=$(find profiling_nvme/stage0a_contract_no_nvme -path '*/source_profile.json' -print -quit)
CAND=$(find profiling_nvme/stage0b_plumbing_no_nvme -path '*/source_profile.json' -print -quit)
.venv/bin/python scripts/lf/compare_nvme_profiles.py \
  --baseline "$BASE" \
  --candidate "$CAND" \
  --target no_change \
  --memory-metric memory.gpu.peak_allocated_hbm_bytes \
  --extra-memory-metric memory.gpu.peak_reserved_hbm_bytes \
  --extra-memory-metric memory.process.rss_peak_bytes \
  --max-memory-drift-bytes $((512*1024*1024)) \
  --timing-metric trainer.timing.measured_e2e_step_milliseconds \
  --extra-timing-metric step_samples.forward_milliseconds \
  --extra-timing-metric step_samples.backward_milliseconds \
  --max-latency-regression-pct 2 \
  --expect-nvme-enabled false
```

Acceptance:

- All config values appear in `source_profile.json.config`.
- Invalid role combinations fail during argument validation.
- Baseline profile still completes with `ASYM_NVME_ENABLE=false`.
- Postprocess writes `step_samples.csv`, `asym_cpu_adamw.csv`, and, for synthetic NVMe profiles, `asym_nvme.csv`.
- `compare_nvme_profiles.py` validates required files, measured steps, finite losses, e2e timing, forward/backward step-sample timing, HBM/RSS memory fields, and requested NVMe role/config values.
- `compare_nvme_profiles.py` can reject synthetic profiles with no memory drop or excessive latency.
- No meaningful memory or latency change versus the accepted Stage 0A no-NVMe profile.

Risks to watch:

- LlamaFactory argument names must stay snake_case CLI names matching dataclass fields.
- Run labels must include NVMe role/cache knobs or stale source profiles can be reused incorrectly.
- Do not make `ASYM_NVME_ENABLE=true` imply runtime behavior until later stages implement the role.

## Stage 1: Local Disk Substrate

Purpose: implement reusable NVMe tensor storage using DeepSpeed's low-level AIO path, with pinned/aligned transfer buffers and stats. This stage is accepted by unit/I/O tests plus a no-NVMe LF regression profile because no training path should use the substrate yet.

Files to modify:

- add `asym_gemm/training/disk_offload.py`
  - implement `LocalAsymNVMeBackend(PlacementBackend)`
- update `asym_gemm/training/__init__.py` if exports are used
- add `tests/training/test_disk_offload.py`

Classes/functions to implement:

```text
DiskOffloadConfig
DiskTensorHandle
DiskIOFuture
DeepSpeedAIOHandles
LocalSwapBufferManager only if DeepSpeed SwapBufferManager cannot run without DS comm
DiskTensorStore
DiskOffloadStats
LocalAsymNVMeBackend
get_local_nvme_backend()
```

Code changes:

1. Configuration, DeepSpeed imports, and alignment:

```python
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import AsyncIOBuilder
from deepspeed.runtime.swap_tensor.utils import (
    AIO_ALIGNED_BYTES,
    MIN_AIO_BYTES,
    SwapBufferManager,
    SwapBufferPool,
    get_sized_buffer,
    get_sized_buffers,
    swap_in_tensors,
    swap_out_tensors,
)

@dataclass(frozen=True)
class DiskOffloadConfig:
    enabled: bool
    path: Path
    roles: frozenset[str]
    min_swappable_bytes: int = 1048576
    cpu_cache_bytes: int = 0
    transfer_buffer_bytes: int = 134217728
    transfer_buffer_count: int = 5
    prefetch_depth: int = 1
    aio_block_size: int = 1048576
    aio_queue_depth: int = 8
    aio_intra_op_parallelism: int = 1
    aio_single_submit: bool = False
    aio_overlap_events: bool = True
    require_aio: bool = True

    @property
    def effective_min_swappable_bytes(self) -> int:
        return max(MIN_AIO_BYTES, self.min_swappable_bytes, self.aio_block_size)

    @property
    def aligned_bytes(self) -> int:
        return AIO_ALIGNED_BYTES * max(1, self.aio_intra_op_parallelism)

def aligned_numel(numel: int, dtype: torch.dtype, cfg: DiskOffloadConfig) -> int:
    elem = torch.empty((), dtype=dtype).element_size()
    nbytes = numel * elem
    padded = round_up(nbytes, cfg.aligned_bytes)
    return padded // elem
```

2. Handles and stats:

```python
@dataclass
class DiskTensorHandle:
    id: str
    role: str
    path: Path
    shape: tuple[int, ...]
    dtype: torch.dtype
    numel: int
    aligned_numel: int
    logical_nbytes: int
    stored_nbytes: int
    dirty: bool = False
    resident_cpu_nbytes: int = 0

@dataclass
class DiskIOFuture:
    kind: Literal["read", "write"]
    handle: DiskTensorHandle
    buffers: list[torch.Tensor]
    op_count: int = 1
    release_buffers_on_wait: bool = True

@dataclass
class DiskOffloadStats:
    bytes_read: int = 0
    bytes_written: int = 0
    read_ops: int = 0
    write_ops: int = 0
    wait_read_ms: float = 0.0
    wait_write_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_resident_bytes: int = 0
    cache_peak_bytes: int = 0
```

3. AIO handle creation. Use DeepSpeed's builder and preserve DeepSpeed knob names:

```python
@dataclass
class DeepSpeedAIOHandles:
    read: object | None
    write: object | None
    available: bool = True

def _build_aio_handles(cfg: DiskOffloadConfig):
    try:
        factory = AsyncIOBuilder().load(verbose=False).aio_handle
        # The vendored DeepSpeed tests call aio_handle positionally. Keep the same call
        # shape unless the local build is proven to support keyword arguments.
        read = factory(
            cfg.aio_block_size,
            cfg.aio_queue_depth,
            cfg.aio_single_submit,
            cfg.aio_overlap_events,
            cfg.aio_intra_op_parallelism,
        )
        write = factory(
            cfg.aio_block_size,
            cfg.aio_queue_depth,
            cfg.aio_single_submit,
            cfg.aio_overlap_events,
            cfg.aio_intra_op_parallelism,
        )
        return DeepSpeedAIOHandles(read=read, write=write)
    except Exception:
        if cfg.require_aio:
            raise
        return DeepSpeedAIOHandles(read=None, write=None, available=False)  # debug-only synchronous fallback
```

4. Swap buffer management. Prefer DeepSpeed's `SwapBufferManager`; fall back only for the local single-process DeepSpeed-comm issue:

```python
class LocalSwapBufferManager:
    """Compatibility shim for local AsymGEMM when DeepSpeed comm is not initialized."""
    def __init__(self, num_elems, count, dtype):
        self.num_elems = num_elems
        self.count = count
        self.dtype = dtype
        self.all_buffers = [
            get_accelerator().pin_memory(torch.zeros(num_elems, device="cpu", dtype=dtype), align_bytes=0)
            for _ in range(count)
        ]
        self.free_ids = list(range(count))
        self.used: dict[int, int] = {}

    def allocate(self, num_elems, count, dtype):
        assert dtype == self.dtype
        assert num_elems <= self.num_elems
        if count > len(self.free_ids):
            return None
        ids = self.free_ids[-count:]
        self.free_ids = self.free_ids[:-count]
        bufs = [self.all_buffers[i].narrow(0, 0, num_elems) for i in ids]
        for i, b in zip(ids, bufs):
            self.used[id(b)] = i
        return bufs

    def allocate_all(self, num_elems, dtype):
        return self.allocate(num_elems=num_elems, count=len(self.free_ids), dtype=dtype)

    def free(self, buffers):
        for b in buffers:
            i = self.used.pop(id(b))
            self.free_ids.append(i)

def build_swap_buffer_manager(num_elems, count, dtype):
    try:
        return SwapBufferManager(num_elems=num_elems, count=count, dtype=dtype)
    except AssertionError as exc:
        if "DeepSpeed backend not set" not in str(exc):
            raise
        return LocalSwapBufferManager(num_elems=num_elems, count=count, dtype=dtype)
```

`LocalSwapBufferManager` is intentionally tiny. It exists only because DeepSpeed's manager prints through `deepspeed.comm`. Do not add independent packing or async-I/O behavior here; use `SwapBufferPool`, `swap_in_tensors`, and `swap_out_tensors` for that.

5. Tensor store:

```python
class DiskTensorStore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.aio = _build_aio_handles(cfg)
        self.buffer_managers: dict[torch.dtype, object] = {}

    def _buffer_manager(self, dtype):
        mgr = self.buffer_managers.get(dtype)
        if mgr is None:
            max_numel = bytes_to_numel(self.cfg.transfer_buffer_bytes, dtype=dtype)
            mgr = build_swap_buffer_manager(max_numel, self.cfg.transfer_buffer_count, dtype)
            self.buffer_managers[dtype] = mgr
        return mgr

    def register_tensor(self, *, stable_id, role, tensor) -> DiskTensorHandle:
        handle = DiskTensorHandle(
            id=sanitize(stable_id),
            role=role,
            path=self.root / role / f"{sanitize(stable_id)}.tensor.swp",
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            numel=tensor.numel(),
            aligned_numel=aligned_numel(tensor.numel(), tensor.dtype, self.cfg),
            logical_nbytes=tensor.numel() * tensor.element_size(),
            stored_nbytes=aligned_numel(...) * tensor.element_size(),
        )
        return handle

    def should_swap(self, tensor_or_numel, dtype) -> bool:
        return numel * element_size(dtype) >= self.cfg.effective_min_swappable_bytes

    def write(self, handle, tensor, *, async_op=False) -> DiskIOFuture:
        if not self.aio.available:
            return self._debug_sync_write(handle, tensor)
        src = tensor.detach().reshape(-1)
        mgr = self._buffer_manager(handle.dtype)
        buffers = mgr.allocate(num_elems=handle.aligned_numel, count=1, dtype=handle.dtype)
        if buffers is None:
            raise RuntimeError("NVMe write ran out of DeepSpeed swap buffers; increase transfer_buffer_count")
        buf = buffers[0]
        buf.narrow(0, 0, handle.numel).copy_(src, non_blocking=False)
        if handle.aligned_numel > handle.numel:
            buf.narrow(0, handle.numel, handle.aligned_numel - handle.numel).zero_()
        swap_out_tensors(self.aio.write, [buf], [str(handle.path)])
        future = DiskIOFuture("write", handle, buffers, release_buffers_on_wait=True)
        if not async_op:
            self.wait(future)
        return future

    def read(self, handle, *, async_op=False) -> tuple[torch.Tensor, DiskIOFuture]:
        if not self.aio.available:
            return self._debug_sync_read(handle)
        mgr = self._buffer_manager(handle.dtype)
        buffers = mgr.allocate(num_elems=handle.aligned_numel, count=1, dtype=handle.dtype)
        if buffers is None:
            raise RuntimeError("NVMe read ran out of DeepSpeed swap buffers; increase transfer_buffer_count")
        dst = buffers[0]
        swap_in_tensors(self.aio.read, [dst], [str(handle.path)])
        # Read buffers back the returned tensor. Do not free them in wait().
        # The backend releases them after cache eviction or MaterializedTensor.release().
        future = DiskIOFuture("read", handle, buffers, release_buffers_on_wait=False)
        if async_op:
            return dst.narrow(0, 0, handle.numel).view(handle.shape), future
        self.wait(future)
        return dst.narrow(0, 0, handle.numel).view(handle.shape), future

    def wait(self, future: DiskIOFuture):
        started = time.perf_counter()
        aio = self.aio.write if future.kind == "write" else self.aio.read
        completed = aio.wait()
        assert completed == future.op_count
        if future.release_buffers_on_wait:
            self.release_buffers(future)
        update_stats(...)

    def release_buffers(self, future: DiskIOFuture):
        self._buffer_manager(future.handle.dtype).free(future.buffers)
```

Full tensor reads/writes must use DeepSpeed `swap_in_tensors`/`swap_out_tensors`. Offset slice reads/writes for optimizer tiles may call the DeepSpeed AIO handle directly because the vendored helper hardcodes offset `0`; keep that direct-call code small and tested.
Read-buffer lifetime is explicit: if a returned CPU tensor aliases a DeepSpeed swap buffer, that buffer is owned by the materialized tensor or CPU cache until release/eviction. Never return a read buffer to the free list immediately after `aio.wait()`.

6. Local backend and CPU cache. `DiskTensorHandle` stays private to the local backend. Public users hold `TensorRef`.

```python
class LocalAsymNVMeBackend:
    def __init__(self, cfg):
        self.store = DiskTensorStore(cfg)
        self._handles_by_ref: dict[str, DiskTensorHandle] = {}
        # Policy-neutral resident CPU buffers. Role owners decide what to admit/evict.
        self.cache: dict[str, torch.Tensor] = {}
        self.cache_owners: dict[str, DiskIOFuture] = {}
        self._uncached_read_owners: dict[str, DiskIOFuture] = {}
        self.inflight_reads: dict[str, tuple[DiskIOFuture, torch.Tensor]] = {}
        self.cache_bytes = 0

    def has_role(self, role: TensorRole) -> bool:
        return role in self.cfg.roles

    def register_tensor_ref(self, *, stable_id, role, tensor, mutable=False) -> TensorRef:
        ref = TensorRef(
            stable_id=stable_id,
            role=role,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            logical_layout="contiguous",
            required_alignment=self.cfg.aligned_bytes,
            mutable=mutable,
        )
        handle = self.store.register_tensor(stable_id=stable_id, role=role, tensor=tensor)
        self._handles_by_ref[ref.stable_id] = handle
        return ref

    def _handle(self, ref: TensorRef) -> DiskTensorHandle:
        try:
            return self._handles_by_ref[ref.stable_id]
        except KeyError as exc:
            raise RuntimeError(f"unknown tensor ref {ref.stable_id}") from exc

    def materialize(self, ref, *, target_device, target_layout="contiguous", stream=None) -> MaterializedTensor:
        if target_device not in {"cpu", "cpu_pinned"}:
            raise ValueError("local disk backend materializes NVMe tensors to CPU first; caller stages CUDA slabs")
        handle = self._handle(ref)
        tensor = self.materialize_cpu_handle(handle, cache=True)
        if target_device == "cpu_pinned" and torch.cuda.is_available() and not tensor.is_pinned():
            tensor = tensor.pin_memory()
        if target_layout == "contiguous" and not tensor.is_contiguous():
            tensor = tensor.contiguous()
        owner = self._uncached_read_owners.pop(handle.id, None)
        return MaterializedTensor(
            ref=ref,
            tensor=tensor.view(ref.shape),
            device=tensor.device,
            layout=target_layout,
            owner_token=owner,
            must_release=owner is not None,
        )

    def materialize_cpu_handle(self, handle, *, cache=True) -> torch.Tensor:
        cached = self.cache.get(handle.id)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached.narrow(0, 0, handle.numel).view(handle.shape)
        self.stats.cache_misses += 1
        tensor, owner = self.store.read(handle, async_op=False)
        if cache and self.cfg.cpu_cache_bytes > 0:
            self._admit(handle, tensor.reshape(-1), owner=owner)
        else:
            self._uncached_read_owners[handle.id] = owner
        return tensor

    def write_cpu(self, ref_or_handle, tensor, *, async_op=False):
        handle = self._handle(ref_or_handle) if isinstance(ref_or_handle, TensorRef) else ref_or_handle
        future = self.store.write(handle, tensor, async_op=async_op)
        handle.dirty = False
        return future

    def prefetch(self, ref, *, target_device, target_layout="contiguous", stream=None) -> PrefetchHandle | None:
        if target_device not in {"cpu", "cpu_pinned"}:
            return None
        handle = self._handle(ref)
        future = self.prefetch_cpu_handle(handle)
        return None if future is None else PrefetchHandle(ref=ref, target_device=target_device, target_layout=target_layout, token=future)

    def prefetch_cpu_handle(self, handle) -> DiskIOFuture | None:
        if handle.id in self.cache or handle.id in self.inflight_reads:
            return None
        tensor, future = self.store.read(handle, async_op=True)
        self.inflight_reads[handle.id] = (future, tensor)
        return future

    def wait(self, prefetch: PrefetchHandle) -> MaterializedTensor | None:
        handle = self._handle(prefetch.ref)
        tensor = self.wait_prefetch_handle(handle)
        return MaterializedTensor(ref=prefetch.ref, tensor=tensor.view(prefetch.ref.shape), device=tensor.device, layout=prefetch.target_layout)

    def wait_prefetch_handle(self, handle):
        future, tensor = self.inflight_reads.pop(handle.id)
        self.store.wait(future)
        self._admit(handle, tensor.reshape(-1), owner=future)
        return self.cache[handle.id].view(handle.shape)

    def _admit(self, handle, tensor, *, owner):
        old_owner = self.cache_owners.pop(handle.id, None)
        if old_owner is not None:
            self.store.release_buffers(old_owner)
        self.cache[handle.id] = tensor
        self.cache_owners[handle.id] = owner
        self.cache_bytes += handle.logical_nbytes
        # Role-specific coordinator chooses victims; the substrate only provides release.

    def release_cached(self, handle):
        tensor = self.cache.pop(handle.id, None)
        owner = self.cache_owners.pop(handle.id, None)
        if owner is not None:
            self.store.release_buffers(owner)
        if tensor is not None:
            self.cache_bytes -= handle.logical_nbytes

    def release(self, materialized):
        if not materialized.must_release:
            return
        owner = materialized.owner_token
        if owner is not None:
            self.store.release_buffers(owner)
```

Validation before Stage 2:

```bash
.venv/bin/python -m pytest tests/training/test_disk_offload.py -q

.venv/bin/python - <<'PY'
from deepspeed.ops.op_builder import AsyncIOBuilder
h = AsyncIOBuilder().load(verbose=False).aio_handle(1048576, 8, False, True, 1)
print("async_io ok", h.get_block_size(), h.get_queue_depth())
PY
```

No-NVMe regression profile after the substrate is imported/plumbed:

```bash
OUTPUT_ROOT=profiling_nvme/stage1_no_nvme_regression \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
WORKLOADS='4096|4|1' \
WARMUP_STEPS=5 MAX_STEPS=10 \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_EXTERNAL_MEMORY=true \
PROFILE_MEMORY_SNAPSHOT=false \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
ASYM_NVME_ENABLE=false \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
scripts/lf/profile_lora_lf.sh --gpus 0 --overwrite true
```

Compare against the accepted Stage 0B no-NVMe profile:

```bash
BASE=$(find profiling_nvme/stage0b_plumbing_no_nvme -path '*/source_profile.json' -print -quit)
CAND=$(find profiling_nvme/stage1_no_nvme_regression -path '*/source_profile.json' -print -quit)
.venv/bin/python scripts/lf/compare_nvme_profiles.py \
  --baseline "$BASE" \
  --candidate "$CAND" \
  --target no_change \
  --memory-metric memory.gpu.peak_allocated_hbm_bytes \
  --extra-memory-metric memory.gpu.peak_reserved_hbm_bytes \
  --extra-memory-metric memory.process.rss_peak_bytes \
  --min-memory-drop-bytes 0 \
  --min-memory-drop-pct 0 \
  --max-memory-drift-bytes $((512*1024*1024)) \
  --timing-metric trainer.timing.measured_e2e_step_milliseconds \
  --extra-timing-metric step_samples.forward_milliseconds \
  --extra-timing-metric step_samples.backward_milliseconds \
  --max-latency-regression-pct 2 \
  --expect-nvme-enabled false
```

Required tests:

- `LocalAsymNVMeBackend` satisfies `PlacementBackend` contract for register/materialize/prefetch/wait/flush/stats.
- Local single-process construction works when `deepspeed.comm` is not initialized. If DeepSpeed `SwapBufferManager` asserts, the local compatibility wrapper is used and reported in stats.
- BF16/FP32 roundtrip for tensors below, equal to, and above 1 MiB.
- Misaligned numel pads for I/O but returns the exact logical shape.
- Direct AIO handle slice I/O works: write two aligned tiles at different `file_offset` values, read each tile back, and verify no cross-tile corruption. This proves Stage 3 can use offset reads/writes instead of rewriting whole optimizer-state files.
- CPU cache byte accounting is correct; role-specific code, not the substrate, decides which tensors to evict.
- Async prefetch followed by wait returns correct data.
- `require_aio=true` fails clearly when DeepSpeed async I/O is unavailable.
- No `torch.save`, `torch.load`, NumPy memmap, Python file-object tensor serialization, or ad hoc per-tensor binary loops are used on the performance path.

Acceptance:

- Unit tests pass.
- AIO smoke test succeeds on the target machine.
- No-NVMe LF profile artifacts are complete and show no meaningful HBM/RSS change and no more than 2% e2e/forward/backward latency regression versus Stage 0B.
- `asym_nvme.enabled=false` in the no-NVMe source profile; importing the substrate must not allocate transfer buffers or build AIO handles unless NVMe is enabled.

Risks to watch:

- BF16 raw writes must use tensor AIO, not NumPy conversion.
- Pinned allocation may fail under host pressure; record and report fallback.
- DeepSpeed `SwapBufferManager` has a `deepspeed.comm` logging dependency. The fallback wrapper is allowed only for buffer ownership; AIO submission and buffer packing must still use DeepSpeed utilities.
- Debug fallback is for correctness tests only and must not be used in performance acceptance.

## Stage 2: NVMe-Backed LoRA Weight Homes

Purpose: move current `LoRAWeightOffloadCoordinator` CPU home slabs to a bounded CPU cache backed by NVMe. This targets host pinned/RSS capacity, not additional HBM savings beyond current weight offload.

Files to modify:

- `asym_gemm/training/weight_offload.py`
  - `_LayerGroup`
  - `LoRAWeightOffloadCoordinator.__init__`
  - `register_group`
  - `gather_group`
  - `refresh_home_from_master`
  - `summary`
  - add `begin_optimizer_refresh`, `flush_dirty_groups`, `_ensure_group_home`, `_prefetch_next_group`, `_evict_group_homes_to_budget`
- `asym_gemm/training/cpu_adam.py`
  - `AsymCPUAdamW.step`
  - `asym_cpu_adamw_summary`
- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/train/trainer_utils.py`
  - `_create_asym_cpu_adamw_optimizer`
- add tests:
  - `tests/training/test_lora_weight_nvme_offload.py`
  - extend `tests/test_lf_memory_breakdown.py`

No Stage 0B script/profile refactor is expected here; use the existing `asym_nvme_summary()` and `stats_by_role` hooks.

Code changes:

1. Extend group state:

```python
class _LayerGroup:
    __slots__ = (...,
        "home_ref",
        "home_resident",
        "dirty",
        "pending_read",
        "pending_write",
    )

    def __init__(self):
        ...
        self.home_ref: TensorRef | None = None
        self.home_resident: bool = True
        self.dirty: bool = False
        self.pending_read = None
        self.pending_write = None
```

2. Coordinator constructor:

```python
class LoRAWeightOffloadCoordinator:
    def __init__(
        self,
        *,
        pin_memory=True,
        persistence_threshold_numel=_DEFAULT_PERSIST_NUMEL,
        nvme_backend: LocalAsymNVMeBackend | None = None,
        nvme_cpu_cache_bytes: int | None = None,
        nvme_prefetch_depth: int = 1,
    ):
        self._nvme = nvme_backend if nvme_backend and nvme_backend.has_role("lora_weights") else None
        self._dirty_groups: set[_LayerGroup] = set()
        self._groups_by_order: list[_LayerGroup] = []
        self._resident_groups: set[int] = set()
        self._trace_build: list[_LayerGroup] = []
        self._trace: list[_LayerGroup] = []
        self._trace_cursor: int = 0
        self._trace_frozen: bool = False
        self._trace_disabled: bool = False
        self._reuse_window_groups = max(1, int(nvme_prefetch_depth))
        self._forced_near_reuse_evictions = 0
        self._optimizer_refresh_depth = 0
```

3. Register groups:

```python
def register_group(...):
    ...
    home = build_current_pinned_cpu_home()
    group.home = home

    if self._nvme and self._nvme.should_swap(home, role="lora_weights"):
        ref = self._nvme.register_tensor_ref(
            stable_id=f"lora_weight/{group.group_name}",
            role="lora_weights",
            tensor=home,
            mutable=True,
        )
        self._nvme.write_cpu(ref, home, async_op=False)
        group.home_ref = ref

        if not self._nvme.admit_existing_cpu(ref, home):
            group.home = None
            group.home_resident = False
        else:
            group.home_resident = True
```

Groups below the 1 MiB/effective min threshold remain CPU-resident as today.

4. Materialize one CPU home per group:

```python
def _ensure_group_home(self, group: _LayerGroup) -> torch.Tensor:
    if group.home is not None:
        self._resident_groups.add(id(group))
        return group.home
    if group.home_ref is None:
        raise RuntimeError("group has neither CPU home nor backend tensor ref")
    materialized = self._nvme.materialize(group.home_ref, target_device="cpu_pinned", target_layout="contiguous")
    home = materialized.tensor.reshape(-1)
    if home.dtype != group.dtype or home.numel() != group.total_numel:
        raise RuntimeError("materialized LoRA home mismatch")
    group.home = home
    group.home_resident = True
    self._resident_groups.add(id(group))
    self._evict_group_homes_to_budget()
    return home
```

The coordinator must clear `group.home` for evicted groups. A backend-only cache eviction is not enough because the group object would still keep a Python reference to the CPU tensor. Eviction is schedule/reuse-distance based, not adaptive hotness/LRU.

```python
def _resident_home_bytes(self) -> int:
    total = 0
    for group in self._groups:
        if group.home is not None and group.home_ref is not None:
            total += group.home.numel() * group.home.element_size()
    return total

def _record_group_access(self, group: _LayerGroup) -> None:
    if self._trace_disabled:
        return
    if not self._trace_frozen:
        self._trace_build.append(group)
        return
    if not self._trace:
        return
    n = len(self._trace)
    for delta in range(n):
        pos = (self._trace_cursor + delta) % n
        if self._trace[pos] is group:
            self._trace_cursor = pos
            return
    # Dynamic group order: stop prefetching rather than guessing from hotness.
    self._trace_disabled = True
    self._trace = []
    self._trace_build.clear()

def seal_step_trace_if_needed(self) -> None:
    # Called by AsymCPUAdamW.step() after the first full forward/backward has completed.
    if self._trace_disabled or self._trace_frozen or not self._trace_build:
        return
    self._trace = list(self._trace_build)
    self._trace_build.clear()
    self._trace_frozen = True
    self._trace_cursor = 0

def _next_use_distance(self, group: _LayerGroup) -> int:
    if not self._trace:
        return 10**12
    n = len(self._trace)
    for delta in range(1, n + 1):
        if self._trace[(self._trace_cursor + delta) % n] is group:
            return delta
    return 10**12

def _eviction_candidates_by_reuse_distance(self) -> list[_LayerGroup]:
    candidates = [
        group for group in self._groups
        if id(group) in self._resident_groups
        and group.home is not None
        and group.buf is None
        and group.live <= 0
        and group.pending_read is None
    ]
    # Farthest next use first. Keep near-reuse groups resident when possible.
    return sorted(candidates, key=self._next_use_distance, reverse=True)

def _evict_group_homes_to_budget(self):
    if self._nvme is None or self._nvme.cfg.cpu_cache_bytes <= 0:
        return
    budget = self._nvme.cfg.cpu_cache_bytes
    while self._resident_home_bytes() > budget:
        candidates = self._eviction_candidates_by_reuse_distance()
        far_candidates = [
            group for group in candidates
            if self._next_use_distance(group) > self._reuse_window_groups
        ]
        if far_candidates:
            victim = far_candidates[0]
        elif candidates:
            victim = candidates[0]  # forced eviction; still farthest known reuse
            self._forced_near_reuse_evictions += 1
        else:
            break
        if victim.dirty:
            self._nvme.write_cpu(victim.home_ref, victim.home, async_op=False)
            victim.dirty = False
        self._nvme.release_cpu_cache(victim.home_ref)
        victim.home = None
        victim.home_resident = False
        self._resident_groups.discard(id(victim))
```

5. Gather remains one H2D per group:

```python
def gather_group(self, module):
    group = self._group_of_module.get(id(module))
    if group is None or group.buf is not None:
        return
    self._record_group_access(group)
    home = self._ensure_group_home(group)
    self._prefetch_next_group(group)

    buf = self._take_slab(group.total_numel, group.dtype, group.device)
    buf.copy_(home, non_blocking=home.is_pinned())  # still ONE H2D for the whole group
    group.buf = buf
    group.live = len(group.params)
    for index, param in enumerate(group.params):
        start = group.offsets[index]
        param.data = buf[start:start + group.numels[index]].view(group.shapes[index])
```

6. Prefetch follows the recorded group-use trace, not adaptive hotness:

```python
def _prefetch_next_group(self, current: _LayerGroup) -> None:
    if self._nvme is None or self._nvme_prefetch_depth <= 0 or not self._trace:
        return
    issued = 0
    n = len(self._trace)
    for delta in range(1, n + 1):
        if issued >= self._nvme_prefetch_depth:
            break
        group = self._trace[(self._trace_cursor + delta) % n]
        if group.home is not None or group.home_ref is None:
            continue
        if group.pending_read is not None:
            continue
        handle = self._nvme.prefetch(group.home_ref, target_device="cpu_pinned", target_layout="contiguous")
        if handle is not None:
            group.pending_read = handle
            issued += 1

def _ensure_group_home(self, group):
    if group.home is not None:
        ...
    if group.pending_read is not None:
        materialized = self._nvme.wait(group.pending_read)
        group.pending_read = None
    else:
        materialized = self._nvme.materialize(group.home_ref, target_device="cpu_pinned", target_layout="contiguous")
    ...
```

Do not scan experts or route decisions to decide prefetch. The next-use order is the actual group gather trace observed from the first complete train step. Before that trace is sealed, prefetch is disabled and the implementation falls back to synchronous materialization. This mirrors DeepSpeed's trace/order-prefetch model more closely than a hotness cache.

7. Refresh after optimizer step without per-bank disk writes:

```python
def begin_optimizer_refresh(self):
    self._optimizer_refresh_depth += 1

def refresh_home_from_master(self, param, master_fp32):
    group = self._group_of_key.get(id(param))
    if group is None:
        return
    home = self._ensure_group_home(group)
    index = group.bank_index[id(param)]
    start = group.offsets[index]
    home[start:start + group.numels[index]].copy_(master_fp32.reshape(-1))
    if group.home_ref is not None:
        group.dirty = True
        self._dirty_groups.add(group)

def flush_dirty_groups(self):
    if self._optimizer_refresh_depth:
        self._optimizer_refresh_depth -= 1
    if self._optimizer_refresh_depth:
        return
    futures = []
    for group in list(self._dirty_groups):
        if group.home_ref is None or group.home is None:
            continue
        futures.append(self._nvme.write_cpu(group.home_ref, group.home, async_op=True))
        group.dirty = False
    self._nvme.wait_all(futures)
    self._dirty_groups.clear()
    self._evict_group_homes_to_budget()
```

8. Hook optimizer step:

```python
def step(self, closure=None):
    ...
    copyback_start = time.perf_counter()
    if self.weight_offload and self._coordinator is not None:
        seal_trace = getattr(self._coordinator, "seal_step_trace_if_needed", None)
        if callable(seal_trace):
            seal_trace()
        begin = getattr(self._coordinator, "begin_optimizer_refresh", None)
        if callable(begin):
            begin()
    try:
        for mapping in self._mappings:
            if mapping.last_had_grad:
                self._copy_master_to_compute_param(mapping)
                copyback_count += 1
    finally:
        flush = getattr(self._coordinator, "flush_dirty_groups", None)
        if callable(flush):
            flush()
    self._last_weight_copyback_ms = ...
```

9. Summary fields:

```python
def summary(self):
    return {
        ...,
        "weight_offload_home_bytes_logical": total_logical_home_bytes,
        "weight_offload_cpu_home_resident_bytes": resident_cpu_home_bytes,
        "weight_offload_pinned_home_resident_bytes": resident_pinned_bytes,
        "weight_offload_nvme_home_bytes": nvme_stored_bytes,
        "weight_offload_nvme_group_count": nvme_group_count,
        "weight_offload_nvme_cache_budget_bytes": self._nvme.cfg.cpu_cache_bytes if self._nvme else 0,
        "weight_offload_nvme_trace_frozen": bool(self._trace_frozen),
        "weight_offload_nvme_trace_disabled": bool(self._trace_disabled),
        "weight_offload_nvme_trace_length": len(self._trace),
        "weight_offload_nvme_prefetch_depth": int(self._nvme_prefetch_depth),
        "weight_offload_nvme_forced_near_reuse_evictions": int(self._forced_near_reuse_evictions),
        "weight_offload_nvme_stats": self._nvme.summary() if self._nvme else {},
    }
```

Validation before Stage 3:

```bash
.venv/bin/python -m pytest \
  tests/training/test_lora_weight_nvme_offload.py \
  tests/training/test_asym_cpu_adamw.py \
  tests/test_lf_memory_breakdown.py \
  -q
```

Baseline:

```bash
OUTPUT_ROOT=profiling_nvme/stage2_baseline \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
WORKLOADS='4096|4|1' \
WARMUP_STEPS=5 MAX_STEPS=10 \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_EXTERNAL_MEMORY=true \
PROFILE_MEMORY_SNAPSHOT=false \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
ASYM_NVME_ENABLE=false \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
scripts/lf/profile_lora_lf.sh --gpus 0 --overwrite true
```

Candidate:

```bash
rm -rf /local_nvme/asymgemm_stage2
OUTPUT_ROOT=profiling_nvme/stage2_lora_nvme \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
WORKLOADS='4096|4|1' \
WARMUP_STEPS=5 MAX_STEPS=10 \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_EXTERNAL_MEMORY=true \
PROFILE_MEMORY_SNAPSHOT=false \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
ASYM_NVME_ENABLE=true \
ASYM_NVME_PATH=/local_nvme/asymgemm_stage2 \
ASYM_NVME_ROLES=lora_weights \
ASYM_NVME_CPU_CACHE_BYTES=$((512*1024*1024)) \
ASYM_NVME_PREFETCH_DEPTH=1 \
ASYM_NVME_REQUIRE_AIO=true \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
scripts/lf/profile_lora_lf.sh --gpus 0 --overwrite true
```

Compare:

```bash
BASE=$(find profiling_nvme/stage2_baseline -path '*/source_profile.json' -print -quit)
CAND=$(find profiling_nvme/stage2_lora_nvme -path '*/source_profile.json' -print -quit)
.venv/bin/python scripts/lf/compare_nvme_profiles.py \
  --baseline "$BASE" \
  --candidate "$CAND" \
  --target lora_host \
  --memory-metric asym_cpu_adamw.weight_offload_pinned_home_resident_bytes \
  --min-memory-drop-bytes $((2*1024*1024*1024)) \
  --min-memory-drop-pct 10 \
  --timing-metric trainer.timing.measured_e2e_step_milliseconds \
  --extra-timing-metric step_samples.forward_milliseconds \
  --extra-timing-metric step_samples.backward_milliseconds \
  --max-latency-regression-pct 5 \
  --require-nvme-role lora_weights
```

Acceptance:

- Required artifacts exist: `source_profile.json`, `lat.md`, `memory.md`, `step_samples.csv`, `step_samples.json`, `asym_cpu_adamw.csv`, `asym_nvme.csv`, and memory-breakdown artifacts.
- `source_profile.json.config.asym_nvme_enable=true`, `asym_nvme.roles` contains `lora_weights`, and `asym_nvme.stats_by_role.lora_weights` exists.
- `weight_offload_nvme_group_count > 0`.
- `weight_offload_nvme_trace_frozen=true` after warmup and `weight_offload_nvme_trace_length > 0`.
- `weight_offload_nvme_trace_disabled=false`; if it is true, this run did not exercise the optimized prefetch policy.
- `bytes_read`, `bytes_written`, `read_ops`, and `write_ops` are nonzero for role `lora_weights`.
- Prefetch reads appear after the first full step; first-step synchronous reads are allowed while the trace is being built.
- Resident pinned/CPU home bytes obey the cache budget plus active/prefetch buffers.
- Host memory decreases meaningfully.
- HBM peak does not increase materially.
- E2E timing and `step_samples` forward/backward timing regression are each <= 5%.
- Measured losses remain finite and measured step count is at least 5.
- Grouped GEMM count and H2D staging pattern are unchanged: one group slab, no per-expert GEMMs.
- `weight_offload_nvme_forced_near_reuse_evictions` should be zero or rare; frequent forced near-reuse evictions mean the CPU cache budget is too small for this policy.

Risks to watch:

- Small CPU cache can turn every forward/backward into a blocking NVMe miss. If wait time dominates, reject or raise cache/prefetch budget.
- This stage may not improve HBM compared with existing CPU-home weight offload. That is expected; do not claim it as default HBM improvement.
- Dirty flush must be group-batched after optimizer copyback. Per-bank disk writes are rejected.
- If the observed group trace changes across steps because of dynamic module paths, disable prefetch for that run and fall back to synchronous materialization; do not guess with hotness scoring.

## Stage 3: Local Optimizer-State Paging

Purpose: reduce host memory from CPU masters and Adam state by paging optimizer data in large tiles. This is the first stage that can make very large LoRA ranks feasible when host RAM is also the bottleneck.

Files to modify:

- add `asym_gemm/training/paged_cpu_adam.py`
  - `PagedAsymCPUAdamW`
  - `_PagedParamMapping`
  - `_OptimizerTile`
  - `PagedAdamStateStore`
- `asym_gemm/training/cpu_adam.py`
  - keep `AsymCPUAdamW` unchanged for the default path
  - optionally factor shared grad-hook helpers only if it avoids duplication
- `asym_gemm/training/weight_offload.py`
  - add `refresh_home_from_master_tile` for flat tile updates if Stage 2 is active
- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/train/trainer_utils.py`
  - select `PagedAsymCPUAdamW` when `ASYM_NVME_ROLES` includes `optimizer_state`
- add tests:
  - `tests/training/test_paged_cpu_adam.py`
  - `tests/training/test_paged_cpu_adam_state_dict.py`

No Stage 0B profile/postprocess refactor is expected here; add optimizer role metrics through `asym_cpu_adamw_summary()` and `asym_nvme_summary()`.

Code changes:

1. Use a separate optimizer class. Do not mutate `DeepSpeedCPUAdam` state tensors behind its back in the first implementation; it expects resident CPU tensors and owns its state dict.

```python
class PagedAsymCPUAdamW(torch.optim.Optimizer):
    def __init__(..., nvme_backend, optimizer_tile_bytes=268435456, pipeline_read=False, pipeline_write=False):
        validate_lora_cuda_params()
        super().__init__([{"params": cuda_params, ...}], defaults)
        self._flat = FlatLoRAIndex.from_named_params(trainable_lora)
        self._store = PagedAdamStateStore(nvme_backend, self._flat)
        self._grad_flat_buffer = allocate_pinned_cpu_float32(self._flat.total_numel)
        self._tiles = plan_tiles(self._flat.total_numel, optimizer_tile_bytes, min_tile_bytes=nvme_min)
        self.pipeline_read = pipeline_read
        self.pipeline_write = pipeline_write
        self._register_grad_offload_hooks()
```

2. Flat parameter index:

```python
@dataclass
class _PagedParamMapping:
    name: str
    cuda_param: torch.nn.Parameter
    offset: int
    numel: int
    shape: tuple[int, ...]
    model_dtype: torch.dtype

class FlatLoRAIndex:
    @classmethod
    def from_named_params(cls, named_params):
        offset = 0
        mappings = []
        for name, param in unique_lora_params(named_params):
            mappings.append(_PagedParamMapping(name, param, offset, param.numel(), tuple(param.shape), param.dtype))
            offset += param.numel()
        return cls(mappings, total_numel=offset)
```

3. State store:

```python
class PagedAdamStateStore:
    def __init__(self, backend, flat):
        self.master = backend.register_tensor("optimizer/master", "optimizer_state", flat_shape_float32)
        self.exp_avg = backend.register_tensor("optimizer/exp_avg", "optimizer_state", flat_shape_float32)
        self.exp_avg_sq = backend.register_tensor("optimizer/exp_avg_sq", "optimizer_state", flat_shape_float32)
        self.initialized = False

    def initialize_from_cuda_params(self, mappings):
        # Fill large CPU transfer buffer by slices, then write full flat files.
        master = allocate_pinned_cpu_float32(total_numel)
        for m in mappings:
            master[m.offset:m.offset+m.numel].copy_(m.cuda_param.detach().reshape(-1).float())
        backend.write_cpu(self.master, master, async_op=False)
        backend.write_cpu(self.exp_avg, zeros_like(master), async_op=False)
        backend.write_cpu(self.exp_avg_sq, zeros_like(master), async_op=False)
        self.initialized = True

    def read_tile(self, handle, start, length, *, async_op=False) -> TensorOrFuture:
        return backend.read_slice_cpu(handle, start, length, async_op=async_op)

    def write_tile(self, handle, start, tensor, *, async_op=False):
        return backend.write_slice_cpu(handle, start, tensor, async_op=async_op)
```

Stage 1 `DiskTensorStore` may need slice reads/writes. Implement them with offset-aware AIO if available; otherwise read aligned tile into a transfer buffer and write aligned tile back. Do not read/write the entire optimizer state for each tile.

4. Grad offload hooks write into the flat grad buffer:

```python
def _offload_grad_from_hook(self, mapping, param):
    grad = mapping.cuda_param.grad
    dst = self._grad_flat_buffer.narrow(0, mapping.offset, mapping.numel).view(mapping.shape)
    dst.copy_(grad.detach(), non_blocking=False)
    mapping.last_had_grad = True
    mapping.cuda_param.grad = None
    if self._coordinator is not None:
        self._coordinator.release(mapping.cuda_param)
```

5. Vectorized tiled AdamW update:

```python
def step(self, closure=None):
    self._step += 1
    beta1, beta2 = group["betas"]
    lr = group["lr"]
    wd = group["weight_decay"]
    eps = group["eps"]
    bias_correction1 = 1 - beta1 ** self._step
    bias_correction2 = 1 - beta2 ** self._step
    step_size = lr * math.sqrt(bias_correction2) / bias_correction1

    previous_write_futures = []
    next_prefetch = self._prefetch_tile(self._tiles[0]) if self.pipeline_read else None

    for tile_index, tile in enumerate(self._tiles):
        if next_prefetch is not None:
            master, exp_avg, exp_avg_sq = self._wait_prefetch(next_prefetch)
        else:
            master = self._store.read_tile(self._store.master, tile.start, tile.length)
            exp_avg = self._store.read_tile(self._store.exp_avg, tile.start, tile.length)
            exp_avg_sq = self._store.read_tile(self._store.exp_avg_sq, tile.start, tile.length)

        next_prefetch = self._prefetch_tile(self._tiles[tile_index + 1]) if has_next and self.pipeline_read else None

        grad_tile = self._grad_flat_buffer.narrow(0, tile.start, tile.length)
        active_ranges = self._flat.active_ranges_in_tile(tile)
        for start, length in active_ranges:
            local = start - tile.start
            master_view = master.narrow(0, local, length)
            grad_view = grad_tile.narrow(0, local, length)
            exp_avg_view = exp_avg.narrow(0, local, length)
            exp_avg_sq_view = exp_avg_sq.narrow(0, local, length)

            if wd:
                master_view.add_(master_view, alpha=-lr * wd)
            exp_avg_view.mul_(beta1).add_(grad_view, alpha=1 - beta1)
            exp_avg_sq_view.mul_(beta2).addcmul_(grad_view, grad_view, value=1 - beta2)
            denom = exp_avg_sq_view.sqrt().add_(eps)
            master_view.addcdiv_(exp_avg_view, denom, value=-step_size)

        self._refresh_compute_or_lora_home_from_tile(tile, master)

        if self.pipeline_write:
            previous_write_futures += [
                self._store.write_tile(self._store.master, tile.start, master, async_op=True),
                self._store.write_tile(self._store.exp_avg, tile.start, exp_avg, async_op=True),
                self._store.write_tile(self._store.exp_avg_sq, tile.start, exp_avg_sq, async_op=True),
            ]
            self._store.wait_if_too_many(previous_write_futures)
        else:
            self._store.write_tile(..., async_op=False)

    self._store.wait_all(previous_write_futures)
    if self._coordinator is not None:
        self._coordinator.flush_dirty_groups()
```

6. Refresh LoRA homes from a tile:

```python
def _refresh_compute_or_lora_home_from_tile(self, tile, master_tile):
    for mapping in self._flat.mappings_overlapping(tile):
        local_start = mapping.offset - tile.start
        src = master_tile.narrow(0, local_start, mapping.numel)
        if self._coordinator and self._coordinator.is_registered(mapping.cuda_param):
            self._coordinator.refresh_home_from_master(mapping.cuda_param, src.view(mapping.shape))
        else:
            mapping.cuda_param.data.copy_(src.view(mapping.shape), non_blocking=False)
```

The active-range loop is over coalesced LoRA ranges during optimizer step only. It must not touch expert GEMM scheduling or forward/backward grouped kernels. Do not use a dense per-element mask for inactive parameters; update contiguous active ranges so no-grad parameters are not decayed or have moments changed.

7. Summary fields:

```python
def asym_cpu_adamw_summary(self):
    return {
        "enabled": True,
        "optimizer_paged": True,
        "optimizer_state_logical_bytes": 3 * total_numel * 4,
        "optimizer_state_resident_bytes": active_tile_buffers_bytes + grad_flat_bytes,
        "optimizer_tile_bytes": tile_bytes,
        "optimizer_tile_count": len(self._tiles),
        "optimizer_nvme_stats": backend.summary(role="optimizer_state"),
        "last_optimizer_tile_read_ms": ...,
        "last_optimizer_tile_write_ms": ...,
        "last_cpu_adam_step_ms": ...,
    }
```

Validation before Stage 4:

```bash
.venv/bin/python -m pytest \
  tests/training/test_paged_cpu_adam.py \
  tests/training/test_paged_cpu_adam_state_dict.py \
  tests/training/test_lora_weight_nvme_offload.py \
  -q
```

Numerical parity test requirements:

- Same toy LoRA model, same gradients, same hyperparameters.
- Compare `PagedAsymCPUAdamW` against existing `AsymCPUAdamW(backend="torch")` for at least 3 steps.
- Tolerances: exact-ish for FP32 master (`rtol=1e-5`, `atol=1e-6`) before BF16 copy/home cast.
- Verify state dict save/load materializes CPU state correctly or explicitly mark paged checkpoint unsupported with a parser guard.

E2E candidate:

```bash
rm -rf /local_nvme/asymgemm_stage3
OUTPUT_ROOT=profiling_nvme/stage3_optimizer_nvme \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
WORKLOADS='4096|4|1' \
WARMUP_STEPS=5 MAX_STEPS=10 \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_EXTERNAL_MEMORY=true \
PROFILE_MEMORY_SNAPSHOT=false \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
ASYM_NVME_ENABLE=true \
ASYM_NVME_PATH=/local_nvme/asymgemm_stage3 \
ASYM_NVME_ROLES=lora_weights,optimizer_state \
ASYM_NVME_CPU_CACHE_BYTES=$((512*1024*1024)) \
ASYM_NVME_OPTIMIZER_TILE_BYTES=$((256*1024*1024)) \
ASYM_NVME_OPTIMIZER_PIPELINE_READ=true \
ASYM_NVME_OPTIMIZER_PIPELINE_WRITE=true \
ASYM_NVME_PREFETCH_DEPTH=1 \
ASYM_NVME_REQUIRE_AIO=true \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
scripts/lf/profile_lora_lf.sh --gpus 0 --overwrite true
```

Compare against the accepted Stage 2 candidate:

```bash
BASE=$(find profiling_nvme/stage2_lora_nvme -path '*/source_profile.json' -print -quit)
CAND=$(find profiling_nvme/stage3_optimizer_nvme -path '*/source_profile.json' -print -quit)
.venv/bin/python scripts/lf/compare_nvme_profiles.py \
  --baseline "$BASE" \
  --candidate "$CAND" \
  --target optimizer \
  --memory-metric asym_cpu_adamw.optimizer_state_resident_bytes \
  --min-memory-drop-bytes $((4*1024*1024*1024)) \
  --min-memory-drop-pct 20 \
  --timing-metric trainer.timing.measured_e2e_step_milliseconds \
  --extra-timing-metric step_samples.forward_milliseconds \
  --extra-timing-metric step_samples.backward_milliseconds \
  --extra-timing-metric step_samples.optimizer_update_side_milliseconds \
  --max-latency-regression-pct 5 \
  --require-nvme-role optimizer_state
```

Acceptance:

- Required artifacts exist: `source_profile.json`, `lat.md`, `memory.md`, `step_samples.csv`, `step_samples.json`, `asym_cpu_adamw.csv`, `asym_nvme.csv`, and memory-breakdown artifacts.
- `source_profile.json.config.asym_nvme_enable=true`, `asym_nvme.roles` contains `optimizer_state`, and `asym_nvme.stats_by_role.optimizer_state` exists.
- `asym_cpu_adamw.optimizer_paged=true`, `optimizer_tile_count > 0`, and `optimizer_tile_bytes >= effective_min_swappable_bytes`.
- Host optimizer/master/state residency drops meaningfully.
- No HBM regression.
- E2E timing, `step_samples` forward/backward timing, and `step_samples.optimizer_update_side_milliseconds` regression are each <= 5% for default acceptance.
- If timing regression is 5-10% but memory drop is essential for a capacity-only profile, keep it behind an explicit capacity-mode flag and do not use it as the default fair-comparison path.
- No per-parameter/tiny-tile disk I/O. Tile size must be large enough to amortize AIO overhead.
- `optimizer_nvme_stats.bytes_read`, `bytes_written`, `read_ops`, and `write_ops` are nonzero; tile read/write wait time is reported separately from Adam compute time.
- Measured losses remain finite and numerical parity tests pass versus `AsymCPUAdamW(backend="torch")`.

Risks to watch:

- A pure PyTorch vectorized CPU Adam tile may be slower than `DeepSpeedCPUAdam`. If it fails timing, reject default use and investigate a DeepSpeedCPUAdam tile backend separately.
- Offset AIO support must be verified. If the local DeepSpeed AIO build lacks efficient offset reads/writes, whole-file state rewrites are unacceptable.
- Gradient paging during backward is intentionally not in this stage; writing tiny grad chunks from hooks would be too latency sensitive.

## Stage 4: Optional Frozen/Base Parameter NVMe

Purpose: reduce host memory from frozen/base parameters after LoRA adapter homes and optimizer state have already been handled. This is optional because base weights are hot. A blocked NVMe read per layer can erase the memory benefit, so this stage is accepted only if trace prefetch and CPU cache keep wait time low.

Files for optional local base-weight NVMe:

- `asym_gemm/training/frozen_linear.py`
- `asym_gemm/training/offload.py`
- `asym_gemm/training/host_weight.py` if present or added
- model wrappers that request frozen/base weights
- `asym_gemm/training/placement.py` for the existing `TensorRef`/`PlacementBackend` API
- `asym_gemm/training/disk_offload.py` for the local backend implementation
- add tests:
  - `tests/training/test_base_weight_nvme_offload.py`

No Stage 0B profile/postprocess refactor is expected here; add base-weight role metrics through `asym_nvme_summary()`.

Code changes for optional local base weights:

1. Register frozen/base tensors as large refs, preferably per layer or packed module block. Do not create per-row or per-expert disk files.

```python
@dataclass
class FrozenWeightRef:
    name: str
    ref: TensorRef
    shape: tuple[int, ...]
    dtype: torch.dtype
    view_offset: int = 0
    view_numel: int | None = None

def register_frozen_weight(name, tensor):
    ref = backend.register_tensor_ref(
        stable_id=f"base_weight/{name}",
        role="base_weight",
        tensor=tensor.detach().contiguous(),
        mutable=False,
    )
    replace_param_with_lightweight_placeholder(name)
    return FrozenWeightRef(name=name, ref=ref, shape=tuple(tensor.shape), dtype=tensor.dtype)
```

2. Add a base-weight coordinator with the same execution-order idea used for LoRA weights:

```python
class BaseWeightOffloadCoordinator:
    def __init__(self, backend, cpu_cache_bytes, prefetch_depth):
        self.backend = backend
        self._trace_build = []
        self._trace = []
        self._trace_frozen = False
        self._trace_cursor = 0
        self._prefetch_depth = prefetch_depth

    def materialize_for_layer(self, weight_ref, *, target_device):
        self._record_access(weight_ref.ref.stable_id)
        self._prefetch_next()
        materialized = self.backend.materialize(
            weight_ref.ref,
            target_device=target_device,
            target_layout="contiguous",
        )
        return materialized.tensor.view(weight_ref.shape)

    def finish_layer(self, weight_ref):
        self.backend.release_cached_if_over_budget(weight_ref.ref, role="base_weight")
```

3. Keep compute layout explicit at the call site:

```python
def materialize_base_weight_for_gemm(weight_ref, *, compute_device):
    target = "cuda" if compute_device.type == "cuda" else "cpu_pinned"
    # Kernels still receive a real tensor in the required layout.
    return base_weight_coordinator.materialize_for_layer(weight_ref, target_device=target)
```

Do not use base-weight NVMe for hot weights unless the CPU cache hit rate is high enough. The accepted path must be large-block reads, bounded resident CPU cache, trace prefetch, and explicit release after the layer/group no longer needs the weight.

Validation before Stage 5:

```bash
.venv/bin/python -m pytest \
  tests/training/test_base_weight_nvme_offload.py \
  tests/test_lf_memory_breakdown.py \
  -q
```

E2E candidate:

```bash
rm -rf /local_nvme/asymgemm_stage4
OUTPUT_ROOT=profiling_nvme/stage4_base_weight_nvme \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
WORKLOADS='4096|4|1' \
WARMUP_STEPS=5 MAX_STEPS=10 \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_EXTERNAL_MEMORY=true \
PROFILE_MEMORY_SNAPSHOT=false \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
ASYM_NVME_ENABLE=true \
ASYM_NVME_PATH=/local_nvme/asymgemm_stage4 \
ASYM_NVME_ROLES=lora_weights,optimizer_state,base_weight \
ASYM_NVME_CPU_CACHE_BYTES=$((8*1024*1024*1024)) \
ASYM_NVME_REQUIRE_AIO=true \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
scripts/lf/profile_lora_lf.sh --gpus 0 --overwrite true
```

Compare against the accepted Stage 3 candidate:

```bash
BASE=$(find profiling_nvme/stage3_optimizer_nvme -path '*/source_profile.json' -print -quit)
CAND=$(find profiling_nvme/stage4_base_weight_nvme -path '*/source_profile.json' -print -quit)
.venv/bin/python scripts/lf/compare_nvme_profiles.py \
  --baseline "$BASE" \
  --candidate "$CAND" \
  --target host \
  --memory-metric memory.process.rss_peak_bytes \
  --min-memory-drop-bytes $((2*1024*1024*1024)) \
  --min-memory-drop-pct 10 \
  --timing-metric trainer.timing.measured_e2e_step_milliseconds \
  --extra-timing-metric step_samples.forward_milliseconds \
  --extra-timing-metric step_samples.backward_milliseconds \
  --max-latency-regression-pct 5 \
  --require-nvme-role base_weight
```

Acceptance:

- Required artifacts exist: `source_profile.json`, `lat.md`, `memory.md`, `step_samples.csv`, `step_samples.json`, `asym_nvme.csv`, and memory-breakdown artifacts.
- `source_profile.json.config.asym_nvme_enable=true`, `asym_nvme.roles` contains `base_weight`, and `asym_nvme.stats_by_role.base_weight` exists.
- Host memory decreases meaningfully.
- HBM does not regress.
- E2E timing and `step_samples` forward/backward timing regression are each <= 5% for default use, <= 10% only for explicit capacity mode.
- Cache hit/miss and NVMe wait stats prove this is not a blocking read on every hot layer; blocking read wait time must not dominate forward time.
- Measured losses remain finite.

Risks to watch:

- Base weights are usually hot. Disk-staging them may be worse than CPU residency unless the cache policy is strong.
- If the model path needs base weights in HBM every layer and the cache cannot hold the working set, reject this stage for the default fair-comparison path.
- Checkpoint semantics must follow the active owner. Local backend writes local checkpoints; future DeepSpeed backend must use DeepSpeed checkpoint APIs.

## Stage 5: Future DeepSpeed Adapter and Deferred Roles

Purpose: keep later ownership paths possible without delaying the useful local single-GPU NVMe work. Do not implement this stage until the local results show which roles are worth preserving.

Future DeepSpeed adapter files:

- add `asym_gemm/training/deepspeed_placement.py`
  - `DeepSpeedZeROBackend`
- `asym_gemm/training/placement.py`
  - keep the protocol stable
- model/optimizer construction glue only after deciding to run under DeepSpeed ownership

Future DeepSpeed adapter design:

```python
class DeepSpeedZeROBackend:
    def materialize(self, ref, *, target_device, target_layout="contiguous", stream=None):
        ds_param = self.lookup[ref.stable_id]
        # DeepSpeed owns partition/NVMe/all-gather state.
        # Adapter only asks ZeRO to gather/swap and then returns a view/copy matching AsymGEMM layout.
        with deepspeed.zero.GatheredParameters([ds_param], modifier_rank=None):
            tensor = ds_param.data
            if target_device == "cuda" and tensor.device.type != "cuda":
                tensor = tensor.to(device="cuda", non_blocking=True)
            if target_device in {"cpu", "cpu_pinned"} and tensor.device.type != "cpu":
                tensor = tensor.to(device="cpu", non_blocking=False)
            if target_device == "cpu_pinned" and torch.cuda.is_available() and not tensor.is_pinned():
                tensor = tensor.pin_memory()
            if not layout_ok(tensor, target_layout):
                tensor = tensor.contiguous()
            return MaterializedTensor(ref=ref, tensor=tensor, device=tensor.device, layout=target_layout, owner_token=...)
```

DeepSpeed integration means "DeepSpeed owns the model-state lifecycle, AsymGEMM supplies compute kernels/policies." It does not mean running current local `AsymCPUAdamW` and LoRA `.data` placeholder swapping on top of ZeRO-owned params.

Deferred activation NVMe:

- Do not add `activation_spill` to the accepted role set.
- Prefer activation checkpointing/recompute and existing CPU activation offload.
- Revisit only with a separate profile showing `activation_offload.cpu_peak_bytes_live` or process RSS is the dominant remaining bottleneck and parameter/optimizer NVMe is already accepted.

Deferred gradient NVMe:

- Gradients are not Adam state; they are transient optimizer-step inputs.
- Default Stage 3 keeps gradients in a CPU flat buffer because writing each autograd hook result to NVMe would add backward-critical tiny writes.
- If host RAM is still too high, add an explicit `gradient` capacity mode inside `PagedAsymCPUAdamW` only after Stage 3 is accepted. It must use large coalesced tiles and optimizer-step reads, not per-parameter disk I/O.

Sketch for a later gradient capacity mode:

```python
class PagedGradientStore:
    def __init__(self, backend, flat, tile_bytes):
        self.grad = backend.register_tensor("optimizer/grad", "gradient", flat_shape_float32)
        self._tile_buffers = allocate_small_number_of_pinned_tiles(tile_bytes)

    def record_grad(self, mapping, grad):
        # First acceptable implementation may still keep CPU grad flat resident.
        # A true capacity mode must aggregate into large tile buffers and flush
        # full contiguous tiles. Do not issue one NVMe write per parameter hook.
        self._accumulate_or_mark_range(mapping.offset, grad.detach().float())

    def read_grad_tile_for_step(self, tile):
        return backend.read_slice_cpu(self.grad, tile.start, tile.length, async_op=True)
```

Future validation before accepting any Stage 5 implementation:

```bash
.venv/bin/python -m pytest \
  tests/training/test_deepspeed_placement_contract.py \
  tests/training/test_deepspeed_placement_no_local_owner_mix.py \
  -q
```

DeepSpeed-backed e2e profile must use a separate output root and a DeepSpeed config with ZeRO offload enabled. Do not compare it against local `AsymCPUAdamW` as if ownership were identical; compare against the corresponding DeepSpeed baseline plus AsymGEMM kernels enabled:

```bash
OUTPUT_ROOT=profiling_nvme/stage5_deepspeed_adapter \
BACKEND_SPECS='zero3_offload|norecomp' \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
WORKLOADS='4096|4|1' \
WARMUP_STEPS=5 MAX_STEPS=10 \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_EXTERNAL_MEMORY=true \
PROFILE_MEMORY_SNAPSHOT=false \
ASYM_NVME_ENABLE=false \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
scripts/lf/profile_lora_lf.sh --gpus 0 --overwrite true
```

Stage 5 acceptance, if implemented later:

- No local `AsymCPUAdamW`, local LoRA `.data` placeholder owner, or local NVMe owner is active for ZeRO-owned trainable params.
- DeepSpeed/ZeRO owns checkpointing, optimizer state, partition state, and parameter residency.
- AsymGEMM receives real materialized tensors/views and keeps the same grouped compute path.
- E2E, forward/backward, optimizer-step, HBM, RSS, and NVMe stats are compared against a DeepSpeed baseline with the same ZeRO offload config.
- If the adapter adds latency without a clear memory/capacity win, reject it for the single-GPU local target.

Risks to watch:

- The DeepSpeed adapter is feasible because both systems can expose flat buffers plus shaped views, but ownership must be exclusive.
- Gradient NVMe may not reduce peak RSS unless the CPU flat grad buffer is genuinely removed or tiled. A design that writes a duplicate grad copy to disk while keeping the full CPU buffer is rejected.
- Activation NVMe is intentionally not part of this plan. Adding it later should require a new design and e2e acceptance gate, not a quiet extension of the current roles.

## Implementation Order

Use this order:

```text
Stage 0A: additive contracts/no-op refactor
Stage 0B: config/profile/compare plumbing
Stage 1: disk substrate
Stage 2: LoRA NVMe homes
Stage 3: optimizer-state paging
Stage 4: optional frozen/base parameter NVMe
Stage 5: future DeepSpeed adapter / deferred roles
```

Do not start Stage 1 until Stage 0A and Stage 0B both have clean no-change profiles. Do not start Stage 3 until Stage 2 has a clean accepted profile. Do not start Stage 4 until Stage 3 proves whether host memory is still the limiter. Do not implement gradient NVMe, activation NVMe, or the DeepSpeed adapter until local single-GPU NVMe results prove they are necessary.
