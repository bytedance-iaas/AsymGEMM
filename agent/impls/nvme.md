# AsymGEMM NVMe Offload Implementation Plan

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

DeepSpeed pieces to reuse directly where practical:

```text
AsyncIOBuilder            required local AIO performance path
GDSBuilder                later optional GPU-direct experiment
DeepSpeed AIO knobs       same names/defaults where possible
DeepSpeedCPUAdam          only where it fits resident/tiled CPU tensors
SwapBuffer patterns       adapt buffer-pool shape, do not import ZeRO owner classes
```

DeepSpeed pieces not to reuse directly in local mode:

```text
ZeRO parameter coordinator
AsyncPartitionedParameterSwapper as-is
PartitionedOptimizerSwapper as-is
ZeRO optimizer/checkpoint ownership
```

Those high-level classes assume `ds_tensor`, `ds_id`, distributed partition state, all-gather, ZeRO release state, and ZeRO checkpoint ownership. Local AsymGEMM must not mix those owners with its current LoRA `.data` placeholder swapping and `AsymCPUAdamW` CPU-master ownership.

## Acceptance Gate

Each stage is accepted only with full LF LoRA profiling, except Stage 1 disk-substrate unit/micro tests because it is isolated infrastructure.

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

## Stage 0: Config, Profile Schema, Acceptance Tooling

Purpose: add the feature surface and reporting before changing runtime behavior. This prevents untracked NVMe behavior and gives every later stage a clear pass/fail gate.

Files to modify:

- add `asym_gemm/training/placement.py`
  - `TensorRole`
  - `TensorRef`
  - `MaterializedTensor`
  - `PrefetchHandle`
  - `PlacementBackend`
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
  - new `tests/training/test_placement_backend_contract.py`
  - `tests/lf/test_asym_cpu_adamw_args.py`
  - `tests/test_lf_memory_breakdown.py`
  - new `tests/lf/test_nvme_profile_compare.py`

Code changes:

1. Add the replaceable placement contract now, before wiring any runtime role to NVMe:

```python
# asym_gemm/training/placement.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch

TensorRole = Literal[
    "lora_weights",
    "optimizer_state",
    "activation_spill",
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

2. Add LF dataclass args:

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
    asym_nvme_activation_cpu_budget_bytes: int = field(default=0)
    asym_nvme_activation_spill_tags: str = field(default="")
```

Allowed initial roles:

```text
lora_weights
optimizer_state
activation_spill
base_weight
```

2. Add parser validation:

```python
def _parse_nvme_roles(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}

def _verify_asym_nvme_args(model_args, training_args, finetuning_args):
    if not finetuning_args.asym_nvme_enable:
        return
    roles = _parse_nvme_roles(finetuning_args.asym_nvme_roles)
    allowed = {"lora_weights", "optimizer_state", "activation_spill", "base_weight"}
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

3. Add script plumbing:

```bash
ASYM_NVME_ENABLE=${ASYM_NVME_ENABLE:-false}
ASYM_NVME_PATH=${ASYM_NVME_PATH:-}
ASYM_NVME_ROLES=${ASYM_NVME_ROLES:-}
...
ASYM_NVME_ENABLE="$(bool_string ASYM_NVME_ENABLE "${ASYM_NVME_ENABLE}")"

CMD_ARGS+=(--asym_nvme_enable "${ASYM_NVME_ENABLE}")
CMD_ARGS+=(--asym_nvme_path "${ASYM_NVME_PATH}")
CMD_ARGS+=(--asym_nvme_roles "${ASYM_NVME_ROLES}")
...

ASYM_GEMM_LF_CONFIG_ASYM_NVME_ENABLE="${ASYM_NVME_ENABLE}"
ASYM_GEMM_LF_CONFIG_ASYM_NVME_PATH="${ASYM_NVME_PATH}"
ASYM_GEMM_LF_CONFIG_ASYM_NVME_ROLES="${ASYM_NVME_ROLES}"
```

In `profile_lora_lf.sh`, add NVMe fields to run labels and completion checks so baseline and candidate profiles cannot be confused:

```bash
nvme_tag="nvmeoff"
if [[ "$(bool_value "${ASYM_NVME_ENABLE}")" == "true" ]]; then
  nvme_tag="nvme_${ASYM_NVME_ROLES//,/+}"
fi
run_dir_name="${run_dir_name}_${nvme_tag}"
```

4. Add source-profile summary hook:

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

5. Add comparison tool:

```python
def main():
    baseline = load_profile(args.baseline)
    candidate = load_profile(args.candidate)
    target = args.target  # hbm, host, optimizer, lora_host, activation_host

    base_mem = extract_metric(baseline, args.memory_metric)
    cand_mem = extract_metric(candidate, args.memory_metric)
    base_ms = extract_timing(baseline, args.timing_metric)
    cand_ms = extract_timing(candidate, args.timing_metric)

    mem_drop = base_mem - cand_mem
    mem_drop_pct = mem_drop / max(base_mem, 1) * 100.0
    latency_regression_pct = (cand_ms - base_ms) / max(base_ms, 1e-9) * 100.0

    assert candidate_is_complete(candidate)
    assert mem_drop >= args.min_memory_drop_bytes
    assert mem_drop_pct >= args.min_memory_drop_pct
    assert latency_regression_pct <= args.max_latency_regression_pct
    if args.require_nvme_role:
        assert args.require_nvme_role in candidate["asym_nvme"]["roles"]
```

Validation before Stage 1:

```bash
bash -n scripts/lf/run_lf_lora_sft.sh scripts/lf/profile_lora_lf.sh

.venv/bin/python -m pytest \
  tests/training/test_placement_backend_contract.py \
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
scripts/lf/profile_lora_lf.sh --dry-run true --gpus 0
```

Acceptance:

- All config values appear in `source_profile.json.config`.
- Invalid role combinations fail during argument validation.
- Baseline profile still completes with `ASYM_NVME_ENABLE=false`.
- `compare_nvme_profiles.py` can reject synthetic profiles with no memory drop or excessive latency.

Risks to watch:

- LlamaFactory argument names must stay snake_case CLI names matching dataclass fields.
- Run labels must include NVMe role/cache knobs or stale source profiles can be reused incorrectly.
- Do not make `ASYM_NVME_ENABLE=true` imply runtime behavior until later stages implement the role.

## Stage 1: Local Disk Substrate

Purpose: implement reusable NVMe tensor storage using DeepSpeed's low-level AIO path, with pinned/aligned transfer buffers and stats. This stage is accepted by unit and I/O sanity tests only because no training behavior changes yet.

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
PinnedTransferPool
DiskTensorStore
DiskOffloadStats
LocalAsymNVMeBackend
get_local_nvme_backend()
```

Code changes:

1. Configuration and alignment:

```python
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
        return max(1048576, self.min_swappable_bytes, self.aio_block_size)

    @property
    def aligned_bytes(self) -> int:
        return 1024 * max(1, self.aio_intra_op_parallelism)

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

3. AIO handle creation:

```python
def _build_aio_handles(cfg: DiskOffloadConfig):
    try:
        from deepspeed.ops.op_builder import AsyncIOBuilder
        factory = AsyncIOBuilder().load(verbose=False).aio_handle
        read = factory(
            block_size=cfg.aio_block_size,
            queue_depth=cfg.aio_queue_depth,
            single_submit=cfg.aio_single_submit,
            overlap_events=cfg.aio_overlap_events,
            intra_op_parallelism=cfg.aio_intra_op_parallelism,
        )
        write = factory(...)
        return read, write
    except Exception:
        if cfg.require_aio:
            raise
        return None, None  # debug-only synchronous fallback
```

4. Transfer pool:

```python
class PinnedTransferPool:
    def __init__(self, cfg, dtype):
        self.buffers = [
            allocate_pinned_aligned(cfg.transfer_buffer_bytes, dtype)
            for _ in range(cfg.transfer_buffer_count)
        ]
        self.free = list(range(len(self.buffers)))

    def acquire(self, aligned_numel, dtype) -> torch.Tensor:
        if aligned_numel > max_buffer_numel:
            return allocate_pinned_aligned(aligned_numel * element_size(dtype), dtype)
        idx = self.free.pop()
        return self.buffers[idx].narrow(0, 0, aligned_numel)

    def release(self, tensor):
        return_buffer_to_free_list_if_pool_owned(tensor)
```

5. Tensor store:

```python
class DiskTensorStore:
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
        src = tensor.detach().reshape(-1)
        buf = self.pool.acquire(handle.aligned_numel, handle.dtype)
        buf.narrow(0, 0, handle.numel).copy_(src, non_blocking=False)
        if handle.aligned_numel > handle.numel:
            buf.narrow(0, handle.numel, handle.aligned_numel - handle.numel).zero_()
        status = self.write_handle.async_pwrite(buf, str(handle.path), 0)
        if status != 0:
            raise IOError(...)
        future = DiskIOFuture("write", handle, buf)
        if not async_op:
            self.wait(future)
        return future

    def read(self, handle, *, async_op=False) -> tuple[torch.Tensor, DiskIOFuture | None]:
        dst = self.pool.acquire(handle.aligned_numel, handle.dtype)
        status = self.read_handle.async_pread(dst, str(handle.path), 0)
        if status != 0:
            raise IOError(...)
        future = DiskIOFuture("read", handle, dst)
        if async_op:
            return dst.narrow(0, 0, handle.numel).view(handle.shape), future
        self.wait(future)
        return dst.narrow(0, 0, handle.numel).view(handle.shape), None

    def wait(self, future_or_kind):
        started = time.perf_counter()
        completed = handle.wait()
        assert completed >= expected_count
        update_stats(...)
```

6. Local backend and CPU cache. `DiskTensorHandle` stays private to the local backend. Public users hold `TensorRef`.

```python
class LocalAsymNVMeBackend:
    def __init__(self, cfg):
        self.store = DiskTensorStore(cfg)
        self._handles_by_ref: dict[str, DiskTensorHandle] = {}
        # Policy-neutral resident CPU buffers. Role owners decide what to admit/evict.
        self.cache: dict[str, torch.Tensor] = {}
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
        return MaterializedTensor(ref=ref, tensor=tensor.view(ref.shape), device=tensor.device, layout=target_layout)

    def materialize_cpu_handle(self, handle, *, cache=True) -> torch.Tensor:
        cached = self.cache.get(handle.id)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached.narrow(0, 0, handle.numel).view(handle.shape)
        self.stats.cache_misses += 1
        tensor, _ = self.store.read(handle, async_op=False)
        if cache and self.cfg.cpu_cache_bytes > 0:
            self._admit(handle, tensor.reshape(-1))
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
        self._admit(handle, tensor.reshape(-1))
        return self.cache[handle.id].view(handle.shape)
```

Validation before Stage 2:

```bash
.venv/bin/python -m pytest tests/training/test_disk_offload.py -q

.venv/bin/python - <<'PY'
from deepspeed.ops.op_builder import AsyncIOBuilder
h = AsyncIOBuilder().load(verbose=False).aio_handle(
    block_size=1048576,
    queue_depth=8,
    single_submit=False,
    overlap_events=True,
    intra_op_parallelism=1,
)
print("async_io ok", h.get_block_size(), h.get_queue_depth())
PY
```

Required tests:

- `LocalAsymNVMeBackend` satisfies `PlacementBackend` contract for register/materialize/prefetch/wait/flush/stats.
- BF16/FP32 roundtrip for tensors below, equal to, and above 1 MiB.
- Misaligned numel pads for I/O but returns the exact logical shape.
- CPU cache byte accounting is correct; role-specific code, not the substrate, decides which tensors to evict.
- Async prefetch followed by wait returns correct data.
- `require_aio=true` fails clearly when DeepSpeed async I/O is unavailable.

Acceptance:

- Unit tests pass.
- AIO smoke test succeeds on the target machine.
- No training profile acceptance is required yet because no training path uses the substrate.

Risks to watch:

- BF16 raw writes must use tensor AIO, not NumPy conversion.
- Pinned allocation may fail under host pressure; record and report fallback.
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
- Stage 0 script/profile files if not already completed
- add tests:
  - `tests/training/test_lora_weight_nvme_offload.py`
  - extend `tests/test_lf_memory_breakdown.py`

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
  --max-latency-regression-pct 5 \
  --require-nvme-role lora_weights
```

Acceptance:

- `weight_offload_nvme_group_count > 0`.
- `weight_offload_nvme_trace_frozen=true` after warmup and `weight_offload_nvme_trace_length > 0`.
- `weight_offload_nvme_trace_disabled=false`; if it is true, this run did not exercise the optimized prefetch policy.
- `bytes_read` and `bytes_written` are nonzero.
- Prefetch reads appear after the first full step; first-step synchronous reads are allowed while the trace is being built.
- Resident pinned/CPU home bytes obey the cache budget plus active/prefetch buffers.
- Host memory decreases meaningfully.
- HBM peak does not increase materially.
- E2E and forward/backward timing regression is <= 5%.
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
- Stage 0 profile/postprocess files
- add tests:
  - `tests/training/test_paged_cpu_adam.py`
  - `tests/training/test_paged_cpu_adam_state_dict.py`

Code changes:

1. Use a separate optimizer class. Do not mutate `DeepSpeedCPUAdam` state tensors behind its back in the first implementation; it expects resident CPU tensors and owns its state dict.

```python
class PagedAsymCPUAdamW(torch.optim.Optimizer):
    def __init__(..., nvme_backend, optimizer_tile_bytes=268435456):
        validate_lora_cuda_params()
        super().__init__([{"params": cuda_params, ...}], defaults)
        self._flat = FlatLoRAIndex.from_named_params(trainable_lora)
        self._store = PagedAdamStateStore(nvme_backend, self._flat)
        self._grad_flat_buffer = allocate_pinned_cpu_float32(self._flat.total_numel)
        self._tiles = plan_tiles(self._flat.total_numel, optimizer_tile_bytes, min_tile_bytes=nvme_min)
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
  --max-latency-regression-pct 5 \
  --require-nvme-role optimizer_state
```

Acceptance:

- Host optimizer/master/state residency drops meaningfully.
- No HBM regression.
- E2E timing regression <= 5% for default acceptance.
- If timing regression is 5-10% but memory drop is essential for a capacity-only profile, keep it behind an explicit capacity-mode flag and do not use it as the default fair-comparison path.
- No per-parameter/tiny-tile disk I/O. Tile size must be large enough to amortize AIO overhead.

Risks to watch:

- A pure PyTorch vectorized CPU Adam tile may be slower than `DeepSpeedCPUAdam`. If it fails timing, reject default use and investigate a DeepSpeedCPUAdam tile backend separately.
- Offset AIO support must be verified. If the local DeepSpeed AIO build lacks efficient offset reads/writes, whole-file state rewrites are unacceptable.
- Gradient paging during backward is intentionally not in this stage; writing tiny grad chunks from hooks would be too latency sensitive.

## Stage 4: Activation NVMe Spill

Purpose: spill selected large CPU activation handles to NVMe when CPU activation memory is the bottleneck. This is high risk because many backward paths need these tensors on the critical path.

Files to modify:

- `asym_gemm/training/activation_offload.py`
  - `CPUActivationHandle`
  - `ActivationOffloadManager.__init__`
  - `offload`
  - `adopt_cpu`
  - `wait_cpu_ready`
  - `stage`
  - `stage_concat_columns`
  - `release_cpu`
  - `snapshot`
- direct `handle.tensor` users must be audited and adjusted:
  - `asym_gemm/training/qwen3_moe.py`
  - `asym_gemm/training/llama4_experts.py`
  - `asym_gemm/training/llama4_shared_mlp.py`
  - `asym_gemm/training/attention_activation_offload.py`
  - `asym_gemm/training/exp_act_offload_lora.py`
- add tests:
  - `tests/training/test_activation_nvme_spill.py`
  - extend activation-offload e2e/profile schema tests

Code changes:

1. Replace the assumption that every handle always has a resident CPU tensor:

```python
@dataclass(frozen=True)
class CPUActivationHandle:
    tag: str
    tensor: torch.Tensor | None
    original_device: torch.device
    original_dtype: torch.dtype
    original_shape: tuple[int, ...]
    spill_ref: TensorRef | None = None
    resident: Literal["cpu", "nvme"] = "cpu"

    @property
    def nbytes(self) -> int:
        if self.tensor is not None:
            return _tensor_nbytes(self.tensor)
        return math.prod(self.original_shape) * torch.empty((), dtype=self.original_dtype).element_size()
```

2. Centralize tensor access:

```python
class ActivationOffloadManager:
    def cpu_tensor(self, handle: CPUActivationHandle) -> torch.Tensor:
        if handle.tensor is not None:
            self.wait_cpu_ready(handle)
            return handle.tensor
        if handle.spill_ref is None:
            raise RuntimeError("activation handle has no CPU tensor or spill ref")
        cached = self._materialized_spills.get(handle.spill_ref.stable_id)
        if cached is not None:
            return cached
        materialized = self._nvme.materialize(handle.spill_ref, target_device="cpu_pinned", target_layout="contiguous")
        tensor = materialized.tensor
        self._materialized_spills[handle.spill_ref.stable_id] = tensor
        self._mark_cpu_live_tensor(tensor, handle.tag)
        return tensor
```

Every direct external `handle.tensor` use that may see a spilled handle must become `manager.cpu_tensor(handle)`. Do not leave mixed direct access in MoE helper code. The manager cannot secretly replace a frozen handle object that caller frames already hold; it must keep any re-materialized CPU tensor in manager-owned side storage keyed by the disk handle.

3. Spill policy:

```python
@dataclass
class ActivationSpillConfig:
    enabled: bool
    cpu_budget_bytes: int
    min_spill_bytes: int
    tags: set[str]

def _should_spill(self, handle):
    if not self._spill.enabled:
        return False
    if handle.nbytes < self._spill.min_spill_bytes:
        return False
    if self._spill.tags and handle.tag not in self._spill.tags:
        return False
    return self.stats.cpu_owned_bytes > self._spill.cpu_budget_bytes
```

4. Offload and maybe spill:

```python
def offload(self, tensor, tag):
    handle = self.empty_cpu(...)
    handle.tensor.copy_(tensor.detach(), non_blocking=handle.tensor.is_pinned())
    self._record_offload(handle)
    return self._maybe_spill(handle)

def _maybe_spill(self, handle):
    if not self._should_spill(handle):
        return handle
    self.wait_cpu_ready(handle)
    ref = self._nvme.register_tensor_ref(
        stable_id=f"activation/{unique_id()}",
        role="activation_spill",
        tensor=handle.tensor,
        mutable=False,
    )
    self._nvme.write_cpu(ref, handle.tensor, async_op=True)
    self._nvme.wait_writes_for(ref)  # first implementation: wait before releasing CPU
    self._unmark_cpu_live(handle)
    _return_cpu(handle.tensor, pin_memory=self.pin_memory)
    self.stats.spilled_bytes += handle.nbytes
    return replace(handle, tensor=None, spill_ref=ref, resident="nvme")
```

5. Stage from either CPU or NVMe:

```python
def stage(self, handle, *, tag=None):
    cpu = self.cpu_tensor(handle)
    stage_tag = handle.tag if tag is None else tag
    stage = get_or_alloc_stage(cpu.shape, cpu.dtype, handle.original_device, stage_tag)
    stage.copy_(cpu, non_blocking=cpu.is_pinned())
    self._mark_stage_live(stage, stage_tag)
    return stage
```

7. Release CPU for resident or materialized-spilled handles:

```python
def release_cpu(self, handle):
    if handle is None:
        return 0
    tensor = handle.tensor
    if tensor is not None:
        self.wait_cpu_ready(handle)
    elif handle.spill_ref is not None:
        tensor = self._materialized_spills.pop(handle.spill_ref.stable_id, None)
    if tensor is None:
        return 0
    released = self._unmark_cpu_live_tensor(tensor, handle.tag)
    _return_cpu(tensor, pin_memory=self.pin_memory)
    if handle.spill_ref is not None:
        self._nvme.delete_if_ephemeral(handle.spill_ref)
    return released
```

8. Summary fields:

```python
def snapshot(self):
    base = self.stats.as_dict()
    base.update({
        "activation_nvme_enabled": self._spill.enabled,
        "activation_nvme_spilled_bytes": self.stats.spilled_bytes,
        "activation_nvme_materialized_bytes": self.stats.materialized_bytes,
        "activation_nvme_wait_ms": self.stats.nvme_wait_ms,
        "activation_nvme_by_tag": dict(self.stats.spilled_bytes_by_tag),
    })
    return base
```

Validation before Stage 5:

```bash
.venv/bin/python -m pytest \
  tests/training/test_activation_nvme_spill.py \
  tests/test_lf_memory_breakdown.py \
  -q
```

E2E candidate:

```bash
rm -rf /local_nvme/asymgemm_stage4
OUTPUT_ROOT=profiling_nvme/stage4_activation_nvme \
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
ASYMM_EXP_ACT_POLICIES='none|true|true|true' \
ASYM_NVME_ENABLE=true \
ASYM_NVME_PATH=/local_nvme/asymgemm_stage4 \
ASYM_NVME_ROLES=lora_weights,activation_spill \
ASYM_NVME_CPU_CACHE_BYTES=$((512*1024*1024)) \
ASYM_NVME_ACTIVATION_CPU_BUDGET_BYTES=$((8*1024*1024*1024)) \
ASYM_NVME_ACTIVATION_SPILL_TAGS='X,gate,up,S_gate,S_up,S_down' \
ASYM_NVME_REQUIRE_AIO=true \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
scripts/lf/profile_lora_lf.sh --gpus 0 --overwrite true
```

Compare against a same-activation-offload baseline without activation NVMe:

```bash
.venv/bin/python scripts/lf/compare_nvme_profiles.py \
  --baseline "$BASE" \
  --candidate "$CAND" \
  --target activation_host \
  --memory-metric activation_offload.cpu_peak_bytes_live \
  --min-memory-drop-bytes $((2*1024*1024*1024)) \
  --min-memory-drop-pct 10 \
  --timing-metric trainer.timing.measured_e2e_step_milliseconds \
  --max-latency-regression-pct 5 \
  --require-nvme-role activation_spill
```

Acceptance:

- CPU activation peak/RSS drops meaningfully.
- Forward/backward/e2e timing regression <= 5%.
- `activation_nvme_wait_ms` is not the dominant step time.
- No direct `handle.tensor` access remains for handles that may spill.

Risks to watch:

- Many activation users currently read `handle.tensor` directly. Missing one can crash only under spill pressure.
- Activation reads are on the backward critical path. This stage is likely useful only with careful tag selection and enough CPU cache.
- Immediate wait-after-write is simpler and safe but may reduce overlap. Only add async release after correctness and lifetime tests pass.

## Stage 5: Optional Base-Weight NVMe and Future DeepSpeed Adapter

Purpose: keep later ownership paths possible without delaying the useful local NVMe work. Implement this stage only if profiling shows host memory is still the blocker after Stages 2-4.

Files for optional local base-weight NVMe:

- `asym_gemm/training/frozen_linear.py`
- `asym_gemm/training/offload.py`
- `asym_gemm/training/host_weight.py` if present or added
- model wrappers that request frozen/base weights
- `asym_gemm/training/placement.py` for the existing `TensorRef`/`PlacementBackend` API
- `asym_gemm/training/disk_offload.py` for the local backend implementation

Code changes for optional local base weights:

```python
def materialize_base_weight_for_cpu_gemm(ref):
    # AsymGEMM CPU-left/base paths still need a real CPU tensor pointer.
    materialized = backend.materialize(ref, target_device="cpu_pinned", target_layout="contiguous")
    return materialized.tensor
```

Do not use base-weight NVMe for hot weights unless the CPU cache hit rate is high enough. A blocked NVMe read per layer can erase the benefit.

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

Validation for optional base-weight NVMe:

```bash
OUTPUT_ROOT=profiling_nvme/stage5_base_weight_nvme \
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
ASYM_NVME_PATH=/local_nvme/asymgemm_stage5 \
ASYM_NVME_ROLES=base_weight \
ASYM_NVME_CPU_CACHE_BYTES=$((8*1024*1024*1024)) \
ASYM_NVME_REQUIRE_AIO=true \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
scripts/lf/profile_lora_lf.sh --gpus 0 --overwrite true
```

Acceptance:

- Host memory decreases meaningfully.
- HBM does not regress.
- E2E timing regression <= 5% for default use, <= 10% only for explicit capacity mode.
- Cache hit/miss and NVMe wait stats prove this is not a blocking read on every hot layer.

Risks to watch:

- Base weights are usually hot. Disk-staging them may be worse than CPU residency unless the cache policy is strong.
- DeepSpeed adapter feasibility is good because both DeepSpeed and AsymGEMM already use flat buffers plus shaped views, but ownership must be exclusive.
- Checkpoint semantics must follow the active owner. Local backend writes local checkpoints; future DeepSpeed backend must use DeepSpeed checkpoint APIs.

## Implementation Order

Use this order:

```text
Stage 0: config/profile/compare
Stage 1: disk substrate
Stage 2: LoRA NVMe homes
Stage 3: optimizer-state paging
Stage 4: activation spill
Stage 5: optional base weights / future DeepSpeed adapter
```

Do not start Stage 3 until Stage 2 has a clean accepted profile. Do not start Stage 4 until direct activation handle access has been audited. Do not implement the DeepSpeed adapter until local single-GPU NVMe results prove which roles are actually worth preserving.
