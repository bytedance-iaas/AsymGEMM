from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time
from typing import Any, Callable, Literal, Sequence

import torch

from .exact_pinned import exact_pinned_enabled, register_inplace
from .lora import _is_lora_parameter_name


def _tensor_storage_nbytes(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().nbytes())
    except Exception:
        return int(tensor.numel() * tensor.element_size())


def _tensor_view_key(param: torch.Tensor) -> tuple[str, int, int, tuple[int, ...], tuple[int, ...], str]:
    try:
        storage_ptr = int(param.untyped_storage().data_ptr())
    except Exception:
        storage_ptr = id(param)
    return (
        str(param.device),
        storage_ptr,
        int(param.storage_offset()),
        tuple(int(dim) for dim in param.shape),
        tuple(int(stride) for stride in param.stride()),
        str(param.dtype),
    )


def _pin_if_requested(tensor: torch.Tensor, *, pin_memory: bool) -> tuple[torch.Tensor, str | None]:
    if not pin_memory or not torch.cuda.is_available() or tensor.is_pinned():
        return tensor, None
    if exact_pinned_enabled():
        # capacity fix 2026-07-25: contiguous CPU tensors page-lock in place at
        # exact size (exact_pinned.py); pow2-bucketed pin_memory() is the fallback.
        if tensor.is_contiguous() and register_inplace(tensor) is None:
            return tensor, None
    try:
        return tensor.pin_memory(), None
    except RuntimeError as exc:
        return tensor, str(exc)


def _recursive_cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, dict):
        return {key: _recursive_cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_recursive_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_recursive_cpu_copy(item) for item in value)
    return value


def _recursive_assert_cpu(value: Any, *, path: str = "state") -> None:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise RuntimeError(f"{path} must be CPU-safe, got tensor on {value.device}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _recursive_assert_cpu(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _recursive_assert_cpu(item, path=f"{path}[{index}]")


@dataclass
class _ParamMapping:
    name: str
    aliases: tuple[str, ...]
    cuda_param: torch.nn.Parameter
    cpu_param: torch.nn.Parameter
    grad_buffer: torch.Tensor | None
    model_dtype: torch.dtype
    master_dtype: torch.dtype
    last_had_grad: bool = False
    grad_buffer_has_data: bool = False
    hook_calls: int = 0
    offloaded_grad_numel: int = 0


def _register_post_accumulate_hook(param: torch.nn.Parameter, hook: Callable[[torch.Tensor], None]) -> Any:
    register = getattr(param, "register_post_accumulate_grad_hook", None)
    if callable(register):
        return register(hook)

    param_tmp = param.expand_as(param)
    grad_acc = param_tmp.grad_fn.next_functions[0][0]
    return grad_acc.register_hook(lambda *unused: hook(param))


class AsymCPUAdamW(torch.optim.Optimizer):
    """CPU-master AdamW for AsymGEMM LoRA params.

    The model keeps its trainable LoRA parameters on CUDA for forward/backward.
    This optimizer owns fp32 CPU master copies and CPU AdamW state, copying only
    LoRA grads and updated masters across the device boundary.
    """

    def __init__(
        self,
        named_params: Sequence[tuple[str, torch.nn.Parameter]],
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        backend: Literal["torch", "deepspeed"] = "deepspeed",
        pin_memory: bool = True,
        fp32_master: bool = True,
        grad_offload: bool = False,
        weight_offload: bool = False,
        coordinator: Any = None,
    ) -> None:
        if backend not in {"torch", "deepspeed"}:
            raise ValueError("backend must be either 'torch' or 'deepspeed'")
        if not fp32_master:
            raise ValueError("AsymCPUAdamW v1 requires fp32_master=True")

        selected: list[tuple[str, torch.nn.Parameter]] = []
        for name, param in named_params:
            if not isinstance(param, torch.nn.Parameter):
                raise TypeError(f"{name} is not a torch.nn.Parameter")
            if not param.requires_grad:
                continue
            if not _is_lora_parameter_name(name):
                raise ValueError(f"AsymCPUAdamW can optimize only LoRA parameters; got {name!r}")
            selected.append((name, param))

        if not selected:
            raise ValueError("AsymCPUAdamW found no trainable LoRA parameters")

        if torch.cuda.is_available():
            not_cuda = [(name, param.device) for name, param in selected if param.device.type != "cuda"]
            if not_cuda:
                details = ", ".join(f"{name} on {device}" for name, device in not_cuda[:8])
                raise ValueError(
                    "AsymCPUAdamW v1 requires trainable LoRA compute params to be CUDA nn.Parameters after "
                    "LlamaFactory's post-adapter Asym CPU-first device move. CPU-resident trainable LoRA is "
                    f"Stage 7 and is not supported by CPUAdamW v1. Offending params: {details}"
                )

        unique_entries: list[tuple[str, torch.nn.Parameter, list[str]]] = []
        by_object: dict[int, int] = {}
        by_view: dict[tuple[str, int, int, tuple[int, ...], tuple[int, ...], str], int] = {}
        for name, param in selected:
            object_key = id(param)
            if object_key in by_object:
                unique_entries[by_object[object_key]][2].append(name)
                continue
            view_key = _tensor_view_key(param)
            if view_key in by_view:
                index = by_view[view_key]
                by_object[object_key] = index
                unique_entries[index][2].append(name)
                continue
            index = len(unique_entries)
            by_object[object_key] = index
            by_view[view_key] = index
            unique_entries.append((name, param, []))

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__([{"params": [param for _, param, _ in unique_entries], **defaults}], defaults)

        self.backend = backend
        self.pin_memory = bool(pin_memory)
        self.fp32_master = bool(fp32_master)
        self.grad_offload = bool(grad_offload)
        # Async grad D2H (agent/fix_throughput.md fix #0): the per-param sync copies in
        # the AccumulateGrad hook drained the stream 672x/step (23-26% of step time).
        # Async path enqueues on the producing stream and drains ONCE before any read.
        self.async_grad_offload = os.environ.get(
            "ASYM_CPU_ADAMW_ASYNC_GRAD_OFFLOAD", "1"
        ).strip().lower() not in ("0", "false", "off")
        self._inflight_grad_srcs: list[torch.Tensor] = []
        # Same-dtype pinned staging is REQUIRED for a truly async D2H: the fp32 master
        # grad buffers force a cross-dtype copy, which synchronizes regardless of
        # non_blocking. Stage bf16 grads verbatim; widen to fp32 on CPU at drain.
        self._grad_staging_flat: torch.Tensor | None = None
        self._grad_staging_views: dict[str, torch.Tensor] = {}
        self._pending_grad_converts: list[_ParamMapping] = []
        self.weight_offload = bool(weight_offload)
        # fix_gb200_ep.md P6: under torchrun the ranks pin the torch/OMP pool small for
        # the bwd host ops, starving DeepSpeedCPUAdam's OMP region (measured 5.7 s @1
        # thread vs 0.53 s). When set, the pool is raised ONLY around inner step() and
        # restored after (scoped; |1 runs leave this unset => byte-identical behavior).
        step_threads = os.environ.get("ASYM_CPU_ADAMW_STEP_THREADS", "").strip()
        self._step_threads = int(step_threads) if step_threads.isdigit() else 0
        self._coordinator = coordinator
        self._mappings: list[_ParamMapping] = []
        self._pin_memory_failures: list[str] = []
        self._grad_offload_handles: list[Any] = []
        self._grad_flat_buffer: torch.Tensor | None = None
        self._grad_accum_staging_buffer: torch.Tensor | None = None
        self._post_prepare_checked = False
        self._last_step_grad_param_count = 0
        self._last_step_copyback_param_count = 0
        self._last_step_skipped_copyback_no_grad_param_count = 0
        self._last_grad_copy_ms = 0.0
        self._last_cpu_adam_step_ms = 0.0
        self._last_weight_copyback_ms = 0.0
        self._current_hook_grad_copy_ms = 0.0
        self._last_hook_offloaded_param_count = 0
        self._last_hook_offloaded_numel = 0
        self._last_hook_call_count = 0
        self._last_hook_grad_copy_ms = 0.0
        self._last_step_used_offloaded_grads = False

        cpu_params: list[torch.nn.Parameter] = []
        for name, cuda_param, aliases in unique_entries:
            master_tensor = cuda_param.detach().to(device="cpu", dtype=torch.float32).contiguous()
            master_tensor, pin_error = _pin_if_requested(master_tensor, pin_memory=self.pin_memory)
            if pin_error is not None:
                self._pin_memory_failures.append(f"{name}: {pin_error}")
            cpu_param = torch.nn.Parameter(master_tensor, requires_grad=True)
            cpu_params.append(cpu_param)
            self._mappings.append(
                _ParamMapping(
                    name=name,
                    aliases=tuple(aliases),
                    cuda_param=cuda_param,
                    cpu_param=cpu_param,
                    grad_buffer=None,
                    model_dtype=cuda_param.dtype,
                    master_dtype=cpu_param.dtype,
                )
            )

        self.inner_optimizer = self._create_inner_optimizer(
            cpu_params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self._refresh_visible_state()
        if self.grad_offload:
            self._ensure_grad_offload_flat_buffer()
            self._register_grad_offload_hooks()

    @property
    def param_names(self) -> list[str]:
        return [mapping.name for mapping in self._mappings]

    @property
    def alias_param_names(self) -> list[tuple[str, ...]]:
        return [mapping.aliases for mapping in self._mappings]

    def attach_weight_offload_coordinator(self, coordinator: Any) -> None:
        """Bind a LoRA weight-offload coordinator built after this optimizer.

        Built afterward so the optimizer's fp32 masters are captured from the full GPU weights
        before the coordinator releases them. After this call, copy-back refreshes the CPU home and
        the post-accumulate grad hook releases the backward-staged weight.
        """
        self._coordinator = coordinator
        self.weight_offload = True

    def _create_inner_optimizer(
        self,
        cpu_params: list[torch.nn.Parameter],
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        if self.backend == "torch":
            return torch.optim.AdamW(cpu_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
        except Exception as exc:
            raise RuntimeError(
                "AsymCPUAdamW backend='deepspeed' requires deepspeed.ops.adam.DeepSpeedCPUAdam; "
                "use BACKEND=asym_cpuadamwtorch for correctness testing if the extension is unavailable."
            ) from exc

        try:
            return DeepSpeedCPUAdam(
                cpu_params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                adamw_mode=True,
                fp32_optimizer_states=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "DeepSpeedCPUAdam failed to initialize for AsymCPUAdamW; use BACKEND=asym_cpuadamwtorch "
                "to validate CPU-master correctness while fixing the DeepSpeed CPUAdam extension."
            ) from exc

    def _copy_group_hyperparameters_to_inner(self) -> None:
        for wrapper_group, inner_group in zip(self.param_groups, self.inner_optimizer.param_groups, strict=True):
            for key in ("lr", "betas", "eps", "weight_decay"):
                if key in wrapper_group:
                    inner_group[key] = wrapper_group[key]

    def _check_post_prepare_devices(self) -> None:
        cpu_moved = [mapping.name for mapping in self._mappings if mapping.cpu_param.device.type != "cpu"]
        if cpu_moved:
            raise RuntimeError(
                "accelerator.prepare moved AsymCPUAdamW CPU master params off CPU; "
                f"offending params: {cpu_moved[:8]}"
            )
        if torch.cuda.is_available():
            not_cuda = [mapping.name for mapping in self._mappings if mapping.cuda_param.device.type != "cuda"]
            if not_cuda:
                raise RuntimeError(
                    "accelerator.prepare or model placement moved AsymCPUAdamW LoRA compute params off CUDA; "
                    "CPU-resident trainable LoRA is Stage 7 and is not supported by CPUAdamW v1. "
                    f"Offending params: {not_cuda[:8]}"
                )
        self._post_prepare_checked = True

    def _ensure_grad_buffer(self, mapping: _ParamMapping) -> torch.Tensor:
        if self.grad_offload:
            if mapping.grad_buffer is None:
                self._ensure_grad_offload_flat_buffer()
            if mapping.grad_buffer is None:
                raise RuntimeError(f"missing offload grad buffer for {mapping.name}")
            return mapping.grad_buffer

        current = mapping.grad_buffer
        if current is not None and current.shape == mapping.cpu_param.shape and current.dtype == mapping.cpu_param.dtype:
            return current
        buffer = torch.empty_like(mapping.cpu_param.data, device="cpu", memory_format=torch.contiguous_format)
        buffer, pin_error = _pin_if_requested(buffer, pin_memory=self.pin_memory)
        if pin_error is not None:
            self._pin_memory_failures.append(f"{mapping.name}.grad: {pin_error}")
        mapping.grad_buffer = buffer
        return buffer

    def _ensure_grad_offload_flat_buffer(self) -> torch.Tensor:
        total = sum(int(mapping.cpu_param.numel()) for mapping in self._mappings)
        current = self._grad_flat_buffer
        if current is not None and current.numel() == total and current.dtype == torch.float32:
            return current

        flat = torch.empty(total, device="cpu", dtype=torch.float32)
        flat, pin_error = _pin_if_requested(flat, pin_memory=self.pin_memory)
        if pin_error is not None:
            self._pin_memory_failures.append(f"grad_flat_buffer: {pin_error}")
        self._grad_flat_buffer = flat

        offset = 0
        for mapping in self._mappings:
            numel = int(mapping.cpu_param.numel())
            mapping.grad_buffer = flat.narrow(0, offset, numel).view_as(mapping.cpu_param.data)
            offset += numel

        if self.async_grad_offload:
            stage_specs = [m for m in self._mappings if m.model_dtype == torch.bfloat16]
            stage_total = sum(int(m.cpu_param.numel()) for m in stage_specs)
            if stage_total:
                stage = torch.empty(stage_total, device="cpu", dtype=torch.bfloat16)
                stage, pin_error = _pin_if_requested(stage, pin_memory=self.pin_memory)
                if pin_error is not None:
                    self._pin_memory_failures.append(f"grad_bf16_staging: {pin_error}")
                self._grad_staging_flat = stage
                self._grad_staging_views = {}
                off = 0
                for m in stage_specs:
                    numel = int(m.cpu_param.numel())
                    self._grad_staging_views[m.name] = stage.narrow(0, off, numel).view_as(m.cpu_param.data)
                    off += numel
        return flat

    def _ensure_grad_accum_staging_buffer(self, numel: int) -> torch.Tensor:
        current = self._grad_accum_staging_buffer
        if current is None or current.numel() < numel or current.dtype != torch.float32:
            staging = torch.empty(numel, device="cpu", dtype=torch.float32)
            staging, pin_error = _pin_if_requested(staging, pin_memory=self.pin_memory)
            if pin_error is not None:
                self._pin_memory_failures.append(f"grad_accum_staging_buffer: {pin_error}")
            self._grad_accum_staging_buffer = staging
        return self._grad_accum_staging_buffer.narrow(0, 0, numel)

    def _register_grad_offload_hooks(self) -> None:
        if self._grad_offload_handles:
            return
        for mapping in self._mappings:
            if not mapping.cuda_param.is_leaf:
                raise RuntimeError(f"AsymCPUAdamW grad offload requires leaf CUDA params; got {mapping.name}")
            self._grad_offload_handles.append(
                _register_post_accumulate_hook(
                    mapping.cuda_param,
                    lambda param, mapping=mapping: self._offload_grad_from_hook(mapping, param),
                )
            )

    @torch.no_grad()
    def _offload_grad_from_hook(self, mapping: _ParamMapping, param: torch.Tensor) -> None:
        if not self.grad_offload:
            return
        if param is not mapping.cuda_param:
            raise RuntimeError(f"Grad hook param mismatch for {mapping.name}")

        grad = mapping.cuda_param.grad
        if grad is None:
            return
        if grad.is_sparse:
            raise RuntimeError("AsymCPUAdamW grad offload does not support sparse LoRA grads.")
        if grad.device.type != "cuda":
            raise RuntimeError(f"AsymCPUAdamW grad offload expected CUDA grad for {mapping.name}, got {grad.device}")
        if tuple(grad.shape) != tuple(mapping.cpu_param.shape):
            raise RuntimeError(
                f"AsymCPUAdamW grad shape mismatch for {mapping.name}: "
                f"grad={tuple(grad.shape)}, param={tuple(mapping.cpu_param.shape)}"
            )

        started = time.perf_counter()
        self._copy_or_accumulate_grad_to_cpu(mapping, grad.detach())
        self._current_hook_grad_copy_ms += (time.perf_counter() - started) * 1000.0

        mapping.last_had_grad = True
        mapping.grad_buffer_has_data = True
        mapping.hook_calls += 1
        mapping.offloaded_grad_numel += int(grad.numel())
        mapping.cpu_param.grad = mapping.grad_buffer
        mapping.cuda_param.grad = None
        if self.weight_offload and self._coordinator is not None:
            # Single backward release point: runs strictly after AccumulateGrad wrote param.grad.
            # No-op for non-offloaded params (e.g. attention LoRA).
            self._coordinator.release(mapping.cuda_param)

    def _drain_grad_offload_copies(self) -> None:
        """Wait for in-flight async grad D2H copies: ONE device sync per step instead of
        one blocking copy per param, then widen staged bf16 grads to the fp32 buffers on
        CPU (exact). Must run before anything reads the CPU grad buffers (step /
        grad-norm / clip / zero_grad / accumulate)."""
        if not self._inflight_grad_srcs and not self._pending_grad_converts:
            return
        torch.cuda.synchronize()
        self._inflight_grad_srcs.clear()
        for mapping in self._pending_grad_converts:
            mapping.grad_buffer.copy_(self._grad_staging_views[mapping.name])
        self._pending_grad_converts.clear()

    def _copy_or_accumulate_grad_to_cpu(self, mapping: _ParamMapping, cuda_grad: torch.Tensor) -> None:
        grad_buffer = self._ensure_grad_buffer(mapping)
        if not grad_buffer.is_contiguous():
            raise RuntimeError(f"CPU grad buffer for {mapping.name} must be contiguous")
        if grad_buffer.dtype != mapping.cpu_param.dtype:
            raise RuntimeError(
                f"CPU grad buffer for {mapping.name} has dtype {grad_buffer.dtype}, "
                f"expected {mapping.cpu_param.dtype}"
            )
        if not mapping.grad_buffer_has_data:
            staging = self._grad_staging_views.get(mapping.name) if self.async_grad_offload else None
            if (staging is not None and cuda_grad.is_cuda
                    and cuda_grad.dtype == staging.dtype and staging.is_pinned()):
                # Same-dtype pinned DMA: enqueued on the producing stream, no host block,
                # no stream drain. bf16->fp32 widening happens on CPU at drain time.
                # Keep the CUDA source alive until drained (the hook nulls
                # cuda_param.grad right after this call).
                staging.copy_(cuda_grad, non_blocking=True)
                self._inflight_grad_srcs.append(cuda_grad)
                self._pending_grad_converts.append(mapping)
            elif (self.async_grad_offload and cuda_grad.is_cuda
                    and cuda_grad.dtype == grad_buffer.dtype and grad_buffer.is_pinned()):
                grad_buffer.copy_(cuda_grad, non_blocking=True)
                self._inflight_grad_srcs.append(cuda_grad)
            else:
                if self.async_grad_offload and not getattr(self, "_async_diag_printed", False):
                    self._async_diag_printed = True
                    import sys
                    print(
                        f"[asym-cpu-adamw] async grad offload FELL BACK to sync for {mapping.name}: "
                        f"staging={'None' if staging is None else 'set'} "
                        f"views={len(self._grad_staging_views)} "
                        f"grad(dtype={cuda_grad.dtype},cuda={cuda_grad.is_cuda},contig={cuda_grad.is_contiguous()}) "
                        f"staging_pinned={staging.is_pinned() if staging is not None else 'n/a'} "
                        f"model_dtype={mapping.model_dtype}",
                        file=sys.stderr, flush=True,
                    )
                grad_buffer.copy_(cuda_grad, non_blocking=False)
            return

        # Grad-accumulation path (ga > 1): drain first so the async first-write of this
        # buffer has landed, then keep the original synchronous staging semantics.
        self._drain_grad_offload_copies()
        staging = self._ensure_grad_accum_staging_buffer(int(cuda_grad.numel())).view_as(grad_buffer)
        staging.copy_(cuda_grad, non_blocking=False)
        grad_buffer.add_(staging)

    def _copy_master_to_compute_param(self, mapping: _ParamMapping) -> None:
        if not mapping.cpu_param.data.is_contiguous():
            raise RuntimeError(f"CPU master for {mapping.name} must be contiguous")
        if self.weight_offload and self._coordinator is not None and self._coordinator.is_registered(mapping.cuda_param):
            # Offloaded bank: refresh the pinned bf16 CPU home from the updated fp32 master (CPU cast,
            # no H2D). The next forward/backward gather re-stages it to GPU. cuda_param.data stays a
            # 0-size placeholder until then.
            self._coordinator.refresh_home_from_master(mapping.cuda_param, mapping.cpu_param.data)
            return
        mapping.cuda_param.data.copy_(mapping.cpu_param.data, non_blocking=mapping.cpu_param.data.is_pinned())

    def _refresh_visible_state(self) -> None:
        for mapping in self._mappings:
            visible_state = self.state[mapping.cuda_param]
            visible_state.clear()
            visible_state["cpu_master"] = mapping.cpu_param.data
            inner_state = self.inner_optimizer.state.get(mapping.cpu_param, {})
            if isinstance(inner_state, dict):
                for key, value in inner_state.items():
                    visible_state[key] = value

    def step(self, closure: Any | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self._post_prepare_checked:
            self._check_post_prepare_devices()

        self._drain_grad_offload_copies()
        self._copy_group_hyperparameters_to_inner()
        grad_param_count = 0
        skipped_no_grad = 0
        copyback_count = 0

        with torch.no_grad():
            grad_copy_start = time.perf_counter()
            if self.grad_offload:
                for mapping in self._mappings:
                    if mapping.grad_buffer_has_data:
                        mapping.cpu_param.grad = mapping.grad_buffer
                        mapping.last_had_grad = True
                        grad_param_count += 1
                    else:
                        mapping.cpu_param.grad = None
                        mapping.last_had_grad = False
                        skipped_no_grad += 1
                self._last_grad_copy_ms = 0.0
                self._last_step_used_offloaded_grads = grad_param_count > 0
                self._last_hook_offloaded_param_count = grad_param_count
                self._last_hook_offloaded_numel = sum(
                    int(mapping.offloaded_grad_numel) for mapping in self._mappings if mapping.last_had_grad
                )
                self._last_hook_call_count = sum(int(mapping.hook_calls) for mapping in self._mappings)
                self._last_hook_grad_copy_ms = self._current_hook_grad_copy_ms
            else:
                for mapping in self._mappings:
                    grad = mapping.cuda_param.grad
                    if grad is None:
                        mapping.cpu_param.grad = None
                        mapping.last_had_grad = False
                        skipped_no_grad += 1
                        continue
                    grad_buffer = self._ensure_grad_buffer(mapping)
                    grad_buffer.copy_(grad.detach(), non_blocking=False)
                    if not grad_buffer.is_contiguous():
                        raise RuntimeError(f"CPU grad buffer for {mapping.name} must be contiguous")
                    if grad_buffer.dtype != mapping.cpu_param.dtype:
                        raise RuntimeError(
                            f"CPU grad buffer for {mapping.name} has dtype {grad_buffer.dtype}, "
                            f"expected {mapping.cpu_param.dtype}"
                        )
                    mapping.cpu_param.grad = grad_buffer
                    mapping.last_had_grad = True
                    grad_param_count += 1
                self._last_grad_copy_ms = (time.perf_counter() - grad_copy_start) * 1000.0
                self._last_step_used_offloaded_grads = False
                self._last_hook_offloaded_param_count = 0
                self._last_hook_offloaded_numel = 0
                self._last_hook_call_count = 0
                self._last_hook_grad_copy_ms = 0.0

            step_start = time.perf_counter()
            if grad_param_count:
                if self._step_threads > 0:
                    _prev_threads = torch.get_num_threads()
                    torch.set_num_threads(self._step_threads)
                    try:
                        self.inner_optimizer.step()
                    finally:
                        torch.set_num_threads(_prev_threads)
                else:
                    self.inner_optimizer.step()
            self._last_cpu_adam_step_ms = (time.perf_counter() - step_start) * 1000.0

            self._refresh_visible_state()

            copyback_start = time.perf_counter()
            for mapping in self._mappings:
                if not mapping.last_had_grad:
                    continue
                self._copy_master_to_compute_param(mapping)
                copyback_count += 1
            self._last_weight_copyback_ms = (time.perf_counter() - copyback_start) * 1000.0

        self._last_step_grad_param_count = grad_param_count
        self._last_step_copyback_param_count = copyback_count
        self._last_step_skipped_copyback_no_grad_param_count = skipped_no_grad
        if self.grad_offload:
            # Make step() self-contained: the trainer's zero_grad may never reach this
            # optimizer (wrappers), which previously left grad_buffer_has_data=True so
            # (a) later steps ACCUMULATED onto stale grads (add_) and (b) every later
            # step took the sync accumulate path, defeating the async D2H offload.
            # Clearing after consumption preserves ga>1 (multiple backwards, one step).
            for mapping in self._mappings:
                mapping.grad_buffer_has_data = False
                mapping.hook_calls = 0
                mapping.offloaded_grad_numel = 0
            self._current_hook_grad_copy_ms = 0.0
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        # Buffers get reused next step: never let an in-flight copy land after a zero.
        self._drain_grad_offload_copies()
        super().zero_grad(set_to_none=True if self.grad_offload else set_to_none)
        try:
            self.inner_optimizer.zero_grad(set_to_none=True if self.grad_offload else set_to_none)
        except TypeError:
            self.inner_optimizer.zero_grad()
        for mapping in self._mappings:
            mapping.cuda_param.grad = None
            if self.grad_offload or set_to_none:
                mapping.cpu_param.grad = None
            elif mapping.cpu_param.grad is not None:
                mapping.cpu_param.grad.zero_()
            mapping.grad_buffer_has_data = False
            mapping.last_had_grad = False
            mapping.hook_calls = 0
            mapping.offloaded_grad_numel = 0
        if self.grad_offload:
            self._current_hook_grad_copy_ms = 0.0

    def _sanitized_param_groups(self) -> list[dict[str, Any]]:
        param_index = {id(mapping.cuda_param): index for index, mapping in enumerate(self._mappings)}
        sanitized: list[dict[str, Any]] = []
        for group in self.param_groups:
            group_copy: dict[str, Any] = {}
            for key, value in group.items():
                if key == "params":
                    group_copy[key] = [param_index[id(param)] for param in value]
                elif isinstance(value, torch.Tensor):
                    group_copy[key] = value.detach().to(device="cpu").clone()
                else:
                    group_copy[key] = value
            sanitized.append(group_copy)
        return sanitized

    def state_dict(self) -> dict[str, Any]:
        state = {
            "format": "asym_cpu_adamw_v1",
            "backend": self.backend,
            "pin_memory": self.pin_memory,
            "fp32_master": self.fp32_master,
            "grad_offload": self.grad_offload,
            "param_names": self.param_names,
            "alias_param_names": [list(aliases) for aliases in self.alias_param_names],
            "param_groups": self._sanitized_param_groups(),
            "cpu_master_params": [mapping.cpu_param.detach().to(device="cpu").clone() for mapping in self._mappings],
            "inner_optimizer": _recursive_cpu_copy(self.inner_optimizer.state_dict()),
        }
        _recursive_assert_cpu(state)
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # type: ignore[override]
        if state_dict.get("format") != "asym_cpu_adamw_v1":
            raise ValueError("unsupported AsymCPUAdamW optimizer state format")
        if state_dict.get("backend") != self.backend:
            raise ValueError(
                f"AsymCPUAdamW state backend mismatch: checkpoint={state_dict.get('backend')!r}, current={self.backend!r}"
            )
        if bool(state_dict.get("fp32_master")) != self.fp32_master:
            raise ValueError("AsymCPUAdamW fp32_master mismatch")

        saved_names = list(state_dict.get("param_names") or [])
        if saved_names != self.param_names:
            raise ValueError(f"AsymCPUAdamW parameter names mismatch: checkpoint={saved_names}, current={self.param_names}")

        saved_masters = list(state_dict.get("cpu_master_params") or [])
        if len(saved_masters) != len(self._mappings):
            raise ValueError("AsymCPUAdamW CPU master parameter count mismatch")

        with torch.no_grad():
            for saved, mapping in zip(saved_masters, self._mappings, strict=True):
                if not isinstance(saved, torch.Tensor):
                    raise TypeError(f"CPU master checkpoint entry for {mapping.name} is not a tensor")
                saved_cpu = saved.detach().to(device="cpu", dtype=mapping.cpu_param.dtype).contiguous()
                if tuple(saved_cpu.shape) != tuple(mapping.cpu_param.shape):
                    raise ValueError(
                        f"AsymCPUAdamW CPU master shape mismatch for {mapping.name}: "
                        f"checkpoint={tuple(saved_cpu.shape)}, current={tuple(mapping.cpu_param.shape)}"
                    )
                mapping.cpu_param.data.copy_(saved_cpu)

        saved_groups = list(state_dict.get("param_groups") or [])
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("AsymCPUAdamW param group count mismatch")
        for saved_group, wrapper_group, inner_group in zip(
            saved_groups, self.param_groups, self.inner_optimizer.param_groups, strict=True
        ):
            if list(saved_group.get("params", [])) != list(range(len(self._mappings))):
                raise ValueError("AsymCPUAdamW saved param group order mismatch")
            for key, value in saved_group.items():
                if key == "params":
                    continue
                wrapper_group[key] = value
                inner_group[key] = value

        inner_state = _recursive_cpu_copy(state_dict.get("inner_optimizer") or {})
        self.inner_optimizer.load_state_dict(inner_state)
        self._copy_group_hyperparameters_to_inner()
        self._validate_inner_state()
        self._refresh_visible_state()

        with torch.no_grad():
            for mapping in self._mappings:
                self._copy_master_to_compute_param(mapping)

    def _validate_inner_state(self) -> None:
        for mapping in self._mappings:
            inner_state = self.inner_optimizer.state.get(mapping.cpu_param, {})
            if not isinstance(inner_state, dict):
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                value = inner_state.get(key)
                if value is None:
                    continue
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"AsymCPUAdamW inner state {key} for {mapping.name} is not a tensor")
                if value.device.type != "cpu":
                    raise ValueError(f"AsymCPUAdamW inner state {key} for {mapping.name} is on {value.device}")
                if value.dtype != torch.float32:
                    raise ValueError(f"AsymCPUAdamW inner state {key} for {mapping.name} must be fp32")
                if tuple(value.shape) != tuple(mapping.cpu_param.shape):
                    raise ValueError(f"AsymCPUAdamW inner state {key} shape mismatch for {mapping.name}")

    def asym_cpu_master_params(self) -> list[torch.nn.Parameter]:
        return [mapping.cpu_param for mapping in self._mappings]

    def asym_cpu_param_name_map(self) -> dict[int, str]:
        return {id(mapping.cpu_param): mapping.name for mapping in self._mappings}

    def asym_cuda_param_name_map(self) -> dict[int, str]:
        return {id(mapping.cuda_param): mapping.name for mapping in self._mappings}

    def asym_cpu_adamw_grad_offload_enabled(self) -> bool:
        return bool(self.grad_offload)

    def asym_cpu_adamw_grad_buffers(self) -> list[torch.Tensor]:
        self._drain_grad_offload_copies()
        return [
            mapping.grad_buffer
            for mapping in self._mappings
            if mapping.grad_buffer_has_data and mapping.grad_buffer is not None
        ]

    def asym_cpu_adamw_grad_norm(self, norm_type: float = 2.0, chunk_elements: int = 8_388_608) -> torch.Tensor:
        chunk_elements_i = int(chunk_elements)
        if chunk_elements_i <= 0:
            raise ValueError("chunk_elements must be positive")

        grads = self.asym_cpu_adamw_grad_buffers()
        if not grads:
            return torch.zeros((), dtype=torch.float32, device="cpu")

        norm_type_f = float(norm_type)
        if math.isinf(norm_type_f):
            max_abs = torch.zeros((), dtype=torch.float64, device="cpu")
            for grad in grads:
                flat = grad.detach().reshape(-1)
                for start in range(0, int(flat.numel()), chunk_elements_i):
                    chunk = flat.narrow(0, start, min(chunk_elements_i, int(flat.numel()) - start))
                    max_abs = torch.maximum(max_abs, chunk.abs().max().to(device="cpu", dtype=torch.float64))
            return max_abs.to(dtype=torch.float32)

        total = torch.zeros((), dtype=torch.float64, device="cpu")
        for grad in grads:
            flat = grad.detach().reshape(-1)
            for start in range(0, int(flat.numel()), chunk_elements_i):
                chunk = flat.narrow(0, start, min(chunk_elements_i, int(flat.numel()) - start))
                chunk_f32 = chunk if chunk.dtype in (torch.float32, torch.float64) else chunk.float()
                total += torch.sum(torch.abs(chunk_f32) ** norm_type_f, dtype=torch.float64).cpu()
        return total.pow(1.0 / norm_type_f).to(dtype=torch.float32)

    def _debug_per_param_grad_norms(self) -> None:
        # Gated diagnostic: top offenders by grad norm, for localizing anomalous
        # global norms (e.g. the >1e10 norms seen at long sequence lengths).
        raw = os.environ.get("ASYM_CPU_ADAMW_PER_PARAM_NORM_DEBUG", "0")
        if str(raw).strip().lower() not in {"1", "true", "yes", "on"}:
            return
        norms = []
        for mapping in self._mappings:
            if mapping.grad_buffer_has_data and mapping.grad_buffer is not None:
                norms.append((float(mapping.grad_buffer.detach().float().norm().item()), mapping.name))
        norms.sort(reverse=True)
        top = ", ".join(f"{name}={value:.3e}" for value, name in norms[:16])
        print(f"[asym_cpu_adamw] per-param grad norms (top {min(16, len(norms))} of {len(norms)}): {top}", flush=True)

    def asym_cpu_adamw_clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        chunk_elements: int = 8_388_608,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        self._debug_per_param_grad_norms()
        total_norm = self.asym_cpu_adamw_grad_norm(norm_type=norm_type, chunk_elements=chunk_elements)
        max_norm_f = float(max_norm)
        total_norm_value = float(total_norm.item())
        nonfinite = not math.isfinite(total_norm_value)
        norm_only = math.isinf(max_norm_f)
        if norm_only:
            clip_coef = 1.0
            clip_coef_clamped = 1.0
            clipped = False
            max_norm_summary: float | str = "inf"
        elif math.isnan(total_norm_value):
            clip_coef = float("nan")
            clip_coef_clamped = 1.0
            clipped = False
            max_norm_summary = max_norm_f
        else:
            clip_coef = max_norm_f / (total_norm_value + 1e-6)
            clip_coef_clamped = min(clip_coef, 1.0) if math.isfinite(clip_coef) else clip_coef
            clipped = bool(clip_coef_clamped < 1.0)
            max_norm_summary = max_norm_f
        if clipped:
            for grad in self.asym_cpu_adamw_grad_buffers():
                grad.mul_(clip_coef_clamped)

        grad_buffers = self.asym_cpu_adamw_grad_buffers()
        if clipped and math.isfinite(total_norm_value) and math.isfinite(clip_coef_clamped):
            result_norm_value = total_norm_value * clip_coef_clamped
        else:
            result_norm_value = total_norm_value
        summary = {
            "enabled": True,
            "path": "asym_cpu_adamw_grad_offload",
            "operation": "norm_only" if norm_only else "clip",
            "max_norm": max_norm_summary,
            "norm_type": float(norm_type),
            "total_norm": total_norm_value,
            "result_norm": result_norm_value,
            "clip_coef": float(clip_coef),
            "clip_coef_clamped": float(clip_coef_clamped),
            "clipped": clipped,
            "nonfinite": nonfinite,
            "cpu_grad_tensors": len(grad_buffers),
            "cpu_grad_numel": sum(int(grad.numel()) for grad in grad_buffers),
            "cuda_grad_tensors": 0,
            "chunk_elements": int(chunk_elements),
        }
        return total_norm, summary

    def asym_cpu_adamw_grad_offload_buffer(self) -> torch.Tensor | None:
        return self._grad_flat_buffer

    def asym_cpu_adamw_summary(self) -> dict[str, Any]:
        master_bytes = sum(_tensor_storage_nbytes(mapping.cpu_param.data) for mapping in self._mappings)
        pinned_master_bytes = sum(
            _tensor_storage_nbytes(mapping.cpu_param.data)
            for mapping in self._mappings
            if mapping.cpu_param.data.is_pinned()
        )
        seen_state: set[tuple[str, int, int, str]] = set()
        optimizer_state_cpu_bytes = 0
        for state in self.inner_optimizer.state.values():
            if not isinstance(state, dict):
                continue
            for value in state.values():
                if not isinstance(value, torch.Tensor):
                    continue
                try:
                    key = (
                        str(value.device),
                        int(value.untyped_storage().data_ptr()),
                        int(value.untyped_storage().nbytes()),
                        str(value.dtype),
                    )
                except Exception:
                    key = (str(value.device), id(value), int(value.numel() * value.element_size()), str(value.dtype))
                if key in seen_state:
                    continue
                seen_state.add(key)
                if value.device.type == "cpu":
                    optimizer_state_cpu_bytes += int(key[2])
        grad_buffer_bytes = _tensor_storage_nbytes(self._grad_flat_buffer) if self._grad_flat_buffer is not None else 0
        pinned_grad_buffer_bytes = (
            _tensor_storage_nbytes(self._grad_flat_buffer)
            if self._grad_flat_buffer is not None and self._grad_flat_buffer.is_pinned()
            else 0
        )
        return {
            "enabled": True,
            "backend": self.backend,
            "grad_offload_enabled": bool(self.grad_offload),
            "grad_offload_hook_count": len(self._grad_offload_handles),
            "grad_offload_buffer_bytes": int(grad_buffer_bytes),
            "pinned_grad_offload_buffer_bytes": int(pinned_grad_buffer_bytes),
            "weight_offload_enabled": bool(self.weight_offload),
            **(self._coordinator.summary() if self._coordinator is not None else {}),
            "param_count": len(self._mappings),
            "param_numel": sum(int(mapping.cpu_param.numel()) for mapping in self._mappings),
            "cpu_master_bytes": int(master_bytes),
            "pinned_cpu_master_bytes": int(pinned_master_bytes),
            "optimizer_state_cpu_bytes": int(optimizer_state_cpu_bytes),
            "all_masters_on_cpu": all(mapping.cpu_param.device.type == "cpu" for mapping in self._mappings),
            "all_cuda_params_on_cuda": all(mapping.cuda_param.device.type == "cuda" for mapping in self._mappings),
            "last_step_grad_param_count": int(self._last_step_grad_param_count),
            "last_step_copyback_param_count": int(self._last_step_copyback_param_count),
            "skipped_copyback_no_grad_param_count": int(self._last_step_skipped_copyback_no_grad_param_count),
            "last_step_used_offloaded_grads": bool(self._last_step_used_offloaded_grads),
            "last_hook_offloaded_param_count": int(self._last_hook_offloaded_param_count),
            "last_hook_offloaded_numel": int(self._last_hook_offloaded_numel),
            "last_hook_call_count": int(self._last_hook_call_count),
            "hook_grad_copy_ms": float(self._last_hook_grad_copy_ms),
            "step_grad_copy_ms": float(self._last_grad_copy_ms),
            "grad_copy_ms": float(self._last_grad_copy_ms),
            "cpu_adam_step_ms": float(self._last_cpu_adam_step_ms),
            "weight_copyback_ms": float(self._last_weight_copyback_ms),
            "pin_memory": self.pin_memory,
            "pin_memory_failures": list(self._pin_memory_failures),
            "param_names": self.param_names,
            "alias_param_names": [list(aliases) for aliases in self.alias_param_names],
        }


__all__ = ["AsymCPUAdamW"]
