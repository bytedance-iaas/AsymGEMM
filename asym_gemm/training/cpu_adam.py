from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Literal, Sequence

import torch

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
        self._mappings: list[_ParamMapping] = []
        self._pin_memory_failures: list[str] = []
        self._post_prepare_checked = False
        self._last_step_grad_param_count = 0
        self._last_step_copyback_param_count = 0
        self._last_step_skipped_copyback_no_grad_param_count = 0
        self._last_grad_copy_ms = 0.0
        self._last_cpu_adam_step_ms = 0.0
        self._last_weight_copyback_ms = 0.0

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

    @property
    def param_names(self) -> list[str]:
        return [mapping.name for mapping in self._mappings]

    @property
    def alias_param_names(self) -> list[tuple[str, ...]]:
        return [mapping.aliases for mapping in self._mappings]

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
        current = mapping.grad_buffer
        if current is not None and current.shape == mapping.cpu_param.shape and current.dtype == mapping.cpu_param.dtype:
            return current
        buffer = torch.empty_like(mapping.cpu_param.data, device="cpu", memory_format=torch.contiguous_format)
        buffer, pin_error = _pin_if_requested(buffer, pin_memory=self.pin_memory)
        if pin_error is not None:
            self._pin_memory_failures.append(f"{mapping.name}.grad: {pin_error}")
        mapping.grad_buffer = buffer
        return buffer

    def _copy_master_to_compute_param(self, mapping: _ParamMapping) -> None:
        if not mapping.cpu_param.data.is_contiguous():
            raise RuntimeError(f"CPU master for {mapping.name} must be contiguous")
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

        self._copy_group_hyperparameters_to_inner()
        grad_param_count = 0
        skipped_no_grad = 0
        copyback_count = 0

        with torch.no_grad():
            grad_copy_start = time.perf_counter()
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

            step_start = time.perf_counter()
            if grad_param_count:
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
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        try:
            self.inner_optimizer.zero_grad(set_to_none=set_to_none)
        except TypeError:
            self.inner_optimizer.zero_grad()
        for mapping in self._mappings:
            if set_to_none:
                mapping.cpu_param.grad = None
            elif mapping.cpu_param.grad is not None:
                mapping.cpu_param.grad.zero_()

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
        return {
            "enabled": True,
            "backend": self.backend,
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
            "grad_copy_ms": float(self._last_grad_copy_ms),
            "cpu_adam_step_ms": float(self._last_cpu_adam_step_ms),
            "weight_copyback_ms": float(self._last_weight_copyback_ms),
            "pin_memory": self.pin_memory,
            "pin_memory_failures": list(self._pin_memory_failures),
            "param_names": self.param_names,
            "alias_param_names": [list(aliases) for aliases in self.alias_param_names],
        }


__all__ = ["AsymCPUAdamW"]
