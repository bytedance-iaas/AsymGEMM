from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .host_weight import HostWeight, tensor_nbytes
from .profile_ranges import is_profile_enabled, prof_range


VALID_BACKENDS = ("asym_only", "asym_or_staged", "asym_or_torch", "torch_only")


@dataclass(frozen=True)
class AsymCapability:
    supported: bool
    reason: Optional[str]


@dataclass
class AsymExecutionStats:
    asym_forward_calls: int = 0
    asym_dx_calls: int = 0
    staged_forward_calls: int = 0
    staged_dx_calls: int = 0
    torch_forward_calls: int = 0
    torch_dx_calls: int = 0
    fallback_reasons: Dict[str, int] = field(default_factory=dict)

    def record_fallback(self, reason: str) -> None:
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1

    @property
    def asym_calls(self) -> int:
        return self.asym_forward_calls + self.asym_dx_calls

    @property
    def staged_calls(self) -> int:
        return self.staged_forward_calls + self.staged_dx_calls

    @property
    def torch_calls(self) -> int:
        return self.torch_forward_calls + self.torch_dx_calls

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["asym_calls"] = self.asym_calls
        data["staged_calls"] = self.staged_calls
        data["torch_calls"] = self.torch_calls
        return data


def _check_backend(backend: str) -> None:
    if backend not in VALID_BACKENDS:
        raise ValueError(f"unsupported backend={backend!r}; expected one of {VALID_BACKENDS}")


def _arch_major(device: torch.device) -> Optional[int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return int(torch.cuda.get_device_capability(device)[0])


def _direct_bf16_reason(a: torch.Tensor, b_cpu: torch.Tensor, *, transpose_b: bool = False) -> Optional[str]:
    if not torch.cuda.is_available():
        return "cuda_unavailable"
    if a.device.type != "cuda":
        return "input_not_cuda"
    if _arch_major(a.device) not in {9, 10}:
        return "requires_sm90_or_sm100"
    if a.dim() != 2 or b_cpu.dim() != 2:
        return "requires_2d_operands"
    if a.dtype != torch.bfloat16 or b_cpu.dtype != torch.bfloat16:
        return "requires_bf16"
    if b_cpu.device.type != "cpu":
        return "weight_not_cpu"
    if not b_cpu.is_pinned():
        return "weight_not_pinned"
    if not a.is_contiguous() or not b_cpu.is_contiguous():
        return "requires_contiguous"
    if transpose_b:
        if a.shape[1] != b_cpu.shape[0]:
            return "shape_mismatch"
    elif a.shape[1] != b_cpu.shape[1]:
        return "shape_mismatch"
    k = int(a.shape[1])
    n = int(b_cpu.shape[1] if transpose_b else b_cpu.shape[0])
    if n <= 0 or k <= 0:
        return "requires_positive_nk"
    if n % 8 != 0 or k % 8 != 0:
        return "requires_8_aligned_nk"

    import asym_gemm

    if not hasattr(asym_gemm, "m_grouped_bf16_asym_gemm_nt_contiguous"):
        return "missing_bf16_asym_binding"
    return None


def _direct_grouped_bf16_reason(a: torch.Tensor, b_cpu: torch.Tensor, *, transpose_b: bool = False) -> Optional[str]:
    if not torch.cuda.is_available():
        return "cuda_unavailable"
    if a.device.type != "cuda":
        return "input_not_cuda"
    if _arch_major(a.device) not in {9, 10}:
        return "requires_sm90_or_sm100"
    if a.dim() != 2 or b_cpu.dim() != 3:
        return "requires_2d_input_3d_weight"
    if a.dtype != torch.bfloat16 or b_cpu.dtype != torch.bfloat16:
        return "requires_bf16"
    if b_cpu.device.type != "cpu":
        return "weight_not_cpu"
    if not b_cpu.is_pinned():
        return "weight_not_pinned"
    if not a.is_contiguous() or not b_cpu.is_contiguous():
        return "requires_contiguous"
    if int(b_cpu.shape[0]) <= 0:
        return "requires_positive_groups"
    if transpose_b:
        if a.shape[1] != b_cpu.shape[1]:
            return "shape_mismatch"
        n = int(b_cpu.shape[2])
        k = int(b_cpu.shape[1])
    else:
        if a.shape[1] != b_cpu.shape[2]:
            return "shape_mismatch"
        n = int(b_cpu.shape[1])
        k = int(b_cpu.shape[2])
    if n <= 0 or k <= 0:
        return "requires_positive_nk"
    if n % 8 != 0 or k % 8 != 0:
        return "requires_8_aligned_nk"

    import asym_gemm

    if not hasattr(asym_gemm, "m_grouped_bf16_asym_gemm_nt_contiguous"):
        return "missing_bf16_asym_binding"
    return None


def can_use_direct_bf16(a: torch.Tensor, host_weight: HostWeight, *, transpose: bool = False) -> Tuple[bool, Optional[str]]:
    reason = _direct_bf16_reason(a, host_weight.weight, transpose_b=transpose)
    return reason is None, reason


def _group_metadata_tensors(
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if offsets.dim() != 1 or experts.dim() != 1:
        raise ValueError("offsets and experts must be 1D tensors")
    if experts.numel() < 2:
        raise ValueError("grouped metadata requires at least one group and a sentinel")
    num_groups = int(experts.numel() - 1)
    if offsets.numel() == experts.numel():
        starts = offsets[:-1]
        ends = offsets[1:]
        pair_offsets = torch.stack((starts, ends), dim=1).reshape(-1)
    elif offsets.numel() >= 2 * num_groups:
        pair_offsets = offsets[: 2 * num_groups]
    else:
        raise ValueError(
            "offsets must be cumulative [num_groups + 1] or pairs [2 * num_groups], "
            f"got offsets={offsets.numel()} experts={experts.numel()}"
        )
    offsets_i32 = pair_offsets.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
    experts_i32 = experts.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
    return offsets_i32, experts_i32, int(experts_i32.numel())


def _pad_grouped_input_for_asym(
    a: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    block_m: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, tuple[list[int], list[int], list[int], list[int]] | None]:
    if offsets.numel() != experts.numel():
        return a, offsets, None

    offsets_cpu = offsets.detach().to(device="cpu", dtype=torch.long).tolist()
    num_groups = int(experts.numel() - 1)
    starts = [int(offsets_cpu[i]) for i in range(num_groups)]
    ends = [int(offsets_cpu[i + 1]) for i in range(num_groups)]
    counts = [max(0, end - start) for start, end in zip(starts, ends)]

    padded_offsets = [0]
    for count in counts:
        padded = ((count + block_m - 1) // block_m) * block_m if count > 0 else 0
        padded_offsets.append(padded_offsets[-1] + padded)

    if padded_offsets == offsets_cpu:
        return a, offsets, None

    total_padded = int(padded_offsets[-1])
    padded = a.new_zeros((total_padded, a.shape[1]))
    for start, end, padded_start, count in zip(starts, ends, padded_offsets[:-1], counts):
        if count > 0:
            padded[padded_start : padded_start + count].copy_(a[start:end])
    padded_offsets_t = torch.tensor(padded_offsets, device=a.device, dtype=offsets.dtype)
    return padded, padded_offsets_t, (starts, ends, padded_offsets[:-1], counts)


def _unpad_grouped_output(
    padded: torch.Tensor,
    unpad: tuple[list[int], list[int], list[int], list[int]] | None,
    *,
    output_m: int,
) -> torch.Tensor:
    if unpad is None:
        return padded
    starts, ends, padded_starts, counts = unpad
    out = padded.new_empty((output_m, padded.shape[1]))
    for start, end, padded_start, count in zip(starts, ends, padded_starts, counts):
        if count > 0:
            out[start:end].copy_(padded[padded_start : padded_start + count])
    return out


def _asym_bf16_nt(
    a: torch.Tensor,
    b_cpu: torch.Tensor,
    *,
    compiled_dims: str = "mnk",
    transpose_b: bool = False,
) -> torch.Tensor:
    import asym_gemm

    reason = _direct_bf16_reason(a, b_cpu, transpose_b=transpose_b)
    if reason is not None:
        raise RuntimeError(f"direct BF16 AsymGEMM is unavailable: {reason}")

    m = int(a.shape[0])
    n = int(b_cpu.shape[1] if transpose_b else b_cpu.shape[0])
    # Use FP32 D so multi-K-block kernels reduce partial K tiles in FP32.
    # The public training contract still returns BF16 activations/gradients.
    d = torch.empty((m, n), device=a.device, dtype=torch.float32)
    offsets = torch.tensor([0, m], device=a.device, dtype=torch.int32)
    experts = torch.tensor([0, -1], device=a.device, dtype=torch.int32)
    b_group = b_cpu.unsqueeze(0)
    asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
        a, b_group, d, offsets, experts, 2, compiled_dims, transpose_b
    )
    return d.to(dtype=torch.bfloat16)


def _asym_grouped_bf16_nt(
    a: torch.Tensor,
    b_cpu: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    compiled_dims: str = "mnk",
    transpose_b: bool = False,
) -> torch.Tensor:
    import asym_gemm

    reason = _direct_grouped_bf16_reason(a, b_cpu, transpose_b=transpose_b)
    if reason is not None:
        raise RuntimeError(f"direct grouped BF16 AsymGEMM is unavailable: {reason}")

    m = int(a.shape[0])
    n = int(b_cpu.shape[2] if transpose_b else b_cpu.shape[1])
    a_kernel, offsets_kernel, unpad = _pad_grouped_input_for_asym(a, offsets, experts)
    d = torch.empty((int(a_kernel.shape[0]), n), device=a.device, dtype=torch.float32)
    offsets_i32, experts_i32, list_size = _group_metadata_tensors(offsets_kernel, experts, device=a.device)
    asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
        a_kernel, b_cpu, d, offsets_i32, experts_i32, list_size, compiled_dims, transpose_b
    )
    d = _unpad_grouped_output(d, unpad, output_m=m)
    return d.to(dtype=torch.bfloat16)


def _staged_nt(a: torch.Tensor, b_cpu: torch.Tensor, *, transpose_b: bool = False) -> torch.Tensor:
    b = b_cpu.to(device=a.device, dtype=a.dtype, non_blocking=b_cpu.is_pinned())
    return a @ b if transpose_b else a @ b.t()


def _grouped_torch_chunks(
    a: torch.Tensor,
    b: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    transpose_b: bool = False,
) -> torch.Tensor:
    offsets_cpu = offsets.detach().to(device="cpu", dtype=torch.long).tolist()
    experts_cpu = experts.detach().to(device="cpu", dtype=torch.long).tolist()
    chunks: list[torch.Tensor] = []
    for group_idx, expert_idx in enumerate(experts_cpu[:-1]):
        start = int(offsets_cpu[group_idx])
        end = int(offsets_cpu[group_idx + 1])
        if end <= start:
            continue
        weight = b[int(expert_idx)]
        chunk = a[start:end] @ weight if transpose_b else a[start:end] @ weight.t()
        chunks.append(chunk)
    if chunks:
        return torch.cat(chunks, dim=0)
    n = int(b.shape[2] if transpose_b else b.shape[1])
    return a.new_empty((0, n))


def _staged_grouped_nt(
    a: torch.Tensor,
    b_cpu: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    transpose_b: bool = False,
) -> torch.Tensor:
    b = b_cpu.to(device=a.device, dtype=a.dtype, non_blocking=b_cpu.is_pinned())
    return _grouped_torch_chunks(a, b, offsets, experts, transpose_b=transpose_b)


def _torch_nt(a: torch.Tensor, b_cpu: torch.Tensor, *, transpose_b: bool = False) -> torch.Tensor:
    if a.device.type == "cpu":
        b = b_cpu.to(dtype=a.dtype)
        return a @ b if transpose_b else a @ b.t()
    b = b_cpu.to(dtype=a.dtype)
    out_cpu = a.to(device="cpu", non_blocking=False) @ (b if transpose_b else b.t())
    return out_cpu.to(device=a.device, non_blocking=False)


def _torch_grouped_nt(
    a: torch.Tensor,
    b_cpu: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    transpose_b: bool = False,
) -> torch.Tensor:
    if a.device.type == "cpu":
        b = b_cpu.to(dtype=a.dtype)
        return _grouped_torch_chunks(a, b, offsets, experts, transpose_b=transpose_b)
    b = b_cpu.to(dtype=a.dtype)
    out_cpu = _grouped_torch_chunks(
        a.to(device="cpu", non_blocking=False),
        b,
        offsets,
        experts,
        transpose_b=transpose_b,
    )
    return out_cpu.to(device=a.device, non_blocking=False)


def _dispatch_nt(
    a: torch.Tensor,
    b_cpu: torch.Tensor,
    *,
    backend: str,
    stats: Optional[AsymExecutionStats],
    phase: str,
    compiled_dims: str,
    transpose_b: bool = False,
    profile_label: str = "",
) -> torch.Tensor:
    _check_backend(backend)

    if backend != "torch_only":
        reason = _direct_bf16_reason(a, b_cpu, transpose_b=transpose_b)
        if reason is None:
            try:
                out = _asym_bf16_nt(a, b_cpu, compiled_dims=compiled_dims, transpose_b=transpose_b)
                if stats is not None:
                    if phase == "forward":
                        stats.asym_forward_calls += 1
                    else:
                        stats.asym_dx_calls += 1
                return out
            except RuntimeError as exc:
                reason = f"direct_runtime_error:{type(exc).__name__}"
                if backend == "asym_only":
                    raise
        if stats is not None:
            stats.record_fallback(f"{phase}:{reason}")
        if backend == "asym_only":
            raise RuntimeError(f"direct BF16 AsymGEMM is unavailable: {reason}")

    if backend == "asym_or_staged":
        if stats is not None:
            if phase == "forward":
                stats.staged_forward_calls += 1
            else:
                stats.staged_dx_calls += 1
        return _staged_nt(a, b_cpu, transpose_b=transpose_b)

    if stats is not None:
        if phase == "forward":
            stats.torch_forward_calls += 1
        else:
            stats.torch_dx_calls += 1
    return _torch_nt(a, b_cpu, transpose_b=transpose_b)


def _dispatch_grouped_nt(
    a: torch.Tensor,
    b_cpu: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    backend: str,
    stats: Optional[AsymExecutionStats],
    phase: str,
    compiled_dims: str,
    transpose_b: bool = False,
) -> torch.Tensor:
    _check_backend(backend)

    if backend != "torch_only":
        reason = _direct_grouped_bf16_reason(a, b_cpu, transpose_b=transpose_b)
        if reason is None:
            try:
                out = _asym_grouped_bf16_nt(
                    a,
                    b_cpu,
                    offsets,
                    experts,
                    compiled_dims=compiled_dims,
                    transpose_b=transpose_b,
                )
                if stats is not None:
                    if phase == "forward":
                        stats.asym_forward_calls += 1
                    else:
                        stats.asym_dx_calls += 1
                return out
            except RuntimeError as exc:
                reason = f"direct_runtime_error:{type(exc).__name__}"
                if backend == "asym_only":
                    raise
        if stats is not None:
            stats.record_fallback(f"{phase}:{reason}")
        if backend == "asym_only":
            raise RuntimeError(f"direct grouped BF16 AsymGEMM is unavailable: {reason}")

    if backend == "asym_or_staged":
        if stats is not None:
            if phase == "forward":
                stats.staged_forward_calls += 1
            else:
                stats.staged_dx_calls += 1
        return _staged_grouped_nt(a, b_cpu, offsets, experts, transpose_b=transpose_b)

    if stats is not None:
        if phase == "forward":
            stats.torch_forward_calls += 1
        else:
            stats.torch_dx_calls += 1
    return _torch_grouped_nt(a, b_cpu, offsets, experts, transpose_b=transpose_b)


class AsymFrozenLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        host_weight: HostWeight,
        bias: Optional[torch.Tensor],
        backend: str,
        stats: Optional[AsymExecutionStats],
        compiled_dims: str,
        profile_name: str,
    ) -> torch.Tensor:
        if x.shape[-1] != host_weight.in_features:
            raise ValueError(f"expected input last dim {host_weight.in_features}, got {x.shape[-1]}")
        input_shape = tuple(x.shape)
        x_2d = x.reshape(-1, host_weight.in_features).contiguous()
        forward_range = "forward.base_frozen_asymgemm"
        backward_range = "backward.base_dx_asymgemm" if not profile_name else f"backward.{profile_name}.base_dx_asymgemm"
        with prof_range(forward_range):
            y = _dispatch_nt(
                x_2d,
                host_weight.weight,
                backend=backend,
                stats=stats,
                phase="forward",
                compiled_dims=compiled_dims,
                profile_label=forward_range,
            )
        if bias is not None:
            y = y + bias.to(device=y.device, dtype=y.dtype)

        ctx.host_weight = host_weight
        ctx.backend = backend
        ctx.stats = stats
        ctx.compiled_dims = compiled_dims
        ctx.profile_backward_range = backward_range
        ctx.profile_enabled = is_profile_enabled()
        ctx.has_bias = bias is not None
        ctx.bias_device = bias.device if bias is not None else None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.input_shape = input_shape
        return y.reshape(*input_shape[:-1], host_weight.out_features)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None, Optional[torch.Tensor], None, None, None, None]:
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_output_2d = grad_output.reshape(-1, ctx.host_weight.out_features).contiguous()
            with prof_range(ctx.profile_backward_range, enabled=ctx.profile_enabled):
                grad_x = _dispatch_nt(
                    grad_output_2d,
                    ctx.host_weight.weight,
                    backend=ctx.backend,
                    stats=ctx.stats,
                    phase="dx",
                    compiled_dims=ctx.compiled_dims,
                    transpose_b=True,
                    profile_label=ctx.profile_backward_range,
                )
            grad_x = grad_x.reshape(ctx.input_shape)

        grad_bias = None
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = (
                grad_output.reshape(-1, ctx.host_weight.out_features)
                .sum(dim=0)
                .to(device=ctx.bias_device, dtype=ctx.bias_dtype)
            )
        return grad_x, None, grad_bias, None, None, None, None


def asym_frozen_linear(
    x: torch.Tensor,
    host_weight: HostWeight,
    *,
    bias: Optional[torch.Tensor] = None,
    backend: str = "asym_or_staged",
    stats: Optional[AsymExecutionStats] = None,
    compiled_dims: str = "mnk",
    profile_name: str = "",
) -> torch.Tensor:
    return AsymFrozenLinearFunction.apply(x, host_weight, bias, backend, stats, compiled_dims, profile_name)


def frozen_linear(
    x: torch.Tensor,
    host_weight: HostWeight,
    bias: Optional[torch.Tensor] = None,
    *,
    backend: str = "asym_or_torch",
    stats: Optional[AsymExecutionStats] = None,
    compiled_dims: str = "mnk",
    profile_name: str = "",
) -> torch.Tensor:
    return asym_frozen_linear(
        x,
        host_weight,
        bias=bias,
        backend=backend,
        stats=stats,
        compiled_dims=compiled_dims,
        profile_name=profile_name,
    )


class AsymGroupedFrozenLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        host_weight: HostWeight,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        backend: str,
        stats: Optional[AsymExecutionStats],
        compiled_dims: str,
        profile_name: str,
    ) -> torch.Tensor:
        if host_weight.weight.dim() != 3:
            raise ValueError(f"grouped host weight must be 3D, got shape {tuple(host_weight.weight.shape)}")
        if x.shape[-1] != int(host_weight.weight.shape[2]):
            raise ValueError(f"expected input last dim {int(host_weight.weight.shape[2])}, got {x.shape[-1]}")
        input_shape = tuple(x.shape)
        x_2d = x.reshape(-1, int(host_weight.weight.shape[2])).contiguous()
        out_features = int(host_weight.weight.shape[1])
        forward_range = "forward.grouped_base_frozen_asymgemm"
        backward_range = "backward.grouped_base_dx_asymgemm" if not profile_name else f"backward.{profile_name}.grouped_base_dx_asymgemm"
        with prof_range(forward_range):
            y = _dispatch_grouped_nt(
                x_2d,
                host_weight.weight,
                offsets,
                experts,
                backend=backend,
                stats=stats,
                phase="forward",
                compiled_dims=compiled_dims,
            )

        ctx.host_weight = host_weight
        ctx.offsets = offsets
        ctx.experts = experts
        ctx.backend = backend
        ctx.stats = stats
        ctx.compiled_dims = compiled_dims
        ctx.profile_backward_range = backward_range
        ctx.profile_enabled = is_profile_enabled()
        ctx.input_shape = input_shape
        return y.reshape(*input_shape[:-1], out_features)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], None, None, None, None, None, None, None]:
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_output_2d = grad_output.reshape(-1, int(ctx.host_weight.weight.shape[1])).contiguous()
            weight_t = ctx.host_weight.transposed_tensor()
            with prof_range(ctx.profile_backward_range, enabled=ctx.profile_enabled):
                grad_x = _dispatch_grouped_nt(
                    grad_output_2d,
                    weight_t,
                    ctx.offsets,
                    ctx.experts,
                    backend=ctx.backend,
                    stats=ctx.stats,
                    phase="dx",
                    compiled_dims=ctx.compiled_dims,
                )
            grad_x = grad_x.reshape(ctx.input_shape)
        return grad_x, None, None, None, None, None, None, None


def asym_grouped_frozen_linear(
    x: torch.Tensor,
    host_weight: HostWeight,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    backend: str = "asym_or_staged",
    stats: Optional[AsymExecutionStats] = None,
    compiled_dims: str = "mnk",
    profile_name: str = "",
) -> torch.Tensor:
    return AsymGroupedFrozenLinearFunction.apply(
        x,
        host_weight,
        offsets,
        experts,
        backend,
        stats,
        compiled_dims,
        profile_name,
    )


class AsymGroupedFrozenLinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        *,
        backend: str = "asym_or_staged",
        pin_memory: bool = True,
        stats: Optional[AsymExecutionStats] = None,
        compiled_dims: str = "mnk",
    ) -> None:
        super().__init__()
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"weight must be a torch.Tensor, got {type(weight)!r}")
        if weight.dim() != 3:
            raise ValueError(f"AsymGroupedFrozenLinear expects [groups, out, in], got {tuple(weight.shape)}")
        _check_backend(backend)
        self.host_weight = HostWeight(weight, pin_memory=pin_memory, clone=True, require_2d=False)
        self.backend = backend
        self.stats = stats if stats is not None else AsymExecutionStats()
        self.compiled_dims = compiled_dims
        self.profile_name = ""
        self.num_groups = int(self.host_weight.weight.shape[0])
        self.out_features = int(self.host_weight.weight.shape[1])
        self.in_features = int(self.host_weight.weight.shape[2])

    @property
    def pinned_cpu_bytes(self) -> int:
        return self.host_weight.pinned_cpu_bytes

    @property
    def weight_hbm_saved_bytes(self) -> int:
        return self.host_weight.weight_nbytes

    @property
    def weight(self) -> torch.Tensor:
        return self.host_weight.weight

    def forward(self, x: torch.Tensor, offsets: torch.Tensor, experts: torch.Tensor) -> torch.Tensor:
        return asym_grouped_frozen_linear(
            x,
            self.host_weight,
            offsets,
            experts,
            backend=self.backend,
            stats=self.stats,
            compiled_dims=self.compiled_dims,
            profile_name=self.profile_name,
        )

    def _save_to_state_dict(self, destination: Dict[str, torch.Tensor], prefix: str, keep_vars: bool) -> None:
        super()._save_to_state_dict(destination, prefix, keep_vars)
        weight = self.host_weight.weight
        destination[prefix + "host_weight"] = weight if keep_vars else weight.detach()

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        weight_key = prefix + "host_weight"
        if weight_key in state_dict:
            weight = state_dict.pop(weight_key)
            if weight.dim() != 3:
                error_msgs.append(f"{weight_key} must be 3D, got shape {tuple(weight.shape)}")
            else:
                self.host_weight = HostWeight(
                    weight,
                    pin_memory=self.host_weight.is_pinned,
                    clone=True,
                    require_2d=False,
                )
                self.num_groups = int(self.host_weight.weight.shape[0])
                self.out_features = int(self.host_weight.weight.shape[1])
                self.in_features = int(self.host_weight.weight.shape[2])
        elif strict:
            missing_keys.append(weight_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extra_repr(self) -> str:
        return (
            f"num_groups={self.num_groups}, in_features={self.in_features}, "
            f"out_features={self.out_features}, backend={self.backend}, "
            f"pinned={self.host_weight.metadata.pinned}"
        )


def direct_asym_capability(
    a: torch.Tensor,
    host_weight: HostWeight,
    *,
    transposed_weight: bool = False,
) -> AsymCapability:
    supported, reason = can_use_direct_bf16(a, host_weight, transpose=transposed_weight)
    return AsymCapability(supported=supported, reason=reason)


class AsymFrozenLinear(nn.Module):
    def __init__(
        self,
        *args: Any,
        bias: Optional[torch.Tensor] = None,
        backend: str = "asym_or_staged",
        pin_memory: bool = True,
        stats: Optional[AsymExecutionStats] = None,
        compiled_dims: str = "mnk",
    ) -> None:
        super().__init__()
        if len(args) == 1:
            weight = args[0]
        elif len(args) == 3:
            in_features, out_features, weight = args
            if tuple(weight.shape) != (int(out_features), int(in_features)):
                raise ValueError(
                    f"weight has shape {tuple(weight.shape)}, expected {(int(out_features), int(in_features))}"
                )
        else:
            raise TypeError("AsymFrozenLinear expects weight or (in_features, out_features, weight)")
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"weight must be a torch.Tensor, got {type(weight)!r}")
        _check_backend(backend)
        self.host_weight = HostWeight.from_tensor(weight, dtype=weight.dtype, pin_memory=pin_memory)
        self.bias_cpu = None if bias is None else bias.detach().to("cpu", dtype=weight.dtype).contiguous()
        if self.bias_cpu is not None and pin_memory:
            try:
                self.bias_cpu = self.bias_cpu.pin_memory()
            except RuntimeError:
                pass
        self.backend = backend
        self.stats = stats if stats is not None else AsymExecutionStats()
        self.compiled_dims = compiled_dims
        self.profile_name = ""
        self.in_features = self.host_weight.in_features
        self.out_features = self.host_weight.out_features

    @classmethod
    def from_gpu_linear(
        cls,
        linear: nn.Linear,
        *,
        backend: str = "asym_or_staged",
        pin_memory: bool = True,
        stats: Optional[AsymExecutionStats] = None,
        compiled_dims: str = "mnk",
    ) -> "AsymFrozenLinear":
        return cls(
            linear.weight.detach(),
            bias=None if linear.bias is None else linear.bias.detach(),
            backend=backend,
            pin_memory=pin_memory,
            stats=stats,
            compiled_dims=compiled_dims,
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        backend: str = "asym_or_staged",
        pin_memory: bool = True,
        stats: Optional[AsymExecutionStats] = None,
        compiled_dims: str = "mnk",
    ) -> "AsymFrozenLinear":
        return cls.from_gpu_linear(
            linear,
            backend=backend,
            pin_memory=pin_memory,
            stats=stats,
            compiled_dims=compiled_dims,
        )

    @property
    def pinned_cpu_bytes(self) -> int:
        total = self.host_weight.pinned_cpu_bytes
        if self.bias_cpu is not None and self.bias_cpu.is_pinned():
            total += tensor_nbytes(self.bias_cpu)
        return total

    @property
    def weight_hbm_saved_bytes(self) -> int:
        return self.host_weight.weight_nbytes

    @property
    def weight(self) -> torch.Tensor:
        return self.host_weight.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.bias_cpu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return asym_frozen_linear(
            x,
            self.host_weight,
            bias=self.bias_cpu,
            backend=self.backend,
            stats=self.stats,
            compiled_dims=self.compiled_dims,
            profile_name=self.profile_name,
        )

    def _save_to_state_dict(self, destination: Dict[str, torch.Tensor], prefix: str, keep_vars: bool) -> None:
        super()._save_to_state_dict(destination, prefix, keep_vars)
        weight = self.host_weight.weight
        destination[prefix + "host_weight"] = weight if keep_vars else weight.detach()
        if self.bias_cpu is not None:
            destination[prefix + "bias_cpu"] = self.bias_cpu if keep_vars else self.bias_cpu.detach()

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        weight_key = prefix + "host_weight"
        bias_key = prefix + "bias_cpu"

        if weight_key in state_dict:
            weight = state_dict.pop(weight_key)
            if weight.dim() != 2:
                error_msgs.append(f"{weight_key} must be 2D, got shape {tuple(weight.shape)}")
            else:
                self.host_weight = HostWeight.from_tensor(
                    weight,
                    dtype=weight.dtype,
                    pin_memory=self.host_weight.is_pinned,
                )
                self.in_features = self.host_weight.in_features
                self.out_features = self.host_weight.out_features
        elif strict:
            missing_keys.append(weight_key)

        if self.bias_cpu is not None:
            if bias_key in state_dict:
                bias = state_dict.pop(bias_key).detach().to("cpu", dtype=self.host_weight.dtype).contiguous()
                if tuple(bias.shape) != (self.out_features,):
                    error_msgs.append(f"{bias_key} must have shape {(self.out_features,)}, got {tuple(bias.shape)}")
                else:
                    if self.bias_cpu.is_pinned():
                        try:
                            bias = bias.pin_memory()
                        except RuntimeError:
                            pass
                    self.bias_cpu = bias
            elif strict:
                missing_keys.append(bias_key)
        elif bias_key in state_dict:
            state_dict.pop(bias_key)
            unexpected_keys.append(bias_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"backend={self.backend}, pinned={self.host_weight.metadata.pinned}"
        )


def measure_gpu_weight_allocation(weight: torch.Tensor, *, device: Optional[torch.device] = None) -> int:
    if not torch.cuda.is_available():
        return 0
    device = device or torch.device("cuda")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated(device)
    gpu_weight = weight.detach().to(device=device)
    torch.cuda.synchronize(device)
    allocated = max(0, torch.cuda.memory_allocated(device) - before)
    del gpu_weight
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return int(allocated)
