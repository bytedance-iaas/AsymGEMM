from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from asym_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8, per_token_cast_to_nvfp4_e4m3

from .host_weight import HostWeight, tensor_nbytes
from .profile_ranges import is_profile_enabled, prof_range


VALID_BACKENDS = ("asym", "torch")
VALID_ASYM_PRECISIONS = ("bf16", "fp8", "fp4")
_FP8_RECIPE = (1, 128, 128)
_FP4_RECIPE = (1, 1, 16)
_TORCH_GROUPED_MM = getattr(torch.nn.functional, "grouped_mm", None)
_TORCH_GROUPED_MM_NAME = "torch.nn.functional.grouped_mm"
if _TORCH_GROUPED_MM is None:
    _TORCH_GROUPED_MM = getattr(torch, "_grouped_mm", None)
    _TORCH_GROUPED_MM_NAME = "torch._grouped_mm"
if _TORCH_GROUPED_MM is None:
    raise RuntimeError("PyTorch grouped torch baseline requires torch.nn.functional.grouped_mm or torch._grouped_mm")

_SINGLE_GROUP_LAUNCH_TENSOR_CACHE: dict[
    tuple[str, int], tuple[torch.Tensor, torch.Tensor]
] = {}


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
    kt_forward_calls: int = 0
    kt_backward_calls: int = 0
    kt_lora_update_calls: int = 0
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


@dataclass(frozen=True)
class QuantizedHostWeight:
    precision: str
    values: torch.Tensor
    scales: torch.Tensor
    logical_shape: tuple[int, ...]

    @property
    def num_groups(self) -> int:
        return int(self.logical_shape[0]) if len(self.logical_shape) == 3 else 1

    @property
    def out_features(self) -> int:
        return int(self.logical_shape[-2])

    @property
    def in_features(self) -> int:
        return int(self.logical_shape[-1])

    @property
    def nbytes(self) -> int:
        return tensor_nbytes(self.values) + tensor_nbytes(self.scales)

    @property
    def pinned_cpu_bytes(self) -> int:
        total = 0
        if self.values.device.type == "cpu" and self.values.is_pinned():
            total += tensor_nbytes(self.values)
        if self.scales.device.type == "cpu" and self.scales.is_pinned():
            total += tensor_nbytes(self.scales)
        return total


def _check_backend(backend: str) -> None:
    if backend not in VALID_BACKENDS:
        raise ValueError(f"unsupported backend={backend!r}; expected one of {VALID_BACKENDS}")


def is_asym_backend(backend: str) -> bool:
    return backend == "asym"


def is_torch_backend(backend: str) -> bool:
    return backend == "torch"


def is_kt_backend(backend: str) -> bool:
    return backend == "kt"


def _normalize_precision(precision: str) -> str:
    normalized = str(precision).lower()
    if normalized not in VALID_ASYM_PRECISIONS:
        raise ValueError(f"unsupported precision={precision!r}; expected one of {VALID_ASYM_PRECISIONS}")
    return normalized


def _pin_cpu_tensor(tensor: torch.Tensor, *, pin_memory: bool) -> torch.Tensor:
    out = tensor.detach()
    if out.device.type != "cpu":
        out = out.to(device="cpu", non_blocking=False)
    if not out.is_contiguous():
        out = out.contiguous()
    if pin_memory and torch.cuda.is_available() and not out.is_pinned():
        try:
            out = out.pin_memory()
        except RuntimeError:
            pass
    out.requires_grad_(False)
    return out


def _transpose_source_for_quantization(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor.t().contiguous()
    if tensor.dim() == 3:
        return tensor.transpose(-1, -2).contiguous()
    raise ValueError(f"cannot transpose host weight with shape {tuple(tensor.shape)}")


def _quantize_host_weight_2d(weight: torch.Tensor, precision: str) -> tuple[torch.Tensor, torch.Tensor]:
    source = weight.detach()
    if source.device.type != "cpu":
        source = source.to(device="cpu", non_blocking=False)
    if not source.is_contiguous():
        source = source.contiguous()
    if source.dtype != torch.bfloat16:
        source = source.to(dtype=torch.bfloat16)

    if precision == "fp8":
        return per_block_cast_to_fp8(source, use_ue8m0=True, gran_k=128)
    if precision == "fp4":
        return per_token_cast_to_nvfp4_e4m3(source, gran_k=16)
    raise ValueError(f"cannot quantize host weight for precision={precision!r}")


def _quantize_host_weight_tensor(
    weight: torch.Tensor,
    precision: str,
    *,
    pin_memory: bool,
) -> QuantizedHostWeight:
    if weight.dim() == 2:
        values, scales = _quantize_host_weight_2d(weight, precision)
    elif weight.dim() == 3:
        values_list: list[torch.Tensor] = []
        scales_list: list[torch.Tensor] = []
        for group in range(int(weight.shape[0])):
            group_values, group_scales = _quantize_host_weight_2d(weight[group], precision)
            values_list.append(group_values)
            scales_list.append(group_scales)
        values = torch.stack(values_list, dim=0).contiguous()
        scales = torch.stack(scales_list, dim=0).contiguous()
    else:
        raise ValueError(f"quantized host weight expects 2D or 3D tensor, got {tuple(weight.shape)}")

    return QuantizedHostWeight(
        precision=precision,
        values=_pin_cpu_tensor(values, pin_memory=pin_memory),
        scales=_pin_cpu_tensor(scales, pin_memory=pin_memory),
        logical_shape=tuple(int(dim) for dim in weight.shape),
    )


def _get_quantized_host_weight(
    host_weight: HostWeight,
    precision: str,
    *,
    transpose: bool = False,
) -> Optional[QuantizedHostWeight]:
    precision = _normalize_precision(precision)
    if precision == "bf16":
        return None

    cache = getattr(host_weight, "_asym_quantized_cache", None)
    if cache is None:
        cache = {}
        setattr(host_weight, "_asym_quantized_cache", cache)
    key = (precision, bool(transpose))
    cached = cache.get(key)
    if cached is not None:
        return cached

    source = _transpose_source_for_quantization(host_weight.weight) if transpose else host_weight.weight
    if precision == "fp4" and int(source.shape[-1]) % 2 != 0:
        return None
    quantized = _quantize_host_weight_tensor(source, precision, pin_memory=host_weight.is_pinned)
    cache[key] = quantized
    return quantized


def _quantized_cache_pinned_bytes(host_weight: HostWeight, precision: str) -> int:
    cache = getattr(host_weight, "_asym_quantized_cache", None)
    if not cache:
        return 0
    return sum(
        int(qweight.pinned_cpu_bytes)
        for (cached_precision, _), qweight in cache.items()
        if cached_precision == precision
    )


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
    if transpose_b and k % 64 != 0:
        return "transpose_b_requires_64_aligned_k"

    import asym_gemm

    if not hasattr(asym_gemm, "m_grouped_bf16_asym_gemm_nt_contiguous"):
        return "missing_bf16_asym_binding"
    return None


def _direct_quantized_reason(
    a: torch.Tensor,
    qweight: Optional[QuantizedHostWeight],
    *,
    precision: str,
    grouped: bool = False,
) -> Optional[str]:
    precision = _normalize_precision(precision)
    if precision == "bf16":
        return "requires_quantized_precision"
    if qweight is None:
        return "requires_quantized_host_weight"
    if qweight.precision != precision:
        return "quantized_precision_mismatch"
    if not torch.cuda.is_available():
        return "cuda_unavailable"
    if a.device.type != "cuda":
        return "input_not_cuda"
    arch = _arch_major(a.device)
    if precision == "fp8" and arch not in {9, 10}:
        return "requires_sm90_or_sm100_for_fp8"
    if precision == "fp4" and arch != 10:
        return "requires_sm100_for_fp4"
    if a.dim() != 2:
        return "requires_2d_input"
    expected_weight_dim = 3 if grouped else 2
    if len(qweight.logical_shape) != expected_weight_dim or qweight.values.dim() != expected_weight_dim:
        return "requires_3d_quantized_weight" if grouped else "requires_2d_quantized_weight"
    if a.dtype != torch.bfloat16:
        return "requires_bf16_input"
    if not a.is_contiguous() or not qweight.values.is_contiguous() or not qweight.scales.is_contiguous():
        return "requires_contiguous"
    if qweight.values.device.type != "cpu" or qweight.scales.device.type != "cpu":
        return "quantized_weight_not_cpu"
    if not qweight.values.is_pinned() or not qweight.scales.is_pinned():
        return "quantized_weight_not_pinned"

    groups = int(qweight.logical_shape[0]) if grouped else 1
    n = int(qweight.logical_shape[-2])
    k = int(qweight.logical_shape[-1])
    if groups <= 0:
        return "requires_positive_groups"
    if n <= 0 or k <= 0:
        return "requires_positive_nk"
    if int(a.shape[1]) != k:
        return "shape_mismatch"
    if n % 128 != 0 or k % 128 != 0:
        return "requires_128_aligned_nk"

    if precision == "fp8":
        if qweight.values.dtype != torch.float8_e4m3fn:
            return "requires_fp8_quantized_values"
        if tuple(qweight.values.shape) != qweight.logical_shape:
            return "quantized_shape_mismatch"
        import asym_gemm

        if not hasattr(asym_gemm, "m_grouped_fp8_asym_gemm_nt_contiguous"):
            return "missing_fp8_asym_binding"
    elif precision == "fp4":
        if k % 2 != 0:
            return "requires_even_k_for_fp4"
        if qweight.values.dtype != torch.uint8 or qweight.scales.dtype != torch.float8_e4m3fn:
            return "requires_fp4_quantized_values"
        expected_values_shape = (*qweight.logical_shape[:-1], k // 2)
        if tuple(qweight.values.shape) != expected_values_shape:
            return "quantized_shape_mismatch"
        import asym_gemm

        if not hasattr(asym_gemm, "m_grouped_fp4_asym_gemm_nt_contiguous"):
            return "missing_fp4_asym_binding"
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


def _single_group_launch_tensors(
    device: torch.device, m: int
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (str(device), int(m))
    cached = _SINGLE_GROUP_LAUNCH_TENSOR_CACHE.get(key)
    if cached is not None:
        return cached
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "single-group AsymGEMM launch tensors must be initialized before CUDA graph capture"
        )
    offsets = torch.tensor([0, int(m)], device=device, dtype=torch.int32)
    experts = torch.tensor([0, -1], device=device, dtype=torch.int32)
    _SINGLE_GROUP_LAUNCH_TENSOR_CACHE[key] = (offsets, experts)
    return offsets, experts


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
    offsets, experts = _single_group_launch_tensors(a.device, m)
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


def _quantize_activation_for_precision(
    a: torch.Tensor,
    *,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if precision == "fp8":
        return per_token_cast_to_fp8(a, use_ue8m0=True, gran_k=128)
    if precision == "fp4":
        return per_token_cast_to_nvfp4_e4m3(a, gran_k=16)
    raise ValueError(f"unsupported quantized activation precision={precision!r}")


def _quantized_output_dtype(a: torch.Tensor, *, precision: str) -> torch.dtype:
    if precision == "fp8" and _arch_major(a.device) == 9:
        return torch.float32
    return torch.bfloat16


def _asym_quantized_nt(
    a: torch.Tensor,
    qweight: QuantizedHostWeight,
    *,
    precision: str,
    compiled_dims: str = "mnk",
) -> torch.Tensor:
    import asym_gemm

    reason = _direct_quantized_reason(a, qweight, precision=precision, grouped=False)
    if reason is not None:
        raise RuntimeError(f"direct {precision.upper()} AsymGEMM is unavailable: {reason}")

    m = int(a.shape[0])
    n = int(qweight.out_features)
    a_quantized = _quantize_activation_for_precision(a, precision=precision)
    b_group = (qweight.values.unsqueeze(0), qweight.scales.unsqueeze(0))
    d = torch.empty((m, n), device=a.device, dtype=_quantized_output_dtype(a, precision=precision))
    offsets, experts = _single_group_launch_tensors(a.device, m)

    if precision == "fp8":
        asym_gemm.m_grouped_fp8_asym_gemm_nt_contiguous(
            a_quantized,
            b_group,
            d,
            offsets,
            experts,
            2,
            recipe=_FP8_RECIPE,
            compiled_dims=compiled_dims,
            disable_ue8m0_cast=False,
        )
    elif precision == "fp4":
        asym_gemm.m_grouped_fp4_asym_gemm_nt_contiguous(
            a_quantized,
            b_group,
            d,
            offsets,
            experts,
            2,
            recipe=_FP4_RECIPE,
            compiled_dims=compiled_dims,
            disable_ue8m0_cast=True,
        )
    else:
        raise ValueError(f"unsupported quantized precision={precision!r}")
    return d.to(dtype=torch.bfloat16)


def _asym_grouped_quantized_nt(
    a: torch.Tensor,
    qweight: QuantizedHostWeight,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    precision: str,
    compiled_dims: str = "mnk",
) -> torch.Tensor:
    import asym_gemm

    reason = _direct_quantized_reason(a, qweight, precision=precision, grouped=True)
    if reason is not None:
        raise RuntimeError(f"direct grouped {precision.upper()} AsymGEMM is unavailable: {reason}")

    m = int(a.shape[0])
    n = int(qweight.out_features)
    a_kernel, offsets_kernel, unpad = _pad_grouped_input_for_asym(a, offsets, experts)
    a_quantized = _quantize_activation_for_precision(a_kernel, precision=precision)
    d = torch.empty(
        (int(a_kernel.shape[0]), n),
        device=a.device,
        dtype=_quantized_output_dtype(a, precision=precision),
    )
    offsets_i32, experts_i32, list_size = _group_metadata_tensors(offsets_kernel, experts, device=a.device)

    if precision == "fp8":
        asym_gemm.m_grouped_fp8_asym_gemm_nt_contiguous(
            a_quantized,
            (qweight.values, qweight.scales),
            d,
            offsets_i32,
            experts_i32,
            list_size,
            recipe=_FP8_RECIPE,
            compiled_dims=compiled_dims,
            disable_ue8m0_cast=False,
        )
    elif precision == "fp4":
        asym_gemm.m_grouped_fp4_asym_gemm_nt_contiguous(
            a_quantized,
            (qweight.values, qweight.scales),
            d,
            offsets_i32,
            experts_i32,
            list_size,
            recipe=_FP4_RECIPE,
            compiled_dims=compiled_dims,
            disable_ue8m0_cast=True,
        )
    else:
        raise ValueError(f"unsupported grouped quantized precision={precision!r}")

    d = _unpad_grouped_output(d, unpad, output_m=m)
    return d.to(dtype=torch.bfloat16)


def _staged_nt(a: torch.Tensor, b_cpu: torch.Tensor, *, transpose_b: bool = False) -> torch.Tensor:
    b = b_cpu.to(device=a.device, dtype=a.dtype, non_blocking=b_cpu.is_pinned())
    return a @ b if transpose_b else a @ b.t()


def _grouped_torch_loop(
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


def _grouped_torch_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    transpose_b: bool = False,
    dense_experts: bool = False,
) -> torch.Tensor:
    if a.device.type != "cuda" or b.device.type != "cuda" or a.device != b.device:
        raise RuntimeError("torch grouped_mm path requires input and grouped weight on the same CUDA device")
    if a.dim() != 2 or b.dim() != 3 or offsets.dim() != 1 or experts.dim() != 1:
        raise ValueError("torch grouped_mm path expects a=[M,K], b=[E,N,K], offsets=1D, experts=1D")
    if int(offsets.numel()) != int(experts.numel()) or int(experts.numel()) < 2:
        raise ValueError("torch grouped_mm path expects cumulative offsets and experts with matching sentinel length")

    n = int(b.shape[2] if transpose_b else b.shape[1])
    if int(a.shape[0]) == 0:
        return a.new_empty((0, n))

    starts = offsets[:-1]
    ends = offsets[1:]
    if dense_experts:
        if int(ends.numel()) != int(b.shape[0]):
            raise ValueError(
                f"dense grouped_mm metadata expects one group per expert, got groups={int(ends.numel())} "
                f"and weights={int(b.shape[0])}"
            )
        active_offsets = ends.to(device=a.device, dtype=torch.int32, non_blocking=True).contiguous()
        selected = b
    else:
        active = ends > starts
        active_experts = experts[:-1]
        if active_experts.device != active.device:
            active_experts = active_experts.to(device=active.device, non_blocking=True)
        active_experts = active_experts[active].to(device=b.device, dtype=torch.long, non_blocking=True)
        active_offsets = ends[active].to(device=a.device, dtype=torch.int32, non_blocking=True).contiguous()
        selected = b.index_select(0, active_experts)
    if int(active_offsets.numel()) == 0:
        return a.new_empty((0, n))

    mat1 = a.contiguous()
    # This is the torch baseline for grouped expert base weights: one grouped
    # GEMM per projection, not fusion across gate/up/down or activation.
    mat2 = selected if transpose_b else selected.transpose(-1, -2)
    return _TORCH_GROUPED_MM(mat1, mat2, offs=active_offsets)


def _grouped_torch_chunks(
    a: torch.Tensor,
    b: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    transpose_b: bool = False,
    dense_experts: bool = False,
) -> torch.Tensor:
    if a.device.type == "cuda" or b.device.type == "cuda":
        return _grouped_torch_mm(a, b, offsets, experts, transpose_b=transpose_b, dense_experts=dense_experts)
    return _grouped_torch_loop(a, b, offsets, experts, transpose_b=transpose_b)


def _staged_grouped_nt(
    a: torch.Tensor,
    b_cpu: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    transpose_b: bool = False,
    dense_experts: bool = False,
) -> torch.Tensor:
    b = b_cpu.to(device=a.device, dtype=a.dtype, non_blocking=b_cpu.is_pinned())
    return _grouped_torch_chunks(a, b, offsets, experts, transpose_b=transpose_b, dense_experts=dense_experts)


def _torch_nt(a: torch.Tensor, b_cpu: torch.Tensor, *, transpose_b: bool = False) -> torch.Tensor:
    if a.device.type == "cpu":
        b = b_cpu.to(dtype=a.dtype)
        return a @ b if transpose_b else a @ b.t()
    b = b_cpu.to(device=a.device, dtype=a.dtype, non_blocking=b_cpu.is_pinned())
    return a @ b if transpose_b else a @ b.t()


def _torch_grouped_nt(
    a: torch.Tensor,
    b_cpu: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    transpose_b: bool = False,
    dense_experts: bool = False,
) -> torch.Tensor:
    if a.device.type == "cpu":
        b = b_cpu.to(dtype=a.dtype)
        return _grouped_torch_chunks(a, b, offsets, experts, transpose_b=transpose_b)
    b = b_cpu.to(device=a.device, dtype=a.dtype, non_blocking=b_cpu.is_pinned())
    return _grouped_torch_chunks(a, b, offsets, experts, transpose_b=transpose_b, dense_experts=dense_experts)


def _dispatch_nt(
    a: torch.Tensor,
    b_cpu: torch.Tensor,
    *,
    backend: str,
    stats: Optional[AsymExecutionStats],
    phase: str,
    compiled_dims: str,
    transpose_b: bool = False,
    precision: str = "bf16",
    quantized_weight: Optional[QuantizedHostWeight] = None,
    profile_label: str = "",
) -> torch.Tensor:
    _check_backend(backend)
    precision = _normalize_precision(precision)

    if backend != "torch":
        reason = (
            _direct_bf16_reason(a, b_cpu, transpose_b=transpose_b)
            if precision == "bf16"
            else _direct_quantized_reason(a, quantized_weight, precision=precision, grouped=False)
        )
        if reason is None:
            try:
                if precision == "bf16":
                    out = _asym_bf16_nt(a, b_cpu, compiled_dims=compiled_dims, transpose_b=transpose_b)
                else:
                    assert quantized_weight is not None
                    out = _asym_quantized_nt(
                        a,
                        quantized_weight,
                        precision=precision,
                        compiled_dims=compiled_dims,
                    )
                if stats is not None:
                    if phase == "forward":
                        stats.asym_forward_calls += 1
                    else:
                        stats.asym_dx_calls += 1
                return out
            except RuntimeError as exc:
                reason = f"direct_runtime_error:{type(exc).__name__}"
                if backend == "asym":
                    raise
        if stats is not None:
            stats.record_fallback(f"{phase}:{reason}")
        if backend == "asym":
            raise RuntimeError(f"direct {precision.upper()} AsymGEMM is unavailable: {reason}")

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
    precision: str = "bf16",
    quantized_weight: Optional[QuantizedHostWeight] = None,
    dense_experts: bool = False,
) -> torch.Tensor:
    _check_backend(backend)
    precision = _normalize_precision(precision)

    if backend != "torch":
        reason = (
            _direct_grouped_bf16_reason(a, b_cpu, transpose_b=transpose_b)
            if precision == "bf16"
            else _direct_quantized_reason(a, quantized_weight, precision=precision, grouped=True)
        )
        if reason is None:
            try:
                if precision == "bf16":
                    out = _asym_grouped_bf16_nt(
                        a,
                        b_cpu,
                        offsets,
                        experts,
                        compiled_dims=compiled_dims,
                        transpose_b=transpose_b,
                    )
                else:
                    assert quantized_weight is not None
                    out = _asym_grouped_quantized_nt(
                        a,
                        quantized_weight,
                        offsets,
                        experts,
                        precision=precision,
                        compiled_dims=compiled_dims,
                    )
                if stats is not None:
                    if phase == "forward":
                        stats.asym_forward_calls += 1
                    else:
                        stats.asym_dx_calls += 1
                return out
            except RuntimeError as exc:
                reason = f"direct_runtime_error:{type(exc).__name__}"
                if backend == "asym":
                    raise
        if stats is not None:
            stats.record_fallback(f"{phase}:{reason}")
        if backend == "asym":
            raise RuntimeError(f"direct grouped {precision.upper()} AsymGEMM is unavailable: {reason}")

    if stats is not None:
        if phase == "forward":
            stats.torch_forward_calls += 1
        else:
            stats.torch_dx_calls += 1
    return _torch_grouped_nt(a, b_cpu, offsets, experts, transpose_b=transpose_b, dense_experts=dense_experts)


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
        precision: str,
        quantized_weight: Optional[QuantizedHostWeight],
    ) -> torch.Tensor:
        precision = _normalize_precision(precision)
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
                precision=precision,
                quantized_weight=quantized_weight,
                profile_label=forward_range,
            )
        if bias is not None:
            y = y + bias.to(device=y.device, dtype=y.dtype)

        ctx.host_weight = host_weight
        ctx.backend = backend
        ctx.stats = stats
        ctx.compiled_dims = compiled_dims
        ctx.precision = precision
        ctx.profile_backward_range = backward_range
        ctx.profile_enabled = is_profile_enabled()
        ctx.has_bias = bias is not None
        ctx.bias_device = bias.device if bias is not None else None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.input_shape = input_shape
        return y.reshape(*input_shape[:-1], host_weight.out_features)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], None, Optional[torch.Tensor], None, None, None, None, None, None]:
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_output_2d = grad_output.reshape(-1, ctx.host_weight.out_features).contiguous()
            quantized_weight_t = (
                _get_quantized_host_weight(ctx.host_weight, ctx.precision, transpose=True)
                if ctx.backend != "torch" and ctx.precision != "bf16"
                else None
            )
            with prof_range(ctx.profile_backward_range, enabled=ctx.profile_enabled):
                grad_x = _dispatch_nt(
                    grad_output_2d,
                    ctx.host_weight.weight,
                    backend=ctx.backend,
                    stats=ctx.stats,
                    phase="dx",
                    compiled_dims=ctx.compiled_dims,
                    transpose_b=True,
                    precision=ctx.precision,
                    quantized_weight=quantized_weight_t,
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
        return grad_x, None, grad_bias, None, None, None, None, None, None


def asym_frozen_linear(
    x: torch.Tensor,
    host_weight: HostWeight,
    *,
    bias: Optional[torch.Tensor] = None,
    backend: str = "asym",
    stats: Optional[AsymExecutionStats] = None,
    compiled_dims: str = "mnk",
    profile_name: str = "",
    precision: str = "bf16",
) -> torch.Tensor:
    precision = _normalize_precision(precision)
    quantized_weight = (
        _get_quantized_host_weight(host_weight, precision, transpose=False)
        if backend != "torch" and precision != "bf16"
        else None
    )
    return AsymFrozenLinearFunction.apply(
        x,
        host_weight,
        bias,
        backend,
        stats,
        compiled_dims,
        profile_name,
        precision,
        quantized_weight,
    )


def frozen_linear(
    x: torch.Tensor,
    host_weight: HostWeight,
    bias: Optional[torch.Tensor] = None,
    *,
    backend: str = "asym",
    stats: Optional[AsymExecutionStats] = None,
    compiled_dims: str = "mnk",
    profile_name: str = "",
    precision: str = "bf16",
) -> torch.Tensor:
    return asym_frozen_linear(
        x,
        host_weight,
        bias=bias,
        backend=backend,
        stats=stats,
        compiled_dims=compiled_dims,
        profile_name=profile_name,
        precision=precision,
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
        precision: str,
        quantized_weight: Optional[QuantizedHostWeight],
        dense_experts: bool,
    ) -> torch.Tensor:
        precision = _normalize_precision(precision)
        if host_weight.weight.dim() != 3:
            raise ValueError(f"grouped host weight must be 3D, got shape {tuple(host_weight.weight.shape)}")
        if x.shape[-1] != int(host_weight.weight.shape[2]):
            raise ValueError(f"expected input last dim {int(host_weight.weight.shape[2])}, got {x.shape[-1]}")
        input_shape = tuple(x.shape)
        x_2d = x.reshape(-1, int(host_weight.weight.shape[2])).contiguous()
        out_features = int(host_weight.weight.shape[1])
        forward_range = (
            "forward.grouped_base_frozen_asymgemm"
            if not profile_name
            else f"forward.{profile_name}.grouped_base_frozen_asymgemm"
        )
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
                precision=precision,
                quantized_weight=quantized_weight,
                dense_experts=dense_experts,
            )

        ctx.host_weight = host_weight
        ctx.offsets = offsets
        ctx.experts = experts
        ctx.backend = backend
        ctx.stats = stats
        ctx.compiled_dims = compiled_dims
        ctx.precision = precision
        ctx.dense_experts = bool(dense_experts)
        ctx.profile_backward_range = backward_range
        ctx.profile_enabled = is_profile_enabled()
        ctx.input_shape = input_shape
        return y.reshape(*input_shape[:-1], out_features)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], None, None, None, None, None, None, None, None, None, None]:
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_output_2d = grad_output.reshape(-1, int(ctx.host_weight.weight.shape[1])).contiguous()
            if ctx.precision == "bf16":
                b_cpu = ctx.host_weight.weight
                transpose_b = True
                quantized_weight_t = None
            else:
                b_cpu = ctx.host_weight.weight
                transpose_b = True
                quantized_weight_t = (
                    _get_quantized_host_weight(ctx.host_weight, ctx.precision, transpose=True)
                    if ctx.backend != "torch"
                    else None
                )
            with prof_range(ctx.profile_backward_range, enabled=ctx.profile_enabled):
                grad_x = _dispatch_grouped_nt(
                    grad_output_2d,
                    b_cpu,
                    ctx.offsets,
                    ctx.experts,
                    backend=ctx.backend,
                    stats=ctx.stats,
                    phase="dx",
                    compiled_dims=ctx.compiled_dims,
                    transpose_b=transpose_b,
                    precision=ctx.precision,
                    quantized_weight=quantized_weight_t,
                    dense_experts=ctx.dense_experts,
                )
            grad_x = grad_x.reshape(ctx.input_shape)
        return grad_x, None, None, None, None, None, None, None, None, None, None


def asym_grouped_frozen_linear(
    x: torch.Tensor,
    host_weight: HostWeight,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    backend: str = "asym",
    stats: Optional[AsymExecutionStats] = None,
    compiled_dims: str = "mnk",
    profile_name: str = "",
    precision: str = "bf16",
    dense_experts: bool = False,
) -> torch.Tensor:
    precision = _normalize_precision(precision)
    quantized_weight = (
        _get_quantized_host_weight(host_weight, precision, transpose=False)
        if backend != "torch" and precision != "bf16"
        else None
    )
    return AsymGroupedFrozenLinearFunction.apply(
        x,
        host_weight,
        offsets,
        experts,
        backend,
        stats,
        compiled_dims,
        profile_name,
        precision,
        quantized_weight,
        dense_experts,
    )


class AsymGroupedFrozenLinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        *,
        backend: str = "asym",
        pin_memory: bool = True,
        stats: Optional[AsymExecutionStats] = None,
        compiled_dims: str = "mnk",
        precision: str = "bf16",
    ) -> None:
        super().__init__()
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"weight must be a torch.Tensor, got {type(weight)!r}")
        if weight.dim() != 3:
            raise ValueError(f"AsymGroupedFrozenLinear expects [groups, out, in], got {tuple(weight.shape)}")
        _check_backend(backend)
        precision = _normalize_precision(precision)
        self.host_weight = HostWeight(weight, pin_memory=pin_memory, clone=True, require_2d=False)
        self.backend = backend
        self.stats = stats if stats is not None else AsymExecutionStats()
        self.compiled_dims = compiled_dims
        self.precision = precision
        self.profile_name = ""
        self.num_groups = int(self.host_weight.weight.shape[0])
        self.out_features = int(self.host_weight.weight.shape[1])
        self.in_features = int(self.host_weight.weight.shape[2])

    @property
    def pinned_cpu_bytes(self) -> int:
        return self.host_weight.pinned_cpu_bytes + _quantized_cache_pinned_bytes(self.host_weight, self.precision)

    @property
    def weight_hbm_saved_bytes(self) -> int:
        return self.host_weight.weight_nbytes

    @property
    def weight(self) -> torch.Tensor:
        return self.host_weight.weight

    def forward(self, x: torch.Tensor, offsets: torch.Tensor, experts: torch.Tensor, *, dense_experts: bool = False) -> torch.Tensor:
        return asym_grouped_frozen_linear(
            x,
            self.host_weight,
            offsets,
            experts,
            backend=self.backend,
            stats=self.stats,
            compiled_dims=self.compiled_dims,
            profile_name=self.profile_name,
            precision=self.precision,
            dense_experts=dense_experts,
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
            f"precision={self.precision}, pinned={self.host_weight.metadata.pinned}"
        )


class TorchGroupedFrozenLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        profile_name: str,
        dense_experts: bool,
    ) -> torch.Tensor:
        if weight.dim() != 3:
            raise ValueError(f"grouped torch weight must be 3D, got shape {tuple(weight.shape)}")
        if x.shape[-1] != int(weight.shape[2]):
            raise ValueError(f"expected input last dim {int(weight.shape[2])}, got {x.shape[-1]}")
        input_shape = tuple(x.shape)
        x_2d = x.reshape(-1, int(weight.shape[2])).contiguous()
        out_features = int(weight.shape[1])
        forward_range = "forward.grouped_base_torch" if not profile_name else f"forward.{profile_name}.grouped_base_torch"
        backward_range = "backward.grouped_base_dx_torch" if not profile_name else f"backward.{profile_name}.grouped_base_dx_torch"
        with prof_range(forward_range):
            y = _grouped_torch_chunks(x_2d, weight, offsets, experts, dense_experts=dense_experts)

        ctx.save_for_backward(weight, offsets, experts)
        ctx.dense_experts = bool(dense_experts)
        ctx.profile_backward_range = backward_range
        ctx.profile_enabled = is_profile_enabled()
        ctx.input_shape = input_shape
        return y.reshape(*input_shape[:-1], out_features)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], None, None, None, None, None]:
        grad_x = None
        if ctx.needs_input_grad[0]:
            weight, offsets, experts = ctx.saved_tensors
            grad_output_2d = grad_output.reshape(-1, int(weight.shape[1])).contiguous()
            with prof_range(ctx.profile_backward_range, enabled=ctx.profile_enabled):
                grad_x = _grouped_torch_chunks(
                    grad_output_2d,
                    weight,
                    offsets,
                    experts,
                    transpose_b=True,
                    dense_experts=ctx.dense_experts,
                )
            grad_x = grad_x.reshape(ctx.input_shape)
        return grad_x, None, None, None, None, None


class TorchGroupedFrozenLinear(nn.Module):
    """GPU-resident grouped frozen linear for the all-HBM torch baseline."""

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"weight must be a torch.Tensor, got {type(weight)!r}")
        if weight.dim() != 3:
            raise ValueError(f"TorchGroupedFrozenLinear expects [groups, out, in], got {tuple(weight.shape)}")
        self.register_buffer("weight", weight.detach().to(device=device, dtype=dtype).contiguous())
        self.profile_name = ""
        self.num_groups = int(self.weight.shape[0])
        self.out_features = int(self.weight.shape[1])
        self.in_features = int(self.weight.shape[2])

    @property
    def pinned_cpu_bytes(self) -> int:
        return 0

    @property
    def weight_hbm_saved_bytes(self) -> int:
        return tensor_nbytes(self.weight)

    @property
    def gpu_resident_weight_bytes(self) -> int:
        return tensor_nbytes(self.weight)

    def forward(self, x: torch.Tensor, offsets: torch.Tensor, experts: torch.Tensor, *, dense_experts: bool = False) -> torch.Tensor:
        return TorchGroupedFrozenLinearFunction.apply(x, self.weight, offsets, experts, self.profile_name, dense_experts)

    def extra_repr(self) -> str:
        return (
            f"num_groups={self.num_groups}, in_features={self.in_features}, "
            f"out_features={self.out_features}, device={self.weight.device}, dtype={self.weight.dtype}"
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
        backend: str = "asym",
        pin_memory: bool = True,
        stats: Optional[AsymExecutionStats] = None,
        compiled_dims: str = "mnk",
        precision: str = "bf16",
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
        precision = _normalize_precision(precision)
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
        self.precision = precision
        self.profile_name = ""
        self.in_features = self.host_weight.in_features
        self.out_features = self.host_weight.out_features

    @classmethod
    def from_gpu_linear(
        cls,
        linear: nn.Linear,
        *,
        backend: str = "asym",
        pin_memory: bool = True,
        stats: Optional[AsymExecutionStats] = None,
        compiled_dims: str = "mnk",
        precision: str = "bf16",
    ) -> "AsymFrozenLinear":
        return cls(
            linear.weight.detach(),
            bias=None if linear.bias is None else linear.bias.detach(),
            backend=backend,
            pin_memory=pin_memory,
            stats=stats,
            compiled_dims=compiled_dims,
            precision=precision,
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        backend: str = "asym",
        pin_memory: bool = True,
        stats: Optional[AsymExecutionStats] = None,
        compiled_dims: str = "mnk",
        precision: str = "bf16",
    ) -> "AsymFrozenLinear":
        return cls.from_gpu_linear(
            linear,
            backend=backend,
            pin_memory=pin_memory,
            stats=stats,
            compiled_dims=compiled_dims,
            precision=precision,
        )

    @property
    def pinned_cpu_bytes(self) -> int:
        total = self.host_weight.pinned_cpu_bytes + _quantized_cache_pinned_bytes(self.host_weight, self.precision)
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
            precision=self.precision,
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
            f"backend={self.backend}, precision={self.precision}, pinned={self.host_weight.metadata.pinned}"
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
