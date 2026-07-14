from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from asym_gemm.utils.math import per_block_cast_to_fp8, per_token_cast_to_fp8, per_token_cast_to_nvfp4_e4m3

from .host_weight import HostWeight, tensor_nbytes
from .profile_ranges import is_profile_enabled, prof_range


# process-constant (fix_gb200_ep.md S2a): |1 rows never set this — the hot grouped-GEMM
# path must pay ZERO per-call env reads for the queued branch check.
import os as _os

_EP_QUEUED_ENABLED = _os.environ.get("ASYM_EP_QUEUED") == "1"
_EP_SEP_ENABLED = _os.environ.get("ASYM_EP_SEP") == "1"  # S6 true-sEP union sharing
# S5b diagnostics: host-block probe on the per-launch pad scalar read (ep_vanilla.py)
_PAD_TIMING = bool(int(_os.environ.get("ASYM_EP_VANILLA_TIMING", "0") or 0))
# fix_ep (2026-07-11): sync-free padding — allocate at the host-computable upper
# bound (m + groups*(BLOCK_M-1)) instead of reading the true padded total back
# with .item(). TRIED AND REJECTED for ep2 (220.2 s vs 169.6 at 32k): losing the
# already-padded early-return forces full pad work + fresh index tensors on
# every backward call — allocator churn worse than the sync. Kept as a
# receipted dead end; default OFF.
_PAD_UPPER_BOUND = _os.environ.get("ASYM_PAD_UPPER_BOUND") == "1"
# fix_ep miss probe: print one stack per unique pad-memo MISS site (diagnosis).
_PAD_DEBUG = _os.environ.get("ASYM_PAD_DEBUG") == "1"
_PAD_MISS_STACKS: set = set()

# fix_ep (2026-07-11): LAYER-SCOPED pad-memo context. The fg entry points
# rebuild offsets tensors per layer/phase (probe receipt: has_memo=False with
# fresh tensor ids at qwen3_moe:2579/2606 + frozen_linear:1884), so
# tensor-attached memos cannot hit there. Within ONE expert-layer invocation
# every grouped call shares the same (m, offsets VALUES) — the vanilla branch
# opens this context and the padder consults it by (block_m, m, device),
# tensor identity be damned. Cleared on layer exit; GC-recompute re-opens it.
import contextlib as _contextlib
import threading as _threading

_PAD_CTX = _threading.local()


@_contextlib.contextmanager
def pad_memo_context():
    prev = getattr(_PAD_CTX, "memo", None)
    _PAD_CTX.memo = {}
    try:
        yield
    finally:
        _PAD_CTX.memo = prev

VALID_BACKENDS = ("asym", "torch")
VALID_ASYM_PRECISIONS = ("bf16", "fp8", "fp4")
VALID_BF16_OUTPUT_DTYPES = ("bf16", "bfloat16", "fp32", "float32")
VALID_GROUPED_WEIGHT_LAYOUTS = ("out_in", "in_out")
_FP8_RECIPE = (1, 128, 128)
_FP4_RECIPE = (1, 1, 16)
_TORCH_GROUPED_MM = getattr(torch.nn.functional, "grouped_mm", None)
_TORCH_GROUPED_MM_NAME = "torch.nn.functional.grouped_mm"
if _TORCH_GROUPED_MM is None:
    _TORCH_GROUPED_MM = getattr(torch, "_grouped_mm", None)
    _TORCH_GROUPED_MM_NAME = "torch._grouped_mm"

_SINGLE_GROUP_LAUNCH_TENSOR_CACHE: dict[
    tuple[str, int], tuple[torch.Tensor, torch.Tensor]
] = {}


def _require_torch_grouped_mm():
    if _TORCH_GROUPED_MM is None:
        raise RuntimeError("PyTorch grouped torch baseline requires torch.nn.functional.grouped_mm or torch._grouped_mm")
    return _TORCH_GROUPED_MM


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
    cpu_left_lora_a_calls: int = 0
    expact_lora_a_forward_grouped_calls: int = 0
    expact_lora_a_forward_cpu_left_grouped_calls: int = 0
    expact_lora_a_forward_hbm_grouped_calls: int = 0
    expact_lora_a_grad_grouped_calls: int = 0
    expact_lora_b_backward_grouped_calls: int = 0
    expact_stage_low_rank_calls: int = 0
    attn_act_base_dx_calls: int = 0
    attn_act_lora_a_forward_calls: int = 0
    attn_act_lora_a_grad_calls: int = 0
    attn_act_stage_low_rank_calls: int = 0
    attn_act_hbm_gemm_calls_by_tag: Dict[str, int] = field(default_factory=dict)
    dense_mlp_finegrained_forward_calls: int = 0
    dense_mlp_finegrained_backward_calls: int = 0
    dense_mlp_finegrained_gate_base_calls: int = 0
    dense_mlp_finegrained_up_base_calls: int = 0
    dense_mlp_finegrained_down_base_calls: int = 0
    dense_mlp_finegrained_stage_concat_columns_calls: int = 0
    dense_mlp_finegrained_gpu_silu_bwd_calls: int = 0
    dense_mlp_finegrained_cpu_silu_bwd_calls: int = 0
    qwen3_moe_finegrained_forward_calls: int = 0
    qwen3_moe_finegrained_nograd_forward_calls: int = 0
    qwen3_moe_finegrained_backward_calls: int = 0
    qwen3_moe_finegrained_gate_base_calls: int = 0
    qwen3_moe_finegrained_up_base_calls: int = 0
    qwen3_moe_finegrained_down_base_calls: int = 0
    qwen3_moe_finegrained_stage_concat_columns_calls: int = 0
    qwen3_moe_finegrained_gpu_silu_bwd_calls: int = 0
    qwen3_moe_finegrained_cpu_silu_bwd_calls: int = 0
    qwen3_moe_finegrained_lora_a_forward_calls: int = 0
    qwen3_moe_finegrained_lora_a_forward_gpu_calls: int = 0
    qwen3_moe_finegrained_lora_a_grad_calls: int = 0
    qwen3_moe_finegrained_da_gpu_calls: int = 0
    qwen3_moe_finegrained_dgrads_hbm_kept: int = 0
    qwen3_moe_finegrained_lora_b_backward_calls: int = 0
    qwen3_moe_finegrained_fused_gate_up_hbm_bytes: int = 0
    qwen3_moe_finegrained_saved_cpu_bytes: int = 0
    qwen3_moe_finegrained_stage_hbm_peak_bytes: int = 0
    qwen3_moe_finegrained_down_scatter_block_experts: int = 0
    qwen3_moe_finegrained_down_scatter_blocks: int = 0
    qwen3_moe_finegrained_down_scatter_max_block_rows: int = 0
    qwen3_moe_finegrained_hidden_route_global_tensors_avoided: int = 0
    qwen3_moe_finegrained_stage_rows_calls: int = 0
    qwen3_moe_routed_base_forward_scatter_calls: int = 0
    qwen3_moe_routed_base_gather_left_calls: int = 0
    qwen3_moe_routed_base_dx_scatter_calls: int = 0
    qwen3_moe_routed_lora_b_forward_scatter_calls: int = 0
    qwen3_moe_routed_lora_b_backward_from_tokens_calls: int = 0
    qwen3_moe_routed_lora_dx_scatter_calls: int = 0
    qwen3_moe_routed_route_space_h_tensors_avoided: int = 0
    reference_fallback_count: int = 0
    fallback_reasons: Dict[str, int] = field(default_factory=dict)

    def record_fallback(self, reason: str) -> None:
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1

    def record_reference_fallback(self, reason: str) -> None:
        self.reference_fallback_count += 1
        self.record_fallback(f"reference:{reason}")

    @property
    def asym_calls(self) -> int:
        return self.asym_forward_calls + self.asym_dx_calls

    @property
    def staged_calls(self) -> int:
        return self.staged_forward_calls + self.staged_dx_calls

    @property
    def torch_calls(self) -> int:
        return self.torch_forward_calls + self.torch_dx_calls

    @property
    def attn_act_hbm_forward_calls(self) -> int:
        return sum(
            int(count)
            for tag, count in self.attn_act_hbm_gemm_calls_by_tag.items()
            if str(tag).rsplit(".", 1)[-1] in {"lora_a_forward", "lora_b_forward"}
        )

    @property
    def attn_act_hbm_backward_calls(self) -> int:
        return sum(
            int(count)
            for tag, count in self.attn_act_hbm_gemm_calls_by_tag.items()
            if str(tag).rsplit(".", 1)[-1] in {"dS", "lora_input_grad", "dB"}
        )

    @property
    def attn_act_hbm_calls(self) -> int:
        return self.attn_act_hbm_forward_calls + self.attn_act_hbm_backward_calls

    @property
    def forward_calls_total(self) -> int:
        return (
            self.asym_forward_calls
            + self.staged_forward_calls
            + self.torch_forward_calls
            + self.kt_forward_calls
            + self.cpu_left_lora_a_calls
            + self.expact_lora_a_forward_hbm_grouped_calls
            + self.attn_act_hbm_forward_calls
            + self.dense_mlp_finegrained_forward_calls
            + self.qwen3_moe_finegrained_forward_calls
            + self.qwen3_moe_finegrained_nograd_forward_calls
        )

    @property
    def backward_calls_total(self) -> int:
        return (
            self.asym_dx_calls
            + self.staged_dx_calls
            + self.torch_dx_calls
            + self.kt_backward_calls
            + self.expact_lora_a_grad_grouped_calls
            + self.expact_lora_b_backward_grouped_calls
            + self.expact_stage_low_rank_calls
            + self.attn_act_hbm_backward_calls
            + self.attn_act_stage_low_rank_calls
            + self.dense_mlp_finegrained_backward_calls
            + self.qwen3_moe_finegrained_backward_calls
        )

    @property
    def calls_total(self) -> int:
        return self.forward_calls_total + self.backward_calls_total

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["asym_calls"] = self.asym_calls
        data["staged_calls"] = self.staged_calls
        data["torch_calls"] = self.torch_calls
        data["attn_act_hbm_forward_calls"] = self.attn_act_hbm_forward_calls
        data["attn_act_hbm_backward_calls"] = self.attn_act_hbm_backward_calls
        data["attn_act_hbm_calls"] = self.attn_act_hbm_calls
        data["forward_calls_total"] = self.forward_calls_total
        data["backward_calls_total"] = self.backward_calls_total
        data["calls_total"] = self.calls_total
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


@dataclass(frozen=True)
class _GroupedPadding:
    padded_rows: torch.Tensor
    original_rows: torch.Tensor


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


def _normalize_bf16_output_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        if dtype in {torch.bfloat16, torch.float32}:
            return dtype
        raise ValueError("BF16 AsymGEMM output dtype must be torch.bfloat16 or torch.float32")
    normalized = str(dtype).lower().removeprefix("torch.")
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported BF16 AsymGEMM output dtype {dtype!r}; expected one of {VALID_BF16_OUTPUT_DTYPES}")


def _normalize_grouped_weight_layout(layout: str) -> str:
    normalized = str(layout).lower().replace("-", "_")
    if normalized not in VALID_GROUPED_WEIGHT_LAYOUTS:
        raise ValueError(f"unsupported grouped weight layout={layout!r}; expected one of {VALID_GROUPED_WEIGHT_LAYOUTS}")
    return normalized


def _grouped_weight_features(weight: torch.Tensor, layout: str) -> tuple[int, int, bool, bool]:
    """Return (in_features, out_features, forward_transpose_b, backward_transpose_b)."""
    layout = _normalize_grouped_weight_layout(layout)
    if weight.dim() != 3:
        raise ValueError(f"grouped weight must be 3D, got shape {tuple(weight.shape)}")
    if layout == "out_in":
        return int(weight.shape[2]), int(weight.shape[1]), False, True
    return int(weight.shape[1]), int(weight.shape[2]), True, False


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
    if transpose_b and k % 64 != 0:
        return "transpose_b_requires_64_aligned_k"

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
) -> tuple[torch.Tensor, torch.Tensor, _GroupedPadding | None]:
    if offsets.numel() != experts.numel():
        return a, offsets, None

    num_groups = int(experts.numel() - 1)
    # The SAME offsets tensor serves every grouped call of a layer (fwd base/LoRA + all
    # backward variants). The pad METADATA depends only on (offsets, block_m) — memoize it
    # on the tensor so the .item() host sync (which drains deep async queues under sTP:
    # measured as the top backward hotspot) happens once per layer instead of ~6-12x.
    memo = getattr(offsets, "_asym_pad_memo", None)
    memo_key = (int(block_m), int(a.shape[0]), str(a.device))
    cached = None if memo is None else memo.get(memo_key)
    ctx_memo = getattr(_PAD_CTX, "memo", None)
    if cached is None and ctx_memo is not None:
        # layer-scoped fallback (fix_ep): same layer => same offsets VALUES,
        # even when the fg path rebuilt the tensor object.
        cached = ctx_memo.get(memo_key)
    if cached is not None:
        total_padded, padded_offsets_long, safe_source_rows, valid_rows = cached
        if total_padded == int(a.shape[0]):
            return a, offsets, None
    else:
        if _PAD_DEBUG:
            import traceback

            stack = "".join(traceback.format_stack(limit=8)[:-1])
            key = hash(stack)
            if key not in _PAD_MISS_STACKS:
                _PAD_MISS_STACKS.add(key)
                print(f"[pad-miss] m={int(a.shape[0])} offsets_id={id(offsets)} "
                      f"has_memo={memo is not None}\n{stack}", flush=True)
        offsets_long = offsets.to(device=a.device, dtype=torch.long, non_blocking=True)
        starts = offsets_long[:-1]
        counts = (offsets_long[1:] - starts).clamp_min(0)
        padded_counts = torch.div(counts + int(block_m) - 1, int(block_m), rounding_mode="floor") * int(block_m)
        padded_offsets_long = torch.cat(
            (
                torch.zeros(1, device=a.device, dtype=torch.long),
                torch.cumsum(padded_counts, dim=0),
            ),
            dim=0,
        )

        # PyTorch allocations still require a Python integer shape. Keep this to one
        # scalar read instead of the previous full offsets D2H copy and Python loops.
        if _PAD_UPPER_BOUND:
            # fix_ep (2026-07-11): host-computable ALLOCATION BOUND instead of the
            # .item() sync (receipt: pad_item_s ~15 s/48-call window in ep2
            # backward — memo misses on derived offsets tensors — feeding the
            # inter-rank oscillation). The kernel reads TRUE segment bounds from
            # the device offsets tensor; rows in [true_total, bound) are dead
            # slack past the last segment (masked to zero, untouched by the
            # kernel, dropped by unpad). Cost: the no-pad early-return cannot
            # trigger (true total unknown host-side), so already-aligned inputs
            # pay one extra masked copy per call.
            total_padded = int(a.shape[0]) + num_groups * (int(block_m) - 1)
        elif _PAD_TIMING:
            import time as _time

            _t0 = _time.perf_counter()
            total_padded = int(padded_offsets_long[-1].item())
            from .ep_vanilla import _timing_add as _ep_timing_add

            _ep_timing_add("pad_item_s", _time.perf_counter() - _t0)
        else:
            total_padded = int(padded_offsets_long[-1].item())
        if not _PAD_UPPER_BOUND and total_padded == int(a.shape[0]):
            if memo is None:
                memo = {}
                try:
                    offsets._asym_pad_memo = memo  # type: ignore[attr-defined]
                except Exception:
                    memo = None
            if memo is not None:
                memo[memo_key] = (total_padded, offsets, None, None)
            if ctx_memo is not None:
                ctx_memo[memo_key] = (total_padded, offsets, None, None)
            return a, offsets, None

        padded_rows = torch.arange(total_padded, device=a.device, dtype=torch.long)
        group_idx = torch.bucketize(padded_rows, padded_offsets_long[1:], right=True)
        group_idx = group_idx.clamp_max(max(num_groups - 1, 0))
        group_starts = starts.index_select(0, group_idx)
        group_counts = counts.index_select(0, group_idx)
        local_rows = padded_rows - padded_offsets_long.index_select(0, group_idx)
        valid_rows = local_rows < group_counts
        source_rows = group_starts + local_rows
        safe_source_rows = torch.where(valid_rows, source_rows, torch.zeros_like(source_rows))
        if memo is None:
            memo = {}
            try:
                offsets._asym_pad_memo = memo  # type: ignore[attr-defined]
            except Exception:
                memo = None
        if memo is not None:
            memo[memo_key] = (total_padded, padded_offsets_long, safe_source_rows, valid_rows)
        if ctx_memo is not None:
            ctx_memo[memo_key] = (total_padded, padded_offsets_long, safe_source_rows, valid_rows)

    padded = a.index_select(0, safe_source_rows)
    if padded.numel() > 0:
        padded = padded * valid_rows.reshape(-1, *([1] * (padded.dim() - 1))).to(dtype=padded.dtype)

    valid_padded_rows = torch.nonzero(valid_rows, as_tuple=False).flatten()
    original_rows = safe_source_rows.index_select(0, valid_padded_rows)
    offsets_out = padded_offsets_long.to(device=offsets.device, dtype=offsets.dtype)
    # fix_ep (2026-07-11): PRE-SEED the RETURNED padded-offsets tensor's memo
    # with its aligned-case entry — backward re-pads saved (padded a, padded
    # offsets) pairs OUTSIDE the layer context, and each such tensor paid one
    # first-touch .item() drain mid-backward (oscillation food). The padded
    # tensor is BLOCK_M-aligned by construction: the check is free.
    try:
        offsets_out._asym_pad_memo = {
            (int(block_m), int(total_padded), str(a.device)):
                (total_padded, offsets_out, None, None)
        }
    except Exception:
        pass
    return (
        padded,
        offsets_out,
        _GroupedPadding(padded_rows=valid_padded_rows, original_rows=original_rows),
    )


def prewarm_pad_memo(offsets: torch.Tensor, experts: torch.Tensor, m_rows: int,
                     *, device, dtype, block_m: int = 128) -> None:
    """fix_ep D2: populate the pad memo BEFORE the vanilla-EP hidden allgather
    enqueues. Runs the production padder on a ZERO-WIDTH dummy (m_rows, 0) — the
    memo key is (block_m, m_rows, device), so every later real grouped call of
    the layer (fwd base/LoRA + all bwd variants incl. GC recompute) hits the
    cache and never host-syncs. The one .item() this costs happens while only
    µs-class tiny-gather work is in the queue."""
    if m_rows <= 0:
        return
    dummy = torch.empty((int(m_rows), 0), device=device, dtype=dtype)
    _pad_grouped_input_for_asym(dummy, offsets, experts, block_m=block_m)


def _unpad_grouped_output(
    padded: torch.Tensor,
    unpad: _GroupedPadding | None,
    *,
    output_m: int,
) -> torch.Tensor:
    if unpad is None:
        return padded
    out = padded.new_empty((output_m, padded.shape[1]))
    out.index_copy_(0, unpad.original_rows, padded.index_select(0, unpad.padded_rows))
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


def _resolve_launch_tensor_device(device: torch.device | str | int) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device("cuda", device)
    return torch.device(device)


def initialize_asym_single_group_launch_tensors(
    device: torch.device | str | int,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-create dense AsymGEMM launch metadata before CUDA graph capture."""
    rows_int = int(rows)
    if rows_int <= 0:
        raise ValueError(f"rows must be positive, got {rows}")
    return _single_group_launch_tensors(_resolve_launch_tensor_device(device), rows_int)


def initialize_asym_cuda_graph_state(
    device: torch.device | str | int,
    rows: int | Sequence[int],
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Pre-create dense/single-group AsymGEMM tensors needed during CUDA graph replay.

    This covers the direct dense AsymFrozenLinear launch path. Grouped MoE routes still
    need graph-stable routing metadata supplied by the caller.
    """
    if isinstance(rows, int):
        row_values = (rows,)
    else:
        row_values = tuple(rows)
    if not row_values:
        raise ValueError("rows must contain at least one row count")
    return tuple(
        initialize_asym_single_group_launch_tensors(device, int(row_count))
        for row_count in row_values
    )


def _asym_bf16_nt(
    a: torch.Tensor,
    b_cpu: torch.Tensor,
    *,
    compiled_dims: str = "mnk",
    transpose_b: bool = False,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    import asym_gemm

    reason = _direct_bf16_reason(a, b_cpu, transpose_b=transpose_b)
    if reason is not None:
        raise RuntimeError(f"direct BF16 AsymGEMM is unavailable: {reason}")

    m = int(a.shape[0])
    n = int(b_cpu.shape[1] if transpose_b else b_cpu.shape[0])
    d = torch.empty((m, n), device=a.device, dtype=output_dtype)
    offsets, experts = _single_group_launch_tensors(a.device, m)
    b_group = b_cpu.unsqueeze(0)
    asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
        a, b_group, d, offsets, experts, 2, compiled_dims, transpose_b
    )
    return d


def _asym_grouped_bf16_nt(
    a: torch.Tensor,
    b_cpu: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    compiled_dims: str = "mnk",
    transpose_b: bool = False,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    import asym_gemm

    reason = _direct_grouped_bf16_reason(a, b_cpu, transpose_b=transpose_b)
    if reason is not None:
        raise RuntimeError(f"direct grouped BF16 AsymGEMM is unavailable: {reason}")

    m = int(a.shape[0])
    n = int(b_cpu.shape[2] if transpose_b else b_cpu.shape[1])
    a_kernel, offsets_kernel, unpad = _pad_grouped_input_for_asym(a, offsets, experts)
    d = torch.empty((int(a_kernel.shape[0]), n), device=a.device, dtype=output_dtype)
    offsets_i32, experts_i32, list_size = _group_metadata_tensors(offsets_kernel, experts, device=a.device)
    if _EP_SEP_ENABLED:
        # fix_gb200_ep.md S6 true-sEP: union work sharing over the shared bank.
        # Consumes POST-PAD pairs (BLOCK_M-aligned — the PR-5 alignment contract).
        # The .tolist() is a LOCAL drain (sdp-benign; no collective sits upstream).
        from .ep_sep import state as _sep_state

        _st = _sep_state()
        # pre_gate declines on host ints BEFORE the .tolist() GPU sync — at
        # decline-regime workloads (m/n_segs > MAX_MPE) the sync cost the whole
        # backend +6.7 s/step at 20k for launches that could never arm.
        if _st is not None and _st.pre_gate(int(a_kernel.shape[0]), int(a_kernel.shape[1]),
                                            list_size - 1):
            pairs_cpu = offsets_i32[: 2 * (list_size - 1)].cpu().tolist()
            ids_cpu = experts_i32[: list_size - 1].cpu().tolist()
            segs = [(ids_cpu[i], pairs_cpu[2 * i], pairs_cpu[2 * i + 1])
                    for i in range(list_size - 1)]
            if _st.try_armed(asym_gemm, a_kernel, b_cpu, d, segs, compiled_dims, transpose_b):
                d = _unpad_grouped_output(d, unpad, output_m=m)
                return d
    if _EP_QUEUED_ENABLED:
        # fix_gb200_ep.md S2a: entry-pop queued variant over THIS rank's own list with a
        # private counter block (side fixed per rank; steal arrives at S2b). Zero-steal
        # claims every item, so d is fully written — validated one step later
        # (head+tail == n_items per launch) instead of zero-initializing d.
        from .ep_queue import get_state

        state = get_state()
        counters = state.next_block(list_size - 1)
        asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous_ep_queued(
            a_kernel, b_cpu, d, offsets_i32, experts_i32, list_size, counters, state.side,
            compiled_dims, transpose_b
        )
    else:
        asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
            a_kernel, b_cpu, d, offsets_i32, experts_i32, list_size, compiled_dims, transpose_b
        )
    d = _unpad_grouped_output(d, unpad, output_m=m)
    return d


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


def _quantized_compiled_dims(compiled_dims: str) -> str:
    # The FP8/FP4 contiguous kernels are specialized around N/K by default.
    # Reusing the BF16 training default "mnk" can compile a numerically invalid
    # SM100 path for these quantized kernels.
    return "nk" if compiled_dims == "mnk" else compiled_dims


def _stage_quantized_tensor_for_kernel(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.device == device:
        return tensor.contiguous()
    return tensor.to(device=device, non_blocking=tensor.device.type == "cpu" and tensor.is_pinned()).contiguous()


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
    # The BF16 path is the CPU-resident training path. Quantized kernels are
    # currently more reliable when the packed cache tensors are staged to CUDA
    # before launch; the source cache remains CPU-resident and frozen.
    b_values = _stage_quantized_tensor_for_kernel(qweight.values, a.device)
    b_scales = _stage_quantized_tensor_for_kernel(qweight.scales, a.device)
    b_group = (b_values.unsqueeze(0), b_scales.unsqueeze(0))
    d = torch.empty((m, n), device=a.device, dtype=_quantized_output_dtype(a, precision=precision))
    offsets, experts = _single_group_launch_tensors(a.device, m)
    kernel_compiled_dims = _quantized_compiled_dims(compiled_dims)

    if precision == "fp8":
        asym_gemm.m_grouped_fp8_asym_gemm_nt_contiguous(
            a_quantized,
            b_group,
            d,
            offsets,
            experts,
            2,
            recipe=_FP8_RECIPE,
            compiled_dims=kernel_compiled_dims,
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
            compiled_dims=kernel_compiled_dims,
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
    b_values = _stage_quantized_tensor_for_kernel(qweight.values, a.device)
    b_scales = _stage_quantized_tensor_for_kernel(qweight.scales, a.device)
    d = torch.empty(
        (int(a_kernel.shape[0]), n),
        device=a.device,
        dtype=_quantized_output_dtype(a, precision=precision),
    )
    offsets_i32, experts_i32, list_size = _group_metadata_tensors(offsets_kernel, experts, device=a.device)
    kernel_compiled_dims = _quantized_compiled_dims(compiled_dims)

    if precision == "fp8":
        asym_gemm.m_grouped_fp8_asym_gemm_nt_contiguous(
            a_quantized,
            (b_values, b_scales),
            d,
            offsets_i32,
            experts_i32,
            list_size,
            recipe=_FP8_RECIPE,
            compiled_dims=kernel_compiled_dims,
            disable_ue8m0_cast=False,
        )
    elif precision == "fp4":
        asym_gemm.m_grouped_fp4_asym_gemm_nt_contiguous(
            a_quantized,
            (b_values, b_scales),
            d,
            offsets_i32,
            experts_i32,
            list_size,
            recipe=_FP4_RECIPE,
            compiled_dims=kernel_compiled_dims,
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
    grouped_mm = _require_torch_grouped_mm()
    return grouped_mm(mat1, mat2, offs=active_offsets)


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


def _asym_unavailable_message(
    *,
    precision: str,
    reason: str,
    grouped: bool,
    phase: str,
    transpose_b: bool,
) -> str:
    grouped_label = "grouped " if grouped else ""
    return (
        f"direct {grouped_label}{precision.upper()} AsymGEMM is unavailable "
        f"during phase={phase} transpose_b={transpose_b}: {reason}"
    )


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


def _gemm_dispatch_staged() -> bool:
    """Phase D5 (agent/impls/fix_throughput.md C1b): route CPU-weight GEMMs to the
    stage-once + native-mm path (`_torch_nt`/`_torch_grouped_nt`) instead of the
    asym streaming kernel. D2 bench receipts: dense attention shapes run at
    6-40x resident on the asym kernel vs 1.06x staged; grouped experts 2.7-3.6x
    vs 1.7x staged (per-call bank copy 3.2 ms). Route-fused ker-bit kernels are
    separate call sites and unaffected. Default off = byte-identical."""
    raw = _os.environ.get(
        "ASYM_GEMM_DISPATCH",
        _os.environ.get("ASYM_GEMM_LF_CONFIG_ASYM_GEMM_DISPATCH", "asym"),
    )
    return str(raw).strip().lower() == "staged"


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
    bf16_output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if getattr(b_cpu, "_stp", None) is not None:
        # A ROW carrier's element order is shard-major, NOT logical [N,K]; consuming it here
        # would silently compute garbage. Every sTP-sharded weight must route through
        # asym_bf16_cpu_right_matmul (the I3 choke point), which dispatches per-shard.
        raise RuntimeError(
            "sTP-sharded weight reached _dispatch_nt unrouted; route via asym_bf16_cpu_right_matmul"
        )
    _check_backend(backend)
    precision = _normalize_precision(precision)
    if backend != "torch" and precision == "bf16" and _gemm_dispatch_staged():
        backend = "torch"  # D5: stage-once + native mm; counted in torch_* stats

    if backend != "torch":
        reason = (
            _direct_bf16_reason(a, b_cpu, transpose_b=transpose_b)
            if precision == "bf16"
            else _direct_quantized_reason(a, quantized_weight, precision=precision, grouped=False)
        )
        if reason is None:
            try:
                if precision == "bf16":
                    out = _asym_bf16_nt(
                        a,
                        b_cpu,
                        compiled_dims=compiled_dims,
                        transpose_b=transpose_b,
                        output_dtype=bf16_output_dtype,
                    )
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
            raise RuntimeError(
                _asym_unavailable_message(
                    precision=precision,
                    reason=reason,
                    grouped=False,
                    phase=phase,
                    transpose_b=transpose_b,
                )
            )

    if stats is not None:
        if phase == "forward":
            stats.torch_forward_calls += 1
        else:
            stats.torch_dx_calls += 1
    return _torch_nt(a, b_cpu, transpose_b=transpose_b)


def _record_attention_cpu_right_call(
    stats: Optional[AsymExecutionStats],
    *,
    phase: str,
    tag: str,
) -> None:
    if stats is None:
        return
    phase_text = str(phase)
    if phase_text in {"attn_act_base_dx", "base_dx"}:
        stats.attn_act_base_dx_calls += 1
    elif phase_text in {"attn_act_dA", "lora_a_grad"}:
        stats.attn_act_lora_a_grad_calls += 1
    if tag:
        stats.attn_act_hbm_gemm_calls_by_tag[tag] = stats.attn_act_hbm_gemm_calls_by_tag.get(tag, 0) + 1


def asym_bf16_cpu_right_matmul(
    left: torch.Tensor,
    right_cpu: torch.Tensor,
    *,
    transpose_b: bool = False,
    backend: str = "asym",
    stats: Optional[AsymExecutionStats] = None,
    phase: str = "attn_act",
    tag: str = "",
    compiled_dims: str = "mnk",
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run a BF16 matmul with a CPU-resident right operand.

    This is the narrow public wrapper used by attention activation offload.
    `backend="asym"` requires the direct pinned-CPU BF16 AsymGEMM path;
    `backend="torch"` is a correctness fallback for focused tests.
    """

    _check_backend(backend)
    if left.dim() != 2 or right_cpu.dim() != 2:
        raise ValueError(f"asym_bf16_cpu_right_matmul expects 2D operands, got {tuple(left.shape)} and {tuple(right_cpu.shape)}")
    if left.dtype != torch.bfloat16 or right_cpu.dtype != torch.bfloat16:
        raise ValueError("asym_bf16_cpu_right_matmul expects BF16 operands")
    if right_cpu.device.type != "cpu":
        raise ValueError(f"right operand must be CPU-resident, got {right_cpu.device}")
    if not left.is_contiguous() or not right_cpu.is_contiguous():
        raise ValueError("asym_bf16_cpu_right_matmul expects contiguous operands")
    if transpose_b:
        if int(left.shape[1]) != int(right_cpu.shape[0]):
            raise ValueError(f"shape mismatch for transpose_b=True: {tuple(left.shape)} vs {tuple(right_cpu.shape)}")
    elif int(left.shape[1]) != int(right_cpu.shape[1]):
        raise ValueError(f"shape mismatch for transpose_b=False: {tuple(left.shape)} vs {tuple(right_cpu.shape)}")
    if backend == "asym":
        reason = _direct_bf16_reason(left, right_cpu, transpose_b=transpose_b)
        if reason is not None:
            raise RuntimeError(
                _asym_unavailable_message(
                    precision="bf16",
                    reason=reason,
                    grouped=False,
                    phase=phase,
                    transpose_b=transpose_b,
                )
            )
    stp_info = getattr(right_cpu, "_stp", None)
    if stp_info is not None:
        out = _stp_base_gemm(
            left,
            stp_info,
            backend=backend,
            stats=stats,
            phase=phase,
            compiled_dims=compiled_dims,
            transpose_b=transpose_b,
            output_dtype=output_dtype,
            tag=tag,
        )
        _record_attention_cpu_right_call(stats, phase=phase, tag=tag)
        return out
    out = _dispatch_nt(
        left,
        right_cpu,
        backend=backend,
        stats=stats,
        phase=phase,
        compiled_dims=compiled_dims,
        transpose_b=transpose_b,
        precision="bf16",
        profile_label=tag,
        bf16_output_dtype=output_dtype,
    )
    _record_attention_cpu_right_call(stats, phase=phase, tag=tag)
    return out


def _stp_base_gemm(
    left: torch.Tensor,
    info,
    *,
    backend: str,
    stats: Optional[AsymExecutionStats],
    phase: str,
    compiled_dims: str,
    transpose_b: bool,
    output_dtype: torch.dtype,
    tag: str,
) -> torch.Tensor:
    """Stage I3 Phase A (gb200_tp.md): ONE logical base GEMM -> two back-to-back async
    device GEMMs over the I2 shard views + one NVLink exchange. The model/residual stays
    on dev0; dev1 receives its operand via bcast01 and returns its piece via to0/to0_sum.
    RAW op — the calling fine-grained Functions own autograd (frozen base => dgrad only).

    dev1 launch order is FIRST so both GEMMs overlap (E4); the dev1 kernel runs under
    torch.cuda.device(dev1) + rt.compute[1] (JIT launches ride the CURRENT device's
    current stream); bcast01's exit contract lands its wait on compute[1] because that is
    dev1's current stream inside the with-block.
    """
    from asym_gemm.training.stp_runtime import get_runtime, _record

    rt = get_runtime()
    s0, s1 = info.shards

    def shard_gemm(operand: torch.Tensor, shard: torch.Tensor, label: str) -> torch.Tensor:
        return _dispatch_nt(
            operand,
            shard,
            backend=backend,
            stats=stats,
            phase=phase,
            compiled_dims=compiled_dims,
            transpose_b=transpose_b,
            precision="bf16",
            profile_label=label,
            bf16_output_dtype=output_dtype,
        )

    def dev1_branch(operand: torch.Tensor):
        with torch.cuda.device(rt.d[1]), torch.cuda.stream(rt.compute[1]):
            x1 = rt.bcast01(operand)
            y1 = shard_gemm(x1, s1, f"{tag}.stp1")
            ready = _record(rt.compute[1])
        return y1, ready

    if info.kind == "col":
        if not transpose_b:
            # fwd: y = x @ W^T, output split on N -> gather [M, N] on dev0
            y1, ev1 = dev1_branch(left)
            y0 = shard_gemm(left, s0, f"{tag}.stp0")
            return torch.cat([y0, rt.to0(y1, producer_event=ev1)], dim=1)
        # dX: dx = g @ W, contraction over the split N -> partial sum
        n0 = info.split
        g1 = left[:, n0:].contiguous()
        y1, ev1 = dev1_branch(g1)
        dx0 = shard_gemm(left[:, :n0].contiguous(), s0, f"{tag}.stp0")
        return rt.to0_sum(dx0, y1, producer_event=ev1)
    if info.kind != "row":
        raise RuntimeError(f"unsupported stp shard kind '{info.kind}' (grouped is Stage I7)")
    k0 = info.split
    if not transpose_b:
        # fwd: y = x @ W^T, contraction over the split K -> partial sum
        x1 = left[:, k0:].contiguous()
        y1, ev1 = dev1_branch(x1)
        y0 = shard_gemm(left[:, :k0].contiguous(), s0, f"{tag}.stp0")
        return rt.to0_sum(y0, y1, producer_event=ev1)
    # dX: dx = g @ W, output split on K -> gather [M, K] on dev0
    y1, ev1 = dev1_branch(left)
    dx0 = shard_gemm(left, s0, f"{tag}.stp0")
    return torch.cat([dx0, rt.to0(y1, producer_event=ev1)], dim=1)


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
    bf16_output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    _check_backend(backend)
    precision = _normalize_precision(precision)
    if backend != "torch" and precision == "bf16" and _gemm_dispatch_staged():
        backend = "torch"  # D5: stage-once + native grouped mm; counted in torch_* stats

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
                        output_dtype=bf16_output_dtype,
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
            raise RuntimeError(
                _asym_unavailable_message(
                    precision=precision,
                    reason=reason,
                    grouped=True,
                    phase=phase,
                    transpose_b=transpose_b,
                )
            )

    if stats is not None:
        if phase == "forward":
            stats.torch_forward_calls += 1
        else:
            stats.torch_dx_calls += 1
    return _torch_grouped_nt(a, b_cpu, offsets, experts, transpose_b=transpose_b, dense_experts=dense_experts)


class AsymFrozenLinearFunction(torch.autograd.Function):
    """Autograd node for a CPU-resident frozen base linear.

    Only the input and optional bias are differentiable. The host weight is
    intentionally data, not a trainable parameter, so backward returns no weight
    gradient and computes only dX with the configured backend.
    """

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
        bf16_output_dtype: torch.dtype,
    ) -> torch.Tensor:
        precision = _normalize_precision(precision)
        bf16_output_dtype = _normalize_bf16_output_dtype(bf16_output_dtype)
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
                bf16_output_dtype=bf16_output_dtype,
            )
        if bias is not None:
            y = y + bias.to(device=y.device, dtype=y.dtype)

        ctx.host_weight = host_weight
        ctx.backend = backend
        ctx.stats = stats
        ctx.compiled_dims = compiled_dims
        ctx.precision = precision
        ctx.bf16_output_dtype = bf16_output_dtype
        ctx.profile_backward_range = backward_range
        ctx.profile_enabled = is_profile_enabled()
        ctx.has_bias = bias is not None
        ctx.bias_device = bias.device if bias is not None else None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.input_shape = input_shape
        ctx.input_dtype = x.dtype
        return y.reshape(*input_shape[:-1], host_weight.out_features)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], None, Optional[torch.Tensor], None, None, None, None, None, None, None]:
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_output_2d = grad_output.reshape(-1, ctx.host_weight.out_features).contiguous()
            if ctx.precision == "bf16" and ctx.backend != "torch" and grad_output_2d.dtype != ctx.host_weight.weight.dtype:
                grad_output_2d = grad_output_2d.to(dtype=ctx.host_weight.weight.dtype)
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
                    bf16_output_dtype=ctx.input_dtype if ctx.precision == "bf16" else ctx.bf16_output_dtype,
                )
            grad_x = grad_x.reshape(ctx.input_shape)

        grad_bias = None
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = (
                grad_output.reshape(-1, ctx.host_weight.out_features)
                .sum(dim=0)
                .to(device=ctx.bias_device, dtype=ctx.bias_dtype)
            )
        return grad_x, None, grad_bias, None, None, None, None, None, None, None


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
    bf16_output_dtype: torch.dtype | str = torch.bfloat16,
) -> torch.Tensor:
    precision = _normalize_precision(precision)
    bf16_output_dtype = _normalize_bf16_output_dtype(bf16_output_dtype)
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
        bf16_output_dtype,
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
    bf16_output_dtype: torch.dtype | str = torch.bfloat16,
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
        bf16_output_dtype=bf16_output_dtype,
    )


class AsymGroupedFrozenLinearFunction(torch.autograd.Function):
    """Autograd node for CPU-resident frozen grouped expert linears.

    Route metadata and host weights are non-differentiable. Backward computes
    only grouped dX; expert/base weights, offsets, and expert ids receive no
    gradients.
    """

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
        bf16_output_dtype: torch.dtype,
        weight_layout: str,
    ) -> torch.Tensor:
        precision = _normalize_precision(precision)
        bf16_output_dtype = _normalize_bf16_output_dtype(bf16_output_dtype)
        weight_layout = _normalize_grouped_weight_layout(weight_layout)
        in_features, out_features, forward_transpose_b, backward_transpose_b = _grouped_weight_features(
            host_weight.weight,
            weight_layout,
        )
        if x.shape[-1] != in_features:
            raise ValueError(f"expected input last dim {in_features}, got {x.shape[-1]}")
        offsets = offsets.detach().contiguous()
        experts = experts.detach().contiguous()
        input_shape = tuple(x.shape)
        x_2d = x.reshape(-1, in_features).contiguous()
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
                transpose_b=forward_transpose_b,
                precision=precision,
                quantized_weight=quantized_weight,
                dense_experts=dense_experts,
                bf16_output_dtype=bf16_output_dtype,
            )

        ctx.host_weight = host_weight
        ctx.offsets = offsets
        ctx.experts = experts
        ctx.backend = backend
        ctx.stats = stats
        ctx.compiled_dims = compiled_dims
        ctx.precision = precision
        ctx.bf16_output_dtype = bf16_output_dtype
        ctx.dense_experts = bool(dense_experts)
        ctx.backward_transpose_b = backward_transpose_b
        ctx.profile_backward_range = backward_range
        ctx.profile_enabled = is_profile_enabled()
        ctx.input_shape = input_shape
        ctx.input_dtype = x.dtype
        ctx.out_features = out_features
        return y.reshape(*input_shape[:-1], out_features)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], None, None, None, None, None, None, None, None, None, None, None, None]:
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_output_2d = grad_output.reshape(-1, int(ctx.out_features)).contiguous()
            if ctx.precision == "bf16" and ctx.backend != "torch" and grad_output_2d.dtype != ctx.host_weight.weight.dtype:
                grad_output_2d = grad_output_2d.to(dtype=ctx.host_weight.weight.dtype)
            if ctx.precision == "bf16":
                b_cpu = ctx.host_weight.weight
                transpose_b = bool(ctx.backward_transpose_b)
                quantized_weight_t = None
            else:
                b_cpu = ctx.host_weight.weight
                transpose_b = bool(ctx.backward_transpose_b)
                quantized_weight_t = (
                    _get_quantized_host_weight(ctx.host_weight, ctx.precision, transpose=transpose_b)
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
                    bf16_output_dtype=ctx.input_dtype if ctx.precision == "bf16" else ctx.bf16_output_dtype,
                )
            grad_x = grad_x.reshape(ctx.input_shape)
        return grad_x, None, None, None, None, None, None, None, None, None, None, None, None


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
    bf16_output_dtype: torch.dtype | str = torch.bfloat16,
    weight_layout: str = "out_in",
) -> torch.Tensor:
    precision = _normalize_precision(precision)
    bf16_output_dtype = _normalize_bf16_output_dtype(bf16_output_dtype)
    weight_layout = _normalize_grouped_weight_layout(weight_layout)
    if precision != "bf16" and weight_layout != "out_in":
        raise NotImplementedError("non-bf16 grouped host weights currently require out_in layout")
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
        bf16_output_dtype,
        weight_layout,
    )


class AsymGroupedFrozenLinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        *,
        backend: str = "asym",
        pin_memory: bool = True,
        clone: bool = True,
        stats: Optional[AsymExecutionStats] = None,
        compiled_dims: str = "mnk",
        precision: str = "bf16",
        bf16_output_dtype: torch.dtype | str = torch.bfloat16,
        weight_layout: str = "out_in",
    ) -> None:
        super().__init__()
        adopted_host_weight = weight if isinstance(weight, HostWeight) else None
        if adopted_host_weight is not None:
            weight = adopted_host_weight.weight
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"weight must be a torch.Tensor, got {type(weight)!r}")
        if weight.dim() != 3:
            raise ValueError(f"AsymGroupedFrozenLinear expects [groups, out, in], got {tuple(weight.shape)}")
        _check_backend(backend)
        precision = _normalize_precision(precision)
        bf16_output_dtype = _normalize_bf16_output_dtype(bf16_output_dtype)
        weight_layout = _normalize_grouped_weight_layout(weight_layout)
        self.host_weight = adopted_host_weight or HostWeight(weight, pin_memory=pin_memory, clone=clone, require_2d=False)
        self.backend = backend
        self.stats = stats if stats is not None else AsymExecutionStats()
        self.compiled_dims = compiled_dims
        self.precision = precision
        self.bf16_output_dtype = bf16_output_dtype
        self.weight_layout = weight_layout
        self.profile_name = ""
        self.num_groups = int(self.host_weight.weight.shape[0])
        self.in_features, self.out_features, self.forward_transpose_b, self.backward_transpose_b = _grouped_weight_features(
            self.host_weight.weight,
            self.weight_layout,
        )

    @property
    def pinned_cpu_bytes(self) -> int:
        return self.host_weight.pinned_cpu_bytes + _quantized_cache_pinned_bytes(self.host_weight, self.precision)

    @property
    def weight_hbm_saved_bytes(self) -> int:
        return self.host_weight.weight_nbytes

    @property
    def weight(self) -> torch.Tensor:
        return self.host_weight.weight

    def forward(
        self,
        x: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        *,
        dense_experts: bool = False,
        profile_name: str | None = None,
        compiled_dims: str | None = None,
    ) -> torch.Tensor:
        effective_profile_name = self.profile_name if profile_name is None else profile_name
        return asym_grouped_frozen_linear(
            x,
            self.host_weight,
            offsets,
            experts,
            backend=self.backend,
            stats=self.stats,
            compiled_dims=self.compiled_dims if compiled_dims is None else compiled_dims,
            profile_name=effective_profile_name,
            precision=self.precision,
            dense_experts=dense_experts,
            bf16_output_dtype=self.bf16_output_dtype,
            weight_layout=self.weight_layout,
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
                self.in_features, self.out_features, self.forward_transpose_b, self.backward_transpose_b = _grouped_weight_features(
                    self.host_weight.weight,
                    self.weight_layout,
                )
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
            f"precision={self.precision}, "
            f"weight_layout={self.weight_layout}, "
            f"bf16_output_dtype={str(self.bf16_output_dtype).removeprefix('torch.')}, "
            f"pinned={self.host_weight.metadata.pinned}"
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
        weight_layout: str,
    ) -> torch.Tensor:
        weight_layout = _normalize_grouped_weight_layout(weight_layout)
        in_features, out_features, forward_transpose_b, backward_transpose_b = _grouped_weight_features(weight, weight_layout)
        if x.shape[-1] != in_features:
            raise ValueError(f"expected input last dim {in_features}, got {x.shape[-1]}")
        input_shape = tuple(x.shape)
        x_2d = x.reshape(-1, in_features).contiguous()
        forward_range = "forward.grouped_base_torch" if not profile_name else f"forward.{profile_name}.grouped_base_torch"
        backward_range = "backward.grouped_base_dx_torch" if not profile_name else f"backward.{profile_name}.grouped_base_dx_torch"
        with prof_range(forward_range):
            y = _grouped_torch_chunks(
                x_2d,
                weight,
                offsets,
                experts,
                transpose_b=forward_transpose_b,
                dense_experts=dense_experts,
            )

        ctx.save_for_backward(weight, offsets, experts)
        ctx.dense_experts = bool(dense_experts)
        ctx.backward_transpose_b = backward_transpose_b
        ctx.profile_backward_range = backward_range
        ctx.profile_enabled = is_profile_enabled()
        ctx.input_shape = input_shape
        ctx.out_features = out_features
        return y.reshape(*input_shape[:-1], out_features)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], None, None, None, None, None, None]:
        grad_x = None
        if ctx.needs_input_grad[0]:
            weight, offsets, experts = ctx.saved_tensors
            grad_output_2d = grad_output.reshape(-1, int(ctx.out_features)).contiguous()
            with prof_range(ctx.profile_backward_range, enabled=ctx.profile_enabled):
                grad_x = _grouped_torch_chunks(
                    grad_output_2d,
                    weight,
                    offsets,
                    experts,
                    transpose_b=bool(ctx.backward_transpose_b),
                    dense_experts=ctx.dense_experts,
                )
            grad_x = grad_x.reshape(ctx.input_shape)
        return grad_x, None, None, None, None, None, None


class TorchGroupedFrozenLinear(nn.Module):
    """GPU-resident grouped frozen linear for the all-HBM torch baseline."""

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
        weight_layout: str = "out_in",
    ) -> None:
        super().__init__()
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"weight must be a torch.Tensor, got {type(weight)!r}")
        if weight.dim() != 3:
            raise ValueError(f"TorchGroupedFrozenLinear expects [groups, out, in], got {tuple(weight.shape)}")
        self.weight_layout = _normalize_grouped_weight_layout(weight_layout)
        self.register_buffer("weight", weight.detach().to(device=device, dtype=dtype).contiguous())
        self.profile_name = ""
        self.num_groups = int(self.weight.shape[0])
        self.in_features, self.out_features, self.forward_transpose_b, self.backward_transpose_b = _grouped_weight_features(
            self.weight,
            self.weight_layout,
        )

    @property
    def pinned_cpu_bytes(self) -> int:
        return 0

    @property
    def weight_hbm_saved_bytes(self) -> int:
        return tensor_nbytes(self.weight)

    @property
    def gpu_resident_weight_bytes(self) -> int:
        return tensor_nbytes(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        *,
        dense_experts: bool = False,
        profile_name: str | None = None,
    ) -> torch.Tensor:
        effective_profile_name = self.profile_name if profile_name is None else profile_name
        return TorchGroupedFrozenLinearFunction.apply(
            x,
            self.weight,
            offsets,
            experts,
            effective_profile_name,
            dense_experts,
            self.weight_layout,
        )

    def extra_repr(self) -> str:
        return (
            f"num_groups={self.num_groups}, in_features={self.in_features}, "
            f"out_features={self.out_features}, weight_layout={self.weight_layout}, "
            f"device={self.weight.device}, dtype={self.weight.dtype}"
        )


def direct_asym_capability(
    a: torch.Tensor,
    host_weight: HostWeight,
    *,
    transposed_weight: bool = False,
) -> AsymCapability:
    supported, reason = can_use_direct_bf16(a, host_weight, transpose=transposed_weight)
    return AsymCapability(supported=supported, reason=reason)


def _prepare_frozen_bias(
    bias: Optional[torch.Tensor],
    *,
    dtype: torch.dtype,
    pin_memory: bool,
    strict_no_copy: bool,
) -> Optional[torch.Tensor]:
    if bias is None:
        return None
    bias_cpu = bias.detach()
    if strict_no_copy:
        if bias_cpu.device.type != "cpu":
            raise RuntimeError(f"frozen bias selected for CPU offload but source tensor is on {bias_cpu.device}")
        if bias_cpu.dtype != dtype:
            raise RuntimeError(f"frozen bias selected for CPU offload has dtype {bias_cpu.dtype}, expected {dtype}")
        if not bias_cpu.is_contiguous():
            raise RuntimeError("frozen bias selected for CPU offload is non-contiguous and would require a CPU copy")
        bias_cpu.requires_grad_(False)
        return bias_cpu

    bias_cpu = bias_cpu.to("cpu", dtype=dtype)
    if not bias_cpu.is_contiguous():
        bias_cpu = bias_cpu.contiguous()
    if pin_memory:
        try:
            bias_cpu = bias_cpu.pin_memory()
        except RuntimeError:
            pass
    bias_cpu.requires_grad_(False)
    return bias_cpu


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
        bf16_output_dtype: torch.dtype | str = torch.bfloat16,
        strict_no_copy_bias: bool = False,
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
        adopted_host_weight = weight if isinstance(weight, HostWeight) else None
        if adopted_host_weight is not None:
            weight = adopted_host_weight.weight
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"weight must be a torch.Tensor, got {type(weight)!r}")
        _check_backend(backend)
        precision = _normalize_precision(precision)
        bf16_output_dtype = _normalize_bf16_output_dtype(bf16_output_dtype)
        self.host_weight = adopted_host_weight or HostWeight.from_tensor(weight, dtype=weight.dtype, pin_memory=pin_memory)
        self.bias_cpu = _prepare_frozen_bias(
            bias,
            dtype=weight.dtype,
            pin_memory=pin_memory,
            strict_no_copy=strict_no_copy_bias,
        )
        self.backend = backend
        self.stats = stats if stats is not None else AsymExecutionStats()
        self.compiled_dims = compiled_dims
        self.precision = precision
        self.bf16_output_dtype = bf16_output_dtype
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
        bf16_output_dtype: torch.dtype | str = torch.bfloat16,
    ) -> "AsymFrozenLinear":
        return cls(
            linear.weight.detach(),
            bias=None if linear.bias is None else linear.bias.detach(),
            backend=backend,
            pin_memory=pin_memory,
            stats=stats,
            compiled_dims=compiled_dims,
            precision=precision,
            bf16_output_dtype=bf16_output_dtype,
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
        bf16_output_dtype: torch.dtype | str = torch.bfloat16,
    ) -> "AsymFrozenLinear":
        return cls.from_gpu_linear(
            linear,
            backend=backend,
            pin_memory=pin_memory,
            stats=stats,
            compiled_dims=compiled_dims,
            precision=precision,
            bf16_output_dtype=bf16_output_dtype,
        )

    @classmethod
    def from_host_weight(
        cls,
        host_weight: HostWeight,
        *,
        bias: Optional[torch.Tensor] = None,
        backend: str = "asym",
        stats: Optional[AsymExecutionStats] = None,
        compiled_dims: str = "mnk",
        precision: str = "bf16",
        bf16_output_dtype: torch.dtype | str = torch.bfloat16,
    ) -> "AsymFrozenLinear":
        return cls(
            host_weight,
            bias=bias,
            backend=backend,
            pin_memory=False,
            stats=stats,
            compiled_dims=compiled_dims,
            precision=precision,
            bf16_output_dtype=bf16_output_dtype,
            strict_no_copy_bias=True,
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
    def cpu_resident_base_weight_bytes(self) -> int:
        return self.host_weight.weight_nbytes

    @property
    def gpu_resident_base_weight_bytes(self) -> int:
        return 0

    @property
    def weight(self) -> torch.Tensor:
        return self.host_weight.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.bias_cpu

    def asym_liger_lm_head_weight(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.bias_cpu is not None:
            raise RuntimeError("Asym Liger lm_head bridge currently requires a bias-free lm_head.")
        weight = self.host_weight.weight
        if weight.requires_grad:
            raise RuntimeError("Asym Liger lm_head bridge supports frozen lm_head only.")
        if weight.ndim != 2:
            raise RuntimeError(f"Asym Liger lm_head bridge expected a 2D weight, got {tuple(weight.shape)}.")
        if not weight.is_contiguous():
            weight = weight.contiguous()
        return weight.to(
            device=device,
            dtype=dtype,
            non_blocking=bool(weight.device.type == "cpu" and weight.is_pinned()),
        )

    def forward(self, x: torch.Tensor, *, profile_name: str | None = None) -> torch.Tensor:
        effective_profile_name = self.profile_name if profile_name is None else profile_name
        return asym_frozen_linear(
            x,
            self.host_weight,
            bias=self.bias_cpu,
            backend=self.backend,
            stats=self.stats,
            compiled_dims=self.compiled_dims,
            profile_name=effective_profile_name,
            precision=self.precision,
            bf16_output_dtype=self.bf16_output_dtype,
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
            f"backend={self.backend}, precision={self.precision}, "
            f"bf16_output_dtype={str(self.bf16_output_dtype).removeprefix('torch.')}, "
            f"pinned={self.host_weight.metadata.pinned}"
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
