import enum
import random
import torch
from typing import Generator, List, Optional, Tuple

from deep_gemm.testing import get_arch_major
from asym_gemm.utils import (
    align, ceil_div,
    per_token_cast_to_fp8, per_channel_cast_to_fp8, per_block_cast_to_fp8,
    per_token_cast_to_fp4, per_block_cast_to_fp4, transpose_packed_fp4,
    get_mk_alignment_for_contiguous_layout
)


class KernelType(enum.Enum):
    Kernel1D1D = 0
    Kernel1D2D = 1
    KernelNoSF = 2

    def is_1d1d(self):
        return self.value == 0

    def is_1d2d(self):
        return self.value == 1

    def is_nosf(self):
        return self.value == 2


class MajorTypeAB(enum.Enum):
    KMajor = 0
    MNMajor = 1

    def is_k_major(self):
        return self.value == 0

    def is_mn_major(self):
        return self.value == 1
    

class QuantConfig:
    _legacy_quant_config = (128, 128, False, False)

    def __init__(self, value: Tuple[int, int, bool, bool] = _legacy_quant_config):
        self.gran_k_a, self.gran_k_b, self.is_fp4_a, self.is_fp4_b = value

    def print(self):
        print(f' > Testing with gran_k_a={self.gran_k_a}, gran_k_b={self.gran_k_b}, '
              f'is_fp4_a={self.is_fp4_a}, is_fp4_b={self.is_fp4_b}')

    def is_legacy(self) -> bool:
        return (self.gran_k_a, self.gran_k_b, self.is_fp4_a, self.is_fp4_b) == self._legacy_quant_config

    def get_recipes(self, is_wgrad: bool = False) -> Tuple[Tuple, Tuple, Tuple]:
        recipe, recipe_a, recipe_b = None, None, None
        if self.is_legacy():
            recipe = (1, 1, 128) if is_wgrad else None
        else:
            recipe_a = (1, self.gran_k_a)
            recipe_b = (1, self.gran_k_b) if self.is_fp4_b or is_wgrad else (self.gran_k_b, self.gran_k_b)
        return recipe, recipe_a, recipe_b

    def max_diff(self) -> float:
        if self.is_fp4_a and self.is_fp4_b:
            return 0.02
        if self.is_fp4_a or self.is_fp4_b:
            return 0.01
        return 0.001

    @staticmethod
    def get_list_from_dtype(dtype: torch.dtype) -> List:
        if dtype == torch.bfloat16:
            return [None]
        quant_config_list = [QuantConfig()]
        if get_arch_major() == 10:
            quant_config_list.append(QuantConfig((128, 32, False, True)))
        return quant_config_list


def reset_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_ue8m0_usage(kernel_type: KernelType) -> bool:
    if get_arch_major() == 9:
        return False
    return kernel_type.is_1d1d()


def get_kernel_types(dtype: torch.dtype) -> tuple:
    if dtype == torch.bfloat16:
        return (KernelType.KernelNoSF, )

    return (KernelType.Kernel1D2D, ) if get_arch_major() == 9 else (KernelType.Kernel1D1D, )


def get_major_ab(allow_a_mn_major: bool, allow_b_mn_major: bool) -> Generator:
    for major_a in (MajorTypeAB.KMajor, MajorTypeAB.MNMajor):
        for major_b in (MajorTypeAB.KMajor, MajorTypeAB.MNMajor):
            if major_a.is_mn_major() and not allow_a_mn_major:
                continue
            if major_b.is_mn_major() and not allow_b_mn_major:
                continue
            yield major_a, major_b

def enumerate_normal(dtype: torch.dtype) -> Generator:
    assert dtype in (torch.float8_e4m3fn, torch.bfloat16)

    quant_config_list = QuantConfig.get_list_from_dtype(dtype)
    fp32_output_nk = [(256, 7168), (129280, 7168)]
    bf16_output_nk = [(2112, 7168), (576, 7168), (24576, 1536), (32768, 512), (7168, 16384), (4096, 7168), (7168, 2048)]
    m_fwd_list, m_bwd_list = [1, 128, 4096], [4096, ]
    nk_list = list(bf16_output_nk)

    # Only BF16 GEMM needs FP32 outputs
    if dtype == torch.bfloat16:
        nk_list += fp32_output_nk

    for kernel_type in get_kernel_types(dtype):
        for quant_config in quant_config_list:
            if len(quant_config_list) > 1:
                quant_config.print()
            reset_seed()

            # Forward
            for m in m_fwd_list:
                for i in range(len(nk_list)):
                    n, k = nk_list[i]
                    out_dtype = torch.bfloat16 if i < len(bf16_output_nk) else torch.float
                    yield kernel_type, quant_config, m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor, False, out_dtype

            # Backward
            for m in m_bwd_list:
                for n, k in nk_list:
                    override_major = MajorTypeAB.MNMajor
                    override_kernel_type = kernel_type
                    if get_arch_major() == 9 and dtype == torch.float8_e4m3fn:
                        override_major = MajorTypeAB.KMajor
                        override_kernel_type = KernelType.Kernel1D1D
                    yield kernel_type,          quant_config, m, k, n, MajorTypeAB.KMajor, override_major, False, torch.bfloat16     # Dgrad
                    yield override_kernel_type, quant_config, n, m, k, override_major,     override_major, True,  torch.float        # Wgrad
                    yield override_kernel_type, quant_config, n, m, k, override_major,     override_major, False, torch.bfloat16     # Wgrad

def enumerate_m_grouped_contiguous(dtype: torch.dtype) -> Generator:
    for kernel_type in get_kernel_types(dtype):
        for num_groups, expected_m_per_group, n, k in ((4, 8192, 4096, 512), (4, 8192, 7168, 2048), (8, 4096, 4096, 7168), (8, 4096, 7168, 2048)):
            for major_a, major_b in get_major_ab(False, get_arch_major() != 9 or dtype != torch.float8_e4m3fn):
                yield kernel_type, num_groups, expected_m_per_group, n, k, major_a, major_b

def enumerate_m_grouped_masked(dtype: torch.dtype) -> Generator:
    quant_config_list = QuantConfig.get_list_from_dtype(dtype)
    max_m = 40960
    m_group_list = [(6, 1024), (32, 192), (32, 50)]
    n_k_list = [(6144, 7168), (7168, 3072), (4096, 4096), (4096, 2048)]
    for kernel_type in get_kernel_types(dtype):
        for quant_config in quant_config_list:
            if len(quant_config_list) > 1:
                quant_config.print()
            for num_groups, m in m_group_list:
                for n, k in n_k_list:
                    yield kernel_type, quant_config, num_groups, max_m, m, n, k, False

def enumerate_k_grouped_contiguous(dtype: torch.dtype):
    # Only K-major is supported for SM90 FP8
    major_a, major_b = (MajorTypeAB.KMajor, MajorTypeAB.KMajor) if get_arch_major() == 9 and dtype == torch.float8_e4m3fn \
                       else (MajorTypeAB.MNMajor, MajorTypeAB.MNMajor)
    # Must with FP32 accumulation and 1D1D kernels
    for num_groups, m, n, expected_k_per_group in (( 4, 4096, 7168, 8192), ( 4, 7168, 2048, 8192),   # EP64
                                                   ( 8, 4096, 7168, 4096), ( 8, 7168, 2048, 4096),   # EP32
                                                   (16, 4096, 7168, 2048), (16, 7168, 2048, 2048)):  # EP16
        ks = [align(int(expected_k_per_group * random.uniform(0.7, 1.3)), get_mk_alignment_for_contiguous_layout()) for _ in range(num_groups)]
        yield num_groups, m, n, major_a, major_b, ks, expected_k_per_group


def enumerate_sf_layout():
    for use_ue8m0 in (False, True):
        for with_transpose in (True, False):
            for mn in (4096, 4097, 8192):
                for k in (128, 7168, 7296):
                    for num_groups in (1, 2, 4):
                        yield mn, k, with_transpose, use_ue8m0, num_groups


def enumerate_k_grouped_sf_layout():
    alignment = get_mk_alignment_for_contiguous_layout()
    assert alignment % 128 == 0
    for mn in (4096, 7168):
        for num_groups, avg_k in ((16, 2048), (8, 4096), (72, 384), (128, 256)):
            ks = [align(int(random.uniform(0.7, 1.3) * avg_k), alignment) for _ in range(num_groups)]
            yield mn, ks, num_groups


def enumerate_transpose():
    for mn in (64, 4096, 16384):
        for delta in (0, 101, 202, 303):
            for k in (128, 1024, 4096, 9984, 16384):
                yield mn + delta, k


def cast_fp8_fp4_with_major(x: torch.Tensor, major: MajorTypeAB, gran_k: int, is_fp4: bool,
                            use_ue8m0: bool, use_block_cast_for_fp8: bool = False):
    if is_fp4:
        x_fp4 = per_token_cast_to_fp4(x, use_ue8m0=use_ue8m0, gran_k=gran_k)
        x = x_fp4 if major.is_k_major() else (transpose_packed_fp4(x_fp4[0]).T, x_fp4[1])
    else:
        x_fp8 = per_block_cast_to_fp8(x, use_ue8m0=use_ue8m0, gran_k=gran_k) if use_block_cast_for_fp8 \
                else per_token_cast_to_fp8(x, use_ue8m0=use_ue8m0, gran_k=gran_k)
        x = x_fp8 if major.is_k_major() else (x_fp8[0].T.contiguous().T, x_fp8[1])
    return x


def grouped_cast_fp8_fp4_with_major(x: torch.Tensor, major: MajorTypeAB, gran_k: int, is_fp4: bool,
                                    use_ue8m0: bool, use_block_cast_for_fp8: bool = False):
    num_groups, mn, k = x.size()
    if is_fp4:
        x_fp4 = (torch.empty((num_groups, mn, k // 2), device='cuda', dtype=torch.uint8) if major.is_k_major() else \
                 torch.empty((num_groups, k, mn // 2), device='cuda', dtype=torch.uint8),
                 torch.empty((num_groups, mn, ceil_div(k, gran_k)), device='cuda', dtype=torch.float))
        for i in range(num_groups):
            x_i_fp4 = per_token_cast_to_fp4(x[i], use_ue8m0=use_ue8m0, gran_k=gran_k)
            x_fp4[0][i], x_fp4[1][i] = x_i_fp4 if major.is_k_major() else (transpose_packed_fp4(x_i_fp4[0]), x_i_fp4[1])
        x = x_fp4 if major.is_k_major() else (x_fp4[0].mT, x_fp4[1])
    else:
        x_fp8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn),
                 torch.empty((num_groups, ceil_div(mn, gran_k), ceil_div(k, gran_k)), device='cuda', dtype=torch.float) if use_block_cast_for_fp8 \
                 else torch.empty((num_groups, mn, ceil_div(k, gran_k)), device='cuda', dtype=torch.float))
        for i in range(num_groups):
            x_fp8[0][i], x_fp8[1][i] = per_block_cast_to_fp8(x[i], use_ue8m0=use_ue8m0, gran_k=gran_k) if use_block_cast_for_fp8 \
                                       else per_token_cast_to_fp8(x[i], use_ue8m0=use_ue8m0, gran_k=gran_k)
        x = x_fp8 if major.is_k_major() else (x_fp8[0].mT.contiguous().mT, x_fp8[1])
    return x


def generate_normal(m: int, n: int, k: int,
                    major_a: MajorTypeAB, major_b: MajorTypeAB,
                    accumulate: bool, out_dtype: torch.dtype,
                    kernel_type: KernelType,
                    use_ue8m0: bool = False, use_bf16: bool = False,
                    quant_config: Optional[QuantConfig] = None):
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    d = torch.randn((m, n), device='cuda', dtype=out_dtype) * 32 if accumulate else \
        torch.empty((m, n), device='cuda', dtype=out_dtype)
    c = d if accumulate else None
    ref_d = (a.float() @ b.float().t() + (c if accumulate else 0)).to(out_dtype)

    if use_bf16:
        a = a if major_a.is_k_major() else a.T.contiguous().T
        b = b if major_b.is_k_major() else b.T.contiguous().T
        return a, b, c, d, ref_d
    
    quant_config = QuantConfig() if quant_config is None else quant_config
    a = cast_fp8_fp4_with_major(a, major_a, quant_config.gran_k_a, quant_config.is_fp4_a, use_ue8m0)
    b = cast_fp8_fp4_with_major(b, major_b, quant_config.gran_k_b, quant_config.is_fp4_b, use_ue8m0,
                                use_block_cast_for_fp8=not (kernel_type.is_1d1d() and accumulate))

    return a, b, c, d, ref_d


def generate_m_grouped_contiguous(num_groups: int, expected_m_per_group: int, n: int, k: int,
                                  major_a: MajorTypeAB, major_b: MajorTypeAB,
                                  use_ue8m0: bool = False, use_bf16: bool = False,
                                  use_psum_layout: bool = False,
                                  quant_config: Optional[QuantConfig] = None,  return_original: bool = False):
    actual_ms = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
    aligned_ms = [align(actual_m, get_mk_alignment_for_contiguous_layout()) for actual_m in actual_ms]
    m = sum(aligned_ms)

    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    m_indices = torch.empty(num_groups, device='cuda', dtype=torch.int32) if use_psum_layout \
                     else torch.empty(m, device='cuda', dtype=torch.int32)
    d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_d = torch.randn((m, n), device='cuda', dtype=torch.bfloat16)

    start = 0
    for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end = start + actual_m
        aligned_end = start + aligned_m
        if use_psum_layout:
            m_indices[i] = actual_end
        else:
            m_indices[start: actual_end] = i
            m_indices[actual_end: aligned_end] = -1
        a[actual_end: aligned_end] = 0
        ref_d[start: aligned_end] = a[start: aligned_end] @ b[i].t()
        start = aligned_end

    if use_bf16:
        b = b if major_b.is_k_major() else b.mT.contiguous().mT
        return m, a, b, m_indices, d, ref_d

    assert major_a.is_k_major()
    a_fp8 = per_token_cast_to_fp8(a, use_ue8m0=use_ue8m0)
    b_fp8 = (torch.empty_like(b, dtype=torch.float8_e4m3fn),
             torch.empty((num_groups, ceil_div(n, 128), ceil_div(k, 128)), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        b_fp8[0][i], b_fp8[1][i] = per_block_cast_to_fp8(b[i], use_ue8m0=use_ue8m0)
    b_fp8 = b_fp8 if major_b.is_k_major() else (b_fp8[0].mT.contiguous().mT, b_fp8[1])
    if return_original:
        return m, a_fp8, b_fp8, m_indices, d, ref_d, a, b
    return m, a_fp8, b_fp8, m_indices, d, ref_d

def layout_masked_to_psum(x: torch.Tensor, psum_m: torch.Tensor):
    num_groups, max_m, _ = x.size()
    x_psum = torch.empty_like(x).view(num_groups * max_m, -1)
    last_psum_m = 0
    for i in range(num_groups):
        x_psum[last_psum_m: psum_m[i]] = x[i, :psum_m[i] - last_psum_m]
        last_psum_m = align(psum_m[i], 128)
    return x_psum


def generate_m_grouped_masked(num_groups: int, max_m: int, expected_m_per_group: int, n: int, k: int,
                              use_ue8m0: bool = False, use_bf16: bool = False,
                              use_psum_layout: bool = False,
                              quant_config: Optional[QuantConfig] = None):
    a = torch.randn((num_groups, max_m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    d = torch.empty((num_groups, max_m, n), device='cuda', dtype=torch.bfloat16)
    ref_d = torch.einsum('gmk,gnk->gmn', a, b)

    masked_m = torch.empty((num_groups, ), device='cuda', dtype=torch.int)
    psum_m = torch.empty((num_groups, ), device='cuda', dtype=torch.int)
    for j in range(num_groups):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
        psum_m[j] = (0 if j == 0 else align(psum_m[j - 1], 128)) + masked_m[j]
    assert masked_m.amax().item() <= max_m

    if use_bf16:
        return a, b, masked_m, psum_m, d, ref_d

    quant_config = QuantConfig() if quant_config is None else quant_config
    a = grouped_cast_fp8_fp4_with_major(a, MajorTypeAB.KMajor, quant_config.gran_k_a, quant_config.is_fp4_a, use_ue8m0)
    b = grouped_cast_fp8_fp4_with_major(b, MajorTypeAB.KMajor, quant_config.gran_k_b, quant_config.is_fp4_b, use_ue8m0, use_block_cast_for_fp8=True)    

    return a, b, masked_m, psum_m, d, ref_d


def generate_k_grouped_contiguous(num_groups: int, m: int, n: int, major_a: MajorTypeAB, major_b: MajorTypeAB, ks: List[int],
                                  use_ue8m0: bool = False, use_bf16: bool = False):
    assert get_mk_alignment_for_contiguous_layout() % 128 == 0
    k = sum(ks)

    a = torch.randn((k, m), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((k, n), device='cuda', dtype=torch.bfloat16)
    c = torch.randn((num_groups, m, n), device='cuda', dtype=torch.float) * 32
    d = c
    ref_d = torch.empty_like(c)

    start = 0
    for i, group_k in enumerate(ks):
        end = start + group_k
        ref_d[i] = c[i] + (a[start:end].T @ b[start:end])
        start = end

    if use_bf16:
        assert (major_a, major_b) == (MajorTypeAB.MNMajor, MajorTypeAB.MNMajor)
        return k, a, b, c, d, ref_d

    a_fp8 = per_channel_cast_to_fp8(a, use_ue8m0=use_ue8m0)
    b_fp8 = per_channel_cast_to_fp8(b, use_ue8m0=use_ue8m0)

    # Transpose for K Major A/B
    if (major_a, major_b) == (MajorTypeAB.KMajor, MajorTypeAB.KMajor):
        a, sfa = a_fp8
        b, sfb = b_fp8
        new_a = torch.empty((sum(ks) * m, ), dtype=a.dtype, device=a.device)
        new_b = torch.empty((sum(ks) * n, ), dtype=b.dtype, device=b.device)
        prefix = 0
        for K in ks:
            new_a[prefix * m : (prefix + K) * m] = a[prefix : prefix + K, ].T.flatten()
            new_b[prefix * n : (prefix + K) * n] = b[prefix : prefix + K, ].T.flatten()
            prefix += K
        a_fp8, b_fp8 = (new_a, sfa.T), (new_b, sfb.T)
    else:
        assert (major_a, major_b) == (MajorTypeAB.MNMajor, MajorTypeAB.MNMajor)

    return k, a_fp8, b_fp8, c, d, ref_d


def enumerate_m_grouped_contiguous_fp4() -> Generator:
    for num_groups, expected_m_per_group, n, k in ((4, 8192, 4096, 512), (4, 8192, 7168, 2048), (8, 4096, 4096, 7168), (8, 4096, 7168, 2048)):
        yield num_groups, expected_m_per_group, n, k

def get_fp4_quantization_module(backend: str = "100"):
    backend_modules = {
        "121": gen_fp4_quantization_sm121_module,
        "120f": gen_fp4_quantization_sm120f_module,
        "120": gen_fp4_quantization_sm120_module,
        "110": gen_fp4_quantization_sm110_module,
        "103": gen_fp4_quantization_sm103_module,
        "100": gen_fp4_quantization_sm100_module,
        "90": gen_fp4_quantization_sm90_module,
    }

    # Prefer 'f' (family / feature-set) variant for SM12x when CUDA >= 12.9,
    # as it enables native FP4 conversion instructions (cvt.rn.satfinite.e2m1x2.f32).
    # sm_120f covers the entire SM12x family (both SM120 and SM121).
    # See: https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/
    if backend in ("120", "121"):
        from .utils import version_at_least

        if version_at_least(torch.version.cuda, "12.9"):
            backend = "120f"

    if backend not in backend_modules:
        raise ValueError(f"Invalid backend: {backend}")

    module = backend_modules[backend]().build_and_load()

    @register_custom_op(
        "flashinfer::fp4_quantize_sm100",
        mutates_args=(""),
    )
    def fp4_quantize_sm100(
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
        is_sf_8x4_layout: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize input tensor to FP4 format.

        Args:
            input (torch.Tensor): Input tensor of shape [M, K] with dtype fp16/bf16/fp8_quantized.
            global_scale (torch.Tensor, optional): Global scale factor of shape [1] and dtype float32.
            sf_vec_size (int, optional): Scale factor vector size. Defaults to 16.
            sf_use_ue8m0 (bool, optional): Whether to use UE8M0 format for scale factors. Defaults to False.
            is_sf_swizzled_layout (bool, optional): Whether to use swizzled layout for scale factors. Defaults to True.
            is_sf_8x4_layout (bool, optional): Whether to use 8x4 layout or 128x4 layout for scale factors. Defaults to False.
            enable_pdl (Optional[bool], optional): Whether to enable PDL (Programmatic Dependent Launch).
                If None, automatically detects based on device capability. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Quantized tensor of shape [M, K/2] with dtype FLOAT4_E2M1X2
                - Scale factors tensor with shape determined by layout and sf_vec_size
        """
        if enable_pdl is None:
            enable_pdl = device_support_pdl(input.device)
        out_val = torch.empty(
            (*input.shape[:-1], input.shape[-1] // 2),
            dtype=torch.uint8,
            device=input.device,
        )
        m = input.numel() // input.shape[-1]
        k = input.shape[-1]
        if is_sf_swizzled_layout:
            out_sf_size = _compute_swizzled_layout_sf_size(
                m, k // sf_vec_size, 8 if is_sf_8x4_layout else 128
            )
            out_sf_size_padded = out_sf_size
        else:
            out_sf_size = m * k // sf_vec_size
            out_sf_size_padded = round_up(m, 16) * k // sf_vec_size
        out_sf = torch.empty(
            (out_sf_size_padded,), dtype=torch.uint8, device=input.device
        )
        module.fp4_quantize(
            input,
            global_scale,
            out_val,
            out_sf,
            sf_vec_size,
            sf_use_ue8m0,
            is_sf_swizzled_layout,
            is_sf_8x4_layout,
            enable_pdl,
        )
        return out_val, out_sf[:out_sf_size]

    @register_fake_op("flashinfer::fp4_quantize_sm100")
    def _fake_fp4_quantize_sm100(
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m, k = input.shape
        return (
            input.new_empty([m, k // 2], dtype=torch.int64),  # FLOAT4_E2M1X2
            input.new_empty([m * k // sf_vec_size], dtype=torch.int32),  # Scale factors
        )

    @register_custom_op(
        "flashinfer::mxfp4_dequantize_host",
        mutates_args=(""),
    )
    def mxfp4_dequantize_host(
        weight: torch.Tensor,
        scale: torch.Tensor,
        group_size: int = 32,
    ) -> torch.Tensor:
        out = torch.empty(
            (weight.shape[0], weight.shape[1] * 2),
            dtype=torch.float32,
            device=weight.device,
        )
        module.mxfp4_dequantize_host(
            weight,
            scale,
            out,
            group_size,
        )
        return out

    @register_fake_op("flashinfer::mxfp4_dequantize_host")
    def _fake_mxfp4_dequantize_host(
        weight: torch.Tensor,
        scale: torch.Tensor,
        group_size: int = 32,
    ) -> torch.Tensor:
        return weight.new_empty(
            [weight.shape[0], weight.shape[1] * 2], dtype=torch.float32
        )

    @register_custom_op(
        "flashinfer::block_scale_interleave_sm100",
        mutates_args=("",),
    )
    def block_scale_interleave_sm100(
        unswizzled_sf: torch.Tensor,
    ) -> torch.Tensor:
        """Swizzle block scale tensor for FP4 format.

        Args:
            unswizzled_sf (torch.Tensor): unswizzled block scale tensor with dtype uint8 or bfloat16.

        Returns:
            torch.Tensor: output tensor for swizzled block scale with dtype uint8 or bfloat16.
        """
        num_experts = unswizzled_sf.shape[0] if unswizzled_sf.dim() == 3 else 1
        expert_out_size = _compute_swizzled_layout_sf_size(
            unswizzled_sf.shape[-2], unswizzled_sf.shape[-1], 128
        )
        out = torch.empty(
            (num_experts * expert_out_size,),
            dtype=unswizzled_sf.dtype,
            device=unswizzled_sf.device,
        )
        module.block_scale_interleave_sm100(unswizzled_sf, out)
        return out

    @register_fake_op("flashinfer::block_scale_interleave_sm100")
    def _fake_block_scale_interleave_sm100(
        unswizzled_sf: torch.Tensor,
    ) -> torch.Tensor:
        return unswizzled_sf.new_empty(
            [unswizzled_sf.shape[0] * unswizzled_sf.shape[1] // 16], dtype=torch.uint8
        )

    @register_custom_op(
        "flashinfer::fp4_batched_quantize_sm100",
        mutates_args=("",),
    )
    def fp4_batched_quantize_sm100(
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a batched tensor to FP4 (E2M1x2) with per-block scale factors.

        This function converts a float/bfloat16 (or FP8-quantized) input tensor into a
        packed FP4 tensor using the E2M1 format (two 4-bit values per byte), along with
        per-block scale factors. Scale factors are encoded as UE4M3 by default, or UE8M0
        when requested, and an optional global scale can be applied.

        Args:
            input (torch.Tensor): Input tensor of shape [B, M, K] with dtype torch.float16,
                torch.bfloat16, or an FP8-quantized dtype supported by the kernel.
            global_scale (torch.Tensor, optional): Global scale factor of shape [1] and
                dtype float32.
            sf_vec_size (int, optional): Scale-factor vector size and alignment unit along K.
                Supported/expected values:
                - 16 (NVFP4 path; supported)
                - 32 (MXFP4 path; not supported yet)
                Defaults to 16.
            sf_use_ue8m0 (bool, optional): Scale-factor encoding type.
                False → UE4M3 (default), True → UE8M0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - self_fp4 (torch.Tensor): Packed FP4 tensor in E2M1x2 format of shape
                [B, M, K // 2] with dtype torch.uint8 (two FP4 lanes per byte).
                - self_block_scale_factors (torch.Tensor): Block scale factors with dtype
                uint8 (UE4M3 or UE8M0), laid out as a flat buffer of shape
                [B, ceil(M / 128) * 128 * ceil(K / sf_vec_size / 4) * 4].

        Notes:
            - K must be even (because outputs pack two FP4 values per byte).
            - For best performance, K should be a multiple of sf_vec_size; the scale-factor
            buffer is aligned to sf_vec_size along K, pads M to multiples of 128, and
            rounds (K / sf_vec_size) up to a multiple of 4 for storage.
            - The batch dimension B is preserved for both outputs.
        """
        b, m, k = input.shape
        out_val = torch.empty(
            (b, m, k // 2),
            dtype=torch.uint8,
            device=input.device,
        )
        out_sf = torch.empty(
            (b, _compute_swizzled_layout_sf_size(m, k // sf_vec_size, 128)),
            dtype=torch.uint8,
            device=input.device,
        )
        module.fp4_batched_quantize(
            input,
            global_scale,
            out_val,
            out_sf,
            sf_vec_size,
            sf_use_ue8m0,
        )
        return out_val, out_sf

    @register_fake_op("flashinfer::fp4_batched_quantize_sm100")
    def _fake_fp4_batched_quantize_sm100(
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, m, k = input.shape
        return (
            input.new_empty([b, m, k // 2], dtype=torch.uint8),  # FLOAT4_E2M1X2
            input.new_empty(
                [b, _compute_swizzled_layout_sf_size(m, k // sf_vec_size, 128)],
                dtype=torch.uint8,
            ),  # swizzled SF buffer
        )

    @register_custom_op(
        "flashinfer::silu_and_mul_scaled_nvfp4_experts_quantize_sm100",
        mutates_args=("",),
    )
    def silu_and_mul_scaled_nvfp4_experts_quantize_sm100(
        input: torch.Tensor,
        mask: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a silu and matmul with masked batched tensor to FP4 (E2M1x2) with per-block scale factors.

        This function first does silu and matmul to a float/bfloat16 input tensor then convect the result
        into a packed FP4 tensor using the E2M1 format (two 4-bit values per byte), along with
        per-block scale factors. Scale factors are encoded as UE4M3 by default, or UE8M0
        when requested, and an optional global scale can be applied.

        Args:
            input (torch.Tensor): Input tensor of shape [B, M, K] with dtype torch.float16,
                torch.bfloat16, or an FP8-quantized dtype supported by the kernel.
            mask (torch.Tensor): mask tensor of shape [B] with dtype torch.int32.
            global_scale (torch.Tensor, optional): Global scale factor of shape [1] and
                dtype float32.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - self_fp4 (torch.Tensor): Packed FP4 tensor in E2M1x2 format of shape
                [B, M, K // 2] with dtype torch.uint8 (two FP4 lanes per byte).
                - self_block_scale_factors (torch.Tensor): Block scale factors with dtype
                uint8 (UE4M3 or UE8M0), laid out as a flat buffer of shape
                [B, ceil(M / 128) * 128 * ceil(K / sf_vec_size / 4) * 4].

        Notes:
            - K must be even (because outputs pack two FP4 values per byte).
            - For best performance, K should be a multiple of sf_vec_size; the scale-factor
            buffer is aligned to sf_vec_size along K, pads M to multiples of 128, and
            rounds (K / sf_vec_size) up to a multiple of 4 for storage.
            - The batch dimension B is preserved for both outputs.
        """
        device = input.device
        l, m, k_by_2 = input.shape
        k = k_by_2 // 2
        sf_vec_size = 16
        assert k % sf_vec_size == 0, f"k must be multiple of 16, but got {k}."

        scale_k = k // sf_vec_size
        padded_k = round_up(scale_k, 4)
        padded_k_int32 = padded_k // 4
        padded_m = round_up(m, 128)
        output = torch.empty(l, m, k // 2, device=device, dtype=torch.uint8)
        output_scales = torch.empty(
            l, padded_m, padded_k_int32, device=device, dtype=torch.int32
        )

        module.silu_and_mul_scaled_nvfp4_experts_quantize(
            output.view(l * m, k // 2),
            output_scales.view(l * padded_m, padded_k_int32),
            input.view(l * m, k_by_2),
            global_scale,
            mask,
            True,
        )
        output = output.permute(1, 2, 0)
        output_scales = output_scales.view(torch.float8_e4m3fn).view(
            l, padded_m // 128, padded_k // 4, 32, 4, 4
        )
        output_scales = output_scales.permute(3, 4, 1, 5, 2, 0)
        return output, output_scales

    @register_fake_op("flashinfer::silu_and_mul_scaled_nvfp4_experts_quantize_sm100")
    def _fake_silu_and_mul_scaled_nvfp4_experts_quantize_sm100(
        input: torch.Tensor,
        mask: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = input.device
        l, m, k_by_2 = input.shape
        k = k_by_2 // 2
        sf_vec_size = 16
        assert k % sf_vec_size == 0, f"k must be multiple of 16, but got {k}."

        scale_k = k // sf_vec_size
        padded_k = round_up(scale_k, 4)
        padded_k_int32 = padded_k // 4
        padded_m = round_up(m, 128)
        output = torch.empty(l, m, k // 2, device=device, dtype=torch.uint8)
        output_scales = torch.empty(
            l, padded_m, padded_k_int32, device=device, dtype=torch.int32
        )

        output_scales = output_scales.view(torch.float8_e4m3fn).view(
            l, padded_m // 128, padded_k // 4, 32, 4, 4
        )
        output_scales = output_scales.permute(3, 4, 1, 5, 2, 0)
        return (output, output_scales)

    @register_custom_op(
        "flashinfer::scaled_fp4_grouped_quant_sm100",
        mutates_args=("",),
    )
    def scaled_fp4_grouped_quant_sm100(
        input_tensor: torch.Tensor,
        input_global_scale: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize input tensor to FP4 and return quantized tensor and scale, for
        grouped gemm inputs (e.g., grouped_gemm_nt_masked for flashinfer).
        Args:
            input: The input tensor to be quantized to FP4, with shape (l, m, k)
                l is number of groups, m is number of tokens per group, k is number of features.
            input_global_scale: A scalar scaling factor for the entire tensor, with
                shape (l,).
        Outputs:
            output: The quantized tensor in FP4, with shape (m, k // 2, l) but the physical
                layout is (l, m, k // 2). `// 2` is because two fp4 values are packed into
                an uint8.
            output_scales: The blockscale tensor in FP8-E4M3, with shape (32, 4, rm, 4, rk, l)
                but the physical layout is (l, rm, rk, 32, 4, 4).
        Note:
            For the shape of output_scales, `32 * 4 * rm` is a padded m to nearest multiple of 128.
            `4 * rk` is a padded `k // 16` to nearest multiple of 4. These layout constants are
            required by the NVIDIA Blackwell MMA operations.
        """
        device = input_tensor.device
        l, m, k = input_tensor.shape
        sf_vec_size = 16
        assert k % sf_vec_size == 0, f"k must be multiple of 16, but got {k}."

        scale_k = k // sf_vec_size
        padded_k = round_up(scale_k, 4)
        padded_k_int32 = padded_k // 4
        padded_m = round_up(m, 128)
        output = torch.empty(l, m, k // 2, device=device, dtype=torch.uint8)
        output_scales = torch.empty(
            l, padded_m, padded_k_int32, device=device, dtype=torch.int32
        )

        module.silu_and_mul_scaled_nvfp4_experts_quantize(
            output.view(l * m, k // 2),
            output_scales.view(l * padded_m, padded_k_int32),
            input_tensor.view(l * m, k),
            input_global_scale,
            mask,
            False,
        )
        # The physical layout of the output is (l, m, k // 2), but we want to return a
        # logical layout (m, k // 2, l) required by the flashinfer masked group gemm.
        output = output.permute(1, 2, 0)
        # The physical layout of the output scales is already swizzled as (l, rm, rk, 32, 4, 4), a
        # requirement for the flashinfer masked group gemm, where rm=m/128 and rk=k/4. The logic
        # layout is (32, 4, rm, 4, rk, l).
        output_scales = output_scales.view(torch.float8_e4m3fn).view(
            l, padded_m // 128, padded_k // 4, 32, 4, 4
        )
        output_scales = output_scales.permute(3, 4, 1, 5, 2, 0)
        return output, output_scales

    @register_fake_op("flashinfer::scaled_fp4_grouped_quant_sm100")
    def _fake_scaled_fp4_grouped_quant_sm100(
        input_tensor: torch.Tensor,
        input_global_scale: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = input_tensor.device
        l, m, k = input_tensor.shape
        sf_vec_size = 16
        assert k % sf_vec_size == 0, f"k must be multiple of 16, but got {k}."

        scale_k = k // sf_vec_size
        padded_k = round_up(scale_k, 4)
        padded_k_int32 = padded_k // 4
        padded_m = round_up(m, 128)
        output = torch.empty(l, m, k // 2, device=device, dtype=torch.uint8)
        output_scales = torch.empty(
            l, padded_m, padded_k_int32, device=device, dtype=torch.int32
        )

        output = output.permute(1, 2, 0)
        output_scales = output_scales.view(torch.float8_e4m3fn).view(
            l, padded_m // 128, padded_k // 4, 32, 4, 4
        )
        output_scales = output_scales.permute(3, 4, 1, 5, 2, 0)
        return output, output_scales

    @register_custom_op(
        "flashinfer::e2m1_and_ufp8sf_scale_to_float_sm100",
        mutates_args=(""),
    )
    def e2m1_and_ufp8sf_scale_to_float_sm100(
        e2m1_tensor: torch.Tensor,
        ufp8_scale_tensor: torch.Tensor,
        global_scale_tensor: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        ufp8_type: int = 1,
        is_sf_swizzled_layout: bool = True,
    ) -> torch.Tensor:
        """Convert E2M1 format tensor and UFP8 scale factors to float tensor.

        This function performs dequantization by converting a packed FP4 tensor in E2M1 format
        back to float values using the associated UFP8 scale factors and global scale.

        Args:
            e2m1_tensor (torch.Tensor): Packed FP4 tensor in E2M1 format of shape [M, K/2] with dtype uint8.
            ufp8_scale_tensor (torch.Tensor): Scale factors tensor in UFP8 format with dtype uint8.
            global_scale_tensor (torch.Tensor, optional): Global scale factor of shape [1] and dtype float32.
            sf_vec_size (int, optional): Scale factor vector size. Defaults to 16.
            ufp8_type (int, optional): UFP8 scale factor type (0 for UE8M0, 1 for E4M3). Defaults to 1.
            is_sf_swizzled_layout (bool, optional): Whether scale factors use swizzled layout. Defaults to True.

        Returns:
            torch.Tensor: Dequantized float tensor of shape [M, K] with dtype float32.
        """
        out = torch.zeros(
            (e2m1_tensor.shape[0], e2m1_tensor.shape[1] * 2),
            dtype=torch.float32,
            device="cpu",
        )
        module.e2m1_and_ufp8sf_scale_to_float_sm100(
            e2m1_tensor.cpu(),
            ufp8_scale_tensor.cpu().reshape(-1),
            global_scale_tensor.cpu(),
            out,
            sf_vec_size,
            ufp8_type,
            is_sf_swizzled_layout,
        )
        return out

    @register_fake_op("flashinfer::e2m1_and_ufp8sf_scale_to_float_sm100")
    def _fake_e2m1_and_ufp8sf_scale_to_float_sm100(
        e2m1_tensor: torch.Tensor,
        ufp8_scale_tensor: torch.Tensor,
        global_scale_tensor: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        ufp8_type: int = 1,
        is_sf_swizzled_layout: bool = True,
    ) -> torch.Tensor:
        return e2m1_tensor.new_empty(
            [e2m1_tensor.shape[0], e2m1_tensor.shape[1] * 2], dtype=torch.float32
        )

    # Register the module
    return SimpleNamespace(
        fp4_quantize_sm100=fp4_quantize_sm100,
        block_scale_interleave_sm100=block_scale_interleave_sm100,
        e2m1_and_ufp8sf_scale_to_float_sm100=e2m1_and_ufp8sf_scale_to_float_sm100,
        mxfp4_dequantize_host=mxfp4_dequantize_host,
        fp4_batched_quantize_sm100=fp4_batched_quantize_sm100,
        silu_and_mul_scaled_nvfp4_experts_quantize_sm100=silu_and_mul_scaled_nvfp4_experts_quantize_sm100,
        scaled_fp4_grouped_quant_sm100=scaled_fp4_grouped_quant_sm100,
    )

def fp4_quantize_sm100(
    input: torch.Tensor,
    global_scale: Optional[torch.Tensor] = None,
    sf_vec_size: int = 16,
    sf_use_ue8m0: bool = False,
    is_sf_swizzled_layout: bool = True,
    is_sf_8x4_layout: bool = False,
    enable_pdl: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize input tensor to FP4 format.

    Args:
        input (torch.Tensor): Input tensor of shape [M, K] with dtype fp16/bf16/fp8_quantized.
        global_scale (torch.Tensor, optional): Global scale factor of shape [1] and dtype float32.
        sf_vec_size (int, optional): Scale factor vector size. Defaults to 16.
        sf_use_ue8m0 (bool, optional): Whether to use UE8M0 format for scale factors. Defaults to False.
        is_sf_swizzled_layout (bool, optional): Whether to use swizzled layout for scale factors. Defaults to True.
        is_sf_8x4_layout (bool, optional): Whether to use 8x4 layout or 128x4 layout for scale factors. Defaults to False.
        enable_pdl (Optional[bool], optional): Whether to enable PDL (Programmatic Dependent Launch).
            If None, automatically detects based on device capability. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Quantized tensor of shape [M, K/2] with dtype FLOAT4_E2M1X2
            - Scale factors tensor with shape determined by layout and sf_vec_size
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    out_val = torch.empty(
        (*input.shape[:-1], input.shape[-1] // 2),
        dtype=torch.uint8,
        device=input.device,
    )
    m = input.numel() // input.shape[-1]
    k = input.shape[-1]
    if is_sf_swizzled_layout:
        out_sf_size = _compute_swizzled_layout_sf_size(
            m, k // sf_vec_size, 8 if is_sf_8x4_layout else 128
        )
        out_sf_size_padded = out_sf_size
    else:
        out_sf_size = m * k // sf_vec_size
        out_sf_size_padded = round_up(m, 16) * k // sf_vec_size
    out_sf = torch.empty(
        (out_sf_size_padded,), dtype=torch.uint8, device=input.device
    )
    module.fp4_quantize(
        input,
        global_scale,
        out_val,
        out_sf,
        sf_vec_size,
        sf_use_ue8m0,
        is_sf_swizzled_layout,
        is_sf_8x4_layout,
        enable_pdl,
    )
    return out_val, out_sf[:out_sf_size]

def generate_m_grouped_contiguous_fp4(num_groups: int, expected_m_per_group: int, n: int, k: int,
                                      use_ue8m0: bool = False, gran_k: int = 16,
                                      return_original: bool = False):
    actual_ms = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
    aligned_ms = [align(actual_m, get_mk_alignment_for_contiguous_layout()) for actual_m in actual_ms]
    m = sum(aligned_ms)
    import ipdb; ipdb.set_trace()
    
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    m_indices = torch.empty(m, device='cuda', dtype=torch.int32)
    d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_d = torch.randn((m, n), device='cuda', dtype=torch.bfloat16)

    start = 0
    for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end = start + actual_m
        aligned_end = start + aligned_m
        m_indices[start: actual_end] = i
        m_indices[actual_end: aligned_end] = -1
        a[actual_end: aligned_end] = 0
        ref_d[start: aligned_end] = a[start: aligned_end] @ b[i].t()
        start = aligned_end

    # A: per-token FP4 quantization -> (m, k//2) uint8, SF (m, ceil(k/gran_k)) float
    a_fp4 = per_token_cast_to_fp4(a, use_ue8m0=use_ue8m0, gran_k=gran_k)
    # B: per-block FP4 quantization -> (G, n, k//2) uint8, SF (G, ceil(n/gran_k), ceil(k/gran_k)) float
    b_fp4_data = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.uint8)
    b_fp4_sf = torch.empty((num_groups, ceil_div(n, gran_k), ceil_div(k, gran_k)), device='cuda', dtype=torch.float)
    for i in range(num_groups):
        b_fp4_data[i], b_fp4_sf[i] = per_block_cast_to_fp4(b[i], use_ue8m0=use_ue8m0, gran_k=gran_k)
    b_fp4 = (b_fp4_data, b_fp4_sf)

    if return_original:
        return m, a_fp4, b_fp4, m_indices, d, ref_d, a, b
    return m, a_fp4, b_fp4, m_indices, d, ref_d


def generate_m_grouped_masked_fp4(num_groups: int, max_m: int, expected_m_per_group: int, n: int, k: int,
                                  use_ue8m0: bool = False, gran_k: int = 16):
    a = torch.randn((num_groups, max_m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    d = torch.empty((num_groups, max_m, n), device='cuda', dtype=torch.bfloat16)
    ref_d = torch.einsum('gmk,gnk->gmn', a, b)

    masked_m = torch.empty((num_groups, ), device='cuda', dtype=torch.int)
    for j in range(num_groups):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
    assert masked_m.amax().item() <= max_m

    # A: per-token FP4, grouped -> (G, max_m, k//2) uint8, SF (G, max_m, ceil(k/gran_k)) float
    a_fp4_data = torch.empty((num_groups, max_m, k // 2), device='cuda', dtype=torch.uint8)
    a_fp4_sf = torch.empty((num_groups, max_m, ceil_div(k, gran_k)), device='cuda', dtype=torch.float)
    for i in range(num_groups):
        a_fp4_data[i], a_fp4_sf[i] = per_token_cast_to_fp4(a[i], use_ue8m0=use_ue8m0, gran_k=gran_k)
    a_fp4 = (a_fp4_data, a_fp4_sf)

    # B: per-block FP4, grouped -> (G, n, k//2) uint8, SF (G, ceil(n/gran_k), ceil(k/gran_k)) float
    b_fp4_data = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.uint8)
    b_fp4_sf = torch.empty((num_groups, ceil_div(n, gran_k), ceil_div(k, gran_k)), device='cuda', dtype=torch.float)
    for i in range(num_groups):
        b_fp4_data[i], b_fp4_sf[i] = per_block_cast_to_fp4(b[i], use_ue8m0=use_ue8m0, gran_k=gran_k)
    b_fp4 = (b_fp4_data, b_fp4_sf)

    return a_fp4, b_fp4, masked_m, d, ref_d