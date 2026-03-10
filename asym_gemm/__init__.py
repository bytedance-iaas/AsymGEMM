import os
import subprocess
import torch
from torch.version import cuda as cuda_version
from packaging import version

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# CUDA extension and kernels (only available when built with CUDA)
try:
    from . import _C
    from ._C import (
        set_num_sms,
        get_num_sms,
        set_tc_util,
        get_tc_util,
        set_compile_mode,
        get_compile_mode,
    )

    if version.parse(cuda_version) >= version.parse('12.1'):
        def _maybe_import_from_C(names):
            for name in names:
                if hasattr(_C, name):
                    globals()[name] = getattr(_C, name)

        def _missing_kernel(kernel_name):
            def _raise_missing(*args, **kwargs):
                raise RuntimeError(
                    f"`{kernel_name}` is not available in this build of asym_gemm. "
                    "Rebuild with matching CUDA/architecture flags to enable this kernel."
                )
            return _raise_missing

        def _export_kernel_alias(alias_name, target_name):
            globals()[alias_name] = globals().get(target_name, _missing_kernel(target_name))

        # DeepGEMM Kernels (may vary by build flags / arch)
        _maybe_import_from_C([
            # FP8 GEMMs
            "m_grouped_fp8_asym_gemm_nt_masked",
            "m_grouped_fp8_asym_gemm_nt_contiguous",
            # BF16 GEMMs
            "m_grouped_bf16_asym_gemm_nt_contiguous",
            "m_grouped_bf16_gemm_nt_contiguous",
            "m_grouped_bf16_asym_gemm_nt_masked",
            # Einsum kernels
            "einsum",
            "fp8_einsum",
            # Attention kernels
            "fp8_mqa_logits",
            "get_paged_mqa_logits_metadata",
            "fp8_paged_mqa_logits",
            # Layout kernels
            "transform_sf_into_required_layout",
            "get_mk_alignment_for_contiguous_layout",
        ])

        # Some alias for legacy supports
        # TODO: remove these later
        _export_kernel_alias("fp8_m_grouped_asym_gemm_nt_masked", "m_grouped_fp8_asym_gemm_nt_masked")
        _export_kernel_alias("fp8_m_grouped_gemm_nt_masked", "m_grouped_fp8_asym_gemm_nt_masked")
        _export_kernel_alias("bf16_m_grouped_asym_gemm_nt_masked", "m_grouped_bf16_asym_gemm_nt_masked")
        _export_kernel_alias("bf16_m_grouped_gemm_nt_masked", "m_grouped_bf16_asym_gemm_nt_masked")

        # Single matrix multiplication wrappers (no MoE grouping)
        def bf16_asym_gemm_nt(a: torch.Tensor, b: torch.Tensor, d: torch.Tensor,
                               compiled_dims: str = "nk") -> None:
            """Single BF16 GEMM: D[M, N] = A[M, K] @ B[N, K].T

            Wraps the grouped GEMM with a single group, so no offsets/experts
            need to be managed by the caller.

            Args:
                a: [M, K] BF16 tensor (K-major)
                b: [N, K] BF16 tensor (K-major)
                d: [M, N] BF16 tensor (N-major, output)
                compiled_dims: dimension compilation string (default "nk")
            """
            m = a.shape[0]
            device = a.device
            offsets = torch.tensor([0, m], dtype=torch.int32, device=device)
            experts = torch.tensor([0, -1], dtype=torch.int32, device=device)
            m_grouped_bf16_asym_gemm_nt_contiguous(
                a, b.unsqueeze(0), d, offsets, experts, 2, compiled_dims
            )

        def fp8_asym_gemm_nt(a, b, d: torch.Tensor,
                              recipe=None, compiled_dims: str = "nk",
                              disable_ue8m0_cast: bool = False) -> None:
            """Single FP8 GEMM: D[M, N] = A[M, K] @ B[N, K].T

            Wraps the grouped FP8 GEMM with a single group, so no offsets/experts
            need to be managed by the caller.

            Args:
                a: (data [M, K], scale_factors) FP8 tensor pair
                b: (data [N, K], scale_factors) FP8 tensor pair
                d: [M, N] BF16 output tensor
                recipe: optional (gran_mn_a, gran_mn_b, gran_k) tuple
                compiled_dims: dimension compilation string (default "nk")
                disable_ue8m0_cast: disable UE8M0 cast (for SM90 compatibility)
            """
            m = a[0].shape[0]
            device = a[0].device
            offsets = torch.tensor([0, m], dtype=torch.int32, device=device)
            experts = torch.tensor([0, -1], dtype=torch.int32, device=device)
            b_grouped = (b[0].unsqueeze(0), b[1].unsqueeze(0))
            m_grouped_fp8_asym_gemm_nt_contiguous(
                a, b_grouped, d, offsets, experts, 2, recipe, compiled_dims, disable_ue8m0_cast
            )

    # Initialize CPP modules
    def _find_cuda_home() -> str:
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        if cuda_home is None:
            try:
                with open(os.devnull, 'w') as devnull:
                    nvcc = subprocess.check_output(['which', 'nvcc'], stderr=devnull).decode().rstrip('\r\n')
                    cuda_home = os.path.dirname(os.path.dirname(nvcc))
            except Exception:
                cuda_home = '/usr/local/cuda'
                if not os.path.exists(cuda_home):
                    cuda_home = None
        assert cuda_home is not None
        return cuda_home

    _C.init(
        os.path.dirname(os.path.abspath(__file__)),
        _find_cuda_home()
    )
except ImportError:
    import warnings
    warnings.warn("CUDA extension (_C) not available. CUDA kernels will not be accessible.")

from importlib.metadata import version as _get_version
__version__ = _get_version('asym_gemm')
