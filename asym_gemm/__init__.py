# Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
# Modified by Bytedance Inc., 2026.
# Original: https://github.com/deepseek-ai/DeepGEMM

import os
import subprocess
from pkgutil import extend_path

import torch
from packaging import version
from torch.version import cuda as cuda_version

__path__ = extend_path(__path__, __name__)

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
            # FP4 GEMMs
            "m_grouped_fp4_asym_gemm_nt_contiguous",
            "m_grouped_fp4_asym_gemm_nt_masked",
            # BF16 GEMMs
            "m_grouped_bf16_asym_gemm_nt_contiguous",
            "m_grouped_bf16_asym_gemm_nt_masked",
            # SM80 MoE GEMM (FP16 + BF16, JIT)
            "m_grouped_moe_gemm_nt_contiguous",
            # SM89 FP8 MoE GEMM (native FP8 MMA, JIT)
            "m_grouped_fp8_asym_gemm_sm89",
            "m_grouped_fp8_asym_gemm_sm89_masked",
            # SM90 INT8 asym GEMM (native S8 WGMMA, JIT)
            "m_grouped_int8_asym_gemm_sm90_masked",
            "m_grouped_int8_asym_gemm_sm90_contiguous",
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

        # Architecture dispatch (facade layer). These unified entry points route
        # to the native-FP8 (SM89/SM90) or TMA/UMMA (SM100) kernels internally, so
        # callers issue one call regardless of GPU. They intentionally override the
        # raw ``_C`` imports of the same name pulled in above.
        from .dispatch import (  # noqa: F401
            get_arch_major,
            get_arch_pair,
            is_blackwell,
            is_dtype_supported,
            supported_archs,
            supported_dtypes,
            m_grouped_fp8_asym_gemm_nt_contiguous,
            m_grouped_fp8_asym_gemm_nt_masked,
            m_grouped_int8_asym_gemm_nt_contiguous,
            m_grouped_int8_asym_gemm_nt_masked,
        )
        globals()["m_grouped_fp8_asym_gemm_nt_contiguous"] = m_grouped_fp8_asym_gemm_nt_contiguous
        globals()["m_grouped_fp8_asym_gemm_nt_masked"] = m_grouped_fp8_asym_gemm_nt_masked
        globals()["m_grouped_int8_asym_gemm_nt_contiguous"] = m_grouped_int8_asym_gemm_nt_contiguous
        globals()["m_grouped_int8_asym_gemm_nt_masked"] = m_grouped_int8_asym_gemm_nt_masked

        # Some alias for legacy supports
        # TODO: remove these later
        _export_kernel_alias("fp8_m_grouped_asym_gemm_nt_masked", "m_grouped_fp8_asym_gemm_nt_masked")
        _export_kernel_alias("fp8_m_grouped_gemm_nt_masked", "m_grouped_fp8_asym_gemm_nt_masked")
        _export_kernel_alias("bf16_m_grouped_asym_gemm_nt_masked", "m_grouped_bf16_asym_gemm_nt_masked")
        _export_kernel_alias("bf16_m_grouped_gemm_nt_masked", "m_grouped_bf16_asym_gemm_nt_masked")

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

# Unified MoE sub-package (CPU AMX via _cpu_C + GPU INT8 via torch._int_mm).
# Independent of the CUDA extension above: a host without CUDA still gets the
# CPU bucket; a host without AMX gets a clear error at first use.
try:
    from . import _cpu_C            # noqa: F401 — register the CPU extension
    from . import unified_moe       # noqa: F401 — Layer + helpers
except ImportError as _e:
    import warnings
    warnings.warn(
        "asym_gemm CPU extension (_cpu_C) not available — unified_moe disabled. "
        f"Cause: {_e}"
    )
    unified_moe = None              # explicit sentinel for callers

from importlib.metadata import version as _get_version
__version__ = _get_version('asym_gemm')
