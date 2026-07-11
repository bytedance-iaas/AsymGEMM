# Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
# Modified by Bytedance Inc., 2026.
# Original: https://github.com/deepseek-ai/DeepGEMM

from . import math
from .math import *

try:
    from . import layout
    from .layout import *
except ImportError:
    layout = None

    def _missing_layout_binding(*args, **kwargs):
        raise RuntimeError("asym_gemm layout CUDA bindings are unavailable in this build")

    get_tma_aligned_size = _missing_layout_binding
    get_mk_alignment_for_contiguous_layout = _missing_layout_binding
    get_m_alignment_for_contiguous_layout = _missing_layout_binding
    get_k_alignment_for_contiguous_layout = _missing_layout_binding
    get_mn_major_tma_aligned_tensor = _missing_layout_binding
    get_mn_major_tma_aligned_packed_ue8m0_tensor = _missing_layout_binding
    get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor = _missing_layout_binding
