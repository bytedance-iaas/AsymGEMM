"""asym_gemm.unified_moe — unified INT8 MoE layer.

Dispatches per-expert work between the CPU AMX path (via asym_gemm._cpu_C,
which wraps the vendored cpu_gemm) and a GPU INT8 path (torch._int_mm,
CUTLASS-backed). See docs/unified_moe.md for the design.

Both backends compute the same INT8 → INT32 → FP32 dequant contract:

    sA[i]       = amax_row_i / 127
    sB[n]       = amax_col_n / 127
    C_int32     = A_int8 @ B_int8.T
    C_fp32      = sA · sB · C_int32       (outer broadcast)
"""

from .dispatch_model import DispatchModel
from .runtime import Layer
from .. import _cpu_C as _C  # re-exported for tests / debugging

__all__ = ["Layer", "DispatchModel", "_C"]
