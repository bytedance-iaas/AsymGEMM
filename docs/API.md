# API Reference

AsymGEMM provides two levels of API:

- **Python API** — PyTorch extension bindings for direct use in Python. This is the primary interface for most users.
- **C++ API** — Header-only C++ interface (`csrc/apis/gemm.hpp`) for integration into custom CUDA/C++ applications without Python overhead.

Both layers expose the same set of kernels with equivalent semantics. This document covers the Python API only — C++ users can refer to the header files directly, as the function signatures and parameter semantics are identical.

---

## Python API

All kernels are exposed as Python functions via PyTorch extension bindings.

### SM100 Asymmetric GEMM (Blackwell)

These kernels require SM100 (GB200) with TMA support. Input tensors `a` and `b` are passed as `(data, scale_factor)` tuples for FP8 and FP4 variants.

#### `asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(a, b, d, offsets, experts, list_size, compiled_dims="nk")`

BF16 asymmetric grouped GEMM with contiguous layout. Weights reside in CPU DRAM.

| Parameter | Shape | Dtype | Location | Description |
|-----------|-------|-------|----------|-------------|
| `a` | `[M, K]` | bfloat16 | GPU | Activations |
| `b` | `[num_groups, N, K]` | bfloat16 | CPU | Expert weights |
| `d` | `[M, N]` | bfloat16 | GPU | Output |
| `offsets` | `[num_groups * 2]` | int32 | GPU | Start/end boundary indices |
| `experts` | `[num_groups + 1]` | int32 | GPU | Expert IDs (-1 terminated) |
| `list_size` | `[1]` | int32 | GPU | Scalar tensor: number of entries in offsets |
| `compiled_dims` | string | — | — | Tile config hint (default `"nk"`) |

#### `asym_gemm.m_grouped_bf16_asym_gemm_nt_masked(a, b, d, offsets, experts, list_size, expected_m, compiled_dims="nk")`

BF16 asymmetric grouped GEMM with masked layout (padded `[G, M_max, K]`).

| Parameter | Shape | Dtype | Location | Description |
|-----------|-------|-------|----------|-------------|
| `a` | `[num_groups, M_max, K]` | bfloat16 | GPU | Padded activations |
| `b` | `[num_groups, N, K]` | bfloat16 | CPU | Expert weights |
| `d` | `[num_groups, M_max, N]` | bfloat16 | GPU | Padded output |
| `offsets` | `[num_groups * 2]` | int32 | GPU | Start/end boundary indices |
| `experts` | `[num_groups + 1]` | int32 | GPU | Expert IDs (-1 terminated) |
| `list_size` | `[1]` | int32 | GPU | Scalar tensor |
| `expected_m` | scalar | int | — | Expected rows per expert (grid sizing hint) |

#### `asym_gemm.m_grouped_fp8_asym_gemm_nt_contiguous(a, b, d, offsets, experts, list_size, recipe=None, compiled_dims="nk", disable_ue8m0_cast=False)`

FP8 asymmetric grouped GEMM with contiguous layout. `a` and `b` are `(data_tensor, scale_factor_tensor)` tuples.

| Parameter | Shape | Dtype | Location | Description |
|-----------|-------|-------|----------|-------------|
| `a` | tuple: `([M, K], scale)` | float8_e4m3fn | GPU | Activations + scale factors |
| `b` | tuple: `([num_groups, N, K], scale)` | float8_e4m3fn | CPU | Expert weights + scale factors |
| `d` | `[M, N]` | bfloat16 | GPU | Output |
| `offsets` | `[num_groups * 2]` | int32 | GPU | Start/end boundary indices |
| `experts` | `[num_groups + 1]` | int32 | GPU | Expert IDs (-1 terminated) |
| `list_size` | `[1]` | int32 | GPU | Scalar tensor |
| `recipe` | tuple | — | — | Optional `(gran_mn_a, gran_mn_b, gran_k)` |

#### `asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(a, b, d, masked_m, expected_m, recipe=None, compiled_dims="nk", disable_ue8m0_cast=False)`

FP8 asymmetric grouped GEMM with masked layout.

#### `asym_gemm.m_grouped_fp4_asym_gemm_nt_contiguous(a, b, d, offsets, experts, list_size, recipe=None, compiled_dims="nk", disable_ue8m0_cast=False)`

NVFP4 (E2M1) asymmetric grouped GEMM with contiguous layout. `a` and `b` are `(packed_data, scale_factor)` tuples.

| Parameter | Shape | Dtype | Location | Description |
|-----------|-------|-------|----------|-------------|
| `a` | tuple: `([M, K//2], scale)` | uint8 | GPU | Packed FP4 activations (2 per byte) + scale |
| `b` | tuple: `([num_groups, N, K//2], scale)` | uint8 | CPU | Packed FP4 weights + scale |
| `d` | `[M, N]` | bfloat16 | GPU | Output |
| `offsets` | `[num_groups * 2]` | int32 | GPU | Start/end boundary indices |
| `experts` | `[num_groups + 1]` | int32 | GPU | Expert IDs (-1 terminated) |
| `list_size` | `[1]` | int32 | GPU | Scalar tensor |

#### `asym_gemm.m_grouped_fp4_asym_gemm_nt_masked(a, b, d, masked_m, expected_m, recipe=None, compiled_dims="nk", disable_ue8m0_cast=False)`

NVFP4 asymmetric grouped GEMM with masked layout.

---

### SM89 FP8 MoE GEMM (Ada Lovelace)

These kernels use native SM89 FP8 MMA with per-tensor scalar scales. Weights may reside in CPU-pinned memory or GPU HBM.

#### `asym_gemm.m_grouped_fp8_asym_gemm_sm80(a, b, d, offsets, experts, list_size, scale_a=1.0, scale_b=1.0)`

FP8 MoE GEMM with contiguous layout and per-tensor scalar scales.

| Parameter | Shape | Dtype | Location | Description |
|-----------|-------|-------|----------|-------------|
| `a` | `[M, K]` | float8_e4m3fn | GPU | Activations |
| `b` | `[num_experts, N, K]` | float8_e4m3fn | CPU pinned or GPU | Expert weights |
| `d` | `[M, N]` | bfloat16 | GPU | Output |
| `offsets` | `[list_size]` | int32 | GPU | Cumulative end-token indices |
| `experts` | `[list_size]` | int32 | GPU | Expert IDs |
| `list_size` | scalar | int | — | Number of active experts |
| `scale_a` | scalar | float | — | Per-tensor activation scale (default 1.0) |
| `scale_b` | scalar | float | — | Per-tensor weight scale (default 1.0) |

**Alignment:** `K % 32 == 0`, `N % 32 == 0`, `K >= 32`.

#### `asym_gemm.m_grouped_fp8_asym_gemm_sm80_masked(a, b, d, masked_m, expected_m, scale_a=1.0, scale_b=1.0)`

FP8 MoE GEMM with masked (padded) layout.

| Parameter | Shape | Dtype | Location | Description |
|-----------|-------|-------|----------|-------------|
| `a` | `[num_groups, M_max, K]` | float8_e4m3fn | GPU | Padded activations |
| `b` | `[num_groups, N, K]` | float8_e4m3fn | CPU pinned or GPU | Expert weights |
| `d` | `[num_groups, M_max, N]` | bfloat16 | GPU | Output |
| `masked_m` | `[num_groups]` | int32 | GPU | Active row count per group |
| `expected_m` | scalar | int | — | Expected rows per group (grid sizing) |
| `scale_a` | scalar | float | — | Per-tensor activation scale |
| `scale_b` | scalar | float | — | Per-tensor weight scale |

---

### SM80 MoE GEMM (Ampere and above)

#### `asym_gemm.m_grouped_moe_gemm_nt_contiguous(a, b, d, offsets, experts, list_size, compiled_dims="nk")`

BF16/FP16 MoE grouped GEMM. All tensors must be on GPU. Dtype resolved at runtime from input tensors.

| Parameter | Shape | Dtype | Location | Description |
|-----------|-------|-------|----------|-------------|
| `a` | `[M, K]` | float16 or bfloat16 | GPU | Activations |
| `b` | `[num_experts, N, K]` | same as `a` | GPU | Expert weights |
| `d` | `[M, N]` | same as `a` | GPU | Output |
| `offsets` | `[list_size]` | int32 | GPU | Cumulative end-token indices |
| `experts` | `[list_size]` | int32 | GPU | Expert IDs |
| `list_size` | scalar | int | — | Number of active experts |

**Alignment:** `K % 16 == 0`, `N % 32 == 0`, `K >= 64`.

---

## Utility Functions

#### `asym_gemm.transform_sf_into_required_layout(scales, ...)`

Transform scale factor tensors into TMA-aligned layout required by SM100 kernels.

#### `asym_gemm.fp8_einsum(equation, a, b, scale_a, scale_b)`

FP8 tensor contraction (einsum-style interface).

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_ENABLE_JIT_ASYMGEMM` | `True` | Enable AsymGEMM JIT compilation in SGLang |
| `SGLANG_JIT_ASYMGEMM_PRECOMPILE` | `True` | Pre-compile kernel variants at startup |
| `SGLANG_JIT_ASYMGEMM_FAST_WARMUP` | `False` | Faster but less thorough precompile |
| `SGLANG_AG_CACHE_DIR` | `~/.asym_gemm/cache/` | JIT kernel cache directory |
| `SGLANG_MASKED_GEMM_CHUNK_SIZE` | `0` | Expert group chunk size (0 = no chunking) |
| `SGLANG_MASKED_GEMM_FAST_ACT` | `False` | Fused SiLU + quantization path |
| `SGLANG_ASYMGEMM_SANITY_CHECK` | `False` | Enable input validation checks |
| `SGLANG_ENABLE_SM89_ASYMGEMM` | `False` | Force SM89 path on non-Ada GPUs (testing) |

---
