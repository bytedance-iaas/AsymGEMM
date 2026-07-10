# API Reference

AsymGEMM provides two levels of API:

- **Python API** — PyTorch extension bindings for direct use in Python. This is the primary interface for most users.
- **C++ API** — Header-only C++ interface (`csrc/apis/gemm.hpp`) for integration into custom CUDA/C++ applications without Python overhead.

Both layers expose the same set of kernels with equivalent semantics. This document covers the Python API only — C++ users can refer to the header files directly, as the function signatures and parameter semantics are identical.

---

## Python API

All kernels are exposed as Python functions via PyTorch extension bindings.

### Architecture-Agnostic Asymmetric GEMM

The `m_grouped_{bf16,fp8,int8}_asym_gemm_nt_*` entry points detect the device's compute capability at call time and route to the matching kernel (SM89 / SM90 / SM100). Callers do not select an architecture explicitly.

| Entry point | SM89 (Ada) | SM90 (Hopper) | SM100 (Blackwell) |
|---|---|---|---|
| `m_grouped_bf16_asym_gemm_nt_contiguous` | — | ✅ | ✅ |
| `m_grouped_bf16_asym_gemm_nt_masked` | ✅ | ✅ | ✅ |
| `m_grouped_fp8_asym_gemm_nt_{contiguous,masked}` | ✅ | ✅ | ✅ |
| `m_grouped_int8_asym_gemm_nt_{contiguous,masked}` | — | ✅ | — |
| `m_grouped_fp4_asym_gemm_nt_{contiguous,masked}` | — | — | ✅ |

**Contiguous layout conventions** (shared by all contiguous variants):

- `offsets` — int32 GPU tensor of flat `(start, end)` row pairs: `offsets[2*i]` / `offsets[2*i+1]` bound the token rows of `experts[i]`.
- `experts` — int32 GPU tensor of expert IDs, terminated by a `-1` sentinel.
- `list_size` — number of entries in `experts` *including* the sentinel. Accepts a Python int or a 1-element int32 tensor.

**Masked layout conventions:** activations/outputs are padded to `[num_groups, M_max, *]`; `masked_m` (int32 GPU tensor, `[num_groups]`) holds the valid row count per group and `expected_m` is an int grid-sizing hint. `masked_m[g] == 0` marks an inactive expert.

#### `asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(a, b, d, offsets, experts, list_size, compiled_dims="nk")`

BF16 asymmetric grouped GEMM with contiguous layout. Weights reside in CPU DRAM.

| Parameter | Shape | Dtype | Location | Description |
|-----------|-------|-------|----------|-------------|
| `a` | `[M, K]` | bfloat16 | GPU | Activations |
| `b` | `[num_groups, N, K]` | bfloat16 | CPU pinned | Expert weights |
| `d` | `[M, N]` | bfloat16 | GPU | Output |
| `offsets` | `[num_groups * 2]` | int32 | GPU | Flat (start, end) row pairs |
| `experts` | `[num_groups + 1]` | int32 | GPU | Expert IDs (-1 terminated) |
| `list_size` | scalar | int | — | Entries in `experts` incl. sentinel |
| `compiled_dims` | string | — | — | Tile config hint (default `"nk"`) |

#### `asym_gemm.m_grouped_bf16_asym_gemm_nt_masked(a, b, d, masked_m, expected_m, compiled_dims="nk")`

BF16 asymmetric grouped GEMM with masked layout (padded `[G, M_max, K]`).

| Parameter | Shape | Dtype | Location | Description |
|-----------|-------|-------|----------|-------------|
| `a` | `[num_groups, M_max, K]` | bfloat16 | GPU | Padded activations |
| `b` | `[num_groups, N, K]` | bfloat16 | CPU pinned | Expert weights |
| `d` | `[num_groups, M_max, N]` | bfloat16 | GPU | Padded output |
| `masked_m` | `[num_groups]` | int32 | GPU | Valid row count per group |
| `expected_m` | scalar | int | — | Expected rows per group (grid sizing hint) |

#### `asym_gemm.m_grouped_fp8_asym_gemm_nt_contiguous(a, b, d, offsets, experts, list_size, recipe=None, compiled_dims="nk", disable_ue8m0_cast=False)`

FP8 asymmetric grouped GEMM with contiguous layout. `a` and `b` are `(data_tensor, scale_factor_tensor)` tuples.

| Parameter | Shape | Dtype | Location | Description |
|-----------|-------|-------|----------|-------------|
| `a` | tuple: `([M, K], scales)` | float8_e4m3fn | GPU | Activations + scale factors |
| `b` | tuple: `([num_groups, N, K], scales)` | float8_e4m3fn | CPU pinned | Expert weights + scale factors |
| `d` | `[M, N]` | bfloat16 | GPU | Output |
| `offsets` | `[num_groups * 2]` | int32 | GPU | Flat (start, end) row pairs |
| `experts` | `[num_groups + 1]` | int32 | GPU | Expert IDs (-1 terminated) |
| `list_size` | scalar | int | — | Entries in `experts` incl. sentinel |
| `recipe` | tuple | — | — | Optional quantization granularity `(gran_mn_a, gran_mn_b, gran_k)` |

On SM90/SM100 the scale factors are transformed into the kernel's compute layout automatically. On SM89 the original scales are consumed natively: a 3-D `b` scale (`[G, ceil(N/128), ceil(K/128)]` fp32, with `a` scales `[M, ceil(K/128)]`) selects the block-scale path; lower-rank scales are treated as per-token (`a`) / per-channel (`b`) tensors.

#### `asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(a, b, d, masked_m, expected_m, recipe=None, compiled_dims="nk", disable_ue8m0_cast=False)`

FP8 asymmetric grouped GEMM with masked layout. `a` is a tuple of padded data `[G, M_max, K]` and scales; `masked_m`/`expected_m` as in the BF16 masked variant.

#### `asym_gemm.m_grouped_int8_asym_gemm_nt_contiguous(a, b, d, offsets, experts, list_size, recipe=None, compiled_dims="nk")`

INT8 asymmetric grouped GEMM with contiguous layout (SM90 only). Fixed `(1, 1, 128)` block recipe: per-token activation scales, per-channel weight scales, both fp32 with `Kb = ceil(K/128)` blocks along K. Output is BF16.

| Parameter | Shape | Dtype | Location | Description |
|-----------|-------|-------|----------|-------------|
| `a` | tuple: `([M, K], scales [M, Kb])` | int8 / float32 | GPU | Activations + per-token scales |
| `b` | tuple: `([num_groups, N, K], scales [num_groups, N, Kb])` | int8 / float32 | CPU pinned | Expert weights + per-channel scales |
| `d` | `[M, N]` | bfloat16 | GPU | Output |
| `offsets` | `[num_groups * 2]` | int32 | GPU | Flat (start, end) row pairs |
| `experts` | `[num_groups + 1]` | int32 | GPU | Expert IDs (-1 terminated) |
| `list_size` | scalar | int | — | Entries in `experts` incl. sentinel |

`recipe` and `compiled_dims` are accepted for signature parity with the other dtypes but ignored. Use `scripts/convert_int8_weights.py` to produce INT8 weights + scales offline from a BF16 MoE checkpoint.

#### `asym_gemm.m_grouped_int8_asym_gemm_nt_masked(a, b, d, masked_m, expected_m, recipe=None, compiled_dims="nk")`

INT8 asymmetric grouped GEMM with masked layout (SM90 only). `a` scales are `[num_groups, M_max, Kb]`.

#### `asym_gemm.m_grouped_fp4_asym_gemm_nt_contiguous(a, b, d, offsets, experts, list_size, recipe=None, compiled_dims="nk", disable_ue8m0_cast=False)`

NVFP4 (E2M1) asymmetric grouped GEMM with contiguous layout (SM100 only). `a` and `b` are `(packed_data, scale_factor)` tuples.

| Parameter | Shape | Dtype | Location | Description |
|-----------|-------|-------|----------|-------------|
| `a` | tuple: `([M, K//2], scale)` | uint8 | GPU | Packed FP4 activations (2 per byte) + scale |
| `b` | tuple: `([num_groups, N, K//2], scale)` | uint8 | CPU pinned | Packed FP4 weights + scale |
| `d` | `[M, N]` | bfloat16 | GPU | Output |
| `offsets` | `[num_groups * 2]` | int32 | GPU | Flat (start, end) row pairs |
| `experts` | `[num_groups + 1]` | int32 | GPU | Expert IDs (-1 terminated) |
| `list_size` | scalar | int | — | Entries in `experts` incl. sentinel |

#### `asym_gemm.m_grouped_fp4_asym_gemm_nt_masked(a, b, d, masked_m, expected_m, recipe=None, compiled_dims="nk", disable_ue8m0_cast=False)`

NVFP4 asymmetric grouped GEMM with masked layout (SM100 only).

> **Note:** the former SM89-specific entry points (`m_grouped_fp8_asym_gemm_sm89[_masked]`) are now internal — SM89 is reached through the architecture-agnostic FP8/BF16 functions above.

---

### SM80 MoE GEMM (Ampere and above)

#### `asym_gemm.m_grouped_moe_gemm_nt_contiguous(a, b, d, offsets, experts, list_size, compiled_dims="nk")`

BF16/FP16 MoE grouped GEMM. All tensors must be on GPU. Dtype resolved at runtime from input tensors.

| Parameter | Shape | Dtype | Location | Description |
|-----------|-------|-------|----------|-------------|
| `a` | `[M, K]` | float16 or bfloat16 | GPU | Activations |
| `b` | `[num_experts, N, K]` | same as `a` | GPU | Expert weights |
| `d` | `[M, N]` | same as `a` | GPU | Output |
| `offsets` | `[num_groups * 2]` | int32 | GPU | Flat (start, end) row pairs |
| `experts` | `[num_groups + 1]` | int32 | GPU | Expert IDs (-1 terminated) |
| `list_size` | scalar | int | — | Entries in `experts` incl. sentinel |

**Alignment:** `K % 16 == 0`, `N % 32 == 0`, `K >= 64`.

---

### Unified MoE Layer (CPU + GPU)

The `asym_gemm.unified_moe` package executes a full INT8 MoE layer as two concurrent buckets — small experts on the host CPU (AMX/AVX-512 via the bundled `cpu_gemm` library), large experts on the GPU — over the same pinned weight bytes. It is importable whenever the CPU extension (`asym_gemm._cpu_C`) is built, independently of the CUDA extension.

#### `unified_moe.Layer`

```python
from asym_gemm.unified_moe import Layer, DispatchModel

layer = Layer.from_bf16(gate, up, down, top_k=k, adaptive=True)
out = layer.forward(x_bf16, expert_ids, route_w)   # [T, hidden] bf16
```

| Method | Description |
|--------|-------------|
| `Layer.from_bf16(gate, up, down, *, top_k, cpu_threads=0, cuda_device=0, m_cpu=16, adaptive=False, dispatch_model=None)` | Quantize BF16 expert masters (`gate`/`up`: `[G, inter, hidden]`, `down`: `[G, hidden, inter]`) to per-channel INT8 and build both backends |
| `Layer.from_int8(...)` | Load pre-quantized weights (see `scripts/convert_int8_weights.py`) |
| `layer.forward(x_bf16, expert_ids, route_w)` | One MoE forward: `x_bf16 [T, hidden]`, `expert_ids [T, top_k]` int, `route_w [T, top_k]` fp32 |
| `layer.set_m_cpu(k)` | Static dispatch: experts with ≤ k routed tokens run on the CPU |
| `layer.set_adaptive(True)` | Enable cost-model-based adaptive dispatch |
| `layer.calibrate()` | Optional forced all-CPU/all-GPU sweeps to seed the cost model |
| `layer.dispatch.snapshot()` | Fitted cost-model coefficients + observation counts |

#### `unified_moe.DispatchModel`

Per-backend linear cost models with a makespan partition solver. One instance can be shared by every same-shape layer of a model (`dispatch_model=` argument), pooling observations across layers; `model.rates()` / `DispatchModel.from_rates(...)` transfer a fitted model across shapes or processes. See [`adaptive_dispatch.md`](../adaptive_dispatch.md) for the design.

---

## Utility Functions

#### `asym_gemm.transform_sf_into_required_layout(scales, ...)`

Transform scale factor tensors into TMA-aligned layout required by SM90/SM100 kernels.

#### `asym_gemm.fp8_einsum(equation, a, b, scale_a, scale_b)`

FP8 tensor contraction (einsum-style interface).

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_ENABLE_JIT_ASYMGEMM` | `True` | Enable AsymGEMM JIT compilation in SGLang |
| `SGLANG_JIT_ASYMGEMM_PRECOMPILE` | `True` | Pre-compile kernel variants at startup |
| `SGLANG_JIT_ASYMGEMM_FAST_WARMUP` | `False` | Faster but less thorough precompile |
| `DG_JIT_CACHE_DIR` | `~/.asym_gemm/` | JIT kernel cache directory |
| `SGLANG_MASKED_GEMM_CHUNK_SIZE` | `0` | Expert group chunk size (0 = no chunking) |
| `SGLANG_MASKED_GEMM_FAST_ACT` | `False` | Fused SiLU + quantization path |
| `SGLANG_ASYMGEMM_SANITY_CHECK` | `False` | Enable input validation checks |
| `SGLANG_ENABLE_SM89_ASYMGEMM` | `False` | Force SM89 path on non-Ada GPUs (testing) |

Unified MoE runtime (`asym_gemm.unified_moe`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ASYMGEMM_GPU_CACHED_EXPERTS` | `0` | Number of experts whose INT8 weights are copied to GPU HBM (always run on GPU) |
| `ASYMGEMM_CPU_PREFILL_FRACTION` | `0.1` | Fixed fraction of streamed experts sent to the CPU bucket (superseded by adaptive dispatch) |
| `ASYMGEMM_RECORD_EXPERT_STATS` | unset | Path (`<stats.pt>`): eager forwards accumulate per-layer expert routing counts to this file |
| `ASYMGEMM_CACHE_HOT_EXPERTS` | unset | Path to a stats file from a `ASYMGEMM_RECORD_EXPERT_STATS` profiling run: cache each layer's hottest experts in HBM |
| `ASYMGEMM_STAGE_STREAMED` | `1` | Stage streamed expert weights through the copy engine; `0` disables |
| `ASYMGEMM_NUMA_TP` | `0` | `1` enables NUMA-aware tensor-parallel CPU thread placement |
| `ASYM_GEMM_FORCE_BACKEND` | unset | Force the CPU INT8 backend: `amx`, `avx512`, or `none` (testing) |

---
