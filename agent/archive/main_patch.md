# Main Patches Needed For `main_kevin`

Apply these on top of `main` if `main_kevin` must support the BF16 `transpose_b=True` / dX path:

```text
A[M, N] @ B[G, N, K] -> D[M, K]
```

## Required Kernel/API Patches

1. `csrc/apis/gemm.hpp`
   - Add `transpose_b` argument to `m_grouped_bf16_asym_gemm_nt_contiguous`.
   - When `transpose_b=True`, treat physical `B[G,N,K]` as logical transposed B:
     - logical `n = K`
     - logical `k = N`
     - force `major_b = MN`
     - pass `b_outer_stride = b.stride(-2)`
   - Allow `D` to be `BF16` or `FP32`.
   - Add `int list_size` overload.
   - Dispatch BF16 to SM90 on arch 9 and SM100 on arch 10.

2. `csrc/jit_kernels/impls/sm100_bf16_asym_gemm.hpp`
   - Add optional `b_outer_stride` parameter to the SM100 BF16 contiguous launcher.
   - Use that stride when building the B TMA descriptor.
   - Use `block_k = 64` when `major_b == MN`; keep `block_k = 512` otherwise.

3. `asym_gemm/include/asym_gemm/impls/sm100_bf16_asym_gemm.cuh`
   - Fix MN-major B indexing for `MGroupedContiguous`.
   - For transposed/grouped B, use:
     - `b_n_idx = blockIdx.x * BLOCK_N`
     - `b_k_idx = k_idx + current_group_idx * shape_k`
   - This makes each group read its own `B[group]` slice.

## Supporting Patches

4. `setup.py`
   - Ensure Python packages and bundled headers are included:
     - `packages=find_packages()`
     - `package_data={'asym_gemm': ['include/**/*']}`

5. `tests/m_grouped/test_sm100_bf16_fp8_fp4_m_grouped.py`
   - Add/keep SM100 coverage for:
     - BF16 tensor `list_size`
     - BF16 `transpose_b=True` dX case
     - masked BF16 path
     - FP8/FP4 status cases

## Not Needed If

If only using the original forward path:

```text
A[M, K] @ B[G, N, K].T -> D[M, N]
```

then main's original SM100 BF16 kernel is sufficient.
