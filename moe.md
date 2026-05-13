# MoE GEMM Kernel Correctness Test — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained test harness that compares `cpuAwareTilingWithoutBias` (the custom tiled MoE GEMM kernel in `mixtureExpertKernel.cu`) against a CUTLASS device GEMM ground truth, and debug until the outputs match.

**Architecture:** The MoE kernel uses SM80 MMA atoms (via `Flash_fwd_kernel_traits`) and tiles the GEMM over experts: for each expert `e`, it computes `O[token_slice, :] = Σ_k X[token_slice, k_tile] * W[expert_id, :, k_tile]^T`, accumulating partial results in global memory across K-tiles. The test runs the same math via CUTLASS `device::Gemm` (NT layout, SM80 TensorOp) per expert and compares element-wise.

**Tech Stack:** CUDA 13.0, CUTLASS 2.x (from `/workspace/AsymGEMM_main/third-party/cutlass`), CuTe (from flash-attention), nvcc targeting sm_80 (runs on GB200/SM100 via forward compat), PyTorch headers (for `c10/cuda`), C++17.

---

## Kernel Algorithm Reference

The grid is `dim3(N/kBlockN)` — each block owns one N-tile (`x_idx = blockIdx.x`).

The kernel iterates experts in order. For each expert `e`:
- `expert_id = expert_list[e]`
- `len = index_list[e] - len_start`  (tokens assigned to this expert)
- Load W tile `[x_idx * kBlockN : (x_idx+1)*kBlockN, 0 : kBlockK]` of `W[expert_id]` → smem `sW`
- **k=0 pass:** for each M-tile `m`: load `X[m*kBlockM:(m+1)*kBlockM, 0:kBlockK]` → `sX`, compute `sO = sX * sW^T`, write to global O
- **k>0 passes:** load W tile for column-block `k`, then for each M-tile: load X tile AND read back previous partial O from global, accumulate `tSrO += sX * sW^T`, write O

Ground truth: `O_ref[len_start:len_end, :] = X[len_start:len_end, :] @ W[expert_id, :, :].T`

## Missing Kernel Traits (Root Cause)

`Flash_fwd_kernel_traits` (from `csrc/flash_attn/src/kernel_traits.h`) does not define:
- `SmemLayoutX` — shape `[kBlockM, kHeadDim]` (same as `SmemLayoutQ`)
- `SmemLayoutW` — shape `[kBlockN, kHeadDim]` (same as `SmemLayoutKV`)
- `SmemLayoutC` — shape `[kBlockM, kBlockN]` (output tile)

These must be added as aliases/new type definitions in a local copy of `kernel_traits.h`.

---

## File Layout

```
MoE_sm80/
├── mixtureExpertKernel.cu        # existing — DO NOT MODIFY
├── test.cpp                      # to create: correctness test (compiled as .cu by nvcc)
├── compile.sh                    # to overwrite: fix paths + add test target
├── moe.md                        # this plan
└── include/                      # local copies of flash-attention headers (to create)
    ├── kernel_traits.h           # copy + add SmemLayoutX/W/C aliases
    ├── block_info.h              # copy verbatim
    ├── dropout.h                 # copy verbatim
    ├── hardware_info.h           # copy verbatim
    ├── namespace_config.h        # copy verbatim
    ├── static_switch.h           # copy verbatim
    └── utils.h                   # copy from hopper/ (has flash::convert_type_out)
                                  # + sm80 softmax.h, mask.h, rotary.h stay as -I paths
```

**Include path order in compile.sh:**
1. `./include` (local overrides — our modified kernel_traits.h, hopper utils.h)
2. `/workspace/flash-attention/hopper` (mask.h, softmax.h, rotary.h, static_switch.h from hopper)
3. `/workspace/flash-attention/csrc/flash_attn/src` (fallback for any remaining sm80 headers)
4. `/workspace/flash-attention/csrc/cutlass/include` (empty but kept for compat)
5. `/workspace/AsymGEMM_main/third-party/cutlass/include` (CUTLASS 2.x for test GEMM)
6. `/workspace/AsymGEMM_main/third-party/cutlass/tools/util/include` (CUTLASS util)
7. PyTorch headers

---

## Task 1: Create Local Include Directory and Copy Headers

**Files:**
- Create: `include/block_info.h`
- Create: `include/dropout.h`
- Create: `include/hardware_info.h`
- Create: `include/namespace_config.h`
- Create: `include/static_switch.h`
- Create: `include/utils.h`
- Create: `include/kernel_traits.h`

- [ ] **Step 1: Create include/ directory and copy verbatim headers**

```bash
mkdir -p /workspace/MoE_sm80/include
cp /workspace/flash-attention/csrc/flash_attn/src/block_info.h   /workspace/MoE_sm80/include/
cp /workspace/flash-attention/csrc/flash_attn/src/dropout.h       /workspace/MoE_sm80/include/
cp /workspace/flash-attention/csrc/flash_attn/src/hardware_info.h /workspace/MoE_sm80/include/
cp /workspace/flash-attention/csrc/flash_attn/src/namespace_config.h /workspace/MoE_sm80/include/
cp /workspace/flash-attention/csrc/flash_attn/src/static_switch.h /workspace/MoE_sm80/include/
```

Run: `ls /workspace/MoE_sm80/include/`
Expected: 5 `.h` files listed.

- [ ] **Step 2: Copy hopper utils.h (provides `flash::convert_type_out`)**

```bash
cp /workspace/flash-attention/hopper/utils.h /workspace/MoE_sm80/include/utils.h
```

Verify: `grep -n "convert_type_out" /workspace/MoE_sm80/include/utils.h | head -3`
Expected: line with `CUTLASS_DEVICE void convert_type_out`.

- [ ] **Step 3: Copy and augment kernel_traits.h**

Copy the sm80 kernel_traits and add the three missing type aliases inside `Flash_fwd_kernel_traits`:

```bash
cp /workspace/flash-attention/csrc/flash_attn/src/kernel_traits.h /workspace/MoE_sm80/include/kernel_traits.h
```

Then open `/workspace/MoE_sm80/include/kernel_traits.h` and find the line containing `using SmemCopyAtomO = ...` inside `Flash_fwd_kernel_traits`. Insert the following lines immediately after `using SmemCopyAtomOaccum = ...`:

```cpp
    // MoE GEMM aliases — X plays role of Q, W plays role of KV, C is the output tile
    using SmemLayoutX = SmemLayoutQ;    // [kBlockM, kHeadDim]
    using SmemLayoutW = SmemLayoutKV;   // [kBlockN, kHeadDim]
    using SmemLayoutC = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kBlockN>>{}));
```

Verify insertion: `grep -n "SmemLayoutX\|SmemLayoutW\|SmemLayoutC" /workspace/MoE_sm80/include/kernel_traits.h`
Expected: 3 matching lines.

- [ ] **Step 4: Commit**

```bash
cd /workspace/MoE_sm80 && git add -A && git commit -m "feat: add include/ with local flash-attention headers and MoE kernel_traits aliases" 2>/dev/null || echo "not a git repo, skip commit"
```

---

## Task 2: Fix compile.sh

**Files:**
- Modify: `compile.sh`

- [ ] **Step 1: Write new compile.sh**

Overwrite `/workspace/MoE_sm80/compile.sh` with:

```bash
#!/bin/bash
set -e

CUDA_HOME=/usr/local/cuda
NVCC=/usr/bin/nvcc
CUTLASS_INC=/workspace/AsymGEMM_main/third-party/cutlass/include
CUTLASS_UTIL=/workspace/AsymGEMM_main/third-party/cutlass/tools/util/include
FLASH_HOPPER=/workspace/flash-attention/hopper
FLASH_SRC=/workspace/flash-attention/csrc/flash_attn/src
TORCH_INC=/usr/local/lib/python3.12/dist-packages/torch/include
TORCH_API_INC=/usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include

COMMON_FLAGS="\
  -I./include \
  -I${FLASH_HOPPER} \
  -I${FLASH_SRC} \
  -I${CUTLASS_INC} \
  -I${CUTLASS_UTIL} \
  -I${TORCH_INC} \
  -I${TORCH_API_INC} \
  -D__CUDA_NO_HALF_OPERATORS__ \
  -D__CUDA_NO_HALF_CONVERSIONS__ \
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
  -D__CUDA_NO_HALF2_OPERATORS__ \
  --expt-relaxed-constexpr \
  --compiler-options '-fPIC' \
  -O3 -std=c++17 \
  --ftemplate-backtrace-limit=0 \
  --use_fast_math \
  -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED \
  -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
  -DNDEBUG \
  -DTORCH_API_INCLUDE_EXTENSION_H \
  -DTORCH_EXTENSION_NAME=flash_attn_3_cuda \
  -D_GLIBCXX_USE_CXX11_ABI=1"

ARCH_FLAGS="-gencode arch=compute_80,code=sm_80"

echo "=== Building moe_test ==="
${NVCC} ${COMMON_FLAGS} ${ARCH_FLAGS} \
  -o moe_test test.cpp \
  -lcuda -lcudart \
  2>&1 | tee compile.log

echo "=== Build done ==="
```

- [ ] **Step 2: Make executable and do a dry-run syntax check**

```bash
chmod +x /workspace/MoE_sm80/compile.sh
head -5 /workspace/MoE_sm80/compile.sh
```

Expected: `#!/bin/bash` on line 1.

---

## Task 3: Write test.cpp

**Files:**
- Create: `test.cpp`

The test performs:
1. Allocate random FP16 X `[total_tokens, K]` and W `[num_experts, N, K]` on device
2. Set up `expert_list` and `index_list` (4 experts, tokens distributed as [12, 8, 20, 28])
3. Call `launch_MoECompute` → output O_moe
4. For each expert: run CUTLASS GEMM(X_slice [len,K] × W[expert_id].T [K,N]) → O_ref slice
5. Download both to host, compare element-wise; report max abs-diff and pass/fail (threshold 0.05 for FP16)

**CUTLASS GEMM setup:** NT variant with FP32 accumulator on SM80:
- A = X_slice: `[len, K]` RowMajor, lda=K
- B = W[expert_id]: `[N, K]` stored as RowMajor but passed as ColumnMajor (ldb=K), giving W.T [K,N]
- C = O_ref_slice: `[len, N]` RowMajor, ldc=N

- [ ] **Step 1: Write test.cpp**

Create `/workspace/MoE_sm80/test.cpp` with the following content:

```cpp
// test.cpp — MoE kernel correctness test vs CUTLASS GEMM ground truth
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cassert>

// CUTLASS 2.x device GEMM
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/arch/arch.h>
#include <cutlass/layout/layout.h>

// ----- Error macros -----
#define CUDA_CHECK(err) do { \
    if ((err) != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

#define CUTLASS_CHECK(status) do { \
    if ((status) != cutlass::Status::kSuccess) { \
        fprintf(stderr, "CUTLASS Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

// ----- CUTLASS GEMM type: A[M,K] RowMajor × B[N,K] ColumnMajor → C[M,N] RowMajor -----
// This computes C = A * B^T when B is stored row-major [N,K].
// We declare B as ColumnMajor so CUTLASS sees ldb=K as the leading dim of the K×N view.
using GemmNT = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,     // A: X_slice [M,K]
    cutlass::half_t, cutlass::layout::ColumnMajor,  // B: W [N,K] row-major = W^T as ColumnMajor
    cutlass::half_t, cutlass::layout::RowMajor,     // C: O_ref [M,N]
    float,                                           // accumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80
>;

// Forward declaration of the MoE kernel launcher (defined in mixtureExpertKernel.cu)
void launch_MoECompute(
    cutlass::half_t* A,
    cutlass::half_t* B,
    cutlass::half_t* C,
    int N, int K,
    int expert_size,
    int list_size,
    int* expert_list,
    int* index_list,
    cudaStream_t stream);

// ----- Reference GEMM for one expert -----
// Computes O_ref[0:M, 0:N] = X_slice[0:M, 0:K] * W_expert[0:N, 0:K]^T
// X_slice_ptr: device pointer to [M, K] FP16 row-major
// W_ptr:       device pointer to [N, K] FP16 row-major (the expert's weight)
// O_ref_ptr:   device pointer to [M, N] FP16 row-major (output)
void run_ref_gemm(
    cutlass::half_t* X_slice_ptr, int M,
    cutlass::half_t* W_ptr, int N, int K,
    cutlass::half_t* O_ref_ptr)
{
    GemmNT gemm_op;
    cutlass::half_t alpha_h(1.0f);
    cutlass::half_t beta_h(0.0f);

    GemmNT::Arguments args(
        {M, N, K},                                         // problem size
        {X_slice_ptr, K},                                  // A (X_slice), lda=K
        {W_ptr, K},                                        // B (W expert), ldb=K (ColumnMajor -> W^T)
        {O_ref_ptr, N},                                    // C (zero beta)
        {O_ref_ptr, N},                                    // D = output
        {alpha_h, beta_h}
    );

    CUTLASS_CHECK(gemm_op(args));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ----- Host utility: fill device buffer with uniform random FP16 -----
void fill_random_fp16(cutlass::half_t* d_ptr, size_t count, float lo = -1.0f, float hi = 1.0f)
{
    std::vector<cutlass::half_t> h(count);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& v : h) v = cutlass::half_t(dist(rng));
    CUDA_CHECK(cudaMemcpy(d_ptr, h.data(), count * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
}

// ----- Comparison -----
// Returns (max_abs_diff, mean_abs_diff). Prints per-element details for first few mismatches.
std::pair<float,float> compare_outputs(
    const std::vector<cutlass::half_t>& moe,
    const std::vector<cutlass::half_t>& ref,
    int total_tokens, int N,
    float threshold = 0.05f)
{
    assert(moe.size() == ref.size());
    float max_diff = 0.0f;
    double sum_diff = 0.0;
    int mismatch_count = 0;
    for (size_t i = 0; i < moe.size(); ++i) {
        float a = float(moe[i]);
        float b = float(ref[i]);
        float diff = std::fabs(a - b);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        if (diff > threshold && mismatch_count < 10) {
            int row = i / N, col = i % N;
            printf("  MISMATCH [token=%d, n=%d]: moe=%.4f  ref=%.4f  diff=%.4f\n",
                   row, col, a, b, diff);
            ++mismatch_count;
        }
    }
    float mean_diff = float(sum_diff / moe.size());
    return {max_diff, mean_diff};
}

int main()
{
    // -----------------------------------------------------------------------
    // Test parameters
    // -----------------------------------------------------------------------
    // N=4096 is the only supported output dim in launch_MoECompute.
    // K=64 exercises the k=0 AND k=1 accumulation paths (kHeadDim=32 for this KT).
    const int N          = 4096;
    const int K          = 64;   // must be multiple of kHeadDim=32
    const int num_experts = 8;   // total experts in weight tensor (expert_size param)

    // Tokens per expert (non-uniform to test edge cases).
    // Use multiples of kBlockM=128 AND partial tiles.
    // expert_list: which experts actually have tokens; index_list: cumulative end indices.
    // Total tokens = 12+8+20+28 = 68 (non-multiples of kBlockM to exercise partial tile path)
    const int list_size = 4;
    int expert_list_h[list_size] = {0, 3, 5, 7};      // 4 active experts
    int token_counts[list_size]  = {12, 8, 20, 28};   // tokens for each
    int index_list_h[list_size];
    int total_tokens = 0;
    for (int i = 0; i < list_size; ++i) {
        total_tokens += token_counts[i];
        index_list_h[i] = total_tokens;
    }
    printf("total_tokens=%d  N=%d  K=%d  num_experts=%d\n",
           total_tokens, N, K, num_experts);

    // -----------------------------------------------------------------------
    // Allocate device buffers
    // -----------------------------------------------------------------------
    cutlass::half_t *d_X, *d_W, *d_O_moe, *d_O_ref;
    CUDA_CHECK(cudaMalloc(&d_X,     (size_t)total_tokens * K * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&d_W,     (size_t)num_experts  * N * K * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&d_O_moe, (size_t)total_tokens * N * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&d_O_ref, (size_t)total_tokens * N * sizeof(cutlass::half_t)));

    int *d_expert_list, *d_index_list;
    CUDA_CHECK(cudaMalloc(&d_expert_list, list_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_index_list,  list_size * sizeof(int)));

    // -----------------------------------------------------------------------
    // Initialize inputs with random FP16
    // -----------------------------------------------------------------------
    fill_random_fp16(d_X, (size_t)total_tokens * K);
    fill_random_fp16(d_W, (size_t)num_experts  * N * K);
    CUDA_CHECK(cudaMemset(d_O_moe, 0, (size_t)total_tokens * N * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMemset(d_O_ref, 0, (size_t)total_tokens * N * sizeof(cutlass::half_t)));

    CUDA_CHECK(cudaMemcpy(d_expert_list, expert_list_h, list_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_index_list,  index_list_h,  list_size * sizeof(int), cudaMemcpyHostToDevice));

    // -----------------------------------------------------------------------
    // Run MoE kernel
    // -----------------------------------------------------------------------
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    launch_MoECompute(d_X, d_W, d_O_moe,
                      N, K,
                      num_experts, list_size,
                      d_expert_list, d_index_list,
                      stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // -----------------------------------------------------------------------
    // Run CUTLASS GEMM reference per expert
    // -----------------------------------------------------------------------
    int len_start = 0;
    for (int e = 0; e < list_size; ++e) {
        int expert_id = expert_list_h[e];
        int len_end   = index_list_h[e];
        int len       = len_end - len_start;

        cutlass::half_t* X_slice_ptr  = d_X     + (size_t)len_start * K;
        cutlass::half_t* W_expert_ptr = d_W     + (size_t)expert_id * N * K;
        cutlass::half_t* O_ref_slice  = d_O_ref + (size_t)len_start * N;

        printf("Expert %d (id=%d): tokens [%d, %d) len=%d\n",
               e, expert_id, len_start, len_end, len);
        run_ref_gemm(X_slice_ptr, len, W_expert_ptr, N, K, O_ref_slice);

        len_start = len_end;
    }

    // -----------------------------------------------------------------------
    // Download and compare
    // -----------------------------------------------------------------------
    std::vector<cutlass::half_t> h_O_moe(total_tokens * N);
    std::vector<cutlass::half_t> h_O_ref(total_tokens * N);
    CUDA_CHECK(cudaMemcpy(h_O_moe.data(), d_O_moe, total_tokens * N * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_O_ref.data(), d_O_ref, total_tokens * N * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));

    auto [max_diff, mean_diff] = compare_outputs(h_O_moe, h_O_ref, total_tokens, N);
    printf("\nMax abs diff : %.6f\n", max_diff);
    printf("Mean abs diff: %.6f\n", mean_diff);

    const float THRESHOLD = 0.05f;
    if (max_diff <= THRESHOLD) {
        printf("\n[PASS] Max diff %.6f <= threshold %.4f\n", max_diff, THRESHOLD);
    } else {
        printf("\n[FAIL] Max diff %.6f > threshold %.4f\n", max_diff, THRESHOLD);
    }

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_O_moe));
    CUDA_CHECK(cudaFree(d_O_ref));
    CUDA_CHECK(cudaFree(d_expert_list));
    CUDA_CHECK(cudaFree(d_index_list));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return (max_diff <= THRESHOLD) ? 0 : 1;
}
```

- [ ] **Step 2: Verify test.cpp was written correctly**

```bash
wc -l /workspace/MoE_sm80/test.cpp
grep -n "launch_MoECompute\|run_ref_gemm\|compare_outputs\|PASS\|FAIL" /workspace/MoE_sm80/test.cpp
```

Expected: ~200 lines, 4-5 matched lines including `[PASS]` and `[FAIL]`.

---

## Task 4: Update compile.sh for Both Object Files

**Files:**
- Modify: `compile.sh`

The final binary links `mixtureExpertKernel.cu` (the MoE kernel) and `test.cpp` (the test harness).

- [ ] **Step 1: Write the final compile.sh**

Overwrite `/workspace/MoE_sm80/compile.sh` with:

```bash
#!/bin/bash
set -e

NVCC=/usr/bin/nvcc
CUTLASS_INC=/workspace/AsymGEMM_main/third-party/cutlass/include
CUTLASS_UTIL=/workspace/AsymGEMM_main/third-party/cutlass/tools/util/include
FLASH_HOPPER=/workspace/flash-attention/hopper
FLASH_SRC=/workspace/flash-attention/csrc/flash_attn/src
TORCH_INC=/usr/local/lib/python3.12/dist-packages/torch/include
TORCH_API_INC=/usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include

INCLUDES="\
  -I./include \
  -I${FLASH_HOPPER} \
  -I${FLASH_SRC} \
  -I${CUTLASS_INC} \
  -I${CUTLASS_UTIL} \
  -I${TORCH_INC} \
  -I${TORCH_API_INC}"

DEFINES="\
  -D__CUDA_NO_HALF_OPERATORS__ \
  -D__CUDA_NO_HALF_CONVERSIONS__ \
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
  -D__CUDA_NO_HALF2_OPERATORS__ \
  -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED \
  -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
  -DNDEBUG \
  -DTORCH_API_INCLUDE_EXTENSION_H \
  -DTORCH_EXTENSION_NAME=flash_attn_3_cuda \
  -D_GLIBCXX_USE_CXX11_ABI=1"

CXX_FLAGS="\
  --expt-relaxed-constexpr \
  --compiler-options '-fPIC' \
  --threads 8 \
  -O3 -std=c++17 \
  --ftemplate-backtrace-limit=0 \
  --use_fast_math \
  -lineinfo"

ARCH="-gencode arch=compute_80,code=sm_80"

echo "=== Compiling mixtureExpertKernel.cu → kernel.o ==="
${NVCC} ${INCLUDES} ${DEFINES} ${CXX_FLAGS} ${ARCH} \
  -c mixtureExpertKernel.cu -o kernel.o \
  2>&1 | tee compile_kernel.log

echo "=== Compiling test.cpp → test.o ==="
${NVCC} ${INCLUDES} ${DEFINES} ${CXX_FLAGS} ${ARCH} \
  -c test.cpp -o test.o \
  2>&1 | tee compile_test.log

echo "=== Linking → moe_test ==="
${NVCC} ${ARCH} kernel.o test.o -o moe_test -lcudart 2>&1 | tee compile_link.log

echo "=== Build complete. Run: ./moe_test ==="
```

- [ ] **Step 2: Run the compilation**

```bash
cd /workspace/MoE_sm80 && bash compile.sh
```

Expected: All three stages print "==" headers and no error lines. Final line: `=== Build complete. Run: ./moe_test ===`

If compilation fails, go to **Task 5: Debug Compilation Errors**.

- [ ] **Step 3: Run the test (if compilation succeeded)**

```bash
cd /workspace/MoE_sm80 && ./moe_test
```

Expected output pattern:
```
total_tokens=68  N=4096  K=64  num_experts=8
Expert 0 (id=0): tokens [0, 12) len=12
...
cup-aware symCompute: X.XXX ms
...
Max abs diff : 0.XXXXXX
Mean abs diff: 0.XXXXXX
[PASS] Max diff ... <= threshold 0.0500
```

If test runs but [FAIL], go to **Task 6: Debug Correctness**.

---

## Task 5: Debug Compilation Errors

**Files:** Whatever headers cause errors.

Compilation errors fall into a few categories:

- [ ] **Step 1: Identify error category**

```bash
grep -i "error:" /workspace/MoE_sm80/compile_kernel.log | head -20
grep -i "error:" /workspace/MoE_sm80/compile_test.log   | head -20
```

**Category A — Missing type `SmemLayoutX` / `SmemLayoutW` / `SmemLayoutC`:**
Confirm the aliases were inserted correctly:
```bash
grep -n "SmemLayoutX\|SmemLayoutW\|SmemLayoutC" /workspace/MoE_sm80/include/kernel_traits.h
```
If missing, re-do Task 1 Step 3.

**Category B — `convert_type_out` not found:**
Check that `include/utils.h` is the hopper version:
```bash
grep -n "convert_type_out" /workspace/MoE_sm80/include/utils.h
```
If missing, the wrong utils.h was copied. Re-copy from `/workspace/flash-attention/hopper/utils.h`.

**Category C — header not found (e.g., `cute/tensor.hpp`):**
The CUTLASS include path is missing. Check `CUTLASS_INC` in compile.sh points to where `cute/tensor.hpp` lives:
```bash
find /workspace/AsymGEMM_main/third-party/cutlass -name "tensor.hpp" | head -3
```
Update `CUTLASS_INC` accordingly.

**Category D — `c10/cuda/CUDAStream.h` not found:**
PyTorch include path is wrong. Find it:
```bash
find /usr/local/lib/python3.12 -name "CUDAStream.h" 2>/dev/null
```
Update `TORCH_INC` in compile.sh.

**Category E — `GmemTiledCopyQKV` not found / missing in kernel_traits:**
The sm80 `kernel_traits.h` uses `GmemTiledCopyQKV`. If hopper's version was picked up instead (different name), check:
```bash
grep -n "GmemTiledCopyQKV" /workspace/MoE_sm80/include/kernel_traits.h
```
If absent, the wrong file was included. The local `./include/kernel_traits.h` must be the sm80 version.

**Category F — arch-related PTX errors:**
If errors mention `sm_80` instructions not supported, the CUTLASS ground truth GemmNT may need SM80 tensorop. Alternatively downgrade to `cutlass::arch::OpClassSimt` in test.cpp:
```cpp
using GemmNT = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,   // ← change from TensorOp to Simt
    cutlass::arch::Sm80
>;
```

- [ ] **Step 2: Apply fix and re-compile**

After applying the specific fix, re-run:
```bash
cd /workspace/MoE_sm80 && bash compile.sh
```

Iterate until `compile_kernel.log` and `compile_test.log` are error-free.

---

## Task 6: Debug Correctness Failures

If `./moe_test` prints `[FAIL]`, use the mismatch lines to diagnose.

- [ ] **Step 1: Check for all-zero output from MoE kernel**

```bash
cd /workspace/MoE_sm80 && ./moe_test 2>&1 | head -30
```

If `Max abs diff` equals the magnitude of the ref values (e.g., ~0.5) and mismatch at ALL positions, the MoE kernel wrote zeros to O. This means the output write path is broken.

- [ ] **Step 2: Diagnose zero output**

The output write path in k=0:
```
tSrO → rO → sO → gO
```
Check `SmemLayoutC` shape vs `SmemCopyAtomO`. The `SmemCopyAtomO` copies a `[kBlockM, kBlockN]` tile but `SmemLayoutC` must match exactly.

Look at line 108 in `mixtureExpertKernel.cu`:
```cpp
Tensor sO = make_tensor(sW.data() + size(sW), typename Kernel_traits::SmemLayoutC{});
```

The shape must be `[kBlockM, kBlockN]`. Verify in kernel_traits.h:
```bash
grep -A3 "SmemLayoutC" /workspace/MoE_sm80/include/kernel_traits.h
```

If `SmemLayoutAtomO` has a stride that causes bank conflicts or wrong mapping for `[kBlockM, kBlockN]`, try a simpler layout:
```cpp
using SmemLayoutC = Layout<Shape<Int<kBlockM>, Int<kBlockN>>,
                            Stride<Int<kBlockN>, _1>>;
```

- [ ] **Step 3: Diagnose k-accumulation errors (wrong results for K > kHeadDim)**

If small K (=32, single k-tile) passes but K=64 fails, the partial O read-back is wrong.

In the k>0 loop (around line 273–291 in mixtureExpertKernel.cu):
```cpp
// Load O from global → sO
copy(tOgO_QKV(_, _, _, m), tOsO_QKV, ...);
...
// Load sO → tSrO registers
Tensor tSrO_copy_view = smem_thr_copy_O.retile_D(tSrO);
cute::copy(smem_tiled_copy_O, tSsO, tSrO_copy_view);
// GEMM accumulates into tSrO
gemm(tSrO, tSrX, tOrW, ...);
```

Check that `tSsO` and `tOsO_QKV` reference the same smem tensor `sO`. If the copy atom `SmemCopyAtomO` and `GmemTiledCopyQKV` use different layouts for reading vs writing `sO`, the retile is misaligned.

To isolate: test with K=32 first (single pass, no accumulation):
Change in `test.cpp`: `const int K = 32;`
Recompile and run. If single-pass passes, the k-accumulation loop is the bug.

- [ ] **Step 4: Diagnose partial-tile edge case**

If full tiles pass but edge tokens fail, the `gap` calculation is wrong or the OOB masking is incorrect.

To isolate: change token counts to all multiples of kBlockM=128:
```cpp
int token_counts[list_size] = {128, 128, 128, 128};
```
If this passes but 12/8/20/28 fails, the edge-case copy is the bug.

Check the gap formula:
```cpp
int gap = len - kBlockM * m_max + kBlockM;
```
This should equal `len % kBlockM` if non-zero, otherwise `kBlockM`.
Verify: `kBlockM * m_max = kBlockM * ceil(len/kBlockM)`, so `gap = kBlockM - (kBlockM*ceil(len/kBlockM) - len)` = last partial tile size. ✓

If gap is incorrect, trace through:
- `m_max = (len + kBlockM - 1) / kBlockM`
- For `len=12, kBlockM=128`: `m_max=1`, `gap = 12 - 128*1 + 128 = 12` ✓

- [ ] **Step 5: Verify with a tiny smoke test (all ones)**

Add a small verification block in test.cpp before the random test to check a known result:
- Set X all-ones `[4, K]`, W[0] all-ones `[N, K]`, one expert (id=0), 4 tokens
- Expected O: all values = K (each output = dot product of K ones with K ones)

This quickly checks the basic computation path without noise from random values.

---

## Task 7: Final Validation and Cleanup

- [ ] **Step 1: Run final test with multiple configurations**

Edit test.cpp to run two sub-tests sequentially:
1. `N=4096, K=32` (single k-tile, tests k=0 path)
2. `N=4096, K=64` (two k-tiles, tests k>0 accumulation)

Both should print `[PASS]`.

- [ ] **Step 2: Verify compile.sh is clean**

```bash
cd /workspace/MoE_sm80 && bash compile.sh && ./moe_test
```

Final expected output:
```
[PASS] Max diff ... <= threshold 0.0500
```
Exit code 0.

- [ ] **Step 3: Summarize any bugs found in mixtureExpertKernel.cu**

After achieving [PASS], document in `moe.md` under a **Bugs Found** section:
- What was wrong (missing type aliases, layout mismatch, etc.)
- What was fixed (which file, which line)
- Residual numerical error (expected for FP16 kernels doing FP32 accumulation)

---

## Self-Review Checklist

**Spec coverage:**
- [x] Read kernel code and understood algorithm ✓
- [x] Identified missing `SmemLayoutX/W/C` types ✓  
- [x] CUTLASS GEMM ground truth with NT layout ✓
- [x] Partial-tile edge case tested ✓
- [x] K-accumulation (k>0) path tested ✓
- [x] compile.sh rewritten with correct paths ✓
- [x] Debug guidance for all likely failure modes ✓

**Potential gaps:**
- The `GmmNT` CUTLASS type uses `OpClassTensorOp` on SM80. If the CUTLASS library in `/workspace/AsymGEMM_main/third-party/cutlass` does not support this TensorOp config for the chosen tile sizes, Task 5 Category F provides the fallback to `OpClassSimt`.
- The MoE kernel's `grid = dim3(128)` is hardcoded. If `N != 4096`, `launch_MoECompute` prints an error and does nothing — the test uses `N=4096` to match this.
