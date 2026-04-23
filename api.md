# `m_grouped_moe_gemm_nt_contiguous` API Alignment Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `m_grouped_moe_gemm_nt_contiguous` have the exact same Python and C++ API signature as `m_grouped_bf16_asym_gemm_nt_contiguous`, while preserving fp16+bf16 support.

**Architecture:** Three layers need touching — the C++ API wrapper (`csrc/apis/gemm.hpp`), the internal kernel dispatcher (`csrc/jit_kernels/impls/sm80_moe_gemm.hpp`), and the test (`tests/test_sm80_moe.py`). The `compiled_dims` parameter is accepted for API parity but ignored at runtime — the SM80 heuristic already selects tile dims automatically.

**Tech Stack:** C++17, pybind11, PyTorch C++ extension, CUDA JIT (NVCC/NVRTC).

---

## API diff

| Position | BF16 asym (target) | Current MoE | Change |
|----------|--------------------|-------------|--------|
| 1 | `a` — `[M,K]` activations | `x` | rename |
| 2 | `b` — `[G,N,K]` weights | `w` | rename |
| 3 | `d` — `[M,N]` output | `o` | rename |
| 4 | `offsets` — cumulative end indices (int32) | `expert_list` — expert IDs | rename + **swap semantic** |
| 5 | `experts` — expert IDs (int32) | `index_list` — cumulative ends | rename + **swap semantic** |
| 6 | `list_size` — int, number of active experts | _(derived internally)_ | **add** |
| 7 | `compiled_dims` — str, default `"nk"` | _(absent)_ | **add** (ignored for SM80) |

Semantic swap: in the current MoE kernel `expert_list` holds expert IDs and `index_list` holds cumulative ends. In the BF16 API `offsets` holds cumulative ends and `experts` holds expert IDs — the opposite order. After alignment, the C++ arg at position 4 is `offsets` (cumulative ends) and position 5 is `experts` (IDs).

---

## File map

| File | Change |
|------|--------|
| `csrc/apis/gemm.hpp:418-484` | Rename params, swap `offsets`/`experts` order, add `list_size` + `compiled_dims`, update internal call |
| `csrc/apis/gemm.hpp:553-559` | Update pybind11 registration with new param names and defaults |
| `csrc/jit_kernels/impls/sm80_moe_gemm.hpp:55-103` | Rename `sm80_m_grouped_moe_gemm_contiguous` params to match |
| `tests/test_sm80_moe.py:118` | Update call-site to new signature |

---

### Task 1: Update the C++ API wrapper signature and body

**Files:**
- Modify: `csrc/apis/gemm.hpp:418-484`

- [ ] **Step 1: Replace the function signature and body**

In `csrc/apis/gemm.hpp`, replace the `m_grouped_moe_gemm_nt_contiguous` function (lines 418–484) with:

```cpp
static void m_grouped_moe_gemm_nt_contiguous(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& d,
    const torch::Tensor& offsets,
    const torch::Tensor& experts,
    const int& list_size,
    const std::string& compiled_dims)
{
    // ── Shape checks ──────────────────────────────────────────────────────────
    DG_HOST_ASSERT(a.dim() == 2);
    DG_HOST_ASSERT(b.dim() == 3);
    DG_HOST_ASSERT(d.dim() == 2);

    const int64_t total_tokens   = a.size(0);
    const int64_t K              = a.size(1);
    const int64_t num_experts    = b.size(0);
    const int64_t N              = b.size(1);
    const int64_t K_b            = b.size(2);
    const int64_t total_tokens_d = d.size(0);
    const int64_t N_d            = d.size(1);

    DG_HOST_ASSERT(K == K_b);
    DG_HOST_ASSERT(N == N_d);
    DG_HOST_ASSERT(total_tokens == total_tokens_d);

    // ── Dtype checks ──────────────────────────────────────────────────────────
    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat16 or a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == a.scalar_type());
    DG_HOST_ASSERT(d.scalar_type() == a.scalar_type());

    // ── CUDA placement ────────────────────────────────────────────────────────
    DG_HOST_ASSERT(a.is_cuda());
    DG_HOST_ASSERT(b.is_cuda());
    DG_HOST_ASSERT(d.is_cuda());

    // ── Contiguity ────────────────────────────────────────────────────────────
    DG_HOST_ASSERT(a.is_contiguous());
    DG_HOST_ASSERT(b.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());

    // ── offsets / experts ─────────────────────────────────────────────────────
    DG_HOST_ASSERT(offsets.is_cuda() and experts.is_cuda());
    DG_HOST_ASSERT(offsets.is_contiguous() and experts.is_contiguous());
    DG_HOST_ASSERT(offsets.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(experts.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(offsets.numel() >= list_size and experts.numel() >= list_size);

    // ── Empty check (before alignment guards so K==0 doesn't trigger K>=64) ──
    if (total_tokens == 0 or N == 0 or K == 0) return;

    // ── Alignment checks ──────────────────────────────────────────────────────
    DG_HOST_ASSERT(K % 16 == 0);
    DG_HOST_ASSERT(N % 32 == 0);
    DG_HOST_ASSERT(K >= 64);

    // ── Resolve element type and dispatch ─────────────────────────────────────
    // compiled_dims is accepted for API parity with m_grouped_bf16_asym_gemm_nt_contiguous
    // but is unused: the SM80 kernel selects tile dims via runtime heuristic.
    const std::string element_type_str =
        (a.scalar_type() == torch::kFloat16) ? "cutlass::half_t" : "cutlass::bfloat16_t";

    sm80_m_grouped_moe_gemm_contiguous(
        a, b, d, experts, offsets,
        N, K,
        static_cast<int32_t>(num_experts),
        static_cast<int32_t>(list_size),
        element_type_str);
}
```

Note: `experts` (IDs) maps to the kernel's `expert_list` parameter; `offsets` (cumulative ends) maps to `index_list`. The call site intentionally passes `experts` before `offsets` because the internal kernel function keeps its original parameter order (`expert_list` then `index_list`).

---

### Task 2: Update the pybind11 registration

**Files:**
- Modify: `csrc/apis/gemm.hpp:553-559`

- [ ] **Step 1: Replace the pybind11 `m.def` block**

Find the registration block (currently under the comment `// SM80 MoE GEMM`) and replace it:

```cpp
    // SM80 MoE GEMM (FP16 + BF16, no arch guard needed: uses >= SM80 primitives)
    m.def("m_grouped_moe_gemm_nt_contiguous",
          static_cast<void(*)(const torch::Tensor&, const torch::Tensor&,
                              const torch::Tensor&, const torch::Tensor&,
                              const torch::Tensor&, const int&,
                              const std::string&)>(
              &m_grouped_moe_gemm_nt_contiguous),
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("offsets"), py::arg("experts"), py::arg("list_size"),
          py::arg("compiled_dims") = "nk");
```

---

### Task 3: Rename params in the internal kernel dispatcher

**Files:**
- Modify: `csrc/jit_kernels/impls/sm80_moe_gemm.hpp:55-103`

- [ ] **Step 1: Replace the free-function signature**

In `sm80_moe_gemm.hpp`, rename the parameters of `sm80_m_grouped_moe_gemm_contiguous` so the body stays identical but names mirror the API layer:

```cpp
static void sm80_m_grouped_moe_gemm_contiguous(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& d,
    const torch::Tensor& expert_list,   // expert IDs   (= API's "experts")
    const torch::Tensor& index_list,    // cumul. ends  (= API's "offsets")
    int64_t N, int64_t K, int32_t num_experts, int32_t list_size,
    const std::string& element_type_str)
```

The body is unchanged — `expert_list` and `index_list` are already the correct names inside the kernel; only the caller-facing rename in Task 1 matters.

- [ ] **Step 2: Update the `SM80MoEParams` initialiser to use the new names**

The struct initialiser already uses `expert_list` and `index_list` which are still the parameter names after Task 3, so no body changes are required. Verify the struct initialiser reads:

```cpp
    const SM80MoEParams params {
        .x_ptr        = a.data_ptr(),
        .w_ptr        = b.data_ptr(),
        .o_ptr        = d.data_ptr(),
        .expert_list  = expert_list.data_ptr<int32_t>(),
        .index_list   = index_list.data_ptr<int32_t>(),
        .list_size    = list_size,
        .expert_size  = num_experts,
        .N            = N,
        .K            = K,
    };
```

---

### Task 4: Update the test call-site

**Files:**
- Modify: `tests/test_sm80_moe.py:99-118`

- [ ] **Step 1: Rename variables and update the kernel call**

In `test_moe_gemm`, rename tensor variables and rebuild the call to match the new signature. Replace the allocation + call block:

```python
        expert_ids   = list(range(len(token_counts)))
        total_tokens = sum(token_counts)
        offsets_h    = list(itertools.accumulate(token_counts))  # cumulative end indices

        a       = torch.randn(total_tokens, K,    dtype=dtype,       device='cuda')
        b       = torch.randn(num_experts,  N, K, dtype=dtype,       device='cuda')
        d       = torch.empty(total_tokens, N,    dtype=dtype,       device='cuda')
        experts = torch.tensor(expert_ids, dtype=torch.int32, device='cuda')
        offsets = torch.tensor(offsets_h,  dtype=torch.int32, device='cuda')

        list_size = experts.numel()

        kernel_fn(a, b, d, offsets, experts, list_size)
        torch.cuda.synchronize()

        ref = ref_moe_gemm(a, b, experts, offsets)  # float32 reference

        diff = calc_diff(d.float(), ref)
```

- [ ] **Step 2: Update `ref_moe_gemm` call signature comment**

Update the `ref_moe_gemm` docstring parameter names to match:

```python
def ref_moe_gemm(a, b, experts, offsets):
    """
    a:       [total_tokens, K]   fp16 or bf16
    b:       [num_experts, N, K] fp16 or bf16
    experts: [list_size]         int32 — expert IDs
    offsets: [list_size]         int32 — cumulative end indices
    returns: [total_tokens, N]   fp32
    """
    total_tokens, K = a.shape
    num_experts, N, K_ = b.shape
    assert K == K_, "K mismatch"

    out = torch.zeros(total_tokens, N, dtype=torch.float32, device=a.device)
    start = 0
    elist = experts.tolist() if isinstance(experts, torch.Tensor) else experts
    ilist = offsets.tolist() if isinstance(offsets, torch.Tensor) else offsets

    for i, expert_id in enumerate(elist):
        end = ilist[i]
        out[start:end] = a[start:end].float() @ b[expert_id].float().t()
        start = end
    return out
```

- [ ] **Step 3: Update the `finally` cleanup block**

```python
        finally:
            del a, b, d, experts, offsets, ref
            torch.cuda.empty_cache()
            gc.collect()
```

---

### Task 5: Rebuild and verify

- [ ] **Step 1: Rebuild the extension in-place**

```bash
cd /workspace/AsymGEMM_SM80   # or /workspace/AsymGEMM on the 4090
python setup.py build_ext --inplace 2>&1 | tail -5
```

Expected: ends with `copying build/.../asym_gemm/_C*.so -> asym_gemm/`

- [ ] **Step 2: Run the test from `tests/`**

```bash
cd tests
python3 test_sm80_moe.py
```

Expected: 16 `[PASS]` lines followed by `All tests passed.`

- [ ] **Step 3: Smoke-check the new keyword-argument names work**

```bash
python3 -c "
import torch, asym_gemm
a = torch.randn(64, 256, dtype=torch.float16, device='cuda')
b = torch.randn(4, 4096, 256, dtype=torch.float16, device='cuda')
d = torch.empty(64, 4096, dtype=torch.float16, device='cuda')
offsets = torch.tensor([16, 32, 48, 64], dtype=torch.int32, device='cuda')
experts = torch.tensor([0, 1, 2, 3],     dtype=torch.int32, device='cuda')
asym_gemm.m_grouped_moe_gemm_nt_contiguous(a, b, d, offsets, experts, list_size=4)
print('OK')
"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
cd /workspace/AsymGEMM_SM80
git add csrc/apis/gemm.hpp csrc/jit_kernels/impls/sm80_moe_gemm.hpp tests/test_sm80_moe.py
git commit -m "api: align m_grouped_moe_gemm_nt_contiguous signature with bf16 asym API"
```
