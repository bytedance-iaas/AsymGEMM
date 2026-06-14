# SM100 BF16 CPU-Left AsymGEMM/SymGEMM for Expert LoRA-A

This document is the scoped implementation and validation plan for the SM100
BF16 CPU-left path. It is separate from the current activation-offload v1 plan.
The activation-offload plan stages CPU activations back to HBM before GPU
GEMMs. This path adds a direct CPU-left math path for the LoRA-A projections
that currently force HBM staging or awkward transposed layouts.

Scope is intentionally narrow:

- Target architecture: **SM100 only**.
- Target dtype/kernel family: **BF16 only**.
- Target operation: grouped expert LoRA-A forward, `A_cpu @ B_cuda.T`.
- Non-targets for the first native patch: SM90, SM89, SM80, FP8, FP4, masked
  layouts, generic GEMM, native backward kernels, and any broad rewrite of
  existing AsymGEMM dispatch.
- Implementation style: add CPU-left-specific functions and bindings. Do not
  change the behavior or body of the current CPU-right kernel entrypoints.

The target use case is memory-first expert LoRA forward:

```text
S_gate = D_gate(X_cpu)   @ A_gate.T   -> [M, r]
S_up   = D_up(X_cpu)     @ A_up.T     -> [M, r]
S_down = D_down(act_cpu) @ A_down.T   -> [M, r]
```

`X_cpu` and `act_cpu` are pinned CPU, route-packed, row-major tensors. `A_*`
are normal trainable CUDA LoRA parameters with shape `[E, r, K]`. The output is
a CUDA row-major `[M, r]` tensor in the same packed row order as the current
expert path.

## Bottom Line

Add an explicit SM100 CPU-left grouped BF16 path instead of trying to use
current CPU-right AsymGEMM through transposes.

This must be a sibling path, not an overload:

- Keep `m_grouped_bf16_asym_gemm_nt_contiguous` as the CPU-right contract.
- Add a new native binding for CPU-left BF16 on SM100.
- Add a new Python helper name for CPU-left LoRA-A.
- Keep the default `grouped_expert_lora()` behavior unchanged in the first
  patch.

Prohibited shortcuts:

- Do not satisfy CPU-left by calling the existing CPU-right function with
  transposed operands.
- Do not stage the full `[M,K]` CPU input to CUDA and then call torch or
  grouped-mm.
- Do not broaden this into SM90, FP8, FP4, masked, or generic GEMM support.
- Do not edit existing CPU-right kernel bodies to make this work.

Definition of done for the first native patch:

- New SM100 BF16 CPU-left binding exists under an explicit CPU-left name.
- Native correctness passes against torch and CPU-right square parity.
- Native latency square parity is measured and close to CPU-right.
- Python helper passes no-staging contract tests.
- Existing SM100 CPU-right BF16 tests still pass unchanged.

Current implementation status in this branch:

- Added `sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous` as a new
  pybind entrypoint. It is additive and SM100-only.
- Added a CPU-left-specific SM100 JIT runtime and sibling kernel:
  `sm100_bf16_cpu_left_asym_gemm_impl`.
- The CPU-left kernel launches CTAs over `(M block, route segment)`, loads the
  pinned CPU-left A tile once per `(M block, K block)`, and iterates over CUDA
  B/N tiles. The CUDA-side B operand is double-buffered across N tiles to mirror
  the existing CPU-right kernel's double-buffered CUDA-side A operand.
- The CPU-left kernel has one persistent SMEM tile for CPU A, not a staged CPU
  A pipeline. The two-stage pipeline is only for CUDA B, which is now the inner
  loop operand. This is the direct mirror of CPU-right: CPU-right keeps one CPU
  B tile resident and pipelines CUDA A across M.
- Stale CPU-left delegation/debug scaffolding has been removed from the native
  CPU-left kernel. The remaining historical references below are profiling
  audit data, not executable paths.
- Added `asym_gemm/training/cpu_left.py` and explicit
  `grouped_expert_lora_cpu_left()` exports. The default
  `grouped_expert_lora()` behavior is unchanged.
- `grouped_expert_lora_cpu_left()` mirrors the high-level CPU-right padding and
  unpadding contract for uneven grouped routes while keeping the padded input
  CPU, pinned, BF16, and contiguous.
- Added correctness, guard, Python no-staging, and opt-in latency validation.
- Existing CPU-right APIs and kernel function bodies remain unchanged.

SM100 NCU performance audit:

```text
shape: M=N=256, K=1024, groups=1, BF16 output

path                    kernel ns  C2C/user bytes  sysmem read sectors  long scoreboard
old delegated CPU-left     33,632       2,099,200              65,536            14.98
CPU-right baseline         26,720         528,384              16,384            10.86
left, M-loop inverted      33,536         526,336              16,384            16.40
left, double-buffered B    27,232         528,384              16,384            11.19
```

Interpretation:

- The original CPU-left wrapper was correct only because it reused the CPU-right
  kernel, but it was B-centric. It loaded the CPU-left A tile once per N tile,
  causing about 4x C2C/sysmem reads for the 256 square case.
- Inverting ownership to `(M block, route segment)` fixed CPU traffic, but a
  single B buffer serialized the inner CUDA-side B TMA loads and left long
  scoreboard stalls high.
- The current CPU-left kernel uses one reusable A tile and two B stages. This
  brings C2C bytes, sysmem sectors, DRAM reads, long-scoreboard stalls, and NCU
  kernel duration to parity with CPU-right.
- For the tiny `128x128x512` two-block case, NCU kernel duration is also at
  parity (`13,248 ns` left vs `13,504 ns` right). CUDA-event timings can show a
  larger ratio because the event is recorded before host-side wrapper work and
  tiny kernels amplify that launch-path gap.

Correctness audit after running the native/helper path on SM100:

- The new CPU-left binding is present and exported in the project `.venv`.
- Aligned group cases pass and match both torch grouped-mm and CPU-right.
- Raw native repeated/uneven route cases fail without padding in the same way as
  raw CPU-right:

```text
lengths = [64, 96, 32, 128]
experts = [2, 0, 2, 1]

CPU-left native vs torch grouped-mm:       9.57000482e-02
raw CPU-right native vs torch grouped-mm:  9.57000482e-02
CPU-left native vs raw CPU-right native:   0.00000000e+00
high-level CPU-right vs torch grouped-mm:  4.06611380e-06
```

This is why the high-level CPU-left helper pads and unpads exactly like the
high-level CPU-right wrapper. The raw native binding remains a block-aligned
kernel surface; the Python helper is the user-facing correctness surface for
normal uneven routes.

## CPU-Left Padding Parity

`grouped_expert_lora_cpu_left()` handles grouped metadata the same way
high-level CPU-right handles grouped BF16 AsymGEMM.

The intended behavior is exact:

1. Accept normal cumulative offsets at the Python boundary:
   `offsets=[0, ... , M]`, `experts=[..., -1]`.
2. If the route groups are already block-aligned, call native directly with no
   extra input allocation.
3. If any group boundary is not block-aligned, create a padded CPU-left input
   tensor whose rows are grouped and zero-padded exactly like CPU-right's
   `_pad_grouped_input_for_asym()` does for CUDA-left inputs.
4. The padded tensor must remain CPU, BF16, contiguous, and pinned. It must not
   be staged to CUDA.
5. Convert the padded cumulative offsets to the existing native pair-offset
   convention with `_group_metadata_tensors()`.
6. Launch `sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous()` on the
   padded CPU tensor and original full CUDA expert weight bank.
7. Unpad the CUDA output back to visible shape `[M, r]` in the original packed
   row order, matching CPU-right's `_unpad_grouped_output()` semantics.

This is the desired comparison target:

```text
CPU-left helper == torch grouped-mm
CPU-left helper == high-level CPU-right wrapper for square/parity cases
```

This is not the desired comparison target for uneven routes:

```text
CPU-left helper == raw CPU-right native
```

Raw native CPU-right has the same scheduler limitation on uneven group
boundaries. The correct user-facing CPU-right behavior is the high-level padded
wrapper, so CPU-left must mirror that behavior.

Concrete implementation constraints:

- Reuse or share CPU-right metadata/padding conventions where possible.
- Keep the default padding block size aligned with CPU-right
  `_pad_grouped_input_for_asym()` unless a measured kernel-specific reason is
  documented in the patch.
- Do not call `x_cpu.to("cuda")`, `x_cpu.cuda()`, CUDA `index_select()` over the
  full `[M, K]` input, or torch grouped-mm from the CPU-left helper.
- Do not pre-gather `weight.index_select(0, active_experts)` for dense expert
  metadata. Native should receive the original CUDA expert weight bank.
- Do not modify existing CPU-right native kernels, CPU-right Python dispatch,
  or default `grouped_expert_lora()` behavior as part of this fix.
- Keep the patch SM100 BF16 CPU-left only.

Acceptance for the immediate fix:

```text
lengths=[64,96,32,128], experts=[2,0,2,1]
CPU-left helper vs torch grouped-mm:       diff < 1e-3
CPU-left helper vs high-level CPU-right:   diff < 1e-3 for square/parity cases
No CUDA allocation with M*K elements from the CPU-left helper
Existing high-level CPU-right tests unchanged and passing
```

## Backward-Compatibility Contract

Yes: if implemented according to this design, existing APIs, kernels,
interfaces, and call sites that use the original AsymGEMM internals should
continue to work unchanged. The CPU-left path is additive and opt-in.

Non-negotiable compatibility requirements:

- Keep existing Python API names, signatures, argument order, defaults, return
  dtypes, and error behavior unchanged for:
  `m_grouped_bf16_asym_gemm_nt_contiguous`,
  `m_grouped_bf16_asym_gemm_nt_masked`,
  `grouped_expert_lora`, `grouped_expert_lora_pair`,
  `AsymFrozenLinear`, and `AsymGroupedFrozenLinear`.
- Keep existing native pybind overloads unchanged, including both
  `m_grouped_bf16_asym_gemm_nt_contiguous(..., list_size: Tensor, ...)` and
  `m_grouped_bf16_asym_gemm_nt_contiguous(..., list_size: int, ...)`.
- Keep existing CPU-right JIT build names, launch wrappers, kernel template
  names, and environment variable semantics unchanged. Do not rename or retune
  existing CPU-right BF16 kernels as part of CPU-left work.
- Add the new binding to `asym_gemm/__init__.py` only as an optional export
  when `_C` exposes it. Importing `asym_gemm` in older builds that do not have
  the CPU-left symbol must still work.
- Do not route existing `AsymGroupedFrozenLinear` or `grouped_expert_lora`
  calls into CPU-left automatically. Existing users should get byte-for-byte the
  same code path unless they call the new CPU-left helper or enable a new
  explicit experimental flag.
- Qwen3 integration must default off. Existing Qwen3/LoRA paths should behave
  exactly as they do today unless `use_cpu_left_lora_a` or an equivalent
  explicit opt-in is set.
- Existing SM90 behavior must remain untouched. The new CPU-left binding should
  fail or be absent on non-SM100; it should not change SM90 dispatch.

Compatibility regression tests that must pass before merging:

```bash
python -m pytest -q tests/m_grouped/test_sm100_bf16_fp8_fp4_m_grouped.py -k bf16
python -m pytest -q tests/test_bf16_asym_gemm.py
python -m pytest -q tests/training/test_cpu_resident_frozen_base.py -k 'grouped_expert_lora or direct_grouped_bf16'
python -m pytest -q tests/training/test_asym_lora_sft_smoke.py -k bf16
```

Compatibility review rule: any change that requires downstream users to rename
a function, pass a new argument, accept a changed default, rebuild different
non-CPU-left kernels, or alter existing BF16 CPU-right behavior is outside this
design and should be rejected.

The first useful API should be:

```python
out = grouped_expert_lora_cpu_left(
    x_cpu,          # pinned CPU [M, K], bf16, contiguous
    a_weight_cuda,  # CUDA [E, r, K], bf16, contiguous
    offsets,
    experts,
)
```

Semantics:

```text
for each active route group g:
    e = experts[g]
    out[offsets[g]:offsets[g + 1]] =
        x_cpu[offsets[g]:offsets[g + 1]] @ a_weight_cuda[e].T
```

This preserves the wanted `[M, r]` orientation and avoids materializing
`X_cpu.T`, `act_cpu.T`, or full `[M, K]` HBM staging tensors.

## Current Code Facts

Current direct AsymGEMM is CPU-right:

- `asym_gemm/training/frozen_linear.py::_dispatch_nt` and
  `_dispatch_grouped_nt` call native `m_grouped_*_asym_gemm_nt_contiguous`.
- The CUDA operand is the left input `a=[M,K]`.
- The CPU-resident operand is the right weight `b_cpu=[N,K]` or
  `b_cpu=[E,N,K]`.
- `transpose_b=True` only changes the logical interpretation of the right CPU
  operand. It does not allow a CPU left operand.
- Direct BF16 currently requires CUDA input, CPU pinned weight, contiguous
  operands, bf16 dtype, SM90/SM100, positive `N/K`, and `N/K` alignment.
- In `csrc/apis/gemm.hpp`, `m_grouped_bf16_asym_gemm_nt_contiguous` dispatches
  to SM90 or SM100 by runtime architecture. This existing function should
  remain CPU-right and should only be used as a reference/regression surface.
- In `csrc/jit_kernels/impls/sm100_bf16_asym_gemm.hpp`, the SM100 wrapper builds
  TMA descriptors for `a`, `b`, and `d`, then launches
  `sm100_bf16_asym_gemm_impl`.
- In `asym_gemm/include/asym_gemm/impls/sm100_bf16_asym_gemm.cuh`, the current
  implementation assumes the existing grouped scheduler and stores output in
  row-major `[M,N]`. This file is useful for understanding the template, but
  the CPU-left patch should not edit the existing kernel function body.
- Current SM100 BF16 coverage lives in
  `tests/m_grouped/test_sm100_bf16_fp8_fp4_m_grouped.py`; it tests the
  CPU-right BF16 binding with a pinned CPU right operand.
- `asym_gemm/training/frozen_linear.py::_direct_grouped_bf16_reason` is the
  existing Python model for named capability failures. A CPU-left helper should
  use the same style, but with SM100-only reasons such as `requires_sm100`,
  `input_not_cpu`, `input_not_pinned`, `weight_not_cuda`, and
  `missing_sm100_cpu_left_bf16_binding`.
- `asym_gemm/training/frozen_linear.py::_group_metadata_tensors` already
  converts cumulative offsets to native pair offsets. The CPU-left helper
  should reuse or share this convention rather than adding a separate metadata
  format.

Current grouped expert LoRA is CUDA grouped-mm based:

- `asym_gemm/training/lora.py::_grouped_lora_torch_mm` does
  `mat1 = x.contiguous()` and passes `selected.transpose(-1, -2)` to
  `torch.nn.functional.grouped_mm` or `torch._grouped_mm`.
- `grouped_expert_lora_pair` does
  `torch.cat((x0.contiguous(), x1.contiguous()), dim=0)`.
- If either operand is CUDA, the helper chooses the CUDA grouped-mm path. A CPU
  activation with CUDA LoRA weights is therefore not a supported direct path.

Current Qwen3 expert flow:

- `asym_gemm/training/qwen3_moe.py::AsymQwen3Experts` owns grouped base weights
  as `[E, 2I, H]` for gate/up and `[E, H, I]` for down.
- With `backend="asym"` and `offload=True`, those base weights are
  `AsymGroupedFrozenLinear` CPU pinned `HostWeight`s.
- LoRA weights remain trainable CUDA parameters:
  `[E,r,H]`, `[E,I,r]`, `[E,r,I]`, and `[E,H,r]`.
- `_forward_gate_up_lora` computes gate/up low-rank tensors, then applies LoRA-B
  through `grouped_expert_lora_pair`.
- `_forward_down_lora` computes down low-rank, then applies down LoRA-B.

Test/style facts from the existing suite:

- `tests/m_grouped/test_sm100_bf16_fp8_fp4_m_grouped.py` uses an SM100 skip
  marker, local `_dense_offsets()`, `_pin_cpu()`, deterministic seeds, and
  `_assert_grouped_close()` with `calc_diff`.
- `tests/training/test_cpu_resident_frozen_base.py` already checks grouped LoRA
  parity, dense expert metadata, direct grouped BF16 forward/dx parity, and
  no-staging stats. CPU-left tests should mirror this style instead of adding a
  separate testing vocabulary.
- `asym_gemm/testing/bench.py::bench_kineto` and `bench()` are already
  available. The CPU-left latency tests should use those helpers so the output
  looks like the rest of AsymGEMM validation.

Code-reading evidence that constrains this design:

| Existing file | Relevant fact | Design implication |
| --- | --- | --- |
| `csrc/apis/gemm.hpp` | BF16 contiguous API checks shape as `[M,K] @ [G,N,K].T`, requires CUDA metadata, and dispatches by `arch_major == 9` or `10`. | Do not overload this function for CPU-left. Add a new SM100-only name with inverted residency checks. |
| `csrc/jit_kernels/impls/sm100_bf16_asym_gemm.hpp` | SM100 wrapper builds `tensor_map_a` from `a` and `tensor_map_b` from `b`, then launches `sm100_bf16_asym_gemm_impl`. | CPU-left may reuse helper concepts, but should use a separate wrapper/template if host-resident `a` needs different descriptor assumptions. |
| `asym_gemm/include/asym_gemm/impls/sm100_bf16_asym_gemm.cuh` | The existing template owns the scheduler/TMA/MMA/store path for CPU-right grouped BF16. | Treat it as reference code; do not edit its existing function body in the first CPU-left patch. |
| `asym_gemm/training/frozen_linear.py` | `_direct_grouped_bf16_reason`, `_group_metadata_tensors`, and `_asym_grouped_bf16_nt` define current Python guards, pair offsets, padding, and native call style. | Mirror the guard style and metadata conversion, but make capability checks SM100 CPU-left-specific. |
| `asym_gemm/training/lora.py` | `grouped_expert_lora()` sends any CUDA operand to CUDA grouped-mm and `grouped_expert_lora_pair()` may index-select active weights. | Add an explicit CPU-left helper; do not silently change generic LoRA dispatch. |
| `tests/m_grouped/test_sm100_bf16_fp8_fp4_m_grouped.py` | Existing SM100 BF16 test pins the right operand and validates against torch. | Add a sibling SM100 CPU-left test file and keep existing CPU-right tests as regression coverage. |

## Why CPU-Right Is the Wrong Fit

The desired gate LoRA-A math for one expert is:

```text
X_cpu_e [m_e, H] @ A_gate_e.T [H, r] -> S_gate_e [m_e, r]
```

Current CPU-right AsymGEMM can only put the CPU tensor on the right. The
transpose workaround is:

```text
A_gate_e [r, H] @ X_cpu_e.T [H, m_e] -> [r, m_e]
```

That creates three problems:

1. The output orientation is `[r, m_e]`, not the route-packed `[m_e, r]` needed
   by the LoRA-B grouped path.
2. For grouped experts, the natural output becomes per-expert/transposed
   fragments, not one packed `[M, r]` tensor.
3. The existing grouped LoRA helpers call `.contiguous()` on the row-major
   `mat1`. Any attempt to pass transposed views for orientation repair can
   materialize the extra HBM tensor this path is intended to avoid.

So `transpose_b=True` in the existing CPU-right path is not enough. We need a
CPU-left contract whose output is already `[M, r]`.

## Target Contracts

### Low-Level Native BF16 API

Add an explicit SM100 binding name rather than silently overloading the current
CPU-right helper:

```python
asym_gemm.sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous(
    a_cpu,       # [M, K], CPU pinned, bf16, contiguous
    b_cuda,      # [E, N, K], CUDA, bf16, contiguous/K-major
    d_cuda,      # [M, N], CUDA, bf16 or fp32, contiguous row-major
    offsets,     # CUDA int32 pair offsets or normalized by Python
    experts,     # CUDA int32 expert ids with sentinel
    list_size,   # CUDA int32 scalar preferred for graph capture
    compiled_dims="nk",
)
```

The native math is still `A @ B.T`; `cpu_left` describes residency, not a
mathematical transpose.

Initial BF16 constraints:

- `a_cpu.device.type == "cpu"` and `a_cpu.is_pinned()`.
- `b_cuda.device.type == d_cuda.device.type == "cuda"`.
- `a_cpu`, `b_cuda`, `d_cuda` are contiguous in their physical contracts.
- `a_cpu.dtype == b_cuda.dtype == torch.bfloat16`.
- `a_cpu.shape == [M, K]`, `b_cuda.shape == [E, N, K]`,
  `d_cuda.shape == [M, N]`.
- `N` is the LoRA rank or packed rank count; require `N % 8 == 0` for the first
  native path.
- `K % 8 == 0`; keep stricter kernel-specific alignment checks in the C++ guard
  if the reused BF16 kernel needs them.
- `offsets`/`experts` describe row groups in the same order used by
  `AsymGroupedFrozenLinear`.
- Runtime arch guard requires `device_runtime->get_arch_major() == 10`; the
  wrapper must fail clearly on non-SM100 instead of falling through to SM90 or
  generic code.

Important native boundary rule:

- The raw native binding consumes already-normalized pair offsets. Like raw
  CPU-right native, it should be tested directly on block-aligned grouped ranges
  and square parity ranges.
- Arbitrary cumulative offsets from real routing, including non-block-aligned
  repeated/uneven groups, are the Python helper contract. The helper must pad
  CPU-left rows and update offsets before native launch.
- Do not call the raw native binding directly with uneven cumulative routing and
  use that as the user-facing correctness target. Existing high-level CPU-right
  also relies on padding before native launch.

The first implementation should not assume the existing BF16 JIT template can
simply swap operand residency. The current Python dispatch is explicitly
CPU-right, and the native grouped CUDA MM APIs require CUDA/contiguous left
inputs. Add a CPU-left-specific wrapper/template path, even if it reuses shared
descriptor helpers. Existing CPU-right functions must stay behaviorally
unchanged and covered by regression tests.

### Scoped Native Change Rules

The first patch may add or register new functions, but should not modify the
body or contract of these existing functions:

- `m_grouped_bf16_asym_gemm_nt_contiguous` in `csrc/apis/gemm.hpp`.
- `sm100_m_grouped_bf16_asym_gemm_contiguous` in
  `csrc/jit_kernels/impls/sm100_bf16_asym_gemm.hpp`.
- `sm100_bf16_asym_gemm_impl` in
  `asym_gemm/include/asym_gemm/impls/sm100_bf16_asym_gemm.cuh`.

Reviewer rule: if a patch changes any of the functions below, it is no longer
the scoped first CPU-left patch and needs a separate design review.

| Existing symbol | File | Allowed in first patch |
| --- | --- | --- |
| `m_grouped_bf16_asym_gemm_nt_contiguous` | `csrc/apis/gemm.hpp` | No body/contract changes. Only nearby registration of the new CPU-left binding is allowed. |
| `m_grouped_bf16_asym_gemm_nt_masked` | `csrc/apis/gemm.hpp` | No changes. CPU-left masked support is out of scope. |
| `sm100_m_grouped_bf16_asym_gemm_contiguous` | `csrc/jit_kernels/impls/sm100_bf16_asym_gemm.hpp` | No changes. Add a sibling CPU-left wrapper instead. |
| `sm100_bf16_asym_gemm_impl` | `asym_gemm/include/asym_gemm/impls/sm100_bf16_asym_gemm.cuh` | No changes. Add a sibling CPU-left template/file if required. |
| `grouped_expert_lora` | `asym_gemm/training/lora.py` | No default CPU-left auto-dispatch in the first patch. |
| `grouped_expert_lora_pair` | `asym_gemm/training/lora.py` | No default behavior change. Optional new CPU-left-specific helper may be added separately. |
| `_dispatch_grouped_nt` / `_asym_grouped_bf16_nt` | `asym_gemm/training/frozen_linear.py` | No CPU-left routing through frozen-base CPU-right dispatch. |

Acceptable first-patch changes are:

- New SM100 CPU-left native wrapper and pybind registration.
- New SM100 CPU-left JIT runtime wrapper/header, if required.
- New CPU-left kernel/template file, if required.
- New Python capability/helper code.
- New tests and optional benchmark scripts.
- Adding the new exported name to `asym_gemm/__init__.py`.

Expected new symbols/files:

| New surface | Purpose |
| --- | --- |
| `sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous` | Native pybind entrypoint for pinned CPU-left BF16 and CUDA-right BF16 on SM100. |
| `sm100_m_grouped_bf16_cpu_left_asym_gemm_contiguous` | C++ wrapper/runtime launch helper if the native pybind should not call the template directly. |
| `sm100_bf16_cpu_left_asym_gemm_impl` | Optional sibling kernel template if the current template cannot safely load host-resident `a_cpu`. |
| `asym_gemm/training/cpu_left.py` | Python capability checks, metadata normalization, and native-call wrapper. |
| `grouped_expert_lora_cpu_left` | Explicit LoRA-A helper for pinned CPU input and CUDA LoRA-A weights. |

### Python Helper API

Add the Python helper in a separate CPU-left utility module or in `lora.py` with
a CPU-left-specific name:

```python
def grouped_expert_lora_cpu_left(
    x_cpu: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    metadata: GroupedLoRAMetadata | None = None,
    compiled_dims: str = "nk",
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    ...
```

Do not make `grouped_expert_lora()` automatically accept CPU-left inputs in the
first patch. An explicit helper makes call sites auditable and prevents silent
fallback to HBM staging.

Expected behavior:

- Returns CUDA `[M, out]` where `out == weight.shape[1]`.
- Supports arbitrary cumulative route offsets by padding the CPU input in pinned
  CPU memory before native launch and unpadding the CUDA output afterward.
- Does not call `x_cpu.to("cuda")`, `x_cpu.cuda()`, or CUDA full-input gather
  operations on the `[M, K]` CPU input.
- May allocate a new padded CPU pinned tensor when route boundaries are not
  block-aligned. For already block-aligned routes, the native call should receive
  the original `x_cpu.data_ptr()`.
- Does not call `weight.index_select()` for the common dense expert case.
- Uses the existing `_group_metadata_tensors` pair-offset convention or a shared
  equivalent helper.
- Records a new stat such as `AsymExecutionStats.cpu_left_lora_a_calls`.

### Dropout Contract

`D_gate`, `D_up`, and `D_down` mean LoRA dropout, including inverted dropout
scale. The staged rollout should handle it in this order:

1. Phase 1 supports `lora_dropout_p == 0.0` only.
2. Phase 2 supports separate gate/up/down dropout masks. The kernel should
   accept an optional packed mask and apply dropout while reading `a_cpu`, so it
   still avoids an HBM dropped-input tensor.
3. Keep RNG ownership in Python/Qwen3 code. The native kernel consumes an
   already-created mask; it should not generate random bits.

For `p == 0`, gate/up can use one packed call:

```python
gate_up_a = torch.cat((gate_lora_A, up_lora_A), dim=1)  # [E, 2r, H]
low_rank = grouped_expert_lora_cpu_left(X_cpu, gate_up_a, offsets, experts)
S_gate, S_up = low_rank.split(r, dim=-1)
```

For `p > 0`, use separate calls unless a mask-aware packed kernel is added.

## Grouped Expert Semantics

The CPU-left path consumes the same packed route order as current Qwen3 experts.

- `offsets` is cumulative `[num_groups + 1]` at the Python boundary.
- `experts` is `[num_groups + 1]` with a final `-1` sentinel.
- Group `g` owns rows `[offsets[g], offsets[g + 1])`.
- Empty groups are legal no-ops.
- The output row order is exactly the input packed row order.
- `dense_experts=True` means one group per expert in expert-id order, but empty
  groups can still exist. The native path should use `experts[g]` directly and
  avoid pre-gathering active weights.
- Routing weights for `forward_input_scaled()` are applied before this helper;
  CPU-left LoRA consumes the already-scaled packed rows.

The helper must not introduce an `[E, M_max, r]` padded public output. Padding
inside the kernel is fine, but the visible output stays `[M, r]`.

## Validation Plan

Validation is a first-class deliverable for this design. The CPU-left path is
only acceptable when correctness, failure guards, no-staging behavior, and
latency are easy to run and easy to interpret.

### Test Files to Add

| File | Purpose |
| --- | --- |
| `tests/m_grouped/test_sm100_bf16_cpu_left.py` | Native binding correctness, guards, route semantics, and CPU-left/CPU-right parity. |
| `tests/m_grouped/test_sm100_bf16_cpu_left_latency.py` | Opt-in latency validation for square and LoRA-rank shapes. Skips unless `ASYM_RUN_PERF_TESTS=1`. |
| `tests/training/test_cpu_left_lora.py` | Python helper correctness, metadata handling, no full-input HBM staging, and error messages. |
| `scripts/lora/profile_cpu_left_bf16_sm100.py` | Manual latency report that prints a small table and optional JSON for review. |

Every new pytest file should skip unless CUDA is available, the current GPU is
SM100, and `asym_gemm.sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous`
exists. Non-SM100 behavior is a skip for tests and a clear runtime error for
the helper.

### Correctness Matrix

Raw native tests should cover:

- Single group: `A_cpu[M,K] @ B_cuda[N,K].T -> D[M,N]`.
- Dense grouped experts with block-aligned row ranges, including empty groups.
- Non-dense route order with repeated expert ids when row ranges are
  block-aligned.
- Pair offsets at the native boundary.
- LoRA ranks `N in {8, 16, 64, 128}`.
- K values aligned to the kernel contract, such as `K in {512, 1024, 4096}`.
- Output dtypes `bf16` and `fp32` if the native path supports both.
- Zero active rows and empty groups as no-ops, with output shape preserved.
- Guard failures for pageable CPU input, CUDA left input, CPU right input,
  non-BF16 operands, non-contiguous operands, non-SM100 arch, unaligned rank,
  and shape mismatches.

Exact native shape matrix:

| Case | Groups | Row counts | N/rank | K | Purpose |
| --- | ---: | --- | ---: | ---: | --- |
| Single LoRA rank | 1 | `[128]` | 8 | 512 | Minimum supported rank and simple reference. |
| Packed gate/up rank | 1 | `[256]` | 16 | 1024 | Represents two rank-8 LoRA-A calls packed as `[E,2r,K]`. |
| Dense grouped | 4 | `[128, 0, 192, 64]` | 16 | 512 | Empty group handling and dense expert ids. |
| Repeated expert route | 4 | `[128, 128, 128, 128]` with experts `[2, 0, 2, 1, -1]` | 64 | 1024 | Non-dense order and repeated expert id with block-aligned ranges. |
| Large K | 2 | `[128, 128]` | 128 | 4096 | Larger hidden dimension and rank alignment. |
| FP32 output | 2 | `[64, 64]` | 64 | 1024 | Optional fp32 output contract. |
| Zero active | 3 | `[0, 0, 0]` | 16 | 512 | No-op launch/output-shape behavior. |
| Square parity | 1 | `[S]`, `S in {128, 256, 512}` | `S` | `{512, 1024, 4096}` | Accuracy and latency symmetry against CPU-right. |

Exact Python helper shape matrix:

| Case | Groups | Row counts | N/rank | K | Purpose |
| --- | ---: | --- | ---: | ---: | --- |
| Repeated uneven route | 4 | `[64, 96, 32, 128]` with experts `[2, 0, 2, 1, -1]` | 64 | 1024 | Required proof that CPU-left padding/unpadding matches torch grouped-mm. |
| Dense uneven route | 4 | `[17, 0, 129, 63]` with experts `[0, 1, 2, 3, -1]` | 16 | 512 | Empty group plus non-block-aligned dense metadata. |
| Already aligned route | 2 | `[128, 128]` with experts `[0, 1, -1]` | 64 | 1024 | Proves no padded input is allocated when unnecessary. |

Do not expand this matrix to FP8/FP4, SM90, masked, or generic GEMM tests in
the first patch. Those are different kernel families.

Concrete native test names:

- `test_cpu_left_single_group_matches_torch_sm100_bf16`
- `test_cpu_left_dense_grouped_matches_torch_with_empty_groups`
- `test_cpu_left_repeated_aligned_expert_route_order_matches_torch`
- `test_cpu_left_accepts_cumulative_and_pair_offsets`
- `test_cpu_left_bf16_and_fp32_outputs_match_torch`
- `test_cpu_left_zero_active_rows_is_noop`
- `test_cpu_left_guard_failures_are_named`
- `test_cpu_left_square_matches_cpu_right_asym_and_torch`

References:

- Native torch reference:
  `torch.cat([a_cpu[s:e].cuda().float() @ b_cuda[eid].float().T for ...])`.
- Current CPU-right AsymGEMM reference on square cases:
  `A_cuda @ B_cpu.T`.
- Python helper reference for real grouped LoRA:
  existing `grouped_expert_lora(A_cuda, B_cuda, offsets, experts)`.
- High-level CPU-right reference for square/parity cases:
  `_asym_grouped_bf16_nt(A_cuda, B_cpu, offsets, experts)`, not raw native
  CPU-right on uneven offsets.

Correctness acceptance:

- `calc_diff(cpu_left_native, torch_ref) < 1e-3` for native block-aligned test
  cases.
- `calc_diff(grouped_expert_lora_cpu_left(...), grouped_expert_lora(...)) <
  1e-3` for arbitrary cumulative route offsets, including repeated/uneven
  groups.
- `calc_diff(cpu_left, cpu_right) < 1e-3` for square CPU-left/CPU-right parity
  cases, where CPU-right is the high-level padded wrapper when grouped offsets
  are involved.
- Invalid rows or empty groups are excluded exactly the same way current
  grouped BF16 tests mask invalid rows.
- No test should silently stage `a_cpu` to CUDA in the helper path.

Guard failure acceptance:

- `a_cpu` is CUDA: fail with `input_not_cpu`.
- `a_cpu` is pageable CPU: fail with `input_not_pinned`.
- `b_cuda` is CPU: fail with `weight_not_cuda`.
- `d_cuda` is CPU: fail with `output_not_cuda`.
- Any operand is not bf16, except optional fp32 output: fail with
  `requires_bf16`.
- Any operand is non-contiguous: fail with `requires_contiguous`.
- Rank or K is unaligned: fail with `requires_8_aligned_nk`.
- Shapes disagree: fail with `shape_mismatch`.
- Current GPU is not SM100: fail with `requires_sm100`.
- Native binding is missing: fail with `missing_sm100_cpu_left_bf16_binding`.

The tests should assert the reason string, not just that some `RuntimeError`
was raised.

### Native Test Sketch

Use local helpers that make the CPU-left and CPU-right comparisons symmetric:

```python
def _cpu_left_ref(a_cpu, b_cuda, offsets, experts):
    chunks = []
    off = offsets.detach().cpu().tolist()
    exp = experts.detach().cpu().tolist()
    for g, e in enumerate(exp[:-1]):
        s, t = int(off[g]), int(off[g + 1])
        if t > s:
            chunks.append(a_cpu[s:t].cuda().float() @ b_cuda[e].float().t())
    return torch.cat(chunks, dim=0).to(torch.bfloat16)

def _run_cpu_left_native(a_cpu, b_cuda, offsets, experts, out_dtype=torch.bfloat16):
    d = torch.empty((a_cpu.shape[0], b_cuda.shape[1]), device="cuda", dtype=out_dtype)
    pair_offsets, experts_i32, list_size = _group_metadata_tensors(
        offsets, experts, device=torch.device("cuda")
    )
    asym_gemm.sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous(
        a_cpu, b_cuda, d, pair_offsets, experts_i32, list_size, compiled_dims="nk"
    )
    return d
```

For the square CPU-right comparison:

```python
S, K = 256, 1024
a = torch.randn((S, K), device="cuda", dtype=torch.bfloat16)
b = torch.randn((1, S, K), device="cuda", dtype=torch.bfloat16)
a_cpu = a.cpu().pin_memory()
b_cpu = b.cpu().pin_memory()
offsets = torch.tensor([0, S], device="cuda", dtype=torch.int32)
experts = torch.tensor([0, -1], device="cuda", dtype=torch.int32)

left = _run_cpu_left_native(a_cpu, b, offsets, experts)
right = torch.empty_like(left)
asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
    a, b_cpu, right, torch.tensor([0, S], device="cuda", dtype=torch.int32),
    experts, torch.tensor([2], device="cuda", dtype=torch.int32), compiled_dims="nk"
)
torch_ref = a.float() @ b[0].float().t()
```

Assertions:

- `calc_diff(left, torch_ref) < 1e-3`.
- `calc_diff(right, torch_ref) < 1e-3`.
- `calc_diff(left, right) < 1e-3`.

### Square Latency Symmetry

The explicit latency sanity check is:

```text
CPU-right existing: D_right = A_cuda[S,K] @ B_cpu[S,K].T
CPU-left new:       D_left  = A_cpu[S,K]  @ B_cuda[S,K].T
```

Use identical `A` and `B` values, both square in the output dimension
`M == N == S`, with the CPU operand pinned and page-touched during warmup.
Both paths perform the same FLOPs and read one BF16 matrix from CPU pinned
memory. On stable warm runs, their latency should be very close.

Latency acceptance:

- Measure kernel-level behavior with NCU when changing the native dataflow. CUDA
  events are useful smoke signals, but for tiny kernels they include host-side
  wrapper gaps if the start event is queued before the Python/C++ call.
- For CUDA-event smoke tests, measure after warmup and after one JIT compile
  run.
- Report median or trimmed mean over at least 30 measured iterations.
- For square cases `S in {128, 256, 512}` and `K in {512, 1024, 4096}`,
  the opt-in event test is a broad regression smoke check:
  `S<=256: ratio <= 1.60`, `S=512: ratio <= 1.25`.
- For NCU kernel duration on the scoped `256x256x1024` case, CPU-left should be
  within roughly 5% of CPU-right and should have equal C2C/sysmem read volume.
- If CI variance is too high, keep the ratio as an opt-in perf assertion under
  `ASYM_RUN_PERF_TESTS=1`; always print the ratio in the manual script.
- Also print `torch_cuda_us` for context, but do not require CPU-left to match
  pure CUDA torch latency.

The manual script should emit rows like:

```text
S    K     groups  left_us  right_us  ratio  diff_torch  diff_right
256  1024  1       123.4    119.8     1.03   4.2e-5      3.9e-5
```

Latency test details:

- Run one unmeasured call first to trigger JIT compilation.
- Synchronize before and after every measured section.
- Page-touch pinned CPU tensors by reading or copying them once before timing.
- Use the same CUDA stream and no host-to-device copies inside measured
  closures.
- Measure CPU-left and CPU-right in alternating order to reduce thermal/drift
  bias.
- Print raw timings even when the ratio assertion fails.
- Keep this perf assertion opt-in; correctness tests must remain fast enough for
  normal development.

Concrete latency test names:

- `test_cpu_left_square_latency_close_to_cpu_right`
- `test_cpu_left_lora_rank_latency_report_only`
- `test_cpu_left_profile_script_prints_diff_and_ratio`

The LoRA-rank latency case should report ranks `8, 16, 64, 128`, but only the
square case should have the strict ratio assertion. Small-rank LoRA is allowed
to be slower initially, as long as the result is correct and visible in the
report.

### Python Helper Tests

`tests/training/test_cpu_left_lora.py` should include:

- `test_grouped_expert_lora_cpu_left_matches_cuda_grouped_lora`
- `test_grouped_expert_lora_cpu_left_repeated_uneven_route_matches_cuda_grouped_lora`
- `test_grouped_expert_lora_cpu_left_dense_metadata_avoids_weight_gather`
- `test_grouped_expert_lora_cpu_left_does_not_stage_full_input`
- `test_grouped_expert_lora_cpu_left_pads_cpu_input_without_cuda_staging`
- `test_grouped_expert_lora_cpu_left_aligned_route_uses_original_input_ptr`
- `test_grouped_expert_lora_cpu_left_empty_groups_match_reference`
- `test_grouped_expert_lora_cpu_left_guard_reasons`
- `test_grouped_expert_lora_cpu_left_records_stats`

For no-staging validation, monkeypatch the native binding with a fake function
that records tensor devices and data pointers, then assert:

- `a_cpu.device.type == "cpu"` at the native call boundary.
- `a_cpu.is_pinned()` is true for the tensor passed to native.
- In the already aligned case, `a_cpu.data_ptr()` matches the original pinned
  input.
- In the repeated/uneven case, `a_cpu.data_ptr()` may differ because padding is
  required, but the tensor must remain CPU pinned and have shape
  `[padded_M, K]`.
- The helper did not create a CUDA tensor with `M * K` or `padded_M * K`
  elements.
- In dense expert metadata, `b_cuda.data_ptr()` matches the original full
  weight bank instead of an `index_select()` result.

This test can run without exercising the real native kernel; it verifies the
Python call contract separately from native correctness. A real-kernel helper
test must still run on SM100 to prove repeated/uneven route correctness against
torch grouped-mm.

### Easy Commands

After build/install:

```bash
python -m pytest -q tests/m_grouped/test_sm100_bf16_cpu_left.py
python -m pytest -q tests/training/test_cpu_left_lora.py
python -m pytest -q tests/m_grouped/test_sm100_bf16_fp8_fp4_m_grouped.py -k bf16
```

Optional latency validation:

```bash
ASYM_RUN_PERF_TESTS=1 python -m pytest -q -s tests/m_grouped/test_sm100_bf16_cpu_left_latency.py
python scripts/lora/profile_cpu_left_bf16_sm100.py --json /tmp/cpu_left_bf16_sm100.json
```

Expected pass/fail interpretation:

- Correctness failure in block-aligned native tests means the native CPU-left
  path is wrong; do not hide it behind fallback.
- Correctness failure in repeated/uneven helper tests means CPU-left
  padding/unpadding is wrong or missing.
- Pageable or unsupported input should fail with a named reason.
- Latency ratio outside the accepted square band means the implementation is not
  yet a credible mirror of CPU-right for the same data movement.
- Existing SM100 CPU-right BF16 tests must still pass, proving the new path did
  not alter current behavior.

### Review Checklist

A reviewer should be able to answer all of these from test output and diff
inspection:

- Does the diff add a new SM100 BF16 CPU-left binding instead of changing
  `m_grouped_bf16_asym_gemm_nt_contiguous`?
- Does the new binding reject non-SM100 devices before launch?
- Does every block-aligned native correctness case print or assert
  `diff_torch < 1e-3`?
- Does the Python helper repeated/uneven route case assert diff against torch
  grouped-mm below `1e-3`?
- Does the square parity case assert `diff_left_right < 1e-3`?
- Does the square latency test print `left_us`, `right_us`, and `ratio`?
- Is the square latency ratio inside the documented opt-in event band, and does
  NCU show kernel-level C2C/sysmem parity against CPU-right?
- Do Python helper tests prove `a_cpu.data_ptr()` reaches native unchanged for
  already aligned routes?
- Do Python helper tests prove repeated/uneven routes pass a CPU pinned padded
  tensor to native rather than a CUDA staged full input?
- Do Python helper tests prove no CUDA tensor with `M * K` elements is created
  by the helper?
- Do guard tests assert exact reason strings?
- Do existing SM100 CPU-right BF16 tests still pass?
- Are there no new SM90, FP8, FP4, masked, or generic GEMM tests/dispatch paths?

Suggested pass summary for the manual profile script:

```text
status  S    K     left_us  right_us  ratio  diff_torch  diff_left_right
PASS    256  1024  123.4    119.8     1.03   4.2e-5      3.9e-5
```

Suggested failure messages:

- `CPU-left correctness failed: diff_torch=... shape=(M,N,K)=...`.
- `CPU-left/right parity failed: diff_left_right=... S=... K=...`.
- `CPU-left latency ratio out of range: ratio=... expected event band for S=...`.
- `CPU-left helper staged full input: saw CUDA allocation with M*K elements`.
- `CPU-left guard reason mismatch: expected requires_sm100, got ...`.

## Implementation Phases

### Phase 0: SM100 Native CPU-Left Binding

Goal: add a new SM100-only BF16 CPU-left native binding with explicit guards and
no behavior change to existing CPU-right functions.

Likely files:

| File | Work |
| --- | --- |
| `csrc/apis/gemm.hpp` | Add `sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous` and pybind registration. Validate CPU-left/CUDA-right residency explicitly. Keep existing BF16 binding untouched. |
| `csrc/jit_kernels/impls/sm100_bf16_cpu_left_asym_gemm.hpp` | New SM100 CPU-left runtime wrapper if the existing wrapper cannot be reused without editing it. |
| `asym_gemm/include/asym_gemm/impls/sm100_bf16_cpu_left_asym_gemm.cuh` | New CPU-left kernel/template if the current template assumes CPU-right behavior internally. |
| `asym_gemm/__init__.py` | Export the new binding only when `_C` exposes it. |
| `tests/m_grouped/test_sm100_bf16_cpu_left.py` | New native parity, guard, and square CPU-left/CPU-right tests. |
| `tests/m_grouped/test_sm100_bf16_cpu_left_latency.py` | New opt-in latency assertions. |
| `scripts/lora/profile_cpu_left_bf16_sm100.py` | New manual latency report. |

Validation:

```bash
python -m pytest -q tests/m_grouped/test_sm100_bf16_cpu_left.py
python -m pytest -q tests/m_grouped/test_sm100_bf16_fp8_fp4_m_grouped.py -k bf16
ASYM_RUN_PERF_TESTS=1 python -m pytest -q -s tests/m_grouped/test_sm100_bf16_cpu_left_latency.py
```

Acceptance:

- Native binding exists only when the extension supports SM100 CPU-left BF16.
- Pinned CPU-left input produces the same `[M, r]` output as the torch reference
  for block-aligned native test cases.
- Square `A_cpu @ B_cuda.T` matches square `A_cuda @ B_cpu.T` in accuracy and
  has latency ratio inside the accepted band.
- Pageable CPU input fails with a clear reason; it must not stage silently.
- Existing SM100 CPU-right BF16 grouped tests still pass.
- Guard and latency tests are explicit enough that an incorrect CPU-left
  implementation cannot pass by falling back to torch, staging `a_cpu`, or
  calling the existing CPU-right function with transposed inputs.

### Phase 1: Python CPU-Left LoRA-A Helper

Goal: expose a minimal Python helper for LoRA-A forward, still independent from
Qwen3 activation offload.

Likely files:

| File | Work |
| --- | --- |
| `asym_gemm/training/cpu_left.py` | New capability checks, metadata normalization, and native-call wrappers. |
| `asym_gemm/training/lora.py` | Add `grouped_expert_lora_cpu_left()` and optional `grouped_expert_lora_pair_cpu_left_a()` for packed gate/up A. |
| `asym_gemm/training/frozen_linear.py` | Add stats fields or move/share metadata helpers if needed. Do not route CPU-left through `AsymFrozenLinearFunction`. |
| `asym_gemm/training/__init__.py` | Export explicit CPU-left helper names. |
| `tests/training/test_cpu_left_lora.py` | New parity, shape, fallback, and no-HBM-staging tests. |

Validation:

```bash
python -m pytest -q tests/training/test_cpu_left_lora.py
python -m pytest -q tests/training/test_cpu_resident_frozen_base.py -k 'grouped_expert_lora or direct_grouped_bf16'
```

Acceptance:

- `grouped_expert_lora_cpu_left(X_cpu, A_cuda, ...)` matches current
  `grouped_expert_lora(X_cuda, A_cuda, ...)`.
- Output is CUDA `[M, r]`.
- The helper does not call `.to("cuda")` on the full `[M, K]` input.
- The helper pads non-block-aligned route groups in pinned CPU memory and
  unpads the CUDA output back to original packed row order.
- The repeated/uneven route case
  `lengths=[64,96,32,128], experts=[2,0,2,1,-1]` passes against torch
  grouped-mm with `calc_diff < 1e-3`.
- Rank-alignment failures name the unsupported rank instead of falling back.
- The helper checks SM100 and BF16 before calling native code.
- Metadata normalization reuses the current pair-offset convention without a
  full CPU round trip for common CUDA metadata.

### Phase 2: Qwen3 Expert Call-Site Integration

Goal: let Qwen3 expert code use SM100 BF16 CPU-left LoRA-A when a CPU owner is
already available, without changing the default all-HBM LoRA path.

Likely files:

| File | Work |
| --- | --- |
| `asym_gemm/training/qwen3_moe.py::_forward_gate_up_lora` | Add an explicit CPU-left branch for `X_cpu` and `lora_dropout_p == 0.0`; pack gate/up A as `[E,2r,H]`. |
| `asym_gemm/training/qwen3_moe.py::_forward_down_lora` | Add CPU-left branch for `act_cpu @ A_down.T`. |
| `asym_gemm/training/qwen3_moe.py::AsymQwen3Experts.__init__` | Add an experimental flag/env gate such as `use_cpu_left_lora_a=False`; keep default off. |
| `tests/training/test_lf_qwen3_asym_backend.py` | Add CPU-left parity tests against the torch backend on synthetic pinned CPU inputs. |

Validation:

```bash
python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py -k 'qwen3_experts and cpu_left'
python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py -k 'qwen3_experts_sm100_backward_matches_torch_backend'
```

Acceptance:

- Existing Qwen3 behavior is unchanged unless the experimental flag is enabled.
- Gate/up CPU-left low-rank tensors match the current CUDA grouped-mm path.
- Down CPU-left low-rank tensor matches the current CUDA grouped-mm path.
- Base `AsymGroupedFrozenLinear` CPU-right execution remains unchanged.
- The branch is disabled for non-SM100, non-BF16, dropout-enabled, or unpinned
  inputs until later phases explicitly support them.

### Phase 3: Dropout and Saved Low-Rank Semantics

Goal: support `lora_dropout_p > 0` without staging dropped inputs to HBM.

Likely files:

| File | Work |
| --- | --- |
| `csrc/apis/gemm.hpp` | Add optional packed-mask arguments or a sibling SM100 BF16 mask-aware CPU-left binding. |
| `asym_gemm/include/asym_gemm/impls/sm100_bf16_cpu_left_asym_gemm*.cuh` | Apply inverted dropout scale during CPU-left tile load if mask-aware native support is chosen. |
| `asym_gemm/training/qwen3_moe.py` | Reuse current packed dropout mask lifecycle; store masks for backward exactly once. |
| `tests/training/test_cpu_left_lora.py` | Add deterministic dropout parity. |
| `tests/training/test_lf_qwen3_asym_backend.py` | Add Qwen3 dropout parity and backward tests. |

Validation:

```bash
python -m pytest -q tests/training/test_cpu_left_lora.py -k dropout
python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py -k 'recompute_lora_dropout or cpu_left'
```

Acceptance:

- CPU-left dropout consumes the same masks as the reference path.
- Backward consumes no extra RNG.
- Saved `S_gate`, `S_up`, and `S_down` remain dropout-applied low-rank tensors.

### Phase 4: Memory-First Expert Path

Goal: wire CPU-left LoRA-A into the future memory-first expert body that owns
`X_cpu` and `act_cpu`.

Likely files:

| File | Work |
| --- | --- |
| `asym_gemm/training/qwen3_moe.py` | Add or extend a custom expert autograd function that passes explicit CPU owners into LoRA-A. |
| `asym_gemm/training/offload.py` | Reuse pinned CPU owner/buffer discipline if a common owner abstraction exists by then. |
| `asym_gemm/profiling/*` | Add counters for avoided HBM staging and CPU-left native calls if profiler support is required. |
| `tests/training/test_lf_qwen3_asym_backend.py` | Add full expert forward/backward parity against torch and current Asym paths. |

Validation:

```bash
python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py -k 'cpu_left and backward'
python -m pytest -q tests/training/test_qwen3_gate_up_windowed_bwd.py
python -m pytest -q tests/lf/test_asym_cpu_adamw_args.py -k offload_modules
```

Acceptance:

- Peak HBM no longer includes full `[M,H]` or `[M,I]` LoRA-A input staging for
  gate/up/down when CPU-left is enabled.
- Low-rank `[M,r]` outputs are still CUDA because LoRA-B and LoRA gradients use
  CUDA trainable parameters.
- LoRA gradients match the current torch backend within existing tolerances.

## Risks and Constraints

- The CPU-left BF16 JIT kernel must stay M-owned: load pinned A once per M/K
  tile and double-buffer CUDA B across N. Accidentally delegating back to the
  CPU-right B-centric loop reintroduces repeated CPU A reads.
- Small ranks can underutilize the current `BLOCK_N` choices. The first path
  should require rank alignment and benchmark rank 8, 16, 64, and 128.
- NCU is the source of truth for native dataflow changes. CUDA-event latency is
  still useful, but small kernels can include launch-path gaps that are not
  kernel body time.
- CPU-left inputs must be pinned and preferably page-touched during setup or
  warmup. Pageable CPU fallback should be a hard error.
- CUDA graph capture should prefer tensor `list_size` and avoid host reads of
  dynamic route metadata.
- Trainable LoRA weights stay CUDA in this design. The helper must not adopt
  them as `HostWeight` or hide them from optimizers.
- Autograd should be explicit. A CPU-left native call used inside Qwen3 expert
  code should be covered by the custom expert backward, not assumed to give
  PyTorch gradients to a CPU activation input.
- Dropout is correctness-sensitive. Do not generate random bits in the native
  GEMM kernel.
- `dA = dS.T @ X_cpu` without HBM staging is a related but distinct gradient
  reduction problem. It can use later CPU-right/K-grouped work, but it should
  not block the first CPU-left LoRA-A forward path.
- Multi-GPU and distributed routing add stream, device, and NUMA placement
  constraints. Keep the first path single-process/single-GPU.

## Explicitly Out of Scope

- Editing the current activation-offload v1 design or making it depend on this
  path.
- Editing existing CPU-right BF16 kernel function bodies to make CPU-left work.
- Any SM90 implementation or SM90 validation requirement.
- Generic CPU-left support for every `grouped_expert_lora()` call site.
- FP8/FP4 CPU-left kernels.
- CPU-resident trainable LoRA parameters or CPUAdam ownership changes.
- Replacing current CPU-right frozen base `AsymFrozenLinear` and
  `AsymGroupedFrozenLinear`.
- A generic saved-tensor-hook activation offloader.
- Async prefetch/overlap, row-window scheduling, or stream pipelining in the
  first CPU-left implementation.
- Non-Qwen3 expert families until the Qwen3 path is validated.
