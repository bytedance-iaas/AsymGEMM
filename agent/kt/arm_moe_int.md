# ARM/aarch64 MoE INT inference adaptation

This document owns the design for porting ktransformers MoE INT inference
kernels to non-x86 ARM/aarch64. It is based on the source checkouts at:

- `KT_ARM_ROOT=/workspace/AsymGEMM-SFT/third_party/ktransformers-arm`
- `KT_ROOT=$KT_ARM_ROOT/kt-kernel`
- `LF_ROOT=$KT_ARM_ROOT/LlamaFactory`
- Repos to clone for the isolated feature workspace:
  - `https://github.com/kvcache-ai/ktransformers.git` into `$KT_ARM_ROOT`
  - `https://github.com/hiyouga/LlamaFactory.git` into `$LF_ROOT`
- Reference-only sources:
  - `KT_REF_ROOT=/workspace/AsymGEMM-SFT/third_party/ktransformers`
  - `KT_REF_KERNEL_ROOT=$KT_REF_ROOT/kt-kernel`
  - `LF_REF_ROOT=/workspace/AsymGEMM-SFT/third_party/LlamaFactory`
  - `ASYM_REF_ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM`
- `AGENT_DOC_ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/kt_adaptation`
- `AGENT_PROBE_REF_ROOT=/workspace/AsymGEMM-SFT/agent/kt_adaptation/probes`
  is a read-only historical probe source. Copy any probe needed for execution or
  editing under `$KT_ARM_ROOT/agent/kt_adaptation/probes` first.

Paths below are relative to `KT_ROOT`, `LF_ROOT`, `AGENT_DOC_ROOT`, or
`$KT_ARM_ROOT/agent/kt_adaptation/probes` unless an absolute path is shown.

The implementation owner should edit ktransformers source, tests, and packaging
only inside `KT_ARM_ROOT`. This document intentionally does not change source
code.

## Current Implementation Status

As of 2026-06-04, the isolated `ktransformers-arm` tree has ARM import/build
hygiene sufficient for the SFT work and now includes the first ARM INT8
inference baseline.

Current proven state:

- The ARM extension can build/import without KML, AMX, AVX, or x86 feature
  variants, and `kt_kernel.__cpu_variant__` reports an ARM variant instead of
  `avx2`.
- The generic `kt_kernel_ext.moe.MOE` binding remains available.
- The SFT document's native `ARMBF16_SFT_MOE` binding is present, but that is
  training BF16 SFT and does not provide INT inference.

Current INT-specific state:

- `kt_kernel_ext.moe.Int8_KERNEL_MOE` is present on ARM in the accepted isolated
  build. It is implemented by the scalar/reference ARM class
  `operators/arm/int8_moe.hpp`, not by KML or x86 `USE_MOE_KERNEL`.
- Python dispatch accepts both existing `method="MOE_INT8"` and explicit
  `method="ARMINT8"`; both route to the ARM-bound `Int8_KERNEL_MOE`.
- Online BF16 tensor loading is supported through
  `GeneralMoEWrapper.load_weights_from_tensors()`, which copies tensors to CPU
  BF16 before native rowwise-symmetric INT8 quantization.
- The existing non-x86 KML path is not usable in this checkout because the
  referenced `operators/kml` and
  `operators/moe_kernel/mat_kernel/kml_kernel` source paths are missing.
- Do not enable `CPU_USE_KML` or treat the missing KML path as the ARM INT8
  solution. The current accepted implementation intentionally bypasses KML and
  binds the existing public class name `Int8_KERNEL_MOE` directly on aarch64.

Validation accepted so far:

- Rebuilt the isolated extension on ARM with KML/AMX/AVX disabled.
- Import smoke reported ARM variant `arm_svebf16`, Python fallback disabled, and
  `kt_kernel_ext.moe.Int8_KERNEL_MOE` present.
- Focused INT8 pytest passed: `test_armint8_inference_reference.py` validates
  both `MOE_INT8` and `ARMINT8` against an FP32 BF16-weight reference.
- Combined focused ARM suite passed: `14 passed` across INT8 inference,
  `ARMBF16_SFT`, `TORCHBF16_SFT`, native parity, lifecycle tests,
  Trainer-style `zero_grad()` grad-view reattachment, and KT fused expert LoRA
  sidecar save/load.
- INT8 smoke benchmark wrote
  `$KT_ARM_ROOT/profiling_kt/bench_armint8_inference_smoke.json`.
- Qwen-shape scalar baseline wrote
  `$KT_ARM_ROOT/profiling_kt/bench_armint8_inference_qwen_shape.json`;
  with `E=128`, `topk=8`, `H=2048`, `I=768`, one CPUInfer thread, observed
  qlen 1 latency `5.84 ms` and qlen 64 latency `368.46 ms`.

Remaining INT inference work is optimization and broader integration, not the
first correctness baseline: add SVE/I8MM tiled kernels, validate packed layouts
against the scalar reference, support file-loaded INT8 weight artifacts if
needed by an inference stack, and run an end-to-end LF or ktransformers
inference smoke once the caller path is selected.

## Isolated implementation rule

Do not develop the ARM INT inference port directly in the user's active
`third_party/ktransformers` or `third_party/LlamaFactory` checkouts. Those
checkouts are read-only references for source inspection. Copy or clone the
needed code under `third_party/ktransformers-arm`, create a separate
`$KT_ARM_ROOT/.venv` or conda env such as `kt-arm`, and install/test from that
environment only.

All ARM INT build products, Python wheels, pybind extensions, benchmark output,
temporary scripts, and LF smoke configs must stay under `KT_ARM_ROOT` until the
port is proven correct and reasonably efficient. If a step below references a
file in `third_party/ktransformers/kt-kernel`, apply the change to the matching
file under `$KT_ROOT`. Do not modify the reference checkout during this proof
phase.

Use physical GPU 3 for any LF or CUDA-assisted smoke associated with this port:
set `CUDA_VISIBLE_DEVICES=3` or `GPU_POOL=3`. Pure CPU INT kernel tests do not
need CUDA, but any optional LF inference smoke should use the copied
Qwen/Qwen3-30B-A3B data under `$LF_ROOT/data`, not the reference LF data files.
Copy the existing Qwen 30B smoke data from `$LF_REF_ROOT/data` into `$LF_ROOT/data`
under a `kt_long_sft_smoke*` prefix when an LF smoke needs realistic inputs.

Recommended fresh bootstrap for the isolated workspace. This creates a separate
ktransformers feature checkout at `$KT_ARM_ROOT` and a separate LLaMA-Factory
checkout inside it, then checks both out to the same commits as the current
local reference trees when those commits are available upstream:

```bash
export KT_ARM_ROOT=/workspace/AsymGEMM-SFT/third_party/ktransformers-arm
export KT_ROOT="$KT_ARM_ROOT/kt-kernel"
export LF_ROOT="$KT_ARM_ROOT/LlamaFactory"
export KT_REF_ROOT=/workspace/AsymGEMM-SFT/third_party/ktransformers
export LF_REF_ROOT=/workspace/AsymGEMM-SFT/third_party/LlamaFactory

mkdir -p "$(dirname "$KT_ARM_ROOT")"
git clone https://github.com/kvcache-ai/ktransformers.git "$KT_ARM_ROOT"
git -C "$KT_ARM_ROOT" checkout "$(git -C "$KT_REF_ROOT" rev-parse HEAD)"
git clone https://github.com/hiyouga/LlamaFactory.git "$LF_ROOT"
git -C "$LF_ROOT" checkout "$(git -C "$LF_REF_ROOT" rev-parse HEAD)"

python3 -m venv "$KT_ARM_ROOT/.venv"
"$KT_ARM_ROOT/.venv/bin/python" -m pip install -U pip setuptools wheel
```

If a reference checkout contains local commits or uncommitted files that are not
available from GitHub, copy those files into the matching path under
`$KT_ARM_ROOT` or `$LF_ROOT` explicitly and record what was copied in the
validation notes. Do not edit the reference trees to make the isolated checkout
work. Do not clone or patch separate `transformers-kt` / `accelerate-kt` source
trees for the INT proof unless a validation failure proves that package-level
source changes are required; they are dependencies, not primary implementation
repos for this port. Do not clone AsymGEMM into the feature workspace; use
`$ASYM_REF_ROOT` only as a read-only source for docs, scripts, and existing
smoke data. Any script copied from AsymGEMM for KT validation must be copied
under `$KT_ARM_ROOT` before editing.

If an LF smoke needs realistic model inputs, copy the existing Qwen 30B smoke
data into the isolated LF tree and rename only the copies:

```bash
mkdir -p "$LF_ROOT/data"
cp "$LF_REF_ROOT/data/dataset_info.json" "$LF_ROOT/data/"
cp "$LF_REF_ROOT/data/lf_dataset_manifest.json" "$LF_ROOT/data/" 2>/dev/null || true
cp "$LF_REF_ROOT/data/asym_long_sft_smoke__qwen3-30b-a3b__s4096.jsonl" \
  "$LF_ROOT/data/kt_long_sft_smoke__qwen3-30b-a3b__s4096.jsonl"
cp "$LF_REF_ROOT/data/asym_long_sft_smoke__qwen3-30b-a3b__s4096__eval.jsonl" \
  "$LF_ROOT/data/kt_long_sft_smoke__qwen3-30b-a3b__s4096__eval.jsonl"
```

After copying, update only the `$LF_ROOT/data` copies of `dataset_info.json` and
`lf_dataset_manifest.json` so every `kt_long_sft_smoke__qwen3-30b-a3b__s4096*`
dataset entry points to the copied
`$LF_ROOT/data/kt_long_sft_smoke__qwen3-30b-a3b__s4096*.jsonl` files. No copied
metadata used by an LF smoke may retain absolute `$LF_REF_ROOT` paths or require
the original `asym_long_sft_smoke*` file names.

The copied `dataset_info.json` must contain keys
`kt_long_sft_smoke__qwen3-30b-a3b__s4096` and
`kt_long_sft_smoke__qwen3-30b-a3b__s4096__eval` whose `file_name` values are the
matching copied `kt_long_sft_smoke*.jsonl` files. The copied
`lf_dataset_manifest.json`, if used by the smoke, must contain matching
`datasets` entries whose `train_name`, `eval_name`, `train_file`, and
`eval_file` all point to the isolated copies under `$LF_ROOT/data`.

After this bootstrap, all INT source edits, package installs, feature branches,
test-only script edits, logs, and profiling outputs belong under `$KT_ARM_ROOT`.
The reference trees are inspected and copied from, not edited.

## Scope

In scope:

- ktransformers CPU MoE INT inference paths for `MOE_INT8` and, after the
  int8 path is correct, `MOE_INT4`.
- Build, import, pybind, runtime dispatch, fallback behavior, tests, and
  benchmark acceptance on aarch64.
- LlamaFactory integration points that select or configure ktransformers.

Out of scope:

- SFT/training kernels, LoRA expert training, and `AMX*_SFT_MOE`; those are
  covered separately by `AGENT_DOC_ROOT/arm_moe_sft.md`.
- CUDA kernels and GPU expert execution.
- Replacing the portable `LLAMAFILE` MoE path.
- Adding source changes in LlamaFactory until the ktransformers ARM backend is
  stable. LLaMA-Factory training/SFT integration is scoped in
  `AGENT_DOC_ROOT/arm_moe_sft.md`.
- Supporting the missing `operators/kml` backend in this checkout.

## Current code map

### Build and packaging

- `CMakeLists.txt`
  - Detects x86 and ARM processors.
  - Enables AMX/AVX only under `HOST_IS_X86`.
  - On ARM/non-MSVC currently appends
    `-march=armv8.2-a+fp16+dotprod+sve+bf16` globally. This is not a safe
    portable baseline because it can produce illegal instructions on ARMv8
    systems without those extensions.
  - Adds `operators/moe_kernel/la` only when
    `KTRANSFORMERS_CPU_MOE_KERNEL=ON`.
  - For non-x86 KML builds, references missing paths in this checkout:
    `operators/kml` and `operators/moe_kernel/mat_kernel/kml_kernel`.

- `setup.py`
  - Exposes x86-oriented environment variables such as
    `CPUINFER_CPU_INSTRUCT=NATIVE|FANCY|AVX512|AVX2`,
    `CPUINFER_ENABLE_AMX`, `CPUINFER_ENABLE_BLIS`, and
    `CPUINFER_ENABLE_KML`.
  - Currently relies too much on vendor strings for ARM detection; on Neoverse
    hosts the vendor can be `unknown`. The ARM path must detect architecture
    first from `platform.machine()` / CMake processor values such as `aarch64`
    or `arm64`, then use vendor strings only as optional diagnostics.
  - Auto-enables `KTRANSFORMERS_CPU_MOE_KERNEL=ON` only when BLIS or KML is
    requested. On this checkout, `CPUINFER_ENABLE_KML=ON` is not buildable
    because the KML sources are absent.
  - Multi-variant extension builds are x86-only.

- `cmake/DetectCPU.cmake` and `cmake/FindSIMD.cmake`
  - Mostly serve x86 SIMD/AMX decisions. ARM feature detection must be added
    separately and must not reuse x86 fallback assumptions.

### Native bindings

- `ext_bindings.cpp`
  - Always binds the generic llamafile implementation as
    `kt_kernel_ext.moe.MOE` via `bind_moe_module<LLAMA_MOE_TP>(..., "MOE")`.
  - Defines `_is_plain_ = true` only under
    `__aarch64__ && CPU_USE_KML`; otherwise `_is_plain_ = false`. A non-KML
    ARM `MOE_KERNEL_TP` build will take the packed-weight path unless this flag
    is decoupled.
  - Under `__x86_64__ && USE_AMX_AVX_KERNEL`, binds AMX classes:
    `AMXInt8_MOE`, `AMXInt4_MOE`, `AMXInt4_1_MOE`,
    `AMXInt4_1KGroup_MOE`, `AMXInt4_KGroup_MOE`, and AVX512-only BF16/FP8/FP4
    classes.
  - Under `__x86_64__`, binds AVX2/AVXVNNI classes:
    `AVX2BF16_MOE`, `AVX2FP8_MOE`, `AVX2GPTQInt4_MOE`,
    `AVX2RawInt4_MOE`, `AVXVNNI256GPTQInt4_MOE`,
    `AVXVNNI256RawInt4_MOE`.
  - Under `USE_MOE_KERNEL`, binds
    `MOE_KERNEL_TP<moe_kernel::GemmKernelInt8, _is_plain_>` as
    `Int8_KERNEL_MOE`. The ARM INT8 task must use this existing class name; it
    is mainly a build/link task for a non-KML mat backend plus guard cleanup,
    not a new pybind class.
  - Binds `Int4_KERNEL_MOE` only under `__aarch64__ && CPU_USE_KML`.
  - Under `__aarch64__ && CPU_USE_KML`, includes `operators/kml/*.hpp`.
    Those headers do not exist in this checkout.

### Python import and wrappers

- `python/__init__.py`
  - Imports `kt_kernel_ext`, exposes `KTMoEWrapper`,
    `generate_gpu_experts_masks`, and `__cpu_variant__`.

- `python/_cpu_detect.py`
  - Valid variants are currently x86-only:
    `amx`, `avx512_bf16`, `avx512_vbmi`, `avx512_vnni`, `avx512_base`,
    `avx2`.
  - `KT_KERNEL_CPU_VARIANT` only accepts those values; an unsupported override
    is currently ignored and detection continues, so the ARM change must turn
    invalid overrides into clear errors instead of silently choosing another
    backend.
  - If detection fails, the fallback is `avx2`. That is wrong on ARM.

- `python/experts.py`
  - `KTMoEWrapper(..., method="MOE_INT8")` and `method="MOE_INT4"` route to
    `GeneralMoEWrapper`, which expects native classes
    `kt_kernel_ext.moe.Int8_KERNEL_MOE` or `Int4_KERNEL_MOE`.
  - `method="LLAMAFILE"` routes to `LlamafileMoEWrapper` and uses
    `kt_kernel_ext.moe.MOE`.
  - `AMXINT8` and `AMXINT4` route to `AMXMoEWrapper` and are x86-only.
  - `RAWINT4`, `GPTQ_INT4`, `FP8`, `FP8_PERCHANNEL`, `BF16`, and `MXFP4`
    route to `NativeMoEWrapper` and currently depend on x86 AMX/AVX bindings.

- `python/utils/moe_kernel.py`
  - Loads and configures the `Int8_KERNEL_MOE` / `Int4_KERNEL_MOE` classes.
  - Error messages should be updated to name ARM build flags once the ARM
    backend is added.

- `python/utils/amx.py`
  - Uses `getattr(...)` to tolerate missing x86 native classes, then raises
    "recompile with AVX512/AVX2" style errors when classes are absent. On ARM,
    those messages must instead say the requested method is x86-only and point
    ARM INT callers to `MOE_INT8` / `MOE_INT4`; otherwise an ARM user is told to
    chase impossible x86 flags.

- `scripts/convert_cpu_weights.py`,
  `scripts/convert_cpu_weights_ds4.py`, and
  `scripts/merge_cpu_weights.py`
  - `convert_cpu_weights.py` and `convert_cpu_weights_ds4.py` recognize
    `moe_int8` / `MOE_INT8` and `moe_int4` / `MOE_INT4`.
  - Those conversion scripts also map bare `int8` / `int4` to `AMXINT8` /
    `AMXINT4` for wrapper conversion; those AMX names should remain
    x86-specific.
  - `merge_cpu_weights.py` detects file prefixes `MOE_INT8_`, `MOE_INT4_`,
    `INT8_`, and `INT4_`; it does not map bare names through AMX wrappers.
  - ARM INT inference should reuse `MOE_INT8` and `MOE_INT4` weight naming
    rather than inventing AMX-like aliases.

### MoE implementations

- `operators/llamafile/moe.hpp`
  - Defines `LLAMA_MOE_TP`, bound as `kt_kernel_ext.moe.MOE`.
  - Uses llamafile/ggml quantized matmul (`llamafile_sgemm`) and ggml quant
    types such as `Q4_K` and `Q6_K`.
  - This is the current portable CPU fallback, but it is not the
    `MOE_INT8`/`MOE_INT4` safetensor kernel path.

- `operators/moe_kernel/moe.hpp`
  - Defines `MOE_KERNEL_TP<T, PLAIN>`, used for `Int8_KERNEL_MOE` and
    `Int4_KERNEL_MOE`.
  - Loads prequantized safetensor weights or online-quantizes BF16 tensors.
  - The forward path uses `forward_unified`, row-quantizes inputs, calls a
    selected `GemmFn`, applies scales and activation, quantizes the down input,
    calls a second `GemmFn`, and merges top-k expert outputs.

- `operators/moe_kernel/la/kernel.hpp`
  - Defines `GemmKernelInt8` and `GemmKernelInt4`.
  - Contains scalar quantization loops with TODO comments for SVE.
  - Expects GEMM entry points from the mat-kernel C API.
  - `PACKED` is currently a global `true` constant from
    `operators/moe_kernel/api/common.h`. Even when `MOE_KERNEL_TP<T, PLAIN>` is
    instantiated with `PLAIN=true`, `BufferB::from_mat(..., if_pack=true,
    plain=true)` stores weights in KT's internal 8-by-32 blocked layout, not in
    ordinary row-major `[n, k]`.
  - Current tiling constants require some dimensions to be divisible by block
    sizes. The first ARM baseline should preserve the existing supported
    dimension contract unless it also changes `BufferB` packing or adds padding.

- `operators/moe_kernel/la/mat_kernel.cpp`
  - Maps `GemmKernelInt8` / `GemmKernelInt4` to these C symbols:
    `decode_cblas_gemm_s8s8s32`,
    `prefill_cblas_gemm_s8s8s32`,
    `decode_int4_cblas_gemm_s8s8s32`,
    `prefill_int4_cblas_gemm_s8s8s32`.

- `operators/moe_kernel/mat_kernel/batch_gemm_api.hpp`
  - Declares the mat-kernel ABI consumed by `la/mat_kernel.cpp`.
  - A new ARM backend must implement the same ABI.
  - The exact C ABI is:

```cpp
void decode_cblas_gemm_s8s8s32(
    const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transa,
    const KERNEL_CBLAS_TRANSPOSE transb, const KERNEL_CBLAS_OFFSET offsetc,
    const size_t m, const size_t n, const size_t k, const float alpha,
    const void* a, const size_t lda, const BLASINT8 oa, const void* b,
    const size_t ldb, const BLASINT8 ob, const float beta, int32_t* c,
    const size_t ldc, const int32_t* oc);

void prefill_cblas_gemm_s8s8s32(
    const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transa,
    const KERNEL_CBLAS_TRANSPOSE transb, const KERNEL_CBLAS_OFFSET offsetc,
    const size_t m, const size_t n, const size_t k, const float alpha,
    const void* a, const size_t lda, const BLASINT8 oa, const void* b,
    const size_t ldb, const BLASINT8 ob, const float beta, int32_t* c,
    const size_t ldc, const int32_t* oc);

void decode_int4_cblas_gemm_s8s8s32(
    const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transa,
    const KERNEL_CBLAS_TRANSPOSE transb, const KERNEL_CBLAS_OFFSET offsetc,
    const size_t m, const size_t n, const size_t k, const float alpha,
    const void* a, const size_t lda, const BLASINT8 oa, const void* b,
    const size_t ldb, const BLASINT8 ob, const float beta, int32_t* c,
    const size_t ldc, const int32_t* oc);

void prefill_int4_cblas_gemm_s8s8s32(
    const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transa,
    const KERNEL_CBLAS_TRANSPOSE transb, const KERNEL_CBLAS_OFFSET offsetc,
    const size_t m, const size_t n, const size_t k, const float alpha,
    const void* a, const size_t lda, const BLASINT8 oa, const void* b,
    const size_t ldb, const BLASINT8 ob, const float beta, int32_t* c,
    const size_t ldc, const int32_t* oc);

void reorder_B_gemm(
    const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transb,
    const size_t k, const size_t n, const size_t ldb, const void* b,
    void* b_reordered);

size_t get_reorder_B_size(
    const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transb,
    const size_t k, const size_t n);
```

- `operators/moe_kernel/mat_kernel/aocl_kernel/kernel.cpp`
  - Useful backend pattern for exported symbols.
  - Implements int8 through AOCL/BLIS.
  - Its int4 symbols throw; do not copy that behavior for a claimed ARM int4
    backend.

### Resolved `BufferB` layout

Local probes against `GemmKernelInt8::BufferB::from_mat` show a critical layout
detail for the first ARM implementation:

- `MOE_KERNEL_TP` constructs `BufferB(..., PACKED, ..., PLAIN)`.
- The intended ARM scalar path sets `PLAIN=true` to avoid AOCL/KML reordered
  `b_pack`, but `PACKED` remains true.
- Therefore the `b` pointer passed to
  `decode_cblas_gemm_s8s8s32` / `prefill_cblas_gemm_s8s8s32` is not an ordinary
  row-major `B[n, k]` matrix.
- `moe.hpp` still calls the GEMM ABI as:

```cpp
cblas_gemm_s8s8s32(
    KernelCblasRowMajor, KernelCblasNoTrans, KernelCblasTrans,
    KernelCblasFixOffset, m_block, n_block, k, 1.0, a_ptr, k, 0,
    b_ptr, k, 0, 0.0, c_ptr, ldc, &oc);
```

For the current int8 `PLAIN=true, PACKED=true` path, the ARM scalar GEMM must
interpret `b_ptr` with KT's blocked accessor:

```cpp
// n_col is 0..n_block-1 relative to the b_ptr block.
// k_idx is 0..k-1.
size_t kt_int8_b_index(size_t n_col, size_t k_idx, size_t k) {
  constexpr size_t PACK_SIZE_N = 8;
  constexpr size_t PACK_SIZE_K = 32;
  return (n_col / PACK_SIZE_N) * PACK_SIZE_N * k
       + (k_idx / PACK_SIZE_K) * PACK_SIZE_N * PACK_SIZE_K
       + (n_col % PACK_SIZE_N) * PACK_SIZE_K
       + (k_idx % PACK_SIZE_K);
}
```

A standard row-major `B[n_col * ldb + k_idx]` implementation is wrong for this
path. The local probe result was:

- standard row-major CBLAS `transb=Trans` access did not match `A @ B.T`;
- the blocked accessor above did match `A @ B.T`;
- non-divisible `n_block` or `k` values can overflow the current packing formula
  unless padding or a true unpacked path is added first.

The first ARM int8 milestone should therefore either:

1. keep `PLAIN=true, PACKED=true` and implement the blocked accessor above in the
   scalar ARM C ABI, or
2. deliberately change the caller/`BufferB` construction to a real unpacked
   `PACKED=false` mode and update save/load tests for that new layout.

Use option 1 for the scoped first milestone because it preserves the existing
`MOE_KERNEL_TP` data path and `.kt` weight layout assumptions. Do not describe it
as ordinary row-major `B`.

The current `PLAIN=true` online-quantize-and-save path has source bugs that must
be fixed before claiming online quantization or `.kt` save/load parity:

1. In `operators/moe_kernel/la/kernel.hpp`,
   `GemmKernelInt8::BufferB::from_mat(...)` calls
   `split_range_n(n, ith, nth, block_size)`, but the `PLAIN=true, if_pack=true`
   constructor path never initializes `block_size`. `MOE_KERNEL_TP::load_weights()`
   calls exactly this path for online BF16 quantization. Initialize/store the
   correct block size for `mat_type='u'`, `'d'`, and `'n'` even when
   `plain=true`, or pass the block size explicitly into `from_mat(...)`. The
   validation must prove gate/up and down online quantization cover the intended
   `n` ranges without relying on uninitialized memory.
2. In `operators/moe_kernel/moe.hpp`, `MOE_KERNEL_TP::load_weights()` saves
   `up_bb_` / `gate_bb_` / `down_bb_` through `b_pack[0]` unconditionally. In the
   ARM scalar `PLAIN=true` path, `b_pack` is empty and the valid pointer is
   `BufferB::b`. The implementation must change each save call to use the
   selected storage:

```cpp
char* saved_b = PLAIN
    ? reinterpret_cast<char*>(bb->b)
    : reinterpret_cast<char*>(bb->b_pack[0]);
write_weights(prefix, suffix, saved_b, expert_idx, size, scale_size);
```

Add a save/load round-trip test for `PLAIN=true, PACKED=true` before relying on
online BF16 quantization to produce reusable `.kt` files.

### ARM-specific optimized layouts

ARM CPU features may prefer different physical layouts than the KT blocked
baseline:

- NEON dot-product may prefer 4-lane or 16-lane interleaving.
- i8mm / SVE-i8mm may prefer MMLA-friendly K/N tiles.
- SVE code must be vector-length agnostic and may need a different tile shape
  than fixed-width NEON.

Those are internal ARM mat-kernel layouts, not new public weight formats. The
contract with the rest of KT must stay explicit:

- The baseline `PLAIN=true, PACKED=true` path consumes the KT 8-by-32 blocked
  `BufferB` layout directly through `load_int8_blocked_b(...)`.
- An optimized ARM layout must be introduced only behind
  `reorder_B_gemm(...)` / `get_reorder_B_size(...)`, with
  `MOE_KERNEL_TP<..., PLAIN=false>` or an equivalent explicit source change that
  makes `BufferB` use `b_pack`.
- The current `batch_gemm_api.hpp` C ABI has no layout-id argument. Therefore,
  if an optimized path uses `PLAIN=false`, the layout id must live in the packed
  `b_pack` buffer written by `reorder_B_gemm`. Add a fixed header before the
  payload, for example:

```cpp
struct ArmPackedBHeader {
  uint32_t magic;      // "KTAI"
  uint16_t version;    // start at 1
  uint16_t layout;     // ArmMoeBLayout value
  uint32_t k;
  uint32_t n_block;
  uint32_t payload_offset;
  uint32_t payload_bytes;
};
```

  `get_reorder_B_size(...)` must include this header and required alignment.
  `reorder_B_gemm(...)` writes the header and payload. Every packed-layout
  accessor and optimized GEMM first validates `magic`, `version`, `layout`,
  `k`, and `n_block`; a mismatch is an error before any numeric computation.
  The scalar reference backend must also read `ArmPackedBHeader` so
  `KT_MOE_ARM_BACKEND=ref` can validate weights loaded or saved in an optimized
  layout.
- Each optimized layout needs a named layout id in the ARM backend, for example
  an internal enum in `operators/moe_kernel/mat_kernel/arm_kernel/layout.hpp`:

```cpp
enum class ArmMoeBLayout {
  KtBlockedPlainInt8,
  NeonDotprodInterleavedInt8,
  I8mmMmlaInterleavedInt8,
  SveI8mmInterleavedInt8,
  KtBlockedInt4,
};
```

- `get_reorder_B_size` must return the exact bytes required by the selected
  packed layout for the current `k` and `n` block.
- `reorder_B_gemm` must transform from KT's logical source passed by
  `BufferB::from_mat` into that selected packed layout.
- The optimized GEMM entry point must reject a layout id it was not compiled to
  support before executing any feature-specific instruction.
- Fallback contract: either an optimized backend consumes the same
  `PLAIN=true` KT blocked `BufferB::b` layout as the scalar path, or its packed
  layout is readable by the scalar reference through `ArmPackedBHeader` and
  `layout.hpp` accessors. Do not implement an optimized layout that can only be
  read by the optimized kernel. After loading optimized-layout weights, forcing
  `KT_MOE_ARM_BACKEND=ref` must still produce correct results or fail before
  load with a clear unsupported-layout error.

Testing for ARM-specific layouts is cheap and required before performance claims:

- Build a small logical `B[n, k]` matrix with sentinel values.
- Pack it through the exact `reorder_B_gemm` implementation for each layout.
- Verify the corresponding scalar/optimized accessor reconstructs every
  `B[col, k]`.
- Verify `A @ B.T` matches a plain reference for decode and prefill blocks.
- Run the same test under forced backend overrides:
  `KT_MOE_ARM_BACKEND=ref|neon_dotprod|i8mm|sve_i8mm`.

Do not reuse the same `.kt` saved files across incompatible physical layouts
unless the file name or metadata records the layout id. Otherwise a checkpoint
saved by one ARM backend can be silently misread by another.

- `operators/amx/*` and `operators/avx2/*`
  - x86-specific and should remain guarded by x86 checks.

### Current KML trap

`operators/moe_kernel/api/common.h` currently auto-defines `CPU_USE_KML` on
ARM:

```cpp
#if defined(__aarch64__) || defined(__arm__) || defined(CPU_USE_KML)
#ifndef CPU_USE_KML
#define CPU_USE_KML
#endif
#endif
```

This is unsafe in this checkout. If `USE_MOE_KERNEL` is enabled on aarch64,
`ext_bindings.cpp` includes `operators/moe_kernel/moe.hpp`, which includes
`api/common.h`, which defines `CPU_USE_KML`; the later
`__aarch64__ && CPU_USE_KML` block can then include missing `operators/kml`
headers. The port must remove this implicit definition or replace it with an
explicit build macro such as `KTRANSFORMERS_CPU_USE_KML` that is set only when
the KML sources are present.

`operators/common.hpp` has additional KML-coupled ARM behavior:

- includes `<arm_sve.h>` under `__aarch64__ && CPU_USE_KML`;
- accepts `GGML_TYPE_F16` conversion only under
  `__aarch64__ && CPU_USE_KML`, otherwise throws
  `"GGML_TYPE_F16 is not supported on this platform"`.

The ARM INT baseline must not depend on those KML branches. Keep the native
`MOE_KERNEL_TP` tests on BF16 tensors, convert F32 to BF16 before native entry
if F32 is accepted at the Python boundary, and add a portable non-KML F16
conversion path before claiming F16 support on ARM.

## LlamaFactory integration points

The LlamaFactory checkout is SFT-oriented for ktransformers:

- `src/llamafactory/extras/packages.py`
  - `is_kt_available()` checks for the `kt_kernel` package.

- `src/llamafactory/extras/misc.py`
  - `use_kt()` checks environment variable `USE_KT`.

- `src/llamafactory/v1/utils/env.py`
  - `use_kt()` currently returns `False`; the v1 launcher path is not the
    active KT integration path for this document.

- `src/llamafactory/hparams/model_args.py`
  - `KTransformersArguments` includes `use_kt`, `kt_weight_path`,
    `kt_expert_checkpoint_path`, `kt_use_lora_experts`,
    `kt_lora_expert_num`, and `kt_lora_expert_intermediate_size`.
  - `apply_kt_config()` writes `ACCELERATE_KT_*` environment variables and
    updates `training_args.hf_kt_config`.

- `src/llamafactory/hparams/parser.py`
  - If `model_args.use_kt` is true, requires `kt-kernel`, `transformers-kt`,
    and `accelerate-kt`.
  - Rejects incompatible combinations such as LoRA reward-model loading and
    DeepSpeed ZeRO-3.

- `src/llamafactory/model/adapter.py`
  - With `use_kt`, only one adapter is accepted and ktransformers is limited to
    LoRA finetuning.

- `examples/ktransformers/accelerate/*.yaml`
  - Current examples use `kt_backend: AMXINT8`, `AMXINT4`, or `AMXBF16`.
  - These names are x86 AMX-specific and should not be reused for ARM.

For SFT/training, ARM support should remain under the existing
`use_kt: true` switch and add new `kt_backend` values such as `TORCHBF16` and
`ARMBF16`; do not add a separate public `use_kt_arm` or `kt-arm` path. See
`AGENT_DOC_ROOT/arm_moe_sft.md` for the full LLaMA-Factory integration
flow.

For INT inference, the direct integration surface is ktransformers
`KTMoEWrapper` with existing method names `MOE_INT8` and `MOE_INT4`. No
LLaMA-Factory source change is required for the ARM INT kernel acceptance gate.
A later LLaMA-Factory inference integration may add examples or parser checks for
those existing method names, but it must not introduce AMX-like ARM aliases or
make INT correctness depend on unavailable patched `transformers-kt` /
`accelerate-kt` packages.

## Target behavior

The port should provide three levels of support:

1. Portable aarch64 build
   - `pip install` succeeds on ARM without KML, AMX, AVX, BLIS, or x86 flags.
   - `kt_kernel_ext.moe.MOE` remains available for `method="LLAMAFILE"`.
   - `MOE_INT8` is either available through the new ARM mat-kernel backend or
     fails with a clear build-time/runtime message naming the missing flag.

2. Correct portable `MOE_INT8` baseline
   - `ext_bindings.cpp` already exposes
     `kt_kernel_ext.moe.Int8_KERNEL_MOE` whenever `USE_MOE_KERNEL` is defined.
     The ARM work is to make `KTRANSFORMERS_CPU_MOE_ARM=ON` define
     `USE_MOE_KERNEL`, link a non-KML implementation of the mat-kernel ABI, and
     prevent `CPU_USE_KML` side effects.
   - The baseline uses scalar C++ loops plus thread partitioning already
     provided by `MOE_KERNEL_TP`; it prioritizes correctness and tail handling
     over speed.
   - The baseline does not require NEON dot-product, i8mm, SVE, BF16 ISA
     instructions, or KML. Its native tensor entry still uses BF16 storage as
     described below.

3. Optional optimized ARM paths
   - Optimized int8 kernels may use NEON dot-product, i8mm, or SVE/SVE2 when
     compile-time and runtime feature checks both pass.
   - Optimized int4 kernels should be implemented only after int8 is correct,
     using a clearly documented nibble layout compatible with
     `GemmKernelInt4::BufferB` and its `int4_2_t` storage.
   - Unsupported features or unsupported shapes fall back to the portable
     scalar backend for that call.

`AMXINT8`, `AMXINT4`, `RAWINT4`, `GPTQ_INT4`, `FP8`, `FP8_PERCHANNEL`, `BF16`,
and `MXFP4` must not silently map to ARM behavior. If a caller requests those
x86 backend names on ARM, keep raising an explicit unsupported-backend error.

## Host facts and priority

The current host is a useful ARM validation machine, but it must not define the
portable wheel baseline:

- architecture: `aarch64`;
- CPU: ARM Neoverse-V2, 144 online CPUs, 2 sockets, 72 cores per socket;
- CPU-populated NUMA nodes: node 0 has CPUs `0-71`, node 1 has CPUs `72-143`;
- relevant flags: `asimd`, `asimddp` for dot-product, `i8mm`, `bf16`, `sve`,
  `sve2`, `svei8mm`, and `svebf16`.

Implementation priority on this host:

1. Build/import and scalar `MOE_INT8` correctness with no KML and no feature
   flags beyond the compiler's normal aarch64 baseline.
2. NEON/ASIMD and dot-product correctness, still with safe runtime dispatch.
3. Optional i8mm and SVE/SVE2/SVE-i8mm variants after scalar and NEON tests are
   stable.

Do not bake Neoverse-V2, SVE, i8mm, or BF16 into the portable wheel. Those
paths must be separately compiled, runtime-gated, and bypassed on weaker ARM
hosts.

## Build design

### CMake options

Add explicit ARM MoE options in `CMakeLists.txt`:

```cmake
option(KTRANSFORMERS_CPU_MOE_ARM
       "Build portable ARM/aarch64 MoE INT mat-kernel backend" OFF)
option(KTRANSFORMERS_CPU_MOE_ARM_OPT
       "Build optional ARM optimized MoE INT mat-kernels" OFF)
option(KTRANSFORMERS_CPU_MOE_ARM_INT4
       "Expose ARM/aarch64 MoE INT4 mat-kernel backend after INT4 tests pass" OFF)
set(KTRANSFORMERS_CPU_MOE_ARM_OPT_LEVEL "baseline" CACHE STRING
    "ARM MoE INT backend: baseline, auto, neon_dotprod, i8mm, or sve_i8mm")
set_property(CACHE KTRANSFORMERS_CPU_MOE_ARM_OPT_LEVEL PROPERTY STRINGS
             baseline auto neon_dotprod i8mm sve_i8mm)
if(NOT KTRANSFORMERS_CPU_MOE_ARM_OPT_LEVEL MATCHES
   "^(baseline|auto|neon_dotprod|i8mm|sve_i8mm)$")
  message(FATAL_ERROR
          "Invalid KTRANSFORMERS_CPU_MOE_ARM_OPT_LEVEL=${KTRANSFORMERS_CPU_MOE_ARM_OPT_LEVEL}; valid values are baseline, auto, neon_dotprod, i8mm, sve_i8mm")
endif()
if(KTRANSFORMERS_CPU_MOE_ARM_INT4 AND NOT KTRANSFORMERS_CPU_MOE_ARM)
  message(FATAL_ERROR
          "KTRANSFORMERS_CPU_MOE_ARM_INT4=ON requires KTRANSFORMERS_CPU_MOE_ARM=ON")
endif()
```

On aarch64, `setup.py` may turn `KTRANSFORMERS_CPU_MOE_ARM=ON` by default once
the scalar backend exists. Until then, it should be opt-in.
Do not default `KTRANSFORMERS_CPU_MOE_ARM_OPT=ON` or
`KTRANSFORMERS_CPU_MOE_ARM_OPT_LEVEL=auto` until the scalar baseline has passing
layout, save/load, and benchmark artifacts. Optimized kernels are opt-in work
for the optimized-int8 subphase.

Option precedence:

- `KTRANSFORMERS_CPU_MOE_ARM_OPT_LEVEL` selects the source set.
- `baseline` means only portable scalar sources are compiled.
- `auto`, `neon_dotprod`, `i8mm`, and `sve_i8mm` require
  `KTRANSFORMERS_CPU_MOE_ARM_OPT=ON`; if the level is non-`baseline` while
  `KTRANSFORMERS_CPU_MOE_ARM_OPT=OFF`, fail CMake configuration with a message
  naming both values.
- `KTRANSFORMERS_CPU_MOE_ARM_OPT=ON` with `OPT_LEVEL=baseline` is allowed but
  compiles no optimized sources; print a status/warning so the build log is not
  misleading.
- CMake must hard-fail invalid `KTRANSFORMERS_CPU_MOE_ARM_OPT_LEVEL` values;
  the cache `STRINGS` property is only UI metadata and is not validation. The
  fatal error must include the full valid-value list.
- CMake must hard-fail `KTRANSFORMERS_CPU_MOE_ARM_INT4=ON` unless
  `KTRANSFORMERS_CPU_MOE_ARM=ON`.
- `setup.py` should derive the boolean from `CPUINFER_ARM_OPT_LEVEL`: pass
  `-DKTRANSFORMERS_CPU_MOE_ARM_OPT=OFF` for `baseline` and
  `-DKTRANSFORMERS_CPU_MOE_ARM_OPT=ON` for every non-`baseline` level. Direct
  CMake users who set inconsistent values get the CMake error above.

When `KTRANSFORMERS_CPU_MOE_ARM=ON`:

- Require an ARM target (`aarch64`, `arm64`, or an equivalent non-MSVC ARM64
  processor value). If this option is set on x86, fail CMake configuration
  immediately instead of compiling a misleading hybrid build.
- Force `KTRANSFORMERS_CPU_MOE_KERNEL=ON`.
- Compile `operators/moe_kernel/la`.
- Compile the new ARM mat-kernel sources.
- Define `USE_MOE_KERNEL=1` and `USE_MOE_KERNEL_ARM=1`.
- Define `USE_MOE_KERNEL_ARM_INT4=1` only when
  `KTRANSFORMERS_CPU_MOE_ARM_INT4=ON` and the scalar INT4 C ABI, pybind class,
  layout tests, and accuracy tests have passed. Until then, INT4 symbols can
  exist only as internal unsupported ABI stubs and `Int4_KERNEL_MOE` must remain
  unbound or clearly unavailable.
- Do not define `CPU_USE_KML`.
- Do not include `operators/kml`.

The force should be explicit in CMake so setup and direct CMake builds behave
the same:

```cmake
if(KTRANSFORMERS_CPU_MOE_ARM)
  set(KTRANSFORMERS_CPU_MOE_KERNEL ON CACHE BOOL
      "ktransformers: CPU use moe kernel" FORCE)
  add_compile_definitions(USE_MOE_KERNEL=1 USE_MOE_KERNEL_ARM=1)
endif()
```

The source selection should mirror the existing AMD/KML mat-kernel structure:

```cmake
if(KTRANSFORMERS_CPU_MOE_KERNEL)
  aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/operators/moe_kernel/la SOURCE_DIR7)
  if(KTRANSFORMERS_CPU_MOE_ARM)
    set(KT_ARM_MOE_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/operators/moe_kernel/mat_kernel/arm_kernel)
    set(SOURCE_DIR7_KERNEL
        ${KT_ARM_MOE_DIR}/kernel.cpp
        ${KT_ARM_MOE_DIR}/reference.cpp)
    if(KTRANSFORMERS_CPU_MOE_ARM_OPT
       AND KTRANSFORMERS_CPU_MOE_ARM_OPT_LEVEL MATCHES "^(auto|neon_dotprod)$")
      list(APPEND SOURCE_DIR7_KERNEL ${KT_ARM_MOE_DIR}/neon_dotprod.cpp)
      set_source_files_properties(${KT_ARM_MOE_DIR}/neon_dotprod.cpp
          PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod")
    endif()
    if(KTRANSFORMERS_CPU_MOE_ARM_OPT
       AND KTRANSFORMERS_CPU_MOE_ARM_OPT_LEVEL MATCHES "^(auto|i8mm)$")
      list(APPEND SOURCE_DIR7_KERNEL ${KT_ARM_MOE_DIR}/i8mm.cpp)
      set_source_files_properties(${KT_ARM_MOE_DIR}/i8mm.cpp
          PROPERTIES COMPILE_OPTIONS "-march=armv8.6-a+i8mm")
    endif()
    if(KTRANSFORMERS_CPU_MOE_ARM_OPT
       AND KTRANSFORMERS_CPU_MOE_ARM_OPT_LEVEL MATCHES "^(auto|sve_i8mm)$")
      list(APPEND SOURCE_DIR7_KERNEL ${KT_ARM_MOE_DIR}/sve_i8mm.cpp)
      set_source_files_properties(${KT_ARM_MOE_DIR}/sve_i8mm.cpp
          PROPERTIES COMPILE_OPTIONS "-march=armv8.6-a+sve+i8mm")
    endif()
    add_compile_definitions(USE_MOE_KERNEL_ARM=1)
  elseif(KTRANSFORMERS_CPU_MOE_AMD)
    aux_source_directory(
      ${CMAKE_CURRENT_SOURCE_DIR}/operators/moe_kernel/mat_kernel/aocl_kernel
      SOURCE_DIR7_KERNEL)
    add_compile_definitions(USE_MOE_KERNEL_AMD=1)
  elseif(NOT HOST_IS_X86 AND KTRANSFORMERS_CPU_USE_KML)
    aux_source_directory(
      ${CMAKE_CURRENT_SOURCE_DIR}/operators/moe_kernel/mat_kernel/kml_kernel
      SOURCE_DIR7_KERNEL)
  endif()
  list(APPEND SOURCE_DIR7 ${SOURCE_DIR7_KERNEL})
  add_compile_definitions(USE_MOE_KERNEL=1)
endif()
```

Do not keep the current global ARM flag
`-march=armv8.2-a+fp16+dotprod+sve+bf16`. Use one of these safer policies:

- Portable baseline: no global `-march`, or at most a conservative
  `-march=armv8-a` when the compiler accepts it.
- Optimized sources: attach feature flags to only those source files or object
  libraries after `check_cxx_compiler_flag` succeeds:
  - NEON dot-product: `-march=armv8.2-a+dotprod`
  - i8mm: `-march=armv8.6-a+i8mm`
  - SVE: `-march=armv8.2-a+sve`
  - SVE i8mm: `-march=armv8.6-a+sve+i8mm`
  - BF16 helpers, only if needed: `-march=armv8.6-a+bf16`

Keep x86 AMX/AVX logic under `HOST_IS_X86`.

### KML guards

Replace KML checks with explicit source availability. Checking only
`operators/kml/moe.hpp` is not enough: the current KML path also references
`operators/kml` headers such as `deepseekv3.hpp` when MLA is enabled and
`operators/moe_kernel/mat_kernel/kml_kernel` subdirectories for GEMM sources.
The required KML path set for this checkout is:

- `operators/kml/moe.hpp`;
- when `KTRANSFORMERS_CPU_MLA=ON`, `operators/kml/deepseekv3.hpp`,
  `operators/kml/gate.hpp`, `operators/kml/mla.hpp`, and
  `operators/kml/mla_int8.hpp`;
- `operators/moe_kernel/mat_kernel/kml_kernel/batch_gemm.cpp`;
- `operators/moe_kernel/mat_kernel/kml_kernel/batch_gemm_kernels.cpp`;
- `operators/moe_kernel/mat_kernel/kml_kernel/prefillgemm/CMakeLists.txt`;
- `operators/moe_kernel/mat_kernel/kml_kernel/prefillgemm_int4/CMakeLists.txt`.

```cmake
set(KT_KML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/operators/kml")
set(KT_KML_KERNEL_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/operators/moe_kernel/mat_kernel/kml_kernel")
set(KT_KML_MLA_HEADERS_PRESENT TRUE)
if(KTRANSFORMERS_CPU_MLA)
  foreach(header deepseekv3.hpp gate.hpp mla.hpp mla_int8.hpp)
    if(NOT EXISTS "${KT_KML_DIR}/${header}")
      set(KT_KML_MLA_HEADERS_PRESENT FALSE)
    endif()
  endforeach()
endif()
set(KT_KML_KERNEL_SOURCES_PRESENT TRUE)
foreach(path
    "${KT_KML_KERNEL_DIR}/batch_gemm.cpp"
    "${KT_KML_KERNEL_DIR}/batch_gemm_kernels.cpp"
    "${KT_KML_KERNEL_DIR}/prefillgemm/CMakeLists.txt"
    "${KT_KML_KERNEL_DIR}/prefillgemm_int4/CMakeLists.txt")
  if(NOT EXISTS "${path}")
    set(KT_KML_KERNEL_SOURCES_PRESENT FALSE)
  endif()
endforeach()
if(KTRANSFORMERS_CPU_USE_KML
   AND EXISTS "${KT_KML_DIR}/moe.hpp"
   AND KT_KML_KERNEL_SOURCES_PRESENT
   AND KT_KML_MLA_HEADERS_PRESENT)
  add_compile_definitions(CPU_USE_KML=1)
  # add KML sources
elseif(KTRANSFORMERS_CPU_USE_KML)
  message(FATAL_ERROR
          "KTRANSFORMERS_CPU_USE_KML=ON but KML sources are not present")
else()
  set(KTRANSFORMERS_CPU_USE_KML OFF)
endif()
```

Then update `operators/moe_kernel/api/common.h` so ARM never implies KML.
`CPU_USE_KML` should mean only "the KML backend was explicitly requested and
the KML source files are present".

For the first ARM backend, delete the current ARM auto-define block or reduce
it to a no-op for non-KML builds:

```cpp
// Do not define CPU_USE_KML from __aarch64__ or __arm__.
// CPU_USE_KML is set only by CMake after KML source presence checks pass.
```

Also audit `operators/common.hpp` after this change. If the ARM backend needs
FP16 tensors, add portable FP16 conversion there without requiring
`CPU_USE_KML`. If it does not, keep FP16 unsupported with a clear error and
validate the ARM INT tests with BF16 native tensors.

For this port's first `MOE_INT8` milestone, the native `MOE_KERNEL_TP` entry is
BF16-only for online hidden states and online expert weights. The current C++
path uses `ggml_bf16_t` inputs and casts `gate_proj`, `up_proj`, and
`down_proj` pointers to `ggml_bf16_t*`. Therefore F32 and FP16 are supported
only if Python converts them to contiguous BF16 before native entry; otherwise
they must be rejected before `MOE_KERNEL_TP` sees their `data_ptr()`. Use clear
messages:

- `MOE_INT8 ARM baseline requires BF16 tensors; convert F32 to BF16 before native entry`
- `MOE_INT8 ARM baseline requires BF16 tensors; FP16 conversion is not enabled`

Do not rely on the current KML-gated FP16 helpers for this behavior. Validation
must include F32 and FP16 cases proving the chosen policy: either they are
converted to BF16 and match the explicitly BF16-converted reference, or they
raise the documented error before kernel launch.

### ARM backend source layout

Add a new backend directory:

```text
operators/moe_kernel/mat_kernel/arm_kernel/
  kernel.cpp          # exports the existing C ABI and dispatches internally
  reference.cpp       # portable scalar int8; int4 stubs until int4 milestone
  reference.hpp
  dispatch.hpp        # feature detection and function pointer table
  layout.hpp          # layout ids and B access/reorder helpers
  neon_dotprod.cpp    # optional internal int8 kernels
  i8mm.cpp            # optional internal int8 kernels
  sve_i8mm.cpp        # optional internal kernels, only when implemented
```

Only `kernel.cpp` should export the C symbols required by
`batch_gemm_api.hpp`. Optimized files should expose internal names to avoid
duplicate symbols.

The baseline build source list is exactly `kernel.cpp` and `reference.cpp`
from this directory. Do not add optional files with `aux_source_directory`;
`neon_dotprod.cpp`, `i8mm.cpp`, and `sve_i8mm.cpp` are compiled only when the
corresponding optimized source level is requested, and only with per-source
feature flags. This is required so `CPUINFER_ARM_OPT_LEVEL=baseline` can run on
plain aarch64 without illegal instructions.

The exported symbols must all exist so `la/mat_kernel.cpp` links:

- `decode_cblas_gemm_s8s8s32`
- `prefill_cblas_gemm_s8s8s32`
- `decode_int4_cblas_gemm_s8s8s32`
- `prefill_int4_cblas_gemm_s8s8s32`
- `reorder_B_gemm`
- `get_reorder_B_size`

For the first int8 milestone, `reorder_B_gemm` and `get_reorder_B_size` are
linkage obligations, not a supported packed-layout contract. Because the
recommended `PLAIN=true, PACKED=true` path reads `BufferB::b` directly, these
two functions should either:

- throw a clear unsupported-layout error if called, or
- implement an explicitly named identity layout that is covered by a test and is
  not selected by default.

Do not silently make `reorder_B_gemm` a byte copy and then let a `PLAIN=false`
path use it. That would reinterpret KT's source layout as a packed optimized
layout without a tested accessor.

With the recommended first-milestone `PLAIN=true, PACKED=true` path, the scalar
int8 implementation in `reference.cpp` must read `B` through the KT blocked
index formula from "Resolved `BufferB` layout", not through row-major CBLAS
storage. Name this helper explicitly, for example:

```cpp
int8_t load_int8_blocked_b(const int8_t* b, size_t n_col, size_t k_idx, size_t k);
```

Unit-test this helper before wiring it into GEMM.

During the int8-only milestone, the two int4 GEMM symbols must still be defined
to satisfy the linked ABI, but they should be explicit unsupported stubs that
throw a clear `"ARM Int4 MoE kernel not implemented"` error if reached.
Because `Int4_KERNEL_MOE` remains unbound on ARM at this stage, normal Python
paths should not reach those stubs. Replace the stubs with real scalar int4 in
the int4 milestone.

Do not leave `_is_plain_ = false` for a non-KML ARM scalar build unless
`reorder_B_gemm`, `get_reorder_B_size`, and packed `b_pack` loading have been
implemented and tested.

### Pybind guards

In `ext_bindings.cpp`:

- Keep `MOE` always bound.
- Keep AMX/AVX bindings x86-only.
- Decouple `_is_plain_` from `CPU_USE_KML`. For the first ARM scalar backend,
  bind `MOE_KERNEL_TP<moe_kernel::GemmKernelInt8, true>` or set
  `_is_plain_ = true` under `USE_MOE_KERNEL_ARM`.
- Do not create a new ARM pybind class name for int8. Keep the existing
  `Int8_KERNEL_MOE` binding under `USE_MOE_KERNEL`; make
  `USE_MOE_KERNEL_ARM` or an equivalent CMake path compile and link the generic
  `MOE_KERNEL_TP<moe_kernel::GemmKernelInt8, ...>` path safely.
- Bind `Int4_KERNEL_MOE` only when a real int4 backend is linked:
  `USE_MOE_KERNEL_ARM_INT4` from `KTRANSFORMERS_CPU_MOE_ARM_INT4=ON`, or
  explicit KML with present sources.
- Do not make `__aarch64__` imply `CPU_USE_KML`.
- `kt_kernel_ext.moe.tiling.get_int4` / `set_int4` / `set_all` are currently
  exposed under the broad `USE_MOE_KERNEL` guard, even when `Int4_KERNEL_MOE`
  is not bound. Either guard those int4 tiling helpers with the same int4
  backend macro or document them as inert configuration helpers; tests must use
  the class binding, not tiling helper presence, to decide whether ARM int4 is
  available. If `set_all` remains exposed without `Int4_KERNEL_MOE`, it must not
  make Python report INT4 support.

A safe first-pass guard shape is:

```cpp
#if defined(USE_MOE_KERNEL_ARM)
static const bool _is_plain_ = true;
#elif defined(__aarch64__) && defined(CPU_USE_KML)
#if defined(KTRANSFORMERS_CPU_MLA)
#include "operators/kml/deepseekv3.hpp"
#include "operators/kml/gate.hpp"
#include "operators/kml/mla.hpp"
#include "operators/kml/mla_int8.hpp"
#endif
#include "operators/kml/moe.hpp"
static const bool _is_plain_ = true;
#else
static const bool _is_plain_ = false;
#endif
```

### setup.py

Add ARM-specific env handling in `setup.py`:

- `CPUINFER_ENABLE_ARM_MOE=ON|OFF`
  - Forwards to `-DKTRANSFORMERS_CPU_MOE_ARM=ON|OFF`.
- `CPUINFER_ENABLE_ARM_MOE_INT4=ON|OFF`
  - Forwards to `-DKTRANSFORMERS_CPU_MOE_ARM_INT4=ON|OFF`.
  - Defaults to `OFF`; setup.py must refuse `ON` unless
    `CPUINFER_ENABLE_ARM_MOE=ON` is also set.
- `CPUINFER_ARM_OPT_LEVEL=baseline|auto|neon_dotprod|i8mm|sve_i8mm`
  - Forwards a CMake cache value such as `KTRANSFORMERS_CPU_MOE_ARM_OPT_LEVEL`.
  - Also forwards `KTRANSFORMERS_CPU_MOE_ARM_OPT=OFF` for `baseline` and
    `KTRANSFORMERS_CPU_MOE_ARM_OPT=ON` for every non-`baseline` value.
  - Invalid values must fail in setup.py before CMake is invoked, with the same
    valid-value list as the CMake check.
  - `baseline` must compile without extension flags.
  - `auto` may compile multiple optimized source files and runtime-dispatch.
- `CPUINFER_CPU_INSTRUCT=ARM_BASELINE`
  - On ARM, forwards `-DLLAMA_NATIVE=OFF` and no x86 `LLAMA_AVX*` flags.
  - This is the required mode for portable baseline validation. A host-specific
    `NATIVE` wheel may be tested later, but it must not be the correctness
    baseline.

On ARM:

- `CPUINFER_CPU_INSTRUCT` must not emit x86 flags.
  - Move CPU detection before `cmake_args += cpu_feature_flags()`, or make
    `cpu_feature_flags()` accept detected CPU info and return no x86 feature
    flags on ARM except a safe host-specific setting when explicitly requested.
- Multi-variant x86 extension builds must be skipped.
- Do not auto-enable `CPUINFER_ENABLE_KML`.
- If `CPUINFER_ENABLE_ARM_MOE=ON`, also pass
  `-DKTRANSFORMERS_CPU_MOE_KERNEL=ON`.

Terminology: `CPUINFER_ARM_OPT_LEVEL=baseline` is the build configuration for
the portable scalar backend. `KT_MOE_ARM_BACKEND=ref` is the runtime override
name for that same scalar backend. Do not create separate "baseline" and "ref"
implementations.

## Runtime dispatch

Use a single ARM extension and dispatch inside the ARM mat-kernel backend.
This avoids needing separate Python extension files per ARM feature level.

Feature detection should be Linux-first:

- Prefer `getauxval(AT_HWCAP)` / `getauxval(AT_HWCAP2)` in C++ with
  `<sys/auxv.h>` and `<asm/hwcap.h>` or the platform-equivalent hwcap header.
  - Dot-product: `HWCAP_ASIMDDP`.
  - SVE: `HWCAP_SVE`.
  - SVE2: `HWCAP2_SVE2`.
  - i8mm: `HWCAP2_I8MM`.
  - SVE i8mm, for `sve_i8mm`: `HWCAP2_SVEI8MM`.
  - BF16, if used: `HWCAP2_BF16`.
  - SVE BF16, if used: `HWCAP2_SVEBF16`.
- Accept an override environment variable, for example
  `KT_MOE_ARM_BACKEND=auto|ref|neon_dotprod|i8mm|sve_i8mm`.
- Treat unknown overrides as errors with valid choices.
- `KT_MOE_ARM_BACKEND=auto` may fall back to `ref` when optimized hardware or
  compiled support is unavailable.
- Forced optimized overrides such as `neon_dotprod`, `i8mm`, or `sve_i8mm` must
  fail before launch when the binary or runtime CPU features do not support that
  backend; they must not silently fall back to `ref`.
- Never execute an optimized kernel unless both the binary was compiled for it
  and runtime CPU features are present.

Python `_cpu_detect.py` still needs ARM awareness so import diagnostics and
`KT_KERNEL_CPU_VARIANT` are sane. Add `import platform`, reject invalid
`KT_KERNEL_CPU_VARIANT` overrides with a message listing valid values, and add
valid variants:

- `arm_baseline`
- `arm_neon_dotprod`
- `arm_i8mm`
- `arm_sve_i8mm`

Detect ARM via `platform.machine()` values such as `aarch64` and `arm64`, and
parse `/proc/cpuinfo` `Features` for:

- baseline NEON/ASIMD: `asimd`
- dot-product: `asimddp`
- i8mm: `i8mm`
- SVE: `sve`
- SVE i8mm: `svei8mm`
- BF16: `bf16`

If no optimized ARM features are found, report `arm_baseline`. Do not fall back
to `avx2` on ARM.

Also update `load_extension()` fallback behavior. On ARM, its fallback chain
must be ARM-only, for example
`arm_sve_i8mm -> arm_i8mm -> arm_neon_dotprod -> arm_baseline -> single-variant`.
If the implementation uses a single extension file for ARM, it may load
`kt_kernel_ext.*.so` directly after reporting the detected ARM variant for
diagnostics. It must not try `_kt_kernel_ext_avx2.*.so` on ARM.

## C++ implementation details

### Int8 baseline

Implement int8 first. The scalar baseline must support both decode and prefill:

- Inputs:
  - `A`: signed int8 activation tile from `GemmKernelInt8::BufferAImpl`.
  - `B`: signed int8 quantized weights from `GemmKernelInt8::BufferB`, using the
    KT 8-by-32 blocked layout described in "Resolved `BufferB` layout" when the
    first milestone keeps `PLAIN=true, PACKED=true`.
  - Output accumulator: signed int32.
- Operation:
  - Compute logical `C[row, col] = sum_k int32(A[row, k]) *
    int32(B_logical[col, k])`.
  - `A` is ordinary row-major with `lda == k`.
  - `C` is ordinary row-major, but `ldc` is the full destination leading
    dimension (`intermediate_size` for up/gate, `hidden_size` for down), not
    necessarily the current `n_block`.
  - `B_logical[col, k]` must be loaded through the KT blocked index for the
    current `PLAIN=true, PACKED=true` path. Do not use plain
    `B[col * ldb + k]` unless the implementation first changes `BufferB` to
    `PACKED=false` and tests that path.
  - Support `m_block` tails; `moe.hpp` already passes the final partial local
    expert token block as a smaller `m_block`.
  - For the first milestone, keep existing `n`/`k` dimension restrictions instead
    of silently pretending arbitrary tails work. Current `BufferB` packing
    assumes `k % 32 == 0`, and `recommended_nth_up_gate/down` require `n` to be a
    multiple of `N_BLOCK_UP_GATE`/`N_BLOCK_DOWN` for the selected decode/prefill
    mode. If non-divisible `n` or `k` support is required, add padding or a true
    unpacked `PACKED=false` path before enabling those tests.
  - `MOE_KERNEL_TP::forward_unified()` is generic. Today it uses
    `N_BLOCK_UP_GATE_PREFI` / `N_BLOCK_DOWN_PREFI` to compute prefill `nth`, but
    several B pointer offsets, GEMM `n` values, C offsets, output offsets, and
    scale blocks still use decode `N_BLOCK_UP_GATE` / `N_BLOCK_DOWN`. For the
    first ARM INT8 milestone, require
    `N_BLOCK_UP_GATE_PREFI == N_BLOCK_UP_GATE` and
    `N_BLOCK_DOWN_PREFI == N_BLOCK_DOWN`. If different prefill tile sizes are
    needed, first make `forward_unified()` carry the selected up/gate and down
    block sizes through every offset, GEMM, output, and scale calculation, then
    add decode-vs-prefill tiling tests before enabling that configuration.
- Minimum ABI subset used by `MOE_KERNEL_TP` today:
  - `moe.hpp` calls the selected int8/int4 `GemmFn` as
    `KernelCblasRowMajor`, `KernelCblasNoTrans`, `KernelCblasTrans`,
    `KernelCblasFixOffset`, `alpha=1.0`, `oa=0`, `ob=0`, `beta=0.0`,
    `oc=&0`.
  - `GemmKernelInt8::BufferB` packing calls `reorder_B_gemm` with
    `KernelCblasColMajor`, `KernelCblasNoTrans`, `ldb=k`.
  - `get_reorder_B_size` is called with `KernelCblasRowMajor`,
    `KernelCblasNoTrans`.
  - The scalar backend may either implement the full enum space or explicitly
    validate this subset and throw a clear unsupported-argument error for other
    enum/offset combinations.
- Threading:
  - Start with the thread partitioning already done by `MOE_KERNEL_TP`.
  - Do not add a second global thread pool in the scalar backend.

`GemmKernelInt8` currently has block divisibility checks and packing assumptions.
Do not leave shape restrictions undocumented. The first milestone should document
supported `hidden_size` / `intermediate_size` multiples and add negative tests for
unsupported non-divisible dimensions. A later milestone may relax those
restrictions by adding padding or changing `BufferB` layout, but that must be a
deliberate source change with save/load compatibility tests.

### Int4 baseline

Implement int4 only after int8 is green.

Requirements:

- Use the exact nibble order from `GemmKernelInt4::BufferB::from_mat`.
  For each `n_col` and 64-wide K group:
  - one byte stores two signed 4-bit values for the same `n_col`;
  - the high nibble stores `k_start + j`, where `j` is `0..31`;
  - the low nibble stores `k_start + j + 32`;
  - the byte offset relative to the selected B block is:

```cpp
size_t kt_int4_b_byte_index(size_t n_col, size_t k_idx, size_t k) {
  constexpr size_t PACK_SIZE_N = 8;
  constexpr size_t PACK_SIZE_K = 32;
  const size_t k_start = (k_idx / (PACK_SIZE_K * 2)) * (PACK_SIZE_K * 2);
  const size_t j = (k_idx - k_start) % PACK_SIZE_K;
  return (n_col / PACK_SIZE_N) * PACK_SIZE_N * (k / 2)
       + (k_start / (PACK_SIZE_K * 2)) * PACK_SIZE_N * PACK_SIZE_K
       + (n_col % PACK_SIZE_N) * PACK_SIZE_K
       + j;
}
```

  Decode the nibble to the int8-equivalent value used by KT accumulation, not to
  the raw `[-8, 7]` nibble. `GemmKernelInt4::BufferB::from_mat` computes
  `b0 = round(f0 / (d * 16.0)) * 16`, `b1 = round(f1 / (d * 16.0)) * 16`, then
  stores `(b0 & 0xF0) | ((b1 >> 4) & 0x0F)`, with `d = amax / 112.0`.

```cpp
int8_t int4_nibble_to_int8_scaled(uint8_t nibble) {
  int v = static_cast<int>(nibble & 0x0F);
  if (v >= 8) {
    v -= 16;
  }
  return static_cast<int8_t>(v * 16);
}

int8_t load_int4_blocked_b(const int8_t* b, size_t n_col, size_t k_idx, size_t k) {
  const uint8_t byte =
      static_cast<uint8_t>(b[kt_int4_b_byte_index(n_col, k_idx, k)]);
  return (k_idx % 64) < 32
      ? int4_nibble_to_int8_scaled(byte >> 4)
      : int4_nibble_to_int8_scaled(byte);
}
```

- Implement decode and prefill symbols without throwing.
- Support signed int4 values consistently with existing quantization and scale
  logic in `GemmKernelInt4`, including the current `d = amax / 112.0` scale.
- Preserve the existing shape contract: `GemmKernelInt4::BufferB` requires
  `k % 64 == 0` because each byte spans a 64-wide K group. It also inherits
  the `GemmKernelInt4` `n` tiling checks: up/gate `intermediate_size` must be a
  multiple of `N_BLOCK_UP_GATE` for decode and `N_BLOCK_UP_GATE_PREFI` for
  prefill, and down `hidden_size` must be a multiple of `N_BLOCK_DOWN` for
  decode and `N_BLOCK_DOWN_PREFI` for prefill. With current defaults those are
  256 for up/gate and 1024 for down. If smaller INT4 validation shapes are used,
  first call the existing int4 tiling setter (`set_int4` /
  `GemmKernelInt4::set_tiling`) and validate that override explicitly. Add
  negative tests for
  non-divisible `k` and `n` unless padding or a new layout is deliberately
  implemented.
- Current `MOE_KERNEL_TP::forward_unified()` is generic and has the same
  prefill/decode tile-size mismatch risk for both INT8 and INT4: it uses
  `N_BLOCK_UP_GATE_PREFI` / `N_BLOCK_DOWN_PREFI` to compute `nth` for prefill,
  but several pointer offsets, GEMM `n` arguments, C offsets, output offsets,
  and scale blocks still use the decode `N_BLOCK_UP_GATE` / `N_BLOCK_DOWN`
  values. Until that forward path is made fully mode-specific, require
  `N_BLOCK_UP_GATE_PREFI == N_BLOCK_UP_GATE` and
  `N_BLOCK_DOWN_PREFI == N_BLOCK_DOWN` for all ARM INT paths. If different
  prefill tile sizes are needed, first change the forward path to carry the
  selected block size through every pointer offset/GEMM/output/scale calculation
  and add a prefill-vs-decode layout test.
- `GemmKernelInt4::BufferB::from_mat(...)` currently calls `split_range_n(...)`
  with the generic `N_BLOCK`, while `MOE_KERNEL_TP::load_weights()` schedules
  work using `recommended_nth_up_gate(...)` and `recommended_nth_down(...)`.
  Before exposing ARM INT4, fix `from_mat(...)` so it uses the selected
  up/gate/down block size for the current `mat_type`, or pass that block size
  explicitly. Add full-row sentinel validation proving online INT4 quantization
  covers every expected `n` row exactly once for gate, up, and down.
- Add tests that compare both online-quantized BF16 weights and prequantized
  weight loading.

Until this is implemented, do not bind `Int4_KERNEL_MOE` on ARM. A missing
class with a clear error is better than a class whose first call throws inside
the backend.

### Optimized kernels

After scalar int8 correctness:

- NEON dot-product path:
  - Use `sdot` intrinsics only in files compiled with dot-product support.
  - Runtime gate on `asimddp`.
  - Fall back to scalar for unsupported tails until tail kernels are tested.

- i8mm path:
  - Use i8mm intrinsics only in files compiled with i8mm support.
  - Runtime gate on `i8mm`.
  - Prefer prefill and large blocks first; decode may need a separate kernel.

- SVE/SVE2 path:
  - Compile separately.
  - Runtime gate on `sve`; gate SVE2 code on `sve2`, SVE-i8mm code on
    `svei8mm`, and SVE-BF16 code on `svebf16`.
  - Do not assume a fixed vector length.

For each optimized path, add a self-report string returned in benchmark logs
and optionally exposed through a small debug binding or environment log. The
benchmark output must state whether it used `ref`, `neon_dotprod`, `i8mm`, or
`sve_i8mm`.

## Python behavior

`python/experts.py` should keep method semantics explicit:

- `LLAMAFILE`
  - Uses `kt_kernel_ext.moe.MOE`.
  - Portable fallback when users have GGUF/ggml-compatible quantized weights.
  - It is not a transparent fallback for `MOE_INT8` safetensors.

- `MOE_INT8`
  - Uses `kt_kernel_ext.moe.Int8_KERNEL_MOE`.
  - On ARM, works only when built with `KTRANSFORMERS_CPU_MOE_ARM=ON`.
  - If the class is absent, raise an error naming
    `CPUINFER_ENABLE_ARM_MOE=ON` and `KTRANSFORMERS_CPU_MOE_ARM=ON`.

- `MOE_INT4`
  - Uses `kt_kernel_ext.moe.Int4_KERNEL_MOE`.
  - On ARM, should remain unavailable until the int4 backend is complete.

- `AMXINT8` / `AMXINT4`
  - Remain x86 AMX names and should raise on ARM with an x86-only message, not
    a suggestion to recompile ARM with AVX512.

- `RAWINT4`, `GPTQ_INT4`, `FP8`, `FP8_PERCHANNEL`, `BF16`, and `MXFP4`
  - Remain x86 AMX/AVX native-wrapper names in this checkout. On ARM, errors
    should identify them as unavailable x86-oriented backends unless a future
    ARM design explicitly ports one of those formats.

Do not silently substitute `LLAMAFILE` for `MOE_INT8` or substitute ARM kernels
for AMX names. Those paths use different weight formats and accuracy/perf
expectations.

## Validation plan

Run validation on a real ARM/aarch64 host. Also keep x86 CI green by ensuring
all new ARM code is behind ARM guards.

### Build smoke

On this workspace machine, first confirm source assumptions and ARM host
identity. Before the implementation exists this does not certify the future ARM
backend, but it catches stale paths and accidental KML assumptions:

```bash
KT_ARM_ROOT=/workspace/AsymGEMM-SFT/third_party/ktransformers-arm
KT_ROOT=$KT_ARM_ROOT/kt-kernel
uname -m
test ! -d "$KT_ROOT/operators/kml"
test ! -d "$KT_ROOT/operators/moe_kernel/mat_kernel/kml_kernel"
"$KT_ARM_ROOT/.venv/bin/python" - <<'PY'
from pathlib import Path

kt = Path("/workspace/AsymGEMM-SFT/third_party/ktransformers-arm/kt-kernel")
for rel in [
    "ext_bindings.cpp",
    "operators/moe_kernel/api/common.h",
    "operators/moe_kernel/la/mat_kernel.cpp",
    "operators/moe_kernel/mat_kernel/batch_gemm_api.hpp",
    "python/_cpu_detect.py",
    "python/experts.py",
]:
    assert (kt / rel).exists(), rel
print("source map ok")
PY
```

After adding the KML source-presence guard, force KML once and prove it fails
with the intended fatal error before any missing include or `add_subdirectory`
is reached:

```bash
KT_ARM_ROOT=/workspace/AsymGEMM-SFT/third_party/ktransformers-arm
cd "$KT_ARM_ROOT/kt-kernel"
mkdir -p "$KT_ARM_ROOT/build" "$KT_ARM_ROOT/profiling_kt/int"
rm -rf "$KT_ARM_ROOT/build/kt-kml-fail-check"
set +e
cmake -S . -B "$KT_ARM_ROOT/build/kt-kml-fail-check" \
  -DKTRANSFORMERS_CPU_USE_KML=ON \
  -DKTRANSFORMERS_CPU_MOE_KERNEL=ON \
  -DKTRANSFORMERS_CPU_MLA=ON \
  2>&1 | tee "$KT_ARM_ROOT/profiling_kt/int/kt-kml-fail-check.log"
status=${PIPESTATUS[0]}
set -e
test "$status" -ne 0
grep -F "KTRANSFORMERS_CPU_USE_KML=ON but KML sources are not present" \
  "$KT_ARM_ROOT/profiling_kt/int/kt-kml-fail-check.log"
```

After source changes, this aarch64 machine should run the generic import smoke:

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers-arm/kt-kernel
CPUINFER_USE_CUDA=0 \
CPUINFER_CPU_INSTRUCT=ARM_BASELINE \
/workspace/AsymGEMM-SFT/third_party/ktransformers-arm/.venv/bin/python -m pip install -v .
/workspace/AsymGEMM-SFT/third_party/ktransformers-arm/.venv/bin/python - <<'PY'
import kt_kernel
import kt_kernel_ext
print("variant:", kt_kernel.__cpu_variant__)
print("MOE:", hasattr(kt_kernel_ext.moe, "MOE"))
PY
/workspace/AsymGEMM-SFT/third_party/ktransformers-arm/.venv/bin/pytest \
  test/per_commit/test_basic_cpu.py
```

On ARM:

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers-arm/kt-kernel
CPUINFER_USE_CUDA=0 \
CPUINFER_CPU_INSTRUCT=ARM_BASELINE \
CPUINFER_ENABLE_ARM_MOE=ON \
CPUINFER_ARM_OPT_LEVEL=baseline \
/workspace/AsymGEMM-SFT/third_party/ktransformers-arm/.venv/bin/python -m pip install -v .
```

Then verify imports:

```bash
/workspace/AsymGEMM-SFT/third_party/ktransformers-arm/.venv/bin/python - <<'PY'
import kt_kernel
import kt_kernel_ext

print("variant:", kt_kernel.__cpu_variant__)
print("MOE:", hasattr(kt_kernel_ext.moe, "MOE"))
print("Int8:", hasattr(kt_kernel_ext.moe, "Int8_KERNEL_MOE"))
print("Int4:", hasattr(kt_kernel_ext.moe, "Int4_KERNEL_MOE"))
PY
```

Expected baseline result:

- `MOE: True`
- `Int8: True` once the ARM int8 backend is implemented
- `Int4: False` until the ARM int4 backend is implemented
- no x86 variant such as `avx2` on ARM

### Unit and smoke tests

Existing tests in this checkout:

```bash
KT_ARM_ROOT=/workspace/AsymGEMM-SFT/third_party/ktransformers-arm
KT_ROOT=$KT_ARM_ROOT/kt-kernel
"$KT_ARM_ROOT/.venv/bin/pytest" "$KT_ROOT/test/per_commit/test_basic_cpu.py"
"$KT_ARM_ROOT/.venv/bin/python" "$KT_ROOT/examples/test_moe_kernel.py"
```

`test/per_commit/test_basic_cpu.py`, `examples/test_moe_kernel.py`, and
`bench/bench_moe_kernel.py` exist. The ARM-specific MoE kernel pytest does not
exist yet. Add a new focused pytest, for example
`test/per_commit/test_moe_kernel_arm_int8.py`, because
`examples/test_moe_kernel.py` uses large DeepSeek-like dimensions:

- experts: 4
- hidden size: 128 or 256
- intermediate size: 64 or 128
- top-k: 2
- qlen cases: 1, 3, 17, 64
- `m_block` tail cases where the number of tokens routed to an expert is not a
  multiple of `M_BLOCK`
- supported int8 `n`/`k` cases for the current KT blocked layout:
  `k % 32 == 0`, `intermediate_size % N_BLOCK_UP_GATE == 0`, and
  `hidden_size % N_BLOCK_DOWN == 0`
- supported int4 cases only after the int4 milestone, including `k % 64 == 0`
  and the current `GemmKernelInt4` `n` constraints:
  `intermediate_size % N_BLOCK_UP_GATE == 0`,
  `intermediate_size % N_BLOCK_UP_GATE_PREFI == 0` for prefill,
  `hidden_size % N_BLOCK_DOWN == 0`, and
  `hidden_size % N_BLOCK_DOWN_PREFI == 0` for prefill; alternatively, set and
  log smaller int4 tiling values through the existing tiling setter before the
  small-shape test; keep `N_BLOCK_*_PREFI == N_BLOCK_*` unless
  `forward_unified()` has been made fully mode-specific and tested
- negative tests for non-divisible `n`/`k` unless the implementation deliberately
  adds padding or a real `PACKED=false` layout
- a direct layout unit test for `load_int8_blocked_b`: build a synthetic
  `[n_block, k]` matrix, pack it with the `BufferB::from_mat` index formula, and
  prove the scalar GEMM matches `A @ B.T` while plain row-major access does not
- BF16 torch reference matching the logic in `examples/test_moe_kernel.py`
- online quantization from BF16 weights
- save/load through the `.kt` path after fixing the `PLAIN=true` save pointer
  selection in `MOE_KERNEL_TP::load_weights()`

The layout-only probe
`AGENT_PROBE_REF_ROOT/moe_int_layout_probe.py` is a small dependency-free
reference for the `load_int8_blocked_b` and int4 nibble indexing/scaling tests.
Copy it under `$KT_ARM_ROOT/agent/kt_adaptation/probes` before running or
editing it, and keep all probe outputs or derived validation artifacts under
`$KT_ARM_ROOT`. The canonical design docs live under `AGENT_DOC_ROOT`; the probe
is just a workspace utility until its assertions are ported into the real pytest
or kept as a pre-implementation smoke.

For every ARM-specific optimized B layout added later, extend the same test
style with sentinel values:

- generate logical `B[n, k]` where each element uniquely identifies `(n, k)`;
- call that layout's `reorder_B_gemm`;
- assert the optimized layout accessor reconstructs every logical element;
- run a small `A @ B.T` comparison through the forced backend override;
- verify a mismatched layout id fails loudly instead of producing numeric output.
- after loading or saving an optimized packed layout, force
  `KT_MOE_ARM_BACKEND=ref` and prove the scalar reference either reads the
  header/accessor correctly and matches the same output, or fails at load time
  with the documented unsupported-layout error.

Accuracy acceptance:

- int8: same metric as `examples/test_moe_kernel.py`, mean relative/error ratio
  less than `0.05`; also require no systematic NaN/Inf and max absolute error
  recorded in logs.
- int4: do not enable until it passes the existing rough threshold
  `0.35`; prefer tightening to `0.25` on small deterministic tests if practical.
- `LLAMAFILE`: existing `examples/test_moe.py` threshold remains separate and
  should not be used to certify `MOE_INT8`.

Fallback tests:

- `KT_KERNEL_CPU_VARIANT=arm_baseline` imports the extension.
- `KT_MOE_ARM_BACKEND=ref` forces scalar path.
- Invalid `KT_MOE_ARM_BACKEND` value raises with valid choices.
- On hardware without i8mm/SVE, forcing those backends must fail before any
  optimized instruction executes.
- Requesting `AMXINT8` or `AMXINT4` on ARM raises an unsupported-backend error.

### Performance smoke

Add or adapt a benchmark from `bench/bench_moe_kernel.py` that reports:

- CPU model and `/proc/cpuinfo` ARM feature flags.
- selected backend: `ref`, `neon_dotprod`, `i8mm`, or `sve_i8mm`.
- thread count, NUMA policy if applicable, qlen, top-k, expert count,
  hidden size, and intermediate size.
- latency per forward and tokens/s.

`bench/bench_moe_kernel.py` currently creates several random tensors on CUDA
before copying them to CPU. The ARM benchmark must be a CPU-only variant or mode
for this host, with random tensors created directly on `device="cpu"`, so CPU
MoE validation does not require CUDA.

Acceptance:

- Scalar baseline: correctness, no illegal instructions, no crashes, and stable
  memory usage. It has no hard throughput gate.
- NEON dot-product: at least 1.5x scalar throughput on representative int8
  prefill or decode shapes, or disabled by default with an open performance
  issue.
- i8mm/SVE: at least 2x scalar throughput on shapes they claim to accelerate,
  or disabled by default.
- Optimized fallback: unsupported tails or shapes must fall back to scalar and
  must not be more than 10 percent slower than scalar-only execution for those
  fallback shapes.

## Implementation order

Treat this as a gated implementation order, not just a task list. Every major
subphase below has an immediate validation checkpoint. Do not continue to the
next subphase until that checkpoint passes and its logs/artifacts are saved
under `$KT_ARM_ROOT/profiling_kt/int/` or the relevant pytest log directory under
`$KT_ARM_ROOT`.

1. Build hygiene
   - `operators/moe_kernel/api/common.h`: remove ARM-implies-KML behavior.
   - `CMakeLists.txt`: make KML source inclusion explicit and
     source-presence guarded.
   - `CMakeLists.txt`: remove unsafe global ARM `-march` flags.
   - `CMakeLists.txt`: add `KTRANSFORMERS_CPU_MOE_ARM`,
     `KTRANSFORMERS_CPU_MOE_ARM_OPT`, and
     `KTRANSFORMERS_CPU_MOE_ARM_OPT_LEVEL`.
   - `CMakeLists.txt`: add `KTRANSFORMERS_CPU_MOE_ARM_INT4`, default `OFF`,
     and require `KTRANSFORMERS_CPU_MOE_ARM=ON` before it can be enabled.
   - `setup.py`: forward `CPUINFER_ENABLE_ARM_MOE` and
     `CPUINFER_ARM_OPT_LEVEL`.
   - `setup.py`: forward `CPUINFER_ENABLE_ARM_MOE_INT4`, default `OFF`, and
     fail if it is enabled without `CPUINFER_ENABLE_ARM_MOE=ON`.
   - `operators/common.hpp`: audit the current ARM/KML-gated FP16 conversion;
     first ARM `MOE_INT8` native entry supports BF16 only, so convert F32/FP16
     to BF16 before `MOE_KERNEL_TP` or reject them clearly before native entry.

   Validation immediately after this subphase:
   - The forced-KML CMake command above fails with the intended
     `KTRANSFORMERS_CPU_USE_KML=ON but KML sources are not present` error before
     any missing include or `prefillgemm`/`prefillgemm_int4`
     `add_subdirectory` failure.
   - Invalid `KTRANSFORMERS_CPU_MOE_ARM_OPT_LEVEL` fails in CMake with the full
     valid-value list.
   - Invalid `CPUINFER_ARM_OPT_LEVEL` fails in setup.py before CMake with the
     same valid-value list.
   - A non-`baseline` `KTRANSFORMERS_CPU_MOE_ARM_OPT_LEVEL` with
     `KTRANSFORMERS_CPU_MOE_ARM_OPT=OFF` fails in CMake and names both values.
   - `KTRANSFORMERS_CPU_MOE_ARM_INT4=ON` without
     `KTRANSFORMERS_CPU_MOE_ARM=ON` fails in CMake.
   - `CPUINFER_ENABLE_ARM_MOE_INT4=ON` without `CPUINFER_ENABLE_ARM_MOE=ON`
     fails in setup.py before CMake.
   - Configure and build on ARM with `CPUINFER_USE_CUDA=0`,
     `CPUINFER_CPU_INSTRUCT=ARM_BASELINE`, `CPUINFER_ENABLE_ARM_MOE=ON`, and
     `CPUINFER_ARM_OPT_LEVEL=baseline`.
   - Confirm build logs do not include `operators/kml`,
     `kml_kernel`, AMX/AVX sources, or unsafe global ARM `-march` flags.
   - Confirm build logs include only the baseline ARM source files
     `arm_kernel/kernel.cpp` and `arm_kernel/reference.cpp`, not
     `neon_dotprod.cpp`, `i8mm.cpp`, or `sve_i8mm.cpp`.
   - Import `kt_kernel`, `kt_kernel_ext`, and `kt_kernel_ext.moe.MOE`.
   - Run `pytest test/per_commit/test_basic_cpu.py` and save the import/build
     log before changing runtime dispatch.

2. Import/runtime detection
   - Add ARM variants to `python/_cpu_detect.py`.
   - Ensure ARM never falls back to `avx2`.
   - Keep single-extension import working on ARM.

   Validation immediately after this subphase:
   - `KT_KERNEL_CPU_VARIANT=arm_baseline python -c "import kt_kernel"` succeeds.
   - Detection on the real ARM host reports an ARM/generic variant, never
     `avx2`.
   - Invalid `KT_KERNEL_CPU_VARIANT` values raise with valid choices.
   - Requests for AMX methods on ARM raise unsupported-backend errors, while
     `LLAMAFILE` still constructs through `kt_kernel_ext.moe.MOE`.

3. Scalar int8 backend
   - Add `operators/moe_kernel/mat_kernel/arm_kernel/kernel.cpp`.
   - Add `operators/moe_kernel/mat_kernel/arm_kernel/reference.cpp` and
     `reference.hpp`.
   - Implement all six `batch_gemm_api.hpp` symbols. For
     `reorder_B_gemm` / `get_reorder_B_size`, fail clearly if called until a
     real packed ARM layout is tested; do not use an untested byte-copy layout.
   - In `reference.cpp`, implement and unit-test
     `load_int8_blocked_b(...)` for the current `PLAIN=true, PACKED=true`
     `BufferB` layout before wiring it into GEMM.
   - Make the int4 GEMM symbols unsupported stubs during the int8-only
     milestone; do not bind `Int4_KERNEL_MOE` yet.
   - Review `operators/moe_kernel/la/mat_kernel.cpp` only if ARM needs a
     different selector, `divide_elements_size`, or debug reporting; otherwise
     keep the existing generic selector.
   - Touch `ext_bindings.cpp` only to decouple `_is_plain_` from KML and to
     keep the existing `Int8_KERNEL_MOE` binding safe under `USE_MOE_KERNEL`.
   - Add new `test/per_commit/test_moe_kernel_arm_int8.py`.
   - Run `test/per_commit/test_basic_cpu.py` and `examples/test_moe_kernel.py`.

   Validation immediately after this subphase:
   - `hasattr(kt_kernel_ext.moe, "Int8_KERNEL_MOE")` is true on the ARM build;
     `Int4_KERNEL_MOE` remains absent until the ARM INT4 backend is actually
     implemented and tested. Only internal C ABI stubs may raise a clear
     not-implemented error.
   - `test_moe_kernel_arm_int8.py` proves the blocked `BufferB` layout with
     sentinel values before any GEMM comparison is trusted.
   - Scalar int8 GEMM matches `A @ B.T` for decode and prefill shapes, including
     routed-token tail cases and supported divisibility constraints.
   - INT8 prefill and decode tiling keep up/gate and down prefill block sizes
     equal to decode block sizes, or the updated `forward_unified()` proves
     selected prefill block sizes are used consistently for B offsets, GEMM `n`,
     C offsets, output offsets, and scale blocks.
   - Online BF16-to-int8 quantization and `.kt` save/load both match the BF16
     reference within the documented int8 tolerance.
   - `GemmKernelInt8::BufferB` initializes the correct block size for
     `PLAIN=true, if_pack=true` online quantization; gate/up/down online
     quantization sentinel tests cover every expected `n` row exactly once.
   - F32 and FP16 tensors follow the chosen first-milestone policy: either they
     are converted to BF16 before `MOE_KERNEL_TP` and match the
     BF16-converted reference, or they fail before kernel launch with the
     documented unsupported-dtype error.
   - `examples/test_moe_kernel.py` passes or has an explicit shape/size skip
     justified by the new focused pytest.

4. Fallback and diagnostics
   - Add `KT_MOE_ARM_BACKEND` override in the new
     `operators/moe_kernel/mat_kernel/arm_kernel/dispatch.hpp` /
     `kernel.cpp` path.
   - Improve errors in `python/utils/moe_kernel.py` and wrapper paths.
   - Verify `LLAMAFILE`, `MOE_INT8`, and AMX names have distinct behavior.

   Validation immediately after this subphase:
   - `KT_MOE_ARM_BACKEND=ref` forces the scalar path and logs `ref`.
   - Invalid `KT_MOE_ARM_BACKEND` values fail before kernel launch and list the
     valid choices.
   - Forcing unavailable optimized backends fails before executing optimized
     instructions.
   - `MOE_INT8` never silently falls back to `LLAMAFILE`, and AMX method names
     never alias to ARM INT names.

5. Optimized int8 kernels
   - Add `operators/moe_kernel/mat_kernel/arm_kernel/neon_dotprod.cpp` first.
   - Add `operators/moe_kernel/mat_kernel/arm_kernel/layout.hpp` before any
     packed optimized layout, with an explicit `ArmMoeBLayout` id and accessors
     for each physical B layout.
   - Implement layout-specific `get_reorder_B_size` / `reorder_B_gemm` paths
     before switching any optimized backend to `PLAIN=false`.
   - Add `operators/moe_kernel/mat_kernel/arm_kernel/i8mm.cpp` where hardware
     and compiler support are available.
   - Add `operators/moe_kernel/mat_kernel/arm_kernel/sve_i8mm.cpp` only with
     vector-length-agnostic tests.
   - Add sentinel-value layout tests for each optimized B layout before using it
     in `examples/test_moe_kernel.py` or benchmarks.
   - Extend `bench/bench_moe_kernel.py` with a CPU-only mode or add an
     ARM-specific benchmark that logs backend selection, CPU flags, thread
     count, and NUMA placement.

   Validation immediately after this subphase:
   - Each optimized layout id round-trips logical `B[n, k]` sentinel values and
     rejects mismatched layout ids loudly.
   - Runtime feature checks gate NEON dotprod, i8mm, and SVE/i8mm separately;
     `auto` falls back to `ref` on unsupported hardware, while forced optimized
     overrides fail before executing unsupported instructions.
   - Optimized kernels compare against scalar int8 for decode, prefill, top-k
     routing, and routed-token tails before they are enabled by default.
   - After saving/loading optimized packed weights, `KT_MOE_ARM_BACKEND=ref`
     validates the same layout through `ArmPackedBHeader` and `layout.hpp`
     accessors, or load fails with the documented unsupported-layout error.
   - The CPU-only benchmark records backend, CPU flags, thread count, shape,
     latency, and tokens/s; claimed optimized paths must meet the performance
     gates from `Performance smoke` or remain opt-in/disabled.

6. Int4 backend
   - Implement and unit-test `load_int4_blocked_b(...)` using the documented
     high-nibble / low-nibble layout from `GemmKernelInt4::BufferB::from_mat`.
   - Implement scalar int4 in `operators/moe_kernel/mat_kernel/arm_kernel`.
   - Touch `ext_bindings.cpp` for `Int4_KERNEL_MOE` only after the ARM int4 C
     ABI is implemented and tested.
   - Add optimized int4 only if it has a measurable benefit.

   Validation immediately after this subphase:
   - Do not start int4 until scalar int8 and selected optimized int8 paths have
     passing correctness and benchmark artifacts.
   - `load_int4_blocked_b(...)` has sentinel layout tests for nibble order,
     scaling, and tails before `Int4_KERNEL_MOE` is exposed.
   - `GemmKernelInt4::BufferB::from_mat(...)` uses the selected up/gate/down
     block size instead of generic `N_BLOCK`; full-row sentinel tests prove
     gate/up/down online INT4 quantization covers every row exactly once.
   - INT4 prefill tests also either keep prefill tile sizes equal to decode tile
     sizes, or prove the updated `forward_unified()` uses the selected prefill
     block size consistently for B offsets, GEMM `n`, C offsets, and scale
     blocks.
   - Scalar int4 passes the documented int4 accuracy threshold against the BF16
     reference on deterministic small shapes.
   - `Int4_KERNEL_MOE` is either fully bound and tested or remains absent with a
     clear unsupported-backend error; no throwing placeholder may be exposed as a
     working backend.

7. LlamaFactory follow-up
   - After ktransformers ARM inference is stable, add LLaMA-Factory examples or
     parser validation only for existing KT INT method names: `MOE_INT8` and,
     after completion, `MOE_INT4`.
   - Do not add ARM aliases that look like `AMX*`, and do not depend on external
     `transformers-kt` / `accelerate-kt` packages unless that path is separately
     pinned and smoke-tested.

   Validation immediately after this subphase:
   - LF examples/configs accept only `MOE_INT8` and, after completion,
     `MOE_INT4`; AMX-like ARM aliases are rejected.
   - A tiny LF inference smoke proves the selected KT method reaches
     `Int8_KERNEL_MOE` rather than `LLAMAFILE`. If this smoke uses CUDA or LF
     training/inference helpers, run with `CUDA_VISIBLE_DEVICES=3` and the
     copied Qwen/Qwen3-30B-A3B smoke data under `$LF_ROOT/data`; do not read or
     modify the reference LF data directory during the proof run.
   - Any optional external `transformers-kt` / `accelerate-kt` route is pinned
     and smoke-tested before it is mentioned as supported; otherwise the design
     remains local-direct.

## Risks

- KML references are stale in this checkout. Any accidental `CPU_USE_KML` on
  ARM can break compilation before the ARM backend is reached.
- Global ARM feature flags can create binaries that import successfully but
  crash with illegal instructions on weaker ARM hosts.
- `MOE_INT8` and `LLAMAFILE` use different weight formats. Silent fallback
  between them would hide configuration errors.
- Existing block-size assumptions in `GemmKernelInt8` / `GemmKernelInt4` may
  mask tail bugs if tests use only model-friendly dimensions.
- FP16 tensor conversion in `operators/common.hpp` is currently KML-gated on
  ARM. The first portable ARM backend should validate BF16 native tensors,
  prove any accepted F32 tensors are converted to BF16 before native entry, or
  add a non-KML FP16 conversion before accepting FP16.
- Optimized kernels need both compile-time and runtime guards. Compile-only
  guards are insufficient for redistributable wheels.
- LlamaFactory examples currently use AMX backend names. Reusing those names for
  ARM would make logs and checkpoints ambiguous.
- Accidentally editing the reference `third_party/ktransformers` or
  `third_party/LlamaFactory` checkouts would make the proof hard to reproduce
  and could affect unrelated work. Keep all ARM code and env changes under
  `third_party/ktransformers-arm` until the port is accepted.

## Convergence checklist

This design is considered implemented when:

- the implementation, wheels, local LF patches, and benchmark/profiling scripts
  live under `third_party/ktransformers-arm`, with a separate env, and the
  reference ktransformers/LF trees remain untouched by the proof phase;
- aarch64 build succeeds without KML, AMX, AVX, BLIS, or unsafe global `-march`;
- `kt_kernel.__cpu_variant__` reports an ARM variant on ARM;
- `LLAMAFILE` remains available;
- `MOE_INT8` works through `Int8_KERNEL_MOE` with scalar fallback;
- optimized int8 backends runtime-dispatch and fall back safely if they are
  implemented or enabled;
- `MOE_INT4` is either absent with a clear error or fully implemented and
  tested;
- AMX/x86 backend names still raise on ARM;
- small and representative MoE tests pass the accuracy thresholds above;
- benchmark logs identify the selected ARM backend and CPU features.
