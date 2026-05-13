# SM80 MoE GEMM — Vectorised FP32 → BF16/FP16 Conversion (`optimized_fp32bf16.md`)

## 1. Problem: Scalar FP32 → Element Conversion of the MMA Accumulator

### Where the issue is

`sm80_moe_gemm.cuh` contains **three identical scalar conversion loops** that turn the
FP32 MMA accumulator `tSrO` into the output dtype (BF16 or FP16) before staging into
shared memory:

| Site | Kernel | Context | Lines (approx.) |
|------|--------|---------|------------------|
| ① | `sm80_moe_gemm_impl` (BF16/FP16) | end of K-loop, per M-tile | 345-349 |
| ② | `sm80_moe_fp8_gemm_impl` | k=0 path, per M-tile | 633-637 |
| ③ | `sm80_moe_fp8_gemm_impl` | k>0 path, per (k, M) | 732-736 |

Each of them runs the same scalar pattern:

```cpp
// // Convert FP32 accumulator → Element  /  // FP32 → BF16 register buffer …
Tensor rO = make_tensor<Element>(shape(tSrO));
CUTE_UNROLL
for (int i = 0; i < size(tSrO); i++)
    rO(i) = Element(static_cast<float>(tSrO(i)));
```

`size(tSrO)` is the per-thread accumulator count.  For the default config
(`BLOCK_M = BLOCK_N = 128`, `NWARPS = 4`, FP8 MMA atom `16×8×32` → 4 acc/thread/atom,
1 M-atom × 8 N-atoms per warp), each thread holds **32 FP32 accumulators**.

The `static_cast<float>` is a no-op (tSrO is already FP32) and `Element(...)` performs
a per-element conversion through the CUTLASS scalar `NumericConverter<bfloat16_t, float>`
(or `<half_t, float>`), which does NOT vectorise — NVCC has to insert one conversion
sequence per element.

**Total cost per M-tile, per K-iteration:** 32 scalar conversions where 16 hardware
instructions would suffice.

---

## 2. Reference: How FA2 and `mixtureExpertKernel.cu` Convert in One Shot

### 2.1 FA2 — `convert_type` helper (`flash-attention/csrc/flash_attn/src/utils.h:228-236`)

```cpp
template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}
```

Two key ideas:

1. **Reinterpret the register tensor as a contiguous `cutlass::Array<From_type, N>`** —
   safe because CuTe register fragments are stored as plain register arrays.
2. **Dispatch through `cutlass::NumericArrayConverter<To, From, N>`** — CUTLASS picks
   the SIMD specialisation that exists for FP32→BF16 / FP32→FP16 of length 2, and the
   N>2 generic specialisation simply unrolls calls to the N=2 path.

Used everywhere in FA2 for the epilogue dtype change, e.g. `flash_fwd_kernel.h:436`:

```cpp
Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
```

### 2.2 CUTLASS provides the PTX SIMD path

`third-party/cutlass/include/cutlass/numeric_conversion.h:1076-1096`:

```cpp
template <>
struct NumericArrayConverter<cutlass::bfloat16_t, float, 2, FloatRoundStyle::round_to_nearest> {
    static result_type convert(source_type const & source) {
        unsigned d;
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n"
            : "=r"(d)
            : "f"(source[1]), "f"(source[0]));
        return reinterpret_cast<result_type const &>(d);
    }
    ...
};
```

`cvt.rn.bf16x2.f32` is a single SM80+ PTX instruction that converts 2 FP32 inputs into
2 BF16 packed in one 32-bit register.  The general-N specialisation
(`NumericArrayConverter<bfloat16_t, float, N, Round>`, line 1131) loops in pairs:

```cpp
for (int i = 0; i < N / 2; ++i) {
    result_ptr[i] = convert_vector_(source_ptr[i]);   // calls the N=2 SIMD version
}
```

So a 32-element conversion compiles to **16** `cvt.rn.bf16x2.f32` instructions.  An
identical specialisation exists for `<half_t, float, 2>` using `cvt.rn.f16x2.f32`.

### 2.3 `mixtureExpertKernel.cu` — same pattern (line 232-233)

```cpp
Tensor rO = make_tensor_like<Element>(tSrO);
flash::convert_type_out(tSrO, rO);
```

`convert_type_out` is the in-place variant of `convert_type` (writes into a pre-
allocated tensor).  Either form works; we'll use the value-returning form because it
matches the existing CuTe idiom in the file.

---

## 3. Proposed Fix

### 3.1 Add a local `moe_convert_type` helper

`sm80_moe_gemm.cuh` does not include FA2 headers.  Mirror the existing `moe_predicated_copy`
convention by adding a self-contained helper just below it (already inside
`namespace asym_gemm`):

```cpp
// ──────────────────────────────────────────────────────────────────────────────
// moe_convert_type: vectorised register-fragment conversion via cutlass NumericArrayConverter.
//
// Port of FLASH_NAMESPACE::convert_type (flash-attention/csrc/flash_attn/src/utils.h).
// For FP32 → BF16 the inner 2-elem specialisation emits one `cvt.rn.bf16x2.f32`
// per pair (SM80+); for FP32 → FP16 it emits `cvt.rn.f16x2.f32`. CUTLASS handles
// the dispatch; this wrapper just adapts CuTe register tensors to a contiguous
// `cutlass::Array<From, N>`.
//
// Requirements:
//   - tensor must be a register-residency tensor whose data() returns a contiguous
//     register array (true for tensors from partition_fragment_C / make_tensor<T>(layout)
//     and any CUTE_UNROLL-safe register fragment).
//   - tensor's element count must be known at compile time.
//
// Usage:
//   Tensor rO_bf16 = moe_convert_type<ElementOut>(tSrO);
// ──────────────────────────────────────────────────────────────────────────────
template <typename To_type, typename Engine, typename Layout>
CUTE_DEVICE auto moe_convert_type(cute::Tensor<Engine, Layout> const& tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(cute::size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    auto frag = convert_op(
        *reinterpret_cast<const cutlass::Array<From_type, numel>*>(tensor.data()));
    return cute::make_tensor(cute::make_rmem_ptr<To_type>(&frag), tensor.layout());
}
```

This is a verbatim port of FA2's `convert_type` with `cute::` namespace prefixes.

### 3.2 Replace the three scalar loops

#### Site ① — `sm80_moe_gemm_impl` (BF16/FP16, lines 345-349)

**Before:**
```cpp
// ── Convert FP32 accumulator → Element ─────────────────
Tensor rO = make_tensor<Element>(shape(tSrO));
CUTE_UNROLL
for (int i = 0; i < size(tSrO); i++)
    rO(i) = Element(static_cast<float>(tSrO(i)));
```

**After:**
```cpp
// ── Convert FP32 accumulator → Element via cvt.rn.{bf16x2,f16x2}.f32 ──
Tensor rO = moe_convert_type<Element>(tSrO);
```

#### Site ② — `sm80_moe_fp8_gemm_impl`, k=0 path (lines 633-637)

**Before:**
```cpp
// FP32 → BF16 register buffer, then write to sO → gO
Tensor rO = make_tensor<ElementOut>(shape(tSrO));
CUTE_UNROLL
for (int i = 0; i < size(tSrO); i++)
    rO(i) = ElementOut(static_cast<float>(tSrO(i)));
Tensor gO_m = gO(_, _, m);
write_output(rO, gO_m, m_actual);
```

**After:**
```cpp
// FP32 → BF16 via cvt.rn.bf16x2.f32, then write sO → gO
Tensor rO = moe_convert_type<ElementOut>(tSrO);
Tensor gO_m = gO(_, _, m);
write_output(rO, gO_m, m_actual);
```

#### Site ③ — `sm80_moe_fp8_gemm_impl`, k>0 path (lines 732-736)

Identical replacement to Site ②.

### 3.3 Verify CUTLASS dependency is already in scope

`sm80_moe_gemm.cuh` already pulls `cutlass::float_e4m3_t` etc. via `cutlass/numeric_types.h`,
which transitively includes `cutlass/array.h` and `cutlass/numeric_conversion.h`.  No new
includes needed.  If a build flag fails this assumption, add at the top of the file:

```cpp
#include <cutlass/numeric_conversion.h>
#include <cutlass/array.h>
```

---

## 4. Expected Performance Gain

Per M-tile, per K-iteration (where the conversion runs):

| | Old (scalar) | New (SIMD) |
|--|--|--|
| Per-thread instructions | `numel` (e.g., 32) `cvt.rn.bf16.f32` | `numel/2` (e.g., 16) `cvt.rn.bf16x2.f32` |
| Throughput multiplier | 1× | 2× |

Effect on overall TFLOPS depends on how often the conversion is invoked relative to
MMA work:

- **k_max = 1** (small K): conversion runs once per M-tile.  Modest impact (~1-2%).
- **k_max ≫ 1** (large K, e.g., FFN with K=4096+ at BLOCK_K=128 → k_max=32): site ③
  runs once per (k, m).  More noticeable (~3-5% on memory-light shapes).
- **Decode phase** (small total_tokens): conversion is on the critical path for every
  partial M-tile — combined with the data_access.md vectorisation, this finishes the
  epilogue savings story.

The improvement is largest when combined with `moe_predicated_copy`: after the store
path is vectorised (bytes/instruction up 8×), the conversion is what's left to eat
into the epilogue tail.

---

## 5. Out-of-Scope (Follow-up)

The FP8 kernel's **k>0 partial-tile seed path** also performs scalar BF16 → FP32
conversion when reading `gO` back as the partial sum:

```cpp
// Lines ~609-612 (full tile) and ~644-649 (partial tile)
for (int i = 0; i < size(tSrO); i++)
    tSrO(i) = static_cast<float>(tSgO(i));      // or rO_seed(i)
```

These are the inverse direction (BF16 → FP32) and benefit from the **same helper**
called as `moe_convert_type<float>(...)`.  CUTLASS provides
`NumericArrayConverter<float, bfloat16_t, N>` so the same SIMD dispatch happens
(BF16 → FP32 is essentially a left-shift; CUTLASS still picks the array path which
NVCC vectorises predictably).

This optimisation is worth doing but is **scoped out of this plan** to keep the change
minimal and the correctness review focused on a single direction.  Propose handling it
in a follow-up plan once the FP32 → BF16 change is verified.

---

## 6. Files to Change

| File | Change |
|------|--------|
| `asym_gemm/include/asym_gemm/impls/sm80_moe_gemm.cuh` | Add `moe_convert_type` helper; replace 3 scalar conversion loops |

Single file, header-only — JIT-compiled per kernel launch, so no `_C.so` rebuild required.

---

## 7. Correctness Verification

### 7.1 Why correctness is preserved

1. **Same rounding mode.**  Default `Round = FloatRoundStyle::round_to_nearest` matches
   the implicit rounding used by `Element(static_cast<float>(x))` (round-to-nearest-even,
   which is what BF16/FP16 conversion uses by default in CUTLASS).
2. **Same domain.**  `cvt.rn.bf16x2.f32` produces IEEE bit-equivalent BF16 values for
   normal/subnormal/inf/NaN inputs.  No saturation difference vs. the scalar path.
3. **Same data layout.**  `make_tensor(make_rmem_ptr<To>(&frag), tensor.layout())`
   preserves the register-residency tensor's logical layout — downstream consumers
   (`smem_thr_copy_O.retile_S(rO)`) see the same shape.
4. **`numel` is compile-time.**  `decltype(size(tensor))::value` resolves at compile
   time for a register fragment whose layout is a tuple of `Int<>` constants — true
   for `tSrO` from `partition_fragment_C(sO)`.

### 7.2 Test plan

Run the existing suite with **no tolerance changes**:

```bash
python tests/test_sm80_moe.py
```

All 8 `TEST_CASES` must pass with `diff < 0.01` (current diffs after data_access.md
were 0.0003-0.0009, all noise).  Numeric output must be **bit-identical** to the
scalar path because both invoke the same underlying CUTLASS `NumericConverter` with
the same rounding mode — the only difference is whether the conversion is vectorised
across pairs.

Edge cases that already exist in `TEST_CASES` and stay relevant:
- Single-expert / multi-expert (covers all parallelism paths)
- Partial M-tiles `[7, 13, 31, 127]` (covers the predicated copy path that consumes `rO`)
- Single BLOCK_K tile and many BLOCK_K tiles (covers k_max==1 vs k_max>1 conversion sites)

If any test diff increases vs. the pre-change baseline by more than 1e-6, that
indicates a rounding-mode mismatch and the helper should be invoked with explicit
`FloatRoundStyle::round_to_nearest` template parameter.
