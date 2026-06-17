# Stride-aware AMX INT8 Kernel — Implementation Plan

**Status:** design proposal (pre-implementation).
**Targets:** `third-party/cpu_gemm/` (new INT8 kernel + dispatcher hook) and
`asym_gemm/unified_moe/runtime.py` (drop the VNNI side buffer).
**Companion docs:** `layout.md` §5.3 (Phase 2), `int8.md`,
`unified_kernel.md`, `third-party/cpu_gemm/analysis.md` §4 item 3 + §5.3.

Goal: let the CPU AMX INT8 kernel consume **row-major `[N, K]` int8
weights** directly — the same byte layout the SM90 INT8 GPU kernel reads.
Once landed, the unified-MoE runtime keeps **one** pinned weight copy per
expert, with zero per-call B-pack overhead.

---

## 1. Why the obvious approach doesn't work

The reflex idea — "the AMX `_tile_loadd` instruction takes a stride, so
just stride-load B from row-major `[N, K]`" — is wrong. Walking through
the Intel SDM pseudo-code for `TDPBSSD(dst, src1, src2)`:

```
for m in [0, dst.rows):
  for k_quad in [0, src1.colsb / 4):
    for n_lane in [0, src2.colsb / 4):
      dst.row[m].dword[n_lane] += DPBD(src1.row[m].dword[k_quad],
                                       src2.row[k_quad].dword[n_lane])
```

The two operands have **different** required layouts:

| Operand | Logical shape | Byte at tile position…                                       |
|---------|---------------|--------------------------------------------------------------|
| src1    | `[M, K]`      | `src1.row[m].byte[k]      = A[m, k]`                         |
| src2    | `[K/4, N·4]`  | `src2.row[k/4].byte[n·4 + (k%4)] = B[n, k]` (VNNI-4 packing) |

src1 is row-major. **src2 is a `K`-row-major / `N`-int32-lane / 4-K-bytes
packing of B^T**. Strided `_tile_loadd` of row-major `B[N, K]` would
deliver `[N=16 rows, K=64 cols]` into the tile — that is the **src1**
layout, not the src2 layout. The byte at tile row `r`, column `c` would
be `B[n_begin+r, k_begin+c]`, but src2 needs the byte at row `r`, column
`c` to be `B[(c/4) + n_begin, k_begin + 4r + (c%4)]`. Different layouts,
not interchangeable.

The `transpose_16x16_32bit` step in `BufferBInt8::pack_tiles`
(`int8_buffers.h:304`) exists precisely to bridge this gap — to rotate a
naturally-loaded `[N, K]` slab into the `[K/4, N·4]` VNNI form `_tile_dpbssd`
demands. Removing it without a structural change to the kernel would
miscompute B.

---

## 2. The structural fix — swap operand roles

`_tile_dpbssd` doesn't care which operand we *call* "A" and which "B".
Mathematically:

```
C  = A · B^T          ⟺          C^T = B · A^T
```

Feeding **B into src1** (the row-major slot) and a VNNI-packed view of
**A into src2** (the K/4-row slot) computes `C^T[n, m]` instead of
`C[m, n]`. That's exactly what we want for unifying the weight layout —
because the large, fixed, expensive-to-repack tensor is B, and we want
B to come in raw row-major straight from pinned memory.

| Operand    | Was (current packed kernel) | After swap                         |
|------------|------------------------------|-------------------------------------|
| AMX src1   | A row-major `[M, K]`         | **B row-major `[N, K]`** (strided!) |
| AMX src2   | B VNNI-packed `[K/4, N·4]`   | A VNNI-packed `[K/4, M·4]`          |
| AMX dst    | C `[M, N]` int32             | C^T `[N, M]` int32                  |
| Per-call pack | A (cheap) + B (heavy)     | A (cheap); B reads pinned bytes raw  |

The strided B load is now **legal and natural**: a single
`_tile_loadd(tile, B + n_begin*K + k_begin, K)` fetches a 16×64-byte
slab `B[n_begin..n_begin+15, k_begin..k_begin+63]` directly into the
src1-shaped tile.

The A-side VNNI pack stays — but A is small. Per-call A-pack already
exists in the current code (`int8_pack_a_bf16`); only its **target
layout** changes from `[M_pad, K]` int8 row-major (current
`BufferAInt8`) to the same `[K/4, M·4]` VNNI form the current
`BufferBInt8` produces for B. Implementation reuses
`BufferBInt8::from_bf16` / `from_int8` byte-for-byte; only the per-row
quant input is A's BF16 instead of B's BF16.

The output tile lands as `C^T`, so the unpack pass must transpose
32×32 int32 blocks while it scales by `(sA × sB)`. This is cheap (§5).

---

## 3. Kernel design

### 3.1 Tile assignment

Same `_tile_dpbssd` arithmetic; only the labels swap.

| Tile | Was                       | After swap (this kernel)            |
|------|---------------------------|--------------------------------------|
| 0, 1 | A row-major (M-rows, K)   | **B row-major (N-rows, K), strided** |
| 2, 3 | B VNNI (K/4, N·4)         | **A VNNI (K/4, M·4)**                |
| 4–7  | C int32 (M·N)             | **C^T int32 (N·M)**                  |

Each `_tile_dpbssd` thus accumulates `dst[n, m] += B[n, k] · A[m, k]`.

`TileConfig` (`int8_gemm.cpp:55-61`) is **bit-identical**: rows=16,
colsb=64 for tiles 0–3; rows=16, colsb=64 for tiles 4–7. Only the
semantic role per tile differs.

### 3.2 Inner kernel — `amx_kernel_rm`

```cpp
// Compute one (n_begin, m_begin) tile group across the full K-block.
// Output goes to C^T scratch (laid out [N_pad, M_pad] int32 in blocked form).
static void amx_kernel_rm(int /*m*/, int n, int k,
                          int n_begin, int m_begin, int k_block_begin,
                          int32_t* ct_sub,                  // 32x32 int32 C^T tile
                          const int8_t* b_rm, std::size_t ldb_bytes,   // row-major B
                          BufferA_VNNI& aa) {                          // VNNI-packed A
  if (k_block_begin == 0) {
    clean_c();
  } else {
    load_c(ct_sub, N_STEP * sizeof(int32_t));
  }
  for (int k_begin = 0; k_begin < K_BLOCK && k_block_begin + k_begin < k;
       k_begin += K_STEP) {
    // src1: 2 tiles of row-major B, 16 N-rows each, strided load (stride = K bytes).
    const int8_t* b_ptr = b_rm + (size_t)n_begin * ldb_bytes + (k_block_begin + k_begin);
    _tile_loadd(0, b_ptr,                                       ldb_bytes);
    _tile_loadd(1, offset_pointer(b_ptr, ldb_bytes * TILE_N),   ldb_bytes);

    // src2: 2 tiles of VNNI-packed A (cached, contiguous; ldb still 64 bytes).
    const int8_t* a_ptr = aa.get_submat(m_begin, k_block_begin + k_begin);
    _tile_loadd(2, a_ptr,                                K_STEP * sizeof(int8_t));
    _tile_loadd(3, offset_pointer(a_ptr, K_STEP * TILE_N), K_STEP * sizeof(int8_t));

    _tile_dpbssd(4, 0, 2);   // C^T[N0..15, M0..15]  +=  B[N0..15] · A[M0..15]
    _tile_dpbssd(5, 0, 3);   // C^T[N0..15, M16..31] +=  B[N0..15] · A[M16..31]
    _tile_dpbssd(6, 1, 2);   // C^T[N16..31, M0..15]
    _tile_dpbssd(7, 1, 3);   // C^T[N16..31, M16..31]
  }
  store_c(ct_sub, N_STEP * sizeof(int32_t));    // emits C^T sub-tile to scratch
}
```

The instruction sequence and tile reuse pattern is **identical** to the
current packed kernel. The only changes are:

1. The src1 base pointer is computed from the row-major B tensor with
   `ldb_bytes = K`, not from a pre-packed slab.
2. The outer loop variable iteration order in `integer_mat_mul_rm` is
   `(n_begin outer, m_begin inner)` — so for each n-stripe we sweep all
   M, keeping B's k-block hot in L2.

### 3.3 Outer driver — `integer_mat_mul_rm`

```cpp
// Drops the n_blocks split (still partitions across threads by N_BLOCK).
for (int blk = 0; blk < n_blocks; ++blk) {
  if (blk % nth != ith) continue;
  int n_start = blk * K::N_BLOCK;
  int n_end   = std::min(n_pad, n_start + K::N_BLOCK);
  for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K::K_BLOCK) {
    for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {     // outer N
      for (int m_begin = 0; m_begin < m;          m_begin += K::M_STEP) {    // inner M
        int32_t* ct_sub = bc_t.get_submat(n_begin, m_begin);    // C^T scratch
        K::amx_kernel_rm(m, n_pad, k, n_begin, m_begin, k_block_begin,
                         ct_sub, b_rm, k, aa);
      }
    }
  }
}
```

Threading boundary unchanged: each thread owns whole `N_BLOCK` stripes;
no inter-thread aliasing on `bc_t`.

### 3.4 Reused buffer types

* `BufferA_VNNI` is the existing `BufferBInt8` rewired for A. The byte
  layout it produces — `[K_BLOCK chunks][K_STEP groups-of-4][lanes × 4]`
  with the in-place `transpose_16x16_32bit` — is exactly what AMX src2
  expects. The current A buffer (`BufferAInt8`, row-major `[M, K]` int8)
  was sized for the *old* src1 role; in the swapped kernel it would still
  feed the *new* src1, which now holds B. So the rename + role-swap is:

  | Buffer name in new file       | Role in `amx_kernel_rm` | Bytes hold        |
  |-------------------------------|--------------------------|-------------------|
  | `BufferA_VNNI` (≡ old BufferB) | AMX src2 (was B)         | A in VNNI form    |
  | (no B buffer; raw row-major)  | AMX src1 (was A)         | B in pinned host  |
  | `BufferCT_int32`              | AMX dst                  | C^T blocked       |

* The A pack scales **per row of A**, i.e. per token. Identical to the
  current `BufferAInt8::from_bf16` per-row quant. Just emit into the VNNI
  layout instead of the row-major one.

* The B side has no scratch on the GEMM path. The per-channel scales
  `[N]` for B come in as a plain `const float*` parameter, like SFB on
  the GPU side.

### 3.5 Per-token / per-channel quant

Quantization values are byte-identical to the current kernel — symmetric
`amax/127`. Storage moves:

| Scale | Current kernel              | Stride-aware kernel               |
|-------|-----------------------------|------------------------------------|
| SFA (per-row A)  | `BufferAInt8::d[max_m]` (trailing) | `BufferA_VNNI::d[max_m]` (trailing — same offset math) |
| SFB (per-channel B) | `BufferBInt8::d[n_pad]` (trailing) | plain `const float*` argument; no internal buffer |

This matches the GPU side's per-channel `[N]` FP32 scales held as a flat
pinned tensor.

### 3.6 Output unpack with implicit transpose

The scratch `bc_t` is in C^T layout: `[n_pad, m_pad]` int32 in blocked
form `[n_blocks][m_block][m_step][n_step]`. The row-major output
`C[m, n]` needs:

```
C[m, n] = α · sA[m] · sB[n] · int32(C^T[n, m])  +  β · C_prev[m, n]
```

Per 32×32 tile group, this is a 32×32 int32 transpose with per-element
scale and accumulate. Implementation:

* Load 32 rows of C^T from scratch into 32× `__m512i` (1 KiB working set).
* Transpose using a 32×32 int32 in-place permutation built from two
  16×16 32-bit transposes (the existing `transpose_16x16_32bit` from
  `amx_utils.h:57-126`) plus a 16-row swap.
* Convert int32 → FP32, multiply by `sA[m_row] · sB[n_col]`, FMA against
  `β · C_prev`, store to `C[m, n]` row-major.

The transpose adds one pass over the scratch (32×32 int32 = 4 KiB per
tile) before the existing scale-apply work — every byte of scratch is
touched exactly once during unpack, same as today. Expected unpack-time
overhead: < 5% of total kernel time at AMX-dominated shapes.

A simpler first implementation skips the SIMD 32×32 transpose and emits
one element per inner iteration (scalar loop), prioritising correctness
over speed — the unpack is already not on the critical path.

---

## 4. Public ABI

No change to `cg_gemm_desc_t`. The stride-aware path is selected by
dispatcher heuristics, not by a new dtype. Specifically:

* `dtype_b == CG_INT8`, `b_scales != nullptr`, `trans_b == CG_TRANS`:
  this descriptor is the same as today's "B is int8 + per-channel scales,
  pre-pack on first touch" path. After the new kernel lands, the
  dispatcher routes it through `run_amx_int8_rm` instead of
  `run_amx_int8`.
* `CG_INT8_PACKED_AMX` remains supported for callers that pre-packed B
  offline; routed through `run_amx_int8_prepacked` unchanged.

The `ldb == k` constraint in `amx_int8_eligible` (`gemm.cpp:100`) is
relaxed to `ldb >= k`. The dispatcher's existing `is_bf16_int8_f32_nt`
predicate is unchanged.

### 4.1 New header

`src/kernels/amx/int8_gemm_rm.h`

```cpp
namespace cpu_gemm::kernels::amx {

// Scratch sizes for the stride-aware INT8 path. Differences vs the packed path:
//   - No bytes_b: B is read straight from caller memory.
//   - bytes_a now matches the packed BufferB shape (VNNI), not the row-major
//     BufferA — A's pack target moved to AMX src2.
//   - bytes_c is C^T scratch, sized [n_pad, max_m_pad] int32.
struct Int8RmScratch {
  std::size_t bytes_a;     // VNNI-packed A + per-row scales
  std::size_t bytes_c;     // C^T int32 blocked
  std::size_t total() const { return bytes_a + bytes_c; }
};

Int8RmScratch int8_rm_scratch(int m, int n, int k);

void int8_rm_pack_a_bf16(int m, int k, const cg_bf16_t* a_rm, void* scratch_a);

void int8_rm_run(int m, int n, int k,
                 const int8_t* b_rm, std::size_t ldb,
                 void* scratch_a, void* scratch_c,
                 int ith, int nth);

void int8_rm_unpack_transposed(int m, int n,
                               const float* a_scales,    // per-row A
                               const float* b_scales,    // per-channel B  (from caller)
                               const void*  scratch_ct,
                               float alpha, float beta,
                               float* c_rm, std::size_t ldc,
                               int ith, int nth);

}  // namespace
```

### 4.2 Dispatcher hook

`src/dispatch/gemm.cpp` adds, alongside the existing `run_amx_int8`:

```cpp
bool amx_int8_rm_eligible(const cg_gemm_desc_t* d) {
  using T = cpu_gemm::kernels::amx::Int8KernelTraits;
  if (!cpu_gemm::kernels::amx::amx_available()) return false;
  if (!is_bf16_int8_f32_nt(d)) return false;
  if (d->k % T::K_STEP != 0) return false;
  if (d->lda != d->k)          return false;        // A still row-major BF16
  if (d->ldb <  d->k)          return false;        // B row-major, stride >= k
  return true;
}

cg_status_t run_amx_int8_rm(cg_runtime_t* rt, const cg_gemm_desc_t* d) { /* mirrors run_amx_int8 */ }
```

Priority order in `run_cg_gemm`:

```
AMX-INT8-prepacked     (CG_INT8_PACKED_AMX; unchanged)
AMX-INT8-rm            (CG_INT8 row-major + scales; NEW, default for raw B)
AMX-BF16               (unchanged)
AVX2-BF16              (unchanged)
```

The `run_amx_int8` (packed-via-runtime-scratch) wrapper can be retired
once the rm path is parity-clean — it becomes dead code because anything
that reached it (caller passes raw B + scales) is now eligible for the
rm path with strictly less scratch.

### 4.3 Python and unified-runtime wiring

* `csrc_cpu/cpu_module.cpp`: drop `pack_b_int8_amx_size`,
  `pack_b_int8_amx`, and `gemm_bf16_int8_packed`. Keep
  `gemm_bf16_int8` — its signature already matches the rm path's
  expected inputs (BF16 A row-major, int8 B row-major + scales,
  fp32 C row-major).
* `asym_gemm/unified_moe/runtime.py`:
  * `ExpertSlab` loses `gate_packed`, `up_packed`, `down_packed`,
    `packed_array(...)`, and the `_PinnedAmxBuffer` wrapper.
  * `from_bf16(...)` no longer calls `_C.pack_b_int8_amx`.
  * `_cpu_expert_forward` calls `_C.gemm_bf16_int8(self.rt,
    x_bf16_bits, slab.gate_int8.numpy(), slab.gate_scales.numpy(),
    c_gate, 1.0, 0.0)` instead of the packed variant.
* `asym_gemm/unified_moe/__init__.py` docstring updated.

---

## 5. Verification

Mirror the §7 ladder from `int8.md` (GPU INT8) so each stage isolates
one source of risk.

### Stage A — compile-only
* `cpu_gemm/tests/CMakeLists.txt`: add `test_amx_int8_rm`.
* `nvcc`/`g++ -c` builds the new TU; no link.
* Pass: no warnings under `-Wall -Wextra -Wpedantic`.

### Stage B — single-tile numeric (no transpose surprises)
* Pick `M=32, N=32, K=64` (one tile group, one K_STEP).
* B row-major `[N, K]` int8 with known small ints, per-channel scales
  all 1.0; A BF16 with values such that the quantized A is integer.
* Reference: scalar int32 dot, then dequant.
* Pass: bit-identical to ref (no rounding involved at this size).
* Failure modes localised here: wrong stride argument to `_tile_loadd`,
  wrong dst-tile interpretation as C^T, wrong A-VNNI pack target.

### Stage C — multi-K-block (exercises K accumulation across blocks)
* Same as B but `K = 256` so `K/K_STEP = 4` inner k iterations occur.
* Pass: bit-identical to ref. Confirms accumulator carry across iterations.

### Stage D — multi-tile / multi-N_BLOCK with transpose unpack
* `M=64, N=128, K=512`. Two 32×32 tile groups in N, two in M.
* Compare to dense FP32 reference within `1 ULP` (only the FP32 scale
  multiply rounds).
* Pass criterion: `max_abs / max_ref < 1e-6`.
* Failure modes localised here: row/column swap in the 32×32 transpose,
  wrong indexing into `bc_t` scratch, scale broadcast confused
  per-row/per-col after the swap.

### Stage E — stride ≠ K
* `ldb = k + 64`, real n_used < n_alloc. Confirms strided
  `_tile_loadd` reads only the in-bounds bytes per row.
* Pass: bit-identical to a contiguous-ldb run.

### Stage F — parity vs packed kernel
* Sweep `(M, N, K) ∈ {(1, 4096, 4096), (8, 4096, 4096),
  (32, 4096, 4096), (64, 14336, 4096), (256, 4096, 4096)}` (matches
  `bench_amx_vs_avx2`).
* Run both `run_amx_int8` (current packed) and `run_amx_int8_rm`
  (new) on the same `(A, B, scales)`.
* Pass: `max_abs / max_ref < 1e-6`. The two should be bit-identical
  if scale arithmetic is in the same order, modulo (`a_scale * b_scale`)
  vs `(b_scale * a_scale)` FP32 rounding.

### Stage G — multi-thread determinism
* `nth ∈ {1, 4, 16, ceil(N/N_BLOCK)}`. The partition is along N
  whole-`N_BLOCK` stripes, so no cross-thread aliasing on the C^T
  scratch. Pass: bit-identical across nth.

### Stage H — guard-page memory safety
* Replicate the `PROT_NONE`-guard harness from `analysis.md` §4.1: put
  B abutting an unmapped page, run with `n` not a multiple of
  `N_STEP`. Pass: no SIGSEGV. (The strided `_tile_loadd` reads
  `TILE_N` rows per tile of B; for the last N-block we must zero
  the padded rows of B into a stash, not load them from caller
  memory — see §6 risk 2.)

### Stage I — unified-MoE end-to-end
* `tests/test_unified_moe.py::test_cpu_vs_gpu_single_expert` becomes
  trivially pass (both backends read the same bytes). Verify
  CPU bucket numerics remain within `1e-3` of GPU on the mixed-bucket
  parity test.

CI gating: stages B–G blocking; H gated by sanitizer lane; I run on
the SPR+H20 matrix only.

---

## 6. Performance targets and experiments

Target: **≥ 95% of the current packed-kernel throughput** at
M ≥ 32 shapes (the AMX-dominant regime per `analysis.md` §3 table).
Acceptable performance window:

| M    | N    | K    | Packed today | Stride-aware target | Floor       |
|------|------|------|--------------|---------------------|-------------|
| 32   | 4096 | 4096 | 561 GFLOPS   | ≥ 533               | 95 %        |
| 64   | 14336| 4096 | 2675 GFLOPS  | ≥ 2541              | 95 %        |
| 256  | 4096 | 4096 | 2852 GFLOPS  | ≥ 2709              | 95 %        |
| 1    | 4096 | 4096 |  18 GFLOPS   | ≥ 17  (no regression) | unchanged |
| 8    | 4096 | 4096 | 137 GFLOPS   | ≥ 137  (no regression) | unchanged |

The small-M cases (1, 8) are already below AVX2 and the dispatcher
should route them away from AMX anyway (`analysis.md` §5 item 1). The
new kernel only needs to not regress them while in eligibility.

Where the headroom is at risk:

* **Strided L1 prefetch.** Each `_tile_loadd(tile, B + n_begin*K + k,
  K)` reads 16 cache lines that are `K` bytes apart. For
  `K ∈ {1024, 4096, 14336}` the stride is a multiple of 64 (one line)
  but page-crossing every ~64 N rows. Sapphire Rapids's stream
  prefetcher locks onto constant strides up to ~2 KiB; at K = 14336
  the prefetcher may give up after a few lines and rely on L2 demand
  fetch. Mitigation: software prefetch `_mm_prefetch(B + (n_begin +
  TILE_N + …) * K + k_begin, _MM_HINT_T0)` two K-steps ahead inside
  `amx_kernel_rm`.

* **`K_BLOCK` retention in L2.** With the swap, the B k-block stays
  pinned in L2 across the inner M loop. The current `K_BLOCK = 3584`
  was sized so the **A** k-block (M_STEP × K_BLOCK ≈ 112 KiB) fit
  comfortably. The new constraint is on the **B** k-block (N_STEP ×
  K_BLOCK ≈ 112 KiB), which is the same byte budget. So `K_BLOCK`
  stays at 3584.

* **C^T unpack cost.** Scalar transpose adds ~1 cycle per int32 over
  the whole output (~16 KiB for M=64, N=4096). SIMD'd, the cost is
  hidden by store bandwidth.

Bench plan:

1. Extend `bench/bench_amx_vs_avx2.cpp` with the stride-aware path
   under the same shape grid. Output the new column in the markdown
   table.
2. Re-run the unified-MoE autotune that populates `M_CPU(N, K)`. The
   cross-over should move *downward* a few units (Phase 2 removes the
   ~50 µs first-touch tax Phase 1 was paying), validating the
   theoretical analysis in `layout.md` §5.4.
3. Vtune / `perf c2c` snapshot for one decode-sized shape to confirm
   strided B loads hit L2 not DRAM after the first k-block.

---

## 7. Risk register

| # | Risk                                                                                       | Mitigation                                                                                  |
|---|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| 1 | Padded N rows read from raw B fault when the caller's B is not page-padded.                | Last partial `N_BLOCK` stripes are computed against a per-thread 32-row stash that zero-fills rows `[n_real, n_pad)`; we never `_tile_loadd` past `n_real` for the last `N_STEP` of the block. |
| 2 | Software prefetch placement regresses warm-cache throughput.                               | Land prefetches behind `#if CPU_GEMM_AMX_PF_HINT`; default off. Enable only when bench shows ≥1% win on a representative shape.  |
| 3 | C^T → C transpose-in-unpack scalar version too slow.                                       | Stage D passes first with scalar; later optimization replaces the inner 32×32 transpose with two `transpose_16x16_32bit` calls. Drop-in change behind unit test.  |
| 4 | `ldb` other than `k` exposes alignment hazards (`_tile_loadd` requires 64B alignment).      | Dispatcher: `ldb % 64 == 0` *and* `((uintptr_t)b) % 64 == 0`. Else fall through to AVX2 path. |
| 5 | Per-channel B scales arriving on the call seam instead of trailing the packed buffer breaks the existing `int8_unpack_explicit` API. | New `int8_rm_unpack_transposed` takes `b_scales` as an explicit `const float*` (mirrors `int8_unpack_explicit`'s `b_scales` argument; no API churn for the caller). |
| 6 | Bit-identical parity with the packed kernel fails because of FP32 multiply order.          | Pass: `< 1e-6` relative, not bit-identical. INT32 accumulator is exact; only the FP32 scale apply rounds. |
| 7 | Phase rollout breaks `M_CPU` cross-over LUT keyed by the old path.                         | LUT version bump; autotune at first run on new build. Documented in `unified_kernel.md` §11. |

---

## 8. Sizing the change

Lines of code (estimated, including tests):

| File                                                | LoC delta |
|-----------------------------------------------------|----------:|
| `src/kernels/amx/int8_gemm_rm.h` (new)              |   ~80    |
| `src/kernels/amx/int8_gemm_rm.cpp` (new)            |  ~260    |
| `src/dispatch/gemm.cpp`                              |   ~60    |
| `tests/test_amx_int8_rm.cpp` (new)                  |  ~240    |
| `bench/bench_amx_vs_avx2.cpp`                       |   ~30    |
| `csrc_cpu/cpu_module.cpp`                           |   −60    |
| `asym_gemm/unified_moe/runtime.py`                  |   −90    |
| `analysis.md` (mark §5.3 done)                      |   ~10    |
| `int8.md` / `unified_kernel.md` (cross-link)        |   ~20    |
| **Net**                                              | **~+550** |

No header dependencies move; the new kernel reuses `BufferBInt8`,
`BufferAInt8` (renamed inline), `TileConfig`, `transpose_16x16_32bit`.
All AVX-512 intrinsics already in use.

---

## 9. Implementation milestones

| #  | Milestone                                                                              | Exit criterion                                                          |
|----|----------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| 0  | Land `BufferA_VNNI` alias / rewire `BufferBInt8::from_bf16` to accept A's quant inputs | Existing `test_amx_int8` still passes (proves the rewire is a no-op)    |
| 1  | `int8_gemm_rm.{h,cpp}` skeleton + Stage A compile-clean                                | TU builds; throwaway kernel instantiates                                 |
| 2  | Stages B–C (single-tile + multi-K accumulate, scalar transpose unpack)                  | `test_amx_int8_rm.SmokeKB1`, `MultiKB` pass bit-identical                |
| 3  | Stage D (multi-tile + transpose unpack)                                                | `MultiTile` passes within `1e-6` rel                                     |
| 4  | Stage E (stride ≠ K)                                                                   | `StridedB` passes                                                        |
| 5  | Dispatcher hook + Stage F parity-vs-packed                                              | Both backends bit-identical on the §6 shape grid                        |
| 6  | Stage G (multi-thread) + Stage H (guard page)                                          | Determinism + no SIGSEGV under guard harness                            |
| 7  | Bench + autotune refresh                                                               | `bench_amx_vs_avx2` adds the rm column; new `M_CPU(N,K)` LUT             |
| 8  | Wire into unified-MoE runtime; retire `gate_packed/up_packed/down_packed`              | `test_cpu_vs_gpu_single_expert` and mixed-bucket parity pass             |
| 9  | Retire `run_amx_int8` (packed-via-scratch) and `pack_b_int8_amx` pybind                | All callers route to the rm path or `CG_INT8_PACKED_AMX` (offline pack) |

Milestones 0–4 are local to `cpu_gemm`; 5–7 are dispatch-level; 8–9
touch the AsymGEMM repo. Milestones 1–6 are the critical path.

---

## 10. What this plan deliberately does not change

* AMX BF16 paths (`bf16_gemm.{h,cpp}`, `bf16_buffers.h`). Same packed
  layout, same dispatcher entry, unchanged numerics.
* SM90 INT8 GPU kernel. No edits to `sm90_int8_asym_gemm_1d1d.cuh` or
  its launcher. The unification is host-side.
* Per-token activation quant on the GPU side
  (`quantize_per_token_int8_gpu`). Untouched.
* `M_CPU` dispatch policy *mechanism* (the LUT plumbing); only the
  measured *values* shift downward after Phase 2 lands.
* The `CG_INT8_PACKED_AMX` offline-pack path. Stays for callers that
  pre-packed at load time; the unified MoE runtime simply stops using
  it.

---

## 11. Appendix — file pointers

| Topic                                          | File / line                                                                            |
|------------------------------------------------|-----------------------------------------------------------------------------------------|
| Current packed AMX kernel                      | `third-party/cpu_gemm/src/kernels/amx/int8_gemm.cpp:30-133`                              |
| Tile config (unchanged for rm path)            | `third-party/cpu_gemm/src/kernels/amx/int8_gemm.cpp:55-61`                               |
| BufferBInt8 (becomes `BufferA_VNNI` for rm)    | `third-party/cpu_gemm/src/kernels/amx/int8_buffers.h:138-322`                            |
| `transpose_16x16_32bit` (reused in unpack)     | `third-party/cpu_gemm/src/kernels/amx/amx_utils.h:57-126`                                |
| Dispatcher INT8 eligibility (relax `ldb==k`)   | `third-party/cpu_gemm/src/dispatch/gemm.cpp:94-114, 172-222`                             |
| Existing `int8_unpack_explicit` (model for new) | `third-party/cpu_gemm/src/kernels/amx/int8_gemm.cpp:223-264`                             |
| Stride-aware AMX called out as future work     | `third-party/cpu_gemm/analysis.md:165-193`                                               |
| Unified-runtime dual layout being retired       | `asym_gemm/unified_moe/runtime.py:155-322`                                              |
| Layout-level Phase 2 rationale                 | `layout.md` §5.3, §5.4                                                                   |
| Bench harness to extend                        | `third-party/cpu_gemm/bench/bench_amx_vs_avx2.cpp`                                       |
| Guard-page harness reference                   | `third-party/cpu_gemm/analysis.md` §4.1                                                  |
