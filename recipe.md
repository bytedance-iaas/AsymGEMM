# NVFP4 AsymGEMM: Recipe Explained

This document traces the full data-flow of the NVFP4 grouped GEMM — from the raw float tensors in the test
(`test_nvfp4.py`) through quantization, scale-factor layout transformation, and kernel execution inside
`sm100_fp4_asym_gemm_1d1d.cuh`.

---

## 1. The `recipe` Tuple — `(gran_mn_a, gran_mn_b, gran_k)`

```python
recipe = (1, 16, 16)          # used in both contiguous and masked tests
disable_ue8m0_cast = True
```

The recipe is a 3-tuple that controls *scale-factor granularity*:

| Index | Name       | Value | Meaning |
|-------|------------|-------|---------|
| 0     | `gran_mn_a`| 1     | Every row of **A** has its own scale (per-token) |
| 1     | `gran_mn_b`| 16    | Every 16 rows of **B** share one scale (block along N) |
| 2     | `gran_k`   | 16    | Every 16 K-elements share one scale (matches SM100 UMMA granularity) |

The default recipe for FP4+E4M3 scales on SM100 (from `csrc/utils/layout.hpp`) is `(1, 1, 16)`.
The test uses `(1, 16, 16)` which relaxes the B-matrix N-granularity from per-element to per-16-rows
(standard block scaling), matching NVIDIA's MX4 spec.

`disable_ue8m0_cast = True` keeps scale-factor bytes in **E4M3** format throughout — no conversion
to UE8M0 (unsigned exponent-8, zero mantissa) is performed. The kernel path in `transform_sf_into_required_layout`
used is:

```cpp
// (E4M3, 1/16, 16) on SM100 FP4 kernels:
if (sf.scalar_type() == torch::kFloat8_e4m3fn and (gran_mn == 1 or gran_mn == 16)
    and gran_k == 16 and arch_major == 10)
    return get_mn_major_tma_aligned_packed_byte_sf_tensor(sf);
```

---

## 2. Data Formats

### 2.1 NVFP4 — E2M1 Format

Each FP4 value is 4 bits: `[sign(1) | exp(2) | mantissa(1)]`.

The 8 positive magnitudes are: `{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`.

Two E2M1 values are packed into one `uint8` byte (low nibble = even element, high nibble = odd element):
- `packed[c//2] = code_even & 0x0F`
- `packed[c//2] |= (code_odd & 0x0F) << 4`

One 8-byte unit holds **16 FP4 elements** (= 2 NVFP4 "packed units" of 8 elements each).

### 2.2 Scale Factors — FP8 E4M3 Format

Scale factors are stored as `torch.float8_e4m3fn` (NVIDIA FP8 E4M3 with no infinities, NaN=0x7F/0xFF).
Each scale covers a **16-element group** in the K dimension.

Encoding: `sf = amax / 6.0` → quantized to E4M3 (the divisor 6.0 = max positive E2M1 magnitude).

---

## 3. Quantization Procedure

### 3.1 A-matrix (Activations) — Per-token, per-K-group

Shape: `A_f32 [M, K]` → `A_packed [M, K//2]` (uint8) + `SFA [M, ceil(K/16)]` (E4M3)

```python
# For each row `i` and K-group `g`:
chunk = A[i, g*16 : (g+1)*16]
amax  = max(|chunk|, 1e-4)
sf    = amax / 6.0                         # float scale
sf_e4m3 = quantize_to_fp8_e4m3(sf)        # stored as uint8
sf_decoded = decode_fp8_e4m3(sf_e4m3)     # used for encoding (or 1.0 if zero/nan)

for each element c in the group:
    code = encode_e2m1(A[i, c] / sf_decoded)
    pack into A_packed[i, c//2]            # low/high nibble
```

C++ equivalent: `quantize_bf16_to_nvfp4()` in `asymCompKernelMain_fp4.cu`.

### 3.2 B-matrix (Weights) — Per-block (N-group × K-group)

Shape: `B_f32 [N, K]` → `B_packed [N, K//2]` + `SFB [ceil(N/16), ceil(K/16)]` (E4M3)

```python
# For each N-group `gn` and K-group `gk`:
chunk = B[gn*16:(gn+1)*16, gk*16:(gk+1)*16]
amax  = max(|chunk|, 1e-4)
sf    = amax / 6.0
# store one scale per (gn, gk) block
for each element (r, c) in the block:
    code = encode_e2m1(B[r, c] / sf_decoded)
    pack into B_packed[r, c//2]
```

Note the asymmetry: **A is per-row** (1D scale along K), **B is 2D block-scaled** (separate scale per N×K tile).

---

## 4. Scale Factor Layout Transformation

Before the kernel can use the scale factors, they must be reformatted into the layout that TMA and
the UTCCP hardware expect. This is done in `csrc/apis/layout.hpp`:
`transform_sf_into_required_layout(sf, mn, k, recipe, num_groups, is_sfa, disable_ue8m0_cast)`

### Step A — `get_mn_major_tma_aligned_packed_byte_sf_tensor`

For `E4M3` scales with `(gran_mn=1/16, gran_k=16)`:

1. View the E4M3 tensor as raw `uint8` bytes.
2. Pad MN dimension to TMA-aligned size (multiple of `tma_align_bytes / sizeof(element)`).
3. Pack 4 consecutive E4M3 bytes (covering 4 consecutive K-groups of 16 = 64 K-elements) into one `int32`.
4. Store in **MN-major** strided layout: stride(-2)=1 (MN stride), stride(-1)=aligned_mn (K-packed stride).

Result shape: `[aligned_mn, ceil(K/16)/4]` as `int32`, MN-major.

This packed layout is what the UTCCP instruction reads from shared memory and what TMA loads.

### Step B — `broadcast_packed_ue8m0_sf` (K granularity broadcast, called in `gemm.hpp`)

```cpp
const int sf_gran_k    = std::get<2>(recipe.value());  // = 16
const int sf_replication = sf_gran_k / fp4_sf_quant_k; // = 16/16 = 1
```

With `gran_k=16` and `fp4_sf_quant_k=16`, `sf_replication=1` — no broadcast is needed.
(If `gran_k=128`, each packed byte would be replicated 8× to expand to 16-element granularity.)

### Step C — MN broadcast for B (when `gran_mn_b > 1`)

For `gran_mn_b=16` in the recipe, `sfb` has shape `[ceil(N/16), ceil(K/16)/4]` which needs to be expanded
to match the N dimension of B for kernel addressing:

```cpp
if (gran_mn_b > 1 && sfb.size(-2) < n) {
    idx = arange(n).floor_divide_(16);    // map each row to its group
    sfb = sfb.index_select(-2, idx);      // repeat: [N, ceil(K/16)/4]
    sfb = empty_strided(..., MN-major);   // re-stride TMA-aligned
}
```

After all transforms, `sfa` has shape `[M, ceil(K/64)]` int32 MN-major, and `sfb` has shape `[G, N, ceil(K/64)]` int32 MN-major.

---

## 5. Kernel Internals: `sm100_fp4_asym_gemm_1d1d`

### 5.1 Template Parameters (key ones)

```
BLOCK_M, BLOCK_N, BLOCK_K  — tile sizes
kSFQuantK = 16              — HW granularity of FP4 UMMA scale lookup
kNumSFPerPack = 4           — 4 E4M3 bytes per int32 pack
kSFPacksPerBlockK = BLOCK_K / (kSFQuantK * kNumSFPerPack) = BLOCK_K / 64
```

For `BLOCK_K=256`: `kSFPacksPerBlockK = 256/64 = 4`.

### 5.2 Shared Memory Layout

From lowest to highest address:

```
[SMEM_CD]       — output staging (2 TMA-store stages × STORE_BLOCK_M × kSwizzleCDMode)
[SMEM_A × stages]  — FP4 A tiles (LOAD_BLOCK_M × BLOCK_K/2 bytes per stage)
[SMEM_B × 1]    — FP4 B tile (LOAD_BLOCK_N × BLOCK_K/2 bytes, single-buffered)
[SMEM_SFA × stages] — A scale factors (SF_BLOCK_M × kSFPacksPerBlockK int32 per stage)
[SMEM_SFB × 1]  — B scale factors (SF_BLOCK_N × kSFPacksPerBlockK int32)
[Barriers]      — all pipeline barriers
[tmem_ptr]      — tensor memory address (uint32)
```

### 5.3 Tensor Memory Layout

```
[0 .. kNumAccumTmemCols)      — FP32 accumulators (kNumEpilogueStages × kNumMWaves × BLOCK_N columns)
[kTmemStartColOfSFA ..)       — SFA in tensor memory (SF_BLOCK_M/32 × kSFPacksPerBlockK columns)
[kTmemStartColOfSFB ..)       — SFB in tensor memory (SF_BLOCK_N/32 × kSFPacksPerBlockK columns)
```

### 5.4 Warp Roles

The kernel assigns each warp a fixed role at dispatch time:

#### Warp 0 — TMA Load (B-centric sweep)

For each K block (outer), for each M block in this CTA's segment (inner):

1. **B tile load** (once per K block): `tma_copy<BLOCK_K/2, LOAD_BLOCK_N>(&tensor_map_b, full_barriers_b, smem_b[0], k_packed_idx, b_n_idx)`
2. **SFB load** (once per K block): `tma_copy<BLOCK_N, kSFPacksPerBlockK>(&tensor_map_sfb, full_barriers_b, smem_sfb[0], n_block*BLOCK_N, group*shape_sf_k + sf_k_idx)`
3. **A tile load** (once per M block per K block): `tma_copy<BLOCK_K/2, LOAD_BLOCK_M>(&tensor_map_a, full_barriers[stage], smem_a[stage], k_packed_idx, m_idx)`
4. **SFA load** (once per M block per K block): `tma_copy<BLOCK_M, kSFPacksPerBlockK>(&tensor_map_sfa, full_barriers[stage], smem_sfa[stage], local_m_idx, sfa_k_idx)`

Both B+SFB fire a combined `arrive_and_expect_tx(SMEM_B_SIZE + BLOCK_N×kSFPacksPerBlockK×4)`.
Both A+SFA fire a combined `arrive_and_expect_tx(SMEM_A_SIZE + BLOCK_M×kSFPacksPerBlockK×4)`.

#### Warp 2 — UTCCP Transposer

UTCCP (Unit for Tensor Core Co-Processor) requires a specific memory layout. Warp 2 transposes
SFA and SFB in shared memory before they can be copied to tensor memory:

```cpp
// Transpose a 128×128-bit block (4×32 uint32 → 32×4 uint32)
// Input:  smem[row*32 + lane_idx] for row in 0..3 (with XOR shuffle)
// Output: smem[lane_idx*4 + (i ^ (lane_idx>>3))] for i in 0..3
```

After transpose, warp 2 signals `with_sf_full_barriers[stage]` (for SFA) and
`with_sf_full_barriers_b[0]` (for SFB).

#### Warp 1 (leader CTA) — MMA Issue

For each K block → for each M block:

1. Wait `with_sf_full_barriers_b[0]` (SFB transposed and ready).
2. Copy SFB from shared memory → tensor memory via `SM100_UTCCP_4x32dp128bit`:
   ```cpp
   for p in 0..kSFPacksPerBlockK:
       for i in 0..SF_BLOCK_N/128:
           utccp::copy(smem_sfb + p*SF_BLOCK_N + i*128, kTmemStartColOfSFB + p*(SF_BLOCK_N/32) + i*4)
   ```
3. Wait `with_sf_full_barriers[stage]` (SFA transposed and ready).
4. Copy SFA from shared memory → tensor memory similarly.
5. Issue block-scaled UMMA: iterates over `NUM_K_ATOMS × UMMA_ITERS_PER_ATOM`:
   ```cpp
   // sf_id selects which E4M3 byte within a packed int32 to use for this UMMA step
   sf_id = (k_block * kSFAtomsPerBlockK + atom*(DESC_ATOM_K/16) + ki*(UMMA_K/16)) % kNumSFPerPack
   instr_desc_with_sf = make_runtime_instr_desc_with_sf_id(instr_desc, sf_id)

   SM100_MMA_MXF4NVF4_SS::fma(
       a_desc, b_desc,
       tmem_col = accum_stage*kNumMWaves*BLOCK_N + wave*BLOCK_N,
       accumulate = (atom > 0 || ki > 0),    // false only for first step
       instr_desc_with_sf,
       tmem_sfa_col, tmem_sfb_col
   )
   ```

The `MXF4NVF4` instruction performs block-scaled FP4 matrix multiply:
`D[m,n] += dequant(A[m,k]) * dequant(B[k,n])` where dequantization uses the tensor-memory scales.

#### Epilogue Warps — TMEM → SMEM → Global

For each M block (after MMA completes):

1. Load FP32 accumulator from tensor memory (`SM100_TMEM_LOAD_32dp32b4x` or `8x`).
2. Cast to BF16 (or keep FP32) and store to shared memory with swizzled bank addressing.
3. Issue TMA store to global memory:
   - K-block 0: `SM90_TMA_STORE_2D` (plain write, initialises the tile)
   - K-block > 0: `SM90_TMA_REDUCE_ADD_2D` (accumulate partial sums in global memory)

---

## 6. End-to-End Data-Flow Summary

```
BF16 A [M, K]                          BF16 B [G, N, K]
     │                                        │
     ▼  quantize_a (per-row, gran_k=16)       ▼  quantize_b (block, gran_n=16, gran_k=16)
A_packed [M, K/2] uint8                B_packed [G, N, K/2] uint8
SFA      [M, K/16] E4M3                SFB      [G, N/16, K/16] E4M3
     │                                        │
     ▼  transform_sf (pack bytes, MN-major)   ▼  same + broadcast N dim
SFA_tmem [M, K/64] int32 MN-major      SFB_tmem [G, N, K/64] int32 MN-major
     │                                        │
     └───────────────┬────────────────────────┘
                     │  Kernel call: m_grouped_fp4_asym_gemm_nt_contiguous
                     │
         ┌───────────▼────────────────────────────────┐
         │  Warp 0 (TMA):                             │
         │    A tile + SFA → SMEM                     │
         │    B tile + SFB → SMEM (once per K-block)  │
         │                                            │
         │  Warp 2 (UTCCP transpose):                 │
         │    SFA/SFB in SMEM: 128-bit block transpose│
         │                                            │
         │  Warp 1 (MMA):                             │
         │    SFA/SFB: SMEM → TMEM (via UTCCP)        │
         │    A+B in SMEM → UMMA block-scaled FP4 MMA │
         │    Result accumulates into TMEM (FP32)     │
         │                                            │
         │  Epilogue warps:                           │
         │    TMEM FP32 → SMEM BF16 (swizzled)        │
         │    SMEM → global D via TMA store/reduce    │
         └────────────────────────────────────────────┘
                     │
                     ▼
              D [M, N] BF16
```

---

## 7. Correctness Status

### What is verified

- `test_m_grouped_nvfp4_contiguous_cpp_flow`: Checks NaN/Inf in both kernel output and manual reference.
- `test_m_grouped_nvfp4_masked_cpp_flow`: Same NaN/Inf check per group.
- Both tests print `kernel_vs_manual`, `kernel_vs_gt`, and `manual_vs_gt` diff stats.

### What is NOT asserted (disabled)

The numerical correctness gate is **commented out**:

```python
# assert kernel_vs_manual["max_abs"] < 1.0, f"kernel_vs_manual max_abs too large: ..."
```

This means the tests pass as long as the kernel does not produce NaN/Inf, but numerical accuracy
vs the manual NVFP4 dequant reference is not enforced.

### Known correctness concerns

1. **`sf_id` computation**: The formula
   ```cpp
   sf_id = (k_block * kSFAtomsPerBlockK + atom*(DESC_ATOM_K/16) + ki*(UMMA_K/16)) % kNumSFPerPack
   ```
   maps each 16-element K-segment to a byte within the packed int32 scale factor.
   With `gran_k=16` and `kNumSFPerPack=4`, this cycles `sf_id ∈ {0,1,2,3}`.
   Must match the byte packing order in `get_mn_major_tma_aligned_packed_byte_sf_tensor`.

2. **TMA accumulation split**: K-block 0 uses a plain TMA store, subsequent blocks use `TMA_REDUCE_ADD`.
   This is correct for multi-K-block GEMMs where the tile is larger than BLOCK_K.

3. **`gran_mn_b=16` broadcast**: After the MN-broadcast step in `gemm.hpp`, `sfb` has full N rows,
   so each of the N rows within a B-group correctly maps to its 16-row scale group.
   The kernel sees `SFB [G, N, K/64]` as if it were per-row — the repeated rows ensure correct lookup.

4. **No subbyte address correctness for FP4**: The kernel uses `BLOCK_K/2` as the byte-count
   for TMA loads (correct: 2 FP4 elements per byte), and divides by 2 again in UMMA descriptor
   construction (`DESC_ATOM_K/2`) to pass byte counts — consistent with `sizeof(fp4_input_element_t)==1`
   but packing two FP4 values per byte.

### How to re-enable the correctness gate

Uncomment the assertion in `test_nvfp4.py`:

```python
assert kernel_vs_manual["max_abs"] < 1.0, f"kernel_vs_manual max_abs too large: {kernel_vs_manual}"
```

A value of `max_abs < 1.0` is appropriate given the quantization noise introduced by E2M1 (max value 6.0)
and E4M3 scale factor rounding.
