# FP4 Recipe Shape and Assertion Error — Analysis & Fix Plan

---

## 1. Recipe Shape Derived from `fp4_model.json`

`fp4_model.json` contains:
```json
"input_activations": { "num_bits": 4, "type": "float", "group_size": 16 },
"weights":           { "num_bits": 4, "type": "float", "group_size": 16 }
```

`group_size: 16` means every 16 elements in the **K dimension** share one scale factor.  
For activations (A), quantization is per-token (per row), so there is no MN grouping: `gran_mn_a = 1`.  
For weights (B), modelopt NVFP4 also quantizes per-row (not 2D block): `gran_mn_b = 1`.

Therefore the production recipe is:

```python
recipe = (1, 1, 16)   # (gran_mn_a, gran_mn_b, gran_k)
```

This matches the default returned by `get_default_recipe` in `csrc/utils/layout.hpp` for
`sfb_dtype == torch::kFloat8_e4m3fn` on SM100:
```cpp
if (sfb_dtype == torch::kFloat8_e4m3fn)
    return {1, 1, 16};   // FP4 native E4M3 scales
```

---

## 2. Root Cause of the Assertion Error

```
RuntimeError: Assertion error (csrc/utils/layout.hpp:80):
sf.size(-2) == ceil_div(mn, gran_mn)
```

The assertion is inside `check_sf_layout`, called from `transform_sf_into_required_layout` for SFB:

```cpp
// layout.hpp line 80
DG_HOST_ASSERT(sf.size(-2) == ceil_div(mn, gran_mn));
//  mn       = n         = 1024
//  gran_mn  = gran_mn_b = 1       (from recipe[1])
//  expected = ceil_div(1024, 1)   = 1024
```

But in `test_nvfp4.py` the B scale tensor is allocated as:

```python
sf_n = (n + gran_k - 1) // gran_k   # = ceil(1024 / 16) = 64  ← 2D block scaling
b_scales_u8 = np.empty((num_groups, sf_n, sf_k), dtype=np.uint8)
# sfb.size(-2) = sf_n = 64  ≠  n = 1024
```

`_quantize_b_nvfp4_e4m3` computes one scale per **16-row × 16-column block** of B (i.e. `gran_mn_b = 16`),
producing SFB shape `[ceil(N/16), ceil(K/16)]`.  
The recipe `(1, 1, 16)` expects one scale **per row** of B (i.e. `gran_mn_b = 1`),
requiring SFB shape `[N, ceil(K/16)]`.

The mismatch: `sf_n = 64` when `n = 1024` → assertion `64 == 1024` fails.

The same mismatch exists in both test functions:

| Location | Variable | Current shape (2D block) | Required shape (per-row) |
|---|---|---|---|
| `test_m_grouped_nvfp4_contiguous_cpp_flow` | `b_scales_u8` | `(G, N/16, K/16)` | `(G, N, K/16)` |
| `test_m_grouped_nvfp4_masked_cpp_flow` | `b_scales_u8` | `(G, N/16, K/16)` | `(G, N, K/16)` |
| `asymCompKernelMain_fp4.cu` | `b_quants[g].scales` | `(G, N/16, K/16)` | `(G, N, K/16)` |

---

## 3. Fix Plan

### Step 1 — Change `_quantize_b_nvfp4_e4m3` to per-row quantization

**File**: `tests/test_nvfp4.py`, lines 163–193

**Current behaviour** (2D block, `gran_mn_b = 16`):
```python
def _quantize_b_nvfp4_e4m3(b_f32, gran_k):
    n, k = b_f32.shape
    sf_n = (n + gran_k - 1) // gran_k          # ← ceil(N/16) rows of scales
    scales = np.zeros((sf_n, sf_k), ...)        # ← 2D block shape

    for gn in range(sf_n):                      # ← iterates over N-blocks
        for gk in range(sf_k):
            chunk = b_f32[gn*gran_k:(gn+1)*gran_k, gk*gran_k:(gk+1)*gran_k]
            ...
            scales[gn, gk] = sf_bits
```

**Required behaviour** (per-row, `gran_mn_b = 1`):
```python
def _quantize_b_nvfp4_e4m3(b_f32, gran_k):
    n, k = b_f32.shape
    sf_k = (k + gran_k - 1) // gran_k
    packed = np.zeros((n, k // 2), dtype=np.uint8)
    scales = np.zeros((n, sf_k), dtype=np.uint8)    # ← one scale row per B row

    for r in range(n):                              # ← iterate over every row
        for gk in range(sf_k):
            c0 = gk * gran_k
            c1 = min(c0 + gran_k, k)
            chunk = b_f32[r, c0:c1]                 # ← single row, K-group slice
            amax = max(float(np.max(np.abs(chunk))), 1e-4)
            sf = amax / 6.0
            sf_bits, sf_decoded = _to_e4m3_bits_and_decoded(sf)
            scales[r, gk] = sf_bits
            for c in range(c0, c1):
                code = _encode_e2m1_scalar(float(b_f32[r, c] / sf_decoded))
                pidx = c // 2
                if (c & 1) == 0:
                    packed[r, pidx] = code & 0x0F
                else:
                    packed[r, pidx] |= (code & 0x0F) << 4
    return packed, scales
```

### Step 2 — Update allocation shapes in both contiguous and masked tests

**File**: `tests/test_nvfp4.py`

In `test_m_grouped_nvfp4_contiguous_cpp_flow` (around line 338):
```python
# Before
sf_n = (n + gran_k - 1) // gran_k      # = 64
recipe = (1, 16, 16)
b_scales_u8 = np.empty((num_groups, sf_n, sf_k), dtype=np.uint8)

# After
sf_n = n                                # = 1024
recipe = (1, 1, 16)
b_scales_u8 = np.empty((num_groups, sf_n, sf_k), dtype=np.uint8)
```

In `test_m_grouped_nvfp4_masked_cpp_flow` (around line 460):
```python
# Before
sf_n = (n + gran_k - 1) // gran_k
recipe = (1, 16, 16)
b_scales_u8 = np.empty((num_groups, sf_n, sf_k), dtype=np.uint8)

# After
sf_n = n
recipe = (1, 1, 16)
b_scales_u8 = np.empty((num_groups, sf_n, sf_k), dtype=np.uint8)
```

### Step 3 — Update `_manual_dequant_reference` to use per-row B scales

**File**: `tests/test_nvfp4.py`, lines 243–246

The current dequantization expands 2D block scales `[sf_n, sf_k]` → `[n, k]` by repeating both
axes. With per-row scales `[n, sf_k]`, only the K axis needs repeating:

```python
# Before (2D block: repeat both N and K axes)
b_sf = _decode_e4m3_bits(b_scales_u8[gid])            # shape: (sf_n, sf_k)
b_sf = np.repeat(np.repeat(b_sf, gran_k, axis=0), gran_k, axis=1)[:n, :k]

# After (per-row: repeat only the K axis)
b_sf = _decode_e4m3_bits(b_scales_u8[gid])            # shape: (n, sf_k)
b_sf = np.repeat(b_sf, gran_k, axis=1)[:, :k]
```

The same inline dequantisation in `test_m_grouped_nvfp4_masked_cpp_flow` (lines 540–543) needs the
identical change:
```python
# Before
b_sf = _decode_e4m3_bits(b_scales_u8[gid])
b_sf = np.repeat(np.repeat(b_sf, gran_k, axis=0), gran_k, axis=1)[:n, :k]

# After
b_sf = _decode_e4m3_bits(b_scales_u8[gid])
b_sf = np.repeat(b_sf, gran_k, axis=1)[:, :k]
```

### Step 4 — Update `asymCompKernelMain_fp4.cu`

**File**: `tests/asymCompKernelMain_fp4.cu`, lines 194–251

`quantize_bf16_to_nvfp4_block` uses 2D block scaling.  Change it to per-row:

```cpp
// Before
struct NvFP4BlockQuantResult {
    std::vector<uint8_t> scales;  // [ceil(n/gran_k) * ceil(k/gran_k)]
    int64_t num_groups_n;         // ceil(n/gran_k)
    ...
};

static NvFP4BlockQuantResult quantize_bf16_to_nvfp4_block(...) {
    const int64_t num_groups_n = (n + gran_k - 1) / gran_k;  // N/16 scale rows
    result.scales.resize(num_groups_n * num_groups_k);
    for (int64_t gn = 0; gn < num_groups_n; ++gn)            // iterates N-blocks
        for (int64_t gk = 0; gk < num_groups_k; ++gk) {
            chunk = b[gn*gran_k..(gn+1)*gran_k, gk*gran_k..(gk+1)*gran_k];
            ...
            result.scales[gn * num_groups_k + gk] = sf_fp8;
        }
}

// After — per-row
static NvFP4BlockQuantResult quantize_bf16_to_nvfp4_block(...) {
    result.scales.resize(n * num_groups_k);   // n × K/16 (one scale row per B row)
    for (int64_t r = 0; r < n; ++r)           // iterate every row
        for (int64_t gk = 0; gk < num_groups_k; ++gk) {
            chunk = b[r, gk*gran_k..(gk+1)*gran_k];   // single row, K-group
            ...
            result.scales[r * num_groups_k + gk] = sf_fp8;
        }
}
```

Also update `compute_manual_nvfp4_e4m3_reference` which reads `b_scales` indexed as `b_scales[g][(r / gran_k) * sf_k + (c / gran_k)]` — change the row index from `r / gran_k` to `r`:
```cpp
// Before
const float sf = b_scales[g][(r / gran_k) * sf_k + (c / gran_k)];

// After
const float sf = b_scales[g][r * sf_k + (c / gran_k)];
```

And update the `recipe` variable and `sf_n` in `main()` (lines 531–538):
```cpp
// Before
const std::optional<std::tuple<int,int,int>> recipe = std::make_tuple(1, 16, 16);
const int64_t sf_n = (n + gran_k - 1) / gran_k;   // = 64

// After
const std::optional<std::tuple<int,int,int>> recipe = std::make_tuple(1, 1, 16);
const int64_t sf_n = n;                            // = 1024
```

Also change the `SFB_u8_cpu` allocation size from `{num_groups, sf_n, sf_k}` (where `sf_n=64`) to
`{num_groups, n, sf_k}` (where `n=1024`), and update the `memcpy` loop to write `n * sf_k` bytes
per group.

### Step 5 — Verify that the kernel path remains unchanged

With `recipe = (1, 1, 16)` and `gran_mn_b = 1`, no MN-broadcast is needed in `gemm.hpp`:

```cpp
// This branch in m_grouped_fp4_asym_gemm_nt_contiguous is SKIPPED (gran_mn_b == 1):
if (gran_mn_b > 1 && sfb.size(-2) < n) { ... broadcast ... }
```

The SFB tensor passed into `sm100_m_grouped_fp4_asym_gemm_contiguous_1d1d` has shape
`[G, N, ceil(K/64)]` (int32, MN-major, TMA-aligned) — same shape the kernel already expects.
**No kernel changes are required.**

The kernel's TMA load for SFB:
```cpp
tma_copy<BLOCK_N, kSFPacksPerBlockK, 0>(&tensor_map_sfb, full_barriers_b[0], smem_sfb[0],
    n_block_idx * BLOCK_N,
    scheduler.current_group_idx * shape_sf_k + sf_k_idx);
```
correctly loads `BLOCK_N` contiguous scale rows starting at `n_block_idx * BLOCK_N` —
this works identically whether the rows come from a broadcast copy (`gran_mn_b=16`) or are
original per-row data (`gran_mn_b=1`).

---

## 4. Summary Table

| File | Location | Change |
|---|---|---|
| `tests/test_nvfp4.py` | `_quantize_b_nvfp4_e4m3` | Rewrite: per-row scales, `scales.shape = (n, sf_k)` |
| `tests/test_nvfp4.py` | `_manual_dequant_reference` lines 243-246 | Remove axis-0 repeat, keep only axis-1 repeat |
| `tests/test_nvfp4.py` | masked test inline dequant, lines 540-543 | Same: remove axis-0 repeat |
| `tests/test_nvfp4.py` | contiguous test, line 337-339 | `sf_n = n`, `recipe = (1, 1, 16)`, `b_scales_u8` shape `(G, n, sf_k)` |
| `tests/test_nvfp4.py` | masked test, line 458-460 | `sf_n = n`, `recipe = (1, 1, 16)`, `b_scales_u8` shape `(G, n, sf_k)` |
| `tests/asymCompKernelMain_fp4.cu` | `quantize_bf16_to_nvfp4_block` | Rewrite: per-row, `scales.size = n * sf_k` |
| `tests/asymCompKernelMain_fp4.cu` | `compute_manual_nvfp4_e4m3_reference` | Row-index from `r/gran_k` → `r` |
| `tests/asymCompKernelMain_fp4.cu` | `main()` lines 531-538 | `sf_n = n`, `recipe = (1, 1, 16)`, update SFB alloc/memcpy |
| kernel `.cuh` / `gemm.hpp` | — | No changes needed |

---

## 5. Why `(1, 16, 16)` Appeared to Work Before

The test was using `recipe = (1, 16, 16)` with `gran_mn_b = 16`, which matched the 2D block
quantization (`sf_n = ceil(N/16)` scale rows). That recipe is numerically valid but **not what the
production model uses** — `fp4_model.json` implies `gran_mn_b = 1` (per-row B scales).

The `recipe = (1, 16, 16)` path triggered the MN-broadcast in `gemm.hpp`, which expanded the 64
scale rows back to 1024 before handing off to the kernel. That broadcast is correct arithmetic
but hides the mismatch between the quantization granularity and what the model actually stores.

Using the correct `recipe = (1, 1, 16)` removes the broadcast and requires the test to supply
SFB with the full `N` rows from the start, which is what this fix plan achieves.
