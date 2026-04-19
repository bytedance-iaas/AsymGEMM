# AsymGEMM FP8 Kernel Optimization Plan

**Target**: `sm100_fp8_asym_gemm_1d1d` on 4× GB200 (SM100) for MoE GEMM  
**Date**: 2026-04-02  
**Hardware**: 132 SMs per GPU, HBM3e ~8 TB/s, NVLink-C2C CPU↔GPU ~900 GB/s

---

## Executive Summary

The kernel already uses B-centric traversal (load B once per K-block, sweep all M-tiles), which is the right strategy for GB200 where B arrives from CPU via NVLink-C2C. Three concrete performance gaps remain:

1. **Hard-coded suboptimal tile sizes**: `(block_m=128, block_n=128, block_k=512)` generates 4 CTA waves for the critical `(8 groups, N=7168, K=2048)` shape → ~85% SM utilization. Switching to `block_n=224, block_k=256` gives 2 waves → ~97% utilization because `7168/224 = 32` tiles exactly.

2. **Auto-tuner bypassed + stray debug printf**: `get_best_config_asym` is available but bypassed by the hard-coded config. It also has a stray `printf` at `common.hpp:520` that must be removed.

3. **B SMEM uses only 1 stage**: B cannot be prefetched for K+1 while computing K. Adding a second B stage (while reducing A stages from 2→1) allows overlap of NVLink transfer with MMA for the next K-block. SMEM budget: 1 A-stage (64KB) + 2 B-stages (128KB) + CD (32KB) + SF/barriers (~4KB) = 228KB < 232KB limit.

Estimated overall improvement on the worst-case shape: **~1.7–1.9× TFLOPS**.

---

## Wave Efficiency Analysis

GB200 has 132 SMs. For `(8 groups, N=7168, K=2048)`:

| block_n | N-tiles | × 8 groups | Waves | Last-wave SMs | Efficiency |
|---------|---------|------------|-------|---------------|------------|
| 128     | 56      | 448 CTAs   | 4     | 112/132       | **85%**    |
| 192     | 38*     | 304 CTAs   | 3     | 40/132        | 77% (bad)  |
| 224     | 32      | 256 CTAs   | 2     | 124/132       | **97%**    |
| 256     | 28      | 224 CTAs   | 2     | 92/132        | 85% (+illegal TMEM) |

*7168 not divisible by 192 → ceil gives 38 tiles, terrible last-wave.  
7168 / 224 = 32 exactly → block_n=224 is the sweet spot.

**TMEM check** for block_n=224, block_m=128 (FP8):  
`2×224 + align(128,128)/32 + align(224,128)/32 = 448 + 4 + 8 = 460 ≤ 512` ✓  
→ 2 epilogue stages available (kNumEpilogueStages=2)

**SMEM check** for (128, 224, 256, 2 A-stages, 1 B-stage):  
`2×(128×256) + 224×256 + 2×(128×128) + SF = 64+56+32+4 ≈ 156 KB < 232 KB` ✓

---

## kBlockKPerSFLoad: SF Load Frequency

```
kSFAtomsPerBlockK = block_k / 128
kBlockKPerSFLoad  = 4 / kSFAtomsPerBlockK
```

- `block_k=512`: `kSFAtomsPerBlockK=4`, `kBlockKPerSFLoad=1` → SF loaded **every** K-block
- `block_k=256`: `kSFAtomsPerBlockK=2`, `kBlockKPerSFLoad=2` → SF loaded every **2** K-blocks

With block_k=256, SFB transpose (UTCCP) happens half as often. SFA transpose count stays the same (per M-tile, per SF load event).

---

## Phase 1 — Baseline + Test Expansion

**File**: `tests/test_fp8.py` lines 499–502  
Uncomment all 4 test functions and enable the masked GEMM performance benchmark (lines 438–450).

**File**: `tests/generators.py` line 148  
Add 16-group and 32-group shapes:

```python
((4,  8192, 4096,  512), (4,  8192, 7168, 2048),
 (8,  4096, 4096, 7168), (8,  4096, 7168, 2048),
 (16, 2048, 4096, 7168), (16, 2048, 7168, 2048),
 (32, 1024, 4096, 7168), (32, 1024, 7168, 2048))
```

**Run baseline**:
```bash
cd /asymGEMMFP8/AsymGEMM && bash install.sh
python3 tests/test_fp8.py 2>&1 | tee logs/baseline.log
```

---

## Phase 2 — Remove Debug Printf + Enable Auto-Tuner

### 2.1 Remove stray printf

**File**: `csrc/jit_kernels/heuristics/common.hpp` line ~520  
Remove the line:
```cpp
printf("candidate_smem_config.smem_size: %d", candidate_smem_config.smem_size);
```

### 2.2 Enable auto-tuner in contiguous path

**File**: `csrc/jit_kernels/impls/sm100_fp8_asym_gemm_1d1d.hpp` lines ~104–106  
Change hard-coded values to 0 to fall through to `get_best_config_asym`:
```cpp
const int block_m = 0;
const int block_n = 0;
const int block_k = 0;
```

### 2.3 Same change in masked path (lines ~255–257)

**Run**: `bash install.sh && python3 tests/test_fp8.py 2>&1 | tee logs/phase2.log`

---

## Phase 3 — Tile Size Optimization for Wave Efficiency

### 3.1 Expand block_n candidates in auto-tuner

**File**: `csrc/jit_kernels/heuristics/common.hpp` line ~447  
Change:
```cpp
auto block_ns = std::vector{64, 128};
```
to:
```cpp
auto block_ns = std::vector{64, 128, 192, 224};
```

### 3.2 Shape-adaptive manual config selection

**File**: `csrc/jit_kernels/impls/sm100_fp8_asym_gemm_1d1d.hpp`  
In both `sm100_m_grouped_fp8_asym_gemm_contiguous_1d1d()` and `sm100_m_grouped_fp8_asym_gemm_masked_1d1d()`, replace the hard-coded block sizes with:

```cpp
// Shape-adaptive tile selection for GB200 (132 SMs)
int block_m = 0, block_n = 0, block_k = 0;
if (n % 224 == 0 && k > 512) {
    // block_n=224: 7168/224=32 tiles => 2 waves for 8 groups = 97% SM util
    block_m = 128; block_n = 224; block_k = 256;
} else if (n % 192 == 0 && k > 512) {
    block_m = 128; block_n = 192; block_k = 256;
} else if (k <= 512) {
    // Small K: block_k=512 fits in 1 K-block, maximize block_k
    block_m = 128; block_n = 128; block_k = k;
} else {
    // Fallback to auto-tuner
    block_m = 0; block_n = 0; block_k = 0;
}
```

**Run**: `bash install.sh && python3 tests/test_fp8.py 2>&1 | tee logs/phase3.log`

---

## Phase 4 — B Double-Buffering (NVLink Latency Hiding)

### Goal
Overlap B load for K-block `k+1` with MMA for K-block `k`.

### SMEM layout change

**File**: `asym_gemm/include/asym_gemm/impls/sm100_fp8_asym_gemm_1d1d.cuh` line 112  
Change:
```cpp
constexpr uint32_t SMEM_B_SIZE = SMEM_B_SIZE_PER_STAGE;           // 1 stage
```
to:
```cpp
constexpr uint32_t kNumBStages = 2;
constexpr uint32_t SMEM_B_SIZE = kNumBStages * SMEM_B_SIZE_PER_STAGE;  // 2 stages
```
And simultaneously reduce A stages from 2 to 1 in the host heuristic (to keep SMEM within 232KB for block_k=512).

### Barrier additions
Add `full_barriers_b[kNumBStages]` and `empty_barriers_b[kNumBStages]`.  
Add `b_stage_idx` variable (cycling 0..kNumBStages-1) analogous to `stage_idx`.

### TMA load warp restructure (warp 0)
```
// Pre-load B[0] before K-loop
load_b(k=0, slot=0)

for k = 0..K-1:
    // Start loading B[k+1] into the next slot early
    if k+1 < K:
        wait empty_barriers_b[(k+1) % kNumBStages]
        load_b(k+1, slot=(k+1)%kNumBStages)
    
    // M-loop uses B[k] from slot k%kNumBStages
    for m in M:
        load A[m, k]
        ...
```

### Host-side SMEM accounting
**File**: `csrc/jit_kernels/heuristics/common.hpp` line ~123  
```cpp
const int smem_b_total = is_asym ? num_b_stages * smem_b_per_stage 
                                 : num_stages * smem_b_per_stage;
```
Pass `num_b_stages=2` from the launcher when using this optimization.

**Run**: `bash install.sh && python3 tests/test_fp8.py 2>&1 | tee logs/phase4.log`

---

## Phase 5 — NCU Profiling

### MMA utilization + bandwidth
```bash
ncu --target-processes all \
  --metrics smsp__sass_thread_inst_executed_op_hmma_pred_on.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,\
smsp__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_elapsed \
  python3 tests/test_fp8.py 2>&1 | tee logs/ncu_mma_bw.log
```

### Stall analysis
```bash
ncu --target-processes all \
  --metrics smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_wait_per_warp_active.pct,\
smsp__warp_issue_stalled_membar_per_warp_active.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct \
  python3 tests/test_fp8.py 2>&1 | tee logs/ncu_stalls.log
```

### Key diagnostics
- **stall_barrier > 20%**: too many sync points; check barrier count
- **stall_wait > 15%**: data latency; B NVLink not hidden → Phase 4 not working
- **stall_membar > 10%**: fence overhead; investigate tcgen05 fences
- **local memory spills > 0**: register pressure; examine ptx in JIT cache

### PTX inspection
```bash
find /tmp -name "*.ptx" 2>/dev/null | xargs grep -l "sm100_fp8_asym_gemm" | head -1 | xargs head -200
```

---

## Phase 6 — Shape-Specific Compiled Dims

**File**: caller of `sm100_m_grouped_fp8_asym_gemm_contiguous_1d1d`  
For the two most common N values, compile N as a template constant:
```cpp
std::string compiled_dims = "";
if (n == 7168 || n == 4096) compiled_dims += "n";
if (k == 512)               compiled_dims += "k";  // single K-block: fully unroll K-loop
```
This eliminates runtime `ceil_div(n, block_n)` and allows compiler to fully unroll the K-block loop for single-pass K=512 shapes.

---

## Expected TFLOPS Improvement

| Shape | Current config | Waves | Baseline | After Ph3 | After Ph4 |
|-------|---------------|-------|----------|-----------|-----------|
| 4g, N=4096, K=512   | (128,128,512) | 1 | ~350 | ~350 | ~380 |
| 4g, N=7168, K=2048  | (128,128,512) | 2 | ~650 | ~720 | ~800 |
| 8g, N=4096, K=7168  | (128,128,512) | 2 | ~750 | ~750 | ~860 |
| **8g, N=7168, K=2048**  | **(128,128,512)** | **4** | **~500** | **~780** | **~900** |
| 16g, N=7168, K=2048 | (128,128,512) | 4 | ~480 | ~750 | ~860 |
| 32g, N=7168, K=2048 | (128,128,512) | 7-8 | ~350 | ~620 | ~720 |

**Overall estimated improvement on critical shape: 1.7–1.9×**

---

## Implementation Order

1. `bash install.sh` → baseline
2. Fix printf + enable auto-tuner → rebuild → measure
3. Add block_n=224 shape-adaptive config → rebuild → measure  
4. B double-buffering → rebuild → measure
5. NCU profile → identify remaining bottlenecks → targeted fixes
6. Compiled dims for N=7168 → rebuild → measure

---

## Critical Files

| File | Change |
|------|--------|
| `csrc/jit_kernels/heuristics/common.hpp:520` | Remove debug printf |
| `csrc/jit_kernels/heuristics/common.hpp:447` | Add 192, 224 to block_n candidates |
| `csrc/jit_kernels/impls/sm100_fp8_asym_gemm_1d1d.hpp:104-106` | Shape-adaptive config (contiguous) |
| `csrc/jit_kernels/impls/sm100_fp8_asym_gemm_1d1d.hpp:255-257` | Shape-adaptive config (masked) |
| `asym_gemm/include/asym_gemm/impls/sm100_fp8_asym_gemm_1d1d.cuh:112` | B double-buffering |
| `tests/test_fp8.py:499-502` | Uncomment all test functions |
| `tests/test_fp8.py:438-450` | Enable masked GEMM perf benchmark |
| `tests/generators.py:148` | Add 16g/32g shapes |
