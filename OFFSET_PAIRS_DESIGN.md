# Offset Pairs Design for Sparse M-Grouped GEMM

## Overview

This document describes the updated design for handling expert-token mappings in m-grouped GEMM operations, supporting sparse and contiguous layouts with offset pairs.

## Key Changes

### 1. Offset Pair Structure

Instead of cumulative single offsets, we now use **offset pairs** for each expert:

```
offsets array layout: [start_0, end_0, start_1, end_1, ..., start_N, end_N]
experts array layout: [expert_id_0, expert_id_1, ..., expert_id_N, -1]
```

Each pair `(start_i, end_i)` defines the M-dimension range for expert `i`.

### 2. Sparse Masked Layout (Fixed Per-Group Allocation)

#### Purpose
Support cases where only a subset of expert groups have tokens, with each group allocated fixed `max_m` space.

#### Example
```python
num_groups = 4
max_m = 4096
masked_m = torch.tensor([0, 12, 0, 129])  # 0 tokens, 12 tokens, 0 tokens, 129 tokens

# Expected output:
offsets = [4096, 4224, 12288, 12544]  # 2 pairs
experts = [1, 3, -1]                   # 2 active experts + terminator
```

**Mapping:**
- Group 0: 0 tokens → skipped
- Group 1: 12 tokens → offset pair (4096, 4224) = [1*4096, 1*4096 + ceil(12/128)*128]
- Group 2: 0 tokens → skipped
- Group 3: 129 tokens → offset pair (12288, 12544) = [3*4096, 3*4096 + ceil(129/128)*128]

#### Implementation
Function: `build_offsets_experts_from_masked_m(masked_m, num_groups, max_m, block_m=128)`

Located in:
- `demo/AsymGEMM/tests/test_fp8.py`
- `demo/AsymGEMM/tests/test_bf16.py`

### 3. Contiguous Layout with Offset Pairs

#### Purpose
Support contiguous token-to-expert layouts where tokens are grouped by expert in sequence.

#### Example
```python
m_indices = [0, 0, 2, 2, 2, 1, -1, -1]
block_m = 128

# Expected output:
offsets = [0, 128, 128, 256, 256, 384]  # 3 pairs (padded to 128)
experts = [0, 2, 1, -1]                  # 3 experts + terminator
```

**Mapping:**
- Tokens [0:2] with expert 0 → pair (0, 128)
- Tokens [2:5] with expert 2 → pair (128, 256)
- Tokens [5:6] with expert 1 → pair (256, 384)
- Tokens [6:8] with expert -1 → skipped (invalid)

#### Implementation
Function: `build_offsets_experts_from_m_indices_pairs(m_indices, block_m=128)`

Located in:
- `demo/AsymGEMM/tests/test_fp8.py`
- `demo/AsymGEMM/tests/test_bf16.py`

## Kernel Implementation

### asymScheduler.cuh Changes

The kernel scheduler was updated to correctly interpret offset pairs:

**Constructor Logic:**
```cpp
__device__ __forceinline__ explicit asymScheduler(
    const uint32_t& shape_m,
    const uint32_t& shape_n,
    uint32_t* experts,
    uint32_t* offsets) {

    expert_id = experts[blockIdx.y];

    // Key change: multiply blockIdx.y by 2 to index into pairs
    uint32_t offset_pair_idx = blockIdx.y * 2;
    m_start = ceil_div_device(offsets[offset_pair_idx], BLOCK_M);
    m_end = ceil_div_device(offsets[offset_pair_idx + 1], BLOCK_M);
}
```

**Member Variables:**
- `m_start`: Block index for the start of this expert's M range (computed as `ceil(offsets[pair_idx] / BLOCK_M)`)
- `m_end`: Block index for the end of this expert's M range (computed as `ceil(offsets[pair_idx+1] / BLOCK_M)`)

### Block Grid Layout

For `num_experts` active experts and `num_n_blocks` N-dimension blocks:

```
Grid dimensions:
- blockIdx.x: ranges over N blocks (one per BLOCK_N)
- blockIdx.y: ranges over active experts (0 to num_experts-1)

Each block (x, y) processes:
- Expert y with M range [m_start, m_end) and N range [blockIdx.x * BLOCK_N, ...)
```

## Usage Examples

### Sparse Masked GEMM (FP8)

```python
import torch
from test_fp8 import build_offsets_experts_from_masked_m

num_groups = 4
max_m = 4096
masked_m = torch.tensor([0, 12, 0, 129], device='cuda', dtype=torch.int)

offsets, experts, list_size = build_offsets_experts_from_masked_m(
    masked_m, num_groups, max_m, block_m=128
)

# offsets: [4096, 4224, 12288, 12544]
# experts: [1, 3, -1]
# list_size: 3

# Call kernel with these offsets/experts
asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(
    a, b, d_asym, offsets, experts, list_size,
    expected_m_per_group, disable_ue8m0_cast=False
)
```

### Contiguous Layout GEMM (BF16)

```python
import torch
from test_bf16 import build_offsets_experts_from_m_indices_pairs

m_indices = torch.tensor([0, 0, 2, 2, 2, 1, -1, -1], device='cuda', dtype=torch.int32)

offsets, experts, list_size = build_offsets_experts_from_m_indices_pairs(
    m_indices, block_m=128
)

# offsets: [0, 128, 128, 256, 256, 384]
# experts: [0, 2, 1, -1]
# list_size: 4

# Call kernel with these offsets/experts
asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
    a, b_pinned, d_asym, offsets, experts, list_size, compiled_dims="mnk"
)
```

## Benefits

1. **Sparse Support**: Efficiently handles sparse expert usage with fixed per-group allocation
2. **Memory Efficiency**: No need to store padding for inactive experts
3. **Block-Aligned**: All offset pairs are aligned to block boundaries for efficient kernel execution
4. **Flexible Scheduling**: Kernel can easily map blockIdx.y to expert offset pairs

## Related Files

- **Scheduler**: `asym_gemm/include/asym_gemm/common/asymScheduler.cuh`
- **Test FP8**: `demo/AsymGEMM/tests/test_fp8.py`
- **Test BF16**: `demo/AsymGEMM/tests/test_bf16.py`
- **Generators**: `demo/AsymGEMM/tests/generators.py`

## Migration Guide

If you have existing code using the old cumulative offset format:

**Old format (cumulative):**
```
offsets = [0, 256, 512]  # Cumulative positions
experts = [0, 2, -1]
```

**New format (pairs):**
```
offsets = [0, 256, 256, 512]  # [start_0, end_0, start_1, end_1]
experts = [0, 2, -1]
```

The kernel will multiply `blockIdx.y` by 2 to access the correct offset pair, so update your kernel calls accordingly.
