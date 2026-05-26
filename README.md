# AsymGEMM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![CUDA 12.9+](https://img.shields.io/badge/CUDA-12.9+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**AsymGEMM** is a high-performance GEMM library for NVIDIA GPUs that enables weight matrices
to reside in **CPU DRAM** while GPU kernels access them directly — without staging them through
HBM. It is designed to work across the full range of CPU–GPU interconnects: **NVLink-C2C**
(≥ 900 GB/s) on Grace-Hopper and Grace-Blackwell Superchips, and **PCIe** on standard server
and workstation platforms.

The library targets **Mixture-of-Experts (MoE)** inference workloads, where the aggregate
expert parameter count far exceeds GPU HBM capacity. By keeping weights in CPU DRAM and fetching
them on-demand, AsymGEMM enables single-node deployment of models such as DeepSeek V3 and
Qwen3-235B that would otherwise require multi-node tensor parallelism.

AsymGEMM builds on concepts from [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM),
[CUTLASS](https://github.com/nvidia/cutlass), and
[CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute).
All CUDA kernels are **JIT-compiled at first use** via NVRTC — no CUDA compilation is required
at install time.

---

## News

- **2026.05**: AsymGEMM now supports **NVFP4 (E2M1 + E4M3 scales)** asymmetric GEMM for SM100.
- **2026.05**: AsymGEMM now supports **FP8 (E4M3)** asymmetric GEMM for SM100 and SM89.
- **2026.02**: AsymGEMM released with **BF16** MoE GEMM support.

---

## Key Features

- **Asymmetric memory**: the weight matrix (`B`) lives in CPU-pinned DRAM; only the activation
  matrix (`A`) and the output (`D`) reside in HBM.
- **Three precision modes**: BF16, FP8 (E4M3), and FP4 (NVFP4 E2M1 + E4M3 scales).
- **Two token layouts**: *contiguous* (variable-length expert segments concatenated in a flat
  buffer) and *masked* (fixed-stride padded layout, one slice per expert).
- **K-outer M-inner scheduling**: each weight tile is fetched once from CPU DRAM and reused
  across all tokens assigned to that expert, amortizing the fetch cost.
- **asymScheduler**: a per-expert CTA ownership model that eliminates atomic contention and
  maximises utilization of the CPU–GPU interconnect, whether NVLink-C2C or PCIe.
- **JIT compilation**: kernels are compiled on first call and cached under `~/.asym_gemm`
  (overrideable via `DG_JIT_CACHE_DIR`); no ahead-of-time build step.
- **Flexible deployment**: runs on high-end Blackwell Superchips (GB200 NVL72) over NVLink-C2C
  and on consumer-grade Ada Lovelace GPUs (RTX 4090) over PCIe, covering a wide range of
  deployment environments.

---

## Supported Use Cases

### Single-GPU Deployment for Large-Scale MoE Models

AsymGEMM removes the HBM capacity bottleneck that forces large MoE models onto multi-node
clusters. By keeping all expert weight matrices in CPU DRAM and fetching them on demand,
a single GPU can serve models such as **DeepSeek V3** and **Qwen3-235B** that would
otherwise require expensive multi-node tensor-parallel setups.

### Prefill-Optimised Inference

AsymGEMM is primarily designed for the **prefill phase**, where the token batch is large
enough to amortize the cost of fetching each weight tile from CPU DRAM across many tokens.

> **Decode phase** optimization (small batch, memory-bandwidth bound) is ongoing — see the
> Roadmap.

### Flexible Precision

AsymGEMM supports **BF16**, **FP8 (E4M3)**, and **FP4 (NVFP4 E2M1)**, allowing users to
select the precision level that best suits their accuracy and throughput requirements without
changing the calling API.

### Broad GPU Compatibility

The same library runs on:
- **GB200 NVL72** (Blackwell Superchip) — weights fetched via NVLink-C2C at ≥ 900 GB/s.
- **RTX 4090** (Ada Lovelace, consumer PCIe) — weights fetched via PCIe, making asymmetric
  inference accessible without dedicated data-centre hardware.

---

## Roadmap

- [x] BF16 asymmetric GEMM (SM100, SM89)
- [x] FP8 asymmetric GEMM (SM100, SM89)
- [x] FP4 (NVFP4) asymmetric GEMM (SM100)
- [ ] SM90 (Hopper) support — H100, H200, H20, GH200 and other Hopper-class GPUs

---

## Hardware Support Matrix

| Architecture | BF16 | FP8 (E4M3) | FP4 (E2M1) | CPU→GPU path |
|---|---|---|---|---|
| SM100 — Blackwell (GB200) | ✅ | ✅ | ✅ | NVLink-C2C via TMA |
| SM89 — Ada Lovelace (RTX 4090) | ✅ | ✅ | — | PCIe (`cp.async`) |

> **Note:** FP4 requires SM100 with CUDA ≥ 12.9. FP8 on SM89 uses the
> `m_grouped_fp8_asym_gemm_sm89[_masked]` API, which accepts per-tensor scale scalars in
> addition to per-token/per-channel scale tensors.

---

## Quick Start

### Requirements

- NVIDIA GPU: SM89 (Ada Lovelace) or SM100 (Blackwell)
- CUDA Toolkit ≥ 12.9 for SM100 / FP4; CUDA ≥ 12.1 for SM89
- Python ≥ 3.8
- PyTorch ≥ 2.1
- C++17-capable host compiler (GCC ≥ 9, Clang ≥ 10)
- CUTLASS ≥ 4.0 (bundled as a Git submodule)

### Installation

We use [UV](https://docs.astral.sh/uv/) to manage the Python environment.

```bash
# Clone with submodules (CUTLASS, fmt)
git clone --recurse-submodules https://github.com/bytedance-iaas/AsymGEMM.git
cd AsymGEMM

# Create virtual environment and install
uv sync
source .venv/bin/activate

# Verify
python -c "import asym_gemm; print('AsymGEMM', asym_gemm.__version__)"
```

Kernels are compiled on first use and cached in `~/.asym_gemm/` (override with
`DG_JIT_CACHE_DIR`). The initial call for a new shape/dtype combination may take a few seconds.

### Running the Tests

```bash
cd tests/

# FP8 asymmetric GEMM — contiguous + masked, correctness + throughput
python test_fp8_asym_gemm.py

# BF16 asymmetric GEMM
python test_bf16_asym_gemm.py

# FP4 (NVFP4) asymmetric GEMM
python test_fp4_asym_gemm.py

# SM89 MoE path
python test_sm89_moe.py
```

---

## Design Overview

### Asymmetric Memory Path

In conventional MoE inference, all expert weight matrices are pre-loaded into HBM. On large
models (e.g., 671B parameters) the total expert weight size dwarfs available HBM, forcing
multi-node deployments. AsymGEMM breaks this constraint by keeping expert weights in CPU DRAM
and fetching them into shared memory on-demand during the GEMM kernel.

On SM100 (GB200), the GPU accesses CPU DRAM via **NVLink-C2C** using the Tensor Memory
Accelerator (TMA). On SM89, the `cp.async` instruction reads from PCIe-mapped pinned CPU
memory. In both cases the host CPU is uninvolved — the GPU autonomously drives all data
movement.

### K-outer, M-inner Loop

Naively, fetching each weight tile once per *token* would saturate the CPU–GPU interconnect.
AsymGEMM
inverts the standard GEMM loop order:

```
for k_tile in range(K // BLOCK_K):            # outer: weight K dimension
    fetch W[expert, N_tile, k_tile] from CPU   # one fetch per K-tile
    for m_tile in range(M // BLOCK_M):         # inner: token dimension
        MMA(A[m_tile, k_tile], W)              # reuse weight across all tokens
    write partial sums to HBM                  # read-modify-write between K-tiles
```

Each weight tile is fetched **once** and reused across all `ceil(M / BLOCK_M)` token tiles.
Partial sums are staged to HBM between K-tiles; because HBM bandwidth is roughly 50× higher
than PCIe throughput, this round-trip cost is negligible.

### asymScheduler

The CTA grid has shape `(ceil(N / BLOCK_N), num_active_experts)` — the M dimension is absent
from the grid. Each CTA owns a fixed **(N-tile, expert)** pair and iterates over the full M
extent for its expert by reading the token range from the `offsets/experts` pair list. This
design eliminates atomic contention between CTAs and ensures each CTA fetches weight tiles only
for its assigned expert, maximising reuse.

### Token Layout Protocols

**Contiguous layout** — All expert tokens are concatenated in a flat buffer of shape
`[M_total, K]`. The `offsets` tensor stores the start position of each expert's segment;
`experts` stores the corresponding expert ID; `list_size` is the count including the `-1`
sentinel.

**Masked layout** — Each expert occupies a fixed-stride slice at `g * max_m` of a buffer of
shape `[num_groups * max_m, K]`. The `masked_m[g]` tensor records the actual valid token count
for expert `g`. The `offsets/experts/list_size` triple is derived from `masked_m` via the
`build_offsets_experts_from_masked_m` helper in `tests/generators.py`.

---

## API Reference

All kernel functions are exported from the top-level `asym_gemm` package.

### Notation

- `a`, `b` are passed as `(data_tensor, scale_tensor)` tuples for FP8/FP4 kernels.
- `offsets` — 1-D `int32` tensor of segment boundaries.
- `experts` — 1-D `int32` tensor of expert IDs, terminated by a `-1` sentinel.
- `list_size` — `int32` scalar (or tensor): total entry count in `experts` including sentinel.
- `recipe` — optional `(gran_mn_a, gran_k_a, gran_k_b)` quantization granularity tuple.
- `disable_ue8m0_cast` — if `True`, scale factors stay in `float32` rather than UE8M0.

---

### FP8 Asymmetric GEMM

```python
asym_gemm.m_grouped_fp8_asym_gemm_nt_contiguous(
    a:                  Tuple[Tensor, Tensor],   # (fp8_data [M, K], scales)
    b:                  Tuple[Tensor, Tensor],   # (fp8_data [G, N, K], scales) — CPU-pinned
    d:                  Tensor,                  # output [M, N], bfloat16
    offsets:            Tensor,                  # int32, segment start positions
    experts:            Tensor,                  # int32, expert IDs + sentinel
    list_size:          Tensor,                  # int32 scalar
    recipe:             Optional[Tuple[int, int, int]] = None,
    disable_ue8m0_cast: bool = False,
) -> None
```

```python
asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(
    a:                    Tuple[Tensor, Tensor],   # (fp8_data [G, M_max, K], scales)
    b:                    Tuple[Tensor, Tensor],   # (fp8_data [G, N, K], scales) — CPU-pinned
    d:                    Tensor,                  # output [G, M_max, N], bfloat16
    offsets:              Tensor,                  # int32, (start, end) pairs per expert
    experts:              Tensor,                  # int32, expert IDs + sentinel
    list_size:            Tensor,                  # int32 scalar
    expected_m_per_group: int,                     # hint for CTA tiling
    disable_ue8m0_cast:   bool = False,
) -> None
```

---

### FP4 Asymmetric GEMM (SM100 only)

```python
asym_gemm.m_grouped_fp4_asym_gemm_nt_contiguous(
    a:                  Tuple[Tensor, Tensor],   # (fp4_packed uint8 [M, K//2], fp8 scales)
    b:                  Tuple[Tensor, Tensor],   # (fp4_packed uint8 [G, N, K//2], fp8 scales) — CPU-pinned
    d:                  Tensor,                  # output [M, N], bfloat16
    offsets:            Tensor,
    experts:            Tensor,
    list_size:          int,
    recipe:             Optional[Tuple[int, int, int]] = None,
    disable_ue8m0_cast: bool = False,
) -> None
```

```python
asym_gemm.m_grouped_fp4_asym_gemm_nt_masked(
    a:                    Tuple[Tensor, Tensor],
    b:                    Tuple[Tensor, Tensor],   # CPU-pinned
    d:                    Tensor,
    masked_m:             Tensor,                  # int32 [G], valid token count per expert
    expected_m_per_group: int,
    recipe:               Optional[Tuple[int, int, int]] = None,
    disable_ue8m0_cast:   bool = False,
) -> None
```

> FP4 data is packed two elements per byte (E2M1 encoding); scale factors use FP8 E4M3.
> Use `asym_gemm.utils.per_token_cast_to_fp4` to quantize activation tensors.

---

### BF16 Asymmetric GEMM

```python
asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
    a:             Tensor,   # bfloat16 [M, K], on GPU
    b:             Tensor,   # bfloat16 [G, N, K], CPU-pinned
    d:             Tensor,   # bfloat16 [M, N], on GPU
    offsets:       Tensor,
    experts:       Tensor,
    list_size:     Tensor,
    compiled_dims: str = "nk",
) -> None
```

```python
asym_gemm.m_grouped_bf16_asym_gemm_nt_masked(
    a:                    Tensor,   # bfloat16 [G, M_max, K]
    b:                    Tensor,   # bfloat16 [G, N, K], CPU-pinned
    d:                    Tensor,   # bfloat16 [G, M_max, N]
    offsets:              Tensor,
    experts:              Tensor,
    list_size:            Tensor,
    expected_m_per_group: int,
    compiled_dims:        str = "nk",
) -> None
```

---

### SM89 FP8 MoE GEMM

For Ada Lovelace (SM89) systems (e.g. RTX 4090) without NVLink-C2C:

```python
asym_gemm.m_grouped_fp8_asym_gemm_sm89(
    a:               Tensor,                   # fp8 [M, K], on GPU
    b:               Tensor,                   # fp8 [G, N, K], CPU-pinned
    d:               Tensor,                   # bfloat16 [M, N], on GPU
    offsets:         Tensor,
    experts:         Tensor,
    list_size:       int,
    scale_a:         float = 1.0,
    scale_b:         float = 1.0,
    scale_a_tensor:  Optional[Tensor] = None,  # per-token scales, overrides scale_a
    scale_b_tensor:  Optional[Tensor] = None,
) -> None

asym_gemm.m_grouped_fp8_asym_gemm_sm89_masked(
    a:               Tensor,
    b:               Tensor,                   # CPU-pinned
    d:               Tensor,
    masked_m:        Tensor,
    expected_m:      int,
    scale_a:         float = 1.0,
    scale_b:         float = 1.0,
    scale_a_tensor:  Optional[Tensor] = None,
    scale_b_tensor:  Optional[Tensor] = None,
) -> None
```

---

## Performance

Benchmarks run on **GB200 NVL72** (Grace-Blackwell, NVLink-C2C ≥ 900 GB/s) with MoE shapes
representative of production LLM inference.

**Methodology:** kernel timing via Torch Kineto (`bench_kineto`), 30 iterations, L2 cache flush
between runs. Throughput = `2 × M × N × K / elapsed_time`. *Baseline* is DeepGEMM with weights
in HBM (compute-roofline upper bound); *AsymGEMM* fetches weights from CPU DRAM via NVLink-C2C.

### FP8 MoE — SM100 (GB200)

| num\_experts | tokens/expert | N | K | AsymGEMM (TFLOPS) | HBM Baseline (TFLOPS) |
|---|---|---|---|---|---|
| 8 | 128 | 7168 | 7168 | — | — |
| 8 | 1024 | 7168 | 7168 | — | — |
| 64 | 128 | 7168 | 7168 | — | — |

> Numbers will be filled in from `python tests/test_fp8_asym_gemm.py` on GB200 hardware.

---

## Acknowledgements

AsymGEMM builds on the following open-source projects:

- **[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)** (DeepSeek, MIT) — JIT GEMM
  framework, quantization utilities, and benchmarking infrastructure. Several source files in
  AsymGEMM are adapted from DeepGEMM and retain the original copyright notice.
- **[CUTLASS](https://github.com/nvidia/cutlass)** (NVIDIA, BSD-3) — CuTe tensor abstractions,
  TMA descriptors, and WGMMA instruction wrappers used in SM100 kernel implementations.
- **[{fmt}](https://github.com/fmtlib/fmt)** (MIT) — String formatting in the C++ JIT
  infrastructure.

---

## License

AsymGEMM is released under the [MIT License](LICENSE).  
Copyright © 2026 Bytedance Inc.

Files adapted from DeepGEMM retain their original copyright notice:
> Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
