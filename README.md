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

- **2026.07**: AsymGEMM **v0.2.0** is internally released! See the [release notes](docs/release_notes/v0.2.0.md).
- **2026.07**: AsymGEMM now supports **SM90 (Hopper)** — H100, H200, H20, GH200 — with **BF16**, **FP8 (E4M3)**, and **INT8** asymmetric GEMM kernels.
- **2026.07**: New **unified MoE runtime** (`asym_gemm.unified_moe`): expert GEMMs run **concurrently on CPU (AMX INT8) and GPU**, with an optional cost-model-based adaptive dispatcher that partitions experts to minimize makespan. See [`adaptive_dispatch.md`](adaptive_dispatch.md).
- **2026.07**: **INT8 asymmetric GEMM** (SM90) with per-token/per-channel block scales, plus an offline INT8 weight converter (`scripts/convert_int8_weights.py`) for Qwen3-family MoE checkpoints.
- **2026.07**: SM89 FP8 kernels now accept **native block scales** (1×128 activation, 128×128 weight) in addition to per-tensor/per-token scales.
- **2026.05**: AsymGEMM **v0.1.0** is internally released! See the [release notes](docs/release_notes/v0.1.0.md).
- **2026.05**: AsymGEMM now supports **NVFP4 (E2M1 + E4M3 scales)** asymmetric GEMM for SM100.
- **2026.05**: AsymGEMM now supports **FP8 (E4M3)** asymmetric GEMM for SM100 and SM89.
- **2026.02**: AsymGEMM released with **BF16** MoE GEMM support.

---

## Key Features

- **Asymmetric memory**: the weight matrix (`B`) lives in CPU-pinned DRAM; only the activation
  matrix (`A`) and the output (`D`) reside in HBM.
- **Four precision modes**: BF16, FP8 (E4M3), INT8 (block-scaled), and FP4 (NVFP4 E2M1 + E4M3 scales).
- **Two token layouts**: *contiguous* (variable-length expert segments concatenated in a flat
  buffer) and *masked* (fixed-stride padded layout, one slice per expert).
- **K-outer M-inner scheduling**: each weight tile is fetched once from CPU DRAM and reused
  across all tokens assigned to that expert, amortizing the fetch cost.
- **asymScheduler**: a per-expert CTA ownership model that eliminates atomic contention and
  maximizes utilization of the CPU-GPU interconnect, whether NVLink-C2C or PCIe.
- **Unified CPU + GPU MoE execution**: the `asym_gemm.unified_moe` package runs each MoE
  forward as two concurrent buckets — small experts on the CPU via an AMX/AVX-512 INT8
  GEMM library (`csrc/cpu/cpu_gemm`), large experts on the GPU — joined at the end. An
  adaptive dispatcher fits per-backend cost models online and picks the expert partition
  that minimizes the slower bucket.
- **JIT compilation**: kernels are compiled on first call and cached under `~/.asym_gemm`
  (overridable via `DG_JIT_CACHE_DIR`); no ahead-of-time build step.
- **Flexible deployment**: runs on Blackwell Superchips (GB200 NVL72) over NVLink-C2C,
  Hopper data-center GPUs (H100/H200/H20/GH200), and consumer-grade Ada Lovelace GPUs
  (RTX 4090) over PCIe, covering a wide range of deployment environments.

---

## Supported Use Cases

### Single-GPU Deployment for Large-Scale MoE Models

AsymGEMM removes the HBM capacity bottleneck that forces large MoE models onto multi-node
clusters. By keeping all expert weight matrices in CPU DRAM and fetching them on demand,
a single GPU can serve models such as **DeepSeek V3** and **Qwen3-235B** that would
otherwise require expensive multi-node tensor-parallel setups.

### Prefill-Optimized Inference

AsymGEMM is primarily designed for the **prefill phase**, where the token batch is large
enough to amortize the cost of fetching each weight tile from CPU DRAM across many tokens.

> **Decode phase** optimization (small batch, memory-bandwidth bound) is ongoing — see the
> Roadmap.

### CPU-GPU Unified Kernel in MoE Execution

The **unified MoE runtime** (`asym_gemm.unified_moe.Layer`) treats the host CPU as a second
compute backend rather than just a weight store: experts with few routed tokens run on the
CPU (AMX INT8, worker pool) while the rest run on the GPU, overlapping in time. In adaptive
mode, a lightweight linear cost model per backend is fitted online from observed timings,
and each forward solves for the expert partition with the minimum predicted makespan.

### Flexible Precision

AsymGEMM supports **BF16**, **FP8 (E4M3)**, **INT8**, and **FP4 (NVFP4 E2M1)**, allowing
users to select the precision level that best suits their accuracy and throughput
requirements without changing the calling API.

### Broad GPU Compatibility

The same library runs on:
- **GB200 NVL72** (Blackwell Superchip) — weights fetched via NVLink-C2C at ≥ 900 GB/s.
- **H100 / H200 / H20 / GH200** (Hopper) — BF16, FP8, and INT8 asymmetric kernels using
  TMA + WGMMA over pinned host memory.
- **RTX 4090** (Ada Lovelace, consumer PCIe) — weights fetched via PCIe, making asymmetric
  inference accessible without dedicated data-center hardware.

---

## Roadmap

- [x] BF16 asymmetric GEMM (SM100, SM90, SM89)
- [x] FP8 asymmetric GEMM (SM100, SM90, SM89 — incl. SM89 block scales)
- [x] FP4 (NVFP4) asymmetric GEMM (SM100)
- [x] SM90 (Hopper) support — H100, H200, H20, GH200 and other Hopper-class GPUs
- [x] INT8 asymmetric GEMM (SM90) + offline INT8 weight conversion
- [x] Unified CPU-GPU kernel with adaptive dispatch

---

## Hardware Support Matrix

| Architecture | BF16 | FP8 (E4M3) | INT8 | FP4 (E2M1) | CPU→GPU path |
|---|---|---|---|---|---|
| SM100 — Blackwell (GB200) | ✅ | ✅ | — | ✅ | NVLink-C2C via TMA |
| SM90 — Hopper (H100/H200/H20/GH200) | ✅ | ✅ | ✅ | — | PCIe / NVLink-C2C via TMA |
| SM89 — Ada Lovelace (RTX 4090) | ✅ | ✅ | — | — | PCIe |
| Host CPU (AMX / AVX-512-VNNI / AVX2) | ✅ | — | ✅ | — | in-DRAM (unified MoE) |

> **Note:** FP4 requires SM100 with CUDA ≥ 12.9. FP8 on SM89 is reached through the
> architecture-agnostic `m_grouped_fp8_asym_gemm_nt_*` APIs and accepts per-token/per-channel
> scale tensors or native block scales (1×128 / 128×128).
> The CPU row refers to the `cpu_gemm` library used by the unified MoE runtime; the
> backend (AMX → AVX-512 → AVX2) is selected at runtime via CPUID probing.

---

## Quick Start

### Requirements

- NVIDIA GPU: SM89 (Ada Lovelace), SM90 (Hopper), or SM100 (Blackwell)
- CUDA Toolkit ≥ 12.9 for SM100 / FP4; CUDA ≥ 12.1 for SM89 / SM90
- Python ≥ 3.8
- PyTorch ≥ 2.1
- C++17-capable host compiler (GCC ≥ 9, Clang ≥ 10)
- CUTLASS ≥ 4.0 (bundled as a Git submodule)

### Installation

```bash
# Clone with submodules (CUTLASS, fmt)
git clone --recurse-submodules https://github.com/bytedance-iaas/AsymGEMM.git
cd AsymGEMM

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install AsymGEMM
bash scripts/install.sh
```

`scripts/install.sh` installs dependencies from `requirements.txt`, cleans previous local build
artifacts, installs AsymGEMM in editable mode, and verifies that `asym_gemm` can be imported.

Kernels are compiled on first use and cached in `~/.asym_gemm/` (override with
`DG_JIT_CACHE_DIR`). The initial call for a new shape/dtype combination may take a few seconds.

### Running the Tests

The tests require a CUDA-capable machine. One command runs the subset of
tests applicable to the current GPU:

```bash
bash scripts/test.sh
```

The script detects compute capability via `torch.cuda.get_device_capability`
and selects:

| Compute cap | Test files                                                                                       |
| ----------- | ------------------------------------------------------------------------------------------------ |
| sm_89       | `tests/test_sm89_moe.py`                                                                         |
| sm_90       | `tests/test_bf16_asym_gemm.py`, `tests/test_fp8_asym_gemm.py`, `tests/test_sm90_int8.py`         |
| sm_100      | `tests/test_bf16_asym_gemm.py`, `tests/test_fp8_asym_gemm.py`, `tests/test_fp4_asym_gemm.py`     |

`tests/test_unified_moe.py` (CPU/GPU parity for the unified MoE layer) is always
appended; it skips internally when the host lacks AMX. The standalone CPU GEMM
library has its own CTest suite, runnable via `bash scripts/test_cpu_gemm.sh`.

It prints a pass/fail summary at the end and exits non-zero if any test
fails. Each test file prints its per-format accuracy diffs as it runs.

### Running with SGLang

AsymGEMM integrates with a downstreaming fork of [SGLang](https://github.com/bytedance-iaas/sglang/tree/asym_gemm_integration) as a MoE runner backend.

Please refer to [Quick Start](docs/quick_start.md) for detailed instructions.

---

## Design Overview

### Asymmetric Memory Path

In conventional MoE inference, all expert weight matrices are pre-loaded into HBM. On large
models (e.g., 671B parameters) the total expert weight size dwarfs available HBM, forcing
multi-node deployments. AsymGEMM breaks this constraint by keeping expert weights in CPU DRAM
and fetching them into shared memory on-demand during the GEMM kernel.

On SM100 (GB200), the GPU accesses CPU DRAM via **NVLink-C2C** using the Tensor Memory
Accelerator (TMA). On SM90 (Hopper), the same TMA path reads pinned host memory over PCIe
(or NVLink-C2C on GH200). On SM89, the `cp.async` instruction reads from PCIe-mapped pinned
CPU memory. In all cases the host CPU is uninvolved — the GPU autonomously drives all data
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
for its assigned expert, maximizing reuse.

### Token Layout Protocols

**Contiguous layout** — All expert tokens are concatenated in a flat buffer of shape
`[M_total, K]`. The `offsets` tensor stores the start position of each expert's segment;
`experts` stores the corresponding expert ID; `list_size` is the count including the `-1`
sentinel.

**Masked layout** — Each expert occupies a fixed-stride slice at `g * max_m` of a buffer of
shape `[num_groups, max_m, K]`. The `masked_m[g]` tensor records the actual valid token count
for expert `g`; `masked_m[g] == 0` marks an inactive expert and the kernel skips that CTA.
The masked APIs (`m_grouped_{bf16,fp8,fp4}_asym_gemm_nt_masked`) consume `masked_m` directly —
no offsets/experts/list_size triple is needed for this layout.

---

## API Reference

All kernel functions are exported from the top-level `asym_gemm` package.

### Notation

- `a`, `b` are passed as `(data_tensor, scale_tensor)` tuples for FP8/FP4 kernels.
- `offsets` — 1-D `int32` tensor of flat `(start, end)` row pairs, one pair per expert.
- `experts` — 1-D `int32` tensor of expert IDs, terminated by a `-1` sentinel.
- `list_size` — `int`: total entry count in `experts` including the sentinel (a 1-element
  `int32` tensor is also accepted).
- `recipe` — optional `(gran_mn_a, gran_mn_b, gran_k)` quantization granularity tuple.
- `disable_ue8m0_cast` — if `True`, scale factors stay in `float32` rather than UE8M0.

---

### FP8 Asymmetric GEMM

```python
asym_gemm.m_grouped_fp8_asym_gemm_nt_contiguous(
    a:                  Tuple[Tensor, Tensor],   # (fp8_data [M, K], scales)
    b:                  Tuple[Tensor, Tensor],   # (fp8_data [G, N, K], scales) — CPU-pinned
    d:                  Tensor,                  # output [M, N], bfloat16
    offsets:            Tensor,                  # int32, flat (start, end) pairs per expert
    experts:            Tensor,                  # int32, expert IDs + sentinel
    list_size:          int,                     # entries in `experts` incl. sentinel
    recipe:             Optional[Tuple[int, int, int]] = None,
    disable_ue8m0_cast: bool = False,
) -> None
```

```python
asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(
    a:                  Tuple[Tensor, Tensor],   # (fp8_data [G, M_max, K], scales)
    b:                  Tuple[Tensor, Tensor],   # (fp8_data [G, N, K], scales) — CPU-pinned
    d:                  Tensor,                  # output [G, M_max, N], bfloat16
    masked_m:           Tensor,                  # int32 [G], valid token count per expert
    expected_m:         int,                     # hint for CTA tiling
    disable_ue8m0_cast: bool = False,
) -> None
```

Dispatches to SM89 (native block/per-token scales), SM90, or SM100 based on the
device's compute capability.

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
    list_size:     int,
    compiled_dims: str = "nk",
) -> None
```

```python
asym_gemm.m_grouped_bf16_asym_gemm_nt_masked(
    a:             Tensor,   # bfloat16 [G, M_max, K]
    b:             Tensor,   # bfloat16 [G, N, K], CPU-pinned
    d:             Tensor,   # bfloat16 [G, M_max, N]
    masked_m:      Tensor,   # int32 [G], valid token count per expert
    expected_m:    int,
    compiled_dims: str = "nk",
) -> None
```

Contiguous runs on SM90/SM100; masked additionally supports SM89.

---

### INT8 Asymmetric GEMM (SM90 only)

Activations and weights are `(int8_data, fp32_scale)` pairs with a fixed `(1, 1, 128)`
block recipe: per-token scales for `a` (`[M, K/128]`), per-channel scales for `b`
(`[G, N, K/128]`). The facade transposes scales into the K-major layout the kernel expects.

```python
asym_gemm.m_grouped_int8_asym_gemm_nt_contiguous(
    a:             Tuple[Tensor, Tensor],   # (int8 [M, K], fp32 scales [M, K//128])
    b:             Tuple[Tensor, Tensor],   # (int8 [G, N, K], fp32 scales [G, N, K//128]) — CPU-pinned
    d:             Tensor,                  # output [M, N], bfloat16
    offsets:       Tensor,
    experts:       Tensor,
    list_size:     int,
) -> None

asym_gemm.m_grouped_int8_asym_gemm_nt_masked(
    a:             Tuple[Tensor, Tensor],   # (int8 [G, M_max, K], fp32 scales [G, M_max, K//128])
    b:             Tuple[Tensor, Tensor],   # CPU-pinned
    d:             Tensor,                  # output [G, M_max, N], bfloat16
    masked_m:      Tensor,                  # int32 [G]
    expected_m:    int,
) -> None
```

> Use `scripts/convert_int8_weights.py` to quantize a BF16 MoE checkpoint (e.g. Qwen3
> family) to this format offline; the output is byte-identical to online quantization
> via `unified_moe.Layer.from_bf16`.

---

### Unified MoE Layer (CPU + GPU)

The `asym_gemm.unified_moe` package provides an INT8 MoE layer that executes experts
concurrently on the CPU (AMX, via the bundled `cpu_gemm` library) and the GPU:

```python
from asym_gemm.unified_moe import Layer, DispatchModel

layer = Layer.from_bf16(gate, up, down, top_k=k, adaptive=True)  # quantize + build both backends
out = layer.forward(x_bf16, expert_ids, route_w)                 # CPU & GPU buckets overlap

layer.set_m_cpu(16)         # static mode: experts with m_e <= 16 go to the CPU
layer.calibrate()           # optional forced all-CPU/all-GPU sweeps to seed the model
layer.dispatch.snapshot()   # fitted cost-model coefficients + observation counts

# Share one cost model across all same-shape layers of a model
shared = DispatchModel(hidden=H, inter=I)
layers = [Layer.from_bf16(..., dispatch_model=shared) for _ in range(n_layers)]
```

Defaults preserve static dispatch (`adaptive=False`, threshold 16). See
[`adaptive_dispatch.md`](adaptive_dispatch.md) for the cost model, partition solver,
and measured behavior.

---

### SM89 FP8 MoE GEMM

Ada Lovelace (SM89) systems (e.g. RTX 4090) are served through the architecture-agnostic
`m_grouped_fp8_asym_gemm_nt_{contiguous,masked}` entry points above: on SM89 the kernel
consumes the original (untransformed) scales natively — a 3-D weight scale
(`[G, ceil(N/128), ceil(K/128)]`, with `[M, ceil(K/128)]` activation scales) selects the
128×128/1×128 block-scale path, lower-rank scales the per-token/per-channel path. The
former `m_grouped_fp8_asym_gemm_sm89[_masked]` functions are internal-only.

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
- **[ktransformers](https://github.com/kvcache-ai/ktransformers/tree/main)** (kvcache-ai,
  Apache-2.0) — The bundled `cpu_gemm` library (`csrc/cpu/cpu_gemm`) used by the unified
  MoE runtime is extracted and adapted from kt-kernel's CPU GEMM kernels (AMX/AVX).

---

## License

AsymGEMM is released under the [MIT License](LICENSE).
Copyright © 2026 Bytedance Inc.

Files adapted from DeepGEMM retain their original copyright notice:
> Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
