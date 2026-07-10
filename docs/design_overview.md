# Design Overview

## Purpose and Motivation

AsymGEMM is a high-performance CUDA kernel library that turns CPU DRAM into a practical extension of GPU compute for MoE inference. The core idea: expert weights that are not actively computing can reside in cheap, abundant CPU memory instead of scarce GPU HBM — and the GPU fetches them at near-peak interconnect bandwidth only when needed.

### The Problem

Large MoE models (hundreds of billions of parameters) require many GPUs simply to *hold* all expert weights in HBM — even though only a handful of experts fire per token. The GPUs aren't compute-bound; they're memory-bound by idle parameters. This is an expensive waste: you might be paying for 4 GPUs' worth of HBM to serve a workload that only needs 1 GPU's worth of compute.

### Our Approach

AsymGEMM eliminates this waste by keeping expert weights in CPU DRAM and fetching only active weight tiles into GPU shared memory on demand. The kernel issues direct loads from CPU memory (PCIe `cp.async` on SM89, TMA over PCIe or NVLink-C2C on SM90 and SM100) — no host-side orchestration, no staging copies, no CPU involvement at runtime. The GPU drives the entire data movement autonomously at peak hardware bandwidth.

To sustain high compute utilization despite the lower CPU→GPU bandwidth, we designed a **K-outer, M-inner kernel loop** that loads each weight tile exactly once and reuses it across all tokens assigned to that expert. The longer the input sequence, the more compute we extract per byte fetched from CPU — making the GPU nearly as efficient as if the weights were local.

### What This Enables

The result is a fundamentally better cost-performance point for MoE serving:

- **Significantly fewer GPUs** — MoE expert weights are offloaded to CPU DRAM, freeing GPU HBM for activations and KV cache. Serving a large MoE model requires substantially fewer GPUs.
- **Higher per-GPU throughput** — Each GPU spends its compute on active experts rather than wasting HBM bandwidth on idle parameters.
- **CPU memory becomes a first-class resource** — Servers with large DRAM (which is 10× cheaper per GB than HBM) become viable inference platforms for models previously out of reach.

We have demonstrated significant per-GPU throughput advantages on prefill-heavy workloads. Optimization for decode-heavy scenarios is ongoing — stay tuned.

> **Relationship to DeepGEMM:** AsymGEMM draws significant inspiration from [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)'s JIT compilation framework and CuTe-based kernel design. However, AsymGEMM is an independent project with its own development direction — the asymmetric memory scenario requires fundamentally different algorithmic choices (loop order, scheduling, pipeline depth) that go beyond the scope of a conventional GEMM library.

---

## Design Decisions

### 1. K-outer, M-inner Loop — Amortize Weight Fetch Cost

The fundamental challenge of asymmetric memory is weight fetch latency. CPU→GPU bandwidth (PCIe ~64 GB/s, NVLink-C2C ~900 GB/s) is far below HBM bandwidth (~3 TB/s). A standard GEMM loop that reloads the same weight tile for every M-tile would bottleneck entirely on interconnect bandwidth.

AsymGEMM solves this with a K-outer, M-inner loop: load each weight K-tile **once** from CPU memory into shared memory, then sweep across all token M-tiles before moving to the next K-tile.

```
for each K-tile:                    ← weight loaded once from CPU
    load W[expert, N_tile, k] → shared memory
    for each M-tile:                ← reuse the cached weight tile
        load X[m, k] from HBM → shared memory
        MMA: accumulate partial result
        write partial sum to HBM
```

The more tokens assigned to an expert (larger M), the more the weight fetch cost is amortized — which is exactly why AsymGEMM excels on prefill-heavy workloads with long input sequences.

**Trade-off:** Partial sums must be written to HBM (as BF16) after each K-tile and read back to seed the accumulator on the next iteration. This adds HBM traffic proportional to `num_k_tiles × M × N × 2 bytes`. But HBM bandwidth is 50×+ higher than PCIe — the extra read/write traffic is negligible compared to the weight fetch savings.

### 2. `asymScheduler` — Per-Expert CTA Ownership

Standard grouped GEMM kernels assign one (M-tile, N-tile) output block to each CTA. This doesn't work for K-outer loops because it would require cross-CTA synchronization on the shared weight tile.

AsymGEMM's `asymScheduler` gives each CTA ownership of an entire **(N-tile, expert)** pair:

- Grid shape: `(ceil(N/BLOCK_N), num_active_experts)` — no M dimension in the grid.
- Each CTA reads its expert's token range from the offset array to determine its M-extent.
- Internally, the CTA sweeps all M-tiles within that range for every K-tile.

This means fewer, longer-running CTAs that maximize weight reuse. A single CTA processes the full M dimension for its expert — load W once, use it `ceil(M/BLOCK_M)` times.

### 3. Maximize BLOCK_K, Fix Pipeline Depth

In a standard GEMM, deeper pipelines (more stages) overlap TMA loads with computation. But for asymmetric memory, the weight tile comes from CPU with much higher and less predictable latency — deep pipelining yields diminishing returns.

AsymGEMM fixes pipeline depth to **2 stages** for asymmetric kernels and uses the freed shared memory to maximize **BLOCK_K**:

- Larger BLOCK_K → fewer K-tiles → fewer weight fetches from CPU.
- Fewer K-tiles also means fewer partial-sum round-trips to HBM.
- The heuristic (`get_best_config_asym`) searches for the largest BLOCK_K that fits in shared memory under 2-stage budget, then picks BLOCK_M/N.

For example, on SM100 with FP8 (229 KB smem), this allows BLOCK_K up to 512 elements — processing half the K dimension of a typical MoE layer (K=7168) in just ~14 tiles.

### 4. Unified CPU + GPU MoE Execution — the Host as a Second Backend

Once expert weights live in CPU DRAM, the CPU sitting next to them can compute too. The unified MoE runtime (`asym_gemm.unified_moe`) splits each MoE forward into two buckets that run **concurrently** over the same pinned INT8 weight bytes:

- **CPU bucket** — experts with few routed tokens run on the host via the bundled `cpu_gemm` library (AMX INT8, with an AVX-512-VNNI fallback selected at runtime), on a work-stealing thread pool.
- **GPU bucket** — the remaining experts run on the GPU INT8 asymmetric kernel; the GPU bucket is enqueued first so host AMX work hides under the GPU stream.

This matters most at decode-like batch sizes, where the GPU's per-expert weight-fetch constant dominates: an expert with 4 routed tokens costs the GPU a full PCIe weight transfer, while the CPU reads the same bytes from local DRAM. Because the buckets overlap, the objective is not "pick the faster backend per expert" but minimizing the **makespan** — `max(T_cpu, T_gpu)` — over the whole routing histogram.

Dispatch is a static token-count threshold by default (`m_cpu = 16`). In **adaptive mode**, a linear wall-time cost model per backend (intercept + per-expert + per-row terms, priors derived from the layer shape and nominal hardware rates) is fitted online from observed bucket timings, and each forward scans all prefix splits of the experts (sorted by token count) for the minimum predicted makespan. See [`adaptive_dispatch.md`](../adaptive_dispatch.md) for the full design.

---

## Data Flow

```
CPU DRAM / Pinned Memory              GPU HBM
    W[expert, N_tile, k]                 X[tokens, k]
          │                                   │
          │  cp.async (SM89, PCIe)            │  cp.async / TMA (HBM)
          │  TMA (SM100, NVLink-C2C)          │
          ▼                                   ▼
      sW (shared memory)               sX (shared memory)
          │                                   │
          └────────── MMA (tensor core) ──────┘
                           │
                    FP32 accumulator
                           │  convert + store
                           ▼
                   O[tokens, N_tile] in HBM
                   (partial sum between K-tiles)
```

The GPU kernel accesses CPU weight memory directly — there is no host-side copy loop. Activations and output stay in GPU HBM. The output buffer doubles as the partial-sum accumulation buffer between K-tiles (BF16 read-modify-write).
