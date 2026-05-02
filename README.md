<p align="center">
  <h1 align="center">AsymGEMM</h1>
  <p align="center">High-Performance Asymmetric GEMM & MoE Kernels for NVIDIA GPUs</p>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python 3.8+"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12.9%2B-green" alt="CUDA 12.9+"></a>
  <a href="#supported-operations"><img src="https://img.shields.io/badge/SM80-A100-lightgrey" alt="SM80"></a>
  <a href="#supported-operations"><img src="https://img.shields.io/badge/SM89-Ada%20Lovelace-orange" alt="SM89"></a>
  <a href="#supported-operations"><img src="https://img.shields.io/badge/SM100-Blackwell-red" alt="SM100"></a>
  <a href="#supported-operations"><img src="https://img.shields.io/badge/BF16-FP8-FP4-brightgreen" alt="Precision"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
</p>

---

## About

AsymGEMM is a high-performance CUDA kernel library for general matrix multiplication (GEMM) on NVIDIA GPUs, with two core capabilities:

1. **Asymmetric Memory Access** — MoE expert weights are stored in CPU memory, and only the weights needed for active experts are moved to GPU on demand. 

2. **Mixture-of-Experts (MoE) Grouped GEMM** — Specialized grouped GEMM kernels for MoE inference workloads with support for masked execution (skipping padding rows) and contiguous dispatch layouts.

By leveraging abundant, low-cost CPU DRAM and high-bandwidth interconnects, this design dramatically cuts the number of GPUs required and reduces total serving cost. AsymGEMM is particularly well-suited for **prefill-heavy, less latency-sensitive workloads** (e.g., document scoring, batch classification, embedding generation, long-context summarization) where it delivers **2×–4× higher per-GPU throughput** compared to conventional deployments using 4× more GPUs.


### Key Features

- **Asymmetric Memory** — MoE weights reside in CPU memory; only active expert weights are moved to GPU on demand, dramatically reducing GPU count and total serving cost
- **MoE Grouped GEMM** — Efficient expert-parallel dispatch with K-outer M-inner tiling to amortize weight load latency
- **Multi-Architecture** — Native optimized kernels for SM80 (A100), SM89 (RTX 4090 / L40S), and SM100 (B200), each exploiting architecture-specific features
- **Flexible Quantization** — BF16/FP16, FP8 (E4M3) with per-token-group scales, and NVFP4 (E2M1 packed) with per-row scales are supported. 

---

## Getting Started

### Requirements

- NVIDIA GPU: SM80 (A100), SM89 (RTX 4090 / L40S), or SM100 (GB200)
- Python 3.8+
- C++20 compiler
- CUDA Toolkit 12.9+
- PyTorch 2.1+
- CUTLASS 4.0+ (included as Git submodule)

### Installation

```bash
# Clone with submodules (CUTLASS, fmt)
git clone --recurse-submodules https://github.com/bytedance-iaas/AsymGEMM.git
cd AsymGEMM

# Install
bash install.sh
```

Verify the installation:

```bash
python -c "import asym_gemm; print(asym_gemm.__version__)"
```

### Usage with SGLang

AsymGEMM integrates with [SGLang](https://github.com/bytedance-iaas/sglang/tree/asym_gemm_integration) as a MoE runner backend:

```bash
# Install SGLang with AsymGEMM integration
git clone -b asym_gemm_integration https://github.com/bytedance-iaas/sglang.git
cd sglang
pip install -e "python[all]"

# Launch server with AsymGEMM backend
# --mem-fraction-static could be set low (e.g., 0.25 in GB200) since MoE weights
# are offloaded to CPU
python -m sglang.launch_server \
    --model Qwen/Qwen3.5-397B-A17B-FP8 \
    --moe-runner-backend asym_gemm \
    --mem-fraction-static 0.25
```

### Standalone Usage

```python
import torch
import asym_gemm

# Asymmetric BF16 GEMM: matrix A in CPU memory, B in GPU
a = torch.randn(4096, 7168, dtype=torch.bfloat16, device="cpu").pin_memory()
b = torch.randn(8, 4096, 7168, dtype=torch.bfloat16, device="cuda")
d = torch.empty(4096, 4096, dtype=torch.bfloat16, device="cuda")
m_indices = torch.tensor([512, 1024, 1536, 2048, 2560, 3072, 3584, 4096],
                         dtype=torch.int32, device="cuda")

asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(a, b, d, m_indices)
```

```python
# MoE Grouped GEMM: FP8 with expert routing
a_fp8 = torch.randn(4096, 7168, dtype=torch.float8_e4m3fn, device="cuda")
b_fp8 = torch.randn(8, 4096, 7168, dtype=torch.float8_e4m3fn, device="cuda")
d = torch.empty(4096, 4096, dtype=torch.bfloat16, device="cuda")
offsets = torch.tensor([512, 1024, 1536, 2048, 2560, 3072, 3584, 4096],
                       dtype=torch.int32, device="cuda")
experts = torch.arange(8, dtype=torch.int32, device="cuda")
scale_a = torch.ones(4096, 7168 // 128, dtype=torch.float32, device="cuda")
scale_b = torch.ones(8, 4096, 7168 // 128, dtype=torch.float32, device="cuda")

asym_gemm.m_grouped_fp8_asym_gemm_sm80(
    a_fp8, b_fp8, d, offsets, experts, 8, scale_a, scale_b
)
```

### Running Tests

```bash
pytest tests/ -v
```

---

## Benchmark and Performance

By offloading MoE weights to CPU memory and serving large models with fewer GPUs, AsymGEMM delivers significantly higher per-GPU throughput — turning cheap DRAM into a direct cost advantage.

### Per-GPU Throughput: AsymGEMM (1 GPU) vs Vanilla (4 GPUs)

**Model:** Qwen3.5-397B-A17B-FP8 | **Hardware:** GB200 | **Benchmark:** 64 prompts, max concurrency 32

AsymGEMM fits the full 397B-parameter MoE model on **1 GPU**, while the vanilla baseline requires **4 GPUs** (TP=4 EP=4). The table below compares per-GPU throughput (total tok/s ÷ #GPUs):

| Input Len | Output Len | AsymGEMM TP1EP1<br/>(1 GPU) | Vanilla TP4EP4<br/>(4 GPUs) | Per-GPU Speedup |
|----------:|-----------:|----------------------------:|----------------------------:|:---------------:|
| 1000 | 2 | 625.87 tok/s | 161.91 tok/s | **3.87×** |
| 1000 | 50 | 357.71 tok/s | 117.20 tok/s | **3.05×** |
| 3500 | 2 | 1487.07 tok/s | 376.51 tok/s | **3.95×** |
| 3500 | 50 | 1098.70 tok/s | 555.50 tok/s | **1.98×** |
| 5000 | 2 | 2837.34 tok/s | 734.80 tok/s | **3.86×** |
| 5000 | 50 | 1606.78 tok/s | 780.91 tok/s | **2.06×** |

> AsymGEMM dominates the **prefill-heavy / short-output regime** (output ≤ 50 tokens), delivering **2×–4× higher per-GPU throughput** while using 4× fewer GPUs. Ideal for workloads like document scoring, batch classification, MMLU-style evaluation, retrieval reranking, and embedding generation.


---

## Contact Us

- **GitHub Issues**: [Report bugs or request features](https://github.com/bytedance-iaas/AsymGEMM/issues)
- **Email**: TBD

---

## Acknowledgement

We would like to thank the following projects that inspired and informed the development of AsymGEMM:

- **[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)** — We learned a great deal from DeepGEMM's elegant JIT-based approach to high-performance GEMM. Its clean design philosophy around runtime kernel specialization was an invaluable reference throughout our development.
- **[CUTLASS](https://github.com/nvidia/cutlass)** — For the foundational tile algebra abstractions and CuTe tensor layout primitives that underpin our kernel implementations.
- **[SGLang](https://github.com/sgl-project/sglang)** — For providing the serving infrastructure and MoE dispatch framework that AsymGEMM integrates with.


## License

AsymGEMM is released under the [MIT License](LICENSE).
