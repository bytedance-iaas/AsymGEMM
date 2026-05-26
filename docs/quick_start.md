# Quick Start

## Requirements

- NVIDIA GPU: SM89 (RTX 4090 / L40S) or SM100 (GB200)
- Python 3.8+
- C++17 compiler
- CUDA Toolkit 12.1+ for SM89; CUDA Toolkit 12.9+ for SM100 / FP4
- PyTorch 2.1+
- CUTLASS (included as Git submodule)

## Installation

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

## Usage with SGLang

AsymGEMM integrates with [SGLang](https://github.com/bytedance-iaas/sglang/tree/asym_gemm_integration) as a MoE runner backend:

```bash
# Install SGLang with AsymGEMM integration
git clone -b asym_gemm_integration https://github.com/bytedance-iaas/sglang.git
cd sglang
pip install -e "python[all]"

# Launch server with AsymGEMM backend
# --mem-fraction-static can be set low (e.g., 0.25) since MoE weights
# are offloaded to CPU
python -m sglang.launch_server \
    --model Qwen/Qwen3.5-397B-A17B-FP8 \
    --moe-runner-backend asym_gemm \
    --mem-fraction-static 0.25
```

## Standalone Usage

### Asymmetric BF16 GEMM (SM100, weights in CPU DRAM)

```python
import torch
import asym_gemm

num_groups = 8
N, K = 4096, 7168
M = 2048

# Activations on GPU, expert weights in CPU pinned memory
a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
b = torch.randn(num_groups, N, K, dtype=torch.bfloat16, device="cpu").pin_memory()
d = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

# Offset pairs (start, end) and expert IDs for contiguous layout
offsets = torch.tensor([0, 256, 256, 512, 512, 768, 768, 1024,
                        1024, 1280, 1280, 1536, 1536, 1792, 1792, 2048],
                       dtype=torch.int32, device="cuda")
experts = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, -1],
                       dtype=torch.int32, device="cuda")
list_size = torch.tensor([len(offsets)], dtype=torch.int32, device="cuda")

asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
    a, b, d, offsets, experts, list_size
)
```

### FP8 MoE GEMM (SM89, weights in CPU-pinned memory)

```python
num_experts = 8
N, K = 4096, 7168
token_counts = [512, 256, 128, 64, 300, 100, 200, 400]
total_tokens = sum(token_counts)

# Activations on GPU, weights in CPU pinned memory
a = torch.randn(total_tokens, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
b = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device="cpu").pin_memory().to(torch.float8_e4m3fn)
d = torch.empty(total_tokens, N, dtype=torch.bfloat16, device="cuda")

# Cumulative end-token indices and expert IDs
import itertools
offsets = torch.tensor(list(itertools.accumulate(token_counts)),
                       dtype=torch.int32, device="cuda")
experts = torch.arange(num_experts, dtype=torch.int32, device="cuda")

asym_gemm.m_grouped_fp8_asym_gemm_sm89(
    a, b, d, offsets, experts, num_experts,
    scale_a=1.0, scale_b=1.0
)
```

## Running Tests

```bash
pytest tests/ -v
```

## JIT Compilation Notes

The first call to any kernel variant triggers JIT compilation via NVRTC (~5–30 seconds depending on complexity). Compiled kernels are cached under `~/.asym_gemm/` — subsequent calls with the same configuration are near-instant.

To precompile kernels at server startup (recommended for production):

```bash
# SGLang handles this automatically with:
export SGLANG_JIT_ASYMGEMM_PRECOMPILE=True
```

To force recompilation (e.g., after upgrading):

```python
asym_gemm.set_compile_mode(1)  # force compile
# ... run kernels ...
asym_gemm.set_compile_mode(0)  # back to cache mode
```
