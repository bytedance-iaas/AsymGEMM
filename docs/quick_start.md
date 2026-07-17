# Quick Start

## Requirements

- NVIDIA GPU: SM89 (RTX 4090 / L40S), SM90 (H100 / H200 / H20 / GH200), or SM100 (GB200)
- Python 3.8+
- C++17 compiler
- CUDA Toolkit 12.1+ for SM89 / SM90; CUDA Toolkit 12.9+ for SM100 / FP4
- PyTorch 2.1+
- CUTLASS (included as Git submodule)
- Optional, for the unified CPU + GPU MoE runtime: an x86 host with Intel AMX
  (Sapphire Rapids or newer) or AVX-512-VNNI

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

### Unified CPU + GPU MoE in SGLang (experimental)

The unified MoE kernel turns the host CPU into a second compute engine
instead of just a weight store. Expert weights are quantized to INT8 once at
load time and kept pinned in host RAM (no VRAM copy); on every forward each
expert is dispatched by its routed token count — small experts run on the CPU
(Intel AMX INT8) while large experts run on the GPU (SM90 INT8 grouped
kernel, streaming weights over PCIe via TMA). The two buckets execute
concurrently, so a large MoE model runs in a fraction of the VRAM with the
CPU absorbing the long tail of lightly-routed experts.

**Why environment variables?** This is an experimental feature. While it
stabilizes, it is configured entirely through `SGLANG_ASYMGEMM_UNIFIED_*`
environment variables so the existing `asym_gemm` backend, its CLI surface,
and its default behavior stay completely untouched — with the variables unset,
nothing changes. Once the feature matures, these knobs will migrate to proper
`sglang.launch_server` CLI options.

Requirements: an SM90 (Hopper) GPU, an x86 host with Intel AMX-INT8
(Sapphire Rapids or newer), a gated-SiLU MoE model whose hidden and
intermediate sizes are multiples of 128, and no expert parallelism
(`--ep-size 1`). If any prerequisite is missing, SGLang logs the reason and
falls back to the regular `asym_gemm` path.

```bash
# Enable the unified CPU + GPU MoE path (BF16 checkpoint,
# quantized to INT8 online at load time)
export SGLANG_ASYMGEMM_UNIFIED_MOE=1

python -m sglang.launch_server \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --moe-runner-backend asym_gemm \
    --mem-fraction-static 0.25
```

Online INT8 quantization is a per-expert loop over every MoE layer and can
dominate startup on large models. To skip it, convert the checkpoint offline
once (byte-identical to online quantization) and point the server at the
result:

```bash
# One-time offline conversion
python scripts/convert_int8_weights.py \
    --input-path /models/Qwen3-Coder-30B-A3B-Instruct \
    --output     /models/Qwen3-Coder-30B-A3B-Instruct-asymint8

export SGLANG_ASYMGEMM_UNIFIED_MOE=1
export SGLANG_ASYMGEMM_UNIFIED_INT8_PATH=/models/Qwen3-Coder-30B-A3B-Instruct-asymint8

python -m sglang.launch_server \
    --model /models/Qwen3-Coder-30B-A3B-Instruct \
    --moe-runner-backend asym_gemm \
    --mem-fraction-static 0.25
```

The offline path is required for block-scaled FP8 checkpoints, which cannot
be quantized to INT8 online — pass `--input-type fp8` to the converter.

| Environment variable | Default | Meaning |
|----------------------|---------|---------|
| `SGLANG_ASYMGEMM_UNIFIED_MOE` | `0` | Master switch. `1` enables the unified CPU + GPU MoE path for the `asym_gemm` backend. Off, the existing paths are untouched. |
| `SGLANG_ASYMGEMM_UNIFIED_M_CPU` | `16` | Per-expert dispatch threshold: experts with ≤ this many routed tokens run on the CPU AMX bucket, the rest on the GPU INT8 grouped kernel. Raise it to shift more work to the CPU. |
| `SGLANG_ASYMGEMM_UNIFIED_CPU_THREADS` | `0` | Total worker threads for the CPU bucket. One process-wide pool is shared by all MoE layers (read once when the first layer is built). `0` = use all hardware threads. |
| `SGLANG_ASYMGEMM_UNIFIED_INT8_PATH` | unset | Directory of pre-quantized INT8 expert weights from `scripts/convert_int8_weights.py`. When a matching `layer_{id}.safetensors` exists, the layer loads it directly instead of quantizing online; any missing file or shape mismatch falls back to online quantization. Empty = always quantize online. |

The lower-level `ASYMGEMM_*` runtime knobs (HBM expert caching, NUMA thread
placement, expert routing stats, …) also apply — see the
[environment variable reference](API.md#environment-variables).

Notes:

- Piecewise (prefill) CUDA graph capture is automatically disabled when the
  unified path is on; decode CUDA graphs work normally via pre-allocated
  capturable buffers.
- After conversion the BF16/FP8 master weights are released to reclaim host
  memory; the unified layer serves every forward for converted layers.
- Layers that cannot be converted (shape constraints, unsupported config)
  keep their master weights and continue on the existing `asym_gemm` path —
  check the server log for per-layer `AsymGEMM unified MoE:` messages.

## Standalone Usage

### Asymmetric BF16 GEMM (SM90 / SM100, weights in CPU DRAM)

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

# Flat (start, end) row pairs per expert, expert IDs with -1 sentinel
offsets = torch.tensor([0, 256, 256, 512, 512, 768, 768, 1024,
                        1024, 1280, 1280, 1536, 1536, 1792, 1792, 2048],
                       dtype=torch.int32, device="cuda")
experts = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, -1],
                       dtype=torch.int32, device="cuda")
list_size = experts.numel()   # entries in `experts`, including the sentinel

asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
    a, b, d, offsets, experts, list_size
)
```

The same call runs unchanged on any supported architecture — the library
dispatches to the matching kernel at call time.

### FP8 MoE GEMM (SM89 / SM90 / SM100, weights in CPU-pinned memory)

FP8 inputs are `(data, scales)` tuples. With 1×128 activation scales and
128×128 weight block scales:

```python
num_experts = 8
N, K = 4096, 7168
token_counts = [512, 256, 128, 64, 300, 100, 200, 400]
total_tokens = sum(token_counts)

a  = torch.randn(total_tokens, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
sa = torch.ones(total_tokens, K // 128, dtype=torch.float32, device="cuda")
b  = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device="cpu").pin_memory().to(torch.float8_e4m3fn)
sb = torch.ones(num_experts, N // 128, K // 128, dtype=torch.float32, device="cuda")
d  = torch.empty(total_tokens, N, dtype=torch.bfloat16, device="cuda")

# Flat (start, end) row pairs and -1-terminated expert IDs
import itertools
bounds  = [0] + list(itertools.accumulate(token_counts))
offsets = torch.tensor([x for s, e in zip(bounds, bounds[1:]) for x in (s, e)],
                       dtype=torch.int32, device="cuda")
experts = torch.tensor(list(range(num_experts)) + [-1],
                       dtype=torch.int32, device="cuda")

asym_gemm.m_grouped_fp8_asym_gemm_nt_contiguous(
    (a, sa), (b, sb), d, offsets, experts, experts.numel()
)
```

INT8 (SM90) follows the same pattern via `m_grouped_int8_asym_gemm_nt_contiguous`,
with per-token/per-channel fp32 scales — see `tests/test_sm90_int8.py`, and
`scripts/convert_int8_weights.py` for producing INT8 weights offline.

### Unified CPU + GPU MoE Layer

On a host with Intel AMX (or AVX-512-VNNI), the unified runtime executes
small experts on the CPU and large experts on the GPU concurrently:

```python
import torch
from asym_gemm.unified_moe import Layer

G, hidden, inter, top_k = 32, 1024, 2048, 4

gate = torch.randn(G, inter, hidden, dtype=torch.bfloat16)
up   = torch.randn(G, inter, hidden, dtype=torch.bfloat16)
down = torch.randn(G, hidden, inter, dtype=torch.bfloat16)

layer = Layer.from_bf16(gate, up, down, top_k=top_k, adaptive=True)

T = 256
x          = torch.randn(T, hidden, dtype=torch.bfloat16, device="cuda")
expert_ids = torch.randint(0, G, (T, top_k), device="cuda")
route_w    = torch.rand(T, top_k, device="cuda")

out = layer.forward(x, expert_ids, route_w)   # [T, hidden] bf16
```

With `adaptive=True`, the CPU/GPU expert partition is chosen per forward by an
online-fitted cost model; the default is a static threshold (experts with
≤ 16 routed tokens run on the CPU). See
[`adaptive_dispatch.md`](../adaptive_dispatch.md).

## Running Tests

One command runs the subset of GPU tests applicable to the detected
architecture, plus the unified MoE CPU/GPU parity suite:

```bash
bash scripts/test.sh
```

The standalone CPU GEMM library (`csrc/cpu/cpu_gemm`) has its own CTest suite:

```bash
bash scripts/test_cpu_gemm.sh
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
