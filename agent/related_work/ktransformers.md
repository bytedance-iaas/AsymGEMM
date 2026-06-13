# KTransformers Heterogeneous MoE Notes

KTransformers is a CPU-GPU heterogeneous execution system for large MoE
inference and LoRA SFT. Its core design point is not LoRA kernel fusion and not
host-weight GPU GEMM. It keeps the large routed-expert weights out of GPU HBM,
runs the expert operators on optimized CPU kernels, and leaves the rest of the
model/training stack visible through normal framework modules.

The high-level split is:

```text
KTransformers:
  CPU-resident expert weights
  CPU expert forward/backward kernels
  GPU attention/router/dense work and orchestration

AsymGEMM:
  CPU-resident frozen weights
  GPU tensor-core GEMM compute
```

So KTransformers is a close systems baseline for "large weights do not fit in
HBM", but the compute placement is different.

## Shapes

For a routed MoE layer, tokens are first assigned to experts.

```text
X        [M, H]        hidden states
topk     [M, K]        selected experts per token
score    [M, K]        routing weights

For one expert e:
X_e      [M_e, H]      tokens routed to expert e
W_gate   [I, H]
W_up     [I, H]
W_down   [H, I]
Y_e      [M_e, H]
```

The expert computation is the standard gated FFN:

```text
gate = X_e W_gate^T
up   = X_e W_up^T
act  = silu(gate) * up
Y_e  = act W_down^T
```

The global MoE output scatters each expert output back to token order and
combines it with the router scores.

## Core System Split

### Background

For large MoE models, expert parameters dominate model size even though each
token uses only a small number of experts. Attention, routing, KV-cache work,
and dense layers still benefit from GPU execution, but placing every expert
weight in HBM is the main memory problem.

KTransformers attacks that problem by making MoE execution heterogeneous at the
operator level.

### Method

The system places different parts of the model on different devices:

```text
GPU:
  attention
  router
  dense / non-expert layers
  loss and training orchestration

CPU:
  routed expert weights
  expert forward kernels
  expert backward kernels for SFT
  NUMA-aware scheduling and thread pools

Transfers:
  selected token activations -> CPU
  routed expert outputs -> GPU
  expert-output gradients -> CPU
  input gradients -> GPU
```

This is not generic tensor offload. The MoE module itself is replaced by a
backend operator whose implementation knows about expert routing, CPU memory
layout, quantized weights, NUMA placement, and forward/backward reuse.

### Why

The main memory saving comes from not storing all expert weights in GPU HBM.
This lets very large MoE models run or fine-tune with commodity GPU memory, at
the cost of depending on CPU throughput, CPU memory bandwidth, PCIe transfers,
and careful NUMA placement.

## Inference Path

For inference, KTransformers exposes optimized MoE kernels through `kt-kernel`
and integrates with serving stacks such as SGLang.

The key mechanism is hot/cold expert placement:

```text
hot experts:
  placed on GPU when selected frequently enough

cold experts:
  remain on CPU
  executed by optimized CPU kernels
```

The CPU backends include AMX INT4/INT8 paths, AVX512 native precision paths
such as BF16/FP8/RAWINT4, llamafile/GGUF support, and BLIS-based support on
AMD CPUs. The wheel can select CPU variants at runtime based on the available
instruction set.

Secondary serving features include CLI wrappers, prefix-cache support,
multi-concurrency support, conversion utilities, and broad model-specific
optimize rules. Those make the system usable, but the central contribution is
still the heterogeneous expert execution.

## SFT Path

### Background

LoRA SFT freezes the base model weights, but the forward pass still needs the
base expert computation and the backward pass still needs gradients through the
expert inputs. For MoE layers, that means the backend cannot be inference-only:
autograd must see a differentiable operator.

### Method

KTransformers uses LLaMA-Factory for the outer training stack:

```text
LLaMA-Factory:
  dataset and training loop
  LoRA configuration
  optimizer and scheduler
  checkpointing and inference entry points

KTransformers:
  injected attention / MoE modules
  CPU expert kernels
  placement policy from optimize-rule YAML
```

For attention projections, KTransformers combines its injected linear module
with PEFT's LoRA layer:

```text
KTransformersLinearLora =
  KTransformersLinear fast path
  + LoRA parameters lora_A / lora_B
```

This preserves the KTransformers prefill/generate linear implementation while
allowing the usual LoRA adapters on Q/K/V/O projections.

For MoE, the important SFT object is a custom autograd node, commonly described
as `KSFTExpertsCPU`:

```text
forward:
  copy hidden states, expert ids, and routing weights to pinned CPU buffers
  run the CPU routed-expert kernel
  copy expert outputs back to GPU

backward:
  copy dY to pinned CPU buffers
  run the CPU expert backward kernel
  copy dX back to GPU
```

The CPU implementation caches useful forward intermediates and precomputes or
caches transposed expert weights when the backward kernel needs `W^T`. That is
what makes the CPU MoE operator usable for training rather than only serving.

## Why Not Treat It As Generic Offload?

Generic offload moves tensors between host and device around an otherwise
ordinary PyTorch graph. KTransformers changes the operator boundary. The routed
expert block becomes a CPU kernel with its own memory layout, quantization
format, scheduling, and backward implementation.

That distinction matters:

```text
Plain offload:
  move parameters or activations
  still rely mostly on framework-level operator decomposition

KTransformers:
  replace the MoE operator
  execute expert compute inside a specialized backend
  expose only the module/autograd boundary to PyTorch
```

This is why the work is better understood as operator-level heterogeneous
execution than as a ZeRO-Offload-style parameter movement system.

## Why It Is Not AsymGEMM

KTransformers and AsymGEMM share the same pressure point: large model weights
do not fit comfortably in GPU memory. They differ in where the large matrix
multiplications execute.

```text
KTransformers:
  expert weights live on CPU
  expert GEMMs run on CPU
  performance depends on AMX/AVX/BLIS/llamafile kernels, NUMA, and PCIe

AsymGEMM:
  frozen weights live on CPU
  GEMM compute runs on GPU tensor cores
  performance depends on host-to-GPU weight movement and GPU-side scheduling
```

So KTransformers is strongest when the CPU backend is powerful enough to absorb
the sparse expert workload. AsymGEMM is aimed at preserving GPU GEMM execution
while relieving HBM pressure from frozen weights.

## Multi-GPU Placement

KTransformers uses explicit placement rather than normal data parallel
replication for very large models. The training wrapper controls where modules
are constructed and prevents the full model from being moved onto one GPU.

The rough placement rule is:

```text
layers:
  constructed on target GPUs according to the optimize rule

activations:
  mostly stay local to the owning GPU

gradients:
  reduced to a chosen device when needed

experts:
  usually CPU-resident unless selected for GPU placement
```

This is a necessary systems detail for 100B+ MoE SFT, but it is secondary to
the main operator split.

## Kernel Summary

```text
Inference:
  CPUInfer / KT MoE wrappers
  AMX INT4 / INT8
  AVX512 BF16 / FP8 / RAWINT4
  llamafile / GGUF
  BLIS backend
  SGLang integration

SFT:
  KTransformersLinearLora for attention projections
  KSFTExpertsCPU-style custom autograd node
  AMX BF16 / AMX INT8 / llamafile expert backward paths
  pinned CPU transfer buffers
  cached W^T and forward intermediates

Control surface:
  LLaMA-Factory configs
  use_kt: true
  kt_optimize_rule YAML placement
```

## Positioning for AsymGEMM

KTransformers is useful related work because it demonstrates that large MoE
LoRA/SFT can be made practical by keeping expert weights out of GPU memory and
by specializing the execution path around the routed expert operator.

The comparison should be framed as:

```text
Shared goal:
  reduce GPU HBM pressure from large frozen MoE weights
  support large-model inference or SFT on limited GPU memory

Different mechanism:
  KTransformers computes routed experts on CPU
  AsymGEMM computes frozen base GEMMs on GPU from CPU-resident weights

Different bottleneck:
  KTransformers is CPU-kernel / NUMA / PCIe sensitive
  AsymGEMM is GPU scheduling / host-weight streaming sensitive
```

The clean one-line summary is:

```text
KTransformers moves massive sparse expert compute to CPU;
AsymGEMM keeps the compute on GPU and moves only the frozen-weight residency.
```

## Secondary Contributions

KTransformers also contributes a significant amount of engineering around the
main idea:

```text
serving:
  SGLang integration
  CLI and OpenAI-compatible server paths
  prefix cache and multi-concurrency support

model support:
  DeepSeek, Qwen, Mixtral, and other MoE optimize rules
  hot/cold expert placement policies
  quantization and conversion utilities

training UX:
  LLaMA-Factory integration
  YAML-based placement control
  multi-GPU construction rules
```

These pieces matter for adoption, but they should not dominate the technical
explanation. The core contribution is the same throughout: place the large MoE
experts outside HBM, execute them with specialized CPU kernels, and keep the
upper framework interface close to ordinary modules and autograd.
