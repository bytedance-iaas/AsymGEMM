# Research Direction Audit for AsymGEMM LoRA SFT

## Context

The current project integrates AsymGEMM as a backend for LLaMA-Factory LoRA SFT, especially for MoE routed experts. This is useful engineering, but by itself it is not a strong ML systems research contribution.

The research question is narrower:

```text
What new problem appears specifically when applying AsymGEMM to LoRA SFT,
and what nontrivial mechanism solves it?
```

The contribution cannot be:

- just another LLaMA-Factory/KTransformers-style integration,
- just wrapping another model family,
- just using the existing AsymGEMM kernel,
- just choosing better block sizes,
- just storing weights in a different precision,
- just fusing generic LoRA forward/backward,
- just fusing gate/up MLP kernels,
- just saying CPU offload/heterogeneous MoE training.

Reviewers will likely reject those as incremental engineering or already-covered ideas.

## Related Work Boundaries

### LoRAFusion

LoRAFusion already covers generic LoRA fine-tuning fusion:

```text
forward:
  fuse base output + LoRA output

backward:
  fuse LoRA dB/dS
  fuse dX_base + dX_lora
```

Therefore, a contribution should not be framed as:

```text
we fuse LoRA forward/backward for SFT
```

Even if the implementation uses AsymGEMM, reviewers can view this as applying LoRAFusion's idea to a different GEMM backend.

### DeepGEMM / Fused MoE Kernels

DeepGEMM and similar systems already cover high-performance GEMM kernels and fused MoE-style operations such as gate/up fusion. Therefore, a contribution should not be:

```text
we fuse gate/up
we tune tiling/block sizes
we choose transposed vs non-transposed layout
```

Those are kernel engineering unless tied to a genuinely new asymmetric-memory training issue.

### KTransformers

KTransformers already claims CPU/GPU heterogeneous MoE inference and SFT:

```text
expert weights on CPU
expert compute on CPU
attention/dense parts on GPU
LF integration for SFT
```

AsymGEMM differs because:

```text
expert weights live in CPU memory
but expert GEMMs execute on GPU tensor cores
```

The research direction must exploit this distinction. It should not simply say "CPU offload for MoE SFT."

### Original AsymGEMM

The original AsymGEMM contribution is already the host-resident weight, GPU-computed GEMM mechanism. We cannot claim:

```text
CPU-resident frozen weights with GPU GEMM
```

as new. The new contribution must be about what changes under LoRA SFT/training.

## Hard Novelty Filter

A plausible paper-level contribution should satisfy most of these:

1. It addresses a bottleneck caused by AsymGEMM's asymmetric memory model.
2. It is specific to training or LoRA SFT, not just inference.
3. It is not a simple module integration or wrapper.
4. It does not copy generic LoRA fusion.
5. It has measurable impact on latency, throughput, or memory.
6. It preserves exact training semantics unless explicitly framed as approximate training with strong evidence.
7. It has enough technical depth that a reviewer cannot dismiss it as "one hour of engineering."

## Ideas Considered And Why They Are Weak/Risky

### 1. Plain LF Integration

Useful, but not research.

```text
AsymGEMM + LF + MoE LoRA
```

is a backend integration unless paired with a deeper mechanism.

### 2. More Model Wrappers

Adding Gemma4, Llama4, or more packed expert formats is engineering support, not a core contribution.

### 3. Base Weight Quantization

Plainly storing frozen base weights as FP8/FP4 and running similar kernels is likely too incremental.

It may help memory/bandwidth, but without a deeper training-specific mechanism reviewers can say:

```text
this is just quantized storage for frozen weights
```

### 4. Transposed/Non-Transposed Layouts

If profiling shows transposed vs non-transposed does not materially affect runtime and block size dominates, this is not a contribution.

### 5. Generic Expert Backward Fusion

A fused/tiled expert backward sketch like:

```text
stream W_down
compute dAct
apply activation derivative
stream W_gate/W_up
accumulate dX
accumulate LoRA gradients
```

is suspicious as a main contribution. It resembles generic fused MLP/LoRA backward ideas. Saying "the weights are CPU-resident" may not be enough to separate it from LoRAFusion-style fusion unless the new mechanism is fundamentally about asymmetric host-weight streaming.

### 6. Cross-Microbatch Token Coalescing

Idea:

```text
collect expert tokens across gradient-accumulation microbatches
run larger expert GEMMs
```

Benefit:

```text
larger M per expert
better amortization of CPU-resident weight streaming
```

Problem:

To coalesce across microbatches, the layer input activations for multiple microbatches must be live at the same time:

```text
C * batch * seq * hidden
```

This raises HBM requirements and can fight the purpose of using AsymGEMM in a memory-limited setting. Windowing, recomputation, and chunking can reduce the problem, but then benefits may shrink. This is a risky main direction.

### 7. Microbatch Temporal Weight Cache

Idea:

```text
reuse expert weights across nearby microbatches
without increasing expert token batch size
```

Full expert caching in HBM is usually impractical because it consumes the memory AsymGEMM is trying to save.

Tile/panel caching is more plausible but still needs extra HBM. Persistent-kernel reuse of a weight tile across multiple microbatch token ranges avoids full HBM caching, but still requires multiple microbatch layer activations to be available. Therefore it inherits some of the cross-microbatch memory issue.

### 8. Dropping Frozen-Weight Jacobians

Idea:

```text
skip some dX = dY W_frozen paths
only compute gradients needed for LoRA adapters
let residual paths carry upstream gradients
```

This has a clear punchline, but it changes gradients. It is approximate training, not exact LoRA SFT.

This is risky and potentially suspicious:

- upstream adapter gradients change,
- convergence can degrade,
- reviewers may reject it as unjustified gradient dropping,
- it is no longer a clean systems optimization.

Unless explicitly pursuing approximate training with strong quality results, discard this as a main direction.

## Current Honest Assessment

The clean systems space is narrow.

Exact training semantics strongly constrain possible optimizations:

```text
Forward must compute the same expert outputs.
Backward must propagate dX through frozen weights.
LoRA gradients must match normal gradient accumulation.
```

Avoiding CPU weight streaming is hard because exact backward still needs frozen-weight Jacobians. Reducing repeated streaming without larger live activation windows is also hard because reuse across microbatches requires those microbatches' layer activations to be available together.

Therefore, many "obvious" ideas either:

- become generic fusion,
- increase HBM,
- require approximate gradients,
- or reduce to engineering/tuning.

This is why the research direction must be chosen carefully.

## More Defensible Direction Family

The most defensible family is not a single easy optimization. It is:

```text
Training-aware asymmetric-memory execution for exact LoRA SFT
```

But this needs a sharp mechanism. A vague "adaptive runtime" is too weak unless tied to a precise new problem and a strong empirical result.

Possible sharper variants:

### A. Fixed-Memory Host-Streaming Scheduler

Goal:

```text
reduce host-weight streaming stalls without increasing activation memory beyond a budget
```

Potential mechanism:

- model AsymGEMM LoRA SFT as a constrained scheduling problem,
- inputs are per-layer expert token counts, host-link bandwidth, HBM budget, and forward/backward phase,
- choose per expert/layer whether to direct-stream, stage a panel, recompute, or fall back,
- guarantee fixed HBM budget.

Risk:

This still may look like engineering unless the scheduler has a crisp new primitive or provable/strong empirical advantage.

### B. Persistent Multi-Range AsymGEMM Kernel

Goal:

```text
load a CPU-resident expert weight tile once
apply it to multiple independent token ranges
discard it
```

Unlike token coalescing, do not create one giant packed tensor. The kernel takes route descriptors:

```text
RouteDesc {
  microbatch_id or buffer_id
  expert_id
  x_ptr / x_offset
  y_ptr / y_offset
  num_tokens
}
```

The scheduler does:

```text
for expert e:
  for weight tile:
    load W_e tile from CPU once
    for route range using expert e:
      compute output for that range
    discard W_e tile
```

This could reduce repeated CPU weight streaming without full expert HBM caching.

Risk:

It still requires multiple layer input activation buffers to be live if ranges come from multiple microbatches. It is more plausible as a kernel/runtime contribution than plain token coalescing, but memory pressure remains a core issue.

### C. Within-Microbatch Multi-Range Reuse

A safer variant avoids cross-microbatch activation blowup:

```text
reuse weight tiles across multiple token ranges within the same microbatch/layer
```

This may exploit fragmentation caused by routing, chunking, top-k, or sequence partitioning. It is less risky for memory, but the benefit may be smaller because current packed routing already groups tokens by expert inside a microbatch.

### D. Exact Backward Traffic Analysis And New Bottleneck Decomposition

A paper can become stronger if it first proves a non-obvious bottleneck:

```text
In AsymGEMM LoRA SFT, the dominant cost is not forward expert compute,
but repeated frozen-weight dX streaming under exact backward.
```

Then the contribution must directly address that bottleneck. Without this profiling result, proposed mechanisms are speculative.

## Recommended Next Step

Before committing to a paper direction, run a profiling study that answers:

1. How much step time is MoE forward base AsymGEMM vs MoE backward dX AsymGEMM?
2. How much time is LoRA path vs frozen base path?
3. What is the per-expert token count distribution per layer and per microbatch?
4. How often do the same experts repeat across gradient-accumulation microbatches?
5. Is host-weight streaming actually the dominant limiter, or is HBM/activation/LoRA/optimizer dominant?
6. How much spare HBM exists during AsymGEMM SFT for staging/caching/persistent windows?

Only after this can we know whether a research contribution is viable.

## Tentative Conclusion

At this point, the strongest honest statement is:

```text
The LF integration is engineering.
Most obvious kernel ideas are either already covered by related work or too incremental.
Cross-microbatch reuse is AsymGEMM-specific but risks increasing activation memory.
Dropping frozen-weight Jacobians gives a punchline but becomes approximate training.
The most promising exact direction is a training-aware asymmetric-memory scheduler/kernel
that reduces repeated host-weight streaming under a strict HBM budget.
```

The project needs one sharp mechanism in that last category to become a credible ML systems paper. Otherwise, reviewers may view it as a backend integration plus routine optimizations.

## Working Research Question

The most useful current framing:

```text
Can exact LoRA SFT with CPU-resident, GPU-computed frozen experts reduce repeated
host-weight streaming without increasing peak HBM or relying on approximate gradients?
```

If the answer is yes, that mechanism is likely the paper.
If the answer is no, then the contribution should be scoped as engineering infrastructure rather than a full research paper.
