# Cross-Microbatch Expert Coalescing for AsymGEMM LoRA SFT

## One-Line Idea

Standard LoRA SFT executes each gradient-accumulation microbatch independently, which gives AsymGEMM small per-expert token groups and poor CPU-resident weight reuse. Coalesce routed expert work across gradient-accumulation microbatches so each active expert sees a larger `M` before fetching host-resident weights.

## Why This Is AsymGEMM-Specific

AsymGEMM keeps frozen expert weights in CPU DRAM and lets GPU kernels fetch active weight tiles directly. Its efficiency depends on amortizing each CPU-to-GPU weight tile over enough tokens assigned to the same expert.

Normal SFT scheduling is mismatched to this cost model:

```text
for microbatch in gradient_accumulation:
  full forward
  full backward
```

Each MoE layer only sees one microbatch at a time. If expert `e` receives few routed tokens in each microbatch, AsymGEMM repeatedly fetches the same expert weights with little reuse.

Coalescing changes the execution order without changing training semantics:

```text
expert e tokens from mb0 + mb1 + ... + mbK
  -> one larger grouped AsymGEMM
  -> scatter outputs back to original microbatches
```

This targets the core AsymGEMM bottleneck: host-weight fetch amortization. It is not generic LoRA fusion, not CPU-executed MoE, and not an easy model-wrapper integration.

## Non-Goals

- Do not claim LLaMA-Factory integration as the research contribution.
- Do not claim generic LoRA forward/backward fusion; LoRAFusion covers that space.
- Do not claim fused gate/up alone; DeepGEMM-style MoE kernels cover that space.
- Do not claim CPU-side MoE SFT; KTransformers covers CPU-executed heterogeneous MoE SFT.
- Do not frame this as block-size tuning, transpose layout selection, or plain quantized storage.

## Proposed Training Schedule

Instead of processing one microbatch end-to-end, use a layerwise gradient-accumulation schedule.

Forward for layer `L`:

```text
for mb in GA window:
  run non-MoE sublayers up to MoE layer L
  compute router top-k
  save hidden states, expert ids, routing weights, and return positions

pack all routed tokens from mb0..mbK
group by expert
run coalesced AsymGEMM expert MLP
scatter outputs back to each microbatch
continue to next layer
```

Backward for layer `L`:

```text
for mb in GA window:
  collect dY at MoE layer L

pack dY by saved routing metadata
group by expert across microbatches
run coalesced expert backward:
  dX_base = dY W
  LoRA gradient paths
scatter dX back to each microbatch
continue reverse layer traversal
```

The math should match ordinary gradient accumulation. Only the order of independent expert GEMM work changes.

## Compatibility With Expert-Level Recomputation

This idea is compatible with expert-level activation recomputation and likely needs it for memory control.

Coalescing improves latency:

```text
larger tokens_per_expert -> better host-weight reuse
```

Recomputation controls HBM:

```text
save fewer expert intermediates -> lower activation memory
```

In backward, dropped expert intermediates can be recomputed on the same coalesced token groups. That can make recomputation cheaper per token because recomputed base GEMMs also have larger `M`.

## Memory Implications

CPU memory should not grow much beyond existing host-resident weights and metadata. The risk is GPU HBM.

This is the main weakness of the direction. Cross-microbatch coalescing is only possible if multiple microbatches' layer input activations are live at the same layer. The dataloader only provides raw input ids; the kernel needs GPU hidden states produced by earlier layers. Therefore a coalescing window of `C` microbatches inherently increases live layer-boundary activation memory by roughly `C`.

Naive coalescing can hold multiple microbatch activations at once:

```text
extra HBM ~= C * batch * seq * hidden
```

For `C = 2`, this can already roughly double the relevant layer-boundary activation footprint. For `C = 4`, it can be much worse. This can directly conflict with the reason for using AsymGEMM in the first place: fitting LoRA SFT under tight HBM.

The intended design avoids full-model activation blowup:

```text
execute layerwise
keep only layer-boundary activations and routing metadata
release packed expert buffers after each layer
use recomputation for selected expert intermediates
```

This reduces but does not eliminate the activation-memory increase. The direction is only practical if a small window, e.g. `C = 2`, gives enough host-weight reuse to offset the extra activation memory and scheduling complexity.

If peak HBM is already near the limit, this direction may be impractical as a main optimization. In that case it should be treated as a risky secondary direction or a microbenchmark study rather than the core paper contribution.

## Implementation Strategy

### Milestone 1: Kernel-Level Microbenchmark

Use saved or synthetic routing metadata.

Compare:

```text
separate microbatches:
  run expert grouped AsymGEMM K times

coalesced:
  concatenate routes across K microbatches
  run one larger grouped AsymGEMM
```

Measure:

- per-expert token counts,
- AsymGEMM time,
- effective bandwidth,
- sensitivity to GA size, sequence length, top-k, and route skew.

This proves whether the core cost-model hypothesis is real before building a trainer.

### Milestone 2: Single MoE Layer Correctness

Build a standalone coalesced MoE layer wrapper:

- accepts a list of microbatch hidden states,
- computes or consumes routing metadata,
- packs tokens across microbatches,
- calls existing packed Asym expert kernels,
- scatters outputs back,
- checks numerical equivalence against independent execution.

Then add backward equivalence:

- input gradients,
- LoRA gradients,
- route-weight semantics.

### Milestone 3: Toy Layerwise Trainer

Use a small MoE transformer or reduced Qwen/Gemma-style block.

Implement manual gradient accumulation:

```text
for layer in layers:
  forward all GA microbatches through layer
  coalesce MoE inside layer

compute loss per microbatch

for layer in reversed(layers):
  backward all GA microbatches through layer
  coalesce MoE backward inside layer
```

This bypasses normal HuggingFace autograd assumptions and makes the schedule explicit.

### Milestone 4: Real Model Integration

Only after proving the schedule:

- integrate with Qwen/Gemma/Llama MoE blocks,
- preserve LF dataset/optimizer/checkpoint interfaces where possible,
- keep the coalescing runtime separate from the baseline LF integration.

## Why A Plain Module Queue Is Not Enough

A normal PyTorch module cannot easily wait for future microbatches:

- `forward()` must return a tensor immediately,
- autograd expects a complete graph for that microbatch,
- delayed outputs can deadlock the trainer,
- HF/LF assumes normal forward/backward ordering.

Therefore, full coalescing is a trainer/runtime redesign, not just a module replacement.

## Research Claims To Test

1. Standard gradient accumulation leaves cross-microbatch expert reuse unexploited.
2. AsymGEMM LoRA SFT is sensitive to tokens per active expert because host-weight fetch is first-order.
3. Cross-microbatch expert coalescing increases effective `M` per expert and improves step latency.
4. Expert-level recomputation can bound the activation memory overhead of coalescing.
5. The combined schedule preserves exact LoRA SFT semantics up to normal floating-point ordering differences.

## Key Risks

- Full training implementation is substantial.
- Layerwise execution necessarily increases live layer-boundary activations for window size `C`; recomputation only reduces internal expert intermediates.
- Even `C = 2` can roughly double the relevant layer-boundary activation footprint.
- Benefits may shrink for long sequences or large batches where per-expert `M` is already high.
- Route distributions may be skewed; some experts may still receive too few tokens.
- Distributed training and pipeline/data parallelism complicate the scheduler.

## Evaluation Plan

Baselines:

- current LF + AsymGEMM MoE LoRA SFT,
- Torch/GPU-resident baseline when it fits,
- KTransformers as a heterogeneous CPU-executed MoE SFT baseline where applicable.

Metrics:

- step time,
- MoE forward/backward time,
- tokens/sec,
- peak HBM,
- CPU memory,
- CPU-to-GPU weight traffic estimate,
- per-expert token histogram,
- loss equivalence and final task quality.

Sweeps:

- GA size,
- sequence length,
- batch size,
- top-k,
- number of experts,
- route skew,
- recomputation policy.

## Paper Framing

Possible title:

```text
Amortizing Host-Resident Expert Weights in LoRA SFT via Cross-Microbatch Asymmetric GEMM Scheduling
```

Core contribution:

```text
We identify that standard gradient-accumulation scheduling is inefficient for
CPU-resident GPU-computed expert weights, and propose an exact layerwise
coalescing schedule that increases expert-token reuse across microbatches.
```

This is a credible systems contribution because it changes the execution schedule to match AsymGEMM's asymmetric memory cost model, rather than adding another backend wrapper or small kernel tweak.
