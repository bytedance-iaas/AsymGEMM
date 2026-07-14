# MoE LoRA-SFT AsymGEMM Story

## Contributions

This project should be framed as a LoRA-SFT system built around AsymGEMM, not
as a generic LoRA optimizer. The contributions are:

1. **AsymGEMM LoRA-SFT runtime for CPU-resident frozen weights.**

   We build the missing training path where frozen base weights live on CPU,
   but the GPU still computes the frozen base GEMMs through AsymGEMM. This is
   different from normal PyTorch/PEFT/Unsloth, where the base weights are
   usually GPU-resident or quantized GPU-resident, and different from
   KTransformers, where MoE expert compute is moved to CPU. The required SFT
   semantics are: no base-weight `dW`, correct forward through `W_base`,
   correct backward `dX = dY @ W_base`, and correct LoRA adapter gradients and
   optimizer state.

2. **MoE-aware AsymGEMM block primitive for routed expert base GEMMs.**

   We move beyond treating each projection as an independent AsymGEMM call.
   In a MoE MLP block, `gate` and `up` share the same routed tokens, expert
   ids, offsets, and input activation. The system should exploit that structure
   by fusing routed `gate + up` forward and routed `gate + up` backward `dX`.
   This targets the measured dominant bottleneck: repeated frozen expert base
   AsymGEMMs, not small LoRA A/B adapter GEMMs.

3. **Training-time dX support as a first-class systems target.**

   In inference, only the frozen base forward matters. In LoRA-SFT, frozen
   base weights still participate in backward because gradients must flow to
   earlier layers and earlier adapters. Therefore `dX` through CPU-resident
   frozen expert weights is not a side detail; it is half of the training
   problem. Optimizing backward `dX` is a real LoRA-SFT-specific contribution
   and is not covered by inference-only AsymGEMM usage.

4. **Low-precision CPU-resident expert-weight study for SFT.**

   The memory and bandwidth story should include BF16, FP8, and eventually FP4
   CPU-resident frozen expert weights. The point is not to replace QLoRA. The
   point is to ask when host-weight GPU compute becomes practical for LoRA-SFT
   as CPU-to-GPU bandwidth pressure drops. This gives the paper a broader
   speed-memory tradeoff beyond just one fusion optimization.

5. **Rigorous characterization against strong LoRA and MoE baselines.**

   The paper should report where AsymGEMM LoRA-SFT wins, loses, and breaks
   even against PyTorch/PEFT, QLoRA/bitsandbytes, Unsloth, LoRAFusion if
   buildable, and KTransformers as the CPU-expert MoE comparison. This is a
   contribution because the value of CPU-resident GPU-computed frozen weights
   is regime-dependent. The result should be a decision framework, not just a
   single speedup number.

6. **One-block MoE LoRA-SFT profiling methodology.**

   We isolate one full MoE block with real LoRA-SFT semantics so the profile is
   interpretable without full-framework noise. The benchmark should attribute
   time and memory to frozen base forward, frozen base `dX`, LoRA adapters,
   routing/pack/scatter, activation, optimizer, and CPU/no-kernel bubbles. This
   makes the bottleneck claim defensible and prevents the paper from looking
   like an anecdotal kernel microbenchmark.

## Motivation

LoRA-SFT freezes the base weights, but it does not remove the cost of applying
the frozen base weights.

For one frozen projection:

```text
Y = X @ W_base^T + LoRA(X)
```

Backward still needs:

```text
dX_base = dY @ W_base
```

LoRA removes `dW_base`, but it does not remove the frozen-base forward GEMM or
the frozen-base dX GEMM. Therefore, the core bottleneck in MoE LoRA-SFT is not
the small LoRA compute. It is the repeated routed frozen expert GEMM work.

In the current one-block profile, the dominant rows are:

```text
gate base AsymGEMM: 11.651 ms
down base AsymGEMM:  9.091 ms
up base AsymGEMM:    9.045 ms
```

Together these are about `29.787 ms`, roughly half of the listed one-block
time. This is the core issue the paper should target.

## Related-Work Boundary

We should not position this as another generic LoRA optimization system. That
would be too close to existing work and likely weaker than mature baselines.

The contribution boundary should be:

| Existing direction | What they already cover | What we should avoid claiming |
|---|---|---|
| Unsloth / QLoRA / bitsandbytes | practical fast LoRA/QLoRA training, quantized GPU-resident bases, kernelized training paths | do not claim generic faster LoRA or better QLoRA unless directly benchmarked |
| LoRAFusion | fused LoRA fine-tuning overhead, memory-bound LoRA ops, multi-LoRA batching/scheduling | do not make generic LoRA A/B fusion or elementwise LoRA epilogues the main contribution |
| mLoRA / ALTO / tLoRA | many concurrent adapters/jobs, shared base execution, scheduling | do not claim multi-adapter scheduling or cluster-level LoRA training |
| KTransformers | CPU-resident MoE experts computed on CPU with CPU kernels | do not claim CPU expert execution; our design point is CPU-resident weights with GPU compute |
| PyTorch / PEFT | standard LoRA semantics and correctness | use as correctness/runtime baseline, not as novelty |

Therefore, the paper should focus on the missing design point:

> LoRA-SFT over MoE blocks where frozen expert weights are CPU-resident, the
> expert base GEMMs still execute on GPU via AsymGEMM, and the training-time
> forward plus dX path is optimized as a routed MoE block primitive.

This is different from LoRAFusion/Unsloth because the target bottleneck is not
ordinary LoRA adapter overhead. It is the repeated CPU-resident frozen expert
base GEMM path exposed by LoRA-SFT.

## Why Fusion Makes Sense

In a MoE MLP block, gate and up projections share the same routed tokens,
expert ids, expert offsets, and input activation:

```text
gate = X @ W_gate^T
up   = X @ W_up^T
```

Executing them as independent AsymGEMM calls repeats routing metadata traversal,
input reads, launches, and related setup. LoRA-SFT also still needs backward dX
through both frozen base matrices, so the same structural redundancy exists in
training backward.

## Optimization 1: Fuse Gate + Up Forward

Replace two routed grouped GEMMs:

```text
gate = X @ W_gate^T
up   = X @ W_up^T
```

with one wider routed grouped GEMM:

```text
[gate, up] = X @ [W_gate; W_up]^T
```

Then split the output and continue with the normal activation path.

Expected benefit:

- one AsymGEMM launch instead of two
- one routing metadata traversal instead of two
- one input read stream instead of two
- potentially better occupancy and arithmetic intensity from the wider output

Based on the current profile:

```text
gate + up = 11.651 + 9.045 = 20.696 ms
```

If fusion saves 10-30% of gate+up base time, that is about `2.1-6.2 ms`, or
roughly `4-11%` one-block speedup. A conservative expectation for forward-only
fusion is `4-8%` overall.

## Optimization 2: Fuse Gate + Up Backward dX

Frozen base weights do not get `dW`, but they still need dX:

```text
dX_gate = dGate @ W_gate
dX_up   = dUp   @ W_up
dX      = dX_gate + dX_up
```

Fuse this as:

```text
dX = [dGate, dUp] @ [W_gate; W_up]
```

This is valid for real LoRA-SFT because gradients must still flow through the
frozen base projections to earlier layers and earlier adapters.

Expected benefit:

- one backward AsymGEMM call instead of two
- one routing metadata traversal instead of two
- less dX accumulation overhead
- larger inner dimension can improve GEMM efficiency

This is likely as important as forward gate/up fusion, because LoRA-SFT still
backpropagates through frozen base layers. Expected additional improvement is
around `2-5 ms` in the current profile, depending on how much of the current
time is launch/setup/input traffic versus pure math.

## Optimization 3: AsymGEMM-Aware Epilogue Cleanup

The current path is conceptually:

```text
gate_base = AsymGEMM(...)
gate_lora = LoRA(...)
gate = gate_base + gate_lora

up_base = AsymGEMM(...)
up_lora = LoRA(...)
up = up_base + up_lora

activated = silu(gate) * up
```

After gate/up base fusion, a later optimization is to reduce the elementwise
and memory traffic around this sequence:

```text
activated = silu(gate_base + gate_lora) * (up_base + up_lora)
```

This should be treated as a secondary cleanup, not the main paper contribution.
Generic LoRA epilogue and memory-traffic fusion overlaps with LoRAFusion and
Unsloth. It is only a good contribution if it is tied to the AsymGEMM-specific
MoE block layout, for example:

- consume the fused gate/up AsymGEMM output without materializing extra copies
- reduce casts/adds introduced specifically by CPU-resident AsymGEMM output
- combine base output, adapter output, and `silu(gate) * up` inside the routed
  MoE block path
- preserve the normal LoRA A/B implementation so we are not competing on
  generic adapter GEMM optimization

Expected benefit if implemented only as AsymGEMM-aware cleanup:

- fewer add/cast/activation kernels
- fewer intermediate tensors
- lower memory traffic

This is probably smaller than the base GEMM fusions. A rough target is
`1-3 ms`, or `2-5%` depending on the profile, but it should not be sold as
"we fuse LoRA better than LoRAFusion/Unsloth." It should be sold as removing
overhead created by our fused AsymGEMM MoE block interface.

## Optimization 4: Static Metadata and CUDA Graphs

For fixed-shape one-block SFT profiling, route metadata can be precomputed and
reused when using static routing:

```text
precompute topk, route metadata, expert offsets
reuse them inside the measured loop
```

CUDA graph replay can further reduce CPU launch overhead for fixed-shape torch
and AsymGEMM block measurements.

This does not change the AsymGEMM kernel itself. It reduces benchmark/runtime
bubbles and makes the kernel contribution easier to measure. It should be
reported separately from kernel speedup.

## Paper Claim

The paper should not claim that LoRA removes the AsymGEMM bottleneck. The better
claim is:

> LoRA-SFT freezes base weights, but frozen expert GEMMs remain the dominant
> cost because forward and dX through `W_base` are still required. In MoE
> blocks, gate and up projections share routing and input, creating redundant
> routed AsymGEMM calls. We adapt AsymGEMM from a standalone frozen-weight GEMM
> primitive into a LoRA-SFT-aware MoE block primitive by fusing gate/up forward
> and dX grouped GEMMs, reducing redundant launches, metadata traversal, input
> reads, and dX accumulation.

The negative claim is just as important:

> We are not proposing a generic LoRA training framework, multi-LoRA scheduler,
> QLoRA replacement, or CPU-expert MoE engine. Those are covered by prior
> systems. Our contribution is the AsymGEMM-specific training block: preserving
> LoRA-SFT semantics while making CPU-resident frozen expert GEMMs efficient in
> both forward and backward dX.

## Expected Combined Improvement

Conservative targets:

```text
forward gate/up fusion:  ~4-8%
backward dX fusion:      ~4-8%
epilogue fusion:         ~2-5%
static metadata/graphs:  reduces bubbles; benchmark-dependent
```

Realistic combined target:

```text
~10-20% one-block LoRA-SFT speedup
```

The key message is that the improvement does not come from LoRA eliminating
AsymGEMM. It comes from recognizing that MoE LoRA-SFT exposes repeated,
structured frozen-base GEMMs and making AsymGEMM exploit that structure.

## What Not To Spend Paper Effort On

Avoid work that would be similar to, or weaker than, existing LoRA systems:

- generic LoRA A/B GEMM fusion independent of AsymGEMM
- multi-adapter batching/scheduling
- broad QLoRA/4-bit training claims
- generic CUDA graph speedups as the main result
- end-to-end training framework features already handled by PEFT/TRL/Unsloth
- CPU expert execution, unless used only as a KTransformers comparison point

Useful LoRA-side work is still allowed, but only when it supports the core
AsymGEMM story:

- ensure base weights are frozen and never receive `dW`
- keep LoRA adapter gradients and optimizer state correct
- compare against strong LoRA baselines rather than claiming novelty there
- make adapter work coexist cleanly with fused gate/up AsymGEMM forward and dX



2. b_outer_stride what can be value to thisarg? 
3. "reduction dim is original N, often 128/192, so 64 is needed." epxain this better ot me why? ive an exampole
4. " The kernel indexing change is needed so each group reads the right slice of B.: epxain this bette wiht an exmaple

Answr conciselt tho

