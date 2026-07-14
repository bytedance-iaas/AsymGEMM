# ZeRO3 Offload GPU OOM Sizing For Qwen3 MoE

Headline: with ZeRO3 offload plus activation recompute, Qwen3 GPU OOM is usually
not caused by total model size on GPU. It is caused by the largest live execution
window, dominated first by `batch_size * seq_len * vocab_size` loss/logits memory
and then by `batch_size * seq_len * top_k * hidden_size` routed-MoE memory.

For Qwen3-30B-A3B and Qwen3-235B-A22B, the first long-sequence wall can be
similar when the loss path is unfused because both use `vocab_size = 151936`.
The 235B-class model becomes harder after that because hidden and MoE
intermediate sizes are 2x larger, and one-layer expert weight windows are about
4x larger.

This note is for the case where ZeRO3 offloads parameters and optimizer states
to CPU and activation recompute is enabled. In that setup, CPU capacity decides
whether model state can be stored, but GPU HBM still must fit the largest live
execution window of one training step.

The useful peak model is:

```text
T = batch_size * seq_len
R = T * top_k
b16 = 2 bytes
fp32 = 4 bytes

GPU_peak ~= Loss(T)
          + RoutedMoE(T)
          + Attention(T)
          + GatheredParamsWindow
          + ZeROBuffers
          + AllocatorSlack
```

The dominant terms are explicit:

```text
Loss(T) ~= c_loss * T * vocab_size * fp32
```

`c_loss` is the number of live fp32 full-vocab logits/loss-equivalent buffers.
For the current Qwen3-30B profile, the measured unfused CE/loss path behaves like
about `2.5-2.6` fp32 `[T, vocab]` equivalents at peak. This is often the first
long-context OOM wall because it scales with `batch_size * seq_len * vocab_size`.

```text
RoutedMoE(T) ~= c_x * R * hidden_size * b16
             + c_mid * R * moe_intermediate_size * b16
             + router/index/scatter buffers
```

Here `R = T * top_k`, so routed expert activation memory grows with
`batch_size * seq_len * top_k`, not just `batch_size * seq_len`.

```text
Attention(T) ~= c_attn * T * hidden_size * b16
```

This assumes a memory-efficient attention implementation. A non-memory-efficient
attention path has the bad quadratic term:

```text
AttentionBad(T) ~= batch_size * num_heads * seq_len^2 * b16
```

The gathered parameter window is not the whole model under ZeRO3, but it is not
zero:

```text
GatheredParamsWindow ~= current module params
                     + prefetched params
                     + allgather/reduce buckets
```

For a Qwen3 MoE expert block, a useful upper-bound scale for all expert weights
in one MoE layer is:

```text
MoEExpertWeightsOneLayer ~= num_experts * 3 * hidden_size * moe_intermediate_size * b16
```

ZeRO3 offload mostly removes total model parameter size and optimizer state from
steady GPU residency, but it cannot remove the tensors used by the current GPU
kernels: full-vocab loss/logits, routed MoE buffers, current gathered module
weights, communication buckets, and CUDA workspace/allocator slack.

## Qwen3 30B vs 235B Scaling

Local configs:

```text
Qwen3-30B-A3B:
  vocab_size = 151936
  hidden_size = 2048
  moe_intermediate_size = 768
  num_experts = 128
  top_k = 8
  layers = 48

Qwen3-235B-A22B:
  vocab_size = 151936
  hidden_size = 4096
  moe_intermediate_size = 1536
  num_experts = 128
  top_k = 8
  layers = 94
```

At the same `batch_size` and `seq_len`:

```text
Loss(T):                 about the same, because vocab_size is the same
RoutedMoE(T):            about 2x larger on 235B
Attention(T):            about 2x larger on 235B
MoEExpertWeightsOneLayer: about 4x larger on 235B
Layer count:             mostly affects time with recompute, not peak
```

So if the CE/loss path is unfused or not chunked, Qwen3-30B and Qwen3-235B can
hit a similar long-sequence wall from:

```text
c_loss * batch_size * seq_len * vocab_size * fp32
```

If the loss/logits term is fused or chunked away, the next limits are the larger
235B hidden-size and MoE-intermediate terms:

```text
batch_size * seq_len * top_k * hidden_size
batch_size * seq_len * top_k * moe_intermediate_size
num_experts * hidden_size * moe_intermediate_size
```

## Measured Qwen3-30B Sanity Check

Current smoke profile:

```text
batch_size = 8
seq_len = 8192
T = 65536
top_k = 8
R = 524288
vocab_size = 151936
hidden_size = 2048
moe_intermediate_size = 768
```

Core tensor sizes:

```text
one fp32 logits tensor [T, vocab] = 65536 * 151936 * 4 = 37.09 GiB
one bf16 routed X [R, hidden]     = 524288 * 2048 * 2 = 2.00 GiB
gate+up outputs [R, moe_i] * 2    = 2 * 524288 * 768 * 2 = 1.50 GiB
one dense hidden [T, hidden]      = 65536 * 2048 * 2 = 0.25 GiB
```

The artifact peak closes as:

```text
loss saved/live tensors       43.344 GiB
loss temporary workspace      52.075 GiB
routed expert memory          14.129 GiB
lm_head memory                 6.799 GiB
attention memory               3.851 GiB
norm memory                    4.331 GiB
embed memory                   1.635 GiB
---------------------------------------
GPU allocated peak           126.164 GiB
measured allocated peak      126.165 GiB

allocator reserved slack      20.032 GiB
---------------------------------------
GPU reserved peak            146.196 GiB
measured reserved peak       146.197 GiB
```

For this exact run, the GPU must accommodate about `126 GiB` of real live
allocations and about `146 GiB` of reserved HBM. The main breakpoint is therefore
not total model size on GPU; it is the peak execution window dominated by
full-vocab loss/logits plus routed MoE and allocator/bucket overhead.
