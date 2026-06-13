# ZeRO-3 CPU Offload Parameter Impact Table

| DeepSpeed arg | Main impact | Memory effect | Latency effect | Tuning note |
|---|---|---|---|---|
| `zero_optimization.stage` | Enables ZeRO stage | `3` partitions params, grads, and optimizer states; lower stages use more GPU memory | ZeRO-3 adds gather/offload overhead versus simpler stages | Fixed to `3` for this baseline |
| `zero_optimization.offload_param.device` | Parameter offload | `cpu` greatly reduces persistent GPU parameter residency | Adds CPU<->GPU param fetch/release traffic | Fixed to `cpu` when using ZeRO-3 CPU offload |
| `zero_optimization.offload_optimizer.device` | Optimizer-state offload | `cpu` greatly reduces GPU optimizer-state residency | Can make optimizer step CPU/offload-transfer bound | Fixed to `cpu` when using ZeRO-3 CPU offload |
| `zero_optimization.offload_param.pin_memory` | Param transfer speed | Little direct GPU-memory impact; increases pinned host-memory pressure | Usually faster CPU<->GPU param movement | Usually keep `true` unless host RAM/pinned memory is constrained |
| `zero_optimization.offload_optimizer.pin_memory` | Optimizer transfer speed | Little direct GPU-memory impact; increases pinned host-memory pressure | Usually faster optimizer offload movement | Usually keep `true` for CPU offload |
| `zero_optimization.stage3_max_live_parameters` | Live GPU param cap | Lower values reduce live GPU parameter residency | Lower values can slow training via more frequent refetch/release | One of the highest-impact memory/latency tradeoff knobs |
| `zero_optimization.stage3_max_reuse_distance` | Param reuse retention | Lower values release params sooner and reduce GPU memory | Lower values can increase CPU<->GPU refetch stalls | Tune with `stage3_max_live_parameters` |
| `zero_optimization.stage3_prefetch_bucket_size` | Param prefetch size | Larger buckets increase transient GPU memory | Larger buckets can reduce stalls if transfer overlaps with compute; too large can hurt memory | High-impact latency/GPU-memory tradeoff knob |
| `zero_optimization.stage3_param_persistence_threshold` | Small-param persistence | Higher values keep more small params on GPU | Higher values can avoid repeated fetches for small tensors | Raise for latency, lower for memory |
| `zero_optimization.reduce_bucket_size` | Gradient reduction bucket | Larger buckets use more transient memory | Larger buckets can improve comm efficiency, mostly multi-GPU | Less important for 1-GPU compatibility runs |
| `zero_optimization.overlap_comm` | Comm/offload overlap | May increase temporary memory | Can reduce exposed transfer/comm latency if overlap works | Relevant on one GPU for CPU->GPU prefetch stream behavior; multi-GPU adds collective overlap |
| `zero_optimization.contiguous_gradients` | Gradient layout | Can reduce fragmentation | Usually modest positive impact | Usually keep `true` |
| `zero_optimization.sub_group_size` | Partition/update granularity | Smaller values can reduce local working set | Too small can add scheduling overhead | Secondary tuning knob |

## Current LlamaFactory ZeRO-3 Offload Settings

The LlamaFactory ZeRO-3 offload and SuperOffload configs use CPU offload, not
NVMe offload:

```json
"offload_optimizer": {
  "device": "cpu",
  "pin_memory": true
},
"offload_param": {
  "device": "cpu",
  "pin_memory": true
}
```

The SuperOffload config adds `super_offload: true` and `cpuadam_cores_perc:
0.8` under `offload_optimizer`, but still uses `device: "cpu"` for both
optimizer and parameter offload. There is no `nvme_path` in these configs, so
NVMe offload is disabled.

## Explicit Offload Blocks

For standard DeepSpeed ZeRO-3 configs, the two explicit `offload_*` blocks are:

```text
zero_optimization.offload_param
zero_optimization.offload_optimizer
```

There is no separate user-facing `offload_gradients` block in the normal
ZeRO-3 config. Gradients are partitioned by ZeRO and, when optimizer offload is
enabled, the gradient partition is handled through the optimizer/offload path.
So `offload_optimizer` is not the same thing as gradients, but it is the
offload setting that affects the optimizer-side gradient/update path.

Other offload behavior is configured inside those same blocks. For example,
NVMe offload is selected by setting `device: "nvme"` plus `nvme_path` under
`offload_param` and/or `offload_optimizer`; it is not a third top-level
`offload_*` block. DeepSpeed also has internal offload state names for buffers
such as params, gradients, and optimizer states, but those are implementation
details rather than separate config keys in the LlamaFactory ZeRO-3 JSON.

## One-GPU ZeRO-3 CPU Offload Step

For a one-GPU ZeRO-3 CPU-offload run, ZeRO-3 still uses the normal
partition/fetch/release/update machinery, but with `world_size=1` it does not
gain cross-GPU sharding. Each ZeRO "partition" is effectively the whole local
tensor chunk. The main GPU-memory saving comes from moving parameter storage,
gradients, fp32 master weights, and Adam states out of GPU memory.

### Runtime Storage

Important DeepSpeed runtime storage objects:

| Object | Meaning | Main location with current config |
|---|---|---|
| `param.data` | Full tensor used by PyTorch module compute | GPU only while the param is gathered for compute |
| `param.ds_tensor` | ZeRO low-precision partition/offload storage | Usually CPU pinned memory for `offload_param=cpu` |
| `fp16_partitioned_groups_flat` | Flat low-precision trainable param chunks grouped by `sub_group_size` | CPU |
| `fp32_partitioned_groups_flat` | CPUAdam fp32 master param chunks | CPU |
| `fp32_partitioned_groups_flat[i].grad` | CPUAdam fp32 gradient chunks | CPU |
| `exp_avg`, `exp_avg_sq` | Adam first/second moment optimizer states | CPU |
| Activations, logits, temporary workspaces | Forward/backward live tensors | GPU |

The low-precision trainable storage and fp32 master storage are separate.
`fp16_partitioned_groups_flat` and `fp32_partitioned_groups_flat` are
DeepSpeed's flat tensors for update chunks, not user model layers. CPUAdam
updates the fp32 chunk; DeepSpeed then copies it back into the low-precision
chunk so future forwards fetch updated weights. Persistent small params can
keep full `param.data` resident on GPU, and frozen/non-trainable params are
still ZeRO-managed even when they are not in optimizer flat groups.

Implementation anchors in the local DeepSpeed tree:

| Component | Main code path |
|---|---|
| Module hooks | `runtime/zero/parameter_offload.py` |
| Trace, parameter queue, prefetch, release | `runtime/zero/partitioned_param_coordinator.py` |
| One-GPU local param gather and partition/free | `runtime/zero/partition_parameters.py` |
| Grad partitioning, CPUAdam step, fp32 writeback | `runtime/zero/stage3.py` |

Big Picture:

- ZeRO-3 offloads model parameter storage, gradients, fp32 master weights, and
  Adam states.

- It does not offload activations, logits, attention workspaces, MLP workspaces,
  or loss tensors.

- On one GPU, there is no useful cross-GPU sharding, but DeepSpeed still uses the
  ZeRO-3 fetch/release/update machinery.

- Main memory saving comes from keeping params and optimizer state on CPU most of
  the time.

1. Initialization

- DeepSpeed wraps every parameter.
    - Adds a parameter id.
    - Adds offload/partition storage: param.ds_tensor.
    - Tracks whether the full GPU param is available or not.

- It builds low-precision param chunks.
    - Usually fp16/bf16.
    - With offload_param=cpu, these chunks live on CPU.
    - param.ds_tensor points into these CPU chunks.

- It builds optimizer-side CPU state.
    - fp32_partitioned_groups_flat[i]: fp32 master weight chunk.
    - fp32_partitioned_groups_flat[i].grad: fp32 grad chunk.
    - exp_avg: Adam first moment.
    - exp_avg_sq: Adam second moment.

- Small params may be marked persistent.
    - Example: norm weights are small, so they often stay available.
    - This is based on size thresholds, not module semantics.

2. Trace And Queue

- On early steps, DeepSpeed records module execution order.
    - Forward/backward hooks see which modules run.
    - DeepSpeed turns that module order into a parameter fetch order.

- Later steps reuse that order.
    - It keeps a queue of upcoming parameters.
    - This queue drives prefetching.

3. Forward Per Module

- Before module compute:
    - DeepSpeed checks needed params.
    - If a param is not on GPU, it copies the param from CPU storage to GPU.
    - On one GPU, this is just CPU -> GPU copy, not real multi-rank all-gather.

- Then:
    - Current params become normal GPU param.data.
    - Module forward runs normally on GPU.
    - Activations/hidden states stay on GPU.

- Prefetch:
    - DeepSpeed may start fetching params for upcoming modules.
    - Limited by stage3_prefetch_bucket_size and stage3_max_live_parameters.
    - If overlap_comm=false, fetch/prefetch is more serialized with compute.

- After module compute:
    - DeepSpeed may release the full GPU param.
    - param.data becomes empty.
    - CPU offload storage remains the source copy.
    - No weight update happens here.

4. Loss

- Loss/logits still allocate GPU memory.
- ZeRO-3 does not save activation/loss workspace memory.
- Large vocab logits or long sequence loss can still OOM even with param offload.

5. Backward Per Module

- Before backward compute:
    - If forward released a param, DeepSpeed fetches it again to GPU.
    - With activation checkpointing, recompute can also trigger fetches.

- Backward compute:
    - GPU computes param.grad.

- Gradient handling:
    - DeepSpeed copies param.grad into its gradient buffer.
    - With one GPU, there is no real cross-GPU reduction.
    - The result is the local gradient chunk.

- At gradient accumulation boundary:
    - DeepSpeed copies the gradient into CPUAdam’s grad buffer:
      fp32_partitioned_groups_flat[i].grad.

    - Then it clears normal GPU param.grad.
    - Params may be released from GPU again.

6. Optimizer Step

- DeepSpeed steps subgroup by subgroup.
    - Not layer-by-layer.
    - Not whole model at once.
    - It uses flat parameter chunks.

- For each subgroup, CPUAdam reads:
    - fp32_partitioned_groups_flat[i]
    - fp32_partitioned_groups_flat[i].grad
    - exp_avg
    - exp_avg_sq

- CPUAdam updates fp32 master weights on CPU.
    - Adam is elementwise math.
    - No matrix multiplication happens in the optimizer.

- DeepSpeed writes updated weights back:
    - Updated fp32 master chunk is cast/copied into low-precision param chunk.
    - That low-precision CPU chunk is what future forwards fetch.

7. Next Forward

- Next forward fetches updated low-precision weights from CPU to GPU.
- GPU compute uses those updated weights.
- Cycle repeats.

NVMe Case

- NVMe is only storage.
- It does not compute.
- If params or optimizer states are on NVMe:
    - DeepSpeed swaps them NVMe -> CPU buffer.
    - CPUAdam computes on CPU.
    - Needed params are copied CPU -> GPU for forward/backward.
    

## Meaning of `auto`

In the Hugging Face/Transformers DeepSpeed integration, `"auto"` is resolved
once during trainer setup, before DeepSpeed initializes. It is not dynamic and
does not change mid-training.

For ZeRO-3, Transformers derives these values from model hidden size `H`:

```text
reduce_bucket_size = H * H
stage3_prefetch_bucket_size = int(0.9 * H * H)
stage3_param_persistence_threshold = 10 * H
```

`H` is taken from `model.config.hidden_size`, or from the maximum entry in
`hidden_sizes`, or from the equivalent `text_config` fields for multimodal/text
subconfigs.

## LoRA SFT Implication

For LoRA SFT, frozen base weights are not optimized by Adam, so CPU optimizer
offload applies to trainable LoRA weights only. However, `offload_param: cpu`
can still move frozen base parameters through the ZeRO-3 CPU offload path
because those parameters are needed for forward/backward compute.

When the config uses `"auto"`, Transformers resolves the persistence threshold
to `10 * hidden_size`. The threshold applies to all ZeRO-managed params, subject
to the model persistence cap. A persistent small param is kept available across
module releases, and persistent params are gathered again after optimizer step.
Lowering `stage3_param_persistence_threshold` makes ZeRO-3 more aggressive
about release/refetch; raising it keeps more full `param.data` resident on GPU
and can reduce fetch overhead.

## Practical Memory Notes

If GPU OOM happens with a large reserved-but-unallocated gap, allocator
fragmentation may be part of the issue. In that case,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` can help by reducing
fragmentation from changing allocation sizes. It does not reduce live tensor
memory, and it can keep larger segments reserved, so keep it as a per-run knob
rather than a permanent global default.

If GPU OOM happens with very little reserved-but-unallocated memory, the issue
is live GPU-memory pressure. In that case, changing allocator settings will not be
enough. The main options are to reduce batch size, sequence length, activation
memory, loss/logit workspace, or make ZeRO-3 keep fewer parameters resident on
GPU.

High-impact ZeRO-3 GPU-memory tuning knobs:

| Knob | Lower-memory direction | Main tradeoff |
|---|---|---|
| `stage3_max_live_parameters` | Lower value | Fewer live GPU params, more refetches |
| `stage3_max_reuse_distance` | Lower value | Earlier eviction, more refetch stalls |
| `stage3_prefetch_bucket_size` | Lower value | Less transient prefetch memory, less overlap |
| `stage3_param_persistence_threshold` | Lower value, even `0` for aggressive testing | Fewer persistent small params, more fetch overhead |
| `reduce_bucket_size` | Lower value | Smaller comm/grad buckets, possible slower communication |
| `sub_group_size` | Lower value | Smaller local working set, more scheduling overhead |

`overlap_comm: false` is already the lower-memory choice for this profile.
`contiguous_gradients: true` usually reduces fragmentation and should normally
stay enabled unless there is a specific reason to test the opposite.

NVMe offload can reduce CPU memory pressure and extend offload capacity, but it
is much slower than CPU RAM offload and should only be tested on fast local
NVMe. Avoid using a shared filesystem or NFS path for DeepSpeed NVMe offload.
