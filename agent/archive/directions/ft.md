# Modular AsymGEMM Full Fine-Tuning Design

## Goal

Add a separate full fine-tuning path for CPU-resident AsymGEMM weights without disturbing the current LoRA SFT design.

The target workload is memory-first full fine-tuning of large dense or MoE expert projections where:

- the served/fetched weight lives in CPU pinned memory and is consumed by AsymGEMM;
- the optimizer master weight and optimizer state live on CPU;
- the GPU only holds activations, output gradients, small temporary `dW` blocks, and normal non-offloaded model parameters;
- each backward pass computes `dX`, computes blockwise `dW`, updates CPU weights, then refreshes the pinned fetch copy for the next forward.

This is worth prototyping for expert-heavy MoE and FFN projections. It is not expected to beat normal GPU full fine-tuning on speed. The purpose is to make full fine-tuning fit under much lower HBM pressure.

## High-Level Summary

The full-FT design adds a new offloaded-training path beside the existing LoRA SFT path.

At a high level:

```text
forward:
  GPU activation reads CPU pinned W_fetch through AsymGEMM

backward:
  compute dX first with the old W_fetch
  compute dW in output-row blocks on GPU
  copy each dW block to CPU
  CPU optimizer updates W_master
  refresh the matching W_fetch block

next forward:
  read the refreshed W_fetch
```

There are always two weight concepts:

```text
W_master
  CPU pageable
  true trainable weight
  owns optimizer state
  saved in checkpoints

W_fetch
  CPU pinned
  BF16 in V1
  AsymGEMM-serving copy
  refreshed from W_master
  disposable cache, not optimizer truth
```

The key design decision is that offloaded full-FT weights are not `nn.Parameter`s. They are owned by a dedicated full-FT subsystem and updated by a blockwise CPU optimizer. Normal PyTorch optimizers only see normal model parameters such as router, attention, layernorm, embeddings, and any projections not converted to full-FT.

## High-Level Target

The first useful target is:

```text
routed MoE expert projections only
gate/up/down expert weights offloaded
BF16 pinned fetch weights
FP32 CPU master weights
CPU AdamW or SGD
gradient_accumulation_steps = 1
block_n = 1024
single fetch copy with strict finalize barrier
```

This target gives the best chance of being worthwhile because inactive experts are skipped. Dense all-layer full fine-tuning is possible as a correctness path, but it is less attractive because every dense weight pays the `dW` D2H and CPU optimizer cost every step.

## High-Level Success Criteria

The implementation is useful if:

- HBM usage drops by roughly the offloaded weight plus optimizer-state footprint;
- small dense and MoE tests match PyTorch full fine-tuning;
- next forward always sees fresh refreshed weights after `finalize_asym_full_ft(model)`;
- only active MoE experts are updated;
- LoRA SFT behavior stays unchanged;
- profiling clearly shows where the time goes: `dW` GEMM, D2H copy, CPU optimizer, refresh, and scheduler wait.

The expected performance result is:

```text
slower than GPU full fine-tuning
much lower HBM use
potentially viable for MoE expert-heavy workloads
likely bandwidth-limited for dense all-layer FT over PCIe
```

## Non-Goals

- Do not modify the current LoRA module contract in `asym_gemm/training/lora.py`.
- Do not turn `HostWeight` into a trainable `nn.Parameter`.
- Do not make existing `AsymFrozenLinear` produce normal `.grad` tensors.
- Do not require PyTorch optimizers to own offloaded full-FT weights as normal parameters.
- Do not start with quantized full-FT fetch weights. Build BF16 first, then add FP8/FP4 refresh once correctness is stable.
- Do not hide the optimizer/update lifecycle inside a Python loop that allocates new tensors per block.

## Existing Facts To Preserve

Current SFT is frozen-base LoRA:

- `HostWeight` is CPU-only, detached, non-parameter storage. It intentionally refuses `.cuda()` and CUDA `.to(...)`.
- `AsymFrozenLinear` stores the base weight as `HostWeight`, then only computes `grad_x` and optional `grad_bias` in backward.
- `AsymFrozenLinearFunction.backward` returns `None` for the host weight gradient.
- The existing LoRA path trains only adapter tensors while the base weight remains frozen.
- The grouped AsymGEMM BF16 path consumes CPU weight tensors through TMA tensor maps and only requires `offsets` and `experts` to be CUDA tensors.
- The SM90 BF16 path uses `block_m=64`, `block_n=64`, and `block_k=512` for normal B layout. Transposed B uses `block_k=64`.

These are good properties. Full fine-tuning should be a sibling path, not a mutation of the frozen path.

## High-Level Architecture

Add a new full-FT package under `asym_gemm/training/full_ft/`. This package owns all offloaded full-FT behavior:

```text
asym_gemm/training/full_ft/
  __init__.py
  config.py         public dataclasses and validation
  weight.py          CPU master/fetch weight ownership and refresh
  optimizer.py       CPU blockwise SGD/AdamW update kernels
  scheduler.py       block queues, streams, CPU worker pool, version barriers
  linear.py          dense Asym full-FT linear autograd function/module
  grouped.py         grouped/MoE Asym full-FT autograd function/module
  injection.py       optional target-module replacement helpers
  state.py           state_dict save/load helpers
  stats.py           counters and profiler ranges
```

The module boundaries are:

```text
config.py
  public knobs and validation

weight.py
  CPU master/fetch storage, refresh, checkpoint state

optimizer.py
  blockwise CPU SGD/AdamW math

scheduler.py
  D2H staging, CUDA events, CPU worker pool, finalize barrier

linear.py
  dense full-FT autograd wrapper

grouped.py
  grouped/MoE expert full-FT autograd wrapper

injection.py
  optional model replacement helpers

state.py
  save/load helpers for offloaded weights and optimizer state

stats.py
  memory, timing, and operation counters
```

Existing files should only get narrow integration points:

```text
asym_gemm/training/__init__.py      export new full-FT APIs
asym_gemm/training/moe.py           optional backend dispatch to full-FT MoE expert path
scripts/profile_lora.py             only if a new benchmark mode is added; do not mix into LoRA mode
tests/...                           add full-FT tests beside existing LoRA tests
```

The current LoRA files stay as they are except for importing/exporting shared helpers if truly needed.

## High-Level Execution Contract

The training loop must explicitly finalize offloaded updates:

```python
loss = model(batch)
loss.backward()

finalize_asym_full_ft(model)

torch_optimizer.step()
torch_optimizer.zero_grad(set_to_none=True)
zero_asym_full_ft_grad_accumulators(model)
```

Ordering matters:

- `loss.backward()` computes `dX`, creates `dW` blocks, and queues or performs CPU updates.
- `finalize_asym_full_ft(model)` waits for all D2H copies, CPU optimizer tasks, and fetch refreshes.
- `torch_optimizer.step()` updates only normal PyTorch parameters.
- The next forward must not start until full-FT finalize is complete.

## High-Level Rollout

Build the feature in this order:

1. Scaffold the new `full_ft` package and exports.
2. Implement CPU weight ownership with `W_master` and `W_fetch`.
3. Implement blockwise SGD and match PyTorch on tiny CPU tests.
4. Implement blockwise AdamW and verify step counting is once per logical step, not once per block.
5. Implement dense `AsymFullFTLinear` with synchronous block updates.
6. Move dense updates behind the scheduler API.
7. Add async D2H staging and CPU worker pool.
8. Implement grouped/MoE `AsymGroupedFullFTLinear`.
9. Integrate a separate MoE full-FT backend or injection hook.
10. Add checkpoint save/load and profiling.
11. Consider FP8/FP4 fetch refresh only after BF16 is correct.

## Core Objects

### `AsymFullFTWeight`

Owns all CPU-side copies for one trainable offloaded matrix.

```python
class AsymFullFTWeight:
    name: str
    shape: tuple[int, int]          # [out_features, in_features]
    dtype_fetch: torch.dtype        # bf16 for V1
    dtype_master: torch.dtype       # fp32 or bf16

    master: torch.Tensor            # CPU pageable, true optimizer weight
    fetch: torch.Tensor             # CPU pinned, AsymGEMM-facing weight
    bias_master: torch.Tensor | None
    bias_fetch: torch.Tensor | None

    adam_m: torch.Tensor | None     # CPU pageable
    adam_v: torch.Tensor | None     # CPU pageable
    step: int

    version: int
    in_flight_read_event: torch.cuda.Event | None
    dirty_blocks: BlockSet
```

Placement rule:

- `fetch` is pinned because AsymGEMM reads it from the GPU.
- `master`, `adam_m`, and `adam_v` are not pinned. They are CPU-only optimizer storage.
- Small D2H staging buffers are pinned and reused.
- Backward dX reads the same pinned `fetch` tensor with `transpose_b=True`.

### `BlockwiseCPUOptimizer`

Owns the CPU math for updating a block.

```python
class BlockwiseCPUOptimizer:
    def update_block(
        self,
        weight: AsymFullFTWeight,
        grad_cpu: torch.Tensor,
        row_start: int,
        row_end: int,
        *,
        col_start: int = 0,
        col_end: int | None = None,
        scale: float = 1.0,
    ) -> None:
        ...
```

V1 should support:

- SGD with weight decay for simplest correctness.
- AdamW CPU update after SGD correctness is proven.
- FP32 master weights even if fetch is BF16.

Do not use a standard `torch.optim.Optimizer` for these offloaded weights in V1. The offloaded optimizer needs block-level updates and refreshes, not full `.grad` tensors.

### `AsymFullFTUpdateScheduler`

Owns concurrency and ordering.

Responsibilities:

- preallocate GPU `dW` buffers;
- preallocate pinned CPU grad staging buffers;
- run a CPU thread pool for optimizer updates;
- record CUDA events that prove AsymGEMM has finished reading an old fetch block;
- prevent the next forward from reading partially refreshed weights;
- expose `finalize_backward()` to wait for all queued CPU updates and refreshes.

Conceptually copy the useful KTransformers pattern:

- wrapper owns CPU weights and worker threads;
- forward/backward are submitted through a wrapper boundary;
- an explicit post-backward or post-step hook updates CPU-resident trainable storage;
- cache depth and preallocated buffers are configuration knobs;
- active experts are processed, inactive experts are skipped.

The full-FT version should expose:

```python
class AsymFullFTUpdateScheduler:
    def begin_forward(self, weight: AsymFullFTWeight) -> None: ...
    def mark_fetch_read(self, weight: AsymFullFTWeight, event: torch.cuda.Event) -> None: ...
    def enqueue_update(self, task: GradBlockTask) -> None: ...
    def finalize_backward(self) -> None: ...
    def zero_grad_accumulators(self) -> None: ...
```

### `AsymFullFTLinear`

Dense linear replacement:

```python
class AsymFullFTLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        *,
        optimizer_config: FullFTOptimizerConfig,
        backend: str = "asym",
        precision: str = "bf16",
        pin_memory: bool = True,
        block_n: int = 1024,
        block_k: int | None = None,
        scheduler: AsymFullFTUpdateScheduler | None = None,
    ) -> None:
        ...
```

Forward calls the existing AsymGEMM dispatch with `weight.fetch`.

Backward computes:

- `grad_x`;
- optional `grad_bias`;
- blockwise `dW`;
- CPU optimizer update;
- refresh of `weight.fetch`.

It returns no normal gradient for the offloaded weight object.

### `AsymGroupedFullFTLinear`

Grouped/MoE replacement:

```python
class AsymGroupedFullFTLinear(nn.Module):
    def __init__(
        self,
        expert_weight: torch.Tensor,       # [num_experts, out, in]
        *,
        optimizer_config: FullFTOptimizerConfig,
        backend: str = "asym",
        precision: str = "bf16",
        block_n: int = 1024,
        scheduler: AsymFullFTUpdateScheduler | None = None,
    ) -> None:
        ...
```

This is the important path for viability. It should:

- reuse existing sorted/grouped token layout, `offsets`, and `experts`;
- update only experts that appear in the current batch;
- skip experts with zero routed tokens;
- share one scheduler across gate/up/down/downstream projection modules in a layer if possible.

## Exact Dense Backward Algorithm

For a linear layer:

```text
Y = X @ W.T + b
X: [M, K] on GPU
W_fetch: [N, K] on CPU pinned BF16
W_master: [N, K] on CPU pageable FP32 or BF16
dY: [M, N] on GPU
```

Backward must use the same weight version that forward used.

Algorithm:

1. Flatten saved activation to `X2d = X.reshape(M, K).contiguous()`.
2. Flatten output gradient to `dY2d = dY.reshape(M, N).contiguous()`.
3. Compute `grad_x` first:

```text
grad_x = dY2d @ W_fetch_old
```

Use AsymGEMM with `transpose_b=True` or the existing dispatch path. Record a CUDA event after the `grad_x` kernel so CPU refresh is not allowed to overwrite the old fetch block before the read completes.

4. Compute `grad_bias` if bias is trainable:

```text
grad_bias = sum(dY2d, dim=0)
```

5. Loop over output-row blocks:

```text
for n0 in range(0, N, block_n):
    n1 = min(n0 + block_n, N)
    dY_block = dY2d[:, n0:n1]
    dW_gpu = dY_block.T @ X2d
    copy dW_gpu -> pinned CPU grad staging buffer
    enqueue CPU optimizer update for W_master[n0:n1, :]
    refresh W_fetch[n0:n1, :] from W_master[n0:n1, :]
```

6. Return `grad_x.reshape(input_shape)`, `None` for the offloaded weight object, and `grad_bias` only if the bias is exposed as a normal trainable tensor.

Critical invariant:

```text
No block of W_fetch may be updated until every grad_x read of that block for the current backward is complete.
```

The simplest V1 implementation computes all `grad_x` first, records one event, then allows all block updates after that event. Later versions can use per-block events.

## Exact Grouped/MoE Backward Algorithm

For grouped expert linear:

```text
X: [total_tokens_for_layer, K] on GPU
W_fetch: [E, N, K] on CPU pinned BF16
offsets: GPU int tensor with token ranges per active group
experts: GPU int tensor with physical expert ids
dY: [total_tokens_for_layer, N] on GPU
```

Forward:

1. Sort or pack routed tokens exactly as the existing grouped path does.
2. Call grouped AsymGEMM with `W_fetch`.
3. Save `X`, `offsets`, `experts`, active expert ids, and weight version.

Backward:

1. Compute `grad_x` first using the old fetch weight:

```text
grad_x_grouped = grouped_asymgemm(dY, W_fetch_old, offsets, experts, transpose_b=True)
```

2. Record an event that protects old fetch blocks from CPU overwrite.
3. For each active expert `e`:

```text
token rows = offsets range for expert e
X_e = X[token rows, :]
dY_e = dY[token rows, :]
for n0:n1 over output rows:
    dW_e_block = dY_e[:, n0:n1].T @ X_e
    D2H copy into pinned staging
    CPU optimizer update W_master[e, n0:n1, :]
    refresh W_fetch[e, n0:n1, :]
```

4. If multiple top-k routes write into the same physical expert, the packed representation must already combine all rows for that expert. If there are multiple ranges for the same expert, accumulate or process them into the same block update before applying AdamW.
5. Return `grad_x` scattered back to the original token order if the forward packed tokens.

Important MoE optimization:

```text
Only active experts should produce dW or optimizer work.
```

Inactive experts remain untouched and do not consume D2H bandwidth.

## After Autograd: What Happens

The training loop must call the full-FT scheduler explicitly.

Recommended loop shape:

```python
loss = model(batch)
loss.backward()

full_ft_scheduler.finalize_backward()

torch_optimizer.step()      # only normal GPU params, router, attention, layernorm, etc.
torch_optimizer.zero_grad(set_to_none=True)
full_ft_scheduler.zero_grad_accumulators()
```

`loss.backward()` schedules or performs the offloaded block updates. `finalize_backward()` waits for:

- GPU `dW` block kernels;
- D2H copies;
- CPU optimizer tasks;
- refresh writes into pinned `W_fetch`;
- version increments.

The next forward is not allowed to start until `finalize_backward()` has completed, unless a later double-buffered implementation keeps two fetch copies and versions them.

## Gradient Accumulation

V1 should use `gradient_accumulation_steps=1`.

If accumulation is required:

1. Do not update `W_master` after every microbatch.
2. Compute `grad_x` for each microbatch against the unchanged old weight.
3. Accumulate blockwise `dW` on CPU or in a preallocated CPU accumulator.
4. At the accumulation boundary, apply one optimizer update and refresh `W_fetch`.

Updating after each microbatch changes the math and no longer matches standard full fine-tuning.

Memory-conscious accumulation options:

- CPU accumulator per active block, flushed at boundary.
- Recompute `dW` from saved or recomputed activations at boundary.
- Restrict V1 to accumulation step 1, then add accumulation after correctness is proven.

## Weight Freshness And Refresh

Keep two concepts separate:

```text
W_master: true trainable optimizer weight, CPU pageable, usually FP32
W_fetch: AsymGEMM-serving weight, CPU pinned, BF16 in V1
```

Refresh means:

```text
W_fetch[block] = cast_or_quantize(W_master[block])
```

Refresh rules:

- Refresh only after the CPU optimizer updates `W_master`.
- Refresh only after CUDA has finished all reads of the old `W_fetch` block.
- Refresh whole aligned row blocks when possible.
- If using quantized fetch weights later, refresh values and scales together as an atomic logical unit.
- Increment `weight.version` after all blocks for a layer are refreshed.

For V1, prefer one fetch copy and strict barriers. Later, add double-buffered fetch pages:

```text
forward reads fetch version v
CPU refreshes inactive fetch version v+1
next forward swaps to v+1 after all blocks are ready
```

Double buffering costs more pinned CPU memory but reduces CPU/GPU idle time.

## Block Size Policy

Start with row blocking over `N`:

```text
block_n = 1024 or 2048 for large projections
block_k = full K for V1
```

Why row blocking first:

- AsymGEMM kernels are already B/weight-tile oriented.
- Output-row blocks align naturally with `dY[:, n0:n1].T @ X`.
- CPU optimizer state is contiguous for `W[n0:n1, :]`.
- Refreshing `W_fetch[n0:n1, :]` is simple.

Use alignment constraints:

- BF16 direct path should keep dimensions aligned for AsymGEMM eligibility.
- Quantized path later should prefer 128-aligned dimensions and block boundaries.
- Pad tails or use a small fallback path for tail rows rather than complicating the hot path.

If `block_n * K` is still too large for the temporary `dW_gpu`, add `K` chunking:

```text
for n0:n1:
    for k0:k1:
        dW_gpu_tile = dY_block.T @ X[:, k0:k1]
        update W_master[n0:n1, k0:k1]
        refresh W_fetch[n0:n1, k0:k1]
```

K chunking lowers peak memory but increases CPU optimizer overhead and may hurt AdamW locality. It should be a fallback, not the default.

## Efficient Scheduling

Use three lanes:

```text
CUDA compute stream:       grad_x and dW GEMMs
CUDA copy stream:          D2H dW block copies
CPU optimizer pool:        AdamW/SGD update and fetch refresh
```

Pipeline:

1. Compute `dW_gpu` for block `i` on compute stream.
2. Copy block `i` to pinned CPU staging on copy stream.
3. While CPU updates block `i`, GPU computes block `i+1`.
4. Reuse staging buffer only after CPU update completes.

Preallocate:

- 2 or 3 GPU `dW` block buffers;
- 2 or 3 pinned CPU grad staging buffers;
- per-layer task objects;
- CPU thread pool workers;
- optional per-block CPU accumulation buffers.

Avoid:

- allocating tensors inside the block loop;
- converting entire weights per step;
- creating a Python future per tiny tile;
- updating inactive MoE experts;
- using pinned memory for full optimizer state.

## KTransformers-Inspired Ideas To Copy

Copy concepts, not implementation details:

- Put CPU-resident trainable expert ownership behind one wrapper object.
- Use explicit submit/sync/finalize lifecycle instead of relying on hidden PyTorch parameter grads.
- Keep a CPU thread pool close to the weight owner.
- Expose `threadpool_count`, worker count, and cache depth as tunables.
- Process active routed experts only.
- Keep update hooks explicit, like the current KT LoRA path calls an update hook after optimizer work.
- Preload and reuse expert weight layouts rather than rebuilding metadata each step.
- Keep CPU/GPU boundary copies coarse enough to amortize overhead.

For AsymGEMM full-FT, the analogous wrapper is:

```text
AsymFullFTMoEWrapper
  owns gate/up/down AsymFullFTWeight objects
  owns grouped forward/backward calls
  owns blockwise CPU optimizer
  owns refresh/version lifecycle
```

## Public API

Dense/manual:

```python
from asym_gemm.training.full_ft import AsymFullFTLinear, FullFTOptimizerConfig

layer = AsymFullFTLinear.from_linear(
    linear,
    optimizer_config=FullFTOptimizerConfig(
        optimizer="adamw",
        lr=1e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        master_dtype=torch.float32,
    ),
    precision="bf16",
    block_n=1024,
)
```

Model injection:

```python
from asym_gemm.training.full_ft import add_asym_full_ft, finalize_asym_full_ft

report = add_asym_full_ft(
    model,
    target_modules=("gate_proj", "up_proj", "down_proj"),
    optimizer_config=optimizer_config,
    precision="bf16",
    block_n=1024,
    strict=True,
)
```

Training loop:

```python
loss.backward()
finalize_asym_full_ft(model)
torch_optimizer.step()
torch_optimizer.zero_grad(set_to_none=True)
```

The report should include:

- replaced module names;
- offloaded parameter count;
- CPU master bytes;
- CPU optimizer-state bytes;
- pinned fetch bytes;
- pinned staging bytes;
- trainable normal GPU parameter count;
- unsupported modules skipped with reasons.

## Optimizer Ownership

Split model parameters into two groups:

1. Normal PyTorch parameters:
   - router;
   - attention if not offloaded;
   - layernorm;
   - embeddings/head if not offloaded;
   - biases if left as normal parameters.

2. Offloaded full-FT weights:
   - owned by `AsymFullFTWeight`;
   - updated by `BlockwiseCPUOptimizer`;
   - not passed to `torch.optim`.

Provide helper:

```python
def normal_torch_parameters(model: nn.Module) -> list[nn.Parameter]:
    ...
```

This prevents accidental double-updates.

## State Dict

State dict must save the trainable master weights, not only the fetch copy.

For each offloaded full-FT layer, save:

```text
<prefix>.full_ft_master_weight
<prefix>.full_ft_fetch_dtype
<prefix>.full_ft_optimizer.step
<prefix>.full_ft_optimizer.m
<prefix>.full_ft_optimizer.v
<prefix>.bias_master             if owned by full-FT path
```

Loading:

1. Load master weight to CPU.
2. Allocate/restore optimizer state on CPU.
3. Recreate pinned fetch.
4. Refresh fetch from master.
5. Reset version and clear in-flight events.

Do not save temporary pinned staging buffers.

## Profiling Counters

Add stats for:

- full-FT forward AsymGEMM calls;
- full-FT `dX` AsymGEMM calls;
- `dW` GEMM calls;
- D2H grad bytes;
- CPU optimizer bytes touched;
- refresh bytes;
- active expert count;
- skipped inactive expert count;
- pinned fetch bytes;
- CPU master bytes;
- CPU optimizer-state bytes;
- scheduler wait time;
- CPU update time;
- copy stream time.

Profiler range names:

```text
forward.full_ft_asymgemm
backward.full_ft_dx_asymgemm
backward.full_ft_dw_gemm
backward.full_ft_dw_d2h
optimizer.full_ft_cpu_update
optimizer.full_ft_refresh_fetch
optimizer.full_ft_finalize
```

## Implementation Phases

### Phase 1: BF16 Dense Correctness

Implement:

- `AsymFullFTWeight`;
- SGD CPU block update;
- `AsymFullFTLinear`;
- strict barrier: compute all `grad_x`, then update blocks;
- no gradient accumulation;
- no quantized fetch;
- one layer at a time.

Tests:

- compare one training step against `nn.Linear` with SGD on tiny shapes;
- compare loss decrease over several steps;
- assert no `.grad` is created for offloaded weight;
- assert `W_fetch` changes after `finalize_backward()`;
- assert current LoRA tests still pass.

### Phase 2: AdamW CPU Optimizer

Add:

- FP32 master;
- AdamW state;
- blockwise update;
- state dict save/load;
- CPU thread pool.

Tests:

- compare one AdamW step with PyTorch AdamW for small shapes;
- compare block sizes produce the same result within tolerance;
- verify weight decay and bias correction.

### Phase 3: Grouped MoE Full-FT

Implement:

- `AsymGroupedFullFTLinear`;
- active expert detection;
- per-expert blockwise `dW`;
- skip inactive experts;
- layer wrapper for gate/up/down expert projections.

Tests:

- compare against tiny PyTorch MoE full fine-tuning;
- test top-k routing with repeated expert ids;
- test inactive experts remain unchanged;
- test offsets/expert ordering.

### Phase 4: Scheduler Pipelining

Add:

- copy stream;
- ring buffers;
- CPU worker pool;
- double-buffered staging;
- configurable queue depth;
- profiler stats.

Benchmark:

- block_n sweep: 256, 512, 1024, 2048;
- CPU thread sweep;
- active expert count sweep;
- PCIe vs NVLink-C2C if available.

### Phase 5: Quantized Fetch Refresh

Only after BF16 is correct:

- keep `W_master` FP32/BF16;
- refresh quantized `W_fetch` block values and scales;
- preserve a single pinned host-weight fetch copy;
- test forward/backward tolerance separately from optimizer correctness.

Quantized full-FT fetch saves CPU bandwidth and pinned bytes, but it makes refresh and dX cache consistency harder. It should not be in V1.

## Correctness Rules

- `dX` must be computed with the same weight version used by forward.
- `W_fetch` must not be modified while any in-flight AsymGEMM may read it.
- If gradient accumulation is enabled, update only at the accumulation boundary.
- Offloaded weights must not appear in PyTorch optimizer parameter groups.
- `W_master` is the source of truth.
- `W_fetch` is disposable serving/cache state.
- Refresh must be complete before the next forward reads a new version.
- Existing LoRA modules must keep the same behavior and tests.

## Viability Criteria

Proceed if the prototype shows:

- HBM savings close to full offloaded weight plus optimizer state size;
- step correctness matching PyTorch on small cases;
- acceptable slowdown for target MoE expert workloads;
- no LoRA regression;
- clear profiling evidence that CPU update and D2H copies are the main costs.

Stop or narrow scope if:

- dense all-layer full-FT over PCIe is dominated by `dW` D2H bandwidth;
- block scheduling overhead dominates useful compute;
- CPU AdamW cannot keep up even for active experts;
- pinned fetch memory becomes the limiting resource.

Best first target:

```text
BF16 fetch
FP32 CPU master
CPU AdamW
gradient_accumulation_steps = 1
routed MoE expert gate/up/down projections only
block_n = 1024
single fetch copy with strict finalize barrier
```

That target gives the clearest signal while keeping the current LoRA design untouched.

## Detailed Implementation Plan

This section is the implementation checklist. Treat it as the source of truth for what should exist after the work is done.

### Final File Layout

Create these new files:

```text
asym_gemm/training/full_ft/
  __init__.py
  config.py
  weight.py
  optimizer.py
  scheduler.py
  linear.py
  grouped.py
  injection.py
  state.py
  stats.py
```

Add only small integration edits to existing files:

```text
asym_gemm/training/__init__.py
asym_gemm/training/moe.py                  optional, only for a new full_ft backend path
scripts/profile_lora.py                    optional, only if adding a benchmark flag
tests/test_full_ft_weight.py               new
tests/test_full_ft_optimizer.py            new
tests/test_full_ft_linear.py               new
tests/test_full_ft_grouped.py              new
tests/test_full_ft_injection.py            new
```

Do not edit these unless a test exposes a real shared-helper need:

```text
asym_gemm/training/lora.py
asym_gemm/training/frozen_linear.py
asym_gemm/training/host_weight.py
```

The reason is simple: LoRA SFT already has a working frozen-base contract. Full fine-tuning needs a different lifecycle, so it should be isolated.

### `config.py`

This file owns public configuration dataclasses and validation.

Expected classes:

```python
@dataclass(frozen=True)
class FullFTOptimizerConfig:
    optimizer: Literal["sgd", "adamw"] = "adamw"
    lr: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.0
    master_dtype: torch.dtype = torch.float32
    maximize: bool = False
    bias_correction: bool = True


@dataclass(frozen=True)
class FullFTSchedulerConfig:
    block_n: int = 1024
    block_k: int | None = None
    staging_depth: int = 2
    cpu_threads: int | None = None
    threadpool_count: int = 1
    use_copy_stream: bool = True
    strict_fetch_barrier: bool = True
    gradient_accumulation_steps: int = 1


@dataclass(frozen=True)
class FullFTModuleConfig:
    backend: Literal["asym", "torch"] = "asym"
    precision: Literal["bf16", "fp8", "fp4"] = "bf16"
    pin_fetch: bool = True
    train_bias: bool = True
    optimizer: FullFTOptimizerConfig = FullFTOptimizerConfig()
    scheduler: FullFTSchedulerConfig = FullFTSchedulerConfig()
```

Validation rules:

- `precision="bf16"` is the only supported precision in V1.
- `gradient_accumulation_steps` must be `1` in V1.
- `block_n` must be positive.
- `staging_depth` must be at least `1`; use `2` for pipelining.
- AdamW betas must be in `[0, 1)`.
- `master_dtype` must be `torch.float32`, `torch.bfloat16`, or `torch.float16`; start with `torch.float32`.
- `backend="torch"` exists only for correctness fallback tests, not the target performance path.

Expected helpers:

```python
def normalize_dtype(dtype: torch.dtype | str) -> torch.dtype: ...
def validate_full_ft_config(config: FullFTModuleConfig) -> None: ...
```

### `stats.py`

This file owns counters and profiling names.

Expected class:

```python
@dataclass
class FullFTStats:
    forward_calls: int = 0
    backward_calls: int = 0
    dx_gemm_calls: int = 0
    dw_gemm_calls: int = 0
    d2h_copies: int = 0
    cpu_update_tasks: int = 0
    refresh_tasks: int = 0
    finalized_steps: int = 0

    d2h_grad_bytes: int = 0
    refresh_bytes: int = 0
    master_bytes: int = 0
    optimizer_state_bytes: int = 0
    pinned_fetch_bytes: int = 0
    pinned_staging_bytes: int = 0

    active_expert_updates: int = 0
    inactive_expert_skips: int = 0

    scheduler_wait_seconds: float = 0.0
    cpu_update_seconds: float = 0.0
    refresh_seconds: float = 0.0
```

Expected constants:

```python
RANGE_FORWARD = "forward.full_ft_asymgemm"
RANGE_DX = "backward.full_ft_dx_asymgemm"
RANGE_DW = "backward.full_ft_dw_gemm"
RANGE_D2H = "backward.full_ft_dw_d2h"
RANGE_CPU_UPDATE = "optimizer.full_ft_cpu_update"
RANGE_REFRESH = "optimizer.full_ft_refresh_fetch"
RANGE_FINALIZE = "optimizer.full_ft_finalize"
```

Use the existing profiling helper style from the training package where available.

### `weight.py`

This file owns trainable offloaded storage.

Expected classes:

```python
@dataclass(frozen=True)
class WeightBlock:
    row_start: int
    row_end: int
    col_start: int = 0
    col_end: int | None = None
    expert: int | None = None


class AsymFullFTWeight:
    def __init__(
        self,
        tensor: torch.Tensor,
        *,
        name: str,
        optimizer_config: FullFTOptimizerConfig,
        fetch_dtype: torch.dtype = torch.bfloat16,
        pin_fetch: bool = True,
        clone: bool = True,
        expert_dim: bool = False,
    ) -> None:
        ...
```

Required fields:

```python
self.master              # CPU pageable source of truth
self.fetch               # CPU pinned AsymGEMM-facing copy
self.exp_avg             # CPU pageable Adam m or None
self.exp_avg_sq          # CPU pageable Adam v or None
self.step                # int
self.version             # int
self.name                # str
self.fetch_dtype         # torch.dtype
self.master_dtype        # torch.dtype
self.expert_dim          # bool, true for [E, N, K]
self._quantized_fetch    # optional future cache, invalidated on update
```

Constructor behavior:

1. Detach input tensor.
2. Move to CPU.
3. Convert to `optimizer_config.master_dtype`.
4. Store as contiguous pageable `master`.
5. Allocate `fetch = master.to(fetch_dtype).contiguous()`.
6. Pin `fetch` if requested and CUDA is available.
7. Allocate Adam state only if optimizer is AdamW.
8. Set `requires_grad_(False)` for every CPU tensor.

Methods:

```python
@classmethod
def from_linear(cls, linear: nn.Linear, *, name: str, config: FullFTModuleConfig) -> "AsymFullFTWeight": ...

@classmethod
def from_expert_tensor(cls, tensor: torch.Tensor, *, name: str, config: FullFTModuleConfig) -> "AsymFullFTWeight": ...

def fetch_tensor(self) -> torch.Tensor: ...
def grouped_fetch_tensor(self) -> torch.Tensor: ...
def master_block(self, block: WeightBlock) -> torch.Tensor: ...
def fetch_block(self, block: WeightBlock) -> torch.Tensor: ...
def refresh_block(self, block: WeightBlock) -> None: ...
def refresh_all(self) -> None: ...
def invalidate_derived_caches(self, block: WeightBlock | None = None) -> None: ...
def pinned_cpu_bytes(self) -> int: ...
def master_bytes(self) -> int: ...
def optimizer_state_bytes(self) -> int: ...
def state_dict(self) -> dict[str, torch.Tensor | int | str]: ...
def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
```

Refresh behavior:

```python
fetch_block.copy_(master_block.to(dtype=fetch_dtype))
version += 1 only after a logical update group is complete
```

For grouped weights shaped `[E, N, K]`, `WeightBlock.expert` selects the first dimension. Dense weights use `expert=None`.

Important rule:

```text
AsymFullFTWeight is not an nn.Parameter and must not be returned by model.parameters().
```

### `optimizer.py`

This file owns CPU update math.

Expected classes:

```python
class BlockwiseCPUOptimizer:
    def __init__(self, config: FullFTOptimizerConfig, *, stats: FullFTStats | None = None) -> None:
        ...

    def update_block(
        self,
        weight: AsymFullFTWeight,
        grad_cpu: torch.Tensor,
        block: WeightBlock,
        *,
        grad_scale: float = 1.0,
    ) -> None:
        ...
```

SGD update:

```text
grad = grad_cpu.float() * grad_scale
if weight_decay:
    grad = grad + weight_decay * master_block
master_block -= lr * grad
```

AdamW update:

```text
step += 1 once per logical optimizer step, not once per block
grad = grad_cpu.float() * grad_scale
if maximize:
    grad = -grad
exp_avg = beta1 * exp_avg + (1 - beta1) * grad
exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
if bias_correction:
    step_size = lr * sqrt(1 - beta2**step) / (1 - beta1**step)
else:
    step_size = lr
master_block *= (1 - lr * weight_decay)
master_block -= step_size * exp_avg / (sqrt(exp_avg_sq) + eps)
```

Step counting detail:

- For dense layer blocks, every block belongs to the same global update step.
- For grouped experts, all active expert blocks in one backward belong to the same global update step.
- Increment `weight.step` once per `finalize_backward()` per weight, before or after block updates consistently.
- Do not increment `step` per block, or AdamW will be wrong.

To support this, the scheduler should call:

```python
optimizer.begin_weight_step(weight)
optimizer.update_block(...)
optimizer.end_weight_step(weight)
```

or pass a `logical_step` into every block task.

Tests must verify this against PyTorch AdamW.

### `scheduler.py`

This file owns ordering, staging buffers, copy streams, and CPU workers.

Expected dataclasses:

```python
@dataclass
class GradBlockTask:
    weight: AsymFullFTWeight
    block: WeightBlock
    grad_gpu: torch.Tensor | None
    grad_cpu: torch.Tensor | None
    ready_event: torch.cuda.Event | None
    read_protect_event: torch.cuda.Event | None
    grad_scale: float = 1.0


@dataclass
class StagingSlot:
    gpu_buffer: torch.Tensor
    cpu_buffer: torch.Tensor
    ready_event: torch.cuda.Event | None = None
    in_use: bool = False
```

Expected class:

```python
class AsymFullFTUpdateScheduler:
    def __init__(
        self,
        config: FullFTSchedulerConfig,
        optimizer: BlockwiseCPUOptimizer,
        *,
        stats: FullFTStats | None = None,
    ) -> None:
        ...
```

Required methods:

```python
def allocate_slots(self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> None: ...
def acquire_slot(self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> StagingSlot: ...
def release_slot(self, slot: StagingSlot) -> None: ...

def begin_backward(self) -> None: ...
def protect_fetch_until(self, weight: AsymFullFTWeight, event: torch.cuda.Event) -> None: ...
def enqueue_d2h_update(self, task: GradBlockTask) -> None: ...
def enqueue_cpu_update(self, task: GradBlockTask) -> None: ...
def finalize_backward(self) -> None: ...
def zero_grad_accumulators(self) -> None: ...
```

V1 scheduling policy:

```text
strict_fetch_barrier = True
1. Backward computes grad_x for a layer.
2. Record one CUDA event after grad_x.
3. Every update task for that weight waits on that event before refreshing fetch.
4. dW blocks may be computed after grad_x.
5. D2H copies complete before CPU update.
6. finalize_backward waits for every submitted task.
```

V1 can be synchronous internally if needed:

```text
compute dW block
copy D2H blocking
CPU update
refresh fetch
next block
```

But the interface must already look asynchronous so Phase 4 can add real pipelining without changing autograd modules.

Concurrency model:

- A CUDA compute stream may be the current stream in V1.
- A copy stream is optional in V1.
- CPU update workers should not write into `fetch` until `read_protect_event.synchronize()` has completed.
- Multiple blocks of the same weight can update in parallel only if they are disjoint.
- Blocks for different weights can update in parallel.

Failure behavior:

- Store exceptions from worker threads.
- `finalize_backward()` re-raises the first exception.
- After an exception, mark scheduler unhealthy and reject new work until reset.

### `linear.py`

This file owns dense full-FT linear.

Expected public module:

```python
class AsymFullFTLinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        name: str = "",
        config: FullFTModuleConfig,
        scheduler: AsymFullFTUpdateScheduler | None = None,
        stats: FullFTStats | None = None,
    ) -> None:
        ...

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, name: str, config: FullFTModuleConfig, scheduler=None, stats=None) -> "AsymFullFTLinear": ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

Expected autograd function:

```python
class AsymFullFTLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_owner, bias, backend, precision, scheduler, stats, profile_name): ...

    @staticmethod
    def backward(ctx, grad_output): ...
```

Forward:

1. Validate `x.shape[-1] == weight.in_features`.
2. Ensure no in-flight refresh is pending for this weight.
3. Flatten `x` to `[M, K]`.
4. Call existing `asym_frozen_linear`-style dispatch using `weight.fetch`.
5. Add bias if present.
6. Save `x` or a recompute handle for backward.
7. Save `weight.version` used by forward.

Backward:

1. Flatten `grad_output` to `[M, N]`.
2. Compute `grad_x` before any update:

```python
grad_x = _dispatch_nt(
    grad_output_2d,
    weight.fetch,
    backend=backend,
    transpose_b=True,
    precision=precision,
)
```

3. Record `read_done_event` on the current stream.
4. Compute bias grad if needed.
5. For `n0:n1` blocks:

```python
dY_block = grad_output_2d[:, n0:n1]
dW_gpu = torch.matmul(dY_block.transpose(0, 1), x_2d)
scheduler.enqueue_d2h_update(
    GradBlockTask(
        weight=weight,
        block=WeightBlock(n0, n1),
        grad_gpu=dW_gpu,
        read_protect_event=read_done_event,
    )
)
```

6. Return `grad_x.reshape(input_shape)`.

Returned gradients:

```text
x: grad_x or None
weight_owner: None
bias: grad_bias or None
backend/config/scheduler/stats/profile: None
```

Bias policy:

- V1 can leave bias as a normal `nn.Parameter` if that is easier.
- If bias is offloaded, add `AsymFullFTBias` later.
- Do not mix hidden bias updates into `weight.py` unless tests cover it.

Memory policy:

- Saving `x` costs HBM. That is required for exact `dW`.
- Activation checkpointing can reduce this later by recomputing `x`.
- Do not save `dY`; backward receives it.

### `grouped.py`

This file owns grouped/MoE full-FT linear.

Expected public module:

```python
class AsymGroupedFullFTLinear(nn.Module):
    def __init__(
        self,
        expert_weight: torch.Tensor,
        *,
        name: str,
        config: FullFTModuleConfig,
        scheduler: AsymFullFTUpdateScheduler | None = None,
        stats: FullFTStats | None = None,
    ) -> None:
        ...

    def forward(self, x: torch.Tensor, offsets: torch.Tensor, experts: torch.Tensor) -> torch.Tensor: ...
```

Expected autograd function:

```python
class AsymGroupedFullFTLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_owner, offsets, experts, backend, precision, scheduler, stats, profile_name): ...

    @staticmethod
    def backward(ctx, grad_output): ...
```

Forward:

1. Validate `weight.fetch` shape `[E, N, K]`.
2. Validate `x` shape `[M, K]`.
3. Validate `offsets` and `experts` are CUDA int tensors following existing grouped AsymGEMM contract.
4. Call existing grouped AsymGEMM dispatch.
5. Save `x`, `offsets`, `experts`, active expert ids, and weight version.

Backward:

1. Compute grouped `grad_x` first with old fetch.
2. Record one read-protect CUDA event.
3. Build active expert ranges from `offsets` and `experts`.
4. For each active expert:

```python
rows = token rows for expert e
X_e = x_2d[rows, :]
dY_e = grad_output_2d[rows, :]
for n0:n1:
    dW_gpu = dY_e[:, n0:n1].T @ X_e
    enqueue block update for WeightBlock(n0, n1, expert=e)
```

5. If one physical expert appears in multiple ranges, combine the ranges before optimizer update:

```text
dW_e_block = sum over all ranges for expert e of dY_range[:, n0:n1].T @ X_range
```

6. Scatter `grad_x` back if the caller expects original token order.

The initial implementation can require the existing MoE packing to present one contiguous range per active expert. If not true, fail clearly with:

```text
AsymGroupedFullFTLinear V1 requires one packed contiguous range per active expert.
```

Then add repeated-range accumulation in Phase 3.

### `injection.py`

This file owns model replacement helpers.

Expected dataclasses:

```python
@dataclass
class FullFTSetupReport:
    replaced_modules: list[str]
    skipped_modules: dict[str, str]
    offloaded_param_count: int
    normal_param_count: int
    master_bytes: int
    optimizer_state_bytes: int
    pinned_fetch_bytes: int
    pinned_staging_bytes: int
```

Expected functions:

```python
def add_asym_full_ft(
    model: nn.Module,
    *,
    target_modules: Sequence[str] | str,
    config: FullFTModuleConfig,
    scheduler: AsymFullFTUpdateScheduler | None = None,
    stats: FullFTStats | None = None,
    strict: bool = True,
) -> FullFTSetupReport:
    ...

def iter_asym_full_ft_modules(model: nn.Module) -> Iterator[nn.Module]: ...
def finalize_asym_full_ft(model: nn.Module) -> None: ...
def zero_asym_full_ft_grad_accumulators(model: nn.Module) -> None: ...
def normal_torch_parameters(model: nn.Module) -> list[nn.Parameter]: ...
```

Target matching:

- Match `nn.Linear` modules by suffix, same style as LoRA target matching.
- Do not replace existing `AsymLoRALinear` or LoRA internals.
- Do not replace modules already inside `asym_gemm.training.full_ft`.
- `strict=True` raises if no target modules matched.

Replacement algorithm:

1. Traverse `model.named_modules()`.
2. Find leaf `nn.Linear` modules whose names match `target_modules`.
3. Replace each with `AsymFullFTLinear.from_linear(...)`.
4. Preserve bias according to config.
5. Remove original weight from normal parameter groups by replacing the module.
6. Return setup report.

For MoE packed expert weights, use model-specific adapter hooks later:

```python
if hasattr(module, "replace_experts_with_full_ft"):
    module.replace_experts_with_full_ft(config, scheduler=scheduler, stats=stats)
```

Keep those hooks small. They should identify the packed expert tensors, not implement optimizer logic.

### `state.py`

This file owns checkpoint helpers.

Expected functions:

```python
def full_ft_state_dict(model: nn.Module) -> dict[str, Any]: ...
def load_full_ft_state_dict(model: nn.Module, state: Mapping[str, Any], *, strict: bool = True) -> None: ...
def merge_full_ft_into_model_state_dict(model: nn.Module, destination: dict[str, Any], prefix: str = "") -> None: ...
```

Each full-FT module should save:

```text
<name>.master
<name>.fetch_dtype
<name>.master_dtype
<name>.optimizer
<name>.step
<name>.exp_avg
<name>.exp_avg_sq
<name>.version
```

Loading sequence:

1. Restore CPU master.
2. Restore optimizer state.
3. Allocate pinned fetch.
4. Refresh fetch from master.
5. Set version.
6. Clear in-flight events.

Do not save:

- GPU staging buffers;
- pinned CPU grad staging buffers;
- CUDA events;
- thread pool state;
- derived quantized fetch metadata.

### `__init__.py`

Export only stable APIs:

```python
from .config import FullFTModuleConfig, FullFTOptimizerConfig, FullFTSchedulerConfig
from .weight import AsymFullFTWeight, WeightBlock
from .optimizer import BlockwiseCPUOptimizer
from .scheduler import AsymFullFTUpdateScheduler
from .linear import AsymFullFTLinear
from .grouped import AsymGroupedFullFTLinear
from .injection import add_asym_full_ft, finalize_asym_full_ft, normal_torch_parameters
from .state import full_ft_state_dict, load_full_ft_state_dict
from .stats import FullFTStats
```

Do not export internal task classes unless tests need them.

## End-To-End Dataflow

### Forward

```text
input GPU activation
  -> AsymFullFTLinear.forward
  -> read weight.fetch from CPU pinned memory through AsymGEMM
  -> output GPU activation
```

No optimizer state is touched in forward.

### Backward

```text
grad_output GPU
  -> compute grad_x with old weight.fetch
  -> record read_done_event
  -> compute dW block on GPU
  -> D2H copy dW block into pinned staging
  -> CPU optimizer updates weight.master block
  -> CPU refreshes weight.fetch block
  -> finalize waits
```

### Next Forward

```text
wait finalize complete
  -> next forward reads refreshed weight.fetch
```

This is intentionally conservative. It gives correct math first. Pipelining can come after correctness.

## Training Loop Contract

User code should look like:

```python
full_ft_stats = FullFTStats()
full_ft_config = FullFTModuleConfig(
    optimizer=FullFTOptimizerConfig(optimizer="adamw", lr=1e-5, weight_decay=0.01),
    scheduler=FullFTSchedulerConfig(block_n=1024, gradient_accumulation_steps=1),
)
report = add_asym_full_ft(
    model,
    target_modules=("gate_proj", "up_proj", "down_proj"),
    config=full_ft_config,
    stats=full_ft_stats,
)

torch_optimizer = torch.optim.AdamW(normal_torch_parameters(model), lr=normal_lr)

for batch in loader:
    loss = model(**batch).loss
    loss.backward()

    finalize_asym_full_ft(model)

    torch_optimizer.step()
    torch_optimizer.zero_grad(set_to_none=True)
    zero_asym_full_ft_grad_accumulators(model)
```

Important ordering:

- `finalize_asym_full_ft(model)` must happen before the next forward.
- `torch_optimizer.step()` does not update offloaded full-FT weights.
- Offloaded full-FT weights are already updated by the full-FT scheduler.
- If the scheduler only queues updates in backward, `finalize` performs the real wait/update completion.

## First Prototype Scope

Implement the first prototype with these constraints:

```text
precision: bf16 fetch only
master dtype: fp32
optimizer: SGD first, then AdamW
gradient accumulation: 1
module type: dense nn.Linear first
MoE: after dense correctness
copy scheduling: synchronous allowed behind async-shaped API
fetch buffering: single fetch copy
update barrier: one read_done_event per layer backward
```

Do not implement in the first prototype:

```text
FP8/FP4 fetch refresh
double-buffered fetch copies
K chunking unless dW buffer is too large
ZeRO/FSDP integration
distributed expert ownership
gradient accumulation > 1
activation recompute
Triton custom dW kernels
```

## Step-By-Step Milestones

### Milestone 0: Scaffold

Create package files and exports.

Deliverables:

- package imports work;
- config validation works;
- no existing LoRA tests broken.

Validation:

```bash
python - <<'PY'
from asym_gemm.training.full_ft import FullFTModuleConfig, AsymFullFTWeight
print("full_ft imports ok")
PY
```

### Milestone 1: CPU Weight Ownership

Implement `AsymFullFTWeight`.

Tests:

- CPU input tensor creates CPU pageable `master`.
- GPU input tensor is copied to CPU.
- `fetch` is BF16 and pinned when CUDA supports pinning.
- `master.requires_grad` and `fetch.requires_grad` are false.
- `refresh_block` updates only the selected block.
- grouped `[E, N, K]` block selection works.
- state dict roundtrip recreates fetch from master.

Acceptance:

```text
W_master changes when edited
refresh_block copies changed rows into W_fetch
no weight is an nn.Parameter
```

### Milestone 2: CPU Optimizer Correctness

Implement `BlockwiseCPUOptimizer` with SGD.

Tests:

- one block SGD equals manual PyTorch math.
- multiple row blocks equal one full-matrix update.
- weight decay matches expected behavior.

Then implement AdamW.

AdamW tests:

- one full update equals PyTorch AdamW on tiny tensor within tolerance.
- row-blocked update equals unblocked update.
- `step` increments once per logical step, not per block.
- bias correction matches PyTorch.

Acceptance:

```text
Blockwise optimizer can update all blocks and reproduce torch optimizer math on small tensors.
```

### Milestone 3: Dense Full-FT Linear With Synchronous Updates

Implement `AsymFullFTLinear` and autograd.

Start with a simple synchronous path:

```text
for each block:
    dW_gpu = dY_block.T @ X
    dW_cpu = dW_gpu.cpu()
    optimizer.update_block(...)
    weight.refresh_block(...)
```

Keep the scheduler interface in place even if internally synchronous.

Tests:

- forward output matches `nn.Linear` for tiny shape.
- one SGD step matches `nn.Linear` + PyTorch SGD.
- one AdamW step matches `nn.Linear` + PyTorch AdamW.
- `grad_x` matches PyTorch.
- bias grad matches PyTorch if bias remains normal.
- no `.grad` appears on offloaded weight.
- LoRA tests still pass.

Acceptance:

```text
Dense full-FT trains correctly on tiny shapes and does not use HBM for weight/optimizer state.
```

### Milestone 4: Scheduler Interface And Finalize Hook

Move update work through `AsymFullFTUpdateScheduler`.

V1 can still run synchronously, but all autograd code should call scheduler APIs.

Tests:

- `finalize_backward()` waits for queued work.
- exceptions in worker tasks re-raise in finalize.
- next forward refuses to start if previous update is pending.
- stats counters update.

Acceptance:

```text
The autograd function no longer owns update lifecycle details directly.
```

### Milestone 5: Async D2H And CPU Worker Pool

Add ring-buffer staging and optional copy stream.

Tests:

- staging slots are reused only after completion.
- D2H bytes are counted.
- CPU tasks update disjoint blocks correctly.
- simultaneous blocks of different weights do not interfere.

Acceptance:

```text
GPU dW computation can overlap with CPU update for prior block.
```

### Milestone 6: Grouped/MoE Full-FT Linear

Implement `AsymGroupedFullFTLinear`.

Start with one contiguous range per expert.

Tests:

- grouped forward matches PyTorch grouped expert matmul.
- grouped backward `grad_x` matches PyTorch.
- active expert `dW` update matches PyTorch.
- inactive expert weights remain unchanged.
- repeated expert ids either work or fail with a clear V1 error.

Acceptance:

```text
Routed expert full-FT works for tiny MoE and updates only active experts.
```

### Milestone 7: MoE Integration

Add an optional full-FT expert backend path in `moe.py`, or add a model injection hook if the existing toy MoE exposes packed expert tensors.

Rules:

- Existing `backend="asym"` LoRA path must not change.
- New path should use a new name, for example `backend="asym_full_ft"` or a separate benchmark flag.
- Shared experts are optional in V1.
- Router and non-expert params stay normal PyTorch params.

Acceptance:

```text
Toy MoE full-FT benchmark can run without changing LoRA benchmark semantics.
```

### Milestone 8: State Dict And Resume

Implement `state.py`.

Tests:

- train one step, save, load into new model, outputs match.
- optimizer state resumes and next AdamW step matches uninterrupted training.
- fetch is regenerated from master on load.

Acceptance:

```text
Full-FT checkpointing saves true trainable state, not just serving fetch cache.
```

### Milestone 9: Profiling And Memory Accounting

Wire `FullFTStats` into profile output.

Report:

- HBM allocated before/after replacement;
- CPU master bytes;
- CPU optimizer state bytes;
- pinned fetch bytes;
- pinned staging bytes;
- active expert count;
- D2H gradient bytes per step;
- CPU optimizer time;
- scheduler wait time;
- full step time.

Acceptance:

```text
Benchmark output shows whether memory savings justify slowdown.
```

### Milestone 10: Quantized Fetch Follow-Up

Only after BF16 passes correctness and profiling:

- add FP8/FP4 fetch cache support;
- add blockwise quantization refresh;
- update scales with values;
- preserve the single pinned host-weight fetch-copy invariant;
- compare loss and update behavior against BF16 fetch.

Acceptance:

```text
Quantized fetch is an optional serving/cache format, not the optimizer source of truth.
```

## Test Matrix

### CPU-Only Tests

These should run without GPU:

- config validation;
- weight CPU master allocation;
- optimizer SGD/AdamW block math;
- state dict roundtrip for master and optimizer state;
- injection target matching.

### CUDA Tests

These require CUDA:

- pinned fetch allocation;
- AsymFullFTLinear forward;
- dense backward `grad_x`;
- dense `dW` block update;
- scheduler CUDA event barrier;
- grouped AsymGEMM forward/backward if supported on the current arch.

### AsymGEMM Hardware Tests

These require direct AsymGEMM support:

- BF16 direct dense linear;
- BF16 grouped expert linear;
- alignment behavior;
- fallback errors when shape or arch is unsupported.

### Regression Tests

Always run:

```text
existing LoRA tests
existing frozen linear tests
existing profile backend smoke tests
```

The full-FT work is not allowed to change LoRA SFT behavior.

## Shape And Memory Examples

For a dense projection:

```text
N = 14336
K = 4096
block_n = 1024
dW block = 1024 * 4096 * 2 bytes BF16 = 8 MiB
FP32 master block = 16 MiB
Adam m block = 16 MiB
Adam v block = 16 MiB
```

Peak GPU temp for one block is approximately:

```text
dW_gpu block + dY block view + saved X
```

The big savings are:

```text
full W not in HBM
Adam m/v not in HBM
master weight not in HBM
```

The cost is:

```text
D2H dW transfer every step
CPU optimizer bandwidth
fetch refresh bandwidth
extra synchronization
```

For MoE:

```text
Only active experts pay dW/update/refresh cost.
```

This is the main reason MoE expert full-FT is a better target than dense all-layer full-FT.

## Expected Classes After Implementation

You should see these classes:

```text
FullFTOptimizerConfig
FullFTSchedulerConfig
FullFTModuleConfig
FullFTStats
WeightBlock
AsymFullFTWeight
BlockwiseCPUOptimizer
GradBlockTask
StagingSlot
AsymFullFTUpdateScheduler
AsymFullFTLinear
AsymFullFTLinearFunction
AsymGroupedFullFTLinear
AsymGroupedFullFTLinearFunction
FullFTSetupReport
```

You may also see these optional helpers:

```text
AsymFullFTMoEWrapper
FullFTStateDictMixin
FullFTUnsupportedError
FullFTPendingUpdateError
```

## Expected User-Facing APIs

Minimum API:

```python
config = FullFTModuleConfig(...)
report = add_asym_full_ft(model, target_modules=("gate_proj", "up_proj", "down_proj"), config=config)
optimizer = torch.optim.AdamW(normal_torch_parameters(model), lr=...)

loss.backward()
finalize_asym_full_ft(model)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

Manual layer API:

```python
layer = AsymFullFTLinear.from_linear(linear, name="mlp.up_proj", config=config)
```

Grouped expert API:

```python
experts = AsymGroupedFullFTLinear(expert_weight, name="layers.0.experts.up", config=config)
out = experts(packed_x, offsets, expert_ids)
```

Checkpoint API:

```python
state = full_ft_state_dict(model)
load_full_ft_state_dict(model, state)
```

## Error Messages To Implement

Use explicit errors:

```text
Asym full-FT V1 supports precision='bf16' only.
Asym full-FT V1 requires gradient_accumulation_steps=1.
Offloaded full-FT weights must not be passed to torch.optim.
Cannot start forward: previous full-FT update has not been finalized.
AsymGroupedFullFTLinear V1 requires one contiguous packed range per active expert.
Unsupported AsymGEMM shape for full-FT fetch: expected aligned BF16 [N, K].
```

These errors are better than silent fallbacks because silent fallbacks hide memory use and invalidate benchmarks.

## Main Risks

### Risk: `dW` D2H Bandwidth Dominates

Mitigation:

- use larger row blocks;
- prioritize MoE active experts;
- benchmark D2H bytes per step;
- avoid dense full-model FT over PCIe as the first target.

### Risk: CPU AdamW Too Slow

Mitigation:

- start with SGD for correctness;
- use vectorized PyTorch CPU ops over large contiguous blocks;
- use CPU threads only after block math is correct;
- consider fused C++ CPU optimizer later.

### Risk: Weight Freshness Bugs

Mitigation:

- strict finalize barrier in V1;
- version counters;
- tests that compare to PyTorch for multi-step training;
- fail if next forward starts with pending updates.

### Risk: LoRA Regression

Mitigation:

- no full-FT code inside `lora.py`;
- new backend name or explicit injection API;
- run existing LoRA tests after every phase.

### Risk: Too Many Tiny Blocks

Mitigation:

- default `block_n=1024`;
- no Python task per 64x64 tile;
- one task should represent a useful contiguous row block.

## Definition Of Done

The implementation is complete enough when:

- dense full-FT one-step SGD and AdamW match PyTorch on small CUDA tests;
- grouped/MoE full-FT updates only active experts and matches PyTorch on small tests;
- offloaded weights and optimizer states are absent from HBM;
- `finalize_asym_full_ft(model)` makes the next forward see refreshed weights;
- checkpoint save/load resumes correctly;
- profile output shows memory savings and update/copy overhead;
- existing LoRA SFT behavior and tests are unchanged.

The first useful demo should be:

```text
toy MoE
expert gate/up/down full-FT offloaded
router/attention/layernorm normal GPU training
BF16 AsymGEMM fetch
FP32 CPU AdamW master
block_n=1024
gradient_accumulation_steps=1
```

Expected result:

```text
The run is slower than GPU full fine-tuning, but HBM use is much lower.
If active expert count is small enough, the slowdown may be acceptable.
If dense all-layer FT is attempted over PCIe, D2H dW bandwidth will likely dominate.
```
