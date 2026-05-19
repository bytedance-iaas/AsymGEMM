# AsymGEMM-SFT: Backend Contributions For A Top ML Systems Paper

## Artifact Boundary

The paper artifact should be an AsymGEMM-SFT backend used from LLaMA-Factory.
The local PyTorch dense/MoE modules are correctness scaffolding only. They are
not the final system contribution.

LLaMA-Factory already owns:

- dataset loading, templates, packing, preprocessing, and CLI/YAML UX;
- SFT trainer entry points and launch plumbing;
- PEFT/LoRA adapter creation, names, dropout, scaling, save/load;
- optimizer and scheduler construction for trainable adapter/router parameters;
- Accelerate/DeepSpeed/FSDP integration.

AsymGEMM-SFT should own only the backend execution and placement of frozen
base/expert matrices after LLaMA-Factory and PEFT have built the training model.

```text
LLaMA-Factory/PEFT:
  model, tokenizer, dataset, trainer, LoRA adapters, optimizer

AsymGEMM-SFT backend:
  frozen base/expert storage
  direct-fetch forward and dX
  host-weight layout runtime
  routed MoE backend execution
  direct/staged/HBM placement policy
  backend timing and memory reporting
```

## SOTA Boundary And Non-Claims

Do not claim standard framework capabilities:

- LoRA freezes base weights and trains adapters.
- PyTorch autograd can propagate `dX` through frozen layers.
- PEFT/Transformers manage adapters and portable adapter checkpoints.
- LLaMA-Factory runs LoRA SFT from normal training YAMLs.
- ZeRO/FSDP/DeepSpeed offload or shard parameters, gradients, and optimizer
  states.
- KTransformers exposes LoRA SFT through LLaMA-Factory with KT backends such as
  AMX BF16/INT8/INT4 for CPU-oriented expert execution.
- Generic MoE routing, top-k selection, and scatter are already common runtime
  mechanisms.

The new claim must be:

```text
Frozen BF16 base/expert matrices remain CPU-pinned host operands, but GPU tensor
cores compute both forward and backward-input GEMMs through direct host fetch.
```

Everything below must support that claim. A backend flag plus Python wrappers is
not enough for a top ML systems paper.

## Contribution Map

```text
Contribution                                      Main implementation layer
------------------------------------------------  ----------------------------
1. Host-operand training operators                native operator/kernel + runtime
2. LLaMA-Factory real-model backend patcher       PyTorch/LLaMA-Factory runtime
3. Host-weight layout and memory runtime          backend runtime + native support
4. Routed MoE direct-fetch execution engine       CUDA kernels + backend runtime
5. Direct/staged/HBM placement scheduler          backend runtime
6. Backend fusion and launch reduction            CUDA kernels + backend runtime
```

For this document, "PyTorch runtime" means the production backend layer behind
LLaMA-Factory, not the toy demonstration modules.

## Current Status Versus Paper Target

The current local implementation is a correctness ladder, not the final paper
system:

- existing AsymGEMM BF16 grouped kernels are the kernel substrate;
- current dense `dX` is demonstrated by calling the NT direct-fetch path on a
  lazily transposed CPU host weight;
- current `HostWeight` clones, pins, blocks CUDA migration, and lazily builds a
  transpose, but it is not yet a pool/descriptor/NUMA runtime;
- current tiny MoE metadata, packing, scatter, and per-expert execution use
  PyTorch operations and Python loops;
- there is no real LLaMA-Factory backend, no CUDA MoE route engine, no placement
  scheduler, and no KT/staging end-to-end comparison yet.

Therefore every contribution below is a target for the production backend. Do
not describe the current prototype as already having these full mechanisms.

## Contribution 1: Host-Operand Training Operators

**Implementation locus:** native operator/kernel work plus PyTorch backend
bindings.

### Motivation

Existing AsymGEMM kernels provide direct host-memory GEMM, but LoRA SFT needs a
training operator, not only forward execution. The frozen weight has no `dW`,
but it is still required for `dX`.

```text
forward:  X_gpu_bf16  @ W_cpu_bf16.T -> Y_gpu
backward: dY_gpu_bf16 @ W_cpu_bf16   -> dX_gpu
unused:   dW_cpu
```

The technical challenge is preserving normal LoRA math while `W_cpu_bf16` is not
a CUDA parameter, not staged into HBM, and not saved as an autograd tensor.

### Needed Components

- **Native forward operator**
  - Takes GPU BF16 activation, CPU-pinned BF16 host-weight descriptor, optional
    bias, grouped metadata when needed, and output dtype policy.
  - Emits GPU output with FP32 accumulation/output handling where required for
    tiled-K correctness.
  - Rejects unsupported dtype/layout/device combinations in no-fallback mode.

- **Operator ABI shape**
  - Dense forward: `(x_gpu, host_weight_fwd_desc, bias?, stream?, policy) -> y`.
  - Dense `dX`: `(dy_gpu, host_weight_dx_desc, stream?, policy) -> dx`.
  - Grouped forward/`dX`: add route/group metadata descriptors and expected
    token counts.
  - ABI version, layout version, dtype, shape, alignment, and kernel capability
    must be checked before launch.
  - Unsupported ABI/version combinations must fail loudly in direct-only mode.

- **Native backward-input operator**
  - Computes `dX = dY @ W` from the host-resident backward layout.
  - May initially reuse an NT direct-fetch GEMM over a prepared `W_dx` host
    layout, but the backend must expose it as a first-class no-fallback `dX`
    operator path.
  - Does not compute or allocate `dW`.
  - Shares descriptor and layout ABI with the forward operator.
  - Supports dense and grouped expert shapes needed by target MoE models.
  - Must not be overclaimed as a general native `dgrad/wgrad` API unless such an
    API is actually implemented and exported.

- **Host descriptor ABI**
  - Stable descriptor for CPU pointer, shape, stride/layout, dtype, pin status,
    NUMA node, alignment, and layout kind.
  - Descriptor must be valid across repeated steps and safe against Python tensor
    lifetime mistakes.
  - Runtime must detect stale or invalid descriptors.
  - Descriptor ownership rules must define which object owns host memory, when
    descriptors are invalidated, and how kernel ABI versions are checked.

- **Grouped metadata ABI**
  - Offsets, expert ids, masked counts, route counts, and sentinel conventions
    must be documented and validated.
  - Same metadata contract must be usable in forward and `dX`.

- **Numerical contract**
  - BF16 input and host weights.
  - FP32 accumulation or FP32 output buffer when partial K tiles would otherwise
    be rounded through BF16.
  - Final model output/gradient emitted in the configured training dtype.

- **Autograd binding**
  - Saves host descriptor id/layout metadata, not a GPU copy of the weight.
  - Backward calls the direct-fetch `dX` operator.
  - Returns `None` for frozen weight gradients.
  - Exposes counters for direct forward, direct `dX`, staged fallback, and Torch
    fallback.

- **Acceptance checks**
  - Direct forward and direct `dX` counters positive with fallback disabled.
  - No CUDA allocation proportional to frozen `W` in direct mode.
  - Dense LoRA gradients match standard LLaMA-Factory/PEFT math.
  - Large-K tests show no BF16 partial-reduction numerical blocker.
  - Supported shape/layout/alignment matrix is documented and enforced.

## Contribution 2: LLaMA-Factory Real-Model Backend Patcher

**Implementation locus:** PyTorch/LLaMA-Factory backend runtime. This is not
kernel work.

### Motivation

Toy `AsymFrozenLinear` modules do not prove the system. The backend must patch
real models loaded by LLaMA-Factory after PEFT has injected LoRA, without
replacing LLaMA-Factory's trainer or PEFT semantics.

### Needed Components

- **Backend configuration**
  - YAML/config fields such as `use_asymgemm_sft` and `asymgemm_sft_config`.
  - Options for direct-only, direct-or-staged, scheduler-enabled, and report
    output.
  - Explicit unsupported-layer behavior: reject, stage, or GPU-resident.
  - Initial scope is single GPU; Accelerate/DeepSpeed/FSDP launches must be
    either explicitly blocked, warned, or supported with rank-local host pools.

- **Registration hook**
  - Runs after LLaMA-Factory model load and PEFT `init_adapter`/LoRA injection.
  - Runs before optimizer construction in the preferred path.
  - If patching occurs after optimizer construction, it must prove optimizer
    groups are still correct or rebuild them through LLaMA-Factory utilities.
  - Does not change dataset, trainer, optimizer, or adapter save/load paths.

- **Layer discovery**
  - Finds eligible frozen base linear layers inside PEFT LoRA wrappers.
  - Handles dense attention/MLP targets and MoE experts.
  - Supports model-family patterns such as `gate_proj`, `up_proj`, `down_proj`,
    shared experts, routed experts, and possible fan-in/fan-out conventions.
  - Records unsupported modules and fails loudly in no-fallback mode.
  - Maintains a supported-model matrix, for example dense Qwen, Qwen MoE,
    DeepSeek-style MoE, and explicit rejected architectures.

- **PEFT compatibility matrix**
  - Defines support for `LoraLayer`/`base_layer`, fan-in/fan-out, bias,
    modules-to-save, active adapter names, merged versus unmerged adapters, and
    adapter enable/disable states.
  - Rejects merged adapters or unsupported multi-adapter states unless explicitly
    implemented.

- **Module replacement**
  - Replaces only frozen base compute with AsymGEMM-SFT backend modules.
  - Leaves LoRA A/B, dropout, scaling, adapter names, and trainable flags
    unchanged.
  - Preserves PEFT adapter state dict compatibility.
  - Filters adapter saves so host weights do not leak into PEFT adapter
    artifacts.
  - Prevents `.to(cuda)`, mixed-precision wrappers, or trainer utilities from
    migrating host weights into HBM.

- **Optimizer and state invariants**
  - Frozen host weights are absent from optimizer groups.
  - LoRA/router parameters remain normal trainable PyTorch parameters.
  - AdamW/optimizer state remains standard for trainable parameters only.

- **Acceptance checks**
  - A normal LLaMA-Factory YAML launches the backend.
  - Adapter save/load remains PEFT-compatible.
  - Model patch report lists every patched, staged, GPU-resident, and rejected
    module.
  - Gradient checkpointing, mixed precision, and single-rank Accelerate launch
    do not violate host-weight invariants.
  - No silent fallback in claimed direct experiments.

## Contribution 3: Host-Weight Layout And Memory Runtime

**Implementation locus:** backend runtime with native support for pinned
allocation, descriptors, alignment checks, and layout metadata.

### Motivation

Direct fetch is not just `tensor.pin_memory()`. Training needs persistent host
layouts for forward and backward-input access, plus lifetime, NUMA, checkpoint,
and memory-accounting semantics.

```text
checkpoint W
-> CPU BF16 host pool
-> forward layout for X @ W.T
-> dX layout for dY @ W
-> optional staged/HBM copies selected by scheduler
-> descriptor table consumed by backend operators
```

### Needed Components

- **Host allocator**
  - CPU-pinned allocation pool for frozen dense and expert weights.
  - Alignment compatible with direct-fetch kernels.
  - Configurable pin budget and failure behavior.
  - Per-NUMA-node allocation support.

- **Layout builder**
  - Builds forward host layout and backward-input host layout once.
  - Supports transposed or kernel-specific layout caches for `dX`.
  - Stores layout kind, shape, stride, dtype, and byte size.
  - Rebuilds deterministically on resume when caches are not checkpointed.

- **Descriptor table**
  - Maps model layer/expert ids to host descriptors.
  - Tracks placement state: direct, staged, GPU-resident, rejected.
  - Keeps Python/PyTorch objects alive only as owners, not as autograd tensors.
  - Exposes native pointer metadata safely to kernels.
  - Stores descriptor ABI version and layout version to reject incompatible
    manifests or kernels.

- **Staging and HBM alternatives**
  - Reusable H2D staging workspace for scheduler-selected layers.
  - Optional GPU-resident cache for hot or small matrices.
  - Same mathematical output across direct, staged, and GPU-resident modes.

- **NUMA and bandwidth calibration**
  - Measures host-path bandwidth per NUMA placement.
  - Binds host pools or records locality decisions.
  - Reports local vs remote bandwidth impact.

- **Checkpoint/resume**
  - Adapter-only checkpoint remains PEFT-compatible.
  - Adapter checkpoint must not include host weights or backend descriptors.
  - Full-system resume stores enough metadata to recreate host-resident weights
    directly.
  - Backend manifest stores source checkpoint keys, shapes, dtypes, layout ids,
    checksums or version ids, placement choices, and backend version.
  - Manifest records whether placement choices are fixed, recalibrated, or
    invalidated on resume or hardware change.
  - Host pointers are never serialized.
  - Resume must not restore frozen weights to GPU and then clean them up later.

- **Memory accounting**
  - Original frozen bytes.
  - Forward layout bytes.
  - Backward layout/cache bytes.
  - Pinned CPU bytes.
  - Staged workspace bytes.
  - GPU-resident selected bytes.
  - HBM avoided versus ordinary GPU-resident base weights.

- **Acceptance checks**
  - Host descriptors survive repeated training steps.
  - `.to(cuda)` and trainer movement do not migrate host weights.
  - Full resume reproduces direct execution counters and memory accounting.
  - Old or incompatible backend manifests fail with a clear version error.

## Contribution 4: Routed MoE Direct-Fetch Execution Engine

**Implementation locus:** CUDA kernels plus PyTorch backend runtime.

### Motivation

Generic MoE routing is not novel. The contribution is executing routed expert
base compute while expert matrices remain CPU-resident and GPU tensor cores
still compute both forward and `dX`.

```text
router output from model
-> backend route metadata
-> GPU route packing
-> grouped direct-fetch expert forward
-> weighted scatter
-> scatter backward
-> grouped direct-fetch expert dX
```

### Needed Components

- **Router integration**
  - Consumes router top-k ids and routing weights from the model.
  - Does not replace the model's router semantics.
  - Preserves router gradients.

- **Route metadata representation**
  - Common metadata object for forward packing, grouped GEMM, scatter, scatter
    backward, and expert `dX`.
  - Contains token ids, expert ids, route ids, routing weights, expert counts,
    offsets, padded capacity, and masks.
  - Supports contiguous and masked layouts.

- **GPU route packing**
  - CUDA kernel or graph-compatible operator to pack token activations by expert.
  - Avoids Python loops in the hot path.
  - Handles empty experts, repeated experts, and skewed load.
  - Produces contiguous active-token buffer or masked `[expert, slot]` buffer.

- **Grouped expert forward**
  - Uses AsymGEMM direct-fetch grouped BF16 operators for frozen expert
    `gate_proj`, `up_proj`, and `down_proj`.
  - Avoids Python per-expert loops in the measured strong-paper path.
  - Supports shared experts and routed experts with independent placement.
  - Uses the same route metadata as packing/scatter.

- **Weighted scatter and scatter backward**
  - CUDA implementation of weighted scatter to original token order.
  - Backward computes gradients for expert outputs and routing weights.
  - Reuses metadata from forward.
  - Preserves router and LoRA gradients.

- **Grouped expert `dX`**
  - Direct-fetch grouped `dX` through host-resident expert weights.
  - Correctly accumulates gradients for tokens receiving multiple expert routes.
  - Uses backward host layout descriptors.

- **Workspace and graph support**
  - Persistent buffers for packed tokens, route metadata, expert outputs,
    scatter gradients, and masks.
  - Masked/fixed-shape path for CUDA graph capture where feasible.
  - Explicit handling of dynamic route counts in contiguous mode.

- **Acceptance checks**
  - Dense and MoE gradients match the ordinary LLaMA-Factory model.
  - Empty/skewed/repeated expert tests pass.
  - Packing, scatter, and route overhead are reported separately.
  - No Python loops remain in the measured hot path for the strong-paper claim.

## Contribution 5: Direct/Staged/HBM Placement Scheduler

**Implementation locus:** backend runtime using kernel measurements and memory
signals.

### Motivation

Direct host fetch will not dominate every shape on H200. The backend needs to
choose the correct execution mode without changing training semantics.

```text
direct:       CPU pinned W -> AsymGEMM direct-fetch GEMM
staged:       async H2D W copy into reusable workspace -> GPU GEMM
gpu-resident: selected hot/small W stays in HBM
cpu-expert:   external KT-style baseline, not AsymGEMM-SFT fast path
```

### Needed Components

- **Calibration**
  - Measure direct-fetch kernel throughput by shape.
  - Measure staged H2D bandwidth and staged GEMM throughput.
  - Measure GPU-resident GEMM throughput.
  - Measure route packing/scatter overhead.
  - Measure host bandwidth under NUMA placements.

- **Cost model**
  - Inputs: M/N/K, expert type, active routes, top-k, skew, reuse across gradient
    accumulation, host bandwidth, HBM budget, staging workspace pressure.
  - Outputs: direct, staged, GPU-resident, or reject.
  - Must account for both forward and `dX`, not forward only.

- **Policy runtime**
  - Computes placement plan before training or after calibration warmup.
  - Can update plan when route distributions shift, with hysteresis to avoid
    thrashing.
  - Maintains identical math across placements.
  - Caches selected GPU-resident matrices under an HBM budget.
  - Persists placement plans only when hardware, kernel ABI, model shape, and
    calibration version match; otherwise recalibrates.

- **Staging overlap**
  - Reuses H2D workspaces.
  - Uses streams/events to overlap copy, route packing, LoRA compute, and GEMM
    when the scheduler selects staging.

- **Reporting**
  - Per-layer/expert placement table.
  - Predicted vs measured time.
  - HBM and pinned CPU bytes per placement.
  - Direct/staged/GPU-resident call counts.

- **Acceptance checks**
  - Direct-only, staged-only, GPU-resident, and scheduled modes produce matching
    training outputs/gradients.
  - Scheduler beats or matches the best static placement across representative
    route distributions.
  - Break-even curves are reported instead of only one speedup number.

## Contribution 6: Backend Fusion And Launch Reduction

**Implementation locus:** mixed, with the most kernel-heavy optional work.

### Motivation

If the backend is a sequence of disconnected operations, launch overhead and
materialization can erase direct-fetch gains:

```text
base direct-fetch GEMM
LoRA A GEMM
LoRA B GEMM
scatter
down projection
```

The strong systems path is to co-design kernels, streams, and workspaces so the
base path and LoRA overlay behave like one backend execution plan.

### Needed Components

- **Base + LoRA epilogue path**
  - Target computation:

    ```text
    FP32 accumulator  = direct_fetch_base(X_bf16, W_cpu_bf16)
    FP32 accumulator += lora_update(X, A_fp32, B_fp32)
    BF16 output       = cast(accumulator)
    ```

  - Can start as a semi-fused path with shared workspace and fewer launches.
  - Full fusion is strongest if shape/rank constraints make it worthwhile.

- **MoE gate/up batching**
  - Gate and up projections use the same routed token set.
  - Batch or fuse them to reduce packing and launch overhead.
  - Avoid materializing intermediate buffers beyond what activation functions
    require.

- **Stream overlap**
  - Overlap route packing with direct-fetch launches where dependencies allow.
  - Overlap staged H2D transfers with LoRA compute.
  - Use events to enforce correct sequencing without global synchronizations.

- **Persistent workspaces**
  - Preallocate activation, route, expert output, LoRA, and gradient buffers.
  - Avoid per-step tiny tensor allocations.
  - Maintain stable addresses for graph capture where possible.

- **CUDA graph compatibility**
  - Masked/fixed-shape path should be graph-capturable.
  - Contiguous dynamic path should still reuse workspaces and limit launch count.

- **Acceptance checks**
  - Launch count and materialized bytes decrease versus unfused backend.
  - Fusion or overlap improves end-to-end step time, not only microbenchmarks.
  - Numerical parity with unfused direct backend is maintained.

## Proof Obligations

These are required to make the claims credible, but are not standalone novelty:

- LLaMA-Factory YAML/CLI launches the backend.
- PEFT adapter artifacts remain compatible with ordinary save/load.
- Optimizer state exists only for trainable LoRA/router parameters.
- Frozen host weights are absent from CUDA parameters, CUDA gradients, and
  optimizer groups.
- Direct forward and direct `dX` counters are positive with fallback disabled.
- Dense and MoE gradients match ordinary LLaMA-Factory/PEFT math within
  BF16/FP32 tolerance.
- Full resume recreates host-resident execution directly.
- Backend manifest, host-layout rebuild, RNG/sampler/scheduler state, gradient
  accumulation state, and active PEFT adapter names resume through existing
  Trainer mechanics without changing adapter artifacts.
- Single-GPU is the initial scope, but rank-local host pools and failure modes
  must be explicit if launched under Accelerate/DeepSpeed/FSDP.
- Reports separate operator time, route packing, scatter, LoRA compute, staging,
  scheduler choice, optimizer, peak HBM, CPU RSS, and pinned bytes.

## Go/No-Go Bar For Top ML Systems

Minimum backend artifact:

```text
LLaMA-Factory backend registration/config path
real model patcher after PEFT injection
native direct-fetch forward + dX for frozen BF16 weights
persistent host-weight layout runtime
no-fallback dense LoRA correctness on a real model
no-fallback MoE LoRA correctness on a real model
direct vs staged comparison
```

Strong-paper artifact:

```text
CUDA route packing/scatter backward
grouped direct-fetch MoE expert forward + dX
direct/staged/HBM placement scheduler with break-even curves
NUMA-aware host-weight placement
launch reduction, stream overlap, or fused base+LoRA epilogue
end-to-end LLaMA-Factory MoE SFT result vs KT AMX backend and staging baseline
```

## Critical Verdict

The contribution set is strong enough for a top ML systems paper only if the
backend delivers the native operator path, host-layout runtime, MoE engine, and
placement scheduler. Those are meaningful technical mechanisms not already
provided by AsymGEMM, LLaMA-Factory, PEFT, ZeRO/FSDP, or KTransformers.

Three-reviewer consensus: this is a **no-go today** as a top ML systems paper
because the existing artifact is a correctness scaffold. It becomes a credible
top-paper target only after the strong-paper artifact above exists and is
validated against optimized staging, GPU-resident execution where it fits, and a
KTransformers-style AMX backend on the same machine.

If the final artifact is only:

```text
existing AsymGEMM forward kernel
+ Python autograd wrapper
+ PEFT module replacement
+ LLaMA-Factory flag
```

then it should be positioned as useful integration or a workshop systems note,
not a top ML systems paper.

## Target Final Claim After The Backend Exists

```text
AsymGEMM-SFT is a LLaMA-Factory backend for LoRA/MoE LoRA SFT that treats frozen
BF16 base and expert matrices as CPU-pinned host operands while GPU tensor cores
compute both forward and backward-input GEMMs through direct fetch. It adds a
host-weight layout runtime, routed MoE direct-fetch engine, and placement
scheduler so real SFT jobs can trade CPU memory, HBM, staging, and GPU
tensor-core compute without changing PEFT training semantics.
```
