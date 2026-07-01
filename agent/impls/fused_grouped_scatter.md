# Fused Grouped Scatter For Qwen3 MoE

## Goal

Implement the next Qwen3 MoE memory fix:

```text
original grouped expert compute
+ fused route-aware output/input placement
- route-space [R,H] intermediates
```

This is for:

```text
q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; <seq>|8|1 ; none|false|false|false|false|false
```

`asym_cpuadamwds` is the primary target backend for all real comparisons. Plain
`asym` is diagnostic-only and must not be used for the final scoreboard against
`superoffload_mem|unsloth` or `superoffload_mem|unsloth-off`.

Do not confuse these two meanings:

```text
RUNS backend = asym_cpuadamwds
  The end-to-end training backend used for apples-to-apples comparisons with
  superoffload_mem baselines.

internal grouped-linear backend = base.backend == "asym"
  The AsymGroupedFrozenLinear kernel backend inside the model. This internal check is
  still correct inside asym_cpuadamwds runs because the frozen expert linears use the
  AsymGEMM CPU-resident weight path.
```

with the real comparison baselines:

```text
q3-30b-a3b|1 ; superoffload_mem|unsloth|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false
q3-30b-a3b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false
```

The target is not blockwise expert splitting. The previous
`ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=16/18` path proved that avoiding the
global `[R,H]` tensor reduces HBM, but it exploded grouped calls and is not the final
performance design.

The correct target is new route-aware grouped kernels. Keep the grouped scheduling over
route-sorted expert rows, but do not materialize the full `[R,H]` output/input when the
next operation is only scatter/gather.

For Qwen3-30B-A3B at `s80000,b8,top_k=8`:

```text
M = batch * seq = 640000
R = M * top_k = 5120000
H = 2048
I = 768
[R,H] bf16 = about 19.5 GiB
[R,I] bf16 = about 7.3 GiB
```

Removing `[R,H]` is the main target. `[R,I]` is still large but is materially smaller
and is required by the expert activation math unless a later design changes the expert
body itself.

## Design Format Reference

This doc should follow the same discipline as:

```text
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/agent/impls/fix_finegrained_moe.md
```

Use that document as the reference for:

1. explicit goals before implementation detail;
2. apples-to-apples baseline comparison rules;
3. stage gates with expected results written before running;
4. artifact-first interpretation;
5. no conclusions from stale, partial, or mislabeled runs;
6. serialized experiments only;
7. no benchmark artifact overwrite or reuse;
8. final claims only after the real Qwen3 MoE workload is audited.

This document is narrower: it owns the fused grouped route-placement kernels. The
behavioral goal and validation style should match `fix_finegrained_moe.md`.

## Kernel Implementation References

Use these references before writing the kernels. The target is not to copy another MoE
kernel wholesale. The target is to reuse the right implementation ideas while preserving
AsymGEMM's CPU-resident packed-weight path.

### AsymGEMM local code

```text
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/asym_gemm/include/asym_gemm/impls/sm100_bf16_asym_gemm.cuh
```

Primary reference for the actual implementation. Reuse its SM100 grouped AsymGEMM
mainloop, TMA descriptor setup, UMMA/tensor-memory flow, shared-memory staging, and
current contiguous output epilogue patch points. The new kernels should be derived from
this code path, not from a separate generic GEMM framework, because this is the path
that already knows how to consume CPU-resident packed Asym weights.

Critical AsymGEMM detail from this file: the SM100 BF16 Asym kernel streams a B/weight
tile from CPU-pinned memory once for the current expert/N/K tile, keeps it in shared
memory, and reuses it across the inner route-row/M tile loop. The current loop shape is
effectively:

```text
for block_k_iter:
  stream/load B CPU-weight tile into smem_b once
  for block_m_iter in this expert group's route-row range:
    load or gather A tile
    run UMMA using the same smem_b tile
    store/scatter the tile result
```

That loop order is central to AsymGEMM. The routed kernels must preserve it. Do not
invert to an M-outer loop that reloads the same CPU-weight tile for every route-row
tile, and do not implement one standalone GPU-native GEMM per expert or per route block.

```text
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/asym_gemm/include/asym_gemm/common/asymScheduler.cuh
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/asym_gemm/include/asym_gemm/common/scheduler.cuh
```

Reference for the route-sorted grouped scheduling contract. The routed kernels must
keep the current grouped scheduling and must not devolve into one small GEMM per expert.

```text
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/csrc/apis/gemm.hpp
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/csrc/apis/qwen3_moe.hpp
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/csrc/python_api.cpp
```

Reference for extension registration, shape checks, stream/device handling, and existing
grouped-Asym API conventions. New Qwen3 routed APIs should be registered through the
Qwen3 MoE API surface, not through the generic frozen-linear dispatch.

```text
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/asym_gemm/training/qwen3_moe_finegrained.py
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/asym_gemm/training/frozen_linear.py
```

Reference for the exact Python call sites, saved-tensor behavior, packed CPU weight
ownership, and current route-space allocations. Use these files to make sure each new
kernel replaces only the intended base path and does not perturb dense full-fg.

```text
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/csrc/qwen3/qwen3_gate_up_windowed_bwd.cu
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/tests/training/test_qwen3_gate_up_windowed_bwd.py
```

Reference for Qwen3-specific CUDA extension style and focused Qwen3 kernel tests.

```text
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lora/profile_ncu_asymgemm.py
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lora/postprocess_ncu_asymgemm.py
```

Reference for local NCU launch/postprocess patterns. The new routed-kernel microbench
should follow this style where possible.

### CUTLASS examples inside AsymGEMM

```text
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/third-party/cutlass/examples/52_hopper_gather_scatter_fusion/gather_gemm.hpp
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/third-party/cutlass/examples/52_hopper_gather_scatter_fusion/scatter_epilogue.hpp
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/third-party/cutlass/examples/common/gather_tensor.hpp
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/third-party/cutlass/examples/52_hopper_gather_scatter_fusion/52_hopper_gather_scatter_fusion.cu
```

Reference for gather/scatter as a first-class GEMM placement transform. Useful for the
conceptual shape of "virtual gathered A" and "scatter epilogue D". Do not directly copy
the Hopper collective into the AsymGEMM SM100 TMA path, but use it to avoid inventing
the coordinate transform contract from scratch.

```text
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/third-party/cutlass/examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm.cu
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/third-party/cutlass/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_grouped.cu
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/third-party/cutlass/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_rcgrouped.cu
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/third-party/cutlass/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_blockscaled_rcgrouped.cu
```

Reference for Blackwell grouped-GEMM and MoE grouped-GEMM launch structure, grouping
metadata, and profiler harness patterns. Useful for validating that we keep grouped
compute and do not accidentally create a slow per-expert GEMM loop.

```text
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/third-party/cutlass/examples/36_gather_scatter_fusion/gather_scatter_fusion.cu
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/third-party/cutlass/examples/59_ampere_gather_scatter_conv/ampere_gather_scatter_conv.cu
```

Secondary references for older gather/scatter coordinate handling and validation ideas.

### DeepGEMM

```text
/home/kevinni/AsymGEMM-SFT/third_party/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh
/home/kevinni/AsymGEMM-SFT/third_party/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh
/home/kevinni/AsymGEMM-SFT/third_party/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh
/home/kevinni/AsymGEMM-SFT/third_party/DeepGEMM/deep_gemm/include/deep_gemm/layout/mega_moe.cuh
/home/kevinni/AsymGEMM-SFT/third_party/DeepGEMM/deep_gemm/include/deep_gemm/epilogue/sm100_store_cd.cuh
/home/kevinni/AsymGEMM-SFT/third_party/DeepGEMM/deep_gemm/include/deep_gemm/common/tma_copy.cuh
/home/kevinni/AsymGEMM-SFT/third_party/DeepGEMM/tests/test_mega_moe.py
```

Reference for SM100 pipeline organization, TMA copy utility style, epilogue factoring,
and MoE layout/scheduler design. DeepGEMM is especially useful for understanding how a
high-performance MoE kernel separates dispatch, compute, and combine. It is not a direct
replacement because our weights are CPU-resident AsymGEMM weights and our target is
training/backward route placement.

### ktransformers / SGLang local third-party kernels

```text
/home/kevinni/AsymGEMM-SFT/third_party/ktransformers/third_party/sglang/sgl-kernel/csrc/moe/prepare_moe_input.cu
/home/kevinni/AsymGEMM-SFT/third_party/ktransformers/third_party/sglang/sgl-kernel/csrc/moe/moe_align_kernel.cu
/home/kevinni/AsymGEMM-SFT/third_party/ktransformers/third_party/sglang/sgl-kernel/csrc/moe/moe_sum.cu
/home/kevinni/AsymGEMM-SFT/third_party/ktransformers/third_party/sglang/sgl-kernel/csrc/moe/moe_sum_reduce.cu
/home/kevinni/AsymGEMM-SFT/third_party/ktransformers/third_party/sglang/sgl-kernel/csrc/moe/moe_fused_gate.cu
/home/kevinni/AsymGEMM-SFT/third_party/ktransformers/third_party/sglang/sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu
/home/kevinni/AsymGEMM-SFT/third_party/ktransformers/third_party/sglang/sgl-kernel/csrc/moe/cutlass_moe_helper.cu
/home/kevinni/AsymGEMM-SFT/third_party/ktransformers/third_party/sglang/sgl-kernel/python/sgl_kernel/fused_moe.py
```

Reference for route preprocessing, token/expert alignment, top-k combine, and MoE test
contracts. Useful for checking whether our token-space scatter/gather semantics match
normal MoE combine behavior. Do not copy the inference-only assumptions or GPU-resident
weight assumptions into the training Asym path.

```text
/home/kevinni/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/cuda/moe/moe_topk_softmax_kernels.cu
/home/kevinni/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/bench/bench_moe_kernel.py
/home/kevinni/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/bench/bench_bf16_moe.py
```

Reference for local MoE benchmarking conventions and routing helper kernels.

### Liger-Kernel

```text
/home/kevinni/AsymGEMM-SFT/third_party/Liger-Kernel/src/liger_kernel/ops/fused_moe.py
/home/kevinni/AsymGEMM-SFT/third_party/Liger-Kernel/src/liger_kernel/ops/fused_moe_kernels.py
/home/kevinni/AsymGEMM-SFT/third_party/Liger-Kernel/benchmark/scripts/benchmark_fused_moe.py
/home/kevinni/AsymGEMM-SFT/third_party/Liger-Kernel/test/transformers/test_fused_moe.py
```

Reference for a clean MoE routing-metadata pipeline, token aggregation, memory-efficient
backward framing, and benchmark/test structure. Liger is useful for validation style and
route metadata semantics; it is not the kernel base because it uses Triton/GPU-resident
weights rather than the AsymGEMM CPU-weight SM100 path.

### ScatterMoE

```text
/home/kevinni/AsymGEMM-SFT/third_party/scattermoe/README.md
/home/kevinni/AsymGEMM-SFT/third_party/scattermoe/scattermoe/parallel_experts.py
/home/kevinni/AsymGEMM-SFT/third_party/scattermoe/scattermoe/mlp.py
/home/kevinni/AsymGEMM-SFT/third_party/scattermoe/scattermoe/kernels/ops.py
/home/kevinni/AsymGEMM-SFT/third_party/scattermoe/scattermoe/kernels/single.py
/home/kevinni/AsymGEMM-SFT/third_party/scattermoe/tests/test_mlp.py
```

First-class reference for MoE route-sorted metadata and placement contracts. The
important pieces are `flatten_sort_count()`, `ParallelLinear.forward/backward()`,
and `scatter2scatter()`:

```text
flatten_sort_count:
  flattened expert ids -> sorted expert ids, route permutation, expert offsets

scatter2scatter:
  x_grouped controls whether A is token-space or route-space
  y_grouped controls whether D is route-space or token/expanded-space
  the expert id range for each M tile is derived from sorted route metadata

ParallelLinear backward:
  shows which tensors training needs for gates, grouped grad_out, grouped input,
  expert dW, and dX
```

This is directly relevant to our route metadata and API contracts. It is not a
drop-in kernel base because it is Triton/GPU-resident-weight oriented, uses a
different mainloop, and still materializes route-expanded tensors in training
paths that SonicMoE explicitly tries to remove.

### SonicMoE

```text
/home/kevinni/AsymGEMM-SFT/third_party/sonic-moe/README.md
/home/kevinni/AsymGEMM-SFT/third_party/sonic-moe/assets/2026-04-22-sonicmoe-blackwell.md
/home/kevinni/AsymGEMM-SFT/third_party/sonic-moe/sonicmoe/functional/forward.py
/home/kevinni/AsymGEMM-SFT/third_party/sonic-moe/sonicmoe/functional/backward.py
/home/kevinni/AsymGEMM-SFT/third_party/sonic-moe/sonicmoe/functional/reduction_over_k_gather.py
/home/kevinni/AsymGEMM-SFT/third_party/sonic-moe/sonicmoe/functional/tile_scheduler.py
/home/kevinni/AsymGEMM-SFT/third_party/sonic-moe/sonicmoe/functional/triton_kernels/__init__.py
/home/kevinni/AsymGEMM-SFT/third_party/sonic-moe/tests/metadata_test.py
/home/kevinni/AsymGEMM-SFT/third_party/sonic-moe/tests/moe_test.py
/home/kevinni/AsymGEMM-SFT/third_party/sonic-moe/benchmarks/moe-cute.py
```

Most relevant algorithmic reference for this Qwen3 MoE work. SonicMoE's local
blog and code make the core memory target explicit: avoid caching or
materializing tensors with size `O(T*K*d)` such as gathered `X`, gathered `dO`,
down-projection `Y`, scattered `Y`, and `dY`. The specific lessons to carry into
this AsymGEMM design are:

```text
metadata:
  TC_topk_router_metadata_triton builds expert offsets, grouped route order,
  inverse route order, and x_gather_idx for Qwen3-style top-k routing.
  tests/metadata_test.py gives the exact invariants to copy into our unit tests.

forward aggregation:
  _router_forward uses token_gather_and_sum_varlen_K_triton to make each token
  own the final sum over its activated expert outputs.

down backward:
  _down_projection_backward_act uses a gathered-A grouped GEMM and a fused
  epilogue to compute dH, dS, and activation-backward outputs without cached Y
  or dY.

hardware lesson:
  the design depends on gather fusion, epilogue customization, L2 locality, and
  NCU-driven validation of memory traffic, not just lower PyTorch allocation
  counters.
```

SonicMoE is still not a direct implementation base. It assumes GPU-resident
weights through QuACK/CUTLASS/CuTeDSL grouped GEMM. Our target is
`asym_cpuadamwds|recomp-off-full-fg` with SM100 BF16 AsymGEMM CPU-resident
packed weights, so the AsymGEMM CPU-weight tile streaming and reuse order remain
non-negotiable.

### Online primary references checked

```text
https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html
https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/52_hopper_gather_scatter_fusion/gather_gemm.hpp
https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/52_hopper_gather_scatter_fusion/scatter_epilogue.hpp
```

CUTLASS confirms the standard high-performance GEMM structure: operands are staged
through shared memory/iterators or fragments before tensor-core MMA. This supports the
Stage 2 design choice: a gathered left operand should be staged into the same shared
memory layout consumed by the existing SM100 AsymGEMM mainloop, rather than launching a
separate gather kernel. The gather/scatter example also makes the placement-transform
boundary explicit: gather belongs at operand loading and scatter belongs in the epilogue.

```text
https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm_block_scaled.cu
https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_grouped.cu
```

CUTLASS's Blackwell grouped example is SM100 grouped/TMA oriented. It reinforces the
constraint that grouped scheduling must remain device/kernel-side and must not become a
Python loop over experts.

```text
https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
```

Nsight Compute's profiling guide explicitly points memory-stall and shared-memory-bank
issues back to Memory Workload Analysis. Therefore every kernel stage must include NCU
inspection of memory sectors, replay/stalls, shared-memory behavior, atomics, and
tensor-core utilization before e2e conclusions.

```text
https://github.com/deepseek-ai/DeepGEMM
```

DeepGEMM is useful as an SM100/MoE pipeline reference, but it is not a drop-in
implementation because this work must preserve AsymGEMM's CPU-resident packed-weight
path and training backward semantics.

```text
https://github.com/shawntan/scattermoe
https://github.com/Dao-AILab/sonic-moe
https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/2026-04-22-sonicmoe-blackwell.md
```

ScatterMoE and SonicMoE are the most relevant MoE-specific references. ScatterMoE
is the cleanest compact reference for route-sorted `scatter2scatter` semantics.
SonicMoE is the strongest reference for the activation-memory target and for the
Blackwell gather/scatter aggregation tradeoff. Their kernels are not copied
directly because neither one is an AsymGEMM CPU-weight kernel.

```text
https://raw.githubusercontent.com/sgl-project/sglang/main/sgl-kernel/csrc/moe/moe_align_kernel.cu
https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/src/liger_kernel/ops/fused_moe.py
https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/src/liger_kernel/ops/fused_moe_kernels.py
```

SGLang/ktransformers and Liger are useful for route alignment, top-k combine, and
route-metadata test structure. They do not replace the AsymGEMM kernel path because
their MoE kernels assume GPU-resident weights and a different training/inference
contract.

### Reference-derived constraints

The references above resolve the main design ambiguity:

```text
Use gather/scatter as GEMM placement transforms, not as pre/post global tensors.
Keep grouped scheduling inside one routed kernel launch; never replace it with Python
expert loops or many small GEMMs.
Preserve AsymGEMM's CPU-weight loop order: load one B/weight tile, reuse it across
route-row tiles, then advance the K/N tile.
Keep routing metadata construction outside the kernel and use the existing repo route
metadata helpers for tests and e2e paths.
Separate scheduler/mainloop/epilogue responsibilities so each routed variant changes
only the A loader or D placement it actually owns.
Treat atomic scatter-add as a first implementation hypothesis, not as proven best:
validate it against SonicMoE's gather-and-sum evidence with NCU before accepting it.
```

Concrete lessons:

```text
CUTLASS gather/scatter examples:
  Gather/scatter belongs at iterator/epilogue placement boundaries. Do not materialize
  a route-space tensor just to immediately scatter or gather it.

CUTLASS Blackwell grouped/MoE examples:
  Grouped work stays in the device-side grouped launch/scheduler. Per-expert host loops
  are a regression, not an implementation shortcut.

DeepGEMM:
  Reuse the separation between scheduling, layout, mainloop, and epilogue. Do not copy
  its kernels directly because its weights are GPU-resident and this project streams
  CPU-resident packed Asym weights.

ScatterMoE:
  Use flatten_sort_count/scatter2scatter as route metadata and grouped placement
  references. Its x_grouped/y_grouped contract is exactly the kind of API boundary
  this design needs. Do not inherit the route-expanded training tensor ownership.

SonicMoE:
  Use it as the main memory-owner reference. The target is to remove O(T*K*d)
  tensors by gathering at kernel runtime and fusing output placement/reduction into
  GEMM boundaries. Its token-owned gather-and-sum aggregation is a serious fallback
  if NCU shows atomic scatter-add is the wrong Blackwell epilogue choice.

SGLang/ktransformers and Liger:
  Use their route sorting, reverse mapping, combine, and benchmark/test structure as
  sanity checks. Do not inherit inference-only assumptions or GPU-resident-weight
  assumptions.
```

## Current Blocker

The current fast Qwen3 MoE fine-grained path is grouped, but its output contract is
route-space:

```text
grouped GEMM returns [R,N]
Python then index_add_ scatters [R,N] into [M,N]
```

Important current sites:

```text
asym_gemm/training/qwen3_moe_finegrained.py
  _scatter_routes_add_(): consumes already-materialized [R,N]
  _route_grad_from_tokens(): materializes grad_routes [R,H]
  down_base forward: _base_forward(...) -> output [R,H]
  gate/up backward dX: _base_dx(...) -> grad_packed [R,H]

asym_gemm/training/frozen_linear.py
  _asym_grouped_bf16_nt(): allocates d = torch.empty((R,N))

csrc/apis/gemm.hpp
  m_grouped_bf16_asym_gemm_nt_contiguous(): requires d.shape == [R,N]

asym_gemm/include/asym_gemm/impls/sm100_bf16_asym_gemm.cuh
  epilogue stores via contiguous TMA to tensor_map_cd at (m_idx,n_idx)
```

The existing AsymGEMM epilogue cannot be "pointed" at token-space output. It uses TMA
stores/reduce-adds into a contiguous `[R,N]` tensor. Route-aware placement needs:

```text
token = token_indices[route_row]
out[token, n] += routing_weight[route_row] * acc(route_row, n)
```

Multiple route rows map to the same token row, so this is an add/reduction. It needs
atomic add or a dedicated route reduction strategy. TMA store cannot perform arbitrary
per-row scatter with token indices.

## Required New Native Primitives

The three AsymGEMM primitives below are required to remove the base expert `[R,H]`
tensors while preserving grouped compute. They are necessary but not always sufficient:
LoRA-B and LoRA-dX can also create `[R,H]`, but they are not the first implementation
target. Implement the base routed kernels first. Only implement the LoRA routed helpers
later if profiling artifacts show LoRA-owned `[R,H]` tensors are a meaningful peak
owner after the base fix.

## Risk Register After Code Exploration

Important clarification: the new routed kernels must not use expert loops, block-expert
loops, or per-expert small GEMMs. The earlier block-expert path is a legacy diagnostic
and mitigation path only. It is a risk because it already exists in the repo and could
be accidentally enabled or compared as if it were the final routed-kernel design. It is
not part of the target implementation.

```text
Current legacy fallback:
  asym_gemm/training/qwen3_moe_finegrained.py
    _down_scatter_block_experts()
    _expert_blocks()
    down_base forward block loop
    gate/up backward block loops

Target routed-kernel path:
  ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0
  one grouped routed kernel launch per logical base operation
  no Python loop over experts
  no block-expert loop
  no many-small-GEMM decomposition
```

| Risk | Code/search evidence | Status after exploration | Required guard/validation |
| --- | --- | --- | --- |
| Accidentally using the block-expert mitigation | `qwen3_moe_finegrained.py` has `_expert_blocks()` and block loops in down/gate/up. `FrozenLinearStats` has `qwen3_moe_finegrained_down_scatter_block_experts`, `qwen3_moe_finegrained_down_scatter_blocks`, and `qwen3_moe_finegrained_down_scatter_max_block_rows`. The profile script also propagates `ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS`. | De-risked as a design issue: the target path explicitly disables it. Still an integration risk if a run label hides the fallback. | For every routed stage, export `ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0`; require recorded config to match; require those three block stats to stay `0`; require no block profile ranges; require NCU launch count not scale with expert blocks. |
| Treating existing contiguous AsymGEMM epilogue as scatter-capable | `sm100_bf16_asym_gemm.cuh` stores through contiguous `tensor_map_cd` TMA store/reduce-add. That cannot directly express `out[token_indices[r], n] += ...`. | Real kernel work remains. A wrapper-only change will not solve this. | Fork the epilogue for routed store/gather semantics. NCU must prove the old `[R,H]` global write/read is gone and the new store path is not pathological. |
| Breaking the AsymGEMM CPU-weight streaming order | Current Asym kernels stream a CPU B tile once and reuse it across the route-row tile loop. ScatterMoE/SonicMoE references assume GPU-resident weights, so they cannot be copied blindly. | Partially de-risked by making the loop-order invariant explicit. Still a kernel implementation risk. | NCU must show B/CPU-weight traffic does not scale with route-row tile count or token count beyond the old grouped AsymGEMM expectation. Reject any implementation that reloads CPU B per route block. |
| Remaining `[R,H]` is LoRA-owned, not base-owned | Current code can still create `down_delta [R,H]`, LoRA `grad_2d [R,H]`, `gate_lora_dx [R,H]`, `up_lora_dx [R,H]`, and mixed `grad_packed [R,H]`. | Scoped but not eliminated by base kernels. This must not be misdiagnosed as base-kernel failure. | Artifact names must split owners: `base_down_routes`, `lora_down_delta`, `lora_grad_routes`, `lora_gate_dx`, `lora_up_dx`, `grad_packed_base_or_lora`. Implement LoRA routed helpers only if route=111 proves LoRA owns a meaningful remaining peak. |
| fp32 token accumulator shifts the peak | At `s80000,b8`, `[R,H]` bf16 is about 19.5 GiB; fp32 `[M,H]` is about 4.9 GiB. | Mostly de-risked by size math, but lifetime must be tight. | Stage artifacts must show one token-space accumulator live, no duplicate scratch, and immediate release/cast after use. If peak merely moves to duplicated scratch, fix lifetime before next stage. |
| Padding/metadata mismatch | `_pad_grouped_input_for_asym()` pads route rows to Asym block boundaries; `_group_metadata_tensors()` converts cumulative offsets to pair offsets. | Real integration risk. The routed kernels need the same group semantics while skipping invalid padded rows in the epilogue/gather. | Unit tests must cover empty experts, one-token experts, non-128 row counts, all tokens routed to one expert, and random top-k. Compare against the old route-space path before e2e. |
| Weighting semantics mismatch | Qwen3 paths distinguish output-weighted down forward/backward and input-weighted gate/up dX. | De-risked by explicit API args and tests, not by inspection alone. | Each routed kernel test must run weighted and unweighted cases. E2E config must record input/output weighting assumptions. |
| Reference repos copied too literally | ScatterMoE has useful `flatten_sort_count` and `scatter2scatter` contracts; SonicMoE has metadata and token-owned gather/sum reduction. Both use GPU-resident weights and different kernel constraints. | De-risked as reference scope: use them for routing/reduction contracts, not as direct Asym kernel code. | Keep Asym CPU-weight streaming and SM100 BF16 constraints. If atomics are bad, evaluate SonicMoE-style token-owned gather/sum as the reduction strategy, not the legacy block-expert loop. |
| Config/path collision with dense full-fg | `profile_lora_lf_test_source.sh` enables Qwen3 MoE fine-grained only for `Qwen3-30B-A3B` and `recomp-off-full-fg`; dense full-fg has separate behavior. | Mostly de-risked by script validation. Still needs new routed flags recorded. | Routed flags default to `0`, appear in run labels/artifacts, and never change dense-model code paths. Treat missing or stale labels as inconclusive. |

### Primitive 1: grouped base forward scatter-add

Explicit kernel goal:

```text
Remove the down-base forward route-space output [R,H] while keeping one grouped
AsymGEMM-style computation over route-sorted expert rows. The kernel must write the
computed hidden columns directly into token-space [M,H] using route-aware add semantics.
```

What this kernel must not do:

```text
- allocate [R,H]
- loop over experts from Python
- use ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS or any block-expert fallback
- replace grouped AsymGEMM with many small GEMMs
- change LoRA handling in this stage
```

Purpose:

```text
down expert base forward
act [R,I] @ down_weight[e] [H,I].T -> directly add into out [M,H]
```

API shape for this Qwen3-specific SM100 BF16 primitive:

```cpp
void qwen3_moe_bf16_down_forward_scatter_add_(
    Tensor a,                  // CUDA bf16 [R,K]
    Tensor b_cpu,              // CPU pinned bf16 [E,N,K], existing packed Asym weight
    Tensor out,                // CUDA fp32 [M,N], pre-zeroed token accumulator
    Tensor offsets,            // CUDA int32 route offsets
    Tensor experts,            // CUDA int32 expert ids
    Tensor token_indices,      // CUDA int64 [R]
    Tensor routing_weights,    // CUDA bf16/fp32 [R]
    int64_t list_size,
    bool weighted,
    string compiled_dims);
```

This is not a generic `m_grouped_bf16_asym_gemm_nt_*` API. Keep it in the Qwen3 MoE
native namespace and hard-fail outside SM100 BF16.

Semantics:

```text
for each route row r in grouped expert order:
  e = expert_for_row(r)
  token = token_indices[r]
  scale = weighted ? routing_weights[r] : 1
  for n in H:
    out[token,n] += scale * dot(a[r,:], b_cpu[e,n,:])
```

Correct usage:

```python
scattered = torch.zeros((num_tokens, hidden_dim), device=..., dtype=input_dtype)
base_out_fp32 = torch.zeros((num_tokens, hidden_dim), device=act_stage.device, dtype=torch.float32)

# Stage 1 changes only the base down path. LoRA-B forward remains on the old path
# unless Stage 4 artifacts prove LoRA-owned [R,H] is a meaningful peak owner.

down_forward_scatter_add_(
    layer.down_base,
    act_stage,
    base_out_fp32,
    offsets,
    experts,
    token_indices,
    routing_weights,
    weighted=ctx.output_weighted,
)
scattered.add_(base_out_fp32.to(dtype=scattered.dtype))
del base_out_fp32
```

Do not allocate:

```text
output = _base_forward(... )          # [R,H]
_scatter_routes_add_(scattered, output, ...)
```

Mandatory NCU validation focus:

```text
Check that the removed [R,H] global store is really gone.
Check token-space atomic/store traffic: coalescing by hidden column, atomic sectors,
L2 write pressure, replay/stall reasons.
Check route metadata loads: token_indices and routing_weights should be small relative
to A/B/GEMM traffic and should not dominate.
Check tensor-core utilization stayed plausible relative to the old grouped GEMM.
```

### Primitive 2: grouped base backward gather-left

Explicit kernel goal:

```text
Remove the down-base backward route-space grad input [R,H] by treating
grad_token[token_indices[r], :] * routing_weight[r] as a virtual left operand for the
grouped AsymGEMM. The output [R,I] remains real because it feeds the expert activation
backward path.
```

What this kernel must not do:

```text
- allocate grad_routes [R,H]
- hide the [R,H] allocation under a different tensor name
- route through the generic frozen-linear backward if that backward materializes [R,H]
- use ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS or any block-expert fallback
- change LoRA backward unless the optional LoRA stage is explicitly enabled
```

Purpose:

```text
down expert base dX
virtual grad_routes [R,H] @ down_weight[e] [H,I] -> grad_act [R,I]
```

The left operand is logically `[R,H]`, but it must not be materialized. Load it from
token-space on demand:

```text
grad_routes[r,h] = grad_token[token_indices[r], h] * routing_weights[r]
```

API shape for this Qwen3-specific SM100 BF16 primitive:

```cpp
void qwen3_moe_bf16_down_dx_gather_left_(
    Tensor grad_token,         // CUDA bf16 [M,K]; wrapper casts if incoming grad is fp32
    Tensor b_cpu,              // CPU pinned bf16 existing down weight
    Tensor out,                // CUDA bf16 [R,N]
    Tensor offsets,            // CUDA int32
    Tensor experts,            // CUDA int32
    Tensor token_indices,      // CUDA int64 [R]
    Tensor routing_weights,    // CUDA bf16/fp32 [R]
    int64_t list_size,
    bool weighted,
    string compiled_dims);
```

This native API expects bf16 `grad_token`. If the recompute backward produces fp32
`grad_output`, the Python wrapper must cast token-space `[M,H]` to bf16 before the
native call. Do not cast or allocate route-space `[R,H]`.

For down backward:

```text
grad_token: grad_output [M,H]
b_cpu: down_base host weight with the same logical transpose used by existing
       `_grouped_base_dx(..., transpose_b=True)`
out: grad_act [R,I]
weighted: ctx.output_weighted
```

Correct usage:

```python
# Do not create grad_2d = _route_grad_from_tokens(...), which is [R,H].
grad_token = grad_output.reshape(num_tokens, hidden_dim)
if grad_token.dtype != torch.bfloat16:
    grad_token = grad_token.to(torch.bfloat16)
grad_token = grad_token.contiguous()

grad_act = _base_dx_gather_left(
    layer.down_base,
    grad_token,
    offsets,
    experts,
    token_indices,
    routing_weights,
    weighted=ctx.output_weighted,
    output_dim=intermediate_dim,
)
```

Do not allocate:

```text
grad_2d = grad_output.index_select(0, token_indices)   # [R,H]
grad_act = _base_dx(layer.down_base, grad_2d, ...)
```

Mandatory NCU validation focus:

```text
Check gathered grad_token load efficiency: global load sectors, L2 hit rate, replay,
and whether token_indices causes badly uncoalesced access.
Check shared-memory A-tile staging: bank conflicts and smem throughput.
Check tensor-core utilization after replacing the A-side TMA load with gathered loads.
Check that no separate [R,H] allocation appears in CUDA memory traces or artifacts.
```

### Primitive 3: grouped base dX scatter-add

Explicit kernel goal:

```text
Remove gate-base and up-base backward dX route-space outputs [R,H]. Each grouped
AsymGEMM computes route-row contributions but directly accumulates them into
token-space grad_hidden [M,H].
```

What this kernel must not do:

```text
- allocate gate_dx_routes [R,H]
- allocate up_dx_routes [R,H]
- combine gate and up into a new larger temporary
- use ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS or any block-expert fallback
- force blockwise/per-expert GEMM decomposition
```

Purpose:

```text
gate/up expert base dX
grad_gate_or_up [R,I] @ gate_or_up_weight[e] [I,H] -> directly add into grad_hidden [M,H]
```

API shape for this Qwen3-specific SM100 BF16 primitive:

```cpp
void qwen3_moe_bf16_gateup_dx_scatter_add_(
    Tensor grad_expert,        // CUDA bf16 [R,K], K=I
    Tensor b_cpu,              // CPU pinned bf16 existing gate/up weight
    Tensor grad_hidden,        // CUDA fp32 [M,H] token accumulator
    Tensor offsets,            // CUDA int32
    Tensor experts,            // CUDA int32
    Tensor token_indices,      // CUDA int64 [R]
    Tensor routing_weights,    // CUDA bf16/fp32 [R]
    int64_t list_size,
    bool weighted,
    string compiled_dims);
```

For gate/up backward:

```text
grad_expert: grad_gate_stage or grad_up_stage [R,I]
b_cpu: gate_base or up_base transposed for dX
grad_hidden: token-space input gradient [M,H]
weighted: ctx.input_weighted
```

Correct usage:

```python
grad_hidden_fp32 = torch.zeros((num_tokens, hidden_dim), device=..., dtype=torch.float32)

gateup_dx_scatter_add_(
    gate_base,
    grad_gate_stage,
    grad_hidden_fp32,
    offsets,
    experts,
    token_indices,
    routing_weights,
    weighted=ctx.input_weighted,
)

gateup_dx_scatter_add_(
    up_base,
    grad_up_stage,
    grad_hidden_fp32,
    offsets,
    experts,
    token_indices,
    routing_weights,
    weighted=ctx.input_weighted,
)

grad_hidden.add_(grad_hidden_fp32.to(dtype=grad_hidden.dtype))
```

Do not allocate:

```text
grad_packed = _base_dx(gate_base, grad_gate_stage, ...)  # [R,H]
grad_packed.add_(_base_dx(up_base, grad_up_stage, ...))  # [R,H]
grad_hidden.index_add_(0, token_indices, grad_packed)
```

Mandatory NCU validation focus:

```text
Check token-space accumulation traffic for both gate and up calls.
Check whether atomics serialize because many top-k routes hit the same token row.
Check hidden-column vectorization and store coalescing.
Check that the kernel remains grouped and that launch count does not become one launch
per expert.
Check that base-owned gate/up [R,H] tensors disappear from e2e artifacts.
```

## Conditional LoRA Routed Helpers

The three base AsymGEMM primitives are the required first pass. They remove base
`[R,H]` tensors. They do not remove all possible `[R,H]` route-space tensors. Qwen3 MoE
LoRA paths can still materialize:

```text
down_delta = grouped_expert_lora(down_low_rank, down_lora_B)      # [R,H]
gate_lora_dx = grouped_expert_lora(d_s_gate, gate_lora_A.T)       # [R,H]
up_lora_dx = grouped_expert_lora(d_s_up, up_lora_A.T)             # [R,H]
grad_2d for down LoRA-B backward if built from token grads         # [R,H]
```

Do not implement these LoRA helpers up front unless artifacts already prove LoRA is the
peak owner. After the base routed kernels are active, inspect the peak snapshot and
live activation details. If LoRA-owned `[R,H]` tensors are not a meaningful peak owner,
skip this section.

If LoRA is a meaningful remaining peak owner, implement only the proven-needed helper
behind the LoRA flag. The base routed flags should stay at the Stage 3 effective
configuration (`fwd=1,gather=1,dx=1`). The master flag may remain `0` for isolated
artifact labels because per-kernel overrides are the staged comparison mechanism.

```text
ASYMM_QWEN3_MOE_ROUTE_LORA=1
```

### LoRA helper A: routed LoRA-B forward scatter-add

Purpose:

```text
down_low_rank [R,r] @ down_lora_B[e] [H,r].T -> add into scattered [M,H]
```

API:

```cpp
void grouped_lora_b_forward_scatter_add(
    Tensor low_rank,           // CUDA bf16 [R,r]
    Tensor lora_b,             // CUDA bf16 [E,H,r]
    Tensor out,                // CUDA fp32 [M,H] token accumulator
    Tensor offsets,
    Tensor experts,
    Tensor token_indices,
    Tensor routing_weights,
    float scale,
    bool weighted);
```

Use this instead of:

```python
down_delta = grouped_expert_lora(down_low_rank, down_lora_B, ...)
_scatter_routes_add_(scattered, down_delta, ...)
```

### LoRA helper B: routed LoRA-B backward from token grad

Purpose:

```text
read grad_token[token_indices[r], h] directly
produce dS_down [R,r] and grad_down_lora_B [E,H,r]
```

API:

```cpp
void grouped_lora_b_backward_from_tokens(
    Tensor grad_token,         // CUDA bf16 [M,H]
    Tensor low_rank,           // CUDA bf16 [R,r]
    Tensor lora_b,             // CUDA bf16 [E,H,r]
    Tensor dS,                 // CUDA bf16 [R,r]
    Tensor grad_b,             // CUDA bf16 or fp32 [E,H,r], match existing LoRA grad dtype
    Tensor offsets,
    Tensor experts,
    Tensor token_indices,
    Tensor routing_weights,
    float scale,
    bool weighted);
```

This is a routed variant of the existing grouped LoRA-B backward kernel. It must not
require `grad_out_cpu [R,H]`.

### LoRA helper C: routed LoRA dX scatter-add

Purpose:

```text
dS_gate [R,r] @ gate_lora_A[e] [r,H] -> add into grad_hidden [M,H]
dS_up   [R,r] @ up_lora_A[e]   [r,H] -> add into grad_hidden [M,H]
```

API:

```cpp
void grouped_lora_dx_scatter_add(
    Tensor dS,                 // CUDA bf16 [R,r]
    Tensor lora_a_t,           // CUDA bf16 [E,H,r] = lora_A.transpose(-1, -2).contiguous()
    Tensor grad_hidden,        // CUDA fp32 [M,H] token accumulator
    Tensor offsets,
    Tensor experts,
    Tensor token_indices,
    Tensor routing_weights,
    float scale,
    bool weighted);
```

Use this instead of:

```python
gate_lora_dx = grouped_expert_lora(d_s_gate, gate_lora_A.transpose(-1, -2), ...)
grad_packed.add_(gate_lora_dx)
```

## Kernel Implementation Notes

### Do not modify the dense path

This work is Qwen3 MoE-only. Keep dense full-fg behavior intact.

### New entry points, shared AsymGEMM core

The three base primitives are new native entry points and may compile to distinct CUDA
kernel symbols, but they must not be three unrelated from-scratch GEMM implementations.
They should be routed variants of the existing SM100 BF16 AsymGEMM core:

```text
shared:   grouped scheduler, CPU-weight TMA path, B tile smem residency/reuse,
          UMMA/tensor-memory mainloop, barrier/pipeline structure

variant:  A-side loader only for gather-left
variant:  epilogue placement only for scatter-add
variant:  API/wrapper shape checks and route metadata
```

The implementation should therefore factor a common routed AsymGEMM core in
`csrc/qwen3/qwen3_moe_routed_gemm.cu` / `.hpp`, parameterized by:

```text
left operand mode:
  contiguous route-space A
  gathered token-space A

output mode:
  contiguous route-space D
  token-space scatter-add D

projection:
  down forward
  down dX
  gate/up dX
```

Do not clone and separately evolve three full kernels unless the common template becomes
unmaintainable. If cloning is used temporarily for bring-up, Stage 0/1 must mark it as
technical debt and later refactor before performance claims.

### CPU-weight tile streaming/reuse invariant

AsymGEMM performance depends on amortizing CPU-pinned weight streaming. The B/weight
tile is expensive because it comes from CPU-resident packed weights. The kernel must
stream a B tile once, keep it in shared memory, and reuse it across all route-row/M tiles
assigned to that expert/N/K tile.

Required loop ordering:

```cuda
for block_k_iter in K_tiles:
    load_B_cpu_tile_to_smem_once(expert, n_tile, block_k_iter)

    for block_m_iter in expert_route_row_tiles:
        load_or_gather_A_tile_to_smem(block_m_iter, block_k_iter)
        umma(acc, smem_A, smem_B)
        epilogue_store_or_scatter(block_m_iter, n_tile)
```

Forbidden loop ordering:

```cuda
for block_m_iter in expert_route_row_tiles:
    for block_k_iter in K_tiles:
        load_B_cpu_tile_to_smem_again(...)   // pays CPU streaming cost repeatedly
        ...
```

This is a central difference from normal GPU-resident GEMMs. A GPU-native grouped GEMM
can often choose tile order mostly for occupancy/coalescing; an AsymGEMM CPU-weight
kernel must choose tile order to amortize CPU-to-GPU/TMA weight streaming.

NCU validation must explicitly check this invariant:

```text
B/weight TMA or CPU-read traffic per expert/N/K tile should not scale with the number
of M route-row tiles.
new routed kernels should have similar B-side traffic to the old contiguous AsymGEMM
for the same shape and flags.
if B traffic grows roughly proportional to route-row tile count, the loop order is wrong.
```

Allowed new files:

```text
csrc/qwen3/qwen3_moe_routed_gemm.cu
csrc/qwen3/qwen3_moe_routed_gemm.hpp
asym_gemm/training/qwen3_moe_routed_gemm.py
```

Allowed small shared edits:

```text
csrc/apis/qwen3_moe.hpp             # declare/register Qwen3 MoE routed wrappers
csrc/python_api.cpp                 # only if include/registration plumbing requires it
setup/CMake/build registration      # only what is needed to compile new CU
```

Avoid changing:

```text
asym_gemm/training/qwen3_moe_finegrained.py dense-neutral code except MoE call sites
asym_gemm/training/dense/offload paths
existing sm100_bf16_asym_gemm.cuh behavior for normal contiguous outputs
```

### Scatter-add epilogue strategy

The existing SM100 AsymGEMM kernel uses TMA store into contiguous output. A routed
scatter output cannot use that store path directly.

For the first correct implementation:

1. Reuse the existing grouped scheduler and mainloop tile structure. Fork only the
   A-load path for gather-left and the epilogue store path for scatter-add.
   Preserve the AsymGEMM B/CPU-weight tile streaming order: stream one B tile, reuse it
   across the inner M/route-row tile loop, then advance to the next K tile.
2. In the epilogue, after accumulator values are available for a tile, store via
   explicit global writes/atomics instead of TMA:

   ```text
   route_row = m_idx + local_m
   token_row = token_indices[route_row]
   value = accumulator(local_m, local_n) * optional_routing_weight
   atomic_add(out[token_row, n_idx + local_n], value)
   ```

3. Use atomic add because multiple route rows for the same token contribute to the same
   output row.
4. Do not introduce split-K in the first implementation. The existing Asym mainloop
   should finish the K accumulation for a tile before the routed epilogue writes it.
   If a future split-K variant is added, it needs a separate numerical and atomic
   validation plan.
5. Use fp32 output accumulators in this plan. Do not use bf16 global atomics in the
   first implementation. Document the expected extra `[M,H]` fp32 memory and prove the
   removed `[R,H]` owner is larger.

Do not write route-space scratch tiles to global memory. Tile-local shared memory is
fine; global `[R,H]` is the thing being removed.

SonicMoE design check:

```text
SonicMoE deliberately chooses grouped GEMM with contiguous route-space output plus a
token-owned gather-and-sum aggregation kernel for its GPU-resident-weight path. It
also documents scatter-fusion and atomic epilogue designs as alternatives that need
hardware validation.

For this AsymGEMM stage, atomic scatter-add is still the first implementation because
it can remove the full global [R,H] owner without adding a second global aggregation
tensor and it fits an AsymGEMM epilogue fork. That is a hypothesis, not a conclusion.
```

Acceptance consequence:

```text
If NCU shows atomic replay/serialization or synchronous global-store behavior dominates,
do not keep tuning atomics blindly.

First check whether SM100 async store/TMA-scatter style placement can be used without
breaking the AsymGEMM CPU-weight mainloop. If that is still poor, evaluate a
SonicMoE-style token-owned gather/sum variant only if it can avoid a full persistent
[R,H] route-space owner. A direct SonicMoE-style full [R,H] grouped output plus
aggregation is not acceptable as the memory-saving endpoint for this stage.
```

### Gather-left strategy

`gather_left` changes how A is loaded, not how D is stored:

```text
A_virtual[r,k] = grad_token[token_indices[r], k] * optional_weight[r]
D[r,n] = grouped_gemm(A_virtual, B[e])
```

Output remains `[R,I]` for down dX, which is acceptable for this stage. The key is to
avoid materializing `A_virtual [R,H]`.

For performance, load `token_indices` once per route row tile, then vector-load K
columns from `grad_token[token,k]`. The memory access is gathered by row but contiguous
across H for each route row.

### Route metadata assumptions

The route rows are already sorted/grouped by expert. Preserve that invariant:

```text
offsets/expert ids describe expert-contiguous route row ranges
token_indices is aligned with those sorted route rows
routing_weights is aligned with those sorted route rows
```

The Python wrapper must normalize metadata exactly like the existing
`_group_metadata_tensors()` helper in `asym_gemm/training/frozen_linear.py`:

```text
input offsets may be cumulative [num_groups + 1] or pair offsets [2 * num_groups]
native offsets_i32 must be CUDA int32 pair offsets [2 * num_groups]
native experts_i32 must be CUDA int32 with sentinel length list_size
list_size = experts_i32.numel()
native grid_y = list_size - 1
```

Do not rebuild or sort route metadata inside kernels. Validate shapes and contiguity at
the Python/C++ boundary.

Route-metadata validation must borrow the exact invariants from ScatterMoE and
SonicMoE:

```text
ScatterMoE invariants:
  sorted route rows are expert-contiguous
  sorted_scattered_idxs/token_indices maps each grouped row back to its token/top-k slot
  expert_offsets boundaries match the sorted expert ids

SonicMoE invariants:
  expert_frequency_offset[-1] == R
  expert_frequency.sum() == R
  each expert offset range contains only that expert
  route permutation and inverse route permutation reconstruct identity when the inverse
  is produced
  x_gather_idx or token_indices is aligned with grouped route order
  non-power-of-two top-k, empty experts, all-same-expert, and single-token cases pass
```

If the current Asym path does not need `s_reverse_scatter_idx`, do not add it to the
runtime API just for symmetry. Still add unit-test-only inverse permutation checks by
constructing the inverse from the emitted `token_indices`/route order and verifying it
matches SonicMoE's metadata semantics.

### Dtypes and numerical checks

Use bf16 tensor-core operands to match the current BF16 Asym path. The first
scatter-add implementation must use fp32 token-space output accumulators for correctness
and debuggability. Do not implement or validate bf16 global atomics in the first pass.

```text
scatter-add kernels:
  inputs: bf16
  token-space accumulator: fp32 [M,H]
  caller casts/adds back to bf16 only after the routed kernel returns

gather-left kernel:
  grad_token source: bf16 [M,H]
  if incoming grad_output is fp32, wrapper casts [M,H] to bf16 before the native call
  output grad_act: bf16 [R,I]
```

Tiny parity tests must compare against the old route-space implementation:

```python
old = torch.zeros((M,H), device="cuda", dtype=dtype)
tmp = grouped_old(...)                  # [R,H]
old.index_add_(0, token_indices, tmp * weights[:, None])

new_fp32 = torch.zeros((M,H), device="cuda", dtype=torch.float32)
routed_new(..., new_fp32, ...)

assert_close(new_fp32.to(old.dtype), old, atol=..., rtol=...)
```

Set tolerances for fp32 accumulation cast back to bf16. Do not require bitwise equality.

### Token-space fp32 accumulator lifetime

The fp32 `[M,H]` scratch is intentional for scatter-add correctness, but it must be
owned tightly:

```text
allocate one fp32 token accumulator per logical token-space result
zero it once
accumulate all enabled base scatter-add contributions into that scratch
cast/add back to the caller dtype once
release it before staging the next large route-space tensor
```

Do not allocate one fp32 `[M,H]` scratch per routed kernel call if two calls contribute
to the same logical result. Stage 3 gate/up dX should use one `grad_hidden_accum_fp32`
for both gate and up base scatter-adds. Stage 1 down forward should use one
`base_out_fp32` only for the base down contribution and release it immediately after it
is added into the returned hidden output.

## Python Wiring

Create a small wrapper module:

```text
asym_gemm/training/qwen3_moe_routed_gemm.py
```

Suggested wrapper functions:

```python
def down_forward_scatter_add_(base, act, out_token, offsets, experts,
                              token_indices, routing_weights,
                              weighted: bool) -> None: ...

def down_dx_gather_left(base, grad_token, out_shape, offsets, experts,
                        token_indices, routing_weights,
                        weighted: bool) -> torch.Tensor: ...

def gateup_dx_scatter_add_(base, grad_expert, grad_hidden, offsets, experts,
                           token_indices, routing_weights,
                           weighted: bool) -> None: ...

def lora_b_forward_scatter_add_(low_rank, lora_b, out, offsets, experts,
                                token_indices, routing_weights, *,
                                scale: float, weighted: bool) -> None: ...

def lora_b_backward_from_tokens(grad_token, low_rank, lora_b, offsets, experts,
                                token_indices, routing_weights, *,
                                scale: float, weighted: bool) -> tuple[Tensor, Tensor]: ...

def lora_dx_scatter_add_(dS, lora_a, grad_hidden, offsets, experts,
                         token_indices, routing_weights, *,
                         scale: float, weighted: bool) -> None: ...
```

The wrappers should:

1. check the feature flag;
2. check dtype/device/contiguity;
3. increment explicit Qwen3 MoE routed counters;
4. fall back to old code only when the flag is disabled, not silently after a native
   error;
5. record enough debug metadata to identify whether routed base and routed LoRA paths
   fired.

Suggested counters:

```text
qwen3_moe_routed_base_forward_scatter_calls
qwen3_moe_routed_base_gather_left_calls
qwen3_moe_routed_base_dx_scatter_calls
qwen3_moe_routed_lora_b_forward_scatter_calls
qwen3_moe_routed_lora_b_backward_from_tokens_calls
qwen3_moe_routed_lora_dx_scatter_calls
qwen3_moe_routed_route_space_h_tensors_avoided
```

## Fine-Grained MoE Call-Site Changes

All routed changes in `qwen3_moe_finegrained.py` must be gated by the effective flags
returned by `routed_kernel_flags()`. The master flag is only a convenience switch; the
stage plan intentionally uses per-kernel overrides with the master flag set to `0` so
artifacts can isolate each kernel.

```text
ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER
ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER
ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER
ASYMM_QWEN3_MOE_ROUTE_LORA
```

and should be active only for the `recomp-off-full-fg` MoE target path.

### Forward down

Current bad shape:

```text
down_delta [R,H]
output/down_base [R,H]
scatter to [M,H]
```

Target:

```text
scattered [M,H]
down base scatter-add directly into scattered
old down LoRA-B path may remain route-space until the Stage 4 profile gate
conditional down LoRA-B scatter-add only if LoRA is a meaningful peak owner
return scattered
```

Expected avoided tensors:

```text
down_base output [R,H]
down_lora_B output [R,H] only after conditional LoRA stage
```

### Backward down

Current bad shape:

```text
grad_2d = index_select(grad_output, token_indices)  # [R,H]
down LoRA-B backward reads grad_2d
down base dX reads grad_2d
```

Target:

```text
grad_token [M,H]
down base gather-left reads grad_token by token_indices and returns grad_act [R,I]
old down LoRA-B backward may still need route-space grad until the Stage 4 profile gate
conditional down LoRA-B backward reads grad_token by token_indices only if LoRA is a meaningful peak owner
```

Expected avoided tensors:

```text
base-owned grad_2d [R,H]
LoRA-owned grad_2d [R,H] only after conditional LoRA stage
```

### Backward gate/up dX

Current bad shape:

```text
gate base dX [R,H]
gate LoRA dX [R,H]
up base dX [R,H]
up LoRA dX [R,H]
grad_packed [R,H]
scatter to grad_hidden [M,H]
```

Target:

```text
grad_hidden [M,H]
gate base dX scatter-add into grad_hidden
up base dX scatter-add into grad_hidden
old gate/up LoRA dX may remain route-space until the Stage 4 profile gate
conditional gate/up LoRA dX scatter-add only if LoRA is a meaningful peak owner
```

Expected avoided tensors:

```text
gate base dX [R,H]
up base dX [R,H]
base-owned grad_packed [R,H]
gate/up LoRA dX [R,H] only after conditional LoRA stage
```

## Stage Plan

### SM100 BF16 scope lock

This plan is only for SM100 BF16 Qwen3 MoE routed kernels.

Do not implement or touch:

```text
SM90 routed kernels
FP8 routed kernels
FP4 routed kernels
generic frozen-linear routed kernels
dense full-fg kernels
blockwise/per-expert GEMM fallbacks
```

Target/e2e profiling must resolve to:

```text
RUNS backend == asym_cpuadamwds for target/e2e profiling
```

Every native routed API must hard-fail unless:

```text
device arch major == 10
left operand dtype == torch.bfloat16
packed Asym weight dtype == torch.bfloat16
base.backend == "asym"
base.precision == "bf16"
ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS == 0 for target profiling
```

The `base.backend == "asym"` line is an internal module check, not permission to use
the plain `asym` RUNS backend for final comparisons.

The first scatter-add implementation uses fp32 token-space accumulation, not bf16
atomics. That makes correctness and NCU interpretation simpler. A bf16 atomic variant
can be a later optimization only after fp32 accumulation proves the memory owner is
removed.

### Stage 0: behavior freeze and plumbing only

Goal: add feature flags, artifact labels, and validation scaffolding without changing
math.

Files/functions/classes:

```text
scripts/lf/profile_lora_lf_test_source.sh
scripts/lf/profile_lora_lf_test_both.sh
asym_gemm/training/qwen3_moe_finegrained.py
agent/impls/fused_grouped_scatter_validation.md
```

Required code behavior:

```python
# Pseudocode in qwen3_moe_finegrained.py
def _route_flag(name):
    if name in os.environ:
        return env_bool(name)
    return env_bool("ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM")

@dataclass(frozen=True)
class RoutedKernelFlags:
    fwd_scatter: bool
    down_dx_gather: bool
    gateup_dx_scatter: bool
    lora: bool
    accum_dtype: str
    debug: bool

def routed_kernel_flags():
    return RoutedKernelFlags(
        fwd_scatter=_route_flag("ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER"),
        down_dx_gather=_route_flag("ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER"),
        gateup_dx_scatter=_route_flag("ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER"),
        lora=env_bool("ASYMM_QWEN3_MOE_ROUTE_LORA"),
        accum_dtype=os.getenv("ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE", "fp32"),
        debug=env_bool("ASYMM_QWEN3_MOE_ROUTE_KERNEL_DEBUG"),
    )
```

Script labels must include:

```text
q3rt_fwd{0|1}_gather{0|1}_dx{0|1}_lora{0|1}_accfp32
```

Validation before Stage 1:

```bash
python -m py_compile asym_gemm/training/qwen3_moe_finegrained.py

export ROUTE_STAGE=route000_smoke
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__${ROUTE_STAGE}__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export RUN_NAME="q3moe_${ROUTE_RUN_ID}"
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_routed"
export OVERWRITE=false
export COLLECT_EXISTING=false
export RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false'
export ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM=0
export ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=0
export ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=0
export ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=0
export ASYMM_QWEN3_MOE_ROUTE_LORA=0
export ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32
export ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0
export UNSLOTH_GC_OUTER_HBM_EVERY_N=0
bash scripts/lf/profile_lora_lf_test_source.sh
```

Must prove:

```text
flags are recorded in command.txt/source_profile config
route=000 label is present
old behavior is unchanged
no blockwise counters fire
```

Risk/watch:

```text
If the shell scripts do not propagate a new env var into the run environment, artifacts
will look like route=000 even when the outer shell has route=100. Treat that as a
plumbing failure, not a kernel result.
```

### Stage 1: SM100 BF16 down-base forward scatter-add

Goal: remove only the down-base forward `[R,H]` route-space output.

Files/functions/classes:

```text
csrc/qwen3/qwen3_moe_routed_gemm.hpp
csrc/qwen3/qwen3_moe_routed_gemm.cu
csrc/apis/qwen3_moe.hpp
setup.py
asym_gemm/training/qwen3_moe_routed_gemm.py
asym_gemm/training/qwen3_moe_finegrained.py
tests/qwen3/test_qwen3_moe_routed_gemm.py
scripts/testing/profile_qwen3_moe_routed_gemm.py
```

Exact local call site to change:

```text
asym_gemm/training/qwen3_moe_finegrained.py
  _Qwen3MoeFinegrainedFunction.forward()
  current down-base path:
    output = _base_forward(layer, layer.down_base, act_stage, offsets, experts, part="down")
    _scatter_routes_add_(scattered, output, token_indices, routing_weights, ...)
```

Native API pseudocode:

```cpp
void qwen3_moe_bf16_down_forward_scatter_add_(
    Tensor act, Tensor weight_cpu, Tensor out_fp32,
    Tensor offsets_i32, Tensor experts_i32,
    Tensor token_indices_i64, optional<Tensor> routing_weights,
    int64_t list_size, bool weighted, string compiled_dims) {
  check_arch_sm100();
  check_bf16_cuda_contiguous(act);             // [R,I]
  check_bf16_cpu_pinned(weight_cpu);           // [E,H,I]
  check_fp32_cuda_contiguous(out_fp32);         // [M,H]
  check_i32_cuda_contiguous(offsets_i32, experts_i32);
  check_i64_cuda_contiguous(token_indices_i64);
  check(compiled_dims == "nk");

  // Use existing sm100_bf16_asym_gemm mainloop and asymScheduler.
  launch_sm100_bf16_grouped_scatter_add(
      act, weight_cpu, out_fp32, offsets_i32, experts_i32,
      token_indices_i64, routing_weights, list_size, weighted);
}
```

Kernel pseudocode:

```cuda
// Same scheduler/mainloop shape as sm100_bf16_asym_gemm_impl.
tile = scheduler.next_tile();  // grouped route rows and hidden columns
load A route rows act[r, k] using existing A path;
load B expert weight W_down[e, h, k] using existing CPU/TMA Asym path;
UMMA accumulate acc[route_m, hidden_n] in tensor memory;
copy accumulator tile to registers/shared-memory as current epilogue does;

// Replace contiguous TMA output store.
for each valid element (local_m, local_n) owned by epilogue threads:
    r = tile.route_m0 + local_m
    h = tile.hidden_n0 + local_n
    token = token_indices_i64[r]
    scale = weighted ? routing_weights[r] : 1.0f
    atomicAdd(&out_fp32[token * H + h], float(acc[local_m, local_n]) * scale)
```

Python wrapper pseudocode:

```python
def down_forward_scatter_add_(base, act, out_fp32, offsets, experts, token_indices, routing_weights, weighted):
    require base.backend == "asym" and base.precision == "bf16"
    require torch.cuda.get_device_capability(act.device)[0] == 10
    require out_fp32.dtype == torch.float32
    _C.qwen3_moe_bf16_down_forward_scatter_add_(
        act.contiguous(), base.host_weight.weight, out_fp32,
        offsets_i32, experts_i32, token_indices.long().contiguous(),
        routing_weights.contiguous(), list_size, weighted, "nk")
```

Integration pseudocode:

```python
if route_flags.fwd_scatter:
    act_stage = manager.stage(act_cpu, tag="moe.act_for_down_base")
    base_out_fp32 = torch.zeros((num_tokens, hidden_dim), device=act_stage.device, dtype=torch.float32)
    down_forward_scatter_add_(
        layer.down_base, act_stage, base_out_fp32, offsets, experts,
        token_indices, routing_weights, weighted=ctx.output_weighted)
    scattered.add_(base_out_fp32.to(dtype=scattered.dtype))
    del base_out_fp32
    manager.release_stage(act_stage, drop_cache=True)
else:
    output = _base_forward(...)
    _scatter_routes_add_(scattered, output, ...)
```

Memory/latency expectations:

```text
Removes one base-owned [R,H] global allocation/store/read.
Adds token-space fp32 [M,H] accumulation scratch if scattered is bf16.
At s80000: removes about 19.5 GiB [R,H] bf16; fp32 [M,H] scratch is about 4.9 GiB.
Launch count should stay one grouped base-down launch, not E launches.
Latency risk is atomics into token-space; NCU decides whether coalescing is acceptable.
```

Validation before Stage 2:

```bash
python -m pip install -e . --no-build-isolation
python -m pytest tests/qwen3/test_qwen3_moe_routed_gemm.py -q -s -k 'down_forward_scatter'

export ROUTE_STAGE=stage1_fwd_scatter_route100
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__${ROUTE_STAGE}__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export NCU_OUT_ROOT="${PWD}/artifacts/ncu/qwen3_moe_routed/${ROUTE_RUN_ID}"
test ! -e "${NCU_OUT_ROOT}" || { echo "refusing to overwrite ${NCU_OUT_ROOT}" >&2; exit 1; }
mkdir -p "${NCU_OUT_ROOT}"

CUDA_VISIBLE_DEVICES=0 ncu \
  --target-processes all \
  --set roofline \
  --kernel-name regex:qwen3_moe_.*down.*scatter.* \
  --launch-skip 10 \
  --launch-count 20 \
  --export "${NCU_OUT_ROOT}/q3_moe_stage1_down_fwd_scatter" \
  python scripts/testing/profile_qwen3_moe_routed_gemm.py --kernel fwd_scatter --M 8192 --top-k 8 --H 2048 --I 768 --E 128 --iters 50 --warmup 10 --weighted 1 --output-dir "${NCU_OUT_ROOT}/microbench_fwd_scatter"

export ROUTE_STAGE=route100_smoke
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__${ROUTE_STAGE}__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export RUN_NAME="q3moe_${ROUTE_RUN_ID}"
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_routed"
export OVERWRITE=false
export COLLECT_EXISTING=false
export RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false'
export ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM=0
export ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=0
export ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=0
export ASYMM_QWEN3_MOE_ROUTE_LORA=0
export ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32
export ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0
export UNSLOTH_GC_OUTER_HBM_EVERY_N=0
bash scripts/lf/profile_lora_lf_test_source.sh
```

Must prove:

```text
unit parity versus old _base_forward(...)->_scatter_routes_add_
NCU shows no [R,H] down-forward global output write
source profile label is route=100
old down-base forward [R,H] owner is absent or zero
no blockwise loop/counter is used
```

Real-workload validation before Stage 2:

```bash
export ROUTE_STAGE=route100_real
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__${ROUTE_STAGE}__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export RUN_NAME="q3moe_${ROUTE_RUN_ID}"
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_routed"
export OVERWRITE=false
export COLLECT_EXISTING=false
export RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false'
export ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM=0
export ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=0
export ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=0
export ASYMM_QWEN3_MOE_ROUTE_LORA=0
export ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32
export ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0
export UNSLOTH_GC_OUTER_HBM_EVERY_N=0
bash scripts/lf/profile_lora_lf_test_source.sh
```

The route=100 real run is required even if total peak HBM does not improve. The expected
signal is that the down-base forward `[R,H]` owner disappears in
`memory_live_activation_details.csv` / `peak_snapshot_attrib_allblocks.*`. If the real
s80000 run OOMs before writing complete artifacts, preserve the OOM directory and run
the largest non-OOM diagnostic length only to inspect attribution; do not replace the
real workload row with the smaller diagnostic row.

Risk/watch:

```text
fp32 token scratch may shift peak if [M,H] overlaps with LoRA down_delta. If peak does
not drop, inspect owner attribution before changing the kernel.
```

### Stage 2: SM100 BF16 down-base backward gather-left

Goal: remove only the base-owned down-backward `grad_routes [R,H]`.

Exact local call site to change:

```text
asym_gemm/training/qwen3_moe_finegrained.py
  _Qwen3MoeFinegrainedFunction.backward()
  current no-block down path:
    grad_2d = _route_grad_from_tokens(...)
    grad_act = _base_dx(layer, layer.down_base, grad_2d, offsets, experts, ...)
```

Native API pseudocode:

```cpp
void qwen3_moe_bf16_down_dx_gather_left_(
    Tensor grad_token, Tensor weight_cpu, Tensor grad_act,
    Tensor offsets_i32, Tensor experts_i32,
    Tensor token_indices_i64, optional<Tensor> routing_weights,
    int64_t list_size, bool weighted, string compiled_dims) {
  check_arch_sm100();
  check_bf16_cuda_contiguous(grad_token);      // [M,H]
  check_bf16_cpu_pinned(weight_cpu);           // down weight, transpose_b=true
  check_bf16_cuda_contiguous(grad_act);         // [R,I]
  check_i32_cuda_contiguous(offsets_i32, experts_i32);
  check_i64_cuda_contiguous(token_indices_i64);

  launch_sm100_bf16_grouped_gather_left(
      grad_token, weight_cpu, grad_act, offsets_i32, experts_i32,
      token_indices_i64, routing_weights, list_size, weighted);
}
```

Kernel pseudocode:

```cuda
tile = scheduler.next_tile();  // route rows x intermediate columns

// Replace A-side TMA because A is virtual:
// A_virtual[r, h] = grad_token[token_indices[r], h] * route_weight[r]
for each A tile element (local_m, local_k) loaded by copy threads:
    r = tile.route_m0 + local_m
    h = tile.hidden_k0 + local_k
    token = token_indices_i64[r]
    scale = weighted ? routing_weights[r] : 1.0f
    smem_A[local_m, local_k] = bf16(float(grad_token[token, h]) * scale)

load B down_weight[e]^T from CPU pinned weight using existing B/TMA Asym path;
UMMA accumulate;
store contiguous grad_act[r, i] to [R,I] using current contiguous epilogue;
```

Integration pseudocode:

```python
if route_flags.down_dx_gather:
    grad_act_base = down_dx_gather_left(
        layer.down_base, grad_output.reshape(num_tokens, hidden_dim),
        out_shape=(num_routes, intermediate_dim),
        offsets=offsets, experts=experts,
        token_indices=token_indices, routing_weights=routing_weights,
        weighted=ctx.output_weighted)
else:
    grad_routes_base = _route_grad_from_tokens(...)
    grad_act_base = _base_dx(layer, layer.down_base, grad_routes_base, ...)

# If LoRA still needs route-space grad, create grad_routes_lora separately.
if lora_needs_grad_routes:
    grad_routes_lora = _route_grad_from_tokens(...)
```

Memory/latency expectations:

```text
Removes base-owned [R,H] grad_routes.
Keeps [R,I] grad_act because the expert activation backward needs it.
Launch count stays one grouped down-dX launch.
Latency risk is noncoalesced grad_token gather; NCU must check L2 sectors/replay and
whether gathered A staging starves tensor cores.
```

Validation before Stage 3:

```bash
python -m pip install -e . --no-build-isolation
python -m pytest tests/qwen3/test_qwen3_moe_routed_gemm.py -q -s -k 'down_dx_gather'

export ROUTE_STAGE=stage2_down_dx_gather_route110
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__${ROUTE_STAGE}__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export NCU_OUT_ROOT="${PWD}/artifacts/ncu/qwen3_moe_routed/${ROUTE_RUN_ID}"
test ! -e "${NCU_OUT_ROOT}" || { echo "refusing to overwrite ${NCU_OUT_ROOT}" >&2; exit 1; }
mkdir -p "${NCU_OUT_ROOT}"

CUDA_VISIBLE_DEVICES=0 ncu \
  --target-processes all \
  --set roofline \
  --kernel-name regex:qwen3_moe_.*down.*gather.* \
  --launch-skip 10 \
  --launch-count 20 \
  --export "${NCU_OUT_ROOT}/q3_moe_stage2_down_dx_gather" \
  python scripts/testing/profile_qwen3_moe_routed_gemm.py --kernel down_dx_gather --M 8192 --top-k 8 --H 2048 --I 768 --E 128 --iters 50 --warmup 10 --weighted 1 --output-dir "${NCU_OUT_ROOT}/microbench_down_dx_gather"

export ROUTE_STAGE=route110_smoke
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__${ROUTE_STAGE}__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export RUN_NAME="q3moe_${ROUTE_RUN_ID}"
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_routed"
export OVERWRITE=false
export COLLECT_EXISTING=false
export RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false'
export ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM=0
export ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=1
export ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=0
export ASYMM_QWEN3_MOE_ROUTE_LORA=0
export ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32
export ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0
export UNSLOTH_GC_OUTER_HBM_EVERY_N=0
bash scripts/lf/profile_lora_lf_test_source.sh
```

Must prove:

```text
unit parity versus old _route_grad_from_tokens(...)->_base_dx(...)
NCU shows gathered-A traffic is understood and not pathological
source profile label is route=110
base-owned down grad_routes [R,H] is absent
any remaining grad_routes [R,H] is explicitly LoRA-owned
```

Real-workload validation before Stage 3:

```bash
export ROUTE_STAGE=route110_real
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__${ROUTE_STAGE}__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export RUN_NAME="q3moe_${ROUTE_RUN_ID}"
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_routed"
export OVERWRITE=false
export COLLECT_EXISTING=false
export RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false'
export ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM=0
export ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=1
export ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=0
export ASYMM_QWEN3_MOE_ROUTE_LORA=0
export ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32
export ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0
export UNSLOTH_GC_OUTER_HBM_EVERY_N=0
bash scripts/lf/profile_lora_lf_test_source.sh
```

The route=110 real run must show that the base-owned down backward `grad_routes [R,H]`
owner is gone. If peak HBM does not drop, identify whether the remaining peak is
gate/up base dX, LoRA, `[R,I]`, attention, allocator state, or fallback to old path.

Risk/watch:

```text
If grad_output is fp32, the wrapper casts token-space [M,H] to bf16 before the native
call. That may add a small [M,H] temporary and can change numerics versus an fp32 ideal.
Compare against the current bf16 Asym path tolerance, not an fp32 mathematical ideal.
```

### Stage 3: SM100 BF16 gate/up base dX scatter-add

Goal: remove base-owned gate and up dX `[R,H]` route-space outputs.

Exact local call sites to change:

```text
asym_gemm/training/qwen3_moe_finegrained.py
  _Qwen3MoeFinegrainedFunction.backward()
  current gate/up no-block and block-compatible paths:
    gate_dx = _base_dx(layer, gate_base, grad_gate_stage, ...)
    grad_hidden.index_add_(0, token_indices, gate_dx)
    up_dx = _base_dx(layer, up_base, grad_up_stage, ...)
    grad_hidden.index_add_(0, token_indices, up_dx)
```

Native API pseudocode:

```cpp
void qwen3_moe_bf16_gateup_dx_scatter_add_(
    Tensor grad_expert, Tensor weight_cpu, Tensor grad_hidden_fp32,
    Tensor offsets_i32, Tensor experts_i32,
    Tensor token_indices_i64, optional<Tensor> routing_weights,
    int64_t list_size, bool weighted, string compiled_dims) {
  check_arch_sm100();
  check_bf16_cuda_contiguous(grad_expert);      // [R,I]
  check_bf16_cpu_pinned(weight_cpu);            // gate or up [E,I,H] logical transpose
  check_fp32_cuda_contiguous(grad_hidden_fp32); // [M,H]
  launch_sm100_bf16_grouped_scatter_add(
      grad_expert, weight_cpu, grad_hidden_fp32, offsets_i32, experts_i32,
      token_indices_i64, routing_weights, list_size, weighted);
}
```

Integration pseudocode:

```python
if route_flags.gateup_dx_scatter and ctx.needs_input_grad[0]:
    grad_hidden_accum = torch.zeros((num_tokens, hidden_dim), device=grad_gate.device, dtype=torch.float32)
    gateup_dx_scatter_add_(gate_base, grad_gate_stage, grad_hidden_accum, offsets, experts, token_indices, routing_weights, ctx.input_weighted)
    gateup_dx_scatter_add_(up_base, grad_up_stage, grad_hidden_accum, offsets, experts, token_indices, routing_weights, ctx.input_weighted)

    # LoRA dX remains old route-space unless Stage 5 is enabled.
    if gate_lora_dx is not None:
        _scatter_routes_add_(grad_hidden_accum, gate_lora_dx, token_indices, routing_weights, weighted=ctx.input_weighted)
    if up_lora_dx is not None:
        _scatter_routes_add_(grad_hidden_accum, up_lora_dx, token_indices, routing_weights, weighted=ctx.input_weighted)
    grad_hidden.add_(grad_hidden_accum.to(dtype=grad_hidden.dtype))
else:
    old _base_dx(...)->index_add_ path
```

Memory/latency expectations:

```text
Removes gate-base [R,H] and up-base [R,H].
Uses two grouped launches per layer for gate and up, not per expert.
Does not fuse gate and up into a larger temporary.
Latency risk is atomics under top-k collisions; NCU must determine whether this is
acceptable before real e2e interpretation.
```

Validation before Stage 4:

```bash
python -m pip install -e . --no-build-isolation
python -m pytest tests/qwen3/test_qwen3_moe_routed_gemm.py -q -s -k 'gateup_dx_scatter'

export ROUTE_STAGE=stage3_gateup_dx_scatter_route111
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__${ROUTE_STAGE}__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export NCU_OUT_ROOT="${PWD}/artifacts/ncu/qwen3_moe_routed/${ROUTE_RUN_ID}"
test ! -e "${NCU_OUT_ROOT}" || { echo "refusing to overwrite ${NCU_OUT_ROOT}" >&2; exit 1; }
mkdir -p "${NCU_OUT_ROOT}"

CUDA_VISIBLE_DEVICES=0 ncu \
  --target-processes all \
  --set roofline \
  --kernel-name regex:qwen3_moe_.*gateup.*scatter.* \
  --launch-skip 10 \
  --launch-count 20 \
  --export "${NCU_OUT_ROOT}/q3_moe_stage3_gateup_dx_scatter" \
  python scripts/testing/profile_qwen3_moe_routed_gemm.py --kernel gateup_dx_scatter --M 8192 --top-k 8 --H 2048 --I 768 --E 128 --iters 50 --warmup 10 --weighted 1 --output-dir "${NCU_OUT_ROOT}/microbench_gateup_dx_scatter"

export ROUTE_STAGE=route111_smoke
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__${ROUTE_STAGE}__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export RUN_NAME="q3moe_${ROUTE_RUN_ID}"
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_routed"
export OVERWRITE=false
export COLLECT_EXISTING=false
export RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false'
export ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM=0
export ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=1
export ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_LORA=0
export ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32
export ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0
export UNSLOTH_GC_OUTER_HBM_EVERY_N=0
bash scripts/lf/profile_lora_lf_test_source.sh
```

Must prove:

```text
unit parity versus old _base_dx(...)->_scatter_routes_add_
NCU shows grouped launches, not per-expert launches
source profile label is route=111
base-owned gate/up [R,H] dX tensors are absent
latency is much closer to route=000 than blockwise 16/18
```

Real-workload validation before Stage 4:

```bash
export ROUTE_STAGE=route111_real_base_only
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__${ROUTE_STAGE}__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export RUN_NAME="q3moe_${ROUTE_RUN_ID}"
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_routed"
export OVERWRITE=false
export COLLECT_EXISTING=false
export RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false'
export ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM=0
export ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=1
export ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_LORA=0
export ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32
export ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0
export UNSLOTH_GC_OUTER_HBM_EVERY_N=0
bash scripts/lf/profile_lora_lf_test_source.sh
```

The route=111 real run must show all base-owned `[R,H]` route-space owners removed.
Only after this run should Stage 4 decide whether remaining `[R,H]` peak ownership is
LoRA-owned and worth Stage 5.

Risk/watch:

```text
If old LoRA dX still creates [R,H], base kernel success may not move e2e peak. Do not
mix that up with base scatter-add failure.
```

### Stage 4: base-only routed real-workload gate

Goal: decide from real artifacts whether base routed kernels are sufficient or LoRA
routed kernels are required.

Run serially. If the Stage 3 route=111 real run already exists as a fresh preserved
artifact, reuse it only as an input to the comparison table; do not rerun into the same
directory and do not overwrite it.

Run this route=111 command only if Stage 3 did not already produce a fresh
`route111_real_base_only` artifact:

```bash
export ROUTE_STAGE=route111_real_base_only
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__${ROUTE_STAGE}__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export RUN_NAME="q3moe_${ROUTE_RUN_ID}"
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_routed"
export OVERWRITE=false
export COLLECT_EXISTING=false
export RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false'
export ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM=0
export ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=1
export ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_LORA=0
export ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32
export ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0
export UNSLOTH_GC_OUTER_HBM_EVERY_N=0
bash scripts/lf/profile_lora_lf_test_source.sh
```

Must compare against:

```bash
export ROUTE_STAGE=baseline_superoffload_unsloth_real
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__${ROUTE_STAGE}__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export RUN_NAME="q3moe_${ROUTE_RUN_ID}"
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_routed"
export OVERWRITE=false
export COLLECT_EXISTING=false
export RUNS='q3-30b-a3b|1 ; superoffload_mem|unsloth|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false'
bash scripts/lf/profile_lora_lf_test_source.sh

export ROUTE_STAGE=baseline_superoffload_unsloth_off_real
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__${ROUTE_STAGE}__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export RUN_NAME="q3moe_${ROUTE_RUN_ID}"
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_routed"
export OVERWRITE=false
export COLLECT_EXISTING=false
export RUNS='q3-30b-a3b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false'
bash scripts/lf/profile_lora_lf_test_source.sh
```

Decision gate:

```text
If base-owned [R,H] tensors are gone and peak beats or meaningfully improves over
superoffload_mem|unsloth-off: skip Stage 5.

If base-owned [R,H] tensors are gone but LoRA-owned [R,H] is the remaining peak owner:
implement only the specific LoRA routed helper that removes that owner.

If base-owned [R,H] tensors are still present: fix wiring before adding LoRA kernels.
```

Risk/watch:

```text
s2048 can prove wiring, but it cannot prove real memory effectiveness. Stage 4 must use
s80000 or it is not accepted as the final decision.
```

### Stage 5: conditional SM100 BF16 LoRA routed helpers

Goal: implement only the LoRA route-space removal proven necessary by Stage 4. Do not
preemptively implement all LoRA kernels.

Allowed files:

```text
csrc/qwen3/qwen3_moe_routed_gemm.hpp
csrc/qwen3/qwen3_moe_routed_gemm.cu
csrc/apis/qwen3_moe.hpp
asym_gemm/training/qwen3_moe_routed_gemm.py
asym_gemm/training/qwen3_moe_finegrained.py
tests/qwen3/test_qwen3_moe_routed_gemm.py
scripts/testing/profile_qwen3_moe_routed_gemm.py
```

Possible helper pseudocode:

```cuda
// down LoRA-B forward scatter-add, only if down_delta [R,H] is peak owner.
for route row r and hidden h:
    val = dot(down_low_rank[r, rank], down_lora_B[expert, h, rank])
    atomicAdd(out_fp32[token_indices[r], h], val * lora_scale * route_weight[r])

// gate/up LoRA dX scatter-add, only if gate_lora_dx/up_lora_dx [R,H] is peak owner.
for route row r and hidden h:
    val = dot(dS_gate_or_up[r, rank], lora_A[expert, rank, h])
    atomicAdd(grad_hidden_fp32[token_indices[r], h], val * route_weight[r])
```

Validation:

```bash
python -m pytest tests/qwen3/test_qwen3_moe_routed_gemm.py -q -s -k 'lora and routed'

export ROUTE_STAGE=stage5_lora_route111_lora1
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__${ROUTE_STAGE}__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export NCU_OUT_ROOT="${PWD}/artifacts/ncu/qwen3_moe_routed/${ROUTE_RUN_ID}"
test ! -e "${NCU_OUT_ROOT}" || { echo "refusing to overwrite ${NCU_OUT_ROOT}" >&2; exit 1; }
mkdir -p "${NCU_OUT_ROOT}"

CUDA_VISIBLE_DEVICES=0 ncu \
  --target-processes all \
  --set roofline \
  --kernel-name regex:qwen3_moe_.*lora.* \
  --launch-skip 10 \
  --launch-count 20 \
  --export "${NCU_OUT_ROOT}/q3_moe_stage5_lora_routed" \
  python scripts/testing/profile_qwen3_moe_routed_gemm.py --kernel lora_selected --M 8192 --top-k 8 --H 2048 --I 768 --E 128 --rank 64 --iters 50 --warmup 10 --weighted 1 --output-dir "${NCU_OUT_ROOT}/microbench_lora_selected"

export ROUTE_STAGE=route111_lora1_real
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__${ROUTE_STAGE}__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export RUN_NAME="q3moe_${ROUTE_RUN_ID}"
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_routed"
export OVERWRITE=false
export COLLECT_EXISTING=false
export RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false'
export ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM=0
export ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=1
export ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_LORA=1
export ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32
export ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0
export UNSLOTH_GC_OUTER_HBM_EVERY_N=0
bash scripts/lf/profile_lora_lf_test_source.sh
```

Risk/watch:

```text
LoRA rank is much smaller than H/I, but LoRA route-space outputs can still be [R,H].
Only optimize LoRA if artifacts show those outputs are peak owners.
```

### Stage 6: final real workload validation

Goal: make the apples-to-apples final claim.

Required final rows:

```text
q3-30b-a3b|1 ; superoffload_mem|unsloth|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false
q3-30b-a3b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false
q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false
```

Final result is accepted only if:

```text
routed flags are present in artifact path and resolved config
runtime counters prove the routed kernels fired
legacy blockwise counters are zero:
  qwen3_moe_finegrained_down_scatter_block_experts == 0
  qwen3_moe_finegrained_down_scatter_blocks == 0
  qwen3_moe_finegrained_down_scatter_max_block_rows == 0
old route-space base [R,H] owners are absent from peak attribution
NCU reports exist for every enabled routed kernel
HBM is meaningfully lower than superoffload_mem|unsloth-off, or the remaining owner is
identified concretely and the stage is marked incomplete
latency is closer to route=000 grouped compute than to blockwise 16/18
```

## Validation Artifacts To Inspect

For every stage, inspect before interpreting:

```text
command.txt
train.log
source_profile.json or profile.json
runtime_counters.json
memory_breakdown_summary.json
memory_live_activation_details.csv
memory_actual_peak_breakdown.csv
peak_snapshot_attrib_allblocks.md/csv/json
process_memory.csv
```

Expected artifact signals:

```text
RUNS backend resolved to asym_cpuadamwds for target runs
the effective routed flags match the stage under test
  route=000: master=0 fwd=0 gather=0 dx=0
  route=100: master=0 fwd=1 gather=0 dx=0
  route=110: master=0 fwd=1 gather=1 dx=0
  route=111: master=0 fwd=1 gather=1 dx=1
ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0
UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=true
UNSLOTH_GC_OUTER_HBM_EVERY_N=0
old ASYMM_EXPERT_ACT_OFFLOAD path disabled for target
```

Treat a run as inconclusive if:

1. artifact label and resolved config disagree;
2. the new routed counters are missing;
3. old blockwise counters fire, especially any nonzero
   `qwen3_moe_finegrained_down_scatter_block_experts`,
   `qwen3_moe_finegrained_down_scatter_blocks`, or
   `qwen3_moe_finegrained_down_scatter_max_block_rows`;
4. dense full-fg counters are used to explain MoE behavior;
5. a partial profile is interpreted as a completed backward;
6. stale artifacts are reused;
7. `UNSLOTH_GC_OUTER_HBM_EVERY_N` is nonzero;
8. the run was concurrent with another GPU experiment.
9. the target run resolves to plain `asym` instead of `asym_cpuadamwds`.

## Expected Memory Movement

At s80000, each removed `[R,H]` bf16 tensor is about 19.5 GiB. Not all of that becomes
peak reduction because lifetimes overlap differently, but if the routed kernels are
working the peak snapshot should no longer show owners like:

```text
model.layers.*.mlp.experts.down_base        [5120000,2048]
down_delta / down_lora_B output             [5120000,2048]
grad_2d / routed grad output                [5120000,2048]
grad_packed                                 [5120000,2048]
gate_lora_dx / up_lora_dx                   [5120000,2048]
gate/up base dX                             [5120000,2048]
```

If HBM does not drop after all routed base and LoRA paths are active, do not conclude
the design is impossible. First identify the new peak owner from the artifacts. Likely
remaining owners are:

```text
[R,I] gate/up/act/grad tensors
LoRA rank tensors [R,r]
attention/norm saved activations
allocator fragmentation/reserved memory
CPUAdamW/LoRA staging
incorrect fallback to old route-space path
```

## Concrete Implementation Blueprint

This section is the implementation contract. Do not treat the earlier sections as a loose proposal once this section exists. The implementation must follow these boundaries unless an artifact proves that a specific boundary is wrong.

### Non-negotiable behavior

1. The dense-model `recomp-off-full-fg` path must not be modified for MoE fixes.
2. The existing generic `AsymGroupedFrozenLinearFunction` behavior must remain valid for all existing call sites.
3. New Qwen3 MoE route-aware kernels must be separately gated and separately labeled in artifacts.
4. Each new kernel must be independently disableable so we can compare:
   - old route-space grouped path,
   - only down forward scatter fused,
   - down forward plus down backward gather fused,
   - down forward plus down backward plus gate/up dX scatter fused.
5. `ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS` must stay `0` for the routed-kernel experiments unless the experiment is explicitly about blockwise fallback. Blockwise is a fallback, not the target.
6. `UNSLOTH_GC_OUTER_HBM_EVERY_N` must stay disabled for these experiments. It is not a principled fix for this issue.
7. Base routed kernels are the first target. LoRA routed kernels are implemented only if memory artifacts show that LoRA route-space tensors are still a meaningful peak contributor after the base route-space tensors are removed.
8. Every test, microbench, NCU capture, smoke profile, and real e2e profile must write to a unique artifact directory. Never overwrite or reuse a previous result directory, even if the flags are identical.
9. Every stage must be accepted from fresh artifacts for that exact stage, then compared against all earlier preserved stage artifacts. Do not infer effectiveness from memory counters alone; inspect the actual live activation and peak-owner artifacts.

### Required environment flags

Add these flags, defaulting to disabled:

```bash
# Master flag. If set to 1, enable every base routed kernel whose per-kernel
# override is unset.
ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM=0

# Stage 1: down projection base forward:
# [R,I] @ W_down[e].T is accumulated directly into token-space [M,H].
ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=0

# Stage 2: down projection base backward dX:
# virtual grad_routes[r, h] = grad_token[token_indices[r], h] * route_weight[r]
# is used as left operand without materializing [R,H].
ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=0

# Stage 3: gate/up projection base backward dX:
# [R,I] @ W_gate_or_up[e].T is accumulated directly into token-space grad_hidden [M,H].
ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=0

# Optional later stage. Must remain 0 until base artifacts prove LoRA route-space
# tensors matter at peak.
ASYMM_QWEN3_MOE_ROUTE_LORA=0

# This plan accepts only fp32 token-space accumulation.
# Other values should hard-fail until a separate bf16-atomic optimization stage exists.
ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32

# Debug assertions and CUDA synchronizes around routed kernels.
ASYMM_QWEN3_MOE_ROUTE_KERNEL_DEBUG=0
```

Flag resolution must be:

```python
def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")

def _route_flag(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is not None:
        return _env_flag(name)
    return _env_flag("ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM", False)
```

The profiling scripts must record these flags in both `command.txt` and the run directory label. Use a compact suffix such as:

```text
q3rt_fwd{0|1}_gather{0|1}_dx{0|1}_lora{0|1}_accfp32
```

Example label:

```text
asym_cpuadamwds__recomp-off-full-fg__q3rt_fwd1_gather0_dx0_lora0_accfp32
```

### Artifact retention and no-overwrite contract

This project needs staged comparisons. Artifact retention is part of correctness.

Hard rules:

1. Do not overwrite any output directory, NCU export, JSON/CSV summary, source profile,
   or LF profile directory from a previous run.
2. Do not use `COLLECT_EXISTING=true` for validation. It is for postprocessing old runs,
   not for accepting a new kernel stage.
3. Do not use `OVERWRITE=true` for validation.
4. Do not accept a run if the script prints `Skipping existing`.
5. Do not compare against a directory whose `command.txt`/resolved config does not
   exactly match the stage being evaluated.
6. Keep all stage artifacts so route `000`, `100`, `110`, `111`, and optional LoRA
   runs can be re-opened and compared later.

Every run must include a fresh run id:

```bash
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__${ROUTE_STAGE}__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
```

For LF e2e profiles, set both:

```bash
export RUN_NAME="q3moe_${ROUTE_RUN_ID}"
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_routed"
export OVERWRITE=false
export COLLECT_EXISTING=false
```

`RUN_NAME` is part of `config_root_path()` in `profile_lora_lf_test_source.sh` /
`profile_lora_lf_test_both.sh`, so it isolates the LF output path. If a future script
change removes `RUN_NAME` from the path, Stage 0 must restore that behavior before any
kernel validation.

Before launching an LF run, print and check the intended artifact root:

```bash
test -n "${RUN_NAME}"
test "${OVERWRITE}" = "false"
test "${COLLECT_EXISTING}" = "false"
```

For NCU and microbench artifacts, use a separate output root:

```bash
export NCU_OUT_ROOT="${PWD}/artifacts/ncu/qwen3_moe_routed/${ROUTE_RUN_ID}"
test ! -e "${NCU_OUT_ROOT}" || { echo "refusing to overwrite ${NCU_OUT_ROOT}" >&2; exit 1; }
mkdir -p "${NCU_OUT_ROOT}"
```

For the kernel microbench script, require an explicit output directory:

```bash
python scripts/testing/profile_qwen3_moe_routed_gemm.py ... --output-dir "${NCU_OUT_ROOT}/microbench"
```

The microbench script must fail if `--output-dir` already exists. It should write:

```text
command.txt
metrics.json
timing.csv
memory.csv
parity.json
```

Do not keep only the latest result. The validation tables must cite the concrete
artifact directory for every row.

### New files

Add exactly these new implementation files:

```text
asym_gemm/training/qwen3_moe_routed_gemm.py
csrc/qwen3/qwen3_moe_routed_gemm.hpp
csrc/qwen3/qwen3_moe_routed_gemm.cu
tests/qwen3/test_qwen3_moe_routed_gemm.py
scripts/testing/profile_qwen3_moe_routed_gemm.py
```

Allowed small registration edits:

```text
csrc/apis/qwen3_moe.hpp
csrc/python_api.cpp
setup.py
scripts/lf/profile_lora_lf_test_source.sh
scripts/lf/profile_lora_lf_test_both.sh
asym_gemm/training/qwen3_moe_finegrained.py
```

Do not add the routed kernels to the generic frozen-linear dispatch. These kernels are route-aware Qwen3 MoE operators, not a generic grouped GEMM API.

### Native API contract

Register the APIs under the existing Qwen3 MoE extension namespace, not under the generic GEMM namespace.

These APIs are SM100 BF16 only. They must not expose a generic architecture or dtype
contract. Do not add SM90, FP8, FP4, or generic grouped-frozen-linear support in this
work.

Every native entry point must start with the equivalent of:

```cpp
static void check_sm100_bf16_route_kernel(
    const torch::Tensor& left,
    const torch::Tensor& weight_cpu,
    const std::string& compiled_dims) {
  const auto arch_major = device_runtime->get_arch_major();
  TORCH_CHECK(arch_major == 10, "Qwen3 routed Asym kernels are SM100-only");
  TORCH_CHECK(left.is_cuda(), "left operand must be CUDA");
  TORCH_CHECK(left.scalar_type() == torch::kBFloat16, "left operand must be bf16");
  TORCH_CHECK(weight_cpu.device().is_cpu(), "packed Asym weight must be CPU resident");
  TORCH_CHECK(weight_cpu.is_pinned(), "packed Asym weight must be pinned CPU memory");
  TORCH_CHECK(weight_cpu.scalar_type() == torch::kBFloat16, "packed Asym weight must be bf16");
  TORCH_CHECK(compiled_dims == "nk", "first routed implementation supports compiled_dims='nk' only");
}
```

The Python wrapper must also check:

```python
if base.backend != "asym" or base.precision != "bf16":
    raise RuntimeError("Qwen3 routed kernels require AsymGroupedFrozenLinear backend='asym', precision='bf16'")
if torch.cuda.get_device_capability(act_or_grad.device)[0] != 10:
    raise RuntimeError("Qwen3 routed kernels are SM100-only")
```

Add declarations in `csrc/qwen3/qwen3_moe_routed_gemm.hpp`:

```cpp
#pragma once

#include <torch/extension.h>
#include <string>

namespace asym_gemm {
namespace qwen3_moe {

void qwen3_moe_bf16_down_forward_scatter_add_(
    torch::Tensor a,                 // [R, I], CUDA bf16, contiguous
    torch::Tensor b_cpu,             // [E, H_phys, I_phys], CPU pinned packed asym weight
    torch::Tensor out,               // [M, H], CUDA fp32, contiguous, zeroed by caller
    torch::Tensor offsets,           // [2 * num_groups], CUDA int32 pair offsets
    torch::Tensor experts,           // [list_size], CUDA int32 expert ids plus sentinel
    torch::Tensor token_indices,     // [R], CUDA int64 contiguous
    c10::optional<torch::Tensor> routing_weights, // [R], CUDA fp32/bf16 contiguous
    int64_t list_size,
    bool weighted,
    std::string compiled_dims);

void qwen3_moe_bf16_down_dx_gather_left_(
    torch::Tensor grad_token,        // [M, H], CUDA bf16, contiguous
    torch::Tensor b_cpu,             // [E, I_phys, H_phys] or packed transpose-compatible weight
    torch::Tensor out,               // [R, I], CUDA bf16, contiguous
    torch::Tensor offsets,
    torch::Tensor experts,
    torch::Tensor token_indices,     // [R], CUDA int64 contiguous
    c10::optional<torch::Tensor> routing_weights,
    int64_t list_size,
    bool weighted,
    std::string compiled_dims);

void qwen3_moe_bf16_gateup_dx_scatter_add_(
    torch::Tensor grad_expert,       // [R, I], CUDA bf16, contiguous
    torch::Tensor b_cpu,             // [E, H_phys, I_phys], CPU pinned packed asym weight
    torch::Tensor grad_hidden,       // [M, H], CUDA fp32, contiguous, already initialized
    torch::Tensor offsets,
    torch::Tensor experts,
    torch::Tensor token_indices,     // [R], CUDA int64 contiguous
    c10::optional<torch::Tensor> routing_weights,
    int64_t list_size,
    bool weighted,
    std::string compiled_dims);

} // namespace qwen3_moe
} // namespace asym_gemm
```

Register in `csrc/apis/qwen3_moe.hpp`:

```cpp
m.def(
    "qwen3_moe_bf16_down_forward_scatter_add_",
    &asym_gemm::qwen3_moe::qwen3_moe_bf16_down_forward_scatter_add_,
    "Qwen3 MoE down base forward: grouped asym GEMM with token-space scatter-add");

m.def(
    "qwen3_moe_bf16_down_dx_gather_left_",
    &asym_gemm::qwen3_moe::qwen3_moe_bf16_down_dx_gather_left_,
    "Qwen3 MoE down base dX: grouped asym GEMM with gathered token-space left operand");

m.def(
    "qwen3_moe_bf16_gateup_dx_scatter_add_",
    &asym_gemm::qwen3_moe::qwen3_moe_bf16_gateup_dx_scatter_add_,
    "Qwen3 MoE gate/up base dX: grouped asym GEMM with token-space scatter-add");
```

Add `csrc/qwen3/qwen3_moe_routed_gemm.cu` to `setup.py` CUDA extension sources.

### C++ validation code

Every API wrapper must validate shape and dtype before launching. Fail loudly; do not silently fall back in these kernels.

Use this structure:

```cpp
static void check_common_route_inputs(
    const torch::Tensor& offsets,
    const torch::Tensor& experts,
    const torch::Tensor& token_indices,
    const c10::optional<torch::Tensor>& routing_weights,
    int64_t R,
    int64_t list_size,
    bool weighted) {
  TORCH_CHECK(token_indices.is_cuda(), "token_indices must be CUDA");
  TORCH_CHECK(token_indices.scalar_type() == torch::kInt64,
              "token_indices must be int64; normalize in Python before calling");
  TORCH_CHECK(token_indices.is_contiguous(), "token_indices must be contiguous");
  TORCH_CHECK(token_indices.numel() == R, "token_indices must have length R");

  TORCH_CHECK(offsets.is_cuda() && experts.is_cuda(), "offsets and experts must be CUDA");
  TORCH_CHECK(offsets.is_contiguous() && experts.is_contiguous(), "offsets and experts must be contiguous");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt32 && experts.scalar_type() == torch::kInt32,
              "offsets and experts must be int32");
  TORCH_CHECK(list_size >= 2, "list_size must include at least one group and one sentinel");
  TORCH_CHECK(experts.numel() == list_size, "experts must have exactly list_size entries");
  TORCH_CHECK(offsets.numel() == 2 * (list_size - 1),
              "offsets must be pair offsets [2 * (list_size - 1)]");

  if (weighted) {
    TORCH_CHECK(routing_weights.has_value(), "weighted route kernel requires routing_weights");
    const auto& w = routing_weights.value();
    TORCH_CHECK(w.is_cuda(), "routing_weights must be CUDA");
    TORCH_CHECK(w.is_contiguous(), "routing_weights must be contiguous");
    TORCH_CHECK(w.numel() == R, "routing_weights must have length R");
    TORCH_CHECK(w.scalar_type() == torch::kFloat32 || w.scalar_type() == torch::kBFloat16,
                "routing_weights must be fp32 or bf16");
  }
}
```

Each wrapper must also check:

```cpp
TORCH_CHECK(a.is_cuda(), "left operand must be CUDA");
TORCH_CHECK(a.scalar_type() == torch::kBFloat16, "left operand must be bf16");
TORCH_CHECK(a.is_contiguous(), "left operand must be contiguous");

TORCH_CHECK(b_cpu.device().is_cpu(), "packed asym weight must be CPU resident");
TORCH_CHECK(b_cpu.is_pinned(), "packed asym weight must be pinned CPU memory");
TORCH_CHECK(b_cpu.scalar_type() == torch::kBFloat16, "packed asym weight must be bf16");

TORCH_CHECK(out.is_cuda(), "output must be CUDA");
TORCH_CHECK(out.is_contiguous(), "output must be contiguous");
TORCH_CHECK(compiled_dims == "nk",
            "first routed qwen3 moe kernels support compiled_dims='nk' only");
```

For scatter-add APIs, require fp32 token-space accumulation:

```cpp
TORCH_CHECK(out.scalar_type() == torch::kFloat32,
            "scatter-add token-space output must be fp32");
```

Do not pass an `accum_dtype` string into native C++. The first implementation has only
one native scatter accumulation mode: fp32. The Python wrapper may keep
`ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32` for labels and should hard-fail if it is set to
anything else, but native code should validate the actual output dtype rather than parse
a string knob.

For gather-left down dX, require bf16 route-space output:

```cpp
TORCH_CHECK(out.scalar_type() == torch::kBFloat16,
            "down dX gather-left output grad_act [R,I] must be bf16");
```

The Python wrapper can cast the final token-space scratch back to bf16 after the routed kernel if the caller needs bf16.

### Kernel 1: down forward scatter-add

Implementation goal:

```text
Delete the base down-forward [R,H] route-space output from the execution graph. Keep the
existing grouped AsymGEMM compute shape and replace only the final placement with
route-aware token-space scatter-add into [M,H].
```

Purpose:

```text
old:
  tmp_routes = grouped_asym_nt(act [R,I], W_down[e] [H,I])      # [R,H]
  tmp_routes *= routing_weights[:, None]
  out_tokens.index_add_(0, token_indices, tmp_routes)           # [M,H]

new:
  out_tokens.zero_()
  grouped_asym_nt_scatter_add(act, W_down, token_indices, routing_weights, out_tokens)
```

The new kernel removes the base down forward `[R,H]` allocation.

Host function:

```cpp
void qwen3_moe_bf16_down_forward_scatter_add_(
    torch::Tensor a,
    torch::Tensor b_cpu,
    torch::Tensor out,
    torch::Tensor offsets,
    torch::Tensor experts,
    torch::Tensor token_indices,
    c10::optional<torch::Tensor> routing_weights,
    int64_t list_size,
    bool weighted,
    std::string compiled_dims) {
  const int64_t R = a.size(0);
  const int64_t I = a.size(1);
  const int64_t M = out.size(0);
  const int64_t H = out.size(1);

  // validate all inputs here
  // require b_cpu logical shape [E, H_phys, I_phys]
  // require token_indices[r] in [0, M)

  launch_qwen3_moe_down_forward_scatter_add(
      a, b_cpu, out, offsets, experts, token_indices,
      routing_weights, list_size, weighted, compiled_dims,
      R, M, H, I);
}
```

CUDA implementation strategy:

1. Derive from the existing SM100 grouped AsymGEMM mainloop used by
   `sm100_bf16_asym_gemm_contiguous`. This is a routed variant of that kernel, not an
   independent GPU-native GEMM.
2. Keep the same expert scheduler:
   - `offsets[g]..offsets[g + 1]` are route rows for expert `experts[g]`.
   - route row `r` maps to token row `token_indices[r]`.
3. Keep the same A and B tile loading path, including the AsymGEMM B/CPU-weight tile
   reuse order: load the B tile once for a K tile, then iterate over all M/route-row
   tiles that can reuse that B tile.
4. Keep tensor-core accumulation unchanged.
5. Replace the contiguous TMA store epilogue with a token-space scatter-add epilogue.

Epilogue helper:

```cuda
template <typename scalar_t>
__device__ __forceinline__ float route_weight_or_one(
    const scalar_t* weights,
    int64_t r,
    bool weighted) {
  if (!weighted) {
    return 1.0f;
  }
  return static_cast<float>(weights[r]);
}

template <>
__device__ __forceinline__ float route_weight_or_one<__nv_bfloat16>(
    const __nv_bfloat16* weights,
    int64_t r,
    bool weighted) {
  if (!weighted) {
    return 1.0f;
  }
  return __bfloat162float(weights[r]);
}

__device__ __forceinline__ void atomic_add_fp32(
    float* out,
    int64_t row,
    int64_t col,
    int64_t stride,
    float value) {
  atomicAdd(out + row * stride + col, value);
}
```

Scatter epilogue shape:

```cuda
// Called after a CTA has produced a tile for route rows [m0, m0 + BM)
// and hidden columns [n0, n0 + BN).
template <typename WeightT>
__device__ void scatter_add_epilogue_fp32(
    float* out,                         // [M,H]
    int64_t out_stride_h,
    const int64_t* token_indices,        // [R]
    const WeightT* routing_weights,      // nullable
    bool weighted,
    int64_t route_m0,
    int64_t hidden_n0,
    int64_t R,
    int64_t H) {
  // Existing code already transfers accumulator tile from tensor memory to a
  // register/shared-memory representation before TMA store. Reuse that path.
  //
  // For every valid accumulator element acc[local_m, local_n]:
  int64_t r = route_m0 + local_m;
  int64_t h = hidden_n0 + local_n;
  if (r < R && h < H) {
    int64_t tok = token_indices[r];
    float w = route_weight_or_one(routing_weights, r, weighted);
    float v = acc_value_as_float(local_m, local_n) * w;
    atomic_add_fp32(out, tok, h, out_stride_h, v);
  }
}
```

Important: the first implementation should not use a bf16 output atomic. Use fp32 output scratch. The e2e path should cast once after all down base and down LoRA contributions are accumulated if the caller needs bf16.

Acceptance for kernel 1:

1. Unit parity with old `tmp_routes + index_add_`.
2. No allocation of base down `[R,H]`.
3. E2E artifact must show the base down forward route-space owner is gone or reduced to zero.
4. If peak does not drop, inspect peak owner attribution before making any conclusion.

### Kernel 2: down backward dX gather-left

Implementation goal:

```text
Delete the base down-backward grad_routes [R,H] tensor. Feed the grouped AsymGEMM from a
virtual gathered left operand backed by grad_token[token_indices], then produce only
the required [R,I] grad_act.
```

Purpose:

```text
old:
  grad_routes = grad_token.index_select(0, token_indices)       # [R,H]
  grad_routes *= routing_weights[:, None]
  grad_act = grouped_asym_nt(grad_routes [R,H], W_down[e].T)    # [R,I]

new:
  grad_act = grouped_asym_nt_gather_left(
      grad_token, W_down[e].T, token_indices, routing_weights)  # [R,I]
```

The new kernel removes the base-owned down backward `grad_routes [R,H]` allocation. LoRA may still materialize a route-space gradient until the optional LoRA stage; that must be named separately in artifacts.

Host function:

```cpp
void qwen3_moe_bf16_down_dx_gather_left_(
    torch::Tensor grad_token,
    torch::Tensor b_cpu,
    torch::Tensor out,
    torch::Tensor offsets,
    torch::Tensor experts,
    torch::Tensor token_indices,
    c10::optional<torch::Tensor> routing_weights,
    int64_t list_size,
    bool weighted,
    std::string compiled_dims) {
  const int64_t R = out.size(0);
  const int64_t I = out.size(1);
  const int64_t M = grad_token.size(0);
  const int64_t H = grad_token.size(1);

  // validate all inputs here
  // require b_cpu logical shape compatible with [E, I_phys, H_phys]

  launch_qwen3_moe_down_dx_gather_left(
      grad_token, b_cpu, out, offsets, experts, token_indices,
      routing_weights, list_size, weighted, compiled_dims,
      R, M, H, I);
}
```

CUDA implementation strategy:

The existing grouped asym GEMM mainloop expects a contiguous CUDA left operand. Here the left operand is virtual:

```text
A_virtual[r, h] = grad_token[token_indices[r], h] * routing_weight[r]
```

Do not materialize `A_virtual`.

Implementation:

1. Reuse the existing expert scheduler and B-side CPU/pinned asym TMA load.
2. Replace the A-side TMA load with a cooperative gather loader.
3. Store gathered A tile into the same shared-memory layout consumed by the existing tensor-core mainloop.
4. Keep tensor-core MMA and contiguous output store to `[R,I]`.

Do not move the B-side load into the M loop when replacing the A loader. Kernel 2 still
must stream each down-weight B tile once and reuse it across the gathered-A route-row
tiles. The only changed producer for `smem_A` is the gather-left loader.

Gather loader skeleton:

```cuda
template <typename GradT, typename WeightT>
__device__ void load_gathered_a_tile_to_smem(
    const GradT* __restrict__ grad_token,       // [M,H]
    int64_t grad_stride_h,
    const int64_t* __restrict__ token_indices,  // [R]
    const WeightT* __restrict__ routing_weights,
    bool weighted,
    int64_t route_m0,
    int64_t hidden_k0,
    int64_t R,
    int64_t H,
    SmemAView smem_a) {
  for (int elem = threadIdx.x; elem < BM * BK; elem += blockDim.x) {
    int local_m = elem / BK;
    int local_k = elem % BK;
    int64_t r = route_m0 + local_m;
    int64_t h = hidden_k0 + local_k;

    __nv_bfloat16 value = __float2bfloat16(0.0f);
    if (r < R && h < H) {
      int64_t tok = token_indices[r];
      float w = route_weight_or_one(routing_weights, r, weighted);
      float g = static_cast<float>(grad_token[tok * grad_stride_h + h]);
      value = __float2bfloat16(g * w);
    }
    smem_a.store(local_m, local_k, value);
  }
}
```

The native kernel expects bf16 `grad_token`. If the incoming autograd `grad_output` is
fp32, the Python wrapper casts token-space `[M,H]` to bf16 before calling this kernel.
If this causes too much numerical drift, document it against the current bf16 Asym path
tolerance; do not introduce a route-space `[R,H]` cast.

Acceptance for kernel 2:

1. Unit parity with old `index_select + grouped_asym_nt`.
2. No allocation of base-owned `grad_routes [R,H]`.
3. E2E artifact must show the down backward base route-space gradient owner is gone.
4. If LoRA still allocates `grad_routes [R,H]`, artifact naming must distinguish it from base. Do not claim kernel 2 failed unless the remaining tensor is base-owned.

### Kernel 3: gate/up backward dX scatter-add

Implementation goal:

```text
Delete base gate/up dX route-space outputs [R,H]. Keep grouped AsymGEMM for gate and up,
but write each contribution directly into token-space grad_hidden [M,H].
```

Purpose:

```text
old:
  gate_dx_routes = grouped_asym_nt(grad_gate [R,I], W_gate[e].T) # [R,H]
  gate_dx_routes *= routing_weights[:, None]
  grad_hidden.index_add_(0, token_indices, gate_dx_routes)

  up_dx_routes = grouped_asym_nt(grad_up [R,I], W_up[e].T)       # [R,H]
  up_dx_routes *= routing_weights[:, None]
  grad_hidden.index_add_(0, token_indices, up_dx_routes)

new:
  grouped_asym_nt_scatter_add(grad_gate, W_gate, token_indices, routing_weights, grad_hidden)
  grouped_asym_nt_scatter_add(grad_up,   W_up,   token_indices, routing_weights, grad_hidden)
```

The new kernel removes the base gate and base up dX route-space `[R,H]` allocations.

Host function:

```cpp
void qwen3_moe_bf16_gateup_dx_scatter_add_(
    torch::Tensor grad_expert,
    torch::Tensor b_cpu,
    torch::Tensor grad_hidden,
    torch::Tensor offsets,
    torch::Tensor experts,
    torch::Tensor token_indices,
    c10::optional<torch::Tensor> routing_weights,
    int64_t list_size,
    bool weighted,
    std::string compiled_dims) {
  const int64_t R = grad_expert.size(0);
  const int64_t I = grad_expert.size(1);
  const int64_t M = grad_hidden.size(0);
  const int64_t H = grad_hidden.size(1);

  // validate all inputs here

  launch_qwen3_moe_gateup_dx_scatter_add(
      grad_expert, b_cpu, grad_hidden, offsets, experts, token_indices,
      routing_weights, list_size, weighted, compiled_dims,
      R, M, H, I);
}
```

CUDA implementation strategy:

This is the same route-space scatter epilogue as kernel 1, but the logical projection is gate/up dX instead of down forward.

It must use the same AsymGEMM CPU-weight streaming/reuse order as kernel 1:

```text
stream gate/up B tile once for the expert/N/K tile
for each route-row tile using that B tile:
  load grad_gate or grad_up A tile
  run UMMA
  scatter-add directly into token-space grad_hidden
```

Do not implement this as M tile outermost with repeated CPU-weight tile loads.

Implementation must share the scatter epilogue helper with kernel 1:

```cuda
launch_grouped_asym_nt_scatter_add(
    /*left=*/grad_expert,
    /*weight=*/gate_or_up_b_cpu,
    /*out_token=*/grad_hidden,
    /*token_indices=*/token_indices,
    /*routing_weights=*/routing_weights,
    /*weighted=*/weighted,
    /*projection_name=*/"gate_dx" or "up_dx");
```

The Python call site should invoke the same native kernel twice:

```python
gateup_dx_scatter_add_(
    layer.gate_base, grad_gate, grad_hidden, offsets, experts, token_indices,
    routing_weights, weighted=ctx.input_weighted)
gateup_dx_scatter_add_(
    layer.up_base, grad_up, grad_hidden, offsets, experts, token_indices,
    routing_weights, weighted=ctx.input_weighted)
```

LoRA dX remains old-path until optional LoRA routed kernels are justified. If LoRA dX still materializes `[R,H]`, it must be named as LoRA-owned in attribution.

Acceptance for kernel 3:

1. Unit parity with old `grouped_asym_nt + index_add_`.
2. No allocation of base gate/up dX `[R,H]`.
3. E2E artifact must show base gate/up dX route-space tensors are gone.
4. Peak should decrease if those tensors were at the previous peak. If not, inspect the new peak before concluding.

### Python wrapper implementation

Create `asym_gemm/training/qwen3_moe_routed_gemm.py`:

```python
import os
from dataclasses import dataclass
from typing import Optional

import torch

import asym_gemm


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def route_flag(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is not None:
        return env_flag(name)
    return env_flag("ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM", False)


@dataclass(frozen=True)
class RoutedKernelFlags:
    fwd_scatter: bool
    down_dx_gather: bool
    gateup_dx_scatter: bool
    lora: bool
    accum_dtype: str
    debug: bool


def routed_kernel_flags() -> RoutedKernelFlags:
    return RoutedKernelFlags(
        fwd_scatter=route_flag("ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER"),
        down_dx_gather=route_flag("ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER"),
        gateup_dx_scatter=route_flag("ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER"),
        lora=env_flag("ASYMM_QWEN3_MOE_ROUTE_LORA", False),
        accum_dtype=os.environ.get("ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE", "fp32").strip().lower(),
        debug=env_flag("ASYMM_QWEN3_MOE_ROUTE_KERNEL_DEBUG", False),
    )


def _normalize_token_indices(token_indices: torch.Tensor) -> torch.Tensor:
    if token_indices.dtype != torch.long:
        token_indices = token_indices.to(torch.long)
    if not token_indices.is_contiguous():
        token_indices = token_indices.contiguous()
    return token_indices


def _routing_weights_or_none(routing_weights: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if routing_weights is None:
        return None
    if not routing_weights.is_contiguous():
        routing_weights = routing_weights.contiguous()
    return routing_weights


def _compiled_dims(base) -> str:
    compiled_dims = getattr(base, "compiled_dims", "nk")
    if compiled_dims != "nk":
        raise RuntimeError("Qwen3 routed kernels currently require compiled_dims='nk'")
    return compiled_dims


def _check_supported_base(base, device: torch.device) -> None:
    if getattr(base, "backend", None) != "asym" or getattr(base, "precision", None) != "bf16":
        raise RuntimeError("Qwen3 routed kernels require AsymGroupedFrozenLinear backend='asym', precision='bf16'")
    if torch.cuda.get_device_capability(device)[0] != 10:
        raise RuntimeError("Qwen3 routed kernels are SM100-only")


def _packed_cpu_weight(base) -> torch.Tensor:
    # Use the same HostWeight object used by AsymGroupedFrozenLinear.
    host_weight = getattr(base, "host_weight", None)
    weight = None if host_weight is None else getattr(host_weight, "weight", None)
    if weight is None:
        raise RuntimeError("Qwen3 routed kernel requires existing packed CPU asym weight")
    return weight


def _group_metadata_for_kernel(
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    # Mirrors asym_gemm.training.frozen_linear._group_metadata_tensors().
    if offsets.dim() != 1 or experts.dim() != 1:
        raise ValueError("offsets and experts must be 1D tensors")
    if experts.numel() < 2:
        raise ValueError("grouped metadata requires at least one group and a sentinel")
    num_groups = int(experts.numel() - 1)
    if offsets.numel() == experts.numel():
        starts = offsets[:-1]
        ends = offsets[1:]
        pair_offsets = torch.stack((starts, ends), dim=1).reshape(-1)
    elif offsets.numel() >= 2 * num_groups:
        pair_offsets = offsets[: 2 * num_groups]
    else:
        raise ValueError(
            "offsets must be cumulative [num_groups + 1] or pairs [2 * num_groups], "
            f"got offsets={offsets.numel()} experts={experts.numel()}"
        )
    offsets_i32 = pair_offsets.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
    experts_i32 = experts.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
    return offsets_i32, experts_i32, int(experts_i32.numel())


def down_forward_scatter_add_(
    base,
    act: torch.Tensor,
    out_token: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    token_indices: torch.Tensor,
    routing_weights: Optional[torch.Tensor],
    weighted: bool,
) -> None:
    flags = routed_kernel_flags()
    _check_supported_base(base, act.device)
    if flags.accum_dtype != "fp32":
        raise RuntimeError("Qwen3 routed scatter-add kernels currently require fp32 accumulation")
    if out_token.dtype != torch.float32:
        raise RuntimeError("down_forward_scatter_add_ requires fp32 token accumulator")
    offsets_i32, experts_i32, list_size = _group_metadata_for_kernel(offsets, experts, device=act.device)
    token_indices = _normalize_token_indices(token_indices)
    routing_weights = _routing_weights_or_none(routing_weights)
    asym_gemm._C.qwen3_moe_bf16_down_forward_scatter_add_(
        act,
        _packed_cpu_weight(base),
        out_token,
        offsets_i32,
        experts_i32,
        token_indices,
        routing_weights,
        list_size,
        bool(weighted),
        _compiled_dims(base),
    )
    if flags.debug:
        torch.cuda.synchronize()


def down_dx_gather_left(
    base,
    grad_token: torch.Tensor,
    out_shape: tuple[int, int],
    offsets: torch.Tensor,
    experts: torch.Tensor,
    token_indices: torch.Tensor,
    routing_weights: Optional[torch.Tensor],
    weighted: bool,
) -> torch.Tensor:
    _check_supported_base(base, grad_token.device)
    if grad_token.dtype != torch.bfloat16:
        grad_token = grad_token.to(torch.bfloat16)
    grad_token = grad_token.contiguous()
    offsets_i32, experts_i32, list_size = _group_metadata_for_kernel(offsets, experts, device=grad_token.device)
    token_indices = _normalize_token_indices(token_indices)
    routing_weights = _routing_weights_or_none(routing_weights)
    out = torch.empty(out_shape, device=grad_token.device, dtype=torch.bfloat16)
    asym_gemm._C.qwen3_moe_bf16_down_dx_gather_left_(
        grad_token,
        _packed_cpu_weight(base),
        out,
        offsets_i32,
        experts_i32,
        token_indices,
        routing_weights,
        list_size,
        bool(weighted),
        _compiled_dims(base),
    )
    if routed_kernel_flags().debug:
        torch.cuda.synchronize()
    return out


def gateup_dx_scatter_add_(
    base,
    grad_expert: torch.Tensor,
    grad_hidden: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    token_indices: torch.Tensor,
    routing_weights: Optional[torch.Tensor],
    weighted: bool,
) -> None:
    flags = routed_kernel_flags()
    _check_supported_base(base, grad_expert.device)
    if flags.accum_dtype != "fp32":
        raise RuntimeError("Qwen3 routed scatter-add kernels currently require fp32 accumulation")
    if grad_hidden.dtype != torch.float32:
        raise RuntimeError("gateup_dx_scatter_add_ requires fp32 token accumulator")
    offsets_i32, experts_i32, list_size = _group_metadata_for_kernel(offsets, experts, device=grad_expert.device)
    token_indices = _normalize_token_indices(token_indices)
    routing_weights = _routing_weights_or_none(routing_weights)
    asym_gemm._C.qwen3_moe_bf16_gateup_dx_scatter_add_(
        grad_expert,
        _packed_cpu_weight(base),
        grad_hidden,
        offsets_i32,
        experts_i32,
        token_indices,
        routing_weights,
        list_size,
        bool(weighted),
        _compiled_dims(base),
    )
    if flags.debug:
        torch.cuda.synchronize()
```

Do not repack weights in this wrapper. The wrapper must use `base.host_weight.weight`,
matching `AsymGroupedFrozenLinear` in `asym_gemm/training/frozen_linear.py`.

### Integration points in `qwen3_moe_finegrained.py`

Import:

```python
from asym_gemm.training.qwen3_moe_routed_gemm import (
    routed_kernel_flags,
    down_forward_scatter_add_,
    down_dx_gather_left,
    gateup_dx_scatter_add_,
)
```

At the beginning of the relevant forward/backward helper, compute once:

```python
route_flags = routed_kernel_flags()
```

#### Stage 1 integration: down base forward

Replace only the down-base route-space output path:

```python
if route_flags.fwd_scatter:
    base_out = torch.zeros((num_tokens, hidden_size), device=act.device, dtype=torch.float32)

    down_forward_scatter_add_(
        layer.down_base,
        act,
        base_out,
        offsets,
        experts,
        token_indices,
        routing_weights,
        weighted=ctx.output_weighted,
    )

    hidden_out.add_(base_out.to(dtype=hidden_out.dtype))
else:
    down_routes = _base_forward(layer.down_base, act, offsets, experts, ...)
    _scatter_routes_add_(hidden_out, down_routes, token_indices, routing_weights, ...)
```

Do not change LoRA forward in this stage. If LoRA still produces a route-space `[R,H]`, leave it as-is and attribute it as LoRA-owned.

#### Stage 2 integration: down base backward dX

Replace only the base `grad_routes -> base_dx` path:

```python
if route_flags.down_dx_gather:
    grad_token = grad_output.reshape(num_tokens, hidden_dim)
    grad_act_base = down_dx_gather_left(
        layer.down_base,
        grad_token,
        out_shape=(num_routes, intermediate_size),
        offsets=offsets,
        experts=experts,
        token_indices=token_indices,
        routing_weights=routing_weights,
        weighted=ctx.output_weighted,
    )
else:
    grad_routes_base = _route_grad_from_tokens(grad_output, token_indices, routing_weights, ...)
    grad_act_base = _base_dx(layer.down_base, grad_routes_base, offsets, experts, ...)
```

If LoRA backward still needs token-gathered route gradients, create a separately named tensor:

```python
grad_routes_lora = _route_grad_from_tokens(...)
```

Do not reuse the name `grad_routes` for both base and LoRA in the routed-kernel path. The profiler attribution must make it clear whether a remaining `[R,H]` came from base or LoRA.

#### Stage 3 integration: gate/up base dX

Replace only the base gate/up dX route-space output:

```python
if route_flags.gateup_dx_scatter:
    grad_hidden_accum = torch.zeros_like(grad_hidden, dtype=torch.float32)

    gateup_dx_scatter_add_(
        layer.gate_base,
        grad_gate,
        grad_hidden_accum,
        offsets,
        experts,
        token_indices,
        routing_weights,
        weighted=ctx.input_weighted,
    )
    gateup_dx_scatter_add_(
        layer.up_base,
        grad_up,
        grad_hidden_accum,
        offsets,
        experts,
        token_indices,
        routing_weights,
        weighted=ctx.input_weighted,
    )

    grad_hidden.add_(grad_hidden_accum.to(grad_hidden.dtype))
else:
    gate_dx = _base_dx(layer.gate_base, grad_gate, offsets, experts, ...)
    _scatter_routes_add_(grad_hidden, gate_dx, token_indices, routing_weights, ...)

    up_dx = _base_dx(layer.up_base, grad_up, offsets, experts, ...)
    _scatter_routes_add_(grad_hidden, up_dx, token_indices, routing_weights, ...)
```

If existing code already accumulates other contributions into `grad_hidden`, do not allocate a second full fp32 accumulator unnecessarily. Prefer to initialize the fp32 accumulator once at the start of the backward helper when `route_flags.gateup_dx_scatter` is enabled, add all base scatter contributions into it, then cast/add once.

### Unit tests

Add `tests/qwen3/test_qwen3_moe_routed_gemm.py`.

Every test must compare the new kernel against the old route-space expression, not against a separately written mathematical formula.

Test data generator:

```python
from asym_gemm.training.moe import build_contiguous_route_metadata, make_dense_group_metadata


def make_route_case(M=128, H=256, I=128, E=8, top_k=4, dtype=torch.bfloat16):
    # Use the same metadata construction as the real MoE path. Do not hand-roll sorted
    # offsets/token_indices in ordinary parity tests.
    topk_indices = torch.randint(0, E, (M, top_k), device="cuda", dtype=torch.long)
    topk_weights = torch.rand((M, top_k), device="cuda", dtype=torch.float32)

    route_meta = build_contiguous_route_metadata(topk_indices, topk_weights, num_experts=E)
    offsets, experts = make_dense_group_metadata(route_meta.expert_offsets, num_groups=E, device=torch.device("cuda"))
    token_indices = route_meta.token_indices.contiguous()
    routing_weights = route_meta.routing_weights.contiguous()

    R = int(token_indices.numel())
    act = torch.randn((R, I), device="cuda", dtype=dtype)
    grad_hidden = torch.randn((M, H), device="cuda", dtype=dtype)
    return act, grad_hidden, offsets, experts, token_indices, routing_weights
```

Only use hand-rolled metadata in narrow edge-case tests that intentionally corrupt or
stress metadata. For empty-expert coverage, construct `topk_indices` so at least one
expert id is absent and still pass through `build_contiguous_route_metadata()` and
`make_dense_group_metadata()`. For many-routes-to-same-token coverage, set several
columns of `topk_indices[fixed_token, :]` and validate the same helpers produce the
route order used by the kernel.

Required cases:

```text
test_route_metadata_matches_scattermoe_expert_offsets
test_route_metadata_matches_sonic_inverse_permutation_invariants
test_route_metadata_handles_non_power_of_two_topk
test_route_metadata_handles_empty_experts
test_route_metadata_handles_all_same_expert
test_route_metadata_handles_single_token
test_down_forward_scatter_matches_old_path_weighted
test_down_forward_scatter_matches_old_path_unweighted
test_down_forward_scatter_handles_empty_experts
test_down_forward_scatter_handles_many_routes_to_same_token
test_down_dx_gather_left_matches_old_path_weighted
test_down_dx_gather_left_matches_old_path_unweighted
test_gateup_dx_scatter_matches_old_path_weighted
test_gateup_dx_scatter_matches_old_path_unweighted
test_routed_flags_do_not_change_old_path_when_disabled
test_no_base_route_space_allocation_when_kernel_enabled
```

The metadata tests are not optional. They prevent false kernel failures caused by
wrong route order. Use ScatterMoE's `flatten_sort_count()` contract and SonicMoE's
`tests/metadata_test.py` invariants as the reference behavior:

```text
expert offsets sum to R
each expert range contains only that expert
token_indices/x_gather_idx maps grouped row -> source token
constructed inverse permutation recovers identity
top-k values such as 1, 3, 7, 8, 10, and 16 are covered
empty experts and heavily imbalanced routing are covered
```

Tolerance:

```python
torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
```

Memory assertion for the no-allocation tests:

```python
torch.cuda.reset_peak_memory_stats()
run_old_path()
old_peak = torch.cuda.max_memory_allocated()

torch.cuda.reset_peak_memory_stats()
run_new_path()
new_peak = torch.cuda.max_memory_allocated()

removed_bytes = R * H * torch.finfo(torch.bfloat16).bits // 8
assert new_peak <= old_peak - int(0.60 * removed_bytes)
```

For kernel 2, use `R * H * 2` as the removed tensor size. For kernel 3, gate and up can run separately, so require at least one `[R,H]` tensor reduction unless the old path never had both live simultaneously.

### NCU microbench

Add `scripts/testing/profile_qwen3_moe_routed_gemm.py`.

The script must support:

```bash
python scripts/testing/profile_qwen3_moe_routed_gemm.py \
  --kernel fwd_scatter \
  --M 8192 \
  --top-k 8 \
  --H 2048 \
  --I 768 \
  --E 128 \
  --iters 50 \
  --warmup 10 \
  --weighted 1 \
  --output-dir "${NCU_OUT_ROOT}/microbench_fwd_scatter"

python scripts/testing/profile_qwen3_moe_routed_gemm.py \
  --kernel down_dx_gather \
  --M 8192 \
  --top-k 8 \
  --H 2048 \
  --I 768 \
  --E 128 \
  --iters 50 \
  --warmup 10 \
  --weighted 1 \
  --output-dir "${NCU_OUT_ROOT}/microbench_down_dx_gather"

python scripts/testing/profile_qwen3_moe_routed_gemm.py \
  --kernel gateup_dx_scatter \
  --M 8192 \
  --top-k 8 \
  --H 2048 \
  --I 768 \
  --E 128 \
  --iters 50 \
  --warmup 10 \
  --weighted 1 \
  --output-dir "${NCU_OUT_ROOT}/microbench_gateup_dx_scatter"
```

NCU is mandatory for every new kernel before any e2e conclusion. It is not enough to
show numerical parity and a lower PyTorch allocation counter. The NCU report must be
used to check memory-access patterns and identify whether the first implementation has
obvious problems or tuning room.

NCU is an optimization loop, not a checkbox. For each kernel stage, use the NCU report
to decide whether to accept, tune, or reject the implementation before moving forward.
If NCU shows a clear kernel problem, fix the kernel and rerun NCU with a fresh
`ROUTE_RUN_ID`; do not proceed to the next stage using a known-bad kernel just because
the unit test passed.

NCU-driven iteration loop:

```text
1. Implement one routed kernel.
2. Run unit parity and allocation tests.
3. Run NCU microbench with fresh artifacts.
4. Read the NCU report and classify the bottleneck.
5. Apply the smallest targeted kernel change.
6. Rerun unit tests and NCU into a new artifact directory.
7. Only then run LF smoke and real e2e for that stage.
```

Common NCU findings and required responses:

```text
target [R,H] owner still exists:
  not a performance issue; wiring or fallback is wrong
  fix call site / flags / fallback before tuning

B/CPU-weight traffic scales with M route-row tile count:
  core AsymGEMM loop order is broken
  restore B-tile stream-once/reuse-across-M loop order before e2e

tensor-core utilization collapses:
  check A/B smem layout, UMMA descriptors, barrier phases, and gathered-A staging
  do not accept a route-placement win that turns compute into a non-tensor-core path

scatter atomics dominate:
  tune epilogue ownership, hidden-column vectorization, route ordering/coalescing,
  and fp32 accumulator write pattern
  check SM100 async-store/TMA-scatter options only if they preserve AsymGEMM B-tile
  stream-once/reuse-across-M loop order
  if atomics remain the limiter, evaluate a SonicMoE-style token-owned gather/sum
  alternative only if it avoids a persistent full [R,H] owner
  do not switch to per-expert/per-route small GEMMs

gather-left global loads dominate:
  tune token-index loading, row reuse, vectorized grad_token loads, and smem staging
  keep gathered A virtual; do not materialize [R,H]
  compare against SonicMoE's gather-fusion expectation: compact token-space sources
  should not behave like a pre-materialized route-space tensor in HBM traffic

shared-memory bank conflicts or replay spikes:
  fix smem layout/swizzle compatibility before e2e

register/smem pressure causes poor occupancy:
  trim epilogue state, split helper templates, or adjust tile shape only if it keeps
  the CPU-weight reuse invariant
```

Each NCU iteration must preserve its own report. The final stage write-up must cite:

```text
initial NCU artifact
issue observed
kernel change made
follow-up NCU artifact
why the remaining bottleneck is acceptable for e2e
```

NCU commands must be run serially. Each NCU command must use a fresh `ROUTE_RUN_ID`
and `NCU_OUT_ROOT`; never reuse the export path:

```bash
export ROUTE_STAGE=stage1_fwd_scatter_route100
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__${ROUTE_STAGE}__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export NCU_OUT_ROOT="${PWD}/artifacts/ncu/qwen3_moe_routed/${ROUTE_RUN_ID}"
test ! -e "${NCU_OUT_ROOT}" || { echo "refusing to overwrite ${NCU_OUT_ROOT}" >&2; exit 1; }
mkdir -p "${NCU_OUT_ROOT}"

CUDA_VISIBLE_DEVICES=0 ncu \
  --target-processes all \
  --set roofline \
  --kernel-name regex:qwen3_moe_.*scatter.* \
  --launch-skip 10 \
  --launch-count 20 \
  --export "${NCU_OUT_ROOT}/q3_moe_fwd_scatter" \
  python scripts/testing/profile_qwen3_moe_routed_gemm.py --kernel fwd_scatter --M 8192 --top-k 8 --H 2048 --I 768 --E 128 --output-dir "${NCU_OUT_ROOT}/microbench_fwd_scatter"

export ROUTE_STAGE=stage2_down_dx_gather_route110
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__${ROUTE_STAGE}__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export NCU_OUT_ROOT="${PWD}/artifacts/ncu/qwen3_moe_routed/${ROUTE_RUN_ID}"
test ! -e "${NCU_OUT_ROOT}" || { echo "refusing to overwrite ${NCU_OUT_ROOT}" >&2; exit 1; }
mkdir -p "${NCU_OUT_ROOT}"

CUDA_VISIBLE_DEVICES=0 ncu \
  --target-processes all \
  --set roofline \
  --kernel-name regex:qwen3_moe_.*gather.* \
  --launch-skip 10 \
  --launch-count 20 \
  --export "${NCU_OUT_ROOT}/q3_moe_down_dx_gather" \
  python scripts/testing/profile_qwen3_moe_routed_gemm.py --kernel down_dx_gather --M 8192 --top-k 8 --H 2048 --I 768 --E 128 --output-dir "${NCU_OUT_ROOT}/microbench_down_dx_gather"

export ROUTE_STAGE=stage3_gateup_dx_scatter_route111
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__${ROUTE_STAGE}__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export NCU_OUT_ROOT="${PWD}/artifacts/ncu/qwen3_moe_routed/${ROUTE_RUN_ID}"
test ! -e "${NCU_OUT_ROOT}" || { echo "refusing to overwrite ${NCU_OUT_ROOT}" >&2; exit 1; }
mkdir -p "${NCU_OUT_ROOT}"

CUDA_VISIBLE_DEVICES=0 ncu \
  --target-processes all \
  --set roofline \
  --kernel-name regex:qwen3_moe_.*scatter.* \
  --launch-skip 10 \
  --launch-count 20 \
  --export "${NCU_OUT_ROOT}/q3_moe_gateup_dx_scatter" \
  python scripts/testing/profile_qwen3_moe_routed_gemm.py --kernel gateup_dx_scatter --M 8192 --top-k 8 --H 2048 --I 768 --E 128 --output-dir "${NCU_OUT_ROOT}/microbench_gateup_dx_scatter"
```

For each kernel, record:

```text
kernel wall time
old route-space expression wall time
new peak allocated bytes
old peak allocated bytes
SM throughput percentage
DRAM throughput percentage
atomic throughput / atomic sectors
achieved occupancy
registers per thread
shared memory per CTA
```

For each kernel, also write a short NCU memory-access report:

```text
kernel:
  qwen3_moe_bf16_down_forward_scatter_add_ |
  qwen3_moe_bf16_down_dx_gather_left_ |
  qwen3_moe_bf16_gateup_dx_scatter_add_

goal:
  exact [R,H] tensor this kernel is supposed to remove

global reads:
  A/grad_token load pattern
  token_indices load pattern
  routing_weights load pattern
  B CPU/TMA path health
  B CPU-weight tile reuse across M/route-row tiles
  L2 hit rate / sectors / replays

global writes:
  token-space scatter-add or [R,I] output pattern
  atomic sectors / serialization if scatter-add
  contiguous output efficiency if gather-left

shared memory:
  A/B tile staging traffic
  bank conflict symptoms
  smem throughput

tensor core:
  tensor-core utilization versus old grouped AsymGEMM
  whether gathered A or scatter epilogue starves MMA

launch structure:
  launch count
  grouped scheduling still active
  no per-expert Python loop
  no blockwise fallback accidentally enabled
  qwen3_moe_finegrained_down_scatter_block_experts == 0
  qwen3_moe_finegrained_down_scatter_blocks == 0
  qwen3_moe_finegrained_down_scatter_max_block_rows == 0

memory goal:
  old peak allocated bytes
  new peak allocated bytes
  target [R,H] owner absent from trace

action:
  accept as good enough for e2e |
  tune epilogue coalescing |
  tune gathered A staging |
  tune CPU-weight reuse / loop order |
  tune UMMA/smem layout |
  investigate atomics |
  reject because target tensor still exists
```

Kernel-specific NCU checks:

```text
down forward scatter-add:
  Must show the old contiguous [R,H] write is gone.
  Inspect token-space atomic/store traffic, L2 write pressure, atomic replay/serialization,
  hidden-column coalescing, and route metadata load overhead.
  B-side CPU/TMA traffic must remain comparable to the old grouped AsymGEMM; it must
  not grow with the number of route-row tiles.

down backward gather-left:
  Must show no [R,H] grad_routes allocation/write/read.
  Inspect gathered grad_token load coalescing, L2 hit rate, shared-memory A-tile bank
  conflicts, and whether replacing A-side TMA with gathered loads starves tensor cores.
  B-side down-weight traffic must still be amortized across the gathered-A route rows.

gate/up dX scatter-add:
  Must show no base gate_dx_routes/up_dx_routes [R,H].
  Inspect token-space atomic pressure for top-k collisions, launch count for gate and up,
  hidden-column coalescing, and tensor-core utilization versus old grouped dX.
  B-side gate/up weight traffic must not show repeated CPU streaming for every M tile.
```

Efficiency acceptance:

1. Correctness is mandatory.
2. The target tensor allocation must be removed.
3. Kernel time must be in a plausible range before e2e use:
   - first implementation: not more than 2.0x old grouped GEMM plus scatter expression for the same shape,
   - after NCU tuning: aim for not more than 1.25x old grouped GEMM plus scatter expression.
4. If NCU shows atomics dominate, do not switch to blockwise small GEMMs. First try
   route sorting/coalescing, vectorized epilogue writes, and SM100 async-store/TMA-scatter
   style placement if it can preserve the AsymGEMM CPU-weight loop order. If those are
   still poor, evaluate a SonicMoE-style token-owned gather/sum alternative only if it
   does not reintroduce a persistent full `[R,H]` route-space tensor.
5. If NCU shows gathered-A loads dominate kernel 2, tune gathered tile staging before concluding the design is bad.
6. If NCU shows the target [R,H] tensor still exists, the kernel is not correctly wired, regardless of timing.
7. If NCU shows B/CPU-weight traffic scaling with route-row tile count, the routed
   kernel broke the core AsymGEMM loop order and must be rejected before e2e profiling.
8. If NCU shows a clear avoidable bottleneck and the target kernel is more than 2.0x
   the old grouped expression, iterate on the kernel before LF smoke/e2e.
9. If NCU shows a bottleneck but e2e is needed to decide whether it matters, mark the
   stage as "NCU risk accepted for one e2e run" and keep the NCU artifact linked in the
   table. Do not silently treat that as a clean pass.

### E2E validation sequence

Run experiments one at a time. Never run profiling experiments in parallel.
Every e2e invocation must use a fresh `ROUTE_RUN_ID`/`RUN_NAME`; do not rely on a
default output path that could skip or overwrite an earlier run.

For every kernel stage:

1. Run focused unit tests.
2. Run the NCU microbench for that kernel.
3. Run a small Qwen3 MoE LF smoke profile.
4. Run the real Qwen3 MoE LF workload for that exact routed stage.
5. Inspect fine-grained activation and peak-owner artifacts to prove the intended
   owner disappeared in the actual workload.
6. Compare against the immediately previous stage and both superoffload baselines.

Small smoke profiles prove wiring only. They do not prove memory effectiveness. For
Stages 1, 2, and 3, the real `s80000,b8` e2e profile is required before moving to the
next kernel stage. A single kernel may not lower total `step_H` by itself because a later
unfixed `[R,H]` tensor may still own the peak. That is acceptable only if the
fine-grained activation accounting proves the kernel's target owner disappeared:

```text
route=100: down-base forward [R,H] owner gone
route=110: route=100 owner still gone, plus base down-backward grad_routes [R,H] gone
route=111: all base-owned down/gate/up [R,H] route-space owners gone
```

If a real stage run OOMs, preserve the OOM artifact directory and record it in the
comparison table. Then run the largest non-OOM diagnostic sequence only to inspect
owner attribution. Do not let a smaller diagnostic replace the real-workload row.

Use these target baselines:

```text
q3-30b-a3b|1 ; superoffload_mem|unsloth|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false
q3-30b-a3b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false
```

Use this target Asym config:

```text
q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false
```

Small smoke command template:

```bash
export ROUTE_STAGE=routeXXX
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__${ROUTE_STAGE}__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export RUN_NAME="q3moe_${ROUTE_RUN_ID}"
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_routed"
export OVERWRITE=false
export COLLECT_EXISTING=false
test -n "${RUN_NAME}"

export RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false'
export ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0
export UNSLOTH_GC_OUTER_HBM_EVERY_N=0
export ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32

# stage-specific flags set here
bash scripts/lf/profile_lora_lf_test_source.sh
```

Real workload command template:

```bash
export ROUTE_STAGE=routeXXX
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__${ROUTE_STAGE}__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export RUN_NAME="q3moe_${ROUTE_RUN_ID}"
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_routed"
export OVERWRITE=false
export COLLECT_EXISTING=false
test -n "${RUN_NAME}"

export RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false'
export ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0
export UNSLOTH_GC_OUTER_HBM_EVERY_N=0
export ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32

# stage-specific flags set here
bash scripts/lf/profile_lora_lf_test_source.sh
```

Stage commands:

```bash
# Stage 0: old current path, routed kernels disabled
export ROUTE_STAGE=route000
export ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM=0
export ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=0
export ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=0
export ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=0

# Stage 1: only down forward fused
export ROUTE_STAGE=route100
export ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM=0
export ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=0
export ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=0

# Stage 2: down forward fused plus down backward gather-left
export ROUTE_STAGE=route110
export ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM=0
export ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=1
export ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=0

# Stage 3: all base routed kernels
export ROUTE_STAGE=route111
export ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM=0
export ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=1
export ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=1
```

Required artifact files to inspect after every e2e run:

```text
command.txt
runtime_counters.json
process_memory.csv
memory_actual_peak_breakdown.csv
memory_live_activation_details.csv
memory_timeline.csv
peak_snapshot_attrib_allblocks.csv
peak_snapshot_attrib_allblocks.json
```

Required comparison table after every stage:

```text
Workload    Backend          Config/flags                         fwd_s  bwd_s  opt_s  step_s  fwd_H  bwd_H  step_H  RAM   artifact_dir   ncu_artifact
----------  ---------------  -----------------------------------  -----  -----  -----  ------  -----  -----  ------  ----  -------------  ----------------
s80000.b8   superoffload_mem unsloth                              ...                                                ...   /abs/path/...  n/a
s80000.b8   superoffload_mem unsloth-off                          ...                                                ...   /abs/path/...  n/a
s80000.b8   asym_cpuadamwds  recomp-off-full-fg route=000         ...                                                ...   /abs/path/...  n/a
s80000.b8   asym_cpuadamwds  recomp-off-full-fg route=100         ...                                                ...   /abs/path/...  /abs/path/...
s80000.b8   asym_cpuadamwds  recomp-off-full-fg route=110         ...                                                ...   /abs/path/...  /abs/path/...
s80000.b8   asym_cpuadamwds  recomp-off-full-fg route=111         ...                                                ...   /abs/path/...  /abs/path/...
```

Required memory-decomposition table:

```text
Workload    Backend/config                 Peak owner                         Bytes/GiB  Expected change                 artifact_dir
----------  -----------------------------  ---------------------------------  ---------  ------------------------------  ----------------
s80000.b8   route=000                      down_base_forward_routes [R,H]      ...        present                         /abs/path/...
s80000.b8   route=100                      down_base_forward_routes [R,H]      0 or tiny  removed by kernel 1             /abs/path/...
s80000.b8   route=100                      down_base_backward_grad [R,H]       ...        still present                   /abs/path/...
s80000.b8   route=110                      down_base_backward_grad [R,H]       0 or tiny  removed by kernel 2             /abs/path/...
s80000.b8   route=110                      gate/up_base_dx_routes [R,H]        ...        still present                   /abs/path/...
s80000.b8   route=111                      gate/up_base_dx_routes [R,H]        0 or tiny  removed by kernel 3             /abs/path/...
```

Do not conclude from `step_H` alone. If memory does not move as expected, inspect owner attribution. The correctness criterion for each kernel is that the exact target `[R,H]` owner disappears. End-to-end peak reduction is expected only if that owner was live at or near the peak.

The artifact path columns are mandatory. A table row without a concrete preserved
artifact directory is not evidence. For routed-kernel rows, `ncu_artifact` must point to
the NCU report used to accept or risk-accept that kernel version.

### Optional LoRA routed kernels

Only start this section if `route=111` still has a large LoRA-owned route-space `[R,H]` tensor at peak.

Optional LoRA primitives:

```text
qwen3_moe_lora_b_forward_scatter_add_
qwen3_moe_lora_b_dx_gather_left_
qwen3_moe_lora_a_dx_scatter_add_
```

Do not implement these preemptively. Base routed kernels first.

## Summary

The desired design is possible, but it is real kernel work:

```text
not Python-only
not current TMA epilogue
not blockwise expert splitting
not per-expert small GEMMs
```

Implement route-aware grouped kernels so the compute remains grouped while the large
`[R,H]` tensors are never written to global HBM. The base kernels are:

```text
1. grouped base forward scatter-add
2. grouped base backward gather-left
3. grouped base dX scatter-add
```

Base routed kernels are the first target. After those are active, implement routed
LoRA-B forward/backward and routed LoRA dX scatter-add only if artifacts show LoRA-owned
`[R,H]` tensors are a meaningful remaining peak owner. Do not spend kernel time on LoRA
before that gate.
