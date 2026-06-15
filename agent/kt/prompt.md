You are continuing KT ARM BF16 optimization in `AsymGEMM` and must pick up from the current state without regression risk.  
Working directory: `/workspace/AsymGEMM-SFT/third_party/AsymGEMM`.

Hard boundary (required):
- All implementation and profiling work for this iteration must be in the KT repo or KT-side scripts:
  - `../ktransformers` for kernel/model code changes
  - `agent/kt/scripts` and `scripts/kt` for profiling launch and validation
- Do **not** edit core `AsymGEMM` product training/profiling code outside KT-specific paths.
- If a behavior can be implemented via KT scripts, do it there. This preserves AsymGEMM development independence.

Context to preserve:
- KT path is **native ARM BF16**, not DeepSpeed; no DeepSpeed/ZeRO routing assumptions for KT kernels.
- User requirement: training/profile on **GPU 1 or 2 only** (not 0/3), `batch_size=4`, `seq_len=4096`, `lora_rank=64`, `lora_dropout=0.00`, `warmup_steps=5`, `measure_steps=10`, `total_steps=15`.
- Keep KT and AsymGEMM changes isolated so AsymGEMM can evolve independently while KT profiling remains in its repo/scripts.
- No loops over experts in implementation path; must be grouped / batched route-expert execution style.
- Remove stale or dead paths; no carrying old/unused script or implementation branches.
- Any claim must be backed by e2e profiling (not toy-only), and results should be appended/updated in the fix doc.

Latest completed benchmark status (already observed):
- KT profile is native ARM BF16, not DeepSpeed.
- Full e2e KT ARM BF16 was completed with fixed config.
- Current best known numbers:
  - Avg step improved from `311.745s` -> `276.813s`
  - Avg forward/backward improved from `74.428/235.798` -> `75.246/199.535`
  - `peak allocated ~34.48GiB`, `peak reserved ~44.03GiB`, `process RSS ~184.47GiB`
- Dominant remaining bottlenecks:
  - forward expert schedule: `1078.092 ms/layer`
  - backward grouped tile: `2457.110 ms/layer`
  - backward_task_sum base grad: `66154.476 ms/layer`
  - backward_lora_grad: `15930.130 ms/layer`
- Thread-reduce and scratch allocation regressions were reduced; those are no longer the primary bottleneck.

Attained e2e metrics table (KT BF16, batch=4, seq=4096, rank=64, dropout=0.00):

| Metric | Reference (`before`, v6_stage4) | Current (`after`, v6_accept) |
|---|---:|---:|
| avg_step | 275.536 s | 276.813 s |
| avg_forward | 74.657 s | 75.246 s |
| avg_backward | 199.304 s | 199.535 s |
| peak_allocated | 34.478 GiB | 34.479 GiB |
| peak_reserved | 40.111 GiB | 44.029 GiB |
| process_rss | 179.638 GiB | 184.467 GiB |
| expert_schedule_wall_ms | N/A | 1078.092 ms/layer |
| backward_grouped_tile_ms | 2444.008 ms/layer | 2457.110 ms/layer |
| backward_tile_recompute_ms | 44108.256 task-ms/layer | 44131.841 task-ms/layer |
| backward_route_grad_accum_ms | 82858.797 task-ms/layer | 82977.276 task-ms/layer |
| backward_base_grad_ms | 66050.799 task-ms/layer | 66154.476 task-ms/layer |
| backward_lora_grad_ms | 15915.865 task-ms/layer | 15930.130 task-ms/layer |
| backward_local_alloc_zero_ms | 60.352 ms/layer | 56.582 ms/layer |
| backward_thread_reduce_ms | 0.000 ms/layer | 0.000 ms/layer |
| sparse_backward_scratch_bytes | 3.043 GiB | 3.043 GiB |
| train loss (max/last/train) | 1.6572/1.6701/1.8459 | 1.8324/1.4692/1.6074 |

Notes:
- The “before” row is the best-relevant prior KT run kept for continuity; the “after” row is the accepted v6-stage result to continue optimizing.
- Use this table as the baseline when deciding if a stage is an improvement.

Mission:
1. Continue optimization with highest priority on **backward grouped kernel efficiency** (compute-time dominated), not allocation/repack/reduction cleanup.
2. Keep grouped layout semantics; do not reintroduce expert-wise loops.
3. Keep all experiments reproducible with fixed command lines and clearly logged artifacts.
4. Update `agent/kt/fix_arm_v6.md` only for in-progress/completed priorities and append quantitative results after run-level validation.

Primary code inspection targets:
- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
- KT script pipeline under `agent/kt/scripts/`
- Existing KT profiling docs/results in `agent/kt/fix_arm_v6.md` (or newer `fix_arm_v7.md` if present)

Next priorities (in order):

1) Confirm exact hotspot attribution in grouped backward path
- Verify any implicit serial work remains around expert route scheduling/partitioning in `bf16_sft_moe.hpp`.
- Keep task partitioning and reductions fully grouped; replace any loop-like bottleneck with grouped, vectorized work-sharing and better tiling.
- Validate:
  - Run e2e KT script on GPU1 or GPU2.
  - Capture `source_profile.json`, `profile.json`, train log, native profile counters.

2) Optimize backward grouped expert matmul scheduling path
- Focus on `backward_tile_recompute`, `backward_base_grad`, and `backward_lora_grad` timer regions.
- Prioritize:
  - higher per-layer occupancy,
  - lower scratch traffic on hot loops,
  - memory layout for contiguous traversal,
  - no fallback paths and minimal branch divergence.

3) Validate kernel quality without touching AsymGEMM core
- Keep edits KT-only.
- Preserve KT-specific wrappers/scripts and avoid collisions with non-KT profiling variants.

4) Remove remaining stale/obsolete script paths
- Remove or disable duplicated and unused profiling branches that can route to old implementations.
- Keep one supported KT path and explicit launch into KT native profile script.

Mandatory validation before stage transition:
- Rebuild/reinstall KT as needed.
- Use canonical command configuration:
  - GPU: 1 or 2 only
  - seq=4096, batch=4, rank=64, dropout=0.00, warmup=5, measure=10
- Confirm stage if all pass:
  - `avg_step_s`, `avg_forward`, `avg_backward`
  - `peak_allocated_MiB` / `peak_reserved_MiB`
  - `expert_schedule_wall_ms`, `backward_grouped_tile_ms`, `backward_route_grad_accum_ms`, `backward_base_grad_ms`, `backward_lora_grad_ms`
  - process RSS peak
- Reject stage if no improvement in step/hotspot metrics or regressions in memory stability/ correctness.

Deliverables:
- Create/update `agent/kt/fix_arm_v7.md` with staged implementation plan and “implemented / measured / failed” status.
- Record exact command lines, artifact paths, and before/after metrics each stage.
- Keep explicit note in every handoff: **KT remains native ARM BF16; DeepSpeed not used**.
