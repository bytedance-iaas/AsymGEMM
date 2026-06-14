# Expert Activation Offload V3 Active Implementation Plan

This is the active design and implementation plan. The v2 document is retained
only as a historical record of prior attempts, failed changes, and profiling
evidence.

Goal: reduce peak HBM during Qwen3 MoE LoRA SFT by offloading almost all
forward routed expert activations to pinned CPU memory, then consuming those
activations in backward through grouped CPU-source AsymGEMM/native kernels.

The fair claim is narrow:

```text
With model, optimizer, LoRA config, precision, routing, batch, sequence length,
profiler, and global recompute mode held fixed, replacing expert activation
recompute with expert activation offload plus grouped CPU-source AsymGEMM
fetchback reduces peak HBM.
```

Do not use unrelated memory optimizers to prove this claim. LoRAFusion,
alternate fused MoE stacks, different loss kernels, different checkpointing, or
optimizer changes are useful design references only if both sides of the
comparison get the same change. The main comparison remains:

```text
BACKEND_SPECS=asym_cpuadamwds|norecomp
ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false
```

This is the active
`/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh`
workflow. `none|true` is the target implementation. `gc-exp|false` is the
expert checkpoint/recompute baseline in this workflow. `none|false` is the
no-expact/no-expert-recompute control that shows whether the workload fits
without either expert memory strategy. Global transformer recompute remains a
useful lower bound, not the primary comparison.

Priority order:

1. Preserve and increase the HBM reduction of `none|true`.
2. Improve `none|true` latency whenever the change does not materially increase
   peak HBM or only increases it within a clearly justified, roughly flat
   budget.
3. Reject latency optimizations that erase the memory win, even if they make
   the path faster.

Acceptance discipline:

- Same peak HBM with worse latency is always a regression. Revert it.
- A trivial peak-HBM reduction with a large latency increase is not acceptable.
- Toy or microbenchmark memory wins do not count unless the active LF workflow
  also shows a meaningful peak-HBM reduction or a clearly reported expert-block
  peak reduction.
- A memory-first change is acceptable only when the active LF workflow shows a
  meaningful peak-HBM reduction and latency does not blow up.
- A latency-first change is acceptable only when it improves `none|true`
  latency while keeping peak HBM flat or lower.
- If the tradeoff is ambiguous, keep the current baseline and collect better
  attribution; do not land the change.

The implementation loop continues stage by stage: first remove avoidable HBM
materialization, then use source-profile, attribution, standalone kernel timing,
and NCU evidence to reduce latency without giving back the memory win. Stop only
when the remaining latency comes from necessary CPU-source movement/compute or a
documented unresolved kernel issue.

## Roadmap

Execute the stages in order. Do not advance a memory-changing stage until the
previous validation passes and the current profile explains the peak. The
current active comparison baseline is the fresh LF `b4_s4096` sweep recorded
below. Larger sequence lengths can be stress checks, but they are not the active
baseline unless rerun and recorded here.

| Stage | Purpose | Main output |
| --- | --- | --- |
| S0 | Ground-truth peak diagnosis before new changes | peak live-set ownership, expert/non-expert split, and a verdict on whether scatter no-save is peak-relevant |
| S1 | Measurement, lifetime, and CPU Adam guardrails | expact counters, saved/lifetime labels, CPU Adam health, and bounded stage accounting |
| S2 | Remove route-expanded scatter saved output | router-no-grad scatter path that does not save the full `[routes, hidden]` expert output |
| S3 | Remove full HBM base activation restaging | grouped CPU-source base forward/dX kernels with no full `[M,I]` or `[M,2I]` HBM stage |
| S4 | Replace wide LoRA-B staging | CPU-source/tiled LoRA-B backward that keeps grouped calls and avoids full wide gradient staging |
| S5 | Tile LoRA-A gradient reductions | no full FP32 LoRA-A accumulator workspaces beyond the real parameter gradients |
| S6 | True paired gate/up scheduling | paired CPU-tile reuse for gate/up LoRA paths, accepted only when LF timing improves under the same memory budget |
| S7 | Hidden materialization and workspace cleanup | profile-visible bounded scratch, no untagged full-width expact materializations |
| S8 | Final fair comparison and CPU Adam acceptance | canonical `none|true,gc-exp|false,none|false` LF comparison and CPU Adam pass/fail evidence |

Current active comparison baseline, recorded 2026-06-13 from the exact LF
workflow:

- command:
  `bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh`
- output root:
  `/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/profiling/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b4_s4096_w5_s10_r64_a16_drop000`
- model/workload: `Qwen/Qwen3-30B-A3B`, `b4_s4096`, `logical_qlen=16384`
- dataset: `asym_long_sft_smoke__qwen3-30b-a3b__s4096`
- backend/profiler: `asym_cpuadamwds|norecomp`, `PROFILERS=source`
- policy axis: `ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false`
- precision/LoRA: `bf16`, rank `64`, alpha `16`, dropout `0.00`, target `all`
- steps: `WARMUP_STEPS=5`, measured `MAX_STEPS=10`, trainer total `15`
- profiling overhead flags:
  `PROFILE_MEMORY_ATTRIBUTION=false`,
  `PROFILE_MEMORY_BREAKDOWN=false`,
  `PROFILE_MEMORY_SNAPSHOT=false`
- CPU Adam config:
  `USE_ASYM_CPU_ADAMW=true`, backend `deepspeed`, pinned memory `true`,
  fp32 master `true`

Baseline results:

| policy | implementation | status | peak allocated HBM | peak reserved HBM | avg step | avg forward | avg backward |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `none|true` | activation offload target | completed | `126.312 GiB` | `131.766 GiB` | `43.742 s` | `11.147 s` | `32.508 s` |
| `gc-exp|false` | expert torch checkpoint baseline | completed | `150.312 GiB` | `164.617 GiB` | `3.803 s` | `1.390 s` | `2.367 s` |
| `none|false` | no expact/no expert recompute control | OOM in first forward/loss | `178.891 GiB` partial | `179.520 GiB` partial | n/a | n/a | n/a |

Runtime call counts from the same run:

- `none|true`:
  `asym_forward_calls=5055`, `asym_dx_calls=4290`,
  `torch_forward_calls=0`, `torch_dx_calls=0`,
  `reference_fallback_count=0`, `fallback_reasons=none`
- `gc-exp|false`:
  `asym_forward_calls=6495`, `asym_dx_calls=4290`,
  `torch_forward_calls=0`, `torch_dx_calls=0`,
  `reference_fallback_count=0`, `fallback_reasons=none`
- `none|false`:
  `asym_forward_calls=337`, `asym_dx_calls=0`,
  `torch_forward_calls=0`, `torch_dx_calls=0`,
  `reference_fallback_count=0`, `fallback_reasons=none`

OOM detail for `none|false`: cross-entropy tried to allocate `9.27 GiB` with
`4.14 GiB` free; PyTorch reported `178.89 GiB` allocated and the partial
profile captured the peak above.

Acceptance rule against this baseline: future changes must either materially
lower `none|true` peak HBM without a large latency regression, or improve
`none|true` latency while keeping peak HBM flat or lower. Same memory with worse
latency is a regression and must be reverted. Tiny HBM savings do not justify a
major latency blow-up.

All LF profiling and acceptance validation must use the workflow script:

```bash
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Use direct unit tests and `scripts/testing/profile_qwen3_activation_offload.py`
only for fast kernel or wrapper checks. They do not replace the LF workflow
comparison. The canonical LF profile command shape is:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
OUTPUT_ROOT="$PWD/outputs/<run_name>" \
GPU_POOL=<gpu_id_or_pool> \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=<seq_len> \
PER_DEVICE_TRAIN_BATCH_SIZE=<batch_size> \
MAX_STEPS=<measured_steps> \
WARMUP_STEPS=<warmup_steps> \
PLOT=false \
RUN_POST=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus <gpu_id_or_pool>
```

For the current active comparison validation, use `SEQ_LENS=4096`,
`PER_DEVICE_TRAIN_BATCH_SIZE=4`, `LORA_DROPOUT=0.00`, and the same policy
axis above. Set `LORA_DROPOUT` explicitly; otherwise the LF script may also
run its default dropout `0.08` axis and mix stale, non-canonical results into
the output tree. For a fair latency check, do not turn on extra memory
attribution/breakdown flags unless the comparison baseline used the same flags.
For diagnostic runs, enable `PROFILE_MEMORY_ATTRIBUTION=true`,
`PROFILE_MEMORY_BREAKDOWN=true`, and `PROFILE_MEMORY_BREAKDOWN_INTERVAL=1` to
explain a peak before changing kernels or lifetimes. For final acceptance, keep
this script and only vary batch/sequence/profiler settings deliberately.
Set `RUN_POST=false` for targeted or single-policy runs; otherwise
`profile_lora_lf.sh` calls `scripts/lf/test_profiling.sh` after it finishes and
can launch an unrelated follow-up sweep.

Peak-memory attribution is part of the implementation loop. After each stage,
inspect the source profile, memory breakdown rows, stage tags, saved activation
attribution, live activation attribution, and allocator peak-growth attribution.
If the current attribution cannot explain the new peak or cannot distinguish
expert-block memory from loss/cross-entropy or other non-expert peaks, improve
the attribution first before guessing at an optimization. It is acceptable to
add stage-local tags, counters, saved-tensor labels, allocator snapshots, or
profile JSON fields when needed to surface the real issue.

## Non-Negotiable Invariants

- No Python loop or host loop that launches one GEMM per expert.
- No small-GEMM decomposition of expert work. Normal training uses grouped
  kernels/GEMMs.
- No full HBM staging of `X_cpu`, `act_cpu`, `dgate_cpu`, or `dup_cpu`.
- No full HBM concat solely to satisfy a convenient grouped_mm API.
- No full CUDA FP32 accumulator shaped like a full LoRA-A grad, LoRA-B grad, or
  full low-rank `dS`.
- `ctx` for activation offload stores CPU handles, metadata, weights, scalar
  config, and trainable LoRA tensors only. It must not keep CUDA activation
  views or staged CUDA tensors alive across forward.
- Every CUDA intermediate is released at its last use before the next
  high-memory region.
- D2H copies that feed CPU-side math have explicit completion before CPU reads.
- H2D/CPU-source kernel reads are stream ordered with their producer copies.
- Async prefetch/staging must be bounded. Async itself is not free: it can raise
  peak HBM if a staged buffer overlaps a still-live producer or consumer. The
  design must count in-flight bytes and cap concurrent staged/prefetch buffers.
- LoRA trainable parameters stay CUDA `nn.Parameter`s with normal `.grad`
  tensors. `AsymCPUAdamW` owns CPU master/state, but backward must still return
  CUDA gradients for the LoRA parameters.

Allowed CUDA materialization:

- final expert output `[M,H]`
- final input gradient `[M,H]` when the caller needs `dX`
- low-rank `[M,r]` tensors when they are true LoRA-A outputs
- final LoRA gradients with true parameter shapes
- tile-sized scratch whose size is independent of full activation width

## Current Blocking Issues And Implementation State

The original `ASYMM_EXPERT_ACT_OFFLOAD=true` path ran, but it was not the target
algorithm. The rejected v2 implementation has been reverted from the hot path,
so these are again current blockers, not solved state:

- `qwen3_moe.py::_ActivationOffloadQwen3ExpertFunction.forward` can restage the
  down-base activation through a full CUDA `act_for_down_base` buffer. A future
  replacement must prove a peak-HBM drop in the LF workflow before it stays.
- `qwen3_moe.py::_ActivationOffloadQwen3ExpertFunction.backward` can build the
  full CUDA `dgate_up_for_gate_up_base` concat for gate/up base dX. Removing it
  is still desirable, but the previous CPU-source-base attempt did not lower
  the full-workflow peak and regressed latency, so it must be redesigned or
  guarded by a strict acceptance gate.
- Offloaded CPU activation lifetime and stream ordering still need explicit
  accounting: D2H readiness, CPU-read waits, CPU-source GPU-consumer waits,
  stage-cache drops, and release-at-last-use must be visible in the profile.
- Gate/up LoRA-B backward can still recreate wide staged CUDA gradients or full
  FP32 helper workspaces. CPU-source/tiled replacements remain candidates only
  if they reduce peak HBM or improve latency without increasing peak HBM.
- LoRA-A gate/up forward still does not have a proven paired CPU-tile-reuse
  kernel in the accepted path. Pairing remains a latency candidate, not an
  accepted memory optimization.
- Regression tests should keep checking that any future accepted implementation
  removes `act_for_down_base` and `dgate_up_for_gate_up_base` stage tags only
  when LF profiling confirms the removal actually lowers the relevant peak.

## Related Code Constraints

Local code checked for design constraints:

- Megatron-LM keeps MoE work grouped by `tokens_per_expert` and has fine-grained
  activation offload/checkpoint lifetimes. Map that lesson to grouped offsets
  and explicit release points here, not to a new execution stack.
- DeepSpeed activation checkpointing and ZeRO offload use explicit buffer
  ownership/reset points. Map that to `ActivationOffloadManager` live-byte and
  cache accounting.
- DeepSpeed FPDT uses chunked `load_to_gpu`, `offload`, stream waits, and
  double-buffer style scheduling. Use that as the async model only after HBM is
  counted and bounded.
- LlamaFactory is the harness. Preserve its labels and CPU Adam integration.
- LoRAFusion is not the fair-comparison mechanism. It only motivates paired
  LoRA scheduling and avoiding repeated LoRA passes.

## Kernel Optimization Notes

Use Nsight Compute (`ncu`) whenever a newly developed helper kernel, wrapper
kernel, or expact-specific native kernel becomes the bottleneck. Profile those
kernels separately and alone, with a small reproducible harness that isolates
one launch shape at a time. The point of these profiles is to inspect memory
flow, CPU/host-mapped load behavior, SMEM movement, occupancy, replay/stall
reasons, and whether the kernel is actually reusing the CPU-sourced tile as
intended. Do not infer kernel quality only from end-to-end LF step time.

For kernels that load from CPU/pinned host memory into SMEM, the core
optimization rule is CPU tile reuse: load the CPU tile once into SMEM, then
reuse that tile across the relevant N/K/output-tile loops instead of repeatedly
fetching the same host tile. The existing AsymGEMM CPU-left and CPU-right
designs are built around this principle. Any new expact base, LoRA-A, LoRA-B,
or elementwise-fusion kernel should follow the same pattern and should be
rejected or revised if NCU shows repeated host reads where SMEM reuse is
possible.

When improving these kernels, keep memory accounting and launch accounting in
the profile:

- number of AsymGEMM/GEMM/native helper launches
- host-to-SMEM bytes implied by CPU-source tiles
- HBM temporary bytes and peak live stage bytes
- whether the implementation uses one grouped launch rather than per-expert
  launches
- whether tile scratch remains bounded and independent of full activation
  width

## Current Baseline Context And Historical Guardrails

The active comparison numbers are only the fresh `b4_s4096` sweep above. Older
numeric runs are intentionally not carried forward in v3 because they caused
confusion and are no longer the comparison baseline. Detailed historical numbers
belong in `fix_exp_acc_offload_v2.md`.

Preserve these non-numeric lessons from the rejected v2 work:

- Do not re-land CPU-source base or paired LoRA changes just because they remove
  a local stage tag.
- A candidate memory change must reduce the global LF peak or a clearly
  reported expert-block peak under the active comparison workload.
- A candidate latency change must improve the active `none|true` baseline while
  keeping HBM roughly flat.
- Toy-profile wins, local stage-tag wins, and tiny HBM reductions are not enough
  if the active LF workflow does not show a meaningful memory benefit or if
  latency becomes materially worse.
- Full HBM stages such as `act_for_down_base`, `dgate_up_for_gate_up_base`, and
  wide LoRA-B staging remain suspicious, but removing them is only useful if the
  LF peak or an explicitly reported expert-block peak moves.
- The likely failure mode from v2 was trading small or non-peak HBM stages for
  more CPU-source traffic, stream waits, CPU-side read waits, and helper-kernel
  work without moving the real global peak.
- Loss/cross-entropy peaks can hide the expert result, so expert-block peaks and
  loss peaks must be reported separately.

Keep the three baseline categories separate:

- canonical workflow comparison:
  `BACKEND_SPECS=asym_cpuadamwds|norecomp` and
  `ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false`
- global recompute lower bound:
  `BACKEND_SPECS=asym_cpuadamwds|recomp` and
  `ASYMM_EXP_ACT_POLICIES=none|false`
- optional global-plus-expert recompute sanity check:
  `BACKEND_SPECS=asym_cpuadamwds|recomp` and
  `ASYMM_EXP_ACT_POLICIES=gc-exp|false`

Older thresholded token-policy examples are not the default workflow for this
plan. Use them only as explicit extra experiments if we need to compare custom
thresholded recompute against `gc-exp`.

### Current Attribution Direction: Active `b4_s4096`

Use the fresh `b4_s4096` sweep above as the no-hook comparison control. Do not
reuse older hook-enabled or stress-run numbers as baselines.

Saved-tensor hooks and full memory breakdown can perturb memory and timing. Use
them only for diagnosis, and never replace the no-hook source-profile baseline
with hook-enabled values. If a hook-enabled diagnostic run changes the peak, it
must be recorded as a profiler artifact, not as a comparison result.

The main expert-side lifetime candidate remains scatter output saving in router
no-grad mode. For the active `b4_s4096` workload:

- route-expanded scatter tensor shape: `[131072, 2048]`
- dtype: `torch.bfloat16`
- size per layer: `536,870,912` bytes (`512 MiB`)
- model depth: `48` layers
- expected upper-bound expert scatter saved bytes if all layers are peak-live:
  `24.0 GiB`

Implementation conclusion:

- The next expert-side memory change should target `scatter_contiguous` in
  router no-grad mode so it does not save the full `[tokens * top_k, hidden]`
  `expert_output`.
- If router gradients are re-enabled, the no-save path must fall back because
  it cannot compute `grad_weights` without `expert_output`.
- After any scatter lifetime change, rerun the active three-policy LF sweep and
  compare against the fresh `b4_s4096` table above.
- If the global peak does not move, report the expert-block peak separately and
  identify the new global peak owner before claiming success.
- Do not use LoRAFusion or other fused-kernel baselines to claim this memory
  win. The fair comparison is still the same LF workflow and backend with only
  the expert activation-offload implementation changed.

## Stage 0: Ground-Truth Peak Diagnosis

The diagnosis plan in `agent/impls/plan.md` is important enough to fold into
this active plan. It should run before landing Stage 2 or any renewed
CPU-source kernel rewrite because it provides allocator-ground-truth peak
ownership without relying only on saved-tensor hooks.

Important correction: do not blindly accept the draft claim that
`scatter_contiguous` cannot save a route-expanded `[M,H]` tensor. A local
saved-tensor microcheck with detached route weights still shows
`index_add`/`index_add_` saving a route-expanded source tensor. Therefore the
scatter no-save path remains a valid candidate, but it must be accepted only
after the snapshot analyzer and LF peak validation prove the saved tensor is
real and peak-relevant.

### Scope

- New `scripts/testing/analyze_cuda_memory_snapshot.py`
  - torch-free analyzer for `memory_snapshot.pickle`
  - replays `device_traces` to reconstruct the true peak live-set
- New `tests/testing/test_analyze_cuda_memory_snapshot.py`
  - synthetic-pickle unit test; no torch import
- `scripts/lf/run_lf_profiled_train.py`
  - `_start_memory_snapshot_recording`
  - `_dump_memory_snapshot`
- `asym_gemm/training/qwen3_moe.py`
  - optional non-disturbing expert timeline probes gated by
    `ASYM_EXPACT_PEAK_PROBE=1`
- `asym_gemm/training/activation_offload.py`
  - optional weak live-manager registry and aggregate staged-HBM counters
- `asym_gemm/profiling/lf_trace.py`
  - debug dump and reconciliation hooks only after the analyzer identifies a
    mismatch

### Code Changes

1. Harden existing memory snapshot capture. Do not invent a new snapshot env:
   the LF path already uses `PROFILE_MEMORY_SNAPSHOT=true`, which maps to
   `ASYM_GEMM_LF_PROFILE_MEMORY_SNAPSHOT`.

```python
try:
    torch.cuda.memory._record_memory_history(
        max_entries=200_000,
        stacks="python",
        context="all",
    )
except TypeError:
    torch.cuda.memory._record_memory_history()
```

The dump happens at end-of-run, so the analyzer must replay
`device_traces`; it must not treat the final `segments` as the peak live-set.

2. Implement a torch-free snapshot analyzer:

```text
read pickle
for each alloc/free event in device_traces[device]:
  update live_blocks[addr]
  track current live bytes
  copy live_blocks when current reaches a new max
attribute peak live blocks by deepest useful Python frame
emit json + markdown: peak bytes, top blocks, buckets, source frames
```

Buckets must distinguish expert frames (`qwen3_moe.py`, `moe.py`,
`exp_act_offload_*`, `frozen_linear.py`, `activation_offload.py`) from
attention, norms, loss/lm_head, allocator/unframed, params/grads, activation,
and workspace/residual.

3. Add non-disturbing expert probes only if the snapshot alone cannot identify
   phase. Use `torch.cuda.memory_allocated()` current values and JSONL logs.
   Do not call `reset_peak_memory_stats()` inside the LF step because that
   corrupts the source profile's global `peak_allocated_hbm_bytes`.

4. Add ctx/manager live-HBM audit behind `ASYM_EXPACT_PEAK_PROBE=1`:
   - weak registry of live `ActivationOffloadManager`s
   - aggregate live staged bytes across all 48 layers
   - forward-exit invariant that expact ctx stores CPU handles and metadata,
     not CUDA activation tensors
   - log any live staged tag, tensor shape, and bytes

5. Reconcile `LFMemoryBreakdownProfiler` only after the snapshot analyzer
   proves a mismatch:
   - dump per-storage rows for the routed-expert bucket
   - move bytes only when the snapshot frame evidence shows misattribution
   - preserve closure to the actual peak allocated/reserved HBM

### Risks And Watch Items

- `_record_memory_history` keyword support is torch-version sensitive; keep the
  `TypeError` fallback and record whether Python frames were present.
- `max_entries=200_000` may truncate a long run. Use `MAX_STEPS=1`; if analyzer
  peak is below `memory.md` peak, rerun with a higher entry limit or a per-step
  dump.
- C-only/unframed allocations may remain; bucket them explicitly instead of
  forcing them into `routed_experts`.
- Timeline probes are current-memory probes, not peak probes. They are for
  phase ordering only; allocator peak ownership comes from the snapshot replay.
- The diagnosis stage is not a memory optimization. It passes by identifying
  the true peak-owner stack, not by lowering HBM.

### Validation Before Stage 1

Torch-free analyzer unit:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/testing/test_analyze_cuda_memory_snapshot.py
```

Shared diagnostic LF run:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
OUTPUT_ROOT="$PWD/outputs/expact_v3_stage0_diag_b4s6144_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=0 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES=none|true \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=6144 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PROFILE_MEMORY_ATTRIBUTION=true \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_MEMORY_BREAKDOWN_INTERVAL=1 \
PROFILE_MEMORY_SNAPSHOT=true \
PLOT=false \
RUN_POST=false \
CONTINUE_ON_ERROR=false \
scripts/lf/profile_lora_lf.sh --gpus 0
```

Snapshot analysis:

```bash
PYTHONPATH="$PWD" .venv/bin/python scripts/testing/analyze_cuda_memory_snapshot.py \
  --snapshot "<run_dir>/memory_snapshot.pickle" \
  --device 0 \
  --top 40 \
  --min-bytes 268435456 \
  --output-json outputs/expact_v3_stage0_peak_attrib.json
```

Stage 0 passes only if:

- analyzer peak matches the source-profile `peak_allocated_hbm_bytes` within
  allocator rounding, or the mismatch is explained as trace truncation and
  rerun
- every live block at least `256 MiB` has a source frame or is explicitly
  `allocator/unframed`
- routed-expert bytes at peak are split into kept LoRA params, kept LoRA grads,
  real activation, real workspace, and misattributed/non-expert bytes
- the output states whether Stage 2 scatter no-save is peak-relevant or only a
  non-peak/local lifetime improvement

## Stage 1: Measurement, Lifetime, And CPU Adam Guardrails

### Scope

- `asym_gemm/training/activation_offload.py`
  - `CPUActivationHandle`
  - `ActivationOffloadStats`
  - `ActivationOffloadManager.empty_cpu`
  - `ActivationOffloadManager.offload`
  - `ActivationOffloadManager.stage`
  - `ActivationOffloadManager.stage_concat_columns`
  - `ActivationOffloadManager.release_stage`
  - `ActivationOffloadManager.release_cpu`
  - `ActivationOffloadManager.snapshot`
- `asym_gemm/training/frozen_linear.py`
  - `AsymExecutionStats`
- `asym_gemm/training/qwen3_moe.py`
  - `_ActivationOffloadQwen3ExpertFunction.forward`
  - `_ActivationOffloadQwen3ExpertFunction.backward`
  - `_activation_offload_cpu_silu_mul`
  - `_activation_offload_cpu_silu_backward`
  - `AsymQwen3Experts._last_activation_offload_stats`
- `asym_gemm/integrations/lf.py`
  - `LFAsymReport.runtime_log_string`
- `scripts/lf/run_lf_profiled_train.py`
  - source profile JSON assembly near `SourceProfileRecorder.build_profile`
  - add model/runtime extraction for Asym execution stats if needed
- Tests:
  - `tests/training/test_lf_qwen3_asym_backend.py`
  - `tests/training/test_asym_cpu_adamw.py`
  - `tests/lf/test_asym_cpu_adamw_lf_integration.py`
  - `tests/lf/test_asym_cpu_adamw_args.py`

### Code Changes

1. Extend `CPUActivationHandle` with copy-order metadata:
   - `producer_stream` or `producer_event`
   - `ready_for_cpu_read`
   - `copy_direction`
   - `requires_cpu_sync_before_read`

2. Split offload APIs by consumer:
   - `offload(..., cpu_read=False)` for CUDA CPU-source kernel consumers
   - `mark_ready_for_cpu_read(handle)` or `wait_for_cpu_read(handle)` before
     CPU elementwise code

   First implementation can conservatively synchronize before CPU reads. Later
   stages can replace this with event polling or stream/event waits if timing
   requires it.

3. Make live memory accounting explicit:
   - live CPU bytes by tag
   - peak live CPU bytes by tag
   - live CUDA staged bytes by tag
   - peak live staged bytes by tag
   - stage-cache live/cached bytes
   - D2H bytes by tag
   - H2D staged bytes by tag
   - CPU-read wait count and waited bytes by tag
   - in-flight async source/stage bytes if async is enabled

4. Extend `AsymExecutionStats` with expact counters:
   - `expact_base_cpu_source_forward_calls`
   - `expact_base_cpu_source_dx_calls`
   - `expact_lora_a_pair_forward_grouped_calls`
   - `expact_lora_b_pair_backward_grouped_calls`
   - `expact_small_gemm_fallback_count`
   - `expact_cpu_source_kernel_bytes`
   - `expact_stage_full_activation_bytes`

5. Export these counters:
   - `LFAsymReport.runtime_log_string` prints the new fields.
   - `run_lf_profiled_train.py` writes an `asym_execution_stats` object in the
     source profile JSON.
   - `scripts/testing/profile_qwen3_activation_offload.py` already writes
     `stats`; keep that path compatible with new fields.

6. Improve peak-memory attribution when existing output is insufficient:
   - add expact phase rows to the source profile if peak ownership is unclear
   - label saved tensors and live activations from expert offload separately
     from loss, attention, router, and dense MLP
   - expose allocator peak-growth owner, current live CPU bytes, current live
     CUDA stage bytes, and in-flight async bytes at each expact phase
   - include enough profile JSON fields to tell whether the peak comes from
     expert offload, loss/cross-entropy, optimizer, padding/unpadding, or
     another component

7. Tighten lifetime in current expact without changing math:
   - Add `del` after last use of wide CUDA tensors.
   - Always `release_stage(..., drop_cache=True)` before entering the next
     high-memory region.
   - Release CPU handles once their last dependent gradient is produced.
   - Snapshot before and after each expact block.
   - Add `prof_range` names around every wide region and CPU wait.

8. Add CPU Adam assertions to small integration tests:
   - LoRA compute params remain CUDA.
   - CPU masters and optimizer state remain CPU.
   - `last_step_grad_param_count == last_step_copyback_param_count` after an
     expact backward + optimizer step when all LoRA params receive gradients.

### Risks And Watch Items

- Pinned D2H plus CPU read ordering is easy to get wrong. Stage 0 should prefer
  a conservative synchronization over silent races.
- A global sync before CPU SiLU will hurt timing, but it should not change HBM
  correctness. Later stages can reduce sync scope.
- Async copies can increase peak HBM if staged buffers overlap too aggressively.
  The plan does not allow async prefetch until the live-byte counters prove it
  is bounded.
- The current source profile may not have a direct handle to `LFAsymReport`.
  If so, implement model introspection in `run_lf_profiled_train.py` to collect
  `module.stats.as_dict()` from wrapped Asym modules.

### Validation Before Stage 2

Static audit:

```bash
rg -n "ctx\\.|manager.stage|stage_concat_columns|release_stage|release_cpu|offload\\(|wait_for_cpu|ready_for_cpu" \
  asym_gemm/training/qwen3_moe.py \
  asym_gemm/training/activation_offload.py
```

Unit correctness:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend \
  tests/training/test_asym_cpu_adamw.py \
  tests/lf/test_asym_cpu_adamw_lf_integration.py \
  tests/lf/test_asym_cpu_adamw_args.py
```

Small profile:

```bash
mkdir -p outputs
PYTHONPATH="$PWD" .venv/bin/python scripts/testing/profile_qwen3_activation_offload.py \
  --tokens 1024 \
  --top-k 2 \
  --num-experts 8 \
  --hidden-dim 4096 \
  --intermediate-dim 11008 \
  --rank 8 \
  --warmup 1 \
  --iters 2 \
  --output-json outputs/expact_v3_stage1_small.json
```

LF dry-run validation of labels and CPU Adam flags:

```bash
DRY_RUN=true \
OUTPUT_ROOT="$PWD/outputs/expact_v3_stage1_dryrun" \
GPU_POOL=0 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=8192 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PLOT=false \
RUN_POST=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus 0
```

Stage 1 passes only if:

- no CPU-side activation math reads an offloaded handle before explicit
  completion
- source profile or small profile exposes expact stats and AsymGEMM/GEMM counts
- CPU Adam summaries report `all_masters_on_cpu=true` and
  `all_cuda_params_on_cuda=true`
- peak HBM is not worse than the previous expact profile except for explicitly
  tagged measurement overhead

## Stage 2: Remove Route-Expanded Scatter Saved Output

### Scope

- `asym_gemm/training/moe.py`
  - `scatter_contiguous`
  - `scatter_backward_contiguous`
  - `ContiguousRouteMetadata`
  - `__all__` export list if a new helper is public
- `asym_gemm/training/qwen3_moe.py`
  - `AsymQwen3Experts.forward`
  - `AsymQwen3MoeBlock._compute_routing` as the router no-grad guard source
- Tests:
  - `tests/training/test_toy_moe_lora_sft.py`
  - `tests/training/test_lf_qwen3_asym_backend.py`
- Profiling:
  - the canonical LF script
    `scripts/lf/profile_lora_lf.sh`

### Code Changes

1. Add a router-no-grad scatter autograd path that saves only the tensors needed
   to backpropagate to expert output:

```python
class _ScatterContiguousRouterNoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_output, token_indices, routing_weights, num_tokens):
        flat = expert_output.reshape(routing_weights.numel(), -1)
        weights = routing_weights.reshape(-1, 1)
        out = torch.zeros((num_tokens, flat.shape[1]), device=flat.device, dtype=flat.dtype)
        out.index_add_(0, token_indices, flat * weights)
        ctx.expert_shape = tuple(expert_output.shape)
        ctx.num_tokens = int(num_tokens)
        ctx.save_for_backward(token_indices, routing_weights)
        return out.reshape(num_tokens, *expert_output.shape[1:])

    @staticmethod
    def backward(ctx, grad_output):
        token_indices, routing_weights = ctx.saved_tensors
        grad_flat = grad_output.reshape(ctx.num_tokens, -1)
        gathered = grad_flat.index_select(0, token_indices)
        grad_expert = gathered * routing_weights.reshape(-1, 1)
        return grad_expert.reshape(ctx.expert_shape), None, None, None
```

The target path must not save `expert_output`. It should save only
`token_indices` and detached `routing_weights`, whose size is tiny relative to
`[tokens * top_k, hidden]`.

2. Route `scatter_contiguous` through this path when router weights do not need
   gradients:

```python
if not metadata.routing_weights.requires_grad:
    return _ScatterContiguousRouterNoGrad.apply(
        expert_output,
        metadata.token_indices,
        metadata.routing_weights,
        metadata.num_tokens,
    )
```

If `metadata.routing_weights.requires_grad=True`, keep the current autograd
implementation or an explicit backward that saves `expert_output` and returns
`grad_weights`. The target LF workflow uses router no-grad:
`AsymQwen3MoeBlock._compute_routing` runs under `torch.no_grad()` and detaches
`top_k_weights` unless `router_debug_grad` is enabled.

3. Keep the change at the scatter boundary. Do not alter expert math, LoRA
   math, route sorting, grouped AsymGEMM calls, CPU Adam, or recompute policy.
   This is a lifetime change for the route-expanded scatter output, not a
   fused-kernel comparison.

4. Add counters or profile fields if needed:
   - `scatter_router_nograd_calls`
   - `scatter_saved_expert_output_bytes_avoided`
   - per-layer expected avoided bytes:
     `metadata.num_routes * hidden_size * element_size`

5. Make the saved-tensor profiler easy to verify:
   - before this stage, `forward.layers.N.mlp.experts.scatter_combine` rows
     show `[131072, 2048]` bf16 tensors at `b4_s4096`
   - after this stage, those rows should be gone or replaced only by
     `token_indices`/`routing_weights`-scale tensors

### Risks And Watch Items

- If router gradients are re-enabled, the no-save path cannot compute
  `grad_weights` without `expert_output`. Fall back when
  `routing_weights.requires_grad=True`.
- PyTorch may still keep `expert_output` alive through graph references even if
  saved-tensor hooks no longer report it. Validate with actual peak HBM, not
  only saved-tensor rows.
- `index_add_` backward semantics must match existing autograd for repeated
  token indices and arbitrary route ordering.
- The new autograd function is not allowed to introduce per-expert loops,
  per-expert GEMMs, or smaller GEMM fragmentation. Scatter backward is a gather
  and scale over the route-expanded gradient, followed by the existing grouped
  expert backward.
- Removing this tensor should reduce routed-expert saved activations by up to
  the active-workload scatter size recorded above. If the global peak does not
  move, inspect the new actual peak phase before claiming success.

### Validation Before Stage 3

Static audit:

```bash
rg -n "scatter_contiguous|scatter_backward_contiguous|ScatterContiguous|routing_weights.requires_grad" \
  asym_gemm/training/moe.py \
  asym_gemm/training/qwen3_moe.py
```

Unit correctness:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_toy_moe_lora_sft.py::test_moe_module_scatter_backward_matches_autograd_and_repeated_backward \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend
```

Add or update a direct router-no-grad scatter test in
`tests/training/test_toy_moe_lora_sft.py`:

```text
expert_output.requires_grad=True
routing_weights.requires_grad=False
compare output and expert_output.grad against old scatter_contiguous
assert saved tensors do not include a [num_routes, hidden] expert_output row
```

Primary attribution validation, using the successful `b4_s4096` workload:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v3_stage2_scatter_nosave_b4s4096" \
SFT_ROOT=/workspace/AsymGEMM-SFT \
GPU_POOL=0 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES="none|true" \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PROFILE_MEMORY_ATTRIBUTION=true \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_MEMORY_BREAKDOWN_INTERVAL=1 \
PROFILE_LEVEL=op \
PLOT=false \
RUN_POST=false \
CONTINUE_ON_ERROR=false \
scripts/lf/profile_lora_lf.sh --gpus 0
```

Full active comparison validation, without attribution overhead:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v3_stage2_scatter_nosave_b4s4096_nohook" \
SFT_ROOT=/workspace/AsymGEMM-SFT \
GPU_POOL=0 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES="none|true,gc-exp|false,none|false" \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=10 \
WARMUP_STEPS=5 \
PROFILE_MEMORY_ATTRIBUTION=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_LEVEL=op \
PLOT=false \
RUN_POST=false \
CONTINUE_ON_ERROR=false \
scripts/lf/profile_lora_lf.sh --gpus 0
```

Stage 2 passes only if:

- Stage 0 says the route-expanded scatter/index-add saved tensor is real and
  peak-relevant, or the run explicitly reports it as an expert-block local
  reduction with no global-peak claim
- routed-expert saved activation rows drop materially in the `b4_s4096`
  memory breakdown
- no `scatter_combine` saved tensor row remains at shape `[131072, 2048]`
- `b4_s4096` `none|true` global peak allocated HBM drops versus `126.312 GiB`,
  or the new peak phase is explicitly explained and the expert-block peak drops
- `b4_s4096` `none|true` latency does not regress versus `43.742 s` average
  step unless the HBM reduction is large enough to justify the tradeoff
- any latency increase must be explicitly justified by a meaningful workflow
  HBM reduction; tiny or local-only memory reductions do not justify a large
  slowdown
- AsymGEMM counts and CPU Adam health remain valid

## Stage 3: Remove Full HBM Base Activation Restaging

### Scope

- `asym_gemm/training/qwen3_moe.py`
  - `_ActivationOffloadQwen3ExpertFunction.forward`
  - `_ActivationOffloadQwen3ExpertFunction.backward`
- New or extended Python wrapper:
  - prefer new `asym_gemm/training/exp_act_offload_base.py`
  - update `asym_gemm/training/__init__.py` if public imports are needed
- Existing abstractions:
  - `asym_gemm/training/host_weight.py::HostWeight`
  - `asym_gemm/training/frozen_linear.py::AsymGroupedFrozenLinear`
- Native binding:
  - `csrc/apis/exp_act_offload.hpp`
  - `csrc/exp_act_offload/exp_act_offload_kernels.cu`
  - `csrc/python_api.cpp` only if a new namespace/register path is added
  - `setup.py` only if a new `.cu` file is added instead of extending
    `exp_act_offload_kernels.cu`
- Tests:
  - new `tests/training/test_exp_act_offload_base_cpu_source.py`
  - update `tests/training/test_lf_qwen3_asym_backend.py`

### Code Changes

1. Add grouped CPU-source base forward:

```python
output = grouped_down_base_cpu_source(
    act_cpu.tensor,
    layer.down_base.host_weight.tensor,
    offsets,
    experts,
    metadata=lora_metadata,
    stats=layer.stats,
)
```

Contract:

```text
act_cpu: pinned CPU BF16 [M,I]
down_base_weight: pinned CPU BF16 HostWeight [E,H,I]
offsets/experts: grouped route metadata
return: CUDA BF16 [M,H]
```

2. Add grouped CPU-source gate/up base dX:

```python
grad_packed = grouped_gate_up_base_dx_cpu_source(
    grad_gate_cpu.tensor,
    grad_up_cpu.tensor,
    layer.gate_up_base.host_weight.tensor,
    offsets,
    experts,
    metadata=lora_metadata,
    stats=layer.stats,
)
```

Contract:

```text
dgate_cpu: pinned CPU BF16 [M,I]
dup_cpu: pinned CPU BF16 [M,I]
gate_up_base_weight: pinned CPU BF16 HostWeight [E,2I,H]
offsets/experts: grouped route metadata
return: CUDA BF16 [M,H]
```

3. Native scheduling requirements:
   - one grouped kernel per projection, not one launch per expert
   - consume CPU activation tiles directly
   - consume CPU `HostWeight` tiles directly or via bounded tile scratch
   - write only the final `[M,H]` output/grad to HBM
   - no full CUDA `[M,I]` or `[M,2I]` source tensor
   - increment `expact_base_cpu_source_forward_calls` and
     `expact_base_cpu_source_dx_calls`

4. Remove from expact:
   - `manager.stage(act_cpu, tag="act_for_down_base")`
   - `manager.stage_concat_columns(..., tag="dgate_up_for_gate_up_base")`
   - `manager.release_stage(grad_gate_up, ...)`

5. Update the Qwen3 unit test:
   - assert `stage_peak_by_tag` does not contain `act_for_down_base`
   - assert `stage_peak_by_tag` does not contain `dgate_up_for_gate_up_base`
   - assert new base CPU-source counters are positive

### Risks And Watch Items

- This is the biggest uncertainty: the kernel has CPU activations and CPU base
  weights. If both are fetched over PCIe/NVLink host mapping, bandwidth may
  dominate. Correctness and HBM reduction still come first.
- `HostWeight.metadata.can_map_host_memory` may be false or unknown on some
  systems. The kernel must fail closed rather than silently staging full HBM.
- Route metadata uses cumulative offsets with a sentinel expert entry. Tests
  must cover empty groups, repeated experts, dense experts, and uneven row
  counts.
- If native code needs a new `.cu` file, update `setup.py`; otherwise editable
  builds will not compile it.

### Validation Before Stage 4

Native correctness:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_exp_act_offload_base_cpu_source.py
```

End-to-end unit:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend
```

Static no-restage audit:

```bash
rg -n "act_for_down_base|dgate_up_for_gate_up_base|stage_concat_columns|manager.stage\\(" \
  asym_gemm/training/qwen3_moe.py \
  asym_gemm/training/activation_offload.py
```

Small profile:

```bash
mkdir -p outputs
PYTHONPATH="$PWD" .venv/bin/python scripts/testing/profile_qwen3_activation_offload.py \
  --tokens 1024 \
  --top-k 2 \
  --num-experts 8 \
  --hidden-dim 4096 \
  --intermediate-dim 11008 \
  --rank 8 \
  --warmup 1 \
  --iters 2 \
  --output-json outputs/expact_v3_stage3_small.json
```

Stage 3 passes only if:

- both old full-stage tags are absent
- base CPU-source counters are positive
- HBM peak drops relative to the prior accepted stage and the Stage 1 guardrail
  baseline
- trace shows grouped base CPU-source launches, not one launch per expert

## Stage 4: Replace Wide LoRA-B Staging With CPU-Source Kernels

### Scope

- `asym_gemm/training/qwen3_moe.py`
  - `_ActivationOffloadQwen3ExpertFunction.backward`
  - remove expact dependence on `_grouped_lora_weight_grads_torch`
  - remove expact dependence on `_grouped_lora_cuda_view`
- `asym_gemm/training/exp_act_offload_lora.py`
  - `grouped_lora_b_backward_cpu_source`
  - `_try_lora_b_ds_cpu_left`
  - `ASYM_EXPACT_LORA_B_SPLIT` auto/force/disable guard
  - `require_expert_activation_offload_kernels`
- `asym_gemm/training/frozen_linear.py`
  - `AsymExecutionStats`
- Native:
  - `csrc/apis/exp_act_offload.hpp`
  - `csrc/exp_act_offload/exp_act_offload_kernels.cu`
  - `sm100_grouped_lora_b_grad_b_bf16_cpu_source`
- Profiling:
  - add or repair a standalone LoRA-B CPU-source profiler before using NCU for
    this stage; do not depend on stale helpers that are not present in the tree
  - `scripts/testing/ncu_exp_act_offload_kernel_profile.sh` only after its
    target profiler exists and is verified with `bash -n`
- Tests:
  - update `tests/training/test_exp_act_offload_native.py`

### Code Changes

1. Stage 4A candidate: remove full HBM gate/up LoRA-B staging from
   `qwen3_moe.py`. Gate/up LoRA-B backward must consume `grad_gate_cpu` and
   `grad_up_cpu` directly:

```python
dS_gate, grad_gate_lora_B = grouped_lora_b_backward_cpu_source(
    grad_gate_cpu.tensor,
    gate_low_rank,
    gate_lora_B,
    offsets,
    experts,
    scale=layer.lora_scale,
    stats=layer.stats,
    tag="gate",
)
```

2. Stage 4A candidate: split the target wide/low-rank LoRA-B path:

```text
dS      = grouped CPU-left AsymGEMM(grad_out_cpu, lora_b.T)
grad_b  = grouped native CPU-source reducer(grad_out_cpu, low_rank)
```

If this split is reintroduced, gate it for `out_dim >= 512`, `rank <= 64`, and at least
`256 * 512 * sizeof(bf16)` CPU-source bytes. This covers the canonical LF
Qwen3 shape (`rank=64`, expert width `768`) and the wider low-rank shapes where
standalone timing shows the split wins, while preserving fallback for the known
small-width `rank=64,width=256` loss case. Set `ASYM_EXPACT_LORA_B_SPLIT=1` to
force the split for profiling, or `ASYM_EXPACT_LORA_B_SPLIT=0` to force the old
combined native helper. The split is accepted only if the canonical LF
`none|true` run improves latency or lowers peak HBM versus the active
activation-offload baseline.

3. Stage 4B, still open: replace the `grad_b` reducer internals with a tiled
   implementation:
   - no full FP32 `grad_b_acc`
   - no atomic-heavy per-output-element reduction when a tiled reduction can be
     used
   - use CPU `grad_gate_cpu` and `grad_up_cpu` directly
   - tile accumulators are allowed; full-output FP32 workspaces are not
   - low-rank CPU input is preferred; low-rank CUDA staging is allowed only if
     Stage 1 counters show it does not move peak HBM
   - increment LoRA-B native/helper counters

4. Keep CPU Adam compatible:
   - returned gradients for `gate_lora_B`, `up_lora_B`, and `down_lora_B` must
     be normal CUDA tensors with the same dtype/shape contract as current
     backward
   - do not write directly to `AsymCPUAdamW` CPU masters
   - do not move LoRA params to CPU to make the kernel simpler

### Risks And Watch Items

- Numerical tolerance may change because tiled reductions reorder
  accumulation. Define tolerances in tests against the current torch reference.
- The split reads `grad_out_cpu` twice for the target LoRA-B path: once through
  grouped CPU-left AsymGEMM for `dS`, once through the native `grad_b` reducer.
  This is acceptable only while it improves latency without increasing HBM
  stage bytes; keep `expact_cpu_source_kernel_bytes` visible and compare
  `expact_lora_b_ds_cpu_left_calls` against LoRA-B grouped-call counts.
- The split materializes a contiguous transposed LoRA-B weight for the CPU-left
  `dS` AsymGEMM. This is parameter-shaped, not activation-shaped, and should
  not erase the memory win, but it must stay under memory attribution.
- Down LoRA-B consumes `grad_output`, which already exists in HBM. It may not
  need CPU offload unless peak attribution shows down LoRA-B staging matters.
  Gate/up must not use full HBM gradients.
- If `dS` is produced in BF16 for memory, LoRA-A grad accuracy must be checked.
  If FP32 `dS` is required, it must be final-sized by mathematical necessity
  and counted as such.

### Validation Before Stage 5

Native correctness:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_exp_act_offload_native.py
```

Qwen expact correctness:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend
```

Static audit:

```bash
rg -n "_grouped_lora_weight_grads_torch|_grouped_lora_cuda_view|dgate_for_lora_b|dup_for_lora_b" \
  asym_gemm/training/qwen3_moe.py \
  asym_gemm/training/exp_act_offload_lora.py \
  csrc/exp_act_offload
```

Representative isolated LoRA-B timing:

- First add a checked-in standalone profiler for the exact accepted wrapper
  shape, or repair `scripts/testing/ncu_exp_act_offload_kernel_profile.sh` so
  it targets an existing profiler.
- Then time at least these shapes before a full LF run:
  `m=1024,n=11008,rank=8,groups=8`,
  `m=24576,n=768,rank=64,groups=128`, and
  `m=256,n=256,rank=64,groups=8`.
- Run NCU only on that isolated helper so host-memory load reuse, SMEM reuse,
  occupancy, and stall reasons are visible without LF noise.

CPU Adam one-step checks:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_asym_cpu_adamw.py \
  tests/lf/test_asym_cpu_adamw_lf_integration.py
```

Canonical LF target-policy rerun after changing the auto split guard:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v3_stage4_rank64_split_b4s6144" \
GPU_POOL=0 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES=none|true \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=6144 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PROFILE_MEMORY_ATTRIBUTION=true \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_MEMORY_BREAKDOWN_INTERVAL=1 \
PLOT=false \
RUN_POST=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus 0
```

Target-policy iteration runs should set `ASYMM_EXP_ACT_POLICIES=none|true`.
Full comparison runs must use `none|true,gc-exp|false,none|false`.

Stage 4 passes only if:

- expact backward no longer calls `_grouped_lora_weight_grads_torch` for LoRA-B
- expact backward no longer stages full gate/up gradients for LoRA-B
- Stage 4A passes when the target rank-64/width-768 LoRA-B helper uses grouped
  CPU-left AsymGEMM for `dS`, keeps low-rank-only HBM stages, and improves
  latency versus the old combined CPU-source helper
- the active LF `none|true` run either lowers peak HBM meaningfully without
  material latency blow-up, or improves latency with flat/lower HBM
- local helper latency wins are not accepted if the full LF workflow has the
  same memory and worse latency
- Stage 4B is not complete until no full FP32 `grad_b_acc` allocation remains
  in the CPU-source LoRA-B path
- CPU Adam sees normal LoRA gradients and updates all LoRA params that had grads

## Stage 5: Tile LoRA-A Gradient Reductions

### Scope

- `asym_gemm/training/exp_act_offload_lora.py`
  - `grouped_lora_a_grad_cpu_right`
  - `grouped_lora_a_pair_grad_cpu_right`
- Native:
  - `csrc/apis/exp_act_offload.hpp`
  - `csrc/exp_act_offload/exp_act_offload_kernels.cu`
- `asym_gemm/training/qwen3_moe.py`
  - no callsite changes if wrapper names stay stable
- Tests:
  - update `tests/training/test_exp_act_offload_native.py`
  - new `tests/training/test_exp_act_offload_lora_a_grad_tiled.py`

### Code Changes

1. Replace full-accumulator native LoRA-A grad internals:

```text
sm100_grouped_lora_a_grad_bf16_cpu_right
sm100_grouped_lora_a_pair_grad_bf16_cpu_right
```

The wrapper names can stay the same, but implementation must not allocate:

```text
grad_acc
grad_gate_acc
grad_up_acc
```

2. Tiled contract:

```text
dS_cuda: [M,r]
X_cpu: pinned CPU BF16 [M,K]
grad_A_cuda: [E,r,K]
offsets/experts: grouped route metadata
```

3. Pair policy:
   - one paired gate/up kernel consumes `dS_gate`, `dS_up`, and `X_cpu`
   - the same `X_cpu` tile is reused for both gate and up reductions
   - output two final gradient tiles
   - increment paired LoRA-A grad counters separately from single calls

4. Reduction policy:
   - accumulate in registers/shared memory or tile scratch
   - write the final tile to `grad_A_cuda`
   - no one-kernel-per-expert fallback in normal path

### Risks And Watch Items

- Atomic accumulation may be nondeterministic. Tests should use tolerant
  numerical comparison and fixed seeds.
- Rank and hidden dimensions may not always be multiples of the ideal tile.
  Keep existing fail-closed alignment checks unless the kernel supports tails.
- Full final `grad_A` is allowed because it is the real parameter gradient. Full
  extra FP32 grad workspaces are not allowed.

### Validation Before Stage 6

Native correctness:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_exp_act_offload_lora_a_grad_tiled.py \
  tests/training/test_exp_act_offload_native.py::test_grouped_lora_a_grad_cpu_right_matches_reference \
  tests/training/test_exp_act_offload_native.py::test_grouped_lora_a_pair_grad_cpu_right_matches_reference
```

Static audit:

```bash
rg -n "grad_acc|grad_gate_acc|grad_up_acc|torch::zeros" \
  csrc/exp_act_offload
```

Small profile:

```bash
mkdir -p outputs
PYTHONPATH="$PWD" .venv/bin/python scripts/testing/profile_qwen3_activation_offload.py \
  --tokens 1024 \
  --top-k 2 \
  --num-experts 8 \
  --hidden-dim 4096 \
  --intermediate-dim 11008 \
  --rank 8 \
  --warmup 1 \
  --iters 2 \
  --output-json outputs/expact_v3_stage5_small.json
```

Stage 5 passes only if:

- no full FP32 LoRA-A accumulator remains in the expact native path
- grouped kernel counts remain bounded and do not scale with expert count
- peak HBM and backward timing improve or the profile identifies another
  dominant bottleneck

## Stage 6: True Paired Gate/Up Scheduling

### Scope

- `asym_gemm/training/exp_act_offload_lora.py`
  - `grouped_lora_a_pair_forward_cpu_left`
  - paired backward wrappers added in Stage 4 and Stage 5
- `asym_gemm/training/qwen3_moe.py`
  - `_ActivationOffloadQwen3ExpertFunction.forward`
  - `_ActivationOffloadQwen3ExpertFunction.backward`
- `asym_gemm/training/lora.py`
  - `grouped_expert_lora_pair`
  - metadata reuse and `torch.cat` accounting
- Native:
  - `csrc/apis/exp_act_offload.hpp`
  - `csrc/exp_act_offload/exp_act_offload_kernels.cu`
- Tests:
  - update `tests/training/test_cpu_left_lora.py`
  - new `tests/training/test_exp_act_offload_gate_up_paired.py`

### Code Changes

1. Stage 6A candidate: replace the pair-forward wrapper that called the
   single CPU-left kernel twice with one grouped CPU-left call over a
   concatenated gate/up LoRA-A weight:

```python
gate_low_rank, up_low_rank = grouped_lora_a_pair_forward_cpu_left(
    x_cpu.tensor,
    gate_lora_A,
    up_lora_A,
    offsets,
    experts,
    metadata=lora_metadata,
    stats=layer.stats,
    tag="gate_up",
)
```

Interim implementation:

```python
combined_weight = torch.cat((lora_a_gate, lora_a_up), dim=1)
combined = grouped_lora_a_forward_cpu_left(source_cpu, combined_weight, ...)
gate_low_rank, up_low_rank = combined.split((gate_rank, up_rank), dim=-1)
```

This is still one grouped CPU-left AsymGEMM and one CPU-source read of `X_cpu`.
The temporary is parameter-shaped and the outputs are low-rank; it must not
become a full activation staging path. A previous experiment did not beat the
active `none|true` latency baseline under a flat-memory result, so this
candidate must be re-landed only with a measured LF improvement.

2. Stage 6B, still open: replace the Python-side weight concatenation with a
   native paired CPU-left schedule:
   - load one `X_cpu` tile
   - compute gate and up low-rank tiles before advancing the source tile
   - one grouped paired call, not two independent source passes
   - reuse the CPU tile in SMEM across both LoRA-A weight tiles

3. Backward schedule:
   - use paired LoRA-B from Stage 4
   - use paired LoRA-A grad from Stage 5
   - if `need_grad_packed`, compute gate/up LoRA `dX` through grouped calls and
     add them into the single final `grad_packed`
   - do not recreate `stage_concat_columns`

4. Track allocations:
   - count paired calls distinctly from two singles
   - record CPU source bytes loaded for paired gate/up
   - tag any `torch.cat` in `grouped_expert_lora_pair`

### Risks And Watch Items

- Pairing is mostly a timing and bandwidth fix, but it can reduce HBM by
  shortening overlapping low-rank/delta lifetimes. Do not make correctness
  harder unless profiles show repeated CPU source reads are material.
- Stage 6A uses a parameter-shaped `torch.cat` for LoRA-A weights. The measured
  LF peak stayed unchanged, but a native Stage 6B kernel is still preferred if
  NCU or LF timing shows the concatenation limits the benefit.
- `grouped_expert_lora_pair` currently uses concatenation for CUDA low-rank
  inputs and weights. That may be acceptable for low-rank tensors, but it must
  be counted and revisited if it moves peak.

### Validation Before Stage 7

Correctness:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_exp_act_offload_gate_up_paired.py \
  tests/training/test_cpu_left_lora.py \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend
```

Static audit:

```bash
rg -n "grouped_lora_a_pair_forward_cpu_left|stage_concat_columns|torch.cat" \
  asym_gemm/training/exp_act_offload_lora.py \
  asym_gemm/training/qwen3_moe.py \
  asym_gemm/training/lora.py
```

Stage 6 passes only if:

- stats show one physical grouped CPU-left LoRA-A forward call for gate/up,
  not two single calls
- no per-expert GEMM launch pattern appears in source/NSYS stats
- paired CPU source bytes are lower than two independent passes
- the active LF `none|true` run improves latency with flat/lower HBM, or lowers
  peak HBM meaningfully without material latency blow-up
- do not keep paired scheduling if it only reduces call/source-byte counters
  while full-workflow memory stays flat and latency regresses
- Qwen3 expact remains numerically matched to the torch backend

## Stage 7: Hidden Materialization And Workspace Cleanup

### Scope

- `asym_gemm/training/activation_offload.py`
  - add optional bounded workspace/pool policy
  - global `_CPU_BUFFER_POOL`
- `asym_gemm/training/frozen_linear.py`
  - `_pad_grouped_input_for_asym`
  - `_unpad_grouped_output`
  - `_dispatch_grouped_nt`
- `asym_gemm/training/cpu_left.py`
  - `_pad_cpu_left_grouped_input_for_asym`
  - `_unpad_grouped_output`
  - `grouped_expert_lora_cpu_left`
- `asym_gemm/training/lora.py`
  - `prepare_grouped_lora_metadata`
  - `grouped_expert_lora_pair`
- `asym_gemm/training/qwen3_moe.py`
  - wide tensor lifetimes in expact forward/backward
- `asym_gemm/profiling/lf_trace.py`
  - expose allocation tags if source profile lacks enough detail
- Tests:
  - update focused allocation-count tests in `tests/training/test_cpu_left_lora.py`
  - update `tests/training/test_cpu_resident_frozen_base.py`
  - add expact allocation regression checks in
    `tests/training/test_lf_qwen3_asym_backend.py`

### Code Changes

1. Add bounded workspace ownership if profiling shows pool/cached buffers are
   moving peak:

```text
ActivationOffloadWorkspace
  get_cuda_scratch(tag, shape_or_tile, dtype)
  get_cpu_scratch(tag, shape_or_tile, dtype, pinned=True)
  release_cuda_scratch(tag)
  release_cpu_scratch(tag)
  snapshot_live_bytes()
```

2. Tag and shorten materializations:
   - `_unpad_grouped_output` padded and unpadded overlap
   - CPU-left padded source buffers
   - `torch.cat` in `grouped_expert_lora_pair`
   - wide `.contiguous()` on routed tensors
   - active expert `index_select`
   - LoRA deltas `gate_delta`, `up_delta`, `down_delta`

3. Cleanup rules:
   - scratch is tile-sized or low-rank-sized unless explicitly justified
   - scratch has profile-visible tags
   - CPU pinned cache has a max cached-byte policy or explicit clear hook
   - final output allocation is allowed
   - duplicate padded/unpadded output overlap is allowed only for the shortest
     possible scope

4. If forward peak is still high after Stages 2-6, add add-into-output variants:
   - LoRA-B delta directly into base output where safe
   - gate/up delta addition immediately followed by CPU offload and delete
   - do not start direct-to-CPU base gate/up output unless profiles show this is
     now the remaining dominant peak

### Risks And Watch Items

- Some padding/unpadding is required by existing AsymGEMM kernels. Removing it
  without changing kernel contracts can break correctness.
- CPU pinned memory pressure can become the next bottleneck. Track CPU live and
  cached bytes separately from HBM.
- Async double buffering should be enabled only after counters prove max
  in-flight HBM remains below the synchronous path.

### Validation Before Stage 8

Static audit:

```bash
rg -n "act_for_down_base|dgate_up_for_gate_up_base|grad_acc|grad_gate_acc|grad_up_acc|dS_acc|grad_b_acc|stage_concat_columns|_grouped_lora_weight_grads_torch|torch.cat|contiguous\\(|index_select|_unpad_grouped_output" \
  asym_gemm/training/qwen3_moe.py \
  asym_gemm/training/activation_offload.py \
  asym_gemm/training/exp_act_offload_lora.py \
  asym_gemm/training/cpu_left.py \
  asym_gemm/training/frozen_linear.py \
  asym_gemm/training/lora.py \
  csrc/exp_act_offload
```

Unit allocation regressions:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_cpu_left_lora.py \
  tests/training/test_cpu_resident_frozen_base.py \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend
```

Memory-breakdown profile:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v3_stage7_cleanup_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=8192 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_MEMORY_BREAKDOWN_MODULES=experts,lora,loss \
PLOT=false \
RUN_POST=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus 3
```

Stage 7 passes only if:

- no untagged full-width materialization remains in expact hot regions
- `max_stage_bytes_live` is bounded and does not include full activation stages
- source profile distinguishes expert-block peak from loss/cross-entropy peak
- grouped call counts remain bounded

## Stage 8: Final Fair Comparison And CPU Adam Acceptance

### Scope

- `/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh`
  - only if output naming or counter propagation is missing
- `scripts/lf/run_lf_lora_sft.sh`
  - only if `asym_cpuadamwds` profile logs omit needed CPU Adam markers
- `scripts/lf/run_lf_profiled_train.py`
  - ensure final profile includes `asym_execution_stats`,
    `activation_offload_stats`, and `asym_cpu_adamw`
- `scripts/lf/postprocess_lf_profile_artifacts.py`
  - only if CSV/plot artifacts need the new counters
- `scripts/lf/check_deepspeed_cpuadam_run.py`
  - no change expected; use manually when DeepSpeedCPUAdam markers are present

### Code Changes

1. Ensure the final profile JSON reports:
   - peak allocated/reserved HBM
   - expert-block peak separate from loss peak
   - `activation_offload_stats`
   - `asym_execution_stats`
   - CPU pinned live/cached bytes
   - D2H/H2D bytes by tag
   - CPU-read wait counts and bytes
   - grouped CPU-source kernel counts
   - base AsymGEMM forward/dX counts
   - LoRA-A/LoRA-B grouped counts
   - fallback reasons and small-GEMM fallback count
   - `asym_cpu_adamw` summary

2. CPU Adam acceptance:
   - `asym_cpu_adamw.enabled=true`
   - `all_masters_on_cpu=true`
   - `all_cuda_params_on_cuda=true`
   - `last_step_grad_param_count > 0`
   - `last_step_grad_param_count == last_step_copyback_param_count`
   - `backend=deepspeed` for the primary `asym_cpuadamwds` run
   - if the profile/log contains a DeepSpeedCPUAdam runtime marker, this manual
     checker must pass:

```bash
.venv/bin/python scripts/lf/check_deepspeed_cpuadam_run.py \
  --profile-json "$PROFILE_SOURCE_JSON" \
  --train-log "$TRAIN_LOG" \
  --require-enabled
```

### Risks And Watch Items

- `check_deepspeed_cpuadam_run.py` is primarily wired for `zero3_cpuadam`.
  For `asym_cpuadamwds`, the stronger required signal is the
  `asym_cpu_adamw` source-profile summary from `AsymCPUAdamW`.
- If `4x8192` still OOMs after Stage 7, record whether the failure is inside
  expert blocks or later loss/cross-entropy. The claim is not satisfied by a
  loss-side OOM explanation alone, but the next fix depends on where the peak
  moved.
- If async is introduced, compare sync vs async with identical kernels. Async
  must not increase peak HBM through excessive prefetch overlap.

### Final Validation

Canonical active workflow comparison, `b4_s4096`:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v3_final_b4s4096_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=10 \
WARMUP_STEPS=5 \
PLOT=false \
RUN_POST=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus 3
```

Optional stress workflow comparison, `b2_s8192`:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v3_final_b2s8192_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=8192 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PLOT=false \
RUN_POST=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus 3
```

Optional stress workflow comparison, `b4_s8192`:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v3_final_b4s8192_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=8192 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PLOT=false \
RUN_POST=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus 3
```

Optional global recompute lower bound, `b2_s8192`:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v3_final_global_recompute_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|recomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES=none|false \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=8192 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PLOT=false \
RUN_POST=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus 3
```

Optional NSYS kernel-count run after the canonical source profile passes:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v3_final_nsys_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=nsys,source \
ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PLOT=false \
RUN_POST=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus 3
```

Final acceptance requires:

- `none|true` peak allocated HBM is below `gc-exp|false` in the active
  `b4_s4096` workflow comparison
- `none|true` at `b4_s4096` is below the current `126.312 GiB` baseline, or has
  the same memory with a measured latency improvement versus `43.742 s` average
  step that does not erase the memory win
- same or higher HBM with worse latency is rejected; trivial HBM reduction with
  a large latency increase is also rejected
- toy-only or microbenchmark-only wins are not accepted without the active LF
  workflow result
- optional `b2_s8192` and `b4_s8192` stress runs fit with clear HBM margin, or
  fail later for a specifically identified non-expert peak after expert-block
  HBM is below recompute
- no full activation stage tags remain
- no full FP32 expact accumulator workspaces remain
- no one-launch-per-expert GEMM pattern appears in source/NSYS traces
- CPU Adam summary proves CPU masters/state and CUDA LoRA compute params are in
  the intended places
- timing regression is explained by useful transfer/compute work, not long
  CPU-idle or GPU-idle gaps caused by ungrouped scheduling
