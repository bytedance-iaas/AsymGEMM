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
BACKEND_SPECS="asym_cpuadamwds|norecomp"
ASYMM_EXP_ACT_POLICIES="none|true,gc-exp|false,none|false"
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
| S2 | Tight CPU handle lifetime and bounded pinned pool | release CPU handles at last use, report post-release stats, and cap/clear cached pinned CPU buffers |
| S3 | Remove route-expanded scatter saved output | router-no-grad scatter path that does not save the full `[routes, hidden]` expert output |
| S4 | Mandatory pause gate after scatter no-save | active LF comparison, attribution verdict, and explicit stop before any S5+ work |
| S5 | Integrated gate/up backward redesign, only if proven peak-relevant | remove/avoid `dgate_up_for_gate_up_base` as a coordinated base-dX plus LoRA-B plan; no standalone CPU-source LoRA-B retry |
| S6 | Conditional LoRA-A accumulator tiling | no full FP32 LoRA-A accumulator workspaces, only if attribution proves they are peak-relevant |
| S7 | Deferred native paired gate/up tile reuse | latency-only CPU-tile reuse path, not a repeat of the reverted paired LoRA experiment |
| S8 | Hidden materialization and workspace cleanup | profile-visible bounded scratch, no untagged full-width expact materializations |
| S9 | Final fair comparison and CPU Adam acceptance | canonical `none|true,gc-exp|false,none|false` LF comparison and CPU Adam pass/fail evidence |

Current active comparison baseline, recorded 2026-06-13 from the exact LF
workflow:

- command:
  `bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh`
- output root:
  `/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/profiling/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b4_s4096_w5_s10_r64_a16_drop000`
- model/workload: `Qwen/Qwen3-30B-A3B`, `b4_s4096`, `logical_qlen=16384`
- dataset: `asym_long_sft_smoke__qwen3-30b-a3b__s4096`
- backend/profiler: `asym_cpuadamwds|norecomp`, `PROFILERS=source`
- policy axis: `ASYMM_EXP_ACT_POLICIES="none|true,gc-exp|false,none|false"`
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
ASYMM_EXP_ACT_POLICIES="none|true,gc-exp|false,none|false" \
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
- CPU activation handles are released at last use, not only at the end of
  backward. Post-backward live CPU-owned bytes must return to zero.
- Cached pinned CPU buffers are bounded and profile-visible. Releasing a CPU
  handle may return storage to the pool, but the pool must not grow without a
  configured cap or clear hook.
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
  Current code releases CPU handles at the end of backward, but that is too
  coarse: handles remain live after last use, `_last_activation_offload_stats`
  is snapshotted before CPU releases, and `_CPU_BUFFER_POOL` is global and
  unbounded.
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

V2 repeat-avoidance map:

- S2 is CPU lifetime/pool cleanup. It is not a memory-claim stage; it must make
  CPU handle release and pinned-pool caching bounded before HBM work continues.
- S3 is the new primary HBM target. It attacks route-expanded scatter output
  lifetime, which was not the accepted v2 kernel rewrite path.
- S4 is not an implementation stage. It is the stop/go measurement gate after
  S3. It must select one next target from the measured peak owner, or stop
  expert-side memory work if the peak moved outside the expert block.
- S5 is not the failed standalone LoRA-B staging/split work. It is only an
  integrated gate/up backward redesign, and only if S4 proves
  `dgate_up_for_gate_up_base` is peak-relevant. CPU-source LoRA-B by itself is
  forbidden while the full gate/up base-dX concat remains live.
- S6 runs only if S4 or S5 shows full LoRA-A accumulator scratch is a peak
  owner. It is not a generic "make the kernel nicer" task.
- S7 is deferred latency work only. Do not re-land the v2 paired LoRA-A/call
  reduction; only a native tile-reuse design can be tried, and only if active LF
  timing improves at flat/lower HBM.
- S8 remains valid only if attribution shows hidden full-width workspaces,
  padding/unpadding overlap, or allocator/workspace caches are peak
  contributors.

Keep the three baseline categories separate:

- canonical workflow comparison:
  `BACKEND_SPECS=asym_cpuadamwds|norecomp` and
  `ASYMM_EXP_ACT_POLICIES="none|true,gc-exp|false,none|false"`
- global recompute lower bound:
  `BACKEND_SPECS=asym_cpuadamwds|recomp` and
  `ASYMM_EXP_ACT_POLICIES="none|false"`
- optional global-plus-expert recompute sanity check:
  `BACKEND_SPECS=asym_cpuadamwds|recomp` and
  `ASYMM_EXP_ACT_POLICIES="gc-exp|false"`

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

### Why This Stage Is Needed

Stage 0 does not directly reduce HBM. It prevents false wins by identifying the
actual allocator peak owner before memory-changing code lands. It can unlock
HBM reduction only by proving which expert tensors are peak-live and movable.

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
OUTPUT_ROOT="$PWD/outputs/expact_v3_stage0_diag_b4s4096_$(date -u +%Y%m%dT%H%M%SZ)" \
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
- the output states whether Stage 3 scatter no-save is peak-relevant or only a
  non-peak/local lifetime improvement

## Stage 1: Measurement, Lifetime, And CPU Adam Guardrails

### Why This Stage Is Needed

Stage 1 does not directly reduce HBM. It makes later reductions enforceable by
making activation lifetimes, staged HBM bytes, D2H/H2D traffic, grouped
AsymGEMM/GEMM counts, fallback counts, and CPU Adam health visible. Without
this, a change can hide a leak, introduce per-expert small GEMMs, or break CPU
Adam while appearing to save memory.

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
ASYMM_EXP_ACT_POLICIES="none|true,gc-exp|false,none|false" \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
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

## Stage 2: Tight CPU Handle Lifetime And Bounded Pinned Pool

### Why This Stage Is Needed

This stage does not claim a direct HBM win. It prevents the activation-offload
path from turning HBM savings into unbounded CPU pinned-memory growth or hidden
latency. Current code releases CPU handles only at the end of backward, records
activation-offload stats before those releases, and returns released CPU tensors
to a global unbounded `_CPU_BUFFER_POOL`. That is logically safe for one step,
but it is not tight lifetime management and it can hide leaks or excessive
cached pinned memory across steps.

This stage must prove that earlier CPU release and bounded caching do not break
the main memory goal: active `none|true` HBM must stay flat/lower, latency must
not materially regress, AsymGEMM/GEMM counts must not blow up, and CPU Adam must
remain correct.

### Scope

- `asym_gemm/training/activation_offload.py`
  - `_CPU_BUFFER_POOL`
  - `_alloc_cpu`
  - `_return_cpu`
  - `CPUActivationHandle`
  - `ActivationOffloadStats`
  - `ActivationOffloadManager.release_cpu`
  - `ActivationOffloadManager.snapshot`
  - new pool stats/clear helpers if needed
- `asym_gemm/training/qwen3_moe.py`
  - `_ActivationOffloadQwen3ExpertFunction.backward`
  - release CPU handles at last use instead of only in the final loop
  - record both pre-release and post-release activation-offload stats
- `scripts/lf/run_lf_profiled_train.py`
  - include post-release CPU live/cached bytes if missing from source profile
- `scripts/testing/profile_qwen3_activation_offload.py`
  - include live and cached CPU pool stats in the JSON output
- Tests:
  - `tests/training/test_lf_qwen3_asym_backend.py`
  - add or extend focused tests for CPU release and pool bounds

### Code Changes

1. Add explicit CPU-pool accounting:

```text
cpu_live_bytes
cpu_live_peak_bytes
cpu_live_bytes_by_tag
cpu_live_peak_bytes_by_tag
cpu_pool_cached_bytes
cpu_pool_cached_bytes_by_shape
cpu_pool_num_cached_tensors
cpu_pool_evictions
cpu_pool_max_cached_bytes
```

2. Bound the global CPU pool:
   - add an env/config limit such as `ASYM_EXPACT_CPU_POOL_MAX_BYTES`
   - default should be conservative and documented
   - when returning a tensor would exceed the limit, drop it instead of caching
   - add a test-only or profile-only clear helper, e.g.
     `clear_activation_offload_cpu_pool()`
   - never keep tensors with stale CUDA dependencies or non-contiguous views

3. Move `release_cpu()` calls to last use:
   - release `ctx.down_low_rank_cpu` after `grad_down_lora_B`
   - release `ctx.act_cpu` after `grad_down_lora_A`
   - release `grad_act_cpu` after `_activation_offload_cpu_silu_backward`
   - release `ctx.gate_cpu` and `ctx.up_cpu` after CPU SiLU backward
   - release `ctx.gate_low_rank_cpu` after gate LoRA-B `dB`
   - release `ctx.up_low_rank_cpu` after up LoRA-B `dB`
   - release `ctx.x_cpu` after paired gate/up LoRA-A grad
   - release `grad_gate_cpu` and `grad_up_cpu` after their final consumer
   - keep the final cleanup loop as idempotent safety, but it should release
     zero additional live bytes in the normal path

4. Fix stats timing:
   - record `activation_offload_stats_pre_release` before final cleanup only if
     useful for debugging
   - set `_last_activation_offload_stats` from the post-release snapshot
   - source profile must expose both live and cached CPU bytes so a post-release
     live value of zero does not hide a huge cached pinned pool

5. Add lifetime assertions in tests:
   - after backward, `cpu_owned_bytes == 0`
   - final cleanup loop is idempotent
   - cached CPU bytes are `<= ASYM_EXPACT_CPU_POOL_MAX_BYTES`
   - clearing the pool returns cached bytes to zero
   - HBM stage bytes are not increased by CPU lifetime cleanup

### Risks And Watch Items

- Releasing a CPU handle before its last CPU-source kernel consumer is a
  correctness bug. Move one handle at a time and validate against the torch
  backend.
- A too-small CPU pool can increase CPU allocation overhead and hurt latency.
  Compare pool disabled, default bounded pool, and current unbounded behavior if
  latency regresses.
- CPU memory reduction is not an HBM win. Do not count this stage as progress on
  the core claim unless HBM also stays flat/lower and later HBM stages still
  pass.
- If async is added later, early release must wait until any CPU-source GPU
  consumer has finished reading the CPU tile. This stage should keep sync
  semantics conservative until async live-byte accounting exists.

### Validation Before Stage 3

Unit correctness and lifetime:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend
```

Add or update focused tests in `tests/training/test_lf_qwen3_asym_backend.py`:

```text
run one activation-offload forward/backward
assert post-release activation_offload_stats["cpu_owned_bytes"] == 0
assert final cleanup releases zero additional bytes in the normal path
assert cached CPU pool bytes are present and <= configured cap
clear CPU pool and assert cached bytes == 0
```

Small profile with pool stats:

```bash
mkdir -p outputs
ASYM_EXPACT_CPU_POOL_MAX_BYTES=$((8 * 1024 * 1024 * 1024)) \
PYTHONPATH="$PWD" .venv/bin/python scripts/testing/profile_qwen3_activation_offload.py \
  --tokens 1024 \
  --top-k 2 \
  --num-experts 8 \
  --hidden-dim 4096 \
  --intermediate-dim 11008 \
  --rank 8 \
  --warmup 1 \
  --iters 2 \
  --output-json outputs/expact_v3_stage2_cpu_lifetime_small.json
```

Full active LF comparison:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v3_stage2_cpu_lifetime_b4s4096_nohook" \
SFT_ROOT=/workspace/AsymGEMM-SFT \
GPU_POOL=0 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES="none|true,gc-exp|false,none|false" \
ASYM_OFFLOAD_MODULES=all \
ASYM_EXPACT_CPU_POOL_MAX_BYTES=$((32 * 1024 * 1024 * 1024)) \
LORA_DROPOUT=0.00 \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=10 \
WARMUP_STEPS=5 \
PROFILE_MEMORY_ATTRIBUTION=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_SNAPSHOT=false \
PROFILE_LEVEL=op \
PLOT=false \
RUN_POST=false \
CONTINUE_ON_ERROR=false \
scripts/lf/profile_lora_lf.sh --gpus 0
```

Stage 2 passes only if:

- post-backward `cpu_owned_bytes == 0` for the activation-offload manager
- cached pinned CPU bytes are reported and stay under the configured cap
- active `b4_s4096` `none|true` peak HBM is flat/lower versus `126.312 GiB`
- active `b4_s4096` `none|true` latency does not materially regress versus
  `43.742 s`; if it regresses, compare pool cap settings before proceeding
- AsymGEMM/GEMM counts, fallback counts, and CPU Adam health remain valid
- any CPU-memory improvement is recorded separately from the HBM claim

## Stage 3: Remove Route-Expanded Scatter Saved Output

### Why This Stage Is Needed

This is the strongest current expert-side HBM candidate. In active `b4_s4096`,
the route-expanded scatter tensor is `[131072, 2048]` bf16, `512 MiB` per
layer, up to `24.0 GiB` across 48 layers if peak-live. Router no-grad means
scatter backward should only need `token_indices` and detached
`routing_weights` to reconstruct `grad_expert`, not the full `expert_output`.

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

### Validation Before Stage 4

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
OUTPUT_ROOT="$PWD/outputs/expact_v3_stage3_scatter_nosave_b4s4096" \
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
OUTPUT_ROOT="$PWD/outputs/expact_v3_stage3_scatter_nosave_b4s4096_nohook" \
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

Stage 3 passes only if:

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

## Stage 4: Mandatory Pause Gate After Scatter No-Save

### Why This Stage Is Needed

Stage 3 is the first new HBM target because router scatter currently uses
ordinary autograd around route-expanded tensors. After changing that lifetime,
stop and measure. This stage is a mandatory pause gate, not another
implementation stage. It prevents repeating v2 by requiring the active LF
workflow to prove what changed, name the new global peak owner and expert-block
peak owner, and get explicit approval before any S5+ work starts.

### Scope

- `scripts/lf/profile_lora_lf.sh`
  - run the canonical workflow comparison exactly, with the explicit dataset
    and sequence settings from the active baseline section
- `scripts/lf/run_lf_profiled_train.py`
  - only add missing summary fields if the profile JSON cannot report peak
    owner, expert-block peak, loss peak, or AsymGEMM/GEMM counts
- `asym_gemm/training/lf_trace.py`
  - only improve attribution labels if expert/loss/non-expert peak attribution
    is ambiguous
- `asym_gemm/training/qwen3_moe.py`
  - only add non-disturbing counters or NVTX ranges if Stage 0/1 attribution is
    insufficient

### Code Changes

1. Do not change expert math in this stage.
2. Re-run the active comparison after Stage 3:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v3_stage4_reprofile_b4s4096_nohook" \
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
PROFILE_MEMORY_SNAPSHOT=false \
PROFILE_LEVEL=op \
PLOT=false \
RUN_POST=false \
CONTINUE_ON_ERROR=false \
scripts/lf/profile_lora_lf.sh --gpus 0
```

3. If attribution is still unclear, run a single-step diagnostic only:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v3_stage4_reprofile_b4s4096_attr" \
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
WARMUP_STEPS=0 \
PROFILE_MEMORY_ATTRIBUTION=true \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_MEMORY_SNAPSHOT=true \
PROFILE_MEMORY_BREAKDOWN_INTERVAL=1 \
PROFILE_LEVEL=op \
PLOT=false \
RUN_POST=false \
CONTINUE_ON_ERROR=false \
scripts/lf/profile_lora_lf.sh --gpus 0
```

4. Record one decision proposal:
   - `next_target=none`: peak moved to non-expert/loss/attention/norm; stop
     expert-side HBM work and use later stages only for latency at flat HBM
   - `next_target=integrated_gate_up_backward`: `dgate_up_for_gate_up_base` is
     live at the expert or global peak
   - `next_target=down_base_forward_stage`: `act_for_down_base` is live at the
     expert or global peak
   - `next_target=lora_a_accumulator`: full LoRA-A accumulator scratch is live
     at the peak
   - `next_target=workspace_cleanup`: hidden full-width temporary tensors are
     live at the peak
5. Mandatory pause rule:
   - after recording the S4 metrics and `next_target` proposal, the agent stops
     implementation work and reports the result
   - no S5+ design or code work starts until the user explicitly approves the
     selected next target
   - if S4 proves the scatter fix achieved the intended memory result and no
     expert-side peak remains, stop the expert memory project here
   - if S4 shows no meaningful HBM reduction, do not continue into S5+ without
     explaining why the scatter hypothesis failed
6. Future branch rule, only after explicit approval:
   - if `next_target=integrated_gate_up_backward`, proceed to Stage 5
   - if `next_target=down_base_forward_stage`, do not reuse the reverted v2
     CPU-source base rewrite; write a separate evidence-gated Stage 5
     replacement that explains why the new peak evidence is different and how
     the kernel will avoid extra host traffic and latency blow-up
   - if `next_target=lora_a_accumulator`, skip Stage 5 and proceed to Stage 6
   - if `next_target=workspace_cleanup`, skip Stage 5-7 and proceed to Stage 8
   - if `next_target=none`, stop expert-side memory work and report the
     non-expert bottleneck

### Risks And Watch Items

- Saved-tensor hooks, memory snapshots, and full breakdown can perturb peak HBM.
  They are diagnostic only; the no-hook source-profile run remains the
  comparison result.
- If the peak owner is C++/CUDA-only with weak Python frame labels, improve the
  attribution labels before choosing a rewrite.
- If loss/cross-entropy owns the global peak, expert-side changes may still be
  useful only if an expert-block peak is separately reported and reduced.
- If `act_for_down_base` or `dgate_up_for_gate_up_base` is not peak-live, do not
  implement CPU-source base rewrites just because those tags look large locally.

### Validation Before Stage 5

Stage 4 passes only if:

- the active `b4_s4096` comparison is recorded for all three policies
- `none|true` peak allocated/reserved HBM, step/fwd/bwd timing, AsymGEMM/GEMM
  counts, fallback counts, and CPU Adam health are recorded
- global peak owner and expert-block peak owner are identified, or attribution
  gaps are listed as exact missing fields to implement
- exactly one `next_target` proposal is recorded
- the mandatory pause is honored: no S5+ implementation starts in the same
  agent run unless the user explicitly approves continuing past S4
- the proposed branch is written next to the results before any future
  implementation
- no v2-like CPU-source base or standalone LoRA-B rewrite is implemented in this
  stage

## Stage 5: Integrated Gate/Up Backward Redesign, Conditional Only

### Why This Stage Is Needed

This stage exists only if Stage 4 selects
`next_target=integrated_gate_up_backward`. The current backward creates a full
CUDA `dgate_up_for_gate_up_base` concat with
`ActivationOffloadManager.stage_concat_columns`, then reuses that same tensor
for gate/up LoRA-B and gate/up base dX. Therefore CPU-source LoRA-B alone is not
a memory fix: the large `[M,2I]` HBM stage still exists for base dX, and reading
the CPU gradients again can make latency worse while peak HBM stays flat.

The better design is integrated: avoid or shrink the full gate/up gradient stage
while computing both gate/up base dX and any LoRA-B work from the same CPU grad
tiles. This preserves the useful v2 lesson but does not repeat the failed v2
standalone helper path.

### Scope

- `asym_gemm/training/qwen3_moe.py`
  - `_ActivationOffloadQwen3ExpertFunction.backward`
- New wrapper only if Stage 4 selects this target:
  - `asym_gemm/training/exp_act_offload_gateup.py`
  - update `asym_gemm/training/__init__.py` only if public imports are needed
- Native only for the accepted integrated design:
  - `csrc/apis/exp_act_offload.hpp`
  - `csrc/exp_act_offload/exp_act_offload_kernels.cu`
  - `csrc/python_api.cpp` only if new bindings are added
  - `setup.py` only if a new `.cu` file is added
- Existing helper may be reused only as a subroutine:
  - `asym_gemm/training/exp_act_offload_lora.py::grouped_lora_b_backward_cpu_source`
- Stats:
  - `asym_gemm/training/frozen_linear.py::AsymExecutionStats`
- Tests:
  - new focused gate/up wrapper tests if a native wrapper is added
  - update `tests/training/test_lf_qwen3_asym_backend.py`

### Code Changes

1. Add a precondition in the implementation notes or code path:

```text
assert stage4_next_target == "integrated_gate_up_backward"
```

2. Preferred design: one integrated grouped CPU-source backward path:

```python
grad_packed, dS_gate, dS_up, grad_gate_lora_B, grad_up_lora_B = (
    grouped_gate_up_backward_cpu_source_integrated(
        grad_gate_cpu.tensor,
        grad_up_cpu.tensor,
        ctx.gate_low_rank_cpu.tensor,
        ctx.up_low_rank_cpu.tensor,
        layer.gate_up_base.host_weight.tensor,
        gate_lora_A,
        gate_lora_B,
        up_lora_A,
        up_lora_B,
        offsets,
        experts,
        need_grad_packed=need_grad_packed,
        scale=layer.lora_scale,
        metadata=lora_metadata,
        stats=layer.stats,
    )
)
```

Scheduling requirements:

- grouped launch or grouped kernel schedule, never a Python loop over experts
- load a CPU grad tile once, keep/reuse it in SMEM/registers for base dX and
  LoRA-B work before advancing
- reuse CPU tiles across inner loops the same way AsymGEMM left-CPU and
  right-CPU kernels are designed to do
- write final `grad_packed [M,H]`, `dS_gate [M,R]`, `dS_up [M,R]`, and LoRA-B
  gradients only; do not materialize full CUDA `[M,2I]`
- count `expact_gate_up_integrated_bwd_calls`,
  `expact_gate_up_full_concat_avoided_bytes`,
  `expact_lora_b_integrated_calls`, and CPU-source bytes by tag

3. Lower-risk fallback if the integrated kernel is too large for one step:

```text
stage grad_gate_cpu [M,I] only -> gate LoRA-B + gate contribution to base dX -> release
stage grad_up_cpu   [M,I] only -> up LoRA-B   + up contribution to base dX   -> release
accumulate final grad_packed [M,H]
```

The half-stage fallback is accepted only if active LF peak HBM drops
meaningfully versus the full `[M,2I]` concat and latency stays within the
acceptance rule. It is not a final design if it only changes local stage bytes.

4. Explicitly forbidden in this stage:
   - switching gate/up LoRA-B to CPU-source while leaving
     `dgate_up_for_gate_up_base` live for base dX
   - reintroducing the v2 rank-64 LoRA-B split as a standalone stage
   - adding per-expert Python loops or many small GEMMs
   - accepting a local helper win when the active LF peak is unchanged and
     latency is worse

5. Keep CPU Adam compatible:
   - returned LoRA gradients stay normal CUDA tensors with current shapes/dtypes
   - do not write directly to `AsymCPUAdamW` CPU masters
   - do not move trainable LoRA params to CPU

### Risks And Watch Items

- This path reads CPU gradients and CPU base weights. If NCU shows poor host
  load reuse, bad SMEM reuse, or host-memory stalls, tune the isolated kernel
  before accepting the LF result.
- Numerical tolerance may change because tiled reductions reorder accumulation.
  Tests must compare against the current torch/reference path.
- The half-stage fallback is pragmatic but weaker: it still stages `[M,I]` at a
  time. Reject it if the measured peak is unchanged.
- If Stage 4 selected `down_base_forward_stage` instead, do not run this stage;
  write a separate, evidence-gated design for `act_for_down_base`.

### Validation Before Stage 6

Static audit:

```bash
rg -n "dgate_up_for_gate_up_base|stage_concat_columns|expact_gate_up_integrated|expact_gate_up_half_stage" \
  asym_gemm/training/qwen3_moe.py \
  asym_gemm/training/activation_offload.py \
  asym_gemm/training/exp_act_offload_gateup.py \
  csrc/exp_act_offload || true
```

Correctness:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend
```

CPU Adam one-step checks:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_asym_cpu_adamw.py \
  tests/lf/test_asym_cpu_adamw_lf_integration.py
```

Isolated NCU after the wrapper exists:

```bash
bash -n scripts/testing/ncu_exp_act_offload_kernel_profile.sh
scripts/testing/ncu_exp_act_offload_kernel_profile.sh gateup_integrated
```

Full active comparison:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v3_stage5_gateup_integrated_b4s4096_nohook" \
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
PROFILE_MEMORY_SNAPSHOT=false \
PROFILE_LEVEL=op \
PLOT=false \
RUN_POST=false \
CONTINUE_ON_ERROR=false \
scripts/lf/profile_lora_lf.sh --gpus 0
```

Stage 5 passes only if:

- Stage 4 selected `integrated_gate_up_backward`
- full `dgate_up_for_gate_up_base` concat is absent, or the accepted half-stage
  fallback proves lower active LF peak HBM
- grouped launch behavior is preserved with no per-expert loops or small-GEMM
  decomposition
- AsymGEMM/GEMM counts, CPU-source bytes, avoided full-concat bytes, fallback
  counts, and CPU Adam health are recorded
- active LF `none|true` meaningfully lowers peak HBM without material latency
  blow-up, or improves latency with flat/lower HBM

## Stage 6: Conditional LoRA-A Accumulator Tiling

### Why This Stage Is Needed

LoRA-A gradient reduction can allocate full FP32 accumulator workspaces larger
than the final parameter gradients. Tiling keeps scratch bounded while still
returning real CUDA `.grad` tensors for CPU Adam. This can reduce backward peak
HBM only if those accumulators overlap the active peak. This stage is skipped
unless Stage 4, Stage 5, or a later attribution run proves the accumulator is a
peak owner.

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

1. Add an evidence precondition:

```text
stage4_next_target == "lora_a_accumulator"
or Stage 5 attribution reports LoRA-A accumulator scratch at the active peak
```

If this precondition is false, do not implement this stage.

2. Replace full-accumulator native LoRA-A grad internals:

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

3. Tiled contract:

```text
dS_cuda: [M,r]
X_cpu: pinned CPU BF16 [M,K]
grad_A_cuda: [E,r,K]
offsets/experts: grouped route metadata
```

4. Pair policy:
   - one paired gate/up kernel consumes `dS_gate`, `dS_up`, and `X_cpu`
   - the same `X_cpu` tile is reused for both gate and up reductions
   - output two final gradient tiles
   - increment paired LoRA-A grad counters separately from single calls

5. Reduction policy:
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

### Validation Before Stage 7

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
  --output-json outputs/expact_v3_stage6_small.json
```

Stage 6 passes only if:

- the evidence precondition was true; otherwise Stage 6 is explicitly skipped
- no full FP32 LoRA-A accumulator remains in the expact native path
- grouped kernel counts remain bounded and do not scale with expert count
- active LF peak HBM drops meaningfully, or latency improves with flat/lower HBM

## Stage 7: Deferred Native Paired Gate/Up Tile Reuse

### Why This Stage Is Needed

This is a deferred latency stage, not a memory-primary stage. Gate and up LoRA-A
both consume the same offloaded input activation, so a native paired schedule
could load one CPU tile and reuse it for both paths. The v2 paired attempt
reduced counters but did not improve the accepted LF workflow, so do not
re-land the concatenation-based or call-count-only version. Try this only after
the active memory target is handled and profiles show repeated CPU-source reads
are a real latency bottleneck at flat/lower HBM.

### Scope

- `asym_gemm/training/exp_act_offload_lora.py`
  - native-backed `grouped_lora_a_pair_forward_cpu_left` only if it performs
    one physical paired CPU-source schedule
  - paired backward wrappers only if Stage 5/6 made them necessary and
    peak-safe
- `asym_gemm/training/qwen3_moe.py`
  - `_ActivationOffloadQwen3ExpertFunction.forward`
  - `_ActivationOffloadQwen3ExpertFunction.backward`
- `asym_gemm/training/lora.py`
  - `grouped_expert_lora_pair`
  - metadata reuse and `torch.cat` accounting only for visibility, not as the
    accepted paired design
- Native:
  - `csrc/apis/exp_act_offload.hpp`
  - `csrc/exp_act_offload/exp_act_offload_kernels.cu`
- Tests:
  - update `tests/training/test_cpu_left_lora.py`
  - new `tests/training/test_exp_act_offload_gate_up_paired.py`

### Code Changes

1. Evidence precondition:

```text
active LF memory target has passed
and profiling shows repeated gate/up CPU-source reads are a latency bottleneck
and the proposed change keeps peak HBM flat/lower
```

If this precondition is false, skip Stage 7.

2. Native paired CPU-left schedule only:
   - load one `X_cpu` tile
   - compute gate and up low-rank tiles before advancing the source tile
   - one grouped paired call, not two independent source passes
   - reuse the CPU tile in SMEM across both LoRA-A weight tiles
   - do not implement the v2-style Python-side weight concatenation as the
     accepted path

3. Backward pairing is optional and evidence-gated:
   - use integrated gate/up work from Stage 5 only if that stage passed
   - use tiled LoRA-A grad from Stage 6 only if that stage passed
   - do not recreate `stage_concat_columns`
   - do not add another CPU-source pass just to improve a local counter

4. Track allocations:
   - count paired calls distinctly from two singles
   - record CPU source bytes loaded for paired gate/up
   - tag any `torch.cat` in `grouped_expert_lora_pair`; if it appears at the
     active peak, the native path must remove it or the stage fails

### Risks And Watch Items

- Pairing is mostly a timing and bandwidth fix. Do not make correctness harder
  unless profiles show repeated CPU source reads are material.
- The v2 concatenation/call-count approach is not an accepted fallback. It can
  remain as code only if it is already present and profile-visible, not as the
  planned optimization.
- `grouped_expert_lora_pair` currently uses concatenation for CUDA low-rank
  inputs and weights. That may be acceptable for low-rank tensors, but it must
  be counted and revisited if it moves peak.

### Validation Before Stage 8

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

Stage 7 passes only if:

- the evidence precondition was true; otherwise Stage 7 is explicitly skipped
- stats show one physical native paired CPU-left LoRA-A forward schedule for
  gate/up, not two single calls hidden behind one wrapper
- no per-expert GEMM launch pattern appears in source/NSYS stats
- paired CPU source bytes are lower than two independent passes
- the active LF `none|true` run improves latency with flat/lower HBM
- do not keep paired scheduling if it only reduces call/source-byte counters
  while full-workflow memory stays flat and latency regresses
- Qwen3 expact remains numerically matched to the torch backend

## Stage 8: Hidden Materialization And Workspace Cleanup

### Why This Stage Is Needed

After the obvious expert tensors are addressed, remaining peak HBM may come
from untagged convenience materializations: padding buffers, concat buffers,
stage caches, allocator leftovers, or scratch shaped like full activations.
This stage makes those buffers profile-visible and bounded, so accidental
full-width HBM materialization can be removed.

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

### Validation Before Stage 9

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
OUTPUT_ROOT="$PWD/outputs/expact_v3_stage8_cleanup_b4s4096_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=0 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES="none|true,gc-exp|false,none|false" \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=1 \
WARMUP_STEPS=0 \
PROFILE_MEMORY_ATTRIBUTION=true \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_MEMORY_SNAPSHOT=true \
PROFILE_MEMORY_BREAKDOWN_MODULES=experts,lora,loss \
PLOT=false \
RUN_POST=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus 0
```

Stage 8 passes only if:

- no untagged full-width materialization remains in expact hot regions
- `max_stage_bytes_live` is bounded and does not include full activation stages
- source profile distinguishes expert-block peak from loss/cross-entropy peak
- grouped call counts remain bounded

## Stage 9: Final Fair Comparison And CPU Adam Acceptance

### Why This Stage Is Needed

This proves the claim under the fair LF workflow. Microbenchmarks, standalone
kernels, and local counters are supporting evidence only. The final run must
compare `none|true`, `gc-exp|false`, and `none|false` with the same model,
batch, sequence length, LoRA config, precision, optimizer, router mode, and
profiler settings, while verifying CPU Adam, grouped kernels, and fallback
counts.

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
- If `4x8192` still OOMs after Stage 8, record whether the failure is inside
  expert blocks or later loss/cross-entropy. The claim is not satisfied by a
  loss-side OOM explanation alone, but the next fix depends on where the peak
  moved.
- If async is introduced, compare sync vs async with identical kernels. Async
  must not increase peak HBM through excessive prefetch overlap.

### Final Validation

Canonical active workflow comparison, `b4_s4096`:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v3_final_b4s4096_$(date -u +%Y%m%dT%H%M%SZ)" \
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
PLOT=false \
RUN_POST=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus 0
```

Optional stress workflow comparison, `b2_s8192`:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v3_final_b2s8192_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=0 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES="none|true,gc-exp|false,none|false" \
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

Optional stress workflow comparison, `b4_s8192`:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v3_final_b4s8192_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=0 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES="none|true,gc-exp|false,none|false" \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=8192 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PLOT=false \
RUN_POST=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus 0
```

Optional global recompute lower bound, active `b4_s4096`:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v3_final_global_recompute_b4s4096_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=0 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|recomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES="none|false" \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PLOT=false \
RUN_POST=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus 0
```

Optional NSYS kernel-count run after the canonical source profile passes:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_v3_final_nsys_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=0 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=nsys,source \
ASYMM_EXP_ACT_POLICIES="none|true,gc-exp|false,none|false" \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PLOT=false \
RUN_POST=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus 0
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
