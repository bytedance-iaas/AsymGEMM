# Expert Activation Offload V4: Remaining Safe Math-Lifetime Targets

This document starts from the accepted V3 post-S4 state. V2 is historical and
must not be reintroduced. V3 keeps the full measurement record and the
post-S4/S8 attribution verdict. V4 is narrower: it lists only the remaining
math-lifetime/kernel candidates that have not already been proven bad.

Core goal: reduce peak HBM for the `none|true` expert activation-offload path,
or improve its latency while keeping peak HBM flat or lower. The mechanism must
remain expert activation offload plus grouped CPU-source AsymGEMM/native kernels.
Do not use unrelated fused stacks, optimizer changes, different recompute modes,
per-expert loops, or small-GEMM decomposition to make the comparison look good.

Accepted comparison workflow:

```text
scripts/lf/profile_lora_lf.sh --gpus 0
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1"
BACKEND_SPECS="asym_cpuadamwds|norecomp"
ASYMM_EXP_ACT_POLICIES="none|true,gc-exp|false,none|false"
SEQ_LENS=4096
PER_DEVICE_TRAIN_BATCH_SIZE=4
LORA_DROPOUT=0.00
PROFILERS=source
PROFILE_MEMORY_ATTRIBUTION=false
PROFILE_MEMORY_BREAKDOWN=false
PROFILE_MEMORY_SNAPSHOT=false
WARMUP_STEPS=5
MAX_STEPS=10
RUN_POST=false
```

Accepted post-S4 baseline:

| policy | implementation | peak allocated HBM | peak reserved HBM | avg step | avg forward | avg backward |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `none|true` | activation offload target | `102.312 GiB` | `107.469 GiB` | `45.173 s` | `10.036 s` | `35.137 s` |
| `gc-exp|false` | expert checkpoint baseline | `126.312 GiB` | `131.414 GiB` | `3.953 s` | `1.570 s` | `2.383 s` |
| `none|false` | no expact/no expert recompute | `170.503 GiB` | `183.055 GiB` | `3.129 s` | `1.562 s` | `1.567 s` |

Current accepted call counts for `none|true`: `asym_forward_calls=5055`,
`asym_dx_calls=4290`, `torch_forward_calls=0`, `torch_dx_calls=0`,
`reference_fallback_count=0`.

The 1M-entry snapshot replay matched the source peak and showed that exact
routed-expert live blocks at the global peak are small. The earlier
`routed_experts:temporary_workspace:inferred_peak_workspace=5.795 GiB` row is
an inferred residual bucket, not exact expert ownership. Do not chase that row
as if it were a real expert tensor.

## Acceptance Rules

- Same peak HBM with worse latency is a regression.
- Tiny HBM savings with a large latency increase are rejected.
- Latency-only changes are accepted only if `none|true` peak HBM stays flat or
  lower and AsymGEMM/GEMM counts do not blow up.
- Memory-changing work must compare against the accepted post-S4 baseline above.
- Unit tests and microbenchmarks are not acceptance. They only allow an LF run.
- Every accepted run must record peak allocated/reserved HBM, forward/backward
  time, step time, AsymGEMM/GEMM counts, fallback count, and activation-offload
  stage/cpu-live counters.

## Do Not Retry

These are already ruled out as standalone changes:

- CPU-source LoRA-B backward by itself while full `grad_gate_up` still exists.
  V2 showed this can keep peak HBM unchanged and worsen latency.
- Re-land the V2 paired LoRA-A wrapper/call-count change. Only a real native
  paired kernel with CPU-tile reuse is eligible.
- Offload full `dY` to CPU only to compute LoRA-B grads. This adds a wide D2H
  path and has no current peak evidence.
- Chase the inferred `5.795 GiB` expert-workspace residual without exact
  snapshot proof.
- Unbounded async prefetch/staging. Async may improve latency, but only with a
  hard cap on staged HBM.
- Add Python expert loops or split the expert path into per-expert small GEMMs.

CPU handle cleanup is not a V4 HBM optimization. It is already mostly working:
source profile reports `total_cpu_live_bytes=0` after cleanup. Keep debug-only
lifetime checks if useful, but run acceptance profiles with checks disabled.

## AsymGEMM-Specific Scope

The main body only tracks novel AsymGEMM-specific work that can support the
memory claim:

```text
Forward:
  offload routed expert activations/state to CPU at math last-use points

Backward:
  consume the CPU-resident activations through grouped CPU-source
  AsymGEMM/native kernels, avoiding full HBM fetchback whenever that tensor is
  a meaningful peak-HBM owner

Comparison:
  same model, optimizer, LoRA config, routing, sequence/batch, profiler, and
  global recompute setting; compare `none|true` against `gc-exp|false`
```

The real AsymGEMM-specific implementation work is CPU-resident expert-state
lifetime plus grouped CPU-source consumption. Generic LoRA fusion, generic
saved-tensor CPU offload, generic activation checkpointing, and HBM-resident
LoRA epilogue fusion are moved to the overlap notes at the end of this
document. They can inform implementation, but they are not the central claim.

## V4-S0: Evidence Gate Before Each Candidate

Purpose: verify the candidate is actually live near the active peak before
changing code. This stage does not reduce memory by itself; it prevents another
V2-style change that only improves a local tag while the global peak stays the
same.

Files/functions:

- `scripts/lf/run_lf_profiled_train.py`
  - source-profile fields for activation-offload counters
  - memory snapshot capture
- `scripts/testing/analyze_cuda_memory_snapshot.py`
  - torch-free snapshot replay and peak live-set attribution
- `asym_gemm/profiling/lf_trace.py`
  - breakdown labeling; inferred residual rows must be clearly labeled

Diagnostic command for one-step peak ownership:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
OUTPUT_ROOT="$PWD/outputs/expact_v4_s0_snapshot_b4s4096_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=0 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES="none|true" \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
WARMUP_STEPS=0 \
MAX_STEPS=1 \
RUN_POST=false \
PLOT=false \
PROFILE_MEMORY_ATTRIBUTION=true \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_MEMORY_BREAKDOWN_INTERVAL=1 \
PROFILE_MEMORY_SNAPSHOT=true \
ASYM_GEMM_LF_PROFILE_MEMORY_SNAPSHOT_MAX_ENTRIES=1000000 \
scripts/lf/profile_lora_lf.sh --gpus 0
```

Then run:

```bash
PYTHONPATH="$PWD" .venv/bin/python scripts/testing/analyze_cuda_memory_snapshot.py \
  <path-to-memory_snapshot.pickle> --top 40
```

Proceed only if the snapshot or exact counters show one of the V4 target tensors
below is live at the peak or the change is strictly latency-only with flat/lower
HBM expected.

## V4-S1: Conditional Gate/Up Backward Stage Lifetime Reduction

Why this can help: current backward stages full
`grad_gate_up = stage_concat_columns(dgate_cpu, dup_cpu)` before gate/up LoRA
backward and keeps it until after base `dX`. That full `[M,2I]` HBM stage can
overlap low-rank stages, `dS_*`, LoRA grad work, and LoRA `dX` temporaries.
This is the main remaining math-lifetime mismatch.

This is conditional because V3 diagnostics showed
`dgate_up_for_gate_up_base=402653184` bytes (`384 MiB`) per layer and not a
confirmed global-peak owner. Do not implement this unless V4-S0 or a later
AsymGEMM-specific profile shows the full stage is peak-live, or the
implementation also improves latency with flat/lower HBM.

Files/functions:

- `asym_gemm/training/qwen3_moe.py`
  - `_ActivationOffloadQwen3ExpertFunction.backward`
  - regions: `gate_up_stage`, `gate_lora`, `up_lora`, `gate_up_base_dx`
- `asym_gemm/training/activation_offload.py`
  - `ActivationOffloadManager.stage`
  - `ActivationOffloadManager.stage_concat_columns`
- native base-dX kernel only if using true CPU-source or bounded tiled staging

Allowed implementation directions:

1. Native integrated CPU-source gate/up base `dX`:
   - consume `dgate_cpu` and `dup_cpu` from pinned CPU
   - reuse CPU tiles in SMEM across the needed K-loop work
   - write one `grad_packed [M,H]`
   - avoid full `[M,2I]` HBM concat

2. Bounded half-stage fallback, only if it beats baseline:
   - stage `dgate_cpu [M,I]`, compute/add gate base `dX`, release
   - stage `dup_cpu [M,I]`, compute/add up base `dX`, release
   - use grouped kernels, not per-expert loops
   - record the extra grouped GEMM count explicitly

Rejected form:

```python
# Not enough; this repeats the V2 failure mode if full grad_gate_up remains.
dS_gate, dB_gate = grouped_lora_b_backward_cpu_source(dgate_cpu, ...)
dS_up, dB_up = grouped_lora_b_backward_cpu_source(dup_cpu, ...)
grad_packed = _grouped_base_dx(..., grad_gate_up, ...)
```

Validation:

- First run a one-policy diagnostic with memory snapshot and confirm the full
  stage was removed or bounded.
- Then run the full LF acceptance sweep with `OUTPUT_ROOT` renamed to
  `expact_v4_s1_gate_up_lifetime_b4s4096_<timestamp>`.
- Accept only if peak HBM drops meaningfully below the prior accepted baseline,
  or latency improves with flat/lower HBM and call-count growth is justified.

## V4-S2: True Native Paired Gate/Up CPU-Tile Reuse

Why this can help: latency is the main weakness of `none|true`. The math note
assumes CPU tiles loaded from pinned CPU are reused while resident in SMEM. The
current Python pair wrapper for gate/up LoRA-A forward still calls two logical
CPU-left paths and does not prove physical tile reuse. A true native paired
kernel can reduce CPU-source traffic and latency without raising HBM.

Why this stays in the main body: the novel part is not paired LoRA fusion by
itself. The novel part is reusing a CPU-resident activation tile inside the
AsymGEMM/native CPU-source schedule. If the implementation becomes ordinary
HBM-resident LoRAFusion-style pairing, it belongs in the LoRAFusion appendix
and must not be claimed as core V4 work.

This is latency-first, not memory-first. It is eligible only after V4-S1 is
accepted, rejected, or explicitly skipped by evidence.

Files/functions:

- `asym_gemm/training/exp_act_offload_lora.py`
  - `grouped_lora_a_pair_forward_cpu_left`
  - possible paired backward helpers only if needed
- `asym_gemm/training/qwen3_moe.py`
  - forward gate/up LoRA-A call site
- native kernel/binding for a real paired CPU-left schedule

Rules:

- One native grouped schedule must load an `X_cpu` tile once and reuse it for
  both gate and up LoRA-A tiles.
- No Python expert loop.
- No increase in full HBM stages.
- Record physical native paired call counts separately from wrapper counts.
- Use NCU on the standalone kernel to verify CPU-load behavior before LF
  acceptance.

Standalone kernel validation should use NCU on the paired kernel only, not a
full training step, so memory traffic and tile reuse are visible:

```bash
ncu --set full --target-processes all -o <report_name> \
  <standalone paired-kernel microbenchmark command>
```

Full acceptance is the same LF sweep, with `OUTPUT_ROOT` renamed to
`expact_v4_s2_native_pair_b4s4096_<timestamp>`.

Accept only if `none|true` latency improves and peak HBM stays flat/lower.

## Stop Conditions

Stop V4 work when one of these is true:

- V4-S0 shows the remaining candidate tensors are not peak-live and V4-S2 has no
  proven latency path.
- A candidate keeps memory the same and worsens latency.
- A candidate saves only tiny HBM while materially worsening latency.
- The implementation needs per-expert loops, small GEMM decomposition, or a
  comparison change outside the accepted LF workflow.

If all V4 stages are skipped or rejected, the honest conclusion is that the
remaining global peak is outside exact routed expert activation-offload tensors.
Further HBM reduction would need a fair change applied to both expact and
checkpoint baselines, such as non-expert activation/loss/norm memory work.

## Appendix: Overlap With LoRAFusion

LoRAFusion, local path `/workspace/AsymGEMM-SFT/third_party/lorafusion`, is
prior art for fused LoRA linear kernels, not for this memory claim. Its single
LoRA path computes and saves `S = dropout(X) A.T`, then fuses:

```text
forward:  Y = X W.T + alpha * S B.T
backward: dB, dS = fused(dY, S, B)
          dX     = fused(dY W + dS A)
```

Overlapping ideas:

- low-rank `S` as the saved LoRA state
- add/beta epilogues that avoid materializing `down_delta` or LoRA `dX`
- fused `dY W + dS A` structure as inspiration for appendix candidate LF-2
- NCU/kernel traffic analysis and per-kernel tuning discipline

Overlapping candidate LF-1, not a core AsymGEMM stage: remove
`down_delta [M,H]` materialization.

- Current expact forward materializes `down_delta`, keeps it alive through down
  base, then does `output.add_(down_delta)`.
- The LoRAFusion-like version is `Y_down += LoRA_B(S_down)` with an add/beta
  epilogue.
- Files would be `asym_gemm/training/qwen3_moe.py`,
  `asym_gemm/training/lora.py`, and a native grouped LoRA add-into-output
  binding if needed.
- This must not be implemented as `tmp = grouped_expert_lora(...);
  output.add_(tmp)`, because that is the current lifetime problem.
- This is only acceptable if it lowers peak HBM or improves latency with flat
  HBM in the full LF workflow. It is not a new AsymGEMM claim by itself.

Overlapping candidate LF-2, not a core AsymGEMM stage: add LoRA `dX` directly
into `grad_packed`.

- Current expact backward can materialize `grad_gate_lora_x [M,H]` and
  `grad_up_lora_x [M,H]`, then add them to `grad_packed`.
- The LoRAFusion-like version is `dX += dS_gate @ A_gate` and
  `dX += dS_up @ A_up` through grouped add-into-output epilogues.
- Files would be `asym_gemm/training/qwen3_moe.py`,
  `asym_gemm/training/lora.py`, and a native grouped LoRA add-into-output
  binding if needed.
- Risk: computing `grad_packed` earlier can make it overlap with LoRA backward
  work. If peak HBM does not improve, revert.
- This is only acceptable if it lowers peak HBM or improves latency with flat
  HBM in the full LF workflow. It is not a new AsymGEMM claim by itself.

How to use this overlap:

- It is valid prior art and implementation inspiration.
- Do not claim normal HBM-resident LoRAFusion-style fusion as our contribution.
- If a V4 change only fuses HBM LoRA operands, it is latency/local-memory
  engineering, not the central AsymGEMM activation-offload claim.
- The claim becomes AsymGEMM-specific only when the fused schedule consumes
  CPU-resident expert activations/low-rank state through grouped CPU-source
  kernels or preserves the fair expact-vs-recompute memory comparison.

## Appendix: Overlap With Megatron-LM

Megatron Core fine-grained activation offloading is prior art for module-level
activation offload. The local implementation is in
`/workspace/AsymGEMM-SFT/third_party/megatron-lm/megatron/core/pipeline_parallel/fine_grained_activation_offload.py`.
It uses saved-tensor hooks, D2H/H2D streams, group start/commit markers,
`forced_released_tensors`, inflight caps, and module choices such as
`expert_fc1` and `moe_act`.

Overlapping ideas:

- explicit group start/commit lifetime boundaries
- bounded async offload/reload queues
- forced release after offload when PyTorch GC is not enough
- offload summaries by module/group
- skipping offload for the last immediately-needed group to avoid blocking

Key difference:

- Megatron reloads saved CPU tensors back into full GPU tensors before backward
  consumers use them.
- Our core path must consume CPU-resident expert activations directly through
  grouped CPU-source AsymGEMM/native kernels where possible.
- If we only implement Megatron-style generic offload/reload, we are recreating
  prior art and should not claim a new AsymGEMM memory mechanism.

## Appendix: Generic Prior Art

PyTorch `saved_tensors_hooks` / `save_on_cpu` and DeepSpeed activation
checkpointing/CPU checkpointing are generic activation-memory mechanisms. They
are useful references for hook semantics, lifetime, and correctness, but they
do not replace the AsymGEMM claim. A fair result cannot be claimed from merely
wrapping the expert block in generic `save_on_cpu` or DeepSpeed checkpointing.

If an optimization comes from LoRAFusion/Megatron/PyTorch/DeepSpeed, record it
as borrowed prior art and only use it if both the target and baseline get the
same unrelated improvement, or if it is strictly internal latency work that
keeps the accepted memory comparison unchanged.
