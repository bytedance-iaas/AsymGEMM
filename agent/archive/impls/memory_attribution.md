# LF Memory Attribution Implementation Plan

## Goal

Replace the current LF `memory.md` "Fine-Grained Memory Attribution" with a
defensible memory breakdown that explains the observed peak HBM by component.

The output should answer:

- Whose weights are live?
- Whose gradients are live?
- Whose optimizer state is live?
- Whose activations are live?
- What remaining memory is framework/temp/workspace/allocator memory?

The stack should close to peak HBM:

```text
peak_hbm =
  gpu_weights
+ gradients
+ optimizer_state
+ attributed_activations
+ temp_workspace_framework
+ allocator_reserved_unallocated
```

Rows may be approximate, but every row must state its attribution method. Avoid
an unexplained "unknown residual" in user-facing summaries. If memory cannot be
semantically attributed exactly, assign it to a named approximate bucket such as
`framework_temp_workspace` or `allocator_reserved_unallocated`.

## Current Problem

Current LF memory reporting is split across:

- `asym_gemm/profiling/lf_trace.py`
- `scripts/lf/run_lf_profiled_train.py`
- `scripts/lora/postprocess_nsys_lora.py`
- `scripts/lf/postprocess_lf_profile_artifacts.py`

For LF runs, `memory.md` currently shows:

- coarse CUDA allocator stage snapshots
- tensor-size accounting for model parameters and CPU/pinned expert weights
- optional saved-tensor hook rows if `PROFILE_MEMORY_ATTRIBUTION=true`

This is not enough for a memory attribution figure. For the inspected run, peak
HBM was about `66.56 GiB`, while GPU tensor-size rows accounted for only about
`9.48 GiB`. The remaining memory was mostly activations, temporaries, workspaces,
and allocator effects, but the report did not split it by owner.

The section title "Fine-Grained Memory Attribution" is therefore misleading for
normal LF runs.

## Reference Method

Use the Gemma4 memory profiler style from:

- `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/gemma4-finetune/src/train/memory_breakdown.py`
- `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/gemma4-finetune/scripts/plots/plot_memory_breakdown.py`

That method combines:

- phase snapshots
- exact persistent tensor accounting
- selected module forward pre/post hook activation deltas
- residual-to-peak closure
- plot bands that sum to the observed CUDA peak

We should port the method conceptually, not copy the Gemma code directly. The
Gemma code is specialized for PLE/FSDP and expects `memory_breakdown_rank*.jsonl`.
AsymGEMM needs Qwen3/LF/AsymGEMM-specific component names and output schema.

## Files To Change

### 1. `asym_gemm/profiling/lf_trace.py`

Primary collection changes.

Add a new memory breakdown collector beside the existing `LFTraceHandle`:

```python
class LFMemoryBreakdownProfiler:
    ...
```

Responsibilities:

- install selected module forward hooks
- record phase snapshots
- collect exact tensor-size accounting
- collect activation deltas by semantic owner
- emit per-step rows for `memory_breakdown.jsonl`

Keep the existing `SavedTensorTracker`, but make it optional and secondary. It is
useful for saved-activation validation, not for the main peak-memory stack.

New pieces:

- `MemoryBreakdownConfig`
- `MemoryBreakdownRecord`
- `_component_from_param_name(name, param)`
- `_semantic_module_name(name, module)`
- `_install_activation_delta_hooks(model)`
- `_collect_persistent_tensor_bytes(model, optimizer)`
- `_phase_record(phase, step, ...)`

Suggested semantic component names:

```text
embedding
attention
mlp_dense
routed_experts
router
lora_attention
lora_mlp
lora_experts
norms
lm_head
loss_logits
optimizer
framework_temp_workspace
allocator_reserved_unallocated
cpu_host_expert_weight
cpu_pinned_expert_weight
other
```

Activation hook selection should start coarse:

```text
self_attn
mlp
mlp.experts
experts
gate/router
lora_A/lora_B wrappers if visible
embed_tokens
lm_head
norm
```

Avoid hooking every small op by default. Coarse hooks reduce overhead and make
the figure readable.

### 2. `scripts/lf/run_lf_profiled_train.py`

Wire the profiler into the real LF training loop.

Current source recorder already wraps training through `run_lf_profiled_train.py`.
Extend it to:

- construct `LFMemoryBreakdownProfiler` from env/config
- call `record("step_begin", step, ...)`
- call `record("after_forward", step, ...)`
- call `record("after_backward", step, ...)`
- call `record("before_optimizer_step", step, ...)`
- call `record("after_optimizer_step", step, ...)`
- write:
  - `memory_breakdown.jsonl`
  - `memory_breakdown_summary.json`
  - memory schema/version metadata in `source_profile.json`

The existing trainer patching is currently stage-based. If phase boundaries are
not exposed cleanly through the LF source wrapper, add a small monkey patch to
the Trainer training step, similar to the Gemma `GemmaSFTTrainer` path.

### 3. `scripts/lf/run_lf_lora_sft.sh`

Add user-facing controls and pass them to the source launcher.

Suggested env/options:

```text
PROFILE_MEMORY_BREAKDOWN=0|1
PROFILE_MEMORY_BREAKDOWN_INTERVAL=N
PROFILE_MEMORY_BREAKDOWN_STEPS=comma-list
PROFILE_MEMORY_BREAKDOWN_MODULES=attention,mlp,experts,lora,embedding,loss
PROFILE_MEMORY_BREAKDOWN_OUTPUT=memory_breakdown
```

Keep `PROFILE_MEMORY_ATTRIBUTION` as the saved-tensor-hook option. Do not reuse
that name for the new profiler because the overhead and semantics differ. The
shell wrappers should accept `PROFILE_MEMORY_ATTRIBUTION=auto|true|false` and
resolve `auto` before launching Python:

```text
source -> PROFILE_MEMORY_ATTRIBUTION=true
nsys   -> PROFILE_MEMORY_ATTRIBUTION=false
```

That keeps nsys timing clean by default while making a source pass collect the
extra memory rows unless the user explicitly overrides it to `false`.

Separate the profiler roles:

```text
nsys   -> existing timing/perf plots
source -> detailed memory-breakdown artifacts and memory-attribution plots
```

For the existing generic plotter, the sweep wrappers should prefer nsys rows
whenever `nsys` is among the requested profilers:

```text
--profilers nsys        -> generic plots use nsys
--profilers source,nsys -> generic plots use nsys only
```

Source rows from memory-attribution runs must not enter
`plot_activation_recompute_sweep.py`. In the mixed-profiler workflow, source
should not emit the generic timing/perf plot set either; it should only feed the
dedicated memory-breakdown artifacts/plots.

If a user runs `--profilers source` alone with `PROFILE_MEMORY_BREAKDOWN=1`, the
driver should skip the generic timing/perf plotter by default and run the
dedicated memory plotter instead. A lightweight source timing run with
`PROFILE_MEMORY_BREAKDOWN=0` may still use the generic plotter if explicitly
requested, but those plots are not timing truth.

### 4. `scripts/lf/profile_lora_lf.sh`

Add sweep-driver pass-through:

```text
--profile-memory-breakdown true|false
--profile-memory-breakdown-interval N
--profile-memory-breakdown-steps LIST
--profile-memory-breakdown-modules LIST
--plot-memory-breakdown true|false
--memory-breakdown-plot-output-dir DIR
--memory-breakdown-plot-y-scale shared|per-plot|global
```

Pass those into `run_lf_lora_sft.sh`.

Important behavior:

- `--profilers nsys --profile-memory-breakdown false` remains the clean timing
  mode.
- `--profilers source --profile-memory-breakdown true` is the memory attribution
  mode; it should generate source memory artifacts plus memory-attribution
  plots, not generic timing plots.
- `--profilers source,nsys --profile-memory-breakdown true` is the recommended
  full workflow: nsys feeds generic timing/perf plots, source feeds only
  memory-attribution artifacts/plots.
- If a user runs `nsys` with memory breakdown enabled, allow it but mark timing
  as hook-skewed.

Memory plotting behavior:

- Source run dirs should generate their own Gemma4-style memory plots.
- Config-level memory combined plots should scan source run dirs only.
- Outer memory combined plots should scan all source memory summaries under the
  precision/output root.
- Existing timing `combined/` remains nsys-backed and must not consume source
  memory rows.

### 5. `scripts/lf/run_lf_lora_sft_kt.sh` and `scripts/lf/profile_lora_lf_kt.sh`

Mirror the same options for KT runs if KT memory plots are needed. Otherwise,
explicitly reject `PROFILE_MEMORY_BREAKDOWN=1` for unsupported backends with a
clear message.

### 6. `scripts/lora/postprocess_nsys_lora.py`

Replace or demote the current misleading memory attribution renderer.

Current target function:

```python
_memory_attribution_markdown(...)
```

Needed changes:

- Rename current table heading to something honest, for example:
  `Persistent Tensor Accounting`
- Add a new renderer:
  `_memory_breakdown_markdown(source_profile, memory_breakdown_summary)`
- Read `memory_breakdown_summary.json` or embedded source-profile breakdown.
- Render a closure table:

```text
Peak HBM Breakdown
| Group | Component | Kind | bytes | MiB | % peak | Method |
| weights | attention | exact tensor size | ... |
| gradients | lora | exact tensor size after backward | ... |
| optimizer | lora | exact optimizer state tensor size | ... |
| activations | routed_experts | measured forward delta | ... |
| temp/workspace | attention | prorated peak closure | ... |
| allocator | reserved_unallocated | allocator snapshot | ... |
```

For user-facing output, do not leave a row named only `unattributed`. Use named
approximate buckets with `Method = inferred residual`.

### 7. `scripts/lf/postprocess_lf_profile_artifacts.py`

Update source-only memory markdown generation to use the same breakdown summary.

This is needed because `source` profiler runs may never go through the Nsight
postprocessor path.

### 8. New plotting script

Add one of:

- `scripts/plotting/plot_lf_memory_breakdown.py`
- or extend `scripts/plotting/plot_activation_recompute_sweep.py`

Prefer a new dedicated script:

```text
scripts/plotting/plot_lf_memory_breakdown.py
```

This is the Gemma4-style plotting path for source memory runs. It should not
reuse `plot_activation_recompute_sweep.py`, because that script is for
timing/perf comparisons and should stay nsys-backed when nsys exists.

Input:

- one run directory containing `memory_breakdown.jsonl`
- optionally many run directories for comparison
- source profiler result directories only, for example:
  `.../asym__source__norecomp__poltok-le1024/s4096/`
- never nsys result directories unless explicitly running a diagnostic memory
  experiment

Output:

- `memory_breakdown_stacked.png`
- `memory_breakdown_stacked.pdf`
- `memory_over_steps_stacked.png`
- `memory_peak_stack.png`
- `memory_breakdown.csv`
- optional combined memory-attribution output under a separate directory such as
  `memory_combined/`, not the existing timing `combined/`

Plot output hierarchy:

```text
<source_seq_root>/memory_plots/
  memory_over_steps_stacked.png
  memory_peak_stack.png
  memory_breakdown.csv

<config_root>/memory_combined/
  combined_memory_over_steps_stacked.png
  combined_memory_peak_stack.png
  combined_memory_breakdown.csv

<precision_root>/memory_combined/
  combined_memory_over_steps_stacked.png
  combined_memory_peak_stack.png
  combined_memory_breakdown.csv

<precision_root>/memory_combined/<workload_base>/
  combined_memory_over_steps_stacked.png
  combined_memory_peak_stack.png
  combined_memory_breakdown.csv
```

This mirrors the existing timing plot hierarchy, but it is a separate source
memory pipeline.

Plot bands:

```text
GPU weights
GPU gradients
GPU optimizer state
Attention activations
MLP activations
Routed expert activations
LoRA activations
Loss/logits/input activations
Temp/workspace/framework
Allocator reserved-unallocated
```

CPU/pinned expert weights should be separate bars or a separate panel, not added
to GPU HBM peak.

Y-axis policy:

- For per-run memory-over-steps plots, use the run peak HBM as the default y max.
- For config-level combined memory plots, use one shared absolute GiB y-axis
  limit across all included source runs so the stacks are visually comparable.
- For outer combined memory plots, use one shared absolute GiB y-axis limit
  across all included source runs, or group by model/sequence if a single global
  axis would make smaller runs unreadable.
- The default should keep the y-axis in absolute GiB. Optionally add a percent
  of peak view, but do not replace the absolute-memory view.

Compared with the Gemma4 plot, the LF plot should expose more detail:

- separate frozen weights, LoRA weights, gradients, and optimizer state
- separate attention, dense MLP, routed expert, LoRA, and loss/logits activation
  rows when measurable
- named temp/workspace/framework closure rows
- named allocator reserved-unallocated rows
- per-row method/accuracy labels in the CSV/JSON sidecar
- step-wise stacked memory attribution, not only a single peak snapshot

The plot may include approximate closure buckets, but it must not show a large
unnamed or unexplained residual.

### 9. Tests

Update or add tests in:

- `tests/training/test_profile_lora_backends.py`
- possibly new file: `tests/training/test_lf_memory_breakdown.py`

Test cases:

1. Synthetic `memory_breakdown.jsonl` closes to peak.
2. Persistent tensor accounting groups LoRA, attention, embedding, routed expert
   CPU/pinned weights correctly.
3. Activation delta rows appear when module hooks are enabled.
4. `memory.md` no longer calls persistent tensor rows "fine-grained" by
   themselves.
5. Saved-tensor hook schema and memory-breakdown schema can coexist.
6. Plot script renders from synthetic input.
7. `PROFILE_MEMORY_BREAKDOWN=1` with unsupported backend gives a clear error or
   a supported output.

## Output Schema

### `memory_breakdown.jsonl`

One JSON row per rank, step, and phase.

Example:

```json
{
  "schema_version": 1,
  "rank": 0,
  "world_size": 1,
  "step": 6,
  "phase": "after_forward",
  "allocated_bytes": 63905503898,
  "reserved_bytes": 71072481280,
  "peak_allocated_since_step_begin": 68567071232,
  "peak_reserved_since_step_begin": 71072481280,
  "persistent_bytes": {
    "attention": {"weight": 1811963904, "grad": 0, "optimizer_state": 0},
    "lora": {"weight": 6857687040, "grad": 6857687040, "optimizer_state": 914358272}
  },
  "activation_bytes": {
    "attention": 123456789,
    "routed_experts": 234567890,
    "mlp_dense": 345678901
  },
  "closure_bytes": {
    "framework_temp_workspace": 123456789,
    "allocator_reserved_unallocated": 456789012
  },
  "methods": {
    "persistent_bytes": "exact tensor size",
    "activation_bytes": "measured forward allocated delta",
    "framework_temp_workspace": "inferred residual to peak",
    "allocator_reserved_unallocated": "reserved - allocated snapshot"
  }
}
```

### `memory_breakdown_summary.json`

Aggregated summary for postprocessing and plotting.

Required fields:

```text
peak_hbm_bytes
selected_phase
selected_step
rank_count
breakdown_rows[]
closure_ok
closure_error_bytes
notes[]
```

Each `breakdown_rows[]` entry:

```json
{
  "memory_space": "GPU HBM",
  "group": "activation",
  "component": "routed_experts",
  "kind": "activation",
  "bytes": 123456789,
  "method": "measured forward allocated delta",
  "accuracy": "approximate"
}
```

## Attribution Rules

### Exact rows

Use exact tensor sizes for:

- GPU model parameters
- GPU gradients
- GPU optimizer state tensors
- CPU host expert weights
- CPU pinned expert weights
- CUDA buffers visible as model buffers

### Activation rows

Use forward pre/post hook deltas:

```text
delta = memory_allocated_after_module_forward - memory_allocated_before_module_forward
```

Only positive deltas count as activation ownership. Negative deltas are ignored
for owner attribution but still affect global phase snapshots.

If activation checkpointing is enabled:

- for per-layer modules inside a checkpointed region, use `max(layer)` instead
  of `sum(layer)`
- for top-level modules outside checkpointed regions, use `sum`

This mirrors the Gemma profiler logic.

### Residual closure rows

Do not call this "unknown" in the main report. Split it into named approximate
rows:

```text
framework_temp_workspace
allocator_reserved_unallocated
loss_logits_input_temp
optimizer_temp_workspace
```

Initial implementation can use a simple rule:

- `allocator_reserved_unallocated = max(0, reserved_bytes - allocated_bytes)`
- `framework_temp_workspace = max(0, peak_allocated_since_step_begin - known_allocated_stack)`

Later implementation can improve this by assigning temp/workspace residual to
the nearest active semantic phase or module.

## Accuracy Labels

Every row must carry one of:

```text
exact tensor size
exact allocator snapshot
measured forward allocated delta
saved-tensor hook unique bytes
inferred residual to peak
prorated residual by module peak delta
not collected
```

The Markdown should display this label. Approximate rows are acceptable, but
they must be explicit.

## Recommended User Workflows

Timing truth:

```bash
--profilers nsys --profile-memory-breakdown false --profile-memory-attribution auto
```

Memory attribution only:

```bash
--profilers source --profile-memory-breakdown true --profile-memory-attribution auto
```

Timing plus memory attribution:

```bash
--profilers source,nsys --profile-memory-breakdown true --profile-memory-attribution auto
```

Expected outputs for the mixed workflow:

```text
existing timing/perf plots      -> nsys rows only
timing combined/               -> nsys rows only
source memory_breakdown.*      -> source rows only
per-run memory plots           -> source memory_breakdown.* only
config memory_combined/        -> source memory_breakdown.* only
outer memory_combined/         -> source memory_breakdown.* only
```

Lightweight source timing/memory without hook overhead:

```bash
--profilers source --profile-memory-breakdown false --profile-memory-attribution false
```

Do not use hook-enabled runs for clean timing claims.

## Migration Plan

### Phase 1: Honest current report

- Rename current `Fine-Grained Memory Attribution` to `Persistent Tensor
  Accounting`.
- Add explicit text that activations are not decomposed unless memory breakdown
  or saved-tensor hooks are enabled.
- Fix LF saved-tensor schema mismatch if present:
  `saved_tensors.rows` and `saved_tensors.by_owner` should both render.

### Phase 2: Add memory breakdown collection

- Implement `LFMemoryBreakdownProfiler`.
- Emit `memory_breakdown.jsonl`.
- Add source-profile pointers to the output.

### Phase 3: Add summary and Markdown

- Build `memory_breakdown_summary.json`.
- Render `memory_breakdown.md`.
- Replace current top memory section with closure-to-peak rows.

### Phase 4: Add plot

- Add `plot_lf_memory_breakdown.py`.
- Create stacked GPU HBM plot plus separate CPU/pinned panel.
- Create memory-over-steps stacked plots.
- Use shared absolute GiB y-axis limits for combined memory plots.
- Generate per-source-run memory plots under each source result directory.
- Generate config-level `memory_combined/` plots from source dirs under one
  config root.
- Generate outer `memory_combined/` plots from all source dirs under the
  precision/output root, plus optional model-split combined memory plots.
- Wire the sweep driver so memory-attribution source runs call this dedicated
  plotter, while generic timing/perf plots continue to use nsys rows only when
  nsys is present.

### Phase 5: Validate on real LF runs

Run one Qwen3 `s4096` source memory profile:

```bash
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_MEMORY_ATTRIBUTION=auto \
scripts/lf/profile_lora_lf.sh --profilers source --seq-lens 4096 ...
```

Check:

- closure error is near zero
- peak matches `torch.cuda.max_memory_allocated`
- CPU host/pinned expert weights match adapter setup log
- activation rows are nonzero for attention/mlp/experts
- no giant unnamed residual remains in the main table

## Non-Goals

- Do not make memory profiling timing authoritative.
- Do not require Nsight for memory breakdown.
- Do not pretend kernel workspaces or allocator cache can be exactly assigned to
  a Python module without approximation.
- Do not remove existing coarse allocator stage metrics.

## Expected Result

After implementation, `memory.md` should no longer show only persistent tensors
beside a huge peak. It should show a readable stack like:

```text
GPU HBM peak: 66.56 GiB

weights:
  lora
  attention frozen
  embedding frozen
gradients:
  lora
optimizer:
  lora AdamW state
activations:
  attention
  routed_experts
  mlp_dense
  lora
temp/workspace/framework:
  framework_temp_workspace
allocator:
  reserved_unallocated
```

Each row states whether it is exact, measured, or inferred.
