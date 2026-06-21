# Fix LF Profiling Attribution (timing + memory, forward/backward + phases)

Goal: make the LoRA-SFT LF profiler attribute **timing** and **memory** to the
forward pass, the backward pass, and the phases within them with **no vagueness** —
peak-vs-total clearly separated, per-stage numbers that are genuinely per-stage (not a
collapsed step max), backward attributed (not a black box), and tables/plots that label
exactly what they show. The cross-config (activation-offload policy) comparison must
actually span the configs that were run.

This doc is the staged implementation spec. Audit findings that motivate it are in the
session notes; the concrete code sites are listed per stage below.

Non-goals:
- **Do not add new user-facing runtime knobs.** Reuse the existing surface only:
  `PROFILE_SYNC`, `PROFILE_LEVEL`, `PROFILE_MEMORY_BREAKDOWN`,
  `PROFILE_MEMORY_BREAKDOWN_MODULES`, `PROFILE_MODULE_FILTER`, `PROFILE_MEMORY_ATTRIBUTION`,
  `CONTINUE_ON_ERROR`. All five fixes are achievable without a new env axis or a new
  `profile_lora_lf.sh` option.
- Do not change the forward/backward boundary semantics (`step.forward` = `compute_loss`,
  `step.backward` = `Accelerator.backward`). They are correct; recompute already bills to
  backward. We only fix *attribution faithfulness*, not the boundaries.
- Do not break `validate_lf_memory_capacity_schema.py` closure (`closure_ok`). Every memory
  change must keep `sum(component rows) == peak_allocated` (+ allocator gap == reserved).
- Do not delete the `memory_attribution.saved_tensors.by_owner` block; only relabel it.

Knob defaults to keep in mind (set in `scripts/lf/run_lf_lora_sft.sh`, parsed in
`scripts/lf/profile_lora_lf.sh`):
- `PROFILE_SYNC=false` → host wall-clock timing, no per-range CUDA sync.
- `PROFILE_LEVEL=op` → forward op NVTX ranges on; backward module/op ranges still gated.
- `PROFILE_MEMORY_BREAKDOWN=true` (in the e2e sweep) → enables `LFMemoryBreakdownProfiler`
  **and** disables per-stage peak reset (the Stage 1 bug).
- `CONTINUE_ON_ERROR=false` → sweep aborts on first failing config, so combined plots never
  regenerate across the policy axis (the Stage 5 bug).

---

## Current local facts (verified file:class/function anchors)

> Line numbers are as of this audit. `asym_gemm/profiling/lf_trace.py` has uncommitted
> working-tree edits, so **re-grep the symbol name before editing** rather than trusting a
> raw line number.

`scripts/lf/run_lf_profiled_train.py`
- `class LFProfileRecorder` (2317). Fields: `reset_stage_peak_stats: bool=True` (2320),
  `global_peak_allocated_bytes:int=0` (2322), `global_peak_reserved_bytes:int=0` (2323).
- `LFProfileRecorder.stage(name, *, sync=False)` (2326): conditional
  `torch.cuda.reset_peak_memory_stats()` (2336-2340); `max_memory_allocated()` read (2354-2355);
  **Python running max** `self.global_peak_allocated_bytes = max(...)` (2358-2359) — already
  reset-proof.
- `LFProfileRecorder._stage_memory_rows()` (2397) — averages StageRecord fields into
  `stage_memory.rows`.
- `LFProfileRecorder._step_sample_rows(...)` (2448) — builds the
  `forward_peak_allocated_bytes` / `backward_peak_allocated_bytes` (+ `_delta_`) per-step fields.
- `LFProfileRecorder.report(...)` (2616) — emits `memory.gpu`, `forward.*`, `backward.*`,
  `step_samples`, `stage_memory`.
- Construction coupling: `reset_stage_peak_stats=not trace_config.memory_breakdown` (2726).

`asym_gemm/profiling/lf_trace.py`
- `class LFTraceConfig` (73); `@property backward_module_ranges_enabled` (141) — true iff
  `level=="deep"` or `"backward"` in module_filter.
- `class LFMemoryBreakdownProfiler` (720); `self._component_stack` (744).
- Breakdown module hooks: nested `forward_pre` (813) / `forward_post` (818); registered via
  `register_forward_pre_hook` / `register_forward_hook` (847-848). **No backward hooks here.**
- `record_phase(phase, model, optimizer)` (890).
- `_remember_live_activation(...)` (964); `_capture_saved_activation_peak(...)` (1060).
- `_update_peak_values(...)` (1095) — **cumulative** running max, never reset per phase
  (1115-1116); `_refresh_peak_snapshot()` (1118); `saved_tensor_pack/unpack` (1124/1143);
  `_snapshot_values()` (1158) reads bare `torch.cuda.max_memory_allocated()` (1164).
- NVTX op-range installer (separate from breakdown): `_install_module_hooks_once(handle, model)`
  (1957); forward NVTX `forward_pre/forward_post` (1987-2006); backward NVTX gated at
  `if handle.config.backward_module_ranges_enabled:` →
  `register_full_backward_pre_hook` / `register_full_backward_hook` (2007-2010).

`scripts/lf/postprocess_lf_profile_artifacts.py`
- `_source_summary_markdown(profile)` (1133); stage table header (1173); stage rows
  `step.forward`/`step.backward`/`step.forward + step.backward` (1180-1182).
- `_source_latency_markdown(profile)` (1410); `_source_memory_markdown(profile)` (1736).
- nsys readers keyed on `profile["stages"]`: `_timing_by_stage` (1967), `_timing_by_op` (1996),
  `_timing_by_layer` (2017), `_timing_by_module` (2031).
- Writers: `_write_source_artifacts(...)` (2109), `_write_profile_csv_artifacts(...)` (2174).

`scripts/lf/postprocess_nsys_lora.py`
- Slices the GPU timeline on the two NVTX ranges `step.forward` / `step.backward`
  (≈1326-1327, 1490-1491, 3057-3058). Source of `profile["stages"]` and per-op kernel time.

`scripts/plotting/plot_lf_memory_breakdown.py`
- `_legend_handles(keys, include_peak=True)` (868) — only "Peak allocated" dashed line label.
- `_plot_single_peak` (1364), `_plot_single_steps` (1455), `_plot_single_phases` (1504),
  `_plot_combined_peak` (1534), `_plot_combined_phases` (1628). Every one calls
  `ax.set_ylabel("Memory (GiB)")` (1398/1461/1509/1566/1612/1658) with no peak/reserved qualifier.

`scripts/plotting/plot_activation_recompute_sweep.py`
- Per-step plot specs include `forward_peak_*` / `backward_peak_*` and `forward_timing` /
  `backward_timing` (≈1979). `collect_rows` builds the combined index from discovered leaves.

`scripts/lf/profile_lora_lf.sh`
- `CONTINUE_ON_ERROR` default false (71), normalized (1910), gates (3153/3230/3273).
- `plot_running_combined()` (2785), `plot_memory_running_combined()` (2859);
  `memory_plot_roots` registration (2145/2430); final combined plotting block (3101+).

Real measured artifact used as the regression baseline (config #1, source leaf):
`profiling_both/asym_long_sft_smoke__lora__lf__bf16/llama-4-scout-17b-16e__gpus1__b4_s4096_ga1_w1_s1_r64_a16_drop000/asym_cpuadamwds__source__norecomp__polnone__routerwhole__expact1__attnact0__layeract0__layergc0__loraafwdhbm__actrecomp0__xunpack0__ligerloss0__gradofftrue__weightofftrue/b4_s4096_ga1/`
- `forward.total_milliseconds=5448.6`, `backward.total_milliseconds=54465.3` (time: already clean).
- `forward_peak_allocated_bytes == backward_peak_allocated_bytes == 149.024 GiB` (COLLAPSED).
- `forward_peak_allocated_delta_bytes=148.877 GiB`, `backward_peak_allocated_delta_bytes=12.724 GiB`
  (only the delta distinguishes the stages).
- `memory_breakdown_summary.json`: `peak_allocated_hbm_bytes=149.02`, `peak_reserved=150.42`,
  `closure_ok=true`; phase rows all stamp `peak_allocated_since_step_begin=148.99 GiB`.

---

## Stage 0 — Preflight (no code change)

Scope: establish the small, fits-in-memory validation workload and capture a golden baseline
so every later stage can diff against "before".

Steps:
1. Pick a workload that fits HBM (the s4096/b4 gc-exp config OOMs; use a small one):
   `WORKLOADS='2048|1|1'`, `MAX_STEPS=2`, `WARMUP_STEPS=1`.
2. Capture baseline source artifacts for one offload config and one GC config:
   ```bash
   OVERWRITE=true PROFILERS=source \
   PROFILE_MEMORY_BREAKDOWN=true PROFILE_SYNC=true PROFILE_LEVEL=op \
   MODEL_SPECS='meta-llama/Llama-4-Scout-17B-16E|1' \
   BACKEND_SPECS='asym_cpuadamwds|norecomp' \
   ASYMM_EXP_ACT_POLICIES='none|true|true|false|false,gc-exp|false|false|false|false' \
   CONTINUE_ON_ERROR=true WORKLOADS='2048|1|1' MAX_STEPS=2 WARMUP_STEPS=1 GPU_POOL=3 \
   OUTPUT_ROOT="$PWD/profiling_fixcheck_baseline" \
   bash scripts/lf/profile_lora_lf.sh
   ```
3. Record, from each leaf's `step_samples.json` and `memory_breakdown_summary.json`:
   forward/backward time, `forward_peak_allocated_bytes`, `backward_peak_allocated_bytes`,
   the two delta fields, `peak_allocated_hbm_bytes`, `closure_ok`, and `selected_phase`.

Pass condition: baseline runs complete and `closure_ok=true`. This is the reference for
Stages 1–5 (no behavior change yet).

Risk to watch: if even `2048|1|1` OOMs for `gc-exp`, drop to `MODEL_SPECS` Qwen3-30B-A3B or
use the offload config alone for Stages 1/3/4 and revisit gc-exp under Stage 2.

---

## Stage 1 — Faithful per-stage (forward vs backward) PEAK memory

Problem: `forward_peak_allocated_bytes` and `backward_peak_allocated_bytes` collapse to the
same step-level cumulative max whenever `PROFILE_MEMORY_BREAKDOWN=true`, because
`reset_stage_peak_stats=not trace_config.memory_breakdown` (run_lf_profiled_train.py:2726)
turns off the per-stage `reset_peak_memory_stats()`. Result: the forward/backward peak split
is carried only by the secondary `_delta_` fields, and `summary.md` prints the identical peak
on both stage rows.

Scope:
- `scripts/lf/run_lf_profiled_train.py`
  - `LFProfileRecorder.stage()` (2326)
  - `LFProfileRecorder` construction site (2726)
- `asym_gemm/profiling/lf_trace.py`
  - `LFMemoryBreakdownProfiler._update_peak_values` (1095), `_refresh_peak_snapshot` (1118),
    `_snapshot_values` (1158), and wherever the summary's `actual_peak_allocated_hbm_bytes`
    is finalized (search `actual_peak_allocated_hbm_bytes` / `_current_peak_allocated`).

Intended code changes:
1. **Always reset per stage.** Remove the coupling at 2726: set
   `reset_stage_peak_stats=True` unconditionally (or delete the field and always reset).
   This makes `max_memory_allocated()` at 2354 a true *within-stage* peak again, so the
   forward and backward StageRecords diverge.
2. **Keep the step-global peak reset-proof for the breakdown.** The recorder already holds
   `self.global_peak_allocated_bytes` (running max across resets, 2358) — make the breakdown
   summary's step-global peak come from a running max, not a bare post-hoc
   `torch.cuda.max_memory_allocated()`:
   - `LFMemoryBreakdownProfiler` already maintains `self._current_peak_allocated`
     (running max, 1115). Confirm the summary's `actual_peak_allocated_hbm_bytes` is sourced
     from `self._current_peak_allocated` (resilient) and **not** re-read from
     `torch.cuda.max_memory_allocated()` after a stage reset. If it re-reads, change it to use
     the running max.
   - Guarantee a breakdown sample is taken at the forward→backward boundary **before** the
     backward stage's reset. `record_phase("after_forward")` already runs before `step.backward`
     begins, so the forward peak is captured; assert this ordering holds (it is the invariant the
     reset relies on).
3. **Schema/labels unchanged.** `forward_peak_allocated_bytes` / `backward_peak_allocated_bytes`
   keep their names but now hold real per-stage peaks; `validate_lf_memory_capacity_schema.py`
   already mandates these fields (no schema edit needed).

Validation (reuse Stage 0 command, `OUTPUT_ROOT=.../profiling_fixcheck_stage1`):
- `step_samples.json`: assert `forward_peak_allocated_bytes != backward_peak_allocated_bytes`
  for the measured step (was equal). Forward peak should be ≈ step peak; backward peak should
  be lower for the offload config.
- `memory_breakdown_summary.json`: `closure_ok=true` still, and `peak_allocated_hbm_bytes`
  unchanged from baseline (the step-global peak must not regress).

Pass conditions: per-stage peaks differ and reconcile (`max(forward_peak, backward_peak)`
≈ `peak_allocated_hbm_bytes`); closure preserved; baseline step-global peak within rounding.

Risk to watch: if the breakdown summary silently used the bare torch counter for its peak,
the naive reset will *lower* the reported step peak. The running-max sourcing in change (2)
is the guard — verify `peak_allocated_hbm_bytes` is byte-identical to baseline.

---

## Stage 2 — Backward per-module attribution (memory + timing)

Problem: backward is a black box. (a) `LFMemoryBreakdownProfiler` installs **only** forward
hooks (lf_trace.py:847-848), so backward allocations fall into a residual `source_runtime`
bucket. (b) Backward op/module NVTX ranges are gated off by default
(`backward_module_ranges_enabled` requires `PROFILE_LEVEL=deep` or `"backward"` in
`PROFILE_MODULE_FILTER`), and in the audited run the backward stage's per-op kernel time was
unattributed — so `timing_by_module.csv` is a forward-only picture even though backward is
~10× the wall time.

Scope:
- `asym_gemm/profiling/lf_trace.py`
  - `LFMemoryBreakdownProfiler`: the module-hook installer around 805-848 (add backward hooks),
    `_component_stack` push/pop, `_update_peak_values` (1095) owner resolution.
  - `_install_module_hooks_once` (1957) + `backward_module_ranges_enabled` (141): ensure backward
    NVTX ranges are on for the profile levels the sweep uses.
- `scripts/lf/postprocess_nsys_lora.py`: verify the `step.backward` slice actually attributes
  per-op kernel time (the 1326-1327 / 3057-3058 slicing).
- `scripts/lf/postprocess_lf_profile_artifacts.py`: `_timing_by_module` (2031) /
  `_timing_by_op` (1996) consume `profile["stages"]`; confirm backward rows are non-empty.

Intended code changes:
1. **Backward memory hooks in the breakdown profiler.** In the installer that currently adds
   `register_forward_pre_hook(forward_pre)` / `register_forward_hook(forward_post)` (847-848),
   also add `register_full_backward_pre_hook` / `register_full_backward_hook` that push the same
   `_component` onto `_component_stack` on backward entry and pop on exit — mirroring
   `forward_pre`/`forward_post` (813-845). Then backward `_update_peak_values` owner resolution
   (1106-1112) attributes backward peak growth to the real component instead of `source_runtime`.
   - Keep the storage-key dedup (`_saved_tensor_storage_key`) so backward attribution does not
     double-count tensors already owned by forward.
2. **Turn backward NVTX ranges on for the standard sweep level.** Two options (pick the lower-risk
   one after a dry check):
   - (a) Make `backward_module_ranges_enabled` also true for `level=="op"` (not only `deep`), so
     the default `PROFILE_LEVEL=op` sweep captures backward op ranges; **or**
   - (b) Leave the property and instead add `backward` to the default `PROFILE_MODULE_FILTER` in
     `scripts/lf/run_lf_lora_sft.sh` (no new knob; just a default-value change).
   Prefer (a) if op-level forward already implies op-level backward is wanted; otherwise (b).
3. **Confirm the nsys backward slice is populated.** If `timing_by_stage.csv` `step.backward`
   shows `cuda_kernel_busy_milliseconds=0` while the nsys `profile.json` `stages[]` backward
   kernel time is non-zero, the bug is in `postprocess_nsys_lora.py` per-stage aggregation, not in
   capture — fix the aggregation so backward kernel-busy is summed into the stage row.

Validation:
- Re-run Stage 0 command with `PROFILE_LEVEL=op` (and `PROFILERS=both` for the nsys half on one
  config). Check:
  - `memory_breakdown.jsonl` `after_backward` rows: component growth attributed to real modules
    (experts/attention/lora), `source_runtime` share materially reduced.
  - nsys `timing_by_module.csv` / `timing_by_op.csv`: rows with `stage=step.backward` and non-zero
    `gpu_kernel_milliseconds` for `experts`/`attention`.
- `closure_ok=true` unchanged.

Pass conditions: backward has per-module memory growth and per-module kernel time; the
`source_runtime` residual in backward drops below a set threshold (e.g. <10% of backward growth).

Risk to watch: full_backward_hook fires once per module-output-grad and can mis-order with
re-entrant gradient checkpointing (the gc-* and recompute configs). Validate on a `gc-exp`
config (small workload) that the push/pop stack stays balanced (no stack underflow / leaked
component). Guard the pop with the same `if self._component_stack:` check used at 844.

---

## Stage 3 — Per-phase PEAK (de-cumulate) so after_forward/after_backward are real

Problem: each `memory_breakdown.jsonl` phase row carries a `peak_allocated_since_step_begin`
that is a *cumulative* running max (`_update_peak_values` only ever takes `max`, never resets
between phases — lf_trace.py:1115). So `after_forward`, `after_backward`,
`before/after_optimizer_step` all report the identical step peak (148.99 GiB), and
`memory_by_phase_stacked.png`'s dashed peak overlay is wrong. The per-phase **current**
allocation (`allocated_bytes`) is already correct and meaningful.

Scope:
- `asym_gemm/profiling/lf_trace.py`
  - `record_phase` (890), `_update_peak_values` (1095), `_refresh_peak_snapshot` (1118),
    and the per-phase row emission (search where phase rows get `peak_allocated_since_step_begin`).
- `scripts/plotting/plot_lf_memory_breakdown.py`
  - `_plot_single_phases` (1504), `_plot_combined_phases` (1628).

Intended code changes:
1. **Add a phase-local peak alongside the cumulative one.** Keep
   `peak_allocated_since_step_begin` (it is the honest step-global running max and is used for
   closure), but also record a `peak_allocated_within_phase` per phase: snapshot the torch peak
   at `record_phase` entry, `reset_peak_memory_stats()` at the start of each phase window, and read
   `max_memory_allocated()` at the next phase boundary. (Coordinate with Stage 1's reset ownership
   so the step-global running max is preserved across these resets.)
2. **Name them unambiguously** in the JSONL/CSV: `peak_allocated_since_step_begin` (cumulative)
   vs `peak_allocated_within_phase` (phase-local). Update
   `validate_lf_memory_capacity_schema.py` to accept (not require, to stay back-compatible) the
   new field.
3. **Fix the phase plot.** In `_plot_single_phases` / `_plot_combined_phases`, draw the dashed
   peak line from `peak_allocated_within_phase`, and label the stacked bars as **current
   allocation at phase boundary** (they already are), so after_forward vs after_backward is a true
   memory-timeline view.

Validation: in `memory_breakdown.jsonl`, `peak_allocated_within_phase` must differ across phases
(after_forward high, after_optimizer_step low), while `peak_allocated_since_step_begin` stays
constant. `memory_by_phase_stacked.png` peak line should step down across phases.

Pass conditions: phase-local peak is non-constant and ≤ step-global peak at every phase; closure
still computed from the cumulative field.

Risk to watch: extra `reset_peak_memory_stats()` calls interact with Stage 1 and with the
`saved_tensor_pack/unpack` refresh (1140/1149). Land Stage 1 first; make a single owner of the
torch peak counter (the recorder) and have the breakdown read running maxes only.

---

## Stage 4 — Table/plot labeling honesty (no metric change)

Problem: artifacts don't say what they show. (a) Every stacked memory plot Y axis is just
"Memory (GiB)" while the bar height is *reserved-stack-sum* and the dashed line is
*allocated-peak*. (b) `summary.md` stage table repeats the same step-level peak on the forward
and backward rows (looks per-stage, isn't — Stage 1 fixes the data; Stage 4 fixes the label).
(c) `memory_attribution.saved_tensors.by_owner` reports 256.7 GiB on a 150 GiB device
(lifetime/reference bytes) but is labeled plainly `saved_tensors`. (d) Source-mode
`timing_by_op/module/layer.csv` are written as empty 2-byte files with no note.

Scope:
- `scripts/plotting/plot_lf_memory_breakdown.py`: `set_ylabel` sites
  (1398/1461/1509/1566/1612/1658), `_legend_handles` (868).
- `scripts/lf/postprocess_lf_profile_artifacts.py`: `_source_summary_markdown` stage table
  (1173-1182), `_source_memory_markdown` (1736) for the saved-tensor-owner heading,
  `_write_source_artifacts` (2109) for the empty-CSV note.

Intended code changes:
1. Y-axis label → `"Memory (GiB) — bars: peak reserved stack; dashed: peak allocated"` (or split
   into ylabel + a one-line subtitle). Add a "Peak reserved (stack total)" legend entry in
   `_legend_handles`.
2. Stage table: after Stage 1, the forward/backward memory columns are genuinely per-stage; rename
   the header to make that explicit (`fwd/bwd peak allocated MiB`) and, if a column is a
   step-level value, label it `step peak` rather than putting it on a stage row.
3. Rename the saved-tensor-owner block to `saved_tensors_by_owner_reference_bytes` (or add a
   `note` field: "reference/lifetime bytes, may exceed device HBM; not live-resident") in the
   markdown and the emitted JSON. Do not change the numbers.
4. When a source-mode run writes empty `timing_by_*` CSVs, write a single header row or a
   sibling `timing_by_op.NOTE.txt` stating "op/module timing requires PROFILERS=nsys|both".

Validation: visual diff of the four touched artifacts; confirm numbers are byte-identical to
Stage 1/3 outputs (labels only). No schema change.

Pass conditions: every memory axis/column states peak-vs-reserved and fwd/bwd/step scope; no
silently-empty CSV.

Risk to watch: keep `table.md == summary.md` (they are emitted identical); change both via the
single `_source_summary_markdown` producer, not in two places.

---

## Stage 5 — Combined plots show only a sliver of the configs present (filter over-narrowing)

Root cause (corrected after direct inspection — this is NOT primarily a staleness/abort bug):
the config-root and precision-level combined plots are **filtered to the current sweep's axis
values**, then the combined dir is **overwritten on every sweep**. So a narrow sweep clobbers a
broad one and excludes every sibling config that is physically present on disk.

### New behavior (the contract this stage must deliver)

The combined comparison artifacts (`combined/`, `memory_combined/`, `c2c_combined/` at the
config-root, and the precision-level equivalents) must **plot every previously-completed config
that is available on disk under that root — not only the configs of the sweep currently running.**

Precisely:
- **Discovery, not sweep-scope:** the set of configs in a combined plot is determined by what is
  present and schema-valid under `--input-root`, never by the axes of the invoking sweep.
- **Additive across runs:** a new sweep ADDS its configs to the comparison and may refresh existing
  ones; it must **never remove or hide** configs that earlier sweeps completed in the same root.
- **Completed only:** include a config iff its `memory_breakdown_summary.json` (memory) /
  `profile.json` (timing) is present and passes schema validation. Partial/failed configs (e.g. a
  gc-exp that OOM'd) are skipped, not errored.
- **Profiler dedup is the only narrowing allowed:** collapse the both-mode `__nsys__`/`__source__`
  pair to one row per config (memory → `source`, timing/C2C → the plot profiler). No
  backend/recompute/policy/activation-axis narrowing.
- **Rebuild-from-disk, idempotent:** each write fully recomputes the combined from the current
  on-disk set, so the result is the same regardless of which sweep triggered it and order does not
  matter.

How the filter narrows today (confirmed):
- The combined plotters discover runs by rglob of `--input-root` (the config-root) for
  `memory_breakdown_summary.json` (plot_lf_memory_breakdown.py:595) / `b*_s*_ga*` dirs
  (plot_activation_recompute_sweep.py:874). Discovery sees **all** leaves.
- `_matches_filters` (plot_lf_memory_breakdown.py:715) drops a run unless its metadata is in the
  allowed set for **every** filter dimension. Critically, an **empty filter set == match-all**
  (lines 736-738: `if not allowed: continue`). So passing *no* `--backend/--expact/...` includes
  everything under `--input-root`.
- But the orchestrator passes the **current sweep's** axis arrays as filters:
  `append_backend_filters` → `${backends[@]}`, `append_activation_axis_filters` →
  `${plot_expact_values[@]}` / `plot_attnact_values` / `plot_layeract_values` / `plot_layergc_values`
  (profile_lora_lf.sh:1490-1548), and `plot_config_root` additionally passes
  `--expert-recompute-policies "${expert_policies[@]}"` (2779). These are AND-ed across dimensions,
  so any leaf whose backend / expact / attnact / layergc / expert_policy isn't in the running
  sweep's value-set is excluded — even though it is present and schema-valid.
- `plot_config_root` writes `${config_root}/combined` and `plot_memory_config_root` writes
  `${config_root}/memory_combined` every sweep → the last (often narrower) sweep clobbers the
  broader one.

Evidence (this `profiling_both` config-root, verified):
- 15 leaf dirs present = **8 distinct configs**, all schema-valid: the 6 `asym_cpuadamwds|norecomp`
  policies (gc-exp partial after OOM, the other 5 full `[PSM]`), plus `asym_cpuadamwds|recomp|polnone`
  and `zero3_offload|recomp|polnone` baselines.
- Combined dirs were rewritten at 16:46, **after** the last policy leaf finished (16:44) — so this
  is not staleness.
- Yet `memory_combined/combined_memory_breakdown.csv` has only **2 configs** (asym + zero3,
  `recomp/polnone`); `combined/activation_recompute_sweep_index.csv` has **4 rows** (those same 2
  configs × nsys/source). The 6 policy configs were filtered out by the last sweep's narrow axes.

`CONTINUE_ON_ERROR=false` is a secondary contributor: a sweep that dies at the gc-exp OOM never
reaches its own `plot_config_root`, so it cannot even write its (correctly-scoped) combined —
leaving whatever a prior narrow sweep wrote.

Scope:
- `scripts/lf/profile_lora_lf.sh`
  - Filter builders: `append_backend_filters` (1490), `append_recompute_filters` (1502),
    `append_activation_axis_filters` (1544) and its `append_expact/attnact/layeract/layergc_filters`
    (1520-1542), `append_sweep_plot_filters` (1567), `memory_plot_filters` (1585).
  - Config-root plotters: `plot_config_root` (2768; passes `append_sweep_plot_filters` +
    `--expert-recompute-policies "${expert_policies[@]}"`), `plot_memory_config_root` (2876;
    passes `memory_plot_filters`).
  - Precision-level: `run_precision_combined_plot` (3023), `memory_precision_combined_cmd` (3003),
    invoked at 3084/3104.
  - `--input-root` is set by `plot_cmd_base` (1437) / `memory_combined_plot_cmd_base` (1455).
  - `CONTINUE_ON_ERROR` gates (3153/3230/3273).
- `scripts/plotting/plot_lf_memory_breakdown.py`: `_matches_filters` (715), empty=match-all
  (736-738), rglob discovery (595).
- `scripts/plotting/plot_activation_recompute_sweep.py`: `--input-root` rglob (874), filter
  (854-861) — same empty=match-all semantics; same change applies.

Intended code changes:
1. **Make the config-root + precision combined disk-driven (the real fix).** In `plot_config_root`
   (2768), `plot_memory_config_root` (2876), and the precision combined commands (3003/3023/3084/3104),
   **stop passing the sweep-axis narrowing filters** — drop `append_sweep_plot_filters` /
   `memory_plot_filters` / `append_activation_axis_filters` / `append_backend_filters` /
   `append_recompute_filters` and the `--expert-recompute-policies "${expert_policies[@]}"` line
   (2779). Rely on `--input-root <config_root>` rglob + empty=match-all so **every** present,
   schema-valid config is included.
   - Keep only the dedup-necessary filter: `--profiler source` for memory, `--profiler` (plot
     profilers) for timing, so the both-mode nsys/source pair collapses to one row per config.
   - The config-root is already one workload/precision/model, so workload/precision filters are
     unnecessary there; the precision-combined may keep a workload-base split.
2. **Alternative (only if mixing must be guarded):** instead of dropping filters, build the filter
   value-sets from a **glob of the config-root** (union of axis values actually on disk) rather than
   from the sweep arrays. More code than option 1; pick option 1 unless a real need to exclude
   present configs appears.
3. **Leave the running/per-iteration previews alone.** `plot_running_combined` (2785) and
   `plot_memory_running_combined` (2859) intentionally scope to the current config via
   `append_current_activation_axis_filters` (1551) and write under the per-run `plots/_combined`;
   they are previews, not the canonical config-root combined. Do not change them.
4. **Rebuild-from-disk makes the clobber harmless** (the combined always reflects everything
   present). Keep `--clean-output` so a removed config does not linger as a stale series.
5. **Secondary:** run sweeps with `CONTINUE_ON_ERROR=true` so a failing/OOM config records
   `failed:1` and the sweep still reaches `plot_config_root`. OOM configs are simply absent from the
   lines (the intended signal).

Validation:
```bash
# 1) Prove the fix on the EXISTING profiling_both config-root, no re-train, no filters:
CR='profiling_both/asym_long_sft_smoke__lora__lf__bf16/llama-4-scout-17b-16e__gpus1__b4_s4096_ga1_w1_s1_r64_a16_drop000'
.venv/bin/python scripts/plotting/plot_lf_memory_breakdown.py \
  --input-root "$CR" --output-dir /tmp/memcombo_allconfigs --clean-output --combined-only --profiler source
# expect: combined_memory_breakdown.csv distinct configs == count of schema-valid leaves (≈7-8 here),
#         not 2.
```
```bash
# 2) End-to-end after the orchestrator change (small workload):
CONTINUE_ON_ERROR=true OVERWRITE=true PROFILERS=both \
PROFILE_MEMORY_BREAKDOWN=true PROFILE_SYNC=true \
MODEL_SPECS='meta-llama/Llama-4-Scout-17B-16E|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
ASYMM_EXP_ACT_POLICIES='none|true|false|false|false,none|true|true|false|false,none|true|true|false|true,gc-exp|false|false|false|false,gc-attn-exp|false|false|false|false,gc-layer|false|false|false|false' \
WORKLOADS='2048|1|1' MAX_STEPS=2 WARMUP_STEPS=1 GPU_POOL=3 \
bash scripts/lf/profile_lora_lf.sh
```
- `memory_combined/combined_memory_breakdown.csv` and `combined/activation_recompute_sweep_index.csv`
  contain one entry per non-OOM config present in the config-root (all 6 policies, plus any baseline
  leaves that coexist there).
- Re-running a narrow sub-sweep into the same config-root no longer shrinks the combined.

Pass conditions: combined comparison count == number of schema-valid leaves under the config-root;
a subsequent narrow sweep cannot drop previously-present configs from the combined.

Risk to watch:
- Including everything means a config-root that accumulated genuinely-unrelated experiments shows
  them all. That is the intended behavior for "compare everything here"; to scope an experiment, use
  a fresh `OUTPUT_ROOT` or `RUN_NAME` per experiment (document this).
- Series/segment keys must stay distinct across the now-larger set — verify the dedup key
  (backend/policy/expact/attnact/layeract/layergc/recompute/liger, RunRecord.metadata at
  plot_lf_memory_breakdown.py:195) does not collapse two distinct policies into one line.
- `gc-exp` (OOM) has only a partial `memory_breakdown_summary.json`; confirm the schema-v2 filter
  excludes it cleanly (logged as a filter/schema failure) rather than crashing the combined.

---

## Stage 6 — Legend placement: ABOVE the plot box, never inside it

Requirement (global; applies to EVERY plot produced by all three plotters, at every level and in
every subfolder): the legend must sit **above the axes box** (or above the figure for multi-panel
figures), **never inside the data area** and never overlapping bars/lines. Current state is mixed:
the sweep plotter's bare `ax.legend()` calls render *inside* the box (matplotlib default), the
memory plotter anchors to the *right* (`bbox_to_anchor=(1.02, 1.0)`), and C2C uses `loc="lower
right"` (inside). All must move above.

Scope:
- `scripts/plotting/plot_lf_memory_breakdown.py`: `_plot_legend` (≈878), `fig.legend` (≈1615, ≈1661).
- `scripts/plotting/plot_activation_recompute_sweep.py`: `add_legend` (≈1740) and every bare
  `ax.legend(...)` (≈1906, 2116, 2180, 2314, 2380, 2429, 2479, 2543, 2593).
- `scripts/plotting/plot_lf_interconnect_ctc.py`: `fig.legend` (≈712), `ax.legend(loc="lower right")` (≈745).

Intended change:
- Route every legend through a single helper that anchors above the axes:
  `ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=<fit>, borderaxespad=0.0, frameon=False, fontsize=...)`.
  For figure-level legends: `fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=<fit>)`.
- Choose `ncol` so the legend lays out horizontally across the top (e.g. `min(len(labels), 4–6)`).
- Ensure every `savefig` uses `bbox_inches="tight"` so the above-axes legend is not clipped, and
  reserve headroom (tight bbox or `subplots_adjust(top=…)`) so the legend never collides with the title.

Execution (an agent owns this loop, runs FIRST):
- Apply the change, then REGENERATE every subplot in every subfolder under the active precision root
  `profiling_both/asym_long_sft_smoke__lora__lf__bf16/`: per-leaf `plots/`, `memory_plots/`, the
  interconnect plots; config-root `combined/`, `memory_combined/`, `c2c_combined/`; and the
  precision-level combined + model-split subfolders.
- VERIFY by reading a representative PNG from each category and confirming the legend is above the
  box. Iterate (adjust `ncol`/anchor, regenerate) until every sampled plot passes.

Pass condition: in every regenerated plot the legend sits above the box and overlaps no data.

---

## Suggested landing order

0. Stage 6 (legends above the box) — runs FIRST via a dedicated agent loop; presentation-only,
   independent of Stages 1–5; regenerate + visually verify all subplots before executing the rest.
1. Stage 1 (per-stage peak) — unblocks honest fwd/bwd memory; smallest change.
2. Stage 2 (backward attribution) — biggest correctness gain; depends on Stage 1's reset owner.
3. Stage 3 (per-phase peak) — builds on Stage 1's reset ownership.
4. Stage 4 (labels) — pure presentation; do after the data is correct.
5. Stage 5 (combined comparison) — independent of 1–4; the real fix is dropping the sweep-axis
   filters from the config-root/precision combined so it includes every config present on disk
   (`CONTINUE_ON_ERROR=true` is only a secondary contributor). Can be done first if the missing-config
   plots are the priority.

Each stage is independently shippable and independently validated against the Stage 0 baseline.
Keep `validate_lf_memory_capacity_schema.py` `closure_ok=true` green at every stage.
