# Profiling Fixes for Expert Activation Offload

This is a profiling-only plan. It must not change model math, activation
offload lifetime behavior, kernels, LoRA schedules, CPU Adam, or training
computation. The purpose is only to make future memory/latency comparisons
harder to misread.

## Current Code Check

The current profiling code already has most of the machinery that older V4
notes assumed was missing:

- `scripts/lf/profile_lora_lf.sh`
  - owns the main workflow and already passes `PROFILE_MEMORY_ATTRIBUTION`,
    `PROFILE_MEMORY_BREAKDOWN`, `PROFILE_MEMORY_SNAPSHOT`, `PROFILE_SYNC`, and
    `PROFILE_EXTERNAL_MEMORY` into each run.
  - writes `command.txt`, `jobs.tsv`, per-run output folders, config-level
    `ARTIFACTS.md`, memory plots, timing plots, and nsys/C2C plot roots.
  - supports the current policy tuple syntax:
    `policy|expert_act|attn_act[|layer_act]`.
- `scripts/lf/run_lf_profiled_train.py`
  - `_config_from_args` already records the main workload/profile config in
    `source_profile.json`.
  - `_activation_offload_counters_from_model` already gathers activation
    offload rows, stage bytes, CPU live/owned bytes, transfer totals, and
    per-module execution stats when available.
  - `_start_memory_snapshot_recording` / `_dump_memory_snapshot` already capture
    `memory_snapshot.pickle` and support `PROFILE_MEMORY_SNAPSHOT_MAX_ENTRIES`.
  - `LFProfileRecorder.report` already writes memory, timing, LoRA, KT,
    CPUAdam, activation-offload, stage-memory, and memory-breakdown sections.
- `scripts/lf/postprocess_lf_profile_artifacts.py`
  - already writes `memory_breakdown.csv/md`,
    `memory_actual_peak_breakdown.csv`, `lora_counters.csv`,
    `process_memory.csv`, `cpuadam.csv`, `asym_cpu_adamw.csv`,
    `trainable_surface.csv`, timing CSVs, and kernel summaries where available.
- `scripts/testing/analyze_cuda_memory_snapshot.py`
  - already replays torch CUDA `device_traces` and attributes the true peak
    live-set without importing torch.
- `scripts/lf/build_lf_sft_eval_pair.py`
  - already writes dataset `manifest.json`, `token_stats.json`, and
    `validation.json` under the dataset validation result directory.

So the real profiling gap is not "build a profiler". The gap is that the
existing evidence is too scattered: some critical run identity and runtime count
information is still hard to consume automatically.

## Current Artifact Facts

Current b4_s4096 source no-hook acceptance sweep:

- `Qwen/Qwen3-30B-A3B`, `asym_cpuadamwds|norecomp`, batch 4, seq 4096,
  rank 64, alpha 16, dropout 0.00, CPU Adam DS.
- `none|true|false`: `102.312 GiB` peak allocated, `107.469 GiB` peak
  reserved, `45.173s` step, `10.036s` fwd, `35.137s` bwd.
- `gc-exp|false|false`: `126.312 GiB` peak allocated, `131.414 GiB` peak
  reserved, `3.953s` step.
- `none|false|false`: `170.503 GiB` peak allocated, `183.055 GiB` peak
  reserved, `3.129s` step.

Current diagnostic attribution/snapshot run agrees with target peak:

- actual peak: `102.312 GiB` allocated, `107.469 GiB` reserved at
  `after_backward`.
- snapshot replay: `443541` events, one trace, `0` unknown frees.
- exact routed-expert live HBM at peak is tiny: `0.059 GiB`.
- large peak buckets are norms, loss, attention, allocator-unframed, and other
  Python allocations.
- activation-offload lifetime stats are good: 48 expert modules,
  `total_cpu_live_bytes=0`, `total_cpu_owned_bytes=0`, max staged HBM about
  `0.391 GiB`, CPU pool cache capped at `32 GiB`.

## Minimal Changes Needed

### P1. Make Run Identity and Dataset State Self-Contained

Why:

`command.txt` currently records the derived dataset name and `CUTOFF_LEN`, but
not all dataset-prep knobs or direct links to the dataset manifest/token stats.
This can recreate confusion about whether a run is really b4_s4096 or something
else.

Files/functions:

- `scripts/lf/profile_lora_lf.sh`
  - `run_job`
  - `ensure_jobs_tsv`
  - `append_job_record`
- `scripts/lf/run_lf_profiled_train.py`
  - `_config_from_args`

Concrete change:

- Add a small per-run `run_manifest.json` or equivalent `source_profile["run"]`
  section. Do not duplicate large data; link existing artifacts.
- Add these fields:
  - `artifact_role`: `acceptance_source_nohook`,
    `diagnostic_source_attribution`, or `diagnostic_nsys`
  - `expert_policy_tuple`: e.g. `none|true|false`
  - `profile_memory_attribution`, `profile_memory_breakdown`,
    `profile_memory_snapshot`, `profile_sync`, `profile_external_memory`
  - `dataset`, `prepare_datasets`, `dataset_min_tokens`,
    `dataset_eval_rows`, `dataset_overwrite`
  - `dataset_manifest`, `dataset_token_stats`, `dataset_validation` if present
  - `source_profile_json`, `profile_json`, `train_log`, `command_txt`
- Extend `jobs.tsv` only at the end with optional columns:
  `expact`, `attnact`, `layeract`, `artifact_role`, `dataset`,
  `run_manifest`.

Validation:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

RUN_NAME=profile_identity_dryrun \
DRY_RUN=true \
OVERWRITE=true \
PROFILERS=source \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYMM_EXP_ACT_POLICIES="none|true|false,gc-exp|false|false,none|false|false" \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
WARMUP_STEPS=5 \
MAX_STEPS=10 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
DATASET_MIN_TOKENS=4096 \
DATASET_OVERWRITE=true \
bash scripts/lf/profile_lora_lf.sh
```

Pass condition:

- dry-run artifacts show the exact policy tuple, hook mode, dataset prep values,
  and artifact role without reading shell history.

### P2. Surface Runtime Counts as JSON/CSV

Why:

The accepted no-hook run already prints the important count line in `train.log`,
for example `asym_forward_calls`, `asym_dx_calls`, `torch_forward_calls`,
`torch_dx_calls`, `reference_fallback_count`, and fallback reasons. But future
acceptance should not require manually grepping `train.log`. We must
machine-check that we did not create per-expert loops, torch fallback, reference
fallback, or a launch-count explosion.

Files/functions:

- `scripts/lf/run_lf_lora_sft.sh`
  - final AsymGEMM runtime verification block near the regex for
    `AsymGEMM LoRA-SFT runtime`
- `scripts/lf/run_lf_profiled_train.py`
  - `_activation_offload_counters_from_model`
  - `LFProfileRecorder.report`
- `scripts/lf/postprocess_lf_profile_artifacts.py`
  - `_write_source_artifacts`
  - `_write_profile_csv_artifacts`
  - new helper like `_runtime_counter_rows`

Concrete change:

- Parse the final runtime line from `train.log`.
- Write:
  - `runtime_counters.json`
  - `runtime_counters.csv`
- Embed the same data under `source_profile["runtime_counters"]` or
  `profile["runtime_counters"]`.
- For activation-offload diagnostic runs, aggregate per-module
  `execution_stats` into a top-level
  `activation_offload_execution_totals` section.
- Normalize counts by measured steps when possible.

Required fields:

```json
{
  "available": true,
  "source": "train_log",
  "asym_forward_calls": 5055,
  "asym_dx_calls": 4290,
  "torch_forward_calls": 0,
  "torch_dx_calls": 0,
  "reference_fallback_count": 0,
  "fallback_reasons": {},
  "router_no_grad": true,
  "expert_recompute_policy": "none",
  "router_mode": "whole",
  "measured_steps": 10,
  "asym_forward_calls_per_measured_step": 505.5,
  "asym_dx_calls_per_measured_step": 429.0
}
```

Validation:

```bash
RUN_NAME=profile_counts_b4s4096 \
OVERWRITE=true \
CONTINUE_ON_ERROR=false \
PROFILERS=source \
PROFILE_MEMORY_ATTRIBUTION=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_SNAPSHOT=false \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYMM_EXP_ACT_POLICIES="none|true|false,gc-exp|false|false,none|false|false" \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
WARMUP_STEPS=5 \
MAX_STEPS=10 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
DATASET_MIN_TOKENS=4096 \
DATASET_OVERWRITE=true \
bash scripts/lf/profile_lora_lf.sh
```

Pass condition:

- `runtime_counters.json/csv` exist for each run.
- Target `none|true|false` has positive AsymGEMM forward/dX counts.
- `torch_forward_calls == 0`, `torch_dx_calls == 0`,
  `reference_fallback_count == 0`.

### P3. Add One Compact Verdict and Diagnostic Links

Why:

The source sweep, memory attribution, snapshot analyzer, and nsys data should
stay separate, but the final comparison needs one small artifact that tells us
what to believe. This is mostly aggregation, not new profiling.

Files/functions:

- `scripts/lf/profile_lora_lf.sh`
  - after all jobs for a config root finish
- optional small new script:
  - `scripts/lf/write_lf_profile_verdict.py`
- `scripts/lf/postprocess_lf_profile_artifacts.py`
  - only if sharing readers/helpers is easier

Concrete change:

- Write config-level:
  - `profile_verdict.json`
  - `profile_verdict.md`
  - optionally `profile_verdict.csv`
- It should compare only matching acceptance runs:
  - target: `none|true|false`
  - baseline: `gc-exp|false|false`
  - reference no-offload: `none|false|false`
- It should include:
  - exact peak allocated/reserved
  - avg step/fwd/bwd
  - memory deltas
  - latency ratios
  - runtime counters
  - dataset/run identity
  - links to diagnostic snapshot and nsys artifacts when present
- It must state whether metrics are from:
  - `acceptance_source_nohook`
  - `diagnostic_source_attribution`
  - `diagnostic_nsys`

Keep/reject rules to encode:

- Keep only if peak HBM reduction is meaningful on the real b4_s4096 workload
  and latency does not blow up unjustifiably.
- Reject same-memory plus worse latency.
- Reject trivial memory saving plus large latency increase.
- Reject torch fallback, reference fallback, or unexpected launch-count growth.
- Do not use snapshot/attribution hook runs for latency acceptance.
- Do not describe inferred workspace rows as exact ownership.

Snapshot integration:

- Do not rebuild the analyzer; `scripts/testing/analyze_cuda_memory_snapshot.py`
  already exists.
- If `PROFILE_MEMORY_SNAPSHOT=true`, auto-run or auto-link analyzer output:
  `peak_snapshot_attrib_allblocks.json/md`.
- Warn if:
  - unknown frees are nonzero
  - replayed peak differs from source peak by more than about `0.25 GiB`
  - event count hits the configured max entries
  - snapshot is missing even though the flag is true

Nsys integration:

- A current b4_s4096 nsys target run is optional for memory attribution, but
  required before accepting kernel/latency optimization claims.
- Use source no-hook for memory/timing acceptance; use nsys for kernel timing,
  launch counts, and interconnect/C2C diagnosis.

Validation:

```bash
python -m json.tool <config_root>/profile_verdict.json >/dev/null
sed -n '1,160p' <config_root>/profile_verdict.md
```

Pass condition:

- The verdict contains all three policy rows with exact metrics.
- It clearly marks `none|true|false` as memory-saving but latency-heavy, not an
  unconditional win.
- It links diagnostic snapshot/nsys artifacts only as diagnostics.

## Final Commands

Acceptance source no-hook:

```bash
RUN_NAME=acceptance_b4s4096_source_nohook \
OVERWRITE=true \
CONTINUE_ON_ERROR=false \
PROFILERS=source \
PROFILE_MEMORY_ATTRIBUTION=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_SNAPSHOT=false \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYMM_EXP_ACT_POLICIES="none|true|false,gc-exp|false|false,none|false|false" \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
WARMUP_STEPS=5 \
MAX_STEPS=10 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
DATASET_MIN_TOKENS=4096 \
DATASET_OVERWRITE=true \
bash scripts/lf/profile_lora_lf.sh
```

Diagnostic source attribution/snapshot:

```bash
RUN_NAME=diagnostic_b4s4096_source_snapshot \
OVERWRITE=true \
CONTINUE_ON_ERROR=false \
PROFILERS=source \
PROFILE_MEMORY_ATTRIBUTION=true \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_MEMORY_SNAPSHOT=true \
PROFILE_MEMORY_SNAPSHOT_MAX_ENTRIES=1000000 \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYMM_EXP_ACT_POLICIES="none|true|false" \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
WARMUP_STEPS=5 \
MAX_STEPS=1 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
DATASET_MIN_TOKENS=4096 \
DATASET_OVERWRITE=false \
bash scripts/lf/profile_lora_lf.sh
```

Diagnostic nsys target:

```bash
RUN_NAME=diagnostic_b4s4096_nsys_target \
OVERWRITE=true \
CONTINUE_ON_ERROR=false \
PROFILERS=nsys \
PROFILE_MEMORY_ATTRIBUTION=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_SNAPSHOT=false \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYMM_EXP_ACT_POLICIES="none|true|false" \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
WARMUP_STEPS=5 \
MAX_STEPS=3 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
DATASET_MIN_TOKENS=4096 \
DATASET_OVERWRITE=false \
bash scripts/lf/profile_lora_lf.sh
```

