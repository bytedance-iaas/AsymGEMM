# CUDA Graph for AsymGEMM LF LoRA SFT

## Current conclusion

CUDA Graph should be treated as a partial latency optimization for the current
AsymGEMM LoRA SFT path, not as a full training-step capture yet.

Status after the 2026-06-17 Qwen3 e2e validation: `ASYM_CUDA_GRAPH=compile`
is implemented and launchable, but it is **not accepted as an effective mode**
for the profiled Qwen3 LoRA SFT workload. It reduced peak HBM in the source
profile, but measured latency regressed, especially p90. Keep the default at
`off`.

The implemented entry point is:

```bash
ASYM_CUDA_GRAPH=compile
```

This enables PyTorch compile with the Inductor `reduce-overhead` mode for
capturable regions. It is intentionally not a hand-written full
`torch.cuda.CUDAGraph` training loop.

Default remains:

```bash
ASYM_CUDA_GRAPH=off
```

With `off`, the launch path must remain behaviorally inert:

- no `--torch_compile` arguments are passed
- no CUDA graph or Inductor mode is enabled
- no graph prewarm tensors are allocated
- no `_cgcompile` output/run-id suffix is added
- `ASYM_CUDA_GRAPH` and `ASYM_GEMM_LF_CONFIG_ASYM_CUDA_GRAPH` are removed from
  the child training environment, even if the caller explicitly set
  `ASYM_CUDA_GRAPH=off`

## Validation status: 2026-06-17

Implemented code changes:

- `scripts/lf/profile_lora_lf.sh` and `scripts/lf/run_lf_lora_sft.sh` now wire
  `ASYM_CUDA_GRAPH=compile` through PyTorch compile/Inductor
  `reduce-overhead`, add `_cgcompile` run labels, record compile config fields,
  and keep `off` launch-inert.
- Compile runs disable source trace ranges with `ASYM_GEMM_LF_TRACE_RANGES=0`
  to avoid Dynamo graph breaks from NVTX/profile range hooks.
- `scripts/lf/run_lf_profiled_train.py` now writes compile-health and global
  Asym execution stats into `profile.json`.
- `scripts/lf/compare_cuda_graph_profiles.py` compares same-shape off/compile
  profiles and fails if latency does not improve or p90 regresses.
- `asym_gemm/training/profile_ranges.py`,
  `asym_gemm/profiling/lf_trace.py`, `asym_gemm/training/frozen_linear.py`,
  `asym_gemm/training/lora.py`, and `asym_gemm/training/qwen3_moe.py` were
  adjusted so compile mode avoids known profiler/Dynamo churn and dynamic
  routed-expert metadata recompiles.

Validation commands that passed:

```bash
.venv/bin/python -m py_compile \
  asym_gemm/training/qwen3_moe.py \
  asym_gemm/training/lora.py \
  asym_gemm/training/frozen_linear.py \
  asym_gemm/training/profile_ranges.py \
  asym_gemm/profiling/lf_trace.py \
  scripts/lf/run_lf_profiled_train.py \
  scripts/lf/compare_cuda_graph_profiles.py

bash -n scripts/lf/profile_lora_lf.sh scripts/lf/run_lf_lora_sft.sh

.venv/bin/python -m pytest -q \
  tests/lf/test_asym_cpu_adamw_args.py -k "cuda_graph" \
  tests/lf/test_compare_cuda_graph_profiles.py

.venv/bin/python -m pytest -q \
  tests/test_lf_memory_breakdown.py -k "asym_execution_stats or compile_health"
```

Qwen3 e2e A/B command shape:

```bash
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
ASYMM_EXP_ACT_POLICIES='none|false|false|false' \
GPU_POOL='0' \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=false \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
PREPARE_DATASETS=false \
LORA_DROPOUT=0.00 \
WARMUP_STEPS=5 \
MAX_STEPS=20 \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false \
ASYM_CUDA_GRAPH=off bash scripts/lf/profile_lora_lf.sh \
  --output-root profiling_both/cuda_graph_validation_fixed

MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
ASYMM_EXP_ACT_POLICIES='none|false|false|false' \
GPU_POOL='0' \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=false \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
PREPARE_DATASETS=false \
LORA_DROPOUT=0.00 \
WARMUP_STEPS=5 \
MAX_STEPS=20 \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false \
ASYM_CUDA_GRAPH=compile \
ASYM_CUDA_GRAPH_TORCH_LOGS=recompiles \
bash scripts/lf/profile_lora_lf.sh \
  --output-root profiling_both/cuda_graph_validation_patched
```

Comparator command:

```bash
.venv/bin/python scripts/lf/compare_cuda_graph_profiles.py \
  --baseline-profile profiling_both/cuda_graph_validation_fixed/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b4_s4096_w5_s20_r64_a16_drop000/asym_cpuadamwds__source__norecomp__polnone__routerwhole__expact0__attnact0__layeract0__loraafwdhbm__actrecomp0__xunpack0__gradofftrue__weightofffalse/b4_s4096/profile.json \
  --candidate-profile profiling_both/cuda_graph_validation_patched/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b4_s4096_w5_s20_r64_a16_drop000_cgcompile/asym_cpuadamwds__source__norecomp__polnone__routerwhole__expact0__attnact0__layeract0__loraafwdhbm__actrecomp0__xunpack0__gradofftrue__weightofffalse/b4_s4096/profile.json \
  --json-output profiling_both/cuda_graph_validation_patched/qwen3_cuda_graph_compare.json
```

Qwen3 e2e result:

| Mode | Samples | Step median | Step p90 | Fwd+Bwd median | Fwd+Bwd p90 | Fwd median | Bwd median | Opt median | Peak alloc | Peak resv | CPU RSS peak | Asym calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `off` | 20 | 7479.64 ms | 7646.61 ms | 4768.10 ms | 4960.60 ms | 1355.53 ms | 3417.15 ms | 2681.06 ms | 170.32 GiB | 179.84 GiB | 196.78 GiB | 15575 |
| `compile` | 20 | 7779.92 ms | 11691.10 ms | 5023.22 ms | 5541.72 ms | 1414.44 ms | 3581.17 ms | 2780.18 ms | 134.84 GiB | 138.20 GiB | 194.34 GiB | 15575 |

Decision:

- Comparator result: **FAIL**.
- Failure reasons: step median regressed by `+4.01%`, forward/backward median
  regressed by `+5.35%`, and step p90 regressed by `+52.89%`.
- HBM peak improved materially, but the project acceptance rule rejects a
  change that saves memory while increasing latency.
- Compile health sidecar recorded the requested compile config, but it did not
  prove zero post-warmup recompiles. The train log still showed empty
  CUDAGraph capture warnings, so the heavy AsymGEMM regions should not be
  claimed as effectively captured.
- Global Asym counters matched exactly: `asym_forward_calls=8425`,
  `asym_dx_calls=7150`, `calls_total=15575`, `torch_calls=0`,
  `reference_fallback_count=0`.

Other findings from validation:

- `BACKEND_SPECS=asym|norecomp` without CPUAdamW OOMed on the Qwen3
  b4/s4096 shape during backward because the GPU optimizer/trainable surface
  consumed too much HBM.
- `asym_cpuadamwds|norecomp` with
  `ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true` hit a shape mismatch
  (`0` vs `64`) in the no-activation-offload path. The valid Qwen3 CUDA-graph
  A/B path used `ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false`.
- Full Llama4 CUDA-graph A/B was not run after Qwen3 failed acceptance. Existing
  Llama4 b4/s4096 activation-offload source profiles already show about
  `67.89s` median step time and `~900 GiB` CPU RSS, while the default main
  Llama4 b8/s8192 profile attempt exited with status `137`. Do not claim
  Llama4 CUDA-graph benefit until a separate full off/compile A/B completes.

## Why partial graph is the right first step

AsymGEMM has many CPU-side control and launch costs, so CUDA Graph can reduce
latency by reducing CPU launch overhead and launch-side synchronization. It
does not remove the actual host-memory traffic inside AsymGEMM kernels.

Full training-step graph capture is not currently safe because the real SFT
path still contains graph blockers:

- grouped MoE routes still have host scalar/list reads in several fallback and
  metadata paths, for example `offsets.tolist()`, `experts.tolist()`, and
  `padded_offsets[-1].item()`
- activation offload uses saved-tensor hooks, pinned CPU copies,
  `torch.cuda.Event`, and `event.synchronize()`
- CPU AdamW / DeepSpeed CPU AdamW is not capturable; the optimizer stage can
  still dominate end-to-end step time even if forward/backward improves
- dropout must be disabled for stable graph behavior
- gradient checkpointing/recompute changes the captured region and is blocked
  for compile mode
- Qwen3 and Llama4 routed-expert paths can produce dynamic route metadata; any
  deeper capture needs fixed shapes and graph-stable routing metadata

## Implemented code groups

### Group 0: off-path no-op safety

Status: implemented.

Purpose: make the default path exactly the old training path for latency and
memory.

Changes:

- `ASYM_CUDA_GRAPH` defaults to `off`
- `off`, `false`, `0`, `none`, `no`, and `n` normalize to `off`
- compile-only env/config variables are not passed when off
- the child process is launched with:

```bash
env -u ASYM_CUDA_GRAPH -u ASYM_GEMM_LF_CONFIG_ASYM_CUDA_GRAPH ...
```

Validation required:

- `ASYM_CUDA_GRAPH=off` does not add `--torch_compile`
- child LF env has blank/unset `ASYM_CUDA_GRAPH`
- child LF env has blank/unset `ASYM_GEMM_LF_CONFIG_ASYM_CUDA_GRAPH`
- output paths have no `_cgcompile` suffix
- no profile JSON field `asym_cuda_graph` is written for off runs

Current lightweight validation:

```bash
pytest \
  tests/lf/test_asym_cpu_adamw_args.py::test_run_lf_lora_sft_cuda_graph_off_is_launch_inert \
  -q
```

Acceptance:

- no training-time Python, CPU, HBM, CUDA allocator, or graph-pool overhead is
  attributable to the CUDA graph feature when off
- only shell argument normalization runs before launching training

### Group 1: compile-mode launch

Status: implemented.

Purpose: enable partial CUDA graph capture through PyTorch compile for regions
that are already capturable.

Compile mode adds:

```bash
--torch_compile true
--torch_compile_backend inductor
--torch_compile_mode reduce-overhead
```

Guardrails:

- direct `run_lf_lora_sft.sh` backend must normalize to AsymGEMM:
  raw `BACKEND=asym`, `asym_torch`, `asym_cpuadamwtorch`, or
  `asym_cpuadamwds`
- profile `BACKEND_SPECS` may include only AsymGEMM backends:
  `asym`, `asym_torch`, `asym_cpuadamwtorch`, or `asym_cpuadamwds`
- `GRADIENT_CHECKPOINTING=false`
- direct `run_lf_lora_sft.sh` requires `LORA_DROPOUT` to be numerically zero
- profile sweeps require the fixed label `LORA_DROPOUT=0.00`
- `ASYMM_EXPERT_ACT_OFFLOAD=false`
- `ASYMM_ATTN_ACT_OFFLOAD=false`
- `ASYMM_LAYER_ACT_OFFLOAD=false`
- profile sweeps must use `norecomp`
- profile sweeps cannot mix non-Asym baselines in the same
  `ASYM_CUDA_GRAPH=compile` invocation

Validation required:

- compile flags appear only when `ASYM_CUDA_GRAPH=compile`
- bad combinations fail before training starts
- generated run IDs and profile roots include `_cgcompile`
- profile config records `asym_cuda_graph`, `torch_compile`,
  `torch_compile_backend`, and `torch_compile_mode` only for compile runs

Current lightweight validation:

```bash
pytest \
  tests/lf/test_asym_cpu_adamw_args.py::test_run_lf_lora_sft_asym_cuda_graph_compile_args_and_env \
  tests/lf/test_asym_cpu_adamw_args.py::test_run_lf_lora_sft_cuda_graph_rejects_activation_offload \
  tests/lf/test_asym_cpu_adamw_args.py::test_profile_lora_lf_dry_run_asym_cuda_graph_compile \
  -q
```

### Group 2: dense AsymGEMM graph prewarm

Status: implemented as a helper, not wired into full LF training.

Purpose: avoid allocating single-group launch tensors inside manual CUDA graph
capture for dense `AsymFrozenLinear` paths.

API:

```python
from asym_gemm.training import initialize_asym_cuda_graph_state

initialize_asym_cuda_graph_state("cuda:0", rows=batch_size * seq_len)
```

This only covers the direct dense/single-group metadata path. It does not make
grouped MoE routing graph-safe.

Validation required:

- repeated calls return cached tensors for the same device and row count
- a capture-time allocation is not attempted after prewarm
- invalid or empty row counts fail early

Current lightweight validation:

```bash
pytest \
  tests/training/test_cpu_resident_frozen_base.py::test_initialize_asym_cuda_graph_state_caches_single_group_tensors_cpu \
  -q
```

## Required metric packet

The CUDA graph question must be answered from real
`scripts/lf/profile_lora_lf.sh` artifacts, not from wrapper tests. Each
candidate comparison needs the raw `profile.json`, `source_profile.json`,
`summary.md`, and the parent `jobs.tsv`. Nsight runs also need the trace
artifacts under the same run directory.

The report must answer three questions:

1. Is compile mode faster after warmup?
2. Is the speedup in the intended model/AsymGEMM scope rather than dataloader,
   optimizer, or logging noise?
3. Is steady-state CPU RSS and GPU HBM materially unchanged?

### Required comparison matrix

Run this matrix for both Qwen3 and Llama4:

- graph A/B, realistic end-to-end:
  `BACKEND_SPECS=asym_cpuadamwds|norecomp`,
  `ASYM_CUDA_GRAPH=off` versus `ASYM_CUDA_GRAPH=compile`
- graph A/B, model-region sensitivity:
  `BACKEND_SPECS=asym|norecomp`,
  `ASYM_CUDA_GRAPH=off` versus `ASYM_CUDA_GRAPH=compile`
- baseline positioning:
  run the existing non-Asym baselines from the normal sweep separately, for
  example the current `zero3_offload|recomp` baseline, with CUDA graph off

Do not put non-Asym baselines in the same `ASYM_CUDA_GRAPH=compile` sweep.
Compile mode is only supported for Asym backends.

### Same-shape proof

Before reading timing deltas, prove that the off and compile artifacts are the
same workload. Compare these `profile.json` fields:

- `config.model_name_or_path`
- `config.backend`
- `config.precision`
- `config.dataset`
- `config.template`
- `config.seq_len`
- `config.cutoff_len`
- `config.logical_qlen`
- `config.per_device_train_batch_size`
- `config.gradient_accumulation_steps`
- `config.lora_rank`
- `config.lora_alpha`
- `config.lora_dropout`
- `config.qwen_moe_expert_lora_impl`
- `config.activation_recompute`
- `config.asymm_expert_act_offload`
- `config.asymm_attn_act_offload`
- `config.asymm_layer_act_offload`
- `config.expert_policy_label`
- `config.profile_sync`
- `config.warmup_steps`
- `config.measure_steps`

For off runs, `config.asym_cuda_graph`, `config.torch_compile`,
`config.torch_compile_backend`, and `config.torch_compile_mode` must be absent.
For compile runs, they must be:

```json
{
  "asym_cuda_graph": "compile",
  "torch_compile": true,
  "torch_compile_backend": "inductor",
  "torch_compile_mode": "reduce-overhead"
}
```

### Efficiency metrics from source profiles

Use only non-warmup rows:

```python
rows = [r for r in profile["step_samples"]["rows"] if not r["is_warmup"]]
```

Required timing numbers:

- median and p90 of `rows[*].step_milliseconds`
- median and p90 of `rows[*].trainer_e2e_step_milliseconds`
- median and p90 of `rows[*].forward_backward_milliseconds`
- median and p90 of `rows[*].forward_milliseconds`
- median and p90 of `rows[*].backward_milliseconds`
- median and p90 of `rows[*].profiled_training_step_milliseconds`
- median and p90 of `rows[*].optimizer_update_side_milliseconds`
- median of `rows[*].heartbeat_dataloader_fetch_milliseconds`, when present
- median of `rows[*].heartbeat_training_step_milliseconds`, when present
- median of `rows[*].heartbeat_optimizer_step_milliseconds`, when present
- `trainer.timing.measured_e2e_step_milliseconds`
- `step.total_milliseconds`
- `step.rows[name=="step.forward"].milliseconds`
- `step.rows[name=="step.backward"].milliseconds`
- `step.rows[name=="lf.step.total"].milliseconds`
- `step.rows[name=="lf.optimizer.step"].milliseconds`, when present
- `step.rows[name=="lf.grad_clip"].milliseconds`, when present
- `step.rows[name=="lf.inputs.prepare"].milliseconds`, when present
- `step.rows[name=="lf.data.next"].milliseconds`, when present

Derived numbers:

- end-to-end speedup:
  `(off_median_step_ms - compile_median_step_ms) / off_median_step_ms`
- model-region speedup:
  `(off_median_forward_backward_ms - compile_median_forward_backward_ms) /
  off_median_forward_backward_ms`
- optimizer masking:
  `median(optimizer_update_side_milliseconds) / median(step_milliseconds)`
- scope ratio:
  `model_region_speedup / end_to_end_speedup`

Interpretation:

- if end-to-end improves but forward/backward does not, the run is noise or an
  unrelated scheduling change, not a CUDA graph win
- if forward/backward improves but end-to-end does not, CPUAdamW or dataloader
  time is masking the benefit; report it as a model-region win only
- if p90 regresses while median improves, do not call it effective until the
  long-tail cause is identified

### Scope proof from source and Nsight

Source `profile.json` can prove stage-level scope. The gain should appear in
`step.forward`, `step.backward`, or `forward_backward_milliseconds`. It should
not be credited to CUDA graph if the only improvement is in:

- `lf.data.next`
- `lf.inputs.prepare`
- `lf.optimizer.step`
- `lf.scheduler.step`
- `lf.log_save_eval`
- `heartbeat_dataloader_fetch_milliseconds`
- `heartbeat_optimizer_step_milliseconds`

Nsight is required for module/kernel scope. Run with:

```bash
PROFILERS=nsys \
PROFILE_LEVEL=op \
PROFILE_MODULE_FILTER=attention,mlp,experts,lora,optimizer \
PROFILE_SYNC=0 \
scripts/lf/profile_lora_lf.sh
```

Nsight acceptance:

- smaller CPU launch gaps inside `lf.forward_loss`, `lf.backward`,
  `forward.layers.*.mlp`, `forward.layers.*.mlp.experts`,
  `backward.layers.*.mlp`, or `backward.layers.*.mlp.experts`
- fewer CPU-launched CUDA operations in the Asym-heavy forward/backward ranges
- no new synchronization boundary after warmup
- no repeated graph recapture or Inductor recompilation in measured steps
- optimizer and dataloader ranges do not explain the measured speedup

Current source-profile limitation: the no-activation-offload path does not
write one global AsymGEMM execution-counter block. Some modules expose
`stats.as_dict()` internally, and activation-offload reports may include
per-module `execution_stats`, but this is not enough for source-only scope
proof when activation offload is disabled. Add a global counter export before
claiming source-only module scope. The required fields are:

- `asym_forward_calls`
- `asym_dx_calls`
- `asym_calls`
- `staged_forward_calls`
- `staged_dx_calls`
- `staged_calls`
- `torch_forward_calls`
- `torch_dx_calls`
- `torch_calls`
- `reference_fallback_count`
- `fallback_reasons`
- `forward_calls_total`
- `backward_calls_total`
- `calls_total`

### Memory metrics from source profiles

Latency validation and memory validation can use the same source profile if
`PROFILE_MEMORY_BREAKDOWN=false`. If memory attribution is needed, run a
separate pass with `PROFILE_MEMORY_BREAKDOWN=true`; saved-tensor hooks perturb
timing, so do not use that pass for latency acceptance.

Required GPU HBM numbers:

- `memory.gpu.peak_allocated_hbm_bytes`
- `memory.gpu.peak_reserved_hbm_bytes`
- `memory.gpu.reserved_unallocated_bytes`
- `stage_memory.max_stage_peak_allocated_bytes`
- `stage_memory.max_stage_peak_reserved_bytes`
- max over non-warmup `step_samples.rows[*].peak_allocated_hbm_bytes`
- max over non-warmup `step_samples.rows[*].peak_reserved_hbm_bytes`
- max over `stage_memory.rows[*].max_peak_allocated_bytes`
- max over `stage_memory.rows[*].max_peak_reserved_bytes`

Required CPU memory numbers:

- `memory.process.rss_bytes`
- `memory.process.rss_peak_bytes`
- max over non-warmup `step_samples.rows[*].process_rss_end_bytes`
- max over non-warmup `step_samples.rows[*].process_rss_peak_bytes`
- max over `stage_memory.rows[*].max_process_rss_end_bytes`
- max over `stage_memory.rows[*].max_process_rss_peak_end_bytes`
- `optimizer_memory.process_memory_at_start.rss_bytes`, when present
- `optimizer_memory.process_memory_before_step.rss_bytes`, when present
- `optimizer_memory.process_memory_after_step.rss_bytes`, when present
- `optimizer_memory.process_rss_delta_bytes`, when present

Memory-breakdown pass, if run:

- `memory_breakdown.summary.peak_allocated_hbm_bytes`
- `memory_breakdown.summary.peak_reserved_hbm_bytes`
- `memory_breakdown.summary.breakdown_rows`
- `memory_breakdown.summary.actual_peak_breakdown_rows`
- `memory_attribution.rows`
- `memory_attribution.saved_tensors.by_owner`

### Minimal extractor for a pair of profiles

Until a checked-in compare script exists, use this style of extractor for every
off/compile pair:

```bash
python - off/profile.json compile/profile.json <<'PY'
import json
import statistics
import sys

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def rows(profile):
    return [r for r in profile["step_samples"]["rows"] if not r.get("is_warmup")]

def values(rows, key):
    return [float(r[key]) for r in rows if r.get(key) is not None]

def median(rows, key):
    vals = values(rows, key)
    return statistics.median(vals) if vals else None

def p90(rows, key):
    vals = sorted(values(rows, key))
    if not vals:
        return None
    index = min(len(vals) - 1, int(round(0.90 * (len(vals) - 1))))
    return vals[index]

def stage_ms(profile, name):
    for row in profile.get("step", {}).get("rows", []):
        if row.get("name") == name:
            return row.get("milliseconds")
    return None

def peak(rows, key):
    vals = values(rows, key)
    return max(vals) if vals else None

def metrics(profile):
    r = rows(profile)
    return {
        "config": profile.get("config", {}),
        "median_step_ms": median(r, "step_milliseconds"),
        "p90_step_ms": p90(r, "step_milliseconds"),
        "median_forward_backward_ms": median(r, "forward_backward_milliseconds"),
        "p90_forward_backward_ms": p90(r, "forward_backward_milliseconds"),
        "median_forward_ms": median(r, "forward_milliseconds"),
        "median_backward_ms": median(r, "backward_milliseconds"),
        "median_optimizer_side_ms": median(r, "optimizer_update_side_milliseconds"),
        "trainer_measured_e2e_ms": profile.get("trainer", {}).get("timing", {}).get(
            "measured_e2e_step_milliseconds"
        ),
        "stage_forward_ms": stage_ms(profile, "step.forward"),
        "stage_backward_ms": stage_ms(profile, "step.backward"),
        "stage_optimizer_ms": stage_ms(profile, "lf.optimizer.step"),
        "peak_step_allocated_hbm": peak(r, "peak_allocated_hbm_bytes"),
        "peak_step_reserved_hbm": peak(r, "peak_reserved_hbm_bytes"),
        "peak_step_rss": peak(r, "process_rss_peak_bytes"),
        "memory_peak_allocated_hbm": profile.get("memory", {}).get("gpu", {}).get(
            "peak_allocated_hbm_bytes"
        ),
        "memory_peak_reserved_hbm": profile.get("memory", {}).get("gpu", {}).get(
            "peak_reserved_hbm_bytes"
        ),
        "process_rss_peak": profile.get("memory", {}).get("process", {}).get("rss_peak_bytes"),
    }

off = metrics(load(sys.argv[1]))
compile_ = metrics(load(sys.argv[2]))

def delta(name):
    a = off.get(name)
    b = compile_.get(name)
    if a in (None, 0) or b is None:
        return None
    return (b - a) / a

report = {
    "off": {k: v for k, v in off.items() if k != "config"},
    "compile": {k: v for k, v in compile_.items() if k != "config"},
    "relative_deltas_compile_vs_off": {
        key: delta(key)
        for key in (
            "median_step_ms",
            "p90_step_ms",
            "median_forward_backward_ms",
            "p90_forward_backward_ms",
            "peak_step_allocated_hbm",
            "peak_step_reserved_hbm",
            "peak_step_rss",
            "memory_peak_allocated_hbm",
            "memory_peak_reserved_hbm",
            "process_rss_peak",
        )
    },
}
print(json.dumps(report, indent=2, sort_keys=True))
PY
```

## Remaining fixes before claiming real benefit

### Fix A: compile health evidence in profile artifacts

Status: implemented, but current counters are still conservative.

Problem: `torch_compile=true` in config proves the flag was passed, but it does
not prove that the expensive model regions were actually compiled, that CUDA
Graphs were used, or that graph breaks/recompiles stopped after warmup.

Implemented evidence collection for compile runs:

- `TORCH_LOGS` is set by `ASYM_CUDA_GRAPH_TORCH_LOGS`
- a concise `compile_health.json` sidecar is written in the run output
  directory and merged into `profile.json`
- graph-break/recompile counters are explicit and remain `None` when the local
  PyTorch runtime does not expose a robust counter
- first-step compile timing is represented as an explicit nullable field

Validation gate:

- compile run has no recompiles after warmup
- graph breaks are either absent from the target modules or explained and
  outside the claimed optimized region
- first-step compile cost is excluded from latency acceptance

### Fix B: profile completeness must distinguish graph mode

Status: implemented.

Profile output paths already include `_cgcompile`, so off and compile runs do
not normally collide. `existing_profile_complete` now validates graph-mode
config fields and rejects stale off-mode JSON in compile paths.

Validation gate:

- `COLLECT_EXISTING=true ASYM_CUDA_GRAPH=compile` rejects stale or off-mode
  profile JSON even if it is manually copied into a compile output path
- off-mode completeness remains unchanged

### Fix C: real fixed-shape workload validation

Status: required.

Toy tests are not enough. Qwen3 and Llama4 have different routing, module
wrapping, and activation shapes. Validate both with the LF SFT path.

Required workload pair:

- Qwen3: `Qwen/Qwen3-30B-A3B`
- Llama4: `meta-llama/Llama-4-Scout-17B-16E`

Required common config:

- single GPU first
- same GPU ID for off and compile
- same model revision/cache
- same dataset file
- same seed
- same `CUTOFF_LEN`
- same `PER_DEVICE_TRAIN_BATCH_SIZE`
- same `GRADIENT_ACCUMULATION_STEPS`
- same `LORA_RANK` and `LORA_ALPHA`
- `LORA_DROPOUT=0.00`
- `BACKEND_SPECS=asym_cpuadamwds|norecomp` for end-to-end realism
- also run `BACKEND_SPECS=asym|norecomp` if optimizer CPU time hides model
  latency gains
- `ASYMM_EXP_ACT_POLICIES=none|false|false|false`
- `ASYM_CUDA_GRAPH=off` and `ASYM_CUDA_GRAPH=compile` in separate runs
- `WARMUP_STEPS >= 5`
- `MAX_STEPS >= 20` for timing, preferably 50 if wall time allows

Example profile commands:

```bash
# Qwen3 baseline
ASYM_CUDA_GRAPH=off \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
ASYMM_EXP_ACT_POLICIES='none|false|false|false' \
LORA_DROPOUT=0.00 \
WARMUP_STEPS=5 \
MAX_STEPS=20 \
PROFILERS=source \
PROFILE_SYNC=1 \
scripts/lf/profile_lora_lf.sh

# Qwen3 compile
ASYM_CUDA_GRAPH=compile \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
ASYMM_EXP_ACT_POLICIES='none|false|false|false' \
LORA_DROPOUT=0.00 \
WARMUP_STEPS=5 \
MAX_STEPS=20 \
PROFILERS=source \
PROFILE_SYNC=1 \
scripts/lf/profile_lora_lf.sh

# Llama4 baseline
ASYM_CUDA_GRAPH=off \
MODEL_SPECS='meta-llama/Llama-4-Scout-17B-16E|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
ASYMM_EXP_ACT_POLICIES='none|false|false|false' \
LORA_DROPOUT=0.00 \
WARMUP_STEPS=5 \
MAX_STEPS=20 \
PROFILERS=source \
PROFILE_SYNC=1 \
scripts/lf/profile_lora_lf.sh

# Llama4 compile
ASYM_CUDA_GRAPH=compile \
MODEL_SPECS='meta-llama/Llama-4-Scout-17B-16E|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
ASYMM_EXP_ACT_POLICIES='none|false|false|false' \
LORA_DROPOUT=0.00 \
WARMUP_STEPS=5 \
MAX_STEPS=20 \
PROFILERS=source \
PROFILE_SYNC=1 \
scripts/lf/profile_lora_lf.sh
```

Validation gate:

- both models finish training and produce complete `profile.json`
- source stage metrics show any speedup in forward/backward or Nsight proves it
  in Asym-heavy module/kernel ranges
- LoRA trainable-surface checks still pass
- loss is finite for every measured step
- compile run has no post-warmup recompiles
- measured step samples exclude warmup

### Fix D: real latency proof

Status: required.

Measure both stage-level and end-to-end latency using the metric packet above.
The minimum report must include median and p90 deltas for:

- non-warmup `step_samples.rows[*].step_milliseconds`
- non-warmup `step_samples.rows[*].forward_backward_milliseconds`
- non-warmup `step_samples.rows[*].forward_milliseconds`
- non-warmup `step_samples.rows[*].backward_milliseconds`
- non-warmup `step_samples.rows[*].optimizer_update_side_milliseconds`
- `trainer.timing.measured_e2e_step_milliseconds`
- `step.rows[name=="step.forward"].milliseconds`
- `step.rows[name=="step.backward"].milliseconds`
- `step.rows[name=="lf.optimizer.step"].milliseconds`, when present

Nsight must be used for per-op/module scope:

- CPU launch gaps inside Asym-heavy forward/backward ranges
- CUDA kernel launch count per measured step
- Python/CPU time in the training loop
- visible graph break, recapture, or synchronization points

Acceptance target:

- compile mode improves median measured step latency by at least 5% on at
  least one real model workload, or improves forward/backward combined latency
  by at least 8% if CPUAdamW masks end-to-end gains
- p90 measured step latency does not regress
- no new long-tail stalls after warmup

Nsight validation:

```bash
ASYM_CUDA_GRAPH=off \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
ASYMM_EXP_ACT_POLICIES='none|false|false|false' \
PROFILERS=nsys \
WARMUP_STEPS=5 \
MAX_STEPS=10 \
scripts/lf/profile_lora_lf.sh

ASYM_CUDA_GRAPH=compile \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
ASYMM_EXP_ACT_POLICIES='none|false|false|false' \
PROFILERS=nsys \
WARMUP_STEPS=5 \
MAX_STEPS=10 \
scripts/lf/profile_lora_lf.sh
```

Nsight acceptance:

- fewer CPU-launched CUDA operations or smaller CPU launch gaps in the
  compiled region
- no new synchronization boundary in the steady-state measured steps
- no visible graph recapture/recompile in measured steps

### Fix E: real memory neutrality proof

Status: required.

Do not assume CUDA Graph is memory-neutral. PyTorch compile and CUDA graph pools
can increase CPU RSS, reserved HBM, or code/cache memory. The desired outcome is
latency improvement without material memory growth; this must be measured.

Use the exact GPU and CPU memory fields listed in the metric packet above.
The minimum report must include:

- top-level HBM allocated/reserved peak deltas
- non-warmup per-step HBM allocated/reserved peak deltas
- stage-memory HBM allocated/reserved peak deltas
- top-level process RSS/RSS-peak deltas
- non-warmup per-step process RSS/RSS-peak deltas
- optimizer memory RSS fields when CPUAdamW is used
- memory-breakdown summaries only from a separate memory pass, not from the
  latency pass

Acceptance target:

- peak allocated HBM: compile <= off + 1% or <= off + 256 MiB, whichever is
  larger
- peak reserved HBM: compile <= off + 3% or <= off + 512 MiB, whichever is
  larger
- steady-state CPU RSS after warmup: compile <= off + 2% or <= off + 1 GiB,
  whichever is larger
- first-step compile RSS may be higher, but must be reported separately and
  must not persist into measured steady-state steps

If compile mode improves latency but violates these memory thresholds, do not
claim success. Record it as a latency/memory tradeoff.

### Fix F: compare script or report

Status: implemented.

The profile artifacts already contain most raw data, but a small comparison
script would make acceptance objective.

The checked-in script takes:

```bash
--baseline-profile path/to/off/profile.json
--candidate-profile path/to/compile/profile.json
```

It should print:

- median/p90 measured step latency delta
- forward/backward latency deltas
- HBM allocated/reserved peak deltas
- process RSS peak delta
- compile config fields
- pass/fail against the thresholds above

Validation gate:

- script exits nonzero if latency does not improve or memory thresholds fail
- script can compare both source and materialized-nsys profile JSON

### Fix G: global AsymGEMM execution counters

Status: implemented for LF source profiles.

Problem addressed: no-activation-offload source profiles previously did not
guarantee a single global `AsymExecutionStats` block. Compile mode disables
activation offload, so source `profile.json` needed a global counter block to
prove that the same number of AsymGEMM/LoRA operations ran in off and compile
mode.

The profile now includes:

```json
{
  "asym_execution_stats": {
    "available": true,
    "asym_forward_calls": 0,
    "asym_dx_calls": 0,
    "asym_calls": 0,
    "staged_forward_calls": 0,
    "staged_dx_calls": 0,
    "staged_calls": 0,
    "torch_forward_calls": 0,
    "torch_dx_calls": 0,
    "torch_calls": 0,
    "reference_fallback_count": 0,
    "fallback_reasons": {},
    "forward_calls_total": 0,
    "backward_calls_total": 0,
    "calls_total": 0
  }
}
```

Validation gate:

- off and compile runs have nonzero `asym_calls`
- off and compile runs have comparable `forward_calls_total`,
  `backward_calls_total`, and `calls_total`
- `torch_calls` and `reference_fallback_count` do not increase in compile mode
- fallback reasons are absent or identical between off and compile
- if counters disagree, do not interpret timing deltas as CUDA graph efficiency

## Deferred deeper CUDA Graph work

Manual CUDA graph capture should only happen after compile mode proves a real
benefit on Qwen3 and Llama4. If compile mode does not move real latency, manual
capture is unlikely to be worth the implementation risk unless Nsight shows
large remaining CPU launch gaps.

Potential deeper work:

1. Make grouped MoE route metadata graph-stable.
   - remove CPU `.item()` and `.tolist()` from hot grouped paths
   - preallocate padded buffers for fixed maximum sequence rows
   - keep route offsets/experts on GPU
   - avoid Python shape allocation from GPU scalars inside capture

2. Add real CUDA graph smoke tests for direct AsymGEMM kernels.
   - gated by `ASYM_GEMM_TEST_CUDA_GRAPH=1`
   - run BF16 direct forward/backward on SM90/SM100
   - compare graph replay output with eager output

3. Investigate optimizer capture only for a GPU/capturable optimizer.
   - CPUAdamW should stay outside graph
   - end-to-end full-step graph is not compatible with current CPU optimizer
     and offload design

4. Consider module-level manual graph only for dense non-MoE AsymFrozenLinear.
   - useful as a correctness proof
   - not sufficient to claim Qwen3/Llama4 SFT benefit

## Final acceptance checklist

CUDA graph support can be called effective only when all of these pass:

- off path is launch-inert and has no runtime/memory overhead
- Qwen3 off and compile runs both complete with identical workload config
- Llama4 off and compile runs both complete with identical workload config
- compile run records compile health and has no post-warmup recompiles
- median steady-state step latency improves or forward/backward latency
  improves enough to matter
- p90 latency does not regress
- peak allocated HBM stays within threshold
- peak reserved HBM stays within threshold
- steady-state CPU RSS stays within threshold
- Nsight confirms reduced launch overhead or reduced CPU gaps in measured
  steps
- loss remains finite
- global AsymGEMM counters remain valid and comparable; Nsight is still needed
  before claiming exact per-kernel launch-scope improvements

Until those real workload validations pass, the correct claim is:

> `ASYM_CUDA_GRAPH=compile` is available as an experimental partial graph mode.
> The 2026-06-17 Qwen3 e2e profile rejected it for the tested workload because
> latency regressed despite lower HBM. Do not enable it by default or claim it
> effective until a later off/compile A/B passes the acceptance gate.
