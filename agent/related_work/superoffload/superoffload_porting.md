> NOTE (2026-07-13): `scripts/lf/profile_lora_lf.sh` mentioned throughout was later renamed/split into `profile_lora_lf_test_{source,both}.sh`; path mentions below predate that split and are kept as written.

# SuperOffload Porting Plan

## Short Answer

Current production already has zero-policy backend labels:

```text
torch, zero2, zero3, zero3_offload, asym_torch, asym, kt_torchbf16, kt_armbf16
```

So port `superoffload` as a new zero-policy backend beside
`zero3_offload`, not as an AsymGEMM kernel backend. The correct comparison is:

```text
zero3_offload   = normal DeepSpeed ZeRO-3 CPU offload baseline
superoffload    = DeepSpeed ZeRO-3 CPU offload + offload_optimizer.super_offload=true
```

Use the local DeepSpeed repo:

```text
/home/shutianluo/kevin/AsymGEMM-SFT/third_party/deepspeed
```

and force it first on `PYTHONPATH` for SuperOffload launches.

## Non-Negotiable Design

| Area | Decision |
|---|---|
| Backend name | Canonical name is `superoffload` |
| Baseline to compare against | `zero3_offload`, not plain `torch` |
| Model path | Ordinary LF PEFT LoRA |
| AsymGEMM model rewrite | Disabled for SuperOffload |
| KT model rewrite | Disabled for SuperOffload |
| DeepSpeed stage | ZeRO stage 3 |
| Offload type | CPU parameter offload + CPU optimizer offload |
| Runtime proof | Must show `SuperOffloadOptimizer_Stage3` |

Do not add `superoffload` to:

```text
asym_gemm/training/frozen_linear.py::VALID_BACKENDS
asym_gemm/integrations/lf.py --asym_backend
asym_gemm/integrations/peft_lf.py backend validation
KT wrappers
scripts/lora/profile_lora_e2e.py
```

## Current Production Shape To Respect

`scripts/lf/run_lf_lora_sft.sh` currently maps zero-policy labels like this:

```text
zero2         -> BACKEND=torch, ZERO_BACKEND_LABEL=zero2
zero3         -> BACKEND=torch, ZERO_BACKEND_LABEL=zero3
zero3_offload -> BACKEND=torch, ZERO_BACKEND_LABEL=zero3_offload
```

Port `superoffload` into that same abstraction:

```text
superoffload -> BACKEND=torch, ZERO_BACKEND_LABEL=superoffload
```

This is cleaner than copying the `AsymGEMM-SO` script branch literally, because
production now already has a zero-policy backend layer.

## File-Level Porting Steps

| File | Needed changes |
|---|---|
| `scripts/lf/profile_lora_lf.sh` | Add `superoffload` backend parsing, scheduling, config rendering, dry-run recording, and comparison support |
| `scripts/lf/run_lf_lora_sft.sh` | Add `superoffload` as a zero-policy run label using the rendered SuperOffload DeepSpeed config |
| `scripts/lf/run_lf_profiled_train.py` | Record SuperOffload config/runtime proof in `source_profile.json` |
| `scripts/lf/render_superoffload_deepspeed_config.py` | New helper to render deterministic ZeRO-3 CPU-offload + SuperOffload config |
| `scripts/lf/check_superoffload_run.py` | New helper to fail if runtime did not select `SuperOffloadOptimizer_Stage3` |
| `tests/lf/test_superoffload_backend_scripts.py` | New focused tests for config rendering, checker, dry-run, single-GPU DeepSpeed launch, and recompute modes |

## `profile_lora_lf.sh`

Add defaults near the existing LF/KT defaults:

```bash
DEEPSPEED_DIR=${DEEPSPEED_DIR:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/deepspeed}
SUPER_OFFLOAD_DEEPSPEED_CONFIG=${SUPER_OFFLOAD_DEEPSPEED_CONFIG:-}
SUPER_OFFLOAD_CPUADAM_CORES_PERC=${SUPER_OFFLOAD_CPUADAM_CORES_PERC:-0.8}
CHECK_SUPEROFFLOAD=${CHECK_SUPEROFFLOAD:-true}
SUPER_OFFLOAD_CONFIG_RENDERER="${ASYM_DIR}/scripts/lf/render_superoffload_deepspeed_config.py"
```

Update backend handling:

1. `backend_label()` accepts:

```text
superoffload, super_offload, so, ds_superoffload
```

and canonicalizes all of them to:

```text
superoffload
```

2. `append_backend_spec()` accepts `superoffload|norecomp`,
   `superoffload|recomp`, and `superoffload|both`.

3. `backend_gpu_count()` treats `superoffload` like `zero3_offload`:

```bash
torch|zero2|zero3|zero3_offload|superoffload)
  printf '%s\n' "${model_gpu_count}"
  ;;
```

4. `is_zero_backend()` includes `superoffload`.

5. `is_policy_independent_backend()` includes `superoffload`.

6. `selected_has_zero=true` when `superoffload` is selected.

7. Default comparison candidates include `superoffload`, and the preferred
   baseline for this comparison is `zero3_offload`.

Render the config once per output root after `base_output_root` is known:

```bash
if selected_has_superoffload; then
  SUPER_OFFLOAD_DEEPSPEED_CONFIG="${SUPER_OFFLOAD_DEEPSPEED_CONFIG:-${base_output_root}/deepspeed/ds_z3_superoffload_config.json}"
  python3 "${SUPER_OFFLOAD_CONFIG_RENDERER}" \
    --base "${LF_DIR}/examples/deepspeed/ds_z3_offload_config.json" \
    --output "${SUPER_OFFLOAD_DEEPSPEED_CONFIG}" \
    --cpuadam-cores-perc "${SUPER_OFFLOAD_CPUADAM_CORES_PERC}"
fi
```

Validate before launching:

```text
DEEPSPEED_DIR/deepspeed/runtime/superoffload/superoffload_stage3.py exists
SUPER_OFFLOAD_CPUADAM_CORES_PERC is in [0.0, 1.0]
SUPER_OFFLOAD_DEEPSPEED_CONFIG exists after rendering
```

Pass these through `run_job()`:

```bash
DEEPSPEED_DIR
SUPER_OFFLOAD_DEEPSPEED_CONFIG
SUPER_OFFLOAD_CPUADAM_CORES_PERC
CHECK_SUPEROFFLOAD
ASYM_GEMM_LF_CONFIG_SUPEROFFLOAD_CONFIG
ASYM_GEMM_LF_CONFIG_SUPEROFFLOAD_CPUADAM_CORES_PERC
ASYM_GEMM_LF_CONFIG_DEEPSPEED_DIR
```

Dry-run fix:

Current production dry-run prints the command and returns before writing
`command.txt`/`jobs.tsv`. Port the SO dry-run behavior so dry-run still writes:

```text
<seq-root>/command.txt
<config-root>/jobs.tsv
```

This is needed for script tests and for reviewing generated SuperOffload
commands without launching GPUs.

## `run_lf_lora_sft.sh`

Add defaults near the existing backend config:

```bash
DEEPSPEED_DIR=${DEEPSPEED_DIR:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/deepspeed}
SUPER_OFFLOAD_DEEPSPEED_CONFIG=${SUPER_OFFLOAD_DEEPSPEED_CONFIG:-}
SUPER_OFFLOAD_CPUADAM_CORES_PERC=${SUPER_OFFLOAD_CPUADAM_CORES_PERC:-0.8}
CHECK_SUPEROFFLOAD=${CHECK_SUPEROFFLOAD:-true}
```

Add `superoffload` to the backend case using the current zero-policy pattern:

```bash
superoffload|super_offload|so|ds_superoffload)
  PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-superoffload}
  ZERO_BACKEND_LABEL=superoffload
  BACKEND=torch
  TORCH_DEEPSPEED_CONFIG="${SUPER_OFFLOAD_DEEPSPEED_CONFIG}"
  ;;
```

Add helpers:

```bash
is_superoffload_zero_run() {
  [[ "${ZERO_BACKEND_LABEL}" == "superoffload" ]]
}
```

Update `zero_deepspeed_config()` only if useful for validation. Since the
SuperOffload config is rendered dynamically, it can return
`${SUPER_OFFLOAD_DEEPSPEED_CONFIG}` for `superoffload`, or validation can handle
SuperOffload separately.

Validation additions:

```text
BACKEND=superoffload requires SUPER_OFFLOAD_DEEPSPEED_CONFIG
SUPER_OFFLOAD_DEEPSPEED_CONFIG must exist
DEEPSPEED_DIR must contain deepspeed/runtime/superoffload/superoffload_stage3.py
SUPER_OFFLOAD_CPUADAM_CORES_PERC must be in [0.0, 1.0]
```

DeepSpeed scope:

`assert_deepspeed_scope()` should still allow `--deepspeed` only for zero-policy
runs. Since SuperOffload uses `ZERO_BACKEND_LABEL=superoffload`, it remains
inside the zero-policy scope.

Command args:

```bash
if is_zero_backend_run; then
  CMD_ARGS+=(--pure_bf16 false)
  CMD_ARGS+=(--deepspeed "${TORCH_DEEPSPEED_CONFIG}")
else
  CMD_ARGS+=(--pure_bf16 true)
fi
```

SuperOffload must not add any of:

```text
--use_asym_gemm
--asym_backend
--use_kt
--kt_backend
```

That should already be true if SuperOffload maps to `BACKEND=torch` and
`TORCH_USE_ASYM_GEMM_LORA=false`.

Runtime env:

For SuperOffload, set local DeepSpeed first:

```bash
RUN_PYTHONPATH="${DEEPSPEED_DIR}:${ASYM_DIR}:${LF_DIR}/src:${PYTHONPATH:-}"
```

Export profile metadata:

```bash
ASYM_GEMM_LF_CONFIG_BACKEND="${PROFILE_BACKEND_LABEL:-superoffload}"
ASYM_GEMM_LF_CONFIG_SUPEROFFLOAD_CONFIG="${SUPER_OFFLOAD_DEEPSPEED_CONFIG}"
ASYM_GEMM_LF_CONFIG_SUPEROFFLOAD_CPUADAM_CORES_PERC="${SUPER_OFFLOAD_CPUADAM_CORES_PERC}"
ASYM_GEMM_LF_CONFIG_DEEPSPEED_DIR="${DEEPSPEED_DIR}"
```

Torchrun fallback:

Current production requires `${ENV_DIR}/bin/torchrun`. Keep that path when it
exists, but add the SO fallback for environments where only Python is present:

```bash
TORCHRUN_CMD=()
if [[ -x "${TORCHRUN_BIN}" ]]; then
  TORCHRUN_CMD=("${TORCHRUN_BIN}")
elif [[ -x "${ENV_PYTHON}" ]]; then
  TORCHRUN_CMD=("${ENV_PYTHON}" -m torch.distributed.run)
fi
```

Use `"${TORCHRUN_CMD[@]}"` in the launch command. This matters for 1-GPU
DeepSpeed because LF still requires a distributed launch for DeepSpeed.

After training, verify runtime:

```bash
if is_superoffload_zero_run && [[ "${CHECK_SUPEROFFLOAD}" == "true" ]]; then
  "${ENV_PYTHON}" "${ASYM_DIR}/scripts/lf/check_superoffload_run.py" \
    --profile-json "${PROFILE_SOURCE_JSON}" \
    --train-log "${LOG_FILE}" \
    --require-enabled
fi
```

Failure means the run did not actually use SuperOffload and should fail the
job.

## `run_lf_profiled_train.py`

Add SuperOffload fields to `_config_from_args()` from env:

```text
ASYM_GEMM_LF_CONFIG_SUPEROFFLOAD_CONFIG
ASYM_GEMM_LF_CONFIG_SUPEROFFLOAD_CPUADAM_CORES_PERC
ASYM_GEMM_LF_CONFIG_DEEPSPEED_DIR
```

Add a SuperOffload profile summary:

```json
{
  "superoffload": {
    "config_super_offload": true,
    "enabled": true,
    "runtime_verified": true,
    "optimizer_class": "SuperOffloadOptimizer_Stage3",
    "deepspeed_config": ".../ds_z3_superoffload_config.json",
    "deepspeed_dir": ".../third_party/deepspeed",
    "cpuadam_cores_perc": 0.8
  }
}
```

Runtime proof should come from the actual trainer/DeepSpeed engine optimizer
class when available. Config-only proof is not enough.

## Helper Scripts

### `render_superoffload_deepspeed_config.py`

Input base:

```text
${LF_DIR}/examples/deepspeed/ds_z3_offload_config.json
```

Output must force:

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true,
      "super_offload": true,
      "cpuadam_cores_perc": 0.8
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

Write stable sorted JSON with two-space indentation.

### `check_superoffload_run.py`

Pass if either:

```text
source_profile.json has superoffload.optimizer_class == SuperOffloadOptimizer_Stage3
train.log contains "DeepSpeed Final Optimizer = SuperOffloadOptimizer_Stage3"
```

With `--require-enabled`, missing proof exits with code `2`.

## Focused Tests

Add `tests/lf/test_superoffload_backend_scripts.py` with these cases:

| Test | Required assertion |
|---|---|
| Config renderer | Stage 3, CPU param offload, CPU optimizer offload, `super_offload=true`, requested `cpuadam_cores_perc` |
| Checker positive | Synthetic profile/log marker exits `0` |
| Checker negative | Missing marker with `--require-enabled` exits `2` |
| Dry-run SuperOffload | Writes `command.txt` and `jobs.tsv` |
| Dry-run command isolation | Command contains `BACKEND=superoffload` and `--deepspeed`, but no AsymGEMM/KT flags |
| Single-GPU DeepSpeed | `zero3_offload` and `superoffload` can launch with `NUM_GPUS=1` via one-rank torchrun |
| Recompute modes | `superoffload|both` emits both `recomp` and `norecomp` rows |
| Policy skipping | Non-`none` expert policies are skipped for SuperOffload |

## Validation Commands

Static/script validation:

```bash
bash -n scripts/lf/run_lf_lora_sft.sh scripts/lf/profile_lora_lf.sh
.venv/bin/python -m pytest -q tests/lf/test_superoffload_backend_scripts.py
git diff --check -- scripts/lf tests/lf asym_gemm
```

Dry-run validation:

```bash
BACKEND_SPECS='zero3_offload|norecomp,zero3_offload|recomp,superoffload|norecomp,superoffload|recomp' \
GPU_POOL='0' \
PROFILERS='source' \
SEQ_LENS='4096' \
MAX_STEPS='1' \
WARMUP_STEPS='0' \
PREPARE_DATASETS='false' \
DRY_RUN='true' \
PLOT='false' \
PLOT_MEMORY_BREAKDOWN='false' \
scripts/lf/profile_lora_lf.sh \
  --model-specs 'Qwen/Qwen3-30B-A3B|1' \
  --output-root /tmp/asymgemm_superoffload_dryrun
```

1-GPU live source profile:

```bash
BACKEND_SPECS='zero3_offload|norecomp,zero3_offload|recomp,superoffload|norecomp,superoffload|recomp' \
GPU_POOL='1' \
PROFILERS='source' \
SEQ_LENS='4096' \
MAX_STEPS='10' \
WARMUP_STEPS='5' \
PREPARE_DATASETS='true' \
MAX_SAMPLES='128' \
LORA_RANK='64' \
LORA_ALPHA='16' \
LORA_DROPOUT='0.00' \
CHECK_SUPEROFFLOAD='true' \
PLOT='false' \
PLOT_MEMORY_BREAKDOWN='false' \
scripts/lf/profile_lora_lf.sh \
  --model-specs 'Qwen/Qwen3-30B-A3B|1' \
  --output-root /tmp/asymgemm_superoffload_source
```

1-GPU live Nsight profile:

```bash
BACKEND_SPECS='zero3_offload|norecomp,zero3_offload|recomp,superoffload|norecomp,superoffload|recomp' \
GPU_POOL='1' \
PROFILERS='nsys' \
SEQ_LENS='4096' \
MAX_STEPS='10' \
WARMUP_STEPS='5' \
PREPARE_DATASETS='true' \
MAX_SAMPLES='128' \
LORA_RANK='64' \
LORA_ALPHA='16' \
LORA_DROPOUT='0.00' \
CHECK_SUPEROFFLOAD='true' \
PLOT='false' \
PLOT_MEMORY_BREAKDOWN='false' \
scripts/lf/profile_lora_lf.sh \
  --model-specs 'Qwen/Qwen3-30B-A3B|1' \
  --output-root /tmp/asymgemm_superoffload_nsys
```

## Acceptance Criteria

| Criterion | Required result |
|---|---|
| Normal baseline | `zero3_offload` uses `DeepSpeedZeroOptimizer_Stage3` |
| SuperOffload runtime | `superoffload` uses `SuperOffloadOptimizer_Stage3` |
| Config baseline | Both use ZeRO stage 3 CPU param offload and CPU optimizer offload |
| SuperOffload config | `offload_optimizer.super_offload=true` |
| Recompute coverage | Both `recomp` and `norecomp` run for `zero3_offload` and `superoffload` |
| Loss comparison | Passes for matching recompute mode |
| Source metrics | Step, forward, backward, peak HBM recorded |
| Nsight artifacts | `trace.nsys-rep`, `trace.sqlite`, `profile.json`, `source_profile.json` exist |
| Existing backends | `torch`, `zero2`, `zero3`, `zero3_offload`, `asym`, `asym_torch`, and KT behavior unchanged unless `superoffload` is selected |

