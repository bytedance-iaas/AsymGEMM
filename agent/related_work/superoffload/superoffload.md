# AsymGEMM-SO SuperOffload LF Backend Integration Plan

## Goal

Create `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO` as an isolated copy of the current `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM` tree, then add `superoffload` as a selectable LlamaFactory profiling and training backend inside that copy beside the existing LF baseline labels `torch`, `asym`, `kt_torchbf16`, and `kt_armbf16`.

`AsymGEMM-SO` is the SuperOffload compatibility and baseline entry point for the eval stack. Production `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM` remains unchanged except for this planning document. The SuperOffload backend must use ordinary LlamaFactory PEFT LoRA plus DeepSpeed ZeRO-3 SuperOffload. It must not be added to AsymGEMM frozen-linear dispatch, `asym_backend`, or the AsymGEMM kernel registry. This plan is a planning artifact only; it does not include source implementation.

## Current Facts From Code Inspection

- `asym_gemm/training/frozen_linear.py` defines `VALID_BACKENDS = ("asym", "torch")`. `_check_backend()` rejects every other value, while `_dispatch_nt()` and `_dispatch_grouped_nt()` treat every non-`torch` backend as an AsymGEMM kernel path. Adding `superoffload` here would route SuperOffload requests into AsymGEMM kernels, which is incorrect.
- The current source tree resolves to `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM`.
- The target SuperOffload tree is `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO`.
- `setup.py` declares the Python package name as `asym_gemm` and the native extension name as `asym_gemm._C`. Keeping the package name unchanged preserves imports, function names, tests, and LF integration paths.
- `scripts/lf/profile_lora_lf.sh` defaults `ROOT` to `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM`.
- `scripts/lf/run_lf_lora_sft.sh` defaults `ROOT` to `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM`.
- `asym_gemm/training/frozen_linear.py` already has execution counters for `asym`, `torch`, and `kt`; these counters describe fused frozen-linear and LoRA paths, not DeepSpeed optimizer offload.
- `asym_gemm/integrations/lf.py` exposes `apply_lf_asym_lora(... backend: Literal["asym", "torch"], ...)` and validates `backend in {"asym", "torch"}`. This integration is for AsymGEMM model rewrites inside LF.
- `asym_gemm/integrations/peft_lf.py` forwards `adapt_lf_asym_peft_lora(... backend: Literal["asym", "torch"], ...)` into `apply_lf_asym_lora`.
- `asym_gemm/training/qwen3_moe.py`, `asym_gemm/training/llama4_moe.py`, and `asym_gemm/training/packed_moe.py` pass `backend: Literal["asym", "torch"]` into AsymGEMM expert wrappers. These wrappers own CPU-resident frozen weights for the `asym` path and GPU-resident torch baselines for the `torch` path.
- `asym_gemm/training/kt_moe.py` implements the KT fused MoE backend through `KTRoutedExpertMoE`, `_KTRoutedMoEFunction`, and `KTMoEWrapper`. KT is separate from `VALID_BACKENDS` and separate from LF `asym_backend`.
- `scripts/lf/profile_lora_lf.sh` is the primary LF sweep wrapper. It currently accepts backend specs for `asym`, `torch`, `kt_torchbf16`, and `kt_armbf16`; it maps backend labels to GPU counts, artifact roots, recompute modes, loss comparisons, and `scripts/lf/run_lf_lora_sft.sh` environment variables.
- `scripts/lf/run_lf_lora_sft.sh` currently accepts `BACKEND=asym`, `BACKEND=torch`, `BACKEND=kt_torchbf16`, and `BACKEND=kt_armbf16`. It sets `--use_asym_gemm true --asym_backend asym` for `asym`, sets `--use_asym_gemm true --asym_backend torch` only when `TORCH_USE_ASYM_GEMM_LORA=true`, and sets `--use_kt true` for KT.
- `scripts/lf/run_lf_lora_sft.sh` limits DeepSpeed launcher scope to `BACKEND=torch` through `is_torch_distributed_run()` and `assert_deepspeed_scope()`. SuperOffload requires a new DeepSpeed path outside the torch backend branch.
- `scripts/lf/run_lf_profiled_train.py` builds `profile.json` and already records `config.backend`, KT counters, memory summaries, and source-profiler timing ranges. It can carry a `superoffload` runtime summary without touching model-layer AsymGEMM counters.
- `scripts/plotting/plot_lf_interconnect_ctc.py` and `scripts/plotting/plot_lf_memory_breakdown.py` accept arbitrary `--backend` filters. They do not contain hard backend choices.
- `scripts/plotting/plot_lora_operator.py` has marker and hatch maps only for `asym`, `torch`, and `kt`; this script targets LoRA operator microbenchmarks, not LF DeepSpeed training.
- LlamaFactory `src/llamafactory/hparams/model_args.py` contains `use_asym_gemm`, `asym_backend: Literal["asym", "torch"]`, and KT fields. LlamaFactory `src/llamafactory/hparams/parser.py` rejects AsymGEMM with DeepSpeed ZeRO-3 and rejects KT with DeepSpeed ZeRO-3.
- LlamaFactory `src/llamafactory/model/adapter.py` uses ordinary PEFT LoRA when `use_asym_gemm=False` and `use_kt=False`. That ordinary path is the correct model path for SuperOffload.
- Local DeepSpeed lives at `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/deepspeed`.
- DeepSpeed `deepspeed/runtime/zero/offload_config.py` defines `DeepSpeedZeroOffloadOptimizerConfig.super_offload: bool = False` and `cpuadam_cores_perc: float = Field(0.8, ge=0.0, le=1.0)`.
- DeepSpeed `deepspeed/runtime/engine.py` exposes `super_offload()` and selects `SuperOffloadOptimizer_Stage3` instead of `DeepSpeedZeroOptimizer_Stage3` during ZeRO stage 3 optimizer construction when `offload_optimizer.super_offload` is true.
- DeepSpeed `deepspeed/runtime/engine.py` logs `DeepSpeed Final Optimizer = <class name>` after optimizer construction, so a live run can prove SuperOffload selection through the train log.
- DeepSpeed `deepspeed/runtime/superoffload/superoffload_stage3.py` implements `SuperOffloadOptimizer_Stage3` and requires CUDA through `_validate_superoffload_accelerator()`.
- DeepSpeed `deepspeed/runtime/superoffload/superoffload_utils.py` implements the CPU optimizer worker and CPU affinity split used by SuperOffload.
- Tests currently assert the public E2E LoRA backends as `("torch", "asym", "kt")` in `tests/training/test_profile_lora_backends.py` and assert frozen-linear `VALID_BACKENDS == ("asym", "torch")` in `tests/training/test_cpu_resident_frozen_base.py`.

## Non-Negotiable Design

- Implementation edits after the copy must target `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO`.
- Production `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM` is a read-only source for copying and inspection after this planning document is updated.
- Every relative file path in the implementation sections below refers to the `AsymGEMM-SO` copy.
- The copy operation must use the current filesystem state of `AsymGEMM`, including uncommitted working-tree edits and untracked source files.
- The copy operation must exclude Git history, Python caches, build outputs, virtual environments, and generated profiling artifact directories.
- The copied repo keeps the Python package name `asym_gemm`, the native extension name `asym_gemm._C`, and existing function/file names.
- `AsymGEMM-SO` scripts must place `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO` before the production tree and before site packages on `PYTHONPATH`.
- `AsymGEMM-SO` scripts must default `ROOT` and `ASYM_DIR` to `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO`.
- `superoffload` remains outside `asym_gemm/training/frozen_linear.py::VALID_BACKENDS`.
- `superoffload` remains outside LF `--asym_backend`; legal LF AsymGEMM backend values stay `asym` and `torch`.
- `superoffload` must launch ordinary LF PEFT LoRA with `--use_asym_gemm false` and `--use_kt false`.
- `superoffload` must launch with DeepSpeed ZeRO stage 3 and a config whose `zero_optimization.offload_optimizer.super_offload` value is `true`.
- `superoffload` is policy-independent for AsymGEMM expert recompute policies. Sweep logic must run it only with `expert_policy=none`.
- `superoffload` uses the model GPU count requested by the LF workload, matching the `torch` distributed baseline.
- The local DeepSpeed repository must be placed before site packages on `PYTHONPATH` for `superoffload` runs unless the active environment already imports a DeepSpeed package that contains `deepspeed.runtime.superoffload.superoffload_stage3`.
- Source-profiler timing is diagnostic only. Nsight Systems postprocessed `profile.json` remains the authoritative step-latency artifact.
- Source-profiler memory data remains the authoritative memory-attribution artifact for LF sweep summaries.
- Existing `asym`, `torch`, and KT behavior must remain byte-for-byte equivalent for generated commands unless the selected backend list contains `superoffload`.

## Files To Change Or Create

All file paths in this section are under `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO` after Stage 0 creates the copy.

0. Create the isolated repository directory `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO`.

   - Copy from `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM/` with `rsync -a --delete`.
   - Exclude `.git/`, `.venv/`, `build/`, `dist/`, `*.egg-info/`, `__pycache__/`, `.pytest_cache/`, `profiling/`, `profiling_kt/`, `wandb/`, and generated trace files.
   - Initialize a new Git repository in `AsymGEMM-SO` with `git init`.
   - Commit the copied baseline in `AsymGEMM-SO` before SuperOffload edits.
   - Do not create symlinks from `AsymGEMM-SO` back into production `AsymGEMM`.

1. Change `scripts/lf/profile_lora_lf.sh`.

   - Change the default `ROOT` to `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO`.
   - Add `superoffload` to help text, backend spec examples, backend comparison argument text, and artifact README text.
   - Add defaults:
     - `DEEPSPEED_DIR=${DEEPSPEED_DIR:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/deepspeed}`
     - `SUPER_OFFLOAD_DEEPSPEED_CONFIG=${SUPER_OFFLOAD_DEEPSPEED_CONFIG:-}`
     - `SUPER_OFFLOAD_CPUADAM_CORES_PERC=${SUPER_OFFLOAD_CPUADAM_CORES_PERC:-0.8}`
     - `CHECK_SUPEROFFLOAD=${CHECK_SUPEROFFLOAD:-true}`
   - After `base_output_root="$(abs_path "${output_root}")"` is computed, set an empty `SUPER_OFFLOAD_DEEPSPEED_CONFIG` to `${base_output_root}/deepspeed/ds_z3_superoffload_config.json`.
   - Extend `backend_gpu_count()` so `superoffload` returns the current workload model GPU count.
   - Extend `backend_label()` so accepted aliases are `superoffload`, `super_offload`, `so`, and `ds_superoffload`, all canonicalized to `superoffload`.
   - Extend `expand_backend_spec()` so `superoffload|recompute` and `superoffload|norecompute` both produce `superoffload|norecompute`, because SuperOffload does not use AsymGEMM activation recompute.
   - Add `selected_has_superoffload` beside `selected_has_asym`, `selected_has_torch`, and `selected_has_kt`.
   - Add validation that selected SuperOffload runs have:
     - a readable `DEEPSPEED_DIR/deepspeed/runtime/superoffload/superoffload_stage3.py`, or an importable DeepSpeed package with that module;
     - `SUPER_OFFLOAD_CPUADAM_CORES_PERC` in `[0.0, 1.0]`;
     - a rendered DeepSpeed config generated before the first SuperOffload job.
   - Generate the SuperOffload DeepSpeed config once per sweep root by calling `scripts/lf/render_superoffload_deepspeed_config.py`.
   - Pass `DEEPSPEED_DIR`, `SUPER_OFFLOAD_DEEPSPEED_CONFIG`, `SUPER_OFFLOAD_CPUADAM_CORES_PERC`, and `CHECK_SUPEROFFLOAD` through `run_job()` for `BACKEND=superoffload`.
   - Set `ASYM_GEMM_LF_CONFIG_BACKEND=superoffload` for source profile metadata.
   - Set `ASYM_GEMM_LF_CONFIG_SUPEROFFLOAD_CONFIG`, `ASYM_GEMM_LF_CONFIG_SUPEROFFLOAD_CPUADAM_CORES_PERC`, and `ASYM_GEMM_LF_CONFIG_DEEPSPEED_DIR` for source profile metadata.
   - Treat SuperOffload as policy-independent in run scheduling and loss comparison skipping. The existing skip message text becomes `torch/KT/SuperOffload backends are policy-independent.`
   - Keep `scripts/lf/profile_lora_lf_fused.sh` unchanged. That script remains a fused AsymGEMM-vs-torch wrapper and already rejects non-`asym`/`torch` labels.

2. Change `scripts/lf/run_lf_lora_sft.sh`.

   - Change the default `ROOT` to `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO`.
   - Add `superoffload` to backend validation and usage text.
   - Add `DEEPSPEED_DIR`, `SUPER_OFFLOAD_DEEPSPEED_CONFIG`, `SUPER_OFFLOAD_CPUADAM_CORES_PERC`, and `CHECK_SUPEROFFLOAD` environment defaults.
   - Replace `is_torch_distributed_run()` with two predicates:
     - `is_torch_distributed_run`: `BACKEND=torch && NUM_GPUS > 1`
     - `is_deepspeed_required_run`: `BACKEND=superoffload || (BACKEND=torch && NUM_GPUS > 1 && TORCH_DISTRIBUTED_BACKEND=deepspeed)`
   - Update `assert_deepspeed_scope()` so `--deepspeed` is accepted for `BACKEND=torch` and `BACKEND=superoffload`, and rejected for `BACKEND=asym` and KT.
   - For `BACKEND=superoffload`, do not append `--use_asym_gemm`, `--asym_backend`, `--use_kt`, or KT flags.
   - For `BACKEND=superoffload`, append `--deepspeed "${SUPER_OFFLOAD_DEEPSPEED_CONFIG}"` and set `--pure_bf16 false`, matching DeepSpeed ZeRO-3 BF16 behavior.
   - For `BACKEND=superoffload`, set `ASYM_DIR="${ROOT}"`.
   - For `BACKEND=superoffload`, build `RUN_PYTHONPATH` as `${DEEPSPEED_DIR}:${ASYM_DIR}:${LF_DIR}/src:<remaining entries>` so the `AsymGEMM-SO` package wins over production `AsymGEMM`.
   - For `BACKEND=superoffload`, set `USE_ASYM_GEMM=0`, do not export `USE_KT`, and export `ASYM_GEMM_LF_CONFIG_BACKEND=superoffload`.
   - For `BACKEND=superoffload`, launch through the same Accelerate/DeepSpeed path used by torch ZeRO-3 distributed runs, with `NUM_GPUS` equal to the model GPU count.
   - For `BACKEND=superoffload`, preserve DeepSpeed rank-0 log output in `${LOG_FILE}` so the checker can read `DeepSpeed Final Optimizer = SuperOffloadOptimizer_Stage3`.
   - After training, when `CHECK_SUPEROFFLOAD=true`, call `scripts/lf/check_superoffload_run.py --profile-json "${SOURCE_PROFILE_JSON}" --train-log "${LOG_FILE}" --require-enabled`.
   - Emit the exact failure text `backend=superoffload completed without a SuperOffload runtime marker; inspect ${LOG_FILE}` when the checker fails.

3. Create `scripts/lf/render_superoffload_deepspeed_config.py`.

   - CLI arguments:
     - `--base PATH`, default `${LF_DIR}/examples/deepspeed/ds_z3_offload_config.json`
     - `--output PATH`, required from the caller
     - `--cpuadam-cores-perc FLOAT`, default `0.8`
   - Validate `0.0 <= cpuadam_cores_perc <= 1.0`.
   - Load the base JSON with `json.load`.
   - Ensure `zero_optimization.stage == 3`.
   - Ensure `zero_optimization.offload_optimizer.device == "cpu"`.
   - Ensure `zero_optimization.offload_optimizer.pin_memory == true`.
   - Set `zero_optimization.offload_optimizer.super_offload = true`.
   - Set `zero_optimization.offload_optimizer.cpuadam_cores_perc = <float>`.
   - Ensure `zero_optimization.offload_param.device == "cpu"` and `zero_optimization.offload_param.pin_memory == true`.
   - Preserve non-offload DeepSpeed keys from the base file.
   - Write stable, sorted JSON with two-space indentation and a trailing newline.

4. Create `scripts/lf/check_superoffload_run.py`.

   - CLI arguments:
     - `--profile-json PATH`
     - `--train-log PATH`
     - `--require-enabled`
   - Load `profile.json` when present.
   - Read the train log when present.
   - Pass when either source is true:
     - `profile["superoffload"]["config_super_offload"] is True` and the train log contains `DeepSpeed Final Optimizer = SuperOffloadOptimizer_Stage3`;
     - `profile["superoffload"]["optimizer_class"] == "SuperOffloadOptimizer_Stage3"`.
   - Fail with exit code `2` when `--require-enabled` is set and no marker is found.
   - Print a one-line JSON diagnostic containing `enabled`, `profile_json`, `train_log`, `optimizer_class`, and `marker_source`.

5. Change `scripts/lf/run_lf_profiled_train.py`.

   - Extend `_config_from_args()` with:
     - `superoffload_config` from `ASYM_GEMM_LF_CONFIG_SUPEROFFLOAD_CONFIG`
     - `superoffload_cpuadam_cores_perc` from `ASYM_GEMM_LF_CONFIG_SUPEROFFLOAD_CPUADAM_CORES_PERC`
     - `deepspeed_dir` from `ASYM_GEMM_LF_CONFIG_DEEPSPEED_DIR`
   - Add `_superoffload_summary_from_config(config)`:
     - Load `config["superoffload_config"]` when the path exists.
     - Read `zero_optimization.offload_optimizer.super_offload`.
     - Read `zero_optimization.offload_optimizer.cpuadam_cores_perc`.
     - Return `{"enabled": <bool>, "runtime_verified": false, "optimizer_class": null, "config_super_offload": <bool>, "cpuadam_cores_perc": <float or null>, "deepspeed_config": <path or null>}`.
   - Add `"superoffload": _superoffload_summary_from_config(self.config)` to the emitted profile report.
   - Keep existing `"kt"` and AsymGEMM stats unchanged.

6. Add `tests/lf/test_superoffload_backend_scripts.py`.

   - `test_render_superoffload_config_sets_deepspeed_keys`: render from LF `ds_z3_offload_config.json`, load the output JSON, and assert ZeRO stage 3, CPU optimizer offload, CPU parameter offload, `super_offload is True`, and the requested `cpuadam_cores_perc`.
   - `test_check_superoffload_run_accepts_profile_and_log_marker`: create a temporary profile with `{"superoffload": {"config_super_offload": true}}`, create a log containing `DeepSpeed Final Optimizer = SuperOffloadOptimizer_Stage3`, and assert the checker exits `0`.
   - `test_check_superoffload_run_accepts_profile_optimizer_class`: create a temporary profile with `{"superoffload": {"optimizer_class": "SuperOffloadOptimizer_Stage3"}}` and assert the checker exits `0`.
   - `test_check_superoffload_run_rejects_missing_marker`: create a profile with `enabled=false`, an empty log, run with `--require-enabled`, and assert exit code `2`.
   - `test_profile_lora_lf_dry_run_accepts_superoffload`: run `scripts/lf/profile_lora_lf.sh` in dry-run mode with `BACKEND_SPECS=superoffload|norecompute`, then assert `jobs.tsv` contains `superoffload` and the generated command contains `--deepspeed`.
   - `test_profile_lora_lf_skips_expert_policy_for_superoffload`: run a dry-run sweep with `EXPERT_POLICIES=none,layer_0` and assert the SuperOffload job count equals the number of `expert_policy=none` combinations.

7. Change existing tests only where assertions cover backend lists.

   - Keep `tests/training/test_cpu_resident_frozen_base.py` assertion `VALID_BACKENDS == ("asym", "torch")`.
   - Keep `tests/training/test_profile_lora_backends.py` E2E toy profiler backend assertion unchanged unless `scripts/lora/profile_lora_e2e.py` gains a DeepSpeed launch mode in a separate task.
   - Add LF SuperOffload script tests under `tests/lf/` rather than broadening the AsymGEMM frozen-linear test surface.

## Algorithm Details

### Repository Copy And Entry Point

Create the SO copy from the production tree before code edits:

```bash
SRC=/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM
DST=/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO
mkdir -p "$(dirname "${DST}")"
rsync -a --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'build/' \
  --exclude 'dist/' \
  --exclude '*.egg-info/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude 'profiling/' \
  --exclude 'profiling_kt/' \
  --exclude 'wandb/' \
  --exclude '*.nsys-rep' \
  --exclude '*.sqlite' \
  "${SRC}/" "${DST}/"
git -C "${DST}" init
git -C "${DST}" add .
git -C "${DST}" -c user.name=asymgemm-so -c user.email=asymgemm-so@example.invalid commit -m "Initial AsymGEMM-SO baseline copy"
```

The copied repo keeps `setup.py` package name `asym_gemm`. Import resolution is controlled by environment and install path:

```bash
SO_DIR=/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO
LF_DIR=/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory
"${LF_DIR}/.venv/bin/python" -m pip install -e "${SO_DIR}"
PYTHONPATH="${SO_DIR}:${PYTHONPATH:-}" "${LF_DIR}/.venv/bin/python" - <<'PY'
import pathlib
import asym_gemm
path = pathlib.Path(asym_gemm.__file__).resolve()
assert str(path).startswith("/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO/")
print(path)
PY
```

No Python module, class, function, or script import name changes for the SO copy. The entry point changes from the production path to the SO repository path.

### Backend Dispatch

`profile_lora_lf.sh` canonicalizes backend labels before any scheduling:

```text
asym              -> asym
torch             -> torch
kt, kt_torchbf16  -> kt_torchbf16
kt_armbf16        -> kt_armbf16
superoffload      -> superoffload
super_offload     -> superoffload
so                -> superoffload
ds_superoffload   -> superoffload
```

For `superoffload`, recompute is canonicalized to `norecompute`, GPU count is the workload model GPU count, and expert policy is forced to `none`.

### SuperOffload Config Rendering

The sweep wrapper renders a SuperOffload config before job scheduling. The renderer starts from LF `ds_z3_offload_config.json`, then applies the exact ZeRO-3 optimizer-offload fields DeepSpeed reads:

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

The renderer preserves the remaining DeepSpeed config keys from the base file.

### Launch Path

The SuperOffload run path is:

```text
scripts/lf/profile_lora_lf.sh
  -> BACKEND=superoffload scripts/lf/run_lf_lora_sft.sh
    -> PYTHONPATH=<local deepspeed>:<AsymGEMM-SO>:<LF src>
    -> accelerate launch ... scripts/lf/run_lf_profiled_train.py ... --deepspeed <rendered superoffload config>
      -> LlamaFactory ordinary PEFT LoRA path
      -> DeepSpeed engine
      -> SuperOffloadOptimizer_Stage3
```

The command must not contain `--use_asym_gemm`, `--asym_backend`, `--use_kt`, or `--kt_backend`.

### Runtime Marker

`run_lf_profiled_train.py` records config truth under:

```json
{
  "config": {
    "backend": "superoffload"
  },
  "superoffload": {
    "enabled": true,
    "runtime_verified": false,
    "optimizer_class": null,
    "config_super_offload": true,
    "cpuadam_cores_perc": 0.8,
    "deepspeed_config": "<path>"
  }
}
```

`check_superoffload_run.py` upgrades proof from config truth to runtime truth by requiring `config_super_offload is True` and the train-log line `DeepSpeed Final Optimizer = SuperOffloadOptimizer_Stage3`. It also accepts an explicit profile marker with `optimizer_class == "SuperOffloadOptimizer_Stage3"`.

### Artifact Layout

SuperOffload artifacts use the existing LF layout:

```text
<config_root>/superoffload__<profiler>__norecompute__polnone/s<seq_len>/
  command.txt
  train.log
  lf_run/
  source_profile.json
  profile.json
  table.md
  trace.nsys-rep
  trace.sqlite
```

Combined plots and comparison outputs stay under the existing `combined/`, `memory_combined/`, `c2c_combined/`, and `comparisons/` roots.

## Stage Plan And Gates

### Stage 0: Create AsymGEMM-SO Copy

Create `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO` from the current production tree, initialize it as an independent Git repository, and install it into the LF environment as the `asym_gemm` package.

Correctness gate:

- `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO/setup.py` exists.
- `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO/asym_gemm/training/frozen_linear.py` exists.
- `git -C /home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO status --short` exits `0`.
- `PYTHONPATH=/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO` import smoke prints an `asym_gemm.__file__` path under `AsymGEMM-SO`.
- Production status captured before Stage 0 and production status captured after Stage 4 are byte-identical after excluding `agent/superoffload/superoffload_integration.md`.

Latency gate:

- Stage 0 has no live latency gate. It is complete only after the copied repo is the active `asym_gemm` import source.

### Stage 1: Script Backend Plumbing

Implement `superoffload` backend canonicalization, scheduling, dry-run command generation, artifact naming, policy skipping, and comparison bookkeeping in `scripts/lf/profile_lora_lf.sh` and `scripts/lf/run_lf_lora_sft.sh`.

Correctness gate:

- `bash -n scripts/lf/profile_lora_lf.sh scripts/lf/run_lf_lora_sft.sh` exits `0`.
- A dry-run sweep with `BACKEND_SPECS=superoffload|norecompute` writes one SuperOffload job per profiler and sequence length.
- SuperOffload dry-run `command.txt` contains `--deepspeed <rendered config>`.
- SuperOffload dry-run `command.txt` does not contain `--use_asym_gemm`, `--asym_backend`, `--use_kt`, or `--kt_backend`.
- Runs with `EXPERT_POLICIES` containing non-`none` values skip SuperOffload jobs for those policies.

Latency gate:

- Stage 1 has no live latency gate. It is complete only after dry-run command generation is deterministic.

### Stage 2: DeepSpeed Config And Runtime Checker

Implement `scripts/lf/render_superoffload_deepspeed_config.py`, `scripts/lf/check_superoffload_run.py`, and source-profile metadata fields in `scripts/lf/run_lf_profiled_train.py`.

Correctness gate:

- The renderer output contains `zero_optimization.stage == 3`.
- The renderer output contains `zero_optimization.offload_optimizer.device == "cpu"`.
- The renderer output contains `zero_optimization.offload_optimizer.super_offload is True`.
- The renderer output contains the requested `cpuadam_cores_perc`.
- The checker exits `0` for a synthetic profile/log containing SuperOffload markers.
- The checker exits `2` for a synthetic profile/log without SuperOffload markers when `--require-enabled` is set.

Latency gate:

- Stage 2 has no live latency gate. It is complete only after config and checker tests pass.

### Stage 3: LF Source-Profiler Smoke

Run a short LF source-profiler sweep comparing `torch` and `superoffload` on the target GH/GB machine.

Correctness gate:

- Both backends complete training with identical dataset, seed, LoRA rank, LoRA alpha, dropout, BF16 mode, max steps, and sequence length.
- Existing LF loss comparison exits `0`.
- SuperOffload `source_profile.json` has `config.backend == "superoffload"`.
- SuperOffload `source_profile.json` has `superoffload.config_super_offload is True`.
- `scripts/lf/check_superoffload_run.py --require-enabled` exits `0` for the SuperOffload run.

Latency gate:

- SuperOffload source profile emits finite `train_step`, forward, and backward timings.
- Source-profiler timings are recorded for diagnostics and not used as the final latency ranking.

### Stage 4: Nsight Systems Baseline

Run an Nsight Systems sweep comparing `torch` DeepSpeed ZeRO-3 offload and `superoffload` on the target GH/GB machine.

Correctness gate:

- Every SuperOffload Nsight run produces `trace.nsys-rep`, `trace.sqlite`, `profile.json`, and `table.md`.
- Combined LF plots include a `superoffload` backend row.
- Loss comparison passes for the matching source-profiler run from Stage 3.

Latency gate:

- The authoritative latency metric is the postprocessed Nsight Systems `profile.json`.
- SuperOffload median measured step latency must be less than or equal to `1.10 * torch_zero3_offload_median_step_latency` for the same model, sequence length, GPU count, and profiler settings.
- SuperOffload peak GPU memory from source-profile memory attribution must be less than or equal to torch ZeRO-3 offload peak GPU memory for the same model, sequence length, and GPU count.

## Validation Commands

Create the isolated SO repository:

```bash
SRC=/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM
DST=/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO
git -C "${SRC}" status --short -- ':(exclude)agent/superoffload/superoffload_integration.md' > /tmp/asymgemm_prod_status.before
mkdir -p "$(dirname "${DST}")"
rsync -a --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'build/' \
  --exclude 'dist/' \
  --exclude '*.egg-info/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude 'profiling/' \
  --exclude 'profiling_kt/' \
  --exclude 'wandb/' \
  --exclude '*.nsys-rep' \
  --exclude '*.sqlite' \
  "${SRC}/" "${DST}/"
git -C "${DST}" init
git -C "${DST}" add .
git -C "${DST}" -c user.name=asymgemm-so -c user.email=asymgemm-so@example.invalid commit -m "Initial AsymGEMM-SO baseline copy"
git -C "${SRC}" status --short -- ':(exclude)agent/superoffload/superoffload_integration.md' > /tmp/asymgemm_prod_status.after
cmp /tmp/asymgemm_prod_status.before /tmp/asymgemm_prod_status.after
```

Verify the SO import entry point:

```bash
SO_DIR=/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO
LF_DIR=/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory
"${LF_DIR}/.venv/bin/python" -m pip install -e "${SO_DIR}"
PYTHONPATH="${SO_DIR}:${PYTHONPATH:-}" "${LF_DIR}/.venv/bin/python" - <<'PY'
import pathlib
import asym_gemm
path = pathlib.Path(asym_gemm.__file__).resolve()
assert str(path).startswith("/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO/")
print(path)
PY
```

Syntax and unit validation from the SO copy:

```bash
cd /home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO
bash -n scripts/lf/profile_lora_lf.sh scripts/lf/run_lf_lora_sft.sh
pytest tests/lf/test_superoffload_backend_scripts.py
pytest tests/training/test_cpu_resident_frozen_base.py tests/training/test_profile_lora_backends.py
```

Render and inspect the DeepSpeed config:

```bash
python scripts/lf/render_superoffload_deepspeed_config.py \
  --base /home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory/examples/deepspeed/ds_z3_offload_config.json \
  --output /tmp/ds_z3_superoffload_config.json \
  --cpuadam-cores-perc 0.8

python - <<'PY'
import json
cfg = json.load(open("/tmp/ds_z3_superoffload_config.json"))
zero = cfg["zero_optimization"]
assert zero["stage"] == 3
assert zero["offload_optimizer"]["device"] == "cpu"
assert zero["offload_optimizer"]["super_offload"] is True
assert zero["offload_optimizer"]["cpuadam_cores_perc"] == 0.8
assert zero["offload_param"]["device"] == "cpu"
PY
```

Dry-run the LF SuperOffload backend:

```bash
BACKEND_SPECS="superoffload|norecompute,torch|norecompute" \
PROFILERS="source" \
SEQ_LENS="4096" \
MAX_STEPS="2" \
WARMUP_STEPS="1" \
PREPARE_DATASETS="false" \
DRY_RUN="true" \
/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO/scripts/lf/profile_lora_lf.sh \
  --model-specs "Qwen/Qwen3-30B-A3B|2" \
  --output-root /tmp/asymgemm_lf_superoffload_dryrun
```

Live source-profiler smoke on the GH/GB target:

```bash
BACKEND_SPECS="torch|norecompute,superoffload|norecompute" \
PROFILERS="source" \
SEQ_LENS="4096" \
MAX_STEPS="3" \
WARMUP_STEPS="1" \
PREPARE_DATASETS="true" \
TORCH_DISTRIBUTED_BACKEND="deepspeed" \
CHECK_SUPEROFFLOAD="true" \
/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO/scripts/lf/profile_lora_lf.sh \
  --model-specs "Qwen/Qwen3-30B-A3B|2" \
  --output-root /tmp/asymgemm_lf_superoffload_source
```

Live Nsight Systems baseline on the GH/GB target:

```bash
BACKEND_SPECS="torch|norecompute,superoffload|norecompute" \
PROFILERS="nsys" \
SEQ_LENS="4096" \
MAX_STEPS="6" \
WARMUP_STEPS="2" \
PREPARE_DATASETS="true" \
TORCH_DISTRIBUTED_BACKEND="deepspeed" \
CHECK_SUPEROFFLOAD="true" \
/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO/scripts/lf/profile_lora_lf.sh \
  --model-specs "Qwen/Qwen3-30B-A3B|2" \
  --output-root /tmp/asymgemm_lf_superoffload_nsys
```

Collect existing artifacts without rerunning training:

```bash
BACKEND_SPECS="torch|norecompute,superoffload|norecompute" \
PROFILERS="source,nsys" \
SEQ_LENS="4096" \
COLLECT_EXISTING="true" \
PREPARE_DATASETS="false" \
/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO/scripts/lf/profile_lora_lf.sh \
  --model-specs "Qwen/Qwen3-30B-A3B|2" \
  --output-root /tmp/asymgemm_lf_superoffload_nsys
```

## Efficiency And Safety Constraints

- Do not implement SuperOffload edits in production `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM`.
- Use production `AsymGEMM` only as the source for the initial `rsync` copy and for read-only inspection.
- Keep `AsymGEMM-SO` independent; do not symlink source files, script files, or package directories back to production `AsymGEMM`.
- Keep the import package name `asym_gemm`; isolate by `PYTHONPATH`, editable install target, and script defaults.
- Do not widen AsymGEMM frozen-linear dispatch for SuperOffload.
- Do not call AsymGEMM expert wrappers for SuperOffload.
- Do not set `USE_ASYM_GEMM=1` for SuperOffload.
- Do not set `USE_KT=1` for SuperOffload.
- Do not allow SuperOffload under `scripts/lora/profile_lora_e2e.py`; that toy profiler lacks a DeepSpeed trainer path.
- Keep DeepSpeed config generation deterministic so repeated sweeps share the same command hash and artifact layout.
- Validate the local DeepSpeed module path before launching so a missing SuperOffload runtime fails before GPU allocation.
- Keep SuperOffload comparison against torch DeepSpeed ZeRO-3 offload, not against single-GPU torch without ZeRO.
- Use source-profile memory attribution for memory comparisons.
- Use Nsight Systems postprocessed timing for latency comparisons.
- Keep SuperOffload run directories separate from `torch`, `asym`, and KT directories.
- Keep AsymGEMM and KT correctness counters unchanged.

## Final Acceptance Criteria

- `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO` exists as an independent repository copied from the current production filesystem state.
- `asym_gemm.__file__` resolves under `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM-SO` for SO eval runs.
- Production `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM` has no SuperOffload implementation edits.
- `scripts/lf/profile_lora_lf.sh` accepts `BACKEND_SPECS=superoffload|norecompute` and rejects invalid SuperOffload configuration before training.
- `scripts/lf/run_lf_lora_sft.sh` launches SuperOffload through DeepSpeed ZeRO-3 with a rendered config containing `offload_optimizer.super_offload=true`.
- SuperOffload launch commands use ordinary PEFT LoRA and contain no AsymGEMM or KT model-rewrite flags.
- `source_profile.json` for SuperOffload records `config.backend == "superoffload"` and `superoffload.config_super_offload is True`.
- `check_superoffload_run.py --require-enabled` passes for live SuperOffload runs.
- LF loss comparison passes between torch DeepSpeed ZeRO-3 offload and SuperOffload for the same workload.
- Nsight Systems artifacts are produced for SuperOffload and included in combined LF latency tables and plots.
- SuperOffload median Nsight step latency is within the Stage 4 threshold relative to torch DeepSpeed ZeRO-3 offload.
- SuperOffload peak GPU memory is within the Stage 4 threshold relative to torch DeepSpeed ZeRO-3 offload.
- Existing `asym`, `torch`, and KT tests continue to pass without backend-list churn in AsymGEMM frozen-linear code.
