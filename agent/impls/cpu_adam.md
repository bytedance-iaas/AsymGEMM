# AsymGEMM LoRA-SFT CPU AdamW Implementation Plan

This document is the implementation contract for adding CPU AdamW optimizer-state offload to the single-GPU AsymGEMM LoRA-SFT path. The target is to remove GPU-resident Adam moment state for trainable LoRA tensors while preserving the current AsymGEMM compute layout.

The first implementation must not put AsymGEMM under DeepSpeed ZeRO. It must use a normal LlamaFactory/HF Trainer optimizer object whose update tensors live on CPU.

Path variables used below:

```bash
export ASYM_DIR=${ASYM_DIR:-$(pwd)}
export SFT_ROOT=${SFT_ROOT:-$(cd "${ASYM_DIR}/../.." && pwd)}
export LF_DIR=${LF_DIR:-${SFT_ROOT}/third_party/LlamaFactory}
export DEEPSPEED_DIR=${DEEPSPEED_DIR:-${SFT_ROOT}/third_party/deepspeed}
```

## Core Decision

Use an Asym-owned CPU-master optimizer wrapper:

```text
model compute copy:
  CUDA bf16 LoRA nn.Parameter tensors remain in the model and are used by AsymGEMM forward/backward.

optimizer update copy:
  CPU master LoRA tensors are optimized by torch.optim.AdamW or DeepSpeedCPUAdam.

per optimizer step:
  CUDA LoRA grad -> CPU master grad, only when grad exists
  CPU AdamW update on CPU master
  updated CPU master weight -> CUDA LoRA param, only for params updated this step
```

This keeps AsymGEMM kernels and current autograd unchanged for the initial win. It removes GPU Adam state, not GPU LoRA weights or GPU LoRA gradients.

Do not use ZeRO-owned flat partitions for this path. DeepSpeed ZeRO flattens and partitions parameters to support distributed sharding, all-gather/release hooks, optimizer subgrouping, and CPU/NVMe swapping. Those are useful for generic distributed training, but they are the wrong owner for AsymGEMM's expert-local CPU fetch layout.

## Per-Module Offload Impact On The Optimizer

The new LF `asym_offload_modules` selector changes frozen/base-weight residency, not optimizer ownership. CPUAdamW must not optimize or create masters for any frozen offloaded component. That includes routed expert bases, routers, shared experts, attention bases, embeddings, LM head, norms, `HostWeight` objects, and `AsymFrozen*` wrappers.

The optimizer contract remains:

```text
selected trainable LoRA nn.Parameter -> one CPU fp32 master -> CPU AdamW state
selected frozen/base module weight    -> AsymGEMM HostWeight/frozen wrapper only
```

Required copy policy:

1. Constructor: copy each unique trainable LoRA CUDA parameter to one CPU fp32 master exactly once. Deduplicate by parameter object identity, or by an identical tensor view key `(device, storage data ptr, storage offset, shape, stride, dtype)` for true tied/shared aliases. Do not collapse different sliced views merely because they share the same underlying allocation.
2. Step: copy CUDA grad to CPU only for LoRA params with `grad is not None`.
3. Step: after the CPU AdamW update, copy CPU master back only for LoRA params that had a grad/update in that step. A no-grad LoRA param should not pay a CPU-to-GPU copyback.
4. Checkpoint load: copy restored CPU masters back into CUDA LoRA params once after `load_state_dict()`.
5. Never copy frozen/base `HostWeight` tensors into optimizer masters or optimizer state.
6. Never run a whole-model parameter copy to support CPUAdamW.

If a future wrapping path leaves trainable LoRA compute params on CPU after model preparation, v1 CPUAdamW must reject that with a precise error. CPU-fetched trainable LoRA is Stage 7, because it changes the forward/backward tensor ownership model.

Allowed one-time non-optimizer copies are separate from CPUAdamW:

- LlamaFactory CPU-first load may call `model.to(current_device)` after adapter conversion. This is the intended placement step for trainable LoRA parameters and ordinary non-offloaded model state.
- Linear/router/expert offload paths that use AsymGEMM kernels may call `HostWeight(..., clone=False, pin_memory=True)` through `adopt_host_weight(..., pin_memory_policy="auto")`. This may replace pageable CPU storage with one pinned CPU copy. It is allowed because AsymGEMM needs pinned CPU host memory for direct fetch. It is still one CPU owner, not an optimizer master.
- Embedding and norm offload wrappers currently use strict no-copy CPU adoption (`pin_memory_policy="none"`). Their forward methods may copy small CPU weights or outputs to the active device as part of executing the frozen module. Those runtime copies are not CPUAdam masters and must be accounted as frozen/base offload overhead, not optimizer state.
- Llama4 packed expert conversion uses a one-time `transpose(...).contiguous()` layout normalization before creating the Asym packed expert wrapper. That is a kernel-layout construction cost, not CPUAdam optimizer state.
- Checkpoint save/load may serialize or rehydrate CPU host weights. That path must remain outside optimizer-state accounting.

Forbidden copies:

- CPUAdamW must not make a second CPU copy of frozen/base weights.
- CPUAdamW must not repair CPU LoRA compute params by moving them to CUDA itself.
- CPUAdamW must not flatten/repack LoRA or base weights into ZeRO-style buffers.
- Profiling/postprocessing must not count pinned HostWeight replacements as CPUAdam optimizer-state savings.
- Profiling/postprocessing must not count embedding/norm forward copies as CPUAdam optimizer-state savings or losses.

## Verified Code Facts

- `DeepSpeedCPUAdam` is `deepspeed.ops.adam.cpu_adam.DeepSpeedCPUAdam`; it is a `torch.optim.Optimizer` wrapper over `CPUAdamBuilder().load()`.
- `DeepSpeedCPUAdam.step()` asserts optimized parameters with non-`None` grads are on CPU. It cannot directly step current CUDA LoRA parameters once those parameters have grads.
- `DeepSpeedCPUAdam` does not require flat weights. It accepts normal CPU tensors in normal optimizer param groups.
- `DeepSpeedCPUAdam` initializes `exp_avg` and `exp_avg_sq` lazily per parameter only after that parameter has a grad.
- AsymGEMM LF setup freezes all non-LoRA params and validates that only LoRA params remain trainable.
- AsymGEMM expert base weights are `HostWeight` objects, intentionally not parameters or buffers, so DeepSpeed/ZeRO will not see or manage them.
- Current LF per-module offload supports `routed_experts`, `router`, `shared_experts`, attention targets, `embed_tokens`, `lm_head`, and `norms`.
- `mlp_dense` may appear in classifier/profiler code as a component label, but it is not a supported `asym_offload_modules` selector in the current LF path.
- Selected CPU offload is active only for execution `backend="asym"`. `backend="torch"`/`asym_torch` may parse the selector for reporting compatibility, but it must not move selected frozen/base weights into CPU HostWeight owners.
- `asym_gemm/training/offload.py::adopt_host_weight()` uses `clone=False` and strict CPU-source checks for selected CPU offload. If pinning is not requested, strict mode requires storage adoption with no copy. If pinning is requested and succeeds, a single pinned CPU replacement is allowed and counted as HostWeight memory, not optimizer state.
- Current linear/router/expert/LM-head Asym offload uses `adopt_host_weight(..., pin_memory_policy="auto")`; current embedding and norm wrappers use `pin_memory_policy="none"` and are strict no-copy CPU owners. CPUAdam must treat both categories as frozen/base residency and must not create masters or optimizer state for either category.
- `asym_gemm/training/offload.py::validate_lf_offload_residency()` rejects selected frozen/base CUDA residues and any trainable non-LoRA parameter.
- `asym_gemm/integrations/lf.py::LFAsymReport` records CPU/GPU frozen/base bytes by component, so memory comparisons must separate frozen-base offload savings from CPUAdam optimizer-state savings.
- LlamaFactory `model/loader.py::_use_asym_cpu_first_load()` loads the model on CPU when any Asym CPU offload component is selected, then `model/loader.py::_move_asym_cpu_first_model_to_device()` runs after `init_adapter(...)` and calls `model.to(current_device)`. That post-adapter move is what should move trainable LoRA `nn.Parameter`s to CUDA while leaving `HostWeight` attributes CPU-resident because they are not parameters or buffers.
- Current `scripts/lf/run_lf_lora_sft.sh` defaults `ASYM_OFFLOAD_MODULES=routed_experts`; current `scripts/lf/profile_lora_lf.sh` defaults `ASYM_OFFLOAD_MODULES=all`; current `scripts/lf/profile_lora_lf_test.sh` defaults `ASYM_OFFLOAD_MODULES=routed_experts`. Stage gates must set `ASYM_OFFLOAD_MODULES` explicitly so command behavior does not depend on those script defaults.
- In the supported CPUAdamW v1 path, selected LF Asym LoRA compute weights must be CUDA `nn.Parameter` tensors after wrapping/Trainer preparation.
- HF Trainer saves and loads the optimizer through `optimizer.state_dict()` and `optimizer.load_state_dict()` as `optimizer.pt`.
- Current LlamaFactory parser rejects AsymGEMM with DeepSpeed/ZeRO and rejects explicit checkpoint resume for AsymGEMM.
- Current implementation scope is SM100 BF16 Asym kernels. Do not broaden CPUAdam milestones to FP8/FP4/other kernel families unless a later stage explicitly adds those validations.

Primary code anchors:

- DeepSpeed CPU Adam: `${DEEPSPEED_DIR}/deepspeed/ops/adam/cpu_adam.py`
- ZeRO optimizer ownership: `${DEEPSPEED_DIR}/deepspeed/runtime/zero/stage3.py`
- Asym LF parser guardrails: `${LF_DIR}/src/llamafactory/hparams/parser.py`
- LF CPU-first model load/device move: `${LF_DIR}/src/llamafactory/model/loader.py`
- LF Asym adapter conversion: `${LF_DIR}/src/llamafactory/model/adapter.py`
- LlamaFactory optimizer hook: `${LF_DIR}/src/llamafactory/train/trainer_utils.py`
- SFT trainer optimizer entry: `${LF_DIR}/src/llamafactory/train/sft/trainer.py`
- Asym LoRA parameter helpers: `${ASYM_DIR}/asym_gemm/training/lora.py`
- Asym LF wrapping/save path: `${ASYM_DIR}/asym_gemm/integrations/lf.py`
- HostWeight CPU-resident base weights: `${ASYM_DIR}/asym_gemm/training/host_weight.py`
- LF per-module offload helpers: `${ASYM_DIR}/asym_gemm/training/offload.py`
- Source memory profiler: `${ASYM_DIR}/asym_gemm/profiling/lf_trace.py`

## Non-Goals For The First CPUAdam Path

- Do not implement Adam math or a CPU Adam kernel manually.
- Do not integrate AsymGEMM into DeepSpeed ZeRO.
- Do not make LoRA compute weights CPU-fetched in the first path.
- Do not change AsymGEMM frozen base-weight kernels.
- Do not enable distributed/DDP.
- Do not relax adapter resume/checkpoint resume until the wrapper state is proven.

## Stage 0: Preflight And Guardrails

Goal: make the implementation target explicit and avoid hidden Trainer/Accelerate behavior.

Files to inspect before coding:

- `deepspeed/deepspeed/ops/adam/cpu_adam.py`
- `LlamaFactory/src/llamafactory/train/sft/trainer.py`
- `LlamaFactory/src/llamafactory/train/trainer_utils.py`
- `LlamaFactory/src/llamafactory/model/loader.py`
- `LlamaFactory/src/llamafactory/model/adapter.py`
- `transformers/src/transformers/trainer.py`
- `AsymGEMM/asym_gemm/training/lora.py`
- `AsymGEMM/asym_gemm/training/offload.py`
- `AsymGEMM/asym_gemm/integrations/lf.py`
- `AsymGEMM/asym_gemm/profiling/lf_trace.py`
- `AsymGEMM/scripts/lf/run_lf_lora_sft.sh`
- `AsymGEMM/scripts/lf/profile_lora_lf.sh`
- `AsymGEMM/scripts/lf/profile_lora_lf_test.sh`

Required checks:

1. Confirm `DeepSpeedCPUAdam.step()` still asserts CPU params when `p.grad is not None`. Tests for this must set a non-`None` grad; otherwise DeepSpeed skips the parameter before the device assertion.
2. Confirm `CustomSeq2SeqTrainer.create_optimizer()` still calls `create_custom_optimizer()` before the HF default optimizer.
3. Confirm HF Trainer still saves `self.optimizer.state_dict()` to `optimizer.pt` and reloads via `self.optimizer.load_state_dict(...)`.
4. Confirm current runtime environment has `accelerate` installed when running through the LF env. The default shell Python may not have it. After implementation, run an actual LF smoke, not only a direct unit test.
5. Confirm `parse_lf_offload_modules()` and `validate_lf_offload_residency()` are the authority for selected frozen/base component residency. CPUAdam must not reimplement per-module offload selection.
6. Confirm the scripts already pass `ASYM_OFFLOAD_MODULES` into LF runs; CPUAdam alias plumbing must preserve that pass-through and must not replace it with a CPUAdam-specific selector.
7. Confirm CPU-first Asym loading order: `load_model()` must call `init_adapter(...)` before `_move_asym_cpu_first_model_to_device(model)`, and `_move_asym_cpu_first_model_to_device()` must call `model.to(current_device)`. CPUAdam must rely on this loader/device-placement invariant instead of moving model params itself.
8. Add an explicit smoke assertion after `accelerator.prepare` by checking the wrapper at runtime: all CPU master params must remain on CPU after Trainer preparation and all selected LoRA compute params must be CUDA. Implement this as `asym_gemm/training/cpu_adam.py::AsymCPUAdamW._check_post_prepare_devices()`, called from the first `AsymCPUAdamW.step()`. This should be tested through an LF run because `Trainer` passes the optimizer through `accelerator.prepare`.

Do not proceed to Stage 2 until these checks are either verified or encoded as tests.

## Parallel Baseline Stage: DeepSpeed `zero3_cpuadam`

Goal: add a clean DeepSpeed CPUAdam baseline for comparison. This is not the Asym CPUAdamW integration path. It verifies what DeepSpeed does when DeepSpeed owns the optimizer.

Use a new backend label, `zero3_cpuadam`, instead of silently changing `zero3_offload`.

### B.1 Add DeepSpeed config

Add outside this repo:

`${LF_DIR}/examples/deepspeed/ds_z3_cpuadam_config.json`

Base it on `ds_z3_offload_config.json`, but add explicit DeepSpeed-owned optimizer creation:

```json
{
  "zero_allow_untested_optimizer": true,
  "zero_force_ds_cpu_optimizer": true,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto",
      "torch_adam": false,
      "adam_w_mode": true
    }
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu", "pin_memory": true},
    "offload_param": {"device": "cpu", "pin_memory": true},
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

Reason: local DeepSpeed creates `DeepSpeedCPUAdam` from config only when CPU optimizer offload is active and DeepSpeed owns optimizer creation.

### B.2 Add launcher/backend plumbing

Modify `scripts/lf/run_lf_lora_sft.sh`:

- Document `BACKEND=zero3_cpuadam` in the user-facing backend comment and backend error message.
- Extend `zero_deepspeed_config()` to map `zero3_cpuadam` to `ds_z3_cpuadam_config.json`.
- Add a backend normalization case beside `zero3_offload` that sets:
  - `PROFILE_BACKEND_LABEL=zero3_cpuadam`
  - `ZERO_BACKEND_LABEL=zero3_cpuadam`
  - `BACKEND=torch`
  - `TORCH_DEEPSPEED_CONFIG="$(zero_deepspeed_config zero3_cpuadam)"`
- Add `CHECK_CPUADAM=${CHECK_CPUADAM:-1}`.
- Export `ASYM_GEMM_LF_CONFIG_DEEPSPEED_CONFIG="${TORCH_DEEPSPEED_CONFIG}"` for all ZeRO backends.
- Export `ASYM_GEMM_LF_CONFIG_CPUADAM_CONFIG="${TORCH_DEEPSPEED_CONFIG}"` for `zero3_cpuadam`.
- Add a runtime check analogous to SuperOffload. Accept either:
  - `source_profile.json["cpuadam"]["runtime_verified"] == true`
  - train log marker showing `DeepSpeedCPUAdam`
- Add `scripts/lf/check_deepspeed_cpuadam_run.py` with `summarize(profile_json, train_log, require_enabled)` and `main()`. Do not reuse `scripts/lf/check_superoffload_run.py::summarize()` because it hardcodes `SuperOffloadOptimizer_Stage3`.

Modify `scripts/lf/profile_lora_lf.sh`:

- Document `zero3_cpuadam|recomp` in `--backend-specs`.
- Add it to `backend_gpu_count()`, `zero_deepspeed_config()`, `is_zero_backend()`, `is_policy_independent_backend()`, `backend_label()`, and `append_backend_spec()`.
- Add it to all backend validation error messages in those functions.
- Pass `CHECK_CPUADAM` through `run_env`.
- Include the CPUAdam label in both `job_root_path()` and `run_id` so profiles do not collide with existing `asym` or `zero3_offload` outputs.
- Keep defaults unchanged until runtime verification is stable.

Modify `scripts/lf/run_lf_profiled_train.py`:

- In `_install_deepspeed_optimizer_capture_hook()`, capture both the ZeRO wrapper and the underlying optimizer. Use `self.optimizer.__class__.__name__` for the wrapper, `self.basic_optimizer.__class__.__name__` when the `DeepSpeedEngine` exposes it, and fall back to `self.optimizer.optimizer.__class__.__name__` when the ZeRO wrapper exposes the inner optimizer there. For ZeRO, the wrapper may be `DeepSpeedZeroOptimizer_Stage3` while the actual CPU optimizer is `DeepSpeedCPUAdam`.
- Record general `--deepspeed` config path, not only `superoffload_config`, in `_config_from_args()` / `_env_config()`.
- Emit a `cpuadam` JSON object:

```json
{
  "enabled": true,
  "runtime_verified": true,
  "basic_optimizer_class": "DeepSpeedCPUAdam",
  "optimizer_wrapper_class": "DeepSpeedZeroOptimizer_Stage3",
  "offload_optimizer_device": "cpu",
  "offload_param_device": "cpu",
  "deepspeed_config": "..."
}
```

Modify postprocessing/plotting if this backend is included in plots:

- `scripts/lf/postprocess_lf_profile_artifacts.py`: update `_source_summary_markdown()`, `_write_source_artifacts()`, and `_write_profile_csv_artifacts()` to add a DeepSpeed optimizer table/CSV for `cpuadam` and include the runtime verification fields in summary outputs.
- `scripts/plotting/plot_activation_recompute_sweep.py`: add `zero3_cpuadam` to `BACKENDS` and `BACKEND_MARKERS`; update `parse_result_dir()` compatibility paths and `row_from_result_dir()` so parsed rows keep the backend label.
- Add tests that postprocess emits CPUAdam summary/CSV fields from a synthetic profile artifact.
- Add tests that plotting accepts/parses `zero3_cpuadam`, `asym_cpuadamwtorch`, and `asym_cpuadamwds`.

### B.3 Baseline tests

Extend script tests:

- Fake `ds_z3_cpuadam_config.json` in `tests/lf/test_superoffload_backend_scripts.py`.
- Add a profile dry-run test for `BACKEND_SPECS='zero3_cpuadam|recomp'` and assert it schedules `BACKEND=zero3_cpuadam` for `run_lf_lora_sft.sh`.
- Add a separate fake-launcher or stubbed-command test for `run_lf_lora_sft.sh` itself and assert it computes the correct `--deepspeed ds_z3_cpuadam_config.json` path and emits no Asym/KT flags.
- Assert runtime check accepts a synthetic `DeepSpeedCPUAdam` marker.
- Assert the existing SuperOffload checker still rejects missing SuperOffload markers; CPUAdam verification must be separate.

Tiny checks:

```bash
bash -n scripts/lf/run_lf_lora_sft.sh
bash -n scripts/lf/profile_lora_lf.sh
bash -n scripts/lf/profile_lora_lf_test.sh
python -m py_compile scripts/lf/run_lf_profiled_train.py scripts/lf/postprocess_lf_profile_artifacts.py scripts/lf/check_deepspeed_cpuadam_run.py
python -m pytest -q tests/lf/test_superoffload_backend_scripts.py -k 'cpuadam or zero3_offload'
```

## Stage 1: Add Public Flags And Validation

Goal: expose CPU AdamW only for AsymGEMM LoRA-SFT, with a safe default-off behavior.

### 1.1 Add finetuning arguments

Modify:

`${LF_DIR}/src/llamafactory/hparams/finetuning_args.py`

Add fields to `FinetuningArguments`.

Canonical Python/CLI names must include `adamw`, because this path is AdamW mode:

```python
use_asym_cpu_adamw: bool = field(
    default=False,
    metadata={"help": "Use AsymGEMM CPU-master AdamW for LoRA params."},
)
asym_cpu_adamw_backend: Literal["torch", "deepspeed"] = field(
    default="deepspeed",
    metadata={"help": "CPU AdamW backend for AsymGEMM LoRA params."},
)
asym_cpu_adamw_pin_memory: bool = field(
    default=True,
    metadata={"help": "Pin CPU master and grad-copy buffers when CUDA is available."},
)
asym_cpu_adamw_fp32_master: bool = field(
    default=True,
    metadata={"help": "Keep CPU master LoRA params in fp32."},
)
```

Initial implementation must only support `asym_cpu_adamw_fp32_master=True`. Keep the flag to make the precision decision explicit, but reject `False` until a bf16-master path is validated.

### 1.2 Validate in parser

Modify:

`${LF_DIR}/src/llamafactory/hparams/parser.py`

Inside the existing `if model_args.use_asym_gemm:` block:

- Allow `finetuning_args.use_asym_cpu_adamw` only when:
  - `stage == "sft"`
  - `do_train == True`
  - `finetuning_type == "lora"`
  - `pure_bf16 == True`
  - `training_args.deepspeed is None`
  - `not is_deepspeed_zero3_enabled()`
  - `training_args.parallel_mode == ParallelMode.NOT_PARALLEL`
- Reject `asym_cpu_adamw_backend` outside `{"torch", "deepspeed"}`.
- Reject `asym_cpu_adamw_fp32_master=False` in the first implementation.
- Reject non-default `training_args.optim` values, because this flag replaces the Trainer optimizer. Start by allowing only the normal AdamW default used by the current LF/HF path; broaden later only with a test.
  Do not hard-code `"adamw_torch"`: compare against the current `TrainingArguments` dataclass default or the installed Transformers `OptimizerNames` default, which may be `adamw_torch_fused` on newer torch/Transformers versions.
- Reject `training_args.load_best_model_at_end=True` until Stage 4 adds an Asym-aware best-checkpoint load path.
- Reject combinations with other custom optimizer paths:
  - `use_galore`
  - `use_apollo`
  - `use_badam`
  - `use_adam_mini`
  - `use_muon`
  - `use_mca`
  - `use_hyper_parallel`
  - `loraplus_lr_ratio is not None`

Outside the Asym block, reject `use_asym_cpu_adamw=True` when `model_args.use_asym_gemm` is false.

Add a CPUAdam-specific parallelism guard independent of `model_args.asym_backend`. The current Asym DDP guard only applies to `asym_backend == "asym"` and misses single-process multi-GPU/DataParallel. CPUAdamW must require `training_args.parallel_mode == ParallelMode.NOT_PARALLEL`; reject `NOT_DISTRIBUTED`, `DISTRIBUTED`, TPU, SageMaker, and any other non-single-process/single-device mode.

Keep checkpoint resume rejected in Stage 1. This requires two guards:

1. Reject explicit `training_args.resume_from_checkpoint is not None` in the Asym block, as current Asym does.
2. Prevent LlamaFactory's later output-dir auto-detection from setting `training_args.resume_from_checkpoint`. Do this by modifying the existing `can_resume_from_checkpoint` block near the auto-detect code, or by adding a second hard rejection immediately after the auto-detect block. Do not set `can_resume_from_checkpoint=False` only inside the earlier Asym block; the later generic assignment would overwrite it. Do not let an existing `output_dir/checkpoint-*` silently enable resume before Stage 4.

### 1.3 Add launcher plumbing

Modify:

`${ASYM_DIR}/scripts/lf/run_lf_lora_sft.sh`

Add env defaults near the other Asym settings. These remain the LF/Trainer plumbing values, but profiling and user-facing benchmark selection must use the explicit backend aliases below instead of plain `asym` plus a global flag.

```bash
USE_ASYM_CPU_ADAMW=${USE_ASYM_CPU_ADAMW:-false}
ASYM_CPU_ADAMW_BACKEND=${ASYM_CPU_ADAMW_BACKEND:-deepspeed}
ASYM_CPU_ADAMW_PIN_MEMORY=${ASYM_CPU_ADAMW_PIN_MEMORY:-true}
ASYM_CPU_ADAMW_FP32_MASTER=${ASYM_CPU_ADAMW_FP32_MASTER:-true}
```

Add two public backend aliases:

```text
asym_cpuadamwtorch -> execution backend asym, USE_ASYM_CPU_ADAMW=true, ASYM_CPU_ADAMW_BACKEND=torch
asym_cpuadamwds    -> execution backend asym, USE_ASYM_CPU_ADAMW=true, ASYM_CPU_ADAMW_BACKEND=deepspeed
```

Normalize these aliases before `RUN_BACKEND_LABEL` and `DEFAULT_RUN_ID` are computed:

- `BACKEND=asym_cpuadamwtorch`: set `PROFILE_BACKEND_LABEL=asym_cpuadamwtorch`, `RUN_BACKEND_LABEL=asym_cpuadamwtorch`, `USE_ASYM_CPU_ADAMW=true`, `ASYM_CPU_ADAMW_BACKEND=torch`, then normalize the execution backend to `BACKEND=asym` before building LF args.
- `BACKEND=asym_cpuadamwds`: set `PROFILE_BACKEND_LABEL=asym_cpuadamwds`, `RUN_BACKEND_LABEL=asym_cpuadamwds`, `USE_ASYM_CPU_ADAMW=true`, `ASYM_CPU_ADAMW_BACKEND=deepspeed`, then normalize the execution backend to `BACKEND=asym` before building LF args.

Plain `BACKEND=asym` and `BACKEND=asym_torch` remain non-CPUAdam Asym baselines. Hard reject any direct non-alias run with `USE_ASYM_CPU_ADAMW=true`, including `BACKEND=asym USE_ASYM_CPU_ADAMW=true`, `BACKEND=asym_torch USE_ASYM_CPU_ADAMW=true`, and all ZeRO/SuperOffload/KT/torch backends. The error must say to use `BACKEND=asym_cpuadamwtorch` or `BACKEND=asym_cpuadamwds`. New tests and profile commands must use the explicit aliases.

Validate `ASYM_CPU_ADAMW_BACKEND` is `torch` or `deepspeed` after alias normalization. For the `asym_cpuadamwds` alias, keep the user-facing label `ds`; only the LF flag value should be `deepspeed`.

When normalized execution `BACKEND=asym` or `BACKEND=asym_torch`, append:

```bash
CMD_ARGS+=(--use_asym_cpu_adamw "${USE_ASYM_CPU_ADAMW}")
CMD_ARGS+=(--asym_cpu_adamw_backend "${ASYM_CPU_ADAMW_BACKEND}")
CMD_ARGS+=(--asym_cpu_adamw_pin_memory "${ASYM_CPU_ADAMW_PIN_MEMORY}")
CMD_ARGS+=(--asym_cpu_adamw_fp32_master "${ASYM_CPU_ADAMW_FP32_MASTER}")
```

For `BACKEND=asym` and `BACKEND=asym_torch` reached directly, `USE_ASYM_CPU_ADAMW` must already be `false` because the direct true case is rejected before command construction. The explicit false flag is useful for parser/profile visibility and for proving the baseline path did not enable CPUAdamW.

Do not pass these flags for `torch`, `zero*`, `superoffload`, or `kt_*` backends.

### 1.4 Add profile launcher plumbing

Modify:

`${ASYM_DIR}/scripts/lf/profile_lora_lf.sh`

Also modify:

`${ASYM_DIR}/scripts/lf/profile_lora_lf_test.sh`

That script is a validation harness copy of `profile_lora_lf.sh`. Either mirror every CPUAdamW option/result-label/filter change into it, or replace it with a thin wrapper that execs the main profile script. Do not let the two profile scripts drift after this stage.

Add CLI/env controls:

```bash
USE_ASYM_CPU_ADAMW=${USE_ASYM_CPU_ADAMW:-false}
ASYM_CPU_ADAMW_BACKEND=${ASYM_CPU_ADAMW_BACKEND:-deepspeed}
ASYM_CPU_ADAMW_PIN_MEMORY=${ASYM_CPU_ADAMW_PIN_MEMORY:-true}
ASYM_CPU_ADAMW_FP32_MASTER=${ASYM_CPU_ADAMW_FP32_MASTER:-true}
```

Add options:

- `--use-asym-cpu-adamw`
- `--asym-cpu-adamw-backend`
- `--asym-cpu-adamw-pin-memory`
- `--asym-cpu-adamw-fp32-master`

These options are low-level forwarding controls only. They must not be the primary user-facing way to select CPUAdam profiling jobs. New profile commands and tests must select CPUAdam with `BACKEND_SPECS=asym_cpuadamwtorch|...` or `BACKEND_SPECS=asym_cpuadamwds|...`.

Pass the per-job values through `run_env` so `run_lf_lora_sft.sh` receives them.

Concrete shell-script edit points for both `profile_lora_lf.sh` and `profile_lora_lf_test.sh`:

1. `usage()`: document env vars and CLI options.
2. Top-level defaults: define the four `ASYM_CPU_ADAMW` env defaults near other workload/profile defaults.
3. CLI `case` parser: accept the four new `--asym-cpu-adamw...` options and normalize booleans with `bool_value()`.
4. `backend_label()` and `append_backend_spec()`: accept `asym_cpuadamwtorch` and `asym_cpuadamwds` as first-class user-facing backend specs, alongside `asym`.
5. `backend_gpu_count()`: treat both aliases like `asym` for GPU-count validation.
6. `selected_has_asym` detection: set it for `asym_cpuadamwtorch` and `asym_cpuadamwds` so dataset preparation, router-mode handling, and Asym-specific checks use the same path as `asym`.
7. `is_policy_independent_backend()`: do not include the aliases. CPUAdam Asym jobs still support expert-policy and router-mode labels like normal Asym.
8. Add a helper such as:

```bash
cpuadam_backend_for_label() {
  case "$1" in
    asym_cpuadamwtorch) printf 'torch\n' ;;
    asym_cpuadamwds) printf 'deepspeed\n' ;;
    *) return 1 ;;
  esac
}
```

9. `job_root_path()`: use the public backend label directly, so the directory becomes `asym_cpuadamwtorch__source__...` or `asym_cpuadamwds__source__...` when CPUAdamW is enabled.
10. `run_job()`: keep separate values and pass all four CPUAdamW env vars:

```bash
profile_backend_label="${backend}"
job_use_asym_cpu_adamw=false
job_asym_cpu_adamw_backend="${ASYM_CPU_ADAMW_BACKEND}"
if cpuadam_backend="$(cpuadam_backend_for_label "${backend}")"; then
  job_use_asym_cpu_adamw=true
  job_asym_cpu_adamw_backend="${cpuadam_backend}"
fi
```

Use `BACKEND="${profile_backend_label}"` in `run_env`; do not replace it with execution `asym` inside the profile script. `run_lf_lora_sft.sh` must implement the aliases and normalize them internally before building LF args. `command.txt`, `jobs.tsv`, job roots, and plot rows must expose `asym_cpuadamwtorch` or `asym_cpuadamwds`, not plain `asym`.
Use `job_use_asym_cpu_adamw` for `USE_ASYM_CPU_ADAMW` and `job_asym_cpu_adamw_backend` for `ASYM_CPU_ADAMW_BACKEND`.
11. `append_backend_filters()`: filters should use the public backend labels selected in `BACKEND_SPECS`, including `asym_cpuadamwtorch` and `asym_cpuadamwds`.

Mixed comparison sweeps are valid. For example, `BACKEND_SPECS=asym_cpuadamwtorch|recomp,zero3_offload|recomp` should run the CPUAdam Asym job and the ZeRO job without leaking CPUAdamW flags into ZeRO.

Include CPUAdam in result naming to avoid overwriting old Asym profiles.

Required CPUAdam backend/result labels:

```text
asym_cpuadamwtorch
asym_cpuadamwds
```

Modify `scripts/plotting/plot_activation_recompute_sweep.py` exactly enough to accept the derived labels:

- Add `asym_cpuadamwtorch` and `asym_cpuadamwds` to `BACKENDS`.
- Add markers for both labels to `BACKEND_MARKERS`.
- In `row_from_result_dir()`, treat `{"asym", "asym_torch", "asym_cpuadamwtorch", "asym_cpuadamwds"}` as Asym-family labels so expert-policy labels are preserved instead of canonicalized to `none`.
- Keep these labels as public backend labels in the profile scripts and plotting. They may normalize internally to execution `asym`, but they must not disappear from output directories or profile config.

### 1.5 Stage 1 tests

Add:

`${ASYM_DIR}/tests/lf/test_asym_cpu_adamw_args.py`

Use the fake LF/fake launcher pattern from `tests/lf/test_superoffload_backend_scripts.py`.

Required tests:

1. `BACKEND=asym_cpuadamwtorch` makes `run_lf_lora_sft.sh` pass:
   - `--use_asym_gemm true`
   - `--use_asym_cpu_adamw true`
   - `--asym_cpu_adamw_backend torch`
   - `--asym_cpu_adamw_pin_memory ...`
   - `--asym_cpu_adamw_fp32_master ...`
   - no `--deepspeed`
2. `BACKEND=asym USE_ASYM_CPU_ADAMW=false` still passes an explicit `--use_asym_cpu_adamw false` and remains otherwise identical to the existing Asym command shape.
3. `BACKEND=asym_cpuadamwds` makes `run_lf_lora_sft.sh` pass `--asym_cpu_adamw_backend deepspeed` while using the public label `asym_cpuadamwds`.
4. Direct `run_lf_lora_sft.sh` with `BACKEND=zero3_offload USE_ASYM_CPU_ADAMW=true` must hard reject before building the LF command.
5. Direct `run_lf_lora_sft.sh` with `BACKEND=asym USE_ASYM_CPU_ADAMW=true` or `BACKEND=asym_torch USE_ASYM_CPU_ADAMW=true` must hard reject so CPUAdam runs are not hidden under plain Asym-family labels.
6. `profile_lora_lf.sh` dry-run with `BACKEND_SPECS=asym_cpuadamwtorch|recomp` writes `command.txt` and jobs under `asym_cpuadamwtorch`, with CPUAdamW enabled and `ASYM_CPU_ADAMW_BACKEND=torch`.
7. `profile_lora_lf.sh` mixed dry-run with `BACKEND_SPECS=asym_cpuadamwtorch|recomp,zero3_offload|recomp` writes the CPUAdam Asym command under `asym_cpuadamwtorch`, but writes the ZeRO command with `USE_ASYM_CPU_ADAMW=false` and `PROFILE_BACKEND_LABEL=zero3_offload`.
8. `profile_lora_lf_test.sh` has the same CPUAdamW behavior as `profile_lora_lf.sh`. If it is converted to a thin wrapper, test that it execs the main script and preserves `--output-root`.
9. Plot parser tests assert `asym_cpuadamwtorch` and `asym_cpuadamwds` result directories are accepted and keep expert-policy metadata.

Add parser tests in the LlamaFactory tree, exercising `llamafactory.hparams.parser.get_train_args()`:

`${LF_DIR}/tests/hparams/test_asym_cpu_adamw_args.py`

Required parser tests:

1. Accept the exact supported combination: AsymGEMM, SFT, LoRA, train, pure bf16, no DeepSpeed, single process, `use_asym_cpu_adamw=True`.
2. Reject CPUAdamW without `use_asym_gemm`.
3. Reject DeepSpeed/ZeRO, DDP/non-single-device parallel modes, non-LoRA finetuning, non-SFT stages, `asym_cpu_adamw_fp32_master=False`, custom optimizer flags, LoRA+, explicit resume, auto-resume from an existing `output_dir/checkpoint-*`, and `load_best_model_at_end=True`.

## Stage 2: Implement The CPU-Master Optimizer Wrapper

Goal: return a real optimizer from LlamaFactory that keeps CPU master weights and CPU optimizer state while the model keeps CUDA LoRA parameters.

### 2.1 Add named LoRA parameter helper

Modify:

`${ASYM_DIR}/asym_gemm/training/lora.py`

Add:

```python
def named_lora_parameters(model: nn.Module, *, adapter_name: str = "default") -> list[tuple[str, torch.nn.Parameter]]:
    try:
        named_params = model.named_parameters(remove_duplicate=False)
    except TypeError:
        named_params = model.named_parameters()
    return [
        (name, param)
        for name, param in named_params
        if _is_lora_parameter_name(name, adapter_name=adapter_name)
    ]
```

Keep `lora_parameters()` unchanged, but make it call `named_lora_parameters()` to avoid duplicated matching. `remove_duplicate=False` is required when available so tied/shared LoRA aliases are visible to the CPUAdam wrapper; the wrapper still creates exactly one CPU master after deduplicating by parameter object or identical tensor view key.

The helper must use the existing `_is_lora_parameter_name()` predicate. That predicate covers both PEFT-style dense adapter names such as `.lora_A.default.weight` / `.lora_B.default.weight` and Asym packed expert names such as `gate_lora_A`, `up_lora_B`, and `down_lora_A`. Do not implement a PEFT-only selector.

Export this helper from:

`${ASYM_DIR}/asym_gemm/training/__init__.py`

if that file already exports training helpers.

### 2.2 Add optimizer module

Add:

`${ASYM_DIR}/asym_gemm/training/cpu_adam.py`

Export `AsymCPUAdamW` from:

`${ASYM_DIR}/asym_gemm/training/__init__.py`

Do not export it from top-level `asym_gemm` in the first implementation.

Define:

```python
class AsymCPUAdamW(torch.optim.Optimizer):
    ...
```

Constructor signature:

```python
def __init__(
    self,
    named_params: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    backend: Literal["torch", "deepspeed"] = "deepspeed",
    pin_memory: bool = True,
    fp32_master: bool = True,
) -> None:
```

Required constructor behavior:

1. Filter to params with `requires_grad=True`.
2. Fail if no parameters are selected.
3. Fail if any selected parameter is not CUDA when CUDA is available. The first path is for GPU LoRA compute params. If this triggers for a per-module offload configuration, the error should say CPU-resident trainable LoRA is Stage 7 and is not supported by CPUAdamW v1, and also mention that LlamaFactory's post-adapter Asym CPU-first device move should have moved trainable LoRA params to CUDA before optimizer creation.
4. Fail unless `fp32_master=True`.
5. Deduplicate selected params before constructing CPU masters. Keep a primary name and alias-name list for diagnostics/checkpoint validation, but create exactly one CPU master for each unique trainable LoRA `nn.Parameter`. The dedupe key must be parameter object identity or an identical tensor view key `(device, storage data ptr, storage offset, shape, stride, dtype)`, not only the storage data pointer.
6. Reject any selected name that is not a LoRA name (`"lora_"`, `.lora_A.`, or `.lora_B.`). `HostWeight` and frozen wrappers should never appear here because they are not trainable `nn.Parameter`s; if they do, raise before any CPU copy.
7. Build wrapper param groups around the original unique CUDA params by calling `super().__init__(...)`.
8. For every unique CUDA LoRA param, create one CPU master `nn.Parameter`:

```python
master = torch.nn.Parameter(
    gpu_param.detach().to(device="cpu", dtype=torch.float32).contiguous(),
    requires_grad=True,
)
```

If `pin_memory=True` and CUDA is available, pin `master.data` after the CPU copy. If pinning fails, keep pageable CPU memory and record the failure string for diagnostics; do not silently move to CUDA.

9. Build exactly one wrapper param group and one matching inner optimizer param group in v1. The group contains all selected CPU master params and uses the flat hyperparameters passed to the constructor from `TrainingArguments`. "Preserve group hyperparameters" means that later scheduler mutations to wrapper `param_groups[0]` are copied into the matching inner optimizer group immediately before every `step()`. Do not claim multi-group support unless the constructor is changed to accept named parameter groups and tests cover it.
10. Backend behavior:
   - `backend="torch"`: use `torch.optim.AdamW`.
   - `backend="deepspeed"`: import `deepspeed.ops.adam.DeepSpeedCPUAdam` lazily and instantiate it with `adamw_mode=True` and `fp32_optimizer_states=True`.
11. Keep `self.state` keyed by the original CUDA model parameters, not by CPU master params. This is important for HF/profiler compatibility. Each visible state entry should contain CPU tensors such as `cpu_master`, `exp_avg`, and `exp_avg_sq` after the inner optimizer initializes them. DeepSpeed CPUAdam initializes moment state lazily only for parameters with grads, so visible state entries may contain only `cpu_master` until a parameter has stepped. These entries may reference the same tensor storage as the inner optimizer state; memory attribution deduplicates by storage pointer.
12. Store metadata:
    - `param_names`
    - `alias_param_names` for tied/shared LoRA params, if any
    - model dtype
    - master dtype
    - backend
    - pinning status

Object ids are process-local diagnostics only. Do not use CUDA or CPU parameter object ids for checkpoint matching. Checkpoint/load matching must use `param_names`, count, shape, dtype, and ordering.

Do not use `amsgrad`; DeepSpeed CPUAdam accepts the field but does not support the algorithm.

Do not call `model.to(...)`, `param.to("cuda")`, or `param.data = ...` from the optimizer constructor/factory to repair device placement. That would hide a loader/Trainer placement bug and can introduce extra full-parameter copies. Device placement belongs to `load_model()`/Trainer; CPUAdam only validates the final invariant and creates CPU masters for already-CUDA LoRA params.

### 2.3 Param mapping data structure

Inside `cpu_adam.py`, add a small dataclass:

```python
@dataclass
class _ParamMapping:
    name: str
    aliases: tuple[str, ...]
    cuda_param: torch.nn.Parameter
    cpu_param: torch.nn.Parameter
    grad_buffer: torch.Tensor | None
    model_dtype: torch.dtype
    master_dtype: torch.dtype
    last_had_grad: bool = False
```

Use this mapping everywhere. Do not rely on matching by tensor order outside this mapping.

### 2.4 Step algorithm

Implement `AsymCPUAdamW.step()`:

```text
for each wrapper param group:
  copy current lr/betas/eps/weight_decay from wrapper group to matching inner group

for each mapping:
  if cuda_param.grad is None:
    cpu_param.grad = None
    mark mapping as not updated this step
    continue

  grad_cpu = cuda_param.grad.detach().to(
      device="cpu",
      dtype=cpu_param.dtype,
      non_blocking=False,
  ).contiguous()

  if grad_buffer already exists:
    copy into the persistent grad_buffer with the same shape/dtype
    cpu_param.grad = grad_buffer
  else:
    cpu_param.grad = grad_cpu
  mark mapping as updated this step

if at least one mapping was updated:
  inner_optimizer.step()

for each mapping:
  if mapping was not updated this step:
    skip copyback
    continue
  if cpu_param.dtype == cuda_param.dtype:
    cuda_param.data.copy_(cpu_param.data, non_blocking=cpu_param.data.is_pinned())
  else:
    copy CPU master into CUDA param with dtype conversion and without retaining a CUDA temporary
```

For the first fp32-master/bf16-model path, avoid `cpu_param.data.to(device=cuda_param.device, ...)` because that materializes a CUDA temporary and can add transient HBM pressure. Use a copy path that casts during CPU-to-CUDA copy without retaining a CUDA temporary; if PyTorch cannot do this directly for the dtype pair, make the temporary allocation explicit in timing/memory diagnostics.

Rules:

- Use `torch.no_grad()`.
- Require CPU master tensors to be contiguous.
- Require copied CPU grad tensors to be contiguous. Persistent pinned grad buffers are optional in Stage 2; if not implemented yet, allocating `grad_cpu` each step is acceptable.
- Require `cpu_param.grad.dtype == cpu_param.dtype`; for the first fp32-master path, copy grads to fp32.
- Do not mutate CUDA param shapes or replace the model parameter objects.
- Do not leave CPU grads pointing to CUDA tensors.
- Do not call `.cuda()` on CPU master params.
- Do not call `DeepSpeedCPUAdam` with CUDA params.
- Do not copy back CPU masters for LoRA params with no grad in the current step. AdamW in both torch and DeepSpeed only updates params with a non-`None` grad, including decoupled weight decay, so a no-grad param has no update to copy back.
- After the inner optimizer initializes moments, refresh the visible `self.state[cuda_param]` entries for parameters whose inner optimizer state exists so profiler/checkpoint code can see CPU optimizer state associated with the original model param.
- Return closure loss if provided, matching normal optimizer behavior.

### 2.5 Zero grad behavior

Implement `zero_grad(set_to_none: bool = True)`:

- Clear CUDA LoRA grads using normal optimizer behavior.
- Clear CPU master grads.
- Do not clear CPU optimizer moment state.

### 2.6 State dict behavior

Override `state_dict()` and `load_state_dict()`.

`state_dict()` must return only CPU-safe tensors and Python scalars:

```python
{
    "format": "asym_cpu_adamw_v1",
    "backend": self.backend,
    "pin_memory": self.pin_memory,
    "fp32_master": self.fp32_master,
    "param_names": [...],
    "alias_param_names": [[...], ...],
    "param_groups": sanitized_wrapper_param_groups,
    "cpu_master_params": [cpu_param.detach().cpu() ...],
    "inner_optimizer": self.inner_optimizer.state_dict(),
}
```

`sanitized_wrapper_param_groups` must not contain `torch.nn.Parameter` objects or CUDA tensors. Store stable parameter references as integer indices plus `param_names`, or as names directly. Add a recursive assertion in tests that every tensor in the returned state dict is on CPU.

`load_state_dict()` must:

1. Validate `format == "asym_cpu_adamw_v1"`.
2. Validate `state["backend"] == self.backend`. Reject torch<->DeepSpeed backend mismatch unless an explicit conversion path is implemented and tested.
3. Validate param count, names, shapes, and dtypes.
4. Load CPU master tensors onto CPU, not onto the current CUDA device. This matters because external wrappers may load optimizer state with a map location.
5. Load the inner optimizer state.
6. Validate loaded inner optimizer tensors when present:
   - `exp_avg` and `exp_avg_sq` shapes match the CPU master param
   - moment tensors are on CPU
   - moment tensors are fp32 for the first `fp32_master=True` implementation
7. Rebuild and refresh visible `self.state[cuda_param]` entries from the loaded inner optimizer state immediately after `inner_optimizer.load_state_dict(...)`. This must happen before the next `step()` because resumed DeepSpeed moment state may already exist.
8. Reapply wrapper param-group hyperparameters.
9. Copy CPU master weights back into the CUDA model params immediately after load.
10. Rebuild or re-pin grad buffers if needed.

This load path must tolerate saved tensors being loaded on CUDA by an external wrapper by explicitly moving them back to CPU.

### 2.7 Diagnostics API

Add methods/properties:

```python
def asym_cpu_master_params(self) -> list[torch.nn.Parameter]
def asym_cpu_param_name_map(self) -> dict[int, str]
def asym_cpu_adamw_summary(self) -> dict[str, Any]
```

`asym_cpu_param_name_map()` must be keyed by `id(cpu_param)`, because profiling uses it to label CPU master params and any inner optimizer state keyed by CPU master params. If CUDA-param names are also needed, add a separate `asym_cuda_param_name_map()` keyed by `id(cuda_param)`.

The summary should include:

- backend
- param_count
- param_numel
- cpu_master_bytes
- pinned_cpu_master_bytes
- optimizer_state_cpu_bytes if initialized
- all_masters_on_cpu
- all_cuda_params_on_cuda
- last_step_grad_param_count
- last_step_copyback_param_count
- skipped_copyback_no_grad_param_count

These methods are used by profiling and tests.

### 2.8 Unit tests for wrapper only

Add:

`${ASYM_DIR}/tests/training/test_asym_cpu_adamw.py`

Tests:

1. CPU-master wrapper with `backend="torch"` updates a CUDA LoRA param and keeps CPU masters on CPU.
2. `state_dict/load_state_dict` restores CPU masters and CUDA compute params for the torch backend.
3. Scheduler-like LR mutation on wrapper `param_groups[0]["lr"]` propagates to the inner optimizer before step.
4. If DeepSpeed is importable and CPUAdam extension loads, `backend="deepspeed"` runs one tiny step; otherwise skip with a clear pytest skip reason.
5. After a fake "external load" where saved CPU master tensors are moved to CUDA before `load_state_dict`, the wrapper moves them back to CPU.
6. DeepSpeed backend parity: compare one deterministic step against `torch.optim.AdamW` on CPU with nonzero `weight_decay`, same lr/betas/eps, and fp32 params/grads. This catches accidentally constructing DeepSpeed CPUAdam with Adam mode instead of AdamW mode.
7. DeepSpeed CUDA-param rejection test: set a non-`None` CUDA grad before calling raw `DeepSpeedCPUAdam.step()`; without a grad, DeepSpeed skips the parameter before the CPU-device assertion.
8. If the DeepSpeed CPUAdam extension is available, `state_dict/load_state_dict` also restores the DeepSpeed backend and visible CUDA-keyed state; assert `inner_optimizer.adam_w_mode is True` and `inner_optimizer.fp32_optimizer_states is True`.
9. Duplicate/tied LoRA params produce one CPU master, preserve alias names in diagnostics, and copy back only once.
10. A no-grad LoRA param has `cpu_param.grad is None`, does not trigger CPU master copyback, and increments `skipped_copyback_no_grad_param_count`.
11. Two different LoRA `nn.Parameter` objects that share the same allocation but have different storage offsets, shapes, or strides produce separate CPU masters. Only exact aliases collapse to one master.
12. Passing a non-LoRA trainable parameter or any CPU trainable LoRA compute param raises before creating optimizer state.
13. Tiny modules containing `HostWeight`/`AsymFrozenLinear`, `AsymFrozenEmbedding`, `AsymFrozenLayerNorm`, or `AsymFrozenRMSNorm` plus LoRA params create CPU masters only for LoRA params and never for frozen/base CPU owners.

If CUDA is unavailable, skip runtime CUDA tests but keep pure validation tests.

## Stage 3: Integrate With LlamaFactory Optimizer Creation

Goal: use the wrapper in actual AsymGEMM SFT without changing Trainer internals.

### 3.1 Add optimizer factory

Modify:

`${LF_DIR}/src/llamafactory/train/trainer_utils.py`

Add:

```python
def _create_asym_cpu_adamw_optimizer(
    model: "PreTrainedModel",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> "torch.optim.Optimizer":
```

Algorithm:

1. Import concrete symbols lazily from their implementation modules:

```python
from asym_gemm.training.lora import named_lora_parameters
from asym_gemm.training.cpu_adam import AsymCPUAdamW
```

Do not import these from top-level `asym_gemm`; the package root currently exposes extension/version symbols, not training helpers. Export `named_lora_parameters` and `AsymCPUAdamW` from `asym_gemm.training.__init__` for local tests/public convenience, but the LF hook should still use concrete module imports to avoid package-root side effects.

2. Collect named LoRA params from the already-wrapped model. This must run after AsymGEMM/PEFT conversion, not before, so routed expert LoRA, dense LoRA, and shared-expert LoRA names are all visible.
3. Build `trainable_lora` by filtering the named LoRA params to `param.requires_grad`.
4. Deduplicate `trainable_lora` by parameter object id for equality checks and wrapper construction. Preserve alias names for diagnostics.
5. Collect all trainable params from `model.named_parameters()`.
6. Validate exact equality:

```text
trainable parameter object ids == trainable LoRA parameter object ids
```

If any trainable non-LoRA param exists, raise a hard error.

7. Treat `model._asym_offload_modules` only as a diagnostic/profile field. Do not use it to select optimizer params; frozen/base offload is validated by `validate_lf_offload_residency()`, while CPUAdam selects trainable LoRA params only.
8. Validate every selected trainable LoRA param is CUDA before constructing `AsymCPUAdamW`. If not, raise and name the parameter plus the expected loader invariant: CPU-first Asym load must run `_move_asym_cpu_first_model_to_device()` after `init_adapter(...)`. Do not move the parameter here.
9. Instantiate:

```python
AsymCPUAdamW(
    named_lora,
    lr=training_args.learning_rate,
    betas=(training_args.adam_beta1, training_args.adam_beta2),
    eps=training_args.adam_epsilon,
    weight_decay=training_args.weight_decay,
    backend=finetuning_args.asym_cpu_adamw_backend,
    pin_memory=finetuning_args.asym_cpu_adamw_pin_memory,
    fp32_master=finetuning_args.asym_cpu_adamw_fp32_master,
)
```

10. Log the summary on rank 0.

### 3.2 Hook into `create_custom_optimizer`

Modify `create_custom_optimizer(...)`:

Put this branch before other custom optimizer branches:

```python
if finetuning_args.use_asym_cpu_adamw:
    return _create_asym_cpu_adamw_optimizer(model, training_args, finetuning_args)
```

This must be before GaLore/APOLLO/BAdam/Adam-mini/Muon. Parser validation should make combinations impossible, but order should still be deterministic.

### 3.3 Verify after Trainer preparation

Because HF Trainer calls `accelerator.prepare(self.model, self.optimizer)`, add a lightweight runtime check after prepare. The least invasive place is an optimizer method invoked from the first `step()`:

```python
if not self._post_prepare_checked:
    if not all(mapping.cpu_param.device.type == "cpu" for mapping in self._mappings):
        raise RuntimeError("accelerator.prepare moved AsymCPUAdamW CPU master params off CPU")
    if not all(mapping.cuda_param.device.type == "cuda" for mapping in self._mappings):
        raise RuntimeError("accelerator.prepare moved AsymCPUAdamW CUDA LoRA params off CUDA")
    self._post_prepare_checked = True
```

If this fails, the error message must say that `accelerator.prepare` moved CPU masters and that `load_state_dict` or wrapper construction needs adjustment.

Do not patch HF Trainer unless this check fails in a real LF smoke and there is no wrapper-only fix.

### 3.4 Stage 3 tests

Add:

`${ASYM_DIR}/tests/lf/test_asym_cpu_adamw_lf_integration.py`

Required tests:

1. Import `create_custom_optimizer()` from LlamaFactory with `PYTHONPATH` pointed at the local LlamaFactory source and AsymGEMM source.
2. Build a tiny module with PEFT-style `.lora_A.default.weight` / `.lora_B.default.weight` trainables and packed expert-style `gate_lora_A`, `gate_lora_B`, `up_lora_A`, `up_lora_B`, `down_lora_A`, `down_lora_B` trainables; assert `finetuning_args.use_asym_cpu_adamw=True` returns `asym_gemm.training.cpu_adam.AsymCPUAdamW` and selects all of them.
3. Add one trainable non-LoRA parameter and assert `_create_asym_cpu_adamw_optimizer()` raises a hard error naming the offending parameter.
4. Assert wrapper hyperparameters come from `training_args.learning_rate`, `adam_beta1`, `adam_beta2`, `adam_epsilon`, and `weight_decay`.
5. Assert `CustomSeq2SeqTrainer.create_optimizer()` still delegates to `create_custom_optimizer()` before falling back to HF's default optimizer.
6. Exercise `AsymCPUAdamW._check_post_prepare_devices()` through the first `step()` after a fake `accelerator.prepare` path, or through an LF smoke, and prove CPU masters are still on CPU while LoRA compute params remain CUDA.
7. Build tiny Asym-wrapped modules with `HostWeight`/`AsymFrozenLinear`, `AsymFrozenEmbedding`, `AsymFrozenLayerNorm`, and `AsymFrozenRMSNorm` frozen bases plus trainable LoRA params; assert `_create_asym_cpu_adamw_optimizer()` creates masters only for LoRA params.
8. Build a tiny Asym-wrapped module where one trainable LoRA param is CPU after wrapping; assert the optimizer factory fails with the Stage 7 unsupported message instead of copying it to a second CPU master silently.
9. Add a loader-order/source test or LF smoke assertion proving `init_adapter(...)` runs before `_move_asym_cpu_first_model_to_device(model)` in CPU-first Asym runs.
10. In a tiny CPU-first Asym model, assert `model.to(cuda)` moves trainable LoRA `nn.Parameter`s to CUDA while `HostWeight` tensors remain CPU-resident and are not registered as parameters or buffers.
11. Add a copy-budget test: selected frozen/base HostWeights may be adopted in place or replaced by one pinned CPU owner, selected embedding/norm wrappers may adopt CPU storage in place, but CPUAdam wrapper construction must not create any additional CPU tensor for those frozen/base weights.

## Stage 4: Checkpoint And Resume

Goal: make CPUAdam usable for normal training checkpoints. This is a separate milestone because current Asym parser rejects resume.

### 4.1 First checkpoint support without parser relaxation

Before allowing resume, verify that checkpoints are written correctly:

- Run with `save_strategy=steps`, `save_steps=1`, `max_steps=2`.
- Confirm checkpoint contains:
  - adapter weights
  - `optimizer.pt`
  - `scheduler.pt`
  - trainer state
- Inspect `optimizer.pt` and confirm:
  - `format == "asym_cpu_adamw_v1"`
  - CPU master tensors are stored
  - inner optimizer state exists for every selected LoRA param that had a grad during the step

If a tiny smoke does not exercise every LoRA tensor, do not treat missing moment tensors for never-stepped params as a checkpoint failure. For checkpoint tests that require all state entries, force non-`None` grads for all selected LoRA params.

### 4.2 Enable resume

Modify:

`${LF_DIR}/src/llamafactory/hparams/parser.py`

Current Asym block rejects `training_args.resume_from_checkpoint is not None`, and Stage 1 must also block LlamaFactory's output-dir auto-resume. Relax this only when:

```text
finetuning_args.use_asym_cpu_adamw == True
```

and after Stage 4.1 has passed.

Before relaxing the parser, add an Asym-aware model checkpoint load path. Optimizer resume is not enough.

Required model-load behavior:

1. Add an override in `CustomSeq2SeqTrainer._load_from_checkpoint(...)` or an equivalent SFT workflow hook that runs after the model has been converted to AsymGEMM modules and before optimizer/scheduler resume.
2. For AsymGEMM, call:

```python
from asym_gemm.integrations.lf import load_asym_peft_adapter

load_asym_peft_adapter(model, resume_from_checkpoint, strict=True)
```

3. For non-Asym paths, leave HF/PEFT loading unchanged.
4. Verify all-expert/non-dense Asym checkpoints load correctly. These are not guaranteed to load through HF's normal PEFT branch because PEFT wrapping is only used for dense target modules.
5. Handle best-checkpoint loading. Either keep rejecting `training_args.load_best_model_at_end=True`, or override/patch HF Trainer's `_load_best_model(...)` for AsymGEMM so it calls `load_asym_peft_adapter` on the best checkpoint path.

For v1, the intended `CustomSeq2SeqTrainer._load_from_checkpoint(...)` behavior is:

1. If not AsymGEMM, call `super()._load_from_checkpoint(...)` unchanged.
2. If AsymGEMM, call `load_asym_peft_adapter(...)` and return without calling `super()._load_from_checkpoint(...)`, because the superclass model/PEFT load path does not understand Asym expert-side LoRA state.
3. Do not bypass HF Trainer's optimizer/scheduler/trainer-state resume. Those are handled after model loading by the normal Trainer train loop, and must still load `optimizer.pt`, `scheduler.pt`, RNG state, and trainer state.
4. If a future implementation calls `super()._load_from_checkpoint(...)` for Asym, it must explicitly suppress only the superclass model/adapter load while preserving optimizer/scheduler/trainer-state resume. Do not allow a double model load.

Keep `adapter_name_or_path` resume disabled unless separately implemented. This stage is for Trainer checkpoint resume from an Asym-saved checkpoint directory, not arbitrary external adapter loading.

### 4.3 Add checkpoint launcher controls

Modify:

`${ASYM_DIR}/scripts/lf/run_lf_lora_sft.sh`

Current launcher arguments hardcode `--save_strategy no`. Add explicit env controls so Stage 4 can validate checkpointing without hand-editing the script:

```bash
SAVE_STRATEGY=${SAVE_STRATEGY:-no}
SAVE_STEPS=${SAVE_STEPS:-}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
LOAD_BEST_MODEL_AT_END=${LOAD_BEST_MODEL_AT_END:-false}
OVERWRITE_OUTPUT_DIR=${OVERWRITE_OUTPUT_DIR:-true}
```

Launcher behavior:

1. Replace the hard-coded `--save_strategy no` with `--save_strategy "${SAVE_STRATEGY}"`.
2. Append `--save_steps "${SAVE_STEPS}"` only when `SAVE_STEPS` is non-empty.
3. Append `--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}"` only when non-empty.
4. Append `--load_best_model_at_end "${LOAD_BEST_MODEL_AT_END}"`.
5. Replace the hard-coded `--overwrite_output_dir true` with `--overwrite_output_dir "${OVERWRITE_OUTPUT_DIR}"`.
6. Keep parser rejection for `load_best_model_at_end=true` until the Asym-aware best-model load path is implemented.

### 4.4 Resume test

Add test or manual smoke:

1. Train 2 steps with checkpoint at step 1.
2. Resume from checkpoint-1 and train to step 2.
3. Verify:
   - `optimizer.pt` loads without CPU/GPU device mismatch.
   - CPU master params are CPU after resume.
   - CUDA LoRA params match CPU masters immediately after optimizer load.
   - Loss continues and no trainable parameter count changes.

### 4.5 Stage 4 tests

Add:

`${ASYM_DIR}/tests/lf/test_asym_cpu_adamw_checkpoint_resume.py`

Required tests:

1. `run_lf_lora_sft.sh` passes `--save_strategy`, `--save_steps`, `--resume_from_checkpoint`, and `--load_best_model_at_end` only according to the new env controls.
2. Parser keeps explicit and auto checkpoint resume rejected until `use_asym_cpu_adamw=True` and Stage 4 model-load support is enabled.
3. `CustomSeq2SeqTrainer._load_from_checkpoint(...)`, or the chosen equivalent hook, calls `asym_gemm.integrations.lf.load_asym_peft_adapter(model, checkpoint, strict=True)` for AsymGEMM and leaves non-Asym loading unchanged.
4. A wrapper checkpoint round trip saves `optimizer.pt`, reloads it, keeps CPU masters on CPU, and copies CPU master weights back into CUDA LoRA params immediately after `load_state_dict()`.
5. `load_best_model_at_end=True` remains rejected unless the best-checkpoint hook is implemented and tested through `_load_best_model(...)`.

## Stage 5: Profiling And Memory Attribution

Goal: prove the expected HBM reduction and avoid misleading memory reports.

### 5.1 Add optimizer summary to source profile

Modify:

`${ASYM_DIR}/scripts/lf/run_lf_profiled_train.py`

Also modify:

`${ASYM_DIR}/asym_gemm/profiling/lf_trace.py`

Add a helper similar to `_superoffload_summary_from_config`:

```python
def _asym_cpu_adamw_summary_from_model_or_optimizer(...) -> dict[str, Any]:
```

The summary should read the optimizer if available and include:

- enabled
- backend
- param_count
- param_numel
- cpu_master_bytes
- pinned_cpu_master_bytes
- optimizer_state_cpu_bytes
- all_masters_on_cpu
- all_cuda_params_on_cuda

Add this under:

```python
    "asym_cpu_adamw": ...
```

in `LFProfileRecorder.report(...)`.

Also preserve the existing Asym offload report fields in the source profile/config:

- `asym_offload_modules`
- `cpu_resident_base_bytes_by_component`
- `gpu_resident_base_bytes_by_component`
- `selected_gpu_resident_base_bytes_by_component`

These component fields are required because CPUAdam and per-module frozen/base offload save different HBM categories. A CPUAdam memory claim should point to optimizer-state rows; a per-module offload claim should point to frozen/base component residency rows.

Also update `LFProfileRecorder._measured_records()` and `_stage_rows()` so warmup filtering applies to every per-step timing range, not only `step.forward` and `step.backward`. `_stage_rows()` must call `_measured_records(name)` for the reported `milliseconds` and `samples`; if raw sample counts are useful, add a separate `raw_samples` field. At minimum, skip the first `warmup_steps` records for:

- `lf.step.total`
- `step.forward`
- `step.backward`
- `lf.optimizer.step`
- `lf.optimizer.zero_grad`
- `lf.scheduler.step`

Without this change, `step.total_milliseconds` is post-warmup but `lf.optimizer.step` and `lf.step.total` include warmup, making Stage 5/6 latency comparisons misleading.

Implementation detail: `scripts/lf/run_lf_profiled_train.py::LFProfileRecorder.report()` currently receives only the `LFTraceHandle`. In `asym_gemm/profiling/lf_trace.py`, add two optimizer fields to `LFTraceHandle`:

- `prepared_optimizer`: the object currently in `trainer.optimizer`, used for step/zero-grad hook wrapping
- `optimizer`: the unwrapped optimizer used for summaries and memory introspection

In `lf_trace.py::_patch_optimizer_objects()`, keep wrapping the prepared optimizer for runtime hooks, but unwrap known wrappers for introspection. At minimum, if `trainer.optimizer` lacks `asym_cpu_master_params` and has an `.optimizer` attribute, inspect that wrapped object too. `LFProfileRecorder.report()`, `LFMemoryBreakdownProfiler.set_optimizer()`, and memory-breakdown helper code must use the unwrapped optimizer for `asym_cpu_adamw_summary()`, `asym_cpu_master_params()`, and `asym_cpu_param_name_map()`. Do not rely on private `LFMemoryBreakdownProfiler._optimizer`.

### 5.2 Update memory attribution

Modify:

`${ASYM_DIR}/asym_gemm/profiling/lf_trace.py`

Current `_collect_persistent_bytes(...)` uses exact unique storage accounting and only sees:

- model parameters
- model parameter gradients
- model buffers
- `HostWeight`
- `optimizer.state`

The wrapper must expose CPU master and CPU optimizer-state tensors through visible `optimizer.state` entries keyed by original CUDA model params. With that design, current attribution can map CPU optimizer state to the correct LoRA component.

Still add wrapper-aware detection for clearer labels:

```python
if hasattr(optimizer, "asym_cpu_master_params"):
    collect each CPU master param as cpu_master_weight using optimizer.asym_cpu_param_name_map()
```

Collect CPU master params before walking generic `optimizer.state`, and add their storage keys to the dedupe set first. When later walking visible `optimizer.state`, skip the `cpu_master` entry if that storage key was already recorded as `cpu_master_weight`; only count moment tensors such as `exp_avg` and `exp_avg_sq` as `optimizer_state`. Also use `asym_cpu_param_name_map()` as a fallback if any inner optimizer state keyed by CPU master param is observed. In the current `LFMemoryBreakdownProfiler._collect_persistent_bytes()` helper, CPU-side kinds are emitted with device suffixes, so expected rows are `cpu_master_weight_cpu` and, when pinned, `cpu_master_weight_cpu_pinned`; moment rows are `optimizer_state_cpu` and optional `optimizer_state_cpu_pinned`.

Expected memory report after Stage 5:

- GPU optimizer state for LoRA should drop near zero.
- CPU optimizer state should appear under CPU host or CPU pinned subset.
- CPU master params should appear under CPU host or CPU pinned subset.
- GPU LoRA weights and GPU LoRA grads should remain.
- Frozen/base CPU offload bytes should remain attributed by component and should not be counted as CPUAdam optimizer-state savings. This includes pinned HostWeight replacements for AsymGEMM-fetched linear/router/expert/LM-head weights and strict no-copy CPU owners for embedding/norm wrappers.
- Embedding/norm forward-time device copies are execution overhead for frozen/base offload, not persistent CPUAdam memory. Do not include them in CPU master or optimizer-state byte totals.
- Selected frozen/base components should have no CUDA parameter/buffer residue under strict mode unless the report explicitly marks them unsupported/skipped.

Use `memory_breakdown` exact storage accounting for CPUAdam acceptance. The older aggregate `memory_attribution` path can overcount HostWeight CPU bytes because it mixes aggregate module byte properties with raw `host_weight` tensor traversal. If CPUAdam reporting uses `memory_attribution`, first dedupe `_model_memory_summary()` by tensor storage and stop mixing aggregate byte properties with raw tensor traversal.

### 5.3 Add launcher/profile labels

Modify:

- `scripts/lf/run_lf_lora_sft.sh`
- `scripts/lf/profile_lora_lf.sh`
- `scripts/lf/postprocess_lf_profile_artifacts.py` if summary tables need a column

The profile config should include:

```text
use_asym_cpu_adamw
asym_cpu_adamw_backend
asym_cpu_adamw_pin_memory
asym_cpu_adamw_fp32_master
```

In `run_lf_lora_sft.sh`, export these as profile config env vars whenever profiling is enabled:

```bash
ASYM_GEMM_LF_CONFIG_USE_ASYM_CPU_ADAMW="${USE_ASYM_CPU_ADAMW}"
ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_BACKEND="${ASYM_CPU_ADAMW_BACKEND}"
ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_PIN_MEMORY="${ASYM_CPU_ADAMW_PIN_MEMORY}"
ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_FP32_MASTER="${ASYM_CPU_ADAMW_FP32_MASTER}"
```

`run_lf_profiled_train.py::_env_config()` already copies `ASYM_GEMM_LF_CONFIG_*` keys into `profile["config"]`, so no special parser is needed for these fields unless they need type normalization. If type normalization is added, test both string env capture and normalized JSON values.

Do not compare a CPUAdam run against old Asym runs under the same directory label.

### 5.4 Stage 5 tests

Extend:

`${ASYM_DIR}/tests/test_lf_memory_breakdown.py`

Add or extend:

`${ASYM_DIR}/tests/lf/test_superoffload_backend_scripts.py`

Required tests:

1. Build a synthetic optimizer exposing `asym_cpu_master_params()`, `asym_cpu_param_name_map()`, and `asym_cpu_adamw_summary()`. Assert `LFMemoryBreakdownProfiler._collect_persistent_bytes(...)` labels CPU masters as `cpu_master_weight_cpu` or `cpu_master_weight_cpu_pinned`, and moment tensors as `optimizer_state_cpu` or `optimizer_state_cpu_pinned`.
2. Assert the same CPU master storage is not double-counted when it appears both in `asym_cpu_master_params()` and in visible `optimizer.state[cuda_param]["cpu_master"]`.
3. Assert `LFProfileRecorder.report(...)` emits an `asym_cpu_adamw` object from the unwrapped optimizer, even if the prepared optimizer is wrapped by Trainer/Accelerate.
4. Assert `_measured_records()` and `_stage_rows()` skip warmup records for `lf.step.total` and `lf.optimizer.step`, not only forward/backward records.
5. Assert postprocessing emits CPU master bytes and CPU optimizer-state bytes into markdown/CSV summaries.
6. Assert profile/result labels for `asym_cpuadamwtorch` and `asym_cpuadamwds` parse as Asym-family labels and are not dropped by plot filters.
7. Assert a synthetic Asym report with `offload_modules=all` keeps frozen/base component bytes separate from `asym_cpu_adamw.cpu_master_bytes` and optimizer-state bytes, including both pinned HostWeight components and no-copy embedding/norm components.

## Stage 6: Performance Cleanup

Goal: reduce copy overhead after correctness and memory accounting are stable.

Only start this after Stage 5 proves the expected HBM reduction.

### 6.1 Persistent grad buffers

The initial wrapper may allocate CPU grad tensors every step. Replace this with persistent grad buffers:

- allocate one CPU grad buffer per CPU master param
- match shape and dtype
- pin when `pin_memory=True` and CUDA is available
- copy CUDA grad into that buffer every step
- set `cpu_param.grad = grad_buffer`

### 6.2 Copy timing

Add lightweight timers inside `AsymCPUAdamW.step()`:

- `grad_copy_ms`
- `cpu_adam_step_ms`
- `weight_copyback_ms`

Expose them through `asym_cpu_adamw_summary()`.

Add optional source profiler rows only if this does not add meaningful overhead. Otherwise log aggregate timings at the end.

### 6.3 Optional flat CPU master buffer

Flat CPU master buffers are not required for correctness. Add them only if per-tensor CPUAdam overhead is measurable.

Design if needed:

```text
one flat CPU master tensor per optimizer group
per-LoRA CPU param views into the flat tensor
same logical ordering as param_names
state_dict stores flat buffers plus view metadata
```

Do not use DeepSpeed ZeRO flat buffers. If flat buffers are added, they must be Asym-owned and stable for checkpointing.

### 6.4 Stage 6 tests

Extend:

`${ASYM_DIR}/tests/training/test_asym_cpu_adamw.py`

Required tests:

1. Persistent grad buffers are allocated once per selected LoRA param, remain CPU, match shape/dtype, and are reused across multiple steps.
2. `zero_grad(set_to_none=True)` clears CUDA LoRA grads and CPU master grads without clearing Adam moment state.
3. `asym_cpu_adamw_summary()` reports `grad_copy_ms`, `cpu_adam_step_ms`, and `weight_copyback_ms` after at least one step.
4. Optional flat CPU master buffers, if implemented, preserve `param_names`, views, state dict round trip, and weight copyback correctness.

## Stage 7: Optional CPU-Fetched Trainable LoRA

Goal: remove GPU LoRA weight HBM too. This is a research/second-project stage, not part of CPUAdam MVP.

Current Stage 2 design leaves LoRA compute weights on GPU. That saves only Adam moment HBM.

Do not confuse this with the current `asym_offload_modules` work. Per-module offload covers frozen/base weights. Stage 7 is only for trainable LoRA A/B tensors becoming CPU-owned fetchable tensors.

CPU-fetched trainable LoRA would require:

1. CPU pinned LoRA A/B tensors in an Asym-owned layout.
2. Forward kernels that fetch LoRA A/B from CPU like frozen base weights.
3. Backward kernels/autograd that compute and accumulate LoRA gradients for CPU-resident trainable weights.
4. CPUAdam updating the same CPU-resident trainable LoRA tensors.
5. A cache/version protocol so forward never reads while CPUAdam is updating.
6. New state_dict logic for trainable CPU LoRA tensors.

This is explicitly not part of the CPUAdam MVP contract. If it becomes active work, likely targets are:

- `asym_gemm/training/lora.py::AsymLoRALinear`
- `asym_gemm/training/lora.py::PackedExpertLoRA`
- `asym_gemm/training/qwen3_moe.py::AsymQwen3Experts`
- `asym_gemm/training/qwen3_moe.py::AsymQwen3MoeBlock`
- `asym_gemm/training/qwen35_moe.py::AsymQwen35MoeBlock`
- `asym_gemm/training/llama4_moe.py::AsymLlama4Moe`

This can be compatible with CPUAdam if AsymGEMM owns the CPU tensor layout. It is not compatible with treating ZeRO-owned flat fp32 partitions as the compute layout without gather/repack overhead.

Do not start Stage 7 until the CPU-master optimizer path is correct and profiled.

### 7.1 Stage 7 tests

Add only if Stage 7 is implemented:

`${ASYM_DIR}/tests/training/test_asym_cpu_fetched_lora.py`

Required tests:

1. CPU-resident trainable LoRA tensors are not registered as GPU model parameters.
2. Forward fetch reads the CPU-owned LoRA tensor version intended for that step.
3. Backward accumulates gradients into the CPU-owned LoRA layout or a documented CPU grad layout.
4. CPUAdam updates the same CPU-resident trainable tensor that the fetch path will read next.
5. Checkpoint save/load restores CPU trainable LoRA layout, optimizer state, and fetch metadata.

## Stage Validation Gates

These gates are the minimum commands/processes to run before moving from one stage to the next. Each gate assumes the code changes for that stage have already been implemented, so commands that mention new flags or backends are expected to fail before their stage is coded. Keep gates short. Use the heavier `scripts/lf/profile_lora_lf_test.sh` only with explicit small overrides; its defaults are intended for broad profiling and are too large for stage gates. Because `profile_lora_lf_test.sh` is not executable in this checkout, call it with `bash`.

Common small profile knobs for gates:

```bash
export ASYM_DIR=${ASYM_DIR:-$(pwd)}
export SFT_ROOT=${SFT_ROOT:-$(cd "${ASYM_DIR}/../.." && pwd)}
export LF_DIR=${LF_DIR:-${SFT_ROOT}/third_party/LlamaFactory}
export DEEPSPEED_DIR=${DEEPSPEED_DIR:-${SFT_ROOT}/third_party/deepspeed}
export ENV_PYTHON=${ENV_PYTHON:-${ASYM_DIR}/.venv/bin/python}
GPU_POOL=0
MODEL_SPECS=meta-llama/Llama-4-Scout-17B-16E\|1
SEQ_LENS=16
PER_DEVICE_TRAIN_BATCH_SIZE=1
MAX_SAMPLES=4
WARMUP_STEPS=5
MAX_STEPS=2
PROFILERS=source
PREPARE_DATASETS=true
PLOT=false
PLOT_MEMORY_BREAKDOWN=false
```

Always set `ASYM_OFFLOAD_MODULES` explicitly in stage gates. Current script defaults differ: `run_lf_lora_sft.sh` and `profile_lora_lf_test.sh` default to `routed_experts`, while `profile_lora_lf.sh` defaults to `all`.

Use `ASYM_OFFLOAD_MODULES=all` only in dry-run selector-plumbing gates unless the target model is known to support every selected component. Runtime correctness/latency gates use `ASYM_OFFLOAD_MODULES=routed_experts` for Llama-4-Scout on SM100 BF16. In this checkout, router offload reaches a frozen-linear dx shape with `k = num_experts = 16`, while the direct BF16 dx kernel requires transpose-B `k` to be 64-aligned. A router runtime gate therefore validates router-kernel/fallback support, not CPUAdam correctness.

After any source-profile gate, set `OUT` to the same output root used by the gate command and print the profile summary with:

```bash
export OUT=/tmp/asym_cpuadam_stage_check
python - <<'PY'
import glob, json, os
root = os.environ["OUT"]
profiles = sorted(glob.glob(root + "/**/source_profile.json", recursive=True))
assert profiles, f"no source_profile.json under {root}"
for path in profiles:
    data = json.load(open(path, "r", encoding="utf-8"))
    rows = data.get("step", {}).get("rows", [])
    row_ms = {row.get("name"): row.get("milliseconds") for row in rows if isinstance(row, dict)}
    print(path)
    print("step_total_ms", data.get("step", {}).get("total_milliseconds"))
    print("lf_step_total_ms", row_ms.get("lf.step.total"))
    print("lf_optimizer_step_ms", row_ms.get("lf.optimizer.step"))
    print("asym_cpu_adamw", data.get("asym_cpu_adamw"))
    print("cpuadam", data.get("cpuadam"))
PY
```

### Stage 0 Gate

Correctness:

```bash
rg -n "class DeepSpeedCPUAdam|def step" "${DEEPSPEED_DIR}/deepspeed/ops/adam/cpu_adam.py"
rg -n "def create_optimizer|def create_custom_optimizer" "${LF_DIR}/src/llamafactory/train/sft/trainer.py" "${LF_DIR}/src/llamafactory/train/trainer_utils.py"
rg -n "_use_asym_cpu_first_load|_move_asym_cpu_first_model_to_device|init_adapter|model.to\\(device\\)" "${LF_DIR}/src/llamafactory/model/loader.py"
rg -n "def _collect_persistent_bytes|def _patch_optimizer_objects" asym_gemm/profiling/lf_trace.py
rg -n "def parse_lf_offload_modules|def validate_lf_offload_residency|adopt_host_weight|clone=False" asym_gemm/integrations/lf.py asym_gemm/training/offload.py
rg -n "ASYM_OFFLOAD_MODULES|asym_offload_modules" scripts/lf/run_lf_lora_sft.sh scripts/lf/profile_lora_lf.sh scripts/lf/profile_lora_lf_test.sh
python - <<'PY'
import os
from pathlib import Path
src = Path(os.environ["DEEPSPEED_DIR"], "deepspeed/ops/adam/cpu_adam.py").read_text()
assert "if p.grad is None" in src
assert "assert p.device == device" in src
assert "adam_w_mode" in src
print("DeepSpeedCPUAdam CPU-param assertion verified by source inspection")
PY
python - <<'PY'
import os
from pathlib import Path
lf_dir = Path(os.environ["LF_DIR"])
trainer = Path(lf_dir, "src/llamafactory/train/sft/trainer.py").read_text()
utils = Path(lf_dir, "src/llamafactory/train/trainer_utils.py").read_text()
parser = Path(lf_dir, "src/llamafactory/hparams/parser.py").read_text()
assert "create_custom_optimizer(self.model, self.args, self.finetuning_args)" in trainer
assert "def create_custom_optimizer" in utils
assert "resume_from_checkpoint" in parser and "can_resume_from_checkpoint" in parser
print("LF optimizer hook and resume-auto-detection anchors verified")
PY
python - <<'PY'
import os
from pathlib import Path
loader = Path(os.environ["LF_DIR"], "src/llamafactory/model/loader.py").read_text()
init_pos = loader.index("model = init_adapter(config, model, model_args, finetuning_args, is_trainable)")
move_pos = loader.index("_move_asym_cpu_first_model_to_device(model)", init_pos)
assert init_pos < move_pos
assert "model.to(device)" in loader
assert "selection.any_cpu_offload" in loader
print("LF Asym CPU-first load moves non-HostWeight state after adapter conversion")
PY
python -m pytest -q tests/training/test_cpu_resident_frozen_base.py -k 'module_to_cuda_does_not_move_host_weight or host_weight_exposes_only_single_pinned_weight_copy'
python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py tests/training/test_lf_qwen35_asym_backend.py -k 'offload or adopts_cpu_storage or all_selector or single_cpu_owner'
"${ENV_PYTHON}" - <<'PY'
import accelerate
import inspect
from transformers import Trainer
save_src = inspect.getsource(Trainer._save_optimizer_and_scheduler)
load_src = inspect.getsource(Trainer._load_optimizer_and_scheduler)
assert "optimizer.pt" in save_src or "OPTIMIZER_NAME" in save_src
assert "load_state_dict" in load_src
print("LF env accelerate import and HF optimizer save/load anchors verified")
PY
```

Latency: no latency gate. Stage 0 is read-only design validation.

### Baseline Stage Gate: `zero3_cpuadam`

Correctness:

```bash
bash -n scripts/lf/run_lf_lora_sft.sh
bash -n scripts/lf/profile_lora_lf.sh
bash -n scripts/lf/profile_lora_lf_test.sh
python -m py_compile scripts/lf/run_lf_profiled_train.py scripts/lf/postprocess_lf_profile_artifacts.py
python -m pytest -q tests/lf/test_superoffload_backend_scripts.py -k 'zero3_cpuadam or zero3_offload or superoffload'
export OUT=/tmp/asym_cpuadam_b0_dryrun
BACKEND_SPECS=zero3_cpuadam\|recomp PROFILERS=source SEQ_LENS=16 MAX_SAMPLES=4 \
PREPARE_DATASETS=false DRY_RUN=true PLOT=false PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf_test.sh --output-root "${OUT}"
```

Latency:

```bash
export OUT=/tmp/asym_cpuadam_b0_latency
GPU_POOL=0 MODEL_SPECS=meta-llama/Llama-4-Scout-17B-16E\|1 \
BACKEND_SPECS=zero3_cpuadam\|recomp PROFILERS=source SEQ_LENS=16 MAX_SAMPLES=4 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 WARMUP_STEPS=5 MAX_STEPS=2 OVERWRITE=true \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf_test.sh --output-root "${OUT}"
```

Acceptance: source profile has `cpuadam.runtime_verified == true` or the train log has `DeepSpeedCPUAdam`; `step.total_milliseconds` is present.

### Stage 1 Gate

Correctness:

```bash
bash -n scripts/lf/run_lf_lora_sft.sh
bash -n scripts/lf/profile_lora_lf.sh
bash -n scripts/lf/profile_lora_lf_test.sh
python -m pytest -q tests/lf/test_asym_cpu_adamw_args.py tests/lf/test_superoffload_backend_scripts.py -k 'asym_cpu_adamw or cpuadam'
PYTHONPATH="${LF_DIR}/src:${PYTHONPATH:-}" \
python -m pytest -q "${LF_DIR}/tests/hparams/test_asym_cpu_adamw_args.py"

# Torch CPUAdam alias dry-run: public label and CPUAdam flags must survive.
export OUT=/tmp/asym_cpuadam_s1_dryrun
BACKEND_SPECS=asym_cpuadamwtorch\|recomp ASYM_OFFLOAD_MODULES=all \
PROFILERS=source SEQ_LENS=16 MAX_SAMPLES=4 PREPARE_DATASETS=false DRY_RUN=true \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf_test.sh --output-root "${OUT}"
rg -n -- 'BACKEND=asym_cpuadamwtorch' "${OUT}"
rg -n -- 'PROFILE_BACKEND_LABEL=asym_cpuadamwtorch' "${OUT}"
rg -n -- 'ASYM_OFFLOAD_MODULES=all|--asym_offload_modules all|--asym_offload_modules=all' "${OUT}"
rg -n -- 'USE_ASYM_CPU_ADAMW=true|--use_asym_cpu_adamw true|--use_asym_cpu_adamw=true' "${OUT}"
rg -n -- 'ASYM_CPU_ADAMW_BACKEND=torch|--asym_cpu_adamw_backend torch|--asym_cpu_adamw_backend=torch' "${OUT}"
find "${OUT}" -type d -name 'asym_cpuadamwtorch__source__recomp__polnone__router*' -print -quit | grep -q .
if rg -n -- 'PROFILE_BACKEND_LABEL=asym($|[[:space:]])' "${OUT}"; then
  echo "CPUAdam job was labeled as plain asym" >&2
  exit 1
fi

# DeepSpeed CPUAdam alias dry-run: public label stays ds, LF backend value is deepspeed.
export OUT=/tmp/asym_cpuadam_s1_ds_dryrun
BACKEND_SPECS=asym_cpuadamwds\|recomp \
PROFILERS=source SEQ_LENS=16 MAX_SAMPLES=4 PREPARE_DATASETS=false DRY_RUN=true \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf_test.sh --output-root "${OUT}"
rg -n -- 'BACKEND=asym_cpuadamwds' "${OUT}"
rg -n -- 'PROFILE_BACKEND_LABEL=asym_cpuadamwds' "${OUT}"
rg -n -- 'USE_ASYM_CPU_ADAMW=true|--use_asym_cpu_adamw true|--use_asym_cpu_adamw=true' "${OUT}"
rg -n -- 'ASYM_CPU_ADAMW_BACKEND=deepspeed|--asym_cpu_adamw_backend deepspeed|--asym_cpu_adamw_backend=deepspeed' "${OUT}"
find "${OUT}" -type d -name 'asym_cpuadamwds__source__recomp__polnone__router*' -print -quit | grep -q .

# Default-off Asym dry-run: plain asym must not enable CPUAdamW.
export OUT=/tmp/asym_cpuadam_s1_false_dryrun
BACKEND_SPECS=asym\|recomp USE_ASYM_CPU_ADAMW=false \
PROFILERS=source SEQ_LENS=16 MAX_SAMPLES=4 PREPARE_DATASETS=false DRY_RUN=true \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf_test.sh --output-root "${OUT}"
rg -n -- 'BACKEND=asym' "${OUT}"
if rg -n -- 'asym_cpuadamw|USE_ASYM_CPU_ADAMW=true|--use_asym_cpu_adamw true|--use_asym_cpu_adamw=true|PROFILE_BACKEND_LABEL=asym_cpuadamw' "${OUT}"; then
  echo "default-off Asym path unexpectedly enabled CPUAdamW" >&2
  exit 1
fi

# Mixed dry-run: CPUAdam flags must not leak into ZeRO/SuperOffload/KT/torch jobs.
export OUT=/tmp/asym_cpuadam_s1_mixed_dryrun
BACKEND_SPECS=asym_cpuadamwtorch\|recomp,zero3_offload\|recomp \
PROFILERS=source SEQ_LENS=16 MAX_SAMPLES=4 PREPARE_DATASETS=false DRY_RUN=true \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf_test.sh --output-root "${OUT}"
rg -n -- 'BACKEND=asym_cpuadamwtorch' "${OUT}"
rg -n -- 'PROFILE_BACKEND_LABEL=asym_cpuadamwtorch' "${OUT}"
zero_cmd="$(find "${OUT}" -path '*zero3_offload__source__recomp__polnone__router*/*/command.txt' -print -quit)"
test -n "${zero_cmd}"
rg -n -- 'BACKEND=zero3_offload' "${zero_cmd}"
rg -n -- 'PROFILE_BACKEND_LABEL=zero3_offload' "${zero_cmd}"
rg -n -- 'USE_ASYM_CPU_ADAMW=false' "${zero_cmd}"
if rg -n -- 'USE_ASYM_CPU_ADAMW=true|--use_asym_cpu_adamw true|--use_asym_cpu_adamw=true' "${zero_cmd}"; then
  echo "profile wrapper leaked Asym CPUAdamW into zero3_offload job" >&2
  exit 1
fi
```

Latency: no latency gate. Stage 1 must not change training runtime when `USE_ASYM_CPU_ADAMW=false`; the false-path dry-run above validates command routing and result labels.

### Stage 2 Gate

Correctness:

```bash
python -m pytest -q tests/training/test_asym_cpu_adamw.py
```

Latency:

```bash
ASYM_CPU_ADAMW_BENCH_STEPS=20 python -m pytest -q tests/training/test_asym_cpu_adamw.py -k 'latency or bench'
```

Acceptance: torch backend updates CUDA LoRA params, CPU masters stay on CPU, `state_dict/load_state_dict` restores weights/state, DeepSpeed backend either passes one-step/parity tests or is skipped with a precise extension-build reason, and micro latency reports grad-copy, CPU-step, and copyback timing if those timers already exist.

### Stage 3 Gate

Correctness:

```bash
python -m pytest -q tests/training/test_asym_cpu_adamw.py tests/lf/test_asym_cpu_adamw_lf_integration.py
```

Small LF smoke and latency:

```bash
export OUT=/tmp/asym_cpuadam_s3_torch
GPU_POOL=0 MODEL_SPECS=meta-llama/Llama-4-Scout-17B-16E\|1 \
BACKEND_SPECS=asym_cpuadamwtorch\|recomp ASYM_OFFLOAD_MODULES=routed_experts \
PROFILERS=source SEQ_LENS=16 MAX_SAMPLES=4 PER_DEVICE_TRAIN_BATCH_SIZE=1 \
WARMUP_STEPS=5 MAX_STEPS=2 PROFILE_MEMORY_BREAKDOWN=true OVERWRITE=true \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf_test.sh --output-root "${OUT}"
```

Acceptance: the LF smoke completes, trainer loss rows exist, `step.total_milliseconds` is present, and `step.rows` includes `lf.optimizer.step` so optimizer latency is measurable. The setup log/profile config must show `asym_offload_modules=routed_experts`, and CPU-first runs must log `Moving AsymGEMM non-HostWeight model state to ...` before optimizer stepping. Do not require the `asym_cpu_adamw` summary block yet; Stage 5 owns that profiler/reporting field. Normal `BACKEND_SPECS=asym|recomp USE_ASYM_CPU_ADAMW=false` must still follow the unchanged Asym path, as checked by the Stage 1 false-path dry-run.

### Stage 4 Gate

Correctness:

```bash
python -m pytest -q tests/lf/test_asym_cpu_adamw_checkpoint_resume.py
export OUT=/tmp/asym_cpuadam_s4_run
BACKEND=asym_cpuadamwtorch \
SAVE_STRATEGY=steps SAVE_STEPS=1 OUT_DIR="${OUT}" CUTOFF_LEN=16 MAX_SAMPLES=4 MAX_STEPS=2 \
scripts/lf/run_lf_lora_sft.sh
BACKEND=asym_cpuadamwtorch \
RESUME_FROM_CHECKPOINT="${OUT}/checkpoint-1" OVERWRITE_OUTPUT_DIR=false \
OUT_DIR="${OUT}" CUTOFF_LEN=16 MAX_SAMPLES=4 MAX_STEPS=3 \
scripts/lf/run_lf_lora_sft.sh
```

Latency: use source profiling for the resume comparison; `trainer_log.jsonl` is not a timing artifact.

```bash
export OUT=/tmp/asym_cpuadam_s4_profile
BACKEND=asym_cpuadamwtorch \
PROFILE=1 PROFILE_PROFILER=source PROFILE_SOURCE_JSON="${OUT}/fresh_source_profile.json" \
PROFILE_OUTPUT_DIR="${OUT}" SAVE_STRATEGY=steps SAVE_STEPS=1 OUT_DIR="${OUT}/lf_run" \
CUTOFF_LEN=16 MAX_SAMPLES=4 MAX_STEPS=2 \
scripts/lf/run_lf_lora_sft.sh
BACKEND=asym_cpuadamwtorch \
PROFILE=1 PROFILE_PROFILER=source PROFILE_SOURCE_JSON="${OUT}/resume_source_profile.json" \
PROFILE_OUTPUT_DIR="${OUT}" RESUME_FROM_CHECKPOINT="${OUT}/lf_run/checkpoint-1" OVERWRITE_OUTPUT_DIR=false \
OUT_DIR="${OUT}/lf_run" \
CUTOFF_LEN=16 MAX_SAMPLES=4 MAX_STEPS=3 \
scripts/lf/run_lf_lora_sft.sh
```

Acceptance: compare `step.rows[].name == "lf.optimizer.step"` and `lf.step.total` between `fresh_source_profile.json` and `resume_source_profile.json`. Resume overhead must be limited to checkpoint load; steady-state optimizer and step latency should match Stage 3 within normal run-to-run noise.

### Stage 5 Gate

Correctness and memory attribution:

```bash
export OUT=/tmp/asym_cpuadam_s5_mem
GPU_POOL=0 MODEL_SPECS=meta-llama/Llama-4-Scout-17B-16E\|1 \
BACKEND_SPECS=asym_cpuadamwds\|recomp ASYM_OFFLOAD_MODULES=routed_experts \
PROFILERS=source SEQ_LENS=16 MAX_SAMPLES=4 PER_DEVICE_TRAIN_BATCH_SIZE=1 \
WARMUP_STEPS=5 MAX_STEPS=2 PROFILE_MEMORY_BREAKDOWN=true OVERWRITE=true \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf_test.sh --output-root "${OUT}"
```

If the DeepSpeed CPUAdam extension fails before training, run the same memory-attribution gate with the torch backend to validate profiler correctness while keeping the DeepSpeed failure visible:

```bash
export OUT=/tmp/asym_cpuadam_s5_mem_torch_fallback
GPU_POOL=0 MODEL_SPECS=meta-llama/Llama-4-Scout-17B-16E\|1 \
BACKEND_SPECS=asym_cpuadamwtorch\|recomp ASYM_OFFLOAD_MODULES=routed_experts \
PROFILERS=source SEQ_LENS=16 MAX_SAMPLES=4 PER_DEVICE_TRAIN_BATCH_SIZE=1 \
WARMUP_STEPS=5 MAX_STEPS=2 PROFILE_MEMORY_BREAKDOWN=true OVERWRITE=true \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf_test.sh --output-root "${OUT}"
```

Latency:

```bash
export OUT=/tmp/asym_cpuadam_s5_latency
GPU_POOL=0 MODEL_SPECS=meta-llama/Llama-4-Scout-17B-16E\|1 \
BACKEND_SPECS=asym_cpuadamwds\|recomp ASYM_OFFLOAD_MODULES=routed_experts \
PROFILERS=source SEQ_LENS=16 MAX_SAMPLES=4 PER_DEVICE_TRAIN_BATCH_SIZE=1 \
WARMUP_STEPS=5 MAX_STEPS=2 PROFILE_MEMORY_BREAKDOWN=true OVERWRITE=true \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf_test.sh --output-root "${OUT}"
```

Acceptance: GPU optimizer-state bytes for LoRA are removed or near zero, CPU master and CPU optimizer-state bytes are visible, `asym_cpu_adamw.enabled == true`, `asym_offload_modules` is `routed_experts`, frozen/base component bytes are reported separately from optimizer-state bytes, and `step.rows` includes `lf.optimizer.step` for optimizer latency comparison. `step.total_milliseconds` is forward+backward only in the current source profiler, so do not use it as the CPUAdam optimizer-latency number. The torch fallback validates memory/profiler correctness only; do not use it to claim DeepSpeed CPUAdam latency.

### Stage 6 Gate

Correctness:

```bash
python -m pytest -q tests/training/test_asym_cpu_adamw.py -k 'grad_buffer or timing or state_dict'
```

Latency:

```bash
export OUT=/tmp/asym_cpuadam_s6_latency
GPU_POOL=0 MODEL_SPECS=meta-llama/Llama-4-Scout-17B-16E\|1 \
BACKEND_SPECS=asym_cpuadamwds\|recomp ASYM_OFFLOAD_MODULES=routed_experts \
PROFILERS=source SEQ_LENS=16 MAX_SAMPLES=4 PER_DEVICE_TRAIN_BATCH_SIZE=1 \
WARMUP_STEPS=5 MAX_STEPS=2 PROFILE_MEMORY_BREAKDOWN=true OVERWRITE=true \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf_test.sh --output-root "${OUT}"
```

Acceptance: correctness matches Stage 5, `grad_copy_ms`, `cpu_adam_step_ms`, and `weight_copyback_ms` are present in `asym_cpu_adamw_summary()`, `lf.optimizer.step` is present in `step.rows`, and total optimizer latency is not worse than Stage 5.

### Stage 7 Gate

Correctness:

```bash
python -m pytest -q tests/training/test_asym_cpu_fetched_lora.py
```

Latency and memory:

```bash
export OUT=/tmp/asym_cpuadam_s7_fetch_lora
GPU_POOL=0 MODEL_SPECS=meta-llama/Llama-4-Scout-17B-16E\|1 \
BACKEND_SPECS=asym_cpuadamwds\|recomp ASYM_OFFLOAD_MODULES=routed_experts \
PROFILERS=source SEQ_LENS=16 MAX_SAMPLES=4 PER_DEVICE_TRAIN_BATCH_SIZE=1 \
WARMUP_STEPS=5 MAX_STEPS=2 PROFILE_MEMORY_BREAKDOWN=true OVERWRITE=true \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf_test.sh --output-root "${OUT}"
```

Acceptance: GPU LoRA weight residency drops in addition to optimizer-state bytes, CPU fetch/cache protocol tests pass, and latency is reported against Stage 5/6 CPU-master-only runs.

### Optional Real-Size Performance Benchmark

This benchmark is not a minimum stage gate. Run it after Stage 5, 6, or 7 passes the small gate and you need paper-scale latency/memory numbers:

```bash
export OUT=/tmp/asym_cpuadam_real_latency
GPU_POOL=0 MODEL_SPECS=meta-llama/Llama-4-Scout-17B-16E\|1 \
BACKEND_SPECS=asym_cpuadamwds\|recomp ASYM_OFFLOAD_MODULES=routed_experts \
PROFILERS=source SEQ_LENS=7168 MAX_SAMPLES=128 PER_DEVICE_TRAIN_BATCH_SIZE=4 \
WARMUP_STEPS=5 MAX_STEPS=10 PROFILE_MEMORY_BREAKDOWN=true OVERWRITE=true \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf_test.sh --output-root "${OUT}"
```

Acceptance: compare `lf.optimizer.step`, `lf.step.total`, memory breakdown optimizer-state rows, and CPUAdam copy/step/copyback timers when Stage 6 timing is available.

## Test Matrix

### Unit tests

Add `tests/training/test_asym_cpu_adamw.py`:

- wrapper selects only LoRA trainables
- torch backend one-step update
- optional DeepSpeed backend one-step update
- DeepSpeed backend AdamW parity against `torch.optim.AdamW` with nonzero weight decay
- state_dict/load_state_dict
- LR scheduler mutation propagates to inner optimizer
- CPU masters remain CPU after any simulated load
- raw DeepSpeedCPUAdam rejects CUDA params if someone bypasses the wrapper and the CUDA param has a non-`None` grad

### LlamaFactory dry-run/script tests

Extend:

`${ASYM_DIR}/tests/lf/test_superoffload_backend_scripts.py`

or add a new script test file.

Tests:

- `BACKEND=asym_cpuadamwtorch` command includes:
  - `--use_asym_gemm true`
  - `--use_asym_cpu_adamw true`
  - `--asym_cpu_adamw_backend torch`
  - no `--deepspeed`
- Direct `BACKEND=zero3_offload USE_ASYM_CPU_ADAMW=true` hard rejects before building an LF command.
- Direct `BACKEND=asym USE_ASYM_CPU_ADAMW=true` and `BACKEND=asym_torch USE_ASYM_CPU_ADAMW=true` hard reject before building an LF command.
- `BACKEND=asym_cpuadamwds` command includes `--asym_cpu_adamw_backend deepspeed` while keeping the public backend label `asym_cpuadamwds`.
- Profile-wrapper mixed sweeps use `asym_cpuadamwtorch` or `asym_cpuadamwds` for CPUAdam jobs and pass `USE_ASYM_CPU_ADAMW=false` to ZeRO/SuperOffload/KT/torch jobs.
- profile dry-run job root includes a CPUAdam-specific label when enabled.

### Real smoke tests

Use tiny GPU smoke first:

```bash
BACKEND_SPECS=asym_cpuadamwtorch\|recomp ASYM_OFFLOAD_MODULES=routed_experts \
MAX_STEPS=2 \
WARMUP_STEPS=0 \
SEQ_LENS=16 \
MAX_SAMPLES=4 \
scripts/lf/profile_lora_lf.sh --dry-run
```

Then actual run on one free GPU:

```bash
BACKEND=asym_cpuadamwtorch \
ASYM_OFFLOAD_MODULES=routed_experts \
CUTOFF_LEN=16 \
MAX_SAMPLES=4 \
MAX_STEPS=2 \
scripts/lf/run_lf_lora_sft.sh
```

Then DeepSpeed CPUAdam backend:

```bash
BACKEND=asym_cpuadamwds \
ASYM_OFFLOAD_MODULES=routed_experts \
CUTOFF_LEN=16 \
MAX_SAMPLES=4 \
MAX_STEPS=2 \
scripts/lf/run_lf_lora_sft.sh
```

The DeepSpeed backend smoke may JIT-build CPUAdam. If it fails because the extension is unavailable, the error must say this is a DeepSpeed CPUAdam build/runtime issue and suggest `BACKEND=asym_cpuadamwtorch` for correctness testing.

### Profiling acceptance

Run the small source-profiler memory breakdown first:

```bash
export OUT=/tmp/asym_cpuadam_profile_accept
GPU_POOL=0 MODEL_SPECS=meta-llama/Llama-4-Scout-17B-16E\|1 \
BACKEND_SPECS=asym_cpuadamwds\|recomp \
ASYM_OFFLOAD_MODULES=routed_experts \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=true \
SEQ_LENS=16 MAX_SAMPLES=4 PER_DEVICE_TRAIN_BATCH_SIZE=1 \
WARMUP_STEPS=5 MAX_STEPS=2 OVERWRITE=true \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf_test.sh --output-root "${OUT}"
```

Use the optional real-size benchmark command above only after this passes.

Acceptance:

- report has `"asym_cpu_adamw": {"enabled": true, ...}`
- GPU optimizer-state bytes for LoRA are removed or near zero
- CPU optimizer-state bytes are visible
- CPU master bytes are visible
- CPU optimizer-state bytes are expected only for LoRA params that have stepped. If the smoke must validate all moment tensors, force grads for all selected LoRA params.
- `lf.optimizer.step` is present, and `lf.step.total` does not regress so badly that CPUAdam makes the run unusable

## Expected Memory Impact

For the current measured Llama4 high-rank/all-expert LoRA case, CPUAdam should remove the existing GPU-resident Adam moment chunk. If the current optimizer state is bf16 moments, that is roughly:

```text
2x LoRA parameter bytes from GPU HBM
```

because Adam has first and second moments. If a future run already keeps fp32 moments on GPU, the removed GPU chunk is larger: roughly `2x LoRA parameter numel * 4 bytes`.

It will not remove:

- GPU LoRA parameter bytes
- GPU LoRA gradient bytes
- GPU-resident dense frozen weights
- temporary/workspace memory
- saved activations

Per-module `asym_offload_modules` can remove additional frozen/base HBM for selected components such as routed experts, router, shared experts, attention bases, embeddings, LM head, or norms. Attribute those savings separately. CPUAdam only removes optimizer-state HBM and adds CPU masters plus CPU optimizer state.

So if the observed Asym persistent GPU train state is roughly:

```text
LoRA weights:    ~4.6 GiB
LoRA grads:      ~4.6 GiB
Adam moments:    ~9.2 GiB
```

Stage 2 should mainly target the `~9.2 GiB` Adam-moment chunk. Stage 7 is required to attack GPU LoRA weight residency.

## Failure Modes And Required Errors

The implementation must fail loudly for these cases:

- `use_asym_cpu_adamw=True` without `use_asym_gemm=True`
- DeepSpeed/ZeRO enabled with Asym CPUAdam
- distributed/DDP enabled
- non-LoRA trainable params remain
- no LoRA trainable params found
- CPUAdam selects a frozen/base `HostWeight`, `AsymFrozenLinear`, `AsymFrozenEmbedding`, `AsymFrozenLayerNorm`, `AsymFrozenRMSNorm`, router wrapper, or any other non-LoRA tensor
- selected trainable LoRA compute params are CPU-resident in v1; the error must point to Stage 7 CPU-fetched trainable LoRA and the expected LlamaFactory CPU-first post-adapter device move
- `asym_cpu_adamw_fp32_master=False`
- `DeepSpeedCPUAdam` backend selected but import/build fails
- CPU masters are moved to CUDA after Trainer/Accelerate preparation
- checkpoint param names/shapes do not match on load
- `BACKEND=asym USE_ASYM_CPU_ADAMW=true`, `BACKEND=asym_torch USE_ASYM_CPU_ADAMW=true`, or any other direct non-alias CPUAdam enablement; CPUAdam jobs must use `BACKEND=asym_cpuadamwtorch` or `BACKEND=asym_cpuadamwds`
- a CPUAdam profile is written under plain `asym__...` directories or `PROFILE_BACKEND_LABEL=asym`
- `BACKEND=asym_cpuadamwds` passes any LF backend value other than `--asym_cpu_adamw_backend deepspeed`
- `BACKEND=asym_cpuadamwtorch` passes any LF backend value other than `--asym_cpu_adamw_backend torch`
- CPUAdam memory reporting double-counts CPU masters because the same storage appears both in `asym_cpu_master_params()` and visible `optimizer.state`
- CPUAdam copies CPU masters back to CUDA for LoRA params with `grad is None`
- CPUAdam constructor/factory calls `model.to(...)`, mutates `param.data`, or otherwise repairs LoRA device placement instead of failing loudly

## Final Acceptance Criteria

The stage is complete only when all are true:

1. `BACKEND=asym_cpuadamwtorch` runs a tiny LF SFT smoke.
2. `BACKEND=asym_cpuadamwds` runs a tiny LF SFT smoke or fails with a precise CPUAdam-extension error before training.
3. Unit tests prove state_dict/load_state_dict restores CPU masters and CUDA LoRA params.
4. Profiler reports CPU master and CPU optimizer-state bytes.
5. Memory breakdown shows GPU optimizer-state reduction.
6. Per-module frozen/base offload bytes are reported separately from CPUAdam optimizer bytes.
7. Normal Asym without CPUAdam still runs unchanged.
8. DeepSpeed/ZeRO/SuperOffload backend behavior is unchanged.
