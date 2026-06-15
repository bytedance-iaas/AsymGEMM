# LoRA Gradient Offload Plan

Goal: add an opt-in LoRA gradient offload path for single-GPU AsymGEMM LoRA SFT. Keep LoRA compute weights on CUDA. Do not add LoRA weight offload in this plan.

Implementation status:

- Stages 1-4 are implemented: the LF toggle is plumbed, `AsymCPUAdamW` has hook-time LoRA grad offload, LF gradient clipping/norm reads the optimizer-owned CPU buffers, and source/memory-breakdown reporting exposes grad-offload counters.
- The implemented path keeps CUDA LoRA weights resident and offloads only grads. Each unique LoRA CUDA parameter gets a post-accumulate hook. The hook copies/accumulates the CUDA grad into one contiguous fp32 CPU grad slab, assigns that CPU view to the CPU master param `.grad`, and immediately clears the CUDA param `.grad`.
- The default remains `asym_cpu_adamw_grad_offload=false`; A/B scripts can sweep `ASYM_CPU_ADAMW_GRAD_OFFLOADS=false,true`, and CPUAdamW job paths include `__gradofffalse` or `__gradofftrue`.
- A shortened real-script A/B smoke passed in the Python 3.12 enroot runtime with host repo mounted: Qwen3-30B-A3B, `asym_cpuadamwds|recomp`, `b1_s128`, `warmup=5`, `measure=1`, `ASYM_CPU_ADAMW_GRAD_OFFLOADS=false,true`. The `grad_offload=true` profile reported 672 hooks, a 12.574 GiB pinned CPU grad slab, `optimizer_memory.cuda_grad_bytes=0`, `step_grad_copy_ms=0`, `grad_clip.path=asym_cpu_adamw_grad_offload`, and peak allocated HBM `6.592 GiB` versus `12.764 GiB` for `grad_offload=false`.
- Full real-script A/B acceptance was executed in the Python 3.12 enroot runtime with host repo mounted: Qwen3-30B-A3B, `asym_cpuadamwds|norecomp`, `b4_s4096`, `warmup=5`, `measure=10`, `ASYMM_EXP_ACT_POLICIES=none|true|true|true`, `ASYM_CPU_ADAMW_GRAD_OFFLOADS=false,true`. Output root: `profiling/lora_grad_offload_accept_20260615T073937Z`.
- Full-run functional grad-offload checks passed: `grad_offload=true` reported 672 hooks, a 12.574 GiB pinned CPU grad slab, `optimizer_memory.cuda_grad_bytes=0`, `step_grad_copy_ms=0`, `grad_clip.path=asym_cpu_adamw_grad_offload`, and `last_step_used_offloaded_grads=true`.
- Full-run peak-HBM acceptance failed: `grad_offload=false` peak allocated HBM was `34.593 GiB`; `grad_offload=true` peak allocated HBM was also `34.593 GiB`, missing the `<= 30.5 GiB` target. Measured step latency stayed within the planned 15% bound: `46.974 s` baseline versus `53.509 s` with grad offload, ratio `1.139x`.
- The failure is not stale CUDA grad residency. The `grad_offload=true` step samples drop live allocation at backward end from `13.012 GiB` to `6.725 GiB`, which matches removing the `6.287 GiB` CUDA LoRA grads. The global peak remains unchanged because this workload also has large loss/lm-head allocations live around the peak, so deleting persistent LoRA grad tensors does not move the allocator high-water mark.
- A focused `grad_offload=true` debug run with `PROFILE_MEMORY_SNAPSHOT=true` confirms the actual peak live set. Output root: `profiling/lora_grad_offload_peakdebug_20260615T083310Z`. The snapshot replay reports peak live HBM `34.593 GiB`, with `18.547 GiB` in two allocator-unframed blocks, `9.273 GiB` in one `cross_entropy`/loss block from `transformers/models/qwen3_moe/modeling_qwen3_moe.py:688`, `6.221 GiB` in long-lived model CUDA params, and only small norm/attention/expert blocks. The three `9.273 GiB` blocks match fp32-sized `[4 * (4096 - 1), 151936]` loss/logit workspace scale. Therefore the unchanged peak is real loss-side memory, not memory appearing from nowhere and not hidden CUDA LoRA grads.
- Stage 5 is therefore not accepted for the stated `<= 30.5 GiB` peak target. The implemented LoRA grad offload is functionally correct and DS-like, but the target requires reducing the loss/lm-head backward temporary peak or changing the acceptance metric to post-backward live allocation; neither is LoRA grad offload alone.

Facts resolved from local code and docs:

- Current `AsymCPUAdamW` keeps CUDA LoRA compute params and CPU fp32 masters, but copies CUDA `.grad` into CPU buffers inside `AsymCPUAdamW.step()` only. See `asym_gemm/training/cpu_adam.py:80`, `:118`, `:299`.
- DeepSpeed `DeepSpeedCPUAdam` is only the CPU optimizer kernel. It requires CPU params and CPU grads; it is not the offload mechanism by itself. Local proof: `third_party/deepspeed/deepspeed/ops/adam/cpu_adam.py:138`. Official doc: https://deepspeed.readthedocs.io/en/latest/optimizers.html
- DeepSpeed ZeRO-3 offload owns the offload mechanics: it registers post-accumulate grad hooks, copies grads into contiguous buffers, copies offloaded grads into fp32 optimizer grads, and sets model `param.grad = None`. Local proof: `third_party/deepspeed/deepspeed/runtime/zero/stage3.py:1279`, `:1379`, `:1703`, `:1754`.
- PyTorch `Tensor.register_post_accumulate_grad_hook` runs after `.grad` is updated, is leaf-tensor only, and the hook can access and modify `.grad`. Official doc: https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.register_post_accumulate_grad_hook.html
- DeepSpeed docs confirm `offload_optimizer` moves optimizer state/computation to CPU and `offload_param` is separate parameter offload. Official doc: https://deepspeed.readthedocs.io/en/latest/zero3.html
- LoRA weight offload is out of scope because current Asym LoRA math and expert activation offload expect trainable LoRA `nn.Parameter`s on CUDA. CPU-resident/fetched LoRA weights would require a separate parameter-staging design, not just this grad path.

What "Python hook" means here:

- A hook is a Python callback that PyTorch autograd calls at a specific point in backward. It is not a CUDA kernel and it does not change LoRA math by itself.
- `register_post_accumulate_grad_hook` is the right hook because it fires after autograd has written the final accumulated gradient for a leaf parameter into `param.grad`.
- The callback receives the parameter tensor, not the gradient tensor. The implementation must read `param.grad`, copy it to CPU, then set `param.grad = None`.
- This adds one Python callback per trainable LoRA parameter per backward. The current optimizer already loops in Python over every LoRA parameter during `step()` to copy grads, so the first implementation moves that per-param copy earlier instead of adding a new dense math kernel. The possible regression is Python callback overhead and less-coalesced copy timing, which is why every nontrivial stage has an e2e source-profile A/B gate.
- The memory win comes from lifetime shortening: CUDA LoRA grads stop living until optimizer step and instead live only until their post-accumulate hook has copied them to CPU.

Peak HBM target for the real acceptance workload:

- Baseline artifact: `reports/qwen3_layer_act/stage1_profile_matrix/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b4_s4096_w5_s10_r64_a16_drop000/asym_cpuadamwds__source__norecomp__polnone__routerwhole__expact1__attnact1__layeract1/b4_s4096/profile.json`.
- Baseline config: Qwen3-30B-A3B, single GPU, `b4_s4096`, `warmup=5`, `measure=10`, rank 64, `asym_cpuadamwds|norecomp`, and activation offload `none|true|true|true` for expert, attention, and layer activation offload.
- Baseline profile numbers: peak allocated HBM `34.593 GiB`, peak reserved HBM `39.676 GiB`, forward peak allocated `29.988 GiB`, backward peak allocated `34.562 GiB`, trainable LoRA params `3,375,366,144`, CUDA bf16 LoRA grads `6.287 GiB`.
- Naively subtracting all CUDA grads gives `34.593 - 6.287 = 28.306 GiB`, but that is not the practical target because the forward pass already peaks around `29.988 GiB` before LoRA grads exist.
- The original expected post-grad-offload peak allocated HBM was `max(forward_peak, backward_peak - cuda_lora_grad_bytes) = max(29.988, 34.562 - 6.287) ~= 30.0 GiB`.
- The full-run A/B invalidated that estimate for this workload. `grad_offload=true` removes CUDA LoRA grads from end-of-backward live allocation, but the recorded global peak still reaches `34.593 GiB` due to loss/lm-head temporary workspace. The allocator snapshot shows three `9.273 GiB` loss-side blocks live at the peak, so the old `~30.0 GiB` estimate was missing loss temporaries rather than failing to offload grads.
- Acceptance target for the real workload remains the user-requested `grad_offload=true` peak allocated HBM `<= 30.5 GiB`, but it is currently not met by LoRA grad offload alone. Peak reserved HBM is tracked but is not the hard target because CUDA allocator caching can keep reserved blocks even after live allocated HBM drops.

## Stage 1: Add the Toggle and A/B Routing With No Behavior Change

Scope:

- `third_party/LlamaFactory/src/llamafactory/hparams/finetuning_args.py`
  - `FinetuningArguments`
- `third_party/LlamaFactory/src/llamafactory/hparams/parser.py`
  - `_verify_asym_cpu_adamw_args`
- `third_party/LlamaFactory/src/llamafactory/train/trainer_utils.py`
  - `_create_asym_cpu_adamw_optimizer`
- `scripts/lf/run_lf_lora_sft.sh`
- `scripts/lf/profile_lora_lf.sh`
  - top-level env defaults
  - usage
  - CLI arg parse
  - bool validation
  - `job_root_path`
  - `existing_profile_complete`
  - `run_job`
  - `ensure_jobs_tsv` / `append_job_record`
- `scripts/lf/test_profiling.sh`
  - same sweep/script functions as `profile_lora_lf.sh`
- `scripts/lf/run_lf_profiled_train.py`
  - `_config_from_args`
- Tests:
  - `tests/lf/test_asym_cpu_adamw_args.py`
  - `tests/lf/test_asym_cpu_adamw_lf_integration.py`
  - `tests/lf/test_lf_profile_postprocess.py`

Implementation:

1. Add a default-off LF argument:

```python
# third_party/LlamaFactory/src/llamafactory/hparams/finetuning_args.py
asym_cpu_adamw_grad_offload: bool = field(
    default=False,
    metadata={"help": "Offload AsymGEMM LoRA grads to CPU during backward instead of copying them in optimizer.step."},
)
```

2. Validate the dependency in the general parser path, then keep CPUAdamW-specific checks in `_verify_asym_cpu_adamw_args`:

```python
# parser.py, near the existing use_asym_cpu_adamw/use_asym_gemm checks
if finetuning_args.asym_cpu_adamw_grad_offload and not finetuning_args.use_asym_cpu_adamw:
    raise ValueError("`asym_cpu_adamw_grad_offload=true` requires `use_asym_cpu_adamw=true`.")

# parser.py::_verify_asym_cpu_adamw_args
if finetuning_args.asym_cpu_adamw_grad_offload and training_args.parallel_mode != ParallelMode.NOT_PARALLEL:
    raise ValueError("AsymGEMM CPU AdamW grad offload is single-process single-device only.")
```

3. Pass the flag into the optimizer constructor, but do not implement behavior yet:

```python
optimizer = AsymCPUAdamW(
    trainable_lora,
    lr=training_args.learning_rate,
    betas=(training_args.adam_beta1, training_args.adam_beta2),
    eps=training_args.adam_epsilon,
    weight_decay=training_args.weight_decay,
    backend=finetuning_args.asym_cpu_adamw_backend,
    pin_memory=finetuning_args.asym_cpu_adamw_pin_memory,
    fp32_master=finetuning_args.asym_cpu_adamw_fp32_master,
    grad_offload=finetuning_args.asym_cpu_adamw_grad_offload,
)
```

4. Add shell env and LF CLI forwarding:

```bash
# scripts/lf/run_lf_lora_sft.sh
ASYM_CPU_ADAMW_GRAD_OFFLOAD=${ASYM_CPU_ADAMW_GRAD_OFFLOAD:-false}

ASYM_CPU_ADAMW_GRAD_OFFLOAD="$(bool_string ASYM_CPU_ADAMW_GRAD_OFFLOAD "${ASYM_CPU_ADAMW_GRAD_OFFLOAD}")"

CMD_ARGS+=(--asym_cpu_adamw_grad_offload "${ASYM_CPU_ADAMW_GRAD_OFFLOAD}")

log_kv ASYM_CPU_ADAMW_GRAD_OFFLOAD "${ASYM_CPU_ADAMW_GRAD_OFFLOAD}"
ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_GRAD_OFFLOAD="${ASYM_CPU_ADAMW_GRAD_OFFLOAD}"
```

5. Add a profile sweep axis that can run both modes in one output root:

```bash
# scripts/lf/profile_lora_lf.sh and scripts/lf/test_profiling.sh
ASYM_CPU_ADAMW_GRAD_OFFLOADS=${ASYM_CPU_ADAMW_GRAD_OFFLOADS:-${ASYM_CPU_ADAMW_GRAD_OFFLOAD:-false}}

case "$1" in
  --asym-cpu-adamw-grad-offloads) need_value "$1" "${2-}"; ASYM_CPU_ADAMW_GRAD_OFFLOADS="$2"; shift 2 ;;
  --asym-cpu-adamw-grad-offloads=*) ASYM_CPU_ADAMW_GRAD_OFFLOADS="${1#*=}"; shift ;;
esac

mapfile -t asym_cpu_adamw_grad_offload_modes < <(
  tokens "${ASYM_CPU_ADAMW_GRAD_OFFLOADS}" | while read -r value; do bool_value "${value}"; done | dedupe
)
```

6. Thread the axis through `run_job`. Non-CPUAdamW backends must force `false`; CPUAdamW backends should use the requested axis.

```bash
# scripts/lf/profile_lora_lf.sh and scripts/lf/test_profiling.sh, inside the existing
# backend/profiler loop before calling run_job.
if cpuadam_backend_for_label "${backend}" >/dev/null; then
  grad_offload_modes_for_job=("${asym_cpu_adamw_grad_offload_modes[@]}")
else
  grad_offload_modes_for_job=(false)
fi

for grad_offload in "${grad_offload_modes_for_job[@]}"; do
  if ! run_job \
      "${backend}" "${profiler}" "${recompute}" "${seq_len}" "${gpu}" "${gpu_count}" \
      "${expert_policy}" "${job_router_mode}" "${current_dataset}" "${lf_expert_lora_impl}" \
      "${grad_offload}"; then
    failures=$((failures + 1))
    if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
      exit 1
    fi
  fi
done

run_job() {
  local backend="$1"
  local profiler="$2"
  local recompute="$3"
  local seq_len="$4"
  local gpu="$5"
  local gpu_count="$6"
  local expert_policy="$7"
  local router_mode="$8"
  local dataset_name="$9"
  local lf_expert_lora_impl="${10}"
  local grad_offload="${11:-false}"

  if ! cpuadam_backend_for_label "${backend}" >/dev/null; then
    grad_offload=false
  fi

  job_root="$(job_root_path "${config_root}" "${backend}" "${profiler}" "${recompute}" "${expert_policy}" "${router_mode}" "${lf_expert_lora_impl}" "${grad_offload}")"
  run_id="..._gradoff${grad_offload}_..."

  run_env+=(
    ASYM_CPU_ADAMW_GRAD_OFFLOAD="${grad_offload}"
    ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_GRAD_OFFLOAD="${grad_offload}"
  )
}
```

7. Include the axis in output paths and profile-completeness checks:

```bash
job_root_path() {
  local config_root="$1"
  local backend="$2"
  local profiler="$3"
  local recompute="$4"
  local expert_policy="$5"
  local router_mode="$6"
  local lf_expert_lora_impl_value="${7:-${lf_expert_lora_impl:-split-target-parameters}}"
  local grad_offload="${8:-false}"
  local suffix=""
  if cpuadam_backend_for_label "${backend}" >/dev/null; then
    suffix="__gradoff${grad_offload}"
  fi
  printf '%s/%s\n' "${config_root}" "$(safe_label "${backend}__${profiler}__${recompute}__pol${expert_policy}__router${router_mode}__${expact_label}__${attnact_label}__${layeract_label}__${expact_lora_a_fwd_label}__qwenexpert${lf_expert_lora_impl_value}${suffix}")"
}

ensure_jobs_tsv() {
  ...
  printf 'status\tgpu\tseq_len\trecompute\texpert_policy\trouter_mode\tbackend\tprofiler\tgrad_offload\tjob_dir\tprofile_json\tlog\tqwen_expert_lora_impl\texpert_lora_a_fwd\n' > "${config_root}/jobs.tsv"
}

append_job_record() {
  local config_root="$1"
  local status="$2"
  shift 2
  ensure_jobs_tsv "${config_root}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${status}" "$@" "${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}" >> "${config_root}/jobs.tsv"
}
```

Add an `expected_grad_offload` argument to `existing_profile_complete` and compare it against `source_profile.config.asym_cpu_adamw_grad_offload` for CPUAdamW labels. This prevents `COLLECT_EXISTING=true` or skip logic from accepting a stale `gradofffalse` profile for a `gradofftrue` run.

```bash
existing_profile_complete() {
  ...
  local expected_expact_lora_a_fwd="${12:-}"
  local expected_grad_offload="${13:-}"
  ...
  "${ENV_PYTHON}" - "${profile_json}" ... "${expected_expact_lora_a_fwd}" "${expected_grad_offload}" <<'PY'
...
expected_grad_offload = sys.argv[22] if len(sys.argv) > 22 else ""
if expected_grad_offload:
    actual = str(config.get("asym_cpu_adamw_grad_offload", "")).lower()
    expected = str(expected_grad_offload).lower()
    if actual != expected:
        raise SystemExit(f"grad offload mismatch: expected {expected}, got {actual}")
PY
}
```

Thread `expected_grad_offload` through every `existing_profile_complete` call in `run_job`, including skip, `COLLECT_EXISTING=true`, and KT matching-source helpers. Thread `grad_offload` through every `append_job_record` call in `run_job`, including `skipped`, `dry-run`, `ok`, and `failed:*`.

8. Store the value in source profiles:

```python
# scripts/lf/run_lf_profiled_train.py::_config_from_args
"asym_cpu_adamw_grad_offload": (
    os.environ.get("ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_GRAD_OFFLOAD")
    or _option_value(args, "--asym_cpu_adamw_grad_offload")
    or "false"
).lower() in {"1", "true", "yes", "on"},
```

Risks to watch:

- If `job_root_path` changes for all existing CPUAdamW profiles, old collection/skip behavior may miss old profiles. Keep `legacy_job_root_path` fallback only for `grad_offload=false`.
- `profile_lora_lf.sh` and `test_profiling.sh` are similar but not identical. Patch both; do not assume one covers the other.

Validation before Stage 2:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
export ENV_PYTHON=${ENV_PYTHON:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python}

"${ENV_PYTHON}" -m pytest -q \
  tests/lf/test_asym_cpu_adamw_args.py \
  tests/lf/test_asym_cpu_adamw_lf_integration.py

OUT=/tmp/asym_lora_grad_offload_stage1_dryrun
rm -rf "${OUT}"
BACKEND_SPECS="asym_cpuadamwds|recomp" \
ASYM_CPU_ADAMW_GRAD_OFFLOADS="false,true" \
PROFILERS=source \
SEQ_LENS=128 \
MAX_STEPS=1 \
WARMUP_STEPS=0 \
PREPARE_DATASETS=false \
DRY_RUN=true \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf.sh --gpus 0 --output-root "${OUT}"

rg -n "ASYM_CPU_ADAMW_GRAD_OFFLOAD=false|--asym_cpu_adamw_grad_offload false|gradofffalse" "${OUT}"
rg -n "ASYM_CPU_ADAMW_GRAD_OFFLOAD=true|--asym_cpu_adamw_grad_offload true|gradofftrue" "${OUT}"
```

## Stage 2: Implement ZeRO-Style Hook-Time LoRA Grad Offload in `AsymCPUAdamW`

Scope:

- `asym_gemm/training/cpu_adam.py`
  - `_ParamMapping`
  - `AsymCPUAdamW.__init__`
  - `_check_post_prepare_devices`
  - `_ensure_grad_buffer`
  - new `_ensure_grad_offload_flat_buffer`
  - new `_register_grad_offload_hooks`
  - new `_offload_grad_from_hook`
  - new `_copy_or_accumulate_grad_to_cpu`
  - `step`
  - `zero_grad`
  - `state_dict`
  - `load_state_dict`
  - `asym_cpu_adamw_summary`
- Tests:
  - `tests/training/test_asym_cpu_adamw.py`

Implementation:

Memory and efficiency design:

- Current path: autograd leaves every LoRA CUDA `.grad` tensor live until `AsymCPUAdamW.step()`. Step then runs a Python loop over every LoRA param, does `CPU fp32 grad_buffer.copy_(CUDA bf16/fp32 grad)`, runs CPUAdam, copies CPU masters back to CUDA, and only later does `zero_grad()` clear CUDA grads.
- New path: autograd creates one LoRA param grad, the post-accumulate hook copies that grad into a CPU fp32 view, and the hook immediately clears `param.grad`. The D2H bytes are roughly the same as the current step-time copy, but the CUDA lifetime is shorter, so peak HBM should drop by close to the live LoRA grad footprint.
- CPU memory is not free: the flat CPU grad buffer is `param_numel * 4` bytes because CPUAdam consumes fp32 grads. This is comparable to the current lazy per-param CPU grad buffers, but it is allocated as one contiguous slab to avoid per-step allocation churn and duplicate storage accounting.
- Kernel/copy behavior: each `copy_` is still one D2H copy operation with dtype conversion when model grads are bf16. This does not add GEMM kernels or change AsymGEMM math. It moves copy timing from optimizer step into backward hooks.
- Do not batch by keeping CUDA grads in a Python list for a later grouped copy. That would preserve the HBM peak we are trying to remove. If copy overhead is too high, Stage 6 may use a dedicated copy stream, but it must still release each source grad promptly.
- Gradient accumulation with `GRADIENT_ACCUMULATION_STEPS>1` is supported by accumulating into the CPU grad buffer. This is memory-conservative but can be slower than DeepSpeed's GPU-side accumulation trick. The e2e acceptance uses `GRADIENT_ACCUMULATION_STEPS=1`; if accumulation is needed later, add a separate A/B gate for `GRADIENT_ACCUMULATION_STEPS=2`.

1. Extend the mapping state. Keep existing `grad_buffer` field, but make it a view into one flat CPU grad slab when `grad_offload=true`.

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
    grad_buffer_has_data: bool = False
    hook_calls: int = 0
    offloaded_grad_numel: int = 0
```

2. Add constructor args and state:

```python
def __init__(..., fp32_master: bool = True, grad_offload: bool = False) -> None:
    ...
    self.grad_offload = bool(grad_offload)
    self._grad_offload_handles: list[Any] = []
    self._grad_flat_buffer: torch.Tensor | None = None
    self._grad_accum_staging_buffer: torch.Tensor | None = None
    self._current_hook_grad_copy_ms = 0.0
    self._last_hook_offloaded_param_count = 0
    self._last_hook_offloaded_numel = 0
    self._last_hook_call_count = 0
    self._last_hook_grad_copy_ms = 0.0
    self._last_step_used_offloaded_grads = False
    ...
    if self.grad_offload:
        self._ensure_grad_offload_flat_buffer()
        self._register_grad_offload_hooks()
```

3. Allocate one contiguous CPU grad buffer, like DeepSpeed's `grad_partitions_flat_buffer`, but unsharded because this plan is single GPU only.

```python
def _ensure_grad_offload_flat_buffer(self) -> torch.Tensor:
    total = sum(int(mapping.cpu_param.numel()) for mapping in self._mappings)
    current = self._grad_flat_buffer
    if current is None or current.numel() != total or current.dtype != torch.float32:
        flat = torch.empty(total, device="cpu", dtype=torch.float32)
        flat, pin_error = _pin_if_requested(flat, pin_memory=self.pin_memory)
        if pin_error is not None:
            self._pin_memory_failures.append(f"grad_flat_buffer: {pin_error}")
        self._grad_flat_buffer = flat

        offset = 0
        for mapping in self._mappings:
            numel = int(mapping.cpu_param.numel())
            mapping.grad_buffer = flat.narrow(0, offset, numel).view_as(mapping.cpu_param.data)
            offset += numel
    return self._grad_flat_buffer

def _ensure_grad_buffer(self, mapping: _ParamMapping) -> torch.Tensor:
    if self.grad_offload:
        if mapping.grad_buffer is None:
            self._ensure_grad_offload_flat_buffer()
        if mapping.grad_buffer is None:
            raise RuntimeError(f"missing offload grad buffer for {mapping.name}")
        return mapping.grad_buffer

    # Existing per-param lazy CPU grad buffer path for grad_offload=false.
    ...
```

4. Register post-accumulate hooks. Mirror DeepSpeed's compatibility helper: use PyTorch's native hook for torch >= 2.1, else the grad-accumulator fallback. The hook must be on the CUDA compute param, not the CPU master.

```python
def _register_post_accumulate_hook(param: torch.nn.Parameter, hook: Callable[[torch.Tensor], None]) -> Any:
    register = getattr(param, "register_post_accumulate_grad_hook", None)
    if callable(register):
        return register(hook)

    # DeepSpeed fallback for older torch.
    param_tmp = param.expand_as(param)
    grad_acc = param_tmp.grad_fn.next_functions[0][0]
    return grad_acc.register_hook(lambda *unused: hook(param))

def _register_grad_offload_hooks(self) -> None:
    if self._grad_offload_handles:
        return
    for mapping in self._mappings:
        if not mapping.cuda_param.is_leaf:
            raise RuntimeError(f"AsymCPUAdamW grad offload requires leaf CUDA params; got {mapping.name}")
        self._grad_offload_handles.append(
            _register_post_accumulate_hook(
                mapping.cuda_param,
                lambda param, mapping=mapping: self._offload_grad_from_hook(mapping, param),
            )
        )
```

5. Hook behavior: copy the just-produced CUDA grad into the CPU buffer and immediately clear CUDA `.grad`. This is the key memory fix.

```python
@torch.no_grad()
def _offload_grad_from_hook(self, mapping: _ParamMapping, param: torch.Tensor) -> None:
    if not self.grad_offload:
        return
    if param is not mapping.cuda_param:
        raise RuntimeError(f"Grad hook param mismatch for {mapping.name}")

    grad = mapping.cuda_param.grad
    if grad is None:
        return
    if grad.is_sparse:
        raise RuntimeError("AsymCPUAdamW grad offload does not support sparse LoRA grads.")
    if grad.device.type != "cuda":
        raise RuntimeError(f"AsymCPUAdamW grad offload expected CUDA grad for {mapping.name}, got {grad.device}")
    if tuple(grad.shape) != tuple(mapping.cpu_param.shape):
        raise RuntimeError(f"AsymCPUAdamW grad shape mismatch for {mapping.name}")

    started = time.perf_counter()
    self._copy_or_accumulate_grad_to_cpu(mapping, grad.detach())
    self._current_hook_grad_copy_ms += (time.perf_counter() - started) * 1000.0

    mapping.last_had_grad = True
    mapping.grad_buffer_has_data = True
    mapping.hook_calls += 1
    mapping.offloaded_grad_numel += int(grad.numel())
    mapping.cpu_param.grad = mapping.grad_buffer

    # Free the model-side CUDA grad like ZeRO-3 partition_grads().
    mapping.cuda_param.grad = None
```

6. Copy/accumulate logic. Start with synchronous copies for correctness and peak-memory validation. Do not add async copy until Stage 5 because retaining source grads incorrectly would erase the memory win.

```python
def _ensure_grad_accum_staging_buffer(self, numel: int) -> torch.Tensor:
    current = self._grad_accum_staging_buffer
    if current is None or current.numel() < numel:
        staging = torch.empty(numel, device="cpu", dtype=torch.float32)
        staging, pin_error = _pin_if_requested(staging, pin_memory=self.pin_memory)
        if pin_error is not None:
            self._pin_memory_failures.append(f"grad_accum_staging_buffer: {pin_error}")
        self._grad_accum_staging_buffer = staging
    return self._grad_accum_staging_buffer.narrow(0, 0, numel)

def _copy_or_accumulate_grad_to_cpu(self, mapping: _ParamMapping, cuda_grad: torch.Tensor) -> None:
    grad_buffer = self._ensure_grad_buffer(mapping)
    if grad_buffer.dtype != mapping.cpu_param.dtype:
        raise RuntimeError(...)
    if not mapping.grad_buffer_has_data:
        grad_buffer.copy_(cuda_grad, non_blocking=False)
        return

    # Correct gradient accumulation across multiple backward calls before optimizer.step().
    staging = self._ensure_grad_accum_staging_buffer(int(cuda_grad.numel())).view_as(grad_buffer)
    staging.copy_(cuda_grad, non_blocking=False)
    grad_buffer.add_(staging)
```

7. Update `step()` to consume already-offloaded CPU grads. The old path remains untouched when `grad_offload=false`.

```python
def step(self, closure=None):
    ...
    self._copy_group_hyperparameters_to_inner()
    grad_param_count = 0
    skipped_no_grad = 0
    copyback_count = 0

    with torch.no_grad():
        if self.grad_offload:
            for mapping in self._mappings:
                if mapping.grad_buffer_has_data:
                    mapping.cpu_param.grad = mapping.grad_buffer
                    mapping.last_had_grad = True
                    grad_param_count += 1
                else:
                    mapping.cpu_param.grad = None
                    mapping.last_had_grad = False
                    skipped_no_grad += 1
            self._last_grad_copy_ms = 0.0
            self._last_step_used_offloaded_grads = grad_param_count > 0
            self._last_hook_offloaded_param_count = grad_param_count
            self._last_hook_offloaded_numel = sum(
                int(mapping.offloaded_grad_numel) for mapping in self._mappings if mapping.last_had_grad
            )
            self._last_hook_call_count = sum(int(mapping.hook_calls) for mapping in self._mappings)
            self._last_hook_grad_copy_ms = self._current_hook_grad_copy_ms
        else:
            # Existing step-time CUDA grad copy path.
            ...

        if grad_param_count:
            self.inner_optimizer.step()
        self._refresh_visible_state()

        for mapping in self._mappings:
            if mapping.last_had_grad:
                self._copy_master_to_compute_param(mapping)
                copyback_count += 1

    self._last_step_grad_param_count = grad_param_count
    ...
```

8. Update `zero_grad()` so CPU offload buffers are reusable but logically empty.

```python
def zero_grad(self, set_to_none: bool = True) -> None:
    super().zero_grad(set_to_none=True if self.grad_offload else set_to_none)
    try:
        self.inner_optimizer.zero_grad(set_to_none=True)
    except TypeError:
        self.inner_optimizer.zero_grad()

    for mapping in self._mappings:
        mapping.cuda_param.grad = None
        mapping.cpu_param.grad = None
        mapping.grad_buffer_has_data = False
        mapping.last_had_grad = False
        mapping.hook_calls = 0
        mapping.offloaded_grad_numel = 0

    self._current_hook_grad_copy_ms = 0.0
```

9. Extend summaries and checkpoints:

```python
def state_dict(self):
    state = {
        "format": "asym_cpu_adamw_v1",
        "backend": self.backend,
        "pin_memory": self.pin_memory,
        "fp32_master": self.fp32_master,
        "grad_offload": self.grad_offload,
        ...
    }

def load_state_dict(self, state_dict):
    # Do not require checkpoint grad_offload to match; it is runtime behavior, not optimizer math state.
    # Still accept and expose the saved value for debugging if present.
    ...

def asym_cpu_adamw_summary(self):
    grad_buffer_bytes = _tensor_storage_nbytes(self._grad_flat_buffer) if self._grad_flat_buffer is not None else ...
    return {
        ...
        "grad_offload_enabled": self.grad_offload,
        "grad_offload_hook_count": len(self._grad_offload_handles),
        "grad_offload_buffer_bytes": int(grad_buffer_bytes),
        "pinned_grad_offload_buffer_bytes": int(pinned_grad_bytes),
        "last_step_used_offloaded_grads": bool(self._last_step_used_offloaded_grads),
        "last_hook_offloaded_param_count": int(self._last_hook_offloaded_param_count),
        "last_hook_offloaded_numel": int(self._last_hook_offloaded_numel),
        "last_hook_call_count": int(self._last_hook_call_count),
        "hook_grad_copy_ms": float(self._last_hook_grad_copy_ms),
    }
```

Risks to watch:

- If hooks are registered after optimizer creation but before `accelerator.prepare`, single-GPU behavior should be stable. If a future wrapper replaces parameter objects, hook registration must move after final model placement.
- Gradient accumulation is correct but may be slower when `GRADIENT_ACCUMULATION_STEPS>1` because CPU accumulation is synchronous. This is acceptable for the first grad-memory validation; Stage 5 can optimize if the e2e timing shows it matters.
- `param.grad = None` inside the hook is required for memory savings. Any later code that assumes CUDA `.grad` exists must be moved to the CPU-buffer path in Stage 3.

Validation before Stage 3:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
export ENV_PYTHON=${ENV_PYTHON:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python}

# Hook semantics probe. Use a Python environment with torch; the local LF venv currently reports torch 2.12.0+cu130.
/home/kevinni/AsymGEMM-SFT/third_party/LlamaFactory/.venv/bin/python - <<'PY'
import torch

p = torch.nn.Parameter(torch.tensor([1.0]))
events = []

def hook(param):
    events.append((param is p, None if param.grad is None else param.grad.detach().clone().tolist()))
    param.grad = None

handle = p.register_post_accumulate_grad_hook(hook)
(p * 3).sum().backward()
print("events", events)
print("grad_after", p.grad)
handle.remove()
if events != [(True, [3.0])]:
    raise SystemExit(f"unexpected post-accumulate hook semantics: {events}")
if p.grad is not None:
    raise SystemExit(f"hook failed to clear grad: {p.grad}")
PY

"${ENV_PYTHON}" -m pytest -q tests/training/test_asym_cpu_adamw.py

# Required new tests inside tests/training/test_asym_cpu_adamw.py:
# - grad_offload=false matches existing CPU reference behavior
# - grad_offload=true autograd hook clears every CUDA LoRA param.grad after backward
# - grad_offload=true step updates CUDA LoRA params identically to grad_offload=false for one backward
# - two backward calls before step accumulate CPU grad buffers correctly
# - zero_grad clears CPU grad ownership and leaves CUDA param.grad None
# - deepspeed backend still uses DeepSpeedCPUAdam when extension is available

OUT=/tmp/asym_lora_grad_offload_stage2_smoke_$(date -u +%Y%m%dT%H%M%SZ)
BACKEND_SPECS="asym_cpuadamwds|recomp" \
ASYM_CPU_ADAMW_GRAD_OFFLOADS="false,true" \
PROFILERS=source \
SEQ_LENS=512 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 \
MAX_STEPS=2 \
WARMUP_STEPS=0 \
MAX_SAMPLES=8 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
MAX_GRAD_NORM=0 \
ASYMM_EXP_ACT_POLICIES="none|false|false|false" \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf.sh --gpus 0 --output-root "${OUT}"

"${ENV_PYTHON}" - <<'PY' "${OUT}"
import json
import pathlib
import sys

profiles = sorted(pathlib.Path(sys.argv[1]).rglob("profile.json"))
if len(profiles) != 2:
    raise SystemExit(f"expected two Stage 2 A/B profiles, found {len(profiles)}")

seen = {}
for path in profiles:
    data = json.loads(path.read_text())
    source = data.get("source_profile") if isinstance(data.get("source_profile"), dict) else data
    config = source.get("config", {})
    cpu = source.get("asym_cpu_adamw", {})
    optimizer_memory = source.get("optimizer_memory", {})
    mode = bool(config.get("asym_cpu_adamw_grad_offload"))
    step_copy_ms = cpu.get("step_grad_copy_ms", cpu.get("grad_copy_ms"))
    seen[mode] = {"path": str(path), "cpu": cpu, "optimizer_memory": optimizer_memory}
    print(json.dumps({
        "path": str(path),
        "mode": mode,
        "grad_offload_enabled": cpu.get("grad_offload_enabled"),
        "hook_params": cpu.get("last_hook_offloaded_param_count"),
        "grad_params": cpu.get("last_step_grad_param_count"),
        "hook_copy_ms": cpu.get("hook_grad_copy_ms"),
        "step_grad_copy_ms": step_copy_ms,
    }, sort_keys=True))

if set(seen) != {False, True}:
    raise SystemExit(f"missing false/true grad-offload modes: {seen.keys()}")
true_cpu = seen[True]["cpu"]
false_cpu = seen[False]["cpu"]
if true_cpu.get("grad_offload_enabled") is not True:
    raise SystemExit("grad_offload=true profile did not enable optimizer grad offload")
if false_cpu.get("grad_offload_enabled") is not False:
    raise SystemExit("grad_offload=false profile unexpectedly enabled optimizer grad offload")
if true_cpu.get("last_hook_offloaded_param_count") != true_cpu.get("last_step_grad_param_count"):
    raise SystemExit(f"hook/step grad param mismatch: {true_cpu}")
true_step_copy_ms = true_cpu.get("step_grad_copy_ms", true_cpu.get("grad_copy_ms"))
if true_step_copy_ms not in (0, 0.0):
    raise SystemExit(f"grad_offload=true still reports step-time grad copy: {true_cpu}")
if int(true_cpu.get("last_hook_offloaded_numel") or 0) <= 0:
    raise SystemExit(f"grad_offload=true did not report offloaded grad numel: {true_cpu}")
optimizer_memory = seen[True].get("optimizer_memory", {})
if isinstance(optimizer_memory, dict) and int(optimizer_memory.get("cuda_grad_bytes") or 0) != 0:
    raise SystemExit(f"grad_offload=true left model CUDA grads visible at optimizer summary time: {optimizer_memory}")
PY
```

## Stage 3: Add CPU-Buffer Grad Norm and Clipping for Offloaded Grads

Scope:

- `asym_gemm/training/cpu_adam.py`
  - new `asym_cpu_adamw_grad_offload_enabled`
  - new `asym_cpu_adamw_grad_buffers`
  - new `asym_cpu_adamw_grad_norm`
  - new `asym_cpu_adamw_clip_grad_norm_`
- `third_party/LlamaFactory/src/llamafactory/train/sft/trainer.py`
  - new helper `_asym_cpu_adamw_optimizer`
  - `CustomSeq2SeqTrainer._clip_grad_norm`
  - `CustomSeq2SeqTrainer._get_grad_norm`
- `scripts/lf/run_lf_profiled_train.py`
  - `_install_trainer_heartbeat_hooks`
- Tests:
  - `tests/training/test_asym_cpu_adamw.py`
  - `tests/lf/test_asym_cpu_adamw_lf_integration.py`
  - `tests/lf/test_lf_profile_postprocess.py`

Implementation:

1. Add optimizer APIs that operate only on CPU grad buffers with real data.

```python
def asym_cpu_adamw_grad_offload_enabled(self) -> bool:
    return bool(self.grad_offload)

def asym_cpu_adamw_grad_buffers(self) -> list[torch.Tensor]:
    return [
        mapping.grad_buffer
        for mapping in self._mappings
        if mapping.grad_buffer_has_data and mapping.grad_buffer is not None
    ]
```

2. Implement chunked CPU norm and clipping. Return `(total_norm, summary)` so LF profiler can record the path.

```python
def _iter_flat_chunks(tensor: torch.Tensor, chunk_elements: int):
    flat = tensor.detach().reshape(-1)
    for start in range(0, int(flat.numel()), chunk_elements):
        yield flat.narrow(0, start, min(chunk_elements, int(flat.numel()) - start))

def asym_cpu_adamw_grad_norm(self, norm_type: float = 2.0, chunk_elements: int = 8_388_608) -> torch.Tensor:
    grads = self.asym_cpu_adamw_grad_buffers()
    if not grads:
        return torch.zeros((), dtype=torch.float32, device="cpu")

    if norm_type == math.inf:
        max_abs = torch.zeros((), dtype=torch.float64, device="cpu")
        for grad in grads:
            for chunk in _iter_flat_chunks(grad, chunk_elements):
                max_abs = torch.maximum(max_abs, chunk.abs().max().to(device="cpu", dtype=torch.float64))
        return max_abs.to(dtype=torch.float32)

    total = torch.zeros((), dtype=torch.float64, device="cpu")
    for grad in grads:
        for chunk in _iter_flat_chunks(grad, chunk_elements):
            chunk_f32 = chunk if chunk.dtype in (torch.float32, torch.float64) else chunk.float()
            total += torch.sum(torch.abs(chunk_f32) ** norm_type, dtype=torch.float64).cpu()
    return total.pow(1.0 / norm_type).to(dtype=torch.float32)

def asym_cpu_adamw_clip_grad_norm_(self, max_norm: float, norm_type: float = 2.0, chunk_elements: int = 8_388_608):
    total_norm = self.asym_cpu_adamw_grad_norm(norm_type=norm_type, chunk_elements=chunk_elements)
    max_norm_f = float(max_norm)
    if math.isinf(max_norm_f):
        clip_coef = 1.0
        clipped = False
        max_norm_summary: float | str = "inf"
    else:
        clip_coef = max_norm_f / (float(total_norm.item()) + 1e-6)
        clipped = bool(clip_coef < 1.0)
        max_norm_summary = max_norm_f
    if clipped:
        for grad in self.asym_cpu_adamw_grad_buffers():
            grad.mul_(clip_coef)
    summary = {
        "enabled": True,
        "path": "asym_cpu_adamw_grad_offload",
        "max_norm": max_norm_summary,
        "norm_type": norm_type,
        "total_norm": float(total_norm.item()),
        "clip_coef": float(clip_coef),
        "clipped": clipped,
        "cpu_grad_tensors": len(self.asym_cpu_adamw_grad_buffers()),
        "cpu_grad_numel": sum(int(g.numel()) for g in self.asym_cpu_adamw_grad_buffers()),
        "cuda_grad_tensors": 0,
        "chunk_elements": chunk_elements,
    }
    return total_norm, summary
```

3. Route LF SFT clipping to this API when grad offload is active. Keep the existing KT path unchanged.

```python
def _asym_cpu_adamw_optimizer(optimizer: Any) -> Any | None:
    seen = set()
    current = optimizer
    for _ in range(5):
        if current is None or id(current) in seen:
            return None
        seen.add(id(current))
        enabled = getattr(current, "asym_cpu_adamw_grad_offload_enabled", None)
        if callable(enabled) and enabled():
            return current
        for attr in ("optimizer", "base_optimizer", "wrapped_optimizer"):
            next_optimizer = getattr(current, attr, None)
            if next_optimizer is not None:
                current = next_optimizer
                break
        else:
            return None

class CustomSeq2SeqTrainer(...):
    def _clip_grad_norm(self, model):
        asym_opt = _asym_cpu_adamw_optimizer(getattr(self, "optimizer", None))
        if asym_opt is not None:
            if hasattr(self.accelerator, "unscale_gradients"):
                self.accelerator.unscale_gradients()
            total_norm, summary = asym_opt.asym_cpu_adamw_clip_grad_norm_(float(self.args.max_grad_norm))
            self._asym_cpu_adamw_grad_clip_last_summary = summary
            _emit_asym_gemm_heartbeat("asym_cpu_adamw_grad_clip", trainer_class=self.__class__.__name__, **summary)
            return total_norm

        if not _is_kt_arm_backend(self.model_args):
            return super()._clip_grad_norm(model)
        ...

    def _get_grad_norm(self, model, grad_norm=None):
        asym_opt = _asym_cpu_adamw_optimizer(getattr(self, "optimizer", None))
        if asym_opt is not None and grad_norm is None:
            if hasattr(self.accelerator, "unscale_gradients"):
                self.accelerator.unscale_gradients()
            total_norm, summary = asym_opt.asym_cpu_adamw_clip_grad_norm_(float("inf"))
            summary["operation"] = "norm_only"
            self._asym_cpu_adamw_grad_clip_last_summary = summary
            _emit_asym_gemm_heartbeat("asym_cpu_adamw_grad_norm", trainer_class=self.__class__.__name__, **summary)
            return total_norm
        ...
```

4. Let the profiler wrapper pick up either Asym or KT grad-clip summaries.

```python
# scripts/lf/run_lf_profiled_train.py::_install_trainer_heartbeat_hooks
summary = getattr(self, "_asym_cpu_adamw_grad_clip_last_summary", None)
if not isinstance(summary, dict):
    summary = getattr(self, "_kt_grad_clip_last_summary", None)
if isinstance(summary, dict):
    record.update(summary)
else:
    record.update({"enabled": False, "path": "default"})
```

Risks to watch:

- If `Trainer` calls a non-`CustomSeq2SeqTrainer` clipping path for this workload, offloaded grads would be invisible to default clipping. The Stage 3 profiler test must prove the source profile records `path=asym_cpu_adamw_grad_offload`.
- `max_grad_norm=0` should behave as HF clipping does: the trainer normally skips clipping when disabled. If this method is called with `0`, it should zero grads only if HF would have called it. Validate with LF behavior before special-casing.

Validation before Stage 4:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
export ENV_PYTHON=${ENV_PYTHON:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python}

"${ENV_PYTHON}" -m pytest -q \
  tests/training/test_asym_cpu_adamw.py \
  tests/lf/test_asym_cpu_adamw_lf_integration.py \
  tests/lf/test_lf_profile_postprocess.py -k "grad_clip or asym_cpu_adamw"

OUT=/tmp/asym_lora_grad_offload_stage3_clip_$(date -u +%Y%m%dT%H%M%SZ)
BACKEND_SPECS="asym_cpuadamwds|recomp" \
ASYM_CPU_ADAMW_GRAD_OFFLOADS="true" \
PROFILERS=source \
SEQ_LENS=512 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 \
MAX_STEPS=2 \
WARMUP_STEPS=0 \
MAX_SAMPLES=8 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
MAX_GRAD_NORM=1.0 \
ASYMM_EXP_ACT_POLICIES="none|false|false|false" \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf.sh --gpus 0 --output-root "${OUT}"

"${ENV_PYTHON}" - <<'PY' "${OUT}"
import json
import pathlib
import sys

profiles = sorted(pathlib.Path(sys.argv[1]).rglob("profile.json"))
if len(profiles) != 1:
    raise SystemExit(f"expected one Stage 3 profile, found {len(profiles)}")
data = json.loads(profiles[0].read_text())
source = data.get("source_profile") if isinstance(data.get("source_profile"), dict) else data
cpu = source.get("asym_cpu_adamw", {})
clip = source.get("grad_clip", {})
print(json.dumps({
    "profile": str(profiles[0]),
    "grad_clip_path": clip.get("path"),
    "cpu_grad_numel": clip.get("cpu_grad_numel"),
    "hook_numel": cpu.get("last_hook_offloaded_numel"),
    "result_norm": clip.get("result_norm"),
}, sort_keys=True))
if clip.get("path") != "asym_cpu_adamw_grad_offload":
    raise SystemExit(f"grad clip did not use Asym offloaded CPU grads: {clip}")
if int(clip.get("cpu_grad_numel") or 0) != int(cpu.get("last_hook_offloaded_numel") or -1):
    raise SystemExit(f"grad clip CPU numel does not match offloaded hook numel: clip={clip}, cpu={cpu}")
if cpu.get("grad_offload_enabled") is not True:
    raise SystemExit(f"Asym CPUAdamW summary did not report grad_offload_enabled=true: {cpu}")
PY

# Required assertions:
# - No CUDA param.grad remains after backward in the grad-offload unit test.
# - The source profile records grad_clip.path == asym_cpu_adamw_grad_offload.
# - CPU grad numel in grad_clip equals AsymCPUAdamW last_hook_offloaded_numel.
```

## Stage 4: Expose Memory/Timing Counters and Postprocess Columns

Scope:

- `asym_gemm/training/cpu_adam.py`
  - `asym_cpu_adamw_summary`
- `scripts/lf/run_lf_profiled_train.py`
  - `_config_from_args`
  - `_asym_cpu_adamw_summary_from_trace`
- `scripts/lf/postprocess_lf_profile_artifacts.py`
  - `_asym_cpu_adamw_rows`
  - summary table generation if present
- `asym_gemm/profiling/lf_trace.py`
  - memory breakdown labels for CPU grad offload buffer if persistent-byte collection needs a new accessor
- Tests:
  - `tests/test_lf_memory_breakdown.py`
  - `tests/lf/test_lf_profile_postprocess.py`

Implementation:

1. Add stable summary fields:

```python
{
    "grad_offload_enabled": bool(self.grad_offload),
    "grad_offload_hook_count": len(self._grad_offload_handles),
    "grad_offload_buffer_bytes": int(grad_buffer_bytes),
    "pinned_grad_offload_buffer_bytes": int(pinned_grad_buffer_bytes),
    "last_step_used_offloaded_grads": bool(self._last_step_used_offloaded_grads),
    "last_hook_offloaded_param_count": int(self._last_hook_offloaded_param_count),
    "last_hook_offloaded_numel": int(self._last_hook_offloaded_numel),
    "last_hook_call_count": int(self._last_hook_call_count),
    "hook_grad_copy_ms": float(self._last_hook_grad_copy_ms),
    "step_grad_copy_ms": float(self._last_grad_copy_ms),
}
```

2. Do not rely on `optimizer_memory.cpu_grad_bytes` for offloaded grads. The current `scripts/lf/run_lf_profiled_train.py::_optimizer_memory_summary` only scans `model.parameters()` and their `.grad` fields. A correct grad-offload run clears model CUDA `.grad` and keeps grads inside `AsymCPUAdamW`, so the authoritative fields are `asym_cpu_adamw.grad_offload_buffer_bytes`, `last_hook_offloaded_numel`, and `optimizer_memory.cuda_grad_bytes == 0`.

3. Add a persistent CPU-memory accessor only if memory breakdown cannot infer the flat grad buffer from summary:

```python
def asym_cpu_adamw_grad_offload_buffer(self) -> torch.Tensor | None:
    return self._grad_flat_buffer
```

If `LFMemoryBreakdownProfiler` supports optimizer-specific accessors, label this storage as `optimizer_grad_cpu` or `offloaded_grad_cpu`, not `optimizer_state_cpu`, so CPU master/state/grad are not double-counted.

4. Keep `_asym_cpu_adamw_rows` generic, but add tests to require the new scalar columns in `asym_cpu_adamw.csv`.

```python
def _asym_cpu_adamw_rows(profile):
    cpuadamw = profile.get("asym_cpu_adamw", {})
    row = {key: value for key, value in cpuadamw.items() if not isinstance(value, (dict, list))}
    return [row] if row else []
```

Risks to watch:

- Memory breakdown must not count `mapping.grad_buffer` views multiple times. Count the flat storage once by storage pointer or use the single accessor.
- A successful grad-offload run may report `optimizer_memory.cpu_grad_bytes=0` because the generic model scanner cannot see optimizer-owned CPU grad buffers. Do not treat that as failure if `asym_cpu_adamw.grad_offload_buffer_bytes` and `last_hook_offloaded_numel` are correct.
- Step timing fields must distinguish hook-copy time from old step-copy time. Otherwise A/B results will look like the optimizer got faster just because copy time moved into backward.

Validation before Stage 5:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
export ENV_PYTHON=${ENV_PYTHON:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python}

"${ENV_PYTHON}" -m pytest -q \
  tests/test_lf_memory_breakdown.py \
  tests/lf/test_lf_profile_postprocess.py

OUT=/tmp/asym_lora_grad_offload_stage4_reporting_$(date -u +%Y%m%dT%H%M%SZ)
BACKEND_SPECS="asym_cpuadamwds|recomp" \
ASYM_CPU_ADAMW_GRAD_OFFLOADS="true" \
PROFILERS=source \
SEQ_LENS=512 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 \
MAX_STEPS=2 \
WARMUP_STEPS=0 \
MAX_SAMPLES=8 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
MAX_GRAD_NORM=1.0 \
ASYMM_EXP_ACT_POLICIES="none|false|false|false" \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_MEMORY_BREAKDOWN_INTERVAL=1 \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf.sh --gpus 0 --output-root "${OUT}"

PROFILE="$(find "${OUT}" -name profile.json | sort | tail -n 1)"
test -n "${PROFILE}"
CSV_DIR="$(dirname "${PROFILE}")"
test -f "${CSV_DIR}/asym_cpu_adamw.csv"
test -f "${CSV_DIR}/summary.md"
rg -n "grad_offload_enabled|grad_offload_hook_count|grad_offload_buffer_bytes|last_step_used_offloaded_grads|last_hook_offloaded_param_count|last_hook_offloaded_numel|hook_grad_copy_ms" "${CSV_DIR}/asym_cpu_adamw.csv"
rg -n "offloaded_grad_cpu|optimizer_grad_cpu|grad_offload_buffer_bytes" "${CSV_DIR}" || {
  echo "Expected offloaded grad CPU memory to appear in summary, CSV, or memory-breakdown artifacts" >&2
  exit 1
}
```

## Stage 5: E2E A/B Validation With Grad Offload On and Off

Scope:

- No new source changes unless Stage 5 finds a correctness or performance regression.
- This stage accepts or rejects the implementation with real LF LoRA profiling, not toy profiling.

Validation before Stage 6:

1. Run the smoke A/B first. This catches command routing, LF parser issues, hook registration, clipping, and postprocess output.

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
export ENV_PYTHON=${ENV_PYTHON:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python}

OUT=/tmp/asym_lora_grad_offload_smoke_$(date -u +%Y%m%dT%H%M%SZ)
BACKEND_SPECS="asym_cpuadamwds|recomp" \
ASYM_CPU_ADAMW_GRAD_OFFLOADS="false,true" \
PROFILERS=source \
SEQ_LENS=512 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 \
MAX_STEPS=2 \
WARMUP_STEPS=0 \
MAX_SAMPLES=8 \
PREPARE_DATASETS=false \
MAX_GRAD_NORM=1.0 \
ASYMM_EXP_ACT_POLICIES="none|false|false|false" \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf.sh --gpus 0 --output-root "${OUT}"

find "${OUT}" -name profile.json -print | sort
rg -n '"asym_cpu_adamw_grad_offload": (false|true)|"grad_offload_enabled": (false|true)|"path": "asym_cpu_adamw_grad_offload"' "${OUT}"
```

2. Run the meaningful A/B profile using the same workload shape for both modes. This is the acceptance run.

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
export ENV_PYTHON=${ENV_PYTHON:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python}

OUT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/profiling/lora_grad_offload_ab_$(date -u +%Y%m%dT%H%M%SZ)
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYM_CPU_ADAMW_GRAD_OFFLOADS="false,true" \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
PROFILERS=source \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=1 \
MAX_STEPS=10 \
WARMUP_STEPS=5 \
MAX_SAMPLES=128 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
LORA_RANK=64 \
LORA_ALPHA=16 \
LORA_DROPOUT=0.00 \
MAX_GRAD_NORM=1.0 \
ASYMM_EXP_ACT_POLICIES="none|true|true|true" \
PROFILE_MEMORY_BREAKDOWN=true \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf.sh --gpus 0 --output-root "${OUT}"
```

3. Compare the two runs from the same output root:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
export ENV_PYTHON=${ENV_PYTHON:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python}

"${ENV_PYTHON}" - <<'PY' "${OUT}"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.rglob("profile.json")):
    data = json.loads(path.read_text())
    source = data.get("source_profile") if isinstance(data.get("source_profile"), dict) else data
    config = source.get("config", {})
    cpu = source.get("asym_cpu_adamw", {})
    memory = source.get("memory", {}).get("gpu", {})
    step_rows = source.get("step_samples", {}).get("rows", [])
    measured = [
        row for row in step_rows
        if not row.get("is_warmup") and isinstance(row.get("step_milliseconds"), (int, float))
    ]
    if not measured:
        measured = [
            row for row in step_rows
            if isinstance(row.get("step_milliseconds"), (int, float))
        ]
    avg_step_ms = (
        sum(float(row["step_milliseconds"]) for row in measured) / len(measured)
        if measured
        else None
    )
    rows.append({
        "path": str(path),
        "grad_offload_config": config.get("asym_cpu_adamw_grad_offload"),
        "grad_offload_summary": cpu.get("grad_offload_enabled"),
        "peak_allocated_gib": (memory.get("peak_allocated_hbm_bytes") or source.get("memory", {}).get("peak_allocated_hbm_bytes") or 0) / 2**30,
        "peak_reserved_gib": (memory.get("peak_reserved_hbm_bytes") or source.get("memory", {}).get("peak_reserved_hbm_bytes") or 0) / 2**30,
        "hook_params": cpu.get("last_hook_offloaded_param_count"),
        "grad_params": cpu.get("last_step_grad_param_count"),
        "hook_numel": cpu.get("last_hook_offloaded_numel"),
        "hook_copy_ms": cpu.get("hook_grad_copy_ms"),
        "step_copy_ms": cpu.get("step_grad_copy_ms"),
        "cpu_adam_step_ms": cpu.get("cpu_adam_step_ms"),
        "weight_copyback_ms": cpu.get("weight_copyback_ms"),
        "avg_step_ms": avg_step_ms,
    })
for row in rows:
    print(json.dumps(row, sort_keys=True))

if len(rows) != 2:
    raise SystemExit(f"expected exactly two A/B profile.json files, found {len(rows)}")
configs = {row["grad_offload_config"] for row in rows}
if configs != {False, True}:
    raise SystemExit(f"expected grad_offload configs false/true, got {configs}")
true_row = next(row for row in rows if row["grad_offload_config"] is True)
false_row = next(row for row in rows if row["grad_offload_config"] is False)
if true_row["grad_offload_summary"] is not True:
    raise SystemExit("grad_offload=true run did not report grad_offload_enabled=true")
if not true_row["hook_params"] or true_row["hook_params"] != true_row["grad_params"]:
    raise SystemExit(f"offloaded hook param count mismatch: {true_row}")
if not (33.5 <= false_row["peak_allocated_gib"] <= 35.5):
    raise SystemExit(
        "grad_offload=false baseline is not the expected Qwen3 b4_s4096 "
        "exp+attn+layer activation-offload profile near 34.593 GiB: "
        f"off={false_row['peak_allocated_gib']} GiB"
    )
if true_row["peak_allocated_gib"] >= false_row["peak_allocated_gib"]:
    raise SystemExit(f"expected lower peak allocated HBM with grad offload: off={false_row['peak_allocated_gib']}, on={true_row['peak_allocated_gib']}")
if true_row["peak_allocated_gib"] > 30.5:
    raise SystemExit(
        "grad_offload=true missed target peak allocated HBM <= 30.5 GiB for "
        "Qwen3 b4_s4096 exp+attn+layer activation offload: "
        f"on={true_row['peak_allocated_gib']} GiB"
    )
if true_row["avg_step_ms"] is not None and false_row["avg_step_ms"] is not None:
    max_allowed = false_row["avg_step_ms"] * 1.15
    if true_row["avg_step_ms"] > max_allowed:
        raise SystemExit(
            "grad offload latency regression exceeds 15%: "
            f"off={false_row['avg_step_ms']} ms, on={true_row['avg_step_ms']} ms"
        )
PY
```

Acceptance criteria:

- Both A/B runs complete under the same output root with distinct `gradofffalse` and `gradofftrue` job paths.
- `grad_offload=false` preserves current behavior: CUDA grad copy happens in optimizer step and `grad_offload_enabled=false`.
- `grad_offload=true` reports `grad_offload_enabled=true`, `last_step_used_offloaded_grads=true`, and `last_hook_offloaded_param_count == last_step_grad_param_count` for the real profile.
- `grad_clip.path == asym_cpu_adamw_grad_offload` when `MAX_GRAD_NORM=1.0`.
- The `grad_offload=false` baseline for the acceptance workload must be near the known layer-offload profile: `33.5 GiB <= peak allocated HBM <= 35.5 GiB`. If it is outside that range, the command shape or profiler selection changed and the `<= 30.5 GiB` target is not comparable.
- Peak allocated HBM decreases in the `grad_offload=true` run.
- For the real `Qwen3-30B-A3B`, `b4_s4096`, `asym_cpuadamwds|norecomp`, `none|true|true|true` acceptance workload, `grad_offload=true` must reach peak allocated HBM `<= 30.5 GiB`. The expected value is about `30.0 GiB`: current best profile is `34.593 GiB`, CUDA LoRA grads are `6.287 GiB`, and the forward-only peak floor is about `29.988 GiB`.
- Do not require peak reserved HBM to hit `<= 30.5 GiB`; reserved HBM may remain higher due to CUDA allocator caching. Use peak allocated HBM for the hard memory target.
- Average measured `step_milliseconds` must not regress by more than 15% versus the same-run `grad_offload=false` baseline. If it does, inspect `hook_grad_copy_ms`, backward time, `cpu_adam_step_ms`, and `weight_copyback_ms` before proceeding to Stage 6.

Risks to watch:

- If the false baseline is not near `34.593 GiB`, do not reinterpret the `<= 30.5 GiB` target. First fix the run shape, stale-profile skip logic, or profiler completeness checks.
- If the true peak lands above `30.5 GiB` but `optimizer_memory.cuda_grad_bytes == 0`, inspect stage peaks. The target may be blocked by a new forward/workspace allocation, not by grad residency.

## Stage 6: Optional Performance Refinement if A/B Timing Regresses

Scope:

- `asym_gemm/training/cpu_adam.py`
  - `_offload_grad_from_hook`
  - `_copy_or_accumulate_grad_to_cpu`
  - `step`
  - `zero_grad`
- Tests:
  - `tests/training/test_asym_cpu_adamw.py`

Implementation:

Only do this after Stage 5 proves the sync hook implementation is correct and memory-meaningful. Add a dedicated copy stream for the first-copy case, but keep accumulation synchronous until a real `GRADIENT_ACCUMULATION_STEPS>1` profile needs more.

```python
def _init_grad_copy_stream(self):
    if self.grad_offload and torch.cuda.is_available() and self.pin_memory:
        self._grad_copy_stream = torch.cuda.Stream(device=self._mappings[0].cuda_param.device)

def _copy_or_accumulate_grad_to_cpu(self, mapping, cuda_grad):
    if mapping.grad_buffer_has_data or self._grad_copy_stream is None:
        # Existing synchronous path.
        ...
        return

    current_stream = torch.cuda.current_stream(cuda_grad.device)
    with torch.cuda.stream(self._grad_copy_stream):
        self._grad_copy_stream.wait_stream(current_stream)
        mapping.grad_buffer.copy_(cuda_grad, non_blocking=True)
        cuda_grad.record_stream(self._grad_copy_stream)
    mapping.pending_grad_copy = True

def _wait_for_grad_offload_copies(self):
    if self._grad_copy_stream is not None:
        torch.cuda.current_stream().wait_stream(self._grad_copy_stream)
```

Call `_wait_for_grad_offload_copies()` at the start of `step()`, before CPUAdam reads CPU grad buffers. Do not keep Python references to all CUDA grad tensors until `step()`, because that would preserve the HBM peak this feature is supposed to remove.

Risks to watch:

- Async D2H with dtype conversion may not be faster than sync for small tensors. The e2e profile decides.
- Incorrect stream lifetime handling can silently corrupt CPU grad buffers. Keep the sync path as a flag-controlled fallback until the async path has passed the same Stage 5 A/B gate.

Validation:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
export ENV_PYTHON=${ENV_PYTHON:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python}

"${ENV_PYTHON}" -m pytest -q tests/training/test_asym_cpu_adamw.py

OUT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/profiling/lora_grad_offload_async_ab_$(date -u +%Y%m%dT%H%M%SZ)
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYM_CPU_ADAMW_GRAD_OFFLOADS="false,true" \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
PROFILERS=source \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=1 \
MAX_STEPS=10 \
WARMUP_STEPS=5 \
MAX_SAMPLES=128 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
LORA_RANK=64 \
LORA_ALPHA=16 \
LORA_DROPOUT=0.00 \
MAX_GRAD_NORM=1.0 \
ASYMM_EXP_ACT_POLICIES="none|true|true|true" \
PROFILE_MEMORY_BREAKDOWN=true \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf.sh --gpus 0 --output-root "${OUT}"

# Run the exact Stage 5 comparison Python block against this OUT.
# The async refinement is accepted only if it preserves the memory reduction and passes the same 15% latency guard.
```
