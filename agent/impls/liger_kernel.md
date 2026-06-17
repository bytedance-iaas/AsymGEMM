# Staged Implementation Plan: Liger Loss-Only Axis

Goal: add a per-job `ligerloss0/ligerloss1` axis for Qwen3-MoE Liger fused linear cross entropy. The axis must work for AsymGEMM and ZeRO-style backends, but it must remain loss-only. Do not encode Liger in backend names. Do not enable Liger experts, SwiGLU, RoPE, RMSNorm, or any default Liger model patch when this profiling path asks for Liger.

Global accept/reject rule:
- Keep the feature only if full e2e `profile_lora_lf.sh` runs with `PROFILERS=both` show meaningful HBM reduction without timing blowup.
- Default pass threshold: peak CUDA allocated drops by at least 10 GiB and source-attributed `lm_head`/`loss` memory drops by at least 20 GiB.
- Reject if median measured step latency is above 1.10x no-Liger baseline, or median forward/backward latency is above 1.15x baseline.
- Reject same-memory/slower and trivial-memory/slower outcomes.
- Toy tests prove compatibility only. They are not acceptance evidence for memory or latency.

Canonical naming:
- `BACKEND_SPECS` is the only sweep input for this axis: `backend|recompute|ligerloss0` or `backend|recompute|ligerloss1`.
- The third field accepts exactly `ligerloss0` or `ligerloss1`. Do not add aliases like `true`, `1`, `liger`, or `noliger`.
- The third field is optional only for backward compatibility; missing means `ligerloss0`.
- Every new per-job output path, run ID, `jobs.tsv` row, profile config, plot row, and comparison row must contain `ligerloss0` or `ligerloss1`.
- The profile metadata key is exactly `liger_loss`. Do not add duplicate artifact keys such as `enable_liger_kernel`, `liger_enabled`, or `liger_loss_enabled`.
- `run_lf_lora_sft.sh` uses exactly two runtime env inputs: `ENABLE_LIGER_KERNEL` and `LIGER_LOSS_ONLY`.
- LF CLI args use LF's existing snake_case convention only: `--enable_liger_kernel` and `--liger_loss_only`.
- Plot CLIs use the existing plotting kebab-case convention only: `--liger-loss`.

Current codebase facts this plan must preserve:
- `scripts/lf/profile_lora_lf.sh` currently parses two-field backend specs in `append_backend_spec()` and stores specs as `backend|recompute`.
- `append_backend_spec()` currently supports `recompute=both` by expanding one backend into `backend|norecomp` and `backend|recomp`; keep this behavior while appending the Liger field.
- The main sweep loop currently does:
  - `backend="${backend_recompute%%|*}"`
  - `recompute="${backend_recompute##*|}"`
  This must be replaced with explicit three-field parsing. Otherwise `recompute` becomes `ligerloss1`.
- `PROFILERS=both` is already implemented as one Nsight run plus materialized source artifacts. With no `OUTPUT_ROOT`, it writes under `${ASYM_DIR}/profiling_both`; otherwise it uses the supplied `OUTPUT_ROOT`.
- `job_root_path()` currently includes backend, profiler, recompute, policy, router, `expact`, `attnact`, `layeract`, `loraafwd`, `actrecomp`, `xunpack`, and CPUAdam offload suffixes. Add `ligerloss0/1` without removing those axes.
- `run_lf_lora_sft.sh` accepts no command-line args. It is controlled through env vars and builds LF CLI args internally.
- `run_lf_profiled_train.py::_config_from_args()` already imports `ASYM_GEMM_LF_CONFIG_*` env vars into the profile config. Add an explicit `liger_loss` config entry anyway so missing metadata is easy to catch.
- The three plotting scripts currently reject unknown path-tail tokens unless they are parsed or listed as optional axes. `ligerloss0/1` must be parsed as a real axis, not silently ignored.

Checked Liger/LF facts:
- LF applies Liger before model load in `third_party/LlamaFactory/src/llamafactory/model/loader.py::load_model`.
- Local LF currently calls Liger with default kwargs for SFT, which can enable non-loss patches.
- Local Liger Qwen3-MoE exposes `fused_linear_cross_entropy`, `cross_entropy`, `rope`, `rms_norm`, and `swiglu`.
- Local Liger `apply_liger_kernel_to_qwen3_moe()` can patch only fused linear CE if called with `fused_linear_cross_entropy=True` and all other boolean patch flags false.
- With Transformers v5, default Liger Qwen3-MoE `swiglu=True` replaces `Qwen3MoeExperts` with `LigerExperts`; that must not happen for AsymGEMM.
- Upstream Liger supports source install with `pip install -e .`; use `--no-deps` here so pip cannot alter torch/triton. Sources: https://github.com/linkedin/Liger-Kernel and https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py

## Stage 0: Dependency And Current Baseline

Files/functions/classes:
- No production code changes.
- Read/verify:
  - `third_party/Liger-Kernel/setup.py`
  - `third_party/Liger-Kernel/src/liger_kernel/transformers/monkey_patch.py`
  - `third_party/Liger-Kernel/src/liger_kernel/transformers/model/qwen3_moe.py`
  - `third_party/LlamaFactory/src/llamafactory/model/model_utils/liger_kernel.py`

Implementation steps:
- Install the vendored Liger checkout into the exact venv used by the LF scripts.
- Use `--no-deps`; dependency churn invalidates profiling.
- Run an optional pre-change baseline with the current two-field `BACKEND_SPECS` format. Do not expect `ligerloss0` in this pre-change path because the current script cannot parse the third field yet.
- Treat this baseline as a sanity check only. Final acceptance must use the post-Stage-2 axis paths.

Commands:

```bash
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python - <<'PY'
from packaging.version import Version
import torch
import triton

assert Version(torch.__version__.split("+")[0]) >= Version("2.1.2"), torch.__version__
assert Version(triton.__version__) >= Version("2.3.1"), triton.__version__
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("triton", triton.__version__)
PY

/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python \
  -m pip install --no-deps -e /home/kevinni/AsymGEMM-SFT/third_party/Liger-Kernel

/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python - <<'PY'
import inspect
from importlib.metadata import version
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe

print("liger-kernel", version("liger-kernel"))
sig = inspect.signature(apply_liger_kernel_to_qwen3_moe)
required = {"fused_linear_cross_entropy", "cross_entropy", "rope", "rms_norm", "swiglu"}
assert not (required - set(sig.parameters)), sig
print(sig)
PY

OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/profiling_liger_prechange_baseline \
RUN_NAME=qwen3_asym_prechange_baseline \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
PROFILERS=both \
GPU_POOL=3 \
WARMUP_STEPS=5 \
MAX_STEPS=5 \
OVERWRITE=true \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Validation before Stage 1:
- Liger import/version/signature check passes in `${ASYM_DIR}/.venv`.
- Optional pre-change baseline completes and produces `profile.json`, `summary.md`, source-memory artifacts, and Nsight artifacts.
- Train log confirms Qwen3-MoE modules were wrapped by AsymGEMM.
- Record baseline peak allocated/reserved, `lm_head`/`loss` source memory, forward median, backward median, and step median.

Risks to watch:
- First Liger/Triton use can include compile overhead. Acceptance timing must use measured post-warmup steps only.
- Pre-change artifacts do not have the `liger_loss` axis and must not be mixed with final acceptance artifacts.

## Stage 1: LF Loss-Only Resolver

Files/functions/classes:
- Modify `third_party/LlamaFactory/src/llamafactory/hparams/model_args.py`
  - Add `liger_loss_only: bool = False`.
- Modify `third_party/LlamaFactory/src/llamafactory/model/model_utils/liger_kernel.py`
  - `apply_liger_kernel(config, model_args, is_trainable, require_logits)`
  - add `LigerApplySpec`
  - add `_LIGER_APPLY_SPECS`
  - add `_resolve_liger_apply(model_type)`
  - add `_build_liger_loss_only_kwargs(apply_fn)`
- Add `tests/lf/test_liger_loss_only_qwen3_moe.py`
  - `test_qwen3_moe_loss_only_kwargs_disable_non_loss_patches`
  - `test_asym_liger_skips_unvalidated_model_type`
  - `test_zero3_liger_loss_only_uses_same_loss_patch`
  - `test_qwen3_moe_loss_only_preserves_hf_experts_for_asym_wrap`
  - `test_qwen3_moe_loss_only_cuda_forward_backward`

Implementation steps:
- Preserve current non-loss-only Liger behavior for users outside this profiling path.
- Add the explicit `liger_loss_only` LF model arg so ZeRO-3 can use the same loss-only path as AsymGEMM.
- Treat `model_args.use_asym_gemm=True` as requiring loss-only behavior whenever Liger is enabled.
- When loss-only is active, call the model-specific Liger apply function with `fused_linear_cross_entropy=True` and every other boolean patch option set to `False`.
- Initially mark only `qwen3_moe` as validated for loss-only Liger.
- Skip loss-only Liger if `require_logits=True`.
- Skip loss-only Liger for unvalidated model types such as `llama4`/`llama4_text`.

Pseudocode:

```python
@dataclass(frozen=True)
class LigerApplySpec:
    model_types: tuple[str, ...]
    import_name: str
    loss_only_supported: bool = False


_LIGER_APPLY_SPECS = (
    LigerApplySpec(("qwen3_moe",), "apply_liger_kernel_to_qwen3_moe", loss_only_supported=True),
    LigerApplySpec(("qwen3",), "apply_liger_kernel_to_qwen3"),
    LigerApplySpec(("qwen3_next",), "apply_liger_kernel_to_qwen3_next"),
    LigerApplySpec(("qwen3_5",), "apply_liger_kernel_to_qwen3_5"),
    # Preserve existing local mappings: gemma, llama, mistral, mixtral, phi3, etc.
)


def _build_liger_loss_only_kwargs(apply_fn):
    sig = inspect.signature(apply_fn)
    if "fused_linear_cross_entropy" not in sig.parameters:
        return None

    kwargs = {}
    for name, param in sig.parameters.items():
        if name == "model":
            continue
        if name == "fused_linear_cross_entropy":
            kwargs[name] = True
        elif isinstance(param.default, bool):
            kwargs[name] = False
    if "cross_entropy" in sig.parameters:
        kwargs["cross_entropy"] = False
    return kwargs


def apply_liger_kernel(config, model_args, is_trainable, require_logits):
    if not is_trainable or not model_args.enable_liger_kernel:
        return

    apply_fn, spec = _resolve_liger_apply(getattr(config, "model_type", None))
    if apply_fn is None:
        logger.warning_rank0("Current model does not support liger kernel.")
        return

    loss_only = bool(getattr(model_args, "liger_loss_only", False) or getattr(model_args, "use_asym_gemm", False))
    if loss_only:
        if require_logits:
            logger.warning_rank0("Skipping Liger loss-only because logits are required.")
            return
        if spec is None or not spec.loss_only_supported:
            logger.warning_rank0("Skipping Liger loss-only: model type is not validated.")
            return
        kwargs = _build_liger_loss_only_kwargs(apply_fn)
        if kwargs is None:
            logger.warning_rank0("Skipping Liger loss-only: fused CE is unavailable.")
            return
        apply_fn(**kwargs)
        logger.info_rank0("Liger loss-only kernel has been applied.")
        return

    # Existing non-loss-only behavior.
    if require_logits and "fused_linear_cross_entropy" in inspect.signature(apply_fn).parameters:
        apply_fn(fused_linear_cross_entropy=False, cross_entropy=True)
    else:
        apply_fn()
```

Validation before Stage 2:

```bash
PYTHONPATH=/home/kevinni/AsymGEMM-SFT/third_party/LlamaFactory/src:/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python \
  -m pytest /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/tests/lf/test_liger_loss_only_qwen3_moe.py -q
```

Required evidence:
- Qwen3-MoE loss-only kwargs are exactly loss-only: `fused_linear_cross_entropy=True`; `cross_entropy/rope/rms_norm/swiglu=False`.
- The same loss-only kwargs are used when `use_asym_gemm=True` and when `liger_loss_only=True` for a non-Asym ZeRO-style job.
- `llama4_text` or another unvalidated model type is skipped in loss-only mode.
- With Qwen3-MoE, Liger patches only the causal-LM forward; HF `Qwen3MoeExperts` remains recognizable.
- AsymGEMM still wraps the MoE block as `AsymQwen3MoeBlock`.
- Tiny CUDA forward/backward produces finite loss, `outputs.logits is None`, and expected LoRA grads exist.

Risks to watch:
- Liger monkey-patches global classes. Tests that compare patched/unpatched behavior must run in fresh Python subprocesses.
- Future Liger versions may add new boolean patch toggles. `_build_liger_loss_only_kwargs` must keep disabling all boolean toggles except `fused_linear_cross_entropy`.

## Stage 2: Run/Profile Path, Metadata, And Plot Axis

Files/functions/classes:
- Modify `scripts/lf/run_lf_lora_sft.sh`
  - user/env defaults: add `ENABLE_LIGER_KERNEL=${ENABLE_LIGER_KERNEL:-false}` and `LIGER_LOSS_ONLY=${LIGER_LOSS_ONLY:-false}`
  - bool normalization block near the other user params
  - `DEFAULT_RUN_ID` construction
  - `CMD_ARGS`
  - logging block
  - `RUN_ENV`
  - profile env block that emits `ASYM_GEMM_LF_CONFIG_*`
- Modify `scripts/lf/profile_lora_lf.sh`
  - usage text and path examples
  - `append_backend_spec()`
  - backend-spec normalization around `backend_specs_raw`, `backend_specs`, `backends`, and `recompute_modes`
  - new `liger_loss_modes` array for plot filters
  - `job_root_path()`
  - `kt_arm_matching_source_profile_json_candidates()` only as needed to pass/default the new argument
  - `ensure_jobs_tsv()`
  - `append_job_record()`
  - `job_profile_complete()`
  - `existing_profile_complete()`
  - `append_sweep_plot_filters()`
  - `memory_plot_filters()`
  - `interconnect_plot_filters()`
  - `run_job()`
  - main sweep loop that currently parses `backend_recompute`
- Modify `scripts/lf/run_lf_profiled_train.py`
  - `_config_from_args()`
- Modify plotting scripts:
  - `scripts/plotting/plot_activation_recompute_sweep.py`
  - `scripts/plotting/plot_lf_memory_breakdown.py`
  - `scripts/plotting/plot_lf_interconnect_ctc.py`
- Modify tests:
  - `tests/lf/test_asym_cpu_adamw_args.py`
  - add `tests/lf/test_liger_loss_plot_axes.py`

Implementation steps:
- Keep backend labels unchanged. Liger is not a backend.
- Extend `BACKEND_SPECS` from `backend|recompute` to `backend|recompute|ligerloss`.
- Missing third field normalizes to `ligerloss0`, but all new output paths still include `__ligerloss0`.
- Keep `recompute=both` expansion:
  - `asym_cpuadamwds|both` becomes `asym_cpuadamwds|norecomp|ligerloss0` and `asym_cpuadamwds|recomp|ligerloss0`.
  - `asym_cpuadamwds|both|ligerloss1` becomes `asym_cpuadamwds|norecomp|ligerloss1` and `asym_cpuadamwds|recomp|ligerloss1`.
- Add `ligerloss0/1` to both the Nsight job folder and the materialized source sibling folder when `PROFILERS=both`.
- Generated profile configs must contain `config["liger_loss"]`.
- New plot output must keep `ligerloss0` and `ligerloss1` as separate series/groups. Do not collapse them into the same plot row.

`profile_lora_lf.sh` pseudocode:

```bash
liger_loss_label() {
  case "${1}" in
    ligerloss0|ligerloss1) printf '%s\n' "${1}" ;;
    *) die "liger loss field must be exactly ligerloss0 or ligerloss1, got '${1}'" ;;
  esac
}

append_backend_spec() {
  local raw="$1"
  local backend_part recompute_part liger_part backend recompute_token recompute_mode liger_loss
  local -a fields recompute_tokens

  IFS='|' read -r -a fields <<< "${raw}"
  ((${#fields[@]} == 2 || ${#fields[@]} == 3)) ||
    die "backend spec must be backend|recompute or backend|recompute|ligerloss0/1, got '${raw}'"

  backend_part="${fields[0]}"
  recompute_part="${fields[1]}"
  liger_part="${fields[2]:-ligerloss0}"

  backend="$(backend_label "${backend_part}")"
  liger_loss="$(liger_loss_label "${liger_part}")"

  mapfile -t recompute_tokens < <(tokens "${recompute_part}")
  ((${#recompute_tokens[@]} > 0)) || die "empty recompute mode in backend spec '${raw}'"
  for recompute_token in "${recompute_tokens[@]}"; do
    if [[ "${recompute_token,,}" == "both" ]]; then
      backend_specs_raw+=("${backend}|norecomp|${liger_loss}" "${backend}|recomp|${liger_loss}")
      continue
    fi
    recompute_mode="$(recompute_label "${recompute_token}")"
    backend_specs_raw+=("${backend}|${recompute_mode}|${liger_loss}")
  done
}

mapfile -t backends < <(printf '%s\n' "${backend_specs[@]}" | cut -d '|' -f1 | dedupe)
mapfile -t recompute_modes < <(printf '%s\n' "${backend_specs[@]}" | cut -d '|' -f2 | dedupe)
mapfile -t liger_loss_modes < <(printf '%s\n' "${backend_specs[@]}" | cut -d '|' -f3 | dedupe)

append_liger_loss_filters() {
  local -n _cmd_ref="$1"
  local liger_loss
  for liger_loss in "${liger_loss_modes[@]}"; do
    _cmd_ref+=(--liger-loss "${liger_loss}")
  done
}

job_root_path() {
  local config_root="$1"
  local backend="$2"
  local profiler="$3"
  local recompute="$4"
  local expert_policy="$5"
  local router_mode="$6"
  local grad_offload="${7:-false}"
  local weight_offload="${8:-false}"
  local liger_loss="${9:-ligerloss0}"
  local grad_offload_suffix=""
  if cpuadam_backend_for_label "${backend}" >/dev/null; then
    grad_offload_suffix="__gradoff${grad_offload}__weightoff${weight_offload}"
  fi
  printf '%s/%s\n' "${config_root}" "$(safe_label \
    "${backend}__${profiler}__${recompute}__pol${expert_policy}__router${router_mode}__${expact_label}__${attnact_label}__${layeract_label}__${expact_lora_a_fwd_label}__${actrecomp_label}__${xunpack_label}__${liger_loss}${grad_offload_suffix}")"
}

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
  local weight_offload="${12:-false}"
  local liger_loss="${13:-ligerloss0}"
  local job_enable_liger_kernel=false

  [[ "${liger_loss}" == "ligerloss1" ]] && job_enable_liger_kernel=true

  job_root="$(job_root_path "${config_root}" "${backend}" "${run_profiler}" "${recompute}" "${expert_policy}" "${router_mode}" "${grad_offload}" "${weight_offload}" "${liger_loss}")"
  source_materialized_job_root="$(job_root_path "${config_root}" "${backend}" source "${recompute}" "${expert_policy}" "${router_mode}" "${grad_offload}" "${weight_offload}" "${liger_loss}")"
  run_id="lf_${backend}_${run_profiler}_${recompute}_pol${expert_policy}_router${router_mode}_${expact_label}_${attnact_label}_${layeract_label}_${expact_lora_a_fwd_label}_${actrecomp_label}_${xunpack_label}_${liger_loss}${grad_offload_run_label}_b${PER_DEVICE_TRAIN_BATCH_SIZE}_s${seq_len}_ga${GRADIENT_ACCUMULATION_STEPS}_${lora_dropout_label_value}"

  run_env+=(
    ENABLE_LIGER_KERNEL="${job_enable_liger_kernel}"
    LIGER_LOSS_ONLY="${job_enable_liger_kernel}"
    ASYM_GEMM_LF_CONFIG_LIGER_LOSS="${liger_loss}"
  )
}

for backend_spec in "${backend_specs[@]}"; do
  IFS='|' read -r backend recompute liger_loss <<< "${backend_spec}"
  [[ -n "${backend}" && -n "${recompute}" && -n "${liger_loss}" ]] || die "internal backend spec is malformed: ${backend_spec}"
  ...
  run_job "${backend}" "${profiler}" "${recompute}" "${seq_len}" "${gpu}" "${gpu_count}" \
    "${expert_policy}" "${job_router_mode}" "${current_dataset}" "${lf_expert_lora_impl}" \
    "${grad_offload}" "${weight_offload}" "${liger_loss}"
done
```

`jobs.tsv` and profile-completeness changes:

```bash
ensure_jobs_tsv() {
  printf 'status\tgpu\tseq_len\tbatch_size\tgradient_accumulation_steps\trecompute\texpert_policy\trouter_mode\tbackend\tprofiler\tliger_loss\tgrad_offload\tjob_dir\tprofile_json\tlog\tqwen_expert_lora_impl\texpert_lora_a_fwd\n'
}

append_job_record ... "${backend}" "${run_profiler}" "${liger_loss}" "${grad_offload}" ...

job_profile_complete ... "${expected_grad_offload}" "${expected_liger_loss}" "${profile_memory_breakdown}"
existing_profile_complete ... "${expected_grad_offload}" "${expected_liger_loss}"
```

Inside the Python check in `existing_profile_complete()`, require:

```python
expected_liger_loss = sys.argv[24] if len(sys.argv) > 24 else "ligerloss0"
actual_liger_loss = str(config.get("liger_loss") or "")
if actual_liger_loss != expected_liger_loss:
    raise SystemExit(
        f"profile liger_loss mismatch: expected {expected_liger_loss}, got {actual_liger_loss or '<missing>'}"
    )
```

`run_lf_lora_sft.sh` pseudocode:

```bash
ENABLE_LIGER_KERNEL=${ENABLE_LIGER_KERNEL:-false}
LIGER_LOSS_ONLY=${LIGER_LOSS_ONLY:-false}

ENABLE_LIGER_KERNEL="$(bool_string ENABLE_LIGER_KERNEL "${ENABLE_LIGER_KERNEL}")"
LIGER_LOSS_ONLY="$(bool_string LIGER_LOSS_ONLY "${LIGER_LOSS_ONLY}")"
if [[ "${ENABLE_LIGER_KERNEL}" == "true" && "${LIGER_LOSS_ONLY}" != "true" ]]; then
  echo "ENABLE_LIGER_KERNEL=true requires LIGER_LOSS_ONLY=true in this profiling script" >&2
  exit 2
fi
if [[ "${ENABLE_LIGER_KERNEL}" == "true" ]]; then
  LIGER_LOSS_TAG=ligerloss1
else
  LIGER_LOSS_TAG=ligerloss0
fi

DEFAULT_RUN_ID="..._${LIGER_LOSS_TAG}_..."

CMD_ARGS+=(
  --enable_liger_kernel "${ENABLE_LIGER_KERNEL}"
  --liger_loss_only "${LIGER_LOSS_ONLY}"
)

log_kv ENABLE_LIGER_KERNEL "${ENABLE_LIGER_KERNEL}"
log_kv LIGER_LOSS_ONLY "${LIGER_LOSS_ONLY}"
log_kv LIGER_LOSS "${LIGER_LOSS_TAG}"

RUN_ENV+=(
  ASYM_GEMM_LF_CONFIG_LIGER_LOSS="${LIGER_LOSS_TAG}"
)
```

`run_lf_profiled_train.py` pseudocode:

```python
liger_loss = os.environ.get("ASYM_GEMM_LF_CONFIG_LIGER_LOSS", "ligerloss0")
if liger_loss not in {"ligerloss0", "ligerloss1"}:
    raise ValueError(f"invalid ASYM_GEMM_LF_CONFIG_LIGER_LOSS: {liger_loss!r}")

config = {
    ...
    "liger_loss": liger_loss,
    ...
}
```

Plotting pseudocode for all three plotting scripts:

```python
parser.add_argument("--liger-loss", action="append", default=[], choices=["ligerloss0", "ligerloss1"])

def parse_liger_loss_part(part: str) -> str | None:
    value = part.strip().lower()
    if value in {"ligerloss0", "ligerloss1"}:
        return value
    return None

def parse_job_dir_parts(job_dir_name):
    ...
    liger_loss = "ligerloss0"
    for part in tail:
        parsed_liger_loss = parse_liger_loss_part(part)
        if parsed_liger_loss is not None:
            liger_loss = parsed_liger_loss
            continue
        ...
    return {
        ...
        "liger_loss": liger_loss,
    }

def matches_filters(record_or_meta, args):
    if args.liger_loss and metadata.get("liger_loss", "ligerloss0") not in set(args.liger_loss):
        return False
```

Add `liger_loss` to:
- `plot_activation_recompute_sweep.py`
  - `row_from_result_dir()`
  - `passes_filters()`
  - `trainable_surface_comparison_key()`
  - `group_key()`
  - `threshold_group_key()`
  - `comparison_key_fields()`
  - `threshold_comparison_key_fields()`
  - `combined_label()`
  - `combined_threshold_label()`
  - group output directory names and group plot title suffixes
- `plot_lf_memory_breakdown.py`
  - `SUMMARY_FIELDS`, `DETAIL_FIELDS`, and any index field list
  - `_metadata_label()`
  - `_infer_metadata()`
  - `_matches_filters()`
  - `_group_label()`
- `plot_lf_interconnect_ctc.py`
  - `SUMMARY_FIELDS`, `STEP_FIELDS`, `INDEX_FIELDS`
  - `RunRecord.label`
  - `_infer_metadata()`
  - `_matches_filters()`
  - `_sort_key()`
  - `_group_label()`

Validation before Stage 3:

```bash
PYTHONPATH=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python \
  -m pytest /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/tests/lf/test_asym_cpu_adamw_args.py -q -k 'liger'

PYTHONPATH=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python \
  -m pytest /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/tests/lf/test_liger_loss_plot_axes.py -q

OUTPUT_ROOT=/tmp/asym_liger_axis_dryrun \
RUN_NAME=qwen3_liger_axis_dryrun \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0,asym_cpuadamwds|norecomp|ligerloss1,zero3_offload|recomp|ligerloss0,zero3_offload|recomp|ligerloss1' \
PROFILERS=both \
GPU_POOL=3 \
WARMUP_STEPS=1 \
MAX_STEPS=1 \
DRY_RUN=true \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Required evidence:
- Dry-run produces four Nsight job roots and four materialized source job roots.
- Every root contains either `__ligerloss0` or `__ligerloss1`; no new job root lacks the Liger axis.
- `ligerloss1` command files contain `ENABLE_LIGER_KERNEL=true`, `LIGER_LOSS_ONLY=true`, and `ASYM_GEMM_LF_CONFIG_LIGER_LOSS=ligerloss1`.
- `ligerloss0` command files contain `ENABLE_LIGER_KERNEL=false`, `LIGER_LOSS_ONLY=false`, and `ASYM_GEMM_LF_CONFIG_LIGER_LOSS=ligerloss0`.
- `jobs.tsv` has exactly one Liger axis column named `liger_loss`.
- Plot scripts accept `--liger-loss ligerloss0 --liger-loss ligerloss1`.
- Plot metadata/CSV rows include `liger_loss`, and combined plots do not collapse `ligerloss0` and `ligerloss1` into the same series.
- Negative dry-run with `BACKEND_SPECS='asym_cpuadamwds|norecomp|true'` fails because only `ligerloss0/1` are valid.

Risks to watch:
- Existing profiles without `liger_loss` are legacy. Do not reuse them for final acceptance.
- Adding `ligerloss0/1` only to `_known_optional_job_axis()` would hide the token but not make it available for filtering/grouping. Parse it as real metadata.
- `PROFILERS=both` creates two sibling artifact trees from one run. Both siblings must carry the same `liger_loss` axis.

## Stage 3: Acceptance Comparison Tool

Files/functions/classes:
- Add `scripts/lf/compare_liger_loss_profiles.py`
  - `ProfileMetrics`
  - `find_profile_artifacts(root, backend, liger_loss)`
  - `load_memory_metrics(root)`
  - `load_timing_metrics(root)`
  - `compare_metrics(baseline, candidate, thresholds)`
  - `main()`
- Add `tests/lf/test_compare_liger_loss_profiles.py`
  - `test_compare_same_profile_fails_memory_threshold`
  - `test_compare_rejects_latency_regression`
  - `test_compare_accepts_meaningful_memory_drop_without_latency_regression`
  - `test_parser_selects_liger_loss_axis`
  - `test_parser_reports_missing_required_metrics`

Implementation steps:
- Consume only e2e profile artifacts produced by `profile_lora_lf.sh`.
- Require explicit `--backend`, `--baseline-liger-loss`, and `--candidate-liger-loss`.
- The two Liger-loss CLI values accept only `ligerloss0` or `ligerloss1`.
- Select artifacts by profile metadata first and path label second; fail if metadata and path disagree.
- Parse source-memory and timing summaries from existing JSON/CSV artifacts, recursively if needed.
- Print the exact files used for each metric.
- Exit nonzero on trivial memory drop, same memory, missing metrics, metadata mismatch, or latency regression.
- Do not hard-code that baseline must be `ligerloss0` inside `compare_metrics()`. That would make baseline-vs-itself validation impossible. The acceptance command supplies `ligerloss0` vs `ligerloss1`.

Pseudocode:

```python
@dataclass
class ProfileMetrics:
    peak_allocated_gib: float
    peak_reserved_gib: float
    lm_head_loss_gib: float
    forward_median_ms: float
    backward_median_ms: float
    step_median_ms: float
    backend: str
    liger_loss: str
    parsed_files: dict[str, str]


def compare_metrics(base, cand, thresholds):
    failures = []
    if base.backend != cand.backend:
        failures.append("backend mismatch")
    if base.liger_loss == cand.liger_loss:
        failures.append("baseline and candidate use the same liger_loss axis")

    peak_drop = base.peak_allocated_gib - cand.peak_allocated_gib
    loss_drop = base.lm_head_loss_gib - cand.lm_head_loss_gib
    fwd_ratio = cand.forward_median_ms / base.forward_median_ms
    bwd_ratio = cand.backward_median_ms / base.backward_median_ms
    step_ratio = cand.step_median_ms / base.step_median_ms

    if peak_drop < thresholds.min_peak_drop_gib:
        failures.append("peak memory drop below threshold")
    if loss_drop < thresholds.min_lm_head_loss_drop_gib:
        failures.append("lm_head/loss memory drop below threshold")
    if step_ratio > thresholds.max_step_ratio:
        failures.append("step latency regression")
    if fwd_ratio > thresholds.max_forward_ratio:
        failures.append("forward latency regression")
    if bwd_ratio > thresholds.max_backward_ratio:
        failures.append("backward latency regression")
    return failures
```

Validation before Stage 4:

```bash
PYTHONPATH=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python \
  -m pytest /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/tests/lf/test_compare_liger_loss_profiles.py -q

BASE_JOB="$(
  find /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/profiling_liger_acceptance \
    -type d -name '*asym_cpuadamwds*ligerloss0*' -print -quit
)"
test -n "${BASE_JOB}"

/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python \
  /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/compare_liger_loss_profiles.py \
  --baseline "${BASE_JOB}" \
  --candidate "${BASE_JOB}" \
  --backend asym_cpuadamwds \
  --baseline-liger-loss ligerloss0 \
  --candidate-liger-loss ligerloss0
```

Required evidence:
- Unit tests prove pass/fail behavior for meaningful memory drop, latency regression, missing metrics, and `liger_loss` selection.
- Baseline-vs-itself comparison exits nonzero and reports same-axis/zero-drop failures.
- Tool output names the artifact files used for memory and timing.

Risks to watch:
- Current profile artifact schemas may not expose every metric in one stable file. Keep the parser tolerant, but fail loudly with missing metric names and searched paths.

## Stage 4: Full E2E Acceptance

Files/functions/classes:
- Exercise:
  - `third_party/LlamaFactory/src/llamafactory/model/model_utils/liger_kernel.py`
  - `third_party/LlamaFactory/src/llamafactory/hparams/model_args.py`
  - `scripts/lf/run_lf_lora_sft.sh`
  - `scripts/lf/profile_lora_lf.sh`
  - `scripts/lf/run_lf_profiled_train.py`
  - plotting scripts under `scripts/plotting/`
  - `scripts/lf/compare_liger_loss_profiles.py`

Implementation steps:
- Run no-Liger and Liger-loss variants for each backend being judged.
- Use `PROFILERS=both` so the one training run produces Nsight timing plus materialized source memory artifacts.
- Confirm the loss optimization did not introduce inefficient kernels:
  - Do not split experts into small GEMMs.
  - Do not loop over experts in Python.
  - Do not enable Liger experts/SwiGLU.
  - Expert compute must remain AsymGEMM-owned for AsymGEMM jobs.

E2E command for four-way Asym/ZeRO comparison:

```bash
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/profiling_liger_acceptance \
RUN_NAME=qwen3_ligerloss_fourway \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0,asym_cpuadamwds|norecomp|ligerloss1,zero3_offload|recomp|ligerloss0,zero3_offload|recomp|ligerloss1' \
PROFILERS=both \
GPU_POOL=3 \
WARMUP_STEPS=5 \
MAX_STEPS=5 \
OVERWRITE=true \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Comparison commands:

```bash
ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/profiling_liger_acceptance
ASYM_BASE="$(find "${ROOT}" -type d -name '*asym_cpuadamwds*ligerloss0*' -print -quit)"
ASYM_CAND="$(find "${ROOT}" -type d -name '*asym_cpuadamwds*ligerloss1*' -print -quit)"
ZERO_BASE="$(find "${ROOT}" -type d -name '*zero3_offload*ligerloss0*' -print -quit)"
ZERO_CAND="$(find "${ROOT}" -type d -name '*zero3_offload*ligerloss1*' -print -quit)"
test -n "${ASYM_BASE}" && test -n "${ASYM_CAND}"
test -n "${ZERO_BASE}" && test -n "${ZERO_CAND}"

/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python \
  /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/compare_liger_loss_profiles.py \
  --baseline "${ASYM_BASE}" \
  --candidate "${ASYM_CAND}" \
  --backend asym_cpuadamwds \
  --baseline-liger-loss ligerloss0 \
  --candidate-liger-loss ligerloss1 \
  --min-peak-drop-gib 10 \
  --min-lm-head-loss-drop-gib 20 \
  --max-step-ratio 1.10 \
  --max-forward-ratio 1.15 \
  --max-backward-ratio 1.15

/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python \
  /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/compare_liger_loss_profiles.py \
  --baseline "${ZERO_BASE}" \
  --candidate "${ZERO_CAND}" \
  --backend zero3_offload \
  --baseline-liger-loss ligerloss0 \
  --candidate-liger-loss ligerloss1 \
  --min-peak-drop-gib 10 \
  --min-lm-head-loss-drop-gib 20 \
  --max-step-ratio 1.10 \
  --max-forward-ratio 1.15 \
  --max-backward-ratio 1.15
```

Required evidence:
- Every per-job folder includes `ligerloss0` or `ligerloss1`.
- `jobs.tsv`, profile metadata, timing CSVs, memory CSVs, and plot metadata include `liger_loss`.
- LF log for `ligerloss1` contains `Liger loss-only kernel has been applied`.
- LF log for `ligerloss0` does not apply Liger.
- AsymGEMM `ligerloss1` log still confirms Qwen3-MoE blocks are wrapped as `AsymQwen3MoeBlock`.
- Candidate artifacts include source memory attribution, source plots, Nsight timeline, timing plots, `profile.json`, and `summary.md`.
- Source attribution shows `lm_head`/`loss` memory reduction.
- Nsight timeline does not show a large number of new tiny GEMMs or expert-loop launches.
- Comparison tool passes all memory and latency thresholds for the backend being accepted.

Decision:
- Accept Liger loss for a backend only if that backend's comparison passes and visual artifact inspection agrees.
- Reject for a backend if memory reduction is below threshold, latency exceeds threshold, or the win comes from an inefficient launch pattern.

Risks to watch:
- If another earlier module dominates peak allocation, peak memory may not drop as much as `lm_head`/`loss` attribution. Do not relax thresholds unless total peak reserved does not increase and the source-attributed loss reduction is clearly meaningful.
- Liger fused CE may trade memory for loss-kernel time. Keep it only if the e2e timing guardrails pass.

## Stage 5: Future Compatible Models

Files/functions/classes:
- Modify only `_LIGER_APPLY_SPECS` in `third_party/LlamaFactory/src/llamafactory/model/model_utils/liger_kernel.py`.
- Add a model-specific compatibility test next to `tests/lf/test_liger_loss_only_qwen3_moe.py`.

Implementation steps:
- Add another model only after proving:
  - Liger exposes fused linear CE for that model.
  - Loss-only kwargs disable all other boolean patches.
  - Original model modules remain recognizable to the matching AsymGEMM wrapper when used with AsymGEMM.
  - Tiny CUDA forward/backward produces finite loss, no logits, and LoRA grads.
  - Full e2e profiling passes Stage 4 memory and latency thresholds.

Validation:
- Run the new model-specific pytest.
- Run the same e2e baseline/candidate/compare sequence from Stages 0 and 4.

Risks to watch:
- Do not add Llama4, Qwen3 dense, Qwen3.5 MoE, or any other model by name similarity. Add only after the compatibility and e2e gates pass.
