# Staged Implementation Plan: Liger Loss-Only Axis

Goal: add a `liger_loss` profiling axis with values `ligerloss0` and `ligerloss1`. `ligerloss1` means Liger fused linear cross entropy only. It must not enable Liger RoPE, RMSNorm, SwiGLU, expert, or other non-loss patches.

Repository layout used by this plan:
- AsymGEMM root: `/workspace/AsymGEMM-SFT/third_party/AsymGEMM`
- LlamaFactory sibling: `../LlamaFactory`
- Liger-Kernel sibling: `../Liger-Kernel`

Hard design decisions:
- Do not edit `../Liger-Kernel` source. The local Liger package is an external dependency. We may import its public functions/classes, but the compatibility bridge lives in AsymGEMM/LlamaFactory integration code.
- Use the vendored Liger checkout, not PyPI. Install it editable into the AsymGEMM `.venv` with `--no-deps --no-build-isolation` so pip does not change the existing Torch/Triton/CUDA stack.
- Do not modify `scripts/lf/profile3.sh` for this Liger work. That script is for the separate Qwen3.5 testing path. The Liger loss-only plan uses `scripts/lf/profile_lora_lf.sh` and shared LF wrapper/profile/plotting interfaces.
- CPU offload profiling must bind host allocations only to real CPU DRAM NUMA
  nodes. On the current GB200 system, the only accepted profiling placement is
  `NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 NUMACTL_MODE=membind`.
  This means CPU execution is bound to nodes `0,1` and host allocations are
  bound to CPU RAM nodes `0,1` only. Do not use `NUMACTL_MODE=interleave`, do
  not widen `NUMACTL_MEMBIND` beyond `0,1`, and do not include any GPU/HBM NUMA
  node. If this placement causes host-memory pressure, report that result
  instead of changing NUMA placement. Discard any artifact produced with
  different placement.
- Keep `lm_head` Asym-wrapped when `ASYM_OFFLOAD_MODULES` selects `lm_head`. The Asym + Liger path must stage the Asym CPU-resident `lm_head` weight explicitly for the fused CE call.
- Do not make `AsymFrozenLinear.weight` secretly stage to GPU. Direct `.weight` stays CPU host storage. Staging must use an explicit method so other code paths are not surprised.
- Common script/runtime interfaces are implemented first. Model-specific and Asym-specific behavior is wired only after folder names, metadata, CLI flags, and validation plumbing are stable.
- Use one user-facing runtime env only: `ENABLE_LIGER_KERNEL`. Do not add `LIGER_LOSS_ONLY` or aliases. Stage 0 maps the profiling axis to this env; Stage 1 makes LlamaFactory resolve it as loss-only for validated model types.

Canonical interface:
- Sweep axis: `BACKEND_SPECS=backend|recompute|ligerloss0_or_ligerloss1`.
- Accepted third-field values are exactly `ligerloss0` and `ligerloss1`.
- Missing third field is backward-compatible input and means `ligerloss0`, but every new output folder must still include `__ligerloss0`.
- Artifact metadata key is exactly `liger_loss`.
- Plot CLI flag is exactly `--liger-loss`.
- Runtime env input is exactly `ENABLE_LIGER_KERNEL`.
- Derived metadata env is `ASYM_GEMM_LF_CONFIG_LIGER_LOSS`; scripts set it from the selected axis. Users do not set it directly.
- `run_lf_lora_sft.sh` derives LF loss-only behavior from `ENABLE_LIGER_KERNEL`:
  - `ligerloss0` -> `ENABLE_LIGER_KERNEL=false`, `--enable_liger_kernel false`
  - `ligerloss1` -> `ENABLE_LIGER_KERNEL=true`, `--enable_liger_kernel true`

Acceptance rule:
- Toy/unit tests only prove plumbing and compatibility. They do not justify keeping the feature.
- Keep `ligerloss1` for a backend only if full e2e `PROFILERS=both` LoRA profiling shows meaningful memory reduction without timing blowup.
- Default acceptance thresholds:
  - peak CUDA allocated drops by at least 10 GiB
  - source-attributed `lm_head`/`loss` HBM drops by at least 20 GiB
  - median measured step latency <= 1.10x `ligerloss0`
  - median forward latency <= 1.15x `ligerloss0`
  - median backward latency <= 1.15x `ligerloss0`
- Reject same-memory/slower results, trivial-memory/slower results, and any result that adds many tiny GEMMs or per-expert launch loops.

Resolved facts:
- The common profiling script must normalize every backend spec to three fields before any real `ligerloss1` run.
- Plot parsers must parse `ligerloss0/1` as real metadata, not hide it as an optional/ignored token.
- `../Liger-Kernel/src/liger_kernel/transformers/monkey_patch.py::apply_liger_kernel_to_qwen3_moe()` exposes `rope`, `cross_entropy`, `fused_linear_cross_entropy`, `rms_norm`, `swiglu`, and `model`; default Liger Qwen3-MoE patching enables non-loss pieces unless we force loss-only kwargs.
- Local Liger also exposes `apply_liger_kernel_to_qwen3_5()` and `apply_liger_kernel_to_qwen3_5_moe()`, both with `fused_linear_cross_entropy`, but those model types remain out of scope for this Qwen3-MoE loss-only implementation. Do not enable them from this plan.
- Current Qwen3.5/common-profiling changes add `linear_attention` classification/default filters and strict-mode metadata. Treat those as existing profiling-surface requirements; do not mix them with Liger runtime changes.
- Liger Qwen3-MoE loss forward passes `lm_head_weight=self.lm_head.weight`. That works with normal HF/DeepSpeed-managed parameters, but not with Asym offloaded `lm_head` because `AsymFrozenLinear.weight` is CPU host storage.
- AsymGEMM and DeepSpeed ZeRO3 are mutually exclusive in LlamaFactory. `asym_cpuadamwds` only uses DeepSpeed CPUAdamW optimizer pieces, not ZeRO3 parameter offload.
- LlamaFactory `lora_target=all` excludes `lm_head`, and AsymGEMM already rejects `additional_target`. First implementation can require frozen `lm_head`.
- Kernel smoke test completed locally with `PYTHONPATH=/workspace/AsymGEMM-SFT/third_party/Liger-Kernel/src`: CUDA was visible, `LigerForCausalLMLoss` ran on GPU 3 in BF16, matched torch CE within about `0.0014`, and produced finite hidden/weight gradients.

Environment setup:
- The current AsymGEMM `.venv` must be able to import the local vendored Liger package before any `ligerloss1` run.
- Current local state before installation: `.venv/bin/python -c "import liger_kernel"` fails with `ModuleNotFoundError`. `PYTHONPATH=../Liger-Kernel/src` is fine for API/kernel smoke tests, but e2e profiling should use the editable install below.
- Recommended install command:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
.venv/bin/python -m pip install \
  --no-deps \
  --no-build-isolation \
  -e /workspace/AsymGEMM-SFT/third_party/Liger-Kernel
```

- Do not use `pip install liger-kernel` for this workflow unless intentionally switching away from the vendored source.
- Do not install with dependencies enabled; the existing profiling venv already owns the Torch/Triton/CUDA versions.
- Verify the install with:

```bash
.venv/bin/python - <<'PY'
import liger_kernel
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe
print(liger_kernel.__file__)
print(apply_liger_kernel_to_qwen3_moe)
PY
```

## Stage 0: Common Interfaces, Scripts, Artifact Axis, And `ligerloss0` Migration

Purpose: update all shared user-facing and artifact-facing interfaces before changing model math. This stage owns backend-spec parsing, runtime env mapping, LF script command construction, profile metadata, postprocessing, plotting, and migration. Every no-Liger run becomes explicit as `ligerloss0`.

Scope:
- Modify `scripts/lf/profile_lora_lf.sh`
  - usage text and examples
  - `append_backend_spec()`
  - backend-spec normalization arrays
  - `job_root_path()`
  - KT source-profile matching helpers
  - `ensure_jobs_tsv()`
  - `append_job_record()`
  - `job_profile_complete()`
  - `existing_profile_complete()`
  - plot-filter helpers
  - `run_job()`
  - the main nested loop over backend specs
- Modify `scripts/lf/run_lf_lora_sft.sh`
  - env default for `ENABLE_LIGER_KERNEL`
  - boolean normalization
  - `DEFAULT_RUN_ID`
  - `CMD_ARGS`
  - log block
  - `RUN_ENV`
- Modify `scripts/lf/run_lf_profiled_train.py`
  - `_config_from_args()`
- Modify plotting scripts:
  - `scripts/plotting/plot_activation_recompute_sweep.py`
  - `scripts/plotting/plot_lf_memory_breakdown.py`
  - `scripts/plotting/plot_lf_interconnect_ctc.py`
- Add `scripts/lf/migrate_liger_loss_axis.py`
- Extend tests:
  - `tests/lf/test_superoffload_backend_scripts.py`
  - `tests/lf/test_asym_cpu_adamw_args.py`

Actual code changes:
- Add `liger_loss_label()` in `profile_lora_lf.sh` next to existing label helpers.
- Change `append_backend_spec()` to accept exactly two or three fields. The third field must validate as `ligerloss0` or `ligerloss1`; missing means `ligerloss0`.
- Normalize every backend spec internally to exactly `backend|recompute|ligerloss`.
- Preserve `recompute=both` expansion:
  - `asym_cpuadamwds|both` -> `asym_cpuadamwds|norecomp|ligerloss0`, `asym_cpuadamwds|recomp|ligerloss0`
  - `asym_cpuadamwds|both|ligerloss1` -> `asym_cpuadamwds|norecomp|ligerloss1`, `asym_cpuadamwds|recomp|ligerloss1`
- Replace all two-field splits with explicit three-field parsing.
- Change `job_root_path()` signature to:
  - `job_root_path config_root backend profiler recompute expert_policy router_mode liger_loss grad_offload weight_offload`
- Insert the Liger token before CPUAdam suffixes:
  - no CPUAdam suffix: `...__xunpack0__ligerloss0`
  - CPUAdam suffix: `...__xunpack0__ligerloss0__gradofftrue__weightofffalse`
- Add `_ligerloss0/1` to `run_id`.
- Add exactly one `jobs.tsv` column: `liger_loss`, immediately after `profiler`.
- In `profile_lora_lf.sh::run_job()`:
  - validate `liger_loss`
  - map `ligerloss0` to `ENABLE_LIGER_KERNEL=false`
  - map `ligerloss1` to `ENABLE_LIGER_KERNEL=true`
  - pass the same value to both `nsys` and materialized `source` runs
  - include `ENABLE_LIGER_KERNEL` and `ASYM_GEMM_LF_CONFIG_LIGER_LOSS` in dry-run command/status artifacts
- In `run_lf_lora_sft.sh`, add and normalize:
  - `ENABLE_LIGER_KERNEL=${ENABLE_LIGER_KERNEL:-false}`
  - derive `LIGER_LOSS_TAG=ligerloss1` when true, else `ligerloss0`
  - append `${LIGER_LOSS_TAG}` to KT and non-KT `DEFAULT_RUN_ID`
  - add `--enable_liger_kernel "${ENABLE_LIGER_KERNEL}"` to `CMD_ARGS`
  - add `ENABLE_LIGER_KERNEL` and `ASYM_GEMM_LF_CONFIG_LIGER_LOSS="${LIGER_LOSS_TAG}"` to `RUN_ENV`
  - log `ENABLE_LIGER_KERNEL` and `LIGER_LOSS_TAG`
- Completion checks must reject mismatched metadata. `existing_profile_complete()` and `job_profile_complete()` need expected `liger_loss` arguments and must compare against `profile.config.liger_loss` or nested `source_profile.config.liger_loss`.
- Plot scripts must parse `ligerloss0/1` as metadata, add `--liger-loss`, filter by it, and include it in CSV/index/JSON rows, labels, grouping keys, and group output directory names.
- `run_lf_profiled_train.py::_config_from_args()` must read `ASYM_GEMM_LF_CONFIG_LIGER_LOSS`, default missing to `ligerloss0`, reject other values, and write `config["liger_loss"]`.
- The migration script must rename legacy job dirs to include `__ligerloss0`, update `jobs.tsv`, update profile/source/memory JSON metadata, refuse overwrites, and write `liger_loss_migration.json`.

Pseudocode:

```bash
liger_loss_label() {
  case "${1}" in
    ligerloss0|ligerloss1) printf '%s\n' "${1}" ;;
    *) die "liger loss must be exactly ligerloss0 or ligerloss1, got '${1}'" ;;
  esac
}

append_backend_spec() {
  local raw="$1" backend recompute_token recompute_mode liger_loss
  local -a fields recompute_tokens
  IFS='|' read -r -a fields <<< "${raw}"
  ((${#fields[@]} == 2 || ${#fields[@]} == 3)) ||
    die "backend spec must be backend|recompute or backend|recompute|ligerloss0/1"

  backend="$(backend_label "${fields[0]}")"
  liger_loss="$(liger_loss_label "${fields[2]:-ligerloss0}")"
  mapfile -t recompute_tokens < <(tokens "${fields[1]}")

  for recompute_token in "${recompute_tokens[@]}"; do
    if [[ "${recompute_token,,}" == "both" ]]; then
      backend_specs_raw+=("${backend}|norecomp|${liger_loss}" "${backend}|recomp|${liger_loss}")
    else
      recompute_mode="$(recompute_label "${recompute_token}")"
      backend_specs_raw+=("${backend}|${recompute_mode}|${liger_loss}")
    fi
  done
}

for backend_spec in "${backend_specs[@]}"; do
  IFS='|' read -r backend recompute liger_loss extra <<< "${backend_spec}"
  [[ -n "${backend}" && -n "${recompute}" && -n "${liger_loss}" && -z "${extra}" ]] ||
    die "internal backend spec is malformed: ${backend_spec}"
  run_job "${backend}" "${profiler}" "${recompute}" "${liger_loss}" ...
done

case "${ENABLE_LIGER_KERNEL,,}" in
  1|true|yes|y|on) ENABLE_LIGER_KERNEL=true ;;
  0|false|no|n|off) ENABLE_LIGER_KERNEL=false ;;
  *) echo "ENABLE_LIGER_KERNEL must be true or false, got '${ENABLE_LIGER_KERNEL}'" >&2; exit 2 ;;
esac

if [[ "${ENABLE_LIGER_KERNEL}" == "true" ]]; then
  LIGER_LOSS_TAG=ligerloss1
else
  LIGER_LOSS_TAG=ligerloss0
fi

CMD_ARGS+=(--enable_liger_kernel "${ENABLE_LIGER_KERNEL}")
RUN_ENV+=(ASYM_GEMM_LF_CONFIG_LIGER_LOSS="${LIGER_LOSS_TAG}")
```

Validation before Stage 1:

```bash
.venv/bin/python -m pytest \
  tests/lf/test_asym_cpu_adamw_args.py \
  -q

.venv/bin/python -m pytest \
  tests/lf/test_superoffload_backend_scripts.py \
  -q -k 'profile_lora_lf or uses_deepspeed_for_single_gpu_zero3_offload or uses_deepspeed_for_single_gpu_superoffload'
```

```bash
OUT=/tmp/asym_liger_axis_dryrun
LOG=/tmp/asym_liger_axis_dryrun.log
rm -rf "${OUT}" "${LOG}"
OUTPUT_ROOT="${OUT}" \
RUN_NAME=qwen3_liger_axis_dryrun \
NUMACTL_MEMBIND=0,1 \
NUMACTL_CPUNODEBIND=0,1 \
NUMACTL_MODE=membind \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp,zero3_offload|recomp' \
PROFILERS=both \
GPU_POOL=3 \
WORKLOADS='128|1|1' \
WARMUP_STEPS=0 \
MAX_STEPS=1 \
DRY_RUN=true \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
PREPARE_DATASETS=false \
ASYMM_EXP_ACT_POLICIES='none|false|false|false' \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=false \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false \
bash scripts/lf/profile_lora_lf.sh 2>&1 | tee "${LOG}"

rg '__ligerloss0' "${LOG}"
! rg '__source__.*__ligerloss1|__nsys__.*__ligerloss1' "${LOG}"
```

Four-way common-interface dry run:

```bash
OUT=/tmp/asym_liger_axis_fourway
LOG=/tmp/asym_liger_axis_fourway.log
rm -rf "${OUT}" "${LOG}"
OUTPUT_ROOT="${OUT}" \
RUN_NAME=qwen3_liger_axis_fourway \
NUMACTL_MEMBIND=0,1 \
NUMACTL_CPUNODEBIND=0,1 \
NUMACTL_MODE=membind \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0,asym_cpuadamwds|norecomp|ligerloss1,zero3_offload|recomp|ligerloss0,zero3_offload|recomp|ligerloss1' \
PROFILERS=both \
GPU_POOL=3 \
WORKLOADS='128|1|1' \
WARMUP_STEPS=0 \
MAX_STEPS=1 \
DRY_RUN=true \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
PREPARE_DATASETS=false \
ASYMM_EXP_ACT_POLICIES='none|false|false|false' \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=false \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false \
bash scripts/lf/profile_lora_lf.sh 2>&1 | tee "${LOG}"

rg '__ligerloss0|__ligerloss1' "${LOG}"
rg 'ENABLE_LIGER_KERNEL=true' "${LOG}"
rg 'ENABLE_LIGER_KERNEL=false' "${LOG}"
rg 'ASYM_GEMM_LF_CONFIG_LIGER_LOSS=ligerloss1|ligerloss1' "${LOG}"
rg 'ASYM_GEMM_LF_CONFIG_LIGER_LOSS=ligerloss0|ligerloss0' "${LOG}"
```

Migration validation on a copied root:

```bash
SRC=/workspace/AsymGEMM-SFT/third_party/AsymGEMM/profiling_both
DST=/tmp/asym_liger_axis_migration_check
test -d "${SRC}"
rm -rf "${DST}"
cp -a "${SRC}" "${DST}"

.venv/bin/python scripts/lf/migrate_liger_loss_axis.py --root "${DST}" --dry-run
.venv/bin/python scripts/lf/migrate_liger_loss_axis.py --root "${DST}" --apply

find "${DST}" -type d \( -name '*__source__*' -o -name '*__nsys__*' \) \
  ! -name '*__ligerloss0*' ! -name '*__ligerloss1*' -print -quit | grep -q . && exit 1 || true
rg -n 'liger_loss' "${DST}" --glob 'profile.json' --glob 'source_profile.json' --glob 'jobs.tsv'
```

Required evidence:
- Fresh dry-run commands include `__ligerloss0` even when `BACKEND_SPECS` omits the third field.
- `PROFILERS=both` paths include both `__nsys__...__ligerloss0` and `__source__...__ligerloss0`.
- Four-way dry-run commands include `__ligerloss0` and `__ligerloss1`.
- Profile dry-run command artifacts contain `ENABLE_LIGER_KERNEL=true/false` and `ASYM_GEMM_LF_CONFIG_LIGER_LOSS=ligerloss0/1`.
- Direct `run_lf_lora_sft.sh` fake-LF tests show final LF argv contains `--enable_liger_kernel true/false`.
- `jobs.tsv` has exactly one `liger_loss` column.
- Migrated artifacts have explicit path labels and `config.liger_loss == "ligerloss0"`.
- Plot outputs can be filtered by `--liger-loss ligerloss0`.
- Invalid aliases such as `true`, `false`, `liger`, or `loss1` fail.

Efficiency rationale:
- Stage 0 validates script/runtime plumbing through dry runs and fake-LF tests. It must not require real model execution.
- Any real runtime or memory change from Stage 0 alone is a bug; actual Liger behavior is only enabled after Stage 1 implements the LF resolver.

Risks to watch:
- Do not run real `ligerloss1` training after Stage 0 alone. Stage 0 can set `--enable_liger_kernel true`, but LlamaFactory does not yet constrain that to loss-only until Stage 1.
- Do not migrate a real profiling root in place until copied-root validation passes.
- If a migration destination already exists, fail. Do not merge directories.
- Do not touch `scripts/lf/profile3.sh` for this Liger work. It belongs to the separate Qwen3.5 testing path. Ignore `scripts/lf/profile_lora_lf_test*.sh` and `../archive/*` unless one is explicitly promoted as canonical.

## Stage 1: LlamaFactory Loss-Only Resolver

Purpose: make the Stage 0 common runtime interface executable by LlamaFactory. `ENABLE_LIGER_KERNEL=true` now resolves to loss-only Liger for validated model types. This stage must not contain Asym-specific `lm_head` staging logic.

Scope:
- Modify `../LlamaFactory/src/llamafactory/hparams/model_args.py`
  - class `ModelArguments`
- Modify `../LlamaFactory/src/llamafactory/model/model_utils/liger_kernel.py`
  - `apply_liger_kernel(config, model_args, is_trainable, require_logits)`
  - add `_LOSS_ONLY_SUPPORTED_MODEL_TYPES`
  - add `_resolve_liger_apply_fn(model_type)`
  - add `_build_liger_loss_only_kwargs(apply_fn)`
- Extend tests:
  - `tests/lf/test_liger_loss_only_qwen3_moe.py`

Actual code changes:
- In LlamaFactory `model_utils/liger_kernel.py`:
  - in this AsymGEMM/LF integration, `model_args.enable_liger_kernel=True` means loss-only
  - initially support only `config.model_type == "qwen3_moe"`
  - skip when `require_logits=True`
  - build kwargs by introspecting the Liger apply function and setting only `fused_linear_cross_entropy=True`; every other boolean patch must be `False`
- Do not change `../Liger-Kernel`.

Pseudocode:

```python
_LOSS_ONLY_SUPPORTED_MODEL_TYPES = {"qwen3_moe"}

def _build_liger_loss_only_kwargs(apply_fn):
    signature = inspect.signature(apply_fn)
    if "fused_linear_cross_entropy" not in signature.parameters:
        return None

    kwargs = {}
    for name, param in signature.parameters.items():
        if name == "model":
            continue
        if name == "fused_linear_cross_entropy":
            kwargs[name] = True
        elif isinstance(param.default, bool):
            kwargs[name] = False

    return kwargs if kwargs.get("fused_linear_cross_entropy") is True else None

def apply_liger_kernel(config, model_args, is_trainable, require_logits):
    if not is_trainable or not model_args.enable_liger_kernel:
        return

    model_type = getattr(config, "model_type", None)
    apply_fn = _resolve_liger_apply_fn(model_type)
    if apply_fn is None:
        return

    if require_logits:
        logger.warning_rank0("Skipping Liger loss-only because logits are required.")
        return
    if model_type not in _LOSS_ONLY_SUPPORTED_MODEL_TYPES:
        logger.warning_rank0(f"Skipping Liger loss-only for unvalidated model_type={model_type}.")
        return
    kwargs = _build_liger_loss_only_kwargs(apply_fn)
    if kwargs is None:
        logger.warning_rank0(f"Skipping Liger loss-only for model_type={model_type}; fused linear CE is unavailable.")
        return
    apply_fn(**kwargs)
    logger.info_rank0("Liger loss-only kernel has been applied.")
```

Validation before Stage 2:

```bash
PYTHONPATH=/workspace/AsymGEMM-SFT/third_party/Liger-Kernel/src \
.venv/bin/python - <<'PY'
import inspect
import liger_kernel
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe

print(liger_kernel.__file__)
sig = inspect.signature(apply_liger_kernel_to_qwen3_moe)
expected = {"rope", "cross_entropy", "fused_linear_cross_entropy", "rms_norm", "swiglu", "model"}
missing = expected - set(sig.parameters)
assert not missing, missing
PY
```

```bash
PYTHONPATH=/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src:/workspace/AsymGEMM-SFT/third_party/Liger-Kernel/src:/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
.venv/bin/python -m pytest \
  tests/lf/test_liger_loss_only_qwen3_moe.py \
  -q
```

Required evidence:
- Qwen3-MoE loss-only kwargs are exactly:
  - `fused_linear_cross_entropy=True`
  - `cross_entropy=False`
  - `rope=False`
  - `rms_norm=False`
  - `swiglu=False`
- `enable_liger_kernel=True` selects loss-only for validated Qwen3-MoE.
- ZeRO3-offload and AsymGEMM use the same `ENABLE_LIGER_KERNEL`/`--enable_liger_kernel` knob.
- `require_logits=True` skips fused linear CE.
- Unvalidated model types skip loss-only.
- Stage 0 four-way dry-run still passes after the LF resolver is added.

Efficiency rationale:
- This stage changes only flag routing and loss-only Liger patch selection.
- It must not add Asym staging, change kernel granularity directly, or loop over experts.

Risks to watch:
- Liger monkey-patches global classes. Unit tests that inspect patched/unpatched behavior must use fresh Python subprocesses.
- Future Liger releases can add boolean toggles. `_build_liger_loss_only_kwargs()` must keep disabling every boolean except `fused_linear_cross_entropy`.

## Stage 2: Asym-Wrapped `lm_head` Staging Bridge

Purpose: keep `lm_head` offloaded by AsymGEMM while allowing Liger fused CE to consume a temporary CUDA weight. This is the specific compatibility bridge. It must live in AsymGEMM/LF integration code, not in Liger-Kernel.

Scope:
- Modify `asym_gemm/training/frozen_linear.py`
  - class `AsymFrozenLinear`: add explicit staging method
- Add `asym_gemm/integrations/liger_loss.py`
  - `_resolve_liger_lm_head_weight(lm_head, hidden_states)`
  - `asym_qwen3_moe_lce_forward(...)`
  - `install_asym_liger_qwen3_moe_loss_bridge(model, *, strict=True)`
- Modify `../LlamaFactory/src/llamafactory/model/adapter.py`
  - call the bridge after `adapt_lf_asym_peft_lora(...)`
- Modify `scripts/lf/run_lf_profiled_train.py`
  - `_config_from_args()`
  - add `_asym_liger_lm_head_bridge_from_model()`
  - include bridge metadata in the final profile dict next to `asym_execution_stats`
- Add tests:
  - `tests/lf/test_asym_liger_lm_head_bridge.py`

Actual code changes:
- `AsymFrozenLinear.asym_liger_lm_head_weight(self, *, device, dtype) -> torch.Tensor`
  - require `self.bias_cpu is None`; otherwise raise, because the local Liger causal LM loss call does not pass a bias.
  - require `self.host_weight.weight.requires_grad is False`; otherwise raise.
  - require source weight is 2D and contiguous; if not contiguous, make CPU contiguous before staging and record this in tests.
  - stage with:
    - `non_blocking=True` only when the CPU tensor is pinned
    - `dtype` equal to hidden state dtype
    - `device` equal to hidden state device
  - return a temporary CUDA tensor. Do not cache it on the module.
- `asym_gemm/integrations/liger_loss.py`
  - Import Liger public loss/output helpers:
    - `LigerForCausalLMLoss`
    - `unpack_cross_entropy_result`
    - `LigerMoeCausalLMOutputWithPast`
  - Copy the Qwen3-MoE loss-only forward structure from the installed Liger version, but replace only the weight line with `_resolve_liger_lm_head_weight(...)`.
  - Preserve the non-loss fallback path: if `skip_logits` is false, call `self.lm_head(kept_hidden_states)` normally.
  - Install as an instance-level method with `types.MethodType`, not a global class patch. This prevents leaking Asym-specific behavior into non-Asym runs in the same Python process.
- `../LlamaFactory/src/llamafactory/model/adapter.py`
  - after `model, report = adapt_lf_asym_peft_lora(...)` and before `return model`, conditionally call:
    - `install_asym_liger_qwen3_moe_loss_bridge(model, strict=model_args.asym_strict)`
  - call only when:
    - `model_args.use_asym_gemm`
    - `model_args.enable_liger_kernel`
    - `model.config.model_type == "qwen3_moe"`
  - fail fast if `ASYM_OFFLOAD_MODULES` selected `lm_head` but `model.lm_head` does not expose `asym_liger_lm_head_weight`.
  - fail fast if `model.lm_head` has trainable params, which indicates explicit `lm_head` LoRA or unsupported wrapping.
- `run_lf_profiled_train.py`
  - `_config_from_args()` records `liger_loss`.
  - add `_asym_liger_lm_head_bridge_from_model()` near `_asym_execution_stats_from_model()`.
  - the helper reads the captured model and returns:
    - `enabled`
    - `weight_source` with values like `asym_host_staged`, `normal_parameter`, or `disabled`
    - `staged_bytes` when measurable from `model.lm_head.cpu_resident_base_weight_bytes`
    - `lm_head_type`
  - final profile JSON includes this helper output as `asym_liger_lm_head_bridge`.

Pseudocode:

```python
# asym_gemm/training/frozen_linear.py
class AsymFrozenLinear(nn.Module):
    ...
    def asym_liger_lm_head_weight(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.bias_cpu is not None:
            raise RuntimeError("Asym Liger lm_head bridge currently requires bias-free lm_head.")
        weight = self.host_weight.weight
        if weight.requires_grad:
            raise RuntimeError("Asym Liger lm_head bridge supports frozen lm_head only.")
        if weight.ndim != 2:
            raise RuntimeError(f"expected 2D lm_head weight, got {tuple(weight.shape)}")
        if not weight.is_contiguous():
            weight = weight.contiguous()
        return weight.to(
            device=device,
            dtype=dtype,
            non_blocking=bool(weight.is_pinned()),
        )
```

```python
# asym_gemm/integrations/liger_loss.py
def _resolve_liger_lm_head_weight(lm_head: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    resolver = getattr(lm_head, "asym_liger_lm_head_weight", None)
    if callable(resolver):
        return resolver(device=hidden_states.device, dtype=hidden_states.dtype)

    weight = getattr(lm_head, "weight", None)
    if weight is None:
        raise TypeError(f"lm_head has no weight for Liger fused CE: {type(lm_head).__name__}")
    return weight

def asym_qwen3_moe_lce_forward(self, ..., skip_logits=None, return_dict=None, **kwargs):
    outputs = self.model(...)
    hidden_states = outputs.last_hidden_state
    kept_hidden_states = hidden_states[:, slice_indices, :]
    shift_labels = kwargs.pop("shift_labels", None)

    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        lm_head_weight = _resolve_liger_lm_head_weight(self.lm_head, kept_hidden_states)
        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=lm_head_weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.hidden_size,
            **kwargs,
        )
        loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)
    else:
        logits = self.lm_head(kept_hidden_states)
        ...
    return LigerMoeCausalLMOutputWithPast(...)

def install_asym_liger_qwen3_moe_loss_bridge(model: nn.Module, *, strict: bool = True) -> bool:
    if getattr(model.config, "model_type", None) != "qwen3_moe":
        if strict:
            raise ValueError("Asym Liger bridge only supports qwen3_moe.")
        return False

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Qwen3-MoE model has no lm_head.")

    resolver = getattr(lm_head, "asym_liger_lm_head_weight", None)
    if not callable(resolver):
        if strict:
            raise RuntimeError("lm_head is not AsymFrozenLinear-compatible for Liger staging.")
        return False

    if any(p.requires_grad for p in lm_head.parameters(recurse=True)):
        raise RuntimeError("Asym Liger bridge supports frozen lm_head only.")

    model.forward = MethodType(asym_qwen3_moe_lce_forward, model)
    model._asym_liger_lm_head_bridge_enabled = True
    model._asym_liger_lm_head_weight_source = "asym_host_staged"
    return True
```

Validation before Stage 3:

```bash
PYTHONPATH=/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src:/workspace/AsymGEMM-SFT/third_party/Liger-Kernel/src:/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
.venv/bin/python -m pytest tests/lf/test_asym_liger_lm_head_bridge.py -q
```

Small CUDA bridge validation:

```bash
CUDA_VISIBLE_DEVICES=3 \
PYTHONPATH=/workspace/AsymGEMM-SFT/third_party/Liger-Kernel/src:/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
.venv/bin/python - <<'PY'
import torch
from asym_gemm.training.frozen_linear import AsymFrozenLinear

lin = torch.nn.Linear(128, 256, bias=False, dtype=torch.bfloat16, device="cuda")
wrapped = AsymFrozenLinear.from_gpu_linear(lin, backend="asym", pin_memory=True)
assert wrapped.weight.device.type == "cpu"
staged = wrapped.asym_liger_lm_head_weight(device=torch.device("cuda"), dtype=torch.bfloat16)
assert staged.device.type == "cuda"
assert staged.dtype == torch.bfloat16
assert staged.shape == (256, 128)
assert not hasattr(wrapped, "_cached_liger_lm_head_weight")
print("ok")
PY
```

Required evidence:
- `AsymFrozenLinear.weight` still returns a CPU tensor.
- `asym_liger_lm_head_weight()` stages to CUDA, uses hidden-state dtype, and does not cache.
- Bias or trainable `lm_head` is rejected.
- The installed bridge is instance-local, not a global class patch.
- A tiny Qwen3-MoE-compatible forward/backward subprocess produces finite loss and LoRA grads.
- AsymGEMM logs still show Qwen3-MoE blocks and selected dense modules wrapped, including `lm_head` when selected.

Efficiency rationale:
- The bridge stages exactly one full `vocab x hidden` `lm_head` weight per loss call.
- It must not split LM-head work into Python loops or small GEMMs.
- It must not keep a persistent GPU cache, because that would defeat `lm_head` offload.
- On one GPU, this is conceptually equivalent to DeepSpeed ZeRO3 offload making the parameter GPU-available for Liger, but implemented explicitly.

Risks to watch:
- The local forward copies Liger Qwen3-MoE forward structure. If local Liger updates its Qwen3 forward signature or output class, tests must catch drift.
- If Qwen3-MoE ever has a biased `lm_head`, the bridge must either pass bias into a supported fused CE path or reject.
- Explicit `lm_head` LoRA is unsupported in the first bridge. Supporting it would require a mathematically correct fused CE over `W + B @ A`, not a naive materialization loop.

## Stage 3: Acceptance Comparison Tool

Purpose: make the memory/timing decision reproducible instead of manual.

Scope:
- Add `scripts/lf/compare_liger_loss_profiles.py`
  - `ProfileMetrics`
  - `load_profile_config(run_dir)`
  - `load_memory_metrics(run_dir)`
  - `load_timing_metrics(run_dir)`
  - `compare_metrics(baseline, candidate, thresholds)`
  - `main()`
- Add `tests/lf/test_compare_liger_loss_profiles.py`

Actual code changes:
- Add explicit CLI args:
  - `--baseline`
  - `--candidate`
  - `--backend`
  - `--baseline-liger-loss`
  - `--candidate-liger-loss`
  - `--min-peak-drop-gib`
  - `--min-lm-head-loss-drop-gib`
  - `--max-step-ratio`
  - `--max-forward-ratio`
  - `--max-backward-ratio`
- Validate `config.backend`, `config.liger_loss`, and path labels.
- Read memory metrics from `memory_breakdown_summary.json`:
  - prefer `actual_peak_allocated_hbm_bytes`
  - fallback to `peak_allocated_hbm_bytes`
  - fallback to `allocated_stack_sum_bytes`
  - compute `lm_head_loss_hbm_bytes` by summing HBM rows whose component/module/path contains `lm_head` or `loss`
- Read timing metrics from `step_samples.csv`:
  - median `step_milliseconds`
  - median `forward_milliseconds`
  - median `backward_milliseconds`
  - fallback to profile summary only when step samples are absent, and report fallback in output
- Print every metric source file and field.
- Exit nonzero on same-axis comparison, missing metrics, metadata mismatch, insufficient memory drop, or latency regression.

Pseudocode:

```python
@dataclass(frozen=True)
class ProfileMetrics:
    run_dir: Path
    backend: str
    liger_loss: str
    peak_allocated_hbm_bytes: int
    lm_head_loss_hbm_bytes: int
    median_step_ms: float
    median_forward_ms: float
    median_backward_ms: float
    sources: dict[str, str]

def compare_metrics(base, cand, args):
    if base.liger_loss == cand.liger_loss:
        fail("baseline and candidate use the same liger_loss")
    peak_drop = base.peak_allocated_hbm_bytes - cand.peak_allocated_hbm_bytes
    lm_loss_drop = base.lm_head_loss_hbm_bytes - cand.lm_head_loss_hbm_bytes
    step_ratio = cand.median_step_ms / base.median_step_ms
    fwd_ratio = cand.median_forward_ms / base.median_forward_ms
    bwd_ratio = cand.median_backward_ms / base.median_backward_ms
    require(peak_drop >= gib(args.min_peak_drop_gib), "peak drop too small")
    require(lm_loss_drop >= gib(args.min_lm_head_loss_drop_gib), "lm_head/loss drop too small")
    require(step_ratio <= args.max_step_ratio, "step latency regression")
    require(fwd_ratio <= args.max_forward_ratio, "forward latency regression")
    require(bwd_ratio <= args.max_backward_ratio, "backward latency regression")
```

Validation before Stage 4:

```bash
PYTHONPATH=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
.venv/bin/python -m pytest tests/lf/test_compare_liger_loss_profiles.py -q
```

Required evidence:
- Tests cover meaningful memory drop, trivial memory drop, latency regression, missing metrics, metadata/path mismatch, and same-axis failure.
- Baseline-vs-itself exits nonzero.
- Output reports all metric source files.

Efficiency rationale:
- This stage is analysis-only and must not change training execution.
- The compare tool must reject memory wins that come with forward/backward/step timing blowups.

Risks to watch:
- Source-memory schemas can evolve. The tool can tolerate alternate key names, but it must fail loudly with searched paths and missing metric names.
- If `lm_head`/`loss` attribution cannot be found, fail instead of accepting only peak-HBM evidence.

## Stage 4: Full E2E Acceptance

Purpose: decide whether `ligerloss1` is actually worth keeping for AsymGEMM and ZeRO3-offload.

Scope exercised:
- Stage 0 common interfaces, scripts, artifact axis, and plotting
- Stage 1 LF loss-only resolver
- Stage 2 Asym `lm_head` staging bridge
- Stage 3 compare tool

Actual code changes:
- None by default. This stage is the e2e acceptance gate for earlier stages.
- If acceptance fails, fix the responsible earlier-stage implementation and rerun that stage's validation before rerunning Stage 4.
- Do not tune thresholds or loosen checks to pass.

E2E run:

```bash
ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM/profiling_liger_acceptance
rm -rf "${ROOT}"
OUTPUT_ROOT="${ROOT}" \
RUN_NAME=qwen3_ligerloss_fourway \
NUMACTL_MEMBIND=0,1 \
NUMACTL_CPUNODEBIND=0,1 \
NUMACTL_MODE=membind \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0,asym_cpuadamwds|norecomp|ligerloss1,zero3_offload|recomp|ligerloss0,zero3_offload|recomp|ligerloss1' \
PROFILERS=both \
GPU_POOL=3 \
WARMUP_STEPS=5 \
MAX_STEPS=5 \
OVERWRITE=true \
bash scripts/lf/profile_lora_lf.sh 2>&1 | tee /tmp/qwen3_ligerloss_fourway.log
```

Artifact checks:

```bash
find "${ROOT}" -type d \( -name '*__source__*' -o -name '*__nsys__*' \) \
  ! -name '*__ligerloss0*' ! -name '*__ligerloss1*' -print -quit | grep -q . && exit 1 || true

rg -n 'Liger loss-only kernel has been applied|asym_liger_lm_head_bridge' "${ROOT}" --glob 'train.log' --glob 'profile.json' --glob 'source_profile.json'
rg -n 'liger_loss' "${ROOT}" --glob 'profile.json' --glob 'source_profile.json' --glob 'jobs.tsv' --glob '*.csv' --glob '*.json'
```

Plot rerun checks:

```bash
.venv/bin/python scripts/plotting/plot_activation_recompute_sweep.py \
  --input-root "${ROOT}" \
  --output-dir "${ROOT}/plot_check/timing" \
  --combined-output-dir "${ROOT}/plot_check/timing/_combined" \
  --clean-output --combined-only \
  --liger-loss ligerloss0 --liger-loss ligerloss1

.venv/bin/python scripts/plotting/plot_lf_memory_breakdown.py \
  --input-root "${ROOT}" \
  --output-dir "${ROOT}/plot_check/memory" \
  --clean-output --combined-only \
  --liger-loss ligerloss0 --liger-loss ligerloss1

.venv/bin/python scripts/plotting/plot_lf_interconnect_ctc.py \
  --input-root "${ROOT}" \
  --output-dir "${ROOT}/plot_check/c2c" \
  --clean-output --combined-only \
  --liger-loss ligerloss0 --liger-loss ligerloss1
```

Comparison commands:

```bash
ASYM_BASE="$(find "${ROOT}" -type d -path '*asym_cpuadamwds*__source__*ligerloss0*' -name 'b*_s*_ga*' -print -quit)"
ASYM_CAND="$(find "${ROOT}" -type d -path '*asym_cpuadamwds*__source__*ligerloss1*' -name 'b*_s*_ga*' -print -quit)"
ZERO_BASE="$(find "${ROOT}" -type d -path '*zero3_offload*__source__*ligerloss0*' -name 'b*_s*_ga*' -print -quit)"
ZERO_CAND="$(find "${ROOT}" -type d -path '*zero3_offload*__source__*ligerloss1*' -name 'b*_s*_ga*' -print -quit)"
test -n "${ASYM_BASE}" && test -n "${ASYM_CAND}" && test -n "${ZERO_BASE}" && test -n "${ZERO_CAND}"

.venv/bin/python scripts/lf/compare_liger_loss_profiles.py \
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

.venv/bin/python scripts/lf/compare_liger_loss_profiles.py \
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

Kernel launch sanity check:

```bash
.venv/bin/python - "${ROOT}" <<'PY'
import csv, sys
from pathlib import Path
root = Path(sys.argv[1])
for csv_path in sorted(root.rglob("timing_by_op.csv")):
    if "ligerloss1" not in str(csv_path):
        continue
    rows = list(csv.DictReader(csv_path.open()))
    gemm_rows = [r for r in rows if "gemm" in " ".join(str(v).lower() for v in r.values())]
    tiny = [
        r for r in gemm_rows
        if any(float(r.get(k) or 9999) < 0.05 for k in ("milliseconds", "total_milliseconds", "duration_ms"))
    ]
    print(csv_path, "gemm_rows", len(gemm_rows), "tiny_gemm_rows", len(tiny))
PY
```

Required evidence:
- Every job folder and run folder is unambiguous by `ligerloss0/1`.
- `jobs.tsv`, profile metadata, timing CSVs, memory CSVs, plot metadata, and grouped plot paths carry `liger_loss`.
- `ligerloss1` logs show `Liger loss-only kernel has been applied`.
- AsymGEMM `ligerloss1` profile metadata shows `asym_liger_lm_head_bridge.enabled=true` and `asym_liger_lm_head_bridge.weight_source=asym_host_staged`.
- AsymGEMM `ligerloss1` logs still show `lm_head` wrapped/offloaded when `ASYM_OFFLOAD_MODULES=all`.
- Source attribution shows meaningful `lm_head`/`loss` memory reduction.
- Nsight timing does not show a large new population of tiny GEMMs or per-expert launch loops.
- Compare tool passes thresholds for a backend before accepting that backend.

Efficiency rationale:
- The intended memory win is avoiding full final logits/loss materialization.
- The accepted Asym path pays one full `lm_head` CPU-to-GPU staging copy per loss call. This must still be outweighed by the logits/loss memory saving.
- If HBM drops but forward/backward/step timing regresses past threshold, reject the backend.

Decision:
- Accept `ligerloss1` per backend, not globally.
- If AsymGEMM passes and ZeRO3-offload fails, keep the axis but mark only AsymGEMM accepted.
- If AsymGEMM fails because staged `lm_head` cost erases the win, reject the Asym `ligerloss1` path rather than disabling `lm_head` offload silently.

Risks to watch:
- Warmup 1/step 1 is allowed for smoke tests only. Acceptance must use at least `WARMUP_STEPS=5 MAX_STEPS=5`.
- Logits memory scales with `batch * seq_len * vocab`, so small toy runs can understate savings.

## Stage 5: Future Compatible Models

Purpose: add future model support without accidentally enabling incompatible model patches.

Scope:
- Modify only:
  - `_LOSS_ONLY_SUPPORTED_MODEL_TYPES` in `../LlamaFactory/src/llamafactory/model/model_utils/liger_kernel.py`
  - model-specific bridge code in `asym_gemm/integrations/liger_loss.py`
  - model-specific tests next to `tests/lf/test_liger_loss_only_qwen3_moe.py`
- Reuse Stage 1 through Stage 4 validation commands with the future model's real e2e workload.

Actual code changes:
- Add one exact `config.model_type` value only after compatibility is proven.
- Do not add model families, broad prefixes, or name-similarity matches.
- Do not modify `_build_liger_loss_only_kwargs()` unless the new model exposes a genuinely incompatible apply signature; if modified, rerun Qwen3-MoE tests.
- Add a model-specific local forward bridge only if the model's Liger fused-loss forward directly reads `self.lm_head.weight` and needs Asym staging.

Rule:
- Add a model only after proving all of the following:
  - Liger exposes `fused_linear_cross_entropy` for that model.
  - `_build_liger_loss_only_kwargs()` disables every other boolean patch.
  - The model's original modules remain recognizable to any AsymGEMM wrappers.
  - The Asym bridge preserves `lm_head` offload when selected.
  - Tiny CUDA forward/backward passes in a fresh subprocess.
  - Full e2e `PROFILERS=both` profiling passes Stage 4 thresholds.
- Do not add Llama4, dense Qwen3, Qwen3.5, Qwen3.5-MoE, or any other model by name similarity.

Efficiency rationale:
- New models must reuse fused-loss-only behavior.
- Do not add model-specific expert loops, split expert GEMMs, or hidden persistent GPU copies.

Risks to watch:
- Public Liger support for a model does not imply AsymGEMM compatibility. Treat each model as unvalidated until Stage 4 passes.
- If a model requires logits for the active training stage, skip loss-only Liger for that model.

References checked:
- Local Liger fused CE source: `../Liger-Kernel/src/liger_kernel/ops/fused_linear_cross_entropy.py`
- Local Liger Qwen3-MoE fused-loss forward: `../Liger-Kernel/src/liger_kernel/transformers/model/qwen3_moe.py`
- Local Liger Qwen3-MoE apply function: `../Liger-Kernel/src/liger_kernel/transformers/monkey_patch.py`
- Local Asym frozen linear wrapper: `asym_gemm/training/frozen_linear.py`
- Local Asym LF adapter path: `asym_gemm/integrations/lf.py`, `asym_gemm/integrations/peft_lf.py`, and `../LlamaFactory/src/llamafactory/model/adapter.py`
