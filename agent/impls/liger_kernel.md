# Staged Plan: AsymGEMM + Liger Loss-Only

Goal: enable Liger only for Qwen3-MoE causal-LM fused linear cross entropy in the AsymGEMM LoRA-SFT path. Do not let Liger replace MoE experts, RoPE, RMSNorm, SwiGLU, or other model internals. The change is accepted only if e2e LoRA profiling shows a meaningful memory reduction and forward/backward/step latency does not regress beyond the configured guardrails.

Acceptance rule for the whole feature:
- Keep the change only if e2e `profile_lora_lf.sh` source+nsys artifacts show a meaningful HBM reduction. Default threshold: peak CUDA allocated drops by at least 10 GiB and source attribution for `lm_head`/`loss` drops by at least 20 GiB, unless a later measured baseline justifies a stricter threshold.
- Reject the change if median measured step latency is above 1.10x baseline, or if forward or backward median latency is above 1.15x baseline. Also reject same-memory/slower and trivial-memory/slower outcomes.

Toy tests are only functional preflight. They are not enough to accept the feature.

## Stage 0: Dependency And Baseline

Scope:
- No production code changes.
- Confirm local dependency behavior against:
  - `third_party/Liger-Kernel/src/liger_kernel/transformers/monkey_patch.py`
  - `third_party/Liger-Kernel/src/liger_kernel/transformers/model/qwen3_moe.py`
  - `third_party/Liger-Kernel/src/liger_kernel/transformers/model/loss_utils.py`
  - `third_party/LlamaFactory/src/llamafactory/model/model_utils/liger_kernel.py`
- Install local Liger into the exact virtualenv used by LF profiling.

Concrete checks:
- The local Liger `apply_liger_kernel_to_qwen3_moe(...)` exposes `fused_linear_cross_entropy`, `cross_entropy`, `swiglu`, `rms_norm`, and `rope`.
- Default Liger Qwen3-MoE patches experts when `swiglu=True`; this must not be used in AsymGEMM.
- Liger loss-only mode with `swiglu=False` keeps HF `Qwen3MoeExperts`, so AsymGEMM can still wrap MoE blocks.
- Upstream primary source check: the current Liger public `monkey_patch.py` has the same Qwen3-MoE apply-function shape, so the local implementation is not relying on a stale private API: `https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py`.

Commands:

```bash
/workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python - <<'PY'
import inspect
from packaging.version import Version
import torch
import triton

assert Version(torch.__version__.split("+")[0]) >= Version("2.1.2"), torch.__version__
assert Version(triton.__version__) >= Version("2.3.1"), triton.__version__
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("triton", triton.__version__)
PY

/workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python \
  -m pip install --no-deps -e /workspace/AsymGEMM-SFT/third_party/Liger-Kernel

/workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python - <<'PY'
import inspect
from importlib.metadata import version
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe

print("liger-kernel", version("liger-kernel"))

sig = inspect.signature(apply_liger_kernel_to_qwen3_moe)
required = {"fused_linear_cross_entropy", "cross_entropy", "swiglu", "rms_norm", "rope"}
missing = required - set(sig.parameters)
assert not missing, missing
print(sig)
PY
```

Baseline e2e profile, before code changes:

```bash
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/profiling_liger_acceptance \
RUN_NAME=qwen3_asym_noliger_baseline \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
PROFILERS=both \
ENABLE_LIGER_KERNEL=false \
GPU_POOL=3 \
WARMUP_STEPS=5 \
MAX_STEPS=5 \
OVERWRITE=true \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Validation before Stage 1:
- The baseline run creates both source and nsys artifacts under `profiling_liger_acceptance`.
- The expected config root is discoverable as `profiling_liger_acceptance/qwen3_asym_noliger_baseline__drop0p00*`.
- The setup log confirms Qwen3-MoE AsymGEMM blocks are wrapped.
- The memory summary, source attribution plots, and timing breakdown plots are present.
- Record baseline peak CUDA allocated/reserved, source-attributed `lm_head`/`loss` memory, forward median latency, backward median latency, and total step median latency.

Risks to watch:
- The first Liger/Triton run may include compile overhead. Do not use first-step timing for acceptance.
- If Liger is not installed in the same Python env used by LF, LF will fail early through its existing `check_version("liger-kernel")` check.

## Stage 1: LlamaFactory Loss-Only Resolver

Files and functions:
- Modify `third_party/LlamaFactory/src/llamafactory/model/model_utils/liger_kernel.py`
  - Refactor `apply_liger_kernel(config, model_args, is_trainable, require_logits)`.
  - Add `LigerApplySpec`.
  - Add `_LIGER_APPLY_SPECS`.
  - Add `_resolve_liger_apply(model_type)`.
  - Add `_build_liger_loss_only_kwargs(apply_fn)`.
- Add `tests/lf/test_liger_loss_only_qwen3_moe.py`
  - `test_qwen3_moe_loss_only_kwargs_disable_non_loss_patches`.
  - `test_asym_liger_skips_unvalidated_model_type`.
  - `test_qwen3_moe_loss_only_preserves_hf_experts_for_asym_wrap`.
  - `test_qwen3_moe_loss_only_cuda_forward_backward`.

Implementation:
- Preserve current non-AsymGEMM behavior.
- In the AsymGEMM path, never call a Liger apply function with default kwargs.
- Initially validate only `qwen3_moe` for AsymGEMM loss-only. Do not enable Llama4 or any other model until a matching wrapper-compatibility test exists.
- If `use_asym_gemm=true` and the model type is not explicitly validated, skip Liger and log a warning.
- If `require_logits=true`, skip Liger in the AsymGEMM path because fused linear CE intentionally avoids materializing logits during training loss.

Pseudocode:

```python
from dataclasses import dataclass
import inspect


@dataclass(frozen=True)
class LigerApplySpec:
    model_types: tuple[str, ...]
    import_name: str
    asym_loss_only_supported: bool = False


_LIGER_APPLY_SPECS = (
    # Preserve every existing model mapping from the current if/elif chain.
    LigerApplySpec(("qwen3_moe",), "apply_liger_kernel_to_qwen3_moe", asym_loss_only_supported=True),
    LigerApplySpec(("qwen3",), "apply_liger_kernel_to_qwen3"),
    LigerApplySpec(("qwen3_next",), "apply_liger_kernel_to_qwen3_next"),
    LigerApplySpec(("qwen3_5",), "apply_liger_kernel_to_qwen3_5"),
    # existing entries: llama, mixtral, mistral, phi, gemma, deepseek, etc.
)


def _resolve_liger_apply(model_type: str | None):
    if model_type is None:
        return None, None

    import liger_kernel.transformers as liger_transformers

    for spec in _LIGER_APPLY_SPECS:
        if model_type in spec.model_types:
            return getattr(liger_transformers, spec.import_name, None), spec

    return None, None


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

    model_type = getattr(config, "model_type", None)
    apply_fn, spec = _resolve_liger_apply(model_type)
    if apply_fn is None:
        logger.warning_rank0("Current model does not support liger kernel.")
        return

    if getattr(model_args, "use_asym_gemm", False):
        if require_logits:
            logger.warning_rank0("Skipping Liger loss-only for AsymGEMM because logits are required.")
            return
        if spec is None or not spec.asym_loss_only_supported:
            logger.warning_rank0(f"Skipping AsymGEMM Liger loss-only: {model_type!r} is not validated.")
            return

        kwargs = _build_liger_loss_only_kwargs(apply_fn)
        if kwargs is None:
            logger.warning_rank0(f"Skipping AsymGEMM Liger loss-only: {model_type!r} lacks fused CE.")
            return

        apply_fn(**kwargs)
        logger.info_rank0(f"Liger loss-only kernel has been applied for AsymGEMM model_type={model_type}.")
        return

    sig = inspect.signature(apply_fn)
    if require_logits and "fused_linear_cross_entropy" in sig.parameters:
        kwargs = {"fused_linear_cross_entropy": False, "cross_entropy": True}
    else:
        kwargs = {}
    apply_fn(**kwargs)
    logger.info_rank0("Liger kernel has been applied to the model.")
```

Functional validation before Stage 2:

```bash
PYTHONPATH=/home/kevinni/AsymGEMM-SFT/third_party/LlamaFactory/src:/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/src \
/workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python - <<'PY'
from types import SimpleNamespace
from transformers import Qwen3MoeConfig
from llamafactory.model.model_utils.liger_kernel import (
    _resolve_liger_apply,
    _build_liger_loss_only_kwargs,
    apply_liger_kernel,
)

fn, spec = _resolve_liger_apply("qwen3_moe")
assert fn is not None
assert spec.asym_loss_only_supported is True
kwargs = _build_liger_loss_only_kwargs(fn)
assert kwargs["fused_linear_cross_entropy"] is True
assert kwargs.get("cross_entropy") is False
assert kwargs.get("swiglu") is False
assert kwargs.get("rms_norm") is False
assert kwargs.get("rope") is False

cfg = Qwen3MoeConfig()
args = SimpleNamespace(enable_liger_kernel=True, use_asym_gemm=True)
apply_liger_kernel(cfg, args, is_trainable=True, require_logits=False)
print("loss-only resolver ok")
PY

PYTHONPATH=/home/kevinni/AsymGEMM-SFT/third_party/LlamaFactory/src:/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM \
/workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python \
  -m pytest /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/tests/lf/test_liger_loss_only_qwen3_moe.py -q
```

Required pytest evidence:
- `test_qwen3_moe_loss_only_kwargs_disable_non_loss_patches` proves `fused_linear_cross_entropy=True` and `cross_entropy/swiglu/rms_norm/rope=False`.
- `test_asym_liger_skips_unvalidated_model_type` uses a fresh Python subprocess for a non-validated model type such as `llama4_text` and proves no Liger patch is applied in the AsymGEMM branch.
- `test_qwen3_moe_loss_only_preserves_hf_experts_for_asym_wrap` builds a tiny `Qwen3MoeForCausalLM`, applies the LF wrapper with `use_asym_gemm=true`, confirms `type(model).forward.__module__ == "liger_kernel.transformers.model.qwen3_moe"`, confirms the experts module still comes from `transformers.models.qwen3_moe`, then runs AsymGEMM adapter setup and confirms the MLP becomes `AsymQwen3MoeBlock`.
- `test_qwen3_moe_loss_only_cuda_forward_backward` runs in a fresh Python subprocess on CUDA, executes one tiny forward/backward, and proves finite loss, `outputs.logits is None`, and expected LoRA grads exist.

Risks to watch:
- Liger monkey patches global classes in a Python process. Run positive and negative tests in fresh Python processes.
- Future Liger versions may add non-boolean patch options. The loss-only builder must only disable boolean feature toggles and must not pass unknown values for `model`.

## Stage 2: Script Plumbing And Artifact Isolation

Files and functions:
- Modify `third_party/AsymGEMM/scripts/lf/run_lf_lora_sft.sh`
  - Add `ENABLE_LIGER_KERNEL=${ENABLE_LIGER_KERNEL:-false}` near user training/model toggles.
  - Add `--enable_liger_kernel "${ENABLE_LIGER_KERNEL}"` to `CMD_ARGS`.
  - Add a default run-id suffix such as `ligerloss` only when enabled and only for script-created default `RUN_ID`s. Do not mutate a caller-provided `RUN_ID`, because `profile_lora_lf.sh` already owns profiled job naming.
  - Add logging for `ENABLE_LIGER_KERNEL`.
- Modify `third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh`
  - Add `ENABLE_LIGER_KERNEL=${ENABLE_LIGER_KERNEL:-false}`.
  - Add CLI parsing/help for `--enable-liger-kernel true|false` if this script already exposes similar env-backed options.
  - In `job_root_path()`, append `ligerloss` only for jobs that actually pass `ENABLE_LIGER_KERNEL=true`.
  - In `run_job()`, pass `ENABLE_LIGER_KERNEL` into the launched LF run.
  - Gate the profiler sweep so `ENABLE_LIGER_KERNEL=true` is passed only to AsymGEMM backends. For `zero3_offload`, pass `false` unless a future explicit non-Asym Liger comparison flag is added.
- Modify `tests/lf/test_asym_cpu_adamw_args.py`
  - Add `test_run_lf_lora_sft_passes_enable_liger_kernel_arg`.
  - Add `test_profile_lora_lf_dry_run_liger_only_for_asym_backend`.
  - Add `test_profile_lora_lf_dry_run_liger_paths_do_not_collide`.

Implementation details:
- Treat the profile script's top-level `ENABLE_LIGER_KERNEL=true` as "try Liger loss-only for compatible AsymGEMM jobs".
- Do not let a mixed sweep accidentally apply default Liger to `zero3_offload`; that would corrupt baseline comparison.
- Artifact names must not collide:
  - No-Liger Asym path: existing path shape.
  - Liger Asym path: same path plus `__ligerloss`.
  - `PROFILERS=both` still writes under `profiling_both` unless `OUTPUT_ROOT` overrides it.
- Avoid double tags:
  - `profile_lora_lf.sh` should append `ligerloss` to its `job_root_path()` and generated `run_id`.
  - `run_lf_lora_sft.sh` should append `ligerloss` only to `DEFAULT_RUN_ID` when `RUN_ID` was not provided by the caller.

Pseudocode:

```bash
# run_lf_lora_sft.sh
ENABLE_LIGER_KERNEL=${ENABLE_LIGER_KERNEL:-false}
LIGER_KERNEL_TAG=""

if [[ "${ENABLE_LIGER_KERNEL}" == "true" ]]; then
  LIGER_KERNEL_TAG="_ligerloss"
fi

# Include ${LIGER_KERNEL_TAG} in DEFAULT_RUN_ID before RUN_ID=${RUN_ID:-${DEFAULT_RUN_ID}}.
# Do not append to RUN_ID after that assignment.

CMD_ARGS+=(
  --enable_liger_kernel "${ENABLE_LIGER_KERNEL}"
)
```

```bash
# profile_lora_lf.sh
ENABLE_LIGER_KERNEL=${ENABLE_LIGER_KERNEL:-false}

is_asym_backend() {
  case "$1" in
    asym|asym_torch|asym_cpuadamwtorch|asym_cpuadamwds) return 0 ;;
    *) return 1 ;;
  esac
}

job_liger_enabled() {
  local backend="$1"
  [[ "${ENABLE_LIGER_KERNEL}" == "true" ]] && is_asym_backend "${backend}"
}

job_root_path() {
  local liger_suffix=""
  if job_liger_enabled "${backend}"; then
    liger_suffix="__ligerloss"
  fi
  printf "%s/%s__%s%s" "${config_root}" "${backend}" "${profiler}" "${liger_suffix}"
}

run_job() {
  local job_enable_liger_kernel=false
  local liger_run_suffix=""
  if job_liger_enabled "${backend}"; then
    job_enable_liger_kernel=true
    liger_run_suffix="_ligerloss"
  fi

  run_id="lf_${backend}_${run_profiler}${liger_run_suffix}_${recompute}_..."

  env_args+=(
    "ENABLE_LIGER_KERNEL=${job_enable_liger_kernel}"
    "ASYM_GEMM_LF_CONFIG_ENABLE_LIGER_KERNEL=${job_enable_liger_kernel}"
  )
}
```

Validation before Stage 3:

```bash
PYTHONPATH=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM \
/workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python \
  -m pytest /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/tests/lf/test_asym_cpu_adamw_args.py -q -k 'liger'

OUTPUT_ROOT=/tmp/asym_liger_dryrun \
RUN_NAME=qwen3_liger_dryrun \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp,zero3_offload|recomp' \
PROFILERS=both \
ENABLE_LIGER_KERNEL=true \
GPU_POOL=3 \
WARMUP_STEPS=1 \
MAX_STEPS=1 \
DRY_RUN=true \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Required evidence:
- The fake-LF unit test for `run_lf_lora_sft.sh` proves `ENABLE_LIGER_KERNEL=true` produces LF args `--enable_liger_kernel true`, and false produces `--enable_liger_kernel false`.
- The profile dry-run command file for `asym_cpuadamwds` contains `ENABLE_LIGER_KERNEL=true`.
- The profile dry-run command file for `zero3_offload` contains `ENABLE_LIGER_KERNEL=false`.
- Only the AsymGEMM job path includes `ligerloss`.
- The plotted/source/nsys output roots for `source`, `nsys`, and `both` remain separated as they are today.

Risks to watch:
- If any plotting script groups jobs by path tokens, the new `ligerloss` suffix may need to be added as a parsed feature column.
- If profile metadata has an explicit schema, add `enable_liger_kernel` there too; otherwise the path suffix and LF log are the minimum required traceability.

## Stage 3: Acceptance Comparison Tool

Files and functions:
- Add `third_party/AsymGEMM/scripts/lf/compare_liger_loss_profiles.py`
  - `find_profile_artifacts(root)`.
  - `load_memory_metrics(path_or_root)`.
  - `load_timing_metrics(path_or_root)`.
  - `compare_metrics(baseline, candidate, thresholds)`.
  - `main()`.
- Add `tests/lf/test_compare_liger_loss_profiles.py`
  - `test_compare_same_profile_fails_memory_threshold`.
  - `test_compare_rejects_latency_regression`.
  - `test_compare_accepts_meaningful_memory_drop_without_latency_regression`.
  - `test_parser_reports_missing_required_metrics`.

Implementation:
- The tool consumes only e2e profile artifacts from `profile_lora_lf.sh`; it must not use toy runs.
- It should accept either job roots or the shared `OUTPUT_ROOT` plus run-name filters.
- It should parse the existing JSON/CSV summary artifacts if available. If an artifact name varies by profiler, search recursively for files containing memory summaries, source attribution, and timing breakdowns.
- It should emit a JSON report and a concise markdown summary.
- It should exit nonzero when memory reduction is below threshold or latency exceeds threshold.

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


def compare_metrics(base, cand, thresholds):
    peak_drop = base.peak_allocated_gib - cand.peak_allocated_gib
    loss_drop = base.lm_head_loss_gib - cand.lm_head_loss_gib
    fwd_ratio = cand.forward_median_ms / base.forward_median_ms
    bwd_ratio = cand.backward_median_ms / base.backward_median_ms
    step_ratio = cand.step_median_ms / base.step_median_ms

    failures = []
    if peak_drop < thresholds.min_peak_drop_gib:
        failures.append(f"peak drop {peak_drop:.2f} GiB is below threshold")
    if loss_drop < thresholds.min_lm_head_loss_drop_gib:
        failures.append(f"lm_head/loss drop {loss_drop:.2f} GiB is below threshold")
    if step_ratio > thresholds.max_step_ratio:
        failures.append(f"step latency ratio {step_ratio:.3f} exceeds threshold")
    if fwd_ratio > thresholds.max_forward_ratio:
        failures.append(f"forward latency ratio {fwd_ratio:.3f} exceeds threshold")
    if bwd_ratio > thresholds.max_backward_ratio:
        failures.append(f"backward latency ratio {bwd_ratio:.3f} exceeds threshold")

    return failures
```

Default thresholds:

```text
--min-peak-drop-gib 10
--min-lm-head-loss-drop-gib 20
--max-step-ratio 1.10
--max-forward-ratio 1.15
--max-backward-ratio 1.15
```

Validation before Stage 4:
- Run the tool on the no-Liger baseline against itself and confirm it fails for insufficient memory reduction.
- Run it on a copied baseline artifact with artificially edited timing/memory values and confirm pass/fail behavior is correct.
- Confirm the tool points to the exact artifact files it parsed, so bad comparisons are debuggable.

Commands:

```bash
PYTHONPATH=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM \
/workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python \
  -m pytest /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/tests/lf/test_compare_liger_loss_profiles.py -q

BASE_ROOT="$(
  find /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/profiling_liger_acceptance \
    -maxdepth 1 -type d -name 'qwen3_asym_noliger_baseline__drop0p00*' -print -quit
)"
test -n "${BASE_ROOT}"

/workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python \
  /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/compare_liger_loss_profiles.py \
  --baseline "${BASE_ROOT}" \
  --candidate "${BASE_ROOT}"
```

Expected result: nonzero exit with a message that the memory drop is below threshold.

Risks to watch:
- Existing profile artifact schemas may not expose all metrics in one stable file. Resolve by making the parser tolerant but explicit: print missing metric names and parsed file paths.

## Stage 4: Full E2E Liger Acceptance

Files and functions:
- No new implementation files unless Stage 3 exposed missing metadata.
- Exercise the implemented files:
  - `third_party/LlamaFactory/src/llamafactory/model/model_utils/liger_kernel.py`
  - `third_party/AsymGEMM/scripts/lf/run_lf_lora_sft.sh`
  - `third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh`
  - `third_party/AsymGEMM/scripts/lf/compare_liger_loss_profiles.py`

Candidate command:

```bash
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/profiling_liger_acceptance \
RUN_NAME=qwen3_asym_ligerloss_candidate \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
PROFILERS=both \
ENABLE_LIGER_KERNEL=true \
GPU_POOL=3 \
WARMUP_STEPS=5 \
MAX_STEPS=5 \
OVERWRITE=true \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Comparison command:

```bash
BASE_ROOT="$(
  find /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/profiling_liger_acceptance \
    -maxdepth 1 -type d -name 'qwen3_asym_noliger_baseline__drop0p00*' -print -quit
)"
CAND_ROOT="$(
  find /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/profiling_liger_acceptance \
    -maxdepth 1 -type d -name 'qwen3_asym_ligerloss_candidate__drop0p00*' -print -quit
)"
test -n "${BASE_ROOT}"
test -n "${CAND_ROOT}"

/workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python \
  /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/compare_liger_loss_profiles.py \
  --baseline "${BASE_ROOT}" \
  --candidate "${CAND_ROOT}" \
  --min-peak-drop-gib 10 \
  --min-lm-head-loss-drop-gib 20 \
  --max-step-ratio 1.10 \
  --max-forward-ratio 1.15 \
  --max-backward-ratio 1.15
```

Required artifact checks:
- LF log contains `Liger loss-only kernel has been applied for AsymGEMM model_type=qwen3_moe`.
- LF/AsymGEMM setup log still confirms Qwen3-MoE blocks are wrapped by `AsymQwen3MoeBlock`.
- Candidate outputs contain both source and nsys timing artifacts.
- Candidate source memory attribution shows the drop in `lm_head`/`loss` memory, not a suspicious shift into many new tiny operations.
- Candidate nsys timeline does not show a large number of new small GEMMs or expert loops. The loss path may have Liger chunked vocab work, but expert computation must remain AsymGEMM-owned and not devolve into per-expert small GEMMs.
- Candidate timing plots show forward, backward, and total measured step latency within the thresholds.

Decision:
- Accept and keep the feature only if the comparison tool passes and the visual artifacts agree with the numeric result.
- Reject or disable by default if memory reduction is below threshold, if latency exceeds threshold, or if the memory win comes with inefficient kernel-launch patterns.

Risks to watch:
- Fused CE can reduce memory but increase loss-kernel time depending on sequence length, vocab size, and chunking. The nsys timing check is mandatory.
- If the peak allocator metric does not move because another earlier module dominates the peak, require a clear source-attributed `lm_head`/`loss` reduction and no increase in total peak reserved before considering a threshold adjustment.

## Stage 5: Future Compatible Models

Files and functions:
- Modify only the registry in `third_party/LlamaFactory/src/llamafactory/model/model_utils/liger_kernel.py`.
- Add a new tiny compatibility test for the model's AsymGEMM wrapper before setting `asym_loss_only_supported=True`.

Implementation rule:
- A future model type may be added only when all of these are true:
  - Liger apply function exposes `fused_linear_cross_entropy`.
  - Loss-only kwargs disable every other boolean patch feature.
  - The model's original modules remain recognizable by the corresponding AsymGEMM wrapper.
  - A CUDA forward/backward produces finite loss, no logits for training loss, and expected LoRA grads.
  - E2E profiling passes the same memory and latency guardrails.

Do not add Llama4, Qwen3 dense, or other model types by assumption. Add them only after the same compatibility and e2e profiling gates pass.
