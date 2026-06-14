# Activation-Offload LoRA-A A/B Implementation Plan

Target workload:

```text
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1"
BACKEND_SPECS="asym_cpuadamwds|norecomp"
ASYMM_EXP_ACT_POLICIES="none|true|true|true"
SEQ_LENS=4096
PER_DEVICE_TRAIN_BATCH_SIZE=4
LORA_DROPOUT=0.00
WARMUP_STEPS=5
MAX_STEPS=10
PROFILERS=source
```

The A/B compares the same full expert + attention + decoder-layer activation
offload stack and changes only expert activation-offload forward LoRA-A source:

```text
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu_left  # current behavior, default
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=gpu_hbm   # experimental behavior
```

Baseline A is `34.593 GiB` peak allocated, `39.676 GiB` peak reserved,
`45.861 s` avg step, `11.490 s` avg forward, and `34.213 s` avg backward.
Reject any B that approaches the old `58.343 GiB` expert+attention-only row.
Treat `<=36 GiB` peak allocated as acceptable, investigate `36-40 GiB`, and
reject `>40 GiB`.

External API assumptions checked before implementation:

- `torch.cat` returns a concatenated tensor, so `cat(A_gate, A_up, dim=1)` is a
  real temporary and must stay bounded to LoRA-A weights only.
- `torch.split` returns views, so split low-rank outputs must keep their owner
  live until both views have been consumed/offloaded.
- `torch.nn.functional.grouped_mm` supports the 2D-left, 3D-right, int32-offset
  MoE shape that `grouped_expert_lora(...)` already uses.

## Stage 0: Lock Comparable Trainable Surface

Purpose: make sure A/B rows compare the same attention+expert LoRA trainable
surface. This stage should already be implemented in the current tree; do not
move on if any validation fails.

Files, functions, and classes:

- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/model/adapter.py`
  - `_setup_lora_tuning`
- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/model/model_utils/fused_moe_lora.py`
  - `apply_fused_moe_lora`
  - `count_fused_moe_lora`
- `scripts/lf/run_lf_profiled_train.py`
  - `_lora_counters_from_model`
  - `_kt_optimizer_memory_preflight`
- `scripts/lf/postprocess_lf_profile_artifacts.py`
  - `_trainable_surface_summary`
- `scripts/lf/run_lf_lora_sft.sh`
  - `check_trainable_surface_if_requested`
- `tests/lf/test_asym_cpu_adamw_lf_integration.py`
- `tests/lf/test_lf_profile_postprocess.py`

Implementation if validation shows a regression:

```python
# adapter.py::_setup_lora_tuning
model = peft_wrap_dense_attention_lora(...)
if lora_target == "all" and model_has_qwen3_packed_experts(model):
    apply_fused_moe_lora(model, rank=finetuning_args.lora_rank,
                         alpha=finetuning_args.lora_alpha)
return model

# fused_moe_lora.py::apply_fused_moe_lora
for module in model.modules():
    if looks_like_qwen3_packed_experts(module):  # has 3D gate_up_proj/down_proj
        attach _lf_fused_lora_params with:
            gate_lora_a/b, up_lora_a/b, down_lora_a/b
        patch/wrap forward so expert delta is applied
        do not add LoRA to module.gate/router

# run_lf_profiled_train.py::_lora_counters_from_model
count peft attention LoRA params
count LF fused expert LoRA params through count_fused_moe_lora(model)
return trainable_parameters and lf_fused_expert_lora_parameters

# postprocess_lf_profile_artifacts.py::_trainable_surface_summary
expert = peft_expert_lora_parameters + lf_fused_expert_lora_parameters
if non_expert_peft_lora_parameters > 0 and expert > 0:
    surface = "attention+expert LoRA"
elif non_expert_peft_lora_parameters > 0:
    surface = "attention-only LoRA"
else:
    surface = "unknown"
```

Risks to watch:

- A `zero3_offload|recomp` source profile with only `53,477,376` trainable
  params is attention-only and is not comparable to AsymGEMM rows.
- If fused expert LoRA is absent in the LF environment, the postprocessed
  `profile.json` must fail `check_trainable_surface_if_requested`, not silently
  pass.

Validation:

```bash
.venv/bin/python -m pytest -q \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_lf_plain_lora_all_adapter_adds_qwen3_fused_expert_lora \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_lf_peft_lora_all_does_not_add_adapter_to_qwen_moe_routers \
  tests/lf/test_lf_profile_postprocess.py::test_source_summary_flags_lf_fused_attention_plus_expert_surface \
  tests/lf/test_lf_profile_postprocess.py::test_source_summary_flags_attention_only_surface
```

Then force a fresh ZeRO source profile before using ZeRO as a comparison row:

```bash
OUTPUT_ROOT=outputs/lf_ab_stage0_zero \
OVERWRITE=true \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="zero3_offload|recomp" \
ASYMM_EXP_ACT_POLICIES="none|false|false|false" \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
DATASET_MIN_TOKENS=4096 \
DATASET_OVERWRITE=true \
LORA_DROPOUT=0.00 \
PROFILERS=source \
PROFILE_MEMORY_ATTRIBUTION=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_SNAPSHOT=false \
WARMUP_STEPS=5 \
MAX_STEPS=10 \
RUN_POST=false \
scripts/lf/profile_lora_lf.sh --gpus 0
```

Acceptance:

- `reference_fallback_count=0` if present.
- Postprocessed `profile.json` has
  `trainable_surface.surface == "attention+expert LoRA"`.
- Postprocessed `profile.json` has
  `trainable_surface.expert_lora_parameters == 3,321,888,768`.
- `lora.trainable_parameters` is close to `3,375,366,144`.

## Stage 1: Add Selector, Run Identity, and Counters

Purpose: add the A/B control plane without changing default behavior. After
this stage, `cpu_left` must produce byte-for-byte equivalent behavior to the
current path, and `gpu_hbm` must fail clearly until Stage 2 implements it.

Files, functions, and classes:

- `asym_gemm/training/qwen3_moe.py`
  - module-level selector helper, for example
    `_expert_act_offload_lora_a_fwd_mode`
  - `AsymQwen3Experts._activation_offload_unsupported_reasons`
  - `_ActivationOffloadQwen3ExpertFunction.forward`
- `asym_gemm/training/frozen_linear.py`
  - `AsymExecutionStats`
- `asym_gemm/training/exp_act_offload_lora.py`
  - `grouped_lora_a_forward_cpu_left`
  - `grouped_lora_a_pair_forward_cpu_left`
- `scripts/lf/run_lf_lora_sft.sh`
  - top-level env defaults
  - activation-offload env normalization
  - logging block
  - `RUN_ENV`
  - profiler config env block
- `scripts/lf/profile_lora_lf.sh`
  - top-level env defaults
  - new selector normalization/tag helpers
  - `job_root_path`
  - `existing_profile_complete`
  - `kt_arm_matching_source_profile_complete`
  - `run_one`
- `scripts/lf/run_lf_profiled_train.py`
  - `_build_config` / config dict construction
- `scripts/testing/profile_qwen3_activation_offload.py`
  - `VariantResult`
  - `_profile_variant`
  - result config/comparison output
- tests:
  - `tests/lf/test_asym_cpu_adamw_args.py`
  - `tests/test_lf_memory_breakdown.py`
  - `tests/training/test_lf_qwen3_asym_backend.py`
  - `tests/training/test_cpu_left_lora.py`

Implementation:

1. Add the selector helper in `qwen3_moe.py`.

```python
EXPERT_ACT_OFFLOAD_LORA_A_FWD_ENV = "ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD"
VALID_EXPERT_ACT_OFFLOAD_LORA_A_FWD = {"cpu_left", "gpu_hbm"}

def _expert_act_offload_lora_a_fwd_mode() -> str:
    raw = os.environ.get(EXPERT_ACT_OFFLOAD_LORA_A_FWD_ENV, "cpu_left")
    mode = str(raw).strip().lower().replace("-", "_")
    if mode == "":
        mode = "cpu_left"
    if mode not in VALID_EXPERT_ACT_OFFLOAD_LORA_A_FWD:
        valid = ", ".join(sorted(VALID_EXPERT_ACT_OFFLOAD_LORA_A_FWD))
        raise ValueError(
            f"{EXPERT_ACT_OFFLOAD_LORA_A_FWD_ENV} must be one of {valid}, "
            f"got {raw!r}"
        )
    return mode
```

2. Fail closed for unimplemented `gpu_hbm`.

```python
# AsymQwen3Experts._activation_offload_unsupported_reasons
try:
    mode = _expert_act_offload_lora_a_fwd_mode()
except ValueError as exc:
    reasons.append(str(exc))
else:
    if mode == "gpu_hbm":
        reasons.append("gpu_hbm expert forward LoRA-A is not implemented yet")
```

Do not put a silent fallback in `_ActivationOffloadQwen3ExpertFunction.forward`.
For Stage 1, the forward body may read and record the mode but must still route
only `cpu_left`.

3. Add mode-specific expert counters while keeping the existing aggregate name.

```python
@dataclass
class AsymExecutionStats:
    cpu_left_lora_a_calls: int = 0  # low-level CPU-left calls, shared by expert and attention
    expact_lora_a_forward_grouped_calls: int = 0  # aggregate, kept for compatibility
    expact_lora_a_forward_cpu_left_grouped_calls: int = 0
    expact_lora_a_forward_hbm_grouped_calls: int = 0

    @property
    def forward_calls_total(self) -> int:
        return (
            self.asym_forward_calls
            + self.staged_forward_calls
            + self.torch_forward_calls
            + self.kt_forward_calls
            + self.cpu_left_lora_a_calls
            + self.expact_lora_a_forward_hbm_grouped_calls
            + self.attn_act_hbm_forward_calls
        )
```

The generic `cpu_left_lora_a_calls` already counts CPU-left kernels from both
expert and attention activation offload. Do not add
`expact_lora_a_forward_cpu_left_grouped_calls` to `forward_calls_total`, or the
CPU-left expert calls will be double counted. Add only the new HBM counter.

4. Update CPU-left expert helpers to fill the new subcounter.

```python
# exp_act_offload_lora.py::grouped_lora_a_forward_cpu_left
out = grouped_expert_lora_cpu_left(..., stats=stats)
if stats is not None:
    stats.expact_lora_a_forward_grouped_calls += 1
    stats.expact_lora_a_forward_cpu_left_grouped_calls += 1
return out

# grouped_lora_a_pair_forward_cpu_left stays a wrapper over two calls.
```

5. Add shell selector plumbing.

```bash
# profile_lora_lf.sh and run_lf_lora_sft.sh
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD:-cpu_left}

normalize_expact_lora_a_fwd() {
  case "${1,,}" in
    ""|cpu|cpu_left|cpu-left) printf 'cpu_left\n' ;;
    hbm|gpu|gpu_hbm|gpu-hbm) printf 'gpu_hbm\n' ;;
    *) die "ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD must be cpu_left or gpu_hbm, got '$1'" ;;
  esac
}

expact_lora_a_fwd_tag() {
  case "$(normalize_expact_lora_a_fwd "$1")" in
    cpu_left) printf 'loraafwdcpu\n' ;;
    gpu_hbm) printf 'loraafwdhbm\n' ;;
  esac
}
```

Apply the normalized value once and pass it everywhere:

```bash
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD="$(normalize_expact_lora_a_fwd "${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}")"
expact_lora_a_fwd_label="$(expact_lora_a_fwd_tag "${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}")"

# profile_lora_lf.sh::job_root_path
safe_label "${backend}__${profiler}__${recompute}__pol${expert_policy}__router${router_mode}__${expact_label}__${attnact_label}__${layeract_label}__${expact_lora_a_fwd_label}"

# profile_lora_lf.sh::run_one run_env
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD="${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}"
ASYM_GEMM_LF_CONFIG_ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD="${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}"

# run_lf_lora_sft.sh::RUN_ENV and profiler env
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD="${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}"
ASYM_GEMM_LF_CONFIG_ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD="${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD}"
```

6. Extend profile completeness checks.

```python
# embedded Python inside profile_lora_lf.sh::existing_profile_complete
expected_lora_a_fwd = sys.argv[20]
actual = normalize_mode(config.get("asymm_expert_act_offload_lora_a_fwd"))
wanted = normalize_mode(expected_lora_a_fwd)
if expected_lora_a_fwd and actual != wanted:
    raise SystemExit(
        "profile asymm_expert_act_offload_lora_a_fwd mismatch: "
        f"expected {wanted}, got {actual or '<missing>'}"
    )
```

Update all callers of `existing_profile_complete(...)`, including
`kt_arm_matching_source_profile_complete(...)`, to pass the selector. This
prevents a `cpu_left` source profile from being reused for a `gpu_hbm` run.

7. Record the selector in `source_profile.json`.

```python
# run_lf_profiled_train.py config dict
"asymm_expert_act_offload_lora_a_fwd": os.environ.get(
    "ASYM_GEMM_LF_CONFIG_ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD",
    os.environ.get("ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD", "cpu_left"),
),
```

8. Update the isolated profiler output, but do not use it as acceptance.

```python
# scripts/testing/profile_qwen3_activation_offload.py
@dataclass
class VariantResult:
    variant: str
    lora_a_forward_mode: str
    ...

def _profile_variant(..., lora_a_forward_mode: str = "cpu_left"):
    previous_mode = os.environ.get("ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD")
    os.environ["ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD"] = lora_a_forward_mode
    ...
    return VariantResult(..., lora_a_forward_mode=lora_a_forward_mode)
```

Risks to watch:

- Existing dry-run tests assert exact path substrings. Update them to include
  `__loraafwdcpu` for default runs.
- `cpu_left_lora_a_calls` will not drop to zero in full-stack `gpu_hbm` runs
  because attention activation offload still uses CPU-left LoRA-A. Use the new
  expert-specific subcounters for the A/B assertion.
- If `ASYMM_EXPERT_ACT_OFFLOAD=false`, the selector is recorded and labeled but
  has no runtime effect. That is intentional for artifact uniqueness.

Validation before Stage 2:

```bash
.venv/bin/python -m py_compile \
  asym_gemm/training/qwen3_moe.py \
  asym_gemm/training/exp_act_offload_lora.py \
  asym_gemm/training/frozen_linear.py \
  scripts/lf/run_lf_profiled_train.py

.venv/bin/python -m pytest -q \
  tests/lf/test_asym_cpu_adamw_args.py \
  tests/test_lf_memory_breakdown.py::test_source_profile_reports_activation_offload_counters \
  tests/training/test_cpu_left_lora.py::test_expact_lora_a_forward_wrappers_use_real_cpu_left \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend
```

Dry-run label and command artifact check:

```bash
OUTPUT_ROOT=/tmp/lf_ab_stage1_dryrun \
DRY_RUN=true \
PREPARE_DATASETS=false \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYMM_EXP_ACT_POLICIES="none|true|true|true" \
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu_left \
SEQ_LENS=128 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
MAX_STEPS=1 \
WARMUP_STEPS=0 \
PROFILERS=source \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
scripts/lf/profile_lora_lf.sh --gpus 0

find /tmp/lf_ab_stage1_dryrun -name command.txt -print -exec rg -n \
  "ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu_left|ASYM_GEMM_LF_CONFIG_ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu_left|loraafwdcpu" {} +
```

Fresh default-mode e2e regression profile:

```bash
OUTPUT_ROOT=outputs/lf_ab_stage1_cpu_left \
OVERWRITE=true \
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu_left \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYMM_EXP_ACT_POLICIES="none|true|true|true" \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
DATASET_MIN_TOKENS=4096 \
DATASET_OVERWRITE=true \
LORA_DROPOUT=0.00 \
PROFILERS=source \
PROFILE_MEMORY_ATTRIBUTION=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_SNAPSHOT=false \
WARMUP_STEPS=5 \
MAX_STEPS=10 \
RUN_POST=false \
scripts/lf/profile_lora_lf.sh --gpus 0
```

Acceptance:

- Source profile config has
  `asymm_expert_act_offload_lora_a_fwd == "cpu_left"`.
- `expact_lora_a_forward_cpu_left_grouped_calls > 0`.
- `expact_lora_a_forward_hbm_grouped_calls == 0`.
- `reference_fallback_count == 0`.
- HBM and timing are consistent with the current A row.

## Stage 2: Implement `gpu_hbm` Expert Forward LoRA-A

Purpose: route only expert activation-offload forward LoRA-A through existing
HBM grouped GEMM windows. Backward stays unchanged and still uses CPU-resident
saved activations.

Files, functions, and classes:

- `asym_gemm/training/exp_act_offload_lora.py`
  - imports from `.lora`
  - new `grouped_lora_a_forward_hbm`
  - `__all__`
- `asym_gemm/training/__init__.py`
  - export `grouped_lora_a_forward_hbm` if the helper is public
- `asym_gemm/training/qwen3_moe.py`
  - import `grouped_lora_a_forward_hbm`
  - `AsymQwen3Experts._activation_offload_unsupported_reasons`
  - `_ActivationOffloadQwen3ExpertFunction.forward`
- tests:
  - `tests/training/test_lf_qwen3_asym_backend.py`
  - `tests/training/test_cpu_left_lora.py`
  - `tests/test_lf_memory_breakdown.py`

Implementation:

1. Add the HBM helper.

```python
# exp_act_offload_lora.py
from .lora import GroupedLoRAMetadata, grouped_expert_lora, grouped_expert_lora_cpu_left

def _check_hbm_lora_a_inputs(source_hbm: torch.Tensor, lora_a: torch.Tensor, tag: str) -> None:
    if source_hbm.device.type != "cuda":
        raise RuntimeError(f"{tag}: HBM LoRA-A source must be CUDA, got {source_hbm.device}")
    if lora_a.device.type != "cuda":
        raise RuntimeError(f"{tag}: HBM LoRA-A weight must be CUDA, got {lora_a.device}")
    if source_hbm.dtype != torch.bfloat16 or lora_a.dtype != torch.bfloat16:
        raise RuntimeError(f"{tag}: HBM LoRA-A requires BF16 source and weight")
    if not source_hbm.is_contiguous() or not lora_a.is_contiguous():
        raise RuntimeError(f"{tag}: HBM LoRA-A requires contiguous source and weight")
    if source_hbm.dim() != 2 or lora_a.dim() != 3:
        raise RuntimeError(f"{tag}: HBM LoRA-A expects source [M,K] and weight [E,r,K]")
    if int(source_hbm.shape[1]) != int(lora_a.shape[2]):
        raise RuntimeError(f"{tag}: HBM LoRA-A shape mismatch")

def grouped_lora_a_forward_hbm(
    source_hbm: torch.Tensor,
    lora_a: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    metadata: GroupedLoRAMetadata | None,
    stats: AsymExecutionStats | None,
    tag: str,
) -> torch.Tensor:
    _check_hbm_lora_a_inputs(source_hbm, lora_a, tag)
    out = grouped_expert_lora(source_hbm, lora_a, offsets, experts, metadata=metadata)
    if stats is not None:
        stats.expact_lora_a_forward_grouped_calls += 1
        stats.expact_lora_a_forward_hbm_grouped_calls += 1
    return out
```

Do not add a HBM pair helper that calls `grouped_expert_lora_pair(packed,
packed, ...)`; that path materializes `[2M,H]`. Gate/up must concatenate only
LoRA-A weights.

2. Enable `gpu_hbm` in unsupported-reason checks.

```python
# AsymQwen3Experts._activation_offload_unsupported_reasons
mode = _expert_act_offload_lora_a_fwd_mode()
if mode == "gpu_hbm":
    try:
        _require_lora_grouped_mm()
    except RuntimeError as exc:
        reasons.append(str(exc))
```

Keep the existing full-kernel check because backward still needs CPU-right
LoRA-A gradient kernels:

```python
kernel_reason = require_expert_activation_offload_kernels(scope="full", check_only=True)
```

3. Rewrite gate/up forward LoRA-A branch inside
`_ActivationOffloadQwen3ExpertFunction.forward`.

```python
mode = _expert_act_offload_lora_a_fwd_mode()

with prof_range(layer._forward_range("activation_offload", "gate_up_lora_a")):
    if mode == "cpu_left":
        gate_low_rank, up_low_rank = grouped_lora_a_pair_forward_cpu_left(
            x_cpu.tensor,
            gate_lora_A,
            up_lora_A,
            offsets,
            experts,
            metadata=lora_metadata,
            stats=layer.stats,
            tag="gate_up",
        )
        gate_up_low_rank_owner = None
    elif mode == "gpu_hbm":
        gate_up_a = torch.cat((gate_lora_A, up_lora_A), dim=1).contiguous()
        try:
            gate_up_low_rank_owner = grouped_lora_a_forward_hbm(
                packed,
                gate_up_a,
                offsets,
                experts,
                metadata=lora_metadata,
                stats=layer.stats,
                tag="gate_up",
            )
            gate_low_rank, up_low_rank = gate_up_low_rank_owner.split(layer.lora_rank, dim=-1)
        finally:
            del gate_up_a
    else:
        raise AssertionError(f"unreachable LoRA-A forward mode {mode}")
```

Use `split`, not `chunk`, so the rank is explicit. The split outputs are views;
do not delete `gate_up_low_rank_owner` until after both low-rank views have
been consumed by LoRA-B and offloaded.

4. Keep gate/up LoRA-B unchanged, then offload the split views.

```python
with prof_range(layer._forward_range("activation_offload", "gate_up_lora_b")):
    gate_delta, up_delta = grouped_expert_lora_pair(
        gate_low_rank,
        up_low_rank,
        gate_lora_B,
        up_lora_B,
        offsets,
        experts,
        metadata=lora_metadata,
    )
    if layer.lora_scale != 1.0:
        gate_delta = gate_delta.mul(layer.lora_scale)
        up_delta = up_delta.mul(layer.lora_scale)
    gate.add_(gate_delta.to(dtype=gate.dtype))
    up.add_(up_delta.to(dtype=up.dtype))
    del gate_delta, up_delta

with prof_range(layer._forward_range("activation_offload", "save_gate_up_cpu")):
    gate_cpu = manager.offload(gate, "gate")
    up_cpu = manager.offload(up, "up")
    gate_low_rank_cpu = manager.offload(gate_low_rank, "S_gate")
    up_low_rank_cpu = manager.offload(up_low_rank, "S_up")
    del gate, up, gate_up, gate_low_rank, up_low_rank
    if gate_up_low_rank_owner is not None:
        del gate_up_low_rank_owner
```

Do not call `.contiguous()` on the split views before `manager.offload(...)`;
that would create extra `[M,R]` HBM clones. `ActivationOffloadManager.offload`
copies the view into a contiguous CPU owner.

5. Rewrite down LoRA-A scheduling for `gpu_hbm`.

```python
with prof_range(layer._forward_range("activation_offload", "activation_cpu")):
    act_cpu = _activation_offload_cpu_silu_mul(gate_cpu, up_cpu, manager, tag="act")

act_stage = None
if mode == "cpu_left":
    with prof_range(layer._forward_range("activation_offload", "down_lora_a")):
        down_low_rank = grouped_lora_a_forward_cpu_left(
            act_cpu.tensor,
            down_lora_A,
            offsets,
            experts,
            metadata=lora_metadata,
            stats=layer.stats,
            tag="down",
        )
    with prof_range(layer._forward_range("activation_offload", "down_lora_b")):
        down_delta = grouped_expert_lora(down_low_rank, down_lora_B, offsets, experts, metadata=lora_metadata)
        if layer.lora_scale != 1.0:
            down_delta = down_delta.mul(layer.lora_scale)
        down_low_rank_cpu = manager.offload(down_low_rank, "S_down")
        del down_low_rank
    with prof_range(layer._forward_range("activation_offload", "down_base_stage")):
        act_stage = manager.stage(act_cpu, tag="act_for_down_base")
        output = layer.down_base(act_stage, offsets, experts, dense_experts=True,
                                 profile_name=layer._profile_name("down", "base"))
        manager.release_stage(act_stage, drop_cache=True)
        act_stage = None
        output.add_(down_delta.to(dtype=output.dtype))
        del down_delta
else:
    try:
        with prof_range(layer._forward_range("activation_offload", "down_base_stage")):
            act_stage = manager.stage(act_cpu, tag="act_for_down_base")
        with prof_range(layer._forward_range("activation_offload", "down_lora_a")):
            down_low_rank = grouped_lora_a_forward_hbm(
                act_stage,
                down_lora_A,
                offsets,
                experts,
                metadata=lora_metadata,
                stats=layer.stats,
                tag="down",
            )
        with prof_range(layer._forward_range("activation_offload", "down_lora_b")):
            down_delta = grouped_expert_lora(down_low_rank, down_lora_B, offsets, experts, metadata=lora_metadata)
            if layer.lora_scale != 1.0:
                down_delta = down_delta.mul(layer.lora_scale)
            down_low_rank_cpu = manager.offload(down_low_rank, "S_down")
            del down_low_rank
        with prof_range(layer._forward_range("activation_offload", "down_base")):
            output = layer.down_base(act_stage, offsets, experts, dense_experts=True,
                                     profile_name=layer._profile_name("down", "base"))
            output.add_(down_delta.to(dtype=output.dtype))
            del down_delta
    finally:
        if act_stage is not None:
            manager.release_stage(act_stage, drop_cache=True)
```

This uses the already-required `act_stage` window for both down LoRA-A and
down base. It must not stage a second copy of `act_cpu`.

6. Record mode in activation-offload stats for easier profile inspection.

```python
snapshot = manager.snapshot()
snapshot["expert_lora_a_forward_mode"] = mode
layer._last_activation_offload_stats = snapshot
```

Apply the same key to the final backward snapshot if forward already stores a
preliminary snapshot and backward overwrites it.

7. Update tests.

```python
# tests/training/test_lf_qwen3_asym_backend.py
@pytest.mark.parametrize("mode", ["cpu_left", "gpu_hbm"])
def test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend(monkeypatch, mode):
    monkeypatch.setenv("ASYMM_EXPERT_ACT_OFFLOAD", "1")
    monkeypatch.setenv("ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD", mode)
    ...
    if mode == "cpu_left":
        assert asym_backend.stats.expact_lora_a_forward_cpu_left_grouped_calls == 3
        assert asym_backend.stats.expact_lora_a_forward_hbm_grouped_calls == 0
    else:
        assert asym_backend.stats.expact_lora_a_forward_cpu_left_grouped_calls == 0
        assert asym_backend.stats.expact_lora_a_forward_hbm_grouped_calls == 2
    assert asym_backend.stats.expact_lora_a_forward_grouped_calls == (
        asym_backend.stats.expact_lora_a_forward_cpu_left_grouped_calls
        + asym_backend.stats.expact_lora_a_forward_hbm_grouped_calls
    )
```

Add a non-SM100 unit test for invalid selector handling if the existing SM100
test is skipped on development machines.

Risks to watch:

- `gate_up_a = torch.cat(...)` creates a temporary `[E,2R,H]` HBM tensor. For
  Qwen3 `E=128,R=64,H=2048`, this is about `64 MiB`. If the e2e peak moves
  above the acceptance band, replace it with a native same-source pair kernel.
- The split `S_gate`/`S_up` tensors are views. Keep the owner alive until both
  views are consumed and offloaded.
- `grouped_expert_lora_pair` still materializes a `[2M,R]` input internally for
  LoRA-B, matching current behavior. Do not accidentally use it with identical
  `[M,H]` packed inputs.
- If `torch.nn.functional.grouped_mm` / `torch._grouped_mm` is unavailable,
  `gpu_hbm` must raise. Do not fall back to CPU-left.

Validation before Stage 3:

```bash
.venv/bin/python -m py_compile \
  asym_gemm/training/qwen3_moe.py \
  asym_gemm/training/exp_act_offload_lora.py \
  asym_gemm/training/frozen_linear.py \
  scripts/lf/run_lf_profiled_train.py

.venv/bin/python -m pytest -q \
  tests/training/test_cpu_left_lora.py \
  tests/test_lf_memory_breakdown.py::test_source_profile_reports_activation_offload_counters \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend \
  tests/lf/test_asym_cpu_adamw_args.py
```

Focused one-MoE sanity check, not acceptance:

```bash
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu_left \
PYTHONPATH=. .venv/bin/python scripts/testing/profile_qwen3_activation_offload.py \
  --tokens 16384 \
  --top-k 8 \
  --num-experts 128 \
  --hidden-dim 2048 \
  --intermediate-dim 768 \
  --rank 64 \
  --alpha 16 \
  --warmup 2 \
  --iters 5 \
  --profile-breakdown \
  --output-json /tmp/qwen3_expact_cpu_left_lora_a_one_moe.json

ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=gpu_hbm \
PYTHONPATH=. .venv/bin/python scripts/testing/profile_qwen3_activation_offload.py \
  --tokens 16384 \
  --top-k 8 \
  --num-experts 128 \
  --hidden-dim 2048 \
  --intermediate-dim 768 \
  --rank 64 \
  --alpha 16 \
  --warmup 2 \
  --iters 5 \
  --profile-breakdown \
  --output-json /tmp/qwen3_expact_gpu_hbm_lora_a_one_moe.json
```

Short e2e smoke before production A/B:

```bash
OUTPUT_ROOT=outputs/lf_ab_stage2_gpu_hbm_smoke \
OVERWRITE=true \
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=gpu_hbm \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYMM_EXP_ACT_POLICIES="none|true|true|true" \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
DATASET_MIN_TOKENS=4096 \
DATASET_OVERWRITE=true \
LORA_DROPOUT=0.00 \
PROFILERS=source \
PROFILE_MEMORY_ATTRIBUTION=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_SNAPSHOT=false \
WARMUP_STEPS=1 \
MAX_STEPS=2 \
RUN_POST=false \
scripts/lf/profile_lora_lf.sh --gpus 0
```

Acceptance:

- Unit parity passes for both modes.
- Smoke `source_profile.json` records `gpu_hbm`; postprocessed `profile.json`
  preserves the same config.
- Expert counters show CPU-left expert forward calls are `0` and HBM expert
  forward calls are positive for `gpu_hbm`.
- Postprocessed `profile.json` has
  `trainable_surface.surface == "attention+expert LoRA"`.
- `reference_fallback_count == 0`.
- Smoke peak allocated HBM is not near `58 GiB`.

## Stage 3: Production E2E A/B

Purpose: decide whether `gpu_hbm` is actually better on the production workload.
This is the acceptance gate. Do not accept based only on unit tests or one-MoE
profiling.

Files, functions, and classes:

- No code changes expected.
- `scripts/lf/profile_lora_lf.sh`
  - source-profile artifact paths and skip/completeness behavior
- `scripts/lf/postprocess_lf_profile_artifacts.py`
  - source-profile CSV/summary generation if summaries are needed
- `agent/status.md`
  - update only after an accepted result

Run A:

```bash
OUTPUT_ROOT=outputs/lf_ab_stage3_cpu_left \
OVERWRITE=true \
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu_left \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYMM_EXP_ACT_POLICIES="none|true|true|true" \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
DATASET_MIN_TOKENS=4096 \
DATASET_OVERWRITE=true \
LORA_DROPOUT=0.00 \
PROFILERS=source \
PROFILE_MEMORY_ATTRIBUTION=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_SNAPSHOT=false \
WARMUP_STEPS=5 \
MAX_STEPS=10 \
RUN_POST=false \
scripts/lf/profile_lora_lf.sh --gpus 0
```

Run B:

```bash
OUTPUT_ROOT=outputs/lf_ab_stage3_gpu_hbm \
OVERWRITE=true \
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=gpu_hbm \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYMM_EXP_ACT_POLICIES="none|true|true|true" \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
DATASET_MIN_TOKENS=4096 \
DATASET_OVERWRITE=true \
LORA_DROPOUT=0.00 \
PROFILERS=source \
PROFILE_MEMORY_ATTRIBUTION=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_SNAPSHOT=false \
WARMUP_STEPS=5 \
MAX_STEPS=10 \
RUN_POST=false \
scripts/lf/profile_lora_lf.sh --gpus 0
```

Post-run checks:

```bash
.venv/bin/python - <<'PY' outputs/lf_ab_stage3_cpu_left outputs/lf_ab_stage3_gpu_hbm
import json
import sys
from pathlib import Path

for root in map(Path, sys.argv[1:]):
    profiles = sorted(root.rglob("profile.json"))
    if len(profiles) != 1:
        raise SystemExit(f"{root}: expected one profile.json, found {len(profiles)}")
    profile = json.loads(profiles[0].read_text())
    cfg = profile["config"]
    lora = profile.get("lora", {})
    surface = profile.get("trainable_surface", {})
    rows = profile.get("activation_offload", {}).get("rows", [])
    exec_stats = [row.get("execution_stats", {}) for row in rows if isinstance(row, dict)]
    total_hbm = sum(int(row.get("expact_lora_a_forward_hbm_grouped_calls", 0) or 0) for row in exec_stats)
    total_cpu_exp = sum(int(row.get("expact_lora_a_forward_cpu_left_grouped_calls", 0) or 0) for row in exec_stats)
    fallback = sum(int(row.get("reference_fallback_count", 0) or 0) for row in exec_stats)
    print(root)
    print("  selector:", cfg.get("asymm_expert_act_offload_lora_a_fwd"))
    print("  peak allocated GiB:", profile["memory"]["peak_allocated_hbm_bytes"] / 1024**3)
    print("  peak reserved GiB:", profile["memory"]["peak_reserved_hbm_bytes"] / 1024**3)
    print("  avg step s:", profile["step"]["total_milliseconds"] / 1000.0)
    print("  trainable surface:", surface.get("surface"))
    print("  trainable params:", lora.get("trainable_parameters"))
    print("  expert CPU-left/HBM/fallback:", total_cpu_exp, total_hbm, fallback)
PY
```

Acceptance for B:

- `config.asymm_expert_act_offload_lora_a_fwd == "gpu_hbm"`.
- Peak allocated HBM `<=36 GiB` preferred; reject `>40 GiB` or near `58 GiB`.
- Peak reserved HBM has no material increase over A.
- Avg step improves by at least `20%`.
- Avg forward materially improves.
- Avg backward does not increase enough to erase forward savings.
- `trainable_surface.surface == "attention+expert LoRA"`.
- Trainable-surface parameter counts match A.
- `reference_fallback_count == 0`.
- Loss max/last/train remains stable relative to A.
- Expert counters:
  - A: `expact_lora_a_forward_cpu_left_grouped_calls > 0`, HBM calls `0`.
  - B: `expact_lora_a_forward_hbm_grouped_calls > 0`, expert CPU-left calls `0`.

Risks to watch:

- Source timings are host wall-clock ranges without per-range CUDA
  synchronization. Use the same profiler, same GPU, same batch, same sequence
  length, same dataset, and same warmup/measure steps for A and B.
- If B improves forward but worsens reserved memory substantially, inspect
  whether `gate_up_a` or split low-rank owner lifetime crosses the peak.

## Stage 4: Promote or Roll Back

Purpose: make a default decision only after Stage 3 passes.

Files, functions, and classes if `gpu_hbm` wins:

- `asym_gemm/training/qwen3_moe.py`
  - `_expert_act_offload_lora_a_fwd_mode`
- `scripts/lf/run_lf_lora_sft.sh`
  - default `ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD`
- `scripts/lf/profile_lora_lf.sh`
  - default `ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD`
- tests:
  - `tests/lf/test_asym_cpu_adamw_args.py`
  - `tests/training/test_lf_qwen3_asym_backend.py`
- `agent/status.md`

Implementation if accepted:

```python
# qwen3_moe.py
raw = os.environ.get(EXPERT_ACT_OFFLOAD_LORA_A_FWD_ENV, "gpu_hbm")
```

```bash
# run_lf_lora_sft.sh and profile_lora_lf.sh
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=${ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD:-gpu_hbm}
```

Keep `cpu_left` as an override for one release:

```bash
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu_left scripts/lf/profile_lora_lf.sh --gpus 0
```

If B is faster but above the memory band:

- Keep default `cpu_left`.
- Implement a native same-source gate/up LoRA-A grouped kernel that avoids the
  `[E,2R,H]` concat or schedules the concat outside the peak.
- Do not use `grouped_expert_lora_pair(packed, packed, ...)`, because that
  materializes `[2M,H]`.

If B does not improve latency:

- Keep default `cpu_left`.
- Next work should target backward LoRA-A grad and CPU SiLU-mul fusion.

Validation:

```bash
.venv/bin/python -m pytest -q \
  tests/lf/test_asym_cpu_adamw_args.py \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend
```

Update `agent/status.md` with:

```text
backend spec
policy tuple
forward LoRA-A mode
peak allocated HBM
peak reserved HBM
avg step
avg forward
avg backward
forward-end HBM
saved CPU peak
AsymGEMM fwd/dx
expert CPU-left LoRA-A calls
expert HBM LoRA-A calls
generic CPU-left LoRA-A calls
loss max/last/train
```
