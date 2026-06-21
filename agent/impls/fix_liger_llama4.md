# Liger Loss-Only for Llama4

Goal: make `ENABLE_LIGER_KERNEL=true` enable only Liger fused linear cross entropy for Llama4, with no RoPE, norm, SwiGLU, or standalone CE patches. The same knob must work for regular LF/DeepSpeed runs and for AsymGEMM runs where `lm_head` can be CPU-resident or wrapped by `AsymFrozenLinear`.

Non-goals:
- Do not add a second runtime knob. The only runtime knob is `ENABLE_LIGER_KERNEL`, represented in sweeps as `BACKEND_SPECS=...|ligerloss0` or `...|ligerloss1`.
- Do not enable non-loss Liger kernels.
- Do not patch unsupported model types.
- Do not modify `third_party/Liger-Kernel` unless local validation proves the repo-local Llama4 loss path is incorrect.

Current local facts to keep the implementation grounded:
- `scripts/lf/profile_lora_lf.sh` derives `SFT_ROOT`, `ROOT`, `ASYM_DIR`, `ENV_DIR`, and `ENV_PYTHON`; docs and commands should use those dynamic paths, not a hardcoded virtualenv.
- `scripts/lf/profile_lora_lf.sh` already parses `BACKEND_SPECS='backend|recompute[|ligerloss0/1]'`, exports `ENABLE_LIGER_KERNEL`, records `ASYM_GEMM_LF_CONFIG_LIGER_LOSS`, writes `liger_loss` into `jobs.tsv`, and includes `__ligerloss0/1__` in run paths.
- `scripts/lf/profile_lora_lf.sh` now supports six-field activation tuples: `policy|expert_act|attn_act|layer_act[|layer_gc[|sdpa_recompute]]`. Run paths include `__sdparecomp0/1__`, and `ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_SDPA_RECOMPUTE` is exported.
- The Liger validation runs must pin one six-field `ASYMM_EXP_ACT_POLICIES` tuple. The default script sweep varies activation and SDPA recompute axes, so it is not a clean Liger off/on comparison.
- `scripts/plotting/plot_activation_recompute_sweep.py` and `scripts/plotting/plot_lf_memory_breakdown.py` parse `sdparecomp`, but the combined plot grouping/labels are not fully split by SDPA everywhere. Pinning the SDPA axis is required unless Stage 0 expands plot grouping.
- `../Liger-Kernel/src/liger_kernel/transformers/monkey_patch.py` maps both `llama4_text` and `llama4` to `apply_liger_kernel_to_llama4(...)`.
- `apply_liger_kernel_to_llama4(...)` accepts `rope`, `cross_entropy`, `fused_linear_cross_entropy`, `rms_norm`, `swiglu`, `model`, and `layer_norm`.
- The repo-local Liger Llama4 fused CE forward patches `transformers.models.llama4.modeling_llama4.Llama4ForCausalLM.forward` and calls `LigerForCausalLMLoss(hidden_states=..., lm_head_weight=self.lm_head.weight, ...)`.
- `Llama4TextConfig.model_type == "llama4_text"` and uses `Llama4ForCausalLM`.
- `Llama4Config.model_type == "llama4"` and uses `Llama4ForConditionalGeneration`, which owns `language_model = Llama4ForCausalLM(config.text_config)`.
- `Llama4ForConditionalGeneration.forward` calls `self.language_model(...)` without labels, materializes logits, and then computes CE at the top level. Therefore a class-level patch of `Llama4ForCausalLM.forward` is not enough to prove fused CE for Scout/Maverick-style `model_type="llama4"` runs.
- `../LlamaFactory/src/llamafactory/model/model_utils/liger_kernel.py` currently whitelists only `qwen3_moe`.
- `asym_gemm/integrations/liger_loss.py` currently has only the Qwen3-MoE Asym bridge.
- `../LlamaFactory/src/llamafactory/model/loader.py` applies the LF Liger hook before model construction, then loads the model, then calls `init_adapter(...)`.
- `../LlamaFactory/src/llamafactory/model/loader.py` selects `AutoModelForImageTextToText` when the config is in that mapping; real Llama4 Scout/Maverick configs should be treated as likely `Llama4ForConditionalGeneration` until the run proves otherwise.
- `scripts/lf/run_lf_lora_sft.sh` puts `${ASYM_DIR}` and `${LF_DIR}/src` on `PYTHONPATH` for normal and DeepSpeed runs, so the post-load bridge can live in `asym_gemm.integrations.liger_loss` and still be importable during zero3 profiling.

Install precondition:

```bash
SFT_ROOT=${SFT_ROOT:-/home/kevinni/AsymGEMM-SFT}
ASYM_DIR=${ASYM_DIR:-${SFT_ROOT}/third_party/AsymGEMM}
ENV_PYTHON=${ENV_PYTHON:-${ASYM_DIR}/.venv/bin/python}

"${ENV_PYTHON}" -m pip install -e "${SFT_ROOT}/third_party/Liger-Kernel"
"${ENV_PYTHON}" - <<'PY'
import inspect
from liger_kernel.transformers import apply_liger_kernel_to_llama4
from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN

sig = inspect.signature(apply_liger_kernel_to_llama4)
required = {"fused_linear_cross_entropy", "rope", "cross_entropy", "rms_norm", "swiglu", "layer_norm", "model"}
missing = required - set(sig.parameters)
assert not missing, missing
assert MODEL_TYPE_TO_APPLY_LIGER_FN["llama4"] is apply_liger_kernel_to_llama4
assert MODEL_TYPE_TO_APPLY_LIGER_FN["llama4_text"] is apply_liger_kernel_to_llama4
print(sig)
PY
```

Accept this precondition only if it uses the same `ENV_PYTHON` that `scripts/lf/profile_lora_lf.sh` will use.

## Stage 0 - Common Script and Interface Preflight

Scope:
- `scripts/lf/profile_lora_lf.sh`
  - `usage`
  - `parse_exp_act_policy_tuple`
  - `sdparecomp_tag`
  - `append_backend_spec`
  - `liger_loss_label`
  - `job_root_path`
  - `job_profile_complete`
  - `append_liger_loss_filters`
  - `append_current_activation_axis_filters`
  - `run_job`
  - `plot_single_run`
  - `plot_running_combined`
  - `plot_memory_single_run`
  - `plot_memory_running_combined`
  - `write_config_artifact_readme`
  - `write_precision_artifact_readme`
- `scripts/lf/run_lf_lora_sft.sh`
  - `ENABLE_LIGER_KERNEL` normalization
  - `--enable_liger_kernel` forwarding
  - `ASYMM_ATTN_SDPA_RECOMPUTE` forwarding
  - `ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_SDPA_RECOMPUTE`
  - `NUMACTL_MEMBIND`, `NUMACTL_CPUNODEBIND`, `NUMACTL_MODE`
- `scripts/lf/run_lf_profiled_train.py`
  - `_env_config`
  - `_config_from_args`
  - emitted top-level `source_profile["config"]`
  - emitted top-level `source_profile["asym_liger_lm_head_bridge"]`
- Only if unpinned SDPA sweeps are needed for acceptance:
  - `scripts/plotting/plot_activation_recompute_sweep.py`
    - `collect_rows` dedupe key
    - sort key in `collect_rows`
    - `combined_group_key`
    - `combined_threshold_group_key`
    - `varied_fields`
    - `varied_threshold_fields`
    - `combined_label`
    - `combined_threshold_label`
    - grouped plot key destructuring and filename suffixes around the per-group writers
  - `scripts/plotting/plot_lf_memory_breakdown.py`
    - `_group_label`
    - optional `--sdparecomp` filter only if accepting unpinned SDPA sweeps
  - `scripts/plotting/plot_lf_interconnect_ctc.py`
    - `_parse_job_dir_parts`
    - `_infer_metadata`
    - labels/filters only if accepting unpinned SDPA sweeps

Intended code changes:
- No LF, AsymGEMM, or Liger runtime changes in this stage.
- Keep exactly one runtime env var for Liger: `ENABLE_LIGER_KERNEL=true|false`.
- Keep exactly one sweep axis spelling: `ligerloss0` and `ligerloss1`.
- Keep `BACKEND_SPECS` format as `backend|recomp|ligerloss0/1`.
- If the current script help or artifact READMEs omit `sdpa_recompute`, update only those text strings so they match the current six-field tuple and `__sdparecomp0/1__` path axis.
- Do not touch `scripts/lf/profile3.sh`; it is out of scope for this Llama4 Liger-loss change.

Validation before moving on:

```bash
DRY_RUN=true \
MODEL_SPECS='meta-llama/Llama-4-Scout-17B-16E|1' \
BACKEND_SPECS='zero3_offload|recomp|ligerloss0,zero3_offload|recomp|ligerloss1,asym_cpuadamwds|norecomp|ligerloss0,asym_cpuadamwds|norecomp|ligerloss1' \
ASYMM_EXP_ACT_POLICIES='none|true|false|false|false|false' \
WORKLOADS='8192|2|1' \
PROFILERS=both GPU_POOL=3 \
NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 NUMACTL_MODE=membind \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Pass conditions:
- Dry-run commands contain `ENABLE_LIGER_KERNEL=true` only for `ligerloss1`.
- Dry-run commands contain `ENABLE_LIGER_KERNEL=false` only for `ligerloss0`.
- Dry-run commands pass `ASYMM_ATTN_SDPA_RECOMPUTE=false` and `ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_SDPA_RECOMPUTE=false`.
- Run paths and run IDs contain both `__sdparecomp0__` and `__ligerloss0/1__`.
- `jobs.tsv` header contains `liger_loss`.
- `PROFILERS=both` shows one Nsight execution path plus a sibling materialized `source` artifact path.
- `NUMACTL_MEMBIND=0,1` and `NUMACTL_CPUNODEBIND=0,1` are forwarded unchanged.

Risks to watch:
- If validation uses the default `ASYMM_EXP_ACT_POLICIES`, the result is not a clean Liger comparison because activation offload and SDPA recompute vary too.
- If unpinned SDPA sweeps become required, update the plot grouping functions listed above before accepting performance plots.

## Stage 1 - Enable LF Loss-Only Gate for Llama4 Model Types

Scope:
- `../LlamaFactory/src/llamafactory/model/model_utils/liger_kernel.py`
  - `_LOSS_ONLY_SUPPORTED_MODEL_TYPES`
  - `_resolve_liger_apply_fn`
  - `_build_liger_loss_only_kwargs` only if tests prove it mishandles `layer_norm`
  - `apply_liger_kernel`
- `tests/lf/test_liger_loss_only_qwen3_moe.py`
  - keep Qwen3-MoE coverage
  - add `llama4_text` and `llama4` coverage
  - update unsupported-model tests so they no longer use `llama4` as unsupported

Intended code changes:

```python
# ../LlamaFactory/src/llamafactory/model/model_utils/liger_kernel.py

_LOSS_ONLY_SUPPORTED_MODEL_TYPES = {"qwen3_moe", "llama4_text", "llama4"}


def _resolve_liger_apply_fn(model_type: str | None) -> Callable[..., None] | None:
    if model_type == "qwen3_moe":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe
        return apply_liger_kernel_to_qwen3_moe

    if model_type in {"llama4_text", "llama4"}:
        from liger_kernel.transformers import apply_liger_kernel_to_llama4
        return apply_liger_kernel_to_llama4

    return None
```

`_build_liger_loss_only_kwargs` should remain generic:

```python
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
```

For Llama4 this must produce:

```python
{
    "rope": False,
    "cross_entropy": False,
    "fused_linear_cross_entropy": True,
    "rms_norm": False,
    "swiglu": False,
    "layer_norm": False,
}
```

Add tests:

```python
@pytest.mark.parametrize("model_type", ["llama4_text", "llama4"])
def test_apply_liger_kernel_uses_loss_only_for_llama4_model_types(monkeypatch, model_type):
    from llamafactory.model.model_utils import liger_kernel

    calls = []

    def fake_apply(
        rope=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rms_norm=True,
        swiglu=True,
        model=None,
        layer_norm=True,
    ):
        calls.append({
            "rope": rope,
            "cross_entropy": cross_entropy,
            "fused_linear_cross_entropy": fused_linear_cross_entropy,
            "rms_norm": rms_norm,
            "swiglu": swiglu,
            "layer_norm": layer_norm,
        })

    monkeypatch.setattr(liger_kernel, "_resolve_liger_apply_fn", lambda mt: fake_apply)

    liger_kernel.apply_liger_kernel(
        SimpleNamespace(model_type=model_type),
        SimpleNamespace(enable_liger_kernel=True),
        is_trainable=True,
        require_logits=False,
    )

    assert calls == [{
        "rope": False,
        "cross_entropy": False,
        "fused_linear_cross_entropy": True,
        "rms_norm": False,
        "swiglu": False,
        "layer_norm": False,
    }]
```

Add an explicit source check:

```python
def test_local_liger_llama4_signature_and_dispatch_are_supported():
    import inspect
    from liger_kernel.transformers import apply_liger_kernel_to_llama4
    from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN

    sig = inspect.signature(apply_liger_kernel_to_llama4)
    assert {"rope", "cross_entropy", "fused_linear_cross_entropy", "rms_norm", "swiglu", "layer_norm", "model"} <= set(sig.parameters)
    assert MODEL_TYPE_TO_APPLY_LIGER_FN["llama4_text"] is apply_liger_kernel_to_llama4
    assert MODEL_TYPE_TO_APPLY_LIGER_FN["llama4"] is apply_liger_kernel_to_llama4
```

Validation before moving on:

```bash
SFT_ROOT=${SFT_ROOT:-/home/kevinni/AsymGEMM-SFT}
ASYM_DIR=${ASYM_DIR:-${SFT_ROOT}/third_party/AsymGEMM}
ENV_PYTHON=${ENV_PYTHON:-${ASYM_DIR}/.venv/bin/python}

"${ENV_PYTHON}" -m pytest -q tests/lf/test_liger_loss_only_qwen3_moe.py
"${ENV_PYTHON}" - <<'PY'
from transformers.models.llama4.configuration_llama4 import Llama4Config, Llama4TextConfig
assert Llama4TextConfig.model_type == "llama4_text"
assert Llama4Config.model_type == "llama4"
print("llama4 config model types verified")
PY
```

Pass conditions:
- Existing Qwen3-MoE loss-only tests still pass.
- `llama4_text` and `llama4` both call Liger with only `fused_linear_cross_entropy=True`.
- Unsupported model types still skip cleanly.

Risks to watch:
- This stage only enables the LF pre-load class patch. It is enough for `Llama4ForCausalLM` / `model_type="llama4_text"` but is not enough to accept `Llama4ForConditionalGeneration` / `model_type="llama4"` memory results.

## Stage 2 - Add Post-Load Llama4 Loss Bridge for Normal and Asym Runs

Scope:
- `asym_gemm/integrations/liger_loss.py`
  - `_candidate_language_models`
  - `_base_causal_lm_model`
  - `_root_model`
  - `_is_llama4_conditional_generation`
  - `_resolve_liger_lm_head_weight`
  - `_lm_head_weight_source`
  - `_validate_liger_lm_head`
  - `_mark_liger_bridge_installed`
  - `_bridge_metadata_target`
  - `_make_liger_shift_labels`
  - add `asym_llama4_causal_lm_lce_forward`
  - add `asym_llama4_conditional_lce_forward`
  - add `install_asym_liger_llama4_loss_bridge`
  - add `install_asym_liger_loss_bridge`
  - keep `install_asym_liger_qwen3_moe_loss_bridge` as a compatibility wrapper
  - update `asym_liger_lm_head_bridge_metadata`
  - update `__all__`
- `../LlamaFactory/src/llamafactory/model/adapter.py`
  - import `install_asym_liger_loss_bridge` only inside the branches that need it
  - install the generic post-load bridge after Asym wrapping
  - install the generic post-load bridge after normal PEFT wrapping for Llama4 conditional-generation runs
- `tests/lf/test_asym_liger_lm_head_bridge.py`
  - keep Qwen3-MoE coverage
  - add `llama4_text` causal-LM coverage
  - add `llama4` conditional-generation coverage
  - cover normal-parameter and `AsymFrozenLinear` `lm_head` sources

Intended code changes:

1. Resolve PEFT, top-level conditional wrappers, and nested language models.

```python
def _root_model(model: nn.Module) -> nn.Module:
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        try:
            base = get_base_model()
        except Exception:
            base = None
        if isinstance(base, nn.Module):
            return base
    return model


def _candidate_language_models(model: nn.Module) -> list[nn.Module]:
    root = _root_model(model)
    candidates = [model, root]

    for candidate in list(candidates):
        language_model = getattr(candidate, "language_model", None)
        if isinstance(language_model, nn.Module):
            candidates.append(language_model)

        inner = getattr(candidate, "model", None)
        if isinstance(inner, nn.Module):
            candidates.append(inner)
            inner_language_model = getattr(inner, "language_model", None)
            if isinstance(inner_language_model, nn.Module):
                candidates.append(inner_language_model)

    deduped = []
    seen = set()
    for candidate in candidates:
        ident = id(candidate)
        if isinstance(candidate, nn.Module) and ident not in seen:
            deduped.append(candidate)
            seen.add(ident)
    return deduped


def _base_causal_lm_model(model: nn.Module) -> nn.Module:
    for candidate in _candidate_language_models(model):
        if hasattr(candidate, "lm_head") and hasattr(candidate, "model"):
            return candidate
    return model


def _is_llama4_conditional_generation(model: nn.Module) -> bool:
    root = _root_model(model)
    return (
        getattr(getattr(root, "config", None), "model_type", None) == "llama4"
        and isinstance(getattr(root, "language_model", None), nn.Module)
        and hasattr(root.language_model, "lm_head")
        and hasattr(root.language_model, "model")
    )
```

2. Share `lm_head` validation for Qwen3-MoE, Llama4 text, and Llama4 conditional.

```python
def _validate_liger_lm_head(lm_head: nn.Module | None, *, model_label: str, strict: bool):
    if lm_head is None:
        if strict:
            raise RuntimeError(f"{model_label} model has no lm_head.")
        return None
    if not isinstance(lm_head, nn.Module):
        raise RuntimeError(f"{model_label} lm_head is not a torch module: {type(lm_head).__name__}.")

    weight_source = _lm_head_weight_source(lm_head)
    if weight_source == "unavailable":
        if strict:
            raise RuntimeError("lm_head is not compatible with Liger fused CE weight resolution.")
        return None

    if getattr(lm_head, "bias", None) is not None or getattr(lm_head, "bias_cpu", None) is not None:
        raise RuntimeError("Liger loss bridge currently requires a bias-free lm_head.")

    if any(param.requires_grad for param in lm_head.parameters(recurse=True)):
        raise RuntimeError("Liger loss bridge supports frozen lm_head only.")

    return lm_head, weight_source
```

3. Preserve bridge metadata for source profiling.

```python
def _mark_liger_bridge_installed(target_model, lm_head, weight_source, model_type, bridge_kind):
    target_model._asym_liger_lm_head_bridge_enabled = True
    target_model._asym_liger_lm_head_weight_source = weight_source
    target_model._asym_liger_lm_head_type = type(lm_head).__name__
    target_model._asym_liger_lm_head_staged_bytes = int(getattr(lm_head, "cpu_resident_base_weight_bytes", 0) or 0)
    target_model._asym_liger_model_type = model_type
    target_model._asym_liger_bridge_kind = bridge_kind


def _bridge_metadata_target(model: nn.Module) -> nn.Module:
    # Conditional Llama4 is marked on the top-level wrapper, while causal-LM
    # bridges are marked on the causal LM. Check both before falling back.
    candidates = [model, _root_model(model), *_candidate_language_models(model)]
    seen = set()
    for candidate in candidates:
        ident = id(candidate)
        if not isinstance(candidate, nn.Module) or ident in seen:
            continue
        seen.add(ident)
        if getattr(candidate, "_asym_liger_lm_head_bridge_enabled", False):
            return candidate
    return _base_causal_lm_model(model)
```

`asym_liger_lm_head_bridge_metadata(...)` must call `_bridge_metadata_target(...)`, not only `_base_causal_lm_model(...)`. Otherwise conditional Llama4 bridges marked on the top-level wrapper will look disabled in `source_profile.json`. It should return at least `enabled`, `weight_source`, `staged_bytes`, `lm_head_type`, `model_type`, and `bridge_kind`.

4. Add a Llama4 causal-LM bridge for `llama4_text` and nested causal LMs.

Use `../Liger-Kernel/src/liger_kernel/transformers/model/llama4.py::lce_forward` as the template. The only intended behavior change is replacing `self.lm_head.weight` with `_resolve_liger_lm_head_weight(self.lm_head, kept_hidden_states)`.

Required imports:

```python
from liger_kernel.transformers.model.output_classes import LigerCausalLMOutputWithPast
from transformers.models.llama4.modeling_llama4 import Llama4CausalLMOutputWithPast
```

```python
def asym_llama4_causal_lm_lce_forward(self, ..., labels=None, logits_to_keep=0, **kwargs):
    outputs = self.model(..., return_dict=True, ...)
    hidden_states = outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    shift_labels = kwargs.pop("shift_labels", None)
    logits = None
    loss = None
    token_accuracy = None
    predicted_tokens = None

    if self.training and (labels is not None or shift_labels is not None):
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
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, shift_labels=shift_labels, vocab_size=self.config.vocab_size, **kwargs)

    return LigerCausalLMOutputWithPast(...)
```

5. Add a top-level conditional-generation bridge for real `model_type="llama4"` runs.

This bridge must mirror `transformers.models.llama4.modeling_llama4.Llama4ForConditionalGeneration.forward` through embedding/image merging, but it must not call `self.language_model(...)` in the loss path because that materializes logits. Instead, call `self.language_model.model(...)` to get hidden states, then call `LigerForCausalLMLoss` with `self.language_model.lm_head`.

```python
def _select_sequence_positions(tensor: torch.Tensor, slice_indices: slice | torch.Tensor) -> torch.Tensor:
    if isinstance(slice_indices, slice):
        return tensor[:, slice_indices].contiguous()
    if isinstance(slice_indices, torch.Tensor):
        return tensor.index_select(1, slice_indices.to(device=tensor.device, dtype=torch.long)).contiguous()
    return tensor[:, slice_indices].contiguous()


def _make_liger_shift_labels(labels, attention_mask, *, slice_indices: slice | torch.Tensor, ignore_index: int = -100):
    if labels is None:
        return None

    shifted = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
    if attention_mask is not None:
        active = torch.nn.functional.pad(attention_mask[..., 1:], (0, 1), value=0).to(dtype=torch.bool)
        shifted = shifted.masked_fill(~active.to(device=shifted.device), ignore_index)

    return _select_sequence_positions(shifted, slice_indices)


def _coerce_existing_shift_labels(
    shift_labels: torch.Tensor,
    *,
    slice_indices: slice | torch.Tensor,
    full_seq_len: int,
    kept_seq_len: int,
) -> torch.Tensor:
    if shift_labels.shape[1] == full_seq_len:
        return _select_sequence_positions(shift_labels, slice_indices)
    if shift_labels.shape[1] == kept_seq_len:
        return shift_labels.contiguous()
    raise ValueError(
        f"shift_labels length {shift_labels.shape[1]} does not match full seq {full_seq_len} or kept seq {kept_seq_len}"
    )


def asym_llama4_conditional_lce_forward(
    self,
    input_ids=None,
    pixel_values=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    vision_feature_select_strategy=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    logits_to_keep=0,
    **kwargs,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if pixel_values is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both pixel_values and inputs_embeds at the same time.")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_features = None
    if pixel_values is not None:
        image_features = self.get_image_features(
            pixel_values=pixel_values,
            vision_feature_select_strategy=vision_feature_select_strategy,
            return_dict=True,
        ).last_hidden_state
        vision_flat = image_features.view(-1, image_features.size(-1))
        projected = self.multi_modal_projector(vision_flat).to(inputs_embeds.device, inputs_embeds.dtype)
        special_image_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=projected)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, projected)

    causal_lm = self.language_model
    outputs = causal_lm.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        **kwargs,
    )

    hidden_states = outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    shift_labels = kwargs.pop("shift_labels", None)
    if shift_labels is None:
        shift_labels = _make_liger_shift_labels(labels, attention_mask, slice_indices=slice_indices)
    else:
        shift_labels = _coerce_existing_shift_labels(
            shift_labels,
            slice_indices=slice_indices,
            full_seq_len=hidden_states.shape[1],
            kept_seq_len=kept_hidden_states.shape[1],
        )

    logits = None
    loss = None
    token_accuracy = None
    predicted_tokens = None
    skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        lm_head_weight = _resolve_liger_lm_head_weight(causal_lm.lm_head, kept_hidden_states)
        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=lm_head_weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.text_config.hidden_size,
            **kwargs,
        )
        loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)
    else:
        logits = causal_lm.lm_head(kept_hidden_states)
        if labels is not None or shift_labels is not None:
            loss = causal_lm.loss_function(
                logits=logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return Llama4CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
    )
```

This keeps kernel launch efficiency acceptable: there is still one text-backbone forward and one fused linear CE loss path. Do not split expert work into Python loops, do not create per-expert GEMMs, and do not materialize full `[batch, seq, vocab]` logits in the training loss path.

6. Add installers.

```python
def install_asym_liger_llama4_loss_bridge(model: nn.Module, *, strict: bool = True) -> bool:
    root = _root_model(model)

    if _is_llama4_conditional_generation(root):
        causal_lm = root.language_model
        validated = _validate_liger_lm_head(getattr(causal_lm, "lm_head", None), model_label="Llama4", strict=strict)
        if validated is None:
            return False
        lm_head, weight_source = validated
        root.forward = MethodType(asym_llama4_conditional_lce_forward, root)
        _mark_liger_bridge_installed(root, lm_head, weight_source, "llama4", "conditional_generation")
        return True

    target = _base_causal_lm_model(root)
    model_type = getattr(getattr(target, "config", None), "model_type", None)
    if model_type not in {"llama4_text", "llama4"}:
        if strict:
            raise ValueError("Llama4 Liger loss bridge only supports llama4_text or llama4.")
        return False

    validated = _validate_liger_lm_head(getattr(target, "lm_head", None), model_label="Llama4", strict=strict)
    if validated is None:
        return False
    lm_head, weight_source = validated
    target.forward = MethodType(asym_llama4_causal_lm_lce_forward, target)
    _mark_liger_bridge_installed(target, lm_head, weight_source, model_type, "causal_lm")
    return True


def install_asym_liger_loss_bridge(model: nn.Module, *, strict: bool = True) -> bool:
    root = _root_model(model)
    root_type = getattr(getattr(root, "config", None), "model_type", None)
    causal = _base_causal_lm_model(root)
    causal_type = getattr(getattr(causal, "config", None), "model_type", None)

    if root_type == "llama4" or causal_type in {"llama4", "llama4_text"}:
        return install_asym_liger_llama4_loss_bridge(model, strict=strict)
    if causal_type == "qwen3_moe":
        return install_asym_liger_qwen3_moe_loss_bridge(model, strict=strict)
    return False
```

The installer may patch normal-parameter `lm_head` for Llama4 conditional-generation runs because the class-level Liger CausalLM patch is not sufficient there. For Qwen3-MoE and `llama4_text`, normal non-Asym runs can rely on the class-level Liger patch; the post-load bridge is primarily needed when `lm_head` is Asym-wrapped.

7. Wire LF adapter after wrapping.

```python
# ../LlamaFactory/src/llamafactory/model/adapter.py

if model_args.use_asym_gemm:
    ...
    model, report = adapt_lf_asym_peft_lora(...)
    if model_args.enable_liger_kernel:
        from asym_gemm.integrations.liger_loss import install_asym_liger_loss_bridge

        bridge_installed = install_asym_liger_loss_bridge(
            model,
            strict=bool(model_args.asym_strict and selection.lm_head),
        )
        if bridge_installed:
            logger.info_rank0("Asym Liger loss bridge has been installed.")
    logger.info_rank0(report.to_log_string())
    return model

# In the normal LoRA/OFT branch after get_peft_model(...), before the final return.
# Only the top-level Llama4 conditional-generation path needs this post-load
# normal-parameter bridge; Qwen3-MoE and llama4_text can use the class patch.
if model_args.enable_liger_kernel and getattr(config, "model_type", None) == "llama4":
    from asym_gemm.integrations.liger_loss import install_asym_liger_loss_bridge

    bridge_installed = install_asym_liger_loss_bridge(model, strict=False)
    if bridge_installed:
        logger.info_rank0("Post-load Liger loss bridge has been installed.")
```

Patch order:
- LF pre-load Liger hook applies the class-level loss-only patch.
- LF loads the model.
- PEFT and/or Asym wrapping happens.
- The post-load bridge patches only the resolved model instance.
- For Asym runs, the bridge stages `lm_head` through `AsymFrozenLinear.asym_liger_lm_head_weight(...)`.
- For normal `Llama4ForConditionalGeneration`, the bridge avoids the top-level logits materialization that the class-level CausalLM patch cannot avoid.

Validation before moving on:

```bash
SFT_ROOT=${SFT_ROOT:-/home/kevinni/AsymGEMM-SFT}
ASYM_DIR=${ASYM_DIR:-${SFT_ROOT}/third_party/AsymGEMM}
ENV_PYTHON=${ENV_PYTHON:-${ASYM_DIR}/.venv/bin/python}

"${ENV_PYTHON}" -m pytest -q tests/lf/test_asym_liger_lm_head_bridge.py
"${ENV_PYTHON}" -m pytest -q tests/lf/test_liger_loss_only_qwen3_moe.py
```

New bridge tests must prove:
- The Llama4 bridge patches only the model instance, not the class.
- `llama4_text` causal-LM bridge passes a staged weight to `LigerForCausalLMLoss`.
- `llama4` conditional-generation bridge does not call `language_model.lm_head(...)` in the training loss path.
- `llama4` conditional-generation bridge passes `normal_parameter` weight for normal/DeepSpeed and `asym_host_staged` weight for Asym.
- `_make_liger_shift_labels` preserves `-100` labels, masks out padded positions from `attention_mask`, and selects the same sequence positions as `logits_to_keep` for both integer and tensor forms.
- `_coerce_existing_shift_labels` accepts already-kept `shift_labels`, slices full-length `shift_labels`, and rejects any ambiguous shape.
- `asym_liger_lm_head_bridge_metadata(...)` reports an enabled bridge when the marker is on the top-level conditional Llama4 wrapper and when it is on a nested causal LM.
- Conditional Llama4 bridge supports `return_dict=False` without materializing logits in the training loss path.
- Trainable or biased `lm_head` is rejected.
- Unsupported model types return `False` when `strict=False`.
- `asym_liger_lm_head_bridge_metadata(...)` reports `enabled`, `model_type`, `bridge_kind`, `weight_source`, `lm_head_type`, and `staged_bytes`.

Risks to watch:
- If a real LF Llama4 run loads a different wrapper shape than `Llama4ForConditionalGeneration(language_model=Llama4ForCausalLM)`, update `_candidate_language_models` and add a fixture before accepting.
- The conditional bridge must remain aligned with the repo-local Transformers `Llama4ForConditionalGeneration.forward`. If Transformers changes that forward, diff the function and update the bridge before profiling.
- `attention_mask` handling must be validated numerically against the unfused HF loss on a small CPU/GPU fixture before running large E2E comparisons.

## Stage 3 - E2E Llama4 Validation on Real Workload

Scope:
- No implementation files should change in this stage unless Stage 0 or Stage 2 validation fails.
- Use:
  - `scripts/lf/profile_lora_lf.sh`
  - `scripts/lf/compare_liger_loss_profiles.py`
  - generated `source_profile.json`
  - generated `memory_breakdown_summary.json`
  - generated timing and memory plots

Run zero3 first:

```bash
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/profiling_liger_llama4_zero3 \
RUN_NAME=llama4_liger_zero3 \
MODEL_SPECS='meta-llama/Llama-4-Scout-17B-16E|1' \
BACKEND_SPECS='zero3_offload|recomp|ligerloss0,zero3_offload|recomp|ligerloss1' \
ASYMM_EXP_ACT_POLICIES='none|true|false|false|false|false' \
WORKLOADS='8192|2|1' \
PROFILERS=both GPU_POOL=3 \
WARMUP_STEPS=5 MAX_STEPS=5 \
NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 NUMACTL_MODE=membind \
OVERWRITE=true CONTINUE_ON_ERROR=true \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Then run Asym:

```bash
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/profiling_liger_llama4_asym \
RUN_NAME=llama4_liger_asym \
MODEL_SPECS='meta-llama/Llama-4-Scout-17B-16E|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0,asym_cpuadamwds|norecomp|ligerloss1' \
ASYMM_EXP_ACT_POLICIES='none|true|false|false|false|false' \
WORKLOADS='8192|2|1' \
PROFILERS=both GPU_POOL=3 \
WARMUP_STEPS=5 MAX_STEPS=5 \
NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 NUMACTL_MODE=membind \
OVERWRITE=true CONTINUE_ON_ERROR=true \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Use `WORKLOADS='8192|2|1'` as the first real validation workload. If the no-Liger baseline OOMs, record that as an OOM-avoidance result, then rerun both off/on at the largest common workload that completes so latency can still be compared fairly. Do not accept from tiny toy profiling.

Compare each off/on pair:

```bash
SFT_ROOT=${SFT_ROOT:-/home/kevinni/AsymGEMM-SFT}
ASYM_DIR=${ASYM_DIR:-${SFT_ROOT}/third_party/AsymGEMM}
ENV_PYTHON=${ENV_PYTHON:-${ASYM_DIR}/.venv/bin/python}

"${ENV_PYTHON}" "${ASYM_DIR}/scripts/lf/compare_liger_loss_profiles.py" \
  --baseline /path/to/zero3/ligerloss0/run_dir \
  --candidate /path/to/zero3/ligerloss1/run_dir \
  --backend zero3_offload \
  --baseline-liger-loss ligerloss0 \
  --candidate-liger-loss ligerloss1 \
  --min-peak-drop-gib 10 \
  --min-lm-head-loss-drop-gib 20 \
  --max-step-ratio 1.10 \
  --max-forward-ratio 1.15 \
  --max-backward-ratio 1.15

"${ENV_PYTHON}" "${ASYM_DIR}/scripts/lf/compare_liger_loss_profiles.py" \
  --baseline /path/to/asym/ligerloss0/run_dir \
  --candidate /path/to/asym/ligerloss1/run_dir \
  --backend asym_cpuadamwds \
  --baseline-liger-loss ligerloss0 \
  --candidate-liger-loss ligerloss1 \
  --min-peak-drop-gib 10 \
  --min-lm-head-loss-drop-gib 20 \
  --max-step-ratio 1.10 \
  --max-forward-ratio 1.15 \
  --max-backward-ratio 1.15
```

Acceptance criteria:
- Run paths contain `__sdparecomp0__` and `__ligerloss0/1__`.
- `source_profile.json["config"]["liger_loss"]` matches the folder path.
- `source_profile.json["config"]["asymm_attn_sdpa_recompute"] == "false"`.
- Liger-on logs contain `Liger loss-only kernel has been applied.`
- For real `model_type="llama4"` runs, logs also show the post-load Llama4 conditional bridge was installed. A class-level Liger message alone is not sufficient.
- zero3 Llama4 conditional Liger-on has `source_profile.json["asym_liger_lm_head_bridge"]["enabled"] == true`, `bridge_kind == "conditional_generation"`, and `weight_source == "normal_parameter"`.
- zero3 Llama4 Liger-on reduces peak allocated HBM by at least 10 GiB and reduces lm_head/loss HBM attribution by at least 20 GiB.
- Asym Llama4 Liger-on reduces peak allocated HBM by at least 10 GiB and reduces lm_head/loss HBM attribution by at least 20 GiB.
- Asym Liger-on has `source_profile.json["asym_liger_lm_head_bridge"]["enabled"] == true`.
- Asym Liger-on bridge metadata has `weight_source == "asym_host_staged"` if `lm_head` is Asym-wrapped.
- Mean and median step time do not exceed no-Liger by more than 10%.
- Median forward and backward time do not exceed no-Liger by more than 15%.
- Losses remain finite and close between off/on runs at the same seed/workload. NaNs, collapse, or large systematic divergence reject the change.
- Source and Nsight/materialized-source artifacts exist under separate `ligerloss0` and `ligerloss1` paths.

Final implementation checklist:
- LF allows `qwen3_moe`, `llama4_text`, and `llama4` through the loss-only gate.
- LF passes only `fused_linear_cross_entropy=True` to Liger and disables every other bool patch.
- Llama4 conditional-generation runs have a post-load bridge that bypasses full logits in the training loss path.
- AsymGEMM runs stage `lm_head` through `_resolve_liger_lm_head_weight`.
- Normal/DeepSpeed Llama4 conditional runs use the same top-level bridge with `normal_parameter` weight.
- Profiling commands pin `ASYMM_EXP_ACT_POLICIES='none|true|false|false|false|false'`.
- E2E zero3 and Asym Llama4 comparisons show meaningful HBM reduction without forward/backward or step-time blow-up.
