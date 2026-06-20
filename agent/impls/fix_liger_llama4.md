# Liger Loss-Only for Llama4

Goal: make `--enable_liger_kernel true` enable only Liger fused linear cross entropy for Llama4, the same way it already does for Qwen3-MoE. This must work for regular LF/DeepSpeed runs and for AsymGEMM runs where `lm_head` can be offloaded/wrapped. Do not enable any other Liger kernels, and do not patch unsupported model types.

Non-goals:
- Do not modify `third_party/Liger-Kernel` unless local validation proves its existing Llama4 loss path is broken.
- Do not add a second runtime knob. The only user-facing runtime knob remains `ENABLE_LIGER_KERNEL`, surfaced in sweeps as `BACKEND_SPECS=...|ligerloss0` or `...|ligerloss1`.
- Do not special-case profiling paths outside the existing `ligerloss0/1` axis unless validation finds a stale artifact filter.

Current local facts:
- `../Liger-Kernel/src/liger_kernel/transformers/monkey_patch.py` already has `apply_liger_kernel_to_llama4(...)`.
- Its signature is:

```python
(
    rope=True,
    cross_entropy=False,
    fused_linear_cross_entropy=True,
    rms_norm=True,
    swiglu=True,
    model=None,
    layer_norm=True,
)
```

- `../Liger-Kernel/src/liger_kernel/transformers/model/llama4.py::lce_forward` already uses `LigerForCausalLMLoss`.
- That Liger Llama4 forward calls `lm_head_weight=self.lm_head.weight`. This is fine for normal/DeepSpeed models but not sufficient for AsymGEMM when `lm_head` is an `AsymFrozenLinear`, because Asym needs to stage the CPU-resident weight to the active device/dtype.
- `../LlamaFactory/src/llamafactory/model/model_utils/liger_kernel.py` currently whitelists only `qwen3_moe`.
- `asym_gemm/integrations/liger_loss.py` currently has only the Qwen3-MoE Asym bridge.
- `scripts/lf/profile_lora_lf.sh` already parses `ligerloss0/1`, includes `liger_loss` in output paths, completeness checks, jobs TSV, plot filters, and exports `ENABLE_LIGER_KERNEL`.

Install precondition:

```bash
python -m pip install -e /home/kevinni/AsymGEMM-SFT/third_party/Liger-Kernel
python - <<'PY'
import inspect
from liger_kernel.transformers import apply_liger_kernel_to_llama4
print(inspect.signature(apply_liger_kernel_to_llama4))
PY
```

Accept this precondition only if the signature includes `fused_linear_cross_entropy` and `layer_norm`. If the import resolves to a non-local package without Llama4 support, fix the environment before changing LF/Asym code.

## Stage 0 - Common Interface Preflight

Scope:
- `scripts/lf/profile_lora_lf.sh`
  - `append_backend_spec`
  - `liger_loss_label`
  - `job_root_path`
  - `job_profile_complete`
  - `append_liger_loss_filters`
  - `run_job`
  - `plot_single_run`
  - `plot_running_combined`
  - `plot_memory_single_run`
  - `plot_memory_running_combined`
- `scripts/lf/run_lf_lora_sft.sh`
  - `ENABLE_LIGER_KERNEL` normalization
  - `--enable_liger_kernel` argument forwarding
- `scripts/lf/run_lf_profiled_train.py`
  - `_asym_liger_lm_head_bridge_from_model`
  - emitted `config["asym_liger_lm_head_bridge"]`

Intended code changes:
- None expected before Stage 1.
- Keep the interface exactly as:
  - sweep axis: `BACKEND_SPECS='backend|recomp|ligerloss0,backend|recomp|ligerloss1'`
  - runtime env: `ENABLE_LIGER_KERNEL=true|false`
  - no `LIGER_LOSS_ONLY` env, because loss-only is the only supported behavior in this repo.

Validation before moving on:

```bash
DRY_RUN=true \
MODEL_SPECS='meta-llama/Llama-4-Scout-17B-16E|1' \
BACKEND_SPECS='zero3_offload|recomp|ligerloss0,zero3_offload|recomp|ligerloss1,asym_cpuadamwds|norecomp|ligerloss0,asym_cpuadamwds|norecomp|ligerloss1' \
PROFILERS=both GPU_POOL=3 \
NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 NUMACTL_MODE=membind \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Pass conditions:
- Dry-run commands contain `ENABLE_LIGER_KERNEL=true` only for `ligerloss1`.
- Output paths and run IDs contain `ligerloss0` or `ligerloss1`.
- `jobs.tsv` header includes `liger_loss`.
- No path or plot code needs to be changed if those pass.

Risk to watch:
- `profile3.sh` is out of scope for this change. Do not modify it for Llama4 Liger loss.

## Stage 1 - Enable Llama4 in LF Loss-Only Gate

Scope:
- Modify `../LlamaFactory/src/llamafactory/model/model_utils/liger_kernel.py`
  - `_LOSS_ONLY_SUPPORTED_MODEL_TYPES`
  - `_resolve_liger_apply_fn`
  - `_build_liger_loss_only_kwargs` only if tests prove it mishandles Llama4's `layer_norm` bool.
  - `apply_liger_kernel` should remain structurally unchanged.
- Modify `tests/lf/test_liger_loss_only_qwen3_moe.py`
  - Add Llama4 coverage while keeping existing Qwen3-MoE coverage.

Intended code changes:

```python
# ../LlamaFactory/src/llamafactory/model/model_utils/liger_kernel.py

_LOSS_ONLY_SUPPORTED_MODEL_TYPES = {"qwen3_moe", "llama4"}


def _resolve_liger_apply_fn(model_type: str | None) -> Callable[..., None] | None:
    if model_type == "qwen3_moe":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe
        return apply_liger_kernel_to_qwen3_moe

    if model_type == "llama4":
        from liger_kernel.transformers import apply_liger_kernel_to_llama4
        return apply_liger_kernel_to_llama4

    return None
```

`_build_liger_loss_only_kwargs` should already do the right thing:

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

That means LF applies only the class-level Llama4 fused CE forward and does not replace RoPE, RMSNorm, SwiGLU, LayerNorm, or standalone CE.

Tests to add:

```python
def test_build_liger_loss_only_kwargs_disables_llama4_non_loss_patches():
    def fake_apply(
        rope=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rms_norm=True,
        swiglu=True,
        model=None,
        layer_norm=True,
    ):
        pass

    assert liger_kernel._build_liger_loss_only_kwargs(fake_apply) == {
        "rope": False,
        "cross_entropy": False,
        "fused_linear_cross_entropy": True,
        "rms_norm": False,
        "swiglu": False,
        "layer_norm": False,
    }


def test_apply_liger_kernel_uses_loss_only_for_llama4(monkeypatch):
    calls = []
    monkeypatch.setattr(liger_kernel, "_resolve_liger_apply_fn", lambda model_type: lambda **kwargs: calls.append(kwargs))

    liger_kernel.apply_liger_kernel(
        SimpleNamespace(model_type="llama4"),
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

Also update the existing unsupported-model test so it no longer uses `llama4` as the unsupported example. Use `qwen3` or another unvalidated model type.

Validation before moving on:

```bash
pytest -q tests/lf/test_liger_loss_only_qwen3_moe.py
python - <<'PY'
import inspect
from liger_kernel.transformers import apply_liger_kernel_to_llama4
sig = inspect.signature(apply_liger_kernel_to_llama4)
required = {"fused_linear_cross_entropy", "rope", "cross_entropy", "rms_norm", "swiglu", "layer_norm", "model"}
missing = required - set(sig.parameters)
raise SystemExit(f"missing Llama4 Liger params: {missing}") if missing else None
print(sig)
PY
```

Pass conditions:
- Qwen3-MoE tests still pass.
- Llama4 tests prove only `fused_linear_cross_entropy=True` and every other bool patch is false.
- Unsupported model types still skip cleanly.

Risk to watch:
- This stage validates normal LF/DeepSpeed class patching only. It does not make AsymGEMM `lm_head` staging correct; that is Stage 2.

## Stage 2 - Add AsymGEMM Llama4 Loss Bridge

Scope:
- Modify `asym_gemm/integrations/liger_loss.py`
  - `_base_causal_lm_model`
  - add `_validate_liger_lm_head`
  - add `_mark_liger_bridge_installed`
  - add `asym_llama4_lce_forward`
  - add `install_asym_liger_llama4_loss_bridge`
  - add `install_asym_liger_loss_bridge`
  - keep `install_asym_liger_qwen3_moe_loss_bridge` as a compatibility wrapper
  - update `__all__`
- Modify `../LlamaFactory/src/llamafactory/model/adapter.py`
  - import `install_asym_liger_loss_bridge`
  - replace the Qwen3-only bridge install block with the generic dispatcher.
- Modify `tests/lf/test_asym_liger_lm_head_bridge.py`
  - add Llama4 tiny-model coverage.
  - update the unsupported-model test so `llama4` is no longer expected to skip.

Implementation details:

1. Make target-model resolution handle PEFT and possible Llama4 wrappers.

```python
def _candidate_language_models(model: nn.Module) -> list[nn.Module]:
    candidates = [model]

    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        try:
            base = get_base_model()
        except Exception:
            base = None
        if isinstance(base, nn.Module):
            candidates.append(base)

    expanded = []
    for candidate in candidates:
        expanded.append(candidate)
        language_model = getattr(candidate, "language_model", None)
        if isinstance(language_model, nn.Module):
            expanded.append(language_model)
        inner = getattr(candidate, "model", None)
        if isinstance(inner, nn.Module):
            expanded.append(inner)
            inner_language_model = getattr(inner, "language_model", None)
            if isinstance(inner_language_model, nn.Module):
                expanded.append(inner_language_model)

    return expanded


def _base_causal_lm_model(model: nn.Module) -> nn.Module:
    for candidate in _candidate_language_models(model):
        if isinstance(candidate, nn.Module) and hasattr(candidate, "lm_head"):
            return candidate
    return model
```

This keeps the existing Qwen3 path working and makes Llama4 robust if LF returns a wrapper whose causal LM lives under `language_model`.

2. Factor the shared `lm_head` validation.

```python
def _validate_liger_lm_head(target_model: nn.Module, *, model_label: str, strict: bool) -> tuple[nn.Module, str] | None:
    lm_head = getattr(target_model, "lm_head", None)
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
        raise RuntimeError("Asym Liger loss bridge currently requires a bias-free lm_head.")

    if any(param.requires_grad for param in lm_head.parameters(recurse=True)):
        raise RuntimeError("Asym Liger loss bridge supports frozen lm_head only.")

    return lm_head, weight_source
```

3. Preserve metadata for profiling.

```python
def _mark_liger_bridge_installed(target_model, lm_head, weight_source, model_type):
    target_model._asym_liger_lm_head_bridge_enabled = True
    target_model._asym_liger_lm_head_weight_source = weight_source
    target_model._asym_liger_lm_head_type = type(lm_head).__name__
    target_model._asym_liger_lm_head_staged_bytes = int(getattr(lm_head, "cpu_resident_base_weight_bytes", 0) or 0)
    target_model._asym_liger_model_type = model_type
```

`asym_liger_lm_head_bridge_metadata` should continue returning `enabled`, `weight_source`, `staged_bytes`, and `lm_head_type`; adding `model_type` is acceptable if tests and profile consumers are updated, but not required.

4. Add `asym_llama4_lce_forward`.

Use the local Liger implementation as the template:
`../Liger-Kernel/src/liger_kernel/transformers/model/llama4.py::lce_forward`.

The only behavior change is replacing `self.lm_head.weight` with the Asym-aware resolver.

```python
def asym_llama4_lce_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    cache_position=None,
    logits_to_keep=0,
    **kwargs,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

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
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

    if not return_dict:
        output = (logits,) + outputs[1:]
        output = ((loss,) + output) if loss is not None else output
        output = output + (token_accuracy,) if token_accuracy is not None else output
        output = output + (predicted_tokens,) if predicted_tokens is not None else output
        return output

    return LigerCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )
```

Required imports:

```python
from liger_kernel.transformers.model.output_classes import LigerCausalLMOutputWithPast
```

Keep the existing Qwen3 imports for `LigerMoeCausalLMOutputWithPast` and `MoeModelOutputWithPast`.

5. Add the Llama4 installer and generic dispatcher.

```python
def install_asym_liger_llama4_loss_bridge(model: nn.Module, *, strict: bool = True) -> bool:
    target_model = _base_causal_lm_model(model)
    config = getattr(target_model, "config", None)
    if getattr(config, "model_type", None) != "llama4":
        if strict:
            raise ValueError("Asym Liger loss bridge only supports llama4.")
        return False

    validated = _validate_liger_lm_head(target_model, model_label="Llama4", strict=strict)
    if validated is None:
        return False

    lm_head, weight_source = validated
    target_model.forward = MethodType(asym_llama4_lce_forward, target_model)
    _mark_liger_bridge_installed(target_model, lm_head, weight_source, "llama4")
    return True


def install_asym_liger_qwen3_moe_loss_bridge(model: nn.Module, *, strict: bool = True) -> bool:
    target_model = _base_causal_lm_model(model)
    config = getattr(target_model, "config", None)
    if getattr(config, "model_type", None) != "qwen3_moe":
        if strict:
            raise ValueError("Asym Liger loss bridge only supports qwen3_moe.")
        return False

    validated = _validate_liger_lm_head(target_model, model_label="Qwen3-MoE", strict=strict)
    if validated is None:
        return False

    lm_head, weight_source = validated
    target_model.forward = MethodType(asym_qwen3_moe_lce_forward, target_model)
    _mark_liger_bridge_installed(target_model, lm_head, weight_source, "qwen3_moe")
    return True


def install_asym_liger_loss_bridge(model: nn.Module, *, strict: bool = True) -> bool:
    target_model = _base_causal_lm_model(model)
    model_type = getattr(getattr(target_model, "config", None), "model_type", None)

    if model_type == "qwen3_moe":
        return install_asym_liger_qwen3_moe_loss_bridge(model, strict=strict)
    if model_type == "llama4":
        return install_asym_liger_llama4_loss_bridge(model, strict=strict)

    if strict:
        raise ValueError(f"Asym Liger loss bridge does not support model_type={model_type}.")
    return False
```

6. Wire the generic bridge in LF adapter after Asym wrapping.

```python
# ../LlamaFactory/src/llamafactory/model/adapter.py
from asym_gemm.integrations.liger_loss import install_asym_liger_loss_bridge

# after adapt_lf_asym_peft_lora(...)
if model_args.enable_liger_kernel and getattr(model.config, "model_type", None) in {"qwen3_moe", "llama4"}:
    bridge_installed = install_asym_liger_loss_bridge(
        model,
        strict=bool(model_args.asym_strict and selection.lm_head),
    )
    if bridge_installed:
        logger.info_rank0("Asym Liger lm_head bridge has been installed.")
```

Patch order:
- LF loader applies the global Liger class patch before model construction.
- PEFT/Asym adapter then wraps/offloads modules.
- Asym adapter then installs the instance-level bridge for Qwen3-MoE or Llama4.
- The instance bridge intentionally overrides the class-level Liger forward for Asym runs only, so `AsymFrozenLinear.asym_liger_lm_head_weight(...)` stages `lm_head` correctly.
- DeepSpeed/normal runs do not use the Asym bridge; they use Liger's normal Llama4 class-level fused CE forward.

Validation before moving on:

```bash
pytest -q tests/lf/test_asym_liger_lm_head_bridge.py
pytest -q tests/lf/test_liger_loss_only_qwen3_moe.py
```

New bridge tests must prove:
- `install_asym_liger_llama4_loss_bridge` patches only the model instance, not the class.
- `asym_llama4_lce_forward` passes a staged `AsymFrozenLinear` weight into `LigerForCausalLMLoss`.
- The Llama4 bridge rejects trainable or biased `lm_head`.
- The generic `install_asym_liger_loss_bridge` dispatches both `qwen3_moe` and `llama4`.
- Unsupported model types still return `False` with `strict=False`.

Risks to watch:
- Some Llama4 HF classes may be wrappers around a nested `language_model`; `_base_causal_lm_model` must locate the actual causal LM with `lm_head`.
- Do not enable `llama4_text` unless a real LF-loaded config proves that is the correct top-level `model_type`.
- If a future Llama4 loss implementation adds router auxiliary loss, compare against HF/Liger behavior before adding aux-loss code. The current local Liger Llama4 `lce_forward` does not add an MoE aux loss.

## Stage 3 - E2E Llama4 Validation

Scope:
- No new implementation files unless Stage 0 found stale profiling filters.
- Use:
  - `scripts/lf/profile_lora_lf.sh`
  - `scripts/lf/compare_liger_loss_profiles.py`
  - generated `source_profile.json`
  - generated `memory_breakdown_summary.json`
  - generated timing/memory plots.

Run zero3 first to validate Liger's normal Llama4 path:

```bash
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/profiling_liger_llama4_zero3 \
RUN_NAME=llama4_liger_zero3 \
MODEL_SPECS='meta-llama/Llama-4-Scout-17B-16E|1' \
BACKEND_SPECS='zero3_offload|recomp|ligerloss0,zero3_offload|recomp|ligerloss1' \
WORKLOADS='8192|2|1' \
PROFILERS=both GPU_POOL=3 \
WARMUP_STEPS=5 MAX_STEPS=5 \
NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 NUMACTL_MODE=membind \
OVERWRITE=true CONTINUE_ON_ERROR=true \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Then validate AsymGEMM staging:

```bash
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/profiling_liger_llama4_asym \
RUN_NAME=llama4_liger_asym \
MODEL_SPECS='meta-llama/Llama-4-Scout-17B-16E|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0,asym_cpuadamwds|norecomp|ligerloss1' \
WORKLOADS='8192|2|1' \
PROFILERS=both GPU_POOL=3 \
WARMUP_STEPS=5 MAX_STEPS=5 \
NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 NUMACTL_MODE=membind \
OVERWRITE=true CONTINUE_ON_ERROR=true \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Use `WORKLOADS='8192|2|1'` as the first real validation workload. If the no-Liger baseline OOMs, record that as an OOM-avoidance result, then rerun both off/on at the largest common workload that completes so latency can still be compared fairly. Do not accept from tiny toy workloads.

Compare each pair:

```bash
python /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/compare_liger_loss_profiles.py \
  --baseline /path/to/ligerloss0/run_dir \
  --candidate /path/to/ligerloss1/run_dir \
  --backend zero3_offload \
  --baseline-liger-loss ligerloss0 \
  --candidate-liger-loss ligerloss1 \
  --min-peak-drop-gib 10 \
  --min-lm-head-loss-drop-gib 20 \
  --max-step-ratio 1.10 \
  --max-forward-ratio 1.15 \
  --max-backward-ratio 1.15

python /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/compare_liger_loss_profiles.py \
  --baseline /path/to/ligerloss0/run_dir \
  --candidate /path/to/ligerloss1/run_dir \
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
- `train.log` for Liger-on runs contains `Liger loss-only kernel has been applied.`
- `source_profile.json["config"]["liger_loss"]` is `ligerloss0` or `ligerloss1` and matches the folder path.
- zero3 Llama4 Liger-on run completes and reduces peak allocated HBM by at least 10 GiB and reduces lm_head/loss HBM attribution by at least 20 GiB.
- Asym Llama4 Liger-on run completes and has:
  - `source_profile.json["config"]["asym_liger_lm_head_bridge"]["enabled"] == true`
  - `weight_source == "asym_host_staged"`
  - `lm_head_type == "AsymFrozenLinear"`
- Mean/median step time does not exceed no-Liger by more than 10%.
- Median forward and backward time do not exceed no-Liger by more than 15%.
- Losses remain finite and close between off/on runs at the same seed/workload. Small differences are acceptable, but NaNs, large divergence, or systematic loss collapse reject the change.
- Plots and postprocessed artifacts exist for both `source` and `nsys` materialized outputs under separate `ligerloss0` and `ligerloss1` paths.

If plots miss the Liger axis:
- Only then touch `scripts/lf/profile_lora_lf.sh`.
- The exact functions to inspect are `append_liger_loss_filters`, `plot_single_run`, `plot_running_combined`, `plot_memory_single_run`, `plot_memory_running_combined`, `job_root_path`, and `job_profile_complete`.
- The fix should be to propagate the existing `liger_loss` argument, not to add another naming convention.

## Final Implementation Checklist

- `../LlamaFactory/src/llamafactory/model/model_utils/liger_kernel.py` allows `model_type == "llama4"` and resolves `apply_liger_kernel_to_llama4`.
- `asym_gemm/integrations/liger_loss.py` has a Llama4 Asym bridge that stages `lm_head` through `_resolve_liger_lm_head_weight`.
- `../LlamaFactory/src/llamafactory/model/adapter.py` installs the generic Asym bridge after Asym wrapping for Qwen3-MoE and Llama4.
- Unit tests cover Qwen3-MoE, Llama4, unsupported model skip, and Asym staged `lm_head`.
- E2E Llama4 zero3 and Asym comparisons show meaningful HBM reduction without forward/backward or step-time blow-up.
