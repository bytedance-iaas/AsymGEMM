# Fix LF Qwen MoE Expert LoRA

Goal: make plain LlamaFactory/ZeRO attach, train, save, reload, and profile Qwen fused MoE expert LoRA correctly, while keeping the fast Transformers Qwen expert execution path. This is not an AsymGEMM feature and should not use AsymGEMM naming.

The accepted correction follows PEFT `ParamWrapper` runtime style:

```text
compute full 3D LoRA delta weight
temporarily parametrize original expert parameter as W + delta
call the original Qwen experts module unchanged
remove parametrization
```

Local evidence:

- `/home/kevinni/AsymGEMM-SFT/third_party/peft/src/peft/tuners/lora/layer.py::ParamWrapper`
  - `get_delta_weight()` materializes a 3D expert delta.
  - `_activate_lora()` registers `_LoraParameterProxy(delta_weight)` with `nn.utils.parametrize`.
  - `forward()` calls `self.base_layer(...)`, preserving the original module implementation.
- `/workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/lib/python3.12/site-packages/transformers/integrations/moe.py`
  - `use_experts_implementation()` dispatches Qwen experts to `grouped_mm` by config.
  - `grouped_mm_experts_forward()` sorts tokens by expert and uses grouped GEMM.
- Official PEFT docs say MoE expert `nn.Parameter`s should use `target_parameters`; they also warn that PEFT materializes LoRA contribution for each expert, which is expected overhead, but that is still much faster than replacing grouped expert execution with Python loops.

Accepted A/B baseline, Qwen3-30B-A3B, `zero3_offload|recomp`, `b4_s4096`, `WARMUP_STEPS=5`, `MAX_STEPS=10`:

| impl | peak allocated HBM | peak reserved HBM | avg step | avg forward | avg backward | trainable params | expert LoRA params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `peft-target-parameters` | `33.107 GiB` | `38.756 GiB` | `3.610 s` | `1.035 s` | `2.510 s` | `2,570,059,776` | `2,516,582,400` |
| `split-target-parameters` | `33.138 GiB` | `38.268 GiB` | `5.155 s` | `1.424 s` | `3.730 s` | `3,375,366,144` | `3,321,888,768` |

Acceptance target:

- Keep the split gate/up/down expert LoRA trainable surface: `3,321,888,768` expert LoRA params for Qwen3-30B-A3B rank 64.
- Preserve original Qwen expert module forward and Transformers `grouped_mm` dispatch.
- E2E timing must be close to `peft-target-parameters`; any small overhead must be justified by the larger trainable surface.
- If peak memory stays the same but latency remains materially worse, reject the change.

## Stage 0 - Add a new PEFT-style mode and keep A/B baselines

Modify:

- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/model/model_utils/fused_moe_lora.py`
  - constants: add `QWEN_EXPERT_LORA_SPLIT_TARGET_PARAMETERS = "split-target-parameters"`
  - `QWEN_EXPERT_LORA_MODES`
  - `prepare_qwen_moe_expert_lora_config`
  - `infer_qwen_expert_lora_impl`
- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/model/adapter.py`
  - mode validation remains env-driven through `LF_QWEN_MOE_EXPERT_LORA_IMPL`
- `scripts/lf/profile_lora_lf.sh`
  - allow `LF_EXPERT_LORA_IMPLS=split-target-parameters`
- `scripts/lf/run_lf_lora_sft.sh`
  - allow `LF_QWEN_MOE_EXPERT_LORA_IMPL=split-target-parameters`
- `scripts/lf/run_lf_profiled_train.py`
  - no schema rename; keep `qwen_moe_expert_lora_impl`
- `scripts/lf/postprocess_lf_profile_artifacts.py`
  - render the new mode label
- `tests/lf/test_asym_cpu_adamw_args.py`
  - add dry-run coverage for the new mode

Implementation:

```python
QWEN_EXPERT_LORA_SPLIT_TARGET_PARAMETERS = "split-target-parameters"
QWEN_EXPERT_LORA_MODES = {
    "peft-target-parameters",
    "split-target-parameters",
    "off",
}

def prepare_qwen_moe_expert_lora_config(model, peft_config, mode, raw_lora_target, resolved_target_modules):
    mode = normalize(mode)
    if mode == "off":
        return peft_config
    if mode == "peft-target-parameters":
        _patch_peft_param_wrapper_zero3_shape()
        return _add_qwen_moe_target_parameters(...)
    if mode == "split-target-parameters":
        _patch_peft_param_wrapper_zero3_shape()
        return _add_qwen_moe_split_target_parameters(...)
    raise ValueError(...)
```

Rules:

- Default is now `split-target-parameters` after Stage 5 acceptance.
- Keep `peft-target-parameters` as the fast stock PEFT A/B baseline.
- Only `peft-target-parameters`, `split-target-parameters`, and `off` remain supported.

Unresolved risks to watch:

- Existing cached profile rows must not be reused across mode changes. The existing profile completeness check must reject mismatched `qwen_moe_expert_lora_impl`.

Validation before Stage 1:

```bash
bash -n scripts/lf/profile_lora_lf.sh scripts/lf/run_lf_lora_sft.sh
.venv/bin/python -m py_compile \
  scripts/lf/run_lf_profiled_train.py \
  scripts/lf/postprocess_lf_profile_artifacts.py \
  /workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/model/model_utils/fused_moe_lora.py \
  /workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/model/adapter.py
```

Dry-run sweep:

```bash
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="zero3_offload|recomp" \
ASYMM_EXP_ACT_POLICIES="none|false|false|false" \
LF_EXPERT_LORA_IMPLS="peft-target-parameters,split-target-parameters" \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=false \
LORA_TARGET=all \
LORA_DROPOUT=0.00 \
PROFILERS=source \
WARMUP_STEPS=0 \
MAX_STEPS=1 \
RUN_POST=false \
DRY_RUN=true \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
scripts/lf/profile_lora_lf.sh --gpus 0 --output-root /tmp/asymgemm_lf_lora_split_dryrun
```

Required dry-run assertions:

- three run directories are emitted;
- labels include `qwenexpertsplit-target-parameters`;
- every command sets `LF_QWEN_MOE_EXPERT_LORA_IMPL`;
- `peft-target-parameters` still rejects nonzero `LORA_DROPOUT`.

## Stage 1 - Implement a PEFT-style split expert parameter wrapper

Modify:

- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/model/model_utils/fused_moe_lora.py`
  - add class `QwenSplitMoeExpertParamWrapper`
  - add helper `_LoraParameterProxy` import or local equivalent
  - add helper `_split_expert_param_shape`
  - add helper `_selected_qwen_expert_module_names` reuse
  - add helper `_add_qwen_moe_split_target_parameters`
  - do not modify Qwen model source files

Design:

- Follow PEFT `ParamWrapper` structure.
- The wrapper must wrap the existing Qwen experts module and call `self.base_layer(...)`.
- The wrapper owns split gate/up/down LoRA tensors, but only materializes full parameter deltas:
  - `delta_gate_up` for `gate_up_proj [E, 2I, H]`
  - `delta_down` for `down_proj [E, H, I]`
- Runtime must preserve original Qwen expert forward and therefore preserve Transformers `grouped_mm`.

Concrete class shape:

```python
class QwenSplitMoeExpertParamWrapper(nn.Module, LoraLayer):
    adapter_layer_names = (
        "lora_gate_A", "lora_gate_B",
        "lora_up_A", "lora_up_B",
        "lora_down_A", "lora_down_B",
    )
    parameter_names = ("gate_up_proj", "down_proj")

    def __init__(
        self,
        base_layer,
        adapter_name,
        r,
        lora_alpha,
        config=None,
        lora_dropout=0.0,
        init_lora_weights=True,
        use_rslora=False,
        use_dora=False,
        lora_bias=False,
        **kwargs,
    ):
        nn.Module.__init__(self)
        LoraLayer.__init__(self, base_layer, **kwargs)
        validate_qwen_fused_experts(base_layer)
        reject lora_dropout != 0, fan_in_fan_out, use_dora, lora_bias
        self.experts, self.hidden_size, self.intermediate_size = _qwen_expert_dims(base_layer)
        self._active_adapter = adapter_name
        create ParameterDicts
        self.update_layer(...)
```

Parameter shapes:

```python
# gate/up LoRA, separate rank-r paths
lora_gate_A[adapter]: [E, r, H]
lora_gate_B[adapter]: [E, I, r]
lora_up_A[adapter]:   [E, r, H]
lora_up_B[adapter]:   [E, I, r]

# down LoRA
lora_down_A[adapter]: [E, r, I]
lora_down_B[adapter]: [E, H, r]
```

Delta construction:

```python
def _expert_delta(weight_b, weight_a, scaling):
    # weight_b [E, O, r], weight_a [E, r, In]
    return torch.einsum("eor,eri->eoi", weight_b, weight_a) * scaling

def get_delta_weight(self, adapter_name, parameter_name):
    if parameter_name == "gate_up_proj":
        delta_gate = _expert_delta(lora_gate_B[adapter], lora_gate_A[adapter], scale)  # [E,I,H]
        delta_up = _expert_delta(lora_up_B[adapter], lora_up_A[adapter], scale)        # [E,I,H]
        delta = torch.cat((delta_gate, delta_up), dim=1)                               # [E,2I,H]
        param = self.get_base_layer().gate_up_proj
        return delta.to(device=param.device, dtype=param.dtype)

    if parameter_name == "down_proj":
        delta = _expert_delta(lora_down_B[adapter], lora_down_A[adapter], scale)       # [E,H,I]
        param = self.get_base_layer().down_proj
        return delta.to(device=param.device, dtype=param.dtype)

    raise ValueError(parameter_name)
```

Activation:

```python
@contextmanager
def _activate_lora(self, active_adapters):
    if no active split adapters:
        yield
        return

    base = self.get_base_layer()
    registered = []
    for parameter_name in ("gate_up_proj", "down_proj"):
        delta_weight = None
        for adapter in active_adapters:
            if adapter not in self.lora_gate_A:
                continue
            candidate = self.get_delta_weight(adapter, parameter_name)
            delta_weight = candidate if delta_weight is None else delta_weight + candidate

        if delta_weight is None:
            continue

        requires_grad_before = getattr(base, parameter_name).requires_grad
        nn.utils.parametrize.register_parametrization(
            base,
            parameter_name,
            _LoraParameterProxy(delta_weight),
        )
        base.parametrizations[parameter_name].original.requires_grad_(requires_grad_before)
        registered.append(parameter_name)

    try:
        with nn.utils.parametrize.cached():
            yield
    finally:
        remove only the split LoRA parametrizations registered above
```

Forward:

```python
def forward(self, hidden_states, top_k_index, top_k_weights):
    self._check_forward_args(hidden_states)
    if self.disable_adapters or self.merged:
        return self.base_layer(hidden_states, top_k_index, top_k_weights)
    with self._activate_lora(self.active_adapters):
        return self.base_layer(hidden_states, top_k_index, top_k_weights)
```

Merge/unmerge:

```python
def merge(self, safe_merge=False, adapter_names=None):
    for adapter in check_adapters_to_merge(...):
        for parameter_name in ("gate_up_proj", "down_proj"):
            param = getattr(base, parameter_name)
            delta = self.get_delta_weight(adapter, parameter_name)
            if safe_merge:
                merged = param.data.clone() + delta.to(param.dtype)
                assert finite
                param.data = merged
            else:
                param.data += delta.to(param.dtype)
        self.merged_adapters.append(adapter)

def unmerge(self):
    while self.merged_adapters:
        adapter = self.merged_adapters.pop()
        for parameter_name in ("gate_up_proj", "down_proj"):
            getattr(base, parameter_name).data -= self.get_delta_weight(adapter, parameter_name)
```

Important implementation rules:

- Do not call `F.linear` inside this wrapper forward.
- Do not loop over active experts or tokens in wrapper forward.
- Do not reimplement router/top-k/scatter/index-add behavior.
- Use PEFT naming conventions (`lora_*`) so adapter save/load keeps working.

Unresolved risks to watch:

- Full delta materialization creates temporary `[E,2I,H]` and `[E,H,I]` tensors per layer. This is the same class of overhead as PEFT `target_parameters`; it should be far cheaper than Python expert loops, but must be measured.
- Nested parametrization with two parameters on the same module must remove only the wrappers registered by this class, not unrelated parametrizations.
- ZeRO-3 empty local params require logical `ds_shape` during initialization. Reuse and extend `_tensor_logical_shape`.

Validation before Stage 2:

```bash
.venv/bin/python -m py_compile \
  /workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/model/model_utils/fused_moe_lora.py
```

Unit tests to add/run:

```bash
.venv/bin/python -m pytest -q \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_qwen_split_param_wrapper_shapes_and_trainable_names \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_qwen_split_param_wrapper_delta_shapes \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_qwen_split_param_wrapper_preserves_base_forward_call
```

Required assertions:

- trainable names include:
  - `mlp.experts.lora_gate_A.default`
  - `mlp.experts.lora_up_A.default`
  - `mlp.experts.lora_down_A.default`
- no trainable router LoRA under `.mlp.gate`;
- `get_delta_weight("default", "gate_up_proj").shape == [E, 2I, H]`;
- `get_delta_weight("default", "down_proj").shape == [E, H, I]`;
- wrapper forward calls the original base expert module exactly once;
- wrapper forward does not call the old custom per-expert `_lora_linear`.

## Stage 2 - Register split wrapper through PEFT-compatible injection

Modify:

- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/model/model_utils/fused_moe_lora.py`
  - `_add_qwen_moe_split_target_parameters`
  - `prepare_qwen_moe_expert_lora_config`
  - `infer_qwen_expert_lora_impl`
- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/model/adapter.py`
  - `_setup_lora_tuning`
  - `_load_peft_with_qwen_expert_lora`
- `tests/lf/test_asym_cpu_adamw_lf_integration.py`

Preferred registration:

- Use PEFT custom module registration for the Qwen experts module type, but the custom module must be `QwenSplitMoeExpertParamWrapper`.
- Do not use `target_parameters` for the split wrapper itself because stock PEFT can only create one A/B pair per targeted parameter and cannot express separate gate/up ranks inside one fused `gate_up_proj`.
- Keep `peft-target-parameters` mode as stock PEFT `target_parameters` for comparison.

Pseudocode:

```python
def _add_qwen_moe_split_target_parameters(model, peft_config, raw_lora_target, resolved_target_modules):
    expert_names = _selected_qwen_expert_module_names(model, raw_lora_target, resolved_target_modules)
    if not expert_names:
        logger.info_rank0("Qwen split expert LoRA selected 0 fused expert modules.")
        return peft_config

    custom_types = {}
    for name, module in model.named_modules():
        if name in expert_names:
            custom_types[type(module)] = QwenSplitMoeExpertParamWrapper

    targets = set(peft_config.target_modules or [])
    targets.update(expert_names)
    peft_config.target_modules = sorted(targets)
    peft_config._register_custom_module(custom_types)
    logger.info_rank0(
        f"Qwen split expert LoRA selected {len(expert_names)} fused expert modules."
    )
    return peft_config
```

Adapter load/resume:

```python
def infer_qwen_expert_lora_impl(peft_config, requested_mode="auto"):
    if requested_mode != "auto":
        return requested_mode
    target_modules = set(peft_config.target_modules or [])
    target_parameters = set(peft_config.target_parameters or [])
    if adapter state/config contains lora_gate_A/lora_up_A/lora_down_A:
        return "split-target-parameters"
    if any(target endswith "gate_up_proj" for target in target_parameters):
        return "peft-target-parameters"
    if any(target endswith ".mlp.experts" or target == "experts" for target in target_modules):
        return "split-target-parameters"
    return "off"
```

Rules:

- New adapters with `LF_QWEN_MOE_EXPERT_LORA_IMPL=split-target-parameters` use `QwenSplitMoeExpertParamWrapper`.
- Resuming an adapter must re-register the split wrapper before `PeftModel.from_pretrained`.
- Do not silently load split-wrapper adapter state into stock `ParamWrapper`; tensor names and math differ.

Unresolved risks to watch:

- PEFT config alone may not distinguish old saved split expert adapters from new `split-target-parameters` if both target the same module names. Use adapter state key detection when loading from disk if needed.
- If PEFT custom module registration passes a newer `config=` kwarg instead of old individual kwargs, support both forms.

Validation before Stage 3:

```bash
.venv/bin/python -m pytest -q \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_lf_plain_lora_all_uses_qwen_split_param_wrapper \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_lf_qwen_split_param_wrapper_respects_explicit_experts_target \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_lf_qwen_split_param_wrapper_non_expert_target_does_not_wrap_experts \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_lf_qwen_split_param_wrapper_auto_load_registers_custom_module
```

Required assertions:

- `lora_target=all` wraps all 48 Qwen3 fused expert modules on the full model or all expert modules in a tiny model;
- explicit `LORA_TARGET=q_proj,v_proj` does not wrap experts;
- explicit `LORA_TARGET=experts` wraps experts;
- adapter reload finds split wrapper tensors and restores them.

## Stage 3 - Prove math equivalence without losing grouped expert dispatch

Modify:

- `tests/lf/test_asym_cpu_adamw_lf_integration.py`
  - add direct math/reference tests
- Optional isolated debug script only if needed:
  - `scripts/lf/debug_qwen_split_lora_wrapper.py`

Reference math:

```python
delta_gate[e] = lora_gate_B[e] @ lora_gate_A[e]      # [I,H]
delta_up[e]   = lora_up_B[e]   @ lora_up_A[e]        # [I,H]
delta_down[e] = lora_down_B[e] @ lora_down_A[e]      # [H,I]

gate_up_eff = gate_up_proj + cat(delta_gate, delta_up, dim=1)
down_eff = down_proj + delta_down

expected = original_qwen_experts_forward_with_parametrized_weights(
    hidden_states,
    top_k_index,
    top_k_weights,
)
```

Tests:

```python
def test_qwen_split_param_wrapper_forward_matches_manual_parametrized_reference():
    build tiny Qwen experts module
    build QwenSplitMoeExpertParamWrapper
    set nonzero deterministic LoRA tensors
    run wrapper(hidden, top_k_index, top_k_weights)
    manually add full delta to cloned base weights
    run base(hidden, top_k_index, top_k_weights)
    assert_allclose

def test_qwen_split_param_wrapper_backward_updates_all_six_lora_families():
    set B tensors nonzero so A gradients are visible on first backward
    loss = wrapper(...).float().square().mean()
    loss.backward()
    assert finite nonzero grads for gate/up/down A and B

def test_qwen_split_param_wrapper_preserves_grouped_mm_dispatch():
    monkeypatch base.config._experts_implementation = "grouped_mm"
    monkeypatch transformers.integrations.moe.grouped_mm_experts_forward counter
    wrapper(...)
    assert grouped_mm path was called
```

Unresolved risks to watch:

- Monkeypatching `grouped_mm_experts_forward` may not catch dispatch if the interface stores the function before patching. If so, validate by setting `config._experts_implementation="eager"` versus `"grouped_mm"` in a timing smoke, and assert the wrapper does not alter `base.config._experts_implementation`.
- Numerical comparison must use `lora_dropout=0.0`; dropout is unsupported for PEFT parameter-style expert LoRA.

Validation before Stage 4:

```bash
.venv/bin/python -m pytest -q \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_qwen_split_param_wrapper_forward_matches_manual_parametrized_reference \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_qwen_split_param_wrapper_backward_updates_all_six_lora_families \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_qwen_split_param_wrapper_preserves_grouped_mm_dispatch
```

Run existing LF integration group:

```bash
.venv/bin/python -m pytest -q tests/lf/test_asym_cpu_adamw_lf_integration.py
```

## Stage 4 - Update counters, save/load, and postprocess for the accepted split path

Modify:

- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/model/model_utils/fused_moe_lora.py`
  - `count_qwen_moe_expert_lora`
  - `count_fused_moe_lora`
- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/model/adapter.py`
  - `_maybe_write_lora_surface_sidecar`
  - `_load_peft_with_qwen_expert_lora`
- `scripts/lf/run_lf_profiled_train.py`
  - `_lora_counters_from_model`
  - sidecar merge logic
  - optimizer memory preflight
- `scripts/lf/run_lf_lora_sft.sh`
  - source-profile gate
- `scripts/lf/postprocess_lf_profile_artifacts.py`
  - summary rows
- `tests/lf/test_lf_profile_postprocess.py`

Counter rules:

```text
split-target-parameters:
  qwen_moe_expert_lora_impl == "split-target-parameters"
  qwen_moe_expert_lora_modules == number of wrapped expert modules
  qwen_moe_expert_lora_tensors == modules * 6
  qwen_moe_expert_lora_parameters == split gate/up/down expert parameter total
  peft_expert_lora_parameters == qwen_moe_expert_lora_parameters

peft-target-parameters:
  qwen_moe_expert_lora_impl == "peft-target-parameters"
  qwen_moe_expert_lora_parameters == 0
  peft_expert_lora_parameters == stock PEFT target_parameters expert total
```

Sidecar:

```python
sidecar = {
    "qwen_moe_expert_lora_impl": mode,
    "qwen_moe_expert_lora_modules": modules,
    "qwen_moe_expert_lora_tensors": tensors,
    "qwen_moe_expert_lora_parameters": params,
    "peft_expert_lora_parameters": params,
    "peft_lora_parameters": total_lora_params,
    "trainable_parameters": trainable_params,
}
```

Save/load validation:

```python
model = tiny Qwen3 + split wrapper
fill six LoRA families deterministically
model.save_pretrained(tmpdir)
loaded = load_peft_with_qwen_expert_lora(base_model, tmpdir, requested_mode="auto")
assert all six tensor families exist
assert state tensors match
assert forward output matches before save
```

Unresolved risks to watch:

- `get_peft_model_state_dict` key format may strip adapter names differently from direct `state_dict`. Tests must verify the actual saved adapter files, not only in-memory keys.
- ZeRO-3 partitioned model views can report zero local `numel`; sidecar counters must use logical `ds_numel` where available.

Validation before Stage 5:

```bash
.venv/bin/python -m pytest -q \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_qwen_split_param_wrapper_save_load_roundtrip \
  tests/lf/test_lf_profile_postprocess.py::test_lora_counters_count_qwen_split_param_wrapper \
  tests/lf/test_lf_profile_postprocess.py::test_lora_counters_use_sidecar_when_split_wrapper_model_view_is_partitioned \
  tests/lf/test_lf_profile_postprocess.py::test_source_summary_labels_split_target_parameters
```

Then:

```bash
.venv/bin/python -m pytest -q \
  tests/lf/test_asym_cpu_adamw_args.py \
  tests/lf/test_lf_profile_postprocess.py \
  tests/lf/test_asym_cpu_adamw_lf_integration.py
```

## Stage 5 - Run real Qwen3/ZeRO A/B profiling and decide acceptance

Status: completed and accepted.

Modify:

- No model-code edits unless profiling exposes a correctness or performance bug.
- Update this doc and `agent/status.md` with accepted measured rows.

Smoke profile:

```bash
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="zero3_offload|recomp" \
ASYMM_EXP_ACT_POLICIES="none|false|false|false" \
LF_EXPERT_LORA_IMPLS="peft-target-parameters,split-target-parameters" \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
DATASET_MIN_TOKENS=4096 \
DATASET_OVERWRITE=false \
LORA_TARGET=all \
LORA_DROPOUT=0.00 \
PROFILERS=source \
PROFILE_MEMORY_ATTRIBUTION=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_SNAPSHOT=false \
WARMUP_STEPS=5 \
MAX_STEPS=2 \
RUN_POST=false \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
CONTINUE_ON_ERROR=false \
OVERWRITE=true \
scripts/lf/profile_lora_lf.sh --gpus 0 --output-root profiling/lf_lora_split_smoke
```

Acceptance profile:

```bash
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="zero3_offload|recomp" \
ASYMM_EXP_ACT_POLICIES="none|false|false|false" \
LF_EXPERT_LORA_IMPLS="peft-target-parameters,split-target-parameters" \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
DATASET_MIN_TOKENS=4096 \
DATASET_OVERWRITE=false \
LORA_TARGET=all \
LORA_DROPOUT=0.00 \
PROFILERS=source \
PROFILE_MEMORY_ATTRIBUTION=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_SNAPSHOT=false \
WARMUP_STEPS=5 \
MAX_STEPS=10 \
RUN_POST=false \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
CONTINUE_ON_ERROR=false \
OVERWRITE=true \
scripts/lf/profile_lora_lf.sh --gpus 0 --output-root profiling/lf_lora_split_accept
```

Accepted result table from `profiling/lf_lora_split_accept`:

| backend spec | qwen expert LoRA impl | peak allocated HBM | peak reserved HBM | avg step | avg forward | avg backward | trainable params | PEFT expert params | Qwen split expert params | fallback | loss max/last/train |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `zero3_offload|recomp` | `peft-target-parameters` | `33.107 GiB` | `38.756 GiB` | `4.932 s` | `1.436 s` | `3.496 s` | `2,570,059,776` | `2,516,582,400` | `0` | `0` | `2.326/1.273/1.914` |
| `zero3_offload|recomp` | `split-target-parameters` | `33.138 GiB` | `38.268 GiB` | `6.026 s` | `1.671 s` | `4.355 s` | `3,375,366,144` | `3,321,888,768` | `3,321,888,768` | `0` | `2.273/1.231/1.874` |

Accepted conclusion:

- `split-target-parameters` fixes the trainable surface: `3,321,888,768` expert LoRA params and `3,375,366,144` total trainable params.
- It preserves the fast grouped expert path: avg step is `6.026 s`, close to stock PEFT target-parameters and far faster than the removed Python-loop wrapper.
- HBM stays in the accepted ZeRO range: `33.138 GiB` peak allocated.
- `reference_fallback_count=0`; losses are finite and in the same range.

Artifacts:

- `peft-target-parameters`: `profiling/lf_lora_split_accept/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b4_s4096_w5_s10_r64_a16_drop000/zero3_offload__source__recomp__polnone__routerhf__expact0__attnact0__layeract0__loraafwdcpu__qwenexpertpeft-target-parameters/b4_s4096/source_profile.json`
- `split-target-parameters`: `profiling/lf_lora_split_accept/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b4_s4096_w5_s10_r64_a16_drop000/zero3_offload__source__recomp__polnone__routerhf__expact0__attnact0__layeract0__loraafwdcpu__qwenexpertsplit-target-parameters/b4_s4096/source_profile.json`

Acceptance checks:

- `split-target-parameters` has `reference_fallback_count == 0`.
- losses are finite and in the same range as the A/B rows.
- `split-target-parameters` expert LoRA params equal `3,321,888,768`.
- `split-target-parameters` total trainable params equal `3,375,366,144`.
- `split-target-parameters` avg step is close to `peft-target-parameters`, not close to the removed Python-loop wrapper.
- peak HBM remains close to the existing ZeRO rows; no meaningful memory regression is accepted.
- if latency is materially worse than `peft-target-parameters`, inspect whether grouped expert dispatch was lost before accepting.

Unresolved risks to watch:

- Full delta materialization may make `split-target-parameters` slightly slower than stock PEFT because it materializes two deltas for fused gate/up before concatenating. That is acceptable only if the overhead is small relative to the corrected trainable surface.
- If `split-target-parameters` regresses toward removed-wrapper timing, the implementation likely replaces or bypasses the Qwen expert grouped path and must be rejected.

## Stage 6 - Promote the new mode only after acceptance

Status: completed.

Modify after Stage 5 passes:

- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/model/model_utils/fused_moe_lora.py`
  - make `QWEN_EXPERT_LORA_SPLIT_TARGET_PARAMETERS` the default corrected mode
  - keep only supported modes: `split-target-parameters`, `peft-target-parameters`, and `off`
- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/model/adapter.py`
  - default env fallback becomes `split-target-parameters`
- `scripts/lf/profile_lora_lf.sh`
  - default `LF_EXPERT_LORA_IMPLS=split-target-parameters`
- `scripts/lf/run_lf_lora_sft.sh`
  - default `LF_QWEN_MOE_EXPERT_LORA_IMPL=split-target-parameters`
- `agent/status.md`
  - record accepted table

Implementation:

```python
def _get_qwen_moe_expert_lora_impl(default: str = "split-target-parameters"):
    return os.environ.get("LF_QWEN_MOE_EXPERT_LORA_IMPL", default).strip().lower()
```

Validation:

```bash
.venv/bin/python -m pytest -q \
  tests/lf/test_asym_cpu_adamw_args.py \
  tests/lf/test_lf_profile_postprocess.py \
  tests/lf/test_asym_cpu_adamw_lf_integration.py
```

One final default-mode smoke:

```bash
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="zero3_offload|recomp" \
ASYMM_EXP_ACT_POLICIES="none|false|false|false" \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=false \
LORA_TARGET=all \
LORA_DROPOUT=0.00 \
PROFILERS=source \
WARMUP_STEPS=5 \
MAX_STEPS=2 \
RUN_POST=false \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
CONTINUE_ON_ERROR=false \
OVERWRITE=true \
scripts/lf/profile_lora_lf.sh --gpus 0 --output-root profiling/lf_lora_split_default_smoke
```

Default smoke result from `profiling/lf_lora_split_default_smoke`:

| qwen expert LoRA impl | peak allocated HBM | peak reserved HBM | avg step | avg forward | avg backward | trainable params | expert LoRA params | fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `split-target-parameters` | `33.138 GiB` | `38.268 GiB` | `5.155 s` | `1.424 s` | `3.730 s` | `3,375,366,144` | `3,321,888,768` | `0` |

Required final state:

- default row records `qwen_moe_expert_lora_impl == "split-target-parameters"`;
- expert LoRA param count matches `3,321,888,768`;
- source-profile surface gate passes;
- no trainable router LoRA;
- no stale expert LoRA mode remains in scripts.
