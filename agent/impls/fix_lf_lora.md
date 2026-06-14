# Fix LF Qwen MoE Expert LoRA

Goal: make plain LlamaFactory/ZeRO attach, train, save, and reload Qwen fused MoE expert LoRA correctly whenever LF/PEFT target selection reaches Qwen expert modules. This is not an AsymGEMM feature and should not use AsymGEMM naming.

Resolved implementation facts:

- Local Transformers Qwen3/Qwen3.5 fused experts use `Qwen3MoeExperts` / `Qwen3_5MoeExperts.forward(hidden_states, top_k_index, top_k_weights)` and store `gate_up_proj [E, 2I, H]`, `down_proj [E, H, I]`.
- PEFT custom LoRA modules are the clean path: upstream docs require `nn.Module + LoraLayer`, adapter tensors in `ParameterDict`/`ModuleDict`, attribute names starting with `lora_`, and re-registering custom modules before adapter load.
- PEFT `target_parameters` can target 3D MoE tensors, but stock behavior materializes per-expert LoRA deltas and does not express separate gate/up LoRA at rank `r`. Do not use it for this correction.
- Current LF helper `_lf_fused_lora_params` trains in-memory but is not PEFT-native enough for reliable normal adapter save/load. Replace it for new runs.
- The A/B comparison is central: the profiling flow must run both stock PEFT expert LoRA and the corrected custom PEFT-compatible expert LoRA as selectable modes on the same workload.
- The custom wrapper is not only for `lora_target=all`: if target selection includes `experts`, `.mlp.experts`, or exact fused expert module names, use the custom wrapper in `custom-peft` mode.

## Stage 0 - Add LF expert-LoRA mode selection to profiling

Modify:

- `scripts/lf/profile_lora_lf.sh`
  - add profile dimension `LF_EXPERT_LORA_IMPLS`
  - add CLI option `--lf-expert-lora-impls`
  - include the mode in run labels, cache completeness checks, and env passed to `run_lf_lora_sft.sh`
- `scripts/lf/run_lf_lora_sft.sh`
  - add env var `LF_QWEN_MOE_EXPERT_LORA_IMPL`
  - propagate it to the training process and source-profile config
- `scripts/lf/run_lf_profiled_train.py`
  - record `qwen_moe_expert_lora_impl` in the profile JSON config
- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/model/adapter.py`
  - read `LF_QWEN_MOE_EXPERT_LORA_IMPL` during LoRA setup

Modes:

```text
custom-peft             corrected PEFT custom module with separate gate/up/down LoRA
peft-target-parameters  stock PEFT target_parameters on gate_up_proj/down_proj
off                     no fused expert LoRA; dense attention/shared-expert LoRA only
```

Implementation:

```bash
# profile_lora_lf.sh
LF_EXPERT_LORA_IMPLS=${LF_EXPERT_LORA_IMPLS:-custom-peft}

case "${arg}" in
  --lf-expert-lora-impls) LF_EXPERT_LORA_IMPLS="$2"; shift 2 ;;
  --lf-expert-lora-impls=*) LF_EXPERT_LORA_IMPLS="${arg#*=}"; shift ;;
esac

parse LF_EXPERT_LORA_IMPLS as comma list
validate each item in {custom-peft,peft-target-parameters,off}

for model_spec in ...
  for backend_spec in ...
    for expert_lora_impl in "${lf_expert_lora_impls[@]}"
      RUN_ENV+=(
        LF_QWEN_MOE_EXPERT_LORA_IMPL="${expert_lora_impl}"
        ASYM_GEMM_LF_CONFIG_QWEN_EXPERT_LORA_IMPL="${expert_lora_impl}"
      )
      run_id includes "qwenexpert-${expert_lora_impl}"
```

```bash
# run_lf_lora_sft.sh
LF_QWEN_MOE_EXPERT_LORA_IMPL=${LF_QWEN_MOE_EXPERT_LORA_IMPL:-custom-peft}
RUN_ENV+=(
  LF_QWEN_MOE_EXPERT_LORA_IMPL="${LF_QWEN_MOE_EXPERT_LORA_IMPL}"
  ASYM_GEMM_LF_CONFIG_QWEN_EXPERT_LORA_IMPL="${LF_QWEN_MOE_EXPERT_LORA_IMPL}"
)
```

```python
# run_lf_profiled_train.py
config["qwen_moe_expert_lora_impl"] = os.environ.get(
    "ASYM_GEMM_LF_CONFIG_QWEN_EXPERT_LORA_IMPL",
    os.environ.get("LF_QWEN_MOE_EXPERT_LORA_IMPL", "custom-peft"),
)
```

Risk and uncertainty:

- If the profile cache key does not include `qwen_moe_expert_lora_impl`, stale artifacts from the other mode can be reused. The completeness check must reject mismatched mode.
- `peft-target-parameters` requires `lora_dropout=0.0` because PEFT `ParamWrapper` rejects dropout for raw parameter targets. The script should fail early if this mode is requested with nonzero dropout.

Validation before Stage 1:

```bash
.venv/bin/python -m pytest -q \
  tests/lf/test_asym_cpu_adamw_args.py \
  tests/lf/test_lf_profile_postprocess.py::test_profile_config_records_qwen_expert_lora_impl
```

Dry-run or one-step smoke must show two distinct profile rows/artifacts:

```bash
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="zero3_offload|recomp" \
ASYMM_EXP_ACT_POLICIES="none|false|false|false" \
LF_EXPERT_LORA_IMPLS="peft-target-parameters,custom-peft" \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
DATASET_MIN_TOKENS=4096 \
DATASET_OVERWRITE=true \
LORA_TARGET=all \
LORA_DROPOUT=0.00 \
PROFILERS=source \
WARMUP_STEPS=0 \
MAX_STEPS=1 \
RUN_POST=true \
scripts/lf/profile_lora_lf.sh --gpus 0
```

## Stage 1 - Implement a PEFT-native Qwen expert LoRA layer

Modify:

- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/model/model_utils/fused_moe_lora.py`
  - Replace `_lf_fused_lora_params`, `_make_fused_lora_params`, `_fused_experts_lora_forward`, and `apply_fused_moe_lora`.
  - Add class `QwenMoeExpertLoraLayer`.
  - Add helpers `_is_qwen_fused_experts_module`, `_qwen_expert_dims`, `prepare_qwen_moe_expert_lora_config`, `infer_qwen_expert_lora_impl`, `count_qwen_moe_expert_lora`.
  - Add stock-PEFT helper `_add_qwen_moe_target_parameters` for the A/B mode.

Implementation:

```python
def _is_qwen_fused_experts_module(module):
    gate_up = getattr(module, "gate_up_proj", None)
    down = getattr(module, "down_proj", None)
    return (
        isinstance(gate_up, torch.Tensor)
        and isinstance(down, torch.Tensor)
        and gate_up.ndim == 3
        and down.ndim == 3
        and gate_up.shape[0] == down.shape[0]      # E
        and gate_up.shape[1] == 2 * down.shape[2]  # 2I
        and gate_up.shape[2] == down.shape[1]      # H
        and hasattr(module, "act_fn")
        and hasattr(module, "num_experts")
    )

def _qwen_expert_dims(module):
    gate_up = module.gate_up_proj
    down = module.down_proj
    experts = int(gate_up.shape[0])
    intermediate = int(down.shape[2])
    hidden = int(down.shape[1])
    return experts, hidden, intermediate
```

```python
class QwenMoeExpertLoraLayer(nn.Module, LoraLayer):
    adapter_layer_names = (
        "lora_gate_A", "lora_gate_B",
        "lora_up_A", "lora_up_B",
        "lora_down_A", "lora_down_B",
    )

    def __init__(
        self,
        base_layer,
        adapter_name,
        r=0,
        lora_alpha=1,
        lora_dropout=0.0,
        init_lora_weights=True,
        use_rslora=False,
        use_dora=False,
        lora_bias=False,
        **kwargs,
    ):
        if use_dora or lora_bias:
            raise ValueError("QwenMoeExpertLoraLayer supports vanilla LoRA only")
        if not _is_qwen_fused_experts_module(base_layer):
            raise TypeError("expected Qwen fused MoE experts module")

        nn.Module.__init__(self)
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.experts, self.hidden_size, self.intermediate_size = _qwen_expert_dims(base_layer)
        self._active_adapter = adapter_name

        self.lora_gate_A = nn.ParameterDict()
        self.lora_gate_B = nn.ParameterDict()
        self.lora_up_A = nn.ParameterDict()
        self.lora_up_B = nn.ParameterDict()
        self.lora_down_A = nn.ParameterDict()
        self.lora_down_B = nn.ParameterDict()

        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
        )
```

```python
def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, **_):
    if r <= 0:
        raise ValueError("r must be positive")

    E, H, I = self.experts, self.hidden_size, self.intermediate_size
    dtype = self.base_layer.gate_up_proj.dtype
    device = self.base_layer.gate_up_proj.device

    self.r[adapter_name] = r
    self.lora_alpha[adapter_name] = lora_alpha
    self.scaling[adapter_name] = lora_alpha / math.sqrt(r) if use_rslora else lora_alpha / r
    self.use_rslora[adapter_name] = use_rslora
    self.lora_dropout[adapter_name] = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

    self.lora_gate_A[adapter_name] = nn.Parameter(torch.empty(E, r, H, dtype=dtype, device=device))
    self.lora_gate_B[adapter_name] = nn.Parameter(torch.zeros(E, I, r, dtype=dtype, device=device))
    self.lora_up_A[adapter_name] = nn.Parameter(torch.empty(E, r, H, dtype=dtype, device=device))
    self.lora_up_B[adapter_name] = nn.Parameter(torch.zeros(E, I, r, dtype=dtype, device=device))
    self.lora_down_A[adapter_name] = nn.Parameter(torch.empty(E, r, I, dtype=dtype, device=device))
    self.lora_down_B[adapter_name] = nn.Parameter(torch.zeros(E, H, r, dtype=dtype, device=device))

    if init_lora_weights is True:
        for A in (self.lora_gate_A, self.lora_up_A, self.lora_down_A):
            nn.init.kaiming_uniform_(A[adapter_name].view(E * r, -1), a=math.sqrt(5))
    elif init_lora_weights in (False, None):
        pass
    else:
        raise ValueError("only default LoRA init is supported for Qwen expert LoRA")

    self._move_adapter_to_device_of_base_layer(adapter_name)
    self.set_adapter(self.active_adapters, inference_mode=False)
```

```python
def _lora_linear(self, x, A, B, expert_idx, scale):
    # x [N, in], A[e] [r, in], B[e] [out, r]
    return F.linear(F.linear(x, A[expert_idx]), B[expert_idx]) * scale

def forward(self, hidden_states, top_k_index, top_k_weights):
    base = self.base_layer
    final = torch.zeros_like(hidden_states)

    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=int(base.num_experts)).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        x = hidden_states[token_idx]

        gate, up = F.linear(x, base.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        for adapter in self.active_adapters:
            if adapter not in self.lora_gate_A:
                continue
            x_lora = self.lora_dropout[adapter](x)
            scale = self.scaling[adapter]
            gate = gate + self._lora_linear(x_lora, self.lora_gate_A[adapter], self.lora_gate_B[adapter], expert_idx, scale)
            up = up + self._lora_linear(x_lora, self.lora_up_A[adapter], self.lora_up_B[adapter], expert_idx, scale)

        act = base.act_fn(gate) * up
        down = F.linear(act, base.down_proj[expert_idx])
        for adapter in self.active_adapters:
            if adapter not in self.lora_down_A:
                continue
            act_lora = self.lora_dropout[adapter](act)
            scale = self.scaling[adapter]
            down = down + self._lora_linear(act_lora, self.lora_down_A[adapter], self.lora_down_B[adapter], expert_idx, scale)

        down = down * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, down.to(final.dtype))

    return final
```

Risk and uncertainty:

- `merge_and_unload()` is not required for training/profiling, but loading an adapter for merge/export will need `merge`, `unmerge`, and `unload_and_optionally_merge_module`.
- Custom PEFT module registration is an experimental PEFT API. The implementation must be covered by save/load tests because the wrapper type is not stored in the adapter config.
- Stock PEFT `target_parameters` is kept only for profiling comparison. It should not replace the corrected custom path unless e2e memory/latency clearly wins and the trainable surface is acceptable.

Validation before Stage 2:

```bash
.venv/bin/python -m py_compile \
  /workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/model/model_utils/fused_moe_lora.py
```

Add a unit test in `tests/lf/test_asym_cpu_adamw_lf_integration.py` and run:

```bash
.venv/bin/python -m pytest -q \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_qwen_moe_expert_lora_layer_registers_peft_native_params
```

Required assertions:

- one Qwen expert module is wrapped;
- `E=4,H=16,I=8,r=2` produces `576` expert LoRA params;
- trainable names include `mlp.experts.lora_gate_A.default` and `mlp.experts.lora_down_B.default`;
- no trainable names under `.mlp.gate`;
- `get_peft_model_state_dict(model)` includes six expert LoRA tensor families after adapter-name removal.
- separate test for `peft-target-parameters` confirms PEFT wraps `mlp.experts.gate_up_proj` and `mlp.experts.down_proj` without creating custom `lora_gate_*` tensors.

## Stage 2 - Wire the wrapper into LF new-adapter creation

Modify:

- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/model/adapter.py`
  - `_setup_lora_tuning`
- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/model/model_utils/fused_moe_lora.py`
  - `prepare_qwen_moe_expert_lora_config`
  - `_add_qwen_moe_target_parameters`
  - `_target_selection_reaches_qwen_experts`

Implementation:

```python
def _target_selection_reaches_qwen_experts(raw_lora_target, resolved_target_modules, expert_module_name):
    raw = {str(t).lower() for t in (raw_lora_target or [])}
    resolved = {str(t).lower() for t in (resolved_target_modules or [])}
    name = expert_module_name.lower()
    suffix = name.rsplit(".", 1)[-1]

    selectors = raw | resolved
    if {"all", "all-linear", "all_linear"} & selectors:
        return True
    if "experts" in selectors:
        return True
    if name in selectors or suffix in selectors:
        return True
    if any(name.endswith("." + target) for target in selectors):
        return True
    if any(target.endswith(".mlp.experts") or target == "mlp.experts" for target in selectors):
        return True
    return False

def _selected_qwen_expert_module_names(model, raw_lora_target, resolved_target_modules):
    names = []
    for name, module in model.named_modules():
        if not _is_qwen_fused_experts_module(module):
            continue
        if _target_selection_reaches_qwen_experts(raw_lora_target, resolved_target_modules, name):
            names.append(name)
    return names

def _add_qwen_moe_target_parameters(model, peft_config, raw_lora_target, resolved_target_modules):
    selected_experts = _selected_qwen_expert_module_names(model, raw_lora_target, resolved_target_modules)
    if not selected_experts:
        return peft_config

    target_parameters = set(getattr(peft_config, "target_parameters", None) or [])

    # PEFT suffix matching covers all decoder layers.
    target_parameters.update({
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    })
    peft_config.target_parameters = sorted(target_parameters)
    return peft_config

def prepare_qwen_moe_expert_lora_config(model, peft_config, mode, raw_lora_target, resolved_target_modules):
    if mode == "off":
        return peft_config
    if mode == "peft-target-parameters":
        if float(getattr(peft_config, "lora_dropout", 0.0) or 0.0) != 0.0:
            raise ValueError("peft-target-parameters requires lora_dropout=0.0")
        return _add_qwen_moe_target_parameters(model, peft_config, raw_lora_target, resolved_target_modules)
    if mode != "custom-peft":
        raise ValueError(f"unknown Qwen MoE expert LoRA mode: {mode}")

    expert_names = _selected_qwen_expert_module_names(model, raw_lora_target, resolved_target_modules)
    custom_types = {}
    for name, module in model.named_modules():
        if name in expert_names:
            custom_types[type(module)] = QwenMoeExpertLoraLayer

    if not expert_names:
        return peft_config

    targets = set(peft_config.target_modules or [])
    targets.update(expert_names)
    peft_config.target_modules = sorted(targets)
    peft_config._register_custom_module(custom_types)
    return peft_config
```

In `_setup_lora_tuning`, plain LF branch only:

```python
expert_lora_impl = os.environ.get("LF_QWEN_MOE_EXPERT_LORA_IMPL", "custom-peft")
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, **peft_kwargs)
if finetuning_args.finetuning_type == "lora":
    from .model_utils.fused_moe_lora import prepare_qwen_moe_expert_lora_config
    peft_config = prepare_qwen_moe_expert_lora_config(
        model,
        peft_config,
        expert_lora_impl,
        raw_lora_target=finetuning_args.lora_target,
        resolved_target_modules=target_modules,
    )
model = get_peft_model(model, peft_config)
```

Delete the old post-wrap manual call:

```python
from .model_utils.fused_moe_lora import apply_fused_moe_lora
apply_fused_moe_lora(model, ...)
```

Do not change `use_asym_gemm` or `use_kt` paths in this stage.

Risk and uncertainty:

- `find_all_linear_modules` still intentionally skips routers and fused expert tensor modules. The wrapper must add only Qwen fused expert modules that the raw or resolved target selection actually requested.
- PEFT may warn that the custom module type is unsupported. That is expected when using custom dispatch and should not fail tests.
- `peft-target-parameters` and `custom-peft` are not mathematically identical. The profile artifact must record the mode and expert parameter count so memory/latency differences are interpretable.
- Explicit non-expert targets such as `q_proj,v_proj` must not attach expert LoRA.

Validation before Stage 3:

```bash
.venv/bin/python -m pytest -q \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_lf_plain_lora_all_adapter_adds_qwen3_fused_expert_lora \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_lf_peft_lora_all_does_not_add_adapter_to_qwen_moe_routers \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_lf_qwen_moe_expert_lora_mode_peft_target_parameters \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_lf_qwen_moe_expert_lora_target_experts_uses_custom_wrapper \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_lf_qwen_moe_expert_lora_non_expert_target_does_not_wrap_experts
```

Update expected names to the PEFT-native wrapper names:

```text
mlp.experts.lora_gate_A.default
mlp.experts.lora_gate_B.default
mlp.experts.lora_up_A.default
mlp.experts.lora_up_B.default
mlp.experts.lora_down_A.default
mlp.experts.lora_down_B.default
```

## Stage 3 - Prove expert forward and backward correctness

Modify:

- `tests/lf/test_asym_cpu_adamw_lf_integration.py`
  - add `test_lf_qwen_moe_expert_lora_forward_matches_manual_reference`
  - add `test_lf_qwen_moe_expert_lora_backward_updates_expert_lora_params`

Implementation:

```python
def manual_expert_reference(layer, hidden, top_k_index, top_k_weights):
    final = zeros_like(hidden)
    for expert e used by top_k_index:
        x = hidden[token_idx]
        gate, up = linear(x, base.gate_up_proj[e]).chunk(2, -1)
        gate += linear(linear(x, layer.lora_gate_A["default"][e]), layer.lora_gate_B["default"][e]) * scale
        up += linear(linear(x, layer.lora_up_A["default"][e]), layer.lora_up_B["default"][e]) * scale
        act = base.act_fn(gate) * up
        down = linear(act, base.down_proj[e])
        down += linear(linear(act, layer.lora_down_A["default"][e]), layer.lora_down_B["default"][e]) * scale
        final.index_add_(0, token_idx, down * top_k_weights[token_idx, top_k_pos, None])
    return final
```

Forward test:

```python
model = tiny Qwen3MoeForCausalLM wrapped through _setup_lora_tuning(lora_target=["all"])
layer = model.base_model.model.model.layers[0].mlp.experts
fill all six expert LoRA tensors with deterministic nonzero values
hidden = torch.randn(M, H)
top_k_index = tensor covering all experts
top_k_weights = normalized weights
assert_allclose(layer(hidden, top_k_index, top_k_weights), manual_expert_reference(...))
```

Backward test:

```python
set all lora_B tensors nonzero so lora_A gradients are observable on first backward
out = layer(hidden.requires_grad_(), top_k_index, top_k_weights)
loss = out.square().mean()
loss.backward()
for each of lora_gate_A/B, lora_up_A/B, lora_down_A/B:
    assert grad is finite
    assert grad.abs().sum() > 0
```

Risk and uncertainty:

- With zero-initialized LoRA-B, LoRA-A gradients can be zero on the first backward. Tests must set LoRA-B nonzero or run two optimizer steps.
- Router-driven full-model tests can miss experts. Use direct expert-module tests with fixed `top_k_index` for correctness.

Validation before Stage 4:

```bash
.venv/bin/python -m pytest -q \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_lf_qwen_moe_expert_lora_forward_matches_manual_reference \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_lf_qwen_moe_expert_lora_backward_updates_expert_lora_params
```

## Stage 4 - Fix PEFT adapter save/load and resume

Modify:

- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/model/adapter.py`
  - `adapter_to_resume` branch in `_setup_lora_tuning`
  - `adapter_to_merge` branch in `_setup_lora_tuning`
- `tests/lf/test_asym_cpu_adamw_lf_integration.py`
  - add `test_lf_qwen_moe_expert_lora_save_load_roundtrip`

Implementation:

```python
def infer_qwen_expert_lora_impl(peft_config, requested_mode="auto"):
    if requested_mode != "auto":
        return requested_mode
    target_parameters = set(getattr(peft_config, "target_parameters", None) or [])
    if {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"} & target_parameters:
        return "peft-target-parameters"
    targets = set(getattr(peft_config, "target_modules", None) or [])
    if any(
        str(target).endswith(".mlp.experts") or str(target) in {"experts", "mlp.experts"}
        for target in targets
    ):
        return "custom-peft"
    return "off"

def load_peft_with_qwen_expert_lora(base_model, adapter_path, is_trainable, init_kwargs, requested_mode="auto"):
    from peft import PeftConfig, PeftModel
    from .model_utils.fused_moe_lora import (
        infer_qwen_expert_lora_impl,
        prepare_qwen_moe_expert_lora_config,
    )

    peft_config = PeftConfig.from_pretrained(adapter_path, **init_kwargs)
    if getattr(peft_config, "peft_type", None).value == "LORA":
        mode = infer_qwen_expert_lora_impl(peft_config, requested_mode)
        saved_targets = getattr(peft_config, "target_modules", None) or []
        peft_config = prepare_qwen_moe_expert_lora_config(
            base_model,
            peft_config,
            mode,
            raw_lora_target=saved_targets,
            resolved_target_modules=saved_targets,
        )
    return PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=is_trainable,
        config=peft_config,
        **init_kwargs,
    )
```

Use this helper for:

```python
requested_mode = os.environ.get("LF_QWEN_MOE_EXPERT_LORA_IMPL", "auto")
model = load_peft_with_qwen_expert_lora(model, adapter_to_resume, is_trainable, init_kwargs, requested_mode)
model = load_peft_with_qwen_expert_lora(model, adapter, False, init_kwargs, requested_mode).merge_and_unload()
```

Save/load test:

```python
model_a = tiny Qwen3 lora_target=all model
fill expert LoRA params with deterministic values
model_a.save_pretrained(tmpdir)

model_b = fresh tiny Qwen3 base
model_b = load_peft_with_qwen_expert_lora(model_b, tmpdir, is_trainable=True, init_kwargs={}, requested_mode="auto")

assert six expert LoRA tensor families exist in model_b
assert saved and loaded tensors match exactly
assert logits match for fixed input_ids
```

Risk and uncertainty:

- Old `_lf_fused_lora_params` checkpoints will not load through this new path unless a migration mapper is added. Treat this as a legacy checkpoint migration, not part of the new accepted path.
- `merge_and_unload()` remains a watch item until Stage 1 adds merge methods. If merge is not required for profiling, defer export merge support.
- `auto` mode inference is based on saved PEFT config. If a custom adapter config does not retain full expert module target names, pass `LF_QWEN_MOE_EXPERT_LORA_IMPL=custom-peft` explicitly during resume.

Validation before Stage 5:

```bash
.venv/bin/python -m pytest -q \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_lf_qwen_moe_expert_lora_save_load_roundtrip \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_lf_qwen_moe_expert_lora_peft_target_parameters_save_load_roundtrip
```

## Stage 5 - Update profiling counters and surface checks

Modify:

- `scripts/lf/run_lf_profiled_train.py`
  - `_collect_lora_summary`
- `scripts/lf/postprocess_lf_profile_artifacts.py`
  - trainable-surface summary logic
- `scripts/lf/run_lf_lora_sft.sh`
  - source-ok surface validation
- `tests/lf/test_lf_profile_postprocess.py`

Implementation:

```python
def is_peft_lora_name(name):
    return ".lora_" in name or "lora_" in name

def is_expert_lora_name(name):
    lowered = name.lower()
    return is_peft_lora_name(name) and any(marker in lowered for marker in (
        ".experts.",
        ".expert.",
        ".shared_expert",
        "shared_experts",
        "block_sparse_moe",
        "moe.experts",
    ))
```

Expected new profile counters:

```text
custom-peft:
  qwen_moe_expert_lora_impl == "custom-peft"
  peft_expert_lora_parameters > 0
  peft_expert_lora_tensors includes lora_gate/up/down A/B tensors
  lf_fused_expert_lora_parameters == 0
  expert_lora_parameters == peft_expert_lora_parameters

peft-target-parameters:
  qwen_moe_expert_lora_impl == "peft-target-parameters"
  peft_expert_lora_parameters > 0
  peft_expert_lora_tensors includes PEFT ParamWrapper lora_A/lora_B tensors
  lf_fused_expert_lora_parameters == 0
  expert_lora_parameters == peft_expert_lora_parameters
```

Keep `_lf_fused_lora_params` counting only as legacy detection. It must not be the accepted new LF expert LoRA path.

Risk and uncertainty:

- The broader `lora_` counter must not accidentally count non-LoRA metadata. Count only trainable `nn.Parameter`s.
- Postprocess tests currently expect old `LF fused expert LoRA` wording. Update wording to `PEFT expert LoRA`.
- Postprocess output must group or label rows by `qwen_moe_expert_lora_impl`; otherwise A/B memory/timing rows are ambiguous.

Validation before Stage 6:

```bash
.venv/bin/python -m pytest -q \
  tests/lf/test_lf_profile_postprocess.py::test_lora_counters_count_peft_expert_lora_by_parameter_name \
  tests/lf/test_lf_profile_postprocess.py::test_source_summary_flags_peft_attention_plus_expert_surface \
  tests/lf/test_lf_profile_postprocess.py::test_profile_summary_keeps_qwen_expert_lora_impl_in_rows
```

Then run the local unit group:

```bash
.venv/bin/python -m pytest -q \
  tests/lf/test_asym_cpu_adamw_lf_integration.py \
  tests/lf/test_lf_profile_postprocess.py
```

## Stage 6 - Validate with real LF/ZeRO profiling

Modify:

- No model-code edits unless Stage 6 exposes a bug.
- Use `scripts/lf/profile_lora_lf.sh` artifacts to compare `peft-target-parameters` against `custom-peft` on the same workload.
- This stage is required before accepting the LF expert-LoRA correction because the user-facing question is memory/latency, not only correctness.

Smoke validation:

```bash
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="zero3_offload|recomp" \
ASYMM_EXP_ACT_POLICIES="none|false|false|false" \
LF_EXPERT_LORA_IMPLS="peft-target-parameters,custom-peft" \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
DATASET_MIN_TOKENS=4096 \
DATASET_OVERWRITE=true \
LORA_TARGET=all \
LORA_DROPOUT=0.00 \
PROFILERS=source \
PROFILE_MEMORY_ATTRIBUTION=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_SNAPSHOT=false \
WARMUP_STEPS=1 \
MAX_STEPS=2 \
RUN_POST=true \
scripts/lf/profile_lora_lf.sh --gpus 0
```

Acceptance validation:

```bash
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="zero3_offload|recomp" \
ASYMM_EXP_ACT_POLICIES="none|false|false|false" \
LF_EXPERT_LORA_IMPLS="peft-target-parameters,custom-peft" \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
DATASET_MIN_TOKENS=4096 \
DATASET_OVERWRITE=true \
LORA_TARGET=all \
LORA_DROPOUT=0.00 \
PROFILERS=source \
PROFILE_MEMORY_ATTRIBUTION=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_SNAPSHOT=false \
WARMUP_STEPS=5 \
MAX_STEPS=10 \
RUN_POST=true \
scripts/lf/profile_lora_lf.sh --gpus 0
```

Acceptance checks from artifacts:

- `reference_fallback_count == 0`
- finite stable loss
- `lora_target == "all"`
- exactly two comparable rows exist for `qwen_moe_expert_lora_impl in {"peft-target-parameters", "custom-peft"}`
- both rows have `peft_expert_lora_parameters > 0`
- both rows have `lf_fused_expert_lora_parameters == 0`
- no trainable router LoRA under `.mlp.gate`
- no missing/unexpected expert LoRA keys in save/load validation
- peak HBM and timing are recorded in the same table format used by prior `profile_lora_lf.sh` comparisons
- report table must include:
  - `backend spec`
  - `qwen_moe_expert_lora_impl`
  - `peak allocated HBM`
  - `peak reserved HBM`
  - `avg step`
  - `avg forward`
  - `avg backward`
  - `peft_expert_lora_parameters`
  - `loss max/last/train`

Risk and uncertainty:

- This correction may change ZeRO memory/timing because it changes how expert LoRA is represented and saved. Accept `custom-peft` as the default only if it fixes the trainable surface and does not create a meaningful profiling regression versus `peft-target-parameters`.
- If `peft-target-parameters` is faster/lower-memory but has the wrong gate/up math, keep it as an explicit A/B baseline, not the default corrected behavior.
