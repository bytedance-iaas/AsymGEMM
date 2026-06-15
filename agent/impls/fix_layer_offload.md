# Active Qwen3 Decoder-Layer Activation Offload Plan

Goal: replace the current hook-only `ASYMM_LAYER_ACT_OFFLOAD=true`
implementation with an explicit active offload/backfetch path for the remaining
Qwen3 decoder-layer activations not already owned by `attn_math.md` or
`mlp_math.md`.

Acceptance is profile-based, not toy-shape based. The change is accepted only
if the real LF LoRA workflow shows lower peak HBM on at least one measured HBM
metric versus the current traditional saved-tensor hook baseline, does not
increase any peak HBM metric by more than `0.5 GiB`, and does not regress
latency beyond the stage gate below. Use `scripts/lf/profile_lora_lf.sh` for
every acceptance profile.

The current `none|true|true|true` tuple is important:

```text
EXPERT_SELECTION_POLICY|ASYMM_EXPERT_ACT_OFFLOAD|ASYMM_ATTN_ACT_OFFLOAD|ASYMM_LAYER_ACT_OFFLOAD
none|true|true|true
```

Capture it before replacing the layer path. At that point it is the
traditional layer saved-tensor offload/staging baseline. After the active layer
implementation lands, rerun the same tuple and compare active backfetching
against that baseline artifact.

Resolved facts used by this plan:

- Local code currently installs `DecoderSavedTensorOffloadWrapper` from
  `asym_gemm/training/decoder_activation_offload.py` through
  `asym_gemm/integrations/lf.py::_wrap_qwen3_decoder_saved_tensor_offload_modules`.
- The current hook profile offloads float32 `[4,4096,2048]` tensors, with most
  decoder layers peaking at `0.5 GiB` CPU-owned saved tensors per layer.
- Qwen3 RMSNorm casts hidden states to float32, computes the RMS variance over
  the hidden dimension, multiplies by `rsqrt(variance + eps)`, then applies the
  RMSNorm weight. This was checked against the Hugging Face
  [`Qwen3RMSNorm` source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py).
- In the LF path, `ASYM_OFFLOAD_MODULES=all` can replace norms with
  `AsymFrozenRMSNorm` before layer activation offload is installed. The active
  wrapper must therefore work for both HF-style `Qwen3RMSNorm` and local
  `AsymFrozenRMSNorm`.

## Stage 0: Capture Baselines And Confirm The Target

Scope:

- No code changes.
- Read-only inspection of:
  - `asym_gemm/training/decoder_activation_offload.py`
  - `asym_gemm/integrations/lf.py`
  - `scripts/lf/run_lf_profiled_train.py`
  - `scripts/lf/profile_lora_lf.sh`
  - `agent/mlp_math.md`
  - `agent/attn_math.md`

Implementation steps:

1. Capture the current no-layer baseline:

   ```text
   none|true|true|false
   ```

   This shows expert and attention active offload without decoder-layer
   boundary offload.

2. Capture the current traditional layer saved-tensor baseline:

   ```text
   none|true|true|true
   ```

   Before this plan is implemented, this is the hook-only baseline. Keep this
   artifact for final comparison.

3. Extract the existing hook evidence from the profile:

   ```python
   rows = profile["activation_offload"]["rows"]
   decoder_hook_rows = [
       row for row in rows
       if row["activation_offload_stats"].get("decoder_saved_tensor_offload")
   ]
   shape_counts = Counter()
   for row in decoder_hook_rows:
       shape_counts.update(row["activation_offload_stats"].get("shape_counts", {}))
   assert shape_counts["float32:(4, 4096, 2048)"] > 0
   ```

Ambiguity and risk checks:

- If `none|true|true|true` already reports an active RMSNorm implementation,
  this doc is stale. Do not proceed until the baseline is recaptured from a
  pre-active-layer commit or from an explicit debug hook path.
- If the profile no longer shows large float32 `[B,T,H]` hook saves, redo the
  target selection before implementing. The current target list assumes those
  tensors still dominate layer-hook offload.

Validation before Stage 1:

```bash
OUTPUT_ROOT=outputs/lf_layer_active_stage0 \
OVERWRITE=true \
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=hbm \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYMM_EXP_ACT_POLICIES="none|true|true|false,none|true|true|true" \
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
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf.sh --gpus 0
```

Stage 0 passes only if both profiles finish, both use the real
`Qwen/Qwen3-30B-A3B` LF LoRA workflow, and the `none|true|true|true` artifact
contains decoder saved-tensor hook rows.

## Stage 1: Implement The Active RMSNorm Offload Primitive

Scope:

- Modify `asym_gemm/training/decoder_activation_offload.py`:
  - add `RMSNormActivationOffloadWrapper`
  - add `_RMSNormActivationOffloadFunction`
  - add `_rmsnorm_eps`
  - add `_rmsnorm_weight_for_input`
  - add `_rmsnorm_active_forward`
  - add `install_rmsnorm_activation_offload`
  - add `is_rmsnorm_activation_offload_wrapper`
  - add `rmsnorm_activation_offload_module_names`
  - keep `DecoderSavedTensorOffloadWrapper` available only for debug/fallback
    comparison, not as the final `ASYMM_LAYER_ACT_OFFLOAD=true` path
- Modify `asym_gemm/training/__init__.py`:
  - export the new active RMSNorm wrapper/install/query helpers
- Modify `tests/training/test_decoder_activation_offload.py`:
  - add HF-style fake RMSNorm parity tests
  - add `AsymFrozenRMSNorm` parity tests when norms are CPU-resident
  - keep existing decoder hook tests intact

Intended code changes:

```python
def _rmsnorm_eps(module: nn.Module) -> float:
    if hasattr(module, "variance_epsilon"):
        return float(module.variance_epsilon)
    if hasattr(module, "eps"):
        return float(module.eps)
    raise TypeError(f"{type(module).__name__} has no RMSNorm eps field")


def _rmsnorm_weight_for_input(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        host_weight = getattr(module, "host_weight", None)
        weight = getattr(host_weight, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"{type(module).__name__} has no RMSNorm weight tensor")
    return weight.to(device=x.device, non_blocking=True)
```

```python
class RMSNormActivationOffloadWrapper:
    def __init__(self, module: nn.Module, *, name: str, pin_memory: bool = True):
        self.module = module
        self.original_forward = module.forward
        self.name = name
        self.tag = f"decoder.rmsnorm.{name}"
        self.manager = ActivationOffloadManager(pin_memory=pin_memory)
        self.calls = 0
        self.skipped_calls = 0
        self.forward_offloads = 0
        self.backward_stages = 0
        self.dtype_counts: dict[str, int] = {}
        self.shape_counts: dict[str, int] = {}
        self._sync_module_stats()

    def install(self) -> None:
        self.module._asym_rmsnorm_activation_offload_wrapper = self
        self.module.forward = types.MethodType(_rmsnorm_active_forward, self.module)

    def run(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self.calls += 1
        if args or kwargs:
            raise TypeError("Qwen3 RMSNorm active offload expects only hidden_states")
        if not self.module.training or not torch.is_grad_enabled():
            self.skipped_calls += 1
            return self.original_forward(hidden_states)
        if hidden_states.device.type != "cuda" or not hidden_states.requires_grad:
            self.skipped_calls += 1
            return self.original_forward(hidden_states)
        if bool(getattr(self.module, "gated", False)) or bool(getattr(self.module, "shifted_weight", False)):
            raise RuntimeError("active layer offload supports Qwen3 RMSNorm, not gated/shifted RMSNorm")

        weight = _rmsnorm_weight_for_input(self.module, hidden_states)
        eps = _rmsnorm_eps(self.module)
        return _RMSNormActivationOffloadFunction.apply(hidden_states, weight, eps, self)

    def record_forward_offload(self, x: torch.Tensor) -> None:
        self.forward_offloads += 1
        dtype_key = str(x.dtype).replace("torch.", "")
        shape_key = f"{dtype_key}:{tuple(int(dim) for dim in x.shape)}"
        self.dtype_counts[dtype_key] = self.dtype_counts.get(dtype_key, 0) + 1
        self.shape_counts[shape_key] = self.shape_counts.get(shape_key, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        stats = self.manager.snapshot()
        return {
            **stats,
            "rmsnorm_active_activation_offload": True,
            "layer_act_offload_impl": "rmsnorm_active",
            "name": self.name,
            "calls": self.calls,
            "skipped_calls": self.skipped_calls,
            "num_forward_offloads": self.forward_offloads,
            "num_backward_stages": self.backward_stages,
            "dtype_counts": dict(self.dtype_counts),
            "shape_counts": dict(self.shape_counts),
        }

    def _sync_module_stats(self) -> None:
        self.module._last_activation_offload_stats = self.snapshot()
```

```python
class _RMSNormActivationOffloadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float, wrapper):
        x_cpu = wrapper.manager.offload(x, tag=f"{wrapper.tag}.x")
        wrapper.record_forward_offload(x)

        input_dtype = x.dtype
        x_f = x.to(torch.float32)
        inv = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + float(eps))
        y = (x_f * inv).to(dtype=input_dtype) * weight
        if y.dtype != input_dtype:
            y = y.to(dtype=input_dtype)

        ctx.wrapper = wrapper
        ctx.x_cpu = x_cpu
        ctx.eps = float(eps)
        ctx.input_dtype = input_dtype
        ctx.save_for_backward(weight)
        wrapper._sync_module_stats()
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        (weight,) = ctx.saved_tensors
        wrapper = ctx.wrapper
        x_stage = wrapper.manager.stage(ctx.x_cpu, tag=f"{wrapper.tag}.x.backward")
        wrapper.backward_stages += 1
        try:
            x_f = x_stage.to(torch.float32)
            gy_f = grad_y.to(torch.float32)
            w_f = weight.to(device=grad_y.device, dtype=torch.float32, non_blocking=True)

            inv = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + ctx.eps)
            u = gy_f * w_f
            c = (u * x_f).mean(dim=-1, keepdim=True)
            grad_x = (inv * u - x_f * inv.pow(3) * c).to(dtype=ctx.input_dtype)

            reduce_dims = tuple(range(grad_y.dim() - 1))
            grad_weight = (gy_f * x_f * inv).sum(dim=reduce_dims).to(dtype=weight.dtype)
            return grad_x, grad_weight, None, None
        finally:
            wrapper.manager.release_stage(x_stage, drop_cache=True)
            wrapper.manager.release_cpu(ctx.x_cpu)
            wrapper._sync_module_stats()
```

```python
def _rmsnorm_active_forward(module: nn.Module, hidden_states: torch.Tensor, *args, **kwargs):
    wrapper = getattr(module, "_asym_rmsnorm_activation_offload_wrapper", None)
    if not isinstance(wrapper, RMSNormActivationOffloadWrapper):
        raise RuntimeError("RMSNorm active activation-offload wrapper is missing")
    return wrapper.run(hidden_states, *args, **kwargs)


def install_rmsnorm_activation_offload(module: nn.Module, *, name: str) -> RMSNormActivationOffloadWrapper:
    existing = getattr(module, "_asym_rmsnorm_activation_offload_wrapper", None)
    if isinstance(existing, RMSNormActivationOffloadWrapper):
        return existing
    wrapper = RMSNormActivationOffloadWrapper(module, name=name)
    wrapper.install()
    return wrapper
```

Ambiguity and risk checks:

- The wrapper must not rely on the norm weight being a CUDA `nn.Parameter`.
  The real LF profile can use `AsymFrozenRMSNorm`, whose weight is CPU-resident.
- The active wrapper recomputes float32 RMSNorm intermediates from the original
  bf16 input. This should match Qwen3 training behavior, but it must be tested
  against autograd output, input grad, and weight grad.
- Gated or shifted RMSNorm variants are not part of this Qwen3 layer plan.
  Raise clearly instead of silently using the wrong formula.

Validation before Stage 2:

```bash
.venv/bin/python -m pytest -q \
  tests/training/test_decoder_activation_offload.py::test_rmsnorm_active_activation_offload_matches_hf_style_forward_backward \
  tests/training/test_decoder_activation_offload.py::test_rmsnorm_active_activation_offload_matches_asym_frozen_rmsnorm_forward_backward \
  tests/training/test_decoder_activation_offload.py::test_decoder_saved_tensor_offload_preserves_backward_and_records_stats
```

Also run a real LF no-op regression profile to prove the new primitive has not
changed the production path before integration:

```bash
OUTPUT_ROOT=outputs/lf_layer_active_stage1_noop \
OVERWRITE=true \
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=hbm \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYMM_EXP_ACT_POLICIES="none|true|true|false" \
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
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf.sh --gpus 0
```

Stage 1 passes if unit tests pass, the no-op profile completes, and profile
JSON contains no `rmsnorm_active_activation_offload` rows when
`ASYMM_LAYER_ACT_OFFLOAD=false`.

## Stage 2: Install Active Wrappers On Qwen3 Decoder RMSNorms

Scope:

- Modify `asym_gemm/integrations/lf.py`:
  - imports from `asym_gemm.training.decoder_activation_offload`
  - replace `_wrap_qwen3_decoder_saved_tensor_offload_modules` with
    `_wrap_qwen3_decoder_rmsnorm_activation_offload_modules`
  - update `apply_lf_asym_lora` near the existing `layer_act_enabled` block
  - update `LFAsymReport.layer_act_offload_modules` semantics to record RMSNorm
    module names, not decoder parent names
  - update `asym_lora_config_from_model`
- Modify `tests/training/test_lf_qwen3_asym_backend.py`:
  - replace the current layer hook test
  - assert exact active RMSNorm module inventory
  - assert no decoder parent hook is installed

Intended code changes:

```python
from asym_gemm.training.decoder_activation_offload import (
    decoder_saved_tensor_offload_module_names,
    install_rmsnorm_activation_offload,
    is_decoder_saved_tensor_offload_wrapper,
    is_rmsnorm_activation_offload_wrapper,
    rmsnorm_activation_offload_module_names,
)
```

```python
def _wrap_qwen3_decoder_rmsnorm_activation_offload_modules(
    model: nn.Module,
    *,
    strict: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    wrapped: list[str] = []
    skipped: list[str] = []
    modules = dict(model.named_modules())

    for layer_name, layer in list(model.named_modules()):
        if not _is_qwen3_decoder_layer_module_name(layer_name, layer):
            continue

        for leaf in ("input_layernorm", "post_attention_layernorm"):
            full_name = f"{layer_name}.{leaf}"
            norm = modules.get(full_name)
            if norm is None:
                skipped.append(f"{full_name}:missing")
                continue
            weight = getattr(norm, "weight", None)
            host_weight = getattr(getattr(norm, "host_weight", None), "weight", None)
            if not isinstance(weight, torch.Tensor) and not isinstance(host_weight, torch.Tensor):
                skipped.append(f"{full_name}:unsupported_norm:{type(norm).__name__}")
                continue
            if not (hasattr(norm, "variance_epsilon") or hasattr(norm, "eps")):
                skipped.append(f"{full_name}:missing_rmsnorm_eps:{type(norm).__name__}")
                continue
            install_rmsnorm_activation_offload(norm, name=full_name)
            wrapped.append(full_name)

    if strict and not wrapped:
        raise RuntimeError("Qwen3 decoder layer activation offload requested but no supported RMSNorms were found")
    return tuple(wrapped), tuple(skipped)
```

```python
layer_act_modules: tuple[str, ...] = ()
layer_act_skipped: tuple[str, ...] = ()
if layer_act_enabled:
    layer_act_modules, layer_act_skipped = _wrap_qwen3_decoder_rmsnorm_activation_offload_modules(
        model,
        strict=strict,
    )

report.layer_act_offload_wrapped = len(layer_act_modules)
report.layer_act_offload_modules = tuple(layer_act_modules)
report.layer_act_offload_skipped = tuple(layer_act_skipped)
setattr(model, "_asym_layer_act_offload_enabled", bool(layer_act_enabled))
setattr(model, "_asym_layer_act_offload_impl", "rmsnorm_active" if layer_act_enabled else "none")
setattr(model, "_asym_layer_act_offload_modules", tuple(layer_act_modules))
setattr(model, "_asym_layer_act_offload_skipped", tuple(layer_act_skipped))
```

```python
def asym_lora_config_from_model(...):
    ...
    layer_act_modules = tuple(getattr(model, "_asym_layer_act_offload_modules", ()))
    if bool(getattr(model, "_asym_layer_act_offload_enabled", False)) or layer_act_modules:
        config["asym_layer_act_offload_enabled"] = bool(...)
        config["asym_layer_act_offload_impl"] = getattr(model, "_asym_layer_act_offload_impl", "unknown")
        config["asym_layer_act_offload_modules"] = list(layer_act_modules)
        config["asym_layer_act_offload_skipped"] = list(...)
        config["asym_rmsnorm_activation_offload_modules"] = list(
            rmsnorm_activation_offload_module_names(model)
        )
        config["asym_decoder_saved_tensor_offload_modules"] = list(
            decoder_saved_tensor_offload_module_names(model)
        )
```

Test pseudocode:

```python
def test_layer_act_wraps_only_qwen3_decoder_rmsnorms(monkeypatch):
    monkeypatch.setenv("ASYMM_LAYER_ACT_OFFLOAD", "true")
    model = FakeQwen3DecoderModel(num_layers=2)
    model, report = apply_lf_asym_lora(..., backend="asym", offload_modules="all")

    assert report.layer_act_offload_enabled is True
    assert report.layer_act_offload_wrapped == 4
    assert report.layer_act_offload_modules == (
        "layers.0.input_layernorm",
        "layers.0.post_attention_layernorm",
        "layers.1.input_layernorm",
        "layers.1.post_attention_layernorm",
    )
    assert is_rmsnorm_activation_offload_wrapper(model.layers[0].input_layernorm)
    assert is_rmsnorm_activation_offload_wrapper(model.layers[0].post_attention_layernorm)
    assert not is_decoder_saved_tensor_offload_wrapper(model.layers[0])
    assert not is_rmsnorm_activation_offload_wrapper(model.layers[0].self_attn)
    assert not is_rmsnorm_activation_offload_wrapper(model.layers[0].mlp)
    assert getattr(model, "_asym_layer_act_offload_impl") == "rmsnorm_active"
```

Ambiguity and risk checks:

- Install after norm replacement, because the real `ASYM_OFFLOAD_MODULES=all`
  path can turn RMSNorms into `AsymFrozenRMSNorm`.
- Do not wrap residual adds. Add backward is gradient routing only and has no
  large saved tensor target in the Stage 0 evidence.
- Do not wrap `self_attn` or `mlp`. Their activation ownership remains in
  `attn_math.md` and `mlp_math.md`.

Validation before Stage 3:

```bash
.venv/bin/python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_layer_act_wraps_only_qwen3_decoder_rmsnorms \
  tests/training/test_lf_qwen3_asym_backend.py::test_qwen3_decoder_layer_activation_offload_requires_policy_none
```

Run the first active candidate on the real workflow:

```bash
OUTPUT_ROOT=outputs/lf_layer_active_stage2 \
OVERWRITE=true \
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=hbm \
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
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf.sh --gpus 0
```

Stage 2 passes only if:

- profile JSON has `config.asym_layer_act_offload_impl == "rmsnorm_active"`;
- active rows exist for every decoder `input_layernorm` and
  `post_attention_layernorm`;
- no row reports `decoder_saved_tensor_offload == true`;
- `reference_fallback_count == 0` if that field is present;
- trainable surface is still LoRA-only.

This stage does not yet accept the design. It only proves the active path is
installed and runs on the real workflow.

## Stage 3: Add Stale-Profile Guards And Profile Summaries

Scope:

- Modify `scripts/lf/run_lf_profiled_train.py`:
  - update `_activation_offload_counters_from_model`
  - add active RMSNorm row counts to the `activation_offload` summary
- Modify `scripts/lf/profile_lora_lf.sh`:
  - update `existing_profile_complete`
  - reject stale `layer_act=true` profiles that lack
    `config.asym_layer_act_offload_impl == "rmsnorm_active"`
- Modify `tests/lf/test_lf_profile_postprocess.py`:
  - verify active RMSNorm summary counts are emitted
- Modify `tests/lf/test_superoffload_backend_scripts.py` or
  `tests/lf/test_asym_cpu_adamw_args.py`:
  - verify stale hook-era `layer_act=true` profile JSON is not treated as
    complete

Intended code changes:

```python
def _activation_offload_counters_from_model() -> dict[str, Any]:
    ...
    active_rmsnorm_rows = []
    decoder_hook_rows = []
    for row in rows:
        stats = row.get("activation_offload_stats")
        if not isinstance(stats, dict):
            continue
        if stats.get("rmsnorm_active_activation_offload"):
            active_rmsnorm_rows.append(row)
        if stats.get("decoder_saved_tensor_offload"):
            decoder_hook_rows.append(row)

    return {
        "available": bool(rows),
        "module_count": len(rows),
        "rows": rows,
        "active_rmsnorm_activation_offload_row_count": len(active_rmsnorm_rows),
        "decoder_saved_tensor_hook_row_count": len(decoder_hook_rows),
        ...
    }
```

```python
# scripts/lf/profile_lora_lf.sh::existing_profile_complete
actual_layeract = normalize_bool(config.get("asymm_layer_act_offload"))
wanted_layeract = normalize_bool(expected_layeract)
if wanted_layeract == "true":
    impl = str(config.get("asym_layer_act_offload_impl", ""))
    if impl != "rmsnorm_active":
        raise SystemExit("profile layer activation offload impl missing or stale")
```

Ambiguity and risk checks:

- Existing artifacts produced before this implementation have the same public
  tuple `none|true|true|true`. The stale-profile guard is required so future
  sweeps do not accidentally reuse hook-era profiles as active-layer profiles.
- If a deliberate hook baseline is needed after Stage 3, use the Stage 0
  baseline artifact or a temporary debug branch. Do not make the hook path the
  default meaning of `ASYMM_LAYER_ACT_OFFLOAD=true`.

Validation before Stage 4:

```bash
bash -n scripts/lf/profile_lora_lf.sh
.venv/bin/python -m pytest -q \
  tests/lf/test_lf_profile_postprocess.py \
  tests/lf/test_superoffload_backend_scripts.py::test_existing_profile_complete_rejects_stale_layer_hook_profile \
  tests/lf/test_asym_cpu_adamw_args.py::test_profile_lora_lf_four_part_layer_axis_dry_run
```

Run the real workflow again with overwrite disabled to exercise the stale
profile path in normal script routing:

```bash
OUTPUT_ROOT=outputs/lf_layer_active_stage3 \
OVERWRITE=true \
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=hbm \
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
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf.sh --gpus 0

OUTPUT_ROOT=outputs/lf_layer_active_stage3 \
OVERWRITE=false \
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=hbm \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYMM_EXP_ACT_POLICIES="none|true|true|true" \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=false \
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
bash scripts/lf/profile_lora_lf.sh --gpus 0
```

Stage 3 passes if the first command produces active RMSNorm summary counts and
the second command reuses only a complete active profile, not a hook-era stale
profile.

## Stage 4: Compare And Accept Or Reject

Scope:

- No functional code changes unless the comparison reveals a correctness or
  performance issue.
- Optional doc update after acceptance:
  - `agent/status.md`
  - `agent/impls/act_offload_ab_testing.md`
  - this file's result notes

Implementation steps:

1. Compare Stage 0 hook baseline `none|true|true|true` against Stage 3 active
   candidate `none|true|true|true`.
2. Also compare Stage 0 no-layer baseline `none|true|true|false` so the final
   result is clear:
   - no layer offload;
   - traditional hook layer offload;
   - active RMSNorm layer offload.
3. Use measured steps only. Ignore warmup rows.
4. Compare both peak allocated HBM and peak reserved HBM.
5. Compare both `forward_backward_milliseconds` and
   `trainer_e2e_step_milliseconds`.

Concrete comparison script:

```bash
.venv/bin/python - <<'PY'
import glob
import json
import statistics
from pathlib import Path

def load_one(root: str, *, layeract: str, impl: str | None = None):
    matches = []
    for path in glob.glob(f"{root}/**/profile.json", recursive=True):
        data = json.loads(Path(path).read_text())
        cfg = data.get("config", {})
        if str(cfg.get("asymm_layer_act_offload")).lower() != layeract:
            continue
        if impl is not None and cfg.get("asym_layer_act_offload_impl") != impl:
            continue
        matches.append((path, data))
    if len(matches) != 1:
        raise SystemExit(f"expected one match for {root} layeract={layeract} impl={impl}, got {len(matches)}")
    return matches[0]

def metrics(data):
    rows = [
        row for row in data.get("step_samples", {}).get("rows", [])
        if not row.get("is_warmup")
    ]
    if not rows:
        raise SystemExit("profile has no measured step_samples rows")
    return {
        "peak_alloc_gib": max(row["peak_allocated_hbm_bytes"] for row in rows) / 1024**3,
        "peak_reserved_gib": max(row["peak_reserved_hbm_bytes"] for row in rows) / 1024**3,
        "fwd_bwd_s": statistics.mean(row["forward_backward_milliseconds"] for row in rows) / 1000.0,
        "e2e_s": statistics.mean(row["trainer_e2e_step_milliseconds"] for row in rows) / 1000.0,
    }

hook_path, hook = load_one("outputs/lf_layer_active_stage0", layeract="true")
active_path, active = load_one("outputs/lf_layer_active_stage3", layeract="true", impl="rmsnorm_active")
no_layer_path, no_layer = load_one("outputs/lf_layer_active_stage0", layeract="false")

for label, path, data in [
    ("no_layer", no_layer_path, no_layer),
    ("hook_baseline", hook_path, hook),
    ("active_candidate", active_path, active),
]:
    print(label, path, metrics(data))

hm = metrics(hook)
am = metrics(active)
alloc_delta = am["peak_alloc_gib"] - hm["peak_alloc_gib"]
reserved_delta = am["peak_reserved_gib"] - hm["peak_reserved_gib"]
e2e_regression = (am["e2e_s"] / hm["e2e_s"]) - 1.0
fwd_bwd_regression = (am["fwd_bwd_s"] / hm["fwd_bwd_s"]) - 1.0
memory_improved = alloc_delta < -0.1 or reserved_delta < -0.1

print("delta_vs_hook", {
    "peak_alloc_gib": alloc_delta,
    "peak_reserved_gib": reserved_delta,
    "e2e_regression_pct": e2e_regression * 100.0,
    "fwd_bwd_regression_pct": fwd_bwd_regression * 100.0,
})

if not memory_improved:
    raise SystemExit("REJECT: active layer offload did not reduce peak HBM by at least 0.1 GiB")
if alloc_delta > 0.5:
    raise SystemExit("REJECT: active layer offload increased peak allocated HBM by more than 0.5 GiB")
if reserved_delta > 0.5:
    raise SystemExit("REJECT: active layer offload increased peak reserved HBM by more than 0.5 GiB")
if e2e_regression > 0.05:
    raise SystemExit("REJECT: active layer offload regressed trainer E2E latency by more than 5%")
if fwd_bwd_regression > 0.05:
    raise SystemExit("REJECT: active layer offload regressed fwd+bwd latency by more than 5%")
print("ACCEPT: active layer offload reduced memory within latency gate")
PY
```

Acceptance criteria:

- Active candidate peak allocated HBM or peak reserved HBM is at least
  `0.1 GiB` lower than the Stage 0 hook baseline. Prefer both to decrease.
- Neither peak allocated HBM nor peak reserved HBM is more than `0.5 GiB`
  higher than the Stage 0 hook baseline.
- Active candidate trainer E2E measured step time is no more than `5%` slower
  than the Stage 0 hook baseline.
- Active candidate forward+backward time is no more than `5%` slower than the
  Stage 0 hook baseline.
- `activation_offload.decoder_saved_tensor_hook_row_count == 0`.
- `activation_offload.active_rmsnorm_activation_offload_row_count == 96` for a
  48-layer Qwen3 model.
- Active row shape/dtype evidence shows original activation dtype
  `[4,4096,2048]`, not hook-saved `float32 [4,4096,2048]`.
- Trainable parameters remain LoRA-only.
- No reference fallback is present.

Risks to watch after Stage 4:

- If active RMSNorm lowers CPU saved bytes but not HBM, the hook baseline may
  already be eliminating the relevant HBM save. In that case this path is only
  accepted if reserved HBM or allocated HBM still drops in the real profile.
- If latency regresses by more than `5%`, the likely issue is RMSNorm backward
  staging or unfused vector/reduction math. The next step would be a fused CUDA
  RMSNorm backward kernel, not an AsymGEMM matrix kernel.
- If `ASYM_OFFLOAD_MODULES` excludes `norms`, validate again. The accepted
  default workflow uses the real script default/target workflow, but the
  wrapper should still work when norm weights remain GPU-resident.

## Decoder-Layer Math Target

These are the only remaining layer targets for this plan:

| Module / boundary | Active target? | Reason |
|---|---:|---|
| `model.layers[i].input_layernorm` | yes | pre-attention RMSNorm needs its input for backward |
| `model.layers[i].post_attention_layernorm` | yes | pre-MLP RMSNorm needs its input for backward |
| residual add after attention | no | backward is gradient routing/addition; no large saved tensor required |
| residual add after MLP | no | backward is gradient routing/addition; no large saved tensor required |
| `self_attn` internals | no | owned by `attn_math.md` / `ASYMM_ATTN_ACT_OFFLOAD` |
| `mlp` / routed experts | no | owned by `mlp_math.md` / `ASYMM_EXPERT_ACT_OFFLOAD` |

Notation:

```text
B = batch
T = sequence length
M = B * T
H = hidden size

X0 = decoder-layer input / first residual source       [B,T,H]
N0 = input RMSNorm output, attention input             [B,T,H]
A  = attention output                                  [B,T,H]
X1 = post-attention residual, second residual source   [B,T,H]
N1 = post-attention RMSNorm output, MLP input          [B,T,H]
M1 = MLP output                                        [B,T,H]
Y  = decoder-layer output                              [B,T,H]

gamma0 = input_layernorm.weight                        [H]
gamma1 = post_attention_layernorm.weight               [H]
eps    = RMSNorm epsilon
```

RMSNorm:

```text
rms(x) = sqrt(mean_h(x_h^2) + eps)                     [B,T,1]
inv(x) = 1 / rms(x)                                    [B,T,1]
rmsnorm(x; gamma) = x * inv(x) * gamma                 [B,T,H]
```

RMSNorm backward:

```text
u = dN * gamma                                         [B,T,H]
c = mean_h(u * x)                                      [B,T,1]
dx = inv(x) * u - x * inv(x)^3 * c                    [B,T,H]
dgamma = sum_{b,t}(dN * x * inv(x))                    [H]
```

Forward active schedule:

```text
X0 = hidden_states                                      # [B,T,H] HBM
X0_cpu = offload(X0)                                    # CPU owner for input RMSNorm backward
N0 = rmsnorm(X0; gamma0)                                # HBM

A = attention(N0)                                       # attn active path owns attn internals
X1 = X0 + A                                             # residual add, no offloaded save

X1_cpu = offload(X1)                                    # CPU owner for post-attn RMSNorm backward
N1 = rmsnorm(X1; gamma1)                                # HBM

M1 = qwen3_moe(N1)                                      # expert active path owns MLP internals
Y = X1 + M1                                             # residual add, no offloaded save
```

Backward active schedule:

```text
dX1_from_final = dY
dM1 = dY
dN1 = qwen3_moe_backward(dM1)

X1_stage = stage(X1_cpu)                                # immediate use only
dX1_norm, dgamma1 = rmsnorm_backward(dN1, X1_stage, gamma1, eps)
release(X1_stage, X1_cpu)
dX1 = dX1_from_final + dX1_norm

dX0_from_residual = dX1
dA = dX1
dN0 = attention_backward(dA)

X0_stage = stage(X0_cpu)                                # immediate use only
dX0_norm, dgamma0 = rmsnorm_backward(dN0, X0_stage, gamma0, eps)
release(X0_stage, X0_cpu)
dX0 = dX0_from_residual + dX0_norm
```

RMSNorm is vector/reduction math, not a GEMM. The AsymGEMM-style requirement
for this layer boundary is explicit ownership, minimal CPU save, just-in-time
stage, and immediate release. Attention and MLP remain the places where the
actual AsymGEMM matrix kernels apply.
