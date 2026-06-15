# Active Qwen3 Decoder-Layer Activation Offload Plan

Status: ON HOLD. Do not implement this plan unless new production profiling
shows decoder-layer RMSNorm staging is a material peak-HBM source, or the goal
changes from peak-HBM reduction to CPU-transfer / CPU-activation reduction.

Reason for hold:

- The only remaining in-layer targets outside attention and MLP/expert are
  `input_layernorm` and `post_attention_layernorm`. RMSNorm is vector/reduction
  math, not a GEMM, so this is not an AsymGEMM kernel optimization.
- The current `DecoderSavedTensorOffloadWrapper` already does the simple
  offload-and-stage behavior. The active RMSNorm path can only make that more
  explicit and save/stage the original RMSNorm input dtype instead of PyTorch's
  float32 RMSNorm temporary.
- For the target b4/s4096 Qwen3-30B shape:

  ```text
  B*T*H = 4*4096*2048 = 33,554,432 elements

  current hook staged tensor:
    float32 [4,4096,2048] = 33,554,432 * 4 bytes = 0.125 GiB

  active RMSNorm staged tensor:
    bf16 [4,4096,2048] = 33,554,432 * 2 bytes = 0.0625 GiB

  best local peak-HBM reduction:
    0.125 GiB - 0.0625 GiB = 0.0625 GiB
  ```

- Because the current hook stages only one such tensor at a time, the expected
  global peak-HBM reduction is only `0` to `0.0625 GiB`. Peak reserved HBM is
  likely `0 GiB` better because allocator reservation may not shrink from a
  transient 64 MiB reduction.
- The current profile shows about `356.25 GiB` of decoder-hook norm
  offload/stage traffic over the measured run, so active bf16 RMSNorm saves
  could reduce CPU/H2D/D2H traffic. That is useful diagnostics, but it is not
  the required peak-HBM win.
- Therefore this is unlikely to beat the current simple offload/stage path on
  the actual acceptance target: lower real production peak HBM without too much
  latency regression.

Reference goal if this is resumed later: add an `asym_layer_offload` selector
so `ASYMM_LAYER_ACT_OFFLOAD=true` can run either the current hook-only layer
offload path or the active backfetch path for the remaining Qwen3 decoder-layer
modules not already owned by `attn_math.md` or `mlp_math.md`.

Acceptance is profile-based, not toy-shape based. The canonical production
profiling script is
`/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh`;
all memory and latency numbers that drive design decisions must come from that
real LF LoRA workflow, not microbenchmarks or toy settings. The change is
accepted only if the real workflow shows lower peak HBM on at least one
measured HBM metric versus the current traditional saved-tensor hook baseline,
does not increase any peak HBM metric by more than `0.5 GiB`, and does not
regress latency beyond the stage gate below.

The production memory envelope is the combined expert+attention+layer
activation-offload profile, not a layer-only microbenchmark. Today that is the
`none|true|true|true` real profile. If that run is around `34 GiB` for the
chosen peak HBM metric, the active implementation for the remaining layer
modules must land in the same combined envelope with at most `+0.5 GiB`
fluctuation. Use the exact `profile.json` fields
`peak_allocated_hbm_bytes` and `peak_reserved_hbm_bytes` as source of truth if
the summary label differs.

Control semantics:

```text
ASYMM_LAYER_ACT_OFFLOAD=false
  no decoder-layer activation offload

ASYMM_LAYER_ACT_OFFLOAD=true, asym_layer_offload=false
  current traditional decoder saved-tensor hook offload/backfetch
  implementation: DecoderSavedTensorOffloadWrapper

ASYMM_LAYER_ACT_OFFLOAD=true, asym_layer_offload=true
  new active AsymGEMM-style layer backfetching for every remaining
  decoder-layer module outside attention and experts
  source-verified active targets: input_layernorm and
  post_attention_layernorm
```

Use `--asym-layer-offload true|false` in `scripts/lf/profile_lora_lf.sh`; the
profile config field is `asym_layer_offload`. Default it to `false` until the
active path passes Stage 4, so existing `ASYMM_LAYER_ACT_OFFLOAD=true` behavior
continues to mean the current hook path unless explicitly toggled.

The current `none|true|true|true` tuple is important:

```text
EXPERT_SELECTION_POLICY|ASYMM_EXPERT_ACT_OFFLOAD|ASYMM_ATTN_ACT_OFFLOAD|ASYMM_LAYER_ACT_OFFLOAD
none|true|true|true
```

Capture it with `asym_layer_offload=false` before adding the active path. At
that point it is the traditional layer saved-tensor offload/staging baseline.
After the active layer implementation lands, rerun the same tuple with
`asym_layer_offload=true` and compare active backfetching against that baseline
artifact. The target is not "RMSNorm offload works in isolation"; the target is
"expert+attention+active-layer offload together stays at the current production
HBM envelope, with no more than `0.5 GiB` upward fluctuation on any peak HBM
metric."

Resolved facts used by this plan:

- Local code currently installs `DecoderSavedTensorOffloadWrapper` from
  `asym_gemm/training/decoder_activation_offload.py` through
  `asym_gemm/integrations/lf.py::_wrap_qwen3_decoder_saved_tensor_offload_modules`.
- The current hook profile offloads float32 `[4,4096,2048]` tensors, with most
  decoder layers peaking at `0.5 GiB` CPU-owned saved tensors per layer.
- The active path does not intentionally reduce model precision. It saves the
  original RMSNorm input dtype, which is bf16 in the target LoRA workflow, and
  recomputes the float32 RMSNorm temporaries during backward. The hook's
  float32 saved tensors are PyTorch RMSNorm intermediates, not higher-precision
  model activations that must be preserved as stored tensors.
- Qwen3 RMSNorm casts hidden states to float32, computes the RMS variance over
  the hidden dimension, multiplies by `rsqrt(variance + eps)`, then applies the
  RMSNorm weight. This was checked against the Hugging Face
  [`Qwen3RMSNorm` source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py).
- Local source inspection confirms the Qwen3/Qwen3-MoE decoder-layer body is:
  `input_layernorm`, `self_attn`, residual add, `post_attention_layernorm`,
  `mlp`, residual add. LlamaFactory does not add another Qwen3-30B-A3B
  decoder-layer module outside that Hugging Face structure. `q_norm` and
  `k_norm` live inside attention, while MoE routing/gating lives inside `mlp`.
- `model.embed_tokens`, the final `model.norm`, and `lm_head` are outside the
  decoder layer. They are not targets for `asym_layer_offload`; if any of them
  becomes a material HBM source, handle it as a separate model-boundary or loss
  path optimization, not as remaining decoder-layer offload.
- The active RMSNorm path is not an AsymGEMM kernel optimization. It only uses
  the same explicit ownership, CPU save, just-in-time stage, and immediate
  release pattern used by the AsymGEMM activation-offload paths. Therefore the
  expected peak-HBM improvement is bounded by the currently staged norm tensor,
  not by the sum of all norm tensors across layers.
- In the current b4/s4096 hook profile, each decoder hook stages at most one
  float32 `[4,4096,2048]` tensor live on HBM, or `0.125 GiB`. The active path
  would stage the original bf16 RMSNorm input, or `0.0625 GiB`, so the local
  peak-stage saving from norms is about `0.0625 GiB` per live staged tensor.
  If the global peak is dominated by attention, experts, logits, or allocator
  reservation, measured peak HBM may improve by `0 GiB`.
- The possible benefit is more likely lower CPU-owned activation bytes and
  lower D2H/H2D traffic for decoder norms. The existing profile shows about
  `356.25 GiB` of decoder-hook offload/stage traffic over the measured run;
  active bf16 RMSNorm input saves would nominally halve that part. Do not count
  this traffic reduction as peak-HBM reduction.
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
  - `scripts/lf/run_lf_lora_sft.sh`
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
   ASYMM_EXP_ACT_POLICIES=none|true|true|true
   asym_layer_offload=false
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

4. Record the combined activation-offload production memory envelope from the
   same `none|true|true|true` profile:

   ```python
   measured = [
       row for row in profile["step_samples"]["rows"]
       if not row.get("is_warmup")
   ]
   combined_peak_alloc_gib = max(row["peak_allocated_hbm_bytes"] for row in measured) / 1024**3
   combined_peak_reserved_gib = max(row["peak_reserved_hbm_bytes"] for row in measured) / 1024**3
   print("combined exp+attn+layer envelope", {
       "peak_alloc_gib": combined_peak_alloc_gib,
       "peak_reserved_gib": combined_peak_reserved_gib,
       "max_allowed_active_peak_alloc_gib": combined_peak_alloc_gib + 0.5,
       "max_allowed_active_peak_reserved_gib": combined_peak_reserved_gib + 0.5,
   })
   ```

   This is the target envelope for the active remaining-layer implementation.
   If the current profile reports about `34 GiB` for the metric being used in
   analysis, the active implementation must stay within that number plus
   `0.5 GiB` when expert, attention, and layer activation offload are enabled
   together.

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
ASYM_LAYER_OFFLOAD=false \
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

Stage 0 passes only if both profiles finish through
`/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh`,
both use the real `Qwen/Qwen3-30B-A3B` LF LoRA workflow, the
`none|true|true|true` artifact contains decoder saved-tensor hook rows, and
the combined expert+attention+layer HBM envelope is recorded from measured
steps. This recorded envelope, not a toy-profile number, is the memory target
for the active remaining-layer implementation.

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
    comparison and for `asym_layer_offload=false`, not for the active selector
    path
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
        self.cast_output_to_input_dtype = type(module).__name__ == "AsymFrozenRMSNorm"
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
        normalized = x_f * inv
        if wrapper.cast_output_to_input_dtype:
            y = (normalized * weight.to(dtype=torch.float32)).to(dtype=input_dtype)
        else:
            y = weight * normalized.to(dtype=input_dtype)

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
  input tensor dtype. In the target Qwen3 LoRA workflow that input is bf16, so
  this is recompute from the actual forward input, not a precision downgrade
  from an fp32 activation. It must still be tested against autograd output,
  input grad, and weight grad.
- The active wrapper must preserve the installed module's output dtype
  semantics: HF-style Qwen3 RMSNorm multiplies by its weight after casting the
  normalized activation back to input dtype, while local `AsymFrozenRMSNorm`
  returns the final result in the input dtype.
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
ASYM_LAYER_OFFLOAD=false \
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

## Stage 2: Add `asym_layer_offload` Selector And Install Active Wrappers

Scope:

- Modify `asym_gemm/integrations/lf.py`:
  - imports from `asym_gemm.training.decoder_activation_offload`
  - keep `_wrap_qwen3_decoder_saved_tensor_offload_modules` for
    `asym_layer_offload=false`
  - add `_wrap_qwen3_decoder_rmsnorm_activation_offload_modules` for
    `asym_layer_offload=true`
  - add `_asym_layer_offload_enabled`
  - update `apply_lf_asym_lora` near the existing `layer_act_enabled` block
  - update `LFAsymReport.layer_act_offload_modules` semantics to record RMSNorm
    module names in active mode and decoder parent names in hook mode
  - update `asym_lora_config_from_model`
- Modify `scripts/lf/profile_lora_lf.sh`:
  - add `ASYM_LAYER_OFFLOAD=${ASYM_LAYER_OFFLOAD:-false}`
  - add `--asym-layer-offload true|false`
  - forward `ASYM_LAYER_OFFLOAD` and
    `ASYM_GEMM_LF_CONFIG_ASYM_LAYER_OFFLOAD` into jobs
  - include the selector in run labels, `jobs.tsv`, and
    `existing_profile_complete`
- Modify `scripts/lf/run_lf_lora_sft.sh`:
  - accept and forward `ASYM_LAYER_OFFLOAD`
- Modify `scripts/lf/run_lf_profiled_train.py`:
  - record `config["asym_layer_offload"]`
- Modify `tests/training/test_lf_qwen3_asym_backend.py`:
  - keep a hook-mode test for `asym_layer_offload=false`
  - add an active-mode test for `asym_layer_offload=true`
  - assert exact active RMSNorm module inventory
  - assert no decoder parent hook is installed
- Modify `tests/lf/test_asym_cpu_adamw_args.py`:
  - verify the dry-run command forwards and labels both selector values

Intended code changes:

```python
from asym_gemm.training.decoder_activation_offload import (
    decoder_saved_tensor_offload_module_names,
    install_decoder_saved_tensor_offload,
    install_rmsnorm_activation_offload,
    is_decoder_saved_tensor_offload_wrapper,
    is_rmsnorm_activation_offload_wrapper,
    rmsnorm_activation_offload_module_names,
)
```

```python
def _asym_layer_offload_enabled() -> bool:
    return _env_true(os.environ.get("ASYM_LAYER_OFFLOAD")) or _env_true(
        os.environ.get("ASYM_GEMM_LF_CONFIG_ASYM_LAYER_OFFLOAD")
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
layer_act_impl = "none"
asym_layer_enabled = _asym_layer_offload_enabled()
if layer_act_enabled:
    if asym_layer_enabled:
        layer_act_impl = "rmsnorm_active"
        layer_act_modules, layer_act_skipped = _wrap_qwen3_decoder_rmsnorm_activation_offload_modules(
            model,
            strict=strict,
        )
    else:
        layer_act_impl = "decoder_saved_tensor_hook"
        layer_act_modules, layer_act_skipped = _wrap_qwen3_decoder_saved_tensor_offload_modules(
            model,
            strict=strict,
        )

report.layer_act_offload_wrapped = len(layer_act_modules)
report.layer_act_offload_modules = tuple(layer_act_modules)
report.layer_act_offload_skipped = tuple(layer_act_skipped)
setattr(model, "_asym_layer_act_offload_enabled", bool(layer_act_enabled))
setattr(model, "_asym_layer_offload_enabled", bool(asym_layer_enabled))
setattr(model, "_asym_layer_act_offload_impl", layer_act_impl)
setattr(model, "_asym_layer_act_offload_modules", tuple(layer_act_modules))
setattr(model, "_asym_layer_act_offload_skipped", tuple(layer_act_skipped))
```

```python
def asym_lora_config_from_model(...):
    ...
    layer_act_modules = tuple(getattr(model, "_asym_layer_act_offload_modules", ()))
    if bool(getattr(model, "_asym_layer_act_offload_enabled", False)) or layer_act_modules:
        config["asym_layer_act_offload_enabled"] = bool(...)
        config["asym_layer_offload"] = bool(getattr(model, "_asym_layer_offload_enabled", False))
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
def test_layer_act_hook_mode_keeps_decoder_saved_tensor_wrapper(monkeypatch):
    monkeypatch.setenv("ASYMM_LAYER_ACT_OFFLOAD", "true")
    monkeypatch.setenv("ASYM_LAYER_OFFLOAD", "false")
    model = FakeQwen3DecoderModel(num_layers=2)
    model, report = apply_lf_asym_lora(..., backend="asym", offload_modules="all")

    assert report.layer_act_offload_enabled is True
    assert report.layer_act_offload_wrapped == 2
    assert report.layer_act_offload_modules == ("layers.0", "layers.1")
    assert is_decoder_saved_tensor_offload_wrapper(model.layers[0])
    assert not is_rmsnorm_activation_offload_wrapper(model.layers[0].input_layernorm)
    assert getattr(model, "_asym_layer_offload_enabled") is False
    assert getattr(model, "_asym_layer_act_offload_impl") == "decoder_saved_tensor_hook"
```

```python
def test_asym_layer_offload_wraps_only_qwen3_decoder_rmsnorms(monkeypatch):
    monkeypatch.setenv("ASYMM_LAYER_ACT_OFFLOAD", "true")
    monkeypatch.setenv("ASYM_LAYER_OFFLOAD", "true")
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
    assert getattr(model, "_asym_layer_offload_enabled") is True
    assert getattr(model, "_asym_layer_act_offload_impl") == "rmsnorm_active"
```

Ambiguity and risk checks:

- Install after norm replacement, because the real `ASYM_OFFLOAD_MODULES=all`
  path can turn RMSNorms into `AsymFrozenRMSNorm`.
- Do not wrap residual adds. Add backward is gradient routing only and has no
  large saved tensor target in the Stage 0 evidence.
- Do not wrap `self_attn` or `mlp`. Their activation ownership remains in
  `attn_math.md` and `mlp_math.md`.
- `asym_layer_offload=true` means the active path owns all remaining
  decoder-layer activation targets outside attention and experts. The current
  real profile identifies RMSNorm boundary tensors as that remaining set. If a
  later production profile exposes another non-attention, non-expert saved
  tensor target, add it to the active selector before accepting Stage 4.

Validation before Stage 3:

```bash
.venv/bin/python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_layer_act_hook_mode_keeps_decoder_saved_tensor_wrapper \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_layer_offload_wraps_only_qwen3_decoder_rmsnorms \
  tests/training/test_lf_qwen3_asym_backend.py::test_qwen3_decoder_layer_activation_offload_requires_policy_none \
  tests/lf/test_asym_cpu_adamw_args.py::test_profile_lora_lf_dry_run_labels_asym_layer_offload_modes
```

Run the first active candidate on the real workflow:

```bash
OUTPUT_ROOT=outputs/lf_layer_active_stage2 \
OVERWRITE=true \
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=hbm \
ASYM_LAYER_OFFLOAD=true \
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

- profile JSON has `config.asym_layer_offload == true`;
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
  - compare expected `asym_layer_offload` against
    `config.asym_layer_offload`
  - reject stale active profiles when `asym_layer_offload=true` and
    `config.asym_layer_act_offload_impl != "rmsnorm_active"`
  - reject stale hook profiles when `asym_layer_offload=false` and
    `config.asym_layer_act_offload_impl != "decoder_saved_tensor_hook"`
- Modify `tests/lf/test_lf_profile_postprocess.py`:
  - verify active RMSNorm summary counts are emitted
- Modify `tests/lf/test_superoffload_backend_scripts.py` or
  `tests/lf/test_asym_cpu_adamw_args.py`:
  - verify stale hook-era `layer_act=true` profile JSON is not treated as a
    complete active profile
  - verify hook-era `layer_act=true` profile JSON is still valid for
    `asym_layer_offload=false`

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
# shell signature adds:
#   expected_asym_layer_offload="${13:-}"
# and every caller passes "${ASYM_LAYER_OFFLOAD}" after the existing
# expert/attention/layer activation-offload expectations.
actual_layeract = normalize_bool(config.get("asymm_layer_act_offload"))
wanted_layeract = normalize_bool(expected_layeract)
if wanted_layeract == "true":
    actual_asym_layer = normalize_bool(config.get("asym_layer_offload"))
    wanted_asym_layer = normalize_bool(expected_asym_layer_offload)
    if actual_asym_layer != wanted_asym_layer:
        raise SystemExit(
            "profile asym_layer_offload mismatch: "
            f"expected {wanted_asym_layer}, got {actual_asym_layer or '<missing>'}"
        )
    impl = str(config.get("asym_layer_act_offload_impl", ""))
    if wanted_asym_layer == "true" and impl != "rmsnorm_active":
        raise SystemExit("profile active layer offload impl missing or stale")
    if wanted_asym_layer == "false" and impl != "decoder_saved_tensor_hook":
        raise SystemExit("profile hook layer offload impl missing or stale")
```

Ambiguity and risk checks:

- Existing artifacts produced before this implementation have the same public
  tuple `none|true|true|true`. The `asym_layer_offload` selector is required
  so future sweeps do not accidentally reuse hook-era profiles as active-layer
  profiles.
- Hook baselines remain valid and rerunnable after Stage 3 by setting
  `asym_layer_offload=false`; active candidates require
  `asym_layer_offload=true`.

Validation before Stage 4:

```bash
bash -n scripts/lf/profile_lora_lf.sh
.venv/bin/python -m pytest -q \
  tests/lf/test_lf_profile_postprocess.py \
  tests/lf/test_superoffload_backend_scripts.py::test_existing_profile_complete_rejects_hook_profile_for_asym_layer_offload_true \
  tests/lf/test_superoffload_backend_scripts.py::test_existing_profile_complete_accepts_hook_profile_for_asym_layer_offload_false \
  tests/lf/test_asym_cpu_adamw_args.py::test_profile_lora_lf_four_part_layer_axis_dry_run
```

Run the real workflow in both selector modes. This is the production A/B for
the new argument:

1. Hook baseline under the new selector:

```bash
OUTPUT_ROOT=outputs/lf_layer_active_stage3_hook \
OVERWRITE=true \
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=hbm \
ASYM_LAYER_OFFLOAD=false \
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

2. Active candidate:

```bash
OUTPUT_ROOT=outputs/lf_layer_active_stage3_active \
OVERWRITE=true \
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=hbm \
ASYM_LAYER_OFFLOAD=true \
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

3. Active candidate reuse check:

```bash
OUTPUT_ROOT=outputs/lf_layer_active_stage3_active \
OVERWRITE=false \
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=hbm \
ASYM_LAYER_OFFLOAD=true \
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

Stage 3 passes if the hook run records
`config.asym_layer_offload == false` and
`config.asym_layer_act_offload_impl == "decoder_saved_tensor_hook"`, the active
run records `config.asym_layer_offload == true` and
`config.asym_layer_act_offload_impl == "rmsnorm_active"`, and the reuse check
reuses only a complete active profile, not a hook-era stale profile.

## Stage 4: Compare And Accept Or Reject

Scope:

- No functional code changes unless the comparison reveals a correctness or
  performance issue.
- Optional doc update after acceptance:
  - `agent/status.md`
  - `agent/impls/act_offload_ab_testing.md`
  - this file's result notes

Implementation steps:

1. Compare the Stage 3 hook baseline `none|true|true|true` with
   `asym_layer_offload=false` against the Stage 3 active candidate
   `none|true|true|true` with `asym_layer_offload=true`.
2. Also compare Stage 0 no-layer baseline `none|true|true|false` so the final
   result is clear:
   - no layer offload;
   - traditional hook layer offload with `asym_layer_offload=false`;
   - active RMSNorm layer offload with `asym_layer_offload=true`.
3. Use measured steps only. Ignore warmup rows.
4. Compare both peak allocated HBM and peak reserved HBM.
5. Compare both `forward_backward_milliseconds` and
   `trainer_e2e_step_milliseconds`.
6. Treat the Stage 3 hook-mode `none|true|true|true,
   asym_layer_offload=false` measured HBM values as the production combined
   activation-offload envelope. The Stage 3 active candidate must stay within
   `+0.5 GiB` of that envelope on both peak allocated and peak reserved HBM.
   This is the real acceptance target for the remaining modules together.

Concrete comparison script:

```bash
.venv/bin/python - <<'PY'
import glob
import json
import statistics
from pathlib import Path

def norm_bool(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return "true"
    if text in {"0", "false", "no", "n", "off"}:
        return "false"
    return ""

def load_one(root: str, *, layeract: str, asym_layer: str | None = None, impl: str | None = None):
    matches = []
    for path in glob.glob(f"{root}/**/profile.json", recursive=True):
        data = json.loads(Path(path).read_text())
        cfg = data.get("config", {})
        if norm_bool(cfg.get("asymm_layer_act_offload")) != layeract:
            continue
        if asym_layer is not None and norm_bool(cfg.get("asym_layer_offload")) != asym_layer:
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

hook_path, hook = load_one(
    "outputs/lf_layer_active_stage3_hook",
    layeract="true",
    asym_layer="false",
    impl="decoder_saved_tensor_hook",
)
active_path, active = load_one(
    "outputs/lf_layer_active_stage3_active",
    layeract="true",
    asym_layer="true",
    impl="rmsnorm_active",
)
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
print("production_envelope", {
    "baseline_combined_peak_alloc_gib": hm["peak_alloc_gib"],
    "baseline_combined_peak_reserved_gib": hm["peak_reserved_gib"],
    "max_active_peak_alloc_gib": hm["peak_alloc_gib"] + 0.5,
    "max_active_peak_reserved_gib": hm["peak_reserved_gib"] + 0.5,
    "active_peak_alloc_gib": am["peak_alloc_gib"],
    "active_peak_reserved_gib": am["peak_reserved_gib"],
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

- The final acceptance profile is the real
  `/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh`
  production LF LoRA run, not a toy setting.
- The active candidate is evaluated as the combined
  expert+attention+layer-activation-offload profile. If the Stage 3 hook-mode combined
  baseline is around `34 GiB` for the HBM metric used in analysis, the active
  candidate must stay in the same envelope with at most `+0.5 GiB`
  fluctuation on that metric.
- Active candidate peak allocated HBM or peak reserved HBM is at least
  `0.1 GiB` lower than the Stage 3 hook baseline. Prefer both to decrease.
- Neither peak allocated HBM nor peak reserved HBM is more than `0.5 GiB`
  higher than the Stage 3 hook baseline.
- Active candidate trainer E2E measured step time is no more than `5%` slower
  than the Stage 3 hook baseline.
- Active candidate forward+backward time is no more than `5%` slower than the
  Stage 3 hook baseline.
- `activation_offload.decoder_saved_tensor_hook_row_count == 0`.
- `activation_offload.active_rmsnorm_activation_offload_row_count == 96` for a
  48-layer Qwen3 model.
- Active row shape/dtype evidence shows the original RMSNorm input dtype
  `[4,4096,2048]`, not hook-saved float32 RMSNorm temporaries.
- Trainable parameters remain LoRA-only.
- No reference fallback is present.

Risks to watch after Stage 4:

- If active RMSNorm lowers CPU saved bytes but not HBM, the hook baseline may
  already be eliminating the relevant HBM save. In that case this path is only
  accepted if reserved HBM or allocated HBM still drops in the real profile.
- If active RMSNorm only reduces CPU traffic or host activation bytes and does
  not reduce peak allocated/reserved HBM, treat that as a useful diagnostic but
  reject it as a peak-HBM optimization unless the acceptance target is changed.
- If latency regresses by more than `5%`, the likely issue is RMSNorm backward
  staging or unfused vector/reduction math. The next step would be a fused CUDA
  RMSNorm backward kernel, not an AsymGEMM matrix kernel.
- If `ASYM_OFFLOAD_MODULES` excludes `norms`, validate again. The accepted
  default workflow uses the real script default/target workflow, but the
  wrapper should still work when norm weights remain GPU-resident.

## Decoder-Layer Math Target

These are the only remaining layer targets for this plan. They are derived
from the explicit Hugging Face Qwen3/Qwen3-MoE decoder-layer forward and the
active LlamaFactory Qwen3-30B-A3B path, not inferred from profiling:

| Module / boundary | Active target? | Reason |
|---|---:|---|
| `model.layers[i].input_layernorm` | yes | pre-attention RMSNorm needs its input for backward |
| `model.layers[i].post_attention_layernorm` | yes | pre-MLP RMSNorm needs its input for backward |
| residual add after attention | no | backward is gradient routing/addition; no large saved tensor required |
| residual add after MLP | no | backward is gradient routing/addition; no large saved tensor required |
| `self_attn` internals | no | owned by `attn_math.md` / `ASYMM_ATTN_ACT_OFFLOAD` |
| `mlp` / routed experts | no | owned by `mlp_math.md` / `ASYMM_EXPERT_ACT_OFFLOAD` |
| `model.embed_tokens` | no | outside decoder layers; frozen embedding boundary in the target LoRA workflow |
| final model norm outside decoder layers | no | not inside the decoder layer; handle separately if it ever becomes material |
| `lm_head` / loss logits | no | outside decoder layers; handle as a separate head/loss optimization if material |

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
