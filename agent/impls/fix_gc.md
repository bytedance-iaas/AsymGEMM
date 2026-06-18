# ASYMM_LAYER_GC implementation plan

Goal: add a fifth activation tuple field so the LF profiling interface can compare the current layer saved-tensor offload path against a new layer-shell recompute path:

```text
EXPERT_SELECTION_POLICY|ASYMM_EXPERT_ACT_OFFLOAD|ASYMM_ATTN_ACT_OFFLOAD|ASYMM_LAYER_ACT_OFFLOAD|ASYMM_LAYER_GC

current: none|true|true|true|false
new:     none|true|true|false|true
```

`ASYMM_LAYER_GC=true` means recompute/checkpoint only decoder-layer shell work outside the core modules already owned by AsymGEMM: norms, residual glue, and model-family layer forwarding. It must not use full `gc-layer`, because full `gc-layer` reruns attention and experts.

All real memory/latency acceptance numbers must come from:

```bash
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Toy/unit tests are only correctness checks for the wrapper and script plumbing. They do not prove memory efficiency. Every real profiling command in this plan must pin both memory and CPU nodes to CPU NUMA nodes `0,1`:

```bash
NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh ...
```

Do not use GPU-adjacent nodes such as `2,10,18,26` for `NUMACTL_MEMBIND` or `NUMACTL_CPUNODEBIND`.

## Stage 0: command interface and profile plumbing first

Files to modify:

- `scripts/lf/profile_lora_lf.sh`
- `scripts/lf/run_lf_lora_sft.sh`
- `scripts/lf/postprocess_lf_profile_artifacts.py`
- `scripts/plotting/plot_activation_recompute_sweep.py`
- `scripts/plotting/plot_lf_memory_breakdown.py`
- `scripts/lf/validate_lf_memory_capacity_schema.py` only if it validates config keys for activation axes

Concrete changes:

1. Extend `ASYMM_EXP_ACT_POLICIES` to exactly five fields.

   In `profile_lora_lf.sh`, update comments, usage text, default, parse errors, and examples:

   ```bash
   ASYMM_EXP_ACT_POLICIES=${ASYMM_EXP_ACT_POLICIES:-"none|true|true|true|false"}
   # Format:
   # policy|expert_act|attn_act|layer_act|layer_gc
   ```

2. Add a tag helper:

   ```bash
   layergc_tag() {
     case "$(bool_value "$1")" in
       true) printf 'layergc1\n' ;;
       false) printf 'layergc0\n' ;;
     esac
   }
   ```

3. Replace `parse_exp_act_policy_tuple()` with a 5-field parser.

   Pseudocode:

   ```bash
   parse_exp_act_policy_tuple() {
     local raw="$1"
     local policy_part expact_part attnact_part layeract_part layergc_part
     local policy expact attnact layeract layergc
     local -a fields

     IFS='|' read -r -a fields <<< "${raw}"
     ((${#fields[@]} == 5)) || die \
       "ASYMM_EXP_ACT_POLICIES item must be policy|expert_act|attn_act|layer_act|layer_gc, got '${raw}'"

     policy_part="${fields[0]}"
     expact_part="${fields[1]}"
     attnact_part="${fields[2]}"
     layeract_part="${fields[3]}"
     layergc_part="${fields[4]}"

     [[ -n "${policy_part}" && -n "${expact_part}" && -n "${attnact_part}" && -n "${layeract_part}" && -n "${layergc_part}" ]] || die \
       "empty policy/activation/layer_gc value in ASYMM_EXP_ACT_POLICIES item '${raw}'"

     policy="$(normalize_expert_policy "${policy_part}")"
     expact="$(bool_value "${expact_part}")"
     attnact="$(bool_value "${attnact_part}")"
     layeract="$(bool_value "${layeract_part}")"
     layergc="$(bool_value "${layergc_part}")"

     # First implementation keeps layer saved-tensor offload and layer shell GC mutually exclusive.
     if [[ "${layeract}" == "true" && "${layergc}" == "true" ]]; then
       die "ASYMM_LAYER_ACT_OFFLOAD and ASYMM_LAYER_GC must not both be true in '${raw}'"
     fi

     # Keep the target design clean: layer GC is a separate bool, not a gc-* expert policy.
     if [[ "${layergc}" == "true" && "${policy}" != "none" ]]; then
       die "ASYMM_LAYER_GC requires expert policy none, got '${raw}'"
     fi

     if [[ ( "${expact}" == "true" || "${attnact}" == "true" || "${layeract}" == "true" ) && "${policy}" != "none" ]]; then
       die "activation offload tuples must use policy none, got '${raw}'"
     fi

     printf '%s|%s|%s|%s|%s\n' "${policy}" "${expact}" "${attnact}" "${layeract}" "${layergc}"
   }
   ```

4. Thread the fifth field through every script variable and run label.

   Required `profile_lora_lf.sh` edits:

   - Add `layergc_values` from field 5.
   - Add `ASYMM_LAYER_GC="${layergc_values[0]}"`.
   - Add `layergc_label="$(layergc_tag "${ASYMM_LAYER_GC}")"`.
   - Include `layergc_label` in `job_root_path()`, `run_id`, status echo, jobs TSV if activation axes are recorded there, skip messages, plot labels, and any source-materialization matching keys.
   - Add `ASYMM_LAYER_GC` to dynamic locals in `job_root_path()` helpers and `run_job()`.
   - Pass `ASYMM_LAYER_GC` and `ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_GC` into the run env.
   - Extend `existing_profile_complete()` / `job_profile_complete()` argument lists and inline Python validation to check:

     ```python
     actual_layergc = normalize_bool(
         config.get("asymm_layer_gc", config.get("asym_layer_glue_gc_enabled"))
     )
     ```

   - Reject global `recomp` if any of `expact`, `attnact`, `layeract`, or `layergc` is true:

     ```bash
     if ! is_policy_independent_backend "${backend}" && [[ "${recompute}" == "recomp" && (
       "${ASYMM_EXPERT_ACT_OFFLOAD}" == "true" ||
       "${ASYMM_ATTN_ACT_OFFLOAD}" == "true" ||
       "${ASYMM_LAYER_ACT_OFFLOAD}" == "true" ||
       "${ASYMM_LAYER_GC}" == "true"
     ) ]]; then
       die "activation offload/layer GC tuples must use backend recompute=norecomp"
     fi
     ```

5. Add `ASYMM_LAYER_GC` to `run_lf_lora_sft.sh`.

   Required edits:

   ```bash
   ASYMM_LAYER_GC=${ASYMM_LAYER_GC:-false}
   ```

   Add bool validation beside `ASYMM_LAYER_ACT_OFFLOAD`:

   ```bash
   case "${ASYMM_LAYER_GC,,}" in
     1|true|yes|y|on) ASYMM_LAYER_GC=true; LAYER_GC_TAG=layergc1 ;;
     0|false|no|n|off) ASYMM_LAYER_GC=false; LAYER_GC_TAG=layergc0 ;;
     *) echo "ASYMM_LAYER_GC must be true or false, got '${ASYMM_LAYER_GC}'" >&2; exit 2 ;;
   esac
   ```

   Log and pass env:

   ```bash
   log_kv ASYMM_LAYER_GC "${ASYMM_LAYER_GC}"

   RUN_ENV+=(
     ASYMM_LAYER_GC="${ASYMM_LAYER_GC}"
     ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_GC="${ASYMM_LAYER_GC}"
   )
   ```

6. Add postprocess visibility.

   In `postprocess_lf_profile_artifacts.py`, include `asymm_layer_gc` wherever the other activation axes are summarized:

   - config counters around the current `asymm_expert_act_offload`, `asymm_attn_act_offload`, `asymm_layer_act_offload` keys
   - markdown profile header
   - CSV rows if activation-axis columns are emitted

   Use `config.get("asymm_layer_gc", config.get("asym_layer_glue_gc_enabled", "-"))`.
   Do not reuse the existing `asym_layer_gc_enabled` key for this new knob; that key already represents full decoder-layer checkpointing from `gc-layer`.

Acceptance checks before Stage 1:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 \
DRY_RUN=true \
PROFILERS=source \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|true|false,none|true|true|false|true' \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh \
  --models 'Qwen/Qwen3.5-35B-A3B|1' \
  --workloads '2048|1|1'
```

Must prove:

- both tuple forms parse
- run IDs/folders include `layergc0` or `layergc1`
- `ASYMM_LAYER_GC` appears in the generated run env
- 4-field tuples fail loudly instead of being silently misparsed
- `none|true|true|true|true` fails because layer saved-tensor offload and layer GC are mutually exclusive

## Stage 1: profile current limitation with live activation details

Purpose: before implementing `ASYMM_LAYER_GC`, capture the exact tensors still live under the current path.

Command:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_LIVE_ACTIVATION_DETAILS=true \
PROFILE_LIVE_ACTIVATION_TOPK=300 \
PROFILERS=source \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|true|false' \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh \
  --models 'Qwen/Qwen3.5-35B-A3B|1'
```

Artifacts to inspect:

```bash
find . -name memory_live_activation_details.csv -path '*layergc0*' -print
find . -name memory_breakdown_summary.json -path '*layergc0*' -print
```

Acceptance gate:

- The source profile config reports:
  - `asymm_expert_act_offload=true`
  - `asymm_attn_act_offload=true`
  - `asymm_layer_act_offload=true`
  - `asymm_layer_gc=false`
- `memory_live_activation_details.csv` exists.
- The top live rows clearly identify the remaining layer-shell candidates, especially norm outputs and decoder-layer boundary outputs.
- Do not accept any implementation claim yet; this stage is diagnostic only.

## Stage 2: LF integration for ASYMM_LAYER_GC

Files to modify:

- `asym_gemm/integrations/lf.py`
- `asym_gemm/training/__init__.py`
- new file `asym_gemm/training/decoder_layer_glue_gc.py`

Concrete changes in `lf.py`:

1. Import new wrapper APIs:

```python
from asym_gemm.training.decoder_layer_glue_gc import (
    decoder_layer_glue_gc_module_names,
    install_decoder_layer_glue_gc,
)
```

2. Add env helper:

```python
def _layer_glue_gc_enabled() -> bool:
    return _env_true(os.environ.get("ASYMM_LAYER_GC")) or _env_true(
        os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_GC")
    )
```

3. Extend `LFAsymReport`:

```python
layer_glue_gc_enabled: bool = False
layer_glue_gc_wrapped: int = 0
layer_glue_gc_modules: tuple[str, ...] = ()
layer_glue_gc_skipped: tuple[str, ...] = ()
```

4. Add wrapper installer:

```python
def _wrap_decoder_layer_glue_gc_modules(model: nn.Module, *, strict: bool):
    wrapped = []
    skipped = []
    for name, module in list(model.named_modules()):
        if not name:
            continue
        if _is_qwen3_decoder_layer_module_name(name, module):
            install_decoder_layer_glue_gc(module)
            wrapped.append(name)
            continue
        lower = f".{name.lower()}."
        leaf = name.rsplit(".", 1)[-1].lower()
        if leaf in {"decoder_layer", "layer"} and any(marker in lower for marker in _ATTENTION_GC_EXCLUDED_PATH_MARKERS):
            skipped.append(f"{name}:vision_or_multimodal")
    if strict and not wrapped:
        raise RuntimeError("ASYMM_LAYER_GC requested but no supported decoder layers were found")
    return tuple(wrapped), tuple(skipped)
```

5. In `apply_lf_asym_lora()`:

```python
layer_act_enabled = _layer_act_offload_enabled()
layer_glue_gc_enabled = _layer_glue_gc_enabled()

if layer_glue_gc_enabled and backend != "asym":
    raise RuntimeError("ASYMM_LAYER_GC requires backend='asym'")
if layer_glue_gc_enabled and recompute_config.label != "none":
    raise RuntimeError("ASYMM_LAYER_GC requires expert policy none")
if layer_glue_gc_enabled and layer_act_enabled:
    raise RuntimeError("ASYMM_LAYER_GC and ASYMM_LAYER_ACT_OFFLOAD are mutually exclusive")
```

Install after dense/expert replacements have happened, before final report:

```python
layer_glue_modules = ()
layer_glue_skipped = ()
if layer_glue_gc_enabled:
    layer_glue_modules, layer_glue_skipped = _wrap_decoder_layer_glue_gc_modules(model, strict=strict)

report.layer_glue_gc_enabled = bool(layer_glue_gc_enabled)
report.layer_glue_gc_wrapped = len(layer_glue_modules)
report.layer_glue_gc_modules = tuple(layer_glue_modules)
report.layer_glue_gc_skipped = tuple(layer_glue_skipped)

setattr(model, "_asym_layer_glue_gc_enabled", bool(layer_glue_gc_enabled))
setattr(model, "_asym_layer_glue_gc_modules", tuple(layer_glue_modules))
setattr(model, "_asym_layer_glue_gc_skipped", tuple(layer_glue_skipped))
```

6. In runtime config summary:

```python
layer_glue_gc_modules = tuple(getattr(model, "_asym_layer_glue_gc_modules", ())) or decoder_layer_glue_gc_module_names(model)
config["asymm_layer_gc"] = bool(getattr(model, "_asym_layer_glue_gc_enabled", False))
if layer_glue_gc_modules:
    config["asym_layer_glue_gc_enabled"] = bool(getattr(model, "_asym_layer_glue_gc_enabled", False))
    config["asym_layer_glue_gc_modules"] = list(layer_glue_gc_modules)
    config["asym_layer_glue_gc_skipped"] = list(getattr(model, "_asym_layer_glue_gc_skipped", ()))
```

Acceptance checks before Stage 3:

```bash
python -m compileall asym_gemm/integrations/lf.py asym_gemm/training/__init__.py asym_gemm/training/decoder_layer_glue_gc.py
```

Then dry-run config:

```bash
NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 \
DRY_RUN=true \
PROFILERS=source \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|false|true' \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh \
  --models 'Qwen/Qwen3.5-35B-A3B|1' \
  --workloads '2048|1|1'
```

Must prove `ASYMM_LAYER_GC=true` reaches the run env and expected config validation.

## Stage 3: implement decoder layer-shell GC wrapper

File to add:

- `asym_gemm/training/decoder_layer_glue_gc.py`

Supported layer families:

- Qwen3 dense decoder layer: child modules `self_attn`, `mlp`, `input_layernorm`, `post_attention_layernorm`
- Qwen3 MoE decoder layer: same child names
- Qwen3.5 MoE hybrid decoder layer: `linear_attn` or `self_attn`, `mlp`, `input_layernorm`, `post_attention_layernorm`
- Llama4 text decoder layer: `self_attn`, `feed_forward`, `input_layernorm`, `post_attention_layernorm`

Do not call `checkpoint()` around attention, linear attention, routed experts, shared experts, `mlp`, or `feed_forward`.

Core implementation pseudocode:

```python
from __future__ import annotations

from collections.abc import Callable
import inspect
import types
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class DecoderLayerGlueGCWrapper:
    def __init__(self, module: nn.Module, *, use_reentrant: bool = False, preserve_rng_state: bool = True) -> None:
        self.module = module
        self.original_forward: Callable[..., Any] = module.forward
        self.use_reentrant = use_reentrant
        self.preserve_rng_state = preserve_rng_state
        self.calls = 0
        self.checkpoint_norm_calls = 0
        self.skipped_cache_calls = 0

    def install(self) -> None:
        setattr(self.module, "_asym_decoder_layer_glue_gc_wrapper", self)
        self.module.forward = types.MethodType(_decoder_layer_glue_gc_forward, self.module)

    def _checkpoint_norm(self, norm: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        if not hidden.requires_grad:
            return norm(hidden)

        def body(x: torch.Tensor) -> torch.Tensor:
            return norm(x)

        self.checkpoint_norm_calls += 1
        return checkpoint(
            body,
            hidden,
            use_reentrant=self.use_reentrant,
            preserve_rng_state=self.preserve_rng_state,
        )

    def run(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if not self.module.training or not torch.is_grad_enabled():
            return self.original_forward(*args, **kwargs)
        if bool(kwargs.get("use_cache", False)):
            self.skipped_cache_calls += 1
            return self.original_forward(*args, **kwargs)
        return self._manual_forward(*args, **kwargs)

    def _manual_forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        layer = self.module

        residual0 = hidden_states
        normed0 = self._checkpoint_norm(layer.input_layernorm, hidden_states)

        if getattr(layer, "layer_type", "") == "linear_attention" and hasattr(layer, "linear_attn"):
            mixer_out = _call_qwen35_linear_attn(layer.linear_attn, normed0, kwargs)
        else:
            mixer_out = _call_self_attn(layer.self_attn, normed0, kwargs, family=_family(layer))

        if isinstance(mixer_out, tuple):
            mixer_out = mixer_out[0]

        hidden_after_attn = residual0 + mixer_out

        residual1 = hidden_after_attn
        normed1 = self._checkpoint_norm(layer.post_attention_layernorm, hidden_after_attn)

        if hasattr(layer, "feed_forward"):
            mlp_out = layer.feed_forward(normed1)
        else:
            mlp_out = layer.mlp(normed1)

        if isinstance(mlp_out, tuple):
            mlp_out = mlp_out[0]
        if hasattr(layer, "feed_forward"):
            mlp_out = mlp_out.view(residual1.shape)

        return residual1 + mlp_out
```

Helper pseudocode:

```python
def _family(layer: nn.Module) -> str:
    name = type(layer).__name__.lower()
    mod = type(layer).__module__.lower()
    if "qwen3_5" in name or "qwen3_5" in mod or "qwen35" in name or "qwen35" in mod:
        return "qwen35"
    if "llama4" in name or "llama4" in mod:
        return "llama4"
    return "qwen3"


def _call_self_attn(attn: nn.Module, hidden: torch.Tensor, kwargs: dict[str, Any], *, family: str) -> Any:
    if family == "llama4":
        return attn(
            hidden_states=hidden,
            position_embeddings=kwargs.get("position_embeddings"),
            attention_mask=kwargs.get("attention_mask"),
            past_key_values=kwargs.get("past_key_values"),
            use_cache=kwargs.get("use_cache", False),
            **_extra_flash_kwargs(kwargs),
        )

    position_ids = kwargs.get("position_ids")
    if family == "qwen35" and isinstance(position_ids, torch.Tensor) and position_ids.ndim == 3:
        # Match LlamaFactory's Qwen3.5 patch for full-attention layers.
        position_ids = position_ids[None, 0]

    return attn(
        hidden_states=hidden,
        attention_mask=kwargs.get("attention_mask"),
        position_ids=position_ids,
        past_key_values=kwargs.get("past_key_values"),
        use_cache=kwargs.get("use_cache", False),
        position_embeddings=kwargs.get("position_embeddings"),
        **_extra_flash_kwargs(kwargs),
    )


def _extra_flash_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    reserved = {
        "hidden_states",
        "attention_mask",
        "position_ids",
        "past_key_values",
        "cache_position",
        "use_cache",
        "position_embeddings",
    }
    return {k: v for k, v in kwargs.items() if k not in reserved}


def _call_qwen35_linear_attn(linear_attn: nn.Module, hidden: torch.Tensor, kwargs: dict[str, Any]) -> Any:
    call_kwargs = {
        "hidden_states": hidden,
        "cache_params": kwargs.get("past_key_values"),
        "cache_position": kwargs.get("cache_position"),
        "attention_mask": kwargs.get("attention_mask"),
        "position_ids": kwargs.get("position_ids"),
    }
    # Keep compatibility with upstream or patched signatures.
    sig = inspect.signature(linear_attn.forward)
    return linear_attn(**{k: v for k, v in call_kwargs.items() if k in sig.parameters})
```

Install helpers:

```python
def _decoder_layer_glue_gc_forward(module: nn.Module, *args: Any, **kwargs: Any) -> Any:
    wrapper = getattr(module, "_asym_decoder_layer_glue_gc_wrapper", None)
    if not isinstance(wrapper, DecoderLayerGlueGCWrapper):
        raise RuntimeError("decoder layer glue GC wrapper is missing")
    return wrapper.run(*args, **kwargs)


def install_decoder_layer_glue_gc(module: nn.Module) -> DecoderLayerGlueGCWrapper:
    existing = getattr(module, "_asym_decoder_layer_glue_gc_wrapper", None)
    if isinstance(existing, DecoderLayerGlueGCWrapper):
        return existing
    wrapper = DecoderLayerGlueGCWrapper(module)
    wrapper.install()
    return wrapper


def is_decoder_layer_glue_gc_wrapper(module: nn.Module) -> bool:
    return isinstance(getattr(module, "_asym_decoder_layer_glue_gc_wrapper", None), DecoderLayerGlueGCWrapper)


def decoder_layer_glue_gc_module_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(name for name, module in model.named_modules() if name and is_decoder_layer_glue_gc_wrapper(module))
```

Efficiency requirements:

- No per-expert loops.
- No splitting GEMMs.
- No checkpoint around attention, linear attention, MLP, routed experts, or shared experts.
- Only two checkpoint calls per decoder layer, one for each norm.
- Preserve exact original call signatures for Qwen3, Qwen3-MoE, Qwen3.5-MoE, and Llama4.

Correctness risks to watch:

- Qwen3.5 full-attention `position_ids` shape must match the LlamaFactory patch.
- Qwen3.5 linear-attention calls must pass `position_ids` when the patch supports it but must not break an upstream signature without it.
- Llama4 `feed_forward` output must be reshaped with `.view(residual.shape)` as the original layer does.
- This wrapper only targets layer-shell activations; core-module internal live tensors remain owned by their own Asym paths.

Acceptance checks before Stage 4:

```bash
python -m compileall asym_gemm/training/decoder_layer_glue_gc.py asym_gemm/integrations/lf.py
```

Add focused unit tests:

- `tests/training/test_decoder_layer_glue_gc.py`

Test cases:

```python
def test_qwen3_glue_gc_matches_original_forward_backward():
    # Build a tiny fake layer with input_layernorm, self_attn, post_attention_layernorm, mlp.
    # Clone it, wrap one copy, compare forward output and grads.

def test_qwen35_linear_attn_signature_is_supported():
    # Fake layer_type='linear_attention' with linear_attn accepting position_ids.
    # Verify wrapper forwards position_ids and does not call self_attn.

def test_llama4_feed_forward_view_matches_original():
    # Fake feed_forward returns a view-compatible tensor; output shape must equal residual shape.

def test_wrapper_does_not_checkpoint_core_modules():
    # Monkeypatch attention/mlp to count calls.
    # Forward + backward should not call attention/mlp a second time due to layer GC.
```

Run:

```bash
pytest -q tests/training/test_decoder_layer_glue_gc.py
```

## Stage 4: e2e validation of new ASYMM_LAYER_GC path

Run the new path first with source profiler and live activation details:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_LIVE_ACTIVATION_DETAILS=true \
PROFILE_LIVE_ACTIVATION_TOPK=300 \
PROFILERS=source \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|false|true' \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh \
  --models 'Qwen/Qwen3.5-35B-A3B|1'
```

Acceptance gate:

- Source profile config reports:
  - `asymm_layer_act_offload=false`
  - `asymm_layer_gc=true`
  - `asym_layer_glue_gc_enabled=true`
  - non-empty `asym_layer_glue_gc_modules`
- `memory_live_activation_details.csv` for `layergc1` shows a meaningful drop in norm/layer-shell live activations versus Stage 1 `layergc0`.
- Forward/backward timing does not blow up versus Stage 1. Reject if memory is unchanged or only trivially lower and latency increases.

Then run the direct comparison:

```bash
NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_LIVE_ACTIVATION_DETAILS=true \
PROFILE_LIVE_ACTIVATION_TOPK=300 \
PROFILERS=source \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0,zero3_offload|recomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|false|true' \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh \
  --models 'Qwen/Qwen3.5-35B-A3B|1'
```

Final acceptance gate:

- Report peak allocated HBM, peak reserved HBM, forward ms, backward ms, optimizer ms, and e2e step ms.
- `asym_cpuadamwds|norecomp|none|true|true|false|true` must use meaningfully less HBM than `zero3_offload|recomp`.
- Reject if it only moves memory trivially or if forward/backward latency is clearly worse than the saved memory justifies.

## Stage 5: keep old path as an explicit baseline

Keep `none|true|true|true|false` valid and runnable. It remains the baseline for current layer saved-tensor offload.

Do not silently map it to the new path. The point of the fifth field is to make these two runs visibly different:

```text
none|true|true|true|false  -> layer saved-tensor offload
none|true|true|false|true  -> layer shell GC
```

Acceptance command:

```bash
NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 \
DRY_RUN=true \
PROFILERS=source \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|true|false,none|true|true|false|true' \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh \
  --models 'Qwen/Qwen3.5-35B-A3B|1' \
  --workloads '2048|1|1'
```

Must show two distinct run labels:

```text
layeract1__layergc0
layeract0__layergc1
```
