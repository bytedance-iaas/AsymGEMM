# Qwen3.5 V2 Fix Plan: Remaining HBM Gap

This is the implementation plan for the remaining Qwen3.5 memory gap after the
current Qwen3.5 AsymGEMM work. It is based on the current code in this checkout
and the completed e2e profile under:

```text
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/outputs/qwen35_final_profile_lora_lf_v3
```

Use only this profiling entry point for the final Qwen3.5 verdict:

```bash
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Every real run must use CPU RAM nodes only:

```bash
NUMACTL_MEMBIND=0,1
NUMACTL_CPUNODEBIND=0,1
NUMACTL_MODE=membind
```

Do not use NUMA nodes `2,10,18,26`. Do not use `NUMACTL_MODE=interleave`.

The final target remains:

```text
asym_cpuadamwds|norecomp|ligerloss0 peak HBM
  <
zero3_offload|recomp|ligerloss0 peak HBM
```

The benchmark model for fast validation is:

```text
Qwen/Qwen3.5-35B-A3B
```

## Current Evidence

The last valid profile used the correct NUMA placement and exact Qwen3.5 knobs:

```bash
NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 NUMACTL_MODE=membind \
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/outputs/qwen35_final_profile_lora_lf_v3 \
PROFILERS=source PROFILE_MEMORY_BREAKDOWN=true \
MODEL_SPECS='Qwen/Qwen3.5-35B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0,zero3_offload|recomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|true' \
ASYM_OFFLOAD_MODULES=all ASYM_STRICT=true \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
LF_EXPERT_LORA_IMPLS=split-target-parameters \
LORA_DROPOUT=0.00 LORA_RANK=64 LORA_ALPHA=16 \
WORKLOADS='2048|4|1' MAX_STEPS=1 WARMUP_STEPS=1 \
bash scripts/lf/profile_lora_lf.sh --overwrite true
```

Measured result:

| Backend | Peak allocated | Peak reserved | Forward | Backward | Measured e2e |
| --- | ---: | ---: | ---: | ---: | ---: |
| `asym_cpuadamwds|norecomp|ligerloss0` | `30997.28 MiB` | `35846.00 MiB` | `3823.10 ms` | `17059.06 ms` | `25438.02 ms` |
| `zero3_offload|recomp|ligerloss0` | `27486.27 MiB` | `33044.00 MiB` | `4604.77 ms` | `13184.86 ms` | `18810.79 ms` |

Current Asym fails the memory goal by `3511.01 MiB` allocated HBM and is slower.

Important current facts:

- LoRA gradient offload is working:
  `grad_offload_hook_count=940`, `last_step_used_offloaded_grads=True`,
  `cuda_grad_bytes=0`, and `grad_offload_buffer_bytes=20492298240`.
- LoRA weight offload is working for routed, shared, linear-attention, and
  attention LoRA banks:
  `weight_offload_group_count=190`, `weight_offload_param_count=940`,
  `weight_offload_home_bytes=10246149120`, `weight_offload_skipped_group_count=0`.
- Component split from the same profile:
  `routed_experts=40 groups`, `shared_experts=80`, `linear_attention=30`,
  `attention=40`. Therefore dense/shared/GDN LoRA weight offload is no longer
  the main missing feature.
- Qwen3.5 linear-attention saved-tensor offload is firing:
  `linear_attention_saved_tensor_offload_wrapped=30`,
  `layer_act_offload_wrapped=40`, and `activation_offload.total_d2h_offloaded_bytes=87577067520`.
- The remaining HBM gap is not explained by missing LoRA offload.

The exact HBM breakdown at the actual peak:

| Category | Asym | ZeRO-3 |
| --- | ---: | ---: |
| Actual peak allocated | `32503008256` | `28821446144` |
| Actual peak reserved | `37587255296` | `34649145344` |
| Saved activation HBM | `8271273988` | `10597015556` |
| Live activation HBM | `8430551040` | `33554432` |
| Temporary workspace HBM | `14906316244` | `18189295224` |

Asym live activation at the high-water point:

| Component | HBM |
| --- | ---: |
| `lm_head` live activation | `3880 MiB` |
| `linear_attention` live activation | `2272 MiB` |
| `norms` live activation | `1600 MiB` |
| `shared_experts` live activation | `256 MiB` |
| `routed_experts` live activation | `32 MiB` |

Asym frozen CUDA residue:

| Component in current profiler | Frozen CUDA HBM |
| --- | ---: |
| `mlp_dense` | `510.96 MiB` |
| `other_model` | `332.12 MiB` |
| `embed_tokens` | `8.44 MiB` |
| `linear_attention` | `1.89 MiB` |

This frozen residue is mostly Qwen3.5 vision state moved to GPU by
`model.to(device)`, plus tiny Qwen3.5 Gated DeltaNet non-GEMM state. The
`embed_tokens` bucket is misleading here because the profiler classifies names
containing `embed` before it checks vision paths; Qwen3.5 `visual.pos_embed`
can land in `embed_tokens`.

## Verified Code State

Already implemented and should be protected by regression tests:

- `asym_gemm/training/weight_offload.py` registers generic LoRA weight offload
  groups for:
  `AsymQwen3Experts`, `AsymQwen35SharedMLP`, `AsymLlama4SharedMLP`,
  Qwen3.5 `linear_attn` parents, and standalone `AsymLoRALinear` /
  `AsymActivationOffloadLoRALinear` leaves.
- `asym_gemm/training/lora.py` has `_AsymLoRALinearWeightOffloadFunction`,
  which gathers LoRA A/B before forward/backward and releases them after use.
- `asym_gemm/training/llama4_shared_mlp.py` has shared-MLP activation offload
  and LoRA weight-offload hooks. `asym_gemm/training/qwen35_shared_mlp.py`
  reuses that path.
- `asym_gemm/training/linear_attention_activation_offload.py` wraps Qwen3.5 GDN
  saved tensors with `saved_tensors_hooks`.
- `asym_gemm/integrations/lf.py` recognizes Qwen3.5 hybrid decoder layers:
  layers with either `linear_attn` or `self_attn` are covered by the decoder
  saved-tensor layer hook.
- `scripts/lf/profile_lora_lf.sh` already supports `linear_attention` in
  `PROFILE_MEMORY_BREAKDOWN_MODULES` and `PROFILE_MODULE_FILTER`, and forwards
  `ASYM_GEMM_LF_CONFIG_ASYM_STRICT`. Use this script as the Qwen3.5 validation
  entry point. Do not use `run_lf_lora_sft.sh` defaults as the acceptance
  interface for this plan; its default memory-breakdown module list is not the
  contract being validated here.

Do not spend a stage reimplementing the above unless a regression test proves it
has broken.

## Non-Negotiable Constraints

- Do not add loss activation offload in this pass. The previous loss activation
  work was reverted intentionally.
- Do not replace FLA / causal-conv core kernels. Qwen3.5 Gated DeltaNet must
  keep using the optimized FLA and causal-conv path when installed.
- Do not add Python loops over experts or per-token loops.
- Do not split routed expert work into small GEMMs. Routed expert work must stay
  on the existing packed/grouped Qwen3 path.
- Do not use tiny per-leaf H2D staging for LoRA weights. LoRA weight offload
  must stay grouped by parent layer/module.
- Keep `ligerloss0` fixed for these Qwen3.5 comparisons.
- A stage is accepted only when the real `profile_lora_lf.sh` run shows a
  meaningful HBM reduction without unacceptable latency regression. Unit tests
  prove correctness; they do not prove the stage is worth keeping.

Meaningful HBM reduction for this plan:

- For the frozen-residue stage: remove at least `800 MiB` of frozen CUDA HBM and
  do not increase forward/backward by more than `5%`.
- For activation/live-output stages: reduce peak allocated or reserved HBM by at
  least `2 GiB` or `5%`, whichever is larger, and do not increase forward,
  backward, or measured e2e latency by more than `20%`.
- Final verdict: Asym must be lower peak allocated and lower peak reserved than
  ZeRO-3 offload on the exact same `profile_lora_lf.sh` workload.

## Stage 1: Stop Moving Frozen Vision State To GPU

Goal: remove the proven frozen CUDA residue caused by the global CPU-first
`model.to(device)` move, without breaking text compute tensors that really must
be on CUDA.

### Issue

`LlamaFactory/src/llamafactory/model/loader.py` currently does:

```python
def _move_asym_cpu_first_model_to_device(model):
    device = get_current_device()
    model.to(device)
```

This runs after Asym wrapping. It moves ordinary frozen modules that Asym did
not replace. For Qwen3.5-35B-A3B text-only SFT, that includes the frozen
multimodal `visual.*` tower:

- `visual.pos_embed`
- `visual.patch_embed.proj`
- `visual.blocks.*.{qkv,proj,linear_fc1,linear_fc2,norm*}`
- `visual.merger.*`

Those tensors are not used by the text-only profile, so moving them to HBM is
pure overhead. ZeRO-3 offload does not pay that same frozen GPU residue.

Tiny Qwen3.5 GDN non-GEMM state is different:

- `linear_attn.conv1d.weight`
- `linear_attn.dt_bias`
- `linear_attn.A_log`
- `linear_attn.norm.weight`

These are used by text forward and must remain CUDA unless a separate custom
CPU/offload implementation is added. They total about `1.89 MiB`, so they are
not the current memory gap.

### Files To Modify

- `LlamaFactory/src/llamafactory/model/loader.py`
  - `_move_asym_cpu_first_model_to_device`
  - call site in `load_model`
- `AsymGEMM/asym_gemm/integrations/lf.py`
  - add selective move helper
  - add frozen CUDA residue audit helper
  - update `classify_lf_component`
  - extend `LFAsymReport` logging for unselected frozen CUDA residue
- `AsymGEMM/asym_gemm/training/offload.py`
  - update default component classification and residency selection helpers
- `AsymGEMM/asym_gemm/profiling/lf_trace.py`
  - classify vision paths before generic `embed` / `mlp` / `norm`
- tests:
  - `AsymGEMM/tests/training/test_lf_qwen35_asym_backend.py`
  - add `LlamaFactory/tests/model/test_asym_cpu_first_move.py`
  - `AsymGEMM/tests/test_lf_memory_breakdown.py`

### Intended Implementation

Add a helper in `asym_gemm/integrations/lf.py` so LlamaFactory does not need to
duplicate Asym internals. Classify by the real parameter/buffer names, not only
by LlamaFactory `COMPOSITE_MODELS` keys, because the local Qwen3.5-MoE
Transformers model exposes paths such as `model.visual.pos_embed`,
`model.visual.patch_embed`, `model.visual.blocks`, and `model.visual.merger`.

```python
from collections import defaultdict
from asym_gemm.training.host_weight import tensor_nbytes

_VISION_PATH_MARKERS = (
    ".visual.",
    ".vision.",
    ".vision_model.",
    ".vision_tower.",
)

_MULTIMODAL_PROJECTOR_PATH_MARKERS = (
    ".multi_modal_projector.",
    ".visual.merger.",
    ".model.visual.merger.",
)

_QWEN35_LINEAR_ATTN_RUNTIME_MARKERS = (
    ".linear_attn.conv1d.",
    ".linear_attn.dt_bias",
    ".linear_attn.a_log",
    ".linear_attn.norm.",
)

def _is_vision_path(name: str) -> bool:
    lower = f".{name.lower()}."
    return any(marker in lower for marker in _VISION_PATH_MARKERS)

def _is_multimodal_projector_path(name: str) -> bool:
    lower = f".{name.lower()}."
    return any(marker in lower for marker in _MULTIMODAL_PROJECTOR_PATH_MARKERS)

def _is_vision_or_multimodal_path(name: str) -> bool:
    return _is_vision_path(name) or _is_multimodal_projector_path(name)

def _is_qwen35_linear_attn_runtime_state(name: str) -> bool:
    lower = f".{name.lower()}."
    return any(marker in lower for marker in _QWEN35_LINEAR_ATTN_RUNTIME_MARKERS)

def _is_lora_parameter_name(name: str) -> bool:
    return "lora_" in name or ".lora_A." in name or ".lora_B." in name

def _move_tensor_data_in_place(tensor: torch.Tensor, device: torch.device) -> None:
    if tensor.device == device:
        return
    tensor.data = tensor.data.to(device=device, non_blocking=False)
    if tensor.grad is not None:
        tensor.grad.data = tensor.grad.data.to(device=device, non_blocking=False)
```

Update `classify_lf_component` and profiler classification so vision is never
misreported as text MLP/embed/norm:

```python
def classify_lf_component(name: str, module: nn.Module | None = None) -> str:
    lower = name.lower()
    if _is_vision_or_multimodal_path(lower):
        return "vision"
    ...
```

Do not add `vision` to `SUPPORTED_LF_OFFLOAD_COMPONENTS` for text AsymGEMM. The
current goal is to keep frozen inactive vision state CPU-resident, not to run
vision layers through AsymGEMM.

Add a selective post-adapter move:

```python
@torch.no_grad()
def move_lf_asym_cpu_first_model_to_device(
    model: nn.Module,
    device: torch.device,
    *,
    offload_modules: Sequence[str] | str,
    strict: bool,
    keep_frozen_vision_on_cpu: bool,
    keep_frozen_projector_on_cpu: bool,
) -> dict[str, Any]:
    selection = parse_lf_offload_modules(offload_modules)
    moved_bytes_by_reason = defaultdict(int)
    kept_cpu_bytes_by_component = defaultdict(int)

    def should_keep_cpu(name: str, tensor: torch.Tensor) -> bool:
        if bool(getattr(tensor, "requires_grad", False)):
            return False
        if keep_frozen_vision_on_cpu and _is_vision_path(name):
            return True
        if keep_frozen_projector_on_cpu and _is_multimodal_projector_path(name):
            return True
        return False

    def should_force_cuda(name: str, tensor: torch.Tensor) -> bool:
        if bool(getattr(tensor, "requires_grad", False)):
            return True                    # CPUAdamW compute LoRA params must be CUDA
        if _is_lora_parameter_name(name):
            return True
        if _is_qwen35_linear_attn_runtime_state(name):
            return True                    # used by GDN forward
        return False

    for name, param in model.named_parameters(recurse=True):
        if should_keep_cpu(name, param):
            kept_cpu_bytes_by_component["vision_or_projector"] += tensor_nbytes(param)
            continue
        if should_force_cuda(name, param):
            _move_tensor_data_in_place(param, device)
            moved_bytes_by_reason["trainable_or_text_runtime"] += tensor_nbytes(param)
            continue

        # Selected frozen text weights should already be HostWeight/Asym wrappers.
        component = classify_lf_component(name)
        leaf = name.rsplit(".", 1)[-1]
        selected = component_is_selected(component, leaf, selection)
        if selected and strict:
            raise RuntimeError(
                f"{name} is selected for Asym CPU offload but remains a raw parameter before device move"
            )

        # Unknown frozen non-vision state should be tiny. Move it only if needed,
        # then let the strict residue audit decide whether it is acceptable.
        _move_tensor_data_in_place(param, device)
        moved_bytes_by_reason[f"unselected_{component}"] += tensor_nbytes(param)

    for name, buffer in model.named_buffers(recurse=True):
        if should_keep_cpu(name, buffer):
            kept_cpu_bytes_by_component["vision_or_projector"] += tensor_nbytes(buffer)
            continue
        if _is_qwen35_linear_attn_runtime_state(name):
            buffer.data = buffer.data.to(device=device, non_blocking=False)
            moved_bytes_by_reason["linear_attention_runtime_buffer"] += tensor_nbytes(buffer)
            continue
        # Tiny non-persistent buffers may move. Strict audit catches large residue.
        if buffer.device != device and not should_keep_cpu(name, buffer):
            buffer.data = buffer.data.to(device=device, non_blocking=False)

    residue = audit_lf_frozen_cuda_residue(
        model,
        offload_modules=offload_modules,
        strict=strict,
        max_allowed_unselected_cuda_bytes=8 * 1024 * 1024,
        allowed_components={"linear_attention"},
    )
    return {
        "moved_bytes_by_reason": dict(moved_bytes_by_reason),
        "kept_cpu_bytes_by_component": dict(kept_cpu_bytes_by_component),
        "unselected_frozen_cuda_residue_bytes_by_component": dict(residue),
    }
```

Change the LlamaFactory loader to pass the existing `freeze_vision_tower` and
`freeze_multi_modal_projector` flags. In the current loader, this requires
changing both the helper signature and the call site after `init_adapter`:

```python
def _move_asym_cpu_first_model_to_device(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    device = get_current_device()
    from asym_gemm.integrations.lf import move_lf_asym_cpu_first_model_to_device

    report = move_lf_asym_cpu_first_model_to_device(
        model,
        device,
        offload_modules=getattr(model_args, "asym_offload_modules", None),
        strict=bool(getattr(model_args, "asym_strict", True)),
        keep_frozen_vision_on_cpu=bool(getattr(finetuning_args, "freeze_vision_tower", True)),
        keep_frozen_projector_on_cpu=bool(getattr(finetuning_args, "freeze_multi_modal_projector", True)),
    )
    logger.info_rank0(f"Moved AsymGEMM CPU-first model state selectively: {report}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

Update call site:

```python
if asym_cpu_first_load:
    _move_asym_cpu_first_model_to_device(model, model_args, finetuning_args)
```

Do not call `model.to(device)` anywhere in this Asym CPU-first path after
`init_adapter`. The selective helper is the replacement for that global move.
It must still run after `init_adapter`, because PEFT/Asym LoRA modules are
created there and CPUAdamW validation later expects trainable LoRA compute
parameters to be CUDA-resident.

Add a strict residue audit:

```python
def audit_lf_frozen_cuda_residue(
    model: nn.Module,
    *,
    offload_modules: Sequence[str] | str,
    strict: bool,
    max_allowed_unselected_cuda_bytes: int,
    allowed_components: set[str],
) -> dict[str, int]:
    selection = parse_lf_offload_modules(offload_modules)
    rows = collect_lf_offload_residency(model, selection, classify_component=classify_lf_component)
    residue = defaultdict(int)
    examples = defaultdict(list)
    for row in rows:
        if row.kind not in {"parameter", "buffer"}:
            continue
        if row.device != "cuda" or row.requires_grad or _is_lora_parameter_name(row.name):
            continue
        if row.selected_for_cpu:
            continue
        residue[row.component] = residue.get(row.component, 0) + row.bytes
        examples.setdefault(row.component, []).append(row.name)

    bad = {
        component: bytes_value
        for component, bytes_value in residue.items()
        if component not in allowed_components and bytes_value > max_allowed_unselected_cuda_bytes
    }
    if strict and bad:
        bad_examples = {component: examples[component][:8] for component in bad}
        raise RuntimeError(
            "unselected frozen CUDA residue remains under Asym strict mode: "
            f"bytes_by_component={bad}, examples={bad_examples}"
        )
    return residue
```

This audit must run after the selective move. It is separate from the existing
selected-for-CPU audit because the bug is specifically unselected frozen vision
state hidden under `other` / `mlp_dense` / `embed_tokens`.

### Tests

Add a fake multimodal Qwen3.5-shaped model:

```python
class FakeQwen35Vision(nn.Module):
    def __init__(self):
        self.pos_embed = nn.Embedding(1024, 1152, dtype=torch.bfloat16)
        self.patch_embed = nn.Conv3d(3, 1152, kernel_size=(2, 16, 16), bias=True, dtype=torch.bfloat16)
        self.merger = nn.Sequential(
            nn.LayerNorm(1152, dtype=torch.bfloat16),
            nn.Linear(4608, 4608, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(4608, 2048, dtype=torch.bfloat16),
        )

class FakeQwen35Conditional(nn.Module):
    def __init__(self):
        self.visual = FakeQwen35Vision()
        self.language_model = FakeQwen3_5DecoderModel(...)
```

Unit assertions:

- `classify_lf_component("visual.pos_embed.weight") == "vision"`.
- `classify_lf_component("visual.blocks.0.mlp.linear_fc1.weight") == "vision"`.
- `classify_lf_component("model.visual.merger.mlp.0.weight") == "vision"`.
- `classify_lf_component("model.language_model.layers.0.linear_attn.conv1d.weight") == "linear_attention"`.
- selective move keeps all `visual.*` frozen params on CPU when
  `keep_frozen_vision_on_cpu=True`.
- selective move keeps frozen `visual.merger` / `model.visual.merger` projector
  params on CPU when `keep_frozen_projector_on_cpu=True`.
- selective move moves every trainable LoRA parameter to CUDA when CUDA is
  available.
- selective move moves Qwen3.5 GDN non-GEMM runtime state to CUDA.
- strict audit fails if a fake `visual.*` frozen tensor is manually moved to CUDA.
- strict audit allows the tiny `linear_attention` runtime residue only under the
  explicit allowlist.

Run:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
.venv/bin/python -m pytest -q \
  tests/training/test_lf_qwen35_asym_backend.py \
  tests/training/test_lora_weight_offload_generic.py \
  tests/training/test_linear_attention_activation_offload.py \
  tests/test_lf_memory_breakdown.py

cd /home/kevinni/AsymGEMM-SFT/third_party/LlamaFactory
PYTHONPATH=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM:$PYTHONPATH \
  .venv/bin/python -m pytest -q tests/model/test_asym_cpu_first_move.py tests/train/test_sft_trainer.py
```

### E2E Validation

Run the real profile:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 NUMACTL_MODE=membind \
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/outputs/qwen35_stage1_selective_move \
PROFILERS=source PROFILE_MEMORY_BREAKDOWN=true \
MODEL_SPECS='Qwen/Qwen3.5-35B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0,zero3_offload|recomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|true' \
ASYM_OFFLOAD_MODULES=all ASYM_STRICT=true \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
LF_EXPERT_LORA_IMPLS=split-target-parameters \
LORA_DROPOUT=0.00 LORA_RANK=64 LORA_ALPHA=16 \
WORKLOADS='2048|4|1' MAX_STEPS=1 WARMUP_STEPS=1 \
bash scripts/lf/profile_lora_lf.sh --overwrite true
```

Accept Stage 1 only if:

- Asym profile completes.
- `memory_by_category.csv` no longer has large frozen GPU buckets for
  `vision`, `other`, `other_model`, `mlp_dense`, or `embed_tokens`.
- Remaining frozen CUDA residue is only tiny runtime state, expected mostly
  Qwen3.5 `linear_attention` non-GEMM state around `2 MiB`.
- Asym peak allocated drops by at least `800 MiB` vs the v3 Asym baseline.
- Forward/backward latency does not regress by more than `5%`.
- LoRA offload remains intact:
  `weight_offload_group_count=190`,
  `weight_offload_skipped_group_count=0`,
  `last_step_used_offloaded_grads=True`,
  `cuda_grad_bytes=0`.

Risks to watch:

- Multimodal training with actual image/video tensors may need the vision tower
  on CUDA. This benchmark is text-only. If a later multimodal run calls
  `visual.forward`, either fail clearly or lazily move `visual` to CUDA with a
  log warning and mark that artifact invalid for the text-only HBM comparison.
- If global `model.to(device)` is removed incorrectly, generic Asym LoRA compute
  params created from CPU source modules may stay CPU. CPUAdamW already catches
  this; keep that check.

## Stage 2: Make Live Activation Attribution Exact Enough To Target

Goal: identify the exact live tensors behind the `lm_head`, `linear_attention`,
and `norms` live activation buckets before changing math or kernels.

This stage is instrumentation and diagnosis. It is accepted only if it adds
profile-only visibility without changing normal training behavior. It is not a
memory-improvement stage by itself.

### Issue

Current memory breakdown gives component totals, not the exact module names and
shapes responsible for the large live-output buckets:

- `lm_head live_activation = 3880 MiB`
- `linear_attention live_activation = 2272 MiB`
- `norms live_activation = 1600 MiB`

The current selected peak phase is `after_backward`. That makes this stage
especially important: the details must distinguish real tensor retention
through `loss` / model outputs from missing offload around a specific
linear-attention operation.

Saved-tensor offload is already firing, so these bytes are not simply missed
`saved_tensors_hooks`. They are live module outputs / graph-edge tensors or
transient forward/loss surfaces observed at high-water points.

Implementing another broad offload wrapper without knowing exact names will
create bugs and may only move attribution around.

### Files To Modify

- `AsymGEMM/asym_gemm/profiling/lf_trace.py`
  - `LFTraceConfig`
  - `LFTraceConfig.from_env`
  - `LFMemoryBreakdownProfiler._remember_live_activation`
  - `LFMemoryBreakdownProfiler._remember_tensor_components`
  - `LFMemoryBreakdownProfiler._live_activation_bytes`
  - report schema for optional top live activation rows:
    `live_activation_detail_rows` and `live_activation_detail_rows_at_peak`
- `AsymGEMM/scripts/lf/postprocess_lf_profile_artifacts.py`
  - write `memory_live_activation_details.csv` from the selected/actual peak
    detail rows and preserve them in JSON summaries
- tests:
  - `AsymGEMM/tests/test_lf_memory_breakdown.py`

### Intended Implementation

Add an opt-in profile env:

```text
ASYM_GEMM_LF_PROFILE_LIVE_ACTIVATION_DETAILS=1
ASYM_GEMM_LF_PROFILE_LIVE_ACTIVATION_TOPK=100
```

Add fields to `LFTraceConfig`:

```python
live_activation_details: bool = False
live_activation_topk: int = 100
```

`from_env` must parse them from the two env vars above. Detail tracking is
active only when both `memory_breakdown` and `live_activation_details` are true.
Track module name, component, dtype, shape, and storage size for live activation
storages. Keep this disabled by default.

Pseudocode:

```python
@dataclass
class _ActivationRecord:
    component: str
    module_name: str
    dtype: str
    shape: tuple[int, ...]
    bytes: int
    refs: list[weakref.ReferenceType[torch.Tensor]]

def _remember_live_activation(self, tensor, component, module_name=""):
    key = _saved_tensor_storage_key(tensor)
    if key is None or key in self._persistent_storage_keys:
        return
    self._storage_components[key] = component
    self._activation_components.setdefault(key, component)
    self._activation_bytes[key] = key[2]
    self._activation_refs.setdefault(key, []).append(weakref.ref(tensor))
    if not self.config.live_activation_details:
        return
    record = self._activation_records.setdefault(
        key,
        _ActivationRecord(
            component=component,
            module_name=module_name,
            dtype=str(tensor.dtype),
            shape=tuple(tensor.shape),
            bytes=key[2],
            refs=[],
        ),
    )
    record.refs.append(weakref.ref(tensor))

def _live_activation_detail_rows(self):
    if not self.config.live_activation_details:
        return []
    rows = []
    for key, record in self._activation_records.items():
        live_refs = [ref for ref in record.refs if ref() is not None]
        if not live_refs:
            continue
        if self._saved_refcounts.get(key, 0) > 0:
            continue
        rows.append({
            "component": record.component,
            "module": record.module_name,
            "dtype": record.dtype,
            "shape": list(record.shape),
            "bytes": record.bytes,
            "ref_count": len(live_refs),
        })
    return sorted(rows, key=lambda row: row["bytes"], reverse=True)[: self.config.live_activation_topk]
```

When installing module hooks, pass the semantic module name into
`_remember_tensor_components(output, component, module_name=name)`.

Do not retain strong references to tensors. Only store weakrefs and scalar
metadata.

When `_capture_saved_activation_peak` updates `_activation_bytes_at_peak`, also
snapshot `_live_activation_detail_rows()` into
`_activation_detail_rows_at_peak`. `record_phase` must include:

```python
"live_activation_detail_rows": self._live_activation_detail_rows(),
"live_activation_detail_rows_at_peak": list(self._activation_detail_rows_at_peak),
```

`build_memory_breakdown_summary` should copy the selected row's
`live_activation_detail_rows_at_peak` into the summary. The postprocessor should
write that list to `memory_live_activation_details.csv` with columns:

```text
phase,step,component,module,dtype,shape,bytes,ref_count
```

### Tests

Add a fake two-module model where one module output remains live and another
dies. Assert:

- detail rows list only live weakref-backed tensors;
- rows disappear after references die;
- persistent weights are excluded;
- saved-for-backward storages are excluded from `live_activation` if already
  counted as `saved_activation`.

Run:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
.venv/bin/python -m pytest -q tests/test_lf_memory_breakdown.py
```

### E2E Validation

Run after Stage 1:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 NUMACTL_MODE=membind \
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/outputs/qwen35_stage2_live_detail \
PROFILERS=source PROFILE_MEMORY_BREAKDOWN=true \
ASYM_GEMM_LF_PROFILE_LIVE_ACTIVATION_DETAILS=1 \
MODEL_SPECS='Qwen/Qwen3.5-35B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0,zero3_offload|recomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|true' \
ASYM_OFFLOAD_MODULES=all ASYM_STRICT=true \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
LF_EXPERT_LORA_IMPLS=split-target-parameters \
LORA_DROPOUT=0.00 LORA_RANK=64 LORA_ALPHA=16 \
WORKLOADS='2048|4|1' MAX_STEPS=1 WARMUP_STEPS=1 \
bash scripts/lf/profile_lora_lf.sh --overwrite true
```

Accept Stage 2 only if:

- peak HBM changes by less than `1%` vs Stage 1 with details disabled;
- timing changes by less than `3%`;
- `memory_live_activation_details.csv` exists for the Asym run and lists exact
  top live activation module names and shapes for `lm_head`,
  `linear_attention`, and `norms`;
- the detail rows show whether the `after_backward` peak is dominated by
  retained `loss` / `lm_head` tensors or by live tensors inside specific
  `layers.N.linear_attn` modules;
- no profiler-created strong-reference leak appears: enabling details must not
  increase live-activation HBM beyond the `<1%` threshold, and repeated
  `_live_activation_detail_rows()` calls must not keep tensors alive after the
  owning Python references are gone in the unit test.

Risks to watch:

- If detail tracking changes the peak materially, disable it by default and use
  it only for one-off diagnosis.
- If detail rows show most remaining peak is `lm_head` / loss logits under
  `ligerloss0`, do not silently add loss activation offload. Report that this
  goal cannot be met under the current "no loss activation" constraint unless
  another non-loss change removes enough HBM.

## Stage 3: Reduce Qwen3.5 Linear-Attention Live HBM Without Replacing FLA

Goal: reduce the `linear_attention` live activation and temporary workspace
surface while keeping FLA / causal-conv as the core compute.

This stage starts only after Stage 2 proves exact live owners. If Stage 2 shows
that linear-attention live tensors are not a major remaining peak after Stage 1,
skip this stage.

Do not implement a parent wrapper that only preserves the existing five
`AsymLoRALinear` leaves and changes profiling labels. That is not a memory fix.
Stage 3 is valid only if it removes or shortens the lifetime of specific
large tensors named by `memory_live_activation_details.csv`.

### Issue

Current Qwen3.5 GDN execution wraps the five projection linears independently:

```text
in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj
```

The GDN source forward then materializes:

```text
mixed_qkv = in_proj_qkv(X)
z         = in_proj_z(X)
b         = sigmoid(in_proj_b(X))
a         = in_proj_a(X)
mixed_qkv = causal_conv1d(mixed_qkv, conv1d.weight)
q, k, v   = split(mixed_qkv)
g         = -exp(A_log) * softplus(a + dt_bias)
core      = chunk_gated_delta_rule(q, k, v, g, b)
core      = norm(core, z)
Y         = out_proj(core)
```

Saved-tensor offload already covers tensors autograd saves for backward.
Remaining HBM is from live projection/core outputs, decoder graph-edge tensors,
or transient workspaces. Stage 2 decides which one is true.

If Stage 2 shows the large rows are `layers.N.linear_attn.in_proj_*`,
`layers.N.linear_attn.norm`, or same-input pre-core projection tensors, the
efficient direction is a first-class Qwen3.5 GDN parent wrapper that:

- owns the parent `linear_attn` module;
- keeps `causal_conv1d` and `chunk_gated_delta_rule` calls intact;
- fuses the same-input pre-core projections so projection outputs are split and
  consumed promptly instead of leaving multiple independent live outputs;
- exposes the same LoRA A/B parameters to CPUAdamW weight/grad offload;
- avoids per-leaf tiny staging and avoids per-token/per-expert loops.

If Stage 2 instead shows the large row is the parent `layers.N.linear_attn`
output retained as a decoder graph edge until `after_backward`, this stage must
not fake a fix. Offloading that tensor without recompute requires a custom
autograd boundary with a correct optimized backward for the whole GDN core. A
manual Python backward for FLA is not acceptable. In that case, skip Stage 3 and
report that the remaining Qwen3.5 gap is not solvable by a safe GDN wrapper
under the current no-recompute/no-loss-offload constraints.

### Files To Modify

- add `AsymGEMM/asym_gemm/training/qwen35_linear_attention.py`
  - `AsymQwen35GatedDeltaNet`
  - `is_qwen35_gated_deltanet`
  - `wrap_qwen35_gated_deltanet`
  - `_PackedQwen35GDNPreCoreProjection`
  - optional `_Qwen35GDNProjectionFunction` only if a custom autograd path is
    needed for projection lifetime control, not for replacing the FLA core
- `AsymGEMM/asym_gemm/integrations/lf.py`
  - wrap Qwen3.5 `linear_attn` parents before dense leaf wrapping
  - skip children under wrapped `linear_attn` in dense replacement loop
  - report `qwen35_linear_attn_wrapped`
- `AsymGEMM/asym_gemm/training/weight_offload.py`
  - extend the existing parent `linear_attn` LoRA grouping to recognize
    `AsymQwen35GatedDeltaNet`
  - keep exactly one LoRA weight-offload group per linear-attn layer
- `AsymGEMM/asym_gemm/profiling/lf_trace.py`
  - component/range labels for parent GDN wrapper
- tests:
  - add `AsymGEMM/tests/training/test_qwen35_linear_attention_wrapper.py`
  - extend `test_lf_qwen35_asym_backend.py`

### Intended Implementation

Parent detection:

```python
_GDN_LINEAR_LEAVES = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")

def is_qwen35_gated_deltanet(module: nn.Module) -> bool:
    class_name = type(module).__name__.lower().replace("_", "")
    module_name = type(module).__module__.lower()
    if "gateddeltanet" not in class_name and "qwen3_5_moe" not in module_name:
        return False
    return all(isinstance(getattr(module, name, None), nn.Module) for name in _GDN_LINEAR_LEAVES)
```

Wrapper state:

```python
class AsymQwen35GatedDeltaNet(nn.Module):
    def __init__(self, source, *, backend, precision, offload, lora_rank, lora_alpha, lora_dropout, stats, strict):
        self.config = getattr(source, "config", None)
        self.hidden_size = source.hidden_size
        self.key_dim = source.key_dim
        self.value_dim = source.value_dim
        self.num_k_heads = source.num_k_heads
        self.num_v_heads = source.num_v_heads
        self.head_k_dim = source.head_k_dim
        self.head_v_dim = source.head_v_dim
        self.conv_kernel_size = source.conv_kernel_size
        self.layer_idx = source.layer_idx
        self.activation = source.activation

        # Keep optimized non-GEMM operators from source.
        self.conv1d = source.conv1d
        self.dt_bias = source.dt_bias
        self.A_log = source.A_log
        self.norm = source.norm
        self.causal_conv1d_fn = source.causal_conv1d_fn
        self.causal_conv1d_update = source.causal_conv1d_update
        self.chunk_gated_delta_rule = source.chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = source.recurrent_gated_delta_rule

        # Projection leaves keep LoRA names and CPUAdamW compatibility.
        self.in_proj_qkv = AsymLoRALinear.from_host_weight(...)
        self.in_proj_z   = AsymLoRALinear.from_host_weight(...)
        self.in_proj_b   = AsymLoRALinear.from_host_weight(...)
        self.in_proj_a   = AsymLoRALinear.from_host_weight(...)
        self.out_proj    = AsymLoRALinear.from_host_weight(...)
```

The wrapper may expose child attributes named `in_proj_qkv`, `in_proj_z`,
`in_proj_b`, `in_proj_a`, and `out_proj` for state-dict compatibility, but the
memory-saving forward path must not simply call all five leaf modules
independently. It must use the packed pre-core projection path when Stage 2 has
identified those projection tensors as the HBM problem.

Packed pre-core projection path:

```python
def _precore_projection(self, hidden_states):
    flat = hidden_states.reshape(-1, self.hidden_size).contiguous()
    flat_lora = self.lora_dropout(flat).to(self.lora_dtype)

    # Base weights are concatenated once at construction as a HostWeight:
    # [out_qkv + out_z + out_b + out_a, hidden]
    base = self.pre_core_base(flat)  # one Asym frozen linear over the packed HostWeight

    # LoRA-A is one grouped/fused matmul over the four leaves, not four Python loops.
    a_cat = torch.cat(
        [
            self.in_proj_qkv.lora_a,
            self.in_proj_z.lora_a,
            self.in_proj_b.lora_a,
            self.in_proj_a.lora_a,
        ],
        dim=0,
    ).contiguous()
    low = F.linear(flat_lora, a_cat)
    low_qkv, low_z, low_b, low_a = low.split(self.lora_rank, dim=-1)

    # LoRA-B uses a grouped or packed helper over four groups. No per-token loops.
    delta_qkv, delta_z, delta_b, delta_a = grouped_dense_lora_b_4way(
        (low_qkv, low_z, low_b, low_a),
        (
            self.in_proj_qkv.lora_b,
            self.in_proj_z.lora_b,
            self.in_proj_b.lora_b,
            self.in_proj_a.lora_b,
        ),
        scale=self.lora_scale,
    )

    qkv_base, z_base, b_base, a_base = base.split(
        [self.conv_dim, self.value_dim, self.num_v_heads, self.num_v_heads],
        dim=-1,
    )
    return (
        qkv_base + delta_qkv.to(qkv_base.dtype),
        z_base + delta_z.to(z_base.dtype),
        b_base + delta_b.to(b_base.dtype),
        a_base + delta_a.to(a_base.dtype),
    )
```

The `torch.cat` above is over four small LoRA-A matrices, not over token
activations. It is acceptable only if `weight_offload_staged_high_water_bytes`
and kernel counts stay within the Stage 3 acceptance limits. If it becomes a
measured overhead, replace it with a cached packed LoRA-A view refreshed only
after LoRA weight gather.

Forward keeps the FLA core:

```python
def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch, seq, _ = hidden_states.shape

    mixed_qkv, z, b, a = self._precore_projection(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    # Keep source conv/cache semantics exactly.
    if use_precomputed_states and seq == 1:
        mixed_qkv = self.causal_conv1d_update(...)
    else:
        if use_precomputed_states:
            mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
        if cache_params is not None:
            cache_params.update_conv_state(...)
        if self.causal_conv1d_fn is not None:
            mixed_qkv = self.causal_conv1d_fn(...)
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :mixed_qkv.shape[-1]])
        if use_precomputed_states:
            mixed_qkv = mixed_qkv[:, :, -seq:]

    q, k, v = split_and_reshape(mixed_qkv)
    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

    # Keep FLA.
    core, state = self.chunk_gated_delta_rule(...)
    core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    core = core.reshape(batch, seq, -1)
    return self.out_proj(core)
```

LoRA weight offload registration:

```python
def _qwen35_gdn_lora_banks(module):
    banks = []
    for leaf in _GDN_LINEAR_LEAVES:
        child = getattr(module, leaf)
        banks.append((f"{leaf}.lora_A", child.lora_a))
        banks.append((f"{leaf}.lora_B", child.lora_b))
    return banks

coordinator.register_group(
    module,
    _qwen35_gdn_lora_banks(module),
    group_name=module.profile_prefix,
    component="linear_attention",
    force=True,
)
```

This must be integrated with the existing `install_lora_weight_offload`
`linear_attn` parent path. Do not allow both the generic parent matcher and the
new wrapper matcher to register the same five leaves twice. The validation
counter must remain:

```text
weight_offload_group_count_by_component["linear_attention"] == 30
weight_offload_skipped_group_count == 0
```

Correctness requirements:

- Source parity vs local Transformers `Qwen3_5MoeGatedDeltaNet` for forward and
  backward on a small config.
- Preserve cache semantics for `cache_params is None`, cached multi-token, and
  single-token decode. Training profile only needs no-cache, but the wrapper
  must not silently break generation paths.
- Preserve LoRA state names and adapter save/load behavior.
- Preserve CPUAdamW visibility: trainable params must remain LoRA params and
  must be CUDA compute params before optimizer creation.
- Do not wrap vision attention or vision MLP.

### Tests

Add tests:

- `test_qwen35_gdn_wrapper_matches_source_no_cache`
- `test_qwen35_gdn_wrapper_backward_matches_source_no_cache`
- `test_qwen35_gdn_wrapper_preserves_lora_state_names`
- `test_qwen35_gdn_weight_offload_registers_one_group_per_layer`
- `test_lf_apply_wraps_linear_attn_parent_and_skips_child_double_wrap`
- CUDA-gated test: FLA import path still called when installed.

Run:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
.venv/bin/python -m pytest -q \
  tests/training/test_qwen35_linear_attention_wrapper.py \
  tests/training/test_lf_qwen35_asym_backend.py \
  tests/training/test_lora_weight_offload_generic.py \
  tests/training/test_linear_attention_activation_offload.py
```

### E2E Validation

Run only after tests pass and Stage 2 shows linear-attention live HBM is still
large and owned by internal GDN projection/norm tensors that the Stage 3 wrapper
actually changes:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 NUMACTL_MODE=membind \
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/outputs/qwen35_stage3_gdn_parent \
PROFILERS=source PROFILE_MEMORY_BREAKDOWN=true \
MODEL_SPECS='Qwen/Qwen3.5-35B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0,zero3_offload|recomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|true' \
ASYM_OFFLOAD_MODULES=all ASYM_STRICT=true \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
LF_EXPERT_LORA_IMPLS=split-target-parameters \
LORA_DROPOUT=0.00 LORA_RANK=64 LORA_ALPHA=16 \
WORKLOADS='2048|4|1' MAX_STEPS=1 WARMUP_STEPS=1 \
bash scripts/lf/profile_lora_lf.sh --overwrite true
```

Accept Stage 3 only if:

- `linear_attention` live activation drops materially from the current
  `2272 MiB` baseline.
- `memory_live_activation_details.csv` shows the specific Stage 2
  `layers.N.linear_attn.*` rows targeted by the wrapper are gone or much
  smaller, not merely renamed.
- Overall peak allocated/reserved drops by at least `2 GiB` or `5%` vs Stage 1.
- Forward, backward, and e2e latency stay within `20%` of Stage 1.
- Runtime counters do not show many more tiny kernel launches.
- `weight_offload_group_count_by_component["linear_attention"] == 30`.
- `weight_offload_skipped_group_count == 0`.

Reject Stage 3 if it only moves profiler attribution, increases kernel launch
count materially, or reduces HBM by less than the acceptance threshold.

Skip Stage 3 if Stage 2 proves the remaining linear-attention HBM is parent
decoder graph-edge output retained until `after_backward`, because the safe
fix for that is not a projection wrapper. Do not replace it with checkpointing
inside the `norecomp` comparison.

Risks to watch:

- Manual backward for FLA is not acceptable unless it calls the same optimized
  backend primitives or a proven native kernel. Prefer normal autograd plus
  saved-tensor offload until profiling proves it cannot work.
- `in_proj_b` and `in_proj_a` have tiny output dimensions; direct AsymGEMM
  backend may fall back to torch CPU-fetched due alignment. Keep the existing
  fallback explicit and measured.
- Fusing LoRA-B with block-diagonal dense matrices is wasteful. Use grouped or
  packed helpers, not a sparse block diagonal disguised as a dense GEMM.

## Stage 4: Final Verdict And Rollback Rules

Goal: make the real pass/fail decision using only e2e `profile_lora_lf.sh`
metrics.

Run:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 NUMACTL_MODE=membind \
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/outputs/qwen35_v2_final \
PROFILERS=source PROFILE_MEMORY_BREAKDOWN=true \
MODEL_SPECS='Qwen/Qwen3.5-35B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0,zero3_offload|recomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|true' \
ASYM_OFFLOAD_MODULES=all ASYM_STRICT=true \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
LF_EXPERT_LORA_IMPLS=split-target-parameters \
LORA_DROPOUT=0.00 LORA_RANK=64 LORA_ALPHA=16 \
WORKLOADS='2048|4|1' MAX_STEPS=1 WARMUP_STEPS=1 \
bash scripts/lf/profile_lora_lf.sh --overwrite true
```

Extract metrics:

```bash
python - <<'PY'
import csv, pathlib

root = pathlib.Path("/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/outputs/qwen35_v2_final")
index_paths = sorted(root.rglob("memory_breakdown_index.csv"))
index_paths = [path for path in index_paths if "memory_combined" in path.parts]
if index_paths:
    path = index_paths[0]
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    print(path)
    for row in rows:
        print(row["backend"], row["recompute"], row["liger_loss"])
        print("  actual_peak_allocated_hbm_bytes", row["actual_peak_allocated_hbm_bytes"])
        print("  actual_peak_reserved_hbm_bytes", row["actual_peak_reserved_hbm_bytes"])
        print("  activation_hbm_bytes_at_peak", row["activation_hbm_bytes_at_peak"])
        print("  temporary_workspace_hbm_bytes_at_peak", row["temporary_workspace_hbm_bytes_at_peak"])
else:
    profiles = [
        path for path in sorted(root.rglob("memory_breakdown.csv"))
        if "memory_plots" not in path.parts and "memory_combined" not in path.parts
    ]
    for path in profiles:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        first = rows[0]
        print(path)
        print("  actual_peak_allocated_hbm_bytes", first["actual_peak_allocated_hbm_bytes"])
        print("  actual_peak_reserved_hbm_bytes", first["actual_peak_reserved_hbm_bytes"])
        print("  saved_activation_hbm_bytes_at_peak", first["saved_activation_hbm_bytes_at_peak"])
        print("  live_activation_hbm_bytes_at_peak", first["live_activation_hbm_bytes_at_peak"])
        print("  temporary_workspace_hbm_bytes_at_peak", first["temporary_workspace_hbm_bytes_at_peak"])

for path in sorted(root.rglob("step_samples.csv")):
    if "memory_plots" in path.parts or "memory_combined" in path.parts:
        continue
    with path.open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("is_warmup") == "False"]
    if not rows:
        continue
    row = rows[-1]
    print(path)
    print("  forward_ms", row.get("forward_milliseconds"))
    print("  backward_ms", row.get("backward_milliseconds"))
    print("  trainer_e2e_ms", row.get("trainer_e2e_step_milliseconds"))
PY
```

Final pass requires:

- Asym peak allocated < ZeRO peak allocated.
- Asym peak reserved < ZeRO peak reserved.
- Asym does not have large frozen CUDA residue.
- LoRA weight/grad offload remains complete.
- No invalid NUMA placement appears in logs.
- `liger_loss=ligerloss0` is present in the profile config.

Rollback rules:

- Roll back any stage that increases peak HBM.
- Roll back any stage that leaves HBM effectively unchanged and increases
  latency.
- Roll back any stage that reduces HBM trivially while adding custom kernels,
  new maintenance burden, or many kernel launches.
- Keep Stage 1 even if it is not enough for the final goal only if it removes
  the proven frozen vision HBM residue and does not regress timing; it fixes a
  correctness mismatch in the comparison.

## Deferred / Do Not Implement In This Pass

### Routed Expert Temporary Peak

The known routed temp issue is the `torch.cat((gate_lora_A, up_lora_A), dim=1)`
path and related pair-concat temporary paths in `qwen3_moe.py`. This affects
Qwen3, Llama4, and Qwen3.5 routed experts together. It is recorded but not part
of this Qwen3.5 v2 pass unless the final Stage 4 result is still close and the
user explicitly asks to attack routed temp peaks next.

If implemented later, it must be a loop-free paired LoRA-A path or native kernel
that writes `[M, 2r]` directly. Do not add per-expert loops.

### Loss Activation / Full Logits Path

Do not add loss activation offload here. Under `ligerloss0`, the model still
materializes full logits to compute loss. The current trainer clears
`outputs.logits` after extracting `loss`, but that cannot remove the logits HBM
surface during `lm_head` and loss construction.

If final profiles prove the remaining gap is dominated by `lm_head` / loss
logits after Stage 1 and Stage 3, the honest conclusion is that the final memory
goal cannot be met under the current no-loss-activation, no-Liger-loss
constraint. Do not hide that behind unrelated changes.

### FLA Core Replacement

Do not replace `causal_conv1d` or `chunk_gated_delta_rule`. Asym should wrap
around those optimized kernels, not reimplement linear attention core math in
Python.
