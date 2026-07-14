# AsymGEMM Owned-Router MoE Plan

## Goal

Add an explicit AsymGEMM SFT mode where Qwen3 and Llama 4 MoE routing is
computed inside an AsymGEMM-owned MoE wrapper, with router weights frozen and
router forward run under `torch.no_grad()`. This should match KTransformers'
SFT routing semantics: same forward routing math, no router parameter training,
and no saved router/topk/softmax autograd graph.

The important semantic distinction:

- Target forward math: same selected experts and route weights as the HF Qwen3
  or Llama 4 MoE block for the same hidden states, dtype, and model weights.
- Target backward math: same as a detached-router oracle, not full HF router
  autograd. The gradient path through `loss -> top_k_weights -> router ->
  hidden_states` is intentionally dropped, matching KT. Expert LoRA gradients and
  the expert-output gradient to hidden states must remain correct.
- Default behavior must remain unchanged unless the new router mode is enabled.

## Current State

Actual LF script path:

- `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh`
- Defaults to `LF_DIR=/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory`.
- Uses `ENV_DIR=${LF_DIR}/.venv`.
- Sets `PYTHONPATH=${ASYM_DIR}:${LF_DIR}/src:${PYTHONPATH:-}`.
- Qwen3 implementation comes from the LF venv's installed `transformers`, not
  archived KTransformers copies under `third_party/ktransformers/archive`.

Actual installed Qwen3 router behavior in that venv:

```python
router_logits = F.linear(hidden_states, self.weight)          # bf16 in pure bf16 run
router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
if self.norm_topk_prob:
    router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
router_top_value = router_top_value.to(router_logits.dtype)   # bf16 in pure bf16 run
return router_logits, router_top_value, router_indices
```

LF MoE aux-loss scope:

- `moe_aux_loss_coef` defaults to `None`.
- When LF sets a nonzero `moe_aux_loss_coef`, it sets
  `config.output_router_logits=True` for `qwen3_moe` and the model may expect
  router logits for load-balancing loss.
- First implementation of `asym_router_mode=whole` is scoped to the current
  profiling target with `moe_aux_loss_coef=None` / `output_router_logits=False`.
  If router aux loss is enabled, fail fast instead of silently dropping router
  logits.

Current AsymGEMM Qwen3 integration:

- `asym_gemm/integrations/lf.py` replaces only the packed `experts` module when
  `is_qwen3_experts(module)` is true.
- `AsymQwen3Experts.forward(hidden_states, top_k_index, top_k_weights)` consumes
  routing that was already computed by the original HF MoE block.
- Router parameters are validated as frozen by `_validate_trainable_params`.
- Router modules are skipped as dense LoRA targets by LF adapter filtering and by
  AsymGEMM's own dense wrapping logic.
- Because AsymGEMM does not own Qwen3 routing today, router forward can still
  run with autograd enabled, so PyTorch may save fp32 router softmax/topk state.

Current AsymGEMM Llama 4 integration:

- `asym_gemm/integrations/lf.py` already replaces the whole Llama 4 MoE module
  with `AsymLlama4Moe` when `is_llama4_moe(module)` is true.
- `AsymLlama4Moe` owns `router`, `shared_expert`, and wrapped packed `experts`.
  This is already structurally a whole-MoE wrapper, unlike the current Qwen3
  path.
- Current `AsymLlama4Moe.forward(...)` runs the router with normal autograd and
  returns `(out, router_logits)`, matching the installed Llama 4 block contract.
- The missing Llama 4 work is not another wrapper class; it is adding
  `router_mode` to `AsymLlama4Moe` so `router_mode=whole` runs the existing
  owned router under `torch.no_grad()` while `router_mode=hf` preserves current
  behavior.

Actual installed Llama 4 router behavior in the LF venv:

```python
router_logits = F.linear(hidden_states, self.weight)                  # bf16 in pure bf16 run
router_top_value, router_indices = torch.topk(router_logits, top_k, dim=1)
router_scores = torch.full_like(router_logits, -inf).scatter_(1, router_indices, router_top_value)
router_scores = torch.sigmoid(router_scores.float()).to(router_scores.dtype)
return router_scores, router_logits
```

Actual installed Llama 4 MoE behavior:

```python
router_scores, router_logits = self.router(hidden_states_2d)
routed_in = hidden_states_2d.repeat(num_experts, 1)
routed_in = routed_in * router_scores.transpose(0, 1).reshape(-1, 1)
routed_out = self.experts(routed_in)
out = self.shared_expert(hidden_states_2d)
out.add_(routed_out.reshape(num_experts, -1, hidden).sum(dim=0))
return out, router_logits
```

## KTransformers Design Check

Local KT paths reviewed:

- `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/ktransformers/doc/en/SFT/KTransformers-Fine-Tuning_Developer-Technical-Notes.md`
- `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/python/sft/layer.py`
- `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/python/sft/autograd.py`
- `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/python/sft/wrapper.py`
- `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/python/sft/lora.py`
- `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/python/sft/base.py`

KT design points that should influence this AsymGEMM owned-router pass:

- KT replaces the full MoE layer with `KTMoELayerWrapper`, not only the expert
  list. This confirms `asym_router_mode=whole` should wrap the whole Qwen3 MoE
  block and should make the already whole-MoE Llama 4 wrapper run its router
  under the same no-grad boundary.
- KT registers the router/gate before experts inside the replacement wrapper to
  preserve original module traversal order. AsymGEMM must do the same so PEFT
  traversal, deterministic LoRA init order, state-dict order, and diagnostics do
  not change unexpectedly.
- KT computes routing under `torch.no_grad()` and treats the router as frozen in
  LoRA SFT. This confirms the detached-router oracle is the correct reference,
  not full HF router-autograd parity.
- KT detaches tensors before submitting them to its C++ backend because it
  reconnects gradients manually through `KTMoEFunction`. AsymGEMM must not copy
  that detach pattern around the expert call in this first Python/GPU wrapper:
  `flat` must still require grad when passed into `AsymQwen3Experts`, and only
  `top_k_index/top_k_weights` are detached by the router no-grad boundary.
- KT has special non-reentrant-gradient-checkpoint handling with a sentinel
  saved tensor because its C++ forward cache must be populated before backward.
  AsymGEMM should not add that machinery in this first owned-router wrapper, but
  it must add an explicit gradient-checkpointing smoke test so we catch any
  interaction with AsymGEMM expert recompute policies.
- KT packs PEFT expert LoRA into contiguous CPU buffers/views for its C++ kernel.
  AsymGEMM already owns expert LoRA parameters inside `AsymQwen3Experts`, so do
  not add KT-style PEFT view replacement in this pass.
- KT clears/shrinks original expert weights after loading them into the backend
  to avoid FSDP broadcast and duplicate storage. AsymGEMM should not add KT's
  FSDP registration indirection in this first single-GPU pass, but validation must
  ensure the owned wrapper does not retain the original expert module or any
  duplicate base-weight storage after wrapping.

Conclusion: the plan stays structurally correct. The required refinements from
KT are to preserve wrapper module order, make the detach boundary explicit,
validate gradient-checkpointing, and test that no original expert storage is
retained. No KT code or KT-kernel files should be reused or modified for this
AsymGEMM-owned router implementation.

## Desired User-Facing Mode

Add a new router mode while preserving current defaults:

```text
asym_router_mode=hf              # default: current behavior
                                  # Qwen3: expert-only replacement
                                  # Llama 4: existing full-MoE wrapper with router autograd behavior unchanged
asym_router_mode=whole           # Qwen3: wrap whole MoE block, router under no_grad
                                  # Llama 4: use existing full-MoE wrapper, router under no_grad
```

If router-grad parity is needed while developing the wrapper, implement it as a
test-only flag such as `router_debug_grad=True` on `AsymQwen3MoeBlock` and
`AsymLlama4Moe`, not as a public LF option. The production whole-MoE mode always
runs the frozen router under `torch.no_grad()`.

Do not move the router to CPU. The router stays with the dense model on CUDA.
The memory saving comes from not saving router autograd state, not from changing
router placement or dtype.

## Implementation Plan

### Phase 0: Scope Guards

Implementation:

Add explicit guards so unsupported cases fail with a clear error before any
wrapping:

1. In `third_party/LlamaFactory/src/llamafactory/hparams/parser.py`, reject:

```python
if model_args.use_asym_gemm and model_args.asym_router_mode == "whole" and model_args.moe_aux_loss_coef:
    raise ValueError("asym_router_mode=whole does not support moe_aux_loss_coef yet.")
```

2. In `asym_gemm/integrations/lf.py::apply_lf_asym_lora`, reject if the loaded
   model config already requests router logits:

```python
if router_mode == "whole" and bool(getattr(getattr(model, "config", None), "output_router_logits", False)):
    raise ValueError("asym_router_mode=whole requires output_router_logits=False.")
```

3. In `AsymQwen3MoeBlock.__init__`, support only the installed Transformers 5.6
   Qwen3 MoE block contract used by the LF venv:

```text
source has .gate and .experts
source.forward(hidden_states) returns a single hidden tensor
gate(hidden_2d) returns (router_logits, top_k_weights, top_k_index)
experts.forward(hidden_2d, top_k_index, top_k_weights) returns hidden_2d
```

Under `strict=True`, any other contract raises `TypeError`/`ValueError` with the
actual module class name and source file. Legacy tuple-returning MoE blocks and
router-aux-loss support are out of scope for this implementation.

4. In `AsymLlama4Moe.__init__`, keep accepting the existing installed
   Transformers 5.6 Llama 4 MoE contract:

```text
source has .router, .shared_expert, and .experts
source.forward(hidden_states) returns (hidden_2d, router_logits)
router(hidden_2d) returns (router_scores, router_logits)
experts.forward(routed_in) returns routed_out
```

Reject `router_mode` values outside `hf|whole`. In `hf`, preserve the current
`AsymLlama4Moe` router-autograd behavior. In `whole`, run only the router/top-k
metadata computation under `torch.no_grad()` and detach router outputs before
the expert path.

Validation before moving on:

- Add a tiny-model test that sets `model.config.output_router_logits=True` and
  verifies `router_mode="whole"` raises before wrapping.
- Add a test that the default profiling config has
  `model.config.output_router_logits == False` and `moe_aux_loss_coef is None`.
- Add a Llama 4 fake-model test that `router_mode="hf"` is accepted and
  preserves current output/logit parity.
- Add a Llama 4 fake-model test that `router_mode="whole"` is accepted when
  router aux loss is disabled and rejected when `output_router_logits=True`.

### Phase 1: Add Arguments and Pass-Through

Implementation:

1. Add an LF model argument in
   `third_party/LlamaFactory/src/llamafactory/hparams/model_args.py`:

```python
asym_router_mode: Literal["hf", "whole"] = field(
    default="hf",
    metadata={"help": "AsymGEMM MoE router handling for Qwen3/Llama4: hf or whole."},
)
```

2. Validate the option in
   `third_party/LlamaFactory/src/llamafactory/hparams/parser.py` alongside the
   other AsymGEMM checks.

3. Pass `model_args.asym_router_mode` through
   `third_party/LlamaFactory/src/llamafactory/model/adapter.py` into
   `adapt_lf_asym_peft_lora(...)`.

4. Add `router_mode` to:

- `asym_gemm/integrations/peft_lf.py::adapt_lf_asym_peft_lora`
- `asym_gemm/integrations/lf.py::apply_lf_asym_lora`
- `LFAsymReport`, including `router_mode` in `to_log_string()`

5. Update `scripts/lf/run_lf_lora_sft.sh`:

```bash
ASYM_ROUTER_MODE=${ASYM_ROUTER_MODE:-hf}
CMD_ARGS+=(--asym_router_mode "${ASYM_ROUTER_MODE}")
```

When `PROFILE=1`, preserve both the backend identity and router mode in
`profile.json`. The current fallback collapses `asym_torch` to `torch`; replace
that fallback so standalone profiled runs can distinguish plain LF torch from
AsymGEMM's torch expert backend:

```bash
ASYM_GEMM_LF_CONFIG_BACKEND="${PROFILE_BACKEND_LABEL:-${BACKEND}}"
ASYM_GEMM_LF_CONFIG_ROUTER_MODE="${ASYM_ROUTER_MODE}"
```

6. Update `scripts/lf/profile_lora_lf.sh` to sweep router modes:

```bash
ROUTER_MODES=${ROUTER_MODES:-hf}
```

Include router mode in the run directory and `RUN_ID`, and pass
`ASYM_ROUTER_MODE="${router_mode}"` into `run_lf_lora_sft.sh`.

Exact folder-name location:

- Keep `precision_root` unchanged:

```text
${OUTPUT_ROOT}/${dataset_root_label}__lora__lf__${precision_label}
```

- Keep `config_root_path(...)` mostly unchanged because it describes the model,
  batch, sequence length, steps, rank, alpha, and dropout.

- Add router mode to `job_root_path(...)`, because router mode is a per-run
  implementation axis like backend, profiler, recompute, and expert policy:

```bash
job_root_path() {
  local config_root="$1"
  local backend="$2"
  local profiler="$3"
  local recompute="$4"
  local expert_policy="$5"
  local router_mode="$6"
  printf '%s/%s\n' "${config_root}" "$(safe_label "${backend}__${profiler}__${recompute}__pol${expert_policy}__router${router_mode}")"
}
```

- Update the call site in `run_job(...)`:

```bash
job_root="$(job_root_path "${config_root}" "${backend}" "${profiler}" "${recompute}" "${expert_policy}" "${router_mode}")"
seq_root="${job_root}/s${seq_len}"
run_id="lf_${backend}_${profiler}_${recompute}_pol${expert_policy}_router${router_mode}_s${seq_len}_${lora_dropout_label_value}"
```

This yields folders like:

```text
${OUTPUT_ROOT}/${dataset_root_label}__lora__lf__${precision_label}/qwen3-30b-a3b__gpus1__b1_s4096_w0_s1_r8_a16_drop000/
  asym__source__norecomp__polnone__routerhf/s4096/
  asym__source__norecomp__polnone__routerwhole/s4096/
```

Also update:

- `run_job(...)` signature to accept `router_mode`.
- `group_key` to include `router_mode`, otherwise `hf` and `whole` can collide:

```bash
local group_key="${config_root}|${profiler}|${recompute}|${expert_policy}|${router_mode}|${seq_len}"
```

- `compare_group_labels[...]` to include `router_mode`.
- `jobs.tsv` header and `append_job_record(...)` rows to include a
  `router_mode` column.
- `run_env+=(ASYM_ROUTER_MODE="${router_mode}")`.
- The profiled run config must contain router mode. This is satisfied by the
  `ASYM_GEMM_LF_CONFIG_ROUTER_MODE="${ASYM_ROUTER_MODE}"` update in
  `run_lf_lora_sft.sh`; no extra parser change is needed because
  `run_lf_profiled_train.py` imports every `ASYM_GEMM_LF_CONFIG_*` environment
  variable into `profile.json["config"]`.
- Outer sweep loop:

```bash
mapfile -t router_modes < <(tokens "${ROUTER_MODES}" | while read -r value; do router_mode_label "${value}"; done | dedupe)

for seq_len in "${seq_lens[@]}"; do
  current_dataset="${DATASET}"
  if [[ "${PREPARE_DATASETS}" == "true" ]]; then
    current_dataset="$(dataset_name_for_seq "${seq_len}")"
    if [[ "${COLLECT_EXISTING}" != "true" ]]; then
      prepare_dataset_for_seq "${seq_len}" "${current_dataset}"
    fi
  fi
  for lora_dropout in "${lora_dropouts[@]}"; do
    LORA_DROPOUT="${lora_dropout}"
    lora_dropout_label_value="$(lora_dropout_label "${LORA_DROPOUT}")"
    config_root="$(config_root_path "${seq_len}")"
    for router_mode in "${router_modes[@]}"; do
      for expert_policy in "${expert_policies[@]}"; do
        for backend_recompute in "${backend_specs[@]}"; do
          backend="${backend_recompute%%|*}"
          recompute="${backend_recompute##*|}"
          for profiler in "${profilers[@]}"; do
            if [[ "${router_mode}" == "whole" && "${backend}" == "torch" ]]; then
              continue
            fi
            gpu_count="$(backend_gpu_count "${backend}" "${current_model_gpu_count}")"
            gpu="$(gpu_slice "${gpu_count}")"
            run_job "${backend}" "${profiler}" "${recompute}" "${seq_len}" "${gpu}" "${gpu_count}" "${expert_policy}" "${router_mode}" "${current_dataset}"
          done
        done
      done
    done
  done
done
```

- Add a simple normalizer:

```bash
router_mode_label() {
  case "${1,,}" in
    hf|expert|experts) printf 'hf\n' ;;
    whole|owned|owned-moe|owned_moe) printf 'whole\n' ;;
    *) die "router mode must be hf or whole, got '${1}'" ;;
  esac
}
```

Backend-axis fix required for the validation commands:

Current `profile_lora_lf.sh` normalizes `asym_torch` to `torch` in
`backend_label()`. That is not acceptable for this work because `asym_torch`
means "use the AsymGEMM wrappers with torch expert backend", while `torch` means
"plain LF/PEFT without AsymGEMM". Change the script to preserve three internal
backend labels:

```bash
backend_label() {
  case "${1,,}" in
    torch) printf 'torch\n' ;;
    asym_torch|asym-torch) printf 'asym_torch\n' ;;
    asym) printf 'asym\n' ;;
    *) die "backend must be torch, asym_torch, or asym, got '${1}'" ;;
  esac
}

backend_gpu_count() {
  local backend="$1"
  local model_gpu_count="$2"
  case "${backend}" in
    asym|asym_torch) printf '1\n' ;;
    torch) printf '%s\n' "${model_gpu_count}" ;;
    *) die "internal backend label must be torch, asym_torch, or asym, got '${backend}'" ;;
  esac
}
```

Keep passing `BACKEND="${backend}"` into `run_lf_lora_sft.sh`; that script
already accepts `torch`, `asym_torch`, and `asym`.

Plot and postprocess updates required by the new folder/backend axis:

1. Update `scripts/plotting/plot_activation_recompute_sweep.py`:

```python
BACKENDS = ("asym", "asym_torch", "torch", "kt")
BACKEND_MARKERS = {
    "asym": "^",
    "asym_torch": "v",
    "torch": "o",
    "kt": "s",
}
```

2. Update `parse_flat_result_dir(...)` to accept both old and new flat job
   layouts:

```text
old: backend__profiler__recompute__pol${expert_policy}/s${seq_len}
new: backend__profiler__recompute__pol${expert_policy}__router${router_mode}/s${seq_len}
```

Parsing rule:

```python
parts = job_dir.name.split("__")
if len(parts) == 4:
    backend, profiler, recompute, policy_part = parts
    router_mode = "hf"
elif len(parts) == 5:
    backend, profiler, recompute, policy_part, router_part = parts
    if not router_part.startswith("router"):
        return None
    router_mode = router_part[len("router") :]
    if router_mode not in {"hf", "whole"}:
        return None
else:
    return None
```

Return `"router_mode": router_mode` in the parsed metadata. This keeps existing
profiling folders readable while making new owned-router runs explicit.

3. Include `router_mode` in:

- `group_key(...)`
- `threshold_group_key(...)`
- the `groups` / `step_groups` / `family_groups` key tuple types
- per-group output directory labels
- any varied-field or combined-label logic that decides which series labels are
  needed

Do not let `routerhf` and `routerwhole` rows share one plot series or one output
folder. Implement this plotting support before Phase 7. Final validation must
run with plotting enabled; `PLOT=false` is allowed only for a local dry-run
debug command before the plotter patch is complete.

4. Update `scripts/lf/postprocess_lf_profile_artifacts.py` summary markdown to
show router mode from `profile["config"]["router_mode"]`:

```text
Router mode: `hf|whole`
```

Validation before moving on:

- Run a dry run and confirm the command includes `--asym_router_mode`.

```bash
cd /home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_torch|norecompute" \
ROUTER_MODES="hf,whole" \
PROFILERS=source \
MAX_STEPS=1 \
MAX_SAMPLES=1 \
SEQ_LENS=16 \
DRY_RUN=true \
scripts/lf/profile_lora_lf.sh
```

Check:

- The default path remains `ASYM_ROUTER_MODE=hf`.
- Both `hf` and `whole` runs appear in `command.txt`.
- `command.txt` for `asym_torch` still contains `BACKEND=asym_torch`, not
  `BACKEND=torch`.
- `profile.json["config"]["backend"]` is `asym_torch` for `BACKEND=asym_torch`
  profiled runs.
- `profile.json["config"]["router_mode"]` is `hf` or `whole`.
- `jobs.tsv` contains a `router_mode` column.
- Plot outputs create separate `routerhf` and `routerwhole` groups instead of
  merging them.
- No existing script invocation changes behavior unless `ROUTER_MODES` or
  `ASYM_ROUTER_MODE` is explicitly set.

### Phase 2: Add Qwen3 MoE Block Detection

Implementation:

Add helpers in `asym_gemm/training/qwen3_moe.py`:

```python
def is_qwen3_moe_block(module: nn.Module) -> bool:
    gate = getattr(module, "gate", None)
    experts = getattr(module, "experts", None)
    return (
        isinstance(gate, nn.Module)
        and is_qwen3_experts(experts)
        and hasattr(gate, "top_k")
        and hasattr(gate, "num_experts")
        and hasattr(gate, "hidden_dim")
    )
```

Also add a guard so already-wrapped modules are not wrapped again:

```python
if getattr(module, "_is_asym_qwen3_moe_block", False):
    return False
```

Validation before moving on:

- Add a small unit test using a tiny `Qwen3MoeForCausalLM` config from the LF
  venv.
- Assert detection finds exactly the model's Qwen3 MoE block(s).
- Assert `is_qwen3_experts(module.experts)` is true for those blocks.
- Assert router module names remain non-trainable after PEFT.

Suggested test file:

```text
third_party/AsymGEMM/tests/training/test_qwen3_owned_moe.py
```

Suggested command:

```bash
cd /home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM
PYTHONPATH=$PWD:/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory/src \
/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory/.venv/bin/python \
  -m pytest tests/training/test_qwen3_owned_moe.py -q
```

Metrics/checks:

- `num_detected_moe_blocks == config.num_hidden_layers` for the tiny model.
- `router_trainable == []`.
- `expert_base_trainable == []`.

### Phase 3: Implement `AsymQwen3MoeBlock`

Implementation:

Add this import in `asym_gemm/training/qwen3_moe.py`:

```python
from contextlib import nullcontext
```

Add Qwen3 MoE block detection next to `is_qwen3_experts(...)`:

```python
def is_qwen3_moe_block(module: nn.Module) -> bool:
    gate = getattr(module, "gate", None)
    experts = getattr(module, "experts", None)
    if gate is None or experts is None or not is_qwen3_experts(experts):
        return False
    for attr in ("hidden_dim", "top_k", "num_experts"):
        if not isinstance(getattr(gate, attr, None), int):
            return False
    return callable(getattr(gate, "forward", None))
```

Add the owned MoE wrapper below `AsymQwen3Experts`:

```python
class AsymQwen3MoeBlock(nn.Module):
    _is_asym_qwen3_moe_block = True

    def __init__(
        self,
        source: nn.Module,
        *,
        backend: Literal["asym", "torch"],
        precision: Literal["bf16"],
        offload: bool,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        lora_dtype: torch.dtype | str | None,
        expert_recompute_policy: str,
        router_mode: Literal["whole"] = "whole",
        router_debug_grad: bool = False,
        stats: AsymExecutionStats | None = None,
        strict: bool = True,
    ) -> None:
        super().__init__()
        if router_mode != "whole":
            raise ValueError(f"AsymQwen3MoeBlock only implements router_mode='whole', got {router_mode!r}")
        if strict and not is_qwen3_moe_block(source):
            raise TypeError(f"source does not look like a Qwen3 MoE block: {type(source).__name__}")

        self.config = getattr(source, "config", None)
        self.backend = backend
        self.precision = precision
        self.offload = bool(offload)
        self.router_mode = router_mode
        self.router_debug_grad = bool(router_debug_grad)
        self.profile_prefix = "layers.unknown.mlp"

        # Register submodules in the same order as Transformers 5.6 Qwen3:
        # gate first, wrapped experts second. Do not store `source` on self.
        self.gate = getattr(source, "gate")
        self.experts = wrap_qwen3_experts(
            getattr(source, "experts"),
            backend=backend,
            precision=precision,
            offload=offload,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_dtype=lora_dtype,
            expert_recompute_policy=expert_recompute_policy,
            stats=stats,
            strict=strict,
        )

        self.hidden_dim = int(getattr(self.gate, "hidden_dim"))
        self.top_k = int(getattr(self.gate, "top_k"))
        self.num_experts = int(getattr(self.gate, "num_experts"))
        self.gate.requires_grad_(False)

    @property
    def cpu_resident_base_bytes(self) -> int:
        return int(self.experts.cpu_resident_base_bytes)

    @property
    def gpu_resident_base_bytes(self) -> int:
        return int(self.experts.gpu_resident_base_bytes)

    @property
    def trainable_lora_params(self) -> int:
        return int(self.experts.trainable_lora_params)

    def _profile_name(self, *parts: object) -> str:
        return scoped_name(self.profile_prefix, *parts)

    def _forward_range(self, *parts: object) -> str:
        return scoped_name("forward", self.profile_prefix, *parts)
```

This intentionally moves only `source.gate` and `source.experts` into the new
wrapper. Do not store `source` on `self`; keeping the whole original MoE block
as a child module would retain the original expert base weights and defeat the
memory goal.

Add the routing helper:

```python
def _compute_routing(
    self,
    hidden_states_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    context = nullcontext() if self.router_debug_grad else torch.no_grad()
    with context, prof_range(self._forward_range("router")):
        router_out = self.gate(hidden_states_2d)

    # Transformers 5.6 Qwen3MoeTopKRouter:
    #   (router_logits, routing_weights, selected_experts)
    if isinstance(router_out, tuple) and len(router_out) >= 3:
        top_k_weights = router_out[1]
        top_k_index = router_out[2]
        # Router aux loss is rejected for this first implementation, so do not
        # keep router_logits live past this point.
        if not self.router_debug_grad:
            top_k_weights = top_k_weights.detach()
            top_k_index = top_k_index.detach()
        if top_k_weights.dtype != hidden_states_2d.dtype:
            top_k_weights = top_k_weights.to(dtype=hidden_states_2d.dtype)
        return top_k_index, top_k_weights, None

    raise TypeError(
        "AsymQwen3MoeBlock requires a Qwen3MoeTopKRouter-style gate returning "
        "(router_logits, top_k_weights, top_k_index); "
        f"got {type(router_out).__name__}"
    )
```

Forward:

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_shape = hidden_states.shape
    if hidden_states.dim() != 3:
        raise ValueError(f"AsymQwen3MoeBlock expects [batch, seq, hidden], got {tuple(hidden_states.shape)}")
    flat = hidden_states.view(-1, input_shape[-1])
    top_k_index, top_k_weights, _router_logits = self._compute_routing(flat)
    if not self.router_debug_grad and top_k_weights.requires_grad:
        raise RuntimeError("router no-grad mode produced differentiable top_k_weights")
    with prof_range(self._forward_range("experts")):
        out = self.experts(flat, top_k_index, top_k_weights)
    return out.view(input_shape)
```

Add the constructor helper next to `wrap_qwen3_experts(...)`:

```python
def wrap_qwen3_moe_block(
    source: nn.Module,
    *,
    backend: Literal["asym", "torch"],
    precision: Literal["bf16"],
    offload: bool,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_dtype: torch.dtype | str | None = torch.bfloat16,
    expert_recompute_policy: str = "none",
    router_mode: Literal["whole"] = "whole",
    router_debug_grad: bool = False,
    stats: AsymExecutionStats | None = None,
    strict: bool = True,
) -> AsymQwen3MoeBlock:
    return AsymQwen3MoeBlock(
        source,
        backend=backend,
        precision=precision,
        offload=offload,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_dtype=lora_dtype,
        expert_recompute_policy=expert_recompute_policy,
        router_mode=router_mode,
        router_debug_grad=router_debug_grad,
        stats=stats,
        strict=strict,
    )
```

Do not detach `flat` before the expert call. KT detaches submitted hidden states
because `KTMoEFunction.backward()` manually returns `grad_input`; this
AsymGEMM-owned wrapper relies on the existing `AsymQwen3Experts` autograd path
for hidden/input and LoRA gradients.

Efficiency constraints for this wrapper:

- Use `view(...)`, not `reshape(...)`, for flatten/unflatten so an unexpected
  non-contiguous hidden state fails loudly instead of silently allocating a copy.
  The validation path must prove current LF Qwen3 hidden states are contiguous.
- Do not call `.contiguous()`, `.clone()`, `.cpu()`, `.tolist()`, `.item()`,
  `torch.cuda.synchronize()`, or `torch.cuda.empty_cache()` in
  `AsymQwen3MoeBlock.forward(...)` or `_compute_routing(...)`.
- Do not build one-hot expert masks, per-expert Python lists, or per-token
  Python loops in the owned wrapper. The wrapper only computes router top-k and
  forwards `(flat, top_k_index, top_k_weights)` to the existing
  `AsymQwen3Experts` route-metadata path.
- Do not retain `router_logits` in production `whole` mode. Router aux loss is
  rejected, so keeping logits live only increases memory lifetime.
- Do not convert `top_k_weights` if it already matches `hidden_states.dtype`.
  Installed Transformers 5.6 returns bf16 weights in the pure bf16 path; the
  conditional cast is only a safety guard.
- `detach()` is allowed for router outputs because it is metadata/view-like and
  does not allocate new storage. It must not be replaced by `.clone().detach()`.

For the first implementation, scope this to the installed Transformers 5.6
Qwen3 MoE block used by LF, where `Qwen3MoeSparseMoeBlock.forward` returns a
single tensor. Legacy tuple-returning MoE blocks and raw-logit-only router
contracts are explicitly unsupported in this implementation.

Export:

- Add `AsymQwen3MoeBlock`, `is_qwen3_moe_block`, and `wrap_qwen3_moe_block` to
  `asym_gemm/training/__init__.py`.

Validation before moving on:

Unit forward parity:

- Construct a tiny Qwen3 MoE block.
- Clone it into an owned wrapper with `backend="torch"`, `offload=False`,
  `lora_rank=1`, `lora_alpha=1`, `lora_dropout=0`.
- Zero all LoRA B weights so the wrapper is pure base.
- Run both modules in eval/no-grad.

Check:

```text
max_abs(original_out - owned_out) <= 1e-6 for fp32
max_abs(original_out - owned_out) <= 1e-3 for bf16
top_k_index exactly equal
top_k_weights max_abs <= 1e-6 for fp32, <= 1e-3 for bf16
```

For nonzero LoRA weights, compare `whole` against the existing
expert-only AsymGEMM wrapper given the same detached `top_k_index/top_k_weights`.
Do not compare nonzero expert LoRA output against vanilla HF, because vanilla HF
does not contain those packed expert LoRA weights.

Unit router autograd check:

- In `whole`, assert `top_k_weights.requires_grad is False`.
- In `router_debug_grad=True` test mode, assert `top_k_weights.requires_grad is True` when
  input hidden requires grad.
- Assert every `gate.*` parameter has `requires_grad=False`.
- Assert `list(wrapper._modules)[:2] == ["gate", "experts"]`.
- Assert `source` is not present in `wrapper._modules` and no original packed
  expert object remains reachable as a child module outside `wrapper.experts`.

Unit efficiency/allocation check:

- Wrap `AsymQwen3MoeBlock._compute_routing` and `AsymQwen3MoeBlock.forward` with
  tiny test hooks that record input/output storage pointers.
- Assert `flat` is a view of `hidden_states`, not a copy:

```python
assert flat.untyped_storage().data_ptr() == hidden_states.untyped_storage().data_ptr()
```

- Assert the final output reshape is also a view of the flat expert output when
  the expert output is contiguous.
- Use `torch.profiler.profile(profile_memory=True, record_shapes=True)` on one
  forward/backward and fail if these operators appear inside
  `forward.*.mlp.router` or the owned-wrapper glue outside the existing
  `AsymQwen3Experts` ranges:

```text
aten::clone
aten::contiguous
aten::copy_
aten::_to_copy
aten::to with a real dtype/device copy
```

Allowed allocations in the router range are the HF router's actual math outputs:
router logits, softmax probabilities, `topk` values, and `topk` indices. The
owned wrapper must not add extra copies around them.

### Phase 3B: Add Llama 4 Router Mode to Existing Whole-MoE Wrapper

Implementation:

Update `asym_gemm/training/llama4_moe.py`. Do not create a second Llama 4
wrapper class. Extend the existing `AsymLlama4Moe` because it already owns the
full Llama 4 MoE block.

Add imports:

```python
from contextlib import nullcontext
```

Extend `AsymLlama4Moe.__init__(...)`:

```python
router_mode: Literal["hf", "whole"] = "hf",
router_debug_grad: bool = False,
```

Add validation and fields:

```python
if router_mode not in {"hf", "whole"}:
    raise ValueError(f"router_mode must be 'hf' or 'whole', got {router_mode!r}")
self.router_mode = router_mode
self.router_debug_grad = bool(router_debug_grad)
self.router.requires_grad_(False)
```

Keep `router_mode="hf"` as the compatibility default. It must preserve the
current `AsymLlama4Moe.forward(...)` autograd behavior and output contract.

Add a routing helper:

```python
def _compute_routing(
    self,
    flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    no_grad_router = self.router_mode == "whole" and not self.router_debug_grad
    context = torch.no_grad() if no_grad_router else nullcontext()
    with context, prof_range(self._forward_range("router")):
        router_scores, router_logits = self.router(flat)
        _top_values, top_k_index = torch.topk(router_logits, self.top_k, dim=1)
        input_weights = router_scores.gather(1, top_k_index)
    if no_grad_router:
        router_logits = router_logits.detach()
        top_k_index = top_k_index.detach()
        input_weights = input_weights.detach()
    return top_k_index, input_weights, router_logits
```

Update `forward(...)`:

```python
def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape
    if hidden_states.dim() != 3:
        raise ValueError(f"AsymLlama4Moe expects [batch, seq, hidden], got {tuple(hidden_states.shape)}")
    flat = hidden_states.view(-1, self.hidden_dim)
    top_k_index, input_weights, router_logits = self._compute_routing(flat)
    if self.router_mode == "whole" and not self.router_debug_grad and input_weights.requires_grad:
        raise RuntimeError("Llama 4 router no-grad mode produced differentiable input_weights")
    with prof_range(self._forward_range("shared_expert")):
        out = self.shared_expert(flat)
    with prof_range(self._forward_range("experts")):
        routed = self.experts.forward_input_scaled(flat, top_k_index, input_weights)
    return out + routed.to(dtype=out.dtype), router_logits
```

Update `wrap_llama4_moe(...)` to accept and forward `router_mode` and
`router_debug_grad`.

Efficiency constraints:

- Use `view(...)`, not `reshape(...)`, for the wrapper flatten step so hidden
  copies fail loudly in validation.
- Do not use the HF `repeat(num_experts, 1)` dense-routing implementation in
  AsymGEMM. Keep using `top_k_index` plus
  `AsymPackedExperts.forward_input_scaled(...)`, which routes only selected
  experts.
- Do not call `.contiguous()`, `.clone()`, `.cpu()`, `.tolist()`, `.item()`,
  `torch.cuda.synchronize()`, or `torch.cuda.empty_cache()` inside
  `AsymLlama4Moe.forward(...)` or `_compute_routing(...)`.
- Do not detach `flat`; only detach router outputs in `router_mode=whole`.
- Return `router_logits` to preserve the installed Llama 4 MoE tuple contract,
  but in `router_mode=whole` it must be detached. Router aux loss is rejected
  for this mode, so detached logits are only for tuple-shape compatibility.

Validation before moving on:

Unit forward parity:

- Existing `test_asym_llama4_moe_torch_matches_eager_at_zero_delta` remains the
  `router_mode="hf"` parity test.
- Add the same zero-LoRA parity test with `router_mode="whole"`.
- Check both output and returned router logits:

```text
max_abs(hf_out - source_out) <= 3e-3 bf16
max_abs(whole_out - source_out) <= 3e-3 bf16
max_abs(hf_router_logits - source_router_logits) == 0
max_abs(whole_router_logits - source_router_logits) == 0
```

Unit router autograd check:

- In `router_mode="hf"`, with `hidden_states.requires_grad=True`, assert
  `input_weights.requires_grad is True`.
- In `router_mode="whole"`, assert `input_weights.requires_grad is False` and
  returned `router_logits.requires_grad is False`.
- In `router_debug_grad=True`, assert `input_weights.requires_grad is True`.
- Assert every `router.*` parameter has `requires_grad=False`.
- Assert `list(wrapper._modules)[:3] == ["router", "shared_expert", "experts"]`
  after construction.

Gradient correctness:

- Compare `router_mode="whole"` against a detached-router oracle that runs the
  original Llama 4 router once, detaches `router_logits/top_k_index/input_weights`,
  then calls the same shared expert and packed expert path.
- Expert LoRA gradients and hidden gradient from the expert/shared-expert paths
  must match the detached-router oracle.
- Hidden gradient may differ from full `router_mode="hf"` because the
  `loss -> router_scores/router_logits -> hidden` path is intentionally removed.

Integration test:

- Extend `tests/training/test_lf_qwen3_asym_backend.py` so
  `apply_lf_asym_lora(..., router_mode="hf")` keeps wrapping Llama 4 MoE blocks.
- Add `apply_lf_asym_lora(..., router_mode="whole")` for `FakeLlama4Model` and
  assert:

```text
report.llama4_moes_wrapped == num_layers
report.qwen3_moes_wrapped == 0
each feed_forward is AsymLlama4Moe
each feed_forward.router_mode == "whole"
router trainable params == []
expert/shared-expert/dense LoRA trainable params match hf mode
```

### Phase 4: Integrate Owned MoE Wrapping into LF AsymGEMM

Implementation:

Update imports in `asym_gemm/integrations/lf.py`:

```python
from asym_gemm.training.llama4_moe import AsymLlama4Moe, is_llama4_moe, wrap_llama4_moe
from asym_gemm.training.qwen3_moe import (
    AsymQwen3Experts,
    AsymQwen3MoeBlock,
    is_qwen3_experts,
    is_qwen3_moe_block,
    wrap_qwen3_experts,
    wrap_qwen3_moe_block,
)
```

Update `LFAsymReport`:

```python
qwen3_moes_wrapped: int = 0
llama4_moes_wrapped: int = 0
router_mode: str = "hf"
router_no_grad: bool = False
```

Update `count_lora_wrapped_modules(...)` to exclude both `AsymQwen3Experts` and
`AsymQwen3MoeBlock`. No direct `AsymLlama4Moe` exclusion is needed unless the
class later exposes top-level `lora_A/lora_B`; its trainable expert LoRA lives
under `AsymLlama4Moe.experts`.

Replacement logic:

```python
if wrap_experts:
    if router_mode == "whole":
        # Wrap whole MoE modules for model families that can own routing.
        for name, module in list(model.named_modules()):
            if is_qwen3_moe_block(module):
                wrapped = wrap_qwen3_moe_block(
                    module,
                    backend=backend,
                    precision=precision,
                    offload=offload_experts,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    lora_dtype=torch.bfloat16,
                    expert_recompute_policy=recompute_config.label,
                    router_mode="whole",
                    router_debug_grad=False,
                    stats=stats,
                    strict=strict,
                )
                wrapped.profile_prefix = _layer_profile_prefix_from_module_name(name, "mlp")
                wrapped.experts.profile_prefix = f"{wrapped.profile_prefix}.experts"
                expert_replacements.append((name, module, wrapped, f"{name}.experts"))
            elif is_llama4_moe(module):
                wrapped = wrap_llama4_moe(
                    module,
                    backend=backend,
                    precision=precision,
                    offload=offload_experts,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    lora_dtype=torch.bfloat16,
                    expert_recompute_policy=recompute_config.label,
                    router_mode="whole",
                    router_debug_grad=False,
                    stats=stats,
                    strict=strict,
                )
                wrapped.profile_prefix = _llama4_profile_prefix_from_module_name(name)
                wrapped.experts.profile_prefix = f"{wrapped.profile_prefix}.experts"
                expert_replacements.append((name, module, wrapped, f"{name}.experts"))
    else:
        # Existing expert-only replacement path.
        for name, module in list(model.named_modules()):
            if is_qwen3_experts(module):
                family = _packed_expert_family(module)
                wrapped = wrap_qwen3_experts(
                    module,
                    backend=backend,
                    precision=precision,
                    offload=offload_experts,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    lora_dtype=torch.bfloat16,
                    expert_recompute_policy=recompute_config.label,
                    router_mode="hf",
                    router_debug_grad=False,
                    stats=stats,
                    strict=strict,
                )
                wrapped.asym_expert_family = family
                if family == "gemma4":
                    wrapped.profile_prefix = _gemma4_profile_prefix_from_module_name(name)
                else:
                    wrapped.profile_prefix = _qwen3_profile_prefix_from_module_name(name)
                expert_replacements.append((name, module, wrapped, name))
            elif is_llama4_moe(module):
                wrapped = wrap_llama4_moe(
                    module,
                    backend=backend,
                    precision=precision,
                    offload=offload_experts,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    lora_dtype=torch.bfloat16,
                    expert_recompute_policy=recompute_config.label,
                    stats=stats,
                    strict=strict,
                )
                wrapped.profile_prefix = _llama4_profile_prefix_from_module_name(name)
                wrapped.experts.profile_prefix = f"{wrapped.profile_prefix}.experts"
                expert_replacements.append((name, module, wrapped, f"{name}.experts"))
```

Important ordering:

- In `whole`, do not also wrap child `mlp.experts` for Qwen3 or child
  `feed_forward.experts` for Llama 4; the full MoE wrapper already owns
  `experts`.
- Use the existing `expert_prefixes` skip mechanism with `skip_prefix=f"{name}.experts"`.
- Keep dense LoRA wrapping unchanged for attention and non-router dense modules.
- Keep `_validate_trainable_params(model)` unchanged and require it to pass.

Adapter save/load:

- Update `_infer_adapter_config(...)` to recognize `AsymQwen3MoeBlock` and to
  save `asym_router_mode` for `AsymLlama4Moe`.
- For Qwen3 `whole`, save:

```json
{
  "asym_expert_format": "qwen3_owned_moe",
  "asym_router_mode": "whole"
}
```

- For Llama 4, keep the existing format and add router mode:

```json
{
  "asym_expert_format": "llama4_packed_moe",
  "asym_expert_family": "llama4",
  "asym_router_mode": "hf|whole"
}
```

- Ensure the actual adapter state dict still includes dense PEFT LoRA plus
  AsymGEMM expert LoRA tensors, and still excludes router base weights.

Validation before moving on:

Use a tiny LF-style model:

- Apply PEFT attention LoRA first.
- Call `adapt_lf_asym_peft_lora` with `router_mode="whole"`.

Check:

```text
report.qwen3_moes_wrapped == num_hidden_layers
report.llama4_moes_wrapped == 0 for Qwen3 tests
report.packed_experts_wrapped == 0 or counts only via the owned wrapper, not double-counted
each owned wrapper registers child modules in order ["gate", "experts"]
owned wrapper does not keep original source MoE as a child/reference
original packed expert module is not reachable outside wrapper.experts
router trainable params == []
expert LoRA trainable params == expected rank-derived count
dense attention LoRA trainable params unchanged vs hf mode
no duplicate AsymQwen3Experts under the same layer
cpu_resident_base_bytes/gpu_resident_base_bytes do not increase vs hf mode except for router wrapper bookkeeping
```

Repeat the integration check with `FakeLlama4Model`:

```text
report.llama4_moes_wrapped == num_hidden_layers
report.qwen3_moes_wrapped == 0
each feed_forward is AsymLlama4Moe
each feed_forward.router_mode == "whole"
each feed_forward.experts.profile_prefix == "layers.N.feed_forward.experts"
router trainable params == []
adapter_config.json keeps asym_expert_format="llama4_packed_moe"
adapter_config.json records asym_router_mode="whole"
```

### Phase 5: Correctness Tests for Backward Semantics

The Qwen3 and Llama 4 router-owned paths must be validated against two
references:

1. Full HF reference:

```text
router autograd enabled, route-weight gradient to hidden included
```

2. Detached-router oracle:

```text
same router forward, but route metadata is detached before expert call
Qwen3 route metadata: top_k_index/top_k_weights
Llama 4 route metadata: top_k_index/input_weights/router_logits
```

The production `whole` mode should match reference 2, not reference 1.

Implementation:

Add a test helper in `tests/training/test_qwen3_owned_moe.py`:

```python
def run_detached_router_oracle(source_moe, hidden):
    flat = hidden.reshape(-1, hidden.shape[-1])
    with torch.no_grad():
        _, weights, ids = source_moe.gate(flat)
    out = source_moe.experts(flat, ids, weights)
    return out.reshape_as(hidden)
```

Add a Llama 4 helper in `tests/training/test_lf_qwen3_asym_backend.py`:

```python
def run_llama4_detached_router_oracle(wrapper, hidden):
    flat = hidden.view(-1, wrapper.hidden_dim)
    with torch.no_grad():
        router_scores, router_logits = wrapper.router(flat)
        _top_values, top_k_index = torch.topk(router_logits, wrapper.top_k, dim=1)
        input_weights = router_scores.gather(1, top_k_index)
    shared = wrapper.shared_expert(flat)
    routed = wrapper.experts.forward_input_scaled(flat, top_k_index, input_weights)
    return shared + routed.to(dtype=shared.dtype), router_logits.detach()
```

For LoRA gradient parity:

- Initialize owned wrapper with `backend="torch"` and `lora_dropout=0`.
- Use deterministic nonzero LoRA A/B weights.
- Compute loss as `out.float().pow(2).mean()`.
- Compare `whole` against the detached-router oracle implemented using
  the same `AsymQwen3Experts` expert object or an equivalent torch expert path.
- For Llama 4, compare `AsymLlama4Moe(router_mode="whole")` against
  `run_llama4_detached_router_oracle(...)` using the same wrapper instance.

Check:

```text
forward max_abs <= 1e-3 bf16
loss abs <= 1e-4
LoRA grad max_abs <= 2e-3 bf16
hidden grad max_abs <= 2e-3 bf16 vs detached-router oracle
router grad params == none
whole hidden grad may differ from full HF reference; report it, do not fail on that difference
```

Saved-tensor memory unit:

- Use `torch.autograd.graph.saved_tensors_hooks`.
- Run one forward/backward for:
  - Qwen3 current expert-only `hf` path
  - Qwen3 new `whole` path
  - Llama 4 current `hf` wrapper path
  - Llama 4 new `whole` wrapper path
- Attribute saved bytes by `prof_range`.

Check:

```text
whole saves zero tensors under forward.*.router
hf path may save router softmax/topk/sigmoid-related tensors
whole total saved bytes <= hf saved bytes
```

Gradient-checkpointing smoke, added because KT needed special handling for
non-reentrant checkpointing around its C++ forward cache:

- Enable LF/Trainer gradient checkpointing or call a tiny decoder layer through
  `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`.
- Run `router_mode=whole`, `backend="torch"`, `lora_dropout=0`, and one
  forward/backward.
- Compare loss and LoRA gradients against the no-checkpoint detached-router
  oracle within the same tolerances above.
- Assert no "No forward cache available" or saved-tensor hook ordering error is
  raised. If a future backend introduces detached submit/cache behavior like KT,
  add an explicit custom-autograd sentinel then; do not add it for the first
  Python/GPU owned-wrapper path.

### Phase 6: Profiling Instrumentation

Implementation:

1. Add `prof_range` coverage in `AsymQwen3MoeBlock`:

```text
forward.layers.N.mlp.router
forward.layers.N.mlp.experts
```

2. Preserve and audit `prof_range` coverage in `AsymLlama4Moe`:

```text
forward.layers.N.feed_forward.router
forward.layers.N.feed_forward.shared_expert
forward.layers.N.feed_forward.experts
```

3. Update `asym_gemm/profiling/lf_trace.py`:

- `_semantic_module_name(...)` should recognize router-owned MoE blocks and
  router submodules if module hooks can see them.
- `_filter_token(...)` should return `router` for router ranges.
- Default `PROFILE_MODULE_FILTER` can remain unchanged, but scripts used for
  this feature should pass:

```text
PROFILE_MODULE_FILTER=attention,router,mlp,experts,lora,optimizer
```

4. Ensure runtime logs include:

```text
router_mode=whole
qwen3_moes_wrapped=N
llama4_moes_wrapped=N
router_no_grad=True
```

Validation before moving on:

Small source-profile run:

```bash
cd /home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_torch|norecompute" \
ROUTER_MODES="hf,whole" \
PROFILERS=source \
PROFILE_MEMORY_ATTRIBUTION=1 \
PROFILE_LEVEL=op \
PROFILE_MODULE_FILTER=attention,router,mlp,experts,lora,optimizer \
SEQ_LENS=64 \
MAX_SAMPLES=1 \
MAX_STEPS=1 \
WARMUP_STEPS=0 \
LORA_RANK=8 \
LORA_ALPHA=16 \
LORA_DROPOUT=0.00 \
OVERWRITE=true \
scripts/lf/profile_lora_lf.sh
```

Check in each `profile.json`:

```text
peak CUDA allocated/reserved
saved_tensors.total_unique_bytes
saved_tensors.by_owner entries containing "router"
wall time
train loss
AsymGEMM runtime call counts
operator list for owned-wrapper ranges
```

Expected:

- `whole` and `hf` have near-identical first-step forward loss.
- `whole` has lower router-attributed saved tensor bytes.
- AsymGEMM expert call counts remain positive.
- No router trainable params appear in logs.
- No new CPU copies, CUDA synchronizations, `aten::clone`, or unconditional
  dtype/device copies are attributed to the owned-wrapper glue. If profiler
  shows a copy, classify it as one of:

```text
expected HF router output allocation
expected existing packed expert route packing/scatter allocation
bug: owned-wrapper glue copy
```

Only the first two are acceptable before moving on.

Repeat the same source-profile command with Llama 4:

```bash
cd /home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM
GPU_POOL=3 \
MODEL_SPECS="meta-llama/Llama-4-Scout-17B-16E|1" \
BACKEND_SPECS="asym_torch|norecompute" \
ROUTER_MODES="hf,whole" \
PROFILERS=source \
PROFILE_MEMORY_ATTRIBUTION=1 \
PROFILE_LEVEL=op \
PROFILE_MODULE_FILTER=attention,router,mlp,experts,lora,optimizer \
SEQ_LENS=64 \
MAX_SAMPLES=1 \
MAX_STEPS=1 \
WARMUP_STEPS=0 \
LORA_RANK=8 \
LORA_ALPHA=16 \
LORA_DROPOUT=0.00 \
OVERWRITE=true \
scripts/lf/profile_lora_lf.sh
```

Llama 4 checks:

```text
llama4_moes_wrapped > 0
qwen3_moes_wrapped == 0
feed_forward.router ranges appear
feed_forward.shared_expert ranges appear
whole router saved tensor bytes <= hf router saved tensor bytes
whole first-step loss delta vs hf <= 5e-3 bf16
no extra clone/copy/sync in AsymLlama4Moe wrapper glue
```

### Phase 7: Full 4k Validation

Run the real target workload with both source profiling and Nsight.

Source profile:

```bash
cd /home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_torch|norecompute,asym|norecompute" \
ROUTER_MODES="hf,whole" \
PROFILERS=source \
PROFILE_MEMORY_ATTRIBUTION=1 \
PROFILE_LEVEL=op \
PROFILE_MODULE_FILTER=attention,router,mlp,experts,lora,optimizer \
SEQ_LENS=4096 \
MAX_SAMPLES=1 \
MAX_STEPS=1 \
WARMUP_STEPS=0 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 \
LORA_RANK=8 \
LORA_ALPHA=16 \
LORA_DROPOUT=0.00 \
OVERWRITE=true \
scripts/lf/profile_lora_lf.sh
```

Nsight:

```bash
cd /home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym|norecompute" \
ROUTER_MODES="hf,whole" \
PROFILERS=nsys \
PROFILE_LEVEL=op \
PROFILE_MODULE_FILTER=attention,router,mlp,experts,lora,optimizer \
SEQ_LENS=4096 \
MAX_SAMPLES=1 \
MAX_STEPS=1 \
WARMUP_STEPS=0 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 \
LORA_RANK=8 \
LORA_ALPHA=16 \
LORA_DROPOUT=0.00 \
OVERWRITE=true \
scripts/lf/profile_lora_lf.sh
```

Llama 4 full-length validation:

- Run the same source-profile command with
  `MODEL_SPECS="meta-llama/Llama-4-Scout-17B-16E|1"` and `SEQ_LENS=4096` after
  the Qwen3 run passes.
- If the one-GPU Llama 4 4k run is blocked by model access or memory, do not
  block Qwen3 implementation on it. Keep Llama 4 support gated by the unit tests
  plus the Phase 6 Llama 4 seq-64 source-profile smoke, and record the 4k block
  explicitly in the progress doc.

Metrics to record:

```text
backend
router_mode
seq_len
train loss step 1
delta loss vs hf router mode
runtime seconds
peak CUDA allocated
peak CUDA reserved
saved_tensors.total_unique_bytes
saved_tensors router-owned bytes
saved_tensors mlp/expert bytes
asym_forward_calls
asym_dx_calls
torch_forward_calls
torch_dx_calls
fallback_reasons
owned_wrapper_clone_or_copy_ops
owned_wrapper_cpu_copy_ops
owned_wrapper_cuda_sync_ops
nsys total CUDA kernel time
nsys router NVTX time
nsys expert/base GEMM time
```

Expected rough memory effect:

- Router no-grad should save router-side autograd state, likely hundreds of MB
  at 4k for Qwen3-30B-A3B, not tens of GB.
- The exact number must come from `saved_tensors.by_owner` plus peak HBM.
- If peak HBM improvement is below noise, still keep the mode only if
  saved-tensor attribution proves router graph removal and there is no loss or
  runtime regression.

Pass criteria:

```text
first-step loss delta vs hf router mode <= 1e-3 preferred, <= 5e-3 allowed for bf16
whole peak HBM <= hf peak HBM
router saved tensor bytes reduced to zero or near-zero
expert AsymGEMM call counts unchanged
no router trainable params
no new reference fallbacks
no new NaN/Inf
owned_wrapper_clone_or_copy_ops == 0, excluding documented HF router math and existing packed expert route packing/scatter
owned_wrapper_cpu_copy_ops == 0
owned_wrapper_cuda_sync_ops == 0
```

### Phase 8: Regression and Save/Load Validation

Implementation checks:

- Existing `asym_router_mode=hf` must reproduce current results and current
  wrapper counts.
- Adapter config must save and load with the new `asym_router_mode`.
- Loading an `hf` adapter must not require the owned MoE wrapper.
- Loading a `whole` adapter should recreate the same owned MoE wrapper.

Suggested validation:

```bash
cd /home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM
PYTHONPATH=$PWD:/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory/src \
/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory/.venv/bin/python \
  -m pytest tests/training/test_qwen3_owned_moe.py tests/training/test_lf_qwen3_asym_backend.py -q
```

Then run one save/load smoke:

```bash
cd /home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM
GPU_ID=3 \
BACKEND=asym_torch \
ASYM_ROUTER_MODE=whole \
MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B \
DATASET=asym_long_sft_smoke__qwen3-30b-a3b__s64 \
CUTOFF_LEN=64 \
MAX_SAMPLES=1 \
MAX_STEPS=1 \
PROFILE=0 \
OVERWRITE=true \
scripts/lf/run_lf_lora_sft.sh
```

Check:

```text
adapter_config.json contains asym_router_mode=whole
adapter can be loaded for inference/training resume
router params are not in trainable adapter state
expert LoRA tensors are present
```

## Required File Changes

First implementation changes exactly the files below. Do not modify CUDA/C++
kernel files, `third_party/ktransformers`, or LF v1 plugin config for this
owned-router pass.

AsymGEMM Python code:

- `third_party/AsymGEMM/asym_gemm/training/qwen3_moe.py`
  - Add `is_qwen3_moe_block(module)`.
  - Add `AsymQwen3MoeBlock`.
  - Add `wrap_qwen3_moe_block(...)`.
  - Register `gate` before `experts` inside `AsymQwen3MoeBlock`.
  - Do not retain the original source MoE block after wrapping.
  - Detach only router outputs in production `whole` mode; do not detach the
    hidden-state tensor passed to `AsymQwen3Experts`.
  - Add router/expert `prof_range` names:
    `forward.layers.N.mlp.router` and `forward.layers.N.mlp.experts`.
  - Keep existing `AsymQwen3Experts` behavior unchanged for `router_mode=hf`.

- `third_party/AsymGEMM/asym_gemm/training/llama4_moe.py`
  - Add `router_mode: Literal["hf", "whole"] = "hf"` and
    `router_debug_grad: bool = False` to `AsymLlama4Moe`.
  - Keep `router_mode=hf` as the current behavior.
  - In `router_mode=whole`, run router/top-k/input-weight computation under
    `torch.no_grad()` and detach `router_logits`, `top_k_index`, and
    `input_weights`.
  - Do not detach the hidden-state tensor passed to `shared_expert` or
    `AsymPackedExperts.forward_input_scaled(...)`.
  - Use `view(...)`, not `reshape(...)`, for the wrapper flatten step.
  - Preserve router/shared-expert/expert `prof_range` names:
    `forward.layers.N.feed_forward.router`,
    `forward.layers.N.feed_forward.shared_expert`, and
    `forward.layers.N.feed_forward.experts`.
  - Update `wrap_llama4_moe(...)` to forward `router_mode` and
    `router_debug_grad`.

- `third_party/AsymGEMM/asym_gemm/training/__init__.py`
  - Export `AsymQwen3MoeBlock`, `is_qwen3_moe_block`, and
    `wrap_qwen3_moe_block`.

- `third_party/AsymGEMM/asym_gemm/integrations/lf.py`
  - Import `AsymQwen3MoeBlock`, `is_qwen3_moe_block`, and
    `wrap_qwen3_moe_block`.
  - Continue importing `AsymLlama4Moe`, `is_llama4_moe`, and
    `wrap_llama4_moe`; pass `router_mode` to `wrap_llama4_moe(...)`.
  - Add `router_mode: Literal["hf", "whole"] = "hf"` to
    `apply_lf_asym_lora(...)`.
  - Reject `router_mode=whole` when `model.config.output_router_logits=True`.
  - Extend `LFAsymReport` with `qwen3_moes_wrapped`, `llama4_moes_wrapped`,
    `router_mode`, and `router_no_grad`.
  - Update `to_log_string()` and `runtime_log_string()` to include router mode.
  - Update `count_lora_wrapped_modules(...)` so owned MoE wrappers are not
    counted as dense LoRA wrappers.
  - In `router_mode=whole`, wrap full Qwen3 MoE blocks and full Llama 4 MoE
    blocks; skip child `mlp.experts` / `feed_forward.experts` replacement.
  - In `router_mode=hf`, keep the existing Qwen3 expert-only path and existing
    Llama 4 full-MoE wrapper behavior.
  - Update `_infer_adapter_config(...)` to recognize `AsymQwen3MoeBlock` and
    save `asym_expert_format="qwen3_owned_moe"` plus
    `asym_router_mode="whole"`.
  - Update `_infer_adapter_config(...)` for `AsymLlama4Moe` so Llama 4 adapters
    keep `asym_expert_format="llama4_packed_moe"` and also save
    `asym_router_mode`.

- `third_party/AsymGEMM/asym_gemm/integrations/peft_lf.py`
  - Add `router_mode` to `adapt_lf_asym_peft_lora(...)`.
  - Pass it through to `apply_lf_asym_lora(...)`.

- `third_party/AsymGEMM/asym_gemm/profiling/lf_trace.py`
  - Teach `_semantic_module_name(...)` / `_filter_token(...)` to preserve
    owned-router ranges as `router`.
  - Ensure `PROFILE_MODULE_FILTER=attention,router,mlp,experts,lora,optimizer`
    captures the new ranges.

LlamaFactory integration:

- `third_party/LlamaFactory/src/llamafactory/hparams/model_args.py`
  - Add `asym_router_mode: Literal["hf", "whole"] = "hf"` to
    `AsymGEMMArguments`.

- `third_party/LlamaFactory/src/llamafactory/hparams/parser.py`
  - Validate `asym_router_mode`.
  - Reject `use_asym_gemm && asym_router_mode=="whole" && moe_aux_loss_coef`.

- `third_party/LlamaFactory/src/llamafactory/model/adapter.py`
  - Pass `model_args.asym_router_mode` to `adapt_lf_asym_peft_lora(...)`.

- `third_party/LlamaFactory/src/llamafactory/train/sft/trainer.py`
  - Add `asym_router_mode` to the metadata passed to
    `save_asym_peft_adapter(...)`.

Profiling/scripts:

- `third_party/AsymGEMM/scripts/lf/run_lf_lora_sft.sh`
  - Add `ASYM_ROUTER_MODE=${ASYM_ROUTER_MODE:-hf}`.
  - Pass `--asym_router_mode "${ASYM_ROUTER_MODE}"`.
  - Preserve `ASYM_GEMM_LF_CONFIG_BACKEND="${PROFILE_BACKEND_LABEL:-${BACKEND}}"`
    so `asym_torch` is not recorded as plain `torch`.
  - Add `ASYM_GEMM_LF_CONFIG_ROUTER_MODE="${ASYM_ROUTER_MODE}"`.

- `third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh`
  - Preserve `asym_torch` as a distinct backend in `backend_label(...)` and
    `backend_gpu_count(...)`.
  - Add `ROUTER_MODES=${ROUTER_MODES:-hf}` and `router_mode_label(...)`.
  - Add `router_mode` to `run_job(...)`, `job_root_path(...)`, `run_id`,
    `group_key`, `compare_group_labels`, `jobs.tsv`, `run_env`, and the outer
    sweep loop.

- `third_party/AsymGEMM/scripts/plotting/plot_activation_recompute_sweep.py`
  - Add `asym_torch` to `BACKENDS` and `BACKEND_MARKERS`.
  - Update `parse_flat_result_dir(...)` to parse both old four-part flat job
    folders and new five-part `__router${router_mode}` folders.
  - Return and propagate `router_mode` through group keys, threshold group keys,
    varied-field labels, combined labels, and output directory names.

- `third_party/AsymGEMM/scripts/lf/postprocess_lf_profile_artifacts.py`
  - Show a source-profile summary markdown line of `Router mode: \`hf\`` or
    `Router mode: \`whole\`` from `profile["config"]["router_mode"]`.

Tests:

- `third_party/AsymGEMM/tests/training/test_qwen3_owned_moe.py`
  - New focused tests for owned-router detection, forward parity,
    detached-router gradient parity, saved-tensor router memory, and trainable
    parameter audit.
  - Add KT-informed tests for wrapper child order, no retained source MoE, no
    detached expert input, and non-reentrant gradient-checkpointing smoke.
  - Add efficiency tests proving flatten/unflatten are views, router logits are
    not retained, router weights are not cloned, and profiler memory does not
    show owned-wrapper `clone/contiguous/copy_/to/cpu/sync` work.

- `third_party/AsymGEMM/tests/training/test_lf_qwen3_asym_backend.py`
  - Extend existing LF/Qwen3 integration tests for `router_mode=hf` compatibility,
    `router_mode=whole` wrapper counts, no double wrapping, adapter config
    inference, save/load metadata, router trainability, and no duplicate original
    expert storage after owned wrapping.
  - Extend Llama 4 tests for `AsymLlama4Moe(router_mode="hf")` parity,
    `AsymLlama4Moe(router_mode="whole")` detached-router parity, router no-grad
    checks, LF integration wrapper counts, adapter metadata, and profile-prefix
    coverage.

Files intentionally not changed in the first implementation:

- `third_party/AsymGEMM/scripts/lf/run_lf_profiled_train.py`
  - No change needed because it already imports every
    `ASYM_GEMM_LF_CONFIG_*` environment variable into `profile.json["config"]`.

- `third_party/LlamaFactory/src/llamafactory/v1/config/model_args.py`
  - No change for this pass because the current `scripts/lf/profile_lora_lf.sh`
    path uses the classic LF hparams/model adapter path, not LF v1 plugin
    config.

- `third_party/AsymGEMM/scripts/lf/profile_lora_lf_kt.sh` and
  `third_party/AsymGEMM/scripts/lf/profile_lora_lf_fused.sh`
  - No change for the owned-router AsymGEMM validation path. Mirror the router
    axis there only if a later comparison explicitly uses those scripts.

- `third_party/AsymGEMM/asym_gemm/include/**`,
  `third_party/AsymGEMM/asym_gemm/csrc/**`, and kernel binding files
  - No change. This feature changes Python routing ownership/autograd semantics,
    not kernel math.

- `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/ktransformers/**`
  - No change. KT is used as a design reference only; this AsymGEMM feature must
    remain implemented in AsymGEMM/LlamaFactory files.

Do not add a separate helper script in the first implementation. If a later
iteration needs a faster local reproducer, it may add
`third_party/AsymGEMM/scripts/lf/profile_owned_moe_router.py`, but that script
must duplicate the same forward/backward parity and saved-tensor checks as
`tests/training/test_qwen3_owned_moe.py` and the Llama 4 tests in
`tests/training/test_lf_qwen3_asym_backend.py`; it is not a replacement for the
required LF profiling path.

## Risks and How to Resolve Them

1. Full HF gradient is not identical.

   Resolution: document and test against a detached-router oracle. This is the
   same SFT assumption KT makes. Do not compare hidden gradients against full HF
   as a pass/fail criterion for `whole`; report the difference.

2. Legacy Qwen3 blocks may return `(hidden, router_logits)`.

   Resolution: first scope `whole` to the installed Transformers 5.6
   path used by LF, where the block returns only hidden states. Add explicit
   tests before enabling legacy tuple-returning blocks.

3. Double wrapping experts.

   Resolution: when `router_mode=whole`, wrap the full MoE block and
   skip child expert replacement entirely. Add tests that count wrappers.

4. Router accidentally becomes trainable.

   Resolution: keep `_validate_trainable_params` unchanged and add a direct
   `router_trainable == []` test after wrapping.

5. Reported HBM savings are noisy.

   Resolution: require both peak HBM and saved-tensor attribution. The feature
   is proven if router-attributed saved tensors disappear, even if peak HBM
   moves by less than allocator noise on one run.

6. KT detach behavior is copied too broadly.

   Resolution: detach only router outputs. KT detaches submitted hidden states
   because its custom `KTMoEFunction.backward()` manually returns input
   gradients. The first AsymGEMM owned-router wrapper keeps the existing
   `AsymQwen3Experts` / `AsymPackedExperts` autograd path, so detaching `flat`
   before expert compute would incorrectly drop expert-path hidden gradients.
   Add unit tests that `flat.requires_grad` reaches `AsymQwen3Experts.forward(...)`
   for Qwen3 and `AsymPackedExperts.forward_input_scaled(...)` for Llama 4 in
   `whole` mode.

7. The owned wrapper adds hidden allocation/copy overhead.

   Resolution: use view-only flatten/unflatten, drop router logits immediately,
   and forbid glue-code `clone`, `contiguous`, `copy_`, CPU transfer, or
   synchronization in the wrapper forward path. Validate with storage-pointer
   tests and profiler operator checks. If a non-contiguous hidden state appears
   in the real LF path, stop and design an explicit policy instead of silently
   inserting `.contiguous()`.

8. Llama 4 already wraps the whole MoE block.

   Resolution: do not create another Llama 4 wrapper. Extend `AsymLlama4Moe`
   with `router_mode`; `hf` must preserve existing behavior and `whole` must
   only change the router autograd boundary.

9. Llama 4 returns router logits even when aux loss is disabled.

   Resolution: preserve the tuple contract `(hidden, router_logits)`, but return
   detached logits in `whole`. The LF decoder patch discards tuple logits in the
   no-aux-loss path; aux loss is rejected before wrapping.

## Final Acceptance Criteria

The implementation is ready only when all are true:

```text
1. asym_router_mode=hf keeps current behavior.
2. asym_router_mode=whole wraps each Qwen3 MoE block once and sets each Llama 4 AsymLlama4Moe to router_mode=whole.
3. Router weights are frozen and have no grads.
4. Router outputs in whole mode do not require grad.
5. Forward output matches HF for the same weights for Qwen3 and Llama 4.
6. Backward matches a detached-router oracle for LoRA grads and expert hidden grads for Qwen3 and Llama 4.
7. 4k LF SFT on cuda:3 runs for Qwen/Qwen3-30B-A3B.
8. Llama 4 seq-64 LF source-profile smoke passes for hf and whole; Llama 4 4k is run if one-GPU model access/memory allows, otherwise the block is recorded.
9. profile.json shows router saved-tensor bytes removed or near-zero.
10. Peak HBM is no worse than hf router mode.
11. Runtime is not materially worse; any regression above 3 percent must be explained by profile ranges.
12. Adapter save/load still works and records asym_router_mode for Qwen3 and Llama 4.
13. profile.json records backend=asym_torch for AsymGEMM torch-backend runs and router_mode=hf|whole.
14. Plot outputs keep routerhf and routerwhole as separate series/groups.
15. Owned-wrapper glue adds no hidden `clone/contiguous/copy_/cpu/sync` work; any copy shown by profiling is either HF router math or existing packed expert route packing/scatter.
16. Flatten/unflatten in AsymQwen3MoeBlock and AsymLlama4Moe are view-only and do not allocate.
```
