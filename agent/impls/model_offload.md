# Whole-Model Frozen-Weight Offload for LF AsymGEMM LoRA SFT

This is an implementation design, not an implementation patch. The target is
highly configurable CPU residency for frozen base weights in the LLaMA-Factory
AsymGEMM LoRA SFT path.

The default must remain unchanged:

```bash
ASYM_OFFLOAD_MODULES=${ASYM_OFFLOAD_MODULES:-routed_experts}
```

The final design must allow selecting any supported frozen-weight bucket:

```bash
ASYM_OFFLOAD_MODULES=routed_experts
ASYM_OFFLOAD_MODULES=routed_experts,attention
ASYM_OFFLOAD_MODULES=router,shared_experts
ASYM_OFFLOAD_MODULES=attention,embed_tokens,lm_head
ASYM_OFFLOAD_MODULES=norms
ASYM_OFFLOAD_MODULES=all
ASYM_OFFLOAD_MODULES=none
```

For this design, "whole model" means all frozen base weights in supported MoE
decoder-only models. Trainable LoRA parameters, optimizer state, activations,
temporary workspaces, logits, and loss tensors stay outside this selector.

## Current State

The current LF integration is expert-only:

- `scripts/lf/run_lf_lora_sft.sh` defaults `ASYM_OFFLOAD_MODULES` to
  `routed_experts`.
- `third_party/LlamaFactory/src/llamafactory/hparams/model_args.py` types
  `asym_offload_modules` as `Literal["routed_experts", "none"]`.
- `third_party/LlamaFactory/src/llamafactory/hparams/parser.py` rejects every
  selector except `routed_experts` and `none`.
- `asym_gemm/integrations/lf.py::apply_lf_asym_lora()` accepts only
  `routed_experts` and `none`.
- `asym_gemm/integrations/peft_lf.py::adapt_lf_asym_peft_lora()` calls
  `apply_lf_asym_lora()` with `wrap_dense=False`, so real LF dense targets are
  handled by normal PEFT before AsymGEMM wraps packed experts.
- `third_party/LlamaFactory/src/llamafactory/model/loader.py` loads on CPU first
  only when `asym_offload_modules == "routed_experts"`, then calls
  `model.to(cuda)` after adapter setup.

`HostWeight` is the core residency primitive. It is not a parameter or buffer,
so `Module.to("cuda")` cannot migrate it into HBM. Every CPU-selected bucket
must be converted into a HostWeight-backed wrapper before LlamaFactory moves the
rest of the model to CUDA.

The wrapper must not create a second persistent CPU copy of a selected base
weight. For production offload, LF must load the base model on CPU first, then
the offload wrapper must adopt the existing CPU tensor storage into `HostWeight`
and replace the original module/parameter owner. The old module must become
unreferenced before LF moves the remaining model state to CUDA.

## Component Scope

This design is for supported MoE models, not generic dense LLMs. The final
selector buckets are:

| Bucket | Meaning | Final token aliases |
| --- | --- | --- |
| `routed_experts` | Packed routed expert base weights already handled today. | `routed_experts`, `routed`, `experts` |
| `router` | MoE router/gate base projection. | `router`, `gate`, `expert_router` |
| `shared_experts` | Shared expert MLP branch and shared expert gate, when present. | `shared_experts`, `shared_expert`, `shared` |
| `attention` | Attention projection bases. | `attention`, `attn`, `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| `embed_tokens` | Input token embedding table. | `embed_tokens`, `embedding`, `embeddings`, `token_embeddings` |
| `lm_head` | Output projection. | `lm_head`, `output_embedding`, `output_embeddings` |
| `norms` | RMSNorm/LayerNorm weights, including final norm and q/k norms. | `norms`, `norm`, `layernorm`, `rmsnorm` |
| `all` | All implemented buckets above. | `all`, `whole_model`, `model` |
| `none` | No AsymGEMM CPU model-weight offload. | `none` |

`mlp_dense` is intentionally not part of `all` for the MoE-only target.
`mlp_dense` is an invalid `ASYM_OFFLOAD_MODULES` token in this feature. If a
mixed dense/MoE architecture exposes ordinary non-expert FFN projections, the
residency audit must report them as `mlp_dense` and must leave them
GPU-resident.

## Non-Negotiable Semantics

1. `lora_target` decides where adapters are trained.
2. `asym_offload_modules` decides where frozen base weights live.
3. Only `asym_backend=asym` performs CPU-resident execution. `asym_backend=torch`
   remains a GPU-resident reference path even when the same selector string is
   supplied.
4. The production kernel target for this work is SM100 BF16 only. Do not add
   validation, fallback policy, or implementation work for non-SM100 or non-BF16
   kernels in this feature.
5. No unimplemented selector token is silently accepted. At each stage, parser
   validation must reject tokens that are not implemented in that stage.
6. No selected frozen base weight remains as a CUDA parameter or persistent CUDA
   buffer after LF's device move.
7. No frozen base weight receives a gradient.
8. The optimizer parameter set must contain only LoRA parameters.
9. Adapter save/load must continue to save only adapter state plus metadata, not
   frozen base weights.
10. Selected offload wrappers must not duplicate frozen base weights in CPU RAM.
    The steady-state model after wrapper installation has at most one CPU-resident
    copy of each selected base weight, not the original CPU parameter plus a
    cloned HostWeight copy.

## CPU Memory Ownership Policy

The offload path is an ownership transfer, not a duplication pass.

1. CPU offload requires CPU-first model loading for every selected component.
2. If a selected source weight is already on CPU, create the `HostWeight` by
   adopting that tensor storage with `clone=False`.
3. Replace the owning module immediately after constructing the wrapper so the
   original `nn.Parameter` object is no longer reachable from the model.
4. After each replacement batch, drop local references to old modules and allow
   garbage collection before LF calls `model.to(cuda)`.
5. If a selected source weight is already on CUDA in a strict production
   `asym_backend=asym` run, fail with a placement error. Do not copy CUDA weights
   back to CPU as the normal implementation path.
6. Pinning must not leave two persistent CPU copies. GEMM-backed buckets
   (`routed_experts`, `router`, `shared_experts`, `attention`, and `lm_head`)
   need pinned CPU HostWeights when CUDA is available because the direct
   AsymGEMM kernel requires pinned host memory. If the CPU-first source tensor is
   already pinned, adopt it exactly. Otherwise create one pinned replacement CPU
   owner during wrapping, then immediately remove the original parameter owner
   from the model. Non-GEMM buckets (`embed_tokens` and `norms`) do not need
   pinned HostWeights.
7. Temporary Python references during a single module replacement are acceptable;
   retaining both the original CPU parameter storage and a cloned HostWeight
   after wrapping is not acceptable.

This policy applies to experts, router projections, shared experts, attention
projections, embeddings, `lm_head`, and norms.

Layout compatibility rule:

- Qwen3 and Qwen3.5 routed expert weights already match the grouped kernel
  layout, so their selected expert wrappers must exact-adopt the source CPU
  storage with `clone=False`.
- Llama 4 routed expert weights are stored transposed relative to the current
  Qwen-style packed expert wrapper. Until a native transposed grouped wrapper is
  added, Llama 4 may create one layout-normalized CPU owner during replacement.
  This is a replacement, not a duplicate: the original Llama 4 expert parameters
  must be removed from the model immediately, and the residency audit must show
  only the normalized HostWeight owner remains. A future zero-layout-copy Llama
  4 path requires native transposed grouped execution instead of hidden cloning.

## Selector Data Model

Add a single parser in `asym_gemm/integrations/lf.py` and reuse it from
LLaMA-Factory validation, loader placement checks, adapter setup, tests, and
reports.

```python
@dataclass(frozen=True)
class LFOffloadSelection:
    raw: str
    routed_experts: bool
    router: bool
    shared_experts: bool
    attention_targets: frozenset[str]
    embed_tokens: bool
    lm_head: bool
    norms: bool

    @property
    def any_cpu_offload(self) -> bool:
        return (
            self.routed_experts
            or self.router
            or self.shared_experts
            or bool(self.attention_targets)
            or self.embed_tokens
            or self.lm_head
            or self.norms
        )

    @property
    def implemented_components(self) -> frozenset[str]:
        components: set[str] = set()
        if self.routed_experts:
            components.add("routed_experts")
        if self.router:
            components.add("router")
        if self.shared_experts:
            components.add("shared_experts")
        if self.attention_targets:
            components.add("attention")
        if self.embed_tokens:
            components.add("embed_tokens")
        if self.lm_head:
            components.add("lm_head")
        if self.norms:
            components.add("norms")
        return frozenset(components)
```

Required parser behavior:

- split comma-separated strings and sequence inputs;
- normalize case and hyphens to underscores;
- expand aliases;
- expand `attention` to `q_proj/k_proj/v_proj/o_proj`;
- expand `all` to every component implemented at that stage;
- reject `none` mixed with any other token;
- reject unknown tokens with a message listing valid tokens;
- reject known but not-yet-implemented tokens until their stage lands.

Keep a stage-gated constant:

```python
SUPPORTED_LF_OFFLOAD_COMPONENTS = frozenset({
    "routed_experts",
    # expanded by later stages
})
```

Each stage expands this set only after its wrappers, tests, and residency audit
are in place.

## Reporting and Residency Audit

Add an explicit audit utility, because the current setup report only has coarse
`cpu_resident_base_bytes` and `gpu_resident_base_bytes`.

Create in `asym_gemm/training/offload.py`:

```python
@dataclass(frozen=True)
class OffloadResidencyRow:
    name: str
    component: str
    kind: str                 # parameter, buffer, host_weight, host_weight_alias
    device: str               # cpu, cuda, meta
    bytes: int
    requires_grad: bool
    selected_for_cpu: bool
    storage_key: tuple[str, int, int] | None

def collect_lf_offload_residency(model: nn.Module, selection: LFOffloadSelection) -> list[OffloadResidencyRow]:
    return _collect_parameter_buffer_and_host_weight_rows(model, selection)
```

Requirements:

- classify rows using the same component vocabulary as
  `asym_gemm/profiling/lf_trace.py`;
- dedupe by storage key when reporting totals, so accidental aliases are not
  counted twice;
- fail strict validation if `selected_for_cpu=True` and `device == "cuda"` for
  a frozen base weight;
- fail strict validation if any non-LoRA parameter has `requires_grad=True`;
- expose per-component CPU bytes and GPU bytes in `LFAsymReport.to_log_string()`;
- update `_model_memory_summary()` in `asym_gemm/profiling/lf_trace.py` so
  HostWeight rows are attributed to their actual component, not hardcoded to
  `routed_experts`.

The report must distinguish:

- `cpu_resident_base_bytes_by_component`;
- `gpu_resident_base_bytes_by_component`;
- `selected_gpu_resident_base_bytes_by_component`, which must be zero in strict
  `asym_backend=asym` runs after each stage.

## Component Classification Rules

Add one shared classifier and use it for selector splitting, wrapping,
residency audit, and memory reporting:

```python
def classify_lf_component(name: str, module: nn.Module | None = None) -> str:
    return _classify_lf_component_by_priority(name, module)
```

Rules, in priority order:

1. Names containing `.mlp.shared_expert`, `.shared_expert.`, or
   `.shared_experts.` classify as `shared_experts`.
2. Names ending in `.shared_expert_gate` classify as `shared_experts`.
3. Names containing `.mlp.experts.`, ending in `.experts`, or containing
   `.feed_forward.experts.` classify as `routed_experts`.
4. Names ending in `.mlp.gate`, containing `.mlp.gate.`, ending in `.router`,
   containing `.router.`, or containing `.feed_forward.router.` classify as
   `router`.
5. Names containing `.self_attn.`, `.self_attention.`, or `.attention.` and
   whose leaf is one of `q_proj`, `k_proj`, `v_proj`, `o_proj`, `qkv_proj`,
   `gate_proj` only for attention modules, or `out_proj` classify as
   `attention`. For Stage 2, only visible `q_proj/k_proj/v_proj/o_proj` are
   implemented; `qkv_proj` and `out_proj` must be reported as unsupported
   attention layouts.
6. Names ending in `.embed_tokens`, `.embed_in`, or `.wte` classify as
   `embed_tokens`.
7. Names ending in `.lm_head`, `.output`, or `.output_layer` classify as
   `lm_head` only when that module is the model output embedding returned by
   `get_output_embeddings()` or an established model-family output head.
8. Names whose leaf contains `norm`, `layernorm`, `rms_norm`, `q_norm`, or
   `k_norm` classify as `norms`.
9. Names under `.mlp.` with leaves `gate_proj`, `up_proj`, or `down_proj` and
   not already classified as shared or routed classify as `mlp_dense`. This is
   report-only in this MoE feature.
10. Everything else classifies as `other`.

The PEFT split must not use only leaf names. It must compute the component for
each matching module name. Example: selecting `shared_experts` reserves
`model.layers.0.mlp.shared_expert.gate_proj` for Asym wrapping but does not
reserve an unrelated `model.layers.0.mlp.gate_proj`.

Unsupported known layouts are hard failures when their component is selected:

- selected `attention` with fused `qkv_proj`;
- selected `router` with an unrecognized router class;
- selected `norms` with an unrecognized norm class;
- selected `embed_tokens` with mutable `max_norm`.

## Wrapper Strategy

### Routed Experts

Keep the current whole-MoE or packed-expert replacement. Routed experts require
block-level ownership because routing, packing, grouped GEMM, scattering,
combining, router no-grad, and expert recompute are cross-module behavior.

Current classes:

- `asym_gemm/training/qwen3_moe.py::AsymQwen3Experts`
- `asym_gemm/training/qwen3_moe.py::AsymQwen3MoeBlock`
- `asym_gemm/training/qwen35_moe.py::AsymQwen35MoeBlock`
- `asym_gemm/training/llama4_moe.py::AsymLlama4Moe`
- `asym_gemm/training/packed_moe.py::PackedExpertSource`

Do not replace whole transformer blocks for the remaining buckets.

### Linear Leaves

Use HostWeight-backed linear wrappers for frozen base linears:

- `AsymFrozenLinear` for frozen-only linears such as `lm_head` or
  `shared_expert_gate` when it is not a LoRA target.
- `AsymLoRALinear` for linears that are LoRA targets and whose frozen base is
  selected for CPU offload.
- `TorchLoRALinear` or normal PEFT for dense LoRA targets not selected for CPU
  offload.

Important: when LF loads the model on CPU first, the selected wrapper must adopt
the source CPU weight instead of cloning it. Do not call a helper such as
`from_linear()` if that helper clones the source tensor. Add an explicit
HostWeight-adoption constructor path and use `clone=False` for selected
offload bases.

### Router

Routers are model-family modules, not always plain leaf linears. Do not blindly
replace every module named `gate` with `AsymLoRALinear`.

Add model-family router wrappers:

- `AsymQwen3Router` for Qwen3 and Qwen3.5 router modules that return
  `(router_logits, top_k_weights, top_k_index)`;
- `AsymLlama4Router` for Llama 4 router modules that return
  `(router_scores, router_logits)`;
- additional router families require explicit wrappers and parity tests before
  their selectors are enabled.

These wrappers own an `AsymFrozenLinear` base for the router projection and
reproduce the source router's scoring/top-k semantics. They must copy the
source router's required attributes (`hidden_dim`, `num_experts`, `top_k`,
`norm_topk_prob`, score function fields, bias if present, and eps/scale fields
if present).

Integrate them in:

- `AsymQwen3MoeBlock.__init__`;
- `AsymQwen35MoeBlock.__init__`;
- `AsymLlama4Moe.__init__`;
- the `router_mode="hf"` path in `apply_lf_asym_lora()` by replacing visible
  router modules before `model.to(cuda)`.

### Shared Experts

Shared experts are regular MLP branches inside MoE blocks. Use component-aware
leaf wrapping instead of a new whole-block abstraction:

- `shared_expert.gate_proj`;
- `shared_expert.up_proj`;
- `shared_expert.down_proj`;
- `shared_expert_gate`;
- Llama 4 `shared_expert` leaves exposed by its model class.

If a shared expert leaf is a LoRA target, use `AsymLoRALinear`. If it is not a
LoRA target but the base is selected for CPU offload, use `AsymFrozenLinear`.

Selection must be based on component classification, not just leaf names, so
`shared_experts` does not accidentally reserve unrelated `gate_proj` modules in
mixed architectures.

### Attention

Attention offload is projection-only:

- `q_proj`;
- `k_proj`;
- `v_proj`;
- `o_proj`.

SDPA, FlashAttention, RoPE, attention masks, and KV cache stay unchanged. The
projection outputs are normal CUDA tensors regardless of whether the projection
base weight is CPU-resident.

Use `AsymLoRALinear` for attention projections that are both LoRA targets and
CPU-offload targets. Use `AsymFrozenLinear` only if a projection is not a LoRA
target but the selector still requests attention base offload.

### Embeddings

Embeddings are not GEMMs. Add a dedicated wrapper:

```python
class AsymFrozenEmbedding(nn.Module):
    host_weight: HostWeight
    padding_idx: int | None

    @property
    def weight(self) -> torch.Tensor:
        return self.host_weight.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return _cpu_embedding_gather_then_copy_to_input_device(self.host_weight, input_ids, self.padding_idx)
```

Forward contract:

1. keep the embedding table in CPU HostWeight storage;
2. move `input_ids` to CPU for the gather;
3. run `F.embedding` on CPU;
4. copy the gathered activation rows to the original input device;
5. preserve output dtype;
6. support the standard gradient-checkpointing forward hook that calls
   `output.requires_grad_(True)`.

Reject source embeddings with `max_norm` set, because `F.embedding` with
`max_norm` mutates the weight and the frozen HostWeight contract forbids that.
Support `padding_idx`, `scale_grad_by_freq=False`, and `sparse=False`. Since the
embedding is frozen, weight gradients are never produced.

### LM Head

Use `AsymFrozenLinear` for `lm_head` unless the run explicitly makes `lm_head`
a LoRA target, in which case use `AsymLoRALinear`.

The target MoE models are not expected to use tied input/output embeddings.
Do not add a tied-weight registry as a prerequisite for `lm_head` offload.
Instead, add a strict check:

```python
if input_embedding_weight shares storage with lm_head_weight:
    raise ValueError("tied embed/lm_head weights are not supported by this offload stage")
```

If a real target model later requires tied weights, implement that as a separate
stage with explicit tests. Do not add hidden duplicate CPU copies to support a
case that is not needed by the current target models.

### Norms

Norms save little HBM but are required for literal "all frozen model weights on
CPU". Add dedicated frozen norm wrappers:

- `AsymFrozenRMSNorm`;
- `AsymFrozenLayerNorm`.

These wrappers keep norm weights in CPU HostWeight storage and keep bias
tensors, when the source norm has a bias, as frozen CPU tensors. In strict mode,
both must adopt CPU-first source storage without creating a second persistent CPU
copy. Forward copies the tiny scale/bias tensors to the activation device and
runs the same math as the source module. They must preserve source `eps` /
`variance_epsilon`, dtype casting behavior, and bias presence.

Only wrap known-compatible norm classes. In strict mode, if `norms` is selected
and an unknown norm class is present, fail with the module name and class name
instead of silently leaving it on GPU.

## LLaMA-Factory Flow Changes

Modify `third_party/LlamaFactory/src/llamafactory/model/adapter.py` in the
`model_args.use_asym_gemm` branch.

Current flow:

1. compute dense LoRA targets;
2. PEFT-wrap all dense targets;
3. call `adapt_lf_asym_peft_lora()` with `wrap_dense=False`;
4. AsymGEMM wraps packed experts only.

Required flow:

1. parse `model_args.asym_offload_modules`;
2. compute dense LoRA targets as today;
3. split dense targets into PEFT-owned and Asym-owned based on component-aware
   module matching;
4. PEFT-wrap only PEFT-owned dense targets;
5. call `adapt_lf_asym_peft_lora()` with the Asym-owned dense targets and the
   full offload selection;
6. `apply_lf_asym_lora()` wraps packed experts, routers, shared experts,
   attention, embeddings, `lm_head`, and norms according to the stage-supported
   selector.

Do not split solely by leaf name. The split helper must inspect actual module
names and components so `shared_experts` and `attention` reserve only their own
modules.

Modify `third_party/LlamaFactory/src/llamafactory/model/loader.py`:

- `_use_asym_cpu_first_load()` must parse the selector and return true whenever
  `use_asym_gemm`, `asym_backend == "asym"`, and the parsed selection has any
  implemented CPU-offload component.
- The current exact equality check against `"routed_experts"` must be removed.

Modify hparams:

- `model_args.py`: change `asym_offload_modules` from a `Literal` to `str`.
- `parser.py`: validate with `parse_lf_offload_modules()`.
- keep existing BF16, SFT, LoRA, single-process, and distributed restrictions.

## Required Integration API Shape

Update `apply_lf_asym_lora()` to accept a string selector and parse it exactly
once:

```python
def apply_lf_asym_lora(
    model: nn.Module,
    *,
    raw_lora_target: Sequence[str] | str,
    dense_target_modules: Sequence[str] | str,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    backend: Literal["asym", "torch"],
    precision: Literal["bf16"],
    offload_modules: Sequence[str] | str,
    expert_recompute_policy: str = "none",
    router_mode: Literal["hf", "whole"] = "whole",
    wrap_dense: bool = True,
    preexisting_dense_lora_wrapped: int = 0,
    strict: bool = True,
) -> tuple[nn.Module, LFAsymReport]:
    selection = parse_lf_offload_modules(offload_modules)
    plan = build_lf_asym_target_plan(model, raw_lora_target, dense_target_modules, selection)
    return apply_lf_asym_lora_from_plan(
        model,
        plan,
        selection,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        backend=backend,
        precision=precision,
        expert_recompute_policy=expert_recompute_policy,
        router_mode=router_mode,
        strict=strict,
    )
```

Update `adapt_lf_asym_peft_lora()` so it can reserve selected dense modules for
Asym wrapping after PEFT wraps everything else:

```python
def adapt_lf_asym_peft_lora(
    model: nn.Module,
    *,
    raw_lora_target: Sequence[str] | str,
    dense_target_modules: Sequence[str] | str,
    asym_owned_dense_target_modules: Sequence[str] | str,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    backend: Literal["asym", "torch"],
    precision: Literal["bf16"],
    offload_modules: Sequence[str] | str,
    expert_recompute_policy: str = "none",
    router_mode: Literal["hf", "whole"] = "whole",
    strict: bool = True,
) -> tuple[nn.Module, LFAsymReport]:
    return apply_lf_asym_lora(
        model,
        raw_lora_target=raw_lora_target,
        dense_target_modules=asym_owned_dense_target_modules,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        backend=backend,
        precision=precision,
        offload_modules=offload_modules,
        expert_recompute_policy=expert_recompute_policy,
        router_mode=router_mode,
        wrap_dense=bool(asym_owned_dense_target_modules),
        preexisting_dense_lora_wrapped=count_lora_wrapped_modules(model),
        strict=strict,
    )
```

Rules:

- `dense_target_modules` is the full LF dense target list after router filtering.
- `asym_owned_dense_target_modules` is the component-aware subset not already
  PEFT-wrapped.
- `wrap_dense=True` inside `adapt_lf_asym_peft_lora()` only when
  `asym_owned_dense_target_modules` is non-empty.
- non-selected dense targets remain normal PEFT modules.
- selected dense targets become `AsymLoRALinear` or `AsymFrozenLinear` according
  to whether they are LoRA targets.

## Implementation Algorithms

### Algorithm 1: Parse `ASYM_OFFLOAD_MODULES`

Function: `parse_lf_offload_modules(selector)`.

Inputs:

- `selector`: string, sequence of strings, or `None`.
- `SUPPORTED_LF_OFFLOAD_COMPONENTS`: stage-gated set of implemented components.

Steps:

1. If `selector is None`, replace it with `"routed_experts"`.
2. If `selector` is a string, split on commas. If it is a sequence, flatten each
   item by comma as well.
3. Strip whitespace from every token. Drop empty tokens.
4. Normalize each token with `token.lower().replace("-", "_")`.
5. If the token list is empty, treat it as `["none"]`.
6. If `none` appears with any other token, raise `ValueError`.
7. For each token, expand aliases:
   - `routed`, `experts` -> `routed_experts`;
   - `gate`, `expert_router` -> `router`;
   - `shared`, `shared_expert` -> `shared_experts`;
   - `attn` -> `attention`;
   - `embedding`, `embeddings`, `token_embeddings` -> `embed_tokens`;
   - `output_embedding`, `output_embeddings` -> `lm_head`;
   - `norm`, `layernorm`, `rmsnorm` -> `norms`;
   - `whole_model`, `model` -> `all`.
8. Expand component groups:
   - `attention` adds `q_proj`, `k_proj`, `v_proj`, and `o_proj` to
     `attention_targets`;
   - individual attention projection names add only that projection;
   - `all` expands to every component in `SUPPORTED_LF_OFFLOAD_COMPONENTS`.
9. If any expanded component is not in `SUPPORTED_LF_OFFLOAD_COMPONENTS`, raise
   `ValueError` that says the token is known but not implemented in the current
   stage.
10. If a token is neither known nor an implemented component, raise `ValueError`
    with the full valid token list.
11. Return `LFOffloadSelection` with booleans and `attention_targets` populated.

Acceptance tests:

- duplicate tokens do not duplicate bytes or wrappers;
- `none,routed_experts` fails;
- `attention` fails before Stage 2 and succeeds in Stage 2;
- `all` expands only to implemented components at each stage.

### Algorithm 2: Classify Components

Function: `classify_lf_component(name, module)`.

Steps:

1. Convert `name` to lowercase.
2. Apply the priority rules in "Component Classification Rules" exactly in
   order.
3. Use module identity checks for embeddings and output heads when available:
   - compare `module is model.get_input_embeddings()` before wrapping;
   - compare `module is model.get_output_embeddings()` before wrapping.
4. Return one of:

```text
routed_experts
router
shared_experts
attention
embed_tokens
lm_head
norms
mlp_dense
other
```

The classifier must be deterministic. A module name must map to exactly one
component. Shared/routed/router matches take priority over generic attention or
MLP leaf names.

### Algorithm 3: Build the LF Asym Target Plan

Function: `build_lf_asym_target_plan(model, raw_lora_target, dense_target_modules, selection)`.

Output dataclass:

```python
@dataclass(frozen=True)
class LFAsymTargetPlan:
    wrap_experts: bool
    expert_module_names: tuple[str, ...]
    asym_dense_lora_names: tuple[str, ...]
    asym_dense_frozen_names: tuple[str, ...]
    peft_dense_target_suffixes: tuple[str, ...]
    router_names: tuple[str, ...]
    shared_expert_names: tuple[str, ...]
    embedding_names: tuple[str, ...]
    lm_head_names: tuple[str, ...]
    norm_names: tuple[str, ...]
    unsupported_selected_names: tuple[tuple[str, str], ...]
```

Steps:

1. Determine `wrap_experts` from `raw_lora_target` exactly as today:
   `all`, `experts`, `gate`, `up`, `down`, `gate_up_proj`, or `down_proj`
   request routed expert wrapping.
2. Build a set of dense LoRA target suffixes from `dense_target_modules`.
3. Iterate `model.named_modules()` before any replacement. For each module:
   - compute `component = classify_lf_component(name, module)`;
   - compute `leaf = name.rsplit(".", 1)[-1]`;
   - compute `is_lora_target = leaf in dense_target_suffixes or name in dense_target_suffixes`;
   - compute `selected = component_is_selected(component, leaf, selection)`.
4. If `component == "routed_experts"` and `wrap_experts`, add the supported
   packed expert or whole-MoE module name to `expert_module_names`.
5. If `component` is selected and the module is a supported linear leaf:
   - if `is_lora_target`, add name to `asym_dense_lora_names`;
   - else add name to `asym_dense_frozen_names`.
6. If `component == "router"` and selected, add name to `router_names` only if
   it is the supported router owner module for the model family. Do not add
   arbitrary nested router helper modules.
7. If `component == "shared_experts"` and selected, add shared expert leaf
   linear names to `shared_expert_names`.
8. If `component == "embed_tokens"` and selected, add name to `embedding_names`.
9. If `component == "lm_head"` and selected, add name to `lm_head_names`.
10. If `component == "norms"` and selected, add name to `norm_names`.
11. If selected but not supported in the current stage, append
    `(name, reason)` to `unsupported_selected_names`.
12. Build `peft_dense_target_suffixes` by subtracting every Asym-owned dense
    LoRA target from the original dense target suffix set.
13. In strict mode, fail if `unsupported_selected_names` is non-empty.

This plan is the single source of truth for both LF's PEFT split and
AsymGEMM's post-PEFT wrapping.

### Algorithm 4: Split PEFT Targets in LLaMA-Factory

Function to add near `_filter_asym_dense_peft_targets()`:
`split_asym_peft_dense_targets(model, target_modules, selection)`.

Steps:

1. Call `_filter_asym_dense_peft_targets(model, target_modules)` to remove
   router-only targets as today.
2. Call
   `build_lf_asym_target_plan(model, raw_lora_target, dense_target_modules, selection)`.
3. Set `peft_dense_target_modules = plan.peft_dense_target_suffixes`.
4. Set `asym_owned_dense_target_modules` to the suffixes or full names in
   `plan.asym_dense_lora_names`.
5. Call PEFT only when `peft_dense_target_modules` is non-empty.
6. Pass `asym_owned_dense_target_modules` into `adapt_lf_asym_peft_lora()`.

Invariant: a selected Asym-owned module is never PEFT-wrapped first. PEFT-wrapped
modules are no longer plain `nn.Linear`, so the Asym wrapper must own selected
linears before PEFT transforms them.

### Algorithm 5: Apply Wrappers in a Stable Order

Function: `apply_lf_asym_lora_from_plan(model, plan, selection, lora_rank, lora_alpha, lora_dropout, backend, precision, expert_recompute_policy, router_mode, strict)`.

Replacement order:

1. Whole MoE / packed routed expert wrappers.
2. Router wrappers inside whole-MoE wrappers.
3. Shared expert leaf wrappers inside whole-MoE wrappers.
4. Attention projection leaf wrappers.
5. `lm_head` wrapper.
6. Embedding wrapper.
7. Norm wrappers.
8. Freeze non-LoRA params.
9. Run strict residency audit.
10. Build `LFAsymReport`.

Why this order:

- routed expert replacement changes module ownership, so it must run before
  scanning nested MoE leaves;
- router/shared wrappers are installed inside the newly installed MoE wrappers;
- `lm_head` before embeddings makes the strict tied-storage rejection
  deterministic if an unexpected tied model appears;
- freezing happens after all wrapper installation so every new wrapper parameter
  is captured.

### Algorithm 6: Wrap a Linear Leaf

Function: `_wrap_lf_linear_leaf(name, module, *, is_lora_target, backend, precision, stats, lora_rank, lora_alpha, lora_dropout, lora_dtype)`.

Steps:

1. Require `isinstance(module, nn.Linear)`.
2. Resolve source dtype from `module.weight.dtype`.
3. If `backend == "asym"` and the component is selected for CPU offload, require
   `module.weight.device.type == "cpu"` in strict mode. This confirms the
   CPU-first load path is active.
4. Create a HostWeight from `module.weight.detach()` with `clone=False`. For
   GEMM buckets on CUDA systems, request pinned memory so the direct AsymGEMM
   path can read the CPU owner. The HostWeight must share source storage when the
   source was already pinned or when pinning is not needed; otherwise it becomes
   the single pinned replacement owner after the source module is replaced.
5. Resolve LoRA device:
   - if CUDA is available and the run will call `model.to(cuda)`, create LoRA
     params on the module's current device and rely on `model.to(cuda)`;
   - if wrapping happens after device move in a unit test, create LoRA params on
     the module's current device.
6. If `is_lora_target`:
   - create `AsymLoRALinear.from_host_weight(host_weight, bias, rank=lora_rank,
     alpha=lora_alpha, backend=backend, stats=stats, lora_dtype=lora_dtype,
     precision=precision, init_lora_weights="peft", lora_dropout=lora_dropout)`;
   - verify `wrapped.base_layer.host_weight.weight.device.type == "cpu"`;
   - verify `wrapped.lora_A` and `wrapped.lora_B` are parameters.
7. If not `is_lora_target`:
   - create `AsymFrozenLinear.from_host_weight(host_weight, bias, backend=backend,
     precision=precision, stats=stats)`;
   - verify it has no parameters.
8. Replace the child module with `_replace_child(parent, child, wrapped)`.
9. Delete local references to the old module after replacement.
10. Record CPU bytes from the wrapper's HostWeight.
11. The residency audit must prove the original CPU parameter is no longer a
    model parameter and only the HostWeight row remains for that selected base.

### Algorithm 7: Adopt HostWeight Storage

Function: `adopt_host_weight(name, tensor, component, pin_memory_policy)`.

Steps:

1. Require `tensor.device.type == "cpu"` for strict production offload.
2. Compute the source storage key before wrapping:

```python
storage = tensor.untyped_storage()
key = (str(tensor.device), storage.data_ptr(), storage.nbytes(), tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()))
```

3. Create `HostWeight(tensor.detach(), clone=False, require_2d=expected_rank == 2,
   pin_memory=needs_direct_asymgemm, name=name)`.
4. If `pin_memory_policy == "none"`, verify the HostWeight storage key is the
   same as the source storage key.
5. If `pin_memory_policy == "auto"` or `"require"`, verify that CUDA systems
   produce a pinned HostWeight. This may change the storage key when the source
   was pageable CPU memory, but the replacement step below must remove the
   original parameter owner so only one persistent CPU base owner remains.
6. Store component attribution outside `HostWeight`, in the residency/reporting
   layer.
7. Replace the source module so the original parameter owner is removed from the
   model.
8. Do not batch-clone all selected weights into pinned HostWeights while
   retaining the original CPU model.

For layout-normalizing model-family adapters such as the current Llama 4 routed
expert adapter, this algorithm changes from "same storage key" to "single
normalized owner":

1. Require CPU-first loading before layout normalization.
2. Build the normalized CPU tensor required by the existing grouped wrapper.
3. Replace the original module immediately.
4. Audit that the original source parameter names are absent from
   `model.named_parameters()`.
5. Audit that only the normalized HostWeight rows remain for the selected
   component.
6. Do not keep both the original source CPU parameters and normalized
   HostWeights after wrapper installation.

Tied embedding/`lm_head` policy:

1. Before wrapping `lm_head` or embeddings, compare the input embedding and
   output head storage keys.
2. If they share storage in a target run, fail strict mode with a clear error.
3. Tied-weight support is deferred until there is a real target model requiring
   it. The deferred implementation must still obey the no-duplicate-CPU-copy
   rule.

### Algorithm 8: Embedding Forward

Class: `AsymFrozenEmbedding`.

Forward steps:

1. Save `target_device = input_ids.device`.
2. Copy `input_ids_cpu = input_ids.to("cpu", non_blocking=False)` because CPU
   embedding lookup requires CPU indices.
3. Run `out_cpu = F.embedding(input_ids_cpu, host_weight.weight, padding_idx,
   max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)`.
4. Copy `out = out_cpu.to(device=target_device, non_blocking=True)` when the
   target device is CUDA; otherwise return `out_cpu`.
5. Cast only if the source embedding module had a forced output dtype behavior.
   The default is to preserve `host_weight.weight.dtype`.
6. Return `out`.

Backward behavior:

- no embedding weight gradient is produced;
- downstream gradients flow into the returned activation tensor;
- LF gradient checkpointing can call `out.requires_grad_(True)`.

### Algorithm 9: Norm Forward

Classes: `AsymFrozenRMSNorm`, `AsymFrozenLayerNorm`.

RMSNorm steps:

1. Copy CPU weight to `x.device` as `weight_dev`.
2. Compute variance in the same dtype behavior as the source norm, normally
   `x.float().pow(2).mean(-1, keepdim=True)`.
3. Compute `normed = x.float() * torch.rsqrt(variance + eps)`.
4. Multiply by `weight_dev.float()`.
5. Cast to the source output dtype behavior, normally `x.dtype`.

LayerNorm steps:

1. Copy CPU weight and bias tensor, when the source norm has bias, to `x.device`.
2. Call `F.layer_norm(x.float(), normalized_shape, weight_dev.float(),
   bias_dev.float() if present else None, eps)`.
3. Cast to the source output dtype behavior.

Strict support:

- support PyTorch `nn.LayerNorm`;
- support HF-style RMSNorm classes whose forward is standard RMSNorm with
  `weight` and `variance_epsilon` or `eps`;
- reject fused custom norm classes until a parity wrapper is added.

### Algorithm 10: Residency Audit

Function: `validate_lf_offload_residency(model, selection, *, strict)`.

Steps:

1. Collect rows for every `named_parameter()`:
   - classify component;
   - record device, bytes, `requires_grad`, storage key;
   - mark `selected_for_cpu` using the selection.
2. Collect rows for every `named_buffer()` with the same fields.
3. Traverse `model.modules()` and collect every attribute named `host_weight`
   or every wrapper type known to own a HostWeight:
   - `AsymFrozenLinear`;
   - `AsymGroupedFrozenLinear`;
   - `AsymLoRALinear.base_layer`;
   - `AsymFrozenEmbedding`;
   - `AsymFrozenRMSNorm`;
   - `AsymFrozenLayerNorm`.
4. Deduplicate HostWeight rows by storage key.
5. For strict validation, fail if any selected component has a frozen parameter
   or buffer on CUDA.
6. Fail if any non-LoRA parameter has `requires_grad=True`.
7. Fail if an expected selected component has no CPU HostWeight rows and the
   model actually contains that component.
8. Return rows and aggregate byte dictionaries.

### Algorithm 11: Stage Gate

Before enabling a new token in `SUPPORTED_LF_OFFLOAD_COMPONENTS`:

1. Parser accepts the new token and rejects it in all earlier stages.
2. Wrapper installation test proves the selected component is replaced.
3. Residency audit test proves the selected frozen base is not in CUDA HBM.
4. Numeric parity test passes for the component.
5. End-to-end fake LF test reaches backward.
6. Report test proves bytes are attributed to the component.
7. Existing `routed_experts` tests pass unchanged.

### Algorithm 12: Install Model-Family MoE Wrappers

Function: `_wrap_lf_moe_block(name, module, selection, lora_rank, lora_alpha, lora_dropout, backend, precision, recompute_config, router_mode, stats, strict)`.

Inputs:

- `module`: a supported Qwen3, Qwen3.5, or Llama 4 MoE owner module.
- `selection`: parsed `LFOffloadSelection`.
- `router_mode`: `whole` or `hf`, matching the existing LF integration.

Steps:

1. Resolve booleans:
   - `offload_experts = backend == "asym" and selection.routed_experts`;
   - `offload_router = backend == "asym" and selection.router`;
   - `offload_shared_experts = backend == "asym" and selection.shared_experts`.
2. If `is_qwen35_moe_block(module)`, call `wrap_qwen35_moe_block()` with the
   existing expert arguments plus `offload_router` and `offload_shared_experts`.
3. Else if `is_qwen3_moe_block(module)`, call `wrap_qwen3_moe_block()` with the
   existing expert arguments plus `offload_router`.
4. Else if `is_llama4_moe(module)`, call `wrap_llama4_moe()` with the existing
   expert arguments plus `offload_router` and `offload_shared_experts`.
5. Else return `None`; unsupported MoE owners are handled by the target plan's
   `unsupported_selected_names` path.
6. Set `wrapped.profile_prefix = _layer_profile_prefix_from_module_name(name, "mlp")`.
7. If `wrapped.experts` exists, set `wrapped.experts.profile_prefix` to
   `f"{wrapped.profile_prefix}.experts"`.
8. Replace the original child module with `_replace_child(parent, child, wrapped)`.
9. Assert selected subcomponents have HostWeight-backed wrappers:
   - `routed_experts`: packed expert HostWeights exist;
   - `router`: router projection HostWeight exists;
   - `shared_experts`: shared expert leaf HostWeights exist.

Wrapper constructor changes:

- `AsymQwen3MoeBlock.__init__` accepts `offload_router: bool = False`.
- `AsymQwen35MoeBlock.__init__` accepts `offload_router: bool = False` and
  `offload_shared_experts: bool = False`.
- `AsymLlama4Moe.__init__` accepts `offload_router: bool = False` and
  `offload_shared_experts: bool = False`.

### Algorithm 13: Install Router and Shared Leaves

Function: `_wrap_lf_router_and_shared_leaves(model, plan, selection, backend, precision, stats, lora_rank, lora_alpha, lora_dropout, lora_dtype, router_mode)`.

Steps:

1. Iterate `plan.router_names` before generic dense wrapping.
2. For Qwen3 and Qwen3.5 whole-MoE wrappers, router wrapping is performed inside
   `_wrap_lf_moe_block()` by constructing `AsymQwen3Router`.
3. For Llama 4 whole-MoE wrappers, router wrapping is performed inside
   `_wrap_lf_moe_block()` by constructing `AsymLlama4Router`.
4. For `router_mode == "hf"`, replace visible supported router modules directly
   before LF's `model.to(cuda)` call:
   - Qwen3/Qwen3.5 routers become `AsymQwen3Router`;
   - Llama 4 routers become `AsymLlama4Router`.
5. For every `plan.shared_expert_names` entry that is still an `nn.Linear`, call
   `_wrap_lf_linear_leaf()` with `is_lora_target = name in plan.asym_dense_lora_names`.
6. After replacement, validate that every selected router/shared row has no CUDA
   frozen parameter or buffer.

Router wrapper algorithm:

1. Store the source router projection in an adopted HostWeight-backed
   `AsymFrozenLinear` without cloning the source CPU tensor.
2. Copy the source router metadata fields exactly.
3. In forward, run the frozen router projection through the Asym linear wrapper.
4. Apply the same score function, top-k selection, normalization, and bias handling
   as the source router.
5. Return the same tuple shape as the source router family.

### Algorithm 14: CPU-First LF Loading

Function: `_use_asym_cpu_first_load(model_args)`.

Steps:

1. If `model_args.use_asym_gemm` is false, return `False`.
2. If `model_args.asym_backend != "asym"`, return `False`.
3. Parse `model_args.asym_offload_modules` with `parse_lf_offload_modules()`.
4. Return `selection.any_cpu_offload`.
5. Propagate parser errors; do not fall back to GPU placement for invalid selector
   strings.
6. This CPU-first path is mandatory for selected CPU offload. Later wrapper code
   must reject selected CUDA source weights instead of copying them back to CPU.

This replaces the current exact comparison against `"routed_experts"`.

### Algorithm 15: Adapter State and Metadata

Functions:

- `get_asym_lora_state_dict(model, adapter_name="default")`;
- `_infer_adapter_config(model, metadata)`;
- `save_asym_peft_adapter(model, output_dir, adapter_name="default", metadata=None, safe_serialization=True)`;
- `load_asym_peft_adapter(model, adapter_dir, adapter_name="default", strict=True)`.

Save steps:

1. Call `get_lora_state_dict(model, adapter_name=adapter_name)`.
2. Convert every returned LoRA tensor to contiguous CPU storage.
3. Reject any key containing `host_weight`, `base_layer.weight`, `bias_cpu`,
   `embed_tokens.weight`, `lm_head.weight`, `norm.weight`, or `norm.bias`.
4. Build adapter config with `_infer_adapter_config()`.
5. Store the raw selector string under `asym_offload_modules`.
6. Store `asym_adapter_format`, `asym_backend`, `asym_precision`,
   `asym_router_mode`, `asym_expert_recompute_policy`, LoRA rank, LoRA alpha, and
   LoRA dropout.
7. Write `adapter_config.json`.
8. Write `adapter_model.safetensors` or `adapter_model.bin` with LoRA tensors only.

Load steps:

1. Construct the target model and install Asym wrappers from the adapter config
   before loading LoRA tensors.
2. Read adapter tensors on CPU.
3. Call `load_lora_state_dict(model, state, adapter_name=adapter_name, strict=strict)`.
4. Run `validate_lf_offload_residency(model, selection, strict=True)` after load.

### Algorithm 16: Report and Profiling Aggregation

Functions:

- `build_lf_asym_report(plan, rows, stats, skipped, expert_recompute_policy, router_mode)`;
- `asym_gemm/profiling/lf_trace.py::_model_memory_summary(model)`;
- `asym_gemm/profiling/lf_trace.py::_component_from_param_name(name)`;
- `asym_gemm/profiling/lf_trace.py::_component_from_module_name(name)`.

Steps:

1. Consume residency rows from `collect_lf_offload_residency(model, selection)`.
2. Deduplicate rows by `storage_key`; rows with `storage_key is None` are counted
   by name.
3. Aggregate CPU HostWeight bytes by component.
4. Aggregate CUDA parameter and buffer bytes by component.
5. Aggregate selected CUDA frozen bytes by component.
6. Store aggregates in `LFAsymReport`:
   - `cpu_resident_base_bytes_by_component`;
   - `gpu_resident_base_bytes_by_component`;
   - `selected_gpu_resident_base_bytes_by_component`.
7. Keep existing scalar totals as sums over the component maps.
8. Update `to_log_string()` to print the component maps in stable sorted-key order.
9. Update LF trace memory summaries to call the shared component classifier, not
   their own hardcoded routed-expert matching.

## Production Touch Map

Files to modify:

- `asym_gemm/integrations/lf.py`
  - add `LFOffloadSelection`;
  - add `parse_lf_offload_modules()`;
  - add component-aware target splitting helpers;
  - extend `LFAsymReport`;
  - extend `apply_lf_asym_lora()`;
  - import `collect_lf_offload_residency()` from `asym_gemm.training.offload`;
  - update `_infer_adapter_config()` to preserve the raw selector.

- `asym_gemm/integrations/peft_lf.py`
  - pass the raw selector through to `apply_lf_asym_lora()`;
  - allow `adapt_lf_asym_peft_lora()` to wrap selected dense targets after PEFT
    wraps non-selected targets.

- `asym_gemm/training/lora.py`
  - reuse `AsymLoRALinear`;
  - ensure `init_lora_weights="peft"` and the adopted HostWeight constructor
    path are used from LF;
  - add stable CPU/GPU base-byte properties for all wrappers used by the LF
    report.

- `asym_gemm/training/frozen_linear.py`
  - add a constructor path that accepts an adopted existing HostWeight without
    cloning the source tensor;
  - preserve current BF16-only LF behavior.

- `asym_gemm/training/host_weight.py`
  - no semantic change required;
  - keep component attribution outside `HostWeight` in the residency audit.

- New `asym_gemm/training/offload.py`
  - `adopt_host_weight()`;
  - `AsymFrozenEmbedding`;
  - `AsymFrozenRMSNorm`;
  - `AsymFrozenLayerNorm`;
  - residency audit helpers.

- Model-family router wrappers
  - place `AsymQwen3Router` in `asym_gemm/training/qwen3_moe.py`;
  - place Qwen3.5 router reuse in `asym_gemm/training/qwen35_moe.py` by
    importing the Qwen3 router wrapper;
  - place `AsymLlama4Router` in `asym_gemm/training/llama4_moe.py`.

- `asym_gemm/training/qwen3_moe.py`
  - accept per-bucket offload selection or explicit booleans;
  - wrap Qwen3 router when selected;
  - preserve current routed expert path.

- `asym_gemm/training/qwen35_moe.py`
  - accept per-bucket offload selection or explicit booleans;
  - wrap Qwen3.5 router when selected;
  - allow shared expert branch wrapping.

- `asym_gemm/training/llama4_moe.py`
  - accept per-bucket offload selection or explicit booleans;
  - wrap Llama 4 router when selected;
  - allow shared expert branch wrapping.

- `asym_gemm/profiling/lf_trace.py`
  - attribute HostWeight memory by component;
  - include `embed_tokens`, `lm_head`, and `norms` in memory breakdown filters;
  - avoid double counting shared HostWeight storage.

- `third_party/LlamaFactory/src/llamafactory/model/adapter.py`
  - split PEFT-owned and Asym-owned targets;
  - call the expanded adapter API.

- `third_party/LlamaFactory/src/llamafactory/model/loader.py`
  - parse selector in `_use_asym_cpu_first_load()`.

- `third_party/LlamaFactory/src/llamafactory/hparams/model_args.py`
  - relax `asym_offload_modules` type to `str`.

- `third_party/LlamaFactory/src/llamafactory/hparams/parser.py`
  - validate selector through the shared parser.

- `scripts/lf/run_lf_lora_sft.sh`
  - keep the default unchanged;
  - no required behavior change beyond accepting comma-separated values.

## Exact Function and Class Checklist

Add these classes:

| File | Class |
| --- | --- |
| `asym_gemm/integrations/lf.py` | `LFOffloadSelection` |
| `asym_gemm/integrations/lf.py` | `LFAsymTargetPlan` |
| `asym_gemm/training/offload.py` | `OffloadResidencyRow` |
| `asym_gemm/training/offload.py` | `AsymFrozenEmbedding` |
| `asym_gemm/training/offload.py` | `AsymFrozenRMSNorm` |
| `asym_gemm/training/offload.py` | `AsymFrozenLayerNorm` |
| `asym_gemm/training/qwen3_moe.py` | `AsymQwen3Router` |
| `asym_gemm/training/llama4_moe.py` | `AsymLlama4Router` |

Modify these existing classes:

| File | Class | Required modification |
| --- | --- | --- |
| `asym_gemm/integrations/lf.py` | `LFAsymReport` | Add per-component CPU/GPU byte maps, selected GPU byte map, and counts for dense CPU/GPU wrappers. |
| `asym_gemm/training/frozen_linear.py` | `AsymFrozenLinear` | Add constructor path for an adopted existing `HostWeight` with no source clone; expose stable `cpu_resident_base_weight_bytes` and `gpu_resident_base_weight_bytes`. |
| `asym_gemm/training/lora.py` | `AsymLoRALinear` | Add constructor/factory path that accepts an adopted HostWeight; expose stable CPU/GPU byte properties. |
| `asym_gemm/training/lora.py` | `TorchLoRALinear` | Expose stable CPU/GPU byte properties for reports. |
| `asym_gemm/training/qwen3_moe.py` | `AsymQwen3MoeBlock` | Accept offload selection or booleans; wrap router when selected. |
| `asym_gemm/training/qwen35_moe.py` | `AsymQwen35MoeBlock` | Accept offload selection or booleans; wrap router and shared branch when selected. |
| `asym_gemm/training/llama4_moe.py` | `AsymLlama4Moe` | Accept offload selection or booleans; wrap router and shared branch when selected. |

Add these functions:

| File | Function |
| --- | --- |
| `asym_gemm/integrations/lf.py` | `parse_lf_offload_modules(selector)` |
| `asym_gemm/integrations/lf.py` | `classify_lf_component(name, module=None)` |
| `asym_gemm/integrations/lf.py` | `component_is_selected(component, leaf, selection)` |
| `asym_gemm/integrations/lf.py` | `build_lf_asym_target_plan(model, raw_lora_target, dense_target_modules, selection)` |
| `asym_gemm/integrations/lf.py` | `apply_lf_asym_lora_from_plan(model, plan, selection, lora_rank, lora_alpha, lora_dropout, backend, precision, expert_recompute_policy, router_mode, strict)` |
| `asym_gemm/integrations/lf.py` | `_wrap_lf_moe_block(name, module, selection, lora_rank, lora_alpha, lora_dropout, backend, precision, recompute_config, router_mode, stats, strict)` |
| `asym_gemm/integrations/lf.py` | `_wrap_lf_router_and_shared_leaves(model, plan, selection, backend, precision, stats, lora_rank, lora_alpha, lora_dropout, lora_dtype, router_mode)` |
| `asym_gemm/integrations/lf.py` | `_wrap_lf_linear_leaf(name, module, is_lora_target, backend, precision, stats, lora_rank, lora_alpha, lora_dropout, lora_dtype)` |
| `asym_gemm/integrations/lf.py` | `_wrap_lf_attention(model, plan, selection, backend, precision, stats, lora_rank, lora_alpha, lora_dropout, lora_dtype)` |
| `asym_gemm/integrations/lf.py` | `_wrap_lf_lm_head(model, plan, selection, stats)` |
| `asym_gemm/integrations/lf.py` | `_wrap_lf_embeddings(model, plan, selection)` |
| `asym_gemm/integrations/lf.py` | `_wrap_lf_norms(model, plan, selection)` |
| `asym_gemm/integrations/lf.py` | `build_lf_asym_report(plan, rows, stats, skipped, expert_recompute_policy, router_mode)` |
| `asym_gemm/integrations/lf.py` | `_replace_child(parent, child_name, module)` already exists; reuse it. |
| `asym_gemm/training/offload.py` | `adopt_host_weight(name, tensor, component, pin_memory_policy)` |
| `asym_gemm/training/offload.py` | `storage_key(tensor)` |
| `asym_gemm/training/offload.py` | `collect_lf_offload_residency(model, selection)` |
| `asym_gemm/training/offload.py` | `validate_lf_offload_residency(model, selection, strict=True)` |
| `third_party/LlamaFactory/src/llamafactory/model/adapter.py` | `split_asym_peft_dense_targets(model, target_modules, selection)` |

Modify these existing functions:

| File | Function | Required modification |
| --- | --- | --- |
| `asym_gemm/integrations/lf.py` | `_validate_trainable_params(model)` | Keep LoRA-only enforcement; include new wrapper names in error diagnostics. |
| `asym_gemm/integrations/lf.py` | `apply_lf_asym_lora(model, raw_lora_target, dense_target_modules, lora_rank, lora_alpha, lora_dropout, backend, precision, offload_modules, expert_recompute_policy, router_mode, wrap_dense, preexisting_dense_lora_wrapped, strict)` | Parse selector, build target plan, wrap selected buckets, run residency audit. |
| `asym_gemm/integrations/lf.py` | `_infer_adapter_config(model, metadata)` | Persist raw `asym_offload_modules` selector. |
| `asym_gemm/integrations/lf.py` | `get_asym_lora_state_dict(model, adapter_name="default")` | Confirm only LoRA state is saved after new wrappers; no base HostWeight keys. |
| `asym_gemm/integrations/lf.py` | `save_asym_peft_adapter(model, output_dir, adapter_name="default", metadata=None, safe_serialization=True)` | Persist selector metadata and save only LoRA tensors. |
| `asym_gemm/integrations/lf.py` | `load_asym_peft_adapter(model, adapter_dir, adapter_name="default", strict=True)` | Load only LoRA tensors after wrappers are installed and then run residency validation. |
| `asym_gemm/integrations/peft_lf.py` | `adapt_lf_asym_peft_lora(model, raw_lora_target, dense_target_modules, asym_owned_dense_target_modules, lora_rank, lora_alpha, lora_dropout, backend, precision, offload_modules, expert_recompute_policy, router_mode, strict)` | Accept `asym_owned_dense_target_modules` and pass selector through. |
| `asym_gemm/profiling/lf_trace.py` | `_component_from_param_name(name)` | Align with `classify_lf_component`. |
| `asym_gemm/profiling/lf_trace.py` | `_component_from_module_name(name)` | Align with `classify_lf_component`. |
| `asym_gemm/profiling/lf_trace.py` | `_model_memory_summary(model)` | Attribute HostWeight rows by component and dedupe storage. |
| `third_party/LlamaFactory/src/llamafactory/model/adapter.py` | `_filter_asym_dense_peft_targets(model, target_modules)` | Keep router filtering; call new split helper after this filter. |
| `third_party/LlamaFactory/src/llamafactory/model/adapter.py` | `_setup_lora_tuning(config, model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32)` AsymGEMM branch | Split PEFT-owned and Asym-owned dense targets before PEFT wrapping. |
| `third_party/LlamaFactory/src/llamafactory/model/loader.py` | `_use_asym_cpu_first_load(model_args)` | Parse selector and return true for any implemented CPU-offload component. |
| `third_party/LlamaFactory/src/llamafactory/hparams/parser.py` | AsymGEMM validation block | Replace hardcoded set validation with `parse_lf_offload_modules`. |
| `third_party/LlamaFactory/src/llamafactory/hparams/model_args.py` | `AsymGEMMArguments.asym_offload_modules` | Change type to `str` and update help text. |

## Staged Implementation Plan

Each stage must merge only after its validation checklist passes. Do not enable
later selector tokens before the corresponding stage is complete.

### Stage 0: Baseline Lock

Goal: freeze current expert-only behavior before broadening the selector.

Allowed selectors:

```text
none
routed_experts
```

Implementation:

- add no new runtime behavior;
- add tests that document current CPU-first load behavior and current
  expert-only wrapping;
- add a baseline residency audit for `routed_experts`.

Validation:

- existing `tests/training/test_lf_qwen3_asym_backend.py`;
- existing `tests/training/test_lf_qwen35_asym_backend.py`;
- existing `tests/training/test_cpu_resident_frozen_base.py`;
- one LF smoke with `ASYM_OFFLOAD_MODULES=routed_experts`;
- audit proves routed expert HostWeights are CPU-resident and all non-LoRA
  trainable params are frozen.

### Stage 1: Selector, Reports, and Strict Residency Audit

Goal: centralize selector parsing and make residency measurable.

Allowed selectors:

```text
none
routed_experts
all              # expands only to routed_experts in this stage
```

Implementation:

- add `LFOffloadSelection`;
- add `parse_lf_offload_modules()`;
- change LF and LLaMA-Factory hparams validation to use the parser;
- update `_use_asym_cpu_first_load()` to use `selection.any_cpu_offload`;
- add `collect_lf_offload_residency()`;
- extend `LFAsymReport` with per-component bytes.

Validation:

- parser tests for aliases, duplicate tokens, `none` exclusivity, and invalid
  tokens;
- `ASYM_OFFLOAD_MODULES=routed_experts` remains byte-for-byte behaviorally
  equivalent to current setup;
- unsupported final tokens such as `attention` and `router` are rejected until
  their stages land;
- memory attribution no longer hardcodes HostWeight bytes to routed experts.

### Stage 2: Attention Projection Offload

Goal: offload attention projection frozen bases while keeping SDPA unchanged.

New allowed selectors:

```text
attention
attn
q_proj
k_proj
v_proj
o_proj
```

Implementation:

- split PEFT dense targets so selected attention projections are not PEFT
  wrapped first;
- wrap selected attention projections with `AsymLoRALinear` when they are LoRA
  targets;
- wrap selected attention projections with `AsymFrozenLinear` when they are not
  LoRA targets;
- keep SDPA and attention masks untouched.

Validation:

- CPU unit parity: `AsymLoRALinear(backend="torch")` matches `TorchLoRALinear`;
- CUDA parity on SM100 BF16: selected q/k/v/o base forwards and dX use
  AsymGEMM and match a torch reference within existing BF16 tolerances;
- LF fake Qwen3 model with `offload_modules=attention`;
- LF fake Qwen3 model with `offload_modules=routed_experts,attention`;
- residency audit shows selected q/k/v/o bases are HostWeights and no selected
  attention base is a CUDA persistent tensor;
- optimizer contains only LoRA parameters.

### Stage 3: Shared Expert Offload

Goal: offload shared expert branches in MoE models that have them.

New allowed selectors:

```text
shared_experts
shared_expert
shared
```

Implementation:

- add component-aware splitting for `shared_expert.*` and
  `shared_expert_gate`;
- wrap shared expert LoRA-target leaves with `AsymLoRALinear`;
- wrap selected frozen-only shared expert leaves with `AsymFrozenLinear`;
- update `AsymQwen35MoeBlock` and `AsymLlama4Moe` so internal shared branches
  are wrapped after the whole MoE wrapper is installed.

Validation:

- Qwen3.5 fake-model parity for shared branch output;
- Llama 4 fake-model parity for shared branch output;
- `offload_modules=shared_experts` affects only shared expert modules, not
  routed experts or attention;
- `offload_modules=routed_experts,shared_experts` accumulates CPU bytes for both
  buckets with no double counting;
- strict mode fails if `shared_experts` is selected on a model whose shared
  branch class is unsupported.

### Stage 4: Router Offload

Goal: offload MoE router/gate projection bases.

New allowed selectors:

```text
router
gate
expert_router
```

Implementation:

- add Qwen3/Qwen3.5 router wrapper that reproduces the source
  `(router_logits, top_k_weights, top_k_index)` output;
- add Llama 4 router wrapper that reproduces `(router_scores, router_logits)`;
- route wrappers use `AsymFrozenLinear`, not LoRA;
- integrate wrappers in whole-MoE mode and `router_mode="hf"` mode;
- preserve router no-grad behavior.

Validation:

- exact top-k index parity against the source router for fixed logits;
- BF16 tolerance parity for router logits and top-k weights;
- selected router base is CPU HostWeight after `model.to(cuda)`;
- no router parameter is trainable;
- routed expert outputs remain unchanged for fixed input and fixed LoRA state;
- strict mode fails for unknown router classes.

### Stage 5: LM Head Offload

Goal: offload output projection.

New allowed selectors:

```text
lm_head
output_embedding
output_embeddings
```

Implementation:

- wrap `model.get_output_embeddings()` or the `lm_head` module with
  `AsymFrozenLinear`;
- support `AsymLoRALinear` only if LF explicitly made `lm_head` a LoRA target;
- reject tied input/output embedding storage in strict mode for this stage;
- adopt the CPU-loaded `lm_head` source weight into HostWeight without cloning.

Validation:

- logits parity against GPU `lm_head`;
- cross entropy loss parity for a small fake model;
- strict-mode tied-weight rejection test;
- residency audit proves the original `lm_head` parameter owner is gone and only
  one CPU HostWeight row remains for the selected `lm_head` base;
- adapter save excludes `lm_head` base weights.

### Stage 6: Token Embedding Offload

Goal: offload input embedding table.

New allowed selectors:

```text
embed_tokens
embedding
embeddings
token_embeddings
```

Implementation:

- add `AsymFrozenEmbedding`;
- replace `model.get_input_embeddings()` target and any direct module name such
  as `model.embed_tokens`;
- preserve `get_input_embeddings()` behavior;
- reject mutable `max_norm` embeddings;
- reject tied input/output embedding storage in strict mode for this stage;
- adopt the CPU-loaded embedding table into HostWeight without cloning.

Validation:

- embedding output parity for CPU and CUDA input ids;
- gradient checkpointing hook compatibility: output can be marked
  `requires_grad_(True)`;
- selected embedding table is not a CUDA parameter or buffer after device move;
- residency audit proves the original embedding parameter owner is gone and only
  one CPU HostWeight row remains for the selected embedding base;
- end-to-end LF smoke reaches backward with LoRA gradients.

### Stage 7: Norm Offload

Goal: eliminate the remaining persistent norm weights from HBM.

New allowed selectors:

```text
norms
norm
layernorm
rmsnorm
```

Implementation:

- add `AsymFrozenRMSNorm`;
- add `AsymFrozenLayerNorm`;
- wrap known-compatible norm modules only;
- copy tiny CPU scale/bias tensors to the activation device during forward;
- preserve source dtype and eps behavior exactly.

Validation:

- norm parity tests for every supported norm class;
- q/k norm parity if the target model has q/k norms;
- strict mode fails for selected unknown norm classes;
- residency audit shows no selected norm CUDA parameters or buffers;
- memory report shows small but nonzero CPU bytes for norms.

### Stage 8: Whole-Model Selector

Goal: enable the final user-facing whole-model modes.

Final allowed selectors:

```text
all
whole_model
model
```

Final expansion for supported MoE models:

```text
routed_experts,router,shared_experts,attention,embed_tokens,lm_head,norms
```

Implementation:

- expand `SUPPORTED_LF_OFFLOAD_COMPONENTS` to every completed bucket;
- make `all` expand to all completed buckets;
- add an explicit `--strict` failure if a completed bucket is selected and a
  matching model-family module cannot be wrapped.

Validation:

- fake Qwen3 whole-model test;
- fake Qwen3.5 whole-model test;
- fake Llama 4 whole-model test;
- LF smoke:

```bash
BACKEND=asym ASYM_OFFLOAD_MODULES=all MAX_STEPS=2 PROFILE=1 \
  scripts/lf/run_lf_lora_sft.sh
```

- residency audit shows zero selected frozen base bytes in GPU HBM;
- optimizer params are LoRA-only;
- adapter save/load round trip preserves LoRA state;
- source memory breakdown has CPU HostWeight rows for each selected component;
- compare one step of `all` against a GPU-resident `asym_backend=torch`
  reference for finite loss and BF16 tolerance on fake models.

## Validation Matrix

Every stage must pass these categories before the next stage starts:

| Category | Required checks |
| --- | --- |
| Parser | aliases, invalid tokens, duplicate tokens, `none` exclusivity, stage-gated unsupported-token rejection |
| Replacement | expected classes installed, old modules removed, no double wrapping, router modules skipped unless router selected |
| Residency | selected bases are HostWeights or CPU tensors, no selected persistent CUDA param/buffer, original selected parameter owners removed, no duplicate persistent CPU base copy |
| Gradients | only LoRA params require grad, no base weight grad, router remains no-grad |
| Numerics | forward parity for every stage; loss parity for `lm_head` and whole-model stages; LoRA gradient parity for stages that install `AsymLoRALinear`; input-gradient parity for frozen-only linear stages |
| Runtime | SM100 BF16 AsymGEMM calls increase for linear buckets, no kernel fallback hidden by torch |
| Save/load | adapter-only state dict unchanged except metadata, reload restores adapter values |
| Profiling | component bytes attributed to the selected bucket, not `routed_experts` by default |

## Acceptance Criteria for Final Whole-Model Offload

With:

```bash
BACKEND=asym
ASYM_OFFLOAD_MODULES=all
```

on a supported MoE model:

1. `LFAsymReport` lists every selected component with nonzero CPU bytes when the
   model actually has that component.
2. `selected_gpu_resident_base_bytes_by_component` is zero.
3. `named_parameters()` contains only trainable LoRA parameters and any required
   framework wrapper parameters; no frozen base model parameter remains
   trainable.
4. No selected frozen base appears in `named_buffers()` on CUDA.
5. `collect_lf_offload_residency()` reports one persistent CPU owner for each
   selected frozen base weight. It must not report both an original CPU
   parameter and a cloned HostWeight for the same selected base.
6. The training loop completes forward, backward, optimizer step, and adapter
   save.
7. `asym_backend=torch` with the same selector remains a reference mode and does
   not create CPU HostWeights.
8. `ASYM_OFFLOAD_MODULES=routed_experts` remains compatible with existing runs
   and preserves current behavior.

## Explicit Non-Goals

- Do not move activations, KV cache, logits, or optimizer state with this
  selector.
- Do not add generic dense-FFN `mlp_dense` to `all` for this MoE-only design.
- Do not support full fine-tuning of CPU-resident base weights.
- Do not support non-BF16 LF AsymGEMM in this feature.
- Do not support non-SM100 AsymGEMM kernels in this feature.
- Do not support tied `embed_tokens`/`lm_head` offload in this target-model pass.
- Do not silently support unknown router or norm classes.
- Do not change the default selector away from `routed_experts`.

## Implementation Order Summary

1. Baseline lock for current `routed_experts`.
2. Shared selector parser, reports, and strict residency audit.
3. Attention projection offload.
4. Shared expert offload.
5. Router offload.
6. `lm_head` offload.
7. Token embedding offload.
8. Norm offload.
9. Enable `all` / `whole_model`.

This order converts the largest and lowest-risk linear buckets first, then
handles model-family router semantics, then handles non-GEMM wrappers. Each
stage is independently selectable through `ASYM_OFFLOAD_MODULES` only after its
validation gate passes.
