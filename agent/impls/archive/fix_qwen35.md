# Qwen3.5 Remaining Fix Plan

This document supersedes the old note that dense/shared/GDN LoRA weights can
remain CUDA-resident "by design". That is not an acceptable final comparison
against ZeRO-3 offload. ZeRO-3 offloads trainable LoRA parameters too; Asym must
either offload the same trainable LoRA surface or prove with real `profile_lora_lf.sh`
numbers that the missing surface is irrelevant. Current code has not proven
that.

Use only this profiling entry point for acceptance:

```bash
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Use `Qwen/Qwen3.5-35B-A3B` for faster Qwen3.5 validation; the profiling script
will download the model if the checkpoint is not already cached. Do not use
`profile3.sh`, `profile_lora_lf_test.sh`, or `profile_lora_lf_test2.sh` for this
Qwen3.5 fix plan. Every real profiling command below must bind only CPU RAM
nodes:

```bash
NUMACTL_MEMBIND=0,1
NUMACTL_CPUNODEBIND=0,1
NUMACTL_MODE=membind
```

Online/local architecture references checked:

- Hugging Face Qwen3.5 docs: Qwen3.5 uses a 3:1 hybrid stack of Gated DeltaNet
  linear-attention layers and full-attention layers.
- Local Transformers source:
  `third_party/transformers/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`
  defines `Qwen3_5MoeGatedDeltaNet` with leaves
  `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, and `out_proj`, then
  calls `causal_conv1d` and `chunk_gated_delta_rule`.
- FLA project docs/source: `flash-linear-attention` owns the optimized linear
  attention kernels. Asym must wrap around those kernels; do not reimplement
  the GDN core math in Python or split it into small GEMMs.

## Current Hard Facts

1. `asym_gemm/training/weight_offload.py` is only half-general today.
   `LoRAWeightOffloadCoordinator.register_group()` can stage arbitrary
   parameter groups, but `install_lora_weight_offload()` only registers
   `AsymQwen3Experts`. Also, only the routed expert custom autograd paths know
   how to release weights after forward and regather them in backward.

2. Qwen3 routed expert LoRA weight offload works because
   `AsymQwen3Experts` owns six expert banks:
   `gate_lora_A`, `gate_lora_B`, `up_lora_A`, `up_lora_B`,
   `down_lora_A`, `down_lora_B`.

3. Llama4 routed expert LoRA weight offload works because
   `AsymLlama4Experts` subclasses `AsymQwen3Experts`. Llama4 shared expert
   frozen base offload and shared-MLP activation offload exist, but Llama4
   shared expert trainable LoRA A/B weights are not JIT weight-offloaded.

4. Qwen3.5 routed expert LoRA weight offload works through
   `AsymQwen35MoeBlock.experts = wrap_qwen3_experts(...)`. Qwen3.5 shared
   expert and GDN projection LoRA weights do not get the same trainable
   weight-offload treatment today.

5. Current `linear_attention` component selection is already present in
   `asym_gemm/integrations/lf.py` and profiling classification tests. Do not
   re-plan that as missing. The remaining problem is trainable LoRA weight
   staging and GDN activation/saved-tensor behavior.

## Math First

### Routed Expert MLP

This is the existing Qwen3 math and is also used for Qwen3.5 routed experts.

```text
X                  [M,H] routed rows
W_gate_up_cpu       [E,2I,H] frozen CPU base
W_down_cpu          [E,H,I] frozen CPU base
A_gate/A_up         [E,r,H] trainable LoRA-A banks
B_gate/B_up         [E,I,r] trainable LoRA-B banks
A_down              [E,r,I]
B_down              [E,H,r]

gate_up = grouped(X_g @ W_gate_up_cpu[e_g].T)      # [M,2I]
gate, up = split(gate_up)                          # [M,I], [M,I]

S_gate = grouped(X_g @ A_gate[e_g].T)              # [M,r]
S_up   = grouped(X_g @ A_up[e_g].T)                # [M,r]
gate += scale * grouped(S_gate_g @ B_gate[e_g].T)  # [M,I]
up   += scale * grouped(S_up_g   @ B_up[e_g].T)    # [M,I]

act = silu(gate) * up                              # [M,I]

S_down = grouped(act_g @ A_down[e_g].T)            # [M,r]
Y = grouped(act_g @ W_down_cpu[e_g].T)             # [M,H]
Y += scale * grouped(S_down_g @ B_down[e_g].T)     # [M,H]
```

Do not add per-expert Python loops. All routed work stays grouped.

### Qwen3.5 Shared Expert MLP

The shared expert is not an expert bank. There is no expert dimension `E`,
no routing index, and no `top_k` selection. It is one dense SwiGLU MLP that runs
for every token, then gets multiplied by a scalar shared gate:

```text
X                         [B*S,H]
W_s_gate_cpu              [I_s,H]
W_s_up_cpu                [I_s,H]
W_s_down_cpu              [H,I_s]
A_s_gate/A_s_up           [r,H]
B_s_gate/B_s_up           [I_s,r]
A_s_down                  [r,I_s]
B_s_down                  [H,r]
W_shared_gate_cpu         [1,H]

G = X @ W_s_gate_cpu.T + scale * ((X @ A_s_gate.T) @ B_s_gate.T)
U = X @ W_s_up_cpu.T   + scale * ((X @ A_s_up.T)   @ B_s_up.T)
S = silu(G) * U
Y_shared = S @ W_s_down_cpu.T + scale * ((S @ A_s_down.T) @ B_s_down.T)

shared_scale = sigmoid(X @ W_shared_gate_cpu.T)     # [B*S,1]
Y_mlp = Y_routed + shared_scale * Y_shared
```

So "split LoRA" here means separate dense leaves
`shared_expert.gate_proj`, `shared_expert.up_proj`, and
`shared_expert.down_proj`. It does not mean expert-bank splitting.

### Qwen3.5 Gated DeltaNet Linear Attention

The GDN projections are ordinary dense projections around the FLA core:

```text
X = hidden_states                                      # [B,S,H]

mixed_qkv = in_proj_qkv(X)                             # [B,S,2*K+V]
z         = in_proj_z(X)                               # [B,S,V]
b         = sigmoid(in_proj_b(X))                      # [B,S,num_v_heads]
a         = in_proj_a(X)                               # [B,S,num_v_heads]

mixed_qkv = causal_conv1d(mixed_qkv, conv1d.weight)    # optimized conv
q, k, v = split(mixed_qkv)
g = -exp(A_log) * softplus(a + dt_bias)

core, state = chunk_gated_delta_rule(q, k, v, g, b)    # FLA kernel
core = norm(core, z)
Y = out_proj(core)
```

Asym should own/offload the five projection linears and their LoRA A/B weights.
It must leave `causal_conv1d` and `chunk_gated_delta_rule` as optimized FLA
calls. The GDN core is not where to add AsymGEMM.

### Trainable LoRA Weight Offload Protocol

For a group of trainable LoRA parameters `P_i`:

```text
home_cpu = concat(P_i.detach())            # pinned BF16 CPU home
P_i.data = empty CUDA placeholder          # no real HBM residency at rest

before forward use:
    slab = stage(home_cpu)                 # one H2D per group
    P_i.data = views(slab)

after forward use:
    P_i.data = placeholder                 # release HBM before loss/lm_head peak

before backward use:
    slab = stage(home_cpu)                 # one H2D per group
    P_i.data = views(slab)

after AccumulateGrad / grad offload:
    P_i.data = placeholder

after optimizer.step:
    home_cpu[...] = cpu_master_fp32.cast(bf16)
```

This is safe only when the autograd function does not save the full LoRA weight
tensors across forward-to-backward. Normal `nn.Linear` autograd is not safe for
this release/regather protocol.

## Active Stage Order

Current implementation agents should execute only:

```text
Stage 1 -> Stage 2 -> Stage 4 -> Stage 5 final verdict
```

Stage 3 is recorded as future work and must be skipped in the current Qwen3.5
fix pass. Do not modify Stage 3 files, do not run Stage 3 validation, and do not
block the final Stage 5 verdict on Stage 3.

Every validation command in this document must keep the comparison target:

```text
asym_cpuadamwds|norecomp lower peak HBM than zero3_offload|recomp
```

The full explicit backend specs use `ligerloss0` to pin the no-Liger-loss axis:
`asym_cpuadamwds|norecomp|ligerloss0,zero3_offload|recomp|ligerloss0`.

Required Qwen3.5 offload parity before the final verdict:

- Routed expert frozen base weights and routed expert LoRA banks must stay on
  the existing Qwen3/Qwen3.5 `AsymQwen3Experts` path.
- Shared expert frozen base weights and shared expert trainable LoRA A/B weights
  must be offloaded. This is the part Llama4 shared MLP did not fully cover for
  trainable LoRA weight offload.
- Qwen3.5 GDN/`linear_attn` projection frozen base weights and trainable LoRA
  A/B weights must be offloaded as one parent group per `linear_attn` layer.
- Full-attention q/k/v/o projection frozen base weights and trainable LoRA A/B
  weights must remain covered by the attention Asym paths.
- Selected frozen base components under `ASYM_OFFLOAD_MODULES=all` must not
  remain CUDA-resident except for tiny explicitly allowlisted non-GEMM state.

## Stage 1: General Trainable LoRA Weight Offload

Goal: make Asym offload trainable LoRA A/B weights for dense, attention,
shared-expert, and GDN projection leaves, not only routed expert banks.

Files/classes/functions to modify:

- `asym_gemm/training/weight_offload.py`
  - `LoRAWeightOffloadCoordinator`
  - `_LayerGroup`
  - `install_lora_weight_offload`
- `asym_gemm/training/lora.py`
  - `AsymLoRALinear`
  - add `_AsymLoRALinearWeightOffloadFunction`
- `asym_gemm/training/attention_activation_offload.py`
  - `AsymActivationOffloadLoRALinear`
  - `_AsymActivationOffloadLoRALinearFunction`
- `asym_gemm/training/llama4_shared_mlp.py`
  - `AsymLlama4SharedMLP`
  - `_Llama4SharedMLPActivationOffloadFunction`
- `asym_gemm/training/qwen35_shared_mlp.py`
  - inherits shared implementation; add only Qwen3.5-specific tests if needed
- `asym_gemm/training/cpu_adam.py`
  - ensure weight-offload release hooks work even when grad offload behavior
    changes
- `LlamaFactory/src/llamafactory/train/trainer_utils.py`
  - update install error/log text and summary expectations
- `asym_gemm/integrations/lf.py`
  - reuse `classify_lf_component` or a local helper to identify Qwen3.5
    `linear_attn` parents during weight-offload registration
- Tests:
  - `tests/training/test_lf_qwen35_asym_backend.py`
  - `tests/training/test_lf_qwen3_asym_backend.py`
  - add `tests/training/test_lora_weight_offload_generic.py`

Concrete implementation:

1. Keep `LoRAWeightOffloadCoordinator.register_group()` as the low-level group
   primitive, but add explicit group metadata:

```python
class _LayerGroup:
    module_key: int
    group_name: str
    component: str
    params: list[nn.Parameter]
    ...
```

2. Add coordinator helpers:

```python
def is_module_registered(self, module) -> bool:
    return id(module) in self._group_of_module

def release_group(self, module) -> None:
    group = self._group_of_module.get(id(module))
    if group is not None:
        self._release_group(group)

def group_for_module(self, module) -> _LayerGroup | None:
    return self._group_of_module.get(id(module))

def register_group(self, module, named_banks, *, group_name="", component="lora") -> int:
    # existing capture-to-CPU and placeholder replacement
    # store group_name/component for reporting
```

3. Add a dense LoRA branch custom function in `lora.py`. This function handles
   only the LoRA branch; the frozen base branch stays in `AsymFrozenLinear`:

```python
class _AsymLoRALinearWeightOffloadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lora_a, lora_b, module):
        if not isinstance(module.lora_dropout, nn.Identity):
            raise RuntimeError("generic LoRA weight offload v1 requires lora_dropout=0.00")
        module.gather_lora_weights()
        a = module.lora_a
        b = module.lora_b
        x_lora = x.to(dtype=module.lora_dtype).contiguous()
        low = F.linear(x_lora, a)                      # [M,r]
        out = F.linear(low, b) * module.scaling        # [M,out]
        ctx.module = module
        ctx.input_shape = tuple(x.shape)
        ctx.save_for_backward(x_lora, low)
        if module._weight_offload_release_after_forward:
            module.release_lora_weights()              # standalone leaf group
        return out.to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        module = ctx.module
        module.gather_lora_weights()
        a = module.lora_a
        b = module.lora_b
        x_lora, low = ctx.saved_tensors
        go = grad_out.reshape(-1, grad_out.shape[-1]).to(dtype=module.lora_dtype)
        go = go * module.scaling
        grad_b = go.transpose(0, 1).matmul(low)
        dlow = go.matmul(b)
        grad_a = dlow.transpose(0, 1).matmul(x_lora)
        grad_x = dlow.matmul(a).reshape(ctx.input_shape).to(dtype=grad_out.dtype)
        return grad_x, grad_a, grad_b, None
```

4. Update `AsymLoRALinear`:

```python
class AsymLoRALinear(nn.Module):
    self._weight_offload = None
    self._weight_offload_owner = self
    self._weight_offload_release_after_forward = True

    def gather_lora_weights(self):
        if self._weight_offload is not None:
            self._weight_offload.gather_group(self._weight_offload_owner)

    def release_lora_weights(self):
        if self._weight_offload is not None:
            self._weight_offload.release_group(self._weight_offload_owner)

    def _uses_lora_weight_offload(self):
        return self._weight_offload is not None and self.training and torch.is_grad_enabled()

    def forward(self, x):
        base = self.base_layer(x)
        if self._uses_lora_weight_offload():
            flat, shape = flatten_last_dim(x, self.base_layer.in_features)
            lora = _AsymLoRALinearWeightOffloadFunction.apply(flat, self.lora_a, self.lora_b, self)
            lora = restore_last_dim(lora, shape, self.base_layer.out_features)
        else:
            lora_input = self.lora_dropout(x).to(dtype=self.lora_dtype)
            lora = self.lora_B[self.active_adapter](self.lora_A[self.active_adapter](lora_input)) * self.scaling
        return base + lora.to(dtype=base.dtype)
```

5. Update `AsymActivationOffloadLoRALinear`. Its current custom function already
   passes `lora_A` and `lora_B`; modify it so offload mode does not save those
   tensors in `ctx.saved_tensors`. It must save only activation data/low-rank
   data, then call `module.gather_lora_weights()` in backward before reading A/B.

```python
class AsymActivationOffloadLoRALinear(nn.Module):
    self._weight_offload = None
    self._weight_offload_owner = self
    self._weight_offload_release_after_forward = True
    def gather_lora_weights(...): ...
    def release_lora_weights(...): ...

    def forward(self, x):
        return _AsymActivationOffloadLoRALinearFunction.apply(
            x, self.lora_a, self.lora_b, self.base_layer, ..., self
        )
```

Inside the function:

```python
if module_has_weight_offload:
    module.gather_lora_weights()
    # compute forward
    ctx.weight_offload_module = module
    ctx.save_for_backward(activation_handles_or_low_rank_only)
    module.release_lora_weights()
else:
    ctx.save_for_backward(a, b, ...)

backward:
    if ctx.weight_offload_module is not None:
        module.gather_lora_weights()
        a = module.lora_a
        b = module.lora_b
```

6. Update `AsymLlama4SharedMLP` activation-offload function. It currently saves
   the six LoRA weights. In weight-offload mode, save only activations/low-rank
   tensors and the layer object. Regather the six child LoRA weights in
   backward.

```python
class AsymLlama4SharedMLP(nn.Module):
    def _lora_weight_banks(self):
        return [
            ("gate_lora_A", self.gate_proj.lora_a),
            ("gate_lora_B", self.gate_proj.lora_b),
            ("up_lora_A", self.up_proj.lora_a),
            ("up_lora_B", self.up_proj.lora_b),
            ("down_lora_A", self.down_proj.lora_a),
            ("down_lora_B", self.down_proj.lora_b),
        ]

    def gather_lora_weights(self): coordinator.gather_group(self)
    def release_lora_weights(self): coordinator.release_group(self)
```

Use one group per shared MLP layer, not three tiny groups, to keep H2D copies
coalesced.

7. Add Qwen3.5 GDN parent grouping. Do not register
   `linear_attn.in_proj_a` and `linear_attn.in_proj_b` as tiny standalone
   groups. Register all five GDN projection leaves as one group on the
   `linear_attn` parent:

```python
def _is_qwen35_linear_attn_parent(name, module):
    children = dict(module.named_children())
    return (
        name.endswith(".linear_attn")
        and {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"} <= set(children)
    )

def _linear_attn_lora_banks(module):
    banks = []
    for leaf in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
        child = getattr(module, leaf)
        if isinstance(child, (AsymLoRALinear, AsymActivationOffloadLoRALinear)):
            banks.extend([(f"{leaf}.lora_A", child.lora_a), (f"{leaf}.lora_B", child.lora_b)])
    return banks
```

Install parent hooks:

```python
def _parent_gather_hook(module, _inputs):
    module.gather_lora_weights()

def _parent_release_hook(module, _inputs, _output):
    module.release_lora_weights()

linear_attn_parent._weight_offload = coordinator
linear_attn_parent.gather_lora_weights = lambda: coordinator.gather_group(linear_attn_parent)
linear_attn_parent.release_lora_weights = lambda: coordinator.release_group(linear_attn_parent)
linear_attn_parent.register_forward_pre_hook(_parent_gather_hook)
linear_attn_parent.register_forward_hook(_parent_release_hook)

for child in projection_children:
    child._weight_offload = coordinator
    child._weight_offload_owner = linear_attn_parent
    child._weight_offload_release_after_forward = False
```

Forward behavior: the parent gathers one GDN layer group before the GDN forward
and releases it after all five projection leaves have run. Backward behavior:
each projection's custom LoRA function regathers the same parent group before
computing its A/B gradients; CPUAdamW releases the group as its parameter grad
hooks fire.

8. Update `install_lora_weight_offload()`:

```python
def install_lora_weight_offload(model, coordinator):
    installed = 0

    for module in model.modules():
        if isinstance(module, AsymQwen3Experts):
            installed += register six expert banks exactly as today

    for module in model.modules():
        if isinstance(module, (AsymLlama4SharedMLP, AsymQwen35SharedMLP)):
            banks = module._lora_weight_banks()
            registered = coordinator.register_group(
                module, banks, group_name="shared_mlp", component="shared_experts"
            )
            if registered:
                module._weight_offload = coordinator
                installed += registered

    for name, module in model.named_modules():
        if _is_qwen35_linear_attn_parent(name, module):
            banks = _linear_attn_lora_banks(module)
            registered = coordinator.register_group(
                module, banks, group_name=name, component="linear_attention"
            )
            if registered:
                install parent gather/release hooks
                mark all projection children with owner=module and release_after_forward=False
                installed += registered

    for name, module in model.named_modules():
        if isinstance(module, (AsymLoRALinear, AsymActivationOffloadLoRALinear)):
            if getattr(module, "_weight_offload_owner", module) is not module:
                continue
            if is_child_of_registered_shared_mlp(name):
                continue
            component = classify_lf_component(name, module)
            if component not in {"attention", "linear_attention", "lm_head", "mlp_dense", "shared_experts"}:
                continue
            banks = [("lora_A", module.lora_a), ("lora_B", module.lora_b)]
            registered = coordinator.register_group(
                module, banks, group_name=name, component=component
            )
            if registered:
                module._weight_offload = coordinator
                installed += registered

    return installed
```

9. Do not register groups smaller than the existing persistence threshold unless
   profiling proves the threshold hides meaningful memory. Log skipped groups by
   component and total skipped bytes so the final profile can explain what is
   still CUDA-resident. The shared-MLP and GDN parent grouping above is required
   specifically so small-but-repeated leaves are not skipped leaf by leaf.

10. Ensure release after backward is independent of grad-offload details. If
   `AsymCPUAdamW` already releases from the grad offload hook, keep that path.
   Add a separate post-accumulate release hook only for the case where
   weight-offload is enabled and grad-offload hook is not installed.

Validations before moving on:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

pytest -q \
  tests/training/test_lora_weight_offload_generic.py \
  tests/training/test_lf_qwen35_asym_backend.py \
  tests/training/test_lf_qwen3_asym_backend.py
```

Required real Qwen3.5 profile gate:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

NUMACTL_MEMBIND=0,1 \
NUMACTL_CPUNODEBIND=0,1 \
NUMACTL_MODE=membind \
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/outputs/qwen35_stage1_generic_lora_weight_offload \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=true \
MODEL_SPECS='Qwen/Qwen3.5-35B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0,zero3_offload|recomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|true' \
ASYM_OFFLOAD_MODULES=all \
ASYM_STRICT=true \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
LF_EXPERT_LORA_IMPLS=split-target-parameters \
LORA_DROPOUT=0.00 \
LORA_RANK=64 \
LORA_ALPHA=16 \
WORKLOADS='2048|4|1' \
MAX_STEPS=1 \
WARMUP_STEPS=1 \
bash scripts/lf/profile_lora_lf.sh --overwrite true
```

Accept Stage 1 only if:

- `weight_offload_group_count` increases beyond routed expert layers.
- Summary reports offloaded groups for `shared_experts` and
  `linear_attention` when those LoRA leaves exist.
- CUDA-resident trainable LoRA bytes for shared/GDN/dense components drop by at
  least 2 GiB or 5% peak HBM, whichever is larger.
- Forward, backward, and e2e step time are not more than 20% slower than the
  previous accepted Asym row.
- Losses remain finite and trainable parameter counts match the pre-stage
  model surface, including CPU-home offloaded params.

Reject Stage 1 if memory is unchanged while latency increases, or if the only
memory drop is below the acceptance threshold.

Risks to watch:

- Returning gradients for offloaded dense LoRA params must still trigger
  `AccumulateGrad` and CPUAdamW grad offload. Unit tests must prove the params
  receive correct gradients after their `.data` was a placeholder at rest.
- `lora_dropout > 0` is not required by `profile_lora_lf.sh` for Qwen3.5. Keep v1
  strict for `LORA_DROPOUT=0.00`; do not silently produce wrong dropout grads.
- Saving `low [M,r]` is acceptable; saving wide dropped activations or LoRA
  weights is not.

## Stage 2: First-Class Qwen3.5 GDN Saved-Tensor Offload

Goal: reduce the `linear_attention` activation/saved-tensor peak without
touching FLA core kernels.

Files/classes/functions to modify:

- Add `asym_gemm/training/linear_attention_activation_offload.py`
  - `LinearAttentionSavedTensorOffloadWrapper`
  - `install_linear_attention_saved_tensor_offload`
  - `linear_attention_saved_tensor_offload_module_names`
- `asym_gemm/training/__init__.py`
  - export the new wrapper helpers
- `asym_gemm/integrations/lf.py`
  - install the wrapper on Qwen3.5 `linear_attn` modules
  - add report fields:
    `linear_attention_saved_tensor_offload_wrapped`,
    `linear_attention_saved_tensor_offload_modules`,
    `linear_attention_saved_tensor_offload_skipped`
- `asym_gemm/profiling/lf_trace.py`
  - ensure wrapper ranges/stats are tagged as `linear_attention`
- Tests:
  - `tests/training/test_lf_qwen35_asym_backend.py`
  - `tests/test_lf_memory_breakdown.py`

Concrete implementation:

1. Build the new wrapper by copying the structure of
   `DecoderSavedTensorOffloadWrapper`, but use GDN-specific env vars and tags:

```python
ASYM_LINEAR_ATTN_SAVED_TENSOR_OFFLOAD_MIN_BYTES
ASYM_LINEAR_ATTN_SAVED_TENSOR_OFFLOAD_DTYPES
ASYM_LINEAR_ATTN_SAVED_TENSOR_OFFLOAD_REQUIRE_GRAD
```

Default `REQUIRE_GRAD=false` for this wrapper. FLA and convolution kernels may
save tensors that are not leaf trainable tensors but still dominate HBM.
Continue to skip leaf trainable parameters.

2. Install only on modules that look like Qwen3.5 GDN:

```python
def is_qwen35_linear_attention_module(name, module):
    children = dict(module.named_children())
    return (
        name.endswith(".linear_attn")
        and {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"} <= set(children)
        and hasattr(module, "chunk_gated_delta_rule")
    )
```

3. In `lf.py`, after dense projection replacement and before layer wrapper
   installation:

```python
if layer_act_enabled and selection.linear_attention:
    for name, module in model.named_modules():
        if is_qwen35_linear_attention_module(name, module):
            install_linear_attention_saved_tensor_offload(module)
```

Nested saved-tensor hooks are acceptable: the inner `linear_attn` wrapper owns
GDN-specific tensors, while the decoder-layer wrapper can still cover the rest
of the decoder layer.

4. Do not reimplement `Qwen3_5MoeGatedDeltaNet.forward`. The source module keeps
   calling `causal_conv1d_fn`, `chunk_gated_delta_rule`, and
   `fused_recurrent_gated_delta_rule`.

Validations before moving on:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

pytest -q \
  tests/training/test_lf_qwen35_asym_backend.py \
  tests/test_lf_memory_breakdown.py
```

Required real Qwen3.5 profile gate:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

NUMACTL_MEMBIND=0,1 \
NUMACTL_CPUNODEBIND=0,1 \
NUMACTL_MODE=membind \
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/outputs/qwen35_stage2_gdn_saved_tensor_offload \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=true \
MODEL_SPECS='Qwen/Qwen3.5-35B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0,zero3_offload|recomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|true' \
ASYM_OFFLOAD_MODULES=all \
ASYM_STRICT=true \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
LF_EXPERT_LORA_IMPLS=split-target-parameters \
LORA_DROPOUT=0.00 \
LORA_RANK=64 \
LORA_ALPHA=16 \
WORKLOADS='2048|4|1' \
MAX_STEPS=1 \
WARMUP_STEPS=1 \
bash scripts/lf/profile_lora_lf.sh --overwrite true
```

Accept Stage 2 only if:

- `linear_attention_saved_tensor_offload_wrapped > 0`.
- Memory breakdown shows lower `linear_attention` peak or lower total peak HBM
  by at least 2 GiB or 5%.
- FLA fast path is still active in the environment.
- Forward/backward/e2e latency stays within the 20% limit.

Reject Stage 2 if it only moves tensors to CPU but introduces more staging peak
or slows the step without meaningful HBM reduction.

Risks to watch:

- Some FLA saved tensors may be opaque to PyTorch saved-tensor hooks. If wrapper
  stats show zero offloaded bytes, reject this stage and move to profiling the
  exact FLA temporary allocation source instead of adding more hooks.
- Nested hooks must not double-copy the same saved tensor. Stats should show
  plausible bytes, not duplicated multiples of the same shape.

## Stage 3 (On Hold): Routed Expert Temporary Peak Reduction

Status: recorded future work only. Do not implement this stage in the current
Qwen3.5 fix pass. The current implementation path skips directly from Stage 2
to Stage 4.

Future goal: reduce routed expert temporary HBM after Stage 1/2, without
changing the grouped expert math or adding per-expert loops.

Known concrete current temporary sources:

- `qwen3_moe.py::_forward_gate_up_lora()` creates
  `gate_up_a = torch.cat((self.gate_lora_A, self.up_lora_A), dim=1)` for the
  no-dropout HBM path.
- `exp_act_offload_lora.py::grouped_lora_a_forward_hbm()` accepts only one
  LoRA-A bank.
- `lora.py::grouped_expert_lora_pair()` uses `torch.cat` on low-rank inputs and
  selected weights to force one grouped-mm launch.

Files/classes/functions to modify later, when this stage is explicitly resumed:

- `asym_gemm/training/exp_act_offload_lora.py`
  - add `grouped_lora_a_pair_forward_hbm`
- `asym_gemm/training/qwen3_moe.py`
  - `_ActivationOffloadQwen3ExpertFunction.forward`
  - `AsymQwen3Experts._forward_gate_up_lora`
- `asym_gemm/training/llama4_experts.py`
  - mirror routed expert pair path if it uses the same HBM concat
- `asym_gemm/training/lora.py`
  - optional Python fallback only; do not accept fallback as final if it
    increases launches and does not reduce memory
- Native extension, if needed:
  - `csrc/exp_act_offload/exp_act_offload_kernels.cu`
  - `csrc/apis/exp_act_offload.hpp`
  - `csrc/python_api.cpp`
  - `setup.py`
- Tests:
  - `tests/training/test_cpu_left_lora.py`
  - `tests/training/test_lf_qwen3_asym_backend.py`
  - `tests/training/test_lf_qwen35_asym_backend.py`

Future concrete implementation:

1. Add a pair HBM LoRA-A API that produces one `[M,2r]` owner without materializing
   `cat(A_gate, A_up)`:

```python
def grouped_lora_a_pair_forward_hbm(
    source_hbm,
    lora_a_gate,
    lora_a_up,
    offsets,
    experts,
    *,
    metadata,
    stats,
    tag,
) -> tuple[Tensor, Tensor, Tensor]:
    if native sm100_grouped_lora_a_pair_forward_bf16_hbm exists:
        owner = native(source_hbm, lora_a_gate, lora_a_up, offsets, experts, ...)
        gate, up = owner.split(lora_a_gate.shape[1], dim=-1)
        return owner, gate, up
    # fallback is for correctness tests only
    gate = grouped_lora_a_forward_hbm(source_hbm, lora_a_gate, ...)
    up = grouped_lora_a_forward_hbm(source_hbm, lora_a_up, ...)
    owner = torch.cat((gate, up), dim=-1)
    return owner, owner[:, :r], owner[:, r:]
```

The native path is the accepted performance path. It should use active-group
metadata and write the two low-rank outputs contiguously. It must not loop over
experts in Python.

2. Replace the cat path in `AsymQwen3Experts._forward_gate_up_lora()`:

```python
if self.lora_dropout_p == 0.0 and hbm_pair_path_enabled:
    owner, gate_low_rank, up_low_rank = grouped_lora_a_pair_forward_hbm(
        x_lora, self.gate_lora_A, self.up_lora_A, offsets, experts, metadata=metadata, ...
    )
else:
    existing path
```

3. In `_ActivationOffloadQwen3ExpertFunction.forward`, replace:

```python
gate_up_a = torch.cat((gate_lora_A, up_lora_A), dim=1).contiguous()
gate_up_low_rank_owner = grouped_lora_a_forward_hbm(..., gate_up_a, ...)
```

with:

```python
gate_up_low_rank_owner, gate_low_rank, up_low_rank = grouped_lora_a_pair_forward_hbm(
    packed_or_stage, gate_lora_A, up_lora_A, offsets, experts, ...
)
```

4. Release order must be explicit:

```python
gate_delta, up_delta = grouped_expert_lora_pair(...)
gate.add_(gate_delta.to(gate.dtype))
up.add_(up_delta.to(up.dtype))
del gate_delta, up_delta
offload gate_low_rank/up_low_rank if needed
del gate_low_rank, up_low_rank, gate_up_low_rank_owner
```

5. Do not replace one grouped operation with many small GEMMs. One native pair
   grouped kernel is acceptable. Two grouped kernels are acceptable only as a
   diagnostic fallback and must be rejected if latency rises without a memory
   win.

Future validation commands, only after this hold is lifted:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

python -m pip install -e .

pytest -q \
  tests/training/test_cpu_left_lora.py \
  tests/training/test_lf_qwen3_asym_backend.py \
  tests/training/test_lf_qwen35_asym_backend.py
```

Future real Qwen3.5 profile gate, only after this hold is lifted:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

NUMACTL_MEMBIND=0,1 \
NUMACTL_CPUNODEBIND=0,1 \
NUMACTL_MODE=membind \
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/outputs/qwen35_stage3_routed_temp_peak \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=true \
MODEL_SPECS='Qwen/Qwen3.5-35B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0,zero3_offload|recomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|true' \
ASYM_OFFLOAD_MODULES=all \
ASYM_STRICT=true \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
LF_EXPERT_LORA_IMPLS=split-target-parameters \
LORA_DROPOUT=0.00 \
LORA_RANK=64 \
LORA_ALPHA=16 \
WORKLOADS='2048|4|1' \
MAX_STEPS=1 \
WARMUP_STEPS=1 \
bash scripts/lf/profile_lora_lf.sh --overwrite true
```

Future acceptance, only after this hold is lifted:

- `routed_experts` peak or total peak HBM drops by at least 2 GiB or 5%.
- Kernel launch count does not grow materially in the routed expert forward.
- Forward/backward/e2e latency stays within the 20% limit.

Reject if the fallback two-kernel path is the only implemented path and it
raises latency without a meaningful HBM drop.

Risks to watch:

- A native pair kernel must match grouped routing metadata exactly, including
  empty groups and dense-expert metadata.
- In-place adds must not mutate tensors still needed for backward unless the
  custom autograd path saves/offloads the correct pre-add values.

## Stage 4: Frozen CUDA Residue Audit And Fix

Goal: eliminate unexplained frozen CUDA-resident base weights that make
Asym larger than ZeRO after selected components are supposedly offloaded.

This is not a vague cleanup stage. The implementation is an enforced audit:
produce the exact names and components of residual frozen CUDA weights, then fix
classification/wrapping for those names.

Files/functions to modify:

- `asym_gemm/training/offload.py`
  - `collect_lf_offload_residency`
  - `validate_lf_offload_residency`
- `asym_gemm/integrations/lf.py`
  - `classify_lf_component`
  - `component_is_selected`
  - replacement logic in `apply_lf_asym_lora`
- `scripts/lf/run_lf_profiled_train.py`
  - persist residual rows in source profile config/report
- `scripts/lf/postprocess_lf_profile_artifacts.py`
  - summarize residual frozen CUDA bytes by component and top names
- Tests:
  - `tests/test_lf_memory_breakdown.py`
  - `tests/training/test_lf_qwen35_asym_backend.py`

Concrete implementation:

1. Extend residency reporting to always include top residual frozen CUDA rows:

```python
def frozen_cuda_residue(rows):
    return [
        row for row in rows
        if row.kind in {"parameter", "buffer"}
        and row.device == "cuda"
        and not row.requires_grad
        and not is_lora_name(row.name)
    ]
```

Persist:

```json
{
  "frozen_cuda_residue_bytes_by_component": {...},
  "frozen_cuda_residue_top": [
    {"name": "...", "component": "...", "bytes": ..., "selected_for_cpu": false}
  ]
}
```

2. In strict Qwen3.5 `ASYM_OFFLOAD_MODULES=all`, enforce:

```python
if selection.raw == "all" and model_is_qwen35:
    bad = [
        row for row in frozen_cuda_residue(rows)
        if row.bytes >= 16 * 1024**2
        and not allowlisted_small_state(row.name)
    ]
    if sum(row.bytes for row in bad) > 128 * 1024**2:
        raise RuntimeError("unexplained frozen CUDA residue", bad[:50])
```

Allowlist only small non-GEMM state that is not worth offloading:
`linear_attn.A_log`, `linear_attn.dt_bias`, small norm weights, and tiny scalar
buffers. Do not allowlist large `nn.Linear.weight` tensors.

3. For every failed row, fix classification or replacement:

```python
if ".linear_attn.conv1d." in name:
    classify as "linear_attention"
    decide separately whether Conv1d weight should stay CUDA or be CPU-owned.
if ".visual." or ".vision_tower." in name and text-only SFT does not use it:
    move unused frozen module to CPU or exclude it from the loaded training model.
if ".mlp." gate/up/down dense leaf is outside routed/shared wrappers:
    classify as "mlp_dense" and wrap with AsymLoRALinear/AsymFrozenLinear.
```

4. Do not hide residue by changing reporting labels. A fix must reduce actual
CUDA bytes or explicitly prove the residual is a tiny non-GEMM state below the
threshold.

Validations before moving on:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

pytest -q \
  tests/test_lf_memory_breakdown.py \
  tests/training/test_lf_qwen35_asym_backend.py
```

Required real Qwen3.5 profile gate:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

NUMACTL_MEMBIND=0,1 \
NUMACTL_CPUNODEBIND=0,1 \
NUMACTL_MODE=membind \
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/outputs/qwen35_stage4_frozen_cuda_residue \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=true \
MODEL_SPECS='Qwen/Qwen3.5-35B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0,zero3_offload|recomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|true' \
ASYM_OFFLOAD_MODULES=all \
ASYM_STRICT=true \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
LF_EXPERT_LORA_IMPLS=split-target-parameters \
LORA_DROPOUT=0.00 \
LORA_RANK=64 \
LORA_ALPHA=16 \
WORKLOADS='2048|4|1' \
MAX_STEPS=1 \
WARMUP_STEPS=1 \
bash scripts/lf/profile_lora_lf.sh --overwrite true
```

Accept Stage 4 only if:

- `selected_gpu_resident_base_bytes_by_component` is zero for selected large
  components.
- Unexplained frozen CUDA residue above 128 MiB is gone or identified as
  intentionally used non-GEMM state.
- Total HBM drops meaningfully if the previous run had large residue.
- Latency is flat or better. Classification/reporting-only changes cannot be
  accepted as memory fixes unless actual HBM drops.

Risks to watch:

- Qwen3.5 checkpoints may include multimodal/vision modules. Do not offload or
  delete modules that the current training path actually uses. Prove with module
  call counts or forward hooks before moving a large `other_model` module.

## Stage 5: Final Verdict Profile

Goal: decide from real numbers whether Qwen3.5 Asym is memory-better than
ZeRO-3 offload under the exact comparison config.

No implementation changes in this stage. Run this after Stage 4. Stage 3 is on
hold and must not block this verdict.

Final command:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

NUMACTL_MEMBIND=0,1 \
NUMACTL_CPUNODEBIND=0,1 \
NUMACTL_MODE=membind \
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/outputs/qwen35_final_fix_verdict \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=true \
MODEL_SPECS='Qwen/Qwen3.5-35B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0,zero3_offload|recomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|true' \
ASYM_OFFLOAD_MODULES=all \
ASYM_STRICT=true \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
LF_EXPERT_LORA_IMPLS=split-target-parameters \
LORA_DROPOUT=0.00 \
LORA_RANK=64 \
LORA_ALPHA=16 \
WORKLOADS='2048|4|1' \
MAX_STEPS=1 \
WARMUP_STEPS=1 \
bash scripts/lf/profile_lora_lf.sh --overwrite true
```

Final acceptance:

- `asym_cpuadamwds|norecomp|ligerloss0` peak allocated HBM must be lower than
  `zero3_offload|recomp|ligerloss0` by at least 2 GiB or 5%.
- Asym peak reserved HBM must not be worse in a way that invalidates the peak
  allocated win.
- Asym forward, backward, and e2e step time must each be no more than 20% slower
  than ZeRO, unless the memory win is large enough that the tradeoff is
  explicitly accepted by the user.
- `weight_offload_param_numel` must include routed expert LoRA plus accepted
  shared/GDN/dense groups.
- `linear_attention`, `shared_experts`, and `routed_experts` must appear as
  separate memory-breakdown components.
- Artifacts must show `liger_loss=ligerloss0`,
  `ENABLE_LIGER_KERNEL=false`, `ASYM_STRICT=true`,
  `NUMACTL_MEMBIND=0,1`, and `NUMACTL_CPUNODEBIND=0,1`.

Reject the final implementation if Asym remains above ZeRO in peak HBM. That
means the offload surface or temporary peak is still not equivalent.
