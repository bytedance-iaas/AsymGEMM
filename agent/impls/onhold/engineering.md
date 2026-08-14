# AsymGEMM MoE LoRA SFT Fusion Notes

Goal: improve MoE LoRA SFT by reducing duplicate gate/up base GEMMs, reducing activation memory traffic, and keeping LoRA gradients correct.

## Key Constraint

For LoRA SFT, LoRA must be added before the SwiGLU nonlinearity:

```python
gate = gate_base + scale * gate_lora
up = up_base + scale * up_lora
hidden = silu(gate) * up
```

It is mathematically wrong to compute `silu(gate_base) * up_base` first and add LoRA later.

## Current Best Design For Us

1. Packed frozen `gate_up` base
   - Use one AsymGEMM call:
     ```python
     base_pair = AsymGEMM(X, W_gate_up_frozen)  # [T, 2I]
     ```
   - Avoids separate `gate_base` and `up_base` launches.
   - Reference: DeepGEMM MegaMoE, NeMo AutoModel `GroupedExpertsLoRA`.

2. Packed LoRA `gate_up`
   - Use one LoRA adapter pair for gate/up:
     ```python
     lora_pair = grouped_lora(X, A_gate_up, B_gate_up)  # [T, 2I]
     ```
   - Split only after base and LoRA are added.
   - Reference: NeMo AutoModel `lora_gate_and_up_A/B`.

3. Fused `base + LoRA + SwiGLU`
   - Replace separate add/chunk/activation elementwise ops with a custom training op:
     ```python
     hidden = fused_add_swiglu(base_pair, lora_pair, scale)
     ```
   - Computes:
     ```python
     hidden = silu(base_gate + scale * lora_gate) * (base_up + scale * lora_up)
     ```
   - This is the biggest practical next win after packed `gate_up`.
   - Reference: Axolotl LoRA MLP optimization kernels.

4. Custom autograd / recompute
   - Backward must produce gradients for LoRA A/B and activation inputs.
   - Frozen base weights do not need gradients.
   - To save memory, prefer recomputing gate/up pieces in backward rather than saving all intermediates.
   - Reference: Axolotl LoRA optimized kernels.

5. Packed/fused down LoRA
   - Down path should follow the same structure:
     ```python
     out_base = AsymGEMM(hidden, W_down_frozen)
     out_lora = grouped_lora(hidden, A_down, B_down)
     out = out_base + scale * out_lora
     ```
   - Reference: NeMo `GroupedExpertsDeepEPLoRA`, Axolotl ScatterMoE/SonicMoE.

6. Later: deeper routed MoE LoRA fusion
   - More aggressive target: fuse LoRA computation into the grouped/routed expert kernel schedule itself.
   - This is harder than fused add/SwiGLU because it couples routing, grouped GEMM, LoRA A/B matmuls, activation, and backward.
   - Reference: Axolotl ScatterMoE/SonicMoE.

## Practical Roadmap

1. Keep the existing packed AsymGEMM frozen `gate_up` path.
2. Ensure LoRA gate/up is packed as one `[T, 2I]` path.
3. Implement `fused_add_swiglu(base_pair, lora_pair, scale)` with custom autograd.
4. Add recompute in backward to reduce saved activation memory.
5. Apply the same packed LoRA structure to down projection.
6. Only after that, consider ScatterMoE-style deeper fusion inside routed grouped expert kernels.

## What Not To Do First

Do not start with one giant AsymGEMM kernel that includes base GEMM, LoRA A, LoRA B, SwiGLU, down projection, and full backward.

The better engineering sequence is:

```python
base_pair = AsymGEMM(X, W_gate_up_frozen)
lora_pair = grouped_lora(X, A_gate_up, B_gate_up)
hidden = fused_add_swiglu(base_pair, lora_pair, scale)
out = AsymGEMM(hidden, W_down_frozen) + grouped_lora(hidden, A_down, B_down)
```

This keeps correctness for LoRA gradients while still capturing the largest near-term launch and memory savings.

## References

### NeMo AutoModel MoE LoRA

Use as the clean training design reference for packed `gate_and_up` LoRA:

- Docs: https://docs.nvidia.com/nemo/automodel/0.3.0/apidocs/nemo_automodel/nemo_automodel.components._peft.lora_moe.html
- Repo: https://github.com/NVIDIA-NeMo/Automodel

Relevant implementation ideas:

- `GroupedExpertsLoRA`
- `GroupedExpertsDeepEPLoRA`
- `swiglu_with_lora`
- `lora_gate_and_up_A`
- `lora_gate_and_up_B`
- `lora_down_A`
- `lora_down_B`

### Axolotl LoRA Optimization Kernels

Use as the main reference for custom autograd and fused LoRA MLP kernels:

- LoRA optimization docs: https://docs.axolotl.ai/docs/lora_optims.html
- Kernel API docs: https://docs.axolotl.ai/docs/api/kernels.lora.html
- Repo: https://github.com/axolotl-ai-cloud/axolotl

Relevant implementation ideas:

- fused LoRA MLP custom autograd
- fused activation kernels for SwiGLU/GEGLU-style MLPs
- backward recompute to reduce saved activation memory

### Axolotl ScatterMoE / SonicMoE

Use as the main public reference for deeper MoE LoRA fusion:

- Custom integrations docs: https://docs.axolotl.ai/docs/custom_integrations.html
- Repo: https://github.com/axolotl-ai-cloud/axolotl

Relevant implementation ideas:

- ScatterMoE: Triton routed MoE kernels with fused LoRA support
- SonicMoE: CUTLASS-style path with effective/fused LoRA weights
- Useful as later-stage reference after our fused add/SwiGLU op works

### DeepGEMM MegaMoE

Use only as a future reference for true fused MoE gate/up/SwiGLU/down epilogue design, not as the first LoRA SFT implementation target:

- Kernel: https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh
- Repo: https://github.com/deepseek-ai/DeepGEMM

Relevant implementation ideas:

- L1 `gate_up` uses `N = 2 * intermediate`
- gate/up are interleaved so the epilogue can see both values
- epilogue computes `silu(gate) * up`
- stores `[T, I]`, not `[T, 2I]`

### vLLM Fused MoE LoRA

Use as an inference-serving reference only, not a training-gradient reference:

- Docs: https://docs.vllm.ai/en/latest/api/vllm/lora/ops/triton_ops/fused_moe_lora_op/
- Repo: https://github.com/vllm-project/vllm

Relevant implementation ideas:

- fused routed LoRA shrink/expand for MoE serving
- useful for multi-LoRA inference, less directly useful for SFT backward
