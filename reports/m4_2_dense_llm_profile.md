# m4_2_dense_llm Source-Label Coverage Report

This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_m4.py` for kernel-busy, memcpy, and GPU no-kernel percentages.

These source-label tables are for auditing range coverage. In `--timing-mode profile`, inner rows are asynchronous Python/NVTX range timings. In `--timing-mode debug_sync`, rows are synchronized debugging timings and carry profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 0.1654 | 0.35% |
| step.forward | 22.4651 | 47.50% |
| step.loss | 0.3172 | 0.67% |
| step.backward | 23.6499 | 50.00% |
| step.optimizer | 0.6991 | 1.48% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 47.2966 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.embeddings | 0.0574 | 0.26% |
| forward.attention.layernorm | 0.6218 | 2.77% |
| forward.attention.q_proj.base_frozen_asymgemm | 0.9785 | 4.36% |
| forward.attention.q_proj.lora_A | 0.5518 | 2.46% |
| forward.attention.q_proj.lora_B | 0.3919 | 1.74% |
| forward.attention.k_proj.base_frozen_asymgemm | 0.9425 | 4.20% |
| forward.attention.k_proj.lora_A | 0.4914 | 2.19% |
| forward.attention.k_proj.lora_B | 0.3855 | 1.72% |
| forward.attention.v_proj.base_frozen_asymgemm | 0.8856 | 3.94% |
| forward.attention.v_proj.lora_A | 0.4867 | 2.17% |
| forward.attention.v_proj.lora_B | 0.3782 | 1.68% |
| forward.attention.scores_matmul | 0.4787 | 2.13% |
| forward.attention.causal_mask | 0.3081 | 1.37% |
| forward.attention.softmax | 0.1609 | 0.72% |
| forward.attention.value_matmul | 0.3949 | 1.76% |
| forward.attention.o_proj.base_frozen_asymgemm | 0.9654 | 4.30% |
| forward.attention.o_proj.lora_A | 0.5165 | 2.30% |
| forward.attention.o_proj.lora_B | 0.3847 | 1.71% |
| forward.attention.residual_add | 0.2521 | 1.12% |
| forward.mlp.layernorm | 0.5747 | 2.56% |
| forward.mlp.gate_proj.base_frozen_asymgemm | 1.0038 | 4.47% |
| forward.mlp.gate_proj.lora_A | 0.5296 | 2.36% |
| forward.mlp.gate_proj.lora_B | 0.3054 | 1.36% |
| forward.mlp.up_proj.base_frozen_asymgemm | 0.9545 | 4.25% |
| forward.mlp.up_proj.lora_A | 0.4997 | 2.22% |
| forward.mlp.up_proj.lora_B | 0.2959 | 1.32% |
| forward.mlp.silu_mul_activation | 0.2439 | 1.09% |
| forward.mlp.down_proj.base_frozen_asymgemm | 0.8738 | 3.89% |
| forward.mlp.down_proj.lora_A | 0.4936 | 2.20% |
| forward.mlp.down_proj.lora_B | 0.3803 | 1.69% |
| forward.mlp.residual_add | 0.2411 | 1.07% |
| forward.final_norm | 0.1432 | 0.64% |
| forward.lm_head | 0.0764 | 0.34% |
| forward.python_dispatch_cuda_launch_and_sync | 6.2167 | 27.67% |
| **sum** | 22.4651 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.cross_entropy.total | 0.1265 | 0.53% |
| backward.base_dx_asymgemm.total | 3.5296 | 14.92% |
| backward.lm_head.total | 0.0726 | 0.31% |
| backward.final_norm.total | 0.0905 | 0.38% |
| backward.mlp.residual_add.total | 0.1552 | 0.66% |
| backward.mlp.down_proj.base_lora_add.total | 0.1385 | 0.59% |
| backward.mlp.down_proj.add_cast_scale.total | 0.1085 | 0.46% |
| backward.mlp.down_proj.lora_B.total | 0.5174 | 2.19% |
| backward.mlp.down_proj.lora_A.total | 0.2796 | 1.18% |
| backward.mlp.silu_mul_activation.total | 0.3594 | 1.52% |
| backward.mlp.up_proj.base_lora_add.total | 0.1472 | 0.62% |
| backward.mlp.up_proj.add_cast_scale.total | 0.1106 | 0.47% |
| backward.mlp.up_proj.lora_B.total | 0.5047 | 2.13% |
| backward.mlp.up_proj.lora_A.total | 0.4378 | 1.85% |
| backward.mlp.gate_proj.base_lora_add.total | 0.1680 | 0.71% |
| backward.mlp.gate_proj.add_cast_scale.total | 0.1144 | 0.48% |
| backward.mlp.gate_proj.lora_B.total | 0.4823 | 2.04% |
| backward.mlp.gate_proj.lora_A.total | 0.4368 | 1.85% |
| backward.mlp.layernorm.total | 0.3572 | 1.51% |
| backward.attention.residual_add.total | 0.1516 | 0.64% |
| backward.attention.o_proj.base_lora_add.total | 0.1373 | 0.58% |
| backward.attention.o_proj.add_cast_scale.total | 0.1074 | 0.45% |
| backward.attention.o_proj.lora_B.total | 0.5048 | 2.13% |
| backward.attention.o_proj.lora_A.total | 0.4359 | 1.84% |
| backward.attention.value_matmul.total | 0.5036 | 2.13% |
| backward.attention.softmax.total | 0.2177 | 0.92% |
| backward.attention.scores_matmul.total | 0.3821 | 1.62% |
| backward.attention.v_proj.base_lora_add.total | 0.1716 | 0.73% |
| backward.attention.v_proj.add_cast_scale.total | 0.1135 | 0.48% |
| backward.attention.v_proj.lora_B.total | 0.4807 | 2.03% |
| backward.attention.v_proj.lora_A.total | 0.4363 | 1.84% |
| backward.attention.k_proj.base_lora_add.total | 0.1783 | 0.75% |
| backward.attention.k_proj.add_cast_scale.total | 0.1165 | 0.49% |
| backward.attention.k_proj.lora_B.total | 0.4935 | 2.09% |
| backward.attention.k_proj.lora_A.total | 0.4375 | 1.85% |
| backward.attention.q_proj.base_lora_add.total | 0.1702 | 0.72% |
| backward.attention.q_proj.add_cast_scale.total | 0.1166 | 0.49% |
| backward.attention.q_proj.lora_B.total | 0.4916 | 2.08% |
| backward.attention.q_proj.lora_A.total | 0.4343 | 1.84% |
| backward.attention.layernorm.total | 0.3625 | 1.53% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 9.0698 | 38.35% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 23.6499 | 100.00% |

## Memory

| Component | bytes |
|---|---:|
| peak_hbm | 89690112 |
| gpu_parameters | 4461056 |
| gpu_buffers | 264192 |
| pinned_W | 2621440 |
| pinned_W_T | 0 |
| pinned_total | 2621440 |
