# m4_2_dense_llm Source-Label Coverage Report

This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_m4.py` for kernel-busy, memcpy, and GPU no-kernel percentages.

These source-label tables are for auditing range coverage. In `--timing-mode profile`, inner rows are asynchronous Python/NVTX range timings. In `--timing-mode debug_sync`, rows are synchronized debugging timings and carry profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 0.3732 | 0.64% |
| step.forward | 20.4714 | 35.21% |
| step.loss | 0.5055 | 0.87% |
| step.backward | 35.2331 | 60.61% |
| step.optimizer | 1.5500 | 2.67% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 58.1332 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.embeddings | 0.1294 | 0.63% |
| forward.attention.layernorm | 0.7089 | 3.46% |
| forward.attention.q_proj.base_frozen_asymgemm | 1.0499 | 5.13% |
| forward.attention.q_proj.lora_A | 0.4786 | 2.34% |
| forward.attention.q_proj.lora_B | 0.1964 | 0.96% |
| forward.attention.k_proj.base_frozen_asymgemm | 0.8277 | 4.04% |
| forward.attention.k_proj.lora_A | 0.2956 | 1.44% |
| forward.attention.k_proj.lora_B | 0.1857 | 0.91% |
| forward.attention.v_proj.base_frozen_asymgemm | 0.8921 | 4.36% |
| forward.attention.v_proj.lora_A | 0.3238 | 1.58% |
| forward.attention.v_proj.lora_B | 0.1844 | 0.90% |
| forward.attention.scores_matmul | 0.5059 | 2.47% |
| forward.attention.causal_mask | 0.3542 | 1.73% |
| forward.attention.softmax | 0.1627 | 0.79% |
| forward.attention.value_matmul | 0.3401 | 1.66% |
| forward.attention.o_proj.base_frozen_asymgemm | 1.1757 | 5.74% |
| forward.attention.o_proj.lora_A | 0.4499 | 2.20% |
| forward.attention.o_proj.lora_B | 0.2216 | 1.08% |
| forward.attention.residual_add | 0.2461 | 1.20% |
| forward.mlp.layernorm | 0.6750 | 3.30% |
| forward.mlp.gate_proj.base_frozen_asymgemm | 1.0418 | 5.09% |
| forward.mlp.gate_proj.lora_A | 0.4152 | 2.03% |
| forward.mlp.gate_proj.lora_B | 0.2182 | 1.07% |
| forward.mlp.up_proj.base_frozen_asymgemm | 0.9254 | 4.52% |
| forward.mlp.up_proj.lora_A | 0.3413 | 1.67% |
| forward.mlp.up_proj.lora_B | 0.2141 | 1.05% |
| forward.mlp.silu_mul_activation | 0.2681 | 1.31% |
| forward.mlp.down_proj.base_frozen_asymgemm | 0.9512 | 4.65% |
| forward.mlp.down_proj.lora_A | 0.3780 | 1.85% |
| forward.mlp.down_proj.lora_B | 0.2474 | 1.21% |
| forward.mlp.residual_add | 0.2186 | 1.07% |
| forward.final_norm | 0.1416 | 0.69% |
| forward.lm_head | 0.0994 | 0.49% |
| forward.python_dispatch_cuda_launch_and_sync | 5.6073 | 27.39% |
| **sum** | 20.4714 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.cross_entropy.total | 0.3122 | 0.89% |
| backward.lm_head.total | 0.1026 | 0.29% |
| backward.final_norm.total | 0.0994 | 0.28% |
| backward.mlp.residual_add.total | 0.2185 | 0.62% |
| backward.mlp.down_proj.base_dx_asymgemm.total | 0.0000 | 0.00% |
| backward.mlp.down_proj.base_lora_add.total | 0.1640 | 0.47% |
| backward.mlp.down_proj.add_cast_scale.total | 0.1177 | 0.33% |
| backward.mlp.down_proj.lora_B.total | 0.5178 | 1.47% |
| backward.mlp.down_proj.lora_A.total | 0.4344 | 1.23% |
| backward.mlp.silu_mul_activation.total | 0.5543 | 1.57% |
| backward.mlp.up_proj.base_dx_asymgemm.total | 0.0000 | 0.00% |
| backward.mlp.up_proj.base_lora_add.total | 0.2065 | 0.59% |
| backward.mlp.up_proj.add_cast_scale.total | 0.1339 | 0.38% |
| backward.mlp.up_proj.lora_B.total | 0.5737 | 1.63% |
| backward.mlp.up_proj.lora_A.total | 0.3837 | 1.09% |
| backward.mlp.gate_proj.base_dx_asymgemm.total | 0.0000 | 0.00% |
| backward.mlp.gate_proj.base_lora_add.total | 0.2819 | 0.80% |
| backward.mlp.gate_proj.add_cast_scale.total | 0.1692 | 0.48% |
| backward.mlp.gate_proj.lora_B.total | 0.5778 | 1.64% |
| backward.mlp.gate_proj.lora_A.total | 0.3961 | 1.12% |
| backward.mlp.layernorm.total | 0.5860 | 1.66% |
| backward.attention.residual_add.total | 0.2474 | 0.70% |
| backward.attention.o_proj.base_dx_asymgemm.total | 0.0000 | 0.00% |
| backward.attention.o_proj.base_lora_add.total | 0.1883 | 0.53% |
| backward.attention.o_proj.add_cast_scale.total | 0.1466 | 0.42% |
| backward.attention.o_proj.lora_B.total | 0.5713 | 1.62% |
| backward.attention.o_proj.lora_A.total | 0.3822 | 1.08% |
| backward.attention.value_matmul.total | 0.9391 | 2.67% |
| backward.attention.softmax.total | 0.3663 | 1.04% |
| backward.attention.scores_matmul.total | 0.7747 | 2.20% |
| backward.attention.v_proj.base_dx_asymgemm.total | 0.0000 | 0.00% |
| backward.attention.v_proj.base_lora_add.total | 0.2817 | 0.80% |
| backward.attention.v_proj.add_cast_scale.total | 0.1730 | 0.49% |
| backward.attention.v_proj.lora_B.total | 0.5072 | 1.44% |
| backward.attention.v_proj.lora_A.total | 0.3927 | 1.11% |
| backward.attention.k_proj.base_dx_asymgemm.total | 0.0000 | 0.00% |
| backward.attention.k_proj.base_lora_add.total | 0.3031 | 0.86% |
| backward.attention.k_proj.add_cast_scale.total | 0.1474 | 0.42% |
| backward.attention.k_proj.lora_B.total | 0.5226 | 1.48% |
| backward.attention.k_proj.lora_A.total | 0.3611 | 1.02% |
| backward.attention.q_proj.base_dx_asymgemm.total | 0.0000 | 0.00% |
| backward.attention.q_proj.base_lora_add.total | 0.2661 | 0.76% |
| backward.attention.q_proj.add_cast_scale.total | 0.1417 | 0.40% |
| backward.attention.q_proj.lora_B.total | 0.4828 | 1.37% |
| backward.attention.q_proj.lora_A.total | 0.3319 | 0.94% |
| backward.attention.layernorm.total | 0.5361 | 1.52% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 21.3401 | 60.57% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 35.2331 | 100.00% |

## Memory

| Component | bytes |
|---|---:|
| peak_hbm | 89690112 |
| gpu_parameters | 4461056 |
| gpu_buffers | 264192 |
| pinned_W | 2621440 |
| pinned_W_T | 0 |
| pinned_total | 2621440 |
