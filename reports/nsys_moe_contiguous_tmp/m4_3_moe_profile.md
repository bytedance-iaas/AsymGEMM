# m4_3_moe Profile

All timing tables are additive within their parent. Stage residuals are explicitly labeled as Python/autograd dispatch, CUDA launch latency, and synchronization/profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 0.2640 | 0.22% |
| step.forward | 53.8196 | 44.09% |
| step.loss | 0.8471 | 0.69% |
| step.backward | 64.0165 | 52.45% |
| step.optimizer | 3.1141 | 2.55% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 122.0613 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.embeddings | 0.0133 | 0.02% |
| forward.attention.layernorm | 0.9402 | 1.75% |
| forward.attention.q_proj_base | 0.5258 | 0.98% |
| forward.attention.k_proj_base | 0.2690 | 0.50% |
| forward.attention.v_proj_base | 0.2309 | 0.43% |
| forward.attention.scores_matmul | 0.6983 | 1.30% |
| forward.attention.causal_mask | 0.5104 | 0.95% |
| forward.attention.softmax | 0.2121 | 0.39% |
| forward.attention.value_matmul | 0.5264 | 0.98% |
| forward.attention.o_proj_base | 0.3687 | 0.69% |
| forward.attention.residual_add | 0.4174 | 0.78% |
| forward.moe.layernorm | 0.7421 | 1.38% |
| forward.moe.flatten | 0.0708 | 0.13% |
| forward.router | 0.0622 | 0.12% |
| forward.route_metadata | 2.4751 | 4.60% |
| forward.pack_tokens | 0.2835 | 0.53% |
| forward.routed_expert.gate_base_asymgemm | 4.6273 | 8.60% |
| forward.routed_expert.gate_lora_A | 1.8655 | 3.47% |
| forward.routed_expert.gate_lora_B | 1.4412 | 2.68% |
| forward.routed_expert.gate_lora_scale_cast | 0.4639 | 0.86% |
| forward.routed_expert.up_base_asymgemm | 4.1076 | 7.63% |
| forward.routed_expert.up_lora_A | 1.5346 | 2.85% |
| forward.routed_expert.up_lora_B | 1.2282 | 2.28% |
| forward.routed_expert.up_lora_scale_cast | 0.4095 | 0.76% |
| forward.routed_expert.activation_silu_mul | 1.6017 | 2.98% |
| forward.routed_expert.down_base_asymgemm | 3.9240 | 7.29% |
| forward.routed_expert.down_lora_A | 1.5438 | 2.87% |
| forward.routed_expert.down_lora_B | 1.2687 | 2.36% |
| forward.routed_expert.down_lora_scale_cast | 0.4376 | 0.81% |
| forward.routed_expert.add | 1.4491 | 2.69% |
| forward.scatter_combine | 0.4075 | 0.76% |
| forward.shared_expert.gate_base_asymgemm | 1.0156 | 1.89% |
| forward.shared_expert.gate_lora_A | 0.4012 | 0.75% |
| forward.shared_expert.gate_lora_B | 0.3416 | 0.63% |
| forward.shared_expert.gate_lora_scale_cast | 0.1093 | 0.20% |
| forward.shared_expert.up_base_asymgemm | 0.9350 | 1.74% |
| forward.shared_expert.up_lora_A | 0.3443 | 0.64% |
| forward.shared_expert.up_lora_B | 0.3101 | 0.58% |
| forward.shared_expert.up_lora_scale_cast | 0.1233 | 0.23% |
| forward.shared_expert.activation_silu_mul | 0.5016 | 0.93% |
| forward.shared_expert.down_base_asymgemm | 1.1689 | 2.17% |
| forward.shared_expert.down_lora_A | 0.4593 | 0.85% |
| forward.shared_expert.down_lora_B | 0.3591 | 0.67% |
| forward.shared_expert.down_lora_scale_cast | 0.1213 | 0.23% |
| forward.shared_expert.add | 0.4107 | 0.76% |
| forward.moe.combine_shared_routed | 0.2781 | 0.52% |
| forward.moe.residual_add | 0.3598 | 0.67% |
| forward.final_norm | 0.1953 | 0.36% |
| forward.python_dispatch_cuda_launch_and_sync | 11.7286 | 21.79% |
| **sum** | 53.8196 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.mse.total | 0.1743 | 0.27% |
| backward.base_dx_asymgemm.total | 12.3341 | 19.27% |
| backward.final_norm.total | 0.1610 | 0.25% |
| backward.moe.residual_add.total | 0.2530 | 0.40% |
| backward.moe.combine_shared_routed.total | 0.1092 | 0.17% |
| backward.scatter_combine.total | 0.3392 | 0.53% |
| backward.pack_tokens.total | 0.4334 | 0.68% |
| backward.route_metadata.total | 0.0000 | 0.00% |
| backward.router.total | 0.0000 | 0.00% |
| backward.routed_expert.down_base_lora_add.total | 1.0833 | 1.69% |
| backward.routed_expert.down_lora_B.total | 1.8923 | 2.96% |
| backward.routed_expert.down_lora_A.total | 1.5005 | 2.34% |
| backward.routed_expert.activation_silu_mul.total | 2.1840 | 3.41% |
| backward.routed_expert.up_base_lora_add.total | 0.9276 | 1.45% |
| backward.routed_expert.up_lora_B.total | 1.9107 | 2.98% |
| backward.routed_expert.up_lora_A.total | 1.3090 | 2.04% |
| backward.routed_expert.gate_base_lora_add.total | 1.0501 | 1.64% |
| backward.routed_expert.gate_lora_B.total | 1.7959 | 2.81% |
| backward.routed_expert.gate_lora_A.total | 1.2915 | 2.02% |
| backward.shared_expert.down_base_lora_add.total | 0.2279 | 0.36% |
| backward.shared_expert.down_lora_B.total | 0.5246 | 0.82% |
| backward.shared_expert.down_lora_A.total | 0.3883 | 0.61% |
| backward.shared_expert.activation_silu_mul.total | 0.5316 | 0.83% |
| backward.shared_expert.up_base_lora_add.total | 0.1978 | 0.31% |
| backward.shared_expert.up_lora_B.total | 0.5030 | 0.79% |
| backward.shared_expert.up_lora_A.total | 0.3948 | 0.62% |
| backward.shared_expert.gate_base_lora_add.total | 0.2702 | 0.42% |
| backward.shared_expert.gate_lora_B.total | 0.5306 | 0.83% |
| backward.shared_expert.gate_lora_A.total | 0.3406 | 0.53% |
| backward.moe.layernorm.total | 0.6518 | 1.02% |
| backward.attention.residual_add.total | 0.2459 | 0.38% |
| backward.attention.o_proj_base.total | 0.3520 | 0.55% |
| backward.attention.value_matmul.total | 0.5921 | 0.92% |
| backward.attention.softmax.total | 0.3851 | 0.60% |
| backward.attention.scores_matmul.total | 0.4782 | 0.75% |
| backward.attention.v_proj_base.total | 0.2645 | 0.41% |
| backward.attention.k_proj_base.total | 0.1838 | 0.29% |
| backward.attention.q_proj_base.total | 0.1749 | 0.27% |
| backward.attention.layernorm.total | 0.6070 | 0.95% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 27.4231 | 42.84% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 64.0165 | 100.00% |

## Memory

| Component | bytes |
|---|---:|
| peak_hbm | 126937600 |
| gpu_parameters | 11804672 |
| gpu_buffers | 799744 |
| pinned_W | 3932160 |
| pinned_W_T | 0 |
| pinned_total | 3932160 |
