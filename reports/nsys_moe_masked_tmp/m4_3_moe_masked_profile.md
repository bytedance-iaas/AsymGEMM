# m4_3_moe Profile

All timing tables are additive within their parent. Stage residuals are explicitly labeled as Python/autograd dispatch, CUDA launch latency, and synchronization/profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 0.2315 | 0.18% |
| step.forward | 58.1815 | 46.08% |
| step.loss | 0.9455 | 0.75% |
| step.backward | 64.8771 | 51.38% |
| step.optimizer | 2.0314 | 1.61% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 126.2670 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.embeddings | 0.0136 | 0.02% |
| forward.attention.layernorm | 0.8863 | 1.52% |
| forward.attention.q_proj_base | 0.5286 | 0.91% |
| forward.attention.k_proj_base | 0.2749 | 0.47% |
| forward.attention.v_proj_base | 0.2023 | 0.35% |
| forward.attention.scores_matmul | 0.8492 | 1.46% |
| forward.attention.causal_mask | 0.5676 | 0.98% |
| forward.attention.softmax | 0.2323 | 0.40% |
| forward.attention.value_matmul | 0.5238 | 0.90% |
| forward.attention.o_proj_base | 0.3661 | 0.63% |
| forward.attention.residual_add | 0.3903 | 0.67% |
| forward.moe.layernorm | 0.7497 | 1.29% |
| forward.moe.flatten | 0.0684 | 0.12% |
| forward.router | 0.0629 | 0.11% |
| forward.route_metadata | 3.7860 | 6.51% |
| forward.pack_tokens | 0.5166 | 0.89% |
| forward.routed_expert.gate_base_asymgemm | 4.9166 | 8.45% |
| forward.routed_expert.gate_lora_A | 1.8876 | 3.24% |
| forward.routed_expert.gate_lora_B | 1.4035 | 2.41% |
| forward.routed_expert.gate_lora_scale_cast | 0.4593 | 0.79% |
| forward.routed_expert.up_base_asymgemm | 4.2094 | 7.23% |
| forward.routed_expert.up_lora_A | 1.6564 | 2.85% |
| forward.routed_expert.up_lora_B | 1.5277 | 2.63% |
| forward.routed_expert.up_lora_scale_cast | 0.4916 | 0.84% |
| forward.routed_expert.activation_silu_mul | 1.6994 | 2.92% |
| forward.routed_expert.down_base_asymgemm | 4.2332 | 7.28% |
| forward.routed_expert.down_lora_A | 1.7577 | 3.02% |
| forward.routed_expert.down_lora_B | 1.4453 | 2.48% |
| forward.routed_expert.down_lora_scale_cast | 0.4732 | 0.81% |
| forward.routed_expert.add | 1.4514 | 2.49% |
| forward.scatter_combine | 1.2497 | 2.15% |
| forward.shared_expert.gate_base_asymgemm | 1.0832 | 1.86% |
| forward.shared_expert.gate_lora_A | 0.4938 | 0.85% |
| forward.shared_expert.gate_lora_B | 0.3460 | 0.59% |
| forward.shared_expert.gate_lora_scale_cast | 0.1205 | 0.21% |
| forward.shared_expert.up_base_asymgemm | 1.0462 | 1.80% |
| forward.shared_expert.up_lora_A | 0.3885 | 0.67% |
| forward.shared_expert.up_lora_B | 0.3108 | 0.53% |
| forward.shared_expert.up_lora_scale_cast | 0.1051 | 0.18% |
| forward.shared_expert.activation_silu_mul | 0.4073 | 0.70% |
| forward.shared_expert.down_base_asymgemm | 1.0090 | 1.73% |
| forward.shared_expert.down_lora_A | 0.3907 | 0.67% |
| forward.shared_expert.down_lora_B | 0.3321 | 0.57% |
| forward.shared_expert.down_lora_scale_cast | 0.1096 | 0.19% |
| forward.shared_expert.add | 0.3763 | 0.65% |
| forward.moe.combine_shared_routed | 0.2572 | 0.44% |
| forward.moe.residual_add | 0.2958 | 0.51% |
| forward.final_norm | 0.1931 | 0.33% |
| forward.python_dispatch_cuda_launch_and_sync | 12.0358 | 20.69% |
| **sum** | 58.1815 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.mse.total | 0.2284 | 0.35% |
| backward.base_dx_asymgemm.total | 11.5525 | 17.81% |
| backward.final_norm.total | 0.2037 | 0.31% |
| backward.moe.residual_add.total | 0.2623 | 0.40% |
| backward.moe.combine_shared_routed.total | 0.1167 | 0.18% |
| backward.scatter_combine.total | 1.5131 | 2.33% |
| backward.pack_tokens.total | 1.1826 | 1.82% |
| backward.route_metadata.total | 0.0000 | 0.00% |
| backward.router.total | 0.0000 | 0.00% |
| backward.routed_expert.down_base_lora_add.total | 1.1809 | 1.82% |
| backward.routed_expert.down_lora_B.total | 1.9965 | 3.08% |
| backward.routed_expert.down_lora_A.total | 1.5441 | 2.38% |
| backward.routed_expert.activation_silu_mul.total | 2.1720 | 3.35% |
| backward.routed_expert.up_base_lora_add.total | 0.9497 | 1.46% |
| backward.routed_expert.up_lora_B.total | 1.8499 | 2.85% |
| backward.routed_expert.up_lora_A.total | 1.3146 | 2.03% |
| backward.routed_expert.gate_base_lora_add.total | 1.0037 | 1.55% |
| backward.routed_expert.gate_lora_B.total | 1.6663 | 2.57% |
| backward.routed_expert.gate_lora_A.total | 1.3159 | 2.03% |
| backward.shared_expert.down_base_lora_add.total | 0.2232 | 0.34% |
| backward.shared_expert.down_lora_B.total | 0.5239 | 0.81% |
| backward.shared_expert.down_lora_A.total | 0.4134 | 0.64% |
| backward.shared_expert.activation_silu_mul.total | 0.5774 | 0.89% |
| backward.shared_expert.up_base_lora_add.total | 0.2361 | 0.36% |
| backward.shared_expert.up_lora_B.total | 0.5231 | 0.81% |
| backward.shared_expert.up_lora_A.total | 0.3758 | 0.58% |
| backward.shared_expert.gate_base_lora_add.total | 0.2970 | 0.46% |
| backward.shared_expert.gate_lora_B.total | 0.5162 | 0.80% |
| backward.shared_expert.gate_lora_A.total | 0.3235 | 0.50% |
| backward.moe.layernorm.total | 0.6407 | 0.99% |
| backward.attention.residual_add.total | 0.2371 | 0.37% |
| backward.attention.o_proj_base.total | 0.3626 | 0.56% |
| backward.attention.value_matmul.total | 0.6045 | 0.93% |
| backward.attention.softmax.total | 0.3349 | 0.52% |
| backward.attention.scores_matmul.total | 0.4185 | 0.65% |
| backward.attention.v_proj_base.total | 0.2336 | 0.36% |
| backward.attention.k_proj_base.total | 0.1660 | 0.26% |
| backward.attention.q_proj_base.total | 0.1445 | 0.22% |
| backward.attention.layernorm.total | 0.5528 | 0.85% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 27.1194 | 41.80% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 64.8771 | 100.00% |

## Memory

| Component | bytes |
|---|---:|
| peak_hbm | 126937600 |
| gpu_parameters | 11804672 |
| gpu_buffers | 799744 |
| pinned_W | 3932160 |
| pinned_W_T | 0 |
| pinned_total | 3932160 |
