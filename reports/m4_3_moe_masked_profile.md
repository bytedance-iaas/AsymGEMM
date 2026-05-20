# m4_3_moe Source-Label Coverage Report

This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_m4.py` for kernel-busy, memcpy, and GPU no-kernel percentages.

These source-label tables are for auditing range coverage. In `--timing-mode profile`, inner rows are asynchronous Python/NVTX range timings. In `--timing-mode debug_sync`, rows are synchronized debugging timings and carry profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 0.1546 | 0.14% |
| step.forward | 50.0710 | 46.86% |
| step.loss | 0.8705 | 0.81% |
| step.backward | 54.1511 | 50.68% |
| step.optimizer | 1.6026 | 1.50% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 106.8497 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.embeddings | 0.0188 | 0.04% |
| forward.attention.layernorm | 0.6165 | 1.23% |
| forward.attention.q_proj_base | 0.4189 | 0.84% |
| forward.attention.k_proj_base | 0.2102 | 0.42% |
| forward.attention.v_proj_base | 0.1791 | 0.36% |
| forward.attention.scores_matmul | 0.5064 | 1.01% |
| forward.attention.causal_mask | 0.3433 | 0.69% |
| forward.attention.softmax | 0.1691 | 0.34% |
| forward.attention.value_matmul | 0.4057 | 0.81% |
| forward.attention.o_proj_base | 0.2963 | 0.59% |
| forward.attention.residual_add | 0.3040 | 0.61% |
| forward.moe.layernorm | 0.5234 | 1.05% |
| forward.moe.flatten | 0.0768 | 0.15% |
| forward.router | 0.0675 | 0.13% |
| forward.route_metadata | 2.1044 | 4.20% |
| forward.pack_tokens | 0.3791 | 0.76% |
| forward.routed_expert.gate_base_asymgemm | 4.2609 | 8.51% |
| forward.routed_expert.gate_lora_A | 2.0526 | 4.10% |
| forward.routed_expert.gate_lora_B | 1.1784 | 2.35% |
| forward.routed_expert.gate_lora_scale_cast | 0.4028 | 0.80% |
| forward.routed_expert.up_base_asymgemm | 4.1595 | 8.31% |
| forward.routed_expert.up_lora_A | 1.8748 | 3.74% |
| forward.routed_expert.up_lora_B | 1.1626 | 2.32% |
| forward.routed_expert.up_lora_scale_cast | 0.4072 | 0.81% |
| forward.routed_expert.activation_silu_mul | 1.1566 | 2.31% |
| forward.routed_expert.down_base_asymgemm | 3.4404 | 6.87% |
| forward.routed_expert.down_lora_A | 1.8761 | 3.75% |
| forward.routed_expert.down_lora_B | 1.4997 | 3.00% |
| forward.routed_expert.down_lora_scale_cast | 0.4140 | 0.83% |
| forward.routed_expert.add | 1.0488 | 2.09% |
| forward.scatter_combine | 0.7638 | 1.53% |
| forward.shared_expert.gate_base_asymgemm | 1.0025 | 2.00% |
| forward.shared_expert.gate_lora_A | 0.5281 | 1.05% |
| forward.shared_expert.gate_lora_B | 0.2952 | 0.59% |
| forward.shared_expert.gate_lora_scale_cast | 0.1037 | 0.21% |
| forward.shared_expert.up_base_asymgemm | 0.9322 | 1.86% |
| forward.shared_expert.up_lora_A | 0.4628 | 0.92% |
| forward.shared_expert.up_lora_B | 0.2852 | 0.57% |
| forward.shared_expert.up_lora_scale_cast | 0.0996 | 0.20% |
| forward.shared_expert.activation_silu_mul | 0.2843 | 0.57% |
| forward.shared_expert.down_base_asymgemm | 0.8573 | 1.71% |
| forward.shared_expert.down_lora_A | 0.4653 | 0.93% |
| forward.shared_expert.down_lora_B | 0.3774 | 0.75% |
| forward.shared_expert.down_lora_scale_cast | 0.0973 | 0.19% |
| forward.shared_expert.add | 0.2611 | 0.52% |
| forward.moe.combine_shared_routed | 0.2035 | 0.41% |
| forward.moe.residual_add | 0.2236 | 0.45% |
| forward.final_norm | 0.1360 | 0.27% |
| forward.python_dispatch_cuda_launch_and_sync | 11.1382 | 22.24% |
| **sum** | 50.0710 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.mse.total | 0.1195 | 0.22% |
| backward.base_dx_asymgemm.total | 8.1642 | 15.08% |
| backward.final_norm.total | 0.1175 | 0.22% |
| backward.moe.residual_add.total | 0.1876 | 0.35% |
| backward.moe.combine_shared_routed.total | 0.1075 | 0.20% |
| backward.scatter_combine.total | 0.9624 | 1.78% |
| backward.pack_tokens.total | 0.7385 | 1.36% |
| backward.route_metadata.total | 0.0000 | 0.00% |
| backward.router.total | 0.0000 | 0.00% |
| backward.routed_expert.down_base_lora_add.total | 0.7811 | 1.44% |
| backward.routed_expert.down_lora_B.total | 2.1573 | 3.98% |
| backward.routed_expert.down_lora_A.total | 1.2163 | 2.25% |
| backward.routed_expert.activation_silu_mul.total | 1.5847 | 2.93% |
| backward.routed_expert.up_base_lora_add.total | 0.6629 | 1.22% |
| backward.routed_expert.up_lora_B.total | 2.1458 | 3.96% |
| backward.routed_expert.up_lora_A.total | 1.8619 | 3.44% |
| backward.routed_expert.gate_base_lora_add.total | 0.7569 | 1.40% |
| backward.routed_expert.gate_lora_B.total | 2.0563 | 3.80% |
| backward.routed_expert.gate_lora_A.total | 1.8324 | 3.38% |
| backward.shared_expert.down_base_lora_add.total | 0.1643 | 0.30% |
| backward.shared_expert.down_lora_B.total | 0.5877 | 1.09% |
| backward.shared_expert.down_lora_A.total | 0.3176 | 0.59% |
| backward.shared_expert.activation_silu_mul.total | 0.4033 | 0.74% |
| backward.shared_expert.up_base_lora_add.total | 0.1637 | 0.30% |
| backward.shared_expert.up_lora_B.total | 0.5599 | 1.03% |
| backward.shared_expert.up_lora_A.total | 0.4756 | 0.88% |
| backward.shared_expert.gate_base_lora_add.total | 0.1964 | 0.36% |
| backward.shared_expert.gate_lora_B.total | 0.5284 | 0.98% |
| backward.shared_expert.gate_lora_A.total | 0.4681 | 0.86% |
| backward.moe.layernorm.total | 0.4346 | 0.80% |
| backward.attention.residual_add.total | 0.1796 | 0.33% |
| backward.attention.o_proj_base.total | 0.2548 | 0.47% |
| backward.attention.value_matmul.total | 0.4624 | 0.85% |
| backward.attention.softmax.total | 0.2394 | 0.44% |
| backward.attention.scores_matmul.total | 0.3549 | 0.66% |
| backward.attention.v_proj_base.total | 0.1952 | 0.36% |
| backward.attention.k_proj_base.total | 0.1401 | 0.26% |
| backward.attention.q_proj_base.total | 0.1254 | 0.23% |
| backward.attention.layernorm.total | 0.3900 | 0.72% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 22.0569 | 40.73% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 54.1511 | 100.00% |

## Memory

| Component | bytes |
|---|---:|
| peak_hbm | 126937600 |
| gpu_parameters | 11804672 |
| gpu_buffers | 799744 |
| pinned_W | 3932160 |
| pinned_W_T | 0 |
| pinned_total | 3932160 |
