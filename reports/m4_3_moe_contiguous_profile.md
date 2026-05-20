# m4_3_moe Source-Label Coverage Report

This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_m4.py` for kernel-busy, memcpy, and GPU no-kernel percentages.

These source-label tables are for auditing range coverage. In `--timing-mode profile`, inner rows are asynchronous Python/NVTX range timings. In `--timing-mode debug_sync`, rows are synchronized debugging timings and carry profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 0.1230 | 0.12% |
| step.forward | 48.4692 | 49.22% |
| step.loss | 0.4237 | 0.43% |
| step.backward | 48.2496 | 49.00% |
| step.optimizer | 1.2097 | 1.23% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 98.4752 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.embeddings | 0.0172 | 0.04% |
| forward.attention.layernorm | 0.5694 | 1.17% |
| forward.attention.q_proj_base | 0.3575 | 0.74% |
| forward.attention.k_proj_base | 0.2027 | 0.42% |
| forward.attention.v_proj_base | 0.1708 | 0.35% |
| forward.attention.scores_matmul | 0.4724 | 0.97% |
| forward.attention.causal_mask | 0.3226 | 0.67% |
| forward.attention.softmax | 0.1601 | 0.33% |
| forward.attention.value_matmul | 0.3807 | 0.79% |
| forward.attention.o_proj_base | 0.2784 | 0.57% |
| forward.attention.residual_add | 0.2999 | 0.62% |
| forward.moe.layernorm | 0.5116 | 1.06% |
| forward.moe.flatten | 0.0778 | 0.16% |
| forward.router | 0.0662 | 0.14% |
| forward.route_metadata | 1.4625 | 3.02% |
| forward.pack_tokens | 0.2485 | 0.51% |
| forward.routed_expert.gate_base_asymgemm | 4.2671 | 8.80% |
| forward.routed_expert.gate_lora_A | 2.0261 | 4.18% |
| forward.routed_expert.gate_lora_B | 1.1653 | 2.40% |
| forward.routed_expert.gate_lora_scale_cast | 0.4001 | 0.83% |
| forward.routed_expert.up_base_asymgemm | 3.9891 | 8.23% |
| forward.routed_expert.up_lora_A | 1.8356 | 3.79% |
| forward.routed_expert.up_lora_B | 1.1357 | 2.34% |
| forward.routed_expert.up_lora_scale_cast | 0.3858 | 0.80% |
| forward.routed_expert.activation_silu_mul | 1.0975 | 2.26% |
| forward.routed_expert.down_base_asymgemm | 3.4256 | 7.07% |
| forward.routed_expert.down_lora_A | 1.8718 | 3.86% |
| forward.routed_expert.down_lora_B | 1.4771 | 3.05% |
| forward.routed_expert.down_lora_scale_cast | 0.3917 | 0.81% |
| forward.routed_expert.add | 1.0409 | 2.15% |
| forward.scatter_combine | 0.3080 | 0.64% |
| forward.shared_expert.gate_base_asymgemm | 0.9952 | 2.05% |
| forward.shared_expert.gate_lora_A | 0.4950 | 1.02% |
| forward.shared_expert.gate_lora_B | 0.2884 | 0.60% |
| forward.shared_expert.gate_lora_scale_cast | 0.0968 | 0.20% |
| forward.shared_expert.up_base_asymgemm | 0.9775 | 2.02% |
| forward.shared_expert.up_lora_A | 0.4558 | 0.94% |
| forward.shared_expert.up_lora_B | 0.2813 | 0.58% |
| forward.shared_expert.up_lora_scale_cast | 0.0958 | 0.20% |
| forward.shared_expert.activation_silu_mul | 0.2720 | 0.56% |
| forward.shared_expert.down_base_asymgemm | 0.8604 | 1.78% |
| forward.shared_expert.down_lora_A | 0.4640 | 0.96% |
| forward.shared_expert.down_lora_B | 0.3727 | 0.77% |
| forward.shared_expert.down_lora_scale_cast | 0.0972 | 0.20% |
| forward.shared_expert.add | 0.2592 | 0.53% |
| forward.moe.combine_shared_routed | 0.2022 | 0.42% |
| forward.moe.residual_add | 0.2254 | 0.46% |
| forward.final_norm | 0.1337 | 0.28% |
| forward.python_dispatch_cuda_launch_and_sync | 11.4809 | 23.69% |
| **sum** | 48.4692 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.mse.total | 0.0796 | 0.17% |
| backward.base_dx_asymgemm.total | 7.4889 | 15.52% |
| backward.final_norm.total | 0.0962 | 0.20% |
| backward.moe.residual_add.total | 0.1686 | 0.35% |
| backward.moe.combine_shared_routed.total | 0.0928 | 0.19% |
| backward.scatter_combine.total | 0.2238 | 0.46% |
| backward.pack_tokens.total | 0.2578 | 0.53% |
| backward.route_metadata.total | 0.0000 | 0.00% |
| backward.router.total | 0.0000 | 0.00% |
| backward.routed_expert.down_base_lora_add.total | 0.6973 | 1.45% |
| backward.routed_expert.down_lora_B.total | 2.0174 | 4.18% |
| backward.routed_expert.down_lora_A.total | 1.1178 | 2.32% |
| backward.routed_expert.activation_silu_mul.total | 1.4315 | 2.97% |
| backward.routed_expert.up_base_lora_add.total | 0.6013 | 1.25% |
| backward.routed_expert.up_lora_B.total | 2.0082 | 4.16% |
| backward.routed_expert.up_lora_A.total | 1.7559 | 3.64% |
| backward.routed_expert.gate_base_lora_add.total | 0.6857 | 1.42% |
| backward.routed_expert.gate_lora_B.total | 1.9517 | 4.04% |
| backward.routed_expert.gate_lora_A.total | 1.7596 | 3.65% |
| backward.shared_expert.down_base_lora_add.total | 0.1459 | 0.30% |
| backward.shared_expert.down_lora_B.total | 0.5532 | 1.15% |
| backward.shared_expert.down_lora_A.total | 0.2865 | 0.59% |
| backward.shared_expert.activation_silu_mul.total | 0.3691 | 0.76% |
| backward.shared_expert.up_base_lora_add.total | 0.1502 | 0.31% |
| backward.shared_expert.up_lora_B.total | 0.5227 | 1.08% |
| backward.shared_expert.up_lora_A.total | 0.4439 | 0.92% |
| backward.shared_expert.gate_base_lora_add.total | 0.1782 | 0.37% |
| backward.shared_expert.gate_lora_B.total | 0.4982 | 1.03% |
| backward.shared_expert.gate_lora_A.total | 0.4441 | 0.92% |
| backward.moe.layernorm.total | 0.3829 | 0.79% |
| backward.attention.residual_add.total | 0.1643 | 0.34% |
| backward.attention.o_proj_base.total | 0.2230 | 0.46% |
| backward.attention.value_matmul.total | 0.4192 | 0.87% |
| backward.attention.softmax.total | 0.2171 | 0.45% |
| backward.attention.scores_matmul.total | 0.3274 | 0.68% |
| backward.attention.v_proj_base.total | 0.1796 | 0.37% |
| backward.attention.k_proj_base.total | 0.1300 | 0.27% |
| backward.attention.q_proj_base.total | 0.1184 | 0.25% |
| backward.attention.layernorm.total | 0.3545 | 0.73% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 19.7071 | 40.84% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 48.2496 | 100.00% |

## Memory

| Component | bytes |
|---|---:|
| peak_hbm | 126937600 |
| gpu_parameters | 11804672 |
| gpu_buffers | 799744 |
| pinned_W | 3932160 |
| pinned_W_T | 0 |
| pinned_total | 3932160 |
