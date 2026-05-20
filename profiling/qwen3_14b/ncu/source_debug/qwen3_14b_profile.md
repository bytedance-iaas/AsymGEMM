# qwen3_14b Source-Label Coverage Report

This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_m4.py` for kernel-busy, memcpy, and GPU no-kernel percentages.

These source-label tables are for auditing range coverage. In `--timing-mode profile`, inner rows are asynchronous Python/NVTX range timings. In `--timing-mode debug_sync`, rows are synchronized debugging timings and carry profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 1.8606 | 0.01% |
| step.forward | 17131.1008 | 52.49% |
| step.loss | 1.1536 | 0.00% |
| step.backward | 15500.9412 | 47.50% |
| step.optimizer | 1.9160 | 0.01% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 32636.9721 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.embeddings | 0.2381 | 0.00% |
| forward.layers.0.attention.layernorm | 1.5094 | 0.01% |
| forward.layers.0.attention.scores_matmul | 0.6492 | 0.00% |
| forward.layers.0.attention.causal_mask | 0.5070 | 0.00% |
| forward.layers.0.attention.softmax | 0.1826 | 0.00% |
| forward.layers.0.attention.value_matmul | 0.4746 | 0.00% |
| forward.layers.0.attention.residual_add | 0.4932 | 0.00% |
| forward.layers.0.attention.q_proj.base_frozen_asymgemm | 3785.1781 | 22.10% |
| forward.layers.0.attention.q_proj.lora_A | 0.6652 | 0.00% |
| forward.layers.0.attention.q_proj.lora_B | 0.2322 | 0.00% |
| forward.layers.0.attention.q_proj.add_cast_scale | 0.8193 | 0.00% |
| forward.layers.0.attention.k_proj.base_frozen_asymgemm | 2227.2438 | 13.00% |
| forward.layers.0.attention.k_proj.lora_A | 1.0085 | 0.01% |
| forward.layers.0.attention.k_proj.lora_B | 0.2709 | 0.00% |
| forward.layers.0.attention.k_proj.add_cast_scale | 0.9004 | 0.01% |
| forward.layers.0.attention.v_proj.base_frozen_asymgemm | 2134.7744 | 12.46% |
| forward.layers.0.attention.v_proj.lora_A | 0.9752 | 0.01% |
| forward.layers.0.attention.v_proj.lora_B | 0.2671 | 0.00% |
| forward.layers.0.attention.v_proj.add_cast_scale | 0.9047 | 0.01% |
| forward.layers.0.attention.o_proj.base_frozen_asymgemm | 2137.9380 | 12.48% |
| forward.layers.0.attention.o_proj.lora_A | 0.9121 | 0.01% |
| forward.layers.0.attention.o_proj.lora_B | 0.2521 | 0.00% |
| forward.layers.0.attention.o_proj.add_cast_scale | 0.8277 | 0.00% |
| forward.layers.0.mlp.layernorm | 1.3249 | 0.01% |
| forward.layers.0.mlp.silu_mul_activation | 0.5061 | 0.00% |
| forward.layers.0.mlp.residual_add | 0.4610 | 0.00% |
| forward.layers.0.mlp.gate_proj.base_frozen_asymgemm | 2330.8070 | 13.61% |
| forward.layers.0.mlp.gate_proj.lora_A | 0.8997 | 0.01% |
| forward.layers.0.mlp.gate_proj.lora_B | 0.2520 | 0.00% |
| forward.layers.0.mlp.gate_proj.add_cast_scale | 0.7977 | 0.00% |
| forward.layers.0.mlp.up_proj.base_frozen_asymgemm | 2255.3055 | 13.16% |
| forward.layers.0.mlp.up_proj.lora_A | 0.8925 | 0.01% |
| forward.layers.0.mlp.up_proj.lora_B | 0.2478 | 0.00% |
| forward.layers.0.mlp.up_proj.add_cast_scale | 0.7998 | 0.00% |
| forward.layers.0.mlp.down_proj.base_frozen_asymgemm | 2235.9098 | 13.05% |
| forward.layers.0.mlp.down_proj.lora_A | 0.8863 | 0.01% |
| forward.layers.0.mlp.down_proj.lora_B | 0.2647 | 0.00% |
| forward.layers.0.mlp.down_proj.add_cast_scale | 0.7853 | 0.00% |
| forward.final_norm | 1.3260 | 0.01% |
| forward.lm_head | 0.2615 | 0.00% |
| forward.python_dispatch_cuda_launch_and_sync | 2.1491 | 0.01% |
| **sum** | 17131.1008 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.cross_entropy.total | 0.9326 | 0.01% |
| backward.lm_head.total | 0.2084 | 0.00% |
| backward.final_norm.total | 0.8027 | 0.01% |
| backward.layers.0.mlp.residual_add.total | 0.2856 | 0.00% |
| backward.layers.0.mlp.down_proj.base_dx_asymgemm.total | 2230.3733 | 14.39% |
| backward.layers.0.mlp.down_proj.base_lora_add.total | 0.2684 | 0.00% |
| backward.layers.0.mlp.down_proj.add_cast_scale.total | 0.1813 | 0.00% |
| backward.layers.0.mlp.down_proj.lora_B.total | 0.4143 | 0.00% |
| backward.layers.0.mlp.down_proj.lora_A.total | 0.3160 | 0.00% |
| backward.layers.0.mlp.silu_mul_activation.total | 1.7866 | 0.01% |
| backward.layers.0.mlp.up_proj.base_dx_asymgemm.total | 2207.8550 | 14.24% |
| backward.layers.0.mlp.up_proj.base_lora_add.total | 0.3009 | 0.00% |
| backward.layers.0.mlp.up_proj.add_cast_scale.total | 0.2000 | 0.00% |
| backward.layers.0.mlp.up_proj.lora_B.total | 0.4913 | 0.00% |
| backward.layers.0.mlp.up_proj.lora_A.total | 0.2601 | 0.00% |
| backward.layers.0.mlp.gate_proj.base_dx_asymgemm.total | 2301.6547 | 14.85% |
| backward.layers.0.mlp.gate_proj.base_lora_add.total | 0.4359 | 0.00% |
| backward.layers.0.mlp.gate_proj.add_cast_scale.total | 0.2174 | 0.00% |
| backward.layers.0.mlp.gate_proj.lora_B.total | 0.5118 | 0.00% |
| backward.layers.0.mlp.gate_proj.lora_A.total | 0.2613 | 0.00% |
| backward.layers.0.mlp.layernorm.total | 1.0329 | 0.01% |
| backward.layers.0.attention.residual_add.total | 0.3330 | 0.00% |
| backward.layers.0.attention.o_proj.base_dx_asymgemm.total | 2334.9335 | 15.06% |
| backward.layers.0.attention.o_proj.base_lora_add.total | 0.2969 | 0.00% |
| backward.layers.0.attention.o_proj.add_cast_scale.total | 0.2036 | 0.00% |
| backward.layers.0.attention.o_proj.lora_B.total | 0.5106 | 0.00% |
| backward.layers.0.attention.o_proj.lora_A.total | 0.2787 | 0.00% |
| backward.layers.0.attention.value_matmul.total | 0.5893 | 0.00% |
| backward.layers.0.attention.softmax.total | 0.4586 | 0.00% |
| backward.layers.0.attention.scores_matmul.total | 0.3045 | 0.00% |
| backward.layers.0.attention.v_proj.base_dx_asymgemm.total | 2163.5516 | 13.96% |
| backward.layers.0.attention.v_proj.base_lora_add.total | 0.3029 | 0.00% |
| backward.layers.0.attention.v_proj.add_cast_scale.total | 0.1830 | 0.00% |
| backward.layers.0.attention.v_proj.lora_B.total | 0.3679 | 0.00% |
| backward.layers.0.attention.v_proj.lora_A.total | 0.2238 | 0.00% |
| backward.layers.0.attention.k_proj.base_dx_asymgemm.total | 2131.3232 | 13.75% |
| backward.layers.0.attention.k_proj.base_lora_add.total | 0.4218 | 0.00% |
| backward.layers.0.attention.k_proj.add_cast_scale.total | 0.1983 | 0.00% |
| backward.layers.0.attention.k_proj.lora_B.total | 0.5262 | 0.00% |
| backward.layers.0.attention.k_proj.lora_A.total | 0.2527 | 0.00% |
| backward.layers.0.attention.q_proj.base_dx_asymgemm.total | 2102.5150 | 13.56% |
| backward.layers.0.attention.q_proj.base_lora_add.total | 0.3867 | 0.00% |
| backward.layers.0.attention.q_proj.add_cast_scale.total | 0.1991 | 0.00% |
| backward.layers.0.attention.q_proj.lora_B.total | 0.4929 | 0.00% |
| backward.layers.0.attention.q_proj.lora_A.total | 0.2437 | 0.00% |
| backward.layers.0.attention.layernorm.total | 0.9377 | 0.01% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 12.1154 | 0.08% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 15500.9412 | 100.00% |

## Memory

| Component | bytes |
|---|---:|
| peak_hbm | 297625600 |
| gpu_parameters | 27848704 |
| gpu_buffers | 84541440 |
| host_W | 744488960 |
| host_W_T | 0 |
| pinned_W | 744488960 |
| pinned_W_T | 0 |
| pinned_total | 744488960 |
