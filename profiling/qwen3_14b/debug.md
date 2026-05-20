# qwen3_14b Source-Label Coverage Report

This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_m4.py` for kernel-busy, memcpy, and GPU no-kernel percentages.

These source-label tables are for auditing range coverage. In `--timing-mode profile`, inner rows are asynchronous Python/NVTX range timings. In `--timing-mode debug_sync`, rows are synchronized debugging timings and carry profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 2.7640 | 5.74% |
| step.forward | 23.3500 | 48.50% |
| step.loss | 0.1889 | 0.39% |
| step.backward | 21.4232 | 44.50% |
| step.optimizer | 0.4140 | 0.86% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 48.1402 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.embeddings | 0.0528 | 0.23% |
| forward.layers.0.attention.layernorm | 0.1924 | 0.82% |
| forward.layers.0.attention.scores_matmul | 0.1053 | 0.45% |
| forward.layers.0.attention.causal_mask | 0.0792 | 0.34% |
| forward.layers.0.attention.softmax | 0.0376 | 0.16% |
| forward.layers.0.attention.value_matmul | 0.0872 | 0.37% |
| forward.layers.0.attention.residual_add | 0.0532 | 0.23% |
| forward.layers.0.attention.q_proj.base_frozen_asymgemm | 0.2773 | 1.19% |
| forward.layers.0.attention.q_proj.lora_A | 0.1022 | 0.44% |
| forward.layers.0.attention.q_proj.lora_B | 0.0628 | 0.27% |
| forward.layers.0.attention.q_proj.add_cast_scale | 0.1058 | 0.45% |
| forward.layers.0.attention.k_proj.base_frozen_asymgemm | 1.2613 | 5.40% |
| forward.layers.0.attention.k_proj.lora_A | 0.0748 | 0.32% |
| forward.layers.0.attention.k_proj.lora_B | 0.0527 | 0.23% |
| forward.layers.0.attention.k_proj.add_cast_scale | 0.0986 | 0.42% |
| forward.layers.0.attention.v_proj.base_frozen_asymgemm | 1.2823 | 5.49% |
| forward.layers.0.attention.v_proj.lora_A | 0.0709 | 0.30% |
| forward.layers.0.attention.v_proj.lora_B | 0.0518 | 0.22% |
| forward.layers.0.attention.v_proj.add_cast_scale | 0.0977 | 0.42% |
| forward.layers.0.attention.o_proj.base_frozen_asymgemm | 0.9478 | 4.06% |
| forward.layers.0.attention.o_proj.lora_A | 0.0711 | 0.30% |
| forward.layers.0.attention.o_proj.lora_B | 0.0455 | 0.19% |
| forward.layers.0.attention.o_proj.add_cast_scale | 0.1286 | 0.55% |
| forward.layers.0.mlp.layernorm | 0.1309 | 0.56% |
| forward.layers.0.mlp.silu_mul_activation | 0.0531 | 0.23% |
| forward.layers.0.mlp.residual_add | 0.0513 | 0.22% |
| forward.layers.0.mlp.gate_proj.base_frozen_asymgemm | 1.1393 | 4.88% |
| forward.layers.0.mlp.gate_proj.lora_A | 0.0721 | 0.31% |
| forward.layers.0.mlp.gate_proj.lora_B | 0.0481 | 0.21% |
| forward.layers.0.mlp.gate_proj.add_cast_scale | 0.0841 | 0.36% |
| forward.layers.0.mlp.up_proj.base_frozen_asymgemm | 5.3081 | 22.73% |
| forward.layers.0.mlp.up_proj.lora_A | 0.0683 | 0.29% |
| forward.layers.0.mlp.up_proj.lora_B | 0.0460 | 0.20% |
| forward.layers.0.mlp.up_proj.add_cast_scale | 0.0842 | 0.36% |
| forward.layers.0.mlp.down_proj.base_frozen_asymgemm | 5.2702 | 22.57% |
| forward.layers.0.mlp.down_proj.lora_A | 0.0676 | 0.29% |
| forward.layers.0.mlp.down_proj.lora_B | 0.0465 | 0.20% |
| forward.layers.0.mlp.down_proj.add_cast_scale | 0.0845 | 0.36% |
| forward.final_norm | 0.1307 | 0.56% |
| forward.lm_head | 0.0645 | 0.28% |
| forward.python_dispatch_cuda_launch_and_sync | 5.2615 | 22.53% |
| **sum** | 23.3500 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.cross_entropy.total | 0.1377 | 0.64% |
| backward.lm_head.total | 0.0682 | 0.32% |
| backward.final_norm.total | 0.0899 | 0.42% |
| backward.layers.0.mlp.residual_add.total | 0.0361 | 0.17% |
| backward.layers.0.mlp.down_proj.base_dx_asymgemm.total | 0.1681 | 0.78% |
| backward.layers.0.mlp.down_proj.base_lora_add.total | 0.0312 | 0.15% |
| backward.layers.0.mlp.down_proj.add_cast_scale.total | 0.0226 | 0.11% |
| backward.layers.0.mlp.down_proj.lora_B.total | 0.0851 | 0.40% |
| backward.layers.0.mlp.down_proj.lora_A.total | 0.0542 | 0.25% |
| backward.layers.0.mlp.silu_mul_activation.total | 0.0838 | 0.39% |
| backward.layers.0.mlp.up_proj.base_dx_asymgemm.total | 4.4644 | 20.84% |
| backward.layers.0.mlp.up_proj.base_lora_add.total | 0.0328 | 0.15% |
| backward.layers.0.mlp.up_proj.add_cast_scale.total | 0.0227 | 0.11% |
| backward.layers.0.mlp.up_proj.lora_B.total | 0.0737 | 0.34% |
| backward.layers.0.mlp.up_proj.lora_A.total | 0.0508 | 0.24% |
| backward.layers.0.mlp.gate_proj.base_dx_asymgemm.total | 4.2401 | 19.79% |
| backward.layers.0.mlp.gate_proj.base_lora_add.total | 0.0419 | 0.20% |
| backward.layers.0.mlp.gate_proj.add_cast_scale.total | 0.0241 | 0.11% |
| backward.layers.0.mlp.gate_proj.lora_B.total | 0.0719 | 0.34% |
| backward.layers.0.mlp.gate_proj.lora_A.total | 0.0491 | 0.23% |
| backward.layers.0.mlp.layernorm.total | 0.0877 | 0.41% |
| backward.layers.0.attention.residual_add.total | 0.0333 | 0.16% |
| backward.layers.0.attention.o_proj.base_dx_asymgemm.total | 4.1539 | 19.39% |
| backward.layers.0.attention.o_proj.base_lora_add.total | 0.0293 | 0.14% |
| backward.layers.0.attention.o_proj.add_cast_scale.total | 0.0213 | 0.10% |
| backward.layers.0.attention.o_proj.lora_B.total | 0.0712 | 0.33% |
| backward.layers.0.attention.o_proj.lora_A.total | 0.0468 | 0.22% |
| backward.layers.0.attention.value_matmul.total | 0.0958 | 0.45% |
| backward.layers.0.attention.softmax.total | 0.0521 | 0.24% |
| backward.layers.0.attention.scores_matmul.total | 0.0675 | 0.32% |
| backward.layers.0.attention.v_proj.base_dx_asymgemm.total | 0.6602 | 3.08% |
| backward.layers.0.attention.v_proj.base_lora_add.total | 0.0377 | 0.18% |
| backward.layers.0.attention.v_proj.add_cast_scale.total | 0.0227 | 0.11% |
| backward.layers.0.attention.v_proj.lora_B.total | 0.0621 | 0.29% |
| backward.layers.0.attention.v_proj.lora_A.total | 0.0462 | 0.22% |
| backward.layers.0.attention.k_proj.base_dx_asymgemm.total | 1.0481 | 4.89% |
| backward.layers.0.attention.k_proj.base_lora_add.total | 0.0400 | 0.19% |
| backward.layers.0.attention.k_proj.add_cast_scale.total | 0.0230 | 0.11% |
| backward.layers.0.attention.k_proj.lora_B.total | 0.0673 | 0.31% |
| backward.layers.0.attention.k_proj.lora_A.total | 0.0460 | 0.21% |
| backward.layers.0.attention.q_proj.base_dx_asymgemm.total | 1.0888 | 5.08% |
| backward.layers.0.attention.q_proj.base_lora_add.total | 0.0375 | 0.18% |
| backward.layers.0.attention.q_proj.add_cast_scale.total | 0.0224 | 0.10% |
| backward.layers.0.attention.q_proj.lora_B.total | 0.0650 | 0.30% |
| backward.layers.0.attention.q_proj.lora_A.total | 0.0464 | 0.22% |
| backward.layers.0.attention.layernorm.total | 0.0855 | 0.40% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 3.5166 | 16.42% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 21.4232 | 100.00% |

## Memory

| Component | bytes |
|---|---:|
| peak_hbm | 298149888 |
| gpu_parameters | 27848704 |
| gpu_buffers | 84541440 |
| pinned_W | 744488960 |
| pinned_W_T | 0 |
| pinned_total | 744488960 |
