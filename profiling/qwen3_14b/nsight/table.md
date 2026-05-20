# Nsight M4 Trace: reports/qwen_requested_20260520/nsys_qwen3_14b_profile.sqlite

## step.forward

### Stage Timeline

| Component | ms | % stage |
|---|---:|---:|
| cuda_kernel_busy_union | 24.0079 | 95.53% |
| gpu_no_kernel_time | 1.1096 | 4.42% |
| cuda_memcpy_union | 0.0134 | 0.05% |

### Host CUDA API

| Component | ms | % stage |
|---|---:|---:|
| cuda_runtime_api_sum_overlaps_gpu_timeline | 20.0688 | 79.86% |
| cuda_synchronization_api_sum_overlaps_gpu_timeline | 19.0853 | 75.94% |

### Operation Kernel Time

| Component | ms | % stage |
|---|---:|---:|
| forward.layers.0.mlp.up_proj.base_frozen_asymgemm | 5.7946 | 23.06% |
| forward.layers.0.mlp.gate_proj.base_frozen_asymgemm | 5.7895 | 23.04% |
| forward.layers.0.mlp.down_proj.base_frozen_asymgemm | 5.4733 | 21.78% |
| forward.layers.0.attention.v_proj.base_frozen_asymgemm | 1.6232 | 6.46% |
| forward.layers.0.attention.o_proj.base_frozen_asymgemm | 1.6197 | 6.45% |
| forward.layers.0.attention.q_proj.base_frozen_asymgemm | 1.6167 | 6.43% |
| forward.layers.0.attention.k_proj.base_frozen_asymgemm | 1.6141 | 6.42% |
| forward.layers.0.attention.layernorm | 0.0453 | 0.18% |
| forward.layers.0.mlp.layernorm | 0.0446 | 0.18% |
| forward.final_norm | 0.0444 | 0.18% |
| forward.layers.0.mlp.down_proj.lora_A | 0.0237 | 0.09% |
| forward.layers.0.mlp.gate_proj.add_cast_scale | 0.0205 | 0.08% |
| forward.layers.0.mlp.up_proj.add_cast_scale | 0.0203 | 0.08% |
| forward.layers.0.mlp.silu_mul_activation | 0.0173 | 0.07% |
| forward.lm_head | 0.0170 | 0.07% |
| forward.layers.0.attention.q_proj.lora_A | 0.0139 | 0.06% |
| forward.layers.0.mlp.gate_proj.lora_A | 0.0138 | 0.05% |
| forward.layers.0.attention.o_proj.lora_A | 0.0136 | 0.05% |
| forward.layers.0.mlp.up_proj.lora_A | 0.0134 | 0.05% |
| forward.layers.0.attention.k_proj.lora_A | 0.0134 | 0.05% |
| forward.layers.0.attention.v_proj.lora_A | 0.0134 | 0.05% |
| forward.layers.0.mlp.down_proj.add_cast_scale | 0.0123 | 0.05% |
| forward.layers.0.attention.q_proj.add_cast_scale | 0.0122 | 0.05% |
| forward.layers.0.attention.scores_matmul | 0.0121 | 0.05% |
| forward.layers.0.attention.o_proj.add_cast_scale | 0.0121 | 0.05% |
| forward.layers.0.attention.k_proj.add_cast_scale | 0.0121 | 0.05% |
| forward.layers.0.attention.v_proj.add_cast_scale | 0.0121 | 0.05% |
| forward.layers.0.mlp.residual_add | 0.0104 | 0.04% |
| forward.layers.0.attention.value_matmul | 0.0102 | 0.04% |
| forward.layers.0.attention.residual_add | 0.0101 | 0.04% |
| forward.layers.0.mlp.gate_proj.lora_B | 0.0098 | 0.04% |
| forward.layers.0.mlp.up_proj.lora_B | 0.0098 | 0.04% |
| forward.layers.0.mlp.down_proj.lora_B | 0.0055 | 0.02% |
| forward.layers.0.attention.o_proj.lora_B | 0.0054 | 0.02% |
| forward.layers.0.attention.q_proj.lora_B | 0.0053 | 0.02% |
| forward.layers.0.attention.v_proj.lora_B | 0.0052 | 0.02% |
| forward.layers.0.attention.k_proj.lora_B | 0.0051 | 0.02% |
| forward.layers.0.attention.causal_mask | 0.0046 | 0.02% |
| forward.layers.0.attention.softmax | 0.0025 | 0.01% |
| forward.embeddings | 0.0017 | 0.01% |

### Operation CUDA API Time

| Component | ms | % stage |
|---|---:|---:|
| forward.layers.0.mlp.up_proj.base_frozen_asymgemm | 5.3210 | 21.17% |
| forward.layers.0.mlp.down_proj.base_frozen_asymgemm | 5.2537 | 20.91% |
| forward.layers.0.attention.k_proj.base_frozen_asymgemm | 1.1220 | 4.46% |
| forward.layers.0.attention.v_proj.base_frozen_asymgemm | 1.1154 | 4.44% |
| forward.layers.0.mlp.gate_proj.base_frozen_asymgemm | 0.9032 | 3.59% |
| forward.layers.0.attention.o_proj.base_frozen_asymgemm | 0.7727 | 3.07% |
| forward.layers.0.mlp.layernorm | 0.0679 | 0.27% |
| forward.layers.0.attention.layernorm | 0.0658 | 0.26% |
| forward.final_norm | 0.0583 | 0.23% |
| forward.layers.0.attention.q_proj.base_frozen_asymgemm | 0.0438 | 0.17% |
| forward.layers.0.attention.q_proj.add_cast_scale | 0.0335 | 0.13% |
| forward.layers.0.attention.o_proj.add_cast_scale | 0.0316 | 0.13% |
| forward.layers.0.mlp.up_proj.add_cast_scale | 0.0315 | 0.13% |
| forward.layers.0.mlp.down_proj.add_cast_scale | 0.0314 | 0.12% |
| forward.layers.0.mlp.gate_proj.add_cast_scale | 0.0313 | 0.12% |
| forward.layers.0.attention.v_proj.add_cast_scale | 0.0309 | 0.12% |
| forward.layers.0.attention.k_proj.add_cast_scale | 0.0309 | 0.12% |
| forward.layers.0.attention.causal_mask | 0.0293 | 0.12% |
| forward.layers.0.attention.residual_add | 0.0217 | 0.09% |
| forward.layers.0.mlp.silu_mul_activation | 0.0216 | 0.09% |
| forward.layers.0.mlp.residual_add | 0.0212 | 0.08% |
| forward.layers.0.attention.scores_matmul | 0.0202 | 0.08% |
| forward.layers.0.attention.q_proj.lora_A | 0.0191 | 0.08% |
| forward.layers.0.mlp.gate_proj.lora_A | 0.0189 | 0.08% |
| forward.layers.0.attention.o_proj.lora_A | 0.0188 | 0.07% |
| forward.layers.0.mlp.up_proj.lora_A | 0.0182 | 0.07% |
| forward.layers.0.mlp.down_proj.lora_A | 0.0179 | 0.07% |
| forward.layers.0.attention.k_proj.lora_A | 0.0175 | 0.07% |
| forward.layers.0.attention.v_proj.lora_A | 0.0167 | 0.07% |
| forward.layers.0.attention.value_matmul | 0.0152 | 0.06% |
| forward.embeddings | 0.0133 | 0.05% |
| forward.lm_head | 0.0092 | 0.04% |
| forward.layers.0.attention.k_proj.lora_B | 0.0071 | 0.03% |
| forward.layers.0.mlp.up_proj.lora_B | 0.0067 | 0.03% |
| forward.layers.0.attention.q_proj.lora_B | 0.0067 | 0.03% |
| forward.layers.0.mlp.down_proj.lora_B | 0.0066 | 0.03% |
| forward.layers.0.mlp.gate_proj.lora_B | 0.0066 | 0.03% |
| forward.layers.0.attention.o_proj.lora_B | 0.0062 | 0.02% |
| forward.layers.0.attention.v_proj.lora_B | 0.0059 | 0.02% |
| forward.layers.0.attention.softmax | 0.0051 | 0.02% |

## step.backward

### Stage Timeline

| Component | ms | % stage |
|---|---:|---:|
| cuda_kernel_busy_union | 21.6657 | 92.17% |
| gpu_no_kernel_time | 1.8248 | 7.76% |
| cuda_memcpy_union | 0.0161 | 0.07% |

### Host CUDA API

| Component | ms | % stage |
|---|---:|---:|
| cuda_runtime_api_sum_overlaps_gpu_timeline | 17.7190 | 75.38% |
| cuda_synchronization_api_sum_overlaps_gpu_timeline | 16.6341 | 70.76% |

### Operation Kernel Time

| Component | ms | % stage |
|---|---:|---:|
| backward.layers.0.mlp.down_proj.base_dx_asymgemm | 5.3077 | 22.58% |
| backward.layers.0.mlp.gate_proj.base_dx_asymgemm | 4.9980 | 21.26% |
| backward.layers.0.mlp.up_proj.base_dx_asymgemm | 4.9617 | 21.11% |
| backward.layers.0.attention.k_proj.base_dx_asymgemm | 1.4661 | 6.24% |
| backward.layers.0.attention.q_proj.base_dx_asymgemm | 1.4661 | 6.24% |
| backward.layers.0.attention.o_proj.base_dx_asymgemm | 1.4612 | 6.22% |
| backward.layers.0.attention.v_proj.base_dx_asymgemm | 1.4605 | 6.21% |
| backward.final_norm | 0.0384 | 0.16% |
| backward.layers.0.mlp.layernorm | 0.0382 | 0.16% |
| backward.layers.0.attention.layernorm | 0.0377 | 0.16% |
| backward.layers.0.mlp.gate_proj.lora_B | 0.0246 | 0.10% |
| backward.layers.0.mlp.up_proj.lora_B | 0.0244 | 0.10% |
| backward.layers.0.mlp.silu_mul_activation | 0.0207 | 0.09% |
| backward.layers.0.mlp.down_proj.lora_A | 0.0178 | 0.08% |
| backward.layers.0.mlp.down_proj.lora_B | 0.0148 | 0.06% |
| backward.layers.0.attention.q_proj.lora_B | 0.0147 | 0.06% |
| backward.layers.0.attention.o_proj.lora_B | 0.0147 | 0.06% |
| backward.layers.0.attention.v_proj.lora_B | 0.0146 | 0.06% |
| backward.layers.0.attention.k_proj.lora_B | 0.0140 | 0.06% |
| backward.layers.0.mlp.up_proj.lora_A | 0.0116 | 0.05% |
| backward.lm_head | 0.0116 | 0.05% |
| backward.layers.0.mlp.gate_proj.lora_A | 0.0116 | 0.05% |
| backward.layers.0.attention.o_proj.lora_A | 0.0112 | 0.05% |
| backward.layers.0.attention.v_proj.lora_A | 0.0112 | 0.05% |
| backward.layers.0.attention.q_proj.lora_A | 0.0111 | 0.05% |
| backward.layers.0.attention.k_proj.lora_A | 0.0106 | 0.05% |
| backward.layers.0.mlp.gate_proj.base_lora_add | 0.0094 | 0.04% |
| backward.layers.0.mlp.up_proj.base_lora_add | 0.0090 | 0.04% |
| backward.layers.0.attention.softmax | 0.0074 | 0.03% |
| backward.loss.cross_entropy | 0.0071 | 0.03% |
| backward.layers.0.attention.value_matmul.lhs_grad | 0.0071 | 0.03% |
| backward.layers.0.mlp.up_proj.add_cast_scale | 0.0069 | 0.03% |
| backward.layers.0.mlp.gate_proj.add_cast_scale | 0.0069 | 0.03% |
| backward.layers.0.attention.scores_matmul.lhs_grad | 0.0061 | 0.03% |
| backward.layers.0.attention.scores_matmul.rhs_grad | 0.0056 | 0.02% |
| backward.layers.0.attention.q_proj.base_lora_add | 0.0055 | 0.02% |
| backward.layers.0.attention.v_proj.base_lora_add | 0.0055 | 0.02% |
| backward.layers.0.attention.k_proj.base_lora_add | 0.0054 | 0.02% |
| backward.layers.0.mlp.residual_add | 0.0053 | 0.02% |
| backward.layers.0.attention.value_matmul.rhs_grad | 0.0053 | 0.02% |
| backward.layers.0.attention.residual_add | 0.0052 | 0.02% |
| backward.layers.0.mlp.down_proj.base_lora_add | 0.0052 | 0.02% |
| backward.layers.0.attention.o_proj.base_lora_add | 0.0051 | 0.02% |
| backward.layers.0.mlp.down_proj.add_cast_scale | 0.0040 | 0.02% |
| backward.layers.0.attention.q_proj.add_cast_scale | 0.0039 | 0.02% |
| backward.layers.0.attention.v_proj.add_cast_scale | 0.0039 | 0.02% |
| backward.layers.0.attention.o_proj.add_cast_scale | 0.0038 | 0.02% |
| backward.layers.0.attention.k_proj.add_cast_scale | 0.0038 | 0.02% |

### Operation CUDA API Time

| Component | ms | % stage |
|---|---:|---:|
| backward.layers.0.mlp.up_proj.base_dx_asymgemm | 4.6720 | 19.88% |
| backward.layers.0.mlp.gate_proj.base_dx_asymgemm | 4.4453 | 18.91% |
| backward.layers.0.attention.o_proj.base_dx_asymgemm | 4.2474 | 18.07% |
| backward.layers.0.attention.q_proj.base_dx_asymgemm | 0.9252 | 3.94% |
| backward.layers.0.attention.k_proj.base_dx_asymgemm | 0.9061 | 3.85% |
| backward.layers.0.attention.v_proj.base_dx_asymgemm | 0.4098 | 1.74% |
| backward.loss.cross_entropy | 0.0476 | 0.20% |
| backward.layers.0.mlp.down_proj.base_dx_asymgemm | 0.0470 | 0.20% |
| backward.layers.0.mlp.layernorm | 0.0432 | 0.18% |
| backward.final_norm | 0.0415 | 0.18% |
| backward.layers.0.attention.layernorm | 0.0388 | 0.17% |
| backward.layers.0.mlp.silu_mul_activation | 0.0298 | 0.13% |
| backward.layers.0.attention.v_proj.base_lora_add | 0.0270 | 0.12% |
| backward.layers.0.mlp.gate_proj.lora_B | 0.0204 | 0.09% |
| backward.layers.0.mlp.down_proj.lora_B | 0.0189 | 0.08% |
| backward.layers.0.attention.o_proj.lora_B | 0.0186 | 0.08% |
| backward.layers.0.attention.softmax | 0.0182 | 0.08% |
| backward.layers.0.attention.k_proj.lora_B | 0.0170 | 0.07% |
| backward.layers.0.mlp.up_proj.lora_B | 0.0170 | 0.07% |
| backward.layers.0.attention.v_proj.lora_B | 0.0166 | 0.07% |
| backward.layers.0.attention.q_proj.lora_B | 0.0165 | 0.07% |
| backward.layers.0.attention.k_proj.base_lora_add | 0.0140 | 0.06% |
| backward.layers.0.mlp.gate_proj.base_lora_add | 0.0135 | 0.06% |
| backward.layers.0.mlp.residual_add | 0.0134 | 0.06% |
| backward.layers.0.mlp.down_proj.base_lora_add | 0.0134 | 0.06% |
| backward.layers.0.attention.q_proj.base_lora_add | 0.0131 | 0.06% |
| backward.layers.0.attention.o_proj.base_lora_add | 0.0131 | 0.06% |
| backward.layers.0.mlp.up_proj.base_lora_add | 0.0126 | 0.05% |
| backward.layers.0.attention.residual_add | 0.0124 | 0.05% |
| backward.layers.0.mlp.down_proj.lora_A | 0.0120 | 0.05% |
| backward.layers.0.mlp.gate_proj.lora_A | 0.0116 | 0.05% |
| backward.layers.0.mlp.up_proj.lora_A | 0.0115 | 0.05% |
| backward.layers.0.attention.o_proj.lora_A | 0.0110 | 0.05% |
| backward.layers.0.attention.v_proj.lora_A | 0.0108 | 0.05% |
| backward.layers.0.attention.q_proj.lora_A | 0.0108 | 0.05% |
| backward.layers.0.attention.k_proj.lora_A | 0.0101 | 0.04% |
| backward.layers.0.attention.v_proj.add_cast_scale | 0.0095 | 0.04% |
| backward.lm_head | 0.0095 | 0.04% |
| backward.layers.0.attention.o_proj.add_cast_scale | 0.0092 | 0.04% |
| backward.layers.0.attention.k_proj.add_cast_scale | 0.0092 | 0.04% |
| backward.layers.0.mlp.gate_proj.add_cast_scale | 0.0091 | 0.04% |
| backward.layers.0.mlp.up_proj.add_cast_scale | 0.0089 | 0.04% |
| backward.layers.0.attention.q_proj.add_cast_scale | 0.0087 | 0.04% |
| backward.layers.0.mlp.down_proj.add_cast_scale | 0.0085 | 0.04% |
| backward.layers.0.attention.value_matmul.lhs_grad | 0.0060 | 0.03% |
| backward.layers.0.attention.scores_matmul.lhs_grad | 0.0058 | 0.02% |
| backward.layers.0.attention.value_matmul.rhs_grad | 0.0052 | 0.02% |
| backward.layers.0.attention.scores_matmul.rhs_grad | 0.0051 | 0.02% |
