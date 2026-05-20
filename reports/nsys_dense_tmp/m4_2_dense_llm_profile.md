# m4_2_dense_llm Profile

All timing tables are additive within their parent. Stage residuals are explicitly labeled as Python/autograd dispatch, CUDA launch latency, and synchronization/profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 0.2119 | 0.45% |
| step.forward | 21.9871 | 46.44% |
| step.loss | 0.3702 | 0.78% |
| step.backward | 23.8798 | 50.43% |
| step.optimizer | 0.9010 | 1.90% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 47.3499 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.embeddings | 0.0603 | 0.27% |
| forward.attention.layernorm | 0.7930 | 3.61% |
| forward.attention.q_proj.base_frozen_asymgemm | 1.0481 | 4.77% |
| forward.attention.q_proj.lora_A | 0.4634 | 2.11% |
| forward.attention.q_proj.lora_B | 0.2463 | 1.12% |
| forward.attention.k_proj.base_frozen_asymgemm | 1.0836 | 4.93% |
| forward.attention.k_proj.lora_A | 0.3980 | 1.81% |
| forward.attention.k_proj.lora_B | 0.2231 | 1.01% |
| forward.attention.v_proj.base_frozen_asymgemm | 0.9459 | 4.30% |
| forward.attention.v_proj.lora_A | 0.4002 | 1.82% |
| forward.attention.v_proj.lora_B | 0.2344 | 1.07% |
| forward.attention.scores_matmul | 0.5984 | 2.72% |
| forward.attention.causal_mask | 0.3636 | 1.65% |
| forward.attention.softmax | 0.1589 | 0.72% |
| forward.attention.value_matmul | 0.4387 | 2.00% |
| forward.attention.o_proj.base_frozen_asymgemm | 1.0145 | 4.61% |
| forward.attention.o_proj.lora_A | 0.4027 | 1.83% |
| forward.attention.o_proj.lora_B | 0.2199 | 1.00% |
| forward.attention.residual_add | 0.3052 | 1.39% |
| forward.mlp.layernorm | 0.7667 | 3.49% |
| forward.mlp.gate_proj.base_frozen_asymgemm | 0.9935 | 4.52% |
| forward.mlp.gate_proj.lora_A | 0.4188 | 1.90% |
| forward.mlp.gate_proj.lora_B | 0.2305 | 1.05% |
| forward.mlp.up_proj.base_frozen_asymgemm | 0.9650 | 4.39% |
| forward.mlp.up_proj.lora_A | 0.4042 | 1.84% |
| forward.mlp.up_proj.lora_B | 0.2244 | 1.02% |
| forward.mlp.silu_mul_activation | 0.3087 | 1.40% |
| forward.mlp.down_proj.base_frozen_asymgemm | 0.9170 | 4.17% |
| forward.mlp.down_proj.lora_A | 0.3657 | 1.66% |
| forward.mlp.down_proj.lora_B | 0.2168 | 0.99% |
| forward.mlp.residual_add | 0.2955 | 1.34% |
| forward.final_norm | 0.1878 | 0.85% |
| forward.lm_head | 0.0817 | 0.37% |
| forward.python_dispatch_cuda_launch_and_sync | 6.2125 | 28.26% |
| **sum** | 21.9871 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.cross_entropy.total | 0.1588 | 0.67% |
| backward.base_dx_asymgemm.total | 4.3614 | 18.26% |
| backward.lm_head.total | 0.0801 | 0.34% |
| backward.final_norm.total | 0.1216 | 0.51% |
| backward.mlp.residual_add.total | 0.1906 | 0.80% |
| backward.mlp.down_proj.base_lora_add.total | 0.1701 | 0.71% |
| backward.mlp.down_proj.add_cast_scale.total | 0.1257 | 0.53% |
| backward.mlp.down_proj.lora_B.total | 0.3685 | 1.54% |
| backward.mlp.down_proj.lora_A.total | 0.3039 | 1.27% |
| backward.mlp.silu_mul_activation.total | 0.4534 | 1.90% |
| backward.mlp.up_proj.base_lora_add.total | 0.1814 | 0.76% |
| backward.mlp.up_proj.add_cast_scale.total | 0.1269 | 0.53% |
| backward.mlp.up_proj.lora_B.total | 0.3562 | 1.49% |
| backward.mlp.up_proj.lora_A.total | 0.2757 | 1.15% |
| backward.mlp.gate_proj.base_lora_add.total | 0.2138 | 0.90% |
| backward.mlp.gate_proj.add_cast_scale.total | 0.1301 | 0.54% |
| backward.mlp.gate_proj.lora_B.total | 0.3550 | 1.49% |
| backward.mlp.gate_proj.lora_A.total | 0.2662 | 1.11% |
| backward.mlp.layernorm.total | 0.5073 | 2.12% |
| backward.attention.residual_add.total | 0.1929 | 0.81% |
| backward.attention.o_proj.base_lora_add.total | 0.1706 | 0.71% |
| backward.attention.o_proj.add_cast_scale.total | 0.1219 | 0.51% |
| backward.attention.o_proj.lora_B.total | 0.3674 | 1.54% |
| backward.attention.o_proj.lora_A.total | 0.2898 | 1.21% |
| backward.attention.value_matmul.total | 0.5864 | 2.46% |
| backward.attention.softmax.total | 0.2673 | 1.12% |
| backward.attention.scores_matmul.total | 0.4636 | 1.94% |
| backward.attention.v_proj.base_lora_add.total | 0.2362 | 0.99% |
| backward.attention.v_proj.add_cast_scale.total | 0.1268 | 0.53% |
| backward.attention.v_proj.lora_B.total | 0.3323 | 1.39% |
| backward.attention.v_proj.lora_A.total | 0.2827 | 1.18% |
| backward.attention.k_proj.base_lora_add.total | 0.2170 | 0.91% |
| backward.attention.k_proj.add_cast_scale.total | 0.1310 | 0.55% |
| backward.attention.k_proj.lora_B.total | 0.3406 | 1.43% |
| backward.attention.k_proj.lora_A.total | 0.2831 | 1.19% |
| backward.attention.q_proj.base_lora_add.total | 0.2147 | 0.90% |
| backward.attention.q_proj.add_cast_scale.total | 0.1364 | 0.57% |
| backward.attention.q_proj.lora_B.total | 0.3622 | 1.52% |
| backward.attention.q_proj.lora_A.total | 0.2823 | 1.18% |
| backward.attention.layernorm.total | 0.4927 | 2.06% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 9.2350 | 38.67% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 23.8798 | 100.00% |

## Memory

| Component | bytes |
|---|---:|
| peak_hbm | 89690112 |
| gpu_parameters | 4461056 |
| gpu_buffers | 264192 |
| pinned_W | 2621440 |
| pinned_W_T | 0 |
| pinned_total | 2621440 |
