# M4.1-M4.3 Detailed Operation Profiling

The operation tables below do not use an `unwrapped ops` bucket. Explicit tensor work is named; the remaining non-tensor bucket is Python/autograd dispatch, CUDA launch latency, and profiler synchronization overhead.

## MLP

### Forward operation breakdown

| Operation | ms | % forward |
|---|---:|---:|
| forward.fc1.base_frozen_asymgemm | 0.2631 | 22.43% |
| forward.fc2.base_frozen_asymgemm | 0.2216 | 18.89% |
| forward.python_dispatch_cuda_launch_and_sync | 0.1904 | 16.24% |
| forward.fc1.add_cast_scale | 0.1035 | 8.83% |
| forward.fc2.add_cast_scale | 0.0919 | 7.84% |
| forward.fc1.lora_A | 0.0911 | 7.77% |
| forward.fc2.lora_A | 0.0714 | 6.09% |
| forward.fc1.lora_B | 0.0578 | 4.92% |
| forward.fc2.lora_B | 0.0451 | 3.84% |
| forward.activation_relu | 0.0369 | 3.15% |

### Backward operation breakdown

| Operation | ms | % backward |
|---|---:|---:|
| backward.python_autograd_dispatch_cuda_launch_and_sync | 0.6572 | 45.66% |
| backward.base_dx_asymgemm.total | 0.2580 | 17.93% |
| backward.fc2.lora_B.total | 0.0881 | 6.12% |
| backward.fc1.lora_B.total | 0.0763 | 5.30% |
| backward.fc2.lora_A.total | 0.0647 | 4.49% |
| backward.fc1.lora_A.total | 0.0624 | 4.34% |
| backward.loss.mse.total | 0.0541 | 3.76% |
| backward.activation_relu.total | 0.0411 | 2.86% |
| backward.fc2.base_lora_add.total | 0.0406 | 2.82% |
| backward.fc1.base_lora_add.total | 0.0392 | 2.73% |
| backward.fc2.add_cast_scale.total | 0.0292 | 2.03% |
| backward.fc1.add_cast_scale.total | 0.0283 | 1.97% |

## Dense toy

### Forward operation breakdown

| Operation | ms | % forward |
|---|---:|---:|
| forward.python_dispatch_cuda_launch_and_sync | 6.2167 | 27.67% |
| forward.mlp.gate_proj.base_frozen_asymgemm | 1.0038 | 4.47% |
| forward.attention.q_proj.base_frozen_asymgemm | 0.9785 | 4.36% |
| forward.attention.o_proj.base_frozen_asymgemm | 0.9654 | 4.30% |
| forward.mlp.up_proj.base_frozen_asymgemm | 0.9545 | 4.25% |
| forward.attention.k_proj.base_frozen_asymgemm | 0.9425 | 4.20% |
| forward.attention.v_proj.base_frozen_asymgemm | 0.8856 | 3.94% |
| forward.mlp.down_proj.base_frozen_asymgemm | 0.8738 | 3.89% |
| forward.attention.layernorm | 0.6218 | 2.77% |
| forward.mlp.layernorm | 0.5747 | 2.56% |
| forward.attention.q_proj.lora_A | 0.5518 | 2.46% |
| forward.mlp.gate_proj.lora_A | 0.5296 | 2.36% |
| forward.attention.o_proj.lora_A | 0.5165 | 2.30% |
| forward.mlp.up_proj.lora_A | 0.4997 | 2.22% |
| forward.mlp.down_proj.lora_A | 0.4936 | 2.20% |
| forward.attention.k_proj.lora_A | 0.4914 | 2.19% |
| forward.attention.v_proj.lora_A | 0.4867 | 2.17% |
| forward.attention.scores_matmul | 0.4787 | 2.13% |
| forward.attention.value_matmul | 0.3949 | 1.76% |
| forward.attention.q_proj.lora_B | 0.3919 | 1.74% |
| forward.attention.k_proj.lora_B | 0.3855 | 1.72% |
| forward.attention.o_proj.lora_B | 0.3847 | 1.71% |
| forward.mlp.down_proj.lora_B | 0.3803 | 1.69% |
| forward.attention.v_proj.lora_B | 0.3782 | 1.68% |
| forward.attention.causal_mask | 0.3081 | 1.37% |
| forward.mlp.gate_proj.lora_B | 0.3054 | 1.36% |
| forward.mlp.up_proj.lora_B | 0.2959 | 1.32% |
| forward.attention.residual_add | 0.2521 | 1.12% |

### Backward operation breakdown

| Operation | ms | % backward |
|---|---:|---:|
| backward.python_autograd_dispatch_cuda_launch_and_sync | 9.0698 | 38.35% |
| backward.base_dx_asymgemm.total | 3.5296 | 14.92% |
| backward.mlp.down_proj.lora_B.total | 0.5174 | 2.19% |
| backward.attention.o_proj.lora_B.total | 0.5048 | 2.13% |
| backward.mlp.up_proj.lora_B.total | 0.5047 | 2.13% |
| backward.attention.value_matmul.total | 0.5036 | 2.13% |
| backward.attention.k_proj.lora_B.total | 0.4935 | 2.09% |
| backward.attention.q_proj.lora_B.total | 0.4916 | 2.08% |
| backward.mlp.gate_proj.lora_B.total | 0.4823 | 2.04% |
| backward.attention.v_proj.lora_B.total | 0.4807 | 2.03% |
| backward.mlp.up_proj.lora_A.total | 0.4378 | 1.85% |
| backward.attention.k_proj.lora_A.total | 0.4375 | 1.85% |
| backward.mlp.gate_proj.lora_A.total | 0.4368 | 1.85% |
| backward.attention.v_proj.lora_A.total | 0.4363 | 1.84% |
| backward.attention.o_proj.lora_A.total | 0.4359 | 1.84% |
| backward.attention.q_proj.lora_A.total | 0.4343 | 1.84% |
| backward.attention.scores_matmul.total | 0.3821 | 1.62% |
| backward.attention.layernorm.total | 0.3625 | 1.53% |
| backward.mlp.silu_mul_activation.total | 0.3594 | 1.52% |
| backward.mlp.layernorm.total | 0.3572 | 1.51% |
| backward.mlp.down_proj.lora_A.total | 0.2796 | 1.18% |
| backward.attention.softmax.total | 0.2177 | 0.92% |
| backward.attention.k_proj.base_lora_add.total | 0.1783 | 0.75% |
| backward.attention.v_proj.base_lora_add.total | 0.1716 | 0.73% |
| backward.attention.q_proj.base_lora_add.total | 0.1702 | 0.72% |
| backward.mlp.gate_proj.base_lora_add.total | 0.1680 | 0.71% |
| backward.mlp.residual_add.total | 0.1552 | 0.66% |
| backward.attention.residual_add.total | 0.1516 | 0.64% |
| backward.mlp.up_proj.base_lora_add.total | 0.1472 | 0.62% |
| backward.mlp.down_proj.base_lora_add.total | 0.1385 | 0.59% |

## MoE contiguous

### Forward operation breakdown

| Operation | ms | % forward |
|---|---:|---:|
| forward.python_dispatch_cuda_launch_and_sync | 11.4809 | 23.69% |
| forward.routed_expert.gate_base_asymgemm | 4.2671 | 8.80% |
| forward.routed_expert.up_base_asymgemm | 3.9891 | 8.23% |
| forward.routed_expert.down_base_asymgemm | 3.4256 | 7.07% |
| forward.routed_expert.gate_lora_A | 2.0261 | 4.18% |
| forward.routed_expert.down_lora_A | 1.8718 | 3.86% |
| forward.routed_expert.up_lora_A | 1.8356 | 3.79% |
| forward.routed_expert.down_lora_B | 1.4771 | 3.05% |
| forward.route_metadata | 1.4625 | 3.02% |
| forward.routed_expert.gate_lora_B | 1.1653 | 2.40% |
| forward.routed_expert.up_lora_B | 1.1357 | 2.34% |
| forward.routed_expert.activation_silu_mul | 1.0975 | 2.26% |
| forward.routed_expert.add | 1.0409 | 2.15% |
| forward.shared_expert.gate_base_asymgemm | 0.9952 | 2.05% |
| forward.shared_expert.up_base_asymgemm | 0.9775 | 2.02% |
| forward.shared_expert.down_base_asymgemm | 0.8604 | 1.78% |
| forward.attention.layernorm | 0.5694 | 1.17% |
| forward.moe.layernorm | 0.5116 | 1.06% |
| forward.shared_expert.gate_lora_A | 0.4950 | 1.02% |
| forward.attention.scores_matmul | 0.4724 | 0.97% |
| forward.shared_expert.down_lora_A | 0.4640 | 0.96% |
| forward.shared_expert.up_lora_A | 0.4558 | 0.94% |
| forward.routed_expert.gate_lora_scale_cast | 0.4001 | 0.83% |
| forward.routed_expert.down_lora_scale_cast | 0.3917 | 0.81% |
| forward.routed_expert.up_lora_scale_cast | 0.3858 | 0.80% |
| forward.attention.value_matmul | 0.3807 | 0.79% |
| forward.shared_expert.down_lora_B | 0.3727 | 0.77% |
| forward.attention.q_proj_base | 0.3575 | 0.74% |

### Backward operation breakdown

| Operation | ms | % backward |
|---|---:|---:|
| backward.python_autograd_dispatch_cuda_launch_and_sync | 19.7071 | 40.84% |
| backward.base_dx_asymgemm.total | 7.4889 | 15.52% |
| backward.routed_expert.down_lora_B.total | 2.0174 | 4.18% |
| backward.routed_expert.up_lora_B.total | 2.0082 | 4.16% |
| backward.routed_expert.gate_lora_B.total | 1.9517 | 4.04% |
| backward.routed_expert.gate_lora_A.total | 1.7596 | 3.65% |
| backward.routed_expert.up_lora_A.total | 1.7559 | 3.64% |
| backward.routed_expert.activation_silu_mul.total | 1.4315 | 2.97% |
| backward.routed_expert.down_lora_A.total | 1.1178 | 2.32% |
| backward.routed_expert.down_base_lora_add.total | 0.6973 | 1.45% |
| backward.routed_expert.gate_base_lora_add.total | 0.6857 | 1.42% |
| backward.routed_expert.up_base_lora_add.total | 0.6013 | 1.25% |
| backward.shared_expert.down_lora_B.total | 0.5532 | 1.15% |
| backward.shared_expert.up_lora_B.total | 0.5227 | 1.08% |
| backward.shared_expert.gate_lora_B.total | 0.4982 | 1.03% |
| backward.shared_expert.gate_lora_A.total | 0.4441 | 0.92% |
| backward.shared_expert.up_lora_A.total | 0.4439 | 0.92% |
| backward.attention.value_matmul.total | 0.4192 | 0.87% |
| backward.moe.layernorm.total | 0.3829 | 0.79% |
| backward.shared_expert.activation_silu_mul.total | 0.3691 | 0.76% |
| backward.attention.layernorm.total | 0.3545 | 0.73% |
| backward.attention.scores_matmul.total | 0.3274 | 0.68% |
| backward.shared_expert.down_lora_A.total | 0.2865 | 0.59% |
| backward.pack_tokens.total | 0.2578 | 0.53% |
| backward.scatter_combine.total | 0.2238 | 0.46% |
| backward.attention.o_proj_base.total | 0.2230 | 0.46% |
| backward.attention.softmax.total | 0.2171 | 0.45% |
| backward.attention.v_proj_base.total | 0.1796 | 0.37% |
| backward.shared_expert.gate_base_lora_add.total | 0.1782 | 0.37% |
| backward.moe.residual_add.total | 0.1686 | 0.35% |

## MoE masked

### Forward operation breakdown

| Operation | ms | % forward |
|---|---:|---:|
| forward.python_dispatch_cuda_launch_and_sync | 11.1382 | 22.24% |
| forward.routed_expert.gate_base_asymgemm | 4.2609 | 8.51% |
| forward.routed_expert.up_base_asymgemm | 4.1595 | 8.31% |
| forward.routed_expert.down_base_asymgemm | 3.4404 | 6.87% |
| forward.route_metadata | 2.1044 | 4.20% |
| forward.routed_expert.gate_lora_A | 2.0526 | 4.10% |
| forward.routed_expert.down_lora_A | 1.8761 | 3.75% |
| forward.routed_expert.up_lora_A | 1.8748 | 3.74% |
| forward.routed_expert.down_lora_B | 1.4997 | 3.00% |
| forward.routed_expert.gate_lora_B | 1.1784 | 2.35% |
| forward.routed_expert.up_lora_B | 1.1626 | 2.32% |
| forward.routed_expert.activation_silu_mul | 1.1566 | 2.31% |
| forward.routed_expert.add | 1.0488 | 2.09% |
| forward.shared_expert.gate_base_asymgemm | 1.0025 | 2.00% |
| forward.shared_expert.up_base_asymgemm | 0.9322 | 1.86% |
| forward.shared_expert.down_base_asymgemm | 0.8573 | 1.71% |
| forward.scatter_combine | 0.7638 | 1.53% |
| forward.attention.layernorm | 0.6165 | 1.23% |
| forward.shared_expert.gate_lora_A | 0.5281 | 1.05% |
| forward.moe.layernorm | 0.5234 | 1.05% |
| forward.attention.scores_matmul | 0.5064 | 1.01% |
| forward.shared_expert.down_lora_A | 0.4653 | 0.93% |
| forward.shared_expert.up_lora_A | 0.4628 | 0.92% |
| forward.attention.q_proj_base | 0.4189 | 0.84% |
| forward.routed_expert.down_lora_scale_cast | 0.4140 | 0.83% |
| forward.routed_expert.up_lora_scale_cast | 0.4072 | 0.81% |
| forward.attention.value_matmul | 0.4057 | 0.81% |
| forward.routed_expert.gate_lora_scale_cast | 0.4028 | 0.80% |

### Backward operation breakdown

| Operation | ms | % backward |
|---|---:|---:|
| backward.python_autograd_dispatch_cuda_launch_and_sync | 22.0569 | 40.73% |
| backward.base_dx_asymgemm.total | 8.1642 | 15.08% |
| backward.routed_expert.down_lora_B.total | 2.1573 | 3.98% |
| backward.routed_expert.up_lora_B.total | 2.1458 | 3.96% |
| backward.routed_expert.gate_lora_B.total | 2.0563 | 3.80% |
| backward.routed_expert.up_lora_A.total | 1.8619 | 3.44% |
| backward.routed_expert.gate_lora_A.total | 1.8324 | 3.38% |
| backward.routed_expert.activation_silu_mul.total | 1.5847 | 2.93% |
| backward.routed_expert.down_lora_A.total | 1.2163 | 2.25% |
| backward.scatter_combine.total | 0.9624 | 1.78% |
| backward.routed_expert.down_base_lora_add.total | 0.7811 | 1.44% |
| backward.routed_expert.gate_base_lora_add.total | 0.7569 | 1.40% |
| backward.pack_tokens.total | 0.7385 | 1.36% |
| backward.routed_expert.up_base_lora_add.total | 0.6629 | 1.22% |
| backward.shared_expert.down_lora_B.total | 0.5877 | 1.09% |
| backward.shared_expert.up_lora_B.total | 0.5599 | 1.03% |
| backward.shared_expert.gate_lora_B.total | 0.5284 | 0.98% |
| backward.shared_expert.up_lora_A.total | 0.4756 | 0.88% |
| backward.shared_expert.gate_lora_A.total | 0.4681 | 0.86% |
| backward.attention.value_matmul.total | 0.4624 | 0.85% |
| backward.moe.layernorm.total | 0.4346 | 0.80% |
| backward.shared_expert.activation_silu_mul.total | 0.4033 | 0.74% |
| backward.attention.layernorm.total | 0.3900 | 0.72% |
| backward.attention.scores_matmul.total | 0.3549 | 0.66% |
| backward.shared_expert.down_lora_A.total | 0.3176 | 0.59% |
| backward.attention.o_proj_base.total | 0.2548 | 0.47% |
| backward.attention.softmax.total | 0.2394 | 0.44% |
| backward.shared_expert.gate_base_lora_add.total | 0.1964 | 0.36% |
| backward.attention.v_proj_base.total | 0.1952 | 0.36% |
| backward.moe.residual_add.total | 0.1876 | 0.35% |

## Nsight Systems MLP Bubble Check

Captured with `--timing-mode profile`, so inner ranges do not force per-op CUDA synchronization. The GPU timeline rows are additive inside each stage.

### step.forward GPU timeline

| Component | ms | % stage |
|---|---:|---:|
| gpu_no_kernel_time | 1.3688 | 93.14% |
| cuda_kernel_busy_union | 0.0979 | 6.66% |
| cuda_memcpy_union | 0.0030 | 0.20% |

### step.forward operation kernel time

| Operation | ms | % stage |
|---|---:|---:|
| forward.fc1.base_frozen_asymgemm | 0.0364 | 2.48% |
| forward.fc2.base_frozen_asymgemm | 0.0155 | 1.06% |
| forward.fc1.add_cast_scale | 0.0099 | 0.67% |
| forward.fc2.add_cast_scale | 0.0095 | 0.64% |
| forward.fc1.lora_A | 0.0082 | 0.56% |
| forward.fc2.lora_A | 0.0080 | 0.54% |
| forward.fc2.lora_B | 0.0070 | 0.47% |
| forward.fc1.lora_B | 0.0024 | 0.16% |
| forward.activation_relu | 0.0010 | 0.07% |

### step.backward GPU timeline

| Component | ms | % stage |
|---|---:|---:|
| gpu_no_kernel_time | 1.6085 | 94.96% |
| cuda_kernel_busy_union | 0.0823 | 4.86% |
| cuda_memcpy_union | 0.0031 | 0.18% |

### step.backward operation kernel time

| Operation | ms | % stage |
|---|---:|---:|
| backward.fc1.lora_B | 0.0083 | 0.49% |
| backward.fc2.lora_B | 0.0081 | 0.48% |
| backward.fc2.lora_A | 0.0074 | 0.44% |
| backward.fc1.lora_A | 0.0073 | 0.43% |
| backward.activation_relu | 0.0049 | 0.29% |
| backward.fc1.base_lora_add | 0.0046 | 0.27% |
| backward.fc2.base_lora_add | 0.0042 | 0.25% |
| backward.fc1.add_cast_scale | 0.0034 | 0.20% |
| backward.fc2.add_cast_scale | 0.0033 | 0.19% |
| backward.loss.mse | 0.0024 | 0.14% |


## Profiling Modes

- `--timing-mode profile` is the real profiling mode. Inner operation labels are NVTX/record_function ranges and do not force per-op CUDA synchronization. Use this under Nsight Systems, then postprocess the SQLite export with `scripts/postprocess_nsys_m4.py`.
- `--timing-mode debug_sync` is only a source coverage/debug mode. It synchronizes around each labeled region so the toy source tables are easy to audit, but those numbers must not be used for performance claims.
- The same strategy is intended for LLaMA-Factory: wrap module/function boundaries with `asym_gemm.training.profile_ranges.prof_range(...)`, run with profiling enabled, capture Nsight Systems, and postprocess NVTX ranges. The toy script is just the local harness for validating the labels.

## Findings

- W.T materialization remains eliminated: all generated M4 reports show `pinned_w_t_bytes = 0`.
- There is no `unwrapped ops` bucket in the generated reports. The source-level non-tensor residual is explicitly named `python_dispatch_cuda_launch_and_sync` or `python_autograd_dispatch_cuda_launch_and_sync`.
- Nsight Systems is the authority for true GPU bubbles. In the MLP profile-mode trace, backward has 4.86% CUDA kernel busy, 0.18% memcpy, and 94.96% `gpu_no_kernel_time`.
- The no-kernel time is caused by launch/autograd scheduling granularity for the tiny workload, not W.T copy or device memcpy.
