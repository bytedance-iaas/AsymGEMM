# Nsight M4 Trace: reports/nsys_mlp_profile.sqlite

## step.forward

### Stage Timeline

| Component | ms | % stage |
|---|---:|---:|
| gpu_no_kernel_time | 1.1925 | 92.18% |
| cuda_kernel_busy_union | 0.0982 | 7.59% |
| cuda_memcpy_union | 0.0030 | 0.23% |

### Host CUDA API

| Component | ms | % stage |
|---|---:|---:|
| cuda_runtime_api_sum_overlaps_gpu_timeline | 0.1857 | 14.35% |
| cuda_synchronization_api_sum_overlaps_gpu_timeline | 0.0244 | 1.89% |

### Operation Kernel Time

| Component | ms | % stage |
|---|---:|---:|
| forward.fc1.base_frozen_asymgemm | 0.0341 | 2.63% |
| forward.fc2.base_frozen_asymgemm | 0.0181 | 1.40% |
| forward.fc1.add_cast_scale | 0.0098 | 0.76% |
| forward.fc2.add_cast_scale | 0.0095 | 0.74% |
| forward.fc1.lora_A | 0.0080 | 0.62% |
| forward.fc2.lora_A | 0.0080 | 0.62% |
| forward.fc2.lora_B | 0.0070 | 0.54% |
| forward.fc1.lora_B | 0.0026 | 0.20% |
| forward.activation_relu | 0.0011 | 0.08% |

### Operation CUDA API Time

| Component | ms | % stage |
|---|---:|---:|
| forward.fc1.base_frozen_asymgemm | 0.0409 | 3.16% |
| forward.fc2.base_frozen_asymgemm | 0.0369 | 2.85% |
| forward.fc1.add_cast_scale | 0.0260 | 2.01% |
| forward.fc2.add_cast_scale | 0.0247 | 1.91% |
| forward.fc1.lora_A | 0.0181 | 1.40% |
| forward.fc2.lora_A | 0.0156 | 1.20% |
| forward.fc2.lora_B | 0.0085 | 0.66% |
| forward.fc1.lora_B | 0.0058 | 0.45% |
| forward.activation_relu | 0.0040 | 0.31% |

## step.backward

### Stage Timeline

| Component | ms | % stage |
|---|---:|---:|
| gpu_no_kernel_time | 1.4405 | 94.48% |
| cuda_kernel_busy_union | 0.0812 | 5.32% |
| cuda_memcpy_union | 0.0030 | 0.20% |

### Host CUDA API

| Component | ms | % stage |
|---|---:|---:|
| cuda_runtime_api_sum_overlaps_gpu_timeline | 0.2099 | 13.77% |
| cuda_synchronization_api_sum_overlaps_gpu_timeline | 0.0247 | 1.62% |

### Operation Kernel Time

| Component | ms | % stage |
|---|---:|---:|
| backward.fc1.base_dx_asymgemm | 0.0138 | 0.90% |
| backward.fc2.base_dx_asymgemm | 0.0084 | 0.55% |
| backward.fc1.lora_B | 0.0082 | 0.54% |
| backward.fc2.lora_B | 0.0080 | 0.52% |
| backward.fc2.lora_A | 0.0074 | 0.48% |
| backward.fc1.lora_A | 0.0073 | 0.48% |
| backward.activation_relu | 0.0046 | 0.30% |
| backward.fc1.base_lora_add | 0.0045 | 0.30% |
| backward.fc2.base_lora_add | 0.0042 | 0.27% |
| backward.fc1.add_cast_scale | 0.0034 | 0.22% |
| backward.fc2.add_cast_scale | 0.0034 | 0.22% |
| backward.loss.mse | 0.0025 | 0.16% |

### Operation CUDA API Time

| Component | ms | % stage |
|---|---:|---:|
| backward.fc2.base_dx_asymgemm | 0.0359 | 2.36% |
| backward.fc1.base_dx_asymgemm | 0.0354 | 2.32% |
| backward.fc2.lora_B | 0.0144 | 0.94% |
| backward.fc1.lora_B | 0.0133 | 0.87% |
| backward.fc2.lora_A | 0.0126 | 0.82% |
| backward.fc1.lora_A | 0.0124 | 0.82% |
| backward.fc2.base_lora_add | 0.0116 | 0.76% |
| backward.fc1.base_lora_add | 0.0111 | 0.73% |
| backward.loss.mse | 0.0101 | 0.66% |
| backward.activation_relu | 0.0087 | 0.57% |
| backward.fc1.add_cast_scale | 0.0073 | 0.48% |
| backward.fc2.add_cast_scale | 0.0072 | 0.47% |
