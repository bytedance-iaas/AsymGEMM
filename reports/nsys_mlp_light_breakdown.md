# Nsight M4 Trace: reports/nsys_mlp_light.sqlite

## step.forward

### Stage Timeline

| Component | ms | % stage |
|---|---:|---:|
| gpu_no_kernel_time | 1.3688 | 93.14% |
| cuda_kernel_busy_union | 0.0979 | 6.66% |
| cuda_memcpy_union | 0.0030 | 0.20% |

### Host CUDA API

| Component | ms | % stage |
|---|---:|---:|
| cuda_runtime_api_sum_overlaps_gpu_timeline | 0.2404 | 16.36% |
| cuda_synchronization_api_sum_overlaps_gpu_timeline | 0.0277 | 1.88% |

### Operation Kernel Time

| Component | ms | % stage |
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

### Operation CUDA API Time

| Component | ms | % stage |
|---|---:|---:|
| forward.fc1.base_frozen_asymgemm | 0.0475 | 3.23% |
| forward.fc2.base_frozen_asymgemm | 0.0473 | 3.22% |
| forward.fc1.add_cast_scale | 0.0375 | 2.55% |
| forward.fc2.add_cast_scale | 0.0334 | 2.27% |
| forward.fc1.lora_A | 0.0235 | 1.60% |
| forward.fc2.lora_A | 0.0191 | 1.30% |
| forward.fc2.lora_B | 0.0113 | 0.77% |
| forward.fc1.lora_B | 0.0089 | 0.61% |
| forward.activation_relu | 0.0051 | 0.35% |

## step.backward

### Stage Timeline

| Component | ms | % stage |
|---|---:|---:|
| gpu_no_kernel_time | 1.6085 | 94.96% |
| cuda_kernel_busy_union | 0.0823 | 4.86% |
| cuda_memcpy_union | 0.0031 | 0.18% |

### Host CUDA API

| Component | ms | % stage |
|---|---:|---:|
| cuda_runtime_api_sum_overlaps_gpu_timeline | 0.2783 | 16.43% |
| cuda_synchronization_api_sum_overlaps_gpu_timeline | 0.0279 | 1.65% |

### Operation Kernel Time

| Component | ms | % stage |
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

### Operation CUDA API Time

| Component | ms | % stage |
|---|---:|---:|
| backward.fc2.lora_B | 0.0202 | 1.19% |
| backward.fc1.lora_B | 0.0200 | 1.18% |
| backward.fc2.lora_A | 0.0186 | 1.10% |
| backward.fc1.lora_A | 0.0167 | 0.99% |
| backward.activation_relu | 0.0157 | 0.93% |
| backward.fc1.base_lora_add | 0.0155 | 0.92% |
| backward.fc2.base_lora_add | 0.0152 | 0.90% |
| backward.loss.mse | 0.0128 | 0.76% |
| backward.fc2.add_cast_scale | 0.0097 | 0.57% |
| backward.fc1.add_cast_scale | 0.0087 | 0.52% |
