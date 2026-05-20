# Nsight M4 Trace: reports/test_base_dx_nsys2.sqlite

## step.forward

### Stage Timeline

| Component | ms | % stage |
|---|---:|---:|
| gpu_no_kernel_time | 1.8074 | 93.42% |
| cuda_kernel_busy_union | 0.1243 | 6.42% |
| cuda_memcpy_union | 0.0030 | 0.15% |

### Host CUDA API

| Component | ms | % stage |
|---|---:|---:|
| cuda_runtime_api_sum_overlaps_gpu_timeline | 0.2790 | 14.42% |
| cuda_synchronization_api_sum_overlaps_gpu_timeline | 0.0298 | 1.54% |

### Operation Kernel Time

| Component | ms | % stage |
|---|---:|---:|
| forward.fc1.base_frozen_asymgemm | 0.0564 | 2.91% |
| forward.fc2.base_frozen_asymgemm | 0.0223 | 1.15% |
| forward.fc1.add_cast_scale | 0.0097 | 0.50% |
| forward.fc2.add_cast_scale | 0.0095 | 0.49% |
| forward.fc1.lora_A | 0.0080 | 0.41% |
| forward.fc2.lora_A | 0.0079 | 0.41% |
| forward.fc2.lora_B | 0.0070 | 0.36% |
| forward.fc1.lora_B | 0.0024 | 0.12% |
| forward.activation_relu | 0.0010 | 0.05% |

### Operation CUDA API Time

| Component | ms | % stage |
|---|---:|---:|
| forward.fc1.base_frozen_asymgemm | 0.0645 | 3.34% |
| forward.fc2.base_frozen_asymgemm | 0.0493 | 2.55% |
| forward.fc1.add_cast_scale | 0.0437 | 2.26% |
| forward.fc2.add_cast_scale | 0.0350 | 1.81% |
| forward.fc1.lora_A | 0.0304 | 1.57% |
| forward.fc2.lora_A | 0.0208 | 1.08% |
| forward.fc2.lora_B | 0.0131 | 0.68% |
| forward.fc1.lora_B | 0.0086 | 0.44% |
| forward.activation_relu | 0.0064 | 0.33% |

## step.backward

### Stage Timeline

| Component | ms | % stage |
|---|---:|---:|
| gpu_no_kernel_time | 2.2586 | 96.37% |
| cuda_kernel_busy_union | 0.0821 | 3.50% |
| cuda_memcpy_union | 0.0030 | 0.13% |

### Host CUDA API

| Component | ms | % stage |
|---|---:|---:|
| cuda_runtime_api_sum_overlaps_gpu_timeline | 0.3361 | 14.34% |
| cuda_synchronization_api_sum_overlaps_gpu_timeline | 0.0288 | 1.23% |

### Operation Kernel Time

| Component | ms | % stage |
|---|---:|---:|
| backward.fc1.base_dx_asymgemm | 0.0140 | 0.60% |
| backward.fc2.base_dx_asymgemm | 0.0087 | 0.37% |
| backward.fc2.lora_B | 0.0083 | 0.35% |
| backward.fc1.lora_B | 0.0082 | 0.35% |
| backward.fc2.lora_A | 0.0074 | 0.31% |
| backward.fc1.lora_A | 0.0072 | 0.31% |
| backward.activation_relu | 0.0048 | 0.20% |
| backward.fc1.base_lora_add | 0.0047 | 0.20% |
| backward.fc2.base_lora_add | 0.0043 | 0.18% |
| backward.fc1.add_cast_scale | 0.0034 | 0.15% |
| backward.fc2.add_cast_scale | 0.0033 | 0.14% |
| backward.loss.mse | 0.0025 | 0.11% |

### Operation CUDA API Time

| Component | ms | % stage |
|---|---:|---:|
| backward.fc2.base_dx_asymgemm | 0.0535 | 2.28% |
| backward.fc1.base_dx_asymgemm | 0.0437 | 1.86% |
| backward.loss.mse | 0.0410 | 1.75% |
| backward.fc2.lora_B | 0.0270 | 1.15% |
| backward.fc1.lora_B | 0.0211 | 0.90% |
| backward.fc2.lora_A | 0.0207 | 0.88% |
| backward.fc2.base_lora_add | 0.0174 | 0.74% |
| backward.fc1.base_lora_add | 0.0163 | 0.69% |
| backward.fc1.lora_A | 0.0161 | 0.68% |
| backward.activation_relu | 0.0144 | 0.61% |
| backward.fc2.add_cast_scale | 0.0104 | 0.45% |
| backward.fc1.add_cast_scale | 0.0090 | 0.38% |
