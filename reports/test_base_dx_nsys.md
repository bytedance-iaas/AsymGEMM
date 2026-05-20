# Nsight M4 Trace: reports/test_base_dx_nsys.sqlite

## step.forward

### Stage Timeline

| Component | ms | % stage |
|---|---:|---:|
| gpu_no_kernel_time | 2.6244 | 95.08% |
| cuda_kernel_busy_union | 0.1329 | 4.81% |
| cuda_memcpy_union | 0.0030 | 0.11% |

### Host CUDA API

| Component | ms | % stage |
|---|---:|---:|
| cuda_runtime_api_sum_overlaps_gpu_timeline | 0.3897 | 14.12% |
| cuda_synchronization_api_sum_overlaps_gpu_timeline | 0.0350 | 1.27% |

### Operation Kernel Time

| Component | ms | % stage |
|---|---:|---:|
| forward.fc1.base_frozen_asymgemm | 0.0641 | 2.32% |
| forward.fc2.base_frozen_asymgemm | 0.0232 | 0.84% |
| forward.fc1.add_cast_scale | 0.0097 | 0.35% |
| forward.fc2.add_cast_scale | 0.0095 | 0.34% |
| forward.fc2.lora_A | 0.0080 | 0.29% |
| forward.fc1.lora_A | 0.0080 | 0.29% |
| forward.fc2.lora_B | 0.0070 | 0.25% |
| forward.fc1.lora_B | 0.0025 | 0.09% |
| forward.activation_relu | 0.0010 | 0.04% |

### Operation CUDA API Time

| Component | ms | % stage |
|---|---:|---:|
| forward.fc1.base_frozen_asymgemm | 0.0826 | 2.99% |
| forward.fc1.add_cast_scale | 0.0753 | 2.73% |
| forward.fc2.base_frozen_asymgemm | 0.0652 | 2.36% |
| forward.fc2.add_cast_scale | 0.0499 | 1.81% |
| forward.fc1.lora_A | 0.0372 | 1.35% |
| forward.fc2.lora_A | 0.0279 | 1.01% |
| forward.fc2.lora_B | 0.0174 | 0.63% |
| forward.fc1.lora_B | 0.0151 | 0.55% |
| forward.activation_relu | 0.0094 | 0.34% |

## step.backward

### Stage Timeline

| Component | ms | % stage |
|---|---:|---:|
| gpu_no_kernel_time | 3.0307 | 97.26% |
| cuda_kernel_busy_union | 0.0825 | 2.65% |
| cuda_memcpy_union | 0.0030 | 0.10% |

### Host CUDA API

| Component | ms | % stage |
|---|---:|---:|
| cuda_runtime_api_sum_overlaps_gpu_timeline | 0.4347 | 13.95% |
| cuda_synchronization_api_sum_overlaps_gpu_timeline | 0.0361 | 1.16% |

### Operation Kernel Time

| Component | ms | % stage |
|---|---:|---:|
| backward.fc2.lora_B | 0.0083 | 0.27% |
| backward.fc1.lora_B | 0.0082 | 0.26% |
| backward.fc2.lora_A | 0.0074 | 0.24% |
| backward.fc1.lora_A | 0.0072 | 0.23% |
| backward.activation_relu | 0.0048 | 0.15% |
| backward.fc1.base_lora_add | 0.0045 | 0.14% |
| backward.fc2.base_lora_add | 0.0043 | 0.14% |
| backward.fc1.add_cast_scale | 0.0034 | 0.11% |
| backward.fc2.add_cast_scale | 0.0033 | 0.11% |
| backward.loss.mse | 0.0026 | 0.08% |

### Operation CUDA API Time

| Component | ms | % stage |
|---|---:|---:|
| backward.loss.mse | 0.0535 | 1.72% |
| backward.fc2.lora_B | 0.0319 | 1.02% |
| backward.fc1.lora_B | 0.0274 | 0.88% |
| backward.fc2.base_lora_add | 0.0242 | 0.78% |
| backward.fc2.lora_A | 0.0241 | 0.77% |
| backward.fc1.lora_A | 0.0225 | 0.72% |
| backward.fc1.base_lora_add | 0.0219 | 0.70% |
| backward.activation_relu | 0.0167 | 0.54% |
| backward.fc2.add_cast_scale | 0.0147 | 0.47% |
| backward.fc1.add_cast_scale | 0.0137 | 0.44% |
