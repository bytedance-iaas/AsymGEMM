# m4_1_mlp Source-Label Coverage Report

This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_m4.py` for kernel-busy, memcpy, and GPU no-kernel percentages.

These source-label tables are for auditing range coverage. In `--timing-mode profile`, inner rows are asynchronous Python/NVTX range timings. In `--timing-mode debug_sync`, rows are synchronized debugging timings and carry profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 0.1720 | 3.38% |
| step.forward | 1.9190 | 37.71% |
| step.loss | 0.2301 | 4.52% |
| step.backward | 2.3266 | 45.72% |
| step.optimizer | 0.4410 | 8.67% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 5.0886 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.fc1.base_frozen_asymgemm | 0.5207 | 27.13% |
| forward.fc1.lora_A | 0.2154 | 11.23% |
| forward.fc1.lora_B | 0.1076 | 5.61% |
| forward.fc1.add_cast_scale | 0.1819 | 9.48% |
| forward.activation_relu | 0.0582 | 3.03% |
| forward.fc2.base_frozen_asymgemm | 0.3100 | 16.15% |
| forward.fc2.lora_A | 0.1131 | 5.89% |
| forward.fc2.lora_B | 0.0619 | 3.23% |
| forward.fc2.add_cast_scale | 0.1386 | 7.22% |
| forward.python_dispatch_cuda_launch_and_sync | 0.2116 | 11.03% |
| **sum** | 1.9190 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.mse.total | 0.1837 | 7.90% |
| backward.fc2.base_dx_asymgemm.total | 0.2151 | 9.25% |
| backward.fc2.base_lora_add.total | 0.0737 | 3.17% |
| backward.fc2.add_cast_scale.total | 0.0405 | 1.74% |
| backward.fc2.lora_B.total | 0.1795 | 7.71% |
| backward.fc2.lora_A.total | 0.0831 | 3.57% |
| backward.activation_relu.total | 0.0644 | 2.77% |
| backward.fc1.base_dx_asymgemm.total | 0.1701 | 7.31% |
| backward.fc1.base_lora_add.total | 0.0581 | 2.50% |
| backward.fc1.add_cast_scale.total | 0.0358 | 1.54% |
| backward.fc1.lora_B.total | 0.1094 | 4.70% |
| backward.fc1.lora_A.total | 0.0737 | 3.17% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 1.0396 | 44.68% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 2.3266 | 100.00% |

## Memory

| Component | bytes |
|---|---:|
| peak_hbm | 67611136 |
| gpu_parameters | 24576 |
| gpu_buffers | 0 |
| host_W | 131072 |
| host_W_T | 0 |
| pinned_W | 131072 |
| pinned_W_T | 0 |
| pinned_total | 131072 |
