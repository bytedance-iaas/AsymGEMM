# m4_1_mlp Source-Label Coverage Report

This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_m4.py` for kernel-busy, memcpy, and GPU no-kernel percentages.

These source-label tables are for auditing range coverage. In `--timing-mode profile`, inner rows are asynchronous Python/NVTX range timings. In `--timing-mode debug_sync`, rows are synchronized debugging timings and carry profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 0.3025 | 4.25% |
| step.forward | 2.7701 | 38.91% |
| step.loss | 0.2869 | 4.03% |
| step.backward | 3.1248 | 43.90% |
| step.optimizer | 0.6343 | 8.91% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 7.1187 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.fc1.base_frozen_asymgemm | 0.7491 | 27.04% |
| forward.fc1.lora_A | 0.2705 | 9.77% |
| forward.fc1.lora_B | 0.1612 | 5.82% |
| forward.fc1.add_cast_scale | 0.3044 | 10.99% |
| forward.activation_relu | 0.0774 | 2.80% |
| forward.fc2.base_frozen_asymgemm | 0.4563 | 16.47% |
| forward.fc2.lora_A | 0.1511 | 5.45% |
| forward.fc2.lora_B | 0.0894 | 3.23% |
| forward.fc2.add_cast_scale | 0.1998 | 7.21% |
| forward.python_dispatch_cuda_launch_and_sync | 0.3108 | 11.22% |
| **sum** | 2.7701 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.mse.total | 0.2732 | 8.74% |
| backward.fc2.base_dx_asymgemm.total | 0.0000 | 0.00% |
| backward.fc2.base_lora_add.total | 0.0977 | 3.13% |
| backward.fc2.add_cast_scale.total | 0.0540 | 1.73% |
| backward.fc2.lora_B.total | 0.2101 | 6.72% |
| backward.fc2.lora_A.total | 0.1135 | 3.63% |
| backward.activation_relu.total | 0.0915 | 2.93% |
| backward.fc1.base_dx_asymgemm.total | 0.0000 | 0.00% |
| backward.fc1.base_lora_add.total | 0.0788 | 2.52% |
| backward.fc1.add_cast_scale.total | 0.0509 | 1.63% |
| backward.fc1.lora_B.total | 0.1498 | 4.80% |
| backward.fc1.lora_A.total | 0.1030 | 3.30% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 1.9021 | 60.87% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 3.1248 | 100.00% |

## Memory

| Component | bytes |
|---|---:|
| peak_hbm | 67611136 |
| gpu_parameters | 24576 |
| gpu_buffers | 0 |
| pinned_W | 262144 |
| pinned_W_T | 0 |
| pinned_total | 262144 |
