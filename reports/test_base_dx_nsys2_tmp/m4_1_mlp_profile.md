# m4_1_mlp Source-Label Coverage Report

This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_m4.py` for kernel-busy, memcpy, and GPU no-kernel percentages.

These source-label tables are for auditing range coverage. In `--timing-mode profile`, inner rows are asynchronous Python/NVTX range timings. In `--timing-mode debug_sync`, rows are synchronized debugging timings and carry profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 0.2148 | 4.17% |
| step.forward | 1.9413 | 37.73% |
| step.loss | 0.2024 | 3.93% |
| step.backward | 2.3505 | 45.68% |
| step.optimizer | 0.4367 | 8.49% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 5.1457 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.fc1.base_frozen_asymgemm | 0.5372 | 27.67% |
| forward.fc1.lora_A | 0.2307 | 11.88% |
| forward.fc1.lora_B | 0.0993 | 5.12% |
| forward.fc1.add_cast_scale | 0.1868 | 9.62% |
| forward.activation_relu | 0.0568 | 2.93% |
| forward.fc2.base_frozen_asymgemm | 0.3119 | 16.07% |
| forward.fc2.lora_A | 0.1106 | 5.70% |
| forward.fc2.lora_B | 0.0650 | 3.35% |
| forward.fc2.add_cast_scale | 0.1304 | 6.71% |
| forward.python_dispatch_cuda_launch_and_sync | 0.2127 | 10.95% |
| **sum** | 1.9413 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.mse.total | 0.2013 | 8.56% |
| backward.fc2.base_dx_asymgemm.total | 0.0000 | 0.00% |
| backward.fc2.base_lora_add.total | 0.0677 | 2.88% |
| backward.fc2.add_cast_scale.total | 0.0394 | 1.68% |
| backward.fc2.lora_B.total | 0.1767 | 7.52% |
| backward.fc2.lora_A.total | 0.0872 | 3.71% |
| backward.activation_relu.total | 0.0662 | 2.82% |
| backward.fc1.base_dx_asymgemm.total | 0.0000 | 0.00% |
| backward.fc1.base_lora_add.total | 0.0717 | 3.05% |
| backward.fc1.add_cast_scale.total | 0.0358 | 1.52% |
| backward.fc1.lora_B.total | 0.1125 | 4.79% |
| backward.fc1.lora_A.total | 0.0859 | 3.66% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 1.4061 | 59.82% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 2.3505 | 100.00% |

## Memory

| Component | bytes |
|---|---:|
| peak_hbm | 67611136 |
| gpu_parameters | 24576 |
| gpu_buffers | 0 |
| pinned_W | 262144 |
| pinned_W_T | 0 |
| pinned_total | 262144 |
