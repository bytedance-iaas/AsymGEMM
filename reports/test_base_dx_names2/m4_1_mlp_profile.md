# m4_1_mlp Source-Label Coverage Report

This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_m4.py` for kernel-busy, memcpy, and GPU no-kernel percentages.

These source-label tables are for auditing range coverage. In `--timing-mode profile`, inner rows are asynchronous Python/NVTX range timings. In `--timing-mode debug_sync`, rows are synchronized debugging timings and carry profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 0.1341 | 3.34% |
| step.forward | 1.4885 | 37.11% |
| step.loss | 0.1704 | 4.25% |
| step.backward | 1.8412 | 45.90% |
| step.optimizer | 0.3772 | 9.40% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 4.0114 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.fc1.base_frozen_asymgemm | 0.4429 | 29.75% |
| forward.fc1.lora_A | 0.1811 | 12.16% |
| forward.fc1.lora_B | 0.0894 | 6.00% |
| forward.fc1.add_cast_scale | 0.1288 | 8.65% |
| forward.activation_relu | 0.0488 | 3.28% |
| forward.fc2.base_frozen_asymgemm | 0.2336 | 15.70% |
| forward.fc2.lora_A | 0.0656 | 4.41% |
| forward.fc2.lora_B | 0.0415 | 2.79% |
| forward.fc2.add_cast_scale | 0.0851 | 5.71% |
| forward.python_dispatch_cuda_launch_and_sync | 0.1718 | 11.54% |
| **sum** | 1.4885 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.mse.total | 0.1522 | 8.27% |
| backward.fc2.base_dx_asymgemm.total | 0.0000 | 0.00% |
| backward.fc2.base_lora_add.total | 0.0484 | 2.63% |
| backward.fc2.add_cast_scale.total | 0.0274 | 1.49% |
| backward.fc2.lora_B.total | 0.1428 | 7.76% |
| backward.fc2.lora_A.total | 0.0567 | 3.08% |
| backward.activation_relu.total | 0.0529 | 2.88% |
| backward.fc1.base_dx_asymgemm.total | 0.0000 | 0.00% |
| backward.fc1.base_lora_add.total | 0.0351 | 1.91% |
| backward.fc1.add_cast_scale.total | 0.0227 | 1.23% |
| backward.fc1.lora_B.total | 0.0723 | 3.93% |
| backward.fc1.lora_A.total | 0.0494 | 2.68% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 1.1811 | 64.15% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 1.8412 | 100.00% |

## Memory

| Component | bytes |
|---|---:|
| peak_hbm | 67611136 |
| gpu_parameters | 24576 |
| gpu_buffers | 0 |
| pinned_W | 262144 |
| pinned_W_T | 0 |
| pinned_total | 262144 |
