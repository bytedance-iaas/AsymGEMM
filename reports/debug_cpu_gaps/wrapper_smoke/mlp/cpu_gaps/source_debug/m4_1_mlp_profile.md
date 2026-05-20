# m4_1_mlp Source-Label Coverage Report

This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_m4.py` for kernel-busy, memcpy, and GPU no-kernel percentages.

These source-label tables are for auditing range coverage. In `--timing-mode profile`, inner rows are asynchronous Python/NVTX range timings. In `--timing-mode debug_sync`, rows are synchronized debugging timings and carry profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 0.6019 | 0.03% |
| step.forward | 1135.6010 | 60.13% |
| step.loss | 52.8407 | 2.80% |
| step.backward | 649.1586 | 34.37% |
| step.optimizer | 50.3972 | 2.67% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 1888.5994 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.fc1.base_frozen_asymgemm | 641.2979 | 56.47% |
| forward.fc1.lora_A | 104.6322 | 9.21% |
| forward.fc1.lora_B | 30.6221 | 2.70% |
| forward.fc1.add_cast_scale | 13.9997 | 1.23% |
| forward.activation_relu | 27.1589 | 2.39% |
| forward.fc2.base_frozen_asymgemm | 316.0000 | 27.83% |
| forward.fc2.lora_A | 0.4257 | 0.04% |
| forward.fc2.lora_B | 0.3157 | 0.03% |
| forward.fc2.add_cast_scale | 0.3003 | 0.03% |
| forward.python_dispatch_cuda_launch_and_sync | 0.8486 | 0.07% |
| **sum** | 1135.6010 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.mse.total | 0.3667 | 0.06% |
| backward.fc2.base_dx_asymgemm.total | 311.5979 | 48.00% |
| backward.fc2.base_lora_add.total | 0.0867 | 0.01% |
| backward.fc2.add_cast_scale.total | 0.0430 | 0.01% |
| backward.fc2.lora_B.total | 1.4086 | 0.22% |
| backward.fc2.lora_A.total | 0.1431 | 0.02% |
| backward.activation_relu.total | 9.3384 | 1.44% |
| backward.fc1.base_dx_asymgemm.total | 314.2655 | 48.41% |
| backward.fc1.base_lora_add.total | 0.1427 | 0.02% |
| backward.fc1.add_cast_scale.total | 0.0468 | 0.01% |
| backward.fc1.lora_B.total | 0.2869 | 0.04% |
| backward.fc1.lora_A.total | 0.1154 | 0.02% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 11.3170 | 1.74% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 649.1586 | 100.00% |

## Memory

| Component | bytes |
|---|---:|
| peak_hbm | 67427328 |
| gpu_parameters | 24576 |
| gpu_buffers | 0 |
| host_W | 131072 |
| host_W_T | 0 |
| pinned_W | 131072 |
| pinned_W_T | 0 |
| pinned_total | 131072 |
