# m4_1_mlp Source-Label Coverage Report

This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_m4.py` for kernel-busy, memcpy, and GPU no-kernel percentages.

These source-label tables are for auditing range coverage. In `--timing-mode profile`, inner rows are asynchronous Python/NVTX range timings. In `--timing-mode debug_sync`, rows are synchronized debugging timings and carry profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 0.0671 | 2.00% |
| step.forward | 1.2910 | 38.50% |
| step.loss | 0.1302 | 3.88% |
| step.backward | 1.5781 | 47.06% |
| step.optimizer | 0.2868 | 8.55% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 3.3532 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.fc1.base_frozen_asymgemm | 0.2793 | 21.63% |
| forward.fc1.lora_A | 0.1059 | 8.20% |
| forward.fc1.lora_B | 0.0615 | 4.76% |
| forward.fc1.add_cast_scale | 0.1303 | 10.09% |
| forward.activation_relu | 0.0375 | 2.90% |
| forward.fc2.base_frozen_asymgemm | 0.2507 | 19.42% |
| forward.fc2.lora_A | 0.0865 | 6.70% |
| forward.fc2.lora_B | 0.0483 | 3.75% |
| forward.fc2.add_cast_scale | 0.1213 | 9.40% |
| forward.python_dispatch_cuda_launch_and_sync | 0.1698 | 13.15% |
| **sum** | 1.2910 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.mse.total | 0.0550 | 3.49% |
| backward.fc2.base_dx_asymgemm.total | 0.1588 | 10.07% |
| backward.fc2.base_lora_add.total | 0.0495 | 3.14% |
| backward.fc2.add_cast_scale.total | 0.0326 | 2.07% |
| backward.fc2.lora_B.total | 0.0941 | 5.96% |
| backward.fc2.lora_A.total | 0.0688 | 4.36% |
| backward.activation_relu.total | 0.0444 | 2.81% |
| backward.fc1.base_dx_asymgemm.total | 0.1494 | 9.47% |
| backward.fc1.base_lora_add.total | 0.0509 | 3.23% |
| backward.fc1.add_cast_scale.total | 0.0329 | 2.09% |
| backward.fc1.lora_B.total | 0.0844 | 5.35% |
| backward.fc1.lora_A.total | 0.0682 | 4.32% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 0.6889 | 43.65% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 1.5781 | 100.00% |

## Memory

| Component | bytes |
|---|---:|
| peak_hbm | 67627520 |
| gpu_parameters | 24576 |
| gpu_buffers | 0 |
| pinned_W | 262144 |
| pinned_W_T | 0 |
| pinned_total | 262144 |
