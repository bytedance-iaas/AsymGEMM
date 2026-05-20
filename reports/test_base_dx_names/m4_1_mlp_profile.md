# m4_1_mlp Source-Label Coverage Report

This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_m4.py` for kernel-busy, memcpy, and GPU no-kernel percentages.

These source-label tables are for auditing range coverage. In `--timing-mode profile`, inner rows are asynchronous Python/NVTX range timings. In `--timing-mode debug_sync`, rows are synchronized debugging timings and carry profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 1.4681 | 12.88% |
| step.forward | 3.6969 | 32.44% |
| step.loss | 0.4514 | 3.96% |
| step.backward | 4.9610 | 43.54% |
| step.optimizer | 0.8174 | 7.17% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 11.3948 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.fc1.base_frozen_asymgemm | 1.7495 | 47.32% |
| forward.fc1.lora_A | 0.1792 | 4.85% |
| forward.fc1.lora_B | 0.0827 | 2.24% |
| forward.fc1.add_cast_scale | 0.1363 | 3.69% |
| forward.activation_relu | 0.0510 | 1.38% |
| forward.fc2.base_frozen_asymgemm | 0.9478 | 25.64% |
| forward.fc2.lora_A | 0.1131 | 3.06% |
| forward.fc2.lora_B | 0.0580 | 1.57% |
| forward.fc2.add_cast_scale | 0.1247 | 3.37% |
| forward.python_dispatch_cuda_launch_and_sync | 0.2544 | 6.88% |
| **sum** | 3.6969 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.mse.total | 0.2700 | 5.44% |
| backward.fc2.base_dx_asymgemm.total | 0.0000 | 0.00% |
| backward.fc2.base_lora_add.total | 0.1074 | 2.16% |
| backward.fc2.add_cast_scale.total | 0.0665 | 1.34% |
| backward.fc2.lora_B.total | 0.2181 | 4.40% |
| backward.fc2.lora_A.total | 0.1089 | 2.19% |
| backward.activation_relu.total | 0.1163 | 2.34% |
| backward.fc1.base_dx_asymgemm.total | 0.0000 | 0.00% |
| backward.fc1.base_lora_add.total | 0.0707 | 1.42% |
| backward.fc1.add_cast_scale.total | 0.0397 | 0.80% |
| backward.fc1.lora_B.total | 0.1607 | 3.24% |
| backward.fc1.lora_A.total | 0.0931 | 1.88% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 3.7096 | 74.78% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 4.9610 | 100.00% |

## Memory

| Component | bytes |
|---|---:|
| peak_hbm | 67627520 |
| gpu_parameters | 24576 |
| gpu_buffers | 0 |
| pinned_W | 262144 |
| pinned_W_T | 0 |
| pinned_total | 262144 |
