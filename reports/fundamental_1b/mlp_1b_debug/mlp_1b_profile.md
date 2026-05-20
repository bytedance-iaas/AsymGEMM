# mlp_1b Source-Label Coverage Report

This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_m4.py` for kernel-busy, memcpy, and GPU no-kernel percentages.

These source-label tables are for auditing range coverage. In `--timing-mode profile`, inner rows are asynchronous Python/NVTX range timings. In `--timing-mode debug_sync`, rows are synchronized debugging timings and carry profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 0.2360 | 0.16% |
| step.forward | 76.5560 | 52.38% |
| step.loss | 0.1617 | 0.11% |
| step.backward | 69.1705 | 47.33% |
| step.optimizer | 0.0283 | 0.02% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 146.1525 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.fc1.base_frozen_asymgemm | 0.3126 | 0.41% |
| forward.activation_relu | 0.0633 | 0.08% |
| forward.fc2.base_frozen_asymgemm | 33.8258 | 44.18% |
| forward.python_dispatch_cuda_launch_and_sync | 42.3544 | 55.32% |
| **sum** | 76.5560 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.mse.total | 0.0771 | 0.11% |
| backward.fc2.base_dx_asymgemm.total | 0.1492 | 0.22% |
| backward.activation_relu.total | 0.0430 | 0.06% |
| backward.fc1.base_dx_asymgemm.total | 40.4444 | 58.47% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 28.4568 | 41.14% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 69.1705 | 100.00% |

## Memory

| Component | bytes |
|---|---:|
| peak_hbm | 37750784 |
| gpu_parameters | 0 |
| gpu_buffers | 0 |
| host_W | 2147483648 |
| host_W_T | 0 |
| pinned_W | 2147483648 |
| pinned_W_T | 0 |
| pinned_total | 2147483648 |
