# matrix_1b Source-Label Coverage Report

This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_m4.py` for kernel-busy, memcpy, and GPU no-kernel percentages.

These source-label tables are for auditing range coverage. In `--timing-mode profile`, inner rows are asynchronous Python/NVTX range timings. In `--timing-mode debug_sync`, rows are synchronized debugging timings and carry profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 0.5322 | 0.00% |
| step.forward | 6715.7307 | 55.38% |
| step.loss | 1.0331 | 0.01% |
| step.backward | 5409.1815 | 44.61% |
| step.optimizer | 0.1307 | 0.00% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 12126.6082 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.matrix.base_frozen_asymgemm | 6715.5761 | 100.00% |
| forward.python_dispatch_cuda_launch_and_sync | 0.1546 | 0.00% |
| **sum** | 6715.7307 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.mse.total | 0.4727 | 0.01% |
| backward.matrix.base_dx_asymgemm.total | 5406.8030 | 99.96% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 1.9058 | 0.04% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 5409.1815 | 100.00% |

## Memory

| Component | bytes |
|---|---:|
| peak_hbm | 54528512 |
| gpu_parameters | 0 |
| gpu_buffers | 0 |
| host_W | 2147483648 |
| host_W_T | 0 |
| pinned_W | 2147483648 |
| pinned_W_T | 0 |
| pinned_total | 2147483648 |
