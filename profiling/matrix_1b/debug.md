# matrix_1b Source-Label Coverage Report

This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_m4.py` for kernel-busy, memcpy, and GPU no-kernel percentages.

These source-label tables are for auditing range coverage. In `--timing-mode profile`, inner rows are asynchronous Python/NVTX range timings. In `--timing-mode debug_sync`, rows are synchronized debugging timings and carry profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 0.0895 | 0.06% |
| step.forward | 72.1443 | 49.18% |
| step.loss | 0.2887 | 0.20% |
| step.backward | 74.1581 | 50.55% |
| step.optimizer | 0.0209 | 0.01% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 146.7017 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.matrix.base_frozen_asymgemm | 0.3193 | 0.44% |
| forward.python_dispatch_cuda_launch_and_sync | 71.8250 | 99.56% |
| **sum** | 72.1443 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.mse.total | 0.1019 | 0.14% |
| backward.matrix.base_dx_asymgemm.total | 0.1671 | 0.23% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 73.8891 | 99.64% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 74.1581 | 100.00% |

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
