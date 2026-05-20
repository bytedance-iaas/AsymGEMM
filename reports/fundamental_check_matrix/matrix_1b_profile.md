# matrix_1b Source-Label Coverage Report

This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_m4.py` for kernel-busy, memcpy, and GPU no-kernel percentages.

These source-label tables are for auditing range coverage. In `--timing-mode profile`, inner rows are asynchronous Python/NVTX range timings. In `--timing-mode debug_sync`, rows are synchronized debugging timings and carry profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 2.9443 | 0.06% |
| step.forward | 2572.4255 | 50.37% |
| step.loss | 51.2906 | 1.00% |
| step.backward | 2480.4642 | 48.57% |
| step.optimizer | 0.0835 | 0.00% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 5107.2081 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.matrix.base_frozen_asymgemm | 2572.3260 | 100.00% |
| forward.python_dispatch_cuda_launch_and_sync | 0.0995 | 0.00% |
| **sum** | 2572.4255 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.mse.total | 6.6625 | 0.27% |
| backward.matrix.base_dx_asymgemm.total | 2393.8411 | 96.51% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 79.9607 | 3.22% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 2480.4642 | 100.00% |

## Memory

| Component | bytes |
|---|---:|
| peak_hbm | 46139392 |
| gpu_parameters | 0 |
| gpu_buffers | 0 |
| pinned_W | 2147483648 |
| pinned_W_T | 0 |
| pinned_total | 2147483648 |
