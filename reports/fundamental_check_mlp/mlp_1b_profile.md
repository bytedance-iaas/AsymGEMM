# mlp_1b Source-Label Coverage Report

This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_m4.py` for kernel-busy, memcpy, and GPU no-kernel percentages.

These source-label tables are for auditing range coverage. In `--timing-mode profile`, inner rows are asynchronous Python/NVTX range timings. In `--timing-mode debug_sync`, rows are synchronized debugging timings and carry profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 2.6931 | 0.03% |
| step.forward | 5080.1237 | 50.62% |
| step.loss | 51.8027 | 0.52% |
| step.backward | 4901.9870 | 48.84% |
| step.optimizer | 0.0634 | 0.00% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 10036.6699 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.fc1.base_frozen_asymgemm | 2487.7600 | 48.97% |
| forward.activation_relu | 15.0156 | 0.30% |
| forward.fc2.base_frozen_asymgemm | 2534.7439 | 49.90% |
| forward.python_dispatch_cuda_launch_and_sync | 42.6042 | 0.84% |
| **sum** | 5080.1237 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.mse.total | 6.7515 | 0.14% |
| backward.fc2.base_dx_asymgemm.total | 2374.9203 | 48.45% |
| backward.activation_relu.total | 40.9153 | 0.83% |
| backward.fc1.base_dx_asymgemm.total | 2442.2263 | 49.82% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 37.1736 | 0.76% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 4901.9870 | 100.00% |

## Memory

| Component | bytes |
|---|---:|
| peak_hbm | 37750784 |
| gpu_parameters | 0 |
| gpu_buffers | 0 |
| pinned_W | 2147483648 |
| pinned_W_T | 0 |
| pinned_total | 2147483648 |
