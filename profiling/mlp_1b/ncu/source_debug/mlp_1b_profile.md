# mlp_1b Source-Label Coverage Report

This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_m4.py` for kernel-busy, memcpy, and GPU no-kernel percentages.

These source-label tables are for auditing range coverage. In `--timing-mode profile`, inner rows are asynchronous Python/NVTX range timings. In `--timing-mode debug_sync`, rows are synchronized debugging timings and carry profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 0.8521 | 0.00% |
| step.forward | 9397.0439 | 54.78% |
| step.loss | 1.0076 | 0.01% |
| step.backward | 7753.7841 | 45.20% |
| step.optimizer | 0.1037 | 0.00% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 17152.7915 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.fc1.base_frozen_asymgemm | 5294.8328 | 56.35% |
| forward.activation_relu | 0.4304 | 0.00% |
| forward.fc2.base_frozen_asymgemm | 4101.5634 | 43.65% |
| forward.python_dispatch_cuda_launch_and_sync | 0.2173 | 0.00% |
| **sum** | 9397.0439 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.mse.total | 0.5066 | 0.01% |
| backward.fc2.base_dx_asymgemm.total | 4018.0731 | 51.82% |
| backward.activation_relu.total | 0.4237 | 0.01% |
| backward.fc1.base_dx_asymgemm.total | 3732.4568 | 48.14% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 2.3239 | 0.03% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 7753.7841 | 100.00% |

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
