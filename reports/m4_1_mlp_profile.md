# m4_1_mlp Source-Label Coverage Report

This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_m4.py` for kernel-busy, memcpy, and GPU no-kernel percentages.

These source-label tables are for auditing range coverage. In `--timing-mode profile`, inner rows are asynchronous Python/NVTX range timings. In `--timing-mode debug_sync`, rows are synchronized debugging timings and carry profiler overhead.

## Step

| Component | ms | % |
|---|---:|---:|
| step.input_preparation | 0.0452 | 1.50% |
| step.forward | 1.1727 | 39.02% |
| step.loss | 0.1009 | 3.36% |
| step.backward | 1.4392 | 47.89% |
| step.optimizer | 0.2471 | 8.22% |
| step.accounting_gap | 0.0000 | 0.00% |
| **sum** | 3.0052 | 100.00% |

## Forward

| Component | ms | % |
|---|---:|---:|
| forward.fc1.base_frozen_asymgemm | 0.2631 | 22.43% |
| forward.fc1.lora_A | 0.0911 | 7.77% |
| forward.fc1.lora_B | 0.0578 | 4.92% |
| forward.fc1.add_cast_scale | 0.1035 | 8.83% |
| forward.activation_relu | 0.0369 | 3.15% |
| forward.fc2.base_frozen_asymgemm | 0.2216 | 18.89% |
| forward.fc2.lora_A | 0.0714 | 6.09% |
| forward.fc2.lora_B | 0.0451 | 3.84% |
| forward.fc2.add_cast_scale | 0.0919 | 7.84% |
| forward.python_dispatch_cuda_launch_and_sync | 0.1904 | 16.24% |
| **sum** | 1.1727 | 100.00% |

## Backward

| Component | ms | % |
|---|---:|---:|
| backward.loss.mse.total | 0.0541 | 3.76% |
| backward.base_dx_asymgemm.total | 0.2580 | 17.93% |
| backward.fc2.base_lora_add.total | 0.0406 | 2.82% |
| backward.fc2.add_cast_scale.total | 0.0292 | 2.03% |
| backward.fc2.lora_B.total | 0.0881 | 6.12% |
| backward.fc2.lora_A.total | 0.0647 | 4.49% |
| backward.activation_relu.total | 0.0411 | 2.86% |
| backward.fc1.base_lora_add.total | 0.0392 | 2.73% |
| backward.fc1.add_cast_scale.total | 0.0283 | 1.97% |
| backward.fc1.lora_B.total | 0.0763 | 5.30% |
| backward.fc1.lora_A.total | 0.0624 | 4.34% |
| backward.python_autograd_dispatch_cuda_launch_and_sync | 0.6572 | 45.66% |
| backward.accounting_gap | 0.0000 | 0.00% |
| **sum** | 1.4392 | 100.00% |

## Memory

| Component | bytes |
|---|---:|
| peak_hbm | 67627520 |
| gpu_parameters | 24576 |
| gpu_buffers | 0 |
| pinned_W | 262144 |
| pinned_W_T | 0 |
| pinned_total | 262144 |
