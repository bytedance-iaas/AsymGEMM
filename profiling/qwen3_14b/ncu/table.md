# NCU AsymGEMM Kernel Report: qwen3_14b

Source: `profiling/qwen3_14b/ncu/raw.csv`

## Kernel Summary

| ID | Operation | duration ms | tensor pipe % | SM throughput % | memory throughput % | DRAM % | L2 % | issue active % | active warps % | regs/thread | smem/block | replay passes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `forward.attention.q_proj.base_frozen_asymgemm` | 1.7701 | 0.2500 | 0.2500 | 6.2900 | 0.0100 | 6.2900 | 0.1200 | 7.8100 | 248.0000 | 230488.0000 | 32.0000 |
| 1 | `forward.attention.k_proj.base_frozen_asymgemm` | 1.6451 | 0.2600 | 0.2600 | 5.8500 | 0.0100 | 5.8500 | 0.1300 | 7.8100 | 248.0000 | 230488.0000 | 32.0000 |
| 2 | `forward.attention.v_proj.base_frozen_asymgemm` | 1.6286 | 0.2600 | 0.2600 | 6.7400 | 0.0100 | 6.7400 | 0.1300 | 7.8100 | 248.0000 | 230488.0000 | 32.0000 |
| 3 | `forward.attention.o_proj.base_frozen_asymgemm` | 1.6283 | 0.2600 | 0.2600 | 6.7000 | 0.0100 | 6.7000 | 0.1300 | 7.8100 | 248.0000 | 230488.0000 | 32.0000 |
| 4 | `forward.mlp.gate_proj.base_frozen_asymgemm` | 6.0791 | 0.2400 | 0.3900 | 7.2100 | 0.0000 | 7.2100 | 0.1500 | 7.8100 | 248.0000 | 230488.0000 | 32.0000 |
| 5 | `forward.mlp.up_proj.base_frozen_asymgemm` | 5.8651 | 0.2500 | 0.4000 | 7.1700 | 0.0000 | 7.1700 | 0.1600 | 7.8100 | 248.0000 | 230488.0000 | 32.0000 |
| 6 | `forward.mlp.down_proj.base_frozen_asymgemm` | 5.4940 | 0.2700 | 0.2700 | 5.8500 | 0.0100 | 5.8500 | 0.1200 | 7.8100 | 248.0000 | 230584.0000 | 32.0000 |
| 7 | `backward.mlp.down_proj.base_dx_asymgemm` | 5.4933 | 0.3000 | 0.6000 | 2.7200 | 0.0000 | 2.7200 | 0.2900 | 7.8100 | 248.0000 | 58736.0000 | 32.0000 |
| 8 | `backward.mlp.up_proj.base_dx_asymgemm` | 5.0405 | 0.3200 | 0.4900 | 2.9800 | 0.0100 | 2.9800 | 0.2800 | 7.8100 | 248.0000 | 59504.0000 | 32.0000 |
| 9 | `backward.mlp.gate_proj.base_dx_asymgemm` | 4.9790 | 0.3200 | 0.4900 | 3.0100 | 0.0100 | 3.0100 | 0.2800 | 7.8100 | 248.0000 | 59504.0000 | 32.0000 |
| 10 | `backward.attention.o_proj.base_dx_asymgemm` | 1.4819 | 0.3200 | 0.4900 | 2.9800 | 0.0100 | 2.9800 | 0.2800 | 7.8100 | 248.0000 | 58736.0000 | 32.0000 |
| 11 | `backward.attention.v_proj.base_dx_asymgemm` | 1.4838 | 0.3200 | 0.4900 | 2.9700 | 0.0100 | 2.9700 | 0.2800 | 7.8100 | 248.0000 | 58736.0000 | 32.0000 |
| 12 | `backward.attention.k_proj.base_dx_asymgemm` | 1.4947 | 0.3200 | 0.4900 | 2.9700 | 0.0100 | 2.9700 | 0.2800 | 7.8100 | 248.0000 | 58736.0000 | 32.0000 |
| 13 | `backward.attention.q_proj.base_dx_asymgemm` | 1.4920 | 0.3200 | 0.4800 | 2.9600 | 0.0100 | 2.9600 | 0.2800 | 7.8100 | 248.0000 | 58736.0000 | 32.0000 |

## Kernel 0: `forward.attention.q_proj.base_frozen_asymgemm`

`void sm90_bf16_asym_gemm_impl<0, 0, 64, 5120, 5120, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 80, 1, 0, float, ...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 548.9500 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 6.0500 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 3.2800 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.7000 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.1500 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 1.1100 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.4900 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.1400 | inst |
| `smsp__average_warps_issue_stalled_misc_per_issue_active.ratio` | 0.1400 | inst |

## Kernel 1: `forward.attention.k_proj.base_frozen_asymgemm`

`void sm90_bf16_asym_gemm_impl<0, 0, 64, 5120, 5120, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 80, 1, 0, float, ...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 531.3300 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 6.2500 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 3.2400 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.7000 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 1.3700 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.1400 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.5100 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.1300 | inst |
| `smsp__average_warps_issue_stalled_misc_per_issue_active.ratio` | 0.1300 | inst |

## Kernel 2: `forward.attention.v_proj.base_frozen_asymgemm`

`void sm90_bf16_asym_gemm_impl<0, 0, 64, 5120, 5120, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 80, 1, 0, float, ...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 545.7900 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 6.9700 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 3.2600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.7100 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 1.3000 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.1500 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.4800 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.1700 | inst |
| `smsp__average_warps_issue_stalled_misc_per_issue_active.ratio` | 0.1400 | inst |

## Kernel 3: `forward.attention.o_proj.base_frozen_asymgemm`

`void sm90_bf16_asym_gemm_impl<0, 0, 64, 5120, 5120, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 80, 1, 0, float, ...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 532.9500 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 6.6500 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 3.2600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.7000 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 1.2400 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.1500 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 0.9900 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.4800 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.1600 | inst |
| `smsp__average_warps_issue_stalled_misc_per_issue_active.ratio` | 0.1300 | inst |

## Kernel 4: `forward.mlp.gate_proj.base_frozen_asymgemm`

`void sm90_bf16_asym_gemm_impl<0, 0, 64, 17408, 5120, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 91, 1, 0, float,...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 749.9800 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 13.0400 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 4.7300 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 4.4500 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.7700 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 0.9900 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.8800 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.5000 | inst |
| `smsp__average_warps_issue_stalled_misc_per_issue_active.ratio` | 0.1900 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.0500 | inst |

## Kernel 5: `forward.mlp.up_proj.base_frozen_asymgemm`

`void sm90_bf16_asym_gemm_impl<0, 0, 64, 17408, 5120, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 91, 1, 0, float,...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 752.3500 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 11.8200 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 4.5900 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 3.6500 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.7700 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 0.9900 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.9000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.5300 | inst |
| `smsp__average_warps_issue_stalled_misc_per_issue_active.ratio` | 0.1900 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.0700 | inst |

## Kernel 6: `forward.mlp.down_proj.base_frozen_asymgemm`

`void sm90_bf16_asym_gemm_impl<0, 0, 64, 5120, 17408, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 80, 1, 0, float,...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 557.9200 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 3.3700 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 1.7700 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.7400 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.1600 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.9300 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.3900 | inst |
| `smsp__average_warps_issue_stalled_misc_per_issue_active.ratio` | 0.1400 | inst |
| `smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio` | 0.0500 | inst |

## Kernel 7: `backward.mlp.down_proj.base_dx_asymgemm`

`void sm90_bf16_asym_gemm_impl<0, 1, 64, 17408, 5120, 64, 64, 64, 1, 128, 128, 0, 2, 128, 128, 1, 0, 91, 1, 0, float, ...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 394.2100 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 2.2400 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 2.1100 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.3500 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 0.8000 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.2800 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.0800 | inst |
| `smsp__average_warps_issue_stalled_misc_per_issue_active.ratio` | 0.0700 | inst |

## Kernel 8: `backward.mlp.up_proj.base_dx_asymgemm`

`void sm90_bf16_asym_gemm_impl<0, 1, 64, 5120, 17408, 64, 64, 64, 1, 128, 128, 0, 2, 128, 128, 1, 0, 80, 1, 0, float, ...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 260.9100 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 2.2700 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5000 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 1.1700 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.2100 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 0.1600 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.0700 | inst |
| `smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio` | 0.0600 | inst |

## Kernel 9: `backward.mlp.gate_proj.base_dx_asymgemm`

`void sm90_bf16_asym_gemm_impl<0, 1, 64, 5120, 17408, 64, 64, 64, 1, 128, 128, 0, 2, 128, 128, 1, 0, 80, 1, 0, float, ...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 262.9200 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 2.2700 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5000 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 1.2600 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.2100 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 0.1700 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.0700 | inst |
| `smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio` | 0.0600 | inst |

## Kernel 10: `backward.attention.o_proj.base_dx_asymgemm`

`void sm90_bf16_asym_gemm_impl<0, 1, 64, 5120, 5120, 64, 64, 64, 1, 128, 128, 0, 2, 128, 128, 1, 0, 80, 1, 0, float, 1...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 261.5200 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 2.2700 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5100 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 1.1700 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 0.5700 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5500 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.2500 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.0900 | inst |
| `smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio` | 0.0600 | inst |

## Kernel 11: `backward.attention.v_proj.base_dx_asymgemm`

`void sm90_bf16_asym_gemm_impl<0, 1, 64, 5120, 5120, 64, 64, 64, 1, 128, 128, 0, 2, 128, 128, 1, 0, 80, 1, 0, float, 1...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 255.5100 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 2.2700 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5000 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 1.2500 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 0.5500 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5500 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.2600 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.0900 | inst |
| `smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio` | 0.0600 | inst |

## Kernel 12: `backward.attention.k_proj.base_dx_asymgemm`

`void sm90_bf16_asym_gemm_impl<0, 1, 64, 5120, 5120, 64, 64, 64, 1, 128, 128, 0, 2, 128, 128, 1, 0, 80, 1, 0, float, 1...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 255.0200 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 2.2700 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5100 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 1.1700 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.2500 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.0900 | inst |
| `smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio` | 0.0600 | inst |

## Kernel 13: `backward.attention.q_proj.base_dx_asymgemm`

`void sm90_bf16_asym_gemm_impl<0, 1, 64, 5120, 5120, 64, 64, 64, 1, 128, 128, 0, 2, 128, 128, 1, 0, 80, 1, 0, float, 1...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 259.7400 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 2.2700 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5100 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 1.2200 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 0.5300 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.2500 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.0900 | inst |
| `smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio` | 0.0500 | inst |

## tensor_core_util

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 0.7343 | 0.3700 | 0.8900 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 0.7343 | 0.3700 | 0.8900 | % |
| `sm__ops_path_tensor_op_hgmma_src_bf16_dst_fp32_sparsity_off.max.pct_of_peak_sustained_elapsed` | 0.4607 | 0.3500 | 0.5300 | % |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_elapsed` | 0.4607 | 0.3500 | 0.5300 | % |
| `sm__pipe_tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma_cycles_active.max.pct_of_peak_sustained_elapsed` | 0.4607 | 0.3500 | 0.5300 | % |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 0.4507 | 0.2500 | 0.5400 | % |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 0.4507 | 0.2500 | 0.5400 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 0.4507 | 0.2500 | 0.5400 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 0.4507 | 0.2500 | 0.5400 | % |
| `sm__ops_path_tensor_op_hgmma_src_bf16_dst_fp32_sparsity_off.avg.pct_of_peak_sustained_elapsed` | 0.2864 | 0.2400 | 0.3200 | % |
| `sm__ops_path_tensor_op_hgmma_src_bf16_dst_fp32_sparsity_off.sum.pct_of_peak_sustained_elapsed` | 0.2864 | 0.2400 | 0.3200 | % |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 0.2864 | 0.2400 | 0.3200 | % |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_elapsed` | 0.2864 | 0.2400 | 0.3200 | % |
| `sm__pipe_tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma_cycles_active.avg.pct_of_peak_sustained_elapsed` | 0.2864 | 0.2400 | 0.3200 | % |
| `sm__pipe_tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma_cycles_active.sum.pct_of_peak_sustained_elapsed` | 0.2864 | 0.2400 | 0.3200 | % |
| `sm__ops_path_tensor_src_bf16_dst_fp32.max.pct_of_peak_sustained_elapsed` | 0.2293 | 0.1700 | 0.2700 | % |
| `sm__ops_path_tensor_src_bf16_dst_fp32.avg.pct_of_peak_sustained_elapsed` | 0.1436 | 0.1200 | 0.1600 | % |
| `sm__ops_path_tensor_src_bf16_dst_fp32.sum.pct_of_peak_sustained_elapsed` | 0.1436 | 0.1200 | 0.1600 | % |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 0.0929 | 0.0500 | 0.1100 | % |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0.0579 | 0.0300 | 0.0700 | % |

## memory_throughput

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `gpu__compute_memory_throughput.max.pct_of_peak_sustained_elapsed` | 7.2307 | 3.0200 | 12.0000 | % |
| `lts__throughput.max.pct_of_peak_sustained_elapsed` | 7.2307 | 3.0200 | 12.0000 | % |
| `gpu__compute_memory_access_throughput.max.pct_of_peak_sustained_elapsed` | 6.7700 | 2.1900 | 12.0000 | % |
| `gpu__compute_memory_access_throughput_internal_activity.max.pct_of_peak_sustained_elapsed` | 5.9900 | 0.9200 | 12.0000 | % |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 4.7429 | 2.7200 | 7.2100 | % |
| `gpu__compute_memory_throughput.sum.pct_of_peak_sustained_elapsed` | 4.7429 | 2.7200 | 7.2100 | % |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | 4.7429 | 2.7200 | 7.2100 | % |
| `lts__throughput.sum.pct_of_peak_sustained_elapsed` | 4.7429 | 2.7200 | 7.2100 | % |
| `gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed` | 4.0779 | 1.5100 | 7.2100 | % |
| `gpu__compute_memory_access_throughput.sum.pct_of_peak_sustained_elapsed` | 4.0779 | 1.5100 | 7.2100 | % |
| `gpu__compute_memory_access_throughput_internal_activity.avg.pct_of_peak_sustained_elapsed` | 3.7493 | 0.8800 | 7.2100 | % |
| `gpu__compute_memory_access_throughput_internal_activity.sum.pct_of_peak_sustained_elapsed` | 3.7493 | 0.8800 | 7.2100 | % |
| `l1tex__throughput.max.pct_of_peak_sustained_active` | 2.9721 | 0.6000 | 5.3400 | % |
| `gpu__compute_memory_request_throughput.max.pct_of_peak_sustained_elapsed` | 2.4264 | 1.2300 | 3.8200 | % |
| `gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed` | 2.0914 | 1.1600 | 3.0100 | % |
| `gpu__compute_memory_request_throughput.sum.pct_of_peak_sustained_elapsed` | 2.0914 | 1.1600 | 3.0100 | % |
| `gpu__compute_memory_throughput.min.pct_of_peak_sustained_elapsed` | 2.0257 | 1.1100 | 2.8600 | % |
| `lts__throughput.min.pct_of_peak_sustained_elapsed` | 2.0257 | 1.1100 | 2.8600 | % |
| `l1tex__throughput.max.pct_of_peak_sustained_elapsed` | 1.8664 | 0.5600 | 3.1800 | % |
| `l1tex__throughput.avg.pct_of_peak_sustained_active` | 1.8229 | 0.4100 | 3.2400 | % |

## l2_dram_behavior

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `lts__t_sector_op_write_hit_rate.pct` | 102.3007 | 89.5200 | 118.7400 | % |
| `l1tex__t_sector_hit_rate.pct` | 91.1036 | 90.0000 | 95.1500 | % |
| `l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct` | 91.1036 | 90.0000 | 95.1500 | % |
| `lts__t_sector_hit_rate.pct` | 58.0571 | 42.0600 | 72.5800 | % |
| `lts__t_sector_op_read_hit_rate.pct` | 36.6464 | 31.9800 | 39.2700 | % |
| `lts__throughput.max.pct_of_peak_sustained_elapsed` | 7.2307 | 3.0200 | 12.0000 | % |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | 4.7429 | 2.7200 | 7.2100 | % |
| `lts__throughput.sum.pct_of_peak_sustained_elapsed` | 4.7429 | 2.7200 | 7.2100 | % |
| `lts__xbar2lts_cycles_active.max.pct_of_peak_sustained_elapsed` | 2.4264 | 1.2300 | 3.8200 | % |
| `lts__xbar2lts_cycles_active.avg.pct_of_peak_sustained_elapsed` | 2.0914 | 1.1600 | 3.0100 | % |
| `lts__xbar2lts_cycles_active.sum.pct_of_peak_sustained_elapsed` | 2.0914 | 1.1600 | 3.0100 | % |
| `lts__throughput.min.pct_of_peak_sustained_elapsed` | 2.0257 | 1.1100 | 2.8600 | % |
| `lts__xbar2lts_cycles_active.min.pct_of_peak_sustained_elapsed` | 1.7729 | 1.1000 | 2.4900 | % |
| `lts__t_sectors.max.pct_of_peak_sustained_elapsed` | 1.0043 | 0.5600 | 1.4500 | % |
| `lts__t_sectors_srcunit_tex.avg.pct_of_peak_sustained_elapsed` | 1.0043 | 0.5900 | 1.4100 | % |
| `lts__t_sectors_srcunit_tex.sum.pct_of_peak_sustained_elapsed` | 1.0043 | 0.5900 | 1.4100 | % |
| `lts__t_sectors.avg.pct_of_peak_sustained_elapsed` | 0.9143 | 0.5200 | 1.3000 | % |
| `lts__t_sectors.sum.pct_of_peak_sustained_elapsed` | 0.9143 | 0.5200 | 1.3000 | % |
| `lts__t_sectors.min.pct_of_peak_sustained_elapsed` | 0.8179 | 0.4500 | 1.1700 | % |
| `lts__d_atomic_input_cycles_active.max.pct_of_peak_sustained_elapsed` | 0.7750 | 0.1300 | 1.4500 | % |

## occupancy

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `derived__pct_occupancy_per_shared_mem_size` | 5460.0000 | 5460.0000 | 5460.0000 | %/byte |
| `derived__pct_occupancy_per_register_count` | 3084.0000 | 3084.0000 | 3084.0000 | %/register |
| `derived__pct_occupancy_per_barrier_count` | 780.0000 | 780.0000 | 780.0000 |  |
| `derived__pct_occupancy_per_block_size` | 105.0000 | 105.0000 | 105.0000 | % |
| `sm__maximum_warps_per_active_cycle_pct` | 12.5000 | 12.5000 | 12.5000 | % |
| `sm__warps_active.max.pct_of_peak_sustained_active` | 12.2900 | 8.0600 | 14.0100 | % |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 7.8100 | 7.8100 | 7.8100 | % |
| `sm__warps_active.sum.pct_of_peak_sustained_active` | 7.8100 | 7.8100 | 7.8100 | % |
| `sm__warps_active.min.pct_of_peak_sustained_active` | 1.4771 | 0.0000 | 7.6800 | % |
| `launch__occupancy_cluster_gpu_pct` | 0.0000 | 0.0000 | 0.0000 | % |
| `launch__occupancy_cluster_pct` | 0.0000 | 0.0000 | 0.0000 | % |
| `smsp__warps_active.sum.peak_sustained` | 8448.0000 | 8448.0000 | 8448.0000 | warp |
| `launch__occupancy_per_shared_mem_size` | 3640.0000 | 3640.0000 | 3640.0000 |  |
| `launch__occupancy_per_register_count` | 2056.0000 | 2056.0000 | 2056.0000 |  |
| `smsp__warps_active.sum.per_cycle_active` | 662.2671 | 645.6300 | 680.1700 | warp |
| `sm__warps_active.sum.per_cycle_active` | 660.0143 | 659.5200 | 660.1600 | warp |
| `launch__occupancy_per_barrier_count` | 520.0000 | 520.0000 | 520.0000 |  |
| `smsp__average_warps_active_per_inst_executed.ratio` | 486.4657 | 269.9600 | 880.7400 | cycle |
| `launch__occupancy_per_block_size` | 73.0000 | 73.0000 | 73.0000 |  |
| `launch__occupancy_limit_blocks` | 32.0000 | 32.0000 | 32.0000 | block |

## warp_stall_reasons

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 440.6500 | 255.0200 | 752.3500 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 3.9921 | 0.1600 | 13.0400 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 2.5057 | 1.1700 | 4.7300 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.9964 | 1.7000 | 2.2700 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.2793 | 0.8800 | 1.5100 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 1.2779 | 0.5000 | 4.4500 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 0.9979 | 0.9900 | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.3636 | 0.2100 | 0.5300 | inst |
| `smsp__average_warps_issue_stalled_misc_per_issue_active.ratio` | 0.0950 | 0.0300 | 0.1900 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.0679 | 0.0200 | 0.0900 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.0671 | 0.0100 | 0.1700 | inst |
| `smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio` | 0.0286 | 0.0000 | 0.0600 | inst |
| `smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio` | 0.0236 | 0.0000 | 0.0500 | inst |
| `smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio` | 0.0014 | 0.0000 | 0.0100 | inst |
| `smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio` | 0.0000 | 0.0000 | 0.0000 | inst |
| `smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio` | 0.0000 | 0.0000 | 0.0000 | inst |
| `smsp__average_warps_issue_stalled_membar_per_issue_active.ratio` | 0.0000 | 0.0000 | 0.0000 | inst |
| `smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio` | 0.0000 | 0.0000 | 0.0000 | inst |
| `smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio` | 0.0000 | 0.0000 | 0.0000 | inst |

## scheduler_stats

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `smsp__issue_inst0.max.pct_of_peak_sustained_active` | 156.6857 | 104.1400 | 178.5300 | % |
| `smsp__issue_inst0.avg.pct_of_peak_sustained_active` | 99.6750 | 99.5300 | 99.8400 | % |
| `smsp__issue_inst0.sum.pct_of_peak_sustained_active` | 99.6750 | 99.5300 | 99.8400 | % |
| `smsp__issue_inst0.min.pct_of_peak_sustained_active` | 19.0214 | 0.0000 | 98.1600 | % |
| `smsp__issue_active.max.pct_of_peak_sustained_active` | 0.7229 | 0.2100 | 1.1100 | % |
| `sm__inst_issued.max.pct_of_peak_sustained_active` | 0.5264 | 0.2000 | 0.7900 | % |
| `sm__issue_active.max.pct_of_peak_sustained_elapsed` | 0.3314 | 0.1900 | 0.4800 | % |
| `sm__inst_issued.avg.pct_of_peak_sustained_active` | 0.3250 | 0.1600 | 0.4700 | % |
| `sm__inst_issued.sum.pct_of_peak_sustained_active` | 0.3250 | 0.1600 | 0.4700 | % |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | 0.3250 | 0.1600 | 0.4700 | % |
| `smsp__issue_active.sum.pct_of_peak_sustained_active` | 0.3250 | 0.1600 | 0.4700 | % |
| `sm__issue_active.avg.pct_of_peak_sustained_elapsed` | 0.2079 | 0.1200 | 0.2900 | % |
| `sm__issue_active.sum.pct_of_peak_sustained_elapsed` | 0.2079 | 0.1200 | 0.2900 | % |
| `sm__mio_inst_issued.max.pct_of_peak_sustained_elapsed` | 0.1821 | 0.0900 | 0.2800 | % |
| `sm__mio_inst_issued.avg.pct_of_peak_sustained_elapsed` | 0.1143 | 0.0500 | 0.1800 | % |
| `sm__mio_inst_issued.sum.pct_of_peak_sustained_elapsed` | 0.1143 | 0.0500 | 0.1800 | % |
| `sm__inst_issued.min.pct_of_peak_sustained_active` | 0.0421 | 0.0000 | 0.3000 | % |
| `sm__issue_active.min.pct_of_peak_sustained_elapsed` | 0.0400 | 0.0000 | 0.2800 | % |
| `smsp__issue_active.min.pct_of_peak_sustained_active` | 0.0371 | 0.0000 | 0.2700 | % |
| `sm__mio_inst_issued.min.pct_of_peak_sustained_elapsed` | 0.0214 | 0.0000 | 0.1700 | % |

## roofline_position

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 0.7343 | 0.3700 | 0.8900 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 0.4507 | 0.2500 | 0.5400 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 0.4507 | 0.2500 | 0.5400 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 0.0571 | 0.0000 | 0.3000 | % |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 0.0464 | 0.0200 | 0.0600 | % |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 0.0279 | 0.0200 | 0.0300 | % |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 0.0279 | 0.0200 | 0.0300 | % |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 0.0043 | 0.0000 | 0.0200 | % |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.avg.pct_of_peak_sustained_elapsed` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.max.pct_of_peak_sustained_elapsed` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.min.pct_of_peak_sustained_elapsed` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.sum.pct_of_peak_sustained_elapsed` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_on.avg.pct_of_peak_sustained_elapsed` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_on.max.pct_of_peak_sustained_elapsed` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_on.min.pct_of_peak_sustained_elapsed` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_on.sum.pct_of_peak_sustained_elapsed` | 0.0000 | 0.0000 | 0.0000 | % |

## launch_stats

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `launch__occupancy_cluster_gpu_pct` | 0.0000 | 0.0000 | 0.0000 | % |
| `launch__occupancy_cluster_pct` | 0.0000 | 0.0000 | 0.0000 | % |
| `launch__shared_mem_config_size` | 149504.0000 | 65536.0000 | 233472.0000 | byte |
| `launch__shared_mem_per_block_allocated` | 144758.8571 | 58752.0000 | 230656.0000 | byte/block |
| `launch__shared_mem_per_block` | 144728.5714 | 58736.0000 | 230584.0000 | byte/block |
| `launch__shared_mem_per_block_dynamic` | 143704.5714 | 57712.0000 | 229560.0000 | byte/block |
| `launch__thread_count` | 31012.5714 | 20480.0000 | 69632.0000 | thread |
| `launch__occupancy_per_shared_mem_size` | 3640.0000 | 3640.0000 | 3640.0000 |  |
| `launch__occupancy_per_register_count` | 2056.0000 | 2056.0000 | 2056.0000 |  |
| `launch__shared_mem_per_block_driver` | 1024.0000 | 1024.0000 | 1024.0000 | byte/block |
| `launch__stack_size` | 1024.0000 | 1024.0000 | 1024.0000 |  |
| `launch__occupancy_per_barrier_count` | 520.0000 | 520.0000 | 520.0000 |  |
| `launch__block_dim_x` | 256.0000 | 256.0000 | 256.0000 | block |
| `launch__block_size` | 256.0000 | 256.0000 | 256.0000 |  |
| `launch__registers_per_thread` | 248.0000 | 248.0000 | 248.0000 | register/thread |
| `launch__registers_per_thread_allocated` | 248.0000 | 248.0000 | 248.0000 | register/thread |
| `launch__sm_count` | 132.0000 | 132.0000 | 132.0000 | SM |
| `launch__grid_dim_x` | 121.1429 | 80.0000 | 272.0000 |  |
| `launch__grid_size` | 121.1429 | 80.0000 | 272.0000 |  |
| `launch__occupancy_per_block_size` | 73.0000 | 73.0000 | 73.0000 |  |
