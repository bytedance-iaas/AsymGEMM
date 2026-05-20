# NCU AsymGEMM Kernel Report: qwen3_30b_a3b

Source: `profiling/qwen3_30b_a3b/ncu/raw.csv`

## Kernel Summary

| ID | Operation | duration ms | tensor pipe % | SM throughput % | memory throughput % | DRAM % | L2 % | issue active % | active warps % | regs/thread | smem/block | replay passes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `asymgemm_kernel_0` | 0.0964 | 0.2600 | 0.2600 | 1.0600 | 0.0100 | 1.0600 | 0.1000 | 7.8500 | 248.0000 | 230464.0000 | 32.0000 |
| 1 | `asymgemm_kernel_1` | 0.0998 | 0.2700 | 0.2700 | 1.0300 | 0.0100 | 1.0300 | 0.1000 | 7.8500 | 248.0000 | 230464.0000 | 32.0000 |
| 2 | `asymgemm_kernel_2` | 0.3595 | 0.0900 | 0.0900 | 1.6200 | 0.0000 | 1.6200 | 0.0500 | 7.8200 | 248.0000 | 230456.0000 | 32.0000 |
| 3 | `asymgemm_kernel_3` | 0.0976 | 0.2700 | 0.2700 | 1.0500 | 0.0100 | 1.0500 | 0.1000 | 7.8600 | 248.0000 | 230464.0000 | 32.0000 |
| 4 | `asymgemm_kernel_4` | 0.0975 | 0.2700 | 0.2700 | 1.0500 | 0.0100 | 1.0500 | 0.1000 | 7.8700 | 248.0000 | 230464.0000 | 32.0000 |
| 5 | `asymgemm_kernel_5` | 0.4146 | 0.0800 | 0.0800 | 1.5000 | 0.0000 | 1.5000 | 0.0500 | 7.8200 | 248.0000 | 230456.0000 | 32.0000 |
| 6 | `asymgemm_kernel_6` | 0.0967 | 0.2600 | 0.2600 | 1.0700 | 0.0100 | 1.0700 | 0.1000 | 7.8500 | 248.0000 | 230464.0000 | 32.0000 |
| 7 | `asymgemm_kernel_7` | 0.1102 | 0.2700 | 0.2700 | 0.9400 | 0.0100 | 0.9400 | 0.1000 | 7.8500 | 248.0000 | 230464.0000 | 32.0000 |
| 8 | `asymgemm_kernel_8` | 0.4062 | 0.0800 | 0.0800 | 2.0600 | 0.0000 | 2.0600 | 0.0400 | 7.8300 | 248.0000 | 230456.0000 | 32.0000 |
| 9 | `asymgemm_kernel_9` | 0.0964 | 0.2700 | 0.2700 | 1.0800 | 0.0100 | 1.0800 | 0.1000 | 7.8600 | 248.0000 | 230464.0000 | 32.0000 |
| 10 | `asymgemm_kernel_10` | 0.0967 | 0.2600 | 0.2600 | 1.0700 | 0.0100 | 1.0700 | 0.1000 | 7.8500 | 248.0000 | 230464.0000 | 32.0000 |
| 11 | `asymgemm_kernel_11` | 0.4405 | 0.0900 | 0.0900 | 1.4400 | 0.0000 | 1.4400 | 0.0500 | 7.8200 | 248.0000 | 230456.0000 | 32.0000 |
| 12 | `asymgemm_kernel_12` | 0.0957 | 0.2700 | 0.2700 | 1.0900 | 0.0100 | 1.0900 | 0.1000 | 7.8600 | 248.0000 | 230464.0000 | 32.0000 |
| 13 | `asymgemm_kernel_13` | 0.0968 | 0.2700 | 0.2700 | 1.0800 | 0.0100 | 1.0800 | 0.1000 | 7.8500 | 248.0000 | 230464.0000 | 32.0000 |
| 14 | `asymgemm_kernel_14` | 0.4175 | 0.1000 | 0.1000 | 2.0000 | 0.0000 | 2.0000 | 0.0500 | 7.8300 | 248.0000 | 230456.0000 | 32.0000 |
| 15 | `asymgemm_kernel_15` | 0.0972 | 0.2600 | 0.2600 | 1.0800 | 0.0100 | 1.0800 | 0.1000 | 7.8500 | 248.0000 | 230464.0000 | 32.0000 |
| 16 | `asymgemm_kernel_16` | 0.0978 | 0.2700 | 0.2700 | 1.0700 | 0.0100 | 1.0700 | 0.1000 | 7.8500 | 248.0000 | 230464.0000 | 32.0000 |
| 17 | `asymgemm_kernel_17` | 0.4535 | 0.0600 | 0.0700 | 2.1000 | 0.0000 | 2.1000 | 0.0400 | 7.8200 | 248.0000 | 230456.0000 | 32.0000 |
| 18 | `asymgemm_kernel_18` | 0.0976 | 0.2600 | 0.2600 | 1.0800 | 0.0100 | 1.0800 | 0.1000 | 7.8600 | 248.0000 | 230464.0000 | 32.0000 |
| 19 | `asymgemm_kernel_19` | 0.1130 | 0.2700 | 0.2700 | 0.9300 | 0.0100 | 0.9300 | 0.1000 | 7.8500 | 248.0000 | 230464.0000 | 32.0000 |
| 20 | `asymgemm_kernel_20` | 0.3293 | 0.0900 | 0.0900 | 1.7900 | 0.0000 | 1.7900 | 0.0500 | 7.8300 | 248.0000 | 230456.0000 | 32.0000 |
| 21 | `asymgemm_kernel_21` | 0.0987 | 0.2700 | 0.2700 | 1.0700 | 0.0100 | 1.0700 | 0.1000 | 7.8500 | 248.0000 | 230464.0000 | 32.0000 |
| 22 | `asymgemm_kernel_22` | 0.0989 | 0.2700 | 0.2700 | 1.0700 | 0.0100 | 1.0700 | 0.1000 | 7.8600 | 248.0000 | 230464.0000 | 32.0000 |
| 23 | `asymgemm_kernel_23` | 0.4583 | 0.0700 | 0.0700 | 2.0300 | 0.0000 | 2.0300 | 0.0400 | 7.8200 | 248.0000 | 230456.0000 | 32.0000 |
| 24 | `asymgemm_kernel_24` | 0.0978 | 0.2700 | 0.2700 | 1.0800 | 0.0100 | 1.0800 | 0.1000 | 7.8600 | 248.0000 | 230464.0000 | 32.0000 |
| 25 | `asymgemm_kernel_25` | 0.1008 | 0.2600 | 0.2600 | 1.0500 | 0.0100 | 1.0500 | 0.1000 | 7.8500 | 248.0000 | 230464.0000 | 32.0000 |
| 26 | `asymgemm_kernel_26` | 0.3897 | 0.0800 | 0.0800 | 2.0700 | 0.0000 | 2.0700 | 0.0500 | 7.8200 | 248.0000 | 230456.0000 | 32.0000 |
| 27 | `asymgemm_kernel_27` | 0.0982 | 0.2500 | 0.2500 | 1.0700 | 0.0100 | 1.0700 | 0.0900 | 7.8500 | 248.0000 | 230464.0000 | 32.0000 |
| 28 | `asymgemm_kernel_28` | 0.0988 | 0.2700 | 0.2700 | 1.0700 | 0.0100 | 1.0700 | 0.1000 | 7.8500 | 248.0000 | 230464.0000 | 32.0000 |
| 29 | `asymgemm_kernel_29` | 0.4091 | 0.0900 | 0.0900 | 2.0500 | 0.0000 | 2.0500 | 0.0500 | 7.8200 | 248.0000 | 230456.0000 | 32.0000 |
| 30 | `asymgemm_kernel_30` | 0.0973 | 0.2700 | 0.2700 | 1.0600 | 0.0100 | 1.0600 | 0.1000 | 7.8500 | 248.0000 | 230464.0000 | 32.0000 |
| 31 | `asymgemm_kernel_31` | 0.0980 | 0.2700 | 0.2700 | 1.0800 | 0.0100 | 1.0800 | 0.1000 | 7.8500 | 248.0000 | 230464.0000 | 32.0000 |

## Kernel 0: `asymgemm_kernel_0`

`void sm90_bf16_asym_gemm_impl<0, 0, 1, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 90.8800 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 3.9500 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5200 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.5000 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4600 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.3200 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.1700 | inst |

## Kernel 1: `asymgemm_kernel_1`

`void sm90_bf16_asym_gemm_impl<0, 0, 1, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 91.9500 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 4.5000 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5300 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.5700 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.4900 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4400 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.1800 | inst |

## Kernel 2: `asymgemm_kernel_2`

`void sm90_bf16_asym_gemm_impl<0, 0, 1, 2048, 1024, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 32, 1, 0, float, 1...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 408.5200 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 72.8400 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 2.3100 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5800 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 1.2000 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.1300 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0100 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.5800 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.3800 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.2400 | inst |

## Kernel 3: `asymgemm_kernel_3`

`void sm90_bf16_asym_gemm_impl<0, 0, 2, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 92.4600 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 3.4200 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5800 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5300 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.7700 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4700 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.2700 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.1700 | inst |

## Kernel 4: `asymgemm_kernel_4`

`void sm90_bf16_asym_gemm_impl<0, 0, 2, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 91.4400 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 4.0700 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5200 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.6900 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4500 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.3400 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.1800 | inst |

## Kernel 5: `asymgemm_kernel_5`

`void sm90_bf16_asym_gemm_impl<0, 0, 2, 2048, 1024, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 32, 1, 0, float, 1...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 401.7900 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 93.2900 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 2.8100 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.6400 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.1200 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 1.0300 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0100 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.3700 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.2500 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.1600 | inst |

## Kernel 6: `asymgemm_kernel_6`

`void sm90_bf16_asym_gemm_impl<0, 0, 3, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 93.1500 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 3.4200 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5700 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5300 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 1.1800 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4600 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.2300 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.1800 | inst |

## Kernel 7: `asymgemm_kernel_7`

`void sm90_bf16_asym_gemm_impl<0, 0, 3, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 91.3000 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 4.0400 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5300 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.9900 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4500 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.3000 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.1800 | inst |

## Kernel 8: `asymgemm_kernel_8`

`void sm90_bf16_asym_gemm_impl<0, 0, 3, 2048, 1024, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 32, 1, 0, float, 1...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 410.0200 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 84.9900 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 2.5900 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5300 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.0800 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 1.0600 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 0.9800 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.4400 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.3600 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.2300 | inst |

## Kernel 9: `asymgemm_kernel_9`

`void sm90_bf16_asym_gemm_impl<0, 0, 4, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 91.3000 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 4.0600 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5300 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.5000 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.4800 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4500 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.1700 | inst |

## Kernel 10: `asymgemm_kernel_10`

`void sm90_bf16_asym_gemm_impl<0, 0, 4, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 91.5600 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 3.7700 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5300 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.4800 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4600 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.4500 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.1900 | inst |

## Kernel 11: `asymgemm_kernel_11`

`void sm90_bf16_asym_gemm_impl<0, 0, 4, 2048, 1024, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 32, 1, 0, float, 1...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 433.4600 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 78.9100 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 2.4800 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5800 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.1200 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.8600 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.3700 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.2700 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.2400 | inst |

## Kernel 12: `asymgemm_kernel_12`

`void sm90_bf16_asym_gemm_impl<0, 0, 5, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 92.6000 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 4.0000 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5200 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.5700 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.4700 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4500 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.1700 | inst |

## Kernel 13: `asymgemm_kernel_13`

`void sm90_bf16_asym_gemm_impl<0, 0, 5, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 91.6400 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 4.1000 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5300 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.5800 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.4700 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4500 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.1900 | inst |

## Kernel 14: `asymgemm_kernel_14`

`void sm90_bf16_asym_gemm_impl<0, 0, 5, 2048, 1024, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 32, 1, 0, float, 1...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 384.8700 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 74.0600 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 2.3300 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5900 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.1000 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 0.9800 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.7700 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.4900 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.3700 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.2400 | inst |

## Kernel 15: `asymgemm_kernel_15`

`void sm90_bf16_asym_gemm_impl<0, 0, 6, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 94.2300 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 3.6500 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5200 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.4900 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4700 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.3900 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.2000 | inst |

## Kernel 16: `asymgemm_kernel_16`

`void sm90_bf16_asym_gemm_impl<0, 0, 6, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 91.3700 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 2.7400 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5200 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.5400 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4800 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.2400 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.1700 | inst |

## Kernel 17: `asymgemm_kernel_17`

`void sm90_bf16_asym_gemm_impl<0, 0, 6, 2048, 1024, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 32, 1, 0, float, 1...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 539.8800 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 84.8400 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 2.5500 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.6100 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.0100 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.7900 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.4600 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.3400 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.2200 | inst |

## Kernel 18: `asymgemm_kernel_18`

`void sm90_bf16_asym_gemm_impl<0, 0, 7, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 91.2000 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 3.1700 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5700 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5300 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.8600 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4700 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.1700 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.1500 | inst |

## Kernel 19: `asymgemm_kernel_19`

`void sm90_bf16_asym_gemm_impl<0, 0, 7, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 91.8700 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 2.7700 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5300 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.5500 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.4900 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4800 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.1700 | inst |

## Kernel 20: `asymgemm_kernel_20`

`void sm90_bf16_asym_gemm_impl<0, 0, 7, 2048, 1024, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 32, 1, 0, float, 1...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 359.7500 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 65.6500 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 2.1600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5400 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.1300 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 1.0400 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 0.9800 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.3800 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.3400 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.2500 | inst |

## Kernel 21: `asymgemm_kernel_21`

`void sm90_bf16_asym_gemm_impl<0, 0, 8, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 109.6300 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 3.5700 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5200 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.7900 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4700 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.2700 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.1700 | inst |

## Kernel 22: `asymgemm_kernel_22`

`void sm90_bf16_asym_gemm_impl<0, 0, 8, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 94.3100 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 3.0600 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5200 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.7700 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4600 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.3300 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.1700 | inst |

## Kernel 23: `asymgemm_kernel_23`

`void sm90_bf16_asym_gemm_impl<0, 0, 8, 2048, 1024, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 32, 1, 0, float, 1...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 523.6700 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 105.7600 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 3.0700 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.6500 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.0500 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.9200 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.3500 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.3000 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.1400 | inst |

## Kernel 24: `asymgemm_kernel_24`

`void sm90_bf16_asym_gemm_impl<0, 0, 8, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 92.8600 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 2.9200 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5300 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.7300 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4800 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.2100 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.1800 | inst |

## Kernel 25: `asymgemm_kernel_25`

`void sm90_bf16_asym_gemm_impl<0, 0, 8, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 93.2300 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 3.7200 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5300 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.8500 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4500 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.3500 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.2000 | inst |

## Kernel 26: `asymgemm_kernel_26`

`void sm90_bf16_asym_gemm_impl<0, 0, 8, 2048, 1024, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 32, 1, 0, float, 1...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 494.3400 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 88.0600 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 2.7100 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.6000 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.1400 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0200 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.9700 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.5400 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.3800 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.2500 | inst |

## Kernel 27: `asymgemm_kernel_27`

`void sm90_bf16_asym_gemm_impl<0, 0, 8, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 97.2100 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 4.5000 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5300 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.7800 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4500 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.2200 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.2000 | inst |

## Kernel 28: `asymgemm_kernel_28`

`void sm90_bf16_asym_gemm_impl<0, 0, 8, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 91.7900 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 4.5900 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5200 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.8100 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4500 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.3000 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.2100 | inst |

## Kernel 29: `asymgemm_kernel_29`

`void sm90_bf16_asym_gemm_impl<0, 0, 8, 2048, 1024, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 32, 1, 0, float, 1...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 454.2500 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 83.9400 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 2.5800 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.6300 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.1000 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.9500 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.6100 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.3700 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.2400 | inst |

## Kernel 30: `asymgemm_kernel_30`

`void sm90_bf16_asym_gemm_impl<0, 0, 8, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 92.1700 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 4.1300 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5300 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.7000 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4500 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.2900 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.2000 | inst |

## Kernel 31: `asymgemm_kernel_31`

`void sm90_bf16_asym_gemm_impl<0, 0, 8, 768, 2048, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 12, 1, 0, float, 10...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 91.6600 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 4.2600 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.5700 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5300 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 1.0800 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 0.4700 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.3300 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.2100 | inst |

## tensor_core_util

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 25.4656 | 1.3500 | 36.8700 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 25.4656 | 1.3500 | 36.8700 | % |
| `sm__inst_executed_pipe_tensor_op_gmma.max.pct_of_peak_sustained_active` | 3.1834 | 0.1700 | 4.6100 | % |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 2.4019 | 0.3300 | 3.3500 | % |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 2.4019 | 0.3300 | 3.3500 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 2.4019 | 0.3300 | 3.3500 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 2.4019 | 0.3300 | 3.3500 | % |
| `sm__ops_path_tensor_op_hgmma_src_bf16_dst_fp32_sparsity_off.max.pct_of_peak_sustained_elapsed` | 2.1128 | 0.2600 | 2.9700 | % |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_elapsed` | 2.1128 | 0.2600 | 2.9700 | % |
| `sm__pipe_tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma_cycles_active.max.pct_of_peak_sustained_elapsed` | 2.1128 | 0.2600 | 2.9700 | % |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 1.5909 | 0.0800 | 2.3000 | % |
| `sm__ops_path_tensor_src_bf16_dst_fp32.max.pct_of_peak_sustained_elapsed` | 1.0569 | 0.1300 | 1.4900 | % |
| `sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active` | 0.3006 | 0.0400 | 0.4200 | % |
| `sm__inst_executed_pipe_tensor_op_gmma.sum.pct_of_peak_sustained_active` | 0.3006 | 0.0400 | 0.4200 | % |
| `sm__ops_path_tensor_op_hgmma_src_bf16_dst_fp32_sparsity_off.avg.pct_of_peak_sustained_elapsed` | 0.2091 | 0.0600 | 0.2700 | % |
| `sm__ops_path_tensor_op_hgmma_src_bf16_dst_fp32_sparsity_off.sum.pct_of_peak_sustained_elapsed` | 0.2091 | 0.0600 | 0.2700 | % |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 0.2091 | 0.0600 | 0.2700 | % |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_elapsed` | 0.2091 | 0.0600 | 0.2700 | % |
| `sm__pipe_tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma_cycles_active.avg.pct_of_peak_sustained_elapsed` | 0.2091 | 0.0600 | 0.2700 | % |
| `sm__pipe_tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma_cycles_active.sum.pct_of_peak_sustained_elapsed` | 0.2091 | 0.0600 | 0.2700 | % |

## memory_throughput

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `l1tex__throughput.max.pct_of_peak_sustained_active` | 38.7891 | 2.0800 | 56.1600 | % |
| `gpu__compute_memory_access_throughput.max.pct_of_peak_sustained_elapsed` | 6.7412 | 4.1800 | 14.2400 | % |
| `gpu__compute_memory_throughput.max.pct_of_peak_sustained_elapsed` | 6.7412 | 4.1800 | 14.2400 | % |
| `lts__throughput.max.pct_of_peak_sustained_elapsed` | 5.4728 | 1.5900 | 14.2400 | % |
| `gpu__compute_memory_access_throughput_internal_activity.max.pct_of_peak_sustained_elapsed` | 4.2025 | 0.6500 | 14.2400 | % |
| `l1tex__throughput.avg.pct_of_peak_sustained_active` | 3.6559 | 0.5000 | 5.1000 | % |
| `l1tex__throughput.sum.pct_of_peak_sustained_active` | 3.6559 | 0.5000 | 5.1000 | % |
| `l1tex__throughput.max.pct_of_peak_sustained_elapsed` | 3.2200 | 0.4000 | 4.5300 | % |
| `gpu__compute_memory_request_throughput.max.pct_of_peak_sustained_elapsed` | 2.3816 | 0.9500 | 3.0100 | % |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 1.3091 | 0.9300 | 2.1000 | % |
| `gpu__compute_memory_throughput.sum.pct_of_peak_sustained_elapsed` | 1.3091 | 0.9300 | 2.1000 | % |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | 1.3091 | 0.9300 | 2.1000 | % |
| `lts__throughput.sum.pct_of_peak_sustained_elapsed` | 1.3091 | 0.9300 | 2.1000 | % |
| `gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed` | 0.8734 | 0.4100 | 2.1000 | % |
| `gpu__compute_memory_access_throughput.sum.pct_of_peak_sustained_elapsed` | 0.8734 | 0.4100 | 2.1000 | % |
| `gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed` | 0.8319 | 0.3000 | 1.0900 | % |
| `gpu__compute_memory_request_throughput.sum.pct_of_peak_sustained_elapsed` | 0.8319 | 0.3000 | 1.0900 | % |
| `gpu__compute_memory_access_throughput_internal_activity.avg.pct_of_peak_sustained_elapsed` | 0.8034 | 0.2800 | 2.1000 | % |
| `gpu__compute_memory_access_throughput_internal_activity.sum.pct_of_peak_sustained_elapsed` | 0.8034 | 0.2800 | 2.1000 | % |
| `gpu__compute_memory_request_throughput.min.pct_of_peak_sustained_elapsed` | 0.6444 | 0.2000 | 0.8600 | % |

## l2_dram_behavior

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `lts__t_sector_op_write_hit_rate.pct` | 106.9769 | 20.4700 | 197.5100 | % |
| `l1tex__t_sector_hit_rate.pct` | 90.0000 | 90.0000 | 90.0000 | % |
| `l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct` | 90.0000 | 90.0000 | 90.0000 | % |
| `lts__t_sector_hit_rate.pct` | 21.6972 | 19.4400 | 23.6500 | % |
| `lts__t_sector_op_read_hit_rate.pct` | 17.9644 | 16.4500 | 19.6300 | % |
| `lts__throughput.max.pct_of_peak_sustained_elapsed` | 5.4728 | 1.5900 | 14.2400 | % |
| `l1tex__m_xbar2l1tex_read_sectors.max.pct_of_peak_sustained_elapsed` | 2.1134 | 0.2600 | 2.9700 | % |
| `lts__xbar2lts_cycles_active.max.pct_of_peak_sustained_elapsed` | 1.4566 | 0.9500 | 1.7000 | % |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | 1.3091 | 0.9300 | 2.1000 | % |
| `lts__throughput.sum.pct_of_peak_sustained_elapsed` | 1.3091 | 0.9300 | 2.1000 | % |
| `lts__lts2xbar_cycles_active.max.pct_of_peak_sustained_elapsed` | 1.1722 | 0.3800 | 1.5300 | % |
| `lts__xbar2lts_cycles_active.avg.pct_of_peak_sustained_elapsed` | 0.8319 | 0.3000 | 1.0900 | % |
| `lts__xbar2lts_cycles_active.sum.pct_of_peak_sustained_elapsed` | 0.8319 | 0.3000 | 1.0900 | % |
| `lts__t_sectors.max.pct_of_peak_sustained_elapsed` | 0.7975 | 0.2700 | 1.0500 | % |
| `lts__d_sectors_fill_sysmem.max.pct_of_peak_sustained_elapsed` | 0.6756 | 0.5600 | 0.9300 | % |
| `lts__t_tag_requests.max.pct_of_peak_sustained_elapsed` | 0.6594 | 0.2600 | 0.8700 | % |
| `lts__throughput.min.pct_of_peak_sustained_elapsed` | 0.6444 | 0.2000 | 0.8600 | % |
| `lts__xbar2lts_cycles_active.min.pct_of_peak_sustained_elapsed` | 0.6444 | 0.2000 | 0.8600 | % |
| `lts__d_sectors.max.pct_of_peak_sustained_elapsed` | 0.4981 | 0.1800 | 0.6500 | % |
| `lts__d_sectors_fill_sysmem.avg.pct_of_peak_sustained_elapsed` | 0.4972 | 0.1800 | 0.6500 | % |

## occupancy

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `derived__pct_occupancy_per_shared_mem_size` | 5460.0000 | 5460.0000 | 5460.0000 | %/byte |
| `derived__pct_occupancy_per_register_count` | 3084.0000 | 3084.0000 | 3084.0000 | %/register |
| `derived__pct_occupancy_per_barrier_count` | 780.0000 | 780.0000 | 780.0000 |  |
| `derived__pct_occupancy_per_block_size` | 105.0000 | 105.0000 | 105.0000 | % |
| `sm__warps_active.max.pct_of_peak_sustained_active` | 77.6287 | 38.7000 | 96.0700 | % |
| `sm__maximum_warps_per_active_cycle_pct` | 12.5000 | 12.5000 | 12.5000 | % |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 7.8441 | 7.8200 | 7.8700 | % |
| `sm__warps_active.sum.pct_of_peak_sustained_active` | 7.8441 | 7.8200 | 7.8700 | % |
| `launch__occupancy_cluster_gpu_pct` | 0.0000 | 0.0000 | 0.0000 | % |
| `launch__occupancy_cluster_pct` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__warps_active.min.pct_of_peak_sustained_active` | 0.0000 | 0.0000 | 0.0000 | % |
| `smsp__warps_active.sum.peak_sustained` | 8448.0000 | 8448.0000 | 8448.0000 | warp |
| `launch__occupancy_per_shared_mem_size` | 3640.0000 | 3640.0000 | 3640.0000 |  |
| `launch__occupancy_per_register_count` | 2056.0000 | 2056.0000 | 2056.0000 |  |
| `sm__warps_active.sum.per_cycle_active` | 662.6522 | 660.6000 | 664.4700 | warp |
| `smsp__warps_active.sum.per_cycle_active` | 656.1828 | 596.2100 | 749.5300 | warp |
| `launch__occupancy_per_barrier_count` | 520.0000 | 520.0000 | 520.0000 |  |
| `smsp__average_warps_active_per_inst_executed.ratio` | 243.1259 | 101.2900 | 655.0700 | cycle |
| `launch__occupancy_per_block_size` | 73.0000 | 73.0000 | 73.0000 |  |
| `sm__warps_active.max.per_cycle_active` | 49.6822 | 24.7700 | 61.4800 | warp |

## warp_stall_reasons

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 201.8862 | 90.8800 | 539.8800 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 28.5859 | 2.7400 | 105.7600 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.5478 | 1.5200 | 1.6500 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.4172 | 1.0100 | 1.5800 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 1.1159 | 0.4400 | 3.0700 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 0.9994 | 0.9800 | 1.0200 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.7844 | 0.4700 | 1.2000 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.4997 | 0.3400 | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.3653 | 0.1400 | 0.6100 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.2028 | 0.1700 | 0.3000 | inst |
| `smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio` | 0.0541 | 0.0400 | 0.0600 | inst |
| `smsp__average_warps_issue_stalled_misc_per_issue_active.ratio` | 0.0462 | 0.0200 | 0.1200 | inst |
| `smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio` | 0.0100 | 0.0100 | 0.0100 | inst |
| `smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio` | 0.0031 | 0.0000 | 0.0100 | inst |
| `smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio` | 0.0000 | 0.0000 | 0.0000 | inst |
| `smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio` | 0.0000 | 0.0000 | 0.0000 | inst |
| `smsp__average_warps_issue_stalled_membar_per_issue_active.ratio` | 0.0000 | 0.0000 | 0.0000 | inst |
| `smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio` | 0.0000 | 0.0000 | 0.0000 | inst |
| `smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio` | 0.0000 | 0.0000 | 0.0000 | inst |

## scheduler_stats

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `smsp__issue_inst0.max.pct_of_peak_sustained_active` | 982.4987 | 493.1600 | 1206.7100 | % |
| `smsp__issue_inst0.avg.pct_of_peak_sustained_active` | 99.0831 | 98.7400 | 99.8000 | % |
| `smsp__issue_inst0.sum.pct_of_peak_sustained_active` | 99.0831 | 98.7400 | 99.8000 | % |
| `smsp__issue_active.max.pct_of_peak_sustained_active` | 12.4322 | 1.0500 | 18.1100 | % |
| `sm__inst_issued.max.pct_of_peak_sustained_active` | 9.6756 | 0.8700 | 13.8000 | % |
| `sm__inst_issued.avg.pct_of_peak_sustained_active` | 0.9222 | 0.2000 | 1.2500 | % |
| `sm__inst_issued.sum.pct_of_peak_sustained_active` | 0.9222 | 0.2000 | 1.2500 | % |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | 0.9169 | 0.2000 | 1.2600 | % |
| `smsp__issue_active.sum.pct_of_peak_sustained_active` | 0.9169 | 0.2000 | 1.2600 | % |
| `sm__issue_active.max.pct_of_peak_sustained_elapsed` | 0.8166 | 0.1700 | 1.1100 | % |
| `sm__mio_inst_issued.max.pct_of_peak_sustained_elapsed` | 0.2113 | 0.0800 | 0.2700 | % |
| `sm__issue_active.avg.pct_of_peak_sustained_elapsed` | 0.0831 | 0.0400 | 0.1000 | % |
| `sm__issue_active.sum.pct_of_peak_sustained_elapsed` | 0.0831 | 0.0400 | 0.1000 | % |
| `sm__mio_inst_issued.avg.pct_of_peak_sustained_elapsed` | 0.0200 | 0.0200 | 0.0200 | % |
| `sm__mio_inst_issued.sum.pct_of_peak_sustained_elapsed` | 0.0200 | 0.0200 | 0.0200 | % |
| `sm__inst_issued.min.pct_of_peak_sustained_active` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__issue_active.min.pct_of_peak_sustained_elapsed` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__mio_inst_issued.min.pct_of_peak_sustained_elapsed` | 0.0000 | 0.0000 | 0.0000 | % |
| `smsp__issue_active.min.pct_of_peak_sustained_active` | 0.0000 | 0.0000 | 0.0000 | % |
| `smsp__issue_inst0.min.pct_of_peak_sustained_active` | 0.0000 | 0.0000 | 0.0000 | % |

## roofline_position

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 25.4656 | 1.3500 | 36.8700 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 2.4019 | 0.3300 | 3.3500 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 2.4019 | 0.3300 | 3.3500 | % |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 1.5909 | 0.0800 | 2.3000 | % |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 0.1503 | 0.0200 | 0.2100 | % |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 0.1503 | 0.0200 | 0.2100 | % |
| `sm__inst_executed_pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__inst_executed_pipe_tensor_op_dmma.max.pct_of_peak_sustained_active` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__inst_executed_pipe_tensor_op_dmma.min.pct_of_peak_sustained_active` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__inst_executed_pipe_tensor_op_dmma.sum.pct_of_peak_sustained_active` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.avg.pct_of_peak_sustained_elapsed` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.max.pct_of_peak_sustained_elapsed` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.min.pct_of_peak_sustained_elapsed` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.sum.pct_of_peak_sustained_elapsed` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_on.avg.pct_of_peak_sustained_elapsed` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_on.max.pct_of_peak_sustained_elapsed` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_on.min.pct_of_peak_sustained_elapsed` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_on.sum.pct_of_peak_sustained_elapsed` | 0.0000 | 0.0000 | 0.0000 | % |
| `sm__ops_path_tensor_op_hmma_src_fp16_dst_fp16_sparsity_off.avg.pct_of_peak_sustained_elapsed` | 0.0000 | 0.0000 | 0.0000 | % |

## launch_stats

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `launch__occupancy_cluster_gpu_pct` | 0.0000 | 0.0000 | 0.0000 | % |
| `launch__occupancy_cluster_pct` | 0.0000 | 0.0000 | 0.0000 | % |
| `launch__shared_mem_config_size` | 233472.0000 | 233472.0000 | 233472.0000 | byte |
| `launch__shared_mem_per_block_allocated` | 230528.0000 | 230528.0000 | 230528.0000 | byte/block |
| `launch__shared_mem_per_block` | 230461.5000 | 230456.0000 | 230464.0000 | byte/block |
| `launch__shared_mem_per_block_dynamic` | 229437.5000 | 229432.0000 | 229440.0000 | byte/block |
| `launch__thread_count` | 4672.0000 | 3072.0000 | 8192.0000 | thread |
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
| `launch__occupancy_per_block_size` | 73.0000 | 73.0000 | 73.0000 |  |
| `launch__tpc_count` | 66.0000 | 66.0000 | 66.0000 |  |
| `launch__occupancy_limit_blocks` | 32.0000 | 32.0000 | 32.0000 | block |
