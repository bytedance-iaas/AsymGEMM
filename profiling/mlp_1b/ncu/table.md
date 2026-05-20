# NCU AsymGEMM Kernel Report: mlp_1b

Source: `profiling/mlp_1b/ncu/raw.csv`

## Kernel Summary

| ID | Operation | duration ms | tensor pipe % | SM throughput % | memory throughput % | DRAM % | L2 % | issue active % | active warps % | regs/thread | smem/block | replay passes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `forward.fc1.base_frozen_asymgemm` | 33.7658 | 0.2600 | 0.4000 | 6.6200 | 0.0000 | 6.6200 | 0.1500 | 7.8100 | 248.0000 | 230512.0000 | 32.0000 |
| 1 | `forward.fc2.base_frozen_asymgemm` | 39.7380 | 0.2200 | 0.3800 | 4.1300 | 0.0100 | 4.1300 | 0.1400 | 7.8100 | 248.0000 | 230960.0000 | 32.0000 |
| 2 | `backward.fc2.base_dx_asymgemm` | 37.3491 | 0.2500 | 0.5800 | 2.4200 | 0.0000 | 2.4200 | 0.2600 | 7.8100 | 248.0000 | 58928.0000 | 32.0000 |
| 3 | `backward.fc1.base_dx_asymgemm` | 27.6400 | 0.3000 | 0.5700 | 3.2600 | 0.0100 | 3.2600 | 0.2900 | 7.8100 | 248.0000 | 62512.0000 | 32.0000 |

## Kernel 0: `forward.fc1.base_frozen_asymgemm`

`void sm90_bf16_asym_gemm_impl<0, 0, 64, 65536, 8192, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 128, 1, 0, float...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 773.4600 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 8.9000 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 4.6600 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 3.7300 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.7800 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.9100 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.4500 | inst |
| `smsp__average_warps_issue_stalled_misc_per_issue_active.ratio` | 0.1900 | inst |
| `smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio` | 0.0400 | inst |

## Kernel 1: `forward.fc2.base_frozen_asymgemm`

`void sm90_bf16_asym_gemm_impl<0, 0, 64, 8192, 65536, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 128, 1, 0, float...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 820.9400 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 4.9500 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 3.2300 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.8000 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0200 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.8700 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 0.5600 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.4700 | inst |
| `smsp__average_warps_issue_stalled_misc_per_issue_active.ratio` | 0.2100 | inst |
| `smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio` | 0.0400 | inst |

## Kernel 2: `backward.fc2.base_dx_asymgemm`

`void sm90_bf16_asym_gemm_impl<0, 1, 64, 65536, 8192, 64, 64, 64, 1, 128, 128, 0, 2, 128, 128, 1, 0, 128, 1, 0, float,...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 460.9300 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 2.4400 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 2.2300 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.2800 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 0.6800 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.4700 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.2900 | inst |
| `smsp__average_warps_issue_stalled_misc_per_issue_active.ratio` | 0.0900 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.0600 | inst |

## Kernel 3: `backward.fc1.base_dx_asymgemm`

`void sm90_bf16_asym_gemm_impl<0, 1, 64, 8192, 65536, 64, 64, 64, 1, 128, 128, 0, 2, 128, 128, 1, 0, 128, 1, 0, float,...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 423.1400 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 2.2800 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 1.8200 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.3900 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.5200 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.2400 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 0.0600 | inst |
| `smsp__average_warps_issue_stalled_misc_per_issue_active.ratio` | 0.0600 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.0600 | inst |

## tensor_core_util

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 0.3100 | 0.2500 | 0.3800 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 0.3100 | 0.2500 | 0.3800 | % |
| `sm__ops_path_tensor_op_hgmma_src_bf16_dst_fp32_sparsity_off.max.pct_of_peak_sustained_elapsed` | 0.2825 | 0.2300 | 0.3300 | % |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_elapsed` | 0.2825 | 0.2300 | 0.3300 | % |
| `sm__pipe_tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma_cycles_active.max.pct_of_peak_sustained_elapsed` | 0.2825 | 0.2300 | 0.3300 | % |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 0.2800 | 0.2400 | 0.3600 | % |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 0.2800 | 0.2400 | 0.3600 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 0.2800 | 0.2400 | 0.3600 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 0.2800 | 0.2400 | 0.3600 | % |
| `sm__ops_path_tensor_op_hgmma_src_bf16_dst_fp32_sparsity_off.avg.pct_of_peak_sustained_elapsed` | 0.2575 | 0.2200 | 0.3000 | % |
| `sm__ops_path_tensor_op_hgmma_src_bf16_dst_fp32_sparsity_off.sum.pct_of_peak_sustained_elapsed` | 0.2575 | 0.2200 | 0.3000 | % |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 0.2575 | 0.2200 | 0.3000 | % |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_elapsed` | 0.2575 | 0.2200 | 0.3000 | % |
| `sm__pipe_tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma_cycles_active.avg.pct_of_peak_sustained_elapsed` | 0.2575 | 0.2200 | 0.3000 | % |
| `sm__pipe_tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma_cycles_active.sum.pct_of_peak_sustained_elapsed` | 0.2575 | 0.2200 | 0.3000 | % |
| `sm__ops_path_tensor_src_bf16_dst_fp32.max.pct_of_peak_sustained_elapsed` | 0.1450 | 0.1200 | 0.1700 | % |
| `sm__ops_path_tensor_src_bf16_dst_fp32.avg.pct_of_peak_sustained_elapsed` | 0.1300 | 0.1100 | 0.1500 | % |
| `sm__ops_path_tensor_src_bf16_dst_fp32.sum.pct_of_peak_sustained_elapsed` | 0.1300 | 0.1100 | 0.1500 | % |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 0.1100 | 0.0000 | 0.2400 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 0.1100 | 0.0000 | 0.2400 | % |

## memory_throughput

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `gpu__compute_memory_throughput.max.pct_of_peak_sustained_elapsed` | 6.1050 | 2.5400 | 10.4800 | % |
| `lts__throughput.max.pct_of_peak_sustained_elapsed` | 6.1050 | 2.5400 | 10.4800 | % |
| `gpu__compute_memory_access_throughput.max.pct_of_peak_sustained_elapsed` | 5.2000 | 1.3300 | 10.4800 | % |
| `gpu__compute_memory_access_throughput_internal_activity.max.pct_of_peak_sustained_elapsed` | 4.9675 | 0.8400 | 10.4800 | % |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 4.1075 | 2.4200 | 6.6200 | % |
| `gpu__compute_memory_throughput.sum.pct_of_peak_sustained_elapsed` | 4.1075 | 2.4200 | 6.6200 | % |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | 4.1075 | 2.4200 | 6.6200 | % |
| `lts__throughput.sum.pct_of_peak_sustained_elapsed` | 4.1075 | 2.4200 | 6.6200 | % |
| `gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed` | 3.3950 | 1.2900 | 6.6200 | % |
| `gpu__compute_memory_access_throughput.sum.pct_of_peak_sustained_elapsed` | 3.3950 | 1.2900 | 6.6200 | % |
| `gpu__compute_memory_access_throughput_internal_activity.avg.pct_of_peak_sustained_elapsed` | 3.1450 | 0.7800 | 6.6200 | % |
| `gpu__compute_memory_access_throughput_internal_activity.sum.pct_of_peak_sustained_elapsed` | 3.1450 | 0.7800 | 6.6200 | % |
| `gpu__compute_memory_throughput.min.pct_of_peak_sustained_elapsed` | 2.6625 | 1.9500 | 3.8100 | % |
| `lts__throughput.min.pct_of_peak_sustained_elapsed` | 2.6625 | 1.9500 | 3.8100 | % |
| `gpu__compute_memory_request_throughput.max.pct_of_peak_sustained_elapsed` | 2.2575 | 1.1600 | 3.9900 | % |
| `gpu__compute_memory_access_throughput.min.pct_of_peak_sustained_elapsed` | 2.1025 | 1.1700 | 3.8100 | % |
| `gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed` | 2.0150 | 1.1000 | 3.2600 | % |
| `gpu__compute_memory_request_throughput.sum.pct_of_peak_sustained_elapsed` | 2.0150 | 1.1000 | 3.2600 | % |
| `gpu__compute_memory_access_throughput_internal_activity.min.pct_of_peak_sustained_elapsed` | 1.8700 | 0.7600 | 3.8100 | % |
| `gpu__compute_memory_request_throughput.min.pct_of_peak_sustained_elapsed` | 1.7825 | 1.0200 | 2.5800 | % |

## l2_dram_behavior

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `l1tex__t_sector_hit_rate.pct` | 94.3550 | 90.0000 | 98.7100 | % |
| `l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct` | 94.3550 | 90.0000 | 98.7100 | % |
| `lts__t_sector_op_write_hit_rate.pct` | 89.2725 | 57.0200 | 108.2800 | % |
| `lts__t_sector_hit_rate.pct` | 59.4400 | 46.0200 | 72.6600 | % |
| `lts__t_sector_op_read_hit_rate.pct` | 38.7425 | 37.4200 | 39.7000 | % |
| `lts__throughput.max.pct_of_peak_sustained_elapsed` | 6.1050 | 2.5400 | 10.4800 | % |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | 4.1075 | 2.4200 | 6.6200 | % |
| `lts__throughput.sum.pct_of_peak_sustained_elapsed` | 4.1075 | 2.4200 | 6.6200 | % |
| `lts__throughput.min.pct_of_peak_sustained_elapsed` | 2.6625 | 1.9500 | 3.8100 | % |
| `lts__xbar2lts_cycles_active.max.pct_of_peak_sustained_elapsed` | 2.2575 | 1.1600 | 3.9900 | % |
| `lts__xbar2lts_cycles_active.avg.pct_of_peak_sustained_elapsed` | 2.0150 | 1.1000 | 3.2600 | % |
| `lts__xbar2lts_cycles_active.sum.pct_of_peak_sustained_elapsed` | 2.0150 | 1.1000 | 3.2600 | % |
| `lts__xbar2lts_cycles_active.min.pct_of_peak_sustained_elapsed` | 1.7825 | 1.0200 | 2.5800 | % |
| `lts__t_sectors_srcunit_tex.avg.pct_of_peak_sustained_elapsed` | 0.9825 | 0.5900 | 1.5200 | % |
| `lts__t_sectors_srcunit_tex.sum.pct_of_peak_sustained_elapsed` | 0.9825 | 0.5900 | 1.5200 | % |
| `lts__t_sectors.max.pct_of_peak_sustained_elapsed` | 0.9350 | 0.5200 | 1.5200 | % |
| `lts__t_sectors.avg.pct_of_peak_sustained_elapsed` | 0.8825 | 0.5100 | 1.4000 | % |
| `lts__t_sectors.sum.pct_of_peak_sustained_elapsed` | 0.8825 | 0.5100 | 1.4000 | % |
| `lts__t_sectors.min.pct_of_peak_sustained_elapsed` | 0.8325 | 0.5000 | 1.2800 | % |
| `lts__d_atomic_input_cycles_active.max.pct_of_peak_sustained_elapsed` | 0.7400 | 0.1300 | 1.5500 | % |

## occupancy

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `derived__pct_occupancy_per_shared_mem_size` | 5460.0000 | 5460.0000 | 5460.0000 | %/byte |
| `derived__pct_occupancy_per_register_count` | 3084.0000 | 3084.0000 | 3084.0000 | %/register |
| `derived__pct_occupancy_per_barrier_count` | 780.0000 | 780.0000 | 780.0000 |  |
| `derived__pct_occupancy_per_block_size` | 105.0000 | 105.0000 | 105.0000 | % |
| `sm__maximum_warps_per_active_cycle_pct` | 12.5000 | 12.5000 | 12.5000 | % |
| `sm__warps_active.max.pct_of_peak_sustained_active` | 8.1750 | 7.9500 | 8.5800 | % |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 7.8100 | 7.8100 | 7.8100 | % |
| `sm__warps_active.sum.pct_of_peak_sustained_active` | 7.8100 | 7.8100 | 7.8100 | % |
| `sm__warps_active.min.pct_of_peak_sustained_active` | 3.6600 | 0.0000 | 7.4100 | % |
| `launch__occupancy_cluster_gpu_pct` | 0.0000 | 0.0000 | 0.0000 | % |
| `launch__occupancy_cluster_pct` | 0.0000 | 0.0000 | 0.0000 | % |
| `smsp__warps_active.sum.peak_sustained` | 8448.0000 | 8448.0000 | 8448.0000 | warp |
| `launch__occupancy_per_shared_mem_size` | 3640.0000 | 3640.0000 | 3640.0000 |  |
| `launch__occupancy_per_register_count` | 2056.0000 | 2056.0000 | 2056.0000 |  |
| `smsp__average_warps_active_per_inst_executed.ratio` | 660.8500 | 371.0100 | 937.9700 | cycle |
| `sm__warps_active.sum.per_cycle_active` | 659.9500 | 659.8300 | 660.0100 | warp |
| `smsp__warps_active.sum.per_cycle_active` | 659.1625 | 651.0200 | 665.1100 | warp |
| `launch__occupancy_per_barrier_count` | 520.0000 | 520.0000 | 520.0000 |  |
| `launch__occupancy_per_block_size` | 73.0000 | 73.0000 | 73.0000 |  |
| `launch__occupancy_limit_blocks` | 32.0000 | 32.0000 | 32.0000 | block |

## warp_stall_reasons

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 619.6175 | 423.1400 | 820.9400 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 3.4675 | 1.8200 | 4.9500 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 2.5500 | 0.0600 | 8.9000 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 2.0225 | 1.7800 | 2.2800 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 1.9875 | 0.4700 | 3.7300 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.1125 | 0.8700 | 1.3900 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0050 | 1.0000 | 1.0200 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.3625 | 0.2400 | 0.4700 | inst |
| `smsp__average_warps_issue_stalled_misc_per_issue_active.ratio` | 0.1375 | 0.0600 | 0.2100 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.0375 | 0.0000 | 0.0600 | inst |
| `smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio` | 0.0250 | 0.0000 | 0.0500 | inst |
| `smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio` | 0.0200 | 0.0000 | 0.0400 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.0050 | 0.0000 | 0.0100 | inst |
| `smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio` | 0.0025 | 0.0000 | 0.0100 | inst |
| `smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio` | 0.0000 | 0.0000 | 0.0000 | inst |
| `smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio` | 0.0000 | 0.0000 | 0.0000 | inst |
| `smsp__average_warps_issue_stalled_membar_per_issue_active.ratio` | 0.0000 | 0.0000 | 0.0000 | inst |
| `smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio` | 0.0000 | 0.0000 | 0.0000 | inst |
| `smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio` | 0.0000 | 0.0000 | 0.0000 | inst |

## scheduler_stats

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `smsp__issue_inst0.max.pct_of_peak_sustained_active` | 104.0900 | 101.3800 | 108.9600 | % |
| `smsp__issue_inst0.avg.pct_of_peak_sustained_active` | 99.7700 | 99.6600 | 99.8500 | % |
| `smsp__issue_inst0.sum.pct_of_peak_sustained_active` | 99.7700 | 99.6600 | 99.8500 | % |
| `smsp__issue_inst0.min.pct_of_peak_sustained_active` | 46.3425 | 0.0000 | 93.1100 | % |
| `smsp__issue_active.max.pct_of_peak_sustained_active` | 0.3025 | 0.1900 | 0.5000 | % |
| `sm__inst_issued.max.pct_of_peak_sustained_active` | 0.2450 | 0.1600 | 0.3600 | % |
| `sm__inst_issued.avg.pct_of_peak_sustained_active` | 0.2300 | 0.1500 | 0.3400 | % |
| `sm__inst_issued.sum.pct_of_peak_sustained_active` | 0.2300 | 0.1500 | 0.3400 | % |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | 0.2300 | 0.1500 | 0.3400 | % |
| `smsp__issue_active.sum.pct_of_peak_sustained_active` | 0.2300 | 0.1500 | 0.3400 | % |
| `sm__issue_active.max.pct_of_peak_sustained_elapsed` | 0.2225 | 0.1500 | 0.2900 | % |
| `sm__issue_active.avg.pct_of_peak_sustained_elapsed` | 0.2100 | 0.1400 | 0.2900 | % |
| `sm__issue_active.sum.pct_of_peak_sustained_elapsed` | 0.2100 | 0.1400 | 0.2900 | % |
| `sm__mio_inst_issued.max.pct_of_peak_sustained_elapsed` | 0.1275 | 0.0800 | 0.1800 | % |
| `sm__mio_inst_issued.avg.pct_of_peak_sustained_elapsed` | 0.1200 | 0.0700 | 0.1700 | % |
| `sm__mio_inst_issued.sum.pct_of_peak_sustained_elapsed` | 0.1200 | 0.0700 | 0.1700 | % |
| `sm__inst_issued.min.pct_of_peak_sustained_active` | 0.0975 | 0.0000 | 0.2500 | % |
| `sm__issue_active.min.pct_of_peak_sustained_elapsed` | 0.0925 | 0.0000 | 0.2400 | % |
| `smsp__issue_active.min.pct_of_peak_sustained_active` | 0.0900 | 0.0000 | 0.2300 | % |
| `sm__mio_inst_issued.min.pct_of_peak_sustained_elapsed` | 0.0550 | 0.0000 | 0.1500 | % |

## roofline_position

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 0.3100 | 0.2500 | 0.3800 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 0.2800 | 0.2400 | 0.3600 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 0.2800 | 0.2400 | 0.3600 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 0.1100 | 0.0000 | 0.2400 | % |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 0.0200 | 0.0200 | 0.0200 | % |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 0.0200 | 0.0200 | 0.0200 | % |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 0.0200 | 0.0200 | 0.0200 | % |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 0.0050 | 0.0000 | 0.0100 | % |
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
| `launch__thread_count` | 147456.0000 | 32768.0000 | 262144.0000 | thread |
| `launch__shared_mem_per_block_allocated` | 145792.0000 | 59008.0000 | 231040.0000 | byte/block |
| `launch__shared_mem_per_block` | 145728.0000 | 58928.0000 | 230960.0000 | byte/block |
| `launch__shared_mem_per_block_dynamic` | 144704.0000 | 57904.0000 | 229936.0000 | byte/block |
| `launch__occupancy_per_shared_mem_size` | 3640.0000 | 3640.0000 | 3640.0000 |  |
| `launch__occupancy_per_register_count` | 2056.0000 | 2056.0000 | 2056.0000 |  |
| `launch__shared_mem_per_block_driver` | 1024.0000 | 1024.0000 | 1024.0000 | byte/block |
| `launch__stack_size` | 1024.0000 | 1024.0000 | 1024.0000 |  |
| `launch__grid_dim_x` | 576.0000 | 128.0000 | 1024.0000 |  |
| `launch__grid_size` | 576.0000 | 128.0000 | 1024.0000 |  |
| `launch__occupancy_per_barrier_count` | 520.0000 | 520.0000 | 520.0000 |  |
| `launch__block_dim_x` | 256.0000 | 256.0000 | 256.0000 | block |
| `launch__block_size` | 256.0000 | 256.0000 | 256.0000 |  |
| `launch__registers_per_thread` | 248.0000 | 248.0000 | 248.0000 | register/thread |
| `launch__registers_per_thread_allocated` | 248.0000 | 248.0000 | 248.0000 | register/thread |
| `launch__sm_count` | 132.0000 | 132.0000 | 132.0000 | SM |
| `launch__occupancy_per_block_size` | 73.0000 | 73.0000 | 73.0000 |  |
