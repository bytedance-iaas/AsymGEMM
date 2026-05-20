# NCU AsymGEMM Kernel Report: matrix_1b

Source: `profiling/matrix_1b/ncu/raw.csv`

## Kernel Summary

| ID | Operation | duration ms | tensor pipe % | SM throughput % | memory throughput % | DRAM % | L2 % | issue active % | active warps % | regs/thread | smem/block | replay passes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `forward.matrix.base_frozen_asymgemm` | 78.7386 | 0.2200 | 0.3900 | 4.9500 | 0.0000 | 4.9500 | 0.1400 | 7.8100 | 248.0000 | 230704.0000 | 32.0000 |
| 1 | `backward.matrix.base_dx_asymgemm` | 80.1419 | 0.2400 | 0.5700 | 2.2600 | 0.0000 | 2.2600 | 0.2500 | 7.8100 | 248.0000 | 60464.0000 | 32.0000 |

## Kernel 0: `forward.matrix.base_frozen_asymgemm`

`void sm90_bf16_asym_gemm_impl<0, 0, 64, 32768, 32768, 64, 64, 512, 1, 128, 128, 0, 2, 128, 128, 1, 0, 128, 1, 0, floa...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 838.7700 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 5.1100 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 3.4400 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 2.4800 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.8100 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.8500 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.4800 | inst |
| `smsp__average_warps_issue_stalled_misc_per_issue_active.ratio` | 0.2100 | inst |
| `smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio` | 0.0300 | inst |

## Kernel 1: `backward.matrix.base_dx_asymgemm`

`void sm90_bf16_asym_gemm_impl<0, 1, 64, 32768, 32768, 64, 64, 64, 1, 128, 128, 0, 2, 128, 128, 1, 0, 128, 1, 0, float...`

### Top Warp Stalls

| Metric | value | unit |
|---|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 485.3000 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 2.6200 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 2.2200 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.2400 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.4600 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.3000 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 0.1600 | inst |
| `smsp__average_warps_issue_stalled_misc_per_issue_active.ratio` | 0.1000 | inst |
| `smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio` | 0.0500 | inst |

## tensor_core_util

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_active` | 0.2750 | 0.2500 | 0.3000 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 0.2750 | 0.2500 | 0.3000 | % |
| `sm__ops_path_tensor_op_hgmma_src_bf16_dst_fp32_sparsity_off.max.pct_of_peak_sustained_elapsed` | 0.2650 | 0.2400 | 0.2900 | % |
| `sm__pipe_tensor_cycles_active.max.pct_of_peak_sustained_elapsed` | 0.2650 | 0.2400 | 0.2900 | % |
| `sm__pipe_tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma_cycles_active.max.pct_of_peak_sustained_elapsed` | 0.2650 | 0.2400 | 0.2900 | % |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 0.2350 | 0.2300 | 0.2400 | % |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_active` | 0.2350 | 0.2300 | 0.2400 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 0.2350 | 0.2300 | 0.2400 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 0.2350 | 0.2300 | 0.2400 | % |
| `sm__ops_path_tensor_op_hgmma_src_bf16_dst_fp32_sparsity_off.avg.pct_of_peak_sustained_elapsed` | 0.2300 | 0.2200 | 0.2400 | % |
| `sm__ops_path_tensor_op_hgmma_src_bf16_dst_fp32_sparsity_off.sum.pct_of_peak_sustained_elapsed` | 0.2300 | 0.2200 | 0.2400 | % |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 0.2300 | 0.2200 | 0.2400 | % |
| `sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_elapsed` | 0.2300 | 0.2200 | 0.2400 | % |
| `sm__pipe_tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma_cycles_active.avg.pct_of_peak_sustained_elapsed` | 0.2300 | 0.2200 | 0.2400 | % |
| `sm__pipe_tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma_cycles_active.sum.pct_of_peak_sustained_elapsed` | 0.2300 | 0.2200 | 0.2400 | % |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_active` | 0.1850 | 0.1800 | 0.1900 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 0.1850 | 0.1800 | 0.1900 | % |
| `sm__ops_path_tensor_op_hgmma_src_bf16_dst_fp32_sparsity_off.min.pct_of_peak_sustained_elapsed` | 0.1750 | 0.1700 | 0.1800 | % |
| `sm__pipe_tensor_cycles_active.min.pct_of_peak_sustained_elapsed` | 0.1750 | 0.1700 | 0.1800 | % |
| `sm__pipe_tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma_cycles_active.min.pct_of_peak_sustained_elapsed` | 0.1750 | 0.1700 | 0.1800 | % |

## memory_throughput

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `gpu__compute_memory_throughput.max.pct_of_peak_sustained_elapsed` | 5.1400 | 2.3700 | 7.9100 | % |
| `lts__throughput.max.pct_of_peak_sustained_elapsed` | 5.1400 | 2.3700 | 7.9100 | % |
| `gpu__compute_memory_access_throughput.max.pct_of_peak_sustained_elapsed` | 4.5750 | 1.2400 | 7.9100 | % |
| `gpu__compute_memory_access_throughput_internal_activity.max.pct_of_peak_sustained_elapsed` | 4.3350 | 0.7600 | 7.9100 | % |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 3.6050 | 2.2600 | 4.9500 | % |
| `gpu__compute_memory_throughput.sum.pct_of_peak_sustained_elapsed` | 3.6050 | 2.2600 | 4.9500 | % |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | 3.6050 | 2.2600 | 4.9500 | % |
| `lts__throughput.sum.pct_of_peak_sustained_elapsed` | 3.6050 | 2.2600 | 4.9500 | % |
| `gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed` | 3.0750 | 1.2000 | 4.9500 | % |
| `gpu__compute_memory_access_throughput.sum.pct_of_peak_sustained_elapsed` | 3.0750 | 1.2000 | 4.9500 | % |
| `gpu__compute_memory_access_throughput_internal_activity.avg.pct_of_peak_sustained_elapsed` | 2.8400 | 0.7300 | 4.9500 | % |
| `gpu__compute_memory_access_throughput_internal_activity.sum.pct_of_peak_sustained_elapsed` | 2.8400 | 0.7300 | 4.9500 | % |
| `gpu__compute_memory_throughput.min.pct_of_peak_sustained_elapsed` | 2.1700 | 2.1500 | 2.1900 | % |
| `lts__throughput.min.pct_of_peak_sustained_elapsed` | 2.1700 | 2.1500 | 2.1900 | % |
| `gpu__compute_memory_request_throughput.max.pct_of_peak_sustained_elapsed` | 1.7650 | 1.1600 | 2.3700 | % |
| `gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed` | 1.6800 | 1.1000 | 2.2600 | % |
| `gpu__compute_memory_request_throughput.sum.pct_of_peak_sustained_elapsed` | 1.6800 | 1.1000 | 2.2600 | % |
| `gpu__compute_memory_access_throughput.min.pct_of_peak_sustained_elapsed` | 1.6250 | 1.0600 | 2.1900 | % |
| `gpu__compute_memory_request_throughput.min.pct_of_peak_sustained_elapsed` | 1.6050 | 1.0600 | 2.1500 | % |
| `gpu__compute_memory_access_throughput_internal_activity.min.pct_of_peak_sustained_elapsed` | 1.4500 | 0.7100 | 2.1900 | % |

## l2_dram_behavior

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `lts__t_sector_op_write_hit_rate.pct` | 100.5600 | 98.1600 | 102.9600 | % |
| `l1tex__t_sector_hit_rate.pct` | 97.4200 | 97.4200 | 97.4200 | % |
| `l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct` | 97.4200 | 97.4200 | 97.4200 | % |
| `lts__t_sector_hit_rate.pct` | 59.7450 | 46.8300 | 72.6600 | % |
| `lts__t_sector_op_read_hit_rate.pct` | 39.2550 | 38.6200 | 39.8900 | % |
| `lts__throughput.max.pct_of_peak_sustained_elapsed` | 5.1400 | 2.3700 | 7.9100 | % |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | 3.6050 | 2.2600 | 4.9500 | % |
| `lts__throughput.sum.pct_of_peak_sustained_elapsed` | 3.6050 | 2.2600 | 4.9500 | % |
| `lts__throughput.min.pct_of_peak_sustained_elapsed` | 2.1700 | 2.1500 | 2.1900 | % |
| `lts__xbar2lts_cycles_active.max.pct_of_peak_sustained_elapsed` | 1.7650 | 1.1600 | 2.3700 | % |
| `lts__xbar2lts_cycles_active.avg.pct_of_peak_sustained_elapsed` | 1.6800 | 1.1000 | 2.2600 | % |
| `lts__xbar2lts_cycles_active.sum.pct_of_peak_sustained_elapsed` | 1.6800 | 1.1000 | 2.2600 | % |
| `lts__xbar2lts_cycles_active.min.pct_of_peak_sustained_elapsed` | 1.6050 | 1.0600 | 2.1500 | % |
| `lts__t_sectors_srcunit_tex.avg.pct_of_peak_sustained_elapsed` | 0.8250 | 0.5900 | 1.0600 | % |
| `lts__t_sectors_srcunit_tex.sum.pct_of_peak_sustained_elapsed` | 0.8250 | 0.5900 | 1.0600 | % |
| `lts__t_sectors.max.pct_of_peak_sustained_elapsed` | 0.7700 | 0.5400 | 1.0000 | % |
| `lts__t_sectors.avg.pct_of_peak_sustained_elapsed` | 0.7400 | 0.5100 | 0.9700 | % |
| `lts__t_sectors.sum.pct_of_peak_sustained_elapsed` | 0.7400 | 0.5100 | 0.9700 | % |
| `lts__t_sectors.min.pct_of_peak_sustained_elapsed` | 0.7150 | 0.4800 | 0.9500 | % |
| `lts__lts2xbar_cycles_active.max.pct_of_peak_sustained_elapsed` | 0.6300 | 0.6100 | 0.6500 | % |

## occupancy

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `derived__pct_occupancy_per_shared_mem_size` | 5460.0000 | 5460.0000 | 5460.0000 | %/byte |
| `derived__pct_occupancy_per_register_count` | 3084.0000 | 3084.0000 | 3084.0000 | %/register |
| `derived__pct_occupancy_per_barrier_count` | 780.0000 | 780.0000 | 780.0000 |  |
| `derived__pct_occupancy_per_block_size` | 105.0000 | 105.0000 | 105.0000 | % |
| `sm__maximum_warps_per_active_cycle_pct` | 12.5000 | 12.5000 | 12.5000 | % |
| `sm__warps_active.max.pct_of_peak_sustained_active` | 8.1100 | 8.0900 | 8.1300 | % |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 7.8100 | 7.8100 | 7.8100 | % |
| `sm__warps_active.sum.pct_of_peak_sustained_active` | 7.8100 | 7.8100 | 7.8100 | % |
| `sm__warps_active.min.pct_of_peak_sustained_active` | 6.8200 | 6.1800 | 7.4600 | % |
| `launch__occupancy_cluster_gpu_pct` | 0.0000 | 0.0000 | 0.0000 | % |
| `launch__occupancy_cluster_pct` | 0.0000 | 0.0000 | 0.0000 | % |
| `smsp__warps_active.sum.peak_sustained` | 8448.0000 | 8448.0000 | 8448.0000 | warp |
| `launch__occupancy_per_shared_mem_size` | 3640.0000 | 3640.0000 | 3640.0000 |  |
| `launch__occupancy_per_register_count` | 2056.0000 | 2056.0000 | 2056.0000 |  |
| `smsp__average_warps_active_per_inst_executed.ratio` | 730.0300 | 519.0500 | 941.0100 | cycle |
| `smsp__warps_active.sum.per_cycle_active` | 662.4350 | 659.9700 | 664.9000 | warp |
| `sm__warps_active.sum.per_cycle_active` | 659.9750 | 659.9500 | 660.0000 | warp |
| `launch__occupancy_per_barrier_count` | 520.0000 | 520.0000 | 520.0000 |  |
| `launch__occupancy_per_block_size` | 73.0000 | 73.0000 | 73.0000 |  |
| `launch__occupancy_limit_blocks` | 32.0000 | 32.0000 | 32.0000 | block |

## warp_stall_reasons

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 662.0350 | 485.3000 | 838.7700 | inst |
| `smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio` | 3.8650 | 2.6200 | 5.1100 | inst |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 2.0150 | 1.8100 | 2.2200 | inst |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 1.9500 | 0.4600 | 3.4400 | inst |
| `smsp__average_warps_issue_stalled_drain_per_issue_active.ratio` | 1.3200 | 0.1600 | 2.4800 | inst |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 1.0450 | 0.8500 | 1.2400 | inst |
| `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` | 1.0000 | 1.0000 | 1.0000 | inst |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.3900 | 0.3000 | 0.4800 | inst |
| `smsp__average_warps_issue_stalled_misc_per_issue_active.ratio` | 0.1550 | 0.1000 | 0.2100 | inst |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.0300 | 0.0100 | 0.0500 | inst |
| `smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio` | 0.0250 | 0.0000 | 0.0500 | inst |
| `smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio` | 0.0150 | 0.0000 | 0.0300 | inst |
| `smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio` | 0.0050 | 0.0000 | 0.0100 | inst |
| `smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio` | 0.0000 | 0.0000 | 0.0000 | inst |
| `smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio` | 0.0000 | 0.0000 | 0.0000 | inst |
| `smsp__average_warps_issue_stalled_membar_per_issue_active.ratio` | 0.0000 | 0.0000 | 0.0000 | inst |
| `smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio` | 0.0000 | 0.0000 | 0.0000 | inst |
| `smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio` | 0.0000 | 0.0000 | 0.0000 | inst |
| `smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio` | 0.0000 | 0.0000 | 0.0000 | inst |

## scheduler_stats

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `smsp__issue_inst0.max.pct_of_peak_sustained_active` | 103.7650 | 103.4600 | 104.0700 | % |
| `smsp__issue_inst0.avg.pct_of_peak_sustained_active` | 99.7950 | 99.7400 | 99.8500 | % |
| `smsp__issue_inst0.sum.pct_of_peak_sustained_active` | 99.7950 | 99.7400 | 99.8500 | % |
| `smsp__issue_inst0.min.pct_of_peak_sustained_active` | 86.1800 | 79.1700 | 93.1900 | % |
| `smsp__issue_active.max.pct_of_peak_sustained_active` | 0.2200 | 0.1800 | 0.2600 | % |
| `sm__inst_issued.max.pct_of_peak_sustained_active` | 0.2150 | 0.1700 | 0.2600 | % |
| `sm__issue_active.max.pct_of_peak_sustained_elapsed` | 0.2100 | 0.1600 | 0.2600 | % |
| `sm__inst_issued.avg.pct_of_peak_sustained_active` | 0.2050 | 0.1500 | 0.2600 | % |
| `sm__inst_issued.sum.pct_of_peak_sustained_active` | 0.2050 | 0.1500 | 0.2600 | % |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | 0.2050 | 0.1500 | 0.2600 | % |
| `smsp__issue_active.sum.pct_of_peak_sustained_active` | 0.2050 | 0.1500 | 0.2600 | % |
| `sm__issue_active.avg.pct_of_peak_sustained_elapsed` | 0.1950 | 0.1400 | 0.2500 | % |
| `sm__issue_active.sum.pct_of_peak_sustained_elapsed` | 0.1950 | 0.1400 | 0.2500 | % |
| `sm__inst_issued.min.pct_of_peak_sustained_active` | 0.1650 | 0.1300 | 0.2000 | % |
| `sm__issue_active.min.pct_of_peak_sustained_elapsed` | 0.1550 | 0.1200 | 0.1900 | % |
| `smsp__issue_active.min.pct_of_peak_sustained_active` | 0.1400 | 0.1100 | 0.1700 | % |
| `sm__mio_inst_issued.max.pct_of_peak_sustained_elapsed` | 0.1200 | 0.0800 | 0.1600 | % |
| `sm__mio_inst_issued.avg.pct_of_peak_sustained_elapsed` | 0.1150 | 0.0700 | 0.1600 | % |
| `sm__mio_inst_issued.sum.pct_of_peak_sustained_elapsed` | 0.1150 | 0.0700 | 0.1600 | % |
| `sm__mio_inst_issued.min.pct_of_peak_sustained_elapsed` | 0.0950 | 0.0700 | 0.1200 | % |

## roofline_position

| Metric | avg | min | max | unit |
|---|---:|---:|---:|---|
| `sm__pipe_tensor_op_hmma_cycles_active.max.pct_of_peak_sustained_active` | 0.2750 | 0.2500 | 0.3000 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 0.2350 | 0.2300 | 0.2400 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active` | 0.2350 | 0.2300 | 0.2400 | % |
| `sm__pipe_tensor_op_hmma_cycles_active.min.pct_of_peak_sustained_active` | 0.1850 | 0.1800 | 0.1900 | % |
| `sm__inst_executed_pipe_tensor_op_hmma.max.pct_of_peak_sustained_active` | 0.0200 | 0.0200 | 0.0200 | % |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | 0.0150 | 0.0100 | 0.0200 | % |
| `sm__inst_executed_pipe_tensor_op_hmma.sum.pct_of_peak_sustained_active` | 0.0150 | 0.0100 | 0.0200 | % |
| `sm__inst_executed_pipe_tensor_op_hmma.min.pct_of_peak_sustained_active` | 0.0100 | 0.0100 | 0.0100 | % |
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
| `launch__shared_mem_per_block_allocated` | 145664.0000 | 60544.0000 | 230784.0000 | byte/block |
| `launch__shared_mem_per_block` | 145584.0000 | 60464.0000 | 230704.0000 | byte/block |
| `launch__shared_mem_per_block_dynamic` | 144560.0000 | 59440.0000 | 229680.0000 | byte/block |
| `launch__thread_count` | 131072.0000 | 131072.0000 | 131072.0000 | thread |
| `launch__occupancy_per_shared_mem_size` | 3640.0000 | 3640.0000 | 3640.0000 |  |
| `launch__occupancy_per_register_count` | 2056.0000 | 2056.0000 | 2056.0000 |  |
| `launch__shared_mem_per_block_driver` | 1024.0000 | 1024.0000 | 1024.0000 | byte/block |
| `launch__stack_size` | 1024.0000 | 1024.0000 | 1024.0000 |  |
| `launch__occupancy_per_barrier_count` | 520.0000 | 520.0000 | 520.0000 |  |
| `launch__grid_dim_x` | 512.0000 | 512.0000 | 512.0000 |  |
| `launch__grid_size` | 512.0000 | 512.0000 | 512.0000 |  |
| `launch__block_dim_x` | 256.0000 | 256.0000 | 256.0000 | block |
| `launch__block_size` | 256.0000 | 256.0000 | 256.0000 |  |
| `launch__registers_per_thread` | 248.0000 | 248.0000 | 248.0000 | register/thread |
| `launch__registers_per_thread_allocated` | 248.0000 | 248.0000 | 248.0000 | register/thread |
| `launch__sm_count` | 132.0000 | 132.0000 | 132.0000 | SM |
| `launch__occupancy_per_block_size` | 73.0000 | 73.0000 | 73.0000 |  |
