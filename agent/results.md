# Profiling Results

Workload is `seq,batch,ga`. Policy is `expert,exp_act,attn_act,layer_act,layer_gc`.
Rows are deduplicated by visible config; if an older artifact omitted the Liger-loss path tag, it is treated as `ligerloss0`, and the newer explicit `__ligerloss0` artifact is kept for duplicate configs.

## Qwen/Qwen3-30B-A3B

| workload | backend | policy | tag | peak_alloc_hbm | peak_reserved_hbm | fwd_ms | bwd_ms | optimizer_ms | step_ms |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 4096,8,1 | asym_cpuadamwds,norecomp | gc-layer,false,false,false,false | ligerloss0 | 63,831.98 MiB | 74,966.00 MiB | 2,395.38 | 9,479.91 | 2,100.50 | 14,295.84 |
| 4096,8,1 | asym_cpuadamwds,norecomp | none,true,true,false,true | ligerloss0 | 58,024.19 MiB | 68,706.00 MiB | 5,242.57 | 75,717.71 | 2,279.37 | 83,706.18 |
| 4096,8,1 | asym_cpuadamwds,norecomp | none,true,true,true,false | ligerloss0 | 58,036.80 MiB | 68,566.00 MiB | 5,667.31 | 75,512.12 | 2,191.08 | 83,851.12 |
| 4096,8,1 | asym_cpuadamwds,recomp | none,false,false,false,false | ligerloss0 | 63,831.98 MiB | 74,966.00 MiB | 4,920.24 | 10,061.31 | 2,219.17 | 17,402.35 |
| 4096,8,1 | zero3_offload,recomp | none,false,false,false,false | ligerloss0 | 65,686.09 MiB | 76,046.00 MiB | 6,094.26 | 26,464.71 | 31.76 | 35,104.66 |
| 8192,8,1 | asym_cpuadamwds,norecomp | none,true,true,false,true | ligerloss0 | 115,755.06 MiB | 136,366.00 MiB | 8,923.24 | 139,973.60 | 2,372.62 | 151,439.55 |
| 8192,8,1 | asym_cpuadamwds,norecomp | none,true,true,true,false | ligerloss0 | 115,928.49 MiB | 136,108.00 MiB | 9,198.84 | 138,589.37 | 2,193.87 | 150,267.20 |
| 8192,8,1 | zero3_offload,recomp | none,false,false,false,false | ligerloss0 | 129,192.99 MiB | 149,706.00 MiB | 6,521.65 | 27,844.55 | 23.90 | 36,761.19 |

## Qwen/Qwen3.5-35B-A3B

| workload | backend | policy | tag | peak_alloc_hbm | peak_reserved_hbm | fwd_ms | bwd_ms | optimizer_ms | step_ms |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 2048,2,1 | asym_cpuadamwds,norecomp | gc-attn-exp,false,false,false,false | ligerloss0 | 33,166.30 MiB | 35,186.00 MiB | 13,055.67 | 14,752.07 | 3,445.70 | 30,281.21 |
| 2048,2,1 | asym_cpuadamwds,norecomp | gc-exp,false,false,false,false | ligerloss0 | 36,311.64 MiB | 38,368.00 MiB | 32,699.28 | 19,211.72 | 3,404.72 | 54,490.87 |
| 2048,2,1 | asym_cpuadamwds,norecomp | gc-layer,false,false,false,false | ligerloss0 | 22,018.12 MiB | 24,326.00 MiB | 9,799.86 | 8,917.65 | 3,398.64 | 21,171.69 |
| 2048,2,1 | asym_cpuadamwds,norecomp | none,true,false,false,false | ligerloss0 | 31,191.64 MiB | 33,248.00 MiB | 34,179.04 | 71,679.17 | 3,493.88 | 108,438.46 |
| 2048,2,1 | asym_cpuadamwds,norecomp | none,true,true,false,false | ligerloss0 | 27,891.64 MiB | 29,948.00 MiB | 14,654.96 | 34,475.62 | 3,427.64 | 51,654.03 |
| 2048,2,1 | asym_cpuadamwds,norecomp | none,true,true,false,true | ligerloss0 | 19,544.91 MiB | 19,546.00 MiB | 5,390.45 | 16,982.91 | 3,401.07 | 24,852.27 |
| 2048,2,1 | zero3_offload,recomp | none,false,false,false,false | ligerloss0 | 19,605.80 MiB | 19,724.00 MiB | 5,209.80 | 21,399.51 | 40.37 | 29,117.88 |
| 2048,4,1 | asym_cpuadamwds,norecomp | none,true,false,false,false | ligerloss0 | 62,069.73 MiB | 66,150.00 MiB | 3,899.38 | 16,496.84 | 3,462.82 | 22,950.55 |
| 2048,4,1 | asym_cpuadamwds,norecomp | none,true,true,false,false | ligerloss0 | 55,469.74 MiB | 59,530.00 MiB | 3,890.64 | 17,774.94 | 3,453.95 | 24,219.44 |
| 2048,4,1 | asym_cpuadamwds,norecomp | none,true,true,false,true | ligerloss0 | 30,142.65 MiB | 35,026.00 MiB | 2,215.53 | 15,895.72 | 3,445.40 | 20,639.82 |
| 4096,4,1 | asym_cpuadamwds,norecomp | none,true,false,false,false | ligerloss0 | 123,825.92 MiB | 132,086.00 MiB | 16,712.84 | 50,947.17 | 3,509.70 | 70,345.10 |
| 4096,4,1 | asym_cpuadamwds,norecomp | none,true,true,false,false | ligerloss0 | 110,615.92 MiB | 118,906.00 MiB | 5,780.82 | 29,176.14 | 3,422.99 | 37,554.28 |
| 4096,4,1 | asym_cpuadamwds,norecomp | none,true,true,false,true | ligerloss0 | 59,932.37 MiB | 69,546.00 MiB | 2,474.04 | 24,475.70 | 3,472.46 | 29,330.06 |
| 4096,8,1 | asym_cpuadamwds,norecomp | gc-layer,false,false,false,false | ligerloss0 | 99,074.56 MiB | 115,646.00 MiB | 9,554.83 | 20,647.19 | 3,593.69 | 32,695.84 |
| 4096,8,1 | asym_cpuadamwds,norecomp | none,true,true,false,true | ligerloss0 | 119,551.81 MiB | 138,426.00 MiB | 9,641.03 | 58,340.06 | 4,290.60 | 70,493.91 |
| 4096,8,1 | asym_cpuadamwds,recomp | none,false,false,false,false | ligerloss0 | 99,074.56 MiB | 115,646.00 MiB | 2,626.35 | 14,396.32 | 3,409.16 | 19,511.43 |
| 4096,8,1 | zero3_offload,recomp | none,false,false,false,false | ligerloss0 | 101,461.68 MiB | 131,144.00 MiB | 5,071.38 | 22,019.68 | 41.01 | 29,616.98 |
| 8192,4,1 | asym_cpuadamwds,norecomp | none,true,true,false,true | ligerloss0 | 119,551.81 MiB | 138,426.00 MiB | 23,733.20 | 90,192.55 | 4,353.76 | 116,597.25 |
| 8192,4,1 | zero3_offload,recomp | none,false,false,false,false | ligerloss0 | 101,461.68 MiB | 131,144.00 MiB | 5,273.19 | 22,216.92 | 41.79 | 30,138.60 |

## meta-llama/Llama-4-Scout-17B-16E

| workload | backend | policy | tag | peak_alloc_hbm | peak_reserved_hbm | fwd_ms | bwd_ms | optimizer_ms | step_ms |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 4096,4,1 | asym_cpuadamwds,norecomp | none,true,true,false,true | ligerloss0 | 47,626.36 MiB | 54,568.00 MiB | 12,142.93 | 57,760.70 | 2,413.22 | 72,503.25 |
| 4096,4,1 | asym_cpuadamwds,norecomp | none,true,true,true,false | ligerloss0 | 47,634.38 MiB | 54,628.00 MiB | 10,702.12 | 60,642.77 | 3,209.83 | 73,897.80 |
| 4096,4,1 | asym_cpuadamwds,recomp | none,false,false,false,false | ligerloss0 | 28,094.45 MiB | 29,188.00 MiB | 3,916.45 | 13,124.33 | 2,220.84 | 19,744.44 |
| 4096,4,1 | zero3_offload,recomp | none,false,false,false,false | ligerloss0 | 50,716.93 MiB | 57,268.00 MiB | 29,535.95 | 27,583.55 | 47.55 | 59,637.52 |
| 4096,8,1 | zero3_offload,recomp | none,false,false,false,false | ligerloss0 | 92,327.46 MiB | 104,008.00 MiB | 33,206.31 | 35,247.82 | 47.75 | 71,055.11 |
