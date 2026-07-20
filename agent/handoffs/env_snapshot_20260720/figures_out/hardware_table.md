| Spec                                | GH200 (Grace Hopper)        | GB200 (Grace Blackwell)             |
| ----------------------------------- | --------------------------- | ----------------------------------- |
| Composition                         | 1 Grace CPU + 1 Hopper GPU  | 1 Grace CPU + 2 Blackwell B200 GPUs |
| Grace : GPU ratio                   | 1 : 1                       | 1 : 2                               |
| CPU cores (Neoverse V2)             | 72                          | 72                                  |
| CPU memory (LPDDR5X)                | 480 GB                      | 480 GB                              |
| Compute per GPU (BF16)              | 990 TFLOP/s                 | 2,500 TFLOP/s                       |
| HBM capacity per GPU                | 96 GB HBM3                  | 186 GB HBM3e                        |
| C2C per GPU — Send / Recv           | 450 / 450 GB/s              | 225 / 225 GB/s                      |

<!-- | HBM bandwidth per GPU               | 4.0 TB/s                    | 8 TB/s                              | -->

Table 1. Per-GPU resources in NVIDIA GH200 and GB200 Superchips. GB200 pairs two
B200 GPUs with one Grace CPU, so each GPU gets half the Grace-side resources and
coherent C2C bandwidth of a GH200 GPU (225/225 versus 450/450 GB/s send/recv).
At the same time, each GPU has much larger HBM and BF16 compute. This is the
regime AsymGEMM targets: longer sequences require more activation and optimizer
state per GPU, and GB200's larger HBM helps consolidate training onto fewer GPUs,
but the smaller per-GPU C2C budget makes naive offload increasingly
bandwidth-limited. Efficient offloading therefore depends on scheduling that
overlaps transfers, prioritizes C2C traffic, and avoids unnecessary HBM staging.
