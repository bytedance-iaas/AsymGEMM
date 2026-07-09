# Week 2 Report: CPU-Resident Expert Weights for MoE LoRA SFT

Date: May 22, 2026

## Goals:

- Profile AsymGEMM as a memory-hierarchy technique for MoE LoRA SFT.
- Compare torch with GPU-resident routed expert weights against AsymGEMM with CPU-pinned routed expert weights.
- Measure the HBM savings, throughput cost, and main bottlenecks for remote routed expert GEMM.

## Progress:

This week tested the core systems idea: keep routed expert base weights outside GPU HBM while still running the expert GEMMs on the GPU. The comparison used a torch backend with routed expert base weights in HBM and an asym backend with routed expert base weights in CPU pinned memory, read by GPU AsymGEMM.

The main result is a clear memory-throughput tradeoff. AsymGEMM cut peak HBM from 4520 MiB to 2216 MiB, saving 2304 MiB. The cost was throughput: step time was 1.67x slower, forward was 2.34x slower, and backward was 1.91x slower.

| Metric | AsymGEMM | Torch | Delta |
| --- | ---: | ---: | --- |
| Source-level step time | 56.92 ms | 33.99 ms | asym 1.67x slower |
| Nsight forward time | 31.73 ms | 13.57 ms | asym 2.34x slower |
| Nsight backward time | 23.37 ms | 12.22 ms | asym 1.91x slower |
| Optimizer time | 1.74 ms | 1.81 ms | similar |
| GPU peak HBM | 2216 MiB | 4520 MiB | asym saves 2304 MiB |
| GPU base-weight and buffer memory | 104 MiB | 2408 MiB | asym moves base weights off GPU |
| CPU pinned expert weights | 2304 MiB | 0 MiB | one host-resident expert-weight copy |
| Saved activations | 908 MiB | 908 MiB | unchanged |
| LoRA params, grads, AdamW state | 264 MiB, 264 MiB, 528 MiB | same | unchanged |

The experiment used a Qwen3-MoE-style 30B-A3B matched configuration with 2 profiled layers, batch size 32, sequence length 64, 2048 logical tokens, hidden size 2048, 128 experts, top-k 8, expert intermediate size 768, LoRA rank 64, LoRA alpha 128, BF16 LoRA, and activation recompute off.

The bottleneck is routed expert base compute. Gate, up, and down projection time increased from 1.46 ms total with torch to 29.79 ms total with AsymGEMM.

| Routed Expert Base Op | AsymGEMM | Torch |
| --- | ---: | ---: |
| Gate projection | 11.65 ms | 0.50 ms |
| Down projection | 9.09 ms | 0.49 ms |
| Up projection | 9.04 ms | 0.48 ms |
| Total | 29.79 ms | 1.46 ms |

The direct AsymGEMM path was used for all routed expert base calls: 240 total calls, 120 forward calls, 120 backward input-gradient calls, and no fallback reasons. That isolates the current throughput issue to the remote expert GEMM path instead of fallback behavior.

Saved activations did not change between torch and asym. The largest saved activation owners were routed expert gate/up LoRA at 312 MiB, routed expert SiLU-times-up activation at 288 MiB, attention scores matmul at 64 MiB, routed expert down LoRA at 52 MiB, attention layernorm at 32 MiB, and MoE layernorm at 32 MiB.

The main risks are also clear from the profile. Remote expert GEMM may stay too slow without tiling, grouping, layout, and fusion work. CPU pinned memory will become more important at larger expert counts. Short expert-token groups may keep kernels underutilized. Output-side LoRA fusion may help less than base AsymGEMM optimization. Hardware and interconnect sensitivity still need to be reported directly.

## Next Steps:

- Profile output-side LoRA fusion for base output plus LoRA output, with the goal of reducing large-output materialization and separate add or cast work.
- Test sequence length sensitivity at 500, 700, and 10000 tokens to understand how expert-token group size changes the asym bottleneck.
- Evaluate dynamic recompute by expert token count so small-token experts can be recomputed when saving activation state is not worth it.
- Profile fused gate/up input-gradient so the two input-gradient projections and the final accumulation can be reduced into one fused path.
- Track CPU pinned expert-weight footprint separately from HBM and report bandwidth or interconnect sensitivity as the runs scale up.
