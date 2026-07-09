# Week 4 Report: Larger MoE SFT Baselines

Date: June 5, 2026

## Goals:

- Move the MoE LoRA SFT evaluation from small integration runs to heavier training workloads.
- Add external baselines so AsymGEMM can be compared against SuperOffload, ZeRO-3 offload, and KTransformers ARM CPU.
- Compare memory and speed across the systems and identify the next memory target for AsymGEMM.

## Progress:

This week focused on making the MoE SFT evaluation more realistic. The main larger run was Qwen3-30B-A3B with batch size 4, sequence length 8192, BF16 training, and LoRA rank 64. We also ran additional Qwen and Llama-family MoE configurations to make sure the profiling and training paths worked beyond the original small tests.

The baseline stack is now in place. AsymGEMM is still much faster than a CPU expert-compute system like KTransformers ARM, but on this heavier Qwen3 run it does not yet beat SuperOffload or ZeRO-3 offload on peak GPU memory.

| System | Workload | Forward | Backward | Step Time | Peak Reserved GPU Memory | Main Takeaway |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| AsymGEMM | Qwen3-30B-A3B, batch 4, seq 8192 | 2.28 s | 4.40 s | 6.68 s | 91.0 GB | Moves routed expert base weights off GPU, but total GPU memory is still high |
| SuperOffload | Qwen3-30B-A3B, batch 4, seq 8192 | 1.68 s | 3.17 s | 4.85 s | 78.8 GB | Strongest memory baseline on this run |
| ZeRO-3 offload | Qwen3-30B-A3B, batch 4, seq 8192 | 1.68 s | 3.12 s | 4.80 s | 78.8 GB | Matches SuperOffload closely for this workload |
| KTransformers ARM CPU | Qwen3-30B-A3B, batch 1, seq 4096 | not recorded | not recorded | 160.46 s | 79.1 GB | Much slower because expert compute runs on CPU |

The KTransformers result is not the same batch-4 workload, so it is not a direct speed comparison. It is useful as the CPU expert-compute reference point: AsymGEMM should stay much faster than that path while reducing GPU memory.

AsymGEMM did move the routed expert base weights out of GPU memory. On the large Qwen3 run, routed expert weights used 221.2 GB of CPU memory, with 55.3 GB pinned. The remaining GPU footprint was still large: LoRA trainable parameters used 6.5 GB on GPU, LoRA optimizer state used 13.1 GB, and the full run peaked at 91.0 GB reserved GPU memory. That means the next memory problem is outside the offloaded expert base weights.

The largest runtime costs were still the routed expert matrix multiplications. Gate/up expert projection took 0.78 s, gate/up input-gradient projection took 0.74 s, down expert projection took 0.67 s, and down input-gradient projection took 0.58 s. Those costs matter more at the larger batch and sequence length because activations, LoRA state, and non-expert model weights all put more pressure on GPU memory.

## Next Steps:

- Profile larger MoEs, especially Qwen3-235B-A22B and larger Qwen3.5-style expert configurations, using the same baseline table.
- Reduce AsymGEMM GPU memory outside expert base weights, especially LoRA optimizer state and saved activations.
- Improve expert projection speed for the GPU-compute, CPU-weight path.
- Use SuperOffload as the near-term memory target and KTransformers ARM as the speed baseline that AsymGEMM should comfortably beat.
