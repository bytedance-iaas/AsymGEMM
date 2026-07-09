# Week 3 Report: LlamaFactory Integration and LoRA SFT Profiling

Date: May 29, 2026

## Goals:

- Integrate AsymGEMM into the LlamaFactory SFT workflow for end-to-end MoE LoRA training.
- Profile LoRA SFT forward and backward latency, memory, module, operation, and kernel breakdowns.
- Add and profile token-count-aware activation recompute for routed experts.

## Progress:

This week moved the work from synthetic profiling into the LlamaFactory SFT workflow. The integration now supports a torch backend with routed expert weights in GPU memory and an asym backend with routed expert base weights served through AsymGEMM. The profiling scripts handle LlamaFactory launch, Nsight collection, source-level metrics, loss comparison, and postprocessing. The torch-vs-asym loss comparison completed successfully for the profiled Qwen3 MoE SFT configuration.

The representative LlamaFactory profile used Qwen3-30B-A3B in BF16 with batch size 1, sequence length 4096, LoRA rank 8, LoRA alpha 16, 5 warmup steps, 15 measured steps, and Nsight profiling.

| Metric | AsymGEMM | Torch | Delta |
| --- | ---: | ---: | --- |
| Step time | 1458.98 ms | 729.73 ms | asym 2.00x slower |
| Forward time | 773.10 ms | 396.84 ms | asym 1.95x slower |
| Backward time | 686.25 ms | 333.28 ms | asym 2.06x slower |
| Peak HBM | 55155.98 MiB | 110483.62 MiB | asym saves 55327.65 MiB |

The LlamaFactory result preserves the Week 2 memory-capacity result at end-to-end training scale. AsymGEMM saves substantial HBM, while the throughput cost is still dominated by routed expert base AsymGEMM.

| Operation | Total Time |
| --- | ---: |
| MLP up base AsymGEMM | 551.05 ms |
| MLP down base AsymGEMM | 340.02 ms |
| Base AsymGEMM gaps and overhead | 206.47 ms |

LoRA, routing, scatter/combine, and activation work are visible in the trace, but they remain secondary compared with the remote base expert GEMM path. Isolated LoRA operator profiling also showed that LoRA memory is small compared with routed expert base weights, while LoRA backward can still be nontrivial.

| Shape | Backend | Forward | Backward |
| --- | --- | ---: | ---: |
| 8192 x 768 to 2048, rank 16 | asym | 0.324 ms | 4.327 ms |
| 8192 x 768 to 2048, rank 16 | torch | 0.078 ms | 0.152 ms |
| 8192 x 2048 to 768, rank 16 | asym | 0.554 ms | 3.564 ms |
| 8192 x 2048 to 768, rank 16 | torch | 0.075 ms | 0.156 ms |

Token-count-aware expert recompute was implemented and profiled with threshold policies and activation-drop variants. The machinery works, reports recomputed experts and routes, and can select small-token experts by threshold. The result is mixed: naive recompute can increase runtime and does not automatically reduce peak HBM. In the current implementation, full activation recompute gives clearer HBM savings than partial expert recompute.

## Next Steps:

- Optimize routed expert base AsymGEMM, especially the gate/up and down projection paths.
- Improve the token-count-aware recompute heuristic using measured saved bytes versus recompute cost.
- Continue profiling LlamaFactory runs across longer sequences and larger MoE models.
- Revisit LoRA backward fusion after the base expert bottleneck is reduced.
