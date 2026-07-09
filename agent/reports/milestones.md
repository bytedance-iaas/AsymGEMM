# Milestones

- Milestone 1: Profile AsymGEMM vs GPU-resident PyTorch for MoE LoRA SFT and quantify HBM savings, step latency, forward/backward latency, and GEMM bottlenecks.
- Milestone 2: Profile token-count-aware recompute, LoRA memory/timing, sequence-length sensitivity, and backward bottlenecks; decide which bottlenecks are worth prioritizing.
- Milestone 3: Build and evaluate a simple threshold heuristic for recomputing small routed-expert groups.
- Milestone 4: Prototype or simulate fused gate/up `dX`, then compare separate vs fused backward on launches, remote reads, memory, and latency.
- Milestone 5: Re-run the main profile with the best current optimizations and freeze baseline configs, metrics, and comparison methodology.
- Milestone 6: Improve grouped AsymGEMM scheduling/layout for routed expert token groups and finish LLaMA-Factory integration for a real training workflow.
- Milestone 7: Bring up KTransformers on the same MoE SFT workload, adapt it for BH200 ARM CPU constraints, and compare against AsymGEMM.
- Milestone 8: Deliver a working AsymGEMM MoE LoRA SFT system integrated into LLaMA-Factory with end-to-end comparison against KTransformers.
- Milestone 9: Package final results, plots, ablations, limitations, and paper-style report materials.
- Stretch Milestone: Finish paper writing after the system and comparison results are stable.
