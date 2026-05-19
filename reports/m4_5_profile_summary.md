# M4.5 Profile Summary

Generated: 2026-05-18T21:59:01.026393+00:00

This summary compares latency, GPU/HBM memory, CPU memory, and pinned CPU cost across MLP, dense LLM, and MoE profiling workloads.

| Workload | Total s | Forward s | Backward s | Optimizer s | Peak HBM | HBM Saved | Pinned CPU | W Host | W.T Host | Direct Fwd/dX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mlp | 0.701179 | 0.000790 | 0.000814 | 0.000308 | 67498496 | 131072 | 262144 | 131072 | 131072 | True/True |
| dense_llm | 0.211579 | 0.012641 | 0.010641 | 0.000749 | 89690112 | 1310720 | 2621440 | 1310720 | 1310720 | True/True |
| tiny_moe | 0.520009 | 0.023832 | 0.022195 | 0.001325 | 126932992 | 3932160 | 7864320 | 3932160 | 3932160 | True/True |

Component timers may overlap their parent forward phase; top-level phase percentages are non-overlapping.
