# M4.5 Profile Summary

Generated: 2026-05-19T09:53:02.598376+00:00

This summary compares latency, GPU/HBM memory, CPU memory, and pinned CPU cost across MLP, dense LLM, and MoE profiling workloads.

| Workload | Total+Setup s | Steady s | Mean s | p95 s | Setup s | Forward s | Backward s | Optimizer s | Host Init s | Host Pin s | W.T Mat s | Peak HBM | HBM Saved | Pinned CPU | W Host | W.T Host | Direct Fwd/dX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mlp | 0.678504 | 0.001802 | 0.001888 | 0.002023 | 0.676702 | 0.000726 | 0.000734 | 0.000276 | 0.009309 | 0.001111 | 0.000000 | 67498496 | 131072 | 131072 | 131072 | 0 | True/True |
| dense_llm | 0.211855 | 0.023016 | 0.023479 | 0.023993 | 0.188840 | 0.011629 | 0.010556 | 0.000730 | 0.161812 | 0.001051 | 0.000000 | 89690112 | 1310720 | 1310720 | 1310720 | 0 | True/True |
| tiny_moe | 0.545695 | 0.048777 | 0.049273 | 0.050516 | 0.496918 | 0.024826 | 0.022584 | 0.001306 | 0.407877 | 0.004521 | 0.000000 | 126932992 | 3932160 | 3932160 | 3932160 | 0 | True/True |
| qwen3_8b | 15.034283 | 0.030531 | 0.030741 | 0.031449 | 15.003752 | 0.016195 | 0.013833 | 0.000422 | 0.402592 | 0.227472 | 0.000000 | 241222656 | 436207616 | 436207616 | 436207616 | 0 | True/True |
| qwen3_14b | 40.546161 | 0.047233 | 0.047457 | 0.048200 | 40.498928 | 0.024793 | 0.021939 | 0.000421 | 0.695514 | 0.450718 | 0.000000 | 300348416 | 744488960 | 744488960 | 744488960 | 0 | True/True |
| qwen3_30b_a3b | 37.207014 | 0.168508 | 0.170157 | 0.176016 | 37.038505 | 0.082610 | 0.081818 | 0.004003 | 27.382430 | 0.780582 | 0.000000 | 1970814976 | 1207959552 | 1207959552 | 1207959552 | 0 | True/True |
| qwen3_32b | 3.930439 | 0.061185 | 0.061391 | 0.062241 | 3.869253 | 0.032146 | 0.028533 | 0.000427 | 0.598282 | 0.066761 | 0.000000 | 326562816 | 996147200 | 996147200 | 996147200 | 0 | True/True |

Component timers may overlap their parent forward phase; top-level phase percentages are non-overlapping.
Deep profile fields are Python-observed setup and host-weight costs; direct host fetch traffic inside kernels requires Nsight/NCU for hardware counters.
