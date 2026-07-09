# Week 6 Report: Dense Model Integration and Profiling

Date: June 19, 2026

## Goals:

- Extend the Week 5 offload stack beyond MoE models and make it work on dense LLMs without adding model-specific code.
- Profile the full stack across both MoE and dense models against zero3_offload and superoffload baselines.
- Measure whether activation offload still gives a meaningful HBM reduction at larger dense-model scales.

## Progress:

This week focused on getting the dense path working cleanly and then checking the numbers across a wider set of models. A generic dense decoder matcher now covers Qwen3-32B, Llama-3.3-70B-Instruct, and Qwen2.5-72B-Instruct. Dense MLP activations are handled through the whole-layer layer_gc path, while attention uses the existing attention offload path. The expert offload flag does not do anything for dense models right now, since that path is only wired into MoE expert engines.

The main result is that the HBM savings still hold on dense models, and the win gets stronger as the FFN gets wider:

| Model | Workload | Offload HBM | Recompute HBM | zero3_offload HBM | HBM Change vs Recompute | HBM Change vs zero3_offload | Step Time | Host RAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-32B | s4096 b4 | 28.6 GiB | 38.5 GiB | 40.3 GiB | -25.9% | -29.1% | 20.3 s | 582.9 GiB |
| Llama-3.3-70B-Instruct | s4096 b4 | 24.6 GiB | 44.6 GiB | 46.4 GiB | -44.9% | -47.1% | 27.9 s | 746.3 GiB |
| Qwen2.5-72B-Instruct | s4096 b4 | 28.9 GiB | 48.9 GiB | 51.1 GiB | -40.9% | -43.4% | 30.1 s | 762.3 GiB |

The MoE runs show the same tradeoff seen in Week 5. On Qwen3-30B-A3B, offload reduced peak HBM from 31.3 GiB to 28.3 GiB at s4096 b4, from 62.3 GiB to 56.5 GiB at s8192 b4, and from 124.5 GiB to 112.7 GiB at s8192 b8. The cost is still CPU-bound backward time and much higher host RAM. For example, Qwen3-30B-A3B at s8192 b8 reached 112.7 GiB HBM with offload, but step time rose to 152.1 s and host RAM reached 665.2 GiB.

Llama-4-Scout stayed memory-limited on the larger offload cases. The s4096 b4 offload run reached 19.7 GiB HBM with a 71.4 s step time and 802.4 GiB host RAM, but the s8192 offload runs failed on host RAM. That makes host-side scheduling and placement the next bottleneck to address.

## Next Steps:

- Develop a scheduler that allocates tensors and transfer work across CPU, GPU, and NVMe instead of treating host memory as one large fallback pool.
- Add placement rules that decide when activations should stay on GPU, move to CPU, or spill to NVMe based on size, reuse distance, and transfer cost.
- Use the scheduler to reduce host RAM pressure on the MoE long-context runs, especially the Llama-4-Scout s8192 cases that currently fail on host RAM.
- Re-profile the dense and MoE workloads after scheduler integration to check whether the HBM savings can be kept without pushing backward time and host RAM as high.
