# Week 5 Report: Full Training-State Offload Stack

Date: June 12, 2026

## Goals:

- Extend AsymGEMM from expert-weight offload into a full training-state offload stack.
- Move non-expert frozen weights, optimizer state, LoRA weights, LoRA grads, and activations out of HBM behind independent toggles.
- Beat SuperOffload on peak allocated and reserved HBM at b4 s8192 and b8 s8192.

## Progress:

Week 5 addressed the main gap from Week 4. Before this, only routed expert base weights left HBM. Other frozen weights, optimizer state, LoRA weights, LoRA grads, and activations still stayed on GPU, which is why AsymGEMM was above SuperOffload on peak memory.

The new stack adds five offload paths: whole-model frozen-weight offload, CPU AdamW optimizer state, LoRA grad offload, LoRA weight gather and release, and fine-grained activation offload. These paths can be enabled independently, so the stack now gives a memory and latency curve instead of one fixed operating point.

The headline numbers were still marked preliminary in this report, with final runs pending, but they show the memory crossover that Week 4 was targeting:

| Workload | Tokens | Backend | Status | Alloc HBM | Reserved HBM | Step Time | Forward | Backward | Optimizer |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| b4 s8192 | 32,768 | AsymGEMM | ok | 56.8 GiB | 66.7 GiB | 78.0 s | 5.2 s | 69.5 s | 0.9 s |
| b4 s8192 | 32,768 | SuperOffload | ok | 64.1 GiB | 74.2 GiB | 8.3 s | 1.8 s | 5.9 s | 0.02 s |
| b8 s8192 | 65,536 | AsymGEMM | ok | 113.2 GiB | 132.9 GiB | 153.4 s | 9.2 s | 140.5 s | 1.2 s |
| b8 s8192 | 65,536 | SuperOffload | ok | 126.2 GiB | 146.2 GiB | 10.0 s | 2.3 s | 7.1 s | 0.02 s |

AsymGEMM is below SuperOffload on both peak allocated and reserved HBM: 7.3 GiB allocated and 7.5 GiB reserved lower at b4 s8192, and 13.0 GiB allocated and 13.3 GiB reserved lower at b8 s8192. That reverses the Week 4 result, where AsymGEMM was at 91.0 GB reserved and SuperOffload was at 78.8 GB.

The cost is step time. The full activation-offload configuration is about 9x slower at b4 s8192 and about 15x slower at b8 s8192. Almost all of that slowdown is in backward, where CPU-side elementwise work and activation staging dominate.

The cheaper memory wins came from the frozen-weight, optimizer, and LoRA paths. On Llama-4-Scout at b4 s8192, switching from routed experts only to all frozen-weight buckets cut the persistent HBM residual from 40.31 GiB to 17.45 GiB. CPU AdamW saved 8.15 GiB peak allocated HBM and 11.48 GiB peak reserved HBM for about 1.2 s of optimizer time per step. On Qwen3-30B-A3B at b4 s4096, LoRA weight offload dropped peak allocated HBM from 34.59 GiB to 28.54 GiB, removed about 6.19 GiB of resident expert-LoRA weights at the loss peak, kept latency near 1.02x, and kept loss tracking the baseline step by step.

Fine-grained activation offload was the largest new capability and the main reason the memory crossover happened. It offloads individual expert activations to pinned CPU memory, runs cheap elementwise work on CPU, and fetches only the operands needed by GPU GEMMs through AsymGEMM. Combined with attention and layer-level activation offload, Qwen3-30B-A3B at b4 s4096 ran at about 34 to 39 GiB peak HBM instead of about 180 GiB with no activation offload or recompute.

## Next Steps:

- Profile the same full stack on larger MoEs, especially Qwen3-235B-A22B, Qwen3.5, and Llama4, to confirm the memory crossover holds at scale.
- Reduce the CPU-bound backward by overlapping CPU elementwise work and staging with GPU GEMMs on a dedicated stream.
- Cut down CPU to GPU round trips in the fine-grained activation path.
- Add a CPU-activation budget and HBM watermark so idle GPU memory can keep the lowest-latency configuration that still fits the target sequence length.
