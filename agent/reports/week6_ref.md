## Profiling Metrics Subset

### llama-4-scout-17b-16e

| Workload | Backend | Config | Status | fwd_s | bwd_s | opt_s | step_s | fwd_H | bwd_H | step_H | RAM |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s4096 b4 | asym_cpuadamwds (norecomp) | none+exp+attn-offload+layerGC [lg- sd+] | ok | 6.4 | 58.9 | 1.4 | 71.4 | 19.7 | 19.3 | 19.7 | 802.4 |
| s4096 b4 | asym_cpuadamwds (recomp) | none (no offload) [lg- sd-] | ok | 3.4 | 10.6 | 1.3 | 18.9 | 27.4 | 27.2 | 27.4 | 580.6 |
| s4096 b4 | zero3_offload (recomp) | none (no offload) [lg- sd-] | ok | 8.3 | 19.1 | 0.1 | 29.2 | 39.0 | 52.1 | 52.1 | 525.9 |
| s4096 b4 | superoffload (recomp) | none (no offload) [lg- sd-] | ok | 8.1 | 18.7 | 0.0 | 28.5 | 39.0 | 52.1 | 52.1 | 525.4 |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| s8192 b4 | asym_cpuadamwds (norecomp) | none+exp+attn-offload+layerGC [lg- sd-] | OOM (host RAM) | - | - | - | - | - | - | - | - |
| s8192 b4 | asym_cpuadamwds (recomp) | none (no offload) [lg- sd-] | ok | 5.5 | 15.4 | 1.3 | 26.0 | 54.4 | 54.2 | 54.4 | 581.5 |
| s8192 b4 | zero3_offload (recomp) | none (no offload) [lg- sd-] | ok | 10.6 | 27.1 | 0.1 | 39.6 | 66.0 | 94.3 | 94.3 | 525.7 |
| s8192 b4 | superoffload (recomp) | none (no offload) [lg- sd-] | not recorded | - | - | - | - | - | - | - | - |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| s8192 b8 | asym_cpuadamwds (norecomp) | none+exp+attn-offload+layerGC [lg- sd-] | OOM (host RAM) | - | - | - | - | - | - | - | - |
| s8192 b8 | asym_cpuadamwds (recomp) | none (no offload) [lg- sd-] | ok | 10.1 | 25.2 | 1.6 | 40.7 | 108.7 | 108.3 | 108.7 | 582.0 |
| s8192 b8 | zero3_offload (recomp) | none (no offload) [lg- sd-] | OOM (GPU) | - | - | - | - | - | - | - | - |
| s8192 b8 | superoffload (recomp) | none (no offload) [lg- sd-] | OOM (GPU) | - | - | - | - | - | - | - | - |

### qwen3-30b-a3b

| Workload | Backend | Config | Status | fwd_s | bwd_s | opt_s | step_s | fwd_H | bwd_H | step_H | RAM |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s4096 b4 | asym_cpuadamwds (norecomp) | none+exp+attn-offload+layerGC [lg- sd+] | ok | 2.7 | 35.2 | 1.1 | 41.9 | 23.8 | 28.3 | 28.3 | 326.3 |
| s4096 b4 | asym_cpuadamwds (recomp) | none (no offload) [lg- sd-] | ok | 1.7 | 7.9 | 1.1 | 13.4 | 26.7 | 31.3 | 31.3 | 205.4 |
| s4096 b4 | zero3_offload (recomp) | none (no offload) [lg- sd-] | ok | 1.8 | 8.4 | 0.0 | 11.1 | 28.5 | 33.1 | 33.1 | 196.2 |
| s4096 b4 | superoffload (recomp) | none (no offload) [lg- sd-] | ok | 1.8 | 8.7 | 0.0 | 11.4 | 28.5 | 33.1 | 33.1 | 196.3 |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| s8192 b4 | asym_cpuadamwds (norecomp) | none+exp+attn-offload+layerGC [lg- sd+] | ok | 5.1 | 71.4 | 1.1 | 80.5 | 47.3 | 56.5 | 56.5 | 440.1 |
| s8192 b4 | asym_cpuadamwds (recomp) | none (no offload) [lg- sd-] | ok | 3.9 | 10.2 | 1.1 | 18.1 | 53.2 | 62.3 | 62.3 | 205.5 |
| s8192 b4 | zero3_offload (recomp) | none (no offload) [lg- sd-] | ok | 1.9 | 8.5 | 0.0 | 11.3 | 54.9 | 64.1 | 64.1 | 196.6 |
| s8192 b4 | superoffload (recomp) | none (no offload) [lg- sd-] | ok | 1.9 | 8.6 | 0.0 | 11.5 | 54.9 | 64.1 | 64.1 | 196.3 |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| s8192 b8 | asym_cpuadamwds (norecomp) | none+exp+attn-offload+layerGC [lg- sd+] | ok | 9.3 | 138.2 | 1.5 | 152.1 | 94.4 | 112.7 | 112.7 | 665.2 |
| s8192 b8 | asym_cpuadamwds (recomp) | none (no offload) [lg- sd-] | ok | 4.5 | 12.7 | 1.4 | 21.4 | 106.2 | 124.5 | 124.5 | 211.6 |
| s8192 b8 | zero3_offload (recomp) | none (no offload) [lg- sd-] | ok | 2.4 | 9.3 | 0.0 | 12.7 | 107.6 | 126.2 | 126.2 | 197.3 |
| s8192 b8 | superoffload (recomp) | none (no offload) [lg- sd-] | ok | 2.4 | 9.3 | 0.0 | 12.6 | 107.6 | 126.2 | 126.2 | 196.7 |

### qwen3-32b

Dense model (no routed experts). The dense-MLP activations **are** offloaded to CPU — but via the whole-layer `layer_gc` hooks (`layer_glue_gc` wraps all 64 decoder layers incl. their MLP), not via a dedicated `exp` path: `ASYMM_EXPERT_ACT_OFFLOAD` is currently wired only to MoE expert engines, so the `exp` flag is a no-op for dense (redundant with `layer_gc`). A surgical dense-MLP `exp` offload (CPU-side down-proj LoRA backward through AsymGEMM, like the expert engine) is a TODO. The HBM win (asym-offload 28.6 GiB; `−25.9%` vs asym-recompute, `−29.1%` vs zero3_offload) comes from attention act-offload + layer glue-GC (MLP+rest) + SDPA recompute. All 65.5 GB of base weights sit on CPU in every asym row; the step-peak is workspace (~18.5 GiB, logits-dominated) + resident activations (9.9 offload vs 19.9 recompute), paid for in host RAM (582.9 GiB vs ~85) and step time.

| Workload | Backend | Config | Status | fwd_s | bwd_s | opt_s | step_s | fwd_H | bwd_H | step_H | RAM |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s4096 b4 | asym_cpuadamwds (norecomp) | none+exp+attn-offload+layerGC [lg- sd+] | ok | 6.7 | 9.4 | 1.1 | 20.3 | 24.1 | 28.6 | 28.6 | 582.9 |
| s4096 b4 | asym_cpuadamwds (recomp) | none (no offload) [lg- sd-] | ok | 2.9 | 10.8 | 1.1 | 17.5 | 34.0 | 38.5 | 38.5 | 137.6 |
| s4096 b4 | zero3_offload (recomp) | none (no offload) [lg- sd-] | ok | 2.8 | 8.8 | 0.1 | 12.9 | 35.6 | 40.3 | 40.3 | 85.0 |

### llama-3.3-70b-instruct

Dense model, **bias-free** attention, 80 layers / FFN 28672. Enabled by one generic dense decoder-matcher branch (`LlamaDecoderLayer` isn't `qwen3`); dense MLP offloaded via `layer_gc`, attention via attn-offload (`exp` is a no-op on dense — same as qwen3-32b). `TEMPLATE=llama3`. asym-offload **24.6 GiB**: `−44.9%` vs asym-recompute, `−47.1%` vs zero3_offload — the wider FFN makes the activation-offload win larger than Qwen3-32B. All 141 GB of base weights on CPU.

| Workload | Backend | Config | Status | fwd_s | bwd_s | opt_s | step_s | fwd_H | bwd_H | step_H | RAM |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s4096 b4 | asym_cpuadamwds (norecomp) | none+exp+attn-offload+layerGC [lg- sd+] | ok | 9.5 | 13.0 | 1.4 | 27.9 | 20.9 | 24.6 | 24.6 | 746.3 |
| s4096 b4 | asym_cpuadamwds (recomp) | none (no offload) [lg- sd-] | ok | 4.8 | 16.2 | 1.3 | 25.8 | 40.9 | 44.6 | 44.6 | 288.8 |
| s4096 b4 | zero3_offload (recomp) | none (no offload) [lg- sd-] | ok | 4.0 | 10.5 | 0.1 | 15.8 | 42.5 | 46.4 | 46.4 | 169.0 |

### qwen2.5-72b-instruct

Dense model with q/k/v **bias** (Qwen2 hardcodes it; handled by the standard LoRA + attn-offload paths — no extra code). 80 layers / FFN 29568. Same generic decoder-matcher branch; `TEMPLATE=qwen`. asym-offload **28.9 GiB**: `−40.9%` vs asym-recompute, `−43.4%` vs zero3_offload. 141 GB base on CPU.

| Workload | Backend | Config | Status | fwd_s | bwd_s | opt_s | step_s | fwd_H | bwd_H | step_H | RAM |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s4096 b4 | asym_cpuadamwds (norecomp) | none+exp+attn-offload+layerGC [lg- sd+] | ok | 10.6 | 14.1 | 1.4 | 30.1 | 24.5 | 28.9 | 28.9 | 762.3 |
| s4096 b4 | asym_cpuadamwds (recomp) | none (no offload) [lg- sd-] | ok | 6.7 | 17.4 | 1.3 | 29.2 | 44.5 | 48.9 | 48.9 | 296.5 |
| s4096 b4 | zero3_offload (recomp) | none (no offload) [lg- sd-] | ok | 4.1 | 10.7 | 0.1 | 16.3 | 46.5 | 51.1 | 51.1 | 173.6 |

<!-- ### qwen3_5-35b-a3b

| Workload | Backend | Config | Status | fwd_s | bwd_s | opt_s | step_s | fwd_H | bwd_H | step_H | RAM |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s4096 b4 | asym_cpuadamwds (norecomp) | none+exp+attn-offload+layerGC [lg- sd+] | not recorded | - | - | - | - | - | - | - | - |
| s4096 b4 | asym_cpuadamwds (recomp) | none (no offload) [lg- sd-] | not recorded | - | - | - | - | - | - | - | - |
| s4096 b4 | zero3_offload (recomp) | none (no offload) [lg- sd-] | ok | 2.7 | 10.6 | 0.0 | 14.6 | 43.3 | 50.9 | 50.9 | 192.0 |
| s4096 b4 | superoffload (recomp) | none (no offload) [lg- sd-] | ok | 2.8 | 11.1 | 0.0 | 15.1 | 43.3 | 50.9 | 50.9 | 171.8 | -->





┌───────────────────────┬──────────┬────────────────────────────┬─────────────────────────────────────────┬───────┬───────┬───────┬────────┬───────┬───────┬────────┬───────┐
│         Model         │ Workload │          Backend           │                 Config                  │ fwd_s │ bwd_s │ opt_s │ step_s │ fwd_H │ bwd_H │ step_H │  RAM  │
├───────────────────────┼──────────┼────────────────────────────┼─────────────────────────────────────────┼───────┼───────┼───────┼────────┼───────┼───────┼────────┼───────┤
│ llama-4-scout-17b-16e │ s4096·b8 │ asym_cpuadamwds (norecomp) │ none+exp+attn-offload+layerGC [lg+ sd+] │ 10.0  │ 36.8  │ 1.4   │ 48.1   │ 5.9   │ 6.4   │ 6.4    │ 901.4 │
├───────────────────────┼──────────┼────────────────────────────┼─────────────────────────────────────────┼───────┼───────┼───────┼────────┼───────┼───────┼────────┼───────┤
│ llama-4-scout-17b-16e │ s4096·b8 │ superoffload_mem (recomp)  │ none (no offload) [lg+ sd−]             │ 10.0  │ 30.4  │ 0.0   │ 42.5   │ 68.7  │ 95.3  │ 95.3   │ 525.4 │
├───────────────────────┼──────────┼────────────────────────────┼─────────────────────────────────────────┼───────┼───────┼───────┼────────┼───────┼───────┼────────┼───────┤
│ llama-4-scout-17b-16e │ s4096·b8 │ superoffload_mem (unsloth) │ none (no offload) [lg+ sd−]             │ 10.1  │ 30.5  │ 0.0   │ 42.6   │ 53.8  │ 80.4  │ 80.4   │ 525.4 │
├───────────────────────┼──────────┼────────────────────────────┼─────────────────────────────────────────┼───────┼───────┼───────┼────────┼───────┼───────┼────────┼───────┤
│ qwen3-30b-a3b         │ s4096·b8 │ asym_cpuadamwds (norecomp) │ none+exp+attn-offload+layerGC [lg+ sd+] │ 4.2   │ 6.6   │ 1.1   │ 14.1   │ 6.8   │ 8.9   │ 8.9    │ 424.4 │
├───────────────────────┼──────────┼────────────────────────────┼─────────────────────────────────────────┼───────┼───────┼───────┼────────┼───────┼───────┼────────┼───────┤
│ qwen3-30b-a3b         │ s4096·b8 │ superoffload_mem (recomp)  │ none (no offload) [lg+ sd−]             │ 2.3   │ 10.9  │ 0.0   │ 14.5   │ 14.4  │ 17.3  │ 17.3   │ 196.1 │
├───────────────────────┼──────────┼────────────────────────────┼─────────────────────────────────────────┼───────┼───────┼───────┼────────┼───────┼───────┼────────┼───────┤
│ qwen3-30b-a3b         │ s4096·b8 │ superoffload_mem (unsloth) │ none (no offload) [lg+ sd−]             │ 2.7   │ 11.4  │ 0.0   │ 14.0   │ 8.7   │ 11.5  │ 11.5   │ 196.7 │
└───────────────────────┴──────────┴────────────────────────────┴─────────────────────────────────────────┴───────┴───────┴───────┴────────┴───────┴───────┴────────┴───────┘


❯ the thing is there is ONLY one ting I wanna show is that if we can avoid materialxion
  weights/activations in the HBM we can save nontirvial amount og HBM. However, I am sturgleing to
  show this becase why the expeirmtsn kida show that if we materisalion in HBM and then to SMMEM
  seems to us ten same amoiuntof memory