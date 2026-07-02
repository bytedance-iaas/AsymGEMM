AsymLoRA: A Superchip-based Offloading System for LoRA SFT

Background:
1. Full FT has optimizer states as significant memory/timing footprint but LoRA has litte optimizer footprints and a lot of activation memory instead, especially in long sequences.
2. Current offloading systems targets full finetuning and optimizes model and optimizer states offloading (zero3 offload). 
Recent work (superoffload) 
    a. improves offloading on superchips by improving optimizer scheduling (not targeting activation bottlenecks) 
    b. uses CPU for optimizerion (unsuitable for arm CPUs due to lower compute powers and underutilizes GPU compute)
There is a gap for superchip-based offloading system for LoRA SFT

Contribs:
1. We profile the SOTA offloading systems for LoRA SFT on GB200 systems and discover its current bottlenecks in activation memory and under utilization  
2. We adapt AsymGEMM for lora SFT to avoid materialization of activations and weights in HBM directly + delegate elemenwise computations on CPU => saves memory to enable longer sequnces / larger batch sizes
3. We develop a scheduler for offloading weights and activations using both CPU and NVME => saves memory to enable even longer sequnces / larger batch sizes
Current systems dont have activation + model weights + optimizer all offloading using CPU and NVME

Completed Milestones:
1. We adapted AsymGEMM for lora SFT and profiled to have lower memory usage
2. We integrated various dense and MoE models into the system

Notes:
- Superoffloads uses GH200 and we uses GB200 (2 GPUs 2x198G + 1 CPU 450G) => Model/EP/SP
- CPU accelerator kernels that might be used 4x128-bit 8 SVE2 to speed up CPU computations
- Next gen's chips e.g. VERA cpu-side accelerators => even larger bandwidths


llama-4-scout-17b-16e
Workload  Backend                      Config                                    fwd_s  bwd_s  opt_s  step_s  fwd_H  bwd_H  step_H    RAM
--------  ---------------------------  ----------------------------------------  ---------------------------  --------------------  -----
s4096·b4  asym_cpuadamwds (norecomp)   none+exp+attn-offload+layerGC  [lg- sd+]    6.5   58.4    5.8    70.7   19.7   19.3    19.7  802.
s4096·b4  zero3_offload (recomp)       none (no offload)  [lg- sd-]                8.3   18.1    1.6    28.0   39.0   52.1    52.1  525.9

Model: qwen3-30b-a3b    LoRA: r64/a16/d0.00
Workload    Backend                      Config                                    fwd_s  bwd_s  opt_s  step_s  fwd_H  bwd_H  step_H    RAM
--------  ---------------------------  ----------------------------------------  ---------------------------  --------------------  
s4096·b4    asym_cpuadamwds (norecomp)   none+exp+attn+layerOF  [lg- sd+]            2.7   34.6    3.7    41.1   23.8   28.3    28.3  343.5
s4096·b4    zero3_offload (recomp)       none (no offload)  [lg- sd-]                1.8    7.5    0.7    10.0   28.5   33.1    33.1  196.2

Setting:
- 1-2 GPUs (2x198G) + (1x450G)

Next Steps:
0. Seq Parallelism / Context Parallelism
1. [WIP] Implement and test various finegrained activation offloading strategies (to not stress CPU but still saves HBM). Then add NVME to accomodate even more offloading.
2. [WIP] Find the OOM scenarios for current systems (larger model, longer seq) that ours can accomodate
3. [TODO] Improve the system throughput with scheduling based on the required model size + sequence length


#############################################################################################################

Motivations:
- RuntimOptimizer is trivial in timing
    - Euperoffload does not target the truth bottleneck in LoRA SFT
- Memory decomposition of LoRA SFT
    - Even with recompute activation memory is core aspect
    - Other aspects taken care of by superoffload but activation not taken care
- Show C2C receiving underutilization of SuperOffload
    - Motivate more AsyGEMM to utilize RX
- Show CPU underutilization of SuperOffload
    - Motivate more computes on CPU
- Show compute vs memory bound for each activation tensor
    - Motivate to recompute on GPU / recompute on CPU / offload and fetch via AsymGEMM
- Show hardware specs difference between GH200 and GB200 in a table 
    - More scheduling needed to achieve better throughput + storage needed form nvme
- [?] Why does it have issues with extending to 2 GPUs (Superoffload + deepspeed / Superoffload + seq parallel)?

System Design:
- AsymGEMM-enabled design for lora forward/backward during recompute. Utilizing more CPU computes based on module/op heterogenuity
- Multi-tier activation storage system / NVME-based activation offload and prefetching
- Intgeration with SP/deepspeed for multiple superchips

Baselines:
- KTransformers
- FSDP
- FSDP Offload (Optimizer State + Model Params)
- Zero2
- Zero2 Offload (Optimizer State)
- Zero3
- Zero3 Offload (Optimizer State)
- Zero3 Offload (Optimizer State + Model Params)
- Superoffload (Optimizer State + Model Params)
<!-- - Megatron (Optimizer State) -->

Exps:
- 
-
-

Ablations:
- 
-
-




