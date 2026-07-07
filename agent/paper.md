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
3. We root-caused and optimized the MoE fine-grained recompute-offload path (operand
   placement + pinned-pool reuse + compact-X GPU dA): q3-30b-a3b s80000.b8 step time
   -70% (1043s -> 315s) at equal-or-lower peak HBM, closing the latency gap to
   superoffload_mem|unsloth-off to 1.15x while keeping the HBM advantage

Notes:
- Superoffloads uses GH200 and we uses GB200 (2 GPUs 2x198G + 1 CPU 450G) => Model/EP/SP
- CPU accelerator kernels that might be used 4x128-bit 8 SVE2 to speed up CPU computations
- Next gen's chips e.g. VERA cpu-side accelerators => even larger bandwidths

Setting:
- 1-2 GPUs (2x198G) + (1x450G)

Next Steps:
0. Seq Parallelism / Context Parallelism
1. [WIP] Implement and test various finegrained activation offloading strategies (to not stress CPU but still saves HBM). Then add NVME to accomodate even more offloading.
2. [WIP] Find the OOM scenarios for current systems (larger model, longer seq) that ours can accomodate
3. [TODO] Improve the system throughput with scheduling based on the required model size + sequence length





#############################################################################################################
Results:
Model: q3-30b-a3b  (asym rows = optimized fg path 2026-07-02, see agent/impls/fix_qwen3.md)

Workload   Backend           Config                     Status  fwd_s  bwd_s   opt_s  step_s  fwd_H  bwd_H  step_H    RAM
---------  ----------------  -------------------------  ------  -----  ------  -----  ------  -----  -----  ------  -----
s80000.b8  superoffload_mem  unsloth                    PASS     29.6   130.3    0.0   160.0   91.9  176.9   176.9  360.0
s80000.b8  superoffload_mem  unsloth-off                PASS     33.2   240.7    0.0   274.0   91.9   94.4    94.4  588.5
s80000.b8  asym_cpuadamwds   recomp-off-full-fg-ker000  PASS     62.9   252.1    3.3   315.1   86.0  105.6   105.6  557.4
s80000.b8  asym_cpuadamwds   recomp-off-full-fg-ker101  PASS     61.6   261.0    3.3   322.8   78.6   73.9    73.9  557.3

Model: q3-32b

Workload   Backend           Config                    Status  fwd_s  bwd_s   opt_s  step_s  fwd_H  bwd_H  step_H    RAM
---------  ----------------  ------------------------  ------  -----  ------  -----  ------  -----  -----  ------  -----
s50000.b8  superoffload_mem  unsloth                   PASS     57.0   219.2    0.1   276.4   91.6  180.9   180.9  340.4
s50000.b8  superoffload_mem  unsloth-off               PASS     56.7   368.3    0.1   425.2   91.6  110.7   110.7  644.4
s50000.b8  asym_cpuadamwds   recomp-off-full-fg-ker000  PASS    101.0   561.6    2.5   662.9   91.7   96.4    96.4  657.7

Model: q2.5-32b

Workload   Backend           Config                    Status  fwd_s  bwd_s   opt_s  step_s  fwd_H  bwd_H  step_H    RAM
---------  ----------------  ------------------------  ------  -----  ------  -----  ------  -----  -----  ------  -----
s50000.b8  superoffload_mem  unsloth                   PASS     49.2   114.7    0.1   164.1   97.7  171.6   171.6  382.7
s50000.b8  superoffload_mem  unsloth-off               PASS     48.5   307.5    0.1   356.2   97.7  118.4   118.4  633.0
s50000.b8  asym_cpuadamwds   recomp-off-full-fg-ker000  PASS     89.2   520.2    1.9   609.7   97.8   81.2    81.2  617.0

Model: q2.5-72b

Workload   Backend           Config                    Status  fwd_s  bwd_s   opt_s  step_s  fwd_H  bwd_H  step_H    RAM
---------  ----------------  ------------------------  ------  -----  ------  -----  ------  -----  -----  ------  -----
s30000.b8  superoffload_mem  unsloth                   PASS     64.3   108.6    0.1   173.1   67.6  125.0   125.0  492.2
s30000.b8  superoffload_mem  unsloth-off               PASS     64.1   308.1    0.1   372.5   67.6   80.8    80.8  662.7
s30000.b8  asym_cpuadamwds   recomp-off-full-fg-ker000  PASS    105.4   498.7    2.5   604.4   67.6   62.5    62.5  733.6

Model: llama3.3-70b

Workload   Backend           Config                     Status  fwd_s  bwd_s   opt_s  step_s  fwd_H  bwd_H  step_H    RAM
---------  ----------------  -------------------------  ------  -----  ------  -----  ------  -----  -----  ------  -----
s25000.b8  superoffload_mem  unsloth                    PASS     34.5    87.0    0.1   122.7   57.0  102.2   102.2  489.6
s25000.b8  superoffload_mem  unsloth-off                PASS     64.2   215.8    0.1   280.2   55.0   65.7    65.7  657.6
s25000.b8  asym_cpuadamwds   recomp-off-full-fg-ker000  PASS     77.7   960.4    3.2  1038.4   55.0   52.1    52.1  730.4

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
- AsymLoRA kernels: Enable efficient ops with cpu-resifent weights and activation tensors
    - MoE routing kernels
        fused routed GEMM epilogues that scatter-add directly into token-space accumulators
    - LoRA offload kernels
        CPU-left LoRA-A forward and CPU-right LoRA-A gradient kernels that operate directly on pinned offloaded activations

- AsymGEMM-guided activation offloading and scheduling: Decompose LoRA MLP/MoE forward and backward into fine-grained operators so large activations are recomputed, offloaded, or consumed directly from pinned CPU memory.
    - Operator-aware activation policy
        split gate/up/activation/down/LoRA paths and retain only the tensors each later operator truly needs.
    - C2C-aware execution and GEMM
        stream offloaded activations through NVLink-C2C into AsymGEMM/LoRA kernels and briefly stage tensors only at point of use.
    - Heterogeneous compute
        move memory-bound elementwise and low-rank work to CPU/offload-aware kernels, while keeping compute-dense GEMMs on GPU.

- [?]New hardware arichiecture/new module/how to do GB200 diffeent than GH200? Hardware-aware/GB200-aware module?
- Ownerless EP on 2 GPUs for balancing MoE models' expert computes
    - Debugging.....
- Intgeration with TP on 2 GPUs for dense models
    - Extend with TP for longer sequences
- Multi-tier activation offload system using CPU and NVME
    - Enable spilling into NVME using FIFO asynchronously and prefetching based on LILO as efficient additional storage for activations.

Baselines:
- KTransformers
- FSDP2
- FSDP2 Offload (Optimizer State + Model Params)
- Zero2
- Zero2 Offload (Optimizer State)
- Zero3
- Zero3 Offload (Optimizer State)
- Zero3 Offload (Optimizer State + Model Params)
- Superoffload (Optimizer State + Model Params)
- *Superoffload (Optimizer State + Model Params + Act)
<!-- - Megatron (Optimizer State) -->

Exps:
GB200
- 1 GPU, Dense / MoEs
    - Throughput vs seq length where each legend is a method
    - Memory saving vs seq length where each legend is a method 
    - C2C Utilization where each legend is a method 
    - GPU Utilization where each legend is a method 
- 2 GPUs, Dense / MoEs
    - The same as above
GB200 + NVME
- 1 GPU, Dense / MoEs
- 2 GPUs, Dense / MoEs
<!-- GH200
- 1 GPU, Dense / MoEs
    - Throughput vs seq length where each legend is a method
    - Memory saving vs seq length where each legend is a method 
    - C2C Utilization where each legend is a method 
    - GPU Utilization where each legend is a method 
- 2 GPUs, Dense / MoEs
    - The same as above
GH200 + NVME
-  -->

Ablations
- 
- 

