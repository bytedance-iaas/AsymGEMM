Research
- Activation memory usage heavy => Recompute based on fine-grained expert-level heuristics. 
- Recompute forces extra weight streaming => Stream selected gate/up W panel once, use it for recompute, cache it in HBM, and reuse it for dX. Reduce selected W stream 2.0x -> 1.0x.

- [x] Small SMEM and Long CPU-GPU transfer wait time => Joint CPU and GPU finetuning of MoEs (not just GPUs / CPUs alone). Finetune experts with few tokens on CPU and large ones on GPU. Need the same int8 layout.

Questions
- [x] AsymGEMM on optimizer states offloading? 
- Is there any fusion opporutnity for our selective expert fusion in forward or backward?

Engineering
- [Minor|?] Saving full gate/up/activated is expensive but full recompute streams too much W => Save-gate/recompute-up baseline. Save gate only and recompute only up. Reduce selected W stream 2.0x -> 1.5x.
- Fuse gate/up LoRA-A compute without storing persistent fused weights => Avoid repeated gate/up LoRA-A materialization (less memory). Keep separate PEFT-compatible gate/up adapters while removing transient concat/copy overhead.
- Prefetch the next N tile in L2 Cache => Long CPU-GPU transfer wait time starving SMEM from doing math
- Tune the block sizes => Tensor sizes affect AsymGEMM speed. Improve kernel memory and compute utilization.
(large M => fewer writes per tile, larger N => more parallel calls / more weight streamining, larger K => more contiguous weight streaming / more weight streaming per tile)
- Fuse gate/up GEMM => Reduce intermediates (reference: DeepGEMM)
- Fuse lora forward ops and lora backward ops separately => Reduce intermediates and host weight streaming (reference: LoraFusion)
- Own the entire MoE block (router+experts not just experts) => Run router in torhc.nograd(). Save autograd memory from the frozen router (reference: KT)
- Host weight quantization => Reduce weight traffic (reference: QLoRA)

Next steps
- Activation can be offloaded to CPU and staged to HBM/L2
- Profile recompute fused intop backward

Questions
- Varying batch size length's impact on training?
- Even longer datasets?
- Need to compare with Megatron?
- int8 SFT?








####################################################################

### Notations 
@ = GEMM
@^ = AsymGEMM A @ B.T where B is on CPU
@^^ = (Hypothetical) AsymGEMM A @ B.T where A is on CPU
* = Elementwise Multiplication
CPU = Compute and keep on CPU
HBM = Persistent in HBM until offload / last usage
Temp means this result is in HBM but can be released right after the next reuse/ next 2 reuse right after. The reuse lines are immediately right after.
Grad is the gradient
Tensors with _cpu are on CPUs. Those without _cpu are on HBM.
offload is offloading from HBM to CPU
stage is staging from CPU to HBM
Lines that start with # are comments

###  Forward
```
X = routed_rows_for_this_expert                         # [M, H] HBM

gate_up_base = X @^ W_gate_up_cpu.T                     # [M,2I] Temp
gate_base, up_base = split(gate_up_base)                # [M,I], [M,I] Temp
S_gate = X @ A_gate.T                                   # [M,r] HBM
S_up   = X @ A_up.T                                     # [M,r] HBM

X_cpu      = offload(X)

LoRA_gate = scale * (S_gate @ B_gate.T)                 # [M,I] Temp
gate = gate_base + LoRA_gate                            # [M,I] HBM
LoRA_up   = scale * (S_up   @ B_up.T)                   # [M,I] Temp
up   = up_base   + LoRA_up                              # [M,I] HBM

gate_cpu   = offload(gate)                              # Can later fuse as   gate_cpu = offload(gate_base + LoRA_gate)
up_cpu     = offload(up)                                # Can later fuse as   up_cpu   = offload(up_base   + LoRA_up)
S_gate_cpu = offload(S_gate)                            # save for dB_gate
S_up_cpu   = offload(S_up)                              # save for dB_up

sig_cpu = sigmoid(gate_cpu)                           # [M, I] CPU
silu_gate_cpu = sig_cpu * gate_cpu                    # [M, I] CPU
act_cpu       = silu_gate_cpu * up_cpu                # [M, I] CPU

S_down    = act_cpu @^^ A_down.T                      # [M, r] HBM
LoRA_down = scale * (S_down @ B_down.T)               # [M, H] Temp
S_down_cpu = offload(S_down)

act = stage(act_cpu)
Y_down = act @^ W_down_cpu.T + LoRA_down               # [M, H] HBM
```

### Backward
```
dY = dL/dY_down                                      # [M, H] HBM

# ---------------- down backward ----------------



dS_down   = scale * (dY @ B_down)                    # [M, r] Temp
dact_lora = dS_down @ A_down                         # [M, I] Temp
dact_base = dY @^ W_down_cpu                         # [M, I] Temp Can later fuse dact_cpu = offload(dY @^ W_down_cpu + dact_lora)
dact = dact_base + dact_lora                         # [M, I] Temp

dact_cpu = offload(dact)

# S_down = act_cpu @^^ A_down.T                          # [M, r] Recomp/Reuse S_down_cpu

dA_down = dS_down.T @^ act_cpu                        # [r, I] Grad
dB_down = scale * (dY.T @^ S_down_cpu)                    # [H, r] Grad


# ---------------- activation backward ----------------

# sig = sigmoid(gate_cpu)                              # [M, I] Recomp/Reuse sig_cpu
# silu_gate = gate_tmp * sig                           # [M, I] Recomp/Resue silu_gate_cpu
silu_grad_cpu = sig_cpu * (1 + gate_cpu * (1 - sig_cpu))  # [M, I] CPU

dgate_cpu = dact_cpu * up_cpu * silu_grad_cpu           # [M, I] CPU 
dup_cpu   = dact_cpu * silu_gate_cpu                    # [M, I] CPU

# ---------------- gate/up base backward ----------------

dgate = stage(dgate_cpu)                              # [M, I] HBM
dup   = stage(dup_cpu)                                # [M, I] HBM

dgate_up = concat(dgate, dup)                         # [M, 2I] HBM, gate first then up
dX = dgate_up @^ W_gate_up_cpu                        # [M, H] HBM


# ---------------- gate LoRA backward ----------------

# S_gate = X_cpu @^^ A_gate.T                         # [M, r] Recomp / Reuse S_gate_cpu. Not needed.

dS_gate      = scale * (dgate @ B_gate)               # [M, r] Temp
dX_gate_lora = dS_gate @ A_gate                       # [M, H] Temp
dX += dX_gate_lora

dA_gate = dS_gate.T @^ X_cpu                          # [r, H] Grad
dB_gate = scale * (dgate.T @^ S_gate_cpu)             # [I, r] Grad


# ---------------- up LoRA backward ----------------

# S_up = X_cpu @^^ A_up.T                             # [M, r] Recomp / Reuse S_up_cpu. Not needed.

dS_up      = scale * (dup @ B_up)                     # [M, r] Temp
dX_up_lora = dS_up @ A_up                             # [M, H] Temp
dX += dX_up_lora

dA_up = dS_up.T @^ X_cpu                              # [r, H] Grad
dB_up = scale * (dup.T @^ S_up_cpu)                   # [I, r] Grad


# ---------------- final input gradient ----------------
# dX = dX_gate_base + dX_gate_lora + dX_up_base + dX_up_lora  # [M, H] HBM should have been accumualted along the way
```

Notes:
- Did LoRAFusion make S_up and S_down persistent? across forward and backward? Are we currently doing that?

####################################################################

### Contributions
- We develop

- We test it comprehensively 


### Questions
1. How to fix CPU OOM with partial recompute / NVME offloading?
2. Is there more kernels for the backward process? 
- Compute silu and directly write to HBM not storing on CPU
- Compute grad and directly write to CPU not through HBM
3. What are the gains and bottlenecks for timing and memory currently across Qwen3 and Llama4?
4. fused CE with AsymGEMM?

### Discussion
0. Current numbers
┌──────────────────────┬───────────────────┬─────────────────┬────────┬────────────────┬───────────────┬──────────────┬───────────┬─────────────┐
│        Model         │  Workload (tok)   │     Backend     │ Recomp │ Peak alloc HBM │ Peak resv HBM │ Peak RAM/RSS │ Mean step │ Median step │
├──────────────────────┼───────────────────┼─────────────────┼────────┼────────────────┼───────────────┼──────────────┼───────────┼─────────────┤
│ qwen3-30B-A3B        │ b8_s4096 (32 768) │ asym_cpuadamwds │   no   │          56.78 │         67.06 │       483.04 │   75.75 s │     75.67 s │
├──────────────────────┼───────────────────┼─────────────────┼────────┼────────────────┼───────────────┼──────────────┼───────────┼─────────────┤
│ qwen3-30B-A3B        │ b8_s4096 (32 768) │ zero3_offload   │  yes   │          64.15 │         74.26 │       195.54 │    7.82 s │      7.61 s │
├──────────────────────┼───────────────────┼─────────────────┼────────┼────────────────┼───────────────┼──────────────┼───────────┼─────────────┤
│ qwen3-30B-A3B        │ b8_s8192 (65 536) │ asym_cpuadamwds │   no   │         113.21 │        132.92 │       760.53 │  151.99 s │    151.01 s │
├──────────────────────┼───────────────────┼─────────────────┼────────┼────────────────┼───────────────┼──────────────┼───────────┼─────────────┤
│ qwen3-30B-A3B        │ b8_s8192 (65 536) │ zero3_offload   │  yes   │         126.17 │        146.20 │       196.53 │   12.30 s │     12.30 s │
├──────────────────────┼───────────────────┼─────────────────┼────────┼────────────────┼───────────────┼──────────────┼───────────┼─────────────┤
│ llama4-scout-17B-16E │ b4_s4096 (16 384) │ asym_cpuadamwds │   no   │          27.83 │         29.03 │       857.45 │  68.13 s¹ │    68.13 s¹ │
├──────────────────────┼───────────────────┼─────────────────┼────────┼────────────────┼───────────────┼──────────────┼───────────┼─────────────┤
│ llama4-scout-17B-16E │ b4_s4096 (16 384) │ zero3_offload   │  yes   │          49.53 │         55.93 │       525.83 │  39.33 s¹ │    39.33 s¹ │
└──────────────────────┴───────────────────┴─────────────────┴────────┴────────────────┴───────────────┴──────────────┴───────────┴─────────────┘

Comments:
1. Comine Suoerlaod with ours to reduce the latency (meomry and latency tradeoff)
- For qwen, we need to do more swapping
3. More plots on tradeoffs
<!-- 4. Can we modify fused CE so that we write the iutputs diretl into CPU 
for vocab tile b:
    Z_b = H @ W_b.T        # [M,Bv], temporary tile only
    update online max/sum   # [M] stats
    discard Z_b -->


1. Can we build a cuda graph for kernel launch (partially)?
- Normally can for attention, exp router but we add in offloading and staging which requires CPU sync
- Cannot for experts because the each receives a diff token count (M) so the cpu needs to read that count to launch the kernel

2. Scheduling between recomputing/offloading/caching
Motivation:
- Memory: caching > recomputing > offloading
- Latency: offloading > recomputing > caching
Method:
- Split by layers / required storage/ required compute
- Early layers offload. Late layers recompute. Middle layers in HBM. 
- Save smaller tensors and recompute larger tensors => save RAM/HBM

3. Where to store tensors? 
- Model params / optimizer states / activations in nvme / cpu
- Activation tensors in CPU instead of nvme, and more weight tensors in nvme

4. What to use for compute? cpu or GPU
- Elementwise (silu(x) = x * sigmoid(x), optimizer updates) use CPU
- GEMMs (AsymGEMM, Native GEMM) use GPU

5. Can we fully use back and forward beween CPU <-> GPU whihc uses separte bandwith not like smem <-> hbm which shares the bandwith betwen read and write
- Compute silu on CPU and directly write to HBM => save RAM

6. Ask agents to optimize bf16 sm100 AsymGEMM

7. What is the motivation?
- We identify a memory-bottleneck shift in LoRA SFT for large MoEs: adapter weights, gradients, and optimizer states occupy only [X] GiB ([X]% of peak HBM), while frozen model weights and activations reach [X] GiB and [X] GiB ([X]% of peak HBM). The main challenge reducing frozen weights and activations' memory footprints.

- We develop SuperLoRA, a Superchip-native LoRA SFT system that keeps frozen MoE weights and most activations in CPU memory, using NVLink-C2C’s
    [Z] GB/s bandwidth to treat host DRAM as an active extension of HBM. 

- We redesign forward and backward execution with C2C-streaming GEMM kernels that load CPU-resident weights and activations tile-wise from CPU directly into GPU shared memory, eliminating HBM materialization. Meanwhile, we delegate lightweight elementwise ops to the CPU, including SiLU, SiLU backward, LoRA grad accomulation, and adapter updates, reserving GPU resources for high-arithmetic-intensity GEMM operations.

- We introduce a specialized scheduler that chooses, per layer and tensor, whether to keep activations in HBM, offload them, or
recompute them. It exploits the structure of LoRA SFT: frozen weights are read-only, trainable
adapters are small, expert usage is sparse, and backward needs CPU-resident weights and activations only at precise GEMM boundaries. 
This enables better memory-latency tradeoffs for fitting larger MoE SFT workloads on a single GB200.

Given a config => deterin an optial poitn in the search space

Schedule needs to cover these cases:
storage {nvme, cpu, gpu} x compute {cpu, gpu}



