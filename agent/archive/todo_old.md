TODO
1. Profile  Y = X W^T + S B^T * scale # large output

Test 500,700, 10k for seq length
Test if we can do it dynamically based on ex0er reevief token lengths. Smal token sexpet can recompute


2. Profile 
```text
dX_gate = dGate @ W_gate
dX_up   = dUp   @ W_up
dX      = dX_gate + dX_up
```

Fuse this as:

```text
dX = [dGate, dUp] @ [W_gate; W_up]
```

3. Compare AsymGEMM and DeepGEMM

4. Find grained way to recompute activtions that are AsymGEMM friendly

5. understand the lorafusion and lora (handwrite the formula to understand the process)


Instructions
1. Currently we can recompute activations on and off and vary different seq lengths. However, I wanna see for a given seq length, we can aslso add a var called expert length therosld, for those below that, we can recomptue them. Duing traing each experir willl recevei different # of tokens. This means that for the experts who receive fewer than k tokens, we can jsutrecompute for them. so this is finegrained. Implement this for me. And make sure /home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM/scripts/profile_lora_driver.sh this can aslo profile for this by adding an extra var. The plotting results should include the per expert threshold as well. And it shod take a list of exper threolsds (also overridinable ) so that the plotting shod also plt ehen the seq legth is fixed, if we increas the threhso (x axis) and y is sitl memory usage, how wil that plot beliek. the other oplot is what it currenelr does (ignore the expert throelds / we always recomute), this plot still needs to eb on their owns, wihc is eactl what we have now.


1. Keep current code only as a correctness/profiling prototype.
   It proves expert-threshold recompute works, but mixed thresholds are too expensive.
2. Add/use the new expert token stats.
   Rerun profiling and check when thresholds actually move experts into recompute vs keep. This tells us whether thresholds like 128/256/512/... are
   meaningful for Qwen routing.
3. Replace PyTorch subset-split checkpointing.
   Current bad path:
   keep subset + recompute subset + two grouped runs + index_copy_.

   Better path:
   one grouped expert autograd path that keeps/recomputes per expert internally.
4. Implement AsymGEMM-aware grouped autograd.
   Save activations only for experts with tokens >= threshold; for experts below threshold, save metadata/input refs and recompute inside backward.
5. Reprofile memory and time.
   Expected: memory should drop with higher threshold; timing should increase more smoothly, without huge mixed-path spikes.
6. Only consider custom fused kernels after step 4.
   The first fix is graph/autograd structure, not necessarily kernel fusion.






Questions
0. Why backward soo slow? Is it becasue of the transposed w reading breaks the cache layout. Test X @ W vs X @ W.T for accuracy and performance.
Test how AsumGEMM is applied to individual GEMM (how does it use goruped GEMM?)
0.1 Test different MoEs?
0.2 how much activation is % of weights?
0.3 How does normal lora sft take care of W.T? When do they tranpose?
0.4 Fuse Y = X W^T + alpha S B^T + bias   + dX_base = dY W


0.  BF16 for AsyGEMM ok?

1. Anything we can borrow from LoraFusion?
Backward:
   Fuse:
   dB = alpha * dY^T S       # [N, r]
   dS = alpha * dY B         # [M, r]

   dA = dS^T Xdrop           # [r, K]

   Fuse:
   dX_base = dY W            # [M, K]
   dX_lora = dropout_backward(dS A)
   dX      = dX_base + dX_lora

-  Current code does:

  base = AsymGEMM(...)
  lora = LoRA(...)
  return base + lora

  So a real contribution would be:

  AsymGEMM epilogue accepts S and B
  computes base accumulator from streamed W
  computes/adds LoRA tile before final store
  never materializes base-only Y or lora-only Y


-  Fuse recompute with backward so streamed W is reused?
  backward:
    dact = dY W_down

    recompute gate = X W_gate^T
    recompute up   = X W_up^T

    dgate, dup = activation backward

    dX_gate = dgate W_gate
    dX_up   = dup W_up

  That streams W_gate/W_up once for recompute and again for dX.

  A meaningful fused kernel would try:

  for expert / tile:
    stream W_gate/W_up tile once
    recompute gate/up tile
    compute dgate/dup tile
    immediately accumulate partial dX using same W tile


- Any separate streamining operations?
   - Prefetch next expert's W while current expert computes

   - while W tiles are being prefetched:
      run small GPU-resident LoRA GEMMs

   when W tile is ready:
      run streamed base GEMM

   in epilogue:
      combine base accumulator + LoRA accumulator

4. Some current ideas:
- Recompute activation memory but store up/gate 





