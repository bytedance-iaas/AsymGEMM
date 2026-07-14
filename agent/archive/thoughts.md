Codex
• Yes. Do it in three layers.

  1. Replace Count Threshold With AsymGEMM Cost Policy
  Current code does:

  recompute_groups = active & (counts < threshold)

  in third_party/AsymGEMM/asym_gemm/training/moe.py:1393.

  That is not AsymGEMM-aware. Replace it with a per-group score:

  saved_hbm_bytes(g)
  /
  extra_asym_recompute_cost(g)

  For expert group g:

  tokens = t
  m_tiles = ceil(t / BLOCK_M)

  saved_hbm_bytes =
      t * intermediate_size * dtype_bytes * saved_tensor_count

  For current save policy, saved_tensor_count = 3 because it saves gate, up, and activated.

  Extra recompute cost is not proportional to tokens. It is tile-rounded:

  extra_remote_weight_bytes =
      m_tiles * bytes(W_gate + W_up)

  So for AsymGEMM:

  t = 1      -> reread almost full gate/up expert weights, save tiny activations
  t = 120    -> reread same tile-rounded weights, save 120x more activations

  This means the current “recompute small experts” idea may be wrong for AsymGEMM. A better policy recomputes groups with good tile utilization:

  score(g) =
      saved_hbm_bytes(g)
      /
      (remote_weight_bytes(g) / c2c_bandwidth + flops(g) / tensorcore_rate + launch_cost)

  Then choose groups by score until you hit a target HBM saving.

  That is the first research contribution: tile- and remote-weight-aware MoE recomputation, not token-count thresholding.

  2. Exploit Frozen Base + LoRA Split
  This is the most important implementation angle.

  Because base expert weights are frozen and CPU-resident, we should prepack gate/up base weights into one host layout:

  W_gate_up_cpu[e] = concat(W_gate[e], W_up[e], dim=0)

  shape: [num_experts, 2 * intermediate_size, hidden_size]

  Then forward gate/up becomes one AsymGEMM:

  gate_up = x @ W_gate_up_cpu.T
  gate, up = split(gate_up)

  Instead of two calls here:

  third_party/AsymGEMM/asym_gemm/training/moe.py:1238

  Backward dX also becomes one AsymGEMM:

  grad_gate_up = concat(grad_gate, grad_up, dim=-1)
  grad_x_base = grad_gate_up @ W_gate_up_cpu

  Instead of two calls here:

  third_party/AsymGEMM/asym_gemm/training/moe.py:1022

  This is a real AsymGEMM-specific trick. Normal GPU MoE does not care as much because weights are already in HBM. For AsymGEMM, fusing gate/up reduces launches, metadata
  traversal, A reads, and remote host-weight scheduling overhead. Since the base is frozen, the packed CPU layout never needs optimizer updates.

  LoRA stays separate:

  base: frozen, CPU-resident, AsymGEMM
  LoRA: trainable, GPU-resident, small

  So the backward policy should save/recompute base activations while freely recomputing cheap LoRA low-rank intermediates.

  3. Use Multi-Tier Activation Saving
  Do not only choose save-all vs recompute-all. Use three modes:

  Mode A: save gate, up, activated
      fastest, most HBM

  Mode B: save gate, up only
      recompute activated = silu(gate) * up in backward
      good default

  Mode C: save nothing for gate/up/activated
      recompute gate/up through AsymGEMM in backward
      saves most HBM, costs extra CPU-resident weight traffic

  Mode B is likely very strong for LoRA-SFT because activated is mostly needed for down-LoRA gradient, not frozen base gradient. The frozen down base only needs:

  grad_activated = grad_out @ W_down

  So saving activated is optional.

  4. Move Recompute Out Of Python Loop
  Current recompute happens group-by-group in Python:

  third_party/AsymGEMM/asym_gemm/training/moe.py:973

  That is fine for correctness, but not the research kernel.

  Replace it with:

      W_gate_up_cpu,
      W_down_cpu,
      offsets,
      experts,
      recompute_plan,
      saved_gate_up,
      saved_activated_optional,
      lora_weights
  )

  Internally:

  1. grad_activated_base = grouped AsymGEMM(grad_out, W_down, transpose_b=True)

  2. for recompute groups:
         gate_up = grouped AsymGEMM(packed_subset, W_gate_up_cpu)

  3. for saved groups:
         load saved gate/up

  4. fused SwiGLU backward:
         grad_gate, grad_up

  5. grad_x_base = grouped AsymGEMM(
         concat(grad_gate, grad_up),
         W_gate_up_cpu,
         transpose_b=True
     )

  6. add LoRA gradients / LoRA dX

  Step 5 is a key kernel contribution: combined gate/up dX through one AsymGEMM using frozen packed host weights.

  Implementation Order

  1. Add cost-model policy replacing counts < threshold.
  2. Add packed W_gate_up_cpu HostWeight layout.
  3. Change forward to use one grouped AsymGEMM for gate/up base.
  4. Change backward to use one grouped AsymGEMM for gate/up base dX.
  5. Add tiered save policy: save gate/up, optionally activated, or recompute all.
  6. Finally move recompute-group handling from Python loop into a native grouped backward op.

  The strongest paper claim is:

  > Existing MoE recompute treats recomputation as extra GPU compute. In AsymGEMM SFT, recomputation rereads frozen expert weights from CPU memory, so the correct policy is tile-
  > rounded and fabric-aware. We exploit frozen CPU-resident gate/up packing to fuse forward recompute and backward dX, making activation recomputation aware of AsymGEMM’s
  > remote-weight execution model.


################################################################################################################################################################

Contributions
1. Recompute activations in AsymGEMM-friendly ways instead of thresholding on seq length / expert token length 
-   Cost model for recomputing expert-level activations
    Normal MoE recompute asks:
        How many activation bytes can I save?
    AsymGEMM recompute must ask:
        How many activation bytes do I save,
        and how much extra CPU-resident weight traffic do I create?
    - But will this really generate concrete benefis comparing to pure token length based thresholding? - Probably not
-   Split lora activation and base weight activation recomputations
    LoRA has much smaller traffic but base weights on CPU has much higher traffic
    - But only checkpointing the LoRA branch is not effective casue it onlt saves [tokens, r] which is small? - Marginal
-   Compute activations for different layers
    Tier 1: save gate, up, activated
        fastest, most memory

    Tier 2: save gate and up only
        recompute activated = silu(gate) * up
        good middle ground

    Tier 3: save nothing for that expert group
        recompute gate/up/activated in backward
        lowest memory, most extra AsymGEMM traffic
    Tier 2 may be very good because recomputing activated is cheap GPU math, while recomputing gate/up requires rereading CPU-resident weights.
    - How much activtion can it even saved tho? Will it reallty be effective?


2. Fuse forward + backward AsymGEMM LoRA kernels
- Fuse the recomputation inside AsmGEMM
- Refer to LoRA fusion's ideas. Anything that we can adapt from agent/related_work/lorafusion.md to apply to our AsymGEMM LoRA SFT case? We cant redo the same things from lorafusion but neeed to do it in ways specific to AsymGEMM

3. Improve AsymGEMM MoE kernels
- Fuse up/gate forward
    - But this is alreyad done by all MoE kernels. Is tehre some special implementations needed for better fusing AsymGEMM MoE kernels? I dont think so? Otherwises this is pure engineering
- Backward can do the same trick:
    grad_gate_up = concat(grad_gate, grad_up)
    grad_x = AsymGEMM(grad_gate_up, W_gate_up, transpose_b=True)
    - Same question is this alreayd solved by normal MoE kernels? Any AsymGEMM specific angles? Cant be pure engineering.



########################################################################################################################################################################

Feedback
2. S = X @ A_lora.T                  # small [M, r]
    Y = AsymGEMM(X, W_cpu) + S @ B.T  # fused inside one output tile
    Can we fuse all of them together for ASynGEMM becase the bottl enck si only AsymGEMM(X, W_cpu). Can we stream in paralle for eample to compute this big kernel so that fusing all might even be better for AsymGEMM?

    Option C: Compute S once per M-tile and share across N-tiles
    This is the most researchy but hardest. A cluster/persistent kernel could compute:
    S_tile = X_tile A.T
    once, keep it in shared memory / cluster memory, then multiple N tile CTAs use it while streaming W_cpu.
    That would be genuinely AsymGEMM-specific:
    use GPU-resident LoRA math to hide remote CPU-weight latency
    But it is much harder than Option A: Precompute S, then fuse output (lorafusion)

3. 
  Backward kernel target:
  dX = AsymGEMM(dY, W_cpu, transpose_b=True) + dX_lora
  bt

5.   The research claim should not be “we fuse gate/up.” The claim should be:

  we prepack frozen CPU-resident gate/up experts to support recompute-aware backward
  and avoid two remote-weight AsymGEMM launches
  Hm but tbis s literal stil the samething but just a differn verbal phrasing wiht no impoementation wise diffence??

  "It becomes meaningful when combined with remote-weight-aware recompute and mixed base+LoRA fusion" - How

