Compare 2 options for the activation-offload baseline (offload acts to CPU, stage back in backward, normal GEMM — no AsymGEMM).

1. **Bring-your-own offload via PyTorch `saved_tensors_hooks` (pack/unpack)**
   - Mechanism: register `torch.autograd.graph.saved_tensors_hooks`; pack hook copies each forward-saved tensor to CPU pinned RAM, unpack hook copies it back to HBM in backward (`save_on_cpu` is the built-in form).
   - Granularity: per saved tensor, with a size threshold to skip small ones. Model-agnostic — hooks at the autograd level, so one implementation covers Qwen3 / Qwen3.5 / Llama4 with no per-model code.
   - Same primitive torchtune (`OffloadActivations`) and TRL `SFTTrainer(activation_offloading=True)` use → this baseline IS the real torchtune/TRL approach, not a strawman.
   - vs AsymGEMM: stages the full BF16 activation CPU→HBM then runs a normal GEMM, paying the HBM cost (staged-tensor buffer + CPU→HBM→SMEM round-trip) that AsymGEMM eliminates. CPU footprint and interconnect bytes are identical to AsymGEMM, so the comparison isolates exactly the HBM staging cost.
   - Caveat: built-in `save_on_cpu` is synchronous (~6× slowdown); a fair baseline needs a second CUDA stream + pinned memory to overlap the copy with compute (as torchtune does).

2. **DeepSpeed's native activation offload (`cpu_checkpointing`)**
   - Mechanism: DeepSpeed JSON flag `cpu_checkpointing` (= `checkpoint_in_cpu` in `deepspeed.checkpointing.configure()`); offloads checkpointed activations to CPU in forward, reloads on recompute. Requires `partition_activations=True`.
   - Granularity: per activation-checkpoint boundary — offloads only the checkpoint *input* tensors, not every activation. Coupled to recompute.
   - Composes with ZeRO-3 (this is the ZeRO-Infinity combination: param/optimizer offload + activation-checkpoint CPU offload).
   - Limit: not a general "offload all activations" feature, so it cannot match the AsymGEMM offload-everything regime — only the checkpoint tensors reach CPU.
   - Config gotcha: JSON key is `cpu_checkpointing`, API arg is `checkpoint_in_cpu`; both require `partition_activations`.
