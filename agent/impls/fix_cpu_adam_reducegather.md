# CPU AdamW x multi-GPU: reduce-then-offload (+ the reduce-scatter option)

Companion note to `fix_gb200_ep.md` S1 DELTA 3 (asym_ep2). Brief by design.

## The rule

Gradients are ALWAYS averaged ON GPU first; the CPU optimizer never sees unreduced
grads. Per step under ep2: backward (GPU grads) -> ONE coalesced NCCL allreduce, div 2
(collective — BOTH ranks end with the same mean; no "one GPU averages") -> clip ->
step-time D2H into pinned CPU buffers -> per-rank DeepSpeedCPUAdam on fp32 masters ->
H2D writeback of updated LoRA params. Adam is elementwise, so identical averaged grads
=> bitwise-identical masters on both ranks — no param sync ever.

## Precedent (verified in vendored third_party/deepspeed, ZeRO-Offload stage 1/2)

- Reduce first, on GPU: `reduce_ipg_grads` -> `average_tensor` /
  `gradient_reduction_w_predivide` (stage_1_and_2.py:1144).
- Only then offload: `copy_grads_in_partition` (:1512) ->
  `async_accumulate_grad_in_cpu_via_gpu` (:1378) into pinned CPU fp32 partition buffers
  (pinned at :808). `DeepSpeedCPUAdam` itself is comm-free.

## The refinement we deliberately skip (for now)

ZeRO stage-2 default (`reduce_scatter=True`, :154/:292): grads are reduce-SCATTERED so
each rank averages + downloads only its 1/N partition, CPU-steps only that slice, then
allgathers updated params. Halves D2H volume and CPU-Adam work per rank at the cost of
one param allgather.

At our scale (~1 GB LoRA grads, ms-class allreduce, ~1 s/step optimizer D2H) this is
noise -> ep2 keeps the simple full-allreduce + replicated CPU Adam (zero param comm).

STATUS: OPTION, not scheduled. Trigger = fix_gb200_ep.md STALL PLAYBOOK P6: adopt only
if a census ranks the optimizer D2H/step path in the top frames at an SG row.
