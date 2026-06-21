• Verdict: CODA is useful as a design pattern, not as a drop-in codebase. I would not port CODA wholesale into this repo. The strongest applicable ideas are:

  1. Fuse LoRA-B into the base GEMM epilogue/finalization path.
     Your generic LoRA path currently does base_layer(x), then lora_B(lora_A(x)), then adds them: asym_gemm/training/lora.py:311. In Qwen3 activation offload, the
     same pattern appears as separate gate_delta/up_delta/down_delta tensors before add_: asym_gemm/training/qwen3_moe.py:1003, asym_gemm/training/
     qwen3_moe.py:1077. A CODA-inspired epilogue could load S = x @ A_lora and B_lora, compute the small rank-r dot, and add it to the frozen base output tile.
     This would remove full-size LoRA delta HBM buffers and extra add/read/write traffic.

  2. Build an AsymGEMM epilogue/finalizer interface, but restrict it carefully.
     CODA’s epilogue visitor model is the right abstraction shape: tile-local transforms, side outputs, reductions, and auxiliary matrix/vector loads. But
     AsymGEMM’s SM100 base kernel stores partial K-tile results with TMA store/reduce-add: asym_gemm/include/asym_gemm/impls/sm100_bf16_asym_gemm.cuh:984. That
     means the complete output tile is generally not resident on chip. Linear transforms can be distributed or added once on the last K tile; nonlinear transforms
     like SwiGLU, RMSNorm stats, and cross-entropy cannot be naively attached.

  3. Use CODA’s side-output idea for direct CPU activation offload.
     Your current manager copies CUDA tensors to pinned CPU and later stages them back: asym_gemm/training/activation_offload.py:187, asym_gemm/training/
     activation_offload.py:237. The Qwen3 path offloads X, gate, up, and low-rank tensors, then stages selected ones in backward: asym_gemm/training/
     qwen3_moe.py:950, asym_gemm/training/qwen3_moe.py:1020, asym_gemm/training/qwen3_moe.py:1168. A CODA-like epilogue could side-store backward-only
     intermediates directly from SMEM/registers to CPU-pinned memory, avoiding temporary HBM residency.

  4. Consider a logits/cross-entropy finalizer later.
     CODA’s partial cross-entropy epilogue is very relevant to SFT because full logits can dominate HBM. For AsymGEMM this likely needs a special final-K path that
     reads prior partial sums, forms final logits, emits CE stats, and avoids materializing full logits. That is promising but more invasive than LoRA-B fusion.

  What I would not prioritize:

  - Directly importing CODA kernels. The public repo targets Hopper/H100, CuTeDSL Python custom ops, HBM-resident GEMMs, and standard transformer blocks. Your
    system is C++/NVRTC AsymGEMM, CPU-pinned operands, grouped expert routing, and activation offload.

  - CODA RMSNorm/SwiGLU/RoPE fusion on the core AsymGEMM path. These require complete output tiles or nonlinear functions of complete outputs. Your K-outer
    schedule intentionally sacrifices that to reuse CPU-resident weight tiles.

  - Treating CODA as a latency fix. CODA reduces GPU HBM round trips around GEMMs. Your main axis is HBM capacity, and the hard bottlenecks are CPU↔GPU movement
    and staged activation lifetimes.

  Best next step: prototype a narrow fused LoRA-B epilogue/finalizer for Qwen3 gate/up/down expert projections. It is linear, aligned with the current activation-
  offload design, and should reduce HBM temporaries without fighting the AsymGEMM schedule.

  Sources: CODA paper (https://arxiv.org/pdf/2605.19269), CODA codebase (https://github.com/open-lm-engine/coda-kernels).

  