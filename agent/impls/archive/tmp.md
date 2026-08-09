# TASK: name the two execution-mode columns of the paper's op table

## Context (self-contained)

MLSys paper (AsymLoRA): long-context LoRA fine-tuning on Grace–Blackwell
superchips (GPU HBM + CPU memory over a fast coherent NVLink-C2C link;
GPU kernels can read CPU memory directly). A design table lists every
training operator and shows, per op, its operand placements and kernel
under TWO executions of the same step:

- **Column A (works when the job fits GPU memory — the fastest execution):**
  activations kept in GPU memory; frozen weights live in CPU memory and are
  copied in per layer ("swapped"); everything runs cuBLAS/FlashAttention/CUDA.
- **Column B (used when the job does not fit — the leanest execution):**
  saved activations live in CPU memory; GEMMs consume them *in place* over
  the link (custom "Asym" streaming kernels, no copy); frozen weights are
  streamed the same way; non-GEMM ops (SDPA/norm/SiLU) cannot read CPU
  memory, so their saves are swapped back by prefetched copy, or their
  operands are rebuilt from smaller swapped saves.

The table (current headers = the placeholder to replace):

```
                    ┌────────────── Mode A ────────────────┬───────────────── Mode B ────────────────────┐
 Op                 │ Placement               │ Kernel     │ Placement                    │ Kernel        │
────────────────────┼─────────────────────────┼────────────┼──────────────────────────────┼───────────────┤
 Y = XW^T           │ X (GPU), W (swapped)    │ cuBLAS     │ X (GPU), W (CPU)             │ Asym          │
 S = XA^T           │ X (GPU), A (GPU)        │ cuBLAS     │ X (CPU), A (GPU)             │ Asym          │
 dA = dS^T·X        │ dS (GPU), X (GPU)       │ cuBLAS     │ dS (GPU), X (CPU)            │ Asym          │
 dX = dY·W          │ dY (GPU), W (swapped)   │ cuBLAS     │ dY (GPU), W (CPU)            │ Asym          │
 MoE expert GEMM    │ X (GPU), W (swapped)    │ cuBLAS     │ X (CPU), W (CPU)             │ Asym          │
 SDPA               │ q, k, v (GPU, kept)     │ FlashAttn  │ q, k, v (GPU, rebuilt)       │ FlashAttn     │
 Norm / RoPE        │ pre-norm q, k (GPU)     │ CUDA       │ pre-norm q, k (swapped)      │ CUDA          │
 SiLU·mul           │ gate, up (GPU)          │ CUDA       │ gate, up (swapped)           │ CUDA          │
 Grad ingestion     │ grads (CPU)             │ GraceArm   │ –                            │ –             │
 Optimizer update   │ state (CPU)             │ GraceArm   │ –                            │ –             │
```

Configurations (tiers) interpolate: deeper tiers execute more rows in
column B. Throughput is ALWAYS the objective — the system runs column A
whenever it fits (it is fastest) and shifts rows to column B only when
GPU capacity forces it (column B is then the fastest execution that fits).

## Naming attempts REJECTED so far, with the reason

1. "Resident vs Offloaded" — wrong: column A's weights are swapped in,
   not resident.
2. "When memory fits vs When memory is tight" — describes the situation,
   too wordy/vague as column headers.
3. "Save memory? No / Yes" — unclear.
4. "Swapping Mode vs Streaming Mode" — mechanism names fail: swaps occur
   in BOTH columns (weights swapped in A; norm/SiLU saves swapped back
   in B), and B also contains rebuilds. Contradictory on its face.
5. "Throughput Mode vs Memory Mode" — wrong intent split: throughput is
   always the target; B is not "preferring memory", it is the fastest
   feasible execution under a tighter budget.
6. "High-Memory vs Low-Memory Mode", "Full vs Lean" — judged flat/bad.
7. "In-Core vs Out-of-Core Mode" — judged bad naming by the author.
8. "HBM Mode vs DRAM Mode" — runner-up, not loved.

## The ask

Propose column-header PAIRS (2–4 candidates, ranked) that:
- name the actual difference: where the working set lives / how much GPU
  memory the execution requires — NOT the mechanism (swap/stream appear
  in both), NOT the intent (throughput is always the goal);
- are short (1–3 words per header), standard/intuitive for a systems
  (MLSys/OSDI) audience, no self-invented jargon;
- read correctly in the sentence: "the system runs <A> whenever the job
  fits and falls back to <B> when capacity forces it, which is then the
  fastest execution that fits."

Give each candidate a one-line defense and note any prior-art usage of
the terms.
