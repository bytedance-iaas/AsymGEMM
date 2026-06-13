# ARM KT MoE SFT Port: Short Plan

This is the short version of `arm_moe_sft.md`.

## Goal

Port KTransformers MoE LoRA-SFT to this ARM/aarch64 machine in a way that works
first, then gets faster later.

KT SFT replaces each HuggingFace MoE layer with a KT wrapper:

- router and non-expert model parts stay on GPU;
- routed expert base weights move to CPU and stay frozen;
- PEFT still creates LoRA parameters;
- KT rewires expert LoRA weights and gradients into KT-managed CPU buffers;
- trainer hooks must clear, reattach, sync, and save those KT-managed buffers.

## Backend Names

Keep the existing LLaMA-Factory switch:

```yaml
use_kt: true
```

Select the implementation with `kt_backend`. Do not add `use_kt_arm` or
`kt-arm`.

Current x86 KT SFT backends:

```text
AMXBF16 -> AMXBF16_SFT
AMXINT8 -> AMXINT8_SFT
AMXINT4 -> AMXINT4_SFT
```

ARM backends to add:

```text
TORCHBF16 -> TORCHBF16_SFT   # first correctness backend
ARMBF16   -> ARMBF16_SFT     # later native ARM kernel
ARMINT8   -> ARMINT8_SFT     # optional later INT SFT
ARMINT4   -> ARMINT4_SFT     # optional later INT SFT
```

On ARM, explicit `AMX*` requests must fail with a clear error.

## First Deliverable

The first target is single-process/single-GPU:

```yaml
use_kt: true
kt_backend: TORCHBF16
```

`TORCHBF16_SFT` should use portable PyTorch/BF16 CPU expert computation. It is
the correctness oracle. It does not need to be AMX-fast.

FSDP/DDP is last-priority follow-up work. Keep helper APIs compatible with later
FSDP/DDP, but do not block the first ARM port on it.

## LLaMA-Factory Integration

Do not depend on patched `transformers-kt` / `accelerate-kt` for the first ARM
path. Those packages are integration glue, not the kernel hot path.

Instead, add local LLaMA-Factory glue:

```text
third_party/LlamaFactory/src/llamafactory/model/kt_adapter.py
```

Required helpers:

```text
build_lf_kt_config
wrap_lf_kt_model
adapt_lf_kt_peft_lora
clear_and_reattach_lf_kt_lora_grads
sync_lf_kt_lora_gradients
mark_lf_kt_lora_pointers_dirty
save_lf_kt_adapter_extras
```

Load/train order:

```text
load base model CPU-first
-> wrap MoE layers with KT
-> apply PEFT LoRA
-> adapt KT expert LoRA buffers
-> train
-> save PEFT adapter plus KT extras
```

Phase 2a parser validation should require one process and one GPU. FSDP/DDP
should be rejected with a phase-guard error, not treated as impossible forever.

## Proof Order

1. Import `kt_kernel` on aarch64 without x86 fallback.
2. Implement `TORCHBF16_SFT`.
3. Run tiny forward/backward parity tests.
4. Run one-process LLaMA-Factory SFT for 2 to 5 steps.
5. Prove KT LoRA grad buffers are nonzero, cleared/reattached across steps,
   saved, reloaded, and used after reload.
6. Add `ARMBF16_SFT` native ARM kernels only after `TORCHBF16_SFT` is correct.
7. Add `ARMINT8_SFT` / `ARMINT4_SFT` only later if needed.
8. Add FSDP/DDP last, only if multi-GPU training becomes important.

## Key Point

The first ARM SFT port is a training-semantics proof, not a peak-performance
kernel project. Once single-GPU `TORCHBF16_SFT` is correct, it becomes the oracle
for native ARM BF16 and later INT SFT backends.
