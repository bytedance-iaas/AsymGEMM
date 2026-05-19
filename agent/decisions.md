# Decisions

## M0 Decisions

### D-0001: H200/SM90 Is The Initial Target

M0-M2 target H200 / SM90 only. Do not rely on GH200, GB200, SM100, or
NVLink-C2C coherent-link assumptions for the initial training path.

### D-0002: M0 Records Reality Before Training Work

M0 records actual imports, bindings, arch gates, missing training APIs, and
smoke status. It does not add autograd modules or training wrappers.

### D-0003: Smoke Tests Reflect The Installed Build

The M0 smoke checks the real `asym_gemm._C` bindings and selected top-level
mirrors. Generated stubs are not treated as authoritative because they can list
unregistered APIs such as `einsum`.

### D-0004: Hardware Assertions Skip Without CUDA

CUDA-only checks skip when CUDA is unavailable. On CUDA/H200 runs, the smoke
reports the device and verifies H200/SM90 support before executing a hardware kernel.

### D-0005: BF16 M-Grouped Contiguous Is The M0 Kernel Smoke

The hardware smoke uses one small BF16 m-grouped contiguous forward call because
BF16 is the M1-M2 dtype and the wrapper has explicit SM90 dispatch.

## M0-M2 Baseline

- Scope is H200/SM90 BF16 only.
- CPU-resident base weights are represented by `asym_gemm.training.HostWeight`.
- M1 uses the existing one-group `m_grouped_bf16_asym_gemm_nt_contiguous` binding as the first dense direct-fetch path.
- Forward computes `Y = X @ W.T`.
- Backward input gradient computes `dX = dY @ W` using a CPU-resident transposed copy of `W`.
- Base-weight gradients are intentionally omitted for LoRA SFT; frozen base weights have `grad is None`.
- Fallbacks are explicit and counted. A result that falls back to staged/Torch is not reported as a direct-fetch win.
- M2 uses a small deterministic MLP LoRA demo before dense LLM or MoE integration.
- HBM savings are reported as actual CUDA allocation avoided by keeping base weights out of GPU memory, plus runtime peak HBM for the demo step.
