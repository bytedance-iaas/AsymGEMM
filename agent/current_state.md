# Current State - M0 Baseline

Date: 2026-05-18

Hardware scope:

- Target machine: NVIDIA H200, compute capability `sm_90`.
- CPU-GPU path: normal host/PCIe/NUMA path, not Grace Hopper or GB200 NVLink-C2C.
- Initial training scope: BF16, single GPU, frozen base/expert weights, LoRA SFT.

Existing AsymGEMM repo capability:

- Public forward BF16 AsymGEMM bindings are present:
  - `m_grouped_bf16_asym_gemm_nt_contiguous`
  - `m_grouped_bf16_asym_gemm_nt_masked`
- Public forward FP8 grouped bindings are present.
- Existing tests cover BF16/FP8 forward grouped kernels and host-pinned weight inputs.
- The JIT emits architecture-specific cubins for the current device.

Observed binding/export reality:

- `setup.py` builds `asym_gemm._C` only when CUDA extension tooling and `CUDA_HOME` are available.
- `asym_gemm/__init__.py` imports `_C` opportunistically; without `_C`, import warns and CUDA kernels are unavailable.
- `csrc/python_api.cpp` registers GEMM, layout, and runtime APIs only.
- Runtime bindings: `set_num_sms`, `get_num_sms`, `set_tc_util`, `get_tc_util`, `set_compile_mode`, `get_compile_mode`, `init`.
- Layout bindings include `transform_sf_into_required_layout`, `get_tma_aligned_size`, `get_mk_alignment_for_contiguous_layout`, and MN-major TMA layout helpers.
- GEMM bindings include BF16 m-grouped kernels; with FP8/TensorMap support they also include FP8, FP4, and k-grouped forward kernels.
- `csrc/apis/einsum.hpp` defines `einsum` and `fp8_einsum`, and generated stubs may mention them, but `csrc/python_api.cpp` does not currently register them.
- Top-level `asym_gemm` mirrors selected `_C` bindings. Some `_C` layout helpers are extension-only, not top-level exports.
- Legacy top-level aliases exist for masked grouped FP8/BF16 names.

Architecture gates and dispatch:

- `DG_TENSORMAP_COMPATIBLE` requires CUDA Driver API `>= 12.1`.
- `DG_FP8_COMPATIBLE` requires PyTorch `>= 2.1`.
- BF16 m-grouped dispatch routes arch major `9` to SM90 kernels and arch major `10` to SM100 kernels.
- FP8 m-grouped dispatch routes arch major `9` to SM90 kernels with FP32 output and arch major `10` to SM100 kernels with BF16 output.
- `fp8_gemm_nt` and `k_grouped_fp8_gemm_nt_contiguous` currently assert the SM90 implementation path.
- FP4 m-grouped wrappers are bound under the FP8/TensorMap gate but call SM100 implementations.

Missing at the M0 checkpoint, before M1-M2 work:

- No PyTorch training/autograd package.
- No frozen-weight linear primitive.
- No direct-fetch backward `dX = dY @ W_base`.
- No runnable LoRA training demo.
- No M0-M2 reports that separate direct AsymGEMM execution from staged/Torch fallback.
- No public dgrad or wgrad pybind API.
- No implemented LoRA frozen-base contract; base-weight `dW = None` is still a roadmap requirement.

Current shared-tree note: later M1/M2 files now exist under `asym_gemm/training/`,
`tests/training/test_01_frozen_linear.py`, `tests/training/test_02_mlp_demo.py`,
`examples/asymgemm/mlp_lora_demo.py`, and `reports/mlp_demo.json`. The M0 baseline
above records what was missing before those milestones. The native `_C` extension
still exposes forward GEMM/layout/runtime APIs only; there is still no public native
dgrad or wgrad pybind API.

M0-M2 implementation direction:

- Keep frozen base weights as CPU `HostWeight` objects, not `nn.Parameter`s and not CUDA buffers.
- Use one-group BF16 m-grouped AsymGEMM as the dense direct-fetch primitive when input, weight, and shape constraints match H200.
- Keep explicit fallback modes:
  - `asym_only`
  - `asym_or_staged`
  - `asym_or_torch`
  - `torch_only`
- Tests must report numerical correctness and the HBM difference between CPU-resident weights and GPU-resident baseline weights.

Current stale assumptions:

- `README.md` still describes NVIDIA Superchips, NVLink-C2C, and SM100/CUDA 12.9 as primary requirements.
- Some existing tests call CUDA capability helpers during import and are not safe on CUDA-unavailable hosts.
- `agent/tests.md` still reflects the historical H20 FP8 smoke path and is stale relative to the current BF16 H20/Hopper test files.

M0 smoke command:

```bash
python -m pytest -q -s tests/training/test_00_m0_smoke.py
```

The smoke imports the package, checks observed `_C`/top-level export behavior,
reports CUDA H200/SM90 status when CUDA exists, and runs one BF16 m-grouped
contiguous forward kernel when H200/SM90 hardware is present.
