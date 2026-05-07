import os
import subprocess
import torch
from torch.version import cuda as cuda_version
from packaging import version

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# CUDA extension and kernels (only available when built with CUDA)
try:
    from . import _C
    from ._C import (
        set_num_sms,
        get_num_sms,
        set_tc_util,
        get_tc_util,
        set_compile_mode,
        get_compile_mode,
    )

    if version.parse(cuda_version) >= version.parse('12.1'):
        def _maybe_import_from_C(names):
            for name in names:
                if hasattr(_C, name):
                    globals()[name] = getattr(_C, name)

        def _missing_kernel(kernel_name):
            def _raise_missing(*args, **kwargs):
                raise RuntimeError(
                    f"`{kernel_name}` is not available in this build of asym_gemm. "
                    "Rebuild with matching CUDA/architecture flags to enable this kernel."
                )
            return _raise_missing

        def _export_kernel_alias(alias_name, target_name):
            globals()[alias_name] = globals().get(target_name, _missing_kernel(target_name))

        # DeepGEMM Kernels (may vary by build flags / arch)
        _maybe_import_from_C([
            # FP8 GEMMs
            "m_grouped_fp8_asym_gemm_nt_masked",
            "m_grouped_fp8_asym_gemm_nt_contiguous",
            # FP8 mega GEMMs (Phase 1+2: double-buffered B, dedicated warp roles)
            "m_grouped_fp8_asym_gemm_mega_nt_contiguous",
            "m_grouped_fp8_asym_gemm_mega_nt_masked",
            # FP4 GEMMs
            "m_grouped_fp4_asym_gemm_nt_contiguous",
            "m_grouped_fp4_asym_gemm_nt_masked",
            # BF16 GEMMs
            "m_grouped_bf16_asym_gemm_nt_contiguous",
            "m_grouped_bf16_asym_gemm_nt_masked",
            # Einsum kernels
            "einsum",
            "fp8_einsum",
            # Attention kernels
            "fp8_mqa_logits",
            "get_paged_mqa_logits_metadata",
            "fp8_paged_mqa_logits",
            # Layout kernels
            "transform_sf_into_required_layout",
            "get_mk_alignment_for_contiguous_layout",
        ])

        # Some alias for legacy supports
        # TODO: remove these later
        _export_kernel_alias("fp8_m_grouped_asym_gemm_nt_masked", "m_grouped_fp8_asym_gemm_nt_masked")
        _export_kernel_alias("fp8_m_grouped_gemm_nt_masked", "m_grouped_fp8_asym_gemm_nt_masked")
        _export_kernel_alias("bf16_m_grouped_asym_gemm_nt_masked", "m_grouped_bf16_asym_gemm_nt_masked")
        _export_kernel_alias("bf16_m_grouped_gemm_nt_masked", "m_grouped_bf16_asym_gemm_nt_masked")

    # Initialize CPP modules
    def _find_cuda_home() -> str:
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        if cuda_home is None:
            try:
                with open(os.devnull, 'w') as devnull:
                    nvcc = subprocess.check_output(['which', 'nvcc'], stderr=devnull).decode().rstrip('\r\n')
                    cuda_home = os.path.dirname(os.path.dirname(nvcc))
            except Exception:
                cuda_home = '/usr/local/cuda'
                if not os.path.exists(cuda_home):
                    cuda_home = None
        assert cuda_home is not None
        return cuda_home

    _C.init(
        os.path.dirname(os.path.abspath(__file__)),
        _find_cuda_home()
    )
except ImportError:
    import warnings
    warnings.warn("CUDA extension (_C) not available. CUDA kernels will not be accessible.")

from importlib.metadata import version as _get_version
__version__ = _get_version('asym_gemm')


# =============================================================================
# fp8_asym_mega_moe_nt_contiguous — Single fused CUDA kernel
# =============================================================================
# Mirrors DeepGEMM's `deep_gemm.fp8_fp4_mega_moe`: ONE kernel call does the
# entire L1 → SwiGLU → FP8 re-quant → L2 → combine pipeline.  The kernel is
# implemented in `asym_gemm/include/asym_gemm/impls/sm100_fp8_asym_mega_moe.cuh`
# and launched by `_C.fp8_asym_mega_moe_nt_contiguous_fused`.
#
# First implementation uses CUDA-core BF16 math (not UMMA) so the code is
# auditable in one PR; a UMMA upgrade is tracked in `asym_moe_update.md`.
def fp8_asym_mega_moe_nt_contiguous(
    a,                     # (FP8 packed, FP32 SF) dispatched activations [M_total, H]
    l1_w,                  # (FP8 packed, FP32 SF) L1 weights [E, 2I, H]
    l2_w,                  # (FP8 packed, FP32 SF) L2 weights [E, H, I]
    y,                     # BF16 [num_tokens, H] — OUTPUT (pre-allocated)
    offsets,               # int32 [2*list_size] — sparse (start, end) pairs per active expert
    experts,               # int32 [E+1] — expert ids per active slot, trailing -1 sentinels
    list_size,             # int — number of active experts
    topk_map,              # int32 [M_total, 2] — (orig_token_idx, topk_k) per dispatched row
    row_topk_w,            # float32 [M_total]  — combine weight per dispatched row
    num_tokens,            # int — original token count (output's first dim)
    intermediate_hidden,   # int
    recipe=(1, 128, 128),
    activation_clamp=0.0,
    fast_math=True,
    disable_ue8m0_cast=False,
    m_indices=None,
):
    """Fused 2-layer MoE (L1 → SwiGLU → re-quant → L2 → combine) in ONE launch."""
    import torch

    a_fp8, a_sf = a
    l1_p, l1_sf = l1_w
    l2_p, l2_wsf = l2_w

    M_total     = a_fp8.shape[0]
    hidden      = a_fp8.shape[1]
    num_experts = l1_p.shape[0]
    num_topk    = int(topk_map[:, 1].max().item()) + 1 if topk_map.numel() > 0 else 1
    device      = a_fp8.device

    # Expand sparse (start, end) offsets into a full [2*num_experts] array.
    # offsets[2*i], offsets[2*i+1] = start/end rows for the i-th active expert;
    # experts[i] = actual expert id.  Inactive experts get start==end==0.
    offsets_full = torch.zeros(2 * num_experts, dtype=torch.int32, device=device)
    if list_size > 1:
        off_cpu = offsets.detach().to('cpu')
        exp_cpu = experts.detach().to('cpu')
        for i in range(list_size - 1):
            eid = int(exp_cpu[i].item())
            s   = int(off_cpu[2 * i].item())
            e   = int(off_cpu[2 * i + 1].item())
            if 0 <= eid < num_experts:
                offsets_full[2 * eid]     = s
                offsets_full[2 * eid + 1] = e

    # Normalize SFs to float32.
    if a_sf.dtype != torch.float32:
        a_sf = a_sf.float()
    if l1_sf.dtype != torch.float32:
        l1_sf = l1_sf.float()
    if l2_wsf.dtype != torch.float32:
        l2_wsf = l2_wsf.float()
    if row_topk_w.dtype != torch.float32:
        row_topk_w = row_topk_w.float()

    # Compute workspace byte offsets (mirrors MegaMoEWorkspace::compute_sizes).
    def _align16(x):
        return (x + 15) & ~15

    block_m         = 128
    num_pool_blocks = (M_total + block_m - 1) // block_m
    _off = 0
    off_grid_sync     = _off; _off = _align16(_off + 4 * 4)
    off_l1_arrival    = _off; _off = _align16(_off + num_pool_blocks * 4)
    off_l2_mask       = _off; _off = _align16(_off + num_pool_blocks * 8)
    off_l2_acts       = _off; _off = _align16(_off + M_total * intermediate_hidden)
    off_l2_sf         = _off; _off = _align16(_off + M_total * (intermediate_hidden // 128) * 4)
    off_token_src_map = _off; _off = _align16(_off + M_total * 2 * 4)
    off_l1_topk_w     = _off; _off = _align16(_off + M_total * 4)
    off_combine       = _off; _off = _align16(_off + num_topk * num_tokens * hidden * 2)
    workspace_bytes   = _off

    workspace = torch.zeros(max(workspace_bytes, 1), dtype=torch.uint8, device=device)

    _C.fp8_asym_mega_moe_nt_contiguous(
        a_fp8=a_fp8.contiguous(),
        a_sf=a_sf.contiguous(),
        l1_w=l1_p.contiguous(),
        l1_w_sf=l1_sf.contiguous(),
        l2_w=l2_p.contiguous(),
        l2_w_sf=l2_wsf.contiguous(),
        offsets=offsets_full.contiguous(),
        topk_map=topk_map.contiguous().to(torch.int32),
        row_topk_w=row_topk_w.contiguous(),
        workspace=workspace,
        y=y,
        off_grid_sync=off_grid_sync,
        off_l1_arrival=off_l1_arrival,
        off_l2_mask=off_l2_mask,
        off_l2_acts=off_l2_acts,
        off_l2_sf=off_l2_sf,
        off_token_src_map=off_token_src_map,
        off_l1_topk_w=off_l1_topk_w,
        off_combine=off_combine,
        M_total=int(M_total),
        num_tokens=int(num_tokens),
        num_topk=int(num_topk),
        hidden=int(hidden),
        intermediate=int(intermediate_hidden),
        num_experts=int(num_experts),
        activation_clamp=float(activation_clamp),
        fast_math=bool(fast_math),
    )


__all__ = [name for name in globals() if not name.startswith('_')]
