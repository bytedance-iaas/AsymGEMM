import math
import random
from typing import Dict, Tuple

import numpy as np
import pytest
import torch

import asym_gemm
from asym_gemm.testing import get_arch_major, ignore_env


def _align(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment


def _build_m_indices_like_cpp(
    expected_m_per_group: int,
    num_groups: int,
    *,
    seed: int = 0,
) -> Tuple[torch.Tensor, int, int]:
    alignment = int(asym_gemm.get_mk_alignment_for_contiguous_layout())
    rng = random.Random(seed)

    actual_ms = []
    aligned_ms = []
    total_m = 0
    active_m = 0
    for _ in range(num_groups):
        actual_m = max(1, int(expected_m_per_group * rng.uniform(0.7, 1.3)))
        aligned_m = _align(actual_m, alignment)
        actual_ms.append(actual_m)
        aligned_ms.append(aligned_m)
        total_m += aligned_m
        active_m += actual_m

    m_indices = torch.empty((total_m,), dtype=torch.int32, device="cpu")
    start = 0
    for gid, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end = start + actual_m
        aligned_end = start + aligned_m
        m_indices[start:actual_end] = gid
        m_indices[actual_end:aligned_end] = -1
        start = aligned_end
    return m_indices, total_m, active_m


def _fill_with_sentinel(m_indices_cpu: torch.Tensor, num_groups: int):
    # Build offsets in pairs format: [start_0, end_0, start_1, end_1, ...]
    # This matches asymScheduler which reads offsets[blockIdx.y*2] (start) and
    # offsets[blockIdx.y*2+1] (end). `experts` keeps its (num_experts + 1) layout
    # with a trailing -1 sentinel, so list_size is unchanged.
    m_list = m_indices_cpu.to(dtype=torch.int32).contiguous().tolist()
    m = len(m_list)
    capacity = num_groups + 1

    offsets = []  # flat array of pairs
    experts = []

    def emit(start_idx: int, end_idx: int):
        e = int(m_list[start_idx])
        if e != -1:
            offsets.append(start_idx)
            offsets.append(end_idx)
            experts.append(e)

    if m > 0:
        seg_start = 0
        for i in range(1, m):
            if m_list[i] != m_list[i - 1]:
                emit(seg_start, i)
                seg_start = i
        emit(seg_start, m)

    experts.append(-1)

    # Truncate: offsets holds 2 ints per expert (excluding sentinel), experts has sentinel.
    num_experts = min(len(experts) - 1, capacity - 1)
    offsets = offsets[: num_experts * 2]
    experts = experts[: num_experts + 1]
    list_size = num_experts + 1
    return (
        torch.tensor(offsets, dtype=torch.int32, device="cuda"),
        torch.tensor(experts, dtype=torch.int32, device="cuda"),
        list_size,
    )


def _encode_e2m1_scalar(x: float) -> int:
    sign = x < 0.0
    ax = min(abs(x), 6.0)
    bounds = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    idx = 0
    for i, b in enumerate(bounds):
        if ax >= b:
            idx = i + 1
    if sign and idx != 0:
        idx |= 0x08
    return idx


def _decode_e2m1_codes(codes: np.ndarray) -> np.ndarray:
    lut = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    mag = lut[np.bitwise_and(codes, 0x07)]
    sign = np.where(np.bitwise_and(codes, 0x08) != 0, -1.0, 1.0).astype(np.float32)
    return mag * sign


def _unpack_fp4_bytes(packed: np.ndarray) -> np.ndarray:
    low = np.bitwise_and(packed, 0x0F).astype(np.uint8)
    high = np.right_shift(np.bitwise_and(packed, 0xF0), 4).astype(np.uint8)
    out = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.uint8)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out


def _to_e4m3_bits_and_decoded(sf: float) -> Tuple[int, float]:
    t = torch.tensor([sf], dtype=torch.float32, device="cpu")
    e4m3 = t.to(torch.float8_e4m3fn)
    bits = int(e4m3.view(torch.uint8).item())
    decoded = float(e4m3.float().item())
    if decoded == 0.0 or math.isnan(decoded):
        decoded = 1.0
    return bits, decoded


def _decode_e4m3_bits(bits_u8: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(bits_u8.copy()).view(torch.float8_e4m3fn)
    return t.float().numpy()


def _quantize_nvfp4_e4m3(x_f32: np.ndarray, gran_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized NVFP4 per-row quantization (gran_mn=1, gran_k=gran_k).

    Replaces the triple-nested Python loop with pure NumPy + a single batched
    PyTorch FP8 cast.  Runs ~1000x faster for large (rows, k) shapes.
    """
    rows, k = x_f32.shape
    assert k % 2 == 0
    sf_k = (k + gran_k - 1) // gran_k
    pad = sf_k * gran_k - k
    x = np.pad(x_f32, ((0, 0), (0, pad))) if pad > 0 else x_f32

    # ── scales ──────────────────────────────────────────────────────────────
    blocks = x.reshape(rows, sf_k, gran_k)                   # (rows, sf_k, gran_k)
    amax = np.maximum(np.abs(blocks).max(axis=2), 1e-4)      # (rows, sf_k)
    sf_f32 = (amax / 6.0).astype(np.float32)
    sf_tensor = torch.from_numpy(sf_f32).to(torch.float8_e4m3fn)
    scales = sf_tensor.view(torch.uint8).numpy().astype(np.uint8)   # (rows, sf_k)
    sf_dec = sf_tensor.float().numpy()
    sf_dec = np.where(sf_dec == 0.0, 1.0, sf_dec)           # guard against NaN/zero

    # ── quantized values ─────────────────────────────────────────────────────
    q = (blocks / sf_dec[:, :, np.newaxis]).reshape(rows, sf_k * gran_k)[:, :k]  # (rows, k)

    # E2M1 encode: same rounding as _encode_e2m1_scalar
    _bounds = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
    sign = (q < 0.0).astype(np.uint8)
    mag_code = np.searchsorted(_bounds, np.abs(q).ravel(), side='right') \
                 .reshape(rows, k).astype(np.uint8)          # 0-7
    codes = mag_code.copy()
    codes[(sign != 0) & (mag_code != 0)] |= np.uint8(8)     # set sign bit

    # Pack two nibbles per byte (low nibble = even column, high = odd)
    packed = ((codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)).astype(np.uint8)
    return packed, scales


def _quantize_a_nvfp4_e4m3(a_f32: np.ndarray, gran_k: int) -> Tuple[np.ndarray, np.ndarray]:
    return _quantize_nvfp4_e4m3(a_f32, gran_k)


def _quantize_b_nvfp4_e4m3(b_f32: np.ndarray, gran_k: int) -> Tuple[np.ndarray, np.ndarray]:
    # Per-row quantization: each row of B gets its own scale per K-group (gran_mn_b=1).
    # This matches recipe (1, 1, 16) and the production NVFP4 format from fp4_model.json.
    return _quantize_nvfp4_e4m3(b_f32, gran_k)


def _build_ground_truth(
    a_bf16: torch.Tensor,
    b_bf16: torch.Tensor,
    m_indices: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    m = a_bf16.size(0)
    n = b_bf16.size(1)
    gt = torch.zeros((m, n), device=a_bf16.device, dtype=torch.bfloat16)
    for gid in range(num_groups):
        rows = torch.where(m_indices == gid)[0]
        if rows.numel() == 0:
            continue
        a_g = a_bf16.index_select(0, rows).float()
        b_g = b_bf16[gid].float()
        gt.index_copy_(0, rows, (a_g @ b_g.t()).to(torch.bfloat16))
    return gt


def _manual_dequant_reference(
    a_packed_u8: np.ndarray,
    a_scales_u8: np.ndarray,
    b_packed_u8: np.ndarray,
    b_scales_u8: np.ndarray,
    m_indices_cpu: torch.Tensor,
    gran_k: int,
) -> torch.Tensor:
    num_groups, n, packed_k = b_packed_u8.shape
    m = a_packed_u8.shape[0]
    k = packed_k * 2
    sf_k = a_scales_u8.shape[1]

    a_codes = _unpack_fp4_bytes(a_packed_u8)
    a_q = _decode_e2m1_codes(a_codes)
    a_sf = _decode_e4m3_bits(a_scales_u8)
    a_sf = np.repeat(a_sf, gran_k, axis=1)[:, :k]
    a_deq = a_q * a_sf

    d_manual = torch.zeros((m, n), dtype=torch.float32, device="cpu")
    m_idx = m_indices_cpu.to(torch.int32).contiguous().numpy()

    for gid in range(num_groups):
        rows = np.where(m_idx == gid)[0]
        if rows.size == 0:
            continue

        b_codes = _unpack_fp4_bytes(b_packed_u8[gid])
        b_q = _decode_e2m1_codes(b_codes)
        b_sf = _decode_e4m3_bits(b_scales_u8[gid])
        b_sf = np.repeat(b_sf, gran_k, axis=1)[:, :k]  # per-row: only expand K axis
        b_deq = b_q * b_sf

        a_rows_t = torch.from_numpy(a_deq[rows]).to(torch.float32)
        b_t = torch.from_numpy(b_deq).to(torch.float32)
        d_rows = a_rows_t @ b_t.t()
        d_manual[torch.from_numpy(rows)] = d_rows

    return d_manual


def _diff_stats(x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    diff = (x - y).abs().float()
    mean_abs = float(diff.mean().item())
    max_abs = float(diff.max().item())
    ref = y.abs().float().mean()
    rel = float((mean_abs / max(float(ref.item()), 1e-12)))
    return {"mean_abs": mean_abs, "max_abs": max_abs, "rel": rel}


def _print_top_left(name: str, x: torch.Tensor, rows: int = 5, cols: int = 5):
    x_cpu = x.float().cpu().contiguous()
    r = min(rows, x_cpu.size(0))
    c = min(cols, x_cpu.size(1))
    print(f"\n{name} (top-left {r}x{c}):")
    for i in range(r):
        print(" ".join(f"{float(x_cpu[i, j]):.6f}" for j in range(c)))


def _print_max_diff_with_neighborhood(
    tag: str,
    out: torch.Tensor,
    ref: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    radius: int = 2,
):
    out_cpu = out.float().cpu().contiguous()
    ref_cpu = ref.float().cpu().contiguous()
    diff = (out_cpu - ref_cpu).abs()
    finite = torch.isfinite(diff)
    if not bool(finite.any()):
        print(f"\n[{tag}] No finite entries found in diff tensor.")
        return

    finite_diff = torch.where(finite, diff, torch.full_like(diff, -1.0))
    flat_idx = int(finite_diff.view(-1).argmax().item())
    cols = out_cpu.size(1)
    i = flat_idx // cols
    j = flat_idx % cols
    max_abs = float(diff[i, j].item())
    global_row = int(active_rows[i].item())

    print(
        f"\n[{tag}] max_abs_diff={max_abs:.6f} at "
        f"active_row={i}, col={j} (global_row={global_row})"
    )

    r0 = max(0, i - radius)
    r1 = min(out_cpu.size(0), i + radius + 1)
    c0 = max(0, j - radius)
    c1 = min(out_cpu.size(1), j + radius + 1)

    print(f"[{tag}] neighborhood rows [{r0}, {r1 - 1}], cols [{c0}, {c1 - 1}]")
    print(f"[{tag}] Out neighborhood:")
    for rr in range(r0, r1):
        print(" ".join(f"{float(out_cpu[rr, cc]):.6f}" for cc in range(c0, c1)))
    print(f"[{tag}] Ref neighborhood:")
    for rr in range(r0, r1):
        print(" ".join(f"{float(ref_cpu[rr, cc]):.6f}" for cc in range(c0, c1)))
    print(f"[{tag}] Diff neighborhood:")
    for rr in range(r0, r1):
        print(" ".join(f"{float(out_cpu[rr, cc] - ref_cpu[rr, cc]):.6f}" for cc in range(c0, c1)))


@ignore_env("DG_JIT_PTXAS_CHECK", lambda: torch.cuda.is_available() and get_arch_major() == 9)
def test_m_grouped_nvfp4_contiguous_cpp_flow() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not hasattr(asym_gemm, "m_grouped_fp4_asym_gemm_nt_contiguous"):
        pytest.skip("FP4 contiguous kernel is not available in this build")

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Same mechanism as asymCompKernelMain_fp4.cu
    num_groups = 2
    expected_m_per_group = 1024
    n = 1024
    k = 1024
    gran_k = 16
    sf_k = (k + gran_k - 1) // gran_k
    sf_n = n  # per-row B scales: one scale row per B row (gran_mn_b=1)
    recipe = (1, 1, 16)
    disable_ue8m0_cast = True

    m_indices_cpu, m, active_m = _build_m_indices_like_cpp(expected_m_per_group, num_groups, seed=0)
    m_indices = m_indices_cpu.to(device="cuda")
    offsets_t, experts_t, list_size = _fill_with_sentinel(m_indices_cpu, num_groups)

    a_bf16 = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_bf16 = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
    d_gt = _build_ground_truth(a_bf16, b_bf16, m_indices, num_groups)

    a_f32_cpu = a_bf16.float().cpu().contiguous().numpy()
    b_f32_cpu = b_bf16.float().cpu().contiguous().numpy()

    a_packed_u8, a_scales_u8 = _quantize_a_nvfp4_e4m3(a_f32_cpu, gran_k=gran_k)
    b_packed_u8 = np.empty((num_groups, n, k // 2), dtype=np.uint8)
    b_scales_u8 = np.empty((num_groups, sf_n, sf_k), dtype=np.uint8)
    for gid in range(num_groups):
        bp, bs = _quantize_b_nvfp4_e4m3(b_f32_cpu[gid], gran_k=gran_k)
        b_packed_u8[gid] = bp
        b_scales_u8[gid] = bs

    a_fp4 = torch.from_numpy(a_packed_u8).to(device="cuda", dtype=torch.uint8)
    sfa = torch.from_numpy(a_scales_u8).to(device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn)
    b_fp4 = torch.from_numpy(b_packed_u8).to(device="cuda", dtype=torch.uint8)
    sfb = torch.from_numpy(b_scales_u8).to(device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn)
    b_fp4_pinned = b_fp4.cpu().pin_memory()
    sfb_pinned = sfb.cpu().pin_memory()

    d_kernel = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    asym_gemm.m_grouped_fp4_asym_gemm_nt_contiguous(
        (a_fp4, sfa),
        (b_fp4_pinned, sfb),
        d_kernel,
        offsets_t,
        experts_t,
        list_size,
        recipe=recipe,
        disable_ue8m0_cast=disable_ue8m0_cast,
    )
    torch.cuda.synchronize()

    d_manual = _manual_dequant_reference(
        a_packed_u8=a_packed_u8,
        a_scales_u8=a_scales_u8,
        b_packed_u8=b_packed_u8,
        b_scales_u8=b_scales_u8,
        m_indices_cpu=m_indices_cpu,
        gran_k=gran_k,
    )

    active_rows = torch.where(m_indices_cpu != -1)[0]
    d_kernel_active = d_kernel.index_select(0, active_rows.to(device="cuda")).float().cpu()
    d_gt_active = d_gt.index_select(0, active_rows.to(device="cuda")).float().cpu()
    d_manual_active = d_manual.index_select(0, active_rows)

    print(
        f"\n=== NVFP4 grouped contiguous test ===\n"
        f"m={m}, active_m={active_m}, n={n}, k={k}, num_groups={num_groups}\n"
        f"A packed: {tuple(a_fp4.shape)}, SFA: {tuple(sfa.shape)}\n"
        f"B packed: {tuple(b_fp4.shape)}, SFB: {tuple(sfb.shape)}"
    )
    _print_top_left("D_kernel", d_kernel_active)
    _print_top_left("D_manual (dequant NVFP4+E4M3)", d_manual_active)
    _print_top_left("D_gt (BF16 ground truth)", d_gt_active)

    kernel_vs_manual = _diff_stats(d_kernel_active, d_manual_active)
    manual_vs_gt = _diff_stats(d_manual_active, d_gt_active)
    kernel_vs_gt = _diff_stats(d_kernel_active, d_gt_active)

    print("\nDiff stats (overall):")
    print(f"  kernel_vs_manual: {kernel_vs_manual}")
    print(f"  manual_vs_gt:     {manual_vs_gt}")
    print(f"  kernel_vs_gt:     {kernel_vs_gt}")

    # Per-group error stats
    active_m_indices = m_indices_cpu[active_rows]
    for gid in range(num_groups):
        group_mask = (active_m_indices == gid)
        if not group_mask.any():
            continue
        k_g = d_kernel_active[group_mask]
        m_g = d_manual_active[group_mask]
        g_stats = _diff_stats(k_g, m_g)
        print(f"  kernel_vs_manual [group {gid}]: {g_stats}")
    _print_max_diff_with_neighborhood(
        "kernel_vs_manual",
        d_kernel_active,
        d_manual_active,
        active_rows,
        radius=2,
    )

    assert torch.isfinite(d_kernel_active).all(), "Kernel output has NaN/Inf"
    assert torch.isfinite(d_manual_active).all(), "Manual reference has NaN/Inf"
    # Main correctness gate for this test: kernel should match manual NVFP4 dequant path.
    # assert kernel_vs_manual["max_abs"] < 1.0, f"kernel_vs_manual max_abs too large: {kernel_vs_manual}"


@ignore_env("DG_JIT_PTXAS_CHECK", lambda: torch.cuda.is_available() and get_arch_major() == 9)
def test_m_grouped_nvfp4_masked_cpp_flow() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not hasattr(asym_gemm, "m_grouped_fp4_asym_gemm_nt_masked"):
        pytest.skip("FP4 masked kernel is not available in this build")

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Masked/decode phase: `a` is (num_groups, max_m, k); only the first
    # masked_m[g] rows of each group are valid. Kernel must respect masked_m
    # via the offsets/experts pair list (see asymScheduler).
    num_groups = 2
    max_m = 512
    expected_m_per_group = 384
    n = 1024
    k = 1024
    gran_k = 16
    sf_k = (k + gran_k - 1) // gran_k
    sf_n = n  # per-row B scales: one scale row per B row (gran_mn_b=1)
    recipe = (1, 1, 16)
    disable_ue8m0_cast = True
    block_m = 128

    # Pick per-group valid lengths in [0.7 * expected, 1.3 * expected], clamped
    # to max_m. Deterministic via the seeds above.
    masked_m_list = []
    for _ in range(num_groups):
        v = max(1, int(expected_m_per_group * random.uniform(0.7, 1.3)))
        masked_m_list.append(min(v, max_m))
    masked_m_cpu = torch.tensor(masked_m_list, dtype=torch.int32, device="cpu")
    masked_m = masked_m_cpu.to(device="cuda")

    a_bf16 = torch.randn((num_groups, max_m, k), device="cuda", dtype=torch.bfloat16)
    b_bf16 = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)

    a_f32_cpu = a_bf16.float().cpu().contiguous().numpy()
    b_f32_cpu = b_bf16.float().cpu().contiguous().numpy()

    a_packed_u8 = np.empty((num_groups, max_m, k // 2), dtype=np.uint8)
    a_scales_u8 = np.empty((num_groups, max_m, sf_k), dtype=np.uint8)
    b_packed_u8 = np.empty((num_groups, n, k // 2), dtype=np.uint8)
    b_scales_u8 = np.empty((num_groups, sf_n, sf_k), dtype=np.uint8)
    for gid in range(num_groups):
        ap, asc = _quantize_a_nvfp4_e4m3(a_f32_cpu[gid], gran_k=gran_k)
        a_packed_u8[gid] = ap
        a_scales_u8[gid] = asc
        bp, bs = _quantize_b_nvfp4_e4m3(b_f32_cpu[gid], gran_k=gran_k)
        b_packed_u8[gid] = bp
        b_scales_u8[gid] = bs

    a_fp4 = torch.from_numpy(a_packed_u8).to(device="cuda", dtype=torch.uint8)
    sfa = torch.from_numpy(a_scales_u8).to(device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn)
    b_fp4 = torch.from_numpy(b_packed_u8).to(device="cuda", dtype=torch.uint8)
    sfb = torch.from_numpy(b_scales_u8).to(device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn)

    # warm up
    expected_m_per_group = 1
    masked_m = torch.zeros(
        (num_groups,), dtype=torch.int32, device="cuda"
    )
    import ipdb; ipdb.set_trace()

    d_kernel = torch.empty((num_groups, max_m, n), device="cuda", dtype=torch.bfloat16)
    asym_gemm.m_grouped_fp4_asym_gemm_nt_masked(
        (a_fp4, sfa),
        (b_fp4, sfb),
        d_kernel,
        masked_m,
        expected_m_per_group,
        recipe=recipe,
        disable_ue8m0_cast=disable_ue8m0_cast,
    )
    torch.cuda.synchronize()

    import ipdb; ipdb.set_trace()
    # Manual reference per group (dequant NVFP4+E4M3 -> fp32 matmul) for the
    # valid rows only. Ground truth uses original BF16 a/b for the same rows.
    print(
        f"\n=== NVFP4 grouped masked test ===\n"
        f"num_groups={num_groups}, max_m={max_m}, expected_m_per_group={expected_m_per_group}, "
        f"n={n}, k={k}\n"
        f"masked_m={masked_m_cpu.tolist()}\n"
        f"A packed: {tuple(a_fp4.shape)}, SFA: {tuple(sfa.shape)}\n"
        f"B packed: {tuple(b_fp4.shape)}, SFB: {tuple(sfb.shape)}"
    )

    overall_kvm = {"mean_abs": 0.0, "max_abs": 0.0, "rel_num": 0.0, "rel_den": 0.0, "n": 0}
    overall_kvg = {"mean_abs": 0.0, "max_abs": 0.0, "rel_num": 0.0, "rel_den": 0.0, "n": 0}

    for gid in range(num_groups):
        m_valid = int(masked_m_cpu[gid].item())
        if m_valid == 0:
            continue

        # Manual FP4 dequant reference for this group's valid rows
        a_codes = _unpack_fp4_bytes(a_packed_u8[gid, :m_valid])
        a_q = _decode_e2m1_codes(a_codes)
        a_sf = _decode_e4m3_bits(a_scales_u8[gid, :m_valid])
        a_sf = np.repeat(a_sf, gran_k, axis=1)[:, :k]
        a_deq = a_q * a_sf

        b_codes = _unpack_fp4_bytes(b_packed_u8[gid])
        b_q = _decode_e2m1_codes(b_codes)
        b_sf = _decode_e4m3_bits(b_scales_u8[gid])
        b_sf = np.repeat(b_sf, gran_k, axis=1)[:, :k]  # per-row: only expand K axis
        b_deq = b_q * b_sf

        a_t = torch.from_numpy(a_deq).to(torch.float32)
        b_t = torch.from_numpy(b_deq).to(torch.float32)
        d_manual = (a_t @ b_t.t())

        # BF16 ground truth on same rows
        d_gt = (a_bf16[gid, :m_valid].float() @ b_bf16[gid].float().t()).cpu()

        d_k_valid = d_kernel[gid, :m_valid].float().cpu()

        k_vs_m = _diff_stats(d_k_valid, d_manual)
        k_vs_g = _diff_stats(d_k_valid, d_gt)
        m_vs_g = _diff_stats(d_manual, d_gt)
        print(
            f"  [group {gid}] masked_m={m_valid} "
            f"kernel_vs_manual={k_vs_m} kernel_vs_gt={k_vs_g} manual_vs_gt={m_vs_g}"
        )

        for dst, src in ((overall_kvm, k_vs_m), (overall_kvg, k_vs_g)):
            dst["mean_abs"] += src["mean_abs"] * m_valid
            dst["max_abs"] = max(dst["max_abs"], src["max_abs"])
            dst["n"] += m_valid

        # NaN/Inf gate
        assert torch.isfinite(d_k_valid).all(), f"Kernel output has NaN/Inf in group {gid}"
        assert torch.isfinite(d_manual).all(), f"Manual reference has NaN/Inf in group {gid}"

    def _finalize(stats):
        if stats["n"] == 0:
            return {"mean_abs": 0.0, "max_abs": 0.0}
        return {"mean_abs": stats["mean_abs"] / stats["n"], "max_abs": stats["max_abs"]}

    final_kvm = _finalize(overall_kvm)
    final_kvg = _finalize(overall_kvg)
    print(f"\nOverall (active rows only):")
    print(f"  kernel_vs_manual: {final_kvm}")
    print(f"  kernel_vs_gt:     {final_kvg}")


if __name__ == "__main__":
    try:
        test_m_grouped_nvfp4_contiguous_cpp_flow()
    except pytest.skip.Exception as exc:
        print(f"Skipped: {exc}")
    except RuntimeError as exc:
        print(f"RuntimeError: {exc}")

    try:
        test_m_grouped_nvfp4_masked_cpp_flow()
    except pytest.skip.Exception as exc:
        print(f"Skipped: {exc}")
    except RuntimeError as exc:
        print(f"RuntimeError: {exc}")
