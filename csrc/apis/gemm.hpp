// Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
// Modified by Bytedance Inc., 2026.
// Original: https://github.com/deepseek-ai/DeepGEMM

#pragma once

#include <algorithm>
#include <vector>

#include "../utils/compatibility.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm100_bf16_asym_gemm.hpp"
#include "../jit_kernels/impls/sm100_fp8_asym_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm100_fp4_asym_gemm_1d1d.hpp"
#endif

#include "../jit_kernels/impls/smxx_cublaslt.hpp"
#include "../jit_kernels/impls/sm89_bf16_moe_gemm_masked.hpp"
#include "../jit_kernels/impls/sm80_moe_gemm.hpp"
#include "../jit_kernels/impls/sm89_fp8_moe_gemm.hpp"

#if DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm90_int8_asym_gemm_1d1d.hpp"
#endif

#include "layout.hpp"

namespace asym_gemm::gemm {

// FP4 tensors are passed as packed bytes (two E2M1 elements per uint8).
static void check_packed_fp4_e2m1_tensor(const torch::Tensor& t) {
    DG_HOST_ASSERT(t.scalar_type() == torch::kUInt8);
}

// Broadcast packed UE8M0 scale factors from coarse granularity (e.g., gran_k=128)
// to the fine granularity required by FP4 UMMA (sf_quant_k=16).
static torch::Tensor broadcast_packed_ue8m0_sf(const torch::Tensor& sf,
                                               int replication_factor,
                                               int mn_size) {
    if (replication_factor <= 1) return sf;

    const int ndim = sf.dim();
    const int packed_k_coarse = sf.size(-1);
    const int packed_k_fine = packed_k_coarse * replication_factor;

    int64_t batch = 1;
    for (int i = 0; i < ndim - 1; ++i) batch *= sf.sizes()[i];

    auto flat = torch::empty({batch, packed_k_coarse},
        at::TensorOptions().device(sf.device()).dtype(torch::kInt32));
    flat.copy_(sf.reshape({batch, packed_k_coarse}));

    auto bytes = flat.view(torch::kUInt8).reshape({batch, packed_k_coarse * 4});
    auto rep = bytes.unsqueeze(-1)
                    .expand({batch, packed_k_coarse * 4, static_cast<int64_t>(replication_factor)})
                    .contiguous()
                    .reshape({batch, packed_k_coarse * 4 * replication_factor});
    auto packed = rep.view(torch::kInt32).reshape({batch, packed_k_fine});

    auto sizes = sf.sizes().vec();
    sizes.back() = packed_k_fine;
    packed = packed.reshape(sizes);

    const int tma_aligned_mn = get_tma_aligned_size(mn_size, 4);
    std::vector<int64_t> strides(ndim);
    strides[ndim - 2] = 1;
    strides[ndim - 1] = tma_aligned_mn;
    if (ndim >= 3)
        strides[ndim - 3] = tma_aligned_mn * packed_k_fine;

    auto result = torch::empty_strided(sizes, strides,
        at::TensorOptions().device(sf.device()).dtype(torch::kInt32));
    result.copy_(packed);
    return result;
}

static bool early_return(const int& m, const int &n, const int& k,
                         const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    // Do nothing if the problem is empty
    if (m == 0 or n == 0)
        return true;

    // Checks
    const bool& is_cd_same = c.has_value() and c->data_ptr() == d.data_ptr();
    if (is_cd_same)
        DG_HOST_ASSERT(c->sizes() == d.sizes() and c->strides() == d.strides());
    if (c.has_value()) {
        check_major_type_cd(c.value());
        DG_HOST_ASSERT(d.scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(c.value().scalar_type() == torch::kFloat);
    }

    // No accumulation
    if (k == 0) {
        if (not is_cd_same)
            c.has_value() ? d.copy_(c.value()) : d.zero_();
        return true;
    }

    // With accumulation, do copy before GEMM (assuming the GEMM kernel does not support different C/D)
    if (c.has_value() and not is_cd_same)
        d.copy_(c.value());
    return false;
}

// Validate the list_size tensor shape/dtype without copying it to host.
// The host can no longer read the value (capture-mode incompatibility), so the
// kernel launches with grid Y = num_groups and per-block sentinels handle no-ops.
static void check_list_size_tensor(const torch::Tensor& list_size_t) {
    DG_HOST_ASSERT(list_size_t.numel() == 1);
    DG_HOST_ASSERT(list_size_t.scalar_type() == torch::kInt);
}

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
static void m_grouped_fp8_asym_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const torch::Tensor& offsets, const torch::Tensor& experts,
                                             const torch::Tensor& list_size_t,
                                             std::optional<std::tuple<int, int, int>> recipe,
                                             const std::string& compiled_dims,
                                             const bool& disable_ue8m0_cast) {
    check_list_size_tensor(list_size_t);
    // Shape must be `[M, K] @ [G, N, K].mT`
    const auto& major_a = get_major_type_ab(a.first);
    const auto& major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    if (fp8_requires_k_major())
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    // Type and shape checks
    const auto& [m, k] = get_shape<2>(a.first);
    const auto& [num_groups, n, k_] = get_shape<3>(b.first);
    const auto& [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(offsets.is_cuda() && experts.is_cuda());
    DG_HOST_ASSERT(offsets.is_contiguous() && experts.is_contiguous());
    DG_HOST_ASSERT(offsets.scalar_type() == torch::kInt && experts.scalar_type() == torch::kInt);
    // Dense per-group layout: offsets is num_groups pairs, experts is num_groups + terminator.
    DG_HOST_ASSERT(offsets.numel() >= 2 * num_groups && experts.numel() >= num_groups + 1);

    // D must be N-major
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0)
        return;

    // Transform SFA and SFB into compute-required layout
    if (not recipe.has_value())
        recipe = get_default_recipe(a.second.scalar_type(), b.second.scalar_type());
    const auto& sfa = layout::transform_sf_into_required_layout(a.second, m, k, recipe.value(), std::nullopt,  true, disable_ue8m0_cast);
    const auto& sfb = layout::transform_sf_into_required_layout(b.second, n, k, recipe.value(),   num_groups, false, disable_ue8m0_cast);

    // Dispatch implementation. Grid Y = num_groups; sentinel blocks early-exit in-kernel.
    const auto& arch_major = device_runtime->get_arch_major();
    sm100_m_grouped_fp8_asym_gemm_contiguous_1d1d(a.first, sfa, b.first, sfb, d,
                                                 offsets, experts, /*grid_y=*/num_groups,
                                                 num_groups, m, n, k, major_a, major_b, compiled_dims);
}

static void m_grouped_fp8_asym_gemm_nt_masked(const std::pair<torch::Tensor, torch::Tensor>& a,
                                         const std::pair<torch::Tensor, torch::Tensor>& b,
                                         const torch::Tensor& d,
                                         const torch::Tensor& masked_m,
                                         const int& expected_m,
                                         std::optional<std::tuple<int, int, int>> recipe,
                                         const std::string& compiled_dims,
                                         const bool& disable_ue8m0_cast) {
    // Shape must be `[G, M, K] @ [G, N, K].mT`
    const auto& major_a = get_major_type_ab(a.first);
    const auto& major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    if (fp8_requires_k_major())
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);

    // Type and shape checks
    const auto& [num_groups, m, k] = get_shape<3>(a.first);
    const auto& [num_groups_, n, k_] = get_shape<3>(b.first);
    const auto& [num_groups__, m_, n_] = get_shape<3>(d);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(num_groups == num_groups_ and num_groups == num_groups__);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    // masked_m: int32 GPU tensor of shape [num_groups] with per-group token counts
    DG_HOST_ASSERT(masked_m.is_cuda() && masked_m.is_contiguous());
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(masked_m.numel() == num_groups);

    // D must be N-major (per-group)
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0 or expected_m == 0)
        return;

    // Transform SFA and SFB into compute-required layout
    if (not recipe.has_value())
        recipe = get_default_recipe(a.second.scalar_type(), b.second.scalar_type());
    const auto& sfa = layout::transform_sf_into_required_layout(a.second, m, k, recipe.value(), num_groups, true, disable_ue8m0_cast);
    const auto& sfb = layout::transform_sf_into_required_layout(b.second, n, k, recipe.value(), num_groups, false, disable_ue8m0_cast);

    // Launch with gridDim.y == num_groups (constant) — CUDA-graph safe.
    sm100_m_grouped_fp8_asym_gemm_masked_1d1d(a.first, sfa, b.first, sfb, d, masked_m, expected_m,
                                              num_groups, m, n, k, major_a, major_b, compiled_dims);
}

static void m_grouped_fp4_asym_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const torch::Tensor& offsets, const torch::Tensor& experts,
                                             const int& list_size,
                                             std::optional<std::tuple<int, int, int>> recipe,
                                             const std::string& compiled_dims,
                                             const bool& disable_ue8m0_cast) {
    // FP4 (NVFP4) asym GEMM uses the Blackwell TMA/UMMA block-scaled path only.
    // There is no SM90 (Hopper/H20) FP4 kernel, so fail loudly rather than fall
    // through to the SM100 kernel (which cannot launch on other archs).
    if (device_runtime->get_arch_major() != 10)
        DG_HOST_UNREACHABLE("FP4 (NVFP4) asym GEMM is only supported on SM100 (Blackwell)");

    const auto& major_a = get_major_type_ab(a.first);
    const auto& major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);

    const auto& [m, k_packed] = get_shape<2>(a.first);
    const auto& [num_groups, n, k_packed_] = get_shape<3>(b.first);
    const int k = k_packed * 2;
    const auto& [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k_packed == k_packed_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    check_packed_fp4_e2m1_tensor(a.first);
    check_packed_fp4_e2m1_tensor(b.first);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    check_major_type_cd(d);
    if (m == 0) return;
    if (not recipe.has_value()) recipe = get_default_recipe(a.second.scalar_type(), b.second.scalar_type());
    const auto& sfa_raw = layout::transform_sf_into_required_layout(a.second, m, k, recipe.value(), std::nullopt, true, disable_ue8m0_cast);
    const auto& sfb_raw = layout::transform_sf_into_required_layout(b.second, n, k, recipe.value(), num_groups, false, disable_ue8m0_cast);

    constexpr int fp4_sf_quant_k = 16;
    const int sf_gran_k = std::get<2>(recipe.value());
    const int sf_replication = sf_gran_k / fp4_sf_quant_k;
    auto sfa = broadcast_packed_ue8m0_sf(sfa_raw, sf_replication, m);
    auto sfb = broadcast_packed_ue8m0_sf(sfb_raw, sf_replication, n);

    const int gran_mn_a = std::get<0>(recipe.value());
    const int gran_mn_b = std::get<1>(recipe.value());
    if (gran_mn_a > 1 && static_cast<int>(sfa.size(-2)) < m) {
        const auto idx = torch::arange(m, at::TensorOptions().device(sfa.device()).dtype(torch::kLong)).floor_divide_(gran_mn_a);
        const auto broadcasted = sfa.index_select(-2, idx);
        const int tma_aligned_mn = get_tma_aligned_size(m, static_cast<int>(sfa.element_size()));
        const auto sf_k_dim = broadcasted.size(-1);
        sfa = torch::empty_strided({m, sf_k_dim}, {1, tma_aligned_mn}, broadcasted.options());
        sfa.copy_(broadcasted);
    }
    if (gran_mn_b > 1 && static_cast<int>(sfb.size(-2)) < n) {
        const auto idx = torch::arange(n, at::TensorOptions().device(sfb.device()).dtype(torch::kLong)).floor_divide_(gran_mn_b);
        const auto broadcasted = sfb.index_select(-2, idx);
        const int tma_aligned_mn = get_tma_aligned_size(n, static_cast<int>(sfb.element_size()));
        const auto sf_k_dim = broadcasted.size(-1);
        sfb = torch::empty_strided({num_groups, n, sf_k_dim},
                                   {tma_aligned_mn * sf_k_dim, 1, tma_aligned_mn}, broadcasted.options());
        sfb.copy_(broadcasted);
    }

    sm100_m_grouped_fp4_asym_gemm_contiguous_1d1d(a.first, sfa, b.first, sfb, d,
                                                 offsets, experts, list_size,
                                                 num_groups, m, n, k, major_a, major_b, compiled_dims);
}

static void m_grouped_fp4_asym_gemm_nt_masked(const std::pair<torch::Tensor, torch::Tensor>& a,
                                         const std::pair<torch::Tensor, torch::Tensor>& b,
                                         const torch::Tensor& d,
                                         const torch::Tensor& masked_m,
                                         const int& expected_m,
                                         std::optional<std::tuple<int, int, int>> recipe,
                                         const std::string& compiled_dims,
                                         const bool& disable_ue8m0_cast) {
    // FP4 (NVFP4) asym GEMM is Blackwell-only; no SM90 (Hopper/H20) kernel exists.
    if (device_runtime->get_arch_major() != 10)
        DG_HOST_UNREACHABLE("FP4 (NVFP4) asym GEMM is only supported on SM100 (Blackwell)");

    const auto& major_a = get_major_type_ab(a.first);
    const auto& major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);

    const auto& [num_groups, m, k_packed] = get_shape<3>(a.first);
    const auto& [num_groups_, n, k_packed_] = get_shape<3>(b.first);
    const int k = k_packed * 2;
    const auto& [num_groups__, m_, n_] = get_shape<3>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k_packed == k_packed_);
    check_packed_fp4_e2m1_tensor(a.first);
    check_packed_fp4_e2m1_tensor(b.first);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.is_cuda() && masked_m.is_contiguous());
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(masked_m.numel() == num_groups);

    if (m == 0 or expected_m == 0) return;
    if (not recipe.has_value()) recipe = get_default_recipe(a.second.scalar_type(), b.second.scalar_type());
    const auto& sfa_raw = layout::transform_sf_into_required_layout(a.second, m, k, recipe.value(), num_groups, true, disable_ue8m0_cast);
    const auto& sfb_raw = layout::transform_sf_into_required_layout(b.second, n, k, recipe.value(), num_groups, false, disable_ue8m0_cast);

    constexpr int fp4_sf_quant_k = 16;
    const int sf_gran_k = std::get<2>(recipe.value());
    const int sf_replication = sf_gran_k / fp4_sf_quant_k;
    auto sfa = broadcast_packed_ue8m0_sf(sfa_raw, sf_replication, m);
    auto sfb = broadcast_packed_ue8m0_sf(sfb_raw, sf_replication, n);

    const int gran_mn_a = std::get<0>(recipe.value());
    const int gran_mn_b = std::get<1>(recipe.value());
    if (gran_mn_a > 1 && static_cast<int>(sfa.size(-2)) < m) {
        const auto idx = torch::arange(m, at::TensorOptions().device(sfa.device()).dtype(torch::kLong)).floor_divide_(gran_mn_a);
        const auto broadcasted = sfa.index_select(-2, idx);
        const int tma_aligned_mn = get_tma_aligned_size(m, static_cast<int>(sfa.element_size()));
        const auto sf_k_dim = broadcasted.size(-1);
        sfa = torch::empty_strided({num_groups, m, sf_k_dim},
                                   {tma_aligned_mn * sf_k_dim, 1, tma_aligned_mn}, broadcasted.options());
        sfa.copy_(broadcasted);
    }
    if (gran_mn_b > 1 && static_cast<int>(sfb.size(-2)) < n) {
        const auto idx = torch::arange(n, at::TensorOptions().device(sfb.device()).dtype(torch::kLong)).floor_divide_(gran_mn_b);
        const auto broadcasted = sfb.index_select(-2, idx);
        const int tma_aligned_mn = get_tma_aligned_size(n, static_cast<int>(sfb.element_size()));
        const auto sf_k_dim = broadcasted.size(-1);
        sfb = torch::empty_strided({num_groups, n, sf_k_dim},
                                   {tma_aligned_mn * sf_k_dim, 1, tma_aligned_mn}, broadcasted.options());
        sfb.copy_(broadcasted);
    }

    sm100_m_grouped_fp4_asym_gemm_masked_1d1d(a.first, sfa, b.first, sfb, d, masked_m, expected_m,
                                              num_groups, m, n, k, major_a, major_b, compiled_dims);
}
#endif


#if DG_TENSORMAP_COMPATIBLE
static void m_grouped_bf16_asym_gemm_nt_contiguous(const torch::Tensor& a, const torch::Tensor& b,
                                              const torch::Tensor& d,
                                              const torch::Tensor& offsets, const torch::Tensor& experts,
                                              const torch::Tensor& list_size_t,
                                              const std::string& compiled_dims) {
    check_list_size_tensor(list_size_t);
    // Shape must be `[M, K] @ [G, N, K].mT`
    const auto& major_a = get_major_type_ab(a);
    const auto& major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);

    // Type and shape checks
    const auto& [m, k] = get_shape<2>(a);
    const auto& [num_groups, n, k_] = get_shape<3>(b);
    const auto& [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    DG_HOST_ASSERT(offsets.is_cuda() && experts.is_cuda());
    DG_HOST_ASSERT(offsets.is_contiguous() && experts.is_contiguous());
    DG_HOST_ASSERT(offsets.scalar_type() == torch::kInt && experts.scalar_type() == torch::kInt);
    // Dense per-group layout: offsets is num_groups pairs, experts is num_groups + terminator.
    DG_HOST_ASSERT(offsets.numel() >= 2 * num_groups && experts.numel() >= num_groups + 1);

    // D must be N-major
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0)
        return;

    // Grid Y = num_groups; sentinel blocks early-exit in-kernel.
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 10) {
        sm100_m_grouped_bf16_asym_gemm_contiguous(a, b, d,
                                                offsets, experts, /*grid_y=*/num_groups,
                                                num_groups, m, n, k, major_a, major_b, compiled_dims);
        return;
    }
    // SM90 (Hopper/H20): the contiguous BF16 layout is only reached via DeepEP-normal
    // dispatch. The native SM90 BF16 asym kernel (sm90_bf16_asym_gemm.cuh) is not yet
    // numerically correct, and the contiguous offset layout is not compatible with the
    // SM80-style kernel used by the masked path, so this combination is unsupported for
    // now. Fail loudly rather than silently produce wrong results.
    DG_HOST_UNREACHABLE("BF16 contiguous asym GEMM is not supported on this architecture "
                        "(only SM100 has a verified kernel; the standard serving path uses "
                        "the masked dispatcher, which is supported on SM90)");
}

static void m_grouped_bf16_asym_gemm_nt_masked(const torch::Tensor& a, const torch::Tensor& b,
                                               const torch::Tensor& d,
                                               const torch::Tensor& masked_m,
                                               const int& expected_m,
                                               const std::string& compiled_dims) {
    // Shape must be `[G, M, K] @ [G, N, K].mT`
    const auto& major_a = get_major_type_ab(a);
    const auto& major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);

    // Type and shape checks
    const auto& [num_groups, m, k] = get_shape<3>(a);
    const auto& [num_groups_, n, k_] = get_shape<3>(b);
    const auto& [num_groups__, m_, n_] = get_shape<3>(d);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(num_groups == num_groups_ and num_groups == num_groups__);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    // masked_m: int32 GPU tensor of shape [num_groups] with per-group token counts.
    DG_HOST_ASSERT(masked_m.is_cuda() && masked_m.is_contiguous());
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(masked_m.numel() == num_groups);

    // D must be N-major (per-group)
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0 or expected_m == 0)
        return;

    const auto& [arch_major, arch_minor] = device_runtime->get_arch_pair();
    // Ada (SM89) and Hopper (SM90/H20) both use the arch-aware SM80-style grouped
    // MoE kernel (native SM90 WGMMA asym kernel exists in sm90_bf16_asym_gemm.cuh
    // but is not yet numerically correct). It flattens the padded [G, M_max, K]
    // masked layout and skips padding rows in-kernel, all via GPU tensor ops so it
    // is CUDA-graph capturable.
    if (arch_major == 9 or (arch_major == 8 and arch_minor == 9)) {
        sm89_m_grouped_bf16_moe_gemm_masked(
            a,
            b,
            d,
            masked_m,
            m,
            n,
            k,
            static_cast<int32_t>(num_groups));
        return;
    }

    // Blackwell (SM100): native TMA/UMMA asym kernel.
    // Grid Y = num_groups; inactive groups early-exit in-kernel.
    if (arch_major == 10) {
        sm100_m_grouped_bf16_asym_gemm_masked(a, b, d, masked_m, /*grid_y=*/num_groups, expected_m,
                                              num_groups, m, n, k, major_a, major_b, compiled_dims);
        return;
    }

    DG_HOST_UNREACHABLE("BF16 masked asym GEMM is not supported on this architecture "
                        "(supported: SM89, SM90/H20, SM100)");
}
#endif

static void m_grouped_moe_gemm_nt_contiguous(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& d,
    const torch::Tensor& offsets,
    const torch::Tensor& experts,
    const int& list_size,
    const std::string& compiled_dims)
{
    // ── Shape checks ──────────────────────────────────────────────────────────
    DG_HOST_ASSERT(a.dim() == 2);
    DG_HOST_ASSERT(b.dim() == 3);
    DG_HOST_ASSERT(d.dim() == 2);

    const int64_t total_tokens   = a.size(0);
    const int64_t K              = a.size(1);
    const int64_t num_experts    = b.size(0);
    const int64_t N              = b.size(1);
    const int64_t K_b            = b.size(2);
    const int64_t total_tokens_d = d.size(0);
    const int64_t N_d            = d.size(1);

    DG_HOST_ASSERT(K == K_b);
    DG_HOST_ASSERT(N == N_d);
    DG_HOST_ASSERT(total_tokens == total_tokens_d);

    // ── Dtype checks ──────────────────────────────────────────────────────────
    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat16 or a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == a.scalar_type());
    DG_HOST_ASSERT(d.scalar_type() == a.scalar_type());

    // ── CUDA placement ────────────────────────────────────────────────────────
    DG_HOST_ASSERT(a.is_cuda());
    DG_HOST_ASSERT(b.is_cuda());
    DG_HOST_ASSERT(d.is_cuda());

    // ── Contiguity ────────────────────────────────────────────────────────────
    DG_HOST_ASSERT(a.is_contiguous());
    DG_HOST_ASSERT(b.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());

    // ── offsets / experts ─────────────────────────────────────────────────────
    DG_HOST_ASSERT(offsets.is_cuda() and experts.is_cuda());
    DG_HOST_ASSERT(offsets.is_contiguous() and experts.is_contiguous());
    DG_HOST_ASSERT(offsets.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(experts.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(offsets.numel() >= list_size and experts.numel() >= list_size);

    // ── Empty check (before alignment guards so K==0 doesn't trigger K>=64) ──
    if (total_tokens == 0 or N == 0 or K == 0) return;

    // ── Alignment checks ──────────────────────────────────────────────────────
    DG_HOST_ASSERT(K % 16 == 0);
    DG_HOST_ASSERT(N % 32 == 0);
    DG_HOST_ASSERT(K >= 64);

    // ── Resolve element type and dispatch ─────────────────────────────────────
    // compiled_dims is accepted for API parity with m_grouped_bf16_asym_gemm_nt_contiguous
    // but unused: the SM80 kernel selects tile dims via runtime heuristic.
    const std::string element_type_str =
        (a.scalar_type() == torch::kFloat16) ? "cutlass::half_t" : "cutlass::bfloat16_t";

    sm80_m_grouped_moe_gemm_contiguous(
        a, b, d, experts, offsets,
        N, K,
        static_cast<int32_t>(num_experts),
        static_cast<int32_t>(list_size),
        element_type_str);
}

static void m_grouped_fp8_asym_gemm_sm89(
    const torch::Tensor& a,        // [total_tokens, K]   float8_e4m3fn  (HBM)
    const torch::Tensor& b,        // [num_experts, N, K] float8_e4m3fn  (CPU pinned or HBM)
    const torch::Tensor& d,        // [total_tokens, N]   bfloat16       (HBM)
    const torch::Tensor& offsets,  // [list_size] int32 cumulative end indices
    const torch::Tensor& experts,  // [list_size] int32 expert IDs
    const int&           list_size,
    const float&         scale_a,
    const float&         scale_b,
    const std::optional<torch::Tensor>& scale_a_tensor = std::nullopt,
    const std::optional<torch::Tensor>& scale_b_tensor = std::nullopt)
{
    DG_HOST_ASSERT(a.dim() == 2 && b.dim() == 3 && d.dim() == 2);

    const int64_t total_tokens = a.size(0);
    const int64_t K            = a.size(1);
    const int64_t num_experts  = b.size(0);
    const int64_t N            = b.size(1);
    DG_HOST_ASSERT(b.size(2) == K);
    DG_HOST_ASSERT(d.size(0) == total_tokens && d.size(1) == N);

    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    // a and d must be on CUDA; b may be on CPU pinned memory (PCIe) or CUDA
    DG_HOST_ASSERT(a.is_cuda() && d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() && b.is_contiguous() && d.is_contiguous());

    DG_HOST_ASSERT(offsets.is_cuda() && experts.is_cuda());
    DG_HOST_ASSERT(offsets.is_contiguous() && experts.is_contiguous());
    DG_HOST_ASSERT(offsets.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(experts.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(offsets.numel() >= list_size && experts.numel() >= list_size);

    if (total_tokens == 0 || N == 0 || K == 0) return;

    DG_HOST_ASSERT(K % 32 == 0 && K >= 32);   // SM89 FP8 MMA K-atom = 32
    DG_HOST_ASSERT(N % 32 == 0);

    sm89_m_grouped_fp8_moe_gemm_contiguous(
        a, b, d, experts, offsets,
        N, K,
        static_cast<int32_t>(num_experts),
        static_cast<int32_t>(list_size),
        scale_a, scale_b,
        scale_a_tensor, scale_b_tensor);
}

static void m_grouped_fp8_asym_gemm_sm89_masked(
    const torch::Tensor& a,        // [num_groups, M_max, K]  float8_e4m3fn
    const torch::Tensor& b,        // [num_groups, N, K]      float8_e4m3fn
    const torch::Tensor& d,        // [num_groups, M_max, N]  bfloat16
    const torch::Tensor& masked_m, // [num_groups]            int32
    const int&           expected_m,
    const float&         scale_a,
    const float&         scale_b,
    const std::optional<torch::Tensor>& scale_a_tensor = std::nullopt,
    const std::optional<torch::Tensor>& scale_b_tensor = std::nullopt)
{
    DG_HOST_ASSERT(a.dim() == 3 && b.dim() == 3 && d.dim() == 3);

    const int64_t num_groups = a.size(0);
    const int64_t M_max      = a.size(1);
    const int64_t K          = a.size(2);
    const int64_t N          = b.size(1);
    DG_HOST_ASSERT(b.size(0) == num_groups && b.size(2) == K);
    DG_HOST_ASSERT(d.size(0) == num_groups && d.size(1) == M_max && d.size(2) == N);

    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    // a and d must be on CUDA; b may be on CPU pinned memory (PCIe) or CUDA
    DG_HOST_ASSERT(a.is_cuda() && d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() && b.is_contiguous() && d.is_contiguous());

    DG_HOST_ASSERT(masked_m.is_cuda() && masked_m.is_contiguous());
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(masked_m.numel() == num_groups);

    if (M_max == 0 || N == 0 || K == 0 || expected_m == 0) return;

    DG_HOST_ASSERT(K % 32 == 0 && K >= 32);
    DG_HOST_ASSERT(N % 32 == 0);

    sm89_m_grouped_fp8_moe_gemm_masked(
        a, b, d, masked_m,
        M_max, N, K,
        static_cast<int32_t>(num_groups),
        scale_a, scale_b,
        scale_a_tensor, scale_b_tensor);
}

#if DG_TENSORMAP_COMPATIBLE
// SM90 (Hopper / H20) INT8 asymmetric grouped GEMM. INT8 inputs + FP32 per-token (A)
// / per-channel (B) scales, FP32 output. Scale tensors must arrive pre-transposed
// into the TMA SF layout (done in the Python dispatch layer):
//   sfa: [num_groups, ceil(k/128), m]   (float32)
//   sfb: [ceil(k/128), num_groups * n]  (float32)
static void m_grouped_int8_asym_gemm_sm90_masked(
        const torch::Tensor& a,         // [G, M_max, K] int8
        const torch::Tensor& b,         // [G, N, K]     int8
        const torch::Tensor& d,         // [G, M_max, N] float32
        const torch::Tensor& masked_m,  // [G]           int32
        const int& expected_m,
        const torch::Tensor& sfa,       // [G, ceil(K/128), M_max] float32
        const torch::Tensor& sfb) {     // [ceil(K/128), G*N]      float32
    DG_HOST_ASSERT(a.dim() == 3 and b.dim() == 3 and d.dim() == 3);
    const int64_t num_groups = a.size(0);
    const int64_t m = a.size(1);
    const int64_t k = a.size(2);
    const int64_t n = b.size(1);
    DG_HOST_ASSERT(b.size(0) == num_groups and b.size(2) == k);
    DG_HOST_ASSERT(d.size(0) == num_groups and d.size(1) == m and d.size(2) == n);

    DG_HOST_ASSERT(a.scalar_type() == torch::kChar and b.scalar_type() == torch::kChar);
    DG_HOST_ASSERT(d.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kFloat and sfb.scalar_type() == torch::kFloat);
    // Activations (A), output (D), and both scale tensors (SFA, SFB) live in device memory.
    DG_HOST_ASSERT(a.is_cuda() and d.is_cuda() and sfa.is_cuda() and sfb.is_cuda());
    // Only the expert weights (B) may be PCIe/host-resident: Hopper TMA can fetch
    // directly from CUDA-pinned host memory over the UVA address space.
    DG_HOST_ASSERT(b.is_cuda() or b.is_pinned());
    DG_HOST_ASSERT(a.is_contiguous() and b.is_contiguous() and d.is_contiguous());
    DG_HOST_ASSERT(sfa.is_contiguous() and sfb.is_contiguous());
    DG_HOST_ASSERT(masked_m.is_cuda() and masked_m.is_contiguous());
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt and masked_m.numel() == num_groups);

    if (m == 0 or n == 0 or k == 0 or expected_m == 0) return;

    const auto& major_a = get_major_type_ab(a);
    const auto& major_b = get_major_type_ab(b);
    sm90_m_grouped_int8_asym_gemm_masked_1d1d(a, sfa, b, sfb, d, masked_m, expected_m,
                                              static_cast<int>(num_groups),
                                              static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                                              major_a, major_b, "nk");
}

static void m_grouped_int8_asym_gemm_sm90_contiguous(
        const torch::Tensor& a,         // [M, K]    int8
        const torch::Tensor& b,         // [G, N, K] int8
        const torch::Tensor& d,         // [M, N]    float32
        const torch::Tensor& offsets,   // [>=2*G]   int32 (start,end) pairs
        const torch::Tensor& experts,   // [>=G+1]   int32 (with -1 terminator)
        const int& list_size,
        const torch::Tensor& sfa,       // [ceil(K/128), M]   float32
        const torch::Tensor& sfb) {     // [ceil(K/128), G*N] float32
    DG_HOST_ASSERT(a.dim() == 2 and b.dim() == 3 and d.dim() == 2);
    const int64_t m = a.size(0);
    const int64_t k = a.size(1);
    const int64_t num_groups = b.size(0);
    const int64_t n = b.size(1);
    DG_HOST_ASSERT(b.size(2) == k);
    DG_HOST_ASSERT(d.size(0) == m and d.size(1) == n);

    DG_HOST_ASSERT(a.scalar_type() == torch::kChar and b.scalar_type() == torch::kChar);
    DG_HOST_ASSERT(d.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kFloat and sfb.scalar_type() == torch::kFloat);
    // Activations (A), output (D), and both scale tensors (SFA, SFB) live in device memory.
    DG_HOST_ASSERT(a.is_cuda() and d.is_cuda() and sfa.is_cuda() and sfb.is_cuda());
    // Only the expert weights (B) may be PCIe/host-resident: Hopper TMA can fetch
    // directly from CUDA-pinned host memory over the UVA address space.
    DG_HOST_ASSERT(b.is_cuda() or b.is_pinned());
    DG_HOST_ASSERT(a.is_contiguous() and b.is_contiguous() and d.is_contiguous());
    DG_HOST_ASSERT(sfa.is_contiguous() and sfb.is_contiguous());
    DG_HOST_ASSERT(offsets.is_cuda() and experts.is_cuda());
    DG_HOST_ASSERT(offsets.scalar_type() == torch::kInt and experts.scalar_type() == torch::kInt);

    if (m == 0 or n == 0 or k == 0) return;

    const auto& major_a = get_major_type_ab(a);
    const auto& major_b = get_major_type_ab(b);
    sm90_m_grouped_int8_asym_gemm_contiguous_1d1d(a, sfa, b, sfb, d, offsets, experts,
                                                  /*grid_y=*/list_size - 1,
                                                  static_cast<int>(num_groups),
                                                  static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                                                  major_a, major_b, "nk");
}
#endif

static void register_apis(pybind11::module_& m) {

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    // FP8 GEMMs
    m.def("m_grouped_fp8_asym_gemm_nt_contiguous",
        static_cast<void(*)(const std::pair<torch::Tensor, torch::Tensor>&,
                            const std::pair<torch::Tensor, torch::Tensor>&,
                            const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                            std::optional<std::tuple<int, int, int>>, const std::string&, const bool&)>(
            &m_grouped_fp8_asym_gemm_nt_contiguous),
        py::arg("a"), py::arg("b"), py::arg("d"),
        py::arg("offsets"), py::arg("experts"), py::arg("list_size"),
        py::arg("recipe") = std::nullopt, py::arg("compiled_dims") = "nk",
        py::arg("disable_ue8m0_cast") = false);
    m.def("m_grouped_fp8_asym_gemm_nt_masked",
        static_cast<void(*)(const std::pair<torch::Tensor, torch::Tensor>&,
                            const std::pair<torch::Tensor, torch::Tensor>&,
                            const torch::Tensor&, const torch::Tensor&, const int&,
                            std::optional<std::tuple<int, int, int>>, const std::string&, const bool&)>(
            &m_grouped_fp8_asym_gemm_nt_masked),
        py::arg("a"), py::arg("b"), py::arg("d"),
        py::arg("masked_m"), py::arg("expected_m"),
        py::arg("recipe") = std::nullopt, py::arg("compiled_dims") = "nk",
        py::arg("disable_ue8m0_cast") = false);

    // FP4 GEMMs
    m.def("m_grouped_fp4_asym_gemm_nt_contiguous",
        static_cast<void(*)(const std::pair<torch::Tensor, torch::Tensor>&,
                            const std::pair<torch::Tensor, torch::Tensor>&,
                            const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const int&,
                            std::optional<std::tuple<int, int, int>>, const std::string&, const bool&)>(
            &m_grouped_fp4_asym_gemm_nt_contiguous),
        py::arg("a"), py::arg("b"), py::arg("d"),
        py::arg("offsets"), py::arg("experts"), py::arg("list_size"),
        py::arg("recipe") = std::nullopt, py::arg("compiled_dims") = "nk",
        py::arg("disable_ue8m0_cast") = false);
    m.def("m_grouped_fp4_asym_gemm_nt_masked",
        static_cast<void(*)(const std::pair<torch::Tensor, torch::Tensor>&,
                            const std::pair<torch::Tensor, torch::Tensor>&,
                            const torch::Tensor&, const torch::Tensor&, const int&,
                            std::optional<std::tuple<int, int, int>>, const std::string&, const bool&)>(
            &m_grouped_fp4_asym_gemm_nt_masked),
        py::arg("a"), py::arg("b"), py::arg("d"),
        py::arg("masked_m"), py::arg("expected_m"),
        py::arg("recipe") = std::nullopt, py::arg("compiled_dims") = "nk",
        py::arg("disable_ue8m0_cast") = false);
#endif

#if DG_TENSORMAP_COMPATIBLE
    // BF16 GEMMs
    m.def("m_grouped_bf16_asym_gemm_nt_contiguous",
          static_cast<void(*)(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                              const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const std::string&)>(
              &m_grouped_bf16_asym_gemm_nt_contiguous),
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("offsets"), py::arg("experts"), py::arg("list_size"),
          py::arg("compiled_dims") = "nk");
    m.def("m_grouped_bf16_asym_gemm_nt_masked",
          static_cast<void(*)(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                              const torch::Tensor&, const int&, const std::string&)>(
              &m_grouped_bf16_asym_gemm_nt_masked),
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("masked_m"), py::arg("expected_m"),
          py::arg("compiled_dims") = "nk");
#endif

#if DG_TENSORMAP_COMPATIBLE
    // SM90 (Hopper / H20) INT8 asymmetric grouped GEMM (native S8 WGMMA, FP32 output)
    m.def("m_grouped_int8_asym_gemm_sm90_masked",
          static_cast<void(*)(const torch::Tensor&, const torch::Tensor&,
                              const torch::Tensor&, const torch::Tensor&,
                              const int&, const torch::Tensor&, const torch::Tensor&)>(
              &m_grouped_int8_asym_gemm_sm90_masked),
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("masked_m"), py::arg("expected_m"),
          py::arg("sfa"), py::arg("sfb"));
    m.def("m_grouped_int8_asym_gemm_sm90_contiguous",
          static_cast<void(*)(const torch::Tensor&, const torch::Tensor&,
                              const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                              const int&, const torch::Tensor&, const torch::Tensor&)>(
              &m_grouped_int8_asym_gemm_sm90_contiguous),
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("offsets"), py::arg("experts"), py::arg("list_size"),
          py::arg("sfa"), py::arg("sfb"));
#endif

    // SM89 FP8 MoE GEMM — masked variant (padded [G, M_max, K] layout)
    m.def("m_grouped_fp8_asym_gemm_sm89_masked",
          static_cast<void(*)(const torch::Tensor&, const torch::Tensor&,
                              const torch::Tensor&, const torch::Tensor&,
                              const int&, const float&, const float&,
                              const std::optional<torch::Tensor>&,
                              const std::optional<torch::Tensor>&)>(
              &m_grouped_fp8_asym_gemm_sm89_masked),
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("masked_m"), py::arg("expected_m"),
          py::arg("scale_a") = 1.0f,
          py::arg("scale_b") = 1.0f,
          py::arg("scale_a_tensor") = py::none(),
          py::arg("scale_b_tensor") = py::none());

    // SM89 FP8 MoE GEMM (native FP8 MMA, K-outer M-inner, W may be CPU-pinned)
    m.def("m_grouped_fp8_asym_gemm_sm89",
          static_cast<void(*)(const torch::Tensor&, const torch::Tensor&,
                              const torch::Tensor&, const torch::Tensor&,
                              const torch::Tensor&, const int&,
                              const float&, const float&,
                              const std::optional<torch::Tensor>&,
                              const std::optional<torch::Tensor>&)>(
              &m_grouped_fp8_asym_gemm_sm89),
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("offsets"), py::arg("experts"), py::arg("list_size"),
          py::arg("scale_a") = 1.0f,
          py::arg("scale_b") = 1.0f,
          py::arg("scale_a_tensor") = py::none(),
          py::arg("scale_b_tensor") = py::none());

    // SM80 MoE GEMM (FP16 + BF16, no arch guard needed: uses >= SM80 primitives)
    m.def("m_grouped_moe_gemm_nt_contiguous",
          static_cast<void(*)(const torch::Tensor&, const torch::Tensor&,
                              const torch::Tensor&, const torch::Tensor&,
                              const torch::Tensor&, const int&,
                              const std::string&)>(
              &m_grouped_moe_gemm_nt_contiguous),
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("offsets"), py::arg("experts"), py::arg("list_size"),
          py::arg("compiled_dims") = "nk");
}

} // namespace asym_gemm::gemm
