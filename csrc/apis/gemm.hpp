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
#include "../jit_kernels/impls/sm90_fp8_asym_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm90_bf16_asym_gemm.hpp"
#endif

#include "../jit_kernels/impls/smxx_cublaslt.hpp"
#include "../jit_kernels/impls/sm89_bf16_asym_gemm.hpp"
#include "../jit_kernels/impls/sm80_moe_gemm.hpp"
#include "../jit_kernels/impls/sm80_int8_asym_gemm.hpp"
#include "../jit_kernels/impls/sm89_fp8_asym_gemm.hpp"

#if DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm90_int8_asym_gemm_1d1d.hpp"
// hybridGEMM.md Phase A/B kernels — not yet in the tree; compile the deep/hybrid
// facades only once the headers land.
#if __has_include("../jit_kernels/impls/sm90_int8_gemm.hpp") && \
    __has_include("../jit_kernels/impls/sm90_int8_hybrid_gemm.hpp")
#define DG_HAS_SM90_INT8_DEEP_HYBRID 1
#include "../jit_kernels/impls/sm90_int8_gemm.hpp"
#include "../jit_kernels/impls/sm90_int8_hybrid_gemm.hpp"
#else
#define DG_HAS_SM90_INT8_DEEP_HYBRID 0
#endif
#endif

#ifndef DG_HAS_SM90_INT8_DEEP_HYBRID
#define DG_HAS_SM90_INT8_DEEP_HYBRID 0
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

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
static void m_grouped_fp8_asym_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const torch::Tensor& offsets, const torch::Tensor& experts,
                                             const int& list_size,
                                             std::optional<std::tuple<int, int, int>> recipe,
                                             const std::string& compiled_dims,
                                             const bool& disable_ue8m0_cast) {
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

    // Architecture dispatch.
    const auto& [arch_major, arch_minor] = device_runtime->get_arch_pair();

    // SM89 (Ada): native FP8 MoE kernel. It consumes the *original* (untransformed) scales,
    // so it must run before the SF layout transform below. SM80/SM86 lack FP8 tensor cores
    // and are not supported.
    if (arch_major == 8 and arch_minor == 9) {
        if (b.second.dim() == 3) {
            DG_HOST_ASSERT(b.second.scalar_type() == torch::kFloat32);
            m_grouped_fp8_asym_gemm_sm89(a.first, b.first, d, offsets, experts, list_size,
                                         1.0f, 1.0f, std::nullopt, std::nullopt,
                                         a.second.contiguous(), b.second.contiguous());
        } else {
            m_grouped_fp8_asym_gemm_sm89(a.first, b.first, d, offsets, experts, list_size,
                                         1.0f, 1.0f, a.second.reshape(-1).contiguous(), b.second.contiguous(),
                                         std::nullopt, std::nullopt);
        }
        return;
    }

    // SM90 (Hopper) / SM100 (Blackwell): native asym kernels consume SFA/SFB in the
    // transformed compute layout. Grid Y = num_groups; sentinel blocks early-exit in-kernel.
    if (arch_major == 9 or arch_major == 10) {
        if (not recipe.has_value())
            recipe = get_default_recipe(a.second.scalar_type(), b.second.scalar_type());
        const auto& sfa = layout::transform_sf_into_required_layout(a.second, m, k, recipe.value(), std::nullopt, true, disable_ue8m0_cast);
        const auto& sfb = layout::transform_sf_into_required_layout(b.second, n, k, recipe.value(), num_groups, false, disable_ue8m0_cast);
        if (arch_major == 9)
            sm90_m_grouped_fp8_asym_gemm_contiguous_1d1d(a.first, sfa, b.first, sfb, d,
                                                         offsets, experts, /*grid_y=*/num_groups,
                                                         num_groups, m, n, k, major_a, major_b, compiled_dims);
        else
            sm100_m_grouped_fp8_asym_gemm_contiguous_1d1d(a.first, sfa, b.first, sfb, d,
                                                          offsets, experts, /*grid_y=*/num_groups,
                                                          num_groups, m, n, k, major_a, major_b, compiled_dims);
        return;
    }

    DG_HOST_UNREACHABLE("FP8 contiguous asym GEMM requires SM89, SM90, or SM100 "
                        "(SM80/SM86 lack FP8 tensor cores)");
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

    // Architecture dispatch.
    const auto& [arch_major, arch_minor] = device_runtime->get_arch_pair();

    // SM89 (Ada): native FP8 MoE kernel. It consumes the *original* (untransformed) scales,
    // so it must run before the SF layout transform below. SM80/SM86 lack FP8 tensor cores
    // and are not supported.
    if (arch_major == 8 and arch_minor == 9) {
        if (b.second.dim() == 3) {
            DG_HOST_ASSERT(b.second.scalar_type() == torch::kFloat32);
            m_grouped_fp8_asym_gemm_sm89_masked(a.first, b.first, d, masked_m, expected_m,
                                                1.0f, 1.0f, std::nullopt, std::nullopt,
                                                a.second.contiguous(), b.second.contiguous());
        } else {
            m_grouped_fp8_asym_gemm_sm89_masked(a.first, b.first, d, masked_m, expected_m,
                                                1.0f, 1.0f, a.second.reshape(-1).contiguous(), b.second.contiguous(),
                                                std::nullopt, std::nullopt);
        }
        return;
    }

    // SM90 (Hopper) / SM100 (Blackwell): native asym kernels consume SFA/SFB in the
    // transformed compute layout. Launch with gridDim.y == num_groups (constant) — CUDA-graph safe.
    if (arch_major == 9 or arch_major == 10) {
        if (not recipe.has_value())
            recipe = get_default_recipe(a.second.scalar_type(), b.second.scalar_type());
        const auto& sfa = layout::transform_sf_into_required_layout(a.second, m, k, recipe.value(), num_groups, true, disable_ue8m0_cast);
        const auto& sfb = layout::transform_sf_into_required_layout(b.second, n, k, recipe.value(), num_groups, false, disable_ue8m0_cast);
        if (arch_major == 9)
            sm90_m_grouped_fp8_asym_gemm_masked_1d1d(a.first, sfa, b.first, sfb, d, masked_m, expected_m,
                                                     num_groups, m, n, k, major_a, major_b, compiled_dims);
        else
            sm100_m_grouped_fp8_asym_gemm_masked_1d1d(a.first, sfa, b.first, sfb, d, masked_m, expected_m,
                                                      num_groups, m, n, k, major_a, major_b, compiled_dims);
        return;
    }

    DG_HOST_UNREACHABLE("FP8 masked asym GEMM requires SM89, SM90, or SM100 "
                        "(SM80/SM86 lack FP8 tensor cores)");
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
                                              const int& list_size,
                                              const std::string& compiled_dims) {
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
    if (arch_major == 9) {
        sm90_m_grouped_bf16_asym_gemm_contiguous(a, b, d,
                                                 offsets, experts, /*grid_y=*/num_groups,
                                                 num_groups, m, n, k, major_a, major_b, compiled_dims);
        return;
    }
    if (arch_major == 10) {
        sm100_m_grouped_bf16_asym_gemm_contiguous(a, b, d,
                                                offsets, experts, /*grid_y=*/num_groups,
                                                num_groups, m, n, k, major_a, major_b, compiled_dims);
        return;
    }
    DG_HOST_UNREACHABLE("BF16 contiguous asym GEMM is not supported on this architecture "
                        "(supported: SM90, SM100)");
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
    // Hopper (SM90): native WGMMA asym kernel (sm90_bf16_asym_gemm.cuh).
    if (arch_major == 9) {
        sm90_m_grouped_bf16_asym_gemm_masked(a, b, d, masked_m, expected_m,
                                             num_groups, m, n, k, major_a, major_b, compiled_dims);
        return;
    }
    // Ada (SM89): arch-aware SM80-style grouped MoE kernel. It flattens the padded
    // [G, M_max, K] masked layout and skips padding rows in-kernel, all via GPU tensor
    // ops so it is CUDA-graph capturable.
    if (arch_major == 8 and arch_minor == 9) {
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
                        "(supported: SM89, SM90, SM100)");
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

#if DG_TENSORMAP_COMPATIBLE
// -----------------------------------------------------------------------------
// Architecture-agnostic INT8 asym GEMM facades.
//
// Callers pass `(int8_data, fp32_scale)` pairs with scales in the *natural*
// per-token / per-channel layout. The facade checks the device architecture and
// routes to the arch-specific downlevel kernel, transforming the scale factors
// into the K-major layout that kernel expects. Only SM90 (Hopper/H20) is wired
// up today; new architectures add another branch here.
// -----------------------------------------------------------------------------
static void m_grouped_int8_asym_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                                   const std::pair<torch::Tensor, torch::Tensor>& b,
                                                   const torch::Tensor& d,
                                                   const torch::Tensor& offsets, const torch::Tensor& experts,
                                                   const int& list_size,
                                                   std::optional<std::tuple<int, int, int>> recipe,
                                                   const std::string& compiled_dims) {
    // recipe/compiled_dims accepted for API parity with the other dtypes; INT8
    // uses a fixed (1,1,128) block recipe and "nk" compiled dims.
    (void)recipe;
    (void)compiled_dims;

    const auto& a_data = a.first;
    const auto& sfa    = a.second;   // [M, Kb]    per-token
    const auto& b_data = b.first;
    const auto& sfb    = b.second;   // [G, N, Kb] per-channel
    const int64_t num_groups = b_data.size(0);
    const int64_t n          = b_data.size(1);
    const int64_t kb         = sfa.size(-1);

    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        // SM90 kernel consumes K-major scales: sfa [M,Kb]->[Kb,M], sfb [G,N,Kb]->[Kb,G*N].
        const auto& sfa_k = sfa.transpose(0, 1).contiguous();
        const auto& sfb_k = sfb.permute({2, 0, 1}).reshape({kb, num_groups * n}).contiguous();
        m_grouped_int8_asym_gemm_sm90_contiguous(a_data, b_data, d, offsets, experts,
                                                 list_size, sfa_k, sfb_k);
        return;
    }
    // Ampere data-center (SM80/A100): K-outer cp.async kernel; consumes the
    // natural scale layouts directly (no K-major transform).
    if (arch_major == 8 and device_runtime->get_arch_pair().second == 0) {
        sm80_m_grouped_int8_asym_gemm_contiguous(a_data, b_data, d, offsets, experts,
                                                 list_size, sfa, sfb);
        return;
    }
    DG_HOST_UNREACHABLE("INT8 contiguous asym GEMM is not supported on this architecture "
                        "(supported: SM80, SM90)");
}

static void m_grouped_int8_asym_gemm_nt_masked(const std::pair<torch::Tensor, torch::Tensor>& a,
                                               const std::pair<torch::Tensor, torch::Tensor>& b,
                                               const torch::Tensor& d,
                                               const torch::Tensor& masked_m,
                                               const int& expected_m,
                                               std::optional<std::tuple<int, int, int>> recipe,
                                               const std::string& compiled_dims) {
    (void)recipe;
    (void)compiled_dims;

    const auto& a_data = a.first;
    const auto& sfa    = a.second;   // [G, M, Kb] per-token
    const auto& b_data = b.first;
    const auto& sfb    = b.second;   // [G, N, Kb] per-channel
    const int64_t num_groups = b_data.size(0);
    const int64_t n          = b_data.size(1);
    const int64_t kb         = sfa.size(-1);

    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        // SM90 kernel consumes K-major scales: sfa [G,M,Kb]->[G,Kb,M], sfb [G,N,Kb]->[Kb,G*N].
        const auto& sfa_k = sfa.transpose(1, 2).contiguous();
        const auto& sfb_k = sfb.permute({2, 0, 1}).reshape({kb, num_groups * n}).contiguous();
        m_grouped_int8_asym_gemm_sm90_masked(a_data, b_data, d, masked_m, expected_m, sfa_k, sfb_k);
        return;
    }
    // Ampere data-center (SM80/A100): natural scale layouts, no transform.
    if (arch_major == 8 and device_runtime->get_arch_pair().second == 0) {
        (void)expected_m;  // grid is constant; expected_m is an SM90 tuning hint
        sm80_m_grouped_int8_asym_gemm_masked(a_data, b_data, d, masked_m, sfa, sfb);
        return;
    }
    DG_HOST_UNREACHABLE("INT8 masked asym GEMM is not supported on this architecture "
                        "(supported: SM80, SM90)");
}

// Deep-pattern INT8 grouped GEMM for HBM-resident expert weights
// (hybridGEMM.md Phase A / unified_kernel_sm80.md Phase 3). Same calling
// convention as the asym contiguous facade; routes to a persistent/M-outer
// kernel instead of the K-outer PCIe-oriented one. B must be HBM-resident.
static void m_grouped_int8_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                              const std::pair<torch::Tensor, torch::Tensor>& b,
                                              const torch::Tensor& d,
                                              const torch::Tensor& offsets, const torch::Tensor& experts,
                                              const int& list_size,
                                              std::optional<std::tuple<int, int, int>> recipe,
                                              const std::string& compiled_dims) {
    (void)recipe;
    (void)compiled_dims;

    const auto& a_data = a.first;
    const auto& sfa    = a.second;   // [M, Kb]    per-token
    const auto& b_data = b.first;
    const auto& sfb    = b.second;   // [G, N, Kb] per-channel
    const int64_t num_groups = b_data.size(0);
    const int64_t n          = b_data.size(1);
    const int64_t kb         = sfa.size(-1);

    const auto& arch_major = device_runtime->get_arch_major();
#if DG_HAS_SM90_INT8_DEEP_HYBRID
    if (arch_major == 9) {
        // Kernel reads K-block row 0 of the K-major scale layouts (scales are
        // constant along K): sfa [M,Kb]->[Kb,M], sfb [G,N,Kb]->[Kb,G*N].
        const auto& sfa_k = sfa.transpose(0, 1).contiguous();
        const auto& sfb_k = sfb.permute({2, 0, 1}).reshape({kb, num_groups * n}).contiguous();
        m_grouped_int8_deep_gemm_sm90_contiguous(a_data, b_data, d, offsets, experts,
                                                 list_size, sfa_k, sfb_k);
        return;
    }
#else
    (void)num_groups; (void)n; (void)kb;
#endif
    // Ampere data-center (SM80/A100): natural scale layouts, no transform.
    if (arch_major == 8 and device_runtime->get_arch_pair().second == 0) {
        sm80_m_grouped_int8_deep_gemm_contiguous(a_data, b_data, d, offsets, experts,
                                                 list_size, sfa, sfb);
        return;
    }
    DG_HOST_UNREACHABLE("INT8 deep grouped GEMM is not supported on this architecture "
                        "(supported: SM80, SM90)");
}

#if DG_HAS_SM90_INT8_DEEP_HYBRID
// Fused hybrid INT8 grouped GEMM (hybridGEMM.md Phase B): ONE launch computes
// host-resident segments with the asym K-outer pipeline on CTA ranks
// [0, s_host) and HBM-resident segments with the deep M-outer pipeline on the
// remaining ranks. Same per-side layout convention as the contiguous facades;
// the two segment lists must cover disjoint row ranges of a/d.
static void m_grouped_int8_hybrid_gemm_nt_contiguous(
        const std::pair<torch::Tensor, torch::Tensor>& a,       // (a [M,K] int8, sfa [M,Kb] fp32)
        const std::pair<torch::Tensor, torch::Tensor>& b_host,  // ([Gh,N,K] int8 pinned/cuda, sfb [Gh,N,Kb])
        const std::pair<torch::Tensor, torch::Tensor>& b_hbm,   // ([Gd,N,K] int8 cuda, sfb [Gd,N,Kb])
        const torch::Tensor& d,
        const torch::Tensor& offsets_host, const torch::Tensor& experts_host, const int& list_size_host,
        const torch::Tensor& offsets_hbm, const torch::Tensor& experts_hbm, const int& list_size_hbm,
        const int& s_host, const bool& enable_steal,
        std::optional<std::tuple<int, int, int>> recipe,
        const std::string& compiled_dims) {
    (void)recipe;
    (void)compiled_dims;

    const auto& a_data = a.first;
    const auto& sfa         = a.second;   // [M, Kb]     per-token
    const auto& b_host_data = b_host.first;
    const auto& sfb_host    = b_host.second;
    const auto& b_hbm_data  = b_hbm.first;
    const auto& sfb_hbm     = b_hbm.second;
    const int64_t gh = b_host_data.size(0);
    const int64_t gd = b_hbm_data.size(0);
    const int64_t n  = b_host_data.size(1);
    const int64_t kb = sfa.size(-1);

    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        // Pre-transpose scales K-major (the layout both parents consume):
        // sfa [M,Kb]->[Kb,M], sfb [G,N,Kb]->[Kb,G*N].
        const auto& sfa_k = sfa.transpose(0, 1).contiguous();
        const auto& sfb_host_k = sfb_host.permute({2, 0, 1}).reshape({kb, gh * n}).contiguous();
        const auto& sfb_hbm_k = sfb_hbm.permute({2, 0, 1}).reshape({kb, gd * n}).contiguous();
        m_grouped_int8_hybrid_gemm_sm90_contiguous(
            a_data, b_host_data, b_hbm_data, d,
            offsets_host, experts_host, list_size_host,
            offsets_hbm, experts_hbm, list_size_hbm,
            s_host, enable_steal, sfa_k, sfb_host_k, sfb_hbm_k);
        return;
    }
    DG_HOST_UNREACHABLE("INT8 hybrid grouped GEMM is not supported on this architecture "
                        "(supported: SM90)");
}
#endif  // DG_HAS_SM90_INT8_DEEP_HYBRID

#endif

// ---- SFT integration: first-party asym-GEMM entry points (grafted onto v0.2.0) ----
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

static void check_list_size_tensor(const torch::Tensor& list_size_t) {
    DG_HOST_ASSERT(list_size_t.numel() == 1);
    DG_HOST_ASSERT(list_size_t.scalar_type() == torch::kInt);
}

static void check_cpu_left_condition(const bool condition, const char* reason) {
    if (!condition)
        DG_HOST_UNREACHABLE(reason);
}

static torch::Tensor expand_sm90_asym_sfb(const torch::Tensor& sfb, const int& n) {
    DG_HOST_ASSERT(sfb.dim() == 3);
    return sfb.repeat_interleave(128, 1).narrow(1, 0, n).contiguous();
}

static void m_grouped_bf16_asym_gemm_nt_contiguous(const torch::Tensor& a, const torch::Tensor& b,
                                              const torch::Tensor& d,
                                              const torch::Tensor& offsets, const torch::Tensor& experts,
                                              const int& list_size,
                                              const std::string& compiled_dims,
                                              const bool transpose_b = false) {
    const auto& major_a = get_major_type_ab(a);
    cute::UMMA::Major major_b;
    if (transpose_b) {
        major_check(b);
        major_b = cute::UMMA::Major::MN;
    } else {
        major_b = get_major_type_ab(b);
    }
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);

    const auto& [m, k_a] = get_shape<2>(a);
    const auto& [num_groups, n_phys, k_phys] = get_shape<3>(b);
    const int n = transpose_b ? k_phys : n_phys;
    const int k = transpose_b ? n_phys : k_phys;
    const auto& [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_a);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(offsets.is_cuda() && experts.is_cuda());
    DG_HOST_ASSERT(offsets.is_contiguous() && experts.is_contiguous());
    DG_HOST_ASSERT(offsets.scalar_type() == torch::kInt && experts.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(offsets.numel() >= list_size && experts.numel() >= list_size);
    check_major_type_cd(d);
    if (m == 0)
        return;

    const int b_outer_stride = transpose_b
        ? static_cast<int>(b.stride(-2))
        : static_cast<int>(b.stride(get_non_contiguous_dim(major_b)));
    const int grid_y = list_size - 1;

    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_m_grouped_bf16_asym_gemm_contiguous(a, b, d,
                                                 offsets, experts, grid_y,
                                                 num_groups, m, n, k, major_a, major_b, compiled_dims,
                                                 b_outer_stride);
    } else if (arch_major == 10) {
        sm100_m_grouped_bf16_asym_gemm_contiguous(a, b, d,
                                                  offsets, experts, grid_y,
                                                  num_groups, m, n, k, major_a, major_b, compiled_dims,
                                                  b_outer_stride);
    } else {
        DG_HOST_ASSERT(false && "unsupported BF16 asym GEMM architecture");
    }
}

static void m_grouped_bf16_asym_gemm_nt_contiguous_ep_queued(const torch::Tensor& a, const torch::Tensor& b,
                                              const torch::Tensor& d,
                                              const torch::Tensor& offsets, const torch::Tensor& experts,
                                              const int& list_size,
                                              const torch::Tensor& ep_queue,
                                              const int& ep_side,
                                              const std::string& compiled_dims,
                                              const bool transpose_b = false) {
    // sEP (gb200_ep.md E3): queued variant of the contiguous grouped GEMM. ep_queue is a
    // PINNED HOST int32[>=3] counter block SHARED by both devices' launches; ep_side selects
    // front (0) or back (1) popping so cold segments stay device-local (affinity).
    const auto& major_a = get_major_type_ab(a);
    cute::UMMA::Major major_b;
    if (transpose_b) {
        major_check(b);
        major_b = cute::UMMA::Major::MN;
    } else {
        major_b = get_major_type_ab(b);
    }
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);

    const auto& [m, k_a] = get_shape<2>(a);
    const auto& [num_groups, n_phys, k_phys] = get_shape<3>(b);
    const int n = transpose_b ? k_phys : n_phys;
    const int k = transpose_b ? n_phys : k_phys;
    const auto& [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_a);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(offsets.is_cuda() && experts.is_cuda());
    DG_HOST_ASSERT(offsets.is_contiguous() && experts.is_contiguous());
    DG_HOST_ASSERT(offsets.scalar_type() == torch::kInt && experts.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(offsets.numel() >= list_size && experts.numel() >= list_size);
    DG_HOST_ASSERT(ep_queue.device().is_cpu() && ep_queue.is_pinned());
    DG_HOST_ASSERT(ep_queue.scalar_type() == torch::kInt && ep_queue.numel() >= 3);
    DG_HOST_ASSERT(ep_side == 0 or ep_side == 1);
    check_major_type_cd(d);
    if (m == 0)
        return;

    const int b_outer_stride = transpose_b
        ? static_cast<int>(b.stride(-2))
        : static_cast<int>(b.stride(get_non_contiguous_dim(major_b)));
    const int grid_y = list_size - 1;

    const auto& arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 10 && "sEP queued GEMM is SM100-only");
    sm100_m_grouped_bf16_asym_gemm_contiguous_ep_queued(a, b, d,
                                              offsets, experts, ep_queue, ep_side, grid_y,
                                              num_groups, m, n, k, major_a, major_b, compiled_dims,
                                              b_outer_stride);
}

static int m_grouped_bf16_asym_gemm_nt_contiguous_ep_steal(const torch::Tensor& a, const torch::Tensor& b,
                                              const torch::Tensor& d,
                                              const torch::Tensor& a_peer, const torch::Tensor& d_peer,
                                              const torch::Tensor& offsets, const torch::Tensor& experts,
                                              const int& list_size,
                                              const torch::Tensor& ep_queue,
                                              const int& ep_side,
                                              const int& ep_n_own,
                                              const std::string& compiled_dims,
                                              const bool transpose_b = false) {
    // sEP S2b (fix_gb200_ep.md): union queue + steal. The union list is
    // [side-0 segments | side-1 segments]; ep_n_own is the boundary in SEGMENT units.
    // a_peer = the PEER's packed X in the shared pinned fabric (sysmem TMA loads);
    // d_peer = THIS launch's fabric D staging for the items it steals (sysmem TMA
    // stores); the owner gathers exactly the stolen contiguous row range back.
    const auto& major_a = get_major_type_ab(a);
    cute::UMMA::Major major_b;
    if (transpose_b) {
        major_check(b);
        major_b = cute::UMMA::Major::MN;
    } else {
        major_b = get_major_type_ab(b);
    }
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);

    const auto& [m, k_a] = get_shape<2>(a);
    const auto& [num_groups, n_phys, k_phys] = get_shape<3>(b);
    const int n = transpose_b ? k_phys : n_phys;
    const int k = transpose_b ? n_phys : k_phys;
    const auto& [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_a);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(offsets.is_cuda() && experts.is_cuda());
    DG_HOST_ASSERT(offsets.is_contiguous() && experts.is_contiguous());
    DG_HOST_ASSERT(offsets.scalar_type() == torch::kInt && experts.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(offsets.numel() >= list_size && experts.numel() >= list_size);
    DG_HOST_ASSERT(ep_queue.device().is_cpu() && ep_queue.is_pinned());
    DG_HOST_ASSERT(ep_queue.scalar_type() == torch::kInt && ep_queue.numel() >= 3);
    DG_HOST_ASSERT(ep_side == 0 or ep_side == 1);
    DG_HOST_ASSERT(a_peer.device().is_cpu() && a_peer.is_pinned() && a_peer.is_contiguous());
    DG_HOST_ASSERT(d_peer.device().is_cpu() && d_peer.is_pinned() && d_peer.is_contiguous());
    DG_HOST_ASSERT(a_peer.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d_peer.scalar_type() == d.scalar_type());
    const auto& [m_peer, k_peer] = get_shape<2>(a_peer);
    const auto& [mp_, np_] = get_shape<2>(d_peer);
    DG_HOST_ASSERT(k_peer == k and mp_ == m_peer and np_ == n);
    DG_HOST_ASSERT(ep_n_own >= 0 and ep_n_own <= list_size - 1);
    check_major_type_cd(d);
    if (m == 0)
        return 0;

    const int b_outer_stride = transpose_b
        ? static_cast<int>(b.stride(-2))
        : static_cast<int>(b.stride(get_non_contiguous_dim(major_b)));
    const int grid_y = list_size - 1;

    const auto& arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 10 && "sEP steal GEMM is SM100-only");
    return sm100_m_grouped_bf16_asym_gemm_contiguous_ep_steal(a, b, d, a_peer, d_peer,
                                              offsets, experts, ep_queue, ep_side, ep_n_own, grid_y,
                                              num_groups, m, n, k, m_peer, major_a, major_b, compiled_dims,
                                              b_outer_stride);
}

static void sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous(
                                              const torch::Tensor& a,
                                              const torch::Tensor& b,
                                              const torch::Tensor& d,
                                              const torch::Tensor& offsets, const torch::Tensor& experts,
                                              const torch::Tensor& list_size_t,
                                              const std::string& compiled_dims,
                                              const int& compact_m_blocks = 0) {
    check_list_size_tensor(list_size_t);

    check_cpu_left_condition(device_runtime->get_arch_major() == 10, "requires_sm100");
    check_cpu_left_condition(a.device().is_cpu(), "input_not_cpu");
    check_cpu_left_condition(a.is_pinned(), "input_not_pinned");
    check_cpu_left_condition(b.is_cuda(), "weight_not_cuda");
    check_cpu_left_condition(d.is_cuda(), "output_not_cuda");
    check_cpu_left_condition(a.dim() == 2 && b.dim() == 3 && d.dim() == 2, "requires_2d_input_3d_weight");
    check_cpu_left_condition(a.is_contiguous() && b.is_contiguous() && d.is_contiguous(), "requires_contiguous");
    check_cpu_left_condition(a.scalar_type() == torch::kBFloat16 && b.scalar_type() == torch::kBFloat16,
                             "requires_bf16");
    check_cpu_left_condition(d.scalar_type() == torch::kBFloat16 || d.scalar_type() == torch::kFloat,
                             "requires_bf16_or_fp32_output");

    const auto& major_a = get_major_type_ab(a);
    const auto& major_b = get_major_type_ab(b);
    check_cpu_left_condition(major_a == cute::UMMA::Major::K && major_b == cute::UMMA::Major::K,
                             "requires_k_major_operands");
    check_major_type_cd(d);

    const auto& [m, k_a] = get_shape<2>(a);
    const auto& [num_groups, n, k] = get_shape<3>(b);
    const auto& [m_, n_] = get_shape<2>(d);
    check_cpu_left_condition(num_groups > 0, "requires_positive_groups");
    check_cpu_left_condition(n > 0 && k > 0, "requires_positive_nk");
    check_cpu_left_condition(m == m_ && n == n_ && k == k_a, "shape_mismatch");
    check_cpu_left_condition(n % 8 == 0 && k % 8 == 0, "requires_8_aligned_nk");

    check_cpu_left_condition(offsets.is_cuda() && experts.is_cuda(), "metadata_not_cuda");
    check_cpu_left_condition(offsets.is_contiguous() && experts.is_contiguous(), "metadata_not_contiguous");
    check_cpu_left_condition(offsets.scalar_type() == torch::kInt && experts.scalar_type() == torch::kInt,
                             "metadata_not_int32");
    const int grid_y = static_cast<int>(experts.numel()) - 1;
    check_cpu_left_condition(grid_y >= 0 && offsets.numel() >= 2 * grid_y, "metadata_mismatch");

    if (m == 0 || grid_y <= 0)
        return;

    sm100_m_grouped_bf16_cpu_left_asym_gemm_contiguous(a, b, d,
                                                       offsets, experts, grid_y,
                                                       num_groups, m, n, k,
                                                       major_a, major_b, compiled_dims,
                                                       static_cast<int>(b.stride(get_non_contiguous_dim(major_b))),
                                                       compact_m_blocks);
}

static void sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous(
                                              const torch::Tensor& a,
                                              const torch::Tensor& b,
                                              const torch::Tensor& d,
                                              const torch::Tensor& offsets, const torch::Tensor& experts,
                                              const int& list_size,
                                              const std::string& compiled_dims,
                                              const int& compact_m_blocks = 0) {
    check_cpu_left_condition(device_runtime->get_arch_major() == 10, "requires_sm100");
    check_cpu_left_condition(a.device().is_cpu(), "input_not_cpu");
    check_cpu_left_condition(a.is_pinned(), "input_not_pinned");
    check_cpu_left_condition(b.is_cuda(), "weight_not_cuda");
    check_cpu_left_condition(d.is_cuda(), "output_not_cuda");
    check_cpu_left_condition(a.dim() == 2 && b.dim() == 3 && d.dim() == 2, "requires_2d_input_3d_weight");
    check_cpu_left_condition(a.is_contiguous() && b.is_contiguous() && d.is_contiguous(), "requires_contiguous");
    check_cpu_left_condition(a.scalar_type() == torch::kBFloat16 && b.scalar_type() == torch::kBFloat16,
                             "requires_bf16");
    check_cpu_left_condition(d.scalar_type() == torch::kBFloat16 || d.scalar_type() == torch::kFloat,
                             "requires_bf16_or_fp32_output");

    const auto& major_a = get_major_type_ab(a);
    const auto& major_b = get_major_type_ab(b);
    check_cpu_left_condition(major_a == cute::UMMA::Major::K && major_b == cute::UMMA::Major::K,
                             "requires_k_major_operands");
    check_major_type_cd(d);

    const auto& [m, k_a] = get_shape<2>(a);
    const auto& [num_groups, n, k] = get_shape<3>(b);
    const auto& [m_, n_] = get_shape<2>(d);
    check_cpu_left_condition(num_groups > 0, "requires_positive_groups");
    check_cpu_left_condition(n > 0 && k > 0, "requires_positive_nk");
    check_cpu_left_condition(m == m_ && n == n_ && k == k_a, "shape_mismatch");
    check_cpu_left_condition(n % 8 == 0 && k % 8 == 0, "requires_8_aligned_nk");

    const int grid_y = list_size - 1;
    check_cpu_left_condition(list_size >= 1, "metadata_mismatch");
    check_cpu_left_condition(offsets.is_cuda() && experts.is_cuda(), "metadata_not_cuda");
    check_cpu_left_condition(offsets.is_contiguous() && experts.is_contiguous(), "metadata_not_contiguous");
    check_cpu_left_condition(offsets.scalar_type() == torch::kInt && experts.scalar_type() == torch::kInt,
                             "metadata_not_int32");
    check_cpu_left_condition(offsets.numel() >= 2 * grid_y && experts.numel() >= list_size,
                             "metadata_mismatch");

    if (m == 0 || grid_y <= 0)
        return;

    sm100_m_grouped_bf16_cpu_left_asym_gemm_contiguous(a, b, d,
                                                       offsets, experts, grid_y,
                                                       num_groups, m, n, k,
                                                       major_a, major_b, compiled_dims,
                                                       static_cast<int>(b.stride(get_non_contiguous_dim(major_b))),
                                                       compact_m_blocks);
}

static void sm100_m_grouped_bf16_cpu_left_pair_asym_gemm_nt_contiguous(
                                              const torch::Tensor& a,
                                              const torch::Tensor& b_gate,
                                              const torch::Tensor& b_up,
                                              const torch::Tensor& d_gate,
                                              const torch::Tensor& d_up,
                                              const torch::Tensor& offsets, const torch::Tensor& experts,
                                              const int& list_size,
                                              const std::string& compiled_dims,
                                              const int& compact_m_blocks = 0) {
    check_cpu_left_condition(device_runtime->get_arch_major() == 10, "requires_sm100");
    check_cpu_left_condition(a.device().is_cpu(), "input_not_cpu");
    check_cpu_left_condition(a.is_pinned(), "input_not_pinned");
    check_cpu_left_condition(b_gate.is_cuda() && b_up.is_cuda(), "weight_not_cuda");
    check_cpu_left_condition(d_gate.is_cuda() && d_up.is_cuda(), "output_not_cuda");
    check_cpu_left_condition(a.dim() == 2 && b_gate.dim() == 3 && b_up.dim() == 3 &&
                             d_gate.dim() == 2 && d_up.dim() == 2, "requires_2d_input_3d_weight");
    check_cpu_left_condition(a.is_contiguous() && b_gate.is_contiguous() && b_up.is_contiguous() &&
                             d_gate.is_contiguous() && d_up.is_contiguous(), "requires_contiguous");
    check_cpu_left_condition(a.scalar_type() == torch::kBFloat16 &&
                             b_gate.scalar_type() == torch::kBFloat16 &&
                             b_up.scalar_type() == torch::kBFloat16,
                             "requires_bf16");
    check_cpu_left_condition(d_gate.scalar_type() == d_up.scalar_type(), "output_dtype_mismatch");
    check_cpu_left_condition(d_gate.scalar_type() == torch::kBFloat16 || d_gate.scalar_type() == torch::kFloat,
                             "requires_bf16_or_fp32_output");

    const auto& major_a = get_major_type_ab(a);
    const auto& major_b = get_major_type_ab(b_gate);
    check_cpu_left_condition(major_a == cute::UMMA::Major::K && major_b == cute::UMMA::Major::K,
                             "requires_k_major_operands");
    check_cpu_left_condition(get_major_type_ab(b_up) == major_b, "requires_k_major_operands");
    check_major_type_cd(d_gate);
    check_major_type_cd(d_up);

    const auto& [m, k_a] = get_shape<2>(a);
    const auto& [num_groups, n, k] = get_shape<3>(b_gate);
    const auto& [num_groups_up, n_up, k_up] = get_shape<3>(b_up);
    const auto& [m_gate, n_gate] = get_shape<2>(d_gate);
    const auto& [m_up, n_up_out] = get_shape<2>(d_up);
    check_cpu_left_condition(num_groups > 0, "requires_positive_groups");
    check_cpu_left_condition(num_groups == num_groups_up && n == n_up && k == k_up, "shape_mismatch");
    check_cpu_left_condition(n > 0 && k > 0, "requires_positive_nk");
    check_cpu_left_condition(m == m_gate && m == m_up && n == n_gate && n == n_up_out && k == k_a, "shape_mismatch");
    check_cpu_left_condition(n % 8 == 0 && k % 8 == 0, "requires_8_aligned_nk");

    const int grid_y = list_size - 1;
    check_cpu_left_condition(list_size >= 1, "metadata_mismatch");
    check_cpu_left_condition(offsets.is_cuda() && experts.is_cuda(), "metadata_not_cuda");
    check_cpu_left_condition(offsets.is_contiguous() && experts.is_contiguous(), "metadata_not_contiguous");
    check_cpu_left_condition(offsets.scalar_type() == torch::kInt && experts.scalar_type() == torch::kInt,
                             "metadata_not_int32");
    check_cpu_left_condition(offsets.numel() >= 2 * grid_y && experts.numel() >= list_size,
                             "metadata_mismatch");

    if (m == 0 || grid_y <= 0)
        return;

    sm100_m_grouped_bf16_cpu_left_pair_asym_gemm_contiguous(a, b_gate, b_up, d_gate, d_up,
                                                            offsets, experts, grid_y,
                                                            num_groups, m, n, k,
                                                            major_a, major_b, compiled_dims,
                                                            static_cast<int>(b_gate.stride(get_non_contiguous_dim(major_b))),
                                                            compact_m_blocks);
}

static void register_apis(pybind11::module_& m) {

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    // FP8 GEMMs
    m.def("m_grouped_fp8_asym_gemm_nt_contiguous",
        static_cast<void(*)(const std::pair<torch::Tensor, torch::Tensor>&,
                            const std::pair<torch::Tensor, torch::Tensor>&,
                            const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const int&,
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
                              const torch::Tensor&, const torch::Tensor&, const int&, const std::string&)>(
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

    // Direct SM80 INT8 asym entry points (kernel guard is >= 800, so these also
    // run on SM90+ — used by tests to exercise the A100 code path on H100 boxes;
    // production code should go through m_grouped_int8_asym_gemm_nt_* instead).
    m.def("m_grouped_int8_asym_gemm_sm80_contiguous",
          &sm80_m_grouped_int8_asym_gemm_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("offsets"), py::arg("experts"), py::arg("list_size"),
          py::arg("sfa"), py::arg("sfb"));
    m.def("m_grouped_int8_asym_gemm_sm80_masked",
          &sm80_m_grouped_int8_asym_gemm_masked,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("masked_m"), py::arg("sfa"), py::arg("sfb"));

#if DG_TENSORMAP_COMPATIBLE
    // Architecture-agnostic INT8 asym GEMM (routes to SM90 today)
    m.def("m_grouped_int8_asym_gemm_nt_contiguous",
          static_cast<void(*)(const std::pair<torch::Tensor, torch::Tensor>&,
                              const std::pair<torch::Tensor, torch::Tensor>&,
                              const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                              const int&, std::optional<std::tuple<int, int, int>>, const std::string&)>(
              &m_grouped_int8_asym_gemm_nt_contiguous),
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("offsets"), py::arg("experts"), py::arg("list_size"),
          py::arg("recipe") = std::nullopt, py::arg("compiled_dims") = "nk");
    m.def("m_grouped_int8_asym_gemm_nt_masked",
          static_cast<void(*)(const std::pair<torch::Tensor, torch::Tensor>&,
                              const std::pair<torch::Tensor, torch::Tensor>&,
                              const torch::Tensor&, const torch::Tensor&, const int&,
                              std::optional<std::tuple<int, int, int>>, const std::string&)>(
              &m_grouped_int8_asym_gemm_nt_masked),
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("masked_m"), py::arg("expected_m"),
          py::arg("recipe") = std::nullopt, py::arg("compiled_dims") = "nk");
    m.def("m_grouped_int8_gemm_nt_contiguous",
          static_cast<void(*)(const std::pair<torch::Tensor, torch::Tensor>&,
                              const std::pair<torch::Tensor, torch::Tensor>&,
                              const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                              const int&, std::optional<std::tuple<int, int, int>>, const std::string&)>(
              &m_grouped_int8_gemm_nt_contiguous),
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("offsets"), py::arg("experts"), py::arg("list_size"),
          py::arg("recipe") = std::nullopt, py::arg("compiled_dims") = "nk");
#if DG_HAS_SM90_INT8_DEEP_HYBRID
    m.def("m_grouped_int8_hybrid_gemm_nt_contiguous",
          static_cast<void(*)(const std::pair<torch::Tensor, torch::Tensor>&,
                              const std::pair<torch::Tensor, torch::Tensor>&,
                              const std::pair<torch::Tensor, torch::Tensor>&,
                              const torch::Tensor&,
                              const torch::Tensor&, const torch::Tensor&, const int&,
                              const torch::Tensor&, const torch::Tensor&, const int&,
                              const int&, const bool&,
                              std::optional<std::tuple<int, int, int>>, const std::string&)>(
              &m_grouped_int8_hybrid_gemm_nt_contiguous),
          py::arg("a"), py::arg("b_host"), py::arg("b_hbm"), py::arg("d"),
          py::arg("offsets_host"), py::arg("experts_host"), py::arg("list_size_host"),
          py::arg("offsets_hbm"), py::arg("experts_hbm"), py::arg("list_size_hbm"),
          py::arg("s_host"), py::arg("enable_steal") = false,
          py::arg("recipe") = std::nullopt, py::arg("compiled_dims") = "nk");
#endif  // DG_HAS_SM90_INT8_DEEP_HYBRID

#endif

    // SM89 FP8 MoE GEMM helpers are now internal-only (dispatched from the
    // architecture-agnostic m_grouped_fp8_asym_gemm_nt_* APIs); not exported to Python.

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
    // ---- SFT integration bindings ----
    m.def("m_grouped_bf16_asym_gemm_nt_contiguous",
          static_cast<void(*)(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                              const torch::Tensor&, const torch::Tensor&, const int&,
                              const std::string&, const bool)>(
              &m_grouped_bf16_asym_gemm_nt_contiguous),
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("offsets"), py::arg("experts"), py::arg("list_size"),
          py::arg("compiled_dims") = "nk",
          py::arg("transpose_b") = false);
    m.def("m_grouped_bf16_asym_gemm_nt_contiguous_ep_queued",
          static_cast<void(*)(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                              const torch::Tensor&, const torch::Tensor&, const int&,
                              const torch::Tensor&, const int&,
                              const std::string&, const bool)>(
              &m_grouped_bf16_asym_gemm_nt_contiguous_ep_queued),
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("offsets"), py::arg("experts"), py::arg("list_size"),
          py::arg("ep_queue"), py::arg("ep_side"),
          py::arg("compiled_dims") = "nk",
          py::arg("transpose_b") = false);
    m.def("m_grouped_bf16_asym_gemm_nt_contiguous_ep_steal",
          static_cast<int(*)(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                             const torch::Tensor&, const torch::Tensor&,
                             const torch::Tensor&, const torch::Tensor&, const int&,
                             const torch::Tensor&, const int&, const int&,
                             const std::string&, const bool)>(
              &m_grouped_bf16_asym_gemm_nt_contiguous_ep_steal),
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("a_peer"), py::arg("d_peer"),
          py::arg("offsets"), py::arg("experts"), py::arg("list_size"),
          py::arg("ep_queue"), py::arg("ep_side"), py::arg("ep_n_own"),
          py::arg("compiled_dims") = "nk",
          py::arg("transpose_b") = false);
    m.def("sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous",
          static_cast<void(*)(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                              const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                              const std::string&, const int&)>(
              &sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous),
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("offsets"), py::arg("experts"), py::arg("list_size"),
          py::arg("compiled_dims") = "nk",
          py::arg("compact_m_blocks") = 0);
    m.def("sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous",
          static_cast<void(*)(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                              const torch::Tensor&, const torch::Tensor&, const int&,
                              const std::string&, const int&)>(
              &sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous),
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("offsets"), py::arg("experts"), py::arg("list_size"),
          py::arg("compiled_dims") = "nk",
          py::arg("compact_m_blocks") = 0);
    m.def("sm100_m_grouped_bf16_cpu_left_pair_asym_gemm_nt_contiguous",
          &sm100_m_grouped_bf16_cpu_left_pair_asym_gemm_nt_contiguous,
          py::arg("a"), py::arg("b_gate"), py::arg("b_up"),
          py::arg("d_gate"), py::arg("d_up"),
          py::arg("offsets"), py::arg("experts"), py::arg("list_size"),
          py::arg("compiled_dims") = "nk",
          py::arg("compact_m_blocks") = 0);
}

} // namespace asym_gemm::gemm
