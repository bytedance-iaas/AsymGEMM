#pragma once

#include <algorithm>
#include <vector>

#include "../utils/compatibility.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm100_bf16_asym_gemm.hpp"
#include "../jit_kernels/impls/sm100_fp8_asym_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm100_fp8_asym_gemm_mega.hpp"
#include "../jit_kernels/impls/sm100_fp4_asym_gemm_1d1d.hpp"
#endif

#include "../jit_kernels/impls/smxx_cublaslt.hpp"

#include "layout.hpp"

namespace asym_gemm::gemm {

// PyTorch does not expose an FP4 dtype yet, so FP4 matrices are passed as
// packed bytes (two E2M1 elements per uint8). This matches NVFP4 E2M1 payload.
static void check_packed_fp4_e2m1_tensor(const torch::Tensor& t) {
    DG_HOST_ASSERT(t.scalar_type() == torch::kUInt8);
}

// Broadcast packed UE8M0 scale factors from coarse granularity (e.g., gran_k=128)
// to the fine granularity required by FP4 UMMA (sf_quant_k=16).
// Each UE8M0 byte is replicated `replication_factor` times (e.g., 128/16 = 8).
// Input:  sf with shape (..., packed_k_coarse) int32, MN-major layout
// Output: sf with shape (..., packed_k_fine)   int32, MN-major layout
static torch::Tensor broadcast_packed_ue8m0_sf(const torch::Tensor& sf,
                                               int replication_factor,
                                               int mn_size) {
    if (replication_factor <= 1) return sf;

    const int ndim = sf.dim();
    const int packed_k_coarse = sf.size(-1);
    const int packed_k_fine = packed_k_coarse * replication_factor;

    // Flatten to 2D for byte-level manipulation
    int64_t batch = 1;
    for (int i = 0; i < ndim - 1; ++i) batch *= sf.sizes()[i];

    // Force a truly row-major contiguous copy (MN-major tensors with size-1 last dim
    // may appear "contiguous" to PyTorch but have stride(-1) != 1)
    auto flat = torch::empty({batch, packed_k_coarse},
        at::TensorOptions().device(sf.device()).dtype(torch::kInt32));
    flat.copy_(sf.reshape({batch, packed_k_coarse}));

    // Reinterpret int32 as uint8: (batch, packed_k_coarse * 4)
    auto bytes = flat.view(torch::kUInt8).reshape({batch, packed_k_coarse * 4});

    // Replicate each byte: (batch, N_bytes) -> (batch, N_bytes, rep) -> (batch, N_bytes * rep)
    auto rep = bytes.unsqueeze(-1)
                    .expand({batch, packed_k_coarse * 4, static_cast<int64_t>(replication_factor)})
                    .contiguous()
                    .reshape({batch, packed_k_coarse * 4 * replication_factor});

    // Repack as int32: (batch, packed_k_fine)
    auto packed = rep.view(torch::kInt32).reshape({batch, packed_k_fine});

    // Restore original batch dimensions
    auto sizes = sf.sizes().vec();
    sizes.back() = packed_k_fine;
    packed = packed.reshape(sizes);

    // Create MN-major strided tensor matching TMA requirements
    const int tma_aligned_mn = get_tma_aligned_size(mn_size, 4);
    std::vector<int64_t> strides(ndim);
    strides[ndim - 2] = 1;                           // MN stride
    strides[ndim - 1] = tma_aligned_mn;              // packed_k stride
    if (ndim >= 3)
        strides[ndim - 3] = tma_aligned_mn * packed_k_fine;  // group stride

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
    DG_HOST_ASSERT(offsets.numel() >= list_size && experts.numel() >= list_size);

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

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    sm100_m_grouped_fp8_asym_gemm_contiguous_1d1d(a.first, sfa, b.first, sfb, d,
                                                 offsets, experts, list_size,
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

    // Dispatch implementation
    sm100_m_grouped_fp8_asym_gemm_masked_1d1d(a.first, sfa, b.first, sfb, d, masked_m, expected_m,
                                              num_groups, m, n, k, major_a, major_b, compiled_dims);
}

// mega variant (Phase 1+2: double-buffered B, dedicated warp roles)
static void m_grouped_fp8_asym_gemm_mega_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                                       const std::pair<torch::Tensor, torch::Tensor>& b,
                                                       const torch::Tensor& d,
                                                       const torch::Tensor& offsets, const torch::Tensor& experts,
                                                       const int& list_size,
                                                       std::optional<std::tuple<int, int, int>> recipe,
                                                       const std::string& compiled_dims,
                                                       const bool& disable_ue8m0_cast) {
    const auto& major_a = get_major_type_ab(a.first);
    const auto& major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    if (fp8_requires_k_major())
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
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
    DG_HOST_ASSERT(offsets.numel() >= list_size && experts.numel() >= list_size);
    check_major_type_cd(d);
    if (m == 0) return;
    if (not recipe.has_value())
        recipe = get_default_recipe(a.second.scalar_type(), b.second.scalar_type());
    const auto& sfa = layout::transform_sf_into_required_layout(a.second, m, k, recipe.value(), std::nullopt,  true, disable_ue8m0_cast);
    const auto& sfb = layout::transform_sf_into_required_layout(b.second, n, k, recipe.value(),   num_groups, false, disable_ue8m0_cast);
    sm100_m_grouped_fp8_asym_gemm_mega_contiguous(a.first, sfa, b.first, sfb, d,
                                                  offsets, experts, list_size,
                                                  num_groups, m, n, k, major_a, major_b, compiled_dims);
}

static void m_grouped_fp8_asym_gemm_mega_nt_masked(const std::pair<torch::Tensor, torch::Tensor>& a,
                                                   const std::pair<torch::Tensor, torch::Tensor>& b,
                                                   const torch::Tensor& d,
                                                   const torch::Tensor& masked_m,
                                                   const int& expected_m,
                                                   std::optional<std::tuple<int, int, int>> recipe,
                                                   const std::string& compiled_dims,
                                                   const bool& disable_ue8m0_cast) {
    const auto& major_a = get_major_type_ab(a.first);
    const auto& major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    if (fp8_requires_k_major())
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    const auto& [num_groups, m, k] = get_shape<3>(a.first);
    const auto& [num_groups_, n, k_] = get_shape<3>(b.first);
    const auto& [num_groups__, m_, n_] = get_shape<3>(d);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(num_groups == num_groups_ and num_groups == num_groups__);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.is_cuda() && masked_m.is_contiguous());
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(masked_m.numel() == num_groups);
    check_major_type_cd(d);
    if (m == 0 or expected_m == 0) return;
    if (not recipe.has_value())
        recipe = get_default_recipe(a.second.scalar_type(), b.second.scalar_type());
    const auto& sfa = layout::transform_sf_into_required_layout(a.second, m, k, recipe.value(), num_groups, true, disable_ue8m0_cast);
    const auto& sfb = layout::transform_sf_into_required_layout(b.second, n, k, recipe.value(), num_groups, false, disable_ue8m0_cast);
    sm100_m_grouped_fp8_asym_gemm_mega_masked(a.first, sfa, b.first, sfb, d, masked_m, expected_m,
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
    const auto& major_a = get_major_type_ab(a.first);
    const auto& major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);

    // FP4 packed: uint8 with 2 elements per byte, so shape has k_packed = k/2
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
    const auto& sfa_raw = layout::transform_sf_into_required_layout(a.second, m, k, recipe.value(), std::nullopt,  true, disable_ue8m0_cast);
    const auto& sfb_raw = layout::transform_sf_into_required_layout(b.second, n, k, recipe.value(),   num_groups, false, disable_ue8m0_cast);

    // Broadcast SFs from coarse granularity (gran_k=128) to FP4 UMMA granularity (sf_quant_k=16)
    constexpr int fp4_sf_quant_k = 16;
    const int sf_gran_k = std::get<2>(recipe.value());
    const int sf_replication = sf_gran_k / fp4_sf_quant_k;
    auto sfa = broadcast_packed_ue8m0_sf(sfa_raw, sf_replication, m);
    auto sfb = broadcast_packed_ue8m0_sf(sfb_raw, sf_replication, n);

    // Broadcast SFs along MN when gran_mn > 1 (SF has reduced MN dimension).
    // The kernel assumes SF MN dimension == data MN dimension.
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
    const auto& major_a = get_major_type_ab(a.first);
    const auto& major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);

    // FP4 packed: uint8 with 2 elements per byte, so shape has k_packed = k/2
    const auto& [num_groups, m, k_packed] = get_shape<3>(a.first);
    const auto& [num_groups_, n, k_packed_] = get_shape<3>(b.first);
    const int k = k_packed * 2;
    const auto& [num_groups__, m_, n_] = get_shape<3>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k_packed == k_packed_);
    check_packed_fp4_e2m1_tensor(a.first);
    check_packed_fp4_e2m1_tensor(b.first);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    // masked_m: int32 GPU tensor of shape [num_groups] with per-group token counts
    DG_HOST_ASSERT(masked_m.is_cuda() && masked_m.is_contiguous());
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(masked_m.numel() == num_groups);

    if (m == 0 or expected_m == 0) return;
    if (not recipe.has_value()) recipe = get_default_recipe(a.second.scalar_type(), b.second.scalar_type());
    const auto& sfa_raw = layout::transform_sf_into_required_layout(a.second, m, k, recipe.value(), num_groups, true, disable_ue8m0_cast);
    const auto& sfb_raw = layout::transform_sf_into_required_layout(b.second, n, k, recipe.value(), num_groups, false, disable_ue8m0_cast);

    // Broadcast SFs from coarse granularity (gran_k=128) to FP4 UMMA granularity (sf_quant_k=16)
    constexpr int fp4_sf_quant_k = 16;
    const int sf_gran_k = std::get<2>(recipe.value());
    const int sf_replication = sf_gran_k / fp4_sf_quant_k;
    auto sfa = broadcast_packed_ue8m0_sf(sfa_raw, sf_replication, m);
    auto sfb = broadcast_packed_ue8m0_sf(sfb_raw, sf_replication, n);

    // Broadcast SFs along MN when gran_mn > 1 (SF has reduced MN dimension).
    // The kernel assumes SF MN dimension == data MN dimension.
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
                                              const std::string& compiled_dims,
                                              const bool transpose_b = false) {
    // Shape must be `[M, K] @ [G, N, K].mT`
    // When transpose_b=true the physical b tensor is K-major [G, N_phys, K_phys]; we build an
    // MN-major TMA descriptor so the kernel reads it as the transposed matrix [G, K_phys, N_phys].
    const auto& major_a = get_major_type_ab(a);
    cute::UMMA::Major major_b;
    if (transpose_b) {
        major_check(b);                    // validate layout; physical tensor must be K-major
        major_b = cute::UMMA::Major::MN;   // override: build MN-major TMA descriptor
    } else {
        major_b = get_major_type_ab(b);    // auto-detect from stride(-1)
    }
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);

    // Shape extraction.
    // transpose_b=true: logical N = K_phys (b.size(2)), logical K = N_phys (b.size(1)).
    const auto& [m, k_a] = get_shape<2>(a);
    const auto& [num_groups, n_phys, k_phys] = get_shape<3>(b);
    const int n = transpose_b ? k_phys : n_phys;
    const int k = transpose_b ? n_phys : k_phys;
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    DG_HOST_ASSERT(offsets.is_cuda() && experts.is_cuda());
    DG_HOST_ASSERT(offsets.is_contiguous() && experts.is_contiguous());
    DG_HOST_ASSERT(offsets.scalar_type() == torch::kInt && experts.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(offsets.numel() >= list_size && experts.numel() >= list_size);

    // D must be N-major
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0)
        return;

    // TMA outer stride for B.
    // transpose_b=true: physical tensor is K-major [G, N_phys, K_phys] with b.stride(-2) = K_phys.
    // MN-major logical view [G, K_phys, N_phys]: outer (K_logical=N_phys) stride = K_phys = b.stride(-2).
    // This equals the forward (K-major) outer stride — the physical row width is the same.
    const int b_outer_stride = transpose_b
        ? static_cast<int>(b.stride(-2))
        : static_cast<int>(b.stride(get_non_contiguous_dim(major_b)));

    sm100_m_grouped_bf16_asym_gemm_contiguous(a, b, d,
                                            offsets, experts, list_size,
                                            num_groups, m, n, k, major_a, major_b, compiled_dims,
                                            b_outer_stride);
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

    // masked_m: int32 GPU tensor of shape [num_groups] with per-group token counts
    DG_HOST_ASSERT(masked_m.is_cuda() && masked_m.is_contiguous());
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(masked_m.numel() == num_groups);

    // D must be N-major (per-group)
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0 or expected_m == 0)
        return;

    // Dispatch implementation
    sm100_m_grouped_bf16_asym_gemm_masked(a, b, d, masked_m, expected_m,
                                          num_groups, m, n, k, major_a, major_b, compiled_dims);
}
#endif

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

    // FP8 mega GEMMs (Phase 1+2: double-buffered B, dedicated warp roles)
    m.def("m_grouped_fp8_asym_gemm_mega_nt_contiguous",
        static_cast<void(*)(const std::pair<torch::Tensor, torch::Tensor>&,
                            const std::pair<torch::Tensor, torch::Tensor>&,
                            const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const int&,
                            std::optional<std::tuple<int, int, int>>, const std::string&, const bool&)>(
            &m_grouped_fp8_asym_gemm_mega_nt_contiguous),
        py::arg("a"), py::arg("b"), py::arg("d"),
        py::arg("offsets"), py::arg("experts"), py::arg("list_size"),
        py::arg("recipe") = std::nullopt, py::arg("compiled_dims") = "nk",
        py::arg("disable_ue8m0_cast") = false);
    m.def("m_grouped_fp8_asym_gemm_mega_nt_masked",
        static_cast<void(*)(const std::pair<torch::Tensor, torch::Tensor>&,
                            const std::pair<torch::Tensor, torch::Tensor>&,
                            const torch::Tensor&, const torch::Tensor&, const int&,
                            std::optional<std::tuple<int, int, int>>, const std::string&, const bool&)>(
            &m_grouped_fp8_asym_gemm_mega_nt_masked),
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
                              const torch::Tensor&, const torch::Tensor&, const int&,
                              const std::string&, const bool)>(
              &m_grouped_bf16_asym_gemm_nt_contiguous),
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("offsets"), py::arg("experts"), py::arg("list_size"),
          py::arg("compiled_dims") = "nk",
          py::arg("transpose_b") = false);
    m.def("m_grouped_bf16_asym_gemm_nt_masked",
          static_cast<void(*)(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                              const torch::Tensor&, const int&, const std::string&)>(
              &m_grouped_bf16_asym_gemm_nt_masked),
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("masked_m"), py::arg("expected_m"),
          py::arg("compiled_dims") = "nk");
#endif
}

} // namespace asym_gemm::gemm
