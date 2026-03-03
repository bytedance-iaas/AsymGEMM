#pragma once

#include <algorithm>
#include <vector>

#include "../utils/compatibility.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm100_bf16_asym_gemm.hpp"
#include "../jit_kernels/impls/sm100_bf16_gemm.hpp"
#include "../jit_kernels/impls/sm100_fp8_asym_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm100_fp8_gemm_1d1d.hpp"
#endif

#include "../jit_kernels/impls/smxx_cublaslt.hpp"

#include "layout.hpp"

namespace asym_gemm::gemm {

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
static void m_grouped_fp8_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const torch::Tensor& m_indices,
                                             std::optional<std::tuple<int, int, int>> recipe,
                                             const std::string& compiled_dims,
                                             const bool& disable_ue8m0_cast) {
    // Shape must be `[M, K] @ [G, N, K].mT`
    const auto& major_a = get_major_type_ab(a.first);
    const auto& major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    if (fp8_requires_k_major())
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(m_indices.is_contiguous());

    // Type and shape checks
    const auto& [m, k] = get_shape<2>(a.first);
    const auto& [num_groups, n, k_] = get_shape<3>(b.first);
    const auto& [m_, n_] = get_shape<2>(d);
    const auto& m__ = static_cast<int>(m_indices.numel());
    DG_HOST_ASSERT(m == m_ and m == m__ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(m_indices.scalar_type() == torch::kInt);

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
    sm100_m_grouped_fp8_gemm_contiguous_1d1d(a.first, sfa, b.first, sfb, d, m_indices,
                                                 num_groups, m, n, k, major_a, major_b, compiled_dims);

}

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

static int fill_with_sentinel(
    int* m_indices, int M,
    int* offsets, int* experts, int capacity
) {
    if (!offsets || !experts || capacity <= 0)
        return 0;
    if (M <= 0 || !m_indices)
        return 0;

    int write = 0;
    auto maybe_emit = [&](int start_idx) {
        const int e = m_indices[start_idx];
        if (e != -1) {
            if (write < capacity) {
                offsets[write] = start_idx;
                experts[write] = e;
            }
            ++write;
        }
    };

    maybe_emit(0);
    for (int i = 1; i < M; ++i) {
        if (m_indices[i] != m_indices[i - 1])
            maybe_emit(i);
    }

    if (write < capacity) {
        offsets[write] = M;
        experts[write] = -1;
    }
    ++write;
    return std::min(write, capacity);
}

static void m_grouped_fp8_asym_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const torch::Tensor& m_indices,
                                             std::optional<std::tuple<int, int, int>> recipe,
                                             const std::string& compiled_dims,
                                             const bool& disable_ue8m0_cast) {
    DG_HOST_ASSERT(m_indices.is_contiguous());
    DG_HOST_ASSERT(m_indices.scalar_type() == torch::kInt);

    const auto m_indices_cpu = m_indices.to(torch::kCPU);
    auto* mi = m_indices_cpu.data_ptr<int>();
    const int max_len = static_cast<int>(b.first.size(0)) + 1;
    std::vector<int> offsets_h(max_len);
    std::vector<int> experts_h(max_len);
    const int list_size = fill_with_sentinel(mi, static_cast<int>(m_indices_cpu.numel()),
                                             offsets_h.data(), experts_h.data(), max_len);

    auto opts_i32_cuda = torch::TensorOptions().device(a.first.device()).dtype(torch::kInt32);
    auto offsets_t = torch::empty({max_len}, opts_i32_cuda);
    auto experts_t = torch::empty({max_len}, opts_i32_cuda);
    auto offsets_cpu_t = torch::from_blob(offsets_h.data(), {max_len}, torch::TensorOptions().dtype(torch::kInt32)).clone();
    auto experts_cpu_t = torch::from_blob(experts_h.data(), {max_len}, torch::TensorOptions().dtype(torch::kInt32)).clone();
    offsets_t.copy_(offsets_cpu_t, /*non_blocking=*/false);
    experts_t.copy_(experts_cpu_t, /*non_blocking=*/false);

    m_grouped_fp8_asym_gemm_nt_contiguous(a, b, d, offsets_t, experts_t, list_size, recipe, compiled_dims, disable_ue8m0_cast);
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
    DG_HOST_ASSERT(masked_m.is_contiguous());

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
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);

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
    DG_HOST_ASSERT(offsets.numel() >= list_size && experts.numel() >= list_size);

    // D must be N-major
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0)
        return;

    const auto& arch_major = device_runtime->get_arch_major();
    sm100_m_grouped_bf16_asym_gemm_contiguous(a, b, d,
                                            offsets, experts, list_size,
                                            num_groups, m, n, k, major_a, major_b, compiled_dims);
}

static void m_grouped_bf16_gemm_nt_contiguous(const torch::Tensor& a, const torch::Tensor& b,
                                              const torch::Tensor& d, const torch::Tensor& m_indices,
                                              const std::string& compiled_dims) {
    // Shape must be `[M, K] @ [G, N, K].mT`
    const auto& major_a = get_major_type_ab(a);
    const auto& major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    DG_HOST_ASSERT(m_indices.is_contiguous());

    // Type and shape checks
    const auto& [m, k] = get_shape<2>(a);
    const auto& [num_groups, n, k_] = get_shape<3>(b);
    const auto& [m_, n_] = get_shape<2>(d);
    const auto& m__ = static_cast<int>(m_indices.numel());
    DG_HOST_ASSERT(m == m_ and m == m__ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(m_indices.scalar_type() == torch::kInt);

    // D must be N-major
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0)
        return;

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    sm100_m_grouped_bf16_gemm_contiguous(a, b, d, m_indices,
                                            num_groups, m, n, k, major_a, major_b, compiled_dims);
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
    DG_HOST_ASSERT(masked_m.is_contiguous());

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
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);

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
    m.def("m_grouped_fp8_asym_gemm_nt_contiguous",
        static_cast<void(*)(const std::pair<torch::Tensor, torch::Tensor>&,
                            const std::pair<torch::Tensor, torch::Tensor>&,
                            const torch::Tensor&, const torch::Tensor&,
                            std::optional<std::tuple<int, int, int>>, const std::string&, const bool&)>(
            &m_grouped_fp8_asym_gemm_nt_contiguous),
        py::arg("a"), py::arg("b"), py::arg("d"), py::arg("m_indices"),
        py::arg("recipe") = std::nullopt, py::arg("compiled_dims") = "nk",
        py::arg("disable_ue8m0_cast") = false);
    m.def("m_grouped_fp8_gemm_nt_contiguous", &m_grouped_fp8_gemm_nt_contiguous,
        py::arg("a"), py::arg("b"), py::arg("d"), py::arg("m_indices"),
        py::arg("recipe") = std::nullopt, py::arg("compiled_dims") = "nk",
        py::arg("disable_ue8m0_cast") = false);
    m.def("m_grouped_fp8_asym_gemm_nt_masked", &m_grouped_fp8_asym_gemm_nt_masked,
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
    m.def("m_grouped_bf16_gemm_nt_contiguous", &m_grouped_bf16_gemm_nt_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("m_indices"),
          py::arg("compiled_dims") = "nk");
    m.def("m_grouped_bf16_asym_gemm_nt_masked", &m_grouped_bf16_asym_gemm_nt_masked,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("masked_m"), py::arg("expected_m"),
          py::arg("compiled_dims") = "nk");
#endif
}

} // namespace asym_gemm::gemm
