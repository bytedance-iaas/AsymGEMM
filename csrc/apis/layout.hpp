#pragma once

#include "../utils/layout.hpp"
#include "../jit_kernels/impls/smxx_layout.hpp"

namespace asym_gemm::layout {

// Pack byte-typed SF tensors (e.g. FP8-E4M3) into uint32 packs and convert to
// MN-major, TMA-aligned layout required by FP4 SF TMA loads.
static torch::Tensor get_mn_major_tma_aligned_packed_byte_sf_tensor(const torch::Tensor& sf) {
    DG_HOST_ASSERT(sf.scalar_type() == torch::kFloat8_e4m3fn or sf.scalar_type() == torch::kUInt8);
    const auto sf_reshaped = (sf.dim() == 2) ? sf.unsqueeze(0) : sf;
    const auto [num_groups, mn, k] = get_shape<3>(sf_reshaped);

    const auto sf_bytes = sf_reshaped.view(torch::kUInt8);

    const int aligned_mn = get_tma_aligned_size(mn, 4);
    const int aligned_k = align(k, 4);

    auto padded_u8 = torch::zeros({num_groups, aligned_mn, aligned_k},
                                  at::TensorOptions().device(sf.device()).dtype(torch::kUInt8));
    padded_u8.slice(1, 0, mn).slice(2, 0, k).copy_(sf_bytes);

    auto packed_i32 = padded_u8.view(-1).view(torch::kInt32).view({num_groups, aligned_mn, aligned_k / 4});
    auto out = torch::empty_strided({num_groups, aligned_mn, aligned_k / 4},
                                    {aligned_mn * (aligned_k / 4), 1, aligned_mn},
                                    at::TensorOptions().device(sf.device()).dtype(torch::kInt32));
    out.copy_(packed_i32);
    out = out.slice(1, 0, mn);
    return (sf.dim() == 2) ? out.squeeze(0) : out;
}

static torch::Tensor transform_sf_into_required_layout(const torch::Tensor& sf,
                                                       const int& mn, const int& k,
                                                       const std::tuple<int, int, int>& recipe,
                                                       const std::optional<int>& num_groups,
                                                       const bool& is_sfa,
                                                       const bool& disable_ue8m0_cast) {
    const auto& gran_mn = is_sfa ? std::get<0>(recipe) : std::get<1>(recipe);
    const auto& gran_k = std::get<2>(recipe);
    const auto& arch_major = device_runtime->get_arch_major();

    // Pre-transform checks
    check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups);

    // (FP32, 1, 128/256) on SM90: transform to TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kFloat and gran_mn == 1 and (gran_k == 128 or gran_k == 256) and (arch_major == 9 or disable_ue8m0_cast))
        return get_mn_major_tma_aligned_tensor(sf);

    // (FP32, 1, 128/256) on SM100: transform to packed UE8M0, TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kFloat and gran_mn == 1 and (gran_k == 128 or gran_k == 256) and arch_major == 10) {
        DG_HOST_ASSERT(not disable_ue8m0_cast);
        return get_mn_major_tma_aligned_packed_ue8m0_tensor(sf);
    }

    // (FP32, 128, 128/256) on SM90: no need to transform, check SFB requirements
    if (sf.scalar_type() == torch::kFloat and gran_mn == 128 and (gran_k == 128 or gran_k == 256) and (arch_major == 9 or disable_ue8m0_cast))
        return check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups, false, true, torch::kFloat);

    // (FP32, 128, 128/256) on SM100: broadcast to (FP32, 1, gran_k), then pack to UE8M0
    if (sf.scalar_type() == torch::kFloat and gran_mn == 128 and (gran_k == 128 or gran_k == 256) and arch_major == 10) {
        DG_HOST_ASSERT(not disable_ue8m0_cast);
        const auto& broadcasted = sf.index_select(-2, torch::arange(mn, at::TensorOptions().device(sf.device())).floor_divide_(128));
        return get_mn_major_tma_aligned_packed_ue8m0_tensor(broadcasted);
    }

    // (INT, 1, 128/256) on SM100: transform to TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kInt and gran_mn == 1 and (gran_k == 128 or gran_k == 256) and arch_major == 10)
        return check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups, true, false, torch::kInt);

    // (FP32, 1/16, 16) on SM100 with UE8M0 disabled: FP4 native SF layout.
    if (sf.scalar_type() == torch::kFloat and (gran_mn == 1 or gran_mn == 16) and gran_k == 16 and arch_major == 10 and disable_ue8m0_cast)
        return get_mn_major_tma_aligned_tensor(sf);

    // (E4M3, 1/16, 16) on SM100 FP4 kernels: repack bytes to MN-major TMA-aligned uint32 layout.
    if (sf.scalar_type() == torch::kFloat8_e4m3fn and (gran_mn == 1 or gran_mn == 16) and gran_k == 16 and arch_major == 10)
        return get_mn_major_tma_aligned_packed_byte_sf_tensor(sf);

    DG_HOST_UNREACHABLE("Unknown SF transformation");
}

static torch::Tensor transform_k_grouped_sf_into_required_layout(const torch::Tensor& sf,
                                                                 const std::vector<int>& ks,
                                                                 const torch::Tensor& ks_tensor,
                                                                 const std::tuple<int, int, int>& recipe) {
    DG_HOST_ASSERT(sf.dim() == 2);
    DG_HOST_ASSERT(recipe == std::make_tuple(1, 1, 128));
    const auto& arch_major = device_runtime->get_arch_major();

    // FP32 on SM90
    if (sf.scalar_type() == torch::kFloat and arch_major == 9)
        return get_mn_major_tma_aligned_tensor(sf);

    // FP32 on SM100
    if (sf.scalar_type() == torch::kFloat and arch_major == 10)
        return get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, ks_tensor, ks);

    // INT on SM100
    if (sf.scalar_type() == torch::kInt and arch_major == 10)
        DG_HOST_UNREACHABLE("Unimplemented");

    DG_HOST_UNREACHABLE("Unknown cases");
}

static void register_apis(pybind11::module_& m) {
    m.def("transform_sf_into_required_layout", &transform_sf_into_required_layout,
      py::arg("sf"), py::arg("mn"), py::arg("k"), py::arg("recipe"),
      py::arg("num_groups") = std::nullopt, py::arg("is_sfa") = false,
      py::arg("disable_ue8m0_cast") = false);

    m.def("get_tma_aligned_size", &get_tma_aligned_size);
    m.def("get_mk_alignment_for_contiguous_layout", &get_mk_alignment_for_contiguous_layout);
    m.def("get_mn_major_tma_aligned_tensor", &get_mn_major_tma_aligned_tensor);
    m.def("get_mn_major_tma_aligned_packed_ue8m0_tensor", &get_mn_major_tma_aligned_packed_ue8m0_tensor);
    m.def("get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor", &get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor);
}

} // namespace asym_gemm::layout
