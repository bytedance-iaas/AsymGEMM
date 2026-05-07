#include <pybind11/pybind11.h>
#include <torch/python.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "apis/gemm.hpp"
// #include "apis/asym_gemm.hpp"
#include "apis/layout.hpp"
#include "apis/runtime.hpp"

// C-linkage launcher compiled in csrc/mega_moe_launch.cu (see asym_moe_update.md).
// Entry point for the fused single-launch MoE kernel.
extern "C" int asym_mega_moe_launch(
        const void* a_fp8, const void* a_sf,
        const void* l1_w, const void* l1_w_sf,
        const void* l2_w, const void* l2_w_sf,
        const void* m_indices,
        const void* topk_map,
        const void* row_topk_w,
        void* l1_out,
        void* l2_acts,
        void* l2_sf,
        void* combine_buf,
        void* y,
        void* grid_sync_ctrs,
        uint32_t M_total,
        uint32_t num_tokens,
        uint32_t num_topk,
        uint32_t hidden,
        uint32_t intermediate,
        float clamp,
        int fast_math,
        cudaStream_t stream);

// C-linkage launcher for the new UMMA two-phase fused MoE kernel.
// Compiled in csrc/mega_moe_new_launch.cu
extern "C" int asym_mega_moe_new_launch(
        const void* a_fp8,    const void* a_sf,
        const void* l1_w,     const void* l1_w_sf,
        const void* l2_w,     const void* l2_w_sf,
        const void* offsets,
        const void* topk_map,
        const void* row_topk_w,
        void*       workspace_ptr,
        uint64_t    workspace_bytes,
        uint64_t    off_grid_sync,
        uint64_t    off_l1_arrival,
        uint64_t    off_l2_mask,
        uint64_t    off_l2_acts,
        uint64_t    off_l2_sf,
        uint64_t    off_token_src_map,
        uint64_t    off_l1_topk_w,
        uint64_t    off_combine,
        void*       y,
        uint32_t M_total, uint32_t num_tokens, uint32_t num_topk,
        uint32_t hidden, uint32_t intermediate, uint32_t num_experts,
        float    activation_clamp,
        int      fast_math,
        cudaStream_t stream);

static void fp8_asym_mega_moe_nt_contiguous(
        const torch::Tensor& a_fp8,
        const torch::Tensor& a_sf,
        const torch::Tensor& l1_w,
        const torch::Tensor& l1_w_sf,
        const torch::Tensor& l2_w,
        const torch::Tensor& l2_w_sf,
        const torch::Tensor& offsets,
        const torch::Tensor& topk_map,
        const torch::Tensor& row_topk_w,
        torch::Tensor workspace,
        torch::Tensor y,
        int64_t off_grid_sync,
        int64_t off_l1_arrival,
        int64_t off_l2_mask,
        int64_t off_l2_acts,
        int64_t off_l2_sf,
        int64_t off_token_src_map,
        int64_t off_l1_topk_w,
        int64_t off_combine,
        int64_t M_total,
        int64_t num_tokens,
        int64_t num_topk,
        int64_t hidden,
        int64_t intermediate,
        int64_t num_experts,
        double  activation_clamp,
        bool    fast_math) {
    int rc = asym_mega_moe_new_launch(
        a_fp8.data_ptr(),  a_sf.data_ptr(),
        l1_w.data_ptr(),   l1_w_sf.data_ptr(),
        l2_w.data_ptr(),   l2_w_sf.data_ptr(),
        offsets.data_ptr(),
        topk_map.data_ptr(),
        row_topk_w.data_ptr(),
        workspace.data_ptr(),
        (uint64_t)workspace.numel(),
        (uint64_t)off_grid_sync,
        (uint64_t)off_l1_arrival,
        (uint64_t)off_l2_mask,
        (uint64_t)off_l2_acts,
        (uint64_t)off_l2_sf,
        (uint64_t)off_token_src_map,
        (uint64_t)off_l1_topk_w,
        (uint64_t)off_combine,
        y.data_ptr(),
        (uint32_t)M_total, (uint32_t)num_tokens, (uint32_t)num_topk,
        (uint32_t)hidden, (uint32_t)intermediate, (uint32_t)num_experts,
        (float)activation_clamp,
        fast_math ? 1 : 0,
        c10::cuda::getCurrentCUDAStream());
    TORCH_CHECK(rc == 0, "fp8_asym_mega_moe_nt_contiguous launch failed rc=", rc);
}

static void fp8_asym_mega_moe_nt_contiguous_fused(
        const torch::Tensor& a_fp8, const torch::Tensor& a_sf,
        const torch::Tensor& l1_w, const torch::Tensor& l1_w_sf,
        const torch::Tensor& l2_w, const torch::Tensor& l2_w_sf,
        const torch::Tensor& m_indices,
        const torch::Tensor& topk_map,
        const torch::Tensor& row_topk_w,
        torch::Tensor l1_out,
        torch::Tensor l2_acts,
        torch::Tensor l2_sf,
        torch::Tensor combine_buf,
        torch::Tensor y,
        torch::Tensor grid_sync_ctrs,
        int64_t M_total, int64_t num_tokens, int64_t num_topk,
        int64_t hidden, int64_t intermediate,
        double clamp, bool fast_math) {
    int rc = asym_mega_moe_launch(
        a_fp8.data_ptr(), a_sf.data_ptr(),
        l1_w.data_ptr(), l1_w_sf.data_ptr(),
        l2_w.data_ptr(), l2_w_sf.data_ptr(),
        m_indices.data_ptr(),
        topk_map.data_ptr(),
        row_topk_w.data_ptr(),
        l1_out.data_ptr(),
        l2_acts.data_ptr(),
        l2_sf.data_ptr(),
        combine_buf.data_ptr(),
        y.data_ptr(),
        grid_sync_ctrs.data_ptr(),
        (uint32_t)M_total, (uint32_t)num_tokens, (uint32_t)num_topk,
        (uint32_t)hidden, (uint32_t)intermediate,
        (float)clamp, fast_math ? 1 : 0,
        c10::cuda::getCurrentCUDAStream());
    TORCH_CHECK(rc == 0, "fp8_asym_mega_moe_nt_contiguous_fused launch failed rc=", rc);
}

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepGEMM C++ library";
    asym_gemm::gemm::register_apis(m);
    asym_gemm::layout::register_apis(m);
    asym_gemm::runtime::register_apis(m);

    // UMMA two-phase fused MoE (sm100_fp8_asym_gemm_mega_moe.cuh)
    m.def("fp8_asym_mega_moe_nt_contiguous",
          &fp8_asym_mega_moe_nt_contiguous,
          pybind11::arg("a_fp8"), pybind11::arg("a_sf"),
          pybind11::arg("l1_w"),  pybind11::arg("l1_w_sf"),
          pybind11::arg("l2_w"),  pybind11::arg("l2_w_sf"),
          pybind11::arg("offsets"),
          pybind11::arg("topk_map"),
          pybind11::arg("row_topk_w"),
          pybind11::arg("workspace"),
          pybind11::arg("y"),
          pybind11::arg("off_grid_sync"),
          pybind11::arg("off_l1_arrival"),
          pybind11::arg("off_l2_mask"),
          pybind11::arg("off_l2_acts"),
          pybind11::arg("off_l2_sf"),
          pybind11::arg("off_token_src_map"),
          pybind11::arg("off_l1_topk_w"),
          pybind11::arg("off_combine"),
          pybind11::arg("M_total"),
          pybind11::arg("num_tokens"),
          pybind11::arg("num_topk"),
          pybind11::arg("hidden"),
          pybind11::arg("intermediate"),
          pybind11::arg("num_experts"),
          pybind11::arg("activation_clamp") = 0.0,
          pybind11::arg("fast_math") = true);

    // Fused single-launch MoE (asym_moe_update.md Stage 3+)
    m.def("fp8_asym_mega_moe_nt_contiguous_fused",
          &fp8_asym_mega_moe_nt_contiguous_fused,
          pybind11::arg("a_fp8"), pybind11::arg("a_sf"),
          pybind11::arg("l1_w"),  pybind11::arg("l1_w_sf"),
          pybind11::arg("l2_w"),  pybind11::arg("l2_w_sf"),
          pybind11::arg("m_indices"),
          pybind11::arg("topk_map"),
          pybind11::arg("row_topk_w"),
          pybind11::arg("l1_out"),
          pybind11::arg("l2_acts"),
          pybind11::arg("l2_sf"),
          pybind11::arg("combine_buf"),
          pybind11::arg("y"),
          pybind11::arg("grid_sync_ctrs"),
          pybind11::arg("M_total"),
          pybind11::arg("num_tokens"),
          pybind11::arg("num_topk"),
          pybind11::arg("hidden"),
          pybind11::arg("intermediate"),
          pybind11::arg("clamp"),
          pybind11::arg("fast_math"));
}
