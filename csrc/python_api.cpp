// Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
// Modified by Bytedance Inc., 2026.
// Original: https://github.com/deepseek-ai/DeepGEMM

#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "apis/gemm.hpp"
// #include "apis/asym_gemm.hpp"
#include "apis/dropout.hpp"
#include "apis/exp_act_offload.hpp"
#include "apis/layout.hpp"
#include "apis/qwen3_moe.hpp"
#include "apis/runtime.hpp"

#include "qwen3/qwen3_moe_routed_gemm.cpp"

// sEP S2b completion/gather kernels (csrc/ep_steal/ep_steal_sync.cu)
namespace asym_gemm::ep_steal {
void register_apis(pybind11::module_& m);
}

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepGEMM C++ library";
    asym_gemm::dropout::register_apis(m);
    asym_gemm::ep_steal::register_apis(m);
    asym_gemm::exp_act_offload::register_apis(m);
    asym_gemm::gemm::register_apis(m);
    asym_gemm::layout::register_apis(m);
    asym_gemm::qwen3_moe::register_apis(m);
    asym_gemm::runtime::register_apis(m);
}
