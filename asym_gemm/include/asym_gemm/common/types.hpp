// Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
// Modified by Bytedance Inc., 2026.
// Original: https://github.com/deepseek-ai/DeepGEMM

#pragma once

namespace asym_gemm {

enum class GemmType {
    Normal              = 0,
    MGroupedContiguous  = 1,
    MGroupedMasked      = 2,
    KGroupedContiguous  = 3,
    Batched             = 4
};

enum class KernelType {
    Kernel1D1D = 0,
    Kernel1D2D = 1,
    KernelNoSF = 2
};

} // namespace asym_gemm
