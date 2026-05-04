// Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
// Modified by Bytedance Inc., 2026.
// Original: https://github.com/deepseek-ai/DeepGEMM

#pragma once

#include <functional>
#include <memory>

#define DG_DECLARE_STATIC_VAR_IN_CLASS(cls, name) decltype(cls::name) cls::name

namespace asym_gemm {

template <typename T>
class LazyInit {
public:
    explicit LazyInit(std::function<std::shared_ptr<T>()> factory)
        : factory(std::move(factory)) {}

    T* operator -> () {
        if (ptr == nullptr)
            ptr = factory();
        return ptr.get();
    }

private:
    std::shared_ptr<T> ptr;
    std::function<std::shared_ptr<T>()> factory;
};

} // namespace asym_gemm
