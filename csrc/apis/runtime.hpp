// Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
// Modified by Bytedance Inc., 2026.
// Original: https://github.com/deepseek-ai/DeepGEMM

#pragma once

#include "../jit/compiler.hpp"
#include "../jit/device_runtime.hpp"

namespace asym_gemm::runtime {

// Global compile mode: 0 = execute (use cache), 1 = force compile
static int compile_mode = 0;

static void register_apis(pybind11::module_& m) {
    m.def("set_num_sms", [&](const int& new_num_sms) {
        device_runtime->set_num_sms(new_num_sms);
    });
    m.def("get_num_sms", [&]() {
       return device_runtime->get_num_sms();
    });
    m.def("set_tc_util", [&](const int& new_tc_util) {
        device_runtime->set_tc_util(new_tc_util);
    });
    m.def("get_tc_util", [&]() {
        return device_runtime->get_tc_util();
    });
    m.def("set_compile_mode", [&](const int& mode) {
        compile_mode = mode;
    });
    m.def("get_compile_mode", [&]() {
        return compile_mode;
    });

    // Architecture query — lets the Python facade centralize SM89/SM90/SM100
    // dispatch instead of callers computing the arch from raw SM numbers.
    m.def("get_arch_pair", [&]() {
        return device_runtime->get_arch_pair();
    });
    m.def("get_arch_major", [&]() {
        return device_runtime->get_arch_major();
    });

    m.def("init", [&](const std::string& library_root_path, const std::string& cuda_home_path_by_python) {
        Compiler::prepare_init(library_root_path, cuda_home_path_by_python);
        KernelRuntime::prepare_init(cuda_home_path_by_python);
    });
}

} // namespace asym_gemm::runtime
