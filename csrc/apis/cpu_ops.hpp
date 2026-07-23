// Grace CPU fused training ops (Stage 1+ of agent/impls/cpu_compute.md).
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace asym_gemm::cpu_ops {

void cpu_fused_silu_mul_bf16(const at::Tensor& gate, const at::Tensor& up, at::Tensor& out,
                             int64_t num_threads);
void cpu_fused_silu_backward_bf16(const at::Tensor& gate, const at::Tensor& up,
                                  const at::Tensor& grad_act, at::Tensor& dgate,
                                  at::Tensor& dup, int64_t num_threads);
void cpu_silu_bf16(const at::Tensor& gate, at::Tensor& out, int64_t num_threads);
void cpu_mul_bf16_(at::Tensor& inout, const at::Tensor& other, int64_t num_threads);
void cpu_grouped_lora_a_grad_bf16(const at::Tensor& dS, const at::Tensor& x, at::Tensor& grad_a,
                                  const at::Tensor& pairs, const at::Tensor& group_experts,
                                  int64_t num_threads);
void cpu_rmsnorm_bf16(const at::Tensor& x, const at::Tensor& w, at::Tensor& out,
                      double eps, int64_t num_threads);
double cpu_widen_bf16_sqsum(const at::Tensor& src, at::Tensor& dst, int64_t num_threads);
bool cpu_ops_sve_compiled();

static void register_apis(pybind11::module_& m) {
    namespace py = pybind11;
    m.def("cpu_fused_silu_mul_bf16", &cpu_fused_silu_mul_bf16,
          py::call_guard<py::gil_scoped_release>(),
          py::arg("gate"), py::arg("up"), py::arg("out"), py::arg("num_threads") = -1);
    m.def("cpu_fused_silu_backward_bf16", &cpu_fused_silu_backward_bf16,
          py::call_guard<py::gil_scoped_release>(),
          py::arg("gate"), py::arg("up"), py::arg("grad_act"), py::arg("dgate"),
          py::arg("dup"), py::arg("num_threads") = -1);
    m.def("cpu_silu_bf16", &cpu_silu_bf16, py::call_guard<py::gil_scoped_release>(),
          py::arg("gate"), py::arg("out"), py::arg("num_threads") = -1);
    m.def("cpu_mul_bf16_", &cpu_mul_bf16_, py::call_guard<py::gil_scoped_release>(),
          py::arg("inout"), py::arg("other"), py::arg("num_threads") = -1);
    m.def("cpu_grouped_lora_a_grad_bf16", &cpu_grouped_lora_a_grad_bf16,
          py::call_guard<py::gil_scoped_release>(),
          py::arg("dS"), py::arg("x"), py::arg("grad_a"), py::arg("pairs"),
          py::arg("group_experts"), py::arg("num_threads") = -1);
    m.def("cpu_rmsnorm_bf16", &cpu_rmsnorm_bf16, py::call_guard<py::gil_scoped_release>(),
          py::arg("x"), py::arg("w"), py::arg("out"), py::arg("eps"), py::arg("num_threads") = -1);
    m.def("cpu_widen_bf16_sqsum", &cpu_widen_bf16_sqsum,
          py::call_guard<py::gil_scoped_release>(),
          py::arg("src"), py::arg("dst"), py::arg("num_threads") = -1);
    m.def("cpu_ops_sve_compiled", &cpu_ops_sve_compiled);
}

} // namespace asym_gemm::cpu_ops
