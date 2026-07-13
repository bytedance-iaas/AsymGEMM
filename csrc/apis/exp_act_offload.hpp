// Copyright (c) 2026.

#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace asym_gemm::exp_act_offload {

void sm100_grouped_lora_a_grad_bf16_cpu_right(
    const torch::Tensor& grad_low_rank,
    const torch::Tensor& source_cpu,
    const torch::Tensor& grad_a,
    const torch::Tensor& offsets,
    const torch::Tensor& experts,
    int64_t list_size);

void sm100_grouped_lora_a_pair_grad_bf16_cpu_right(
    const torch::Tensor& dS_gate,
    const torch::Tensor& dS_up,
    const torch::Tensor& x_cpu,
    const torch::Tensor& grad_gate,
    const torch::Tensor& grad_up,
    const torch::Tensor& offsets,
    const torch::Tensor& experts,
    int64_t list_size);

void sm100_grouped_lora_b_backward_bf16_cpu_source(
    const torch::Tensor& grad_out_cpu,
    const torch::Tensor& low_rank,
    const torch::Tensor& lora_b,
    const torch::Tensor& dS,
    const torch::Tensor& grad_b,
    const torch::Tensor& offsets,
    const torch::Tensor& experts,
    int64_t list_size,
    double scale);

static void register_apis(pybind11::module_& m) {
    namespace py = pybind11;
    m.def(
        "sm100_grouped_lora_a_grad_bf16_cpu_right",
        &sm100_grouped_lora_a_grad_bf16_cpu_right,
        py::arg("grad_low_rank"),
        py::arg("source_cpu"),
        py::arg("grad_a"),
        py::arg("offsets"),
        py::arg("experts"),
        py::arg("list_size"));
    m.def(
        "sm100_grouped_lora_a_pair_grad_bf16_cpu_right",
        &sm100_grouped_lora_a_pair_grad_bf16_cpu_right,
        py::arg("dS_gate"),
        py::arg("dS_up"),
        py::arg("x_cpu"),
        py::arg("grad_gate"),
        py::arg("grad_up"),
        py::arg("offsets"),
        py::arg("experts"),
        py::arg("list_size"));
    m.def(
        "sm100_grouped_lora_b_backward_bf16_cpu_source",
        &sm100_grouped_lora_b_backward_bf16_cpu_source,
        py::arg("grad_out_cpu"),
        py::arg("low_rank"),
        py::arg("lora_b"),
        py::arg("dS"),
        py::arg("grad_b"),
        py::arg("offsets"),
        py::arg("experts"),
        py::arg("list_size"),
        py::arg("scale"));
}

}  // namespace asym_gemm::exp_act_offload
