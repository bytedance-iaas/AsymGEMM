// Copyright (c) 2026.

#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace asym_gemm::dropout {

torch::Tensor pack_bool_mask_2d(const torch::Tensor& mask_bool);
torch::Tensor unpack_bool_mask_2d(const torch::Tensor& mask_packed, int64_t width);
torch::Tensor apply_packed_dropout(const torch::Tensor& x, const torch::Tensor& mask_packed, double dropout_p);
torch::Tensor apply_packed_dropout_(torch::Tensor x, const torch::Tensor& mask_packed, double dropout_p);

static void register_apis(pybind11::module_& m) {
    m.def("pack_bool_mask_2d", &pack_bool_mask_2d, pybind11::arg("mask_bool"));
    m.def("unpack_bool_mask_2d", &unpack_bool_mask_2d, pybind11::arg("mask_packed"), pybind11::arg("width"));
    m.def("apply_packed_dropout", &apply_packed_dropout, pybind11::arg("x"), pybind11::arg("mask_packed"), pybind11::arg("dropout_p"));
    m.def("apply_packed_dropout_", &apply_packed_dropout_, pybind11::arg("x"), pybind11::arg("mask_packed"), pybind11::arg("dropout_p"));
}

}  // namespace asym_gemm::dropout
