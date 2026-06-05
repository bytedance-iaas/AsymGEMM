/*
 * asym_gemm._cpu_C — pybind11 wrapper around cpu_gemm for use by the
 * Python-side unified MoE runtime (asym_gemm.unified_moe.Layer).
 *
 * This extension is independent of asym_gemm._C (the CUDA extension): it
 * has no torch dependency, and a host without CUDA still builds and loads
 * this module. Runtime availability of the AMX path is reported via
 * caps()['has_amx_int8'].
 *
 * Uses py::array_t<T> typed parameters so pybind11 enforces dtype matches
 * on the Python boundary (auto-converting where safe; rejecting otherwise).
 * Arrays are required C-contiguous via py::array::c_style.
 */
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpu_gemm/cpu_gemm.h"
#include "cpu_gemm/cpu_gemm.hpp"

namespace py = pybind11;

namespace {

void check(cg_status_t s, const char* where) {
  if (s == CG_OK) return;
  throw std::runtime_error(std::string(where) + " failed: status=" + std::to_string((int)s));
}

struct RuntimeHandle {
  cg_runtime_t* rt = nullptr;
  explicit RuntimeHandle(int n_threads) {
    rt = cg_runtime_create(n_threads);
    if (!rt) throw std::runtime_error("cg_runtime_create returned NULL");
  }
  ~RuntimeHandle() { if (rt) cg_runtime_destroy(rt); }
  int threads() const { return cg_runtime_threads(rt); }
};

py::dict caps_dict() {
  cg_caps_t c = cg_query_caps();
  py::dict d;
  d["has_avx2"]        = (bool)c.has_avx2;
  d["has_fma"]         = (bool)c.has_fma;
  d["has_avx512f"]     = (bool)c.has_avx512f;
  d["has_avx512_bf16"] = (bool)c.has_avx512_bf16;
  d["has_avx_vnni"]    = (bool)c.has_avx_vnni;
  d["has_amx_bf16"]    = (bool)c.has_amx_bf16;
  d["has_amx_int8"]    = (bool)c.has_amx_int8;
  return d;
}

size_t pack_b_int8_amx_size_py(size_t n, size_t k) {
  return cg_pack_b_int8_amx_size(n, k);
}

py::array_t<uint8_t> pack_b_int8_amx_py(
    py::array_t<int8_t,  py::array::c_style | py::array::forcecast> b_int8,
    py::array_t<float,   py::array::c_style | py::array::forcecast> b_scales) {
  if (b_int8.ndim() != 2)
    throw std::invalid_argument("b_int8 must be 2-D");
  if (b_scales.ndim() != 1)
    throw std::invalid_argument("b_scales must be 1-D");
  ssize_t n = b_int8.shape(0);
  ssize_t k = b_int8.shape(1);
  if (b_scales.shape(0) != n)
    throw std::invalid_argument("b_scales length != b_int8 rows");

  size_t bytes = cg_pack_b_int8_amx_size((size_t)n, (size_t)k);
  if (bytes == 0)
    throw std::invalid_argument("k not multiple of 64 or n==0");

  void* buf = nullptr;
  if (posix_memalign(&buf, 64, bytes) != 0 || !buf)
    throw std::bad_alloc();

  cg_status_t st = cg_pack_b_int8_amx(
      (size_t)n, (size_t)k, b_int8.data(), b_scales.data(), buf);
  if (st != CG_OK) { free(buf); check(st, "cg_pack_b_int8_amx"); }

  py::capsule owner(buf, [](void* p) { free(p); });
  return py::array_t<uint8_t>(
      {(ssize_t)bytes}, {(ssize_t)1}, static_cast<uint8_t*>(buf), owner);
}

void gemm_bf16_int8_py(
    RuntimeHandle& rt,
    py::array_t<uint16_t, py::array::c_style | py::array::forcecast> a_bf16,
    py::array_t<int8_t,   py::array::c_style | py::array::forcecast> b_int8,
    py::array_t<float,    py::array::c_style | py::array::forcecast> b_scales,
    py::array_t<float,    py::array::c_style>                         c_fp32,
    float alpha, float beta) {
  if (a_bf16.ndim() != 2 || b_int8.ndim() != 2 || c_fp32.ndim() != 2 || b_scales.ndim() != 1)
    throw std::invalid_argument("bad rank");
  ssize_t m = a_bf16.shape(0), k = a_bf16.shape(1);
  ssize_t n = b_int8.shape(0);
  if (b_int8.shape(1) != k) throw std::invalid_argument("b_int8 k mismatch");
  if (c_fp32.shape(0) != m || c_fp32.shape(1) != n)
    throw std::invalid_argument("c_fp32 shape mismatch");
  if (b_scales.shape(0) != n) throw std::invalid_argument("b_scales length mismatch");
  if (!c_fp32.writeable()) throw std::invalid_argument("c_fp32 must be writeable");

  auto d = cpu_gemm::make_desc();
  d.m = (size_t)m; d.n = (size_t)n; d.k = (size_t)k;
  d.alpha = alpha; d.beta = beta;
  d.a = a_bf16.data();   d.lda = (size_t)k; d.dtype_a = CG_BF16;
  d.b = b_int8.data();   d.ldb = (size_t)k; d.dtype_b = CG_INT8;
  d.b_scales = b_scales.data();
  d.c = c_fp32.mutable_data(); d.ldc = (size_t)n; d.dtype_c = CG_F32;

  py::gil_scoped_release rel;
  check(cg_gemm(rt.rt, &d), "cg_gemm (BF16xINT8)");
}

void gemm_bf16_int8_packed_py(
    RuntimeHandle& rt,
    py::array_t<uint16_t, py::array::c_style | py::array::forcecast> a_bf16,
    py::array_t<uint8_t,  py::array::c_style>                         b_packed,
    py::array_t<float,    py::array::c_style>                         c_fp32,
    size_t n, size_t k,
    float alpha, float beta) {
  if (a_bf16.ndim() != 2 || c_fp32.ndim() != 2 || b_packed.ndim() != 1)
    throw std::invalid_argument("bad rank");
  ssize_t m = a_bf16.shape(0);
  if ((ssize_t)k != a_bf16.shape(1)) throw std::invalid_argument("a_bf16 k mismatch");
  if (c_fp32.shape(0) != m || (size_t)c_fp32.shape(1) != n)
    throw std::invalid_argument("c_fp32 shape mismatch");

  size_t expected = cg_pack_b_int8_amx_size(n, k);
  if ((size_t)b_packed.shape(0) != expected)
    throw std::invalid_argument("b_packed size != cg_pack_b_int8_amx_size(n,k)");
  if (reinterpret_cast<std::uintptr_t>(b_packed.data()) % 64 != 0)
    throw std::invalid_argument("b_packed must be 64-byte aligned");
  if (!c_fp32.writeable()) throw std::invalid_argument("c_fp32 must be writeable");

  auto d = cpu_gemm::make_desc();
  d.m = (size_t)m; d.n = n; d.k = k;
  d.alpha = alpha; d.beta = beta;
  d.a = a_bf16.data();   d.lda = k; d.dtype_a = CG_BF16;
  d.b = b_packed.data(); d.ldb = k; d.dtype_b = CG_INT8_PACKED_AMX;
  d.b_scales = nullptr;
  d.c = c_fp32.mutable_data(); d.ldc = n; d.dtype_c = CG_F32;

  py::gil_scoped_release rel;
  check(cg_gemm(rt.rt, &d), "cg_gemm (BF16xINT8_PACKED)");
}

}  // namespace

PYBIND11_MODULE(_cpu_C, m) {
  m.doc() = "asym_gemm._cpu_C — pybind11 wrapper for cpu_gemm (AMX INT8)";

  py::class_<RuntimeHandle>(m, "Runtime")
      .def(py::init<int>(), py::arg("n_threads") = 0)
      .def_property_readonly("threads", &RuntimeHandle::threads);

  m.def("caps", &caps_dict, "Host CPU capabilities (AMX, AVX-512, ...)");

  m.def("pack_b_int8_amx_size", &pack_b_int8_amx_size_py,
        py::arg("n"), py::arg("k"));
  m.def("pack_b_int8_amx", &pack_b_int8_amx_py,
        py::arg("b_int8"), py::arg("b_scales"));

  m.def("gemm_bf16_int8", &gemm_bf16_int8_py,
        py::arg("rt"), py::arg("a_bf16"), py::arg("b_int8"),
        py::arg("b_scales"), py::arg("c_fp32"),
        py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f);

  m.def("gemm_bf16_int8_packed", &gemm_bf16_int8_packed_py,
        py::arg("rt"), py::arg("a_bf16"), py::arg("b_packed"),
        py::arg("c_fp32"), py::arg("n"), py::arg("k"),
        py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f);
}
