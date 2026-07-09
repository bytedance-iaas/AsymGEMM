/*
 * cpu_gemm/cpu_gemm.hpp — thin C++ wrapper around the C ABI.
 *
 * RAII for the runtime handle; a Desc builder that fills sensible defaults.
 * No new functionality — everything goes through the C entry points.
 */
#ifndef CPU_GEMM_HPP
#define CPU_GEMM_HPP

#include <memory>
#include <stdexcept>
#include <string>

#include "cpu_gemm/cpu_gemm.h"

namespace cpu_gemm {

class Error : public std::runtime_error {
 public:
  Error(cg_status_t s, const std::string& msg)
      : std::runtime_error(msg + " (status=" + std::to_string((int)s) + ")"),
        status(s) {}
  cg_status_t status;
};

inline void check(cg_status_t s, const char* where) {
  if (s != CG_OK) throw Error(s, where);
}

/* Owning wrapper for cg_runtime_t. */
class Runtime {
 public:
  explicit Runtime(int n_threads = 0)
      : rt_(cg_runtime_create(n_threads), &cg_runtime_destroy) {
    if (!rt_) throw Error(CG_E_INTERNAL, "cg_runtime_create");
  }

  cg_runtime_t* raw() noexcept { return rt_.get(); }
  int threads() const noexcept { return cg_runtime_threads(rt_.get()); }

 private:
  std::unique_ptr<cg_runtime_t, void (*)(cg_runtime_t*)> rt_;
};

/* Descriptor helper with reasonable defaults. */
inline cg_gemm_desc_t make_desc() {
  cg_gemm_desc_t d{};
  d.order = CG_ROW_MAJOR;
  d.trans_a = CG_NO_TRANS;
  d.trans_b = CG_TRANS;
  d.offset_c_mode = CG_OFFSET_C_NONE;
  d.alpha = 1.0f;
  d.beta = 0.0f;
  return d;
}

inline void gemm(Runtime& rt, const cg_gemm_desc_t& d) {
  check(cg_gemm(rt.raw(), &d), "cg_gemm");
}

inline void gemm_st(const cg_gemm_desc_t& d, int ith, int nth) {
  check(cg_gemm_st(&d, ith, nth), "cg_gemm_st");
}

}  // namespace cpu_gemm
#endif  // CPU_GEMM_HPP
