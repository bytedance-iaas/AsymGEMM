#ifndef CPU_GEMM_RUNTIME_IMPL_H
#define CPU_GEMM_RUNTIME_IMPL_H

#include <memory>

#include "runtime/scratch_arena.h"
#include "runtime/worker_pool.h"

/* Definition of the opaque cg_runtime_t handle, shared by runtime.cpp and
 * dispatch/gemm.cpp. Not exposed in the public API. */
struct cg_runtime {
  std::unique_ptr<cpu_gemm::WorkerPool>  pool;
  std::unique_ptr<cpu_gemm::ScratchArena> scratch;
  int threads;
};

#endif
