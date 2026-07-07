/*
 * Small thread-safe scratch arena for transient per-call buffers.
 * Direct descendant of cpu_backend/shared_mem_buffer but stripped of the
 * NUMA bookkeeping. Owners (e.g. an operator instance) call reserve() once
 * and get aligned-and-grown storage they can reuse.
 */
#ifndef CPU_GEMM_RUNTIME_SCRATCH_ARENA_H
#define CPU_GEMM_RUNTIME_SCRATCH_ARENA_H

#include <cstddef>
#include <cstdlib>
#include <mutex>

namespace cpu_gemm {

class ScratchArena {
 public:
  ScratchArena() = default;
  ~ScratchArena();

  ScratchArena(const ScratchArena&) = delete;
  ScratchArena& operator=(const ScratchArena&) = delete;

  /* Ensure at least `bytes` of 64-byte-aligned storage is available.
   * Returns a pointer valid until the next reserve() with a larger size or
   * destruction of the arena. Repeated reserve() with smaller sizes are
   * free (no realloc). Thread-safe. */
  void* reserve(std::size_t bytes);

  std::size_t capacity() const noexcept { return capacity_; }

 private:
  std::mutex  mtx_;
  void*       buffer_ = nullptr;
  std::size_t capacity_ = 0;
};

}  // namespace cpu_gemm
#endif
