#include "runtime/scratch_arena.h"

#include <cstdlib>
#include <new>

namespace cpu_gemm {

ScratchArena::~ScratchArena() {
  if (buffer_) std::free(buffer_);
}

void* ScratchArena::reserve(std::size_t bytes) {
  std::lock_guard<std::mutex> lk(mtx_);
  if (bytes <= capacity_ && buffer_) return buffer_;

  // Grow geometrically to avoid thrash, but never shrink.
  std::size_t target = capacity_ ? capacity_ : 4096;
  while (target < bytes) target *= 2;

  void* p = nullptr;
  if (posix_memalign(&p, 64, target) != 0) {
    throw std::bad_alloc{};
  }

  if (buffer_) std::free(buffer_);
  buffer_ = p;
  capacity_ = target;
  return buffer_;
}

}  // namespace cpu_gemm
