#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <string>

#define CHECK_CUDA(call) do {                                    \
  cudaError_t e = (call);                                        \
  if (e != cudaSuccess) {                                        \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                    \
            __FILE__, __LINE__, cudaGetErrorString(e));          \
    std::exit(1);                                                \
  }                                                              \
} while(0)

// BUGGY: publish flag first, then write payload (no ordering)
__global__ void kernel_buggy_publish_first(int* payload, int* flag, int iters) {
  // single thread for clarity
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int i = 0; i < iters; ++i) {
      // publish "new item i is ready" FIRST (wrong)
      atomicExch(flag, i + 1);

      // add some delay to widen the race window
      unsigned long long t0 = clock64();
      while (clock64() - t0 < 1000ULL) {
      }

      // write payload AFTER publishing (CPU may read old value)
      payload[i] = 0x12340000 + i;
    }
  }
}

// FIXED: write payload, make it visible system-wide, then publish flag
__global__ void kernel_fixed_fence_then_publish(int* payload, int* flag, int iters) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int i = 0; i < iters; ++i) {
      payload[i] = 0x12340000 + i;

      // ensure payload writes are visible to CPU before flag publish
      __threadfence_system();

      atomicExch(flag, i + 1);
    }
  }
}

static inline void cpu_spin_pause() {
#if defined(__x86_64__) || defined(_M_X64)
  asm volatile("pause" ::: "memory");
#else
  asm volatile("" ::: "memory");
#endif
}

int main(int argc, char** argv) {
  bool use_fixed = false;
  if (argc >= 2 && std::string(argv[1]) == "fixed") use_fixed = true;

  // Required for host mapping on some setups (must be before any CUDA context work)
  CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));

  const int iters = 200000;

  // Mapped pinned host memory so GPU writes go directly to host memory
  int* h_payload = nullptr;
  int* h_flag    = nullptr;

  CHECK_CUDA(cudaHostAlloc(&h_payload, iters * sizeof(int), cudaHostAllocMapped));
  CHECK_CUDA(cudaHostAlloc(&h_flag,    sizeof(int),         cudaHostAllocMapped));

  // init
  for (int i = 0; i < iters; ++i) h_payload[i] = 0xDEADBEEF;
  *h_flag = 0;

  int* d_payload = nullptr;
  int* d_flag    = nullptr;
  CHECK_CUDA(cudaHostGetDevicePointer(&d_payload, h_payload, 0));
  CHECK_CUDA(cudaHostGetDevicePointer(&d_flag,    h_flag,    0));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  auto t0 = std::chrono::high_resolution_clock::now();

  if (use_fixed) {
    kernel_fixed_fence_then_publish<<<1, 1, 0, stream>>>(d_payload, d_flag, iters);
  } else {
    kernel_buggy_publish_first<<<1, 1, 0, stream>>>(d_payload, d_flag, iters);
  }
  CHECK_CUDA(cudaGetLastError());

  // CPU polling: whenever flag says "k items ready", read payload[k-1]
  int errors = 0;
  int last_seen = 0;

  while (last_seen < iters) {
    int f = *h_flag;          // poll published count
    cpu_spin_pause();

    // consume all newly published items
    while (last_seen < f) {
      int idx = last_seen;    // 0-based
      int v = h_payload[idx];

      int expected = 0x12340000 + idx;
      if (v != expected) {
        // record first few mismatches
        if (errors < 10) {
          printf("MISMATCH at %d: got 0x%08x expected 0x%08x (flag=%d)\n",
                 idx, (unsigned)v, (unsigned)expected, f);
        }
        errors++;
      }
      last_seen++;
    }
  }

  CHECK_CUDA(cudaStreamSynchronize(stream));

  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  printf("\nMode: %s\n", use_fixed ? "fixed(__threadfence_system then publish)" : "buggy(publish then write)");
  printf("iters=%d errors=%d time=%.2f ms\n", iters, errors, ms);

  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFreeHost(h_payload));
  CHECK_CUDA(cudaFreeHost(h_flag));
  return 0;
}
