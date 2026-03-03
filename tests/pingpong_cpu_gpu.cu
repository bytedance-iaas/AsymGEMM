#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <chrono>
#include <string>

#define CHECK(call) do {                                      \
  cudaError_t e = (call);                                     \
  if (e != cudaSuccess) {                                     \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                 \
            __FILE__, __LINE__, cudaGetErrorString(e));       \
    std::exit(1);                                             \
  }                                                           \
} while(0)

static inline void cpu_pause() {
#if defined(__x86_64__) || defined(_M_X64)
  asm volatile("pause" ::: "memory");
#else
  asm volatile("" ::: "memory");
#endif
}

// Put flags on separate cache lines to reduce false sharing
struct alignas(64) Mailbox {
  uint32_t cpu_to_gpu;
  uint32_t pad0[15];
  uint32_t gpu_to_cpu;
  uint32_t pad1[15];
  uint32_t payload;   // a tiny payload (you can expand to an array if you want)
  uint32_t pad2[15];
  uint32_t response;
  uint32_t pad3[15];
};

__global__ void gpu_pingpong(Mailbox* m, int iters) {
  if (blockIdx.x || threadIdx.x) return;

  uint32_t expected = 1;
  for (int i = 0; i < iters; ++i, ++expected) {
    // wait for CPU publish
    while (atomicAdd((unsigned int*)&m->cpu_to_gpu, 0) != expected) {
      // spin
    }

    // read payload, produce response
    uint32_t x = m->payload;
    m->response = x ^ 0xA5A5A5A5u;

    // make payload/response visible to CPU before publishing gpu_to_cpu
    __threadfence_system();
    atomicExch((unsigned int*)&m->gpu_to_cpu, expected);
  }
}

static void print_attr(int dev, cudaDeviceAttr attr, const char* name) {
  int v = 0;
  cudaError_t e = cudaDeviceGetAttribute(&v, attr, dev);
  if (e == cudaSuccess) printf("%s = %d\n", name, v);
  else { cudaGetLastError(); printf("%s = (n/a)\n", name); }
}

int main(int argc, char** argv) {
  std::string mode = "mapped";   // mapped | pageable
  int iters = 200000;
  int warmup = 20000;

  if (argc >= 2) mode = argv[1];
  if (argc >= 3) iters = std::atoi(argv[2]);
  if (argc >= 4) warmup = std::atoi(argv[3]);

  int dev = 0;
  CHECK(cudaSetDevice(dev));

  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("Device: %s\n", prop.name);

  // Useful to compare platforms
  print_attr(dev, cudaDevAttrPageableMemoryAccess, "cudaDevAttrPageableMemoryAccess");
  print_attr(dev, cudaDevAttrPageableMemoryAccessUsesHostPageTables,
             "cudaDevAttrPageableMemoryAccessUsesHostPageTables");

  Mailbox* h = nullptr;
  Mailbox* d = nullptr;

  if (mode == "mapped") {
    // required on some systems before creating the context
    CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    CHECK(cudaHostAlloc(&h, sizeof(Mailbox), cudaHostAllocMapped));
    std::memset(h, 0, sizeof(Mailbox));
    CHECK(cudaHostGetDevicePointer(&d, h, 0));
    printf("Mode: mapped (cudaHostAllocMapped)\n");
  } else if (mode == "pageable") {
    h = (Mailbox*)std::malloc(sizeof(Mailbox));
    if (!h) { printf("malloc failed\n"); return 1; }
    std::memset(h, 0, sizeof(Mailbox));
    d = h;  // pass host pointer directly (works only if pageable access is supported)
    printf("Mode: pageable (malloc host pointer directly)\n");
  } else {
    printf("Usage: %s [mapped|pageable] [iters] [warmup]\n", argv[0]);
    return 1;
  }

  cudaStream_t stream;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Launch persistent kernel (1 block, 1 thread) for warmup + iters
  // Total iterations in kernel = warmup + iters
  int total = warmup + iters;
  gpu_pingpong<<<1, 1, 0, stream>>>(d, total);
  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(launch_err));
    return 1;
  }

  // CPU side ping-pong
  uint32_t expected = 1;

  // Warmup
  for (int i = 0; i < warmup; ++i, ++expected) {
    h->payload = expected;
    std::atomic_thread_fence(std::memory_order_release);
    h->cpu_to_gpu = expected;

    while (h->gpu_to_cpu != expected) cpu_pause();
    std::atomic_thread_fence(std::memory_order_acquire);

    // touch response to prevent optimization
    volatile uint32_t r = h->response;
    (void)r;
  }

  // Timed section
  auto t0 = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iters; ++i, ++expected) {
    h->payload = expected;
    std::atomic_thread_fence(std::memory_order_release);
    h->cpu_to_gpu = expected;

    while (h->gpu_to_cpu != expected) cpu_pause();
    std::atomic_thread_fence(std::memory_order_acquire);

    // validate occasionally
    if ((i & 0x3FFFF) == 0) {
      uint32_t want = (expected ^ 0xA5A5A5A5u);
      if (h->response != want) {
        printf("Mismatch at i=%d expected response=0x%08x got=0x%08x\n",
               i, want, h->response);
        break;
      }
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();

  // Ensure GPU finished
  cudaError_t sync_err = cudaStreamSynchronize(stream);
  if (sync_err != cudaSuccess) {
    printf("cudaStreamSynchronize error: %s\n", cudaGetErrorString(sync_err));
  }

  double rtt_us = (sec * 1e6) / iters;
  double msgs_mpps = (iters / sec) / 1e6;

  printf("\nResults:\n");
  printf("  iters=%d  warmup=%d\n", iters, warmup);
  printf("  total_time=%.6f s\n", sec);
  printf("  avg_round_trip=%.3f us\n", rtt_us);
  printf("  msg_rate=%.3f Mmsgs/s\n", msgs_mpps);

  CHECK(cudaStreamDestroy(stream));

  if (mode == "mapped") CHECK(cudaFreeHost(h));
  else std::free(h);

  return 0;
}