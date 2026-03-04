#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <string>
#include <atomic>

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

// Separate cache lines to reduce false sharing
struct alignas(64) Mailbox {
  uint32_t cpu_to_gpu;
  uint32_t pad0[15];
  uint32_t gpu_to_cpu;
  uint32_t pad1[15];
  uint32_t payload;
  uint32_t pad2[15];
  uint32_t response;
  uint32_t pad3[15];
};

__global__ void gpu_mailbox_host(Mailbox* m, int total_iters) {
  if (blockIdx.x || threadIdx.x) return;

  uint32_t expected = 1;
  for (int i = 0; i < total_iters; ++i, ++expected) {
    while (atomicAdd((unsigned int*)&m->cpu_to_gpu, 0) != expected) { }
    uint32_t x = m->payload;
    m->response = x ^ 0xA5A5A5A5u;

    // publish payload/response to CPU before publishing gpu_to_cpu
    __threadfence_system();
    atomicExch((unsigned int*)&m->gpu_to_cpu, expected);
  }
}

// Same mailbox but in device memory; CPU interacts via cudaMemcpy (heavy)
__global__ void gpu_mailbox_device(Mailbox* m, int total_iters) {
  if (blockIdx.x || threadIdx.x) return;

  uint32_t expected = 1;
  for (int i = 0; i < total_iters; ++i, ++expected) {
    while (atomicAdd((unsigned int*)&m->cpu_to_gpu, 0) != expected) { }
    uint32_t x = m->payload;
    m->response = x ^ 0xA5A5A5A5u;

    __threadfence_system();
    atomicExch((unsigned int*)&m->gpu_to_cpu, expected);
  }
}

static inline uint64_t now_ns() {
  return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  // mode: "fine" (mapped host mailbox) or "copy" (memcpy flag/payload each iter)
  std::string mode = "fine";
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
  printf("Mode: %s | iters=%d warmup=%d\n", mode.c_str(), iters, warmup);

  cudaStream_t stream;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  const int total = warmup + iters;

  if (mode == "fine") {
    // Fine-grained: mapped pinned host mailbox, CPU polls, no per-iter CUDA API
    CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    Mailbox* h = nullptr;
    Mailbox* d = nullptr;
    CHECK(cudaHostAlloc(&h, sizeof(Mailbox), cudaHostAllocMapped));
    std::memset(h, 0, sizeof(Mailbox));
    CHECK(cudaHostGetDevicePointer(&d, h, 0));

    gpu_mailbox_host<<<1,1,0,stream>>>(d, total);
    CHECK(cudaGetLastError());

    uint32_t expected = 1;

    // warmup
    for (int i = 0; i < warmup; ++i, ++expected) {
      h->payload = expected;
      std::atomic_thread_fence(std::memory_order_release);
      h->cpu_to_gpu = expected;

      while (h->gpu_to_cpu != expected) cpu_pause();
      std::atomic_thread_fence(std::memory_order_acquire);
      (void)h->response;
    }

    // timed
    uint64_t t0 = now_ns();
    for (int i = 0; i < iters; ++i, ++expected) {
      h->payload = expected;
      std::atomic_thread_fence(std::memory_order_release);
      h->cpu_to_gpu = expected;

      while (h->gpu_to_cpu != expected) cpu_pause();
      std::atomic_thread_fence(std::memory_order_acquire);

      if ((i & 0x3FFFF) == 0) {
        uint32_t want = expected ^ 0xA5A5A5A5u;
        if (h->response != want) {
          printf("Mismatch at i=%d want=0x%08x got=0x%08x\n", i, want, h->response);
          break;
        }
      }
    }
    uint64_t t1 = now_ns();

    CHECK(cudaStreamSynchronize(stream));
    double sec = (t1 - t0) * 1e-9;
    double rtt_ns = (double)(t1 - t0) / iters;
    printf("Results (fine): avg_RTT = %.1f ns  |  %.3f Mroundtrips/s\n",
           rtt_ns, (iters/sec)/1e6);

    CHECK(cudaFreeHost(h));
  }
  else if (mode == "copy") {
    // Heavy barrier: mailbox in device memory; CPU uses cudaMemcpyAsync each iter.
    // This is “always correct but high overhead”.
    Mailbox* d = nullptr;
    CHECK(cudaMalloc(&d, sizeof(Mailbox)));
    CHECK(cudaMemsetAsync(d, 0, sizeof(Mailbox), stream));
    CHECK(cudaStreamSynchronize(stream));

    // Host staging buffer (pinned to make memcpy as fast as possible)
    Mailbox* h = nullptr;
    CHECK(cudaHostAlloc(&h, sizeof(Mailbox), cudaHostAllocDefault));
    std::memset(h, 0, sizeof(Mailbox));

    gpu_mailbox_device<<<1,1,0,stream>>>(d, total);
    CHECK(cudaGetLastError());

    uint32_t expected = 1;

    auto do_one = [&](bool timed) {
      // CPU writes payload + cpu_to_gpu to device
      h->payload = expected;
      h->cpu_to_gpu = expected;
      CHECK(cudaMemcpyAsync(&d->payload,   &h->payload,   sizeof(uint32_t),
                            cudaMemcpyHostToDevice, stream));
      CHECK(cudaMemcpyAsync(&d->cpu_to_gpu,&h->cpu_to_gpu,sizeof(uint32_t),
                            cudaMemcpyHostToDevice, stream));

      // Poll gpu_to_cpu by copying it back each time (heavy)
      uint32_t flag = 0;
      do {
        CHECK(cudaMemcpyAsync(&flag, &d->gpu_to_cpu, sizeof(uint32_t),
                              cudaMemcpyDeviceToHost, stream));
        CHECK(cudaStreamSynchronize(stream));
      } while (flag != expected);

      // Read response (also a copy)
      CHECK(cudaMemcpyAsync(&h->response, &d->response, sizeof(uint32_t),
                            cudaMemcpyDeviceToHost, stream));
      CHECK(cudaStreamSynchronize(stream));
    };

    // warmup
    for (int i = 0; i < warmup; ++i, ++expected) do_one(false);

    uint64_t t0 = now_ns();
    for (int i = 0; i < iters; ++i, ++expected) {
      do_one(true);
      if ((i & 0x3FFFF) == 0) {
        uint32_t want = expected ^ 0xA5A5A5A5u;
        if (h->response != want) {
          printf("Mismatch at i=%d want=0x%08x got=0x%08x\n", i, want, h->response);
          break;
        }
      }
    }
    uint64_t t1 = now_ns();

    CHECK(cudaStreamSynchronize(stream));
    double sec = (t1 - t0) * 1e-9;
    double rtt_ns = (double)(t1 - t0) / iters;
    printf("Results (copy): avg_RTT = %.1f ns  |  %.6f Mroundtrips/s\n",
           rtt_ns, (iters/sec)/1e6);

    CHECK(cudaFreeHost(h));
    CHECK(cudaFree(d));
  }
  else {
    printf("Usage: %s [fine|copy] [iters] [warmup]\n", argv[0]);
    return 1;
  }

  CHECK(cudaStreamDestroy(stream));
  return 0;
}