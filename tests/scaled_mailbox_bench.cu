#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <atomic>
#include <string>

#define CHECK(call) do {                                      \
  cudaError_t e = (call);                                     \
  if (e != cudaSuccess) {                                     \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                 \
            __FILE__, __LINE__, cudaGetErrorString(e));       \
    std::exit(1);                                             \
  }                                                           \
} while(0)

static inline uint64_t now_ns() {
  return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// One mailbox per thread. Keep flag/ack/payload on separate cache lines to reduce false sharing.
struct alignas(64) Mailbox {
  uint32_t flag;     uint32_t pad0[15];
  uint32_t ack;      uint32_t pad1[15];
  uint32_t payload;  uint32_t pad2[15];
};

__global__ void gpu_consume_mailboxes(Mailbox* m, int n, uint32_t rounds) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  Mailbox* mb = &m[tid];
  uint32_t expect = 1;

  for (uint32_t r = 0; r < rounds; ++r, ++expect) {
    // wait for CPU publish
    while (atomicAdd((unsigned int*)&mb->flag, 0) != expect) { }

    // acquire-like barrier (pair with CPU release fence)
    __threadfence_system();

    // touch payload
    uint32_t x = mb->payload;
    (void)x;

    // publish ack
    __threadfence_system();
    atomicExch((unsigned int*)&mb->ack, expect);
  }
}

int main(int argc, char** argv) {
  // Usage:
  //   ./scaled_mailbox [mapped|pageable] [N] [rounds]
  // N = number of mailboxes/threads
  // rounds = number of updates per mailbox
  std::string mode = "mapped";
  int N = 4096;
  uint32_t rounds = 2000;

  if (argc >= 2) mode = argv[1];
  if (argc >= 3) N = std::atoi(argv[2]);
  if (argc >= 4) rounds = (uint32_t)std::atoi(argv[3]);

  int dev = 0;
  CHECK(cudaSetDevice(dev));
  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("Device: %s\n", prop.name);
  printf("Mode=%s  N=%d  rounds=%u  (total updates = N*rounds = %llu)\n",
         mode.c_str(), N, rounds, (unsigned long long)N * rounds);

  Mailbox* h = nullptr;
  Mailbox* d = nullptr;

  if (mode == "mapped") {
    CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CHECK(cudaHostAlloc(&h, (size_t)N * sizeof(Mailbox), cudaHostAllocMapped));
    std::memset(h, 0, (size_t)N * sizeof(Mailbox));
    CHECK(cudaHostGetDevicePointer(&d, h, 0));
    printf("Using cudaHostAllocMapped (pinned + mapped)\n");
  } else if (mode == "pageable") {
    h = (Mailbox*)std::malloc((size_t)N * sizeof(Mailbox));
    if (!h) { printf("malloc failed\n"); return 1; }
    std::memset(h, 0, (size_t)N * sizeof(Mailbox));
    d = h; // works only if pageable host memory access (ATS/HMM) is enabled
    printf("Using malloc pageable host pointer directly\n");
  } else {
    printf("Usage: %s [mapped|pageable] [N] [rounds]\n", argv[0]);
    return 1;
  }

  // init
  for (int i = 0; i < N; ++i) {
    h[i].flag = 0;
    h[i].ack = 0;
    h[i].payload = 0;
  }

  cudaStream_t stream;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  int threads = 256;
  int blocks  = (N + threads - 1) / threads;

  // Launch persistent consumer kernel
  gpu_consume_mailboxes<<<blocks, threads, 0, stream>>>(d, N, rounds);
  CHECK(cudaGetLastError());

  // CPU producer: for each round, publish to all N mailboxes, then wait for all acks
  uint64_t t0 = now_ns();

  for (uint32_t r = 1; r <= rounds; ++r) {
    // publish all
    for (int i = 0; i < N; ++i) {
      h[i].payload = (uint32_t)i ^ (r * 1315423911u);

      // CPU release fence before publishing flag
      std::atomic_thread_fence(std::memory_order_release);
      h[i].flag = r;
    }

    // wait all acks
    for (int i = 0; i < N; ++i) {
      while (h[i].ack != r) { /* spin */ }
      std::atomic_thread_fence(std::memory_order_acquire);
    }
  }

  uint64_t t1 = now_ns();

  CHECK(cudaStreamSynchronize(stream));

  double sec = (t1 - t0) * 1e-9;
  double total_updates = (double)N * rounds;
  double updates_per_s = total_updates / sec;

  printf("\nResult:\n");
  printf("  time = %.6f s\n", sec);
  printf("  updates/s = %.3f Mupdates/s\n", updates_per_s / 1e6);
  printf("  avg per-update = %.1f ns (includes CPU publish + GPU consume + ack)\n",
         (double)(t1 - t0) / total_updates);

  CHECK(cudaStreamDestroy(stream));

  if (mode == "mapped") CHECK(cudaFreeHost(h));
  else std::free(h);

  return 0;
}