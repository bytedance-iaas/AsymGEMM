#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#define CHECK_CUDA(call)                                                        \
  do {                                                                          \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "   \
                << cudaGetErrorString(err__) << std::endl;                      \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

__global__ void scale_kernel(float* data, int n, float factor) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] *= factor;
  }
}

int main() {
  constexpr int n = 1024;
  constexpr size_t bytes = n * sizeof(float);

  float* host_pinned = nullptr;
  CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&host_pinned), bytes,
                           cudaHostAllocDefault));

  std::cout << "Pinned host pointer: " << static_cast<void*>(host_pinned)
            << std::endl;

  // CPU writes to pinned memory.
  for (int i = 0; i < n; ++i) {
    host_pinned[i] = static_cast<float>(i);
  }

  // CPU reads from pinned memory.
  std::cout << "CPU read before kernel: [0]=" << host_pinned[0]
            << ", [1]=" << host_pinned[1]
            << ", [1023]=" << host_pinned[n - 1] << std::endl;

  int device = 0;
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  CHECK_CUDA(cudaSetDevice(device));

  if (!prop.canMapHostMemory) {
    std::cerr << "Device cannot map host memory (canMapHostMemory=0)." << std::endl;
    CHECK_CUDA(cudaFreeHost(host_pinned));
    return EXIT_FAILURE;
  }

  // Map pinned host memory into device address space and let GPU modify it.
  float* device_alias = nullptr;
  CHECK_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_alias),
                                      host_pinned, 0));

  constexpr int threads = 256;
  int blocks = (n + threads - 1) / threads;
  scale_kernel<<<blocks, threads>>>(device_alias, n, 2.0f);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::cout << "CPU read after kernel:  [0]=" << host_pinned[0]
            << ", [1]=" << host_pinned[1]
            << ", [1023]=" << host_pinned[n - 1] << std::endl;

  bool ok = true;
  for (int i = 0; i < n; ++i) {
    float expected = static_cast<float>(i) * 2.0f;
    if (std::fabs(host_pinned[i] - expected) > 1e-5f) {
      std::cerr << "Mismatch at " << i << ": got " << host_pinned[i]
                << ", expected " << expected << std::endl;
      ok = false;
      break;
    }
  }

  CHECK_CUDA(cudaFreeHost(host_pinned));

  if (!ok) {
    return EXIT_FAILURE;
  }

  std::cout << "Success: CPU could access pinned memory before and after GPU use."
            << std::endl;
  return EXIT_SUCCESS;
}
