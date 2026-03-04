#include <cuda_runtime.h>
#include <cstdio>

struct HostQueue {
  int* buf;        // device pointer that maps host pinned memory
  int  cap;
  int* tail;       // mapped host pinned int
};

__global__ void gpu_enqueue(HostQueue q, int v) {
  int t = atomicAdd(q.tail, 1);
  q.buf[t % q.cap] = v;
  __threadfence_system(); // make visible to CPU without needing a full device sync
}

int main() {
  const int CAP = 1024;

  int *h_buf=nullptr, *h_tail=nullptr;
  cudaHostAlloc(&h_buf,  CAP*sizeof(int), cudaHostAllocMapped);
  cudaHostAlloc(&h_tail, sizeof(int),    cudaHostAllocMapped);
  *h_tail = 0;

  int *d_buf=nullptr, *d_tail=nullptr;
  cudaHostGetDevicePointer(&d_buf,  h_buf,  0);
  cudaHostGetDevicePointer(&d_tail, h_tail, 0);

  HostQueue q{d_buf, CAP, d_tail};

  gpu_enqueue<<<1,1>>>(q, 123);

  // If you want immediate CPU observation without cudaDeviceSynchronize,
  // you can poll *h_tail; otherwise just sync:
  cudaDeviceSynchronize();

  printf("tail=%d buf[0]=%d\n", *h_tail, h_buf[0]);

  cudaFreeHost(h_buf);
  cudaFreeHost(h_tail);
}