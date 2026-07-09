// sEP S2b cross-process completion + gather (fix_gb200_ep.md S2b contract).
//
// Stream-ordered, ZERO host syncs:
//   thief stream : [ep_steal GEMM] -> [flag_set kernel]   (st.release.sys => the GEMM's
//                  sysmem staging stores are visible wherever the flag reads 1)
//   owner stream : [spin_gather kernel] -> [d consumers]  (spins on the flag — coherent
//                  sysmem read, ~us when already set — then gathers ONLY the stolen
//                  item tiles from the fabric staging into local d)
//
// Steal ranges are CONTIGUOUS in item space by the front/back queue construction; the
// meeting point may split a segment MID-WAY, so the gather works at (segment, n_block)
// ITEM-TILE granularity: rows from the offsets pairs, columns = the item's n-block strip.
// n_blk and BLOCK_N are recovered in-kernel from the final counters:
//   n_blk = (head + tail) / n_segments,   block_n = ceil(N / n_blk).

#include <cuda_bf16.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

namespace asym_gemm::ep_steal {

__global__ void flag_set_kernel(int* flag) {
    // release at system scope: prior stream work (the steal GEMM's sysmem TMA stores)
    // is complete when this kernel runs; the release makes the flag write ordered
    // AFTER those stores for any system-scope observer.
    asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(flag), "r"(1) : "memory");
}

__device__ __forceinline__ int load_acquire_sys(const int* ptr) {
    int value;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
    return value;
}

__global__ void spin_gather_kernel(__nv_bfloat16* __restrict__ d,
                                   const __nv_bfloat16* __restrict__ staging,
                                   const int* __restrict__ offsets,  // device int32[2*n_segments]
                                   const int* __restrict__ counters, // pinned [claimed, head, tail]
                                   const int* __restrict__ flag,     // pinned, set by the peer
                                   const int n,
                                   const int n_blk,                  // from the steal launcher
                                   const int n_segments,
                                   const int n_own_segments,
                                   const int side) {
    const int total_items = n_segments * n_blk;
    const int n_own_items = n_own_segments * n_blk;

    // EARLY EXIT (fix_gb200_ep.md D-GATHER): my OWN side's counter is final (my GEMM
    // precedes this kernel in-stream). If I claimed my whole section, nothing of mine
    // was stolen => no wait, no gather — the balanced path costs ~one sysmem read.
    if (side == 0) {
        if (load_acquire_sys(counters + 1) >= n_own_items)
            return;
    } else {
        if (load_acquire_sys(counters + 2) >= total_items - n_own_items)
            return;
    }

    // some of my items were claimed by the peer: wait for its completion flag
    // (release-ordered after its GEMM's staging stores), then gather the contiguous
    // stolen range at item-tile granularity.
    while (load_acquire_sys(flag) == 0)
        __nanosleep(256);

    const int head = load_acquire_sys(counters + 1);
    const int tail = load_acquire_sys(counters + 2);

    int steal_lo, steal_hi;
    if (side == 0) {
        steal_lo = total_items - tail;             // peer(1) claimed [total-tail, total)
        if (steal_lo >= n_own_items) return;
        steal_hi = n_own_items;
    } else {
        steal_lo = n_own_items;                    // peer(0) claimed [0, head)
        steal_hi = head;                           // crossed if head > n_own_items
        if (steal_hi <= steal_lo) return;
    }

    const int block_n = (n + n_blk - 1) / n_blk;
    // grid-stride over stolen item tiles; one thread block per tile chunk of rows
    for (int item = steal_lo + static_cast<int>(blockIdx.x); item < steal_hi;
         item += static_cast<int>(gridDim.x)) {
        const int segment = item / n_blk;
        const int col_blk = item % n_blk;
        const int row_lo = offsets[2 * segment];
        const int row_hi = offsets[2 * segment + 1];
        const int col_lo = col_blk * block_n;
        const int col_hi = min(n, col_lo + block_n);
        const int cols = col_hi - col_lo;
        if (cols <= 0) continue;
        const long long elems = static_cast<long long>(row_hi - row_lo) * cols;
        for (long long e = threadIdx.x; e < elems; e += blockDim.x) {
            const int r = row_lo + static_cast<int>(e / cols);
            const int c = col_lo + static_cast<int>(e % cols);
            d[static_cast<long long>(r) * n + c] = staging[static_cast<long long>(r) * n + c];
        }
    }
}

void flag_set(const torch::Tensor& flag_page, const int64_t index) {
    TORCH_CHECK(flag_page.device().is_cpu() && flag_page.is_pinned(), "flag page must be pinned CPU");
    TORCH_CHECK(flag_page.scalar_type() == torch::kInt, "flag page must be int32");
    TORCH_CHECK(index >= 0 && index < flag_page.numel(), "flag index out of range");
    auto stream = at::cuda::getCurrentCUDAStream();
    flag_set_kernel<<<1, 1, 0, stream.stream()>>>(flag_page.data_ptr<int>() + index);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void spin_gather(const torch::Tensor& d, const torch::Tensor& staging,
                 const torch::Tensor& offsets, const torch::Tensor& counters,
                 const torch::Tensor& flag_page, const int64_t flag_index,
                 const int64_t n_blk, const int64_t n_segments,
                 const int64_t n_own_segments, const int64_t side) {
    TORCH_CHECK(d.is_cuda() && d.scalar_type() == torch::kBFloat16 && d.is_contiguous());
    TORCH_CHECK(staging.device().is_cpu() && staging.is_pinned() && staging.is_contiguous());
    TORCH_CHECK(staging.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(staging.sizes() == d.sizes(), "staging must mirror d's shape");
    TORCH_CHECK(offsets.is_cuda() && offsets.scalar_type() == torch::kInt);
    TORCH_CHECK(offsets.numel() >= 2 * n_segments, "offsets too small");
    TORCH_CHECK(counters.device().is_cpu() && counters.is_pinned() && counters.scalar_type() == torch::kInt);
    TORCH_CHECK(flag_page.device().is_cpu() && flag_page.is_pinned() && flag_page.scalar_type() == torch::kInt);
    TORCH_CHECK(flag_index >= 0 && flag_index < flag_page.numel());
    TORCH_CHECK(side == 0 || side == 1);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = 32;
    spin_gather_kernel<<<blocks, 256, 0, stream.stream()>>>(
        reinterpret_cast<__nv_bfloat16*>(d.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(staging.data_ptr()),
        offsets.data_ptr<int>(),
        counters.data_ptr<int>(),
        flag_page.data_ptr<int>() + flag_index,
        static_cast<int>(d.size(1)),
        static_cast<int>(n_blk),
        static_cast<int>(n_segments),
        static_cast<int>(n_own_segments),
        static_cast<int>(side));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void register_apis(pybind11::module_& m) {
    m.def("ep_steal_flag_set", &flag_set, pybind11::arg("flag_page"), pybind11::arg("index"));
    m.def("ep_steal_spin_gather", &spin_gather,
          pybind11::arg("d"), pybind11::arg("staging"), pybind11::arg("offsets"),
          pybind11::arg("counters"), pybind11::arg("flag_page"), pybind11::arg("flag_index"),
          pybind11::arg("n_blk"), pybind11::arg("n_segments"),
          pybind11::arg("n_own_segments"), pybind11::arg("side"));
}

}  // namespace asym_gemm::ep_steal
