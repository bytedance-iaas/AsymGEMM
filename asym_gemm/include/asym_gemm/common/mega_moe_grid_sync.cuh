#pragma once
// Stage 1 scaffolding — single-GPU grid sync primitive.
//
// Ports the idea of DeepGEMM's `comm::grid_sync<kNumSMs, kIndex>()` without
// the NVLink parts.  A single `uint32` counter per index lets all blocks
// wait until every block has passed the sync point, using a generation-bit
// trick that avoids having to reset the counter across phases.
//
// Usage:
//     mega_grid_sync<kNumSMs, 0>(workspace, sm_idx, thread_idx);
//
// The `sync_scope` in DeepGEMM is a lambda that the caller uses to coordinate
// CTA-local threads before/after the global handshake.  Here we inline a
// plain __syncthreads() since the AsymGEMM mega kernel does not need the
// flexibility DeepGEMM's MoE kernel does for dispatch warps.

#include <cuda_runtime.h>
#include <asym_gemm/common/mega_moe_workspace.cuh>

namespace asym_gemm {

template <uint32_t kNumSMs, uint32_t kIndex = 0>
__device__ __forceinline__ void mega_grid_sync(
        const MegaMoEWorkspace& ws,
        uint32_t sm_idx,
        uint32_t thread_idx) {
    static_assert(kIndex < 4, "only 4 grid-sync counter slots in MegaMoEWorkspace");
    constexpr uint32_t kFinishSumTag = 0x80000000u;

    __syncthreads();
    if (thread_idx == 0) {
        uint32_t* ctr = ws.get_grid_sync_count_ptr(kIndex);
        // SM 0 writes the "completion offset" so the counter wraps exactly
        // to `kFinishSumTag` once all kNumSMs contributions have landed.
        const uint32_t delta = (sm_idx == 0) ? (kFinishSumTag - (kNumSMs - 1u)) : 1u;
        uint32_t old_val = atomicAdd_system(ctr, delta);

        // Spin until the top bit flips — indicates the *generation* advanced
        // because all other SMs have contributed.
        uint32_t new_val;
        do {
            new_val = *reinterpret_cast<volatile uint32_t*>(ctr);
        } while (((new_val ^ old_val) & kFinishSumTag) == 0u);
    }
    __syncthreads();
}

} // namespace asym_gemm
