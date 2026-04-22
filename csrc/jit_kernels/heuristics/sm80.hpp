// csrc/jit_kernels/heuristics/sm80.hpp
#pragma once

#include <cstdint>
#include <utility>
#include <string>

namespace asym_gemm {
namespace sm80 {

struct SM80GemmConfig {
    uint32_t block_m;   // always 128
    uint32_t block_n;   // 128, 64, or 32 (largest that divides N)
    uint32_t block_k;   // largest power-of-2 ≤ arch_max_block_k that divides K, min 64
    uint32_t nwarps;    // always 4

    int num_threads() const { return static_cast<int>(nwarps * 32); }

    // sX [BLOCK_M,BLOCK_K] + sW [BLOCK_N,BLOCK_K] + sO [BLOCK_M,BLOCK_N], 2 bytes each
    int smem_bytes() const {
        return static_cast<int>(block_m * block_k + block_n * block_k + block_m * block_n) * 2;
    }

    // Grid is (ceil_div(N, block_n), 1): one CTA per N-tile, experts processed serially
    int grid_x(int N) const { return (N + static_cast<int>(block_n) - 1) / static_cast<int>(block_n); }
};

// Smem capacity per arch
// SM80 (A100):    163840 B (160 KB)
// SM89 (RTX4090):  98304 B ( 96 KB)
// SM100 (GB200):  163840 B (160 KB)
inline int smem_limit(int arch_major, int arch_minor) {
    if (arch_major == 8 && arch_minor == 9)
        return 98304;
    return 163840;
}

// Max BLOCK_K given smem limit and fixed BLOCK_M=128, BLOCK_N=128
// smem(BLOCK_K) = (128*BLOCK_K + 128*BLOCK_K + 128*128)*2 = 512*BLOCK_K + 32768
// BLOCK_K ≤ (smem_limit - 32768) / 512
// SM80/SM100: BLOCK_K ≤ (163840-32768)/512 = 256
// SM89:       BLOCK_K ≤ ( 98304-32768)/512 = 128
inline int max_block_k(int arch_major, int arch_minor) {
    int cap = smem_limit(arch_major, arch_minor);
    return (cap - 32768) / 512;  // = 256 or 128
}

// Select the largest block_n in {128, 64, 32} that divides N and fits in smem with given block_k
inline uint32_t pick_block_n(int N, uint32_t block_k, int smem_cap) {
    for (uint32_t bn : {128u, 64u, 32u}) {
        if (N % static_cast<int>(bn) != 0) continue;
        int smem = static_cast<int>(128u * block_k + bn * block_k + 128u * bn) * 2;
        if (smem <= smem_cap) return bn;
    }
    return 32u;  // fallback — caller should validate N % 32 == 0
}

// Main entry point called from the API layer.
// arch_major / arch_minor from device_runtime->get_arch_pair().
// N: output column dimension. K: inner dimension (must be ≥ 64, multiple of 16).
inline SM80GemmConfig select_sm80_config(int arch_major, int arch_minor, int N, int K) {
    const int smem_cap = smem_limit(arch_major, arch_minor);

    // Select block_k: start at arch max, halve until K is divisible, min = 64
    uint32_t block_k = static_cast<uint32_t>(max_block_k(arch_major, arch_minor));
    while (block_k > 64u && K % static_cast<int>(block_k) != 0)
        block_k /= 2u;

    // Select block_n
    uint32_t block_n = pick_block_n(N, block_k, smem_cap);

    return SM80GemmConfig{128u, block_n, block_k, 4u};
}

}  // namespace sm80
}  // namespace asym_gemm
