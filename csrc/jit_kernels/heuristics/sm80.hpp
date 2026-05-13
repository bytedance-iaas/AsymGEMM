// csrc/jit_kernels/heuristics/sm80.hpp
#pragma once

#include <cstdint>

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

    // Grid is (ceil_div(N, block_n), list_size): one CTA per (N-tile, expert)
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

// ── FP8 helpers (SM89 native FP8 MMA, K-outer M-inner loop) ─────────────────
//
// Smem formula (FP8 for sX+sW, BF16 for sO):
//   sX: BLOCK_M * BLOCK_K * 1 B
//   sW: BLOCK_N * BLOCK_K * 1 B
//   sO: BLOCK_M * BLOCK_N * 2 B  (output staging + partial-sum seed read-back)
//
// For BLOCK_M=BLOCK_N=128:
//   Total = (128+128)*BLOCK_K + 128*128*2 = 256*BLOCK_K + 32768
//   SM89 (96 KB = 98304 B): BLOCK_K ≤ (98304-32768)/256 = 256
inline int smem_bytes_fp8(uint32_t block_m, uint32_t block_n, uint32_t block_k) {
    return static_cast<int>((block_m + block_n) * block_k          // sX+sW: FP8
                            + block_m * block_n * 2                // sO: BF16
                            + block_m * 4);                        // smem_sa: per-token scales
}

// Max BLOCK_K for the FP8 kernel given the arch's smem limit.
// Derived from smem_bytes_fp8(128,128,block_k) <= limit with BLOCK_M=BLOCK_N=128.
inline int max_block_k_fp8(int arch_major, int arch_minor) {
    return (smem_limit(arch_major, arch_minor) - 32768) / 256;
}

// Config selector for the FP8 kernel.
// BLOCK_K min = 32 (SM89 FP8 MMA K-atom = 32, must be a multiple of 32).
inline SM80GemmConfig select_sm80_fp8_config(int arch_major, int arch_minor,
                                             int N, int K) {
    const int smem_cap = smem_limit(arch_major, arch_minor);

    // Start at arch max, halve until K is divisible; floor at 32
    uint32_t block_k = static_cast<uint32_t>(max_block_k_fp8(arch_major, arch_minor));
    while (block_k > 32u && K % static_cast<int>(block_k) != 0)
        block_k /= 2u;

    // Largest block_n in {128, 64, 32} that divides N and fits in smem
    uint32_t block_n = 32u;
    for (uint32_t bn : {128u, 64u, 32u}) {
        if (N % static_cast<int>(bn) != 0) continue;
        if (smem_bytes_fp8(128u, bn, block_k) <= smem_cap) {
            block_n = bn;
            break;
        }
    }

    return SM80GemmConfig{128u, block_n, block_k, 4u};
}

}  // namespace sm80
}  // namespace asym_gemm
