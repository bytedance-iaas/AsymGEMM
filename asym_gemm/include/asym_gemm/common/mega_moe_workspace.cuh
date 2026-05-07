#pragma once
// Stage 1 scaffolding — single-GPU analogue of DeepGEMM's layout::Workspace.
//
// Holds every intermediate tensor the fused mega-MoE kernel will need *and*
// the per-block arrival signals that let L1 and L2 overlap on the M
// dimension (see asym_moe_update.md §1.3 and §1.7-§1.8).
//
// Stage 1: host-side sizing helper + device accessors.  No consumer yet.
//          asym_gemm/__init__.py will export a single-shot allocator.
//
// Layout (flat uint8 region, 16-byte aligned sections):
//
//   [0..16)                                grid_sync_counters[4]   uint32
//   [16..  +num_max_pool_blocks*4)          l1_arrival_count[]      uint32
//   [  ..  +num_max_pool_blocks*8)          l2_arrival_mask[]       uint64
//   [  ..  +num_max_pool_tokens*H)          l2_act_buffer           FP8 E4M3
//   [  ..  +num_max_pool_tokens*H/128*4)    l2_sf_buffer            UE8M0 pack
//   [  ..  +num_max_pool_tokens*8)          token_src_metadata[]    int2 (orig, topk_k)
//   [  ..  +num_max_pool_tokens*4)          l1_topk_weights[]       float
//   [  ..  +num_topk*num_tokens*H*2)        combine_buffer          BF16
//
// Each section is aligned up to 16 bytes so TMA descriptors over them
// satisfy the 16-byte alignment requirement.

#include <cstdint>
#include <asym_gemm/common/utils.cuh>

namespace asym_gemm {

struct MegaMoEWorkspace {
    void*    base;
    uint32_t num_experts;
    uint32_t num_topk;
    uint32_t num_tokens;           // original (pre-dispatch) token count
    uint32_t num_max_pool_tokens;  // M_total after dispatch, padded
    uint32_t num_max_pool_blocks;  // num_max_pool_tokens / BLOCK_M
    uint32_t hidden;
    uint32_t intermediate_hidden;

    float    activation_clamp;  // SwiGLU gate/up clamp (0.0 = no clamping)

    // Offsets (bytes) into `base`.  Filled by the host-side constructor.
    uint64_t off_grid_sync;
    uint64_t off_l1_arrival;
    uint64_t off_l2_mask;
    uint64_t off_l2_acts;
    uint64_t off_l2_sf;
    uint64_t off_token_src_map;
    uint64_t off_l1_topk_w;
    uint64_t off_combine;
    uint64_t total_bytes;

    // ---- Host-side sizing ------------------------------------------------
    // Returns the total byte size; call once to allocate the workspace
    // tensor from Python / C++ before kernel launch.  Does not touch `base`.
    __host__ __device__ static constexpr uint64_t align16(uint64_t x) {
        return (x + 15ull) & ~15ull;
    }

    __host__ __device__ static uint64_t compute_sizes(
            uint32_t num_experts,
            uint32_t num_topk,
            uint32_t num_tokens,
            uint32_t num_max_pool_tokens,
            uint32_t hidden,
            uint32_t intermediate_hidden,
            uint32_t block_m,
            uint64_t* out_off_grid_sync = nullptr,
            uint64_t* out_off_l1_arrival = nullptr,
            uint64_t* out_off_l2_mask = nullptr,
            uint64_t* out_off_l2_acts = nullptr,
            uint64_t* out_off_l2_sf = nullptr,
            uint64_t* out_off_token_src_map = nullptr,
            uint64_t* out_off_l1_topk_w = nullptr,
            uint64_t* out_off_combine = nullptr) {
        const uint64_t num_max_pool_blocks = num_max_pool_tokens / block_m;
        uint64_t off = 0;
        if (out_off_grid_sync) *out_off_grid_sync = off;
        off = align16(off + 4ull * sizeof(uint32_t));
        if (out_off_l1_arrival) *out_off_l1_arrival = off;
        off = align16(off + num_max_pool_blocks * sizeof(uint32_t));
        if (out_off_l2_mask) *out_off_l2_mask = off;
        off = align16(off + num_max_pool_blocks * sizeof(uint64_t));
        if (out_off_l2_acts) *out_off_l2_acts = off;
        // Intermediate FP8 activations: [M_total, intermediate_hidden]
        off = align16(off + static_cast<uint64_t>(num_max_pool_tokens) * intermediate_hidden);
        if (out_off_l2_sf) *out_off_l2_sf = off;
        // UE8M0 SF packed 4 per uint32, gran_k=128, so 1 uint32 per 128*4=512 K elements
        // For FP8 (gran_k=128 regardless of BLOCK_K) the SF size per row is
        // intermediate_hidden/128 floats, i.e. intermediate_hidden/128*4 bytes.
        off = align16(off + static_cast<uint64_t>(num_max_pool_tokens) * (intermediate_hidden / 128) * sizeof(uint32_t));
        if (out_off_token_src_map) *out_off_token_src_map = off;
        off = align16(off + static_cast<uint64_t>(num_max_pool_tokens) * 2 * sizeof(int32_t));
        if (out_off_l1_topk_w) *out_off_l1_topk_w = off;
        off = align16(off + static_cast<uint64_t>(num_max_pool_tokens) * sizeof(float));
        if (out_off_combine) *out_off_combine = off;
        // Combine buffer is per-topk-slot so every row gets a deterministic
        // destination without atomics: [num_topk, num_tokens, hidden] BF16.
        off = align16(off + static_cast<uint64_t>(num_topk) * num_tokens * hidden * sizeof(uint16_t));
        return off;
    }

    MegaMoEWorkspace() = default;

    __host__ MegaMoEWorkspace(void* base_,
                              uint32_t num_experts_,
                              uint32_t num_topk_,
                              uint32_t num_tokens_,
                              uint32_t num_max_pool_tokens_,
                              uint32_t hidden_,
                              uint32_t intermediate_hidden_,
                              uint32_t block_m)
        : base(base_),
          num_experts(num_experts_),
          num_topk(num_topk_),
          num_tokens(num_tokens_),
          num_max_pool_tokens(num_max_pool_tokens_),
          num_max_pool_blocks(num_max_pool_tokens_ / block_m),
          hidden(hidden_),
          intermediate_hidden(intermediate_hidden_) {
        total_bytes = compute_sizes(
            num_experts_, num_topk_, num_tokens_, num_max_pool_tokens_,
            hidden_, intermediate_hidden_, block_m,
            &off_grid_sync, &off_l1_arrival, &off_l2_mask, &off_l2_acts,
            &off_l2_sf, &off_token_src_map, &off_l1_topk_w, &off_combine);
    }

    // ---- Device accessors ------------------------------------------------
    __device__ __host__ __forceinline__ uint32_t* get_grid_sync_count_ptr(uint32_t idx = 0) const {
        return reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(base) + off_grid_sync) + idx;
    }
    __device__ __host__ __forceinline__ uint32_t* get_l1_arrival_count_ptr(uint32_t pool_block = 0) const {
        return reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(base) + off_l1_arrival) + pool_block;
    }
    __device__ __host__ __forceinline__ uint64_t* get_l2_arrival_mask_ptr(uint32_t pool_block = 0) const {
        return reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(base) + off_l2_mask) + pool_block;
    }
    __device__ __host__ __forceinline__ void* get_l2_acts_ptr() const {
        return static_cast<uint8_t*>(base) + off_l2_acts;
    }
    __device__ __host__ __forceinline__ uint32_t* get_l2_sf_ptr() const {
        return reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(base) + off_l2_sf);
    }
    // Per-row (orig_token_idx, topk_k) as int2
    struct TokenSrcMetadata { int32_t orig_token_idx; int32_t topk_k; };
    __device__ __host__ __forceinline__ TokenSrcMetadata* get_token_src_metadata_ptr(uint32_t pool_row = 0) const {
        return reinterpret_cast<TokenSrcMetadata*>(static_cast<uint8_t*>(base) + off_token_src_map) + pool_row;
    }
    __device__ __host__ __forceinline__ float* get_l1_topk_weight_ptr(uint32_t pool_row = 0) const {
        return reinterpret_cast<float*>(static_cast<uint8_t*>(base) + off_l1_topk_w) + pool_row;
    }
    // combine_buffer[topk_k, token, :] BF16
    __device__ __host__ __forceinline__ void* get_combine_buffer_ptr() const {
        return static_cast<uint8_t*>(base) + off_combine;
    }
};

// UTCCP 4x32 transpose index mapping for UE8M0 SF storage.
// Maps a pool-relative row index to its storage offset within l2_sf_buffer,
// accounting for the 4×32 transpose done by the UTCCP warp.
// `token_idx_in_expert` : row within this expert's pool region
// `block_m`             : BLOCK_M tile height (must be power-of-two multiple of 128)
// `sf_block_m`          : SF rows per BLOCK_M tile = BLOCK_M / 4  (UE8M0 4x32 layout)
__device__ __host__ __forceinline__
uint32_t mega_moe_transform_sf_token_idx(
        uint32_t token_idx_in_expert, uint32_t block_m, uint32_t sf_block_m) {
    const uint32_t idx = token_idx_in_expert % block_m;
    return token_idx_in_expert / block_m * sf_block_m +
           (idx & ~127u) + (idx & 31u) * 4u + ((idx >> 5) & 3u);
}

} // namespace asym_gemm
