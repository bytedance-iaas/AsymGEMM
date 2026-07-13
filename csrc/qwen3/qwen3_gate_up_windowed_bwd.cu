// Copyright (c) 2026.

#include "../apis/qwen3_moe.hpp"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace asym_gemm::qwen3_moe {
namespace {

constexpr int kThreads = 256;

struct CudaTimer {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    cudaStream_t stream{};

    explicit CudaTimer(cudaStream_t stream_) : stream(stream_) {
        C10_CUDA_CHECK(cudaEventCreate(&start));
        C10_CUDA_CHECK(cudaEventCreate(&stop));
        C10_CUDA_CHECK(cudaEventRecord(start, stream));
    }

    float elapsed_ms() {
        C10_CUDA_CHECK(cudaEventRecord(stop, stream));
        C10_CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        C10_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }

    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};

template <typename T>
__device__ __forceinline__ float to_float(T v) {
    return static_cast<float>(v);
}

__device__ __forceinline__ at::BFloat16 to_bf16(float v) {
    return static_cast<at::BFloat16>(v);
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu_grad(float x) {
    const float sig = 1.0f / (1.0f + expf(-x));
    return sig * (1.0f + x * (1.0f - sig));
}

__global__ void fill_row_metadata_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ experts,
    int32_t* __restrict__ row_to_e_rel,
    int32_t* __restrict__ row_to_expert,
    int32_t g_start,
    int32_t g_chunk) {
    const int32_t g_rel = static_cast<int32_t>(blockIdx.x);
    if (g_rel >= g_chunk) return;
    const int32_t g = g_start + g_rel;
    const int32_t row_start = offsets[g_start];
    const int32_t begin = offsets[g] - row_start;
    const int32_t end = offsets[g + 1] - row_start;
    const int32_t expert = experts[g];
    for (int32_t local = begin + static_cast<int32_t>(threadIdx.x);
         local < end;
         local += static_cast<int32_t>(blockDim.x)) {
        row_to_e_rel[local] = g_rel;
        row_to_expert[local] = expert;
    }
}

__global__ void fill_w_cache_kernel(
    const at::BFloat16* __restrict__ gate_up_weight_cpu,
    const int32_t* __restrict__ experts,
    at::BFloat16* __restrict__ w_cache,
    int32_t g_start,
    int32_t g_chunk,
    int32_t e_total,
    int32_t i_total,
    int32_t h_total,
    int32_t iwin0,
    int32_t p,
    int32_t kdx) {
    const int64_t total = static_cast<int64_t>(g_chunk) * kdx * h_total;
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;

    const int32_t h = static_cast<int32_t>(linear % h_total);
    const int64_t tmp = linear / h_total;
    const int32_t n = static_cast<int32_t>(tmp % kdx);
    const int32_t e_rel = static_cast<int32_t>(tmp / kdx);
    const int32_t expert = experts[g_start + e_rel];

    const int32_t bn = 2 * p;
    const int32_t q_rel = n / bn;
    const int32_t in_pair = n - q_rel * bn;
    const bool is_up = in_pair >= p;
    const int32_t i = iwin0 + q_rel * p + (is_up ? in_pair - p : in_pair);

    at::BFloat16 value = to_bf16(0.0f);
    if (expert >= 0 && expert < e_total && i >= 0 && i < i_total) {
        const int32_t weight_i = (is_up ? i_total + i : i);
        const int64_t src = (static_cast<int64_t>(expert) * (2 * i_total) + weight_i) * h_total + h;
        value = gate_up_weight_cpu[src];
    }
    w_cache[linear] = value;
}

__global__ void recompute_activation_kernel(
    const at::BFloat16* __restrict__ x_sel,
    const at::BFloat16* __restrict__ dact_sel,
    const at::BFloat16* __restrict__ dS_down_sel,
    const at::BFloat16* __restrict__ gate_low_rank_sel,
    const at::BFloat16* __restrict__ up_low_rank_sel,
    const at::BFloat16* __restrict__ gate_lora_B,
    const at::BFloat16* __restrict__ up_lora_B,
    const at::BFloat16* __restrict__ w_cache,
    const uint8_t* __restrict__ down_mask_packed,
    const int32_t* __restrict__ row_to_e_rel,
    const int32_t* __restrict__ row_to_expert,
    at::BFloat16* __restrict__ grad_pair_window,
    at::BFloat16* __restrict__ grad_gate_sel,
    at::BFloat16* __restrict__ grad_up_sel,
    float* __restrict__ grad_down_lora_A_acc,
    int32_t row_start,
    int32_t m_work,
    int32_t e_total,
    int32_t h_total,
    int32_t i_total,
    int32_t r_total,
    int32_t r_down,
    int32_t down_mask_packed_width,
    int32_t iwin0,
    int32_t p,
    int32_t kdx,
    float lora_scale,
    float down_dropout_scale,
    bool compute_down_lora_A) {
    const int64_t total = static_cast<int64_t>(m_work) * kdx;
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;

    const int32_t n = static_cast<int32_t>(linear % kdx);
    const int32_t row_local = static_cast<int32_t>(linear / kdx);
    const int32_t row_global = row_start + row_local;
    const int32_t e_rel = row_to_e_rel[row_local];
    const int32_t expert = row_to_expert[row_local];

    const int32_t bn = 2 * p;
    const int32_t q_rel = n / bn;
    const int32_t in_pair = n - q_rel * bn;
    const bool is_up = in_pair >= p;
    const int32_t i = iwin0 + q_rel * p + (is_up ? in_pair - p : in_pair);

    float grad_value = 0.0f;
    if (expert >= 0 && expert < e_total && i >= 0 && i < i_total) {
        float base = 0.0f;
        const int64_t w_base = (static_cast<int64_t>(e_rel) * kdx + n) * h_total;
        const int64_t x_base = static_cast<int64_t>(row_global) * h_total;
        for (int32_t h = 0; h < h_total; ++h) {
            base += to_float(x_sel[x_base + h]) * to_float(w_cache[w_base + h]);
        }

        float delta = 0.0f;
        const int64_t lr_base = static_cast<int64_t>(row_global) * r_total;
        const int64_t b_base = (static_cast<int64_t>(expert) * i_total + i) * r_total;
        for (int32_t r = 0; r < r_total; ++r) {
            const float s = is_up ? to_float(up_low_rank_sel[lr_base + r])
                                  : to_float(gate_low_rank_sel[lr_base + r]);
            const float b = is_up ? to_float(up_lora_B[b_base + r])
                                  : to_float(gate_lora_B[b_base + r]);
            delta += s * b;
        }

        const float branch_value = base + lora_scale * delta;
        const int32_t peer_n = is_up ? (n - p) : (n + p);
        float peer_base = 0.0f;
        const int64_t peer_w_base = (static_cast<int64_t>(e_rel) * kdx + peer_n) * h_total;
        const int64_t x_peer_base = static_cast<int64_t>(row_global) * h_total;
        for (int32_t h = 0; h < h_total; ++h) {
            peer_base += to_float(x_sel[x_peer_base + h]) * to_float(w_cache[peer_w_base + h]);
        }
        const int64_t peer_b_base = (static_cast<int64_t>(expert) * i_total + i) * r_total;
        float peer_delta = 0.0f;
        for (int32_t r = 0; r < r_total; ++r) {
            const float s = is_up ? to_float(gate_low_rank_sel[lr_base + r])
                                  : to_float(up_low_rank_sel[lr_base + r]);
            const float b = is_up ? to_float(gate_lora_B[peer_b_base + r])
                                  : to_float(up_lora_B[peer_b_base + r]);
            peer_delta += s * b;
        }
        const float peer_value = peer_base + lora_scale * peer_delta;
        const float gate = is_up ? peer_value : branch_value;
        const float up = is_up ? branch_value : peer_value;
        const float silu_gate = silu(gate);
        const float act = silu_gate * up;
        const float dact = to_float(dact_sel[static_cast<int64_t>(row_global) * i_total + i]);
        grad_value = is_up ? dact * silu_gate : dact * up * silu_grad(gate);

        if (!is_up) {
            grad_gate_sel[static_cast<int64_t>(row_global) * i_total + i] = to_bf16(grad_value);
            if (compute_down_lora_A) {
                float dropped_act = act;
                if (down_mask_packed != nullptr) {
                    const uint8_t byte = down_mask_packed[
                        static_cast<int64_t>(row_global) * down_mask_packed_width + (i >> 3)];
                    const bool keep = ((byte >> (i & 7)) & 1u) != 0;
                    dropped_act = keep ? act * down_dropout_scale : 0.0f;
                }
                const int64_t ds_base = static_cast<int64_t>(row_global) * r_down;
                const int64_t da_base = static_cast<int64_t>(expert) * r_down * i_total + i;
                for (int32_t rd = 0; rd < r_down; ++rd) {
                    const float ds = to_float(dS_down_sel[ds_base + rd]);
                    atomicAdd(&grad_down_lora_A_acc[da_base + static_cast<int64_t>(rd) * i_total], ds * dropped_act);
                }
            }
        } else {
            grad_up_sel[static_cast<int64_t>(row_global) * i_total + i] = to_bf16(grad_value);
        }
    }

    grad_pair_window[linear] = to_bf16(grad_value);
}

__global__ void dx_window_kernel(
    const at::BFloat16* __restrict__ grad_pair_window,
    const at::BFloat16* __restrict__ w_cache,
    const int32_t* __restrict__ row_to_e_rel,
    float* __restrict__ dx_acc,
    int32_t row_start,
    int32_t m_work,
    int32_t h_total,
    int32_t kdx) {
    const int64_t total = static_cast<int64_t>(m_work) * h_total;
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;

    const int32_t h = static_cast<int32_t>(linear % h_total);
    const int32_t row_local = static_cast<int32_t>(linear / h_total);
    const int32_t row_global = row_start + row_local;
    const int32_t e_rel = row_to_e_rel[row_local];
    float acc = 0.0f;
    const int64_t gp_base = static_cast<int64_t>(row_local) * kdx;
    const int64_t wc_base = static_cast<int64_t>(e_rel) * kdx * h_total + h;
    for (int32_t n = 0; n < kdx; ++n) {
        acc += to_float(grad_pair_window[gp_base + n]) * to_float(w_cache[wc_base + static_cast<int64_t>(n) * h_total]);
    }
    dx_acc[static_cast<int64_t>(row_global) * h_total + h] += acc;
}

__global__ void cast_fp32_to_bf16_kernel(
    const float* __restrict__ src,
    at::BFloat16* __restrict__ dst,
    int64_t total) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    dst[linear] = to_bf16(src[linear]);
}

int blocks_for(int64_t total) {
    if (total <= 0) return 0;
    return static_cast<int>((total + kThreads - 1) / kThreads);
}

void check_cuda_bf16_2d(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must be BF16");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_cuda_bf16_3d(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.dim() == 3, name, " must be 3D");
    TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must be BF16");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_cuda_uint8_2d(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == torch::kUInt8, name, " must be uint8");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_offsets_experts(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.dim() == 1, name, " must be 1D");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == torch::kInt32 || t.scalar_type() == torch::kInt64,
                name, " must be int32 or int64");
}

pybind11::dict empty_stats() {
    pybind11::dict stats;
    stats["mode"] = "cache_first_window";
    stats["native_calls"] = 1;
    stats["selected_recompute_rows"] = 0;
    stats["cpu_weight_bytes_staged"] = 0;
    stats["expected_cpu_weight_bytes_min"] = 0;
    stats["cpu_weight_stream_multiplier"] = 0.0;
    stats["hbm_w_cache_write_bytes"] = 0;
    stats["hbm_w_cache_valid_write_bytes"] = 0;
    stats["hbm_w_cache_padding_zero_write_bytes"] = 0;
    stats["recompute_reads_w_cache_bytes"] = 0;
    stats["dx_reads_w_cache_bytes"] = 0;
    stats["w_cache_bytes_allocated_peak"] = 0;
    stats["grad_pair_window_bytes_allocated_peak"] = 0;
    stats["dx_acc_bytes_allocated_peak"] = 0;
    stats["metadata_ms"] = 0.0;
    stats["fill_w_cache_ms"] = 0.0;
    stats["recompute_grad_window_ms"] = 0.0;
    stats["dx_window_ms"] = 0.0;
    stats["native_total_ms"] = 0.0;
    return stats;
}

}  // namespace

// Not on the current best benchmark paths; retained for native Qwen3 windowed recompute experiments.
pybind11::tuple qwen3_gate_up_recompute_bwd_sm100_bf16_windowed(
    const torch::Tensor& x_sel,
    const torch::Tensor& dact_sel,
    const torch::Tensor& gate_low_rank_sel,
    const torch::Tensor& up_low_rank_sel,
    const torch::Tensor& gate_lora_B,
    const torch::Tensor& up_lora_B,
    const torch::Tensor& gate_up_weight_cpu,
    const torch::Tensor& selected_offsets,
    const torch::Tensor& selected_experts,
    int64_t p,
    int64_t q,
    int64_t bm,
    int64_t bk,
    int64_t g_work,
    double lora_scale,
    const std::string& mode,
    bool return_stats,
    const torch::Tensor& dS_down_sel,
    const torch::Tensor& down_mask_packed,
    double down_dropout_p) {
    namespace py = pybind11;
    check_cuda_bf16_2d(x_sel, "x_sel");
    check_cuda_bf16_2d(dact_sel, "dact_sel");
    check_cuda_bf16_2d(gate_low_rank_sel, "gate_low_rank_sel");
    check_cuda_bf16_2d(up_low_rank_sel, "up_low_rank_sel");
    check_cuda_bf16_3d(gate_lora_B, "gate_lora_B");
    check_cuda_bf16_3d(up_lora_B, "up_lora_B");
    check_offsets_experts(selected_offsets, "selected_offsets");
    check_offsets_experts(selected_experts, "selected_experts");
    TORCH_CHECK(mode == "cache_first_window", "initial native implementation supports mode=cache_first_window only");
    TORCH_CHECK(down_dropout_p >= 0.0 && down_dropout_p < 1.0,
                "down_dropout_p must satisfy 0.0 <= p < 1.0");
    TORCH_CHECK(gate_up_weight_cpu.device().is_cpu(), "gate_up_weight_cpu must be CPU");
    TORCH_CHECK(gate_up_weight_cpu.scalar_type() == torch::kBFloat16, "gate_up_weight_cpu must be BF16");
    TORCH_CHECK(gate_up_weight_cpu.dim() == 3, "gate_up_weight_cpu must be [E,2I,H]");
    TORCH_CHECK(gate_up_weight_cpu.is_contiguous(), "gate_up_weight_cpu must be contiguous");
    TORCH_CHECK(gate_up_weight_cpu.is_pinned(), "gate_up_weight_cpu must be pinned CPU memory");
    TORCH_CHECK(x_sel.device() == dact_sel.device(), "x_sel and dact_sel must be on same device");
    TORCH_CHECK(x_sel.device() == gate_low_rank_sel.device(), "low-rank tensors must be on same device");
    TORCH_CHECK(x_sel.device() == gate_lora_B.device(), "LoRA-B tensors must be on same device");
    TORCH_CHECK(x_sel.device() == selected_offsets.device(), "metadata must be on same CUDA device");

    const c10::cuda::CUDAGuard device_guard(x_sel.device());
    const auto props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props != nullptr && props->major == 10, "qwen3 native windowed backward requires SM100");

    TORCH_CHECK(p > 0 && q > 0 && bm > 0 && bk > 0 && g_work > 0, "p/q/bm/bk/g_work must be positive");
    const int64_t m_selected = x_sel.size(0);
    const int64_t h = x_sel.size(1);
    const int64_t i = dact_sel.size(1);
    const int64_t r = gate_low_rank_sel.size(1);
    const int64_t e = gate_up_weight_cpu.size(0);
    const bool compute_down_lora_A = dS_down_sel.defined() && dS_down_sel.numel() > 0;
    int64_t r_down = 0;
    int64_t down_mask_packed_width = 0;
    TORCH_CHECK(dact_sel.size(0) == m_selected, "dact_sel row mismatch");
    TORCH_CHECK(gate_low_rank_sel.size(0) == m_selected && up_low_rank_sel.sizes() == gate_low_rank_sel.sizes(),
                "low-rank tensor shape mismatch");
    TORCH_CHECK(gate_up_weight_cpu.size(1) == 2 * i && gate_up_weight_cpu.size(2) == h,
                "gate_up_weight_cpu shape must be [E,2I,H]");
    TORCH_CHECK(gate_lora_B.size(0) == e && gate_lora_B.size(1) == i && gate_lora_B.size(2) == r,
                "gate_lora_B shape mismatch");
    TORCH_CHECK(up_lora_B.sizes() == gate_lora_B.sizes(), "up_lora_B shape mismatch");
    if (compute_down_lora_A) {
        check_cuda_bf16_2d(dS_down_sel, "dS_down_sel");
        TORCH_CHECK(dS_down_sel.device() == x_sel.device(), "dS_down_sel must be on same CUDA device as x_sel");
        TORCH_CHECK(dS_down_sel.size(0) == m_selected, "dS_down_sel row mismatch");
        r_down = dS_down_sel.size(1);
        TORCH_CHECK(r_down > 0, "dS_down_sel must have nonzero rank width");
        if (down_dropout_p > 0.0) {
            TORCH_CHECK(down_mask_packed.defined() && down_mask_packed.numel() > 0,
                        "down_mask_packed is required when down_dropout_p > 0");
            check_cuda_uint8_2d(down_mask_packed, "down_mask_packed");
            TORCH_CHECK(down_mask_packed.device() == x_sel.device(),
                        "down_mask_packed must be on same CUDA device as x_sel");
            TORCH_CHECK(down_mask_packed.size(0) == m_selected, "down_mask_packed row mismatch");
            down_mask_packed_width = (i + 7) / 8;
            TORCH_CHECK(down_mask_packed.size(1) == down_mask_packed_width,
                        "down_mask_packed width mismatch");
        }
    } else {
        TORCH_CHECK(down_dropout_p == 0.0,
                    "down_dropout_p requires dS_down_sel so native can compute dA_down");
    }
    TORCH_CHECK(selected_offsets.numel() == selected_experts.numel(), "selected_offsets/expert length mismatch");
    TORCH_CHECK(selected_offsets.numel() >= 1, "selected_offsets must include at least sentinel");

    auto options_bf16 = x_sel.options().dtype(torch::kBFloat16);
    auto options_fp32 = x_sel.options().dtype(torch::kFloat32);
    auto grad_x_base_sel = torch::empty({m_selected, h}, options_bf16);
    auto grad_gate_sel = torch::zeros({m_selected, i}, options_bf16);
    auto grad_up_sel = torch::zeros({m_selected, i}, options_bf16);
    auto grad_down_lora_A_sel = compute_down_lora_A
        ? torch::empty({e, r_down, i}, options_bf16)
        : torch::Tensor();
    py::dict stats = empty_stats();
    stats["mode"] = mode;
    stats["p"] = p;
    stats["q"] = q;
    stats["w"] = p * q;
    stats["kdx"] = 2 * p * q;
    stats["bm"] = bm;
    stats["bk"] = bk;
    stats["g_work"] = g_work;
    stats["selected_recompute_rows"] = m_selected;
    stats["down_lora_A_enabled"] = compute_down_lora_A;
    stats["down_lora_A_rank"] = r_down;
    stats["down_dropout_p"] = down_dropout_p;

    if (m_selected == 0 || selected_offsets.numel() <= 1) {
        grad_x_base_sel.zero_();
        if (compute_down_lora_A) {
            grad_down_lora_A_sel.zero_();
            return py::make_tuple(grad_x_base_sel, grad_gate_sel, grad_up_sel, grad_down_lora_A_sel, return_stats ? stats : py::dict());
        }
        return py::make_tuple(grad_x_base_sel, grad_gate_sel, grad_up_sel, return_stats ? stats : py::dict());
    }

    auto offsets_i32 = selected_offsets.scalar_type() == torch::kInt32
        ? selected_offsets
        : selected_offsets.to(selected_offsets.options().dtype(torch::kInt32));
    auto experts_i32 = selected_experts.scalar_type() == torch::kInt32
        ? selected_experts
        : selected_experts.to(selected_experts.options().dtype(torch::kInt32));

    const int64_t gs = selected_offsets.numel() - 1;
    const int64_t kdx = 2 * p * q;
    TORCH_CHECK(kdx <= 2 * i, "kdx must be <= 2I");

    auto offsets_cpu = offsets_i32.to(torch::kCPU);
    auto offsets_acc = offsets_cpu.accessor<int32_t, 1>();
    TORCH_CHECK(offsets_acc[0] == 0, "selected_offsets[0] must be zero");
    TORCH_CHECK(offsets_acc[gs] == m_selected, "selected_offsets[-1] must equal M_selected");
    for (int64_t g = 0; g < gs; ++g) {
        TORCH_CHECK(offsets_acc[g + 1] >= offsets_acc[g], "selected_offsets must be monotonic");
    }

    auto dx_acc = torch::zeros({m_selected, h}, options_fp32);
    auto grad_down_lora_A_acc = compute_down_lora_A
        ? torch::zeros({e, r_down, i}, options_fp32)
        : torch::Tensor();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CudaTimer total_timer(stream);
    float metadata_ms = 0.0f;
    float fill_ms = 0.0f;
    float recompute_ms = 0.0f;
    float dx_ms = 0.0f;
    int64_t w_cache_peak = 0;
    int64_t grad_pair_peak = 0;

    const int64_t num_windows = (i + p * q - 1) / (p * q);
    int64_t expected_cpu_weight_bytes_min = 0;
    int64_t cpu_weight_bytes_staged = 0;
    int64_t hbm_w_cache_valid_write_bytes = 0;
    int64_t hbm_w_cache_padding_zero_write_bytes = 0;
    int64_t recompute_reads_w_cache_bytes = 0;
    int64_t dx_reads_w_cache_bytes = 0;
    int64_t dS_down_reads_bytes = 0;
    int64_t down_mask_reads_bytes = 0;
    int64_t down_lora_A_atomic_adds = 0;

    for (int64_t iwin0 = 0; iwin0 < i; iwin0 += p * q) {
        const int64_t valid_w = std::min<int64_t>(p * q, i - iwin0);
        for (int64_t g_start = 0; g_start < gs; g_start += g_work) {
            const int64_t g_chunk = std::min<int64_t>(g_work, gs - g_start);
            const int64_t row_start = offsets_acc[g_start];
            const int64_t row_end = offsets_acc[g_start + g_chunk];
            const int64_t m_work = row_end - row_start;
            if (g_chunk <= 0 || m_work <= 0) continue;

            CudaTimer metadata_timer(stream);
            auto row_to_e_rel = torch::empty({m_work}, offsets_i32.options());
            auto row_to_expert = torch::empty({m_work}, offsets_i32.options());
            fill_row_metadata_kernel<<<static_cast<int>(g_chunk), kThreads, 0, stream>>>(
                offsets_i32.data_ptr<int32_t>(),
                experts_i32.data_ptr<int32_t>(),
                row_to_e_rel.data_ptr<int32_t>(),
                row_to_expert.data_ptr<int32_t>(),
                static_cast<int32_t>(g_start),
                static_cast<int32_t>(g_chunk));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            metadata_ms += metadata_timer.elapsed_ms();

            auto w_cache = torch::empty({g_chunk, kdx, h}, options_bf16);
            auto grad_pair_window = torch::empty({m_work, kdx}, options_bf16);
            w_cache_peak = std::max<int64_t>(w_cache_peak, w_cache.numel() * w_cache.element_size());
            grad_pair_peak = std::max<int64_t>(grad_pair_peak, grad_pair_window.numel() * grad_pair_window.element_size());

            CudaTimer fill_timer(stream);
            const int64_t fill_total = g_chunk * kdx * h;
            fill_w_cache_kernel<<<blocks_for(fill_total), kThreads, 0, stream>>>(
                gate_up_weight_cpu.data_ptr<at::BFloat16>(),
                experts_i32.data_ptr<int32_t>(),
                w_cache.data_ptr<at::BFloat16>(),
                static_cast<int32_t>(g_start),
                static_cast<int32_t>(g_chunk),
                static_cast<int32_t>(e),
                static_cast<int32_t>(i),
                static_cast<int32_t>(h),
                static_cast<int32_t>(iwin0),
                static_cast<int32_t>(p),
                static_cast<int32_t>(kdx));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            fill_ms += fill_timer.elapsed_ms();

            CudaTimer recompute_timer(stream);
            const int64_t recompute_total = m_work * kdx;
            recompute_activation_kernel<<<blocks_for(recompute_total), kThreads, 0, stream>>>(
                x_sel.data_ptr<at::BFloat16>(),
                dact_sel.data_ptr<at::BFloat16>(),
                compute_down_lora_A ? dS_down_sel.data_ptr<at::BFloat16>() : nullptr,
                gate_low_rank_sel.data_ptr<at::BFloat16>(),
                up_low_rank_sel.data_ptr<at::BFloat16>(),
                gate_lora_B.data_ptr<at::BFloat16>(),
                up_lora_B.data_ptr<at::BFloat16>(),
                w_cache.data_ptr<at::BFloat16>(),
                (compute_down_lora_A && down_dropout_p > 0.0) ? down_mask_packed.data_ptr<uint8_t>() : nullptr,
                row_to_e_rel.data_ptr<int32_t>(),
                row_to_expert.data_ptr<int32_t>(),
                grad_pair_window.data_ptr<at::BFloat16>(),
                grad_gate_sel.data_ptr<at::BFloat16>(),
                grad_up_sel.data_ptr<at::BFloat16>(),
                compute_down_lora_A ? grad_down_lora_A_acc.data_ptr<float>() : nullptr,
                static_cast<int32_t>(row_start),
                static_cast<int32_t>(m_work),
                static_cast<int32_t>(e),
                static_cast<int32_t>(h),
                static_cast<int32_t>(i),
                static_cast<int32_t>(r),
                static_cast<int32_t>(r_down),
                static_cast<int32_t>(down_mask_packed_width),
                static_cast<int32_t>(iwin0),
                static_cast<int32_t>(p),
                static_cast<int32_t>(kdx),
                static_cast<float>(lora_scale),
                down_dropout_p > 0.0 ? static_cast<float>(1.0 / (1.0 - down_dropout_p)) : 1.0f,
                compute_down_lora_A);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            recompute_ms += recompute_timer.elapsed_ms();

            CudaTimer dx_timer(stream);
            const int64_t dx_total = m_work * h;
            dx_window_kernel<<<blocks_for(dx_total), kThreads, 0, stream>>>(
                grad_pair_window.data_ptr<at::BFloat16>(),
                w_cache.data_ptr<at::BFloat16>(),
                row_to_e_rel.data_ptr<int32_t>(),
                dx_acc.data_ptr<float>(),
                static_cast<int32_t>(row_start),
                static_cast<int32_t>(m_work),
                static_cast<int32_t>(h),
                static_cast<int32_t>(kdx));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            dx_ms += dx_timer.elapsed_ms();

            const int64_t valid_bytes = g_chunk * 2 * valid_w * h * 2;
            const int64_t cache_bytes = w_cache.numel() * w_cache.element_size();
            expected_cpu_weight_bytes_min += valid_bytes;
            cpu_weight_bytes_staged += valid_bytes;
            hbm_w_cache_valid_write_bytes += valid_bytes;
            hbm_w_cache_padding_zero_write_bytes += std::max<int64_t>(0, cache_bytes - valid_bytes);
            recompute_reads_w_cache_bytes += m_work * kdx * h * 2;
            dx_reads_w_cache_bytes += m_work * kdx * h * 2;
            if (compute_down_lora_A) {
                dS_down_reads_bytes += m_work * valid_w * r_down * 2;
                down_lora_A_atomic_adds += m_work * valid_w * r_down;
                if (down_dropout_p > 0.0) {
                    down_mask_reads_bytes += m_work * valid_w;
                }
            }
        }
    }

    const int64_t cast_total = m_selected * h;
    cast_fp32_to_bf16_kernel<<<blocks_for(cast_total), kThreads, 0, stream>>>(
        dx_acc.data_ptr<float>(),
        grad_x_base_sel.data_ptr<at::BFloat16>(),
        cast_total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (compute_down_lora_A) {
        const int64_t cast_da_total = e * r_down * i;
        cast_fp32_to_bf16_kernel<<<blocks_for(cast_da_total), kThreads, 0, stream>>>(
            grad_down_lora_A_acc.data_ptr<float>(),
            grad_down_lora_A_sel.data_ptr<at::BFloat16>(),
            cast_da_total);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const float total_ms = total_timer.elapsed_ms();

    stats["num_windows"] = num_windows;
    stats["num_dx_windows"] = num_windows;
    stats["num_phase_launches"] = num_windows * ((gs + g_work - 1) / g_work) * 4 + 1;
    stats["expected_cpu_weight_bytes_min"] = expected_cpu_weight_bytes_min;
    stats["cpu_weight_bytes_staged"] = cpu_weight_bytes_staged;
    stats["cpu_weight_stream_multiplier"] = expected_cpu_weight_bytes_min == 0
        ? 0.0
        : static_cast<double>(cpu_weight_bytes_staged) / static_cast<double>(expected_cpu_weight_bytes_min);
    stats["hbm_w_cache_write_bytes"] = hbm_w_cache_valid_write_bytes + hbm_w_cache_padding_zero_write_bytes;
    stats["hbm_w_cache_valid_write_bytes"] = hbm_w_cache_valid_write_bytes;
    stats["hbm_w_cache_padding_zero_write_bytes"] = hbm_w_cache_padding_zero_write_bytes;
    stats["recompute_reads_w_cache_bytes"] = recompute_reads_w_cache_bytes;
    stats["dx_reads_w_cache_bytes"] = dx_reads_w_cache_bytes;
    stats["w_cache_bytes_allocated_peak"] = w_cache_peak;
    stats["grad_pair_window_bytes_allocated_peak"] = grad_pair_peak;
    stats["dx_acc_bytes_allocated_peak"] = dx_acc.numel() * dx_acc.element_size();
    stats["grad_down_lora_A_bytes_allocated_peak"] = compute_down_lora_A
        ? grad_down_lora_A_acc.numel() * grad_down_lora_A_acc.element_size()
        : 0;
    stats["dS_down_reads_bytes"] = dS_down_reads_bytes;
    stats["down_mask_reads_bytes"] = down_mask_reads_bytes;
    stats["down_lora_A_atomic_adds"] = down_lora_A_atomic_adds;
    stats["pair_acc_global_bytes"] = 0;
    stats["local_memory_load_bytes"] = 0;
    stats["local_memory_store_bytes"] = 0;
    stats["metadata_ms"] = metadata_ms;
    stats["fill_w_cache_ms"] = fill_ms;
    stats["recompute_grad_window_ms"] = recompute_ms;
    stats["recompute_act_ms"] = recompute_ms;
    stats["down_lora_activation_ms"] = compute_down_lora_A ? recompute_ms : 0.0f;
    stats["recompute_down_lora_activation_ms"] = compute_down_lora_A ? recompute_ms : 0.0f;
    stats["dx_window_ms"] = dx_ms;
    stats["native_total_ms"] = total_ms;
    stats["old_selected_base_dx_rows"] = 0;
    stats["new_selected_base_dx_rows"] = m_selected;
    stats["selected_recompute_rows"] = m_selected;
    stats["workspace_reused"] = true;

    if (compute_down_lora_A) {
        return py::make_tuple(grad_x_base_sel, grad_gate_sel, grad_up_sel, grad_down_lora_A_sel, return_stats ? stats : py::dict());
    }
    return py::make_tuple(grad_x_base_sel, grad_gate_sel, grad_up_sel, return_stats ? stats : py::dict());
}

}  // namespace asym_gemm::qwen3_moe
