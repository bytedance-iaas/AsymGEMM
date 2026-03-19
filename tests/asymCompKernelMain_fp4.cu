#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <tuple>
#include <utility>
#include <string>
#include <random>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <cmath>

#include "../csrc/apis/gemm.hpp"
#include "../csrc/utils/layout.hpp"
#include "../csrc/utils/math.hpp"

// IMPORTANT: include the header that owns `prepare_init` / JIT globals
#include "../csrc/jit/compiler.hpp"

// ============================================================================
// NVFP4 (E2M1) format utilities
// ============================================================================
// NVFP4 uses the E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit
// Representable positive values: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
// Two E2M1 values are packed into one uint8 byte (low nibble + high nibble).
// One 8-byte unit = 16 FP4 elements.
// Scale factors use FP8 E4M3 format, one scale per group of elements.

// Encode a float to 4-bit E2M1 code (nearest rounding)
static inline uint8_t encode_e2m1(float x) {
    const bool sign = (x < 0.0f);
    float ax = std::fabs(x);
    if (ax > 6.0f) ax = 6.0f;

    // Midpoint boundaries for {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    static constexpr float bounds[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    uint8_t idx = 0;
    for (int i = 0; i < 7; ++i) {
        if (ax >= bounds[i]) idx = i + 1;
    }
    if (sign && idx != 0) idx |= 0x08;
    return idx;
}

// Decode FP8 E4M3 to float (matching torch.float8_e4m3fn behavior)
// Bit layout: [sign(1) | exp(4) | man(3)]
// bias = 7, special: no inf, NaN = 0x7F/0xFF
static inline float decode_fp8_e4m3(uint8_t bits) {
    const bool sign = (bits >> 7) & 1;
    const uint8_t exp = (bits >> 3) & 0x0F;
    const uint8_t man = bits & 0x07;

    float val;
    if (exp == 0x0F && man == 0x07) {
        return std::numeric_limits<float>::quiet_NaN();
    } else if (exp == 0) {
        // Subnormal: (-1)^s * 2^(1-bias) * (0.man)
        val = std::ldexp(static_cast<float>(man) / 8.0f, 1 - 7);
    } else {
        // Normal: (-1)^s * 2^(exp-bias) * (1 + man/8)
        val = std::ldexp(1.0f + static_cast<float>(man) / 8.0f, static_cast<int>(exp) - 7);
    }
    return sign ? -val : val;
}

// Encode float to FP8 E4M3 (simple nearest rounding)
static inline uint8_t encode_fp8_e4m3(float x) {
    if (std::isnan(x)) return 0x7F;
    const bool sign = (x < 0.0f);
    float ax = std::fabs(x);
    // E4M3 max = 448.0
    if (ax > 448.0f) ax = 448.0f;

    if (ax == 0.0f) {
        return sign ? 0x80 : 0x00;
    }

    // Try to find best exponent
    int exp_val = static_cast<int>(std::floor(std::log2(ax)));
    if (exp_val < -6) exp_val = -6;  // min subnormal exponent region
    if (exp_val > 8) exp_val = 8;

    uint8_t best_bits = 0;
    float best_err = std::numeric_limits<float>::max();

    for (int e = std::max(0, exp_val + 7 - 1); e <= std::min(14, exp_val + 7 + 1); ++e) {
        for (int m = 0; m < 8; ++m) {
            if (e == 0x0F && m == 0x07) continue;  // NaN
            float candidate;
            if (e == 0) {
                candidate = std::ldexp(static_cast<float>(m) / 8.0f, 1 - 7);
            } else {
                candidate = std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
            }
            float err = std::fabs(candidate - ax);
            if (err < best_err) {
                best_err = err;
                best_bits = (static_cast<uint8_t>(e) << 3) | static_cast<uint8_t>(m);
            }
        }
    }

    if (sign) best_bits |= 0x80;
    return best_bits;
}

// ============================================================================
// NVFP4 quantization: BF16 tensor -> packed uint8 + FP8 scale factors
// ============================================================================
// Quantizes per-token (per-row) with a given group size (gran_k).
// Each group of `gran_k` elements shares one FP8 E4M3 scale factor.
// Returns: {packed_fp4 [m, k/2] as uint8, scale_factors [m, ceil(k/gran_k)] as uint8 (FP8 E4M3)}
struct NvFP4QuantResult {
    std::vector<uint8_t> packed;     // [m * (k/2)]
    std::vector<uint8_t> scales;     // [m * num_groups_k]
    int64_t m, k, gran_k;
    int64_t num_groups_k;
};

static NvFP4QuantResult quantize_bf16_to_nvfp4(
    const float* data,  // [m, k] in float (converted from bf16)
    int64_t m, int64_t k, int64_t gran_k
) {
    const int64_t num_groups_k = (k + gran_k - 1) / gran_k;
    NvFP4QuantResult result;
    result.m = m;
    result.k = k;
    result.gran_k = gran_k;
    result.num_groups_k = num_groups_k;
    result.packed.resize(m * (k / 2));
    result.scales.resize(m * num_groups_k);

    for (int64_t row = 0; row < m; ++row) {
        for (int64_t g = 0; g < num_groups_k; ++g) {
            const int64_t col_start = g * gran_k;
            const int64_t col_end = std::min(col_start + gran_k, k);

            // Find amax in this group
            float amax = 1e-4f;
            for (int64_t c = col_start; c < col_end; ++c) {
                float av = std::fabs(data[row * k + c]);
                if (av > amax) amax = av;
            }

            // Scale factor: amax / 6.0, stored as FP8 E4M3
            float sf = amax / 6.0f;
            uint8_t sf_fp8 = encode_fp8_e4m3(sf);
            float sf_decoded = decode_fp8_e4m3(sf_fp8);
            if (sf_decoded == 0.0f || std::isnan(sf_decoded)) sf_decoded = 1.0f;

            result.scales[row * num_groups_k + g] = sf_fp8;

            // Quantize each element in the group
            for (int64_t c = col_start; c < col_end; ++c) {
                float scaled = data[row * k + c] / sf_decoded;
                uint8_t code = encode_e2m1(scaled);

                // Pack two codes per byte: even index -> low nibble, odd index -> high nibble
                int64_t pack_idx = row * (k / 2) + c / 2;
                if (c % 2 == 0) {
                    result.packed[pack_idx] = code & 0x0F;
                } else {
                    result.packed[pack_idx] |= (code & 0x0F) << 4;
                }
            }
        }
    }

    return result;
}

// Per-block quantization for B tensor [n, k] -> packed + 2D scale factors
struct NvFP4BlockQuantResult {
    std::vector<uint8_t> packed;     // [n * (k/2)]
    std::vector<uint8_t> scales;     // [ceil(n/gran_k) * ceil(k/gran_k)] as FP8 E4M3
    int64_t n, k, gran_k;
    int64_t num_groups_n, num_groups_k;
};

// Decode 4-bit E2M1 code to float.
// Code format: bit3=sign, bits[2:0]=magnitude bucket {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
static inline float decode_e2m1(uint8_t code) {
    static constexpr float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const bool sign = (code & 0x08) != 0;
    const float mag = lut[code & 0x07];
    return sign ? -mag : mag;
}

static NvFP4BlockQuantResult quantize_bf16_to_nvfp4_block(
    const float* data,  // [n, k] in float
    int64_t n, int64_t k, int64_t gran_k
) {
    const int64_t num_groups_n = (n + gran_k - 1) / gran_k;
    const int64_t num_groups_k = (k + gran_k - 1) / gran_k;

    NvFP4BlockQuantResult result;
    result.n = n;
    result.k = k;
    result.gran_k = gran_k;
    result.num_groups_n = num_groups_n;
    result.num_groups_k = num_groups_k;
    result.packed.resize(n * (k / 2), 0);
    result.scales.resize(num_groups_n * num_groups_k, 0);

    for (int64_t gn = 0; gn < num_groups_n; ++gn) {
        for (int64_t gk = 0; gk < num_groups_k; ++gk) {
            const int64_t row_start = gn * gran_k;
            const int64_t row_end = std::min(row_start + gran_k, n);
            const int64_t col_start = gk * gran_k;
            const int64_t col_end = std::min(col_start + gran_k, k);

            // Find amax in this block
            float amax = 1e-4f;
            for (int64_t r = row_start; r < row_end; ++r) {
                for (int64_t c = col_start; c < col_end; ++c) {
                    float av = std::fabs(data[r * k + c]);
                    if (av > amax) amax = av;
                }
            }

            float sf = amax / 6.0f;
            uint8_t sf_fp8 = encode_fp8_e4m3(sf);
            float sf_decoded = decode_fp8_e4m3(sf_fp8);
            if (sf_decoded == 0.0f || std::isnan(sf_decoded)) sf_decoded = 1.0f;

            result.scales[gn * num_groups_k + gk] = sf_fp8;

            // Quantize elements in this block
            for (int64_t r = row_start; r < row_end; ++r) {
                for (int64_t c = col_start; c < col_end; ++c) {
                    float scaled = data[r * k + c] / sf_decoded;
                    uint8_t code = encode_e2m1(scaled);

                    int64_t pack_idx = r * (k / 2) + c / 2;
                    if (c % 2 == 0) {
                        result.packed[pack_idx] = code & 0x0F;
                    } else {
                        result.packed[pack_idx] |= (code & 0x0F) << 4;
                    }
                }
            }
        }
    }

    return result;
}

// Build a manual reference by dequantizing NVFP4(E2M1)+E4M3 scales back to FP32,
// then running grouped GEMM on CPU.
static torch::Tensor compute_manual_nvfp4_e4m3_reference(
    const NvFP4QuantResult& a_quant,
    const std::vector<NvFP4BlockQuantResult>& b_quants,
    const torch::Tensor& m_indices_cpu,
    int64_t m, int64_t n, int64_t k, int64_t gran_k
) {
    auto opts_f32_cpu = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
    auto d_manual = torch::zeros({m, n}, opts_f32_cpu);
    float* d_ptr = d_manual.data_ptr<float>();

    const int64_t num_groups = static_cast<int64_t>(b_quants.size());
    const int64_t sf_k = (k + gran_k - 1) / gran_k;
    const int64_t sf_n = (n + gran_k - 1) / gran_k;
    const auto* mi = m_indices_cpu.data_ptr<int32_t>();

    // Decode A scales once.
    std::vector<float> a_scales(m * sf_k, 1.0f);
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t gk = 0; gk < sf_k; ++gk) {
            float sf = decode_fp8_e4m3(a_quant.scales[row * sf_k + gk]);
            if (sf == 0.0f || std::isnan(sf)) sf = 1.0f;
            a_scales[row * sf_k + gk] = sf;
        }
    }

    // Decode B scales once per group.
    std::vector<std::vector<float>> b_scales(
        num_groups, std::vector<float>(sf_n * sf_k, 1.0f));
    for (int64_t g = 0; g < num_groups; ++g) {
        for (int64_t gn = 0; gn < sf_n; ++gn) {
            for (int64_t gk = 0; gk < sf_k; ++gk) {
                float sf = decode_fp8_e4m3(b_quants[g].scales[gn * sf_k + gk]);
                if (sf == 0.0f || std::isnan(sf)) sf = 1.0f;
                b_scales[g][gn * sf_k + gk] = sf;
            }
        }
    }

    // Collect active rows by expert group.
    std::vector<std::vector<int64_t>> rows_by_group(num_groups);
    for (int64_t row = 0; row < m; ++row) {
        const int32_t gid = mi[row];
        if (gid >= 0 && gid < num_groups) rows_by_group[gid].push_back(row);
    }

    for (int64_t g = 0; g < num_groups; ++g) {
        const auto& rows = rows_by_group[g];
        if (rows.empty()) continue;

        auto a_deq = torch::empty({static_cast<int64_t>(rows.size()), k}, opts_f32_cpu);
        float* a_deq_ptr = a_deq.data_ptr<float>();
        for (int64_t ri = 0; ri < static_cast<int64_t>(rows.size()); ++ri) {
            const int64_t row = rows[ri];
            const uint8_t* a_row = a_quant.packed.data() + row * (k / 2);
            for (int64_t c = 0; c < k; ++c) {
                const uint8_t packed = a_row[c / 2];
                const uint8_t code = (c & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
                const float q = decode_e2m1(code);
                const float sf = a_scales[row * sf_k + (c / gran_k)];
                a_deq_ptr[ri * k + c] = q * sf;
            }
        }

        auto b_deq = torch::empty({n, k}, opts_f32_cpu);
        float* b_deq_ptr = b_deq.data_ptr<float>();
        const uint8_t* b_g_packed = b_quants[g].packed.data();
        for (int64_t r = 0; r < n; ++r) {
            const uint8_t* b_row = b_g_packed + r * (k / 2);
            for (int64_t c = 0; c < k; ++c) {
                const uint8_t packed = b_row[c / 2];
                const uint8_t code = (c & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
                const float q = decode_e2m1(code);
                const float sf = b_scales[g][(r / gran_k) * sf_k + (c / gran_k)];
                b_deq_ptr[r * k + c] = q * sf;
            }
        }

        // [rows, k] @ [k, n] -> [rows, n]
        auto c_g = torch::mm(a_deq, b_deq.t()).contiguous();
        const float* c_ptr = c_g.data_ptr<float>();
        for (int64_t ri = 0; ri < static_cast<int64_t>(rows.size()); ++ri)
            memcpy(d_ptr + rows[ri] * n, c_ptr + ri * n, n * sizeof(float));
    }

    return d_manual;
}

// ============================================================================
// CUDA error check helper
// ============================================================================
static bool cuda_check(const char* tag) {
    const auto launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::cerr << "[CUDA] " << tag << " launch error: " << cudaGetErrorString(launch_err) << "\n";
        return false;
    }
    const auto sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        std::cerr << "[CUDA] " << tag << " sync error: " << cudaGetErrorString(sync_err) << "\n";
        return false;
    }
    return true;
}

// ============================================================================
// Pretty-print comparison of two result tensors
// ============================================================================
static void print_two_way_compare(
    const char* label_a, const float* da,
    const char* label_b, const float* db,
    int64_t rows_total, int64_t cols_total
) {
    const int64_t rows = std::min<int64_t>(5, rows_total);
    const int64_t cols = std::min<int64_t>(5, cols_total);

    std::cout << std::fixed << std::setprecision(6);

    auto print_block = [&](const char* label, const float* d) {
        std::cout << "\n" << label << " (top-left " << rows << "x" << cols << "):\n";
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                std::cout << d[i * cols_total + j];
                std::cout << (j + 1 == cols ? '\n' : ' ');
            }
        }
    };

    print_block(label_a, da);
    print_block(label_b, db);
    // Pairwise diff statistics
    auto diff_stats = [&](const char* name, const float* x, const float* y) {
        double max_abs = 0.0, sum_abs = 0.0;
        int64_t count = rows_total * cols_total;
        for (int64_t i = 0; i < count; ++i) {
            double d = std::fabs(static_cast<double>(x[i]) - static_cast<double>(y[i]));
            if (d > max_abs) max_abs = d;
            sum_abs += d;
        }
        double mean_abs = count > 0 ? sum_abs / count : 0.0;
        // Relative diff: mean(|x-y|) / mean(|y|)
        double sum_ref = 0.0;
        for (int64_t i = 0; i < count; ++i)
            sum_ref += std::fabs(static_cast<double>(y[i]));
        double mean_ref = count > 0 ? sum_ref / count : 1.0;
        double rel_diff = mean_ref > 0 ? mean_abs / mean_ref : mean_abs;

        std::cout << "  " << name << ": max_abs=" << max_abs
                  << ", mean_abs=" << mean_abs
                  << ", rel_diff=" << rel_diff << "\n";
    };

    std::cout << "\nPairwise diff statistics (over " << rows_total << "x" << cols_total << " elements):\n";
    diff_stats("a_vs_b", da, db);
}

// ============================================================================
// Build m_indices for grouped GEMM (same logic as original)
// ============================================================================
static int fill_with_sentinel(
    int* m_indices, int M,
    int* offsets, int* experts, int capacity
) {
    if (!offsets || !experts || capacity <= 0) return 0;
    if (M <= 0 || !m_indices) return 0;

    int write = 0;
    auto maybe_emit = [&](int start_idx) {
        int e = m_indices[start_idx];
        if (e != -1) {
            if (write < capacity) {
                offsets[write] = start_idx;
                experts[write] = e;
            }
            ++write;
        }
    };

    maybe_emit(0);
    for (int i = 1; i < M; ++i) {
        if (m_indices[i] != m_indices[i - 1]) {
            maybe_emit(i);
        }
    }

    if (write < capacity) {
        offsets[write] = M;
        experts[write] = -1;
    }
    ++write;

    return std::min(write, capacity);
}

static torch::Tensor build_m_indices_like_generators(
    int64_t expected_m_per_group,
    int64_t num_groups,
    int64_t* out_m,
    int64_t* out_active_m
) {
    const int64_t alignment = asym_gemm::get_mk_alignment_for_contiguous_layout();
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(0.7f, 1.3f);

    std::vector<int64_t> actual_ms, aligned_ms;
    actual_ms.reserve(num_groups);
    aligned_ms.reserve(num_groups);

    int64_t total_m = 0, active_m = 0;
    for (int64_t i = 0; i < num_groups; ++i) {
        const int64_t actual_m = std::max<int64_t>(1, static_cast<int64_t>(expected_m_per_group * dist(rng)));
        const int64_t aligned_m = asym_gemm::align(actual_m, alignment);
        actual_ms.push_back(actual_m);
        aligned_ms.push_back(aligned_m);
        total_m += aligned_m;
        active_m += actual_m;
    }

    auto m_indices_cpu = torch::empty({total_m},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    auto* mi = m_indices_cpu.data_ptr<int32_t>();

    int64_t start = 0;
    for (int64_t i = 0; i < num_groups; ++i) {
        const int64_t actual_end = start + actual_ms[i];
        const int64_t aligned_end = start + aligned_ms[i];
        for (int64_t j = start; j < actual_end; ++j)
            mi[j] = static_cast<int32_t>(i);
        for (int64_t j = actual_end; j < aligned_end; ++j)
            mi[j] = -1;
        start = aligned_end;
    }

    *out_m = total_m;
    *out_active_m = active_m;
    return m_indices_cpu;
}

// ============================================================================
// Main: NVFP4 GEMM comparison
//   1. BF16 ground truth (bf16 @ bf16^T in float)
//   2. fp4_asym_gemm kernel
// ============================================================================
int main(int argc, char** argv) {
    // JIT setup
    setenv("DG_JIT_CACHE_DIR", "/tmp/deepgemm_jit", 1);
    setenv("DG_JIT_WITH_LINEINFO", "1", 1);
    setenv("DG_JIT_DEBUG", "1", 1);

    asym_gemm::Compiler::prepare_init(
        "/sgl-workspace/sglang/demo/AsymGEMM/asym_gemm",
        "/usr/local/cuda-12.9"
    );
    asym_gemm::KernelRuntime::prepare_init("/usr/local/cuda-12.9");

    torch::NoGradGuard ng;
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA not available.\n";
        return 1;
    }

    auto dev = torch::Device(torch::kCUDA, 0);
    c10::cuda::CUDAGuard device_guard(dev);

    // -----------------------------------------------------------------------
    // Problem dimensions
    // -----------------------------------------------------------------------
    // const int64_t n = 4096, k = 7168, num_groups = 4;
    // const int64_t expected_m_per_group = 2048;

    const int64_t n = 1024, k = 1024, num_groups = 2;
    const int64_t expected_m_per_group = 1024;
    int64_t m = 0, active_m = 0;

    // FP4 scale-factor granularity: 16 elements per group
    // (One 8-byte unit = 16 FP4 values = 2 nvfp4 packed units of 8 elements each)
    constexpr int64_t gran_k = 16;
    const int64_t sf_k = (k + gran_k - 1) / gran_k;
    const int64_t sf_n = (n + gran_k - 1) / gran_k;

    // For FP4 recipe (1,16,16), keep SF payload in E4M3 bytes and only do layout packing.
    // `disable_ue8m0_cast=true` avoids FP32->UE8M0 normalization in transform paths.
    const bool disable_ue8m0_cast = true;
    const std::string compiled_dims = "nk";
    const std::optional<std::tuple<int,int,int>> recipe = std::make_tuple(1, 16, 16);

    // -----------------------------------------------------------------------
    // Build m_indices (group assignment for each row of A)
    // -----------------------------------------------------------------------
    auto m_indices_cpu = build_m_indices_like_generators(expected_m_per_group, num_groups, &m, &active_m);
    auto m_indices = m_indices_cpu.to(dev);
    const int max_len = static_cast<int>(num_groups) + 1;
    std::vector<int> offsets_h(max_len), experts_h(max_len);
    const int list_size = fill_with_sentinel(
        m_indices_cpu.data_ptr<int>(), m_indices_cpu.numel(),
        offsets_h.data(), experts_h.data(), max_len);

    auto opts_i32_cuda = torch::TensorOptions().device(dev).dtype(torch::kInt32);
    auto offsets_t = torch::empty({max_len}, opts_i32_cuda);
    auto experts_t = torch::empty({max_len}, opts_i32_cuda);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    cudaMemcpyAsync(offsets_t.data_ptr<int>(), offsets_h.data(),
                    max_len * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(experts_t.data_ptr<int>(), experts_h.data(),
                    max_len * sizeof(int), cudaMemcpyHostToDevice, stream);

    // -----------------------------------------------------------------------
    // Generate random BF16 data (A and B)
    // -----------------------------------------------------------------------
    auto A_bf16 = torch::randn({m, k}, torch::TensorOptions().device(dev).dtype(torch::kBFloat16));
    auto B_bf16 = torch::randn({num_groups, n, k}, torch::TensorOptions().device(dev).dtype(torch::kBFloat16));

    std::cout << "=== NVFP4 GEMM Comparison Test ===\n";
    std::cout << "m=" << m << ", active_m=" << active_m << ", n=" << n
              << ", k=" << k << ", num_groups=" << num_groups << "\n";
    std::cout << "FP4 format: E2M1 (NVFP4), 2 values per byte, 8 bytes = 16 elements (2 nvfp4 units)\n";
    std::cout << "Scale factor: FP8 E4M3, granularity=" << gran_k << "\n\n";

    // -----------------------------------------------------------------------
    // Step 1: BF16 ground truth  (A_bf16 @ B_bf16[g]^T for each group)
    // -----------------------------------------------------------------------
    std::cout << "[1/2] Computing BF16 ground truth...\n";
    auto D_bf16_gt = torch::zeros({m, n}, torch::TensorOptions().device(dev).dtype(torch::kBFloat16));
    {
        auto mi_cpu = m_indices_cpu.data_ptr<int32_t>();
        for (int64_t g = 0; g < num_groups; ++g) {
            // Find rows belonging to group g
            std::vector<int64_t> rows;
            for (int64_t i = 0; i < m; ++i) {
                if (mi_cpu[i] == g) rows.push_back(i);
            }
            if (rows.empty()) continue;

            auto idx = torch::tensor(std::vector<int64_t>(rows.begin(), rows.end()),
                                     torch::TensorOptions().dtype(torch::kLong).device(dev));
            auto A_g = A_bf16.index_select(0, idx);   // [rows, k]
            auto B_g = B_bf16[g];                       // [n, k]
            // matmul in float for precision
            auto result = torch::mm(A_g.to(torch::kFloat32), B_g.to(torch::kFloat32).t()).to(torch::kBFloat16);
            D_bf16_gt.index_copy_(0, idx, result);
        }
    }
    std::cout << "  BF16 ground truth computed.\n";

    // -----------------------------------------------------------------------
    // Step 2: Quantize A and B to NVFP4 with FP8 E4M3 scale factors
    // -----------------------------------------------------------------------
    std::cout << "[2/2] Quantizing to NVFP4 (E2M1 + FP8 E4M3 scales)...\n";

    // Convert A and B to CPU float for quantization
    auto A_f32_cpu = A_bf16.to(torch::kCPU, torch::kFloat32).contiguous();
    auto B_f32_cpu = B_bf16.to(torch::kCPU, torch::kFloat32).contiguous();
    const float* a_data = A_f32_cpu.data_ptr<float>();
    const float* b_data = B_f32_cpu.data_ptr<float>();

    // Quantize A: per-token, [m, k] -> packed [m, k/2], scales [m, sf_k]
    auto a_quant = quantize_bf16_to_nvfp4(a_data, m, k, gran_k);

    // Quantize B: per-block, [num_groups, n, k] -> for each group: packed [n, k/2], scales [sf_n, sf_k]
    std::vector<NvFP4BlockQuantResult> b_quants(num_groups);
    for (int64_t g = 0; g < num_groups; ++g) {
        b_quants[g] = quantize_bf16_to_nvfp4_block(
            b_data + g * n * k, n, k, gran_k);
    }

    std::cout << "  A packed: [" << m << ", " << k/2 << "] uint8, scales: [" << m << ", " << sf_k << "] FP8\n";
    std::cout << "  B packed: [" << num_groups << ", " << n << ", " << k/2 << "] uint8, "
              << "scales: [" << num_groups << ", " << sf_n << ", " << sf_k << "] FP8\n";

    // Verify 8-byte unit structure
    std::cout << "  Verification: 8 bytes = " << 8*2 << " FP4 elements = 2 nvfp4 packed units\n";

    // -----------------------------------------------------------------------
    // Build torch tensors for the kernel (packed uint8 + scale factors).
    // We pass SF as E4M3 (`torch::kFloat8_e4m3fn`); the layout transform
    // keeps E4M3 payload bytes and packs them into SM100 TMA layout.
    // -----------------------------------------------------------------------

    // A tensor: packed FP4 [m, k/2] as uint8
    auto A_fp4_packed = torch::empty({m, k / 2}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8));
    memcpy(A_fp4_packed.data_ptr<uint8_t>(), a_quant.packed.data(), m * (k / 2));
    auto A_fp4_gpu = A_fp4_packed.to(dev);

    // A scale factors: [m, sf_k] as E4M3
    auto SFA_u8_cpu = torch::empty({m, sf_k}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8));
    memcpy(SFA_u8_cpu.data_ptr<uint8_t>(), a_quant.scales.data(), m * sf_k);
    auto SFA = SFA_u8_cpu.to(dev).view(torch::kFloat8_e4m3fn);

    // B tensor: packed FP4 [num_groups, n, k/2] as uint8 on CPU first.
    auto B_fp4_packed_cpu = torch::empty(
        {num_groups, n, k / 2},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8).pinned_memory(true));
    for (int64_t g = 0; g < num_groups; ++g) {
        memcpy(B_fp4_packed_cpu.data_ptr<uint8_t>() + g * n * (k / 2),
               b_quants[g].packed.data(), n * (k / 2));
    }

    // B scale factors: [num_groups, sf_n, sf_k] as E4M3
    auto SFB_u8_cpu = torch::empty(
        {num_groups, sf_n, sf_k},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8).pinned_memory(true));
    for (int64_t g = 0; g < num_groups; ++g) {
        memcpy(SFB_u8_cpu.data_ptr<uint8_t>() + g * sf_n * sf_k,
               b_quants[g].scales.data(), sf_n * sf_k);
    }
    auto SFB_cpu = SFB_u8_cpu.view(torch::kFloat8_e4m3fn);

    // SM100 FP4 TMA path is sensitive to CPU/pinned memory-domain handling.
    // Use CUDA tensors for B/SFB in this test to avoid host-memory TMA issues.
    auto B_fp4_gpu = B_fp4_packed_cpu.to(dev, /*non_blocking=*/true);
    auto SFB_gpu = SFB_cpu.to(dev, /*non_blocking=*/true);

    auto D_kernel = torch::empty({m, n}, torch::TensorOptions().device(dev).dtype(torch::kBFloat16));

    std::pair<torch::Tensor, torch::Tensor> a_pair{A_fp4_gpu, SFA};
    // std::pair<torch::Tensor, torch::Tensor> b_pair{B_fp4_gpu, SFB_gpu};
    std::pair<torch::Tensor, torch::Tensor> b_pair{B_fp4_packed_cpu, SFB_gpu};

    auto check_k_major = [](const torch::Tensor& t, const char* name) {
        const bool k_major = t.stride(-1) == 1;
        std::cout << "  " << name << " strides=" << t.strides()
                  << " (" << (k_major ? "K-major" : "MN-major") << ")\n";
        if (!k_major)
            std::cerr << "  ERROR: " << name << " must be K-major (stride(-1)==1).\n";
        return k_major;
    };

    std::cout << "\n  Calling fp4_asym_gemm kernel (m_grouped_fp4_asym_gemm_nt_contiguous)...\n";
    std::cout << "  A_fp4=" << A_fp4_gpu.sizes() << " " << A_fp4_gpu.scalar_type()
              << "  SFA=" << SFA.sizes() << " " << SFA.scalar_type() << "\n";
    std::cout << "  B_fp4=" << b_pair.first.sizes() << " " << b_pair.first.scalar_type()
              << "  SFB=" << b_pair.second.sizes() << " " << b_pair.second.scalar_type() << "\n";
    if (!check_k_major(A_fp4_gpu, "A_fp4") || !check_k_major(b_pair.first, "B_fp4"))
        return 3;

    asym_gemm::gemm::m_grouped_fp4_asym_gemm_nt_contiguous(
        a_pair, b_pair, D_kernel, offsets_t, experts_t, list_size,
        recipe, compiled_dims, disable_ue8m0_cast
    );
    if (!cuda_check("m_grouped_fp4_asym_gemm_nt_contiguous")) {
        std::cerr << "KERNEL FAILURE. Aborting.\n";
        return 2;
    }
    std::cout << "  fp4_asym_gemm kernel finished.\n";

    std::cout << "\n[Manual] Computing NVFP4+E4M3 dequantized reference GEMM...\n";
    auto D_manual_f32_cpu = compute_manual_nvfp4_e4m3_reference(
        a_quant, b_quants, m_indices_cpu, m, n, k, gran_k);
    std::cout << "  Manual reference computed.\n";

    // -----------------------------------------------------------------------
    // Compare kernel output vs BF16 ground truth
    // -----------------------------------------------------------------------
    std::cout << "\n=== Kernel vs Ground Truth ===\n";

    // Convert kernel output and bf16 ground truth to CPU float
    auto D_kernel_f32_cpu = D_kernel.to(torch::kCPU, torch::kFloat32).contiguous();
    auto D_gt_f32_cpu = D_bf16_gt.to(torch::kCPU, torch::kFloat32).contiguous();

    // Only compare active (non-padding) rows
    // Build a mask of active rows
    auto mi_ptr = m_indices_cpu.data_ptr<int32_t>();
    std::vector<int64_t> active_rows;
    for (int64_t i = 0; i < m; ++i) {
        if (mi_ptr[i] != -1) active_rows.push_back(i);
    }

    std::cout << "Comparing " << active_rows.size() << " active rows (excluding padding rows with m_index=-1)\n";

    // Gather active rows into contiguous buffers
    const int64_t am = static_cast<int64_t>(active_rows.size());
    std::vector<float> kernel_active(am * n), gt_active(am * n), manual_active(am * n);
    const float* kernel_ptr = D_kernel_f32_cpu.data_ptr<float>();
    const float* gt_ptr = D_gt_f32_cpu.data_ptr<float>();
    const float* manual_ptr = D_manual_f32_cpu.data_ptr<float>();

    for (int64_t ri = 0; ri < am; ++ri) {
        memcpy(kernel_active.data() + ri * n, kernel_ptr + active_rows[ri] * n, n * sizeof(float));
        memcpy(gt_active.data() + ri * n, gt_ptr + active_rows[ri] * n, n * sizeof(float));
        memcpy(manual_active.data() + ri * n, manual_ptr + active_rows[ri] * n, n * sizeof(float));
    }

    print_two_way_compare(
        "D_kernel (fp4_asym_gemm)", kernel_active.data(),
        "D_bf16_gt (BF16 ground truth)", gt_active.data(),
        am, n
    );

    std::cout << "\n=== Manual NVFP4+E4M3 vs Ground Truth ===\n";
    print_two_way_compare(
        "D_manual (dequantized NVFP4+E4M3)", manual_active.data(),
        "D_bf16_gt (BF16 ground truth)", gt_active.data(),
        am, n
    );

    std::cout << "\n=== Kernel vs Manual NVFP4+E4M3 ===\n";
    print_two_way_compare(
        "D_kernel (fp4_asym_gemm)", kernel_active.data(),
        "D_manual (dequantized NVFP4+E4M3)", manual_active.data(),
        am, n
    );

    // Summary statistics
    std::cout << "\n=== Summary ===\n";
    auto kernel_t = torch::from_blob(kernel_active.data(), {am, n});
    auto gt_t = torch::from_blob(gt_active.data(), {am, n});
    auto manual_t = torch::from_blob(manual_active.data(), {am, n});

    std::cout << "D_kernel  mean=" << kernel_t.mean().item<float>()
              << ", std=" << kernel_t.std().item<float>() << "\n";
    std::cout << "D_bf16_gt mean=" << gt_t.mean().item<float>()
              << ", std=" << gt_t.std().item<float>() << "\n";
    std::cout << "D_manual  mean=" << manual_t.mean().item<float>()
              << ", std=" << manual_t.std().item<float>() << "\n";

    std::cout << "\nDone.\n";
    return 0;
}
