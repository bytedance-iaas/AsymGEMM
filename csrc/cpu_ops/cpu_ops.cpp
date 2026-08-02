// Fused CPU SwiGLU kernels for Grace (aarch64 SVE-BF16), Stage 1 of
// agent/impls/cpu_compute.md. Single-pass, no temporaries: the ATen sequence they
// replace makes >=6 sweeps with intermediate allocations over [M, I] bf16 tensors.
//
//   cpu_fused_silu_mul_bf16:      out   = silu(gate) * up
//   cpu_fused_silu_backward_bf16: dgate = ga * up * s * (1 + g - g*s)   [s = sigmoid(g)]
//                                 dup   = ga * g * s
//
// All math in fp32; bf16 storage with round-to-nearest-even (svcvt/svcvtnt).
// Thread count is an explicit argument (own OMP team; never inherits the global
// torch/OMP setting — see cpu_compute.md caveat 0.4-5).

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <memory>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_thread_num() { return 0; }
static inline int omp_get_num_threads() { return 1; }
#endif

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_SVE_BF16)
#define ASYM_CPU_OPS_SVE 1
#include <arm_sve.h>
#else
#define ASYM_CPU_OPS_SVE 0
#endif

namespace asym_gemm::cpu_ops {

namespace {

inline float silu_ref(float x) {
    const float s = 1.0f / (1.0f + std::exp(-x));
    return x * s;
}

#if ASYM_CPU_OPS_SVE

// Vector exp on fp32 lanes: Cephes-style range reduction + degree-6 Taylor.
// Inputs clamped to [-87, 88] so the 2^n exponent scale stays in range.
// Max relative error ~2e-7 on the clamped domain — far below bf16 resolution.
inline svfloat32_t sve_expf(svbool_t pg, svfloat32_t x) {
    const svfloat32_t log2e = svdup_f32(1.442695041f);
    const svfloat32_t ln2_hi = svdup_f32(0.693145752f);   // 0x1.62e400p-1
    const svfloat32_t ln2_lo = svdup_f32(1.42860677e-6f); // 0x1.7f7d1cp-20

    x = svmax_f32_x(pg, svdup_f32(-87.0f), svmin_f32_x(pg, svdup_f32(88.0f), x));
    svfloat32_t n = svrinta_f32_x(pg, svmul_f32_x(pg, x, log2e));
    svfloat32_t r = svmls_f32_x(pg, x, n, ln2_hi);
    r = svmls_f32_x(pg, r, n, ln2_lo);

    // exp(r) = 1 + r + r^2/2! + ... + r^6/6!   (|r| <= 0.3466)
    svfloat32_t p = svdup_f32(1.38888889e-3f);            // 1/720
    p = svmad_f32_x(pg, p, r, svdup_f32(8.33333333e-3f));     // 1/120
    p = svmad_f32_x(pg, p, r, svdup_f32(4.16666667e-2f));     // 1/24
    p = svmad_f32_x(pg, p, r, svdup_f32(1.66666667e-1f));     // 1/6
    p = svmad_f32_x(pg, p, r, svdup_f32(0.5f));
    p = svmad_f32_x(pg, p, r, svdup_f32(1.0f));
    p = svmad_f32_x(pg, p, r, svdup_f32(1.0f));

    // scale by 2^n through the exponent field
    svint32_t ni = svcvt_s32_f32_x(pg, n);
    svfloat32_t scale =
        svreinterpret_f32_s32(svlsl_n_s32_x(pg, svadd_n_s32_x(pg, ni, 127), 23));
    return svmul_f32_x(pg, p, scale);
}

// Kernel-campaign 2026-07-21 NEGATIVE RESULT (variant tried + removed after its
// gate): FRECPE + 2-Newton reciprocal in place of the FDIV measured +19-20% SLOWER
// at both production shapes (the dependent 5-op estimate chain costs more than the
// well-hidden FDIV latency in this bandwidth-bound kernel) AND broke parity (max
// 917 bf16 ulp on the saturated-negative tail). The FDIV stays; see the
// fix_cpu_compute.md kernel-campaign log.
inline svfloat32_t sve_sigmoid(svbool_t pg, svfloat32_t x) {
    const svfloat32_t e = sve_expf(pg, svneg_f32_x(pg, x));
    svfloat32_t s = svdiv_f32_x(pg, svdup_f32(1.0f), svadd_n_f32_x(pg, e, 1.0f));
    // saturate outside the exp clamp range to match fp32 semantics
    // (fp32 sigmoid(-100) == 0 exactly via exp overflow to inf)
    s = svsel_f32(svcmplt_n_f32(pg, x, -87.0f), svdup_f32(0.0f), s);
    s = svsel_f32(svcmpgt_n_f32(pg, x, 87.0f), svdup_f32(1.0f), s);
    return s;
}

// Even/odd bf16<->fp32 lane pattern: u32 lane i of a loaded bf16 vector holds
// element 2i (low u16) and 2i+1 (high u16). "even" fp32 = bits<<16; "odd" fp32 =
// bits & 0xFFFF0000. svcvt_bf16 writes even bf16 lanes, svcvtnt writes odd lanes,
// so ordering round-trips exactly.
inline void widen(svbool_t pg32, svbfloat16_t v, svfloat32_t& even, svfloat32_t& odd) {
    const svuint32_t u = svreinterpret_u32_bf16(v);
    even = svreinterpret_f32_u32(svlsl_n_u32_x(pg32, u, 16));
    odd = svreinterpret_f32_u32(svand_n_u32_x(pg32, u, 0xFFFF0000u));
}

inline svbfloat16_t narrow(svbool_t pg32, svfloat32_t even, svfloat32_t odd) {
    svbfloat16_t r = svcvt_bf16_f32_x(pg32, even);
    return svcvtnt_bf16_f32_m(r, pg32, odd);
}

#endif // ASYM_CPU_OPS_SVE

using bf16_t = uint16_t;

inline float bf16_to_f32(bf16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

inline bf16_t f32_to_bf16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    // round-to-nearest-even (matches ATen / SVE svcvt)
    const uint32_t rounding = 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<bf16_t>((bits + rounding) >> 16);
}

void fused_silu_mul_range(const bf16_t* gate, const bf16_t* up, bf16_t* out,
                          int64_t lo, int64_t hi) {
#if ASYM_CPU_OPS_SVE
    const int64_t step = svcnth();
    int64_t i = lo;
    const svbool_t pg32 = svptrue_b32();
    static const bool use_nt = []() {
        const char* v = std::getenv("ASYM_CPU_SILU_NT");
        return v && *v && *v != '0';
    }();
    for (; i + step <= hi; i += step) {
        const svbool_t pg16 = svptrue_b16();
        // K-10: prefetch 4 iterations ahead (L1) — bf16 streams
        svprfh(pg16, reinterpret_cast<const bfloat16_t*>(gate + i + 4 * step), SV_PLDL1STRM);
        svprfh(pg16, reinterpret_cast<const bfloat16_t*>(up + i + 4 * step), SV_PLDL1STRM);
        svfloat32_t ge, go, ue, uo;
        widen(pg32, svld1_bf16(pg16, reinterpret_cast<const bfloat16_t*>(gate + i)), ge, go);
        widen(pg32, svld1_bf16(pg16, reinterpret_cast<const bfloat16_t*>(up + i)), ue, uo);
        const svfloat32_t re = svmul_f32_x(pg32, svmul_f32_x(pg32, ge, sve_sigmoid(pg32, ge)), ue);
        const svfloat32_t ro = svmul_f32_x(pg32, svmul_f32_x(pg32, go, sve_sigmoid(pg32, go)), uo);
        const svbfloat16_t rv = narrow(pg32, re, ro);
        if (use_nt) {
            svstnt1_bf16(pg16, reinterpret_cast<bfloat16_t*>(out + i), rv);
        } else {
            svst1_bf16(pg16, reinterpret_cast<bfloat16_t*>(out + i), rv);
        }
    }
    for (; i < hi; ++i)
        out[i] = f32_to_bf16(silu_ref(bf16_to_f32(gate[i])) * bf16_to_f32(up[i]));
#else
    for (int64_t i = lo; i < hi; ++i)
        out[i] = f32_to_bf16(silu_ref(bf16_to_f32(gate[i])) * bf16_to_f32(up[i]));
#endif
}

void fused_silu_backward_range(const bf16_t* gate, const bf16_t* up, const bf16_t* ga,
                               bf16_t* dgate, bf16_t* dup, int64_t lo, int64_t hi) {
#if ASYM_CPU_OPS_SVE
    const int64_t step = svcnth();
    int64_t i = lo;
    const svbool_t pg32 = svptrue_b32();
    const svfloat32_t one = svdup_f32(1.0f);
    for (; i + step <= hi; i += step) {
        const svbool_t pg16 = svptrue_b16();
        svfloat32_t ge, go, ue, uo, ae, ao;
        widen(pg32, svld1_bf16(pg16, reinterpret_cast<const bfloat16_t*>(gate + i)), ge, go);
        widen(pg32, svld1_bf16(pg16, reinterpret_cast<const bfloat16_t*>(up + i)), ue, uo);
        widen(pg32, svld1_bf16(pg16, reinterpret_cast<const bfloat16_t*>(ga + i)), ae, ao);

        const svfloat32_t se = sve_sigmoid(pg32, ge);
        const svfloat32_t so = sve_sigmoid(pg32, go);
        const svfloat32_t te = svmul_f32_x(pg32, ge, se);  // silu(g)
        const svfloat32_t to = svmul_f32_x(pg32, go, so);

        // dup = ga * silu(g)
        const svfloat32_t due = svmul_f32_x(pg32, ae, te);
        const svfloat32_t duo = svmul_f32_x(pg32, ao, to);
        // dgate = ga * up * s * (1 + g - g*s)
        const svfloat32_t we = svsub_f32_x(pg32, svadd_f32_x(pg32, one, ge), te);
        const svfloat32_t wo = svsub_f32_x(pg32, svadd_f32_x(pg32, one, go), to);
        const svfloat32_t dge =
            svmul_f32_x(pg32, svmul_f32_x(pg32, svmul_f32_x(pg32, ae, ue), se), we);
        const svfloat32_t dgo =
            svmul_f32_x(pg32, svmul_f32_x(pg32, svmul_f32_x(pg32, ao, uo), so), wo);

        svst1_bf16(svptrue_b16(), reinterpret_cast<bfloat16_t*>(dup + i), narrow(pg32, due, duo));
        svst1_bf16(svptrue_b16(), reinterpret_cast<bfloat16_t*>(dgate + i), narrow(pg32, dge, dgo));
    }
    for (; i < hi; ++i) {
        const float g = bf16_to_f32(gate[i]), u = bf16_to_f32(up[i]), a = bf16_to_f32(ga[i]);
        const float s = 1.0f / (1.0f + std::exp(-g));
        dup[i] = f32_to_bf16(a * g * s);
        dgate[i] = f32_to_bf16(a * u * s * (1.0f + g - g * s));
    }
#else
    for (int64_t i = lo; i < hi; ++i) {
        const float g = bf16_to_f32(gate[i]), u = bf16_to_f32(up[i]), a = bf16_to_f32(ga[i]);
        const float s = 1.0f / (1.0f + std::exp(-g));
        dup[i] = f32_to_bf16(a * g * s);
        dgate[i] = f32_to_bf16(a * u * s * (1.0f + g - g * s));
    }
#endif
}

constexpr int64_t kGranule = 1 << 16; // 64Ki elements per work unit

void check_bf16_same_shape(const at::Tensor& t, const at::Tensor& ref, const char* name) {
    TORCH_CHECK(t.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bf16");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.sizes() == ref.sizes(), name, " shape mismatch");
}

int resolve_threads(int64_t num_threads, int64_t n) {
    int nt = num_threads > 0 ? static_cast<int>(num_threads) : 32;
    const int64_t blocks = (n + kGranule - 1) / kGranule;
    return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, blocks)));
}

} // namespace

void cpu_fused_silu_mul_bf16(const at::Tensor& gate, const at::Tensor& up, at::Tensor& out,
                             int64_t num_threads) {
    check_bf16_same_shape(gate, gate, "gate");
    check_bf16_same_shape(up, gate, "up");
    check_bf16_same_shape(out, gate, "out");
    const int64_t n = gate.numel();
    if (n == 0) return;
    const auto* g = reinterpret_cast<const bf16_t*>(gate.data_ptr());
    const auto* u = reinterpret_cast<const bf16_t*>(up.data_ptr());
    auto* o = reinterpret_cast<bf16_t*>(out.data_ptr());
    const int nt = resolve_threads(num_threads, n);
#pragma omp parallel num_threads(nt)
    {
        const int tid = omp_get_thread_num(), tc = omp_get_num_threads();
        const int64_t chunk = (n + tc - 1) / tc;
        // 64-element alignment keeps vector bodies off shared cache lines
        const int64_t lo = std::min<int64_t>(n, ((tid * chunk + 63) / 64) * 64);
        const int64_t hi = std::min<int64_t>(n, (((tid + 1) * chunk + 63) / 64) * 64);
        if (lo < hi) fused_silu_mul_range(g, u, o, lo, hi);
    }
}

void cpu_fused_silu_backward_bf16(const at::Tensor& gate, const at::Tensor& up,
                                  const at::Tensor& grad_act, at::Tensor& dgate,
                                  at::Tensor& dup, int64_t num_threads) {
    check_bf16_same_shape(gate, gate, "gate");
    check_bf16_same_shape(up, gate, "up");
    check_bf16_same_shape(grad_act, gate, "grad_act");
    check_bf16_same_shape(dgate, gate, "dgate");
    check_bf16_same_shape(dup, gate, "dup");
    const int64_t n = gate.numel();
    if (n == 0) return;
    const auto* g = reinterpret_cast<const bf16_t*>(gate.data_ptr());
    const auto* u = reinterpret_cast<const bf16_t*>(up.data_ptr());
    const auto* a = reinterpret_cast<const bf16_t*>(grad_act.data_ptr());
    auto* dg = reinterpret_cast<bf16_t*>(dgate.data_ptr());
    auto* du = reinterpret_cast<bf16_t*>(dup.data_ptr());
    const int nt = resolve_threads(num_threads, n);
#pragma omp parallel num_threads(nt)
    {
        const int tid = omp_get_thread_num(), tc = omp_get_num_threads();
        const int64_t chunk = (n + tc - 1) / tc;
        const int64_t lo = std::min<int64_t>(n, ((tid * chunk + 63) / 64) * 64);
        const int64_t hi = std::min<int64_t>(n, (((tid + 1) * chunk + 63) / 64) * 64);
        if (lo < hi) fused_silu_backward_range(g, u, a, dg, du, lo, hi);
    }
}


void cpu_silu_bf16(const at::Tensor& gate, at::Tensor& out, int64_t num_threads) {
    check_bf16_same_shape(gate, gate, "gate");
    check_bf16_same_shape(out, gate, "out");
    const int64_t n = gate.numel();
    if (n == 0) return;
    const auto* g = reinterpret_cast<const bf16_t*>(gate.data_ptr());
    auto* o = reinterpret_cast<bf16_t*>(out.data_ptr());
    const int nt = resolve_threads(num_threads, n);
#pragma omp parallel num_threads(nt)
    {
        const int tid = omp_get_thread_num(), tc = omp_get_num_threads();
        const int64_t chunk = (n + tc - 1) / tc;
        const int64_t lo = std::min<int64_t>(n, ((tid * chunk + 63) / 64) * 64);
        const int64_t hi = std::min<int64_t>(n, (((tid + 1) * chunk + 63) / 64) * 64);
#if ASYM_CPU_OPS_SVE
        const int64_t step = svcnth();
        const svbool_t pg32 = svptrue_b32();
        int64_t i = lo;
        for (; i + step <= hi; i += step) {
            svfloat32_t ge, go;
            widen(pg32, svld1_bf16(svptrue_b16(), reinterpret_cast<const bfloat16_t*>(g + i)), ge, go);
            const svfloat32_t re = svmul_f32_x(pg32, ge, sve_sigmoid(pg32, ge));
            const svfloat32_t ro = svmul_f32_x(pg32, go, sve_sigmoid(pg32, go));
            svst1_bf16(svptrue_b16(), reinterpret_cast<bfloat16_t*>(o + i), narrow(pg32, re, ro));
        }
        for (; i < hi; ++i) o[i] = f32_to_bf16(silu_ref(bf16_to_f32(g[i])));
#else
        for (int64_t i = lo; i < hi; ++i) o[i] = f32_to_bf16(silu_ref(bf16_to_f32(g[i])));
#endif
    }
}

void cpu_mul_bf16_(at::Tensor& inout, const at::Tensor& other, int64_t num_threads) {
    check_bf16_same_shape(inout, inout, "inout");
    check_bf16_same_shape(other, inout, "other");
    const int64_t n = inout.numel();
    if (n == 0) return;
    auto* a = reinterpret_cast<bf16_t*>(inout.data_ptr());
    const auto* b = reinterpret_cast<const bf16_t*>(other.data_ptr());
    const int nt = resolve_threads(num_threads, n);
#pragma omp parallel num_threads(nt)
    {
        const int tid = omp_get_thread_num(), tc = omp_get_num_threads();
        const int64_t chunk = (n + tc - 1) / tc;
        const int64_t lo = std::min<int64_t>(n, ((tid * chunk + 63) / 64) * 64);
        const int64_t hi = std::min<int64_t>(n, (((tid + 1) * chunk + 63) / 64) * 64);
#if ASYM_CPU_OPS_SVE
        const int64_t step = svcnth();
        const svbool_t pg32 = svptrue_b32();
        int64_t i = lo;
        for (; i + step <= hi; i += step) {
            svfloat32_t ae, ao, be, bo;
            widen(pg32, svld1_bf16(svptrue_b16(), reinterpret_cast<const bfloat16_t*>(a + i)), ae, ao);
            widen(pg32, svld1_bf16(svptrue_b16(), reinterpret_cast<const bfloat16_t*>(b + i)), be, bo);
            svst1_bf16(svptrue_b16(), reinterpret_cast<bfloat16_t*>(a + i),
                       narrow(pg32, svmul_f32_x(pg32, ae, be), svmul_f32_x(pg32, ao, bo)));
        }
        for (; i < hi; ++i) a[i] = f32_to_bf16(bf16_to_f32(a[i]) * bf16_to_f32(b[i]));
#else
        for (int64_t i = lo; i < hi; ++i) a[i] = f32_to_bf16(bf16_to_f32(a[i]) * bf16_to_f32(b[i]));
#endif
    }
}


// ---- Stage 3: grouped LoRA-A wgrad on CPU (cpu_compute.md) ----------------
// Per expert group g (rows [pairs[2g], pairs[2g+1])):
//   grad_a[experts[g]] = dS_g^T @ X_g      (fp32 accumulate, bf16 store)
// Rank-1-update microkernel: r_block=4 x k_tile=16 fp32 accumulators in
// registers, m streamed; m chunked to 512 rows so the X sub-block stays in L2
// across the r/4 register passes (X read from DRAM once per (g, k_block)).
// Parallelism: OpenMP dynamic over (group, k_block=512) items — grouped/packed,
// no per-expert host loops. Technique adapted from kt-kernel's
// lora_bwd_grad_a_grouped_sve / arm_bf16_grad_matmul_reg (bf16_sft_moe.hpp).

#if ASYM_CPU_OPS_SVE
// widen one 8-lane bf16 vector into two SEQUENTIAL f32 vectors (lo = elems 0..3,
// hi = elems 4..7): zip with zeros puts each bf16 in the high half of a u32 lane.
static inline void widen_seq(svbfloat16_t v, svfloat32_t& lo, svfloat32_t& hi) {
    const svuint16_t z = svdup_u16(0);
    const svuint16_t u = svreinterpret_u16_bf16(v);
    lo = svreinterpret_f32_u16(svzip1_u16(z, u));
    hi = svreinterpret_f32_u16(svzip2_u16(z, u));
}
#endif

#if ASYM_CPU_OPS_SVE
// K-6: BFDOT m-pair microkernel — contracts TWO rows per instruction (2x flops/instr
// vs the widen+FMLA path). X rows are zipped into (m0,m1) pairs per k-lane; the dS
// pair for each r is broadcast as one u32.
static inline void wgrad_tile_bfdot(const bf16_t* dsp, const bf16_t* xp, float* scratch,
                                    int64_t m0, int64_t m1, int64_t R, int64_t K,
                                    int64_t k0, int64_t kb, int64_t r0) {
    // 4 r-rows x kb columns; scratch layout [R][kb]
    const svbool_t pg16 = svptrue_b16();
    const svbool_t pg32 = svptrue_b32();
    for (int64_t kt = 0; kt < kb; kt += 8) {           // 8 f32 lanes = 2 vectors
        float* c0 = scratch + (r0 + 0) * kb + kt;
        float* c1 = scratch + (r0 + 1) * kb + kt;
        float* c2 = scratch + (r0 + 2) * kb + kt;
        float* c3 = scratch + (r0 + 3) * kb + kt;
        svfloat32_t a00 = svld1_f32(pg32, c0), a01 = svld1_f32(pg32, c0 + 4);
        svfloat32_t a10 = svld1_f32(pg32, c1), a11 = svld1_f32(pg32, c1 + 4);
        svfloat32_t a20 = svld1_f32(pg32, c2), a21 = svld1_f32(pg32, c2 + 4);
        svfloat32_t a30 = svld1_f32(pg32, c3), a31 = svld1_f32(pg32, c3 + 4);
        int64_t m = m0;
        for (; m + 2 <= m1; m += 2) {
            const bf16_t* xr0 = xp + m * K + k0 + kt;
            const bf16_t* xr1 = xr0 + K;
            const svbfloat16_t v0 = svld1_bf16(pg16, reinterpret_cast<const bfloat16_t*>(xr0));
            const svbfloat16_t v1 = svld1_bf16(pg16, reinterpret_cast<const bfloat16_t*>(xr1));
            // pairs: lane j of zip1 = (x0[j], x1[j]) for j in 0..3; zip2 = j in 4..7
            const svbfloat16_t zlo = svzip1_bf16(v0, v1);
            const svbfloat16_t zhi = svzip2_bf16(v0, v1);
            const bf16_t* dr0 = dsp + m * R + r0;
            const bf16_t* dr1 = dr0 + R;
            uint32_t p0 = (uint32_t)dr0[0] | ((uint32_t)dr1[0] << 16);
            uint32_t p1 = (uint32_t)dr0[1] | ((uint32_t)dr1[1] << 16);
            uint32_t p2 = (uint32_t)dr0[2] | ((uint32_t)dr1[2] << 16);
            uint32_t p3 = (uint32_t)dr0[3] | ((uint32_t)dr1[3] << 16);
            const svbfloat16_t s0 = svreinterpret_bf16_u32(svdup_u32(p0));
            const svbfloat16_t s1 = svreinterpret_bf16_u32(svdup_u32(p1));
            const svbfloat16_t s2 = svreinterpret_bf16_u32(svdup_u32(p2));
            const svbfloat16_t s3 = svreinterpret_bf16_u32(svdup_u32(p3));
            a00 = svbfdot_f32(a00, zlo, s0); a01 = svbfdot_f32(a01, zhi, s0);
            a10 = svbfdot_f32(a10, zlo, s1); a11 = svbfdot_f32(a11, zhi, s1);
            a20 = svbfdot_f32(a20, zlo, s2); a21 = svbfdot_f32(a21, zhi, s2);
            a30 = svbfdot_f32(a30, zlo, s3); a31 = svbfdot_f32(a31, zhi, s3);
        }
        for (; m < m1; ++m) {  // odd tail row: scalar FMA into lanes via widen
            const bf16_t* xr = xp + m * K + k0 + kt;
            svfloat32_t x0, x1;
            widen_seq(svld1_bf16(pg16, reinterpret_cast<const bfloat16_t*>(xr)), x0, x1);
            const bf16_t* dr = dsp + m * R + r0;
            a00 = svmla_f32_x(pg32, a00, x0, svdup_f32(bf16_to_f32(dr[0])));
            a01 = svmla_f32_x(pg32, a01, x1, svdup_f32(bf16_to_f32(dr[0])));
            a10 = svmla_f32_x(pg32, a10, x0, svdup_f32(bf16_to_f32(dr[1])));
            a11 = svmla_f32_x(pg32, a11, x1, svdup_f32(bf16_to_f32(dr[1])));
            a20 = svmla_f32_x(pg32, a20, x0, svdup_f32(bf16_to_f32(dr[2])));
            a21 = svmla_f32_x(pg32, a21, x1, svdup_f32(bf16_to_f32(dr[2])));
            a30 = svmla_f32_x(pg32, a30, x0, svdup_f32(bf16_to_f32(dr[3])));
            a31 = svmla_f32_x(pg32, a31, x1, svdup_f32(bf16_to_f32(dr[3])));
        }
        svst1_f32(pg32, c0, a00); svst1_f32(pg32, c0 + 4, a01);
        svst1_f32(pg32, c1, a10); svst1_f32(pg32, c1 + 4, a11);
        svst1_f32(pg32, c2, a20); svst1_f32(pg32, c2 + 4, a21);
        svst1_f32(pg32, c3, a30); svst1_f32(pg32, c3 + 4, a31);
    }
}
#endif

void cpu_grouped_lora_a_grad_bf16(const at::Tensor& dS, const at::Tensor& x,
                                  at::Tensor& grad_a, const at::Tensor& pairs,
                                  const at::Tensor& group_experts, int64_t num_threads) {
    TORCH_CHECK(dS.device().is_cpu() && dS.dtype() == at::kBFloat16 && dS.is_contiguous() && dS.dim() == 2);
    TORCH_CHECK(x.device().is_cpu() && x.dtype() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2);
    const bool out_f32 = grad_a.dtype() == at::kFloat;
    TORCH_CHECK(grad_a.device().is_cpu() && (grad_a.dtype() == at::kBFloat16 || out_f32) && grad_a.is_contiguous() && grad_a.dim() == 3);
    TORCH_CHECK(pairs.device().is_cpu() && pairs.dtype() == at::kLong && pairs.is_contiguous());
    TORCH_CHECK(group_experts.device().is_cpu() && group_experts.dtype() == at::kLong && group_experts.is_contiguous());
    const int64_t M = dS.size(0), R = dS.size(1), K = x.size(1);
    TORCH_CHECK(x.size(0) == M && grad_a.size(1) == R && grad_a.size(2) == K);
    const int64_t G = group_experts.numel();
    TORCH_CHECK(pairs.numel() == 2 * G);
    const auto* dsp = reinterpret_cast<const bf16_t*>(dS.data_ptr());
    const auto* xp = reinterpret_cast<const bf16_t*>(x.data_ptr());
    auto* gap = reinterpret_cast<bf16_t*>(grad_a.data_ptr());
    float* gapf = out_f32 ? grad_a.data_ptr<float>() : nullptr;
    const int64_t* pr = pairs.data_ptr<int64_t>();
    const int64_t* ge = group_experts.data_ptr<int64_t>();
    const int64_t E = grad_a.size(0);

    std::memset(grad_a.data_ptr(), 0, grad_a.element_size() * grad_a.numel());

    constexpr int64_t KB = 512;   // k_block (fp32 scratch r*KB = 128 KB @ r=64)
    constexpr int64_t MC = 512;   // m chunk (X sub-block <= 512*512*2B = 512 KB, L2-resident)
    struct Item { int64_t g, k0, kb, im0, im1; };
    std::vector<Item> items;
    const int want = num_threads > 0 ? static_cast<int>(num_threads) : 32;
    // few-group shapes (attention E=1, dB K=64) starve on (g, k-block) items alone:
    // split m so items >= ~4x threads; partial tiles reduce under a per-tile lock.
    int64_t base_items = 0;
    for (int64_t g = 0; g < G; ++g) {
        const int64_t m0 = pr[2 * g], m1 = pr[2 * g + 1];
        if (m1 > m0) base_items += (K + KB - 1) / KB;
    }
    const bool m_split = out_f32 && base_items > 0 && base_items < 4 * want;
    const int64_t m_pieces = m_split ? std::max<int64_t>(1, (4 * want) / std::max<int64_t>(1, base_items)) : 1;
    for (int64_t g = 0; g < G; ++g) {
        const int64_t m0 = pr[2 * g], m1 = pr[2 * g + 1], e = ge[g];
        if (m1 <= m0 || e < 0 || e >= E) continue;
        TORCH_CHECK(m0 >= 0 && m1 <= M, "group rows out of range");
        const int64_t rows = m1 - m0;
        const int64_t mstep = std::max<int64_t>(MC, (rows + m_pieces - 1) / m_pieces);
        for (int64_t k0 = 0; k0 < K; k0 += KB)
            for (int64_t ms = m0; ms < m1; ms += mstep)
                items.push_back({g, k0, std::min(KB, K - k0), ms, std::min(m1, ms + mstep)});
    }
    std::vector<std::unique_ptr<std::mutex>> tile_locks;
    std::unordered_map<int64_t, int64_t> lock_of;  // (g * 1e9 + k0) -> lock idx
    if (m_split) {
        for (const auto& it : items) {
            const int64_t key = it.g * 1000000000LL + it.k0;
            if (!lock_of.count(key)) {
                lock_of[key] = static_cast<int64_t>(tile_locks.size());
                tile_locks.emplace_back(new std::mutex());
            }
        }
    }
    const int nt = num_threads > 0 ? static_cast<int>(std::min<int64_t>(num_threads, std::max<size_t>(1, items.size())))
                                   : 32;
    const bool vec_ok = (R % 4 == 0) && (K % 16 == 0);
#pragma omp parallel num_threads(nt)
    {
        std::vector<float> scratch(static_cast<size_t>(R) * KB);
#pragma omp for schedule(dynamic, 1)
        for (int64_t it = 0; it < static_cast<int64_t>(items.size()); ++it) {
            const int64_t g = items[it].g, k0 = items[it].k0, kb = items[it].kb;
            const int64_t m0 = items[it].im0, m1 = items[it].im1, e = ge[g];
            std::fill(scratch.begin(), scratch.begin() + R * kb, 0.0f);
#if ASYM_CPU_OPS_SVE
            static const bool use_bfdot = []() {
                const char* v = std::getenv("ASYM_CPU_WGRAD_BFDOT");
                return v && *v && *v != '0';
            }();
            if (vec_ok && svcnth() == 8 && use_bfdot) {  // K-6 BFDOT m-pair path
                for (int64_t ms = m0; ms < m1; ms += MC) {
                    const int64_t me = std::min(m1, ms + MC);
                    for (int64_t r0 = 0; r0 < R; r0 += 4)
                        wgrad_tile_bfdot(dsp, xp, scratch.data(), ms, me, R, K, k0, kb, r0);
                }
            } else if (vec_ok && svcnth() == 8) {   // 128-bit SVE path (Grace)
                for (int64_t ms = m0; ms < m1; ms += MC) {
                    const int64_t me = std::min(m1, ms + MC);
                    for (int64_t r0 = 0; r0 < R; r0 += 4) {
                        for (int64_t kt = 0; kt < kb; kt += 16) {
                            float* c0 = scratch.data() + (r0 + 0) * kb + kt;
                            float* c1 = scratch.data() + (r0 + 1) * kb + kt;
                            float* c2 = scratch.data() + (r0 + 2) * kb + kt;
                            float* c3 = scratch.data() + (r0 + 3) * kb + kt;
                            const svbool_t pg = svptrue_b32();
                            svfloat32_t a00 = svld1_f32(pg, c0), a01 = svld1_f32(pg, c0 + 4),
                                        a02 = svld1_f32(pg, c0 + 8), a03 = svld1_f32(pg, c0 + 12);
                            svfloat32_t a10 = svld1_f32(pg, c1), a11 = svld1_f32(pg, c1 + 4),
                                        a12 = svld1_f32(pg, c1 + 8), a13 = svld1_f32(pg, c1 + 12);
                            svfloat32_t a20 = svld1_f32(pg, c2), a21 = svld1_f32(pg, c2 + 4),
                                        a22 = svld1_f32(pg, c2 + 8), a23 = svld1_f32(pg, c2 + 12);
                            svfloat32_t a30 = svld1_f32(pg, c3), a31 = svld1_f32(pg, c3 + 4),
                                        a32 = svld1_f32(pg, c3 + 8), a33 = svld1_f32(pg, c3 + 12);
                            for (int64_t m = ms; m < me; ++m) {
                                const bf16_t* xr = xp + m * K + k0 + kt;
                                svfloat32_t x0, x1, x2, x3;
                                widen_seq(svld1_bf16(svptrue_b16(), reinterpret_cast<const bfloat16_t*>(xr)), x0, x1);
                                widen_seq(svld1_bf16(svptrue_b16(), reinterpret_cast<const bfloat16_t*>(xr + 8)), x2, x3);
                                const bf16_t* dr = dsp + m * R + r0;
                                const svfloat32_t s0 = svdup_f32(bf16_to_f32(dr[0]));
                                const svfloat32_t s1 = svdup_f32(bf16_to_f32(dr[1]));
                                const svfloat32_t s2 = svdup_f32(bf16_to_f32(dr[2]));
                                const svfloat32_t s3 = svdup_f32(bf16_to_f32(dr[3]));
                                a00 = svmla_f32_x(pg, a00, x0, s0); a01 = svmla_f32_x(pg, a01, x1, s0);
                                a02 = svmla_f32_x(pg, a02, x2, s0); a03 = svmla_f32_x(pg, a03, x3, s0);
                                a10 = svmla_f32_x(pg, a10, x0, s1); a11 = svmla_f32_x(pg, a11, x1, s1);
                                a12 = svmla_f32_x(pg, a12, x2, s1); a13 = svmla_f32_x(pg, a13, x3, s1);
                                a20 = svmla_f32_x(pg, a20, x0, s2); a21 = svmla_f32_x(pg, a21, x1, s2);
                                a22 = svmla_f32_x(pg, a22, x2, s2); a23 = svmla_f32_x(pg, a23, x3, s2);
                                a30 = svmla_f32_x(pg, a30, x0, s3); a31 = svmla_f32_x(pg, a31, x1, s3);
                                a32 = svmla_f32_x(pg, a32, x2, s3); a33 = svmla_f32_x(pg, a33, x3, s3);
                            }
                            svst1_f32(pg, c0, a00); svst1_f32(pg, c0 + 4, a01);
                            svst1_f32(pg, c0 + 8, a02); svst1_f32(pg, c0 + 12, a03);
                            svst1_f32(pg, c1, a10); svst1_f32(pg, c1 + 4, a11);
                            svst1_f32(pg, c1 + 8, a12); svst1_f32(pg, c1 + 12, a13);
                            svst1_f32(pg, c2, a20); svst1_f32(pg, c2 + 4, a21);
                            svst1_f32(pg, c2 + 8, a22); svst1_f32(pg, c2 + 12, a23);
                            svst1_f32(pg, c3, a30); svst1_f32(pg, c3 + 4, a31);
                            svst1_f32(pg, c3 + 8, a32); svst1_f32(pg, c3 + 12, a33);
                        }
                    }
                }
            } else
#endif
            {
                for (int64_t m = m0; m < m1; ++m) {
                    const bf16_t* xr = xp + m * K + k0;
                    const bf16_t* dr = dsp + m * R;
                    for (int64_t ri = 0; ri < R; ++ri) {
                        const float s = bf16_to_f32(dr[ri]);
                        float* c = scratch.data() + ri * kb;
                        for (int64_t kk = 0; kk < kb; ++kk) c[kk] += s * bf16_to_f32(xr[kk]);
                    }
                }
            }
            if (out_f32 && m_split) {
                float* out = gapf + (e * R) * K + k0;
                std::lock_guard<std::mutex> lk(*tile_locks[lock_of[g * 1000000000LL + k0]]);
                for (int64_t ri = 0; ri < R; ++ri) {
                    float* orow = out + ri * K;
                    const float* srow = scratch.data() + ri * kb;
                    for (int64_t kk = 0; kk < kb; ++kk) orow[kk] += srow[kk];
                }
            } else if (out_f32) {
                float* out = gapf + (e * R) * K + k0;
                for (int64_t ri = 0; ri < R; ++ri)
                    std::memcpy(out + ri * K, scratch.data() + ri * kb, sizeof(float) * kb);
            } else {
                bf16_t* out = gap + (e * R) * K + k0;
                for (int64_t ri = 0; ri < R; ++ri)
                    for (int64_t kk = 0; kk < kb; ++kk)
                        out[ri * K + kk] = f32_to_bf16(scratch[ri * kb + kk]);
            }
        }
    }
}


void cpu_rmsnorm_bf16(const at::Tensor& x, const at::Tensor& w, at::Tensor& out,
                      double eps, int64_t num_threads) {
    // K-7: HF Qwen3RMSNorm semantics — fp32 math, bf16 storage:
    // y = (w * x / sqrt(mean(x^2) + eps)) rounded to bf16
    TORCH_CHECK(x.device().is_cpu() && x.dtype() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2);
    TORCH_CHECK(out.device().is_cpu() && out.dtype() == at::kBFloat16 && out.is_contiguous());
    TORCH_CHECK(w.device().is_cpu() && w.is_contiguous() && w.numel() == x.size(1));
    TORCH_CHECK(out.sizes() == x.sizes());
    const int64_t M = x.size(0), H = x.size(1);
    if (M == 0) return;
    const auto* xp = reinterpret_cast<const bf16_t*>(x.data_ptr());
    auto* op = reinterpret_cast<bf16_t*>(out.data_ptr());
    const bool w_f32 = w.dtype() == at::kFloat;
    TORCH_CHECK(w_f32 || w.dtype() == at::kBFloat16);
    std::vector<float> wf(static_cast<size_t>(H));
    if (w_f32) {
        const float* wp = w.data_ptr<float>();
        std::copy(wp, wp + H, wf.begin());
    } else {
        const auto* wp = reinterpret_cast<const bf16_t*>(w.data_ptr());
        for (int64_t j = 0; j < H; ++j) wf[j] = bf16_to_f32(wp[j]);
    }
    // kernel-campaign 2026-07-21: hoist the even/odd weight deinterleave OUT of the
    // row loop (it was rebuilt scalar-wise per 8 lanes per row — H/8 times per row).
    // Same values in the same lane order => bit-identical output.
    std::vector<float> we_all(static_cast<size_t>(H / 2 + 8), 0.0f);
    std::vector<float> wo_all(static_cast<size_t>(H / 2 + 8), 0.0f);
    for (int64_t j = 0; j + 1 < H; j += 2) {
        we_all[static_cast<size_t>(j / 2)] = wf[static_cast<size_t>(j)];
        wo_all[static_cast<size_t>(j / 2)] = wf[static_cast<size_t>(j + 1)];
    }
    const float* we_base = we_all.data();
    const float* wo_base = wo_all.data();
    const int nt = resolve_threads(num_threads, M * H);
#pragma omp parallel for schedule(static) num_threads(nt)
    for (int64_t m = 0; m < M; ++m) {
        const bf16_t* xr = xp + m * H;
        bf16_t* orow = op + m * H;
        float ss = 0.0f;
        int64_t j = 0;
#if ASYM_CPU_OPS_SVE
        const svbool_t pg32 = svptrue_b32();
        const int64_t step = svcnth();
        svfloat32_t acc0 = svdup_f32(0.0f), acc1 = svdup_f32(0.0f);
        for (; j + step <= H; j += step) {
            svfloat32_t e, o;
            widen(pg32, svld1_bf16(svptrue_b16(), reinterpret_cast<const bfloat16_t*>(xr + j)), e, o);
            acc0 = svmla_f32_x(pg32, acc0, e, e);
            acc1 = svmla_f32_x(pg32, acc1, o, o);
        }
        ss = svaddv_f32(pg32, acc0) + svaddv_f32(pg32, acc1);
#endif
        for (; j < H; ++j) { const float v = bf16_to_f32(xr[j]); ss += v * v; }
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(H) + static_cast<float>(eps));
        j = 0;
#if ASYM_CPU_OPS_SVE
        const svfloat32_t vinv = svdup_f32(inv);
        for (; j + step <= H; j += step) {
            svfloat32_t e, o;
            widen(pg32, svld1_bf16(svptrue_b16(), reinterpret_cast<const bfloat16_t*>(xr + j)), e, o);
            // weights per even/odd lanes (hoisted deinterleave; identical lane values)
            const svfloat32_t we = svld1_f32(pg32, we_base + (j >> 1));
            const svfloat32_t wo = svld1_f32(pg32, wo_base + (j >> 1));
            const svfloat32_t re = svmul_f32_x(pg32, svmul_f32_x(pg32, e, vinv), we);
            const svfloat32_t ro = svmul_f32_x(pg32, svmul_f32_x(pg32, o, vinv), wo);
            svst1_bf16(svptrue_b16(), reinterpret_cast<bfloat16_t*>(orow + j), narrow(pg32, re, ro));
        }
#endif
        for (; j < H; ++j)
            orow[j] = f32_to_bf16(bf16_to_f32(xr[j]) * inv * wf[j]);
    }
}

double cpu_widen_bf16_sqsum(const at::Tensor& src, at::Tensor& dst, int64_t num_threads) {
    // K-8: dst_f32 = widen(src_bf16) AND return sum(dst^2) in one pass (drain + norm fused)
    TORCH_CHECK(src.device().is_cpu() && src.dtype() == at::kBFloat16 && src.is_contiguous());
    TORCH_CHECK(dst.device().is_cpu() && dst.dtype() == at::kFloat && dst.is_contiguous());
    TORCH_CHECK(src.numel() == dst.numel());
    const int64_t n = src.numel();
    if (n == 0) return 0.0;
    const auto* sp = reinterpret_cast<const bf16_t*>(src.data_ptr());
    float* dp = dst.data_ptr<float>();
    const int nt = resolve_threads(num_threads, n);
    double total = 0.0;
#pragma omp parallel num_threads(nt) reduction(+ : total)
    {
        const int tid = omp_get_thread_num(), tc = omp_get_num_threads();
        const int64_t chunk = (n + tc - 1) / tc;
        const int64_t lo = std::min<int64_t>(n, ((tid * chunk + 63) / 64) * 64);
        const int64_t hi = std::min<int64_t>(n, (((tid + 1) * chunk + 63) / 64) * 64);
        double acc = 0.0;
        for (int64_t i = lo; i < hi; ++i) {
            const float v = bf16_to_f32(sp[i]);
            dp[i] = v;
            acc += static_cast<double>(v) * static_cast<double>(v);
        }
        total += acc;
    }
    return total;
}

bool cpu_ops_sve_compiled() { return ASYM_CPU_OPS_SVE != 0; }

} // namespace asym_gemm::cpu_ops
