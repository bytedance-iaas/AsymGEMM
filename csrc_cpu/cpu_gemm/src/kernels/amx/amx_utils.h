/*
 * Subset of helpers used by the AMX BF16 buffer + kernel paths.
 *
 * Ported from operators/amx/la/utils.hpp and operators/amx/la/amx_utils.hpp
 * with the debug printers, transpose_16x16 stride variant, and VBMI shims
 * dropped — none of them are needed for the BF16 vertical slice.
 */
#ifndef CPU_GEMM_KERNELS_AMX_AMX_UTILS_H
#define CPU_GEMM_KERNELS_AMX_AMX_UTILS_H

#if defined(CPU_GEMM_HAS_AMX)

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace cpu_gemm::kernels::amx {

/* Byte-offset pointer arithmetic that preserves the element type. */
template <typename T>
inline T* offset_pointer(T* ptr, std::size_t byte_offset) {
  return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr) + byte_offset);
}
template <typename T>
inline const T* offset_pointer(const T* ptr, std::size_t byte_offset) {
  return reinterpret_cast<const T*>(reinterpret_cast<const char*>(ptr) + byte_offset);
}

/* Bulk 32-element BF16 copy (one cache line). */
inline void avx512_copy_32xbf16(const __m512i* src, __m512i* dst) {
  _mm512_storeu_si512(dst, _mm512_loadu_si512(src));
}

/* FP32 → BF16 for 32 values via AVX-512 BF16 if available, manual RNE
 * fallback otherwise. */
inline void avx512_32xfp32_to_32xbf16(const __m512* src0, const __m512* src1, __m512i* dst) {
#if defined(__AVX512BF16__)
  _mm512_storeu_si512(dst, __m512i(_mm512_cvtne2ps_pbh(*src1, *src0)));
#else
  __m512i i0 = _mm512_castps_si512(*src0);
  __m512i i1 = _mm512_castps_si512(*src1);
  __m512i r0 = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF),
                                _mm512_and_epi32(_mm512_srli_epi32(i0, 16), _mm512_set1_epi32(1)));
  __m512i r1 = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF),
                                _mm512_and_epi32(_mm512_srli_epi32(i1, 16), _mm512_set1_epi32(1)));
  i0 = _mm512_srli_epi32(_mm512_add_epi32(i0, r0), 16);
  i1 = _mm512_srli_epi32(_mm512_add_epi32(i1, r1), 16);
  __m512i packed = _mm512_packus_epi32(i0, i1);
  packed = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), packed);
  _mm512_storeu_si512(dst, packed);
#endif
}

/* In-place 16x16 32-bit transpose (used to produce the AMX VNNI tile for
 * B). Bit-for-bit copy of amx_utils.hpp:transpose_16x16_32bit. */
inline void transpose_16x16_32bit(__m512i* v) {
  __m512i t[16];
  t[0]  = _mm512_unpacklo_epi32(v[0],  v[1]);
  t[1]  = _mm512_unpackhi_epi32(v[0],  v[1]);
  t[2]  = _mm512_unpacklo_epi32(v[2],  v[3]);
  t[3]  = _mm512_unpackhi_epi32(v[2],  v[3]);
  t[4]  = _mm512_unpacklo_epi32(v[4],  v[5]);
  t[5]  = _mm512_unpackhi_epi32(v[4],  v[5]);
  t[6]  = _mm512_unpacklo_epi32(v[6],  v[7]);
  t[7]  = _mm512_unpackhi_epi32(v[6],  v[7]);
  t[8]  = _mm512_unpacklo_epi32(v[8],  v[9]);
  t[9]  = _mm512_unpackhi_epi32(v[8],  v[9]);
  t[10] = _mm512_unpacklo_epi32(v[10], v[11]);
  t[11] = _mm512_unpackhi_epi32(v[10], v[11]);
  t[12] = _mm512_unpacklo_epi32(v[12], v[13]);
  t[13] = _mm512_unpackhi_epi32(v[12], v[13]);
  t[14] = _mm512_unpacklo_epi32(v[14], v[15]);
  t[15] = _mm512_unpackhi_epi32(v[14], v[15]);

  v[0]  = _mm512_unpacklo_epi64(t[0],  t[2]);
  v[1]  = _mm512_unpackhi_epi64(t[0],  t[2]);
  v[2]  = _mm512_unpacklo_epi64(t[1],  t[3]);
  v[3]  = _mm512_unpackhi_epi64(t[1],  t[3]);
  v[4]  = _mm512_unpacklo_epi64(t[4],  t[6]);
  v[5]  = _mm512_unpackhi_epi64(t[4],  t[6]);
  v[6]  = _mm512_unpacklo_epi64(t[5],  t[7]);
  v[7]  = _mm512_unpackhi_epi64(t[5],  t[7]);
  v[8]  = _mm512_unpacklo_epi64(t[8],  t[10]);
  v[9]  = _mm512_unpackhi_epi64(t[8],  t[10]);
  v[10] = _mm512_unpacklo_epi64(t[9],  t[11]);
  v[11] = _mm512_unpackhi_epi64(t[9],  t[11]);
  v[12] = _mm512_unpacklo_epi64(t[12], t[14]);
  v[13] = _mm512_unpackhi_epi64(t[12], t[14]);
  v[14] = _mm512_unpacklo_epi64(t[13], t[15]);
  v[15] = _mm512_unpackhi_epi64(t[13], t[15]);

  t[0]  = _mm512_shuffle_i32x4(v[0],  v[4],  0x88);
  t[1]  = _mm512_shuffle_i32x4(v[1],  v[5],  0x88);
  t[2]  = _mm512_shuffle_i32x4(v[2],  v[6],  0x88);
  t[3]  = _mm512_shuffle_i32x4(v[3],  v[7],  0x88);
  t[4]  = _mm512_shuffle_i32x4(v[0],  v[4],  0xdd);
  t[5]  = _mm512_shuffle_i32x4(v[1],  v[5],  0xdd);
  t[6]  = _mm512_shuffle_i32x4(v[2],  v[6],  0xdd);
  t[7]  = _mm512_shuffle_i32x4(v[3],  v[7],  0xdd);
  t[8]  = _mm512_shuffle_i32x4(v[8],  v[12], 0x88);
  t[9]  = _mm512_shuffle_i32x4(v[9],  v[13], 0x88);
  t[10] = _mm512_shuffle_i32x4(v[10], v[14], 0x88);
  t[11] = _mm512_shuffle_i32x4(v[11], v[15], 0x88);
  t[12] = _mm512_shuffle_i32x4(v[8],  v[12], 0xdd);
  t[13] = _mm512_shuffle_i32x4(v[9],  v[13], 0xdd);
  t[14] = _mm512_shuffle_i32x4(v[10], v[14], 0xdd);
  t[15] = _mm512_shuffle_i32x4(v[11], v[15], 0xdd);

  v[0]  = _mm512_shuffle_i32x4(t[0],  t[8],  0x88);
  v[1]  = _mm512_shuffle_i32x4(t[1],  t[9],  0x88);
  v[2]  = _mm512_shuffle_i32x4(t[2],  t[10], 0x88);
  v[3]  = _mm512_shuffle_i32x4(t[3],  t[11], 0x88);
  v[4]  = _mm512_shuffle_i32x4(t[4],  t[12], 0x88);
  v[5]  = _mm512_shuffle_i32x4(t[5],  t[13], 0x88);
  v[6]  = _mm512_shuffle_i32x4(t[6],  t[14], 0x88);
  v[7]  = _mm512_shuffle_i32x4(t[7],  t[15], 0x88);
  v[8]  = _mm512_shuffle_i32x4(t[0],  t[8],  0xdd);
  v[9]  = _mm512_shuffle_i32x4(t[1],  t[9],  0xdd);
  v[10] = _mm512_shuffle_i32x4(t[2],  t[10], 0xdd);
  v[11] = _mm512_shuffle_i32x4(t[3],  t[11], 0xdd);
  v[12] = _mm512_shuffle_i32x4(t[4],  t[12], 0xdd);
  v[13] = _mm512_shuffle_i32x4(t[5],  t[13], 0xdd);
  v[14] = _mm512_shuffle_i32x4(t[6],  t[14], 0xdd);
  v[15] = _mm512_shuffle_i32x4(t[7],  t[15], 0xdd);
}

}  // namespace cpu_gemm::kernels::amx

#endif  // CPU_GEMM_HAS_AMX
#endif
