/*
 * Intel AMX tile configuration and permission gating.
 *
 * Direct port of operators/amx/la/amx_config.hpp from ktransformers.
 * Cleaned up: no Windows-specific paths, no AVX-512 BF16/VBMI emulation
 * stubs (those will land in src/kernels/avx512/ when that backend is
 * lifted).
 *
 * Compilation: this header only compiles into TUs flagged with
 * -mamx-tile -mamx-bf16 -mamx-int8 (see CMakeLists.txt).
 */
#ifndef CPU_GEMM_KERNELS_AMX_TILE_CONFIG_H
#define CPU_GEMM_KERNELS_AMX_TILE_CONFIG_H

#if defined(CPU_GEMM_HAS_AMX)

#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <stdexcept>

namespace cpu_gemm::kernels::amx {

#define CG_AMX_ARCH_GET_XCOMP_SUPP   0x1021
#define CG_AMX_ARCH_GET_XCOMP_PERM   0x1022
#define CG_AMX_ARCH_REQ_XCOMP_PERM   0x1023
#define CG_AMX_XFEATURE_XTILECFG     17
#define CG_AMX_XFEATURE_XTILEDATA    18
#define CG_AMX_XFEATURE_MASK_XTILE   ((1 << CG_AMX_XFEATURE_XTILECFG) | (1 << CG_AMX_XFEATURE_XTILEDATA))
#define CG_AMX_XFEATURE_MASK_TDATA   (1 << CG_AMX_XFEATURE_XTILEDATA)

constexpr int TMMCount       = 8;
constexpr int MaxTileHeight  = 16;
constexpr int MaxTileWidth   = 64;
constexpr int AMX_BLK_SIZE   = 32;

/* Request AMX tile-data XSAVE permission from the kernel. Returns true if
 * AMX is now usable on the calling thread.
 *
 * MUST be __attribute__((noinline)): the kernel-side effect of
 * ARCH_REQ_XCOMP_PERM ("ldtilecfg is now legal on this thread") is invisible
 * to the compiler, so without an inlining barrier GCC will happily hoist a
 * downstream ldtilecfg above this call and SIGILL at runtime. */
__attribute__((noinline))
inline bool enable_amx() {
  unsigned long features = 0;
  long rc = syscall(SYS_arch_prctl, CG_AMX_ARCH_GET_XCOMP_SUPP, &features);
  if (rc) return false;
  if ((features & CG_AMX_XFEATURE_MASK_XTILE) != CG_AMX_XFEATURE_MASK_XTILE) return false;

  unsigned long bitmask = 0;
  if (syscall(SYS_arch_prctl, CG_AMX_ARCH_GET_XCOMP_PERM, &bitmask) != 0) return false;
  if (bitmask & CG_AMX_XFEATURE_MASK_TDATA) return true;
  if (syscall(SYS_arch_prctl, CG_AMX_ARCH_REQ_XCOMP_PERM, CG_AMX_XFEATURE_XTILEDATA) != 0) return false;
  if (syscall(SYS_arch_prctl, CG_AMX_ARCH_GET_XCOMP_PERM, &bitmask) != 0) return false;
  return (bitmask & CG_AMX_XFEATURE_MASK_TDATA) != 0;
}

struct alignas(64) TileConfig {
  uint8_t                       palette;
  uint8_t                       start_row;
  std::array<uint8_t, 14>       _pad0{};
  std::array<uint16_t, 8>       colsb;
  std::array<uint8_t, 16>       _pad1{};
  std::array<uint8_t, 8>        rows;
  std::array<uint8_t, 8>        _pad2{};

  TileConfig() : palette(1), start_row(0), colsb{}, rows{} {}

  void set_row_col(int i, uint8_t r, uint16_t c) {
    colsb[i] = c;
    rows[i]  = r;
  }

  void load() { _tile_loadconfig(this); }
};

static_assert(sizeof(TileConfig) == 64, "TileConfig must be exactly 64 bytes");

}  // namespace cpu_gemm::kernels::amx

#endif  // CPU_GEMM_HAS_AMX
#endif  // CPU_GEMM_KERNELS_AMX_TILE_CONFIG_H
