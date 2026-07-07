/* Quick probe: can we actually enable AMX and run a single tile op? */
#include <cstdio>
#include <cstring>

#define CPU_GEMM_HAS_AMX 1
#include "kernels/amx/tile_config.h"

#include <immintrin.h>

int main() {
  bool ok = cpu_gemm::kernels::amx::enable_amx();
  std::printf("enable_amx -> %s\n", ok ? "true" : "false");
  if (!ok) return 1;

  cpu_gemm::kernels::amx::TileConfig cfg;
  // Two A tiles 16x64, two B tiles 16x64, four C tiles 16x64
  for (int i = 0; i < 2; i++) cfg.set_row_col(i, 16, 64);
  for (int i = 2; i < 4; i++) cfg.set_row_col(i, 16, 64);
  for (int i = 4; i < 8; i++) cfg.set_row_col(i, 16, 64);
  cfg.load();
  std::printf("tile config loaded\n");

  alignas(64) uint8_t a[16*64] = {};
  alignas(64) uint8_t b[16*64] = {};
  alignas(64) float   c[16*16] = {};

  _tile_loadd(0, a, 64);
  _tile_loadd(2, b, 64);
  _tile_zero(4);
  _tile_dpbf16ps(4, 0, 2);
  _tile_stored(4, c, 64);
  _tile_release();
  std::printf("tile op succeeded\n");
  return 0;
}
