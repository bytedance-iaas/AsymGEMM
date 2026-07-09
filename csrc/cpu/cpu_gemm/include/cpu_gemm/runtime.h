/*
 * cpu_gemm/runtime.h — opaque runtime handle.
 *
 * Wraps a small work-stealing thread pool. The pool is created once per
 * process; cg_gemm() fans out across it. Single-thread callers can use
 * cg_gemm_st() in cpu_gemm.h instead and skip the runtime entirely.
 */
#ifndef CPU_GEMM_RUNTIME_H
#define CPU_GEMM_RUNTIME_H

#include "cpu_gemm/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cg_runtime cg_runtime_t;

/* Create a runtime backed by `n_threads` worker threads. Pass 0 to use
 * std::thread::hardware_concurrency(). */
cg_runtime_t* cg_runtime_create(int n_threads);

void cg_runtime_destroy(cg_runtime_t* rt);

/* Number of worker threads. */
int cg_runtime_threads(const cg_runtime_t* rt);

/* Capability bits the dispatcher will consider on this host. Useful for
 * diagnostics; the dispatcher does its own probing. */
typedef struct cg_caps {
  int has_avx2;
  int has_fma;
  int has_avx512f;
  int has_avx512_bf16;
  int has_avx_vnni;       /* AVX-VNNI (256-bit, leaf 7.1 EAX[4]) */
  int has_avx512_vnni;    /* AVX-512-VNNI (leaf 7.0 ECX[11]) */
  int has_amx_bf16;
  int has_amx_int8;
} cg_caps_t;

cg_caps_t cg_query_caps(void);

/* Diagnostics for the runtime-selected INT8 row-major backend
 * (dispatch/int8_rm_backend). Name is one of "amx_int8_rm",
 * "avx512_vnni_int8_rm", "none"; ok is 0 when no backend is usable. */
const char* cg_int8_rm_backend_name(void);
int cg_int8_rm_backend_ok(void);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* CPU_GEMM_RUNTIME_H */
