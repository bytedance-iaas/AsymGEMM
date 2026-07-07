/*
 * cpu_gemm/types.h — dtype enum and tiny scalar types.
 *
 * The library does not depend on ggml; ggml interop lives in an optional
 * compat header.
 */
#ifndef CPU_GEMM_TYPES_H
#define CPU_GEMM_TYPES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Data type enumeration. Values are intentionally stable; please append. */
typedef enum cg_dtype {
  CG_F32                  = 0,
  CG_F16                  = 1,
  CG_BF16                 = 2,

  /* FP8 family, all E4M3. */
  CG_FP8_E4M3             = 10,  /* per-channel scale */
  CG_FP8_E4M3_BLK128      = 11,  /* 128x128 block scales */
  CG_FP8_E4M3_PERCHANNEL  = 12,  /* per-output-channel scales */

  /* MXFP4 (E2M1, 32-byte block, 16 elem). */
  CG_MXFP4_E2M1           = 20,

  CG_INT8                 = 30,
  /* INT8 weights pre-packed in the AMX kernel's blocked-VNNI layout with
   * per-channel FP32 scales appended. Produced by cg_pack_b_int8_amx; the
   * dispatcher routes (CG_BF16, CG_INT8_PACKED_AMX, CG_F32) straight to the
   * AMX INT8 path with no per-call B repacking. b_scales must be NULL: the
   * scales live at the tail of the packed buffer. */
  CG_INT8_PACKED_AMX      = 31,

  /* INT4 family. Group size and zero-point conventions differ. */
  CG_INT4_GPTQ_SYM        = 40,  /* zero fixed at 8 */
  CG_INT4_AWQ             = 41,  /* group scales + group zeros */
  CG_INT4_RAW             = 42,  /* signed [-8,7], 2 per byte */
  CG_INT4_K2              = 43,  /* group scales, no zeros */
} cg_dtype_t;

/* Per-operand quantization metadata. group_size == 0 means per-channel. */
typedef struct cg_quant_block {
  size_t       group_size;   /* elements per scale group (64, 128, 0) */
  cg_dtype_t   scale_dtype;  /* CG_F32 or CG_BF16 */
  int          has_zero;     /* 0 = symmetric, 1 = zero-point present */
} cg_quant_block_t;

/* CBLAS-shaped flags. */
typedef enum cg_order      { CG_ROW_MAJOR = 101, CG_COL_MAJOR = 102 } cg_order_t;
typedef enum cg_trans      { CG_NO_TRANS = 111, CG_TRANS = 112      } cg_trans_t;
typedef enum cg_offset_c   { CG_OFFSET_C_NONE = 0, CG_OFFSET_C_FIX,
                             CG_OFFSET_C_ROW, CG_OFFSET_C_COL       } cg_offset_c_t;

/* Status codes. */
typedef enum cg_status {
  CG_OK                   = 0,
  CG_E_INVALID            = -1,  /* malformed descriptor */
  CG_E_UNSUPPORTED        = -2,  /* dtype combo / shape not implemented */
  CG_E_ALIGNMENT          = -3,  /* pointer or stride alignment violated */
  CG_E_INTERNAL           = -4,
} cg_status_t;

/* Returns dtype size in bits — 0 for variable / packed types (caller must
 * know the layout). */
int cg_dtype_bits(cg_dtype_t dt);

/* Human-readable name, never NULL. */
const char* cg_dtype_name(cg_dtype_t dt);

/* ----------------------------------------------------------------------
 * BF16 scalar type — bit-identical to IEEE-754 single's upper 16 bits.
 * Defined here so callers don't need a ggml/torch dependency just to
 * construct a BF16 buffer.
 * ---------------------------------------------------------------------- */
typedef struct cg_bf16 { uint16_t bits; } cg_bf16_t;

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* CPU_GEMM_TYPES_H */
