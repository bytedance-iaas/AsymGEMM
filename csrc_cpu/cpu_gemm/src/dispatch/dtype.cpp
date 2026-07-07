#include "cpu_gemm/types.h"

extern "C" int cg_dtype_bits(cg_dtype_t dt) {
  switch (dt) {
    case CG_F32: return 32;
    case CG_F16: return 16;
    case CG_BF16: return 16;
    case CG_FP8_E4M3:
    case CG_FP8_E4M3_BLK128:
    case CG_FP8_E4M3_PERCHANNEL: return 8;
    case CG_MXFP4_E2M1: return 4;
    case CG_INT8: return 8;
    case CG_INT8_PACKED_AMX: return 8;
    case CG_INT4_GPTQ_SYM:
    case CG_INT4_AWQ:
    case CG_INT4_RAW:
    case CG_INT4_K2: return 4;
  }
  return 0;
}

extern "C" const char* cg_dtype_name(cg_dtype_t dt) {
  switch (dt) {
    case CG_F32: return "f32";
    case CG_F16: return "f16";
    case CG_BF16: return "bf16";
    case CG_FP8_E4M3: return "fp8_e4m3";
    case CG_FP8_E4M3_BLK128: return "fp8_e4m3_blk128";
    case CG_FP8_E4M3_PERCHANNEL: return "fp8_e4m3_perchannel";
    case CG_MXFP4_E2M1: return "mxfp4_e2m1";
    case CG_INT8: return "int8";
    case CG_INT8_PACKED_AMX: return "int8_packed_amx";
    case CG_INT4_GPTQ_SYM: return "int4_gptq_sym";
    case CG_INT4_AWQ: return "int4_awq";
    case CG_INT4_RAW: return "int4_raw";
    case CG_INT4_K2: return "int4_k2";
  }
  return "unknown";
}
