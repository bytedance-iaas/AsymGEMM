#!/bin/bash
set -e

NVCC=/usr/local/cuda/bin/nvcc
CUTLASS_INC=/workspace/AsymGEMM_main/third-party/cutlass/include
CUTLASS_UTIL=/workspace/AsymGEMM_main/third-party/cutlass/tools/util/include
FLASH_SRC=/workspace/flash-attention/csrc/flash_attn/src
INCLUDES="\
  -I./include \
  -I${FLASH_SRC} \
  -I${CUTLASS_INC} \
  -I${CUTLASS_UTIL}"

DEFINES="\
  -D__CUDA_NO_HALF_OPERATORS__ \
  -D__CUDA_NO_HALF_CONVERSIONS__ \
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
  -D__CUDA_NO_HALF2_OPERATORS__ \
  -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED \
  -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
  -DNDEBUG \
  -DTORCH_API_INCLUDE_EXTENSION_H \
  -DTORCH_EXTENSION_NAME=flash_attn_3_cuda \
  -D_GLIBCXX_USE_CXX11_ABI=1"

CXX_FLAGS="\
  --expt-relaxed-constexpr \
  --compiler-options '-fPIC' \
  --threads 8 \
  -O3 -std=c++17 \
  --ftemplate-backtrace-limit=0 \
  --use_fast_math \
  -lineinfo"

ARCH="-gencode arch=compute_100,code=sm_100 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_80,code=sm_80"

echo "=== Compiling mixtureExpertKernel.cu → kernel.o ==="
${NVCC} ${INCLUDES} ${DEFINES} ${CXX_FLAGS} ${ARCH} \
  -c mixtureExpertKernel.cu -o kernel.o \
  2>&1 | tee compile_kernel.log

echo "=== Compiling test.cpp → test.o ==="
${NVCC} ${INCLUDES} ${DEFINES} ${CXX_FLAGS} ${ARCH} \
  -x cu -c test.cpp -o test.o \
  2>&1 | tee compile_test.log

echo "=== Linking → moe_test ==="
${NVCC} ${ARCH} kernel.o test.o -o moe_test \
  -lcudart \
  2>&1 | tee compile_link.log

echo "=== Build complete. Run: ./moe_test ==="
