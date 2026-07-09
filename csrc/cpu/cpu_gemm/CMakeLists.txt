cmake_minimum_required(VERSION 3.18)
project(cpu_gemm LANGUAGES CXX C VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------
option(CPU_GEMM_BUILD_TESTS    "Build correctness tests"         ON)
option(CPU_GEMM_BUILD_BENCH    "Build microbenchmarks"           ON)
option(CPU_GEMM_BUILD_EXAMPLES "Build example programs"          ON)

# Backend toggles (default ON if compiler can target them; user can override).
option(CPU_GEMM_WITH_AVX2      "Build AVX2 backend"              ON)
option(CPU_GEMM_WITH_AVX512    "Build AVX-512 backend"           OFF)
option(CPU_GEMM_WITH_AMX       "Build Intel AMX backend"         ON)
option(CPU_GEMM_WITH_AOCL      "Build AMD AOCL/BLIS bridge"      OFF)

# --------------------------------------------------------------------------
# Warning + sanity flags shared by every TU
# --------------------------------------------------------------------------
add_library(cpu_gemm_common_flags INTERFACE)
target_compile_options(cpu_gemm_common_flags INTERFACE
  -Wall -Wextra -Wno-unused-parameter -Wno-unused-function
  -fno-omit-frame-pointer
)

# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
set(CPU_GEMM_SOURCES
  src/runtime/worker_pool.cpp
  src/runtime/scratch_arena.cpp
  src/runtime/runtime.cpp
  src/dispatch/gemm.cpp
  src/dispatch/dtype.cpp
)

set(CPU_GEMM_BACKEND_LIBS "")

if(CPU_GEMM_WITH_AVX2)
  add_library(cpu_gemm_avx2 OBJECT
    src/kernels/avx2/bf16_gemm.cpp
  )
  target_compile_options(cpu_gemm_avx2 PRIVATE
    -mavx2 -mfma -mf16c
  )
  target_include_directories(cpu_gemm_avx2 PRIVATE include src)
  target_link_libraries(cpu_gemm_avx2 PRIVATE cpu_gemm_common_flags)
  list(APPEND CPU_GEMM_BACKEND_LIBS cpu_gemm_avx2)
endif()

if(CPU_GEMM_WITH_AMX)
  add_library(cpu_gemm_amx OBJECT
    src/kernels/amx/bf16_gemm.cpp
    src/kernels/amx/int8_gemm.cpp
    src/kernels/amx/int8_gemm_rm.cpp
  )
  target_compile_options(cpu_gemm_amx PRIVATE
    -mamx-tile -mamx-bf16 -mamx-int8
    -mavx512f -mavx512bw -mavx512vl -mavx512dq
    -mavx512bf16 -mfma
  )
  target_compile_definitions(cpu_gemm_amx PRIVATE CPU_GEMM_HAS_AMX=1)
  target_include_directories(cpu_gemm_amx PRIVATE include src)
  target_link_libraries(cpu_gemm_amx PRIVATE cpu_gemm_common_flags)
  list(APPEND CPU_GEMM_BACKEND_LIBS cpu_gemm_amx)
endif()

# Placeholders for the other backends — they'll be wired in by their own
# steps in cpu_gemm.md. Each one becomes an OBJECT library with the right
# -march flags, and the dispatcher selects among them.

add_library(cpu_gemm STATIC ${CPU_GEMM_SOURCES})
target_include_directories(cpu_gemm PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_include_directories(cpu_gemm PRIVATE src)
target_link_libraries(cpu_gemm PRIVATE cpu_gemm_common_flags
                                       ${CPU_GEMM_BACKEND_LIBS}
                                       Threads::Threads)

find_package(Threads REQUIRED)

if(CPU_GEMM_WITH_AVX2)
  target_compile_definitions(cpu_gemm PRIVATE CPU_GEMM_HAS_AVX2=1)
  target_sources(cpu_gemm PRIVATE $<TARGET_OBJECTS:cpu_gemm_avx2>)
endif()

if(CPU_GEMM_WITH_AMX)
  target_compile_definitions(cpu_gemm PRIVATE CPU_GEMM_HAS_AMX=1)
  target_sources(cpu_gemm PRIVATE $<TARGET_OBJECTS:cpu_gemm_amx>)
endif()

# --------------------------------------------------------------------------
# Tests / examples / bench
# --------------------------------------------------------------------------
if(CPU_GEMM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(CPU_GEMM_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(CPU_GEMM_BUILD_BENCH)
  add_subdirectory(bench)
endif()

# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS cpu_gemm
        EXPORT cpu_gemm-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

message(STATUS "cpu_gemm: backends — "
  "AVX2=${CPU_GEMM_WITH_AVX2} "
  "AVX512=${CPU_GEMM_WITH_AVX512} "
  "AMX=${CPU_GEMM_WITH_AMX} "
  "AOCL=${CPU_GEMM_WITH_AOCL}")
