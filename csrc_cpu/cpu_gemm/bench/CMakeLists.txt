add_executable(bench_bf16 bench_bf16.cpp)
target_link_libraries(bench_bf16 PRIVATE cpu_gemm)
target_include_directories(bench_bf16 PRIVATE ${CMAKE_SOURCE_DIR}/include
                                              ${CMAKE_SOURCE_DIR}/tests)

if(CPU_GEMM_WITH_AMX)
  add_executable(bench_amx_vs_avx2 bench_amx_vs_avx2.cpp)
  target_link_libraries(bench_amx_vs_avx2 PRIVATE cpu_gemm)
  target_include_directories(bench_amx_vs_avx2 PRIVATE ${CMAKE_SOURCE_DIR}/include
                                                       ${CMAKE_SOURCE_DIR}/tests)
endif()
