add_executable(test_avx2_bf16 test_avx2_bf16.cpp)
target_link_libraries(test_avx2_bf16 PRIVATE cpu_gemm)
target_include_directories(test_avx2_bf16 PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_avx2_bf16 COMMAND test_avx2_bf16)

add_executable(test_runtime test_runtime.cpp)
target_link_libraries(test_runtime PRIVATE cpu_gemm)
target_include_directories(test_runtime PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_runtime COMMAND test_runtime)

if(CPU_GEMM_WITH_AMX)
  add_executable(test_amx_bf16 test_amx_bf16.cpp)
  target_link_libraries(test_amx_bf16 PRIVATE cpu_gemm)
  target_include_directories(test_amx_bf16 PRIVATE ${CMAKE_SOURCE_DIR}/include)
  add_test(NAME test_amx_bf16 COMMAND test_amx_bf16)

  add_executable(test_amx_int8 test_amx_int8.cpp)
  target_link_libraries(test_amx_int8 PRIVATE cpu_gemm)
  target_include_directories(test_amx_int8 PRIVATE ${CMAKE_SOURCE_DIR}/include)
  add_test(NAME test_amx_int8 COMMAND test_amx_int8)

  add_executable(test_amx_int8_prepacked test_amx_int8_prepacked.cpp)
  target_link_libraries(test_amx_int8_prepacked PRIVATE cpu_gemm)
  target_include_directories(test_amx_int8_prepacked PRIVATE ${CMAKE_SOURCE_DIR}/include)
  add_test(NAME test_amx_int8_prepacked COMMAND test_amx_int8_prepacked)

  add_executable(test_amx_int8_rm test_amx_int8_rm.cpp)
  target_link_libraries(test_amx_int8_rm PRIVATE cpu_gemm)
  target_include_directories(test_amx_int8_rm PRIVATE ${CMAKE_SOURCE_DIR}/include)
  add_test(NAME test_amx_int8_rm COMMAND test_amx_int8_rm)
endif()

if(CPU_GEMM_WITH_AMX AND CPU_GEMM_WITH_AVX512)
  add_executable(test_int8_rm_parity test_int8_rm_parity.cpp)
  target_compile_definitions(test_int8_rm_parity PRIVATE
    CPU_GEMM_HAS_AMX=1 CPU_GEMM_HAS_AVX512_VNNI=1)
  target_link_libraries(test_int8_rm_parity PRIVATE cpu_gemm)
  target_include_directories(test_int8_rm_parity PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
  add_test(NAME test_int8_rm_parity COMMAND test_int8_rm_parity)
endif()
