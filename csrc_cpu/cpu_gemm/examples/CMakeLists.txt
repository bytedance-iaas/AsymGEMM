add_executable(simple_bf16 simple_bf16.cpp)
target_link_libraries(simple_bf16 PRIVATE cpu_gemm)
target_include_directories(simple_bf16 PRIVATE ${CMAKE_SOURCE_DIR}/include)
