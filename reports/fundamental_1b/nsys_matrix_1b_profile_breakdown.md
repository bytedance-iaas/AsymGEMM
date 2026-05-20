# Nsight M4 Trace: reports/fundamental_1b/nsys_matrix_1b_profile.sqlite

## step.forward

### Stage Timeline

| Component | ms | % stage |
|---|---:|---:|
| cuda_kernel_busy_union | 76.7354 | 99.63% |
| gpu_no_kernel_time | 0.2795 | 0.36% |
| cuda_memcpy_union | 0.0018 | 0.00% |

### Host CUDA API

| Component | ms | % stage |
|---|---:|---:|
| cuda_runtime_api_sum_overlaps_gpu_timeline | 76.7011 | 99.59% |
| cuda_synchronization_api_sum_overlaps_gpu_timeline | 76.6590 | 99.54% |

### Operation Kernel Time

| Component | ms | % stage |
|---|---:|---:|
| forward.matrix.base_frozen_asymgemm | 76.7354 | 99.63% |

### Operation CUDA API Time

| Component | ms | % stage |
|---|---:|---:|
| forward.matrix.base_frozen_asymgemm | 0.0510 | 0.07% |

### Top Kernels

| Kernel | ms | % stage |
|---|---:|---:|
| `void asym_gemm::sm90_bf16_asym_gemm_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)64, (unsigned int)32768, (unsigned int)32768, (unsigned int)64, (unsigned int)64, (unsigned int)512, (unsigned int)1, (unsigned int)128, (unsigned int)128, (unsigned int)0, (unsigned int)2, (unsigned int)128, (unsigned int)128, (unsigned int)1, (bool)0, (unsigned int)128, (asym_gemm::GemmType)1, (bool)0, float, (unsigned long)100>(unsigned int *, unsigned int *, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)` | 76.7316 | 99.63% |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3)` | 0.0038 | 0.00% |

## step.backward

### Stage Timeline

| Component | ms | % stage |
|---|---:|---:|
| cuda_kernel_busy_union | 80.3023 | 99.42% |
| gpu_no_kernel_time | 0.4650 | 0.58% |
| cuda_memcpy_union | 0.0018 | 0.00% |

### Host CUDA API

| Component | ms | % stage |
|---|---:|---:|
| cuda_runtime_api_sum_overlaps_gpu_timeline | 80.2791 | 99.39% |
| cuda_synchronization_api_sum_overlaps_gpu_timeline | 80.2109 | 99.31% |

### Operation Kernel Time

| Component | ms | % stage |
|---|---:|---:|
| backward.matrix.base_dx_asymgemm | 80.2907 | 99.41% |
| backward.loss.mse | 0.0078 | 0.01% |

### Operation CUDA API Time

| Component | ms | % stage |
|---|---:|---:|
| backward.matrix.base_dx_asymgemm | 0.0430 | 0.05% |
| backward.loss.mse | 0.0225 | 0.03% |

### Top Kernels

| Kernel | ms | % stage |
|---|---:|---:|
| `void asym_gemm::sm90_bf16_asym_gemm_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)1, (unsigned int)64, (unsigned int)32768, (unsigned int)32768, (unsigned int)64, (unsigned int)64, (unsigned int)64, (unsigned int)1, (unsigned int)128, (unsigned int)128, (unsigned int)0, (unsigned int)2, (unsigned int)128, (unsigned int)128, (unsigned int)1, (bool)0, (unsigned int)128, (asym_gemm::GemmType)1, (bool)0, float, (unsigned long)100>(unsigned int *, unsigned int *, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)` | 80.2868 | 99.40% |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3)` | 0.0069 | 0.01% |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 0.0067 | 0.01% |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | 0.0012 | 0.00% |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | 0.0008 | 0.00% |
