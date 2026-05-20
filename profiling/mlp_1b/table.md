# Nsight M4 Trace: reports/fundamental_1b/nsys_mlp_1b_profile.sqlite

## step.forward

### Stage Timeline

| Component | ms | % stage |
|---|---:|---:|
| cuda_kernel_busy_union | 78.5470 | 99.48% |
| gpu_no_kernel_time | 0.4073 | 0.52% |
| cuda_memcpy_union | 0.0032 | 0.00% |

### Host CUDA API

| Component | ms | % stage |
|---|---:|---:|
| cuda_runtime_api_sum_overlaps_gpu_timeline | 78.2856 | 99.15% |
| cuda_synchronization_api_sum_overlaps_gpu_timeline | 78.1995 | 99.04% |

### Operation Kernel Time

| Component | ms | % stage |
|---|---:|---:|
| forward.fc2.base_frozen_asymgemm | 42.3990 | 53.70% |
| forward.fc1.base_frozen_asymgemm | 36.1437 | 45.78% |
| forward.activation_relu | 0.0043 | 0.01% |

### Operation CUDA API Time

| Component | ms | % stage |
|---|---:|---:|
| forward.fc2.base_frozen_asymgemm | 35.9125 | 45.48% |
| forward.fc1.base_frozen_asymgemm | 0.0493 | 0.06% |
| forward.activation_relu | 0.0073 | 0.01% |

### Top Kernels

| Kernel | ms | % stage |
|---|---:|---:|
| `void asym_gemm::sm90_bf16_asym_gemm_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)64, (unsigned int)8192, (unsigned int)65536, (unsigned int)64, (unsigned int)64, (unsigned int)512, (unsigned int)1, (unsigned int)128, (unsigned int)128, (unsigned int)0, (unsigned int)2, (unsigned int)128, (unsigned int)128, (unsigned int)1, (bool)0, (unsigned int)128, (asym_gemm::GemmType)1, (bool)0, float, (unsigned long)100>(unsigned int *, unsigned int *, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)` | 42.3971 | 53.70% |
| `void asym_gemm::sm90_bf16_asym_gemm_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)64, (unsigned int)65536, (unsigned int)8192, (unsigned int)64, (unsigned int)64, (unsigned int)512, (unsigned int)1, (unsigned int)128, (unsigned int)128, (unsigned int)0, (unsigned int)2, (unsigned int)128, (unsigned int)128, (unsigned int)1, (bool)0, (unsigned int)128, (asym_gemm::GemmType)1, (bool)0, float, (unsigned long)100>(unsigned int *, unsigned int *, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)` | 36.1373 | 45.77% |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3)` | 0.0084 | 0.01% |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 9)]::operator ()() const::[lambda(c10::BFloat16) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3)` | 0.0043 | 0.01% |

## step.backward

### Stage Timeline

| Component | ms | % stage |
|---|---:|---:|
| cuda_kernel_busy_union | 70.8772 | 99.31% |
| gpu_no_kernel_time | 0.4859 | 0.68% |
| cuda_memcpy_union | 0.0031 | 0.00% |

### Host CUDA API

| Component | ms | % stage |
|---|---:|---:|
| cuda_runtime_api_sum_overlaps_gpu_timeline | 70.6103 | 98.94% |
| cuda_synchronization_api_sum_overlaps_gpu_timeline | 70.4946 | 98.78% |

### Operation Kernel Time

| Component | ms | % stage |
|---|---:|---:|
| backward.fc2.base_dx_asymgemm | 40.1208 | 56.22% |
| backward.fc1.base_dx_asymgemm | 30.7315 | 43.06% |
| backward.activation_relu | 0.0186 | 0.03% |
| backward.loss.mse | 0.0040 | 0.01% |

### Operation CUDA API Time

| Component | ms | % stage |
|---|---:|---:|
| backward.fc1.base_dx_asymgemm | 39.9769 | 56.02% |
| backward.fc2.base_dx_asymgemm | 0.0434 | 0.06% |
| backward.loss.mse | 0.0161 | 0.02% |
| backward.activation_relu | 0.0119 | 0.02% |

### Top Kernels

| Kernel | ms | % stage |
|---|---:|---:|
| `void asym_gemm::sm90_bf16_asym_gemm_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)1, (unsigned int)64, (unsigned int)65536, (unsigned int)8192, (unsigned int)64, (unsigned int)64, (unsigned int)64, (unsigned int)1, (unsigned int)128, (unsigned int)128, (unsigned int)0, (unsigned int)2, (unsigned int)128, (unsigned int)128, (unsigned int)1, (bool)0, (unsigned int)128, (asym_gemm::GemmType)1, (bool)0, float, (unsigned long)100>(unsigned int *, unsigned int *, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)` | 40.1143 | 56.21% |
| `void asym_gemm::sm90_bf16_asym_gemm_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)1, (unsigned int)64, (unsigned int)8192, (unsigned int)65536, (unsigned int)64, (unsigned int)64, (unsigned int)64, (unsigned int)1, (unsigned int)128, (unsigned int)128, (unsigned int)0, (unsigned int)2, (unsigned int)128, (unsigned int)128, (unsigned int)1, (bool)0, (unsigned int)128, (asym_gemm::GemmType)1, (bool)0, float, (unsigned long)100>(unsigned int *, unsigned int *, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)` | 30.7299 | 43.06% |
| `void at::native::unrolled_elementwise_kernel<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)3>, (int)4, TrivialOffsetCalculator<(int)2, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)2>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7)` | 0.0145 | 0.02% |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3)` | 0.0098 | 0.01% |
| `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::compare_scalar_kernel<c10::BFloat16>(at::TensorIteratorBase &, at::native::<unnamed>::OpType, T1)::[lambda(c10::BFloat16) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3)` | 0.0040 | 0.01% |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 0.0030 | 0.00% |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | 0.0010 | 0.00% |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | 0.0007 | 0.00% |
