# Nsight CPU Gap Debug: reports/debug_cpu_gaps/wrapper_smoke/mlp/cpu_gaps/trace.sqlite

- CPU sample percentages are sampling shares on stage submission threads, not exact elapsed-time shares.
- Small gaps can be below the CPU sampling period; those rows still keep CUDA API, OSRT, scheduler, and enclosing NVTX attribution.
- Use the regular Nsight table.md/profile.json for low-overhead end-to-end timing truth.

## step.forward

- Stage total: 1135.5925 ms
- GPU no-kernel gap total: 1135.4757 ms (99.99% of stage)
- Gap count: 33
- Thread scope: NVTX stage thread plus CUDA runtime submission threads (1 tids)

### CPU Sample Buckets During Gaps

| Bucket | Samples | % samples |
|---|---|---|
| cuda_runtime_or_driver | 155 | 57.84% |
| python_interpreter_or_model_code | 47 | 17.54% |
| pytorch_autograd_engine | 41 | 15.30% |
| allocator_or_memory | 24 | 8.96% |
| pytorch_dispatch_or_aten | 1 | 0.37% |

### OS Runtime API Overlap In Stage

| OSRT API | overlap ms | summed % stage |
|---|---|---|
| fgets | 937.0785 | 82.52% |
| ioctl | 26.5633 | 2.34% |
| popen | 1.4092 | 0.12% |
| fread | 0.0723 | 0.01% |
| mmap | 0.0375 | 0.00% |
| fopen | 0.0292 | 0.00% |
| fclose | 0.0030 | 0.00% |

### CUDA Runtime API Overlap In Stage

| CUDA API | overlap ms | summed % stage |
|---|---|---|
| cudaLaunchKernel_v7000 | 69.1415 | 6.09% |
| cuLibraryLoadData | 17.6067 | 1.55% |
| cudaFree_v3020 | 1.7640 | 0.16% |
| cuKernelGetFunction | 1.4511 | 0.13% |
| cudaGetDeviceProperties_v12000 | 1.3702 | 0.12% |
| cuModuleLoad | 0.7618 | 0.07% |
| cudaMalloc_v3020 | 0.5406 | 0.05% |
| cuGetProcAddress_v2 | 0.1860 | 0.02% |

### Scheduler Off-CPU Overlap In Stage

| Block/state | overlap ms | summed % stage |
|---|---|---|
| unknown/Unknown | 937.9994 | 82.60% |

### Largest GPU No-Kernel Gaps

| offset ms | gap ms | % stage | enclosing NVTX | previous kernel | next kernel | CPU bucket | top CPU leaf | top CUDA API | top OSRT | ctxsw |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5871 | 627.9694 | 55.30% | `forward.fc1.base_frozen_asymgemm` | `<stage_start>` | `void asym_gemm::sm90_bf16_asym_gemm_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)64, (unsigned int)256, (unsigned int)512, (unsigned int)64, (unsigned int)64, (unsigned int)512, (unsigned int)1, (unsigned int)128, (unsigned int)128, (unsigned int)0, (unsigned int)2, (unsigned int)128, (unsigned int)128, (unsigned int)1, (bool)0, (unsigned int)4, (asym_gemm::GemmType)1, (bool)0, float, (unsigned long)100>(unsigned int *, unsigned int *, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)` | allocator_or_memory (4) | `0xffffffffc318fca6 (2)` | cudaGetDeviceProperties_v12000 (1.3702 ms) | fgets (624.1039 ms) | 16 |
| 819.1502 | 314.7165 | 27.71% | `forward.fc2.base_frozen_asymgemm` | `void at::native::vectorized_elementwise_kernel<(int)8, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 9)]::operator ()() const::[lambda(c10::BFloat16) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3)` | `void asym_gemm::sm90_bf16_asym_gemm_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)64, (unsigned int)128, (unsigned int)512, (unsigned int)64, (unsigned int)64, (unsigned int)512, (unsigned int)1, (unsigned int)128, (unsigned int)128, (unsigned int)0, (unsigned int)2, (unsigned int)128, (unsigned int)128, (unsigned int)1, (bool)0, (unsigned int)2, (asym_gemm::GemmType)1, (bool)0, float, (unsigned long)100>(unsigned int *, unsigned int *, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)` | allocator_or_memory (1) | `0xffffffff9e727bfd (1)` | cuModuleLoad (0.4386 ms) | fgets (312.9746 ms) | 8 |
| 645.5036 | 88.0042 | 7.75% | `forward.fc1.lora_A` | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7)` | `void cutlass::Kernel2<cutlass_80_simt_sgemm_32x128_8x5_tn_align1>(T1::Params)` | cuda_runtime_or_driver (63) | `0xffffffffc318fca6 (25)` | cuLibraryLoadData (17.6067 ms) | ioctl (21.4298 ms) | 4 |
| 745.8947 | 30.7745 | 2.71% | `step.forward` | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bool)0, (bool)0>(cublasLt::cublasSplitKParams<T6>, const T4 *, const T10 *, T9 *, T5 *, const T6 *, const T6 *, const T11 *, const T4 *, T11 *, void *, long, T6 *, int *, T6 *, T6 *, const T6 *, const T6 *, const T6 *, const T6 *, const T6 *)` | `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_stage3_warpsize1x2x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas` | cuda_runtime_or_driver (36) | `0xffffffff9ee46337 (9)` | cudaStreamCreate_v3020 (0.0237 ms) | ioctl (0.0062 ms) | 0 |
| 790.8077 | 27.1260 | 2.39% | `step.forward` | `void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3)` | `void at::native::vectorized_elementwise_kernel<(int)8, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorBase &, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 9)]::operator ()() const::[lambda(c10::BFloat16) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3)` | cuda_runtime_or_driver (36) | `0xffffffff9e93ef6f (4)` | cudaLaunchKernel_v7000 (26.8339 ms) | - | 0 |
| 777.1702 | 13.5567 | 1.19% | `forward.fc1.add_cast_scale` | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | pytorch_autograd_engine (17) | `0xffffffff9f000be0 (5)` | cudaLaunchKernel_v7000 (13.4736 ms) | - | 0 |
| 733.5120 | 12.3814 | 1.09% | `forward.fc1.lora_A` | `void cutlass::Kernel2<cutlass_80_simt_sgemm_32x128_8x5_tn_align1>(T1::Params)` | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bool)0, (bool)0>(cublasLt::cublasSplitKParams<T6>, const T4 *, const T10 *, T9 *, T5 *, const T6 *, const T6 *, const T11 *, const T4 *, T11 *, void *, long, T6 *, int *, T6 *, T6 *, const T6 *, const T6 *, const T6 *, const T6 *, const T6 *)` | cuda_runtime_or_driver (15) | `0xffffffff9ee46337 (3)` | cudaLaunchKernel_v7000 (12.3618 ms) | - | 0 |
| 628.6034 | 12.3649 | 1.09% | `forward.fc1.base_frozen_asymgemm` | `void asym_gemm::sm90_bf16_asym_gemm_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)64, (unsigned int)256, (unsigned int)512, (unsigned int)64, (unsigned int)64, (unsigned int)512, (unsigned int)1, (unsigned int)128, (unsigned int)128, (unsigned int)0, (unsigned int)2, (unsigned int)128, (unsigned int)128, (unsigned int)1, (bool)0, (unsigned int)4, (asym_gemm::GemmType)1, (bool)0, float, (unsigned long)100>(unsigned int *, unsigned int *, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)` | `void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3)` | pytorch_autograd_engine (17) | `0xffffffff9f000be0 (2)` | cudaLaunchKernel_v7000 (12.2460 ms) | - | 0 |

### Top CPU Stack Signatures During Gaps

| Bucket | Samples | % samples | Stack leaf-to-root |
|---|---|---|---|
| cuda_runtime_or_driver | 7 | 2.61% | `0xffffffffc318fca6 <- NSYS_OSRT_ioctl_0 <- 0x7f1a5d68dec3 <- 0x7f1a5d698097 <- 0x7f1a5d6740fa` |
| cuda_runtime_or_driver | 7 | 2.61% | `0xffffffff9ee46337 <- 0x7f1a5d707516 <- 0x7f1a5d711a5d <- 0x7f1a5d4b8905 <- 0x7f1a5d4da064` |
| cuda_runtime_or_driver | 6 | 2.24% | `0xffffffffc318fca6 <- NSYS_OSRT_ioctl_0 <- 0x7f1a5d68dec3 <- 0x7f1a5d693481 <- 0x7f1a5d697ae3` |
| cuda_runtime_or_driver | 6 | 2.24% | `0xffffffffc318fca6 <- NSYS_OSRT_ioctl_0 <- 0x7f1a5d68dec3 <- 0x7f1a5d698097 <- 0x7f1a5d67415c` |
| cuda_runtime_or_driver | 6 | 2.24% | `0xffffffff9e93ef6f <- 0x7f1a5d46767c <- 0x7f1a5d4db777 <- 0x7f1a5d4dc615 <- 0x7f1a5d3b8dd2` |
| cuda_runtime_or_driver | 6 | 2.24% | `0x7f1b9e6ae6d2 <- 0x7f191422dcf1 <- 0x7f1914209969 <- 0x7f19142ec04f <- 0x7f19142e1ce8` |
| cuda_runtime_or_driver | 5 | 1.87% | `0xffffffff9ee46337 <- 0x7f1a5d46767c <- 0x7f1a5d4db777 <- 0x7f1a5d4dc615 <- 0x7f1a5d3b8dd2` |
| cuda_runtime_or_driver | 4 | 1.49% | `0xffffffffc318fca6 <- NSYS_OSRT_ioctl_0 <- 0x7f1a5d68dec3 <- 0x7f1a5d698097 <- 0x7f1a5d6796c8` |

## step.backward

- Stage total: 649.1153 ms
- GPU no-kernel gap total: 649.0306 ms (99.99% of stage)
- Gap count: 41
- Thread scope: NVTX stage thread plus CUDA runtime submission threads (2 tids)

### CPU Sample Buckets During Gaps

| Bucket | Samples | % samples |
|---|---|---|
| allocator_or_memory | 15 | 42.86% |
| pytorch_autograd_engine | 12 | 34.29% |
| cuda_runtime_or_driver | 8 | 22.86% |

### OS Runtime API Overlap In Stage

| OSRT API | overlap ms | summed % stage |
|---|---|---|
| pthread_cond_wait | 642.0312 | 98.91% |
| fgets | 622.2643 | 95.86% |
| pthread_create | 1.7099 | 0.26% |
| popen | 0.9266 | 0.14% |
| ioctl | 0.4350 | 0.07% |
| fread | 0.0871 | 0.01% |
| fopen | 0.0112 | 0.00% |
| mmap64 | 0.0072 | 0.00% |

### CUDA Runtime API Overlap In Stage

| CUDA API | overlap ms | summed % stage |
|---|---|---|
| cudaLaunchKernel_v7000 | 15.4365 | 2.38% |
| cuModuleLoad | 0.8361 | 0.13% |
| cudaMalloc_v3020 | 0.5597 | 0.09% |
| cuKernelGetFunction | 0.1613 | 0.02% |
| cuLaunchKernelEx | 0.0702 | 0.01% |
| cudaMemcpyAsync_v3020 | 0.0578 | 0.01% |
| cuLaunchKernel | 0.0438 | 0.01% |
| cudaLaunchKernelExC_v11060 | 0.0355 | 0.01% |

### Scheduler Off-CPU Overlap In Stage

| Block/state | overlap ms | summed % stage |
|---|---|---|
| unknown/Unknown | 1264.8801 | 194.86% |

### Largest GPU No-Kernel Gaps

| offset ms | gap ms | % stage | enclosing NVTX | previous kernel | next kernel | CPU bucket | top CPU leaf | top CUDA API | top OSRT | ctxsw |
|---|---|---|---|---|---|---|---|---|---|---|
| 334.4779 | 313.8635 | 48.35% | `backward.fc1.base_dx_asymgemm` | `void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3)` | `void asym_gemm::sm90_bf16_asym_gemm_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)1, (unsigned int)64, (unsigned int)128, (unsigned int)256, (unsigned int)64, (unsigned int)64, (unsigned int)64, (unsigned int)1, (unsigned int)128, (unsigned int)128, (unsigned int)0, (unsigned int)2, (unsigned int)128, (unsigned int)128, (unsigned int)1, (bool)0, (unsigned int)2, (asym_gemm::GemmType)1, (bool)0, float, (unsigned long)100>(unsigned int *, unsigned int *, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)` | allocator_or_memory (2) | `0xffffffff9ede5f69 (1)` | cuModuleLoad (0.3969 ms) | pthread_cond_wait (313.8635 ms) | 8 |
| 12.0439 | 311.1554 | 47.94% | `backward.fc2.base_dx_asymgemm` | `void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3)` | `void asym_gemm::sm90_bf16_asym_gemm_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)1, (unsigned int)64, (unsigned int)256, (unsigned int)128, (unsigned int)64, (unsigned int)64, (unsigned int)64, (unsigned int)1, (unsigned int)128, (unsigned int)128, (unsigned int)0, (unsigned int)2, (unsigned int)128, (unsigned int)128, (unsigned int)1, (bool)0, (unsigned int)4, (asym_gemm::GemmType)1, (bool)0, float, (unsigned long)100>(unsigned int *, unsigned int *, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)` | allocator_or_memory (2) | `0xffffffff9edd656e (1)` | cuModuleLoad (0.4392 ms) | pthread_cond_wait (311.1554 ms) | 8 |
| 323.7651 | 9.2170 | 1.42% | `step.backward` | `void at::native::vectorized_elementwise_kernel<(int)8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::compare_scalar_kernel<c10::BFloat16>(at::TensorIteratorBase &, at::native::<unnamed>::OpType, T1)::[lambda(c10::BFloat16) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3)` | pytorch_autograd_engine (12) | `0xffffffff9ede2917 (2)` | cudaLaunchKernel_v7000 (8.9491 ms) | pthread_cond_wait (9.2170 ms) | 2 |
| 0.0000 | 6.3143 | 0.97% | `step.backward` | `<stage_start>` | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | cuda_runtime_or_driver (7) | `0xffffffff9f000be0 (2)` | cudaLaunchKernel_v7000 (5.9946 ms) | - | 0 |
| 6.3149 | 3.0497 | 0.47% | `step.backward` | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, std::array<char *, (unsigned long)1>>(int, T2, T3)` | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | allocator_or_memory (4) | `0xffffffff9e727be1 (1)` | cudaLaunchKernel_v7000 (0.0337 ms) | pthread_cond_wait (2.2428 ms) | 8 |
| 9.8760 | 1.1570 | 0.18% | `step.backward` | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nn_align1>(T1::Params)` | allocator_or_memory (1) | `0xffffffff9edcd487 (1)` | cudaMalloc_v3020 (0.5597 ms) | pthread_cond_wait (1.1570 ms) | 0 |
| 323.3608 | 0.4033 | 0.06% | `step.backward` | `void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3)` | `void at::native::vectorized_elementwise_kernel<(int)8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | allocator_or_memory (1) | `0x7f1a5d5e74df (1)` | cudaLaunchKernel_v7000 (0.0699 ms) | pthread_cond_wait (0.4033 ms) | 0 |
| 11.0518 | 0.3445 | 0.05% | `step.backward` | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bool)0, (bool)0>(cublasLt::cublasSplitKParams<T6>, const T4 *, const T10 *, T9 *, T5 *, const T6 *, const T6 *, const T11 *, const T4 *, T11 *, void *, long, T6 *, int *, T6 *, T6 *, const T6 *, const T6 *, const T6 *, const T6 *, const T6 *)` | `sm80_xmma_gemm_f32f32_f32f32_f32_nt_n_tilesize32x32x8_stage3_warpsize1x2x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas` | off_cpu_wait_no_cpu_sample: unknown/Unknown | `no_cpu_sample_in_gap` | cudaLaunchKernelExC_v11060 (0.0131 ms) | pthread_cond_wait (0.3445 ms) | 0 |

### Top CPU Stack Signatures During Gaps

| Bucket | Samples | % samples | Stack leaf-to-root |
|---|---|---|---|
| allocator_or_memory | 1 | 2.86% | `std::pair<ska::detailv3::sherwood_v3_table<c10::cuda::CUDACachingAllocator::N... <- c10::cuda::CUDACachingAllocator::Native::DeviceCachingAllocator::alloc_found_... <- c10::cuda::CUDACachingAllocator::Native::DeviceCachingAllocator::malloc(signe... <- c10::cuda::CUDACachingAllocator::Native::NativeCachingAllocator::malloc(void*... <- c10::cuda::CUDACachingAllocator::Native::NativeCachingAllocator::allocate(uns...` |
| cuda_runtime_or_driver | 1 | 2.86% | `0xffffffff9e8f6c85 <- 0x7f1a5d707516 <- 0x7f1a5d711a5d <- 0x7f1a5d4b8905 <- 0x7f1a5d4da064` |
| cuda_runtime_or_driver | 1 | 2.86% | `0xffffffff9ee46337 <- 0x7f1a5d46766e <- 0x7f1a5d4db777 <- 0x7f1a5d4dc615 <- 0x7f1a5d3b8dd2` |
| cuda_runtime_or_driver | 1 | 2.86% | `0xffffffff9f00153b <- 0x7f1a5d46767c <- 0x7f1a5d4db777 <- 0x7f1a5d4dc615 <- 0x7f1a5d3b8dd2` |
| cuda_runtime_or_driver | 1 | 2.86% | `0xffffffff9f000be0 <- 0x7f1a5d4b8faa <- 0x7f1a5d47e403 <- 0x7f1a5d4db816 <- 0x7f1a5d4dc615` |
| cuda_runtime_or_driver | 1 | 2.86% | `0xffffffff9ee30d1e <- 0x7f1a5d4b8faa <- 0x7f1a5d47e403 <- 0x7f1a5d4db816 <- 0x7f1a5d4dc615` |
| allocator_or_memory | 1 | 2.86% | `0xffffffff9ede2917 <- __libc_malloc <- 0x7f1a5e14eeee <- 0x7f1a5d47eed3 <- 0x7f1a5d47e403` |
| cuda_runtime_or_driver | 1 | 2.86% | `0xffffffff9f000be0 <- 0x7f191422dcf1 <- 0x7f1914209969 <- 0x7f19142ec04f <- 0x7f19142e1ce8` |
