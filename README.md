# AsymGEMM

AsymGEMM is a library for clean and efficient general matrix multiplication (GEMM) on NVIDIA Superchips. Unlike conventional GEMM implementations, AsymGEMM allows one input matrix to reside in CPU memory while the GPU kernel accesses it directly, without copying it into HBM. The key idea is to leverage the high-bandwidth NVLink-C2C interconnect between the CPU and GPU.

AsymGEMM leverages some concepts from [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [CUTLASS](https://github.com/nvidia/cutlass) and [CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute).

## News

- 2026.02.26: AsymGEMM support BF16 MoE computation.

## Roadmap

- [x] Support BF16

## Quick start

### Requirements

- NVIDIA SM100 architecture GPU
- Python 3.8 or higher
- Compilers with C++20 support
- CUDA Toolkit:
  - CUDA 12.9 or higher for SM100
- PyTorch 2.1 or higher
- CUTLASS 4.0 or higher (could be cloned by Git submodule)

### Development


### Installation

```bash
cat install.sh
./install.sh
```

Then, import `asym_gemm` in your Python project, and enjoy!

## Interfaces

#### Notices


#### Utilities

The library provides some utility functions besides the above kernels:

## Acknowledgement

AsymGEMM is inspired by the [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) project. Thanks and respect to the developers!

## License

This code repository is released under [the MIT License](LICENSE).
