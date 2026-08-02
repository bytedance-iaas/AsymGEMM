# Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
# Modified by Bytedance Inc., 2026.
# Original: https://github.com/deepseek-ai/DeepGEMM

import os
import platform
import shutil
import setuptools
import torch
from setuptools import find_packages
from setuptools.command.build_py import build_py
from pathlib import Path

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
except ImportError:
    BuildExtension = None
    CUDAExtension = None
    CUDA_HOME = None

# Project root
current_dir = os.path.dirname(os.path.realpath(__file__))

# Third-party include directories to bundle into the wheel
third_party_include_dirs = [
    'third-party/cutlass/include/cute',
    'third-party/cutlass/include/cutlass',
]


def get_ext_modules():
    if CUDA_HOME is None or CUDAExtension is None:
        return []

    cxx_flags = [
        '-std=c++17', '-O3', '-fPIC',
        '-Wno-psabi', '-Wno-deprecated-declarations',
        f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}',
        # gb200_tp.md I1 FIX A: context-INDEPENDENT runtime-API kernel loading
        # (cudaLibraryLoadFromFile -> cudaKernel_t, materialized per-device on demand), so one
        # process can launch the same JIT'd GEMM on BOTH GPUs. Requires CUDART >= 12.8.
        '-DDG_JIT_USE_RUNTIME_API',
    ]

    # Grace CPU fused ops (csrc/cpu_ops): OpenMP + SVE-BF16 (agent/impls/cpu_compute.md)
    cxx_flags.append('-fopenmp')
    if platform.machine() == 'aarch64':
        cxx_flags.append('-march=armv8.2-a+fp16+dotprod+sve+bf16')

    return [CUDAExtension(
        name='asym_gemm._C',
        sources=[
            os.path.join(current_dir, 'csrc/python_api.cpp'),
            os.path.join(current_dir, 'csrc/cpu_ops/cpu_ops.cpp'),
            os.path.join(current_dir, 'csrc/dropout/dropout_mask.cu'),
            os.path.join(current_dir, 'csrc/ep_steal/ep_steal_sync.cu'),
            os.path.join(current_dir, 'csrc/exp_act_offload/exp_act_offload_kernels.cu'),
            os.path.join(current_dir, 'csrc/qwen3/qwen3_gate_up_windowed_bwd.cu'),
        ],
        extra_link_args=['-fopenmp'],
        include_dirs=[
            f'{CUDA_HOME}/include',
            f'{CUDA_HOME}/include/cccl',
            os.path.join(current_dir, 'asym_gemm/include'),
            os.path.join(current_dir, 'third-party/cutlass/include'),
            os.path.join(current_dir, 'third-party/fmt/include'),
        ],
        libraries=['cudart', 'nvrtc'],
        library_dirs=[f'{CUDA_HOME}/lib64'],
        extra_compile_args={
            'cxx': cxx_flags,
            'nvcc': [
                '-std=c++17', '-O3', '-Xcompiler', '-fPIC',
                f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}',
                '-DDG_JIT_USE_RUNTIME_API',
            ],
        },
    )]


class CustomBuildPy(build_py):
    """Custom build step that bundles third-party includes and generates stubs."""

    def run(self):
        self.prepare_includes()
        self.generate_pyi_file()
        build_py.run(self)

    def generate_pyi_file(self):
        from scripts.generate_pyi import generate_pyi_file
        generate_pyi_file(name='_C', root='./csrc', output_dir='./stubs')
        pyi_source = os.path.join(current_dir, 'stubs', '_C.pyi')
        pyi_target = os.path.join(self.build_lib, 'asym_gemm', '_C.pyi')

        if os.path.exists(pyi_source):
            os.makedirs(os.path.dirname(pyi_target), exist_ok=True)
            shutil.copy2(pyi_source, pyi_target)

    def prepare_includes(self):
        # Copy into both the build directory and the source tree.
        # The source-tree copy is needed for editable installs (`pip install -e .`),
        # where the JIT compiler resolves `library_include_path` to the source
        # `asym_gemm/include/` rather than the build output.
        source_include_dir = os.path.join(current_dir, 'asym_gemm/include')
        build_include_dir = os.path.join(self.build_lib, 'asym_gemm/include')

        for target_dir in [source_include_dir, build_include_dir]:
            os.makedirs(target_dir, exist_ok=True)
            for d in third_party_include_dirs:
                dirname = d.split('/')[-1]
                src_dir = os.path.join(current_dir, d)
                dst_dir = os.path.join(target_dir, dirname)
                if os.path.exists(dst_dir):
                    shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)

        # build_lib is the staging area for what eventually goes to dist-packages
        build_asym_gemm_inc_dir = os.path.join(self.build_lib, 'asym_gemm/include/asym_gemm')

        # Source directory in your workspace
        source_asym_gemm_inc_dir = os.path.join(current_dir, 'asym_gemm/include/asym_gemm')

        # Ensure the destination path in the build directory exists
        os.makedirs(build_asym_gemm_inc_dir, exist_ok=True)

        # Copy the contents of your project's include folder
        if os.path.exists(source_asym_gemm_inc_dir):
            # Using dirs_exist_ok=True (Python 3.8+) to merge contents if needed
            shutil.copytree(source_asym_gemm_inc_dir, build_asym_gemm_inc_dir, dirs_exist_ok=True)
            print(f"Bundled project headers: {source_asym_gemm_inc_dir} -> {build_asym_gemm_inc_dir}")

if __name__ == '__main__':
    setuptools.setup(
        name='asym_gemm',
        version='0.1.0',
        packages=find_packages(),
        ext_modules=get_ext_modules(),
        package_data={'asym_gemm': ['include/**/*']},
        cmdclass={
            'build_py': CustomBuildPy,
            **({'build_ext': BuildExtension} if BuildExtension is not None else {}),
        },
    )
