import os
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
    ]

    nvcc_flags = [
        '-std=c++17', '-O3',
        '-gencode', 'arch=compute_100a,code=sm_100a',  # Blackwell UMMA (GB200)
        '--expt-relaxed-constexpr',
        '-Xcompiler', '-fPIC',
        '-diag-suppress=550',     # used-but-not-set variable warnings
        '-diag-suppress=20012',   # __host__ on defaulted ctor
    ]

    return [CUDAExtension(
        name='asym_gemm._C',
        sources=['csrc/python_api.cpp', 'csrc/mega_moe_launch.cu',
                 'csrc/mega_moe_new_launch.cpp', 'csrc/mega_moe_new_kernel.cu'],
        include_dirs=[
            f'{CUDA_HOME}/include',
            f'{CUDA_HOME}/include/cccl',
            os.path.join(current_dir, 'asym_gemm/include'),
            os.path.join(current_dir, 'third-party/cutlass/include'),
            os.path.join(current_dir, 'third-party/fmt/include'),
        ],
        libraries=['cudart', 'nvrtc'],
        library_dirs=[f'{CUDA_HOME}/lib64'],
        extra_compile_args={'cxx': cxx_flags, 'nvcc': nvcc_flags},
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


if __name__ == '__main__':
    cmdclass = {'build_py': CustomBuildPy}
    if BuildExtension is not None:
        # Needed so setuptools invokes nvcc for .cu sources in CUDAExtension.
        cmdclass['build_ext'] = BuildExtension
    setuptools.setup(
        ext_modules=get_ext_modules(),
        cmdclass=cmdclass,
    )
