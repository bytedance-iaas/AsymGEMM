# Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
# Modified by Bytedance Inc., 2026.
# Original: https://github.com/deepseek-ai/DeepGEMM

import os
import shutil
import subprocess
import sys
import platform
import setuptools
import torch
from setuptools import find_packages, Extension
from setuptools.command.build_py import build_py
from setuptools.command.build_ext import build_ext
from pathlib import Path

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME, include_paths as _torch_include_paths
except ImportError:
    BuildExtension = None
    CUDAExtension = None
    CUDA_HOME = None
    _torch_include_paths = lambda: []

# Project root
current_dir = os.path.dirname(os.path.realpath(__file__))

# Third-party include directories to bundle into the wheel
third_party_include_dirs = [
    'third-party/cutlass/include/cute',
    'third-party/cutlass/include/cutlass',
]


# --------------------------------------------------------------------------- #
# CPU extension (asym_gemm._cpu_C): first-party cpu_gemm (AMX INT8) + pybind11
# wrapper. Built via a CMake sub-step that produces libcpu_gemm.a, which the
# Extension then links statically. Independent of the CUDA build: hosts
# without CUDA still produce a working _cpu_C and unified_moe sub-package.
# --------------------------------------------------------------------------- #

CPU_GEMM_SRC = os.path.join(current_dir, 'csrc', 'cpu', 'cpu_gemm')
CPU_GEMM_BUILD = os.path.join(current_dir, 'build', 'cpu_gemm')
CPU_GEMM_LIB = os.path.join(CPU_GEMM_BUILD, 'libcpu_gemm.a')


def _pybind11_include():
    """Pick a pybind11 include path that works in this environment.

    Preference order:
      1. installed `pybind11` Python package (authoritative)
      2. torch's bundled pybind11 headers (always present when torch is)
      3. system /usr/include (e.g. apt's pybind11-dev)
    """
    try:
        import pybind11
        return pybind11.get_include()
    except ImportError:
        pass
    # torch ships pybind11 headers under <torch_inc>/pybind11/
    for p in _torch_include_paths():
        if os.path.isdir(os.path.join(p, 'pybind11')):
            return p
    if os.path.isdir('/usr/include/pybind11'):
        return '/usr/include'
    raise RuntimeError(
        "Could not find pybind11 headers. Install with `pip install pybind11` "
        "or `apt install pybind11-dev`."
    )


def _sources_mtime(root):
    """Newest mtime under `root`, recursive. Used to skip the CMake build
    when nothing has changed."""
    newest = 0.0
    for dirpath, _, files in os.walk(root):
        # Don't traverse into build dirs the user may have created.
        if os.path.basename(dirpath).startswith('build'):
            continue
        for f in files:
            try:
                m = os.path.getmtime(os.path.join(dirpath, f))
                if m > newest:
                    newest = m
            except OSError:
                pass
    return newest


def build_cpu_gemm_lib():
    """Build csrc/cpu/cpu_gemm via its own CMakeLists, producing
    libcpu_gemm.a at a deterministic path. Idempotent — skips when the .a
    is newer than the newest cpu_gemm source file."""
    if platform.machine() not in ('x86_64', 'AMD64', 'x64'):
        print(f"[cpu_gemm] non-x86 host ({platform.machine()}) — skipping AMX/AVX CPU extension.")
        return
    if not os.path.isdir(CPU_GEMM_SRC):
        raise RuntimeError(f"cpu_gemm sources not found at {CPU_GEMM_SRC}.")

    if os.path.isfile(CPU_GEMM_LIB):
        lib_mtime = os.path.getmtime(CPU_GEMM_LIB)
        if lib_mtime >= _sources_mtime(CPU_GEMM_SRC):
            print(f"[cpu_gemm] libcpu_gemm.a is up to date — skipping CMake build.")
            return

    print(f"[cpu_gemm] configuring → {CPU_GEMM_BUILD}")
    os.makedirs(CPU_GEMM_BUILD, exist_ok=True)
    configure = [
        'cmake', '-S', CPU_GEMM_SRC, '-B', CPU_GEMM_BUILD,
        '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
    ]
    try:
        subprocess.check_call(configure)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(
            "CMake invocation for cpu_gemm failed. Ensure `cmake` (>= 3.18) "
            f"is installed and on PATH. Underlying error: {e}"
        )

    print(f"[cpu_gemm] building libcpu_gemm.a")
    subprocess.check_call([
        'cmake', '--build', CPU_GEMM_BUILD, '--target', 'cpu_gemm',
        '-j', str(max(1, os.cpu_count() or 1)),
    ])
    if not os.path.isfile(CPU_GEMM_LIB):
        raise RuntimeError(
            f"CMake reported success but {CPU_GEMM_LIB} was not produced."
        )


def get_cpu_extension():
    """The asym_gemm._cpu_C pybind11 extension. Plain setuptools.Extension —
    no torch dependency, just links the pre-built libcpu_gemm.a."""
    cxx_flags = [
        '-std=c++17', '-O3', '-fPIC',
        '-Wno-psabi', '-Wno-deprecated-declarations',
    ]
    # Only set the CXX11 ABI flag if torch is available (matches the CUDA
    # extension's choice). When building on a CUDA-less host with torch
    # absent, we just omit the flag.
    try:
        cxx_flags.append(
            f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}'
        )
    except Exception:
        pass

    return Extension(
        name='asym_gemm._cpu_C',
        sources=['csrc/cpu/module.cpp'],
        include_dirs=[
            _pybind11_include(),
            os.path.join(CPU_GEMM_SRC, 'include'),
            # Internal headers: the MoE batch entry drives the worker pool
            # directly (runtime/runtime_impl.h, runtime/worker_pool.h).
            os.path.join(CPU_GEMM_SRC, 'src'),
        ],
        extra_objects=[CPU_GEMM_LIB],
        extra_compile_args=cxx_flags,
        language='c++',
    )


def get_ext_modules():
    cxx_flags = [
        '-std=c++17', '-O3', '-fPIC',
        '-Wno-psabi', '-Wno-deprecated-declarations',
        f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}',
        # gb200_tp.md I1 FIX A: context-INDEPENDENT runtime-API kernel loading
        # (cudaLibraryLoadFromFile -> cudaKernel_t, materialized per-device on demand), so one
        # process can launch the same JIT'd GEMM on BOTH GPUs. Requires CUDART >= 12.8.
        '-DDG_JIT_USE_RUNTIME_API',
    ]

    exts = []

    # CUDA extension. Skipped if no CUDA toolkit is found, so the CPU-only path
    # still produces an installable package. Includes the first-party SFT kernels
    # (dropout / ep-steal / exp-act-offload / qwen3) compiled at build time.
    if CUDA_HOME is not None and CUDAExtension is not None:
        exts.append(CUDAExtension(
            name='asym_gemm._C',
            sources=[
                'csrc/python_api.cpp',
                'csrc/dropout/dropout_mask.cu',
                'csrc/ep_steal/ep_steal_sync.cu',
                'csrc/exp_act_offload/exp_act_offload_kernels.cu',
                'csrc/qwen3/qwen3_gate_up_windowed_bwd.cu',
            ],
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
        ))

    # CPU extension (AMX INT8): x86-only. Skipped on non-x86 hosts (e.g. ARM /
    # Grace-Blackwell) where the AVX/AMX intrinsics don't compile; the CUDA +
    # SFT path is unaffected.
    if platform.machine() in ('x86_64', 'AMD64', 'x64'):
        exts.append(get_cpu_extension())

    return exts


class CustomBuildPy(build_py):
    """Custom build step that bundles third-party includes and generates stubs."""

    def run(self):
        self.prepare_includes()
        build_cpu_gemm_lib()        # produce libcpu_gemm.a before extension link
        self.generate_pyi_file()
        build_py.run(self)

    def generate_pyi_file(self):
        from scripts.generate_pyi import generate_pyi_file
        # csrc/cpu holds the separate _cpu_C module — keep its m.def()s out
        # of the _C stub.
        generate_pyi_file(name='_C', root='./csrc', output_dir='./stubs',
                          exclude_parts=('cpu',))
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

class CustomBuildExt(BuildExtension):
    """Ensure libcpu_gemm.a exists before the _cpu_C extension is linked.

    `build_py` also calls build_cpu_gemm_lib(), but editable installs may
    invoke `build_ext` without first running our `build_py`. Calling here
    is idempotent — the function skips when the .a is already up-to-date.
    """

    def run(self):
        build_cpu_gemm_lib()
        super().run()


if __name__ == '__main__':
    setuptools.setup(
        name='asym_gemm',
        version='0.2.0',
        packages=find_packages(),
        ext_modules=get_ext_modules(),
        package_data={'asym_gemm': ['include/**/*']},
        cmdclass={'build_py': CustomBuildPy, 'build_ext': CustomBuildExt},
    )
