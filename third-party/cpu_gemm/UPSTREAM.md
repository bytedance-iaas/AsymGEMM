# cpu_gemm — vendored snapshot

This directory is a snapshot of the cpu_gemm standalone library, vendored
into AsymGEMM so that one `pip install` builds both the GPU and the CPU
sides of the unified MoE kernel.

| Field      | Value                                                             |
|------------|--------------------------------------------------------------------|
| Source     | `/workspace/cpu_gemm/` (local working tree, snapshot)              |
| Snapshot   | 2026-06-04                                                         |
| License    | See `LICENSE` in this directory.                                   |
| Re-sync    | `bash ../../scripts/sync_cpu_gemm.sh [PATH_TO_UPSTREAM]`           |

Excluded from the snapshot: `build/`, `.git/`. The `bench/` directory
is kept because the upstream CMakeLists references it unconditionally.

The vendored tree's CMakeLists is consumed unmodified by AsymGEMM's
`setup.py` during `pip install`, which shells out to CMake to produce
`libcpu_gemm.a` before linking the `asym_gemm._cpu_C` pybind11 extension.
