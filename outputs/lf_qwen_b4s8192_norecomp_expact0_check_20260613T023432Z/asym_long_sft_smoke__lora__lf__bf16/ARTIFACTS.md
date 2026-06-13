# LF Profiling Precision Root

This precision root is organized as follows:

- `combined/`: global LF timing and allocator-summary plots across config roots.
- `memory_combined/`: global source-memory breakdown plots across config roots plus per-group subfolders split by workload/backend/profiler/router/recompute/policy. If no source-memory rows were collected, this folder contains a README explaining why.
- `c2c_combined/`: global C2C/CTC saturation plots across config roots plus per-group subfolders split by workload/backend/profiler/router/recompute/policy. If old traces lack Nsight GPU metrics, this folder contains a README explaining why.
- `<config_root>/`: one workload/configuration root. Each config root has its own `combined/`, `memory_combined/`, `c2c_combined/`, and per-run backend/profiler folders.

If `PLOT_OUTPUT_DIR` is set, global combined plot folders are written under that external plot output root instead of this precision root.

Fresh nsys runs collect C2C GPU metrics at 100 Hz. Existing traces created without Nsight `GPU_METRICS` tables cannot be converted into C2C saturation plots.
