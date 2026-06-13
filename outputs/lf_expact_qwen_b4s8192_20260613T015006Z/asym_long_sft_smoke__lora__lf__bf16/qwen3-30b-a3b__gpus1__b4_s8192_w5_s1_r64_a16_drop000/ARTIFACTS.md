# LF Profiling Artifacts

This config root is organized as follows:

- `combined/`: config-level LF timing and allocator-summary plots from `profile.json`.
- `memory_combined/`: config-level source-memory breakdown plots plus per-group subfolders split by workload/backend/profiler/router/recompute/policy. If no source-memory rows were collected, this folder contains a README explaining why.
- `c2c_combined/`: config-level C2C/CTC saturation plots plus per-group subfolders split by workload/backend/profiler/router/recompute/policy. If old traces lack GPU metrics, this folder contains a README explaining why.
- `<backend>__<profiler>__<recompute>__pol<policy>__router<mode>/b<batch>_s<seq>/`: per-run artifacts.

If `PLOT_OUTPUT_DIR` is set, combined plot folders are written under that external plot output root instead of this config root.

Per-run nsys folders contain `profile.json`, markdown summaries, `plots/` for per-run LF plots, and `interconnect_ctc_*.png/csv` when C2C GPU metrics are available.
