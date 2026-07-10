# AsymGEMM Documentation

## Contents

| Document | Description |
|----------|-------------|
| [Quick Start](quick_start.md) | Installation, SGLang integration, standalone usage |
| [Design Overview](design_overview.md) | Motivation, design decisions, data flow |
| [API Reference](API.md) | Python API, input formats, environment variables |
| [Release Notes](release_notes/) | [v0.2.0](release_notes/v0.2.0.md) — SM90, INT8, unified CPU + GPU MoE; [v0.1.0](release_notes/v0.1.0.md) — initial release |
| [Adaptive Dispatch](../adaptive_dispatch.md) | Cost-model-based CPU/GPU expert partitioning for the unified MoE runtime |
| [cpu_gemm](../csrc/cpu/cpu_gemm/README.md) | Bundled CPU GEMM library (AMX / AVX-512-VNNI / AVX2) used by the unified runtime |
