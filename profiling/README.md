# Profiling Results Layout

Each workload has its own directory.  The top-level `table.md` and
`profile.json` are the Nsight/profile-mode GPU timeline results and are the
only truth profiling tables.  Source-label coverage artifacts are debug-only
and are kept as `debug.md` / `debug.json`.

| Directory | Workload | Contents |
|---|---|---|
| `matrix_1b/` | Fundamental single 1.073B-parameter frozen matrix | Nsight GPU timeline/kernel/memcpy/no-kernel table |
| `mlp_1b/` | Fundamental 1.073B-parameter two-layer MLP | Nsight GPU timeline/kernel/memcpy/no-kernel table |
| `mlp/` | M4.1 MLP toy | Nsight GPU timeline/kernel/memcpy/no-kernel table |
| `dense_llm/` | M4.2 dense LLM toy | Nsight GPU timeline/kernel/memcpy/no-kernel table |
| `moe_contiguous/` | M4.3 MoE toy, contiguous routing | Nsight GPU timeline/kernel/memcpy/no-kernel table |
| `moe_masked/` | M4.3 MoE toy, masked routing | Nsight GPU timeline/kernel/memcpy/no-kernel table |
| `qwen3_14b/` | Qwen3-14B config-matched profile | Nsight GPU timeline/kernel/memcpy/no-kernel table |
| `qwen3_30b_a3b/` | Qwen3-30B-A3B config-matched profile | Nsight GPU timeline/kernel/memcpy/no-kernel table |
| `*/debug.md`, `*/debug.json` | Matching source-label report | Python/source range coverage only; not performance truth |
| `*/nsight/` | Matching Nsight report | Same data as the workload top-level table |

Use `--timing-mode profile` plus Nsight Systems for real GPU bubble analysis.
Use `--timing-mode debug_sync` only for source label coverage debugging.

The fundamental selectors are:

```bash
python scripts/profile_m4_steps.py --workload matrix_1b --timing-mode profile
python scripts/profile_m4_steps.py --workload mlp_1b --timing-mode profile
```
