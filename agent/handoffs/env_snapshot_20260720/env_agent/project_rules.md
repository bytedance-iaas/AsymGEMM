# PROJECT RULES — read before anything else. Corrections to common agent misconceptions.

These override any assumption you brought with you. Violating them produces wrong
numbers, wrong claims, or broken runs. (Written 2026-07-16 after repeated hallucinations.)

## 1. CPU memory is ~950 GB, NOT >1 TB
- Only NUMA nodes 0 and 1 are CPU (Grace LPDDR) memory: **490 GB + 490 GB ≈ 957 GB total**
  (verify: `numactl --hardware` → node 0/1 sizes).
- `free` reports ~1.69 TB — **that number is a trap**: the extra ~740 GB is GPU HBM
  exposed through the coherent fabric as extra NUMA nodes, NOT host RAM you can spend.
- Consequence: host-memory budgets, watchdog floors, and "C-" numbers are against the
  ~957 GB CPU pool (minus OS/driver overhead). A run at C-898 GiB is near the REAL
  ceiling even though `free` suggests headroom.

## 2. You are ALWAYS inside a container — never assume host access
- All work runs inside an enroot container (this project: image mounted under
  `/scratch_local/.../enroot/data/`, e.g. `asym_sft_46`). The host does NOT have
  `/workspace`: **if `/workspace` exists, you are in the container** — do not reason as
  if you can touch host services, host packages, or host paths.
- Do not try to run training/profiling "on the host directly"; every command belongs in
  the container environment you are already in (with its `.venv`).

## 3. NEVER run experiments/configs in parallel
- One profiling/training run per node at a time, ~30 s settle between runs. Parallel
  runs contend on GPU, LPDDR bandwidth, C2C, and host memory → contaminated numbers.
- This includes "just a quick microbenchmark": CPU microbenchmarks only in clean windows
  (GPU idle, no trainer alive). Numbers taken during a concurrent run are invalid
  (measured: 0.16 TF/s vs 1.78 TF/s clean for the same kernel).
- Ceiling searches: never run two concurrently; never interrupt one (re-run to resume).

## 4. The project goal
- **Develop memory-efficient LoRA SFT systems for LARGE models on 1–2 GPUs** (GB200
  class): long context and big models within tight HBM + ~957 GB host budgets, lossless
  training, with throughput as good as possible under those constraints.
- Corollaries: host RAM is a first-class constraint (the 30B/32B ceilings are host-RAM-
  bound, not HBM-bound); "spend host memory to save time" is NOT automatically a win;
  every feature must state its C (host) and G (HBM) cost next to its speedup.

## Standing measurement law (summary — details in agent/RULES.md + cpu_compute.md §0)
- **Fit/no-fit (capacity) probes: 1 warmup + 1 measured step is ENOUGH** (`WARMUP_STEPS=1
  MAX_STEPS=1`) — the verdict is OOM-or-not, not a timing (Kevin 2026-07-19).
- Latency: `PROFILERS=source`, **1 warmup + 2 measured (`WARMUP_STEPS=1 MAX_STEPS=2`)**,
  steady = mean of the 2 measured. (Rev 2026-07-19, user: measured-step variance is ~1%,
  so 2 measured steps suffice — saves 25-30% GPU time per run. Supersedes the old
  1w+4m protocol everywhere, including throughput_prompt.md and older docs.)
- Same-day A/B only (day drift measured ±1 s @32k, ±10 s @128k).
- Verify flags reached the trainer (`/proc/<gpu-pid>/environ` + ENGAGED markers).
- Artifact leaves get overwritten by same-config runs — snapshot numbers immediately.
- Process truth = `nvidia-smi --query-compute-apps`, not `pgrep -f`.
- **NEVER use `pkill`/`pkill -f` to stop runs (rule added 2026-07-19 after it orphaned a
  setsid trainer that poisoned 3 runs with a 120-GiB squatter). ONLY `kill -9 <PID>` with
  explicit PIDs from (a) `nvidia-smi --query-compute-apps=pid` AND (b) the run's process
  tree (`ps --ppid` walk / torchrun + .venv python children — a trainer in its CPU load
  phase is NOT on the GPU list yet). After killing: re-check BOTH sources until empty.
  Before launching: GPU compute list must be empty (pre-flight guard), else abort.**
