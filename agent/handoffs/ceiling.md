# Handoff: CPU-bound ceiling reconfirmation (asym_cpuadamwds)

You are (should be) picking this up **inside the training enroot container** (e.g. `asym42_enroot_run`).
A previous session ran on the **host by mistake** (no `numactl` there), which let the cpuadam offload
spill onto GPU-HBM NUMA nodes and inflated the memory numbers. Your job: redo the measurement
**correctly, NUMA-bound to the Grace CPU nodes.**

## STEP 0 — GATE (do this first; do NOT skip)
Confirm you are actually in the container with numactl available:

```bash
ls -d /workspace                 # must exist (container mount)
command -v numactl               # must resolve, e.g. /usr/bin/numactl  <-- the path the user expects
numactl --hardware | grep -E 'node (0|1) cpus'   # node0=cpus 0-71, node1=cpus 72-143 (Grace)
```

- If `/workspace` is missing OR `numactl` is not found → **STOP** and tell the user. You are on the
  host / wrong container. Running there spills to GPU-HBM and produces wrong numbers (that's the exact
  bug that made this handoff necessary).
- If both are present → continue.

## Why this matters (GB200 NUMA)
GPU HBM is coherent and shows up as NUMA nodes. `run_lf_lora_sft.sh` binds host offload with
`numactl --membind=0,1 --cpunodebind=0,1` (its default `NUMACTL_ENABLE=1`). CPU nodes:
- **node0** (cpus 0-71, ~478 GiB) + **node1** (cpus 72-143, ~479 GiB) → **~957 GiB total CPU RAM**.
- GPU-HBM nodes = **node2, node10, node18, node26** (184 GiB each). Host allocations must NOT land here.

Verify binding holds during a run — GPU nodes must stay ~empty:
```bash
for n in 2 10 18 26; do awk -v n=$n '/MemUsed/{print "node"n": "$4/1048576" GiB used"}' \
  /sys/devices/system/node/node$n/meminfo; done
```
If a GPU-HBM node grows during training, binding is NOT working — stop and debug; do not trust numbers.

## TASK — reconfirm two asym ceilings, bound
Both rows are already active (uncommented) in `scripts/lf/ceiling_search_both.sh` CONFIGS:
1. `q3-32b     | asym_cpuadamwds | recomp-off-full-fg-ker000-ceil0000` (seq0 50000)
2. `q3-30b-a3b | asym_cpuadamwds | recomp-off-full-fg-ker101-ceil0000` (seq0 128000)

Run the **search** (not `--single`) so it finds the *true bound* ceiling (it changes vs the old unbound run):
```bash
cd <repo>/third_party/AsymGEMM
STATE_DIR=scripts/lf/ceiling_bound_state \
CONFIRM_STEPS=4 WARMUP_STEPS=1 PROBE_STEPS=2 PROFILERS=source \
CEIL_GPU_POOL=<a clean GPU idx> \
bash scripts/lf/ceiling_search_both.sh
```
`NUMACTL_ENABLE` defaults to 1 and the container's numactl binds to 0,1 automatically — **do NOT set
`NUMACTL_ENABLE=0`**. Set `HF_HOME` / `ENV_DIR` / `CUDA_HOME` as the container needs (see Env below).

### Expected under binding (sanity check the results against these)
- **q3-32b 68k/ohbm8** was C-1000 GiB **unbound**; 1000 > 957 GiB CPU cap → **bound it WILL C-OOM below
  68k**. The real ceiling is lower — let the search find it.
- **q3-30b-a3b 172k/ohbm0** was C-899 GiB (< 957) → **likely still ~172k** bound; its boundary is G-OOM
  ~188k (HBM-bound), so binding may not move it much.

## Env (adapt to the container's paths)
- **HF_HOME**: cache holding `Qwen3-32B`, `Qwen3-30B-A3B`, `Llama-3.3-70B`.
- **ENV_DIR**: venv with torch 2.12+cu130, deepspeed, datasets, **liger-kernel**, `asym_gemm` importable.
  `flash_attn` absent → SDPA, which is correct for these base models.
- **CUDA_HOME**: cu13 toolkit. **ASYM_NVME_PATH**: any writable dir (only used for `-ceil>0`; these are ceil0000).
- Each step is ~20 min at these seqs; a full search per config is a few hours. Run **one config at a time**
  (each needs ~900 GiB host RAM; two won't co-fit in 957 GiB).

## Steady-state rule (agent/RULES.md — authoritative)
`lat` = run **4 measured (non-warmup) steps, drop the 1st and last, average the middle 2** (warmup always
excluded). `C` = peak host RSS (GiB, `/proc/self/status` VmHWM). `G` = peak reserved HBM (GiB). Extract from
the run's `summary.md` (`Whole-process peak reserved HBM`, `RSS peak MiB`) + `step_samples.csv`.

## When done
Write bound results into `scripts/lf/profile_lora_lf_test_both.sh` RUNS comments, format
`<lat>s, C-<ram>, G-<hbm>, <boundary> [DONE|IP]`, and note they **supersede** these unbound host numbers
(wrong due to GPU-HBM spillover):
- q3-32b asym (unbound/host): `1109s, C-1000, G-183`  ← redo bound (ceiling drops)
- q3-30b-a3b asym (unbound/host): `1110s, C-899, G-183` ← re-verify bound

## Reference / artifacts
- Prior unbound (host) artifacts: `profiling_results/profiling_both_ceiling/asym_long_sft_smoke__lora__lf__bf16/qwen3-{32b,30b-a3b}__gpus1__b8_s{68000,172000}_ga1_*`.
- Recovered original in-container ceiling records: `scripts/lf/ceiling_search_state/{ledger,results}.jsonl` + `driver.log`.
- Host-only numactl workaround (ONLY if forced to run on host; prefer the container's system numactl):
  `scripts/lf/setup_numaenv.sh` provisions gitignored `${ROOT}/.numaenv/{bin,lib}`.
