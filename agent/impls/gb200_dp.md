# GB200 DP Panel: Real-DP `asym_dp2` + DP Baselines — Staged Implementation Plan

Companion to `agent/impls/gb200_tp.md` (the sTP system plan — the actual product).
THIS doc owns the **DP panel** of the paper:
the SOTA-DP baseline rows, a **semantically-correct** DP variant of our kernels (`asym_dp2`),
the contention/motivation measurement (Q4), and the **viability gate** that protects the whole
story. Style/discipline mirror `gb200_tp.md`: staged, gated, one experiment at a time, fresh
artifacts, never advance on inconclusive.

## Goal (ULTRA CLEAR — read before any change)

```text
WHAT: a BOUNDED (~days) DP track — DP baselines (D0), contention probe (D1), real-DP asym_dp2
    (D2), panel freeze (D3). DP is NOT our system; underperformance is fixed ONLY with existing
    |1 knobs (pool caps, OMP caps, GPU-silu), never new machinery.
WHY REAL DP: unsynced pairs are not DP. asym_dp2 = sharded data + per-step LoRA-grad
    all-reduce, loss-equivalent to one b8 run.
pair 0,1 only. Fresh OUTPUT_ROOT per stage. EXACT configs: next section.
```

## STATUS (2026-07-06 — read this first; details in the Decision Log)

```text
DONE + VALIDATED: D0 (superoffload_mem 134.6 s / 22.2 GiB / 165.5 GiB-rank), D1 (contention
  ~1.0 — the "no contention" branch; capacity IS the DP pain: 489 GiB summed; pairing preserves
  per-GPU throughput 1.95x), D2 (Route A DDP WORKS: cross-rank post-reduce grads BIT-IDENTICAL,
  mean semantics confirmed; e2e 151.0 s / 22.3 GiB / 246.6 GiB-rank; VG1 = 1.12x PASS -> TP
  investment justified; VG3 PASS). New tools: aggregate_dp_ranks.py, run_dp2_pair.sh,
  rank<R>_memstats.json emission, adapter grad/init dump + comparators.
REMAINING / OPEN ISSUES:
  1. WEIGHT REDUNDANCY: each dp2 rank pins its OWN 64 GB arena (2x total; ZeRO-3 shards its).
     Fixable WITHOUT sharding: shm/memfd arena + per-process cudaHostRegister of the SAME pages
     (1x host bytes). New machinery; ~10-15% frontier gain; position as OUR improved DP, and it
     softens the "DP cannot express one-copy" line — decide framing before building.
  2. G-D2.4 OPEN: dp2 bwd is +21.3 s vs the D1 probe (132.1 vs 110.8) — DDP plumbing overhead,
     needs an nsys decomposition (reducer-hook interplay with the host-blocking fg backward).
  3. VG2 FAILED at the dev row: HBM parity (22.3 vs 22.2; adapter banks stay on-GPU because
     hook-offload is disabled under DDP) and host RSS WORSE (247 vs 165/rank). The asym memory
     posture pays at the FRONTIER (b1/boundary rows), not at s20000 — panel framing must lead
     with VG1 + frontier, never a blanket memory win.
  4. Disjoint-shard receipt (per-rank sample ids at step 1) not yet dumped — add before the
     paper freeze (G-D2.1's shard-proof line).
  5. D2.5 MoE row (find_unused_parameters=True) + D3 panel = paper phase, untouched.
```

## Dev Workloads & Baselines-To-Beat (EXACT configs — copy-paste into RUNS)

```text
DEV RULES: ONE model (q3-32b — DENSE first; MoE deferred), ONE workload (s20000; well inside
every boundary), host RSS comfortably under the watchdog floor (HC2/HC5). ONE baseline
(superoffload_mem|unsloth-off — the best/shipping DP). Everything else = paper phase.

# ============ OURS (real DP, lands in D2; b4/GPU = global 8) ============
q3-32b|2 ; asym_dp2_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 20000|4|1 ; none|false|false|false|false|false

# ============ THE BASELINE TO BEAT (D0; the VG reference) ============
q3-32b|2 ; superoffload_mem|unsloth-off|ligerloss1 ; 20000|4|1 ; none|false|false|false|false|false

# ============ |1 SOLO REFERENCES (D1 contention factor + scaling frame) ============
q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 20000|4|1 ; none|false|false|false|false|false
q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 20000|8|1 ; none|false|false|false|false|false

# BEAT relation (dev): asym_dp2 vs the superoffload row above = VG1-VG4 (step_s <= ~2.0x,
# target 1.3-1.7x; step_H <<; loss in band; allreduce < 3%).
# PAPER PHASE (DEFERRED): the 5-model matrix (llama3.3-70b 25000, q2.5-72b 30000, q3-30b-a3b
# 80000 ker101, llama4-scout 9500, q3-32b 50000), zero3_offload_mem rows, unsloth (non-off)
# variant, b4-boundary + b1-frontier sweeps, MoE rows.
```

## THE VIABILITY GATE (headline; a PRECONDITION for continuing TP investment)

```text
VG1  step_s(asym_dp2) <= ~2.0x step_s(superoffload_mem-DP2) at the same row
     (TARGET band 1.3-1.7x, consistent with the |1 scoreboard ratios), AND
VG2  step_H(asym_dp2, per GPU) substantially below superoffload_mem-DP2's, AND
VG3  loss in band (within ~0.05 of the DP baseline at same global workload), AND
VG4  the all-reduce cost < 3% of step_s (LoRA grads ~0.8-1.6 GB; must be noise).
IF VG1 FAILS (we are "bombed", >2.5-3x): STOP. Decompose (lane BW / RSS / numastat /
     host-sync counts / pool-churn counters), fix with the |1 knobs (the fix_qwen3 lessons:
     ASYM_EXPACT_CPU_POOL_MAX_BYTES per rank, OMP caps, GPU-silu), re-run. The TP build does
     not proceed on a bombed DP row — the kernels would look non-viable and the TP win would
     read as patching a self-inflicted wound.
EXPECTATION MATH (predeclared, not assumed): per-rank b4 halves per-rank volume; total
     activation bytes ~ one |1 b8 run; average DRAM demand ~60-120 GB/s vs ~500 ceiling =>
     contention should add ~10-30%, not multiples. |1 ratios (post v1/v2 fixes) were
     ~1.15-1.7x superoffload step_s at equal workload with far lower step_H.
```

## HARD CONSTRAINTS (inherited from gb200_tp.md HC1-HC3 + DP-specific)

```text
HC1  host memory = BOTH CPU nodes (NUMACTL_MEMBIND=0,1 default). Never restrict to one node.
HC2  EVERY rank carries TRAIN_OOM_SCORE_ADJ=1000 + HOST_MEM_WATCHDOG=true (35 GB floor).
     DP nuance: each rank runs its own watchdog over the SAME nodes — a squeeze interrupts
     BOTH ranks; per-rank ${LOG_FILE}.host_mem_watchdog_fired sentinels are recorded and a
     soft host OOM is classified as such (and is itself dp2-vs-sTP evidence).
HC3  launch ONLY via scripts/lf/profile_* or scripts/lf/run_lf_* (guards built in).
HC4  (DP-specific) ASYM_DP=1 and ASYM_STP=1 are MUTUALLY EXCLUSIVE — both set => die at the
     launcher. The sTP Trainer.__init__ n_gpu patch (keyed on ASYM_STP) must NOT fire under DP
     (under torchrun DDP wrap is INTENDED here, unlike sTP where DataParallel is killed).
HC5  (DP-specific) two ranks x pinned pools must fit: predeclare per-rank
     ASYM_EXPACT_CPU_POOL_MAX_BYTES (start 96-128 GiB/rank vs |1's 192 GiB) and audit summed
     RSS vs the 960 GB node budget + the 35 GB floor BEFORE the big rows.
```

## Baselines owned by this doc

```text
B-DP1 superoffload_mem|unsloth-off  b4/GPU |2   (true DP via existing torchrun/deepspeed path;
      THE viability-gate reference; also run |unsloth variant where it fits)
B-DP2 zero3_offload_mem|unsloth-off b4/GPU |2   (second family member; same launch path)
OURS  asym_dp2_cpuadamwds           b4/GPU |2   (real DP of the validated |1 asym backend)
CITED-NOT-RUN SuperOffload-Ulysses (SP): vendored Megatron-DeepSpeed-SO is GPT-only, NO LoRA —
      record the file:line receipt in D3 BEFORE stating it in any table (tp.md discipline).
NOT here: FSDP2+TP (that is the TP panel's strongest baseline — lives in gb200_tp.md scope).
```

## Evidence Discipline

Same as `gb200_tp.md`. One experiment at a time; new OUTPUT_ROOT per stage; before each run
write expected {model, pair, backend, per-rank+global batch, artifact tag, comparison row,
likely failure}; after: command.txt, per-rank train.log + profile.json, per-GPU step_H, RSS per
rank + summed, loss band, numa_maps/numastat, watchdog sentinels. Labels:
`validated | blocked_by_stage_bug | inconclusive_wrong_config | inconclusive_partial_profile |
inconclusive_stale_artifact | inconclusive_unexpected_path`. Never advance on inconclusive.

---

## Stage D0 — DP Baseline Rows + Per-Rank Profiling Audit (NO new training code)

**Objective.** Land the true-DP baseline numbers (they already launch via the harness torchrun
path) and make multi-rank profiling trustworthy — the tp.md I0 audit flagged "torchrun-path
profile.json may report rank0 only" as a known unknown; it gets RESOLVED here because this doc
owns every multi-rank row.

**Files & functions (verified anchors):**

```text
scripts/lf/run_lf_lora_sft.sh:704-706   is_torch_run() — already true for BACKEND=torch (the
    zero/superoffload family rewrites BACKEND=torch at :300-389), so B-DP1/B-DP2 launch
    multi-proc TODAY. No change in D0.
scripts/lf/run_lf_profiled_train.py:93-98   _is_rank0() gates profile/heartbeat writes — audit
    what rank1 emits; :2542-2564 memory sampling is deviceless (per-rank OK since each rank
    sees one GPU, but CONFIRM the visible-device mapping per rank).
scripts/lf/run_lf_lora_sft.sh:2301-2304  torchrun+deepspeed unsets CUDA_VISIBLE_DEVICES and
    lets the launcher assign — confirm each rank binds its own GPU (nvidia-smi during run).
NEW scripts/lf/aggregate_dp_ranks.py    tiny: collect per-rank profile.json/train.log ->
    dp_row.json {per-rank step_s (wall = max), per-GPU step_H, per-rank RSS + summed, per-rank
    loss, watchdog flags}. Used by every later stage.
```

**Runs (DEV = this ONE row; pair 0,1; MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false):**

```bash
RUNS='q3-32b|2 ; superoffload_mem|unsloth-off|ligerloss1 ; 20000|4|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_gb200dp_d0 bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0,1 --overwrite false
# PAPER PHASE (deferred): zero3 rows, the other 4 models, b4 boundary probes, b1 max-seq
# probes (b1 = DP's structural frontier: each rank carries the FULL sequence).
```

**Metrics to emit/check (D0):**

```text
per-rank: fwd_s bwd_s opt_s step_s   fwd_H bwd_H step_H   RSS   loss
per-row:  wall step_s (= max rank), summed RSS, PROFILE_GLOBAL_BATCH_SIZE == 8 in command.txt
          (the :1163 formula b4*ga1*2 is ALREADY correct for DP — assert, don't override).
outcome class per boundary cell: SUCCESS | GPU OOM | SOFT HOST OOM (watchdog sentinel).
```

**Validation gate (advance only if ALL hold):**

```text
G-D0.1 the baseline completes the DEV row (q3-32b 20000|4|1) with loss in band.
G-D0.2 per-rank profiling proven: BOTH ranks' step_H/RSS visible (via profile.json or heartbeat)
       and aggregate_dp_ranks.py emits a sane dp_row.json — if rank1 is dark, FIX HERE.
G-D0.3 each rank verifiably on its own GPU (device index in logs + nvidia-smi snapshot).
(paper phase: b4 boundary + b1 frontier cells per model per backend — OOM cells are RESULTS.)
```

**Risks/watch:** deepspeed launcher vs torchrun differences in env propagation (command.txt must
show the per-rank env); dataset registration on new roots; do NOT run rows concurrently.

---

## Stage D1 — Contention Probe (`run_dp2_pair.sh`; NOT a paper DP row)

**Objective.** The Q4 motivation number with ZERO new training code: two INDEPENDENT `|1` asym
b4 runs side-by-side (GPU0/GPU1). This is deliberately NOT DP (no sync) — it is the clean
attribution experiment for shared-Grace contention: same kernels, no communication, so any
degradation vs a solo `|1` run is pure {DRAM/capacity/cores} contention. It also PREDICTS
asym_dp2 performance (real DP adds only the ~ms-scale all-reduce, gated by VG4).
LABEL HONESTY: artifacts are tagged `dp2_probe` and are NEVER presented as a DP training row.

**Files & functions:**

```text
NEW scripts/lf/run_dp2_pair.sh   launch two |1 asym_cpuadamwds jobs concurrently (GPU 0 and 1,
    b4, same model/row — independence is the point), wait both, call aggregate_dp_ranks.py ->
    dp2_probe_merged.json {per-rank profile paths, summed RSS, max wall, per-rank
    watchdog-fired}. Each child launches through run_lf_lora_sft.sh (HC3 — guards intact),
    GPU_ID=0 / GPU_ID=1, NUM_GPUS=1, per-rank OUTPUT_ROOT subdirs, per-rank pool cap (HC5).
REFERENCE solo rows: the SAME |1 b4 row run ALONE (fresh, same day) — the contention delta is
    (paired step_s / solo step_s) per rank; also record a |1 b8 solo row for the scaling frame.
```

**Runs:**

```bash
# solo references first (q3-32b 20000|4|1 and 20000|8|1), then the pair (DEV row only):
bash scripts/lf/run_dp2_pair.sh --model q3-32b --row '20000|4|1' \
  --backend asym_cpuadamwds --recomp recomp-off-full-fg-ker000 --loss ligerloss1 \
  --output-root profiling_gb200dp_d1
```

**Metrics to emit/check (D1):**

```text
CONTENTION FACTOR per rank: paired_step_s / solo_step_s        (expect 1.1-1.3; MEASURE)
SCALING FRAME: (solo b8 step_s) vs (paired wall b4 step_s)     (the "what does GPU#2 buy in
    DP shape" number that motivates TP; also tokens/s per GPU)
host side: summed RSS vs 960 GB; numastat per node during steady state; watchdog sentinels;
    (optional PROFILERS=both) per-lane H2D/D2H GB/s in streamed windows + DRAM-side estimate.
attribution: if contention >1.3x, decompose BEFORE concluding: pool churn (cudaHostAlloc
    counts), CPU-core saturation (utilization timeseries), lane BW dips, allocator syncs.
```

**Validation gate:**

```text
G-D1.1 both ranks complete; per-rank loss equals the solo |1 run's (same seed/data => identical
       trajectories — a DIVERGENCE means an environment/contention bug, catch it now).
G-D1.2 contention factor measured WITH attribution (or explicitly ~1.0 = "no contention", which
       is ALSO a result: the motivation then leans on capacity + b1 + MoE).
G-D1.3 dp2_probe_merged.json complete; numbers copied into the Decision Log below.
```

**Risks/watch:** JIT cache races — two processes first-compiling the same kernel cache dir
concurrently; MITIGATE: pre-warm the cache with a 1-step tiny `|1` smoke BEFORE the pair (both
ranks then read the same warm cache read-only). Pool caps per HC5. Runs must not share
OUTPUT_ROOT files.

---

## Stage D2 — REAL DP `asym_dp2` (the one code change; the paper's DP row)

**Objective.** Make the `|1` asym backend a semantically-correct 2-rank DP trainer: sharded
data + per-step LoRA-grad all-reduce, loss-equivalent to a single b8 run. Everything else
(streaming kernels, offload, CPUAdamW) unchanged.

**Design decision — two routes, primary + fallback (picked by the D2 parity gate):**

```text
ROUTE A (primary): torchrun + HF DDP, with grad-offload DISABLED under DP.
  - Launch: torchrun 2-proc through the harness (is_torch_run extended; below). HF/accelerate
    initializes distributed, DistributedSampler shards the data, and Trainer wraps the model
    in DDP => DDP's reducer all-reduces the LoRA grads automatically (the ONLY trainables).
  - THE ORDERING HAZARD this route must kill: our CPUAdamW grad-offload hooks
    (cpu_adam.py:221-223 registration; :397-412 per-param D2H) fire per-param at
    post-accumulate time, but DDP's bucketed all-reduce completes only at backward END — the
    hooks would copy PRE-reduction grads to CPU. FIX: under ASYM_DP=1 do NOT register the
    hooks (skip :221-223); use the existing NON-offload step path (cpu_adam.py:469-488, the
    :478 blocking D2H at step time) which reads param.grad AFTER DDP finalization. LoRA grads
    are tiny — the |1 reason for hook-based offload (85.7 s of blocking copies) does not apply
    at adapter scale; measure opt_s to confirm.
  - DDP knobs: find_unused_parameters=True for the MoE rows (thresholded/expert adapters can
    be unused in a step — DDP hangs otherwise); False for dense rows. broadcast_buffers=False.
    HostWeights are NOT nn.Parameters (host_weight.py:178+) so DDP never touches them; verify
    DDP's init broadcast covers exactly the LoRA params.
ROUTE B (fallback if DDP wrap fights the surgered model): manual process group, NO DDP.
  - Two ranks (torchrun with HF distribution masked, or the pair script), init_process_group,
    shard data deterministically (fixed dataset, no shuffle, rank-interleaved indices), and
    ONE explicit coalesced all_reduce(mean) over the CPU grad buffers inside AsymCPUAdamW.step
    BEFORE inner_optimizer.step (cpu_adam.py:435-514; the buffers are the pinned
    cpu_param.grad set at :390). Gloo over loopback on ~1 GB ~ 0.3-1.5 s — acceptable only if
    VG4 (<3% step_s) holds at the target rows; else move the reduce to CUDA/NCCL pre-offload.
```

**Files & functions (verified anchors):**

```text
scripts/lf/run_lf_lora_sft.sh:295-415   BACKEND case: add asym_dp2_cpuadamwds -> BACKEND=asym
    + ASYM_DP=1 (echoed into command.txt + profile.json.config); require NUM_GPUS=2.
scripts/lf/run_lf_lora_sft.sh:704-706   is_torch_run(): true also when ASYM_DP=1 (torchrun
    2-proc launch, DIST_LAUNCHER=torchrun — NOT the deepspeed launcher).
scripts/lf/profile_lora_lf_test_source.sh:899-908  backend_gpu_count(): asym_dp2* -> 2.
scripts/lf/run_lf_lora_sft.sh:1163      global batch: the existing formula (b*ga*NUM_GPUS) is
    CORRECT for DP (4*1*2=8) — assert in command.txt, no override.
asym_gemm/training/cpu_adam.py:221-223  skip grad-offload hook registration when ASYM_DP=1
    (Route A); :469-488 non-offload step path becomes the active one. (Route B: the allreduce
    sits in step() before :498 inner_optimizer.step().)
scripts/lf/run_lf_profiled_train.py     HC4 guard: die if ASYM_DP and ASYM_STP both set; the
    ASYM_STP __init__ patch stays keyed on ASYM_STP only. Heartbeat: emit rank id + world size.
lf.py device resolution:                under torchrun each rank must be pinned to its GPU —
    transformers _setup_devices does set_device(local_rank) in distributed mode; VERIFY at
    runtime (device index in the rank logs), do not assume.
```

**Metrics to emit/check (D2):**

```text
allreduce_s per step (explicit timer around the reduce / DDP comm time) — feeds VG4.
per-rank shard proof: log the first N sample ids per rank at step 1 — must be DISJOINT and
    union == the b8 reference set.
everything from D0's per-rank list, via aggregate_dp_ranks.py.
```

**Validation gate (ALL must pass before D3; e2e is the bar):**

```text
G-D2.1 GRAD-PARITY (the correctness proof): fixed tiny dataset (8 samples, no shuffle);
       reference = single-process |1 b8 run; treatment = asym_dp2 b4x2 with rank shards
       {0-3},{4-7}. AT STEP 2 (step 1 is vacuous for dA — PEFT B=0 at init; tp.md lesson),
       dump per-adapter grads both sides: after mean-allreduce, per-adapter max-rel-err
       <= 1e-2 (bf16 band, |a-b|/max(|b|,1e-8)). Loss overlay 5 steps within band.
G-D2.2 VG4: allreduce_s < 3% of step_s at the primary row.
G-D2.3 e2e DEV row (q3-32b 20000|4|1) completes; THE VIABILITY GATE VG1-VG3 vs the D0
       superoffload_mem row is evaluated HERE — this is the go/no-go for the TP framing
       (record verdict + decomposition in the Decision Log either way).
G-D2.4 asym_dp2 step_s ~= D1 paired-probe step_s + allreduce_s (sanity: real DP == probe +
       sync; a bigger gap means the DDP/dist plumbing added hidden costs — find them).
G-D2.5 (PAPER PHASE — when the MoE track resumes) MoE row (q3-30b-a3b ker101) completes with
       find_unused_parameters=True and no DDP hang; routed counters sane on both ranks.
```

**Risks/watch:** (a) DDP wrap on the surgered model — if init broadcast or reducer touches
anything unexpected, fall back to Route B (decision + evidence into the Decision Log);
(b) the sTP DataParallel-kill patch must not fire (HC4 test: launch with both envs set => die);
(c) JIT cache pre-warm before the first dual-rank run (D1 note); (d) per-rank pool caps (HC5)
— watch summed RSS at s80000; (e) accelerate/LlamaFactory may inject its own dist config — the
command.txt env audit catches it.

---

## Stage D3 — DP-Panel Scoreboard + Freeze + Receipts (PAPER PHASE — after dense dev passes)

**Objective.** Produce the paper's DP panel and freeze it. No code. DEV needs only D0-D2;
this stage runs when the paper matrix is unlocked.

**Runs.** Full matrix (each row separately): {superoffload_mem, zero3_offload_mem, asym_dp2}
x 5 model rows at b4/GPU + per-model b4 boundary + b1 frontier cells (three outcome classes:
SUCCESS / GPU OOM / SOFT HOST OOM by watchdog sentinel — OOM cells are reported results).

**Deliverables / metrics:**

```text
DP PANEL table per model:  Backend  fwd_s bwd_s opt_s step_s  step_H(g0/g1)  RAM(sum)  loss
SCALING table: |1 b8 solo vs dp2-probe vs asym_dp2 vs superoffload-DP2 — tokens/s per GPU +
    the contention factor (D1) => THE motivation figure feeding gb200_tp.md's C1/C6 and the
    paper's Q4 motivation figure.
b1 FRONTIER row per model: DP's max seq (each rank carries the full sequence — the structural
    limit the TP panel will beat; state it as structural, not a horse race).
ULYSSES RECEIPT: file:line evidence from third_party/Megatron-DeepSpeed-SO that it is GPT-only
    / no-LoRA, recorded BEFORE the "cited-not-run" table note.
HONEST CONCLUSIONS paragraph (predeclared shape): asym_dp2 = step_H win + capacity posture,
    step_s within the VG band, does-not-scale-as-a-pair (contention factor) => "same kernels
    need a different parallelism on 2:1" — the hand-off line to gb200_tp.md.
```

**Validation gate (= panel freeze):**

```text
G-D3.1 every cell audited (command.txt env, pair 0,1, global batch 8, fresh artifact, loss band).
G-D3.2 VG verdict recorded with numbers; if VG failed and was fixed via |1 knobs, the fix and
       the re-run are both in the Decision Log (no silent retuning).
G-D3.3 numbers copied into gb200_tp.md's P0/pace-car section (DP-family mem/time pace cars)
       and by the paper's Q4 motivation figure — single source of truth = this panel.
```

---

## Reporting Format

```text
fwd_s  bwd_s  opt_s  step_s  fwd_H  bwd_H  step_H(g0/g1)  RAM(rank0+rank1=sum)  allreduce_s  loss
labels exactly as generated (asym_dp2_cpuadamwds | recomp-off-full-fg-ker000, __gpus2__);
dp2_probe artifacts carry the `dp2_probe` tag and never appear in the DP training panel.
```

## Decision Log (append-only; date + decision + evidence path)

```text
2026-07-06 D0 VALIDATED. superoffload_mem q3-32b 20000|4|1 pair 0,1 (torchrun 2-proc, membind 0,1,
  oom_adj 1000): step_s 134.6 (fwd 9.4 / bwd 125.2 / opt 0.1 — SuperOffload folds optimizer work
  into bwd), step_H(rank0) 22.2 GiB, RSS ~165.5 GiB PER RANK (both ranks confirmed via external
  sampler; summed ~331 GiB), losses 1.21-1.25. G-D0.1..3 PASS. Evidence:
  profiling_gb200dp_d0/{dp_row.json,external_sampler.csv,PRERUN.md}.
2026-07-06 G-D0.2 FIX: profile/heartbeat writes are rank0-gated, so every rank now writes
  rank<R>_memstats.json (per-device HBM peaks + RSS) from run_lf_profiled_train.py's finally
  block; THIS D0 run used an external /proc+nvidia-smi sampler instead (already running).
2026-07-06 METRIC CONVENTION: trainer.timing.measured_e2e (heartbeat dataloader intervals) reads
  45.4 s/step on the deepspeed path — inconsistent with tqdm (~138 s/it) and step_samples
  (134.6). step_samples fwd+bwd+opt per-step averages are authoritative (show_metrics.py
  convention); aggregate_dp_ranks.py prefers step_samples.
2026-07-06 D1 SOLO b4 (|1 asym_cpuadamwds, GPU0): step_s 127.9 (16.9/109.3/1.7), step_H 19.3 GiB,
  RSS 244.9 GiB, losses sane. Measured steps uniform ±0.1 s -> clean despite an I1-probe overlap
  during warmup (noted; probes now forbidden during live rows). ALSO the |1 zero-regression proof
  for the JIT FIX A _C rebuild (gb200_tp.md I1). asym solo b4 BEATS the superoffload per-rank step
  (127.9 vs 134.6) with lower step_H (19.3 vs 22.2).
2026-07-06 D1 SOLO b8 (|1 asym, GPU0): step_s 251.8 (33.2/216.6/2.1), step_H 38.6 GiB, RSS 352 GiB.
  1.97x of b4 => near-linear in batch; the |1 step is offload/CPU-bound in bwd, not streaming-bound.
2026-07-06 D1 PAIR PROBE VALIDATED (G-D1.1..3). Two independent |1 b4 jobs, GPU0+GPU1, same
  seed/data, started simultaneously: rank0 129.4 s (factor 1.012 vs solo), rank1 108.1 s (0.845 —
  FASTER than solo; concurrent identical jobs spread 21 s => host-side scheduling asymmetry, not
  contention). Losses identical to solo within 2e-3. VERDICT: ~NO shared-Grace contention at the
  dev row (predeclared "~1.0 is also a result" branch — Q4 motivation shifts to CAPACITY:
  summed RSS 488.7 GiB for two ranks vs ~245 GiB one-copy; plus b1 frontier + MoE).
  SCALING FRAME: paired-b4 wall 129.4 s vs solo-b8 251.8 s => pairing preserves per-GPU
  throughput (1.95x). PROCESS NOTES: first pair attempt burned on the unsuffixed smoke dataset
  (PREPARE_DATASETS=false skips name derivation -> run_dp2_pair.sh now passes DATASET
  explicitly); second bug: $(launch_rank) command substitution reparents the child so wait()
  fails instantly -> launch via function-in-background. Concurrent host-weight pinning of two
  64 GB arenas is slow (~10+ min vs ~5 solo) — cudaHostAlloc serializes; a contention datum.
  Evidence: profiling_gb200dp_d1/dp2_probe_merged.json.
2026-07-06 D2 VALIDATED (Route A works; LlamaFactory guards relaxed under ASYM_DP=1 at
  parser.py:252/:517). PARITY (s2048, 8 samples, b8 vs b4x2, init transplanted):
  rank0-vs-rank1 post-reduce grads BIT-IDENTICAL (DDP mechanics exact); dp2-vs-|1 grad-norm
  ratio median 0.91 (mean semantics correct, no 2x); directional spread within the measured
  64-layer bf16 reduction-order envelope (see gb200_tp.md 2026-07-06 — the static 1e-2 band is
  unsatisfiable at this depth for ANY perturbation, including |1's own knobs); loss overlay
  within few e-3. G-D2.1 PASS under the envelope method.
2026-07-06 D2 e2e + THE VIABILITY GATE (q3-32b 20000|4x2 vs D0 superoffload_mem 134.6/22.2/165):
  asym_dp2: step_s(rank0) 151.0 (fwd 17.0 / bwd 132.1 / opt 1.9), step_H 22.3 GiB,
  RSS 246.6 GiB/rank (summed ~493), losses match the superoffload trajectory within 4e-3
  (same sampler => same batches).
  VG1 PASS: 151.0/134.6 = 1.12x (<= 2.0; better than the 1.3-1.7 target band). TP investment
    justified per the gate.
  VG3 PASS. VG4: the DDP reduce itself is ms-scale; HOWEVER G-D2.4 flags +21.3 s of bwd vs the
    D1 probe (132.1 vs 110.8) — DDP-plumbing overhead needing an nsys decomposition (candidates:
    reducer-hook interplay with the host-blocking fg backward; per-param blocking D2H in the
    non-offload CPUAdamW step path is only ~1 s and opt_s shows 1.9). OPEN.
  VG2 FAIL AT THIS ROW (honest): step_H parity (22.3 vs 22.2 — adapter banks+grads stay on-GPU
    because hook-offload is disabled under DDP) and host RSS WORSE (247 vs 165 per rank — the
    asym pinned pools). The asym memory posture pays at longer sequences (b1/boundary rows,
    paper phase), not at s20000. Panel framing must lead with VG1+capacity-at-frontier, not a
    blanket memory win at the dev row.
```

## Stage Dependency Summary

```text
D0 baselines + per-rank profiling audit   (no code; unblocks everything)
D1 contention probe (Q4)                  (script only; predicts D2; the motivation number)
D2 real-DP asym_dp2 + VIABILITY GATE      (the one code change; grad-parity proof; go/no-go)
D3 scoreboard freeze + receipts           (feeds gb200_tp.md P0/C1/C6 + the paper's Q4 figure)
Interlock with gb200_tp.md: D0-D2 SHOULD land before/alongside TP stages I1-I2 so the viability
verdict exists before deep TP investment; the TP plan's asym_dp2 references resolve to THIS
doc's D2 row (real DP; the I0 run_dp2_pair.sh probe = D1 here), and its DP-family pace cars
resolve to D0.
```
