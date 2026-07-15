# Handoff: Memory↔Throughput Scheduler for AsymGEMM LoRA-SFT

## The insight driving this task

AsymGEMM (`asym*` backends) and SuperOffload (`superoffload_mem|unsloth-off`) are two
points on ONE frontier: what lives in HBM runs fast (stock cuBLAS, no offload traffic);
what lives on host is memory-cheap but slow (custom asym GEMM ≈ ⅓ cuBLAS speed +
transfer/orchestration). Today the placement is a STATIC config choice. The task:
**a scheduler that, for a requested (model, seq, batch), places each tensor class in
HBM when it fits and degrades to CPU-resident asym-style only for what doesn't** —
spending memory headroom on throughput. At s ≪ ceiling, asym has 100s of GB unused;
it should then run ≈ superoffload speed. Near the ceiling it degrades gracefully to
full-asym and keeps its +20–32% max-seq advantage. One backend, best of both.

## Measured ground truth (2026-07-09, GB200 node, serial 4-step runs @32k×8; see `agent/fix_throughput.md`)

| q3-30b-a3b @32k | step | fwd | of which |
|---|---|---|---|
| superoffload_mem unsloth-off | 66.6 s | 8.1 s | stock kernels, wholesale weight/opt streaming |
| plain `asym` (GPU AdamW) | 109.6 s | 19.0 s | **+43 s = asym GEMM tax + act-offload path** |
| `asym_cpuadamwds` (flagship) | 117.0 s | 18.9 s | **+7.4 s = ENTIRE cpu-adam machinery** |

- Causal control established: the CPU-Adam hooks' 30.4 s/step host-block does NOT convert
  to wall time (GPU overlaps it). The gap is the GEMM engine + activation offload path.
- Ceilings (confirmed, b=8): asym: llama 34k@ohbm3, q3-32b 65k@ohbm8, q3-30b 173k@ohbm0;
  superoffload unsloth-off: 32k / 53k@ohbm4 / 131k. Memory is linear in T=B×s
  (validated R²=1.00); per-config fits + maxB/latency tables: `scripts/lf/ceiling_estimate.py`
  → `scripts/lf/ceiling_table.md` (+ `ceiling_table_record.md` incl. estimates).
- nsys traces (existing, `profiling_results/profiling_both/.../ceiling__*32000*/...__nsys__*/trace.sqlite`):
  asym = 64-70% kernel-busy, custom `sm100_bf16_asym_gemm_impl` ≈ ⅓ of cuBLAS peak
  (llama: ~137 s/step vs ~43 s ideal); superoffload = 36-43% busy, 2-3× copy volume,
  stock nvjet at peak. Decompose with `scripts/lf/analyze_stp_bwd.py <trace.sqlite>`.

## Levers that already exist (no new kernels needed for v1)

1. **ohbm knob** (`-ohbm<N>` suffix): keep every Nth Unsloth-GC root in HBM (share=1/N).
   Already searched per config; scheduler picks N from headroom.
2. **Per-class offload env flags** (see `scripts/lf/run_lf_lora_sft.sh` and the label
   grammar in `profile_lora_lf_test_both.sh`): `ASYMM_ATTN_ACT_OFFLOAD`,
   `ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD` (the `full-fg` in labels),
   `ASYM_CPU_ADAMW_{GRAD,WEIGHT}_OFFLOAD`, `ASYM_OFFLOAD_MODULES`, recompute label
   variants (`recomp-off` without `-full-fg`, `unsloth`, `unsloth-off`...).
3. **Hybrid GEMM dispatch** (v2, code change): route HBM-resident operands to stock
   cuBLAS/nvjet, keep `asym_gemm_impl` only for genuinely CPU-resident operands.
   Kernel: `asym_gemm/include/asym_gemm/impls/sm100_bf16_asym_gemm.cuh`; ncu helper:
   `scripts/lora/profile_ncu_asymgemm.py`.
4. **Memory model for the decision**: peak ≈ intercept + slope·(B×s) per placement
   config; calibrate from 2 anchor runs per config (artifacts:
   `memory_actual_peak_breakdown.csv` — has per-component/per-class bytes! — and
   `process_memory.csv` in each run leaf). Effective capacities anchored by the
   confirmed ceilings.

## The task — part 0: DIAGNOSE AND REASON FIRST. DO NOT IMPLEMENT YET.

Your first deliverable is analysis, not code. Concretely:

1. **Read `agent/impls/scheduler_v2.md`** (370-line formulation + latency-recovery plan, v2,
   already exists) — it formalizes homes {HBM, DRAM}, the memory↔latency dial, and a
   steady-state timing protocol (≥5 steps, drop warmup + first + last measured). Treat
   it as the theoretical frame; reconcile anything below that disagrees with it.
2. **Re-verify the diagnosis yourself from artifacts** (zero GPU): the stall budgets and
   trace decompositions in `agent/fix_throughput.md` §2.2/§2.4, the control-experiment
   result (cpu-adam machinery = only ~7.4 s/step wall; earlier "269 s sync-hook" theory
   REFUTED as a wall-time item), and the per-kernel evidence (`analyze_stp_bwd.py` on the
   existing 32k traces). History warning: one plausible-sounding diagnosis already
   collapsed under a control experiment — treat every remaining claim (GEMM ≈ ⅓ peak;
   act-offload path = the fwd gap; ~147 ms periodic gaps) as UNVERIFIED until you've
   reproduced the number from an artifact or a cheap control run yourself.
3. **Close the open diagnostic holes** before designing: (a) WHERE exactly does the
   remaining 43 s/step go — split it between GEMM-engine time, act-offload transfer
   waits, and orchestration gaps, per phase (use ncu on `sm100_bf16_asym_gemm_impl` via
   `scripts/lora/profile_ncu_asymgemm.py`, and per-stream nsys timelines); (b) why is
   plain-asym FORWARD 2.3× slower than superoffload's with no optimizer involved —
   quantify kernel-tax vs D2H-write share; (c) confirm the v3 cpu_adam fix A/B outcome.
4. Only THEN write the design-space survey (part A) and propose — not build — the
   implementation plan (part B). **Implementation starts only after the diagnosis doc
   and survey are reviewed.**

## The task — part A: EXPLORE the scheduling design space (after part 0)

Before building, produce a written survey (`agent/scheduler_design_space.md`) of the
scheduling opportunities — the goal is a single tunable knob (or small set) that slides
the system between **throughput-oriented** and **memory-oriented**. This space is large;
enumerate and evaluate before committing. Dimensions to cover at minimum (add your own):

- **Granularity of placement decisions**: whole-backend preset → per tensor CLASS
  (weights / optimizer states / grads / attn acts / MLP acts / GC roots — the current
  env-flag granularity) → per LAYER (e.g. first K layers' acts resident, rest offloaded;
  ohbm is already a per-layer-stride instance of this) → per TENSOR/operand (hybrid GEMM
  dispatch) → per PHASE (fwd-resident, bwd-streamed).
- **Decision time**: static per-run (config chosen at launch from the memory model) vs
  runtime-adaptive (watch HBM/host headroom + achieved BW, migrate placements between
  steps; riskier, fingerprint/reproducibility implications).
- **What to spend headroom on first** — rank classes by (throughput gained)/(byte spent):
  e.g. keeping MLP acts on GPU kills both the fg-offload traffic AND moves those GEMMs
  to stock kernels (double win) vs ohbm roots (transfer-only win). Use the per-class
  bytes from `memory_actual_peak_breakdown.csv` + per-class time attribution from the
  nsys traces to build this ranking EMPIRICALLY per model/seq.
- **The knob's semantics**: continuous "memory budget to use" (e.g. `TARGET_HBM_GB`,
  `TARGET_HOST_GB`) that the scheduler fills greedily by the ranking above — vs discrete
  presets (`--profile=throughput|balanced|max-seq`). Recommend one; justify.
- **Interaction with batch**: at fixed s, headroom can also buy larger B (more tokens/step)
  instead of faster placement — when is which better? (t0 is small, so placement usually
  wins; show the math from the fitted latency model.)
- **Safety margins**: distance-to-watchdog / distance-to-HBM-OOM as scheduler inputs;
  never schedule into the thrash zone (see pitfalls).
- **Degradation policy**: what gets evicted first when the requested (s,B) doesn't fit —
  the reverse of the spend ranking; must reproduce today's full-asym behavior at the
  extreme so ceilings are preserved.

Deliverable: the survey doc with a recommended architecture + the ranking table, THEN
implement incrementally:

## The task — part B: implementation plan (PROPOSE ONLY in this pass — build after review)

1. **v0 (static schedule table)**: for each (model, seq-bucket, batch), choose the
   fastest *existing* config that fits with margin (host-avail ≥ 2× the 35 GB watchdog
   floor; HBM peak ≤ ~183 GiB): full-superoffload-style < partial (acts on GPU, weights
   streamed) < full-asym. Encode as a lookup applied by the launcher (env preset per
   bucket). Validate: tok/s @{8k,16k,32k} ≥ 0.9× superoffload while ceiling runs still
   reach the confirmed max seqs.
2. **v1 (auto-tuner)**: compute placements for requested (s,B) from the linear memory
   model + per-class byte costs (peak-breakdown CSV) instead of a hand table.
3. **v2 (hybrid dispatch)**: per-operand runtime choice inside the backend (stock GEMM
   for HBM-resident, asym GEMM otherwise). Biggest win, real code change.

## Protocol & pitfalls (hard-earned; do not skip)

- **Measurement**: strictly SERIAL — one experiment per NODE (shared host RAM/CPU/C2C
  contaminate even different-GPU runs); `MAX_STEPS=4 WARMUP_STEPS=1`; drop warmup + last
  measured step; `PROFILERS=source` for timing, `both` only for diagnosis; never anchor
  near the memory wall (thrash: q3-32b @65k ran 2× slower than @64k, one grid step).
- **Ceiling re-verification**: memory-relevant changes move the config fingerprint in
  `scripts/lf/ceiling_search_both.sh` → re-run the ceiling search for changed configs (the
  ledger replays unchanged ones). Never launch two ceiling drivers (the flock file was
  deleted once — mutual exclusion is broken; `pgrep -f ceiling_search.py` first).
- **Known fixed bug (KEEP the fix)**: `asym_gemm/training/cpu_adam.py` — `zero_grad`
  never reaches the optimizer under LF, so `step()` now self-clears
  `grad_buffer_has_data` (previously grads ACCUMULATED across steps = subtly wrong
  math) and async bf16-staged grad D2H is in (`ASYM_CPU_ADAMW_ASYNC_GRAD_OFFLOAD`,
  default on). Wall win ~≤7 s/step, but correctness matters. v3 A/B may still be
  pending — check `asyncfix_v3_ab` artifacts or rerun the 32k A/B.
- Loss comparisons vs pre-fix runs are NOT bit-comparable (old trajectories carried the
  accumulation bug); validate loss sanity/convergence, or vs plain `asym`.
- Full context: `agent/fix_throughput.md` (stall budgets, trace decomposition, fix
  ranking + control result), `scripts/lf/ceiling_table*.md`, `agent/impls/scheduler_v2.md`
  (earlier planning notes).

## Success criteria

1. s ≤ superoffload's ceiling: scheduled-asym tok/s ≥ superoffload −10% (v0/v1),
   ≥ parity (v2).
2. s > superoffload's ceiling: asym remains the only runner; confirmed ceilings
   (34k/65k/173k @ b8) do not regress.
3. One flag/entry-point, no per-run hand-tuning; placement decisions logged per run.
