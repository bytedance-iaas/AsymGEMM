# GB200 sTP: Streamed Tensor Parallelism for LoRA (`|2`) — Staged Implementation Plan

**What/why.** `asym_stp` = single-process TP-2 LoRA backend for one GB200 superchip: the frozen
base lives ONCE in a pinned Grace arena; each GPU streams its own shard's tiles over its own C2C
lane (AsymGEMM cpu-right; zero weight-HBM); LoRA + CPUAdamW unchanged from `|1`. Lands behind
`ASYM_STP=1` + new backend names; never touches `|1` defaults. Discipline mirrors
`fix_finegrained_*`: staged, gated, one experiment at a time, artifacts never overwritten.

```text
FLAGSHIP:     asym_stp_cpuadamwds ("stp" = streamed TP; "_cpuadamwds" = DeepSpeed CPU-AdamW,
              masters in host DRAM per HC1; NVMe variants opnvme/panvme OUT OF SCOPE).
paper names:  TP-Resident (tp2_resident_*), TP-Staged (tp2_offstage_*),
              AsymLoRA-DP (asym_dp2 — owned by gb200_dp.md), AsymLoRA-sTP (asym_stp_*)
EXACT dev/test workloads + baselines-to-beat: next section (copy-paste RUNS lines).
target models (verified: scripts/lf/profile_lora_lf_test_source.sh:50-60):
  llama3.3-70b = meta-llama/Llama-3.3-70B-Instruct   (80L, H=8192, I=28672, q64/kv8, hd128)
  q2.5-72b     = Qwen/Qwen2.5-72B-Instruct           (80L, H=8192, I=29568, q64/kv8, hd128, QKV-bias)
  q3-32b       = Qwen/Qwen3-32B                       (64L, H=5120, I=25600, q64/kv8, hd128, q/k-norm)
  q3-30b-a3b   = Qwen/Qwen3-30B-A3B                   (48L, H=2048, I=768,  E=128 top8, q32/kv4, no shared expert)
  llama4-scout = meta-llama/Llama-4-Scout-17B-16E     (48L, H=5120, E=16, q40/kv8, hd128, HAS shared expert)
```

## STATUS (2026-07-06 — read this first; details in the Decision Log)

```text
DONE + VALIDATED: I0 (harness/guards; dry-run + 9 negative gates), I1 (JIT FIX A
  -DDG_JIT_USE_RUNTIME_API — dev1 GEMMs work; probe: P2P 778/774 GB/s, dual-lane shared-pinned
  174.7 GB/s/lane, allreduce2 3 GiB 6.11 ms; |1 zero-regression), I2 (arena repack: col =
  zero-copy views, row = shape-preserving carrier + _dispatch_nt guard), I3 kernel parity
  (gather BIT-IDENTICAL; partial-sum within fp32-ref band), I4 FULL TP e2e — built as a
  TWO-INSTANCE design (StpDecoderLayer runs two copies of the |1 machinery over shard
  HostWeights, joined only by boundary Fns; per-layer Bcast01/Join01 => GC-safe; mirrors for
  replicated LoRA pieces + post-bwd merge + post-step resync). Trains s2048+s20000 through the
  harness; LOSS TRACKS |1 to ~1e-3. Pace car #2 built standalone (fsdp2_tp_baseline.py:
  head-split TP2 + FSDP2 CPUOffload + save_on_cpu).
DEV-ROW NUMBERS (s20000 b8): stp 234.9 s / 27.4+25.2 GiB / RSS 395 | |1 b8 251.8 / 38.6 / 352
  | superoffload b4x2 134.6 / 22.2 | fsdp2-tp2 26.8 / 133.4 (GPU-OOM at s32k b8).
REMAINING / OPEN ISSUES (priority order):
  1. I5 NOT BUILT — THE step_s/RSS lever: each branch currently offloads its OWN full-width
     [M,H] copy of U/X/GC-roots (dup => bwd 209 s barely beats |1's 216.6, RSS +43 GiB).
     Dedup = offload once on dev0, restage both lanes; + row-split dA; + BWD SCHEDULING RULE +
     de-sync (a)-(d) (still inherited |1 host-blocking order per branch).
  2. LoRA-bank offload disabled under sTP (mirror-merge ordering) -> step_H 27.4 not <=~20;
     re-enable or fold into I5.
  3. lm_head/loss segment: dev1 idles through final-norm+CE (fwd 22.3 vs |1 33.2 = 0.67x, not
     0.5x) — measure share, split only if >5%.
  4. P-DEV REVISION NEEDED: step_s-vs-pace-car is structurally unwinnable at fits-in-HBM rows
     (pace car 26.8 s); evaluate step_s at a FRONTIER row, keep s20000 for step_H/loss. The
     winning rows: step_H 4.9x under fsdp2-tp2; frontier (it dies s32k; stp headroom multi-x);
     b1 ladder + actnvme vs SuperOffload forms (SP baseline: build DS-Ulysses-on-HF, days —
     NOT the Megatron-DS port).
  5. PARITY METHOD (binding lesson): 64-layer bf16 step-2 adapter grads are ~O(1) sensitive to
     ANY reduction-order change (measured |1-vs-|1 envelope p50 0.46 / max 1.72) — gates MUST
     use the measured-envelope method; sharp instruments = mini-parity
     (stp_full_tp_mini_parity.py) + loss overlay + cross-rank bit-identity.
  6. Phase-A ablation currently FAILS at s2048 AND its artifact label collides with full-TP
     (needs its own tag). I6 ladder rungs, I7 MoE, I8 matrix: untouched.
```

## Dev Workloads & Baselines-To-Beat (EXACT configs — copy-paste into RUNS)

```text
DEV RULES: ONE model (q3-32b — DENSE first; MoE deferred), ONE decently-sized workload
(s20000 b8 ~ 160k tokens — well inside every boundary: |1 asym C-OOM ~53k, SO|unsloth-off
C-OOM ~53k), and host RSS comfortably under the watchdog floor at ALL times (HC2). No heavy
rows, no multi-model matrix during dev.

# ============ DEV TARGET (ours; TP: ONE job on pair 0,1; b8 = global 8) ============
q3-32b|2 ; asym_stp_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 20000|8|1 ; none|false|false|false|false|false
# small parity/loss-gate row (I3/I4 grad-parity + logits dumps):
q3-32b|2 ; asym_stp_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false

# ============ DEV LADDER RUNGS (ablations; same row, one WEIGHT_MODE knob apart) ============
q3-32b|2 ; tp2_offstage_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 20000|8|1 ; none|false|false|false|false|false
q3-32b|2 ; tp2_resident_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 20000|8|1 ; none|false|false|false|false|false

# ============ DEV BASELINE #1 — TP apples-to-apples (THE one to beat) ============
# FSDP2(CPUOffloadPolicy) x TP-2 DeviceMesh + LoRA, q3-32b s20000 b8 global.
# NOT in the RUNS grammar -> NEW scripts/testing/fsdp2_tp_baseline.py (torchrun 2-proc +
# profile.json metrics shim). NeMo Automodel ON HOLD.

# ============ DEV BASELINE #2 — best DP (owned by gb200_dp.md) ============
q3-32b|2 ; superoffload_mem|unsloth-off|ligerloss1 ; 20000|4|1 ; none|false|false|false|false|false
q3-32b|2 ; asym_dp2_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 20000|4|1 ; none|false|false|false|false|false

# ============ |1 DEV REFERENCES ============
q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 20000|8|1 ; none|false|false|false|false|false
q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 20000|4|1 ; none|false|false|false|false|false

# BEAT relation (dev) = P-DEV: asym_stp beats FSDP2+TP2 (b8) AND superoffload-DP2 (b4) on
# step_s AND step_H at the dev row; ladder ordering step_H resident > offstage > stream;
# loss in band everywhere.
# NOTE: per-stage e2e validation commands below are TEMPLATES (some written against
# llama3.3-70b) — during dev SUBSTITUTE this q3-32b row (s20000 target; s2048 parity).
# PAPER PHASE (DEFERRED until dense dev passes): the 5-model matrix at target rows
# (llama3.3-70b 25000, q2.5-72b 30000, q3-30b-a3b 80000 ker101, llama4-scout 9500, q3-32b
# 50000), zero3 + unsloth-variant rows, b4 boundaries, b1 frontiers, Stage I7 MoE, NeMo FW rows.
```

## HARD CONSTRAINTS (non-negotiable — a violation risks a machine-wide host OOM, not just a failed run)

```text
HC1  MEMORY = CPU DRAM ACROSS BOTH GRACE NODES; the 2 GPUs are COMPUTE ONLY. The frozen base
     weights, offloaded activations, and CPUAdamW masters all live in PINNED HOST memory — GPU HBM
     holds NO resident base weights (that IS sTP). Target workloads exceed one Grace node's ~480 GB
     (e.g. q3-30b-a3b s80000 ~ 640 GB RSS), so every run MUST have BOTH CPU/NUMA nodes' DRAM
     available: NUMACTL_MEMBIND=0,1 (the harness DEFAULT — leave it). NEVER restrict the memory pool
     to a single node; it OOMs the large rows. HBM is for live activations + LoRA + streamed tiles
     only. (Locality is a SEPARATE, softer concern: the pair 0,1 is local to node 0, so streaming a
     node-1 tile costs the ~117 GB/s cross-superchip path; the coord knob may FIRST-TOUCH the hot
     weight arena on the pair-local node — but that never restricts capacity, HC1 wins.)

HC2  EVERY launched training process MUST carry the host-OOM guards: TRAIN_OOM_SCORE_ADJ=1000 (the
     training child is the FIRST thing the kernel OOM-killer takes, protecting the machine + other
     jobs) AND HOST_MEM_WATCHDOG=true at its 35 GB floor (gracefully STOPs the child + writes
     ${LOG_FILE}.host_mem_watchdog_fired BEFORE a hard kernel OOM). Both are echoed into command.txt
     + profile.json.config (I0). A run without both is INVALID and must not be reported.

HC3  LAUNCH ONLY through scripts/lf/profile_*  OR  scripts/lf/run_lf_*  — those are the ONLY entry
     points with HC2's guards wired in (oom_score_adj + watchdog run_lf_lora_sft.sh:1610-1681). NEVER
     start real training via a bare `python ...`, a notebook, or a bespoke script — it runs WITHOUT
     the watchdog and WILL risk a hard host OOM that can take down the box. Isolated probes
     (stp_runtime_probe.py, parity probes) may run standalone ONLY because they cap pinned host
     allocations to SMALL test buffers; any probe that would allocate a full model's host arena must
     go through the launcher scripts instead.
```

## Why This Design Is Correct & Efficient (derived + reused from `third_party/`, verified against source)

```text
POSITIONING — the claims (REVISED per the live paper-story analysis: headline baseline =
one-copy STAGED TP-2 (tp2_offstage, kills the TP-vs-DP confound), the streamed~staged tie at
dense large-M is PREDECLARED, and the M3'/M4 mechanisms + qualified flip-boundary claim live in
the paper-story doc; receipts archived in archive/gb200_angle.md):
- The SOTA offloaded-training baseline SuperOffload (= superoffload_mem) is ZeRO-3 DATA-PARALLEL
  (TP=1), assumes 1 GPU:1 Grace, prefetches 64 MB weight BUCKETS into HBM, is full-FT + MoE-agnostic.
  Ported to GB200's shared Grace it duplicates + contends (2 ranks, 1 Grace). sTP IS the redesign
  that 2:1 penalty forces: ONE copy, disjoint TP, shared arena, tile-wise streaming — a structure
  DP/1:1 cannot express. That is the GB200-specific contribution.
- NOVELTY is narrow + honest: (1) tile-wise, NEVER-in-HBM weight streaming on the fwd AND
  BACKWARD/LoRA-grad path (predecessors are inference-only or prefetch whole units to HBM);
  (2) TP (not DP) offloaded streaming on a shared-Grace arena; (3) frozen-base + LoRA; (4) MoE-aware
  ownerless-expert streaming. Do NOT claim as novel: "stream from Grace" (SuperOffload does it),
  "save a copy" (weak), or FP4/quantization (Hopper does int4 QLoRA).
- HONESTY GUARD: the win is HBM-CAPACITY / no CPU->HBM->SM round-trip / SHARED-GRACE SCALING, NEVER
  a raw C2C-BANDWIDTH claim (tile-wise does NOT beat 64 MB buckets on C2C BW — SuperOffload Fig 7).
- COROLLARY (cheap, clean): equal-COUNT token slices make per-GPU compute skew-immune, so the MoE
  load-balance aux loss / capacity-factor token-drop are unnecessary for frozen+LoRA (zero drop).

CORRECT — the TP math is NOT invented; it is copied from proven multi-process SPMD TP, and only the
TRANSPORT changes (NCCL collectives -> single-process P2P allreduce2 on copy streams):
- col/row f/g duality copied verbatim from Megatron-Core (third_party/megatron-lm/megatron/core/
  tensor_parallel/mappings.py:197-233, layers.py) -> our TPRegionFn / AllReduce2Fn. By construction:
  exactly 2 fwd + 2 bwd O(M*H) exchanges per dense block, even with unfused q/k/v/gate/up.
- frozen-base backward = LinearWithFrozenWeight (layers.py:350-382): save weight only, dgrad, NO
  wgrad -> our streamed base's exact bwd contract.
- row-LoRA "scale-before-B, keep both partial, ONE reduce" copied from NeMo Automodel
  (third_party/Automodel/nemo_automodel/components/_peft/lora.py:297-313).
- HF DataParallel kill verified against third_party/transformers Trainer (the __init__ _n_gpu=1 trick).
=> grads are provably identical to battle-tested TP-2; ENFORCED by the I4/I7 step-2 adapter-grad
   parity gate vs the |1 run (not assumed).

EFFICIENT — each choice removes a specific cost the vendored baselines pay:
- weight STREAMING (AsymGEMM cpu-right TMA, zero copy engines) removes TP's HBM-residency blocker ->
  we run TP where Megatron-Bridge / NeMo Automodel CANNOT (neither has weight streaming; Bridge
  cpu_offloading = TE bulk per-layer swap PP=1-only, Automodel = FSDP2 whole-unit offload — receipts
  in the Verified Borrow List). This is the paper's novelty.
- DISJOINT LANES: each GPU streams its OWN shard over its OWN C2C lane; DP (asym_dp2) duplicates
  every frozen byte on both lanes -> sTP moves ~0.5x per-lane weight bytes (C1).
- ONE O(M*H) collective per region (E3); EP-2 MoE with ZERO token all-to-all (residual replicated,
  E2, one grouped kernel/device); replicate-r LoRA (our deliberate deviation) REMOVES the refs'
  extra fwd all-gather on S — strictly better at tp2 with r<=64.
- optimizer = DeepSpeed CPU-AdamW masters in host DRAM (no HBM optimizer state, no NVMe).

BASELINE PROVENANCE (what each vendored folder is used FOR):
  DERIVATION (line-audited): megatron-lm, Automodel, Megatron-Bridge, transformers (above).
  MEASURED baselines (black-box via P0, NOT line-audited — correct treatment): deepspeed
    (zero3_offload*, superoffload*), DeepSpeed-SO / Megatron-DeepSpeed-SO / SuperOffload (Ulysses SP).
  FW rows (Tier 2, I8): NeMo Automodel TP-Resident / TP-Staged. DeepSpeed-inexpressibility claim is
    deferred to I8 with a cite-file:line-before-stating rule.
Full reuse/deviation table with line numbers: the Verified Borrow List below.
```

## Hardware Facts (this node; `spec` = vendor, `meas` = our probe, re-verify in Stage I1)

```text
topology  GPU0+GPU1 share Grace/NUMA node 0; GPU2+GPU3 share Grace/NUMA node 1
C2C/lane  NVLink-C2C is 900 GB/s bidir AGGREGATE per superchip (450/dir), SHARED by both
          GPUs  [spec, verified] -> ~225 GB/s/dir per GPU when both stream concurrently
          (the TP-2 case). A solo GPU can burst toward 450/dir.
          H2D per lane ~190-195 GB/s  [meas/extrapolated — NO public GB200 microbench;
          GH200 published read-from-host ~419, write ~378; ~195 = ~85% of 225 shared spec]
GH200 ref 450 GB/s/dir to ONE GPU -> GB200 halves it per-GPU (two GPUs share the link). [spec]
NVLink    GPU<->GPU ~900 GB/s/dir (1800 bidir), 5th-gen, 18 links ("NV18" in nvidia-smi). [spec]
Grace     LPDDR5X ~500 GB/s total behind BOTH lanes (NOT the "1 TB/s" 2-die CPU Superchip). [spec]
remote    cross-superchip host read (GPU0 -> node-1 Grace) ~117 GB/s [meas; nearest public ~100].
          Streaming a node-1 tile to GPU0/1 pays this; keep the HOT weight arena pair-local (coord).
rule      |2 scoreboard rows MUST use a same-superchip GPU pair (--gpus 0,1 or 2,3); --gpus 0,2 only
          for the contention study (ALLOW_CROSS_SUPERCHIP=1). Host MEMORY spans BOTH nodes (HC1).
```

## Verified Code Map (the real anchors; all line numbers checked against the working tree)

Everything below routes through these. Pseudocode in later stages calls them verbatim.

```text
BASE-WEIGHT GEMM (weight is the HOST-resident operand; streams via compute-stream TMA over
C2C — ZERO copy engines). THE choke point for the primary backend:
  asym_bf16_cpu_right_matmul(left, right_cpu, *, backend, stats, phase, tag,
      transpose_b=False, output_dtype=bf16)                     frozen_linear.py:1140
    left = activation (CUDA; SETS the output device); right_cpu = weight [N,K] host-pinned.
    -> single _dispatch_nt(...) at :1185 -> _asym_bf16_nt (:707) ->
       asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(a, b_cpu.unsqueeze(0), d, ...).
  _dispatch_nt(a, b_cpu, *, backend, stats, phase, compiled_dims, transpose_b=False,
      precision="bf16", quantized_weight=None, profile_label="", bf16_output_dtype=bf16)
      -> a@b^T if transpose_b=False, a@b if True                frozen_linear.py:1051
  _direct_bf16_reason(a, b_cpu, *, transpose_b=False)           frozen_linear.py:400
    gate: both 2D bf16; b_cpu CPU+PINNED+CONTIGUOUS, a contiguous; n%8==0, k%8==0;
    transpose_b -> k%64==0 (k = contraction = out_features on the dX GEMM).
  PRIMARY TRAINING PATH executes these NOT via AsymFrozenLinear.forward (:2062, eval/fallback
  only) but via the fine-grained autograd Functions:
    MLP  -> _FinegrainedDenseMLPFunction (dense_mlp_finegrained.py:183) through helpers
            _asym_base_forward (:64) / _asym_base_dx (:86); base calls at :220(gate fwd),
            :236(up fwd), :274(down fwd), :339(down dX), :399(gate dX), :423(up dX).
    attn -> _AsymActivationOffloadLoRALinearFunction (attention_activation_offload.py:560);
            base calls at :599(q/k/v/o fwd), :692(q/k/v/o dX).
  Three bias-add sites (col shards need a sliced [N/2] bias): dense :1322, fg :81-82, attn :609.

LoRA (AsymLoRALinear, lora.py:238):  lora_a = [r,K] (:350), lora_b = [N,r] (:354),
  scaling = alpha/rank.  => col layers shard lora_b on dim0(N); row layers shard lora_a
  on dim1(K); r never splits.
  fwd-S is CPU-LEFT today: _cpu_left_lora_a (dense_mlp_finegrained.py:790) ->
    grouped_lora_a_forward_cpu_left (exp_act_offload_lora.py:105).
  dA is CPU-RIGHT: grouped_lora_a_grad_cpu_right(dS[M,r] CUDA contig, source_cpu[M,K] pinned
    CONTIGUOUS 2D, offsets, experts, *, num_experts, stats, tag) -> [E,r,K]
    (exp_act_offload_lora.py:231; E = num_experts kwarg).

Activation offload (activation_offload.py):
  global CPU pool _CPU_BUFFER_POOL keyed (dtype,shape,pinned), cap ASYM_EXPACT_CPU_POOL_MAX_BYTES
    default 32 GiB (:10-27) — NO device tag (collides across GPUs).
  offload() D2H records an Event on current_stream(src) keyed by host data_ptr (:194-197,
    no device tag); stage()/H2D paths record NO events. wait_cpu_ready() = host
    event.synchronize() (:230-235) — the blocking drain to convert to stream-ordered.

Weight offload (weight_offload.py): slab pool keyed total_numel (:89, :199-206); gather_group
  = one H2D/layer into a flat slab, repoints param.data views (:208-221); release() (:226-242)
  is driven ONLY by the CPUAdamW post-accumulate hook (cpu_adam.py:395) — a plain-tensor
  mirror gets no auto-release.

HostWeight (host_weight.py:178): the one pinned tensor is self._tensor (:227); .to(cuda)/cuda()
  refuse migration (:317-335). No repack method; a shard view is a plain slice of hw.weight.

CPUAdamW (cpu_adam.py): grad D2H hooks _copy_or_accumulate_grad_to_cpu, non_blocking=False
  (:397-412, +:478); step (:435-514) updates fp32 masters, refresh_home_from_master for
  offloaded banks.

MoE:
  qwen3 fine-grained: _ensure_qwen3_moe_finegrained_bases (qwen3_moe.py:2509-2543) LAZILY
    splits the fused [E,2I,H] gate_up bank into two .contiguous() [E,I,H] pinned COPIES and
    NEVER frees the fused bank -> ~2x pinned gate/up bytes per layer (per device if cloned).
    Router DETACHED (:2569, :3084-3086, :3103-3104). Entry _forward_qwen3_moe_finegrained_offload
    (:2556) already receives (offsets, experts, token_indices, routing_weights).
  llama4: fused [E,H,2I] in_out; splits gate/up on the ACTIVATION via chunk(2,dim=-1)
    (llama4_experts.py:169,:283) — NO weight duplication. Router detached (:287-289, :297-298);
    defeatable only via router_debug_grad=True or router_mode="hf".
  expert X = per-device PACKED gather _rebuild_packed_x_cpu (llama4_experts.py:117,
    index_select) — LOCAL, non-replicated.
  grouped kernel m_grouped_bf16_asym_gemm_nt_contiguous(a, b=[G,N,K], d, offsets_i32[2G],
    experts_i32[G+1 with -1 sentinel], list_size, ...) (gemm.hpp:506; launched frozen_linear.py:753).
    num_groups = b.shape[0]; experts are 0-based LOCAL ids -> EP-2 needs NO kernel change,
    just a per-device bank [E/2,N,K] + local ids/offsets.
  MoE primitives (moe.py): pack_tokens_contiguous (:758), scatter_contiguous (:804),
    _ScatterContiguousRouterNoGrad (:771). Insert DispatchFn at pack, AllReduce2Fn at scatter,
    in the PRODUCTION paths (qwen3_moe.py:2800/3097, llama4_moe.py:292) — NOT AsymMoELayer.
  shared expert (scout only) added AFTER the routed combine (llama4_moe.py:303).
  validate_group_plan does a blocking D2H drain offsets/experts.to(CPU) on every grouped call
    (csrc/exp_act_offload/exp_act_offload_kernels.cu:89-90; called :365/:437/:512).

JIT (BLOCKER for 2 devices): KernelRuntimeCache keyed by dir_path ONLY, no device, no lock,
  process-global (cache.hpp:16-33); default build compiles the driver-API cuModuleLoad path
  (handle.hpp:113-125) whose CUfunction is bound to the context current at first build.
  DG_JIT_USE_RUNTIME_API guard exists (handle.hpp:52) but is NOT set in setup.py. static
  LaunchAttrHandle (handle.hpp:156) is thread-shared under cluster launch. device_runtime
  caches the first device's cudaDeviceProp (:18,58-67).

Kernel alignment for a shard: COL split (out_features N) shard N_i must be %64 (the dX
  transpose contraction dim); it is a contiguous dim0 slice (zero-copy). ROW split (in_features
  K) shard K_i must be %8 but a [:, :K/2] view is NON-contiguous -> must repack into a
  contiguous pinned buffer. block_k drops 256->64 when the transpose-path contraction dim
  K<768 (sm100_bf16_asym_gemm.hpp:298) — a per-shape fact, not a lane fault. Strided B is
  accepted (gemm.hpp:550-552) if the byte row-stride stays 16B-aligned (stride%8).

Harness:
  backend_gpu_count() forces asym*->1 GPU (profile_lora_lf_test_source.sh:903) — THE cap that
    collapses asym|2 to one device. is_torch_run = (BACKEND=="torch") (run_lf_lora_sft.sh:704)
    so asym stays SINGLE-PROCESS. BACKEND case :295-415 (asym_cpuadamwds -> BACKEND=asym :397;
    unknown -> exit 2 :414). GPU_ID is an unvalidated scalar that CAN carry "0,1"; it drives
    CUDA_VISIBLE_DEVICES=${GPU_ID} (:2302). PROFILE_GLOBAL_BATCH_SIZE = per_device*GA*NUM_GPUS
    (:1163). numactl node-level, membind 0,1 (:2512-2532). host-mem watchdog floor 35 GiB,
    sentinel ${LOG_FILE}.host_mem_watchdog_fired (:1610-1681). config_label embeds
    gpus${model_spec_count} (:1885). HF Trainer is NOT patched for DataParallel (heartbeat
    patch is instrumentation only, run_lf_profiled_train.py:1243-1533); per-device memory
    emission is deviceless/dev0-only (:2542-2564).
  lf.py: device resolver _module_device_dtype (:1143-1150), override at the call site :1185;
    wrap loop apply_lf_asym_lora (:1709-2426), Linear-leaf walk :2256-2332, kind classifier
    classify_lf_component (:528-610); repack hook = adopt_host_weight (offload.py:176-213);
    model-state via setattr(model, ...) ~:2418; env via dual-name _env_true idiom (:409).
    NO ASYM_STP / record_stream / wait_stream / with-stream anywhere yet (all net-new).

Cross-device CUDA facts (verified against vendored sources under third_party/):
  A cross-device x1.to(dev0, non_blocking=True) enqueues on the SOURCE device's current stream
    (ATen Copy.cu). Peer access is enabled lazily on the first cross-device COPY, not on
    allocation. Symmetric all-reduce: each direction on its source's copy stream; events order
    the local add after BOTH copy-dones (write-after-read hazard); record_stream the dst temp.
  Megatron duality (megatron-lm/.../tensor_parallel): col f-operator fwd=identity /
    bwd=all-reduce (mappings.py:197-214); row g-operator fwd=all-reduce / bwd=identity (:217);
    LinearWithFrozenWeight saves ONLY weight, dgrad, NO wgrad (layers.py:350-382); bwd order
    dgrad->async-AR->wgrad->wait (:544-653); RowParallel bias AFTER reduce (:1350);
    CUDA_DEVICE_MAX_CONNECTIONS=1 is perf-ordering, NOT correctness (docstring :685-692).
  Automodel LoRA (Automodel/.../_peft/lora.py) SHARDS the adapter (Shard(0) col +all-gather S,
    Shard(1) row) and uses "scale before B, both partial, ONE reduce" (:297-313). Our
    replicate-r layout is a DELIBERATE deviation (below).
```

## Contribution -> Evidence Map (each stage must produce the evidence for its contributions)

```text
C1 disjoint-lane streaming: asym_stp vs asym_dp2; per-lane weight bytes ~0.5x, step_s
   ~0.5-0.6x at equal global workload.                                      [I3, I6]
C2 zero-residency shards: step_H resident > staged > streamed; seq frontier
   asym_stp >= 1.8x tp2_resident (honest: resident WINS step_s where it fits).  [I6]
C3 shared-arena dedup: arena=1 vs 0; residual host bytes + D2H ~0.5x.        [I5]
C4 tile-wise act consumption: asym_stp vs tp2_offstage boundary (2-3x).      [I5, I6, I8]
C5 coordination: coord=1 vs 0; 10-30% step_s or DROP the claim.             [I5]
C6 scaling: {|1 b8} vs {asym_dp2 b4x2} vs {asym_stp b8}; dp2 <=1.2x, stp 1.6-2x. [I6]
C7 aux-loss-free MoE (corollary): equal-count token slices -> skew-immune compute -> the MoE
   load-balance aux loss / capacity-factor token-drop are unnecessary for frozen+LoRA; loss/quality
   parity vs the aux-loss baseline, zero dropped tokens. [I7]
```

## Profiling Goals (dev; real models, real workloads)

E2E LoRA profiling is the acceptance bar for every stage that touches training semantics.
Isolated micro-tests are acceptable ONLY for I1 (pure stream/P2P plumbing) and kernel parity
probes; they never accept a stage on their own.

```text
DEV pace cars (ONLY two; everything else deferred to paper phase):
  DP: superoffload_mem|unsloth-off  q3-32b 20000|4|1        (the shipping-SOTA reference)
  TP: FSDP2(CPUOffloadPolicy)+TP-2  q3-32b s20000 b8-global (apples-to-apples; own script)
Paper-phase pace-car crowning (zero3, unsloth variant, per-model winners) = DEFERRED.
```

Dev goals:

```text
P-DEV (THE dev gate — every stage validates on THIS row ONLY; q3-32b 20000|8|1 ker000):
   step_H(stp) < step_H(superoffload-DP2 b4)  AND  step_s(stp) < step_s(FSDP2+TP2 b8)
   AND step_s(stp) < step_s(superoffload-DP2 b4); loss in band; host RSS well under the
   watchdog floor at all times (HC2). P6's mechanism-health checks (both lanes >= 170 GB/s
   in streamed windows, dup_factor 1.0) also evaluate at P-DEV.
PAPER-PHASE acceptance (full matrix; run ONLY after every stage passes P-DEV):
P1 llama3.3-70b 25000|8|1: step_H(stp) < step_H(mem pace car b4) AND
   step_s(stp) < step_s(time pace car b4); loss in band.            [I4 then I5]
P2 q3-32b 50000|8|1: same       P3 q2.5-72b 30000|8|1: same          [I4/I5]
P4 q3-30b-a3b 80000|8|1 ker101: same (after Stage I7).               [I7]
P5 boundary b8 (llama3.3-70b, q3-32b): max seq(stp) >= 1.5x mem pace car. [I5]
P6 mechanism health at P1: both lanes >= 170 GB/s in streamed windows; dup_factor 1.0;
   per-lane weight bytes ~0.5x of asym_dp2.                          [I4/I5]
DECIDABILITY RULE (P1-P4): prior |1 boundaries make b4 completion at these rows non-certain.
   If a pace car OOMs at the target row, the comparison shifts to that car's max-runnable seq
   at b4, and the stp row is reported as beyond-frontier — the P goal stays decidable.
```

P0 (DEV; runnable BEFORE any sTP code; each row separately, pair 0,1) — exactly TWO baseline
rows, nothing else:

```bash
# DEV pace car #1 (DP; owned by gb200_dp.md D0):
RUNS='q3-32b|2 ; superoffload_mem|unsloth-off|ligerloss1 ; 20000|4|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_gb200tp_p0 MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0,1 --overwrite false
# DEV pace car #2 (TP): scripts/testing/fsdp2_tp_baseline.py — q3-32b s20000 b8 global,
#   torchrun 2-proc, profile.json metrics shim (same reporting format).
# PAPER PHASE (deferred): multi-model pace-car sweep, zero3 rows, unsloth variant, b4
#   boundary probes, b1 frontiers.
```

## Baselines (rules unchanged; full ladder is one knob apart)

```text
Tier 1 (run-as-is): superoffload_mem|unsloth[-off] b4, zero3_offload_mem|unsloth-off b4.
  Ulysses = LF-SP if this venv supports it else cited (DECIDED: no in-stack asym-SP —
  SP==DP on the weight axis; at b=1 TP halves the working set the same as SP).
Tier 2 (stock-API TP, external, Stage I8): both FW rows via vendored NeMo Automodel
  (torchrun, 2 procs) as pure-YAML configs — Automodel ships its own DTensor-aware LoRA, so
  peft-on-DTensor risk is gone:
    FW1 TP-Resident = distributed:{tp_size:2, dp_size:none} + peft block.
    FW2 TP-Staged   = FW1 + fsdp2.offload_policy: CPUOffloadPolicy.
  Metrics shim: profile.json emitter around their recipe loop. BRIDGE RULE: official baseline
  number = max(our in-codebase rung, FW row). Megatron-Bridge: 2 rows (fits-throughput + OOM
  boundary); raw Megatron-LM excluded (no LoRA).
Tier 3 (in-codebase ladder, ours; never presented as prior work): tp2_resident / tp2_offstage
  / asym_stp — one WEIGHT_MODE knob apart — plus asym_dp2 (two |1 asym jobs, attribution row).
Fairness: TP rows b8 (=global 8); DP rows b4/GPU; pair 0,1 only (0,2 for the contention study);
  loss within ~0.05; fresh artifacts; KV heads %2==0 verified per model (llama3.3-70b/q2.5-72b
  kv8->4, q3-32b kv8->4, q3-30b-a3b/scout kv... 4->2 / 8->4 — all even).
```

## Verified Borrow List (2026-07-04 audit; sources vendored under `third_party/`, re-verified)

```text
IMPORT AS-IS (pure functions, no dist): our DENSE modules are UNFUSED (separate q/k/v, gate,
  up) so plain dim0/dim1 chunks at head-group boundaries suffice; no fused-QKV split is ever
  performed. Megatron-Bridge param_mapping split/merge_qkv (GQA interleave) is reference only.
  MCore tensor_parallel/utils split fns; moe_utils permute/unpermute/sort_chunks.
SEMANTICS TO COPY (wiring yes; transport swapped to our P2P exchange):
  MCore f/g duality (mappings.py:197-233): col fwd=identity/bwd=all-reduce (our TPRegionFn),
    row fwd=all-reduce/bwd=identity. => per dense block exactly 2 O(M*H) exchanges fwd AND 2
    bwd, by construction, even with unfused q/k/v/gate/up.
  LinearWithFrozenWeight (layers.py:350-382): frozen base saves ONLY weight, dgrad + one sync
    all-reduce, NO wgrad — our streamed base's backward contract. Its "async==sync" note holds
    only with no wgrad to overlap; WE have dA/dB, so we DO overlap (BWD SCHEDULING RULE, I4).
  Backward order (layers.py:544-653): dgrad -> schedule collective async -> wgrad -> wait.
    RowParallel bias added AFTER reduce (:1350).
  Automodel row-LoRA "partial trick" (lora.py:297-313): scale BEFORE B, keep base and LoRA
    outputs BOTH partial, add, then ONE exchange. We copy this for row layers.
  MoE AllGather dispatcher MINUS its all-gather (residual already replicated) = our I7 no-a2a EP-2.
DELIBERATE DEVIATIONS (verified correct for tp2, r<=64; do NOT "fix" toward SPMD shapes):
  Replicate-r LoRA (NOT the refs' r-sharding): A [r,H] replicated on both devices. Costs r*H
    replicated bytes; REMOVES the refs' extra tiny fwd all-gather on S. Strictly better at tp2.
  One TPRegionFn per region (Megatron's f generalized to unfused q/k/v/gate/up): fused-QKV comm
    cost without weight fusion.
  NO sequence parallelism: at tp2, AR bytes == AG+RS (no comm saving); SP would shard the
    residual and kill the x0==x1 bit-identity that I5 dedup / I7 identical-topk / DEBUG_HASH need.
  dev0 embedding lookup + Bcast01Fn (one P2P copy) beats VocabParallelEmbedding at tp2; all five
    targets untie embeddings.
  Dropout BAN (replaces MCore RNG tracker): every dropout site here is replicated, so a revival
    needs IDENTICAL per-device seeds + a DEBUG_HASH mask assert — the harness default is 0.00.
DO NOT REUSE: all three stacks are multi-process SPMD (one GPU/process, NCCL groups); no
  single-process dual-device path and no weight streaming exists anywhere (Bridge
  cpu_offloading_weights = TE bulk per-layer swap, PP=1 only; Automodel = FSDP2 whole-unit
  offload) — these are the paper's novelty receipts.
```

## Global Efficiency Rules (apply to EVERY stage; a violation is a design bug)

```text
E1 NEVER split M (tokens) on the base/weight-streamed GEMMs. Their B traffic is M-independent
   (TMA loop: one B tile per (n,k) block), so M-split buys no bandwidth. Shards are N/2 or K/2
   — still huge single GEMMs. (EXCEPTION, scoped: the dA grad kernel CONTRACTS over M and
   streams X from host, so M-split IS the correct axis for it — see I5.)
E2 MoE: ONE grouped kernel per device over its E/2 experts. No per-expert Python loops, no
   per-expert launches beyond what the |1 grouped path already does.
E3 One O(M*H) collective per TP region: attention -> 1 AllReduce2Fn fwd + 1 TPRegionFn bwd;
   MLP -> same. EXEMPT: tiny O(M*r)/O(H*r) LoRA-grad exchanges (dS_full per col adapter, row dB),
   budget <=7/layer bwd unfused (q,k,v,gate,up dS + o,down dB); OPTIONAL batching (concat qkv dS
   -> [M,3r], gate/up -> [M,2r]) reduces to 4. allreduce2 = 1 P2P copy each direction on copy
   streams + local add. No NCCL groups, no barriers.
E4 Launch pattern per op: enqueue dev0 kernel, enqueue dev1 kernel, back-to-back, both async;
   cross-device ordering via events ONLY at collective points. No cudaDeviceSynchronize in
   steady state; no new .item()/host reads (the only host reads stay the existing MoE token counts).
E5 Weight repack happens ONCE at load (weights frozen). Zero runtime relayout.
E6 Backward concurrency is free: torch autograd runs one worker thread per device, so dev0/dev1
   backward nodes overlap without extra code — for exchange-FREE nodes (norms, rotary, silu).
```

## Evidence Discipline

One experiment at a time (exception: the two ranks inside one asym_dp2 row). New `OUTPUT_ROOT`
per stage; artifacts never overwritten. Before each run write the expected {model, pair,
backend, WEIGHT_MODE, arena, coord, per-device+global batch, artifact tag, comparison row,
likely failure}. After: `command.txt` (all `ASYM_STP_*` echoed), `train.log`,
`profile.json.config`, per-device `step_H`, loss band, `numa_maps`, and — once their emitting
stage lands — `lane_bw.json` (I1) and `arena_breakdown` dup_factor (I5); earlier runs (P0, dp2)
are judged without them. Labels: `validated | blocked_by_stage_bug | inconclusive_wrong_config
| inconclusive_partial_profile | inconclusive_stale_artifact | inconclusive_unexpected_path`.
Never advance on inconclusive.

## Stage I0 — Harness Plumbing + P0 Baselines

**Objective.** Make `|2` sTP runs launchable, single-process, correctly batched, and auditable
BEFORE any sTP math exists. Land the P0 pace cars and the `asym_dp2` attribution row (zero code
dependencies). No training-semantics change.

**Files & functions (verified anchors):**

```text
scripts/lf/profile_lora_lf_test_source.sh:899-908  backend_gpu_count() — THE cap. Add
    asym_stp*/tp2_* -> 2; make asym*/kt_* -> 1 unchanged. This is what un-collapses |2.
scripts/lf/profile_lora_lf_test_source.sh:887-897   gpu_slice() already returns "0,1" for
    count=2 from a >=2 pool — no change, just feed it count 2.
scripts/lf/profile_lora_lf_test_source.sh (backend->env derivation, NEW helper mode_for_backend)
scripts/lf/run_lf_lora_sft.sh:295-415   BACKEND case: add asym_stp[_cpuadamwds],
    tp2_resident_cpuadamwds, tp2_offstage_cpuadamwds -> BACKEND=asym (keeps is_torch_run false
    => single process); unknown still exits 2 (:414).
scripts/lf/run_lf_lora_sft.sh:56,460-463  NUM_GPUS gate: asym_stp*/tp2_* REQUIRE NUM_GPUS=2.
scripts/lf/run_lf_lora_sft.sh:1163        global-batch override (below).
scripts/lf/run_lf_lora_sft.sh:2512-2532   keep BOTH-node membind under sTP (HC1); assert the HC2
    guards (TRAIN_OOM_SCORE_ADJ=1000, HOST_MEM_WATCHDOG=true) are set or die (below).
scripts/lf/run_lf_lora_sft.sh (env echo)  echo ASYM_STP, ASYM_STP_TP_SIZE, ASYM_STP_WEIGHT_MODE,
    ASYM_STP_SHARED_ARENA, ASYM_STP_COORD + the watchdog knobs (HOST_MEM_WATCHDOG*,
    TRAIN_OOM_SCORE_ADJ) into command.txt + profile.json.config.
scripts/lf/run_lf_profiled_train.py:1243-1533  Trainer.__init__ monkey-patch (DataParallel kill).
scripts/lf/run_lf_profiled_train.py:2542-2564   per-DEVICE memory emission (loop visible devices).
scripts/lf/run_dp2_pair.sh   NEW (asym_dp2 attribution row).
config_label (profile_lora_lf_test_source.sh:1885) already embeds gpus${model_spec_count} ->
    tags gpus2; append stpW<mode>_arena<0|1>_coord<0|1>_tp2 so every ablation gets its own dir.
```

**Implementation.**

1) *Un-collapse `|2` at the cap.* `backend_gpu_count()` currently forces every `asym*` to 1:

```bash
# profile_lora_lf_test_source.sh:899-908 — add the sTP/ladder arm BEFORE the asym arm:
case "${backend}" in
  asym_stp|asym_stp_cpuadamwds|tp2_resident_cpuadamwds|tp2_offstage_cpuadamwds)
      printf '2\n' ;;                                  # <- honor |2 (was silently 1)
  asym|asym_torch|asym_cpuadamwtorch|asym_cpuadamwds)  printf '1\n' ;;   # unchanged
  torch|zero2|...|superoffload_mem_panvme) printf '%s\n' "${model_gpu_count}" ;;
  kt_torchbf16|kt_armbf16) printf '1\n' ;;
esac
```

2) *Backend -> env derivation (self-contained; exists in BOTH the profile script for the dry-run
tag and run_lf_lora_sft.sh for direct launches).* The harness DERIVES `ASYM_STP=1`,
`ASYM_STP_TP_SIZE=2`, and defaults per rung. PER-KNOB override semantics — a blanket
must-match would make the I5 ablations unlaunchable:

```bash
mode_for_backend() { case "$1" in
  asym_stp*)        echo stream ;;   tp2_resident*) echo resident ;;
  tp2_offstage*)    echo stage  ;;   *) echo none ;; esac; }
# defaults: WEIGHT_MODE derived (explicit must MATCH or die). SHARED_ARENA/COORD are DEFAULTS:
#   asym_stp*     -> arena 1 / coord 1  (overridable — that IS the I5 ablation mechanism, BUT
#                    explicit overrides DIE until the I5 code lands, else a mislabeled artifact)
#   tp2_*         -> arena 0 / coord 0  PINNED (explicit override always dies; rung purity —
#                    arena/coord CLAIMS are ablated on asym_stp ONLY)
```

```bash
case "${BACKEND}" in
  asym_stp*|tp2_*)
    [[ "${NUM_GPUS}" == 2 ]] || die "${BACKEND} requires NUM_GPUS=2"
    ASYM_STP=1; ASYM_STP_TP_SIZE=2
    ASYM_STP_WEIGHT_MODE="${ASYM_STP_WEIGHT_MODE:-$(mode_for_backend "${BACKEND}")}"
    [[ "${ASYM_STP_WEIGHT_MODE}" == "$(mode_for_backend "${BACKEND}")" ]] || die "WEIGHT_MODE mismatch"
    ;;                       # is_torch_run stays false (BACKEND=asym) => single process
esac
```

3) *TP-mode global batch.* At `run_lf_lora_sft.sh:1163` the reported global batch multiplies by
`NUM_GPUS`; sTP is model-parallel (both GPUs process the SAME batch), so:

```bash
if [[ "${ASYM_STP:-0}" == 1 ]]; then
  PROFILE_GLOBAL_BATCH_SIZE=$(( PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS ))  # NOT *NUM_GPUS
fi
```

4) *Kill HF DataParallel (RESOLVED design — no existing guard).* With two visible GPUs on the
single-process path, HF Trainer sets `_n_gpu=2` and (a) wraps the model in `nn.DataParallel`
inside `_wrap_model` (would run our surgery twice/step) and (b) freezes
`_train_batch_size = per_device * n_gpu = 16` in `__init__` BEFORE the dataloader is built.
`_setup_devices` is a `@cached_property`, so a write to `_n_gpu` AFTER first access sticks;
therefore patch at `Trainer.__init__`, not `_wrap_model` (too late — batch already 16). Verified
against vendored transformers 5.8.0.dev0 (`trainer.py:591` freeze, `:2419` DP wrap; HF itself
does the same `_n_gpu=1` trick but only `if is_model_parallel`, which won't fire for us):

```python
# inside _install_trainer_heartbeat_hooks (run_lf_profiled_train.py:1243-1533):
if os.environ.get("ASYM_STP") == "1":
    _orig_init = Trainer.__init__
    def _init(self, *a, **k):
        args = k.get("args") or a[1]           # __init__(self, model=None, args=None, ...)
        _ = args.n_gpu                          # force the cached _setup_devices FIRST (sets _n_gpu=2)
        args._n_gpu = 1                         # sticks; runs BEFORE the :591 _train_batch_size freeze
        _orig_init(self, *a, **k)
        assert self._train_batch_size == args.per_device_train_batch_size  # == 8, not 16
    Trainer.__init__ = _init
    _orig_wrap = Trainer._wrap_model            # belt: DP wrap must never appear
    def _wrap_model(self, model, *a, **k):
        w = _orig_wrap(self, model, *a, **k)
        assert not isinstance(w, torch.nn.DataParallel), "stp: DP wrap leaked"
        return w
    Trainer._wrap_model = _wrap_model
```

5) *Per-device memory emission (else a dev1 peak blowup is invisible to every gate).* The step
sampler is deviceless (dev0 only). Loop visible devices:

```python
# run_lf_profiled_train.py:2542-2564 — replace the deviceless calls with a per-device loop:
for d in range(torch.cuda.device_count()):
    peak[d] = int(torch.cuda.max_memory_allocated(d)); torch.cuda.reset_peak_memory_stats(d)
# emit step_H per device into profile.json + heartbeat (consumed by I4 step_H<=0.55x, P1/P6,
# and the failure-decomposition step_H(g0/g1)).
```

6) *NUMA — keep the DEFAULT both-node membind (HC1); do NOT restrict to one node (large rows exceed
one node's ~480 GB and would OOM):*

```bash
: "${NUMACTL_MEMBIND:=0,1}"; : "${NUMACTL_CPUNODEBIND:=0,1}"   # HC1: BOTH nodes for the host mem pool
# arena-locality (pair-local first-touch of the HOT weight arena) is the coord knob's job (I5),
# NOT a hard membind restriction. Also assert the HC2 guards are set: TRAIN_OOM_SCORE_ADJ=1000 and
# HOST_MEM_WATCHDOG=true (die if unset under ASYM_STP=1 — these must never be silently off).
```

7) *`run_dp2_pair.sh` (attribution row, runnable before any sTP code).* Launch two `|1` asym jobs
(GPU0/GPU1, b4, same seed/dataset), wait both, emit `dp2_merged.json` {per-rank profile paths,
summed RSS, max wall-clock, per-rank watchdog-fired state}. This is the `asym_dp2` row AND the
shared-Grace contention evidence: each rank hosts ~2x weights over the SAME node under the SAME
35 GB floor, so a soft host OOM is itself dp2-vs-sTP evidence.

**Validation gate.**

```bash
# positive dry run (DRY_RUN prints the command + writes command.txt; never launches training):
RUNS='q3-32b|2 ; asym_stp_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 128|8|1 ; none|false|false|false|false|false' \
DRY_RUN=true PREPARE_DATASETS=false PLOT=false RUN_POST=false \
OUTPUT_ROOT=profiling_gb200tp_dryrun RUNS_LOG=profiling_gb200tp_dryrun/runs.log \
GPU_POOL=0,1 PROFILERS=source MAX_STEPS=1 WARMUP_STEPS=1 \
bash scripts/lf/profile_lora_lf_test_source.sh
```

```text
PASS:
  command.txt shows GPU_ID=0,1  NUM_GPUS=2  ASYM_STP=1 ASYM_STP_TP_SIZE=2
    ASYM_STP_WEIGHT_MODE=stream ASYM_STP_SHARED_ARENA=1 ASYM_STP_COORD=1, single-process launch
    (NO torchrun), and the DERIVED PROFILE_GLOBAL_BATCH_SIZE=8 (NOT 16).
  artifact path contains __gpus2__ ... stpWstream_arena1_coord1_tp2.
  NEGATIVE guards die: asym_stp at |1; asym_cpuadamwds at |2; WEIGHT_MODE mismatch;
    --gpus 0,2 without ALLOW_CROSS_SUPERCHIP=1; ASYM_STP=1 with LORA_DROPOUT!=0 (breaks the
    x0==x1 invariant I5/I7 depend on); ASYM_STP=1 on llama4 with router_mode!="whole" or
    router_debug_grad set; ASYM_STP=1 on ANY MoE model until I7 (gate on ASYM_STP_MOE, which I7
    introduces); tp2_* with an explicit COORD/ARENA; asym_stp_cpuadamwds with ARENA=0 pre-I5.
  P0 sweep completes; |2 mem/time pace-car winners recorded above; asym_dp2 dp2_merged.json sane.
```

```text
NOTE — the global-batch RECEIPT is NOT "profile.json global batch == 8" (the Python launcher
never reads GLOBAL_BATCH into profile.json; it emits only per_device_train_batch_size). The two
real receipts are (a) command.txt PROFILE_GLOBAL_BATCH_SIZE=8 here, and (b) trainer._train_batch_size
== 8 emitted via the heartbeat at step 1 — verified by the I3 s2048 run (loss band alone cannot
catch a silently-doubled consumed batch).
```

**Risks / watch:** (a) DRY_RUN cannot exercise the `:1163` runtime override (it never runs
`run_lf_lora_sft.sh`) — the derived value must ALSO be printed in the dry-run echo, and the real
override is proven by the I3 s2048 profile; (b) dataset registration loss on new roots — verify
`dataset_info.json` rows exist before runs; (c) accelerate may have its own device-placement pass
— the `_wrap_model` assert covers it; (d) the artifact tag says `gpus2` even for `tp2_*` — correct
now that the cap honors 2.

## Stage I1 — STPRuntime + JIT Per-Device Fix (single-process pair; isolated validation allowed)

**Objective.** One process owns both GPUs: streams, P2P exchanges, NUMA binding, the out-slab
pool, and — critically — the fix that lets both devices JIT-launch the same GEMM signature. This
is pure runtime plumbing; isolated micro-validation is allowed (no training semantics).

**Files & functions:**

```text
asym_gemm/training/stp_runtime.py   NEW: class STPRuntime (env singleton).
asym_gemm/integrations/lf.py:1185   device resolution: when ASYM_STP=1, pin the wrap-leaf device
    to rt.primary (dev0) at the _wrap_lf_linear_leaf call site (NOT only the :1149 fallback).
csrc/jit + setup.py                 JIT per-device fix (BLOCKER — see below).
scripts/testing/stp_runtime_probe.py  NEW probe (topology + lanes + allreduce2 + a dev1 GEMM).
scripts/lf/extract_lane_bw.py       NEW tooling (lands HERE; I3+ gates consume it): parse
    nsys export -> lane_bw.json {per-GPU h2d/d2h GB/s p50/p95 + bytes, nvlink tx/rx, windows =
    streamed-GEMM NVTX spans, PLUS a host-DRAM BW column (2x ~190 GB/s streamed reads + D2H act
    + CPUAdam + dataloader all draw on ONE ~500 GB/s Grace controller — the true shared ceiling).}
```

**Implementation — the JIT blocker (fix FIRST; it gates every dev1 GEMM).** The kernel cache is
keyed by `dir_path` only (no device), and the default build compiles the driver-API
`cuModuleLoad` path whose `CUfunction` is bound to the context current at first build. Two GB200s
have identical arch `sm_100a` -> identical `dir_path` -> dev1 reuses dev0's `CUfunction` ->
`CUDA_ERROR_INVALID_HANDLE` at the first dev1 base GEMM. Pick ONE fix:

```text
FIX A (preferred, one line): enable the context-INDEPENDENT runtime-API path. Add
  -DDG_JIT_USE_RUNTIME_API to setup.py cxx_flags + nvcc args (:34-63). This compiles
  handle.hpp:52-72 (cudaLibraryLoadFromFile + cudaLibraryGetKernel -> cudaKernel_t), which the
  CUDA runtime materializes per-device on demand. Requires CUDART>=12.8 (GB200 toolchains satisfy).
FIX B (if A regresses |1): key the cache by (dir_path, device) + a std::mutex, wrap
  KernelRuntime construction and launch in c10::cuda::CUDAGuard(device), and pre-warm BOTH
  devices before any autograd worker thread splits (E6).
EITHER WAY: pre-warm both devices at STPRuntime init (one tiny asym GEMM per device) so the first
  real launch never races. WATCH: the static LaunchAttrHandle (handle.hpp:156) is thread-shared —
  only a hazard if both devices are driven from two host threads AND use cluster launch
  (num_multicast>1); our GEMMs do not cluster-launch, but assert num_multicast==1 under sTP.
```

**Implementation — STPRuntime.** The base-weight stream uses ZERO copy engines (in-kernel TMA on
the compute stream). Copy engines serve only D2H act offload, H2D restage, and P2P exchanges — so
give P2P its own stream (do not serialize NVLink P2P behind C2C copies):

```python
class STPRuntime:                                  # singleton, built when ASYM_STP=1
    def __init__(self, dev_ids=(0, 1)):
        self.d = [torch.device("cuda", i) for i in dev_ids]
        self.primary = self.d[0]
        for a, b in ((0, 1), (1, 0)):
            assert torch.cuda.can_device_access_peer(dev_ids[a], dev_ids[b])
        # ENABLE peer access NOW via a COPY (an alloc does NOT enable it — verified: Copy.cu):
        torch.zeros(1, device=self.d[0]).to(self.d[1]); torch.zeros(1, device=self.d[1]).to(self.d[0])
        self.compute = [torch.cuda.Stream(x) for x in self.d]   # weight-TMA GEMMs live here
        self.d2h     = [torch.cuda.Stream(x) for x in self.d]   # act offload  (C2C)
        self.h2d     = [torch.cuda.Stream(x) for x in self.d]   # restage/gather (C2C, opp. dir)
        self.p2p     = [torch.cuda.Stream(x) for x in self.d]   # NVLink exchanges
        self._bind_numa()          # HC1: BOTH CPU nodes stay membind (default); optionally
        #                            first-touch the HOT arena pair-local; cap OMP. (coord tunes it.)
        self._jit_prewarm()        # one tiny asym GEMM per device (fixes the JIT blocker path).
        # coord DEFAULTS to 1 from I1 on; the coord=0 path lands in I5. Per HC1 BOTH nodes stay
        # membind — coord only tunes first-touch LOCALITY (pair-local hot arena), never restricts
        # capacity. WATCHDOG NOTE: the 35 GB floor sums MemFree over ALL nodes, so a single-node
        # squeeze can hide below it — sample per-node numastat into lane_bw.

    def allreduce2(self, y0, y1):
        # CONTRACT (verified against ATen Copy.cu):
        #  (1) a cross-device y1.to(dev0) enqueues on the SOURCE device's CURRENT stream -> we must
        #      set the source's copy stream, else it lands on the default stream and serializes.
        #  (2) y_i is READ by its outgoing copy AND WRITTEN by the in-place add -> the add must wait
        #      BOTH copy-done events (write-after-read hazard), not just its incoming one.
        e0 = record(current_stream(y0.device)); e1 = record(current_stream(y1.device))  # producers
        self.p2p[1].wait_event(e1)                                   # src=dev1 will read y1
        with torch.cuda.stream(self.p2p[1]):
            t0 = torch.empty_like(y0); t0.copy_(y1, non_blocking=True)   # dev1->dev0, on p2p[1]
        self.p2p[0].wait_event(e0)                                   # src=dev0 will read y0
        with torch.cuda.stream(self.p2p[0]):
            t1 = torch.empty_like(y1); t1.copy_(y0, non_blocking=True)   # dev0->dev1, on p2p[0]
        c0 = record(self.p2p[1]); c1 = record(self.p2p[0])          # copy-done
        for ev in (c0, c1):
            self.compute[0].wait_event(ev); self.compute[1].wait_event(ev)   # BOTH adds wait BOTH
        with torch.cuda.stream(self.compute[0]): y0.add_(t0); t0.record_stream(self.compute[0])
        with torch.cuda.stream(self.compute[1]): y1.add_(t1); t1.record_stream(self.compute[1])
        # EXIT: ambient stream of each device waits the add, so ordinary consumers (cat, residual
        # add, autograd) are ordered after the exchange without knowing our streams:
        current_stream(y0.device).wait_event(record(self.compute[0]))
        current_stream(y1.device).wait_event(record(self.compute[1]))
        return y0, y1                                # both directions moved concurrently (full-duplex)

    def to0(self, y1):        ...   # dev1->dev0 P2P on p2p[1] (SOURCE side, same Copy.cu rule)
    def bcast01(self, x0):    ...   # dev0->dev1 P2P on p2p[0]
    def to0_sum(self, y0, y1):...   # y0 += to0(y1)   (Phase-A return path)
    def bcast01_from_host(self, h): #... pinned host -> dev1 H2D on h2d[1] (I5 residual restage:
                                    #    both lanes pull the ONE pinned copy concurrently)
```

**STREAM DISCIPLINE (binding for EVERY primitive — stp_base_gemm, to0, bcast01, offloads):** at
entry, the consuming stream waits an event recorded on each operand's PRODUCING stream (inside the
producing with-block if a compute kernel produced it; `current_stream(dev)` if ambient); at exit,
the device's ambient stream waits the result event. There is NO house precedent for this in the
repo (it uses only `Event.record -> event.synchronize()`, host-blocking) — this is net-new, so
review it as such. Violations are silent data races.

**out-slab pool (owned by STPRuntime, keyed (M,N,dtype)):** NEW — the closest existing template
is `weight_offload._pool` (keyed total_numel). LIFETIME RULE: serves ONLY outputs captured by a
saved-tensor/offload reference (qkv/gate-up cat slabs) — TAKEN per forward call, RETURNED when the
reference releases. EXEMPT (use `torch.empty` + `record_stream`): any exchange output NOT so
captured (fwd AllReduce2Fn sums feed only the residual add; all bwd exchange outputs) — pooling
those would have no release trigger and accumulate ~2x[M,H]x80 layers ~525 GB/device in one pass.

**Autograd contract (declared here; the exchange Functions land in I4).** The raw helpers above
are NOT autograd-safe. Every cross-device exchange in the training graph is ONE of five boundary
Functions (mixing regimes computes wrong grads):

```text
AllReduce2Fn : fwd = exchange+sum on both devs ; bwd = identity per dev
Bcast01Fn    : fwd = copy dev0->dev1            ; bwd = pass g0, DROP g1 (g1 == g0 by construction;
               summing double-counts — dev1's share already arrived via the first region bwd)
Join01Fn     : fwd = (x0,x1)->x0                ; bwd = g -> (g, bcast01(g))  [replaces SPMD's
               replicated loss seed — without it dev1's row-shard grads are zero]
TPRegionFn   : fwd = identity on (h0,h1)        ; bwd = allreduce2 of the two ACCUMULATED LOCAL dX
               partials (Megatron's f-operator, placed at each region ENTRY: the norm output
               feeding qkv / gate_up). Col-Fns NEVER exchange internally -> autograd sums
               q+k+v(+LoRA) partials locally, and this ONE Fn does the single exchange => exactly
               2 O(M*H) bwd exchanges per dense block even with 5 unfused col modules.
DispatchFn   : the MoE instance of TPRegionFn (I7).
IN-PLACE RULE: Fn forwards must not do y+=t on Fn INPUTS (version-counter); Fn backwards must not
  mutate incoming grad_outputs -> TPRegionFn/DispatchFn bwd use an OUT-OF-PLACE allreduce2 into
  FRESH torch.empty outputs (+record_stream). (The raw in-place allreduce2 above is for Phase A /
  the probe only.)
```

**Validation (isolated OK — no training semantics):**

```bash
python scripts/testing/stp_runtime_probe.py --pair 0,1   # topology + lanes + allreduce2 + dev1 GEMM
```

```text
PASS:
  can_device_access_peer both ways; P2P copy >= 700 GB/s/dir sustained, both dirs concurrently.
  both C2C lanes ~190 GB/s H2D concurrently reading the SAME pinned buffer (proves shared-arena
    legality + delivery).
  allreduce2 of [200000, 8192] bf16 (3.3 GB) < 8 ms.
  a REAL asym GEMM launches on dev1 (JIT fix works) — NOT INVALID_HANDLE.
  CUDA_DEVICE_MAX_CONNECTIONS A/B (=1 vs unset): measure cross-device kernel overlap + per-kernel
    enqueue cost; PIN the winner and echo it into command.txt from I3 on (expect UNSET wins — it
    is a Megatron perf-ordering hack, not correctness; our exchanges are copy-engine DMA).
  numa_maps shows the probe's TEST pinned allocations on the pair's node; membind + OMP cap applied.
  |1 smoke row with ASYM_STP unset: ZERO regression (also confirms FIX A didn't perturb |1).
```

**Risks / watch:** event/stream deadlock (keep ONE canonical allreduce2, no ad-hoc syncs);
Python per-kernel enqueue cost from one thread for 2 devices — measure in the probe (< 30 us/kernel;
acceptable because our GEMMs are ms-scale, E4); FIX A must be A/B'd against `|1` for any perf drift.

## Stage I2 — Sharded Arena (repack-at-load; contiguity is the law)

**Objective.** One pinned copy of every frozen weight, physically laid out so each device's shard
is a CONTIGUOUS block. Weights are frozen, so repack ONCE at load; zero runtime cost (E5). The
kernel gate `_direct_bf16_reason` requires `b_cpu.is_contiguous()` — we honor it rather than
relax to strided-B (an optional later optimization).

**Files & functions:**

```text
asym_gemm/training/stp_layout.py   NEW:
    plan(model_cfg) -> {module_name: (kind in {"col","row","attn_col","attn_row"}, split_dim)}
    assert_plan_matches_hf(plan): dense/attn vs cfg.base_model_tp_plan; experts+router vs
      cfg.base_model_ep_plan (the tp_plan is UNSATISFIABLE for MoE — qwen3_moe maps experts to
      moe_experts_allreduce = intra-expert width TP, llama4 tp_plan has no experts entry; what
      matches our EP-2 is the ep_plan. Record the deliberate divergence.)
    shard_spec(kind, shape) -> [(dev0, slice), (dev1, slice)]
asym_gemm/training/host_weight.py  NEW method repack_for_stp(kind) + shard_view(dev). No existing
    repack; a shard view today is a plain slice of hw.weight (self._tensor). The .to(cuda)/cuda()
    guards (:317-335) already refuse HBM migration — extend to permit index-1 for dev1 resident
    buffers ONLY under the resident rung (I6).
asym_gemm/integrations/lf.py: adopt_host_weight (offload.py:176-213, called at lf.py:1194) is the
    load-time hook — call repack_for_stp there when ASYM_STP=1.
asym_gemm/training/qwen3_moe.py:2509-2543   arena-awareness fix (below).
```

**Implementation.** Two shard kinds, only one copies:

```python
# COL layers (out_features N split; [N,K], split dim0): shards are ALREADY contiguous dim0 slices
# -> record offsets, NO COPY. N_i MUST be %64 (the dX transpose contraction dim — verified in
# _direct_bf16_reason: transpose_b -> k%64 where k = out_features).
def repack_col(hw):                  # hw.weight : pinned [N, K]
    N0 = ceil64(N // 2); N1 = N - N0            # even for the matrix; pad rule for future models
    return hw.weight[:N0], hw.weight[N0:]       # both contiguous, both pinned, zero copy

# ROW layers (in_features K split; [N,K], split dim1): a [:, :K/2] view is NON-contiguous and
# _direct_bf16_reason rejects it -> allocate ONE pinned buffer and copy each half contiguous.
def repack_row(hw):                  # hw.weight : pinned [N, K]
    h = K // 2                                   # K/2 must be %8 (fwd contraction alignment)
    buf = pin_alloc(N * K)                       # ONE alloc, same total bytes
    b0 = buf[: N*h].view(N, h); b0.copy_(hw.weight[:, :h])
    b1 = buf[N*h :].view(N, h); b1.copy_(hw.weight[:, h:])
    # reassembly check BEFORE freeing the original (nothing to compare against post-load):
    assert allclose(torch.cat([b0, b1], dim=1), hw.weight)   # or record a checksum
    free(hw.weight)                              # repack layer-by-layer, transient < 1 layer
    return b0, b1                                # both contiguous, both pinned, one copy total

# BIAS RULE (q2.5-72b is the ONLY target with qkv bias — modeling_qwen2 hardcodes bias=True on
# q/k/v; NOTHING before the P3 run exercises it): col modules get the [N_i] bias SLICE per device,
# pinned alongside the weight shard (Megatron shards bias with the column dim). All three bias-add
# sites (dense :1322, fg :81-82, attn :609) add a full-[N] bias today and would shape-error on the
# [M, N/2] halves in I4 -> each must add its device's [N_i] slice. Row bias: none of the five
# targets has one; if a future model does, add it ONCE after AllReduce2Fn (never per device).
```

```python
# GROUPED EXPERT BANKS (EP-2 ONLY): per-device bank = contiguous dim0 slice at the E/2 boundary,
# ZERO-COPY. The banks are FUSED (qwen3 [E,2I,H], llama4 [E,H,2I]) and must NEVER be N-bisected
# (that would separate gate from up); EP-2 slices dim0 (experts), so no fused split is performed.
# ARENA-AWARENESS (qwen3-specific): _ensure_qwen3_moe_finegrained_bases lazily materializes SPLIT
# gate/up [E,I,H] COPIES and NEVER frees the fused [E,2I,H] -> ~2x pinned bytes/layer; per-device
# cloning multiplies it. FIX at load: repack the split gate/up banks INTO the arena once (their
# EP-2 dim0 slices stay zero-copy) and make _ensure... return arena views. llama4 splits gate/up
# on the ACTIVATION (chunk(2,-1)) not the weight, so it needs no such fix — but assert its in_out
# [E,H,2I] layout in stp_layout so a future out_in model is caught.
```

**Validation (unit OK — layout only):**

```text
PASS (llama3.3-70b first, all wrapped Linears):
  every shard view is pinned + contiguous; N/2 and K/2 pass the %8 / %64 gates in
    _direct_bf16_reason (checked: 28672/2=14336, 25600/2=12800, 29568/2=14784 are all 64-multiples
    -> even splits work across the whole matrix today; keep the ceil64 pad rule for future models).
  host RSS after load == |1 RSS (ONE copy; repack transient < 1 layer's bytes, freed as you go).
  per-layer reassembly allclose(cat(shards), original) at repack time (before free).
  plan assertion passes vs transformers base_model_tp_plan (dense/attn) and base_model_ep_plan (MoE).
  qwen3 arena-awareness: after a fine-grained forward, pinned gate/up bytes == 1x fused (not 2x),
    dup_factor 1.0 (measured once I5's arena_breakdown lands; until then, an RSS check).
```

**Risks / watch:** split points MUST be 64-aligned (mandatory — the dX transpose gate); verified
for the matrix, keep the pad rule (`N0=ceil64(N/2)`, kernels accept uneven shards). embedding /
lm_head stay UNSHARDED (dev0). The MoE arena-awareness fix must be validated on P4/scout where the
expert banks dominate host RSS.

## Stage I3 — Dense Base GEMMs, Phase A (fan-out only; model stays on dev0)

**Objective.** Streamed base GEMMs (fwd base, bwd dX) execute split across BOTH devices;
everything else (attention math, norms, LoRA, residual, optimizer) stays untouched on dev0. This
is e2e-runnable WITHOUT attention surgery and proves lane pooling. Phase-A broadcasts are a real
cost, accepted only for this stage — they disappear in I4 when the residual is replicated.

**ONE CHOKE POINT (verified).** The primary backend does NOT go through `AsymFrozenLinear.forward`.
Every primary-path base GEMM — MLP gate/up/down (fwd `:220/:236/:274`, dX `:339/:399/:423`) and
attention q/k/v/o (fwd `:599`, dX `:692`) — converges on `asym_bf16_cpu_right_matmul`
(`frozen_linear.py:1140`, single `_dispatch_nt` at `:1185`). So we route THAT one function through
`stp_base_gemm` when `ASYM_STP=1`. (The dense `AsymFrozenLinearFunction` path — `_dispatch_nt` at
`:1310`/`:1356` — is separate, used only by eval/fallback + plain `AsymLoRALinear`; route it too
for completeness, but P1 never executes it.)

**Files & functions:**

```text
asym_gemm/training/stp_runtime.py     NEW stp_base_gemm(...) — the single choke point; I6's
    resident/stage/stream mode dispatch lives INSIDE it.
asym_gemm/training/frozen_linear.py:1140/:1185   asym_bf16_cpu_right_matmul routes to
    stp_base_gemm when ASYM_STP=1 (covers the WHOLE fg + attention primary path). Also route the
    dense AsymFrozenLinearFunction :1310/:1356 for the non-fg path.
asym_gemm/training/cpu_left.py        UNCHANGED in I3 (grouped/expert paths keep the |1
    single-device path until I7; no grouped N/K split exists in this plan).
scripts/testing/stp_gemm_parity_probe.py   NEW (kernel-level parity; declared + used here).
```

**Implementation.** `stp_base_gemm` reads the shard metadata I2 attached to the HostWeight
(`kind in {col,row}`, split point, per-device shard views) and splits the SINGLE logical GEMM into
two back-to-back async device GEMMs + one exchange. It is a RAW op (the existing fg Functions own
autograd); it is a drop-in for `asym_bf16_cpu_right_matmul`, so BOTH fwd (output split) and bwd-dX
(the transpose call) become split with no change to the Functions:

```python
def stp_base_gemm(left, hw, *, phase, transpose_b, mode="stream"):   # I3: mode=="stream", model on dev0
    s0, s1 = hw.shard_view(0), hw.shard_view(1)      # host-pinned CONTIGUOUS shards (I2); N0/K0 = split pt
    G = asym_bf16_cpu_right_matmul                    # (left, right_cpu, *, backend, stats, phase, tag, transpose_b)
    if hw.kind == "col":                             # W=[N,K] split on N (dim0)
        if phase == "forward":                       # y = left @ W^T ; split OUTPUT N, gather
            x1 = rt.bcast01(left)                                       # Phase A: x is on dev0 only
            with stream(rt.compute[0]): y0 = G(left, s0, transpose_b=False)   # [M,N0] @dev0
            with stream(rt.compute[1]): y1 = G(x1,   s1, transpose_b=False)   # [M,N1] @dev1
            return rt.out_slab.cat_(y0, rt.to0(y1))                    # [M,N] @dev0, preallocated slab
        else:                                        # dx = grad @ W ; split grad by output cols, SUM
            g1 = rt.bcast01(left[:, N0:].contiguous())
            with stream(rt.compute[0]): dx0 = G(left[:, :N0].contiguous(), s0, transpose_b=True)  # [M,K]
            with stream(rt.compute[1]): dx1 = G(g1,                        s1, transpose_b=True)
            return rt.to0_sum(dx0, dx1)                                # ONE P2P + add (E3)
    else:                                            # hw.kind=="row": W=[N,K] split on K (dim1)
        if phase == "forward":                       # y = x @ W^T ; split INPUT K, SUM partials
            x1 = rt.bcast01(left[:, K0:].contiguous())
            with stream(rt.compute[0]): y0 = G(left[:, :K0].contiguous(), s0, transpose_b=False)  # [M,N] partial
            with stream(rt.compute[1]): y1 = G(x1,                        s1, transpose_b=False)
            return rt.to0_sum(y0, y1)
        else:                                        # dx = grad @ W ; split W by K -> cat
            g1 = rt.bcast01(left)                                      # grad_y replicated
            with stream(rt.compute[0]): dx0 = G(left, s0, transpose_b=True)   # [M,K0]
            with stream(rt.compute[1]): dx1 = G(g1,   s1, transpose_b=True)   # [M,K1]
            return rt.out_slab.cat_(dx0, rt.to0(dx1))                 # [M,K] @dev0
# EVERY operand hand-off obeys the I1 STREAM DISCIPLINE (bcast/to0 results event-ordered before
# consumption; y1's ready-event recorded ON compute[1] inside its with-block). A literal
# transcription without those waits RACES.
```

**Efficiency notes (E1-E4).** No M split (E1); the two GEMM launches are back-to-back async (E4);
Phase-A broadcast is ~1x[M,K] per call over ~900 GB/s NVLink, accepted ONLY for this stage. LoRA
stays entirely on dev0 in I3.

**Alignment (verified).** The transpose_b dX path requires the shard's contraction dim `%64`; for
a col shard that is `N_i` -> split points MUST be 64-aligned (the I2 pad rule is mandatory, not
defensive). Matrix models checked: 28672/2, 25600/2, 29568/2 are all 64-multiples.

**Validation (kernel parity isolated; acceptance is E2E):**

```bash
python scripts/testing/stp_gemm_parity_probe.py --model llama3.3-70b --cases col,row --mode stream
# e2e loss gate (small):
RUNS='q3-32b|2 ; asym_stp_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_gb200tp_I3_s2048 MAX_STEPS=1 WARMUP_STEPS=0 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0,1 --overwrite false
# e2e target profiling (PROFILERS=both — the lane_bw/nsys gates need the nsys pass; EVERY later
# gate that reads lane_bw/nvlink/class-byte artifacts inherits this convention):
RUNS='q3-32b|2 ; asym_stp_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 20000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_gb200tp_I3_s20000 MAX_STEPS=3 WARMUP_STEPS=1 PROFILERS=both PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0,1 --overwrite false
```

```text
PASS:
  parity MATCH (col, row, uneven shards).
  s2048 loss within ~0.05 of |1 asym at same global workload.
  I0 RECEIPTS land HERE: command.txt PROFILE_GLOBAL_BATCH_SIZE=8 AND train.log step-1 heartbeat
    trainer._train_batch_size == 8 (the Trainer.__init__ patch beat the :591 freeze).
  s20000: nsys/lane_bw shows BOTH lanes active in base-GEMM windows; the base-GEMM component of
    bwd_s ~halves vs the |1 run; step_H(dev0) <= |1 step_H (Phase A adds no residency).
FAIL SIGNATURES: lane1 ~0 -> silent fallback bug -> inconclusive_unexpected_path.
```

**Risks / watch:** `cat([y0, to0(y1)])` allocates — reuse the out-slab pool (I1); bcast01/to0 MUST
ride the p2p copy streams or they serialize compute; the JIT fix (I1) must already be in — the
first dev1 base GEMM is here.

## Stage I4 — Full TP Residency (replicated residual; attention head-split; Phase-A broadcasts deleted)

**Objective.** Both devices hold the residual stream (replicated, Megatron pattern);
norms/rotary/elementwise run redundantly on both (cheap, E4); attention is head-split; MLP col/row
wire to the LOCAL residual copy; ONE `allreduce2` after o_proj and ONE after down_proj per layer;
LoRA layouts final; fwd-S moves CPU->GPU. This is where activation memory ~halves -> the seq
frontier opens, and where P1/P2/P3 are won.

**Per-decoder-layer dataflow (fwd; bwd mirrors with the transposed comm points):**

```text
x0 [M,H]@dev0, x1 [M,H]@dev1 (BIT-IDENTICAL by construction)
attn: n_i = rmsnorm(x_i)                 # both devs, local
      qkv_i = n_i @^ Wqkv_i^T            # col-split by heads (q64->32/32, kv8->4/4), own-lane stream
      a_i = flash_attn(rope(qkv_i))      # full seq, half heads, local
      p_i = a_i @^ Wo_i^T                # row-split, partial [M,H]
      p_i += lora_o_i                    # LoRA partial (row rule: scale-before-B), STAYS partial
      p0,p1 = allreduce2(p0,p1)          # collective #1 (sums base+LoRA together)
      x_i = x_i + p_i                    # residual add, local
mlp:  m_i = rmsnorm(x_i)
      gu_i = m_i @^ Wgateup_i^T          # col-split [M, 2F/2]
      s_i = silu_mul(gu_i)               # local halves (GPU-silu under sTP)
      q_i = s_i @^ Wdown_i^T             # row-split (K=F/2 shard), partial
      q_i += lora_down_i                 # LoRA partial, same row rule
      q0,q1 = allreduce2(q0,q1)          # collective #2
      x_i = x_i + q_i
```

**Boundary-Function placement (ONE regime — mixing computes wrong grads).** Every cross-device
exchange is one of the five I1 Functions OR a tiny O(M*r)/O(H*r) exchange INSIDE a single
two-branch module Function (dS_full, row-dB). Placement:

```text
bottom  Bcast01Fn right after embedding (fwd copy x0->x1; bwd passes g0, DROPS g1 — under this
        regime g1==g0 because the first region-boundary bwd already left the full summed dX on both
        branches; summing here = latent 2x on embedding grads).
top     Join01Fn right before final-norm/lm_head (fwd passes x0, consumes the x1 branch; bwd hands
        the SAME grad to both) — replaces SPMD's per-rank replicated loss. WITHOUT it, dev1's
        row-shard partials get zero gradient and every col allreduce2 sums a real partial with a
        zero -> silently ~half-magnitude grads.
interior AllReduce2Fn after o_proj and after down_proj (fwd sum, bwd identity); TPRegionFn at each
        norm output feeding a col region (attention entry, MLP entry) — col-Fns return LOCAL dX
        partials, the TPRegionFn does the single bwd exchange for base+LoRA of ALL col modules at
        once. lm_head/liger stay dev0-only.
budget  per dense block: fwd = 2 (the two AllReduce2Fn), bwd = 2 (the two TPRegionFn) O(M*H)
        exchanges — holds by construction even with unfused q/k/v/gate/up.
LOSS-SEGMENT MEASURE: dev1 idles through final-norm+lm_head+CE (both refs shard lm_head). Emit the
  loss-segment share of step_s in the I4 nsys pass; adopt a V/2-split lm_head + local max/sum-exp +
  [M]-scalar allreduce2 ONLY if the share exceeds ~5% (it means reworking liger — not the default).
```

**LoRA layout (final; the col-only rule is WRONG for row layers).** Verified `lora_a=[r,K]`,
`lora_b=[N,r]`, `scaling=alpha/rank`:

```text
COL layers (qkv, gate_up) — ONE dataflow:
  fwd: A [r,H] REPLICATED (our deliberate deviation); S = x_i @ A^T identical on both (x replicated);
       B [N_i,r] col-sharded (lora_b dim0); y_i += scale * S @ B_i^T locally (no exchange).
  bwd: dB_i = local ([N_i,r] sharded param -> local grad, no exchange).
       dS_i = g_i @ B_i stays PARTIAL for the dX path: dX_lora_i = dS_i @ A is a LOCAL partial that
         autograd sums into the region's accumulated dX; the region's single TPRegionFn bwd covers
         base+LoRA dX for ALL col modules at once. dS_full must NOT feed dX.
       dA PRIMARY (X offloaded, the I5 kernel path): dS_full = allreduce2(dS_i) (tiny [M,r]);
         per-device grouped_lora_a_grad_cpu_right on its X half; dA = cat(halves). FALLBACK (X not
         offloaded): dA_i = dS_i^T @ x partial -> one tiny allreduce2. NEVER both (double-count).
ROW layers (o, down) — Automodel "partial trick" (verified lora.py:297-313):
  fwd: A [r,K_i] ROW-sharded (lora_a dim1) consumes the local width-shard; apply LoRA scale BEFORE
       B; keep base partial p_i and LoRA partial l_i = B(scale * A_i x_i) BOTH unreduced; the
       block's single allreduce2 sums (p_i + l_i) — one exchange covers base+LoRA, no double-count
       (B(sum_i A_i x_i) == sum_i B(A_i x_i)). B [H,r] REPLICATED.
  bwd: dB = allreduce2 of per-device partials dB_i = g^T (scale * S_i) (tiny [H,r]; B replicated,
       its grad needs the sum); dA_i local ([r,K_i] sharded param); dX_lora_i local (row bwd=identity).
```

**Trainable-state ownership (single source of truth = the EXISTING pinned CPU LoRA slabs).**

```text
nn.Parameters registered ONCE on the dev0 model object (Trainer sees n_gpu=1); dev1 holds
  plain-tensor MIRRORS only.
MIRROR RELEASE RULE (verified: weight_offload.release() is driven ONLY by the CPUAdamW
  post-accumulate hook — a plain tensor gets no such hook): release dev1 mirrors EXPLICITLY at the
  region Function's backward exit, keeping staged high-water ~one layer/device (scout's expert-LoRA
  home is 4.4 GB bf16 — material on the memory scoreboard).
weight_offload.gather_group already re-gathers LoRA banks CPU->GPU per layer per step: extend to
  per-device gathers (dev_i pulls its shards + the replicated pieces over its OWN lane). Staleness
  impossible BY CONSTRUCTION — every step re-gathers from the CPU slab CPUAdamW just updated.
grads: sharded params (col-B_i, row-A_i) -> local grad, D2H over the owning lane; replicated params
  (row-B -> allreduce2 then D2H once from dev0; col-A -> dA halves cat'd on dev0, D2H once). Grad-norm
  sees each logical param exactly once. CPUAdamW unchanged (steps the CPU masters as in |1).
```

**Files & functions:**

```text
asym_gemm/integrations/lf.py:1709-2426   wrap: per-device module CLONES sharing the arena shard
    views (exchange-free ops ONLY — every exchanging op is a single two-branch Function per the I1
    rule); residual duplication at embedding output via Bcast01Fn (the autograd Function, NOT a raw
    .to() — a raw copy leaves a ToCopyBackward edge that SUMS g1, breaking the drop-g1 regime);
    lm_head on dev0 only, consuming x0 via Join01Fn. Hang rt on model via setattr (~:2418).
asym_gemm/training/dense_mlp_finegrained.py + attention_activation_offload.py  THE EXECUTED PATH:
    restructure both fg Functions into two-branch region Functions (per-device silu halves,
    per-device LoRA partials, dS_full/row-dB exchanges inside the Function). fwd-S moves to GPU HERE
    (S = x_i @ A^T from the live replicated x — DROP the cpu-left fwd-S calls at :227/:243; leaving
    them loses dev1's LoRA delta silently). This is I4 scope, NOT I5.
asym_gemm/training/frozen_linear.py       STP fns lose the Phase-A bcast (inputs already local);
    residual_mode selects Phase-A (bcast-in/to0_sum-out) vs I4 (local-in/local-partial-out).
asym_gemm/training/activation_offload.py  PER-DEVICE pools land HERE (moved from I5 — I4's own
    s25000 gates produce dev1 halves ~17+ GB/layer that cannot stay resident, and the existing D2H
    event map + pool key carry NO device tag, so dev1-sourced copies would collide/mis-route). Add
    a device dimension to _CPU_BUFFER_POOL's key and the pending-event map; add H2D-side events
    (stage/restage record none today). Keep the ONE global byte cap for now (the shared token
    bucket lands in I5). Guard capture with is_current_stream_capturing().
scripts/testing/stp_grad_parity_probe.py  NEW + env ASYM_STP_DUMP_GRADS=1: dumps per-adapter grads
    + 1-batch logits AT STEP 2 (PEFT B=0 at init -> dS=0 -> dA identically ZERO at step 1 in BOTH
    |1 and |2, hiding the double-count/placement bugs this gate exists to catch). rel-err uses an
    absolute floor (|a-b| / max(|b|, 1e-8)).
asym_gemm/training/stp_layout.py          head split DERIVED from config (num_attention_heads/2,
    num_key_value_heads/2, assert %2==0); rotary caches PER DEVICE.
```

**BWD SCHEDULING RULE (binding for the restructure; Megatron dgrad->async-comm->wgrad->wait; the
inherited |1 code does the OPPOSITE — dense_mlp_finegrained.py runs dS->dB->dA BEFORE base-dX,
which under sTP queues the dominant dA kernels ahead of the region exchange).** Per region backward:

```text
(a) all dS_i tiny GEMMs + fire the tiny dS allreduce2s immediately;
(b) all base-dX partial GEMMs;
(c) return partials so the region's TPRegionFn O(M*H) exchange launches;
(d) ONLY THEN enqueue dB + the CPU-left dA kernels — leaf wgrads feed nothing upstream; they
    overlap the exchange and the upper layers' bwd (dA needs dS_full, which (a) produced).
```

**DE-SYNC the fg internals (I4 scope — E4 alone is INSUFFICIENT: the |1 machinery host-blocks
inside what becomes the shared autograd thread, so every drain stalls BOTH lanes):**

```text
(a) validate_group_plan does offsets/experts.to(CPU) — a full current-stream drain per grouped call
    (exp_act_offload_kernels.cu:89-90); the dense MLP rebuilds its plan UNCACHED per call
    (_one_expert_plan, dense_mlp_finegrained.py:785-788) = 3 drains/MLP bwd -> cache host-side per
    (device, M) like the attention path (attention_activation_offload.py:96-110).
(b) grad-offload hooks do BLOCKING D2H per LoRA param (cpu_adam.py:397-412, non_blocking=False) ->
    non_blocking=True + ONE event fence before optimizer.step (dest buffers are pinned; the accum
    case grad_buffer.add_(staging) reads immediately -> its own fence before add_).
(c) wait_cpu_ready is a host event.synchronize per handle (activation_offload.py:230-235, ~6-10x/
    layer) -> stream-ordered stream.wait_event wherever the consumer is a GPU kernel.
(d) pin cpu_activation=False under sTP (GPU-silu path exists at dense_mlp_finegrained.py:253-262).
nsys gate: host-sync count per layer ~0 in steady state.
```

**Validation (E2E is the bar):**

```bash
# parity (own declared pair — MAX_STEPS=5, dump on BOTH the |1 reference and the stp run):
RUNS='q3-32b|2 ; asym_stp_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_gb200tp_I4_parity MAX_STEPS=5 WARMUP_STEPS=0 ASYM_STP_DUMP_GRADS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0,1 --overwrite false
# e2e P1 (PROFILERS=both):
RUNS='q3-32b|2 ; asym_stp_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 20000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_gb200tp_I4_s20000 MAX_STEPS=3 WARMUP_STEPS=1 PROFILERS=both PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0,1 --overwrite false
```

```text
PASS:
  PARITY WITH BANDS (exact equality unavailable — fwd-S moved CPU->GPU, reduction orders differ):
    1-batch logits |1 vs stp max-abs-diff <= max(measured envelope, bf16 tol). Derive the envelope
    from a reduction-order-perturbed |1 reference (a different kernel-config run) — a plain
    seed-to-seed |1 envelope is plausibly ZERO (deterministic fixed-order + B=0 at step 1) and would
    fail CORRECT code. Measure once, record in the Decision Log. Logits dump is s2048-ONLY (~4 GB;
    at s25000 ~51 GB) and materializes lm_head explicitly (ligerloss1 never forms logits).
  ADAPTER-GRAD PARITY at STEP 2: per-adapter max-rel-err of dA/dB vs |1 <= 1e-2 (bf16). The loss
    band alone masks a 2x error on a subset of adapters (the exact failure of a wrong exchange
    placement). Step 1 is vacuous for dA (see probe note).
  5-step seeded loss overlay within band.
  e2e P1: step_H per GPU <= 0.55x of |1 asym step_H at s25000 (intermediates halved);
    step_s(stp) < step_s(|2 time pace car b4) [P1 first half]; allreduce2 < 15% of layer time and
    overlapped (nsys); region-exchange copy starts BEFORE the region's dA kernels (BWD SCHEDULING
    receipt); both-lane weight bytes ~W/2 each (lane_bw.json).
  DESIGNATED FIX if the 15% gate fails (predeclared): chunk the row-GEMM OUTPUT along N and fire
    partial-sum copies per finished column block (the single-process analog of Megatron/TE
    tp_comm_overlap; compatible with streamed-weight tile order; E1 forbids splitting M, not
    output-N). Do NOT build preemptively — streamed-GEMM time should dwarf the ~0.4 GB exchange at P1.
  TILE NOTE (not a fault): kv-projection dx shards (contraction k=N_i=512<768) drop transpose
    block_k 256->64 — ~8 MB/GEMM, don't misread those lane_bw windows as a lane fault.
  then P2 (q3-32b 50000|8), P3 (q2.5-72b 30000|8) — same gates. q2.5-72b exercises the col bias slice.
```

**Risks / watch:** (a) per-device autograd overlap — verify in nsys dev0/dev1 backward kernels
overlap; if serialized, a false dependency (usually an accidental same-device intermediate); (b)
rotary/pos-cache device mismatch; (c) Qwen3 q_norm/k_norm operate on head-sharded tensors —
replicate per device, NEVER seq-shard (Automodel `optimized_tp_plans.py:516-518` caveat); (d)
Trainer/accel device moves — the |1 `.to(cuda)` guards must fire (HostWeight already refuses; add
the same for dev1-resident buffers), add a setup counter `stp_dev1_modules_wrapped`; (e)
`CUDA_DEVICE_MAX_CONNECTIONS` pinned by the I1 A/B, echoed into command.txt from I3 on.

## Stage I5 — Activation Path: Shared-Arena Dedup + Row-Split dA + Coordination

**Objective (per-device pools already landed in I4).** Three layout/coordination changes on the
now-working I4 path: (1) residual checkpoints offloaded ONCE (dedup — they are bit-identical across
devices); (2) ROW-split the dA grad (M-split, NOT K-split); (3) the `coord` knob (membind + capped
OMP + GPU-silu + joint prefetch budget). These realize C3, part of C4, and C5.

**Files & functions:**

```text
asym_gemm/training/decoder_activation_offload.py   residual ckpt dedup (below).
asym_gemm/training/exp_act_offload_lora.py:231      dA row-split (grouped_lora_a_grad_cpu_right).
asym_gemm/training/activation_offload.py            residual-dedup path + the shared token bucket.
asym_gemm/integrations/lf.py                         coord knob wiring.
```

**Implementation — residual dedup.** The residual is bit-identical on both devices, so offload it
from ONE device and restage to both from the SAME pinned buffer (both lanes pull concurrently — no
NVLink needed):

```python
def offload_residual(x0, x1):        # bit-identical by construction (assert under DEBUG_HASH)
    return pool[0].offload(x0)        # ONE D2H, dev0 lane; dev1 does NOT offload
def restage_residual(h):
    r0 = pool[0].restage(h, rt.d[0]) # H2D lane0
    r1 = rt.bcast01_from_host(h)      # H2D via lane1 directly from the SAME pinned buffer
    return r0, r1                     # dup_factor accounting -> arena_breakdown.json
```

**Implementation — ROW-split dA (audit-resolved: M-split, not K-split).** `grouped_lora_a_grad_cpu_right`
requires `source_cpu` CONTIGUOUS 2D pinned (verified). A `[:, :K/2]` COLUMN view is non-contiguous
and rejected; a `[:M//2]` ROW view of a contiguous `[M,K]` pinned tensor stays contiguous+pinned
(leading-dim narrow, same K-stride) and is accepted. So split by ROWS. E1 is SCOPED to
base/weight-streamed GEMMs (their B traffic is M-independent); the dA kernel CONTRACTS over M and
streams X from host, so M-split is the CORRECT axis for it — and it is dedup + lane-balanced + zero
compaction kernels (the old K-half design paid a `.contiguous()` + ~1.6 GB transient HBM/layer/dev
for zero bandwidth gain):

```python
# forward: X replicated on both devices (I4); ONE pinned block, unchanged layout; each device
# D2H-writes its ROW half over its own lane (one host copy total = dedup):
with stream(rt.d2h[0]): wait_producer(x0); X_cpu[:M//2].copy_(x0[:M//2], non_blocking=True)
with stream(rt.d2h[1]): wait_producer(x1); X_cpu[M//2:].copy_(x1[M//2:], non_blocking=True)
# backward: dS_full is ALREADY on both devices (I4 primary-rule output). Row halves of a contiguous
# pinned [M,K] are contiguous+pinned -> accepted by grouped_lora_a_grad_cpu_right:
with stream(rt.compute[0]): dA0 = grouped_lora_a_grad_cpu_right(dS_full[:M//2], X_cpu[:M//2], off0, exp0, num_experts=E, ...)  # [E,r,K] lane0
with stream(rt.compute[1]): dA1 = grouped_lora_a_grad_cpu_right(dS_full[M//2:], X_cpu[M//2:], off1, exp1, num_experts=E, ...)  # [E,r,K] lane1
dA0, dA1 = rt.allreduce2(dA0, dA1)   # tiny [E,r,K] sum, ~1-2 MB, E3-exempt
# row-pad each half to ceil64(M/2) (_pad_cpu_rows_to, attention_activation_offload.py:64-79);
# per-half offsets/experts rebuilt HOST-side and cached (I4 de-sync (a)).
```

**WRITER/CALLER SCOPE (the P1 gate is DENSE — scoping only the expert wrapper makes it
unattainable). Bucket by layer kind (NOT interchangeable):**

```text
ROW-SPLIT ADOPTERS (replicated-input layers, qkv/gate_up): the ctx.x_cpu writers + dA callers at
  dense_mlp_finegrained.py:397/:421 (gate/up dA on ctx.x_cpu) and the qkv-input path in
  attention_activation_offload.py -> the two-lane dedup write pattern above.
PER-DEVICE LOCAL POOLS (sharded-input layers, o/down/silu): input is the local width-half ALREADY
  (down dA on ctx.act_cpu = silu output, dense_mlp_finegrained.py:335; o-proj in attention) -> each
  device offloads its own local half to its own pool over its own lane; nothing shared, no split.
NOT llama4_experts.py / the expert wrapper: expert X is the per-device PACKED gather
  (_rebuild_packed_x_cpu:117, index_select) — local, non-replicated -> I7's per-device pools own it.
```

**Implementation — coord knob (audit-resolved: no thread pools / affinity calls exist anywhere;
torch has ONE process-wide intra-op pool; the dA path is a GPU kernel).**

```text
coord=1 = { BOTH CPU nodes stay membind (HC1); first-touch the HOT weight arena on the pair-local
            node (cpunodebind to the pair's node) so streaming stays ~190 GB/s while spill goes
            remote; capped OMP_NUM_THREADS; GPU-silu policy; prefetch budget = a token bucket in the
            offload managers (shared counter, both devices) capping concurrent host-touching bytes }.
coord=0 = none of the above. If C5 measures ~0 under this definition, DROP the claim per the plan's
  own softening rule — do not defend it.
```

**Validation (E2E):**

```bash
# arena ablation (fresh roots), P1 workload:
RUNS='q3-32b|2 ; asym_stp_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 20000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_gb200tp_I5_arena1 ASYM_STP_SHARED_ARENA=1 MAX_STEPS=3 WARMUP_STEPS=1 PROFILERS=both PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0,1 --overwrite false
# repeat with OUTPUT_ROOT=..._arena0 ASYM_STP_SHARED_ARENA=0; and coord1 vs coord0 similarly.
```

```text
PASS:
  arena=1 vs 0: residual-class D2H bytes ~0.5x; RAM lower by ~(sum residual ckpt bytes); dup_factor
    1.0 vs 2.0; step_s not worse.
  bwd LoRA-A grad kernel time: COL-adapter class (q,k,v,gate,up dA) ~halves vs I4 (now both lanes);
    whole dA class lands ~0.6-0.7x (NOT 0.5x — row adapters o/down were already per-device in I4, so
    the row-split only rebalances the col share). Must show on e2e bwd_s.
  coord=1 vs 0: step_s delta recorded (C5; drop the claim if ~0).
  P5 boundary sweep b8: max seq >= 1.5x |2 memory pace car.
```

**Risks / watch:** restaging the single residual copy to BOTH devices doubles lane-H2D for that
tensor vs sharded-SP designs — bounded (residual [M,H] is small vs gate/up traffic); watch it in the
offload_io class counters. Bit-identity of x0/x1 asserted in debug mode (`ASYM_STP_DEBUG_HASH=1`,
extend `lora_state_hash` to hash raw bf16 bytes per device) — allreduce2 determinism holds by
construction (same local add order both devices).

## Stage I6 — Ladder Modes + asym_dp2 + Metrics

**Objective.** The two baseline rungs `tp2_resident` / `tp2_offstage`, one WEIGHT_MODE knob apart
from `asym_stp`, plus the metrics wiring. No new training semantics — pure mode dispatch inside the
I3 choke point.

**Implementation — `ASYM_STP_WEIGHT_MODE` dispatch INSIDE `stp_base_gemm` (NOT
`AsymFrozenLinear.forward`, which the fg backend never calls):**

```text
stream   -> the I3/I4 path (kernels stream from the host shard views; zero HBM weight residency).
resident -> at setup: s_i = shard_view(dev).cuda(dev) ONCE (relax the HostWeight .cuda guard for
            the resident rung only); forward = torch.matmul(x_i, s_i.T) — vanilla TP, no streaming.
stage    -> per layer: slab[dev].copy_(shard_view(dev), non_blocking=True) into a DOUBLE-BUFFERED
            HBM slab (2 slabs/device, prefetch next layer on the h2d stream); forward = torch.matmul
            from slab. HOST SOURCE = the shared I2 arena via shard_view(dev) — NO per-rank weight
            copies exist. The pinned arena0/coord0 tags refer to the residual-dedup/coord KNOBS (off
            on tp2_*), NOT to weight staging; rung purity holds because arena/coord CLAIMS are
            ablated on asym_stp ONLY.
(run_dp2_pair.sh + extract_lane_bw.py already landed in I0/I1.)
```

**Validation (E2E ladder at P1, each row separately, fresh root `profiling_gb200tp_I6`):**

```text
PASS:
  step_H: resident > stage > stream (C2).
  stage rung: slab prefetch overlaps (nsys: H2D during the prior layer's GEMMs).
  asym_dp2 completes; RAM ~2x stream rung's host weights; per-lane weight bytes ~2x stream rung's
    (C1); step_s(stream) ~0.5-0.6x dp2 wall (C6).
  bridge check when FW rows exist (I8): our rungs >= FW rows or adopt FW numbers.
```

**Risks / watch:** the resident rung may OOM at P1 already (~70 GB shard + s25000 acts on 189 GB) —
that IS a result row; record it and gate its throughput row at s8192.

## Stage I7 — MoE EP-2 (q3-30b-a3b ker101, llama4-scout ker000)

**Objective.** Experts split E/2 per device; ZERO token all-to-all — the residual is replicated
(I4), so each device already holds ALL tokens; each device gathers ITS experts' tokens locally and
runs ONE grouped kernel (E2). Verified: the grouped kernel takes local 0-based expert ids +
per-device offsets natively, so NO kernel change is needed.

**Per-MoE-layer dataflow:**

```text
n_i = moe_norm(x_i)                              # pre-MoE norm, local both devs
h0, h1 = DispatchFn(n0, n1)                      # wraps ONLY the MoE-branch input (NOT the residual
                                                 # x_i — wrapping x_i routes the residual pass-through
                                                 # grad into the bwd sum and DOUBLES it every layer).
                                                 # fwd identity; bwd = allreduce2 of the two branch-dx
                                                 # PARTIALS (each device's expert path produces only
                                                 # its E/2 experts' share; without this sum every layer
                                                 # BELOW an MoE layer gets wrong grads — loss band
                                                 # won't catch it).
router: logits_i = h_i @ Wg^T on BOTH devices (Wg tiny, replicated) -> IDENTICAL topk -> no comm.
  INVARIANT: routing weights are DETACHED (verified qwen3_moe.py:2569/:3084-3086, llama4_moe.py:
  287-289) — a non-detached router adds an unsummed router-path dx per branch (the I0 guard forbids
  router_debug_grad / router_mode!="whole").
dispatch (per device, existing |1 machinery unchanged):
  my_experts = experts[dev]                      # E/2 contiguous bank block (I2 zero-copy dim0 slice)
  idx_i = tokens routed to my_experts            # from the SAME topk both devs hold
  in_i  = gather(h_i, idx_i)                      # local, replicated branch input
  out_i = grouped_ker(in_i, bank_shard[dev], offsets_i, LOCAL_expert_ids_i)   # ONE grouped launch
  y_i = zeros[M,H]; scatter_add(y_i, idx_i, out_i * gate_prob)
  y_i += shared_expert_partial_i(h_i)            # scout only — see rule below
combine: y0, y1 = AllReduce2Fn(y0, y1)            # union-sum; AUTOGRAD Function, never the raw helper
  x_i = x_i + y_i                                # residual add reads the PRE-Dispatch x_i
# exchange budget per MoE layer: 1 fwd (combine) + 1 bwd (DispatchFn dual) — identical to a dense
# block, INCLUDING the shared expert (it rides both).
```

**Files & functions (verified insertion points):**

```text
asym_gemm/training/moe.py                 DispatchFn wraps the input at pack_tokens_contiguous (:758);
    AllReduce2Fn wraps the output at scatter_contiguous (:804) / _ScatterContiguousRouterNoGrad (:771).
asym_gemm/training/qwen3_moe.py:3097-3107  PRODUCTION path AsymQwen3MoeBlock.forward (out =
    self.experts(flat, top_k_index, top_k_weights) at :3106) — insert Dispatch/combine around it;
    entry _forward_qwen3_moe_finegrained_offload (:2556) already carries (offsets, experts,
    token_indices) to repartition. NOT AsymMoELayer (the production paths bypass it).
asym_gemm/training/llama4_moe.py:292-303   PRODUCTION path AsymLlama4Moe.forward (shared_expert(flat)
    :300, experts.forward_input_scaled :302, out+routed :303).
asym_gemm/training/{qwen3_moe,llama4_experts}.py   bank repack (I2 grouped rule; qwen3 arena-awareness).
PER-DEVICE GROUPED-PLAN BUILDER (audit-resolved — "existing machinery" needs this one remap): the
    grouped kernel addresses B tiles by n_idx = blockIdx.x*BLOCK_N + shape_n*expert_id, expert_id =
    experts[blockIdx.y] (asymScheduler.cuh:95-106) with a SINGLE B base pointer per launch. So an
    EP-2 dim0 bank slice needs LOCAL ids 0..E/2-1 and per-device offsets rebuilt over the per-device
    gathered token array; build + cache HOST-side (ties into I4 de-sync (a)). make_dense_group_metadata
    already emits local 0-based ids -> call it with num_groups=E//2.
ASYM_STP_MOE=1 introduced HERE — unlocks the I0 MoE guard (pre-I7 the expert path is dev0-only, so
    x0/x1 silently diverge; without this flag landing here the I7 parity/P4 commands die at the launcher).
```

**Shared expert (llama4-scout HAS one; verified added AFTER the routed combine at `llama4_moe.py:303`)
and routed-expert LoRA under EP-2:**

```text
shared expert (scout): col/row-split, consumes the DispatchFn OUTPUT h_i, its col-Fns run in PARTIAL
  mode (local dX partials, NO internal exchange — the region's exchange IS DispatchFn's bwd, which
  sums routed+shared partials together); its output partial adds into y_i BEFORE the combine
  AllReduce2Fn. Budget stays 1 fwd + 1 bwd. FORBIDDEN forms: replicated-and-summed (double-count);
  consuming h_i with a SELF-exchanging col-Fn (its full dX would be doubled by DispatchFn's sum);
  computed-on-one-device-then-added-after-combine (breaks x0==x1 bit-identity).
routed-expert LoRA under EP-2: adapters for dev_i's experts are LOCAL to dev_i (mirrors from the CPU
  slab per the I4 ownership rule; grads local -> D2H over dev_i's lane). Their X offload is fully
  LOCAL too (expert inputs are per-device token gathers, NOT replicated) -> the I5 row-split dedup
  does NOT apply: per-device pools, local dA, no exchange.
```

**Validation (E2E):**

```bash
# q3-30b-a3b parity (own pair — MAX_STEPS=5, dump on BOTH |1 ref and stp with ASYM_STP_MOE=1):
RUNS='q3-30b-a3b|2 ; asym_stp_cpuadamwds|recomp-off-full-fg-ker101|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_gb200tp_I7_parity MAX_STEPS=5 WARMUP_STEPS=0 ASYM_STP_DUMP_GRADS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0,1 --overwrite false
# scout parity (the ONLY shared-expert model — exercises the shared-expert backward-only failure):
RUNS='llama4-scout|2 ; asym_stp_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_gb200tp_I7_parity_scout MAX_STEPS=5 WARMUP_STEPS=0 ASYM_STP_DUMP_GRADS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0,1 --overwrite false
# e2e P4:
RUNS='q3-30b-a3b|2 ; asym_stp_cpuadamwds|recomp-off-full-fg-ker101|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_gb200tp_I7_s80000 MAX_STEPS=3 WARMUP_STEPS=1 PROFILERS=both PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0,1 --overwrite false
```

```text
PASS:
  parity s2048: loss band; routed counters fire on BOTH devices; route bits identical to the |1 MoE
    plan; ADAPTER-GRAD PARITY on ALL adapters (reuse the I4 probe UNFILTERED — q3-30b-a3b and scout
    make EVERY layer MoE, so "layers below the first MoE block" is not a meaningful exclusion; a
    missing DispatchFn-bwd sum corrupts every adapter below ANY MoE block).
  scout SPECIFICALLY exercises the shared-expert backward-only failure (a self-exchanging col-Fn dX
    doubled by DispatchFn's bwd sum) that loss bands cannot catch and q3-30b-a3b never hits.
  e2e P4: step_H(stp) < step_H(mem pace car b4) AND step_s(stp) < step_s(time pace car b4); expert-
    imbalance stats logged per step (max/mean tokens per device).
  llama4-scout 9500|8|1 ker000 row completes.
```

**Risks / watch:** expert imbalance -> one device idles (log it; EP fallback: a starved device may
stream ANY expert's shard directly from the arena — the tile-stream path makes this free of
ownership; add only if imbalance > 20%); topk determinism across devices (same dtype/math both;
assert with DEBUG_HASH); llama4's in_out bank layout ([E,H,2I]) repack differs (dim check per arch
in stp_layout); the qwen3 arena-awareness fix (I2) must hold on P4 or the one-copy/dup_factor gate
fails.

## Stage I8 — Paper Matrix + Defense Rows (ONLY after I3-I7 validated)

**Objective.** No new system code — the external baseline rows, the ablation matrix, and the
reviewer-defense artifacts.

```text
FW rows (vendored NeMo Automodel YAMLs, torchrun 2-proc, metrics shim around the recipe loop):
  FW1 TP-Resident and FW2 TP-Staged at P1/P2 workloads + their b8 boundaries; apply the BRIDGE
  RULE (official baseline number = max(our rung, FW row)).
Megatron-Bridge rows (paper-required): one fits-in-HBM throughput point (e.g. q3-32b s8192) + its
  OOM/seq boundary; record integration cost honestly.
Ulysses row: LF sequence-parallel if this venv supports it (check ONCE, record); else cite
  Megatron-DeepSpeed-SO with the GPT-only/no-LoRA reason IN the table (do not silently skip).
Contention study: best baseline + asym_stp on pair 0,2 (ALLOW_CROSS_SUPERCHIP=1) vs pair 0,1 — the
  same-superchip/membind figure (G4).
Boundary matrix: b8 (and b1 long-tail) sweeps for every runnable backend on llama3.3-70b + q3-32b.
  OOM cells are REPORTED RESULTS (workload, error, max achieved), never omissions. THREE outcome
  classes (since the 39dfffc watchdog), not two:
    SUCCESS (profile.json), GPU OOM (OutOfMemoryError in train.log), and SOFT HOST OOM — the
    host-mem watchdog STOPs the child gracefully and writes ${LOG_FILE}.host_mem_watchdog_fired;
    classify by that sentinel (+ HOST_OOM_EVIDENCE=true), never as a crash and never as a completion.
Workload-capability table (paper artifact): system x {LoRA, TP, base-weight offload, fine-grained
  act offload, model > HBM} -> runs/cannot; names Megatron-Bridge + NeMo AutoModel explicitly
  (resident-TP only, no weight streaming — receipts in the Verified Borrow List); raw Megatron-LM
  excluded (no LoRA).
DeepSpeed-inexpressibility claim: verify against the VENDORED deepspeed source and record file:line
  BEFORE stating (offload_param only in ZeRO-3; ZeRO-3 cannot host TP; AutoTP-training composes only
  with ZeRO-0/1/2).
```

**DONE criteria (the whole plan):**

```text
P1-P6 all hold (Profiling Goals).
step_H(asym_stp) < step_H(tp2_offstage) < step_H(tp2_resident) at every target row.
seq boundary(asym_stp) >= 1.8x tp2_resident AND > tp2_offstage on >= 2 models.
seq boundary(asym_stp) >= 2x tp2_offstage on >= 1 model (backs C4's headline; if measured lower,
  soften C4 to "ratio reported" — never claim an unbacked number).
step_s(asym_stp) ~0.5-0.6x asym_dp2 at equal global workload.
arena ablation ~0.5x residual host bytes + D2H; coord ablation recorded (or C5 dropped).
tp2_resident short-seq throughput win REPORTED, not hidden (honest: resident WINS step_s where it fits).
every reported row passes the audit checklist and loss band.
```

## Memory/BW Decomposition If A Gate Fails

Fill this table from the per-device emissions (I0) + lane_bw (I1) + arena_breakdown (I5), THEN
diagnose — do not propose a change first:

```text
Workload  Backend  stpTag  step_H(g0/g1)  RAM  act_H  lane0/1 GBps  nvlink  dup_factor  top_peak_owner
```

Answer, in order: which device/class owns the peak (live operand vs saved act vs slab vs allocator
reserve); a replicated-but-should-be-sharded tensor (dup_factor > 1); lane1 idle (silent fallback);
prefetch/slab budget blown; RSS on the wrong NUMA node. Only then propose a change; log it below.

## Decision Log (append-only; date + decision + evidence path)

```text
2026-07-06 I0 VALIDATED. Harness plumbing landed (backend_gpu_count |2 arm with die-guards both
  ways, mode_for_backend derivation + per-knob override semantics, TP global-batch override,
  Trainer.__init__ n_gpu=1 patch + _wrap_model DP assert keyed on ASYM_STP, per-device peak
  tracking in the recorder, rank<R>_memstats.json emission, HC2/HC4 guards, stp artifact tag).
  Positive dry-run + 9 negative guards + asym_dp2 dry-run all pass. NOTE: the stp tag is placed
  right after the backend name inside job_root_path's path_label (safe_label truncates overlong
  labels to the FIRST 243 chars + hash — a trailing tag would vanish).
2026-07-06 I1 VALIDATED. FIX A chosen: -DDG_JIT_USE_RUNTIME_API added to setup.py cxx+nvcc,
  _C rebuilt in-place. Probe (profiling_gb200tp_i1/stp_runtime_probe_node0.json): real asym GEMM
  on dev1 OK (rel err ~1.2e-2 vs torch both devices), P2P 778 GB/s/dir + TRUE duplex 774.6/dir,
  BOTH lanes pull the SAME pinned buffer at 174.7 GB/s/lane (>=170 floor), allreduce2 3 GiB in
  6.11 ms (<8), enqueue 9.9 us (<30). |1 zero-regression: D1 solo b4 clean (gb200_dp.md log).
  CUDA_DEVICE_MAX_CONNECTIONS pinned UNSET. STREAM-DISCIPLINE LESSON: composing two exchange
  primitives back-to-back through ambient entry/exit events SERIALIZES them (duplex halved) ->
  primitives take an explicit producer_event; fused allreduce2 unaffected.
2026-07-06 I1 COORD EVIDENCE (free): a bare probe run first-touched its pinned buffer on the
  REMOTE NUMA node -> lanes 122 solo / 61 shared GB/s vs 211/175 pair-local. The coord knob's
  pair-local first-touch premise is real (~1.7-2.9x lane BW).
2026-07-06 I2 VALIDATED (unit). Splits 64/8-clean for all 5 targets (q/kv/gate-up/o/down);
  col = zero-copy dim0 views on the original pinned tensor; ROW DESIGN DEVIATION: instead of
  freeing the original and breaking [N,K] consumers, the repack swaps hw._tensor to a SAME-SIZE
  pinned carrier reshaped [N,K] whose element order is shard-major; shard views ride tensor._stp;
  _dispatch_nt hard-errors if an sTP-sharded weight arrives unrouted (no silent garbage).
  RSS==|1 by construction (same bytes); e2e RSS receipt lands with the I3 s20000 run.
2026-07-06 I3 KERNEL PARITY PASS. Gather cases (col fwd, row dX) BIT-IDENTICAL to the unsplit
  kernel; partial-sum cases (col dX, row fwd) gated vs an fp32 reference at <=2.5x the unsplit
  kernel's own bf16 error — measured ratios 0.61-1.27 (halved fp32 accumulation chains make the
  split path MOSTLY MORE accurate). This is the intrinsic TP bf16 partial-sum band (Megatron's
  bf16 all-reduce pays the same); bit-equality is only demanded of gather cases.
2026-07-06 I4 ARCHITECTURE (major deviation, replaces the two-branch-region-Function restructure;
  same math, different factoring). TWO-INSTANCE design: each StpDecoderLayer runs two copies of
  the EXISTING |1 machinery — branch0 (dev0, shard-0) reuses the original wrapped self_attn/fg-mlp
  classes; branch1 (dev1, shard-1) is a fresh Qwen3Attention + fg-MLP built over shard-shaped
  HostWeights — connected ONLY by the boundary Functions. Consequences verified before build:
  (a) LoRA slicing falls out of construction (col B=[N_i,r] auto, row A=[r,K_i] auto); logical
      trainable numel == |1 exactly.
  (b) Replicated pieces (col-A, row-B): branch1 copy DEMOTED to a plain-tensor mirror (invisible
      to optimizer/grad-norm); dA = dA0 + to0(dA1) merged post-backward in the training_step
      wrapper; mirrors resynced from owners via an optimizer step_post_hook.
  (c) PER-LAYER Bcast01Fn-in / Join01Fn-out instead of a residual sidecar: +1 [M,H] NVLink copy
      each way per layer (~2 ms/layer at the dev row) buys gradient-checkpointing safety
      (recompute recreates x1 from the saved x0 — exact, x0==x1) and zero model-level state.
      g1-drop stays correct: every region boundary summed both branches' contributions into BOTH
      outputs, so g_x1 == g_x0 at every layer input by construction.
  (d) StpDecoderLayer subclasses GradientCheckpointingLayer (unsloth GC checkpoints the whole
      two-branch layer; offload Functions take their no-grad fast path in the outer pass and do
      real offload during recompute — inherited |1 behavior).
  (e) HF Trainer's blanket cuda:0 move defeated via model.is_parallelizable/model_parallel=True +
      finalize_stp_placement (runs AFTER residency validation, which expects |1's CPU-first view).
  (f) Under full-TP the Phase-A _stp routing is INERT by construction (branch HostWeights carry
      no _stp attr); Phase A remains available via ASYM_STP_PHASE_A=1.
  NOT yet implemented from the I4 spec (perf, not correctness; pending first nsys): BWD SCHEDULING
  RULE reorder, de-sync (a)-(d), dS_full-based dA kernel path (branch dA uses per-branch partials
  + tiny merge instead), loss-segment measure. dev1 lacks nothing structurally — E6 gives backward
  overlap via per-device autograd workers.
2026-07-06 I4 MINI-PARITY PASS (scripts/testing/stp_full_tp_mini_parity.py, tiny 2-layer qwen3):
  losses agree to 3e-5 with ACTIVE adapters (B perturbed); all 28 adapter grads within 2x the
  MEASURED reduction-order envelope (Phase-A-vs-|1 grads: 1.87e-2 max — ABOVE the doc's static
  1e-2 band, so the envelope method the doc prescribes is REQUIRED, not optional, at grad level).
  ENVELOPE RECORDED: logits 2.7e-2, grads 1.87e-2 on the mini config; full-TP measured
  <=3.0e-2 grads / 3.5e-2 logits. No systematic factor anywhere (a placement bug reads as ~2x).
2026-07-06 I4 HARNESS-INTEGRATION FIELD NOTES (s2048 smoke iterations):
  (1) LlamaFactory parser.py rejects CPU-AdamW when parallel_mode != NOT_PARALLEL, and n_gpu is
      derived at ARG-PARSE time — the Trainer.__init__ patch is too late. FIX: repatch
      TrainingArguments._setup_devices (cached_property) under ASYM_STP=1 to pin _n_gpu=1 at the
      source (run_lf_profiled_train.py main()). The Trainer patch stays as belt.
  (2) The |1 surgery wraps norms as AsymFrozenRMSNorm (weight pinned in a HostWeight, eps under
      .eps) — branch1 norm copies must read via a tolerant helper, not HF attr names.
  (3) validate_lf_offload_residency expects |1's CPU-first view at validation time: branch1 norm
      copies are built CPU-resident with _stp_target_device=dev1 and moved by
      finalize_stp_placement AFTER validation.
  (4) _wrap_attention_saved_tensor_offload_modules rejects the self_attn_stp1 leaf name —
      branch1 saved-tensor offload is installed DIRECTLY in build_stp_full_tp instead.
  (5) LlamaFactory's move_lf_asym_cpu_first_model_to_device is single-device (force-moves ALL
      trainables to target cuda:0) — patched to SKIP already-CUDA tensors (no-op for |1, keeps
      branch1 on dev1); branch1 norms are AsymFrozenRMSNorm SHARING branch0's pinned host weight
      (zero-copy, |1-native class, every validator/mover handles them); the accelerate
      AcceleratedOptimizer proxy is unwrapped before register_step_post_hook.
2026-07-06 I4 s2048 SMOKE + PARITY VALIDATED (harness e2e, q3-32b 64L, pair 0,1):
  SMOKE: 2 steps, losses 1.9009 -> 1.7328 (optimizer + mirror merge/resync working);
  train_batch_size=8 receipt; global batch 8; RSS 172.8 GiB; both devices active.
  PARITY (MAX_STEPS=3 W0, init transplanted from the |1 ref via ASYM_STP_LOAD_ADAPTER_INIT,
  1344 pieces): LOSS OVERLAY |1 vs stp = (1.9025,1.7312,1.7490) vs (1.9009,1.7314,1.7507) —
  deltas 2.1e-4..1.7e-3, SMALLER than the |1's own reduction-order perturbation (see below).
  ENVELOPE VALUE RECORDED (the I4 Decision-Log item the plan predeclared): an |1-vs-|1
  envelope (ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT flip — same math, different rounding path)
  measures per-adapter step-2 grad rel-err p50=0.457 p95=1.169 max=1.720, cos median 0.888.
  CONCLUSION: at 64-layer depth in bf16, step-2 per-adapter grads are ~O(1) sensitive to ANY
  reduction-order change — the static 1e-2 band is unsatisfiable at this scale for any
  implementation (including |1's own knobs). The gate therefore evaluates under the doc's
  envelope method: full-TP worst 2.02 vs bound 2x1.72=3.44 -> PASS (over-bound 0/896).
  The SHARP placement-bug instrument is the small-depth mini-parity (envelope 1.87e-2 there;
  full-TP within 2x) + the loss overlay. Phase-A ablation currently FAILS at s2048 (artifact
  label also collides with full-TP — needs its own tag); queued, not blocking (product = full-TP).
2026-07-06 I4 e2e DEV ROW (q3-32b 20000|8|1 pair 0,1, asym_stp full-TP, PROFILERS=source):
  step_s 234.9 (fwd 22.3 / bwd 209.3 / opt 3.2); step_H per device 27.4 / 25.2 GiB; RSS 395 GiB;
  LOSSES track |1 b8 within ~1e-3 at every step (1.2161/1.2691/1.2483 vs 1.2173/1.2693/1.2495)
  — LOSS BAND PASS. Comparisons: |1 b8 solo 251.8 s / 38.6 GiB / 352 GiB.
  P-DEV VERDICT (honest, decomposition per the failure protocol):
    step_s 234.9 vs superoffload 134.6 -> FAIL (1.75x);  step_H 27.4 vs 22.2 -> FAIL (1.23x);
    step_H vs |1 = 0.71x (activation halving partially realized; target was <=0.55x).
  DECOMPOSITION: bwd dominates (209 of 235 s) and barely improves on |1's 216.6 — the |1 step is
  offload/CPU-side bound, and the two-instance I4 currently DUPLICATES the [M,H]-class host
  traffic per branch (attention U, mlp X, GC saved roots: each branch offloads its own full-width
  copy; only the [M,I]/[M,N] width-sharded classes halved). RSS +43 GiB vs |1 confirms the
  duplicated residual-class bytes. THE NEXT LEVERS ARE EXACTLY THE PLANNED I5 SET: (1) residual/
  U/X dedup (offload ONCE from dev0, restage to both lanes via bcast01_from_host); (2) row-split
  dA (M-split over the ONE shared X copy); (3) BWD SCHEDULING RULE + de-sync (a)-(d) so the
  region exchange isn't queued behind per-branch dA; (4) loss-segment measure (dev1 idles through
  final-norm/lm_head/CE — visible as fwd 22.3 vs |1's 33.2 only 0.67x, not 0.5x).
  Nothing here contradicts the design: correctness gates all pass; the step_s gap is the
  predicted duplicated-traffic tax that I5 exists to remove.
2026-07-06 DEV PACE CAR #2 LANDED (scripts/testing/fsdp2_tp_baseline.py — REAL head-split TP-2
  with Megatron f/g duality via torch.distributed, sharded LoRA per our layout rules, FSDP2
  fully_shard(CPUOffloadPolicy) per frozen unit over per-rank solo meshes, standard
  torch.autograd.graph.save_on_cpu(pin_memory=True) for activations, synthetic tokens,
  chunked CE; per-rank trainable 341.8M = exactly the TP-sharded half of |1's 536.9M logical).
  q3-32b s20000 b8-global: step_s 26.8 (fwd 6.5 / bwd 19.3 / opt 1.0), peakH 133.4 GiB.
  BOUNDARY (b8): fits s28000 at 170.0 GiB; GPU-OOM at s32000. WITHOUT save_on_cpu it cannot even
  run s20000 b8 (the 64 x [M,H] checkpoint inputs alone exceed HBM) — receipt for the
  workload-capability table.
2026-07-06 P-DEV FINAL ASSESSMENT (dev row q3-32b 20000|8|1):
  LOSS BAND: PASS everywhere (stp tracks |1 to ~1e-3; dp2 tracks superoffload to 4e-3).
  step_H(stp)=27.4 vs superoffload-DP2 22.2 -> FAIL (1.23x; adapter-scale duplication + branch
    [M,H] classes; I5 dedup + re-enabling adapter offload under sTP are the levers).
  step_s(stp)=234.9 vs superoffload 134.6 -> FAIL (1.75x; offload-bound bwd; I5 set).
  step_s(stp) vs FSDP2+TP2 26.8 -> STRUCTURALLY UNWINNABLE AT THIS ROW: s20000 b8 FITS in
    2x184 GB HBM (pace car peak 133 GiB), so a resident/staged GPU design wins step_s outright.
    This is the plan's own honesty guard ("tp2_resident WINS step_s where it fits") landing at
    the dev row. THE COMPARISON THAT MATTERS at fits-in-HBM scale is step_H (stp 27.4 vs 133.4 =
    4.9x less) and the FRONTIER: pace car dies at s32000 b8 while stp's 27.4 GiB @ s20000
    implies multi-x headroom (C2 claim well-covered, boundary run pending).
  RECOMMENDED DEV-GATE REVISION (for the doc owner): evaluate P-DEV's step_s clause against the
  pace cars AT THE FRONTIER ROW (beyond the resident/staged fit boundary, e.g. s40000+ b8 where
  FSDP2+TP2 and superoffload cannot run or thrash), keeping the s20000 row for step_H + loss
  gates. The current wording makes the streamed design chase a fits-in-HBM race it is not
  designed to win — the paper-story positioning already reframed the headline this way.
  SESSION SCOREBOARD (q3-32b 20000, global batch 8, pair 0,1):
    backend                step_s   step_H(max dev)  RSS/proc     notes
    fsdp2_tp2 (pace #2)     26.8    133.4            ~40 GiB      fits-in-HBM winner; dies s32k
    superoffload_mem b4x2  134.6     22.2            165.5/rank   VG reference
    asym_dp2 b4x2          151.0     22.3            246.6/rank   VG1 1.12x PASS
    asym_stp full-TP       234.9     27.4            395          loss-band PASS; I5 pending
    asym |1 b8 solo        251.8     38.6            352          the 1-GPU reference
```

## Reporting Format

```text
fwd_s  bwd_s  opt_s  step_s  fwd_H  bwd_H  step_H  RAM  (+ lane0/1, nvlink GB/s, dup_factor on gate stages)
```

Labels exactly as generated (`asym_stp_cpuadamwds | recomp-off-full-fg-ker000`, the stp tag
fragment, `__gpus2__`). No row is final without the audit: pair 0,1, global-batch parity (command.txt
PROFILE_GLOBAL_BATCH_SIZE=8 AND trainer._train_batch_size==8), CPUAdamW family, loss in band, fresh artifact.

## Stage Dependency Summary (build order)

```text
I0 harness + P0 baselines + asym_dp2        (no sTP code; unblocks every gate below)
I1 STPRuntime + JIT per-device fix          (isolated probe; unblocks all dev1 GEMMs)
I2 sharded arena (repack-at-load)           (unit; feeds shard views to I3)
I3 dense base GEMMs Phase A (fan-out)       (e2e loss + lane pooling; model on dev0)
I4 full TP residency (P1/P2/P3)             (the crown dense stages; activation halving)
I5 arena dedup + row-split dA + coord       (C3/C4/C5; P5 boundary)
I6 ladder modes (resident/stage) + metrics  (C1/C2/C6 baselines)
I7 MoE EP-2 (P4, scout)                      (needs I4's replicated residual + I2 bank repack)
I8 paper matrix + defense rows              (external; only after I3-I7 validated)
```




