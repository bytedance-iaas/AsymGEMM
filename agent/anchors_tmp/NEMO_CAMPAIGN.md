# NeMo (Megatron-Bridge) baseline campaign — q3-30b-a3b + q3.5-35b-a3b, 2-rank EP2
(2026-08-02, node c17, user directive: build the NeMo/Megatron-Bridge baseline,
"multi-rank normal EP, with/without activation offloading; expectation: fails
even against SuperOffload because NeMo does not offload model weights; don't
stop until env validated + throughput results prove it underperforms on the
same seq lengths.")

## Deliverables
- scripts/lf/bootstrap_nemo_venv.sh  -> .venv-nemo   [VALIDATED — see below]
- scripts/lf/bootstrap_nemo_venv_fa4.sh -> .venv-nemo-fa4 (fa4 sibling, LF-fa4 pins)
- scripts/lf/profile_lora_nemo.sh + scripts/lf/run_nemo_lora_sft.py — nemo-only
  driver, SAME protocol as profile_lora_lf_test_source.sh (w1+m2, eff tok/s =
  measured*b*ranks*ga*s / sum(step_s), GLOBAL over ranks; mock full-length data;
  LoRA r64/a16/drop0 all-linear incl. per-expert adapters; serial runs, host
  watchdog, GOOM/COOM verdicts; run dirs + jobs.tsv + step_samples.csv +
  profile.json house layout under profiling_results/profiling_nemo/).

## Stack (why these pins)
- Megatron-Bridge @ dabf51d9d + vendored mcore submodule 69c4868 (editable).
- torch 2.12.0+cu130 (house pin) + TE 2.16.0 (prebuilt aarch64 cu13 wheel;
  lock pins the same 2.16.0 vintage from git).
- nvidia-cublas==13.6.1.10 pinned LAST: torch metadata hard-pins 13.1.1.3
  which lacks TE 2.16's grouped-GEMM cublasLt symbols; every later pip resolve
  that touches torch downgrades it back (recurring landmine — the bootstrap
  orders it last; anything pip-installed into this venv later must re-pin).
- COMPAT PATCH to Megatron-Bridge (src/megatron/bridge/peft/utils.py,
  _forward_te_grouped_linear): released TE 2.16.0 _GroupedLinear takes the
  22-slot non_tensor_args (weight_workspaces, cache_weight, skip, save, debug)
  and returns (out, workspaces); upstream targeted the d64bc14 git layout.
  Without it every grouped per-expert LoRA fwd dies "expected 22, got 21".
- apex absent (pip env): gradient_accumulation_fusion=False (frozen base
  weights under LoRA make it moot), torch-norm fallback warnings benign.
- nvidia-modelopt 0.44.0 + megatron-energon 7.4 = import closure of
  megatron.bridge; fsspec re-pinned <=2026.4.0 for datasets 5.0.0.

## Config (steelman for NeMo)
- 2 ranks (GPUs 0+1), TP1 PP1 CP1 ETP1, EP2 = "normal EP"; DP=2 (batch is
  per-rank like every 2r plot cell). alltoall dispatcher (DeepEP silently
  no-ops on GB200 in this checkout; HybridEP needs extra deps).
- moe_grouped_gemm (TE grouped), moe_permute_fusion, fused CE ("te" impl —
  the ligerloss analogue, always on like the baselines' ligerloss1).
- Arms: `recomp` = recompute_granularity full/uniform/1 (strongest recompute);
  `actoff` = fine_grained_activation_offloading + ALL 7 offloadable modules
  (core_attn, attn_proj, qkv_linear, attn_norm, mlp_norm, expert_fc1, moe_act),
  fraction 1.0 — Megatron's strongest activation offload; upstream validation
  makes cpu_offloading and recompute mutually exclusive, and NOTHING offloads
  the full-recompute layer boundaries or the frozen weights. Weights stay
  HBM-resident: 17.3B params/rank = ~34.6 GiB before a single activation.
- Real weights loaded from the HF snapshot (direct HF-dir pretrained_checkpoint
  path), mock data at exact seq (house runs likewise train synthetic rows).
- 35b = VL checkpoint (Qwen3_5MoeForConditionalGeneration): VL bridge, vision
  tower frozen + no image inputs, MTP off, text-only mock conversations padded
  to full seq (pad rows still run MLP/experts; attention may skip pads →
  throughput generous to NeMo), CE fusion impl "native" (their VL default).

## Reference cells to beat (tp2r plot, plot_tp_vs_seq_2r.py, GLOBAL tok/s)
- q3-30b-a3b @384k: recomp 2370 · unsloth 2367 · asym sEP-T2 2314; uns 1386@640k;
  asym holds 640k..1.04M (1477..901). FSDP2/ZeRO3 track recomp.
- q3.5-35b-a3b @256k: recomp 1620 · uns 1584 · uns-off 1498 · asym T2 2005;
  asym holds 384k..896k (2467..2640); uns/uns-off die past 512k.

## Ops lessons
- Watchdog false-COOM (chain v1, rc384 07:26): a NUMA-"free"-based floor check
  fires the moment the 61 GB safetensors read lands in page cache. Fixed to the
  house method (per-CPU-node MemFree + FilePages − Shmem,
  host_mem_watchdog_avail_kb): 935 GB available reported on the idle node.
  Chain v1's rc384/ao384 dirs deleted; chain v2 re-measures them.

## Smoke (2026-08-02 ~07:2x)
- q3-30b-a3b EP2 s=8192 b1: TRAINED, eff 2674 tok/s global, peak resv 59.0
  GiB/rank; 384 adapter sites (192 expert-grouped); loss ~5.0 sane.

## RESULTS — q3-30b-a3b (2 ranks, EP2, b1 ga1; eff tok/s GLOBAL, resv GiB/rank)
chain v3 2026-08-02 07:32-07:57 c17, all MEASURED, w1+m2:
| seq  | nemo recomp        | nemo actoff        | plot @same seq (rc/uns/asym) |
|------|--------------------|--------------------|------------------------------|
| 32k  | 6026 (80.3)        | 7171 (166.9)       | — (below plot range)         |
| 64k  | 6636 (109.5)       | **GOOM** (187 smi) | —                            |
| 96k  | 6141 (138.4)       | —                  | —                            |
| 128k | 5467 (169.2)       | —                  | —                            |
| 160k | **GOOM** (179+ used, tried 7.89G) | — | —                     |
| 384k (plot rung 1) | **GOOM** (180.5 used, tried 17.4G) | GOOM (redo pending) | 2370 / 2367 / 2314 |
- **recomp wall (128k,160k]** — 2.4x below SuperOffload-recomp's last fit
  (384k), 4x below uns (640k), 6.5x below asym's 1.04M crown. HBM slope ~0.92
  GiB/1k tok/rank (resident 34.6 GiB weights + logits/CE + 48 layer
  boundaries; nothing offloadable).
- **actoff wall (32k,64k]** — WORSE than recomp: upstream makes offload and
  recompute mutually exclusive, and the offload list cannot touch logits/
  router/dispatch buffers, so dropping recompute to enable offload doubles
  the resident footprint (166.9 vs 80.3 GiB at 32k). Where it fits it is
  faster (7171 vs 6026 @32k: offload D2H beats recompute flops at short seq),
  but its ceiling is ~1/6 of recomp's.
- At EVERY tp2r plot rung (384k..1.04M) NeMo = OOM on both arms.
- **Max-composition probe (2026-08-02 15:1x, selrecomp-actoff = selective
  recompute of the whole MoE block + offload of the attention side
  core_attn/attn_proj/qkv_linear/attn_norm/mlp_norm)**: 128k TRAINED 5947
  (174.4 GiB) vs recomp 5467 (169.2) -> +8.8% tok/s for +5.2 GiB; 160k GOOM.
  Wall UNCHANGED at (128k,160k] — checkpoint boundaries, weights, and
  logits/CE are outside both lists, exactly as argued. Best NeMo config =
  small speed win in the mid-range, zero capacity gain.

## RESULTS — q3.5-35b-a3b (2 ranks, EP2, b1 ga1; eff tok/s GLOBAL, resv GiB/rank)
chain + probes 2026-08-02 08:13-08:32 c17, all MEASURED, w1+m2:
| seq  | nemo recomp        | nemo actoff     | plot @same seq (rc/uns/uns-off/asym) |
|------|--------------------|-----------------|---------------------------------------|
| 16k  | 3231 (116.4)       | **GOOM**        | — (below plot range)                  |
| 24k  | **GOOM**           | **GOOM**        | —                                     |
| 32k  | **GOOM** (61.04G unfused-attn scores) | GOOM (same) | —            |
| 256k (plot rung 1) | **GOOM** (tried 244.14 GiB = fp32 logits [256k × 248320 vocab]) | GOOM | 1620 / 1584 / 1498 / 2005 |
- **recomp wall (16k,24k]** — vs SuperOffload-recomp 256k last fit (>=10x),
  uns/uns-off 512k (>=21x), asym 896k crown (>=37x).
- **actoff wall <16k** — offload arm again strictly worse (recomp@16k holds
  116.4 GiB; dropping recompute to enable offload overflows 184).
- **Root cause is architectural**: TE's fused/flash attention does not
  support Qwen3.5's output-gated full-attention layers, so attention_backend
  auto silently degrades to UNFUSED torch-softmax attention — the [1,16,S,S]
  fp32 score tensor is 61.04 GiB at S=32k and O(S^2) beyond (the same wall
  class as the tpfig phi 256k sliding-window mask). On top of that the fused
  CE still materializes fp32 [S, 248320] logits (244 GiB at 256k alone).
  NeMo has NO fused path for this model class today and nothing that offloads
  weights (34.6 GiB/rank for 30b, ~37 GiB/rank language tower for 35b) or the
  recompute layer boundaries.

## RESULTS — 1-RANK (single GPU, EP1: FULL model resident; 2026-08-02 16:30-17:40)
q3-30b-a3b (61 GiB weights on the one GPU; 1-rank plot rungs 80k/128k/...):
| seq | nemo recomp (eff, resv) | 1r plot @same seq (rc/uns/uns-off/asym) |
|-----|--------------------------|------------------------------------------|
| 32k | 2178 (124.5) | — |
| 64k | 2564 (148.1) | — |
| 80k (rung 1) | 2557 (160.1) | 5206 / 5288 / 2693 / 4984 -> nemo = **half** of rc/uns |
| 96k | 2541 (173.7) | — |
| 128k (rung 2) | **GOOM** | 2985 / 3055 / 2032 / 2982 |
- actoff arm: **GOOM already at 32k** (1000 MiB tail alloc with 183/184 used).
- recomp wall (96k,128k] — one rung deep vs recomp/uns 320k+/640k, asym 1.6M.
- At its only shared rung (80k) nemo is ~2x slower than SuperOffload-recomp.

q3.5-35b-a3b (1-rank plot rungs 128k/256k/...; ~70 GiB weights + unfused attn):
| seq | nemo recomp | notes |
|-----|-------------|-------|
| 8k  | 837 (157.2) | |
| 16k | 1215 (177.2) | actoff @16k **GOOM** |
| 24k | **GOOM** | wall (16k,24k], same bracket as 2-rank (S^2 attn dominates) |
| 128k (rung 1) | **GOOM** | plot: rc 1024 / uns 1831 / uns-off 2020 / asym 2484 |
- nemo's best-anywhere tok/s (1215 @16k) is BELOW every baseline's tok/s at
  128k (8x the seq). Every plot rung: OOM.

## Ops lesson (35b 1-rank, IMPORTANT): VLM mock padding silently no-ops at dp=1
First 1-rank 35b ladder produced constant ~4.6s steps / constant 149.9 GiB
across s=8k..128k with "TRAINED" verdicts — bogus. NEMO_DBG instrumentation
proved the model received (1,128)-token batches: cfg.dataset carries
pad_to_max_length=True but the collate receives False when data-parallel size
is 1 (at dp=2 it arrives True; sequence_length arrives correctly in both).
Upstream plumbing bug. run_nemo_lora_sft.py now FORCES pad_to_max_length at
the qwen_vl collate boundary and NEMO_DEBUG_BATCH=1 prints the shapes reaching
the model; all v2 1-rank cells carry "NEMO_DBG_BATCH tokens=(1, S)" proof
lines. Bogus cells deleted; only v2 cells are quotable.

## RESULTS — ADDITIONAL MoEs (2026-08-02 19:05-19:40, chain nemo_moes_chain.sh
+ supplements; eff tok/s GLOBAL, resv GiB/rank; NEMO_DBG shape proofs in logs)

**mixtral-8x22b: upstream-UNSUPPORTED (probe banked), then LOCALLY PORTED
(2026-08-03).** No bridge exists upstream for MixtralForCausalLM; we added one
(src/megatron/bridge/models/mistral/mixtral_bridge.py + __init__ import, ~130
lines on the Qwen3MoEBridge template: block_sparse_moe.{gate,experts.E.w1/w3/
w2} mappings, pre-softmax top-2 router, moe_ffn_hidden_size=intermediate_size).
Port validated: provider fields exact (56L/6144h/48-8heads/8E top2/theta 1e6),
full weight load clean, loss 5.96->5.54 finite grads (mapping-garbage would sit
~10.4/NaN). RESULTS with the ported bridge (llama4-scout remains unsupported —
probe LLAMA4_BRIDGE_UNSUPPORTED, Llama4ForConditionalGeneration):
- r2 EP2 recomp: **4419 tok/s @8k (170.2 GiB — ceiling; ~145 GB/rank weights)**,
  16k GOOM (in-loop) -> wall (8k,16k]. actoff @8k GOOM. r1 @4k load GOOM
  (282 GB > 184, measured).
- Same-seq SO comparison @8k r2: SO-recomp **HOST-C-OOM x3** (b8/b4/b2, house
  watchdog "available 49 GiB < floor 50" — SO's per-rank host machinery
  duplicates the 282 GB model and blows the 957 GB pool; matches tpfig's
  ~870 GB@1-rank note). The one cell where nemo beats SO outright at 2 ranks —
  HBM-resident weights squeak in where host-replicated offload cannot. Mirror
  image at 1 rank: SO runs 64k/128k (1704/1162, house) while nemo cannot load.
  House mixtral panel stays 1-rank, where nemo has zero coverage.

**q3.5-122b-a10b: ZERO COVERAGE at both rank counts.** 2-rank EP2 recomp@4k
GOOMs at 183.47/184 GiB BEFORE the weight-load phase even starts (model
construction + DDP/dist-opt buffers overflow; ~122 GB/rank shards, no weight
offload). rc@128k (tp2r rung 1) GOOM, actoff@4k GOOM, 1-rank load probe GOOM
(244 GB weights > 184). Plot @128k: rc 752 / uns 1445 / asym 1665 — nemo: none.

**glm4.7-flash** (2r plot rungs 32k..192k, rc ref 6688@32k..1560@192k):
| | 16k | 32k+ |
|---|---|---|
| r2 recomp | 3090 (73.2) | **GOOM** — [20,S,S] fp32 unfused scores, 76.29 GiB @32k |
| r2 actoff | **GOOM** | — |
| r1 recomp | 1308 (111.6) | **GOOM** (same S^2 tensor, rank-count independent) |
- TE has no fused kernel for glm4.7's attention variant -> forward_torch_softmax
  fallback -> wall (16k,32k] BELOW the panel's first rung. Same failure class
  as qwen3.5 (gated attention).
- r1 actoff @16k: GOOM (smi peak 188.4 GB; measured 08-03 ~07:0x) — closes the
  matrix; every runnable model x rank x arm now has a measured cell.

**glm4.5-air** (2r plot rungs 16k..128k, rc ref 4730@16k..1886@128k):
| | 8k | 16k (rung 1) | 32k |
|---|---|---|---|
| r2 recomp | 2084 (158.7) | 3113 (171.1, smi 182) | **ncclUnhandledCudaError x2** (reproduced) = G-OOM-class @edge |
| r2 actoff | NCCL@edge FAIL (<8k wall) | GOOM | — |
| r1 | — | — | load GOOM (212 GB > 184) |
- Air's wall is pure resident-weights budget (~106 GB/rank at EP2; fused attn
  fine, grep 0 unfused hits): 16k runs AT the ceiling, 32k tips NCCL over.
  At its single covered rung nemo = 66% of SO-recomp (3113 vs 4730), 45% of
  asym (6844); baselines carry to 128k.

Pattern across all six models: every failure is one of (a) resident weights
(no weight offload): 122b/air/mixtral-by-size, (b) unfused O(S^2) attention on
2025-vintage attention variants: qwen3.5, glm4.7, (c) boundaries+logits under
full recompute: q3-30b — and the offload arm NEVER extends capacity anywhere.

## SAME-SEQ SO-recomp comparison cells (2026-08-03 05:07-06:40, soeq_chain
+ supp; SO on the LF stack, best-over-batch, w1+m2; GLOBAL tok/s; resv GiB/rank)
NeMo's last-fit seqs, head-to-head (NeMo value from the campaign tables):
- 2r q3-30b @128k:  SO 5408 (b2;119.5)  vs nemo 5467 (b1;169.2)  -> tie (+1% nemo), SO -50 GiB
- 2r q3.5-35b @16k: SO 2341 (b16;122.2) vs nemo 3231 (b1;116.4)  -> nemo +38% (SO b8=1163; b24 unprobed ~180 proj)
- 2r glm4.7 @16k:   SO 4813 (b4;143.4)  vs nemo 3090 (b1;73.2)   -> SO +56%  [SO ran ligerloss0*]
- 2r glm4.5-air @16k: SO 4730 (b8, house GLMTP cell) vs nemo 3113 (171.1) -> SO +52%
- 1r q3-30b @96k:   SO 4282 (b2;91.0)   vs nemo 2541 (b1;173.7)  -> SO +69%
- 1r q3-30b @80k:   SO 5206 (b4;~149, house) vs nemo 2557 (160.1) -> SO +104%
- 1r q3.5-35b @16k: SO 1207 (b16;121.2) vs nemo 1215 (b1;177.2)  -> tie
- 1r glm4.7 @16k:   SO 2706 (b4;143.4)  vs nemo 1308 (111.6)     -> SO +107% [ligerloss0*]
*glm4.7 SO cells ran ligerloss0: this tree's Liger lacks apply_liger_kernel_to_
glm4_moe_lite (present in the sibling workspace that measured the house glm
cells); unfused CE only SLOWS SO, so its wins are lower bounds. b8 GOOMed
(logits w/o fused CE); best measured b4. Ops note: LF sanitizes RUN_NAME to
lowercase — verdict greps must use lowercase dirs (soeq chain mislabels fixed
at harvest).
Net: with best-over-batch, SO-recomp WINS or TIES at nemo's own last-fit seq
in 7 of 8 pairs (nemo's sole edge: 35b 2r 16k, +38%) — and SO then continues
2.4-16x deeper in seq. The nemo "fast where it fits" effect survives only on
the qwen3.5 short-seq cells.


## FA4 REMEASURE ADDENDUM (2026-08-08)
The 08-02 qwen3.5-35b and glm4.7-flash walls were measured WITHOUT
FlashAttention (.venv-nemo has no flash-attn; TE fell back to
UnfusedDotProductAttention for their 256-dim heads — qwen3.5 full-attn
hd256+output-gate, glm4.7 MLA qk 192+64=256/v 256; cuDNN fused covers
hd<=128 + the DeepSeek (192,128) special case only). TE detects flash-attn-4
(v4_is_installed=True) and its FA gate covers hd<=256 on sm100, so
profile_lora_nemo.sh now AUTO-ROUTES these families to .venv-nemo-fa4
(nemo_env_for_model). Remeasured (0 unfused hits verified):
- q3.5-35b: 2r 5556@32k (158.7/rank) wall (32k,64k]; 1r wall <32k — after
  FA4 the binding limits are the 248k-vocab fp32 logits + per-layer state.
- glm4.7-flash: 2r 9663/8838/7117/5860/3903 @32-192k wall (192k,256k]
  (8x the unfused-era 32k wall, still one rung short of its panel); 1r
  3648/3799/3324/2817 @32-128k wall (128k,192k] — 2817@128k IS a panel rung:
  megatron's one tall bar (resident-weights speed), dead next rung.
- Other four models: unaffected (never hit the fallback; rows stand).
Figures updated + pushed (6c1741e). Ops: a concurrent figure-editing session
rewrote megatron rows mid-update (["OOM"]*13 style) — all 12 rows were
deduped, length-checked, and bar-alignment-verified before the push.

## Ready-to-paste tp2r DATA cells (if a nemo series is wanted in the figure)
```python
# q3-30b-a3b panel (seqs 384k..1.12M): nemo OOM at every rung, both arms.
"nemo_recomp": ["OOM", "OOM", "OOM", "OOM", "OOM", "OOM", "OOM", "OOM"],
# q3.5-35b-a3b panel (seqs 256k..1.02M): nemo OOM at every rung, both arms.
"nemo_recomp": ["OOM", "OOM", "OOM", "OOM", "OOM", "OOM", "OOM"],
# (walls, measured: 30b recomp (128k,160k] / actoff (32k,64k];
#  35b recomp (16k,24k] / actoff <16k — all below the panels' first rung.)
```

## More ops lessons (35b)
- VLMLoRA's freeze walk crashes on Qwen3VLModel (no .vision_projection attr);
  the upstream VL PEFT recipe pattern (plain LoRA + provider freeze flags) is
  the working path.
- Adapters matched inside the frozen never-run vision tower are
  trainable-yet-gradless and trip DDP's per_param_grad_ready_counts assert —
  scope targets with "*language_model*..." wildcards.
- GDN (delta-net) layers' in_proj/out_proj are NOT in upstream's default LoRA
  target set — only the 20 full-attn layers + experts + shared experts get
  adapters (360 sites). Generous to NeMo; it still walls at (16k,24k].
- mcore GPTDatasetConfig default create_attention_mask=True materializes a
  numpy [S,S] causal mask PER SAMPLE in the dataloader (147 GB @384k -> host
  OOM before step 1). Must be False; TE causal attention needs no mask.

## State
- [x] recon (bridges: Qwen3MoeForCausalLM native; 35b via VL bridge)
- [x] .venv-nemo builds + imports; smoke TRAINED (EP2+LoRA+TE grouped)
- [x] q30b chain COMPLETE (both arms: plot-rung GOOMs + walls + ladder tok/s)
- [x] q35b chain COMPLETE (both arms: plot-rung GOOMs + walls + 16k tok/s)
- [x] fa4 fresh build PASSED (FA4_BOOTSTRAP_RC=0, ~08:5x) — .venv-nemo-fa4
  built from scratch through bootstrap_nemo_venv.sh + the FA4 layer
  (flash-attn-4 4.0.0b16, cutlass-dsl 4.5.2, fla 0.5.0), verify green incl.
  "qwen3-moe + qwen3.5 bridges ok" => the base bootstrap is clean-reproducible.
- [x] results banked here
CAMPAIGN COMPLETE 2026-08-02 08:5x.
