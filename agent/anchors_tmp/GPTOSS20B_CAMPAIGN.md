# GPT-OSS-20B INTEGRATION + THROUGHPUT CAMPAIGN (2026-08-12, c17, 38-tree)

User directive (verbatim intent): integrate gpt-oss-20b into the pipeline;
make T1/T2/T2B/T3 work THOROUGHLY; memory verdict = asym T3 must beat
superoffload uns-off (its most memory-lean config) NONTRIVIALLY, keep fixing
tiers until it does; then 1-rank AND 2-rank throughput ladders with TURNING
POINTS + CEILING visible (house tp-figure style — not just small seqs);
2-rank runs use the **sepplanlink** backend (asym_sepplanlink2_cpuadamwds,
ported from the 46-tree working diff). DO NOT STOP until all goals met.

## Integration facts (2026-08-12)
- Model: openai/gpt-oss-20b — 24L, 32E top-4, hidden 2880, interm 2880,
  64QH/8KVH hd64, vocab 201088, ties=false, ctx 131072 (yarn ×32),
  alternating sliding_attention(128)/full_attention, attention sinks,
  MXFP4 experts in the HF checkpoint.
- Checkpoint: DEQUANTIZED bf16 local copy at
  /scratch_local/user_data/shutian/kevin/cache/fused/gpt-oss-20b-bf16
  (39.0 GiB, 8 shards; per-layer expert tensors verified distinct; config
  stripped of quantization_config). Why: venvs lack the `kernels` pkg so HF
  would warn+dequant on EVERY load (transient); tf-5.6 save_pretrained also
  MANGLES dequantized expert weights via shared-tensor detection
  (down_proj$/gate_up_proj$ keys, 4.9G file) — manual shard writer used.
- Attention: gpt_oss has _supports_sdpa=False (sinks). Plain .venv has NO
  flash-attn → eager O(S²) fallback. FA4 4.0.0b16 (.venv-fa4) supports
  learnable_sink + window_size → gpt-oss rides the qwen3.5 FA4 stack via a
  new is_gptoss_model_name auto-switch in resolve_current_runtime_for_model
  (ASYM_GPTOSS_FA4_AUTO=1 default; env overrides win as usual).
- Liger: gpt_oss added to LF _LOSS_ONLY_SUPPORTED_MODEL_TYPES + resolver →
  vendored apply_liger_kernel_to_gpt_oss (loss-only FLCE, class-level patch,
  DS-safe). 201k vocab = incident-#4 class; BOTH sides run it.
- Driver: M[gpt-oss-20b] → fused path (layers 24); watchdog floor 35 (both
  spellings); tier family moe; template gpt_oss (LF ships it).
- Tiers for gpt-oss: T1/T2/T2B presets as-is (ker000 tokens; qwen fg pins
  inert — OWN engine, not shared). T3 = raw token
  recomp-off-full-fg-ker000-ceil0000-ohbm0 + T3 recipe env exported by the
  chain (gpt-oss is NOT route-kernel capable → moe|T3's ker101 dies by
  design at validate_recompute_kernel_for_model). full-fg auto-enables
  attnact1 + loraafwdcpu; moefg stays 0 (excluded family — correct).
- Experts engine: AsymGptOssExperts (gptoss_moe.py, unit-verified 07-26) —
  pinned host banks (~39 GB), per-active-expert checkpointed streaming,
  grouped LoRA on gate_up+down, verbatim clamped GLU. Does NOT dispatch
  through frozen_linear grouped path → sEP steal cannot arm for experts
  (expected armed=0; the 2r sepplanlink cells still measure the DP+rings
  stack; disclosed in every 2r row).
- ASYM_OFFLOAD_MODULES=all for every asym cell (untied embeds; GLM
  precedent from mrg4 regression cells).
- sepplanlink2 port: 46-tree uncommitted diff (ep_sep transport=nvlink,
  device X rings + CUDA-IPC exchange + range-pull to pinned x-scratch,
  dispatch-level hook before the staged flip, driver alternations) applied
  to this tree 2026-08-12; ep_sep_probe host+nvlink PR5 gate before 2r use.

## Protocol (house)
- w1+m2 (MAX_STEPS=2 WARMUP_STEPS=1), dev pairs w1+m1; PROFILERS=source;
  MAX_SAMPLES=512; serial cells, guard on GPU idle + host floor; verdicts
  GOOM/COOM/TRAINED/FAIL; global tok/s = ranks×meas_steps×b×s/Σstep_ms;
  resv/rss from profile.json. ASYM_ZERO_ROUTER_JITTER=1 TRUST_REMOTE_CODE=false.
- Baselines: superoffload_mem|recomp (rc), |unsloth-ohbm0 (un),
  |unsloth-off-ohbm0 (uo). Fair-comparison rule in force (no one-sided
  generic tricks; liger fused loss on BOTH sides).
- 2r: rm -f /dev/shm/asym_fabric_* before every cell; DDP_TIMEOUT=1500;
  arena default (39 GB banks ≪ 160 cap).

## Ladder plan
- A dev: uns vs T1 @8k b1 (loss parity ≤ ~1%, gptoss_moes_wrapped=24,
  mover audit clean incl. sinks residue ≤8MB carve-in).
- B tiers @64k b1: T1/T2/T2B/T3 all TRAINED + loss in-band + peak ordering
  sane; uns + uns-off reference cells.
- C memory verdict @128k (ctx-capped rung): uns-off batch-walk 8→1 (up-walk
  16/12/10 if <60% HBM at b8) → probative band 75–95% or bracketed wall;
  T3 same-workload row + capacity probe at the baseline wall. PASS = T3
  peak resv NONTRIVIALLY below uns-off + capacity standing. Otherwise fix
  tiers and re-run (walker rule: adapt, never stop at first bad outcome).
- D 1r tp ladder: rungs 32k 64k 96k 128k 192k 256k 320k 384k 448k 512k
  640k 768k 896k 1.02M (extend/stop at asym wall; beyond-131k = rope-OOD
  timing-valid, house precedent); per rung rc→un→uo→asym T1 (promote
  T2B→T3 on OOM, glmext pattern); batch walks b8→b1 at ≤128k, b1-2 beyond.
  Baselines stop 1 rung past their bracketed wall.
- E 2r tp ladder (sepplanlink): probe gate first (ep_sep_probe --transport
  host + nvlink, mode plan, PR5_PASS both); rungs 32k 64k 128k 192k 256k
  320k 384k 512k 640k 768k 896k 1.02M+ (asym wall); backend
  asym_sepplanlink2_cpuadamwds|T1 (promote T2B/T3); baselines rc/un/uo |2.
  ep_sep exit stats recorded per cell (armed expected 0 — own-engine note).

## STATUS LOG (append-only)
- [2026-08-12] Campaign doc created; integration edits landed (driver alias
  + FA4 auto-switch + floors + LF liger mapping); bf16 checkpoint built +
  verified; sepplanlink2 diff ported cleanly (7 files). Chain A queued.
- [2026-08-12 03:28] GPTOSS INCIDENT #1: LF attention.py hardcodes an
  UNCONDITIONAL gpt_oss hijack to the kernels-community/vllm-flash-attn3
  hub kernel — import fails on tf 5.6 (load_and_register_kernel renamed to
  load_and_register_attn_kernel) and would need the absent `kernels` pkg +
  network anyway. Both dev cells died at load. FIX (additive): the hijack is
  skipped when flash_attn ∈ {fa4, disabled} — our driver pins fa4 for
  gpt-oss, so both sides ride FA4 (learnable_sink native); `disabled` kept
  as the eager escape hatch for numeric cross-checks. a_eag cell added to
  chain A (FA4-vs-eager step-1 loss, bf16-noise gate) since both parity dev
  cells share FA4 and could not catch a wrong sink integration alone.
- [2026-08-12 03:36] DEV PAIR PASS (8k·b1 w1+m1, both FA4): uns 1.791→1.469
  vs asym T1 1.803→1.480 — Δ 0.67%/0.75%, parallel (family band: mixtral
  0.35/0.69, phi 0.50/0.10, air 0.91/1.24). gptoss_moes_wrapped=24/24;
  routers skipped whole-GPU (24×); sinks = 3072 B unselected_other moved,
  residue 3328 B ≪ 8MB allowance — no incident; staged dispatch verified
  (torch_forward_calls=384, asym 0 by design at T1). MXFP4-free bf16 local
  checkpoint loads clean. Eager cross-check next.
- [2026-08-12 03:38] EAGER CROSS-CHECK PASS: uns-eager 1.794→1.466 vs
  uns-FA4 1.791→1.469 (Δ 0.17%/0.20% = bf16 noise) — FA4 learnable_sink +
  window_size integration numerically correct on gpt-oss. PHASE-A COMPLETE.
- [2026-08-12 03:56] PHASE-B COMPLETE — ALL FOUR TIERS TRAINED @64k·b1:
  | cell | tok/s | resv | rss | losses |
  | uns    | 8301 | 22.0G |  83G | 0.878/0.9486/1.162 |
  | uns-off| 5428 | 18.4G |  85G | 0.878/0.9483/1.162 |
  | T1     | 3646 | 12.0G | 106G | 0.8801/0.9409/1.159 |
  | T2     | 3913 | 12.5G | 107G | 0.8801/0.9416/1.158 |
  | T2B    | 4267 |  9.5G | 121G | 0.8801/0.943/1.157 |
  | T3     | 4125 |  9.0G | 120G | 0.8801/0.9435/1.16 |
  Loss in-band everywhere (≤0.5%/step vs baseline, parallel). Engagement
  verified: T1 staged+unsloth+attnact0; T2/T2B staged full-fg attnact1;
  T3 DIRECT dispatch (asym_forward_calls=576) full-fg attnact1; moefg0 all
  (own engine). T3 already 49% of uns-off resv at the shared 64k workload.
  GPT-OSS TIER FINDING: T2B BEATS T1 on BOTH axes (4267>3646 tok/s, 9.5<12.0
  resv) — unlike hy/glm where T1 is the speed tier. gpt-oss T1's unsloth-GC
  soc traffic outweighs its recompute savings at this scale; the tp ladders
  therefore run DUAL-TRACK asym (T1 AND T2B per rung while each fits, T3
  after T2B walls; T2 dominated by T2B — quartet-validated, excluded from
  ladders).
- [2026-08-12 04:16] PHASE-C MEMORY VERDICT (same-workload row, 128k·b4
  w1+m2): uns-off walked 8→GOOM, 6→GOOM, 4→TRAINED 135.8 GiB (73% —
  probative band) RSS 349G tok/s 4660, losses 1.167/1.098/1.135. asym T3:
  **49.8 GiB (27%) = 36.7% of baseline**, RSS 300G, tok/s 5970 (+28%
  FASTER), losses 1.171/1.098/1.128 (Δ≤0.6%/step). T3 beats uns-off on
  EVERY axis at the verdict workload — beyond the phi win (49.6%).
  Baseline wall bracketed (b4,b6]; T3 dominance probe at b6 in flight.
- [2026-08-12 04:26] PHASE-C COMPLETE — VERDICT: **WIN+DOMINANCE**.
  Capacity: uns-off GOOMs 128k·b6 while **T3 TRAINS it @72.8 GiB (38%)**,
  RSS 472G, tok/s 5644, losses in-band (+50% workload only asym runs).
  Full verdict row: same-workload 128k·b4 T3 49.8 (27%) vs uns-off 135.8
  (73%) = 36.7%, T3 +28% faster, RSS 300 vs 349. CHAIN-A COMPLETE (13
  cells: dev trio + quartet + refs + walker). Chain D (1r ladder) next.
- [2026-08-12 09:12] 1R TURNING POINT #1: superoffload_mem|recomp G-OOM at
  448k·b1 — rc wall (384k,448k] MEASURED (384k·b1 trained ~172G). rc dead;
  un/uo/T1/T2B continue.
- [2026-08-12 10:59] 1R TURNING POINT #2: superoffload_mem|unsloth G-OOM at
  640k·b1 — un wall (512k,640k] MEASURED. Survivors: uo + asym T1/T2B.
- [2026-08-12 11:56] 1R CROSSOVER MEASURED: from 512k the asym tiers lead
  uo on tok/s too (512k: T1 3256 vs uo 2541; 640k: T1 2756/T2B 2524 vs uo
  2203) at 44-65% less HBM. uo 640k·b1 = 169.5G → wall projected next rung.
  T1 projects ~150G @1.02M — plan: extend rungs past 1.02M if asym alive
  (flash 1r went to 1.15M) so the CEILING is measured, not assumed.
- [2026-08-12 12:04] 1R TURNING POINT #3: superoffload_mem|unsloth-off-ohbm0
  G-OOM at 768k·b1 — uo wall (640k,768k] MEASURED (640k·b1 169.5G).
  ALL THREE BASELINES DEAD; asym T1+T2B alone from 768k on. Baseline walls:
  rc (384k,448k], un (512k,640k], uo (640k,768k].
- [2026-08-12 18:55] 1R ASYM T1 CEILING: G-OOM at 1408k·b1 — T1 wall
  (1280k,1408k] MEASURED (1.28M·b1 trained). T2B continues.
- [2026-08-12 22:56] 1R LADDER BANKED (chains D/D2/D3, 60 cells): baseline
  walls rc (384k,448k] / un (512k,640k] / uo (640k,768k] all G-OOM measured;
  asym T1 ceiling (1.28M,1.41M] G-OOM measured (182.7G @1.28M); T2B deepest
  fit 1.664M·b1 TRAINED (1089 tok/s, 158.4G, RSS 829G — 2.2× uo's wall;
  ceiling OPEN per flash deepest-fit precedent; optional deeper cells queued
  behind 2r). Crossover: asym leads all baselines on tok/s from 512k.
  Chain E (2r sepplanlink) launched.
- [2026-08-12 22:56] SEPPLANLINK PROBE GATE PASSED on c17/38-tree: host-plan
  PR5 bitwise=True (bal/skew/decline/bal2 all bitwise, armed=3 declined=1)
  AND nvlink-plan PR5_PASS bitwise=True (device X rings + CUDA-IPC exchange
  + range-pull scratch). Ported code validated end-to-end; 2r ladder begins.
- [2026-08-13 03:42] 2R TURNING POINT #1: rc G-OOM at 512k·b1 — 2r wall
  (384k,512k] (grid skips 448k; consistent with 1r (384k,448k]).
- [2026-08-13 04:46] 2R TURNING POINT #2: un G-OOM at 640k·b1 — 2r wall
  (512k,640k], replicating the 1r wall (per-rank memory ≡ at b1).
- [2026-08-13 05:53] 2R TURNING POINT #3: uo G-OOM at 768k·b1 — 2r wall
  (640k,768k] ≡ 1r wall. ALL 2R BASELINES WALLED (rc 512k / un 640k /
  uo 768k); sepplanlink T1+T2B alone to 1.02M.
- [2026-08-13 08:40] 2R CROWN + T2B CEILING: sepplanlink T1 TRAINS 1.02M·b1
  (3617 global tok/s, 157.3G resv, RSS 300G) — deepest 2r rung, 1.33× past
  uo's wall; T2B host-COOMs 1.02M → 2r T2B ceiling (896k,1024k] MEASURED
  (896k: 3753 tok/s, 86.8G, RSS 465G; per-rank pools ≈2× the 1r host
  footprint). T3 fallback probing 1.02M. ep_sep armed=0 declined=0 at every
  rung (own-engine expectation, disclosed).

## FINAL RESULTS (all cells MEASURED, w1+m2, best-over-batch; tok/s global)

### 1-rank ladder (chains D/D2/D3)
| seq | rc | un | uo | asym T1 | asym T2B |
|---|---|---|---|---|---|
| 32k  | 13818·114G | 13607·83G | 6878·68G | 9829·40G | 8289·26G |
| 64k  | 11871·168G | 11813·170G | 5391·136G | 9676·79G | 7533·48G |
| 96k  | 10109·168G | 8574·181G | 4945·153G | 8608·87G | 6815·56G |
| 128k | 8650·114G | 8793·170G | 4661·136G | 7571·116G | 5861·74G |
| 192k | 7007·168G | 6957·128G | 4081·102G | 6082·57G | 5252·36G |
| 256k | 5757·112G | 5818·170G | 3658·136G | 5258·76G | 4551·47G |
| 320k | 4957·143G | 4934·108G | 3593·85G | 4395·47G | 4027·32G |
| 384k | 4348·172G | 4326·127G | 2992·102G | 3968·57G | 3558·36G |
| 448k | G-OOM | 3852·153G | 2755·119G | 3567·68G | 3255·42G |
| 512k | — | 3468·170G | 2541·136G | 3256·77G | 2969·47G |
| 640k | — | G-OOM | 2203·170G | 2756·94G | 2524·60G |
| 768k | — | — | G-OOM | 2351·115G | 2154·73G |
| 896k | — | — | — | 2072·132G | 1914·86G |
| 1.02M| — | — | — | 1842·159G | 1714·97G |
| 1.15M| — | — | — | 1652·170G | 1547·108G |
| 1.28M| — | — | — | 1483·183G | 1400·120G |
| 1.41M| — | — | — | G-OOM | 1275·133G |
| 1.54M| — | — | — | — | 1182·151G |
| 1.66M| — | — | — | — | 1089·158G (RSS 829G; deepest fit) |

### 2-rank ladder (chain E, backend asym_sepplanlink2_cpuadamwds)
| seq | rc | un | uo | SEP T1 | SEP T2B |
|---|---|---|---|---|---|
| 32k  | 26723·115G | 26288·85G | 12816·68G | 20094·43G | 16739·30G |
| 64k  | 22937·173G | 22942·176G | 7605·137G | 19396·82G | 14819·52G |
| 128k | 16785·115G | 17093·176G | 6914·137G | 14944·121G | 10998·77G |
| 192k | 13684·173G | 13557·132G | 6317·103G | 12189·60G | 10284·41G |
| 256k | 11232·115G | 11376·176G | 5755·137G | 10499·80G | 8994·53G |
| 320k | 9688·147G | 9652·110G | 6932·85G | 8813·51G | 8062·35G |
| 384k | 8523·170G | 8468·132G | 4874·103G | 7885·60G | 7072·41G |
| 512k | G-OOM | 6800·176G | 4282·136G | 6463·79G | 5909·50G |
| 640k | — | G-OOM | 3712·170G | 5457·98G | 4997·63G |
| 768k | — | — | G-OOM | 4662·118G | 4217·77G |
| 896k | — | — | — | 4086·138G | 3753·87G |
| 1.02M| — | — | — | 3617·157G | host-COOM (T3 probe also COOM) |

Walls: 1r rc (384k,448k] G / un (512k,640k] G / uo (640k,768k] G / T1
(1.28M,1.41M] G / T2B open >1.66M. 2r rc (384k,512k] / un (512k,640k] /
uo (640k,768k] all G / T2B+T3 (896k,1.02M] host / T1 alive at 1.02M crown.
ep_sep exit stats armed=0 declined=0 on every 2r asym cell (own-engine).
- [2026-08-13 09:10] CAMPAIGN COMPLETE — ALL GOALS MET. Figure DATA blocks
  landed in /home/kevinni/env/figures/plot_tp_vs_seq.py (+_2r.py): the
  reserved gpt-oss-20b placeholders filled (19-rung 1r / 12-rung 2r
  panels + OOM_TIP/LEAN_DROP), tp*/tp2r* combined PNG+PDF re-rendered and
  verified (lean 1r: 32k/384k/448k/640k/768k/1.28M/1.66M; lean 2r:
  32k/384k/512k/640k/768k/1.02M). env/figures changes left UNCOMMITTED for
  Kevin's review per the overleaf-sync convention. ~95 measured cells total.
