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
