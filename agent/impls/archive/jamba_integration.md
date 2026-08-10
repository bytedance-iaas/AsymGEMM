# AI21-Jamba2-Mini integration (model_integration.md family #7)

User directive (2026-08-08): full asym integration — T1/T2/T2B/T3 all correct,
like the other MoE models in the throughput plots. Gate: the asym memory tier
must achieve LOWER memory than SuperOffload+Unsloth-Off (correctness first),
then complete 1-rank AND 2-rank throughput ladders (turning-point panels),
bank, render, push. Don't stop until achieved.

## Model facts
52B hybrid MoE (Jamba: 32 layers = 4 attention + 28 Mamba-1; MoE every 2nd
layer → 16 sparse layers × 16 experts top-2, expert width 14336, hidden 4096;
16 dense JambaMLP layers; vocab 65536; ctx 262144; NOT tied embeddings; full
per-layer regime at every seq — no Phi-3.5-style switching; transformers 5.6
native `model_type=jamba`, `JambaExperts` = the packed [E,2I,H]/[E,H,I]
layout the shared AsymQwen3Experts engine consumes byte-for-byte).

## What was built (all in the 39 tree)
1. `asym_gemm/training/jamba_moe.py` — family wrapper (mixtral pattern):
   name-gated detectors; block owns hidden_dim/top_k/num_experts; router is a
   bare nn.Linear — routing replicated out-of-place (softmax fp32 → topk, NO
   renorm, no jitter). Unit parity vs HF block: max|diff| = 0.0 (bf16).
2. `asym_gemm/integrations/lf.py` — 8 sites: imports, report field+repr,
   isinstance report chain, exclusion list, candidate scan (jamba_whole), wrap
   branch (profile prefix child `feed_forward`), decoder-layer detector branch
   ({self_attn|mamba, feed_forward, input_layernorm, pre_ff_layernorm}),
   `classify_lf_component`: `.mamba.*`/JambaMambaMixer → "linear_attention"
   (GPU-resident allowed class, qwen3.5-GDN precedent), llama4-router matcher
   excludes bare-Linear routers, router-offload keeps jamba Linear intact
   (DS-V3-gate precedent), mamba mixers included in the linear-attention
   saved-tensor offload walk.
3. Driver (`profile_lora_lf_test_source.sh`): model key `jamba2-mini`, moe
   classifier `*jamba*`, template → chatml (chat_template.jinja is ChatML),
   `is_shared_engine_moe_family_model` += AI21-Jamba2-Mini (moefg under
   full-fg), per-model ASYM_OFFLOAD_MODULES default excludes lm_head+embed
   (see incident 6).
4. `run_lf_lora_sft.sh`: WATCHDOG_FLOOR_GB_BY_MODEL += 50.
5. Fused loss (BOTH sides, fairness): vendored liger `model/jamba.py`
   (generic lce_forward, glm4_moe pattern) + `apply_liger_kernel_to_jamba`
   in the container liger's monkey_patch (class-level, DS-safe) + LF resolver
   entry + `_LOSS_ONLY_SUPPORTED_MODEL_TYPES` += jamba +
   `_ASYM_LIGER_GENERIC_MOE_MODEL_TYPES` += jamba (instance bridge for asym).
6. Deps: mamba-ssm 2.3.2.post1 built in asym_sft_42 (causal-conv1d present);
   weights at HF cache (snapshot 24cbbd23).
7. T3 token for jamba = raw `recomp-off-full-fg-ker000-ceil0000-ohbm0`
   (ker101 is qwen3-shape-gated; GLM/mixtral precedent).

## Incidents (chronological, each fixed+verified)
1. Watchdog floor unmapped → table entry (50 GB).
2. `.feed_forward.router` captured as Llama4 router (`Linear has no top_k`)
   → bare-Linear exclusion.
3. Router CPU-offload selection raised for the jamba Linear → kept-intact
   skip (mover's router_whole_gpu bucket places it, name-matched).
4. Frozen-residue audit: 5.9 GB `.mamba.*` in "other" → classified
   linear_attention (allowed resident).
5. T3 preset ker101 rejected (qwen-only kernels) → raw ker000 token.
6. MEMORY GATE FAIL №1 (T3 34.2 vs uo 29.7 GiB @32k·b1): breakdown showed
   loss saved 8.39 GB (= 32000×65536 fp32 logits) + CE workspace 10.9 GB —
   liger fused-LCE not engaging (no jamba support anywhere). Fixed by (5)
   above; baseline side verified TRAINED with parity (uo fused 1.2123 vs
   unfused 1.2139).
7. Asym-side FLCE crash (CUBLAS_STATUS_EXECUTION_FAILED in grad @ weight):
   ASYM_OFFLOAD_MODULES=all hosts lm_head RAW (no staged AsymFrozenLinear
   wrap exists for jamba) → CPU pointer into cuBLAS. Fixed by per-model
   selection default (lm_head+embed resident, ~1 GB — conservative against
   asym). Gate retry in flight.

## Measured so far (8k·b1 smokes; rc loss ref 1.7432 @ 23.2 GiB / 8.9 s)
T1 1.7405 · 5.4-5.7 s (−38%) · 14.1 GiB — T2 1.7419 · 13.2 — T2B 1.7425 ·
13.6 — T3 1.7408 · 12.7. All parity Δ≤0.003.

## Next
Gate verdict (T3-fused vs uo-fused @32k) → tier re-smokes w/ fused loss →
1r ladder (in-ctx + beyond-ctx turning points) → 2r → bank both DATA dicts →
render → Overleaf.

## MEMORY GATE: PASSED (2026-08-09)
@32k·b1, fused loss both sides, matched configs:
- **asym T3 = 20.5 GiB HBM** (13.5 s/step, loss 1.2137)
- **uns-off = 24.7 GiB HBM** (15.6 s/step, loss 1.2123)
- → **T3 is −17% HBM and −13% step time.** Gate criterion (asym memory tier
  below SuperOffload+Unsloth-Off) satisfied with parity.
Incident 8 (closed): the ASYM instance liger bridge IMAs in
liger_cross_entropy_kernel on Jamba (every tier; CUDA_LAUNCH_BLOCKING
localized it; T1-fused reproduced it) while the CLASS-level vendored applier
runs the same fused math cleanly on both asym and baselines → jamba is
class-path-only (removed from _ASYM_LIGER_GENERIC_MOE_MODEL_TYPES with
rationale; root cause of the instance-path IMA left documented-unresolved —
the class path is the house-preferred DS-safe mechanism anyway).
