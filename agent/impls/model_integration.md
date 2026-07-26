# model_integration — 6 new MoE families for the asym backend (RECORD)

(2026-07-26, user directive. This doc is the living record for the integration
campaign: plan, effort ranking, per-model status. Update the STATUS LOG as
models land. Rule of the campaign, verbatim intent: **replicate the code path
per model** — every new family gets its OWN module files + its own adapter
branch + its own driver entries; ZERO edits inside the existing families
(qwen3_moe*, qwen35_*, llama4_*, dense paths) so nothing already-validated can
break.)

## Directive & build order (user, 2026-07-26)
Integrate, in this order:
1. **Mixtral-8x22B** → 2. **Phi-3.5-MoE** → 3. **Hunyuan-A13B** →
4. **the two GLMs (GLM-4.5-Air then GLM-4.7-Flash — grouped: Flash is a
   structural clone of Air's arch)** → 5. **gpt-oss-120b**.
Order = reverse of the effort ranking below: the vanilla model validates the
replicated-path skeleton, each later model adds ~one new deviation.

## Verified architecture facts (transformers 5.6.0, installed venv, 2026-07-26)
All six are supported HF architectures, and ALL use the packed 3D expert
layout the repo's `packed_moe.py` → `AsymPackedExperts` grouped-GEMM path
already consumes:
- `mixtral`, `phimoe`, `hunyuan_v1_moe`, `glm4_moe` (=4.5-Air),
  `glm4_moe_lite` (=4.7-Flash): `gate_up_proj [E, 2I, H]` + `down_proj
  [E, H, I]` — same orientation as the existing families.
- `gpt_oss`: **transposed** packing (`gate_up_proj [E, H, 2I]`, `down_proj
  [E, I, H]`) **plus per-expert biases** on both projections — the only model
  with either deviation.
- Routers: mixtral/phimoe/hunyuan = softmax top-k (top-2/2/8);
  glm4_moe(+lite) = DeepSeek-V3-style sigmoid + `e_score_correction_bias` +
  group-limited top-k (shielded by our `router_mode=whole` — router stays an
  intact GPU-resident black box); gpt_oss = softmax-after-topk with clamped
  GLU (`gate·σ(1.702·gate)`, clamp limits) instead of silu·mul.
- Shared experts: hunyuan (1), glm4_moe/+lite (yes) — pattern exists
  (`llama4_shared_mlp.py`, `qwen35_shared_mlp.py`). mixtral/phimoe/gpt_oss: none.
- Attention quirks: phimoe + mixtral sliding-window-capable (masking_utils
  handles it); gpt_oss adds **attention sinks** + alternating sliding layers;
  glm4_moe partial RoPE + first-k dense layers; hunyuan qk-norm variant.
- Checkpoint: gpt-oss ships MXFP4 → needs dequant-to-bf16 on load. Others bf16.

## THE EFFORT TABLE (most → least; GLMs grouped per user)

| rank | model (HF id) | total→host bank | key deviations driving effort |
|---|---|---|---|
| 1 | **gpt-oss-120b** (`openai/gpt-oss-120b`) | ~120B → ~240 GB | transposed packed layout; expert BIASES in grouped GEMM + HostWeight bank; clamped-GLU activation (fg kernels assume silu·mul); attention sinks + alternating sliding window; MXFP4 dequant load; harmony template |
| 2 | **GLM-4.5-Air + GLM-4.7-Flash** (`zai-org/GLM-4.5-Air`, `zai-org/GLM-4.7-Flash`) | 106B → ~212 GB; Flash ~30B-class | sigmoid+bias-corrected group top-k router (mostly shielded by router_mode=whole); shared experts; first-k-dense layers; partial RoPE. **Flash (`glm4_moe_lite`) is a line-level structural clone of Air (`glm4_moe`)** — verified same module classes — so it is near-free once Air's path exists (standalone it would rank #2) |
| 3 | **Hunyuan-A13B** (`tencent/Hunyuan-A13B-Instruct`) | 80B → ~160 GB | shared expert + template/tokenizer quirks; otherwise standard packed + softmax top-k |
| 4 | **Phi-3.5-MoE** (`microsoft/Phi-3.5-MoE-instruct`) | 42B → ~84 GB | vanilla packed top-2; sliding window; LongRoPE via config; smallest = fastest validation |
| 5 | **Mixtral-8x22B** (`mistralai/Mixtral-8x22B-v0.1`) | 141B → ~282 GB | most vanilla of all (classic top-2, no shared expert, no router exotica) — least code; only cost is bank size |

## Replication rule (how each model lands)
Per model, NEW files/branches only:
- `asym_gemm/training/<family>_moe.py` (+ `<family>_shared_mlp.py` where the
  arch has shared experts) — cloned from the nearest existing family
  (mixtral/phimoe/hunyuan/glms clone the qwen3_moe/packed_moe consumption
  pattern; gpt-oss additionally extends the packed layout + bias handling in
  ITS OWN module, not in `packed_moe.py`).
- Adapter: a new wrap branch in LF's adapter setup keyed on the HF model type
  (the `*_moes_wrapped` line) — additive, never touching existing branches.
- Driver: new `M[...]` alias in `scripts/lf/profile_lora_lf_test_source.sh`
  (family comment + layer count), TEMPLATE mapping, and a
  `WATCHDOG_FLOOR_GB_BY_MODEL` entry in `run_lf_lora_sft.sh`.
- Tier recipes: current `moe|T*` env bundles are qwen3-moe-tuned
  (`ASYMM_QWEN3_MOE_FG_*` flags are inert for other families). New families
  start on T1 (unsloth-ohbm0+staged) and T2's attention keep-acts only; their
  own fg tiers come later if/when per-family fg paths are written. Note this
  in any capacity claims.

## Common per-model checklist (fixed side-work, every model)
1. HF weights → node-local cache (disk OK: 12 TB free; ~1 TB total for all six).
2. LF chat template registered + dataset builder tokenizer pass
   (`build_lf_sft_eval_pair.py`; fast builder handles any tokenizer).
3. Driver aliases + floors (above). 4. Wrap module + adapter branch.
5. Smoke: rank-1 64k b1 `asym|T1` w1+m2 — loss sanity + `*_moes_wrapped>0` in
   the setup line + HBM/RSS eyeball. 6. Baseline smoke (superoffload uns) for
   a same-seq reference row. 7. Record results in this doc's STATUS LOG.

## STATUS LOG (update as work proceeds)
- [2026-07-26] Doc created. Order locked: mixtral → phimoe → hunyuan →
  glm4.5-air → glm4.7-flash → gpt-oss.
- [2026-07-26] **ALL SIX FAMILY MODULES CODED + UNIT-VERIFIED** (see below).
- [2026-07-26] **Mixtral-8x22B code COMPLETE, unit PASS.**
  Landed (all additive): `asym_gemm/training/mixtral_moe.py` (name-gated
  detectors, AsymMixtralMoeBlock w/ jitter fidelity, wrap fn; engine =
  AsymQwen3Experts per packed_moe precedent — tf-5.6 MixtralExperts carries
  the identical packed layout + router triple); `integrations/lf.py` 8
  additive edits (import, report.mixtral_moes_wrapped + log line, install
  counter, whole-mode candidate check placed BEFORE qwen3's — qwen3's
  structural detector would otherwise capture Mixtral; mixtral_whole install
  branch, decoder-recognizer stanza, LoRA-count exclusion); driver: M[
  mixtral-8x22b] (layers 56), infer_template mixtral-*→mistral,
  tier_model_family *mixtral*→moe, watchdog floor 50. Weights cached (262
  GB). UNIT: wrapped-vs-HF max|Δ| 6.1e-5 bf16 @ asym+offload on GPU,
  LoRA banks live. SMOKE #1: dataset built clean on the mistral template (p50
  71k), 56/56 layers wrapped (mixtral_moes_wrapped=56), 262 GB expert banks on
  host, trained w1+m2 (train_loss 0.7677) — but jobs.tsv failed:127 from an
  OPS foot-gun: I edited the driver/runner scripts WHILE the smoke was
  executing them (bash reads incrementally → offset skew → post-train shell
  garbage). LESSON (now a campaign rule): never edit driver/runner .sh while
  any run is live. NOTE: asym_forward_calls=0 in the runtime line is EXPECTED
  under T1 (ASYM_GEMM_DISPATCH=staged counts as torch calls — the validated
  q3-30b sEP-T2 runs show the same signature). Clean re-smoke (smkmx2-*)
  relaunched with stable scripts.
- [2026-07-26] **Phi-3.5-MoE coded + UNIT PASS** (`phimoe_moe.py`; router attr
  `.router` (nn.Linear sparsemixer — router-name PEFT exclusion covers it),
  ints on block, input_jitter fidelity). max|Δ| = 0.0 vs HF on GPU asym+offload.
- [2026-07-26] **Hunyuan-A13B coded + UNIT PASS** (`hunyuan_moe.py`; gate
  returns logits → block-level softmax-topk replicated verbatim (fp32),
  shared_mlp kept as original module: GPU-resident, standard PEFT LoRA, grads
  verified flowing; classifier got additive ".shared_mlp" → shared_experts
  attribution). max|Δ| = 6.1e-5.
- [2026-07-26] **GLM-4.5-Air + GLM-4.7-Flash coded + UNIT PASS** (`glm45_moe.py`
  / `glm47_moe.py`, generated from ONE template — Flash verified line-identical
  routing; name gates: air = glm4moe∧¬lite, flash = glm4moelite, lite
  dispatched first). DS-V3 sigmoid+bias group top-k replicated verbatim under
  no-grad; shared_experts kept dense/PEFT. max|Δ| = 6.1e-5 / 3.1e-5. Template
  mapping: glm-4.5*/4.7* → LF `glm4_moe` template.
- [2026-07-26] **gpt-oss-120b coded + UNIT PASS** (`gptoss_moe.py` — OWN
  engine-light expert class, NOT the shared engine: pinned host banks for the
  4 transposed/biased tensors; per-active-expert compute under non-reentrant
  checkpoint with weights fetched no-grad INSIDE (backward re-streams,
  autograd never retains bases); grouped LoRA (kaiming A/zero B) on gate_up +
  down; verbatim interleaved clamped-GLU. Block preserves the HF
  (hidden, router_scores) tuple return. Forward bit-exact (Δ=0.0) vs HF; dX +
  LoRA grads verified through the checkpointed streamed path. No tuned
  kernels yet — cuBLAS-on-streamed (T1-class), tuning is follow-up work.
  MXFP4 load path unverified until its smoke (expect transformers auto-dequant
  on this ARM node).
- [2026-07-26] lf.py cumulative wiring: 6 family imports, 6 report counters +
  log-line fragments, 6 install-counter branches, whole-mode dispatch order
  qwen35 → mixtral → phimoe → hunyuan → glm47 → glm45 → gptoss → qwen3 →
  llama4 (new families name-gated so qwen* can never be captured; qwen3's
  purely-structural detector runs AFTER the name-gated ones because it WOULD
  match Mixtral/GLM shapes), 6 decoder-recognizer stanzas, LoRA-count
  exclusions. Qwen3 regression probe: detector intact, zero cross-capture.
- [2026-07-26] Re-smoke #2: SAME foot-gun, second instance — the gpt-oss
  driver wiring landed while the re-smoke was mid-flight (train again healthy:
  loss 0.7661 ≈ #1's 0.7677; only the post-train shell corrupted). Scripts are
  now FINAL; mixtral verdict run (smkmx3) queued BEHIND the family chain with
  a hard no-edit freeze in force.
- [2026-07-26] Serial smoke chain launched for the 5 remaining families
  (phi → hunyuan → glm4.7 → glm4.5 → gpt-oss; each waits for its HF download,
  builds its 64k dataset on its own template, runs rank-1 64k b1 asym|T1).

## VALIDATION PLAN (user-approved protocol, 2026-07-26)
Two proofs per family, run as A/B pairs — reference = `superoffload_mem|unsloth`
(pure HF native classes, TRUST_REMOTE_CODE=false) vs `asym_cpuadamwds|T1` —
same seed, same dataset, same steps:
1. **Loss parity** at the dev workload — step-1 loss must match to bf16 noise
   (LoRA starts B=0 ⇒ identical math modulo streaming order); trajectory in-band.
2. **Memory benefit** at the validation workload — LEAN-vs-LEAN pair (user,
   2026-07-26): baseline = `superoffload_mem|unsloth-off-ohbm0` (its most
   memory-lean config) vs `asym_cpuadamwds|T3` (tier preset expands via
   tier_recipes.sh; all six families map to the moe family). Peak reserved
   HBM + host RSS; expect asym ≪ baseline and/or fits-where-baseline-OOMs.
   CAVEAT (keep in claims): new-family T3's qwen3-specific fg flags/ker101
   are inert — their T3 = generic recompute + save-on-cpu + asym expert
   streaming; family fg kernels are follow-up work.
STEP PROTOCOL (user, verbatim): **dev = 1 warmup + 1 non-warmup**
(WARMUP_STEPS=1 MAX_STEPS=1); **validation = 1 warmup + 2 non-warmup**
(WARMUP_STEPS=1 MAX_STEPS=2).

### THE RUN TABLE (fill loss/verdict as pairs land; mirror of the user table)
| family | module | dev workload (loss parity) | validation workload (memory proof: uns-OFF vs T3) | loss | verdict |
|---|---|---|---|---|---|
| Mixtral-8x22B | mixtral_moe.py | 8k·b1 w1+m1 | 64k·b2 w1+m2 (ctx-capped 65k; est uns ~120 GiB) + max-batch probe | — | — |
| Phi-3.5-MoE | phimoe_moe.py | 8k·b1 w1+m1 | 128k·b3 w1+m2 (est uns ~140 GiB ≈76%) | — | — |
| Hunyuan-A13B | hunyuan_moe.py | 8k·b1 w1+m1 | 32k·b12 w1+m2 (ctx-capped 32k) | — | — |
| GLM-4.5-Air | glm45_moe.py | 8k·b1 w1+m1 | 128k·b2 w1+m2 (est uns ~132 GiB ≈72%) | — | — |
| GLM-4.7-Flash | glm47_moe.py | 8k·b1 w1+m1 | 192k·b2 w1+m2 (ctx 202k) + b3 probe | — | — |
| gpt-oss-120b | gptoss_moe.py | 8k·b1 w1+m1 | 128k·b4 w1+m2 (est uns ~148 GiB ≈81%; MXFP4 load verified at dev) | — | — |
Sizing basis: each config's ctx cap + H×L activation slope anchored on the
measured q3-30b uns line; probe rule applies (±1 batch/seq rung on OOM/slack).
Order: mixtral → phi → hunyuan → glm4.7 → glm4.5 → gpt-oss; frozen-script
discipline (no driver edits while any run lives).

### VALIDATION WALKER RULE (user, 2026-07-26 — DO NOT GIVE UP ON FIRST FAILURE)
The validation workloads above are FIRST-CUT ANCHORS, not fixed targets. The
agent running a validation pair must ADAPT until the workload is probative,
never stop at the first bad outcome:
- **both sides G-OOM** → step DOWN one rung (batch −1, or seq −1 grid step if
  b1) and re-run; repeat until the baseline side fits or its wall is bracketed.
- **baseline OOMs, asym fits** → that IS a result (capacity dominance): record
  it, then also step down once so a same-workload HBM comparison exists too.
- **HBM too low on the baseline side** (< ~60% peak reserved — the comparison
  is unprobative in the flat region) → step UP one rung (batch +1, or next seq
  grid step) and re-run until baseline lands ~75-95% or OOMs (then rule 2).
- **host C-OOM** on either side → seq/batch down one rung (host walls move
  first at long seq; note which side hit host — that asymmetry is itself data).
- Same walker semantics as every capacity campaign: adjacent-rung bracketing,
  probe rule near edges, clean-shm guard between runs, every probe recorded in
  the STATUS LOG (including the failures — they are the wall measurements).
The pair is DONE when: loss column filled (dev) AND a same-workload memory row
exists with baseline in the probative band or a bracketed baseline wall plus
asym's standing at that wall. Only then move to the next family.
