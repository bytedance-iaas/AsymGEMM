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

## WAVE-1 COMPLETE (2026-07-26 ~08:3x) — stopped before GLM per user
Mixtral: loss PASS (0.35%/0.10%) + memory PASS+DOMINANCE. Phi: loss PASS
(0.50%/0.10%) + memory PARITY-42B (no win; honest). Hunyuan: loss =
parallel-offset finding (user to judge; engine-numerics decision pending) +
memory BASELINE-WINS under generic-T3 (family-fg chunking = fix path).
Fixes landed en route (all additive): TRUST_REMOTE_CODE plumb, jitter-zero
hook, router mover+audit carve-outs, tied-embed audit allowance, hunyuan
z3-leaf + liger registrations. GLM/gpt-oss remain (pre-flagged: liger
mappings + big-vocab). GPUs idle, shm clean.

## PHI T3 MEMORY CAMPAIGN (2026-07-27, user directive: keep iterating until
## T3 peak is WELL below the uns-off baseline 149.9 GiB)
DIAGNOSIS (from vl-t3 run artifacts, memory_breakdown.md):
- pLoRA split path WAS correct (loraafwdcpu; GC save-on-cpu active — layer
  saves tiny). NOT the problem.
- Peak 158.6 GiB = **loss path 79.6 GiB (50%: 45.9 saved logits + 33.7 CE
  workspace — Asym Liger bridge never installed for phimoe; loss runs OUTSIDE
  GC coverage so soc can't touch it)** + 37.7 routed-experts workspace
  (expact0) + ~15.5 norms/attn workspace + 12.0 allocator slack.
FIXES (additive, liger_loss.py + frozen_linear.py):
- Generic-MoE Liger bridge: `_ASYM_LIGER_GENERIC_MOE_MODEL_TYPES` (all 6 new
  families) + `asym_generic_moe_causal_lce_forward` (duck-typed hidden-state,
  aux/router loss not computed — tf-5.6 drops router-logit threading; wave-1
  parity A/Bs confirmed) + installer; umbrella dispatch extended.
- lm_head BIAS support (phi is the one family with real lm_head.bias): liger's
  functional accepts bias; threaded via LigerForCausalLMLoss kwargs;
  `_resolve_liger_lm_head_bias` reads bias/bias_cpu; staged-resolver
  `asym_liger_lm_head_weight(allow_bias=)` bypass (old bridges keep refusing
  so nothing can silently drop bias).
- UNIT: bridged loss bit-exact (0.0000%) vs unbridged on phimoe(+bias),
  mixtral, AND phimoe with staged AsymFrozenLinear lm_head + bias_cpu.
ITERATION LADDER (128k·b3, w1+m2): A = T3+bridge; B = +ASYMM_EXPERT_ACT_OFFLOAD
(targets the 37.7 GiB packed gate-up workspace). Continue to next-largest
consumer until goal met. (Run-1 false start: staged resolver's bias gate —
fixed above, relaunched.)

### PHI T3 LADDER RESULTS (running log)
- Run A (T3+generic bridge): **115.2 GiB (was 158.6, −43.4; baseline 149.9 →
  asym now −34.7 BELOW baseline)**, RSS 511, lw 1.0619 (in-band), bridge=yes.
  New peak anatomy: routed_experts recompute workspace 48.0 GiB (42%),
  allocator slack 31.6 (27%), norms ws 9.7, attn ws 8.4.
- Runs B/C/D: three ways expact/moefg DIDN'T engage (env clobbered by driver;
  spec expact collapsed by recipe — principled: recomp-off retains no expert
  saves, the 48 GiB is BACKWARD-recompute transient; env moefg reset by the
  full-fg stage's qwen3-only gate). Lesson: this driver's recipe layer OWNS
  policy envs — flip them at the recipe/token layer, not via env.
- FIX for the 48 GiB: full-fg's moefg enablement extended to shared-engine
  families via new `is_shared_engine_moe_family_model` (mixtral/phi/hunyuan/
  glm45/glm47; NOT gpt-oss) — deliberately NOT widening
  is_qwen3_moe_routed_model (that would unlock qwen3-shape-tuned ker101).
  lf.py: the 4 family install branches now mirror qwen3's
  `_qwen3_moe_finegrained_enabled` per-instance engine flag (fg is
  family-agnostic engine code; unit: fwd Δ=0.0 + train grads at fg-compliant
  dims; real phi 4096/6400 satisfies the ×64 constraint).
- Run E in flight: token-driven moefg1 (fg bounded-chunk expert recompute).

### RUN-E RESULT + OPS LESSON (2026-07-27)
Run E (token-driven moefg1): **75.6 GiB peak (from 115.2; baseline 149.9 —
now HALF the baseline)**, RSS 563, all 3 steps trained, losses in-band
(1.0619/1.0303/1.1070), fg fully engaged (asym_forward_calls=4032,
torch_forward_calls=0). jobs.tsv verdict was failed:1 from the PROFILE
COMPLETENESS gate, not the run — root cause under investigation on the
rerun's artifact. OPS LESSON: NEVER re-invoke the driver on an existing
RUN_NAME to "revalidate" — an incomplete-judged profile is DELETED and
rerun; a short-timeout kill then destroys the artifacts (run E lost; numbers
preserved here; vl5f is the clean reproduction).

### PHI T3 CAPACITY FLIP (2026-07-27)
- vl5f reproduction: **75.6 GiB deterministic** (twice), RSS 563, loss in-band.
- **CAPACITY DOMINANCE: T3 TRAINS 128k·b4 @100.6 GiB (55%), RSS 567** — the
  rung where BOTH sides GOOM'd pre-fix and uns-off still GOOMs. b5 probing.
- Phi memory verdict upgrades: PARITY → **WIN** (same-workload 128k·b3: 75.6
  vs 149.9 = **49.6% of baseline**; capacity: b4 @100.6 AND b5 @125.8/RSS 734
  — +67% workload beyond the baseline wall; b6 not probed, host ~900 est).
- Verdict-gate mystery ONGOING (does not affect the numbers): driver marks
  these runs failed:1 via job_profile_complete while identical-argv replays
  and write-time watchers all PASS (schema+epc=0 at profile-write). argv
  capture via ENV_PYTHON shim queued (acceptance path on a sacrificial COPY —
  never revalidate in place, see OPS LESSON).

### VERDICT-GATE MYSTERY CLOSED (2026-07-27)
ENV_PYTHON-shim argv capture nailed it: the completeness heredoc had a THIRD
qwen3-family hardcode — `qwen3_moe_target = "Qwen3-30B-A3B" in model_name` →
expected moefg=false for phi while the run truthfully recorded true. (The
gate swallows stderr with >/dev/null 2>&1, which is why three replay attempts
couldn't see it; the shim override ENV_PYTHON=... is the reusable trick.)
Fixed: heredoc extended with the shared-engine family list (mirrors
is_shared_engine_moe_family_model). vl5f/b4/b5 re-accepted by the driver
("Skipping existing" + skipped record) on backed-up copies-first protocol —
no GPU time, no artifact loss. PHI T3 MEMORY CAMPAIGN COMPLETE.

## GLM LEG (2026-07-28, user go: target = asym T3 beats uns-off on memory at
## a big workload)
- PRE-WORK LANDED: baseline-side fused loss for glm4_moe/glm4_moe_lite. The
  vendored Liger ships NO class applier for the text GLM MoEs (only glm4
  dense/glm4v/glm4v_moe) — instead of writing one, LF adapter.py's post-load
  instance-bridge hook (the llama4 site, adapter.py:571) now also fires for
  glm4_moe + glm4_moe_lite: BOTH sides run the identical asym generic-MoE
  bridge → parity by construction, and the 151k/155k-vocab logits stop
  dominating the baseline's peak (hunyuan incident-#4 class fix, instance
  flavor). UNIT: bridged-vs-raw loss on tiny real-config models — glm4_moe
  rel Δ 0.00e+00 (bit-exact), glm4_moe_lite 1.55e-07 (fp32 noise).
- tie_word_embeddings=False both → no hunyuan-style offload carve-out needed.
- CHAIN (glm_chain.sh, serial GPU0): dev pairs 8k·b1 w1+m1 (uns vs T1, both
  models) → Flash 192k·b2 uns-off-vs-T3 pair + b3 probes → Air 128k·b2 pair
  + b3 probes. Walker: batch down-walk on OOM in-chain; band judgment after.
- GLM INCIDENT #1 (instance bridge × ZeRO-3): first approach installed the
  asym generic bridge on the BASELINE instance (adapter.py llama4 hook) —
  faults in liger FLCE's chunk GEMM (CUBLAS_STATUS_EXECUTION_FAILED →
  illegal address) under the DS ZeRO-3 baseline. Class-level patching is the
  mechanism every validated baseline uses (qwen3_moe/hunyuan/llama) →
  REPLACED with vendored-fork appliers `apply_liger_kernel_to_glm4_moe` /
  `_glm4_moe_lite` (shared loss-only lce_forward in
  Liger-Kernel/src/.../model/glm4_moe.py; tf-5.6 minimal signature, no
  router-logit threading) + LF resolver entries. UNIT (via LF's own
  loss-only kwargs builder): glm4_moe bit-exact, lite 1.55e-07. adapter.py
  hook reverted to llama4-only with a comment.
- GLM INCIDENT #2 (Flash = MLA attention): glm4_moe_lite uses DeepSeek-style
  MLA (q_a/q_b, kv_a_proj_with_mqa/kv_b_proj) — the doc's "Flash is a
  structural clone of Air" held for ROUTING only; attention differs (Air is
  standard GQA q/k/v/o). The strict CPU-first mover audited MLA projections
  as unclassified 'other' residue → asym dev FAIL. Fix (additive, lf.py):
  `_MLA_ATTENTION_TARGETS` set; classifier + `_is_text_attention_projection_
  name` + `_is_text_attention_module_name` accept MLA names (kept OUT of
  `_ATTENTION_TARGETS` — module recognizers require ALL members as children);
  profiled-train activation-context key strips MLA suffixes. Unit: classifier
  cases + std/MLA module recognizers pass; std behavior unchanged.
  FOLLOW-UP: `_build_attention_activation_contexts` still requires the
  q/k/v triple → attn-act offload silently no-ops for MLA models (safe
  degradation; Flash hidden=2048 so the lever is small there). MLA-aware
  attn-act = future work if Flash T3 needs it.
- GLM INCIDENT #3 (DS-V3 router swap dropped the bias): `_wrap_lf_router_
  module` replaces any `*.mlp.gate` with the projection-only AsymQwen3Router
  (2D-weight check accepted GLM's Glm4MoeTopkRouter by accident), dropping
  the `e_score_correction_bias` buffer the block's verbatim DS-V3 routing
  reads → Air asym dev crashed at first routing forward. (The Jul-26 unit
  wrapped raw HF blocks and never saw the swap.) Fix (additive, lf.py): the
  swap now SKIPS gates carrying `e_score_correction_bias` — original router
  stays intact, matching the whole-mode "router = intact GPU black box"
  design; mover's router_whole_gpu param bucket + buffer pass place weight
  AND bias on GPU. Unit: DS-gate kept intact, plain gate still swapped.
  Dev-cell reruns queued behind the main chain (g3dev47a Flash, g3dev45a
  Air); the main chain's validation cells import the fixed module.
- **AIR VALIDATION RESULT (2026-07-28 23:42, main chain)**: at 128k·b2 —
  uns-off 131.0 GiB (71%)/RSS 750 vs **T3 121.5 GiB (66%)/RSS 886** → T3
  leaner on HBM (92.7% of baseline). Loss parity AT THE VALIDATION WORKLOAD:
  1.309/1.502/1.372 (uns-off) vs 1.31/1.497/1.359 (T3) — Δ ≤ 0.9%/step.
  **CAPACITY DOMINANCE: uns-off host-COOMs at 128k·b3 (wall (b2,b3], RSS
  750@b2) while T3 TRAINS b3 @176.4 GiB (96%)/RSS 892, losses in-band
  (1.355/1.427/1.406)** → +50% workload only asym runs. fg fully engaged
  (45/45 fg-wrapped, asym_forward_calls=5442). T3 b2 peak anatomy: 85.1 GiB
  transient workspace at after_backward (Air's 96-head attention backward =
  the dominant transient; saved acts 0 — soc active) — attention-backward
  chunking is the recorded next lever if a deeper Air win is ever wanted.
  NOTE both sides show grad_norm ~1e8-1e10 at 128k·b2/b3 (dev-scale norms
  are sane 0.6-1.1): symmetric on A and B → packing/masking artifact of the
  128k dataset, not an asym numerics issue; does not affect the comparison.

## STATUS LOG (update as work proceeds)
- [2026-07-31] **GLM THROUGHPUT-PANEL CAMPAIGN COMPLETE** (~330 cells, both
  GPUs, ~35 h): 4 house-style panels — Flash 1r/2r + Air 1r/2r, >=6 seq
  rungs each, every cell measured solo/serial, zero est. Deliverable:
  `tp_glm_combined.pdf` (2x2) + 4 per-model PDFs in /home/kevinni/env/
  figures/out/ + the overleaf archive figures/. DATA + full per-cell
  provenance comments in plot_tp_vs_seq.py / plot_tp_vs_seq_2r.py; campaign
  record agent/anchors_tmp/GLMTP_CAMPAIGN.md. Headlines: Air-2r asym sdp2
  leads ALL rungs +13-45% (shared-fabric vs ~480 GB/rank replicated host
  machinery; uns-off host-DEAD >=96k); Air-1r asym leads all 6; Flash-1r
  asym leads 4/6 + uns red-OOM 192k; Flash-2r asym leads 5/6. Ops lessons
  recorded in the campaign doc (solo-only throughput cells; sdp2 not
  ep2-vanilla for non-qwen3; ASYM_ARENA_SHM_CAP_GB for >160 GB banks;
  ddp_timeout tiers; buffered-script kill-rewrite-relaunch discipline).
- [2026-07-29] **GLM LEG COMPLETE** (~30 cells): Air = WIN+DOMINANCE, Flash =
  baseline-wins-at-MLA (see run table). Four stack fixes landed en route
  (vendored liger glm appliers; MLA classification; DS-router kept intact;
  MLA selection-parser targets) — all additive, unit-verified, recorded as
  GLM INCIDENTS #1-#3 + the selection fix. Ops incidents: watchdog-escalation
  cross-chain kill (a dying run's sweep killed the next chain's fresh driver
  — leave a settle gap after COOMs); node now SHARED with a concurrent
  campaign (c2cuns192_q3-30b-a3b took the GPU 05:20; Flash asym-dev rerun
  yielded). Follow-ups on record: MLA-aware attn-act offload (the Flash T3
  unlock); intermittent first-forward host transient (mixtral 352k / Flash
  T3-b3 class).
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
| Mixtral-8x22B | mixtral_moe.py | 8k·b1 w1+m1 | 64k·b2 w1+m2 (ctx-capped 65k; est uns ~120 GiB) + max-batch probe | **PASS** Δw 0.35% Δm 0.69% (uns 1.4013→0.9195 vs asym 1.3964→0.9131; learning-step Δ 0.3%; asym 11.0 vs uns 18.5 GiB even at 8k) | **PASS+DOMINANCE**: shared 64k·b2 — T3 host −53 GB (815 vs 868) at +10.8 GiB HBM (80.6 vs 69.8; uns-off is host-pinned); capacity — T3 FIT 64k·b3 @120.6 GiB (66%) where uns-off host-COOM ×3 (+50% workload only asym runs) |
| Phi-3.5-MoE | phimoe_moe.py | 8k·b1 w1+m1 | 128k·b3 w1+m2 (est uns ~140 GiB ≈76%) | **PASS** Δw 0.50% Δm 0.10% (uns 1.2310→0.8953 vs asym 1.2372→0.8962) | **WIN (post fused-loss + fg fixes, 2026-07-27)**: same 128k·b3 — T3 **75.6 GiB vs uns-off 149.9 (49.6%)**, RSS 563 vs 392; capacity — T3 TRAINS 128k·b4 @100.6 (55%) and **128k·b5 @125.8 (68%)/RSS 734** where uns-off GOOMs at b4 (+67% workload only asym runs). [Superseded initial verdict: PARITY — pre-fix T3 was 158.6 with the loss path unbridged and fg dormant] |
| Hunyuan-A13B | hunyuan_moe.py | 8k·b1 w1+m1 | 32k·b12 w1+m2 (ctx-capped 32k) | **PARALLEL-OFFSET** (user to judge): 5-step A/B deltas +3.9/+3.4/+1.8/−0.55/+1.9% — curves parallel, sign-crossing at step 4 ⇒ training-signal equivalent; plus ~1.1% run-to-run nondeterminism WITHIN asym (1.4829 vs 1.4661 same-config reruns; uns is bit-deterministic). Mechanism: engine accumulation-order at top-8/small-expert shapes (0.70%/block probe-isolated) | **BASELINE WINS at 80B/generic-T3**: uns-off 32k·b12 @131.4 GiB (72%)/RSS 725 vs T3 b12 GOOM (23.4 GiB packed gate-up transient), T3 fits only b10 @178.2 (97%)/RSS 693 — no family fg chunking yet; hunyuan-fg = the recorded fix path |
| GLM-4.5-Air | glm45_moe.py | 8k·b1 w1+m1 | 128k·b2 w1+m2 (est uns ~132 GiB ≈72%) | **PASS** dev Δw 0.91% Δm 1.24% (uns 1.534→2.267 vs T1 1.52→2.295, parallel; Air's top-8×128 shapes = sub-hunyuan engine offset); val-workload parity Δ ≤ 0.9%/step (1.309/1.502/1.372 vs 1.31/1.497/1.359) | **WIN+DOMINANCE (2026-07-28)**: same 128k·b2 — **T3 121.5 GiB (66%) vs uns-off 131.0 (71%)** (92.7%), RSS 886 vs 750; capacity — **uns-off host-COOMs 128k·b3 while T3 TRAINS it @176.4 (96%)/RSS 892**, losses in-band (+50% workload only asym runs) |
| GLM-4.7-Flash | glm47_moe.py | 8k·b1 w1+m1 | 192k·b2 w1+m2 (ctx 202k) + b3 probe | **PASS** dev Δw 0.79% Δm 0.25% (uns 1.655→2.444 vs T1 1.642→2.450; g5dev47a, all four MLA/router fixes proven end-to-end, 46/46 wrapped); val-workload parity Δ ≤ 1%/step (192k·b5: 1.321/1.226/1.223 vs 1.322/1.226/1.235) | **BASELINE WINS at MLA/generic-T3**: probative rung 192k·b5 (walker: uns-off b2 28% → b5 128.5 GiB (70%)/RSS 771, wall (b5,b6] host) vs T3 158.8 (86%)/RSS 723; both COOM b6 → capacity tie. Cause: attn-act offload no-ops for MLA (q/k/v-keyed contexts) — T3 fights without its attention lever; MLA-aware attn-act = the recorded fix path. Ref cells: T3 b2 69.2 (38%)/419; T3-b3 attempt COOM'd at load = anomalous host transient (b5 fit supersedes by monotonicity) |
| gpt-oss-120b | gptoss_moe.py | 8k·b1 w1+m1 | 128k·b4 w1+m2 (est uns ~148 GiB ≈81%; MXFP4 load verified at dev) | — | — |
Sizing basis: each config's ctx cap + H×L activation slope anchored on the
measured q3-30b uns line; probe rule applies (±1 batch/seq rung on OOM/slack).
Order: mixtral → phi → hunyuan → glm4.7 → glm4.5 → gpt-oss; frozen-script
discipline (no driver edits while any run lives).

### WAVE-1 INCIDENTS + FIXES (2026-07-26, during mixtral/phi legs)
- **T3 preset is qwen-gated**: driver rejects ker101 for non-Qwen3 models
  ("must use recomp-off-full-fg-ker000"). New families' T3 = raw token
  `recomp-off-full-fg-ker000-ceil0000-ohbm0` (identical effective semantics —
  qwen fg flags were inert). Chain updated; tier_recipes emit for new-family
  T3 tokens is follow-up work with the scheduler.
- **PhiMoE jitter × GC = crash**: HF PhimoeTopKRouter applies jitter IN-PLACE
  on a view → "view ... modified inplace" under unsloth-GC backward hooks —
  breaks the HF REFERENCE side itself, and stochastic jitter is incompatible
  with loss-parity anyway. Fix: `ASYM_ZERO_ROUTER_JITTER=1` hook in
  run_lf_profiled_train.py `_capture_loaded_model` (duck-typed sweep zeroing
  router_jitter_noise/input_jitter_noise/jitter_noise on config+modules;
  no-op unless env set). Validation protocol runs BOTH sides with it.
- Mixtral validation walker banked before the T3 gate hit: uns-off 64k·b2 FIT
  69.8 GiB (38% HBM — host-bound config) / RSS pending in TSV; 64k·b3 host-
  COOM ×3 (bracketed wall). T3 redo runs 64k·b2 (same-workload row) + 64k·b3
  (dominance probe at the baseline's death rung).

### WAVE-1 INCIDENTS #2 (2026-07-26, phi/hunyuan legs)
- **Whole-mode router vs strict mover**: PhiMoE's `.mlp.router` (nn.Linear
  subclass) stayed a raw param inside the wrapper → CPU-first mover strict
  check raised. Fix (lf.py mover): router-component params now FORCE-place to
  GPU (`router_whole_gpu` bucket) — whole-mode routers execute there by
  design, tiny tensors; covers phi/hunyuan-wg/GLM gates in one carve-out.
- **Hunyuan × ZeRO-3 tracing error**: `hunyuan_v1_moe` was the ONE wave family
  missing from LF's z3-leaf registry (moe.py) → DS prefetch tracer died
  ("tracing error at step 59") on the REFERENCE side. Fix: registered
  HunYuanMoEV1Moe as z3 leaf (additive; same fix every other MoE arch has).
- phi validation banked: uns-off 128k·b3 FIT 149.9 GiB (82%) RSS 392 vs T3
  FIT 158.6 GiB RSS 442 — same-workload row has asym HIGHER on both axes at
  42B-scale (bank small, recompute graph on GPU); verdict deferred to the b4
  capacity pair (does uns-off die where T3 stands?) — queued in redo2 with
  the hunyuan family redo.

### WAVE-1 INCIDENT #3 (hunyuan): tied embed/lm_head
Hunyuan-A13B ties embeddings; the asym offload stage rejects tied weights
("tied embed/lm_head weights are not supported by this offload stage").
Fix (run-config, no code): hunyuan asym legs run
ASYM_OFFLOAD_MODULES=routed_experts,shared_experts,attention,mlp_dense,norms
(embed/lm_head stay GPU-resident, ~1-2 GB — immaterial). Applied in redo2.
Follow-up option if ever needed: tied-weight support in the offload stage.

### WAVE-1 INCIDENT #4 (hunyuan): 129k vocab × no Liger = logits-bound OOMs
uns-off GOOM'd at EVERY rung 32k·b12→b7 (both sides would): LF skips Liger
loss-only for unvalidated model types, so full-vocab logits + fp32 softmax
dominate HBM (~8 GB/batch-row at 32k seq + transients). Fix: liger-kernel
SHIPS apply_liger_kernel_to_hunyuan_v1_moe — LF's resolver just never mapped
it. Added hunyuan_v1_moe to _LOSS_ONLY_SUPPORTED_MODEL_TYPES + resolver
(additive). Hunyuan family rerun runs liger-consistent on BOTH sides.
PRE-FLAG for wave 2: GLM (151k vocab) and gpt-oss (201k vocab) will need the
same mapping (liger ships glm4/gpt_oss appliers) — apply BEFORE their
validation pairs, verify loss-parity at dev first.

### HUNYUAN LOSS FINDING (2026-07-26, probe-isolated — decision pending)
Dev loss Δ 5.06% (asym 1.4829 vs uns 1.4114; liger-consistent ref reproduces
1.411377 BIT-EXACT → confound eliminated). Layer-0 REAL-weights probes:
- routing: indices EXACTLY equal; weights differ only by my bf16 cast (1e-3).
- **engine-vs-HF experts at IDENTICAL routing/weights: 0.70% relmean/block**
  — grouped-GEMM accumulation-order at hunyuan's shapes (top-8 × 64 small
  I=3072 experts = high fan-in, small ~0.125 weights); ×32 layers compounds
  to the observed ~5%. fp32 routing weights barely move it (0.44→0.41%).
- Suspected mechanism: weight-application point (pre- vs post-down GEMM) —
  exact math equal, bf16 rounding differs; mixtral top-2 (w≈0.5) shows
  0.35% E2E, within band, same engine.
DECISION PENDING (user): engine numerics changes would touch the VALIDATED
engine shared by all banked families — not doing that unilaterally. In
flight: 5-step trajectory A/B (parallel curves ⇒ level-offset noise, not
signal corruption) + wave-4 T3 rung descent for the memory verdict.

### MEASURED METRICS — MIXTRAL + PHI, memory AND timing (2026-07-27 addendum)
Source of truth: each run's `profile.json` (paths in the runs' `jobs.tsv`).
Timing = fwd/bwd totals over the run's steps (dev = 2 steps: w1+m1;
validation = 3 steps: w1+m2) → compute/step = (fwd+bwd)/steps. Wall
additionally includes model load + host bank setup (mixtral streams a 262 GB
bank at load). Asym cpu-adamw adds ~0.1–0.3 s/step (in wall; baseline's
optimizer not separately instrumented).

Mixtral-8x22B:
| run | workload | peak resv HBM | peak RSS | compute/step | wall |
|---|---|---|---|---|---|
| uns-off dev | 8k·b1 | 18.5 GiB | 869 GB | 8.1 s | 107 s |
| asym T1 dev | 8k·b1 | 11.0 GiB | 659 GB | 3.8 s | 367 s |
| uns-off val | 64k·b2 | 69.8 GiB | 868 GB | 42.2 s | 470 s |
| asym T3 val | 64k·b2 | 80.6 GiB | 815 GB | 55.8 s | 937 s |
| uns-off cap | 64k·b3 | host-OOM ×3 (no artifact) | | | |
| asym T3 cap | 64k·b3 | 120.6 GiB | 912 GB | 80.2 s | 1133 s |

Phi-3.5-MoE:
| run | workload | peak resv HBM | peak RSS | compute/step | wall |
|---|---|---|---|---|---|
| uns-off dev | 8k·b1 | 9.3 GiB | 262 GB | 3.5 s | 42 s |
| asym T1 dev (dv2) | 8k·b1 | 6.1 GiB | 276 GB | 1.5 s | 116 s |
| uns-off val | 128k·b3 | 149.9 GiB | 392 GB | 44.3 s | 468 s |
| asym T3 pre-fix (superseded) | 128k·b3 | 158.6 GiB | 442 GB | 54.8 s | 675 s |
| asym T3 FINAL | 128k·b3 | 75.6 GiB (49.6%) | 563 GB | 54.8 s | 633 s |
| uns-off cap | 128k·b4 | GPU-OOM | | | |
| asym T3 cap | 128k·b4 | 100.6 GiB | 567 GB | 72.9 s | 820 s |
| asym T3 cap | 128k·b5 | 125.8 GiB | 734 GB | 91.9 s | 1026 s |

Timing read: asym is FASTER per step at the 8k dev rung (streamed experts
beat the full graph at short seq) and ~24–32% slower per step at the long-seq
T3 validation rungs (recompute dominates — the price of the memory win). The
phi fused-loss + moefg fixes were memory-pure: 54.8 s/step unchanged pre/post
while peak fell 158.6 → 75.6 GiB.

### MIXTRAL TIER-LADDER ANCHORS (2026-07-27 evening, user-approved runs)
4 anchor runs (at1b2/at2b2/at2b3/at3b2, artifacts + logs in
`agent/anchors_tmp/`) to measure the T1/T2/T3 ladder on CURRENT code —
motivated by: T2 had never been run for any new family, T1's 64k smoke
artifacts were destroyed, and vl-t3's numbers predate the moefg unlock.
All at 64k seq on one GB200 (~183 GiB budget), MAX_SAMPLES=512 w1+m2:

| tier @64k·b2 (128k tok) | peak resv | compute tok/s | profiled-wall tok/s | RSS |
|---|---|---|---|---|
| T1 (unsloth-ohbm0+staged) | 99.2 GiB | 5,818 | 1,773 | 766 GB |
| T2 (moe|T2 preset, moefg1) | 72.5 GiB | 5,071 | 1,319 | 904 GB |
| T3 (raw ker000 token, moefg1) | 58.7 GiB | 2,534 | 704 | 908 GB |

- T2 b3 leg: 109.2 GiB @ 192k tok → slope 0.573 GiB/1k tok, intercept ≈0.
  Per-1k-token slopes: T1 0.775 / T2 0.573 / T3 0.459 → b1 seq walls
  ≈236k / ≈319k / ≈400k (rope-extended; ctx cap 65k, OOD beyond).
- **moefg unlock measured on mixtral T3: 80.6 → 58.7 GiB (−27%)** at
  unchanged loss (0.7748/0.8892 vs T1's 0.7743/0.8904 — engine-consistent).
  The vl-t3 verdict-row numbers are the pre-moefg standing.
- METRIC LESSON: tqdm wall under PROFILERS=source is inflated ~2.5–3.5×
  vs instrumented fwd+bwd on ALL configs — never mix the two metrics when
  comparing tiers (an earlier cross-metric read had T1 "slower" than T3;
  consistent metrics show T1 is 2.3–2.6× FASTER). Tier order on both
  metrics: T1 fastest/fattest → T2 → T3 leanest/slowest.
- Ops: anchors ran in a SECOND enroot instance of asym_sft_42
  (ENROOT_{DATA,RUNTIME,CACHE}_PATH under /scratch_local/.../enroot, -m
  binds for repo + HF cache) — pattern reusable for future runs.

### TP-VS-SEQ FIGURE CAMPAIGN (2026-07-28, user directive: mixtral + phi
### single-rank panels for the paper's combined throughput figure)
~50 measured cells (chains + logs + plan: `agent/anchors_tmp/`, cells JSON:
`tpfig_cells.json`); DATA landed in `/home/kevinni/env/figures/
plot_tp_vs_seq.py` (combined figure now 3x2); PDFs synced to the overleaf
archive figures/ + tex comment & six-model count updated (UNCOMMITTED, and
the overleaf clone is NOT pushed — Kevin reviews). EVERY plotted cell
measured, zero est. Headlines:
- **Mixtral-8x22B**: asym leads EVERY rung it fits (64k best-batch 1826 vs
  uns 1735; 128k 1223 vs 1162; 192k 970 vs 942; 256k 802 vs 788; 320k asym-
  ONLY rung, T2 670). Walls measured: rc/uns-OFF (128k,192k]; uns + asym-T1
  (256k,320k] G-OOM; asym T2/T3 (320k,352k] host-COOM.
- **Phi-3.5-MoE**: 128k best-batch crown asym T1·b3 4074 (uns 4052·b2, rc
  3921, uns-OFF 2873·b3) + T3 capacity b5 (640k tok, +67% vs uns-OFF's b4
  wall); 192k asym 1310 leads; 224k uns dead, uns-OFF 905 vs asym-T3 842
  (baseline +7% at the shared deepest rung — honest); 256k UNIVERSAL wall.
- **PHI MODEL-STACK WALL (new finding)**: at 256k·b1 EVERY system (all asym
  tiers included) dies on the identical 61.04-GiB alloc = 256k² sliding-
  window mask materialization (window=131k). The masked-SDPA path past the
  window also collapses ALL systems ~4000→~1300 tok/s at 192k. Any fix
  (mask-free sliding attention) would lift every system equally.
- **MIXTRAL HOST-POOL BLOWUP (new finding, fix path)**: asym T2@352k, T3@352k
  and T3@384k all host-COOM at FIRST FORWARD (node 1600→49 GiB) while their
  320k runs sit at RSS ~880-910. Suspect: pinned attn-act offload pool
  allocation at (320k,352k]. Fixing it is the unlock for mixtral >320k.
- OPS LESSONS (recorded for reuse): (1) parallel driver chains on one node
  host-contend → serialize + high host floor (1300 GB) + own-GPU guard;
  (2) `failed:1`-counts-as-trained is DEAD — post gate-fix, only `ok` counts,
  and OOM-grep must PRECEDE the tsv check; (3) run_cell logs append per
  tag+batch → a rerun inherits the old attempt's OOM lines (stale-verdict);
  fresh tag names per rerun (mxu064-b2 "COOM" was stale; artifact = clean
  1668 tok/s TRAIN); (4) `superoffload_mem|unsloth` @64k·b4 died as SIGKILL
  (kernel OOM-killer beat the watchdog poll) — treat exit -9 at step-0 as
  host-OOM.

### GLM MEMORY-DESCENT ROUND (2026-07-29/30 — exploration, NON-VERDICT per
### the fair-comparison rule below; ~14 cells, agent/anchors_tmp/glm*_ladder*)
Question: how low can T3 peak go on the GLMs. Answers, all at the verdict
workloads (Air 128k·b2 / Flash 192k·b5):
- Knob probes are DEAD ends: sdparecomp 160.2, expact 158.9, both 158.8
  (vs 158.8 plain) — the Flash peak is pure attention-bwd transient (83.3
  of 158.8); expact can't touch bwd-recompute transients (re-confirmed).
- NEW STAGE `asym_gemm/training/qchunked_attention.py` (query-chunked flex
  attention, env ASYMM_ATTN_QCHUNK_ROWS/_MODE, checkpoint-per-chunk,
  fixed kernel_options): **SYSTEM-AGNOSTIC — excluded from A/B verdicts.**
  On Air it works as designed: attention ws 85.1 → 19.97 GiB (−77%); peak
  moves to expert-bwd (~125 total; fg-chunk env probes 256/128MB → 123.3/
  122.9 = floor; the env does not govern the fg transient). Air b4 with
  chunking still GOOMs (expert transient doubles). Loss note: chunked steps
  within 1–3.2% of plain (flex bf16 ordering); needs a dev A/B before ANY
  claim use.
- On Flash the chunked path is PARKED: MLA's full-width K/V make the
  per-chunk checkpoint saves 18.4 GiB/layer → ~865 GB hosted via save-on-
  cpu → node saturates at first forward (three COOMs incl. fixed-kernel
  retry; host sampler bottomed ≤557G mid-drain). Air survives the same
  code because GQA K/V saves are 0.5 GiB/layer. The proper fix (future
  work, still system-agnostic): a zero-new-saves chunked attention whose
  backward restages q/k/v from the EXISTING attn-act offload handles.
- **GB200 COHERENT-MEMORY SPILL FINDING (matters beyond GLM)**: flex/
  inductor autotune BENCHMARKS candidate kernels at real sizes; near-cap
  GPU demand spills into host pages (unreclaimable "Cached" — the host
  sampler caught a ~7 GB/s drain with instant snap-back on kill). This is
  the likely mechanism class behind the "first-forward host transient"
  incidents (mixtral T2/T3@352k, T3@384k, Flash T3-b3): near-cap GPU +
  coherent-memory spill → watchdog COOM instead of clean CUDA OOM.
  Mitigations recorded: fixed flex kernel_options (no benchmark sweep),
  TORCHINDUCTOR_COMPILE_THREADS=1; those mixtral walls deserve a re-probe
  with allocator spill disabled before being treated as hard.
- Also landed (both-sides-neutral, kept): MLA share-dedupe in the attn-act
  context (q_a+kv_a share one CPU copy; kills the 172-GiB duplicate D2H
  stream; unit + field verified, kv_a_proj_with_mqa.U tag gone).
- FINAL STANDINGS (unchanged by this round): Air VERDICT = WIN+DOMINANCE
  on the clean config (121.5 vs 131.0; b3 dominance). Air absolute floor
  with all levers ≈ 122.9 (chunked, non-verdict). Flash VERDICT = baseline
  wins at generic-T3 (128.5 vs 158.8 @b5); its asym-specific unlock remains
  MLA-aware attn-act offload + the zero-new-saves chunked design.

### FAIR-COMPARISON RULE (user, 2026-07-29 — NO GENERIC TRICKS IN A/B VERDICTS)
Memory/throughput VERDICTS vs the baseline may NOT be won with generic,
system-agnostic techniques applied one-sided (e.g. blockwise/chunked
attention, allocator tuning, activation checkpointing variants): if the
baseline could adopt it identically, it either goes on BOTH sides (the
liger fused-loss precedent, hunyuan incident #4) or in NEITHER. A/B wins
must come from asym-specific machinery (expert streaming/banks, moefg
engine paths, host-offload architecture). Generic levers may still be
built and recorded — labeled as system-agnostic absolute-memory tools,
excluded from verdict rows. (Recorded after the qchunked-attention round:
Air's verdict stands on the CLEAN config — 121.5 vs 131.0 + b3 dominance,
no chunking; the chunked cells are exploration, not evidence.)

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
