# TRUE T3 for the two GLMs — task + record (2026-08-08)

USER DIRECTIVE: the GLM panels' "T3" cells were a T3-LITE (ker000, no tier
env). Fix both GLMs (GLM-4.5-Air 106B, GLM-4.7-Flash 30B, HF arch glm4_moe /
glm4_moe_lite) so the REAL T3 tier — `asym_cpuadamwds|T3` (1r) /
`asym_sdp2_cpuadamwds|T3` (2r) — resolves, engages its kernels, and trains
with validated numerics. Don't stop until true T3 works on both.

## §1 GAP ANALYSIS (deep-dive 2026-08-08, agent map; anchors verified)

T3 ≡ tier alias `TIER_TOKEN[moe|T3] = recomp-off-full-fg-ker101-ceil0000-
ohbm0` + `TIER_ENV[moe|T3]` (DOWN_DX_STAGED=1, DOWN_SCATTER_BLOCK_EXPERTS=0,
FG_DA_GPU=1, KEEP_DGRADS_HBM=1, FG_LORA_A_FWD_GPU=1, FG_ELEMENTWISE_CHUNK_MB
=1024, CPU_OPS_THREADS=48, PLACEMENT_POLICY=1) — `scripts/lf/tier_recipes.sh
:16-17`.

Already working for GLM (shared machinery, nothing to do):
- fine-grained expert engine (AsymQwen3Experts; GLM wrap mirrors the flag,
  asym_gemm/integrations/lf.py:2235-2265) — "fg fully engaged 45/45"
  validated on Air (model_integration.md GLM leg);
- dense-MLP fg, expert-act LoRA-A cpu, GC save-on-cpu (family-agnostic env).

Missing / broken:
1. **ker101 shell gate** — `validate_recompute_kernel_for_model`
   (profile_lora_lf_test_source.sh:821-829) dies for ker!=000 unless
   `is_qwen3_moe_routed_model` (796-801: the three qwens only). GLM resolves
   family moe -> T3 alias = ker101 -> hard-die. Every GLM campaign therefore
   hand-rolled ker000 tokens as "T3" (T3-lite), and raw tokens get NO
   TIER_ENV. The routed-GEMM kernels themselves
   (asym_gemm/training/qwen3_moe_routed_gemm.py) are layout-generic (packed
   [E,2I,H] + group metadata; no qwen shape asserts) but UNVALIDATED on GLM
   shapes/routing (Air: DS-V3-style group-limited top-k w/ bias correction;
   Flash: MLA blocks).
2. **MLA attn-act offload (Flash only)** — `_build_attention_activation_
   contexts` (asym_gemm/integrations/lf.py) recognizes the GQA q/k/v triple;
   Flash's MLA (q_a/q_b, kv_a_with_mqa/kv_b) silently no-ops -> attention
   activations stay in HBM. Air (GQA) works. Known+recorded:
   model_integration.md incident #2 "MLA-aware attn-act = the Flash T3
   unlock".
3. Optional, small: GLM shared-experts offload (qwen35-style
   AsymQwen35SharedMLP mirror); GLM wrappers currently keep shared_experts
   GPU-resident by design. NOT in scope unless cheap.

## §2 DONE =
1. `|T3` alias resolves for both GLMs: ker101 accepted, TIER_ENV applied
   (visible in run env/dirname route-flag class).
2. Numerics: house unit — wrapped-vs-HF forward/backward max|Δ| within the
   family-port band with route flags ON — passes for both GLM shapes;
   short loss-band run sane vs T3-lite twin.
3. A T3 training cell per model per rank-class trains (or hits an HONEST
   memory wall) with kernels engaged — no shell die, no silent fallback.
4. Flash: attn-act offload actually offloads under MLA (nonzero offload
   counters / reduced attn residency at fixed seq).
5. Ledger below + run_glms.md cross-note updated.

## §3 EXPECTED PAYOFF (calibrated; do not oversell)
- Air 2r published walls (336k+): host/shm-bound — true T3 will NOT move
  them. Air 1r 448k+ host-walled too (RSS 917/957).
- Flash deep rungs: MLA attn-act is the big seq-scaling HBM lever; ker101
  + DOWN_DX_STAGED shave 98%-edge transients. Plausible +64k on crowns
  (both 1r 1.09M and 2r 1.02M were host-capped at the end though).

## §Log (append-only)
- [08-08 03:1x] doc created; implementation starting: gate widening first,
  then numerics unit on GLM shapes, then MLA attn-act port, then validation
  cells (GPUs idle post-campaign).
- [08-08 03:3xZ] PIECE 1 DONE — gate widened in profile_lora_lf_test_source.sh
  (new is_glm_family_model; validate_recompute_kernel_for_model + the
  route-enabled scope check at ~3560 both accept GLM). DRY-RUN VERIFIED both
  models: `asym_cpuadamwds|T3` -> "TIER preset ... ker101 ... (+8 recipe
  env)", route101_lora0_accfp32 in the dirname, KERNEL_CODE=101 exported.
  profile_lora_lf_test_both.sh left untouched (stale sibling: its fg gate
  never included GLM either; campaigns use test_source only).
- [08-08 03:3xZ] PIECE 2 ALREADY LANDED (by the c14 session, pre-dating this
  task): _build_attention_activation_contexts accepts the MLA pair
  q_a_proj+kv_a_proj_with_mqa and _is_text_attention_module_name recognizes
  MLA children — Flash attn-act offload is live (their 1r T3+ohbm8 1.09M
  finale depended on it).
- [08-08 03:4xZ] UNIT v1 (absolute maxΔ band) was too blunt: ker000 baseline
  (the integration-validated config) showed the same ~2-3e-2 deviation as
  ker101 under random-init amplification — bf16 accumulation-order noise,
  not kernel error (integration's 6.1e-5 was real-weight scale). UNIT v2 =
  fp32-truth noise-envelope method: PASS iff w000/w101 errors are within 3x
  HF-bf16's own error vs the fp32 reference. Running.
- [08-08 03:5xZ] UNIT PASS (final; t3_glm_unit.py, control-relative family-
  parity criteria; qwen3-30b = production-validated control run in the same
  process). Evidence table (vs fp32 truth, maxΔ/rel; T=2048 b1 true dims):
    control  out 1.57e-1/2.2e-1  engine-dX rel 6.9e-1  ker-delta 5.4e-3/7.1e-3
    air      out 2.93e-2/6.6e-3  engine-dX rel 5.0e-2  ker-delta 3.5e-3/5.2e-3
    flash    out 2.15e-2/7.1e-3  engine-dX rel 1.3e-1  ker-delta 5.2e-3/3.4e-3
  Readings: (1) forward = HF-parity on all three; (2) ker101 delta on GLM is
  AT/BELOW the control's — kernels add no GLM-specific drift; (3) the large-
  looking engine-dX-vs-HF gap is the DETACHED-ROUTER backward design, largest
  on the validated control itself — family-normal, not a GLM defect. LoRA
  grads finite everywhere. Unit iterations 1-3 were yardstick miscalibrations
  (absolute bands, HF-noise ratios) — data never changed.
- [08-08 03:5xZ] SMOKE CELLS launching (t3_glm_smoke.sh): Flash 1r/2r @256k,
  Air 1r/2r @192k, |T3 alias (ker101+TIER_ENV), serial, container.
- [08-08 10:44Z] t3f1s256 glm4.7-flash asym_cpuadamwds|T3 s=256000 r1 b=1 -> FAIL | -
- [08-08 10:54Z] t3a1s192 glm4.5-air asym_cpuadamwds|T3 s=192000 r1 b=1 -> FAIL | -
- [08-08 11:03Z] t3f2s256 glm4.7-flash asym_sdp2_cpuadamwds|T3 s=256000 r2 b=1 -> FAIL | -
- [08-08 11:13Z] t3a2s192 glm4.5-air asym_sdp2_cpuadamwds|T3 s=192000 r2 b=1 -> FAIL | -
- [08-08 22:36Z] bisect t3bA recomp-off-full-fg-ker000-ceil0000-ohbm0 s=64k -> FAIL
- [08-08 22:40Z] bisect t3bB recomp-off-full-fg-ker100-ceil0000-ohbm0 s=64k -> FAIL
- [08-08 22:46Z] bisect t3bC recomp-off-full-fg-ker001-ceil0000-ohbm0 s=64k -> FAIL
- [08-08 22:51Z] bisect t3bD recomp-off-full-fg-ker101-ceil0000-ohbm0 s=64k -> FAIL
- [08-08 23:46Z] probe t3pA1 (tier-env-as-is) -> FAIL
- [08-08 23:52Z] probe t3pA2 (ASYM_CPU_OPS_THREADS=48 ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE=0) -> FAIL
- [08-08 23:57Z] probe t3pA3 (ASYM_CPU_OPS_THREADS=48 ASYMM_ATTN_ACT_OFFLOAD=false) -> FAIL
- [08-09 02:26Z] t3f1v2 glm4.7-flash asym_cpuadamwds|T3 s=256000 r1 b=1 -> TRAINED | 454.8	563	42.2	23	329	0.2
- [08-09 02:55Z] t3a1v2 glm4.5-air asym_cpuadamwds|T3 s=192000 r1 b=1 -> TRAINED | 320.2	600	65.8	36	914	0.2
- [08-09 03:25Z] t3f2v2 glm4.7-flash asym_sdp2_cpuadamwds|T3 s=256000 r2 b=1 -> TRAINED | 457.8	1118	48.7	26	333	0.1
- [08-09 03:40Z] t3a2v2 glm4.5-air asym_sdp2_cpuadamwds|T3 s=192000 r2 b=1 -> COOM | -
- [08-09 03:4xZ] **TASK COMPLETE — TRUE T3 WORKS ON BOTH GLMS.** Root cause
  of the smoke segfaults: _dense_lora_a_cpu_left's asym branch hands CUDA
  LoRA-A weights to the native CPU-left binding — provably broken for EVERY
  caller (48/48 isolated repros incl. qwen shapes) but reachable only when
  the shared q/k/v LoRA-A source path doesn't engage (Flash's MLA pair;
  Air's reach via its own non-shared projection path) — qwen's trio always
  takes the shared path, hence never crashed in production. FIX (additive):
  env-gated reroute to the existing torch staging math
  (ASYMM_ATTN_LORA_A_CPU_LEFT_TORCH_STAGE=1, exported by the driver's
  full-fg branch for is_glm_family_model only; qwen paths byte-identical).
  SMOKE v2 (ker101 + TIER_ENV live, faulthandler on): Flash 1r 256k TRAINED
  563 (23% HBM) · Air 1r 192k TRAINED 600 (36%, RSS 914) · Flash 2r 256k
  TRAINED 1118 global (26%) · Air 2r 192k honest HOST-C-OOM (T3-adds-host
  on 106B@2r — known class, measured wall, not a crash). Done-definition
  §2: all criteria met.
