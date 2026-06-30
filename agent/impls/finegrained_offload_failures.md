# `recomp-off` Status + Failure Cases (companion to `finegrained_offload.md`)

**Target:** beat `q3-32b|1 ; superoffload_mem|unsloth|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false` on a
**184 GiB GPU / ~870 GiB CPU** (`membind=0,1`) — fit a real seq ≥ **s50000** at b8 that the baseline cannot. **This doc =
why the `asym_cpuadamwds|recomp-off|ligerloss1` attempt fails; read alongside `finegrained_offload.md` (the design).**

**Target baseline (measured): `superoffload_mem|unsloth` @ s50k = 181.4 GiB reserved** (185764 MiB; allocated 181.0;
**fits, barely**) / **1736 tok/s** / 230 s-step, 400000 real tokens, loss finite. recomp-off detail below is at
**s45000** (its largest point with complete artifacts) where it **already OOMs (183)** — so it cannot reach s50k (every
tensor is ~11% larger). M=b8·s45000=**360000**, I=25600, H=5120 ⇒ **`[M,I]`=17.17, `[M,2I]`=34.33, `[M,H]`=3.43 GiB**
(bf16, verified from the OOM alloc sizes).

## Headline @ s45k b8

| config | HBM reserved | tok/s | s/step | result |
|---|---:|---:|---:|---|
| **`superoffload_mem\|unsloth`** (target baseline) | **178.4** | **2315** | 155 (2.6m) | fits; loss 1.0091 |
| **recomp-off V2** (expact0·attn1·loraAcpu) | **183.0** | **351** | 1025 (17m) | **WORSE on both axes**; loss 1.0093 |
| **recomp-off V3** (expact1·attn1·loraAcpu) | **OOM @184** | — | — | OOM (+17.17=`[M,I]` on 166.07 alloc) |

*Tuple = `backend | recompute | liger`.* `superoffload_mem` is the **backend** (offloads optimizer state + base weights
to CPU; does NOT offload activations); `unsloth` is the **recompute** mode. recomp-off's backend is `asym_cpuadamwds`
(asym base-weight `@^R` + CPU AdamW) — a *different* backend, so compare recomp-off to the `superoffload_mem` baseline
only at equal seq. recomp-off loses on **both** memory (183 > 178.4; nvidia-smi total ≈180) and speed (351 << 2315
tok/s). Math is correct (loss ≡ superoffload to 3 dp). CPU RSS 472→602 GiB once activations actually offload (<870).

## Failure cases

| # | symptom | root cause | evidence | status |
|---|---|---|---|---|
| **F1** | SURGICAL=1 set but `[M,I]` stays on GPU (V2 holds **72.2 GiB** MLP saved) | **Two-flag footgun.** `ASYMM_DENSE_MLP_SURGICAL_OFFLOAD=1` only *installs* the engine + offloads base **weights**; the activation-offload **Function** is dispatched by a *different* env `ASYMM_EXPERT_ACT_OFFLOAD` (policy `expact`). expact0 ⇒ engine falls to in-graph `_forward_expert_body` (saves 4.2×`[M,I]`=72 GiB) | install `lf.py:1983-1990`; dispatch `qwen3_moe.py:2675` (Function) vs `:2694` (in-graph); flag `:2474`/`:2476`. V2 csv `mlp_dense saved_activation`=72.2 GiB | **root-caused.** fix = expact1 (→F2) |
| **F2** | LoRA materializes `[M,I]` on GPU even at expact1+`loraA=cpu` | down-LoRA-A keeps the act `[M,I]` on GPU instead of streaming `@^L` | s16384 **after_backward**: `mlp.engine.lora_dropout [M,I]` LIVE 6.4 GiB (→**17 @ s45k**) = the #1 *exact held* tensor. V3 OOM alloc = +17.17 GiB (`[M,I]`) | **confirmed (bwd).** avoidable iff LoRA-A streams the act |
| **F3** | "norms/residual on GPU ~36 GiB" | **mostly mis-attributed workspace, NOT held.** Real held norm = one layer's `[M,H]`; the bulk is `inferred_peak_workspace` = the live GEMM set (→F4) | s16384 bwd: `norms live_activation` = **1.28 GiB** (one `layers.1.input_layernorm [M,H]`) vs `norms inferred_peak_workspace` 13.5. Real held norm ~1–14 GiB | **corrected — workspace, not a held lever** |
| **F4** | a large transient survives all offload | **irreducible live GEMM working set** — operands must be on GPU *before* they can be offloaded: gate_up `[M,2I]`=34.3 (must exist to be ⬇) and backward `grad_gate_up` `[M,2I]` (left operand of the `gate_up_base` dX GEMM) + the gate/up/down LoRA GEMM transients | V1 OOM alloc = +34.33 (`[M,2I]`) = the gate_up GEMM. dX GEMM `qwen3_moe.py:1430-1439`. s16384 bwd: ~58 GiB inferred workspace (×2.75 ≈ **158 @ s45k**, dominant) | **inherent** (only chunking shrinks it) |
| **F5** | recomp-off step = **1025 s** (17m) vs baseline 155 s; CPU silu ~640 ms/layer, **GPU 0% idle** | **bandwidth-bound, not slow compute.** silu is elementwise over a 17.2 GiB DRAM-resident `[M,I]` (~400× the cache-resident microbench tensor) ⇒ ~200 GiB CPU traffic/layer, contended on shared Grace C2C with CPU-Adam grad-offload D2H + the ⬇/⬆ copies. Super-linear: 17.8s@s2k→800s@s16k | math `qwen3_moe.py:913-927`; grad D2H `cpu_adam.py:362-389`; offload D2H/H2D `activation_offload.py:187-250`. GPU-silu escape `:930-966` stages gate+up back = **+34 GiB** ⇒ wrong trade for max-seq | **characterized** (the peak↔speed dial) |
| **F6** | design projected ~35–40 GiB; reality 183 / OOM | projection **conflated** *held* acts (already ≈0 under recompute — the baseline doesn't hold them either, that's its 178) with the **transient working set** (the real ceiling, F4). offload only shrinks what's on the offload path (MLP base ✓; LoRA F2 ✗; the live GEMM operands can't be) | `finegrained_offload.md:110-111` vs measured V3 OOM | **root-caused** |

## Backward-peak ground truth (forensic s16384 run, fits at 70.6 GiB reserved)

The captured **after_backward** peak (the data V3 OOM'd before writing) decomposes as **~82% live GEMM workspace**
(`lora` 38.5 + `norms` 13.5 + `attention` 4.0 + `embed` 1.6 = 57.6 GiB) + **~13% exact held** (`lora_dropout [M,I]` 6.4 +
`norms [M,H]` 1.28 + `embed` 1.28). **MLP base fully offloaded** (no `mlp_dense saved`; only its 720 MiB weight).
Scaling ×2.75 → ~158 workspace + ~25 held ≈ **183 GiB @ s45k = exactly the V3 OOM**. ⇒ **the ceiling is the live GEMM
working set, not held activations.**

## Synthesis

Recompute (unsloth GC) already collapses **held** activations 64×→≈0 — the baseline's 178 is its **transient working
set**, not held activations. recomp-off offloads the one held piece recompute was already cheapest on (MLP base
72→3.4), while leaving on GPU: the **LoRA-A act `[M,I]`** (not streamed, F2, ~17) + the **irreducible live GEMM
operands** (gate_up/`grad_gate_up` `[M,2I]`, LoRA GEMM transients, F4, ~158). **Offload cannot lower a peak made of
*live* operands**; the only lever that shrinks the working set itself is **chunking** (`[chunk,I]`) — which recomp-off
lacks.

## Verdict / next

- recomp-off as built **cannot reach the s50k target** (OOMs at s45k, 183 GiB; +11% for s50k ⇒ deeper OOM), and is
  ~6.6× slower than the baseline.
- To fit ≥ s50k the **working set must be chunked** (single-recompute chunked-MLP that recomputes+offloads each
  `[chunk,I]` once). Offload alone is proven insufficient here.
- F2 (stream the LoRA-A act via `@^L`) recovers ~17 GiB held but **not** the ~158 GiB GEMM-operand workspace.
- **Exact bar: superoffload @ s50k = 181.4 GiB / 1736 tok/s (fits).** recomp-off must beat that; it OOMs at s45k, so it
  is not in contention. superoffload's s45k→s50k peak grows only 178.4→181.4 (sublinear — its peak is dominated by the
  per-layer weight-gather, not activations), confirming the baseline is hard to beat by shaving activations alone.

## Run log

| run | config @ s45k | outcome | key numbers |
|---|---|---|---|
| Stage 0 | recomp-off s2048 | OK — label wired, finite, completes | 12.1 / 10.1 GiB |
| V1 | recomp-off expact0·attn0·loraAhbm | OOM @196 | 162.15 + 34.33(`[M,2I]` gate_up); RSS 391 |
| V2 | recomp-off expact0·attn1·loraAcpu | fits-183-but-17m | 183.0 / 351 tok/s; 72.2 GiB MLP held in-graph (F1) |
| V3 | recomp-off expact1·attn1·loraAcpu | OOM @184 | +17.17(`[M,I]` LoRA); MLP saved 72→3.4 ✓; LoRA+workspace remain; RSS 602 |
| s16384 | recomp-off expact1·attn1·loraAcpu | fits — forensic | 70.6 GiB; bwd = 82% GEMM workspace + 13% held (F2/F4 confirmed) |
