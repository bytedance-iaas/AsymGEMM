# Qwen3.5 (qwen3_5-35b-a3b): why asym-offload loses, and the REAL fix (pure offload)

Don't change qwen3/llama4's design — but the fix below is shared code, so we must **prove it does not break them**.

## 🚫 HARD RULE: NO LIGER / NO FUSED OR CHUNKED CE — EVER, IN ANY CONFIG
Every run is `ligerloss0` / `ENABLE_LIGER_KERNEL=false`. The ~45.5 GiB CE loss is **off the table**. It is identical
in A/B/C so it cancels. **The comparison is decided only by NON-LOSS.** Any step that touches the loss is out of scope.

## 🎯 GOAL
**The asym offload config `asym_cpuadamwds | norecomp | none|true|true|false|true|*` MUST beat asym-recompute (A) and
zero3-offload-recompute (B)** in peak GPU HBM — all `ligerloss0`. The win must come from **OFFLOAD, not recompute.**

### The numbers — `scripts/lf/show_metrics.py profiling_both_qwen35` (s4096·b4, ligerloss0)
`step_H` = **peak GPU HBM (GiB)** = the metric that matters · `step_s` = step time (s) · `RAM` = host peak (GiB).

| backend / config | flag tuple `pol\|exp\|attn\|layer\|gc\|sdpa` | step_s | **step_H** (peak) | RAM | non-loss |
|---|---|---:|---:|---:|---:|
| **C** asym `norecomp` exp+attn-offload+layerGC **(TARGET)** | `none\|true\|true\|false\|true\|true` | 30.9 | **57.9** | 275.8 | **12.4** |
| A asym `recomp`, no offload | (recompute) | 16.6 | **48.5** | 189.2 | ~3–4 |
| B zero3-offload `recomp`, no offload | (recompute) | 14.8 | **50.9** | 192.1 | ~4–6 |

Today **C (57.9) LOSES to both A (48.5) and B (50.9)** — that is the bug. **Success = C peak < A and < B by a clear
margin**, loss parity. Only the **non-loss 12.4** is attackable (loss 45.5 = hard floor).

---

## 1. CONFIRMED ROOT CAUSE (diagnostic-proven) — the offload wrapper was SKIPPING the activations

The three saved-tensor offload wrappers (`linear_attention_`, `attention_`, `decoder_activation_offload.py`) all had
this guard in `_should_offload`:
```python
if tensor.is_leaf and tensor.requires_grad:   # intended: "skip parameters"
    return False                              # ...but it also skips leaf+grad ACTIVATIONS
```
It was meant to skip **parameters** (which are leaf + requires_grad). But **qwen3.5's fla `chunk_gated_delta_rule`
mixer creates leaf+requires_grad ACTIVATIONS** (tensors materialized inside fla's custom autograd Functions where
grad-mode is off → `is_leaf=True`, then marked `requires_grad=True`). So the guard wrongly skipped them.

**Diagnostic proof** (`profiling_qwen35_dbg`, qwen3.5 sd0, `ASYM_OFFLOAD_DEBUG_SKIP=1`, 1 step):
- **432 skipped tensors, 100% `param=False`** (i.e. activations, NOT parameters):
  | wrapper | shape | size | what it is |
  |---|---|---:|---|
  | LinearAttention | `(16384, 2048)` = `[4·4096, 2048]` | 64 MiB | delta-net hidden stream |
  | LinearAttention | `(524288, 128)` = `[4·4096·32, 128]` | 128 MiB | delta-net q/k/v-style |
  | Decoder | `(16384, 2048)` | 64 MiB | decoder hidden stream |
- Aggregate counters across the run: **skipped 74.7 GiB vs offloaded 94.4 GiB** — the skip was eating ~44% of all
  offload-eligible traffic. Shapes are batch×seq dependent (`16384=4·4096`) → unambiguously activations, not weights.

**Why qwen3/llama4 don't hit this:** SDPA / standard attention produce normal **non-leaf** saved tensors; they have
~0 leaf+grad activations, so the guard never fired on their activations (their `param=False` skip count ≈ 0). That is
exactly why their lightweight offload path already wins and qwen3.5's didn't.

> This **supersedes** the earlier theory that the delta-net mixer (7.51 GiB) was an un-offloadable "live residual."
> Most of it is **offloadable saved tensors that were being skipped** — so the fix is pure OFFLOAD, no recompute.

## 2. THE FIX — skip ONLY real parameters (3 wrappers)
```python
# was: if tensor.is_leaf and tensor.requires_grad:
if isinstance(tensor, torch.nn.Parameter):   # skip exactly weights; offload leaf+grad activations
    return False
```
**Correctness:** offload copies bf16 → pinned CPU → bf16 GPU, which is **bitwise-exact**; the saved-for-backward VALUE
is unchanged. Leaf-ness only affects `.grad` accumulation (a separate autograd mechanism — the graph edge — untouched
by `saved_tensors_hooks`). So forward/backward math is identical; only HBM residency changes. Applied to
`linear_attention_activation_offload.py`, `attention_activation_offload.py`, `decoder_activation_offload.py`.

---

## 3. VALIDATION PLAN — prove it (a) works for qwen3.5, (b) does NOT break qwen3/llama4, (c) cleanup is neutral

Harness `run1` (fast 1-step correctness) and `run6` (6-step memory) in §5. All `ligerloss0`. GPUs: use **0 and 3 only**
(1,2 = concurrent weight-offload workstream). `ASYM_OFFLOAD_DEBUG_SKIP=1` enables the `[newly-offload leaf+grad]`
counter (logs exactly the tensors the fix newly offloads).

### V0 — qwen3.5 correctness (math must be unchanged) — sd0 PURE OFFLOAD
```bash
run1 0 profiling_qwen35_v0 "Qwen/Qwen3.5-35B-A3B|1" "asym_cpuadamwds|norecomp" "none|true|true|false|true|false"
```
**PASS iff:** no exception; backward completes; `loss[:5]` finite AND within ~5% of the pre-fix diagnostic loss
(`profiling_qwen35_dbg`; fla kernel is non-deterministic so bit-parity is impossible); grad-norm finite.

### V1 — qwen3.5 efficacy (same run as V0)
**PASS iff:** peak HBM **drops vs pre-fix sd0** (pre-fix `profiling_qwen35_dbg` = 60.5 GiB table / 63.5 mem-breakdown);
`offloaded_bytes` up, `skipped_bytes` → params-only; `[newly-offload leaf+grad]` count is large (hundreds).

### V2 — qwen3 + llama4 NOT broken (the safety gate) — sd0
```bash
run1 0 profiling_qwen3_v2  "Qwen/Qwen3-30B-A3B|1"             "asym_cpuadamwds|norecomp" "none|true|true|false|true|false"
run1 3 profiling_llama4_v2 "meta-llama/Llama-4-Scout-17B-16E|1" "asym_cpuadamwds|norecomp" "none|true|true|false|true|false"
```
**PASS iff (per model):** `[newly-offload leaf+grad]` count **≈ 0** (≤ a handful) — this is a *proof* the fix is
byte-identical for them (it offloads no new tensors) → it **cannot** change their result; AND loss finite; AND no crash.
If the count is non-trivial for either, STOP and investigate before trusting the fix.

### V3 — remove the recompute detours, re-validate neutrality
Only after V0–V2 pass. Remove (see §4 surface), keep lora.py bugfixes + the skip-fix. Then:
```bash
run1 0 profiling_qwen35_v3 "Qwen/Qwen3.5-35B-A3B|1" "asym_cpuadamwds|norecomp" "none|true|true|false|true|false"
```
**PASS iff:** peak + loss match V1 within noise (cleanup is behavior-neutral at sd0, where the detour was already
dormant — it's gated on sdparecomp=true). Confirms removal introduced no regression.

### V4 — the GOAL grid (qwen3.5 offload vs A vs B), 6-step
```bash
run6 0 profiling_qwen35_fix "Qwen/Qwen3.5-35B-A3B|1" "asym_cpuadamwds|norecomp" "none|true|true|false|true|false"  # C pure offload
run6 0 profiling_qwen35_fix "Qwen/Qwen3.5-35B-A3B|1" "asym_cpuadamwds|recomp"   "none|false|false|false|false|false" # A
run6 0 profiling_qwen35_fix "Qwen/Qwen3.5-35B-A3B|1" "zero3_offload|recomp"     "none|false|false|false|false|false" # B
.venv/bin/python scripts/lf/show_metrics.py profiling_qwen35_fix
```
**PASS (the GOAL) iff:** C `step_H` **< A AND < B by a clear margin**, loss parity vs A. Record the table here.

---

## 4. CLEANUP SURFACE — detours to REMOVE once V0–V2 pass (these were the wrong approach: recompute, not offload)
- **`asym_gemm/training/delta_net_recompute.py`** — delete the whole file (`install_delta_net_recompute`,
  `offload_checkpoint`/`_OffloadCheckpointFunction` [flawed: an autograd Function holds its input, so the CPU copy
  never frees GPU], `residual_offload_enabled`, `delta_net_recompute_enabled`).
- **`asym_gemm/integrations/lf.py`** — remove the `import ... install_delta_net_recompute` (l.24) and the
  `install_delta_net_recompute(model)` call (l.1528).
- **`asym_gemm/training/decoder_layer_glue_gc.py`** — remove `_recompute_delta_net`, `_call_module_raw`,
  `_call_qwen35_linear_attention_raw`, the `_asym_delta_net_recompute` dispatch, and the `delta_net_recompute` import;
  revert `_checkpoint_norm` to its original (run norm; no `residual_offload` branch).
- **KEEP (genuine bugfixes, unrelated to the detour):** lora.py gather-before-apply (`AsymLoRALinear.forward`) +
  `_uses_lora_weight_offload` dropping the `is_grad_enabled()` gate; and the §2 skip-fix.

---

## 5. Validation harness
```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
# run1 <GPU> <OUTROOT> <MODEL_SPEC> <BACKEND> <POLICY>  — 1-step correctness, DEBUG counter on
run1 () {
  ASYM_OFFLOAD_DEBUG_SKIP=1 DRY_RUN=false OVERWRITE=true PLOT=false PREPARE_DATASETS=false REQUIRE_SM100=1 GPU_POOL="$1" \
  MODEL_SPECS="$3" BACKEND_SPECS="$4" ASYMM_EXP_ACT_POLICIES="$5" WORKLOADS="4096|4|1" MAX_STEPS=1 WARMUP_STEPS=1 \
  PROFILE_MEMORY_BREAKDOWN=true PROFILE_MEMORY_SNAPSHOT=false PROFILE_LIVE_ACTIVATION_DETAILS=false \
  OUTPUT_ROOT="$2" bash scripts/lf/profile_lora_lf.sh
}
# run6 <GPU> <OUTROOT> <MODEL_SPEC> <BACKEND> <POLICY> — 6-step memory + snapshot
run6 () {
  DRY_RUN=false OVERWRITE=true PLOT=false PREPARE_DATASETS=false REQUIRE_SM100=1 GPU_POOL="$1" \
  MODEL_SPECS="$3" BACKEND_SPECS="$4" ASYMM_EXP_ACT_POLICIES="$5" WORKLOADS="4096|4|1" MAX_STEPS=6 WARMUP_STEPS=2 \
  PROFILE_MEMORY_BREAKDOWN=true PROFILE_MEMORY_SNAPSHOT=true PROFILE_LIVE_ACTIVATION_DETAILS=true \
  OUTPUT_ROOT="$2" bash scripts/lf/profile_lora_lf.sh
}
```
Read peak/non-loss with `.venv/bin/python scripts/lf/show_metrics.py <OUTROOT>`; per-tensor at peak via the leaf's
`memory_breakdown.jsonl`; loss parity from `source_profile.json` `step_samples[].loss`.

---

## 6. MEASURED RESULTS (fill in as runs land)
| stage | model | config | peak (GiB) | loss (step1 wu/meas) | newly-offload leaf+grad | verdict |
|---|---|---|---:|---|---:|---|
| pre-fix | qwen3.5 | sd0 | 60.5 / 63.5 | 1.846 / 1.953 | n/a (was skipped) | baseline |
| **V0/V1** | qwen3.5 | sd0 | **47.3** (48476 MiB) | 1.866 / 1.957 (max\|Δ\|=0.004) | **432** | ✅ PASS: −12.1 GiB, loss parity, no crash, already < A(48.6) |
| **V2** | qwen3-30b | sd0 | 28.4 (29112 MiB) | 2.150 / 2.189 (finite) | **0** | ✅ PASS: fix offloads 0 new tensors → byte-identical → cannot break |
| **V2** | llama4-scout | sd0 | 24.6 (25170 MiB) | 0.930 / 1.0 (finite) | **0** | ✅ PASS: byte-identical → cannot break |

**V2 verdict:** both non-delta-net models offload **0** new tensors under the fix → the fix is provably scoped to
qwen3.5's fla delta-net activations and is a guaranteed no-op for qwen3/llama4. Safe to remove the detours (§4).

### Detour decision (V3): PURE OFFLOAD beats recompute → detour REMOVED
qwen3.5 @ 4×4096, 1-step, post-skip-fix:
| config | what | peak | loss |
|---|---|---:|---|
| **sd0** `…\|true\|false` | **pure offload** (delta-net activations → CPU) | **47.3 GiB** | 1.866/1.957 |
| sd1 `…\|true\|true` (+ delta-net recompute detour) | offload + recompute mixer (30 tagged) | 48.0 GiB | 1.869/1.947 |

Pure offload (47.3) **< offload+recompute (48.0)** → the delta-net recompute detour is *worse* than offload. **Removed**
`delta_net_recompute.py` + lf wiring + glue detour (V3 done). Code now = skip-fix (3 wrappers) + lora bugfixes only.

### ⚠️ OOM was PARALLELISM, not the fix
qwen3 8×8192 + llama4 4×4096 at sd1 were SIGKILL'd (status 9) when run **in parallel** — their host RAM is 665 / 802 GiB
(see refs below); two at once + the GPU1/2 workstream blew the ~958 GiB membind ceiling → OOM killer. newly-offload=0
in both before the kill (fix changed nothing). **Rule: run heavy workloads ONE AT A TIME.** Re-running sequentially.

### Reference benchmarks — `profiling_both_cur` (sd1 = `none+exp+attn-offload+layerGC [lg- sd+]`, ligerloss0)
The "do not break" baseline. My fixed code must reproduce these for qwen3/llama4 (newly-offload=0 ⇒ byte-identical):
| model | workload | **sd1 ref peak** | ref RAM |
|---|---|---:|---:|
| qwen3-30b | s8192·b8 | **112.7 GiB** | 665 |
| llama4-scout | s4096·b4 | **19.7 GiB** | 802 |

### V2' — sequential not-broken check at sd1 + real workloads (clean post-cleanup code)
| model | workload | my peak (clean) | ref peak | loss | verdict |
|---|---|---:|---:|---|---|
| **llama4-scout** | s4096·b4 sd1 | **18.5 GiB** | 19.7 | 0.934/1.0 finite | ✅ ≤ ref; no crash; newly-offload=0 |
| **qwen3-30b** | s8192·b8 sd1 | **56.7 GiB** | 2.18/2.27 finite | 112.7 | ✅ runs clean; **2× LOWER than ref — investigated below** |

### ⚠️ The qwen3 56.7 vs ref 112.7 gap is NOT a regression — `profiling_both_cur` is a different-code baseline
Investigated the 2× gap (same config/workload, would expect a match):
1. **`profiling_both_cur` is being concurrently regenerated** by the weight-offload workstream — 92 source-profile files
   modified after 15:00 today; the qwen3/llama4 refs are timestamped **16:13** (mid-session). I never wrote there.
2. **It is materialized from NSYS runs** (`command.txt`: "Materialized source artifacts from nsys run", `nsys_dir=profiling_both/…__nsys__…`) — a *different profiling mode* than my `source`-mode runs, and no git commit recorded.
3. **Weight-offload metadata is byte-identical** in both (240 groups, 6.29 GiB home, 672 params, 3.14 GiB numel) → the
   2× is NOT a weights-in-HBM difference; it's in activation handling / code vintage.
4. My skip-fix is **byte-identical for qwen3** (newly-offload=0), and my run is **lower** (not higher) → not a regression.

**Definitive A/B (DONE — both models bit-identical base-vs-fixed):**
| model | workload | BASE code peak | MINE code peak | verdict |
|---|---|---:|---:|---|
| qwen3-30b | s8192·b8 sd1 | 58057.31 MiB | 58057.31 MiB | ✅ bit-identical (loss 2.186/2.273 both) |
| llama4-scout | s4096·b4 sd1 | 18978.93 MiB | 18978.93 MiB | ✅ bit-identical (loss 0.934/1.0 both) |

⇒ my changes are a **proven no-op** for qwen3 AND llama4 — same peak to the byte, same loss. The `profiling_both_cur`
112.7 (qwen3) is purely the older/nsys baseline. On the transpose concern: a transposed weight isn't an `nn.Parameter`,
but OLD (`is_leaf and requires_grad`) and NEW (`isinstance Parameter`) guards treat it **identically** (both offload it),
and offload is bit-exact — so no behavior change; the A/B bit-identity confirms it.

**V0/V1 detail:** offloaded 94.4→128.9 GiB, skipped 74.7→40.2 GiB (remaining skip = params + sub-1MiB). Pure-offload
sd0 peak 47.3 GiB already ≈/below A(48.6) at smoke scale → the offload thesis holds **without any recompute**.

---

### ✅✅ V4 GOAL MET (qwen3.5, 1-step, 4×4096, current clean code, ligerloss0)
| config | peak GiB | meas loss | verdict |
|---|---:|---|---|
| **C sd1** `none\|true\|true\|false\|true\|true` | **46.08** | 1.945 | 🏆 beats A by **2.51**, B by **4.90** |
| C sd0 pure-offload `…\|true\|false` | 47.34 | 1.940 | beats A by 1.25, B by 3.64 |
| A asym-recompute | 48.59 | 1.957 | (reproduces original 48.5 baseline) |
| B zero3-offload-recomp | 50.99 | 1.889 | (reproduces original 50.9 baseline) |

**GOAL ACHIEVED:** the target offload config (sd1) is lowest at 46.08 GiB — beats asym-recompute and zero3 by a clear
margin, loss parity (C-vs-A meas Δ=0.012 < fla non-det). Win is **offload** (delta-net offloaded; only 10 full-attn
layers recomputed via the existing SDPA path). qwen3/llama4 proven unaffected (A/B bit-identical 58057.31 MiB; newly-off=0).

### ✅ Qwen3.5-122B-A10B goal grid (4096·b4) — fix generalizes, win GROWS
| config | peak GiB | meas loss |
|---|---:|---|
| **C asym-offload sd1** | **46.4** 🏆 | 1.19 |
| B zero3_offload_mem | 50.4 | 1.10 |
| A asym-recompute | 50.9 | 1.19 |

C beats A by **4.5**, zero3_offload_mem by **4.0** (vs 35B's 2.5) — loss parity. **C peak 46.1 (35B) → 46.4 (122B)
despite ~3.5× params** — weights (~244 GiB) + activations offloaded → peak pinned to the ~45.5 loss floor. The win grows
with size (more activations that recompute must keep but offload doesn't). (C sd1 122B run had a cosmetic RC=1 in
table.md gen; peak read from memory_breakdown_summary.json; training succeeded.)

## 7. LAST STAGE — serious benchmark (run ONLY after the GOAL/V4 is met)
**Gate:** V4 green (C has the lowest `step_H` of A/B/C). Then run the canonical production sweep (still `ligerloss0`):
```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
MODEL_SPECS="Qwen/Qwen3.5-35B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp,asym_cpuadamwds|recomp,zero3_offload|recomp" \
ASYMM_EXP_ACT_POLICIES="none|true|true|false|true|false" \
WORKLOADS="4096|4|1,8192|8|1" MAX_STEPS=10 WARMUP_STEPS=5 PLOT=true OVERWRITE=true REQUIRE_SM100=1 \
PROFILE_MEMORY_BREAKDOWN=true PROFILE_MEMORY_SNAPSHOT=true OUTPUT_ROOT=profiling_qwen35_final \
bash scripts/lf/profile_lora_lf.sh
.venv/bin/python scripts/lf/show_metrics.py profiling_qwen35_final
```
**ACCEPT (deliverable) iff** the C row has the **lowest `step_H`** at **each** workload (incl. `s8192·b8`). That is the
proof of offload-cheaper-than-recompute for qwen3.5. Promote `profiling_qwen35_final` over `profiling_both_qwen35`.
