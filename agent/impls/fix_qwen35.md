# Qwen3.5 (qwen3_5-35b-a3b): why asym-offload loses to recompute, and the fixes

All numbers are **snapshot-exact** (per-allocation stack frames) from `profiling_qwen35_attrib`
(2-step, `ligerloss0`, `PROFILE_MEMORY_SNAPSHOT=true`). s4096·b4. Don't change qwen3/llama4.

## 🚫 HARD RULE: NO LIGER / NO FUSED OR CHUNKED CE — EVER, IN ANY CONFIG
Every run is `ligerloss0` / `ENABLE_LIGER_KERNEL=false`. The ~45.5 GiB CE loss is **off the table** — we do not
reduce it, not even "fairly for all configs". It is identical in A/B/C, so it cancels. **The comparison is decided
only by NON-LOSS.** Any plan step that touches the loss is out of scope.

## 🎯 GOAL
**The asym no-recompute offload config `asym_cpuadamwds | norecomp | none|true|true|false|true|true` MUST beat
asym-recompute (A) and zero3-offload-recompute (B)** — all `ligerloss0`.

### The numbers — `scripts/lf/show_metrics.py profiling_both_qwen35` (s4096·b4, ligerloss0)
`step_H` = **peak GPU HBM (GiB)** = the metric that matters · `step_s` = step time (s) · `RAM` = host peak (GiB).

| backend / config | flag tuple `pol\|exp\|attn\|layer\|gc\|sdpa` | step_s | **step_H** (peak) | RAM | non-loss |
|---|---|---:|---:|---:|---:|
| **C** asym `norecomp` exp+attn-offload+layerGC **(TARGET)** | `none\|true\|true\|false\|true\|true` | 30.9 | **57.9** | 275.8 | **12.4** |
| A asym `recomp`, no offload | (recompute) | 16.6 | **48.5** | 189.2 | ~3–4 |
| B zero3-offload `recomp`, no offload | (recompute) | 14.8 | **50.9** | 192.1 | ~4–6 |

Today **C (57.9) LOSES to both A (48.5) and B (50.9)** — that is the bug. (The other offload variants — layerOF, sd−
— all also read 57.9; `s8192·b8` ⚠️ failed.)

**Success = C non-loss < A non-loss by a clear margin**, with loss parity. Measure on **non-loss** (the 45.5 GiB loss
is constant across A/B/C and cancels — see hard rule); absolute peak stays ~46–48 GiB for all and we don't chase it.

### Expected reduction (projection — confirm via validation)
Only the **non-loss 12.4** is attackable (loss 45.5 = hard floor). It is: delta-net mixer **7.51** + norm/residual **4.31** + experts 0.28 + other 0.3.

| stage | knocks off | non-loss | **peak** | vs A 48.5 / B 50.9 |
|---|---:|---:|---:|---|
| today C | — | 12.4 | **57.9** | +9.4 / +7.0 (loses) |
| + CHANGE 1 (recompute mixer) | ~9–10 | ~3 | **~48** | ≈ tie A |
| + CHANGE 3 (offload residual) | ~2 | ~1 | **~46.5** | **−2 / −4.4 (wins)** |
| + CHANGE 4 (in_proj) | ~0 (perf only) | ~1 | ~46.5 | — |

**Total ≈ 11 GiB off (57.9 → ~46.5)** — ~92% of non-loss, ~20% of peak (loss floor dominates). A's exact non-loss is pending its snapshot; a higher A widens C's margin.

---

## 1. Root cause (exact)

Peak 57.9 = **loss/CE 45.5** (constant) + **non-loss 12.4**. The non-loss, by stack frame:

| non-loss component | GiB | frame |
|---|---:|---|
| delta-net mixer | **7.51** | `decoder_layer_glue_gc.py:129 _call_qwen35_linear_attention` |
| norm / residual saves | **4.31** | `decoder_layer_glue_gc.py:178 _checkpoint_norm` |
| routed_experts | 0.28 | (a non-issue) |
| real attention + other | 0.3 | |

Qwen3.5 is hybrid: **30 of 40 layers are `Qwen3_5MoeGatedDeltaNet`** (fla linear attention), 10 full-attention
(`full_attention_interval=4`). The offload path leaves the delta-net mixer resident because:
- **`sdpa_recompute.py` only recomputes attention via `ALL_ATTENTION_FUNCTIONS`**, which only the 10 full-attn
  layers use. The 30 delta-net layers call `chunk_gated_delta_rule` directly → never recomputed. (This is why
  toggling `sdparecomp` changed nothing: 57.89 vs 57.90.)
- **The glue-GC checkpoints only the 2 layernorms**, not the mixer. The delta-net's gated-norm output / out_proj
  input (`[16384×4096]`=128 MiB/layer) is a *live residual-stream tensor*, not an offloadable saved tensor, so the
  saved-tensor offload (which does move the fla kernel's *inputs*, ~16 GiB/step) can't remove it.

By contrast qwen3-30b/llama4 are pure full-attention → this non-loss term is ~0 for them, so their lightweight
offload path already wins. **Full-layer recompute (A/B) zeroes this term** — which is exactly why offload loses today.

> Note: `routed_experts` is **0.28 GiB** resident. An earlier *inferred* breakdown showed "8.98" — that was the
> proportional split mis-assigning the unframed CE scratch to experts; the snapshot disproves it. The shared expert
> engine is fine (no change).

---

## 2. Fixes — each stage has an explicit validation run + pass gate; do not accept a stage until its gate is green

`run_cfg` is defined in §3. All runs are `ligerloss0`. Read snapshot-exact rows (§3 b/d), not inferred workspace.

### CHANGE 0 (prerequisite) — pin the references A and B with the snapshot
Targets for everything below; A/B non-loss are currently only known via the (inflated) inferred method.
```bash
run_cfg profiling_qwen35_fix "asym_cpuadamwds|recomp" "none|false|false|false|false|false"   # A asym-recompute
run_cfg profiling_qwen35_fix "zero3_offload|recomp"   "none|false|false|false|false|false"   # B zero3
```
**Record:** A_nonloss, B_nonloss (harness b/d). **Gate:** both runs OK and produce `memory_breakdown.jsonl`.

### CHANGE 1 — recompute the delta-net mixer (frees ~11.8 GiB → ties A)
**Do:** `asym_gemm/training/delta_net_recompute.py::install_delta_net_recompute(model)` — checkpoint the
`input_layernorm`→`linear_attn` sub-block from the residual input (`torch.utils.checkpoint(use_reentrant=False,
preserve_rng_state=True)`, train+grad only, like `sdpa_recompute.py:25`). Wire next to `install_sdpa_recompute(model)`
(`lf.py:1525`) under the same `ASYMM_ATTN_SDPA_RECOMPUTE` gate.
**Validate (run exactly):**
```bash
run_cfg profiling_qwen35_fix "asym_cpuadamwds|norecomp" "none|true|true|false|true|true"   # recompute ON  (sd1)
run_cfg profiling_qwen35_fix "asym_cpuadamwds|norecomp" "none|true|true|false|true|false"  # control       (sd0)
```
**PASS iff** (harness b/d/c): sd1 delta-net mixer frame **≤0.5 GiB** AND norm frame **≤3.0 GiB** AND peak **≤49 GiB**;
sd0 stays **~57.9 GiB** (proves the flag did it); loss parity sd1-vs-sd0 **max|Δloss| ≤1e-2** (step-0 identical).
Loss drift ⇒ RNG not preserved in checkpoint — fix before continuing.

### CHANGE 2 — confirm the recompute input is offloaded (check only, reuse CHANGE 1's sd1)
**Validate:** harness (d) on the sd1 leaf.
**PASS iff** `live_activation_bytes_at_peak[linear_attention]` **≤1.5 GiB** (the saved layer-input didn't reappear
resident). If it fails: route that input through the decoder saved-tensor offload, re-run sd1.

### CHANGE 3 — make `layeract` offload the residual stream to CPU (this is what BEATS A)
**Do:** `layeract1` is inert today (`enabled, wrapped=40`, frees nothing). Make it copy each layer's residual output
to pinned CPU and fetch in backward (reuse attn-act/saved-tensor machinery), ≤1 layer in flight.
**Validate (run exactly):**
```bash
run_cfg profiling_qwen35_fix "asym_cpuadamwds|norecomp" "none|true|true|true|false|true"   # layeract + delta-net recompute
```
**PASS iff** (harness b/d/c): C non-loss **< A_nonloss** (from CHANGE 0) by **≥2 GiB**; residual-stream frames
resident **≤1 GiB**; loss parity vs A **max|Δloss| ≤1e-2**. (Absolute peak ≈46–47 vs A ~48.5 — expected, loss-bounded.)

### CHANGE 4 — `in_proj_a`/`in_proj_b` 64-alignment (minor)
**Do:** out_features = `num_v_heads` = 32 → not 64-aligned → torch fallback (`lf.py:1142`) on all 30 layers (tiny,
not a memory driver). Keep these two on the normal GPU LoRA path (don't offload), or pad to 64.
**Validate:**
```bash
grep -c "in_proj_[ab]:torch_cpu_fetched" profiling_qwen35_fix/**/train.log
```
**PASS iff** count **= 0** (was 60) AND CHANGE 1/3 peak+loss gates still hold (no regression).

### CHANGE 5 — DROPPED (no validation needed)
Experts are 0.28 GiB resident (snapshot §1). No expert-engine change.

### CHANGE 6 — final fair grid
**Validate (run exactly):**
```bash
run_cfg profiling_qwen35_fix "asym_cpuadamwds|norecomp" "none|true|true|true|false|true"      # C final (CHANGE 1+3)
# A and B already from CHANGE 0
.venv/bin/python scripts/lf/show_metrics.py profiling_qwen35_fix
```
Also triage the `s8192·b8` non-OOM failure (its `train.log` tail).
**FINAL PASS iff (the GOAL):** C non-loss **< A_nonloss AND < B_nonloss by a clear margin**, loss parity vs A. Then
offload-cheaper-than-recompute holds for qwen3.5 at `ligerloss0`.

---

## 3. Validation harness

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
run_cfg () {  # run_cfg <OUTROOT> <BACKEND_SPEC> <POLICY: policy|expact|attnact|layeract|layergc|sdparecomp>
  DRY_RUN=false OVERWRITE=true PLOT=false PREPARE_DATASETS=false REQUIRE_SM100=1 GPU_POOL=0 \
  MODEL_SPECS="Qwen/Qwen3.5-35B-A3B|1" BACKEND_SPECS="$2" ASYMM_EXP_ACT_POLICIES="$3" \
  WORKLOADS="4096|4|1" MAX_STEPS=6 WARMUP_STEPS=2 \
  PROFILE_MEMORY_BREAKDOWN=true PROFILE_MEMORY_SNAPSHOT=true \
  PROFILE_LIVE_ACTIVATION_DETAILS=true PROFILE_LIVE_ACTIVATION_TOPK=100 \
  OUTPUT_ROOT="$1" bash scripts/lf/profile_lora_lf.sh
}
# CHANGE 1 A/B:
run_cfg profiling_qwen35_fix "asym_cpuadamwds|norecomp" "none|true|true|false|true|true"    # delta-net recompute ON
run_cfg profiling_qwen35_fix "asym_cpuadamwds|norecomp" "none|true|true|false|true|false"   # control
# References (run once):
run_cfg profiling_qwen35_fix "asym_cpuadamwds|recomp"   "none|false|false|false|false|false"  # A
run_cfg profiling_qwen35_fix "zero3_offload|recomp"     "none|false|false|false|false|false"  # B
```
Always read **snapshot-exact** rows, not inferred `temporary_workspace` (treat that as transient scratch w/ no owner).

```bash
# (b) peak + non-loss per run
.venv/bin/python scripts/lf/show_metrics.py profiling_qwen35_fix

# (d) exact per-component + per-tensor at peak (LEAF = the .../b4_s4096_ga1 dir)
.venv/bin/python - "$LEAF/memory_breakdown.jsonl" <<'PY'
import json,sys
recs=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
r=max((x for x in recs if x.get("phase")=="after_backward" and not x.get("is_warmup")),
      key=lambda x:x.get("peak_allocated_within_phase",0)); G=2**30
f=lambda d:{k:round(v/G,2) for k,v in (d or {}).items() if isinstance(v,(int,float)) and v>5e7}
print("saved",f(r.get("saved_activation_bytes_at_peak")),"live",f(r.get("live_activation_bytes_at_peak")))
for x in sorted(r.get("live_activation_detail_rows_at_peak") or [],key=lambda z:-z.get("bytes",0))[:10]:
    print(f"  {x.get('bytes',0)/G:5.2f} GiB {x.get('name') or x.get('module')}")
PY
# snapshot stacks: peak_snapshot_attrib_allblocks.md (components/frames) + memory_snapshot.pickle

# (c) loss parity: pull step_samples[].loss from each run's source_profile.json; max|Δ| ≤ 1e-2
```

---

## 4. Liger — OUT OF SCOPE (see the hard rule at top)
We do **not** use liger/fused/chunked CE in any config. (FYI only: it's also currently unsupported for qwen3.5 —
`install_asym_liger_loss_bridge` `liger_loss.py:581` has no `qwen3_5_moe` branch — but that's irrelevant since we
never enable it. The 45.5 GiB loss stays; the win is on non-loss.)

---

## 5. LAST STAGE — serious benchmark (run ONLY after the GOAL is met)
**Gate to enter this stage:** CHANGES 1–6 green on the quick validation, i.e. **C (`none|true|true|false|true|true`)
has the lowest non-loss / lowest `step_H` of A, B, C.** Do **not** run this before then — it's the heavy, multi-config,
multi-hour, production-settings sweep (real warmup/steps, full plots), not the smoke `run_cfg`.

Run the canonical `profile_lora_lf.sh` benchmark (still `ligerloss0`), at production settings + both workloads:
```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
MODEL_SPECS="Qwen/Qwen3.5-35B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp,asym_cpuadamwds|recomp,zero3_offload|recomp" \
ASYMM_EXP_ACT_POLICIES="none|true|true|false|true|true" \
WORKLOADS="4096|4|1,8192|8|1" \
MAX_STEPS=10 WARMUP_STEPS=5 PLOT=true OVERWRITE=true REQUIRE_SM100=1 \
PROFILE_MEMORY_BREAKDOWN=true PROFILE_MEMORY_SNAPSHOT=true \
OUTPUT_ROOT=profiling_qwen35_final \
bash scripts/lf/profile_lora_lf.sh
.venv/bin/python scripts/lf/show_metrics.py profiling_qwen35_final
```
**ACCEPT (deliverable) iff** in that table the C row has the **lowest `step_H`** of all backends at **each** workload
(including `s8192·b8`, which must now also succeed). That is the official proof of offload-cheaper-than-recompute for
qwen3.5. Promote `profiling_qwen35_final` over the stale `profiling_both_qwen35` once accepted.
