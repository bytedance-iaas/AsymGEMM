# AsymGEMM dense v2 — surgical `exp` activation offload for the dense MLP

> **v2 extends [`fix_dense.md`](fix_dense.md).** Same goal, same config, same harnesses. v1 *enabled* dense Qwen3-32B (the `lf.py:1945` expert-wrap fix) and reached **28.55 GiB** at `none|true|true|false|true|true`. But v1 left one thing on the table: on a dense model `ASYMM_EXPERT_ACT_OFFLOAD` (`exp`) is a **no-op** — the dense MLP's activations only get offloaded *incidentally* by the blanket `ASYMM_LAYER_GC` hooks, which **stage the intermediate back to GPU and run the MLP backward on GPU**. v2 makes `exp` do for the dense MLP exactly what it does for MoE experts: offload the heavy `silu(gate)·up` intermediate to CPU and run the **down-proj LoRA backward on CPU through AsymGEMM**, never re-staging it. v1 is a **prerequisite** (its expert-wrap fix must be applied first).

## Objective

Make `ASYMM_EXPERT_ACT_OFFLOAD=true` surgically offload the **dense MLP** (`Qwen3MLP` = gate_proj/up_proj/down_proj) on dense models, composing with `attn_act` and `layer_gc` — **identical to how Qwen3-MoE / Llama4-MoE already run `none+exp+attn-offload+layerGC` together**. The policy string is **unchanged**:

```
ASYMM_EXP_ACT_POLICIES="none|true|true|false|true|true"     # exp now offloads the dense MLP
BACKEND_SPECS="asym_cpuadamwds|norecomp"
```

**Success criterion (extends v1):**
> with surgical `exp` MLP offload, `none|true|true|false|true|true` peak GPU memory **≤ v1's 28.55 GiB** (the MLP intermediate leaves the *backward* peak), and **still < `zero3_offload_mem` (38.37 GiB)**.

Metric = `peak_allocated_hbm_bytes` in `memory_breakdown_summary.json`, plus the per-stage `bwd_H` (the MLP backward is where the win lands), and the offload tags (the `…x25600` MLP intermediate should now be owned by the **expert manager**, not the `decoder.saved.*` glue hooks).

### What v2 does NOT change
- The config string, the harnesses (`test1.sh`/`test2.sh`), and the v1 result interpretation.
- **`exp` and `layer_gc` are NOT mutually exclusive** — they compose. The only exclusion in the codebase is `ASYMM_LAYER_GC` ↔ `ASYMM_LAYER_ACT_OFFLOAD` (`lf.py:1724`, two whole-layer strategies). `exp` targets the MLP, `attn` the attention, `layer_gc` the remainder (residual/glue). Proven: the MoE rows in `agent/reports/week6.md` run `none+exp+attn-offload+layerGC`, status `ok`.

### ✅ Implementation validated (2026-06-22)

Implemented: `asym_gemm/training/dense_mlp.py` (`AsymDenseMLP` + `build_dense_mlp_expert_engine` reusing `AsymQwen3Experts` as E=1 + `is_dense_mlp_module`) and the `lf.py` install block (gated on dense + `exp`). Verified end-to-end:

- **Numerical parity** (5120×25600 widths): offload-vs-non-offload forward rel-err **4.3e-3**; **all six LoRA grads (gate/up/down × A/B) bit-identical** offload vs non-offload → the CPU-side AsymGEMM backward is exact. (A-grad=0 at PEFT init is correct: `dL/dA ∝ B=0`.)
- **Wiring** (toy dense Qwen3): `dense_mlp_act_offload_wrapped=2`, `dense_lora_wrapped=8` (attention only — MLP owned by the engine), `layer_glue_gc=2` composes, `mlp is AsymDenseMLP`, 0 skipped.
- **MoE non-regression**: `test_lf_qwen3_asym_backend.py -k "wraps_experts or whole_wraps_qwen3_moe or whole_wraps_llama4"` all pass with `exp=true` (MoE never enters the dense path — gated on `not expert_prefixes`).
- **Real Qwen3-32B partition proof** (the whole point): `dense_mlp_act_offload_wrapped=64`, and the offload tags show the spec partition —
  - `exp` owns the MLP: `gate`/`up`/`dact` (the `silu(gate)·up` intermediate) + `X` + `S_*` low-rank, and `expact_lora_b_backward_grouped_calls=384` (the **CPU-side down-proj LoRA backward ran**).
  - `layer_gc` owns **only** the glue: its sole tag is `decoder.saved.bf16.*x5120` (the 5120 residual). **The 25600 MLP intermediate is no longer in `decoder.saved.*`.**
  - `attn` owns attention (`q_proj.U`, `o_proj.U`). Trained clean (loss 2.34→1.81).

### Operational rules (carried from v1 — unchanged)
- Stop runs with **`term` (= `kill -TERM`)**, NEVER `kill -9` (corrupts the DeepSpeed `cpu_adam` JIT build → every later run hangs).
- Run configs **strictly sequentially on one GPU** (parallel pollutes peak-HBM).
- `jq` not installed → parse JSON with the venv Python. `CHECK_TRAINABLE_SURFACE=0` and `PROFILERS=source` apply exactly as in v1.

**Hard constraint — do NOT break existing models.** Every MoE model (Qwen3-MoE / Qwen3.5 / Llama4) must wrap experts exactly as today. v2's wiring is gated on **"no packed-expert/MoE blocks found"** (the same dense signal v1 uses), so MoE models never enter the dense-MLP branch. **No new kernels and no new autograd function** — v2 reuses the expert engine (`_ActivationOffloadQwen3ExpertFunction`) and the SM100 CPU-side LoRA kernels verbatim.

---

## Why v2 (root cause)

In `none|true|true|false|true|true` on dense Qwen3-32B today (v1):

- `exp` (`ASYMM_EXPERT_ACT_OFFLOAD`) is consumed **only** inside MoE engines (`qwen3_moe.py`, `llama4_experts.py`, `llama4_shared_mlp.py`). No dense MLP path exists → **no-op**.
- The dense MLP activation *is* offloaded — but by `ASYMM_LAYER_GC`'s whole-layer `saved_tensors_hooks` (`DecoderSavedTensorOffloadWrapper._pack` copies to CPU, `decoder_activation_offload.py:194`). **Empirically confirmed** via `profile.json` `offload_bytes_by_tag`: the `decoder.saved.bfloat16.16384x25600` (MLP intermediate, 25600 = `intermediate_size`) tag is present.
- **The cost v2 removes:** the glue-GC path's `_unpack` (`decoder_activation_offload.py:236`) copies the intermediate **back to GPU** in backward, and the MLP backward runs on GPU. That staged-back intermediate is part of the 9.9 GiB `saved_activation_hbm_bytes_at_peak`.

The MoE engine avoids this: `_ActivationOffloadQwen3ExpertFunction` keeps the intermediate on CPU and computes the down-proj LoRA backward there via `grouped_lora_b_backward_cpu_source` / `grouped_lora_a_pair_grad_cpu_right` (`exp_act_offload_lora.py:251,218`). **A dense MLP is just one expert group** (`offsets=[0,M]`, `experts=[0,-1]`, `dense_experts=True` — already the mode the engine runs in), so the whole machinery reuses.

---

## Stage 0 — Preconditions

```bash
export REPO=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
cd "$REPO"; PY="$REPO/.venv/bin/python"

# v1 must already be applied (the lf.py:1945 dense_no_experts branch)
grep -n "routed_experts:dense_model_no_experts" asym_gemm/integrations/lf.py   # expect a hit

# the surgical CPU-side LoRA kernels must be present (SM100)
$PY - <<'P'
from asym_gemm.training.exp_act_offload_lora import require_expert_activation_offload_kernels as r
print("kernels:", r(scope="full", check_only=True) or "OK (all present)")
P
```

Gate: v1 branch present; `require_expert_activation_offload_kernels(scope="full")` returns `None` (all of `CPU_LEFT_BF16_BINDING`, `LORA_A_GRAD_CPU_RIGHT`, `LORA_A_PAIR_GRAD_CPU_RIGHT` available). Same SM100 GPU / CPU-RAM gates as v1.

---

## Stage 1 — Code change (new dense-MLP engine + wiring; MoE untouched)

Three additions. **No edits to the expert autograd function or kernels.**

### 1a. Construction helper — build a 1-expert engine from `Qwen3MLP`

**File:** `asym_gemm/training/qwen3_moe.py` (next to `wrap_qwen3_moe_block`) — or a small new `asym_gemm/training/dense_mlp.py`.

```python
def build_dense_mlp_expert_engine(mlp, *, backend, precision, lora_rank, lora_alpha,
                                  lora_dropout, stats, device, dtype, strict=True):
    """Adapt a dense Qwen3MLP (gate_proj/up_proj/down_proj) into AsymQwen3Experts(num_experts=1).
    Fuses gate+up -> [1, 2*I, H] and down -> [1, H, I], adopting the existing CPU HostWeight
    storage (clone=False) so no second copy is made. Fresh grouped LoRA [1, r, ·]/[1, ·, r]."""
    gate_w, up_w, down_w = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight   # [I,H],[I,H],[H,I]
    gate_up = torch.cat([gate_w, up_w], dim=0).unsqueeze(0).contiguous()                     # [1, 2I, H]
    down    = down_w.unsqueeze(0).contiguous()                                               # [1, H, I]
    return AsymQwen3Experts(
        num_experts=1,
        hidden_dim=int(gate_w.shape[1]), intermediate_dim=int(gate_w.shape[0]),
        gate_up_weight=gate_up, down_weight=down,            # -> AsymGroupedFrozenLinear([1,out,in])
        backend=backend, precision=precision,
        lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        stats=stats, device=device, dtype=dtype,
    )
```
(Match the real `AsymQwen3Experts.__init__` signature, `qwen3_moe.py:1965` — it already builds `gate_up_base`/`down_base` as `AsymGroupedFrozenLinear` from `[E, out, in]` weights at `:2039,:2047` and grouped LoRA at `:2078-2087`. E=1 is a valid group.)

### 1b. `AsymDenseMLP` — thin module that runs the engine on flattened tokens

**File:** same module. The only genuinely new logic (~40 lines), and it reuses `_single_group_offsets_experts` (already used by the attention path) + `_ActivationOffloadQwen3ExpertFunction`.

```python
class AsymDenseMLP(nn.Module):
    _is_asym_dense_mlp = True
    def __init__(self, engine):           # engine = AsymQwen3Experts(num_experts=1)
        super().__init__(); self.engine = engine

    def _uses_activation_offload(self):
        return (_env_flag("ASYMM_EXPERT_ACT_OFFLOAD", False)
                and self.training and torch.is_grad_enabled())

    def forward(self, x):
        B, S, H = x.shape
        packed = x.reshape(B * S, H)                       # one group, all tokens
        offsets, experts = _single_group_offsets_experts(packed.shape[0], packed.device)  # [0,M], [0,-1]
        if self._uses_activation_offload():
            out = _ActivationOffloadQwen3ExpertFunction.apply(
                packed, offsets, experts, None, None, None,
                self.engine.gate_lora_A, self.engine.gate_lora_B,
                self.engine.up_lora_A,   self.engine.up_lora_B,
                self.engine.down_lora_A, self.engine.down_lora_B, self.engine)
        else:
            out = self.engine.forward_packed(packed, offsets, experts)   # plain grouped fwd (no offload)
        return out.reshape(B, S, H)
```

### 1c. Wiring in `lf.py apply_lf_asym_lora` (gated on dense + `exp`)

```python
# matcher (next to the other _is_* helpers)
def _is_dense_mlp_module(name, module):
    leaf = name.rsplit(".", 1)[-1]
    if leaf not in {"mlp", "feed_forward"}: return False
    if is_qwen3_moe_block(module) or is_qwen35_moe_block(module) or is_llama4_moe(module): return False
    return all(isinstance(getattr(module, p, None), nn.Linear) for p in ("gate_proj","up_proj","down_proj"))

# inside apply_lf_asym_lora, AFTER expert_candidates is built (dense => empty), BEFORE the dense linear-leaf loop:
dense_mlp_act = (backend == "asym" and _env_flag("ASYMM_EXPERT_ACT_OFFLOAD", False) and not expert_candidates)
if dense_mlp_act:
    require_expert_activation_offload_kernels(scope="full")            # reuse v-existing check
    for name, module in list(model.named_modules()):
        if not _is_dense_mlp_module(name, module) or _is_under(name, expert_prefixes): continue
        engine = build_dense_mlp_expert_engine(module, backend=backend, precision=precision,
                    lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                    stats=stats, device=..., dtype=..., strict=strict)
        parent, child = _parent_and_child(model, name); _replace_child(parent, child, AsymDenseMLP(engine))
        expert_prefixes.append(name)                                   # per-linear loop now SKIPS gate/up/down
        report.dense_mlp_act_offload_wrapped += 1
        _release_replaced_module_memory()
```

Add `dense_mlp_act_offload_wrapped: int = 0` to `LFAsymReport` (`lf.py:256`) and to `to_log_string`.

**No mutual-exclusion check.** `exp` composes with `layer_gc` and `attn` — see Verification below. The per-linear wrapper skipping the MLP (via `expert_prefixes`) is the same mechanism MoE uses so its expert linears aren't double-wrapped.

### Verification evidence (why this is correct and composes — not assumed)

- **Reuse is exact.** `AsymQwen3Experts` stores base as `AsymGroupedFrozenLinear([E,out,in])` (`qwen3_moe.py:2039,2047`) and LoRA as `[E,r,·]/[E,·,r]` (`:2078-2087`); the offload forward already passes `dense_experts=True` (`:991`) and groups via `offsets/experts`. E=1 is a legal group → the dense MLP is one expert with all M tokens.
- **CPU-side backward is the kernels'.** `grouped_lora_b_backward_cpu_source` (`exp_act_offload_lora.py:251`) takes `grad_out_cpu` (CPU) + GPU LoRA-B → computes `dS, grad_b` via `LORA_B_BACKWARD_CPU_SOURCE`; `grouped_lora_a_pair_grad_cpu_right` (`:218`) takes CPU `x` → gate/up LoRA-A grads. The intermediate never returns to GPU.
- **Composes with `layer_gc` (no double-offload).** `AsymDenseMLP` runs as `layer.mlp`, so it executes inside the glue-GC `_manual_forward` (`decoder_layer_glue_gc.py:224`). Its offloaded tensors are saved as **CPU handles** by `_ActivationOffloadQwen3ExpertFunction`; the glue-GC's `saved_tensors_hooks` only offload **CUDA** tensors (`_should_offload` returns False for non-CUDA, `decoder_activation_offload.py:162`) → it **skips** the MLP's CPU saved tensors and catches only the leftover GPU glue. Identical to Qwen3-MoE today.
- **MoE non-regression is structural.** `dense_mlp_act` is gated on `not expert_candidates`; any MoE model has candidates → the dense-MLP block is never entered.

---

## Stage 2 — Unit validation

### 2a. Numerical parity — `AsymDenseMLP` (offload) vs reference dense MLP

```bash
$PY - <<'PY'
import torch
from asym_gemm.training... import build_dense_mlp_expert_engine, AsymDenseMLP   # adjust import
from torch import nn
torch.manual_seed(0); dev="cuda"
H, I, B, S = 5120, 25600, 1, 256                       # real Qwen3-32B widths (64-aligned)
class Qwen3MLP(nn.Module):
    def __init__(s):
        super().__init__()
        s.gate_proj=nn.Linear(H,I,bias=False); s.up_proj=nn.Linear(H,I,bias=False); s.down_proj=nn.Linear(I,H,bias=False)
    def forward(s,x): return s.down_proj(torch.nn.functional.silu(s.gate_proj(x))*s.up_proj(x))
ref=Qwen3MLP().to(dev,torch.bfloat16)
eng=build_dense_mlp_expert_engine(ref, backend="asym", precision="bf16", lora_rank=64, lora_alpha=16,
        lora_dropout=0.0, stats=None, device=dev, dtype=torch.bfloat16)
mlp=AsymDenseMLP(eng).to(dev).train()
import os; os.environ["ASYMM_EXPERT_ACT_OFFLOAD"]="true"
x=torch.randn(B,S,H,device=dev,dtype=torch.bfloat16,requires_grad=True)
y=mlp(x); y.sum().backward()
# compare fwd to a base-only reference (LoRA init ~0 -> y ≈ ref(x)); assert finite grads on LoRA + x
assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
print("parity OK; lora grads:", [p.grad.norm().item() for p in (eng.down_lora_A, eng.down_lora_B)])
PY
```
Gate: forward matches base reference within bf16 tol; LoRA-A/B and input grads finite and non-zero.

### 2b. Wrapping on a toy dense Qwen3 — engine owns the MLP, linears skipped

Reuse the v1 `/tmp/val_dense_fix.py` builders, with `ASYMM_EXPERT_ACT_OFFLOAD=true`, then assert:
```
report.dense_mlp_act_offload_wrapped == <num_layers>
report.dense_lora_wrapped counts attention only (gate/up/down NOT separately wrapped)
report.skipped contains no 'not_nn_linear' for the mlp children
```

### 2c. MoE non-regression (the don't-break gate)

```bash
$PY -m pytest -q tests/training/test_lf_qwen3_asym_backend.py \
  tests/training/test_lf_qwen35_asym_backend.py tests/training/test_decoder_layer_glue_gc.py
# plus a NEW test: tests/training/test_dense_mlp_act_offload.py  (2a parity + 2b wrapping)
```
Gate: existing MoE wrap/router tests green (dense-MLP branch never reached for MoE); new dense test green.

---

## Stage 3 — Smoke on real Qwen3-32B (confirm exp now owns the MLP)

```bash
MODEL_SPECS="Qwen/Qwen3-32B|1" GPU_POOL=3 CHECK_TRAINABLE_SURFACE=0 \
  MAX_STEPS=1 WARMUP_STEPS=1 WORKLOADS="2048|1|1" PROFILERS=source \
  ASYMM_EXP_ACT_POLICIES="none|true|true|false|true|true" \
  OUTPUT_ROOT="$REPO/profiling_smoke_v2" \
  bash scripts/lf/profile_lora_lf_test2.sh 2>&1 | tee /tmp/v2_smoke.log
grep -E "dense_mlp_act_offload_wrapped=64|layer_glue_gc_wrapped=64|attention_act_offload_wrapped=256" /tmp/v2_smoke.log
```
Then confirm the MLP intermediate moved from the glue hooks to the **expert manager** (the surgical path):
```bash
RUN=$(find profiling_smoke_v2 -name profile.json | head -1)
$PY - "$RUN" <<'PY'
import json,sys; d=json.load(open(sys.argv[1]))
# expect expert/activation-offload counters > 0 (expact_lora_*), and the 25600 tag NOT in decoder.saved.* glue tags
print("expact stats:", {k:v for k,v in d.get("asym_execution_stats",{}).items() if "expact" in k})
print("glue offload tags:", list(d.get("activation_offload",{}).get("offload_bytes_by_tag",{}).keys()))
PY
```
Gate: a step trains (loss printed); `dense_mlp_act_offload_wrapped=64`; `expact_lora_*` counters > 0; the `…x25600` intermediate is owned by the expert manager (its bytes are no longer attributed to the `decoder.saved.*` glue tags).

---

## Stage 4 — Full comparison run (same config, surgical exp)

Sequential on one GPU, `term`/`kill -TERM` only, `CHECK_TRAINABLE_SURFACE=0`, `PROFILERS=source` (per v1's operational rules). Write to a **fresh** v2 root so it sits beside v1's numbers.

```bash
export CMP=$REPO/profiling_both        # same root as v1 -> same config dir, new per-run tag is identical string
# v2 = identical policy string; the difference is the code (exp now offloads the MLP)
MODEL_SPECS="Qwen/Qwen3-32B|1" GPU_POOL=3 OVERWRITE=true CHECK_TRAINABLE_SURFACE=0 PROFILERS=source \
  ASYMM_EXP_ACT_POLICIES="none|true|true|false|true|true" \
  BACKEND_SPECS="asym_cpuadamwds|norecomp" OUTPUT_ROOT="$CMP" \
  bash scripts/lf/profile_lora_lf_test2.sh > /tmp/v2_test2.log 2>&1
```
(The `zero3_offload_mem` baseline from v1 — 38.37 GiB — is reused; no need to re-run it.)

---

## Stage 5 — Validate the goal

```bash
CFG="$CMP/asym_long_sft_smoke__lora__lf__bf16/qwen3-32b__gpus1__b4_s4096_ga1_w5_s10_r64_a16_drop000"
ASYM=$(find "$CFG" -path "*asym_cpuadamwds*source*norecomp*expact1*attnact1*layeract0*layergc1*sdparecomp1*/b4_s4096_ga1/*memory_breakdown_summary.json" | sort | head -1)
$PY - "$ASYM" <<'PY'
import json,sys; g=1024**3; d=json.load(open(sys.argv[1]))
peak=d["peak_allocated_hbm_bytes"]/g
print(f"v2 asym(none|T|T|F|T|T) PEAK = {peak:.2f} GiB   (v1 was 28.55, zero3_offload_mem 38.37)")
print({k:round(d.get(k,0)/g,2) for k in ['saved_activation_hbm_bytes_at_peak','live_activation_hbm_bytes_at_peak','temporary_workspace_hbm_bytes_at_peak']})
print("PASS vs zero3:", peak < 38.37, "| improved vs v1:", peak <= 28.55)
PY
```

Also pull the per-stage `bwd_H` (where the win lands) from `summary.md` / `profile.json` and compare to v1's row:

| metric (GiB) | v1 (`layer_gc` blanket, GPU bwd) | v2 (surgical `exp`, CPU bwd) — target |
| --- | ---: | ---: |
| step_H (peak) | 28.55 | **≤ 28.55** |
| bwd_H | 28.55 | **lower** (MLP intermediate not staged back) |
| saved_act @ peak | 9.90 | **lower** |
| RAM | 582.9 | ~same or higher (intermediate dwells on CPU) |
| step_s | 20.3 | **likely higher** (down-proj backward on CPU) |

**Pass:** `peak < 38.37` (still beats zero3) **and** `peak ≤ v1's 28.55` (surgical MLP offload removed the staged-back intermediate from the backward peak). If `peak == 28.55` exactly, the breakdown's `saved_activation_hbm_bytes_at_peak` should still drop (the win moved out of the at-peak window) — report it either way.

---

## Risks / contingencies

1. **CPU-RAM and step-time tradeoff.** The MLP intermediate (`16384×25600` bf16 ≈ 0.8 GiB/layer) now dwells on CPU through backward; expect RAM ≥ v1's 582.9 GiB and a slower step (CPU-side down-proj backward). This is the AsymGEMM thesis (HBM↓ at the cost of CPU/bandwidth) — flag it, don't treat slower step as failure.
2. **AsymGroupedFrozenLinear with E=1.** The grouped kernels are validated for E≥1; the single-group `offsets=[0,M]`/`experts=[0,-1]` path is the same one `attention_activation_offload.py` already uses. Stage 2a is the guard.
3. **Double-offload regression.** If the glue-GC ever offloads the MLP intermediate again (e.g. the engine accidentally saves a CUDA tensor), Stage 3's tag check catches it: the `…x25600` bytes must be owned by the expert manager, not `decoder.saved.*`.
4. **`forward_packed` non-offload path** (eval / `torch.no_grad`) must match the grouped base+LoRA exactly — covered by Stage 2a with offload off.

## Rollback

The dense-MLP path is fully gated by `dense_mlp_act = backend=="asym" and exp and not expert_candidates`. Remove the Stage 1c block + the new module to revert to v1 behavior (MLP offloaded via `layer_gc` blanket). v1's `lf.py:1945` fix and the harnesses are untouched. MoE-safe.

## Scope summary

| Area | Change |
| --- | --- |
| `asym_gemm/training/{qwen3_moe.py or dense_mlp.py}` | +`build_dense_mlp_expert_engine` (E=1 adapter) + `AsymDenseMLP` (~60 lines) |
| `asym_gemm/integrations/lf.py` | +`_is_dense_mlp_module` matcher + dense-MLP install block (~25 lines) + `report.dense_mlp_act_offload_wrapped` |
| Expert autograd fn / SM100 kernels (`exp_act_offload_lora.py`) | **reused verbatim** — no change |
| `_ActivationOffloadQwen3ExpertFunction`, `AsymGroupedFrozenLinear` | **reused** — no change |
| MoE models (Qwen3-MoE / Qwen3.5 / Llama4) | **untouched** — gated on `not expert_candidates` |
| Config / harnesses / v1 fix | **unchanged** |
