# Fair activation-offload baselines: `off-layer` (floor) + generic same-policy (apples-to-apples)

AsymGEMM's selective offload (`exp+attn`) has **no fair baseline** in LF: `recomp`/`unsloth` only recompute or
offload the *layer boundary*, never the intra-layer expert/attention stack. Two baselines fix this:

1. **`off-layer`** — generic **whole-layer** activation offload (offload twin of `gc-layer`) via
   `torch.autograd.graph.save_on_cpu`, no recompute. The "how low/slow can naive full offload go" **floor**.
2. **Generic same-policy** — the same `none|T|T|F|T|T` *policy* (offload attn+exp, recompute norms+sdpa) implemented
   with **one generic `save_on_cpu`** offload instead of AsymGEMM's custom offload. The **apples-to-apples**: same
   activations offloaded + same recompute policy; generic vs custom offload *implementation*. (Not a literal
   pack/unpack swap — see "Mechanism" + scope caveat.)

> **🎯 Goal / deliverable:** the point of this whole doc is the **final comparison metrics** — **one table per
> model**, in this exact format:
> `| Workload | Backend | Config | Status | fwd_s | bwd_s | opt_s | step_s | fwd_H | bwd_H | step_H | RAM |`
> comparing **AsymGEMM (custom offload) vs #2 (generic same-policy) vs #1 (off-layer floor)** (`_s`=sec, `_H`=GPU HBM
> peak GiB, `RAM`=host RSS GiB — `show_metrics.sh` emits exactly these columns). Stages 0–5 exist only to make that
> table **fair**; Stage 6 produces it (`profile_lora_lf_test.sh` → `show_metrics.sh`). Success = asym ≤ #2 on both
> peak HBM and step time, with #1 as the low-HBM/slow floor.

> **Verified against code + standalone tests, 2026-06-23.** See "Validation" below; all line refs checked.

## ⚠️ Architecture constraint (drives the design)
The AsymGEMM LF integration (`asym_gemm/integrations/lf.py:apply_lf_asym_lora`) runs **only when
`model_args.use_asym_gemm=true`** — `LlamaFactory/.../model/adapter.py:482` → `adapt_lf_asym_peft_lora`. The harness
sets `--use_asym_gemm true` **only** for `BACKEND ∈ {asym, asym_torch}` (`run_lf_lora_sft.sh:1637,1642`); the asym
branch **returns early (adapter.py:525)**. So `zero3_offload`/`superoffload_mem` never touch `lf.py` → the baselines
**must be a backend-agnostic hook**, applied at the **end of `init_adapter` before the final `return model`** (runs
for the non-asym paths). (`USE_ASYM_GEMM=1` *env* at `run_lf_lora_sft.sh:1914` is the asym runtime/logging flag —
distinct from the `--use_asym_gemm` CLI arg that drives the :482 gate, which stays false for zero3/superoffload.)

## Mechanism, resolved (how glue-GC + offload actually compose)
`DecoderLayerGlueGCWrapper` (`asym_gemm/training/decoder_layer_glue_gc.py`) reconstructs the layer forward and:
- **checkpoints only the norms** (`_checkpoint_norm` :168 wraps just `norm(x)`); **attn + mlp/experts run normally**
  (:214/:222, NOT checkpointed);
- runs the whole manual forward under **`saved_tensors_hooks(pack, unpack)`** (:199) — that *is* the offload.

`torch.autograd.graph.save_on_cpu(pin_memory=True)` **is** `saved_tensors_hooks` with CPU pack/unpack. So #2 = a
glue-GC wrapper whose hooks are `save_on_cpu` — same forward + norm-GC, generic offload of everything the layer
saves (attn+exp). No big module is ever checkpointed → **no offload-inside-recompute problem, no `context_fn`
needed** (proven below).

**Scope note (important):** asym's `none|T|T|F|T|T` does **not** offload via glue-GC alone — it layers *three* custom
mechanisms: attention saved-tensor offload (`lf.py:2252`, attn_act=T), MoE-block expert offload (`:1851`), and
glue-GC (`:2306`); `layer_act` offload is **off** (F). #2 consolidates these into **one** generic whole-forward
`save_on_cpu`. So #2 is **not** a literal pack/unpack swap of asym — it's a generic offload of the *same activations*
(attn+exp) under the *same* recompute policy. Fair comparison ("custom offload machinery vs generic, same policy"),
but **verify offloaded-bytes / peak-HBM scope parity** at impl (caveats).

## Validation (standalone, before touching production code)
`/tmp/test_saveoncpu*.py` on a toy layer (`checkpoint(norm)` + a large linear), bf16, CUDA:
- **Correctness:** `save_on_cpu` around a forward containing `checkpoint(norm)` → forward identical, input-grad &
  weight-grad **bit-identical** (max|Δ| = 0.00e+00) vs no-offload. ✅ glue-GC ↔ save_on_cpu composes exactly.
- **Memory (12 layers, ~128 MiB act/layer):** peak HBM **4768 → 2176 MiB (−54%)**, grads still bit-identical. ✅
- **Caveat (1 layer):** peak *rose* (+864 MiB) — offload only pays off with multi-layer accumulation; expect
  overhead, not savings, at tiny depth. Real models (48 layers) are firmly in the win regime.

## Reuse (verified import paths)
- Matchers (`lf.py`): `_is_qwen3_decoder_layer_module_name` (:1393; already covers qwen3/qwen35/dense **and llama4**),
  `_is_text_attention_module_name`.
- `DecoderLayerGlueGCWrapper` / `install_decoder_layer_glue_gc` — `asym_gemm/training/decoder_layer_glue_gc.py:239`
  (stock-HF-compatible for qwen3/qwen35/llama4: rebuilds forward from `self_attn`/`mlp`, dispatch by class name).
- `install_sdpa_recompute` — `asym_gemm/training/sdpa_recompute.py:39` (**self-gates on `ASYMM_ATTN_SDPA_RECOMPUTE`**).
- Policy parser: `asym_gemm/training/moe.py:parse_expert_recompute_policy_spec` (gc-layer @ :619).
- Env (emitted by harness, `run_lf_lora_sft.sh:1985-2005`): `ASYM_GEMM_LF_CONFIG_EXPERT_POLICY`,
  `…_ASYMM_LAYER_GC`, `…_ASYMM_ATTN_SDPA_RECOMPUTE`.

---

## Stage 0 — new backend-agnostic module
`asym_gemm/integrations/generic_offload_lf.py`:
```python
import os
import torch
from torch import nn
from asym_gemm.integrations.lf import _is_qwen3_decoder_layer_module_name
from asym_gemm.training.decoder_layer_glue_gc import install_decoder_layer_glue_gc
from asym_gemm.training.sdpa_recompute import install_sdpa_recompute

def _env_true(v): return str(v or "").strip().lower() in {"1","true","yes","on"}

def _install_save_on_cpu(module: nn.Module) -> None:        # off-layer: wrap a whole layer forward
    orig = module.forward
    def fwd(*a, _o=orig, **k):
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            return _o(*a, **k)
    module.forward = fwd  # type: ignore[method-assign]
```

## Stage 1 — `off-layer` policy (baseline #1, floor)
**1a. Parser** — `asym_gemm/training/moe.py`: add the spec branch next to `gc-layer` (:619), extend the
unsupported-policy error string at :737. **Do NOT touch `VALID_EXPERT_RECOMPUTE_POLICIES` (:431)** — that validates
the `.policy` field (`none/tok/gc`); off-layer's `policy` is `none`.
```python
if raw == "off-layer":
    return ExpertRecomputeConfig(policy="none", token_threshold=0, activation_save_policy="save_all",
        activation_save_threshold=0, label="off-layer", token_min=1, token_max=None,
        activation_save_min=1, activation_save_max=None, force_custom_autograd=False, torch_checkpoint=False)
```
**1b. Apply** — in `generic_offload_lf.py`:
```python
def apply_off_layer(model, report) -> int:
    n = 0
    for name, m in list(model.named_modules()):
        if name and _is_qwen3_decoder_layer_module_name(name, m):
            _install_save_on_cpu(m); n += 1
    if n == 0: raise RuntimeError("off-layer: no decoder layers matched")
    report.layer_off_wrapped = n; return n          # save_on_cpu per layer, no recompute
```

## Stage 2 — generic same-policy (baseline #2, apples-to-apples)
**2a. Parameterize the glue-GC wrapper** — `decoder_layer_glue_gc.py`: add an optional offload-kernel selector so
line :199 can use generic `save_on_cpu` instead of the custom pack/unpack:
```python
class DecoderLayerGlueGCWrapper:
    def __init__(self, module, *, generic_offload=False, ...):
        self.generic_offload = generic_offload
        ...
    def run(self, *args, **kwargs):
        ...
        if self.generic_offload:
            ctx = torch.autograd.graph.save_on_cpu(pin_memory=True)
        else:
            ctx = saved_tensors_hooks(self.saved_tensor_offload._pack, self.saved_tensor_offload._unpack)
        with ctx:
            return self._manual_forward(values)
```
`install_decoder_layer_glue_gc(module, generic_offload=True)` then gives the generic kernel — **also add the
`generic_offload` kwarg to `install_decoder_layer_glue_gc` (:239)** and forward it to the wrapper (current signature
is just `(module)`). Nothing else changes — same `_manual_forward`, same norm checkpointing.

**2b. Apply** — in `generic_offload_lf.py` (NO separate attn/expert wrappers — glue-GC's whole-forward
`saved_tensors_hooks` already offloads attn+expert+all saves; a second wrapper would double-offload):
```python
def apply_generic_same_policy(model, report) -> int:
    n = 0
    for name, m in list(model.named_modules()):
        if name and _is_qwen3_decoder_layer_module_name(name, m):
            install_decoder_layer_glue_gc(m, generic_offload=True); n += 1
    if n == 0: raise RuntimeError("generic same-policy: no decoder layers matched")
    install_sdpa_recompute(model)          # self-gates on ASYMM_ATTN_SDPA_RECOMPUTE
    report.generic_glue_gc_wrapped = n; return n
```
This offloads the layer's attn+exp activations generically + recomputes norms + sdpa — the **same policy** as asym,
implemented as one generic `save_on_cpu` vs asym's custom multi-mechanism offload (attn wrapper + MoE-block + glue-GC).
Fair head-to-head; confirm scope parity at impl (caveats).

**2c. No `context_fn` needed** — RESOLVED by the validation above: glue-GC checkpoints only the **norms**; attn/mlp
run normally so their (large) saves are offloaded by the hooks. The norm recompute in backward re-saves a
hidden-sized tensor on GPU (one norm at a time, transient — negligible vs the offloaded attn+exp); grads are
bit-identical. (Only if a *future* model checkpointed a *large* submodule would `context_fn` be needed — not the
case here.)

## Stage 3 — the hook (all backends)
End of `init_adapter` in `LlamaFactory/.../model/adapter.py`, just before the final `return model`:
```python
from asym_gemm.integrations.generic_offload_lf import maybe_apply_generic_offload
maybe_apply_generic_offload(model)     # env-gated; no-op unless requested
```
```python
def maybe_apply_generic_offload(model):
    # Reached only on non-asym backends (asym returns early in init_adapter), so any offload the
    # policy requests here must be applied generically — AUTO-DETECT from the policy + offload envs,
    # NO separate flag (matches configs that just set the policy, e.g. superoffload_mem|...).
    policy = os.environ.get("ASYM_GEMM_LF_CONFIG_EXPERT_POLICY", "none").strip()
    exp  = _env_true(os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_EXPERT_ACT_OFFLOAD"))
    attn = _env_true(os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_ACT_OFFLOAD"))
    gc   = _env_true(os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_GC"))
    report = _Report()
    if policy == "off-layer":
        apply_off_layer(model, report)
    elif exp or attn or gc:            # offload/gc requested on a non-asym backend → generic
        apply_generic_same_policy(model, report)
    else:
        return
    log_rank0(report)
```
(`_Report` = a tiny namespace/`SimpleNamespace` holding the counter fields; `log_rank0` = a rank-0 logger — both
trivial helpers to define in the new module.)

## Stage 4 — harness plumbing
1. `scripts/lf/profile_lora_lf_*.sh` (6 share the parser): add `off-layer` to **`normalize_expert_policy`** — the
   `none|gc-exp|gc-attn-exp|gc-layer|…)` case (~:644) **and** its error string (~:653); else the bash rejects it
   (`die "invalid expert policy"`). (Verified: this validator gates the policy field.)
2. **No new env flag.** The hook auto-detects from the already-emitted policy + `ASYMM_*` offload envs
   (`run_lf_lora_sft.sh:1985-2005`) — the policy's offload fields (`expert_act`/`attn_act`/`layer_gc`) drive
   `apply_generic_same_policy`; `off-layer` drives `apply_off_layer`. So just setting the policy is enough.

## Stage 5 — smoke
```bash
PY="$REPO/.venv/bin/python"
$PY -c "from asym_gemm.training.moe import parse_expert_recompute_policy_spec as p; assert p('off-layer').label=='off-layer'; print('ok')"
WORKLOADS="2048|1|1" PROFILERS=source GPU_POOL=3 MAX_STEPS=1 WARMUP_STEPS=1 MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
  BACKEND_SPECS="zero3_offload|norecomp|ligerloss1" ASYMM_EXP_ACT_POLICIES="off-layer|false|false|false|false" \
  OUTPUT_ROOT="$REPO/profiling_baseline_smoke" bash scripts/lf/profile_lora_lf_test.sh 2>&1 | tee /tmp/bl_smoke.log
```
Gate: exit 0, loss printed, `layer_off_wrapped>0`, peak HBM < `norecomp`. Repeat with
`ASYMM_EXP_ACT_POLICIES="none|true|true|false|true|true"` (auto-triggers generic offload — no flag) → gate `generic_glue_gc_wrapped>0`.

## Stage 6 — produce the final metrics (THE deliverable)
This is the goal of the whole effort: the comparison table. `profile_lora_lf_test.sh` → `show_metrics.sh`,
**strictly sequential** (NUMA/PCIe pollute peak HBM):
```bash
REPO=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM; cd $REPO
# Baselines #1 + #2 in ONE invocation — test.sh defaults are already set to this:
#   BACKEND_SPECS="superoffload_mem|norecomp|ligerloss1"
#   ASYMM_EXP_ACT_POLICIES="none|true|true|false|true|true,off-layer|false|false|false|false"
#   MODEL_SPECS="Qwen/Qwen3-30B-A3B|1,meta-llama/Llama-4-Scout-17B-16E|1"  WORKLOADS="4096|8|1"
#   GPU_POOL=3  PROFILERS=both   (no ASYMM_GENERIC_OFFLOAD — auto-triggered by the policy)
bash scripts/lf/profile_lora_lf_test.sh        # → superoffload_mem × {#2 none|T|T|F|T|T, #1 off-layer} × {qwen3, llama4}

# Contender (asym custom offload) — SEPARATE run, same policy/workload/models:
BACKEND_SPECS="asym_cpuadamwds|norecomp|ligerloss1" ASYMM_EXP_ACT_POLICIES="none|true|true|false|true|true" \
  bash scripts/lf/profile_lora_lf_test.sh
# (zero3_offload works identically to superoffload_mem as the baseline backend.)
bash scripts/lf/show_metrics.sh profiling_both
```
**Output — one table per model**, exact columns (what `show_metrics.sh` prints, `_s`=sec, `_H`=GiB, `RAM`=host GiB):
```
| Workload | Backend | Config | Status | fwd_s | bwd_s | opt_s | step_s | fwd_H | bwd_H | step_H | RAM |
# e.g. per model (Qwen3-30B-A3B, Llama-4-Scout), rows =
#   asym_cpuadamwds | none|T|T|F|T|T   (contender, custom offload)
#   zero3_offload   | none|T|T|F|T|T   (#2 generic same-policy)
#   zero3_offload   | off-layer        (#1 floor)
#   + superoffload_mem variants
```
**Success:** asym `none|T|T|F|T|T` ≤ #2 on **both** peak HBM (`step_H`) and step time (`step_s`) — custom kernel
beats generic at the same policy; #1 = low-`step_H` / high-`step_s` floor.

## Caveats
- `save_on_cpu` offloads *every* saved tensor incl. tiny ones → PCIe-heavy/slow; that's the floor — asym's
  selectivity/kernel should beat it.
- Run with LF GC **off** (`norecomp`); the only recompute is the policy's glue-GC + sdpa.
- Keep `ligerloss1` (MoE) so logits don't mask the activation comparison.
- Verify `save_on_cpu` composes with ZeRO-3 param offload (forward ctx; smoke it — Stage 5).
- The hook is an LF-submodule edit (like the existing asym hook); keep it env-gated + no-op by default.
- **Scope parity (must verify):** asym `none|T|T|F|T|T` offloads via 3 custom mechanisms (attn wrapper `:2252` +
  MoE-block `:1851` + glue-GC `:2306`); #2 uses one generic whole-forward `save_on_cpu`. Log offloaded-bytes /
  compare peak-HBM scope so #2 and asym cover the **same** attn+exp activations — else the head-to-head is biased.
- **sdpa env on non-asym path (must verify):** `install_sdpa_recompute` self-gates on the env it reads (verify which
  — likely `ASYMM_ATTN_SDPA_RECOMPUTE`). Confirm the harness exports *that* env (not only the `ASYM_GEMM_LF_CONFIG_*`
  form) to the LF process for zero3/superoffload, else #2's sdpa-recompute silently no-ops.

## Scope summary
| Area | Change |
| --- | --- |
| `asym_gemm/integrations/generic_offload_lf.py` (NEW) | `_install_save_on_cpu`, `apply_off_layer`, `apply_generic_same_policy`, `maybe_apply_generic_offload` |
| `asym_gemm/training/moe.py` | +`off-layer` parser branch + error string (NOT `VALID_EXPERT_RECOMPUTE_POLICIES`) |
| `asym_gemm/training/decoder_layer_glue_gc.py` | +`generic_offload` flag → swap `saved_tensors_hooks` to `save_on_cpu` |
| `LlamaFactory/.../model/adapter.py` | +1 env-gated `maybe_apply_generic_offload(model)` at end of `init_adapter` |
| harness (`profile_lora_lf_*.sh`) | accept `off-layer` in `normalize_expert_policy` (no new env flag — auto-trigger) |
| untouched | `apply_lf_asym_lora` (asym path), custom offload kernels, KT backends |
