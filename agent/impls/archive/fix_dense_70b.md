# AsymGEMM dense 70B — Qwen2.5-72B-Instruct & Llama-3.3-70B-Instruct

> Extends [`fix_dense.md`](fix_dense.md) (v1, the expert-wrap fix) and [`fix_dense_v2..md`](fix_dense_v2..md) (the surgical-`exp` finding) to two more **dense** LLMs. v1's `lf.py:1945` fix is a prerequisite. Net new code here is **one generic dense decoder-matcher branch**; everything else (dense LoRA, CPU base-weight offload, attn-act, sdpa, `asym_cpuadamwds`, profiling, the dense MLP being offloaded by `layer_gc`) is already generic.

## Objective

Run `asym_cpuadamwds|norecomp` with `ASYMM_EXP_ACT_POLICIES="none|true|true|false|true|true,gc-layer|false|false|false|false"` on:
- `meta-llama/Llama-3.3-70B-Instruct` — bias-free attention → **architecturally easiest, do first**
- `Qwen/Qwen2.5-72B-Instruct` — q/k/v have bias (Qwen2 hardcodes it; handled by `lora.py:174` + `attention_activation_offload.py:609-610`)

**Success criterion:** `none|true|true|false|true|true` (asym, MLP offloaded by `layer_gc`) peak HBM **<** `asym_cpuadamwds|recomp` **and** `<` `zero3_offload|recomp`. Metric = `peak_allocated_hbm_bytes`.

### ✅ RESULT — Llama-3.3-70B (2026-06-22, seq4096×b4, 15 steps, source, sequential)
| config | peak HBM |
| --- | ---: |
| **asym OFFLOAD `none\|T\|T\|F\|T\|T`** | **24.56 GiB** |
| asym RECOMPUTE | 44.56 GiB |
| zero3_offload | 46.40 GiB |

**PASS — asym offload lowest: −44.9% vs asym-recompute, −47.1% vs zero3_offload.** All runs exit 0, losses sane (2.18→1.88 in smoke). Enabled by the one generic decoder-matcher branch; dense MLP offloaded by `layer_gc`.

### ✅ RESULT — Qwen2.5-72B-Instruct (q/k/v bias; same drill, `TEMPLATE=qwen`)
| config | peak HBM |
| --- | ---: |
| **asym OFFLOAD `none\|T\|T\|F\|T\|T`** | **28.92 GiB** |
| asym RECOMPUTE | 48.92 GiB |
| zero3_offload | 51.13 GiB |

**PASS — asym offload lowest: −40.9% vs asym-recompute, −43.4% vs zero3_offload.** Q2.5's q/k/v bias handled cleanly (smoke trained loss 2.33→1.67, `layer_glue_gc=80`, `dense_lora_wrapped=560`). No per-model code beyond the shared decoder-matcher branch + `TEMPLATE=qwen`.

Both are 80-layer, hidden 8192, 64/8-kv-head models. Llama-3.3: vocab 128256, `attention_bias=False`. Qwen2.5-72B: vocab 152064, q/k/v bias.

## Why a fix is needed

Both decoder layers are `{self_attn, mlp, input_layernorm, post_attention_layernorm}` — structurally identical to Qwen3 — but the classes are `LlamaDecoderLayer` / `Qwen2DecoderLayer`, **not** `qwen3`. So `_is_qwen3_decoder_layer_module_name` (`lf.py:1393`) didn't match them, and `gc-layer` / `ASYMM_LAYER_GC` would strict-error ("no supported decoder layers"). Everything else already works for any dense arch with `q/k/v/o_proj` + `gate/up/down_proj`.

## Stage 1 — Code change (one generic dense decoder-matcher branch)

**File:** `asym_gemm/integrations/lf.py`, in `_is_qwen3_decoder_layer_module_name`, after the qwen3 branch and before the llama4 branch. Matches any standard dense decoder layer; MoE layers (mlp = sparse block, **no** gate/up/down children) are excluded so routed-expert layers are never reclassified.

```python
# Generic DENSE decoder layer (Qwen2 / Qwen2.5 / Llama-3.x / Mistral / etc.)
if qwen3_required <= child_names:           # {self_attn, mlp, input_layernorm, post_attention_layernorm}
    mlp_child = children["mlp"]
    if getattr(mlp_child, "_is_asym_dense_mlp", False):
        return True
    if {"gate_proj", "up_proj", "down_proj"} <= set(dict(mlp_child.named_children())):
        return True
```

**The dense MLP is offloaded by `layer_gc`, not `exp`.** Per the v2 finding (`fix_dense_v2..md`), the surgical `exp`=dense-MLP path is numerically correct but **stalls at large seq/batch** — a dense MLP runs the *full* compute on CPU every layer (unlike sparse MoE experts). So it is **gated opt-in** (`ASYMM_DENSE_MLP_SURGICAL_OFFLOAD`, default off). With the standard config, `exp=true` is a no-op for the MLP and `ASYMM_LAYER_GC` offloads the MLP activation (the HBM win) while keeping the backward on GPU (fast). This is what reaches the goal — confirmed on Qwen3-32B (v1 = 28.55 GiB).

### Verified (matcher)
`_is_qwen3_decoder_layer_module_name` returns **True** for toy `LlamaDecoderLayer`, `Qwen2DecoderLayer`, `Qwen3DecoderLayer`, and **False** for a MoE block with no gate/up/down children. No MoE regression.

## Stage 2 — Unit validation

```bash
PY="$REPO/.venv/bin/python"
# matcher recognizes the new dense layers, excludes MoE
$PY - <<'P'
# (build toy Llama/Qwen2/Qwen3 decoder layers + a fake MoE; assert matcher True/True/True/False)
P
# MoE non-regression unchanged
$PY -m pytest -q tests/training/test_lf_qwen3_asym_backend.py tests/training/test_lf_qwen35_asym_backend.py \
  -k "wraps_experts or whole_wraps_qwen3_moe or whole_wraps_llama4 or gc_layer"
```
Gate: matcher True for Llama/Qwen2, False for MoE; MoE suites green (the new branch never reclassifies a routed-expert layer).

## Stage 3 — Smoke on the real model (1 step)

```bash
# Llama-3.3 needs TEMPLATE=llama3 (auto-infer would default to qwen3_nothink); Qwen2.5 -> qwen2/qwen
MODEL_SPECS="meta-llama/Llama-3.3-70B-Instruct|1" TEMPLATE=llama3 GPU_POOL=3 CHECK_TRAINABLE_SURFACE=0 \
  MAX_STEPS=1 WARMUP_STEPS=1 WORKLOADS="2048|1|1" PROFILERS=source \
  ASYMM_EXP_ACT_POLICIES="none|true|true|false|true|true" \
  OUTPUT_ROOT="$REPO/profiling_llama33_smoke" \
  bash scripts/lf/profile_lora_lf_test2.sh 2>&1 | tee /tmp/llama33_smoke.log
```
Gate: a step trains (loss printed); report shows `layer_glue_gc_wrapped=80`, `attention_act_offload_wrapped=320`, `dense_lora_wrapped=560` (80×(4 attn + 3 mlp); MLP linears wrapped as dense LoRA since surgical `exp` is off), `dense_mlp_act_offload_wrapped=0`, `cpu_resident_base_bytes≈140 GB`, no "no supported decoder layers" / expert error.

## Stage 4 — Comparison (sequential, one GPU, `term` not `-9`)

Three configs into one `OUTPUT_ROOT`, `CHECK_TRAINABLE_SURFACE=0`, `PROFILERS=source`, default workload `4096|4|1`:

```bash
export CMP=$REPO/profiling_70b ; export M="meta-llama/Llama-3.3-70B-Instruct" ; export T=llama3
# 1) asym offload (the contender)
MODEL_SPECS="$M|1" TEMPLATE=$T GPU_POOL=3 OVERWRITE=true CHECK_TRAINABLE_SURFACE=0 PROFILERS=source \
  BACKEND_SPECS="asym_cpuadamwds|norecomp" ASYMM_EXP_ACT_POLICIES="none|true|true|false|true|true" \
  OUTPUT_ROOT="$CMP" bash scripts/lf/profile_lora_lf_test2.sh
# 2) asym recompute (baseline)
MODEL_SPECS="$M|1" TEMPLATE=$T GPU_POOL=3 OVERWRITE=true CHECK_TRAINABLE_SURFACE=0 PROFILERS=source \
  BACKEND_SPECS="asym_cpuadamwds|recomp" ASYMM_EXP_ACT_POLICIES="none|false|false|false|false|false" \
  OUTPUT_ROOT="$CMP" bash scripts/lf/profile_lora_lf_test2.sh
# 3) zero3_offload recompute (baseline)
MODEL_SPECS="$M|1" TEMPLATE=$T GPU_POOL=3 OVERWRITE=true CHECK_TRAINABLE_SURFACE=0 PROFILERS=source \
  BACKEND_SPECS="zero3_offload|recomp" OUTPUT_ROOT="$CMP" bash scripts/lf/profile_lora_lf_test1.sh
```
Run strictly one at a time (CPU-pin/NUMA contention pollutes peak-HBM). Then repeat with `M="Qwen/Qwen2.5-72B-Instruct" T=qwen` (Qwen2.5 template).

## Stage 5 — Validate the goal

```bash
CFG="$CMP/asym_long_sft_smoke__lora__lf__bf16/<model>__gpus1__b4_s4096_ga1_w5_s10_r64_a16_drop000"
$PY - "$CFG" <<'PY'
import json,glob,sys; cfg=sys.argv[1]; g=1024**3; r={}
for f in glob.glob(cfg+"/**/memory_breakdown_summary.json",recursive=True):
    t=f.split('/')[-3]; d=json.load(open(f)); peak=d['peak_allocated_hbm_bytes']/g
    if 'asym_cpuadamwds' in t and 'norecomp' in t and 'layergc1' in t: r['asym_offload']=peak
    elif 'asym_cpuadamwds' in t and '__recomp__' in t: r['asym_recompute']=peak
    elif 'zero3_offload' in t and 'zero3_offload_mem' not in t: r['zero3_offload']=peak
for k,v in r.items(): print(f"  {k:16s} = {v:.2f} GiB")
a=r.get('asym_offload')
if a: print("PASS:", all(a<r[k] for k in ('asym_recompute','zero3_offload') if k in r))
PY
```
**Pass:** `asym_offload` < `asym_recompute` and < `zero3_offload`.

## Risks / operational notes
- **70B scale:** base ≈140 GB → CPU; activations at s4096×b4 push host RAM high (watch `free -g`; the week5/6 runs OOM'd host RAM at s8192×b8). Run sequentially.
- **Template:** Llama-3.3 → `llama3`; Qwen2.5 → `qwen`/`qwen2`. Auto-infer defaults to `qwen3_nothink` (wrong for both) — pass `TEMPLATE=` explicitly. Wrong template = suboptimal formatting, not a crash, but set it.
- **Qwen2.5 q/k/v bias:** handled by the standard LoRA path and the attn-offload path; verify the smoke report has no bias-related skip.
- **`CHECK_TRAINABLE_SURFACE=0`** and **`term` (never `kill -9`)** as in v1.
- **Do NOT enable `ASYMM_DENSE_MLP_SURGICAL_OFFLOAD`** at s4096×b4 — it stalls (v2 finding). The dense MLP is correctly offloaded by `layer_gc`.

## Scope summary
| Area | Change |
| --- | --- |
| `lf.py` `_is_qwen3_decoder_layer_module_name` | +6 lines: generic dense decoder branch (Qwen2/Llama/…); MoE excluded |
| dense LoRA / CPU offload / attn-act / sdpa / optimizer / profiling | unchanged (generic) |
| dense MLP offload | via `layer_gc` (surgical `exp` gated opt-in per v2) |
| MoE (Qwen3-MoE / Qwen3.5 / Llama4) | unaffected |
