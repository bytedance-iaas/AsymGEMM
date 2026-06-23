# Enable AsymGEMM dense (Qwen3-32B) + memory-baseline comparison

## Objective

Run the AsymGEMM backend on the **dense** `Qwen/Qwen3-32B` with the policy sweep

```
ASYMM_EXP_ACT_POLICIES="none|true|true|false|true|true,gc-layer|false|false|false|false"
BACKEND_SPECS="asym_cpuadamwds|norecomp"
```

then run the two comparison harnesses unchanged:

- `scripts/lf/profile_lora_lf_test1.sh` — baselines: `zero3_offload|recomp`, `zero3_offload_mem|recomp` (inert policy `none|false|false|false|false|false`)
- `scripts/lf/profile_lora_lf_test2.sh` — AsymGEMM: `asym_cpuadamwds|norecomp` with the two policies above

**Success criterion (the whole point):**
> `none|true|true|false|true|true` (test2) peak GPU memory **<** `zero3_offload_mem|recomp` (test1)

Metric = `peak_allocated_hbm_bytes` in each run's `memory_breakdown_summary.json`.

### ✅ RESULT (2026-06-22, Qwen3-32B, seq4096×b4, 15 steps, source profiler, sequential on one GPU)

| metric (GiB) | asym `none\|T\|T\|F\|T\|T` | `zero3_offload_mem\|recomp` |
| --- | ---: | ---: |
| **PEAK allocated HBM** | **28.55** | **38.37** |
| activation @ peak | 10.05 | 21.35 |
| saved (offloaded) act | 9.90 | 21.19 |
| temp workspace | 18.49 | 17.02 |

**PASS — asym 25.6% lower (28.55 < 38.37).** The win is entirely the activation footprint (10.05 vs 21.35 GiB resident at peak); temp workspace and live act are ~equal, so it cleanly isolates the offload advantage. Liger not needed. asym losses 1.65–1.76, zero3 1.69–1.80 (sane).

### Operational rules learned the hard way
- **Stop runs with `term` (= `kill -TERM`, `/workspace/env/bashrc.sh:458`), NEVER `kill -9`.** A `-9` of a DeepSpeed run mid-JIT-compile corrupts the `cpu_adam` build (`/scratch_local/.../torch_extensions/py312_cu130/cpu_adam/` — leaves `.ninja_lock`/`lock`, `.o` newer than `.so`); every later DeepSpeed run (asym AND zero3 both use DeepSpeed CPU-Adam) then hangs in `futex_wait`/`hrtimer_nanosleep` at optimizer init (0% GPU). Recovery: `rm -rf` that op dir to force a clean recompile.
- **Run configs strictly sequentially on one GPU** (no parallel across GPUs) — concurrent CPU-pin/NUMA/bandwidth contention pollutes the peak-HBM numbers.
- **`jq` is not installed** — read JSON with the venv Python.
- When scripting kills, target **exact PIDs**, not `pgrep -f <pattern>` — the pattern matches the killer shell's own command string and it suicides (exit 144).

**Hard constraint — do NOT break existing models.** Qwen3-30B-A3B MoE, Qwen3.5-MoE, and Llama4-MoE must wrap experts exactly as they do today. The change in Stage 1 is an **isolated dense-only branch** gated on "this model has zero packed-expert/MoE blocks." Every MoE model produces expert candidates and therefore takes the byte-for-byte identical original code path; only a genuinely dense model (Qwen3-32B) takes the new branch.

`norecomp` is correct here: it keeps HF full gradient-recompute **off** so the AsymGEMM selective policies (`layergc1` + `sdparecomp1` + activation offload) are the thing being measured. With `recomp` the policy axis is collapsed to inert (`canonicalize_policy_axis_for_inert_run`).

---

## Why a fix is needed at all (root cause)

Everything the config exercises is already generic and works for dense Qwen3-32B **except one strict guard**. Verified by reading + empirical probes:

- `Qwen3DecoderLayer.forward` (dense) is structurally identical to `Qwen3MoeDecoderLayer.forward` in transformers 5.6.0 (both end `hidden_states = self.mlp(...); return hidden_states`). The decoder matcher `_is_qwen3_decoder_layer_module_name` (`lf.py:1390`) already matches it via `"qwen3" in class_name` (`lf.py:1417`), so **gc-layer** and **ASYMM_LAYER_GC** wrap it with no new code.
- `attn_act`, `sdpa_recompute`, dense LoRA, CPU base-weight offload, the `asym_cpuadamwds` optimizer, weight-offload, and profiling are all leaf-name/structure driven and have no MoE assumptions.

**The single blocker:** the script hardcodes `--lora_target all` → `_targets_experts("all")` is True (`lf.py:1051`) → `apply_lf_asym_lora` enters the expert-wrap block, finds **zero** MoE/expert modules in a dense model, and raises at **`lf.py:1945-1946`**:

```
ValueError: AsymGEMM requested routed expert LoRA but found no supported packed expert/MoE modules.
```

This was reproduced directly on a toy dense model (strict=True). It fires for *both* policies (it precedes the decoder matcher) and is independent of `expert_act`.

---

## Stage 0 — Preconditions

```bash
export REPO=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
cd "$REPO"                          # scripts derive SFT_ROOT via `cd ../..`; must run from here
PY="$REPO/.venv/bin/python"

# environment sanity
$PY -c "import torch,transformers;print('torch',torch.__version__,'tf',transformers.__version__)"
$PY -c "import torch;print('cuda',torch.cuda.is_available(),'sm',torch.cuda.get_device_capability())"   # expect sm_100 (REQUIRE_SM100=1)
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv
free -g | awk 'NR==1||/Mem/'        # need headroom: 32B bf16 base ≈ 65 GB offloaded to CPU + optimizer state
```

Notes:
- Pick a free GPU with `GPU_POOL=<id>` (default `3`). `asym_cpuadamwds` uses 1 GPU.
- `Qwen/Qwen3-32B` must be present in the HF cache or downloadable.
- Run the two heavy-offload scripts **sequentially**, not concurrently (CPU RAM).

Gate: all three checks succeed; ≥ ~150 GB free CPU RAM; SM100 GPU visible.

---

## Stage 1 — Code change (isolated dense branch; existing models untouched)

**This change must not alter behavior for any MoE model.** It is written *for the dense Qwen3-32B case only* and is reached only when the model has **zero** packed-expert/MoE blocks.

**File:** `asym_gemm/integrations/lf.py`, function `apply_lf_asym_lora`. The `if wrap_experts:` block opens at **line 1799**; `expert_candidates` is built at **lines 1800-1814** by the MoE detectors `is_qwen35_moe_block` / `is_qwen3_moe_block` / `is_llama4_moe` / `is_qwen3_experts`; the wrapping loop is **lines 1816-1943**; the strict guard to change is **lines 1945-1946** (8-space indent, still inside `if wrap_experts:`, so `expert_candidates` is in scope). **Leave the candidate scan and the wrapping loop exactly as-is** — only the final guard changes. Exact replacement (match the current two lines verbatim):

```python
# BEFORE  (raises for ANY model with no experts found — wrongly trips dense Qwen3-32B)
        if not expert_prefixes and strict:
            raise ValueError("AsymGEMM requested routed expert LoRA but found no supported packed expert/MoE modules.")

# AFTER  (explicit dense-only branch; MoE path is byte-for-byte unchanged)
        # Dense models (e.g. Qwen/Qwen3-32B) have no packed-expert/MoE blocks, but
        # `--lora_target all` still sets wrap_experts. When the MoE detectors find ZERO
        # candidates the model is genuinely dense: record it and skip expert wrapping.
        # MoE models (Qwen3-MoE / Qwen3.5 / Llama4) ALWAYS produce candidates, so they
        # fall through to the original strict guard below and are unaffected.
        dense_no_experts = not expert_candidates
        if dense_no_experts:
            report.skipped.append("routed_experts:dense_model_no_experts")
        elif not expert_prefixes and strict:
            raise ValueError("AsymGEMM requested routed expert LoRA but found no supported packed expert/MoE modules.")
```

**Why this provably cannot regress existing models:**
- MoE model → `expert_candidates` non-empty → `dense_no_experts = False` → the `elif` *is* the original guard, reached on the original path; the wrapping loop above is untouched ⇒ identical behavior, identical reports.
- The "experts present but none wrapped" safety net is preserved (non-empty candidates + empty prefixes + strict → still raises).
- The *only* new behavior is `expert_candidates == []` (no MoE blocks anywhere) → dense → skip instead of raise, with an explicit `report.skipped` breadcrumb.
- Downstream, `expert_act=true` becomes a true no-op for dense (it is only ever consumed inside an `AsymQwen3Experts` wrapper, which is never created when there are no experts).

No existing test asserts that error string (`grep tests/` is clean); the MoE-wrapping tests assert the positive path stays intact (Stage 2c). Equivalent one-liner if you prefer minimal diff: `if expert_candidates and not expert_prefixes and strict:` — same semantics, but the explicit `dense_no_experts` branch above is preferred for readability and the breadcrumb.

### Verification evidence (why this is correct, not assumed)

Every claim below was checked against the running code (transformers 5.6.0 in `.venv`) and direct probes, not inferred:

- **The blocker is real and is the *only* blocker.** A toy dense model (`model_type="qwen3"`, std attention+MLP) through `apply_lf_asym_lora(raw_lora_target="all", strict=True)` raised exactly `ValueError: ...no supported packed expert/MoE modules` — and nothing earlier. With `wrap_experts` sidestepped, the *same* model passed strict end-to-end: report `dense_lora_wrapped=14, attention_act_offload_wrapped=8, attention_saved_tensor_offload_wrapped=2, layer_glue_gc_wrapped=2, trainable_lora_params=32768, skipped=[]`. So once the guard is relaxed, the full config wraps cleanly with zero skips.
- **Decoder GC works for dense.** `_is_qwen3_decoder_layer_module_name` returns True for a `Qwen3DecoderLayer` (matches via `"qwen3" in class_name`, `lf.py:1417`) and False for `LlamaDecoderLayer`; the empirical `layer_glue_gc_wrapped=2` confirms both decoder layers were wrapped by ASYMM_LAYER_GC. `gc-layer` (policy 2) uses `install_decoder_checkpoint` (generic `torch.utils.checkpoint`), also fine.
- **The glue-GC forward reproduces dense Qwen3 exactly.** `decoder_layer_glue_gc.py:_manual_forward` takes the `else: mlp_out = layer.mlp(normed)` branch (Qwen3 has `mlp`, not `feed_forward`), guards the tuple case, and skips the Llama4-only `.view` reshape. In transformers 5.6.0, `Qwen3DecoderLayer.forward` and the already-validated `Qwen3MoeDecoderLayer.forward` are structurally identical (both end `hidden_states = self.mlp(hidden_states); hidden_states = residual + hidden_states; return hidden_states`), and `_layer_family()` returns `"qwen3"` for the dense layer → correct `position_ids`/`mlp` handling.
- **attn_act / sdpa / dense-LoRA / CPU-offload / optimizer / profiling are generic.** Confirmed by reading: `saved_tensors_hooks`-based attention offload, `ALL_ATTENTION_FUNCTIONS`-based sdpa recompute, leaf-name `classify_lf_component` (with `mlp_dense` first-class in `split_asym_peft_dense_targets`, `adapter.py:237`), and `cpu_adam.py`/`weight_offload.py`/profiling all tolerate empty expert/router buckets.
- **MoE non-regression is structural.** For any MoE model `expert_candidates` is non-empty ⇒ `dense_no_experts=False` ⇒ the unchanged `elif` original guard runs on the unchanged loop. `report.skipped` is `list[str]` (`lf.py:299`) consumed only via `"; ".join(...)` in `to_log_string` (`lf.py:307`); appending `"routed_experts:dense_model_no_experts"` follows the existing `"name:reason"` convention used at 13 other sites (e.g. `lf.py:2112,2160`).

---

## Stage 2 — Unit validation of the fix

### 2a. The previously-failing path now passes (dense + target=all + strict)

```bash
cat > /tmp/val_dense_fix.py <<'PY'
import torch
from torch import nn
import asym_gemm.integrations.lf as lf
def lin(o,i): return nn.Linear(i,o,bias=False)
class Cfg:
    model_type="qwen3"; tie_word_embeddings=False
    def __init__(s): s.use_cache=False; s.output_router_logits=False
def mk(c,m,**k):
    def __init__(s):
        nn.Module.__init__(s)
        for a,v in k.items(): setattr(s,a,v() if callable(v) else v)
    return type(c,(nn.Module,),{"__init__":__init__,"__module__":m})
H=128
A=mk("Qwen3Attention","transformers.models.qwen3.modeling_qwen3",
    q_proj=lambda:lin(128,H),k_proj=lambda:lin(64,H),v_proj=lambda:lin(64,H),o_proj=lambda:lin(H,128),
    q_norm=lambda:nn.RMSNorm(64,eps=1e-6),k_norm=lambda:nn.RMSNorm(64,eps=1e-6))
M=mk("Qwen3MLP","transformers.models.qwen3.modeling_qwen3",
    gate_proj=lambda:lin(256,H),up_proj=lambda:lin(256,H),down_proj=lambda:lin(H,256))
DL=mk("Qwen3DecoderLayer","transformers.models.qwen3.modeling_qwen3",
    self_attn=A,mlp=M,input_layernorm=lambda:nn.RMSNorm(H,eps=1e-6),
    post_attention_layernorm=lambda:nn.RMSNorm(H,eps=1e-6),config=Cfg)
class Inner(nn.Module):
    def __init__(s):
        super().__init__(); s.embed_tokens=nn.Embedding(256,H)
        s.layers=nn.ModuleList([DL() for _ in range(2)]); s.norm=nn.RMSNorm(H,eps=1e-6)
class Top(nn.Module):
    def __init__(s):
        super().__init__(); s.model=Inner(); s.lm_head=lin(256,H); s.config=Cfg()
    def get_input_embeddings(s): return s.model.embed_tokens
    def get_output_embeddings(s): return s.lm_head
m=Top().to(torch.bfloat16)
tgt=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
# raw_lora_target="all" sets wrap_experts (the blocker); dense_target_modules is the concrete
# PEFT-style list so dense LoRA actually attaches (mirrors the real LF flow). Using "all" here
# would NOT expand via _matches_target and would trip the zero-trainable-LoRA guard at lf.py:2250.
model,report=lf.apply_lf_asym_lora(m,raw_lora_target="all",dense_target_modules=tgt,
    lora_rank=8,lora_alpha=16,lora_dropout=0.0,backend="asym",precision="bf16",
    offload_modules="all",strict=True)
print("OK strict; dense_lora_wrapped =", report.dense_lora_wrapped, "trainable =", report.trainable_lora_params)
PY
$PY /tmp/val_dense_fix.py        # PRE-fix: ValueError "...no supported packed expert/MoE modules"
                                 # POST-fix: "OK strict; dense_lora_wrapped = 14 trainable = ..."
```

### 2b. Full policy wraps end-to-end (attn_act + layer_gc + sdpa) on dense Qwen3

```bash
ASYMM_ATTN_ACT_OFFLOAD=true ASYMM_LAYER_GC=true ASYMM_ATTN_SDPA_RECOMPUTE=true \
ASYMM_EXPERT_ACT_OFFLOAD=true $PY - <<'PY'
import os,torch; from torch import nn
import asym_gemm.integrations.lf as lf
exec(open("/tmp/val_dense_fix.py").read().split("m=Top()")[0])   # reuse builders
m=Top().to(torch.bfloat16)
tgt=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
# raw_lora_target="attention" dodges the expert-target trigger; dense_target_modules drives LoRA
model,r=lf.apply_lf_asym_lora(m,raw_lora_target="attention",dense_target_modules=tgt,
    lora_rank=8,lora_alpha=16,lora_dropout=0.0,backend="asym",precision="bf16",
    offload_modules="all",strict=True)
assert r.layer_glue_gc_wrapped==2 and r.attention_act_offload_wrapped==8 and r.dense_lora_wrapped==14, vars(r)
print("OK: layer_glue_gc=%d attn_act=%d dense_lora=%d trainable=%d skipped=%d"%(
    r.layer_glue_gc_wrapped,r.attention_act_offload_wrapped,r.dense_lora_wrapped,
    r.trainable_lora_params,len(r.skipped)))
PY
```

Expected: `layer_glue_gc=2 attn_act=8 dense_lora=14 ... skipped=0`.

### 2c. Non-regression for existing MoE models (the "don't break existing models" gate)

```bash
$PY -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py \
  tests/training/test_decoder_layer_glue_gc.py \
  tests/training/test_lf_qwen35_asym_backend.py
```

These suites construct real Qwen3-MoE / Qwen3.5-MoE / Llama4 fixtures and assert experts are still wrapped and routers frozen — they are the explicit guarantee the dense branch did not touch the MoE path. Key cases that must stay green:
- `test_apply_lf_asym_lora_whole_wraps_qwen3_moe_and_freezes_router`
- `test_apply_lf_asym_lora_whole_wraps_llama4_moe_and_reports_router_mode`
- `test_apply_lf_asym_lora_whole_wraps_qwen35_and_dense_shared_modules`
- `test_apply_lf_asym_lora_gc_layer_wraps_qwen3_decoder_layers_only` / `..._wraps_llama4_decoder_layers`

### 2d. Targeted assertion: MoE wrap is unchanged AND dense skips cleanly

```bash
$PY -m pytest -q tests/training/test_lf_qwen3_asym_backend.py \
  -k "wraps_experts or whole_wraps_qwen3_moe or whole_wraps_llama4"
```

Gate: 2a prints OK (and *raises* before the fix); 2b asserts pass with `qwen3_moes_wrapped == 0` for the dense model; 2c + 2d green ⇒ MoE behavior provably unchanged.

### 2e. (Optional, strongest) Live non-regression on the original MoE model

Confirm the default Qwen3-30B-A3B MoE still runs end-to-end after the edit (its expert wrapping is the path most at risk if the edit were wrong):

```bash
GPU_POOL=3 MAX_STEPS=1 WARMUP_STEPS=1 WORKLOADS="2048|1|1" \
  OUTPUT_ROOT="$REPO/profiling_regress_moe" \
  bash scripts/lf/profile_lora_lf_test2.sh 2>&1 | tee /tmp/moe_regress.log   # MODEL_SPECS default = Qwen/Qwen3-30B-A3B
grep -E "qwen3_moes_wrapped|packed_experts_wrapped" /tmp/moe_regress.log     # expect non-zero (experts still wrapped)
```

---

## Stage 3 — Smoke on the real Qwen3-32B (no full sweep)

### 3a. Dry run — confirm command + config-dir wiring, no training

```bash
MODEL_SPECS="Qwen/Qwen3-32B|1" DRY_RUN=true \
  bash scripts/lf/profile_lora_lf_test2.sh 2>&1 | tee /tmp/qwen3_32b_dry.log
grep -E "model_name_or_path|asym_offload_modules|lora_target|polnone|expact1|layergc1" /tmp/qwen3_32b_dry.log | head
```

Confirm: model = `Qwen/Qwen3-32B`, template auto-resolves to `qwen3_nothink`, `--lora_target all`, `--asym_offload_modules all`, and the per-run dir tag carries `...norecomp__polnone__...expact1__attnact1__layeract0__layergc1__sdparecomp1...`.

### 3b. One-step real train — confirm forward/backward survives the policy

```bash
MODEL_SPECS="Qwen/Qwen3-32B|1" GPU_POOL=3 CHECK_TRAINABLE_SURFACE=0 \
  MAX_STEPS=1 WARMUP_STEPS=1 WORKLOADS="2048|1|1" \
  OUTPUT_ROOT="$REPO/profiling_smoke_qwen3_32b" \
  bash scripts/lf/profile_lora_lf_test2.sh 2>&1 | tee /tmp/qwen3_32b_smoke.log
```

Gate: a step completes (loss printed), the run writes `profile.json`, and `lf_run` reports the AsymGEMM report (`dense_lora_wrapped`, `layer_glue_gc_wrapped`, `attention_act_offload_wrapped` non-zero). No expert ValueError.

---

## Stage 4 — Full comparison runs (the exact configs)

Run both into the **same** `OUTPUT_ROOT` so they land in one config dir (same dataset/model/workload/rank) and the combined plot picks up all backends. Default `OUTPUT_ROOT` for `PROFILERS=both` (the default) is `$REPO/profiling_both`; pin it explicitly for clarity.

**REQUIRED env override — `CHECK_TRAINABLE_SURFACE=0` (second dense gotcha).** A post-run guard (`run_lf_lora_sft.sh:83`, gate at `:1421`, raise at `:1456`) aborts the sweep with *"Qwen3 lora_target=all profile has no captured expert LoRA parameters … Set CHECK_TRAINABLE_SURFACE=0"* — it asserts **expert** LoRA params exist, which a dense model never has. Without this, test1 aborts after `zero3_offload` (so `zero3_offload_mem` never runs) and test2 aborts after policy 1. Disabling it is the prescribed setting and is correct for dense (no expert surface to verify). Env override only; no script edit.

GPU 0 and GPU 3 are both free here, so run the two scripts **concurrently** on separate GPUs (≈2× faster). The dataset `asym_long_sft_smoke__qwen3-32b__s4096` is built once by either run and audited read-only by the other.

```bash
export CMP=$REPO/profiling_both         # shared output root for the comparison

# 4a. Baselines (zero3_offload, zero3_offload_mem) on GPU 3
MODEL_SPECS="Qwen/Qwen3-32B|1" GPU_POOL=3 CHECK_TRAINABLE_SURFACE=0 OVERWRITE=true OUTPUT_ROOT="$CMP" \
  bash scripts/lf/profile_lora_lf_test1.sh > /tmp/qwen3_32b_test1.log 2>&1 &

# 4b. AsymGEMM (asym_cpuadamwds|norecomp), policies none|T|T|F|T|T + gc-layer, on GPU 0
MODEL_SPECS="Qwen/Qwen3-32B|1" GPU_POOL=0 CHECK_TRAINABLE_SURFACE=0 OVERWRITE=true OUTPUT_ROOT="$CMP" \
  bash scripts/lf/profile_lora_lf_test2.sh > /tmp/qwen3_32b_test2.log 2>&1 &
wait    # both finish in ~12 min; drop the trailing `&`/`wait` to run sequentially on one GPU
```

Default workload is `4096|4|1`, `MAX_STEPS=10`, `WARMUP_STEPS=5`, `r64/a16/drop000`. The exact config dir (empirically verified by running the script's `safe_label`/`job_root_path` logic — note the double underscore after `gpus1`, and hyphens are preserved):

```
$CMP/asym_long_sft_smoke__lora__lf__bf16/qwen3-32b__gpus1__b4_s4096_ga1_w5_s10_r64_a16_drop000/
```

Per-run dir name = `<backend>__<profiler>__<recompute>__pol<expert_policy>__router<mode>__expact?__attnact?__layeract?__layergc?__sdparecomp?__loraafwdhbm__actrecomp0__xunpack0__<ligerloss>[__gradoff?__weightoff?]` (built at `profile_lora_lf_test*.sh:1326,1330`; the `__gradoff__weightoff` suffix is appended only for `asym_cpuadamw*` backends). **Under `PROFILERS=both` each job emits two sibling dirs** — `<backend>__nsys__...` (Nsight run) and `<backend>__source__...` (materialized source artifacts). The `memory_breakdown_summary.json` is written into the **`__source__`** dir (`run_lf_profiled_train.py` writes it; `materialize_source_artifacts_from_nsys` copies it). Exact dirs for this comparison (the `_source` variant carries the breakdown):

```
# asym policy 1  (none|true|true|false|true|true)  -- the contender
asym_cpuadamwds__source__norecomp__polnone__routerwhole__expact1__attnact1__layeract0__layergc1__sdparecomp1__loraafwdhbm__actrecomp0__xunpack0__ligerloss0__gradofftrue__weightofftrue/b4_s4096_ga1/
# zero3 baseline  -- the bar to beat
zero3_offload_mem__source__recomp__polnone__routerwhole__expact0__attnact0__layeract0__layergc0__sdparecomp0__loraafwdhbm__actrecomp0__xunpack0__ligerloss0/b4_s4096_ga1/
# also produced (secondary): asym ... polgc-layer ... expact0__attnact0__layeract0__layergc0__sdparecomp0 ...
```

Gate: both scripts exit 0; each target `__source__` run dir has `memory_breakdown_summary.json` (directly under `b4_s4096_ga1/`, or in its `memory_breakdown/` subdir). The Stage-5 `find` below matches it regardless of the exact suffix tokens, so you never hand-type these.

---

## Stage 5 — Validate the memory goal

```bash
CFG="$CMP/asym_long_sft_smoke__lora__lf__bf16/qwen3-32b__gpus1__b4_s4096_ga1_w5_s10_r64_a16_drop000"

# Locate the two summaries by backend + policy-tag substring, anchored on the seq leaf and the
# __source__ variant (where the breakdown is materialized under PROFILERS=both). The `*/...` glob
# matches both `b4_s4096_ga1/memory_breakdown_summary.json` and the `memory_breakdown/` subdir form.
ASYM=$(find "$CFG" -path "*asym_cpuadamwds*source*norecomp*expact1*attnact1*layeract0*layergc1*sdparecomp1*/b4_s4096_ga1/*memory_breakdown_summary.json" | sort | head -1)
Z3M=$( find "$CFG" -path "*zero3_offload_mem*source*recomp*/b4_s4096_ga1/*memory_breakdown_summary.json" | sort | head -1)
[ -n "$ASYM" ] && [ -n "$Z3M" ] || { echo "ERROR: summary not found; check run dirs under $CFG"; find "$CFG" -name memory_breakdown_summary.json; }
echo "asym : $ASYM"; echo "zero3: $Z3M"

# schema-validate both
$PY scripts/lf/validate_lf_memory_capacity_schema.py --memory-breakdown-summary "$ASYM" --require-breakdown
$PY scripts/lf/validate_lf_memory_capacity_schema.py --memory-breakdown-summary "$Z3M"  --require-breakdown

# compare peak allocated HBM (jq is NOT installed in this env — use python)
$PY - "$ASYM" "$Z3M" <<'PY'
import json, sys
a=json.load(open(sys.argv[1]))["peak_allocated_hbm_bytes"]
b=json.load(open(sys.argv[2]))["peak_allocated_hbm_bytes"]
g=1024**3
print(f"asym(none|T|T|F|T|T) = {a/g:.2f} GiB")
print(f"zero3_offload_mem    = {b/g:.2f} GiB")
print(f"ratio asym/zero3_mem = {a/b:.3f}  ->  {'PASS (asym lower)' if a<b else 'FAIL (asym not lower)'}")
PY
```

Combined comparison plot (auto-generated when `PLOT=true`, the default) lands under the config dir:

```bash
find "$CFG" -name "*.png" -path "*memory*" | sort
# also the per-run stacked breakdowns under each run's memory_plots/
```

**Pass:** `asym(none|true|true|false|true|true)` `peak_allocated_hbm_bytes` < `zero3_offload_mem|recomp`.

Also report the component split from each summary to explain *where* the win comes from:

```bash
$PY - "$ASYM" "$Z3M" <<'PY'
import json, sys
g=1024**3
for f in sys.argv[1:]:
    d=json.load(open(f))
    k=["peak_allocated_hbm_bytes","activation_hbm_bytes_at_peak","saved_activation_hbm_bytes_at_peak",
       "live_activation_hbm_bytes_at_peak","temporary_workspace_hbm_bytes_at_peak"]
    print(f"== {f.split('/')[-3]} ==")
    print({x: round(d.get(x,0)/g,2) for x in k})
PY
```

---

## Risks / contingencies

1. **The CE-logits peak is shared by both backends — interpret accordingly.** The final cross-entropy logits tensor (`batch 4 × seq 4096 × vocab 151936` ≈ 5.0 GiB bf16, ≈ 9.9 GiB if upcast to fp32) is materialized identically by the asym run and the zero3 run, so it inflates **both** peaks by the same amount. The asym advantage comes from everything *else* (CPU-offloaded + selectively-checkpointed activations vs zero3's resident working set), so under `ligerloss0` expect `asym < zero3` by that delta, not by a huge factor. Prior analogous MoE data (memory note `asymgemm-qwen35-loss-floor`) shows the win holds without fused loss: asym 46.08 vs zero3 50.99 GiB. Use the Stage-5 component split to confirm: the logits term shows up under `live_activation_hbm_bytes_at_peak` and should be comparable across both; the difference should live in `saved_activation_hbm_bytes_at_peak` / resident base.
   **Do NOT expect `ligerloss1` to help here as-is.** `install_asym_liger_loss_bridge` (`asym_gemm/integrations/liger_loss.py:720-732`) branches only on `llama4` / `qwen3_5_moe` / `qwen3_moe`; for dense `model_type=="qwen3"` it falls through to `return False`, so the fused-CE bridge is a **silent no-op** (no error, no fused loss, full logits peak retained). Enabling fused CE for dense Qwen3-32B needs new code — **Appendix A** has the complete, copy-paste implementation. It is **not** required to satisfy the primary goal; treat it as an enhancement for a cleaner margin.
2. **Tied lm_head.** `offload_modules=all` offloads `lm_head`. Qwen3-32B has `tie_word_embeddings=false`, so `_reject_tied_lm_head_offload` is satisfied. If a tied variant is ever used, that guard fires — expected.
3. **Kernel alignment fallbacks.** AsymGEMM kernels need 64-aligned `out_features`. Qwen3-32B dims (q=8192, k/v=1024, o=5120, gate/up=25600, down=5120, vocab=151936) all satisfy this; misaligned leaves would degrade gracefully to `torch_cpu_fetched` (a `report.skipped` note), not error.
4. **CPU RAM.** 32B bf16 base (~65 GB) + fp32 master + optimizer offload. Run test1 then test2 sequentially. Watch `free -g` during the run.

## Rollback

Single-file revert of the Stage 1 branch in `asym_gemm/integrations/lf.py` — restore the original `if not expert_prefixes and strict: raise ValueError(...)` and delete the `dense_no_experts` lines. Nothing else in the source tree is touched, and `profile_lora_lf_test1.sh` / `test2.sh` are unmodified (env overrides only), so reverting is a no-risk, MoE-safe operation.

## Scope summary (what changes vs what is guaranteed untouched)

| Area | Change |
| --- | --- |
| `asym_gemm/integrations/lf.py` expert-wrap guard | +4 lines: explicit `dense_no_experts` branch (dense skips; MoE unchanged) |
| MoE wrapping loop / detectors / reports | **unchanged** |
| Decoder GC, attn_act, sdpa, dense LoRA, optimizer, profiling | **unchanged** (already generic) |
| `profile_lora_lf_test1.sh` / `test2.sh` | **unchanged** (run with `MODEL_SPECS`/`OUTPUT_ROOT` env only) |
| Existing models (Qwen3-MoE / Qwen3.5 / Llama4) behavior | **identical** — they never enter the dense branch |

---

## Appendix A — OPTIONAL: fused-CE (Liger) bridge for dense Qwen3 (only if `ligerloss1` is wanted)

Not needed for the primary goal. Add this only to remove the shared CE-logits peak for a cleaner margin (`BACKEND_SPECS="asym_cpuadamwds|norecomp|ligerloss1"`). As-is, `ligerloss1` on dense Qwen3 is a silent no-op (root cause: `install_asym_liger_loss_bridge` has no `qwen3` branch). The fix mirrors the existing `qwen3_moe` bridge (`liger_loss.py:176-299`) with the MoE/router parts removed. All helpers used below already exist in `liger_loss.py`.

**A1. New dense forward + installer** — add to `asym_gemm/integrations/liger_loss.py` (next to `install_asym_liger_qwen3_moe_loss_bridge`). Add `from transformers.modeling_outputs import CausalLMOutputWithPast` to the imports if not already present.

```python
def asym_qwen3_dense_lce_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs: Any,
):
    # Dense Qwen3: identical to asym_qwen3_moe_lce_forward minus output_router_logits / aux_loss.
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    shift_labels = kwargs.pop("shift_labels", None)
    logits = None
    loss = None
    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        lm_head_weight = _resolve_liger_lm_head_weight(self.lm_head, kept_hidden_states)
        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=lm_head_weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.hidden_size,
            **kwargs,
        )
        loss, _, _token_accuracy, _predicted_tokens = unpack_cross_entropy_result(result)
    else:
        logits = self.lm_head(kept_hidden_states)
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, shift_labels=shift_labels,
                vocab_size=self.vocab_size, **kwargs,
            )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def install_asym_liger_qwen3_dense_loss_bridge(model: nn.Module, *, strict: bool = True) -> bool:
    target_model = _base_causal_lm_model(model)
    config = getattr(target_model, "config", None)
    if getattr(config, "model_type", None) != "qwen3":
        if strict:
            raise ValueError("Asym Liger dense bridge only supports qwen3.")
        return False
    validated = _validate_liger_lm_head(getattr(target_model, "lm_head", None), model_label="Qwen3", strict=strict)
    if validated is None:
        return False
    lm_head, weight_source = validated
    target_model.forward = MethodType(asym_qwen3_dense_lce_forward, target_model)
    _mark_liger_bridge_installed(target_model, lm_head, weight_source, "qwen3", "causal_lm")
    return True
```

**A2. Dispatcher branch** — in `install_asym_liger_loss_bridge` (`liger_loss.py:730`), add the `qwen3` case right after the `qwen3_moe` case:

```python
    if causal_type == "qwen3_moe":
        return install_asym_liger_qwen3_moe_loss_bridge(model, strict=strict)
    if causal_type == "qwen3":                                              # NEW (dense)
        return install_asym_liger_qwen3_dense_loss_bridge(model, strict=strict)
    return False
```

Also add `"asym_qwen3_dense_lce_forward"` and `"install_asym_liger_qwen3_dense_loss_bridge"` to `__all__` (`liger_loss.py:735`).

**A3. Non-regression for this addition.** The new `qwen3` branch is reached only for `causal_type == "qwen3"`; `qwen3_moe` / `qwen3_5_moe` / `llama4` hit their existing earlier branches unchanged. With `ASYM_OFFLOAD_MODULES=all`, `selection.lm_head` is True, so the adapter installs with `strict=asym_strict` (`adapter.py:520`).

**A4. Validation of the bridge (must smoke-test the new forward before trusting it):**

```bash
# 1-step real run with fused loss; expect the install log line and a finished step
MODEL_SPECS="Qwen/Qwen3-32B|1" GPU_POOL=3 MAX_STEPS=1 WARMUP_STEPS=1 WORKLOADS="2048|1|1" \
  BACKEND_SPECS="asym_cpuadamwds|norecomp|ligerloss1" \
  ASYMM_EXP_ACT_POLICIES="none|true|true|false|true|true" \
  OUTPUT_ROOT="$REPO/profiling_liger_smoke" \
  bash scripts/lf/profile_lora_lf_test2.sh 2>&1 | tee /tmp/liger_smoke.log
grep -E "Asym Liger loss bridge has been installed" /tmp/liger_smoke.log   # must appear (else still a no-op)
```

Then, for the cleaner comparison, run the full `ligerloss1` variants of both backends into a *separate* `OUTPUT_ROOT` (do not edit test1/test2's baked configs) and repeat Stage 5; expect `live_activation_hbm_bytes_at_peak` (the logits term) to collapse and the asym/zero3 gap to widen.
