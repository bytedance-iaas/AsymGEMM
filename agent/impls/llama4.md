# Llama 4 — AsymGEMM LoRA-SFT memory-offload port (staged plan)

## State of the world

- **One shared expert engine.** `AsymQwen3Experts` (`asym_gemm/training/qwen3_moe.py:1886`) is the only MoE engine; `packed_moe.py:10` aliases it as `AsymPackedExperts`. Llama 4 routed experts already run it via `AsymLlama4Moe` → `wrap_packed_experts` (`llama4_moe.py:199`). They inherit for free: grouped GEMM over experts (no per-expert Python loop — verified zero `for expert in range(...)` in the hot path), fused gate+up split-LoRA (`qwen3_moe.py:2047-2096`), expert-activation offload (`exp_act_offload_lora.py`, gated by `ASYMM_EXPERT_ACT_OFFLOAD`), expert recompute policy, and LoRA weight-offload (`weight_offload.py:256` scans `isinstance(m, AsymQwen3Experts)` — Llama4 experts ARE that class). Llama4 routing uses the input-scaled + summed combine `forward_input_scaled` (`qwen3_moe.py:2508`), which exactly mirrors HF `Llama4TextMoe.forward` semantics.
- **Scaffold exists and is wired.** `is_llama4_moe`/`wrap_llama4_moe`/`AsymLlama4Moe`/`AsymLlama4Router` (`llama4_moe.py`) normalize `gate_up_proj [E,hidden,2*expert]→[E,2*expert,hidden]` (transpose preserves row order ⇒ `chunk(2)` gate/up correct) and `down_proj [E,expert,hidden]→[E,hidden,expert]`. Dispatched in `lf.py:1088` (router_mode=whole) and `lf.py:1131` (router_mode=hf).
- **What already works:** routed-expert LoRA + base CPU offload (`ASYM_OFFLOAD_MODULES=routed_experts`), expert-activation offload, expert recompute (`gc-exp`), expert LoRA weight offload, router CPU offload (`AsymLlama4Router`), and — because the attention matcher is generic — attention activation offload + `gc-attn-exp` checkpoint on Llama4's `self_attn` (q/k/v/o children present).
- **The port is about NON-EXPERT glue.** Confirmed gaps: **G1** decoder-layer offload/`gc-layer` matcher requires a child named `mlp` (`lf.py:806`); Llama4 uses `feed_forward` ⇒ strict RAISE. **G2** dense-FFN (Maverick interleaved layers) base CPU offload missing: `classify_lf_component` only tags `.mlp.{gate,up,down}_proj` as `mlp_dense` (`lf.py:436`), and `mlp_dense` is not a selectable component. **G3** shared-expert (always-on dense MLP) needs verification of base offload + LoRA, and it gets NO expert-engine act-offload/recompute. **G4** attention correctness with iRoPE internals (verify, expected fine). **G5** bf16-only (Maverick is fp8 ⇒ validate on Scout-bf16/small bf16 config). **G6** router/scaling numerics parity (verify).

### HF ground-truth (re-verify line numbers before editing; from `third_party/transformers/src/transformers/models/llama4/modeling_llama4.py`)
- `Llama4TextDecoderLayer` (~413): children `self_attn`, **`feed_forward`** (Llama4TextMoe if `layer_idx in config.moe_layers` else dense `Llama4TextMLP(intermediate_size_mlp)`), `input_layernorm`, `post_attention_layernorm`. **No `mlp` child.** `self.is_moe_layer` bool records which. `forward(...)` takes `position_embeddings, attention_mask, position_ids, past_key_values, use_cache` and **returns a single tensor** (unpacks `(out, router_logits)` when MoE).
- `Llama4TextMoe` (~157): `forward` → `routed_in = h.repeat(num_experts,1) * router_scores.T.reshape(-1,1)`; `routed_out = experts(routed_in)`; `out = shared_expert(h)`; `out.add_(routed_out.reshape(num_experts,-1,hidden).sum(0))`; returns `(out, router_logits)`. `top_k=num_experts_per_tok` (default 1).
- `Llama4Router` (~142): subclass of `nn.Linear`; `forward` → topk over experts → scatter to `-inf` → `sigmoid(.float())`; returns `(router_scores, router_logits)`. Weight `[num_local_experts, hidden]`, `bias=False`.
- `Llama4TextExperts` (~56): `gate_up_proj` Param `[E,hidden,2*expert_dim]`, `down_proj` Param `[E,expert_dim,hidden]`, `act_fn=ACT2FN[hidden_act]` (silu). Native forward is `bmm` (our grouped GEMM is the win).
- `Llama4TextMLP` (~89): `gate_proj`/`up_proj`/`down_proj` are `nn.Linear(bias=False)`, attr `activation_fn`. Used as dense-layer FFN (`intermediate_size_mlp`=16384) AND as MoE `shared_expert` (`intermediate_size`=8192).
- `Llama4TextAttention` (~321): q/k/v/o are plain `nn.Linear` on every layer. `use_rope=config.no_rope_layers[layer_idx]`; `qk_norm = Llama4TextL2Norm` only `if use_qk_norm and use_rope`; `attn_temperature_tuning` (NoPE layers); `forward` returns **`(attn_output, attn_weights)`**. **All exotic ops (RoPE, L2 qk_norm, temperature, chunked mask) are DOWNSTREAM of q/k/v projections and upstream of o_proj.**
- `Llama4TextL2Norm` (~107): **STATELESS** (`self.eps` only, no `nn.Parameter`).
- Config: `tie_word_embeddings=False`, `output_router_logits=False` (default), `moe_layers` derived in `__post_init__` as `range(interleave_moe_layer_step-1, num_hidden_layers, interleave_moe_layer_step)` (Scout step=1 ⇒ all MoE; Maverick step=2 ⇒ interleaved). `num_experts_per_tok=1`, `num_local_experts` (Scout 16).

### Profiling truth (from `run_lf_profiled_train.py`)
- Peak GPU memory: profile JSON `memory.gpu.peak_allocated_hbm_bytes` (= `global_peak_allocated_bytes`, sourced from `torch.cuda.max_memory_allocated()`, line 2267/2564). Also `memory.gpu.peak_reserved_hbm_bytes`.
- Step latency: top-level `measured_e2e_step_milliseconds` (line 477; warmup-excluded mean from trainer log elapsed). Use this, NOT `step.total_milliseconds` (that is a per-stage profiler sum, higher overhead).
- When `PROFILE_PROFILER=source`, the full report may be nested under `source_profile` in the written `profile.json`; both validation snippets below dereference `source_profile` if present (mirroring the script's own `profile.get("source_profile", profile)`).
- `config` carries `backend`, `seq_len`, `model_name_or_path`, `asym_offload_modules`, `asymm_layer_act_offload`, `asymm_attn_act_offload`, `asymm_expert_act_offload`, `lora_target`, etc.

### Validation harness conventions used by every stage
- Driver: `scripts/lf/run_lf_lora_sft.sh` (env-var driven; `--lora_target all` at line 1493; asym args at `lf.py` CMD 1538-1542; `PROFILE=1` routes through `run_lf_profiled_train.py`).
- **bf16 model for validation.** Either `meta-llama/Llama-4-Scout-17B-16E` (bf16, all-MoE — exercises G1/G3/G4/G6) on a 96GB+ GPU, or, if VRAM-bound, a **small bf16 Llama4-text config** (see "Small bf16 Llama4 fixture" at the end) which also exercises G2 (interleaved dense layers). Maverick is fp8 ⇒ **do not** use for the asym path (G5).
- Each stage's accept/reject compares the change run against its own baseline run with identical model/seq/batch/rank, differing ONLY in the one flag under test. Helper `jq_peak`/`jq_step` defined once below.

```bash
# Common helpers (paste once per shell). PROF=path to the run's profile.json.
jq_peak() { python3 - "$1" <<'PY'
import json,sys; p=json.load(open(sys.argv[1])); p=p.get("source_profile",p)
print(p["memory"]["gpu"]["peak_allocated_hbm_bytes"])
PY
}
jq_step() { python3 - "$1" <<'PY'
import json,sys; p=json.load(open(sys.argv[1])); p=p.get("source_profile",p)
v=p.get("measured_e2e_step_milliseconds"); print(v if v is not None else "null")
PY
}
decide() { # decide BASE_PEAK CHG_PEAK BASE_MS CHG_MS  -> prints ACCEPT/REJECT + deltas
python3 - "$@" <<'PY'
import sys
bp,cp,bm,cm=[float(x) for x in sys.argv[1:5]]
dmem=(cp-bp)/bp*100.0; dms=(cm-bm)/bm*100.0; dmib=(bp-cp)/2**20
ok = (dmib>=512 or dmem<=-3.0) and dms<=5.0   # see per-stage thresholds; this is the generic gate
print(f"peak: {bp/2**20:.0f}->{cp/2**20:.0f} MiB ({dmem:+.1f}%, saved {dmib:.0f} MiB)")
print(f"step: {bm:.1f}->{cm:.1f} ms ({dms:+.1f}%)")
print("ACCEPT" if ok else "REJECT")
PY
}
```

---

## Stage 0 — baseline + correctness harness (MANDATORY; nothing is accepted without this)

### SCOPE
Files touched: **none** (validation only; produce a small bf16 fixture config under your scratch dir, not in-repo). Prove the existing scaffold runs e2e and is numerically correct before changing anything.

Goals:
1. `is_llama4_moe` matches every MoE block; `report.llama4_moes_wrapped == num_moe_layers`.
2. Forward+backward parity asym-vs-torch within tolerance (the asym path replaces routed experts with grouped GEMM + LoRA, and the router runs no-grad in `whole` mode; both must match HF math).
3. Record baseline peak GPU mem + `measured_e2e_step_milliseconds` for: (a) `routed_experts` offload only, (b) `all` offload. These are the reference numbers all later stages diff against.

### CORRECTNESS (isolated, fast — the ONLY place a non-e2e micro-test is allowed, because it is a self-contained numerics check of the wrapper)
Build a tiny bf16 Llama4 MoE block + a few dense layers in-process and compare `AsymLlama4Moe` vs raw HF `Llama4TextMoe`, plus the full decoder layer. Pseudocode:

```python
# tools/_llama4_parity.py (scratch, do not commit)
import torch
from transformers.models.llama4.modeling_llama4 import Llama4TextConfig, Llama4TextMoe, Llama4TextDecoderLayer
from asym_gemm.training.llama4_moe import wrap_llama4_moe, is_llama4_moe

cfg = Llama4TextConfig(hidden_size=512, intermediate_size=256, intermediate_size_mlp=512,
                       num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
                       head_dim=64, num_local_experts=8, num_experts_per_tok=1,
                       interleave_moe_layer_step=2,            # ⇒ moe_layers=[1,3]; layers 0,2 dense (exercises G2)
                       use_qk_norm=True, attn_temperature_tuning=True, torch_dtype=torch.bfloat16)
ref = Llama4TextMoe(cfg).to("cuda", torch.bfloat16).eval()
for p in ref.parameters(): p.requires_grad_(False)
# CPU-first copy for the asym wrapper (offload requires CPU-resident bf16 source)
import copy; src = copy.deepcopy(ref).to("cpu")
assert is_llama4_moe(src)
asym = wrap_llama4_moe(src, backend="asym", precision="bf16", offload=True,
                       lora_rank=8, lora_alpha=16, lora_dropout=0.0,
                       router_mode="whole", strict=True).to("cuda")
# zero LoRA-B ⇒ asym output must equal frozen-base HF output (LoRA delta = 0)
for n,p in asym.named_parameters():
    if "lora_B" in n: torch.nn.init.zeros_(p)
x = torch.randn(2, 16, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    o_ref,_ = ref(x); o_asym,_ = asym(x)
torch.testing.assert_close(o_asym, o_ref, rtol=2e-2, atol=2e-2)   # bf16 grouped-GEMM tolerance
# Backward sanity: nonzero LoRA-B, check grads flow to LoRA only, base frozen
for n,p in asym.named_parameters():
    if "lora_B" in n: torch.nn.init.normal_(p, std=1e-3)
y,_ = asym(x); y.sum().backward()
assert all(p.grad is None for n,p in asym.named_parameters() if "lora_" not in n)
assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in asym.named_parameters() if "lora_A" in n)
```
**Watch:** grouped-GEMM bf16 requires `hidden`/`intermediate` multiples of 64 (see `_activation_offload_unsupported_reasons`); pick fixture dims accordingly. If `torch._grouped_mm` is unavailable on the box, the parity test still runs the eager grouped path; record which path executed.

### EXACT VALIDATION COMMANDS (e2e — the real acceptance baseline)
```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
M=meta-llama/Llama-4-Scout-17B-16E   # or your small bf16 fixture path
# (a) torch reference  — numerics + memory reference
MODEL_NAME_OR_PATH=$M BACKEND=torch  PROFILE=1 PROFILE_PROFILER=source \
  SEQ_LENS_OVERRIDE= CUTOFF_LEN=4096 PER_DEVICE_TRAIN_BATCH_SIZE=1 MAX_STEPS=8 \
  LORA_RANK=16 LORA_ALPHA=16 LORA_DROPOUT=0.0 ASYM_ROUTER_MODE=whole \
  scripts/lf/run_lf_lora_sft.sh
# (b) asym, routed_experts only  (PRIMARY Stage-0 baseline)
MODEL_NAME_OR_PATH=$M BACKEND=asym  PROFILE=1 PROFILE_PROFILER=source \
  CUTOFF_LEN=4096 PER_DEVICE_TRAIN_BATCH_SIZE=1 MAX_STEPS=8 LORA_RANK=16 LORA_ALPHA=16 \
  LORA_DROPOUT=0.0 ASYM_OFFLOAD_MODULES=routed_experts ASYM_EXPERT_RECOMPUTE_POLICY=none \
  ASYM_ROUTER_MODE=whole ASYM_STRICT=true USE_ASYM_CPU_ADAMW=false \
  scripts/lf/run_lf_lora_sft.sh
# (c) asym, all offload  (baseline for Stage 2/3 'all' comparisons)
MODEL_NAME_OR_PATH=$M BACKEND=asym  PROFILE=1 PROFILE_PROFILER=source \
  CUTOFF_LEN=4096 PER_DEVICE_TRAIN_BATCH_SIZE=1 MAX_STEPS=8 LORA_RANK=16 LORA_ALPHA=16 \
  LORA_DROPOUT=0.0 ASYM_OFFLOAD_MODULES=all ASYM_EXPERT_RECOMPUTE_POLICY=none \
  ASYM_ROUTER_MODE=whole ASYM_STRICT=true USE_ASYM_CPU_ADAMW=false \
  scripts/lf/run_lf_lora_sft.sh
# locate each run's profile.json (printed as OUT_DIR in the log) then:
#   jq_peak <OUT_DIR>/.../profile.json ; jq_step <OUT_DIR>/.../profile.json
```
Parity acceptance (b vs a): final-step training loss within ~2% relative (read trainer log `loss`), and run (b) completes without strict RAISE. Record peak/step for (b) and (c).

### ACCEPTANCE RULE
Stage 0 has no "keep/reject" — it gates the project. Block all later stages until: (1) `llama4_moes_wrapped == #MoE layers`, (2) parity passes, (3) baselines (b) and (c) recorded. If parity fails, fix is in the scaffold/engine, out of scope of later stages — STOP and report.

### RISKS / watch later
- `router_mode=whole` runs the router under `no_grad` (`llama4_moe.py:244`) and asserts non-differentiable `input_weights`. HF's `output_router_logits=False` default + `tie_word_embeddings=False` (verified) ⇒ no aux loss and the `lf.py:1018` whole-mode guard passes. **Watch:** if a Scout/Maverick `config.json` ships `output_router_logits=True`, whole-mode RAISES — fall back to `ASYM_ROUTER_MODE=hf`.
- Scout download is large/fp8-free but bf16 ⇒ heavy. The small fixture is the fast path; Scout is the final confirmation.

---

## Stage 1 — generalize the decoder-layer matcher (G1): enable `ASYMM_LAYER_ACT_OFFLOAD` + `gc-layer`

### SCOPE
File: `asym_gemm/integrations/lf.py`. Function: **`_is_qwen3_decoder_layer_module_name`** (currently `lf.py:800-815`). It is reused by BOTH `_wrap_qwen3_decoder_saved_tensor_offload_modules` (`lf.py:905`, activation offload) AND `_wrap_qwen3_decoder_checkpoint_modules` (`lf.py:853`, `gc-layer`). The installers `install_decoder_saved_tensor_offload` / `install_decoder_checkpoint` are fully **structure-agnostic** (they `saved_tensors_hooks`/`checkpoint` the whole layer forward; no child-name dependence) — verified. So **only the matcher** must change; no installer edits.

Today the matcher requires `{self_attn, mlp, input_layernorm, post_attention_layernorm}` ⊆ children. Llama4 has `feed_forward` not `mlp`, and is not qwen3 ⇒ every Llama4 layer is skipped ⇒ with `ASYMM_LAYER_ACT_OFFLOAD=true` or `gc-layer`, strict RAISES "no supported decoder layers".

### INTENDED CODE CHANGE (pseudocode)
Generalize to accept Llama4 lineage with a `feed_forward` child, keeping the Qwen3 path intact. Rename optional but keep the symbol name to avoid touching call sites; just broaden the predicate.

```python
def _is_qwen3_decoder_layer_module_name(name, module):   # now "supported text decoder layer"
    if not name or _has_attention_excluded_path_marker(name):   # excludes vision/multimodal
        return False
    children = set(dict(module.named_children()))
    norms_ok = {"input_layernorm", "post_attention_layernorm"} <= children
    if not ({"self_attn"} <= children and norms_ok):
        return False
    class_name  = type(module).__name__.lower()
    module_name = type(module).__module__.lower()
    config      = getattr(module, "config", None)
    model_type  = str(getattr(config, "model_type", "")).lower()
    # Qwen3 path (unchanged): requires a child named `mlp`
    if "mlp" in children and (
        "qwen3" in class_name or "qwen3" in module_name
        or model_type in {"qwen3_moe", "qwen3_vl_moe"}
        or hasattr(dict(module.named_children())["mlp"], "_is_asym_qwen3_moe_block")
        or is_qwen3_moe_block(dict(module.named_children())["mlp"])
    ):
        return True
    # Llama4 path (NEW): child `feed_forward` + llama4 lineage.
    if "feed_forward" in children and (
        "llama4" in class_name or "llama4" in module_name
        or model_type in {"llama4", "llama4_text"}
    ):
        return True
    return False
```
Notes: match on the FULL decoder layer (so the saved-tensor-hooks region spans attention + feed_forward, the largest activation window). The Llama4 layer forward already returns a single tensor and honors `use_cache` short-circuit in the installer. No change to `_wrap_*` callers; both the offload and checkpoint paths light up automatically. After replacement the `feed_forward` child may itself be an `AsymLlama4Moe` (expert work happens inside) — wrapping the whole layer is still correct because saved-tensor offload composes with the engine's own activation handling (the engine's offloaded tensors are CPU already and won't be re-offloaded; the wrapper only offloads CUDA saved tensors over `min_bytes`).

### VALIDATION (e2e)
`gc-layer` and `ASYMM_LAYER_ACT_OFFLOAD` both require expert policy `none`-or-`gc-layer` and backend `asym` (`lf.py:1039-1042`). Run against the Stage-0 (b) baseline (`routed_experts`, policy none, no layer act).

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
M=meta-llama/Llama-4-Scout-17B-16E   # or small bf16 fixture
COMMON="PROFILE=1 PROFILE_PROFILER=source CUTOFF_LEN=4096 PER_DEVICE_TRAIN_BATCH_SIZE=1 \
  MAX_STEPS=8 LORA_RANK=16 LORA_ALPHA=16 LORA_DROPOUT=0.0 ASYM_ROUTER_MODE=whole \
  ASYM_STRICT=true USE_ASYM_CPU_ADAMW=false BACKEND=asym MODEL_NAME_OR_PATH=$M"
# baseline = Stage-0 (b): ASYM_OFFLOAD_MODULES=routed_experts ASYMM_LAYER_ACT_OFFLOAD=false
# change A: layer activation offload
env $COMMON ASYM_OFFLOAD_MODULES=routed_experts ASYMM_LAYER_ACT_OFFLOAD=true \
  ASYM_EXPERT_RECOMPUTE_POLICY=none scripts/lf/run_lf_lora_sft.sh
# change B: gc-layer (full-layer gradient checkpointing)
env $COMMON ASYM_OFFLOAD_MODULES=routed_experts ASYMM_LAYER_ACT_OFFLOAD=false \
  ASYM_EXPERT_RECOMPUTE_POLICY=gc-layer scripts/lf/run_lf_lora_sft.sh
# then: confirm wrapped count > 0 and compute decision
python3 - <<'PY'
import json
b=json.load(open("BASE/profile.json")); b=b.get("source_profile",b)
c=json.load(open("CHG/profile.json")); c=c.get("source_profile",c)
print("layer_act_wrapped:", c["config"].get("asymm_layer_act_offload"))  # sanity flag set
PY
# decide BASE_PEAK CHG_PEAK BASE_MS CHG_MS   (use jq_peak/jq_step)
```
Also confirm in the run log that `report.layer_act_offload_wrapped` / `report.layer_gc_wrapped` equals the number of decoder layers (grep the train log for the LF asym report line, or read `model._asym_layer_act_offload_modules`). A non-zero count proves the matcher fix; zero means it still skipped.

### ACCEPTANCE RULE (numeric)
Layer activation offload trades activation HBM for H2D/D2H copy traffic. KEEP change A only if peak `peak_allocated_hbm_bytes` drops by **≥ 5% OR ≥ 1024 MiB absolute** at the chosen seq/batch (activations dominate at long seq) AND `measured_e2e_step_milliseconds` rises by **≤ 5%**. KEEP change B (`gc-layer`) only if peak drops **≥ 5%** AND step rises **≤ 10%** (recompute legitimately costs an extra forward; full-layer GC is the heaviest, so the latency band is wider but bounded). REJECT if wrapped-count is 0 (matcher still broken), if memory is flat, or if latency exceeds the band. If at short seq the activation savings are trivial, re-run at the realistic training seq (e.g. CUTOFF_LEN=8192) before deciding — the lever only pays off where activations are large.

### RISKS / watch later
- The Llama4 decoder forward takes `position_embeddings` (rotary tuple) as a kwarg; saved-tensors-hooks wrap the closure transparently, so positional/kwarg passing is unaffected. **Watch:** if some Llama4 wrapper passes `use_cache=True` during training (it should be False under HF Trainer), the installer short-circuits and offload silently no-ops — assert `use_cache` is False in the wrapped path during validation.
- `gc-layer` + the engine's own expert recompute are mutually exclusive by policy (`gc-layer` IS the policy). No double-recompute risk.
- Vision/multimodal exclusion: `_has_attention_excluded_path_marker` already filters `.vision_model.` etc.; Llama4 multimodal checkpoints would route those layers to skip — fine.

---

## Stage 2 — dense-FFN + shared-expert base CPU offload (G2/G3)

### SCOPE
File: `asym_gemm/integrations/lf.py`. Functions/areas:
- **`classify_lf_component`** (`lf.py:372-438`): currently tags dense FFN only under `.mlp.` (line 436). Llama4 dense FFN is `...feed_forward.{gate,up,down}_proj` ⇒ classified `"other"` ⇒ no base offload. Shared expert is `...shared_expert.{gate,up,down}_proj` and IS already caught by the `.shared_expert.` rule (line 375) → `"shared_experts"` (verify in Stage-0 dump).
- **`LFOffloadSelection`** (`lf.py:88-97`) + **`SUPPORTED_LF_OFFLOAD_COMPONENTS`** (`lf.py:71`) + **`parse_lf_offload_modules`** (`lf.py:298-369`) + **`component_is_selected`** (`lf.py:441-456`): add a selectable `dense_mlp` component (covers Maverick interleaved dense layers). There is currently NO selectable dense-MLP component — confirmed.

G3 (shared expert) is mostly **a verification task**: after block replacement, the dense walk (`lf.py:1282`) reaches `{block}.shared_expert.{gate,up,down}_proj`, classifies `shared_experts`, and gets base CPU offload + LoRA **iff** `shared_experts ∈ ASYM_OFFLOAD_MODULES` and `lora_target` includes those leaves (`all` does). The shared expert is dense ⇒ it correctly gets NO expert-engine act-offload/recompute; its activation lever is the Stage-1 decoder saved-tensor offload (it lives inside the decoder layer). So G3 needs: (1) confirm offload+LoRA actually happen, (2) rely on Stage 1 for its activation memory, (3) NO new code beyond the Stage-2 classifier work if `.shared_expert.` already classifies correctly (it does).

G2 (dense FFN, Maverick only — Scout is all-MoE so has zero dense FFN layers) needs real code: classify `feed_forward.{gate,up,down}_proj` as a dense FFN component and make it selectable.

### INTENDED CODE CHANGES (pseudocode)

1) Extend the classifier so Llama4 dense FFN is recognized (and keep the legacy `.mlp.` behavior). IMPORTANT: this rule must run AFTER the `shared_expert`/`experts`/`router` rules so the shared expert (also `feed_forward.shared_expert.*` — but Llama4 shared_expert is `{block}.shared_expert`, a sibling of `feed_forward`, so no collision) and routed experts are not misclassified.
```python
# in classify_lf_component, replace the final dense-mlp rule (line ~436):
if parent_leaf in {"gate_proj", "up_proj", "down_proj"} and (
    ".mlp." in lower or ".feed_forward." in lower    # Llama4 dense FFN lives under .feed_forward.
):
    return "mlp_dense"          # keep the existing label so reporting/plotting keys are stable
```
Rationale for label reuse: the memory-breakdown plotting buckets use `mlp` already; reusing `mlp_dense` avoids a new bucket. The selector name can differ from the classifier label.

2) Make `mlp_dense` selectable. Add component + alias + selection bool:
```python
# lf.py:71
SUPPORTED_LF_OFFLOAD_COMPONENTS = frozenset({
    "routed_experts","router","shared_experts","attention",
    "embed_tokens","lm_head","norms","dense_mlp",          # NEW
})
# LFOffloadSelection (lf.py:88): add field
    dense_mlp: bool = False
# implemented_components / any_cpu_offload: include dense_mlp
# parse_lf_offload_modules aliases (lf.py:306): add
    "mlp": "dense_mlp", "dense": "dense_mlp", "dense_ffn": "dense_mlp", "feed_forward": "dense_mlp",
# 'all' expansion already adds the whole SUPPORTED set ⇒ dense_mlp included under `all`.
# build LFOffloadSelection(..., dense_mlp=("dense_mlp" in expanded), ...)
# component_is_selected (lf.py:441): add
    if component == "mlp_dense":      # classifier label
        return selection.dense_mlp
```
The dense walk (`lf.py:1282-1342`) then naturally CPU-offloads these `nn.Linear` leaves via `_wrap_lf_linear_leaf` (`AsymFrozenLinear` for frozen base, `AsymLoRALinear` when also a LoRA target). No per-token/per-expert loop — these are plain dense GEMMs already; we only move the frozen base weight to pinned CPU and run `AsymFrozenLinear` (single GEMM, CPU-fetched bf16). No GEMM is split.

3) (Verification only, G3) In the Stage-0 model dump, confirm `classify_lf_component("...shared_expert.gate_proj") == "shared_experts"` and that with `ASYM_OFFLOAD_MODULES` containing `shared_experts` + `lora_target=all`, the run log reports those leaves as CPU-offloaded LoRA. No code change expected; if the dump shows the shared expert under `feed_forward.shared_expert` (it is a child of the MoE block, but after we replace the block with `AsymLlama4Moe`, the attribute path is `{block}.shared_expert` since `AsymLlama4Moe` keeps `self.shared_expert` raw — `llama4_moe.py:168`), the existing `.shared_expert.` substring rule still matches. Verify the actual dotted path in the Stage-0 dump and only adjust the classifier substring if the dump disagrees.

### VALIDATION (e2e)
Maverick is fp8 (cannot use asym). **G2 must be validated on the small bf16 fixture with `interleave_moe_layer_step=2`** (so half the layers are dense `Llama4TextMLP(intermediate_size_mlp)`), or any bf16 Llama4-text config that has dense layers. G3 validates on Scout (all-MoE, has shared experts) or the fixture.

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
M=/path/to/small_bf16_llama4_fixture        # interleaved dense+MoE, bf16
COMMON="PROFILE=1 PROFILE_PROFILER=source CUTOFF_LEN=4096 PER_DEVICE_TRAIN_BATCH_SIZE=1 \
  MAX_STEPS=8 LORA_RANK=16 LORA_ALPHA=16 LORA_DROPOUT=0.0 ASYM_ROUTER_MODE=whole \
  ASYM_STRICT=true USE_ASYM_CPU_ADAMW=false BACKEND=asym MODEL_NAME_OR_PATH=$M"
# G2 baseline: dense FFN base stays on GPU
env $COMMON ASYM_OFFLOAD_MODULES=routed_experts scripts/lf/run_lf_lora_sft.sh           # BASE
# G2 change: also offload dense FFN base to CPU
env $COMMON ASYM_OFFLOAD_MODULES=routed_experts,dense_mlp scripts/lf/run_lf_lora_sft.sh # CHG
# G3 verify (shared expert offload + LoRA actually applied):
env $COMMON ASYM_OFFLOAD_MODULES=routed_experts,shared_experts scripts/lf/run_lf_lora_sft.sh
#   grep train log for the LF asym report: shared_experts CPU-resident bytes > 0 and dense_lora_wrapped includes shared_expert leaves.
# decision: decide BASE_PEAK CHG_PEAK BASE_MS CHG_MS
```
To measure the dense-FFN base size that should leave HBM: per dense layer, `gate+up+down ≈ (2*hidden*16384 + 16384*hidden)*2 bytes`. Expect the peak drop to track `#dense_layers * that`. If the model is all-MoE (Scout), G2 has zero applicable modules — assert the selector reports `dense_mlp` selected-but-zero-matched and does NOT strict-RAISE (it must not, because `dense_mlp` is an offload component, not a LoRA target; missing matches are fine). Add that as an explicit no-op assertion.

### ACCEPTANCE RULE (numeric)
- **G2 (dense_mlp offload):** KEEP only if peak drops by **≥ #dense_layers × (per-layer dense base bytes) × 0.8** (i.e. at least 80% of the theoretical frozen-base savings is realized at peak) AND step latency rises **≤ 5%**. The frozen base is CPU-fetched per forward, so a small latency bump is expected; reject if > 5% or if the realized memory drop is < 50% of theory (means the base wasn't actually the peak contributor at this seq — then re-evaluate at a longer seq or drop the lever). REJECT outright on any strict RAISE for an all-MoE model.
- **G3 (shared_experts):** This is verification; "accept" = the run log shows shared-expert base CPU-resident bytes > 0 and (with `lora_target=all`) shared-expert LoRA wrapped. The memory lever for the shared expert's *activations* is Stage 1; do not separately accept/reject G3 on memory beyond confirming the base left HBM (peak drop ≥ shared-expert base bytes × 0.8 when `shared_experts` is the only delta vs baseline, step ≤ 5%).

### RISKS / watch later
- **Classifier ordering bug risk:** the new `.feed_forward.` dense rule MUST sit after the experts/router/shared rules (it already would, being the last rule). Verify the Stage-0 dump shows `feed_forward.experts.*`→`routed_experts`, `feed_forward.router`→`router`, `shared_expert.*`→`shared_experts`, and only `feed_forward.{gate,up,down}_proj`→`mlp_dense`. A misorder would steal expert/router leaves into `mlp_dense`.
- **Naming collision:** Llama4 has NO `feed_forward.gate_proj` on MoE layers (MoE's gate is the `router`, an nn.Linear subclass, classified `router`). Dense layers have `feed_forward.{gate,up,down}_proj`. So the `.feed_forward.` + `{gate,up,down}_proj` predicate is unambiguous. Confirm no MoE block exposes a `feed_forward.gate_proj`.
- **Tied lm_head:** `tie_word_embeddings=False` (verified) ⇒ `_reject_tied_lm_head_offload` won't block `embed_tokens`/`lm_head` under `all`. Watch if a custom config ties them.
- The shared expert is small relative to routed experts; if its base offload yields a trivial peak drop, that is expected — accept on correctness (base off HBM), not on a large memory delta.

---

## Stage 3 — attention activation offload / checkpoint correctness with iRoPE internals (G4)

### SCOPE
File: `asym_gemm/integrations/lf.py` (expected **no code change**; this is a correctness+efficiency verification). The relevant matchers/wrappers are already generic:
- `_is_text_attention_module_name` (`lf.py:788-797`): requires an attention-ish leaf (`self_attn`) with all of `{q_proj,k_proj,v_proj,o_proj}` children — Llama4 `Llama4TextAttention` matches (verified). Used by both attention checkpoint (`gc-attn-exp`) and attention saved-tensor offload.
- `AsymActivationOffloadLoRALinear` (`attention_activation_offload.py:728`) + `AttentionActivationOffloadContext` (`:441`): a drop-in LoRA Linear that offloads its saved INPUT activation to CPU; q/k/v share one offloaded input via the context (built in `_build_attention_activation_contexts`, `lf.py:750`, only when q+k+v are all offload-eligible LoRA targets). Verified: it assumes nothing about RoPE/qk_norm/temperature — only that its input last-dim == `in_features`, bf16. All of Llama4's exotic ops are DOWNSTREAM of q/k/v and upstream of o_proj (verified), so offloading the q/k/v INPUT and o_proj INPUT is safe.
- Saved-tensor offload (`install_attention_saved_tensor_offload`) wraps the whole attention forward with `saved_tensors_hooks` — structure-agnostic (verified).

### WHAT TO VERIFY (no new code unless a check fails)
1. With `ASYMM_ATTN_ACT_OFFLOAD=true` + `lora_target=all` + `ASYM_OFFLOAD_MODULES` including `attention` (so q/k/v/o are CPU-offloaded LoRA targets), the run wraps Llama4 attention: `report.attention_act_offload_wrapped > 0` and `report.attention_saved_tensor_offload_wrapped > 0`. Numerics unchanged vs Stage-0 (b) (loss within ~2%).
2. `gc-attn-exp` (attention + expert gradient checkpointing) wraps Llama4 attention: `report.attention_gc_wrapped > 0`, runs, numerics ok.
3. The L2 `qk_norm` is stateless (verified) ⇒ no parameter is mis-offloaded by `norms` selection (it has no weight; the `norms` walk skips stateless modules via `_is_stateless_module`, `lf.py:1260`). Confirm `Llama4TextL2Norm` is treated as stateless (skipped) and `Llama4TextRMSNorm` (has weight) is the only norm offloaded.
4. **Efficiency:** attention activation offload must NOT introduce per-head or per-token small ops; it offloads the single packed input activation per projection (one D2H/H2D), and q/k/v share one buffer. Confirm via profiler that attention step time rise is bounded and there is no kernel-count explosion.

### INTENDED CODE CHANGE
None expected. **Contingency** (only if a check fails): if `_is_text_attention_module_name` fails to match because a Llama4 attention class name differs, the predicate already accepts leaf `self_attn` + the four children, so it should pass; if a future Llama4 uses fused `qkv_proj`, extend `_ATTENTION_TARGETS` handling — but current Llama4 has separate q/k/v/o (verified), so no change.

### VALIDATION (e2e)
```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
M=meta-llama/Llama-4-Scout-17B-16E   # or small bf16 fixture
COMMON="PROFILE=1 PROFILE_PROFILER=source CUTOFF_LEN=8192 PER_DEVICE_TRAIN_BATCH_SIZE=1 \
  MAX_STEPS=8 LORA_RANK=16 LORA_ALPHA=16 LORA_DROPOUT=0.0 ASYM_ROUTER_MODE=whole \
  ASYM_STRICT=true USE_ASYM_CPU_ADAMW=false BACKEND=asym MODEL_NAME_OR_PATH=$M"
# baseline: attention LoRA on GPU, no attn act offload
env $COMMON ASYM_OFFLOAD_MODULES=routed_experts,attention ASYMM_ATTN_ACT_OFFLOAD=false \
  ASYM_EXPERT_RECOMPUTE_POLICY=none scripts/lf/run_lf_lora_sft.sh                       # BASE
# change A: attention activation offload
env $COMMON ASYM_OFFLOAD_MODULES=routed_experts,attention ASYMM_ATTN_ACT_OFFLOAD=true \
  ASYM_EXPERT_RECOMPUTE_POLICY=none scripts/lf/run_lf_lora_sft.sh                       # CHG-A
# change B: gc-attn-exp (attention + expert checkpoint)
env $COMMON ASYM_OFFLOAD_MODULES=routed_experts,attention ASYMM_ATTN_ACT_OFFLOAD=false \
  ASYM_EXPERT_RECOMPUTE_POLICY=gc-attn-exp scripts/lf/run_lf_lora_sft.sh               # CHG-B
# verify wrapped counts and numerics, then decide
python3 - <<'PY'
import json
for tag in ("BASE","CHG-A","CHG-B"):
    p=json.load(open(f"{tag}/profile.json")); p=p.get("source_profile",p); c=p["config"]
    print(tag, "attn_act=",c.get("asymm_attn_act_offload"), "policy=",c.get("asym_expert_recompute_policy"))
PY
# decide BASE_PEAK CHGA_PEAK BASE_MS CHGA_MS   (and again for CHG-B)
```
LoRA dropout MUST be 0.0 for attention activation offload (`_wrap_lf_linear_leaf` raises NotImplementedError otherwise, `lf.py:605`) — already set.

### ACCEPTANCE RULE (numeric)
- **CHG-A (attn act offload):** KEEP only if peak drops by **≥ 5% OR ≥ 512 MiB** at the long seq (attention input activations scale with seq×hidden) AND step rises **≤ 5%**. REJECT if wrapped-count 0, memory flat, or latency > 5%.
- **CHG-B (`gc-attn-exp`):** KEEP only if peak drops **≥ 5%** AND step rises **≤ 10%** (recompute cost). REJECT otherwise.
- If both are accepted, also sanity-check they compose with Stage-1 layer offload without double-counting (run `ASYM_OFFLOAD_MODULES=all ASYMM_ATTN_ACT_OFFLOAD=true ASYMM_LAYER_ACT_OFFLOAD=true` and confirm no RAISE — note layer-act requires policy none, and attention-act is independent; they should coexist).

### RISKS / watch later
- **NoPE/temperature/chunked-mask layers:** temperature tuning multiplies q AFTER q_proj; chunked attention is a masking choice; both are inside the attention forward, downstream of the offloaded q/k/v input and upstream of o_proj. Saved-tensor offload wraps the entire forward, so the recompute/restore is transparent. **Watch:** confirm numerics on a config with at least one NoPE layer (the small fixture with default `no_rope_layer_interval=4` and ≥4 layers has one).
- **`position_embeddings` device cast:** Llama4 attention does `position_embeddings.to(query_states.device)`; with attention checkpoint (`gc-attn-exp`) the rotary tuple is re-supplied on recompute — fine since it's an input, not a saved tensor. Verify no device mismatch under checkpoint.
- **Attention context sharing requires q+k+v all offload-eligible:** if `ASYM_OFFLOAD_MODULES` omits one of q/k/v, `_build_attention_activation_contexts` won't create the shared context and each projection offloads independently (more D2H traffic). For the memory lever to be efficient, include full `attention` (all four) — the validation above does.

---

## Cross-cutting risks / watch-later

- **G5 (fp8 Maverick):** the asym path is bf16-only and RAISES otherwise (`AsymLlama4Moe.__init__` asserts bf16 source for experts/router; `AsymActivationOffloadLoRALinear` raises on non-bf16). **All validation uses Scout-bf16 or a small bf16 Llama4-text fixture.** Maverick is explicitly out of scope until a bf16 checkpoint or an upcast path exists. State this in any PR.
- **G6 (router/scaling numerics):** `AsymLlama4Router.forward` (`llama4_moe.py:108`) replicates HF `Llama4Router` exactly (topk→scatter(-inf)→`sigmoid(.float())`). `forward_input_scaled` (`qwen3_moe.py:2508`) input-scales by route weight then sums over top_k — matching HF `Llama4TextMoe` (`routed_in *= router_scores`; sum over experts; `+ shared`). Stage-0 parity is the gate. **Watch:** HF scales by the DENSE `router_scores` (per-expert sigmoid, non-selected=0) and repeats hidden `num_experts` times, then sums; the asym path packs only the top_k=1 selected route and scales by the gathered score. For top_k=1 these are numerically identical (only the selected expert has nonzero score). **If a future Llama4 sets top_k>1, re-verify** that gather-of-scores (asym) equals dense-score replication (HF) — they should, since non-selected scores are 0, but confirm with a top_k>1 fixture.
- **`router_mode=whole` aux-loss guard:** depends on `output_router_logits=False` (default). If a checkpoint enables it, use `ASYM_ROUTER_MODE=hf`. The `hf` dispatch path (`lf.py:1131`) wraps experts with `router_mode="hf"`; confirm it also runs (it is wired) if needed.
- **Profiler peak attribution:** `peak_allocated_hbm_bytes` is a global max across the step; CPU-Adam / weight-offload can shift where the peak occurs. Keep `USE_ASYM_CPU_ADAMW=false` in all stage comparisons to isolate the activation/base-offload lever (CPU-Adam is arch-agnostic and orthogonal; validate it separately last if desired).
- **Small bf16 Llama4 fixture (for fast + dense-layer coverage):** construct a `Llama4TextConfig` with small dims (e.g. hidden=1024, intermediate=512, intermediate_size_mlp=1024, num_hidden_layers=8, heads=8/kv=2, head_dim=128, num_local_experts=8, num_experts_per_tok=1, **interleave_moe_layer_step=2** ⇒ moe_layers=[1,3,5,7], dense=[0,2,4,6]), `torch_dtype=bfloat16`, save via `Llama4ForCausalLM(config).save_pretrained(path)` (random init is fine for memory/latency; for loss-parity use the in-process parity test in Stage 0). Dims must be multiples of 64 for grouped-GEMM. This single fixture exercises G1 (decoder layers), G2 (dense FFN), G3 (shared experts), G4 (attention incl. one NoPE layer), G6 (router) without downloading Scout.

## Efficiency invariants (enforced in every stage)
- Routed experts ALWAYS use the grouped GEMM + fused gate+up split-LoRA path; never a Python per-expert loop, never per-token small GEMMs. No stage introduces small-GEMM fan-out.
- Dense-FFN / shared-expert / attention base offload uses single per-Linear CPU-fetched bf16 GEMMs (`AsymFrozenLinear`/`AsymLoRALinear`) — one GEMM per projection, weight staged once per forward.
- Activation offloads (layer/attention) move whole packed activation tensors (one D2H/H2D each), not per-head/per-token slivers; q/k/v share one offloaded input buffer.
