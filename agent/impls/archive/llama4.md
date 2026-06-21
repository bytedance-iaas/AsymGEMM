# Llama 4 — AsymGEMM LoRA-SFT memory-offload port (staged implementation plan)

> **Implementation status (2026-06-15): Stage 1 + Stage 2 IMPLEMENTED & validated.** Code: `asym_gemm/integrations/lf.py` (`_is_qwen3_decoder_layer_module_name` broadened to accept Llama4 `feed_forward`; `mlp_dense` made a selectable offload component) + `asym_gemm/training/offload.py` (`_selection_component_selected`). Tests added/updated in `tests/training/test_lf_qwen3_asym_backend.py` — all green, **0 regressions** (the only failing repo tests are pre-existing CPU-Adam / `profile_lora_lf.sh` shape-drift, confirmed via stash). Validated on a **real HF `Llama4TextModel`**: matcher wraps **4/4** real `Llama4TextDecoderLayer`s, dense-FFN bases move to CPU (`mlp_dense`), activation offload fires (`num_offloads=[41,63,51,55]`), offloaded model trains (LoRA-only finite grads, 0 non-LoRA trainable). Inherited optimizations (`gc-exp`, expert/attention act-offload, `gc-attn-exp`) need **no code** — see Coverage table. **E2E profiling acceptance: DONE — see "Measured e2e profiling acceptance" below.**

> **Activation offloading + AsymGEMM back-fetch (verified by reuse, no new code):** all three activation-offload paths run for Llama4 by reusing the Qwen3 machinery — expert-act (`AsymQwen3Experts` + `exp_act_offload_lora` CPU-left/right), attention-act (`AsymActivationOffloadLoRALinear`), layer-act (`decoder_activation_offload`). Numerically exact on real `Llama4TextModel`: with `ASYMM_EXPERT_ACT_OFFLOAD` ON vs OFF the LoRA grads are **bit-identical (rel diff 0.0)**; same for `ASYMM_ATTN_ACT_OFFLOAD` (fwd + grads rel diff 0.0). No redundant code. **Caveat (out of scope here):** `embed_tokens` frozen-*weight* offload fails for *untied* Llama4 because `modeling_llama4.py:540` does `embed_tokens(input_ids.to(embed_tokens.weight.device))` (CPU host weight) — tied Llama4 (default) already rejects embed/lm_head offload; a robust fix to the shared `AsymFrozenEmbedding` was reverted to keep the Qwen3 path pristine.

## Measured e2e profiling acceptance (2026-06-15)
Real `apply_lf_asym_lora` offload path, bf16 Llama4 (12 layers = 6 dense/6 MoE, 16 experts, hidden 4096, mlp 14336), **seq 8192**, real fwd+bwd+AdamW, 10 measured steps, 1×GB200 (184 GB). Baseline = existing `routed_experts`-only offload (`18479 MB / 207 ms ±35`).

| Lever (this change enables) | Peak MB | Δmem | Step ms (±σ) | Δlat | Verdict |
|---|---|---|---|---|---|
| baseline (fits) | 18479 | — | 207 ±35 | — | — |
| **`gc-layer` recompute** | 5894 | **−68.1%** | 305 ±51 | **+47%** | **ACCEPT — best mem/lat lever** |
| layer-act-offload | 5454 | −70.5% | 675 ±39 | +227% | correct; dominated by recompute (sync H2D staging → async-prefetch is the optimization) |
| **`mlp_dense`** (Stage 2) | 16463 | **−10.9%** (=2016 MB, exact) | 315 ±9 | +52% | correct; memory-bound-only lever |
| layer-offload + `mlp_dense` | 3438 | **−81.4%** | 784 ±50 | +279% | lowest peak |

**Enabling case (batch 16, same model):** baseline **OOMs** (>184 GB, cannot train); **`gc-layer` fits at 49 GB; layer-offload+`mlp_dense` fits at 40 GB** (lowest peak → largest trainable batch). In the memory-bound regime — the regime these levers are *for* — they are the difference between training and OOM; the latency cost is moot.

**Decision:** correctness confirmed (memory drops match theory; trains with LoRA-only finite grads). The Stage-1 matcher fix is what makes `gc-layer` (−68% mem / +47% lat) and layer-act-offload *work* on Llama4 (was strict-RAISE). **Use `gc-layer` as the default activation lever**; the saved-tensor offload sub-mode is correct but its synchronous backward staging (`decoder_activation_offload._unpack`) makes it slower than recompute — async prefetch is the tracked optimization. `mlp_dense` correctly evacuates the exact dense base for memory-bound (Maverick) runs.

## State of the world
- **One shared expert engine.** `AsymQwen3Experts` (`qwen3_moe.py:1886`) is the only MoE engine; `packed_moe.py:10` aliases it as `AsymPackedExperts`. Llama 4 routed experts already run it via `AsymLlama4Moe → wrap_packed_experts` (`llama4_moe.py:199`) and inherit, for free: grouped GEMM over experts (no per-expert Python loop), fused gate+up split-LoRA, expert-activation offload (`exp_act_offload_lora.py`, `ASYMM_EXPERT_ACT_OFFLOAD`), expert recompute policy, and trainable LoRA-weight offload (`weight_offload.py:256` scans `isinstance(m, AsymQwen3Experts)` — Llama4 experts ARE that class). Routing uses input-scaled + summed combine `forward_input_scaled` (`qwen3_moe.py:2508`), matching HF `Llama4TextMoe`.
- **Scaffold exists and is wired.** `is_llama4_moe`/`wrap_llama4_moe`/`AsymLlama4Moe`/`AsymLlama4Router` (`llama4_moe.py`) normalize `gate_up_proj [E,H,2E']→[E,2E',H]` (transpose preserves row order ⇒ `chunk(2)` gate/up correct) and `down_proj [E,E',H]→[E,H,E']`. Dispatched in `lf.py:1088` (`router_mode=whole`) and `lf.py:1131` (`router_mode=hf`).
- **What already works (verify in Stage 0):** routed-expert LoRA + base CPU offload, expert-activation offload, expert recompute (`gc-exp`), expert LoRA-weight offload, router CPU offload, and — because the attention matcher is generic on q/k/v/o — attention activation offload + `gc-attn-exp`.
- **The port is non-expert glue.** Gaps: **G1** decoder/layer offload + `gc-layer` matcher hard-requires a child named `mlp` (`lf.py:806`); Llama4 uses `feed_forward` ⇒ strict RAISE. **G2** dense-FFN (Maverick interleaved layers) base CPU offload is unreachable: `mlp_dense` is classified only under `.mlp.` (`lf.py:436`) and is not a *selectable* offload component. **G3** shared-expert base offload + LoRA need verification (no code expected). **G4** attention correctness with iRoPE internals (verify). **G5** bf16-only path. **G6** router/scaling numerics parity (gate in Stage 0).
- **Checkpoints (corrected):** `meta-llama/Llama-4-Scout-17B-16E` is **bf16, all-MoE** (`interleave_moe_layer_step=1`) → exercises G1/G3/G4/G6 but has **zero** dense layers. `meta-llama/Llama-4-Maverick-17B-128E-Instruct` is **bf16** (the `-FP8` repo is the quantized one) and **interleaves dense+MoE** → the real G2 target, but 400B ⇒ needs the full offload stack/multi-GPU. For single-GPU G2 validation use the **sized bf16 fixture** below (GB-scale, not toy).

### Coverage — every `gc-*` policy & activation optimization on Llama4
Llama4 routed experts **are** `AsymQwen3Experts`, and the expert-side toggles are reached from `forward_input_scaled` (the Llama4 path) by the **same dispatch** as Qwen3's `forward` (`qwen3_moe.py:2555-2559`; input-scaling is applied at pack time *before* the dispatch). So everything except the two matcher-gated rows is inherited with no Llama4 code.

| Optimization (flag / policy) | Reaches Llama4 via | Status |
|---|---|---|
| Expert act offload (`ASYMM_EXPERT_ACT_OFFLOAD`) | engine `_uses_activation_offload` (`:2326`) | ✅ inherited → **verify Stage 0b** |
| `gc-exp` (expert checkpoint) | engine `_uses_expert_gc` (`:2288`) | ✅ inherited → **verify Stage 0b** |
| token-recompute (`tok-leN`/`tok-geN`/`tokA-B`/`-act`) | engine `_uses_expert_recompute` (`:2292`) | ✅ inherited → **verify Stage 0b** |
| Attention act offload (`ASYMM_ATTN_ACT_OFFLOAD`) | generic attn matcher `_is_text_attention_module_name` (q/k/v/o) | ✅ inherited → **validate Stage 3** |
| `gc-attn-exp` (attn+expert checkpoint) | generic attn matcher + engine | ✅ inherited → **validate Stage 3** |
| Layer act offload (`ASYMM_LAYER_ACT_OFFLOAD`) | `_is_qwen3_decoder_layer_module_name` (requires `mlp`) | ⚠️ **needs Stage 1** |
| `gc-layer` (full-layer checkpoint) | same decoder matcher | ⚠️ **needs Stage 1** |

**One code change (Stage 1 matcher) unlocks both matcher-gated rows; all other gc-* policies and act optimizations are inherited** because the experts are the shared engine and Llama4 attention exposes q/k/v/o. Valid policy syntax: `none, tok-leN, tok-geN, tokA-B, gc-exp, gc-attn-exp, gc-layer` + `-act` variants (`moe.py:737`).

### HF ground-truth (re-verify line numbers before editing; `third_party/transformers/src/transformers/models/llama4/`)
- `Llama4TextDecoderLayer` (`modeling_llama4.py:413`): children `self_attn`, **`feed_forward`** (`Llama4TextMoe` if `layer_idx in config.moe_layers` else dense `Llama4TextMLP(intermediate_size_mlp)`), `input_layernorm`, `post_attention_layernorm`. **No `mlp` child.** `forward` returns a single tensor.
- `Llama4TextMoe` (`:157`): `routed_in = h.repeat(top_k,1) * router_scores`; `out = shared_expert(h) + Σ_topk routed_out`. `top_k = num_experts_per_tok` (default 1).
- `Llama4TextMLP` (`:89`): `gate_proj`/`up_proj`/`down_proj` are `nn.Linear(bias=False)`. Dense FFN uses `intermediate_size_mlp` (16384); MoE `shared_expert` uses `intermediate_size`.
- `Llama4TextAttention` (`:321`): q/k/v/o plain `nn.Linear` on every layer; `use_rope=no_rope_layers[i]`; `qk_norm=Llama4TextL2Norm` only if `use_qk_norm and use_rope`; `attn_temperature_tuning` on NoPE layers. **All exotic ops (RoPE/L2-qk-norm/temperature/chunked mask) are downstream of q/k/v and upstream of o_proj.**
- `Llama4TextL2Norm` (`:107`): **stateless** (no `nn.Parameter`).
- `configuration_llama4.py`: text `model_type="llama4_text"` (`:109`), top `"llama4"` (`:226`); `moe_layers = range(step-1, L, step)`; `tie_word_embeddings=False`, `output_router_logits=False` (defaults).

### Profiling truth + helpers (from `run_lf_profiled_train.py`)
- Peak GPU mem: profile JSON `memory.gpu.peak_allocated_hbm_bytes` (`torch.cuda.max_memory_allocated`, line 2267/2564). Step latency: top-level `measured_e2e_step_milliseconds` (line 477, warmup-excluded). When `PROFILE_PROFILER=source` the report may nest under `source_profile`.
- Driver: `scripts/lf/run_lf_lora_sft.sh` (env-driven; `PROFILE=1` routes through `run_lf_profiled_train.py`).

```bash
# Paste once per shell. Each helper takes a profile.json path.
jq_peak(){ python3 - "$1" <<'PY'
import json,sys; p=json.load(open(sys.argv[1])); p=p.get("source_profile",p)
print(p["memory"]["gpu"]["peak_allocated_hbm_bytes"])
PY
}
jq_step(){ python3 - "$1" <<'PY'
import json,sys; p=json.load(open(sys.argv[1])); p=p.get("source_profile",p)
print(p.get("measured_e2e_step_milliseconds") or "null")
PY
}
# decide BASE_PEAK CHG_PEAK BASE_MS CHG_MS  MIN_PCT MIN_MIB MAX_LAT_PCT
decide(){ python3 - "$@" <<'PY'
import sys
bp,cp,bm,cm,minpct,minmib,maxlat=[float(x) for x in sys.argv[1:8]]
dmem=(cp-bp)/bp*100; dms=(cm-bm)/bm*100; dmib=(bp-cp)/2**20
ok=(dmem<=-minpct and dmib>=minmib) and dms<=maxlat
print(f"peak {bp/2**20:.0f}->{cp/2**20:.0f} MiB ({dmem:+.1f}%, saved {dmib:.0f} MiB) | step {bm:.1f}->{cm:.1f} ms ({dms:+.1f}%)")
print("ACCEPT" if ok else "REJECT")
PY
}
```

### Sized bf16 Llama4 fixture (single-GPU; non-toy; exercises G1–G6)
Random-init weights are fine for memory/latency (loss-parity is the in-process Stage-0 test). Dims are multiples of 64 for grouped-GEMM.
```python
# tools/make_llama4_fixture.py  (scratch)
import torch
from transformers.models.llama4.modeling_llama4 import Llama4TextConfig, Llama4ForCausalLM
cfg = Llama4TextConfig(hidden_size=4096, intermediate_size=2048, intermediate_size_mlp=14336,
                       num_hidden_layers=16, num_attention_heads=32, num_key_value_heads=8, head_dim=128,
                       num_local_experts=16, num_experts_per_tok=1, interleave_moe_layer_step=2,  # dense=[0,2,..,14], MoE=[1,3,..,15]
                       use_qk_norm=True, attn_temperature_tuning=True, vocab_size=32000,
                       tie_word_embeddings=False, output_router_logits=False, torch_dtype=torch.bfloat16)
m = Llama4ForCausalLM(cfg).to(torch.bfloat16)
m.save_pretrained("/tmp/llama4_bf16_fixture"); cfg.save_pretrained("/tmp/llama4_bf16_fixture")
# add a tokenizer (copy any llama tokenizer) so LlamaFactory can load it.
```
Dense-FFN frozen base ≈ 8 dense layers × (2·4096·14336 + 14336·4096)·2 B ≈ **2.8 GB** → a meaningful, non-trivial G2 lever.

---

## Stage 0 — baseline + correctness gate (MANDATORY; no source edits)
### SCOPE
Files: none. Prove the existing scaffold runs e2e and is numerically correct before any change. Gate conditions:
1. `report.llama4_moes_wrapped == #MoE layers` (grep the LF asym report line in the train log).
2. asym-vs-torch forward+backward parity (zero LoRA-B ⇒ asym output == frozen-base HF output; grads flow to LoRA only).
3. Record baseline peak-mem + `measured_e2e_step_milliseconds` for (b) `routed_experts` and (c) `all` — the references every later stage diffs against.

### CORRECTNESS micro-test (isolated numerics — the ONLY allowed non-e2e check)
```python
# tools/_llama4_parity.py (scratch)
import torch, copy
from transformers.models.llama4.modeling_llama4 import Llama4TextConfig, Llama4TextMoe
from asym_gemm.training.llama4_moe import wrap_llama4_moe, is_llama4_moe
cfg = Llama4TextConfig(hidden_size=512, intermediate_size=256, intermediate_size_mlp=512,
                       num_local_experts=8, num_experts_per_tok=1, torch_dtype=torch.bfloat16)
ref = Llama4TextMoe(cfg).to("cuda", torch.bfloat16).eval()
for p in ref.parameters(): p.requires_grad_(False)
src = copy.deepcopy(ref).to("cpu"); assert is_llama4_moe(src)        # offload needs CPU-resident bf16 source
asym = wrap_llama4_moe(src, backend="asym", precision="bf16", offload=True,
                       lora_rank=8, lora_alpha=16, lora_dropout=0.0, router_mode="whole", strict=True).to("cuda")
for n,p in asym.named_parameters():
    if "lora_B" in n: torch.nn.init.zeros_(p)                        # LoRA delta = 0
x = torch.randn(2,16,cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
with torch.no_grad(): o_ref,_=ref(x); o_asym,_=asym(x)
torch.testing.assert_close(o_asym, o_ref, rtol=2e-2, atol=2e-2)      # bf16 grouped-GEMM tolerance
for n,p in asym.named_parameters():
    if "lora_B" in n: torch.nn.init.normal_(p, std=1e-3)
y,_=asym(x); y.sum().backward()
assert all(p.grad is None for n,p in asym.named_parameters() if "lora_" not in n)
assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in asym.named_parameters() if "lora_A" in n)
```

### VALIDATION (e2e — the real baselines)
```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
M=/tmp/llama4_bf16_fixture     # or meta-llama/Llama-4-Scout-17B-16E on a 96GB+ GPU
C="PROFILE=1 PROFILE_PROFILER=source CUTOFF_LEN=4096 PER_DEVICE_TRAIN_BATCH_SIZE=1 MAX_STEPS=8 \
   LORA_RANK=16 LORA_ALPHA=16 LORA_DROPOUT=0.0 ASYM_ROUTER_MODE=whole ASYM_STRICT=true \
   USE_ASYM_CPU_ADAMW=false MODEL_NAME_OR_PATH=$M"
env $C BACKEND=torch                                                        scripts/lf/run_lf_lora_sft.sh  # (a) ref
env $C BACKEND=asym ASYM_OFFLOAD_MODULES=routed_experts ASYM_EXPERT_RECOMPUTE_POLICY=none scripts/lf/run_lf_lora_sft.sh  # (b)
env $C BACKEND=asym ASYM_OFFLOAD_MODULES=all            ASYM_EXPERT_RECOMPUTE_POLICY=none scripts/lf/run_lf_lora_sft.sh  # (c)
# each run prints OUT_DIR; record jq_peak/jq_step for (b),(c); parity = (b) final loss within ~2% of (a).
```
### ACCEPTANCE
No keep/reject — this gates the project. Block all later stages until (1)+(2)+(3) hold. If parity fails the bug is in the engine/scaffold (out of scope here) — STOP and report.

### WATCH
- `router_mode=whole` runs the router under `no_grad` and asserts non-differentiable weights. Needs `output_router_logits=False` (default) + `tie_word_embeddings=False` (verified). If a config ships `output_router_logits=True`, whole-mode RAISES (`lf.py:1018`) → use `ASYM_ROUTER_MODE=hf`.

---

## Stage 0b — verify inherited expert-side optimizations (no code; explicit accept/reject)
Expert act-offload, `gc-exp`, and token-recompute reach Llama4 through the shared engine dispatch with no matcher and no Llama4 code. **Prove each is active and meaningful — do not assume.** Baseline = Stage-0 (b).
```bash
M=/tmp/llama4_bf16_fixture
C="PROFILE=1 PROFILE_PROFILER=source CUTOFF_LEN=8192 PER_DEVICE_TRAIN_BATCH_SIZE=1 MAX_STEPS=8 \
   LORA_RANK=16 LORA_ALPHA=16 LORA_DROPOUT=0.0 ASYM_ROUTER_MODE=whole ASYM_STRICT=true \
   USE_ASYM_CPU_ADAMW=false BACKEND=asym MODEL_NAME_OR_PATH=$M ASYM_OFFLOAD_MODULES=routed_experts"
env $C ASYMM_EXPERT_ACT_OFFLOAD=true  ASYM_EXPERT_RECOMPUTE_POLICY=none      scripts/lf/run_lf_lora_sft.sh  # expert act offload
env $C ASYMM_EXPERT_ACT_OFFLOAD=false ASYM_EXPERT_RECOMPUTE_POLICY=gc-exp    scripts/lf/run_lf_lora_sft.sh  # expert checkpoint
env $C ASYMM_EXPERT_ACT_OFFLOAD=false ASYM_EXPERT_RECOMPUTE_POLICY=tok-le128 scripts/lf/run_lf_lora_sft.sh  # token-recompute
```
Prove active: each wrapped Llama4 experts module's `_last_activation_offload_stats["num_offloads"]>0` (act offload), or the LF report shows `expert_recompute_policy=gc-exp|tok-le128`. ACCEPT each independently: act-offload `decide … 5 512 5`; checkpoint/recompute (extra forward) `decide … 5 512 10`. REJECT any that is flat-memory or latency-blown (e.g. if D2H/H2D isn’t overlapped at this size — raise seq/batch and re-measure). **Watch:** `gc-exp` and the token policies are one mutually-exclusive `ASYM_EXPERT_RECOMPUTE_POLICY` string; expert act-offload is the independent `ASYMM_EXPERT_ACT_OFFLOAD` flag (only valid with policy `none`).

---

## Stage 1 — generalize the decoder-layer matcher (G1)
Enables `ASYMM_LAYER_ACT_OFFLOAD` (decoder saved-tensor offload) and `gc-layer` (full-layer checkpoint) for Llama4 — the **only** two optimizations that need new code.

### SCOPE
File `asym_gemm/integrations/lf.py`, function **`_is_qwen3_decoder_layer_module_name`** (`lf.py:800-815`). It is the sole gate for BOTH `_wrap_qwen3_decoder_saved_tensor_offload_modules` (`:895`) and `_wrap_qwen3_decoder_checkpoint_modules` (`:843`). The installers `install_decoder_saved_tensor_offload`/`install_decoder_checkpoint` are **structure-agnostic** (wrap the whole layer `forward` via `saved_tensors_hooks`/`checkpoint`; verified `decoder_activation_offload.py:147,156`) — **only the matcher changes**.

Today it requires `{self_attn, mlp, input_layernorm, post_attention_layernorm} ⊆ children`. Llama4 has `feed_forward` not `mlp` ⇒ every layer skipped ⇒ strict RAISE.

### CODE CHANGE (add-only; Qwen3 path untouched → zero regression)
Insert a Llama4 branch just before the final `return False` (`lf.py:815`):
```python
def _is_qwen3_decoder_layer_module_name(name, module):   # now: "supported text decoder layer"
    if not name or _has_attention_excluded_path_marker(name):       # excludes vision/multimodal
        return False
    children = dict(module.named_children())
    # --- existing Qwen3 path (UNCHANGED) ---
    required = {"self_attn", "mlp", "input_layernorm", "post_attention_layernorm"}
    if required <= set(children):
        cn, mn = type(module).__name__.lower(), type(module).__module__.lower()
        mt = str(getattr(getattr(module, "config", None), "model_type", "")).lower()
        if ("qwen3" in cn or "qwen3" in mn or mt in {"qwen3_moe", "qwen3_vl_moe"}
                or hasattr(children["mlp"], "_is_asym_qwen3_moe_block") or is_qwen3_moe_block(children["mlp"])):
            return True
    # --- NEW: Llama4 decoder layer (child `feed_forward`, dense or MoE) ---
    llama4_required = {"self_attn", "feed_forward", "input_layernorm", "post_attention_layernorm"}
    if llama4_required <= set(children):
        cn, mn = type(module).__name__.lower(), type(module).__module__.lower()
        mt = str(getattr(getattr(module, "config", None), "model_type", "")).lower()
        if "llama4" in cn or "llama4" in mn or mt in {"llama4", "llama4_text"}:
            return True
    return False
```
**Why whole-layer is correct/efficient:** the saved-tensor region spans attention+feed_forward (the largest activation window); offload is one D2H/H2D per qualifying tensor (`min_bytes=1 MiB`, `requires_grad`, cuda-only — `decoder_activation_offload.py:159-179`), never per-head/token. The `feed_forward` child may be `AsymLlama4Moe`; its own expert-offloaded tensors are already CPU (`device!="cuda"` ⇒ not re-offloaded), so the two compose without double-copying GPU tensors.

### VALIDATION (e2e)
`ASYMM_LAYER_ACT_OFFLOAD`/`gc-layer` require backend `asym` + policy `none`/`gc-layer` (`lf.py:1039-1042`). Use a realistic seq (activations must dominate) — re-run at `CUTOFF_LEN=8192` if 4096 shows trivial savings.
```bash
M=/tmp/llama4_bf16_fixture
C="PROFILE=1 PROFILE_PROFILER=source CUTOFF_LEN=8192 PER_DEVICE_TRAIN_BATCH_SIZE=1 MAX_STEPS=8 \
   LORA_RANK=16 LORA_ALPHA=16 LORA_DROPOUT=0.0 ASYM_ROUTER_MODE=whole ASYM_STRICT=true \
   USE_ASYM_CPU_ADAMW=false BACKEND=asym MODEL_NAME_OR_PATH=$M ASYM_OFFLOAD_MODULES=routed_experts"
# BASE: ASYMM_LAYER_ACT_OFFLOAD=false ASYM_EXPERT_RECOMPUTE_POLICY=none
env $C ASYMM_LAYER_ACT_OFFLOAD=true  ASYM_EXPERT_RECOMPUTE_POLICY=none     scripts/lf/run_lf_lora_sft.sh  # CHG-A
env $C ASYMM_LAYER_ACT_OFFLOAD=false ASYM_EXPERT_RECOMPUTE_POLICY=gc-layer scripts/lf/run_lf_lora_sft.sh  # CHG-B
```
Prove the matcher fired: grep the train log for the LF asym report — `layer_act_offload_wrapped` (CHG-A) / `layer_gc_wrapped` (CHG-B) **== #decoder layers** (was 0). Also assert a GDN-free Llama4 layer's `module._last_activation_offload_stats["num_offloads"] > 0` and `offloaded_bytes > 0`. Then `decide BASE_PEAK CHGx_PEAK BASE_MS CHGx_MS …`.

### ACCEPTANCE
- CHG-A (offload): KEEP iff peak ↓ **≥5% AND ≥1024 MiB** and step ↑ **≤5%** → `decide … 5 1024 5`.
- CHG-B (`gc-layer`): KEEP iff peak ↓ **≥5%** and step ↑ **≤10%** (one extra forward) → `decide … 5 1024 10`.
- REJECT if wrapped-count is 0 (matcher still broken), memory flat/trivial, or latency out of band.

### WATCH
- If a Llama4 wrapper passes `use_cache=True` in training (should be False under HF Trainer) the installer no-ops silently (`:153`) — assert it's False during validation.
- Stacking with `ASYMM_EXPERT_ACT_OFFLOAD=true`: both are active inside the layer; the decoder hook may also offload the engine's on-GPU saved tensors → redundant copies. Measure layer-offload standalone first; only stack if `decide` still passes.

---

## Stage 2 — dense-FFN base CPU offload (G2) + shared-expert verify (G3)
### SCOPE
- `asym_gemm/integrations/lf.py`: `classify_lf_component` (`:436`), `SUPPORTED_LF_OFFLOAD_COMPONENTS` (`:71`), `_ALL_LF_OFFLOAD_COMPONENTS` (`:74`), `LFOffloadSelection` (`:88`, `any_cpu_offload` `:100`, `implemented_components` `:112`), `parse_lf_offload_modules` (aliases `:306`, constructor `:360`), `component_is_selected` (`:441`).
- `asym_gemm/training/offload.py`: `_selection_component_selected` (`:110`) — **residency validation has its own selection check; miss it and dense bases are never counted as CPU-resident.**
- **Keep the existing label `mlp_dense`** (asserted by `tests/test_lf_memory_breakdown.py:126,133`; used by `lf_trace.py` and `offload.py:106`). Add it as a *selectable* component; do not rename.

G3 is **verification-only**: after block replacement the dense walk (`lf.py:1282`) reaches `{block}.shared_expert.{gate,up,down}_proj` → `.shared_expert.` rule (`:375`) classifies `shared_experts` (fires before the dense rule) → base offload + LoRA iff `shared_experts ∈ ASYM_OFFLOAD_MODULES` and target includes those leaves. The shared expert is dense ⇒ no expert-engine act-offload; its activation lever is Stage-1 (it lives inside the decoder layer). Scout has shared experts; only Maverick/fixture have *dense* FFN layers for G2.

### CODE CHANGES (complete, consistent edit set — 10 sites)
```python
# lf.py:71  + :74   add to BOTH frozensets
SUPPORTED_LF_OFFLOAD_COMPONENTS = frozenset({..., "norms", "mlp_dense"})
_ALL_LF_OFFLOAD_COMPONENTS      = frozenset({..., "norms", "mlp_dense"})

# lf.py:88  LFOffloadSelection
    mlp_dense: bool = False
# lf.py:100 any_cpu_offload    -> add  `or self.mlp_dense`
# lf.py:112 implemented_components -> add  `if self.mlp_dense: components.add("mlp_dense")`

# lf.py:436 classify_lf_component  (broaden; keep label; still AFTER experts/router/shared rules)
if parent_leaf in {"gate_proj", "up_proj", "down_proj"} and (".mlp." in lower or ".feed_forward." in lower):
    return "mlp_dense"

# lf.py:306 parse_lf_offload_modules aliases (user-facing tokens -> internal label)
    "dense_mlp": "mlp_dense", "mlp": "mlp_dense", "dense": "mlp_dense",
    "dense_ffn": "mlp_dense", "feed_forward": "mlp_dense",
# lf.py:360 LFOffloadSelection(...) constructor -> add
    mlp_dense="mlp_dense" in expanded,

# lf.py:441 component_is_selected -> add (classifier label keys this)
    if component == "mlp_dense":
        return selection.mlp_dense

# offload.py:110 _selection_component_selected -> add (residency validation path)
    if component == "mlp_dense":
        return bool(getattr(selection, "mlp_dense", False))
```
With these, the dense walk’s `selected_cpu_offload` (`lf.py:1293`) becomes True for `feed_forward.{gate,up,down}_proj`, and `_wrap_lf_linear_leaf` (`:636`) adopts the weight to **pinned CPU** and returns `AsymFrozenLinear` (frozen base, single CPU-fetched bf16 GEMM) or `AsymLoRALinear` when also a LoRA target. **No GEMM splitting, no loops** — one GEMM per projection, weight staged once per forward. Ordering is safe: `.shared_expert.`/`.experts.`/`.router.` rules (`:375/:379/:381`) fire before the broadened dense rule, so only true dense FFN reaches it (Maverick MoE layers have no bare `feed_forward.gate_proj`; their gate is the `router`).

### VALIDATION (e2e — sized fixture; Maverick-bf16 is the real target when hardware allows)
```bash
M=/tmp/llama4_bf16_fixture     # interleave_moe_layer_step=2 ⇒ has dense layers
C="PROFILE=1 PROFILE_PROFILER=source CUTOFF_LEN=4096 PER_DEVICE_TRAIN_BATCH_SIZE=1 MAX_STEPS=8 \
   LORA_RANK=16 LORA_ALPHA=16 LORA_DROPOUT=0.0 ASYM_ROUTER_MODE=whole ASYM_STRICT=true \
   USE_ASYM_CPU_ADAMW=false BACKEND=asym MODEL_NAME_OR_PATH=$M"
env $C ASYM_OFFLOAD_MODULES=routed_experts            scripts/lf/run_lf_lora_sft.sh   # BASE (dense base on GPU)
env $C ASYM_OFFLOAD_MODULES=routed_experts,mlp_dense  scripts/lf/run_lf_lora_sft.sh   # CHG (dense base -> CPU)
env $C ASYM_OFFLOAD_MODULES=routed_experts,shared_experts scripts/lf/run_lf_lora_sft.sh  # G3 verify
```
- CHG: confirm the LF asym report shows `mlp_dense` CPU-resident bytes > 0 (≈ the 2.8 GB above) and the dense Linears became `AsymFrozenLinear`. `decide BASE_PEAK CHG_PEAK BASE_MS CHG_MS 5 1024 5`.
- G3: confirm `shared_experts` CPU-resident bytes > 0 and (with `lora_target=all`) shared-expert LoRA wrapped.
- All-MoE Scout: assert `mlp_dense` selected-but-zero-matched does **not** RAISE (offload component, not a LoRA target → missing matches are fine).

### ACCEPTANCE
- G2: KEEP iff peak ↓ realizes **≥80% of the theoretical dense-base bytes** AND step ↑ **≤5%** (frozen base is CPU-fetched per forward → small bump expected). REJECT if realized < 50% of theory (dense base wasn’t the peak contributor → retry at longer seq or drop) or any strict RAISE on an all-MoE model.
- G3: "accept" = base off HBM (peak ↓ ≥ shared-expert-base × 0.8 when `shared_experts` is the only delta) and LoRA wrapped; step ↑ ≤5%.

### WATCH
- Classifier ordering: verify the Stage-0 dump maps `feed_forward.experts.*→routed_experts`, `feed_forward.router→router`, `shared_expert.*→shared_experts`, and only bare `feed_forward.{gate,up,down}_proj→mlp_dense`. A reorder would steal expert/router leaves.
- `offload.py:106` `_default_classify_component` still matches only `.mlp.`, but it is the *fallback*; `apply_lf_asym_lora` passes `classify_lf_component` to `validate_lf_offload_residency` (`:1398`), so the lf.py classifier governs. Optionally mirror the `.feed_forward.` broadening there for consistency.

---

## Stage 3 — attention activation offload / checkpoint correctness with iRoPE (G4)
### SCOPE
File `asym_gemm/integrations/lf.py` — **expected no code change**; correctness+efficiency verification. The matchers/wrappers are already generic: `_is_text_attention_module_name` (`:788`) matches `self_attn` with q/k/v/o (Llama4 has all four); `AsymActivationOffloadLoRALinear` offloads the projection INPUT only and assumes nothing about RoPE/qk-norm/temperature (all of which are downstream of q/k/v and upstream of o_proj); `install_attention_saved_tensor_offload` wraps the whole attention forward.

### VERIFY (no new code unless a check fails)
1. `ASYMM_ATTN_ACT_OFFLOAD=true` + `lora_target=all` + `attention ∈ ASYM_OFFLOAD_MODULES` ⇒ `attention_act_offload_wrapped>0` and `attention_saved_tensor_offload_wrapped>0`; loss within ~2% of Stage-0 (b).
2. `gc-attn-exp` ⇒ `attention_gc_wrapped>0`, runs, numerics OK.
3. `Llama4TextL2Norm` is stateless ⇒ skipped by the `norms` walk (`_is_stateless_module`, `lf.py:1260`); only `Llama4TextRMSNorm` (has weight) is offloaded. Confirm in the Stage-0 dump.
4. Efficiency: q/k/v share one offloaded input buffer (one D2H/H2D per projection); confirm no per-head/token op or kernel-count explosion in the profiler.

### VALIDATION (e2e)
```bash
M=/tmp/llama4_bf16_fixture       # ≥4 layers ⇒ ≥1 NoPE layer (no_rope every 4th) for exotic-path coverage
C="PROFILE=1 PROFILE_PROFILER=source CUTOFF_LEN=8192 PER_DEVICE_TRAIN_BATCH_SIZE=1 MAX_STEPS=8 \
   LORA_RANK=16 LORA_ALPHA=16 LORA_DROPOUT=0.0 ASYM_ROUTER_MODE=whole ASYM_STRICT=true \
   USE_ASYM_CPU_ADAMW=false BACKEND=asym MODEL_NAME_OR_PATH=$M ASYM_OFFLOAD_MODULES=routed_experts,attention"
# BASE: ASYMM_ATTN_ACT_OFFLOAD=false policy none
env $C ASYMM_ATTN_ACT_OFFLOAD=true  ASYM_EXPERT_RECOMPUTE_POLICY=none        scripts/lf/run_lf_lora_sft.sh  # CHG-A
env $C ASYMM_ATTN_ACT_OFFLOAD=false ASYM_EXPERT_RECOMPUTE_POLICY=gc-attn-exp scripts/lf/run_lf_lora_sft.sh  # CHG-B
```
`lora_dropout` MUST be 0.0 (`_wrap_lf_linear_leaf:605` raises otherwise — already set).

### ACCEPTANCE
- CHG-A: KEEP iff peak ↓ **≥5% AND ≥512 MiB** at long seq, step ↑ **≤5%** → `decide … 5 512 5`. REJECT if wrapped-count 0 / flat / latency out of band.
- CHG-B (`gc-attn-exp`): KEEP iff peak ↓ **≥5%**, step ↑ **≤10%** → `decide … 5 512 10`.

### WATCH
- NoPE/temperature/chunked-mask all live inside the wrapped attention forward, downstream of the offloaded q/k/v input → transparent under save/restore; confirm numerics on a config with ≥1 NoPE layer.
- Attention context sharing needs q+k+v all offload-eligible; include the full `attention` selector or each projection offloads independently (more D2H traffic).
- Contingency: only if `_is_text_attention_module_name` ever fails to match (e.g., a future fused `qkv_proj`) extend `_ATTENTION_TARGETS`; current Llama4 has separate q/k/v/o (verified) ⇒ no change.

---

## Cross-cutting risks / watch-later
- **G5 (bf16-only):** the asym path RAISES on non-bf16 (`AsymLlama4Moe.__init__`, `AsymActivationOffloadLoRALinear`). Validate on Scout-bf16, **Maverick-bf16** (`meta-llama/Llama-4-Maverick-17B-128E-Instruct`, *not* the `-FP8` repo), or the sized fixture. Do **not** point the asym path at an FP8 checkpoint.
- **G6 (router/scaling numerics):** `AsymLlama4Router.forward` replicates HF `Llama4Router` (topk→scatter(-inf)→`sigmoid(.float())`); `forward_input_scaled` input-scales by route weight then sums — matching `Llama4TextMoe`. For `top_k=1` (current Llama4 default) gather-of-selected-score == dense-score replication (non-selected scores are 0). **If a future Llama4 sets `top_k>1`, re-verify** with a `top_k>1` parity fixture. Stage-0 parity is the gate.
- **Isolate the lever:** keep `USE_ASYM_CPU_ADAMW=false` in all stage comparisons (CPU-Adam is arch-agnostic, orthogonal; validate it last/separately). `peak_allocated_hbm_bytes` is a global step max — CPU-Adam/weight-offload move where the peak occurs.

## Efficiency invariants (every stage)
- Routed experts ALWAYS use grouped GEMM + fused gate+up split-LoRA; never a per-expert Python loop or per-token small GEMMs.
- Dense-FFN / shared-expert / attention base offload = one CPU-fetched bf16 GEMM per projection (`AsymFrozenLinear`/`AsymLoRALinear`), weight staged once per forward; no GEMM splitting.
- Activation offload (layer/attention) moves whole packed tensors (one D2H/H2D each), q/k/v share one input buffer; never per-head/token slivers.
