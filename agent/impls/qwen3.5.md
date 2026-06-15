# Qwen3.5 (`qwen3_5_moe`) LoRA-SFT Memory-Offload Port — Staged Implementation Plan

## State of the world (read first)

- **One shared expert engine, reused as-is.** `AsymQwen3Experts` (`asym_gemm/training/qwen3_moe.py:1886`) is the only MoE engine. `is_qwen3_experts` (`qwen3_moe.py:86`) matches HF `Qwen3_5MoeExperts` exactly (3D `gate_up_proj`/`down_proj`, int `num_experts/hidden_dim/intermediate_dim`, callable `act_fn`, forward `(hidden_states, top_k_index, top_k_weights)`). VERIFIED at runtime: `is_qwen35_moe_block(layers.N.mlp) == True` on a real tiny `Qwen3_5MoeTextModel`, for BOTH linear-attention and full-attention layers. So routed experts inherit split-LoRA (fused gate+up LoRA-A), grouped GEMM (replacing the native per-`expert_hit` python loop in `Qwen3_5MoeExperts.forward`), `ASYMM_EXPERT_ACT_OFFLOAD`, recompute policy, and LoRA weight-offload (`weight_offload.py`) for free.
- **MoE-block scaffold already complete AND tested.** `AsymQwen35MoeBlock`/`is_qwen35_moe_block`/`wrap_qwen35_moe_block` (`asym_gemm/training/qwen35_moe.py`) are wired into `apply_lf_asym_lora` (`lf.py:1048`, FIRST so it beats the qwen3 branch). `tests/training/test_lf_qwen35_asym_backend.py` already proves: matcher accept/reject, whole-block forward equals HF `Qwen3_5MoeSparseMoeBlock` (incl. real transformers block), router detach, experts CPU-adopt-without-clone, shared_expert + shared_expert_gate LoRA via the dense walk, router offload, shifted+gated RMSNorm offload, and adapter save/load. **The MoE block is done.**
- **The real challenge is the hybrid-attention glue, NOT the experts.** `Qwen3_5MoeDecoderLayer` is `linear_attention` for 3/4 of layers (`full_attention_interval=4`): child `linear_attn` = `Qwen3_5MoeGatedDeltaNet` (GDN), no `self_attn`. The remaining 1/4 are `full_attention` (`self_attn` = `Qwen3_5MoeAttention`, with the q/k/v/o projections). Every layer carries `mlp` (the MoE block), `input_layernorm`, `post_attention_layernorm`.
- **What works today (verified):** experts grouped-GEMM + LoRA; shared expert/gate LoRA + base CPU offload; router offload; norms offload incl. gated/shifted; expert activation offload; full-attention activation offload + attention GC (the 1/4 full-attn layers only); embed/lm_head offload; adapter IO.
- **The gaps (verified, see Q1–Q7 below):** GDN decoder layers get NO decoder-level activation offload/checkpoint (matcher requires `self_attn`); GDN Linear leaves' LoRA behaviour under `target=all` is unvetted (and two of them are degenerate); deps `causal_conv1d`/`fla` are absent in all relevant venvs (GDN runs the pure-torch fallback — verified autograd-safe); MTP head treatment.

**Single biggest memory lever: Stage 1 (generalize the decoder matcher so the existing decoder saved-tensor-offload + checkpoint cover the GDN layers — 3/4 of the model).**

### Corrections to the briefing's "verified context" (re-verified against source + runtime)

- GDN projections are **`in_proj_qkv` / `in_proj_z` / `in_proj_b` / `in_proj_a` / `out_proj`** (five Linears), NOT `in_proj_qkvz`/`in_proj_ba`. Confirmed identical in `third_party/transformers/src/.../modeling_qwen3_5_moe.py:418-421,428` and the runtime copy `LlamaFactory/.venv/.../modeling_qwen3_5_moe.py:418-421`, and on a live tiny model. Q2 below is written against the real names.
- `Qwen3_5MoeSparseMoeBlock` exists on every layer; gate is `Qwen3_5MoeTopKRouter` returning `(router_logits, router_scores, router_indices)`. `AsymQwen35MoeBlock._compute_routing` reads `[1]`=weights, `[2]`=index — matches (`router_scores` is the normalized top-k weight). Q7 OK.
- `causal_conv1d` and `fla` are **not installed** in `AsymGEMM/.venv`, `LlamaFactory/.venv`, or `LlamaFactory-fa4/.conda-lf-fa4`. GDN therefore uses `torch_chunk_gated_delta_rule` + `Qwen3_5MoeRMSNormGated` (NOT `FusedRMSNormGated`). Both verified present and the torch chunk kernel is forward+backward differentiable (isolated test passed). This is the de-risked default; see Stage 0 / Q6.

---

## Profiling harness facts (used by every stage's VALIDATION)

- **Entry:** `scripts/lf/run_lf_lora_sft.sh` (env → `run_lf_profiled_train.py`). Sweep wrapper: `scripts/lf/profile_lora_lf.sh`. The runtime venv is `ENV_DIR=${ROOT}/.venv` = `AsymGEMM/.venv` (`run_lf_lora_sft.sh:163-164`).
- **Peak GPU memory:** `run_lf_profiled_train.py:2267` `peak_allocated = int(torch.cuda.max_memory_allocated())`; surfaced in the source profile JSON as `memory.peak_allocated_hbm_bytes` (and `memory.gpu.peak_allocated_hbm_bytes`), `:2562-2573`. This is the acceptance metric for memory.
- **Step latency:** `measured_e2e_step_milliseconds` (`run_lf_profiled_train.py:432,477`), surfaced under `trainer.timing` (`:2596`) and echoed by postprocess (`postprocess_lf_profile_artifacts.py:293,823,1032`). This is the acceptance metric for latency.
- **Activation-offload counters:** `model._last_activation_offload_stats` snapshots (`decoder_activation_offload.py:248-279`) flow into the profile `activation_offload` block (`:2601`). Use `layer_act_offload_wrapped` / `offloaded_bytes` / `num_offloads` to PROVE the GDN layers were actually wrapped and offloaded.
- **Model spec for the real workload (from the sweep comments, `profile_lora_lf.sh:25,27`):** `Qwen/Qwen3.5-122B-A10B|1`. Template inference returns `qwen3_nothink` for it (`run_lf_lora_sft.sh:310-320`). For Stage 0 use a small **local** Qwen3.5 config dir (recipe below) so we get GDN+full-attn layers cheaply and CPU-first.
- **The flags (all gated through `run_lf_lora_sft.sh`):** `ASYM_OFFLOAD_MODULES`, `ASYMM_EXPERT_ACT_OFFLOAD`(+`_LORA_A_FWD`), `ASYMM_ATTN_ACT_OFFLOAD`, `ASYMM_LAYER_ACT_OFFLOAD`, `ASYM_EXPERT_RECOMPUTE_POLICY` (`none|gc-exp|gc-layer|gc-attn-exp`), `USE_ASYM_CPU_ADAMW`(+grad/weight offload), `BACKEND` (`asym|torch`), `ASYM_PRECISION=bf16`.

### ACCEPTANCE RULE (applied identically in every stage)
Keep a change ONLY if, baseline-vs-change at identical config:
- **Memory:** `memory.peak_allocated_hbm_bytes` drops **meaningfully** — threshold **≥ 5% AND ≥ 512 MiB** on the real workload (toy models: ≥ 5% only, MiB threshold waived). A drop < 2% or < ~100 MiB is "trivial" → REJECT.
- **Latency:** `measured_e2e_step_milliseconds` rises by no more than the **noise band of 5%** (3% target). Take the median of the measured steps (warmup excluded by the harness).
- **Decision matrix:** memory meaningfully down + latency within band → ACCEPT. Memory unchanged but latency up → REJECT. Memory drop trivial → REJECT. Memory down but latency blows up (> 5%) → REJECT (or gate behind a flag, off by default).
- Correctness gate precedes all of the above: forward/backward finite, grads on every LoRA bank, and (Stage 0) numerics equal to `backend=torch` within tolerance.

### EFFICIENCY RULES (binding on every design)
Never split work into many small GEMMs. Never loop over experts in python (the engine already eliminates the native loop). For GDN, never reimplement the delta-rule/conv kernels — wrap ONLY at the decoder-layer boundary (saved-tensor offload / checkpoint) and at the Linear-leaf boundary (LoRA). Prefer the existing grouped-GEMM + fused split-LoRA paths.

---

## Stage 0 — Baseline + correctness harness (MANDATORY; nothing accepted without it)

### SCOPE
Files/functions exercised (no edits): `apply_lf_asym_lora` (`lf.py:993`), `wrap_qwen35_moe_block`/`is_qwen35_moe_block` (`qwen35_moe.py:48,228`), `AsymQwen3Experts` (`qwen3_moe.py:1886`), `is_qwen3_experts` (`qwen3_moe.py:86`). Existing test: `tests/training/test_lf_qwen35_asym_backend.py`. Modeling: `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`.

Goal: a small **bf16** Qwen3.5 text model (a few layers MIXING GDN + full attention, small experts) trains e2e through `run_lf_lora_sft.sh` with `BACKEND=asym, lora_target=all, ASYM_OFFLOAD_MODULES=routed_experts`, and is **numerically equal** to `BACKEND=torch` within tolerance. Record baseline peak memory + step latency. Establish the deps decision.

VALIDATION (must all pass):
1. `is_qwen35_moe_block` matches each `layers.N.mlp` (both layer types); `is_qwen3_experts` passes; `report.qwen35_moes_wrapped == num_layers`; experts wrapped to `AsymQwen3Experts`.
2. Forward/backward finite; every expert-LoRA + dense-LoRA bank has a grad; `_validate_trainable_params` raises nothing.
3. **Numerical equivalence:** loss and (a sample of) LoRA grads from `BACKEND=asym` vs `BACKEND=torch` agree within `atol=4e-3, rtol=2e-2` (the tolerance used in the existing test `_assert_close`).
4. Deps probe recorded: `causal_conv1d`/`fla` availability; confirm the GDN warning "fast path is not available" appears and the run still completes (it does — verified).
5. Record `memory.peak_allocated_hbm_bytes` and `measured_e2e_step_milliseconds` as the BASELINE for Stages 1–3.

ACCEPTANCE RULE: Stage 0 is a gate, not an optimization — it must simply pass (1)–(5). If numerics fail, STOP and fix before any offload stage.

### INTENDED CODE CHANGES
**None to source.** Stage 0 adds only a local test config + (optionally) a tiny e2e test. Build a local Qwen3.5 config directory (CPU-first load → satisfies the `strict` CPU-residency guard in `AsymQwen3Experts.__init__:1956` and `_wrap_lf_linear_leaf:585`):

```python
# scripts/.../make_tiny_qwen35.py  (helper, not shipped src)
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextForCausalLM  # text-only CausalLM
cfg = Qwen3_5MoeTextConfig(
    vocab_size=512, hidden_size=256, num_hidden_layers=8,        # 6 GDN + 2 full-attn (interval=4)
    num_attention_heads=4, num_key_value_heads=2, head_dim=64,
    moe_intermediate_size=128, shared_expert_intermediate_size=128,
    num_experts=16, num_experts_per_tok=4, hidden_act="silu",
    linear_num_key_heads=4, linear_num_value_heads=8,
    linear_key_head_dim=64, linear_value_head_dim=64, linear_conv_kernel_dim=4,
    full_attention_interval=4, max_position_embeddings=4096, output_router_logits=False,
)
m = Qwen3_5MoeTextForCausalLM(cfg).to(dtype="bfloat16")   # keep on CPU
m.save_pretrained("/tmp/tiny-qwen35"); cfg.save_pretrained("/tmp/tiny-qwen35")
# also save the matching tokenizer or point MODEL to a real Qwen3.5 tokenizer dir
```
Reason like real code: keep `head_dim`/`moe_intermediate_size` multiples of 8 so the experts and attention leaves hit the direct-bf16 path; keep `num_hidden_layers` a multiple of 4 so both layer types appear; `output_router_logits=False` because `router_mode=whole` requires it (`lf.py:1018`).

### AMBIGUITY/UNCERTAINTY
- **Is the text-only `*ForCausalLM` class present and named as assumed?** RESOLVE: `grep -n "class Qwen3_5Moe.*ForCausalLM\|Qwen3_5MoeTextModel\|Qwen3_5MoeModel" transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`. The tiny `Qwen3_5MoeTextModel` already built+ran here; for SFT you need the CausalLM head — verify its exact class name and that LF can load it via `MODEL_NAME_OR_PATH=/tmp/tiny-qwen35`. If the only top class is the VL conditional-generation wrapper, build the text config under `text_config` and use the text CausalLM directly.
- **Tokenizer:** LF needs a tokenizer. RESOLVE: point at a real Qwen3.5 tokenizer dir, or copy one into `/tmp/tiny-qwen35`.

### RISKS
- The MTP head (`_keys_to_ignore_on_load_unexpected=[r"^mtp.*"]`, `modeling:903`) may add params; if the chosen CausalLM class instantiates an MTP head it could trip `_validate_trainable_params` (Q5). Covered in Risks/watch.
- Tolerance: the torch chunk delta-rule runs in fp32 internally; asym-vs-torch differences are dominated by the experts' bf16 GEMM, not GDN — `4e-3/2e-2` is the established band but re-confirm on this model.

### WATCH LATER
- If deps get installed later (`fla`/`causal_conv1d`), GDN switches to `FusedRMSNormGated` + fused kernels: re-run Stage 0 numerics AND re-baseline memory/latency, because the norm class change flips `AsymFrozenRMSNorm.gated` detection (Q4) and the saved-tensor population changes (Stage 1).

### EXACT VALIDATION COMMANDS
```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
# (a) deps + GDN autograd probe (records the Q6 decision)
.venv/bin/python - <<'PY'
import importlib.util as u
print("causal_conv1d", u.find_spec("causal_conv1d") is not None, "| fla", u.find_spec("fla") is not None)
PY

# (b) unit harness (the existing suite is the correctness backbone)
.venv/bin/python -m pytest tests/training/test_lf_qwen35_asym_backend.py -q

# (c) build tiny local model (see helper above), then e2e asym baseline
MODEL_NAME_OR_PATH=/tmp/tiny-qwen35 BACKEND=asym ASYM_PRECISION=bf16 \
  LORA_RANK=8 LORA_ALPHA=16 LORA_DROPOUT=0.0 \
  DATASET=asym_long_sft_smoke CUTOFF_LEN=2048 MAX_SAMPLES=16 MAX_STEPS=8 \
  PER_DEVICE_TRAIN_BATCH_SIZE=1 \
  ASYM_OFFLOAD_MODULES=routed_experts ASYM_EXPERT_RECOMPUTE_POLICY=none ASYM_ROUTER_MODE=whole \
  PROFILE=1 PROFILE_PROFILER=source PROFILE_LEVEL=op \
  bash scripts/lf/run_lf_lora_sft.sh

# (d) e2e torch reference (numerical baseline)
MODEL_NAME_OR_PATH=/tmp/tiny-qwen35 BACKEND=torch ASYM_PRECISION=bf16 \
  LORA_RANK=8 LORA_ALPHA=16 LORA_DROPOUT=0.0 \
  DATASET=asym_long_sft_smoke CUTOFF_LEN=2048 MAX_SAMPLES=16 MAX_STEPS=8 \
  PER_DEVICE_TRAIN_BATCH_SIZE=1 ASYM_ROUTER_MODE=whole \
  PROFILE=1 PROFILE_PROFILER=source PROFILE_LEVEL=op \
  bash scripts/lf/run_lf_lora_sft.sh

# (e) read the two acceptance metrics from each run's source profile JSON
.venv/bin/python - "$ASYM_JSON" <<'PY'
import json,sys; p=json.load(open(sys.argv[1]))
print("peak_alloc_bytes", p["memory"]["peak_allocated_hbm_bytes"])
print("e2e_step_ms", p["trainer"]["timing"].get("measured_e2e_step_milliseconds"))
PY
# (loss/grad equivalence: compare the per-step losses in trainer.losses between (c) and (d))
```

---

## Stage 1 — GDN decoder-layer activation offload + checkpoint (THE dominant lever)

### SCOPE
- `asym_gemm/integrations/lf.py:800` `_is_qwen3_decoder_layer_module_name` — generalize so it matches Qwen3.5 GDN decoder layers.
- Consumers (no signature change): `_wrap_qwen3_decoder_saved_tensor_offload_modules` (`lf.py:895`, driven by `ASYMM_LAYER_ACT_OFFLOAD`, `lf.py:1382`) and `_wrap_qwen3_decoder_checkpoint_modules` (`lf.py:843`, driven by `gc-layer`, `lf.py:1183`). Both already call `install_decoder_saved_tensor_offload` / `install_decoder_checkpoint` (`decoder_activation_offload.py:289`, `decoder_checkpoint.py`) which wrap at the WHOLE decoder-layer forward boundary — exactly the right granularity for GDN (never touches the delta-rule kernels).

PROBLEM (verified on a live tiny model): GDN decoder layers expose children `['linear_attn','mlp','input_layernorm','post_attention_layernorm']` (no `self_attn`). The current required set is `{self_attn, mlp, input_layernorm, post_attention_layernorm}` (`lf.py:806`), so with `ASYMM_LAYER_ACT_OFFLOAD=true` (or `gc-layer`) ONLY the 1/4 full-attention layers are wrapped; the GDN layers — the dominant activation memory (3/4 of layers, each producing large `[B,S,H,head_dim]` q/k/v/conv/core_attn tensors) — are NOT offloaded.

VALIDATION (effectiveness):
- After fix, `report.layer_act_offload_wrapped == num_hidden_layers` (was `num_full_attention_layers`); `activation_offload` block shows the GDN layer names with `num_offloads > 0` and nonzero `offloaded_bytes` from GDN-shaped tensors (tags like `decoder.saved.bfloat16.<B>x<S>x...`).
- Same for `gc-layer`: `report.layer_gc_wrapped == num_hidden_layers`.
- Memory: `memory.peak_allocated_hbm_bytes` drops meaningfully vs Stage-0 baseline (expect the bulk of the win here, since the GDN layers were previously un-offloaded). Latency within band (saved-tensor offload is H2D/D2H copy overlapped with compute; `gc-layer` recompute adds a second forward of GDN — measure both, prefer saved-tensor offload if `gc-layer` blows latency).

ACCEPTANCE RULE: Apply the matcher generalization ONLY if peak memory drops ≥ 5% AND ≥ 512 MiB on the real workload with `ASYMM_LAYER_ACT_OFFLOAD=true` (and/or `gc-layer`), and `measured_e2e_step_milliseconds` rises ≤ 5%. If the offload path is net-neutral on memory (it will not be, given 3/4 of layers were excluded) or latency blows up, REJECT that sub-mode and keep only the one that passes.

### INTENDED CODE CHANGES (pseudocode — efficient + correct)
Generalize the matcher to accept either token-mixer child and to recognize the qwen3.5 lineage. Keep it strict (still require `mlp` + both layernorms; still exclude vision via the existing path-marker guard):

```python
# lf.py  (replace _is_qwen3_decoder_layer_module_name)
def _is_qwen3_decoder_layer_module_name(name: str, module: nn.Module) -> bool:
    if not name:
        return False
    if _has_attention_excluded_path_marker(name):   # excludes vision_model/visual/etc.
        return False
    children = dict(module.named_children())
    # Token mixer is EITHER full attention (self_attn) OR GDN linear attention (linear_attn).
    has_token_mixer = ("self_attn" in children) or ("linear_attn" in children)
    base_required = {"mlp", "input_layernorm", "post_attention_layernorm"}
    if not has_token_mixer or not base_required <= set(children):
        return False
    class_name = type(module).__name__.lower()
    module_name = type(module).__module__.lower()
    config = getattr(module, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    # qwen3 + qwen3.5 lineage (class/module/config), unchanged for qwen3.
    if (
        "qwen3" in class_name or "qwen3" in module_name
        or model_type in {"qwen3_moe", "qwen3_vl_moe", "qwen3_5_moe", "qwen3_5_moe_text"}
    ):
        return True
    # Fallback: identify by the MoE block child (covers Asym-wrapped or renamed lineages).
    mlp_child = children["mlp"]
    return (
        hasattr(mlp_child, "_is_asym_qwen3_moe_block")
        or hasattr(mlp_child, "_is_asym_qwen35_moe_block")     # AsymQwen35MoeBlock marker
        or is_qwen3_moe_block(mlp_child)
        or is_qwen35_moe_block(mlp_child)
    )
```
Notes on correctness:
- `is_qwen35_moe_block` returns `False` once the block is already wrapped (it checks `_is_asym_qwen35_moe_block` and returns False, `qwen35_moe.py:49`), which is why the explicit `hasattr(..., "_is_asym_qwen35_moe_block")` check is needed for the post-wrap walk. Decoder wrapping in `apply_lf_asym_lora` happens AFTER expert replacement (`lf.py:1381` vs `:1155`), so the `mlp` child is already `AsymQwen35MoeBlock` — the `hasattr` branch (and the `model_type`/class branch) cover it. The class/module-name branch (`"qwen3_5_moe"` in the decoder layer's own type) is the primary, robust path and fires regardless of wrap order.
- The `class GradientCheckpointingLayer.__call__` base (modeling_layers.py:59) strips `use_cache`/`past_key_values` under its own GC; our wrappers replace `module.forward` and also skip when `use_cache=True` (`decoder_activation_offload.py:153`), so they compose safely. The decoder saved-tensor wrapper only offloads tensors with `requires_grad` and `nbytes ≥ min_bytes` (`:159-179`) — GDN's large activations qualify, the tiny `dt_bias`/`A_log`-derived tensors do not.
- Do NOT also enable HF native gradient checkpointing (`GRADIENT_CHECKPOINTING=true`) together with `gc-layer` — that would double-wrap. The script keeps `GRADIENT_CHECKPOINTING=false` by default (`run_lf_lora_sft.sh:52`); `gc-layer` is the AsymGEMM path.

### AMBIGUITY/UNCERTAINTY
- **Does the GDN forward's `**kwargs` (cache_params/attention_mask/seq_idx/cu_seq_lens) survive the saved-tensors-hooks / checkpoint wrappers?** The wrappers pass `*args, **kwargs` straight through (`decoder_activation_offload.py:149,157`; `decoder_checkpoint.py` `body`), and skip entirely when `use_cache=True`. RESOLVE by the Stage-1 e2e run asserting forward/backward finite + `num_offloads>0` on GDN layers (the tiny model already runs the GDN path).
- **Which sub-mode to ship?** Saved-tensor offload (`ASYMM_LAYER_ACT_OFFLOAD=true`, requires policy `none` + backend `asym`, `lf.py:1039-1042`) vs recompute (`gc-layer`). They are mutually exclusive by config. Measure both; ship whichever passes the ACCEPTANCE RULE with the better memory/latency trade. Given GDN's heavy fp32 internal recompute under the torch fallback, expect `gc-layer` to cost more latency than saved-tensor offload — but verify, don't assume.
- **min_bytes default** is 1 MiB (`decoder_activation_offload.py:14`). On the tiny Stage-0 model many GDN tensors fall below 1 MiB and won't offload — that's expected; validate effectiveness on the REAL workload (long context, hidden=2048), where GDN activations are well above 1 MiB. Use `ASYM_DECODER_SAVED_TENSOR_OFFLOAD_MIN_BYTES` only if needed for the toy correctness run.

### RISKS
- If a future HF refactor renames `linear_attn` (e.g. to `mamba`/`token_mixer`), the `has_token_mixer` set must grow. Low risk; the lineage/model_type branch still catches it.
- Over-broad matching: the added `is_qwen35_moe_block(mlp_child)` fallback could in principle match a non-decoder container that happens to hold a qwen3.5 MoE block; mitigated because we still require BOTH layernorms + a token-mixer child, which only a real decoder layer has.

### WATCH LATER
- With `fla` installed, GDN uses fused kernels whose internal saved tensors differ; re-confirm `num_offloads>0` and re-baseline memory (the chunked FLA kernel may save fewer/different tensors than the torch fallback).
- Interaction with expert activation offload (`ASYMM_EXPERT_ACT_OFFLOAD`) and CPU-Adam weight/grad offload at the real peak — Stage 1 must be measured BOTH standalone and stacked with `ASYM_OFFLOAD_MODULES=all` to confirm the levers are additive, not antagonistic.

### EXACT VALIDATION COMMANDS
```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
# unit: matcher now accepts GDN + full-attn decoder layers (add to the qwen35 test file)
.venv/bin/python - <<'PY'
import torch
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from asym_gemm.integrations.lf import _is_qwen3_decoder_layer_module_name
cfg=Qwen3_5MoeTextConfig(vocab_size=64,hidden_size=16,num_hidden_layers=4,num_attention_heads=2,
  num_key_value_heads=1,head_dim=8,moe_intermediate_size=16,shared_expert_intermediate_size=16,
  num_experts=4,num_experts_per_tok=2,linear_num_key_heads=2,linear_num_value_heads=4,
  linear_key_head_dim=8,linear_value_head_dim=8,linear_conv_kernel_dim=4,full_attention_interval=4)
m=M.Qwen3_5MoeTextModel(cfg)
hits=[n for n,mod in m.named_modules() if _is_qwen3_decoder_layer_module_name(n,mod)]
print("matched decoder layers:", hits)   # EXPECT all 4 (layers.0..3), not just the full-attn one
assert len(hits)==cfg.num_hidden_layers, hits
PY

# e2e: layer activation offload ON; expect layer_act_offload_wrapped == num_layers, memory down
MODEL_NAME_OR_PATH=/tmp/tiny-qwen35 BACKEND=asym ASYM_PRECISION=bf16 \
  LORA_RANK=8 LORA_ALPHA=16 LORA_DROPOUT=0.0 CUTOFF_LEN=2048 MAX_SAMPLES=16 MAX_STEPS=8 \
  PER_DEVICE_TRAIN_BATCH_SIZE=1 ASYM_ROUTER_MODE=whole \
  ASYM_OFFLOAD_MODULES=routed_experts ASYM_EXPERT_RECOMPUTE_POLICY=none \
  ASYMM_LAYER_ACT_OFFLOAD=true \
  PROFILE=1 PROFILE_PROFILER=source PROFILE_LEVEL=op \
  bash scripts/lf/run_lf_lora_sft.sh
# also the gc-layer variant (mutually exclusive with ASYMM_LAYER_ACT_OFFLOAD):
#   ASYM_EXPERT_RECOMPUTE_POLICY=gc-layer ASYMM_LAYER_ACT_OFFLOAD=false ...

# accept/reject: compare to Stage-0 baseline JSON
.venv/bin/python - "$BASELINE_JSON" "$STAGE1_JSON" <<'PY'
import json,sys
b=json.load(open(sys.argv[1])); s=json.load(open(sys.argv[2]))
bm=b["memory"]["peak_allocated_hbm_bytes"]; sm=s["memory"]["peak_allocated_hbm_bytes"]
bl=b["trainer"]["timing"]["measured_e2e_step_milliseconds"]; sl=s["trainer"]["timing"]["measured_e2e_step_milliseconds"]
ao=s.get("activation_offload",{})
print(f"mem {bm}->{sm} ({100*(bm-sm)/bm:.1f}% lower, {(bm-sm)/2**20:.0f} MiB)")
print(f"lat {bl:.1f}->{sl:.1f} ms ({100*(sl-bl)/bl:+.1f}%)")
print("layer_act_offload_wrapped", ao.get("layer_act_offload_wrapped"), "num_offloads", ao.get("num_offloads"))
ok = (bm-sm)/bm>=0.05 and (bm-sm)>=512*2**20 and (sl-bl)/bl<=0.05   # MiB gate on real workload only
print("ACCEPT" if ok else "REJECT")
PY
```

---

## Stage 2 — Shared expert + shared_expert_gate (base CPU offload + LoRA; exclude the degenerate gate)

### SCOPE
- Dense walk in `apply_lf_asym_lora` (`lf.py:1282-1356`) via `_wrap_lf_linear_leaf` (`lf.py:563`) and `classify_lf_component` (`lf.py:372`).
- The wrapped block keeps `shared_expert` (a `Qwen3_5MoeMLP`: `gate_proj/up_proj/down_proj`) and `shared_expert_gate` (`nn.Linear(hidden,1)`) RAW (`qwen35_moe.py:146-147`). After block replacement, the dense walk reaches `{block}.shared_expert.{gate,up,down}_proj` (classify→`shared_experts`) and `{block}.shared_expert_gate` (classify→`shared_experts`, `lf.py:377`).

STATUS (already verified by existing tests, so Stage 2 is largely a CONFIRM-and-guard, not new code):
- `shared_expert.{gate,up,down}_proj` get base CPU offload + LoRA when `shared_experts ∈ ASYM_OFFLOAD_MODULES` and `target` includes those leaves (`test_apply_lf_asym_lora_qwen35_shared_experts_adopt_cpu_storage_without_clone`).
- `shared_expert_gate` (out_features=1) — the existing tests show it CAN be wrapped (`test_..._shared_modules`, `test_..._shape_is_not_direct_bf16_compatible`). out_features=1 violates `_direct_bf16_linear_shape_reason` (`requires_8_aligned_nk`, `lf.py:556-558`) so its base falls back to `backend=torch` CPU-fetch, and LoRA on a `(1,hidden)` matrix is **degenerate** (rank capped at 1, near-zero capacity, wasteful). The clean policy: **base CPU offload YES; LoRA NO**.

VALIDATION (effectiveness):
- Shared expert is DENSE (every token), so its base weights (3×`hidden×shared_intermediate`) and activations are nontrivial. With `ASYM_OFFLOAD_MODULES` including `shared_experts`, `report.cpu_resident_base_bytes_by_component["shared_experts"] > 0` and `selected_gpu_resident_base_bytes_by_component` has no `shared_experts` residue.
- Shared-expert ACTIVATION memory is covered for free by Stage 1 (the shared expert runs inside the decoder layer's forward, so its saved tensors are offloaded by the decoder saved-tensor wrapper). No separate dense-MLP activation path is needed — confirm via the Stage-1 `offloaded_bytes` tags.

ACCEPTANCE RULE: Including `shared_experts` in CPU offload must drop peak memory ≥ 5% AND ≥ 512 MiB on the real workload (shared-expert base is large at hidden=2048, num_layers=40) with latency ≤ 5% up. The LoRA-exclusion of `shared_expert_gate` must not change loss meaningfully (it's a near-degenerate adapter) and must remove a wasteful `(1,hidden)` adapter — keep the exclusion regardless of memory (it's a correctness/efficiency hygiene change, validated by "no `shared_expert_gate.lora_A` in trainable params").

### INTENDED CODE CHANGES
The base-offload path needs **no change** (works today). The only change is to **stop emitting LoRA on the degenerate `shared_expert_gate`**, while still allowing its base to be CPU-offloaded. Two equally valid options; prefer (A) (localized, no new flag):

```python
# Option A — in _wrap_lf_linear_leaf (lf.py:563), force is_lora_target=False for the (·,1) gate.
def _is_degenerate_lora_leaf(name: str, module: nn.Linear) -> bool:
    # A (out=1) projection (Qwen3.5 shared_expert_gate, hidden->1) is not a useful LoRA target.
    return name.rsplit(".", 1)[-1] == "shared_expert_gate" and int(module.weight.shape[0]) == 1

# at the top of _wrap_lf_linear_leaf, after computing device/dtype:
if is_lora_target and _is_degenerate_lora_leaf(name, module):
    is_lora_target = False        # base CPU offload still applies if selected_cpu_offload
```
Then it naturally returns `AsymFrozenLinear.from_host_weight(...)` when `selected_cpu_offload` (base offload, no LoRA), or — if not selected for offload and not a LoRA target — the leaf is simply skipped by the caller's `if not is_lora_target and not selected_cpu_offload: continue` (`lf.py:1295`). Reason: this removes one wasted adapter + its optimizer state per layer (40 layers × `(1+hidden)` params is small but the adapter is pure noise) and avoids a `(1,hidden)` GEMM in the forward.

Correctness guard: `freeze_non_lora_params` + `_validate_trainable_params` (`lf.py:1395-1396`) still pass because the gate's base is frozen and no LoRA bank is created for it.

NOTE on the existing tests: `test_apply_lf_asym_lora_whole_wraps_qwen35_and_dense_shared_modules:360` and `:418` assert `shared_expert_gate` HAS `lora_A`. **These tests encode the CURRENT behaviour and must be updated** to assert the gate is `AsymFrozenLinear`/base-only (or `TorchLoRALinear` with LoRA disabled) — flag this to the reviewer; do not silently break them. (This is the only place Stage 2 touches tests.)

### AMBIGUITY/UNCERTAINTY
- **Is excluding the gate's LoRA a behaviour change users rely on?** PEFT `target=all` would, by default, put LoRA on every Linear including `(·,1)`; but a rank-`r` adapter on a 1-row matrix is mathematically rank-≤1 and contributes negligible capacity. RESOLVE: confirm loss curves are within noise with/without the gate adapter on the tiny model; if a stakeholder wants exact PEFT parity, gate the exclusion behind `ASYM_LORA_SKIP_DEGENERATE=true` (default true) instead of unconditional. Document the choice in the adapter config.
- **`shared_expert` leaf shapes** (`hidden=2048 ↔ shared_intermediate=512`) are 8-aligned and 64-aligned on the out dim for `gate/up_proj` (out=512) and `down_proj` (out=2048) → direct-bf16 LoRA path, efficient. Confirm with `_direct_bf16_linear_shape_reason` on the real config.

### RISKS
- If `shared_expert_intermediate_size` is ever not 64-aligned, `down_proj`/`up_proj` could fall back to torch — measure, but it's the existing dense-leaf behaviour, not new.
- Touching the two assertions in the existing test is mandatory; missing it makes the suite red.

### WATCH LATER
- Whether shared-expert base offload + Stage-1 activation offload + expert offload are additive at the real peak (measure stacked).

### EXACT VALIDATION COMMANDS
```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
# base offload present, gate not a LoRA target, no degenerate adapter
MODEL_NAME_OR_PATH=/tmp/tiny-qwen35 BACKEND=asym ASYM_PRECISION=bf16 \
  LORA_RANK=8 LORA_ALPHA=16 LORA_DROPOUT=0.0 CUTOFF_LEN=2048 MAX_SAMPLES=16 MAX_STEPS=8 \
  PER_DEVICE_TRAIN_BATCH_SIZE=1 ASYM_ROUTER_MODE=whole \
  ASYM_OFFLOAD_MODULES=routed_experts,shared_experts ASYM_EXPERT_RECOMPUTE_POLICY=none \
  ASYMM_LAYER_ACT_OFFLOAD=true \
  PROFILE=1 PROFILE_PROFILER=source PROFILE_LEVEL=op \
  bash scripts/lf/run_lf_lora_sft.sh

.venv/bin/python - "$STAGE2_JSON" <<'PY'
import json,sys; p=json.load(open(sys.argv[1]))
cpu=p.get("lora",{}); 
# residency by component is in the setup log; here assert no degenerate gate adapter survived:
# (grep the train log for shared_expert_gate adapters; expect none)
print("peak_alloc", p["memory"]["peak_allocated_hbm_bytes"], "e2e_ms", p["trainer"]["timing"]["measured_e2e_step_milliseconds"])
PY
grep -E "shared_expert_gate.*(lora_A|lora_B)" "$STAGE2_LOG" && echo "FAIL: degenerate gate adapter present" || echo "OK: no gate adapter"
# accept/reject vs Stage-1 JSON via the same comparison snippet as Stage 1 (mem >=5% & >=512MiB, lat <=5%).
```

---

## Stage 3 — Norms offload for shifted / gated RMSNorm

### SCOPE
- `asym_gemm/training/offload.py:339` `AsymFrozenRMSNorm` (already class-cases `Qwen3_5MoeRMSNorm`→`shifted_weight`, `Qwen3_5MoeRMSNormGated`→`gated`, `:354-355`; forward at `:361-376`).
- Driver: `apply_lf_asym_lora` norms branch (`lf.py:1243-1268`) selecting modules where `classify_lf_component(...)=="norms"`.

STATUS (already verified): `test_asym_frozen_qwen35_rmsnorm_matches_shifted_weight_formula` and `test_asym_frozen_qwen35_gated_rmsnorm_matches_transformers` PASS — both the shifted `(1+w)` formula and the gated `forward(x, gate)` match HF exactly. VERIFIED at runtime that the GDN `norm` is `Qwen3_5MoeRMSNormGated` (because `fla` is absent). So the gated case is the live path.

The Qwen3.5 norm inventory and classification:
- `input_layernorm`, `post_attention_layernorm` (`Qwen3_5MoeRMSNorm`, shifted) → classify `norms` ✓ (`lf.py:430-431`).
- Full-attn `self_attn.q_norm`, `self_attn.k_norm` (`Qwen3_5MoeRMSNorm`, shifted, head-dim) → classify `norms` ✓ (`lf.py:427-428,433-434`).
- Model final `norm` (`Qwen3_5MoeRMSNorm`) → `norms` ✓ (`lf.py:429`).
- GDN `linear_attn.norm` (`Qwen3_5MoeRMSNormGated`, **gated**, lives INSIDE the GDN, invoked as `self.norm(core_attn_out, z)` at `modeling:551`) → leaf `norm`, classify `norms` ✓.

CRITICAL CORRECTNESS POINT for the gated norm: the GDN calls its norm with a positional `gate` (`z`). `AsymFrozenRMSNorm.forward(self, x, gate=None)` accepts that positional second arg and requires it in gated mode (`offload.py:361-365`). So replacing the GDN's `norm` with `AsymFrozenRMSNorm` preserves the call contract. VERIFIED by the passing gated test. The per-call weight staging is a `head_v_dim`-length vector copy (`offload.py:362`) — negligible.

VALIDATION (effectiveness):
- With `ASYM_OFFLOAD_MODULES` including `norms`, `report.cpu_resident_base_bytes_by_component["norms"] > 0`, no `norms` GPU residue; forward/backward finite; loss within Stage-0 tolerance (norms are frozen; only staging changes).
- The gated GDN norm specifically: assert the replaced module is `AsymFrozenRMSNorm` with `.gated==True` and the GDN forward still runs (it does on the tiny model).

ACCEPTANCE RULE: Norm weights are TINY (a few KiB each). Their CPU offload will NOT move peak memory meaningfully → by the ACCEPTANCE RULE, **do NOT enable `norms` offload for a memory win** (it's trivial, REJECT as a standalone lever). KEEP norms offload ONLY as part of `ASYM_OFFLOAD_MODULES=all` correctness coverage, and ONLY if it does not raise latency > 5% (per-call host→device staging of a tiny vector must overlap; if it measurably slows the step, REJECT and leave norms resident). The deliverable here is a CORRECTNESS/coverage guarantee for the gated+shifted variants under `all`, not a memory optimization.

### INTENDED CODE CHANGES
**None expected** — `AsymFrozenRMSNorm` already handles both variants and the driver already selects them. Stage 3 is a VERIFY stage. The only possible change is defensive: if profiling shows the per-call weight `.to(device)` in the gated path (`offload.py:362`) adds latency on the hot GDN path (3/4 of layers × every step), pin the norm weight and stage on a side stream, or simply EXCLUDE GDN gated norms from offload (they are tiny — leaving them resident costs ~nothing in memory and avoids any latency). Pseudocode for the optional exclusion (only if Stage 3 latency check fails):

```python
# lf.py norms branch: skip the GDN gated norm if staging hurts the hot path (tiny weight, no memory cost)
class_name = type(module).__name__.lower()
if "rmsnormgated" in class_name and _env_true(os.environ.get("ASYM_KEEP_GATED_NORM_RESIDENT", "true")):
    continue   # leave Qwen3_5MoeRMSNormGated resident; negligible memory, avoids per-call staging on GDN
```

### AMBIGUITY/UNCERTAINTY
- **Does the gated-norm offload survive being INSIDE a Stage-1-wrapped decoder layer (saved-tensor hooks active)?** The norm forward produces activations that the decoder wrapper may offload; the norm WEIGHT staging is separate (a `.to()` inside forward). RESOLVE by the stacked Stage-1+norms e2e run asserting finite grads + `.gated==True` on the replaced module.
- **`FusedRMSNormGated` (if `fla` later installed)** is NOT detected by `AsymFrozenRMSNorm` (it only matches class name `Qwen3_5MoeRMSNormGated`, `offload.py:355`). If deps get installed, the GDN norm becomes `FusedRMSNormGated` and `classify`/`AsymFrozenRMSNorm` will treat it as a plain RMSNorm (no gate) → WRONG. WATCH item: extend the gated detection to also match `fusedrmsnormgated` if/when `fla` is installed.

### RISKS
- If `norms` offload is enabled and a norm leaf is not actually frozen elsewhere, `_validate_trainable_params` would catch it — but norms are frozen by `freeze_non_lora_params`. Low risk.

### WATCH LATER
- `FusedRMSNormGated` detection gap (above) — only relevant once `fla` is installed.

### EXACT VALIDATION COMMANDS
```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
# unit: both variants already covered
.venv/bin/python -m pytest tests/training/test_lf_qwen35_asym_backend.py -q \
  -k "gated_rmsnorm or shifted_weight"
# runtime: gated GDN norm is the wrapped class, forward finite
.venv/bin/python - <<'PY'
import torch
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from asym_gemm.training.offload import AsymFrozenRMSNorm
cfg=Qwen3_5MoeTextConfig(hidden_size=16,num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,
  head_dim=8,moe_intermediate_size=16,shared_expert_intermediate_size=16,num_experts=4,num_experts_per_tok=2,
  linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=8,linear_value_head_dim=8,
  linear_conv_kernel_dim=4,full_attention_interval=4,vocab_size=64)
gdn=[m for m in M.Qwen3_5MoeTextModel(cfg).modules() if type(m).__name__=="Qwen3_5MoeGatedDeltaNet"][0]
w=AsymFrozenRMSNorm(gdn.norm); print("gated", w.gated, "shifted", w.shifted_weight)  # EXPECT gated True
PY
# e2e: all selector includes norms; accept/reject -> KEEP only if latency <=5% (memory delta expected trivial)
MODEL_NAME_OR_PATH=/tmp/tiny-qwen35 BACKEND=asym ASYM_OFFLOAD_MODULES=all \
  ASYM_PRECISION=bf16 LORA_RANK=8 LORA_ALPHA=16 LORA_DROPOUT=0.0 CUTOFF_LEN=2048 MAX_SAMPLES=16 MAX_STEPS=8 \
  PER_DEVICE_TRAIN_BATCH_SIZE=1 ASYM_ROUTER_MODE=whole ASYMM_LAYER_ACT_OFFLOAD=true \
  PROFILE=1 PROFILE_PROFILER=source PROFILE_LEVEL=op bash scripts/lf/run_lf_lora_sft.sh
```

---

## Risks / watch-later (cross-stage, unresolved)

### Q2 — LoRA on GDN Linear leaves (`in_proj_qkv/z/b/a`, `out_proj`) under `target=all`
VERIFIED leaf set + shapes on a live model: `out_proj` (value_dim→hidden, both 8/64-aligned at real sizes → direct-bf16 LoRA OK), `in_proj_qkv` (hidden→`2*key_dim+value_dim`, large, aligned → OK), `in_proj_z` (hidden→value_dim → OK), **`in_proj_b` and `in_proj_a` (hidden→`num_v_heads`)**. Real `num_v_heads=32` → out_features=32: 8-aligned but NOT 64-aligned → `_direct_bf16_linear_shape_reason(require_backward=True)` returns `dx_transpose_b_requires_64_aligned_out_features` (`lf.py:558`), so under `selected_cpu_offload` they fall back to `backend=torch` (correct, just not the fast path), and a rank-`r` LoRA on a 32-wide output is low-value (near-degenerate, like `shared_expert_gate` but less extreme). conv1d/`A_log`/`dt_bias` are NOT `nn.Linear` → never LoRA-wrapped (correct — never touch the delta-rule params). DECISION: (a) confirm the four big GDN Linears get useful LoRA and feed the GDN forward correctly (Stage-0 finite-grad check already exercises this since the tiny model has GDN layers and `target=all`); (b) consider extending the Stage-2 degenerate-leaf exclusion to `in_proj_b`/`in_proj_a` (out_features ≤ some threshold, e.g. < 64) to avoid wasted adapters — measure loss impact first. These are "other"/unselectable for base CPU offload today (no component bucket); leaving their base resident is fine (GDN base is dominated by `in_proj_qkv`/`out_proj`, not b/a). **Watch:** if GDN base CPU offload is ever wanted, add a `gdn`/`linear_attn` component to `SUPPORTED_LF_OFFLOAD_COMPONENTS` + `classify_lf_component` — out of scope unless memory demands it.

### Q5 — MTP (multi-token-prediction) head
`Qwen3_5MoePreTrainedModel._keys_to_ignore_on_load_unexpected=[r"^mtp.*"]` (`modeling:903`). If the instantiated SFT class builds an MTP head with trainable params, `_validate_trainable_params` (`lf.py:918`) will raise on non-LoRA trainables. DECISION: ensure the MTP head is frozen (it is not a LoRA/offload target → it must carry `requires_grad=False`). `freeze_non_lora_params` (`lf.py:1395`) freezes everything not LoRA, so MTP becomes frozen automatically; confirm at Stage 0 that the chosen CausalLM class either has no MTP head or that `_validate_trainable_params` passes. If an MTP head exists and is large, optionally add it to `ASYM_OFFLOAD_MODULES` coverage (classify currently → `other`, stays resident). **Watch:** verify with `grep -n "mtp" modeling_qwen3_5_moe.py` and a Stage-0 trainable-param dump.

### Q6 — GDN deps (`causal_conv1d`, `fla`)
VERIFIED ABSENT in all three venvs; GDN runs `torch_chunk_gated_delta_rule` + `Qwen3_5MoeRMSNormGated`, which are forward+backward correct (isolated test passed) and complete e2e (tiny model ran). So deps are NOT a hard blocker for correctness, but: (a) the torch fallback is slower and may use MORE activation memory (more intermediate fp32 tensors) than the fused path — this actually makes Stage 1's decoder offload MORE valuable; (b) installing `fla`/`causal_conv1d` later changes the norm class (Q4 detection gap) and the saved-tensor population (Stage 1 re-baseline). DECISION for the plan: develop + accept Stages 1–3 against the torch-fallback path (the shipping default here), and record a WATCH to re-validate numerics + re-baseline memory/latency if deps are installed. **Do not gate any stage on installing deps.**

### Q7 — Router (`Qwen3_5MoeTopKRouter`)
VERIFIED: returns `(router_logits, router_scores, router_indices)`; `router_scores`=normalized top-k weights (`router_top_value /= sum`, `modeling:788`). `AsymQwen35MoeBlock._compute_routing` consumes `[1]`=weights, `[2]`=index, detaches under no-grad, and casts weights to the hidden dtype (`qwen35_moe.py:187-197`). `norm_topk_prob` is effectively always-on in this router (the normalize is unconditional in HF) — matches. The existing `test_asym_qwen35_moe_whole_matches_source_and_detaches_router` covers it. No action; **watch** only if HF adds a `norm_topk_prob=False` config branch.

### General
- Measure every stage BOTH standalone and stacked under `ASYM_OFFLOAD_MODULES=all` + `USE_ASYM_CPU_ADAMW=true` (+grad/weight offload) at the real `Qwen/Qwen3.5-122B-A10B|1` workload, SEQ≈11264, batch as in the sweep — that stacked config is the true acceptance target; the toy model only proves correctness + matcher behaviour, not the real peak.
- The full-attention path (1/4 of layers) already has attention activation offload (`ASYMM_ATTN_ACT_OFFLOAD`, matcher `_is_text_attention_module_name` requires q/k/v/o — only full-attn layers match; GDN has none, correctly skipped). Stage 1's decoder-level offload covers BOTH layer types and supersedes the need for any GDN-specific attention path. Do not attempt a GDN "attention activation offload" — wrap at the decoder boundary only.
