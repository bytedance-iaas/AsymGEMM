# Liger Loss-Only for Qwen3.5-MoE

Wire `ENABLE_LIGER_KERNEL=true` (`ligerloss1`) to enable **only** Liger fused linear cross-entropy
for Qwen3.5-MoE — no RoPE / RMSNorm / SwiGLU / expert / standalone-CE patches — for plain LF/DeepSpeed
(`zero3_offload`) runs and for AsymGEMM runs where `lm_head` is CPU-resident / `AsymFrozenLinear`-wrapped.

This is the qwen3/llama4 Liger path extended to qwen3.5. It is a **new, independent axis** and does **not**
change the `ligerloss0` offload-vs-recompute research in `fix_qwen35.md` (that stays `ligerloss0`; the ~45.5
GiB CE cancels there). Here the CE is exactly what we attack.

## ⚠️ Critical correction vs the first analysis: qwen3.5 is ConditionalGeneration, like llama4

`Qwen/Qwen3.5-35B-A3B` loads as **`Qwen3_5MoeForConditionalGeneration`** (verified: 18 hits in the existing
`profiling_qwen35_goal` train.log; `config.architectures == ['Qwen3_5MoeForConditionalGeneration']`,
`config.model_type == "qwen3_5_moe"`, `text_config.model_type == "qwen3_5_moe_text"`,
`text_config.vocab_size == 248320`, `text_config.hidden_size == 2048`, `tie_word_embeddings == False`).

So it needs the **conditional-generation bridge shape**, NOT the plain `qwen3_moe` causal reuse.
But it is **simpler than llama4-conditional**, because:

| | llama4 (`llama4`) | qwen3.5 (`qwen3_5_moe`) |
|---|---|---|
| Does Liger patch the ConditionalGeneration class? | **No** (only the inner CausalLM) | **Yes** — `Qwen3_5MoeForConditionalGeneration.forward = lce_forward_conditional_generation` |
| Does the Liger conditional forward fuse the loss? | n/a (dead) | **Yes** — reads `self.lm_head.weight`, uses `self.config.text_config.hidden_size`, no top-level logits |
| Normal/zero3: is the LF class patch sufficient? | No → needs post-load bridge | **Yes** for normal; zero3 engages the patch but is subject to the ZeRO-3 gather caveat (see Risks) |
| Bridge must re-do image merge / `language_model.model(...)`? | Yes (manual) | **No** — Liger's forward already calls `self.model(...)`; we copy it and swap one line |

⇒ The **only** reason qwen3.5 needs an Asym post-load bridge is `lm_head` staging (Liger's conditional
forward reads `self.lm_head.weight` directly, which is the CPU tensor under Asym). For normal/`zero3_offload`,
the Stage-1 class patch fuses CE correctly on its own.

## Decision metric (what "works" means)

Per backend, run Liger off (`ligerloss0`) vs on (`ligerloss1`) at one matched config and feed both run dirs
to `scripts/lf/compare_liger_loss_profiles.py`. Liger works for qwen3.5 on a backend iff the comparator
returns `ok: true` (exit 0):
- peak allocated HBM ↓ **≥ 10 GiB**,
- `lm_head`/`loss` HBM attribution ↓ **≥ 20 GiB** (the `[batch, seq, vocab]` logits the fused kernel never
  materializes — with vocab 248320 this is the dominant signal),
- step ≤ **1.10×**, forward/backward ≤ **1.15×** of off,
- loss finite and close to off at the same seed/workload (fla delta-net is non-deterministic → allow ~5%),
- bridge metadata `enabled=true` with the expected `bridge_kind`/`weight_source`.

**We already have the `ligerloss0` baselines** (see Stage 4) — so this work only runs the `ligerloss1` arms
and compares against the existing `ligerloss0` runs. The Stage-1/2 code changes are a **proven no-op for
`ligerloss0`** (resolver early-returns when `enable_liger_kernel=false`; the bridge installs only when
`enable_liger_kernel=true`), so reusing the existing `ligerloss0` runs as baselines is valid.

## Grounding facts (verified against local source)

- Liger ships `apply_liger_kernel_to_qwen3_5_moe(rope=False, cross_entropy=False, fused_linear_cross_entropy=True,
  rms_norm=True, swiglu=True, model=None)` and `MODEL_TYPE_TO_APPLY_LIGER_FN` already maps both `qwen3_5_moe`
  and `qwen3_5_moe_text` to it. The bool set matches `qwen3_moe`, so the existing generic
  `_build_liger_loss_only_kwargs()` produces exactly `{fused_linear_cross_entropy: True, rope/cross_entropy/
  rms_norm/swiglu: False}` with **no special-casing** (no `layer_norm`, unlike llama4).
- Liger `model/qwen3_5_moe.py` exposes two forwards: `lce_forward` (for `Qwen3_5MoeForCausalLM`) and
  `lce_forward_conditional_generation` (for `Qwen3_5MoeForConditionalGeneration`). The conditional one:
  - calls `self.model(input_ids, pixel_values, ..., mm_token_type_ids, ...)` (the `Qwen3_5MoeModel`, which does
    its own vision/text merge) and takes `hidden_states = outputs[0]`,
  - fuses via `LigerForCausalLMLoss(hidden_states=..., lm_head_weight=self.lm_head.weight,
    hidden_size=self.config.text_config.hidden_size, ...)`,
  - returns `LigerQwen3_5MoeCausalLMOutputWithPast` (has `rope_deltas`, `router_logits`, `aux_loss`,
    `token_accuracy`, `predicted_tokens`),
  - aux-loss path uses `self.config.text_config.{num_experts,num_experts_per_tok,router_aux_loss_coef}` and only
    fires on `output_router_logits=True`.
- Imports needed by the bridge:
  - `from liger_kernel.transformers.model.output_classes import LigerQwen3_5MoeCausalLMOutputWithPast`
  - `from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss, unpack_cross_entropy_result`
    (already imported in `liger_loss.py`)
  - `from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import load_balancing_loss_func` (lazy, inside
    the aux branch).
- AsymGEMM already recognizes qwen3.5 (`asym_gemm/integrations/lf.py` and `training/qwen35_moe.py` switch on
  `{"qwen3_5_moe", "qwen3_5_moe_text"}`); the expert engine and the fla delta-net decoder live under
  `model.model.language_model`. The Liger loss-only patch touches **only** the top-level `forward` (CE) — it
  must NOT patch `Qwen3_5MoeExperts`/`Qwen3_5MoeRMSNorm` (those would collide with the Asym expert engine), and
  loss-only kwargs guarantee that (`swiglu=False`, `rms_norm=False`).
- `_base_causal_lm_model()` returns the `Qwen3_5MoeForConditionalGeneration` itself (it has both `.lm_head` and
  `.model`), and its `config.model_type == "qwen3_5_moe"`. The new detection helper must run **before** the
  generic qwen3_moe path so the conditional forward (uses `self.model(...)` + `text_config`) is installed, not
  the qwen3_moe forward (uses `self.config.hidden_size` + `MoeModelOutputWithPast` — would break here).
- Baselines on disk (`profiling_qwen35_goal`, source mode, `b4_s4096_ga1`, all `ligerloss0`): `asym_cpuadamwds`
  sd0 `none|true|true|false|true|false`, sd1 `none|true|true|false|true|true`, A `recomp` no-offload;
  `zero3_offload` recomp (B) and norecomp. `compare_liger_loss_profiles.py` reads `source_profile.json` (peak +
  timing) and `memory_breakdown_summary.json` (`actual_peak_breakdown_rows` → `lm_head`/`loss` HBM).

## Install precondition (once, before any `ligerloss1` run)

```bash
ASYM_DIR=${ASYM_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")}
SFT_ROOT=${SFT_ROOT:-$(cd "${ASYM_DIR}/../.." && pwd)}
ENV_PYTHON=${ENV_PYTHON:-${ASYM_DIR}/.venv/bin/python}

"${ENV_PYTHON}" -m pip install --no-deps --no-build-isolation -e "${SFT_ROOT}/third_party/Liger-Kernel"
"${ENV_PYTHON}" - <<'PY'
import inspect
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5_moe
from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
sig = inspect.signature(apply_liger_kernel_to_qwen3_5_moe)
assert {"rope","cross_entropy","fused_linear_cross_entropy","rms_norm","swiglu","model"} <= set(sig.parameters), sig
assert MODEL_TYPE_TO_APPLY_LIGER_FN["qwen3_5_moe"] is apply_liger_kernel_to_qwen3_5_moe
assert MODEL_TYPE_TO_APPLY_LIGER_FN["qwen3_5_moe_text"] is apply_liger_kernel_to_qwen3_5_moe
print("qwen3_5_moe Liger dispatch OK:", sig)
PY
```

---

## Stage 1 — LF loss-only gate (enables normal + `zero3_offload`)

File: `../LlamaFactory/src/llamafactory/model/model_utils/liger_kernel.py`

```python
# 1a — whitelist
_LOSS_ONLY_SUPPORTED_MODEL_TYPES = {
    "qwen3_moe", "llama4_text", "llama4", "qwen3_5_moe", "qwen3_5_moe_text",
}

# 1b — dispatch (add before `return None` in _resolve_liger_apply_fn)
    if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5_moe
        return apply_liger_kernel_to_qwen3_5_moe
```

No change to `_build_liger_loss_only_kwargs()`. This alone makes `zero3_offload` + plain LF runs fuse CE (the
class patch sets `Qwen3_5MoeForConditionalGeneration.forward = lce_forward_conditional_generation`).

**Validation:**
```bash
"${ENV_PYTHON}" -m pytest -q tests/lf/test_liger_loss_only_qwen3_moe.py
"${ENV_PYTHON}" - <<'PY'
from types import SimpleNamespace
from llamafactory.model.model_utils import liger_kernel
calls=[]
def fake(rope=True,cross_entropy=False,fused_linear_cross_entropy=True,rms_norm=True,swiglu=True,model=None):
    calls.append(dict(rope=rope,cross_entropy=cross_entropy,fused_linear_cross_entropy=fused_linear_cross_entropy,rms_norm=rms_norm,swiglu=swiglu))
liger_kernel._resolve_liger_apply_fn = lambda mt: fake
for mt in ("qwen3_5_moe","qwen3_5_moe_text"):
    calls.clear()
    liger_kernel.apply_liger_kernel(SimpleNamespace(model_type=mt), SimpleNamespace(enable_liger_kernel=True), is_trainable=True, require_logits=False)
    assert calls==[dict(rope=False,cross_entropy=False,fused_linear_cross_entropy=True,rms_norm=False,swiglu=False)], (mt,calls)
print("Stage 1 loss-only kwargs OK")
PY
```

---

## Stage 2 — Asym conditional `lm_head` bridge (enables the Asym staged path)

File: `asym_gemm/integrations/liger_loss.py`

**2a — top-level imports** (next to the existing Liger output-class imports):
```python
from transformers.utils import can_return_tuple
from liger_kernel.transformers.model.output_classes import LigerQwen3_5MoeCausalLMOutputWithPast
```

**2b — detection helper** (mirror `_is_llama4_conditional_generation`):
```python
def _is_qwen3_5_moe_conditional_generation(model: nn.Module) -> bool:
    config = getattr(model, "config", None)
    return (
        getattr(config, "model_type", None) == "qwen3_5_moe"
        and getattr(config, "text_config", None) is not None   # composite config ⇒ conditional wrapper
        and hasattr(model, "lm_head")
        and hasattr(model, "model")
    )
```

**2c — conditional fused forward** — line-for-line copy of Liger `lce_forward_conditional_generation`. The only
*behavior* change is the `lm_head_weight` line; `load_balancing_loss_func` is imported locally so it doesn't
shadow the module-level mixtral import the qwen3_moe bridge uses. **Keep the `@can_return_tuple` decorator** —
Liger binds the decorated function via `MethodType`, and it is what yields a tuple when `return_dict=False`
(the existing llama4 conditional bridge instead took an explicit `return_dict` param; either is valid, but this
mirrors Liger's qwen3.5 forward exactly):
```python
@can_return_tuple
def asym_qwen3_5_moe_conditional_lce_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    mm_token_type_ids=None,
    logits_to_keep=0,
    skip_logits=None,
    **kwargs,
):
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )

    hidden_states = outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    shift_labels = kwargs.pop("shift_labels", None)
    loss = None
    logits = None
    token_accuracy = None
    predicted_tokens = None

    if skip_logits and labels is None and shift_labels is None:
        raise ValueError("skip_logits is True, but labels and shift_labels are None")

    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        lm_head_weight = _resolve_liger_lm_head_weight(self.lm_head, kept_hidden_states)  # <-- only change vs Liger
        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=lm_head_weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.text_config.hidden_size,
            **kwargs,
        )
        loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)
    else:
        logits = self.lm_head(kept_hidden_states)
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size
            )

    aux_loss = None
    if kwargs.get("output_router_logits", False):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import load_balancing_loss_func
        aux_loss = load_balancing_loss_func(
            outputs.router_logits,
            self.config.text_config.num_experts,
            self.config.text_config.num_experts_per_tok,
            attention_mask,
        )
        if loss is not None and aux_loss is not None:
            loss = loss + self.config.text_config.router_aux_loss_coef * aux_loss.to(loss.device)

    return LigerQwen3_5MoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
        router_logits=outputs.router_logits,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )
```

**2d — installer** (conditional first; plain-causal fallback reuses the existing qwen3_moe forward, which equals
Liger's `qwen3_5_moe` causal `lce_forward`):
```python
def install_asym_liger_qwen3_5_moe_loss_bridge(model: nn.Module, *, strict: bool = True) -> bool:
    root = _root_model(model)

    if _is_qwen3_5_moe_conditional_generation(root):
        validated = _validate_liger_lm_head(getattr(root, "lm_head", None), model_label="Qwen3.5-MoE", strict=strict)
        if validated is None:
            return False
        lm_head, weight_source = validated
        root.forward = MethodType(asym_qwen3_5_moe_conditional_lce_forward, root)
        _mark_liger_bridge_installed(root, lm_head, weight_source, "qwen3_5_moe", "conditional_generation")
        return True

    target = _base_causal_lm_model(root)
    model_type = getattr(getattr(target, "config", None), "model_type", None)
    if model_type not in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        if strict:
            raise ValueError("Qwen3.5 Liger loss bridge only supports qwen3_5_moe / qwen3_5_moe_text.")
        return False
    validated = _validate_liger_lm_head(getattr(target, "lm_head", None), model_label="Qwen3.5-MoE", strict=strict)
    if validated is None:
        return False
    lm_head, weight_source = validated
    target.forward = MethodType(asym_qwen3_moe_lce_forward, target)   # text-only fallback; == qwen3_moe shape (mm_token_type_ids rides **kwargs)
    _mark_liger_bridge_installed(target, lm_head, weight_source, model_type, "causal_lm")
    return True
```

**2e — dispatch** in `install_asym_liger_loss_bridge` (add after the llama4 branch):
```python
    if root_type == "qwen3_5_moe" or causal_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        return install_asym_liger_qwen3_5_moe_loss_bridge(model, strict=strict)
```

**2f — `__all__`**: add `asym_qwen3_5_moe_conditional_lce_forward` and
`install_asym_liger_qwen3_5_moe_loss_bridge`.

**No `adapter.py` change.** The Asym branch (≈ line 517) already calls `install_asym_liger_loss_bridge`
generically. The llama4-only normal-branch hook (≈ line 571) stays as-is — qwen3.5 normal/zero3 runs fuse via
the Stage-1 class patch and do **not** need a post-load bridge.

**Validation:**
```bash
"${ENV_PYTHON}" -m pytest -q tests/lf/test_asym_liger_lm_head_bridge.py tests/lf/test_liger_loss_only_qwen3_moe.py
```

---

## Stage 3 — Unit tests (add alongside existing)

- `tests/lf/test_liger_loss_only_qwen3_moe.py`: parametrize `qwen3_5_moe` and `qwen3_5_moe_text` — loss-only
  kwargs exactly `{fused_linear_cross_entropy: True, rest False}`; both in `_LOSS_ONLY_SUPPORTED_MODEL_TYPES`;
  a live `inspect.signature(apply_liger_kernel_to_qwen3_5_moe)` assertion.
- `tests/lf/test_asym_liger_lm_head_bridge.py`: a tiny fake `Qwen3_5MoeForConditionalGeneration`-shaped module
  (`config.model_type="qwen3_5_moe"`, `config.text_config` with `hidden_size`/`vocab_size`, a `.model` returning
  a 1-tuple of hidden states, a frozen `AsymFrozenLinear`-style `lm_head`). Assert:
  - install patches the **instance** only (class forward unchanged),
  - training forward calls `_resolve_liger_lm_head_weight` and does **not** call `self.lm_head(...)` (no logits),
  - `weight_source == "asym_host_staged"` for an Asym `lm_head`, `"normal_parameter"` for a plain one,
  - `bridge_kind == "conditional_generation"`, `model_type == "qwen3_5_moe"`,
  - trainable or biased `lm_head` is rejected.

---

## Stage 4 — E2E validation: run ONLY `ligerloss1`, compare to existing `ligerloss0`

Operational constraints (from project memory — obey strictly):
- **GPUs 0 and 3 only** (1,2 = concurrent weight-offload workstream). `REQUIRE_SM100=1`.
- **Run heavy arms sequentially** — qwen3.5 host RAM is ~665–802 GiB and the membind ceiling is ~958 GiB.
- `NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 NUMACTL_MODE=membind`.
- Write candidates to a **fresh** `OUTPUT_ROOT=profiling_qwen35_liger` — do NOT write into `profiling_qwen35_goal`
  or any canonical `profiling*` dir.
- For the Asym arm, set `ASYM_OFFLOAD_MODULES` to include `lm_head` (match the baseline; use `all`) so the
  candidate reaches `weight_source=asym_host_staged`. Verify against the baseline's `source_profile.json` config.

**Baselines to reuse** (existing `ligerloss0`, `b4_s4096_ga1`):
```bash
ASYM_DIR=${ASYM_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")}
GOAL="${ASYM_DIR}/profiling_qwen35_goal/asym_long_sft_smoke__lora__lf__bf16/qwen3_5-35b-a3b__gpus1__b4_s4096_ga1_w1_s1_r64_a16_drop000"
ASYM_BASE="${GOAL}/asym_cpuadamwds__source__norecomp__polnone__routerwhole__expact1__attnact1__layeract0__layergc1__sdparecomp1__loraafwdhbm__actrecomp0__xunpack0__ligerloss0__gradofftrue__weightofftrue/b4_s4096_ga1"
ZERO_BASE="${GOAL}/zero3_offload__source__recomp__polnone__routerhf__expact0__attnact0__layeract0__layergc0__sdparecomp0__loraafwdhbm__actrecomp0__xunpack0__ligerloss0/b4_s4096_ga1"
test -f "${ASYM_BASE}/source_profile.json" && test -f "${ZERO_BASE}/source_profile.json" || echo "RE-RUN BASELINES (missing)"
```
First, read each baseline's `source_profile.json["config"]` and **match** `WARMUP_STEPS`/`MAX_STEPS`/`SEED`/
`DATASET`/`LORA_*`/`WORKLOADS`/`ASYM_OFFLOAD_MODULES` in the candidate runs below. If a baseline has fewer than
~5 *measured* steps (unstable medians), re-run that backend's `ligerloss0` arm too at `WARMUP_STEPS=5
MAX_STEPS=10` (memory metrics are still valid at low step counts; only timing needs ≥5).

**Candidate run A — `zero3_offload` (clean: no activation offload → unconfounded peak signal). GPU 0:**
```bash
OUTPUT_ROOT="${ASYM_DIR}/profiling_qwen35_liger" \
RUN_NAME=qwen35_liger_zero3 \
MODEL_SPECS='Qwen/Qwen3.5-35B-A3B|1' \
BACKEND_SPECS='zero3_offload|recomp|ligerloss1' \
ASYMM_EXP_ACT_POLICIES='none|false|false|false|false|false' \
WORKLOADS='4096|4|1' \
PROFILERS=both GPU_POOL=0 \
WARMUP_STEPS=5 MAX_STEPS=10 \
PROFILE_MEMORY_BREAKDOWN=true PROFILE_MEMORY_SNAPSHOT=true \
NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 NUMACTL_MODE=membind \
REQUIRE_SM100=1 OVERWRITE=true PLOT=false PREPARE_DATASETS=false \
bash "${ASYM_DIR}/scripts/lf/profile_lora_lf.sh" 2>&1 | tee /tmp/qwen35_liger_zero3.log
```

**Candidate run B — `asym_cpuadamwds` sd1 (matches the GOAL config). Run AFTER A finishes. GPU 3:**
```bash
OUTPUT_ROOT="${ASYM_DIR}/profiling_qwen35_liger" \
RUN_NAME=qwen35_liger_asym \
MODEL_SPECS='Qwen/Qwen3.5-35B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss1' \
ASYMM_EXP_ACT_POLICIES='none|true|true|false|true|true' \
ASYM_OFFLOAD_MODULES=all \
WORKLOADS='4096|4|1' \
PROFILERS=both GPU_POOL=3 \
WARMUP_STEPS=5 MAX_STEPS=10 \
PROFILE_MEMORY_BREAKDOWN=true PROFILE_MEMORY_SNAPSHOT=true \
NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 NUMACTL_MODE=membind \
REQUIRE_SM100=1 OVERWRITE=true PLOT=false PREPARE_DATASETS=false \
bash "${ASYM_DIR}/scripts/lf/profile_lora_lf.sh" 2>&1 | tee /tmp/qwen35_liger_asym.log
```
(If `profile_lora_lf.sh` already defaults `ASYM_OFFLOAD_MODULES`/seed/dataset to the baseline values, drop the
explicit overrides — the rule is *match the baseline*, not hardcode.)

**Locate candidates and compare:**
```bash
ENV_PYTHON=${ENV_PYTHON:-${ASYM_DIR}/.venv/bin/python}
LROOT="${ASYM_DIR}/profiling_qwen35_liger"
ZERO_CAND="$(find "${LROOT}" -type d -path '*zero3_offload*__source__*ligerloss1*' -name 'b*_s*_ga*' -print -quit)"
ASYM_CAND="$(find "${LROOT}" -type d -path '*asym_cpuadamwds*__source__*ligerloss1*' -name 'b*_s*_ga*' -print -quit)"
test -n "${ZERO_CAND}" && test -n "${ASYM_CAND}" || { echo "candidate run dir(s) missing"; exit 1; }

"${ENV_PYTHON}" "${ASYM_DIR}/scripts/lf/compare_liger_loss_profiles.py" \
  --baseline "${ZERO_BASE}" --candidate "${ZERO_CAND}" --backend zero3_offload \
  --baseline-liger-loss ligerloss0 --candidate-liger-loss ligerloss1 \
  --min-peak-drop-gib 10 --min-lm-head-loss-drop-gib 20 \
  --max-step-ratio 1.10 --max-forward-ratio 1.15 --max-backward-ratio 1.15

"${ENV_PYTHON}" "${ASYM_DIR}/scripts/lf/compare_liger_loss_profiles.py" \
  --baseline "${ASYM_BASE}" --candidate "${ASYM_CAND}" --backend asym_cpuadamwds \
  --baseline-liger-loss ligerloss0 --candidate-liger-loss ligerloss1 \
  --min-peak-drop-gib 10 --min-lm-head-loss-drop-gib 20 \
  --max-step-ratio 1.10 --max-forward-ratio 1.15 --max-backward-ratio 1.15
```

**Per-arm artifact checks (before trusting the comparator):**
```bash
rg -n 'Liger loss-only kernel has been applied' /tmp/qwen35_liger_*.log
rg -n 'Asym Liger loss bridge has been installed' /tmp/qwen35_liger_asym.log   # asym arm only
"${ENV_PYTHON}" - "${ASYM_CAND}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]+"/source_profile.json"))
b=p.get("asym_liger_lm_head_bridge",{})
print("bridge:",b)
assert b.get("enabled") is True and b.get("bridge_kind")=="conditional_generation"
assert b.get("weight_source")=="asym_host_staged"   # requires lm_head offloaded
print("loss[:3]:",[s.get("loss") for s in p.get("step_samples",[])[:3]])
PY
```

## Acceptance

Accept `ligerloss1` for qwen3.5 **per backend**:
- `zero3_offload`: comparator `ok: true` — this is the clean proof fused CE fires (no activation-offload
  confounding). Logs show `Liger loss-only kernel has been applied`.
- `asym_cpuadamwds`: comparator `ok: true`, with `asym_liger_lm_head_bridge.enabled=true`,
  `bridge_kind=conditional_generation`, `weight_source=asym_host_staged`; logs show both
  `Liger loss-only kernel has been applied` and `Asym Liger loss bridge has been installed`.
- Both arms: `ligerloss1` loss finite and within ~5% of `ligerloss0` at the same workload (fla non-determinism).

**Confounding fallback (asym only):** the sd1 config offloads activations, so the global *peak* may be set in
backward (offload fetch), not at the logits point. If the asym comparator passes `lm_head/loss-drop ≥ 20 GiB`
but fails `peak-drop ≥ 10 GiB`, fused CE still works — the peak win is masked by offload. In that case run a
clean isolation pair (run BOTH `ligerloss0` and `ligerloss1` fresh at `asym_cpuadamwds|norecomp|
none|false|false|false|false|false`, `ASYM_OFFLOAD_MODULES=all`) and report that comparison. **Do not loosen
the thresholds.** If vocab-scale logits make even `b4_s4096` marginal on `lm_head/loss-drop`, escalate the
isolation pair to `WORKLOADS='8192|8|1'` (run both arms; logits grow 4×).

## Risks / notes
- Loss-only must NOT patch `Qwen3_5MoeExperts`/RMSNorm (would collide with the Asym expert engine) — guaranteed
  by `swiglu=False, rms_norm=False`. Confirm the `ligerloss1` log shows no expert/RMSNorm patch messages.
- The bridge mirrors Liger's repo-local `lce_forward_conditional_generation`. If Liger updates that forward
  (signature/output class), diff and re-sync before profiling.
- zero3 (validation result, not an assumption): the Stage-1 **class patch** (`lce_forward_conditional_generation`)
  reads `self.lm_head.weight` directly, which bypasses the module-forward hook DeepSpeed ZeRO-3 uses to auto-gather
  partitioned params → the weight can arrive 0-numel → non-finite loss. Inherited from the qwen3_moe/llama4 design,
  not introduced here. If the zero3 arm's loss is non-finite or `lm_head/loss`-drop is implausible, gather around
  the fused call (`deepspeed.zero.GatheredParameters(lm_head.weight, ...)`); do **not** apply that to the Asym path,
  which stages explicitly. With `tie_word_embeddings=False` (verified) the head is untied; if a future qwen3.5
  variant ties it, exclude `lm_head` from offload and re-validate.
- This axis is orthogonal to `fix_qwen35.md`. Do not touch the `ligerloss0` offload-vs-recompute runs.

## ✅✅ MEASURED RESULTS (2026-06-21, Qwen3.5-35B-A3B @ 4×4096, ligerloss0 vs ligerloss1)

Code: Stage 1 (LF resolver) + Stage 2 (Asym conditional bridge, `@can_return_tuple`) + Stage 3 (tests, 16/16
pass). Confirmed e2e: model loads as `Qwen3_5MoeForConditionalGeneration`; `Liger loss-only kernel has been
applied.` + (asym) `Asym Liger loss bridge has been installed.` (`bridge_kind=conditional_generation`,
`weight_source=asym_host_staged`, staged 1.017 GiB). Comparator fixed: 0 GPU lm_head/loss rows in the candidate =
elimination (0 bytes), not missing-data error.

| workload | backend | peak GPU HBM off→on | lm_head/loss GPU off→on | step / fwd / bwd | loss off→on | comparator |
|---|---|--:|--:|--:|--:|:--:|
| 4×4096 | asym_cpuadamwds (sd1, 1-step) | 46.08→**4.51** (−41.6) | 43.14→**0.00** (−43.1) | 1.05× / 0.84× / 1.08× | 1.887→1.912 | **ok:true** |
| 4×4096 | zero3_offload (recomp, ms w3/m5) | 50.99→**12.33** (−38.7) | 40.63→**0.50** (−40.1) | 0.96× / 1.00× / 0.96× | 1.479→1.476 | **ok:true** |
| 8×8192 | asym_cpuadamwds (sd1, 1-step) | 92.03→**8.38** (−83.6) | 86.24→**0.00** (−86.2) | **0.51× / 0.36× / 0.54×** | 1.924→1.928 | **ok:true** |
| 8×8192 | zero3_offload (recomp, ms w3/m5) | 99.40→**~19.3** (−80.6) | **−82.0** | 0.99× / 1.05× / 0.98× | 1.523→1.519 | **ok:true** |

Key insight: under heavy activation offload the CE loss is the *entire* HBM peak (43 of 46 GiB on asym @ 4×4096;
86 of 92 GiB @ 8×8192) → fused CE collapses peak ~10× (asym) / ~4× (zero3). It is **fusion+chunking, NOT precision**:
Liger FLCE chunks the `[BT,V]` logits (8×8192 → 64 chunks of 1024 rows, ~0.5 GiB resident vs ~30 GiB full) and
computes grad inline so logits are never saved for backward; the logit GEMM stays bf16 and the softmax/loss is fp32
internally — same as torch CE. Proof: zero3 4×4096 loss off 1.8737 vs on 1.8743 (Δ6e-4, deterministic backend).
At 8×8192 fused CE is also a **speed** win (asym step 0.51× — the 86 GiB logits were that expensive). zero3 timing
only "regresses" at warmup=1 (Triton first-call autotune in the single measured step); at warmup=3/max=5 it is
flat/faster (fwd 1.31×→1.05×). 4×4096 baselines reused from `profiling_qwen35_goal` (reproduce `fix_qwen35.md` V4);
candidates in `profiling_qwen35_liger{,_ms}`; 8×8192 pairs run fresh in `profiling_qwen35_liger_8k{,_ms}` (no
existing baseline). All runs strictly sequential (RAM ceiling). No ZeRO-3 lm_head-partition issue (loss finite).
