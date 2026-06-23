# Megatron-LM LoRA baseline — staged implementation plan

> Goal: a **comparable single-GPU LoRA baseline** for Qwen3-30B-A3B run on **Megatron-Core** (vendored
> `third_party/megatron-lm`, v0.18.0), to sit next to the existing
> `zero3_offload | unsloth | ligerloss1` run from `scripts/lf/profile_lora_lf_short.sh`
> (`MODEL_SPECS=Qwen/Qwen3-30B-A3B|1`, `WORKLOADS=48000|8|1`, `LORA_RANK=64`, `LORA_ALPHA=16`, bf16).
> We add **LoRA on Megatron** (which Megatron does NOT ship), then enable **activation offload**,
> **layer-granularity offload**, and **optimizer-state offload**, and emit the **same `profile.json`**
> the harness already plots — so peak-HBM / step-time / loss land in the existing comparison.

---

## TL;DR — what exists vs net-new

| piece | status | where |
| --- | --- | --- |
| Megatron-Core 0.18.0 source | vendored | `third_party/megatron-lm` (`megatron/core/package_info.py` → 0.18.0) |
| TransformerEngine 2.17 source | vendored, **not installed** | `third_party/TransformerEngine` (VERSION 2.17.0.dev0) |
| `megatron`, `transformer_engine` importable | **NO** — neither in `.venv` | `python -c "import megatron"` → ModuleNotFound |
| Qwen3-30B-A3B Megatron args | **ready-made** | `megatron-lm/examples/rl/model_configs/qwen3_30b_a3b_moe.sh` (exact arch) |
| MoE grouped-GEMM experts (`TEGroupedMLP`) | exists, needs TE | `megatron/core/transformer/moe/experts.py:172` |
| Activation CPU offload | exists, needs TE | `--cpu-offloading*`, `model_parallel_config.py:368-395` |
| Optimizer-state CPU offload | exists | `--optimizer-cpu-offload`, `optimizer/cpu_offloading/hybrid_optimizer.py` |
| Recompute (selective/full) | exists | `transformer_config.py:472-506` |
| **Base-weight CPU offload (ZeRO-3 `offload_param` analog)** | **DOES NOT EXIST** | TE `cpu_offloading_weights` is a deprecated no-op |
| **LoRA / PEFT on Megatron** | **DOES NOT EXIST** in core | only inference-export refs; training LoRA = net-new |
| HF→Megatron Qwen3-MoE converter | **DOES NOT EXIST** | only `loader_mixtral_hf.py` (Mixtral) under `tools/checkpoint/` |
| `profile.json` schema + plots | exists, HF-Trainer-driven | `scripts/lf/run_lf_profiled_train.py`, `scripts/plotting/*` |

**Net-new work:** (0) build/install TE + megatron-core; (1) stand up Megatron Qwen3-30B-A3B single-GPU;
(2) implement LoRA modules (incl. grouped-expert LoRA as grouped GEMMs); (3) a `profile.json` emitter
reusing the harness recorder; (4) wire offload/recompute flags; (5) the e2e head-to-head. Conversion of
real HF weights (6) and orchestrator wiring (7) are optional.

---

## Architecture decision — standalone Megatron entrypoint (Path B), not the LF/mca bridge (Path A)

There are two ways to get "LoRA on Megatron":

- **Path A — LlamaFactory + `mcore_adapter` (alibaba/ROLL).** LF *does* drive real Megatron training via
  `mcore_adapter` (`LlamaFactory/src/llamafactory/train/mca/`, dispatched from `tuner.py:86-100` when
  `use_mca=True`). `mcore_adapter` already implements LoRA, including **correct grouped-expert LoRA**
  (`LoraColumnParallelLinear`/`LoraRowParallelLinear` over `TE*GroupedLinear`), and its `McaTrainer`
  subclasses HF `Trainer`. **But:** it is an external package (not vendored, not installed), LF's own
  examples are `finetuning_type: full # only support full for now` (`examples/megatron/qwen3_moe_full.yaml`),
  it pins/validates against specific mcore minors (the 0.18 `pg_collection`/`ProcessGroupCollection`
  "M4" change is a live compat risk), and profiling fidelity through mca's wrapped `forward_backward_func`
  is unverified.

- **Path B — vendored Megatron-Core 0.18 directly + hand-rolled LoRA + a `profile.json` emitter.**
  Self-contained in the vendored tree, full control of the train loop → reliable, byte-compatible
  `profile.json`, native offload flags that map 1:1 to the user's "model/layer/act offload", no external
  pinned dependency.

**We take Path B as primary** (matches this repo's vendored/full-control/bespoke-harness philosophy and the
"comparable baseline" requirement). We **reuse Path A's code as reference only** — the grouped-expert LoRA
pattern below is exactly what ROLL `mcore_adapter` and NVIDIA Megatron-Bridge (`bridge.peft.lora`,
`GroupedExpertLinearAdapter`) do. Path A stays documented as a fallback in *Risks*.

**Key de-risking insight:** the headline metric is **peak HBM + step time**, which depend on
architecture / shapes / dtype / offload — **not on weight values**. So Stages 1-5 run with **random-init**
Qwen3-30B-A3B (no converter needed). Real HF weights are only needed for a *loss-curve* sanity check
(Stage 6, optional). This removes the single biggest chunk of risk (HF→Megatron MoE conversion) from the
critical path.

---

## The comparison & success criterion

Baseline to compare against (from `agent/impls/unsloth_gc.md`, Qwen3-30B-A3B, seq48000×b8, 1 GPU):

| config | peak HBM (alloc / reserved) | step time |
| --- | ---: | ---: |
| `zero3_offload\|unsloth\|ligerloss1` | **108.40 / 115.46 GiB** | ~64–72 s/step |

**Deliverable:** a Megatron LoRA run at the **identical** workload (Qwen3-30B-A3B, seq 48000, micro-batch 8,
grad-accum 1, LoRA r=64 α=16, bf16, 1 GPU, fused CE) that writes a `profile.json` the harness ingests,
yielding a side-by-side of `memory.gpu.peak_allocated_hbm_bytes` and step time vs the row above.

**Expected shape of the result (state it, don't hide it):** Megatron keeps the **~60 GB frozen base weights
resident in HBM** (no base offload — see caveat), so its peak HBM is expected to be *higher* on the
param axis but it avoids ZeRO-3's per-layer CPU→GPU param gather, so step time may be *lower*. That tradeoff
(offload-everything-to-fit vs keep-base-resident-and-go-fast) is precisely the comparison worth publishing.

---

## ⚠️ Critical caveat — Megatron has no base-weight CPU offload

`zero3_offload` offloads **base params** to CPU (`offload_param`) and gathers them per layer; that is what
lets it train a 30B model on one small GPU. **Megatron-Core 0.18 + TE 2.17 cannot do this.**
`--cpu-offloading-weights` is a **deprecated no-op** in TE 2.17 (`cpu_offload.py:826-837` warns and offloads
nothing). There is no ZeRO-3 `offload_param` analog. So in a Megatron LoRA run the frozen base
(~30B × 2 B = ~60 GiB bf16) **stays in HBM**.

Consequences:
- The comparison is only apples-to-apples on **activation + optimizer-state** memory, not base params.
  Document this explicitly in the result.
- It must run on a GPU whose HBM holds ~60 GiB base + working set (GB200 class — fine; small GPUs — not).
- "Model offload" in the user's list maps, in Megatron, to **optimizer-state offload** only (tiny under
  LoRA). Call that out rather than implying base offload works.

---

## Offload taxonomy — user's terms → Megatron flags

| user term | ZeRO-3 baseline | Megatron mechanism | flags | exists? |
| --- | --- | --- | --- | --- |
| **act offload** | unsloth boundary→CPU | TE per-layer saved-tensor offload→pinned CPU | `--cpu-offloading --cpu-offloading-activations` | ✅ (needs TE) |
| **layer offload** | per-layer param gather | choose how many layers' activations offload | `--cpu-offloading-num-layers N` (0≤N≤L−1) | ✅ (activations, granularity) |
| **model offload** | `offload_param` (base→CPU) | — base offload absent — *optimizer state* only | `--optimizer-cpu-offload …` | ⚠️ partial (opt state) |
| (recompute, alt to act offload) | `gradient_checkpointing` | selective/full recompute | `--recompute-granularity …` | ✅ (⊥ `--cpu-offloading`) |

Hard constraints (all verified in vendored source — bake into the launcher's arg validation):
- `--cpu-offloading` **⊥** `--recompute-granularity` — `transformer_config.py:1539-1542` raises if both set.
- `--cpu-offloading-num-layers` ∈ `[0, num_layers-1]` — `transformer_config.py:1527-1532`. Cannot offload all.
- `--optimizer-cpu-offload` ⇒ `--use-precision-aware-optimizer` (`arguments.py:1664-1668`)
  ⇒ `--use-distributed-optimizer` + `--optimizer adam` (`optimizer_config.py:425-430`)
  ⇒ AdamW i.e. `--decoupled-weight-decay` (`optimizer/__init__.py:508-510`). Works at DP=1.
- `moe_act` recompute requires `--moe-grouped-gemm` (`transformer_config.py:1603-1606`).
- CPU offload needs TE present, else `transformer_block.py:312-315` asserts.

---

## Target config — Qwen3-30B-A3B on Megatron (single GPU)

Arch confirmed from HF `config.json` (`Qwen3MoeForCausalLM`) and the vendored example
`examples/rl/model_configs/qwen3_30b_a3b_moe.sh`. Single-GPU ⇒ **TP=PP=EP=1**.

```
# --- model (Qwen3-30B-A3B) ---
--num-layers 48  --hidden-size 2048  --ffn-hidden-size 6144
--num-attention-heads 32  --group-query-attention --num-query-groups 4  --kv-channels 128
--normalization RMSNorm  --norm-epsilon 1e-6  --qk-layernorm
--position-embedding-type rope  --rotary-base 1000000  --rotary-percent 1.0  --use-rotary-position-embeddings
--swiglu  --disable-bias-linear  --untie-embeddings-and-output-weights
--vocab-size 151936  --make-vocab-size-divisible-by 128
# --- MoE (128 experts, top-8, grouped GEMM, no shared expert) ---
--num-experts 128  --moe-router-topk 8  --moe-ffn-hidden-size 768
--moe-grouped-gemm  --moe-token-dispatcher-type alltoall  --moe-layer-freq 1
--moe-router-load-balancing-type aux_loss  --moe-aux-loss-coeff 0.001  --moe-router-pre-softmax  # norm_topk_prob=True
# --- engine / single GPU ---
--transformer-impl transformer_engine  --attention-backend flash  --bf16
--tensor-model-parallel-size 1  --pipeline-model-parallel-size 1  --expert-model-parallel-size 1
--tokenizer-type HuggingFaceTokenizer  --tokenizer-model Qwen/Qwen3-30B-A3B
# --- optimizer (match baseline AdamW) ---
--optimizer adam  --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8  --lr 1e-4  --weight-decay 0.0  --clip-grad 1.0
```

> Note `head_dim=128` while `hidden/num_heads = 2048/32 = 64`, so `--kv-channels 128` is **required**
> (head_dim ≠ hidden/heads). `norm_topk_prob=True` ⇒ `--moe-router-pre-softmax` (verify exact flag name in
> `arguments.py` during Stage 1). No shared expert (`shared_expert_intermediate_size=None`).

---

## Stage 0 — Environment bring-up (TransformerEngine + megatron-core)

**Why:** nothing Megatron runs today. TE is a *hard* requirement for grouped-GEMM experts and CPU offload.

**Files/dirs:** `third_party/TransformerEngine`, `third_party/megatron-lm`; new helper
`scripts/megatron/bootstrap_megatron_venv.sh` (mirrors `scripts/lf/bootstrap_lf_venv.sh` style).

**Steps**
1. Detect torch CUDA: `python -c "import torch;print(torch.version.cuda)"`. Prefer a **matching prebuilt TE
   wheel** (`pip install transformer-engine-cu12` or `-cu13` to match) — least pain.
2. Fallback (source build of vendored TE): `MAX_JOBS=… pip install --no-build-isolation
   ./third_party/TransformerEngine` with `CUDA_PATH`/`CUDNN_PATH` set (cuDNN from the venv's
   `nvidia/cudnn`). Heavy C++/CUDA build; expect minutes.
3. Install megatron-core from the vendored tree editable:
   `pip install -e ./third_party/megatron-lm --no-build-isolation` (or `PYTHONPATH=third_party/megatron-lm`
   if only pure-python paths are used; the TE-based path we use needs no megatron C++ kernels because
   `--transformer-impl transformer_engine` routes fused ops through TE).
4. Sanity: HuggingFace tokenizer for Qwen3 already resolvable (cache exists at
   `…/models--Qwen--Qwen3-30B-A3B`).

**Validation (must pass before Stage 1)**
```bash
.venv/bin/python - <<'PY'
import torch, transformer_engine.pytorch as te, megatron.core
from megatron.core.extensions.transformer_engine import TEGroupedLinear, get_cpu_offload_context
from megatron.core.transformer.moe.experts import TEGroupedMLP
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("megatron-core", getattr(megatron.core, "__version__", "?"))
print("TE + megatron import OK")
PY
```

**Risks to watch**
- TE build needs CUDA toolkit + cuDNN matching torch's CUDA; mismatch is the most likely failure. Pin the
  wheel variant to torch's CUDA first.
- megatron-core may pull PyPI deps; use `--no-build-isolation` and the existing torch. If `pip install -e`
  fights versions, fall back to `PYTHONPATH`.
- GB200/ARM: confirm a TE wheel exists for the arch; else source build is mandatory.

---

## Stage 1 — Megatron forward/backward of Qwen3-30B-A3B, single GPU, random init

**Why:** prove the whole Megatron stack (model build, MoE grouped GEMM, attention, optimizer, loss) trains
this model on one GPU **before** adding LoRA. Zero custom code — use stock `pretrain_gpt.py` + `--mock-data`.

**Files:** none new. Use `megatron-lm/pretrain_gpt.py` (`model_provider` line 31, `forward_step`,
`pretrain()` at `megatron/training/training.py:1029`).

**Validation (e2e, stock Megatron)**
```bash
cd third_party/megatron-lm
torchrun --nproc_per_node 1 pretrain_gpt.py \
  --num-layers 48 --hidden-size 2048 --ffn-hidden-size 6144 \
  --num-attention-heads 32 --group-query-attention --num-query-groups 4 --kv-channels 128 \
  --normalization RMSNorm --norm-epsilon 1e-6 --qk-layernorm \
  --position-embedding-type rope --rotary-base 1000000 --use-rotary-position-embeddings \
  --swiglu --disable-bias-linear --untie-embeddings-and-output-weights \
  --num-experts 128 --moe-router-topk 8 --moe-ffn-hidden-size 768 \
  --moe-grouped-gemm --moe-token-dispatcher-type alltoall --moe-layer-freq 1 \
  --transformer-impl transformer_engine --attention-backend flash --bf16 \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 1 \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model Qwen/Qwen3-30B-A3B \
  --vocab-size 151936 --make-vocab-size-divisible-by 128 \
  --seq-length 2048 --max-position-embeddings 2048 \
  --micro-batch-size 1 --global-batch-size 1 --train-iters 5 \
  --optimizer adam --lr 1e-4 --weight-decay 0.0 --clip-grad 1.0 \
  --mock-data --log-interval 1 --no-save-optim --no-load-optim
```
**Accept when:** 5 iters complete, loss is finite, `experts` resolve to `TEGroupedMLP` (not `SequentialMLP`),
HBM holds (≈60 GiB base + working set). If it OOMs at random-init on-GPU build, init on `meta`/CPU then
`.to(cuda)` per layer, or drop to a few layers just to prove the path, then restore 48 for the real runs.

**Risks to watch**
- Exact MoE router flag for `norm_topk_prob` (`--moe-router-pre-softmax` vs `--moe-router-topk-scaling-factor`)
  — grep `arguments.py` and match HF semantics; affects loss correctness only, not memory.
- `--qk-layernorm` granularity: Qwen3 normalizes per-head (head_dim) — confirm Megatron's qk-layernorm
  matches (it should under TE spec). Loss-correctness item, deferrable to Stage 6.
- attention at long seq later needs flash/TE fused (set here already).

---

## Stage 2 — LoRA on Megatron (the net-new core)

**Why:** Megatron core ships **no** training LoRA. Implement it as module wrappers. Single-GPU TP=1 makes the
non-expert adapters trivial (no tensor-parallel comm). Experts use grouped GEMM, so expert LoRA must also be
**grouped GEMMs** — never a Python loop over 128 experts, never 128 small GEMMs.

**New file:** `asym_gemm/megatron/lora.py`.

**Injection targets (verified):**
- Attention (`megatron/core/transformer/attention.py`): `self.linear_qkv`, `self.linear_proj`.
- Dense MLP (`megatron/core/transformer/mlp.py:149`): `self.linear_fc1`, `self.linear_fc2`.
- MoE experts (`megatron/core/transformer/moe/experts.py:172`, `TEGroupedMLP`): `self.linear_fc1`
  (`TEColumnParallelGroupedLinear`), `self.linear_fc2` (`TERowParallelGroupedLinear`), both `num_gemms=128`.
- Skip the router (standard PEFT practice) and norms/embeddings.

Base linears return a **`(output, bias)` tuple** (`ColumnParallelLinear.forward`,
`RowParallelLinear.forward`, and the TE `TELinear.forward` → `(out, bias|None)`). The wrappers must preserve
that contract.

**Pseudocode — non-expert adapter (plain, TP=1):**
```python
# asym_gemm/megatron/lora.py
import torch, torch.nn as nn
from megatron.core.transformer.module import MegatronModule

class LoRAParallelLinear(MegatronModule):
    """Wrap a Column/Row/TE ParallelLinear. base is FROZEN; only A,B train."""
    def __init__(self, base, in_features, out_features, r, alpha, dropout, config):
        super().__init__(config)
        self.base = base                      # requires_grad already set False on its params
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)    # B=0 → adapter starts as identity
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout)
        self.lora_A.to(self.base.weight.dtype); self.lora_B.to(self.base.weight.dtype)

    def forward(self, x, *args, **kwargs):
        out, bias = self.base(x, *args, **kwargs)                       # preserve (out,bias)
        delta = self.lora_B(self.lora_A(self.drop(x))) * self.scaling   # 2 GEMMs, r=64
        return out + delta, bias
```

> For the **fused** `TELayerNormColumnParallelLinear` used as `linear_qkv` (layernorm+linear fused), the LoRA
> input `x` is the *pre-norm* hidden — the same input the base module receives. This is the documented NeMo
> behavior (`TEFusedLoRALinear`); the small norm discrepancy is accepted. (Memory/timing identical either
> way; only loss-curve relevant — note for Stage 6.)

**Pseudocode — grouped-expert adapter (grouped GEMMs, no loop):**
```python
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear, TERowParallelGroupedLinear)

class LoRAGroupedExpert(MegatronModule):
    """Wrap TEGroupedMLP.linear_fc1 or linear_fc2 (a TE*GroupedLinear, num_gemms=E experts).
    LoRA-A and LoRA-B are themselves grouped linears → ALL experts in one grouped GEMM each."""
    def __init__(self, base_grouped, in_features, out_features, r, alpha, config, mode):
        super().__init__(config)
        self.base = base_grouped                  # frozen TE*GroupedLinear, num_gemms=E
        E = self.base.num_gemms                    # 128
        ColG, RowG = TEColumnParallelGroupedLinear, TERowParallelGroupedLinear
        AClass, BClass = (RowG, ColG) if mode == "fc1" else (ColG, RowG)   # mirror base parallel mode
        self.lora_A = AClass(num_gemms=E, input_size=in_features, output_size=r,
                             config=config, init_method=kaiming_, bias=False,
                             skip_bias_add=True, is_expert=True, pg_collection=base_grouped.pg_collection)
        self.lora_B = BClass(num_gemms=E, input_size=r, output_size=out_features,
                             config=config, init_method=zeros_, bias=False,
                             skip_bias_add=True, is_expert=True, pg_collection=base_grouped.pg_collection)
        self.scaling = alpha / r

    def forward(self, x, m_splits, *args, **kwargs):
        # m_splits == tokens_per_expert from the dispatcher — reuse the SAME permutation/offsets
        out, bias = self.base(x, m_splits, *args, **kwargs)        # 1 grouped GEMM
        a, _ = self.lora_A(x, m_splits)                            # 1 grouped GEMM → [tokens, r]
        b, _ = self.lora_B(a, m_splits)                            # 1 grouped GEMM → [tokens, out]
        return out + b * self.scaling, bias
```

> `TEGroupedMLP.forward` (experts.py:630) calls `self.linear_fc1(permuted_states, tokens_per_expert)` then
> `self.linear_fc2(act, tokens_per_expert)` (`apply_module(...)(x, tokens_per_expert)` at :684/:778). Wrapping
> `linear_fc1`/`linear_fc2` means our `forward(x, m_splits)` receives `m_splits = tokens_per_expert` for free.
> This is the canonical ROLL/Megatron-Bridge pattern — **2 extra grouped GEMMs per fc for all 128 experts**,
> one kernel launch each, zero Python loops, no rank-64 micro-GEMM storm.

**Pseudocode — apply + freeze:**
```python
def apply_lora(model, r=64, alpha=16, dropout=0.0,
               targets=("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2")):
    for p in model.parameters(): p.requires_grad_(False)        # freeze base
    for name, mod in list(model.named_modules()):
        for attr in targets:
            child = getattr(mod, attr, None)
            if child is None: continue
            if is_te_grouped(child):                             # MoE expert grouped linear
                wrapped = LoRAGroupedExpert(child, child.in_features_, child.out_features_, r, alpha,
                                            model.config, mode=("fc1" if "fc1" in attr else "fc2"))
            else:                                                # attn/dense parallel linear
                wrapped = LoRAParallelLinear(child, child.input_size, child.output_size, r, alpha,
                                             dropout, model.config)
            setattr(mod, attr, wrapped)
    for n, p in model.named_parameters():                        # re-enable adapters only
        if "lora_A" in n or "lora_B" in n: p.requires_grad_(True)
    return model
```

**Optimizer wiring:** Megatron builds the optimizer over `model.parameters()`; frozen params
(`requires_grad=False`) are excluded from the grad buffer. After `apply_lora`, only LoRA params get grads +
optimizer state. With the distributed optimizer (needed later for offload), confirm only LoRA params land in
the param-and-grad buffer.

**Validation (isolated correctness — toy OK here, it's pure kernel/grad logic):**
```bash
# A) overfit-one-batch: loss must drop sharply over ~30 steps with base frozen
.venv/bin/python scripts/megatron/_lora_smoke.py --layers 2 --experts 8 --seq 256 --steps 30
# asserts inside the smoke script:
#   - count(trainable)/count(total) ~ matches r·(in+out)·#targets; print trainable%
#   - every base param: p.grad is None (or main_grad zero); every lora param: grad finite & nonzero
#   - experts resolve to TEGroupedMLP and LoRA path is LoRAGroupedExpert (assert isinstance)
#   - one optimizer.step changes lora_B away from 0 (adapter becomes active)
```
Then a **mini-e2e** (full 48-layer, 128-expert model, random init, seq 2048, 5 steps) to prove LoRA runs at
real width and the grouped LoRA fires per MoE layer without OOM/loop blowup.

**Risks to watch**
- `TEGroupedLinear` attribute names for in/out size (`in_features_`/`input_size`) and `pg_collection`
  presence — read `transformer_engine.py:1906-2050` and adapt; 0.18 uses `pg_collection`
  (`ProcessGroupCollection`), not legacy `tp_group` (M4 change).
- Grouped LoRA dtype: keep A/B in bf16 to match base and feed TE grouped GEMM (fp32 A/B would force a cast).
- Rank normalization: ROLL/Bridge optionally use `r//topk` for experts (`normalize_moe_lora`). Default to
  full `r` for parity with the LF LoRA unless the LF expert-LoRA impl normalizes — verify against
  `asym_gemm/integrations/lf.py` expert-LoRA to match trainable% if a tight match is wanted.
- If TE grouped LoRA proves fiddly, a correct **fallback** is to force `--no-moe-grouped-gemm` →
  `SequentialMLP` and wrap each `local_experts[i].linear_fc*` with `LoRAParallelLinear` — **but** that is the
  per-expert-loop path the perf rules forbid; use only as a last-resort correctness oracle, never for the
  reported numbers.

---

## Stage 3 — `profile.json` emitter (make the run comparable)

**Why:** the harness plots read a specific `profile.json`. Emit the same fields from a Megatron loop so the
run drops into the existing comparison with no plot changes.

**New files:**
- `scripts/megatron/run_megatron_profiled_train.py` — the entrypoint (model+LoRA+data+optimizer+loop+emit).
- `scripts/megatron/run_megatron_lora_sft.sh` — leaf launcher (sets `ASYM_GEMM_LF_CONFIG_*`, `PROFILE_JSON`,
  output dir; mirrors `scripts/lf/run_lf_lora_sft.sh` env contract).
- `asym_gemm/megatron/data.py` — in-memory SFT iterator from the LF dataset JSON.

**Reuse the recorder (it's mostly pure):** `LFProfileRecorder` (`run_lf_profiled_train.py:2319`) — its
`stage(name, sync=)` (`:2328`) and `report(trace_handle=None)` (`:2618`) only use `torch.cuda.*` +
`/proc` reads; they are **not** HF-Trainer-coupled. `report(trace_handle=None)` degrades gracefully without
trainer-log/heartbeat. Peak HBM is global (`torch.cuda.max_memory_allocated`), correct for any loop.

**Minimal `profile.json` field set the plots/validators actually read** (so they don't crash, and they
compare):
`config{backend,batch_size,seq_len,gradient_accumulation_steps,lora_rank,lora_alpha,precision,
model_name_or_path,dataset,activation_recompute,liger_loss,warmup_steps,max_steps,measure_steps,output_dir}`,
`memory.gpu{peak_allocated_hbm_bytes,peak_reserved_hbm_bytes,reserved_unallocated_bytes}`,
`step{total_milliseconds,rows[]}`, `forward.total_milliseconds`, `backward.total_milliseconds`,
`stage_memory.rows[{name,max_peak_allocated_bytes,max_peak_reserved_bytes,...}]`,
`step_samples.rows[{step,is_warmup,loss,step_milliseconds,peak_allocated_hbm_bytes,...}]`,
`trainer.timing{available,measured_e2e_step_milliseconds,measured_steps}`, `lora{available,trainable_parameters}`.

**Pseudocode — entrypoint loop:**
```python
# scripts/megatron/run_megatron_profiled_train.py
from megatron.training.training import setup_model_and_optimizer, train_step
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.training import get_timers
import importlib; rec = importlib.import_module("scripts.lf.run_lf_profiled_train")

def main():
    initialize_megatron(extra_args_provider=add_lora_args)   # parse Qwen3 args + --lora-rank/--lora-alpha
    model, optimizer, sched = setup_model_and_optimizer(model_provider)  # model_provider applies apply_lora()
    data_iter = build_sft_iterator(args)                     # asym_gemm/megatron/data.py (real tokens, seq=48000)
    fwd_bwd = get_forward_backward_func()                    # PP=1 → no-pipeline variant (forward+backward)

    config = build_config_from_env()                         # ASYM_GEMM_LF_CONFIG_* + args  (backend="megatron")
    recorder = rec.LFProfileRecorder(os.environ["PROFILE_JSON"], config)
    timers = get_timers()
    for it in range(warmup + measure):
        torch.cuda.reset_peak_memory_stats()
        with recorder.stage("step.iteration", sync=True):   # wraps the whole optimizer step
            loss_dict, skipped, grad_norm, *_ = train_step(  # Megatron's step: fwd_bwd + opt.step + sched
                forward_step, data_iter, model, optimizer, sched, args_config, fwd_bwd, iteration=it)
        # split fwd/bwd from Megatron's own timers (keeps distributed-optimizer main_grad plumbing intact)
        fwd_ms = timers('forward-compute').elapsed(reset=False) * 1e3
        bwd_ms = timers('backward-compute').elapsed(reset=False) * 1e3
        recorder.note_step(step=it, is_warmup=it < warmup, loss=float(loss_dict.get("lm loss", nan)),
                           forward_ms=fwd_ms, backward_ms=bwd_ms)   # feeds step_samples + fwd/bwd totals
    report = recorder.report(trace_handle=None)
    Path(os.environ["PROFILE_JSON"]).write_text(json.dumps(report))
```

> Use Megatron's `train_step` (not a hand-rolled `loss.backward()`) so the **distributed optimizer /
> optimizer-cpu-offload `main_grad` plumbing stays correct** for Stage 4. Forward/backward split comes from
> Megatron's built-in `forward-compute`/`backward-compute` timers; peak HBM comes from the recorder's
> `torch.cuda` stats. (`note_step` is a thin helper to add a `step_samples` row + accumulate forward/backward
> totals — implement alongside, or set those fields directly on the `report` dict before dumping.)

**Loss (CE) memory — required for parity with `ligerloss1`:** at seq 48000 × vocab 151936 the logits tensor
is enormous and is *the* peak under offload (per the repo's liger findings). Enable Megatron's fused CE
(`--cross-entropy-loss-fusion`; check `--cross-entropy-fusion-impl te|native` in `arguments.py`) so full
logits are never materialized. Treat "fused CE on" as mandatory for the seq-48000 run; record
`config.liger_loss="ligerloss1"` to label it as the fused-CE comparator.

**Data:** `asym_gemm/megatron/data.py` yields Megatron's expected dict
(`tokens,labels,loss_mask,attention_mask,position_ids`) from the same concat-packed source the LF build
produces (`scripts/lf/build_lf_sft_eval_pair.py`, dataset `asym_long_sft_smoke`). For the memory/throughput
baseline, full-sequence LM loss (`loss_mask=1`) is sufficient and length-matched; SFT prompt-masking is a
loss-curve refinement (Stage 6). Tokenize once with the Qwen3 tokenizer; pack/truncate to exactly
`--seq-length`.

**Output path:** write into the harness layout `${LF_DIR}/saves/asymgemm_smoke/<RUN_ID>/profile.json` with a
`<RUN_ID>` following the `..._megatron_bf16_ctx<seq>_bs<b>_ga<ga>_r<r>_a<a>_steps<n>_...` scheme so
`--collect-existing` plotting and `scripts/lf/show_metrics.py` pick it up (`run_lf_lora_sft.sh:468-472,1136`).

**Validation (e2e at a small-but-real workload):**
```bash
# run the megatron leaf launcher at seq 2048, b1, 3 measure steps; then ingest with existing tooling
PROFILE_JSON=.../profile.json scripts/megatron/run_megatron_lora_sft.sh   # writes profile.json
.venv/bin/python scripts/lf/show_metrics.py .../profile.json              # prints peak HBM / step ms / loss
.venv/bin/python scripts/lf/validate_lf_memory_capacity_schema.py \
    --memory-breakdown-summary .../memory_breakdown_summary.json || true  # only if breakdown emitted
```
**Accept when:** `show_metrics.py` parses it and prints sane peak HBM, step time, and a decreasing loss; the
plotting `--collect-existing` path renders the megatron point next to the LF backends without code changes.

**Risks to watch**
- `LFProfileRecorder` constructor/`report` field names — read `run_lf_profiled_train.py:2271-2703` and emit
  exactly those keys; if `note_step` shape mismatches, write the `step_samples`/`forward`/`backward` fields
  directly into the returned dict before dumping.
- Megatron timer names (`forward-compute`/`backward-compute`) — confirm via `--timing-log-level 2`; if
  absent, fall back to one combined `step.iteration` time and split 50/50 (plots tolerate; note it).
- Don't double-count warmup: mark `is_warmup` and set `measure_steps` so `measured_e2e_step_milliseconds`
  excludes warmup (the baseline used warmup=1).

---

## Stage 4 — Offload + recompute levers

**Why:** the user's "act / layer / model offload". Wire Megatron's native flags through the launcher and
measure each lever's HBM effect at a mid workload before the full run.

**Files:** `scripts/megatron/run_megatron_lora_sft.sh` (arg passthrough + validation),
`run_megatron_profiled_train.py` (`add_lora_args` already forwards Megatron args).

**Levers (add launcher toggles, encode in `<RUN_ID>` like the LF tags):**
1. **act offload** — `MEGATRON_CPU_OFFLOAD=true` → `--cpu-offloading --cpu-offloading-activations
   --cpu-offloading-num-layers ${N}` (+ optional `--cpu-offloading-double-buffering`).
2. **layer offload (granularity)** — `MEGATRON_CPU_OFFLOAD_NUM_LAYERS=N` (0≤N≤47). Sweep N to trade HBM↓ vs
   PCIe time↑. This *is* "layer offload": choose how many layers' activations go to CPU.
3. **optimizer-state offload ("model offload" analog)** — `MEGATRON_OPT_CPU_OFFLOAD=true` →
   `--optimizer-cpu-offload --optimizer-offload-fraction 1.0 --use-precision-aware-optimizer
   --use-distributed-optimizer --decoupled-weight-decay --overlap-cpu-optimizer-d2h-h2d`.
4. **recompute (alt to #1)** — `MEGATRON_RECOMPUTE=selective` → `--recompute-granularity selective
   --recompute-modules moe_act core_attn` (moe_act needs `--moe-grouped-gemm`); or `full` →
   `--recompute-granularity full --recompute-method uniform --recompute-num-layers K`.

Launcher must enforce the constraints (cpu-offloading ⊥ recompute; num-layers ≤ L−1; opt-offload arg chain).

**Validation (e2e, mid workload — NOT toy; offload effects are workload-dependent):**
```bash
# seq 8192, micro-batch 8, 1 GPU, 3 measure steps. Compare peak HBM across levers.
for cfg in "none" "act_off=24" "act_off=47+db" "recompute_selective" "opt_off"; do
  RUN_TAG=$cfg scripts/megatron/run_megatron_lora_sft.sh   # seq8192 b8 ga1 r64 a16, writes profile.json
done
.venv/bin/python scripts/lf/show_metrics.py <each profile.json>
```
**Accept when:** `memory.gpu.peak_allocated_hbm_bytes` **drops monotonically** as `--cpu-offloading-num-layers`
rises (activations leaving HBM); recompute-selective shows an HBM drop with a step-time rise; opt-offload runs
green (HBM delta small under LoRA — expected, document it). Capture all into one table.

**Risks to watch**
- TE V2 offload: `--cpu-offloading-double-buffering` is plumbed but a **no-op** in TE 2.17's default V2 path
  (`cpu_offload.py:698`); the V1 path is behind `NVTE_CPU_OFFLOAD_V1=1`. If double-buffering shows no
  speedup, that's why — note it, don't chase it.
- `--cpu-offloading-num-layers L−1` is the max; you cannot offload the last layer. Sweep up to 47.
- opt-offload requires the full arg chain or Megatron asserts at startup — surface a clear launcher error.
- Base weights stay resident regardless (the caveat) — peak HBM floor ≈ 60 GiB + non-offloaded working set.

---

## Stage 5 — E2E head-to-head at seq 48000 × b8 (the deliverable)

**Why:** the actual comparable baseline against `zero3_offload | unsloth | ligerloss1` (108.40/115.46 GiB,
~64–72 s/step).

**Run (single GPU, base resident, act offload + fused CE, recompute as needed to fit):**
```bash
WORKLOADS="48000|8|1" LORA_RANK=64 LORA_ALPHA=16 PRECISION=bf16 \
MEGATRON_CPU_OFFLOAD=true MEGATRON_CPU_OFFLOAD_NUM_LAYERS=47 \
CROSS_ENTROPY_FUSION=true MAX_STEPS=1 WARMUP_STEPS=1 \
scripts/megatron/run_megatron_lora_sft.sh   # model Qwen/Qwen3-30B-A3B, 1 GPU
```
If it OOMs with act-offload alone (base 60 GiB + 48k activations), switch the activation lever to
**recompute** (`MEGATRON_RECOMPUTE=selective` with `moe_act core_attn`) — recompute and cpu-offload are
mutually exclusive, so pick the one that fits, and report both if both fit.

**Validation / accept:** produce the comparison table by re-running the existing baseline collection and the
new megatron `profile.json` through the standard `show_metrics`/plot path:
```bash
.venv/bin/python scripts/lf/show_metrics.py <megatron profile.json> <unsloth baseline profile.json>
```
| config | peak HBM (alloc / reserved) | step time | notes |
| --- | ---: | ---: | --- |
| `zero3_offload\|unsloth\|ligerloss1` | 108.40 / 115.46 GiB | ~64–72 s | base offloaded to CPU |
| `megatron\|lora\|act-offload\|fusedCE` | _tbd_ | _tbd_ | **base ~60 GiB resident** (no param offload) |

**Accept when:** the megatron run completes (exit 0), `profile.json` records the matched config
(seq=48000, b=8, ga=1, r=64, a=16, bf16, fused CE), and the table is populated. The *finding* (base-resident
HBM vs offload-everything, and the step-time delta) is the result — there is no "must beat" threshold, since
the two occupy different points on the fit-vs-speed curve (state that explicitly).

**Risks to watch**
- Logits OOM at 48k if fused CE isn't actually active — verify the CE path materializes no full
  `[b,s,vocab]` tensor (watch peak during the loss stage).
- flash/TE attention must be the memory-efficient kernel at seq 48000 (set `--attention-backend flash`).
- Throughput fairness: same micro-batch (8), same grad-accum (1), same measured-step definition (exclude
  warmup) as the LF run.

---

## Stage 6 (optional) — HF→Megatron weight conversion for loss-curve parity

**Why:** Stages 1-5 use random init (correct for HBM/throughput). Only do this if a *real* decreasing loss vs
the pretrained model is wanted. **Not on the critical path.**

**New file:** `tools/checkpoint/loader_qwen3_moe_hf.py` (template: `loader_mixtral_hf.py`) or a direct
state-dict mapper `asym_gemm/megatron/convert_qwen3_moe.py`.

**Mapping (HF `Qwen3MoeForCausalLM` → mcore GPTModel):**
- attn: HF `q_proj/k_proj/v_proj` → mcore fused `linear_qkv` (interleave per GQA: 32 q heads, 4 kv groups,
  head_dim 128); `o_proj` → `linear_proj`; Qwen3 `q_norm/k_norm` → mcore qk-layernorm.
- experts: per expert HF `gate_proj`+`up_proj` → mcore `linear_fc1` (concat as `[gate;up]` for SwiGLU),
  HF `down_proj` → `linear_fc2`; **pack all 128 experts into the grouped weight tensors** (`weight0..127` or
  the single stacked param of `TE*GroupedLinear`).
- router HF `gate` → mcore `router.weight`; `embed_tokens`→embedding; `lm_head`→output layer (untied);
  final `norm`→`final_layernorm`.

**Validation:** convert, load, run 20 steps on real `asym_long_sft_smoke` tokens; **loss decreases** and
starts near the pretrained model's NLL (not ~ln(vocab)≈11.9 of random init). Cross-check a few logits against
HF forward on a short prompt (tolerance for bf16 + qk-norm nuance).

**Risks to watch**
- GQA QKV interleave order and Qwen3 per-head q/k-norm are the usual conversion bugs — diff against HF logits.
- Reuse ROLL `mcore_adapter`'s `AutoModel.from_pretrained` mapping or Megatron-Bridge `AutoBridge.from_hf`
  as a *reference* for the exact tensor surgery (don't take them as runtime deps).

---

## Stage 7 (optional) — Wire `megatron` into the orchestrator

**Why:** to launch from `profile_lora_lf_*.sh` like any other backend (consistency with the unsloth_gc add).

**Files (mirror `agent/impls/unsloth_gc.md`'s 3-site pattern):**
- `profile_lora_lf_short.sh` `append_backend_spec()` (`:707-722`) — accept `megatron`.
- `run_lf_lora_sft.sh` backend `case` (`:224-302`) — route `megatron` to `scripts/megatron/run_megatron_lora_sft.sh`
  instead of the LF/torchrun path.
- `run_lf_profiled_train.py` backend string checks (`:560-575`) — recognize `megatron` for the config echo
  (no behavior change; the megatron entrypoint emits the profile).

**Validation:** `BACKEND_SPECS="megatron|norecomp|ligerloss1"` smoke at seq 2048 launches the megatron leaf
and produces a `profile.json` in the standard run dir.

**Risk:** the orchestrator assumes one launcher shape; keep `megatron` on a clean branch in the `case` so it
never falls through to the DeepSpeed/torchrun assembly.

---

## Risks to watch (consolidated)

1. **TE build/install (Stage 0)** — hardest gating step; match torch's CUDA, prefer prebuilt wheel. Blocks
   everything (grouped GEMM + offload need TE).
2. **No base-weight offload** — fundamental gap vs ZeRO-3 `offload_param`; ~60 GiB base stays in HBM. Defines
   the comparison; must be stated, and constrains us to large-HBM GPUs.
3. **Grouped-expert LoRA correctness** — `TE*GroupedLinear` ctor args / `pg_collection` (0.18 M4 change) /
   dtype; validate the 2-grouped-GEMM path fires and obeys the no-loop rule. SequentialMLP wrap is a
   correctness oracle only, never the reported run.
4. **CE memory at seq 48000** — fused CE must be active or logits OOM; it's the peak under offload.
5. **Profiling field parity** — emit exactly the keys the plots read; verify with `show_metrics.py` +
   `validate_lf_memory_capacity_schema.py` before trusting any number.
6. **Megatron router/qk-norm semantics** (`norm_topk_prob`, per-head q/k-norm) — loss-correctness only;
   irrelevant to HBM/throughput, fix in Stage 6 if doing loss parity.
7. **Path A fallback** — if Path B's offload/LoRA integration stalls, ROLL `mcore_adapter` already has
   working LoRA (incl. grouped experts) + an HF-`Trainer` subclass and accepts mcore <0.19; cost is the
   external dep + uncertain profiling-hook fidelity through its wrapped step.
8. **`--cpu-offloading-double-buffering` no-op in TE 2.17 V2** — don't chase a speedup it can't give (or test
   `NVTE_CPU_OFFLOAD_V1=1`).
