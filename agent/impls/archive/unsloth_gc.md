# `zero3_offload | unsloth | ligerloss1` — Unsloth GC (activation offload) as a recompute mode

> Adds **`unsloth`** as a third value of the **recompute axis** in the `BACKEND_SPECS` grammar
> (`backend|recompute|ligerloss`), wiring LlamaFactory's existing `use_unsloth_gc` flag through the
> profiling harness. `zero3_offload` is already a backend and `ligerloss1` is already wired
> end-to-end — **the only net-new work is the `unsloth` recompute token + one LF CLI passthrough.**

## TL;DR — is it "just one flag + some scripts"?

**Yes.**

| piece | status | work |
| --- | --- | --- |
| **`zero3_offload`** backend | already exists (`run_lf_profiled_train.py:583`, JSON `examples/deepspeed/ds_z3_offload_config.json` in LlamaFactory) | none |
| **`ligerloss1`** | already wired end-to-end: spec field → `ENABLE_LIGER_KERNEL` → `--enable_liger_kernel` → `enable_liger_kernel` (LF `model_args.py:137`) | **none** — just put `ligerloss1` in the spec |
| **`unsloth`** | **net-new** | 1 LF flag passthrough (`--use_unsloth_gc`) in the leaf launcher + a new `unsloth` recompute token in the orchestrator(s) |

LlamaFactory already defines `use_unsloth_gc` (`src/llamafactory/.../model_args.py:133`) and implements it
(`model_utils/checkpointing.py:43-77`, installed by `prepare_model_for_training` **only when GC is enabled**).
We are not writing any model code — we only plumb the existing flag.

## Semantics — why `unsloth` belongs in the recompute slot

The middle `BACKEND_SPECS` field is the **recompute axis**. Unsloth GC is gradient checkpointing **plus**
CPU-offload of the one saved layer-boundary activation (the residual stream / `hidden_states` input to each
decoder block — confirmed: HF passes it as `args[0]` at `transformers/.../modeling_layers.py:92`, and Unsloth
offloads exactly that tensor at `checkpointing.py:55`). So it is a third sibling of `recomp`, and the user's
notation `zero3_offload|unsloth|ligerloss1` maps cleanly with `unsloth` in the recompute slot.

| recompute token | LF args emitted | activations |
| --- | --- | --- |
| `norecomp` | `--gradient_checkpointing false --disable_gradient_checkpointing true` | all kept on GPU |
| `recomp` | `--gradient_checkpointing true --disable_gradient_checkpointing false` | boundary kept on **GPU**, interior recomputed |
| **`unsloth`** (new) | `… --disable_gradient_checkpointing false --use_unsloth_gc true` | boundary offloaded to **CPU**, interior recomputed |

`unsloth` ⇒ GC is **on** (so LF installs the checkpoint hook) **and** `use_unsloth_gc=true` (so the hook is the
offloading variant). `use_reentrant_gc` is irrelevant in this mode (the custom autograd Function replaces
`torch.utils.checkpoint`).

## Objective

Run `BACKEND_SPECS="zero3_offload|unsloth|ligerloss1"` on an **MoE** model (so `ligerloss1` is not a no-op — see
caveats) and confirm it trains, then beats vanilla GC on peak HBM.

**Success criterion:** `zero3_offload|unsloth|ligerloss1` peak HBM **<** `zero3_offload|recomp|ligerloss1`.
Metric = `peak_allocated_hbm_bytes`. (Offloading the residual stream to CPU must save GPU memory vs keeping it
resident; step time is expected to be ≥ `recomp` — the PCIe tradeoff, also worth recording.)

### ✅ WIRING SMOKE — PASS (2026-06-22, Qwen3-30B-A3B, b1×s2048, 2 steps, source)
`profile_lora_lf_short.sh` with `WORKLOADS=2048|1|1 PROFILERS=source`, default `BACKEND_SPECS=zero3_offload|unsloth|ligerloss1`:
- exit 0; banner `recompute=unsloth liger_loss=ligerloss1`; `--use_unsloth_gc true` emitted + accepted by LF.
- `Liger loss-only kernel has been applied.` (MoE) + `Gradient checkpointing enabled.` + `DeepSpeed ZeRO3 detected`.
- LoRA `trainable% 9.95`; train_loss **2.582 → 2.012** (mean 2.297), 35.4s/2 steps. No errors/OOM.
- profile records `config.use_unsloth_gc=true`, `activation_recompute=true`, `liger_loss=ligerloss1`, `backend=zero3_offload`.
- peak HBM at this tiny workload: **5.42 GiB alloc / 12.64 GiB reserved** (correctness only — the offload win needs large seq; see below).

### ✅ RESULT — Qwen3-30B-A3B, b8, zero3_offload, ligerloss1 (2026-06-23, PROFILERS=both, `show_metrics.py`)
peak HBM = `step_H` GiB, step time = `step_s` s, RAM = host RSS GiB.
| workload | config | status | step_s | step_H | RAM |
| --- | --- | --- | ---: | ---: | ---: |
| s4096·b8 | norecomp | 🔴 OOM (GPU) | — | — | — |
| s4096·b8 | recomp | OK | 39.3 | 18.6 | 191.1 |
| s4096·b8 | **unsloth** | OK | 39.9 | **12.9** | 191.0 |
| s8192·b8 | recomp | OK | 40.1 | 33.1 | 191.0 |
| s8192·b8 | **unsloth** | OK | 39.8 | **21.6** | 191.0 |
| s48000·b8 | **unsloth** | OK | 77.4 | 108.4 | 252.5 |

**unsloth vs recomp: peak HBM −30.6% @ s4096 (18.6→12.9), −34.7% @ s8192 (33.1→21.6), at ~equal step time
(~40s) and equal host RAM (191 GiB).** The win is all in `bwd_H` (boundary activation on CPU). `norecomp` OOMs
(>184 GiB even at s4096) — GC mandatory; unsloth strictly beats recomp. PCIe-contention step-time penalty did
not materialize at these sizes. Tabulate with `scripts/lf/show_metrics.py profiling_both`.
> Smoke was b1×s2048 where activations are tiny, so unsloth vs recomp HBM is ~equal there. Run the comparison at
> large seq (e.g. the short.sh default `48000|8|1`, or `8192|8|1`) — that's where offloading the residual stream
> to CPU shows a peak-HBM drop vs `recomp`.

---

## Stage 1 — Leaf launcher: pass `--use_unsloth_gc` (the "one flag")

**File:** `scripts/lf/run_lf_lora_sft.sh`

1. **Default** (next to `GRADIENT_CHECKPOINTING` at `:72` / `ENABLE_LIGER_KERNEL` at `:77`):
   ```bash
   USE_UNSLOTH_GC=${USE_UNSLOTH_GC:-false}
   ```

2. **Validate** (mirror the `ENABLE_LIGER_KERNEL` block at `:387-390`):
   ```bash
   case "${USE_UNSLOTH_GC,,}" in
     1|true|yes|y|on)  USE_UNSLOTH_GC=true ;;
     0|false|no|n|off) USE_UNSLOTH_GC=false ;;
     *) echo "USE_UNSLOTH_GC must be true or false, got '${USE_UNSLOTH_GC}'" >&2; exit 2 ;;
   esac
   # Unsloth GC only installs when GC is enabled (LF prepare_model_for_training, checkpointing.py:164).
   if [[ "${USE_UNSLOTH_GC}" == "true" && "${GRADIENT_CHECKPOINTING,,}" != "true" ]]; then
     echo "USE_UNSLOTH_GC=true requires GRADIENT_CHECKPOINTING=true" >&2; exit 2
   fi
   ```

3. **Emit the LF arg** (in the `CMD_ARGS` array, right after `--enable_liger_kernel` at `:1591`):
   ```bash
     --use_unsloth_gc "${USE_UNSLOTH_GC}"
   ```

4. **(Optional) metadata passthrough** for run tagging (mirror `:1812-1813`), so profiles record the mode:
   ```bash
     ASYM_GEMM_LF_CONFIG_USE_UNSLOTH_GC="${USE_UNSLOTH_GC}"
   ```

> The existing GC case at `:1609-1611` is untouched — `unsloth` reuses the `true` branch
> (`--gradient_checkpointing true --disable_gradient_checkpointing false`) and just adds `--use_unsloth_gc true`.

## Stage 2 — Orchestrator: add the `unsloth` recompute token

**Files (all 6 share the `recompute_label` / parse / derivation logic):**
`profile_lora_lf_test1.sh` (zero3 path — primary), `profile_lora_lf_test2.sh`, `profile_lora_lf_short.sh`,
`profile_lora_lf_long.sh`, `profile_ft_lf_long.sh`, `profile_lora_lf_nvme.sh`.

1. **Accept the token** — `recompute_label()` (`test1.sh:615-620`):
   ```bash
   case "${1,,}" in
     norecomp|recomp|unsloth) printf '%s\n' "${1,,}" ;;
     unslothgc|unsloth_gc|unsloth-gc) printf 'unsloth\n' ;;
     norecompute|no_recompute|no-recompute) printf 'norecomp\n' ;;
     recompute) printf 'recomp\n' ;;
     *) die "expected recompute mode norecomp/recomp/unsloth; got '${1}'" ;;
   esac
   ```

2. **Derive both vars** — the per-run setup (`test1.sh:2364-2367`):
   ```bash
   local gradient_checkpointing=false use_unsloth_gc=false
   ...
   case "${recompute}" in
     recomp)  gradient_checkpointing=true ;;
     unsloth) gradient_checkpointing=true; use_unsloth_gc=true ;;
   esac
   ```
   (replaces the single `[[ "${recompute}" == "recomp" ]] && gradient_checkpointing=true` line.)

3. **Export to the leaf** — the env block handed to `run_lf_lora_sft.sh` (next to
   `GRADIENT_CHECKPOINTING="${gradient_checkpointing}"` at `test1.sh:2553`):
   ```bash
     USE_UNSLOTH_GC="${use_unsloth_gc}"
   ```

No change needed to the "both" expansion (`:727`) or the run-dir `path_label` (`:1326`): the label already
includes `${recompute}`, so `unsloth` runs land in their own `__unsloth__` dirs and never collide with `recomp`.

## Stage 3 — Recompute validators must treat `unsloth` like `recomp`

`unsloth` sets `gradient_checkpointing=true`, so the profiled `activation_recompute` will be `true`. Any check
that maps a recompute **label** → expected boolean must include `unsloth` on the `true` side:

- `test1.sh:1085-1091` (`expected_recompute` → `wanted_recompute`):
  ```python
  if expected_recompute in ("recomp", "unsloth"):
      wanted_recompute = "true"
  elif expected_recompute == "norecomp":
      wanted_recompute = "false"
  ```
- The source-ok validator in `run_lf_lora_sft.sh:765-770` compares the boolean `activation_recompute` directly,
  so it passes as-is **provided** the `expected_recompute` it is handed is already the boolean (`true`) and not
  the raw `unsloth` label. Verify the value threaded into it (`test1.sh:805/842/865` → `:880`) resolves to the
  boolean for `unsloth` the same way it does for `recomp`.

## Stage 4 — Smoke (1 step) on a real MoE model

```bash
REPO=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
# MoE model so ligerloss1 actually engages (see caveats). Set GPU_POOL to a free GPU.
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" TEMPLATE=qwen3_nothink GPU_POOL=0 CHECK_TRAINABLE_SURFACE=0 \
  MAX_STEPS=1 WARMUP_STEPS=1 WORKLOADS="2048|1|1" PROFILERS=source \
  BACKEND_SPECS="zero3_offload|unsloth|ligerloss1" \
  OUTPUT_ROOT="$REPO/profiling_unslothgc_smoke" \
  bash scripts/lf/profile_lora_lf_test1.sh 2>&1 | tee /tmp/unslothgc_smoke.log
```
**Gate:** exits 0, a step trains (loss printed). In the log confirm:
- `Gradient checkpointing enabled.` (GC installed)
- `--use_unsloth_gc true` present in the launched LF arg line
- (MoE) `Liger loss-only kernel has been applied.`
- no `USE_UNSLOTH_GC=true requires GRADIENT_CHECKPOINTING=true` / arg-parse errors

To assert the *offloading* variant is actually active (not plain GC), check that the layer's
`_gradient_checkpointing_func.__self__.__name__ == "UnslothGradientCheckpointing"` (the LF unit test
`tests/model/model_utils/test_checkpointing.py::test_unsloth_gradient_checkpointing` does exactly this), or grep
the heartbeat/metadata for the `use_unsloth_gc` passthrough from Stage 1.4.

## Stage 5 — Comparison (sequential, one GPU) + validate the goal

```bash
export CMP=$REPO/profiling_unslothgc ; export M="Qwen/Qwen3-30B-A3B" ; export T=qwen3_nothink
MODEL_SPECS="$M|1" TEMPLATE=$T GPU_POOL=0 OVERWRITE=true CHECK_TRAINABLE_SURFACE=0 PROFILERS=source \
  WORKLOADS="4096|4|1" OUTPUT_ROOT="$CMP" \
  BACKEND_SPECS="zero3_offload|unsloth|ligerloss1,zero3_offload|recomp|ligerloss1,zero3_offload|norecomp|ligerloss1" \
  bash scripts/lf/profile_lora_lf_test1.sh
```
Run strictly one config at a time (CPU-pin/NUMA + PCIe contention pollute peak HBM). Then:

```bash
PY="$REPO/.venv/bin/python"
$PY - "$CMP" <<'PY'
import json,glob,sys; root=sys.argv[1]; g=1024**3; r={}
for f in glob.glob(root+"/**/memory_breakdown_summary.json",recursive=True):
    t=f.split('/')[-3]; peak=json.load(open(f))['peak_allocated_hbm_bytes']/g
    if   '__unsloth__'  in t: r['unsloth']=peak
    elif '__recomp__'   in t: r['recomp']=peak
    elif '__norecomp__' in t: r['norecomp']=peak
for k in ('unsloth','recomp','norecomp'):
    if k in r: print(f"  {k:9s} = {r[k]:.2f} GiB")
u=r.get('unsloth')
if u and 'recomp' in r: print("PASS:", u < r['recomp'])
PY
```
**Pass:** `unsloth` peak HBM `<` `recomp` peak HBM.

## Caveats / operational notes

- **Liger here is LOSS-ONLY and MoE-gated.** `enable_liger_kernel` in this fork only fuses linear cross-entropy
  and only for `model_type ∈ {qwen3_moe, llama4(_text), qwen3_5_moe(_text)}` (LF `model_utils/liger_kernel.py:30`),
  for stages `pt`/`sft`. On a **dense** model `ligerloss1` is a **silent no-op** (`Skipping Liger loss-only for
  unvalidated model_type=…`) — use `ligerloss0` there, or it makes the run label misleading.
- **`use_unsloth` ≠ `use_unsloth_gc`.** We only plumb `use_unsloth_gc` (the checkpointing variant, "no need to
  install unsloth"). Do **not** set the full `use_unsloth` integration — it is a separate flag and is not meant to
  run under DeepSpeed ZeRO-3.
- **PCIe contention is the thing to measure.** With `zero3_offload` (params → CPU) **and** `unsloth`
  (residual stream → CPU), the backward pass shares the one CPU→GPU PCIe link between ZeRO-3's per-layer param
  prefetch and Unsloth's per-layer activation fetch. It saves HBM but may be bandwidth-bound — hence Stage 5
  records step time, not just peak HBM.
- **GC must stay on.** `unsloth` forces `gradient_checkpointing=true`; never combine `unsloth` with `norecomp`.
- **No `use_reentrant_gc` conflict.** The two are an `if/else` in `checkpointing.py:123-126`, not cross-validated:
  when `use_unsloth_gc=true` the Unsloth function is chosen and `use_reentrant_gc` is **silently ignored** (dead).
  Setting both raises no error. The harness never emits `--use_reentrant_gc`, so LF keeps its default (`True`,
  ignored on this path). Do **not** confuse this with `ASYM_EXPERT_GC_USE_REENTRANT`
  (`run_lf_profiled_train.py:643`) — that is the AsymGEMM *expert* recompute knob, unrelated and inert under
  `zero3_offload`.
- **`use_unsloth_gc` × ZeRO-3 is not a heavily-tested combo.** The backward recompute re-invokes each layer's
  `__call__`, so ZeRO-3's hooks re-gather params (same as vanilla checkpointing under ZeRO-3) — mechanically
  sound, but the Stage 4 smoke (loss sane, exits 0) is a required gate before trusting Stage 5 numbers.

## Scope summary

| Area | Change |
| --- | --- |
| `run_lf_lora_sft.sh` | +`USE_UNSLOTH_GC` default/validate/guard; +`--use_unsloth_gc` in `CMD_ARGS` (≈4 small edits) |
| `profile_lora_lf_*.sh` ×6 | `recompute_label()` accepts `unsloth`; derive `use_unsloth_gc`; export `USE_UNSLOTH_GC`; validators map `unsloth`→true |
| Liger (`ligerloss1`) | unchanged (already wired end-to-end) |
| `zero3_offload` backend / DeepSpeed JSON | unchanged |
| LlamaFactory model code | unchanged (`use_unsloth_gc` already implemented) |
