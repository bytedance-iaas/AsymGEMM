# Isolated Testing Plan For Qwen3 MoE HBM Reduction

## Goal

Identify, module by module, why the current Qwen3 MoE Asym path does not yet get
below the matching `superoffload_mem|unsloth-off` baseline.

Current scope is attribution only. Do not implement fixes, routed kernels, new memory
placement behavior, or alternate expert execution paths from this document. The only
allowed code changes in this phase are profiling-only instrumentation, artifact labels,
and config recording that do not change LoRA-SFT math or tensor placement.

The active goal is to identify, as concretely as possible, where HBM is coming from for
each operation and module:

```text
phase -> module -> sub-op -> tensor name -> shape -> dtype -> bytes -> live/temporary/saved -> peak owner
```

Future fixes may use the attribution results, but they require a separate implementation
plan after the attribution table is complete.

The core target is:

```text
q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false
```

The primary comparison baseline is:

```text
q3-30b-a3b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false
```

`superoffload_mem|unsloth` is useful context, but the target to beat for this doc is
`superoffload_mem|unsloth-off`.

The intended Asym advantage is stronger than ordinary offload:

```text
SuperOffload:
  stores weights/optimizer/saved activations off HBM,
  but PyTorch/HF kernels still materialize normal HBM operands and outputs.

AsymGEMM:
  streams CPU-resident frozen weights directly into the grouped GEMM path,
  so selected HBM operands/results can be avoided entirely.
```

Therefore the tests below must answer this exact question first:

```text
For each suspicious module, are we failing because a tensor is fundamentally live,
or because current Asym code materializes an avoidable HBM intermediate?
```

## Current Evidence To Explain

Current measured Qwen3 MoE comparison:

```text
Model/workload             Metric                         superoffload-off   asym full-fg   Delta asym
-------------------------  -----------------------------  ----------------  ------------  ----------
qwen3-30b-a3b s80000.b8    peak allocated HBM                     94.40        112.93      +18.53
                           peak reserved HBM                      98.48        137.29      +38.81
                           routed live activation                  7.32         19.53      +12.21
                           routed workspace                       58.02         61.60       +3.58
                           norms live                              0.00          2.44       +2.44
                           attention workspace                     6.55          7.65       +1.11
```

Qwen3-30B-A3B shape constants for `s80000,b8,top_k=8`:

```text
M = batch * seq = 640000
R = M * top_k = 5120000
H = hidden_dim = 2048
I = intermediate_dim = 768

[R,H] bf16 = 5120000 * 2048 * 2 bytes = 19.53 GiB
[R,I] bf16 = 5120000 *  768 * 2 bytes =  7.32 GiB
[M,H] bf16 =  640000 * 2048 * 2 bytes =  2.44 GiB
```

The current Asym live-detail artifact shows:

```text
model.layers.46.mlp.experts.down_base   bfloat16   5120000x2048   19.53 GiB
model.embed_tokens                      bfloat16   8x80000x2048    2.44 GiB
model.layers.46.post_attention_layernorm bfloat16  8x80000x2048    2.44 GiB
```

The matching `superoffload_mem|unsloth-off` live-detail artifact shows:

```text
base_model.model.model.layers.0.mlp.experts.base_layer.act_fn bfloat16 5120000x768 7.32 GiB
base_model.model.model.embed_tokens                           bfloat16 8x80000x2048 2.44 GiB
router tensors                                                small, about 0.20 GiB total
```

Do not summarize this as "router issue." The softmax/top-k router tensors are not the
large problem. The large issue is route-domain hidden-width placement: `[R,H]`.

## Global Test Rules

1. Run experiments one at a time. Never run memory experiments in parallel.
2. Use fresh output roots for every test family and every variant. Do not overwrite or
   reuse old runs.
3. Keep final-comparison runs on `asym_cpuadamwds`, not plain `asym`.
4. Keep `UNSLOTH_GC_OUTER_HBM_EVERY_N=0`.
5. Keep old expert-policy tuple disabled:

```text
none|false|false|false|false|false
```

6. Do not turn on old `ASYMM_EXPERT_ACT_OFFLOAD=true` for target claims.
7. Do not use chunked MLP or route chunking as the claimed win.
8. Always report both allocated and reserved HBM.
9. Interpret allocated HBM before reserved HBM. Reserved includes allocator slack.
10. A result is inconclusive if the resolved config does not match the path label.
11. Fine-grained activation accounting is mandatory for every test, including smoke
    tests. Do not accept a run that only reports aggregate CUDA peak memory.
12. Every test must preserve enough attribution to explain the peak owner, not just
    whether the run fit.

## Required Instrumentation For Every Test

Every isolated test must enable the richest practical activation/memory attribution:

```text
PROFILE_MEMORY_BREAKDOWN=true
PROFILE_MEMORY_BREAKDOWN_INTERVAL=1
PROFILE_MEMORY_BREAKDOWN_MODULES=attention,linear_attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss
```

For full LF runs, collect and archive at minimum:

```text
memory_actual_peak_breakdown.csv
memory_live_activation_details.csv
memory_breakdown.csv
memory_breakdown_summary.json
runtime_counters.csv/json
peak_snapshot_attrib_allblocks.md/csv/json, when available
process_memory.csv
profile.json or source_profile.json
command.txt
train.log
```

For synthetic or expert-only tests, if the standard LF memory-attribution files are not
available, the harness must still record:

```text
max_memory_allocated
max_memory_reserved
reserved_unallocated if available
top live tensors by shape/dtype/module if available
range timing by expert substep
activation offload manager stats
runtime counters proving which path executed
```

Do not use a synthetic result to make a final full-model claim unless the full-model
run also has `memory_live_activation_details.csv` and
`memory_actual_peak_breakdown.csv`.

## Output Directory Discipline

Every run must write to a unique profiling directory. The directory name must include
all knobs that can affect memory placement. At minimum include:

```text
model
seq/batch/grad_accum
backend
recompute label
unsloth vs unsloth-off
moefg flag
routed-kernel flag or stage name
down-scatter-block value
attention flag
norm-placement flag if being tested
UNSLOTH_GC_OUTER_HBM_EVERY_N
grad_offload/weight_offload
timestamp or explicit experiment id
```

Examples:

```text
profiling_isolated_q3moe/stage03_down_fwd_scatter/s8192_b8/current_unfused__exp001/
profiling_isolated_q3moe/stage03_down_fwd_scatter/s8192_b8/fused_down_fwd__exp001/
profiling_isolated_q3moe/stage06_full_routed/s80000_b8/fused_all__exp001/
profiling_isolated_q3moe/stage07_norm/s80000_b8/fused_all_norm_baseline__exp001/
profiling_isolated_q3moe/stage07_norm/s80000_b8/fused_all_norm_candidate__exp001/
```

Never compare two variants if one silently reused the other's output root. If a script
would skip because a profile already exists, either use a new root or explicitly mark
the result as reused and diagnostic-only.

For every run, inspect these artifacts before making a conclusion:

```text
command.txt
train.log
source_profile.json or profile.json
runtime_counters.csv/json
memory_actual_peak_breakdown.csv
memory_live_activation_details.csv
memory_breakdown.csv
memory_breakdown_summary.json
process_memory.csv
asym_cpu_adamw.csv
cpuadam.csv
lora_counters.csv
```

The minimum metrics table for every comparison is:

```text
workload
backend
recompute/config
fwd_s
bwd_s
opt_s
step_s
fwd_H
bwd_H
step_H
peak_allocated_HBM
peak_reserved_HBM
RAM
```

The minimum memory-decomposition table is:

```text
peak allocated HBM
peak reserved HBM
routed live activation
routed workspace
norms live activation
norms workspace
attention workspace
embed live/workspace
allocator reserved-unallocated
```

## Attribution-First Rule

Do not start the routed-kernel implementation plan until the isolated tests identify the
actual peak owners by operation/module/tensor. The routed kernels described later are
future candidates, not existing code and not assumed to be the fix.

The immediate goal is exact attribution:

```text
which phase: forward / backward / optimizer
which module: attention / router / expert MLP / LoRA / norm / embed / allocator
which Qwen3 expert sub-op: gate base, up base, act, down base, down LoRA, down dX, gate dX, up dX, pack_grad
which tensor: shape, dtype, bytes, owner tag, live range
which memory kind: live activation, temporary workspace, saved activation, parameter/LoRA/optimizer, allocator slack
```

Only after the attribution says "this exact tensor is avoidable and material at peak"
should an implementation stage be selected.

## Attribution Test Stages

These tests are the required first work. They can add profiling-only code, but they must
not change LoRA-SFT math or memory placement unless a later implementation stage is
explicitly enabled.

### Attribution Stage A: Full LF Peak Owner Audit

Question:

```text
At the real q3-30b-a3b s80000,b8 peak, which module/component/tensor owns allocated
HBM for superoffload_mem|unsloth-off and current asym_cpuadamwds|recomp-off-full-fg?
```

Code changes:

```text
None expected. Use existing LF memory breakdown first.
If artifacts are missing required detail, do not guess; proceed to Attribution Stage B.
```

Required serial commands:

```bash
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_attribution"

export RUN_NAME="attrA_superoffload_unsloth_off_$(date -u +%Y%m%dT%H%M%SZ)"
export RUNS='q3-30b-a3b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false'
bash scripts/lf/profile_lora_lf_test_source.sh \
  --profile-memory-breakdown true \
  --profile-memory-breakdown-interval 1 \
  --profile-memory-breakdown-modules attention,linear_attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --overwrite false

export RUN_NAME="attrA_asym_fullfg_current_$(date -u +%Y%m%dT%H%M%SZ)"
export RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false'
export ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0
export UNSLOTH_GC_OUTER_HBM_EVERY_N=0
bash scripts/lf/profile_lora_lf_test_source.sh \
  --profile-memory-breakdown true \
  --profile-memory-breakdown-interval 1 \
  --profile-memory-breakdown-modules attention,linear_attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --overwrite false
```

Artifacts to inspect before making any conclusion:

```text
command.txt
source_profile.json
memory_breakdown_summary.json
memory_actual_peak_breakdown.csv
memory_live_activation_details.csv
memory_breakdown.csv
runtime_counters.json/csv
process_memory.csv
train.log
```

Output table required from this stage:

```text
run
peak_allocated_HBM
peak_reserved_HBM
phase_at_peak
top owner component
top owner module_name
top owner tensor shape
top owner dtype
top owner bytes
allocator_reserved_unallocated
```

Pass criteria:

```text
The current gap is explained by concrete owners, not by labels like "router issue".
Every owner larger than 1 GiB has shape/dtype/module attribution or is marked
unattributed and sent to Stage B.
```

### Attribution Stage B: Qwen3 Expert Sub-Op Tensor Ledger

Question:

```text
Inside current qwen3_moe_finegrained.py, exactly which expert sub-op creates the large
route-space tensor: down forward, down backward gather, gate/up dX, LoRA-B, or pack_grad?
```

This stage may require profiling-only code because full LF saved-tensor attribution may
show module-level owners without enough sub-op lifetime detail.

Allowed code changes:

```text
asym_gemm/training/qwen3_moe_finegrained.py
  add a profiling-only ledger enabled by ASYMM_QWEN3_MOE_TENSOR_AUDIT=1
  add record calls immediately after each suspicious tensor allocation or materialization

scripts/lf/run_lf_profiled_train.py
  record ASYMM_QWEN3_MOE_TENSOR_AUDIT in config

scripts/lf/profile_lora_lf_test_source.sh
scripts/lf/profile_lora_lf_test_both.sh
  forward ASYMM_QWEN3_MOE_TENSOR_AUDIT and include audit label in run dir
```

No math or placement behavior may change. The ledger records metadata only.

Ledger pseudocode:

```python
_TENSOR_AUDIT_ENABLED = _env_flag("ASYMM_QWEN3_MOE_TENSOR_AUDIT", False)

def _audit_tensor(layer, phase: str, op: str, name: str, tensor: torch.Tensor | None) -> None:
    if not _TENSOR_AUDIT_ENABLED or tensor is None:
        return
    if not isinstance(tensor, torch.Tensor):
        return
    row = {
        "layer": getattr(layer, "_profile_prefix", ""),
        "phase": phase,
        "op": op,
        "name": name,
        "shape": "x".join(str(int(dim)) for dim in tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "bytes": int(tensor.numel()) * int(tensor.element_size()),
        "requires_grad": bool(tensor.requires_grad),
        "allocated": int(torch.cuda.memory_allocated(tensor.device)) if tensor.is_cuda else 0,
        "reserved": int(torch.cuda.memory_reserved(tensor.device)) if tensor.is_cuda else 0,
    }
    layer.stats.record_qwen3_moe_tensor_audit(row)
```

Minimum audit call sites in `qwen3_moe_finegrained.py`:

```text
forward.gate_base: packed/gate
forward.up_base: up
forward.activation: act_cpu source tensor and staged act
forward.down_lora: down_low_rank, down_delta
forward.down_base: act_stage, output [R,H], scattered [M,H]
backward.scatter_grad: grad_2d [R,H]
backward.down_lora: d_s_down, down_lora_dx
backward.down_base_dx: grad_act [R,I]
backward.activation: grad_up_cpu, grad_gate_cpu, staged gate/up
backward.gate: grad_gate_stage, gate base dX, gate_lora_dx
backward.up: grad_up_stage, up base dX, up_lora_dx
backward.pack_grad: grad_packed [R,H], grad_hidden [M,H]
```

Stats storage pseudocode:

```python
# frozen_linear.py / AsymExecutionStats
qwen3_moe_tensor_audit_rows: list[dict[str, Any]] = field(default_factory=list)

def record_qwen3_moe_tensor_audit(self, row: dict[str, Any]) -> None:
    self.qwen3_moe_tensor_audit_rows.append(row)
```

Validation command:

```bash
python -m py_compile asym_gemm/training/qwen3_moe_finegrained.py scripts/lf/run_lf_profiled_train.py

export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_attribution"
export RUN_NAME="attrB_asym_tensor_audit_s80000_$(date -u +%Y%m%dT%H%M%SZ)"
export RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false'
export ASYMM_QWEN3_MOE_TENSOR_AUDIT=1
export ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0
export UNSLOTH_GC_OUTER_HBM_EVERY_N=0
bash scripts/lf/profile_lora_lf_test_source.sh \
  --profile-memory-breakdown true \
  --profile-memory-breakdown-interval 1 \
  --profile-memory-breakdown-modules attention,linear_attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --overwrite false
```

Required output table:

```text
phase  op                  name              shape          GiB   allocated_at_record  reserved_at_record
-----  ------------------  ----------------  ------------  ----  -------------------  ------------------
fwd    down_base           output            5120000x2048  ...
bwd    scatter_grad        grad_2d           5120000x2048  ...
bwd    gate                gate_base_dx      5120000x2048  ...
bwd    up                  up_base_dx        5120000x2048  ...
bwd    pack_grad           grad_packed       5120000x2048  ...
```

Pass criteria:

```text
Every [R,H], [R,I], [M,H], and LoRA [R,H] tensor has an explicit sub-op owner.
The audit rows agree with memory_live_activation_details.csv on shape and approximate
bytes for tensors present at peak.
If a large peak owner is missing from the ledger, add a ledger call before moving on.
```

### Attribution Stage C: Expert-Only Reproduction

Question:

```text
Can the expert body alone reproduce the same large route-space owners without
attention/norm/loss/full-model noise?
```

Use the existing harness only for localization; it is not a final result:

```text
scripts/testing/profile_qwen3_activation_offload.py
```

Important limitation:

```text
This harness compares current Asym vs old ASYMM_EXPERT_ACT_OFFLOAD. It does not replace
full LF attribution and does not prove the new fine-grained MoE target by itself.
```

Command:

```bash
mkdir -p profiling/qwen3_moe_attribution/expert_only
ASYMM_QWEN3_MOE_TENSOR_AUDIT=1 \
python scripts/testing/profile_qwen3_activation_offload.py \
  --tokens 65536 \
  --top-k 8 \
  --num-experts 128 \
  --hidden-dim 2048 \
  --intermediate-dim 768 \
  --rank 64 \
  --alpha 16 \
  --warmup 1 \
  --iters 3 \
  --profile-breakdown \
  --output-json profiling/qwen3_moe_attribution/expert_only/current_vs_old_expact_s65536.json
```

Scale `--tokens` upward only if memory allows. This is for op localization and timing,
not final HBM claims.

Pass criteria:

```text
The expert-only run shows which Qwen3 expert ranges dominate time and peak allocation.
If it disagrees with the full LF tensor ledger, trust full LF first and explain why the
harness differs before implementing anything.
```

### Attribution Stage D: Toggle Isolation Without New Kernels

Question:

```text
Which existing toggles move the peak owner, without adding new kernels?
```

Only use toggles that already exist. Run one variant at a time.

Variants:

```text
current target:
  ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0

diagnostic block fallback only:
  ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=16
  ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=18

attention isolation:
  ASYMM_ATTN_ACT_OFFLOAD=true
  ASYMM_ATTN_ACT_OFFLOAD=false

no outer HBM:
  UNSLOTH_GC_OUTER_HBM_EVERY_N=0 always
```

Required command pattern:

```bash
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_attribution"
export RUN_NAME="attrD_<variant>_$(date -u +%Y%m%dT%H%M%SZ)"
export RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false'
export ASYMM_QWEN3_MOE_TENSOR_AUDIT=1
export UNSLOTH_GC_OUTER_HBM_EVERY_N=0
bash scripts/lf/profile_lora_lf_test_source.sh \
  --profile-memory-breakdown true \
  --profile-memory-breakdown-interval 1 \
  --profile-memory-breakdown-modules attention,linear_attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --overwrite false
```

Pass criteria:

```text
Each toggle has a before/after peak owner table.
No toggle is accepted as a final solution if it uses blockwise/per-expert looping.
The table must explain whether peak moved from down_base [R,H] to grad_2d [R,H],
gate/up dX [R,H], LoRA [R,H], [R,I], norm [M,H], attention, or allocator slack.
```

### Attribution Gate

Before proposing any implementation stage, write a table like this from real artifacts:

```text
Owner class          Exact op/module             Shape          GiB   Peak?  Existing toggle effect  Avoidable?
-------------------  --------------------------  ------------  ----  -----  ----------------------  ----------
base routed output   forward.down_base.output    [R,H]          ...   yes/no ...
base routed grad     backward.scatter_grad       [R,H]          ...   yes/no ...
base gate/up dX      backward.gate/up.base_dx    [R,H]          ...   yes/no ...
LoRA routed output   forward.down_lora.delta     [R,H]          ...   yes/no ...
norm live            post_attention_layernorm    [M,H]          ...   yes/no ...
attention workspace  attention.*                 ...            ...   yes/no ...
allocator slack      reserved_unallocated        ...            ...   yes/no ...
```

If this table is incomplete, do not implement kernels, placement changes, or alternate
expert paths.

## Future Implementation Appendix: Do Not Execute During Attribution

The active attribution plan stops at the Attribution Gate above. Everything below this
heading is a future-design reference only. Do not execute it in this phase.

These future implementation stages are kept only so a later agent knows the kind of
fixes that might be considered after attribution. They are not evidence, and they are
not part of the current test plan.

Any future implementation change must be gated, default-off, and artifact labeled. With
all future flags left at default, normal LoRA-SFT behavior must use the same execution
path as before those changes, modulo harmless config fields recorded in profiling
output.

These stages are conditional. They start only after the Attribution Gate identifies a
specific avoidable owner at the real workload peak.

Do not use per-expert Python loops, route chunking, or many small GEMM launches as the
target implementation. The route-aware kernels must preserve grouped scheduling and
AsymGEMM CPU-resident weight streaming. The detailed kernel blueprint is:

```text
agent/impls/fused_grouped_scatter.md
```

### Impl Stage 0: Flags, Labels, Counters, No Math Change

Intended code changes:

```text
Add default-off Qwen3-MoE routed-kernel flags.
Propagate and record them through LF profiling scripts and profile config.
Add runtime counters so artifacts prove which path executed.
Do not change any forward/backward math yet.
```

Files and exact scopes:

```text
scripts/lf/profile_lora_lf_test_source.sh
  defaults near existing ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD
  tag helpers near moefg_tag/dscatter_tag
  run_id/run_dir label construction
  run_env forwarding
  ASYM_GEMM_LF_CONFIG_* forwarding

scripts/lf/profile_lora_lf_test_both.sh
  same changes as source script

scripts/lf/run_lf_profiled_train.py
  config payload around existing asymm_qwen3_moe_* fields

asym_gemm/training/frozen_linear.py
  AsymExecutionStats counters only

asym_gemm/training/qwen3_moe_finegrained.py
  flag resolver only; no changed call sites yet
```

Required flags, all default `0` except `ACCUM_DTYPE=fp32`:

```text
ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM=0
ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=0
ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=0
ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=0
ASYMM_QWEN3_MOE_ROUTE_LORA=0
ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32
ASYMM_QWEN3_MOE_ROUTE_KERNEL_DEBUG=0
```

Pseudocode:

```python
def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")

def _route_flag(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is not None:
        return _env_flag(name)
    return _env_flag("ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM", False)

@dataclass(frozen=True)
class RoutedKernelFlags:
    fwd_scatter: bool
    down_dx_gather: bool
    gateup_dx_scatter: bool
    lora: bool
    accum_dtype: str
    debug: bool

def routed_kernel_flags() -> RoutedKernelFlags:
    accum_dtype = os.environ.get("ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE", "fp32").strip().lower()
    if accum_dtype != "fp32":
        raise RuntimeError("Qwen3 MoE routed kernels currently support only fp32 token accumulation")
    return RoutedKernelFlags(
        fwd_scatter=_route_flag("ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER"),
        down_dx_gather=_route_flag("ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER"),
        gateup_dx_scatter=_route_flag("ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER"),
        lora=_env_flag("ASYMM_QWEN3_MOE_ROUTE_LORA", False),
        accum_dtype=accum_dtype,
        debug=_env_flag("ASYMM_QWEN3_MOE_ROUTE_KERNEL_DEBUG", False),
    )
```

Script label pseudocode:

```bash
bool01() {
  case "$(bool_value "$1")" in
    true) printf '1\n' ;;
    false) printf '0\n' ;;
  esac
}

q3rt_tag() {
  printf 'q3rt_fwd%s_gather%s_dx%s_lora%s_acc%s\n' \
    "$(bool01 "${ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER}")" \
    "$(bool01 "${ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER}")" \
    "$(bool01 "${ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER}")" \
    "$(bool01 "${ASYMM_QWEN3_MOE_ROUTE_LORA}")" \
    "${ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE}"
}
```

Validation before moving on:

```bash
python -m py_compile \
  asym_gemm/training/qwen3_moe_finegrained.py \
  scripts/lf/run_lf_profiled_train.py

export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__stage0_flags__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export RUN_NAME="q3moe_${ROUTE_RUN_ID}"
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_routed"
export OVERWRITE=false
export RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false'
export ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM=0
export ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=0
export ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=0
export ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=0
export ASYMM_QWEN3_MOE_ROUTE_LORA=0
export ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32
export ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0
export UNSLOTH_GC_OUTER_HBM_EVERY_N=0
bash scripts/lf/profile_lora_lf_test_source.sh \
  --profile-memory-breakdown true \
  --profile-memory-breakdown-interval 1 \
  --profile-memory-breakdown-modules attention,linear_attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --overwrite false
```

Pass criteria:

```text
resolved config records all route flags as disabled
directory label includes q3rt_fwd0_gather0_dx0_lora0_accfp32
runtime counters show no routed kernels fired
memory/timing matches the previous unfused path within normal run noise
no dense full-fg behavior or old target path changed
```

Risks to watch:

```text
The shell scripts may fail to forward a new env var into training.
The completion checker may not include the new flags and may reuse stale artifacts.
Both are plumbing failures; do not interpret them as kernel results.
```

### Impl Stage 1: Down-Base Forward Scatter-Add

Intended code changes:

```text
Add a Qwen3-specific SM100 BF16 native API that computes down-base forward and
scatter-adds directly into token-space [M,H]. Replace only the base down forward
call site when ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=1.
```

Files and exact scopes:

```text
setup.py
  add csrc/qwen3/qwen3_moe_routed_gemm.cu to asym_gemm._C sources

csrc/qwen3/qwen3_moe_routed_gemm.hpp
csrc/qwen3/qwen3_moe_routed_gemm.cu
  implement qwen3_moe_bf16_down_forward_scatter_add_

csrc/apis/qwen3_moe.hpp
  declare/register pybind wrapper

asym_gemm/training/qwen3_moe_routed_gemm.py
  Python wrapper and shape checks

asym_gemm/training/qwen3_moe_finegrained.py
  _Qwen3MoeFinegrainedFunction.forward down_base branch only

tests/qwen3/test_qwen3_moe_routed_gemm.py
  parity test for down forward scatter-add

scripts/testing/profile_qwen3_moe_routed_gemm.py
  isolated microbench for NCU and allocation checks
```

Native API pseudocode:

```cpp
void qwen3_moe_bf16_down_forward_scatter_add_(
    Tensor act,                 // CUDA bf16 [R,I]
    Tensor weight_cpu,          // CPU pinned bf16 [E,H,I]
    Tensor out_token_fp32,      // CUDA fp32 [M,H], pre-zeroed
    Tensor offsets_i32,
    Tensor experts_i32,
    Tensor token_indices_i64,   // sorted-route-aligned [R]
    Tensor routing_weights,     // sorted-route-aligned [R]
    int64_t list_size,
    bool weighted,
    std::string compiled_dims) {
  check_sm100();
  check(act.is_cuda() && act.dtype() == bf16 && act.is_contiguous());
  check(weight_cpu.device().is_cpu() && weight_cpu.dtype() == bf16);
  check(out_token_fp32.is_cuda() && out_token_fp32.dtype() == float32);
  check(compiled_dims == "nk");

  // Reuse existing grouped AsymGEMM scheduler and CPU-weight tile stream.
  // Fork only the epilogue placement.
  launch_grouped_asym_down_fwd_scatter_add(...);
}
```

Kernel pseudocode:

```cuda
tile = grouped_scheduler.next_tile();  // expert route rows x hidden columns
load_B_cpu_weight_tile_once_to_smem(expert, n_tile, k_tile);
for route_m_tile assigned under that B tile:
    load_contiguous_A_route_tile(act[r, k]) into smem_A;
    umma(acc, smem_A, smem_B);
    for each valid acc element:
        r = route_row_base + local_m;
        h = hidden_col_base + local_n;
        token = token_indices_i64[r];
        scale = weighted ? float(routing_weights[r]) : 1.0f;
        atomicAdd(&out_token_fp32[token * H + h], float(acc[local_m, local_n]) * scale);
```

Python integration pseudocode:

```python
flags = routed_kernel_flags()
act_stage = manager.stage(act_cpu, tag="moe.act_for_down_base")
try:
    if flags.fwd_scatter:
        base_out = torch.zeros((num_tokens, layer.hidden_dim), device=act_stage.device, dtype=torch.float32)
        down_forward_scatter_add_(
            layer.down_base,
            act_stage,
            base_out,
            offsets,
            experts,
            token_indices,
            routing_weights,
            weighted=bool(output_weighted),
        )
        scattered.add_(base_out.to(dtype=scattered.dtype))
        layer.stats.qwen3_moe_routed_route_space_h_tensors_avoided += 1
        del base_out
    else:
        output = _base_forward(layer, layer.down_base, act_stage, offsets, experts, part="down")
        _scatter_routes_add_(scattered, output, token_indices, routing_weights, weighted=bool(output_weighted))
        del output
finally:
    manager.release_stage(act_stage, drop_cache=True)
```

Memory/latency expectation:

```text
s80000 removes one [R,H] bf16 route-space down-base output: about 19.53 GiB.
Adds one short-lived fp32 [M,H] token accumulator: about 4.88 GiB.
Launch count remains one grouped routed base-down launch per layer, not per expert.
NCU must verify B/CPU-weight traffic does not scale with route-row tile count.
```

Validation before moving on:

```bash
python -m pip install -e . --no-build-isolation
python -m pytest tests/qwen3/test_qwen3_moe_routed_gemm.py -q -s -k down_forward_scatter

export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__stage1_fwd_scatter__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export NCU_OUT_ROOT="${PWD}/artifacts/ncu/qwen3_moe_routed/${ROUTE_RUN_ID}"
test ! -e "${NCU_OUT_ROOT}" || { echo "refusing to overwrite ${NCU_OUT_ROOT}" >&2; exit 1; }
mkdir -p "${NCU_OUT_ROOT}"
CUDA_VISIBLE_DEVICES=0 ncu \
  --target-processes all \
  --set roofline \
  --kernel-name regex:qwen3_moe_.*down.*scatter.* \
  --launch-skip 10 \
  --launch-count 20 \
  --export "${NCU_OUT_ROOT}/down_fwd_scatter" \
  python scripts/testing/profile_qwen3_moe_routed_gemm.py \
    --kernel fwd_scatter --M 8192 --top-k 8 --H 2048 --I 768 --E 128 \
    --iters 50 --warmup 10 --weighted 1 --output-dir "${NCU_OUT_ROOT}/microbench"

export RUN_NAME="q3moe_${ROUTE_RUN_ID}"
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_routed"
export RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false'
export ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=0
export ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=0
export ASYMM_QWEN3_MOE_ROUTE_LORA=0
export ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32
export ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0
export UNSLOTH_GC_OUTER_HBM_EVERY_N=0
bash scripts/lf/profile_lora_lf_test_source.sh \
  --profile-memory-breakdown true \
  --profile-memory-breakdown-interval 1 \
  --profile-memory-breakdown-modules attention,linear_attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --overwrite false
```

Pass criteria:

```text
parity against old _base_forward + _scatter_routes_add_
label/config show q3rt_fwd1_gather0_dx0_lora0_accfp32
runtime counters show down-forward routed scatter calls
no down-base [5120000,2048] live owner in the real workload artifacts
no blockwise scatter counters fire
latency is not blockwise/per-expert-loop shaped
```

Risks to watch:

```text
Atomic scatter may be slower than contiguous store plus index_add. If so, keep the
artifact and diagnose with NCU; do not replace it with expert loops.
The fp32 [M,H] scratch can move the peak if it overlaps with LoRA tensors.
```

### Impl Stage 2: Down-Base Backward Gather-Left

Intended code changes:

```text
Add a native API where token-space grad_output [M,H] is used as a virtual gathered
left operand for down-base dX. This removes base-owned grad_routes [R,H].
```

Files and exact scopes:

```text
csrc/qwen3/qwen3_moe_routed_gemm.{hpp,cu}
  implement qwen3_moe_bf16_down_dx_gather_left

csrc/apis/qwen3_moe.hpp
  register wrapper

asym_gemm/training/qwen3_moe_routed_gemm.py
  down_dx_gather_left wrapper

asym_gemm/training/qwen3_moe_finegrained.py
  _Qwen3MoeFinegrainedFunction.backward down_base_dx branch only

tests/qwen3/test_qwen3_moe_routed_gemm.py
  parity test for down_dx_gather

scripts/testing/profile_qwen3_moe_routed_gemm.py
  microbench --kernel down_dx_gather
```

Native API pseudocode:

```cpp
Tensor qwen3_moe_bf16_down_dx_gather_left(
    Tensor grad_token,          // CUDA bf16 [M,H]
    Tensor weight_cpu,          // CPU pinned bf16 down weight
    Tensor offsets_i32,
    Tensor experts_i32,
    Tensor token_indices_i64,
    Tensor routing_weights,
    int64_t list_size,
    bool weighted,
    std::string compiled_dims) {
  check_sm100();
  auto grad_act = torch::empty({R, I}, grad_token.options().dtype(torch::kBFloat16));
  launch_grouped_asym_down_dx_gather_left(..., grad_act);
  return grad_act;
}
```

Kernel pseudocode:

```cuda
tile = grouped_scheduler.next_tile();  // route rows x intermediate columns
load_B_cpu_down_weight_transpose_tile_once_to_smem(expert, i_tile, h_tile);
for route_m_tile assigned under that B tile:
    for each A tile element:
        r = route_row_base + local_m;
        h = hidden_k_base + local_k;
        token = token_indices_i64[r];
        scale = weighted ? float(routing_weights[r]) : 1.0f;
        smem_A[local_m, local_k] = bf16(float(grad_token[token, h]) * scale);
    umma(acc, smem_A, smem_B);
    store_contiguous(grad_act[r, i], acc);
```

Python integration pseudocode:

```python
if flags.down_dx_gather:
    grad_token = grad_output.reshape(ctx.num_tokens, layer.hidden_dim)
    if grad_token.dtype != torch.bfloat16:
        grad_token = grad_token.to(torch.bfloat16)
    grad_token = grad_token.contiguous()
    grad_act = down_dx_gather_left(
        layer.down_base,
        grad_token,
        offsets,
        experts,
        token_indices,
        routing_weights,
        weighted=ctx.output_weighted,
    )
else:
    grad_2d = _route_grad_from_tokens(grad_output, token_indices, routing_weights, ...)
    grad_act = _base_dx(layer, layer.down_base, grad_2d, offsets, experts, part="down", ...)

# If LoRA still needs grad_2d, create it in the LoRA block only and label it LoRA-owned.
```

Memory/latency expectation:

```text
Removes base-owned [R,H] grad_routes.
Keeps [R,I] grad_act because activation backward needs it.
Launch count remains one grouped down-dX launch.
Gathered A loads may be less coalesced; NCU must inspect L2 sectors/replay and tensor-core utilization.
```

Validation before moving on:

```bash
python -m pip install -e . --no-build-isolation
python -m pytest tests/qwen3/test_qwen3_moe_routed_gemm.py -q -s -k down_dx_gather

export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__stage2_down_dx_gather__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export NCU_OUT_ROOT="${PWD}/artifacts/ncu/qwen3_moe_routed/${ROUTE_RUN_ID}"
test ! -e "${NCU_OUT_ROOT}" || { echo "refusing to overwrite ${NCU_OUT_ROOT}" >&2; exit 1; }
mkdir -p "${NCU_OUT_ROOT}"
CUDA_VISIBLE_DEVICES=0 ncu \
  --target-processes all \
  --set roofline \
  --kernel-name regex:qwen3_moe_.*down.*gather.* \
  --launch-skip 10 \
  --launch-count 20 \
  --export "${NCU_OUT_ROOT}/down_dx_gather" \
  python scripts/testing/profile_qwen3_moe_routed_gemm.py \
    --kernel down_dx_gather --M 8192 --top-k 8 --H 2048 --I 768 --E 128 \
    --iters 50 --warmup 10 --weighted 1 --output-dir "${NCU_OUT_ROOT}/microbench"

export RUN_NAME="q3moe_${ROUTE_RUN_ID}"
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_routed"
export RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false'
export ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=1
export ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=0
export ASYMM_QWEN3_MOE_ROUTE_LORA=0
export ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32
export ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0
export UNSLOTH_GC_OUTER_HBM_EVERY_N=0
bash scripts/lf/profile_lora_lf_test_source.sh \
  --profile-memory-breakdown true \
  --profile-memory-breakdown-interval 1 \
  --profile-memory-breakdown-modules attention,linear_attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --overwrite false
```

Pass criteria:

```text
parity against old _route_grad_from_tokens + _base_dx
label/config show q3rt_fwd1_gather1_dx0_lora0_accfp32
runtime counters show down-dX gather-left calls
base-owned grad_routes [5120000,2048] is absent
remaining [R,H], if any, is explicitly LoRA-owned or another known owner
```

Risks to watch:

```text
If LoRA backward still uses grad_2d [R,H], the full peak may not move yet.
If grad_output arrives fp32, token-space bf16 cast creates only [M,H], not [R,H].
```

### Impl Stage 3: Gate/Up Base dX Scatter-Add

Intended code changes:

```text
Add a native API that computes gate/up base dX and scatter-adds directly into
token-space grad_hidden [M,H]. This removes base-owned gate_dx/up_dx/grad_packed [R,H].
```

Files and exact scopes:

```text
csrc/qwen3/qwen3_moe_routed_gemm.{hpp,cu}
  implement qwen3_moe_bf16_gateup_dx_scatter_add_

csrc/apis/qwen3_moe.hpp
  register wrapper

asym_gemm/training/qwen3_moe_routed_gemm.py
  gateup_dx_scatter_add_ wrapper

asym_gemm/training/qwen3_moe_finegrained.py
  _Qwen3MoeFinegrainedFunction.backward gate and up base dX branches only

tests/qwen3/test_qwen3_moe_routed_gemm.py
  parity test for gateup_dx_scatter

scripts/testing/profile_qwen3_moe_routed_gemm.py
  microbench --kernel gateup_dx_scatter
```

Native API pseudocode:

```cpp
void qwen3_moe_bf16_gateup_dx_scatter_add_(
    Tensor grad_expert,         // CUDA bf16 [R,I]
    Tensor weight_cpu,          // CPU pinned bf16 gate/up weight
    Tensor grad_hidden_fp32,    // CUDA fp32 [M,H], pre-zeroed
    Tensor offsets_i32,
    Tensor experts_i32,
    Tensor token_indices_i64,
    Tensor routing_weights,
    int64_t list_size,
    bool weighted,
    std::string compiled_dims) {
  check_sm100();
  check(grad_hidden_fp32.dtype() == float32);
  launch_grouped_asym_gateup_dx_scatter_add(...);
}
```

Kernel pseudocode:

```cuda
tile = grouped_scheduler.next_tile();  // route rows x hidden columns
load_B_cpu_gate_or_up_weight_transpose_tile_once_to_smem(expert, h_tile, i_tile);
for route_m_tile assigned under that B tile:
    load_contiguous_A_route_tile(grad_expert[r, i]) into smem_A;
    umma(acc, smem_A, smem_B);
    for each valid acc element:
        r = route_row_base + local_m;
        h = hidden_col_base + local_n;
        token = token_indices_i64[r];
        scale = weighted ? float(routing_weights[r]) : 1.0f;
        atomicAdd(&grad_hidden_fp32[token * H + h], float(acc[local_m, local_n]) * scale);
```

Python integration pseudocode:

```python
if flags.gateup_dx_scatter and ctx.needs_input_grad[0]:
    grad_hidden_accum = torch.zeros((ctx.num_tokens, layer.hidden_dim), device=grad_output.device, dtype=torch.float32)

    gateup_dx_scatter_add_(gate_base, grad_gate_stage, grad_hidden_accum,
                           offsets, experts, token_indices, routing_weights,
                           weighted=ctx.input_weighted)

    gateup_dx_scatter_add_(up_base, grad_up_stage, grad_hidden_accum,
                           offsets, experts, token_indices, routing_weights,
                           weighted=ctx.input_weighted)

    # LoRA dX stays old-path unless flags.lora is enabled by a later artifact gate.
    grad_hidden = grad_hidden_accum.to(dtype=ctx.input_dtype)
    del grad_hidden_accum
else:
    grad_packed = _base_dx(gate_base, grad_gate_stage, ...)
    grad_packed.add_(_base_dx(up_base, grad_up_stage, ...))
    grad_hidden.index_add_(0, token_indices, grad_packed)
```

Memory/latency expectation:

```text
Removes base-owned gate/up [R,H] dX tensors and base-owned grad_packed [R,H].
Uses two grouped launches per layer for gate and up, not per expert.
Uses one fp32 [M,H] token accumulator shared by gate and up.
Atomic scatter collision is the main latency risk; NCU must inspect replay/stall metrics.
```

Validation before moving on:

```bash
python -m pip install -e . --no-build-isolation
python -m pytest tests/qwen3/test_qwen3_moe_routed_gemm.py -q -s -k gateup_dx_scatter

export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__stage3_gateup_dx_scatter__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export NCU_OUT_ROOT="${PWD}/artifacts/ncu/qwen3_moe_routed/${ROUTE_RUN_ID}"
test ! -e "${NCU_OUT_ROOT}" || { echo "refusing to overwrite ${NCU_OUT_ROOT}" >&2; exit 1; }
mkdir -p "${NCU_OUT_ROOT}"
CUDA_VISIBLE_DEVICES=0 ncu \
  --target-processes all \
  --set roofline \
  --kernel-name regex:qwen3_moe_.*gateup.*scatter.* \
  --launch-skip 10 \
  --launch-count 20 \
  --export "${NCU_OUT_ROOT}/gateup_dx_scatter" \
  python scripts/testing/profile_qwen3_moe_routed_gemm.py \
    --kernel gateup_dx_scatter --M 8192 --top-k 8 --H 2048 --I 768 --E 128 \
    --iters 50 --warmup 10 --weighted 1 --output-dir "${NCU_OUT_ROOT}/microbench"

export RUN_NAME="q3moe_${ROUTE_RUN_ID}"
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_routed"
export RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false'
export ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=1
export ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_LORA=0
export ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32
export ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0
export UNSLOTH_GC_OUTER_HBM_EVERY_N=0
bash scripts/lf/profile_lora_lf_test_source.sh \
  --profile-memory-breakdown true \
  --profile-memory-breakdown-interval 1 \
  --profile-memory-breakdown-modules attention,linear_attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --overwrite false
```

Pass criteria:

```text
parity against old _base_dx + index_add path
label/config show q3rt_fwd1_gather1_dx1_lora0_accfp32
runtime counters show gate/up dX scatter calls
base-owned gate/up [R,H] and grad_packed [R,H] are absent
no per-expert launch pattern in NCU
```

Risks to watch:

```text
If base owners disappear but peak is still high, inspect LoRA, [R,I], norm, attention,
and allocator slack before adding more kernels.
```

### Impl Stage 4: Base-Only Real Workload Decision Gate

Intended code changes:

```text
No new code unless Stage 3 validation shows incorrect counters, missing labels, or
old path fallback. This stage decides from real artifacts whether LoRA routed kernels
are needed.
```

Files and scopes:

```text
No kernel/model changes expected.
Possible script-only fixes if artifact labels or resolved config are incomplete.
```

Validation command:

```bash
export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__stage4_superoffload_unsloth_off__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export RUN_NAME="q3moe_${ROUTE_RUN_ID}"
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_routed"
export RUNS='q3-30b-a3b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false'
bash scripts/lf/profile_lora_lf_test_source.sh \
  --profile-memory-breakdown true \
  --profile-memory-breakdown-interval 1 \
  --profile-memory-breakdown-modules attention,linear_attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --overwrite false
```

Compare this baseline to the fresh Stage 3 `q3rt_fwd1_gather1_dx1_lora0_accfp32`
artifact and the existing `superoffload_mem|unsloth` context row if needed.

Decision rules:

```text
If base-owned [R,H] owners are gone and peak beats superoffload_mem|unsloth-off,
stop routed-kernel work and write the comparison table.

If base-owned [R,H] owners are gone but LoRA-owned [R,H] is the remaining peak owner,
proceed to Impl Stage 5 and implement only the necessary LoRA routed helper.

If base-owned [R,H] owners remain, fix Stage 1-3 wiring before adding LoRA code.

If no [R,H] remains but peak is high, isolate [R,I], norm, attention, allocator slack,
or fallback counters before adding code.
```

Risk to watch:

```text
s2048 smoke proves execution only. The decision gate must use s80000,b8 artifacts.
```

### Impl Stage 5: Conditional LoRA Routed Helpers

Intended code changes:

```text
Implement LoRA routed placement only if Stage 4 proves LoRA-owned [R,H] is a material
peak owner. Do not implement this stage preemptively.
```

Files and exact scopes:

```text
csrc/qwen3/qwen3_moe_routed_gemm.{hpp,cu}
  add only the proven-needed helper

csrc/apis/qwen3_moe.hpp
  register only that helper

asym_gemm/training/qwen3_moe_routed_gemm.py
  wrapper for that helper

asym_gemm/training/qwen3_moe_finegrained.py
  only the corresponding LoRA call site

tests/qwen3/test_qwen3_moe_routed_gemm.py
  parity test for that helper
```

Possible helper pseudocode:

```cuda
// Down LoRA-B forward scatter-add, only if down_delta [R,H] is peak owner.
for route row r and hidden h:
    val = dot(down_low_rank[r, q], down_lora_B[expert, h, q]);
    atomicAdd(out_fp32[token_indices[r], h], val * lora_scale * route_weight[r]);

// Gate/up LoRA dX scatter-add, only if gate_lora_dx/up_lora_dx [R,H] is peak owner.
for route row r and hidden h:
    val = dot(dS_gate_or_up[r, q], lora_A[expert, q, h]);
    atomicAdd(grad_hidden_fp32[token_indices[r], h], val * route_weight[r]);
```

Validation:

```bash
python -m pytest tests/qwen3/test_qwen3_moe_routed_gemm.py -q -s -k 'lora and routed'

export ROUTE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__stage5_lora_routed__pid$$__$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export RUN_NAME="q3moe_${ROUTE_RUN_ID}"
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_routed"
export RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false'
export ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=1
export ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_LORA=1
export ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32
export ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0
export UNSLOTH_GC_OUTER_HBM_EVERY_N=0
bash scripts/lf/profile_lora_lf_test_source.sh \
  --profile-memory-breakdown true \
  --profile-memory-breakdown-interval 1 \
  --profile-memory-breakdown-modules attention,linear_attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --overwrite false
```

Pass criteria:

```text
the specific LoRA-owned [R,H] peak owner disappears
base routed counters still fire
no unrelated LoRA-SFT behavior changes with ASYMM_QWEN3_MOE_ROUTE_LORA=0
peak allocated improves on the real workload, not just the microbench
```

Risks to watch:

```text
LoRA rank is small, but LoRA outputs can still be [R,H]. Only optimize what artifacts
show is actually on the peak path.
```

### Impl Stage 6: Final Comparison And Regression Guard

Intended code changes:

```text
No new code unless validation finds missing artifact fields or accidental dense-path
regressions.
```

Required final serial runs:

```bash
export OUTPUT_ROOT="${PWD}/profiling/qwen3_moe_routed_final"

export RUN_NAME="q3moe_final_superoffload_unsloth_$(date -u +%Y%m%dT%H%M%SZ)"
export RUNS='q3-30b-a3b|1 ; superoffload_mem|unsloth|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false'
bash scripts/lf/profile_lora_lf_test_source.sh --profile-memory-breakdown true --profile-memory-breakdown-interval 1 --profile-memory-breakdown-modules attention,linear_attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss --output-root "${OUTPUT_ROOT}" --run-name "${RUN_NAME}" --overwrite false

export RUN_NAME="q3moe_final_superoffload_unsloth_off_$(date -u +%Y%m%dT%H%M%SZ)"
export RUNS='q3-30b-a3b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false'
bash scripts/lf/profile_lora_lf_test_source.sh --profile-memory-breakdown true --profile-memory-breakdown-interval 1 --profile-memory-breakdown-modules attention,linear_attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss --output-root "${OUTPUT_ROOT}" --run-name "${RUN_NAME}" --overwrite false

export RUN_NAME="q3moe_final_asym_routed_$(date -u +%Y%m%dT%H%M%SZ)"
export RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false'
export ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=1
export ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=1
export ASYMM_QWEN3_MOE_ROUTE_LORA="${ASYMM_QWEN3_MOE_ROUTE_LORA:-0}"
export ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32
export ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0
export UNSLOTH_GC_OUTER_HBM_EVERY_N=0
bash scripts/lf/profile_lora_lf_test_source.sh --profile-memory-breakdown true --profile-memory-breakdown-interval 1 --profile-memory-breakdown-modules attention,linear_attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss --output-root "${OUTPUT_ROOT}" --run-name "${RUN_NAME}" --overwrite false
```

Final acceptance:

```text
Compared rows are fresh, unique, serial, and not reused.
Asym row resolves to asym_cpuadamwds, not plain asym.
old ASYMM_EXPERT_ACT_OFFLOAD path is disabled.
UNSLOTH_GC_OUTER_HBM_EVERY_N=0.
ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0.
routed counters prove enabled kernels fired.
base [R,H] owners are gone from memory_live_activation_details.csv.
peak allocated HBM is below superoffload_mem|unsloth-off, or the remaining owner is
named concretely and the implementation is marked incomplete.
Dense Qwen2.5/Qwen3 full-fg smoke is rerun if any shared file outside Qwen3-MoE-specific
call sites changed.
```

Unresolved implementation risks:

```text
fp32 token-space atomics may be slower than acceptable; NCU decides.
Custom epilogue scatter may require nontrivial factoring of the existing SM100 AsymGEMM
contiguous TMA-store path.
If a future optimization needs bf16 atomics or token-owned gather/reduce instead of
scatter-add, that is a separate stage and must not be mixed into Stages 1-3.
```

## Stage 0: Config And Artifact Audit

### Question

Are the compared runs actually the intended systems?

### Required runs/artifacts

Use the existing full workload artifacts first:

```text
superoffload_mem|unsloth-off|ligerloss1 ; s80000,b8
asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; s80000,b8
```

If either artifact is missing, rerun serially with
`scripts/lf/profile_lora_lf_test_source.sh` or
`scripts/lf/profile_lora_lf_test_both.sh`.

### Must verify

For `superoffload_mem|unsloth-off`:

```text
backend = superoffload_mem
recompute/source = unsloth-off
router mode = hf
old expert act flags = false
ligerloss1 = true
CPUAdam/SuperOffload runtime verified
```

For `asym_cpuadamwds|recomp-off-full-fg`:

```text
backend = asym_cpuadamwds
recomp_off_stage = full-fg
qwen3_moe_finegrained_offload = true
router mode = whole
ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD = 1
ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS = 0 unless explicitly testing fallback
ASYMM_ATTN_ACT_OFFLOAD = true only if this is the intended full-fg target
ASYMM_EXPERT_ACT_OFFLOAD = false
UNSLOTH_GC_OUTER_HBM_EVERY_N = 0
grad_offload = true
weight_offload = true
```

### Expected result

The current valid comparison should reproduce approximately:

```text
superoffload-off allocated/reserved:  94.40 /  98.48 GiB
asym full-fg allocated/reserved:     112.93 / 137.29 GiB
```

### Failure interpretation

If these do not match, do not reason about kernels yet. First check stale artifact,
wrong output root, wrong recompute label, missing env forwarding, or old expert path
accidentally enabled.

## Stage 1: Static Shape Accounting

### Question

Which tensors are large enough to explain the gap?

### Test design

Before running code, compute the exact byte sizes for the workload:

```text
R = batch * seq * top_k
M = batch * seq
[R,H], [R,I], [R,2I], [M,H], router logits [M,E], router top-k tensors [M,top_k]
```

### Expected result

For Qwen3-30B-A3B at `s80000,b8,top_k=8`:

```text
[R,H]  = 19.53 GiB
[R,I]  =  7.32 GiB
[R,2I] = 14.65 GiB
[M,H]  =  2.44 GiB
router logits/top-k are small relative to routed expert tensors
```

### What this answers

This prevents wrong conclusions. A 19.53 GiB live tensor is almost exactly one
`[R,H]`. A 7.32 GiB live tensor is almost exactly one `[R,I]`. A 2.44 GiB live tensor
is almost exactly one `[M,H]`.

## Stage 2: Expert-Only Current Path Test

### Question

Inside Qwen3 experts alone, which HBM tensors does current Asym materialize before any
full-model attention/norm/loss noise?

### Test design

Use the existing expert-only harness where possible:

```text
scripts/testing/profile_qwen3_activation_offload.py
```

If this harness cannot select the new `ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD` path
directly, add a new focused harness during implementation. Do not infer the new
fine-grained path from the old `ASYMM_EXPERT_ACT_OFFLOAD` path.

Recommended synthetic shape:

```text
tokens = 8192 first, then larger if memory allows
top_k = 8
num_experts = 128
hidden_dim = 2048
intermediate_dim = 768
rank = 64
alpha = 16
dtype = bf16
```

Variants:

```text
current fine-grained Asym MoE, no block scatter, no fused routed kernels
block-scatter fallback diagnostic, if available
future fused routed kernels, once implemented
```

### Expected result

Current un-fused path should show a route-domain hidden tensor or a peak consistent
with `[R,H]`.

Block-scatter fallback should reduce or remove full `[R,H]`, but may increase calls and
latency. This is diagnostic only, not the target design.

Fused routed kernels should show no live or peak-attributed `[R,H]` tensor.

### What this answers

Whether the core problem exists inside the expert body itself, independent of
attention/norm/loss. If expert-only still shows `[R,H]`, full-model tests cannot get
below `superoffload-off` reliably.

## Stage 3: Down Forward Placement Test

### Question

Can Asym stream CPU-resident down weights and scatter directly into `[M,H]` without
creating `[R,H]`?

### Old path under test

Current Python behavior in `qwen3_moe_finegrained.py`:

```text
output = grouped_asym_down(act [R,I], W_down_cpu)  # output [R,H]
scattered.index_add_(token_indices, output)        # final [M,H]
```

### Target path

Future routed kernel:

```text
grouped_asym_down_forward_scatter_add_(
  act [R,I],
  W_down_cpu,
  token_indices [R],
  routing_weights [R],
  out [M,H],
)
```

No `[R,H]` output tensor should be allocated.

### Expected result

At full workload, removing this materialization has an upper-bound allocated-HBM win of
about:

```text
[R,H] = 19.53 GiB
```

The measured full-model win may be smaller if the peak shifts to another operation,
but `memory_live_activation_details.csv` must not contain a `[5120000,2048]` down-base
output.

### Pass criteria

1. Kernel parity against old `output + index_add_`.
2. No live `[R,H]` down output in the expert-only test.
3. Full-model `routed live activation` decreases materially in
   `memory_actual_peak_breakdown.csv`.
4. Runtime counters report routed forward-scatter calls.
5. `memory_live_activation_details.csv` identifies no `[R,H]` down-base live tensor.

### Failure interpretation

If parity passes but `[R,H]` remains live, integration is wrong. If `[R,H]` disappears
but peak allocated does not drop, the peak moved to backward, norm, attention, loss, or
allocator-reserved slack.

## Stage 4: Down Backward Gather-Left Test

### Question

Can backward avoid materializing route-domain hidden gradient `[R,H]`?

### Old path under test

Current un-fused behavior:

```text
grad_2d = grad_output.index_select(0, token_indices)  # [R,H]
grad_act = grouped_asym_down_dx(grad_2d, W_down_cpu)  # [R,I]
```

### Target path

Future routed kernel:

```text
grouped_asym_down_dx_gather_left(
  grad_output_tokens [M,H],
  W_down_cpu,
  token_indices [R],
  routing_weights [R],
  grad_act [R,I],
)
```

No `grad_2d [R,H]` should be allocated.

### Expected result

Expert-only backward peak should lose the route-domain hidden gradient. At full
workload this avoids another possible 19.53 GiB allocation site, though not necessarily
another 19.53 GiB of end-to-end peak if it did not overlap the old peak.

### Pass criteria

1. Kernel parity against `index_select + grouped_asym_down_dx`.
2. `memory_live_activation_details.csv` has no `[R,H] grad_2d` owner.
3. `routed workspace` decreases or the peak owner shifts.
4. The peak-owner shift is recorded in a unique profiling directory with full memory
   attribution.

### Failure interpretation

If down forward is fixed but peak still exceeds `superoffload-off`, this test is the
first backward suspect.

## Stage 5: Gate/Up dX Scatter-Add Test

### Question

Can gate/up backward compute route-domain dX and scatter directly into token gradients,
without creating `[R,H]` intermediates?

### Old path under test

Current un-fused behavior can create route-domain hidden intermediates:

```text
gate_dx or up_dx = grouped_asym_dx(grad_gate/up [R,I], W_gate/up_cpu)  # [R,H]
grad_packed accumulates route-domain hidden gradients                  # [R,H]
grad_hidden.index_add_(token_indices, grad_packed)                     # [M,H]
```

### Target path

Future routed kernel:

```text
grouped_asym_gateup_dx_scatter_add_(
  grad_gate_or_up [R,I],
  W_gate_or_up_cpu,
  token_indices [R],
  routing_weights [R],
  grad_hidden [M,H],
)
```

No `grad_packed`, `gate_dx`, or `up_dx` shaped `[R,H]` should be allocated.

### Expected result

After down-forward and down-backward are fixed, this should remove the remaining
route-domain hidden-width backward placement. If this is not fixed, full-model peak can
still stay above `superoffload-off`.

### Pass criteria

1. Kernel parity against old `grouped_asym_dx + index_add_`.
2. Runtime counters show gate/up routed dX scatter calls.
3. No `[R,H]` gate/up dX tensors in live details.
4. Full-model `routed live activation + routed workspace` drops below the
   `superoffload-off` sum or is clearly explained by `[R,I]` live operands.
5. The comparison includes allocated HBM, reserved HBM, and allocator slack.

## Stage 6: Full Routed-Kernel Expert Integration Test

### Question

After all routed placement kernels are enabled together, does the expert path have any
remaining avoidable `[R,H]` materialization?

### Test design

Run serially:

```text
s8192,b8  smoke
s30000,b8 validation
s80000,b8 final comparison
```

For each size compare:

```text
superoffload_mem|unsloth-off
asym_cpuadamwds|recomp-off-full-fg current/unfused
asym_cpuadamwds|recomp-off-full-fg fused-routed
```

Each row must have its own output directory. Do not put current/unfused and
fused-routed artifacts in the same leaf directory.

### Expected result

For the fused-routed Asym run:

```text
memory_live_activation_details.csv has no [R,H] routed expert tensor
routed live activation is near [R,I] scale, not [R,H] scale
routed workspace is <= current Asym and ideally <= superoffload-off
peak allocated HBM is below current Asym by at least 12 GiB
```

At `s80000,b8`, a reasonable first target is:

```text
current Asym full-fg allocated:      112.93 GiB
remove one [R,H]:                    -19.53 GiB
rough routed-only target:             93.40 GiB
superoffload-off allocated:           94.40 GiB
```

This is a tight target. If the fused routed kernels work but norms remain extra, the
measured result may land around the baseline. If routed plus norm cleanup works, the
target should be below baseline.

## Stage 7: Norm Placement Isolation

### Question

Why does Asym keep an extra `[M,H]` norm live tensor while `superoffload-off` does not?

### Evidence to explain

Current Asym extra:

```text
saved/live norms: +2.44 GiB
```

This is exactly one `[M,H]` tensor at `s80000,b8,H=2048`.

### Test design

After routed `[R,H]` is fixed, run a diagnostic that isolates norm placement:

1. same full model and workload;
2. same routed-kernel setting;
3. change only norm saved-tensor/offload behavior, or add temporary instrumentation to
   identify which norm output is live;
4. inspect `memory_live_activation_details.csv` and autograd saved-tensor attribution.

Do not mix this with attention or routed-kernel changes in the same run.

### Expected result

If the extra norm tensor is avoidable, eliminating it should reduce allocated HBM by up
to:

```text
[M,H] = 2.44 GiB
```

### Pass criteria

1. The specific live norm tensor disappears or moves off HBM.
2. Peak allocated decreases by about 2.44 GiB if the norm tensor was on the peak path.
3. No change to routed expert counters.
4. The norm baseline and norm candidate are separate profiling directories with the
   same routed-kernel and attention settings.

### Failure interpretation

If the norm tensor stays but peak is already below `superoffload-off`, this can be a
later optimization. If it blocks beating the baseline after routed kernels are fixed,
it becomes the next required fix.

## Stage 8: Attention Placement Isolation

### Question

Is the attention path a real remaining blocker or just a small workspace difference?

### Evidence to explain

Current Asym extra:

```text
attention workspace: +1.11 GiB
```

### Test design

Run an ablation where only attention activation placement changes:

```text
ASYMM_ATTN_ACT_OFFLOAD=true
ASYMM_ATTN_ACT_OFFLOAD=false
```

Keep all routed settings fixed. This is diagnostic. The final target composition may
still require the dense full-fg attention setting.

### Expected result

The possible win is small, around 1 GiB. It should not be treated as the main reason
for missing the baseline.

### Pass criteria

1. Only attention counters change.
2. Routed/norm live details remain unchanged.
3. Peak allocated movement is consistent with the attention workspace delta.
4. The attention-on and attention-off runs are separate profiling directories.

## Stage 9: Allocator Reserved Isolation

### Question

Is high reserved HBM real tensor memory or PyTorch allocator slack after large
allocations?

### Evidence to explain

Current Asym extra:

```text
allocated HBM gap: +18.53 GiB
reserved HBM gap:  +38.81 GiB
allocator slack:   +20.28 GiB
```

### Test design

For every routed-kernel integration run, compare:

```text
actual_peak_allocated_hbm_bytes
actual_peak_reserved_hbm_bytes
reserved_unallocated_bytes
memory_live_activation_details.csv
```

Run in a fresh process. Do not compare reserved HBM from a process that ran multiple
variants sequentially unless the script explicitly resets and isolates the process.

### Expected result

If fused routed kernels prevent large `[R,H]` allocations from ever happening, reserved
HBM should also improve. If allocated drops but reserved remains high, the kernel is
still useful; the remaining issue is allocator behavior or an earlier large allocation.

### Pass criteria

1. Allocated HBM drops first.
2. Reserved HBM either drops too or the delta is explained by
   `reserved_unallocated_bytes`.
3. No final claim uses reserved HBM alone as proof that the tensor placement failed.

## Stage 10: LoRA And Optimizer Isolation

### Question

Is MoE missing the dense-model win because LoRA/optimizer tensors are still in HBM?

### Evidence

Dense wins were partly because Asym removed large LoRA-attributed HBM:

```text
qwen2.5 dense: lora live+workspace delta = -53.39 GiB
qwen3 dense:   lora live+workspace delta = -40.19 GiB
```

In the Qwen3 MoE `s80000,b8` peak breakdown, the dominant delta is not labeled LoRA;
it is routed experts. This means dense-style LoRA savings are not the obvious remaining
MoE blocker at the actual peak.

### Test design

Inspect:

```text
lora_counters.csv
asym_cpu_adamw.csv
optimizer_memory_preflight.csv
memory_actual_peak_breakdown.csv
```

Only implement MoE LoRA routed kernels if artifacts show LoRA contributes materially to
the peak after base routed `[R,H]` tensors are removed.

### Expected result

Before routed base kernels are fixed, LoRA should not be the main focus. After routed
base kernels are fixed, rerun the decomposition. If LoRA becomes visible at peak, add
LoRA routed scatter/gather tests then.

## Final Full-Workload Gate

No design is considered successful until the real comparison is run serially:

```text
q3-30b-a3b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false
q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false
```

Required final table:

```text
Model/workload
Backend
Config
fwd_s
bwd_s
opt_s
step_s
fwd_H
bwd_H
step_H
peak allocated HBM
peak reserved HBM
RAM
routed live activation
routed workspace
norms live activation
attention workspace
allocator reserved-unallocated
```

Required final artifact audit:

```text
each compared row has a unique output directory
each compared row has memory_live_activation_details.csv
each compared row has memory_actual_peak_breakdown.csv
each compared row has runtime counters
each compared row has command/config evidence
no row is a stale/reused artifact unless explicitly labeled diagnostic-only
```

Expected successful end state:

```text
1. no [R,H] routed tensors in memory_live_activation_details.csv;
2. routed live activation is [R,I]-scale or lower, not [R,H]-scale;
3. peak allocated HBM is below superoffload_mem|unsloth-off;
4. any remaining reserved-HBM excess is explained by allocator slack;
5. no old expert activation-offload path was used;
6. dense full-fg artifacts remain unchanged or are revalidated if shared code changed.
```

## Interpretation Matrix

Use this matrix before making conclusions.

```text
Observation                                      Meaning
-----------------------------------------------  -------------------------------------------
[R,H] appears in live details                    identify exact owner before proposing fixes
[R,H] appears only in tensor audit               temporary route-space tensor; inspect peak timing
allocated lower, reserved still high             allocator slack or earlier large allocation
routed live near 7.32 GiB                        expected [R,I] scale, not the main blocker
norm live exactly 2.44 GiB                       one [M,H] norm tensor, next cleanup target
attention delta around 1 GiB                     secondary, not primary
router tensors around 0.2 GiB                    not the memory blocker
LoRA absent from peak                            do not prioritize LoRA changes
LoRA appears at peak                             identify exact LoRA op before proposing fixes
```

## Active Attribution Roadmap

1. Audit current full-workload artifacts.
2. Add profiling-only tensor ledger if existing artifacts are not detailed enough.
3. Produce the Qwen3 expert sub-op table for every large `[R,H]`, `[R,I]`, and `[M,H]`
   tensor.
4. Use expert-only runs only to localize expert-body behavior, not for final claims.
5. Run existing-toggle isolation only to see how the peak owner moves.
6. Write the Attribution Gate table from real `s80000,b8` artifacts.
7. Stop. Do not implement fixes from this document.

The central attribution question is:

```text
which exact current operation creates each HBM owner, and is it actually live at peak?
```
