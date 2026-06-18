# Qwen3.5 AsymGEMM/LF Implementation Plan

This plan is based on the current Qwen3, Qwen3.5, Llama4, LF integration,
and profiling code in this checkout, plus current Transformers Qwen3.5 MoE
docs/source. The external architecture facts that matter are: Qwen3.5 MoE uses
a 3:1 Gated DeltaNet/full-attention backbone and replaces dense FFNs with MoE
blocks containing routed experts plus one shared expert, with default text
config fields such as `num_experts=256`, `num_experts_per_tok=8`,
`moe_intermediate_size=512`, and `shared_expert_intermediate_size=512`.
See:

- https://huggingface.co/docs/transformers/en/model_doc/qwen3_5_moe
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py

The local vendored source confirms the concrete block shape:

```python
class Qwen3_5MoeSparseMoeBlock(nn.Module):
    gate: Qwen3_5MoeTopKRouter
    experts: Qwen3_5MoeExperts
    shared_expert: Qwen3_5MoeMLP  # gate_proj, up_proj, down_proj
    shared_expert_gate: nn.Linear(hidden_size, 1, bias=False)

    def forward(hidden_states):
        flat = hidden_states.view(-1, hidden)
        shared = shared_expert(flat)
        _, weights, indices = gate(flat)
        routed = experts(flat, indices, weights)
        shared = sigmoid(shared_expert_gate(flat)) * shared
        return (routed + shared).view(batch, seq, hidden)
```

Current repo facts:

- Qwen3 routed experts are comprehensively owned by
  `asym_gemm/training/qwen3_moe.py`. `AsymQwen3Experts` supports packed
  routed expert base offload, LoRA banks, activation offload/backfetch,
  expert recompute, input-scaled routed forward, and CPUAdamW trainable
  expert-LoRA weight offload.
- Llama4 adds the missing pattern for shared branches. `AsymLlama4Moe` in
  `asym_gemm/training/llama4_moe.py` owns the whole MoE block and can wrap
  `shared_expert` with `AsymLlama4SharedMLP` from
  `asym_gemm/training/llama4_shared_mlp.py`, including shared-MLP activation
  offload/backfetch.
- Qwen3.5 already has `AsymQwen35MoeBlock` in
  `asym_gemm/training/qwen35_moe.py`. It wraps the routed experts via
  `wrap_qwen3_experts` and preserves the Qwen3.5 shared branch, but the shared
  branch is not first-class. It is only reached later by generic dense wrapping
  in `asym_gemm/integrations/lf.py`.
- Qwen3.5 decoder layers are not Qwen3 decoder layers with only a different MLP.
  They are hybrid: most layers have `linear_attn` (`Qwen3_5MoeGatedDeltaNet`)
  and the rest have `self_attn`. The current LF decoder-layer matcher must cover
  both shapes, otherwise `ASYMM_LAYER_ACT_OFFLOAD=true` and `gc-layer` only cover
  the full-attention subset.
- Verified dependency state at plan time: `python`,
  `third_party/AsymGEMM/.venv/bin/python`,
  `third_party/LlamaFactory/.venv/bin/python`, and
  `third_party/LlamaFactory-fa4/.conda-lf-fa4/bin/python` all lack importable
  `fla` and `causal_conv1d`. The profile scripts default to
  `third_party/AsymGEMM/.venv/bin/python`, so current Qwen3.5 runs use the
  Transformers torch fallback for Gated DeltaNet unless the environment changes.
  The mandatory pre-stage below must install and verify the package-index
  `flash-linear-attention` path before any profiling baseline or implementation
  work proceeds. In this checkout, Transformers 5.6.0 exposes
  `is_flash_linear_attention_available` from
  `transformers.utils.import_utils`, not from `transformers.utils`.
- The Gated DeltaNet linear leaves are `in_proj_qkv`, `in_proj_z`, `in_proj_b`,
  `in_proj_a`, and `out_proj`. Verified locally with a tiny
  `Qwen3_5MoeTextModel`: LlamaFactory's `find_all_linear_modules` logic includes
  all five leaf names under `lora_target=all`. Verified against current
  `classify_lf_component`: those same `layers.*.linear_attn.*` leaves classify
  as `other` and are not selected by `ASYM_OFFLOAD_MODULES=all`. Therefore,
  current code gives them PEFT LoRA coverage, but not Asym-owned frozen-base CPU
  offload.
- `asym_offload_modules=all` already selects `shared_experts`, router,
  attention, embeddings, LM head, norms, and dense MLPs. For Qwen3.5 this means
  the current code can offload `shared_expert.{gate,up,down}_proj` and
  `shared_expert_gate` as generic dense leaves, but it does not have the
  Llama4-style shared-MLP activation offload path or Qwen3.5-specific shared
  wrapper accounting.
- `scripts/lf/profile3.sh` is the final gate. Before Pre-Stage 0A, its relevant
  backend default is the legacy two-field
  `BACKEND_SPECS=asym_cpuadamwds|norecomp,zero3_offload|recomp`; after
  Pre-Stage 0A, the normalized Qwen3.5 validation form must be
  `asym_cpuadamwds|norecomp|ligerloss0,zero3_offload|recomp|ligerloss0`.
  The other relevant defaults are:
  `ASYMM_EXP_ACT_POLICIES=none|true|true|true`,
  `ASYM_OFFLOAD_MODULES=all`, `ASYM_STRICT=true`,
  `ASYM_CPU_ADAMW_GRAD_OFFLOAD=true`,
  `ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true`, `WORKLOADS=2048|4|1`,
  `MAX_STEPS=1`, `WARMUP_STEPS=1`, `LORA_RANK=64`, `LORA_ALPHA=16`,
  `LORA_DROPOUT=0.00`, and `PROFILE_MEMORY_BREAKDOWN=true`.
  `scripts/lf/run_lf_lora_sft.sh` then runs LF with `--lora_target all`.
  The implementation must therefore support Qwen3.5 under the exact all-target
  LoRA and `ASYM_OFFLOAD_MODULES=all` surface, not under a hand-pruned target
  list.
- Qwen3.5 profiling and final acceptance must use only
  `/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile3.sh` as
  the profiling entry point. Do not use
  `/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh`,
  `/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test2.sh`,
  or
  `/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test.sh`
  for Qwen3.5 validation; those scripts are entry points for other experiments.
  For Qwen3.5, all validation and verdict commands below must explicitly use
  `ligerloss0` and `ENABLE_LIGER_KERNEL=false`; the separate Liger-loss work
  must not change the Qwen3.5 Asym-vs-ZeRO comparison.
- Qwen3.5 CPU offload profiling must bind CPU offload allocations only to real
  CPU DRAM NUMA nodes. On the current GB200 system, the only accepted
  profiling placement is:
  `NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 NUMACTL_MODE=membind`.
  This means CPU execution is bound to nodes `0,1` and host allocations are
  bound to CPU RAM nodes `0,1` only. Do not use `NUMACTL_MODE=interleave`, do
  not widen `NUMACTL_MEMBIND` beyond `0,1`, and do not include any GPU/HBM NUMA
  node. Any artifact produced with different NUMA placement must be discarded
  for the final verdict.

Execution status as of 2026-06-17:

- Pre-Stage 0B is complete in `third_party/AsymGEMM/.venv`:
  `flash-linear-attention==0.5.0`, `fla-core==0.5.0`, and
  `causal-conv1d==1.6.2.post1` import correctly; Transformers reports both FLA
  and causal-conv availability as true.
- Stage 0 ZeRO baseline is complete with the exact `profile3.sh` Qwen3.5 knobs:
  `zero3_offload|recomp|ligerloss0` reached actual peak allocated
  `31.5265 GiB`, actual peak reserved `36.7949 GiB`, forward `27024.0 ms`,
  backward `14800.3 ms`, and measured e2e `43462.7 ms`.
- Stage 1 hybrid Qwen3.5 decoder-layer hooks are implemented and profiled. The
  Asym row completed with `qwen35_moes_wrapped=48`,
  `layer_act_offload_wrapped=48`, `attention_act_offload_wrapped=48`, and
  `attention_saved_tensor_offload_wrapped=12`, but memory was worse than ZeRO:
  actual peak allocated `55.9898 GiB`, actual peak reserved `61.4316 GiB`,
  forward `12747.0 ms`, backward `54761.0 ms`, and e2e `74833.1 ms`.
- Stage 2 Gated DeltaNet projection offload is implemented. The setup report
  showed `linear_attention:6370099200` CPU-resident base bytes and no selected
  frozen-base CUDA residue, but the stage profile did not reach a valid
  training step because the GPU had only about `1.52 GiB` free during setup.
- Stages 3 and 4 are implemented by adding `AsymQwen35SharedMLP`, a Qwen3.5
  adapter over the proven Llama4 shared-MLP base/offload and activation-offload
  path. Tests verify source parity, preexisting LoRA transfer, shared-leaf host
  adoption, and the CUDA-gated activation-offload path when kernels are
  available.
- Stage 5 is implemented as a validation gate: Qwen3.5 routed expert LoRA banks
  are registered for grouped CPUAdamW weight offload; dense/shared/GDN LoRA
  tensors remain CUDA-resident by design because they are much smaller than
  routed expert banks and offloading them as tiny groups is not accepted without
  a real memory win.
- Current validation passed:
  `pytest -q tests/training/test_lf_qwen35_asym_backend.py tests/training/test_lf_qwen3_asym_backend.py tests/test_lf_memory_breakdown.py`
  produced `163 passed, 25 skipped`. A venv-level probe also proved FLA imports
  and LlamaFactory routes selected `linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}`
  leaves into `asym_owned_dense_target_modules`.
- Final Stage 6 verdict is still pending. Do not claim Qwen3.5 memory
  improvement until `scripts/lf/profile3.sh` completes both final rows on a GPU
  with enough free HBM. At this checkpoint, `nvidia-smi` shows GPUs 0-2 almost
  full with no visible process owner and GPU3 occupied by an unrelated Qwen3
  Liger run, so the exact Qwen3.5 final profile is externally blocked.
- A later manually launched `outputs/qwen35_final` attempt used an invalid
  `NUMACTL_MEMBIND` setting that included non-CPU NUMA nodes. That run was
  stopped/discarded and must not be used for acceptance. Final Qwen3.5 runs must
  use CPU RAM nodes only:
  `NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 NUMACTL_MODE=membind`.

Global acceptance rule for every stage:

- Use toy/unit tests only to prove local correctness and kernel safety.
  Accept or reject runtime implementation stages using the real LF e2e Qwen3.5
  profiling path. Pre-Stage 0A is the exception because it is common profiling
  and interface plumbing only; accept it by dry-run command audit, unit tests,
  and proving it does not change model execution or offload selection semantics.
- Keep a runtime implementation stage only if it reduces peak HBM meaningfully
  and does not blow up latency once the exact Asym Qwen3.5 row is runnable.
  "Meaningful" means at least 5% peak allocated/reserved HBM reduction or at
  least 2 GiB, whichever is larger. "Does not blow up latency" means forward,
  backward, and measured e2e step time are each no more than 20% slower than the
  most recent accepted same-backend Qwen3.5 profile. For the first stage that makes
  `asym_cpuadamwds|norecomp|ligerloss0` complete, there is no earlier successful Asym
  Qwen3.5 baseline; require that it does not obviously exceed
  `zero3_offload|recomp|ligerloss0` timing while the later optimization stages establish
  real Asym-vs-Asym deltas.
- Reject a stage if memory is unchanged and latency increases. Also reject it
  if memory drops only trivially, even if correctness tests pass.
- Do not introduce per-expert Python loops, per-token loops, or many small GEMMs.
  Qwen3.5 routed expert work must keep using the packed/grouped Qwen3 expert
  paths. Shared expert work must remain three large dense projections plus the
  scalar shared gate, matching the source architecture.

Required `profile3.sh` knob coverage:

| Knob/config | Qwen3.5 requirement |
| --- | --- |
| `MODEL_SPECS='Qwen/Qwen3.5-122B-A10B|1'` | Model detection must route to Qwen3.5 MoE wrappers and hybrid layer hooks. Template inference may stay `qwen3_nothink` unless LF needs a Qwen3.5-specific template. |
| `BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0,zero3_offload|recomp|ligerloss0'` | Final implementation must run with CPUAdamW-Deepspeed and no recompute for the Asym row; ZeRO-3 offload baseline must still run through normal PEFT/DeepSpeed. The explicit third field pins Qwen3.5 to the no-Liger-loss axis while parallel Liger work evolves. Stage 0 does not run the Asym row; it identifies missing support by code audit after Pre-Stage 0A makes the profiling interface trustworthy and Pre-Stage 0B validates FLA. Later stages validate the exact Asym row after implementation. Do not add code that only works for the Asym row. |
| `liger_loss=ligerloss0` and `ENABLE_LIGER_KERNEL=false` | Pre-Stage 0A must make the common scripts preserve the Liger-loss axis, but Qwen3.5 implementation validation must not enable Liger kernels yet. Every Qwen3.5 profile artifact used for acceptance must prove `liger_loss=ligerloss0`. |
| `LF_EXPERT_LORA_IMPLS=split-target-parameters` and `--lora_target all` | Routed expert LoRA must use the Qwen3 split expert path. Shared expert and GDN projection LoRA must be correctly included or deliberately transferred from PEFT to Asym-owned modules when selected for offload. |
| `LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT=0.00` | All Qwen3.5 wrappers must preserve the requested rank/alpha/dropout. Expert split LoRA remains valid only for dropout `0.00`; do not silently ignore nonzero dropout. |
| `ASYM_OFFLOAD_MODULES=all` and `ASYM_STRICT=true` | Every component included in `all` must either be implemented for Qwen3.5 or fail loudly in strict mode. No selected Qwen3.5 base weight may stay on HBM without being reported. Add `ASYM_GEMM_LF_CONFIG_ASYM_STRICT` to the profile-script env so final artifacts prove strict mode was on. |
| `ASYMM_EXP_ACT_POLICIES=policy|expert_act|attn_act|layer_act` | `expert_act` applies to routed experts and, after Stage 4, shared MLP activation offload. `attn_act` applies only to full-attention `self_attn` q/k/v/o projections. `layer_act` must cover both full-attention and Gated DeltaNet decoder layers. |
| `ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu|hbm` | Keep this routed-expert-specific unless a new shared/GDN implementation explicitly proves benefit. Do not accidentally apply this routed expert knob to shared MLP or GDN projections. |
| `ASYM_CPU_ADAMW_GRAD_OFFLOAD=true` | All trainable LoRA gradients created by Qwen3.5 wrappers must remain visible to the optimizer/offload path. `_validate_trainable_params` must still pass. |
| `ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true` | Any trainable LoRA weights owned by Qwen3.5-specific wrappers must participate in CPUAdamW weight offload, or the stage must prove the existing generic path already handles them. |
| `PROFILE_MEMORY_BREAKDOWN=true` and modules `attention,linear_attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss` | Pre-Stage 0A must add a `linear_attention` profiling component and update script defaults/postprocess classification so GDN rows do not disappear into `other`. |
| `PROFILE_MODULE_FILTER=attention,linear_attention,router,mlp,experts,lora,optimizer,kt` | Pre-Stage 0A must add a filter token for `linear_attention` or deliberately map it into an existing token with tests. Prefer a separate token to keep comparison meaningful. |

## Pre-Stage 0A: Common Profiling, Config Capture, And LF Component Interface

Do this before Qwen3.5 runtime work. The point is to make every later stage
measurable and auditable before changing model execution.

Files/functions/classes to modify:

- `scripts/lf/profile3.sh`
  - default `PROFILE_MEMORY_BREAKDOWN_MODULES`
  - default `PROFILE_MODULE_FILTER`
  - backend-spec parser and normalized backend tuple shape
  - `job_root_path`, `run_id`, `jobs.tsv`, completeness checks, plot filters,
    and `run_job` signature
  - `ENABLE_LIGER_KERNEL`, `ASYM_GEMM_LF_CONFIG_LIGER_LOSS`, and
    `ASYM_GEMM_LF_CONFIG_ASYM_STRICT="${ASYM_STRICT}"` profile env forwarding.
- `scripts/lf/run_lf_lora_sft.sh`
  - profile-mode forwarding for `ASYM_GEMM_LF_CONFIG_ASYM_STRICT`
  - verify existing `ENABLE_LIGER_KERNEL` and `ASYM_GEMM_LF_CONFIG_LIGER_LOSS`
    forwarding remains intact.
- `scripts/lf/run_lf_profiled_train.py`
  - `_env_config`
  - `_config_from_args`
  - verify `ASYM_GEMM_LF_CONFIG_LIGER_LOSS` is persisted as `liger_loss`.
- `scripts/lf/migrate_liger_loss_axis.py`
  - inspect/use for legacy artifact migration only; fresh Qwen3.5 runs must
    write the explicit axis directly.
- `scripts/plotting/plot_activation_recompute_sweep.py`
  - `--liger-loss`, job-dir parsing, row metadata, grouping labels.
- `scripts/plotting/plot_lf_memory_breakdown.py`
  - `--liger-loss`, job-dir parsing, row metadata, grouping labels.
- `scripts/plotting/plot_lf_interconnect_ctc.py`
  - `--liger-loss`, job-dir parsing, row metadata, grouping labels.
- `asym_gemm/profiling/lf_trace.py`
  - `_component_from_param_name`
  - `_component_from_module_name`
  - `_component_from_range_name`
  - `_component_filter_token`
- `scripts/lf/postprocess_lf_profile_artifacts.py`
  - module/timing labels for Qwen3.5 `linear_attn` / Gated DeltaNet rows.
- `asym_gemm/integrations/lf.py`
  - `classify_lf_component` for reporting/classification only.
- `tests/lf/test_asym_cpu_adamw_args.py`
  - dry-run env/config forwarding coverage for strict mode, Liger-loss axis,
    command paths, `jobs.tsv`, and plot filters.
- `tests/lf/test_superoffload_backend_scripts.py`
  - update expected source-profile metadata/path fixtures to include
    `liger_loss`.
- `tests/test_lf_memory_breakdown.py`
  - component classification tests for `linear_attn` params, modules, and ranges.

Intended code changes:

- Make profile artifacts prove the exact common knobs used by later stages:
  `liger_loss`, `asym_strict`, `asym_offload_modules`, CPUAdamW grad/weight
  offload, activation offload knobs, LoRA impl, rank, alpha, and dropout.
- Implement the needed profiling interface directly in `profile3.sh`:
  - accept `backend|recompute` and `backend|recompute|ligerloss0/ligerloss1`;
  - normalize every backend spec to `backend|recompute|ligerloss`;
  - default legacy two-field specs to `ligerloss0`;
  - include `liger_loss` in job directory names, run IDs, `jobs.tsv`, profile
    completeness checks, source-profile matching, and plot filter calls;
  - set `ENABLE_LIGER_KERNEL=false` for `ligerloss0` and `true` for
    `ligerloss1`;
  - forward `ASYM_GEMM_LF_CONFIG_LIGER_LOSS` so
    `run_lf_profiled_train.py` records it in `source_profile.config`.
- For Qwen3.5, do not sweep Liger loss in this plan. All Qwen3.5 commands must
  spell `|ligerloss0`; any accidental `ligerloss1` artifact is excluded from
  Qwen3.5 Asym acceptance.
- Add a profiling component named `linear_attention` for Qwen3.5 Gated DeltaNet
  ranges and projection leaves. This is profile/accounting only in Pre-Stage 0A.
- Update profile defaults so memory breakdown and module filters can emit
  `linear_attention` rows:

```bash
PROFILE_MEMORY_BREAKDOWN_MODULES=attention,linear_attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss
PROFILE_MODULE_FILTER=attention,linear_attention,router,mlp,experts,lora,optimizer,kt
```

- Do not change `ASYM_OFFLOAD_MODULES=all` semantics in this stage. In
  particular, do not add `linear_attention` to `_ALL_LF_OFFLOAD_COMPONENTS` and
  do not route GDN projection leaves into Asym-owned dense targets yet. Stage 2
  makes that decision based on real profile evidence.
- Do not enable Liger kernels for Qwen3.5. The common scripts may support
  `ligerloss1` for the parallel Liger work, but the Qwen3.5 validation row is
  always `ligerloss0` / `ENABLE_LIGER_KERNEL=false`.
- Keep the common interface efficient: this stage should add labels,
  config-capture, and classification only. It must not add model hooks, tensor
  transfers, or new kernels.

Pseudocode:

```bash
append_backend_spec() {
  raw="$1"
  if raw has one pipe:
    backend, recompute = split(raw, "|")
    liger_loss=ligerloss0
  elif raw has two pipes:
    backend, recompute, liger_loss = split(raw, "|")
  else:
    die "backend spec must be backend|recompute[|ligerloss]"

  liger_loss = normalize_liger_loss(liger_loss)  # ligerloss0 or ligerloss1
  for recompute_mode in expand_recompute(recompute):  # norecomp/recomp/both
    backend_specs_raw += "${backend}|${recompute_mode}|${liger_loss}"
}

run_job(backend, profiler, recompute, liger_loss, ...):
  enable_liger_kernel=false
  if liger_loss == "ligerloss1":
    enable_liger_kernel=true
  job_root = job_root_path(..., liger_loss, grad_offload, weight_offload)
  run_env += ENABLE_LIGER_KERNEL="${enable_liger_kernel}"
  run_env += ASYM_GEMM_LF_CONFIG_LIGER_LOSS="${liger_loss}"
  append_job_record(..., profiler, liger_loss, grad_offload, ...)
```

```python
def classify_profile_component(name):
    lower = name.lower()
    leaf = lower.rsplit(".", 1)[-1]
    if ".linear_attn." in lower and leaf in {
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    }:
        return "linear_attention"
    if ".linear_attn" in lower or "gateddeltanet" in lower:
        return "linear_attention"
    return existing_profile_component(name)

def classify_lf_component(name, module=None):
    component = classify_profile_component(name)
    if component == "linear_attention":
        return "linear_attention"  # reporting only until Stage 2
    return existing_classification(name, module)

# Pre-Stage 0A invariant:
# parse_lf_offload_modules("all") does not select linear_attention yet.
```

Validation commands:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

pytest -q tests/lf/test_asym_cpu_adamw_args.py tests/test_lf_memory_breakdown.py

bash -n scripts/lf/profile3.sh scripts/lf/run_lf_lora_sft.sh

DRY_RUN=true \
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/outputs/qwen35_prestage0a_interface_dryrun \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=true \
MODEL_SPECS='Qwen/Qwen3.5-122B-A10B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0,zero3_offload|recomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|true' \
ASYM_OFFLOAD_MODULES=all \
ASYM_STRICT=true \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
LF_EXPERT_LORA_IMPLS=split-target-parameters \
LORA_DROPOUT=0.00 \
LORA_RANK=64 \
LORA_ALPHA=16 \
WORKLOADS='2048|4|1' \
MAX_STEPS=1 \
WARMUP_STEPS=1 \
bash scripts/lf/profile3.sh --overwrite true
```

Acceptance before Pre-Stage 0B:

- Dry-run command artifacts include
  `ASYM_GEMM_LF_CONFIG_ASYM_STRICT=true`,
  `ASYM_GEMM_LF_CONFIG_LIGER_LOSS=ligerloss0`,
  `ENABLE_LIGER_KERNEL=false`, and the expected CPUAdamW, activation-offload,
  LoRA, and workload config values.
- Dry-run command paths include `__ligerloss0` and do not include
  `__ligerloss1`.
- `jobs.tsv` has a `liger_loss` column, and every Qwen3.5 row in this dry run
  has `ligerloss0`.
- Profile completeness checks reject stale Qwen3.5 artifacts whose config lacks
  the expected `liger_loss=ligerloss0`.
- Plot invocations generated by the scripts include `--liger-loss ligerloss0`.
  `plot_activation_recompute_sweep.py`, `plot_lf_memory_breakdown.py`, and
  `plot_lf_interconnect_ctc.py` must parse both legacy job dirs and explicit
  `__ligerloss0` job dirs, defaulting only legacy paths to `ligerloss0`.
- Unit tests prove Qwen3.5 `linear_attn` params/modules/ranges classify as
  `linear_attention`, not `other`.
- `ASYM_OFFLOAD_MODULES=all` still does not select GDN projection offload in
  Pre-Stage 0A. If this changes accidentally, reject the stage because it mixed
  measurement plumbing with model implementation.
- This stage is accepted by correctness and auditability, not by memory
  reduction. It should not change forward/backward latency because it does not
  alter runtime model execution.

Risks to watch:

- `classify_lf_component` may be used by both reporting and offload target
  splitting. Keep tests that prove `linear_attention` is visible in reports but
  not selected for Asym ownership until Stage 2 explicitly enables it.
- `profile_lora_lf.sh`, `profile_lora_lf_test2.sh`, and
  `profile_lora_lf_test.sh` are not Qwen3.5 validation entry points. Do not let
  their behavior or artifacts drive Qwen3.5 acceptance unless a separate task
  explicitly changes the scope.
- `profile3.sh`, postprocess, plotting, and migration must stay aligned;
  otherwise final comparisons can show `linear_attention` in one artifact,
  `other` in another, or mix `ligerloss0` and `ligerloss1` rows.

## Pre-Stage 0B: Mandatory FLA Install And Validation Gate

Run this only after Pre-Stage 0A has made the profiling/script interface
auditable. If this install or validation fails, stop. Do not proceed to the
Stage 0 baseline, Qwen3.5 implementation, profiling, or acceptance/rejection
decisions until the FLA/GDN dependency path works in the same Python environment
used by `profile3.sh`.

Environment to modify:

- `third_party/AsymGEMM/.venv/bin/python`

Files to modify:

- None.

Required install command:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

PY=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python

$PY -m pip show fla-core flash-linear-attention causal-conv1d || true
$PY -m pip index versions flash-linear-attention | head
$PY -m pip index versions fla-core | head
$PY -m pip install "flash-linear-attention[cuda,conv1d]==0.5.0"
```

Use the package-index install, not the local checkout and not editable, unless
the package-index install is proven unusable on this machine. If a local-source
fallback is required later, record it explicitly in the profiling artifact notes
because those numbers no longer match the FA4-style package-index setup. The
local `third_party/flash-linear-attention` checkout currently advertises
`0.5.1`, while the package-index version checked here is `0.5.0`; do not mix
those two in the same verdict run.
If validation fails because an old, local, or editable FLA package is already
present, uninstall `fla-core flash-linear-attention causal-conv1d` and rerun the
install command; do not proceed with a mixed environment.

Required validation command:

```bash
PY=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python

$PY - <<'PY'
import importlib.metadata as md

import fla
from fla.modules.convolution import causal_conv1d as fla_causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
import causal_conv1d

from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)

print("flash-linear-attention", md.version("flash-linear-attention"))
print("fla-core", md.version("fla-core"))
print("fla.__version__", fla.__version__)
print("causal-conv1d", md.version("causal-conv1d"))
print("hf_flash_linear_attention_available", is_flash_linear_attention_available())
print("hf_causal_conv1d_available", is_causal_conv1d_available())

assert md.version("flash-linear-attention") == "0.5.0"
assert is_flash_linear_attention_available()
assert is_causal_conv1d_available()
assert callable(chunk_gated_delta_rule)
assert callable(fused_recurrent_gated_delta_rule)
assert callable(fla_causal_conv1d)
print("Qwen3.5 FLA dependency gate: PASS")
PY
```

Acceptance before Stage 0:

- `flash-linear-attention==0.5.0`, `fla-core`, and `causal-conv1d` are installed
  in `third_party/AsymGEMM/.venv`.
- `fla.ops.gated_delta_rule.{chunk,fused_recurrent}_gated_delta_rule` imports.
- `fla.modules.convolution.causal_conv1d` imports.
- Transformers reports both `is_flash_linear_attention_available()` and
  `is_causal_conv1d_available()` as true.
- If any item fails, fix the environment first. Do not start Stage 0 or later
  work against the torch fallback path.

## Stage 0: Establish Zero3 Baseline And Static Asym Code Audit

Run the numeric baseline only after Pre-Stage 0A has made the common profiling
and config-capture interface trustworthy and Pre-Stage 0B has validated FLA.
Do not collect baseline artifacts with an older schema that lacks strict-mode
config or `linear_attention` labels.

Files to inspect:

- `scripts/lf/profile3.sh`
- `scripts/lf/postprocess_lf_profile_artifacts.py`
- `scripts/lf/run_lf_lora_sft.sh`
- `scripts/lf/run_lf_profiled_train.py`
- `asym_gemm/training/qwen35_moe.py`
- `asym_gemm/training/qwen3_moe.py`
- `asym_gemm/training/llama4_moe.py`
- `asym_gemm/training/llama4_shared_mlp.py`
- `asym_gemm/integrations/lf.py`
- `asym_gemm/training/weight_offload.py`
- `tests/training/test_lf_qwen35_asym_backend.py`

Implementation work:

- Run the known-good `zero3_offload|recomp|ligerloss0` Qwen3.5 profile. This is the only
  required numeric Stage 0 baseline because it does not depend on the missing
  Qwen3.5 Asym implementation.
- Do not run `asym_cpuadamwds|norecomp|ligerloss0` as a Stage 0 probe. The missing work is
  already visible from code inspection: hybrid decoder-layer hooks do not cover
  `linear_attn`, GDN projection offload is intentionally not selected yet,
  shared expert lacks a first-class Qwen3.5 wrapper/activation-offload path, and
  shared-wrapper LoRA weight-offload coverage must be verified.
- Capture allocator peaks and source memory breakdown for the completed ZeRO
  row. Do not accept or reject Qwen3.5 changes from tiny model tests unless the
  changed code is an isolated kernel implementation.
- Record a baseline table with:
  `peak_allocated_hbm_bytes`, `peak_reserved_hbm_bytes`,
  `actual_peak_allocated_hbm_bytes`, `actual_peak_reserved_hbm_bytes`,
  `forward.total_milliseconds`, `backward.total_milliseconds`, and
  `trainer.timing.measured_e2e_step_milliseconds`.

Validation command, required numeric baseline:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

NUMACTL_MEMBIND=0,1 \
NUMACTL_CPUNODEBIND=0,1 \
NUMACTL_MODE=membind \
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/outputs/qwen35_stage0_zero3 \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=true \
MODEL_SPECS='Qwen/Qwen3.5-122B-A10B|1' \
BACKEND_SPECS='zero3_offload|recomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|true' \
ASYM_OFFLOAD_MODULES=all \
ASYM_STRICT=true \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
LF_EXPERT_LORA_IMPLS=split-target-parameters \
LORA_DROPOUT=0.00 \
LORA_RANK=64 \
LORA_ALPHA=16 \
WORKLOADS='2048|4|1' \
MAX_STEPS=1 \
WARMUP_STEPS=1 \
bash scripts/lf/profile3.sh --overwrite true
```

Metric extraction pseudocode:

```python
from pathlib import Path
import json

root = Path("outputs/qwen35_stage0_zero3")

def read_profile(path):
    profile = json.loads(path.read_text())
    source = profile.get("source_profile") or profile
    config = source.get("config", {})
    mem = source.get("memory", {}).get("gpu", {})
    trainer = source.get("trainer", {}).get("timing", {})
    row = {
        "path": str(path),
        "backend": config.get("backend"),
        "liger_loss": config.get("liger_loss"),
        "activation_recompute": config.get("activation_recompute"),
        "model": config.get("model_name_or_path"),
        "lora_target": config.get("lora_target"),
        "offload_modules": config.get("asym_offload_modules"),
        "strict": config.get("asym_strict"),
        "qwen_moe_expert_lora_impl": config.get("qwen_moe_expert_lora_impl"),
        "cpuadam_grad_offload": config.get("asym_cpu_adamw_grad_offload"),
        "cpuadam_weight_offload": config.get("asym_cpu_adamw_weight_offload"),
        "expact": config.get("asymm_expert_act_offload"),
        "attnact": config.get("asymm_attn_act_offload"),
        "layeract": config.get("asymm_layer_act_offload"),
        "peak_alloc": mem.get("peak_allocated_hbm_bytes"),
        "peak_reserved": mem.get("peak_reserved_hbm_bytes"),
        "forward_ms": source.get("forward", {}).get("total_milliseconds"),
        "backward_ms": source.get("backward", {}).get("total_milliseconds"),
        "e2e_ms": trainer.get("measured_e2e_step_milliseconds"),
    }
    for candidate in (
        path.parent / "memory_breakdown_summary.json",
        path.parent / "memory_breakdown" / "memory_breakdown_summary.json",
    ):
        if candidate.exists():
            summary = json.loads(candidate.read_text())
            row["actual_peak_alloc"] = summary.get("actual_peak_allocated_hbm_bytes")
            row["actual_peak_reserved"] = summary.get("actual_peak_reserved_hbm_bytes")
            row["actual_peak_phase"] = summary.get("actual_peak_phase")
            break
    return row

rows = [read_profile(p) for p in root.rglob("profile.json")]
for row in sorted(rows, key=lambda r: (str(r["backend"]), str(r["path"]))):
    print(row)
```

Acceptance before Stage 1:

- The `zero3_offload|recomp|ligerloss0` Stage 0 artifact is complete, not `partial`, and
  `profile3.sh` accepts it through `job_profile_complete`.
- The complete source profile config proves the exact baseline knob surface:
  `model_name_or_path=Qwen/Qwen3.5-122B-A10B`, `lora_target=all`,
  `liger_loss=ligerloss0`,
  `qwen_moe_expert_lora_impl=split-target-parameters`,
  `asym_strict=true`, and the ZeRO row's policy-independent Asym knobs
  canonicalized off by the script.
- The code audit maps every known missing Qwen3.5 Asym runtime surface to a
  later stage: hybrid decoder-layer hooks to Stage 1, GDN projection offload to
  Stage 2, shared MLP wrapper/accounting to Stage 3, shared activation offload
  to Stage 4, and trainable LoRA weight-offload verification to Stage 5.

Risks to watch:

- `Qwen/Qwen3.5-122B-A10B` may require model access and enough CPU memory for
  CPU-first loading. A failed download/load is not an implementation failure.
- `MAX_STEPS=1` is the exact script default and good for the requested verdict
  shape, but timing noise is high. If a latency decision is close, run a second
  same-knob sweep with `MAX_STEPS=10 WARMUP_STEPS=5` as supporting evidence;
  do not replace the exact-config verdict with the longer run.

Reusable Asym e2e validation command for implementation Stages 1-5:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

NUMACTL_MEMBIND=0,1 \
NUMACTL_CPUNODEBIND=0,1 \
NUMACTL_MODE=membind \
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/outputs/qwen35_stageN \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=true \
MODEL_SPECS='Qwen/Qwen3.5-122B-A10B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|true' \
ASYM_OFFLOAD_MODULES=all \
ASYM_STRICT=true \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
LF_EXPERT_LORA_IMPLS=split-target-parameters \
LORA_DROPOUT=0.00 \
LORA_RANK=64 \
LORA_ALPHA=16 \
WORKLOADS='2048|4|1' \
MAX_STEPS=1 \
WARMUP_STEPS=1 \
bash scripts/lf/profile3.sh --overwrite true
```

## Stage 1: Cover Qwen3.5 Hybrid Decoder-Layer Hooks

Files/functions/classes to modify:

- `asym_gemm/training/qwen35_moe.py`
  - `_is_qwen35_family`
  - `is_qwen35_moe_block`
  - `AsymQwen35MoeBlock.__init__`
  - `AsymQwen35MoeBlock._compute_routing`
  - `AsymQwen35MoeBlock.forward`
- `asym_gemm/integrations/lf.py`
  - `_is_qwen3_decoder_layer_module_name`
  - `_wrap_qwen3_decoder_checkpoint_modules`
  - `_wrap_qwen3_decoder_saved_tensor_offload_modules`
- `asym_gemm/training/decoder_activation_offload.py`
  - no algorithm change expected; use existing `install_decoder_saved_tensor_offload`.
- `asym_gemm/training/decoder_checkpoint.py`
  - no algorithm change expected; use existing `install_decoder_checkpoint`.
- `tests/training/test_lf_qwen35_asym_backend.py`
  - add tests for Qwen3.5 decoder layer detection, saved tensor offload, and
    layer checkpoint/activation hook installation.

Intended code changes:

- Make Qwen3.5 model-type checks explicit in LF decoder-layer discovery. The
  current code recognizes classic Qwen3 layers and Llama4 layers; Qwen3.5
  decoder layers are different because they use either `linear_attn` or
  `self_attn` depending on `layer_type`, plus `mlp`, `input_layernorm`, and
  `post_attention_layernorm`.
- This is an optimization stage, not just naming hardening. The intended effect
  is that `ASYMM_LAYER_ACT_OFFLOAD=true` and `ASYM_EXPERT_RECOMPUTE_POLICY=gc-layer`
  wrap all Qwen3.5 text decoder layers, including the Gated DeltaNet layers that
  make up most of the stack. That should reduce saved activation HBM for the
  linear-attention path without changing routed expert math.
- Full-attention `self_attn` layers keep the existing attention path:
  selected q/k/v/o projections can use AsymGEMM frozen-base CPU offload plus
  attention activation offload/backfetch when `ASYMM_ATTN_ACT_OFFLOAD=true`.
  Qwen3.5 `linear_attn` layers do not use that attention projection backfetch
  path. They are covered at this stage only by decoder-layer checkpointing and
  decoder saved-tensor offload around the whole layer.
- Preserve the active Gated DeltaNet core. If `flash-linear-attention` and
  `causal_conv1d` are installed later, the wrapper must still let Transformers
  call FLA's `chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule` and
  `causal_conv1d`. If they are absent, it must leave the current torch fallback
  intact. The layer hook boundary is outside the GDN core in both cases.
- Keep whole-router mode as the e2e path for `profile3.sh`. Qwen3.5
  `output_router_logits=True` conflicts with `router_mode=whole` because whole
  mode intentionally detaches router outputs.
- Keep routed expert execution delegated to `AsymQwen3Experts`; do not copy or
  rewrite the grouped routed expert kernels.
- Strengthen shape assertions for `shared_expert` and `shared_expert_gate` so
  unsupported variants fail before profiling.

Pseudocode:

```python
def _is_qwen35_decoder_layer(module):
    children = dict(module.named_children())
    config = getattr(module, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    has_norms = {"mlp", "input_layernorm", "post_attention_layernorm"} <= set(children)
    has_mixer = "linear_attn" in children or "self_attn" in children
    class_or_module = (type(module).__name__ + " " + type(module).__module__).lower()
    return (
        has_norms
        and has_mixer
        and (
            model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}
            or "qwen3_5_moe" in class_or_module
            or getattr(children["mlp"], "_is_asym_qwen35_moe_block", False)
            or is_qwen35_moe_block(children["mlp"])
        )
    )

def _is_qwen3_decoder_layer_module_name(name, module):
    if not name or _has_attention_excluded_path_marker(name):
        return False
    if _is_qwen35_decoder_layer(module):
        return True
    # existing Qwen3 and Llama4 branches stay unchanged
```

```python
def is_qwen35_moe_block(module):
    if getattr(module, "_is_asym_qwen35_moe_block", False):
        return False
    if not _is_qwen35_family(module):
        return False
    gate = module.gate
    experts = module.experts
    shared = module.shared_expert
    shared_gate = module.shared_expert_gate
    require is_qwen3_experts(experts)
    require shared has gate_proj/up_proj/down_proj
    require shared_gate.in_features == gate.hidden_dim
    require shared_gate.out_features == 1
    require shared.gate_proj.in_features == gate.hidden_dim
    require shared.up_proj.in_features == gate.hidden_dim
    require shared.down_proj.out_features == gate.hidden_dim
    require shared.gate_proj.out_features == shared.up_proj.out_features
    require shared.down_proj.in_features == shared.gate_proj.out_features
    return True
```

Validation commands:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

pytest -q \
  tests/training/test_lf_qwen35_asym_backend.py \
  tests/training/test_lf_qwen3_asym_backend.py \
  tests/training/test_lf_llama4_asym_backend.py
```

Then run the reusable Asym e2e validation command, changing only:

```bash
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/outputs/qwen35_stage1
```

Acceptance before Stage 2:

- Qwen3.5 unit tests prove source/wrapped forward parity and that
  `ASYMM_LAYER_ACT_OFFLOAD=true` installs hooks on Qwen3.5 decoder layers.
- The test/log output records which GDN backend was active: FLA/causal-conv when
  dependencies are importable, or the Transformers torch fallback when they are
  absent. Do not accept a change that silently switches a FLA run to fallback.
- E2E `memory_breakdown_summary.json` and the Asym setup/runtime logs show
  decoder saved-tensor offload wrappers on both `linear_attn` and `self_attn`
  layer types. The wrapped count should match the text decoder layer count, not
  only the full-attention count. Attention activation offload counts should
  continue to count only `self_attn` q/k/v/o parents.
- If this stage is the first one to make the reusable Asym e2e command complete,
  accept it only when the layer-hook counters are correct and
  forward/backward/e2e timing is not obviously worse than the Stage 0 Zero3
  baseline. Otherwise keep it only if it lowers actual peak HBM meaningfully
  versus the previous successful Asym profile without measurable latency
  regression. If it wraps more layers but memory is unchanged and latency
  increases, reject the change.

Risks to watch:

- Qwen3.5 multimodal/vision module names may match generic attention/norm logic.
  Keep `_ATTENTION_GC_EXCLUDED_PATH_MARKERS` exclusions in place and test text
  tower names, not vision tower names.
- `router_mode=hf` can still find Qwen3.5 `experts` because they are Qwen3-style
  packed experts, but the final `profile3.sh` path uses `whole`; optimize whole
  mode first.

## Stage 2: Conditional Gated DeltaNet Projection Offload

Pre-Stage 0A already made Qwen3.5 Gated DeltaNet visible as `linear_attention` in
profiles. This stage decides whether to turn that profile component into an
actual Asym-owned offload selector.

The implementation is conditional: add Asym ownership/offload only for the
dense projection leaves if the real profile shows those frozen base weights are
a meaningful HBM contributor. Never own or rewrite the Gated DeltaNet
recurrent/delta-rule core.

Files/functions/classes to modify only if GDN projection offload is accepted by
the memory gate:

- `asym_gemm/integrations/lf.py`
  - `SUPPORTED_LF_OFFLOAD_COMPONENTS`
  - `_ALL_LF_OFFLOAD_COMPONENTS`
  - `LFOffloadSelection`
  - `parse_lf_offload_modules`
  - `classify_lf_component`
  - `component_is_selected`
  - `_wrap_lf_linear_leaf`
  - report byte accounting in `build_lf_asym_report`
- `asym_gemm/training/offload.py`
  - `collect_lf_offload_residency` classification only if extra component names
    need explicit handling.
- `third_party/LlamaFactory/src/llamafactory/model/adapter.py`
  - `split_asym_peft_dense_targets`
  - `_filter_asym_dense_peft_targets` only if target filtering needs to avoid
    PEFT pre-wrapping selected GDN projection leaves.
- `tests/training/test_lf_qwen35_asym_backend.py`
  - tests for `linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}`
    selection, CPU host adoption, fallback accounting, and forward/backward parity.

Intended code changes:

- Convert Pre-Stage 0A's profiling-only `linear_attention` component into an actual
  `asym_offload_modules=linear_attention` selector only if projection offload is
  accepted by the memory/latency gate. Include it in `all` only in the accepted
  implementation. Do not overload normal `attention` activation offload, because
  Gated DeltaNet is not q/k/v/o attention and should not enter
  `AsymActivationOffloadLoRALinear`.
- Classify only text-tower Qwen3.5 Gated DeltaNet leaves:
  `linear_attn.in_proj_qkv`, `linear_attn.in_proj_z`, `linear_attn.in_proj_b`,
  `linear_attn.in_proj_a`, and `linear_attn.out_proj`.
- Use the existing generic dense offload machinery for base weights:
  `AsymLoRALinear.from_host_weight` when the shape supports direct BF16, or the
  existing torch CPU-fetch fallback when it does not. Do not reimplement the
  delta-rule, causal convolution, or Gated DeltaNet kernels.
- Keep each GDN projection as one dense projection. Do not split
  `in_proj_qkv` into tiny per-head q/k/v GEMMs, do not loop over channels, and
  do not wrap the convolution or delta-rule state update. The only allowed
  AsymGEMM work here is frozen-base CPU offload/backfetch for whole dense
  projection leaves.
- Update LlamaFactory's PEFT/Asym split so selected `linear_attention` leaves are
  assigned to `asym_owned_dense_target_modules`. Without this, `lora_target=all`
  can let PEFT wrap the GDN linears first, after which the generic Asym dense
  pass will not own or offload their frozen bases.
- Keep Pre-Stage 0A profile defaults and postprocess classification intact so the
  final verdict can show whether GDN projection offload actually reduced HBM and
  whether it added latency.
- Expect `in_proj_b` and `in_proj_a` to be small and possibly direct-BF16
  backward-incompatible because their output dimension can be below 64. They
  should either fall back to torch CPU fetch or remain resident if profiling
  shows transfer overhead exceeds memory benefit.

Pseudocode:

```python
SUPPORTED_LF_OFFLOAD_COMPONENTS = frozenset({
    "routed_experts", "router", "shared_experts", "attention",
    "linear_attention", "embed_tokens", "lm_head", "norms", "mlp_dense",
})

def parse_lf_offload_modules(selector):
    aliases = {
        ...,
        "gdn": "linear_attention",
        "linear_attn": "linear_attention",
        "gated_deltanet": "linear_attention",
    }
    if token == "all":
        expanded.update(SUPPORTED_LF_OFFLOAD_COMPONENTS)

# Pre-Stage 0A already classifies Qwen3.5 GDN ranges/leaves as linear_attention.
# Stage 2 makes that component selectable for offload.

def component_is_selected(component, leaf, selection):
    if component == "linear_attention":
        return selection.linear_attention
    return existing_component_selection(component, leaf, selection)

# LlamaFactory adapter.py
if component in {"attention", "linear_attention", "shared_experts", "lm_head", "mlp_dense"} \
   and component_is_selected(component, module_leaf, selection):
    selected_names.append(name)
```

Validation commands:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

pytest -q tests/training/test_lf_qwen35_asym_backend.py -k 'linear_attn or offload_modules or all'
pytest -q tests/training/test_lf_qwen35_asym_backend.py tests/training/test_lf_qwen3_asym_backend.py
```

Then run the reusable Asym e2e validation command, changing only:

```bash
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/outputs/qwen35_stage2
```

Acceptance before Stage 3:

- If projection offload is accepted, residency validation reports no selected
  CUDA-resident frozen base weights for the new `linear_attention` component
  when `ASYM_OFFLOAD_MODULES=all`.
- If projection offload is accepted, `split_asym_peft_dense_targets` routes
  selected GDN projection leaves into `asym_owned_dense_target_modules`; tests
  must fail if those leaves remain ordinary PEFT LoRA modules before Asym
  wrapping.
- Profile/postprocess artifacts contain `linear_attention` rows in memory and
  timing outputs, not only `other`.
- Profile logs show GDN leaves are not passed through attention activation
  offload. Only base CPU offload and LoRA wrapping are expected here.
- Keep the projection-offload implementation only if the real Qwen3.5 profile
  shows a meaningful HBM drop versus the previous successful Asym profile without
  forward/backward/e2e latency blowup. Reject projection offload if most
  affected leaves are tiny torch CPU-fetch fallbacks and transfer overhead
  dominates.

Risks to watch:

- Adding `linear_attention` to `all` changes the meaning of `ASYM_OFFLOAD_MODULES=all`.
  Do that only if projection offload is accepted. Pre-Stage 0A already added
  `linear_attention` to profiling outputs without making it an offload selector.
- Gated DeltaNet optional fused dependencies can change saved tensors and norm
  classes. Re-run this stage if `causal_conv1d` or `flash-linear-attention` is
  installed after the baseline.

## Stage 3: Add A First-Class Qwen3.5 Shared MLP Wrapper

Files/functions/classes to modify:

- Add `asym_gemm/training/qwen35_shared_mlp.py`
  - `AsymQwen35SharedMLP`
  - `is_qwen35_shared_mlp`
  - `is_qwen35_shared_mlp_leaf`
  - helper functions copied from `llama4_shared_mlp.py` with Qwen3.5 names.
- `asym_gemm/training/qwen35_moe.py`
  - `AsymQwen35MoeBlock.__init__`
  - `AsymQwen35MoeBlock.cpu_resident_base_bytes`
  - `AsymQwen35MoeBlock.gpu_resident_base_bytes`
  - `AsymQwen35MoeBlock.trainable_lora_params`
  - `AsymQwen35MoeBlock.forward`
  - `wrap_qwen35_moe_block`
- `asym_gemm/training/__init__.py`
  - export the new wrapper if this package exports training wrappers there.
- `asym_gemm/integrations/lf.py`
  - imports
  - `count_lora_wrapped_modules`
  - `apply_lf_asym_lora`
  - `_infer_adapter_config`
- `tests/training/test_lf_qwen35_asym_backend.py`
  - shared wrapper parity
  - preexisting PEFT LoRA copy
  - CPU host adoption without cloning
  - residency validation and report byte accounting.

Intended code changes:

- Copy `AsymLlama4SharedMLP` as the implementation template, then change only
  architecture-specific naming and activation attribute handling. Qwen3.5
  `Qwen3_5MoeMLP` uses `act_fn`; Llama4 shared MLP uses `activation_fn`.
  The Qwen3.5 wrapper should accept either but store `self.activation_fn`.
- Wrap Qwen3.5 `shared_expert` inside `AsymQwen35MoeBlock` when all three
  leaves are LoRA targets or already have PEFT LoRA, exactly like Llama4.
- Keep `shared_expert_gate` outside the shared MLP wrapper. It is a separate
  scalar gate with shape `[1, hidden]`; the existing generic dense offload path
  correctly handles it, and the Asym direct BF16 dx path is usually not valid
  because `out_features=1`.
- Avoid double wrapping. Once `shared_expert` is an `AsymQwen35SharedMLP`, the
  generic dense traversal should skip its `AsymLoRALinear` children naturally,
  but tests must verify `dense_lora_wrapped` counts only missing leaves.

Pseudocode:

```python
class AsymQwen35SharedMLP(nn.Module):
    def __init__(self, source, *, backend, precision, offload,
                 lora_rank, lora_alpha, lora_dropout,
                 lora_dtype=torch.bfloat16, stats=None, strict=True):
        gate_spec = _extract_shared_linear_leaf(source.gate_proj, strict=strict)
        up_spec = _extract_shared_linear_leaf(source.up_proj, strict=strict)
        down_spec = _extract_shared_linear_leaf(source.down_proj, strict=strict)

        gate = gate_spec.base
        up = up_spec.base
        down = down_spec.base
        require all biases are None
        require gate.in_features == up.in_features == down.out_features
        require gate.out_features == up.out_features == down.in_features

        self.hidden_size = gate.in_features
        self.intermediate_size = gate.out_features
        self.activation_fn = getattr(source, "act_fn", None) or getattr(source, "activation_fn", None)
        require callable(self.activation_fn)

        self.backend = backend
        self.precision = precision
        self.offload = bool(offload)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scale = float(lora_alpha) / float(lora_rank)
        self.lora_dtype = normalize_lora_dtype(lora_dtype)
        self.lora_dropout_p = float(lora_dropout)
        self.stats = stats or AsymExecutionStats()
        self.profile_prefix = "layers.unknown.mlp.shared_expert"

        if backend == "asym" and offload:
            for leaf_name, linear in [("gate_proj", gate), ("up_proj", up), ("down_proj", down)]:
                require linear.weight.device.type == "cpu" when strict
                require linear.weight.dtype == torch.bfloat16 when strict
            self.gate_proj = AsymLoRALinear.from_host_weight(
                adopt_host_weight("shared_expert.gate_proj.weight", gate.weight, "shared_experts",
                                  pin_memory_policy="auto", strict=strict),
                rank=lora_rank, alpha=lora_alpha, backend="asym", stats=self.stats,
                device=cuda_if_available(), lora_dtype=self.lora_dtype,
                precision=precision, init_lora_weights="peft", lora_dropout=lora_dropout)
            self.up_proj = same_for(up)
            self.down_proj = same_for(down)
        elif backend == "asym":
            self.gate_proj = AsymLoRALinear(gate, ...)
            self.up_proj = AsymLoRALinear(up, ...)
            self.down_proj = AsymLoRALinear(down, ...)
        else:
            self.gate_proj = TorchLoRALinear(gate, ...)
            self.up_proj = TorchLoRALinear(up, ...)
            self.down_proj = TorchLoRALinear(down, ...)

        copy preexisting LoRA weights and validate scaling

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(self.activation_fn(gate) * up)
```

```python
def _qwen35_shared_expert_is_lora_target(name, module):
    if not wrap_dense or not selection.shared_experts:
        return False
    shared = getattr(module, "shared_expert", None)
    if not isinstance(shared, nn.Module):
        return False
    for leaf in ("gate_proj", "up_proj", "down_proj"):
        child = getattr(shared, leaf, None)
        supported = isinstance(child, nn.Module) and is_qwen35_shared_mlp_leaf(child)
        targeted = (
            _is_all_target(raw_lora_target)
            or _matches_target(f"{name}.shared_expert.{leaf}", child, dense_target_modules)
        )
        preexisting = hasattr(child, "lora_A") and hasattr(child, "lora_B")
        require supported
    return all(targeted) or all(preexisting)
```

```python
if kind == "qwen35_whole":
    wrapped = wrap_qwen35_moe_block(
        module,
        ...,
        wrap_shared_expert=_qwen35_shared_expert_is_lora_target(name, module),
        offload_shared_expert=backend == "asym" and selection.shared_experts,
    )
    wrapped.profile_prefix = _layer_profile_prefix_from_module_name(name, "mlp")
    wrapped.experts.profile_prefix = f"{wrapped.profile_prefix}.experts"
    if isinstance(wrapped.shared_expert, AsymQwen35SharedMLP):
        wrapped.shared_expert.profile_prefix = f"{wrapped.profile_prefix}.shared_expert"
        report.dense_lora_wrapped += max(0, 3 - wrapped.shared_expert.preexisting_lora_leaf_count)
```

Validation commands:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

pytest -q tests/training/test_lf_qwen35_asym_backend.py
pytest -q tests/training/test_lf_llama4_asym_backend.py tests/training/test_lf_qwen3_asym_backend.py
```

Then run the reusable Asym e2e validation command, changing only:

```bash
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/outputs/qwen35_stage3
```

Acceptance before Stage 4:

- Correctness tests show Qwen3.5 source and wrapped block outputs match in BF16
  within the same tolerances as existing Qwen3.5 tests.
- `validate_lf_offload_residency` reports no selected CUDA-resident frozen base
  weights for `shared_experts`.
- Runtime counters still show Asym routed expert calls, and shared MLP profile
  ranges appear under `layers.*.mlp.shared_expert`.
- Keep the stage only if e2e Qwen3.5 peak HBM improves meaningfully versus the
  previous successful Asym profile or if it is required for Stage 4 activation
  offload. If generic dense wrapping already achieved the same base-weight
  memory reduction and this stage adds latency, reject it or keep only the
  minimal scaffolding needed for Stage 4.

Risks to watch:

- Qwen3.5 generic dense offload may already save most shared expert base memory,
  so this stage alone may have little memory effect.
- Preexisting PEFT LoRA modules may carry adapter-specific scaling. Copy the
  Llama4 scaling validation exactly so load/save roundtrips remain compatible.
- Do not include `shared_expert_gate` in the MLP wrapper. Its scalar output
  shape is a poor fit for the direct Asym BF16 dx kernel, and folding it into
  the MLP would change the source graph and saved tensors.

## Stage 4: Add Qwen3.5 Shared-MLP Activation Offload/Backfetch

Files/functions/classes to modify:

- `asym_gemm/training/qwen35_shared_mlp.py`
  - `_Qwen35SharedMLPActivationOffloadFunction`
  - `_shared_mlp_activation_offload_enabled`
  - `_is_silu_activation`
  - `_asym_base_forward`
  - `_asym_base_dx`
  - `_lora_forward`
  - `_lora_backward`
  - `AsymQwen35SharedMLP._activation_offload_supported`
  - `AsymQwen35SharedMLP.forward`
- `tests/training/test_lf_qwen35_asym_backend.py`
  - CUDA-gated forward/backward parity for shared MLP activation offload
  - saved activation/residency counters for `ASYMM_EXPERT_ACT_OFFLOAD=true`.

Intended code changes:

- Copy Llama4 shared MLP activation offload and rename it for Qwen3.5. This is
  safe because the math is identical: `down(silu(gate(x)) * up(x))`.
- Use the same env knobs as Qwen3/Llama4 so `profile3.sh` exercises the feature:
  `ASYMM_EXPERT_ACT_OFFLOAD=true` or
  `ASYM_GEMM_LF_CONFIG_ASYMM_EXPERT_ACT_OFFLOAD=true`.
- Use the activation offload path only under strict conditions:
  `backend=="asym"`, shared base weights are offloaded, training mode, grad
  enabled, LoRA dropout is zero, LoRA dtype is BF16, input is CUDA BF16, activation
  is SiLU, and all three leaves are `AsymLoRALinear` over BF16
  `AsymFrozenLinear` bases.
- Keep the forward as three large dense projections. The backward should stage
  `x` once, recompute gate/up and activation once, compute down dx/LoRA grads,
  then compute gate/up dx/LoRA grads. No token loops and no expert loops.

Pseudocode:

```python
class _Qwen35SharedMLPActivationOffloadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_a, gate_b, up_a, up_b, down_a, down_b, layer):
        flat = x.reshape(-1, layer.hidden_size).contiguous()
        require flat.dtype == torch.bfloat16

        manager = ActivationOffloadManager(pin_memory=True)
        x_cpu = manager.offload(flat, "qwen35_shared.X")

        gate_base = _asym_base_forward(layer.gate_proj.base_layer, flat, tag="shared_gate.base_forward")
        up_base = _asym_base_forward(layer.up_proj.base_layer, flat, tag="shared_up.base_forward")
        gate_delta, _ = _lora_forward(flat, gate_a, gate_b, scale=layer.lora_scale)
        up_delta, _ = _lora_forward(flat, up_a, up_b, scale=layer.lora_scale)
        gate = gate_base + gate_delta.to(gate_base.dtype)
        up = up_base + up_delta.to(up_base.dtype)

        activated = layer.activation_fn(gate) * up

        down_base = _asym_base_forward(
            layer.down_proj.base_layer,
            activated.to(torch.bfloat16).contiguous(),
            tag="shared_down.base_forward")
        down_delta, _ = _lora_forward(
            activated.to(down_a.dtype).contiguous(),
            down_a, down_b, scale=layer.lora_scale)
        out = down_base + down_delta.to(down_base.dtype)

        ctx.save_for_backward(gate_a, gate_b, up_a, up_b, down_a, down_b)
        ctx.layer = layer
        ctx.manager = manager
        ctx.x_cpu = x_cpu
        ctx.input_shape = tuple(x.shape)
        ctx.input_dtype = x.dtype
        return out.reshape(*x.shape[:-1], layer.hidden_size)

    @staticmethod
    def backward(ctx, grad_output):
        gate_a, gate_b, up_a, up_b, down_a, down_b = ctx.saved_tensors
        layer = ctx.layer
        manager = ctx.manager
        x_stage = manager.stage(ctx.x_cpu, tag="qwen35_shared.X_for_backward")
        x_lora = x_stage.to(gate_a.dtype)

        gate_base = _asym_base_forward(layer.gate_proj.base_layer, x_stage, tag="shared_gate.recompute")
        up_base = _asym_base_forward(layer.up_proj.base_layer, x_stage, tag="shared_up.recompute")
        gate_delta, gate_low_rank = _lora_forward(x_lora, gate_a, gate_b, scale=layer.lora_scale)
        up_delta, up_low_rank = _lora_forward(x_lora, up_a, up_b, scale=layer.lora_scale)
        gate = gate_base + gate_delta.to(gate_base.dtype)
        up = up_base + up_delta.to(up_base.dtype)
        activated = layer.activation_fn(gate) * up

        grad_2d = grad_output.reshape(-1, layer.hidden_size).to(torch.bfloat16).contiguous()
        grad_down_base_x = _asym_base_dx(layer.down_proj.base_layer, grad_2d, input_dtype=torch.bfloat16)
        _, down_low_rank = _lora_forward(activated.to(down_a.dtype).contiguous(), down_a, down_b, scale=layer.lora_scale)
        grad_down_lora_x, grad_down_a, grad_down_b = _lora_backward(
            grad_2d, down_low_rank, activated.to(down_a.dtype).contiguous(),
            down_a, down_b, scale=layer.lora_scale)
        grad_act = grad_down_base_x + grad_down_lora_x.to(grad_down_base_x.dtype)

        grad_up = grad_act * layer.activation_fn(gate)
        grad_gate = torch.ops.aten.silu_backward(grad_act * up, gate)

        grad_gate_base_x = _asym_base_dx(layer.gate_proj.base_layer, grad_gate.to(torch.bfloat16).contiguous())
        grad_up_base_x = _asym_base_dx(layer.up_proj.base_layer, grad_up.to(torch.bfloat16).contiguous())
        grad_gate_lora_x, grad_gate_a, grad_gate_b = _lora_backward(
            grad_gate, gate_low_rank, x_lora, gate_a, gate_b, scale=layer.lora_scale)
        grad_up_lora_x, grad_up_a, grad_up_b = _lora_backward(
            grad_up, up_low_rank, x_lora, up_a, up_b, scale=layer.lora_scale)

        grad_x_2d = grad_gate_base_x + grad_up_base_x
        grad_x_2d.add_(grad_gate_lora_x.to(grad_x_2d.dtype))
        grad_x_2d.add_(grad_up_lora_x.to(grad_x_2d.dtype))
        grad_x = grad_x_2d.to(ctx.input_dtype).reshape(ctx.input_shape)

        manager.release_stage(x_stage, drop_cache=True)
        manager.release_cpu(ctx.x_cpu)
        return grad_x, grad_gate_a, grad_gate_b, grad_up_a, grad_up_b, grad_down_a, grad_down_b, None
```

Validation commands:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

ASYMM_EXPERT_ACT_OFFLOAD=true \
pytest -q tests/training/test_lf_qwen35_asym_backend.py -k 'shared and activation'

pytest -q \
  tests/training/test_lf_qwen35_asym_backend.py \
  tests/training/test_lf_llama4_asym_backend.py \
  tests/training/test_lf_qwen3_asym_backend.py
```

Then run the reusable Asym e2e validation command, changing only:

```bash
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/outputs/qwen35_stage4
```

Acceptance before Stage 5:

- Unit tests prove forward and backward parity for the Qwen3.5 shared MLP
  activation-offload path.
- `memory_breakdown_summary.json` shows actual peak HBM or saved activation HBM
  drops meaningfully versus the previous successful Asym profile. The drop
  should be attributable to `shared_experts`, `mlp`, or activation/saved tensor
  rows, not allocator noise.
- Forward/backward/e2e timing stays within the global latency threshold. If
  shared activation offload reduces memory but adds many small operations or
  excessive H2D/D2H copies, reject it.

Risks to watch:

- The shared branch may not be the actual peak at `WORKLOADS=2048|4|1`; in that
  case this stage can be correct but not worth keeping.
- `ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD` is routed-expert-specific today. Do not
  invent a separate shared-MLP interpretation unless profiling shows shared LoRA
  low-rank staging is a real peak contributor.
- `shared_expert_gate` still saves its input for backward through generic
  autograd. If the breakdown shows the scalar gate is a peak contributor, add a
  separate small activation-offload gate wrapper later, but do not fold it into
  this stage.

## Stage 5: Verify Or Implement Shared-MLP LoRA Weight Offload

`profile3.sh` sets `ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true`, so this stage is not
optional as a validation gate. The implementation work is conditional: if the
existing generic dense/CPUAdamW path already offloads Qwen3.5 shared-MLP LoRA
weights correctly, add tests and keep the code unchanged. If the Stage 3/4
shared wrapper owns LoRA tensors in a way the generic coordinator cannot see,
implement grouped shared-MLP LoRA weight offload here.

Files/functions/classes to inspect in all cases:

- `asym_gemm/training/weight_offload.py`
  - current trainable LoRA tensor discovery and coordinator lifetime model.
- `third_party/LlamaFactory/src/llamafactory/train/trainer_utils.py`
  - `_create_asym_cpu_adamw_optimizer`
- `tests/training/test_lf_qwen35_asym_backend.py`
  - trainable surface and CPUAdamW weight-offload coverage.

Files/functions/classes to modify only if the existing path does not cover the
Qwen3.5 shared wrapper:

- `asym_gemm/training/weight_offload.py`
  - `LoRAWeightOffloadCoordinator`
  - registration/discovery for trainable LoRA tensors
  - forward/backward gather/release hooks
- `asym_gemm/training/qwen35_shared_mlp.py`
  - safe gather/release points if the shared MLP custom autograd function owns
    the LoRA tensors during forward and backward.
- `third_party/LlamaFactory/src/llamafactory/train/trainer_utils.py`
  - `_create_asym_cpu_adamw_optimizer` error/reporting text so it does not imply
    only `AsymQwen3Experts` LoRA banks can be coordinated.
- `tests/training/test_lf_qwen35_asym_backend.py`
  - shared MLP LoRA weight offload parity and lifetime tests.

Intended code changes:

- The current weight offload coordinator is designed for `AsymQwen3Experts`
  LoRA banks. Qwen3.5 routed expert banks already inherit that support because
  Qwen3.5 uses `AsymQwen3Experts`.
- First prove whether Qwen3.5 shared-MLP LoRA tensors are already discovered by
  the generic path under `ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true`. The proof is
  explicit parameter names, residency/lifetime accounting, and a trainable
  surface check, not an assumption.
- Only add new shared MLP trainable LoRA weight-offload code if the existing path
  misses those tensors or if the new shared wrapper uses custom autograd and
  therefore needs explicit gather/release points. They are much smaller than
  routed expert banks, so do not add a new runtime path unless correctness or a
  meaningful memory drop requires it.
- If implemented, register all six shared MLP LoRA tensors as one per-layer
  group, not six independent tiny transfer groups.
- Never release a LoRA tensor before autograd no longer needs it. Generic
  `AsymLoRALinear` autograd will save parameter tensors implicitly, so this
  stage is only safe if the shared MLP is using the custom activation-offload
  function that manually controls LoRA forward/backward math, or if releases are
  delayed by a post-accumulate/backward-completion hook.

Pseudocode:

```python
def discover_lora_weight_offload_groups(model):
    groups = []
    for name, module in model.named_modules():
        if isinstance(module, AsymQwen3Experts):
            groups.append(register_existing_expert_bank_group(name, module))
        if isinstance(module, AsymQwen35SharedMLP) and module.uses_custom_autograd_weight_lifetime:
            groups.append(LoRAWeightGroup(
                name=f"{name}.shared_mlp_lora",
                tensors=[
                    module.gate_proj.lora_a,
                    module.gate_proj.lora_b,
                    module.up_proj.lora_a,
                    module.up_proj.lora_b,
                    module.down_proj.lora_a,
                    module.down_proj.lora_b,
                ],
                gather_before_forward=module.gather_lora_weights,
                release_after_forward=module.release_lora_weights,
                gather_before_backward=module.gather_lora_weights,
                release_after_backward=module.release_lora_weights,
            ))
    return groups
```

Validation commands:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

ASYMM_EXPERT_ACT_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
pytest -q tests/training/test_lf_qwen35_asym_backend.py -k 'weight_offload or shared'
```

Add or update a test that prints/asserts the exact Qwen3.5 trainable LoRA names
seen by the optimizer under `--lora_target all`. The accepted trainable surface is:

- routed expert split LoRA banks from `AsymQwen3Experts`;
- shared expert dense LoRA weights for `shared_expert.gate_proj`,
  `shared_expert.up_proj`, and `shared_expert.down_proj`;
- `shared_expert_gate` LoRA if it remains selected by `lora_target=all`;
- GDN projection LoRA for `linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}`
  if Stage 2 leaves them PEFT-owned or moves them into Asym-owned dense wrappers.

No router base weights, dense base weights, norms, embeddings, or GDN
non-projection tensors may be trainable.

Then run the reusable Asym e2e validation command, changing only:

```bash
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/outputs/qwen35_stage5
```

Acceptance before Stage 6:

- The validation gate is passed only when `ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true`
  correctly covers every Qwen3.5 trainable LoRA tensor that the final config
  creates. If the existing path already does this, keep the verification/tests
  and do not add new runtime code.
- Keep any new runtime offload implementation only if Qwen3.5 e2e peak HBM drops
  meaningfully beyond Stage 4. A small trainable LoRA memory reduction is not
  enough.
- Backward must be numerically correct and must not crash from missing/released
  LoRA parameters.
- Transfer scheduling must not create many tiny H2D/D2H operations. Use grouped
  slabs or one coordinated per-layer transfer.

Risks to watch:

- This stage has the highest correctness risk because LoRA parameter lifetimes
  cross autograd boundaries.
- It is likely not needed for `LORA_RANK=64` unless memory breakdown shows
  trainable dense/shared LoRA weights are unexpectedly large.

## Stage 6: Final Verdict From Exact `profile3.sh` Qwen3.5 Numbers

Files to use, not necessarily modify:

- `scripts/lf/profile3.sh`
- `scripts/lf/postprocess_lf_profile_artifacts.py`
- final output directory selected below.

Final profiling command:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

NUMACTL_MEMBIND=0,1 \
NUMACTL_CPUNODEBIND=0,1 \
NUMACTL_MODE=membind \
OUTPUT_ROOT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/outputs/qwen35_final \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=true \
MODEL_SPECS='Qwen/Qwen3.5-122B-A10B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0,zero3_offload|recomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|true' \
ASYM_OFFLOAD_MODULES=all \
ASYM_STRICT=true \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
LF_EXPERT_LORA_IMPLS=split-target-parameters \
LORA_DROPOUT=0.00 \
LORA_RANK=64 \
LORA_ALPHA=16 \
WORKLOADS='2048|4|1' \
MAX_STEPS=1 \
WARMUP_STEPS=1 \
bash scripts/lf/profile3.sh --overwrite true
```

Final extraction command:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

python - <<'PY'
from pathlib import Path
import json

root = Path("outputs/qwen35_final")
rows = []
for path in root.rglob("profile.json"):
    profile = json.loads(path.read_text())
    source = profile.get("source_profile") or profile
    if source.get("partial") is True:
        continue
    config = source.get("config", {})
    mem = source.get("memory", {}).get("gpu", {})
    timing = source.get("trainer", {}).get("timing", {})
    row = {
        "backend": config.get("backend"),
        "liger_loss": config.get("liger_loss"),
        "activation_recompute": config.get("activation_recompute"),
        "model": config.get("model_name_or_path"),
        "lora_target": config.get("lora_target"),
        "offload_modules": config.get("asym_offload_modules"),
        "strict": config.get("asym_strict"),
        "qwen_moe_expert_lora_impl": config.get("qwen_moe_expert_lora_impl"),
        "cpuadam_grad_offload": config.get("asym_cpu_adamw_grad_offload"),
        "cpuadam_weight_offload": config.get("asym_cpu_adamw_weight_offload"),
        "expert_policy": config.get("expert_recompute_policy"),
        "exp_act": config.get("asymm_expert_act_offload"),
        "attn_act": config.get("asymm_attn_act_offload"),
        "layer_act": config.get("asymm_layer_act_offload"),
        "peak_alloc_gib": (mem.get("peak_allocated_hbm_bytes") or 0) / 2**30,
        "peak_reserved_gib": (mem.get("peak_reserved_hbm_bytes") or 0) / 2**30,
        "forward_ms": source.get("forward", {}).get("total_milliseconds"),
        "backward_ms": source.get("backward", {}).get("total_milliseconds"),
        "e2e_ms": timing.get("measured_e2e_step_milliseconds"),
        "path": str(path),
    }
    for candidate in (
        path.parent / "memory_breakdown_summary.json",
        path.parent / "memory_breakdown" / "memory_breakdown_summary.json",
    ):
        if candidate.exists():
            summary = json.loads(candidate.read_text())
            row["actual_peak_alloc_gib"] = (summary.get("actual_peak_allocated_hbm_bytes") or 0) / 2**30
            row["actual_peak_reserved_gib"] = (summary.get("actual_peak_reserved_hbm_bytes") or 0) / 2**30
            row["actual_peak_phase"] = summary.get("actual_peak_phase")
            break
    rows.append(row)

for row in sorted(rows, key=lambda r: (str(r["backend"]), str(r["path"]))):
    print(json.dumps(row, sort_keys=True))
PY
```

Final acceptance:

- Final source profile configs must prove the exact knobs used for the verdict:
  `model_name_or_path=Qwen/Qwen3.5-122B-A10B`, `lora_target=all`,
  `liger_loss=ligerloss0`,
  `qwen_moe_expert_lora_impl=split-target-parameters`,
  `asym_offload_modules=all`, `asym_strict=true`,
  `asym_cpu_adamw_grad_offload=true`,
  `asym_cpu_adamw_weight_offload=true`, `lora_rank=64`, `lora_alpha=16`, and
  `lora_dropout=0.00`. The run log must also show
  `NUMACTL_MEMBIND=0,1`, `NUMACTL_CPUNODEBIND=0,1`, and
  `NUMACTL_MODE=membind`; reject any final artifact whose `MEMBIND` includes a
  non-CPU NUMA node.
- `asym_cpuadamwds|norecomp|ligerloss0` must have lower Qwen3.5 peak HBM than
  `zero3_offload|recomp|ligerloss0` by at least the global memory rule. Prefer
  `actual_peak_allocated_hbm_bytes` and `actual_peak_reserved_hbm_bytes` from
  `memory_breakdown_summary.json`; fall back to `memory.gpu` allocator peaks
  only if breakdown artifacts are missing.
- `asym_cpuadamwds|norecomp|ligerloss0` forward, backward, and measured e2e step times must
  not exceed the global latency threshold versus the first successful or most
  recent accepted `asym_cpuadamwds|norecomp|ligerloss0` Qwen3.5 profile. If there is only
  one successful Asym profile by Stage 6, judge it directly against
  `zero3_offload|recomp|ligerloss0`: it must win memory meaningfully without obviously
  worse forward/backward/e2e timing.
- Runtime health must show the intended code paths:
  - `qwen35_moes_wrapped > 0`
  - `packed_experts_wrapped`/`qwen3_experts_wrapped` reflects routed expert use
  - `dense_lora_wrapped` includes shared expert leaves or the shared MLP wrapper
  - selected `shared_experts`, `router`, `attention`, `embed_tokens`, `lm_head`,
    and `norms` have no frozen selected CUDA residues
  - if Stage 2 projection offload was accepted, selected `linear_attention` has
    no frozen selected CUDA residues and the breakdown has `linear_attention`
    rows; if Stage 2 projection offload was rejected, the breakdown still has
    `linear_attention` rows so the rejection is evidence-based
  - `asym_forward_calls` and `asym_dx_calls` are nonzero
  - no unexpected reference fallback spikes
- If final memory is the same as zero3 or only trivially lower, reject the last
  implemented stage that failed to move the profile. If memory improves but
  forward/backward latency regresses beyond the threshold, reject the stage that
  introduced the regression.

Unresolved risks to watch through the final verdict:

- The biggest missing Qwen3.5-specific feature is likely shared MLP activation
  offload, not routed expert offload. However, the profile peak may be attention,
  Gated DeltaNet, loss/logits, or optimizer state instead.
- Qwen3.5 linear-attention layers depend on optional FLA/causal-conv kernels.
  If those are missing, Transformers/LlamaFactory may fall back to slower and
  more memory-hungry PyTorch paths, hiding the effect of MoE changes.
- CPU offload latency is sensitive to pinned memory, NUMA placement, C2C/PCIe
  topology, and concurrent processes. Compare stages on the same host/GPU with
  identical `profile3.sh` knobs.
