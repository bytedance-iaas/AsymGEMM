# AsymGEMM-SFT Roadmap

## Goal

Beat KTransformers on LoRA fine-tuning for MoE models by keeping frozen base and expert weights in CPU host memory while GPU tensor cores compute the heavy base/expert matmuls.

The first credible target is not full-weight fine-tuning. It is BF16 LoRA SFT where:

- frozen base/expert weights live in CPU pinned or mapped host memory;
- forward and `dX` through frozen weights run on GPU, not CPU;
- LoRA parameters, activations, routing, optimizer state, and loss stay on GPU;
- base/expert `dW` is intentionally absent because frozen LoRA SFT does not need it;
- KTransformers is measured on the same hardware, model, precision tier, and SFT setup.

Defensible claim:

```text
AsymGEMM can beat CPU-compute KTransformers for MoE LoRA SFT when direct host-weight fetch gives enough reuse over the available CPU-GPU interconnect.
```

## Scope

Initial scope:

- BF16 only.
- Single GPU only.
- LoRA SFT only.
- Frozen base and expert weights only.
- Direct `dX` is required before any training-speed claim.
- Fallback `dX` is allowed for correctness bring-up, but not for the final KT win gate.

Explicitly out of initial scope:

- full fine-tuning and base `wgrad`;
- FP8 training claims;
- DeepSpeed, FSDP, ZeRO, Unsloth, DoRA, OFT, PiSSA conversion, merge/unload;
- multi-GPU and large-model claims before the tiny and small-model ladders pass.

## Hardware Scope

| Class | Use In This Roadmap | Notes |
| --- | --- | --- |
| Intel-AMX CPU + NVIDIA H200 over PCIe/host path | Only target hardware class | KT AMXBF16, H2D staging, and AsymGEMM direct-fetch claims are all measured here. |

This roadmap is H200-only. CPU-resident weight access is over the available host-GPU path on this machine, not a special coherent CPU-GPU link. If direct fetch loses to optimized H2D staging on H200, the result is still useful, but the paper claim must pivot to a boundary analysis instead of a speedup claim.

## Current Reality

AsymGEMM currently provides forward, output-buffer style pybind APIs. There is no public autograd, dgrad, or wgrad API.

Current relevant forward APIs:

- `m_grouped_bf16_asym_gemm_nt_contiguous`
- `m_grouped_bf16_asym_gemm_nt_masked`
- `m_grouped_fp8_asym_gemm_nt_contiguous`
- `m_grouped_fp8_asym_gemm_nt_masked`
- `fp8_gemm_nt`
- `k_grouped_fp8_gemm_nt_contiguous`
- layout and runtime helpers

H200 is SM90. BF16 and FP8 forward smoke coverage is relevant, but this is not training coverage.

## Kernel Layout Terms

`contiguous` grouped mode:

```text
A: [active_tokens, K]
B: [num_experts, N, K]
D: [active_tokens, N]
metadata: offsets + experts + list_size
```

Use it when routed tokens are packed into a compact active-token buffer. It minimizes padded work but requires route sorting and packing.

`masked` grouped mode:

```text
A: [num_experts, max_m, K]
B: [num_experts, N, K]
D: [num_experts, max_m, N]
metadata: masked_m + expected_m
```

Use it when each expert has a fixed slot and only the first `masked_m[g]` rows are valid. It is simpler and more graph-friendly but pays padding cost.

MoE milestones must test both and log active tokens, padded tokens, empty experts, route pack/scatter time, and grouped GEMM time.

## Autograd Contract

For a frozen base linear:

```text
Y = X @ W_base.T + bias + LoRA_B(LoRA_A(X)) * scale
```

Required for LoRA SFT:

- `W_base`: CPU host-memory handle, frozen, not an `nn.Parameter`.
- `W_base_T`: optional CPU host-memory copy shaped `[in_features, out_features]` so an NT kernel can compute `dX = dY @ W_base`.
- forward base term: AsymGEMM direct fetch when supported, staged/Torch fallback otherwise.
- backward base term: direct-fetch `dX` for performance claims; staged/Torch fallback only for correctness bring-up.
- base `dW`: always `None` in LoRA SFT.
- LoRA `dA/dB`: normal PEFT GPU autograd.

Replacement base layer contract:

- same output shape and dtype as the original PEFT `base_layer`;
- accepts `*args, **kwargs` used by PEFT;
- does not detach or run under `torch.no_grad()`;
- does not register CPU weight handles as parameters or persistent buffers;
- blocks accidental `.to(cuda)` movement of frozen base weights;
- preserves `in_features`, `out_features`, `bias`, adapter names, state dict behavior, and optimizer filtering.

Dense linear may start as a one-group BF16 m-grouped call. If that is too awkward or too padded, add a real non-grouped BF16 AsymGEMM binding.

## Progress Tracker

Update this table after each milestone run.

| Milestone | Status | Evidence Artifact | Last Run | Blocker |
| --- | --- | --- | --- | --- |
| M0 Baseline Reality | Done | `agent/current_state.md`, `agent/decisions.md`, `tests/training/test_00_m0_smoke.py` | 2026-05-18 | |
| M1 BF16 Frozen Linear Primitive | Done | `asym_gemm/training/`, `tests/training/test_01_frozen_linear.py` | 2026-05-18 | |
| M2 Viable MLP Demo | Done | `examples/asymgemm/mlp_lora_demo.py`, `tests/training/test_02_mlp_demo.py`, `reports/mlp_demo.json` | 2026-05-18 | |
| M3 Tiny LLM Correctness | Done | `reports/m3_tiny_llm.json`, `tests/training/test_03_tiny_dense_llm.py` | 2026-05-18 | |
| M4 Tiny MoE Correctness | Done - stop point reached | `reports/m4_tiny_moe.json`, `tests/training/test_04_tiny_moe.py` | 2026-05-18 | Review stop report before M5 |
| M4.5 Fine-Grained Profiling Gate | Not started | `reports/m4_5_profile_summary.json`, `reports/m4_5_profile_summary.md` | | Required before M5 |
| M5 LLaMA-Factory Integration | Not started | `reports/m5_llamafactory.json` | | |
| M6 KT Benchmark Reproduction | Not started | `reports/m6_kt_repro.json` | | |
| M7 Optimize And Beat-KT Gate | Not started | `reports/m7_beat_kt.json` | | |

Evidence artifacts must record commit, host, GPU, CPU, NUMA binding, CUDA, PyTorch, LLaMA-Factory, KTransformers, AsymGEMM, precision, fallback counts, peak HBM, CPU RSS, pinned bytes, and command line.

## Milestones

### M0 - Baseline Reality

Deliverables:

- `agent/current_state.md` documents actual exports, arch gates, supported dtypes, missing backward APIs, and stale tests.
- Fast smoke command verifies import, `_C` bindings, and one forward m-grouped kernel when hardware permits.
- Stale exports/tests are fixed or skipped with explicit reason.
- Decision log starts at `agent/decisions.md`.

Acceptance:

- `python -c "import asym_gemm; print(asym_gemm.__version__)"` succeeds.
- Exported Python names match `_C` bindings.
- BF16/FP8 forward tests are marked by real H200 support, not assumed support.
- BF16 SM90/H200 dispatch status is recorded correctly.

### M1 - BF16 Frozen Linear Primitive

Deliverables:

- CPU host-weight handle with allocation metadata:
  - `pin_memory`, `cudaHostAllocMapped`, or `cudaHostRegister`;
  - pointer attributes;
  - `canMapHostMemory`;
  - NUMA node and CPU socket.
- Output-allocating Python wrappers around in-place kernels.
- Dense BF16 frozen linear via one-group m-grouped API or a new dense BF16 binding.
- `AsymFrozenLinearFunction` with:
  - Asym/staged/Torch forward modes;
  - direct-fetch `dX` path;
  - staged/Torch `dX` fallback for bring-up;
  - base `dW = None`.
- Fallback policy:
  - `asym_only`
  - `asym_or_staged`
  - `asym_or_torch`
  - `torch_only`
- Capability microbench records whether the kernel can legally consume the host pointer and measures in-kernel host-read bandwidth.

Acceptance:

- FP64 formula gradcheck passes for the reference wrapper.
- FP32 wrapper vs Torch passes output/loss/grad parity.
- BF16 direct or staged path matches Torch within documented tolerance.
- Frozen base weight has `grad is None`.
- LoRA parameters receive gradients when composed through PEFT.
- No accidental `.to(cuda)` of CPU host weights.
- Direct `dX` is available or the milestone records why the KT speed claim is blocked.

### M2 - Viable MLP LoRA Demo

Purpose: a small runnable demo that proves the training idea before LLM and MoE complexity.

Deliverables:

- `examples/asymgemm/mlp_lora_demo.py`
- `tests/training/test_02_mlp_demo.py`
- One command:

```bash
python examples/asymgemm/mlp_lora_demo.py --backend asym_or_staged --report reports/mlp_demo.json
```

Demo model:

- 2-layer MLP;
- deterministic seeds;
- no dropout;
- TF32 disabled;
- dimensions chosen or padded so real AsymGEMM kernels execute;
- LoRA on both frozen base linears.

Report fields:

- forward parity;
- scalar loss parity;
- `input.grad` parity;
- LoRA `A/B` grad parity;
- one optimizer step loss movement;
- frozen base weights unchanged;
- base weights absent from optimizer state;
- number of AsymGEMM calls;
- fallback counts;
- pinned CPU bytes;
- peak HBM;
- Torch vs staged vs direct-fetch timings.

Acceptance:

- One optimizer step updates only LoRA parameters.
- Frozen base handles remain CPU-resident.
- Output/loss/`dX`/LoRA grad parity is within numeric tolerance.
- `reports/mlp_demo.json` is emitted and includes nonzero AsymGEMM calls or a hard failure in `asym_only`.

### M3 - Tiny Dense LLM Correctness

Deliverables:

- 4-layer decoder-only random model:
  - vocab `512`
  - hidden `128`
  - heads `4`
  - seq `8`
  - batch `2`
  - intermediate `256`
  - causal CE loss
- Progressive replacement:
  1. MLP linears only
  2. attention projections only
  3. all LoRA target linears
- Checkpointing on/off tests.

Acceptance:

- Logits, loss, selected activations, `input.grad`, and LoRA grads match Torch within BF16 tolerance.
- 5 repeated optimizer steps have no NaN/Inf.
- Loss curve tracks Torch within `<= 2%`.
- Adapter save/reload works with unchanged PEFT parameter names.

### M4 - Tiny MoE Correctness

Deliverables:

- 4-layer tiny transformer MoE:
  - experts `4`
  - shared experts `1`
  - top-k `2`
  - vocab `512`
  - batch `2`
  - sequence length `8`
  - hidden `128`
  - attention heads `4`
  - expert intermediate `256`
  - logical tokens `16`
- Dense mini-LLM shell:
  - token embeddings;
  - position embeddings;
  - causal self-attention;
  - frozen layer norms and residual paths;
  - final norm and LM head;
  - shifted-label causal loss.
- GPU route pack/scatter:
  - token expansion by top-k;
  - sort/group by expert;
  - contiguous metadata;
  - masked metadata;
  - scatter with routing weights;
  - scatter backward.
- Grouped expert forward and grouped expert `dX` for gate/up/down projections.
- Tests for balanced, empty, skewed/Zipf, repeated-expert, and repeated-backward cases.

Acceptance:

- Static routing parity passes before learned router.
- Learned router gradients match Torch reference.
- Expert LoRA grads, combined hidden states, and input grads match BF16 tolerances.
- Both contiguous and masked grouped modes are tested.
- 20 toy training steps run without NaN/Inf or stale-buffer failures.

## STOP POINT - Pause After Tiny MoE

When M4 is complete, stop execution and report status before starting M5 or any later milestone.

Required stop report:

- exact M1-M4 pass/fail status;
- correctness gaps, if any;
- direct-fetch vs staged fallback status for forward and `dX`;
- contiguous and masked grouped-mode coverage;
- peak HBM, pinned CPU bytes, and toy-step timing from the M4 report;
- recommendation on whether to continue to LLaMA-Factory integration.

M4 stop report, 2026-05-18:

- M1: pass; M2: pass; M3: pass; M4: pass, stop point reached.
- Correctness gaps: none found for the tiny transformer-style MoE route pack/scatter, shared+routed expert path, or per-expert frozen-linear path. Native fused grouped expert GEMM performance is not proven by M4 and remains an optimization/benchmarking risk before any KT-speed claim.
- Architecture audit: M4 now mirrors the dense mini LLM shell with token/position embeddings, causal attention, residual decoder blocks, final norm, LM head, shifted-label loss, and a MoE FFN made from always-on shared experts plus top-k routed experts.
- Direct fetch: forward used, `dX` used; staged calls `0`, Torch fallback calls `0`.
- Grouped modes: contiguous and masked metadata/pack/scatter covered; balanced, empty, skewed, and repeated-expert routes covered; repeated backward covered.
- M4 report metrics: peak HBM `126,958,592` bytes, pinned CPU `7,208,960` bytes during toy training, toy step `0.066745` seconds/step. The normal-vs-Asym model allocation comparison records `3,932,160` bytes model HBM saved with `7,864,320` bytes pinned CPU after transpose materialization.
- Recommendation: M4 is usable as the tiny MoE SFT correctness gate. Continue to M5 only after reviewing that the remaining risk is performance/benchmarking, not model architecture correctness.

Do not begin LLaMA-Factory integration, KTransformers benchmarking, or larger-model work until this stop point is reviewed.

### M4.5 - Fine-Grained Profiling Gate

Purpose: before integrating with LLaMA-Factory or making any performance claim, produce a comprehensive latency, CPU-memory, and GPU-memory breakdown for the three controlled demos: MLP LoRA, tiny dense LLM, and tiny transformer MoE. This milestone is profiling and analysis only; do not change kernels, model logic, or training behavior as part of M4.5.

Scope:

- Workloads:
  - MLP LoRA demo from M2;
  - tiny dense LLM from M3;
  - tiny transformer MoE from M4.
- Backends/modes:
  - Torch GPU-resident baseline;
  - staged fallback path where supported;
  - direct AsymGEMM path with `asym_only` on H200;
  - dense target modes `mlp_only`, `attention_only`, and `all` where applicable;
  - MoE grouped metadata modes `contiguous` and `masked`.
- Phases:
  - setup and host-weight construction;
  - host pinning and `W.T` materialization;
  - forward;
  - loss;
  - backward;
  - optimizer step;
  - report serialization and cleanup excluded from step timing unless reported separately.

Deliverables:

- Profiling entrypoint, to be implemented after this roadmap update:
  - `scripts/profile_asymgemm_sft.py`
- Per-workload machine-readable reports:
  - `reports/m4_5_mlp_profile.json`
  - `reports/m4_5_dense_llm_profile.json`
  - `reports/m4_5_tiny_moe_profile.json`
- Cross-workload summary:
  - `reports/m4_5_profile_summary.json`
  - `reports/m4_5_profile_summary.md`
- Optional schema/guard test:
  - `tests/training/test_04_5_profile_schema.py`

Latency breakdown requirements:

- Report total step latency and percent-of-step for every major stage:
  - input/token preparation;
  - host-weight pointer preparation;
  - route logits and top-k router for MoE;
  - route pack/sort/metadata construction;
  - frozen base forward AsymGEMM or Torch/staged equivalent;
  - LoRA forward;
  - attention forward for dense LLM and MoE;
  - MoE expert `gate`, `up`, activation, and `down`;
  - route scatter;
  - loss;
  - frozen base `dX`;
  - LoRA gradient computation;
  - router gradient computation for MoE;
  - optimizer step;
  - explicit synchronization/copy/fallback overhead.
- Report latency at multiple granularities:
  - whole training step;
  - forward vs backward vs optimizer;
  - per layer;
  - per projection class such as attention `q/k/v/o`, MLP `gate/up/down`, MoE shared experts, and MoE routed experts;
  - per backend mode.
- Include distribution statistics over repeated measured steps:
  - warmup count;
  - measured step count;
  - mean, median, p90, p95, min, max, and standard deviation;
  - CUDA event time and wall-clock time, with methodology clearly labeled.

GPU memory breakdown requirements:

- Report total and percent breakdown for:
  - frozen base/expert weights that would be GPU-resident in the Torch baseline;
  - trainable LoRA parameters;
  - router parameters;
  - embeddings, LM head, attention weights, and layernorm buffers;
  - optimizer states;
  - activations saved for backward;
  - route metadata, packed token buffers, scatter buffers, and MoE workspace;
  - staged weight buffers if any fallback path uses them;
  - kernel temporary/workspace allocations when measurable;
  - peak allocated, peak reserved, current allocated, and fragmentation/reserved-minus-allocated.
- Report both absolute bytes and percentages:
  - percent of peak allocated HBM;
  - percent of model-state HBM;
  - percent of total training-step HBM delta.

CPU memory breakdown requirements:

- Report total and percent breakdown for:
  - CPU-resident frozen `W` weights;
  - CPU-resident transposed `W.T` weights used for `dX`;
  - pinned/page-locked bytes;
  - pageable CPU bytes;
  - staging buffers;
  - route metadata mirrored on CPU, if any;
  - process RSS and peak RSS;
  - NUMA node and CPU socket placement where available.
- Explicitly separate:
  - CPU memory required by AsymGEMM host weights;
  - CPU memory required only by reporting/profiling;
  - CPU memory that would exist in the Torch baseline.

Required percentage accounting:

- Every profile report must include tables where component bytes and component latency sum to the reported total within a documented tolerance.
- Required percentage views:
  - latency percent of full step;
  - latency percent of forward;
  - latency percent of backward;
  - GPU bytes percent of peak HBM;
  - CPU bytes percent of total RSS;
  - pinned bytes percent of total CPU-resident frozen-weight storage.
- If any category cannot be measured directly, the report must mark it as `estimated` or `unattributed`, include the estimation method, and keep `unattributed_percent` visible.

Comparison requirements:

- For each workload, compare:
  - Torch GPU-resident baseline vs AsymGEMM direct;
  - direct fetch vs staged fallback where supported;
  - forward-only vs full training step;
  - forward base matmul vs backward `dX`;
  - LoRA overhead vs frozen base matmul overhead;
  - memory saved in HBM vs extra pinned CPU memory required.
- For tiny MoE, additionally compare:
  - router time;
  - pack/sort time;
  - contiguous vs masked metadata;
  - routed expert time;
  - shared expert time;
  - scatter time;
  - imbalance statistics: expert token counts, empty experts, max/min/mean routes per expert, and padded-route percent.

Leadership summary requirements:

- The markdown summary must answer, for MLP, dense LLM, and MoE:
  - where the step time goes;
  - where HBM goes;
  - where CPU memory goes;
  - how much HBM AsymGEMM saves;
  - how much pinned CPU memory AsymGEMM costs;
  - whether direct fetch is faster or slower than staged/Torch for forward and `dX`;
  - whether MoE route overhead dominates expert GEMM time;
  - whether `W` and `W.T` host-layout overhead is acceptable;
  - the top three performance blockers before M5.

Acceptance:

- All three workload reports are generated on the H200 target machine with identical hardware metadata.
- Reports include zero hidden fallback in `asym_only`; any staged/Torch fallback causes M4.5 to fail unless explicitly run as a fallback comparison mode.
- Latency and memory component percentages sum to the total within `+/- 5%`, or the report clearly identifies the unattributed remainder.
- At least one table provides total and percent breakdown for every stage listed above.
- At least one table provides total and percent breakdown for CPU memory and GPU memory.
- Results are stable enough for leadership review: median step time coefficient of variation is recorded and any high variance is explained.
- No M5 integration work starts until M4.5 identifies whether the next bottleneck is kernel throughput, host bandwidth, route packing/scatter, `W.T` layout storage, optimizer/activation memory, or framework integration overhead.

### M5 - LLaMA-Factory Integration

Deliverables:

- `src/llamafactory/model/model_utils/asymgemm.py`
- Patch only PEFT LoRA wrapper `base_layer` objects after `init_adapter(...)`.
- Args:
  - `use_asymgemm_sft: bool`
  - `asymgemm_dtype: bf16` initially; parse but reject `fp8`
  - `asymgemm_targets: dense|moe|all`
  - `asymgemm_fallback: asym_only|asym_or_staged|asym_or_torch`
  - `asymgemm_pin_transpose: bool`
  - `asymgemm_profile: bool`
- Reject initially:
  - `finetuning_type != lora`
  - non-SFT stages
  - `use_kt`
  - BNB/HQQ/EETQ and other quantization on patched modules
  - DeepSpeed/FSDP/ZeRO
  - Unsloth and v1-kernel patch paths
  - DoRA/OFT/PiSSA conversion
  - merge/unload
- CPU-first load plan documented:
  - tiny validation may tolerate transient GPU residency;
  - KT-beating claim requires avoiding full base-weight residency in HBM.
- Examples:
  - `examples/asymgemm/train_lora/tiny_dense_lora_sft.yaml`
  - `examples/asymgemm/train_lora/tiny_moe_lora_sft.yaml`

Acceptance:

- Tiny dense SFT runs end to end.
- Tiny MoE SFT runs after M4.
- AdamW and LoRA+ optimizer paths keep base handles out of optimizer state.
- Adapter `save_pretrained` works.
- Adapter reload works into both unpatched HF/PEFT and Asym-patched models.
- Checkpoint resume works.
- Logs include patched layer count, fallback count, pinned bytes, peak HBM, hardware info, and whether direct `dX` was used.

### M6 - KTransformers Reproduction And Benchmark Harness

Deliverables:

- `benchmarks/asym_sft/` scripts for:
  - grouped GEMM microbench;
  - frozen-linear forward+dX;
  - MLP step;
  - tiny dense LLM step;
  - tiny MoE step;
  - LLaMA-Factory SFT smoke.
- Preregistered benchmark matrix:
  - model;
  - active experts;
  - top-k;
  - seq length;
  - microbatch;
  - gradient accumulation;
  - LoRA rank;
  - target modules;
  - precision;
  - backend;
  - H200 host-path details.
- Baselines:
  - GPU-resident LoRA;
  - optimized async H2D staging with stream overlap and buffer reuse;
  - KTransformers AMXBF16 where CPU supports it;
  - KTransformers AMXINT8/INT4 as quality-speed baselines;
  - AsymGEMM direct fetch.
- Metrics:
  - optimizer-step wall time;
  - forward, backward, optimizer, route pack/scatter;
  - dataloader-excluded tokens/sec;
  - time-to-first-step and JIT warmup;
  - p50/p95 step time;
  - fallback counts;
  - GPU utilization;
  - CPU utilization;
  - PCIe/host-path bandwidth when available;
  - CPU DRAM bandwidth;
  - peak HBM;
  - CPU RSS/USS;
  - pinned bytes;
  - page faults and swap.

Acceptance:

- KT baseline is run through the current public LLaMA-Factory `use_kt` path on the same machine.
- Exact versions and commits are recorded for KT, kt-kernel, accelerate-kt, LLaMA-Factory, PyTorch, CUDA, and AsymGEMM.
- CPU controls are recorded: socket binding, AMX availability, thread count, DRAM channels, power/clocks, hugepage or `mlock` limits.
- Tokens/sec definition is explicit and matches the KT run accounting.
- BF16 AsymGEMM is compared to KT AMXBF16 first.
- KT AMXBF16 comparison is run on this Intel-AMX + H200 machine.
- INT8/INT4 comparisons include adapter quality/eval parity.

### M7 - Optimize And Beat-KT Gate

Deliverables:

- Bottleneck reports with before/after traces.
- Direct-fetch vs staged break-even study across at least:
  - balanced routing;
  - skewed routing;
  - real router distribution from a supported MoE checkpoint.
- Optimizations as needed:
  - direct `dX` improvements;
  - host-weight handle cache;
  - optional `W_T` cache with CPU memory accounting;
  - GPU-only route histogram/offset build;
  - buffer reuse;
  - JIT cache warmup.

Acceptance for "beats KTransformers":

- Same hardware, model, dataset, LoRA rank, target modules, precision tier, batch/GAS, seed, and checkpoint format.
- BF16 primary claim compares AsymGEMM BF16 to KT AMXBF16.
- Warmup and JIT are reported separately.
- At least 5 measured runs, p50/p95, and coefficient of variation under `5%`.
- MoE forward+backward is at least `1.25x` faster than KT on the target shape.
- End-to-end SFT tokens/sec is at least `1.15x` faster than KT.
- Peak HBM is no higher than KT or the tradeoff is explicitly called out.
- No swap and no unbounded CPU memory growth.
- Saved adapter reloads and eval/perplexity or task metrics match the KT workflow.

If direct fetch does not beat optimized staging on any named MoE shape, pivot the paper to a boundary analysis instead of a speedup claim.

## Backlog

Only start these after M0-M7 are green or formally narrowed.

- FP8 frozen-base LoRA:
  - fake-quant reference;
  - exact scale-layout tests;
  - forward and direct `dX` parity;
  - quality study before SFT claim.
- Full fine-tuning:
  - routed `wgrad`;
  - optimizer state for offloaded weights;
  - update semantics and checkpointing.
- Multi-GPU:
  - rank-local routing and expert placement;
  - distributed checkpoint/resume;
  - cross-rank CPU memory accounting.
- Larger models:
  - DeepSeek-V2-Lite or Qwen3-30B-A3B first;
  - no DeepSeek-V3-class claim until multi-GPU and small-model ladders pass.

## Decision Gates

| Gate | Decision |
| --- | --- |
| After M1 | Stop direct-fetch work if host-pointer legality or bandwidth is not viable on target hardware. |
| After M2 | Stop training integration if `dX` or LoRA grads cannot match Torch. |
| After M4 | Narrow to dense-only if MoE route pack/scatter or grouped `dX` cannot pass correctness. |
| After M5 | Stop LLaMA-Factory path if PEFT save/reload, optimizer filtering, or checkpoint resume breaks. |
| After M6 | Do not claim a KT comparison unless KT is reproduced on the same machine. |
| After M7 | Do not claim a speedup unless direct fetch beats KT and optimized staging under the registered benchmark matrix. |

Each decision must be recorded in `agent/decisions.md` with evidence artifact, date, command, hardware, and owner.

## Minimum Viable Demo

The first public demo is the MLP demo, not a full LLM:

```bash
python examples/asymgemm/mlp_lora_demo.py --backend asym_or_staged --report reports/mlp_demo.json
```

It must show:

- frozen base weights held in CPU host memory;
- GPU-compute base forward and `dX`, or explicit fallback counts;
- LoRA-only optimizer update;
- Torch parity for output, loss, `dX`, and LoRA gradients;
- lower HBM than GPU-resident base weights for the same demo;
- pinned bytes, peak HBM, and timing in a JSON report.

The first integration demo is later:

```bash
llamafactory-cli train examples/asymgemm/train_lora/tiny_dense_lora_sft.yaml
llamafactory-cli train examples/asymgemm/train_lora/tiny_moe_lora_sft.yaml
```

Only after these pass should real-model KT comparisons begin.
