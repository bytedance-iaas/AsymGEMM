export const meta = {
  name: 'audit-merged-tree-v2',
  description: 'Strong-model re-audit: re-verify every recorded claim, review the applied fixes, fresh per-model discovery until convergence',
  whenToUse: 'Pre-push audit of the merged AsymGEMM tree with a stronger model, superseding the earlier pass',
  phases: [
    { title: 'Re-verify' },
    { title: 'Fix-review' },
    { title: 'Find' },
    { title: 'Verify' },
    { title: 'Critic' },
  ],
}

const REPO = '/home/kevinni/AsymGEMM-SFT-39/third_party/AsymGEMM'

const HARD_RULES = `
ABSOLUTE CONSTRAINTS (violating these ruins the user's machine state):
- DO NOT RUN ANY PROJECT CODE. No python, no pytest, no probes, no training, no \`python -c\`, no imports.
  A GPU is shared with another user and experiments must be strictly sequential; the user said
  "dont run code". STATIC READING ONLY: git show / git diff / git log / grep / rg / read files.
- DO NOT MODIFY ANY FILE anywhere. Read-only audit.
- DO NOT touch /home/kevinni/AsymGEMM-SFT (reference-only clone).
- Work in ${REPO} (branch main_kevin, HEAD ~9897a5c).
IMPORTANT: an earlier audit pass ran on a WEAKER model. Do not trust its conclusions or the doc
(agent/impls/fix_merged.md) as ground truth — re-derive everything from the code itself.
`

const CONTEXT = `
CONTEXT. Two clones of one repo were merged: the "SFT side" (c62aef3, memory/latency work, ~1000+
lines) and the "qwen3.5 side". Common ancestor 1896825. Merge commit fd2eb57; then 4c2b4f2,
389db00, 9897a5c added doc records + small telemetry fixes. Useful diffs:
  git diff 1896825 c62aef3 -- asym_gemm/ scripts/      (the SFT side's incoming work)
  git diff c62aef3 main_kevin -- asym_gemm/ scripts/   (qwen3.5 side + post-merge fixes on top)

DEFAULT-ON behaviours that changed recently (deserve the most suspicion):
- ASYMM_QWEN3_MOE_DOWN_DX_STAGED default ON (stages packed down weight to HBM per layer-backward)
- ASYMM_FG_ELEMENTWISE_CHUNK_MB default 1024 (drives fg_chunk_rows -> row-chunked fg paths)
- ASYMM_QWEN3_MOE_FG_RELEASE_FUSED_HOME default ON (frees fused gate_up host weight once split
  bases exist; running the plain path afterwards in the same process raises "grouped weight must
  be 3D, got shape (0,)" — loud, not silent)
- ASYM_SAVED_TENSOR_ASYNC_UNPACK (side-stream H2D restage on unpack)

MODELS and their code paths — reason about EACH where they differ:
- q3.5-35b-a3b, q3.5-122b-a10b : MoE (qwen3_moe_finegrained) + LINEAR ATTENTION (delta-net/fla) + FA4 venv
- q3-30b-a3b                   : MoE (qwen3_moe_finegrained), ker101 routed default
- q3.5-27b dense               : DENSE (dense_mlp_finegrained) — the only gate covering that file
- q3-32b, q2.5-32b, q2.5-72b, llama3.3-70b : DENSE (dense_mlp_finegrained), ker000
- llama4-scout                 : MoE via llama4_experts.py
Shapes matter: fg_chunk_rows(total_rows, row_width) decides whether chunked paths engage, and
row_width = intermediate_dim differs per model (MoE I=512/768 vs dense I in the thousands), so the
same code takes different branches per model and per sequence length. Flagship qwen3.5 rows run
ker101 + loraafwdcpu (LoRA-A forward on CPU), seq 45k/80k(/120k pre-merge) x b8, top_k=8.
`

const claimSchema = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['CONFIRMED', 'CORRECTED', 'REFUTED'] },
    corrected_statement: { type: 'string', description: 'if CORRECTED/REFUTED: what is actually true' },
    evidence: { type: 'string', description: 'file:line and the code you read' },
    new_concerns: { type: 'string', description: 'anything adjacent you noticed that the claim misses (empty if none)' },
  },
  required: ['verdict', 'evidence'],
}

const CLAIMS = [
  { id: 'F1-bug', text: `offload() in activation_offload.py records num_offloads/offloaded_bytes/offload_bytes_by_tag but empty_cpu() records only num_cpu_allocs, so before the fix, paths that fill an empty_cpu handle by direct row writes + record_cpu_ready moved bytes to CPU invisibly (e.g. moe.act via the chunked act path).` },
  { id: 'F1-fix', text: `The fix folded that accounting INTO record_cpu_ready. Claim: it cannot double-count — offload() does NOT call record_cpu_ready, and every direct-write caller calls record_cpu_ready EXACTLY once per handle. Check ALL callers: qwen3_moe_finegrained.py (~778-780, ~872, ~1463-1464) and dense_mlp_finegrained.py (~316, ~455-456). Also check: are there paths where record_cpu_ready is called on a handle that was filled via offload() (double-count), or direct-write paths that DON'T call it (still uncounted)?` },
  { id: 'F2', text: `qwen3_moe_finegrained_fused_home_released is set dynamically at qwen3_moe.py:~2568 via getattr(...,0)+1 but was not declared on the stats dataclass in frozen_linear.py, so it never reached profile.json; it is now declared. Verify the declaration exists and that the profile serialization actually picks up dataclass fields (how does the stats dict get exported?).` },
  { id: 'F3', text: `The async H2D restage in activation_offload.py stage() gives ZERO copy/compute overlap by construction: side.wait_stream(compute) makes the copy wait on all prior compute; compute.wait_event(done) makes all later compute wait on the copy; nothing is enqueued between. Only benefit is host run-ahead. The comment was corrected to say this. Verify the corrected comment is itself accurate, including its prescription for achieving real overlap.` },
  { id: 'F4', text: `The driver (profile_lora_lf_test_both.sh) now forwards ASYM_SAVED_TENSOR_ASYNC_UNPACK as ASYM_GEMM_LF_CONFIG_ASYM_SAVED_TENSOR_ASYNC_UNPACK="\${ASYM_SAVED_TENSOR_ASYNC_UNPACK:-}". Verify: (a) the line sits in the correct env-forwarding block so it actually reaches the training process; (b) forwarding an EMPTY string when the var is unset does not change behavior — check how the reader (decoder_activation_offload.py:~22 _async_unpack_enabled or similar) treats empty vs unset; (c) _env_config() harvesting means it now lands in profile artifacts.` },
  { id: 'F5', text: `linear_attention_activation_offload.py: a failed pin in _empty_strided_cpu_like silently returns a pageable buffer; _pack then leaves ready_event=None and _unpack takes the host-blocking branch. A module-global _PIN_FALLBACK_CALLS counter was added and exposed in snapshot() as pin_fallback_calls. Verify the mechanics claim, AND critique the fix: the counter is module-global while snapshot() is per-wrapper (same number reported by every wrapper — misleading?), incremented without synchronization (autograd threads?), and never reset between runs in one process. Is the fix correct enough or does it need adjustment?` },
  { id: 'B3', text: `gc_async_offload.py (93 lines) has ZERO external callers — nothing imports or calls it; git log -S "async_save_on_cpu" --all returns only the commit adding it. Its docstring premise (stock LF saves the GC root to an UNPINNED cpu tensor) does not describe this repo, whose gc_boundary_offload.py path already offloads the root through the pinned pool. Verify both statements.` },
  { id: 'V1', text: `wait_cpu_ready in activation_offload.py performs only current_stream().wait_event() (device-side ordering; explicitly commented never block the host), while cpu_left.py:~188 does padded[...].copy_(x_cpu[...]) — a HOST-side memcpy out of that pinned buffer — so nothing orders the host read after the D2H that fills the buffer. Commit 27dde72 (pre-merge) moved a previously-accidental host sync (blocking .to("cpu") + .item() at cpu_left.py:~129,139) under "if cached is None", so the 2nd/3rd CPU-left calls per forward skip it. Verify precisely; also hunt for ANY other synchronization that might cover the host read (e.g. a .synchronize(), an event.synchronize(), a blocking copy upstream on the same buffer) before endorsing the claim.` },
  { id: 'V2', text: `release_cpu in activation_offload.py returns buffers to the module-global _CPU_BUFFER_POOL after only device-side ordering, so a pinned buffer whose D2H copy is still in flight can be recycled by the next _alloc_cpu and handed to an unrelated consumer that overwrites or reads it. Verify by reading release_cpu/_return_cpu and every producer of in-flight copies (offload(), the chunked direct-write paths); state exactly the interleaving that corrupts data, or refute it if some wait actually intervenes.` },
  { id: 'V3', text: `scripts/testing/qwen35_fg_numeric_probe.py at --qwen3 --tokens 655360: fg101 run ALONE scores rel_fro 0.006273 (PASS) but 0.170918 (FAIL) when run after plain+fg000 in the canonical 5-case order — empirically established, so cross-case contamination in the probe, not a kernel bug. Root cause UNKNOWN. From code reading alone, rank the candidate mechanisms: module-global _CPU_BUFFER_POOL reuse; _asym_kernel_meta_memo on the offsets tensor keyed only by str(device) (qwen3_moe_routed_gemm.py:~89,108); _asym_pad_memo (frozen_linear.py:~643-746); os.environ mutation between cases vs module-level caches; engine state mutated by the plain path. Identify which mechanism can actually produce a DETERMINISTIC wrong forward for fg101 given the probe's exact call sequence, and whether the SAME offsets tensor object is reused across cases with route env flags changed.` },
  { id: 'D1', text: `In the flagship qwen3.5 row (ker101 + loraafwdcpu), the SFT blocked forward (fwd_blocks / gateup_act_blocked) NEVER engages because its gate requires da_gpu AND lora_a_fwd_gpu, and lora_a_fwd is CPU there. What runs instead is the act_chunk branch (gated on NOT lora_a_fwd_gpu) plus silu-backward chunking. Verify the gate conditions by reading qwen3_moe_finegrained.py (~686-710 and the activation section), and confirm which branches produce the stage_rows calls observed (2880 at 45k, 4800 at 80k, = 40 layers x steps x chunks?). Sanity-check that arithmetic too.` },
  { id: 'DEAD-int32', text: `Dead-end record: int32 overflow at T=655360 (R*I=4.03e9) is refuted because the routed kernel never forms a 32-bit linear index — act is addressed by TMA row coordinate with 64-bit descriptor strides and the scatter uses atomicAdd(&out[static_cast<uint64_t>(token_row)*stride+col]) (include/asym_gemm/impls/sm100_bf16_asym_gemm.cuh:~1043-1045); token_indices is int64. Re-verify by reading the kernel, INCLUDING any other index arithmetic in the same file that the earlier pass may have skipped (e.g. int32 casts on R, offsets arrays, cumulative sums).` },
  { id: 'DEAD-race', text: `Dead-end record: "fg101 is broken / host-device race" is refuted because (a) rel_fro reproduces to six decimals across three commits and processes, and (b) fg000 and fg101 share the identical code path up to the divergence at qwen3_moe_finegrained.py:~988-1008 (down_forward_scatter_add_ vs _base_forward+_scatter_routes_add_), so an upstream defect would break fg000 too — and fg000 passes. Re-verify the shared-path claim by reading the code at the stated lines: is the divergence REALLY only there under the probe's env (no fg_env flags, route 1,0,1)?` },
  { id: 'CLEAN-rebase', text: `Verified-clean record: _fg_elementwise_blocks/_expert_blocks per-block offsets are correctly rebased so each block's grouped GEMM segments are identical to the full-width call (bit-identical results claim in its docstring). Re-verify the offset rebasing arithmetic line by line, including the pair-offsets form and edge cases (empty experts, single group, uneven blocks).` },
  { id: 'CLEAN-hbmkeep', text: `Verified-clean record: _HBMKeepManager (used when ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1) deliberately lacks record_cpu_ready/stage_rows/empty_cpu, and every call site is hasattr-guarded or unreachable when keep_acts_hbm is on. Re-verify EVERY method call on the manager object in qwen3_moe_finegrained.py under keep_acts_hbm=1, including in the backward and in error paths — one unguarded call = AttributeError mid-training.` },
  { id: 'CLEAN-stagecache', text: `Verified-clean record: the _stage_cache in ActivationOffloadManager (keyed by device/dtype/shape/tag) cannot alias two LIVE stages, and _release_chunk_stages plus the data_ptr-keyed chunk_stages dict are sound. Re-verify: same tag + same shape requested twice while the first stage is still live — what happens? Check stage(), stage_rows(), release_stage(drop_cache=...) and _stage_keys_by_ptr bookkeeping for leaks or aliasing.` },
  { id: 'CLEAN-dtype', text: `Verified-clean record: the blocked forward allocating gate_cpu/up_cpu/act_cpu as bf16 while the non-blocked path offloads gate/up in input_dtype is only latent (flagship is bf16). Re-verify: is there ANY reachable config (fp32 training? fp16?) where input_dtype != bf16 AND the blocked path engages, and does the backward read those handles assuming input_dtype?` },
  { id: 'CLEAN-releasepair', text: `RELEASE_FUSED_HOME (qwen3_moe.py ~2553-2570) + the unsupported_reasons pin-check relaxation (qwen3_moe_finegrained.py ~2257-2264 checking _asym_released_fused_home) form a matched pair; after release, the fg path keeps working (160 forwards at 80k) and the plain path fails LOUDLY (NotImplementedError via unsupported_reasons, or ValueError from _grouped_weight_features). Re-verify there is NO silent path: checkpoint save/load, state_dict, weight_offload.py, stp_wrap.py, nograd forward, or any consumer that reads gate_up_base.host_weight.weight after release and would get the empty tensor WITHOUT raising.` },
]

const findingSchema = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          file: { type: 'string' },
          line: { type: 'integer' },
          summary: { type: 'string' },
          trigger: { type: 'string', description: 'exact model + config + shape that triggers it' },
          models_affected: { type: 'string' },
          severity: { type: 'string', enum: ['correctness-bug', 'latent-trap', 'stats-only', 'doc-wrong', 'style'] },
          confidence: { type: 'string', enum: ['CONFIRMED', 'UNCERTAIN'] },
          evidence: { type: 'string' },
          fixable_by_reasoning: { type: 'boolean' },
        },
        required: ['id', 'file', 'summary', 'trigger', 'severity', 'confidence', 'evidence', 'fixable_by_reasoning'],
      },
    },
  },
  required: ['findings'],
}

const verdictSchema = {
  type: 'object',
  properties: {
    real: { type: 'boolean' },
    reason: { type: 'string' },
    correction: { type: 'string' },
  },
  required: ['real', 'reason'],
}

const KNOWN = `
ALREADY RECORDED (agent/impls/fix_merged.md) — do NOT re-report as new findings:
F1 empty_cpu/record_cpu_ready byte-accounting hole (fixed); F2 fused_home_released undeclared
(fixed); F3 async-restage overlap comment wrong (fixed); F4 ASYM_SAVED_TENSOR_ASYNC_UNPACK not
forwarded (fixed); F5 silent pageable pin-fallback, pin_fallback_calls added (fixed);
B1 cross-model matrix not run (dense q3.5-27b + q3-30b gates missing); B2 G1 ratios use a
pre-merge B; B3 gc_async_offload.py dead but recorded as measured; V1 wait_cpu_ready device-only
vs cpu_left host memcpy; V2 pool recycling with in-flight D2H; V3 probe self-contamination (root
cause unknown); D1 blocked fwd never engages in flagship row (loraafwdcpu); D2 nograd full-R act
not blocked (~1.4 GiB); D3 no fg fwd blocking under dscatter; D4 nothing >80k validated;
D5 _record_attn_hbm_gemm can't distinguish chunk on/off; D6 ATTN_ACT_LORA_CHUNK no-ops when
FG_ELEMENTWISE_CHUNK_MB=0; D7 no tests reference the new knobs; D8 smoke grad_norm slightly wide.
REFUTED (do not resurrect without NEW evidence): int32 overflow in the routed kernel; pool
bucket/narrow aliasing at R=5,242,880 exactly; "ker101 broken / fg101 host-device race".
`

const AREAS = [
  { key: 'moe-fg-fwd', prompt: `asym_gemm/training/qwen3_moe_finegrained.py FORWARD + nograd forward, all branches: fwd_blocks (blocked gate/up/act), act_chunk, keep_acts_hbm/_HBMKeepManager, da_gpu on/off, lora_a_fwd_gpu vs CPU-left. For q3-30b-a3b (I=768), q3.5-35b-a3b (I=512), q3.5-122b-a10b: compute fg_chunk_rows at 45k/80k/120k x b8 top_k=8 and state which branch runs. Hunt for numerics differences BETWEEN branches (a config change that silently changes results), dead branches, and wrong gate logic.` },
  { key: 'moe-fg-bwd', prompt: `asym_gemm/training/qwen3_moe_finegrained.py BACKWARD: silu-bwd chunking, DOWN_DX_STAGED (default ON — verify the staged HBM weight is freed on every path incl. exceptions, and that its dx math equals the asym-dx path it replaced), dscatter blocked loops, keep_dgrads_hbm, dX/dA/dB grads. Verify blocked/chunked and full-width branches compute identical math, and ctx save/restore consistency (what forward saves vs what backward expects, per branch combination — e.g. forward took act_chunk but backward assumes offload()-style handles?).` },
  { key: 'dense-fg', prompt: `asym_gemm/training/dense_mlp_finegrained.py end to end — the ONLY file behind q3.5-27b dense, q3-32b, q2.5-32b, q2.5-72b, llama3.3-70b. It calls record_cpu_ready (~316,455,456) and empty_cpu (~186,201,202,300,433,434); the merge changed record_cpu_ready to also record bytes. Verify once-per-handle discipline (double-fire = double-count NOW). Compute whether chunking engages at dense widths/real seq lens. Check its silu-bwd and act paths mirror the qwen3 ones or have drifted (copy-paste divergence bugs).` },
  { key: 'llama4', prompt: `asym_gemm/training/llama4_experts.py + qwen3_moe.py's own empty_cpu callers (~909,926,927,997). If any path fills an empty_cpu handle by direct writes WITHOUT record_cpu_ready: its bytes stay uncounted AND no cpu-ready event is registered — find who later waits on that handle and whether a stale/unordered read is possible. llama4-scout is the only model on llama4_experts.py.` },
  { key: 'activation-offload-core', prompt: `asym_gemm/training/activation_offload.py in FULL: _CPU_BUFFER_POOL (_alloc_cpu bucketing/_narrow/_asym_pool_base/_return_cpu/trim), stage/stage_rows/_stage_cache/release_stage/_mark_stage_live, offload/empty_cpu/adopt_cpu/record_cpu_ready/wait_cpu_ready/release_cpu/seal, _GOV hooks, stats dataclass, fg_chunk_rows/fg_elementwise_chunk_bytes. Hunt ordering hazards (host vs device), pool poisoning, stage aliasing, accounting drift, and _asym_pool_base narrow-vs-base confusion (e.g. copy into a narrow whose base is recycled).` },
  { key: 'attn-offload', prompt: `asym_gemm/training/attention_activation_offload.py + decoder_activation_offload.py: _add_matmul_rows_/_attn_lora_chunk_enabled, _h2d_restage_stream discipline (events, record_stream, stream cache keyed by device), saved-tensor pack/unpack, skip_in_backward via torch._C._current_graph_task_id() — could that gate skip a NEEDED offload in configs where a grad-enabled forward legitimately runs inside a backward graph task but is NOT an unsloth-GC recompute (e.g. other checkpointing, double-backward)? All models use attention.` },
  { key: 'linattn', prompt: `asym_gemm/training/linear_attention_activation_offload.py (q3.5 models only): async unpack side-stream path, _empty_strided_cpu_like pageable fallback + ready_event=None consequences, _pack/_unpack symmetry (strided tensors, non-contiguous), imports from attention_activation_offload/decoder_activation_offload (cycles? import-order at wrap time?), interaction with the fla/delta-net backward and QWEN35_DELTA_CHUNK (seq-chunked calls -> more, smaller saved tensors -> min_bytes threshold behavior changes?).` },
  { key: 'frozen-linear-cpuleft', prompt: `asym_gemm/training/frozen_linear.py (GEMM dispatch, _grouped_weight_features, _asym_pad_memo ~643-746, stats dataclass) + cpu_left.py in full + qwen3_moe_routed_gemm.py memos (~89-108, keyed only by str(device)). Determine rigorously whether one offsets tensor object can be reused with different experts/env-flags across calls (stale memo). This is the live V3-contamination suspect — proving or disproving a concrete mechanism is the single highest-value outcome of this audit.` },
  { key: 'gc-boundary', prompt: `asym_gemm/training/gc_boundary_offload.py + offload.py merge diff + how unsloth-GC recompute interacts with saved-tensor wrappers. The boundary root path (pinned pool offload at ~69-70,90): correctness of its event/wait discipline, and whether skip_in_backward (new) changes what unsloth-GC saves/restores. gc_async_offload.py is KNOWN dead — skip it.` },
  { key: 'driver-knobs', prompt: `scripts/lf/profile_lora_lf_test_both.sh + profile_lora_lf_test_source.sh + run_lf_lora_sft.sh + rebuild_asymgemm.sh + run_dial_ladder.sh. Enumerate EVERY ASYMM_*/ASYM_* env var read in asym_gemm/**.py, and check each is forwarded as ASYM_GEMM_LF_CONFIG_* by the driver — list every unforwarded knob (provenance holes). Check empty-string-vs-unset semantics for each forwarded knob against its reader (several readers treat empty as "default ON" — e.g. DOWN_DX_STAGED — so forwarding "" is meaningful there; any knob where forwarding "" CHANGES behavior vs not forwarding is a bug). Also the qwen3.5 FA4 auto-switch + QWEN35_DELTA_CHUNK_DEFAULT plumbing.` },
  { key: 'defaults-matrix', prompt: `Cross-interactions of the DEFAULT-ON set, per model: RELEASE_FUSED_HOME x (checkpoint save, state_dict, resume, weight_offload.py, stp_wrap.py, nograd, eval); DOWN_DX_STAGED x (dscatter, keep_acts_hbm, route flags, HBM headroom at 122B); FG_ELEMENTWISE_CHUNK_MB x (dense widths, tiny R, keep_dgrads_hbm). Any default-on path assuming bf16 or specific E/I. Any combination that raises, silently disables another feature, or double-frees.` },
  { key: 'integration-lf', prompt: `asym_gemm/integrations/lf.py + liger_loss.py + training/stp_wrap.py + weight_offload.py + training/__init__.py: verify the model->wrapper mapping (q3.5-35b/122b MoE+linattn; q3.5-27b dense; q3-30b MoE; q3-32b/q2.5-32b/q2.5-72b/llama3.3-70b dense; llama4-scout llama4_experts) and that no enable-condition changed meaning in the merge. Check _env_config() harvesting (what reaches profile.json), and the snapshot/stats aggregation for keys that collide or overwrite across wrappers.` },
  { key: 'qwen35-additions', prompt: `The qwen3.5 side's own +133 lines (git diff c62aef3 main_kevin): lf_trace.py (+26), qwen3_moe.py RELEASE_FUSED_HOME implementation (+21) — including what happens on model checkpoint save AFTER release (state_dict reads host_weight.weight -> empty tensor saved silently?), attention/linear_attention skip_in_backward additions (+27/+28) — default AUTO keyed on UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU env: is that env actually set by the LF stack in the flagship config, and is the auto-default correct when GC is配置ured differently? profile_lora_lf_test_both.sh QWEN35_DELTA_CHUNK plumbing (+12).` },
  { key: 'tests-probes-kernels', prompt: `tests/training/test_lf_qwen35_asym_backend.py + test_lf_qwen3_asym_backend.py: do the 171 tests exercise ANY of the merge's new paths (chunked act, silu-bwd chunking, down_dx_staged, blocked fwd, async unpack, skip_in_backward), or do they all run at shapes below every chunk threshold (silent-cap problem: green tests that touch none of the new code)? Also: check whether csrc/ or include/ changed anywhere in 1896825..main_kevin (git diff --stat that range for csrc/ include/) and read any kernel change nobody audited. Also scripts/gemm/bench_engine_tax.py (new, SFT side).` },
]

const LENSES = [
  'correctness — is the claimed behavior actually wrong given ALL surrounding code (some other guard may cover it)?',
  'reachability — can a REAL model+config in this repo reach the line with the claimed effect? name it, or kill the finding',
  'novelty+materiality — is it already in the KNOWN list, or too trivial to change what the user does before pushing?',
]

// ---- Phase 1: re-verify every recorded claim ----
phase('Re-verify')
log(`Re-deriving ${CLAIMS.length} recorded claims from code (strong-model pass)`)
const reverified = await parallel(CLAIMS.map(c => () =>
  agent(
    `${HARD_RULES}\n${CONTEXT}\n\nYou are RE-VERIFYING one claim that an earlier, weaker audit pass recorded. Re-derive it from the code — do not trust the doc. If the claim is exactly right, CONFIRMED. If directionally right but wrong in a detail that matters, CORRECTED + the corrected statement. If wrong, REFUTED + why.\n\nCLAIM [${c.id}]: ${c.text}\n\nBe rigorous: read every file:line you cite. Also fill new_concerns with anything ADJACENT the claim misses (one or two sentences; empty if none).`,
    { label: `reverify:${c.id}`, phase: 'Re-verify', schema: claimSchema }
  ).then(r => ({ id: c.id, ...r }))
))

// ---- Phase 2: review the applied fixes as diffs ----
phase('Fix-review')
const fixReview = await agent(
  `${HARD_RULES}\n${CONTEXT}\n\nReview the POST-MERGE fix commits as diffs, as a skeptical senior reviewer:\n  git show 389db00 -- asym_gemm/   (declares the fused_home_released stats field)\n  git show 9897a5c -- asym_gemm/ scripts/   (async-restage comment rewrite; pin_fallback_calls counter; ASYM_SAVED_TENSOR_ASYNC_UNPACK forwarding)\n  git diff 1896825 fd2eb57 -- asym_gemm/training/activation_offload.py   (the record_cpu_ready accounting fold + qwen3.5-side changes)\nFor each hunk: is it correct, complete, and inert on the compute path as claimed? Specifically probe: record_cpu_ready double-count/missed-call risks across ALL callers; whether the new bash forwarding line is inside the correct scope/heredoc of the driver script (read the surrounding 30 lines) and preserves unset-vs-empty semantics for its reader; whether _PIN_FALLBACK_CALLS being module-global + per-wrapper snapshot is acceptable or misleading; whether the rewritten comment's technical prescription is accurate. Return a concise list of defects found in the fixes themselves (empty list statement if none), each with file:line and severity.`,
  { label: 'review-applied-fixes', phase: 'Fix-review' }
)

// ---- Phase 3: discovery rounds until dry ----
const seen = new Set()
const confirmed = []
let dry = 0
let round = 0

while (dry < 2 && round < 4) {
  round++
  log(`=== Discovery round ${round} (dry ${dry}/2, confirmed ${confirmed.length}) ===`)
  const seenList = seen.size ? [...seen].join('\n- ') : '(none yet)'

  const found = (await parallel(AREAS.map(a => () => agent(
    `${HARD_RULES}\n${CONTEXT}\n${KNOWN}\n\nAlready found THIS run (do not re-report):\n- ${seenList}\n\nYOUR AREA (round ${round}): ${a.prompt}\n\nRead the actual code; reason per model where models differ. Report ONLY nontrivial findings not already known/refuted: real defects, latent traps reachable by a real config, wrong defaults, docs contradicting code, or things that MUST be tested before pushing. No style nits. An EMPTY findings list is a correct and expected outcome — do not invent findings. fixable_by_reasoning=true only if a safe fix is obvious from reading alone.`,
    { label: `find:${a.key}:r${round}`, phase: 'Find', schema: findingSchema }
  )))).filter(Boolean).flatMap(r => r.findings || [])

  const fresh = found.filter(f => {
    const k = `${f.file}:${f.line || 0}:${f.id}`
    if (seen.has(k)) return false
    seen.add(k)
    return true
  })
  log(`round ${round}: ${found.length} raw, ${fresh.length} fresh`)
  if (!fresh.length) { dry++; continue }
  dry = 0

  const verified = await parallel(fresh.map(f => () =>
    parallel(LENSES.map(lens => () => agent(
      `${HARD_RULES}\n${CONTEXT}\n${KNOWN}\n\nADVERSARIALLY REFUTE this claimed finding — kill it if wrong, already-known, or unreachable by any real model+config. Default real=false when uncertain.\n\nCLAIM: ${f.summary}\nFILE: ${f.file}:${f.line || '?'}\nTRIGGER: ${f.trigger}\nMODELS: ${f.models_affected || '?'}\nEVIDENCE: ${f.evidence}\n\nLENS: ${lens}\n\nRead the code at the location and rule. If directionally right but overstated, real=true with the corrected statement in \`correction\`.`,
      { label: `verify:${f.id}`, phase: 'Verify', schema: verdictSchema }
    ))).then(vs => {
      const votes = vs.filter(Boolean)
      return { ...f, yes: votes.filter(v => v.real).length, votes: votes.length, corrections: votes.map(v => v.correction).filter(Boolean) }
    })
  ))
  const kept = verified.filter(Boolean).filter(v => v.yes >= 2)
  confirmed.push(...kept)
  log(`round ${round}: ${kept.length}/${fresh.length} survived adversarial verification`)
}

// ---- Phase 4: completeness critic ----
phase('Critic')
const critic = await agent(
  `${HARD_RULES}\n${CONTEXT}\n${KNOWN}\n\nA ${round}-round strong-model audit just finished. Areas: ${AREAS.map(a => a.key).join(', ')}. It also re-verified ${CLAIMS.length} recorded claims and reviewed the applied fix diffs.\nNEW confirmed findings this run:\n${confirmed.length ? confirmed.map(c => `- ${c.file}:${c.line || '?'} ${c.summary}`).join('\n') : '(none)'}\n\nYou are the COMPLETENESS CRITIC. Do not re-report findings. Identify concrete GAPS: files changed in 1896825..main_kevin (run the diff --stat yourself, including csrc/, include/, tests/, scripts/) that no area covers; models whose specific path nobody reasoned about; claims in agent/impls/fix_merged.md asserted but never checked against code; assumptions this audit itself made that could hide a bug. If coverage is genuinely complete, say so plainly — do not invent gaps.`,
  { label: 'completeness-critic', phase: 'Critic' }
)

return {
  rounds: round,
  reverified,
  fixReview,
  confirmedCount: confirmed.length,
  confirmed: confirmed.map(c => ({
    id: c.id, file: c.file, line: c.line, summary: c.summary, trigger: c.trigger,
    models: c.models_affected, severity: c.severity, confidence: c.confidence,
    fixable_by_reasoning: c.fixable_by_reasoning, evidence: c.evidence, votes: `${c.yes}/${c.votes}`,
    corrections: c.corrections,
  })),
  gaps: critic,
}
