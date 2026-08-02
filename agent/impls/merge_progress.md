# Merge campaign: 39 ⟵ {46, SFT, then 38}  (progress record)

Goal: make `/home/kevinni/AsymGEMM-SFT-39/third_party/AsymGEMM` (this repo) the union of
- **39** (base): HEAD `33dc443` + uncommitted new-model additions (glm45/glm47/gptoss/hunyuan/mixtral/phimoe MoE wrappers, qchunked_attention, lf.py + liger_loss.py integration hooks, model_integration.md)
- **46** (`/home/kevinni/AsymGEMM-SFT-46/...`): HEAD `d22440d` (= parent of 33dc443) + uncommitted Qwen3.5 TP work
- **SFT** (`/home/kevinni/AsymGEMM-SFT/...`): commits ahead of 33dc443 (motivation-bench, ep_balance_bench, cpu-archive moves) + uncommitted kernel/CPU-path work
- **38** (phase 2, after validation): 3 committed capacity commits on top of 33dc443 (exact_pinned, host_weight, capacity scripts). Branched from *current* 33dc443 in git terms, but user warns the capacity approach may target stale paths — merge selectively.

Then validate near-capacity (no regression in memory/latency/throughput) on: Qwen3-30B-A3B, Qwen3.5-35B, GLM 4.5 MoE, GLM 4.7 MoE — inside container via `asym42_enroot_run` (NEVER on host).

Git facts (verified):
- `d22440d..33dc443` touches only `agent/impls/model_integration.md` → base `33dc443` is a valid 3-way base for every conflicted file.
- All of 46's delta is uncommitted; SFT's delta = commits `288dfc1 + c428a1c..a0ef616` + uncommitted; 38's delta = commits `b37e59d, aacd99c, ef9bc80`, clean tree.
- Per memory: NEVER commit — leave the merged working tree for Kevin.

## Overlap files (need 3-way merge, base = 33dc443)
| file | 39 | SFT | 46 |
|---|---|---|---|
| asym_gemm/integrations/lf.py | +322 (new models) | M (kernel path) | — |
| asym_gemm/training/attention_activation_offload.py | ±16 | M | — |
| asym_gemm/training/frozen_linear.py | ±9 | M | — |
| scripts/lf/profile_lora_lf_test_source.sh | ±32 | ±17 | — |
| scripts/lf/run_lf_lora_sft.sh | ±9 | ±21 | ±12 |
| scripts/lf/run_lf_profiled_train.py | ±38 | ±60 | ±64 |

## Non-overlap intake
- **From SFT (copy working-tree version)** — M: `asym_gemm/__init__.py`, `include/.../sm100_bf16_asym_gemm.cuh`, `include/.../sm100_bf16_cpu_left_asym_gemm.cuh`, `training/{activation_offload,cpu_adam,cpu_left,dense_mlp_finegrained,ep_sep,exp_act_offload_lora,gc_boundary_offload,qwen3_moe,qwen3_moe_finegrained}.py`, `csrc/apis/{exp_act_offload,gemm}.hpp`, `csrc/exp_act_offload/exp_act_offload_kernels.cu`, `csrc/jit_kernels/impls/sm100_bf16_asym_gemm.hpp`, `csrc/python_api.cpp`, `scripts/lf/{asym_scheduler.py,postprocess_lf_profile_artifacts.py,profile_lora_lf_test_both.sh,tier_recipes.sh}`, `scripts/testing/ep_balance_bench.py`, `setup.py`, `tests/training/*.py`, `agent/impls/previous_validation_results.md`
- **From SFT — A**: `training/{cpu_ops,cpu_worker,pinned_ledger,placement_policy,qknorm_recompute,save_dedup}.py`, `csrc/apis/cpu_ops.hpp`, `csrc/cpu_ops/cpu_ops.cpp`, `scripts/motivation_bench/{bench_m2a,bench_m4b,bench_m4b_window,plot_motivation}.py`, `tests/{bench_modules,test_cpu_ops,test_cpu_worker,test_moe_direct_reuse,test_pinned_ledger,test_placement_policy,test_qknorm_recompute,test_restage_prefetch,test_save_dedup}.py`, `agent/impls/cpu_compute.md`, `agent/related_work/*`, `scripts/archive/cpu/*` (campaign archive)
- **From SFT — R100 moves**: root `.mrg_*` scripts/snapshots → `scripts/archive/cpu/` (apply as mv)
- **From SFT — untracked worth taking**: `asym_gemm/include/asym_gemm/impls/asym_lora_dataflow.cuh` (kernel header!), `scripts/motivation_bench/bench_{attn_site,base_stream,m2b,m2c,m3,m3_stack,pair_gateup}.py`, `agent/impls/{aymlora_kernels,ep,system_tier_justification,writing_prompt}.md`, `agent/impls/archive/*.md` (+ corresponding deletes of agent/impls/{fix_cpu_compute,merge_cpu_modules,merge_cpu_modules_compspec,placement}.md)
- **From SFT — skip (scratch/stale)**: `.figtmp/`, `.m6_*.sh` root scratch, screenshots, `agent/AsymLoRA (8).pdf`, `agent/impls/tmp.md`, `agent/prompt.md` (verify before final skip)
- **From 46 — untracked**: `scripts/lf/parse_fill_cell.py`, `scripts/lf/tp_probe_fill.sh`, `agent/impls/fix_qwen3.5_tp.md`, `agent/impls/fix_plot_placeholders.md`; `agent/archive/overleaf/` (size-check first)

## Status log
- [x] Inventory of all four repos (this file)
- [x] Phase 1: SFT kernel/CPU merge into 39 working tree
  - Bulk: 92 paths copied (M/A/R100 from `git diff --name-status 33dc443`), verified byte-identical vs SFT tree; root `.mrg_*` → `scripts/archive/cpu/` moves applied.
  - Untracked taken: `asym_lora_dataflow.cuh`, 7 motivation-bench scripts, `agent/impls/{aymlora_kernels,ep,system_tier_justification,writing_prompt}.md`, `agent/impls/archive/*` (4 docs), recovered-sessions merged (no-clobber).
  - Untracked SKIPPED as scratch: `.figtmp/`, `.m6_{long,short,t1long}.sh` (container queue scratch for /workspace/AsymGEMM-SFT), `agent/prompt.md`, `agent/impls/tmp.md`, screenshots, loose `AsymLoRA (8).pdf`.
  - 3-way merges (base 33dc443): 5/6 files conflict-free; 1 conflict in `lf.py` `_wrap_attention_saved_tensor_offload_modules` — both sides appended self-gated installs; resolved by keeping BOTH (SFT `install_rope_recompute()` then 39 `install_qchunked_attention(model)`).
  - Semantic checks: SFT `shared_lora_a_forward` returns None for non-q/k/v roles → 39's MLA share roles (q_a/kv_a) keep source-sharing and fall back per-projection for compute — correct compose. `frozen_linear` merged holds both `allow_bias` kwarg (39) and `_dense_stream_m_chunks` grid-fill (SFT). No 39 wrapper uses the re-signatured `_single_group_launch_tensors`.
  - Note: SFT flipped default `ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE` 0→1 in profile_lora_lf_test_source.sh (kept — SFT-validated kernel default).
- [x] Phase 2: 46 Qwen3.5 TP merge — CONFIRMED script-level only (no asym_gemm/csrc changes in 46's delta):
  - `run_lf_lora_sft.sh`: EP2 forced-false pair → env-overridable `ASYM_EP2_{GRAD,WEIGHT}_OFFLOAD` (defaults still false).
  - `run_lf_profiled_train.py`: EP2 CPU-flat-buffer grad allreduce path for AsymCPUAdamW grad-offload (chunked H2D→all_reduce→D2H staging).
  - Both merged conflict-free on top of Phase-1 results. Copied: `parse_fill_cell.py`, `tp_probe_fill.sh`, `fix_qwen3.5_tp.md`, `fix_plot_placeholders.md`, `agent/archive/overleaf/` (3.2M).
  - NOT taken: 46 stash `stash@{0} On sm90_h20_kevin: asymgemm h200 e2e local changes` — different hardware line (sm90/H20), predates branch align, stale.
- [~] Phase 3: container build + near-capacity validation
  - [x] `_C` REBUILT in asym_sft_42 from merged csrc (setup.py now compiles cpu_ops.cpp with -fopenmp + SVE-BF16). INCIDENT + FIX: first rebuild used SYSTEM python (torch 2.9.1) — but the drivers run in `.venv` (torch **2.12.0+cu130**), so the smoke HARDFAILED with `missing_sm100_m_grouped_bf16_cpu_left...` (venv couldn't import the mismatched _C). Rebuilt with `.venv/bin/python setup.py build_ext --inplace` — the venv is the ONLY correct interpreter for builds/tests here (recorded rule).
  - [x] compileall clean over asym_gemm/scripts/tests; bash -n clean on merged shell scripts.
  - [x] UNIT GATE under `.venv`: 91/91 passed (tests/test_{cpu_ops,cpu_worker,pinned_ledger,placement_policy,qknorm_recompute,restage_prefetch,save_dedup,moe_direct_reuse}.py + tests/training/*). (An initial 91/91 on system python was superseded — venv is authoritative.)
  - [x] Datasets staged: q3.5-35b s896000·n1024 copied from 46 LF/data (identical bytes ⇒ comparable); q3-30b s120000·n1024 copied from SFT LF/data (CPU-matrix provenance); GLM 128k/192k + q3-30b 1.1M already present; registrations verified in dataset_info.json.
  - [~] e2e smoke: q3-30b-a3b 20k·b1 `|T2|` (tp_probe) — running.
  - Validation queue WRITTEN: `agent/anchors_tmp/mrg_validation_queue.sh` (V1 row7 T2 120k·b8 → V2 Air T3 128k·b3 → V3 Flash T3 192k·b5 → V4 q3.5-35b T2 896k·b1 → V5 row9 T2B 1.1M·b1), tpfig_lib run_cell machinery (solo guard, HOSTFLOOR 1300, OOM/COOM verdicts), references in header. GLM references were recorded ON THIS NODE (c14) — tightest comparisons; q3-30b matrix refs from c06 (band per record protocol), 35B refs from c18 (band).
  - Known default-behavior delta to watch: merged driver defaults `ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE=1` (SFT-validated flip; recorded GLM cells ran =0). If GLM cells breach band, first A/B this flag.
- [ ] Phase 4: selective 38 capacity merge + revalidation — RECON DONE, plan:
  - 38 delta = `b37e59d` (feature: exact_pinned.py + host_weight/cpu_adam hooks + scripts/lf/capacity/ toolkit + model_capacity.md) + `aacd99c` (evidence: profiling_results/capacity_push_c17/) + `ef9bc80` (a stray GITLINK for `agent/archive/overleaf/[MLSys 26 Sub]...` — submodule entry w/o .gitmodules; SKIP: 39 git-ignores overleaf/ and content already on disk from 46).
  - exact_pinned is fully env-gated (ASYM_EXACT_PINNED default OFF ⇒ flags-off bit-identical); mechanism = cudaHostRegister exact-size in place vs pin_memory()'s pow2 bucketing; NOT stale — HostWeight + AsymCPUAdamW `_pin_if_requested` are live paths in the merged tree.
  - TAKE: exact_pinned.py, model_capacity.md, scripts/lf/capacity/*, capacity_push_c17 evidence, host_weight.py hook (verbatim — no one else touched that file).
  - HAND-MERGE cpu_adam.py: SFT rewrote `_pin_if_requested` around pinned_ledger (reserve→pin→register). Composition: after a successful ledger reserve, try exact-pinned register_inplace first and `pinned_ledger.register_tensor(tensor,'adam')` on success; else fall through to ledger-tracked pin_memory. (Keeps ledger accounting truthful for exact-pinned bytes.)
  - Execute ONLY after Phase-3 queue completes (frozen-tree discipline while runs are live). Then: venv import gate + a gated smoke (ASYM_EXACT_PINNED=1 small run) + flags-off invariance spot-check.
  - **APPLIED (2026-08-01 ~20:1x, post-Phase-3)**: exact_pinned.py + model_capacity.md + scripts/lf/capacity/ + profiling_results/capacity_push_c17/ copied; host_weight.py hook applied verbatim; cpu_adam.py `_pin_if_requested` hand-merged (ledger reserve → exact-pinned register_inplace + `pinned_ledger.register_tensor` on success → ledger-tracked pin_memory fallback — keeps item-4 accounting truthful); LF `checkpointing.py` diff from 0132880a applied clean to 39's LF (guarded asym_gemm import; ASYM_EXACT_PINNED_ROOTS / UNSLOTH_GC_OUTER_HBM_AUTO / ASYM_HOST_FLUSH_EVERY_N_LAYERS all default OFF). ef9bc80 overleaf gitlink SKIPPED as planned.
  - Re-gate: compileall OK; **91/91 venv tests**; direct mechanism test PASS (register_inplace pinned 128 MiB in place, register_stats counted, H2D through registered memory OK).
  - Flags-OFF smoke: **FIT**, losses within noise of the pre-Phase-4 smoke (1.6845/2.0747/1.0636 vs 1.6767/2.0726/1.0657) — default invariance holds.
  - **GATED-ON INCIDENT (found + fixed)**: ASYM_EXACT_PINNED=1 smoke crashed in the first fg forward — `cudaErrorInvalidValue` at `_stage_weight_panel` `.to()` of the qwen3 expert bank (confirmed the true site under CUDA_LAUNCH_BLOCKING=1; ROOTS gate exonerated by isolation run). Elimination trail: fresh-clone register+cast H2D OK; safetensors-copy OK; 2/8/24-GiB registrations with deep-offset slices OK; unit tests with MIN_MB=0.05 OK; synthetic HostWeight bank + stage OK — only the REAL loader path failed. ROOT CAUSE: `AsymQwen3Experts` builds banks with `clone=False` and bf16 sources make `gate_up.to(dtype=bf16)` a NO-OP ALIAS → `register_inplace` pinned the **HF param's own storage = a transformers-5.x safetensors MMAP slice**; registration grabs the file-backed slab and async DMA from file-backed registered pages throws invalid-argument on GB200/torch-2.12/CUDA-13. Stock `pin_memory()` COPIES out of the mmap — that's why stock works, and why 38 (synthetic-model capacity toolkit, no mmap sources) never saw it.
  - FIX (in `register_inplace`, protects every call site): refuse storages with `untyped_storage().filename != None` → return "mmap_backed" → callers fall through to the stock pin-copy path. Zero behavior change for the anon-memory homes the feature targets (optimizer masters, GC roots, cloned banks); mmap-aliased weight homes keep stock semantics (which the site's own release step assumed anyway — in-place-pinning an aliased mmap slab would ALSO have silently pinned the whole file mapping, a latent capacity bug in the 38 branch even where DMA works).
  - mmap guard alone did NOT fix it — elimination continued: 480 GiB / 120-range registration scale test OK; side-stream OK; secondary-thread OK; graph-captured copy OK; post-fork OK; per-registration probes **491/491 OK at registration time** yet the same bank was uncopyable (sync AND async) by first forward; synthetic alias+register+release sequence did NOT reproduce ⇒ the surviving variable: registered-in-place aliases of REAL loader-owned params die mid-lifecycle (HF/PEFT param machinery around the release).
  - **FINAL FIX — ownership-gated registration** (host_weight.py): register_inplace fires only when HostWeight `exclusively_owned` the buffer (it cloned, device-copied, or contiguity-copied). Aliased clone=False homes (the qwen3 bank path) fall back to stock pin_memory copy — release-safe by construction; owned homes (cpu_adam fp32 masters/state, cloned banks, RootPool buffers) keep the exact-pinned capacity win. Never page-lock memory you don't own.
  - **RESULT: ASYM_EXACT_PINNED=1 smoke FIT; ASYM_EXACT_PINNED=1+ROOTS=1 smoke FIT; 91/91 venv tests; flags-OFF invariance already proven.** Kept (env-gated, off by default): ASYM_EXACT_PINNED_DEBUG registration probes (exact_pinned.py) + the stage-panel async/sync diagnostic (frozen_linear.py) — today's incident tooling, useful for the next stack port.
  - Phase 4 hardening delta vs 38's original: (1) mmap-backed storages refused (registering an aliased safetensors slab would silently pin the whole file mapping), (2) ownership gate (above). Both make the 38 feature SAFER without changing its validated semantics on owned memory. 38's capacity crowns (267.7B@128k / 340.4B@64k) remain as-recorded on c17's stack; re-crowning on this stack is future work (feature default-OFF).

## Docs-completeness pass (user ask, 2026-08-02)
Directory-level sweep of `agent/` (md/txt/pdf/png) across 46/SFT/38 vs 39 — beyond the git deltas already merged. Gaps found + taken: `agent/archive/results/activation_saving.md` + `agent/archive/results/archive/mm_profiling.md` (untracked leftover records present in all three older checkouts, never git-tracked — 39's checkout predates them), SFT's `agent/prompt.md` + `agent/impls/tmp.md` (working notes, previously skipped as scratch — taken per completeness directive), 2 screenshots, `agent/AsymLoRA (8).pdf`. Re-sweep: ZERO missing. Campaign/design docs already merged earlier: cpu_compute.md, previous_validation_results.md (SFT-updated), aymlora_kernels.md, ep.md, system_tier_justification.md, writing_prompt.md, archive/{fix_cpu_compute,merge_cpu_modules,merge_cpu_modules_compspec,placement}.md, related_work/*, fix_qwen3.5_tp.md, fix_plot_placeholders.md, model_capacity.md, overleaf archive, recovered-sessions, capacity_push_c17 evidence.

## Phase 3-ext (user directive 2026-08-02): 4 more models × 2 near-capacity cells
Queue: `agent/anchors_tmp/mrg_validation_queue2.sh` (X1–X8, serial GPU0, fresh tags). References:
| cell | config | recorded ref (tok/s · peak GiB · RSS; s/it where recorded) |
|---|---|---|
| X1 | llama3.3-70b T2 192k·b2 (MS=1024) | 548 sched-c06 / 545 CPU-matrix / 543 archived · 171.1 EXACT-line · RSS 963–982 |
| X2 | llama3.3-70b T2 448k·b1 WALL 97% | 280 / 279 / 275 · 182.4 EXACT-line · RSS 976–983 |
| X3 | glm4.7-flash T3TOK 192k·b5 | 158.8 · RSS 723 · losses 1.322/1.226/1.235 (c14; +V3 replication 157.9·719) |
| X4 | glm4.7-flash T1 192k·b2 | f1s_t1192 artifacts (c14): 812 compute / 845 wall tok/s · 93.5 · RSS 273 |
| X5 | mixtral-8x22b T2 320k·b1 | mxt2320 artifacts (c14): 670 compute / 685 wall · 173.8 · RSS 882 |
| X6 | mixtral-8x22b T3TOK 64k·b2 | tier-ladder anchor at3b2 (c14, current-code era): 58.7 · 2534 compute · RSS 908 |
| X7 | q3.5-122b T2 448k·b1 | c18 §8: 520.2 s/it · 861 · 171.3 (93%) · 846 (cross-node band) |
| X8 | q3.5-122b T2 480k·b1 | c18 §8: 563.6 s/it · 852 · 177.8 (96% EDGE — c14/c18 fragile-edge caveat on record) · 849 |
Dataset prep: llama 192k·n1024 + 448k·n512 copied from SFT LF/data; 122b 480k present; 122b ≥512k datasets never existed on this host (c18-only) — 480k is the deepest reproducible cell here.

## FINAL STATE (2026-08-01 ~21:0x)
- Phases 1, 1b, 2, 2b, 3 (5/5 PASS), 4 — ALL COMPLETE. This repo + this workspace's LlamaFactory now carry the full union of 39 + 46 + SFT + 38 (selective, gated), validated near-capacity with no regression.
- Uncommitted per house rule (Kevin commits): ~101 AsymGEMM paths + 6 LF files (dataset_info.json, parser.py, adapter.py, checkpointing.py, liger_kernel.py, moe.py) + Liger-Kernel's pre-existing 39 deltas.
- Rebuilt in-container: `asym_gemm/_C*.so` via `.venv/bin/python setup.py build_ext --inplace` (venv = the ONLY correct interpreter — see incident above).

Tree state after Phase 1+2: 96 changed paths (`git status --short | wc -l`), zero conflict markers in code dirs.
## Phase 2b (user directive 2026-08-01): merge SIBLING-repo deltas per workspace
Scope check across ALL editable-install siblings (what the .venv actually imports: asym_gemm, LlamaFactory, Liger-Kernel, deepspeed, ktransformers; transformers 5.6.0 is non-editable site-packages):
- **LlamaFactory** (all four forks at dev `2f2a8a7b`):
  - 39 dirty (KEPT): dataset_info.json + adapter.py + liger_kernel.py + moe.py (model_integration campaign's fixes).
  - 46 dirty (MERGED): `src/llamafactory/hparams/parser.py` — EP2 grad/weight-offload opt-in gate (pairs with the AsymGEMM-side EP2 merge; applied clean via git apply). dataset_info.json → union.
  - SFT dirty (MERGED): dataset_info.json only → union.
  - 38: commit `0132880a` touches model_utils/checkpointing.py (exact-pinned GC roots + HBM root parking + host-cache flush) → **Phase 4 item**; dataset_info dirty → union'd now.
  - dataset_info.json UNION: +80 keys → 724 total, ZERO value conflicts (pure additions in every repo). 68 dataset files referenced by merged-in keys copied into 39 LF/data (35.5 GiB, 0 failures); files for registrations 39 had already culled were NOT resurrected.
- **Liger-Kernel**: all four at `80f19ad`; ONLY 39 carries deltas (glm4_moe applier + monkey_patch) — nothing to merge in.
- **deepspeed**: all at `3dc98deb`, clean everywhere. **ktransformers**: all at `cf6477d` with the IDENTICAL dirty bf16_sft_moe.hpp (same md5 in all four) — pre-existing shared tweak, nothing to merge.
- Ops repair: ported `.repair_dataset_info.py` to the 39 workspace root (paths rewritten) and fixed tp_probe.sh's hardcoded `/workspace/AsymGEMM-SFT/` self-heal call to `${SFT_ROOT}` (stale path in this workspace).
- First validation-queue launch was STOPPED (V1 was still loading, no orphans) so the queue validates the COMPLETE merge incl. LF-side; relaunched after Phase 2b.

## Phase 3 RESULTS (queue relaunched 14:33, node c14, solo GPU0)
| cell | config | recorded ref (tok/s · GiB · RSS) | merged tree | verdict |
|---|---|---|---|---|
| V1 | q3-30b T2 120k·b8 | 2726 · 176.7 · ~539 (CPU-matrix c06); 2762 · 165.7 (sched c06); 2723 · 180.0 · 517 (archived c12/c14) | **2782 · 178.9 · 492** (mean step 345.0s ×2) | **PASS** — tok/s above ALL refs (+2.1% vs CPU-matrix); HBM under same-node archived 180.0 (+2.2 vs c06 ref = the record's own reserved-cache band); RSS leanest |
| V2 | glm4.5-air T3TOK 128k·b3 (96% cell) | 176.4 GiB · RSS 892 · losses 1.355/1.427/1.406 (SAME NODE c14, 2026-07-28) | **178.1 GiB · RSS 834 · losses 1.356/1.428/1.402** (step 464.7s, 826 tok/s) | **PASS** — HBM +1.7 (±2 band), RSS −58, losses match to ≤0.3% (bf16 noise); PAIR_NATIVE=1 flip regression-free on GLM |
| V3 | glm4.7-flash T3TOK 192k·b5 (86% cell) | 158.8 GiB · RSS 723 · losses 1.322/1.226/1.235 (SAME NODE c14) | **157.9 GiB · RSS 719 · losses 1.320/1.225/1.235** (step 1331s, 721 tok/s) | **PASS** — HBM −0.9 (leaner), RSS −4, losses ≤0.15% (step-3 exact to 4 decimals) |
| V4 | q3.5-35b T2 896k·b1 | 1492 · 175.6 · 469 (c18); 600.4 s/it | redo: **1495 · 176.0 · 445; 599.4 s/it** — jobs.tsv `failed:1` but ARTIFACTS COMPLETE (3/3 losses 0.759/0.720/0.723, profile.json + staged-dispatch verify written; trainer died in TEARDOWN) | **PASS** per the repo's own artifacts-complete rule (the recorded 2026-07-20 q3.5 teardown flake — tp_probe's fallback exists for exactly this); +0.2% tok/s, +0.4 GiB, RSS −24 vs c18 |

**PHASE 3 VERDICT: 5/5 near-capacity cells PASS — merged tree shows NO regression in throughput/HBM/RSS/loss on q3-30b (T2+T2B), q3.5-35b (T2), glm4.5-air (T3), glm4.7-flash (T3).**
| V5 | q3-30b T2B 1.1M·b1 | 380 · 144.5 (CPU-matrix c06); 385 · 152.9 · 906 (sched c06); 382 · 151.5 · 906 (archived) | **389 · 152.9 reserved (EXACT sched byte-line) / 120.2 ALLOCATED · RSS 810** (step 2830.4s ×2) | **PASS** — tok/s above ALL refs (+1.0–2.4%); reserved matches the validated byte-line EXACTLY; allocated 120.2 leaner than every line; RSS −96; policy stack verified ENGAGED via profile placement_policy block (P12 qknorm 613 + rope 144 fires, P13 restage 144, P15 gated off) — c06's 144.5 = reserve-cache variance, not a lost lever |

V4 INCIDENT (2026-08-01 16:43): "missing registration" despite the Phase-2b union — ROOT CAUSE: **46's own dataset_info.json had been wiped by an LF git sync** (the documented failure mode .repair_dataset_info.py exists for); the 896k keys were in NO source dataset_info — the on-disk .jsonl files are the source of truth. FIX: ran the ported `~/AsymGEMM-SFT-39/.repair_dataset_info.py` (flock-protected, safe next to the live V5) → 242 registrations repaired, 966 total, 896k train+eval PRESENT. Lesson: after dataset-file merges, ALWAYS run the repair script rather than trusting source dataset_info files. V4 redo queued behind V5 (same cell, MAX_SAMPLES=1024).
