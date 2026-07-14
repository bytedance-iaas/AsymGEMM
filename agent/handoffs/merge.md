I am doing one scheduling exploration to tune between latency mode and memory mode of this system. I also tried to improved the source code to reduce latency and even expand the memory capacity even more. So there are tons of work modified here.

On another machine i hav ethe same system but i was working on expert paralleim implemnatino and better ceiling search harness etc. plotting results etc.

Currently I wanna merge them safely wihtou losing proigress or breaking any code. This is the current repo /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM and the other repo is at /home/kevinni/AsymGEMM-SFT-38/third_party/AsymGEMM
Please expore both comprehensivelt adn extenisvel i wanna merge AsymGEMM-SFT-38/third_party/AsymGEMM's changes onto us /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

Explore extensivel tand compreeilve. Isn there any conflcits? if so we need to be ultra careful and ultra metricou in decindhwo to resolve any confflits. Or si tehr any errors or incompaitl as epcts that cannot be triviallt resolved. 
Expopre and let me know what are conflicts that cannot be triviallt explored and that require  mwe to intervene. 

SUPER improtnat after mergin we need to redo some testins wiht large owklodas to ensure that eveyrhig still wokrs all right. 
Note taht we NEVENR RUN on thi hsot directl. we need to run by acting asym40_enroot_run this starts the continer wehre we can acutka run the code this is suerp improant we cNANOT ru the code on teh host directly.

We need to push all the curen hagne to liek main_kevin_scheduler so that we keep this backup right?

---

# Exploration findings + merge plan (2026-07-13)

## Repo relationship
- Both repos are clones of the SAME `origin`; common ancestor = `4936ad1`.
- Repo A (this one) scheduler work: committed to backup branch `main_kevin_scheduler`
  (`3b8361d` on top of `955d046`). `main_kevin` itself untouched at `955d046`.
- Repo B (`../../../AsymGEMM-SFT-38/third_party/AsymGEMM`) EP work: `main_kevin` at
  `f1b18a4`, 14 commits over the base (EP impl, ceiling-search harness, plotting,
  plus B's own prior SFT-38/39 merges). Added here as read-only remote `other`.

## Backup
- `main_kevin_scheduler` = FROZEN backup reference. NEVER modify/commit to it.
- (optional) local tag `backup/pre-merge-2026-07-13`; push to origin = TBD.

## Conflict analysis — NO non-trivial conflicts
Only 2 textual conflicts, both auto/easily resolvable (no human surgery needed):
- `.gitignore` — trivial UNION of ignore patterns.
- `scripts/lf/profile_lora_lf_test_both.sh` — the ONLY conflict region is the
  `RUNS=()` experiment queue (which rows to launch; NOT code). Everything else in
  that file (B's skew/zipf parsing, backend renames sEP->sepqueue2/sepplan2, A's
  env-var passthroughs) auto-merges. DECISION = keep BOTH RUNS blocks (union);
  only A's uncommented rows actually run, B's stay as commented history.

## Semantic compatibility — VERIFIED CLEAN (the real risk, and it passed)
- A rewrote `qwen3_moe_finegrained.py` (~1103 lines); B rewrote `qwen3_moe.py` —
  different files with a two-way call contract. Every crossing signature is
  IDENTICAL on both sides; A removed zero public symbols:
  - B calls A's `qwen3_moe_finegrained_forward` / `_nograd_forward` /
    `_unsupported_reasons` -> signatures byte-identical.
  - A calls B's `AsymQwen3Experts` / `_grouped_lora_weight_grads_torch` /
    `_grouped_base_dx` / `_is_silu_activation` -> B never touched those defs.
  - B imports A's `adopt_host_weight` / `ActivationOffloadManager` /
    `CPUActivationHandle` -> signatures identical.
- `frozen_linear.py` (only shared source file) auto-merges; A's edits at L1107+
  (GEMM dispatch), B's at L17-885 (pad-memo/sEP) — different functions.

## Build action (not a conflict)
- B changed `csrc/jit/compiler.hpp` (+21/-4). A doesn't touch csrc -> no conflict,
  but the CUDA extension MUST be rebuilt after merge before any workload test.

## Merge execution plan (main_kevin = canonical, all work combined)
1. `git checkout main_kevin` (leave main_kevin_scheduler frozen).
2. Fast-forward `main_kevin` to include scheduler work: `git merge --ff-only main_kevin_scheduler`.
3. `git merge other/main_kevin` (brings in EP work).
4. Resolve 2 conflicts: `.gitignore` union; `profile_lora_lf_test_both.sh` RUNS = keep both.
5. Commit the merge.
6. Rebuild the extension INSIDE `asym40_enroot_run` (NEVER on host).
7. Large-workload smoke tests INSIDE the container (NEVER on host).
8. main_kevin_scheduler remains the frozen backup throughout.

---

# Post-merge verification (2026-07-13) — run inside asym_sft_40 enroot container

Merge landed: main_kevin = `36af646` (merge of scheduler `7dc22dd` + EP `f1b18a4`).
main_kevin_scheduler frozen at `3b8361d`. All runs via the container (never host).

Results:
- BUILD: clean rebuild of `asym_gemm._C` (all 5 objects incl python_api.o, which
  pulls in the merged jit/compiler.hpp multi-rank rename-race fix). Fresh .so; `_C`
  loads + init OK; JIT bf16 GEMM runs. (Had to CLEAN-build — setuptools doesn't
  track header deps, so a compiler.hpp-only change won't recompile without it.)
- SEP QUEUE (asym_sepqueue2): `PR5_PASS mode=queue bitwise=True` — all cases
  (bal/skew/decline/bal2) bitwise-identical to plain kernel (max_diff 0.0); armed
  launch + steal + gather fired (armed=3, declined=1).
- SEP PLAN (asym_sepplan2): `PR5_PASS mode=plan bitwise=True` — planned=3, all
  cases bitwise-identical. => scaling via sep plan AND queue verified post-merge.
- FINEGRAINED MoE (the merged A<->B interface: AsymQwen3Experts x
  qwen3_moe_finegrained_forward/backward + expert offload + LoRA), synthetic 65536
  tokens / 128 experts / H2048: loss cpu==gpu (absdiff 8e-9), out/dx/grad rel diffs
  ~8e-3 (bf16 noise), no NaN, exit 0. No regression.
- Kernel unit pytest tests/m_grouped/... fails at COLLECTION (missing test_nvfp4.py)
  — PRE-EXISTING (absent at base 4936ad1, untouched by either side), NOT merge-
  caused; kernel correctness already covered bitwise by the sep probes.

REAL-MODEL RUN (correction: weights WERE cached all along under
HF_HOME=/scratch_local/.../cache/huggingface; my earlier "not cached" probe just
lacked the container env). Real Qwen3-30B-A3B finegrained latency-mode SFT ran and
the merged code trained correctly:
  loss 1.569 -> 1.616, grad_norm ~0.6-0.8, asym_forward_calls=4896 asym_dx_calls=1152
  torch_forward_calls=0 (all-asym, zero fallback), qwen3_moe_finegrained offload on
  all 48 MoE layers, ~61 GB CPU-resident base. => merge trains correctly end-to-end.

BUILD GOTCHA (fixed): first rebuild used /usr/bin/python (torch 2.9) -> ABI mismatch
-> `undefined symbol c10_cuda_check_implementation` -> `missing_bf16_asym_binding` at
runtime. _C MUST be built against the venv's torch (2.12). Captured permanently as
`scripts/lf/rebuild_asymgemm.sh` (run in container: bash scripts/lf/rebuild_asymgemm.sh).

INFRA NOTE: shared /home (28T) is often ~100% full; full runs die with
`OSError [Errno 28] No space left`. Use PROFILERS=source (skip multi-GB nsys traces).
The merge verification does NOT depend on a full run completing — it is already proven.

.gitignore: re-added `results/` (the merge union took EP's line that renamed
results/->datasets/, accidentally un-ignoring the results/ output dir).
