# merge_plan — TASK PROMPT: merge SFT-38's AsymGEMM onto us (SFT-39)

You are working in /home/kevinni/AsymGEMM-SFT-39/third_party/AsymGEMM ("OURS",
branch main_kevin, HEAD c70c9a1 + uncommitted EP work in the tree). Merge the
committed work from /home/kevinni/AsymGEMM-SFT-38/third_party/AsymGEMM
("THEIRS", branch main_kevin, HEAD ead5d01, clean tree) onto us WITHOUT losing
one line of our uncommitted EP work. Do not stop until the verification
checklist below passes twice in a row.

## EXECUTION ENVIRONMENT (host vs container — verified 2026-07-13)

You run on the HOST, outside the enroot container, and must stay there for
all editing/git work: the container mounts ONLY this workspace, so THEIRS'
path /home/kevinni/AsymGEMM-SFT-38 DOES NOT EXIST inside it — S2's fetch
would fail there. Do ALL git operations, file edits, greps and `test -f`
checks on the host.

NEVER execute code on the host — it won't work there. Anything that invokes
an interpreter (`bash -n`, `python -m py_compile`, any repo script) runs
INSIDE the enroot container `asym_sft_39`, brought up with
`asym39_enroot_run` (defined in ~/env/bashrc.sh, sourced by ~/.bashrc; the
user sometimes writes it "asym49_enroot_run" — only asym38/39/40 exist, 39
is this repo's).

The container bind-mounts /home/kevinni/AsymGEMM-SFT-39 at
/workspace/AsymGEMM-SFT-39 — SAME files, different path: this repo is
/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM inside, and host edits are
instantly visible there; never copy files across.

`asym39_enroot_run` takes no command argument (only --gpus=N) and lands in
an interactive shell, so from a non-interactive agent feed commands via
stdin (this exact pattern is verified working):

    source ~/env/bashrc.sh   # REQUIRED first in the same command: the
                             # tool-shell snapshot has the wrapper but NOT
                             # the _custom_enroot_* helpers it calls
    asym39_enroot_run <<'EOF'
    cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
    bash -n scripts/lf/run_lf_lora_sft.sh && echo CHECK_1_OK
    PYTHONPYCACHEPREFIX=/tmp/pyc python3 -m py_compile \
      scripts/lf/ceiling_search.py && echo CHECK_2_OK
    exit
    EOF

  - Print an explicit OK/FAIL marker per check inside the heredoc and grep
    the captured output for it — don't trust exit-status plumbing through
    enroot + the interactive shell.
  - Ignore the NVIDIA license banner and the "cannot set terminal process
    group" / "no job control" stderr noise — normal for piped stdin.
  - Shells inside run as ROOT: never run git in there (root-owned .git
    writes + dubious-ownership refusals), and keep PYTHONPYCACHEPREFIX=/tmp
    so pyc litter stays out of the host tree.
  - Container asym_sft_39 already exists. If the wrapper prints that it is
    CREATING or SETTING UP a container, something is wrong — stop and
    report instead of letting it install tooling.

## GROUND TRUTH (verified 2026-07-13 — trust this over re-derivation)

Git topology: THEIRS' history CONTAINS our HEAD. ead5d01 = c70c9a1 + exactly 5
commits (c206065, cb744b9, ad3ef05, 89d1be1, ead5d01, all titled "updates").
So this is a textbook 3-way merge with merge-base c70c9a1:

  THEIRS (c70c9a1..ead5d01, committed; 17 files, +484/-117):
    - scripts/lf/run_lf_lora_sft.sh       — per-model watchdog floor map
      WATCHDOG_FLOOR_GB_BY_MODEL + hard error for unmapped models + kill
      grace 60s -> 1s (their hunk @@ lines ~218-250)
    - scripts/lf/ceiling_search.sh        — RENAMED to ceiling_search_both.sh
      (heavily edited) + NEW ceiling_search_source.sh (per-profiler split,
      injects PROFILERS/HOST_TAG env)
    - scripts/lf/ceiling_search.py        — CUDNN_STATUS_INTERNAL_ERROR as
      G-OOM pattern, ohbm ladder [0,16,8,7,6,5,4,3,2,1], per-host+profiler
      artifact roots profiling_{prof}_ceiling_{host}, confirm-OK resets the
      max_confirm_attempts streak
    - scripts/lf/ceiling_table.py         — companion changes (+28)
    - scripts/lf/build_lf_sft_eval_pair.py — NFS-safe concurrent builds:
      fcntl locks + tmp-file + os.replace atomic writes (+27)
    - scripts/lf/{compare_liger_loss_profiles,compare_nvme_profiles,
      migrate_ker_ceil_axis,migrate_liger_loss_axis,migrate_ohbm_axis}.py
      — moved into scripts/lf/archive/
    - scripts/lf/profile_lora_lf_test_{both,source}.sh — ONLY RUNS-array
      comment toggles (their hunks @@ ~91-108; transient bench state)
    - .gitignore — ceiling_search_state/ line -> `ceiling*/**` with
      whitelist !ceiling_reconfirm_state/{extract_metrics.py,
      numa_leak_monitor.sh}; adds archive/
    - agent/RULES.md (new), agent/handoffs/ceiling.md (new, 81 lines),
      agent/handoffs/Screenshot 2026-07-11....png (new)

  OURS (uncommitted working-tree diff vs c70c9a1; 12 files, +920/-136, plus
  4 untracked: agent/impls/fix_ep.md, agent/merge_plan.md [this file],
  agent/plot_ep.md, scripts/testing/print_skew_tables.py):
    - asym_gemm/training/{ep_sep,ep_vanilla,frozen_linear,qwen3_moe}.py,
      csrc/jit/compiler.hpp — the EP balancing + fix_ep detox work. THEIRS
      NEVER TOUCHED THESE. They must survive byte-identical.
    - scripts/lf/run_lf_lora_sft.sh — NAMING EPOCH 4 (sepqueue2/sepplan2),
      ASYM_EP_SEP_MODE, ASYM_EP_VANILLA_FUSED/ALIGN_EVERY exports (our hunks
      @@ ~423-470 and ~637 — DISJOINT from their watchdog hunk)
    - scripts/lf/profile_lora_lf_test_{both,source}.sh — zipf skew z<s>
      model-field parsing, epoch-4 backend tables (our hunks @@ 280+ —
      DISJOINT from their RUNS hunks)
    - scripts/lf/run_lf_profiled_train.py, scripts/testing/
      ep_balance_bench.{py,sh}, ep_sep_probe.py — ours only.

  CONSEQUENCE: in every shared file the two sides' hunks are line-disjoint;
  git should auto-merge everything EXCEPT .gitignore (both edited the same
  region). If you see any other conflict, resolve by the rules below — never
  by dropping a side wholesale.

## MERGE PROCEDURE (staged; do not reorder; ALL of S1-S6 on the HOST — the
## SFT-38 path is unreachable from inside the container)

  S1. Safety snapshot: `git add -A && git commit -m "pre-merge: SFT-39 EP
      work (epoch-4 naming, fix_ep detox, zipf bench)"` then
      `git tag pre-merge-sft38`. Nothing below may run on a dirty tree.
  S2. `git remote add sft38 /home/kevinni/AsymGEMM-SFT-38/third_party/AsymGEMM`
      (skip if exists) and `git fetch sft38 main_kevin`.
  S3. `git merge --no-ff sft38/main_kevin -m "merge SFT-38: ceiling-search
      split, NFS-safe eval-pair builds, per-model watchdog floors"`.
  S4. Resolve conflicts by these rules:
        .gitignore            -> take THEIRS verbatim (their `ceiling*/**`
                                 already covers our added ceiling_confirm_state/
                                 line; keep our line only if some our-side path
                                 is NOT matched by theirs — check with
                                 `git check-ignore`).
        run_lf_lora_sft.sh    -> keep BOTH: their watchdog-floor block
                                 (~218-250) AND our epoch-4 EP block (~423+).
        profile_lora_lf_test_*.sh -> their RUNS toggles are disposable comment
                                 state; on conflict keep OUR RUNS rows, take
                                 theirs everywhere else.
        asym_gemm/**, csrc/**, scripts/testing/** -> OURS, always (theirs has
                                 no changes here; a conflict means you erred).
  S5. Do NOT hand-copy their gitignored run-state dirs (ceiling_bound_state,
      ceiling_confirm_state, ceiling_probe_state, ceiling_reconfirm_state,
      ceiling_search_state_{both,source}_s04-p1-dgx-02-c14, scripts/lf/
      ceiling_bound_state etc.) — they are host-specific artifacts. The two
      whitelisted scripts (extract_metrics.py, numa_leak_monitor.sh) arrive
      via the merge; verify they did and are executable. List anything you
      deliberately skipped in the final report.
  S6. Commit the resolution (if S4 had conflicts) — do NOT push.

## VERIFICATION CHECKLIST (must pass TWICE in a row; log runs below)

  [ ] `git status` clean; `git log --oneline -3` shows the merge commit with
      parents = our pre-merge commit AND ead5d01.
  [ ] Ours intact: `git diff pre-merge-sft38 HEAD -- asym_gemm csrc
      scripts/testing scripts/lf/run_lf_profiled_train.py` is EMPTY.
  [ ] Theirs landed:
        grep -q WATCHDOG_FLOOR_GB_BY_MODEL scripts/lf/run_lf_lora_sft.sh
        grep -q CUDNN_STATUS_INTERNAL_ERROR scripts/lf/ceiling_search.py
        grep -q fcntl scripts/lf/build_lf_sft_eval_pair.py
        test -f scripts/lf/ceiling_search_both.sh
        test -f scripts/lf/ceiling_search_source.sh
        test ! -f scripts/lf/ceiling_search.sh   # renamed away
        test -f agent/RULES.md && test -f agent/handoffs/ceiling.md
        test -f scripts/lf/archive/migrate_ohbm_axis.py
  [ ] Ours still present in shared files:
        grep -q asym_sepqueue2_cpuadamwds scripts/lf/run_lf_lora_sft.sh
        grep -q ASYM_EP_VANILLA_FUSED scripts/lf/run_lf_lora_sft.sh
        grep -qE 'z\(\[0-9\]|parsed_model_zipf' scripts/lf/profile_lora_lf_test_both.sh
  [ ] Syntax — INSIDE the container via the stdin recipe in EXECUTION
      ENVIRONMENT (never on the host): `bash -n` on every changed/new .sh
      (run_lf_lora_sft.sh, profile_lora_lf_test_{both,source}.sh,
      ceiling_search_{both,source}.sh, ep_balance_bench.sh) and
      `python -m py_compile` on every changed .py (ceiling_search.py,
      ceiling_table.py, build_lf_sft_eval_pair.py, plus our four training
      files and the testing scripts) — one OK marker per file, all present
      in the output. Everything else in this checklist (git/grep/test/
      eyeballing) runs on the host.
  [ ] Watchdog logic sane: with HOST_MEM_WATCHDOG=true and an unmapped model
      the script must ERROR OUT (exit 2), not silently default — eyeball the
      merged block, don't run a training job.
  [ ] NO e2e GPU run required for this merge (both sides' training-path code
      is disjoint); do not launch profiling jobs.

## ROLLBACK

  Anything unrecoverable goes wrong -> `git merge --abort` (mid-merge) or
  `git reset --hard pre-merge-sft38` (post-commit), then report instead of
  improvising.

## RUN LOG (append-only: iteration -> what failed -> what changed)

  2026-07-13 EXECUTED (Claude, host + asym_sft_39 container):
  - S1 pre-merge commit 240be99, tag pre-merge-sft38. S2 fetch OK
    (merge-base = c70c9a1 as predicted). S3 merge: ONLY .gitignore
    conflicted, as predicted. S4: took THEIRS' .gitignore verbatim;
    `git check-ignore` confirmed their `ceiling*/**` covers our
    ceiling_confirm_state/ (our line dropped). S6 merge commit d31e522,
    parents 240be99 + ead5d01.
  - DEVIATION at S5: the two whitelisted scripts (ceiling_reconfirm_state/
    extract_metrics.py, numa_leak_monitor.sh) do NOT exist in SFT-38 at
    all — neither tracked in ead5d01 nor on its disk. Nothing to verify or
    copy; their .gitignore whitelist lines are anticipatory. Skipped all
    host-specific run-state dirs per S5.
  - Verification round 1: all host checks OK; all 17 container syntax
    checks OK (bash -n x6, py_compile x11); watchdog block eyeballed —
    unmapped model hard-errors exit 2 at run_lf_lora_sft.sh:240.
  - Verification round 2: all host checks OK; 17/17 container checks OK.
    CHECKLIST PASSED TWICE IN A ROW.
  - Post-merge WORKLOAD TESTS (user-requested, all inside asym_sft_39,
    GPUs 2,3; torch 2.12.0+cu130, 4 GPUs visible):
      * import smoke: ep_sep/ep_vanilla/frozen_linear/qwen3_moe all import
        clean under .venv python.
      * arg-parse smoke: ceiling_search.py --help and
        build_lf_sft_eval_pair.py --help OK. ceiling_table.py exits 1
        "no ceiling_search_state_*/results.jsonl" — its own graceful
        no-data path (state dirs deliberately not copied), NOT a bug.
      * ep_sep_probe.py --gpus 2,3: PR5_PASS bitwise=True for BOTH
        --mode queue and --mode plan (arm/decline/steal/gather all fire).
      * ep_balance_bench.sh (q3-30b-a3b, worst layer, natural+a0.15,
        modes owned/plan/queue, m=1.28M smoke): completes rc=0, sane
        imbalance/timing numbers, JSON written.
      * merged watchdog negative test: MODEL_NAME_OR_PATH=fake/Unmapped
        HOST_MEM_WATCHDOG=true -> error + exit 2 before any launch.
    ALL WORKLOAD TESTS PASS.
