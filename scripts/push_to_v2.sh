#!/usr/bin/env bash
# scripts/push_to_v2.sh — direct fast-forward push to
# origin/unified_kernel_sm90_v2 (no feature branch, no PR).
#
# Strategy:
#   1. Snapshot HEAD as a tag + stash working-tree changes.
#   2. Fetch remote; print per-title diff of local-only commits
#      vs remote-only commits.
#   3. RESET unified_kernel_sm90_v2 to origin/unified_kernel_sm90_v2
#      (drops duplicate local SHAs; picks up the net-new remote commits).
#   4. Pop stash to restore working-tree changes; EXPECT conflicts in:
#        - csrc_cpu/cpu_module.cpp
#        - asym_gemm/unified_moe/runtime.py
#      (script prints resolution recipes when --reset runs).
#   5. Build OFF + ON.
#   6. Stage all files.
#   7. Commit manually (script prints guidance).
#   8. Fast-forward push.
#
# Usage:
#   ./scripts/push_to_v2.sh                # stages 1-4: snapshot/fetch/title check
#   ./scripts/push_to_v2.sh --reset        # stages 5-7: reset + stash pop
#   ./scripts/push_to_v2.sh --continue     # stages 8-10: build + stage + commit guidance
#   ./scripts/push_to_v2.sh --push         # stage 11: fast-forward push
#   ./scripts/push_to_v2.sh -h | --help    # this text
#
# Rollback at any time before the final push:
#   git reset --hard local-pre-rebase-snapshot
#   git stash list && git stash pop stash@{N}

set -euo pipefail

REMOTE=origin
BRANCH=unified_kernel_sm90_v2
SNAPSHOT_TAG=local-pre-rebase-snapshot
WT_STASH_PREFIX="push-script wt-changes"
SCRATCH_WT=../asym_gemm_rebase_scratch

if [[ -t 1 ]]; then
  RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[0;33m'
  BLU='\033[0;34m'; BOLD='\033[1m';   NC='\033[0m'
else
  RED=''; GRN=''; YEL=''; BLU=''; BOLD=''; NC=''
fi

stage() { printf "\n${BOLD}${BLU}=== STAGE %s — %s ===${NC}\n" "$1" "$2"; }
ok()    { printf "  ${GRN}[ok]${NC}    %s\n" "$1"; }
warn()  { printf "  ${YEL}[warn]${NC}  %s\n" "$1"; }
err()   { printf "  ${RED}[fail]${NC}  %s\n" "$1"; }
note()  { printf "  ${BLU}[note]${NC}  %s\n" "$1"; }
pause() {
  printf "\n${YEL}>>> %s${NC}\n" "$1"
  read -r -p "Press Enter to continue, Ctrl-C to abort. " _
}

preflight() {
  stage 0 "Preflight"
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    err "Not inside a git repo."; exit 1
  fi
  REPO_ROOT="$(git rev-parse --show-toplevel)"
  cd "$REPO_ROOT"
  ok "Repo root: $REPO_ROOT"

  local cur
  cur="$(git rev-parse --abbrev-ref HEAD)"
  if [[ "$cur" != "$BRANCH" ]]; then
    err "Expected to be on '$BRANCH' but you are on '$cur'."
    err "Switch branches: git checkout $BRANCH"
    exit 1
  fi
  ok "On branch: $cur"

  if ! git config --get "remote.$REMOTE.url" >/dev/null; then
    err "Remote '$REMOTE' is not configured."; exit 1
  fi
  ok "Remote '$REMOTE': $(git config --get remote.$REMOTE.url)"

  # Clean up leftover scratch worktree from a previous failed dry-run.
  if [[ -d "$SCRATCH_WT" ]]; then
    warn "Found leftover scratch worktree at $SCRATCH_WT."
    git -C "$SCRATCH_WT" rebase --abort 2>/dev/null || true
    if ! git worktree remove --force "$SCRATCH_WT" 2>/dev/null; then
      rm -rf "$SCRATCH_WT"; git worktree prune
    fi
    ok "Removed."
  fi
}

stage1_snapshot() {
  stage 1 "Snapshot current state"
  git tag -f "$SNAPSHOT_TAG" HEAD
  ok "Tagged HEAD as '$SNAPSHOT_TAG' ($(git rev-parse --short HEAD))"

  if [[ -n "$(git status --porcelain)" ]]; then
    local msg="${WT_STASH_PREFIX} $(date -u +%FT%TZ)"
    git stash push -u -m "$msg"
    ok "Stashed working-tree changes: $msg"
  else
    ok "Working tree clean."
  fi
  note "Rollback: git reset --hard $SNAPSHOT_TAG && git stash pop"
}

stage2_fetch() {
  stage 2 "Fetch remote"
  git fetch "$REMOTE" --prune
  ok "Fetched $REMOTE."
  local ahead behind
  ahead="$(git rev-list --count "$REMOTE/$BRANCH..$SNAPSHOT_TAG")"
  behind="$(git rev-list --count "$SNAPSHOT_TAG..$REMOTE/$BRANCH")"
  note "Local snapshot is $ahead commits ahead of remote, $behind behind."
}

stage3_title_check() {
  stage 3 "Title-by-title divergence check"

  printf "\n  ${BOLD}Local-only commits (will be DROPPED by --reset):${NC}\n"
  git log --oneline "$REMOTE/$BRANCH..$SNAPSHOT_TAG" | sed 's/^/    /'

  printf "\n  ${BOLD}Remote-only commits (will be PICKED UP by --reset):${NC}\n"
  git log --oneline "$SNAPSHOT_TAG..$REMOTE/$BRANCH" | sed 's/^/    /'

  printf "\n  ${BOLD}Per-title check:${NC}\n"

  local local_titles remote_titles mb
  mb="$(git merge-base "$SNAPSHOT_TAG" "$REMOTE/$BRANCH")"
  local_titles="$(git log --format='%s' "$REMOTE/$BRANCH..$SNAPSHOT_TAG")"
  remote_titles="$(git log --format='%s' "$mb..$REMOTE/$BRANCH")"

  local only_local=0 match=0
  while IFS= read -r title; do
    [[ -z "$title" ]] && continue
    if echo "$remote_titles" | grep -Fxq "$title"; then
      printf "    ${GRN}[match]${NC}        %s\n" "$title"
      match=$((match + 1))
    else
      printf "    ${RED}[ONLY-LOCAL]${NC}   %s\n" "$title"
      only_local=$((only_local + 1))
    fi
  done <<< "$local_titles"

  printf "\n  Summary: ${GRN}%d match${NC}, ${RED}%d ONLY-LOCAL${NC}\n" "$match" "$only_local"

  if [[ "$only_local" -gt 0 ]]; then
    err "Some local commits have no remote-side equivalent."
    err "Inspect them before --reset (which would drop them)."
  else
    ok "All local commits have remote duplicates — safe to drop."
  fi
}

stage4_summary() {
  stage 4 "Summary"
  cat <<EOF

Next:  ${BOLD}./scripts/push_to_v2.sh --reset${NC}

--reset will:
  1. git reset --hard $REMOTE/$BRANCH    (drops local-only SHAs)
  2. git stash pop                        (restore working-tree changes)
  3. EXPECT conflicts in:
       - csrc_cpu/cpu_module.cpp
       - asym_gemm/unified_moe/runtime.py
     Resolution recipes will be printed.

Bail out instead:
  git tag -d $SNAPSHOT_TAG
  git stash pop

EOF
}

stage5_reset() {
  stage 5 "Reset $BRANCH to $REMOTE/$BRANCH"
  note "Before: $(git rev-parse --short HEAD)"
  git reset --hard "$REMOTE/$BRANCH"
  note "After:  $(git rev-parse --short HEAD)"
  ok "Top of history:"
  git log --oneline -3 | sed 's/^/    /'
}

stage6_restore_stash() {
  stage 6 "Restore working-tree changes"
  local stash_ref
  stash_ref="$(git stash list | grep -F "$WT_STASH_PREFIX" | head -1 | cut -d: -f1 || true)"
  if [[ -z "$stash_ref" ]]; then
    warn "No '$WT_STASH_PREFIX' stash found."
    return 0
  fi
  note "Popping $stash_ref ..."
  if git stash pop "$stash_ref"; then
    ok "Stash popped cleanly. No conflicts!"
    note "Skip to: ${BOLD}./scripts/push_to_v2.sh --continue${NC}"
    return 0
  else
    warn "Stash pop reported conflicts (expected)."
  fi
}

stage7_conflict_instructions() {
  stage 7 "Conflict resolution"
  local conflicts
  conflicts="$(git diff --name-only --diff-filter=U || true)"
  if [[ -z "$conflicts" ]]; then
    ok "No conflict markers. Proceed to --continue."
    return 0
  fi
  printf "\n  Conflict files:\n"; echo "$conflicts" | sed 's/^/    /'

  cat <<'EOF'

============== csrc_cpu/cpu_module.cpp ==============
Both sides ADD; keep everything from both:
  * Remote (f1d6777) added:
      RuntimeHandle::serial_rts vector
      ensure_serial(size_t count)
      fp32_to_bf16_rne helper
      moe_expert_forward_batch pybind entry
  * Local stash adds:
      Second RuntimeHandle ctor:
        RuntimeHandle(const std::vector<int>& numa_map,
                      const std::vector<int>& thread_count)
      n_numa() method
      Runtime.numa() static + n_numa property in PYBIND11_MODULE
      caps['has_numa'] in caps_dict()

============ asym_gemm/unified_moe/runtime.py ============
Take remote's signature + local's forward body:

  def __init__(self, slab, *, top_k, cpu_threads=0, cuda_device=0,
               m_cpu=16,
               runtime: Optional["_C.Runtime"] = None):    # <- remote
      self.slab = slab
      self.top_k = top_k
      self.cuda_device = cuda_device
      self.rt = runtime if runtime is not None else _C.Runtime(cpu_threads)
      self.m_cpu = m_cpu
      try:                                                 # <- keep local
          if torch.get_num_threads() > 1:
              torch.set_num_threads(1)
      except RuntimeError:
          pass

  @classmethod
  def from_bf16(cls, gate, up, down, *, top_k, ..., m_cpu=16,
                runtime: Optional["_C.Runtime"] = None):   # <- remote
      ...
      return cls(slab, top_k=top_k, ..., m_cpu=m_cpu,
                 runtime=runtime)                          # <- pass through

  def forward(...):
      # KEEP LOCAL's BODY — the _C.moe_int8 single-call path.
EOF

  cat <<EOF
After resolving:
  git add csrc_cpu/cpu_module.cpp asym_gemm/unified_moe/runtime.py
  ${BOLD}./scripts/push_to_v2.sh --continue${NC}

Bail out:
  git reset --hard $SNAPSHOT_TAG
  git stash pop

EOF
}

stage8_build_verify() {
  stage 8 "Build verification (OFF + ON)"
  if git ls-files -u | grep -q .; then
    err "Conflict markers still present:"
    git diff --name-only --diff-filter=U | sed 's/^/    /'
    exit 3
  fi
  if ! command -v python >/dev/null 2>&1; then
    warn "python not on PATH — skipping build."; return 0
  fi
  note "Building OFF (no NUMA)"
  if python setup.py build_ext --inplace >/tmp/build_off.log 2>&1; then
    ok "OFF build succeeded ($(wc -l </tmp/build_off.log) log lines)."
  else
    err "OFF build FAILED. Last 20 lines:"
    tail -20 /tmp/build_off.log | sed 's/^/    /'
    exit 4
  fi
  note "Building ON (NUMA)"
  if ASYM_GEMM_WITH_NUMA=1 python setup.py build_ext --inplace >/tmp/build_on.log 2>&1; then
    ok "ON build succeeded ($(wc -l </tmp/build_on.log) log lines)."
  else
    err "ON build FAILED. Last 20 lines:"
    tail -20 /tmp/build_on.log | sed 's/^/    /'
    exit 4
  fi
  ok "Both builds clean."
}

stage9_stage_files() {
  stage 9 "Stage files"

  # Code (existing only)
  local code_files=(
    csrc_cpu/cpu_infer.h csrc_cpu/cpu_infer.cpp
    csrc_cpu/task_queue.h csrc_cpu/task_queue.cpp
    third-party/cpu_gemm/src/dispatch/moe.cpp
    third-party/cpu_gemm/src/runtime/in_numa_pool.h
    third-party/cpu_gemm/src/runtime/in_numa_pool.cpp
    third-party/cpu_gemm/src/runtime/numa_job_distributor.h
    third-party/cpu_gemm/src/runtime/numa_job_distributor.cpp
    third-party/cpu_gemm/tests/test_worker_pool_numa.cpp
    third-party/cpu_gemm/src/dispatch/int8_rm_backend.h
    third-party/cpu_gemm/src/dispatch/int8_rm_backend.cpp
    third-party/cpu_gemm/src/kernels/avx512/int8_gemm_rm.h
    third-party/cpu_gemm/src/kernels/avx512/int8_gemm_rm.cpp
    third-party/cpu_gemm/tests/test_int8_rm_parity.cpp
    tests/bench_backend_throughput.py
    tests/bench_common.py
    tests/bench_compare_ktransformers.py
    tests/bench_estimate_cpu_pressure.py
    scripts/run_pressure_estimate.sh
    scripts/push_to_v2.sh
  )
  # Design docs (existing only)
  local doc_files=(
    estimate_cpu_kt.md cpu_comparison.md performance_compare.md
    performance.md kt_asygGEMM.md kt_synchronize.md
    overlap_unified_cuda_graph.md CPU_gemm_improve.md
    push_conflict_plan.md avx_512.md
  )

  note "Files to stage:"
  local f
  for f in "${code_files[@]}" "${doc_files[@]}"; do
    [[ -e "$f" ]] && printf "    %s\n" "$f"
  done

  pause "Confirm and stage them now"

  for f in "${code_files[@]}" "${doc_files[@]}"; do
    [[ -e "$f" ]] && git add "$f"
  done

  # Tracked-but-modified — pick up too
  git add csrc_cpu/cpu_module.cpp asym_gemm/unified_moe/runtime.py \
          setup.py tests/bench_unified_moe.py \
          third-party/cpu_gemm/CMakeLists.txt \
          third-party/cpu_gemm/include/cpu_gemm/cpu_gemm.h \
          third-party/cpu_gemm/include/cpu_gemm/runtime.h \
          third-party/cpu_gemm/src/dispatch/gemm.cpp \
          third-party/cpu_gemm/src/runtime/runtime.cpp \
          third-party/cpu_gemm/src/runtime/worker_pool.cpp \
          third-party/cpu_gemm/src/runtime/worker_pool.h \
          third-party/cpu_gemm/tests/CMakeLists.txt \
          2>/dev/null || true

  ok "Staged. Cached diff stats:"
  git diff --cached --shortstat | sed 's/^/    /'

  note "Files left untracked:"
  git ls-files --others --exclude-standard | sed 's/^/    /'
}

stage10_commit_guidance() {
  stage 10 "Commit guidance"
  cat <<'EOF'

Quick single-commit option:
  git commit -m "Add NUMA-aware WorkerPool + CPU MoE dispatcher + AVX-512-VNNI fallback + pressure estimator"

Or split into logical commits (see push_conflict_plan.md §5.7 for the
7-commit recipe). When done:
  ./scripts/push_to_v2.sh --push
EOF
}

stage11_push() {
  stage 11 "Fast-forward push to $REMOTE/$BRANCH"
  local cur
  cur="$(git rev-parse --abbrev-ref HEAD)"
  if [[ "$cur" != "$BRANCH" ]]; then
    err "Not on '$BRANCH' (on '$cur')."; exit 5
  fi
  if [[ -n "$(git status --porcelain)" ]]; then
    err "Working tree dirty. Commit first:"
    git status --short | head -20 | sed 's/^/    /'
    exit 5
  fi

  local remote_sha local_sha
  remote_sha="$(git rev-parse "$REMOTE/$BRANCH")"
  local_sha="$(git rev-parse HEAD)"
  if [[ "$remote_sha" == "$local_sha" ]]; then
    warn "Local HEAD == $REMOTE/$BRANCH. Nothing to push."; return 0
  fi
  if ! git merge-base --is-ancestor "$remote_sha" "$local_sha"; then
    err "Local HEAD is NOT a descendant of $REMOTE/$BRANCH."
    err "Refusing non-fast-forward push."; exit 6
  fi
  ok "Push is a fast-forward."

  note "Commits to push:"
  git log --oneline "$REMOTE/$BRANCH..HEAD" | sed 's/^/    /'
  note "Diff stat:"
  git diff --stat "$REMOTE/$BRANCH..HEAD" | tail -5 | sed 's/^/    /'

  pause "Final confirmation: push '$BRANCH' to $REMOTE?"

  git push "$REMOTE" "$BRANCH"
  ok "Pushed."
}

usage() { sed -n '2,30p' "$0"; exit 0; }

MODE=dryrun
case "${1:-}" in
  -h|--help)   usage ;;
  ""|--dryrun) MODE=dryrun ;;
  --reset)     MODE=reset ;;
  --continue)  MODE=continue ;;
  --push)      MODE=push ;;
  *)           err "Unknown flag: $1"; usage; exit 1 ;;
esac

preflight

case "$MODE" in
  dryrun)
    stage1_snapshot
    stage2_fetch
    stage3_title_check
    stage4_summary
    ;;
  reset)
    stage5_reset
    stage6_restore_stash
    stage7_conflict_instructions
    ;;
  continue)
    stage8_build_verify
    stage9_stage_files
    stage10_commit_guidance
    ;;
  push)
    stage11_push
    ;;
esac
