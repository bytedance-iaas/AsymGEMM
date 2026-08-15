# AsymSFT local backup/sync runbook (2026-08-15, supersedes the 08-14 version)

SELF-CONTAINED runbook for an agent running ON THE LOCAL (destination) machine.
Everything you need is in this file: what to ssh into, what to pull, in which
waves, how to verify — plus two standing duties: **always assess existing
progress before transferring anything (§2)** and **always report per-wave
ETAs while running (§3)**. All transfers are resumable (`--partial`) and
re-runnable (incremental delta on re-run). Device/tunnel background lives in
the repo's `agent/devices.md`; you do not need it to execute this.

## 0. Access — what to ssh into

Source of truth is ONE host (everything lives on NFS `/home` shared by all
GB200 nodes, so any node works):

    kevinni@s04-p1-dgx-02-c17     (IP 10.78.202.45; primary work node)
    fallbacks: s04-p1-dgx-02-c12 (10.78.202.40), s04-p1-dgx-02-c14 (10.78.202.42)

These names/IPs only resolve INSIDE the cluster network. All ssh arrives via
the internal bastion **10.78.200.8** — from outside you need VPN/corp access
to that first, then `-J`:

    RSH='ssh -J <you>@10.78.200.8'      # or RSH='ssh' if already routed
    N=kevinni@s04-p1-dgx-02-c17

NOTE: `gb200-kevin-45` and friends are VS Code *tunnel* names, NOT ssh hosts —
never try to ssh to them.

PREFLIGHT (do not skip):
```bash
$RSH $N hostname          # must print: s04-p1-dgx-02-c17
df -h .                   # local free space: ≥25G for waves 1+2; +90G wave 3; +460G wave 4
rsync --version | head -1 # need rsync ≥ 3.1 for --info=progress2
                          # (macOS stock rsync/openrsync is too old: brew install rsync)
```

## 1. What is being backed up — two sources of truth on the remote

    ~/AsymGEMM-SFT-38/   the canonical repo tree (code, agent/ docs+ledgers)
    ~/env/               shared scripts, figures, per-machine docs, overleaf,
                         and outputs/asymlora/ = the central run-artifact store

Sibling trees (`AsymGEMM-SFT{,-39,-46}`) are byte-identical clones — never
sync them. Node-local `/scratch_local` (HF caches, base weights, fused ckpts,
enroot images, tunnel state) is deliberately NOT backed up — rebuildable.

**Code rule — pull only the 5 repos with local work, whole and WITH `.git`:**

| repo (under `AsymGEMM-SFT-38/third_party/`) | size | local work |
|---|---|---|
| `AsymGEMM` | 11G (8.4G `archive/`) | main repo; local-only branches + stash |
| `LlamaFactory` | 85G (84G is `data/` → wave 3) | merged ports, fsdp2, appliers |
| `ktransformers` | 5.5G | arm bf16 SFT MoE kernel (dev, pushed to fork) |
| `Megatron-Bridge` | 364M | local `dev` commit: Mixtral bridge + TE-2.16 compat — EXISTS ONLY IN THIS `.git` |
| `Liger-Kernel` | 57M | liger appliers |

`.git` MUST come along: AsymGEMM has local-only branches (`merge_4way*`,
`main_kevin_nemo`, `main_kevin_model_capacity`, `backup/*`) and a stash that
GitHub does NOT have, and Megatron-Bridge's `dev` commit is unpushed (no fork
exists). A GitHub clone is NOT a substitute.

The other 14 third_party repos are clean upstream checkouts (~1.1G): don't
pull; re-create any later from `env/agent/third_party_patches/MANIFEST.tsv`
(origin URL + pinned SHA + branch; arrives with wave 2) via
`git clone <origin> && git checkout <sha>`.

**Symlink layout rule (why waves 1 and 2 belong under ONE parent):** the
repo's data dirs (`profiling_results`, `runs/{live,history,docs}`,
`scripts/figures`, `agent/{project_rules.md,RESEARCH_RULES.md,overleaf}`) are
RELATIVE symlinks into `../../../env/...`. Mirror both trees side-by-side —
`<DEST>/AsymGEMM-SFT-38/...` and `<DEST>/env/...` — and they resolve locally
by construction. Therefore:
- links DANGLE after wave 1 alone — expected, not an error; do not "fix" them;
- NEVER use `-L`/`--copy-links` (duplicates the store, chokes on two
  intentionally-dead links) and never `--safe-links` (drops the links);
- the store itself has zero symlinks/hardlinks — plain rsync = complete data.

## 2. ALWAYS FIRST: assess current progress — never start blind

`DEST` may already hold a partial backup from an earlier session. Before any
transfer, determine per wave what is already downloaded and what remains, and
REPORT it. The probe is each wave's exact rsync command with `-n --stats`
appended (dry run — transfers nothing):

```bash
DEST=~/asymsft; mkdir -p "$DEST"; cd "$DEST"
OPT=(-az --partial --info=progress2 --no-inc-recursive)
EXC=(--exclude='.venv*' --exclude='.aioenv' --exclude='build/'
     --exclude='__pycache__/' --exclude='*.egg-info' --exclude='stubs/'
     --exclude='.cache/' --exclude='.pytest_cache/' --exclude='.ruff_cache/')

ls -la .; du -sh AsymGEMM-SFT-38 env 2>/dev/null      # what exists at all?
# then for EACH wave below: rsync -n --stats <that wave's exact arguments>
#   "Number of regular files transferred: 0"  → wave already COMPLETE
#   "Total transferred file size: X bytes"    → X bytes still to pull
```

Then report a status table BEFORE starting, e.g.:

    wave 1 code:      partial — 2.1G of 18G remaining
    wave 2 env:       complete
    wave 3 LF data:   not started (84G)
    wave 4 history:   not started (455G) — confirm wanted before pulling

Only after this report, start transferring (remaining waves only).

## 3. The waves — with mandatory ETA reporting

`OPT` above gives one live overall-progress line per transfer
(`--info=progress2` = total %, rate, ETA; `--no-inc-recursive` makes the %
and ETA truthful by scanning the full file list upfront — expect a short
pause before numbers appear; `-v` is deliberately omitted so the progress
line stays readable).

**ETA protocol (do this for every wave):**
- BEFORE the wave: state bytes remaining (from the §2 dry run) and expected
  duration. Until a rate is measured, say "rate unknown — measuring"; after
  ~2 min of the first transfer, use the observed MB/s to give ETAs for this
  AND all remaining waves.
- DURING: relay rsync's progress2 line (%, MB/s, ETA) at the start and then
  every ~5 minutes or every 10% step, whichever is coarser.
- AFTER: report actual duration + average rate, and refresh the ETAs of the
  waves still to come using the measured rate.

```bash
# WAVE 0 — refresh the clean-repo manifest on the remote (fast)
$RSH $N 'bash env/agent/third_party_patches/regen.sh'
# In its output, every listed repo must show dirty_files=0. If any is >0,
# that repo has NEW local work: add it to the wave-1 list below.

# WAVE 1 — SOURCE CODE (~18G): the 5 modified repos, whole, incl. .git
rsync "${OPT[@]}" -R "${EXC[@]}" --exclude='LlamaFactory/data/' -e "$RSH" \
  "$N:AsymGEMM-SFT-38/.repair_dataset_info.py" \
  "$N:AsymGEMM-SFT-38/third_party/AsymGEMM" \
  "$N:AsymGEMM-SFT-38/third_party/LlamaFactory" \
  "$N:AsymGEMM-SFT-38/third_party/ktransformers" \
  "$N:AsymGEMM-SFT-38/third_party/Megatron-Bridge" \
  "$N:AsymGEMM-SFT-38/third_party/Liger-Kernel" .
# (-R recreates the full AsymGEMM-SFT-38/third_party/<repo> paths under DEST —
#  exactly the layout the symlinks need.)

# WAVE 2 — ARTIFACTS, current (~1.6G): env/ = live store, docs, figures,
# overleaf, rules, scripts, third_party_patches manifest
rsync "${OPT[@]}" --exclude='outputs/asymlora/history/' \
  --exclude='outputs/asymlora/.trash_root_owned/' -e "$RSH" \
  "$N:env" .

# WAVE 3 — OPTIONAL, LF DATASETS (+84G): registered jsonl packs
mkdir -p AsymGEMM-SFT-38/third_party/LlamaFactory/data
rsync "${OPT[@]}" -e "$RSH" \
  "$N:AsymGEMM-SFT-38/third_party/LlamaFactory/data/" \
  AsymGEMM-SFT-38/third_party/LlamaFactory/data/

# WAVE 4 — OPTIONAL, RUN HISTORY (+455G): the frozen archive; makes
# runs/history resolve. Selective slices work: append e.g. history/sft38/
# (13G; sft 290G, sft39 94G, sft46 ~59G) to both paths.
mkdir -p env/outputs/asymlora
rsync "${OPT[@]}" --exclude='.trash_root_owned/' -e "$RSH" \
  "$N:env/outputs/asymlora/history" env/outputs/asymlora/
```

Waves are independent: any order, any subset, re-run any time (a re-run
transfers only the delta — that is also how you resume after interruption).
Nothing needs fixing up afterwards — the moment wave 2 lands, wave-1
symlinks start resolving on their own.

## 4. Verify

```bash
# symlinks alive (only after wave 2):
ls AsymGEMM-SFT-38/third_party/AsymGEMM/runs/live/ \
   AsymGEMM-SFT-38/third_party/AsymGEMM/profiling_results/
# local-only git state made it:
git -C AsymGEMM-SFT-38/third_party/AsymGEMM branch -a | grep merge_4way
git -C AsymGEMM-SFT-38/third_party/AsymGEMM stash list          # 1 entry
git -C AsymGEMM-SFT-38/third_party/Megatron-Bridge log --oneline -1  # dev: Mixtral bridge + TE compat
# clean-repo manifest present:
column -t env/agent/third_party_patches/MANIFEST.tsv | head
# final progress statement: re-run the §2 dry-run probes — every wave you
# pulled must now show "Number of regular files transferred: 0".
```

Known-good quirks — do NOT "repair" these:
- `AsymGEMM/third-party/{LlamaFactory,lorafusion}` (hyphen dir INSIDE the
  repo) are dangling absolute links to `/home/shutianluo/...` — dead on the
  cluster too. The real LlamaFactory is the wave-1 parent-level copy.
- `runs/history` dangles until wave 4 — expected.
- venvs are absent by design — rebuilt per machine by
  `scripts/lf/bootstrap_lf_venv*.sh`.
