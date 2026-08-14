# AsymSFT project — meaningful-download rules + rsync commands (2026-08-14)

Run these ON YOUR LOCAL MACHINE to pull everything meaningful from this
GB200 host. After the 08-14 4-way merge + artifact migration there are
exactly TWO sources of truth:

  /home/kevinni/AsymGEMM-SFT-38/   the canonical repo tree (code, docs,
                                   ledgers, all third_party incl. local
                                   patches in kt / Megatron-Bridge)
  /home/kevinni/env/               shared scripts, figures, per-machine
                                   docs, and outputs/asymlora/ = the
                                   central run-artifact store

The sibling trees (-39/-46/-SFT) are byte-identical clones now — never
sync them. Node-local /scratch_local (HF model caches, fused ckpts,
enroot images) is NOT needed: all re-downloadable/rebuildable.

## RULES — what to exclude and why (rebuildable or junk, never needed)
  .venv* .aioenv          python environments — rebuilt by
                          scripts/lf/bootstrap_lf_venv*.sh (per machine)
  build/ __pycache__/ *.egg-info stubs/   build products — regenerated
  .cache/ .pytest_cache/ .ruff_cache/     tool caches
  .git/                   history lives on GitHub (AsymGEMM + LF + Liger
                          all pushed); drop this exclude if you want the
                          full git history mirrored too
Everything else is meaningful: source, agent/ docs+ledgers, datasets/,
scripts, configs, the committed symlinks.

## LAYOUT REQUIREMENT (important)
Mirror BOTH roots side-by-side under one parent dir. The repo's committed
symlinks are RELATIVE (profiling_results -> ../../../env/outputs/...), so
with the sibling layout they resolve on your machine exactly like here:

  <DEST>/AsymGEMM-SFT-38/...
  <DEST>/env/...

Do NOT pass --safe-links to rsync — the links intentionally point outside
the transferred tree and --safe-links would drop them.

## SIZES (measured 08-14)
  tier 1  code+docs (repo tree minus LF data/)      ~16G
  tier 2  LlamaFactory data/ (generated jsonl packs) 84G
  tier 3  env minus store                            80M
  tier 4  artifact store env/outputs/asymlora/      455G

## COMMANDS (pull from your local machine; resumable, re-runnable)
```bash
REMOTE=gb200-kevin-45          # your ssh alias/tunnel for this host
DEST=~/asymsft                 # local parent dir
mkdir -p "$DEST"

# tier 1 — CODE + DOCS (repo tree, no envs/caches/git, LF data deferred)
rsync -avzP \
  --exclude='.venv*' --exclude='.aioenv' --exclude='build/' \
  --exclude='__pycache__/' --exclude='*.egg-info' --exclude='stubs/' \
  --exclude='.cache/' --exclude='.pytest_cache/' --exclude='.ruff_cache/' \
  --exclude='.git/' \
  --exclude='third_party/LlamaFactory/data/' \
  "$REMOTE:/home/kevinni/AsymGEMM-SFT-38/" "$DEST/AsymGEMM-SFT-38/"

# tier 2 — LF DATASETS (84G; the registered jsonl packs — pull when needed)
rsync -avzP \
  "$REMOTE:/home/kevinni/AsymGEMM-SFT-38/third_party/LlamaFactory/data/" \
  "$DEST/AsymGEMM-SFT-38/third_party/LlamaFactory/data/"

# tier 3 — ENV (scripts, figures, per-machine docs; excludes the big store)
rsync -avzP --exclude='outputs/asymlora/history/' \
  --exclude='outputs/asymlora/live/' --exclude='outputs/asymlora/.trash_root_owned/' \
  "$REMOTE:/home/kevinni/env/" "$DEST/env/"

# tier 4 — ARTIFACT STORE (455G; run when you want the run results.
# Selective pulls work too, e.g. append history/sft38/ to both paths.)
rsync -avzP --exclude='.trash_root_owned/' \
  "$REMOTE:/home/kevinni/env/outputs/asymlora/" "$DEST/env/outputs/asymlora/"
```

Notes
- Re-running any tier is an incremental update (rsync delta) — safe.
- Tiers 1+3 (~16G) give a fully browsable project: code, every ledger,
  figures, per-machine docs, and working runs/history symlinks once
  tier 4 (or a selective slice of it) is present.
- To also mirror git history, delete the `--exclude='.git/'` line in
  tier 1 (adds several GB; alternatively `git clone` from GitHub and
  rsync only artifacts).
- kt (`kt-kernel/.../bf16_sft_moe.hpp`) and Megatron-Bridge carry LOCAL
  UNCOMMITTED patches — they ride along in tier 1 automatically (one more
  reason to sync worktrees, not bare clones).
