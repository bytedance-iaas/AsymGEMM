# Session recovery record — writing / plotting agents (b3bf7ddd, 359085da, 4e6a4616)

(2026-07-27, written by the recovery session in the host checkout
`/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM`. Companion to the earlier
`AsymGEMM-SFT-39/.../recovered-sessions/RECOVERY.md` for the MoE agent.)

## What actually happened (root cause — same failure as the MoE loss)

**No session was lost — only the terminal attachment was.** The writing
agents ran inside an enroot container that mounts this repo at
`/workspace/AsymGEMM-SFT/third_party/AsymGEMM`, under tmux server PID
**980214** (socket `/home/kevinni/.tmux.sock`, session name `agent`, created
Jul 26 22:31). A second tmux server (PID 3437015) was later started **on the
same socket path**, unlinking 980214's socket; by this morning no server
owned the path at all. `tmux attach` therefore failed, but the processes
never died.

**Fix applied: `kill -USR1 980214`** — tmux recreates its socket on SIGUSR1.
The session is reachable again:

    tmux -S /home/kevinni/.tmux.sock attach -t agent

That window holds the still-alive claude process **PID 996895** (pts/25,
started Jul 26), whose latest conversation is the writing session b3bf7ddd
(transcript last touched 10:47 today; conversation itself ended cleanly).
Full scrollback is intact.

**Prevention (same lesson as last time):** one tmux server per socket —
use `tmux -L <name>` per agent, or never start a second server on an
explicit shared socket path. Today's new sessions (`agent`/`writing`/
`writing2` on `/scratch_local/user_data/shutian/kevin/.tmux.sock`) share one
NEW server — fine — but do not point anything at `/home/kevinni/.tmux.sock`
again while 980214 lives, and do not SIGUSR1 3437015 (it would re-clobber).

## The three recovered sessions (all ended at clean resting points)

| id | role | span (local) | ending state |
|---|---|---|---|
| `b3bf7ddd` | **Overleaf writing agent** — prose↔figure sync; §3 component renaming (Training Kernels / Configuration Selector / Runtime Controller / Dynamic Expert Balancer); §3.1 rewritten in DirectKV overview register; wrote the module-3 naming handoff `agent/impls/prompt.md` | Jul 26 22:54 → Jul 27 02:16 | "Done and pushed (`e4b4f95`)" — Overleaf local == remote |
| `359085da` | **EP / figure-7 plotting agent** — worked from `agent/impls/ep.md`; M5b fair-EP grain fix, planner spill fix (no over-stream at z=2.0), panel-c re-timing; Overleaf commits `42627b9`, `4ddb1f0` pushed | Jul 26 22:58 → Jul 27 01:56 | Clean; left ONE user action: push `main_kevin` (see below) |
| `4e6a4616` | Short ep.md claim-by-claim verification paste (interrupted) | Jul 26 22:44 | trivial |

**Outstanding user action from 359085da (still true as of this recovery):**
`main_kevin` is **ahead 2** of origin (`a0ef616`, `3a2bb8d` — the
ep_balance_bench fixes). The container had no GitHub credentials. Push from
the host: `git push origin main_kevin`.

## What is archived here (md5-verified against the container originals)

| file | what |
|---|---|
| `b3bf7ddd-….jsonl` | writing agent, raw transcript (1.7 MB, 512 records) |
| `359085da-….jsonl` | plotting agent, raw transcript (3.1 MB) |
| `4e6a4616-….jsonl` | ep.md verification stub (22 KB) |
| `b3bf7ddd_WRITING_conversation.md` | readable extraction (user + assistant text) |
| `359085da_EP_PLOTTING_conversation.md` | readable extraction |

Container originals live in
`/proc/996895/root/scratch_local/.../projects/-workspace-AsymGEMM-SFT-third-party-AsymGEMM/`
(container-private disk — gone if the container is wiped; hence these copies).
Resumable copies are also staged in the HOST project slug
`/scratch_local/user_data/shutian/kevin/.claude-kevin/projects/-home-kevinni-AsymGEMM-SFT-third-party-AsymGEMM/`,
so from this checkout `claude --resume b3bf7ddd-…` / `--resume 359085da-…`
works. **Prefer re-attaching the tmux session over resuming the staged copy
while PID 996895 is alive** — two live copies of one session id diverge.

## Where everything lives (the writing-context map)

- **Overleaf project (git, syncs to Overleaf remote):**
  `/home/kevinni/env/overleaf/[MLSys 26 Sub] Superchip-based LoRA/`
  (= `agent/overleaf` symlink in this repo). Paper: *AsymLoRA — long-context
  LoRA fine-tuning on GB200/Grace-Blackwell superchips*, MLSys'26 sub.
  - `sections/` — the tex: `motivations.tex` (§2), `system.tex` (§3),
    `main_results.tex`, `abstract.tex`; `figures/` — final PDFs incl.
    `AsymLoRA.pdf` system figure.
  - `drafts/` — **the rhetoric + writing docs (ground truth for prose):**
    - `related_work_writing_craft.md` — writing-craft analysis of the
      related-work design sections; **DirectKV (OSDI'26) = the house
      register** §3.1 mimics.
    - `motivation_v2.md` (§2 titles+rhetoric skeleton) and
      `motivation_full_v2.md` (§2 full prose).
    - `system_writing_v2.md` (§3 skeleton + style rules + gates C2/C5) and
      `system_writing_full_v2.md` (§3 full prose).
    - `motivation_v2_plots.md` — §2 figure generation specs (M0a…M6, M5b-row3
      final form) + expected numbers.
    - `toconfirm.md` — built-but-unverified modules; **used-only rule**: a
      module enters §3 prose only after a CONFIRMED in-container measurement.
  - Same drafts are mirrored in `agent/impls/s04-p1-dgx-02-c06/` (the c06
    outputs dir where v0→v2 iterations live; env/ is NOT a git repo).
- **Related-work papers + notes:** `agent/related_work/` — PDFs (directkv,
  c2cserve/superinfer, coda, dwdp, erm, ktransformers, morphserve,
  superoffload) + distilled notes (`ktransformers.md`, `lorafusion.md`,
  `superoffload.md`, `zero3_offload.md`, `directkv.txt`…).
- **Plotting code:** `/home/kevinni/env/figures/` — `plot_m2_row.py/.sh`
  (figs 2–6, uniform size), `plot_ep_balance.py` (fig 7), `constants.py`,
  `plot_tp_vs_seq*.py` (fig 8/9 main results), output → overleaf `figures/`.
  Bench code feeding them: `scripts/motivation_bench/` in this repo.
- **Open feedback notes from Kevin:** `agent/prompt.md` (fig-5 rhetoric
  question; "fig 7b add back") — the overnight M5b/fig-7 commits addressed
  these; confirm on attach. `agent/impls/prompt.md` — module-3 naming
  handoff (RESOLVED: "Runtime Controller", names final per `e4b4f95`).
