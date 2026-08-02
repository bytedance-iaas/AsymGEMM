# Session recovery record — MoE-integration agent (e068fbbb)

(2026-07-27. Written by the third session in the chain: e068fbbb = the lost
MoE-integration agent; f79da6d0 = first recovery session, did the forensics +
backups below, interrupted right before writing this record; this session
verified everything, archived the remaining artifacts, and wrote it.)

## What actually happened (root cause)

**The session was never lost — only the terminal attachment was.** The
MoE-integration agent runs to this minute:

- PID **1315307**: `claude --dangerously-skip-permissions --resume
  e068fbbb-a742-45da-8902-b6d3b53151db`, up 5+ days, inside enroot container
  **asym_sft_42** (repo mounted at `/workspace/AsymGEMM-SFT-39/third_party/
  AsymGEMM` — same repo as `/home/kevinni/...`, different mount point).
- Root cause (confirmed by f79da6d0): **five tmux servers were started on the
  same socket path**; each new server unlinked the previous server's socket.
  The servers holding sessions `run` (this agent) and `merge` are alive but
  their sockets are gone from the filesystem → `tmux attach` can never reach
  them again. Prevention: give each server its own socket (`tmux -L <name>`)
  or never start a second server on an explicit shared socket path.
- The transcript lives inside the CONTAINER's own `/scratch_local` (not a
  bind mount), under the container-side project slug. Host-reachable only via
  `/proc/1315307/root/scratch_local/user_data/shutian/kevin/.claude-kevin/
  projects/-workspace-AsymGEMM-SFT-39-third-party-AsymGEMM/`. If the
  container is wiped or the process dies, that copy is gone — hence the
  backups below.

## Session state at loss: CLEAN RESTING POINT

Last record 2026-07-27T12:08Z. Verified: no monitors, no background tasks,
GPUs idle, shm clean. Final agent message: phi T3 campaign complete
(75.6 GiB = 49.6% of baseline; capacity extended to 128k·b5), all results
recorded in `agent/impls/model_integration.md`, work left uncommitted per
standing practice. Nothing was in flight; nothing was lost.

## What is archived here (all md5-verified against the live original)

| file | what |
|---|---|
| `e068fbbb-a742-45da-8902-b6d3b53151db.jsonl` | full raw transcript, 29.8 MB, 8163 records, Jul 17 → Jul 27 12:08Z, 395 user prompts (md5 `2f6027feed6623d763df63b402294189`) |
| `e068fbbb_FULL_conversation.md` | readable extraction of the whole session (10,661 lines) |
| `e068fbbb_MOE_CAMPAIGN_conversation.md` | readable extraction of just the MoE campaign (Jul 26 10:08Z onward, 1,284 lines) |
| `e068fbbb-subagents/` | the session's subagent transcripts (1.4 MB) |
| `5a7964a4-…jsonl` + `5a7964a4-subagents/` | older container session (Jul 16–19, pre-campaign workstream) — archived because the container copy is wipe-risk |
| `d319ab99-…jsonl` | trivial (an interrupted `/login`), archived for completeness |

A second copy of the main transcript is staged in the HOST project slug so it
is resumable from the host repo checkout:
`/scratch_local/user_data/shutian/kevin/.claude-kevin/projects/-home-kevinni-AsymGEMM-SFT-39-third-party-AsymGEMM/e068fbbb-a742-45da-8902-b6d3b53151db.jsonl`

## How to continue the conversation

From `/home/kevinni/AsymGEMM-SFT-39/third_party/AsymGEMM` (host):

    claude --resume e068fbbb-a742-45da-8902-b6d3b53151db

Notes:
- The resumed context believes the repo lives at `/workspace/...` — same
  files, different prefix; remind the agent it is now on the host path.
- **First kill the orphan** (see below) so two live copies of the same
  session id can't diverge.
- The orphan process holds the only other live copy; it is unreachable
  (socket unlinked), idle at the resting point, and fully backed up — killing
  it loses nothing: `kill 1315307` (left running; user's call).

## Where the MoE integration stands (authoritative: `agent/impls/model_integration.md`)

- **All 6 family modules coded + unit-verified** (max|Δ| ≤ 6.1e-5 vs HF, several
  bit-exact): `mixtral_moe.py`, `phimoe_moe.py`, `hunyuan_moe.py`,
  `glm45_moe.py`, `glm47_moe.py`, `gptoss_moe.py`. lf.py + driver + floors +
  templates fully wired for all six (verified in working tree).
- **Mixtral-8x22B: DONE** — loss PASS, memory PASS+DOMINANCE.
- **Phi-3.5-MoE: DONE** — loss PASS; memory verdict upgraded to **WIN**
  (T3 75.6 vs 149.9 GiB = 49.6%, capacity 128k·b5 where baseline OOMs at b4)
  after the Jul-27 T3 campaign (generic Liger loss bridge for all 6 families +
  fine-grained expert path unlocked for shared-engine families + verdict-gate
  qwen3-hardcode fixed in 3 places).
- **Hunyuan-A13B: 2 open threads** — (1) loss parallel-offset (~5% level
  offset, curves parallel; engine accumulation-order at top-8 shapes;
  **user decision pending** — engine numerics are shared with validated
  families, not to be changed unilaterally); (2) memory verdict was
  BASELINE-WINS under generic T3 but the newly unlocked moefg targets exactly
  its 23.4 GiB packed transient → **re-run recommended**.
- **GLM-4.5-Air, GLM-4.7-Flash, gpt-oss-120b: validation NOT started**
  (wave-1 stopped before GLM per user). Pre-flagged pre-work: add liger
  loss-only mappings for `glm4_moe`/`gpt_oss` (151k/201k vocabs) before their
  pairs; gpt-oss dev smoke must verify the MXFP4→bf16 dequant load; gpt-oss
  has no tuned kernels yet (cuBLAS-on-streamed, T1-class).
- Everything is **deliberately uncommitted** (standing practice: user commits).
  The working tree (6 new modules + 7 modified files) matches the record.
