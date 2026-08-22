# STDTPS96 — c18 lane (Session D): Agent 4 → 1 → 2 → 3 (Kevin's order 2026-08-21)

GH200-96GB simulated TP campaign (agent/impls/s04-p1-dgx-02-c06/standardize_tps_96gb.md).
Models this lane owns first: **Agent 4 = Qwen3.5-122B (weight-offload 2r) + Hunyuan-A13B +
gpt-oss-20B**; then gap-sweeps of 1 (30B), 2 (GLMs), 3 (35B+Mixtral) per the doc's claims.
Status log: `stdtps96_status.log`; run logs `r_s96*.tryN.log`; occupier logs/pids
`stdtps96_occupier_gpu{0,2}.{log,pid}`.

## Setup (2026-08-21 02:4x-03:1x)
- Sim pair = **GPU0 (socket 0) + GPU2 (socket 1)** (`nvidia-smi topo -m`: GPUs 0/1 NUMA0,
  2/3 NUMA1). Occupiers pin both to **97,775 MiB free** (target 97,871 = 95.6 GiB; the
  ~96 MiB delta = the occupier's own context, by design target-free sizing).
- `hbm96_occupy.py` written (in-repo, shared). Renderers `plot_tp_vs_seq{,_2r}_96gb.py`
  created (185G clones, DATA stripped to placeholders, tp96_/tp2r96_ stems, MAIN_RUNGS
  mechanism kept). gpt-oss-20b DEQUANTIZED bf16 fused copy built on c18 scratch (39G,
  `gptoss20_dequant_c18.py`, G20-BUILD rc=0).
- OOM-proof of the budget: hy T1@224k-2r GOOM shows the occupier (88.5 GiB) + trainer
  (86.4 + 13.7 asked) — the allocator enforces ~95.6 GiB exactly as designed.

## Incidents (all fixed in stdtps96_lib.sh; flagged in the doc's LIVE CLAIMS)
1. **Occupier reaped as orphan**: pidfiles lacked trailing newlines → `_occ_pids`' cats
   concatenated into one garbage token → guard whitelisting failed → both occupiers
   killed as ppid-1 orphans. Fix: newline per pid (both the lib reader and the writer).
2. **Expired shared HF token**: env/bashrc.sh + cache/huggingface/token 401 on every hub
   call → hunyuan tokenizer (`additional_chat_templates` listing) FAILs instantly.
   Fix: `HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN=""` (all campaign models public).
3. **Driver same-superchip gate**: `|2` rows reject `--gpus 0,2` → the doc-mandated
   per-socket pair needs `ALLOW_CROSS_SUPERCHIP=1` (pure gate, run_lf_lora_sft.sh:657).
4. **OPEN — fg-tier cross-superchip SIGSEGV**: hunyuan sdp2 T2B/T3 (full-fg recipes)
   segfault on rank 1 (socket-1 GPU) right after the embed/lm_head untie (offload-stage
   prep), while T1 (staged, no fg fabric) runs clean to a genuine budget GOOM on the same
   0,2 pair. No fg-family 2r cell ever ran cross-superchip before (all banked 2r asym
   cells were 0,1). Decisive probe in flight: T2B@64k b2 on 0,2 (the same cell trained at
   59.3 GiB on 0,1 → under-budget; a segfault here proves pair-geometry, not memory).

## Verdicts so far
- hy 2r T1@224k b1: GOOM (96G budget; 185G footprint was 146.4G) — first real 96G wall.

## Incident 4 — RESOLVED-IN-PART, evolving record (03:1x-05:1x)
Chronology of discriminators (all 5-8 min repro cells, tags in stdtps96_status.log):
- fg-tier (T2B/T3) cells segfault in `cpu_left.py:405` (native sm100 cpu-left grouped
  BF16 call) via attn-act-offload dense LoRA-A; T1/staged cells unaffected.
- NOT pair-geometry (fails 1r solo), NOT GPU2 (GPU0 fails), NOT the occupier (fails
  with none), NOT membind, NOT the JIT cache (fresh cache fails), NOT `all`-offload
  (family list fails; `all`+gradofftrue additionally yields trainable=0 — separate
  landmine), NOT gradofftrue (both-false fails identically), NOT git code (path-scoped
  checkout of the Aug-11-good commit 4546c2f fails = env drift, code exonerated; the
  only cpu_left commit in the window is env-gated ablation arms), NOT the venv by
  mtime, NOT the HF snapshot (single Aug-10 revision), NOT hunyuan-specific (qwen-30B
  T2B@32k canary fails), NOT per-GPU state (never-occupied GPU1 fails).
- WORKING at 2026-08-20 21:56 (s1q30sep1024, 2r sepplan fg = cpu-left in-path);
  FIRST failure 2026-08-21 03:19. Node-wide breakage window ~5h across idle time.
- Core-dump attempt produced a 283 GB core that FILLED the shared /home NFS to 100%
  — deleted within minutes (free back to ~142G). DO NOT enable core dumps on these
  address spaces; use live gdb attach if a native frame is ever needed.
- IN FLIGHT: probe14 = the same canary from the -39 tree's checkout/venv on this
  node — discriminates "-46 runtime drifted in place" vs "container/node-wide".

## Incident 5 — c18 CUDA wedge (23:1x-23:2x 08-21) — BLOCKED ON ROOT
After the fg-segfault storm (a dozen segfaulted CUDA processes + kill -9 cleanups),
all 4 GPUs now refuse allocations (context init OK, first cudaMalloc =>
cudaErrorDevicesUnavailable) while idle in nvidia-smi; persistence mode on.
No user-space holder exists (fuser/ps clean). Needs root: nvidia-persistenced
restart -> nvidia-smi --gpu-reset -> reboot. Kevin notified in-session.
RESUME PLAN post-reset: (1) fgprobe17 rerun = LF@ebde34d3 canary (6 min; the
LF-pin test, twice contaminated: stash ate dataset_info.json, then the wedge);
(2) if TRAINED -> pin LF, restart occupiers, relaunch stdtps96_a4_hy.sh; if
FAIL -> run Agent 4 through the -39 stack on c18 (proven working) and file the
-46 diagnosis to Kevin. LF currently pinned at ebde34d3 (dataset_info.json
restored to a valid state; driver re-registers datasets per run).
