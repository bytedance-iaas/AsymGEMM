# fix_dynamic_ep — asym_sepplanlink2 (planned sEP over NVLink)

## §GOAL (Kevin, 2026-08-10 — the DON'T-STOP criterion)
Build asym_sepplanlink2 and KEEP FIXING AND IMPROVING IT until, for EACH
MoE model in scope (GLM-4.7-Flash, GLM-4.5-Air, then Mixtral, then Hunyuan,
then Qwen3-30B / Qwen3.5-35B / Qwen3.5-122B), measured on REAL NEAR-CEILING
2-rank workloads (the §5 ceiling cells — the same cells the throughput
plots bank):
  1. sepplanlink2 is STRICTLY BETTER than the original asym_sepplan2
     (throughput at matched cell/batch; memory not worse than +3%), AND
  2. sepplanlink2 is AT LEAST EQUIVALENT TO — ideally better than — sdp2
     (tok/s >= sdp2 − 2% with loss parity; the sdp2-floor contract must
     hold empirically, not just by design).
The agent CANNOT stop until both hold for every in-scope MoE on its ceiling
cells. If a model resists (e.g., all-declined + overhead), that is not a
stopping point — it is a fix-and-retest loop: tune MAX_MPE, ring sizing,
transport details, hook placement, whatever the evidence demands, until the
criterion is met (worst legitimate outcome per the floor contract: armed=0
AND tok/s == sdp2 within noise AND better-than-sepplan2 — which satisfies
1+2 by making the new backend the safe default). Every fix goes through the
§5 comparison discipline (vs sdp2 AND sepplan2, tok/s+HBM+RSS+loss).

STRICT MODEL SEQUENCE (hard gate — no skipping, no parallelizing ahead):
  1. GLM-4.7-Flash and GLM-4.5-Air   (the GLM pair = stage 1)
  2. Mixtral-8x22B                    (only after BOTH GLMs fully meet §GOAL)
  3. Hunyuan-A13B                     (only after Mixtral fully meets §GOAL)
  4. Qwen3-30B, Qwen3.5-35B, Qwen3.5-122B (only after Hunyuan; regression +
     upside on the paper's existing sEP rows)
DO NOT move to the next model until the CURRENT model's §GOAL criteria are
FULLY met on its near-ceiling cells and logged in §8 with the comparison
table (sepplanlink2 vs sepplan2 vs sdp2: tok/s, peak HBM, peak RSS, loss,
armed/declined). A blocked model blocks the whole ladder — fix it there.

MISSION (Kevin, 2026-08-10): the natural version of dynamic EP — same planned
union split + same steal-only-when-profitable gate as `asym_sepplan2`, minus
the two measured artifacts that make sEP lose on fat-expert models:
(1) stolen payloads ship over HOST staging and contend with weight streaming;
(2) the hook exists only in the DIRECT GEMM path while every tier recipe pins
STAGED dispatch (−11% entry fee measured on GLM-Flash).
New backend: **asym_sepplanlink2_cpuadamwds** (sep + plan + link).
DO NOT START until Kevin says go. Build on c14, dev on SMALL cells, validate
on NEAR-CEILING cells (§5). Jamba is OUT of scope.

## §0 RESOLVED FACTS (measured/verified 2026-08-10 — no open questions)
- GPU0<->GPU1 on c14: **NV18** (18 NVLink links, `nvidia-smi topo -m`);
  warmed peer copy_ = **718 GB/s** (first touch pays ~
  mapping cost — warm up in install). `torch.cuda.can_device_access_peer`
  True both ways.
- Arm-rate diagnosis (run_glms.md §Log 2026-08-09): sepplan2 with the hook
  live declines 100% on Air (0/1755) and Flash (0/1518); direct dispatch
  itself −11% on Flash (3049 vs 3414 staged). sdp2-floor parity ±0.9%.
- The ONLY residual technical unknown — whether the ep_steal TMA kernel
  accepts peer-HBM pointers unchanged — is closed by dev step D0 below
  (existing standalone probe, both outcomes have a specified design).

## §1 DESIGN INVARIANTS (unchanged from sepplan2)
- Plan-mode split: union counts -> deterministic contiguous cut at a segment
  boundary (`ep_sep.py` `_SepState.try_armed`, `mode == "plan"` branch,
  incl. the n_own hysteresis snap). UNTOUCHED.
- Arming gate: `pre_gate` host-int decline (zero-cost, publishes flag=2) +
  `try_armed` decline. UNTOUCHED except `_MAX_MPE` becomes
  transport-dependent (§3.3e).
- Collective-free control: host-pinned `ctrl` flags, host spins, PR-5
  visibility receipt (event.synchronize -> host flag write). UNTOUCHED —
  the same receipt covers P2P writes (device sync => system-wide
  visibility of that device's global-memory writes, incl. peer HBM).
- LoRA/grad semantics: stealing covers only the frozen grouped GEMM. UNTOUCHED.

## §2 WHAT CHANGES (two things only)
A. TRANSPORT: X/D payload slots move host-pinned -> GPU-HBM rings with CUDA
   peer access; stealer peer-READS the owner's X ring, peer-WRITES results
   into the owner's D ring. ctrl stays host-pinned.
B. HOOK PLACEMENT: the sEP attempt moves up into `_dispatch_grouped_nt` so
   it fires BEFORE the staged flip — all tiers (T1/T2/T2B/T3) see it under
   their recipe-pinned dispatch, no env override, no entry fee.

## §3 CHANGE LIST — exact files / functions / classes
### 3.1 `asym_gemm/training/frozen_linear.py`
a. EXTRACT the sEP attempt from `_asym_grouped_bf16_nt` (the block starting
   `if _EP_SEP_ENABLED:` at ~line 1004: pre_gate -> pairs/ids `.tolist()` ->
   segs build -> `try_armed`) into a new module-level helper:
     `_try_ep_sep_grouped(a, b_cpu, offsets, experts, *, compiled_dims,
                          transpose_b, output_dtype) -> torch.Tensor | None`
   The helper OWNS the pad+metadata prep it needs: it calls
   `_pad_grouped_input_for_asym` + `_group_metadata_tensors` itself and, on
   armed success, `_unpad_grouped_output` before returning d. Returns None
   on any decline. It MUST first check `_direct_grouped_bf16_reason(...) is
   None` (the armed kernel is a direct-family kernel) and `_EP_SEP_ENABLED`.
b. `_asym_grouped_bf16_nt`: replace its inline block with a call to the
   helper (behavior byte-identical; it already runs post-pad, so guard the
   helper against double-padding by passing a flag or splitting the helper
   into prep+attempt — implementer's choice, but the direct path must not
   pad twice).
c. `_dispatch_grouped_nt` (line 1601): insert BEFORE the staged flip
   (`if backend != "torch" and precision == "bf16" and _gemm_dispatch_staged()`,
   line 1619):
     if backend != "torch" and precision == "bf16" and _EP_SEP_ENABLED:
         out = _try_ep_sep_grouped(...)
         if out is not None: (count stats.asym_* per phase) ; return out
   Decline => fall through to the existing staged/direct logic UNCHANGED.
   NOTE: with this, the direct path's internal attempt (b) becomes
   redundant-but-harmless (seq consumption happens once per launch either
   way because the dispatch-level attempt returns before reaching
   `_asym_grouped_bf16_nt`); keep (b) for the non-staged path only if the
   dispatch-level attempt is gated to `_gemm_dispatch_staged()` — DECISION:
   simplest correct form = attempt ONCE at `_dispatch_grouped_nt` for BOTH
   dispatch modes and DELETE the inline attempt from
   `_asym_grouped_bf16_nt`. One call site, one seq consumed, no drift.
### 3.2 `asym_gemm/training/ep_sep.py`
a. `install_buffers(*, rank, world, ctrl, x_slots, d_slots)` gains
   `transport: str = "host"`; store on `_SepState` (`self.transport`).
b. `_SepState.try_armed`: the ONLY transport-sensitive lines are
   - X stage: `x_slot[: m*k].view(m,k).copy_(a, non_blocking=True)` on
     `self.side_stream` — IDENTICAL for HBM slots (D2D local copy, faster);
     keep the event-sync + host flag publish as-is.
   - `peer_x = self.x_slots[peer][ring]...` / `d_peer = d_stage_mine...` —
     with transport="nvlink" these are peer-HBM views; pointer plumbing
     into the kernel is unchanged (unified addressing). NO CODE CHANGE if
     D0 passes; see D0-fallback otherwise.
   - `ep_steal_spin_gather(d, self.d_slots[peer][ring]...)`: owner reads
     its OWN d-ring under the ownership convention below.
   OWNERSHIP CONVENTION (nvlink): d_slots[r] lives ON rank r's GPU and is
   the ring where THE PEER writes rank r's stolen rows; x_slots[r] lives on
   rank r's GPU holding rank r's own X. (Host transport keeps today's
   meaning: d_slots[me] = where I write peer rows. The gather call site
   flips accordingly — implement as a small
   `self._d_ring_for_gather()` / `self._d_ring_for_kernel()` pair on
   `_SepState` so both transports read one code path.)
c. `state()` unchanged; `_dump_stats` prints transport.
d. Module constants: `_MAX_MPE` -> resolved in `install_buffers` from
   `ASYM_EP_SEP_MAX_MPE` if set, else per-transport default:
   `_MAX_MPE_DEFAULTS = {"host": 4096, "nvlink": <D1 microbench>}`.
e. Slot dtype/shape identical (bf16, rows*kmax); RING/FLAG_SLOTS/MAX_SEGS
   unchanged.
### 3.3 `scripts/lf/run_lf_profiled_train.py` — fabric-seal install block
   (the `fabric = get_fabric()` block, currently ~lines 1888-1932, guarded
   by `os.environ.get("ASYM_EP_SEP") == "1"`):
a. Read `transport = os.environ.get("ASYM_EP_SEP_TRANSPORT", "host")`.
b. transport=="host": today's fabric `get_or_create` slots — UNCHANGED.
c. transport=="nvlink": allocate per-rank rings in DEVICE memory:
   `sep_xs[r][i] = torch.zeros(rows*kmax, dtype=bf16, device=f"cuda:{local}")`
   for MY rank only + enable peer access
   (`torch.cuda.set_device` + a warmed 1-copy handshake per direction to
   pay the mapping cost); exchange ring POINTERS via two tiny fabric
   get_or_create int64 blocks (device ptr + size published through the
   sealed fabric, read by the peer; wrap with
   `torch.cuda.caching_allocator_enable... `NO — simplest exact mechanism:
   allocate via `torch.zeros(..., device=...)`, publish
   `t.data_ptr()`+numel through the fabric ctrl block, peer wraps with
   `torch.as_tensor` NO — peer CANNOT wrap a raw remote ptr via torch API;
   the kernel takes raw POINTERS anyway (`_C` extension), so publish ptrs
   in the ctrl block and pass ints to the kernel — `try_armed` under
   nvlink passes peer ptr ints (new kernel-arg path, see D0) OR uses CUDA
   IPC handles (`torch.multiprocessing.reductions` /
   `cudaIpcGetMemHandle` via `torch.cuda` IPC: `tensor._share_cuda_()` ->
   peer `torch.cuda.CUDAStorage._new_shared_cuda(...)`) to get a REAL peer
   tensor. DECISION (no ambiguity): use CUDA IPC share — each rank
   `_share_cuda_()`s its rings, publishes the handles through a fabric
   bytes block, peer reconstructs tensors once at install; thereafter
   try_armed code is transport-blind (views of real tensors). IPC across
   the two ranks of one torchrun on one node is supported and NVLink-fast.
d. ctrl block: unchanged (host-pinned fabric).
e. `heartbeat.emit("ep_sep_installed", transport=..., slot_rows=...)`.
### 3.4 `scripts/lf/run_lf_lora_sft.sh`
a. Line ~477 backend case list: add `asym_sepplanlink2_cpuadamwds`.
b. Lines ~486-493 comment table: add the one-liner.
c. Line ~524 case block: new arm
   `asym_sepplanlink2_cpuadamwds)` -> exports `ASYM_EP_SEP=1`,
   `ASYM_EP_SEP_MODE=plan`, `ASYM_EP_SEP_TRANSPORT=nvlink` + echo line.
d. Lines ~1098 and ~1284 (2-rank backend alternations): add the token.
### 3.5 `scripts/lf/profile_lora_lf_test_source.sh`
a. Backend normalization (lines ~1356-1357 pattern): add
   `asym_sepplanlink2) backend=asym_sepplanlink2_cpuadamwds ;;` +
   `asym_sepplanlink2_cpuadamwds)` identity arm.
b. Grep for every alternation naming `asym_sepplan2` and mirror the new
   token (implementer runs
   `grep -n 'sepplan2' scripts/lf/*.sh` and covers ALL hits).
### 3.6 Dev probes (exist already — extend, don't create)
- `scripts/testing/ep_sep_probe.py`: standalone protocol validator with
  injectable buffers. Add `--transport nvlink` flag exercising the IPC
  ring path + peer pointers into the ep_steal kernel.
- `scripts/testing/shared_fabric_probe.py`: untouched (ctrl path).
### 3.7 Docs
- §8 log here (append-only), run_glms.md pointer entry, tier_recipes.sh NOT
  touched (recipes stay staged — that is the point of hook change B).

## §4 DEV STEPS (small workloads; c14; strict order)
D0 KERNEL/PEER PROBE (closes the last unknown): `ep_sep_probe.py
   --transport nvlink` — synthetic segs, both ranks, verify
   `m_grouped_bf16_asym_gemm_nt_contiguous_ep_steal` reads peer-HBM X and
   writes peer-HBM D correctly (bitwise vs the host-transport run).
   IF the TMA path faults on peer pointers: FALLBACK DESIGN (pre-approved):
   keep kernel I/O local — stealer first peer-copies the owner's X slice
   into a local scratch ring (side stream, NVLink 718 GB/s), runs the
   existing kernel purely locally, then peer-copies D back and publishes
   done. Costs one extra hop each way but stays >=20x faster than host
   staging; ownership convention in 3.2b already accommodates it.
D1 BREAK-EVEN MICROBENCH: ep_balance-style sweep on ONE model cell
   (glm4.7-flash 2r 64k b2, T2) with MAX_MPE forced high — measure armed
   step time vs declined across segment sizes; set `_MAX_MPE_DEFAULTS
   ["nvlink"]` at the crossover. Bank the number + receipts in ep_sep.py.
D2 SMALL-CELL SANITY (dev workloads): flash 2r 32k b2 T1 and T2 —
   loss parity vs sdp2 (Δ<=0.02), un-armed overhead <=2%, arm counter >0
   at forced-low MAX_MPE (mechanism live), no step-time inflation with
   MAX_MPE=0 (stagger check).

## §5 VALIDATION LADDER (near-ceiling cells; each tier gates the next;
   compare sepplanlink2 vs BOTH banked sdp2 AND a fresh sepplan2 twin on
   tok/s + peak HBM + peak RSS + loss; /dev/shm/asym_fabric_* cleaned
   before EVERY cell; arena caps per model; floors per model table)
V1 GLMs:
   - glm4.7-flash: dev 64k b2 (T2) then CEILING 1.02M b1 (T2 — banked
     asym 294 global) and 192k b4 (T1 97%-HBM cell, banked 1526-class).
   - glm4.5-air (arena 400 for T2-class, 240 for T1): dev 64k b2, then
     CEILING 320k b1 (T1 — banked 989, 98% HBM).
   PASS = every cell: loss parity, tok/s >= max(sdp2, sepplan2) - 2%, and
   HBM/RSS within +3% of sdp2 (the rings must not sink the edge cells);
   plus armed>0 somewhere OR an explicit all-declined verdict (which is a
   legitimate finding: bank nothing, stop the ladder for that model).
V2 Mixtral (only if V1 passes; arena 285): dev 64k b1, CEILING 304k b1
   (T1 — banked 1110).
V3 Hunyuan (only if V2 passes; arena per its row): dev 64k b2, CEILING
   320k b1 (T2B — banked 1375).
V4 Qwen3/3.5 (regression + upside; their sEP rows are the paper's):
   - q3-30b-a3b CEILING 1.04M (T2 — banked 901)
   - q3.5-35b-a3b CEILING 896k (T2 — banked 2640)
   - q3.5-122b-a10b CEILING 336k (T1 — banked 1498; arena 400)
   PASS = >= banked sepplan2-era values - 2% (no regression) ; record wins.
V5 Re-ladder + rebank ONLY models with measured wins (standing banking
   rules; figures-only Overleaf push; asym-side machinery = fair per the
   house A/B rules).

## §6 RISKS (each with its mitigation wired above)
- Stagger reintroduction: one-sided IPC tensors + host flags only (no
  NCCL anywhere in the steal path); V-gate stagger checks (D2, V1).
- HBM ring cost at 90%+-HBM edge cells: rings sized by
  ASYM_EP_SEP_SLOT_ROWS (right-sized default), allocated ONLY under the
  new backend token; V1 explicitly checks HBM within +3%.
- IPC lifetime: rings are allocated once at install and never freed until
  exit (matches fabric slot lifetime); no re-share per step.
- X-slot stability race: same ring/seq discipline as today (RING slots,
  FLAG_SLOTS hygiene) — unchanged code path.
- First-touch P2P cost (~30 ms): paid once in install warm-up handshake.

## §7 RUNBOOK (self-contained ops — how to actually run every cell)
- MACHINE: c14 only (this session's machine). NEVER run on the host: every
  command goes through the enroot container `asym_sft_42` with the 39 tree
  mounted at /workspace/AsymGEMM-SFT-39 (one-shot runner pattern: enroot
  start --rw --root with the workspace + /scratch_local cache mounts; see
  agent/anchors_tmp/*.sh chains for working examples).
- ONE CELL =
    export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
    export NUMACTL_ENABLE=1 NUMACTL_MODE=membind NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1
    export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512
    export CUDA_VISIBLE_DEVICES=0,1 GPU_POOL=0,1 DDP_TIMEOUT=1500
    rm -f /dev/shm/asym_fabric_*          # MANDATORY before EVERY 2r cell
    RUN_NAME=<tag> RUNS="<modelkey>|2 ; <backend>|<TIER-or-token>|ligerloss1 ; <seq>|<b>|1 ; none|false|false|false|false|false" \
      bash scripts/lf/profile_lora_lf_test_source.sh
  model keys: glm4.7-flash, glm4.5-air, mixtral-8x22b, hunyuan-a13b,
  q3-30b-a3b, q3.5-35b-a3b, q3.5-122b-a10b. Backends under test:
  asym_sepplanlink2_cpuadamwds (new) · asym_sepplan2_cpuadamwds (original
  twin, run AS-IS, no overrides) · asym_sdp2_cpuadamwds (floor reference —
  banked values already exist; rerun only if the banked comment lacks the
  needed batch).
- PER-MODEL ENV: ASYM_ARENA_SHM_CAP_GB — air T1 240 / air T2-class 400 /
  mixtral 285 / q3.5-122b 400 / others default. Host watchdog floors come
  from WATCHDOG_FLOOR_GB_BY_MODEL automatically.
- METRICS per cell from the run dir
  (profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16/
   <RUN_NAME>__b<b>_s<seq>_ga1_drop000/**/):
  step_samples.json -> eff tok/s = 2*seq*b/mean(non-warmup step_milliseconds/1000)
  (2r = global), peak_reserved_hbm_bytes, process_rss_peak_bytes, loss;
  train.log -> `[ep_sep] ... exit stats {armed, declined, spin_wait_s}`.
- DISCIPLINE: strictly SOLO+serial on the node; fresh tag per attempt;
  never edit a script while a run is live; kill by exact PID only
  (bracket-pattern pgrep like '[s]epplanlink'); guards must count only
  /proc-live pids (ghost nvidia-smi entries are a known trap).
- BANKING (only after §GOAL is met per model): figure DATA dicts live in
  scripts/figures/plot_tp_vs_seq_2r.py (hardlinked across trees — bank
  ONCE); house banking rules per run_glms.md §5; Overleaf push is
  figures-only via the [MLSys 26 Sub] repo; verify origin/main after push.

## §8 PROGRESS LOG (append-only)
- [2026-08-10] Doc v2: all ambiguities resolved by code-read + node
  measurements (NV18 topology, 718 GB/s warmed peer bw; hook site =
  `_dispatch_grouped_nt` before the staged flip; transport surface =
  x/d slot provisioning + gather-ring convention only; IPC-share chosen
  as the peer-tensor mechanism; D0 probe closes the TMA question with a
  pre-approved fallback). Validation ladder per Kevin: GLMs -> Mixtral ->
  Hunyuan -> Qwen3/3.5, dev-small + near-ceiling cells, vs sdp2 AND
  sepplan2 on tok/s+HBM+RSS+loss. BUILD NOT STARTED (awaiting go).
