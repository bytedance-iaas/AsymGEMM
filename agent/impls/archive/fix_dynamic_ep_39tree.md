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
DO NOT START until Kevin says go. Run on YOUR OWN machine + repo tree (never
another agent's), dev on SMALL cells, validate
on NEAR-CEILING cells (§5). Jamba is OUT of scope.

## §0 RESOLVED FACTS (measured/verified 2026-08-10 — no open questions)
- GPU pair topology (measured on a GB200 board 2026-08-10; RE-VERIFY on
  your machine as runbook step R0): **NV18** (18 NVLink links,
  `nvidia-smi topo -m`); warmed peer copy_ = **718 GB/s** (first touch pays
  a ~30 ms mapping cost — warm up in install).
  `torch.cuda.can_device_access_peer(0,1)` True both ways. If YOUR pair is
  not NVLink-connected, stop and surface to Kevin — the design premise
  fails without it.
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

## §4 DEV STEPS (small workloads; your machine; strict order)
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
- MACHINE: your own machine with an NVLink-connected 2-GPU pair (R0: run
  the §0 topology + peer-bandwidth check first). NEVER run on the host:
  every command goes through your machine's asym enroot container with
  YOUR repo tree mounted at /workspace/<your-tree>/third_party/AsymGEMM
  (one-shot runner pattern: enroot start --rw --root with the workspace +
  /scratch_local cache mounts; see agent/anchors_tmp/*.sh chains for
  working examples). One machine = one repo tree; coordinate cross-agent
  claims through this doc's §8 ledger.
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
- [2026-08-10] D0 CLOSED — nvlink transport BITWISE-CORRECT (plan mode
  PR5_PASS on the 2-proc probe). Path taken: the relax-the-assert route
  IMA'd (ep_steal TMA faults on legacy-IPC REMOTE mappings), but the D0b
  operand probe (scripts/testing/ep_steal_operand_probe.py) showed all-LOCAL
  device operands are bitwise-clean -> shipped the pre-approved copy-based
  dataflow: peer-X --NVLink copy--> local scratch before launch; local
  d-scratch --NVLink copy--> owner's ring before the done flag (single
  writer per ring per seq; visibility receipts unchanged). csrc asserts
  relaxed at gemm.hpp:722 + ep_steal_sync.cu:116 (pinned-or-cuda), _C
  rebuilt. Host-transport regression PR5_PASS pre-change; re-verify in
  D0T5. OPS: an IMA'd probe leaves hung children HOLDING the GPUs — pkill
  '[e]p_sep_probe' + clean /dev/shm/asym_seprobe_* before ANY rerun.
- [2026-08-10] D2 CLOSED — e2e sepplanlink2 in real training (flash 2r 64k
  b2 T2): hook fires under recipe staged dispatch (552 gates), floor contract
  holds when declining (3426 vs sdp2 3412, +0.4%, loss parity), and with
  arming-regime rings (SLOT_ROWS=655360) EVERY launch arms planned-mode with
  loss dead-parity (1.4662; band 1.4661-1.4684) at 3356 tok/s (-1.6% vs
  floor on a BALANCED cell = protocol overhead only; the win case is skew —
  D1's job). Latent NameError (module import in the extracted helper) found
  by the first armed run and fixed. Ring math: flash fg launches carry
  ~512k rows -> default SLOT_ROWS 163840 capacity-declines everything;
  per-cell SLOT_ROWS is the knob (big on arming cells, floor-tiny on
  ceiling cells to hold the +3% HBM criterion; armed rings @655360 cost
  ~+25 GiB). D1 next: armed-vs-floor across skew regimes to set
  _MAX_MPE_DEFAULTS["nvlink"] at the real crossover.
- [2026-08-10] D1 CLOSED — arming economics on flash (64k b2 T2, all cells
  loss-parity): snap-armed −2.1/−2.5% vs sdp2 at z1.5/z2.0 (skew never
  creates row imbalance under DP — m is tokens*topk, identical per rank);
  no-snap bank-once −1.9/−1.7% (flash banks 0.14GB/layer = nothing to
  consolidate). Floor mode measured +0.4% vs sdp2. POLICY: nvlink MAX_MPE
  default = 4096 (decline fat launches; floor is the win on tiny-bank
  models); bank-once remains the per-cell hypothesis for WIDE-bank models
  (Air 4.4GB/layer, Mixtral 2.8GB — their V-stages probe NO_SNAP+high-MPE
  explicitly). Rings: pure-floor cells run SLOT_ROWS tiny to hold the HBM
  criterion. OPS: skew rows land in profiling_both_skew/ root.
- [2026-08-10] MICROBENCH (probe, armed plan, bitwise-verified): nvlink
  transport vs host — skew-3:1 (m=442k) 18.3 vs 61.1 ms = 3.3x; balanced
  warm 5.6 vs 11.4 ms = 2.0x; decline path 0.09 vs 0.07 ms (both ~free);
  one-time first-armed cost ~86 ms (IPC map; consider a full armed warm
  round at install). CONFIRMS the layering: transport is strictly better;
  the flash e2e -2% is model-regime (DP-equal rows + 0.14GB banks = nothing
  to win there; floor +0.4% is the correct behavior). E2E wins expected
  where launch-level imbalance (qwen chunked fg) or fat banks (Air/Mixtral)
  exist.
- [2026-08-10] §GOAL AMENDMENT (Kevin): a model with NO physically winnable
  e2e cell (Flash class: DP-equal rows + tiny banks) passes on
  floor-tie e2e (>= sdp2 - 2%, loss parity, HBM criterion at ceilings) +
  the armed-path microbench superiority (3.3x) — "strictly better than
  sepplan2" is required only where the mechanism can engage. Move on to
  the next model in that case. V1-Flash ceiling trio still runs (the floor
  contract must hold at 97-98%-HBM edge cells — ring cost check).
- [2026-08-11] V1 RESUME (session kill at ~04:02 took down v1_guarded.sh
  mid-f_sdp960; f_sp2960 + air block never ran). Banked so far (guarded
  chain, c17): f_spl960 316 tok/s vs campaign-fresh sepplan2 twin
  spgf960=317 (-0.3%) vs banked sdp2 313 (+1.0%), loss 6.4125 vs 6.4153
  (parity), resv 180.9G vs 181.5G, rss 469G vs 479G — 960k b1 T2 PASSES
  both criteria (fresh same-node sdp2 control still queued for the table).
  FLASH 192k b4 T1 FAILED -6.3% (f_spl192r 1417 vs f_sdp192r 1512):
  per-phase split shows fwd IDENTICAL, backward +6-11% — the DEFAULT
  163840x2048 device rings (RING=4, x+d = ~5.1 GiB/rank) squeeze the
  allocator at the 97%-HBM cell exactly as the D1 tiny-ring policy
  predicted; the chain forgot ASYM_EP_SEP_SLOT_ROWS on ceiling cells.
  ALSO: armed=0 declined=0 with 4212 launches through the hook site on T1
  — launches bail SILENTLY at _direct_grouped_bf16_reason before pre_gate
  (invisible in exit stats). PATCH: _SepState.count_skip(reason) +
  frozen_linear counts skip_<reason> in exit stats (no seq consumed, both
  ranks bail deterministically — protocol-safe). RESUMED as v1c_resume.sh:
  192k T1 retry w/ SLOT_ROWS=8192, air dev trio, f_sdp960r + f_sp2960,
  bank-once probe pair, air 320k trio (spl tiny rings), f_spl1024 crown
  (twins exist: campaign sp2@1024k=297, banked sdp2=294). Air env applied
  PER-CELL (old chain-level export would leak arena/kmax into flash cells).
- [2026-08-11] TINY-RING FIX CONFIRMED + T1 HOOK-DEATH IS STRUCTURAL + AIR
  DEV TRIO. (a) f_spl192t (SLOT_ROWS=8192): 1487 vs fresh f_sdp192r 1512 =
  -1.7% (was -6.3% default rings), loss 1.26446 vs 1.26408, resv 181.1 vs
  182.1 — 192k T1 now inside the floor criterion vs the fresh same-node
  control. (b) count_skip patch verified live (repo import confirmed in
  .venv-fa4) yet f_spl192t exit stats all-zero with 4212 launches counted
  torch_* — T1 ("unsloth" token, no fg envs) routes expert GEMMs so the
  hook site condition never fires; §2 change B's "all tiers see it" is
  MEASURED-FALSE for T1 on GLM (T2 fg-path launches do gate: 3036@960k).
  No mid-chain code edits — all v1c cells run one consistent tree.
  (c) AIR 64k dev trio (T1, KMAX=4096): sdp2 3951/97.9G/1.4409, sp2
  3990/97.9G/1.4512, spl 3871/110.4G/1.4393 -> spl FAILS tok/s (-3.0% vs
  sp2) AND HBM (+12.8% vs sdp2). IDLE DEVICE RINGS (10.7 GiB/rank at
  kmax4096) cost ~2-3% tok/s + full ring size in resv even at a 60%-HBM
  cell, while sp2's equally-sized HOST slots cost ~nothing. POLICY: any
  hook-dead cell (= every T1 cell) runs SLOT_ROWS=8192 always; rings are
  unusable there. v1c's a_spl320 already complies; a_bo16 (T1) is moot as
  a win probe (hook dead — its NO_SNAP/MPE knobs can't engage) and will
  only measure idle-BIG-ring cost; the real wide-bank probe moves to an
  air T2-class trio (arena 400) queued as v1d_air_fix.sh alongside the
  a_spl64t tiny-ring dev retry. Also noting sp2's loss wobble (+0.010 vs
  sdp/spl agreeing at 1.440) — inside the 0.02 band; watch at 320k.
- [2026-08-11 ~11:45] KEVIN DIRECTIVE: "stop flash — only do the important
  ones first." Killed f_sp2960 ~40min in (twin coverage stands on campaign
  spgf960=317 fresh same-node); DROPPED f_spl1024 crown + a_bo16/a_sdp16
  (T1 hook-dead makes the bo16 probe moot). FLASH 960k FINAL with the fresh
  control: sdp2r 318/181.5G/464G/6.4085 vs spl 316/180.9G/469G/6.4125 =
  -0.6%, PASS (fresh sdp2 +1.6% above banked 313 — same-node discipline
  vindicated). New chain v1e_priority.sh, importance-first: (1) air
  T2-class armed-probe trio 64k b1 arena400 (hook ALIVE on T2; spl rings
  ROWS=327680 covers ~256k+pad fg rows, NO_SNAP+MPE=inf = the wide-bank
  win test), (2) a_spl64t tiny-ring dev retry, (3) air 320k ceiling trio,
  (4) mixtral dev entry: T1 floor trio (spl tiny rings) + T2 armed-probe
  trio (KMAX=6144 covers gate/up k=6144; down k=16384 capacity-declines by
  design; ROWS=131072). Mixtral 304k ceiling trio queues after air data.
  V2 overlap is Kevin-directed (dev-entry cells only; V2 verdict still
  gated on V1 air completion).
- [2026-08-11 ~16:45] SESSION HANDOFF RECONCILE + v1f_delta. Found on
  arrival: v1e_priority.sh began 11:49 but its first cell (a_splt2) was
  SIGINT'd 4min in and the chain died silently (v1e_status.log has no
  CELL-END); the afternoon v1_guarded rerun (14:15-16:26, V1G-DONE)
  re-banked the air 320k twins (sp2 980 / sdp 981, HBM 179.2) and 16k pair
  (sdp 2272/42.5G; bo16 2211/62.5G — T1 hook-dead, +20G = idle big rings),
  and a_spl320 big-ring OOM'd AGAIN identically (19.53 GiB alloc, 19.25
  free: that alloc is the model's own whole-layer packed expert tensor, so
  the +10.7G idle rings are what starve it — tiny-ring retry is the fix and
  has still never run). DIAGNOSIS BUILD: added gate_* skip counters at the
  dispatch call site in frozen_linear.py (counts WHY the hook gate rejects:
  gate_backend_torch / gate_precision / gate_quantized / gate_dense, into
  the same stats dict) — behavior unchanged; every cell from here runs it.
  An orphan cell af_spl64t (= v1e item 2: air T1 64k b2 SLOT_ROWS=8192) is
  in flight and doubles as the T1 gate-diagnosis read. LAUNCHED
  v1f_delta.sh = v1e minus banked/in-flight cells: (1) air T2-class trio
  a_splt2/a_sp2t2/a_sdpt2 (arena 400; spl ROWS=327680 KMAX=4096 NO_SNAP
  MPE=inf), (2) a_spl320t tiny-ring ceiling retry, (3) mixtral dev entry
  m_spl64/m_sp264/m_sdp64 (T1 floor, spl ROWS=8192) + m_splt2/m_sp2t2/
  m_sdpt2 (T2 probe, KMAX=6144 ROWS=131072; down-proj k=16384
  capacity-declines by design). Air §GOAL needs: T1 floor-ties (64k b2 +
  320k tiny-ring) AND the T2 trio verdict; mixtral cells are dev-entry
  only (V2 verdict still gated on air completion).
- [2026-08-11 ~17:05] T1 HOOK-DEAD ROOT CAUSE FOUND + 64k FLOOR NOW PASSES.
  (a) af_spl64t (air T1 64k b2, SLOT_ROWS=8192, diagnosis build): 4003
  tok/s / HBM 93.3 / loss 1.4496 vs sp2 3990/97.9 and sdp 3951/97.9 ->
  spl is now fastest AND lowest-HBM at the 64k dev cell; the V1 -2%/+12.5G
  row was entirely idle-big-ring cost. AIR 64k DEV ROW: PASS.
  (b) Gate counters answered WHY T1 never engages: skip_gate_dense=810 —
  every whole-layer (T1-class) grouped launch carries dense_experts=True
  (one group per expert, identity order, token-grouped rows; verified in
  _grouped_torch_mm + the qwen3/glm45 whole-layer path). The hook's call
  gate excludes dense_experts — an inherited conservatism, not a layout
  problem. Added ASYM_EP_SEP_ALLOW_DENSE=1 (default off, no behavior
  change) to let the hook evaluate dense launches.
  (c) DEEPER: even with dense allowed + NO_SNAP, plan-mode bank-once CANNOT
  fire under equal-DP. The union is [rank0 section | rank1 section] and
  kernel+gather infer ownership POSITIONALLY from the single n_own
  boundary; the row-balanced cut always lands exactly at that boundary
  (per-rank m is identical under equal-DP), so every launch is fully
  local no matter what. Assigning each expert's bank to ONE rank (the
  actual bank-once win) requires per-seg ownership in the kernel contract
  (csrc ep_steal + spin_gather signature change) — parked as future work;
  NOT required for the air §GOAL (amended pass route: floor-ties + micro
  superiority; full pass route: the T2-class trio where fg launches are
  non-dense and the hook is alive).
  (d) v1f_delta relaunched after closing a guard race (model-load minutes
  show zero GPU procs — guard now also waits on launcher processes).
  In flight: air T2 trio -> a_spl320t -> mixtral dev entry (6 cells).
- [2026-08-11 ~17:25] AIR T2-CLASS TRIO DONE (64k b1, arena 400). spl
  forced-arm (NO_SNAP, MPE=inf, ROWS=327680): 2960 tok/s / HBM 80.8 /
  loss 1.5069, **armed 540/540, declined 0, planned 540 — first armed
  launches ever on a GLM** (skip_gate_dense=945 = the whole-layer dx
  calls; fg forward launches all armed). sp2 (defaults): 3189 / 55.8 /
  1.5195 with 0/540 armed (declines are free). sdp: 3205 / 55.8 / 1.5198.
  VERDICT: hook alive + mechanism correct on T2; forced arming costs -7.6%
  vs sdp on BALANCED data exactly as the D1 economics predict (equal-DP =>
  per-rank rows identical => the plan cut has no imbalance to harvest =>
  transport is pure cost; +25G = the probe's big rings). A default-knob
  spl declines everything and ties sp2. The win channel on T2 needs
  launch-level imbalance (skewed data x fg chunking) — not part of air's
  §GOAL route (air ceiling tier is T1). Caveat: the spl number is from the
  16:49 pipeline whose launcher I killed but whose setsid'd children ran
  to completion overlapping af_spl64t's plot phase — if a razor verdict
  ever hangs on it, rerun clean; the -7.6% direction is beyond that noise.
  NOTE the spl cell's own artifacts landed under a jobs.tsv row "skipped"
  (v1f#2 found them already written) — numbers harvested from
  step_samples.json directly. Air remaining: a_spl320t (RUNNING).
- [2026-08-11 ~18:10] **AIR §GOAL: PASS (amended route).** a_spl320t (320k
  T1 ceiling, SLOT_ROWS=8192): 980 tok/s / HBM 179.8 / loss 6.1792, job
  ok — exact floor-tie with sp2 980/179.2/6.1516 and sdp 981/179.2/6.1739;
  the tiny-ring policy eliminated the twice-fatal 19.53-GiB OOM. Air
  criteria closed: 64k dev spl BEST on both axes (4003/93.3), 320k ceiling
  three-way tie, T2 hook-alive with 540/540 armed + cost quantified
  (-7.6% forced-arm on balanced data), microbench 3.3x at 3:1 imbalance.
  No winnable e2e cell exists on air under equal-DP (structural — dense
  whole-layer T1 + positional-ownership union; balanced T2 has no
  imbalance to harvest), so the amendment applies as at flash.
  SEQUENCE GATE: air complete -> V2 MIXTRAL entered. The v1f mixtral
  6-pack insta-failed: the fused local copy path in the model map exists
  only on the other campaign machine (HFValidationError root cause, now
  understood). Repointed [mixtral-8x22b] at mistralai/Mixtral-8x22B-v0.1
  (hub cache verified complete: 59/59 shards, 262G) — pays the load-time
  expert-fusion conversion transient; throughput semantics unchanged.
  Relaunched as v2m_mixtral.sh (same 6 cells).
- [2026-08-11 ~19:00] V2 MIXTRAL DEV: T1 trio DONE, T2 trio on third cap
  try. T1 64k b1 (arena 285, spl SLOT_ROWS=8192 KMAX=6144): spl 3416/47.6/
  0.7811, sp2 3417/46.7/0.7819, sdp 3383/46.7/0.7807 -> spl TIES sp2 and
  edges sdp +1.0%; hook dense-dead on T1 exactly as on GLM
  (skip_gate_dense=1008; whole-layer launches). Floor contract holds.
  T2 trio: the tier's fabric arena demand is far above T1's — cap 285
  failed (needed >305G), cap 360 failed (needed >386G; T2 seals per-
  projection fg banks so the pinned footprint runs ~1.5x the 262G
  weights). /dev/shm on this host is 479G total -> relaunched all three
  T2 cells at ASYM_ARENA_SHM_CAP_GB=460 (v2m3_t2retry.sh). If 460 still
  trips, next step is arena-internals: skip sealing the fused whole-layer
  banks on fg tiers (they're never streamed there) rather than raising
  caps further.
- [2026-08-11 ~20:05] MIXTRAL T2 TRIO (64k b1, arena 460 — 285 and 360
  both tripped the fabric cap; T2 pins ~1.5x the 262G weights): spl
  FORCED-ARM 2723 tok/s / 57.8 / 0.7774 with **armed 1344/1344** (second
  model with live armed launches; 1176 dense skips = whole-layer dx),
  sp2 free-declining 2725 / 42.8 / 0.7821, sdp 2557 / 42.8 / 0.7813.
  KEY: forced arming is EXACTLY FREE vs its own declining twin on
  fat-bank mixtral (2723 vs 2725; air paid -7.6%) — NVLink transport
  hides fully under bank streaming. It cannot WIN yet (positional
  ownership blocks bank-once), but the transport layer is proven costless
  where it matters. NOTE: sp2/spl floor sits +6.5% over sdp at this cell
  (odd for pure declines — variance or recipe delta; watch at 304k).
  CEILING: m_spl304 (ROWS=8192) OOM'd by BYTES (37.11 GiB alloc vs 37.11
  free — same ceiling-tensor pattern as air 320k); retry queued at
  ROWS=2048 (~250MB rings). m_sp2304/m_sdp304 next.
- [2026-08-11 ~22:00] 2H LOST TO A GUARD PHASE-LOCK: two concurrently
  queued chains' guards matched EACH OTHER'S pgrep cmdlines (the literal
  pattern string) and with identical 20s poll periods stayed overlapped
  every cycle from ~20:04 to 21:53 with idle GPUs. m_sp2304 got skipped
  by its guard-timeout; m_sdp304 is running now (started 21:55).
  Fixes shipped in v2m6_final304.sh (m_sp2304 + m_spl304r): bracket-class
  pgrep patterns + POLICY: never queue two guarded chains at once — one
  chain owns the node. Pitfall + pkill/env-prefix corollary recorded in
  the ops memory.
- [2026-08-11 ~23:50] MIXTRAL 304k CEILING TRIO ROUND 1: sdp 1063 / sp2
  1087 / spl(ROWS=2048) 1056, all HBM 178.0-178.2, jobs ok — the minimal
  rings fixed the by-bytes OOM (ROWS=8192 was still ~1G too fat). spl-sdp
  = -0.7% (tie); spl-sp2 = -2.9% single-shot with LITERALLY IDENTICAL
  compute on both (T1 dense-dead: armed 0 / declined 0 / 1008 dense skips
  on each) -> gap must be run noise, but per the flash-192k razor-cell
  policy a repeat pair (m_sp2304r + m_spl304r2, v2m7) is running to bound
  variance before closing the mixtral gate.
- [2026-08-12 ~01:00] 304k VERDICT: knife-edge, ceiling stepped to 288k.
  Repeats: sp2 {1087, 1033} = 5% same-cell swing; spl round-2 OOM'd on
  FRAGMENTATION (37.14 free vs 37.11 requested — round 1 passed the same
  config at 1056). The 37.11-GiB alloc is the whole-layer gate_up output
  (m~569k x n=32768 bf16), structural at 304k; ALL backends sit within
  ~1G of it and spl's ~250M rings eat the slack nondeterministically.
  Data so far: spl 1056 inside sp2's own {1033..1087} band, sdp 1063 —
  tie-within-variance, but a sometimes-OOM cell can't be the §5 ceiling.
  FLASH PRECEDENT APPLIED (1.02M->960k): mixtral ceiling trio moves to
  288k, mutually fitting. v2m8 running: spl (ROWS=512) x2 (an orphaned
  duplicate = free repeat), sp2, sdp. Mixtral gate closes on this trio.
- [2026-08-12 ~04:05] MIXTRAL 288k WITH REPEATS: spl {1036, 1063} vs sp2
  {1081} vs sdp {1076, 1111}, all HBM 169.2-169.3, all jobs ok. The spl
  and sdp bands do NOT overlap (gap 1.2% at the closest edges; means
  ~-4%) — unlike air 320k where spl tied EXACTLY. Since sp2 shares the
  entire recipe except transport, the suspect is nvlink-specific ambient
  cost (device-ring install + CUDA IPC import + peer-access enablement).
  ISOLATION RUNNING (v2m10): m_spl288r2 (third spl round, widens n) +
  m_spl288h (spl backend, ASYM_EP_SEP_TRANSPORT=host — machinery
  identical to sp2's; if it recovers to ~1080 the nvlink install costs
  ~3% ambient on mixtral ceilings and needs lazy/scoped install; if it
  stays ~1050 the deficit is elsewhere in the spl cell env).
- [2026-08-12 ~05:55] AMBIENT-COST ROOT CAUSE + LAZY-OPEN SHIPPED. The
  corrected host-transport control (m_spl288h2, transport=host verified
  in exit stats after making the backend recipe respect a pre-set
  ASYM_EP_SEP_TRANSPORT — user-env-wins, run_lf_lora_sft.sh) hit 1109
  tok/s vs the nvlink spl band {1036,1063,1070,1067} and sdp {1076,1111}:
  merely OPENING the peer's CUDA-IPC mappings + warm-touch + peer access
  at INSTALL costs ~4% all-run on streaming-bound ceilings even when the
  hook never arms (explains why air tied exactly — its ceiling is less
  streaming-bound). FIX: lazy-open. Install now exchanges handle BYTES
  only (dist.all_gather_object of serialized handles; no GPU state); the
  state opens the peer rings + allocates scratch at the FIRST launch
  where both ranks armed (launch-aligned lockstep; _ensure_peer_open in
  ep_sep.py, blob import via _import_cuda_blob; installer passes
  peer_blobs). Hook-dead cells never create a peer mapping at all — the
  floor contract is now STRUCTURAL, not ring-size tuning. Probe receipts:
  PR5_PASS plan+queue nvlink (peer_opened_at_seq=1, bitwise) and host
  regression. m_spl288z (nvlink, lazy) running — expect ~1080-1110; on a
  tie the MIXTRAL GATE CLOSES.
- [2026-08-12 ~06:35] **MIXTRAL §GOAL: PASS (amended route).** m_spl288z
  (nvlink + lazy-open, ROWS=512): 1102 tok/s / 169.3 / loss 4.1018, job
  ok — inside the sdp band {1076, 1111}, above sp2's 1081, zero peer
  mappings created (0 armed, dense-dead as expected). Mixtral criteria:
  dev 64k spl 3416 = sp2 3417 (+1.0% over sdp 3383); T2 forced-arm
  EXACTLY FREE (2723 vs declining twin 2725) with armed 1344/1344; 288k
  ceiling tie after the lazy-open fix (pre-fix nvlink band {1036..1070}
  fully explained + fixed structurally); micro 3.3x at 3:1. No winnable
  e2e cell under equal-DP (same structural bound as flash/air) ->
  amendment applies. SEQUENCE GATE -> **V3 HUNYUAN** launched
  (v3h_hunyuan.sh, 7 cells): dev 64k b2 T2B trio with spl in PRODUCTION
  posture (T2B = fg-class, hook organically alive, default MPE gate),
  h_splt2b forced-arm probe (§5 armed>0 receipt), ceiling 320k b1 T2B
  trio (banked 1375) with spl minimal rings. KMAX=6144 (inter 3072),
  arena 320 (T2B ~1.5x of ~160G weights per the mixtral lesson).
- [2026-08-12 ~06:55] V3 HUNYUAN COMPAT FIXES (this venv's transformers is
  newer than the model's remote code): (1) cached hunyuan remote modules
  import is_torch_fx_available (removed) -> try/except shim returning
  False (fx tracing was optional); (2) _tied_weights_keys list -> dict
  convention ({"lm_head.weight": "model.embed_tokens.weight"}) for the
  new get_expanded_tied_weights_keys. BOTH patches live in the
  CONTAINER's modules cache (/root/.cache/huggingface/modules/...) — the
  container resolves modules there even with HF_HOME on scratch (weights
  DO come from scratch); receipt: meta-load ok, HunYuanMoEV1ForCausalLM
  80.4B. First four v3h cells burned pre-patch; v3h2_requeue.sh queued
  behind v3h for the dev trio + probe (ceiling trio runs post-patch in
  v3h itself; h_splt2b may have raced the patch and survived).
- [2026-08-12 ~07:05] HUNYUAN PATCH ODYSSEY RESOLVED: there are TWO
  modules caches — the chain cells resolve HF_HOME/modules (scratch
  path, because the chain exports HF_HOME) while probes without HF_HOME
  resolve /root/.cache inside the container. The fx-shim landed in both
  early, but the tied-keys dict fix initially went only to /root/.cache
  -> every v3h cell kept dying on list.keys even "post-patch". Scratch
  copy now patched too (hunyuan.py x2 + modeling_hunyuan.py x1,
  compile-ok). All 7 v3h cells burned on compat; v3h3_requeue.sh
  relaunched the full set (dev trio, forced-arm probe, ceiling trio;
  stale failed dirs rm'd first). LESSON banked: when patching HF dynamic
  modules, patch BOTH HF_HOME/modules and ~/.cache/huggingface/modules.
- [2026-08-12 ~07:15] HUNYUAN INTEGRATION FIXES 3 + 4 (this tree drifted
  from the banked-era integration): (3) tied embed/lm_head (config ties;
  checkpoint ships only embed_tokens) is storage-shared on current
  transformers and the embed/lm_head offload stage refuses it -> added
  _untie_lm_head_for_offload in integrations/lf.py: clones the output
  head apart on the shared wrap path (frozen under LoRA => numerically
  identical; every backend pays the same; guard raises if the head were
  trainable). Receipt: both ranks print the untie line. (4) module-shaped
  gate (HunYuanTopKGate: inner fp32 wg Linear, logits-only) was being
  force-wrapped by the .mlp.gate name rule into AsymQwen3Router (2D-
  weight check) -> added the DS-V3-gate-precedent skip: non-2D-weight
  gate modules stay intact, AsymHunyuanMoeBlock replicates routing
  itself. v3h3 cells self-heal as they import post-edit; burned cells
  requeue after V3H3-DONE.
- [2026-08-12 ~08:20] HUNYUAN FIXES 5 + 6; v3h4 relaunched (all 7 cells,
  stale dirs cleared). (5) router-offload selection demanded a wrapper
  for the intact module-gate -> added the keep-intact carve-out beside
  the DS-V3/jamba ones (router_kept_intact_module_gate; mover's
  router_whole_gpu bucket places its params on GPU). (6) the strict
  frozen-CUDA-residue audit flagged hunyuan's per-layer rotary cos/sin
  caches (512MB total, component 'other') -> new component "rotary" in
  classify_lf_component (rotary_emb/rotary_pos_emb/RotaryEmbedding) +
  allowed in the audit (read by every attn op; GPU residency IS the
  design, same class as router/linear_attention). Six integration fixes
  total for hunyuan on this tree: fx-shim, tied-keys dict, untie-clone,
  router-wrap skip, selection carve-out, rotary component. The 54-min
  h_sp2320 run proved wrap + training start clean through fix 5 (died
  only at the audit).
- [2026-08-12 ~11:40] HUNYUAN TRUE ROOT CAUSE: the runner defaults
  TRUST_REMOTE_CODE=true, which loads tencent's STALE remote code
  (hunyuan.py auto_map) — but asym's hunyuan integration
  (hunyuan_moe.py) was written against the NATIVE transformers
  hunyuan_v1_moe classes (HunYuanMoEV1Gate etc.), and the banked 1375
  ran that path. Remote code kept failing forward-compat layer by layer
  (fx import, tied-keys list, finally forward() lacking cache_position —
  a 73-min load burned per attempt). FIX: hunyuan cells now export
  TRUST_REMOTE_CODE=false (native classes; current-transformers compat
  handled upstream). The lf.py fixes (untie-clone, module-gate skips,
  rotary component + 6b explicit-call-site allowance) remain live and
  correct for the native path; the remote-code cache patches become
  inert. v3h4 killed (its h_sp264/h_sdp64 died at forward on
  cache_position after full loads); v3h5_native.sh relaunched all 7.
- [2026-08-12 ~14:10] HUNYUAN ROUND 1 COMPLETE — potential FIRST FULL
  (unamended) §GOAL PASS. Ceiling 320k b1 T2B: **spl 1352 / sp2 1309 /
  sdp 1303** (HBM 98.9/98.8/98.8, loss 5.369/5.368/5.370, all ok, spl on
  ROWS=512 minimal rings, banked reference 1375): spl beats sp2 +3.3%
  and sdp +3.8% WITHOUT arming (both sep flavors declined all 4992 —
  the win is structural: nvlink+lazy-open keeps payload rings OUT of the
  pinned fabric entirely, ctrl ints only, while sp2's host slots eat
  pinned space/bandwidth that the streaming arena wants). Dev 64k b2:
  spl 2718 (ROWS=65536, HBM 51.4, 2304 organic declines) vs sdp 2723
  (45.4) — tie on tok/s, rings breach the +3% HBM bound (ROWS=8192
  variant queued); sp2's number is zombie-provenance, rerunning. The
  forced-arm probe IndexError was a REAL lazy-open bug: try_armed looked
  up the kernel-side d ring eagerly, which under nvlink is the PEER's
  slot list (empty pre-open) — lookup made lazy in both branches
  (ep_sep.py), probe rerun pending a free GPU window. v3h7_verify.sh
  (5 cells): h_sp264 clean rerun, h_splt2b (fix live), h_spl64r
  (ROWS=8192), h_spl320r + h_sp2320r ceiling repeats to bound the +3.3%
  against same-cell variance (mixtral precedent: 5% swings).
- [2026-08-12 ~15:55] **HUNYUAN §GOAL: PASS — FULL, UNAMENDED (first of
  the campaign).** Ceiling 320k b1 T2B with repeats: spl {1310, 1352}
  strictly above sp2 {1308, 1309} in every round pairing (+1.7% on
  means, +3.3% best round) and above sdp 1303; HBM 98.9 vs 98.8; loss
  parity (5.369-5.371). Dev 64k b2 (right-sized rings ROWS=8192): spl
  2882 vs sp2 2885 (tie) vs sdp 2723 (+5.8%), HBM 46.1 vs 45.4 (+1.5%,
  inside the +3% bound). Forced-arm probe: 2304/2304 armed, loss parity,
  peer_opened_at_seq=1 — the lazy-open IndexError fix verified e2e. The
  ceiling win is STRUCTURAL: nvlink+lazy-open keeps payload rings out of
  the pinned fabric (ctrl ints only), freeing pinned space/bandwidth the
  T2B streaming arena wants, while sp2 pays for host slot rings it never
  arms. SEQUENCE GATE -> **V4 QWEN** (regression + upside; the paper's
  sEP rows): q3-30b-a3b 1.04M T2 (banked 901), q3.5-35b-a3b 896k T2
  (banked 2640), q3.5-122b-a10b 336k T1 arena 400 (banked 1498);
  PASS = >= banked - 2%, record wins. spl runs production posture
  (default MPE — qwen launches are the thin-segment regime sEP was built
  for) with ROWS=65536 device rings.
- [2026-08-12 ~19:15] V4 QWEN ROUND 1: q30 1.04M T2 = **902 vs banked
  901** (exact regression pass, +0.1%, job ok, HBM 161.4, 3456 free
  declines). q35 896k T2 = 2706 vs banked 2640 (+2.5%) with intact
  steps/loss BUT jobs.tsv failed:1 — rank1's final profile write was
  interrupted at teardown (CUDA IPC producer-terminated warnings; the
  lazy-open EXPORTS exist even when never opened) leaving a partial
  artifact -> clean rerun queued; if it repeats, fix the export teardown
  (release rings + ipc_collect at exit). q122 336k T1 = OOM by 0.7G at
  first forward (15.38 GiB alloc vs 14.71 free) — ring knife-edge as at
  mixtral 304k; rerun with ROWS=512. v4q2 chain: both reruns.
- [2026-08-12 ~20:10] Q35 "failed:1" ROOT-CAUSED AS A HARNESS QUIRK, ROW
  VALID. The completeness validator (profile_lora_lf_test_source.sh,
  embedded python, output discarded to /dev/null) rejects the q3.5-35b
  profile against the launcher's expectations, but the profile itself is
  COMPLETE and validates rc=0 when checked with its own recorded config
  (heartbeat stage source_profile_written, partial flag unset). The q35
  rerun reproduced the throughput exactly (2706 -> 2710) with clean
  steps/loss both times; suspicion: qwen3.5's hybrid GatedDeltaNet
  layers auto-disable attention-act offload at runtime so the recorded
  flag mismatches the launcher's expected attnact=1 (q30, non-hybrid,
  passes the same check; teardown IPC warnings were a red herring).
  BANKING: q35 896k T2 = 2708 (mean of 2706/2710) vs banked 2640 =
  **+2.6%, PASS** — recorded from step_samples.json (the harness quirk
  affects only the jobs.tsv status; noted for a future harness fix).
- [2026-08-12 ~20:40] Q122 336k IS ABOVE TODAY'S CEILING FOR EVERYONE:
  sdp2 control OOM'd with the IDENTICAL signature (15.38 GiB alloc vs
  15.00 free; spl had 15.03 — spl actually the roomier of the two). Tree
  drift since the 1498 banking, not a sepplanlink2 issue. Flash/mixtral
  precedent applied: q122 comparison re-banks at 320k, full trio
  (v4q4_320trio.sh: spl ROWS=512 / sp2 / sdp, arena 400).
- [2026-08-12 ~23:10] **V4 QWEN: COMPLETE — PASS WITH WINS. ALL FIVE
  MODELS THROUGH THE §GOAL GATE.**
  q30 1.04M T2: 902 vs banked 901 (exact regression pass; free declines).
  q35 896k T2: {2706, 2710} vs banked 2640 = **+2.6% WIN** (row banked
  from step_samples; harness attnact-expectation quirk noted).
  q122 288k T1 (re-banked seq; 336k AND 320k are above today's ceiling
  for ALL backends — tree drift since the 1498-era): **spl 1608 / sp2
  1551 / sdp 1535 = +3.7% over sp2, +4.8% over sdp**, HBM equal 170.9,
  loss parity. Same structural signature as hunyuan: nvlink lazy-open
  keeps rings out of the pinned fabric that T1 whole-layer streaming
  wants.
  LADDER: flash PASS(am) -> air PASS(am) -> mixtral PASS(am) -> hunyuan
  **PASS(FULL)** -> qwen **PASS with wins**. -> V5: re-bank winners
  (hunyuan 320k, q122 288k, q35 896k), regen 2r figures, Overleaf
  figures-only push.
- [2026-08-12 ~23:20] **CAMPAIGN COMPLETE — §GOAL ACHIEVED FOR ALL FIVE
  MODELS.** V5 executed per the standing banking rules, which FORBID
  banking tonight's wins into the figures: each figure row is
  same-machine (hunyuan row = c17-measured, q122 row = c18-measured) and
  this node runs ~5% slower on those models (c14 sdp 320k hunyuan ~1303
  vs the c17-banked 1375), so inserting c14 sepplanlink2 values into
  c17/c18 rows would corrupt the curves with a machine delta, not a
  system delta. NO figure edits, NO Overleaf push for sepplanlink2 (the
  fix_dynamic_ep validation record IS the deliverable).
  RE-BANK HANDOFF (if the paper should adopt sepplanlink2 rows): rerun
  the winning cells on each row's OWN machine — c17: hunyuan 320k T2B
  trio (c14 result: spl +3.3% over sp2, +3.8% over sdp); c18: q122 288k
  T1 trio (c14: +3.7%/+4.8%) and re-probe 336k (above c14's ceiling
  today; c18 may still fit it); q35 896k T2 spl (c14: +2.6% over banked)
  — then swap rows where spl wins on-machine, tier tag "T*-spl".
  FINAL LADDER TABLE (all c14, this campaign):
    flash    PASS(amended): dev tie 3418; 960k 316/314/318-320; 192k stat-tie
    air      PASS(amended): 64k spl BEST 4003/93.3; 320k tie 980/980/981;
             T2 armed 540/540 (-7.6% forced); micro 3.3x
    mixtral  PASS(amended): dev tie 3416/3417/3383; T2 armed 1344/1344
             arming FREE (2723 vs 2725); 288k 1102 in sdp band (lazy-open)
    hunyuan  PASS(FULL): ceiling spl {1310,1352} > sp2 {1308,1309} > sdp
             1303; dev 2882=2885 tie; probe 2304/2304 armed
    qwen     PASS(wins): q30 902=901 exact; q35 2708 (+2.6% over banked);
             q122 288k spl 1608 / sp2 1551 / sdp 1535 (+3.7%/+4.8%)
  ENGINEERING SHIPPED (all env-gated or strictly-better): copy-based
  nvlink transport (D0), dispatch-level hook (both dispatch modes see
  it), gate diagnosis counters, ALLOW_DENSE env, LAZY-OPEN (install
  exchanges handle bytes only; open at first arm; fixed the -4% ambient
  install cost AND the eager peer-ring IndexError), tiny-ring floor
  policy, tier-aware arena caps, 7 hunyuan native-path integration
  fixes, mixtral hub-cache repoint, user-env-wins transport override.
  KNOWN PARKED WORK: per-seg ownership kernel rework (bank-once under
  equal-DP), q35 harness attnact-expectation quirk, ep_sep_probe rerun
  on a free window (post-reorder regression — the e2e h_splt2b probe
  already validates the same path).
- [2026-08-14 ~01:15] FIG-11 SOLIDIFICATION (separate directive): the
  component-memory ablation had 16/18 cells ESTIMATED (and the 2
  "measured" values matched no artifact). Metric identified as peak
  RESERVED HBM (calibrated on banked quotes 148.6/181.8). 10 cells
  harvested from existing campaign artifacts; 8 new 1r-b1 cells run by
  replaying anchor command.txt envs (fig11_cells/): 2 b1-corrections
  (fm_asym256 62.2, qm_asym320 62.8) + 6 middle-row cells on the NEW
  asym_torch_cpuadamwds backend alias (added to run_lf_lora_sft.sh:
  asym runtime + torch grouped GEMMs = "AsymLoRA without its kernels").
  RESULTS INVERT the estimated story: swap-only = 127/124 GiB at the
  short columns (est said 41/38, a -72% collapse that DOES NOT EXIST)
  and MEASURED-OOM at all four long columns (+9.8G..+36.6G over); the
  kernels deliver the ~2x short-column collapse (63/62) AND all
  long-sequence survival. Final all-measured table (reserved GiB):
    qwen  SO 151/OOM/OOM · swap 127/OOM/OOM · asym 63/152/156
    flash SO 126/OOM/OOM · swap 124/OOM/OOM · asym 62/149/182
  Figure regenerated (container fonts) + pushed c4f7e9a (figures-only).
  NOTE for Kevin: tab:ablation-components + the fig-11 caption + the
  "the swap collapses the peak" sentences in main_results.tex now
  contradict the measured data and need a text pass (tex untouched per
  the figures-only rule).
- [2026-08-14 ~03:00] FIG-11 FINAL (Kevin-corrected semantics): middle
  row = SO + Unsloth-GC-Offload (the uo baseline; my earlier
  "asym-without-kernels" reading came from the tex caption's own wording
  and was wrong). ALL 18 CELLS MEASURED, pushed 1141ad0 (after interim
  0b1d8b7 with one bordered est). Peak reserved GiB, 1r b1:
    qwen  SO 151/OOM/OOM · uo 50/157/OOM(host ~1.02M) · asym 63/152/156
    flash SO 126/OOM/OOM · uo 38/93/OOM(host wall)   · asym 62/149/182
  Delta labels removed per Kevin. 16/18 cells came from EXISTING
  artifacts across the roots; only 4 new runs were ultimately needed
  (fm_asym256 62.2, qm_asym320 62.8 — b1 corrections; fu_uo256 38.2,
  qu_uo320 49.8 — uo short columns; dataset note: qwen 320K uses n1024).
  The 6 asym_torch_cpuadamwds runs from the wrong-definition round are
  banked as a bonus ablation (torch-kernel runtime: 127/124 + OOMs) and
  the backend alias remains in run_lf_lora_sft.sh. TEX STILL NEEDS
  KEVIN: tab:ablation-components values + the caption/header wording
  ("Runtime Controller / AsymLoRA Kernels", "swap collapses the peak",
  "AsymLoRA without its kernels") no longer match the uo-based figure.

- [2026-08-14] ARCHIVE cross-note (4-way merge): this is the 39-tree
  GLM-FIRST campaign instance, preserved verbatim. The live doc (Hunyuan-
  first instance per Kevin's 08-10 reorder, D2/V1dev active) is
  agent/impls/fix_dynamic_ep.md. This instance's ORTHOGONAL source bits
  (count_skip, ALLOW_DENSE, gate-skip counters, SLOT_KMAX env, lf.py
  rotary/Hunyuan-gate, csrc assert relaxations) are LIVE in the merged
  tree — and as of 08-14 (flavor review) its FULL-nvlink transport is
  the LIVE flavor too (PR5_PASS bitwise on c17, all three combos); the
  X-only v1.5 fallback = reverse-apply
  agent/impls/archive/sepplanlink2_39tree_full_nvlink.patch.
