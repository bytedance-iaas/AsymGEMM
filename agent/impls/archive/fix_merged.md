# fix_merged.md — post-merge fix list + pre-push validation status (2026-07-15, rewritten 2026-07-16 after the strong-model re-audit; validation CLOSED 2026-07-16 — T9/T9b green, F14 firing-race confirmed & fixed; V3 CLOSED same evening — kernel exonerated, P8 probe green on the fixed tree)

Scope: the SFT-39 (qwen3.5 memory) x SFT (memory/latency) merge. `main_kevin` =
`9897a5c` + **uncommitted working-tree fixes from the 2026-07-16 correction round**
(deliberately left uncommitted for review — see §0b). Backup `main_kevin_qwen35` =
`5a81335`, pushed. Merge completeness is proven by git: `git log sft/main_kevin
^main_kevin` is empty and `git merge-base --is-ancestor sft/main_kevin main_kevin` is
true; the only delta over their tip is the qwen3.5 line (+133 code lines / 7 files).

Audit provenance: a first audit ran on a weaker model (2026-07-15). A strong-model
re-audit (2026-07-16) re-derived **all 18 recorded claims from code** and reviewed the
applied fix diffs; it CONFIRMED 7, CORRECTED 10, and **REFUTED 1** — the corrections are
folded in below and the wrong claims struck. Its fresh-discovery rounds and completeness
critic died on session limits TWICE (second attempt: round-1 finders completed — 35
raw findings, triaged by hand in §0c — but the adversarial verify pass, rounds 2–3,
and the critic never ran), so **formal discovery convergence was NOT reached**. The
audit script is preserved at `agent/handoffs/audit-merged-tree-v2.workflow.js`
(cross-session cache resume is not possible — a new session re-runs it fresh; see
§7c item 3 before deciding to). The V3 kernel read-set hunt, which stopped partway in
that session, was COMPLETED on 2026-07-16 evening by a fresh agent: kernel exonerated,
mechanism = the V1 host race, P8 decisive probe PASSED — V3 is closed (§V3, §6b P8).

Only nontrivial items are recorded. Anything fixable by reading was fixed, not filed.

---

## 0. Fixed by code reasoning — COMMITTED (fd2eb57 / 389db00 / 9897a5c)

| # | Fix | Where | Status after re-audit |
|---|---|---|---|
| F1 | Fold offload-traffic accounting into `record_cpu_ready` | `activation_offload.py` (landed in the merge commit fd2eb57 itself) | **CONFIRMED correct, no double-count** — all 8 call sites fire once per D2H-row-write handle; `offload()` never calls it; CPU-computed fills correctly omit it. Wording corrected: pre-fix the handles were always visible in the cpu-RESIDENCY gauges (`cpu_bytes_by_tag` etc. via `_mark_cpu_live`); only the offload-TRAFFIC view (`num_offloads`/`offloaded_bytes`/`offload_bytes_by_tag`) was blind. The hole covered moe.gate/up (blocked fwd), moe.dup/dgate (chunked silu-bwd), and the dense tags mlp.act/dup/dgate — so pre-merge per-tag D2H-traffic comparisons across chunk settings are apples-to-oranges. |
| F2 | Declare `qwen3_moe_finegrained_fused_home_released` | `frozen_linear.py:139` | **CONFIRMED** — `asdict()` drops undeclared attrs, so the declaration is what makes it reach profile.json; grep confirms this was the only dynamic stats setattr. Proven end-to-end on the 2026-07-16 q3-30b run: `fused_home_released = 48`. |
| F3 | Async-restage comment | `activation_offload.py` | **Amended again 2026-07-16** — see F8. |
| F4 | Forward `ASYM_SAVED_TENSOR_ASYNC_UNPACK` as `ASYM_GEMM_LF_CONFIG_*` | driver | **CONFIRMED placed correctly & byte-inert** (reader treats "" as falsy). Two recorded semantics: (a) `run_lf_profiled_train.py:807` filters empty values, so an UNSET knob yields NO key in the artifact — absence must be read as default-off; only an explicit `=0` records an off-row. (b) The `:-` pattern converts unset→set-empty in the child; safe for this reader, but breaks for any future presence-sensitive reader of the prefixed name. |
| F5 | `pin_fallback_calls` counter (lin-attn) | `linear_attention_activation_offload.py` | **Was incomplete** — extended 2026-07-16, see F9. |
| F6 | D4 receipt correction | `fix_throughput.md` | **Was itself partly wrong** — re-amended 2026-07-16, see F10. |

## 0b. Fixed by code reasoning — UNCOMMITTED (2026-07-16 round, in the working tree for review)

| # | Fix | Where | What/why |
|---|---|---|---|
| F7 | RELEASE_FUSED_HOME now zeroes the freed home's byte telemetry | `qwen3_moe.py` (release block) | Release swapped `_tensor` but never refreshed `HostWeight._metadata`, so `nbytes`/`pinned_cpu_bytes`/`weight_nbytes` kept reporting the freed fused bytes (~E·2I·H·2B per layer) as pinned-resident — hiding exactly the savings the feature delivers. Fix: `_metadata = replace(_metadata, nbytes=0)`. `is_pinned` is DELIBERATELY left true: `_load_from_state_dict` (frozen_linear.py:1986/2388) reads it as the pin intent for a reloaded weight, and a good pre-release checkpoint restored onto a released module must come back pinned. |
| F8 | Rewrote the async-restage comment (again) | `activation_offload.py` stage() | The 07-15 rewrite fixed "overlaps compute" but (a) claimed "host never blocks" as the win — false: a plain `non_blocking` compute-stream copy also never blocks; the construction is pure structural overhead over that equivalent; and (b) its "for real overlap" recipe was race-inducing if followed verbatim (dropping `wait_stream` alone lets the H2D read the pinned source before its producing D2H finishes, and the `_stage_cache` reuses buffers whose prior consumer may still be in flight). New text states both requirements: re-anchor on the D2H ready event (`side.wait_event`) AND use a fresh per-call destination. |
| F9 | Pin-fallback counters extended to ALL saved-tensor modules | `attention_activation_offload.py` (both fallbacks), `decoder_activation_offload.py`, lin-attn updated | The identical silent pageable fallback existed uninstrumented in the attention and decoder siblings — and the DECODER wrapper is live on the same runs that A/B `ASYM_SAVED_TENSOR_ASYNC_UNPACK`, so the exact ambiguity F5 claimed to remove persisted. All three now count (only when a pin was actually REQUESTED — a plain CPU-OOM with pin_memory=False no longer counts), exposed as `pin_fallback_calls_module_global` with the read rule stated in-place: same value on every row of a module; max-across-rows, never sum. |
| F10 | `record_cpu_ready` records its event on the handle's original device stream | `activation_offload.py` | Was `current_stream()` (ambient device) vs `offload()`'s `current_stream(tensor.device)` — on a multi-device process the event could order nothing. Byte-inert today (all call sites single-device). |
| F11 | fix_throughput.md D4 note re-amended | `fix_throughput.md` | My 07-15 correction over-corrected: the dead file's docstring premise IS accurate for flagship rows (LF `checkpointing.py:114` saves the GC boundary root UNPINNED; the pinned `gc_boundary_offload` path only runs under `ASYM_UNSLOTH_GC_NVME=1`, i.e. `_actnvme` backends — run_lf_lora_sft.sh:537), so my attribution of "root saved pinned" to gc_boundary_offload was impossible for those runs; that engagement observation is UNEXPLAINED. Wire-or-delete now leans **wire** (targets a real active bottleneck). |
| F12 | §9c of archive/fix_qwen3.5.md reframed; §9b-post-merge caveat 3 retracted | `agent/impls/archive/fix_qwen3.5.md` | See R1 and V3 below. |

---

## 0c. Late round (2026-07-16): 35 finder results triaged; 9 more fixes — UNCOMMITTED

The audit's discovery finders completed round 1 before session limits killed the
adversarial verify pass, leaving 35 raw findings. Each fixed item below was verified
against the code BY HAND before editing (the finders are otherwise unverified).

Fixed (working tree):
| # | Finding | Fix |
|---|---|---|
| F13 | `wait_cpu_ready_host` POPPED the event, so a second host wait on the same handle (dense waits x_cpu at gate then up) fell into the full-stream-drain fallback every time | pop → **get**: the completed event stays; second wait re-synchronizes instantly. Next fill overwrites the entry (keyed by data_ptr); device-wait/release still pop. |
| F14 | **Missed live site**: attention FORWARD LoRA-A host-pads `u_handle.tensor` whenever M % 128 ≠ 0 (flagship 45k×8 → M=360000 → pad fires) with NO wait | host wait added before `_dense_lora_a_cpu_left` |
| F15 | My :830 backward wait targeted the WRONG manager for shared q/k/v sources (event lives in `attention_context.manager`) — correct-but-slow via the fallback | both attention waits now pick the owning manager (`ctx.attention_context.manager` when shared) |
| F16 | **Missed site**: dense nograd cpu-offload down LoRA-A host-reads `act_cpu` with NO wait (opt-in `NOGRAD_CPU_OFFLOAD` path) | host wait added |
| F17 | `run_job`'s full-fg branch assigned `ASYMM_QWEN3_MOE_FG_{LORA_A_FWD_GPU,DA_GPU,KEEP_DGRADS_HBM}` + the 192 GiB pool WITHOUT `local` → leaked into every LATER row of the same driver invocation | `local` shadows added beside the sibling block |
| F18 | `run_dial_ladder.sh` `rung_ok()` globbed the pre-move `profiling/` path → never matches at HEAD → every rung re-ran | glob updated to `profiling_results/profiling*/` (old path kept as fallback) |
| F19 | `profile_lora_lf_test_source.sh` had NO `QWEN35_DELTA_CHUNK` plumbing (qwen3.5 ≥75k rows launched via source.sh hit the fla illegal-memory fault) and was missing 7 provenance mirrors | full port from both.sh: env-set tracking, 16000 default, resolve/run_job/run_env plumbing, 10 mirror lines |
| F20 | `RELEASE_FUSED_HOME` reader treated a forwarded EMPTY string as OFF (flipping the default if ever mirrored); no provenance mirror; no shared-fabric guard (multi-rank arena-shm banks must not be "released") | reader now empty-safe (empty ⇒ default-on, same convention as DOWN_DX_STAGED) + `_fabric_bank` guard + mirrors for it, `ASYM_EMPTY_CACHE_PHASES`, `QWEN35_DELTA_CHUNK_SIZE` in both.sh |
| F21 | `_alloc_cpu`'s pageable pin-fallback (the POOL one — distinct from the saved-tensor modules) was still silent | `cpu_pool_pin_fallback_calls` counted + exposed in pool stats |

Validation: the FIRST host-sync sweep validated clean (T8: 287.9 s/step vs 291.1
pre-fix, memory byte-identical, loss parity). The re-run covering F13–F21 (which touch
the flagship-live attention forward, F14) is **T9 + T9b — DONE 07-16, both PASS**, and
notably PROVED F14 corrects a FIRING race: T9's flagship losses shifted ~0.05 vs T8/T2
and T9b reproduced that shift to ~0.0008/step (§6a T9 note + §V1). So F14 was NOT just
latency-free insurance — it fixed real, silent numeric corruption (stale attention
LoRA-A input, 40×/step) in the flagship row.

New open items from the same triage (recorded, not fixed — verify before acting; the
adversarial pass never ran on these):
- **D15.** V2 read-direction remainder: the pool can hand a buffer with a pending
  device READ (stage H2D) to a host-WRITING new owner (cpu_left pad `zero_()`), the
  mirror image of the proven write-direction race. Needs the poison-test extension;
  reachable on llama4/dense cpu paths, not flagship. **Sharpened 07-16 evening by the
  independent sweep (§7e NEW-2): pool is LIFO so the riskiest buffer is handed out
  first; concrete sites are the ACT_RECOMPUTE backward silu (`qwen3_moe.py:1297`,
  `llama4:540`) and X_UNPACKED rebuilds (`qwen3_moe.py:997`, `llama4:128`) — both
  default-off knobs. Clean fix: pool-boundary event (record at release_cpu, sync at
  _alloc_cpu).**
- **D16.** `load_state_dict` onto a warm fg module silently trains on STALE splits —
  `_ensure_qwen3_moe_finegrained_bases` early-returns the cache, so restored fused
  weights never propagate (extends D9). Reload-after-fg is the trigger; loud only via
  loss divergence.
- **D17.** DOWN_DX_STAGED silently degrades to the full grad_2d legacy backward when
  `down_bwd_blocks` is empty (`CHUNK_MB=0`, `KEEP_ACTS_HBM=1`, or R ≤ 8192) — the
  default-on knob's OFF path re-appears without any signal; and its staged full down
  bank's HBM headroom at 122B is unquantified.
- **D18.** Tests structurally cannot reach ANY merge-added fg path: every backend test
  runs below the 1 GiB chunk threshold and the fg gates — the 171 green tests certify
  none of the new code (sharpens D7). Same for the production-on `skip_in_backward`
  branch.
- **D19.** Decoder saved-tensor wrapper lacks the `skip_in_backward` gate its two
  siblings have ⇒ `ASYMM_LAYER_ACT_OFFLOAD=true` + unsloth-GC round-trips every ≥1MB
  saved tensor in-backward.
- **D20.** Attention q/k/v share-cache can serve a STALE entry after an exception
  between q_proj and v_proj (cache invalidates only on v_proj); attention `_unpack`
  lacks the released-idempotency flag its siblings have (double-backward
  double-decrements stats); `wait_cpu_ready` device-wait uses the ambient stream
  (multi-device latent, same family as F10).
- **D21.** Lin-attn: strided saved VIEWS under `QWEN35_DELTA_CHUNK` at batch>1
  allocate SPAN-sized pinned buffers (potential pinned blow-up — measure);
  `cpu_owned_bytes` decremented at unpack while the buffer is still alive;
  `stage_concat_columns` does a non-contiguous D2H per llama4 backward (perf).
- **D22.** `bench_engine_tax.py` benches `compiled_dims='mnk'` while training uses a
  different setting — its asym-vs-torch ratios may not be apples-to-apples (UNCERTAIN).

## R. CLAIMS OVERTURNED by the strong-model re-audit (own the record straight)

### R1. REFUTED: "the SFT blocked forward never engages in the flagship row" (was D1)
The `loraafwdcpu` run-dir tag encodes `ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu` — a knob
of the DISABLED (`expact0`) expert-act-offload module. The driver's `full-fg` case
defaults `ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1`, `FG_DA_GPU=1`, `FG_KEEP_DGRADS_HBM=1`
for routed MoE (verified in the script AND in artifacts: 80k
`lora_a_forward_gpu_calls=2400`, `cpu_left=0`, `da_gpu_calls=320`,
`hidden_route_global_tensors_avoided=800`; q3-30b run: `gpu_calls=1152`, `avoided=960`).
Consequences: (a) the post-merge 45k/80k/30b wins **DID exercise the SFT blocked
forward — it is validated by those runs**; (b) the previously proposed "A/B with
FG_LORA_A_FWD_GPU=1 to settle blocked-forward vs H3b" is moot (it already runs at 1);
(c) `stage_rows_calls` is NOT a blocked-forward signal (that branch never calls
stage_rows — the observed counts come from silu-bwd chunks + down-LoRA backward
blocks); use `hidden_route_global_tensors_avoided` and the `gateup_act_blocked` prof
range instead; (d) the `act_chunk` sibling (gated on `not lora_a_fwd_gpu`) CANNOT run
in flagship rows; (e) the CPU-left path (cpu_left.py) is OFF in flagship rows, which
narrows V1's live exposure (see V1).

### R2. OVERSTATED: "the fg101 probe failure is a probe bug; the kernel is fine" (V3)
The isolation A/B proves ORDER-DEPENDENCE, not kernel innocence — see V3 for the
reframed suspect (device-allocator recycling + a possible out-of-contract kernel read)
and the cheap discriminator. (Superseded by R3/V3-closure: the kernel IS fine, but for
a reason neither claim had right.)

### R3. OVERTURNED (2026-07-16 evening): "P5 confirmed a kernel-side out-of-contract DEVICE read"
My P5 conclusion over-read the discriminator: `PYTORCH_NO_CUDA_MEMORY_CACHING=1` makes
every device free a synchronizing `cudaFree`, which ALSO drains pending D2H copies —
masking HOST-read races identically. The static hunt then exonerated the kernel
(every input zero-filled/predicated, no workspace, exact A/B TMA extents) and located
the real mechanism: the probe's non-v2 cases run the CPU-LoRA-A path, whose HOST pad
read of `act_cpu` behind a device-only wait (the V1 race family) captured recycled
LIFO-pool bytes. P8 (canonical order, caching ON, fixed tree) passing at rel_fro
0.006273 byte-identical seals it. Net: **no kernel defect ever existed; the "tolerated
kernel-hygiene defect" framing was wrong; the defect was ours (host sync) and is fixed
in the working tree.**

---

## 1. Blockers before push

### B1. Cross-model coverage — **CLOSED 2026-07-16 by reasoning + ONE lean run**
The rule requires the cross-model matrix for shared-code edits. Closed as follows:

Per-model delta analysis (what each model actually inherits from this merge):
| Model | Path | Untested delta after reasoning | Verdict |
|---|---|---|---|
| q3.5-35b-a3b | MoE fg + lin-attn + FA4 | — | tested post-merge: smoke/45k/80k, all green |
| q3-30b-a3b | MoE fg, ker101 | RELEASE_FUSED_HOME (+ inert counters) — SFT side already validated everything else on 30B@120k pre-merge | **tested 2026-07-16**: s20000 ctl, loss 1.768 step-1 (band 1.775±0.05 ✓), grad_norm 0.41–0.50 (~0.49 ✓), peak 20.0/21.6 GiB, `fused_home_released=48`, GPU LoRA-A + blocked fwd engaged, offload tags correct at I=768. 3.5 min. |
| q3.5-122b-a10b | same code as 35b, bigger E | RELEASE_FUSED_HOME — same shape-independent code path proven on 35b (3 workloads) and 30b | reasoning-closed; no run |
| q3-32b / q2.5-32b / q2.5-72b / llama3.3-70b | dense (`dense_mlp_finegrained`) | counters only: RELEASE_FUSED_HOME lives in qwen3_moe.py (dense never calls it); SFT validated the dense file itself on llama3.3-70b post-merge (their commit 4b14ee8); skip_in_backward moot at attnact0 (their recorded rows) | reasoning-closed; no run |
| llama4-scout | `llama4_experts.py` | counters only: the file is untouched by BOTH sides; its empty_cpu fills are host-side (correctly uncounted); shared-file deltas validated via SFT's real runs | reasoning-closed; no run. (Pre-existing V1-family latent at :333-343 predates the merge — see V1.) |
| q3.5-27b dense | — | — | **stale gate**: model absent from the current driver roster; the s30000 gate in the archive matrix predates the roster. Noted, skipped. |

Note on `skip_in_backward` (qwen3.5-side addition): its AUTO default keys on
`UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU`, which the driver sets by RECOMPUTE MODE
(`recomp-off-*`/`unsloth-off`), not by model — so it engages for ANY model run with
`attnact1` under those modes. The recorded configs for non-qwen3.5 models are
`attnact0` (wrapper not installed, flag moot); engagement is observable via
`skipped_backward_calls` if that ever changes.

Also done pre-close: Stage-1 probe (qwen3.5 shapes + `--zero-b`) PASS; pytest 171/171.

### B2. The G1 headline used a pre-merge B — **CLOSED 2026-07-16 (B re-measured post-merge)**
Fresh B rows (`superoffload_mem|unsloth-off`, same protocol, post-merge tree):
- **B@45k**: 59.44 GiB alloc / 72.43 reserved (2026-07-14 ref: 59.4 / 72.4 — exact),
  lat 281.8 s (ref 261.8, +7.6%, co-tenant noise; memory is the G1 metric).
- **B@80k**: 100.36 GiB alloc / 103.03 reserved (ref: 100.4 / 103.0 — exact),
  lat 363.0 s (ref ~360).
G1 ratios now fresh-on-fresh: **45k = 34.24/72.43 = 0.473×; 80k = 57.37/103.03 =
0.557×** — both well under the 0.80× bar. B's exact reproduction also confirms the
merge left the superoffload path untouched, as reasoned.

### B3. `gc_async_offload.py`: wire it or delete it — OPEN (decision, now leaning WIRE)
Zero callers in tree and history (pickaxe now returns 2 commits: the adder and the
audit doc quoting the string — still no code caller). Its docstring premise is
ACCURATE for flagship rows (unpinned GC-root save via LF checkpointing.py:114; the
pinned gc_boundary path is actnvme-only), so it targets a real bottleneck. The D4
"root saved pinned" engagement observation is unexplained for plain-backend runs and
should be treated as unverified. If wired: fresh A/B with the
`pin_fallback_calls_module_global` counters and the F4-forwarded flag in artifacts.

---

## 2. Real defects found by reading — need isolated tests (cannot be settled by argument)

### V1. Host-side reads of async-D2H-filled pinned buffers are unordered (family, refined)
The load-bearing history: commit `27dde72` (2026-07-07, pre-merge) **weakened
`wait_cpu_ready` itself** from a host-blocking `event.synchronize()` to a device-only
`current_stream().wait_event()` ("never block the host — Megatron rule"), and in the
same commit memoized the cpu-left padder so its incidental host-syncing D2H
(`offsets.to("cpu")`, cpu_left.py:129) runs only on memo miss. Device-side consumers
are correctly ordered; HOST-side consumers of those buffers are not:

- `cpu_left.py:187-188` — host memcpy out of `x_cpu`/`act_cpu`. Within one fg forward
  the 1st call is accidentally covered (memo-miss sync), the 2nd re-reads proven data,
  the **3rd (down, reading act_cpu freshly filled by non-blocking D2H) is genuinely
  unordered**. Exposure: only when `FG_LORA_A_FWD_GPU=0` — **NOT the flagship rows**
  (see R1), so this is a non-default-config latent.
- `attention_activation_offload.py:809` (`_pad_cpu_rows_to` host-reads `u_handle` with
  NO wait at all) — **this one IS on flagship rows** (attnact1; `attn_act_lora_a_forward_calls=160`
  at 80k). Losses in band ⇒ unproven to fire, but the guard is absent.
- `qwen3_moe.py:907-932` — host `F.silu(gate.tensor)`/silu-backward reads right after
  async offloads (non-finegrained qwen3 offload mode — not current flagship).
- `llama4_experts.py:333-343` — same pattern, llama4-scout's active path. Predates the
  merge (the file is untouched by both sides; the exposure was created by `27dde72`).

**RESOLVED 2026-07-16 — race proven in vitro, fixed at all 30 host-reader sites, fix
proven in vitro; latency validation run pending/below.**
- In-vitro proof (`scripts/archive/race_invitro_test.py`, GPU stalled with `torch.cuda._sleep` to
  hold the D2H in flight — the window a busy training stream opens naturally):
  TEST B: `offload()` → `wait_cpu_ready()` → immediate host read returned the
  PRE-COPY poison bytes (7.0 where 3.0 was in flight; 3.0 after sync). TEST A:
  `release_cpu()` → pool re-alloc → new owner's host `zero_()` → the late D2H
  overwrote the new owner (5.0 over the zeros). Both mechanisms REAL; no hidden
  ordering saves them.
- Fix: new `ActivationOffloadManager.wait_cpu_ready_host()` — `event.synchronize()`
  when the D2H event is pending; when an earlier device-side waiter already consumed
  it (wait_cpu_ready pops), falls back to synchronizing the original device's current
  stream (bounded by already-queued work). `_HBMKeepManager` got a no-op counterpart.
  Swapped at every host-reader site — 30 total, each verified by its consumer:
  qwen3_moe.py ×6 (host silu fwd/bwd — llama4 imports these — + index_select rebuild),
  llama4_experts.py ×1 (rebuild), qwen3_moe_finegrained.py ×9 (cpu-left pads +
  cpu-right dA reads; all in non-default cpu/dscatter branches — zero flagship-loop
  cost), dense_mlp_finegrained.py ×13 (mirrors), attention_activation_offload.py ×1
  (the `_pad_cpu_rows_to(u_handle)` read that had NO wait — the one LIVE flagship
  site). The 8 device-reader waits (stage/stage_rows/H2D) are untouched.
- Fix proven in vitro: C1 (event pending) and C2 (event consumed → fallback) both
  read correct data under the same stall that reproduced the race.
- V2's pool-recycle corruption is closed by the same change at the sites that matter:
  the only corrupting interleavings need a HOST first-touch or host read, and every
  such site now synchronizes. Device-side first touches were already stream-ordered.
- [x] Latency validation DONE for the 30-site sweep (T8, 2026-07-16): 287.9 s/step vs
      291.1 pre-fix (−1.1% = noise), alloc/reserved byte-identical (32138.69/35066.00
      MiB), loss per-step parity. The fix is free.
- [x] T9 (covers the later F13–F21 fixes incl. F14, the attention-forward wait) DONE
      07-16 — PASS (memory byte-identical, latency +2.8%, losses in-band). **It also
      PROVED the attention-forward `cpu_left` race was FIRING**: T9's flagship losses
      shifted ~0.05 vs T2/T8 (a pipeline deterministic to ~0.002), and F14 is the only
      F13–F21 fix both live on the flagship forward and able to change numerics (§6a T9
      note). So this site is no longer "unproven to fire" — it fired 40×/step (160 across
      the 4-step run) and F14 corrects it; T9's losses are the correct ones. **T9b
      CONFIRMED stable** (reproduces T9 to ~0.0008/step; stays 0.03–0.055 off T2/T8 every
      step). **Dataset-regeneration alternative eliminated by md5** (§6a T9 note).

### V2. CPU buffer pool can recycle a buffer whose copy is still in flight — host-first-touch only
Refined by the re-audit: `release_cpu` pools after at most a device-side wait, BUT every
DEVICE-side first touch of a recycled buffer (offload's D2H, chunked row writes,
stage/stage_rows) is enqueued on the same compute stream — or the bracketed h2d side
stream — so stream FIFO orders it after the old owner's in-flight copies: **that
dominant path is safe**. Corruption requires the new owner's first touch to be
HOST-side, which exists: `cpu_left.py:185-188` (`padded.zero_()` + row copies — LoRA-A-
on-CPU configs and the attention cpu-left at :641), `qwen3_moe.py:909-932`, and the
host `index_select` rebuilds. Concrete interleaving (GC recompute): layer L+1's
backward enqueues its last chunked stage_rows H2D reads of gate_cpu/up_cpu then
release_cpu's them with no host sync; layer L's recompute-forward pops the same
(bf16, bucketed-rows, pinned) base inside the cpu-left pad and host-writes `zero_()`
before the device executed L+1's reads ⇒ staged chunks read zeros ⇒ silently wrong
dgate/dup. Also noted: `wait_cpu_ready` waits on the ambient-device stream (same
multi-device nuance as F10 — unfixed there, since the right fix depends on the caller),
and cpu_left's padded buffer bypasses the pool entirely, relying on allocator
event-recording that the custom binding never performs.
- [ ] Isolated test: poison-fill recycled buffers in `_alloc_cpu`; assert no consumer
      observes poison, under the GC-recompute interleaving above.

### V3. The fg101 probe failure — **CLOSED 2026-07-16 evening: kernel EXONERATED; it was the V1 host-read race in the probe's CPU-LoRA-A path; fix already in tree; proven end-to-end by P8**
Empirics stand: fg101 alone rel_fro 0.006273 PASS vs 0.170918 BROKEN in the canonical
5-case order; byte-identical across 1896825/c62aef3/fd2eb57. The re-audit EXCLUDED all
five Python-level mechanisms by reading (memos rebuilt per case since the fg Function
detaches its inputs, and value-pure anyway; env re-read per call; plain path mutates
nothing fg reads; pool reads are stream-ordered overwrites; engine-state excluded at
1896825). **Surviving suspect: CUDA caching-allocator DEVICE-memory recycling** — an
out-of-contract read in the ker101/asym-engine path would be deterministic under the
probe's fixed allocation sequence, pass alone (fresh zeroed pages), and pass as
fg101+v2 (different allocation pattern). That would be a kernel-side bug, benign only
while recycled bytes happen to be right. In-band training loss at R≈5M argues against
an always-on miscompute but does NOT bound an allocation-pattern-dependent one.
- [x] **Discriminator RUN 2026-07-16 — allocator recycling CONFIRMED.** Canonical
      5-case order with `PYTORCH_NO_CUDA_MEMORY_CACHING=1`: `fg101 out rel_fro =
      0.006273` — byte-identical to the isolation value — vs 0.170918 with caching
      on. Same script/order/shapes/commit; the only variable was the allocator.
      (fg000 in-sequence: 0.007497, also clean.) Full log:
      `/workspace/qwen35_local/probe_nocache_655360.log`.
      ~~⇒ The ker101/asym-engine path performs an out-of-contract DEVICE-memory read~~
      **CONCLUSION OVERTURNED (see R3 + the hunt findings below): the discriminator
      proved allocator-TIMING dependence, not device-memory recycling — caching-off
      cudaFree syncs also mask HOST-read races, and the true mechanism was exactly
      that (V1 family). The kernel is exonerated (statically + empirically by P8).**
- [x] **Kernel read-set hunt COMPLETE (2026-07-16 evening, static; independent agent,
      68 tool-verified reads). VERDICT: the ker101 fwd-scatter kernel is statically
      EXONERATED — no out-of-contract DEVICE read exists.** Every suspect excluded
      with file:line evidence: (a) scatter accumulator is `torch.zeros` at every call
      site (fg:993/2187/1688), same-stream ordered; (b) swizzle_cd row mapping exact
      (fp32@block_n=64 → swizzle 128 → shortcut `row=lane_idx`); (c) padded act tail
      rows are DATA-zeroed at creation (`padded * valid_rows`, frozen_linear.py:733-735)
      and the K 768→1024 overrun is TMA architectural zero-fill; (d) token/weight pad
      slots `torch.zeros`+`index_copy_` fresh per call, epilogue predicated
      (`route_row < shape_m`, cuh:1035) with 64-bit addressing; (e) A/B TMA extents
      exact; (f) the kernel HAS NO workspace (args = offsets/experts ptrs + 3 TMA
      descs + route ptrs only; barriers/accum in smem/tmem); (g) packed pinned CPU
      weight = the same tensor the passing fg000 multiplies. fg101's device read-set
      minus fg000's = zero-filled token/weight metadata + the zeroed fp32 accumulator:
      nothing uninitialized.
- [x] **P5's interpretation CORRECTED — the discriminator did NOT isolate device
      memory.** `PYTORCH_NO_CUDA_MEMORY_CACHING=1` turns every device free into a
      synchronizing `cudaFree`, which ALSO drains pending D2H copies — masking HOST-side
      read races identically. P5 therefore only proved "some allocator-timing-dependent
      contamination", not a device-side one.
- [x] **Reframed mechanism (the hunt's positive finding): the probe failure is the V1
      HOST-READ RACE, in the probe's own non-v2 path.** The probe's fg cases (non-v2)
      run `FG_LORA_A_FWD_GPU=0` → the CPU-left LoRA-A path: at the failing commits the
      forward did `wait_cpu_ready(act_cpu)` (device-only) then HOST-memcpy'd
      `act_cpu.tensor` in the cpu_left pad (cpu_left.py:187-188) while act's D2H sat
      behind the previous case's still-executing backward ⇒ the pad read the LIFO
      pool's recycled bytes (a previous case's gate/up leftovers — deterministic
      content, hence the 6-decimal-stable 0.170918). Every empiric fits: passes ALONE
      (shallow queue / fresh first-touch pinned allocs ⇒ D2H lands first — timing
      detail inferred, labeled as such); passes with caching OFF (cudaFree syncs);
      passes as fg101+v2 (GPU LoRA-A never runs the CPU-left path); fg000-in-sequence
      passes (its buffers were first-touch fresh after the pool-less plain case).
      **The fix is ALREADY IN THE WORKING TREE** (the V1 sweep: `wait_cpu_ready_host`
      at fg:894 fwd + the backward dA sites). **DECISIVE VERIFICATION PASSED — P8
      (2026-07-16 evening): canonical 5-case order, caching ON, current tree → ALL 5
      PASS; `fg101 out rel_fro = 0.006273`, byte-identical to the isolation (P4) and
      nocache (P5) values; fg101+v2 = 0.006271; fg000 = 0.007497. The ONLY delta vs
      P3's 0.170918 is the working-tree host-sync fix. Log:
      `/workspace/qwen35_local/probe_cacheon_postfix_655360.log`. V3 CLOSED.**
- [x] Adjacent kernel-side LATENTS found by the hunt (record, not bugs today):
      (i) **CD TMA descriptor over-declares** in scatter mode — built with m=P≈5.26M
      over the 655,360-row fp32 output (~43 GB descriptor over a 5.37 GB allocation);
      provably UNREFERENCED today (`if constexpr (!kRouteScatterAdd)` compiles the CD
      store path out, cuh:1184, prefetch skipped :236-239) — a landmine if stores are
      ever routed through it; same pattern in gateup-dx scatter. (ii) **MN-major B
      group-offset mismatch**: kernel advances `b_k_idx += group * aligned_k`
      (cuh:396-399) while the B descriptor uses true-k·num_groups extent — wrong
      whenever k % block_k ≠ 0 on a transpose path (all current transpose shapes
      divide evenly). (iii) The `alone-pass` first-touch-sync inference (cudaHostAlloc
      behavior) is unverified — irrelevant once P8 settles the mechanism.
- [ ] Adjacent latent recorded: under-keyed memos — `_asym_kernel_meta_memo` keyed only
      by `str(device)` (ignores experts), `_asym_route_pad_memo` bakes in
      token_indices/routing_weights, `_asym_lora_meta_memo` bakes in experts — all
      content-pure per-case HERE, dangerous for any caller reusing one offsets tensor
      with changed companions. `_asym_route_meta_memo` on the probe's idx tensor is the
      one memo that DOES cross cases (value-pure; a probe-fix must clear it too).

---

## 3. Dead ends — refuted with evidence, do NOT re-investigate

- **int32 overflow in the routed kernel**: REFUTED (TMA row coordinates fit uint32;
  64-bit descriptor strides; `static_cast<uint64_t>` scatter; int64 token_indices).
  Adjacent note from the re-audit: one host-side `int*int` multiply on the addressing
  path (`make_tma_2d_desc`, csrc/jit_kernels/impls/runtime_utils.hpp:109) widens AFTER
  multiplying — benign at today's row strides (≤ a few K elements), would truncate
  silently if a huge-stride tensor ever reached it.
- **Pool bucket/narrow aliasing at the probe's exact R** (5,242,880 = 80×65536):
  REFUTED narrowly (exact-shape alloc; `zero_()` before use). General pool-recycle
  hazards are V2, host-first-touch only.
- **"ker101 is broken"**: now FULLY refuted (2026-07-16 evening) — statically (hunt:
  every input zero-filled/predicated, no workspace, exact A/B TMA extents, predicated
  64-bit scatter) and empirically (P8: canonical order passes with caching ON once the
  HOST sync fix is in). Ironically "fg101 is a host-device race" — refuted early on —
  turned out to be CORRECT in mechanism class, just at a different site than anyone
  claimed: the CPU-LoRA-A host pad read (V1 family), not the kernel. V3 closed.

---

## 4. Open / deferred — decide, but not push blockers

- **D2.** nograd keeps a full-R `act` [R,I] bf16 (~1.4 GiB @30B-a3b/120k/b8) that the
  qwen3.5 blocked down+scatter avoided. Live under GC. Re-port if fwd peak needs it.
- **D3.** No fg fwd blocking under dscatter (`DOWN_SCATTER_BLOCK_EXPERTS>0` forces
  da_gpu off). Matters only if a dscatter row returns.
- **D4. CLOSED 2026-07-16** — 120k×8 flagship row ran clean post-merge: loss
  0.78–0.89 (continues the 45k/80k trend), grad_norm 0.12–0.22, **70.4 GiB alloc /
  85.7 reserved**, 507.0 s/step, `fused_home_released=40`, GPU LoRA-A + blocked fwd
  engaged, and **`pin_fallback_calls_module_global = 0` in all three saved-tensor
  modules** — no silent pageable degradation at peak pinned pressure. (V1/V2 remain
  latent-by-reading; this adds empirical comfort at max load, not proof.)
- **D5.** `_record_attn_hbm_gemm` increments outside the chunk loop ⇒
  `ASYMM_ATTN_ACT_LORA_CHUNK` on/off indistinguishable in stats. Left alone (changing
  the counter's semantics would rebase history); add a separate chunk counter if dialed.
- **D6.** `ASYMM_ATTN_ACT_LORA_CHUNK=1` silently no-ops if `ASYMM_FG_ELEMENTWISE_CHUNK_MB=0`.
- **D7.** No tests reference the new knobs (`ASYM_SAVED_TENSOR_ASYNC_UNPACK`,
  `_add_matmul_rows_`, `ASYMM_ATTN_ACT_LORA_CHUNK`) — and whether the 171 green tests
  reach ANY new chunked path at their tiny shapes is untested (the audit round asking
  this died on session limits).
- **D8.** smoke grad_norm 0.2136–0.2671 vs recorded 0.22–0.25 — noise, confirm on the
  next smoke.
- **D9 (new).** RELEASE_FUSED_HOME state_dict surfaces: a FULL-model `state_dict()`
  after release silently emits the empty `(0,)` fused tensor (loud only at reload:
  "must be 3D"); adapter-only saves filter the key (the actual LoRA workflow — safe).
  Also the split bases are registered submodules, so a full state_dict grows multi-GiB
  `_qwen3_moe_finegrained_{gate,up}_base.host_weight` keys once any fg forward ran
  (checkpoint key-set depends on timing). And `unsupported_reasons` never validates the
  SPLIT bases' pinning — if their pin failed at creation, release is skipped (its own
  guard) but fg silently runs off unpinned splits (slow, not wrong). Decide if
  full-model saves are ever a real workflow; if yes, guard `_save_to_state_dict`.
- **D10 (new).** `_stage_cache` has NO intrinsic aliasing protection: two LIVE stages
  with the same (device,dtype,shape,tag) key would be the SAME buffer (discipline-only
  safety today); `release_stage(drop_cache=True)` early-returns when the tensor isn't
  accounting-live, silently skipping the drop; the cache has no size cap. Latent traps
  for future callers.
- **D11 (new).** `_HBMKeepManager` safety rests on env immutability mid-step: backward
  re-reads `DOWN_SCATTER_BLOCK_EXPERTS` (fg:1087), so flipping it between a layer's
  forward and backward sends the stand-in into `stage_rows` ⇒ AttributeError.
- **D12 (new).** Vanilla-EP pad-memo collision: the layer-scoped fallback memo (keyed
  `(block_m, rows, device)`, frozen_linear.py:643-654 under pad_memo_context) can serve
  one block's padded offsets to a DIFFERENT block with the same row count but different
  group boundaries ⇒ silent mis-segmentation. EP configs only.
- **D13 (new).** bf16 baking is broader than one branch: act_cpu is always bf16 in both
  fg paths; dgate/dup written bf16 while grad_act is input_dtype; `fg_chunk_rows`
  called without element_size at two sites (budget assumes 2-byte). All fine for the
  bf16 flagship; latent for any future fp32/fp16 config.
- **D14 (new).** `adopt_cpu` (offload() on an already-CPU tensor) increments no traffic
  counters — latent; every current call site passes GPU tensors.

---

## 5. Verified clean (re-derived by the strong-model audit — do not re-review without new evidence)

`_fg_elementwise_blocks`/`_expert_blocks` offset rebasing (bit-identical GEMM segments;
edge cases checked); `_HBMKeepManager` hasattr-guards at every call site under
keep_acts_hbm=1 (modulo D11's env-flip caveat); `fg_chunk_rows`'s `max(8192, rows)`
floor (callers clamp); the F1 fold (no double-count, no missed D2H caller — re-verified
call-site-by-call-site); F2's declaration + the full export chain to profile.json and
runtime_counters; the lin-attn async unpack's use-after-free question (source protected
by `CachingHostAllocator_recordEvent`; the `_asym_restage_keepalive` attr is redundant
and NOT load-bearing — don't rely on it); `_add_matmul_rows_` tiling/dtype chain
(chunk-off math identical to the code it replaced); RELEASE_FUSED_HOME's COMPUTE paths
(every fused-weight consumer fails loudly after release — plain/eval/GC forwards raise
ValueError, act-offload raises NotImplementedError, STP/EP slicing raises; the fg path
runs off split bases indefinitely — 160 forwards @80k + 192 @30b prove the pair).

## 6. VALIDATION LEDGER — every check, config, metrics, status (updated 2026-07-16)

Common config for all training runs unless stated: in-container `asym_sft_39` (never
host), GPU 0 only (GPUs 1–2 = another user's sglang job, untouched), strictly one
experiment at a time; `PROFILERS=source PLOT=false OVERWRITE=true`; 1 warmup + 3
measured steps (4 trainer loss lines); LoRA r64 α16 drop 0.00 target=all, seed 42,
bf16, 1 GPU (GB200 SM100, 184 GiB). qwen3.5 rows auto-resolve: FA4 venv
(`.venv-fa4`), ker101 route code, `QWEN35_DELTA_CHUNK_SIZE=16000`, `moefg1`,
`attnact1`. "tuned env" = `ASYM_CPU_ADAMW_ASYNC_GRAD_OFFLOAD=0` + both
`*_SKIP_IN_BACKWARD=1`. Artifacts under container
`/workspace/qwen35_local/profiling_postmerge_*` (rootfs on /scratch_local). Latency
= `lf.training_step.total` avg over the 3 measured steps; memory = whole-process
peak from memory.md.

### 6a. Training runs

| # | Run (date, tree) | RUNS spec + env | Measured | Reference | Verdict |
|---|---|---|---|---|---|
| T1 | smoke (07-15, fd2eb57) | `q3.5-35b-a3b ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 2048|8|1 ; none|false|true|false|false|false` + tuned env | reserved **19546.00 MiB = 19.09 GiB**, alloc 3129.05 MiB; loss 1.608–1.815 (mean 1.727); grad_norm 0.2136–0.2671; 773 s total | 2026-07-13 row: 19.1 GiB, loss 1.688, gn 0.22–0.25 | **PASS** — memory exact to the MiB (gn marginally wide = D8) |
| T2 | 45k (07-15, fd2eb57) | T1 spec, seq 45000 | **291.1 s/step**; alloc **31.39** / reserved **34.24 GiB**; host RSS 357 GiB; loss 0.8947/0.9203/0.9498/0.9487; `stage_rows_calls=2880`; offload tags `moe.act/gate/up = 110.04 GiB` each (F1 fix live) | pre-merge §9b: 306–308 s, 44.7 / 49.5–50.7 GiB, loss 0.89–0.95 | **PASS** — −29.8% alloc / −32.5% reserved / −5% lat |
| T3 | 80k GOAL row (07-15, fd2eb57) | T1 spec, seq 80000 | **415.0 s/step**; alloc **47.28** / reserved **57.37 GiB**; RSS 500.5 GiB; loss 0.8442/0.9026/0.9096/0.9292; reserved−alloc gap 10.09 GiB; counters: `lora_a_forward_gpu_calls=2400, cpu_left=0, da_gpu_calls=320, hidden_route_global_tensors_avoided=800, stage_rows_calls=4800, forward_calls=160` | §9b: 428–433 s, 78.9 / 91.7 GiB, loss 0.80–0.93, gap 12.8 GiB "irreducible" | **PASS** — −40.1% / −37.4%; **G1 80k = 0.557×** (bar 0.80; §9b recorded 0.89 FAIL) |
| T4 | qwen3-30b control (07-16, 389db00) | `q3-30b-a3b ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 20000|8|1 ; none|false|false|false|false|false`, BARE env (canonical band config; ker101 auto-default for 30b) | loss **1.768**/1.704/1.755/1.623; grad_norm 0.4083–0.5016; alloc 20.0 / reserved 21.6 GiB; 214 s total; `fused_home_released=48` (48/48 MoE layers — F2 counter proven end-to-end), `lora_a_forward_gpu_calls=1152, da_gpu_calls=384, avoided=960, stage_rows=1728`; offload tags at I=768: `moe.act/gate/up = 87.89 GiB` | archive matrix ctl band: loss 1.775 ± 0.05 (step-1 ✓), gn ~0.49 ✓ | **PASS** — closed B1 (cross-model) with one 3.5-min run |
| T5 | 120k long-seq (07-16, 389db00) | T1 spec, seq 120000 | loss 0.7795–0.8869 (continues the 45k→80k trend); gn 0.1233–0.2208; alloc **70.35** / reserved **85.69 GiB**; **507.0 s/step**; 2068 s total; `fused_home_released=40, lora_a_forward_gpu_calls=3840, avoided=800`; **`pin_fallback_calls_module_global = 0` in all 3 saved-tensor modules** | no post-merge reference (pre-merge 120k ran on the SFT tree); the question was headroom + silent pin degradation | **PASS** — closed D4; zero pageable fallbacks at max pinned pressure |
| T6 | B baseline 45k (07-16, 389db00) | `q3.5-35b-a3b ; superoffload_mem|unsloth-off|ligerloss1 ; 45000|8|1 ; none|false|false|false|false|false` | 281.8 s/step; alloc **59.44** / reserved **72.43 GiB** | 2026-07-14 B: 261.8 s, 59.4 / 72.4 | **PASS** — memory exact repro (lat +7.6% = co-tenant noise); closed B2 |
| T7 | B baseline 80k (07-16, 389db00) | T6 spec, seq 80000 | 363.0 s/step; alloc **100.36** / reserved **103.03 GiB** | 359.7–360.5 s, 100.4 / 103.0 | **PASS** — exact repro; **G1 fresh-on-fresh: 45k 34.24/72.43 = 0.473×; 80k 57.37/103.03 = 0.557×** |
| T8 | 45k host-sync validation #1 (07-16, tree = +30-site `wait_cpu_ready_host` sweep) | T2 spec | **287.9 s/step**; alloc/reserved **32138.69 / 35066.00 MiB — byte-identical to T2**; loss 0.8935/0.9202/0.9505/0.9470 (per-step parity with T2); 1186 s vs T2's 1196 s | T2 | **PASS** — the race fix costs nothing (−1.1% lat = noise) |
| T9 | 45k re-validation #2 (07-16, tree = +F13–F21, incl. flagship-live attn-fwd wait F14) | T2 spec, OUTPUT_ROOT `profiling_postmerge_45k_hostsync2` | **296.08 s/step** (lf.training_step.total); alloc **32138.69 MiB — byte-identical to T2/T8** / reserved **35026.00 MiB** (−40 MiB vs T8, allocator noise); measured losses **0.9516/0.9186/0.8937** (warmup 0.9475); `stage_rows=2880` (=T2), `fused_home_released=40`, `lora_a_fwd_gpu=1440, da_gpu=320, avoided=800`, `attn_act_lora_a_fwd=160, cpu_left_lora_a=160`, **`pin_fallback_calls_module_global=0`, `cpu_pool_pin_fallback=0`**; gn 0.158 (not clipped); exit 0 | T2/T8 | **PASS** — memory exact, latency within run-to-run noise (T9 +2.8% but T9b +0.5% vs T2, and T2↔T8 spread is itself −1.1%; F14's true sync cost est. ≲0.3% — ~20 event-wait misses/step × ~7 ms C2C D2H), losses in-band. **Loss trajectory shifted ~0.05 vs T2/T8 → F14 CORRECTED A FIRING RACE (see note below); T9's losses are the correct ones — CONFIRMED STABLE by T9b.** |
| T9b | 45k reproducibility of T9 (07-16, same tree/config) | T9 spec, OUTPUT_ROOT `profiling_postmerge_45k_hostsync2_rep` | **292.63 s/step**; alloc **32138.68 MiB** / reserved **35066.00 MiB** (= T8 exactly); measured losses **0.9524/0.9194/0.8932** (warmup 0.9474) — reproduce T9 to **~0.0008 at every raw step**; exit 0 | T9 | **PASS — F14 correction CONFIRMED STABLE.** T9≈T9b to ~0.001 (as tight as T2≈T8), and both differ from T2/T8 by 0.03–0.055 at every raw step ⇒ the shift is real, deterministic, F14-caused — NOT flaky, NOT noise. |

**T9 FINDING — F14 corrected a firing race (the run's headline result, CONFIRMED by T9b).**
T9's per-step losses (0.9516/0.9186/0.8937, warmup 0.9475) differ from T2/T8 — which
matched *each other* to ~0.002 — by ~0.05, far above pipeline noise. By elimination the
only F13–F21 fix both LIVE on the flagship forward path AND able to change numerics is
**F14** (attn LoRA-A `cpu_left` forward host wait; `cpu_left_lora_a_calls=160`): F13/F16
are dense (off here, `dense_mlp_finegrained_forward_calls=0`); F15 backward is inert
(T8 already synced correctly via the slow wrong-manager fallback → same data); F17–F19
are script/env plumbing on a single row; F20 RELEASE_FUSED_HOME is unchanged (the kernel
reads the *direct* unset env → default-on in BOTH T2/T8 and T9, `fused_home_released=40`
either way — F20 only fixed the empty-*mirror* edge case the kernel never reads); F21 is
telemetry. So F14 **deterministically corrected a host-read race that was FIRING in
T2/T8**: the host read `u_handle.tensor` (the shared q/k/v attention LoRA-A source)
before its non-blocking D2H completed. F14 is verified correct — it waits on the
*owning* manager (`attention_context.manager` for shared sources, the same logic F15
fixed) and P7 proved `wait_cpu_ready_host` returns synced (not stale) data. Consequences:
(a) **T9's losses are the correct ones**; T2/T8 trained on slightly-stale attn-LoRA-A
input 40×/step (10 full-attn layers × q/k/v/o; 160 across the 4-step run). (b) **Upgrades
V1's attention-forward race from "latent / unproven to fire" to PROVEN-FIRING-AND-FIXED**
in the flagship row (this is the second, distinct attn-fwd site — V1/T8 fixed the
backward `:809` read; F14 fixed the forward `cpu_left` read that V1 missed). (c) Latency
is within run-to-run noise (T9b +0.5% vs T2; T2↔T8 spread −1.1%); F14's true cost is the
~20 cache-miss event waits/step (q + o sources; k/v re-sync the completed event
instantly via F13's get-not-pop), est. ≲0.3%. (d) Memory byte-identical ⇒ data-ordering
fix only, zero footprint cost. **T9b CONFIRMED (07-16): losses 0.9524/0.9194/0.8932
reproduce T9 to ~0.0008/step AND stay 0.03–0.055 off T2/T8 at every raw step ⇒ stable
deterministic correction, not flaky, not noise.** **Dataset identity PROVEN (2026-07-16
re-audit): the strongest alternative cause — a dataset regenerated differently after the
container recreation — is eliminated by checksum: `md5 03a587e874869ee88ce5842375791536`
is byte-identical between the 07-13 build (`...s45000.jsonl`, the T2-era file) and the
07-16 regeneration (`...s45000__n256.jsonl`, rewritten during T9b prep); eval file also
identical. The builder is deterministic; T2/T8/T9/T9b all trained on identical bytes.**

### 6b. Numeric probes / unit gates

| # | Check | Config | Result | Verdict |
|---|---|---|---|---|
| P1 | Stage-1 fg numeric probe | qwen3.5 shapes E=256 H=2048 I=512 top_k=8 T=4096; `.venv` python; GPU 0 | all forward paths ≤5% of fp32 ref (worst rel_fro ≈0.0075), nan=0 | **PASS** |
| P2 | Stage-1 `--zero-b` (step-1 condition) | P1 shapes, LoRA-B zeroed | "all forward paths within 5% of fp32 reference" | **PASS** |
| P3 | Stage-1 `--qwen3 --tokens 655360` (canonical 5-case order) | E=128 I=768, R = 5,242,880 routed rows | `fg101 out rel_fro = 0.170918` → BROKEN; **byte-identical at 1896825 / c62aef3 / fd2eb57** | pre-existing, NOT merge-caused → V3 |
| P4 | fg101 ISOLATION (untracked scratch variant, fg101 only) | P3 shapes/inputs | rel_fro **0.006273** → PASS | order-dependence proven |
| P5 | Discriminator: canonical order + `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | P3 shapes | fg101 **0.006273**, fg000 0.007497 — **ALL 5 CASES PASS**; log (container-lost) `probe_nocache_655360.log` | allocator-TIMING dependence proven; original "device-recycling/kernel-read" conclusion **OVERTURNED (R3)** — caching-off syncs mask host races too |
| P8 | **DECISIVE close-out: canonical order, caching ON, FIXED tree** (9897a5c + working-tree host-sync sweep) | P3 shapes; `.venv` python; GPU 0; 2026-07-16 evening | **ALL 5 PASS**: plain 0.007653, fg000 0.007497, **fg101 0.006273** (= P4/P5 byte-identical), fg101+v2 0.006271, fg000+v2 0.007496; log `/workspace/qwen35_local/probe_cacheon_postfix_655360.log` | **V3 CLOSED** — only delta vs P3's 0.170918 is the host-sync fix ⇒ mechanism = V1 host-read race (probe's CPU-LoRA-A path); kernel exonerated statically (hunt) + empirically (this run) |
| P6 | pytest backend suites | `tests/training/test_lf_qwen35_asym_backend.py` + `test_lf_qwen3_asym_backend.py`; `.venv-fa4` (only venv with pytest); GPU 0 | **171 passed** in 39.2 s | **PASS** — matches the recorded 171/171 (but see D18: these tests reach none of the merge-added paths) |
| P7 | In-vitro race test `scripts/archive/race_invitro_test.py` | GPU stalled via `torch.cuda._sleep(4e9)` (~2 s); 16384×512 bf16 pinned handles; poison 7.0/5.0 vs live 3.0 | TEST B (host read after device-only `wait_cpu_ready`): read **7.000** = pre-copy poison (3.000 after sync) → **RACE PROVEN**. TEST A (pool recycle mid-D2H): late copy overwrote the new owner's zeros with **5.000** → **RACE PROVEN**. TEST C1 (fix, event pending): 3.000 → **FIXED**. TEST C2 (fix, event consumed → stream-sync fallback): 3.000 → **FIXED** | V1/V2 mechanisms real; `wait_cpu_ready_host` closes both directions at the host-reader sites |

### 6c. Static verifications (no execution)

| # | Check | Result |
|---|---|---|
| S1 | Merge completeness (git) | `git log sft/main_kevin ^main_kevin` → empty; `git merge-base --is-ancestor sft/main_kevin main_kevin` → true; delta over their tip = the qwen3.5 line only (+133 code lines / 7 files) |
| S2 | Strong-model re-verification of all 18 recorded claims | 7 CONFIRMED, 10 CORRECTED, **1 REFUTED** (old D1: the SFT blocked forward DOES engage in flagship rows — driver defaults FG_LORA_A_FWD_GPU=1; artifacts confirm `gpu_calls=2400, cpu_left=0`). All corrections folded into this doc (§R, §V1–V3, §5) |
| S3 | Adversarial review of my own fix commits | 6 defects found (counter coverage incomplete, race-inducing comment recipe, wrong-manager wait, module-global-counter semantics, event on ambient stream, forwarding pattern) — all addressed (F8/F9/F13–F15) or recorded |
| S4 | 35 discovery-finder results (audit round 1) | hand-triaged, each fixed item re-verified against code before editing: **9 fixed** (F13–F21), rest recorded as **D15–D22**. NOT yet adversarially verified (that pass never ran — 6d) |

### 6d. Paused / incomplete (stopped at user request 2026-07-16; nothing lost, all resumable)

| Item | State | Resume |
|---|---|---|
| **T9** 45k re-validation #2 | **DONE 07-16 — PASS** (memory byte-identical, latency +2.8%, losses in-band). Surfaced the F14 firing-race correction (loss shift, see 6a note). Ran directly in-container (session is already inside `asym_sft_39`; no enroot wrapper needed/available). | — |
| **T9b** 45k reproducibility | **DONE 07-16 — PASS**: losses 0.9524/0.9194/0.8932 reproduce T9 to ~0.0008/step; F14 correction confirmed stable/deterministic. **No training runs left in the queue.** | — |
| Audit adversarial-verify + discovery rounds 2–3 + completeness critic | never completed (session limits ×2, then user pause). Round-1 finders completed → the 35 findings of S4 (hand-triaged only) | `Workflow(scriptPath: …/audit-merged-tree-v2-wf_bfbc73f3-0fd.js, resumeFromRunId: wf_bfbc73f3-0fd)` — re-verify/fix-review phases replay from cache |
| V3 kernel read-set hunt | **COMPLETE 07-16 evening** — kernel exonerated on all suspects (see V3); mechanism = V1 host race in the probe's CPU-LoRA-A path; **P8 decisive probe PASSED** (caching ON, fixed tree, fg101 = 0.006273). V3 CLOSED. Two kernel-side latents recorded in V3 (CD-desc over-declaration; MN-major-B offset mismatch when k % block_k ≠ 0). | — |

### 6e. Bottom line

**T9 + T9b DONE 07-16 — both PASS. All planned validation is now GREEN. Same evening:
V3 CLOSED — the fg101 probe failure was never a kernel bug; it was the V1 host-read
race in the probe's CPU-LoRA-A path, already fixed by the working-tree sweep, and P8
(canonical probe, caching ON, fixed tree) passes with fg101 = 0.006273 byte-identical
(§V3, §6b P8, §R3). The working tree now has ZERO known unfixed defects on any
reachable config; remaining recorded items are latents on default-off knobs.** The pair
surfaced and then CONFIRMED the run's most important result: F14 deterministically
**CORRECTED A FIRING** host-read race on the flagship attention LoRA-A `cpu_left`
forward (§6a T9 note + §V1). T9's losses shifted ~0.05 vs T2/T8 and are the correct
ones; T9b reproduced them to ~0.0008/step ⇒ stable, not flaky. Memory byte-identical
(32138.68–69 MiB alloc), latency within run-to-run noise (F14's true sync cost ≲0.3%),
`pin_fallback_calls_module_global=0`. The working tree (code fixes F7–F21 + all doc updates) is
**UNCOMMITTED** for review; `main_kevin` = `9897a5c` + working tree; backup
`main_kevin_qwen35` = `5a81335` pushed to origin. Remaining decisions: push;
wire-or-delete `gc_async_offload.py` (B3, leaning wire); full-model-save guard (D9);
D15–D22 follow-ups. Rules held throughout: in-container only (NB: this session runs
directly INSIDE `asym_sft_39` — the enroot wrapper is a host-only entry tool and is
absent here; see §7a), one experiment at a time, reference clone
`/home/kevinni/AsymGEMM-SFT` never touched.

---

## 7. HANDOFF — everything a fresh agent needs to finish this (written 2026-07-16)

### 7a. Hard rules (non-negotiable)
1. **Never run anything on the host.** All experiments/tests/builds run inside the
   `asym_sft_39` container. **FIRST check WHERE you are** (the 07-16 T9/T9b session
   found this had changed): if `command -v enroot` is empty, `hostname` ends in `-cNN`,
   cwd is `/workspace/AsymGEMM-SFT-39/...`, and `.venv-fa4/bin/python` gives torch 2.12
   with a working `asym_gemm._C`, you are ALREADY INSIDE the container — run scripts
   **DIRECTLY** (`bash scripts/archive/run45k_hostsync2.sh` from the repo root), and do
   NOT use the wrapper (no enroot to nest; `HOME=/root` so `~/env/bashrc.sh` is absent
   and `_custom_enroot_*` are undefined). Only if you are on the HOST: interactive
   `asym39_enroot_run` (bash function from `~/env/bashrc.sh`; ends in `exec bash -i` so
   it CANNOT take a command); non-interactive
   `scripts/archive/in39_noninteractive.sh '<command>'` (optionally `GPUS=0` to pin
   the GPU). Either way the effect is identical — commands run in-container at
   `/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM`.
2. **Strictly ONE of our experiments at a time.** Before launching:
   `pgrep -f run_lf_lora_sft` must be empty and `nvidia-smi` GPU 0 near-idle.
   Other users' jobs (e.g. sglang on GPUs 1–2) are fine to coexist with — never touch
   them, never use their GPUs. If the machine is busy with the user's other work, WAIT.
3. **Never write to `/home/kevinni/AsymGEMM-SFT`** (reference-only clone; fetch from
   it via the `sft` remote only).
4. **Do NOT commit or push** — the working tree is deliberately uncommitted for
   Kevin's review (see 7c). Only Kevin decides commit/push.
5. Python in-container: use `.venv/bin/python` or `.venv-fa4/bin/python` (torch 2.12;
   the system python is torch 2.9 and `asym_gemm._C` fails with an ABI
   undefined-symbol error). pytest exists only in `.venv-fa4`. Rebuild `_C` (only if
   csrc changed): `bash scripts/lf/rebuild_asymgemm.sh` in-container.
6. `/home` is ~99% full — always `PROFILERS=source`; artifacts go to the container
   rootfs (`/workspace/qwen35_local/...`, physically on /scratch_local). HF weights
   are already cached (`HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface`).

### 7b. State of the world
- Branch `main_kevin` = commit `9897a5c` (5 commits ahead of origin; nothing pushed
  except backup branch `main_kevin_qwen35` = `5a81335`).
- **Backup branch `main_kevin_qwen35_sched_merge_backup` = `e98628c`** (created
  2026-07-16 at Kevin's request, local only, not pushed): full `main_kevin` history +
  ONE snapshot commit holding the entire uncommitted working tree (all F7–F21 edits,
  doc moves, untracked helpers; venvs excluded via .gitignore). Created with the
  **`gbackup <branch>` bash function in `env/bashrc.sh`** (host `~/env/bashrc.sh`,
  bind-mounted in-container at `/workspace/env/bashrc.sh`, next to the other `g*`
  helpers). Kept deliberately simple: `git add -A` → temp commit on the current
  branch → `git branch -f <branch> HEAD` → `git reset HEAD~1` (pointer-only; files
  untouched, changes become uncommitted again). Re-running moves the branch to a
  fresh snapshot; earlier tips (`bcb4ea8`, `083f9e7`) remain in the reflog.
- **Uncommitted working tree** (the 2026-07-16 correction round, F7–F21 + docs):
  code = `activation_offload.py, attention_activation_offload.py,
  decoder_activation_offload.py, dense_mlp_finegrained.py,
  linear_attention_activation_offload.py, llama4_experts.py, qwen3_moe.py,
  qwen3_moe_finegrained.py, profile_lora_lf_test_both.sh,
  profile_lora_lf_test_source.sh, run_dial_ladder.sh`; docs = this file,
  `fix_throughput.md`, `archive/fix_qwen3.5.md` (+ the user's own doc-archive moves:
  the 3 `D` entries in git status are Kevin's, leave them).
- Untracked helpers — MOVED 2026-07-16 (Kevin's request) from repo-root dotfiles to
  **`scripts/archive/`** (leading dots dropped; README.md there maps each to its
  ledger row): `in39_noninteractive.sh` (the runner — HOST-only; no-op inside the
  container), `run45k_hostsync.sh` (T8), `run45k_hostsync2.sh` (T9),
  `run45k_hostsync2_rep.sh` (T9b), `run30b_postmerge.sh` (T4),
  `run120k_postmerge.sh` (T5), `chain_probe_then_B.sh` (P5+T6/T7),
  `race_invitro_test.py` (P7 — the V1/V2 in-vitro proof; TESTs B/A prove the race,
  C1/C2 prove the fix). The `run*.sh` use RELATIVE `scripts/lf/...` paths — run from
  the repo root. Still untracked; `agent/handoffs/audit-merged-tree-v2.workflow.js`
  unmoved.

### 7c. Next-actions queue, in order
1. **T9 + T9b — DONE 07-16, both PASS.** T9 (296.08 s/step, alloc 32138.69 MiB
   byte-identical to T2/T8, losses in-band) surfaced the headline result — **F14
   corrected a FIRING attention-forward race** (loss shift ~0.05, §6a T9 note + §V1) —
   and T9b (292.63 s/step, alloc 32138.68 MiB, losses 0.9524/0.9194/0.8932) reproduced
   T9 to ~0.0008/step, confirming the correction is stable/deterministic. **All planned
   validation is now green; the tree is fully validated for review+push.** Artifacts:
   `/workspace/qwen35_local/profiling_postmerge_45k_hostsync2{,_rep}/...`.
2. **Kevin reviews → commits → pushes** (his action; suggested shape: one commit for
   code fixes F7–F21, one for docs — or a single commit; his call).
3. Audit convergence (optional, token-heavy): cross-session cache resume is NOT
   possible (resumeFromRunId is same-session only); a fresh full run of
   `agent/handoffs/audit-merged-tree-v2.workflow.js` re-does everything (~4–8M
   subagent tokens). The valuable parts (18-claim re-verify, fix review, round-1
   finders) are already folded into this doc — only re-run if Kevin wants the formal
   "two clean discovery rounds + critic" sign-off. If run: mind session limits.
4. ~~Kernel read-set hunt~~ **DONE 07-16 evening — V3 CLOSED.** The hunt exonerated
   the kernel on every suspect; the mechanism was the V1 host-read race in the probe's
   own CPU-LoRA-A path (fix already in the working tree), and the decisive P8 probe
   (canonical order, caching ON, fixed tree) passed with fg101 = 0.006273
   byte-identical. No kernel work needed. Optional hygiene if ever touched: shrink the
   over-declared CD TMA descriptor in scatter mode; guard MN-major-B transpose paths
   against k % block_k ≠ 0 (V3 latents).
5. Decisions for Kevin (no work until he rules): B3 wire-or-delete
   `gc_async_offload.py` (leaning wire); D9 full-model-save guard; D15–D22 follow-ups.

### 7d. Key artifacts & references
- This file = the single source of truth. Deep dives: `agent/handoffs/merge.md`
  (both merge records), `agent/impls/archive/fix_qwen3.5.md` §9b/§9c (scoreboards +
  probe verdict), `agent/impls/fix_throughput.md` (D-table + D4 corrections).
- Recorded baselines used throughout: §6a "Reference" column; qwen3-30b band from
  `agent/impls/archive/fix_finegrained_qwen3.5_moe.md` (Cross-Model Matrix).
- Probe logs: `/workspace/qwen35_local/probe_nocache_655360.log` (in-container path;
  NB the container was recreated 2026-07-16 — this log is gone; the numbers live in
  §6b P5).

### 7e. 2026-07-16 EVENING RE-AUDIT (fresh session, stronger model) — reconfirmation from code+artifacts, no reruns

Method: re-read this doc from scratch, then re-derived the load-bearing claims from
code and surviving artifacts (T2/T8 artifacts died with the container; runs.log,
datasets, and T9/T9b artifacts persist). Independent kernel-hunt and sync-sweep agents
ran in parallel (results folded in below/§V3).

Reconfirmed from code (fresh reads, not trust-the-doc):
- **Manager semantics** (`activation_offload.py`): offload/record_cpu_ready key events
  by data_ptr on the ORIGINAL device's stream; wait_cpu_ready = device-side (pops);
  wait_cpu_ready_host = get-not-pop + `event.synchronize()`, stream-drain fallback only
  when the event is absent. Event keying is consistent across the bucketed pool's
  `_narrow` views (offset-0 ⇒ same data_ptr); release_cpu pops ⇒ pooled buffers never
  carry stale events; no leak (release pops, refill overwrites).
- **F14/F15 owning-manager selection correct in all four sub-cases**: no-context →
  local `manager` (which did the offload, :730); context QKV-shared AND context
  non-shared roles (o_proj etc.) → `attention_context.manager` (acquire_source offloads
  via `self.manager` at :572/:583 in BOTH branches); backward mirrors. Artifact proof
  sharing is live: `source_share_hits=8/misses=4` per layer per 4 steps (q miss;
  k/v hits) — so each shared handle is host-waited 3×/forward, which is exactly what
  F13's get-not-pop keeps cheap.
- **Backward pad no-ops at 45k**: `_align_up(M,64)` == M at M=360000 and
  `_pad_cpu_rows_to` early-returns on equal rows — the backward wait is cheap
  insurance there; the FORWARD pad (M%128=64≠0) is the live host read F14 guards.
- **All 32 `wait_cpu_ready_host` sites re-located** (6 qwen3_moe + 9 fg + 14 dense +
  1 llama4 + 2 attention). The 9 fg sites verified inside `else`-branches of
  `lora_a_fwd_gpu` / `da_gpu` / `dscatter>0` gates ⇒ zero flagship-loop cost, as
  claimed. qwen3_moe/dense wait targets are offload()-filled handles (events present ⇒
  cheap path); host-FILLED handles (act/dgate/dup/rebuilt) are only consumed by DEVICE
  stages afterward (no host wait needed, none paid).
- **Over-sync verdict: none material — with three named exceptions, all empirically
  ≈free** (corrected after the independent sweep, below): (1) fg backward act waits
  (fg:1119/1248/1385) can hit the popped-event stream-drain fallback (the forward
  stage consumed act's event) while guarding cpu_right KERNEL reads that same-stream
  FIFO already orders — an unnecessary drain up to once per layer backward where
  reachable; (2) dense 330/398/735 host-wait HOST-filled act handles (no event ever)
  → case-(iii) full drain per call for an already-valid buffer — dense/CPU_ACT
  configs only; (3) governor bounce-buffer waits (NVMe mode only). T8/T9 latency
  (the sweep tree measured −1.1%/+noise vs pre-fix) proves all of these cost ≈0 in
  practice — the stream is nearly caught up at those points. Micro-cleanup
  candidates, not stalls. Flagship's real cost stays ~20 event waits/step ≲0.3%.
- **F7/F9/F16/F20/F21 all present in code as described** (F20 reader: unset→mirror,
  empty→default-ON, `_fabric_bank` + split-pin guards before release; F7 zeroes
  `nbytes` via `_dataclass_replace` at qwen3_moe.py:2582).
- **D-items hand-checked**: D16 CONFIRMED (warm-cache early-return at :2517-2520 ⇒
  stale splits after any reload; loud-or-silent depends on release state); D19
  CONFIRMED (decoder: 0 `skip_in_backward` occurrences vs 3 in each sibling); D17
  CONFIRMED structurally (empty `down_bwd_blocks` ⇒ silent legacy fallback; indirectly
  observable via `down_base_calls=0`); D20 share-cache key = (storage_ptr, offset,
  shape, stride, dtype) ⇒ stale-hit needs allocator address+geometry reuse after an
  exception — real but narrow; **D22 DOWNGRADED** (bench_engine_tax.py contains no
  `compiled_dims` reference at all — the recorded claim doesn't match the code;
  re-verify before acting).
- **B3 premise CONFIRMED from LF code**: `checkpointing.py` UnslothGradientCheckpointing
  saves the per-layer boundary root via `hidden_states.to("cpu", non_blocking=True)` —
  PAGEABLE. At 45k×8 that is ~2.9 GiB × 40 decoder layers ≈ 118 GiB pageable D2H per
  forward + the same H2D in backward. `gc_async_offload.py` itself REVIEWED CORRECT
  (side-stream H2D anchored on the D2H ready event + fresh per-call destinations +
  record_stream + keepalive — precisely the pattern F8's rewritten comment requires).
  If wired: watch pinned pressure (~118 GiB of live pinned roots) via the
  `pin_fallback_calls_module_global` / `cpu_pool_pin_fallback_calls` counters.
- **Dataset identity PROVEN by md5** (see §6a T9 note) — closes the last alternative
  explanation for the T9 loss shift.
- **V3 CLOSED end-to-end this session**: independent kernel-hunt agent statically
  exonerated the ker101 fwd-scatter kernel (all 7 suspects excluded with file:line
  evidence), identified the true mechanism (V1 host-read race in the probe's non-v2
  CPU-LoRA-A path — the probe was our own victim #2 after F14), corrected P5's
  over-conclusion (caching-off masks host races too → R3), and the decisive P8 probe
  run (canonical order, caching ON, fixed tree) passed all 5 cases with fg101
  rel_fro = 0.006273 byte-identical to P4/P5. Two kernel-side latents recorded
  (CD-descriptor over-declaration, unused in scatter mode; MN-major-B group-offset
  mismatch if k % block_k ≠ 0 ever reaches a transpose path).
- **In-vitro test re-read**: TESTs B/A/C1/C2 are sound and match P7; TEST A covers the
  D2H-write direction only ⇒ D15 (pending device READS handed to a host-writing new
  owner) genuinely still needs the poison-test extension. Suggested pool-boundary fix
  if D15 is pursued: record an event at release_cpu (after its device wait) and
  synchronize it in _alloc_cpu before handing out a recycled buffer — closes both
  directions at the pool contract.

**Independent sync-sweep agent (2026-07-16 evening) — full fresh audit of every host
touch of async-filled CPU buffers across asym_gemm/. VERDICT: flagship airtight.**
All 32 wait sites re-derived owner-correct (the attention pair explicitly routes to
the event-owning manager; no wrong-manager calls anywhere); every flagship
host-first-touch write transitively covered by a same-scope fresh-event host wait;
saved-tensor wrappers / gc offloads / cpu_adam drain / ep_sep / weight_offload all
guarded; no unguarded raw non_blocking-copy-then-host-touch outside the manager
family; event-dict lifecycle leak-free ("no staleness or leak" docstring accurate).
New findings (all non-flagship or latent; none block push):
- **NEW-1 (fragile incidental guard, coarse mode).** `qwen3_moe.py:1066` and
  `llama4_experts.py:288` — the COARSE forward (`ASYMM_EXPERT_ACT_OFFLOAD=1` +
  default `LORA_A_FWD=cpu`; flagship runs expact0 so NOT flagship) host-pads
  `x_cpu.tensor` with NO explicit wait; today safe only because the pad-memo miss
  does a blocking `offsets.to("cpu")` (cpu_left.py:129) that drains the stream.
  Should get an explicit `wait_cpu_ready_host` — one line each.
- **NEW-2 (D15 sharpened to concrete sites).** The pool is LIFO (most-recently
  released = most likely in-flight = handed out first). Concrete unprotected
  host-writes into popped buffers: backward silu-recompute under
  `ASYM_OFFLOAD_ACT_RECOMPUTE=1` (`qwen3_moe.py:1297`, `llama4:540`) and x-rebuilds
  under `ASYM_OFFLOAD_X_UNPACKED=1` (`qwen3_moe.py:997-1006`, `llama4:128-137`) —
  previous owner released right after enqueuing stage H2D / zero-copy kernel reads.
  Both knobs default-off. The pool-boundary event fix above closes all of them.
- **NEW-3 (dead-code latent).** `llama4_experts.py:172` recompute pad read —
  unreachable today (`gate_up_recompute` hardcoded False at :255).
- **NEW-4 (pinned-buffer lifetime question — worth one in-vitro check).** The
  cpu_left `padded` buffer and the attention backward `u_source` pad are Python-freed
  right after the async kernel launch that zero-copy-reads them; no
  CachingHostAllocator recordEvent covers custom-kernel reads (only copy_ paths).
  Likely neutralized because the asym CPU-operand kernels are host-blocking (their
  per-call host cost is documented as ~seconds in-code), and at all recorded
  seq×batch shapes M%64==0 so the backward pad early-returns the source (no fresh
  buffer at all). Latent for odd shapes IF the kernels are ever made async — add to
  the poison-test list.
- **NEW-5 (micro).** `gc_boundary_offload` releases WITHOUT its ev.synchronize on the
  exception path only (`finally`) — pool gets a buffer with a possibly in-flight H2D.
  Exception-only.
