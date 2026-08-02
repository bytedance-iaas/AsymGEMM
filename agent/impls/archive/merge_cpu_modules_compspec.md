# merge_cpu_modules — APPENDIX: full conflict-composition spec (generated 2026-07-22)

All analysis complete. Full composition spec follows.

---

# MERGE COMPOSITION SPEC — 5 conflicted files (cpu_compute → merge_cpu)

Task restated: exact per-hunk resolutions for the 5 conflicted files, OURS as trunk, THEIR features grafted as default-off branches, defaults-off ≡ OURS byte-identical.

**Global invariants used below**
- `pinned_ledger.try_reserve` never denies at default (caps default 0 ⇒ always True, pure accounting) — verified. Prefetch/direct-reuse/cpu-act/deposits/dedup/policy all default-off — verified per-flag.
- KA managers (`_HBMKeepManager` moe/dense, attn `ctx.keep_acts_hbm`) lack `take_cpu_ready_event`/`host_wait_cpu_ready`/`stage_begin`/`empty_cpu` ⇒ every THEIRS branch that touches those must be KA-guarded (listed per site) **plus** defensive no-op methods added to `_HBMKeepManager` (item F-0 below).

---

## FILE 1: activation_offload.py (2 hunks) — pure union

**Hunk @551–617.** Keep BOTH sides, ours first. Final method order: `wait_cpu_ready_host` (OURS verbatim) → `take_cpu_ready_event` (THEIRS verbatim) → `host_wait_cpu_ready` (THEIRS, but replace its body with a delegation — pop-based host wait loses the get-not-pop re-sync and the stream-drain fallback of ours):
```python
    def host_wait_cpu_ready(self, handle: CPUActivationHandle | None) -> None:
        """Block the host until `handle.tensor` is safe to read from CPU code.
        Delegates to wait_cpu_ready_host (get-not-pop + stream-drain fallback —
        the proven-safe host wait); kept as the cpu_compute-era API name."""
        self.wait_cpu_ready_host(handle)
```
→ `stage_buffer` (THEIRS verbatim) → `stage` with **OURS' signature** (`*, tag=None, mutable: bool = True` + the 3-line mutable comment); the shared body after the marker is already correct.

**Hunk @667–739.** Keep BOTH: `record_cpu_ready` (OURS verbatim) then `stage_begin` + `stage_commit` (THEIRS verbatim). No glue. Watch-item (no code change): `stage_begin` pops the ready event via `take_cpu_ready_event`; a subsequent `wait_cpu_ready_host` on the same handle falls to the stream-drain fallback — safe by design.

Checklist: (1) resolve @551 per order above with delegated `host_wait_cpu_ready`; (2) resolve @667 keep-both; (3) `grep -c "def stage("` = 1, signature has `mutable`; (4) `python -c "import ast; ast.parse(open(...).read())"`.

---

## FILE 2: attention_activation_offload.py (4 hunks)

**Hunk @431–464 (`_empty_strided_cpu_like`).** Composed function body (replaces the whole conflict; the post-marker shared tail `return torch.empty_strided(...)` / `if pinned: pinned_ledger.register_tensor(out, "saved")` / `return out` stays):
```python
    want_pin = bool(pin_memory and torch.cuda.is_available())
    pinned = want_pin
    if pinned:
        # item 4 (fix_cpu_compute.md): these per-save fresh pinned allocs are the
        # largest untracked page-locked class on dense models — book them under the
        # "saved" family; cap denial degrades to the unpinned (sync-copy) behaviour.
        from . import pinned_ledger

        nbytes = tensor.numel() * tensor.element_size()
        pinned = pinned_ledger.try_reserve("saved", int(nbytes))
        if not pinned:
            # Pageable fallback: _pack leaves ready_event=None, so _unpack takes the
            # host-blocking branch — count it (a pin was requested but denied) so
            # async-unpack A/Bs can distinguish "on" from "silently degraded".
            _PIN_FALLBACK_CALLS += 1
    try:
        out = torch.empty_strided(shape, stride, device="cpu", dtype=tensor.dtype, pin_memory=pinned)
    except RuntimeError:
        if pinned:
            from . import pinned_ledger

            pinned_ledger.release("saved", int(tensor.numel() * tensor.element_size()))
            _PIN_FALLBACK_CALLS += 1
```
(Default caps=0 ⇒ try_reserve always True ⇒ counter fires exactly where OURS fired — invariant holds.)

**Hunk @673–749 (`run` + `_prefetch_region`).** OURS' skip-guard first, THEIRS' body after:
```python
    def run(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if not self.module.training or not torch.is_grad_enabled():
            return self.original_forward(*args, **kwargs)
        if self.skip_in_backward and _in_backward_graph_task():
            # (ours' 4-line comment verbatim)
            self.skipped_backward_calls += 1
            self._sync_module_stats()
            return self.original_forward(*args, **kwargs)
        from . import save_dedup as _save_dedup
        ... THEIRS verbatim through the try/finally ...
```
Keep `_prefetch_region` (THEIRS) verbatim after `run`. (Verified: `_dedup_seen/_dedup_alias/_region_handles/skip_in_backward` inits and handle fields `ready_event/prefetch_staged/prefetch_done` are already auto-merged at :619–650/:480–483.)

**Hunk @1393–1414 (backward dA).** KA-precedence composition (the post-marker shared tail is OURS' legacy `else:` — restructure the region to):
```python
            if needs_grad_a:
                if d_s is None:
                    raise RuntimeError("internal error: dS was not computed for dA")
                if getattr(ctx, "keep_acts_hbm", False):
                    # (ours' KA branch verbatim: zeros / GPU GEMM on ctx._ka_u)
                else:
                    # KA off ⇒ U is an offloaded pinned handle; the CPU deposit may
                    # claim the wgrad (K-2), else the legacy padded CPU-right kernel.
                    if _attn_lora_a_grad_cpu_deposit_enabled():
                        _sweep_attn_deposit_releases()
                        grad_a = _try_deposit_attn_lora_a_grad(
                            a, d_s, u_handle, manager, ctx.shared_source, role
                        )
                        deposited_u = grad_a is not None
                    m_grad = _align_up(int(d_s.shape[0]), 64)
                    if grad_a is not None:
                        pass
                    elif m_grad == 0:
                        grad_a = torch.zeros_like(a)
                    else:
                        # (ours' legacy padded body verbatim — the 5-line race comment,
                        #  wait_cpu_ready_host(u_handle) on the OWNING manager,
                        #  _pad_cpu_rows_to / _pad_hbm_rows_to / asym_bf16_cpu_right_matmul —
                        #  WITHOUT the duplicate `m_grad = _align_up(...)` line)
```
(`deposited_u = False` init already auto-merged at :1360. Deposit is unreachable under KA — correct, since `u_handle` is None there.)

**Hunk @1452–1466 (finally).** Composed:
```python
        finally:
            if s_stage is not None:
                manager.release_stage(s_stage)
            if s_handle is not None:
                manager.release_cpu(s_handle)
            if deposited_u:
                # U/shared-source release deferred to the worker sweep (K-2) —
                # _try_deposit took ownership of both.
                pass
            elif ctx.shared_source is not None:
                ctx.shared_source.release()
            elif u_handle is not None:
                manager.release_cpu(u_handle)
            ctx._ka_u = None
            ctx._ka_s = None
            _update_snapshot(ctx.snapshot, manager, ctx.attention_context)
```
(Shared tail lines `ctx.shared_source.release()` etc. are consumed into this block — delete the leftovers after the marker.)

Checklist: (1) @431 composed body; (2) @673 skip-guard-first; (3) @1393 KA-else nesting, dedupe m_grad; (4) @1452 composed finally, remove orphaned tail lines; (5) grep: exactly one `m_grad = _align_up` inside needs_grad_a; `deposited_u` referenced 3×; ast.parse.

---

## FILE 3: dense_mlp_finegrained.py (12 hunks + 3 out-of-hunk guard fixes)

- **@20 imports**: union — `from .activation_offload import ActivationOffloadManager, CPUActivationHandle, fg_chunk_rows` + THEIRS' 4 module imports.
- **@69 defs**: keep OURS' `_finegrained_keep_acts_hbm_enabled` **then** THEIRS' block (`_dense_cpu_act_async_enabled`, `_dense_lora_a_grad_cpu_deposit_enabled`, `_DENSE_PAIR_META`, `_DENSE_DEPOSIT_DIAG`, `_try_deposit_dense_lora_a_grad`) verbatim. No collisions.
- **@306 `_cpu_silu_mul`**: THEIRS' fused branch verbatim (its `host_wait_cpu_ready` now delegates to ours), then fallback waits = **OURS'** `wait_cpu_ready_host(gate/up)` (NOT theirs' `wait_cpu_ready` — the race fix is the trunk). Shared tail unchanged.
- **@334 `_cpu_silu_backward`**: same pattern; fallback = OURS' three `wait_cpu_ready_host` calls.
- **@473 activation dispatch**: OURS' 3 prep lines first, then THEIRS' blocks:
```python
        act_rows = int(gate_cpu.tensor.shape[0])
        act_width = int(gate_cpu.tensor.shape[1])
        act_chunk = fg_chunk_rows(act_rows, act_width) if hasattr(manager, "stage_rows") else 0
        if act_task is not None:
            ... THEIRS' _dense_mul_job submission verbatim ...
        if act_task is not None:
            ... THEIRS' wait + act_cpu = act_cpu_async ...
        elif layer.cpu_activation:
            (shared tail: activation_cpu → elif act_chunk > 0: → else: — unchanged)
```
- **@622 down dA**: `manager.wait_cpu_ready_host(ctx.act_cpu)` (OURS) + call **with** `deposit_ctx=deposit_ctx` (THEIRS' kwarg). Host-wait ⊇ theirs' branch pair — drop their if/else.
- **@748 gate stage**: `if grad_gate_hbm is not None: grad_gate_stage = grad_gate_hbm` (THEIRS) `else: grad_gate_stage = manager.stage(grad_gate_cpu, tag="mlp.dgate", mutable=False)` (OURS' mutable) ; `gate_low_rank = manager.stage(ctx.gate_low_rank_cpu, tag="mlp.S_gate_for_dB", mutable=False)`.
- **@767 gate dA**: `wait_cpu_ready_host(ctx.x_cpu)` + `deposit_ctx=deposit_ctx`.
- **@794 up stage**: mirror of @748 (`grad_up_hbm`, `mutable=False` on both stages).
- **@808 up dA**: mirror of @767.
- **@1200 `_cpu_left_lora_a`**: OURS' `if source.tensor.is_cuda:` KA branch first (verbatim), then THEIRS' `if self.backend == "torch" or not source.tensor.is_pinned():` + its item-4 comment; shared tail unchanged.
- **@1236 `_cpu_right_lora_a_grad`**: order = OURS' `is_cuda` KA branch → THEIRS' deposit attempt block verbatim → THEIRS' `backend=="torch" or not pinned` fallback → shared tail. (KA-precedence: deposit needs a pinned CPU source; is_cuda returns first.)

**Out-of-hunk guard fixes (auto-merged code, required by KA precedence):**
- :433 K-3 async condition — add `and not _finegrained_keep_acts_hbm_enabled()` to `if (not layer.cpu_activation and _dense_cpu_act_async_enabled() and ...)`. (Under dense-KA the manager is `_HBMKeepManager`; `take_cpu_ready_event`/CPU tensors don't exist.)
- :600 backward R5 init — `if not layer.cpu_activation and _act_offload.restage_prefetch_enabled() and hasattr(manager, "stage_begin"):`.

Checklist: 12 hunks per above; 2 guard edits; grep `mutable=False` count = 11 (unchanged from ours); `deposit_ctx` referenced at :590/:627/:772/:813/:838/:853; ast.parse.

---

## FILE 4: qwen3_moe.py (1 hunk @908)

Composed `_cpu_silu_mul` head: THEIRS' fused branch verbatim (host_wait delegates), fallback = **OURS'** two `wait_cpu_ready_host` lines (replace theirs' `wait_cpu_ready`); shared tail (`out = manager.empty_cpu...copy_`) unchanged. Handles here are always real CPU-manager handles (expert-act path) — no KA guard needed.

---

## FILE 5: qwen3_moe_finegrained.py (5 hunks; hunks 4+5 resolved as ONE region) 

**F-0 (defensive, in ours' `_HBMKeepManager`, part of hunk @168 resolution):** add
```python
    def take_cpu_ready_event(self, handle: "_HBMKeepHandle | None"):
        return None  # HBM-kept: nothing was copied; no event exists

    def host_wait_cpu_ready(self, handle: "_HBMKeepHandle | None") -> None:
        return None
```

**Hunk @16 imports:** union — `from .activation_offload import ActivationOffloadManager, fg_chunk_rows` + `from . import activation_offload as _act_offload` + `from . import cpu_ops` + `from . import cpu_worker` + `from . import placement_policy`.

**Hunk @168–430 defs:** OURS' block verbatim (reuse_packed_x, down_dx_staged, keep_acts, `_HBMKeepHandle`, noclone, `_HBMKeepManager` + F-0 methods) **then** THEIRS' block verbatim (direct-reuse counters/gates, `_cpu_act_max_bytes`, `_fg_cpu_act_max_rows`, `_fg_cpu_silu_bwd_enabled`, `_fg_lora_b_grad_cpu_deposit_enabled`, `_fg_stage_dedup_enabled`, `_fg_cpu_act_chunked_enabled`, `_fg_cpu_act_async_enabled`, `_fg_cpu_act_enabled`, `_cpu_act_fits`). Zero name collisions (verified def-by-def).

**Hunk @1118–1522 (forward).** Trunk = OURS. Structure:
1. `down_low_rank = None` (ours, already before the hunk) and **`if fwd_blocks:` blocked loop = OURS VERBATIM, untouched.** (Their features were measured on the full-width path with `lora_a_fwd_gpu=0`; under flagship pins `fwd_blocks` engages and their fwd features correctly stay cold. Regimes are naturally disjoint — no hazard. Follow-up note, NOT v1: enabling cpu-act under flagship pins would require adding `and not (_fg_cpu_act_enabled() and _cpu_act_fits(...))` to the `fwd_blocks` gate.)
2. `if not fwd_blocks:` full-width branch — OURS' gate/up blocks as trunk with THEIRS grafted:
   - gate block: ours verbatim (incl. `wait_cpu_ready_host(x_cpu)` and `_add_grouped_lora_b_delta_` — do NOT take theirs' full-width `gate_delta = _lora_b_forward(...)`; efficiency rule), then before `del gate, gate_low_rank`:
     ```python
                gate_cpu = manager.offload(gate, "moe.gate")
                # byte-diet mech 3 (gate-direct): provisional keep; committed after the
                # act-path decision. Cold under keep-acts (manager already keeps).
                _dr_gate_cand = gate if (_fg_direct_reuse_enabled() and not keep_acts_hbm) else None
                gate_low_rank_cpu = manager.offload(gate_low_rank, "moe.S_gate")
                del gate, gate_low_rank
     ```
   - THEIRS' async-silu block after the gate block, with the KA guard added to the big condition:
     ```python
        act_task = None
        act_cpu_async = None
        split_silu = None
        if (
            not keep_acts_hbm
            and _fg_cpu_act_enabled()
            and _fg_cpu_act_async_enabled()
            and cpu_worker.enabled()
            and cpu_ops.fused_silu_applicable(gate_cpu.tensor)
            and _cpu_act_fits(int(gate_cpu.tensor.shape[0]), int(gate_cpu.tensor.numel()) * 2)
        ):
            split_silu = cpu_ops.split_silu_kernels()
        if split_silu is not None:
            ... THEIRS verbatim (_silu_job / submit) ...
     ```
   - THEIRS' mech-3 commit block verbatim (`_dr_gate = _dr_gate_cand if (... act_task is None and _direct_reuse_ok("gate_direct", 2 * gate_cpu.nbytes)) else None; _dr_gate_cand = None`).
   - up block: ours verbatim + graft before `del up, up_low_rank`:
     ```python
                _dr_up = (
                    up
                    if (act_task is None and not keep_acts_hbm
                        and _direct_reuse_ok("up_direct", up_cpu.nbytes))
                    else None
                )
     ```
   - `del packed` (ours), then THEIRS' `act_stage_prefilled/act_stage_done_ev` init + mul-chain block verbatim (incl. K-4 `_mul_stage_job` — it chunk-fills the down_base stage on the restage side stream; this is the sanctioned chunking mechanism, no new full-width sweeps).
   - activation block — composed 4-way chain, one `carried_*` init before it:
     ```python
        carried_act_stage = None  # K-9: one act stage reused by down_lora AND down_base
        carried_act_gpu = None    # byte-diet mech 2
        fused_silu = cpu_ops.fused_silu_kernels() if (_fg_cpu_act_enabled() and not keep_acts_hbm) else None
        with prof_range(layer._forward_range("moe_finegrained", "activation")):
            act_rows = int(gate_cpu.tensor.shape[0])
            act_width = int(gate_cpu.tensor.shape[1])
            act_chunk = (
                fg_chunk_rows(act_rows, act_width)
                if (not lora_a_fwd_gpu and hasattr(manager, "stage_rows"))
                else 0
            )
            if act_task is not None:
                ... THEIRS verbatim (wait; prefilled-commit / stage+dedup / plain) ...
            elif (
                fused_silu is not None
                and cpu_ops.fused_silu_applicable(gate_cpu.tensor, up_cpu.tensor)
                and _cpu_act_fits(int(gate_cpu.tensor.shape[0]), int(gate_cpu.tensor.numel()) * 2)
            ):
                ... THEIRS' fused-sync branch verbatim ...
            elif act_chunk > 0:
                ... OURS' row-chunked branch verbatim (empty_cpu/stage_rows/record_cpu_ready/_release_chunk_stages) ...
            else:
                ... THEIRS' else-body verbatim (_dr_gate/_dr_up short-circuits, mech-2
                    carried_act_gpu, _gate_was_stage/_up_was_stage release guards) ...
        _dr_gate = _dr_gate_cand = _dr_up = None
     ```
     Defaults-off proof: act_task None, fused_silu None, `_dr_*` None ⇒ chain = ours' chunk-vs-legacy exactly, and theirs' else-body with `_dr_*` None ≡ ours' else-body line-for-line (verified equal incl. under KA: HBMKeep stage/offload/no-op releases).
3. Post-activation forward tail (down_lora `if down_low_rank is None:` with `wait_cpu_ready_host`, down offload, scatter, carried_* consumption at :1607–1654, ctx saves incl. `x_packed`/`keep_acts_hbm`, `seal`) — already auto-merged correctly; verify only.

**Hunks @1902–2016 + @2025–2157 (backward): discard the marker split; rebuild the region from the two scratch files as follows.**
1. OURS' `down_bwd_blocks` blocked path verbatim (from `if down_bwd_blocks and down_dx_staged:` through the blocked loop), with ONE graft (G-D1) on its accumulated dA call:
   ```python
                        grad_down_lora_A = _lora_a_grad_cpu(
                            layer, d_s_down_full, ctx.act_cpu.tensor, down_lora_A,
                            offsets, experts, tag="moe.down",
                            allow_deposit=True, ctx=ctx,
                        )
   ```
   (Host wait already at the block top; `_lora_a_grad_cpu`'s internal gates decide; identical operand contract to the measured full-width deposit.)
2. OURS' `if grad_2d is not None:` full-width down path verbatim (stage/dS/dB/KA-vs-CPU dA/`_apply_lora_dx_` — do NOT take theirs' `grouped_expert_lora + grad_act.add_` full-width dx), with the KA-else grafted:
   ```python
                        else:
                            manager.wait_cpu_ready_host(ctx.act_cpu)   # host wait ⊇ theirs' both branches
                            grad_down_lora_A = _lora_a_grad_cpu(
                                layer, d_s_down, ctx.act_cpu.tensor, down_lora_A,
                                offsets, experts, tag="moe.down",
                                allow_deposit=True, ctx=ctx,
                            )
   ```
3. silu-bwd chain, composed (replaces both sides' activation regions):
   ```python
            silu_bwd_rows = int(grad_act.shape[0])
            silu_bwd_width = int(grad_act.shape[1])
            silu_bwd_chunk = fg_chunk_rows(silu_bwd_rows, silu_bwd_width) if hasattr(manager, "stage_rows") else 0
            cpu_silu_bwd_task = None
            _fused_pair = (
                cpu_ops.fused_silu_kernels()
                if (_fg_cpu_silu_bwd_enabled() and not getattr(ctx, "keep_acts_hbm", False))
                else None
            )
            if (
                _fused_pair is not None
                and cpu_worker.enabled()
                and down_scatter_block_experts == 0
                and ctx.gate_cpu.tensor.device.type == "cpu"
                and cpu_ops.fused_silu_applicable(ctx.gate_cpu.tensor, ctx.up_cpu.tensor)
            ):
                ... THEIRS' K-5 block verbatim (dact offload, empty_cpu dgrads, take_cpu_ready_event,
                    _silu_bwd_job, submit, keep_dgrads_hbm = False, _asym_db_plan, _asym_dact_cpu) ...
            elif _pref_gate is not None:
                ... THEIRS' R5 stage_commit branch verbatim ...
            elif _direct_reuse_ok("silu_bwd_single_stage", ctx.gate_cpu.nbytes):
                ... THEIRS' mech-4 branch verbatim ...
            elif silu_bwd_chunk > 0:
                ... OURS' row-chunked branch verbatim ...
            else:
                ... legacy 3-stage branch (ours ≡ theirs, verified identical; use ours' indentation) ...
   ```
4. Downstream (gate/up backward with dB-deposit call sites :2544/:2651, dA deposit inside `_lora_a_grad_cpu`, `_asym_db_plan` consumption, cleanup) — already auto-merged; verify only.

**Out-of-hunk guard fix (fg backward, :1762):** change the R5 init condition to
`if down_scatter_block_experts == 0 and _act_offload.restage_prefetch_enabled() and hasattr(manager, "stage_begin"):` (KA manager lacks `stage_begin`).

Checklist (fg): (1) @16 union; (2) @168 ours+F-0 then theirs; (3) forward per composition — blocked loop untouched, 3 KA guards (`_dr_gate_cand`, async condition, `_dr_up`), `fused_silu` KA-gated, chain order async→fused→chunk→legacy; (4) backward region rebuilt: blocked path + G-D1, full-width with KA-else graft, 5-way silu chain, `:1762` hasattr guard; (5) greps: `record_cpu_ready` calls only inside `fwd_blocks`/chunk paths; `allow_deposit=True` ×2; `keep_dgrads_hbm = False` exactly once (inside K-5); `_apply_lora_dx_` present; `grouped_expert_lora(` NOT re-introduced in down_base_dx full-width; ast.parse.

---

## Efficiency/precedence audit (flagged + corrected in the spec above)
1. Theirs' full-width `gate_delta/up_delta = _lora_b_forward` and `down_lora_dx = grouped_expert_lora + add` — REJECTED (full-width transients); ours' `_add_grouped_lora_b_delta_`/`_apply_lora_dx_` kept.
2. Dense K-3 async + dense/moe R5 prefetch + moe async/fused cpu-act under KA — all AttributeError/wrong-device hazards; fixed via the 3 dense/fg condition guards, `not keep_acts_hbm` gates, `device.type == "cpu"` belt in K-5, and F-0 no-ops.
3. Theirs' pop-based `host_wait_cpu_ready` — replaced by delegation to the get-not-pop host wait; `take_cpu_ready_event` retained only where a worker job owns the event end-to-end (its `ev.synchronize()` inside the job); the one starvation pattern (pop then later host wait) degrades to the stream-drain fallback — safe.
4. K-4 `_mul_stage_job` chunk loop = existing sanctioned chunking (side-stream fill, ~4 chunks, 64Ki-row floor) — accepted as-is.
5. Deposit-on-blocked-path graft (G-D1) is arg-only; internal rows/policy gates decide — no new loops or small GEMMs anywhere.