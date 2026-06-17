# Fix CPU-RAM OOM during the FIRST forward (Qwen3-30B-A3B LoRA-SFT)

## Corrected diagnosis (measured)
- **OOM is CPU host RAM, in the FIRST forward** — heartbeat dies at `model_forward_enter`, `global_step 0`, **before any backward**. No-recompute ⇒ all 48 layers' activations are offloaded to **pinned CPU** and held live at the forward peak.
- **Numbers:** s8192 **completes** at ~**815 GB** peak; s10240 **OOMs** at ~**970 GB**. ~**9.4 MB/token** live. GPU nearly **idle (104 MiB used, 188 GB free)**.
- **Effective ceiling < 958 GiB:** under `--membind=0,1` + first-touch, pinned buffers **skew onto one Grace node**, so practical ceiling ≈ 820 GB (matches s8192-fits / s10240-OOMs).

## Done / ruled out
- **DONE — free duplicated base weights** (`AsymQwen3Experts.__init__` empties `source.gate_up_proj/down_proj`; `gc.collect()` in `apply_lf_asym_lora`). Validated bit-exact + `pytest tests/training/test_lf_qwen3_asym_backend.py` = 108 passed. **Real effect only ~13 GB** (duplicate is mostly mmap-reclaimable) → keep (lossless) but it is **not** the OOM fix.
- **RULED OUT — pool cap** (`ASYM_EXPACT_CPU_POOL_MAX_BYTES`, default 32 GiB): the pool fills only on **backward release**, so it is **empty in the first forward** → 0 saving at the OOM point.

---

## Shared validation harness (used by every lever below)
**Baselines that already exist:** s10240 OOM run (`…/b8_s10240/`, heartbeat `model_forward_enter`, rss_peak ~200 GB); s8192 completed run (`…/b8_s8192/`, `source_profile_written`, rss_peak ~815 GB, has timing).

**Run one config** (single backend, single seq), on an **idle node** (`numactl -H` → check node 0/1 free first; `nvidia-smi` → pick an idle `GPU_POOL`):
```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
BACKEND_SPECS="asym_cpuadamwds|norecomp" SEQ_LENS=10240 GPU_POOL=0 \
PER_DEVICE_TRAIN_BATCH_SIZE=8 MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
bash scripts/lf/profile_lora_lf.sh        # swap SEQ_LENS=8192 for the latency control
```
**Read the three metrics from the run dir:**
```bash
RUN=$(find profiling -name process_memory.csv -newermt '-20 min' -printf '%h\n' | sort | tail -1)
# (1) completion: source_profile_written = OK ; model_forward_enter = OOM
python3 -c "import json;print(json.load(open('$RUN/lf_run/heartbeat.latest.json'))['stage'])"
# (2) peak CPU RSS, GiB (max over rows, col5=rss_peak_bytes)
awk -F, 'NR>1&&$5+0>m{m=$5} END{printf "CPU rss_peak = %.1f GiB\n", m/1073741824}' "$RUN/process_memory.csv"
# (3) fwd/bwd host-ms (latency) + peak HBM
grep -E 'step\.forward|step\.backward' "$RUN/summary.md"; grep -i 'reserved' "$RUN/memory.md"
```

**EVERY change is a runtime TOGGLE — default OFF = today's behavior. Keep the OFF code path intact behind the flag** so we can flip it and instantly revert. A/B = run the **same config twice, flag OFF then ON**, and diff the 3 metrics. Toggles:
| Lever | Toggle env | OFF (baseline) | ON (change) |
|---|---|---|---|
| 1 | `NUMACTL_MODE` | `membind` | `interleave` |
| 2A | `ASYM_OFFLOAD_ACT_RECOMPUTE` | `0` (hold `act`) | `1` (drop+recompute) |
| 2B | `ASYM_OFFLOAD_X_UNPACKED` | `0` (offload packed) | `1` (offload hidden+regather) |
| 3 | `ASYM_ACT_CPU_BUDGET_BYTES` | unset/`0` (no budget) | e.g. `161061273600` (150 GiB) |

**GLOBAL ACCEPT / REJECT (decide per lever from its OFF→ON A/B at the same config):**
- **ACCEPT only if BOTH:** (a) the **target memory drops meaningfully** — s10240 goes **OOM→`source_profile_written`**, *or* CPU `rss_peak` ↓ **≥ 40 GiB** (Lever 3: CPU ↓ ≥ 40 GiB **and** HBM peak stays **< 180 GiB**); **and** (b) **latency not blown up** — at the **s8192 control** `step.forward + step.backward` (ON vs OFF) regress **≤ 5 %**.
- **REJECT if ANY:** memory drop **< 10 GiB** (trivial); **OR HBM/CPU unchanged but latency goes up** (pure cost, no benefit — always reject); OR (fwd+bwd) regression **> 10 %**; OR it still OOMs with rss_peak unmoved.
- 5–10 % latency or 10–40 GiB memory = inconclusive → re-measure / tune before deciding.
- **Bottom line: keep a change only if it cuts HBM/CPU *non-trivially* AND does not blow up time.** A change that doesn't move memory but adds time is rejected outright.

---

## Lever 1 — NUMA interleave  — ❌ **TESTED → REJECTED** (no memory benefit; harmful under imbalance)
> **Measured 2026-06-16.** Pinned-alloc split: `membind=0,1` → 100% one node at 40 GB; `interleave=0,1` → 50/50. **BUT** the s8192 run completed at **815 GB RSS under `membind=0,1`** > 490 GB (one node) ⇒ `membind=0,1` **already spills across both nodes** (cap ~980 GB, not ~490). So interleave **cannot raise the ceiling**; the s10240 OOM (~970) is genuine ~total-CPU exhaustion (~970 vs ~980). And under co-tenant imbalance (node0 438 / node1 314 free) interleave (forced 50/50) OOMs ~124 GB *earlier* than membind. **The earlier "one-node-skew → 820 GB ceiling" was wrong** (micro-bench too small to trigger spill). Toggle code is wired (default `membind` = no-op); leave it, do not enable. Go to Lever 2.

### Implementation (wired, default OFF)  — **Toggle:** `NUMACTL_MODE=interleave` (ON) / `membind` (OFF, default)
Edit `scripts/lf/run_lf_lora_sft.sh`:
- `:30-33` add a mode flag, **default OFF (`membind`) = today's behavior** so nothing changes until you flip it: `NUMACTL_MODE=${NUMACTL_MODE:-membind}` (values `interleave|membind`). (Flip default to `interleave` only after it's accepted.)
- `:1925-1929` build the args by mode:
```bash
if [[ "${NUMACTL_MODE}" == "interleave" ]]; then
  NUMACTL_CMD=( "${NUMACTL_BIN}" "--interleave=${NUMACTL_MEMBIND}" "--cpunodebind=${NUMACTL_CPUNODEBIND}" )
else
  NUMACTL_CMD=( "${NUMACTL_BIN}" "--membind=${NUMACTL_MEMBIND}" "--cpunodebind=${NUMACTL_CPUNODEBIND}" )
fi
```
Mirror the same in `scripts/lf/profile_lora_lf.sh` (the `numactl --membind=0,1 …` near `:297`). Do **not** pass `--membind` and `--interleave` together.

### Validate
```bash
# memory/completion at s10240, interleave ON:
NUMACTL_MODE=interleave BACKEND_SPECS="asym_cpuadamwds|norecomp" SEQ_LENS=10240 GPU_POOL=0 bash scripts/lf/profile_lora_lf.sh
# then run the 3-metric block. Also confirm pages split ~50/50 during forward:
PID=$(pgrep -f 'src/train.py' | head -1); awk '/N0=|N1=/{n0+=gensub(/.*N0=([0-9]+).*/,"\\1",1);n1+=gensub(/.*N1=([0-9]+).*/,"\\1",1)} END{print "N0pages",n0,"N1pages",n1}' /proc/$PID/numa_maps
# latency control at s8192 (compare to existing s8192 baseline timing):
NUMACTL_MODE=interleave SEQ_LENS=8192 BACKEND_SPECS="asym_cpuadamwds|norecomp" GPU_POOL=0 bash scripts/lf/profile_lora_lf.sh
```
### Accept iff
s10240 → `source_profile_written` (was OOM) **and** N0≈N1 (balanced) **and** s8192 (fwd+bwd) ≤ 5 % over baseline. **Reject if** s10240 still `model_forward_enter`, or N0/N1 still ~100/0 (interleave not taking effect), or s8192 latency > 10 %.

---

## Lever 2 — Stop storing derivable activations  (lossless) — ⚠️ **TESTED: reduces RAM but adds latency**
> **MEASURED 2026-06-16 — controlled s8192 A/B, same node, back-to-back (`MAX_STEPS=1` = 5 warmup + 1 measured):**
> | | CPU rss_peak | Δ RAM | step time | Δ time |
> |---|---|---|---|---|
> | OFF | 748.2 GiB | — | ~157 s/it | — |
> | **2A** (`ACT_RECOMPUTE=1`) | 701.0 GiB | **−47 GiB** | ~214 s/it | **+36% → REJECT** |
> | **2B** (`X_UNPACKED=1`) | 666.4 GiB | **−82 GiB** | ~174 s/it | **+11% → borderline** |
>
> Both **bit-exact** (pytest 115/115). Both cut RAM, but both **add latency**: re-materializing in the backward (CPU `silu` for 2A, CPU `index_select` gather for 2B) **adds CPU work to an already CPU-bound step** (DeepSpeed CPU AdamW + offload dominate the ~157 s). My "latency ≈ flat" estimate was wrong. **2A rejected** (−47 not worth +36%); **2B borderline** (big −82 but +11% > the 10% line). 2B is the better of the two and could be revisited if its rebuild is optimized (pool the pinned buffer / GPU gather). **⇒ Prefer Lever 3 (latency-neutral — removes a copy instead of adding compute).** Toggle dirs now encode `actrecomp{0,1}__xunpack{0,1}` so OFF/2A/2B/BOTH land in distinct folders (no `OVERWRITE` needed).

### Implementation (`qwen3_moe.py` `_ActivationOffloadQwen3ExpertFunction` @910; pack `pack_tokens_contiguous` @2504; grad kernels in `exp_act_offload_lora.py`)
**STATUS: implemented + gated behind `ASYM_OFFLOAD_ACT_RECOMPUTE` / `ASYM_OFFLOAD_X_UNPACKED` (default OFF); encoded in the profiler dir name + forced off for non-AsymGEMM backends.**
**Toggles — gate BOTH behind env flags, default OFF = today's offload path kept byte-for-byte:** 2A `ASYM_OFFLOAD_ACT_RECOMPUTE` (0/1), 2B `ASYM_OFFLOAD_X_UNPACKED` (0/1). Read them once into a layer/ctx bool; `if flag: <new path> else: <existing offload>`. Keeping both paths is what lets us A/B each independently (and revert instantly).
> **VERIFIED CAVEAT (code-checked):** there is **NO HBM-source LoRA-A *grad* kernel** — only `grouped_lora_a_forward_hbm` (fwd, @125). Grads are CPU-source only: `grouped_lora_a_grad_cpu_right` @190, `grouped_lora_a_pair_grad_cpu_right` @218. So do **not** "switch to an HBM grad analog" (it doesn't exist). Instead **keep the CPU-source AsymGEMM grad kernels** and feed them a **transiently re-materialised** operand in backward. That cuts the *forward* peak (what OOMs) while the rebuild is short-lived in backward and never on the forward peak — and it keeps the AsymGEMM CPU-source contribution intact. (Alternative if you ever want X on GPU: route through the autograd-backed all-GPU grouped LoRA in `lora.py:766 grouped_expert_lora` — but that bypasses AsymGEMM, so prefer the transient-restage path.)

**2A (do first): stop *holding* `act` across the forward.** Today `act_cpu = _activation_offload_cpu_silu_mul(gate_cpu, up_cpu, …, tag="act")` (`qwen3_moe.py:877,1003`) computes act **on CPU** and `ctx.act_cpu = act_cpu` (`:1075`) holds it pinned the whole forward. **Verified:** its ONLY backward consumer is the down LoRA-A grad `grouped_lora_a_grad_cpu_right(…, ctx.act_cpu.tensor, …)` (`:1170`), and `gate_cpu`/`up_cpu` are still alive there (released only at `:1202-1203`).
1. Forward: still use `act_cpu` for the down path (`:1024-1068`), but **do not put it on `ctx`** — `manager.release_cpu(act_cpu)` after the down forward so it leaves the held set.
2. Backward (just before `:1170`): recompute `act_cpu = _activation_offload_cpu_silu_mul(ctx.gate_cpu, ctx.up_cpu, manager, tag="act")` — **same CPU helper on the same CPU inputs ⇒ bit-identical** (no GPU round-trip, no new kernel) — feed `grouped_lora_a_grad_cpu_right`, then `release_cpu`.
Saving = the `act` hold off the **forward** peak = 8N×768 bf16 ×48 layers ≈ **~48 GB at s10240**.

**2B (bigger): store `X` unpacked (1×) instead of packed (8×).** Today `x_cpu = manager.offload(packed,"X")` (`qwen3_moe.py:932`). **Verified:** `packed = pack_tokens_contiguous(hidden, metadata)` is **exactly** `hidden_flat.index_select(0, token_indices)` (`moe.py:743`); `x_cpu`'s ONLY backward consumer is the gate/up LoRA-A grad `grouped_lora_a_pair_grad_cpu_right(…, ctx.x_cpu.tensor, …)` (`:1287`).
1. Keep passing `packed` for the forward base/LoRA-A GEMMs (transient GPU), but **offload `hidden` (N×2048) instead of `packed`** at `:932`, and save `metadata.token_indices` (`moe.py:207`) to `ctx` (next to `offsets`,`experts` @ `:1086/1088`); thread `hidden` into the Function (`_forward_expert_activation_offload` @ `:2403` passes only `packed` today).
2. Backward (before `:1287`): transiently rebuild `packed_X = pack_tokens_contiguous(hidden, metadata)` into a **short-lived pinned** buffer (one gather, **bit-identical** to forward, no per-expert loop), feed `grouped_lora_a_pair_grad_cpu_right`, then free.
Saving = forward holds N×2048 not 8N×2048 ⇒ the `X` term drops ~8× **on the forward peak** ≈ **~113 GB at s10240**. The transient 8× rebuild (~2.7 GB/layer) is one-layer-at-a-time in *backward*, not on the forward peak.

Both keep grouped GEMMs and the CPU-source AsymGEMM kernels; combined ≈ **−56% of the expert stored set off the forward peak**, losslessly.

### Validate (correctness gate FIRST, then OFF→ON A/B). Test 2A and 2B **separately** (one flag at a time).
```bash
# correctness of the NEW path: run the suite with the flag(s) ON
CUDA_VISIBLE_DEVICES=0 ASYM_OFFLOAD_ACT_RECOMPUTE=1 .venv/bin/python -m pytest tests/training/test_lf_qwen3_asym_backend.py -q
CUDA_VISIBLE_DEVICES=0 ASYM_OFFLOAD_X_UNPACKED=1   .venv/bin/python -m pytest tests/training/test_lf_qwen3_asym_backend.py -q
# A/B at the SAME config: OFF (=0) then ON (=1). Diff loss (numerics), rss_peak (memory), fwd/bwd ms (latency).
for V in 0 1; do
  ASYM_OFFLOAD_ACT_RECOMPUTE=$V SEQ_LENS=8192 BACKEND_SPECS="asym_cpuadamwds|norecomp" GPU_POOL=0 bash scripts/lf/profile_lora_lf.sh
  echo "ACT_RECOMPUTE=$V:"; grep -i 'loss' "$RUN/train.log" | head -3   # + run the 3-metric block
done
# repeat the loop for ASYM_OFFLOAD_X_UNPACKED, and at SEQ_LENS=10240 for the memory/completion check.
```
### Accept iff
**Correctness:** 108/108 tests pass **and** step-0/1 loss within **2e-3** of baseline (else REJECT outright — it's not lossless). **Memory:** s10240 → `source_profile_written` or CPU rss_peak ≥ 40 GiB lower. **Latency:** s8192 (fwd+bwd) ≤ 5 % (it should be ~neutral; the index_select/silu are cheap). **Reject if** loss diverges, memory drop < 10 GiB, or latency > 10 %. (Do 2A and 2B as separate accept/reject decisions — 2A alone may be < 40 GiB = keep only if it stacks toward fitting; 2B is the headroom.)

---

## Lever 3 — HBM-budget watermark  (backstop; uses the idle 188 GB)
### Implementation
**Measured CPU-activation split (s8192 completed run, by tag — approximate, the profiler's per-tag counters overlap so treat as proportions):** roughly **~46% kernel-coupled (must stay pinned-CPU)** = expert `X`/`gate`/`up`/`act`/`S_*` (~37%) + attention-LoRA `q_proj.U`/`o_proj.U`/`*.S` (~9%) — the AsymGEMM CPU-source operands — and **~48% relocatable saved-tensor-hooks** = `decoder.saved.float32.[N×2048]` (residual stream) + `saved.float32/bf16.[attention]`. The **two single largest tensors are FP32** (the residual stream and an attention save) and are in the *relocatable* set — so this lever moves the **biggest** tensors to HBM, expert path untouched. (Earlier "~90% relocatable" was wrong; it's ~half.)
**Toggle = the budget env itself: `ASYM_ACT_CPU_BUDGET_BYTES` unset/`0` ⇒ the `_should_offload` budget branch is a no-op (OFF = today's behavior); a positive value (e.g. 150 GiB) turns it ON.** New shared helpers (mirror `_decoder_saved_tensor_min_bytes`, `decoder_activation_offload.py:37`): a process-global `_GLOBAL_CPU_OWNED_BYTES` + `_activation_cpu_budget_bytes()` from that env (returns ∞ when unset/0). Then:
- `decoder_activation_offload.py` `_should_offload` (`:159-179`): after the `min_bytes` check add `if _global_cpu_owned() + nbytes > budget: return False` (tensor stays on HBM — `_pack` returns it unchanged). Increment the global in `_pack` after the CPU copy (`:201`); decrement in `_unpack` on release (`:243`).
- Repeat in `attention_activation_offload.py` (`:243-263` / `:285` / `:325`).
- Leave the **kernel-coupled** paths on CPU: the expert `manager.offload` tags AND the attention-LoRA `AsymActivationOffloadLoRALinear` `U`/`S` — neither is a saved-hook, so neither reaches `_should_offload`.
Relocatable pool ≈ 48% of the ~615 GB s8192 activation set ≈ **~290 GB**; a 150 GB HBM budget absorbs enough to clear s10240 (linear: 1 GB HBM budget = 1 GB off CPU until the ~290 GB pool is exhausted).

### Validate
```bash
# sweep the budget; lower budget = more on HBM = less on CPU. Run s10240 each:
for B in 999999999999 700 500; do
  ASYM_ACT_CPU_BUDGET_BYTES=$((B*1024*1024*1024)) BACKEND_SPECS="asym_cpuadamwds|norecomp" SEQ_LENS=10240 GPU_POOL=0 bash scripts/lf/profile_lora_lf.sh
  echo "budget=${B}GiB:"; <run the 3-metric block: completion, CPU rss_peak, peak HBM>
done
# latency control at s8192 with the chosen budget vs baseline.
```
### Accept iff
s10240 → `source_profile_written` **and** CPU rss_peak drops by ≈ budget reduction (≥ 40 GiB) **and** **peak reserved HBM < 180 GiB** (no GPU OOM) **and** s8192 (fwd+bwd) ≤ 5 % (should be *faster* — skips H2D/D2H). **Reject if** GPU OOMs (HBM peak ≥ 188 GB), CPU drop < 10 GiB, or latency > 10 %. Pick the **largest** budget (most CPU) that still makes s10240 complete (keeps HBM minimal for the paper).

---

## Lever 4 — Temporal bounded-live window  (future; heavy — implement only if 1–3 insufficient)
### Implementation (sketch)
Add an eviction engine on a dedicated `torch.cuda.Stream`: `register_forward_hook` on each decoder layer (same modules as `lf.py:922-926`) hands its CPU handles to the evictor, which keeps W most-recent layers in CPU/HBM and migrates older layers' storage (swap `CPUActivationHandle.tensor` / `_SavedTensorOffloadHandle.tensor`); `register_full_backward_pre_hook` prefetches them back just-in-time. Tier = **HBM** (NVMe is ~10–20× too slow for the forward rate). Requires un-freezing `CPUActivationHandle` and building the copy-stream/double-buffer infra that does not exist today.
### Validate / Accept
Same harness; additionally vary `W` and require: s10240 completes, CPU rss_peak ≈ `W×~9 GiB`, and (fwd+bwd) ≤ 5 % (overlap must hide the eviction copies — if it stalls, **reject**). Defer unless 1–3 cannot reach the target sequence length.

---

## Recommended order  (updated 2026-06-16 with measured results)
1. ~~Lever 1 (interleave)~~ — REJECTED (membind already spans both nodes; no ceiling gain).
2. ~~Lever 2A (act recompute)~~ — TESTED: −47 GiB but **+36% latency → REJECTED**. **Lever 2B (X unpack)** — TESTED: −82 GiB but **+11% latency → borderline**; revisit only if the backward rebuild is optimized (pool the pinned buffer / GPU gather).
3. **Lever 3 (HBM budget) — ← TRY NEXT.** **Latency-neutral** (removes an H2D/D2H copy instead of adding backward compute), uses the idle 188 GB HBM. This is the lever that sidesteps the latency cost that sank 2A/2B.
4. Lever 4 — only for a much larger envelope.
Each is independently gated by the ACCEPT/REJECT rules: **keep only a meaningful (not trivial) memory win that does not blow up latency.**

## Other CPU-memory options (not levers in this doc; broader strategy in memory `asymgemm-core-research-goal`)
- **NVMe tier** (local `/dev/md0` RAID, ~11 TB free) — lossless capacity, but ~10–20× too slow for hot per-step activations → cold-tail only.
- **fp8 / quantized activation compression** — 2–4× effective capacity, but lossy (rejected — no approximations).
- **Partial recompute** — reduces stored activations but is the baselines' trick + adds compute (undermines the AsymGEMM contribution).
- **Smaller seq/batch** — s8192 already fits (~748 GiB); trivially avoids OOM but shrinks the demonstrable scale.
