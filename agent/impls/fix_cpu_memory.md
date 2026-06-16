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

**GLOBAL ACCEPT / REJECT (applies to every lever):**
- **ACCEPT only if BOTH:** (a) **memory win is meaningful** — s10240 goes **OOM→`source_profile_written`**, *or* CPU `rss_peak` drops **≥ 40 GiB**; **and** (b) **latency not blown up** — at the **s8192 control** (completes either way) `step.forward + step.backward` regress **≤ 5 %** vs the existing s8192 baseline.
- **REJECT if:** memory drop **< 10 GiB** (trivial), OR it still OOMs and rss_peak barely moves, OR (fwd+bwd) regression **> 10 %**.
- 5–10 % latency or 10–40 GiB memory = inconclusive → re-measure / tune before deciding.

---

## Lever 1 — NUMA interleave  (free; try first)
### Implementation
Edit `scripts/lf/run_lf_lora_sft.sh`:
- `:30-33` add a mode flag: `NUMACTL_MODE=${NUMACTL_MODE:-interleave}` (values `interleave|membind`).
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

## Lever 2 — Stop storing derivable activations  (lossless; the principled win)
### Implementation (`asym_gemm/training/qwen3_moe.py` `_ActivationOffloadQwen3ExpertFunction`; `moe.py`; `exp_act_offload_lora.py`)
**2A (do first, easy): drop `act`.** `act = silu(gate)·up` of already-stored `gate`,`up`.
1. In forward, keep the GPU `act` only through the `down_base` path; **remove** the `manager.offload(..., "act")` (`qwen3_moe.py:~1003`) so it does not persist to CPU. Do **not** put `act` on `ctx`.
2. In backward, before the down LoRA-A grad (`:~1170`), recompute `act = F.silu(gate)*up` on GPU from the staged `gate`,`up`; switch that call from the CPU-source kernel (`grouped_lora_a_grad_cpu_right`) to its **HBM-source** analog (operand now GPU).

**2B (bigger): unpack `X` (it is stored packed 8× = R×2048, R=8N).**
1. Stop offloading the packed input; instead offload the **pre-gather hidden `N×2048`** + carry `token_indices` (`ContiguousRouteMetadata`, `moe.py:207`). Add `token_indices` to `ctx.save_for_backward` (alongside `offsets`,`experts` at `:1086/1088`); thread `hidden` into the Function (currently only `packed` passed at `:2403`).
2. In backward, re-gather once on GPU: `X_gpu = hidden.index_select(0, token_indices)`; feed the gate/up LoRA-A grad via the **HBM-source** path (`grouped_lora_a_forward_hbm` analog) instead of `grouped_lora_a_pair_grad_cpu_right` (`exp_act_offload_lora.py:218`).

No per-expert loop; all backward GEMMs stay grouped. Forward gets cheaper (copy N×2048, not 8N×2048).

### Validate (correctness gate FIRST, then the harness)
```bash
# correctness: existing suite must stay green (exercises this Function fwd+bwd vs reference)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest tests/training/test_lf_qwen3_asym_backend.py -q
# numerics e2e: step-0/1 loss must match a pre-change run (bf16 tolerance). Run s8192 with & without:
SEQ_LENS=8192 BACKEND_SPECS="asym_cpuadamwds|norecomp" GPU_POOL=0 bash scripts/lf/profile_lora_lf.sh
grep -i 'loss' "$RUN/train.log" | head -3      # compare to the s8192 baseline's first losses
# memory + completion + latency: run s10240, then s8192, then the 3-metric block on each.
```
### Accept iff
**Correctness:** 108/108 tests pass **and** step-0/1 loss within **2e-3** of baseline (else REJECT outright — it's not lossless). **Memory:** s10240 → `source_profile_written` or CPU rss_peak ≥ 40 GiB lower. **Latency:** s8192 (fwd+bwd) ≤ 5 % (it should be ~neutral; the index_select/silu are cheap). **Reject if** loss diverges, memory drop < 10 GiB, or latency > 10 %. (Do 2A and 2B as separate accept/reject decisions — 2A alone may be < 40 GiB = keep only if it stacks toward fitting; 2B is the headroom.)

---

## Lever 3 — HBM-budget watermark  (backstop; uses the idle 188 GB)
### Implementation
New shared helpers (mirror `_decoder_saved_tensor_min_bytes`, `decoder_activation_offload.py:37`): a process-global `_GLOBAL_CPU_OWNED_BYTES` + `_activation_cpu_budget_bytes()` from env `ASYM_ACT_CPU_BUDGET_BYTES`. Then:
- `decoder_activation_offload.py` `_should_offload` (`:159-179`): after the `min_bytes` check add `if _global_cpu_owned() + nbytes > budget: return False` (tensor stays on HBM — `_pack` returns it unchanged). Increment the global in `_pack` after the CPU copy (`:201`); decrement in `_unpack` on release (`:243`).
- Repeat in `attention_activation_offload.py` (`:243-263` / `:285` / `:325`).
- Leave the expert CPU-kernel path (`X`/`gate`/`act`) on CPU (kernel-coupled).

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

## Recommended order
1. **Lever 1** (interleave) — free; run s10240, likely fits.
2. **Lever 2** (drop `act`, unpack `X`) — lossless, biggest principled reduction.
3. **Lever 3** (HBM budget) — backstop using idle HBM.
4. **Lever 4** — only for a much larger envelope.
Each is independently gated by the ACCEPT/REJECT rules above: **keep only a meaningful (not trivial) memory win that does not blow up latency.**
