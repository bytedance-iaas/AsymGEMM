# SM100 BF16 Qwen3 Gate/Up Backward: Down-LoRA-Aware Windowed W-Cache Plan

This is the implementation target for the V2 selected-recompute gate/up
backward path. It is a backward-only SM100 BF16 path for Qwen3-style MoE LoRA
SFT where selected routed rows use activation recompute.

The corrected design is not the earlier one-call `dact -> dgate/dup -> dX`
prototype. That prototype proves the local cache mechanism, but naive Python
integration leaves the full selected gate/up stream count at `2x -> 2x` because
Python still has to recompute gate/up once to build `act` for down-LoRA
backward.

The V2 implementation target is a down-LoRA-aware `cache_first_window` schedule:

```text
CPU W_gate_up window -> HBM w_cache
outside prework -> selected dact and dS_down
HBM w_cache -> selected gate/up recompute + act panel
act panel + dS_down -> selected dA_down
passed dact + gate/up panel -> dgate/dup
HBM w_cache -> selected gate/up base dX
discard/reuse w_cache for the next window
```

`lazy_direct_window` is only an optimization after the cache-first path is
correct and profiled. It must prove that it does not duplicate CPU weight
streams or spill hidden accumulator state.

## Scope

Included in this implementation:

```text
architecture: SM100 only
dtype: BF16 base weights and BF16 selected activations first
model path: Qwen3-style packed MoE experts
base weight: CPU-pinned HostWeight, shape [E, 2I, H]
gate/up packing: W_gate_up[:, :I, :] is gate, W_gate_up[:, I:, :] is up
rows: selected recompute rows only
math owned:
  selected gate/up base recompute
  selected gate/up LoRA-B contribution from provided low-rank S
  selected act panel production
  selected down-LoRA A gradient contribution, because dA_down needs act
  selected activation backward to dgate/dup
  selected gate/up base dX
outputs:
  grad_x_base_sel
  grad_gate_sel
  grad_up_sel
  grad_down_lora_A_sel
```

Outside this implementation:

```text
down base backward that produces dact_base
selected dS_down = dY B_down * lora_scale
selected dact_lora = dropout_backward(dS_down A_down, down_mask)
selected dact = dact_base + dact_lora
selected and nonselected down LoRA-B gradient
nonselected down LoRA backward
gate/up LoRA A/B gradients and LoRA dX
gate/up LoRA dX
nonselected gate/up base dX
routing scatter/combine and top-level gradient merges
optimizer state offload
forward-path AsymGEMM
generic MoE kernels
SM90, FP8, FP4, quantized weights
```

Dropout ownership:

```text
Native kernel does not own dropout RNG.
If lora_dropout == 0.0:
  Python may compute LoRA-A low-rank S from X before the native call; this does
  not stream base W_gate/W_up.
If lora_dropout > 0.0:
  Python must pass saved forward S_gate/S_up for selected rows.
If down LoRA dropout > 0.0:
  Python must pass the saved down-LoRA dropout mask for selected act rows.
Backward must not rerun dropout.
```

Do not expand V2 into unrelated full MLP backward fusion. The only extra
ownership relative to the legacy direct prototype is selected `dA_down`, because
that gradient needs the recomputed selected `act`. `dact_lora` and final `dact`
do not need gate/up/act and are computed outside and passed in for V2.

## Reviewed Inputs

This file incorporates the corrected design from:

```text
/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM/agent/fused_kernels/kernel_short.md
```

Implementation techniques were checked against:

```text
/home/shutianluo/kevin/AsymGEMM-SFT/third_party/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh
/home/shutianluo/kevin/AsymGEMM-SFT/third_party/DeepGEMM/deep_gemm/include/deep_gemm/layout/mega_moe.cuh
/home/shutianluo/kevin/AsymGEMM-SFT/third_party/DeepGEMM/csrc/jit_kernels/heuristics/mega_moe.hpp
/home/shutianluo/kevin/AsymGEMM-SFT/third_party/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh
/home/shutianluo/kevin/AsymGEMM-SFT/third_party/Megakernels/include/megakernel.cuh
/home/shutianluo/kevin/AsymGEMM-SFT/third_party/Megakernels/include/controller/controller.cuh
https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh
```

Borrowed implementation rules:

```text
prebuild selected row maps and per-expert row offsets before the hot path
schedule by expert waves rather than scanning routing metadata in hot loops
use one explicit aligned workspace sliced into typed views
keep CPU-W load, X/G load, MMA, activation epilogue, and dX as explicit phases
record stage timings and byte counters in machine-readable output
avoid clever persistent overlap until the simple phase schedule is measured
```

Source-derived decisions for this kernel:

```text
Use from DeepGEMM MegaMoE:
  per-expert selected-row counts cached before hot kernels
  expert-wave scheduling over precomputed M-block counts
  one aligned native workspace sliced into typed views
  padded row metadata for grouped GEMM compatibility
  explicit phase timing and byte counters
  static/host checks for supported BM/BN/BK shapes

Use from sm100_fp8_fp4_mega_moe.cuh:
  role separation between weight/load, token/X load, MMA, and epilogue work
  full/empty or launch-boundary barriers between producer/consumer phases
  early descriptor/config validation and alignment checks
  register/local-memory accounting as a first-order profiling signal

Use from Megakernels:
  controller/loader/consumer/storer separation only after phase-split V2A proves
  that launch/sync overhead is material
  instruction/timing style accounting for persistent variants

Do not copy:
  FP8/FP4 quantization layouts
  NVLink/symmetric-buffer dispatch machinery
  token combine/reduce logic
  TMA descriptor paths for CPU-pinned HostWeight loads
  global spin-wait protocols before phase-split cache_first_window passes Stage 7
  an instruction-VM/persistent kernel before Stage 7 proves it is needed
```

Concrete design implications from those sources:

```text
1. Precompute scheduling state, then keep hot kernels simple.
   DeepGEMM caches expert token counts and schedules by expert waves. V2 must
   build selected row maps, chunk offsets, padded metadata, and R_M before the
   timed GEMM phases. Stage 1 fails if kernels repeatedly scan routing metadata
   or allocate/index_select selected rows inside the window loop.

2. Separate roles before trying to overlap them.
   DeepGEMM separates dispatch/load, MMA, epilogue, and combine with explicit
   barriers. Megakernels separates loader, consumer, storer, launcher, and
   controller roles. V2A therefore starts as phase-split cache_first_window and
   uses NSYS to measure whether launch/sync overhead is actually the blocker.

3. Treat reuse mechanisms as measured resources.
   Megakernels pages shared memory with explicit request/release and timing
   records. V2A uses HBM w_cache instead of SMEM pages because CPU W panels are
   too large for per-CTA SMEM lifetime. Stage 8 may borrow the page idea only as
   seed_group_direct: keep the just-streamed W tile in a tiny SMEM/register
   window for one M_group, while still writing w_cache and keeping CPU stream
   multiplier <= 1.01.

4. Make workspace ownership auditable.
   DeepGEMM uses one layout workspace and compile-time/runtime layout checks.
   V2 must use one native workspace sliced into typed views. Stage 5 fails if
   allocator logs or CUDA memory snapshots show per-window allocation.

5. Register and local-memory pressure are first-order signals.
   Both external designs explicitly manage registers, shared memory, barriers,
   and timing. V2 cannot accept a faster CUDA-event result if NCU shows local
   memory spill, low tensor-pipe use, or unexplained HBM traffic.
```

This means V2A must remain a simple phase-split `cache_first_window` until Stage
7. Persistent fusion, role-separated CTAs/warps, ready flags, or instruction
pipelines are Stage 8+ work unless Stage 7 shows launch/sync overhead is the
specific blocker and the user approves continuing after the Stage 7 stop.

## Math

Forward context:

```text
gate = X W_gate^T + LoRA_gate
up   = X W_up^T   + LoRA_up
act  = silu(gate) * up
Y_down = act W_down^T + LoRA_down(act)
```

Before this native schedule starts, Python or the existing backward computes
the down base gradient and the selected down-LoRA activation gradient:

```text
dact_base = dY W_down

S_down = dropout_replay(act, down_mask) A_down^T
LoRA_down(act) = S_down B_down^T * lora_scale

dS_down = dY B_down * lora_scale
dact_lora = dropout_backward(dS_down A_down, down_mask)
dact = dact_base + dact_lora
dB_down += dY^T S_down * lora_scale
```

`dact_lora` and final `dact` do not depend on gate/up/act values. They are
outside prework. The selected down-LoRA term that still needs recomputed act is:

```text
dA_down_selected += dS_down^T dropout_replay(act, down_mask)
```

Boundary rule:

```text
Keep dact_lora outside native:
  dact_lora = dropout_backward((dY B_down * lora_scale) A_down, down_mask)
  This uses dY, B_down, A_down, and the down mask. It does not use W_gate,
  W_up, gate, up, or act.

Pass final dact into native:
  dact = dact_base + dact_lora
  Native reads dact_sel only for the current BM x P activation-backward tile.

Keep dA_down_selected inside native:
  dA_down_selected needs dropout_replay(act, down_mask), and act is exactly what
  the native gate/up recompute produces. Moving dA_down_selected outside would
  force a separate selected gate/up recompute and lose the 2x -> 1x CPU W stream
  reduction.
```

For every selected row `m`, expert `e`, and intermediate index `i`, the native
window schedule receives `dact[m, i]` and `dS_down[m, :]`, then computes:

```text
gate_base = dot(X_sel[m, :], W_gate_up_cpu[e, i, :])
up_base   = dot(X_sel[m, :], W_gate_up_cpu[e, I + i, :])

gate_lora = lora_scale * dot(gate_low_rank_sel[m, :], B_gate[e, i, :])
up_lora   = lora_scale * dot(up_low_rank_sel[m, :],   B_up[e, i, :])

gate = gate_base + gate_lora
up   = up_base   + up_lora
act  = silu(gate) * up

dA_down_selected[:, i] += dS_down[m, :] * dropout_replay(act, down_mask)

silu_gate = silu(gate)
dgate = dact[m, i] * up * silu_grad(gate)
dup   = dact[m, i] * silu_gate

dX_base[m, :] += dgate * W_gate_up_cpu[e, i, :]
dX_base[m, :] += dup   * W_gate_up_cpu[e, I + i, :]
```

Native outputs:

```text
grad_x_base_sel [M_selected, H]
grad_gate_sel   [M_selected, I]
grad_up_sel     [M_selected, I]
grad_down_lora_A_sel [E, r_down, I]
```

`grad_gate_sel` and `grad_up_sel` are returned so the existing LoRA backward can
compute gate/up LoRA A/B gradients and LoRA dX outside this native path.
`grad_down_lora_A_sel` must be merged with the existing nonselected down-LoRA A
gradient path. Down-LoRA B is computed outside from saved `S_down`; native does
not need `down_lora_B` or `down_low_rank_sel`.

## Tile Symbols

Use this notation everywhere:

```text
E:
  number of experts.

H:
  hidden dimension.

I:
  intermediate dimension per gate or up branch.

M_selected:
  selected recompute rows in compact selected-row order.

P:
  intermediate columns per gate or up recompute panel.
  Default P=32.

BN:
  paired gate/up columns per recompute panel.
  BN = 2P.
  Default BN=64.
  Not an independent API or CLI knob in V2.

Q:
  number of P-panels per dX window.
  Default Q=8.

W:
  intermediate columns per gate or up window.
  W = P * Q.
  Default W=256.

Kdx:
  paired gate/up columns in the selected dX window.
  Kdx = 2 * W = 2 * P * Q.
  Default Kdx=512.

BM:
  selected row tile.
  Default BM=64.

BK:
  hidden tile.
  In recompute: K-reduction over H.
  In dX: N-output tile over H.
  Default BK=512.

G_work:
  selected experts processed by one workspace chunk.
  Default G_work=128 if memory allows.

M_work:
  selected rows covered by the current expert chunk.

R_M:
  average selected M-block reuse factor:
    M_blocks_e = ceil(M_e / BM)
    R_M = weighted average M_blocks_e over active selected experts
```

Default shape:

```text
mode = cache_first_window
P = 32
Q = 8
W = 256
BN = 64
Kdx = 512
BM = 64
BK = 512
G_work = 128
```

For Qwen3 `I=768`, the default has:

```text
I / W = 768 / 256 = 3 windows
```

## Traffic And Memory Model

For Qwen3-30B-A3B:

```text
E = 128
H = 2048
I = 768
W_gate_up_cpu per layer = E * 2I * H * 2 bytes = 768 MiB
W_gate_up_cpu over 48 layers = 36 GiB per matrix-size full stream
```

The 36 GiB number is a lower bound. A normal M-tiled GEMM may reload the same W
tile once per selected M block. The timing model must include `R_M`.

For `M_selected=4096*8=32768`, `E=128`, and `BM=64`, uniform routing gives:

```text
average rows per expert = 32768 / 128 = 256
R_M = ceil(256 / 64) = 4
```

Workspace for one expert chunk:

```text
w_cache:
  [G_work, Kdx, H] bf16
  bytes = G_work * Kdx * H * 2

grad_pair_window:
  [M_work, Kdx] bf16
  bytes = M_work * Kdx * 2

dx_acc:
  [M_selected, H] fp32
  bytes = M_selected * H * 4

ready metadata for persistent fused mode only:
  w_tile_ready and grad_panel_ready bitsets or uint32 flags
  expected below 1 MiB for default Qwen3 shape
```

Qwen3 default memory:

```text
Q=4:  Kdx=256,  w_cache=128 MiB, grad_pair_window=16 MiB
Q=8:  Kdx=512,  w_cache=256 MiB, grad_pair_window=32 MiB
Q=12: Kdx=768,  w_cache=384 MiB, grad_pair_window=48 MiB
Q=24: Kdx=1536, w_cache=768 MiB, grad_pair_window=96 MiB

dx_acc fp32 = 32768 * 2048 * 4 = 256 MiB
optional bf16 grad_x_base_sel transient = 128 MiB
```

Default `Q=8` gross active memory:

```text
w_cache              256 MiB
grad_pair_window      32 MiB
dx_acc               256 MiB
optional bf16 output 128 MiB
ready metadata        <1 MiB
total gross          544-672 MiB
```

Matrix-size lower-bound traffic over 48 layers:

```text
current selected recompute + selected dX:
  CPU read: 72 GiB

cache_first_window:
  CPU read:  36 GiB
  HBM write: 36 GiB
  HBM read:  72 GiB

lazy_direct_window best case:
  CPU read:  36 GiB
  HBM write: 36 GiB
  HBM read:  36 GiB
```

Tile-model timing traffic:

```text
current selected recompute + selected dX:
  CPU read ~= 2 * R_M * 36 GiB

cache_first_window:
  CPU read  = 1 * 36 GiB
  HBM write = 1 * 36 GiB
  HBM read ~= 2 * R_M * 36 GiB

lazy_direct_window best case:
  CPU read  = 1 * 36 GiB
  HBM write = 1 * 36 GiB
  HBM read ~= 1 * R_M * 36 GiB
```

With `R_M=4`, cache-first adds about `4 * 36 = 144 GiB` extra recompute-side
HBM read relative to best-case lazy direct. At 2-3 TiB/s idealized HBM
bandwidth that is about 48-72 ms. Profiling must measure physical HBM bytes and
L2 hit rate; lower-bound bytes are not enough to claim a win.

## Implementation Modes

### V2A: down_lora_aware_cache_first_window

Required integration implementation:

```text
fill_w_cache:
  CPU W_gate_up stream -> HBM w_cache

recompute_down_lora_activation_tile:
  for each P panel and BM row tile:
    read HBM w_cache
    recompute gate/up base
    add gate/up LoRA-B deltas
    compute act = silu(gate) * up
    accumulate selected grad_down_lora_A from act and passed dS_down
    compute dgate/dup from passed dact and gate/up
    write grad_pair_window
    write grad_gate_sel/grad_up_sel

dx_window:
  read grad_pair_window
  read HBM w_cache
  accumulate selected base dX into dx_acc
```

Properties:

```text
CPU stream multiplier must be <= 1.01.
Recompute is allowed and expected to read w_cache.
Python must not perform an earlier selected gate/up base recompute for the same
rows and then call this path, because that would change the full selected stream
accounting from the current 2x selected CPU stream into 2x -> 2x instead of
the desired 2x -> 1x.
Normal M-tiled GEMM parallelism is preserved.
This is the correctness and integration baseline.
```

Tile-level principle:

```text
Only w_cache and grad_pair_window need full W-window lifetime.
gate, up, and act must be BM x P or BM x 2P tile-local in the latency-gated
path.
Full-window gate/up/act tensors are allowed only as temporary validation debug
views and must report their extra HBM bytes separately.
```

### Legacy Direct-Op Prototype: dact_input_cache_first_window

This is useful only for op-level proof of the cache mechanism:

```text
input dact is already known
CPU W_gate_up -> w_cache
recompute gate/up from w_cache
dgate/dup from dact and gate/up
dX from grad_pair_window and w_cache
```

It must not be used to claim the integrated selected gate/up stream saving by
itself. The legacy direct op takes final `dact` but does not compute selected
`dA_down`. If Python computes selected `dA_down` outside, it still needs
selected `act`, so a naive one-call integration performs:

```text
old Python selected recompute: W -> gate/up -> act -> dA_down
direct native op:              W -> w_cache -> gate/up -> dgate/dup -> dX
```

That is `2x -> 2x` for the full selected gate/up stream, even though the direct
op itself has `cpu_weight_stream_multiplier == 1.0`.

### V2B: seed_group_direct

Optional after V2A:

```text
while streaming a W tile from CPU:
  write it to w_cache
  also compute one M_group directly from the just-streamed tile
remaining M groups:
  read w_cache like V2A
```

Properties:

```text
CPU stream multiplier must remain <= 1.01.
Only valid inside a persistent fused op or CTA/cluster design.
Not available in the simple phase-split validation path.
Must beat same-shape cache_first_window in full-step latency.
```

### V2C: all_rows_direct

Optional research path after V2A and V2B:

```text
one cooperative work unit covers all rows for an expert/panel
CPU W tile is consumed directly for all rows while also written to w_cache
dX still reads w_cache
```

Reject if:

```text
cpu_weight_stream_multiplier > 1.05
pair_acc spills to local/global memory enough to erase the HBM-read saving
occupancy collapses from huge expert/panel work units
full-step latency is not better than V2A and V2B
```

Do not implement V2B or V2C before V2A is correct and profiled.

## Native API

Add one V2 pybind-visible integration function:

```python
(
  grad_x_base_sel,
  grad_gate_sel,
  grad_up_sel,
  grad_down_lora_A_sel,
  stats,
) = asym_gemm.qwen3_gate_up_down_lora_bwd_sm100_bf16_windowed(
    x_sel,                 # cuda bf16 [M_selected, H], contiguous
    dact_sel,              # cuda bf16 [M_selected, I], dact_base + dact_lora
    dS_down_sel,           # cuda bf16/fp32 [M_selected, r_down], dY B_down * scale
    gate_low_rank_sel,     # cuda bf16 [M_selected, r], contiguous
    up_low_rank_sel,       # cuda bf16 [M_selected, r], contiguous
    gate_lora_B,           # cuda bf16 [E, I, r], contiguous
    up_lora_B,             # cuda bf16 [E, I, r], contiguous
    down_mask_packed_sel,  # cuda uint8 [M_selected, ceil(I/8)] or empty
    gate_up_weight_cpu,    # cpu pinned bf16 [E, 2I, H], contiguous
    selected_offsets,      # cuda int32/int64 [Gs + 1], cumulative
    selected_experts,      # cuda int32/int64 [Gs + 1], original expert ids, sentinel -1
    p: int = 32,
    q: int = 8,
    bm: int = 64,
    bk: int = 512,
    g_work: int = 128,
    lora_scale: float = 1.0,
    down_lora_dropout_p: float = 0.0,
    mode: str = "cache_first_window",
    return_stats: bool = True,
)
```

This function is down-LoRA-aware because it computes selected
`grad_down_lora_A_sel` from recomputed `act`. It does not compute selected
`dact_lora`, final `dact`, or `grad_down_lora_B`; those are outside prework or
outside gradient reductions that do not require recomputing gate/up/act.

The existing direct-op prototype may remain as a debug validation helper:

```python
grad_x_base_sel, grad_gate_sel, grad_up_sel, stats = asym_gemm.qwen3_gate_up_recompute_bwd_sm100_bf16_windowed(...)
```

It is not the V2 integration target because it takes final `dact_sel` as
input but does not compute selected `dA_down`. Without selected `dA_down`
inside the same gate/up recompute, Python would still need an earlier selected
gate/up recompute to produce `act`.

Return empty tensors if `M_selected == 0` or `Gs == 0`. `stats` must still be
returned with `selected_recompute_rows == 0` and `cpu_weight_bytes_staged == 0`
when `return_stats=True`.

C++ hard checks:

```text
current device arch major == 10
all CUDA tensors are on the same CUDA device
x_sel, dact_sel, gate_low_rank_sel, up_low_rank_sel are CUDA BF16
dS_down_sel is CUDA BF16 or FP32
gate_lora_B and up_lora_B are CUDA BF16
down_mask_packed_sel is CUDA uint8 or empty
gate_up_weight_cpu is CPU BF16, contiguous, pinned
gate_up_weight_cpu.shape == [E, 2I, H]
gate_lora_B.shape == up_lora_B.shape == [E, I, r]
dS_down_sel.shape == [M_selected, r_down]
selected_offsets.numel() == selected_experts.numel()
selected_experts[-1] == -1
p > 0
q > 0
bm > 0
bk > 0
2 * p is a supported BN candidate, default 64
bm is a supported selected-row tile candidate, default 64
bk is a supported hidden tile candidate, default 512
kdx = 2 * p * q
kdx <= 2 * I
mode in {"cache_first_window", "seed_group_direct", "all_rows_direct"}
mode != "cache_first_window" requires explicit env flag:
  ASYM_QWEN3_GATE_UP_ENABLE_LAZY_DIRECT=1
```

Selected metadata contract:

```text
selected_offsets and selected_experts describe compact selected rows only.
selected_offsets dtype may be int32 or int64 on input, but native metadata uses int32.
selected_offsets.numel() == selected_experts.numel() == Gs + 1.
selected_offsets[0] == 0.
selected_offsets is monotonically nondecreasing.
selected_offsets[-1] == M_selected.
selected_experts[0:Gs] are original expert ids in [0, E).
selected_experts[-1] == -1 sentinel.
Groups with zero selected rows must be excluded unless the case is Gs == 0.
For Gs == 0:
  selected_offsets == [0]
  selected_experts == [-1]
  M_selected == 0
Rows are in compact selected-row order; selected_offsets[g]:selected_offsets[g+1]
is the contiguous selected-row span for selected_experts[g].
```

## Files To Add Or Edit

Add:

```text
csrc/apis/qwen3_moe.hpp
csrc/jit_kernels/impls/sm100_qwen3_gate_up_windowed_bwd.hpp
asym_gemm/include/asym_gemm/impls/sm100_qwen3_gate_up_windowed_bwd.cuh
scripts/lf/validate_qwen3_gate_up_windowed_bwd.py
scripts/lf/profile_qwen3_gate_up_windowed_ops.sh
scripts/lf/profile_lora_lf_fused.sh
tests/training/test_qwen3_gate_up_windowed_bwd.py
agent/fused_kernels/kernels_progress.md
```

Edit:

```text
csrc/python_api.cpp
  include csrc/apis/qwen3_moe.hpp
  optionally register qwen3_gate_up_recompute_bwd_sm100_bf16_windowed as a
  debug direct-op prototype
  register qwen3_gate_up_down_lora_bwd_sm100_bf16_windowed for V2 integration

asym_gemm/__init__.py
  add both qwen3 native backward APIs to _maybe_import_from_C

csrc/apis/gemm.hpp
  add or expose an internal SM100 BF16 grouped GEMM helper with fp32 accumulate
  for the selected dX window

csrc/jit_kernels/impls/sm100_bf16_asym_gemm.hpp
  reuse tile config/runtime where possible; do not fork unrelated behavior

asym_gemm/training/qwen3_moe.py
  call the V2 down-LoRA-aware native function from
  _ThresholdedQwen3ExpertFunction.backward for selected recompute rows only

scripts/lf/profile_lora_lf.sh
  preserve current profiling entrypoint; add native stats to profile JSON

scripts/lf/profile_lora_lf_fused.sh
  add if absent, otherwise edit; optional E2E wrapper after op-level validation
  passes; do not use it as the primary NCU workflow
```

Use existing style from:

```text
csrc/jit_kernels/impls/sm100_bf16_asym_gemm.hpp
asym_gemm/include/asym_gemm/impls/sm100_bf16_asym_gemm.cuh
csrc/python_api.cpp
```

## Progress Ledger

All implementation, validation, profiling, and decision results must be recorded
in:

```text
agent/fused_kernels/kernels_progress.md
```

Rules:

```text
Append or update the progress ledger after every stage attempt.
Do not rely only on terminal output, chat history, or profile directories.
Every entry must include:
  stage name
  status: not_started, running, passed, failed, blocked, or stopped_for_review
  git/worktree note for files changed in the stage
  exact commands run
  validation.json path
  validation.md path
  latency.csv path
  NSYS/NCU artifact paths when applicable
  key correctness metrics
  key latency metrics
  key memory/traffic metrics
  explicit next action

If a stage fails, write the failed metric, likely cause, and proposed next
kernel/doc/script change before continuing.
```

The Stage 7 entry is special. After the Stage 7 aggregate gate is written, the
agent must stop and return control to the user. Do not start Stage 8, Stage 9,
or Stage 10 unless the user explicitly approves continuing after reviewing
`kernels_progress.md`.

## Per-Stage File And Function Map

This is the audit map for what each milestone is allowed to add or modify.
If a stage changes a file or function not listed here, the validation report
for that stage must call it out explicitly with the reason.

Each stage validation artifact must include:

```text
changed_files_since_previous_stage
changed_functions_since_previous_stage
new_public_apis_since_previous_stage
validation_commands_run
latency_or_ncu_artifacts_checked
```

### Stage 0: Baseline Instrumentation

Edit existing files and add/scaffold new validation files:

```text
asym_gemm/training/qwen3_moe.py
  _ThresholdedQwen3ExpertFunction.backward
  existing profile range/counter helpers used by qwen3_moe.py

asym_gemm/training/profile_ranges.py or asym_gemm/training/profiling.py
  only if current counters cannot express the needed selected-region ranges

scripts/lf/validate_qwen3_gate_up_windowed_bwd.py
  add/scaffold if absent
  stage0_baseline_instrumentation

tests/training/test_qwen3_gate_up_windowed_bwd.py
  add/scaffold if absent
  baseline counter and selected-region unit tests
```

Do not add native kernels in Stage 0.

### Stage 1: Metadata And W-Cache Fill

Add or edit:

```text
csrc/apis/qwen3_moe.hpp
  metadata_fill debug/test entry if direct validation needs it
  qwen3 selected-row metadata conversion helpers

csrc/jit_kernels/impls/sm100_qwen3_gate_up_windowed_bwd.hpp
  metadata/fill launch wrappers

asym_gemm/include/asym_gemm/impls/sm100_qwen3_gate_up_windowed_bwd.cuh
  sm100_qwen3_fill_row_experts
  sm100_qwen3_build_chunk_metadata
  sm100_qwen3_fill_gate_up_w_cache_bf16

csrc/python_api.cpp
  temporary private debug binding only if direct validation needs returned w_cache

scripts/lf/validate_qwen3_gate_up_windowed_bwd.py
  op=metadata_fill

tests/training/test_qwen3_gate_up_windowed_bwd.py
  metadata reference tests
  w_cache byte/order/layout tests
```

### Stage 2: Recompute Gate/Up And Act Tile

Add or edit:

```text
csrc/apis/qwen3_moe.hpp
  recompute_act direct/test entry

csrc/jit_kernels/impls/sm100_qwen3_gate_up_windowed_bwd.hpp
  recompute_gate_up_act_tile launcher

asym_gemm/include/asym_gemm/impls/sm100_qwen3_gate_up_windowed_bwd.cuh
  sm100_qwen3_gate_up_act_tile_bf16
  optional fused GEMM+act epilogue only after simple path passes

csrc/apis/gemm.hpp
  reuse or expose grouped BF16 GEMM helper reading CUDA w_cache

scripts/lf/validate_qwen3_gate_up_windowed_bwd.py
  op=recompute_act

tests/training/test_qwen3_gate_up_windowed_bwd.py
  gate/up recompute tests
  LoRA-B delta tests
  act=silu(gate)*up tests
```

### Stage 3: Selected Down-LoRA A And Activation Tile

Add or edit:

```text
csrc/apis/qwen3_moe.hpp
  selected_down_lora_A_activation direct/test entry

csrc/jit_kernels/impls/sm100_qwen3_gate_up_windowed_bwd.hpp
  selected_down_lora_A_activation_tile launcher

asym_gemm/include/asym_gemm/impls/sm100_qwen3_gate_up_windowed_bwd.cuh
  sm100_qwen3_selected_down_lora_A_activation_tile_bf16
  or fused epilogue code in sm100_qwen3_gate_up_act_tile_bf16

scripts/lf/validate_qwen3_gate_up_windowed_bwd.py
  op=down_lora_activation

tests/training/test_qwen3_gate_up_windowed_bwd.py
  passed dS_down input tests
  down dropout replay tests
  selected grad_down_lora_A tests
  dgate/dup and grad_pair_window tests
```

### Stage 4: Selected dX Window

Add or edit:

```text
csrc/apis/gemm.hpp
  fp32-accumulate helper for selected dX

csrc/jit_kernels/impls/sm100_qwen3_gate_up_windowed_bwd.hpp
  dx_window launcher

asym_gemm/include/asym_gemm/impls/sm100_qwen3_gate_up_windowed_bwd.cuh
  selected dX wrapper or kernel glue

csrc/jit_kernels/impls/sm100_bf16_asym_gemm.hpp
  only if existing grouped GEMM needs explicit accumulation-mode support

scripts/lf/validate_qwen3_gate_up_windowed_bwd.py
  op=dx_window

tests/training/test_qwen3_gate_up_windowed_bwd.py
  grad_x_base_sel reference tests
  no double-count tests
```

### Stage 5: V2 Native Direct End-To-End

Add or edit:

```text
csrc/apis/qwen3_moe.hpp
  qwen3_gate_up_down_lora_bwd_sm100_bf16_windowed host API
  optional qwen3_gate_up_recompute_bwd_sm100_bf16_windowed debug API only if
  needed to preserve the legacy direct-op prototype

csrc/python_api.cpp
  public pybind qwen3_gate_up_down_lora_bwd_sm100_bf16_windowed

asym_gemm/__init__.py
  expose qwen3_gate_up_down_lora_bwd_sm100_bf16_windowed through _maybe_import_from_C

csrc/jit_kernels/impls/sm100_qwen3_gate_up_windowed_bwd.hpp
  full window/expert-chunk orchestration

scripts/lf/validate_qwen3_gate_up_windowed_bwd.py
  op=native_e2e

tests/training/test_qwen3_gate_up_windowed_bwd.py
  full V2 native torch-reference tests for grad_x_base_sel, grad_gate_sel,
  grad_up_sel, and grad_down_lora_A_sel
```

### Stage 6: Python Integration Dropout 0.00 And 0.10

Edit:

```text
asym_gemm/training/qwen3_moe.py
  _ThresholdedQwen3ExpertFunction.backward native selected path
  compute selected dS_down, dact_lora, final dact, and grad_down_lora_B outside native
  compact selected dact/dS_down/down dropout mask inputs
  selected saved S_gate/S_up inputs when gate/up LoRA dropout > 0
  call the V2 down-LoRA-aware native API, not the direct-op prototype
  selected/nonselected row split
  scatter grad_x_base_sel, grad_gate_sel, and grad_up_sel
  merge selected native grad_down_lora_A with nonselected down-LoRA A grads
  existing gate/up LoRA backward call ownership

scripts/lf/validate_qwen3_gate_up_windowed_bwd.py
  stage6_python_integration_drop000
  stage6_python_integration_drop010

tests/training/test_lf_qwen3_asym_backend.py or tests/training/test_qwen3_gate_up_windowed_bwd.py
  dropout-0 integration tests
  dropout-0.10 saved-S and saved-down-mask replay tests
  row-ownership and grad checks
```

Do not pass RNG state into native C++ in Stage 6. Passing saved packed down
dropout masks is allowed only for replay; native must not sample dropout.

### Stage 7: NSYS/NCU Cache-First Profile Gate

Add or edit:

```text
scripts/lf/validate_qwen3_gate_up_windowed_bwd.py
  profile-mode NSYS/NCU child runners
  NCU CSV parser
  Stage 7 gate checker

scripts/lf/profile_qwen3_gate_up_windowed_ops.sh
  op-level correctness, CUDA-event timing, NSYS, and NCU sweep

asym_gemm/training/qwen3_moe.py
  native stats export to profile JSON if missing

scripts/lf/postprocess_lf_profile_artifacts.py
  only if needed to preserve native stats in summaries
```

Stage 7 may make bounded `cache_first_window` kernel fixes and tile/scheduling
tuning when NCU or timing gates fail. It must not introduce new public API
shape, new Python row-ownership behavior, or new lazy/direct modes. New
`seed_group_direct` and `all_rows_direct` work belongs to Stage 8 and Stage 9.

### Stage 8: Optional seed_group_direct

Add or edit:

```text
asym_gemm/include/asym_gemm/impls/sm100_qwen3_gate_up_windowed_bwd.cuh
  seed_group_direct persistent/CTA-cluster variant

csrc/jit_kernels/impls/sm100_qwen3_gate_up_windowed_bwd.hpp
  mode dispatch and launcher

csrc/apis/qwen3_moe.hpp
  mode validation and stats

scripts/lf/validate_qwen3_gate_up_windowed_bwd.py
  mode=seed_group_direct gates

scripts/lf/profile_qwen3_gate_up_windowed_ops.sh
  mode sweep support
```

Stage 8 must not edit Python row-ownership logic except to pass mode/env config.

### Stage 9: Optional all_rows_direct

Add or edit:

```text
asym_gemm/include/asym_gemm/impls/sm100_qwen3_gate_up_windowed_bwd.cuh
  all_rows_direct variant

csrc/jit_kernels/impls/sm100_qwen3_gate_up_windowed_bwd.hpp
  mode dispatch and launcher

csrc/apis/qwen3_moe.hpp
  mode validation and stats

scripts/lf/validate_qwen3_gate_up_windowed_bwd.py
  mode=all_rows_direct gates
  spill and CPU-multiplier rejection checks

scripts/lf/profile_qwen3_gate_up_windowed_ops.sh
  mode sweep support
```

Stage 9 must not merge if CPU W is re-streamed per `M_group` or if `pair_acc`
spills enough to erase the traffic win.

### Stage 10: LF 50-Step End-To-End Gate

Add or edit:

```text
scripts/lf/profile_lora_lf_fused.sh
  set ASYM_QWEN3_GATE_UP_WINDOWED_BWD env vars
  include native mode/P/Q/BM/BK/G_work labels in output paths

scripts/lf/profile_lora_lf.sh
  only if the fused wrapper delegates to it and needs env passthrough

scripts/lf/validate_qwen3_gate_up_windowed_bwd.py
  stage10_lf_50step profile-root validator

scripts/lf/postprocess_lf_profile_artifacts.py
  only if needed to surface native counters in combined profile JSON
```

No new kernel features may be introduced in Stage 10. Kernel iteration belongs
back in Stage 7, Stage 8, or Stage 9 depending on which mode changed.

## Internal Workspaces

Allocate inside the native function:

```text
w_cache              cuda bf16 [G_work, Kdx, H]
grad_pair_window     cuda bf16 [M_work, Kdx]
dx_acc               cuda fp32 [M_selected, H], zero once
grad_gate_sel        cuda bf16 [M_selected, I]
grad_up_sel          cuda bf16 [M_selected, I]
dS_down_sel          cuda bf16/fp32 [M_selected, r_down], input
grad_down_lora_A_sel cuda bf16/fp32 [E or G_work, r_down, I], output/accumulator
pair_acc_tile        logical fp32 [BM, 2*P], register/shared/tensor-memory scoped
gate_up_act_tile     logical fp32/bf16 [BM, P] views, tile-local only
source_offsets       cuda int32 [Gs + 1]
source_experts       cuda int32 [Gs + 1]
compact_experts      cuda int32 [G_work + 1]
pair_offsets         cuda int32 [2 * G_work]
row_experts          cuda int32 [M_selected]
chunk_row_ids        cuda int32 [M_work], maps chunk-local rows to compact selected rows
row_to_chunk_local   cuda int32 [M_selected], maps compact selected rows to chunk-local rows
row_to_local_expert  cuda int32 [M_selected], allocate only if a kernel indexes
                     expert-local row ids; otherwise omit it and report 0 bytes
stats_buffer         cuda/host counters
```

Workspace rules:

```text
No [E, Kdx, H] scratch.
No per-layer persistent W cache.
No per-window [M_selected, H] temporary.
No full-window gate/up/act [M_work, W] tensors in the latency-gated path.
Only w_cache and grad_pair_window are allowed to have full W-window lifetime.
No torch.cat or index_select for W staging.
No Python loop over windows.
No selected base dX in the old path.
No native dropout RNG or dropout sampling; saved masks are read-only replay
inputs only.
```

`G_work` batching:

```text
Default: G_work=min(128, Gs) for Qwen3.
If Gs > G_work, loop over expert chunks inside the native function.
For each chunk:
  w_cache is [G_work, Kdx, H]
  grad_pair_window covers only M_work chunk-local rows for that chunk
  chunk_row_ids maps grad_pair_window local row ids back to compact selected rows
  row_to_chunk_local maps compact selected rows into grad_pair_window local row ids
  dx_acc remains full [M_selected, H]
```

Chunk-local row indexing rule:

```text
Use compact selected row ids for:
  X_sel
  dact_sel
  dS_down_sel
  gate_low_rank_sel/up_low_rank_sel
  down_mask_packed_sel
  grad_gate_sel/grad_up_sel
  grad_down_lora_A_sel
  dx_acc

Use chunk-local row ids only for:
  grad_pair_window

For every expert in the current chunk:
  rows_e_global = selected compact row ids for expert e
  rows_e_local = row_to_chunk_local[rows_e_global]

Do not index grad_pair_window with global compact row ids unless
grad_pair_window is deliberately allocated as [M_selected, Kdx] and the memory
model is updated. The V2 cache budget assumes chunk-local [M_work, Kdx].
```

Grouped-GEMM padding contract:

```text
Existing contiguous grouped GEMM paths pad each group to a block-M multiple and
use pair offsets plus a sentinel expert list. V2 native code must own this
contract inside C++/CUDA; Python must not call _pad_grouped_input_for_asym or
unpad tensors around this path.

Build these metadata views for each expert chunk:
  chunk_offsets_unpadded  int32 [G_chunk + 1]
  chunk_offsets_padded    int32 [G_chunk + 1]
  pair_offsets_padded     int32 [2 * G_chunk]
  grouped_experts         int32 [G_chunk + 1], with trailing -1 sentinel
  grouped_list_size       int32 scalar, equal to G_chunk + 1
  padded_to_global_row    int32 [M_padded_work], -1 for padding rows
  padded_to_local_row     int32 [M_padded_work], -1 for padding rows

Definitions:
  G_chunk <= G_work
  M_work = sum valid selected rows in the expert chunk
  M_padded_work = sum_e ceil(M_e / BM) * BM

Accepted V2 path:
  grouped GEMM loaders may read padded_to_global_row/padded_to_local_row and
  skip padding rows, or write zero for padding rows.
  grad_pair_window remains unpadded [M_work, Kdx].
  dx_acc, grad_gate_sel, and grad_up_sel are written only for valid compact
  selected rows.

Rejected for latency-gated V2:
  Python-side padding/unpadding.
  Materialized padded X or padded grad_pair tensors unless marked debug-only and
  excluded from Stage 7/Stage 10 pass criteria.
```

## Runtime Phase Schedule

The V2 C++ function owns all selected loops and keeps the window cache live
until selected `dA_down`, activation backward, and dX finish:

```text
qwen3_gate_up_down_lora_bwd_sm100_bf16_windowed(...)
  validate inputs
  convert selected_offsets/selected_experts to int32 CUDA metadata
  build row_experts once
  zero dx_acc
  zero selected grad_down_lora_A accumulator

  for iwin0 in 0..I step W:
    for expert_chunk in selected experts step G_work:
      Phase A0: fill_w_cache
      Phase A1: recompute_down_lora_activation_tile
      Phase B:  dx_window

  cast/store grad_x_base_sel
  return grad_x_base_sel, grad_gate_sel, grad_up_sel,
         grad_down_lora_A_sel, stats
```

For default Qwen3 `I=768`, `P=32`, `Q=8`, `G_work=128`, the phase-split
validation path has:

```text
3 windows * 1 expert chunk * 3 phase kernels = 9 launches per layer
48 layers -> 432 launches
```

Captured launch overhead target:

```text
captured_phase_launch_overhead_ms <= 5 ms over all selected layers for the
Qwen3-30B-A3B projected step, or <= 2 percent of native_selected_region_ms.
```

Uncaptured Python launch overhead can be larger; the validation must measure it.
A production implementation must collapse phases into a graph-captured native
orchestration or persistent kernel if NSYS shows launch/runtime overhead exceeds
either threshold above.

### Phase 0: Metadata

Kernels:

```text
sm100_qwen3_fill_row_experts
sm100_qwen3_build_chunk_metadata
```

Inputs:

```text
selected_offsets [Gs + 1]
selected_experts [Gs + 1]
```

Outputs:

```text
row_experts [M_selected]
source_offsets/source_experts int32 CUDA views:
  alias the caller tensors if they are contiguous int32 CUDA tensors
  otherwise create int32 CUDA copies before the timed hot region and report the
  copy bytes/time separately
pair_offsets for grouped GEMM
padded pair offsets/list size for contiguous grouped GEMM compatibility
compact_experts for w_cache
padded_to_global_row and padded_to_local_row maps
tokens_per_expert_histogram
sum_m_blocks
R_M
```

Correctness:

```text
row_experts[row] == selected_experts[g] for offsets[g] <= row < offsets[g+1]
pair_offsets match compact selected row spans for the expert chunk
pair_offsets_padded match ceil(M_e / BM) * BM padded spans
padded_to_global_row == -1 only for padding rows
sum_m_blocks == sum_e ceil(M_e / BM)
```

Latency/profiling:

```text
metadata_ms must be reported
metadata_ms < 1 percent of direct native time for qwen_shape_routed
no CPU copy of expert list or offsets in the hot path
```

### Phase A0: Fill W Cache

Kernel:

```text
sm100_qwen3_fill_gate_up_w_cache_bf16
```

Logical loop:

```text
for e_rel, e in expert_chunk:
  for n0 in 0..Kdx step BN:
    q_panel = n0 / BN
    i0 = iwin0 + q_panel * P
    valid_i = clamp(I - i0, 0, P)

    for h0 in 0..H step BK:
      Wg = stream_cpu(W_gate_up_cpu[e, i0:i0+valid_i, h0:h0+BK])
      Wu = stream_cpu(W_gate_up_cpu[e, I+i0:I+i0+valid_i, h0:h0+BK])
      W_tile = concat_and_zero_pad(Wg, Wu, P)
      w_cache[e_rel, n0:n0+BN, h0:h0+BK] = W_tile
```

Implementation requirements:

```text
linearize over G_work * Kdx * H
vectorize H with 16-byte loads/stores when aligned
zero-fill invalid last-window columns
count exact bytes copied from CPU and written to HBM
do not call torch.cat, torch.index_select, or per-expert cudaMemcpyAsync
```

Correctness:

```text
for every valid gate column:
  w_cache[e_rel, n0 + j, h] == W_gate_up_cpu[e, i0 + j, h]
for every valid up column:
  w_cache[e_rel, n0 + P + j, h] == W_gate_up_cpu[e, I + i0 + j, h]
invalid columns are zero
```

Latency/profiling:

```text
fill_w_cache_ms reported per case and per window
cpu_weight_bytes_staged == expected_cpu_weight_bytes_min
hbm_w_cache_valid_write_bytes == expected_cpu_weight_bytes_min
hbm_w_cache_write_bytes ==
  hbm_w_cache_valid_write_bytes + hbm_w_cache_padding_zero_write_bytes
cpu_weight_stream_multiplier <= 1.01
NCU or internal counters confirm no second CPU read in this phase
```

### Phase A1: Recompute, Down-LoRA A, And Activation Backward

This phase is the V2 correction. It must produce selected `act`,
`grad_down_lora_A_sel`, and `grad_pair_window` before `w_cache` is discarded,
without forcing Python to recompute selected gate/up. Final `dact` is passed in;
native does not compute `dact_lora`.

The desired latency-gated implementation is a grouped GEMM plus a custom tile
epilogue. A temporary validation implementation may split the GEMM and epilogue
into separate native kernels, but it must not allocate full-window gate/up/act
or dact tensors for the Stage 7 pass.

Per `P` panel and `BM` row tile:

```text
for e_rel, e in expert_chunk:
  rows_e_global = selected compact row ids for e
  rows_e_local = row_to_chunk_local[rows_e_global]

  for n0 in 0..Kdx step BN:
    q_panel = n0 / BN
    i0 = iwin0 + q_panel * P
    valid_i = clamp(I - i0, 0, P)

    if valid_i == 0:
      zero-fill grad_pair_window[rows_e_local, n0:n0+BN]
      continue

    for me0 in 0..M_e step BM:
      rows_global = rows_e_global[me0:me0+BM]
      rows_local = rows_e_local[me0:me0+BM]

      pair_acc = zeros([len(rows_global), BN], fp32)
      for h0 in 0..H step BK:
        W_tile = load_hbm(w_cache[e_rel, n0:n0+BN, h0:h0+BK])
        X_tile = X_sel[rows_global, h0:h0+BK]
        pair_acc += X_tile @ W_tile^T

      gate_base_p = pair_acc[:, 0:P]
      up_base_p   = pair_acc[:, P:2*P]

      gate_lora_p = zero_pad(
          gate_low_rank_sel[rows_global] @ gate_lora_B[e, i0:i0+valid_i].T, P
      )
      up_lora_p = zero_pad(
          up_low_rank_sel[rows_global] @ up_lora_B[e, i0:i0+valid_i].T, P
      )

      gate_p = gate_base_p + gate_lora_p
      up_p = up_base_p + up_lora_p
      act_p = silu(gate_p) * up_p

      dS_p = dS_down_sel[rows_global, :]
      mask_valid = down_mask_slice(
          down_mask_packed_sel, rows_global, i0, valid_i
      )
      act_drop_p = dropout_replay_panel(
          act_p[:, 0:valid_i],
          mask_valid,
          down_lora_dropout_p
      )

      dact_v = dact_sel[rows_global, i0:i0+valid_i]

      gate_v = gate_p[:, 0:valid_i]
      up_v = up_p[:, 0:valid_i]
      dgate_v = dact_v * up_v * silu_grad(gate_v)
      dup_v = dact_v * silu(gate_v)

      grad_pair_window[rows_local, n0:n0+BN] = 0
      grad_pair_window[rows_local, n0:n0+valid_i] = dgate_v
      grad_pair_window[rows_local, n0+P:n0+P+valid_i] = dup_v
      grad_gate_sel[rows_global, i0:i0+valid_i] = dgate_v
      grad_up_sel[rows_global, i0:i0+valid_i] = dup_v

      grad_down_lora_A_sel[e, :, i0:i0+valid_i] += dS_p^T @ act_drop_p
```

Correctness:

```text
base gate/up match torch reference against W_gate_up_cpu.to(cuda)
LoRA-B deltas match S_gate/S_up @ B_gate/B_up
act_p == silu(gate_p) * up_p within BF16 tolerance
down dropout mask is replayed exactly; no RNG is advanced
selected grad_down_lora_A contribution matches reference and is not double-counted
nonselected down-LoRA rows remain owned by the existing path
dgate/dup use passed final dact_sel
dgate/dup use FP32 silu derivative
last partial window writes zeros for invalid Kdx columns
grad_gate_sel/grad_up_sel match reference for selected rows
```

Latency/profiling:

```text
recompute_down_lora_activation_ms reported
recompute_reads_w_cache_bytes > 0 for cache_first_window
timing fields follow the Timing field semantics in the NCU Requirements section
native_kernel_consumed_down_dropout_masks reported
native_kernel_consumed_dS_down reported
no_aten_native_dropout_in_backward remains true
grad_pair_window write bytes reported
pair_acc_global_bytes reported and must be 0 for tile-local epilogue
full_window_gate_up_act_bytes reported and must be 0 for Stage 7 pass
tensor-core utilization reported in NCU for GEMM kernels
epilogue register count and local memory bytes reported
```

### Phase B: Selected Base dX

Logical GEMM:

```text
for e_rel, e in expert_chunk:
  rows_e_global = selected compact row ids for e
  rows_e_local = row_to_chunk_local[rows_e_global]
  for h0 in 0..H step BK:
    dx_tile = 0
    for n0 in 0..Kdx step BN:
      G_tile = grad_pair_window[rows_e_local, n0:n0+BN]
      W_tile = w_cache[e_rel, n0:n0+BN, h0:h0+BK]
      dx_tile += G_tile @ W_tile
    dx_acc[rows_e_global, h0:h0+BK] += dx_tile
```

Implementation requirements:

```text
use an SM100 BF16 tensor-core grouped GEMM with transpose-B style layout
the only permitted substitute is an in-repo native C++/CUDA helper called from
this API, read grad_pair_window and w_cache directly, accumulate fp32, emit the
same counters, and appear as the dx_window kernel in NCU
do not use a Python/Torch fallback, materialized padded tensors, the old
selected _grouped_base_dx path, or any path that streams W_gate_up_cpu in dX
output accumulation is fp32
zero dx_acc once before all windows
do not allocate per-window [M_selected, H] and add with torch.add
do not stream W_gate_up_cpu in dX
```

Correctness:

```text
grad_x_base_sel == sum_i dgate_i W_gate_i + dup_i W_up_i within BF16 tolerance
selected rows receive base dX exactly once
nonselected rows are not touched by this native kernel
```

Latency/profiling:

```text
dx_window_ms reported
dx_reads_w_cache_bytes > 0
dx_acc write/read bytes reported
w_cache_read_multiplier_vs_cpu_min and w_cache_read_model_ratio reported
```

## Python Integration

In `_ThresholdedQwen3ExpertFunction.backward` in:

```text
asym_gemm/training/qwen3_moe.py
```

Use the native path only when:

```text
SM100
BF16 selected tensors
layer.gate_up_base is AsymGroupedFrozenLinear
layer.gate_up_base.host_weight.weight is CPU-pinned BF16 [E, 2I, H]
layer.lora_dtype == torch.bfloat16
selected recompute rows are nonempty
ASYM_QWEN3_GATE_UP_WINDOWED_BWD=1
```

Runtime switch contract:

```text
ASYM_QWEN3_GATE_UP_WINDOWED_BWD=0 or unset:
  use the current selected-recompute backward.
  native_calls == 0.
  This is the before/baseline profile.

ASYM_QWEN3_GATE_UP_WINDOWED_BWD=1:
  use qwen3_gate_up_down_lora_bwd_sm100_bf16_windowed for selected recompute
  rows when all guards pass.
  This is the after/candidate profile.

ASYM_QWEN3_GATE_UP_WINDOWED_BWD_MODE:
  cache_first_window by default.
  seed_group_direct/all_rows_direct require
  ASYM_QWEN3_GATE_UP_ENABLE_LAZY_DIRECT=1.

The profiling scripts must run both switch values for the same seed, shape,
dropout, expert policy, and tile config before reporting a speedup.
```

If `layer.gate_up_base` is not `AsymGroupedFrozenLinear`, or
`layer.lora_dtype != torch.bfloat16`, Python must fall back to the current
selected-recompute backward. V2 native C++ does not accept FP16/FP32 LoRA low
rank or LoRA-B tensors.

Backward ownership:

```text
1. existing down base path computes dact_base = dY W_down.
2. existing split down-LoRA prework computes selected:
     dS_down = dY B_down * lora_scale
     dact_lora = dropout_backward(dS_down A_down, down_mask)
     dact = dact_base + dact_lora
     grad_down_lora_B = dY^T S_down * lora_scale
3. compact selected rows into x_sel, dact_sel, dS_down_sel, gate/up S, and
   selected down dropout masks.
4. provide gate_low_rank_sel/up_low_rank_sel:
     dropout 0.0: recompute S in Python or use saved S
     dropout >0.0: use saved forward S
5. call qwen3_gate_up_down_lora_bwd_sm100_bf16_windowed with:
     layer.gate_up_base.host_weight.weight
     dact_sel
     dS_down_sel
     selected down dropout mask
6. scatter grad_x_base_sel to selected rows in grad_packed
7. scatter grad_gate_sel/grad_up_sel for existing gate/up LoRA backward
8. merge native selected grad_down_lora_A with nonselected down-LoRA A grads
9. keep outside grad_down_lora_B result
10. run existing nonselected gate/up base dX only for nonselected rows
11. run existing gate/up LoRA backward for all rows needing LoRA grads/dX
```

No selected row may receive base dX from both the native selected path and the
old `_grouped_base_dx` path.

Naive direct-op integration is rejected for Stage 6 integration:

```text
old Python selected recompute produces act for selected dA_down
qwen3_gate_up_recompute_bwd_sm100_bf16_windowed then recomputes selected gate/up
again from CPU W

This proves local op correctness but keeps full selected gate/up CPU stream at
2x -> 2x. It cannot be used for the Stage 6/7 integrated speedup claim.
```

Nonselected base dX:

```text
nonselected_groups = active_groups minus selected_recompute_groups
if nonselected_rows.numel() > 0:
  run existing _grouped_base_dx for nonselected rows only
```

## Required Counters

The native function and validation script must emit:

```text
mode
p
q
w
kdx
bm
bk
g_work
cpu_weight_bytes_staged
expected_cpu_weight_bytes_min
cpu_weight_stream_multiplier
fill_w_cache_effective_GBps
pinned_h2d_baseline_GBps
same_shape_cache_first_window_recompute_reads_w_cache_bytes
measured_saved_recompute_hbm_read_bytes
hbm_w_cache_write_bytes
hbm_w_cache_valid_write_bytes
hbm_w_cache_padding_zero_write_bytes
recompute_reads_w_cache_bytes
dx_reads_w_cache_bytes
w_cache_bytes_allocated_peak
grad_pair_window_bytes_allocated_peak
dx_acc_bytes_allocated_peak
dS_down_bytes_allocated_peak
grad_down_lora_A_bytes_allocated_peak
full_window_gate_up_act_bytes
pair_acc_global_bytes
local_memory_load_bytes
local_memory_store_bytes
num_windows
num_recompute_panels
num_dx_windows
num_phase_launches
fused_recompute_down_lora_kernel
num_expert_waves
num_m_groups
g_chunk
m_work
m_padded_work
tokens_per_expert_histogram
max_rows_per_expert
sum_m_blocks
m_block_weight_reuse_factor_R_M
expected_current_cpu_weight_bytes_tile_model
expected_w_cache_read_bytes_tile_model
w_cache_read_multiplier_vs_cpu_min
w_cache_read_model_ratio
native_calls
original_selected_base_recompute_calls
original_selected_base_dx_calls
old_selected_base_dx_rows
new_selected_base_dx_rows
nonselected_base_dx_rows
reference_nonselected_rows
selected_recompute_rows
native_kernel_consumed_saved_gate_up_s
native_kernel_consumed_down_dropout_masks
native_kernel_consumed_dact
native_kernel_consumed_dS_down
dropout_backward_rng_advanced
no_aten_native_dropout_in_backward
dropout_replay_mismatches
workspace_reused
metadata_ms
fill_w_cache_ms
recompute_act_ms
down_lora_activation_ms
recompute_down_lora_activation_ms
selected_down_lora_A_ms
activation_backward_ms
dx_window_ms
native_total_ms
baseline_reference_ms
current_selected_region_ms
native_selected_region_ms
projected_full_step_saving_ms
projected_full_step_saving_percent
median_latency_ms
p50_step_ms
p95_step_ms
peak_hbm_bytes
max_abs_error
max_rel_error
```

Required equalities:

```text
expected_cpu_weight_bytes_min ==
  sum over iwin0 and active selected experts:
    2 * clamp(I - iwin0, 0, W) * H * sizeof(bf16)

cpu_weight_stream_multiplier =
  cpu_weight_bytes_staged / expected_cpu_weight_bytes_min

sum_m_blocks == sum over active selected experts ceil(M_e / BM)
m_block_weight_reuse_factor_R_M == sum_m_blocks / active_selected_experts
m_padded_work == sum over experts in the current chunk ceil(M_e / BM) * BM

expected_current_cpu_weight_bytes_tile_model ==
  2 * m_block_weight_reuse_factor_R_M * expected_cpu_weight_bytes_min

expected_w_cache_read_bytes_tile_model:
  cache_first_window: 2 * R_M * expected_cpu_weight_bytes_min
  seed_group_direct: between 1 * R_M and 2 * R_M times expected_cpu_weight_bytes_min
  all_rows_direct:   1 * R_M * expected_cpu_weight_bytes_min

pinned_h2d_baseline_GBps:
  measured by the validation script from one contiguous pinned CPU BF16 tensor
  with the same byte count class as expected_cpu_weight_bytes_min, copied to a
  CUDA BF16 tensor on the same device before the timed stage

fill_w_cache_effective_GBps =
  cpu_weight_bytes_staged / fill_w_cache_ms

same_shape_cache_first_window_recompute_reads_w_cache_bytes:
  copied from the Stage 7 same-shape cache_first_window profile used as the
  baseline for Stage 8/9 direct-mode comparisons

measured_saved_recompute_hbm_read_bytes =
  same_shape_cache_first_window_recompute_reads_w_cache_bytes
  - recompute_reads_w_cache_bytes

w_cache_read_multiplier_vs_cpu_min =
  (recompute_reads_w_cache_bytes + dx_reads_w_cache_bytes)
  / expected_cpu_weight_bytes_min

w_cache_read_model_ratio =
  (recompute_reads_w_cache_bytes + dx_reads_w_cache_bytes)
  / expected_w_cache_read_bytes_tile_model
```

Counter ownership:

```text
native stats:
  mode, p, q, w, kdx, bm, bk, g_work
  cpu/hbm byte counters
  fill_w_cache_effective_GBps, pinned_h2d_baseline_GBps, and Stage 8/9
  comparison byte counters when those modes are profiled
  workspace byte counters
  phase launch counts
  fused_recompute_down_lora_kernel
  selected/nonselected row ownership counters visible to native code
  phase timing counters when return_stats=True

validation script:
  baseline_reference_ms
  current_selected_region_ms
  native_selected_region_ms
  projected_full_step_saving_ms
  projected_full_step_saving_percent
  max_abs_error and max_rel_error
  NCU/NSYS-derived bytes, occupancy, tensor, and local-memory fields

Python LF profile JSON:
  native_calls
  original_selected_base_recompute_calls
  original_selected_base_dx_calls
  old_selected_base_dx_rows
  new_selected_base_dx_rows
  nonselected_base_dx_rows
  native_kernel_consumed_saved_gate_up_s
  native_kernel_consumed_down_dropout_masks
  native_kernel_consumed_dact
  native_kernel_consumed_dS_down
  dropout_backward_rng_advanced
  no_aten_native_dropout_in_backward
```

Mode-specific bounds:

```text
cache_first_window:
  cpu_weight_stream_multiplier <= 1.01
  recompute_reads_w_cache_bytes > 0
  dx_reads_w_cache_bytes > 0
  w_cache_read_model_ratio is within 0.80-1.25 unless NCU explains L2 reuse or extra traffic

seed_group_direct:
  cpu_weight_stream_multiplier <= 1.01
  recompute_reads_w_cache_bytes <
    recompute_reads_w_cache_bytes from same-shape cache_first_window
  dx_reads_w_cache_bytes > 0

all_rows_direct:
  cpu_weight_stream_multiplier <= 1.05
  recompute_reads_w_cache_bytes == 0 or explained by boundary/metadata reads
  dx_reads_w_cache_bytes > 0
```

Common bounds:

```text
old_selected_base_dx_rows == 0
new_selected_base_dx_rows == selected_recompute_rows
nonselected_base_dx_rows == reference_nonselected_rows
w_cache_bytes_allocated_peak <= G_work * Kdx * H * sizeof(bf16)
workspace_reused == true
dropout_replay_mismatches == 0
```

## Required Scripts And Workflow

There must be two separate validation/profiling layers:

```text
op-level layer:
  correctness, CUDA-event timing, NSYS schedule checks, and NCU kernel analysis
  runs before any LF end-to-end SFT profile

LF end-to-end layer:
  50-step SFT profile similar to scripts/lf/profile_lora_lf_fused.sh
  runs only after op-level correctness, timing, NSYS, and NCU gates pass
```

Do not use LF end-to-end results to debug a broken kernel. The implementation
agent must first prove each native op is correct and fast enough in isolation.

Required scripts to write:

```text
scripts/lf/validate_qwen3_gate_up_windowed_bwd.py
  Direct correctness and timing harness.
  Owns torch references, native calls, CUDA-event latency, NSYS child runs, and
  NCU child runs.

scripts/lf/profile_qwen3_gate_up_windowed_ops.sh
  Thin shell sweep around the validation script.
  Similar style to scripts/lf/profile_lora_lf_fused.sh, but op-level only.
  Runs direct correctness, CUDA-event timing sweeps, focused NSYS, and focused
  NCU before any LF E2E profile is allowed.

scripts/lf/profile_lora_lf_fused.sh
  E2E LF profile wrapper.
  May be extended to set ASYM_QWEN3_GATE_UP_WINDOWED_BWD=1 and collect native
  stats from profile JSON, but must not be the primary NCU workflow.
```

Required workflow:

```text
1. Run direct correctness without profiler.
2. Run direct CUDA-event timing.
3. Run paired selected-region timing:
     ASYM_QWEN3_GATE_UP_WINDOWED_BWD=0 baseline/current
     ASYM_QWEN3_GATE_UP_WINDOWED_BWD=1 native/candidate
   using the same seed, shape, dropout, expert policy, P/Q/BM/BK/G_work, and
   warmup/latency iteration counts.
4. Run focused NSYS to verify phase schedule and launch count.
5. Run focused NCU for each major kernel/phase.
6. Inspect NCU metrics and improve kernels.
7. Repeat 1-6 until op-level correctness, NCU gates, and significant op-level
   latency gates pass.
8. Write the Stage 7 aggregate result and decision to
   agent/fused_kernels/kernels_progress.md.
9. HARD STOP: return control to the user after Stage 7. Do not start Stage 8,
   Stage 9, or Stage 10 without explicit user approval.
10. Only after user approval run LF E2E profile through profile_lora_lf_fused.sh or
   profile_lora_lf.sh.
```

Blocking rule:

```text
Stage 10 LF E2E is not allowed to pass unless Stage 7 includes fresh NCU reports
for fill_w_cache, recompute_act, down_lora_activation, and dx_window with all
required metrics present.
Stage 10 LF E2E is not allowed to start for integration unless Stage 7 also
shows a significant same-shape op-level latency win over the current selected
recompute+dX path.
```

Significant op-level latency win:

```text
Required before LF E2E:
  native_selected_region_ms <= 0.90 * current_selected_region_ms
  for qwen_shape_smallM and qwen_shape_routed.

Also required for the Qwen3-30B-A3B estimate:
  projected_full_step_saving_ms >= 40 ms
  or projected_full_step_saving_percent >= 2 percent.

If the native path is correct but does not meet this threshold:
  do not run LF E2E for integration
  return to NCU-driven kernel iteration
  try Q/BM/BK/G_work/expert-wave tuning or seed_group_direct if cache reads
  dominate.
```

## Validation Script

Add:

```text
scripts/lf/validate_qwen3_gate_up_windowed_bwd.py
```

Every stage writes:

```text
profiling/qwen3_gate_up_windowed_bwd/<stage>/validation.json
profiling/qwen3_gate_up_windowed_bwd/<stage>/validation.md
profiling/qwen3_gate_up_windowed_bwd/<stage>/latency.csv
```

Base CLI:

```bash
python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py \
  --stage STAGE_NAME \
  --op OP_NAME \
  --device cuda:1 \
  --seed 1234 \
  --output-dir profiling/qwen3_gate_up_windowed_bwd/STAGE_NAME
```

Common arguments:

```text
--stage
--device
--seed
--op
--cases
--p
--q
--bm
--bk
--g-work
--mode
--lora-dropout
--warmup-iters
--latency-iters
--profile-mode {none,cuda-events,nsys,ncu}
--profile-root
--keep-artifacts
```

There is no `--bn` argument in V2. `BN` is always derived as `2 * p`.

Required `--op` values:

```text
baseline:
  measure the existing selected-recompute path and emit Stage 0 counters

metadata_fill:
  build selected metadata and fill w_cache only

recompute_act:
  run metadata, fill_w_cache, and selected gate/up base recompute from w_cache;
  add gate/up LoRA-B deltas and produce only tile-local or debug gate/up/act
  outputs. This op must not compute dA_down, dgate/dup, or dX.

down_lora_activation:
  run metadata, fill_w_cache, recompute_act, selected down-LoRA A tile
  accumulation, and activation backward. This op consumes dact_sel and
  dS_down_sel, writes grad_pair_window and grad_gate_sel/grad_up_sel, and must
  not compute dact_lora or dB_down.

dx_window:
  run metadata, fill_w_cache, down_lora_activation, and selected base dX from
  grad_pair_window and w_cache

native_e2e:
  run the full pybind path and compare all outputs

all:
  run every direct op-level check for the requested cases

python_integration:
  run the qwen3_moe.py backward integration path and row/dropout ownership checks

cache_first_profile:
  run Stage 7 aggregate NSYS/NCU profile validation from V2 native artifacts

lf_e2e:
  validate the LF end-to-end profile root after Stage 7 has passed
```

The validation script must be able to run a single kernel family repeatedly
without LF, without model loading, and without Python panel/window loops around
the native implementation.

The script must exit nonzero if a required correctness or latency/profiling
gate fails. It must not silently skip non-SM100. On non-SM100 it exits nonzero
with `passed=false` and `reason=requires_sm100`.

Every `validation.md` must include:

```text
exact command line
environment summary
case table: shape, mode, P, Q, max_abs, max_rel, passed
latency table: baseline_ms, native_ms, speedup, warmup_iters, latency_iters
profile gate table: observed value, expected value, passed
artifact links
```

Required JSON top-level shape:

```json
{
  "stage": "stage5_v2_native_e2e",
  "op": "native_e2e",
  "passed": true,
  "device": {"name": "...", "sm": 100},
  "config": {"mode": "cache_first_window", "p": 32, "q": 8, "bm": 64, "bk": 512, "g_work": 128},
  "cases": {},
  "correctness": {},
  "latency": {},
  "counters": {},
  "ncu": {},
  "nsys": {},
  "artifacts": {}
}
```

The script must fail if `latency`, `counters`, or required `ncu` fields are
missing for stages that require them. Missing profiler data is a failure, not a
warning.

Before/after comparison rule:

```text
For any speedup claim, validation.json must contain both:
  baseline_current:
    ASYM_QWEN3_GATE_UP_WINDOWED_BWD=0
  candidate_native:
    ASYM_QWEN3_GATE_UP_WINDOWED_BWD=1

The validator must reject the comparison if seed, case, lora_dropout,
expert_policy, P, Q, BM, BK, G_work, warmup_iters, or latency_iters differ
between baseline_current and candidate_native.

selected_region_speedup =
  current_selected_region_ms / native_selected_region_ms

full_step_speedup =
  baseline_current_step_median_ms / candidate_native_step_median_ms
```

## Op-Level Profiling Script

Add:

```text
scripts/lf/profile_qwen3_gate_up_windowed_ops.sh
```

Purpose:

```text
Run the native op-level validation/profiling ladder quickly and repeatably.
This is the script used during kernel development before any LF E2E profile.
```

Required user parameters:

```text
ROOT
GPU
OUTPUT_ROOT
NATIVE_ENABLED_VALUES
MODES
CASES
P_VALUES
Q_VALUES
BM_VALUES
BK_VALUES
G_WORK_VALUES
WARMUP_ITERS
LATENCY_ITERS
RUN_CORRECTNESS
RUN_CUDA_EVENTS
RUN_NSYS
RUN_NCU
OVERWRITE
DRY_RUN
```

Default fast profile:

```bash
GPU=1 \
NATIVE_ENABLED_VALUES=0,1 \
MODES=cache_first_window \
CASES=ragged_groups,partial_window,qwen_shape_smallM \
P_VALUES=32 \
Q_VALUES=8 \
BM_VALUES=64 \
BK_VALUES=512 \
G_WORK_VALUES=128 \
RUN_CORRECTNESS=1 \
RUN_CUDA_EVENTS=1 \
RUN_NSYS=1 \
RUN_NCU=1 \
OUTPUT_ROOT=profiling/qwen3_gate_up_windowed_bwd/ops \
bash scripts/lf/profile_qwen3_gate_up_windowed_ops.sh
```

The script must run these steps in order:

```text
correctness:
  validate_qwen3_gate_up_windowed_bwd.py --op all --profile-mode none

cuda_events:
  validate_qwen3_gate_up_windowed_bwd.py --op native_e2e --profile-mode cuda-events

nsys:
  validate_qwen3_gate_up_windowed_bwd.py --op native_e2e --profile-mode nsys

ncu_fill:
  validate_qwen3_gate_up_windowed_bwd.py --op metadata_fill --profile-mode ncu

ncu_recompute_act:
  validate_qwen3_gate_up_windowed_bwd.py --op recompute_act --profile-mode ncu

ncu_down_lora_activation:
  validate_qwen3_gate_up_windowed_bwd.py --op down_lora_activation --profile-mode ncu

ncu_dx:
  validate_qwen3_gate_up_windowed_bwd.py --op dx_window --profile-mode ncu
```

Required output layout:

```text
<OUTPUT_ROOT>/native<NATIVE_ENABLED>__<mode>__p<P>_q<Q>_bm<BM>_bk<BK>_gw<G_WORK>/
  correctness/validation.json
  cuda_events/validation.json
  nsys/validation.json
  ncu_fill/validation.json
  ncu_recompute_act/validation.json
  ncu_down_lora_activation/validation.json
  ncu_dx/validation.json
  summary.json
  summary.md
```

`summary.json` must aggregate:

```text
native_enabled
all correctness max_abs/max_rel
native_total_ms, fill_w_cache_ms, recompute_act_ms,
down_lora_activation_ms, recompute_down_lora_activation_ms,
selected_down_lora_A_ms, activation_backward_ms, dx_window_ms
current_selected_region_ms
native_selected_region_ms
selected_region_speedup
paired_baseline_output_dir
paired_candidate_output_dir
projected_full_step_saving_ms
projected_full_step_saving_percent
cpu_weight_stream_multiplier
fill_w_cache_effective_GBps
pinned_h2d_baseline_GBps
w_cache_read_multiplier_vs_cpu_min
w_cache_read_model_ratio
fused_recompute_down_lora_kernel
NCU dram bytes
NCU L2 hit rates
NCU tensor utilization
NCU occupancy/register/local-memory metrics
pass/fail for every gate
```

The shell script must stop on the first failed correctness gate. It may continue
across profiling failures only when `CONTINUE_ON_ERROR=1`, but the final summary
must still be `passed=false`.

## NCU Requirements

NCU is mandatory for kernel development. CUDA-event timing alone is not enough
to accept a kernel change.

The validation script must own focused NCU invocations. Do not run NCU over the
full LF SFT workload by default.

Recommended NCU command shape:

```bash
ncu --target-processes all \
  --force-overwrite \
  --set full \
  --kernel-name 'regex:sm100_qwen3|m_grouped_bf16|qwen3_gate_up' \
  --export <output-dir>/profiles/<case>/<op>.ncu-rep \
  python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py \
    --stage <stage> \
    --op <op> \
    --device cuda:1 \
    --cases <case> \
    --profile-mode none \
    --internal-profile-child ncu
```

The script must also export CSV:

```bash
ncu --import <report>.ncu-rep --csv > <report>.csv
```

Minimum NCU reports to collect before LF E2E:

```text
fill_w_cache:
  op=metadata_fill, kernel regex sm100_qwen3_fill_gate_up_w_cache

recompute_act:
  op=recompute_act, grouped BF16 GEMM kernel reading w_cache

down_lora_activation:
  op=down_lora_activation, selected dA_down plus dgate/dup kernel or fused
  recompute+epilogue kernel

dx_window:
  op=dx_window, native BF16 tensor-core dX kernel reading grad_pair_window and
  w_cache directly
```

If recompute_act and down_lora_activation are fused into one physical kernel,
the validator may point both report entries to the same `.ncu-rep`, but
`validation.json` must set `fused_recompute_down_lora_kernel=true` and still
provide separate logical counters for recompute reads, selected down-LoRA A
work, activation backward work, local-memory bytes, and grad_pair writes.

Timing field semantics:

```text
If recompute_act and down_lora_activation are separate physical kernels:
  recompute_act_ms = measured recompute_act kernel time
  down_lora_activation_ms = measured selected dA_down + dgate/dup kernel time
  recompute_down_lora_activation_ms = recompute_act_ms + down_lora_activation_ms
  selected_down_lora_A_ms and activation_backward_ms are sub-counters when the
  implementation can separate them; otherwise they are null and
  down_lora_activation_ms is the blocking epilogue-overhead metric.

If they are one fused physical kernel:
  fused_recompute_down_lora_kernel = true
  recompute_down_lora_activation_ms = measured fused kernel time
  recompute_act_ms = null
  down_lora_activation_ms = null
  selected_down_lora_A_ms = null
  activation_backward_ms = null
  logical byte counters must still be present.
```

Required NCU metric categories to parse into `validation.json`:

```text
Launch/occupancy:
  grid size, block size, registers per thread, static/dynamic shared memory,
  achieved occupancy, active warps

Speed of light:
  kernel duration, SM throughput percent, DRAM throughput percent

Tensor core / pipe use:
  tensor-pipe active cycles or tensor instruction count

DRAM and L2:
  DRAM read/write bytes, L2 read/write bytes, L2 hit/miss sectors or hit rate

Local/spill:
  local memory load/store bytes

Scheduler stalls:
  barrier, long scoreboard, short scoreboard, not-selected stall fractions
```

Preferred metric names when available:

```text
gpu__time_duration.sum
sm__throughput.avg.pct_of_peak_sustained_elapsed
dram__throughput.avg.pct_of_peak_sustained_elapsed
dram__bytes_read.sum
dram__bytes_write.sum
smsp__inst_executed_pipe_tensor.sum
smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
lts__t_bytes_srcunit_tex_op_read.sum
lts__t_bytes_srcunit_tex_op_write.sum
lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum
lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum
lmem__t_bytes_pipe_lsu_mem_local_op_ld.sum
lmem__t_bytes_pipe_lsu_mem_local_op_st.sum
smsp__warps_issue_stalled_barrier_per_warp_active.pct
smsp__warps_issue_stalled_long_scoreboard_per_warp_active.pct
smsp__warps_issue_stalled_short_scoreboard_per_warp_active.pct
smsp__warps_issue_stalled_not_selected_per_warp_active.pct
```

If a metric name is unavailable on the installed NCU version, the script must
record the missing metric and use the closest section-derived metric. It must
not silently omit the category.

Required NCU checks by op:

```text
fill_w_cache:
  DRAM read bytes are within 0.90-1.20x expected_cpu_weight_bytes_min for the
  profiled case/window, unless cache effects are explicitly explained.
  DRAM write bytes are compared with hbm_w_cache_write_bytes, where
  hbm_w_cache_write_bytes =
    hbm_w_cache_valid_write_bytes + hbm_w_cache_padding_zero_write_bytes.
  hbm_w_cache_valid_write_bytes must equal expected_cpu_weight_bytes_min.
  local_memory_load_bytes + local_memory_store_bytes <= 1 percent of
  dram__bytes_read.sum + dram__bytes_write.sum.
  fill_w_cache_effective_GBps >= 0.50 * pinned_h2d_baseline_GBps for the same
  GPU and host memory path, or the stage fails as scalar/address-overhead bound.

recompute_act:
  DRAM/L2 read bytes are reported and compared with R_M tile model.
  Tensor-pipe utilization is nonzero for all profiled cases and >= 10 percent
  on qwen_shape_smallM/qwen_shape_routed unless DRAM throughput is >= 60 percent
  of peak and w_cache_read_model_ratio is within the accepted bound.
  local_memory_load_bytes + local_memory_store_bytes <= 1 percent of
  dram__bytes_read.sum + dram__bytes_write.sum.
  Long-scoreboard stalls must be reported when tensor utilization is low.

down_lora_activation:
  local_memory_load_bytes + local_memory_store_bytes <= 1 percent of
  dram__bytes_read.sum + dram__bytes_write.sum.
  Register count and achieved occupancy are reported; achieved occupancy below
  25 percent fails unless the same report shows tensor/DRAM throughput is the
  limiting resource.
  If fused_recompute_down_lora_kernel=false:
    down_lora_activation_ms <= 0.25 * native_selected_region_ms for
    qwen_shape_smallM and qwen_shape_routed, or the stage fails as
    epilogue-overhead bound.
  If fused_recompute_down_lora_kernel=true:
    recompute_down_lora_activation_ms is compared against the full
    native_selected_region_ms Stage 7 threshold instead of this sub-threshold.

dx_window:
  Tensor-pipe utilization is nonzero for all profiled cases and >= 10 percent
  on qwen_shape_smallM/qwen_shape_routed unless DRAM throughput is >= 60 percent
  of peak and w_cache_read_model_ratio is within the accepted bound.
  DRAM/L2 read bytes include w_cache reads and are compared with R_M tile model.
  dx_acc read/write bytes are reported.
  dx_acc read/write bytes <= 1.25x the analytical dx_acc traffic model for the
  chosen Q.
```

NCU-driven iteration rules:

```text
If fill_w_cache has poor bandwidth:
  improve vectorization/alignment and CPU pinned-memory access pattern first.

If recompute tensor utilization is low and DRAM bytes are high:
  tune BM, BK, expert-wave scheduling, and L2 locality before changing math.

If dx_window is slow with low tensor utilization:
  increase Q or tune Kdx/BK; avoid Q=4 unless memory forces it.

If local memory bytes exceed the op threshold in down_lora_activation or
all_rows_direct:
  reduce register pressure or reject that fused variant.

If w_cache_read_model_ratio > 1.25:
  inspect scheduling order and L2 locality; do not move to E2E.

If CUDA-event latency improves but NCU shows CPU stream multiplier, spills, or
HBM bytes are wrong:
  reject the change even if the direct latency looks better.
```

## Staged Implementation And Gates

Do not move to the next stage until the current stage has a saved
`validation.json`, a human-readable `validation.md`, and all listed correctness
and latency/profiling gates pass.

### Stage 0: Baseline Instrumentation

Goal:

```text
Measure the existing selected-recompute path before changing math.
```

Implementation:

```text
Add profile counters/ranges in qwen3_moe.py backward:
  selected_recompute_rows
  original_selected_base_recompute_calls
  original_selected_base_dx_calls
  selected_w_gate_up_streams_logical
  selected_w_gate_up_streams_tile_model
  nonselected_base_dx_rows
```

Correctness gate:

```text
existing tests pass
selected rows and nonselected rows partition active rows
native_calls == 0
original_selected_base_recompute_calls > 0 when selected rows exist
original_selected_base_dx_calls > 0 when selected rows exist
```

Latency/profiling gate:

```text
baseline_reference_ms is recorded with cuda events
profile range names for selected recompute and selected base dX are emitted;
focused NSYS capture is optional in Stage 0 and blocking only in Stage 7
expected_current_cpu_weight_bytes_tile_model is reported
```

Command:

```bash
python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py \
  --stage stage0_baseline_instrumentation \
  --op baseline \
  --device cuda:1 \
  --profile-mode cuda-events \
  --output-dir profiling/qwen3_gate_up_windowed_bwd/stage0_baseline_instrumentation
```

### Stage 1: Metadata And W-Cache Fill

Goal:

```text
Validate selected metadata and CPU-to-HBM window staging without GEMM.
```

Implementation:

```text
Add pybind or private test entry for:
  build metadata
  fill_w_cache for one or more windows
Return w_cache and stats for direct validation.
```

Correctness gate:

```text
row_experts match selected_offsets/selected_experts
w_cache equals torch reference slice for gate and up
last partial window invalid columns are zero
cpu_weight_bytes_staged == expected_cpu_weight_bytes_min
cpu_weight_stream_multiplier <= 1.01
```

Latency/profiling gate:

```text
metadata_ms and fill_w_cache_ms are recorded
hbm_w_cache_valid_write_bytes and hbm_w_cache_padding_zero_write_bytes are recorded
fill_w_cache_effective_GBps and pinned_h2d_baseline_GBps are recorded
fill_w_cache_effective_GBps >= 0.50 * pinned_h2d_baseline_GBps
the validation script checks that the implementation path does not call
torch.cat/index_select/cudaMemcpyAsync per expert; focused NSYS verification is
blocking only in Stage 7
current_selected_region_ms is read from the Stage 0 validation artifact for the
same qwen_shape_smallM case, seed, dropout, expert policy, P/Q/BM/BK/G_work, and
latency iteration count; missing or mismatched Stage 0 artifact fails Stage 1
fill_w_cache_ms <= 0.35 * current_selected_region_ms for qwen_shape_smallM,
or the stage is marked failed_latency with reason=fill_staging_overhead
```

Command:

```bash
python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py \
  --stage stage1_metadata_and_fill \
  --op metadata_fill \
  --device cuda:1 \
  --cases one_group,ragged_groups,partial_window,qwen_shape_smallM \
  --profile-mode cuda-events \
  --output-dir profiling/qwen3_gate_up_windowed_bwd/stage1_metadata_and_fill
```

### Stage 2: Recompute Gate/Up And Act Tile

Goal:

```text
Validate cache_first recompute from w_cache, gate/up LoRA-B addition, and
act=silu(gate)*up production.
```

Implementation:

```text
Add recompute_act path:
  grouped BF16 GEMM reads w_cache
  tile epilogue adds LoRA-B deltas
  tile epilogue computes act
  gate/up/act are tile-local or debug-only validation outputs
```

Correctness gate:

```text
base gate/up match torch reference
LoRA-B deltas match torch reference
act=silu(gate)*up matches torch reference:
  max_abs <= 3e-2
  max_rel <= 3e-2
invalid last-window columns are zero
full_window_gate_up_act_bytes == 0 for latency-gated runs, or the run is
marked debug_only and cannot satisfy Stage 7
```

Latency/profiling gate:

```text
recompute_act_ms or recompute_down_lora_activation_ms is recorded
recompute_reads_w_cache_bytes > 0
local_memory_load_bytes and local_memory_store_bytes are reported
when profile-mode ncu is used, local_memory_load_bytes + local_memory_store_bytes
<= 1 percent of dram__bytes_read.sum + dram__bytes_write.sum for the profiled
kernel
pair_acc_global_bytes is reported and must be 0 for tile-local epilogue
NCU fields may be empty in Stage 2; focused NCU tensor-utilization and
epilogue local-memory checks are blocking only in Stage 7
```

Command:

```bash
python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py \
  --stage stage2_recompute_act \
  --op recompute_act \
  --device cuda:1 \
  --cases ragged_groups,partial_window,qwen_shape_smallM \
  --profile-mode cuda-events \
  --output-dir profiling/qwen3_gate_up_windowed_bwd/stage2_recompute_act
```

### Stage 3: Selected Down-LoRA A And Activation Tile

Goal:

```text
Validate selected down-LoRA A grad and activation backward while the recompute
tile is still live.
```

Implementation:

```text
Add selected_down_lora_A_activation path:
  accept dact_sel and dS_down_sel produced outside native
  for each BM x P tile:
    replay saved down dropout mask on act
    accumulate selected grad_down_lora_A from dS_down_sel^T @ act_drop
    read passed dact_sel
    compute dgate/dup
    write grad_pair_window and grad_gate_sel/grad_up_sel
```

Correctness gate:

```text
passed dS_down_sel matches torch reference in validation prework
passed dact_sel matches dact_base + selected dact_lora reference
grad_down_lora_A_sel matches selected-row reference:
  max_abs <= 5e-2
  max_rel <= 5e-2
grad_pair_window, grad_gate_sel, and grad_up_sel match reference:
  max_abs <= 3e-2
  max_rel <= 3e-2
dropout_replay_mismatches == 0
dropout_backward_rng_advanced == false
```

Latency/profiling gate:

```text
timing fields follow the Timing field semantics in the NCU Requirements section
full_window_gate_up_act_bytes == 0 for latency-gated runs
pair_acc_global_bytes == 0
when profile-mode ncu is used, local_memory_load_bytes + local_memory_store_bytes
<= 1 percent of dram__bytes_read.sum + dram__bytes_write.sum for the profiled
kernel
native_kernel_consumed_down_dropout_masks is correct for dropout setting
native_kernel_consumed_dact == true
native_kernel_consumed_dS_down == true
no aten/native dropout kernels appear in backward
```

Command:

```bash
python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py \
  --stage stage3_down_lora_activation \
  --op down_lora_activation \
  --device cuda:1 \
  --cases ragged_groups,partial_window,qwen_shape_smallM,qwen_shape_routed \
  --profile-mode cuda-events \
  --output-dir profiling/qwen3_gate_up_windowed_bwd/stage3_down_lora_activation
```

### Stage 4: Selected dX Window

Goal:

```text
Validate selected base dX accumulation using grad_pair_window and w_cache.
```

Implementation:

```text
Add dx_window path:
  native BF16 tensor-core grouped GEMM reads grad_pair_window and w_cache
  accumulates into dx_acc fp32
  casts final grad_x_base_sel to BF16
```

Correctness gate:

```text
grad_x_base_sel matches torch reference:
  max_abs <= 5e-2
  max_rel <= 5e-2
dx_acc is zeroed exactly once before all windows
dX accumulates once per Q-window, not once per P-panel
selected rows receive base dX exactly once
```

Latency/profiling gate:

```text
dx_window_ms is recorded
dx_reads_w_cache_bytes > 0
w_cache_read_multiplier_vs_cpu_min and w_cache_read_model_ratio are reported
tensor-pipe utilization is reported; Stage 7 requires >= 10 percent on
qwen_shape_smallM/qwen_shape_routed unless DRAM throughput is >= 60 percent
no per-window [M_selected, H] allocation appears
```

Command:

```bash
python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py \
  --stage stage4_dx_window \
  --op dx_window \
  --device cuda:1 \
  --cases ragged_groups,partial_window,qwen_shape_smallM,qwen_shape_routed \
  --profile-mode cuda-events \
  --output-dir profiling/qwen3_gate_up_windowed_bwd/stage4_dx_window
```

### Stage 5: V2 Native Direct End-To-End

Goal:

```text
Validate full V2 native pybind against torch reference without Python model
integration.
```

Implementation:

```text
Expose qwen3_gate_up_down_lora_bwd_sm100_bf16_windowed.
Run all windows and all expert chunks inside C++.
Return grad_x_base_sel, grad_gate_sel, grad_up_sel,
grad_down_lora_A_sel, stats.
Keep qwen3_gate_up_recompute_bwd_sm100_bf16_windowed only as an optional
debug/direct-op prototype; it is not the integration target.
```

Correctness gate:

```text
empty, one_group, ragged_groups, partial_window, qwen_shape_smallM pass
grad_gate_sel max_abs/max_rel <= 3e-2
grad_up_sel   max_abs/max_rel <= 3e-2
grad_x_base_sel max_abs/max_rel <= 5e-2
grad_down_lora_A_sel max_abs/max_rel <= 5e-2
stats equalities and mode-specific bounds pass
cpu_weight_stream_multiplier <= 1.01
```

Latency/profiling gate:

```text
native_total_ms and per-phase ms are recorded
native_total_ms is compared against torch reference and current selected path
num_phase_launches is reported
captured vs uncaptured launch overhead is measured for qwen_shape_smallM
full_window_gate_up_act_bytes == 0 for latency-gated runs
```

Command:

```bash
python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py \
  --stage stage5_v2_native_e2e \
  --op native_e2e \
  --device cuda:1 \
  --cases empty,one_group,ragged_groups,partial_window,qwen_shape_smallM \
  --profile-mode cuda-events \
  --warmup-iters 10 \
  --latency-iters 50 \
  --output-dir profiling/qwen3_gate_up_windowed_bwd/stage5_v2_native_e2e
```

### Stage 6: Python Integration Dropout 0.00 And 0.10

Goal:

```text
Route selected recompute rows through the V2 native path in qwen3_moe.py and
prove dropout replay ownership for both dropout 0.00 and 0.10.
```

Implementation:

```text
Add guarded native call in _ThresholdedQwen3ExpertFunction.backward.
For gate/up LoRA dropout 0.00, recompute or pass S_gate/S_up deterministically.
For gate/up LoRA dropout > 0, pass saved forward S_gate/S_up.
For down-LoRA dropout > 0, pass saved down dropout mask for selected act rows.
Compute selected dS_down, dact_lora, final dact, and grad_down_lora_B outside
native; these do not require recomputed gate/up/act.
Exclude selected rows from old selected base dX.
Disable or split the existing selected down-LoRA A branch so selected dA_down is
not double-counted.
Keep gate/up LoRA backward outside the native path.
```

Correctness gate:

```text
trainable LoRA grads match current selected-recompute reference:
  gate_lora_A/B max_abs/max_rel <= 5e-2
  up_lora_A/B   max_abs/max_rel <= 5e-2
  down_lora_A/B max_abs/max_rel <= 5e-2
old_selected_base_dx_rows == 0
new_selected_base_dx_rows == selected_recompute_rows
nonselected_base_dx_rows == reference_nonselected_rows
native_kernel_consumed_saved_gate_up_s == true when gate/up dropout > 0
native_kernel_consumed_down_dropout_masks == true when down dropout > 0
native_kernel_consumed_dact == true
native_kernel_consumed_dS_down == true
dropout_backward_rng_advanced == false
no_aten_native_dropout_in_backward == true
dropout_replay_mismatches == 0
```

Latency/profiling gate:

```text
native call count per selected layer is 1
old selected recompute/base-dX calls are 0 for selected rows
drop000 and drop010 native-region latency are recorded
drop010 median direct latency <= 1.10 * drop000 same-shape latency
no extra native kernel launches solely for dropout sampling
```

Command:

```bash
python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py \
  --stage stage6_python_integration_drop000 \
  --op python_integration \
  --device cuda:1 \
  --lora-dropout 0.00 \
  --profile-mode cuda-events \
  --output-dir profiling/qwen3_gate_up_windowed_bwd/stage6_python_integration_drop000

python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py \
  --stage stage6_python_integration_drop010 \
  --op python_integration \
  --device cuda:1 \
  --lora-dropout 0.10 \
  --profile-mode cuda-events \
  --output-dir profiling/qwen3_gate_up_windowed_bwd/stage6_python_integration_drop010
```

### Stage 7: NSYS/NCU Cache-First Profile Gate

Goal:

```text
Prove the cache-first schedule is real and quantify HBM/cache overhead before
any LF end-to-end profile.
```

Correctness gate:

```text
Stage 1 through Stage 6 validation JSONs are passed
same test cases still pass under profiler child process
```

Latency/profiling gate:

```text
paired baseline/candidate profiles exist:
  ASYM_QWEN3_GATE_UP_WINDOWED_BWD=0 current selected path
  ASYM_QWEN3_GATE_UP_WINDOWED_BWD=1 native cache_first_window path
both profiles use the same cases, seed, dropout, expert policy, P/Q/BM/BK/G_work,
warmup count, and latency iteration count
NSYS ranges exist:
  qwen3_gate_up_windowed/total
  qwen3_gate_up_windowed/metadata
  qwen3_gate_up_windowed/fill_w_cache
  qwen3_gate_up_windowed/recompute_down_lora_activation
  qwen3_gate_up_windowed/dx_window
NCU reports:
  dram bytes read/write for fill_w_cache
  HBM/L2 stats for w_cache reads
  tensor-core utilization for recompute/down-LoRA activation and dX
  local memory bytes for the down-LoRA/activation epilogue
measured w_cache reads are compared against R_M tile model
all required NCU metric categories are present or explicitly mapped to
available section-derived alternatives
fused_recompute_down_lora_kernel is recorded; if true, both logical report
entries point to the same NCU artifact and expose separate logical counters
native_selected_region_ms <= 0.90 * current_selected_region_ms for
qwen_shape_smallM and qwen_shape_routed
projected_full_step_saving_ms >= 40 ms or projected_full_step_saving_percent
>= 2 percent for the Qwen3-30B-A3B estimate
```

Command:

```bash
python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py \
  --stage stage7_cache_first_profile \
  --op native_e2e \
  --device cuda:1 \
  --cases qwen_shape_smallM,qwen_shape_routed \
  --profile-mode nsys \
  --output-dir profiling/qwen3_gate_up_windowed_bwd/stage7_cache_first_profile
```

Run NCU for focused kernels before Stage 10:

```bash
python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py \
  --stage stage7_cache_first_profile \
  --op all \
  --device cuda:1 \
  --cases qwen_shape_smallM,qwen_shape_routed \
  --profile-mode ncu \
  --output-dir profiling/qwen3_gate_up_windowed_bwd/stage7_cache_first_profile_ncu
```

Then run the Stage 7 aggregate gate over the NSYS/NCU artifacts:

```bash
python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py \
  --stage stage7_cache_first_profile \
  --op cache_first_profile \
  --device cuda:1 \
  --profile-root profiling/qwen3_gate_up_windowed_bwd \
  --output-dir profiling/qwen3_gate_up_windowed_bwd/stage7_cache_first_profile
```

The Stage 7 `validation.json` must include paths to:

```text
fill_w_cache.ncu-rep and fill_w_cache.csv
recompute_act.ncu-rep and recompute_act.csv
down_lora_activation.ncu-rep and down_lora_activation.csv
dx_window.ncu-rep and dx_window.csv
```

If any NCU report is missing, Stage 7 fails.

If the NCU reports pass but the native selected-region latency threshold does
not pass, Stage 7 may be recorded as a useful profiling artifact, but it is not
an integration pass and Stage 10 must not be launched for integration.

Hard stop after Stage 7:

```text
After the Stage 7 aggregate validation completes, write the full decision to
agent/fused_kernels/kernels_progress.md and stop.

Required Stage 7 decision values:
  passed_for_e2e
  failed_correctness
  failed_latency
  failed_ncu_or_traffic
  blocked_missing_artifacts

If passed_for_e2e:
  do not start Stage 10 yet.
  ask the user to review kernels_progress.md and approve E2E.

If failed_* or blocked_*:
  do not start Stage 8/9 automatically.
  write the proposed next kernel iteration and ask the user to approve it.
```

### Stage 8: Optional seed_group_direct

Goal:

```text
Reduce recompute-side w_cache reads without duplicating CPU W streams.
```

Implementation:

```text
Add persistent fused or CTA/cluster path where CPU-W loader computes one
M_group directly while filling w_cache.
```

Correctness gate:

```text
all Stage 5 V2 native correctness cases pass
all Stage 6 integration correctness cases pass
cpu_weight_stream_multiplier <= 1.01
```

Latency/profiling gate:

```text
recompute_reads_w_cache_bytes <
  same-shape cache_first_window recompute_reads_w_cache_bytes
full-step or layer-region latency improves over cache_first_window
pair_acc_global_bytes == 0
local_memory_load_bytes + local_memory_store_bytes does not increase by more
than 5 percent versus same-shape cache_first_window
```

Do not proceed to all_rows_direct unless seed_group_direct fails for a measured
reason or leaves a clear remaining recompute-read bottleneck.

### Stage 9: Optional all_rows_direct

Goal:

```text
Test the theoretical minimum recompute-cache-read schedule.
```

Correctness gate:

```text
all Stage 5 V2 native correctness cases pass
all Stage 6 integration correctness cases pass
cpu_weight_stream_multiplier <= 1.05
```

Latency/profiling gate:

```text
recompute_reads_w_cache_bytes <=
  0.05 * same_shape_cache_first_window_recompute_reads_w_cache_bytes
pair_acc_global_bytes == 0
local_memory_load_bytes + local_memory_store_bytes <=
  0.10 * measured_saved_recompute_hbm_read_bytes
achieved occupancy >= 25 percent unless the same report shows tensor/DRAM
throughput is the limiting resource
full-step latency is better than cache_first_window and seed_group_direct
```

Reject immediately if implementation re-streams CPU W per M_group.

### Stage 10: LF 50-Step End-To-End Gate

Goal:

```text
Prove the path is useful in the real LF SFT workload before merging or broad
E2E/product tuning.
```

Profile command:

```bash
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym|norecompute" \
ASYM_QWEN3_GATE_UP_WINDOWED_BWD_VALUES="0,1" \
EXPERT_POLICIES="none,tok-le512,tok-le1024,tok-le512-act,tok-le1024-act" \
PROFILERS="nsys" \
SEQ_LENS="4096" \
MAX_STEPS=50 \
WARMUP_STEPS=5 \
LORA_RANK=64 \
LORA_ALPHA=16 \
LORA_DROPOUT="0.00,0.10" \
PREPARE_DATASETS=true \
DATASET_OVERWRITE=false \
OUTPUT_ROOT=profiling/qwen3_gate_up_windowed_bwd/stage10_lf_50step \
bash scripts/lf/profile_lora_lf_fused.sh
```

If `profile_lora_lf_fused.sh` does not expose a needed option, extend that
script rather than bypassing the staged validation. It must run both:

```text
ASYM_QWEN3_GATE_UP_WINDOWED_BWD=0
ASYM_QWEN3_GATE_UP_WINDOWED_BWD=1
```

For the enabled run it must set:

```text
ASYM_QWEN3_GATE_UP_WINDOWED_BWD=1
ASYM_QWEN3_GATE_UP_WINDOWED_BWD_MODE=cache_first_window
ASYM_QWEN3_GATE_UP_WINDOWED_BWD_P=32
ASYM_QWEN3_GATE_UP_WINDOWED_BWD_Q=8
ASYM_QWEN3_GATE_UP_WINDOWED_BWD_BM=64
ASYM_QWEN3_GATE_UP_WINDOWED_BWD_BK=512
ASYM_QWEN3_GATE_UP_WINDOWED_BWD_G_WORK=128
```

The E2E script may use NSYS/source profiling, but not NCU by default. NCU for
kernel iteration belongs to Stage 7 and the op-level profiling script.

Validation command:

```bash
python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py \
  --stage stage10_lf_50step \
  --op lf_e2e \
  --profile-root profiling/qwen3_gate_up_windowed_bwd/stage10_lf_50step/asym_long_sft_smoke__lora__lf__bf16 \
  --output-dir profiling/qwen3_gate_up_windowed_bwd/stage10_lf_50step_validation
```

Correctness gate:

```text
Stage 7 cache_first_profile validation passed with fresh NCU artifacts
Stage 7 native selected-region latency threshold passed
profile JSON exists for both ASYM_QWEN3_GATE_UP_WINDOWED_BWD=0 and =1 variants
loss compare passes
profile JSON for every expected run exists
selected/nonselected row ownership counters pass
drop000 and drop010 both pass native/dropout ownership checks
```

Latency/profiling gate:

```text
full-step median and p95 are recorded
selected recompute+dX region time is recorded
native path improves full-step median, or the validation fails for integration
drop010 median step time <= 1.10 * matching drop000 policy median
profile JSON includes native op stats copied from Stage 7 counter schema where
available: cpu_weight_stream_multiplier, w_cache_read_model_ratio, phase ms,
selected/nonselected row ownership counters
```

Only after Stage 10 passes may broad E2E/product tile tuning begin. Limited
op-level kernel tuning is allowed earlier only inside Stage 7, Stage 8, or
Stage 9 gates, and it must rerun the op-level validation/profiling script before
any Stage 10 integration attempt.

## Correctness Test Cases

Direct cases:

```text
empty:
  E=0 or M_selected=0, returns empty outputs and zero counters

one_group:
  E=2, H=256, I=512, M_selected=8, r=8, P=32, Q=4, BM=32, BK=128

ragged_groups:
  E=4, H=384, I=768, M_selected=37, r=16, P=32, Q=4, BM=32, BK=128

partial_window:
  E=3, H=512, I=650, M_selected=19, r=8, P=32, Q=8, BM=32, BK=128

qwen_shape_smallM:
  E=8, H=2048, I=768, M_selected=512, r=64, P=32, Q=8, BM=64, BK=512

qwen_shape_routed:
  E=128, H=2048, I=768, M_selected=32768, r=64, P=32, Q=8, BM=64, BK=512
```

Reference:

```text
move W_gate_up_cpu to CUDA for reference only
compute gate/up base with torch matmul/einsum
compute LoRA-B deltas with torch matmul/einsum
compute dgate/dup in FP32
compute grad_x_base = dgate @ W_gate + dup @ W_up
```

Tolerances:

```text
grad_gate_sel max_abs/max_rel <= 3e-2
grad_up_sel   max_abs/max_rel <= 3e-2
grad_x_base_sel max_abs/max_rel <= 5e-2
```

## Tuning Plan

Do not tune before Stage 5-6 correctness passes and Stage 7 has produced an
initial NSYS/NCU profile for the cache-first schedule and counters. If initial
Stage 7 profiling misses the required latency threshold, only limited op-level
kernel tuning is allowed; LF E2E integration remains blocked until Stage 7
passes.

No-early-conclusion rule:

```text
Do not conclude "cache_first_window is viable" until:
  Stage 5 correctness passes
  Stage 6 Python integration correctness passes
  Stage 7 NCU artifacts exist for fill_w_cache, recompute_act,
  down_lora_activation, and dx_window
  native_selected_region_ms <= 0.90 * current_selected_region_ms

Do not conclude "the idea is too slow" after only the default P=32 Q=8 BM=64
BK=512 G_work=128 run. First classify the bottleneck from NCU, then run the
minimum matrix below or record why a required run is blocked.

Every failed Stage 7 or tuning run must write to kernels_progress.md:
  bottleneck_class:
    fill_bandwidth
    recompute_w_cache_reads
    dx_tensor_utilization
    activation_epilogue_spill
    launch_overhead
    workspace_memory
    correctness
  measured blocking counters
  hypothesis selected from the list below
  exact next configuration to try
  reason if no next configuration is technically valid
```

Overhead-reduction hypotheses to test, in order:

```text
1. Fix fill_w_cache vectorization before changing math.
   Hypothesis: poor fill bandwidth is caused by scalar address generation,
   uncoalesced CPU-pinned reads, or unaligned HBM writes.
   Test: check fill_w_cache_effective_GBps against pinned_h2d_baseline_GBps;
   try vector widths 16B/32B, aligned H strides, and one CTA mapping that
   linearizes contiguous H first.
   Accept only if fill_w_cache_effective_GBps improves and
   cpu_weight_stream_multiplier remains <= 1.01.

2. Increase BM before changing the algorithm.
   Hypothesis: larger BM reduces repeated w_cache reads across M blocks.
   Test: sweep BM=64,96,128,192 with fixed P/Q/BK/G_work.
   Accept only if recompute_reads_w_cache_bytes falls or latency improves, while
   achieved occupancy stays >= 25 percent and local-memory bytes stay within the
   NCU threshold.

3. Tune P/BN for epilogue register pressure versus tensor-core shape.
   Hypothesis: P=32/BN=64 is balanced, but P=16 can reduce epilogue registers
   and P=64 can improve tensor-core efficiency when local memory stays zero.
   Test: sweep P=16,32,64 with fixed Q/BM/BK/G_work.
   Accept only if grad_pair_window traffic, local-memory bytes, and
   native_selected_region_ms improve or remain within 2 percent while memory
   stays in budget.

4. Tune Q for dX efficiency and launch count.
   Hypothesis: Q=12 can improve dX K efficiency and reduce window count; Q=4 can
   reduce active HBM when memory pressure dominates.
   Test: sweep Q=4,8,12,24 with fixed P/BM/BK/G_work.
   Accept only if native_selected_region_ms improves and peak HBM stays within
   budget.

5. Tune BK for tensor-core efficiency and memory locality.
   Hypothesis: BK=512 is good for Qwen3 H=2048, but BK=256 or 128 may improve
   occupancy or reduce scoreboard stalls if register/shared-memory pressure is
   limiting.
   Test: sweep BK=128,256,512 with fixed P/Q/BM/G_work.
   Accept only if recompute_act and dx_window tensor utilization or latency
   improves without raising w_cache_read_model_ratio above the accepted bound.

6. Tune G_work and expert-wave order.
   Hypothesis: G_work=128 minimizes chunks, but smaller G_work may improve cache
   locality or reduce workspace pressure; expert-wave ordering can improve L2
   reuse without changing CPU stream count.
   Test: sweep G_work=16,32,64,128 and num_experts_per_wave=16,32,64,128.
   Accept only if peak HBM, L2 hit rate, and native_selected_region_ms improve
   while selected row ownership and CPU stream counters remain correct.

7. Keep padding metadata-only.
   Hypothesis: materialized padded X or grad_pair tensors erase the memory and
   latency benefit.
   Test: NCU/allocator counters show no padded X/grad_pair allocation in
   Stage 7 pass artifacts; padded_to_global_row/padded_to_local_row metadata is
   used instead.

8. Split or fuse the recompute/down-LoRA epilogue based on NCU, not preference.
   Hypothesis: fusing avoids HBM round trips, but over-fusion can raise register
   pressure and spill; splitting can help only if the saved occupancy exceeds
   the extra launch/HBM cost.
   Test: compare fused_recompute_down_lora_kernel=true and false for the same
   P/Q/BM/BK/G_work with NCU local-memory, register, occupancy, and HBM bytes.
   Accept the fused path only if local-memory bytes stay within threshold and
   native_selected_region_ms is no worse; accept the split path only if it wins
   latency without allocating full-window gate/up/act.

9. Reduce launch/runtime overhead with CUDA graph capture or one native
   orchestration API.
   Hypothesis: phase launches are valid only when captured; uncaptured overhead
   above the launch-overhead threshold must trigger native orchestration, not
   LF E2E.
   Test: NSYS reports launch count, CPU runtime overhead, and captured versus
   uncaptured timing.

10. If cache_first_window is correct but recompute w_cache reads dominate, try
   seed_group_direct.
   Hypothesis: compute one row group directly from the CPU-streamed W tile while
   filling w_cache, keeping cpu_weight_stream_multiplier at 1.0x.
   Test: Stage 8 only; recompute_reads_w_cache_bytes drops versus same-shape
   cache_first_window and full/layer latency improves.

11. Use persistent role-separated fusion only after Stage 7 proves launch/sync or
   repeated HBM cache reads are the measured blocker.
   Hypothesis: DeepGEMM/Megakernels-style loader/consumer/storer separation helps
   only when the simple phase schedule has already exposed that bottleneck.
   Test: Stage 8+ only with fresh NSYS/NCU and an updated kernels_progress.md
   decision.
```

Every tuning iteration must run the op-level profiling script first:

```bash
GPU=1 \
MODES=cache_first_window \
CASES=qwen_shape_smallM \
P_VALUES=32 \
Q_VALUES=8 \
BM_VALUES=64 \
BK_VALUES=512 \
G_WORK_VALUES=128 \
RUN_CORRECTNESS=1 \
RUN_CUDA_EVENTS=1 \
RUN_NSYS=1 \
RUN_NCU=1 \
OUTPUT_ROOT=profiling/qwen3_gate_up_windowed_bwd/tune_ops \
bash scripts/lf/profile_qwen3_gate_up_windowed_ops.sh
```

Only configurations whose `summary.json` passes correctness, CUDA-event timing,
NSYS schedule checks, and NCU metric gates are eligible for LF E2E testing.

Primary sweep:

```text
mode:
  cache_first_window
  seed_group_direct after cache_first
  all_rows_direct only after seed_group_direct

P:
  16, 32, 64

Q for P=32:
  4, 8, 12, 24

G_work:
  16, 32, 64, 128

BM:
  16, 32, 64, 96, 128, 192

BK:
  128, 256, 512

num_experts_per_wave:
  16, 32, 64, 128
```

Minimum matrix:

```text
baseline current selected recompute path
cache_first_window P=32 Q=4  BM=64  BK=512 G_work=128
cache_first_window P=32 Q=8  BM=64  BK=512 G_work=128
cache_first_window P=32 Q=12 BM=64  BK=512 G_work=128
cache_first_window P=64 Q=4  BM=64  BK=512 G_work=128
cache_first_window P=32 Q=8  BM=32  BK=512 G_work=128
cache_first_window P=32 Q=8  BM=128 BK=512 G_work=128
cache_first_window P=32 Q=8  BM=64  BK=256 G_work=128
cache_first_window P=32 Q=8  BM=64  BK=128 G_work=128
cache_first_window P=32 Q=8  BM=64  BK=512 G_work=64
seed_group_direct  P=32 Q=8  BM=64  BK=512 G_work=128 M_group=64
seed_group_direct  P=32 Q=8  BM=64  BK=512 G_work=128 M_group=128
all_rows_direct    P=32 Q=8  BM=64  BK=512 G_work=128 M_group=all
```

Change one variable per run. A tuning change is accepted only if:

```text
all correctness tolerances pass
CPU stream multiplier remains within the mode bound
w_cache read bytes are reported against R_M tile model
NCU reports are present for fill_w_cache, recompute_act,
down_lora_activation, and dx_window
local-memory thresholds pass and w_cache_read_model_ratio remains within the
mode-specific bound
the directly affected op median latency improves or stays within 2 percent
Stage 5 native_e2e correctness still passes
Stage 6 Python integration correctness still passes
Stage 7 aggregate gate improves, or remains within 2 percent if the change is
claimed as neutral cleanup
Stage 10 full-step median improves before merging into the E2E path
```

## Correctness Traps

```text
Do not accumulate dX after every P panel. Accumulate after each Q-window.
Do not tile the cache over H only; dgate/dup require full-H gate/up reduction.
Do not process gate and up in unrelated windows.
Do not pass original expert ids to GEMMs reading compact w_cache.
Do not include selected rows in the later nonselected _grouped_base_dx call.
Do not use global atomics into full grad_packed from the native kernel.
Do not rerun dropout in backward.
Do not silently fall back to torch inside the native path.
Do not claim success from matrix-size traffic bytes; compare against R_M.
Do not merge lazy_direct if it duplicates CPU W streams.
```

## Expected Ceiling

For the Qwen3-30B-A3B `drop010` profile:

```text
step total:                    about 2231 ms
selected gate/up recompute:    about 271.5 ms
selected gate/up base dX:      about 269.8 ms
target region total:           about 541.3 ms
```

The impossible hard ceiling from deleting selected gate/up base dX is:

```text
269.8 / 2231 = 12.1 percent E2E
```

That cannot happen because the new path still computes dX. The larger selected
recompute+dX region is `541.3 / 2231 = 24.3 percent`, also impossible to remove.

Realistic good-case if the current path is CPU-stream limited and measured
cache reads are controlled:

```text
saved time: 80-160 ms
E2E win:    about 4-7 percent
```

If cache-first is correct but `w_cache_read_model_ratio` is high, the next
engineering step is `seed_group_direct` or `all_rows_direct`, not claiming the
cache-first schedule is a timing win.
