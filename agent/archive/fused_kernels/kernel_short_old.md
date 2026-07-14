# V1.5 Record: Windowed W-Cache Gate/Up Recompute + dX Backward

Status: recommended research prototype for the selected-recompute backward path.
This is not a forward-path AsymGEMM change and not a generic optimizer/offload
kernel. It targets Qwen3-style MoE gate/up backward when forward activation
recompute is enabled for selected routed rows.

The key correction in this version is that `cache_first_window` is the required
first implementation. The more aggressive lazy/direct schedule is an
optimization that must prove it does not duplicate CPU weight streams or spill
hidden accumulator state.

How to read this file:

```text
This is the concise design and pseudocode record.
It is representative of kernel.md for:
  scope and math
  runtime switch and before/after comparison
  tile/workspace sizing
  selected metadata and grouped-GEMM padding contracts
  cache_first_window pseudocode
  later lazy/direct variants
  required counters
  Stage 7 stop rule

kernel.md remains authoritative for:
  exact file/function ownership by stage
  exact validation commands and CLI
  full NSYS/NCU artifact requirements
  full progress-ledger template
```

## Scope

This is not the entire MLP backward. It is the selected-recompute gate/up-base
subpath inside the existing custom backward.

```text
Included in V1.5:
  selected gate/up base recompute
  selected activation backward to dgate/dup
  selected gate/up base dX

Inputs expected from the existing backward:
  selected packed X rows
  selected expert metadata and per-expert row offsets
  dact = dY W_down + d LoRA_down / d act
  saved/recomputed gate/up LoRA low-rank outputs or a way to rebuild them

Still outside this kernel in V1:
  down base backward that produces dact_base
  down LoRA-A grad
  down LoRA-B grad and dact_lora
  gate/up LoRA A/B grads and LoRA dX
  nonselected gate/up base dX
  routing scatter/combine and top-level gradient merges
```

The only required performance win is to remove the second selected
`W_gate_up_cpu` stream. Do not expand V1 into a full MLP-backward fusion unless
this narrower kernel already passes correctness and timing gates.

## Original
```
dact_base = dY W_down

dact_lora = dS_down A_down
dact = dact_base + dact_lora

RECOMP: gate_tmp1 = X W_gate^T + LoRA_gate(X)
RECOMP: up_tmp1   = X W_up^T   + LoRA_up(X)
RECOMP: act_tmp1  = silu(gate_tmp1) * up_tmp1
dS_down = scale * dY B_down
dA_down = dS_down^T act_tmp1

RECOMP: gate_tmp2 = X W_gate^T + LoRA_gate(X)
RECOMP: up_tmp2   = X W_up^T   + LoRA_up(X)
dgate = dact * up_tmp2 * silu_grad(gate_tmp2)
dup   = dact * silu(gate_tmp2)

dX_base_selected = dgate W_gate + dup W_up
```

## Math Scope

```text
Context forward:
gate = X W_gate^T + LoRA_gate
up   = X W_up^T   + LoRA_up
act  = silu(gate) * up

Y_down = act W_down^T + LoRA_down(act)

Backward:
dact_base = dY W_down
dact_lora = scale * dY B_down A_down
dact = dact_base + dact_lora

RECOMP: gate_tmp1 = X W_gate^T + LoRA_gate(X)
RECOMP: up_tmp1   = X W_up^T   + LoRA_up(X)
RECOMP: act_tmp1  = silu(gate_tmp1) * up_tmp1

dS_down = scale * dY B_down
dA_down = dS_down^T act_tmp

# # RECOMP: gate_tmp2 = X W_gate^T + LoRA_gate(X)
# # RECOMP: up_tmp2   = X W_up^T   + LoRA_up(X)

sig = sigmoid(gate) # CPU
silu_gate = gate * sig
silu_grad_gate = sig * (1 + gate * (1 - sig))

dgate = dact * up_tmp2 * silu_grad_gate # 2 AsymGEMM
dup   = dact * silu(gate_tmp2) # AsymGEMM where silu(gate_tmp2)
dX_base = dgate W_gate + dup W_up # Cache to HBM
```

Qwen3 packed base weight:

```text
W_gate_up_cpu: [E, 2I, H] CPU-pinned BF16 HostWeight
W_gate = W_gate_up_cpu[:, 0:I, :]
W_up   = W_gate_up_cpu[:, I:2I, :]
```

For the profiled Qwen3-30B-A3B shape used by the current reports:

```text
E = 128 experts
H = 2048 hidden dim
I = 768 intermediate dim per gate/up branch
W_gate_up_cpu per layer = E * 2I * H * 2 bytes = 768 MiB
W_gate_up_cpu over 48 layers = 36 GiB per matrix-size full stream
```

That 36 GiB number is a lower bound for one pass over the selected experts'
weights. A normal M-tiled GEMM may load the same W tile once per selected M
block. The actual traffic model must include:

```text
M_blocks_e = ceil(M_e / BM)
R_M = weighted average M_blocks_e over active selected experts
```

For the Qwen example with `M_selected=32768`, `E=128`, and `BM=64`, the average
selected rows per expert is 256, so a uniform-routing estimate gives
`R_M = ceil(256/64) = 4`. Real routing skew can make the max expert larger.

## Current Selected-Recompute Backward

Current selected recompute streams `W_gate_up_cpu` twice:

```text
# selected rows, existing AsymGEMM-style tiles: BM=64, BN=64, BK=512

# 1. recompute full gate/up, CPU W stream #1
# GEMM: M=selected rows, N=2I gate/up, K=H hidden
for m0 in selected_rows(e) step BM:                         # M loop
  for n0 in 0..2I step BN:                                  # N loop
    acc = 0                                                 # [BM, BN] = [M,N]
    for h0 in 0..H step BK:                                 # K loop
      W_tile = stream_cpu(W_gate_up[e, n0:n0+BN, h0:h0+BK]) # [BN, BK] = [N,K]
      X_tile = X_sel[m0:m0+BM, h0:h0+BK]                    # [BM, BK] = [M,K]
      acc += X_tile @ W_tile^T                              # [BM, BN] = [M,N]
    pair_acc[m0:m0+BM, n0:n0+BN] = acc                      # [BM, BN] = [M,N]

gate, up = split(pair_acc + LoRA_gate_up)                  # [Ms, I], [Ms, I]
activated = silu(gate) * up                                # [Ms, I]
dact = down_backward(activated)                            # outside V1.5
dgate = dact * up * silu_grad(gate)                        # [Ms, I]
dup   = dact * silu(gate)                                  # [Ms, I]
grad_pair = concat(dgate, dup)                             # [Ms, 2I]

# 2. selected base dX, CPU W stream #2
# GEMM: M=selected rows, N=H hidden output, K=2I gate/up reduce
for m0 in selected_rows(e) step BM:                         # M loop
  for h0 in 0..H step BK:                                   # N loop for dX
    acc = 0                                                 # [BM, BK] = [M,N]
    for n0 in 0..2I step BN:                                # K loop for dX
      W_tile = stream_cpu(W_gate_up[e, n0:n0+BN, h0:h0+BK]) # [BN, BK] = [K,N]
      G_tile = grad_pair[m0:m0+BM, n0:n0+BN]                # [BM, BN] = [M,K]
      acc += G_tile @ W_tile                                # [BM, BK] = [M,N]
    dx_acc[m0:m0+BM, h0:h0+BK] = acc                        # [BM, BK] = [M,N]

# selected W_gate_up CPU stream = 2.0x in the logical lower-bound model.
# Actual tile traffic may be 2.0x * R_M if each M block streams W separately.
```

## Design Invariants

V1.5 is a scheduling change, not a math approximation. The non-negotiable
contract is to remove the second selected `W_gate_up_cpu` stream without
changing selected-row math or double-counting selected dX.

Common invariants for all V1.5 modes:

```text
1. Selected W_gate_up CPU bytes are staged once per active selected expert and
   window, unless a mode explicitly reports a larger multiplier and is rejected.
2. The staged window is stored in a reusable HBM cache:
     w_cache[e_rel, 0:Kdx, 0:H]
3. Selected base dX reads w_cache; it does not stream W_gate_up_cpu.
4. Selected rows are removed from the old selected base-dX path.
5. Nonselected rows keep the old path.
6. dX uses a window K dimension, Kdx = 2P*Q, not a skinny K=2P panel.
7. Extra workspaces are scoped to one layer/op window and must be reused, not
   kept alive across layers.
8. Every implementation must report CPU stream bytes, HBM cache read/write
   bytes, launch count, local-memory spills, and selected/nonselected row counts.
```

Runtime switch:

```text
ASYM_QWEN3_GATE_UP_WINDOWED_BWD=0 or unset:
  use the current selected-recompute backward.
  This is the before/baseline profile.

ASYM_QWEN3_GATE_UP_WINDOWED_BWD=1:
  use the V1.5 native windowed gate/up recompute+dX path when all guards pass.
  This is the after/candidate profile.

Every reported speedup must compare both values with the same seed, shape,
dropout, expert policy, P/Q/BM/BK/G_work, warmup count, and latency iterations.
```

One public native API:

```text
qwen3_gate_up_recompute_bwd_sm100_bf16_windowed(
  x_sel, dact_sel,
  gate_low_rank_sel, up_low_rank_sel,
  gate_lora_B, up_lora_B,
  gate_up_weight_cpu,
  selected_offsets, selected_experts,
  p=32, q=8, bm=64, bk=512, g_work=128,
  lora_scale=1.0,
  mode="cache_first_window",
)
```

Python may call it only when:

```text
SM100
selected tensors are BF16
layer.gate_up_base is AsymGroupedFrozenLinear
layer.gate_up_base.host_weight.weight is CPU-pinned BF16 [E, 2I, H]
layer.lora_dtype == torch.bfloat16
selected rows are nonempty
ASYM_QWEN3_GATE_UP_WINDOWED_BWD=1
```

Otherwise Python falls back to the current selected-recompute backward.

The current selected path has this weight movement:

```text
current selected path:
  recompute: CPU W stream
  dX:        CPU W stream
```

The required V1.5 baseline is:

```text
cache_first_window:
  fill:      CPU W stream -> HBM w_cache
  recompute: HBM w_cache read
  dX:        HBM w_cache read
```

This is not the absolute minimum traffic schedule, but it is the correct first
implementation because it guarantees `1.0x` CPU staging while keeping normal
M-tiled GEMM parallelism. It trades the removed CPU stream for HBM traffic,
which is the right trade only if CPU-to-GPU weight movement is the bottleneck.

The optimized mode is:

```text
lazy_direct_window:
  fill/recompute: CPU W stream is also consumed directly by some recompute work
  fill:           same W tile is written once to HBM w_cache
  remaining work: any recompute work not covered directly reads HBM w_cache
  dX:             HBM w_cache read
```

`lazy_direct_window` is allowed only as an optimization over
`cache_first_window`. It must not hide duplicated CPU reads. A normal M-tiled
GEMM cannot broadcast one CPU-streamed W tile across all independent M CTAs for
free. Therefore the lazy mode must prove either:

```text
all_rows_direct:
  one cooperative work unit covers all selected rows for that expert/panel
  cpu_weight_stream_multiplier <= 1.05
  pair_acc/global-spill overhead is below the cache_first recompute-read saving

seed_group_direct:
  the CPU-streaming work unit computes only one M_group while filling w_cache
  all other M_groups read w_cache
  cpu_weight_stream_multiplier <= 1.01
```

If the implementation moves `stream_cpu(W_tile)` inside the M-tile loop, the CPU
stream multiplier becomes roughly `ceil(M_e / M_group)` for that expert. That is
usually worse than the current path and must fail validation.

## Tile Symbols

Use the same `M/N/K` convention as the current pseudocode.

```text
mode:
  cache_first_window is the required first implementation.
  lazy_direct_window is a later optimization gated by counters.

BM:
  M tile, selected routed rows.
  Default 64.
  Sweep 32, 64, 96, 128, 192. Use smaller BM for long-tail experts and larger
  BM when per-expert selected rows are dense.

BK:
  H tile.
  In recompute, BK is the K-reduction tile over hidden H.
  In dX, BK is the N-output tile over hidden H.
  Default 512.

P:
  intermediate columns per gate or up panel.
  BN = 2P paired gate/up columns.
  Default P=32, so BN=64.

BN:
  recompute N tile over paired gate/up columns.
  BN = 2P.
  Not an independent API/CLI knob in V1.

Q:
  number of P-panels per dX window.
  W = P*Q intermediate columns per gate or up window.
  Kdx = 2W = 2P*Q paired gate/up columns used by selected dX.
  Default Q=8, so Kdx=512.

G_work:
  number of active selected experts processed in one cache workspace.
  Default 128 for one-GPU Qwen3-30B-A3B if memory allows.

M_work:
  selected rows covered by the current expert_chunk.
  For one-GPU Qwen3 with G_work=128, worst case is M_selected.

M_group:
  maximum rows per expert/panel that a lazy-direct work unit consumes from the
  just-streamed CPU W tile.
  Applies only to lazy_direct_window.
  For all_rows_direct, M_group must cover M_e.
  For seed_group_direct, M_group is usually BM or 2*BM and only reduces part of
  the recompute-side w_cache read.

num_experts_per_wave:
  scheduler grouping used to keep SMs busy under row imbalance.
  Sweep 16, 32, 64, 128 or derive from expected rows per expert.
```

Default kernel shape:

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
num_experts_per_wave = 32 or 64 first, then tune
```

For Qwen3 `I=768`, the default has exactly three windows:

```text
I / W = 768 / 256 = 3 windows
```

## Workspaces

Required workspace for an expert chunk:

```text
w_cache:
  shape [G_work, Kdx, H] bf16
  bytes = G_work * Kdx * H * 2
        = G_work * 2P * Q * H * 2

grad_pair_window:
  logical shape [M_work, Kdx] bf16
  bytes = M_work * Kdx * 2
  holds concat(dgate_window, dup_window)

pair_acc:
  cache_first_window:
    per-M-tile accumulator [BM, BN] fp32, normally registers/tensor memory.
  lazy_direct all_rows_direct:
    logical accumulator [M_e, BN] fp32 if one work unit covers all rows.
    This is only acceptable if it does not spill enough to erase the win.
  lazy_direct seed_group_direct:
    per-M_group accumulator [M_group, BN] fp32.

dx_acc:
  shape [M_selected, H] fp32 for exact first implementation
  bytes = M_selected * H * 4

ready metadata, persistent-fused implementation only:
  w_tile_ready     [G_work, Kdx/BN, H/BK] uint32 or packed bitset
  grad_panel_ready [M_blocks or expert/panel blocks] uint32 or packed bitset
  usually below 1 MiB for the Qwen3 default
```

Selected metadata and row maps:

```text
selected_offsets and selected_experts describe compact selected rows only.
selected_offsets[0] == 0.
selected_offsets[-1] == M_selected.
selected_offsets is monotonic.
selected_experts[0:Gs] are original expert ids in [0, E).
selected_experts[-1] == -1.
Gs == 0 is represented as selected_offsets=[0], selected_experts=[-1].

Use compact selected row ids for:
  X_sel, dact, low-rank LoRA inputs, grad_gate/up outputs, dx_acc.

Use chunk-local row ids for:
  grad_pair_window [M_work, Kdx].

Build:
  chunk_row_ids      [M_work] maps local rows -> compact selected rows
  row_to_chunk_local [M_selected] maps compact selected rows -> local rows
```

Grouped-GEMM padding contract:

```text
The existing contiguous grouped GEMM style pads each group to a BM multiple and
uses pair offsets plus a sentinel expert list. V1 native code owns that metadata.
Do not pad/unpad around this path in Python for latency-gated runs.

Per expert_chunk build:
  chunk_offsets_unpadded [G_chunk + 1]
  chunk_offsets_padded   [G_chunk + 1]
  pair_offsets_padded    [2 * G_chunk]
  grouped_experts        [G_chunk + 1] with trailing -1
  grouped_list_size      G_chunk + 1
  padded_to_global_row   [M_padded_work], -1 for padding rows
  padded_to_local_row    [M_padded_work], -1 for padding rows

grad_pair_window remains unpadded [M_work, Kdx].
```

For the Qwen3-30B-A3B profile shape, assuming worst selected routed rows
`M_selected = 4096 * 8 = 32768`, `H=2048`, `P=32`, `G_work=128`:

```text
Q=4:  Kdx=256,  w_cache=128 MiB, grad_pair_window=16 MiB
Q=8:  Kdx=512,  w_cache=256 MiB, grad_pair_window=32 MiB
Q=12: Kdx=768,  w_cache=384 MiB, grad_pair_window=48 MiB
Q=24: Kdx=1536, w_cache=768 MiB, grad_pair_window=96 MiB

dx_acc fp32 = 32768 * 2048 * 4 = 256 MiB
optional final grad_x_base_sel bf16 transient = 128 MiB
```

Default `Q=8` incremental active memory is roughly:

```text
w_cache              256 MiB
grad_pair_window      32 MiB
pair_acc            <= 8 MiB logical worst for all_rows_direct only
dx_acc               256 MiB
optional bf16 output 128 MiB
ready metadata        <1 MiB
total gross          552-680 MiB
```

This is around 0.8-1.0 percent of the observed 67.76 GiB peak in the Qwen3
`drop010` profile, if workspaces are reused per layer/op and not kept
persistent across layers.

Full-run weight/cache traffic for Qwen3-30B-A3B, 48 layers, in matrix-size
lower-bound bytes:

```text
one full selected W_gate_up stream = 36 GiB

current selected recompute + selected dX:
  CPU read: 72 GiB

cache_first_window:
  CPU read:  36 GiB
  HBM write: 36 GiB
  HBM read:  72 GiB  # one recompute read, one dX read

lazy_direct_window best case:
  CPU read:  36 GiB
  HBM write: 36 GiB
  HBM read:  36 GiB  # dX read only
```

This lower-bound table is useful for sizing peak cache memory, but it is too
optimistic for timing. The timing model must include M-block reuse:

```text
current selected recompute + selected dX:
  CPU read ~= 2 * R_M * 36 GiB

cache_first_window:
  CPU read  = 1 * 36 GiB
  HBM write = 1 * 36 GiB
  HBM read ~= 2 * R_M * 36 GiB  # recompute and dX, before L2 reuse

lazy_direct_window best case:
  CPU read  = 1 * 36 GiB
  HBM write = 1 * 36 GiB
  HBM read ~= 1 * R_M * 36 GiB  # dX only, before L2 reuse
```

With the uniform Qwen estimate `R_M=4`, the extra recompute-side HBM read in
`cache_first_window` versus best-case lazy direct is about `4 * 36 = 144 GiB`,
or 48-72 ms at 2-3 TiB/s idealized HBM bandwidth. L2 reuse and a weight-stationary
scheduler can reduce physical HBM traffic, but the implementation must measure
it. This is why `cache_first_window` is the practical correctness baseline,
while `seed_group_direct` and `all_rows_direct` are measured optimizations for
reducing recompute-side cache reads.

## Borrowed Implementation Techniques

The design should borrow these patterns from DeepGEMM MegaMoE and Megakernels:

```text
Reviewed source anchors:
  /workspace/AsymGEMM-SFT/third_party/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh
  /workspace/AsymGEMM-SFT/third_party/DeepGEMM/deep_gemm/include/deep_gemm/layout/mega_moe.cuh
  /workspace/AsymGEMM-SFT/third_party/DeepGEMM/csrc/jit_kernels/heuristics/mega_moe.hpp
  /workspace/AsymGEMM-SFT/third_party/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh
  /workspace/AsymGEMM-SFT/third_party/Megakernels/include/megakernel.cuh
  /workspace/AsymGEMM-SFT/third_party/Megakernels/include/controller/controller.cuh
  https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh

DeepGEMM MegaMoE scheduler:
  pre-cache per-expert token counts
  schedule by expert waves, not by scanning routing metadata in hot loops
  keep a persistent for_each_block-style scheduler over block phases

DeepGEMM MegaMoE layout:
  allocate one explicit aligned workspace and slice it into views
  keep arrival masks/counters in workspace when phases overlap
  pad token pools by candidate BM alignment to avoid ragged hot-path branches

DeepGEMM heuristics:
  choose BM from expected rows per expert, not one hard-coded value
  choose num_experts_per_wave so imbalance still leaves enough blocks for SMs
  include smem-based stage count in the kernel configuration
  do not copy FP8/FP4 block sizes blindly; BF16 AsymGEMM should keep BK=512 as
  the first default and only sweep BK=128/256 after profiling SMEM pressure

DeepGEMM SM100 kernel:
  split roles for weight load, token/X load, MMA issue, and epilogue
  use explicit full/empty barriers for staged pipelines
  avoid clever on-demand waits when expert waves already guarantee completion

Megakernels:
  use role-separated loader/consumer/storer/controller structure
  keep stage timing in the kernel-level instruction/workspace state
  avoid Python-level per-panel orchestration
```

For this V1.5 kernel, the practical translation is:

```text
1. Build selected row maps, per-expert row offsets, and selected expert waves
   before entering the hot kernel path.
2. Use an explicit workspace object with typed slices:
     w_cache, grad_pair_window, dx_acc, ready flags, counters.
3. Keep CPU-W loading, X/G loading, MMA, activation-grad epilogue, and dX
   accumulation as explicit phases or warp/CTA roles.
4. First validate with simple phase barriers or separate CUDA-graph-captured
   kernels. Only then collapse to a persistent fused op if launch overhead or
   lost overlap is visible in profiling.
```

What to borrow vs not borrow:

```text
Borrow:
  cached per-expert counts
  expert-wave scheduling
  one aligned workspace with typed slices
  padded grouped-GEMM metadata
  explicit producer/consumer phases
  timing/byte counters
  register/local-memory/occupancy checks

Do not borrow for V1A:
  FP8/FP4 quant layouts
  NVLink/symmetric-buffer dispatch
  token combine/reduce logic
  TMA paths for CPU-pinned HostWeight loads
  global spin-wait protocols
  an instruction-VM/persistent kernel before Stage 7 proves it is needed
```

## Detailed Algorithm

Inputs:

```text
X_sel                [M_selected, H] bf16/bf16-like
dact                 [M_selected, I] bf16/bf16-like
gate_low_rank_sel    [M_selected, r] bf16
up_low_rank_sel      [M_selected, r] bf16
gate_lora_B          [E, I, r] bf16
up_lora_B            [E, I, r] bf16
selected_offsets     per selected expert row ranges
selected_experts     original expert ids for selected groups
W_gate_up_cpu        [E, 2I, H] CPU-pinned bf16
```

Outputs:

```text
grad_x_base_sel      [M_selected, H] bf16, accumulated through dx_acc
grad_gate_sel        [M_selected, I] bf16 dgate for selected rows
grad_up_sel          [M_selected, I] bf16 dup for selected rows
```

`grad_gate_sel` and `grad_up_sel` are not LoRA parameter gradients. They are
returned so the existing Python backward can compute gate/up LoRA A/B gradients
and LoRA dX outside V1.5.

### Algorithm A: cache_first_window

This is the required first implementation. It has one CPU stream and preserves
ordinary M-tiled parallelism. Recompute pays an HBM read from `w_cache`.

The implementation can start as three CUDA-graph-captured kernels per
`iwin0/expert_chunk`:

```text
fill_w_cache
recompute_grad_window
dx_window
```

For production, use one persistent native op with phase counters or ready flags
instead of Python-level per-panel launches. Do not launch one kernel per `P`
panel.

For Qwen3 default `I=768, P=32, Q=8`, there are three windows. With
`G_work=128`, the phase-split validation path is nine launches per layer, or
432 launches over 48 layers. Captured launch overhead should be a few
milliseconds; uncaptured Python launch overhead can be larger and must be
measured. If launch overhead is visible in NSYS, collapse the phases into one
persistent op before judging the design.

Detailed pseudocode:

```text
# W = P*Q intermediate indices per window.
# BN = 2P paired gate/up columns per recompute tile.
# Kdx = 2W paired gate/up columns per selected dX window.

for iwin0 in 0..I step W:
  # Window covers:
  #   gate[iwin0:iwin0+W]
  #   up[iwin0:iwin0+W]
  # Cached packed window width is Kdx = 2W.

  for expert_chunk in active_selected_experts step G_work:
    allocate/reuse w_cache          [G_work, Kdx, H] bf16
    allocate/reuse grad_pair_window [M_work, Kdx] bf16
    build chunk_row_ids and row_to_chunk_local
    build padded grouped-GEMM metadata for this chunk
    zero grad_pair_window

    # Stage A0: fill the W cache.
    # CPU W stream is outside every M loop, so selected CPU W bytes are 1.0x.

    for e_rel, e in enumerate(expert_chunk):
      rows_e_global = selected compact row ids for expert e

      for n0 in 0..Kdx step BN:                            # N loop in recompute view
        q = n0 / BN
        i0 = iwin0 + q * P                                 # gate/up intermediate offset
        valid_i = clamp(I - i0, 0, P)

        if valid_i == 0:
          zero-fill w_cache[e_rel, n0:n0+BN, :]
          continue

        for h0 in 0..H step BK:                            # K loop in recompute view
          Wg = stream_cpu(W_gate_up_cpu[e, i0:i0+valid_i,     h0:h0+BK])
          Wu = stream_cpu(W_gate_up_cpu[e, I+i0:I+i0+valid_i, h0:h0+BK])
          W_tile = concat_and_zero_pad(Wg, Wu, P)           # [BN, BK] = [N,K]

          w_cache[e_rel, n0:n0+BN, h0:h0+BK] = W_tile      # HBM write once
          mark_w_tile_ready(e_rel, n0, h0)                 # persistent fused op only

    # Stage A1: recompute gate/up from w_cache and form grad_pair_window.
    # GEMM: M=selected rows, N=BN paired gate/up columns, K=H hidden.

    for e_rel, e in enumerate(expert_chunk):
      rows_e_global = selected compact row ids for expert e
      rows_e_local = row_to_chunk_local[rows_e_global]
      M_e = number of selected rows for expert e

      for n0 in 0..Kdx step BN:                            # N loop in recompute view
        q = n0 / BN
        i0 = iwin0 + q * P
        valid_i = clamp(I - i0, 0, P)

        if valid_i == 0:
          zero-fill grad_pair_window[rows_e_local, n0:n0+BN]
          continue

        for me0 in 0..M_e step BM:                         # M loop
          me1 = min(me0 + BM, M_e)
          rows_global = rows_e_global[me0:me1]
          rows_local = rows_e_local[me0:me1]
          pair_acc = zeros([me1-me0, BN], fp32)             # [M, N]

          for h0 in 0..H step BK:                          # K loop
            wait_w_tile_ready(e_rel, n0, h0)               # no-op with launch barriers
            W_tile = load_hbm(w_cache[e_rel, n0:n0+BN, h0:h0+BK])
            X_tile = X_sel[rows_global, h0:h0+BK]          # [M, K]
            pair_acc += X_tile @ W_tile^T                  # [M, N]

          lora_gate_p = zero_pad(
              gate_low_rank_sel[rows_global] @ gate_lora_B[e, i0:i0+valid_i].T, P
          )
          lora_up_p = zero_pad(
              up_low_rank_sel[rows_global] @ up_lora_B[e, i0:i0+valid_i].T, P
          )
          gate_p = pair_acc[:, 0:P]   + lora_gate_p        # [M, P]
          up_p   = pair_acc[:, P:2*P] + lora_up_p          # [M, P]
          act_p = silu(gate_p) * up_p                      # [M, P]

          dact_v = dact[rows_global, i0:i0+valid_i]        # [M, valid_i]
          gate_v = gate_p[:, 0:valid_i]
          up_v   = up_p[:, 0:valid_i]

          dgate_v = dact_v * up_v * silu_grad(gate_v)      # [M, valid_i]
          dup_v   = dact_v * silu(gate_v)                  # [M, valid_i]

          grad_pair_window[rows_local, n0:n0+BN] = 0
          grad_pair_window[rows_local, n0:n0+valid_i] = dgate_v
          grad_pair_window[rows_local, n0+P:n0+P+valid_i] = dup_v
          grad_gate_sel[rows_global, i0:i0+valid_i] = dgate_v
          grad_up_sel[rows_global, i0:i0+valid_i] = dup_v
          mark_grad_panel_ready(rows_local, n0)            # persistent fused op only

    # Stage B: selected base dX from the cached window.
    #
    # GEMM: M=selected rows, N=H hidden output, K=Kdx.
    # This reads w_cache from HBM once and does not stream W_gate_up_cpu.

    for e_rel, e in enumerate(expert_chunk):
      rows_e_global = selected compact row ids for expert e
      rows_e_local = row_to_chunk_local[rows_e_global]
      M_e = number of selected rows for expert e

      for me0 in 0..M_e step BM:                           # M loop
        me1 = min(me0 + BM, M_e)
        rows_global = rows_e_global[me0:me1]
        rows_local = rows_e_local[me0:me1]

        for h0 in 0..H step BK:                            # N loop in dX view
          dx_tile = zeros([me1-me0, BK], fp32)              # [M, N]

          for n0 in 0..Kdx step BN:                        # K loop in dX view
            wait_grad_panel_ready(rows_local, n0)          # no-op with launch barriers
            wait_w_tile_ready(e_rel, n0, h0)
            G_tile = grad_pair_window[rows_local, n0:n0+BN] # [M, K]
            W_tile = w_cache[e_rel, n0:n0+BN, h0:h0+BK]    # [K, N]
            dx_tile += G_tile @ W_tile                     # [M, N]

          dx_acc[rows_global, h0:h0+BK] += dx_tile

    discard_or_reuse w_cache and grad_pair_window

grad_x_base_sel = cast_or_store(dx_acc)
return grad_x_base_sel, grad_gate_sel, grad_up_sel
```

Implementation notes:

```text
1. With separate kernels, the launch boundary is the global barrier.
2. With one persistent fused op, `w_tile_ready` and `grad_panel_ready` replace
   launch barriers. Keep these flags coarse and aligned; do not make one flag
   per element.
3. `pair_acc` above is per M tile. Do not materialize [M_e, BN] for
   cache_first_window.
4. Do not add down LoRA-A gradient ownership to V1.5. A later extension may
   revisit it only after the required gate/up-base recompute+dX path passes.
5. `dx_window` must be an in-repo native C++/CUDA BF16 tensor-core path called
   from this API. It reads `grad_pair_window` and `w_cache` directly, accumulates
   fp32, emits the required counters, and appears in NCU. Python/Torch fallback,
   materialized padded tensors, the old selected `_grouped_base_dx` path, or a
   second `W_gate_up_cpu` stream do not count for Stage 7.
```

### Algorithm B: lazy_direct_window

This is an optimization after Algorithm A is correct and profiled.

There are two acceptable lazy/direct variants.

#### B1: seed_group_direct

This variant never duplicates CPU reads. While the fill work unit streams a W
tile from CPU, it also computes one row group directly from that just-streamed
tile. The remaining row groups read `w_cache` like Algorithm A.

This variant requires a persistent fused op or a CTA/cluster design where the
CPU-W loader and seed MMA consumer share the just-loaded W tile. It is not
available in the simple phase-split validation path, because that path writes W
to HBM first and only later launches recompute.

```text
for iwin0 in 0..I step W:
  for expert_chunk in active_selected_experts step G_work:
    allocate/reuse w_cache and grad_pair_window
    build the same chunk_row_ids and row_to_chunk_local maps as Algorithm A

    for e_rel, e in enumerate(expert_chunk):
      rows_e_global = selected compact row ids for expert e
      rows_e_local = row_to_chunk_local[rows_e_global]

      for n0 in 0..Kdx step BN:
        q = n0 / BN
        i0 = iwin0 + q * P
        valid_i = clamp(I - i0, 0, P)

        seed_rows_global = first M_group rows_e_global rows for this expert/panel
        seed_rows_local = row_to_chunk_local[seed_rows_global]
        seed_pair_acc = zeros([len(seed_rows_global), BN], fp32)

        for h0 in 0..H step BK:
          Wg = stream_cpu(W_gate_up_cpu[e, i0:i0+valid_i,     h0:h0+BK])
          Wu = stream_cpu(W_gate_up_cpu[e, I+i0:I+i0+valid_i, h0:h0+BK])
          W_tile = concat_and_zero_pad(Wg, Wu, P)

          w_cache[e_rel, n0:n0+BN, h0:h0+BK] = W_tile
          mark_w_tile_ready(e_rel, n0, h0)

          X_tile = X_sel[seed_rows_global, h0:h0+BK]
          seed_pair_acc += X_tile @ W_tile^T

        compute_activation_backward_for(seed_rows_global, seed_pair_acc)
        write grad_pair_window[seed_rows_local, n0:n0+BN]
        write grad_gate_sel/grad_up_sel[seed_rows_global, i0:i0+valid_i]
        mark_grad_panel_ready(seed_rows_local, n0)

    # Remaining rows use cache_first recompute from w_cache.
    recompute all rows not covered by seed_rows using Algorithm A Stage A1
    run Algorithm A Stage B dX
```

This saves only part of the recompute-side HBM read, but it is low-risk because
CPU bytes stay at `1.0x`. It is the first lazy optimization to test if
`cache_first_window` is correct but the extra recompute HBM read is visible.

#### B2: all_rows_direct

This is the theoretical best lazy schedule:

```text
for each expert e and recompute panel n0 in the current iwin0 window:
  rows_e_global = selected compact row ids for expert e
  rows_e_local = row_to_chunk_local[rows_e_global]
  M_e = len(rows_e_global)
  q = n0 / BN
  i0 = iwin0 + q * P
  valid_i = clamp(I - i0, 0, P)
  pair_acc = zeros([len(rows_e_global), BN], fp32)

  for h0 in 0..H step BK:
    Wg = stream_cpu(W_gate_up_cpu[e, i0:i0+valid_i,     h0:h0+BK])
    Wu = stream_cpu(W_gate_up_cpu[e, I+i0:I+i0+valid_i, h0:h0+BK])
    W_tile = concat_and_zero_pad(Wg, Wu, P)
    w_cache[e_rel, n0:n0+BN, h0:h0+BK] = W_tile

    for me0 in 0..M_e step BM:
      X_tile = X_sel[rows_e_global[me0:me0+BM], h0:h0+BK]
      pair_acc[me0:me0+BM, :] += X_tile @ W_tile^T

  compute activation backward for all rows_e_global
  write grad_pair_window[rows_e_local, n0:n0+BN]
  write grad_gate_sel/grad_up_sel[rows_e_global, i0:i0+valid_i]

run Algorithm A Stage B dX
```

This achieves:

```text
CPU read:  1.0x
HBM write: 1.0x
HBM read:  dX only
```

But it is only acceptable when the implementation proves:

```text
cpu_weight_stream_multiplier <= 1.05
pair_acc_global_bytes and local-memory spill bytes are small
occupancy does not collapse from one huge expert/panel work unit
latency is better than cache_first_window for the same P/Q/BM/BK/G_work
```

For Qwen3 with `M_selected=32768` and `E=128`, average selected rows per expert
is about 256, but routing skew can make some experts much larger. If
`M_group=64`, a direct schedule that re-streams W per group has an average CPU
multiplier around `ceil(256/64)=4`, which is immediately invalid. This is the
main trap the counters must catch.

## Why Windowed dX Is Required

If dX accumulates after every tiny panel with `P=32`, its K dimension is only
`2P=64`. For `I=768`, that gives 24 separate dX updates. Repeated global
`dx_acc` read/write can erase the CPU-stream win.

Windowing uses several recompute panels before dX:

```text
P=32, Q=4:  Kdx=256
P=32, Q=8:  Kdx=512
P=32, Q=12: Kdx=768
P=32, Q=24: Kdx=1536
```

The default `Q=8` keeps memory modest while giving dX a useful K size. `Q=12`
is the first speed-mode candidate. `Q=24` is closest to the old full-K dX but
uses a large cache.

## Why The Pure-SMEM Plan Was Rejected

The dependency is the blocker:

```text
dgate/dup need final gate/up.
final gate/up need the complete H reduction.
```

A normal AsymGEMM tile:

```text
W_tile = W[n0:n0+BN, h0:h0+BK]
```

cannot be immediately reused for dX while it is in SMEM, because `dgate/dup`
are not known until all `h0` tiles have contributed. Holding the full-H panel in
SMEM is too expensive:

```text
resident W bytes = 2 * P * H * sizeof(bf16) = 4 * P * H bytes
```

For example:

```text
H = 4096, P = 8  -> 128 KiB just for W_pair
H = 4096, P = 16 -> 256 KiB just for W_pair, already impossible
```

Existing SM100 BF16 AsymGEMM with `BM=64, BN=64, BK=512` already uses roughly:

```text
A/X double buffer: 2 * 64 * 512 * 2 = 128 KiB
B/W tile:          1 * 64 * 512 * 2 = 64 KiB
C/D staging:       about 16 KiB
total:             about 208 KiB before barriers/alignment
SM100 capacity:    232448 bytes
```

So do not build the pure-SMEM full-H version first.

## Sweep Plan

Correctness test shapes:

```text
one_group:
  E=2, H=256,  I=512, M_selected=8,   P=32, Q=4,  BM=32, BK=128

ragged_groups:
  E=4, H=384,  I=768, M_selected=37,  P=32, Q=4,  BM=32, BK=128

partial_window:
  E=3, H=512,  I=650, M_selected=19,  P=32, Q=8,  BM=32, BK=128

qwen_shape_smallM:
  E=8, H=2048, I=768, M_selected=512, P=32, Q=8,  BM=64, BK=512

qwen_shape_routed:
  E=128, H=2048, I=768, M_selected=32768, P=32, Q=8, BM=64, BK=512
```

Primary performance sweep:

```text
Fixed first:
  H=2048, I=768, dtype=bf16, dx_acc=fp32
  active selected rows from real profile metadata when possible

Sweep mode:
  cache_first_window       required baseline
  seed_group_direct        after cache_first is correct
  all_rows_direct          only for small/medium M_e or after spill counters pass

Sweep P:
  P=16  -> BN=32, lower cache, likely worse tensor utilization
  P=32  -> BN=64, default
  P=64  -> BN=128, speed candidate, higher pair_acc pressure

Sweep Q for P=32:
  Q=4   -> Kdx=256,  w_cache=128 MiB at G_work=128
  Q=8   -> Kdx=512,  w_cache=256 MiB at G_work=128
  Q=12  -> Kdx=768,  w_cache=384 MiB at G_work=128
  Q=24  -> Kdx=1536, w_cache=768 MiB at G_work=128

Sweep G_work:
  16, 32, 64, 128
  choose largest value that does not push peak HBM or allocator fragmentation too high

Sweep num_experts_per_wave:
  16, 32, 64, 128
  choose the smallest value whose M/N block count still keeps SMs busy after
  routing imbalance

Sweep BM:
  16, 32, 64, 96, 128, 192
  default 64
  start 32/64/128; add 96/192 when rows per expert are dense

Sweep BK:
  128, 256, 512
  default 512 for BF16 AsymGEMM-style hidden tiles
  use 128/256 if SMEM pressure or async CPU staging requires more stages

Sweep M_group for seed_group_direct:
  BM, 2*BM, 4*BM
  report fraction of recompute HBM read avoided

Sweep M_group for all_rows_direct:
  64, 128, 256, 512, all
  reject any case with cpu_weight_stream_multiplier > 1.05
```

Implementation staging plan:

```text
Stage 0:
  baseline instrumentation only.

Stage 1:
  selected metadata + CPU W -> HBM w_cache.
  No GEMM yet.

Stage 2:
  recompute gate/up from w_cache, add LoRA-B contribution, compute dgate/dup.

Stage 3:
  selected dX window from grad_pair_window + w_cache.
  No second CPU W stream.

Stage 4:
  expose one full native API that owns all windows/chunks/phases.

Stage 5:
  Python integration for dropout 0.00 behind ASYM_QWEN3_GATE_UP_WINDOWED_BWD.

Stage 6:
  Python integration for dropout 0.10 using saved S_gate/S_up.

Stage 7:
  paired before/after NSYS+NCU decision gate.
  Write agent/fused_kernels/kernels_progress.md and stop.
  Decision is one of:
    passed_for_e2e
    failed_correctness
    failed_latency
    failed_ncu_or_traffic
    blocked_missing_artifacts
```

Do not write the whole kernel at once. Each stage must pass its own correctness
and latency/profiling checks before the next stage starts.

After Stage 7, do not start `seed_group_direct`, `all_rows_direct`, or LF E2E
until the user reviews `kernels_progress.md` and explicitly approves.

Minimum benchmark matrix:

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

Do not tune every parameter at once. Establish `cache_first_window
P=32,Q=8,BM=64,BK=512` first, then vary one parameter per run.

Profiler passes:

```text
NSYS:
  end-to-end step time
  per-layer selected recompute+dX region time
  launch count and CPU runtime overhead

NCU:
  HBM read/write bytes
  L2 hit rate for w_cache reads
  tensor-core utilization
  local memory load/store bytes
  register count and occupancy
  source counters around CPU W stream and w_cache loads

Allocator/memory:
  peak allocated HBM
  workspace reuse address
  fragmentation or allocator retries
```

Reject an optimization if it improves an isolated microbenchmark but loses in
the graph-captured layer benchmark or full SFT step.

Overhead-reduction hypotheses to test:

```text
1. BM tuning first:
   Sweep BM=64,96,128,192. Larger BM should reduce repeated w_cache reads across
   M blocks. Accept only if NCU shows no spill/occupancy regression.

2. Q tuning second:
   Sweep Q=4,8,12,24. Q=12 may improve dX efficiency and reduce windows; Q=4 is
   for memory pressure. Accept only if latency improves within peak-HBM budget.

3. Metadata-only padding:
   Use padded_to_global_row/padded_to_local_row metadata. Reject materialized
   padded X or grad_pair tensors in Stage 7 pass artifacts.

4. Launch/runtime overhead:
   Use CUDA graph capture or one native orchestration API before judging
   cache_first_window. NSYS must show whether launch overhead is material.

5. seed_group_direct only after cache_first:
   Try it only if cache_first is correct and NCU shows recompute w_cache reads
   dominate. It must keep cpu_weight_stream_multiplier <= 1.01.

6. Persistent role-separated fusion only after Stage 7:
   Borrow DeepGEMM/Megakernels loader/consumer/storer structure only if Stage 7
   proves launch/sync or repeated HBM reads are the measured blocker.
```

## Performance Hazards To Check

```text
CPU stream duplication:
  Any CPU multiplier above 1.05 usually invalidates the method.

Extra HBM traffic:
  cache_first_window adds w_cache write + recompute read + dX read.
  This should still be cheaper than the removed CPU stream, but only profiling
  can prove it.

M-block W reloads:
  Matrix-size bytes are a lower bound. If each M block reloads the same W tile,
  cache reads scale with R_M = average ceil(M_e / BM).
  Use expert-wave scheduling, L2 locality, and possible persistent weight-
  stationary ordering to reduce physical HBM reads.
  Reject schedules whose measured w_cache_read_model_ratio is much larger than
  1.0 unless NCU explains L2 reuse or extra traffic.

Tiny dX K:
  Q too small makes dX inefficient and repeats dx_acc updates.
  Prefer Q=8 first; try Q=12 if memory allows.

pair_acc spill:
  all_rows_direct can silently spill [M_e, BN] state.
  Reject if local/global spill traffic approaches the saved recompute HBM read.

launch overhead:
  Separate phase kernels are acceptable for validation if captured in a CUDA
  graph. If uncaptured launch overhead is visible, collapse phases into one
  native op.

routing imbalance:
  Expert row skew can leave SMs idle or make all_rows_direct too large.
  Use selected expert waves and real profile metadata.

future down LoRA-A fusion:
  Down LoRA-A gradient is outside V1.5. Do not add it to this path unless a
  later extension defines separate correctness, contention, and latency gates.

LoRA/dropout replay:
  The recomputed activation must use the exact same dropout mask/scale and
  low-rank gate/up contribution as the reference path.
```

Minimum acceptance thresholds:

```text
cache_first_window:
  cpu_weight_stream_multiplier <= 1.01
  old_selected_base_dx_rows == 0
  new_selected_base_dx_rows == selected_recompute_rows
  max_abs_error/max_rel_error within current recompute tolerance
  peak HBM increase <= configured workspace budget
  before LF E2E integration:
    native_selected_region_ms <= 0.90 * current_selected_region_ms
    for qwen_shape_smallM and one large/routed case
    projected_full_step_saving_ms >= 40 ms or
    projected_full_step_saving_percent >= 2 percent for the Qwen3-30B-A3B estimate
  for integration: full-step latency improves
  for continued research only: isolated target-region latency improves enough to
  justify persistent fusion work

seed_group_direct:
  all cache_first thresholds
  cpu_weight_stream_multiplier <= 1.01
  recompute_reads_w_cache_bytes lower than same-shape cache_first_window
  full-step latency better than cache_first

all_rows_direct:
  all cache_first thresholds
  recompute_reads_w_cache_bytes == 0 or near zero
  cpu_weight_stream_multiplier <= 1.05
  pair_acc spill/local-memory bytes below the measured recompute-read saving
  full-step latency better than cache_first and seed_group_direct
```

## Required Counters

The implementation must emit machine-readable counters for every validation run:

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
hbm_w_cache_write_bytes
hbm_w_cache_valid_write_bytes
hbm_w_cache_padding_zero_write_bytes
recompute_reads_w_cache_bytes
dx_reads_w_cache_bytes
w_cache_bytes_allocated_peak
grad_pair_window_bytes_allocated_peak
pair_acc_bytes_allocated_peak
pair_acc_global_bytes
local_memory_load_bytes
local_memory_store_bytes
dx_acc_bytes_allocated_peak
num_windows
num_recompute_panels
num_dx_windows
num_phase_launches
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
native_kernel_consumed_saved_s
native_kernel_consumed_dropout_masks
dropout_backward_rng_advanced
no_aten_native_dropout_in_backward
dropout_replay_mismatches
workspace_reused
metadata_ms
fill_w_cache_ms
recompute_grad_window_ms
dx_window_ms
native_total_ms
baseline_reference_ms
current_selected_region_ms
native_selected_region_ms
projected_full_step_saving_ms
projected_full_step_saving_percent
max_abs_error
max_rel_error
median_latency_ms
p50_step_ms
p95_step_ms
peak_hbm_bytes
```

Required equalities or bounds:

```text
expected_cpu_weight_bytes_min ==
  sum over iwin0 and active selected experts:
    2 * clamp(I - iwin0, 0, W) * H * sizeof(bf16)

cpu_weight_stream_multiplier =
  cpu_weight_bytes_staged / expected_cpu_weight_bytes_min

cache_first_window:
  cpu_weight_stream_multiplier <= 1.01
  recompute_reads_w_cache_bytes > 0
  dx_reads_w_cache_bytes > 0

seed_group_direct:
  cpu_weight_stream_multiplier <= 1.01
  recompute_reads_w_cache_bytes <
    recompute_reads_w_cache_bytes from the same-shape cache_first_window run
  dx_reads_w_cache_bytes > 0

all_rows_direct:
  cpu_weight_stream_multiplier <= 1.05
  recompute_reads_w_cache_bytes == 0 or explained by boundary/metadata reads
  dx_reads_w_cache_bytes > 0

common:
  sum_m_blocks == sum over active selected experts ceil(M_e / BM)
  m_block_weight_reuse_factor_R_M == sum_m_blocks / active_selected_experts
  m_padded_work == sum over experts in the chunk ceil(M_e / BM) * BM
  expected_current_cpu_weight_bytes_tile_model ==
    2 * m_block_weight_reuse_factor_R_M * expected_cpu_weight_bytes_min
  expected_w_cache_read_bytes_tile_model ==
    mode-specific scheduled cache-read model using actual M blocks:
      cache_first_window: 2 * R_M * expected_cpu_weight_bytes_min
      seed_group_direct: between 1 * R_M and 2 * R_M times
                         expected_cpu_weight_bytes_min, depending on direct rows
      all_rows_direct:   1 * R_M * expected_cpu_weight_bytes_min
  w_cache_read_multiplier_vs_cpu_min =
    (recompute_reads_w_cache_bytes + dx_reads_w_cache_bytes)
    / expected_cpu_weight_bytes_min
  w_cache_read_model_ratio =
    (recompute_reads_w_cache_bytes + dx_reads_w_cache_bytes)
    / expected_w_cache_read_bytes_tile_model
  hbm_w_cache_write_bytes ==
    hbm_w_cache_valid_write_bytes + hbm_w_cache_padding_zero_write_bytes
  hbm_w_cache_valid_write_bytes == expected_cpu_weight_bytes_min
  old_selected_base_dx_rows == 0
  new_selected_base_dx_rows == selected_recompute_rows
  nonselected_base_dx_rows == reference_nonselected_rows
  w_cache_bytes_allocated_peak <= G_work * Kdx * H * sizeof(bf16)
  workspace_reused == true
  dropout_replay_mismatches == 0
```

For the Qwen3 default, expected per-window cache size must match:

```text
Q=8, G_work=128, Kdx=512, H=2048:
w_cache_bytes_allocated_peak <= 128 * 512 * 2048 * 2 = 256 MiB
```

## Required Proof Before Integration

```text
1. CPU-staged selected W_gate_up bytes are 1.0x for cache_first_window.
2. The old selected base-dX call no longer processes selected recompute rows.
3. Nonselected base dX is bitwise or tolerance-equivalent to the fallback path.
4. Default P=32 preserves recompute BN=64; P=16 is not the default.
5. Default Q=8 makes selected base dX use K=512 per window.
6. dX accumulates once per window, not after every P panel.
7. Extra active HBM from w_cache/grad_pair_window/pair_acc/dx_acc is within
   the configured budget and is released/reused promptly.
8. dx_acc memory mode is explicit: fp32 exact first, lower-memory mode later.
9. LoRA gate/up contribution and dropout replay match the reference path.
10. Down LoRA-A gradient remains outside V1.5; base recompute+dX must work
    without it.
11. NCU shows no unexpected local-memory spill or HBM traffic explosion.
12. NSYS shows launch overhead is not erasing the target-region win.
13. Measured w_cache read bytes are compared against the R_M tile model, not
    only against matrix-size lower-bound bytes.
14. Compare cache_first_window against the current two-stream selected path.
15. Stage 7 meets native_selected_region_ms <= 0.90 * current_selected_region_ms
    for qwen_shape_smallM and one large/routed case before LF E2E integration.
16. Compare lazy_direct modes against cache_first_window, not only against the
    old current path.
17. Keep the current selected-recompute implementation as fallback until full
    step correctness and timing gates pass.
18. Write every stage result to agent/fused_kernels/kernels_progress.md. After
    the Stage 7 NSYS/NCU cache-first profile gate, stop and ask the user to
    approve Stage 8/9/10 before continuing.
```

Self-check before claiming success:

```text
If recompute_reads_w_cache_bytes == 0 but cpu_weight_stream_multiplier > 1.05,
the implementation is not a win; it duplicated CPU streams.

If cpu_weight_stream_multiplier == 1.0 but recompute_reads_w_cache_bytes is
large, that is expected for cache_first_window; judge it by end-to-end time.

If microbench time improves but full-step time does not, the design is not ready
for integration.

If peak HBM rises by more than the workspace estimate, the workspace is leaking
or being kept across layers.
```

## Expected Ceiling

For the Qwen3-30B-A3B `drop010` profile:

```text
step total:                    about 2231 ms
selected gate/up recompute:    about 271.5 ms
selected gate/up base dX:      about 269.8 ms
target region total:           about 541.3 ms
```

The impossible hard ceiling is removing all selected gate/up base dX:

```text
269.8 / 2231 = 12.1 percent E2E
```

That cannot happen because dX still performs GEMM work and writes `dx_acc`. The
larger target region, selected gate/up recompute plus selected gate/up base dX,
is about `541.3 / 2231 = 24.3 percent`, but that is also impossible because the
new path still computes recompute, activation backward, and dX.

The realistic good-case range, assuming the current path is CPU-stream limited
and measured cache-read traffic is not much worse than the R_M tile model, is:

```text
saved time: 80-160 ms
E2E win:    about 4-7 percent
```

This method looks better when the current path is CPU-weight-stream limited. It
looks worse when the current time is dominated by tensor-core compute,
launch/runtime overhead, routing/scatter, repeated `dx_acc` HBM traffic, or
repeated `w_cache` HBM reads across many M blocks. If `cache_first_window` is
correct but the measured `w_cache_read_model_ratio` is high, the next step is
`seed_group_direct` or `all_rows_direct`, not claiming the cache-first schedule
is a timing win.
