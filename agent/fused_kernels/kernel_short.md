# V2 Record: Down-LoRA-Aware Windowed W-Cache Backward

Status: concise design and pseudocode record for the selected-recompute Qwen3
gate/up backward path. `kernel.md` remains authoritative for exact file
ownership, validation commands, NCU requirements, and progress-ledger rules.

How to read this file:

```text
1. Scenario and math define exactly what this kernel owns.
2. The continuous pseudocode is the implementation schedule.
3. Required counters and stages are only the minimum audit checklist.
```

## Scenario And Math

This kernel is only for selected routed rows in Qwen3 MoE LoRA SFT backward.
It is not forward AsymGEMM, optimizer offload, or a generic MoE kernel.

The integrated problem is:

```text
current selected backward:
  recompute selected gate/up for act: CPU W_gate_up stream
  selected gate/up base dX:           CPU W_gate_up stream
  selected CPU stream count:          2x

naive direct native op:
  Python still recomputes gate/up for down-LoRA act first
  native direct op then recomputes gate/up again for dX cache
  selected CPU stream count: 2x -> 2x, not a real integrated win

V2 target:
  native owns selected act -> dA_down -> dgate/dup -> dX
  selected CPU stream count: 2x -> 1x
```

Context forward:

```text
gate = X W_gate^T + lora_scale * S_gate B_gate^T
up   = X W_up^T   + lora_scale * S_up   B_up^T
act  = silu(gate) * up

Y_down = act W_down^T + lora_scale * S_down B_down^T
S_down = dropout_replay(act, down_mask) A_down^T
```

Down-LoRA backward produces parameter grads and an activation grad:

```text
Given LoRA_down(act) = lora_scale * S_down B_down^T:
  dB_down   = dY^T S_down * lora_scale
  dS_down   = dY B_down * lora_scale
  dA_down   = dS_down^T dropout_replay(act, down_mask)
  dact_lora = dropout_backward(dS_down A_down, down_mask)

Because Y_down also has the base branch act W_down^T:
  dact_base = dY W_down
  dact = dact_base + dact_lora
```

Boundary rule:

```text
dact_lora and final dact stay outside native:
  they use dY, W_down, B_down, A_down, and the down mask
  they do not use W_gate, W_up, gate, up, or act

dA_down_selected is inside native:
  it needs dropout_replay(act, down_mask)
  act is produced by the same selected gate/up recompute that dgate/dup need

Native therefore consumes dact_sel and dS_down_sel as inputs; it does not
compute dact_lora or dB_down.
```

## Original Selected Schedule

```text
Outside current selected gate/up recompute:
dact_base = dY W_down
dS_down   = dY B_down * lora_scale
dact_raw  = dS_down A_down
dact_lora = dropout_backward(dact_raw, down_mask)
dact      = dact_base + dact_lora
dB_down_selected += dY^T S_down * lora_scale

# CPU W_gate_up stream #1
RECOMP1: gate_tmp1 = X W_gate^T + lora_scale * S_gate B_gate^T
RECOMP1: up_tmp1   = X W_up^T   + lora_scale * S_up   B_up^T
RECOMP1: act_tmp1  = silu(gate_tmp1) * up_tmp1

dA_down_selected += dS_down^T dropout_replay(act_tmp1, down_mask)

# CPU W_gate_up stream #2
RECOMP2: gate_tmp2 = X W_gate^T + lora_scale * S_gate B_gate^T
RECOMP2: up_tmp2   = X W_up^T   + lora_scale * S_up   B_up^T

dgate = dact * up_tmp2 * silu_grad(gate_tmp2)
dup   = dact * silu(gate_tmp2)
dX_base_selected = dgate W_gate + dup W_up

Selected CPU W_gate_up stream count = 2x.
```

## V2 Math Scope

```text
Outside before V2 native:
dact_base = dY W_down
dS_down   = dY B_down * lora_scale
dact_raw  = dS_down A_down
dact_lora = dropout_backward(dact_raw, down_mask)
dact      = dact_base + dact_lora
dB_down_selected += dY^T S_down * lora_scale

Inside V2 native for selected rows:

# CPU W_gate_up stream #1 only: fill reusable HBM cache.
FILL: w_cache = W_gate_up_cpu selected expert/window slice

# Recompute once from w_cache, not directly from CPU W again.
RECOMP: gate_tmp = X W_gate^T + lora_scale * S_gate B_gate^T
RECOMP: up_tmp   = X W_up^T   + lora_scale * S_up   B_up^T
RECOMP: act_tmp  = silu(gate_tmp) * up_tmp

dA_down_selected += dS_down^T dropout_replay(act_tmp, down_mask)

dgate = dact * up_tmp * silu_grad(gate_tmp)
dup   = dact * silu(gate_tmp)

# dX reads w_cache. It must not stream W_gate_up_cpu.
dX_base_selected = dgate W_gate + dup W_up

Outside after V2 native:
merge selected dA_down with nonselected down-LoRA A grads
keep outside dB_down result
use dgate/dup to run existing gate/up LoRA backward and LoRA dX
run existing nonselected gate/up base dX

Selected CPU W_gate_up stream count = 1x.
```

Gate/up LoRA correctness rule:

```text
S_gate and S_up must be the exact forward low-rank tensors after LoRA-A dropout
replay. Native consumes S_gate/S_up and applies lora_scale exactly once.
```

Packed CPU base weight:

```text
W_gate_up_cpu: [E, 2I, H] CPU-pinned BF16
W_gate = W_gate_up_cpu[:, 0:I, :]
W_up   = W_gate_up_cpu[:, I:2I, :]
```

Default tile symbols:

```text
P=32                         # gate/up columns per panel
BN=2P=64                     # paired gate/up columns per recompute panel
Q=8                          # P-panels per dX window
W=P*Q=256                    # gate/up intermediate columns per window
Kdx=2W=2P*Q=512              # paired dX K dimension per window
BM=64                        # selected rows per row tile
BK=512                       # hidden tile
G_work=128                   # selected experts per workspace chunk
```

Only these buffers have full-window lifetime:

```text
w_cache          [G_chunk, Kdx, H] bf16
grad_pair_window [M_work, Kdx] bf16
dx_acc           [M_selected, H] fp32, zero once
grad_gate_sel    [M_selected, I] bf16
grad_up_sel      [M_selected, I] bf16
```

These must be tile-local in the latency-gated path:

```text
pair_acc [BM, 2*P], gate/up/act [BM, P]
```

Do not allocate full-window `gate/up/act [M_work, W]` for Stage 7 pass.

## Detailed Algorithm

`native_rows(e)` below means routed token-expert rows for expert `e` that are
sent to this native kernel. If the integration enables the kernel for all routed
rows, then `native_rows(e)` is all routed rows for that expert.

For readability, `m0` indexes rows in the native row set. The implementation
uses chunk-local row ids for `grad_pair_window` and maps them back to compact
selected-row ids with the metadata described in `kernel.md`.

### Tile-Level View

```text
Outside native:
  dact_base = dY W_down
  dS_down   = dY B_down * lora_scale
  dact_raw  = dS_down A_down
  dact_lora = dropout_backward(dact_raw, down_mask)
  dact      = dact_base + dact_lora
  dB_down   = dY^T S_down * lora_scale

Current selected-recompute backward, shown only as the 2-stream thing V2
replaces:

  # 1. Recompute gate/up/act and produce dgate/dup, CPU W stream #1.
  # GEMM: M=native rows, N=2P paired gate/up panel, K=H hidden.
  for e in active_experts:
    for i0 in 0..I step P:
      for m0 in native_rows(e) step BM:
        pair_acc = zeros([BM, 2P], fp32)
        for h0 in 0..H step BK:
          Wg = stream_cpu(W_gate_up_cpu[e, i0:i0+P,     h0:h0+BK])
          Wu = stream_cpu(W_gate_up_cpu[e, I+i0:I+i0+P, h0:h0+BK])
          W_pair = concat(Wg, Wu)                         # [2P, BK]
          X_tile = X_sel[m0:m0+BM, h0:h0+BK]              # [BM, BK]
          pair_acc += X_tile @ W_pair^T                   # [BM, 2P]

        gate, up = split(pair_acc + LoRA_gate_up_panel)   # [BM,P], [BM,P]
        act = silu(gate) * up                             # [BM,P]
        dA_down[:, i0:i0+P] += dS_down[m0:m0+BM]^T @ dropout_replay(act)

        dact_p = dact[m0:m0+BM, i0:i0+P]
        dgate = dact_p * up * silu_grad(gate)              # [BM,P]
        dup   = dact_p * silu(gate)                       # [BM,P]

        grad_pair[m0:m0+BM, i0:i0+P]   = dgate
        grad_pair[m0:m0+BM, I+i0:I+i0+P] = dup

  # 2. Selected base dX, CPU W stream #2.
  # GEMM: M=native rows, N=H hidden output, K=2I gate/up reduce.
  for e in active_experts:
    for m0 in native_rows(e) step BM:
      for h0 in 0..H step BK:
        dx_tile = zeros([BM, BK], fp32)
        for i0 in 0..I step P:
          Wg = stream_cpu(W_gate_up_cpu[e, i0:i0+P,     h0:h0+BK])
          Wu = stream_cpu(W_gate_up_cpu[e, I+i0:I+i0+P, h0:h0+BK])
          Gg = grad_pair[m0:m0+BM, i0:i0+P]
          Gu = grad_pair[m0:m0+BM, I+i0:I+i0+P]
          dx_tile += Gg @ Wg + Gu @ Wu                    # [BM,BK]
        dx_acc[m0:m0+BM, h0:h0+BK] += dx_tile

V2 native backward:

  for expert_chunk in active_experts step G_work:
    allocate/reuse:
      w_cache          [G_chunk, 2P*Q, H] bf16
      grad_pair_window [M_work, 2P*Q] bf16

    for iwin0 in 0..I step P*Q:

      # A0. Fill HBM cache. CPU W stream is outside row loops and happens once.
      for e_rel, e in enumerate(expert_chunk):
        for q in 0..Q-1:
          i0 = iwin0 + q*P
          n0 = q*(2P)
          valid_i = clamp(I - i0, 0, P)
          for h0 in 0..H step BK:
            Wg = stream_cpu(W_gate_up_cpu[e, i0:i0+valid_i,     h0:h0+BK])
            Wu = stream_cpu(W_gate_up_cpu[e, I+i0:I+i0+valid_i, h0:h0+BK])
            w_cache[e_rel, n0:n0+P,   h0:h0+BK] = zero_pad(Wg, P)
            w_cache[e_rel, n0+P:n0+2P,h0:h0+BK] = zero_pad(Wu, P)

      # A1. Recompute gate/up from w_cache, then immediately consume act for
      # selected dA_down and consume passed dact for dgate/dup.
      for e_rel, e in enumerate(expert_chunk):
        rows = native_rows(e)
        for q in 0..Q-1:
          i0 = iwin0 + q*P
          n0 = q*(2P)
          valid_i = clamp(I - i0, 0, P)

          for m0 in rows step BM:
            # GEMM: M=BM rows, N=2P paired gate/up panel, K=H hidden.
            pair_acc = zeros([BM, 2P], fp32)
            for h0 in 0..H step BK:
              X_tile = X_sel[m0:m0+BM, h0:h0+BK]                  # [BM,BK]
              W_pair = w_cache[e_rel, n0:n0+2P, h0:h0+BK]         # [2P,BK]
              pair_acc += X_tile @ W_pair^T                       # [BM,2P]

            gate_base, up_base = split(pair_acc)                  # [BM,P]
            gate_delta = S_gate[m0:m0+BM] @ B_gate[e, i0:i0+valid_i]^T
            up_delta   = S_up[m0:m0+BM]   @ B_up[e,   i0:i0+valid_i]^T
            gate = gate_base + lora_scale * zero_pad(gate_delta, P)
            up   = up_base   + lora_scale * zero_pad(up_delta, P)
            act  = silu(gate) * up                                # [BM,P]

            # Down-LoRA A grad uses recomputed act. Native does not compute
            # dact_lora or dB_down.
            act_drop = dropout_replay(act[:, 0:valid_i], down_mask)
            dS = dS_down_sel[m0:m0+BM, :]                         # [BM,r_down]
            grad_down_lora_A[e, :, i0:i0+valid_i] += dS^T @ act_drop

            # Activation backward uses final dact passed from outside.
            dact_p = dact_sel[m0:m0+BM, i0:i0+valid_i]             # [BM,valid_i]
            dgate = dact_p * up[:,0:valid_i] * silu_grad(gate[:,0:valid_i])
            dup   = dact_p * silu(gate[:,0:valid_i])

            grad_gate_sel[m0:m0+BM, i0:i0+valid_i] = dgate
            grad_up_sel[m0:m0+BM,   i0:i0+valid_i] = dup

            # Store paired K panel for dX. Invalid columns are zero.
            grad_pair_window[m0:m0+BM, n0:n0+2P] = zero
            grad_pair_window[m0:m0+BM, n0:n0+valid_i] = dgate
            grad_pair_window[m0:m0+BM, n0+P:n0+P+valid_i] = dup

      # B. Selected base dX from HBM cache. No CPU W stream here.
      for e_rel, e in enumerate(expert_chunk):
        rows = native_rows(e)
        for m0 in rows step BM:
          for h0 in 0..H step BK:
            # GEMM: M=BM rows, N=BK hidden output, K=2P*Q paired reduce.
            dx_tile = zeros([BM, BK], fp32)
            for q in 0..Q-1:
              n0 = q*(2P)
              G_pair = grad_pair_window[m0:m0+BM, n0:n0+2P]       # [BM,2P]
              W_pair = w_cache[e_rel, n0:n0+2P, h0:h0+BK]         # [2P,BK]
              dx_tile += G_pair @ W_pair                          # [BM,BK]
            dx_acc[m0:m0+BM, h0:h0+BK] += dx_tile

      discard_or_reuse(w_cache, grad_pair_window)

Result:
  selected CPU W_gate_up stream count changes from 2x to 1x.
  Extra HBM is w_cache + grad_pair_window + dx_acc.
```

### API-Oriented Skeleton

```text
function qwen3_gate_up_down_lora_bwd_sm100_bf16_windowed(
    x_sel,                    # [M_selected, H] bf16
    dact_sel,                 # [M_selected, I] bf16/fp32, final dact
    dS_down_sel,              # [M_selected, r_down] bf16/fp32
    gate_low_rank_sel,        # [M_selected, r_gate] bf16, forward S_gate
    up_low_rank_sel,          # [M_selected, r_gate] bf16, forward S_up
    gate_lora_B,              # [E, I, r_gate] bf16
    up_lora_B,                # [E, I, r_gate] bf16
    down_mask_packed_sel,     # selected forward down-dropout mask, or empty
    gate_up_weight_cpu,       # CPU pinned bf16 [E, 2I, H]
    selected_offsets,         # [Gs+1], compact selected rows by expert
    selected_experts,         # [Gs+1], original expert ids, sentinel -1
    P=32, Q=8, BM=64, BK=512, G_work=128,
    lora_scale,
    down_lora_dropout_p,
):
    BN  = 2 * P
    W   = P * Q
    Kdx = 2 * W

    validate:
      SM100 only
      BF16 CUDA selected tensors and LoRA tensors
      CPU pinned contiguous BF16 gate_up_weight_cpu
      selected_offsets/expert sentinel and monotonic row ranges
      BN, BM, BK, G_work supported

    allocate outputs:
      dx_acc              = zeros([M_selected, H], fp32)
      grad_gate_sel       = zeros([M_selected, I], bf16)
      grad_up_sel         = zeros([M_selected, I], bf16)
      grad_down_lora_A_sel = zeros([E, r_down, I], bf16/fp32)

    build global selected metadata once:
      row_expert[row]      -> original expert id
      row_group[row]       -> selected group id
      active_selected_experts = selected_experts[0:Gs]

    for g_start in 0..Gs step G_work:
      expert_chunk = active_selected_experts[g_start:g_start+G_work]
      G_chunk = len(expert_chunk)

      build chunk metadata:
        chunk_offsets_unpadded [G_chunk+1]
        chunk_row_ids          [M_work]      # local row -> compact selected row
        row_to_chunk_local     [M_selected]  # compact row -> local row for chunk
        row_to_e_rel           [M_work]      # local row -> e_rel in chunk
        padded grouped-GEMM metadata if the implementation uses padded loaders

      allocate/reuse chunk workspaces:
        w_cache          [G_chunk, Kdx, H] bf16
        grad_pair_window [M_work, Kdx] bf16

      # Main window loop. Each window streams selected W_gate_up_cpu exactly once.
      for iwin0 in 0..I step W:

        #######################################################################
        # A0. Fill W cache: CPU W -> HBM w_cache. The CPU weight stream stays
        # outside all selected-row loops.
        #######################################################################
        for e_rel, e in enumerate(expert_chunk):
          for q in 0..Q-1:
            i0 = iwin0 + q * P
            valid_i = clamp(I - i0, 0, P)
            n0 = q * BN

            for h0 in 0..H step BK:
              if valid_i > 0:
                Wg = stream_cpu(gate_up_weight_cpu[e, i0:i0+valid_i, h0:h0+BK])
                Wu = stream_cpu(gate_up_weight_cpu[e, I+i0:I+i0+valid_i, h0:h0+BK])
              else:
                Wg = empty
                Wu = empty

              # Paired layout for this q panel:
              #   w_cache[..., n0:n0+P,   :] = gate columns
              #   w_cache[..., n0+P:n0+BN,:] = up columns
              w_cache[e_rel, n0:n0+P,    h0:h0+BK] = zero
              w_cache[e_rel, n0+P:n0+BN, h0:h0+BK] = zero
              w_cache[e_rel, n0:n0+valid_i, h0:h0+BK] =
                  Wg[0:valid_i, :]
              w_cache[e_rel, n0+P:n0+P+valid_i, h0:h0+BK] =
                  Wu[0:valid_i, :]

        #######################################################################
        # A1. Recompute gate/up, include gate/up LoRA, compute act, selected
        # dA_down, activation backward, and grad_pair_window. Only
        # grad_pair_window and w_cache survive until dX.
        #######################################################################
        for e_rel, e in enumerate(expert_chunk):
          rows_e_global = compact selected row ids for expert e
          rows_e_local  = row_to_chunk_local[rows_e_global]

          for q in 0..Q-1:
            i0 = iwin0 + q * P
            valid_i = clamp(I - i0, 0, P)
            n0 = q * BN

            if valid_i == 0:
              zero grad_pair_window[rows_e_local, n0:n0+BN]
              continue

            for me0 in 0..len(rows_e_global) step BM:
              rows_b  = rows_e_global[me0:me0+BM]
              local_b = rows_e_local[me0:me0+BM]
              M_b = len(rows_b)

              # Recompute base gate/up from cached CPU W window.
              pair_acc = zeros([M_b, BN], fp32)
              for h0 in 0..H step BK:
                X_tile = x_sel[rows_b, h0:h0+BK]                    # [M_b, BK]
                W_pair = w_cache[e_rel, n0:n0+BN, h0:h0+BK]         # [BN, BK]
                pair_acc += X_tile @ W_pair^T                       # [M_b, BN]

              gate_base = pair_acc[:, 0:P]
              up_base   = pair_acc[:, P:BN]

              # Correct LoRA recompute: gate/up include LoRA-B deltas.
              gate_delta = zeros([M_b, P], fp32)
              up_delta   = zeros([M_b, P], fp32)
              gate_delta[:, 0:valid_i] =
                  gate_low_rank_sel[rows_b, :] @ gate_lora_B[e, i0:i0+valid_i, :].T
              up_delta[:, 0:valid_i] =
                  up_low_rank_sel[rows_b, :] @ up_lora_B[e, i0:i0+valid_i, :].T

              gate = gate_base + lora_scale * gate_delta
              up   = up_base   + lora_scale * up_delta
              act  = silu(gate) * up                                # [M_b, P]

              # Selected down-LoRA A gradient for this P panel.
              dS = dS_down_sel[rows_b, :]                             # [M_b, r_down]

              act_valid = act[:, 0:valid_i]
              mask_valid = down_mask_slice(
                  down_mask_packed_sel, rows_b, i0, valid_i
              )
              act_drop = dropout_replay_panel(
                  act_valid,
                  mask_valid,
                  down_lora_dropout_p
              )

              # Selected down-LoRA A gradient for this panel.
              # dS already includes lora_scale, so do not scale again here.
              grad_down_lora_A_sel[e, :, i0:i0+valid_i] += dS^T @ act_drop

              # Gate/up activation backward.
              dact = dact_sel[rows_b, i0:i0+valid_i]
              gate_v = gate[:, 0:valid_i]
              up_v   = up[:, 0:valid_i]
              dgate  = dact * up_v * silu_grad(gate_v)
              dup    = dact * silu(gate_v)

              # Store full paired panel for dX. Invalid columns are zero.
              grad_pair_window[local_b, n0:n0+BN] = zero
              grad_pair_window[local_b, n0:n0+valid_i] = cast_bf16(dgate)
              grad_pair_window[local_b, n0+P:n0+P+valid_i] = cast_bf16(dup)

              # Return dgate/dup so existing gate/up LoRA backward can stay
              # outside this native path.
              grad_gate_sel[rows_b, i0:i0+valid_i] = cast_bf16(dgate)
              grad_up_sel[rows_b, i0:i0+valid_i]   = cast_bf16(dup)

        #######################################################################
        # B. Selected base dX: grad_pair_window @ cached W.
        # This reads HBM w_cache and must never stream W_gate_up_cpu.
        #######################################################################
        for e_rel, e in enumerate(expert_chunk):
          rows_e_global = compact selected row ids for expert e
          rows_e_local  = row_to_chunk_local[rows_e_global]

          for me0 in 0..len(rows_e_global) step BM:
            rows_b  = rows_e_global[me0:me0+BM]
            local_b = rows_e_local[me0:me0+BM]
            M_b = len(rows_b)

            for h0 in 0..H step BK:
              dx_tile = zeros([M_b, BK], fp32)
              for q in 0..Q-1:
                n0 = q * BN
                G_pair = grad_pair_window[local_b, n0:n0+BN]          # [M_b, BN]
                W_pair = w_cache[e_rel, n0:n0+BN, h0:h0+BK]           # [BN, BK]
                dx_tile += G_pair @ W_pair                            # [M_b, BK]

              dx_acc[rows_b, h0:h0+BK] += dx_tile

        discard_or_reuse(w_cache)
        discard_or_reuse(grad_pair_window)

    grad_x_base_sel = cast_bf16(dx_acc)
    return (
      grad_x_base_sel,
      grad_gate_sel,
      grad_up_sel,
      grad_down_lora_A_sel,
      stats
    )
```

## Required Counters

```text
cpu_weight_bytes_staged
expected_cpu_weight_bytes_min
cpu_weight_stream_multiplier                  # <= 1.01 for cache_first_window
fill_w_cache_effective_GBps
pinned_h2d_baseline_GBps
same_shape_cache_first_window_recompute_reads_w_cache_bytes
measured_saved_recompute_hbm_read_bytes
hbm_w_cache_valid_write_bytes
hbm_w_cache_padding_zero_write_bytes
recompute_reads_w_cache_bytes
dx_reads_w_cache_bytes
w_cache_bytes_allocated_peak
grad_pair_window_bytes_allocated_peak
full_window_gate_up_act_bytes                  # must be 0 for Stage 7 pass
fused_recompute_down_lora_kernel
native_kernel_consumed_saved_gate_up_s
native_kernel_consumed_down_dropout_masks
native_kernel_consumed_dact
native_kernel_consumed_dS_down
dropout_backward_rng_advanced                  # must be false
old_selected_base_dx_rows                      # must be 0 after integration
new_selected_base_dx_rows                      # must equal selected rows
fill_w_cache_ms
captured_phase_launch_overhead_ms
recompute_act_ms
down_lora_activation_ms
recompute_down_lora_activation_ms
selected_down_lora_A_ms
activation_backward_ms
dx_window_ms
native_selected_region_ms
current_selected_region_ms
```

Stage 7 pass condition:

```text
native_selected_region_ms <= 0.90 * current_selected_region_ms
fresh NCU exists for fill_w_cache, recompute_act, down_lora_activation, and
dx_window
CPU stream multiplier, w_cache reads, tensor-core use, and local-memory spill
counters all pass
```

## Stage Order

```text
Stage 0: baseline counters only
Stage 1: selected metadata + CPU W -> HBM w_cache
Stage 2: recompute_act = gate/up + gate/up LoRA + act tile
Stage 3: down_lora_activation = selected dA_down + dgate/dup tile
Stage 4: selected dX from grad_pair_window + w_cache
Stage 5: V2 native E2E API correctness
Stage 6: Python integration for dropout 0.00 and 0.10
Stage 7: NSYS/NCU profile gate, write kernels_progress.md, stop
```

Optional Megakernel-style SMEM paging belongs after Stage 7:

```text
seed_group_direct:
  while filling w_cache, keep the just-streamed W tile in an SMEM page and
  compute one seed M_group before releasing the page
  remaining rows still read w_cache
  accept only if cpu_weight_stream_multiplier <= 1.01 and NCU shows lower
  recompute-side w_cache traffic
```
