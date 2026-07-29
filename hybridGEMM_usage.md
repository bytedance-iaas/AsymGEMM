# hybridGEMM usage guide

How to use the SM90 INT8 grouped-GEMM family for mixed expert residency:
one fused kernel launch computes **HBM-resident** experts with a
deepGEMM-style pipeline and **pinned-host-resident** experts with the
asymGEMM pipeline, concurrently, on disjoint SM ranges. This document is the
practical companion to `hybridGEMM.md` (the design/measurement log): it
covers the API contracts, data layout rules, split selection, and the
pitfalls a new user will hit first.

All performance numbers quoted here were measured on an H200 (132 SMs) with
PCIe Gen5 pinned host memory; see `hybridGEMM.md` §10 for the full record.

---

## 1. Which kernel do I want?

Three sibling entry points share one layout convention (contiguous grouped
GEMM: `a[M,K] @ b[G,N,K].mT -> d[M,N]`, INT8 in / FP32 out):

| Entry point | Loop order | Weights live in | Use when |
|---|---|---|---|
| `m_grouped_int8_asym_gemm_nt_contiguous` | K-outer, single-slot B | pinned host (PCIe/UVA) *or* HBM | all experts host-resident; the PCIe-shaped baseline |
| `m_grouped_int8_gemm_nt_contiguous` ("deep") | M-outer, deep A+B pipeline | HBM only (device) | all experts HBM-resident — ~2.2× decode / ~1.9× prefill over the asym kernel on HBM |
| `m_grouped_int8_hybrid_gemm_nt_contiguous` | both, split by CTA rank | both at once | **mixed residency** — one launch instead of two, 1.0–1.3× over two launches when the split is sized right |

Rules of thumb:

- **Pure residency (all-host or all-HBM): call the standalone kernel.**
  The hybrid is parity at best there (measured 0.98–1.00×) and needs dummy
  tensors for the empty side.
- **Mixed residency: call the hybrid** with `s_host` from §6.
- Never run the deep kernel against pinned-host B. It re-reads B per
  m-block, which is ruinous over PCIe (the facade accepts pinned B for
  testing only).

The unified MoE runtime (`asym_gemm/unified_moe/runtime.py`) already routes
its **cached partition** through the deep kernel via
`ASYMGEMM_HBM_KERNEL=deep|asym` (default `deep`); the hybrid kernel is not
yet wired into `Layer.forward` (that is plan Phase D) — today you call it
directly.

## 2. Requirements and build

- SM90 GPU (H100/H200/H800; the kernels compile for `sm_90a`), CUDA ≥ 12.1.
- Build the extension once after any `csrc/` change:

```bash
python setup.py build_ext --inplace     # editable install
```

- The CUDA kernels themselves are JIT-compiled on first call (cached in
  `~/.asym_gemm/cache/`). The first call per (n, k) shape takes extra
  seconds; warm up before timing.
- Availability check:

```python
import asym_gemm
assert hasattr(asym_gemm, "m_grouped_int8_hybrid_gemm_nt_contiguous")
```

## 3. Quick start: mixed residency in ~40 lines

```python
import torch, asym_gemm
from asym_gemm.utils.math import ceil_div, per_token_cast_to_int8, per_channel_cast_to_int8

G_HOST, G_HBM, MPG = 2, 6, 128          # experts per side, rows per expert
N, K = 2048, 1024                        # both must be multiples of 128
G = G_HOST + G_HBM
M = G * MPG                              # MPG must be a multiple of 64

# --- quantize activations (per-token) and weights (per-channel, 128-granular K blocks)
a = torch.randn(M, K, device="cuda")
a_q, sfa = per_token_cast_to_int8(a)     # a_q [M,K] int8, sfa [M, ceil(K/128)] fp32

b = torch.randn(G, N, K, device="cuda")
b_q = torch.empty_like(b, dtype=torch.int8)
sfb = torch.empty(G, N, ceil_div(K, 128), device="cuda")
for g in range(G):
    b_q[g], sfb[g] = per_channel_cast_to_int8(b[g])

b_host = b_q[:G_HOST].cpu().pin_memory()          # host side: PINNED int8 [Gh,N,K]
sfb_host = sfb[:G_HOST].contiguous()              # its scales stay on the GPU
b_hbm, sfb_hbm = b_q[G_HOST:].contiguous(), sfb[G_HOST:].contiguous()  # device

# --- segment lists: expert g owns rows [g*MPG, (g+1)*MPG); ids are LOCAL per side
def layout(expert_range, id_base):
    off, ids = [], []
    for g in expert_range:
        off += [g * MPG, (g + 1) * MPG]
        ids.append(g - id_base)
    return (torch.tensor(off, dtype=torch.int32, device="cuda"),
            torch.tensor(ids + [-1], dtype=torch.int32, device="cuda"),
            len(ids) + 1)

off_h, exp_h, ls_h = layout(range(G_HOST), 0)          # host segments
off_d, exp_d, ls_d = layout(range(G_HOST, G), G_HOST)  # HBM segments (re-based ids)

d = torch.empty(M, N, device="cuda", dtype=torch.float32)
asym_gemm.m_grouped_int8_hybrid_gemm_nt_contiguous(
    (a_q, sfa), (b_host, sfb_host), (b_hbm, sfb_hbm), d,
    off_h, exp_h, ls_h, off_d, exp_d, ls_d,
    s_host=32, enable_steal=True)
```

A runnable, asserting version of exactly this pattern is
`tests/test_sm90_int8_hybrid.py` (`_build_case` + the parity tests).

## 4. API reference

### 4.1 `m_grouped_int8_hybrid_gemm_nt_contiguous`

```python
asym_gemm.m_grouped_int8_hybrid_gemm_nt_contiguous(
    a,            # (a_int8, sfa)          activations, shared by both sides
    b_host,       # (b_int8, sfb)          host-side expert weights
    b_hbm,        # (b_int8, sfb)          HBM-side expert weights
    d,            # [M, N] fp32 cuda       output, shared by both sides
    offsets_host, experts_host, list_size_host,
    offsets_hbm,  experts_hbm,  list_size_hbm,
    s_host,                    # int: CTAs [0, s_host) run the host side
    enable_steal=False,        # bool: drained host CTAs join the HBM side
    recipe=None, compiled_dims="nk")   # accepted for API parity; ignored
```

Tensor contracts (all asserted host-side):

| Tensor | Shape / dtype | Placement | Notes |
|---|---|---|---|
| `a_int8` | `[M, K]` int8, contiguous | cuda | K-major (row-major) |
| `sfa` | `[M, ceil(K/128)]` fp32 | cuda | per-token scales, one per 128-wide K block |
| `b_host[0]` | `[Gh, N, K]` int8, contiguous | **pinned host** (or cuda) | TMA reads it over UVA/PCIe |
| `b_host[1]` | `[Gh, N, ceil(K/128)]` fp32 | cuda | per-channel scales — always device-resident |
| `b_hbm[0]` | `[Gd, N, K]` int8, contiguous | **cuda only** | |
| `b_hbm[1]` | `[Gd, N, ceil(K/128)]` fp32 | cuda | |
| `d` | `[M, N]` fp32, contiguous | cuda | both sides write disjoint rows |
| `offsets_*` | `[2*S]` int32 | cuda | (start, end) row pairs per segment |
| `experts_*` | `[S+1]` int32 | cuda | **side-local** B index per segment, then `-1` terminator |

`list_size_* = S + 1` (segment count including the terminator slot). Both B
tensors must share the same `N` and `K`; `N % 128 == 0` and `K % 128 == 0`
are required.

`s_host` is clamped by the launcher: an empty side gets 0 CTAs, and when
both sides have work each gets at least one CTA. See §6 for choosing it.

### 4.2 The standalone kernels

```python
asym_gemm.m_grouped_int8_gemm_nt_contiguous(       # deep (HBM weights)
    (a_q, sfa), (b_q, sfb), d, offsets, experts, list_size)
asym_gemm.m_grouped_int8_asym_gemm_nt_contiguous(  # asym (pinned or HBM weights)
    (a_q, sfa), (b_q, sfb), d, offsets, experts, list_size)
```

Same contracts as one side of the hybrid, with **global** expert ids
indexing the single B tensor. Outputs are numerically interchangeable with
the hybrid (the degenerate-split parity tests pin this to diff = 0).

## 5. Layout rules (the part that bites)

1. **Segment starts must be multiples of 64** (the kernels' BLOCK_M launch
   granularity). Ends may be true row counts; rows between `end` and the
   next 64-aligned boundary are treated as padding.
2. **Zero-fill padding rows** of `a_int8` and `sfa` (a zeroed activation
   row is harmless; garbage is not). `torch.zeros` + `index_copy_` of real
   rows, as the runtime does, is the safe pattern.
3. **Host and HBM segments must be disjoint row ranges** of `a`/`d`. Each
   expert's rows appear in exactly one side's list. Interleaving segments
   from the two sides in M is fine (the tests do it on purpose).
4. **Expert ids are side-local**: `experts_hbm[i]` indexes `b_hbm`, not the
   global expert numbering. Subtract the base when you split a global
   weight tensor (see quick start).
5. **An empty side** still needs well-formed arguments: any 1-group dummy B
   pair plus `offsets=torch.zeros(2, int32)`, `experts=[-1]`,
   `list_size=1`. The dummy B is never read. (Or just call the standalone
   kernel — preferred, §1.)
6. **Scales are 128-granular along K and constant within a block.** Use
   `per_token_cast_to_int8` (activations) / `per_channel_cast_to_int8`
   (weights); anything else must reproduce their `[.., ceil(K/128)]`
   layouts. The facades transpose scales to the kernel's K-major layout
   internally — you never pre-transpose.
7. `recipe` / `compiled_dims` are accepted for API parity with the FP8/BF16
   entry points and ignored: INT8 is fixed at (1, 1, 128) granularity and
   "nk" compiled dims.

## 6. Choosing `s_host` (the two-regime rule)

`s_host` is the number of CTAs (of `num_sms`, 132 on H200) given to the
host side. The measured rule (hybridGEMM.md §10, ratio-sweep study):

```python
def pick_s_host(rows_per_expert, n_host, n_hbm, num_sms=132,
                asym_tflops=182e12, deep_tflops=337e12, link_bw=24e9):
    """Two-regime C1 prior, validated on H200 + PCIe Gen5."""
    rows_link_bound = asym_tflops / (2 * link_bw)     # ~3.7k rows/expert on Gen5
    if rows_per_expert <= rows_link_bound:
        return 32                                     # host side is LINK-bound
    w = 1.85 * n_host                                 # asym pipeline is ~1.85x
    return max(16, min(num_sms - 16,                  #   less TFLOPS-efficient
                       round(num_sms * w / (w + n_hbm))))
```

Why two regimes:

- **Link-bound** (small decode batches): a handful of CTAs saturates PCIe;
  more only steal HBM throughput. But beware: the saturating count grows
  with rows/expert, because each asym CTA's PCIe duty-cycle falls as it
  computes more m-blocks between B fetches — 16 CTAs that saturate the link
  at decode starve it at prefill. 32 is the safe link-bound choice.
- **Compute-bound** (large prefill batches, above ~3.7k rows/expert on
  Gen5): size by flops share, weighted by the asym pipeline's ~1.85× lower
  efficiency.

Getting this wrong is the main way to lose: a link-sized split at a
compute-bound load measured 0.4–0.9× vs two launches; the rule above
restores ≥ 0.98× at every measured point. When in doubt, enable stealing
(§7) and err toward a **larger** `s_host` — over-allocated host CTAs steal
HBM work back; an under-fed link cannot be helped from the other side.

## 7. Stealing (`enable_steal=True`)

With stealing on, the HBM side's tile enumeration becomes a device-global
atomic ticket counter, and host-side CTAs that finish their segment list
quiesce and join it. Properties:

- **One-directional** (host→HBM only): the reverse would just queue more
  consumers on an already-saturated PCIe link.
- **Cost when nothing needs stealing: below noise** (measured 0.171 vs
  0.172 ms on a well-balanced split). The launcher auto-disables it when
  either side is empty.
- **What it buys**: recovery from a mispredicted `s_host` — a 3/4-wrong
  split at prefill was pulled from 0.350 ms back to 0.172 ms, matching the
  hand-tuned split. It cannot fix an *under-fed host side* (see §6).

Recommended: always pass `enable_steal=True` for mixed loads. The launcher
allocates and zero-fills the ticket counter per call; nothing else to manage.

## 8. Runtime integration (unified_moe)

What is wired today:

```bash
ASYMGEMM_HBM_KERNEL=deep    # default: cached (HBM) partition uses the deep kernel
ASYMGEMM_HBM_KERNEL=asym    # fallback: previous behavior
```

This affects `Layer.forward`'s cached-partition GEMMs only; streamed
(pinned-host) partitions always use the asym kernel, and copy-engine
staging (~48 GB/s) remains the default transport for bulk host experts on
PCIe — in-kernel PCIe reads (~24 GB/s) are the staging-off / ring-overflow
path. The single-launch hybrid replaces the cached/staged partition loop in
plan Phase D (`ASYMGEMM_HYBRID_KERNEL=1`, not yet implemented).

## 9. What performance to expect (H200, measured)

Fused gate+up projection, mean of 3 runs, vs running deep + asym as two
launches, with `s_host` from §6 and stealing on:

| Scenario | Expectation |
|---|---|
| all experts HBM (r=1) | parity with the deep kernel — use the deep kernel directly |
| all experts host (r=0) | parity with the asym kernel — use the asym kernel directly |
| mixed, decode (256 rows/expert) | 1.00–1.24×, growing with the HBM share |
| mixed, prefill (10k rows/expert) | 1.0–1.30×, peaking near r ≈ 0.75 |
| mispredicted split, stealing on | penalty recovered to ≈ the balanced split |

Context that dominates all of the above: moving an expert from host to HBM
is worth ~27–32× at decode (weight-streaming-bound) and ~4× at large
prefill. Residency policy first, kernel choice second. On NVLink-C2C parts
(GH200/GB200, ~20× the link bandwidth) the link-bound regime extends ~20×
and the hybrid becomes the primary path; on PCIe it complements staging.

## 10. Verifying an install / benchmarking

```bash
python tests/test_sm90_int8_deep.py     # deep kernel: parity + HBM bench
python tests/test_sm90_int8_hybrid.py   # hybrid: mixed/degenerate/steal parity + benches
python tests/test_sm90_int8.py          # asym baseline suite
python scratchpad/hybrid_ratio_sweep.py # residency-ratio study (writes JSON)
```

All parity tests should print `diff=0.00000`-class numbers; the hybrid
suite also cross-checks degenerate splits bitwise against the standalone
kernels.

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `DG_HOST_ASSERT` on N or K | `N % 128 != 0` or `K % 128 != 0` — pad the projection or pick block-friendly shapes |
| wrong results, some rows stale | segment start not 64-aligned, overlapping host/HBM row ranges, or padding rows not zero-filled (§5) |
| `b.is_cuda() or b.is_pinned()` assert | host-side B must be `pin_memory()` (pageable host tensors are not TMA-reachable); `b_hbm` must be on-device |
| edited a `.cuh`, behavior unchanged | the JIT cache keys on the instantiation string, not file contents: `rm -rf ~/.asym_gemm/cache/*<kernel_name>*` and rerun |
| first call is slow | JIT compile; warm up per (n, k) shape before timing |
| hybrid slower than two launches | `s_host` mis-sized for the regime — apply §6; confirm stealing is on; at pure residency use the standalone kernel |
| tiny diffs vs a fp64 reference on the host side | expected: the asym side chains fp32 `TMA_REDUCE_ADD` partial sums (order-dependent); the deep side accumulates in registers and is exact vs the int32 reference |
| benchmarking noise / contradictory ratios | interleave the variants and average (this project's sweeps use 3 alternating reps) — one-shot orderings produced up-to-2× phantom effects |

## 12. File map

| Path | What |
|---|---|
| `hybridGEMM.md` | design, correctness analysis, full measurement log |
| `asym_gemm/include/asym_gemm/impls/sm90_int8_hybrid_gemm.cuh` | the fused kernel (both side pipelines + stealing) |
| `asym_gemm/include/asym_gemm/impls/sm90_int8_gemm.cuh` | deep (HBM) kernel |
| `asym_gemm/include/asym_gemm/impls/sm90_int8_asym_gemm_1d1d.cuh` | asym (PCIe) kernel |
| `csrc/jit_kernels/impls/sm90_int8_hybrid_gemm.hpp` | hybrid JIT launcher (smem plan, s_host clamping, steal counter) |
| `csrc/apis/gemm.hpp` | Python-facing facades (`m_grouped_int8_*`) |
| `tests/test_sm90_int8_hybrid.py` | parity + steal + launch-collapse benches; the reference for building layouts |
| `scratchpad/hybrid_ratio_sweep.py`, `hybrid_shost_check*.py` | residency-ratio study and s_host sensitivity harness |
| `asym_gemm/unified_moe/runtime.py` | serving runtime; `ASYMGEMM_HBM_KERNEL` switch lives here |
