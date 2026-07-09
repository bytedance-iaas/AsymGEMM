"""Adaptive CPU/GPU dispatch model for the unified INT8 MoE layer.

Replaces the static ``m_e <= m_cpu`` threshold with a runtime cost model.
Both buckets run concurrently (the GPU bucket is stream-async, the CPU
bucket runs on the host worker pool), so the objective is *makespan*:

    minimize  max(T_cpu(bucket), T_gpu(bucket))

not per-expert argmin — offloading an expert to an idle backend is free
even when that backend is slower for the expert in isolation.

Each backend gets a linear wall-time model over one forward's bucket:

    T_cpu ≈ a0 + a1 · E_cpu + a2 · Σ m_e            (E experts, m rows each)
    T_gpu ≈ b0 + b1 · E_gpu + b2 · Σ pad(m_e)       (pad = round up to BLOCK_M)

The structure mirrors the hardware: with pinned-host weights the GPU pays
a per-expert PCIe weight fetch (b1, nearly independent of m_e) plus padded
WGMMA rows (b2); the CPU pays a per-expert DRAM weight-read floor (a1)
plus per-row AMX work (a2). Coefficients are seeded with rough priors and
refitted online by ridge-regularized least squares over a ring buffer of
observed (E, Σm, seconds) triples, so the model tracks live conditions
(thread contention, PCIe traffic, clocks). The windowed refit doubles as
hysteresis — one outlier observation cannot flap the split.

Timing sources are supplied by the caller (`Layer.forward`): host
``perf_counter`` for the CPU bucket, CUDA event pairs for the GPU bucket.
Event pairs are *harvested lazily* on a later call so the hot path never
synchronizes the stream.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# Priors put the single-expert CPU/GPU crossover near m_e ≈ 16–32 (the
# static default this model replaces). They only steer the first few
# forwards; fitted observations dominate quickly. Used when the model is
# constructed without a layer shape; with (hidden, inter) given, priors are
# derived from _RATES_PRIOR instead.
_CPU_PRIOR = np.array([100e-6, 30e-6, 8e-6])    # a0, a1, a2  (seconds)
_GPU_PRIOR = np.array([150e-6, 120e-6, 0.05e-6])  # b0, b1, b2

# Nominal hardware rates behind the shape-aware priors. The per-expert cost
# is weight traffic (3 int8 projections = 3·H·I bytes) over a bandwidth; the
# per-row cost is GEMM work (6·H·I MAC-flops across the 3 projections) over
# a throughput. Values are chosen to reproduce the legacy hand-tuned priors
# above at the reference shape H=1024, I=2048 — behavior is unchanged there,
# but the derived priors stay physically meaningful at any layer shape.
_RATES_PRIOR = {
    "cpu_launch_s":   100e-6,     # job launch, gather, numpy plumbing
    "cpu_weight_bps": 2.097e11,   # DRAM streaming read of expert weights
    "cpu_flops":      1.573e12,   # effective AMX INT8 at small m
    "gpu_launch_s":   150e-6,     # quant prekernel + launches + layout build
    "gpu_weight_bps": 5.243e10,   # effective PCIe fetch of pinned weights
    "gpu_flops":      2.517e14,   # effective WGMMA INT8 (incl. padding)
}


def _coefs_from_rates(rates: dict, bytes_per_expert: float,
                      flops_per_row: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert hardware rates → (cpu, gpu) coefficient vectors at a shape."""
    B, F = bytes_per_expert, flops_per_row
    cpu = np.array([rates["cpu_launch_s"],
                    B / rates["cpu_weight_bps"], F / rates["cpu_flops"]])
    gpu = np.array([rates["gpu_launch_s"],
                    B / rates["gpu_weight_bps"], F / rates["gpu_flops"]])
    return cpu, gpu

_RING_SIZE = 64
_MIN_OBS_FOR_FIT = 4
_RIDGE_LAMBDA = 1e-3


@dataclass
class _BackendModel:
    prior: np.ndarray
    coef: np.ndarray = field(init=False)
    obs: deque = field(init=False)          # (n_experts, total_m, seconds)
    _dirty: bool = field(default=False, init=False)

    def __post_init__(self):
        self.coef = self.prior.copy()
        self.obs = deque(maxlen=_RING_SIZE)

    def record(self, n_experts: int, total_m: int, seconds: float) -> None:
        if n_experts <= 0 or seconds <= 0.0:
            return
        self.obs.append((float(n_experts), float(total_m), float(seconds)))
        self._dirty = True

    def _refit(self) -> None:
        if len(self.obs) < _MIN_OBS_FOR_FIT:
            return
        arr = np.asarray(self.obs, dtype=np.float64)
        X = np.column_stack([np.ones(len(arr)), arr[:, 0], arr[:, 1]])
        t = arr[:, 2]
        # Ridge toward the prior on column-normalized features: keeps the
        # system well-posed when observations are collinear (e.g. every
        # forward so far had the same expert count).
        s = np.sqrt((X * X).mean(axis=0)).clip(min=1e-12)
        Xn = X / s
        A = Xn.T @ Xn + _RIDGE_LAMBDA * np.eye(3)
        b = Xn.T @ t + _RIDGE_LAMBDA * (self.prior * s)
        self.coef = (np.linalg.solve(A, b) / s).clip(min=0.0)

    def refresh(self) -> None:
        """Fold pending observations into the coefficients."""
        if self._dirty:
            self._refit()
            self._dirty = False

    def predict(self, n_experts: float, total_m: float) -> float:
        if n_experts <= 0:
            return 0.0
        self.refresh()
        c = self.coef
        return float(c[0] + c[1] * n_experts + c[2] * total_m)


class DispatchModel:
    """Owns both backend models, the partition solver, and pending GPU events.

    With ``hidden`` and ``inter`` given, priors are derived analytically from
    nominal hardware rates (weight bytes / bandwidth, flops / throughput), so
    cold-start decisions are sensible at any layer shape. Coefficients stay
    in per-expert / per-row units, so one instance can be shared by every
    same-shape MoE layer in a model (pass it via ``Layer(dispatch_model=…)``)
    — pooling observations across layers. For cross-shape transfer, use
    :meth:`rates` / :meth:`from_rates`.
    """

    def __init__(self, block_m: int = 256,
                 hidden: Optional[int] = None, inter: Optional[int] = None):
        self.block_m = block_m
        self.hidden, self.inter = hidden, inter
        if hidden is not None and inter is not None:
            self.bytes_per_expert = 3.0 * hidden * inter   # int8, 3 projections
            self.flops_per_row = 6.0 * hidden * inter      # MACs, 3 GEMMs
            cpu_prior, gpu_prior = _coefs_from_rates(
                _RATES_PRIOR, self.bytes_per_expert, self.flops_per_row)
        else:
            self.bytes_per_expert = None
            self.flops_per_row = None
            cpu_prior, gpu_prior = _CPU_PRIOR.copy(), _GPU_PRIOR.copy()
        self.cpu = _BackendModel(cpu_prior)
        self.gpu = _BackendModel(gpu_prior)
        # (ev_start, ev_end, n_experts, total_padded_m)
        self._pending_gpu: list[tuple] = []
        # When True, record_cpu/record_gpu_events are no-ops. Used by
        # Layer.calibrate to run one unrecorded warm-up forward per shape:
        # first-touch effects (kernel JIT for a new shape, worker-pool and
        # allocator warm-up) inflate that observation by 5-10x, and with
        # rings this small a single outlier wrecks the ridge fit.
        self.paused = False

    # -- cross-shape transfer ---------------------------------------------

    def rates(self) -> dict:
        """Implied hardware rates from the current coefficients (shape-aware
        models only). A zero coefficient maps to an infinite rate."""
        assert self.hidden is not None and self.inter is not None, \
            "rates() requires a shape-aware model (hidden/inter set)"
        B, F = self.bytes_per_expert, self.flops_per_row
        c, g = self.cpu.coef, self.gpu.coef
        return {
            "cpu_launch_s":   float(c[0]),
            "cpu_weight_bps": B / c[1] if c[1] > 0 else float("inf"),
            "cpu_flops":      F / c[2] if c[2] > 0 else float("inf"),
            "gpu_launch_s":   float(g[0]),
            "gpu_weight_bps": B / g[1] if g[1] > 0 else float("inf"),
            "gpu_flops":      F / g[2] if g[2] > 0 else float("inf"),
        }

    @classmethod
    def from_rates(cls, rates: dict, *, hidden: int, inter: int,
                   block_m: int = 256) -> "DispatchModel":
        """New shape-aware model whose priors *and* coefficients realize the
        given hardware rates at (hidden, inter) — warm-starts a layer of a
        different shape from an already-fitted model (or from disk)."""
        m = cls(block_m=block_m, hidden=hidden, inter=inter)
        cpu, gpu = _coefs_from_rates(rates, m.bytes_per_expert,
                                     m.flops_per_row)
        m.cpu.prior, m.cpu.coef = cpu.copy(), cpu.copy()
        m.gpu.prior, m.gpu.coef = gpu.copy(), gpu.copy()
        return m

    # -- observation intake ---------------------------------------------

    def record_cpu(self, n_experts: int, total_m: int, seconds: float) -> None:
        if self.paused:
            return
        self.cpu.record(n_experts, total_m, seconds)

    def record_gpu_events(self, ev_start, ev_end,
                          n_experts: int, total_padded_m: int) -> None:
        """Register a CUDA event pair bracketing one GPU bucket. Resolved
        later by :meth:`harvest` — never blocks the hot path."""
        if self.paused:
            return
        self._pending_gpu.append((ev_start, ev_end, n_experts, total_padded_m))

    def harvest(self, wait: bool = False) -> int:
        """Resolve completed pending GPU event pairs into observations.

        Returns the number harvested. With ``wait=True`` synchronizes on
        every pending pair (used by calibration, never the hot path).
        """
        remaining, n = [], 0
        for ev_start, ev_end, n_experts, total_m in self._pending_gpu:
            if wait:
                ev_end.synchronize()
            elif not ev_end.query():
                remaining.append((ev_start, ev_end, n_experts, total_m))
                continue
            self.gpu.record(n_experts, total_m,
                            ev_start.elapsed_time(ev_end) * 1e-3)
            n += 1
        self._pending_gpu = remaining
        return n

    # -- partition --------------------------------------------------------

    def padded(self, m: int) -> int:
        return ((m + self.block_m - 1) // self.block_m) * self.block_m

    def partition(
        self,
        m_list: Sequence[int],
        gpu_available: bool = True,
    ) -> tuple[list[int], list[int]]:
        """Whole-expert split of ``m_list`` across backends.

        Returns (cpu_indices, gpu_indices) — indices into ``m_list``.
        Prefix-scan family only; see :meth:`partition_rows` for the version
        that may additionally row-split the hottest expert.
        """
        cpu_idx, gpu_idx, _ = self._scan(m_list, gpu_available,
                                         allow_row_split=False)
        return cpu_idx, gpu_idx

    def partition_rows(
        self,
        m_list: Sequence[int],
        gpu_available: bool = True,
    ) -> tuple[list[int], list[int], Optional[tuple[int, int]]]:
        """Like :meth:`partition`, but the hottest expert's *rows* may be
        split across both backends.

        Returns (cpu_indices, gpu_indices, row_split) where row_split is
        ``None`` or ``(index, rows_to_cpu)`` with ``index ∈ gpu_indices``:
        the first ``rows_to_cpu`` routed rows of that expert run on CPU, the
        remainder on GPU. Both backends read the same pinned weight bytes,
        so the only duplicated cost is one extra per-expert weight fetch —
        which the scan prices in (the CPU side is charged a full extra
        expert). Donated rows are quantized in whole ``block_m`` chunks of
        GPU savings, so the GPU keeps exactly block-aligned rows (zero
        padding waste on the split expert). This removes the makespan floor
        a single hot expert otherwise imposes under skewed routing.
        """
        return self._scan(m_list, gpu_available, allow_row_split=True)

    def _scan(
        self,
        m_list: Sequence[int],
        gpu_available: bool,
        allow_row_split: bool,
    ) -> tuple[list[int], list[int], Optional[tuple[int, int]]]:
        """Minimum-predicted-makespan scan.

        Experts are sorted by m_e ascending and every prefix→CPU /
        suffix→GPU split is scored with prefix sums. Small-m experts go CPU
        first because CPU cost grows with m faster than GPU cost (a2 ≫ b2)
        while GPU carries the larger per-expert constant (b1 > a1). With
        ``allow_row_split``, each prefix split is additionally scored with
        c = 1..nblk BLOCK_M-sized chunks of the largest expert (always in
        the GPU suffix) donated to the CPU bucket; c = nblk is a full move,
        which also covers the non-prefix "largest expert alone on CPU"
        assignments.
        """
        n = len(m_list)
        if n == 0:
            return [], [], None
        if not gpu_available:
            return list(range(n)), [], None

        order = sorted(range(n), key=lambda i: m_list[i])
        m_sorted = [m_list[i] for i in order]

        prefix_m = np.concatenate([[0], np.cumsum(m_sorted)])
        padded = np.array([self.padded(m) for m in m_sorted])
        suffix_pad = np.concatenate([np.cumsum(padded[::-1])[::-1], [0]])

        blk = self.block_m
        m_big, pad_big = m_sorted[-1], int(padded[-1])
        nblk = pad_big // blk

        # Mixed splits pay a serial penalty: the GPU bucket's launch work
        # (b0 — quant prekernel, layout build, kernel enqueue) runs on the
        # host thread *before* the CPU bucket starts, so it delays the CPU
        # side instead of overlapping it. Pure max() would over-favor mixed
        # splits at small batches, where b0 dominates both buckets.
        self.cpu.refresh()
        self.gpu.refresh()
        serial_gpu = float(self.gpu.coef[0])

        def mixed_cost(t_cpu: float, t_gpu: float,
                       cpu_n: int, gpu_n: int) -> float:
            if cpu_n > 0 and gpu_n > 0:
                return max(t_cpu + serial_gpu, t_gpu)
            return max(t_cpu, t_gpu)

        best_k, best_c, best_cost = 0, 0, float("inf")
        for k in range(n + 1):                     # first k experts → CPU
            t_cpu = self.cpu.predict(k, prefix_m[k])
            t_gpu = self.gpu.predict(n - k, suffix_pad[k])
            cost = mixed_cost(t_cpu, t_gpu, k, n - k)
            if cost < best_cost:
                best_k, best_c, best_cost = k, 0, cost
            if not allow_row_split or k >= n:
                continue
            for c in range(1, nblk + 1):           # donate c blocks of L
                if c < nblk:                       # partial: L stays on GPU
                    r = m_big - (pad_big - c * blk)
                    t_cpu = self.cpu.predict(k + 1, prefix_m[k] + r)
                    t_gpu = self.gpu.predict(n - k, suffix_pad[k] - c * blk)
                    cost = mixed_cost(t_cpu, t_gpu, k + 1, n - k)
                else:                              # full move of L to CPU
                    t_cpu = self.cpu.predict(k + 1, prefix_m[k] + m_big)
                    t_gpu = self.gpu.predict(n - k - 1,
                                             suffix_pad[k] - pad_big)
                    cost = mixed_cost(t_cpu, t_gpu, k + 1, n - k - 1)
                if cost < best_cost:
                    best_k, best_c, best_cost = k, c, cost

        cpu_idx = [order[i] for i in range(best_k)]
        gpu_idx = [order[i] for i in range(best_k, n)]
        row_split = None
        if best_c > 0:
            big = order[n - 1]
            if best_c == nblk:                     # whole-expert move
                gpu_idx.remove(big)
                cpu_idx.append(big)
            else:
                row_split = (big, m_big - (pad_big - best_c * blk))
        return sorted(cpu_idx), sorted(gpu_idx), row_split

    # -- introspection ----------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "cpu_coef": self.cpu.coef.tolist(),
            "gpu_coef": self.gpu.coef.tolist(),
            "cpu_obs": len(self.cpu.obs),
            "gpu_obs": len(self.gpu.obs),
            "pending_gpu": len(self._pending_gpu),
            "hidden": self.hidden,
            "inter": self.inter,
        }
