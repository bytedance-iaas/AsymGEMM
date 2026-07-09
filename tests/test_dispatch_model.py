"""Unit tests for asym_gemm.unified_moe.dispatch_model — no AMX/CUDA needed.

Runs both ways (AsymGEMM convention):
    python tests/test_dispatch_model.py
    pytest tests/test_dispatch_model.py -s
"""
from __future__ import annotations

import itertools
import sys

import numpy as np

try:
    from asym_gemm.unified_moe.dispatch_model import DispatchModel, _BackendModel
except ImportError as e:  # asym_gemm extensions not built on this host
    print(f"[SKIP] asym_gemm.unified_moe not importable: {e}")
    sys.exit(0)


def _seed_model(model: DispatchModel,
                cpu_coef=(100e-6, 30e-6, 8e-6),
                gpu_coef=(150e-6, 120e-6, 0.05e-6)) -> None:
    model.cpu.coef = np.array(cpu_coef)
    model.gpu.coef = np.array(gpu_coef)


def _makespan(model: DispatchModel, m_list, cpu_idx, gpu_idx) -> float:
    """The solver's objective: predicted makespan, with the GPU launch (b0)
    charged serially in front of the CPU bucket on mixed splits (the enqueue
    runs on the host thread before the CPU bucket starts)."""
    t_cpu = model.cpu.predict(len(cpu_idx), sum(m_list[i] for i in cpu_idx))
    t_gpu = model.gpu.predict(len(gpu_idx),
                              sum(model.padded(m_list[i]) for i in gpu_idx))
    if cpu_idx and gpu_idx:
        return max(t_cpu + float(model.gpu.coef[0]), t_gpu)
    return max(t_cpu, t_gpu)


def test_empty_and_forced_cases():
    model = DispatchModel()
    assert model.partition([]) == ([], [])
    # No GPU → everything CPU.
    cpu, gpu = model.partition([1, 5, 300], gpu_available=False)
    assert cpu == [0, 1, 2] and gpu == []
    print("  [test_empty_and_forced_cases]  ok")


def test_small_m_prefers_cpu_large_m_prefers_gpu():
    """With the default priors, a lone tiny expert goes CPU and a lone huge
    expert goes GPU — the model reproduces the sane threshold behavior."""
    model = DispatchModel()
    cpu, gpu = model.partition([1])
    assert cpu == [0] and gpu == []
    cpu, gpu = model.partition([4096])
    assert cpu == [] and gpu == [0]
    print("  [test_small_m_prefers_cpu_large_m_prefers_gpu]  ok")


def test_partition_beats_all_single_backend():
    """Predicted makespan of the chosen split is never worse than all-CPU
    or all-GPU (both are members of the scanned prefix-split family)."""
    rng = np.random.default_rng(0)
    model = DispatchModel()
    _seed_model(model)
    for _ in range(50):
        n = int(rng.integers(1, 12))
        m_list = [int(rng.integers(1, 1024)) for _ in range(n)]
        cpu, gpu = model.partition(m_list)
        assert sorted(cpu + gpu) == list(range(n))
        chosen = _makespan(model, m_list, cpu, gpu)
        all_cpu = _makespan(model, m_list, list(range(n)), [])
        all_gpu = _makespan(model, m_list, [], list(range(n)))
        assert chosen <= all_cpu + 1e-12
        assert chosen <= all_gpu + 1e-12
    print("  [test_partition_beats_all_single_backend]  ok")


def test_partition_matches_exhaustive():
    """For realistic coefficient regimes (CPU per-row cost ≫ GPU per-row
    cost, GPU per-expert cost > CPU per-expert cost), the sorted-prefix
    scan finds the same makespan as exhaustive subset search."""
    rng = np.random.default_rng(1)
    model = DispatchModel()
    for trial in range(30):
        _seed_model(
            model,
            cpu_coef=(rng.uniform(20e-6, 300e-6),
                      rng.uniform(10e-6, 80e-6),
                      rng.uniform(2e-6, 20e-6)),
            gpu_coef=(rng.uniform(50e-6, 400e-6),
                      rng.uniform(60e-6, 300e-6),
                      rng.uniform(0.01e-6, 0.2e-6)),
        )
        n = int(rng.integers(1, 9))
        m_list = [int(rng.integers(1, 700)) for _ in range(n)]
        cpu, gpu = model.partition(m_list)
        chosen = _makespan(model, m_list, cpu, gpu)

        best = float("inf")
        for mask in itertools.product([0, 1], repeat=n):
            c = [i for i in range(n) if mask[i] == 0]
            g = [i for i in range(n) if mask[i] == 1]
            best = min(best, _makespan(model, m_list, c, g))
        assert chosen <= best * 1.0 + 1e-12, (
            f"trial {trial}: prefix split {chosen:.3e} worse than "
            f"exhaustive optimum {best:.3e} (m_list={m_list})")
    print("  [test_partition_matches_exhaustive]  ok")


def _makespan_rows(model: DispatchModel, m_list, cpu_idx, gpu_idx,
                   row_split) -> float:
    """Predicted (penalized) makespan of a partition_rows() result."""
    cpu_e, cpu_m = len(cpu_idx), sum(m_list[i] for i in cpu_idx)
    gpu_pad = 0
    for i in gpu_idx:
        m = m_list[i]
        if row_split is not None and i == row_split[0]:
            m -= row_split[1]
        gpu_pad += model.padded(m)
    if row_split is not None:
        cpu_e += 1
        cpu_m += row_split[1]
    t_cpu = model.cpu.predict(cpu_e, cpu_m)
    t_gpu = model.gpu.predict(len(gpu_idx), gpu_pad)
    if cpu_e > 0 and gpu_idx:
        return max(t_cpu + float(model.gpu.coef[0]), t_gpu)
    return max(t_cpu, t_gpu)


def test_partition_rows_never_worse():
    """Row-splitting only ever lowers the predicted makespan (the c=0
    candidates are exactly the whole-expert prefix family)."""
    rng = np.random.default_rng(3)
    model = DispatchModel()
    for _ in range(50):
        _seed_model(
            model,
            cpu_coef=(rng.uniform(20e-6, 300e-6),
                      rng.uniform(10e-6, 80e-6),
                      rng.uniform(0.5e-6, 20e-6)),
            gpu_coef=(rng.uniform(50e-6, 400e-6),
                      rng.uniform(60e-6, 300e-6),
                      rng.uniform(0.01e-6, 0.5e-6)),
        )
        n = int(rng.integers(1, 10))
        m_list = [int(rng.integers(1, 1500)) for _ in range(n)]
        cpu_w, gpu_w = model.partition(m_list)
        cpu_r, gpu_r, split = model.partition_rows(m_list)
        assert sorted(cpu_r + gpu_r) == list(range(n))
        if split is not None:
            assert split[0] in gpu_r and 0 < split[1] < m_list[split[0]]
        whole = _makespan(model, m_list, cpu_w, gpu_w)
        rows = _makespan_rows(model, m_list, cpu_r, gpu_r, split)
        assert rows <= whole + 1e-12, f"{rows:.3e} > {whole:.3e} ({m_list})"
    print("  [test_partition_rows_never_worse]  ok")


def test_row_split_improves_hot_expert():
    """Equal hot experts under a GPU-bound regime: whole-expert splits are
    floor-bound by one expert, donating BLOCK_M-quantized rows to the idle
    CPU strictly improves the predicted makespan."""
    model = DispatchModel()
    _seed_model(model,
                cpu_coef=(100e-6, 30e-6, 1e-6),
                gpu_coef=(150e-6, 120e-6, 0.6e-6))
    m_list = [600, 600, 600]                     # padded(600) = 768, 3 blocks

    cpu_w, gpu_w = model.partition(m_list)
    whole = _makespan(model, m_list, cpu_w, gpu_w)

    cpu_r, gpu_r, split = model.partition_rows(m_list)
    rows = _makespan_rows(model, m_list, cpu_r, gpu_r, split)

    # Expected optimum: expert 0 on CPU, experts 1/2 on GPU, plus 88 rows of
    # expert 2 donated so its GPU remainder (512) exactly fills two blocks.
    assert split == (2, 88), f"unexpected split {split}"
    assert cpu_r == [0] and gpu_r == [1, 2]
    assert rows < whole * 0.95, \
        f"row split should strictly improve: {rows:.3e} vs {whole:.3e}"
    print(f"  [test_row_split_improves_hot_expert]  "
          f"whole={whole*1e6:.0f}us -> rows={rows*1e6:.0f}us split={split}")


def test_shape_aware_priors():
    """Shape-aware priors reproduce the legacy constants at the reference
    shape and scale with weight bytes / flops at other shapes."""
    ref = DispatchModel(hidden=1024, inter=2048)
    assert np.allclose(ref.cpu.coef, [100e-6, 30e-6, 8e-6], rtol=0.02)
    assert np.allclose(ref.gpu.coef, [150e-6, 120e-6, 0.05e-6], rtol=0.02)

    big = DispatchModel(hidden=4096, inter=14336)
    ratio = (4096 * 14336) / (1024 * 2048)
    assert np.isclose(big.cpu.coef[1] / ref.cpu.coef[1], ratio)
    assert np.isclose(big.cpu.coef[2] / ref.cpu.coef[2], ratio)
    assert np.isclose(big.gpu.coef[1] / ref.gpu.coef[1], ratio)
    assert np.isclose(big.cpu.coef[0], ref.cpu.coef[0])   # launch is shape-free

    # A tiny expert still belongs on CPU, a huge one on GPU, at both shapes.
    for m in (ref, big):
        cpu, gpu = m.partition([1])
        assert cpu == [0] and gpu == []
        cpu, gpu = m.partition([8192])
        assert cpu == [] and gpu == [0]
    print("  [test_shape_aware_priors]  ok")


def test_rates_roundtrip():
    """rates()/from_rates() transfer a fitted model across shapes: exact
    round-trip at the same shape, byte/flop-proportional at another."""
    m1 = DispatchModel(hidden=1024, inter=2048)
    m1.cpu.coef = np.array([2e-4, 4e-5, 6e-6])
    m1.gpu.coef = np.array([3e-4, 0.0, 2e-8])    # zero coef → infinite rate

    same = DispatchModel.from_rates(m1.rates(), hidden=1024, inter=2048)
    assert np.allclose(same.cpu.coef, m1.cpu.coef)
    assert np.allclose(same.gpu.coef, m1.gpu.coef)

    other = DispatchModel.from_rates(m1.rates(), hidden=2048, inter=4096)
    ratio = (2048 * 4096) / (1024 * 2048)
    assert np.isclose(other.cpu.coef[0], m1.cpu.coef[0])
    assert np.isclose(other.cpu.coef[1], m1.cpu.coef[1] * ratio)
    assert np.isclose(other.cpu.coef[2], m1.cpu.coef[2] * ratio)
    assert other.gpu.coef[1] == 0.0
    print("  [test_rates_roundtrip]  ok")


def test_online_refit_shifts_decision():
    """Feed observations describing a very slow CPU; the model must move
    work to the GPU that priors would have kept on CPU."""
    model = DispatchModel()
    m_list = [8, 8, 8, 8]
    cpu0, gpu0 = model.partition(m_list)
    assert len(cpu0) > 0, "priors should keep some tiny experts on CPU"

    # Observed: CPU is catastrophically slow (10 ms per row).
    for e in range(1, 6):
        for m in (4, 16, 64):
            model.record_cpu(e, e * m, seconds=e * m * 10e-3)
    # Observed: GPU is fast and flat.
    for e in range(1, 6):
        for m in (4, 16, 64):
            model.gpu.record(e, e * model.padded(m), seconds=50e-6 * e)

    cpu1, gpu1 = model.partition(m_list)
    assert len(gpu1) > len(gpu0), "refit model should offload to the fast GPU"
    print("  [test_online_refit_shifts_decision]  "
          f"before={len(cpu0)}cpu → after={len(cpu1)}cpu/{len(gpu1)}gpu")


def test_refit_recovers_coefficients():
    """Least-squares refit recovers planted coefficients from clean data."""
    truth = np.array([2e-4, 5e-5, 3e-6])
    m = _BackendModel(prior=np.array([1e-4, 1e-5, 1e-6]))
    rng = np.random.default_rng(2)
    for _ in range(40):
        e = int(rng.integers(1, 16))
        s = int(rng.integers(1, 4096))
        m.record(e, s, float(truth @ [1, e, s]))
    pred = m.predict(8, 1000)
    ref = float(truth @ [1, 8, 1000])
    assert abs(pred - ref) / ref < 0.05, f"{pred:.3e} vs {ref:.3e}"
    print(f"  [test_refit_recovers_coefficients]  pred={pred:.3e} ref={ref:.3e}")


def test_outliers_and_degenerate_records_ignored():
    m = _BackendModel(prior=np.array([1e-4, 1e-5, 1e-6]))
    m.record(0, 0, 1.0)       # zero experts → dropped
    m.record(4, 64, -1.0)     # negative time → dropped
    assert len(m.obs) == 0
    print("  [test_outliers_and_degenerate_records_ignored]  ok")


class _FakeEvent:
    """Stand-in for torch.cuda.Event so harvest() is testable without CUDA."""

    def __init__(self, ms: float, done: bool = True):
        self.ms = ms
        self.done = done

    def query(self):
        return self.done

    def synchronize(self):
        self.done = True

    def elapsed_time(self, other):
        return other.ms - self.ms


def test_lazy_event_harvest():
    model = DispatchModel()
    done = (_FakeEvent(0.0), _FakeEvent(2.0), 3, 768)
    pending = (_FakeEvent(0.0), _FakeEvent(5.0, done=False), 2, 512)
    model.record_gpu_events(*done)
    model.record_gpu_events(*pending)

    assert model.harvest() == 1                 # only the completed pair
    assert len(model.gpu.obs) == 1
    assert model.snapshot()["pending_gpu"] == 1

    assert model.harvest(wait=True) == 1        # blocking drain
    assert model.snapshot()["pending_gpu"] == 0
    assert len(model.gpu.obs) == 2
    print("  [test_lazy_event_harvest]  ok")


def test_paused_drops_records():
    """paused=True makes both record paths no-ops (used by Layer.calibrate
    for the per-shape warm-up forward)."""
    model = DispatchModel()
    model.paused = True
    model.record_cpu(2, 64, 1e-3)
    model.record_gpu_events(_FakeEvent(0.0), _FakeEvent(2.0), 3, 768)
    assert len(model.cpu.obs) == 0
    assert model.snapshot()["pending_gpu"] == 0
    model.paused = False
    model.record_cpu(2, 64, 1e-3)
    assert len(model.cpu.obs) == 1
    print("  [test_paused_drops_records] ok")


if __name__ == "__main__":
    test_empty_and_forced_cases()
    test_small_m_prefers_cpu_large_m_prefers_gpu()
    test_partition_beats_all_single_backend()
    test_partition_matches_exhaustive()
    test_partition_rows_never_worse()
    test_row_split_improves_hot_expert()
    test_shape_aware_priors()
    test_rates_roundtrip()
    test_online_refit_shifts_decision()
    test_refit_recovers_coefficients()
    test_outliers_and_degenerate_records_ignored()
    test_lazy_event_harvest()
    test_paused_drops_records()
    print("All dispatch-model tests passed.")
