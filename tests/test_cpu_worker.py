"""Unit tests for asym_gemm.training.cpu_worker (cpu_compute.md Stage 2).

Run: .venv/bin/python tests/test_cpu_worker.py            (CPU-only parts)
     .venv/bin/python tests/test_cpu_worker.py cuda       (+ D2H event ordering test; needs a GPU)
"""

import sys
import time

import torch

from asym_gemm.training import cpu_worker


def test_submit_wait_result():
    t = cpu_worker.submit(lambda: 41 + 1)
    assert cpu_worker.wait(t) == 42


def test_exception_propagates_and_worker_survives():
    def boom():
        raise ValueError("boom")
    t = cpu_worker.submit(boom)
    try:
        cpu_worker.wait(t)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    assert cpu_worker.wait(cpu_worker.submit(lambda: "alive")) == "alive"


def test_no_grad_is_set_in_worker():
    with torch.enable_grad():
        t = cpu_worker.submit(torch.is_grad_enabled)
        assert cpu_worker.wait(t) is False


def test_parallelism():
    # ATen CPU ops release the GIL: total wall for a worker op overlapped with a
    # main-thread op must be well under the serial sum.
    a = torch.randn(4096, 4096)
    b = torch.randn(4096, 4096)
    torch.set_num_threads(8)
    t0 = time.perf_counter(); a @ a; s1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    task = cpu_worker.submit(lambda: a @ a)
    b @ b
    cpu_worker.wait(task)
    both = time.perf_counter() - t0
    assert both < 1.8 * s1, f"no overlap: serial-one={s1:.3f}s both={both:.3f}s"


def test_d2h_event_ordering_cuda():
    # producer stream: slow kernel -> async D2H into pinned; worker host-waits the
    # event then reads. Without the event.synchronize() this reads garbage.
    dev = torch.device("cuda")
    n = 1 << 22
    src = torch.zeros(n, device=dev)
    pinned = torch.full((n,), -1.0, pin_memory=True)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(50):
            src = src + 1.0  # keep the stream busy
        pinned.copy_(src, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(s)

    def job():
        ev.synchronize()
        return float(pinned[::65536].min()), float(pinned[::65536].max())

    lo, hi = cpu_worker.wait(cpu_worker.submit(job))
    assert lo == hi == 50.0, (lo, hi)


if __name__ == "__main__":
    test_submit_wait_result()
    test_exception_propagates_and_worker_survives()
    test_no_grad_is_set_in_worker()
    test_parallelism()
    if len(sys.argv) > 1 and sys.argv[1] == "cuda" and torch.cuda.is_available():
        test_d2h_event_ordering_cuda()
        print("cpu_worker tests passed (incl. CUDA ordering)")
    else:
        print("cpu_worker tests passed (CPU-only)")


def test_bg_submit_from_worker_like_thread():
    # Regression: get_bg_worker() under _LOCK calls get_worker() (re-entrant acquire);
    # with a plain Lock this deadlocked the autograd thread on first submit_deposit.
    import os, threading
    os.environ["ASYM_CPU_WORKER_BG"] = "1"
    result = {}

    def autograd_like():
        t = cpu_worker.submit_deposit(lambda: 7)
        result["v"] = cpu_worker.wait(t)

    th = threading.Thread(target=autograd_like)
    th.start()
    th.join(timeout=10)
    assert not th.is_alive(), "submit_deposit deadlocked (BG lock regression)"
    assert result.get("v") == 7
    os.environ.pop("ASYM_CPU_WORKER_BG", None)


def test_deposit_retention_backpressure():
    # Backpressure: with a slow worker, retained deferred bytes must never exceed
    # the budget — the producer blocks on the oldest task instead.
    import os, time
    os.environ["ASYM_DEPOSIT_RETAIN_BUDGET_GB"] = str(3 / 1024)  # 3 MiB budget
    import importlib
    from asym_gemm.training import attention_activation_offload as at
    importlib.reload(at)
    from asym_gemm.training.activation_offload import ActivationOffloadManager

    mgr = ActivationOffloadManager(pin_memory=False)
    handles = [mgr.offload(torch.zeros(1024 * 512, dtype=torch.bfloat16), f"h{i}") for i in range(6)]  # 1 MiB each
    max_retained = 0
    t0 = time.perf_counter()
    for h in handles:
        task = cpu_worker.submit(lambda: time.sleep(0.25))
        at._defer_deposit_release(task, mgr, h, None)
        max_retained = max(max_retained, at._DEPOSIT_RETAINED_BYTES)
    assert max_retained <= 3 * (1 << 20), f"budget breached: {max_retained}"
    assert time.perf_counter() - t0 > 0.4, "producer never blocked — backpressure inert"
    at._sweep_attn_deposit_releases(force=True)
    assert at._DEPOSIT_RETAINED_BYTES == 0
    os.environ.pop("ASYM_DEPOSIT_RETAIN_BUDGET_GB", None)
