"""In-vitro test for fix_merged.md V1/V2: can the ActivationOffloadManager API,
used exactly as the flagged call sites use it, observe/corrupt data?

No training, no model. One GPU, seconds. The GPU stream is stalled with
torch.cuda._sleep so an enqueued D2H provably hasn't run when the host proceeds —
the same window a busy training stream creates naturally.

TEST B (V1): offload() -> wait_cpu_ready() -> immediate HOST read of handle.tensor
  (what cpu_left.py:188 / qwen3_moe.py:910 / attention:809 do). wait_cpu_ready is
  device-side only, so the host read should be able to observe PRE-COPY bytes.
TEST A (V2): offload() -> release_cpu() -> empty_cpu() (pool pop of the same
  buffer) -> host zero_() by the new owner -> the in-flight D2H lands late and
  overwrites the new owner's bytes.
"""
import torch

from asym_gemm.training.activation_offload import ActivationOffloadManager

assert torch.cuda.is_available()
dev = torch.device("cuda:0")
torch.cuda.set_device(dev)
SLEEP_CYCLES = 4_000_000_000  # ~2s stream stall on GB200
ROWS, WIDTH = 16384, 512      # rows = 2 x 8192 pool granule -> exact-bucket key

mgr = ActivationOffloadManager(pin_memory=True)


def poison_pool(value: float) -> None:
    """Leave a pinned buffer full of `value` in the pool under our shape key."""
    h = mgr.empty_cpu((ROWS, WIDTH), torch.bfloat16, dev, "poison")
    with torch.no_grad():
        h.tensor.fill_(value)
    mgr.release_cpu(h)
    torch.cuda.synchronize()


print("== TEST B (V1): host read after device-only wait_cpu_ready ==")
poison_pool(7.0)
src = torch.full((ROWS, WIDTH), 3.0, device=dev, dtype=torch.bfloat16)
torch.cuda.synchronize()
torch.cuda._sleep(SLEEP_CYCLES)          # stall the stream
h = mgr.offload(src, "testB")            # pops the 7.0-poisoned buffer; D2H of 3.0 pending behind the stall
mgr.wait_cpu_ready(h)                    # device-side wait only (the "never block the host" rule)
host_view = float(h.tensor.float().mean())   # the host read, exactly as the flagged sites do it
torch.cuda.synchronize()
after = float(h.tensor.float().mean())
if abs(host_view - 7.0) < 0.25:
    print(f"  RACE PROVEN: host read observed pre-copy bytes (mean={host_view:.3f}, expected 3.0; after sync={after:.3f})")
elif abs(host_view - 3.0) < 0.25:
    print(f"  clean: host read saw the copied data (mean={host_view:.3f}) — some sync intervened")
else:
    print(f"  TORN/MIXED read: mean={host_view:.3f} (neither 3.0 nor 7.0; after sync={after:.3f})")
mgr.release_cpu(h)
torch.cuda.synchronize()

print("== TEST A (V2): pool recycle while the D2H is still in flight ==")
src2 = torch.full((ROWS, WIDTH), 5.0, device=dev, dtype=torch.bfloat16)
torch.cuda.synchronize()
torch.cuda._sleep(SLEEP_CYCLES)          # stall the stream
h2 = mgr.offload(src2, "testA")          # D2H of 5.0 pending behind the stall
mgr.release_cpu(h2)                      # buffer returns to the pool immediately
nb = mgr.empty_cpu((ROWS, WIDTH), torch.bfloat16, dev, "newowner")  # pops the SAME buffer
same_buffer = nb.tensor.data_ptr() == h2.tensor.data_ptr()
with torch.no_grad():
    nb.tensor.zero_()                    # new owner's host write, race window open
torch.cuda.synchronize()                 # let the stalled D2H land
mx = float(nb.tensor.float().abs().max())
print(f"  recycled same buffer: {same_buffer}")
if not same_buffer:
    print("  INCONCLUSIVE: pool handed back a different buffer — key mismatch, rerun")
elif mx > 0.5:
    print(f"  RACE PROVEN: late D2H overwrote the new owner's zeros (abs max={mx:.3f}, source was 5.0)")
else:
    print("  clean: new owner's zeros survived — some ordering intervened")

print("== TEST C (fix): wait_cpu_ready_host must close both holes ==")
# C1: event still pending -> host variant synchronizes the event
poison_pool(7.0)
src3 = torch.full((ROWS, WIDTH), 3.0, device=dev, dtype=torch.bfloat16)
torch.cuda.synchronize()
torch.cuda._sleep(SLEEP_CYCLES)
h3 = mgr.offload(src3, "testC1")
mgr.wait_cpu_ready_host(h3)
v1 = float(h3.tensor.float().mean())
print(f"  C1 (event pending):   host read = {v1:.3f} -> {'FIXED' if abs(v1-3.0) < 0.25 else 'STILL BROKEN'}")
mgr.release_cpu(h3)
torch.cuda.synchronize()

# C2: event already consumed by a device-side waiter -> fallback stream-sync must cover it
poison_pool(7.0)
src4 = torch.full((ROWS, WIDTH), 3.0, device=dev, dtype=torch.bfloat16)
torch.cuda.synchronize()
torch.cuda._sleep(SLEEP_CYCLES)
h4 = mgr.offload(src4, "testC2")
mgr.wait_cpu_ready(h4)        # device-side waiter POPS the event (the hard case)
mgr.wait_cpu_ready_host(h4)   # must fall back to stream-sync, not no-op
v2 = float(h4.tensor.float().mean())
print(f"  C2 (event consumed):  host read = {v2:.3f} -> {'FIXED' if abs(v2-3.0) < 0.25 else 'STILL BROKEN'}")
mgr.release_cpu(h4)
torch.cuda.synchronize()
