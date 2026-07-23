"""Total pinned-memory accounting + per-tag-family caps (fix_cpu_compute.md item 4).

The dense-32B OOMs were NODE-level page-locked growth invisible to process RSS
(final 32B diagnosis, cpu_compute.md 2026-07-15). This module gives every
pin_memory site we control a tag-family ledger:

- attribution: live pinned bytes + high-water per family (act pool ``moe.*``/
  ``attn.*``/``dense.*``, boundary ``gc.*``, deposit slots, adam buffers,
  save_on_cpu packs), released exactly when the tensor is GC'd
  (``weakref.finalize``), plus the OS truth alongside it in the profile
  (``torch.cuda.host_memory_stats()`` and /proc/meminfo Mlocked/Unevictable —
  the torch caching host allocator keeps freed blocks page-locked, so live
  bytes here are a lower bound on OS page-locked bytes).
- enforcement: per-family caps ``ASYM_PINNED_CAP_GB_<FAMILY>`` and one global
  cap ``ASYM_PINNED_CAP_TOTAL_GB`` (both default 0 = unlimited, i.e. the
  feature is default-off). When a cap would be exceeded the allocation FALLS
  BACK TO UNPINNED (never OOMs, never blocks): non_blocking copies degrade to
  synchronous but stay correct; deposit paths treat a denial as "use the GPU
  path". One WARN line per family on first denial (P11).

Dependency-light (os/sys/threading/weakref + torch); imported from the pinned
alloc chokepoints, must never cycle.
"""

from __future__ import annotations

import os
import sys
import threading
import weakref
from typing import Optional

import torch

__all__ = [
    "family_of",
    "try_reserve",
    "release",
    "register_tensor",
    "pinned_allowed",
    "stats",
    "reset_for_tests",
]

_LOCK = threading.RLock()
_LIVE: dict[str, int] = {}
_HIGH: dict[str, int] = {}
_COUNT: dict[str, int] = {}
_DENIALS: dict[str, int] = {}
_WARNED: set = set()
_TOTAL_LIVE = 0
_TOTAL_HIGH = 0
_CAPS: Optional[dict] = None


def _env_gb(name: str) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return 0
    try:
        return int(float(raw) * (1 << 30))
    except ValueError:
        return 0


def _caps() -> dict:
    global _CAPS
    if _CAPS is None:
        with _LOCK:
            if _CAPS is None:
                caps = {"total": _env_gb("ASYM_PINNED_CAP_TOTAL_GB")}
                prefix = "ASYM_PINNED_CAP_GB_"
                for key, _ in os.environ.items():
                    if key.startswith(prefix) and key != "ASYM_PINNED_CAP_GB_TOTAL":
                        fam = key[len(prefix):].lower()
                        caps[fam] = _env_gb(key)
                _CAPS = caps
    return _CAPS


def family_of(tag: str | None) -> str:
    if not tag:
        return "untagged"
    return str(tag).split(".", 1)[0].lower()


def pinned_allowed(family: str, nbytes: int) -> bool:
    """Check-only (no booking): would a pinned allocation of nbytes fit the caps?"""
    caps = _caps()
    fam_cap = caps.get(family, 0)
    total_cap = caps.get("total", 0)
    with _LOCK:
        if fam_cap and _LIVE.get(family, 0) + int(nbytes) > fam_cap:
            return False
        if total_cap and _TOTAL_LIVE + int(nbytes) > total_cap:
            return False
    return True


def try_reserve(family: str, nbytes: int) -> bool:
    """Book nbytes against the family + total caps. False = caller must fall back
    to unpinned (or its non-pinned code path); nothing is booked on denial."""
    global _TOTAL_LIVE, _TOTAL_HIGH
    family = str(family)
    nbytes = int(nbytes)
    caps = _caps()
    fam_cap = caps.get(family, 0)
    total_cap = caps.get("total", 0)
    warn = False
    with _LOCK:
        if (fam_cap and _LIVE.get(family, 0) + nbytes > fam_cap) or (
            total_cap and _TOTAL_LIVE + nbytes > total_cap
        ):
            _DENIALS[family] = _DENIALS.get(family, 0) + 1
            if family not in _WARNED:
                _WARNED.add(family)
                warn = True
            denied = True
        else:
            _LIVE[family] = _LIVE.get(family, 0) + nbytes
            _COUNT[family] = _COUNT.get(family, 0) + 1
            _HIGH[family] = max(_HIGH.get(family, 0), _LIVE[family])
            _TOTAL_LIVE += nbytes
            _TOTAL_HIGH = max(_TOTAL_HIGH, _TOTAL_LIVE)
            denied = False
    if warn:
        print(
            f"[asym-pinned-ledger] cap DENIAL for family '{family}' "
            f"(live={_LIVE.get(family, 0)} B, request={nbytes} B, fam_cap={fam_cap} B, "
            f"total_live={_TOTAL_LIVE} B, total_cap={total_cap} B) -> unpinned fallback",
            file=sys.stderr,
            flush=True,
        )
    return not denied


def release(family: str, nbytes: int) -> None:
    global _TOTAL_LIVE
    with _LOCK:
        _LIVE[family] = max(0, _LIVE.get(family, 0) - int(nbytes))
        _TOTAL_LIVE = max(0, _TOTAL_LIVE - int(nbytes))


def register_tensor(tensor: torch.Tensor, family: str) -> None:
    """Attach the auto-release: the bytes booked by try_reserve() are released
    when the pinned tensor is GC'd (works through pool caching — a pooled buffer
    stays booked while it stays pinned/alive)."""
    nbytes = tensor.numel() * tensor.element_size()
    weakref.finalize(tensor, release, family, int(nbytes))


def stats() -> dict:
    with _LOCK:
        out = {
            "live_bytes": dict(_LIVE),
            "high_water_bytes": dict(_HIGH),
            "alloc_count": dict(_COUNT),
            "denials": dict(_DENIALS),
            "total_live_bytes": _TOTAL_LIVE,
            "total_high_water_bytes": _TOTAL_HIGH,
            "caps_bytes": dict(_caps()),
        }
    try:
        if torch.cuda.is_available() and hasattr(torch.cuda, "host_memory_stats"):
            hs = torch.cuda.host_memory_stats()
            out["torch_host_allocator"] = {
                "allocated_bytes_current": hs.get("allocated_bytes.all.current"),
                "allocated_bytes_peak": hs.get("allocated_bytes.all.peak"),
                "reserved_bytes_current": hs.get("reserved_bytes.all.current"),
                "reserved_bytes_peak": hs.get("reserved_bytes.all.peak"),
            }
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("Mlocked:", "Unevictable:")):
                    k, v = line.split(":", 1)
                    out[f"meminfo_{k.lower()}_kb"] = int(v.strip().split()[0])
    except OSError:
        pass
    return out


def reset_for_tests() -> None:
    global _CAPS, _TOTAL_LIVE, _TOTAL_HIGH
    with _LOCK:
        _CAPS = None
        _LIVE.clear()
        _HIGH.clear()
        _COUNT.clear()
        _DENIALS.clear()
        _WARNED.clear()
        _TOTAL_LIVE = 0
        _TOTAL_HIGH = 0
