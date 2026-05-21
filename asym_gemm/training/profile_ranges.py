"""Lightweight profiling ranges for AsymGEMM training paths.

The helpers are no-ops unless profiling is explicitly enabled, so they can live
in shared wrappers that are used by toy profiles and later LLaMA-Factory.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

import torch


_enabled: ContextVar[bool] = ContextVar("asym_gemm_profile_ranges_enabled", default=False)
_current_range: ContextVar[str] = ContextVar("asym_gemm_profile_current_range", default="")


def set_profile_enabled(enabled: bool) -> None:
    _enabled.set(bool(enabled))


def is_profile_enabled() -> bool:
    return bool(_enabled.get())


def current_profile_range() -> str:
    return str(_current_range.get())


def scoped_name(*parts: object) -> str:
    return ".".join(str(part).strip(".") for part in parts if str(part).strip("."))


@contextmanager
def prof_range(name: str, *, enabled: bool | None = None) -> Iterator[None]:
    active = is_profile_enabled() if enabled is None else bool(enabled)
    if not active:
        yield
        return

    token = _current_range.set(name)
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
            _current_range.reset(token)
    else:
        try:
            with torch.autograd.profiler.record_function(name):
                yield
        finally:
            _current_range.reset(token)


@contextmanager
def profile_enabled(enabled: bool = True) -> Iterator[None]:
    token = _enabled.set(bool(enabled))
    try:
        yield
    finally:
        _enabled.reset(token)


__all__ = [
    "current_profile_range",
    "is_profile_enabled",
    "prof_range",
    "profile_enabled",
    "scoped_name",
    "set_profile_enabled",
]
