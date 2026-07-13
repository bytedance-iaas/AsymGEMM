"""Shared pinned weight fabric for asym_ep2 (fix_gb200_ep.md S1 DELTA 2).

ONE /dev/shm file holds every frozen weight bank ONCE; each rank mmaps it and
cudaHostRegisters the used range ONCE at seal(). rank0 writes bank bytes at adopt
time; other ranks discard their own (identical) copy and map the same range. Bank
construction order is deterministic (model traversal), cross-checked at seal via a
manifest + per-bank sample hashes. Enabled by ASYM_ARENA_SHM=1 under torchrun
(WORLD_SIZE >= 2); completely inert otherwise.

Lifecycle: get_or_create() during the LF surgery (UNREGISTERED views — no GEMM may
touch them yet); seal() from run_lf_profiled_train right before trainer.train()
(torch.distributed is initialized there): barrier -> manifest cross-check -> sample
hash check -> ONE cudaHostRegister of [0, used). Registration makes every fabric
view report is_pinned()==True, which the asym GEMM host-streaming path requires.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import mmap
import os
import time

import torch

__all__ = ["fabric_enabled", "get_fabric", "SharedFabric"]

_ALIGN = 4096
_SAMPLE_BYTES = 1 << 20  # head+tail sample per bank for the rank>0 divergence check
_FABRIC: "SharedFabric | None" = None


def fabric_enabled() -> bool:
    return (
        os.environ.get("ASYM_ARENA_SHM") == "1"
        and int(os.environ.get("WORLD_SIZE", "1")) >= 2
    )


def get_fabric() -> "SharedFabric | None":
    global _FABRIC
    if _FABRIC is None and fabric_enabled():
        _FABRIC = SharedFabric()
    return _FABRIC


def _sample_hash(flat_u8: torch.Tensor) -> str:
    n = flat_u8.numel()
    take = min(_SAMPLE_BYTES, n)
    h = hashlib.sha1()
    h.update(flat_u8[:take].numpy().tobytes())
    if n > take:
        h.update(flat_u8[n - take :].numpy().tobytes())
    return h.hexdigest()


class SharedFabric:
    def __init__(self) -> None:
        self.rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        cap_gb = float(os.environ.get("ASYM_ARENA_SHM_CAP_GB", "160"))
        self.cap = int(cap_gb * (1 << 30))
        tag = os.environ.get("ASYM_ARENA_SHM_TAG") or os.environ.get("MASTER_PORT", "0")
        self.path = os.environ.get("ASYM_ARENA_SHM_PATH") or f"/dev/shm/asym_fabric_{tag}"
        self._entries: list[dict] = []
        self._used = 0
        self._seq = 0
        self._sample_hashes: dict[str, str] = {}
        self._sealed = False
        self._register_seconds = 0.0
        self._late_banks = 0

        if self.rank == 0:
            # stale-file guard: a crash leftover would silently map OLD weights
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
            for suffix in (".manifest.json", ".done"):
                try:
                    os.unlink(self.path + suffix)
                except FileNotFoundError:
                    pass
            fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_EXCL, 0o600)
        else:
            deadline = time.time() + 300.0
            while not os.path.exists(self.path):
                if time.time() > deadline:
                    raise RuntimeError(f"shared fabric: {self.path} never created by rank0")
                time.sleep(0.05)
            fd = os.open(self.path, os.O_RDWR)
        os.ftruncate(fd, self.cap)  # virtual on tmpfs; pages materialize on first write
        self._mm = mmap.mmap(fd, self.cap)
        os.close(fd)
        # uint8 alias of the whole mapping; every bank is a dtype/shape view into it
        self._base = torch.frombuffer(self._mm, dtype=torch.uint8, count=self.cap)
        atexit.register(self._cleanup)

    # ---- adopt-time API (surgery; pre-seal; views are NOT yet pinned) ----

    def get_or_create(self, name: str, src: torch.Tensor) -> "torch.Tensor | None":
        if self._sealed:
            # latecomer bank (built after seal, e.g. a lazy cache the pre-build pass
            # missed): caller falls back to a private pin_memory copy — correct but
            # NOT deduplicated; counted so the receipt shows what escaped the fabric.
            self._late_banks += 1
            return None
        src = src.detach()
        if not src.is_contiguous():
            src = src.contiguous()
        nbytes = src.numel() * src.element_size()
        key = f"{self._seq:05d}:{name or 'bank'}"
        self._seq += 1
        offset = (self._used + _ALIGN - 1) // _ALIGN * _ALIGN
        if offset + nbytes > self.cap:
            raise RuntimeError(
                f"shared fabric cap exceeded at {key} (used={self._used}, need={nbytes}); "
                f"raise ASYM_ARENA_SHM_CAP_GB (cap={self.cap})"
            )
        self._used = offset + nbytes
        raw = self._base[offset : offset + nbytes]
        view = raw.view(src.dtype).view(src.shape)
        if self.rank == 0:
            view.copy_(src)
        else:
            # keep a cheap fingerprint of OUR OWN bytes; verified vs fabric at seal
            self._sample_hashes[key] = _sample_hash(src.reshape(-1).view(torch.uint8))
        self._entries.append(
            {
                "key": key,
                "offset": offset,
                "nbytes": nbytes,
                "dtype": str(src.dtype),
                "shape": list(src.shape),
            }
        )
        return view

    # ---- seal (call ONCE, post-surgery, pre-train; dist initialized) ----

    def seal(self) -> None:
        if self._sealed:
            return
        manifest_path = self.path + ".manifest.json"
        done_path = self.path + ".done"
        if self.rank == 0:
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump({"used": self._used, "entries": self._entries}, fh)
            with open(done_path, "w", encoding="utf-8") as fh:
                fh.write("ok")
        self._barrier(done_path)
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        if manifest["entries"] != self._entries or manifest["used"] != self._used:
            raise RuntimeError(
                "shared fabric: manifest mismatch across ranks — bank construction order "
                f"diverged (rank{self.rank}: {len(self._entries)} entries/{self._used}B vs "
                f"rank0: {len(manifest['entries'])}/{manifest['used']}B)"
            )
        if self.rank != 0:
            for entry in self._entries:
                key = entry["key"]
                want = self._sample_hashes.get(key)
                if want is None:
                    continue
                raw = self._base[entry["offset"] : entry["offset"] + entry["nbytes"]]
                got = _sample_hash(raw)
                if got != want:
                    raise RuntimeError(
                        f"shared fabric: bank byte divergence at {key} — rank0's write "
                        "does not match this rank's locally-built weights"
                    )
        reg_bytes = (self._used + _ALIGN - 1) // _ALIGN * _ALIGN
        start = time.perf_counter()
        rc = torch.cuda.cudart().cudaHostRegister(self._base.data_ptr(), reg_bytes, 0)
        self._register_seconds = time.perf_counter() - start
        if int(rc) != 0:
            raise RuntimeError(f"cudaHostRegister({reg_bytes}B) failed rc={int(rc)}")
        if self._entries:
            first = self._entries[0]
            probe = self._base[first["offset"] : first["offset"] + first["nbytes"]]
            if not probe.is_pinned():
                raise RuntimeError("shared fabric: registered range does not report is_pinned")
        self._sealed = True

    def stats(self) -> dict:
        return {
            "path": self.path,
            "rank": self.rank,
            "banks": len(self._entries),
            "used_bytes": self._used,
            "sealed": self._sealed,
            "register_seconds": self._register_seconds,
            "late_banks": self._late_banks,
        }

    # ---- internals ----

    def _barrier(self, done_path: str) -> None:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                return
        except Exception:
            pass
        deadline = time.time() + 900.0
        while not os.path.exists(done_path):
            if time.time() > deadline:
                raise RuntimeError("shared fabric: seal barrier timed out (rank0 never sealed)")
            time.sleep(0.1)

    def _cleanup(self) -> None:
        try:
            if self._sealed:
                torch.cuda.cudart().cudaHostUnregister(self._base.data_ptr())
        except Exception:
            pass
        try:
            if self.rank == 0:
                os.unlink(self.path)
        except Exception:
            pass
