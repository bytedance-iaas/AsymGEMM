"""JIT CPU<->GPU staging for trainable expert-LoRA weights (ZeRO-3 offload_param, single GPU).

The trainable expert-LoRA banks (``gate/up/down`` x ``A/B``) for one decoder layer live
in a pinned bf16 CPU "home" slab. Before that layer's experts run we copy the whole layer's
banks to GPU in ONE H2D (a contiguous slab), point each bank's ``.data`` at a view of the
staged buffer, run the unchanged grouped GEMM, then release the buffer. This removes the
~6.19 GiB of always-resident expert-LoRA weights from the cross-entropy/lm-head peak.

Design notes (memory / latency / launch efficiency):
- One contiguous home slab and ONE ``copy_`` per layer per direction (not per bank, never per
  expert). The compute path (``grouped_mm``) is untouched: all experts still run in one grouped
  GEMM. We only move where the weight bytes live, not how they are multiplied.
- A single GPU staging slab is reused across all layers via a size-keyed pool, so steady-state
  staging HBM is ~one layer (~132 MiB for Qwen3-30B-A3B), not 6.19 GiB.
- The 0-size placeholder is a CUDA tensor, so the optimizer's CUDA-residency guards keep passing
  while a bank is "released".
- Backward re-gathers in the custom expert Function; per-bank release is owned by the optimizer's
  post-accumulate grad hook (runs strictly after autograd's AccumulateGrad), refcounted per layer.
"""

from __future__ import annotations

from typing import Any

import torch

_DEFAULT_PERSIST_NUMEL = 1_048_576

_BANK_NAMES = ("gate_lora_A", "gate_lora_B", "up_lora_A", "up_lora_B", "down_lora_A", "down_lora_B")


class _LayerGroup:
    """Staging state for the six LoRA banks of one ``AsymQwen3Experts`` (one decoder layer)."""

    __slots__ = (
        "module_key",
        "params",
        "shapes",
        "numels",
        "offsets",
        "total_numel",
        "dtype",
        "device",
        "home",
        "placeholders",
        "bank_index",
        "buf",
        "live",
    )

    def __init__(self) -> None:
        self.module_key: int = 0
        self.params: list[torch.nn.Parameter] = []
        self.shapes: list[tuple[int, ...]] = []
        self.numels: list[int] = []
        self.offsets: list[int] = []
        self.total_numel: int = 0
        self.dtype: torch.dtype = torch.bfloat16
        self.device: torch.device = torch.device("cpu")
        self.home: torch.Tensor | None = None  # pinned bf16 CPU slab [total_numel]
        self.placeholders: dict[int, torch.Tensor] = {}
        self.bank_index: dict[int, int] = {}
        self.buf: torch.Tensor | None = None  # live GPU slab while gathered, else None
        self.live: int = 0


class LoRAWeightOffloadCoordinator:
    """Owns CPU homes + GPU staging for expert-LoRA weights; gather before use, release after."""

    def __init__(
        self,
        *,
        pin_memory: bool = True,
        persistence_threshold_numel: int = _DEFAULT_PERSIST_NUMEL,
    ) -> None:
        self.pin_memory = bool(pin_memory)
        self.persistence_threshold_numel = int(persistence_threshold_numel)
        self._groups: list[_LayerGroup] = []
        self._group_of_module: dict[int, _LayerGroup] = {}
        self._group_of_key: dict[int, _LayerGroup] = {}
        self._registered_keys: set[int] = set()
        self.registered_params: list[torch.nn.Parameter] = []
        self._pool: dict[int, list[torch.Tensor]] = {}  # total_numel -> free GPU slabs
        self._live_groups = 0
        self.staged_high_water_bytes = 0
        self.pin_memory_failures: list[str] = []
        # Stage 3 (prefetch) fields, unused in the synchronous core.
        self._stream: torch.cuda.Stream | None = None

    # -- registration -----------------------------------------------------------------

    def is_registered(self, param: torch.nn.Parameter) -> bool:
        return id(param) in self._registered_keys

    @torch.no_grad()
    def register_group(self, module: Any, named_banks: list[tuple[str, torch.nn.Parameter]]) -> int:
        """Capture the layer's banks into a pinned bf16 home slab, then release them.

        Must be called while the params are still full on CUDA (before any release), so the
        captured home equals the optimizer's freshly-built fp32 master.
        """
        params = [param for _, param in named_banks]
        if not params:
            return 0
        if id(module) in self._group_of_module:
            return len(self._group_of_module[id(module)].params)
        total = sum(int(param.numel()) for param in params)
        if total < self.persistence_threshold_numel:
            return 0  # keep tiny adapters resident; offloading them is all latency, no benefit

        dtype = params[0].dtype
        device = params[0].device
        for param in params:
            if param.dtype != dtype or param.device != device:
                raise RuntimeError("LoRA weight offload requires all banks in a layer to share dtype/device")

        group = _LayerGroup()
        group.module_key = id(module)
        group.params = params
        group.shapes = [tuple(int(d) for d in param.shape) for param in params]
        group.numels = [int(param.numel()) for param in params]
        offset = 0
        for numel in group.numels:
            group.offsets.append(offset)
            offset += numel
        group.total_numel = total
        group.dtype = dtype
        group.device = device

        home = torch.empty(total, device="cpu", dtype=dtype)
        if self.pin_memory and torch.cuda.is_available() and not home.is_pinned():
            try:
                home = home.pin_memory()
            except RuntimeError as exc:  # pragma: no cover - depends on host
                self.pin_memory_failures.append(str(exc))
        for index, param in enumerate(params):
            start = group.offsets[index]
            home[start : start + group.numels[index]].copy_(param.detach().reshape(-1))
        group.home = home
        group.placeholders = {id(param): torch.empty(0, dtype=dtype, device=device) for param in params}
        group.bank_index = {id(param): index for index, param in enumerate(params)}

        self._groups.append(group)
        self._group_of_module[group.module_key] = group
        for param in params:
            self._group_of_key[id(param)] = group
            self._registered_keys.add(id(param))
            self.registered_params.append(param)

        self._release_group(group)  # start released; nothing resident until first gather
        return len(params)

    # -- staging ----------------------------------------------------------------------

    def _take_slab(self, total_numel: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        pool = self._pool.get(total_numel)
        if pool:
            return pool.pop()
        return torch.empty(total_numel, dtype=dtype, device=device)

    def _return_slab(self, slab: torch.Tensor) -> None:
        self._pool.setdefault(int(slab.numel()), []).append(slab)

    @torch.no_grad()
    def gather_group(self, module: Any) -> None:
        """Stage one layer's six banks to GPU in a single H2D and point each ``.data`` at the slab."""
        group = self._group_of_module.get(id(module))
        if group is None or group.buf is not None or group.home is None:
            return
        buf = self._take_slab(group.total_numel, group.dtype, group.device)
        buf.copy_(group.home, non_blocking=group.home.is_pinned())  # ONE H2D for the whole layer
        group.buf = buf
        group.live = len(group.params)
        for index, param in enumerate(group.params):
            start = group.offsets[index]
            param.data = buf[start : start + group.numels[index]].view(group.shapes[index])
        self._live_groups += 1
        live_bytes = self._live_groups * group.total_numel * buf.element_size()
        if live_bytes > self.staged_high_water_bytes:
            self.staged_high_water_bytes = int(live_bytes)

    @torch.no_grad()
    def release(self, param: torch.nn.Parameter) -> None:
        """Release one bank; free the layer's GPU slab once all six banks are released."""
        group = self._group_of_key.get(id(param))
        if group is None:
            return  # not offloaded (e.g. attention LoRA) -> no-op
        param.data = group.placeholders[id(param)]
        if group.buf is None:
            return
        group.live -= 1
        if group.live <= 0:
            self._return_slab(group.buf)
            group.buf = None
            group.live = 0
            self._live_groups = max(0, self._live_groups - 1)
            for member in group.params:
                member.data = group.placeholders[id(member)]

    def _release_group(self, group: _LayerGroup) -> None:
        for param in group.params:
            param.data = group.placeholders[id(param)]
        if group.buf is not None:
            self._return_slab(group.buf)
            group.buf = None
        group.live = 0

    @torch.no_grad()
    def refresh_home_from_master(self, param: torch.nn.Parameter, master_fp32: torch.Tensor) -> None:
        """After optimizer.step(): write the updated fp32 master into the bf16 home (CPU cast)."""
        group = self._group_of_key.get(id(param))
        if group is None or group.home is None:
            return
        index = group.bank_index[id(param)]
        start = group.offsets[index]
        group.home[start : start + group.numels[index]].copy_(master_fp32.reshape(-1))

    # -- reporting --------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        home_bytes = 0
        pinned_bytes = 0
        for group in self._groups:
            if group.home is None:
                continue
            nbytes = int(group.home.numel() * group.home.element_size())
            home_bytes += nbytes
            if group.home.is_pinned():
                pinned_bytes += nbytes
        return {
            "weight_offload_enabled": True,
            "weight_offload_param_count": len(self.registered_params),
            "weight_offload_group_count": len(self._groups),
            "weight_offload_home_bytes": int(home_bytes),
            "weight_offload_pinned_home_bytes": int(pinned_bytes),
            "weight_offload_staged_high_water_bytes": int(self.staged_high_water_bytes),
            "weight_offload_persistence_threshold_numel": int(self.persistence_threshold_numel),
        }


def install_lora_weight_offload(model: Any, coordinator: LoRAWeightOffloadCoordinator) -> int:
    """Register every ``AsymQwen3Experts`` layer with the coordinator and wire forward hooks.

    Returns the number of offloaded banks (0 means nothing matched; callers should assert > 0).
    """
    from .qwen3_moe import AsymQwen3Experts

    def _gather_hook(module: Any, _inputs: Any) -> None:
        module.gather_lora_weights()

    def _release_hook(module: Any, _inputs: Any, _output: Any) -> None:
        module.release_lora_weights()

    installed = 0
    for module in model.modules():
        if not isinstance(module, AsymQwen3Experts):
            continue
        named_banks = [
            (name, getattr(module, name))
            for name in _BANK_NAMES
            if isinstance(getattr(module, name, None), torch.nn.Parameter)
        ]
        registered = coordinator.register_group(module, named_banks)
        if registered:
            module._weight_offload = coordinator
            module.register_forward_pre_hook(_gather_hook)
            module.register_forward_hook(_release_hook)
            installed += registered
    return installed
