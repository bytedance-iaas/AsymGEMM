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

from collections import OrderedDict
import types
from typing import Any

import torch

_DEFAULT_PERSIST_NUMEL = 1_048_576

_BANK_NAMES = ("gate_lora_A", "gate_lora_B", "up_lora_A", "up_lora_B", "down_lora_A", "down_lora_B")


class _LayerGroup:
    """Staging state for the six LoRA banks of one ``AsymQwen3Experts`` (one decoder layer)."""

    __slots__ = (
        "module_key",
        "group_name",
        "component",
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
        self.group_name: str = ""
        self.component: str = "lora"
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
        self.skipped_groups: list[dict[str, Any]] = []
        # Stage 3 (prefetch) fields, unused in the synchronous core.
        self._stream: torch.cuda.Stream | None = None

    # -- registration -----------------------------------------------------------------

    def is_registered(self, param: torch.nn.Parameter) -> bool:
        return id(param) in self._registered_keys

    def is_module_registered(self, module: Any) -> bool:
        return id(module) in self._group_of_module

    def group_for_module(self, module: Any) -> _LayerGroup | None:
        return self._group_of_module.get(id(module))

    @torch.no_grad()
    def register_group(
        self,
        module: Any,
        named_banks: list[tuple[str, torch.nn.Parameter]],
        *,
        group_name: str = "",
        component: str = "lora",
        force: bool = False,
    ) -> int:
        """Capture the layer's banks into a pinned bf16 home slab, then release them.

        Must be called while the params are still full on CUDA (before any release), so the
        captured home equals the optimizer's freshly-built fp32 master.
        """
        params: list[torch.nn.Parameter] = []
        seen_params: set[int] = set()
        for _, param in named_banks:
            if not isinstance(param, torch.nn.Parameter):
                continue
            key = id(param)
            if key in seen_params:
                continue
            seen_params.add(key)
            params.append(param)
        if not params:
            return 0
        if id(module) in self._group_of_module:
            return len(self._group_of_module[id(module)].params)
        if any(id(param) in self._registered_keys for param in params):
            return 0
        total = sum(int(param.numel()) for param in params)
        if total < self.persistence_threshold_numel and not force:
            self.skipped_groups.append(
                {
                    "group_name": str(group_name or type(module).__name__),
                    "component": str(component or "lora"),
                    "param_count": len(params),
                    "numel": int(total),
                    "bytes": int(total * params[0].element_size()),
                    "reason": "below_persistence_threshold",
                }
            )
            return 0  # keep tiny adapters resident; offloading them is all latency, no benefit

        dtype = params[0].dtype
        device = params[0].device
        for param in params:
            if param.dtype != dtype or param.device != device:
                raise RuntimeError("LoRA weight offload requires all banks in a layer to share dtype/device")

        group = _LayerGroup()
        group.module_key = id(module)
        group.group_name = str(group_name or type(module).__name__)
        group.component = str(component or "lora")
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
            self._live_groups = max(0, self._live_groups - 1)
        group.live = 0

    @torch.no_grad()
    def release_group(self, module: Any) -> None:
        group = self._group_of_module.get(id(module))
        if group is not None:
            self._release_group(group)

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
        home_numel = 0
        pinned_bytes = 0
        group_count_by_component: OrderedDict[str, int] = OrderedDict()
        param_count_by_component: OrderedDict[str, int] = OrderedDict()
        home_bytes_by_component: OrderedDict[str, int] = OrderedDict()
        for group in self._groups:
            if group.home is None:
                continue
            nbytes = int(group.home.numel() * group.home.element_size())
            home_bytes += nbytes
            # Element count survives gather/release (the bank .data is a 0-numel placeholder at rest,
            # but the CPU home slab always holds the real weights), so this is the trustworthy
            # trainable expert-LoRA parameter count for offloaded banks.
            home_numel += int(group.home.numel())
            if group.home.is_pinned():
                pinned_bytes += nbytes
            group_count_by_component[group.component] = group_count_by_component.get(group.component, 0) + 1
            param_count_by_component[group.component] = param_count_by_component.get(group.component, 0) + len(group.params)
            home_bytes_by_component[group.component] = home_bytes_by_component.get(group.component, 0) + nbytes
        skipped_bytes_by_component: OrderedDict[str, int] = OrderedDict()
        skipped_groups_by_component: OrderedDict[str, int] = OrderedDict()
        for item in self.skipped_groups:
            component = str(item.get("component", "lora"))
            skipped_groups_by_component[component] = skipped_groups_by_component.get(component, 0) + 1
            skipped_bytes_by_component[component] = skipped_bytes_by_component.get(component, 0) + int(item.get("bytes", 0))
        return {
            "weight_offload_enabled": True,
            "weight_offload_param_count": len(self.registered_params),
            "weight_offload_param_numel": int(home_numel),
            "weight_offload_group_count": len(self._groups),
            "weight_offload_group_count_by_component": dict(group_count_by_component),
            "weight_offload_param_count_by_component": dict(param_count_by_component),
            "weight_offload_home_bytes": int(home_bytes),
            "weight_offload_home_bytes_by_component": dict(home_bytes_by_component),
            "weight_offload_pinned_home_bytes": int(pinned_bytes),
            "weight_offload_staged_high_water_bytes": int(self.staged_high_water_bytes),
            "weight_offload_persistence_threshold_numel": int(self.persistence_threshold_numel),
            "weight_offload_skipped_group_count": len(self.skipped_groups),
            "weight_offload_skipped_group_count_by_component": dict(skipped_groups_by_component),
            "weight_offload_skipped_bytes_by_component": dict(skipped_bytes_by_component),
            "weight_offload_skipped_groups": list(self.skipped_groups),
        }


def install_lora_weight_offload(model: Any, coordinator: LoRAWeightOffloadCoordinator) -> int:
    """Register trainable Asym LoRA weights with the coordinator and wire staging hooks.

    Returns the number of offloaded banks (0 means nothing matched; callers should assert > 0).
    """
    from .attention_activation_offload import AsymActivationOffloadLoRALinear
    from .llama4_shared_mlp import AsymLlama4SharedMLP
    from .lora import AsymLoRALinear
    from .qwen3_moe import AsymQwen3Experts
    try:
        from .qwen35_shared_mlp import AsymQwen35SharedMLP
    except Exception:  # pragma: no cover - defensive import guard
        AsymQwen35SharedMLP = AsymLlama4SharedMLP  # type: ignore[assignment]

    def _gather_hook(module: Any, _inputs: Any) -> None:
        module.gather_lora_weights()

    def _release_hook(module: Any, _inputs: Any, _output: Any) -> None:
        module.release_lora_weights()

    def _bind_parent_methods(module: Any) -> None:
        if not hasattr(module, "gather_lora_weights"):
            def gather_lora_weights(self: Any) -> None:
                current = getattr(self, "_weight_offload", None)
                if current is not None:
                    current.gather_group(self)

            module.gather_lora_weights = types.MethodType(gather_lora_weights, module)
        if not hasattr(module, "release_lora_weights"):
            def release_lora_weights(self: Any) -> None:
                current = getattr(self, "_weight_offload", None)
                if current is not None:
                    current.release_group(self)

            module.release_lora_weights = types.MethodType(release_lora_weights, module)

    def _mark_child_owned(child: Any, owner: Any) -> None:
        child._weight_offload = coordinator
        object.__setattr__(child, "_weight_offload_owner", owner)
        child._weight_offload_release_after_forward = False

    def _linear_attn_children(module: Any) -> dict[str, Any]:
        return {name: getattr(module, name, None) for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")}

    def _is_linear_attn_parent(name: str, module: Any) -> bool:
        if not name or not name.lower().endswith(".linear_attn"):
            return False
        children = _linear_attn_children(module)
        return all(isinstance(child, (AsymLoRALinear, AsymActivationOffloadLoRALinear)) for child in children.values())

    def _linear_attn_lora_banks(module: Any) -> list[tuple[str, torch.nn.Parameter]]:
        banks: list[tuple[str, torch.nn.Parameter]] = []
        for leaf, child in _linear_attn_children(module).items():
            if isinstance(child, (AsymLoRALinear, AsymActivationOffloadLoRALinear)):
                banks.append((f"{leaf}.lora_A", child.lora_a))
                banks.append((f"{leaf}.lora_B", child.lora_b))
        return banks

    def _component_for_lora_leaf(name: str) -> str:
        lower = name.lower()
        leaf = lower.rsplit(".", 1)[-1]
        if ".linear_attn." in lower or lower.endswith(".linear_attn"):
            return "linear_attention"
        if ".shared_expert" in lower or lower.endswith(".shared_expert_gate") or ".shared_expert_gate." in lower:
            return "shared_experts"
        if leaf in {"q_proj", "k_proj", "v_proj", "o_proj"} or ".self_attn." in lower:
            return "attention"
        if leaf in {"gate_proj", "up_proj", "down_proj"} and (".mlp." in lower or ".feed_forward." in lower):
            return "mlp_dense"
        if leaf in {"lm_head", "output", "output_layer"} or lower.endswith(".lm_head"):
            return "lm_head"
        return "dense_lora"

    installed = 0
    for module in model.modules():
        if not isinstance(module, AsymQwen3Experts):
            continue
        named_banks = [
            (name, getattr(module, name))
            for name in _BANK_NAMES
            if isinstance(getattr(module, name, None), torch.nn.Parameter)
        ]
        registered = coordinator.register_group(
            module,
            named_banks,
            group_name=getattr(module, "profile_prefix", type(module).__name__),
            component="routed_experts",
        )
        if registered:
            module._weight_offload = coordinator
            module.register_forward_pre_hook(_gather_hook)
            module.register_forward_hook(_release_hook)
            installed += registered

    for module in model.modules():
        if not isinstance(module, (AsymLlama4SharedMLP, AsymQwen35SharedMLP)):
            continue
        banks_fn = getattr(module, "_lora_weight_banks", None)
        if not callable(banks_fn):
            continue
        named_banks = banks_fn()
        registered = coordinator.register_group(
            module,
            named_banks,
            group_name=getattr(module, "profile_prefix", type(module).__name__),
            component="shared_experts",
            force=True,
        )
        if registered:
            module._weight_offload = coordinator
            for child in (module.gate_proj, module.up_proj, module.down_proj):
                if isinstance(child, AsymLoRALinear):
                    _mark_child_owned(child, module)
            module.register_forward_pre_hook(_gather_hook)
            module.register_forward_hook(_release_hook)
            installed += registered

    for name, module in model.named_modules():
        if not _is_linear_attn_parent(name, module):
            continue
        named_banks = _linear_attn_lora_banks(module)
        registered = coordinator.register_group(
            module,
            named_banks,
            group_name=name,
            component="linear_attention",
            force=True,
        )
        if registered:
            module._weight_offload = coordinator
            _bind_parent_methods(module)
            for child in _linear_attn_children(module).values():
                if isinstance(child, (AsymLoRALinear, AsymActivationOffloadLoRALinear)):
                    _mark_child_owned(child, module)
            module.register_forward_pre_hook(_gather_hook)
            module.register_forward_hook(_release_hook)
            installed += registered

    for name, module in model.named_modules():
        if not isinstance(module, (AsymLoRALinear, AsymActivationOffloadLoRALinear)):
            continue
        if getattr(module, "_weight_offload_owner", module) is not module:
            continue
        named_banks = [("lora_A", module.lora_a), ("lora_B", module.lora_b)]
        component = _component_for_lora_leaf(name)
        registered = coordinator.register_group(
            module,
            named_banks,
            group_name=name,
            component=component,
            force=component in {"attention", "shared_experts", "linear_attention", "lm_head", "mlp_dense"},
        )
        if registered:
            module._weight_offload = coordinator
            module.register_forward_pre_hook(_gather_hook)
            module.register_forward_hook(_release_hook)
            installed += registered
    return installed
