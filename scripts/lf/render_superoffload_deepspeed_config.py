#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_BASE = (
    "/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory/"
    "examples/deepspeed/ds_z3_offload_config.json"
)


def _dict_at(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.setdefault(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a JSON object")
    return value


def render_config(base: Path, cpuadam_cores_perc: float) -> dict[str, Any]:
    if not 0.0 <= cpuadam_cores_perc <= 1.0:
        raise ValueError("--cpuadam-cores-perc must be between 0 and 1")

    with base.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError("base DeepSpeed config must be a JSON object")

    zero = _dict_at(config, "zero_optimization")
    zero["stage"] = 3

    offload_optimizer = _dict_at(zero, "offload_optimizer")
    offload_optimizer["device"] = "cpu"
    offload_optimizer["pin_memory"] = True
    offload_optimizer["super_offload"] = True
    offload_optimizer["cpuadam_cores_perc"] = float(cpuadam_cores_perc)

    offload_param = _dict_at(zero, "offload_param")
    offload_param["device"] = "cpu"
    offload_param["pin_memory"] = True

    return config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a ZeRO-3 CPU-offload config with SuperOffload enabled."
    )
    parser.add_argument("--base", default=DEFAULT_BASE, help="Base ds_z3_offload_config.json")
    parser.add_argument("--output", required=True, help="Destination SuperOffload config path")
    parser.add_argument(
        "--cpuadam-cores-perc",
        type=float,
        default=0.8,
        help="Fraction of CPU cores used by SuperOffload CPUAdam.",
    )
    args = parser.parse_args()

    rendered = render_config(Path(args.base), args.cpuadam_cores_perc)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(rendered, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
