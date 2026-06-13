#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
QWEN3 = ROOT / "asym_gemm" / "training" / "qwen3_moe.py"

RETIRED = {
    "_activation_offload_lora_a_pair_forward": "grouped_lora_a_pair_forward_cpu_left",
    "_activation_offload_lora_a_forward": "grouped_lora_a_forward_cpu_left",
    "_activation_offload_lora_a_grad": "grouped_lora_a_grad_cpu_right",
    "_activation_offload_cpu_lora_b_grad": "_grouped_lora_cuda_view + _grouped_lora_weight_grads_torch",
}

REQUIRED = {
    "grouped-forward": [
        "grouped_lora_a_pair_forward_cpu_left",
        "grouped_lora_a_forward_cpu_left",
    ],
    "grouped-lora-b": ["_grouped_lora_cuda_view", "_grouped_lora_weight_grads_torch"],
    "grouped-da": [
        "grouped_lora_a_pair_grad_cpu_right",
        "grouped_lora_a_grad_cpu_right",
    ],
    "final": [
        "grouped_lora_a_pair_forward_cpu_left",
        "grouped_lora_a_forward_cpu_left",
        "_grouped_lora_cuda_view",
        "_grouped_lora_weight_grads_torch",
        "grouped_lora_a_pair_grad_cpu_right",
        "grouped_lora_a_grad_cpu_right",
    ],
}

FORBIDDEN_CALLS = {
    "grouped-forward": [
        "_activation_offload_lora_a_pair_forward",
        "_activation_offload_lora_a_forward",
    ],
    "grouped-lora-b": ["_activation_offload_cpu_lora_b_grad"],
    "grouped-da": ["_activation_offload_lora_a_grad"],
    "final": list(RETIRED),
}

SLOW_CALLS = {"_dispatch_nt", "matmul", "copy_", "contiguous"}


def _call_name(node: ast.Call) -> str | None:
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return None


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _calls_in(node: ast.AST) -> list[str]:
    names: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _call_name(child)
            if name is not None:
                names.append(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require", choices=["fail-closed", "grouped-forward", "grouped-lora-b", "grouped-da", "final"], required=True)
    args = parser.parse_args()

    tree = ast.parse(QWEN3.read_text())
    errors: list[str] = []

    for retired, replacement in RETIRED.items():
        fn = _find_function(tree, retired)
        if fn is not None:
            errors.append(f"{retired} must be deleted; replacement is {replacement}")
        if fn is not None:
            slow = sorted(set(_calls_in(fn)) & SLOW_CALLS)
            if slow:
                errors.append(f"{retired} still contains slow calls: {', '.join(slow)}")

    # More precise: inspect all qwen3 module calls for forbidden retired helper
    # calls, since nested autograd methods are also named forward/backward.
    all_calls = _calls_in(tree)

    if args.require != "fail-closed":
        for name in FORBIDDEN_CALLS.get(args.require, []):
            if name in all_calls:
                errors.append(f"retired helper call remains reachable in module: {name}")
        for name in REQUIRED.get(args.require, []):
            if name not in all_calls:
                errors.append(f"required grouped helper call missing: {name}")

    if args.require == "fail-closed" and "require_expert_activation_offload_kernels" not in all_calls:
        errors.append("missing require_expert_activation_offload_kernels fail-closed call")

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"expact structure check passed: {args.require}")


if __name__ == "__main__":
    main()
