#!/usr/bin/env python3
"""I4 step-2 adapter-grad parity: |1 reference vs sTP full-TP dump (branch pieces
reassembled per the dumped stp_param_map), judged against a measured reduction-order
ENVELOPE (Phase-A-vs-|1) per gb200_tp.md I4.

Usage:
  compare_stp_adapter_grads.py REF.pt STP.pt [--envelope ENV.pt] [--static-tol 1e-2]
PASS: every logical adapter grad within max(static_tol, 2 x envelope_max).
"""
from __future__ import annotations

import argparse
import sys

import torch


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-8)).item()


def load(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["grads"], payload.get("stp_param_map")


def reassemble(stp_grads: dict, param_map: dict) -> dict:
    logical: dict[str, torch.Tensor] = {}
    for base, meta in param_map.items():
        kind = meta["kind"]
        branch1 = base.replace(".self_attn.", ".self_attn_stp1.").replace(".mlp.", ".mlp_stp1.")
        for piece in ("lora_A", "lora_B"):
            own = stp_grads.get(f"{base}.{piece}.default.weight")
            other = stp_grads.get(f"{branch1}.{piece}.default.weight")
            key = f"{base}.{piece}.default.weight"
            sliced = (kind == "col" and piece == "lora_B") or (kind == "row" and piece == "lora_A")
            if sliced:
                if own is None or other is None:
                    continue
                dim = 0 if kind == "col" else 1
                logical[key] = torch.cat([own, other], dim=dim)
            elif own is not None:
                logical[key] = own  # replicated piece: mirror already merged into the owner
    return logical


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ref")
    parser.add_argument("stp")
    parser.add_argument("--envelope", default="")
    parser.add_argument("--static-tol", type=float, default=1e-2)
    args = parser.parse_args()

    ref_grads, _ = load(args.ref)
    stp_grads, param_map = load(args.stp)
    if param_map:
        stp_logical = reassemble(stp_grads, param_map)
    else:
        stp_logical = stp_grads

    env_max = 0.0
    if args.envelope:
        env_grads, _ = load(args.envelope)
        for name, g_ref in ref_grads.items():
            g_env = env_grads.get(name)
            if g_env is not None:
                env_max = max(env_max, rel(g_env, g_ref))
        print(f"[cmp] measured envelope max rel-err = {env_max:.3e}")
    bound = max(args.static_tol, 2.0 * env_max)
    print(f"[cmp] bound = {bound:.3e} (static {args.static_tol}, 2x envelope {2*env_max:.3e})")

    worst, worst_name, n_bad, n_cmp = 0.0, "", 0, 0
    missing = []
    for name, g_ref in sorted(ref_grads.items()):
        g_stp = stp_logical.get(name)
        if g_stp is None:
            missing.append(name)
            continue
        if g_stp.shape != g_ref.shape:
            missing.append(f"{name} SHAPE {tuple(g_stp.shape)} vs {tuple(g_ref.shape)}")
            continue
        e = rel(g_stp, g_ref)
        n_cmp += 1
        if e > worst:
            worst, worst_name = e, name
        if e > bound:
            n_bad += 1
            if n_bad <= 8:
                print(f"[cmp] OVER: {name} rel={e:.3e}")
    print(f"[cmp] compared {n_cmp}; missing {len(missing)}; worst {worst:.3e} @ {worst_name}; over-bound {n_bad}")
    for m in missing[:6]:
        print(f"[cmp] missing: {m}")
    ok = n_bad == 0 and not missing and n_cmp > 0
    print("[cmp] PASS" if ok else "[cmp] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
