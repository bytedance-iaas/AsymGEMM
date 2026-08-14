"""fix_dynamic_ep.md D0b: single-process operand-placement isolation for the
ep_steal kernel. Runs the union kernel with side=0 and a synthetic 2-rank
work list where the PEER section is empty-of-steals (n_own = all segments),
so a_peer/d_peer are TOUCHED by the launch path but no cross-rank protocol is
needed. Placements tried per operand: pinned-host (control) vs local-CUDA.
Verdict per combo: RUNS+bitwise / RUNS+corrupt / IMA. This tells us whether
the TMA descriptors are sysmem-typed per-operand (kernel-variant fix) or the
whole peer path faults on device memory (copy-based fallback).
"""
from __future__ import annotations
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

os.environ.setdefault("DG_EP_QUEUE_GRID_PCT", "100")

import torch  # noqa: E402
import asym_gemm  # noqa: E402

E, N, K = 32, 768, 2048
PER = 256  # rows per segment (BLOCK_M-aligned x2)


def main() -> int:
    combo_sel = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    dev = torch.device("cuda", 0)
    torch.manual_seed(0)
    m = PER * E
    a = torch.randn(m, K, device=dev, dtype=torch.bfloat16)
    b = (torch.randn(E, N, K, dtype=torch.bfloat16) * 0.02).pin_memory()
    segs = [(e, e * PER, (e + 1) * PER) for e in range(E)]
    pairs = torch.tensor([[lo, hi] for _, lo, hi in segs], dtype=torch.int32, device=dev).reshape(-1)
    ids = torch.tensor([e for e, _, _ in segs] + [-1], dtype=torch.int32, device=dev)

    d_ref = torch.empty(m, N, device=dev, dtype=torch.bfloat16)
    asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(a, b, d_ref, pairs, ids, E + 1, "nk", False)
    torch.cuda.synchronize()

    m_peer = PER * 2  # small fake peer section
    ctr_host = torch.zeros(8, dtype=torch.int32).pin_memory()

    def mk(where, rows, cols):
        t = torch.zeros(rows, cols, dtype=torch.bfloat16)
        return t.cuda() if where == "cuda" else t.pin_memory()

    combos = [
        ("pinned/pinned (control)", "pin", "pin"),
        ("a_peer=cuda, d_peer=pin", "cuda", "pin"),
        ("a_peer=pin,  d_peer=cuda", "pin", "cuda"),
        ("both cuda", "cuda", "cuda"),
    ]
    for ci, (name, wa, wd) in enumerate(combos):
        if combo_sel >= 0 and ci != combo_sel:
            continue
        a_peer = mk("cuda" if wa == "cuda" else "pin", m_peer, K)
        d_peer = mk("cuda" if wd == "cuda" else "pin", m_peer, N)
        d = torch.empty(m, N, device=dev, dtype=torch.bfloat16)
        ctr_host.zero_()
        try:
            asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous_ep_steal(
                a, b, d, a_peer, d_peer, pairs, ids, E + 1, ctr_host, 0, E, "nk", False)
            torch.cuda.synchronize()
            bit = torch.equal(d, d_ref)
            print(f"[{name}] RUNS bitwise={bit}", flush=True)
        except Exception as exc:  # noqa: BLE001
            torch.cuda.synchronize() if False else None
            print(f"[{name}] FAULT {type(exc).__name__}: {str(exc)[:110]}", flush=True)
            return 1 if name.startswith("pinned/pinned") else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
