import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def main():
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    B, H, S, D = 4, 8, 512, 128
    mk = lambda: torch.randn(B, H, S, D, device=dev, dtype=dt, requires_grad=True)
    q, k, v = mk(), mk(), mk()
    sdpa = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)

    o1 = sdpa(q, k, v)
    o1.float().pow(2).sum().backward()
    g1 = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None

    o2 = checkpoint(sdpa, q, k, v, use_reentrant=False)
    o2.float().pow(2).sum().backward()
    g2 = [t.grad.clone() for t in (q, k, v)]

    assert torch.equal(o1, o2), f"output differs: {(o1 - o2).abs().max()}"
    gdiff = max((a - b).abs().max().item() for a, b in zip(g1, g2))
    assert gdiff == 0.0, f"grad differs: {gdiff}"
    print(f"OK  output exact-equal, grad max-diff={gdiff}")


if __name__ == "__main__":
    main()
