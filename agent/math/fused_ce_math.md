# Qwen3.5 Fused Cross-Entropy (Liger Loss) Math

LM-head + cross-entropy at the top of the net. Above the transformer layers, so
under activation offload these `[M,V]` tensors are the whole remaining peak.

Per `[M,V]` copy: **bf16 = 4.6 GiB, fp32 = 9.3 GiB** (M = B·T = 16384, V ≈ 152k).
Measured liger saving: `46.08 − 4.51 = 41.6`, `50.99 − 12.33 = 38.7` GiB.

## Punchline

**Without Fused CE** — `[M,V]` tensors co-resident at the CE backward:

| tensor  | shape   | dtype | role                   | size       |
|---------|---------|-------|------------------------|------------|
| `Zf`    | `[M,V]` | fp32  | upcast logits          | 9.3 GiB    |
| `P`     | `[M,V]` | fp32  | softmax / log_softmax  | 9.3 GiB    |
| `dZ`    | `[M,V]` | fp32  | logit grad             | 9.3 GiB    |
| `Z`     | `[M,V]` | bf16  | head logits            | 4.6 GiB    |
| **peak**|         |       | coexisting, not summed | **~32–42 GiB** |

**With Fused CE** — no `[M,V]` tensor ever exists:

| tensor  | shape   | role                     | size      |
|---------|---------|--------------------------|-----------|
| `Z_c`   | `[C,V]` | chunk logits, C ≪ M      | ~100s MiB |
| `dX`    | `[M,H]` | hidden grad              | ~10s MiB  |
| `dW_lm` | `[V,H]` | weight grad (if trained) | ~sub-GiB  |
| **peak**|         |                          | **< 1 GiB** |

## Notation

```text
@            = GEMM
softmax, logsumexp  act over vocab axis V, per row
onehot(y_i)  = unit vector, 1 at target y_i
release(...) = listed tensors freed
.float()     = fp32; CE runs in fp32, all else bf16
```

## Shapes

```text
B = batch              M = B * T
T = sequence length    H = hidden_size
V = vocab_size         C = chunk rows, C << M
N = rows with y_i != ignore_index

X      [M,H]   hidden states (LM-head input)
y      [M]     target token ids
W_lm   [V,H]   LM-head weight
Z      [M,V]   logits
P      [M,V]   softmax
dZ     [M,V]   logit grad
dX     [M,H]
dW_lm  [V,H]
L              scalar loss
```

## Without Fused CE

```text
# forward
Z  = X @ W_lm.T                          # [M,V] bf16
Zf = Z.float()                           # [M,V] fp32
release(Z)
L  = (1/N) * sum_i ( logsumexp(Zf_i) - Zf_i[y_i] )

# backward
P     = softmax(Zf)                      # [M,V] fp32   <- Zf + P live = peak
release(Zf)
dZ    = (1/N) * (P - onehot(y))          # [M,V] fp32 -> bf16 for dX
release(P)
dX    = dZ @ W_lm                        # [M,H]
dW_lm = dZ.T @ X                         # [V,H]
release(dZ)
```

Floor = 2 fp32 `[M,V]` (Zf + P at softmax) = 18.6 GiB; stays O(M·V).

## With Fused CE

```text
L = 0;  dW_lm = 0
for chunk c, rows R_c, |R_c| = C:
    X_c  = X[R_c]                        # [C,H]
    Z_c  = X_c @ W_lm.T                  # [C,V]
    Zf_c = Z_c.float()                   # [C,V] fp32
    L   += sum_i ( logsumexp(Zf_c_i) - Zf_c_i[y_c_i] )
    P_c  = softmax(Zf_c)                 # [C,V]
    dZ_c = (1/N) * (P_c - onehot(y_c))   # [C,V]
    dX[R_c] = dZ_c @ W_lm                # [C,H]
    dW_lm  += dZ_c.T @ X_c               # [V,H]
    release(Z_c, Zf_c, P_c, dZ_c)
L = L / N
```

Peak = O(C·V) + O(M·H) + O(V·H); the `[M,V]` term never forms.

## Memory

| case             | tensors                  | HBM       |
|------------------|--------------------------|-----------|
| non-fused floor  | 2 fp32 `[M,V]`           | 18.6 GiB  |
| non-fused actual | 3–4 fp32 + bf16 logits   | 32–42 GiB |
| fused            | 1 `[C,V]` tile (C ≪ M)   | sub-GiB   |
| measured saving  | `46.08 − 4.51`           | 41.6 GiB  |
| measured saving  | `50.99 − 12.33`          | 38.7 GiB  |

Ratio `M_standard / M_fused ~ M / C` (≫ 1). Scales with `M·V` only → grows with
sequence length and vocab.
