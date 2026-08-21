#!/bin/bash
cd /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM
PY=.venv/bin/python
$PY -c "import fitz" 2>/dev/null || $PY -m pip install -q pymupdf 2>&1 | tail -1
$PY - <<'PY'
import fitz, numpy as np
R="/workspace/env/overleaf/[MLSys 26 Sub] Superchip-based LoRA/figures"; O="/workspace/env/figures/out"
def ras(p):
    d=fitz.open(p); pg=d[0]; pix=pg.get_pixmap(dpi=110, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
for f in ["tp_main_combined.pdf","tp2r_main_combined.pdf","tp_combined.pdf","tp2r_combined.pdf"]:
    a=ras(f"{R}/{f}"); b=ras(f"{O}/{f}")
    if a.shape!=b.shape: print(f"{f}: SHAPE DIFFERS {a.shape} vs {b.shape}"); continue
    d=(np.abs(a.astype(int)-b.astype(int))>40)
    H,W=d.shape; rows=5; 
    # per-panel diff fraction (2 cols x 5 rows grid, approx)
    rep=[]
    for r in range(rows):
        for c in range(2):
            sub=d[int(H*(0.06+0.188*r)):int(H*(0.06+0.188*(r+1))), int(W*c/2):int(W*(c+1)/2)]
            rep.append(f"r{r}c{c}={sub.mean()*100:.2f}%")
    print(f"{f}: diff px {d.mean()*100:.3f}% | "+" ".join(rep))
PY
