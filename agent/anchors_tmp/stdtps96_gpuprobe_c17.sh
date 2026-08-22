#!/bin/bash
# Per-GPU context probe (Session E, c17): tiny alloc+free on each visible
# device, try/except so one wedged device doesn't mask the others.
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
.venv/bin/python - <<'PY'
import torch
n = torch.cuda.device_count()
print("ndev", n)
for i in range(n):
    try:
        with torch.cuda.device(i):
            x = torch.ones(1024, 1024, device=f"cuda:{i}")
            s = float(x.sum().item())
            free, total = torch.cuda.mem_get_info(i)
            del x
            torch.cuda.empty_cache()
        print(f"dev{i} OK sum={s:.0f} free={free/2**30:.1f}G/{total/2**30:.1f}G")
    except Exception as e:
        print(f"dev{i} FAIL {type(e).__name__}: {str(e).splitlines()[0][:120]}")
PY
