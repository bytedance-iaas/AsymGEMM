#!/bin/bash
/workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python - <<'PY'
import torch
try:
    t=torch.zeros(1024,device="cuda:0")
    free,total=torch.cuda.mem_get_info(0)
    print("CUDA-OK free",round(free/2**30,1),"GiB")
except Exception as e:
    print("CUDA-FAIL",type(e).__name__,str(e)[:120])
PY
