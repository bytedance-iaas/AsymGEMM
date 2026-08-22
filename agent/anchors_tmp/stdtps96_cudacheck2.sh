#!/bin/bash
cd /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM
echo "start $(date +%H:%M:%S)"
CUDA_VISIBLE_DEVICES=0 timeout 90 .venv/bin/python -u -c "
import time, torch
print('torch imported', flush=True)
t=time.time()
torch.cuda.init()
print('cuda.init ok in', round(time.time()-t,1), 's', flush=True)
x=torch.zeros(1, device='cuda:0')
print('alloc ok', flush=True)
" 2>&1; echo "rc=$? end $(date +%H:%M:%S)"
