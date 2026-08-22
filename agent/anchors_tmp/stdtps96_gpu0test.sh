#!/bin/bash
cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
timeout 60 .venv/bin/python -c "
import torch
torch.cuda.init()
free, total = torch.cuda.mem_get_info(0)
x = torch.empty(1024**3, dtype=torch.uint8, device='cuda:0')
print('GPU0-NEW-CONTEXT-OK free', free/2**30)
del x" 2>&1 | tail -2
