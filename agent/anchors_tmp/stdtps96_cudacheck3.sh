#!/bin/bash
cd /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM
for g in 0 1 2 3; do
  r=$(CUDA_VISIBLE_DEVICES=$g timeout 60 .venv/bin/python -u -c "import torch; torch.zeros(1, device='cuda:0'); print('ALLOC-OK')" 2>&1 | grep -oE "ALLOC-OK|busy or unavailable|out of memory|Error[A-Za-z]*" | head -1)
  echo "gpu$g: ${r:-empty/hang}"
done
