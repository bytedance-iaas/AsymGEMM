#!/bin/bash
cd /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM
for gpu in 0 1 2 3; do
  for venv in /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/.venv /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/.venv; do
    out=$(CUDA_VISIBLE_DEVICES=$gpu timeout 60 $venv/bin/python -c "import torch; torch.zeros(1, device='cuda:0'); print('OK')" 2>&1 | tail -1)
    echo "gpu$gpu $(basename $(dirname $(dirname $venv)))/$(basename $venv): ${out:0:60}"
  done
done
