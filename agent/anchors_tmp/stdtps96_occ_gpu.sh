#!/bin/bash
# one occupier inside the container; arg = host GPU id
exec /workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python \
  /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/hbm96_occupy.py
