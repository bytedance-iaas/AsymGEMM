#!/bin/bash
cd /workspace/env/figures
PY=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/.venv/bin/python
$PY plot_tp_vs_seq_96gb.py 2>&1 | grep -E "WARNING|wrote tp96_main|Traceback" | head -12
$PY plot_tp_vs_seq_2r_96gb.py 2>&1 | grep -E "WARNING|wrote tp2r96_main|Traceback" | head -12
