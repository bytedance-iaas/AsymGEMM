#!/bin/bash
# stdtps_render.sh — re-render both TP figure families + print MAIN_RUNGS state.
cd /workspace/env/figures
PY=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/.venv/bin/python
$PY plot_tp_vs_seq.py    > /tmp/r1.log 2>&1; rc1=$?
$PY plot_tp_vs_seq_2r.py > /tmp/r2.log 2>&1; rc2=$?
echo "== 1r rc=$rc1 =="; grep -E "WARNING|Traceback|Error" /tmp/r1.log | sort -u; grep -c "^wrote" /tmp/r1.log
echo "== 2r rc=$rc2 =="; grep -E "WARNING|Traceback|Error" /tmp/r2.log | sort -u; grep -c "^wrote" /tmp/r2.log
$PY - <<'PYEOF'
import importlib.util, sys
for name in ("plot_tp_vs_seq", "plot_tp_vs_seq_2r"):
    spec = importlib.util.spec_from_file_location(name, f"/workspace/env/figures/{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    print(f"--- {name}: main-variant rungs present/6 ---")
    for k, want in m.MAIN_RUNGS.items():
        seqs = m.DATA[k]["seqs"]; alive = [s for i, s in enumerate(seqs) if m.DATA[k]["asym"][i] not in ("OOM", "OOM*")]
        have = [s for s in want if s in alive]
        print(f"{k:16s} {len(have)}/6 have={have}")
PYEOF
