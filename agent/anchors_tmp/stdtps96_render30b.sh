#!/bin/bash
# one-off: combined-main figures restricted to the 30B panel (Kevin: one panel for now)
cd /workspace/env/figures
PY=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/.venv/bin/python
$PY - <<'PYEOF'
import importlib.util
for name in ("plot_tp_vs_seq_96gb", "plot_tp_vs_seq_2r_96gb"):
    spec = importlib.util.spec_from_file_location(name, f"/workspace/env/figures/{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m._plot_combined("main", keys=["q3-30b-a3b"])
PYEOF
