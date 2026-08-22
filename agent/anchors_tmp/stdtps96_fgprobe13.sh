#!/bin/bash
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_lib.sh
echo "=== FG-PROBE13 (core dump + native bt) BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
occupiers_alive() { return 0; }
guard() { return 0; }
ulimit -c unlimited
POL="none|false|false|false|false|false"
v=$(ONE_RANK_GPU=0 run_cell s96q30t2b032c q3-30b-a3b "asym_cpuadamwds|T2B" 32000 "1" "$POL" 1)
echo "FG-PROBE13 canary w/ coredump -> $v" >> "$S"
core=$(find /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM -maxdepth 2 -name "core*" -newermt "-30 minutes" 2>/dev/null | head -1)
[ -z "$core" ] && core=$(find / -maxdepth 3 -name "core" -newermt "-30 minutes" 2>/dev/null | grep -v proc | head -1)
if [ -n "$core" ] && command -v gdb >/dev/null 2>&1; then
  echo "--- native backtrace ($core):" >> "$S"
  gdb -q -batch -ex "bt 20" .venv/bin/python3 "$core" 2>/dev/null | grep -E "^#" | head -20 >> "$S"
else
  echo "no core file or no gdb (core=$core, gdb=$(command -v gdb))" >> "$S"
fi
echo "=== FG-PROBE13 DONE $(date '+%F %H:%M:%S') ===" >> "$S"
