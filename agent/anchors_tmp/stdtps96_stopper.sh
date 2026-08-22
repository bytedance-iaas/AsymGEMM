#!/bin/bash
# v3 — offset-scoped: only lines appended AFTER arming count (stale-log trap).
S=/home/kevinni/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/stdtps96_status.log
OFF=$(wc -l < "$S")
newlines() { tail -n +"$((OFF+1))" "$S"; }
for i in $(seq 1 720); do
  if newlines | grep -aqE 'WALL nmx1uou |START nmx1t1_'; then break; fi
  sleep 5
done
if newlines | grep -aq 'START nmx1t1_' && ! newlines | grep -aq 'WALL nmx1uou '; then MODE=late; else MODE=boundary; fi
PIDS=$(ps -ef | grep 'stdtps96_a3_mxC1r.sh' | grep -v grep | awk '{print $2}')
DRV=$(ps -ef | grep -E 'profile_lora_lf_test_source' | grep -v grep | awk '{print $2}' | head -1)
if [ "$MODE" = boundary ] && [ -z "$DRV" ]; then
  for p in $PIDS; do kill -INT $p 2>/dev/null; done; sleep 8
  for p in $PIDS; do [ -d /proc/$p ] && kill -TERM $p 2>/dev/null; done; sleep 5
  LEFT=$(ps -ef | grep 'stdtps96_a3_mxC1r.sh' | grep -v grep | wc -l)
  echo "STOPPER: chain halted at uo boundary (left=$LEFT) $(date +%H:%M)" >> "$S"
  echo "STOPPED-CLEAN"
else
  echo "STOPPER: not safe (mode=$MODE drv=$DRV) $(date +%H:%M)" >> "$S"
  echo "NEEDS-MANUAL mode=$MODE drv=$DRV"
fi
