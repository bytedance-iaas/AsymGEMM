#!/bin/bash
# Host-memory sampler: logs meminfo deltas + top-RSS + train-process VmLck/VmPin
# every 3s; verbose dumps once node available < 900 GB. Runs until killed.
OUT="$1"
while :; do
  avail=$(free -g | awk 'NR==2{print $7}')
  ts=$(date +%H:%M:%S)
  if [ "$avail" -lt 900 ]; then
    {
      echo "=== $ts avail=${avail}G"
      grep -E "MemAvailable|AnonPages|Mlocked|PageTables|Cached|Unevictable|KernelStack|VmallocUsed" /proc/meminfo
      ps -eo pid,rss,comm --sort=-rss | head -6
      tp=$(pgrep -f run_lf_profiled_train | head -1)
      [ -n "$tp" ] && grep -E "VmRSS|VmLck|VmPin|VmPTE" /proc/$tp/status
    } >> "$OUT"
  else
    echo "$ts avail=${avail}G" >> "$OUT"
  fi
  sleep 3
done
