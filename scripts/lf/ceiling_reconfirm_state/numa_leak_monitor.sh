#!/usr/bin/env bash
# NUMA GPU-HBM leak monitor for bound cpuadam/superoffload confirm runs.
#
# Grace/GB200 layout: CPU nodes 0,1 (offload belongs here). GPU-HBM NUMA nodes:
#   node2=GPU0  node10=GPU1  node18=GPU2  node26=GPU3
# Training uses GPU 2 (GPU_POOL=2) => node18 legitimately rises to ~G (reserved HBM).
# The idle-GPU nodes 2,10,26 MUST stay near baseline; growth there = host offload
# spilling onto GPU-HBM (numactl binding broken) -- the exact bug being guarded.
#
# Usage: numa_leak_monitor.sh OUT.csv [interval_s] [max_s]
set -u
OUT="${1:?usage: numa_leak_monitor.sh OUT.csv [interval_s] [max_s]}"
INT="${2:-15}"
MAX="${3:-36000}"   # 10h hard safety cap

node_used() { awk '/MemUsed/{printf "%.1f", $4/1048576; f=1} END{if(!f)print "NA"}' \
  "/sys/devices/system/node/node$1/meminfo" 2>/dev/null || echo NA; }

[ -f "$OUT" ] || echo "ts,iso,node0_gib,node1_gib,node2_gib,node10_gib,node18_gib,node26_gib,gpu0_mib,gpu1_mib,gpu2_mib,gpu3_mib" > "$OUT"

end=$(( $(date +%s) + MAX ))
while [ "$(date +%s)" -lt "$end" ]; do
  ts=$(date +%s); iso=$(date -Is)
  n0=$(node_used 0); n1=$(node_used 1); n2=$(node_used 2)
  n10=$(node_used 10); n18=$(node_used 18); n26=$(node_used 26)
  mapfile -t G < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  echo "$ts,$iso,$n0,$n1,$n2,$n10,$n18,$n26,${G[0]:-NA},${G[1]:-NA},${G[2]:-NA},${G[3]:-NA}" >> "$OUT"
  sleep "$INT"
done
