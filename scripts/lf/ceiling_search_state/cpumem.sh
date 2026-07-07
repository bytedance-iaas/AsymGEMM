#!/usr/bin/env bash
# CPU-node-only memory view (GB200: excludes GPU HBM NUMA nodes, which
# free/top wrongly count as system RAM). Mirrors ceiling_search.py's
# host_cpu_mem_avail_gib(): avail = MemFree + max(FilePages - Shmem, 0)
# summed over NUMA nodes with a non-empty cpulist.
tot=0; fre=0; avl=0
for node in /sys/devices/system/node/node[0-9]*; do
  [[ -s "$node/cpulist" && -n "$(cat "$node/cpulist")" ]] || continue
  eval "$(awk '{k=$3; sub(":","",k); v=$4}
    k=="MemTotal"{t=v} k=="MemFree"{f=v} k=="FilePages"{p=v} k=="Shmem"{s=v}
    END{printf "t=%d f=%d p=%d s=%d", t, f, p, s}' "$node/meminfo")"
  r=$(( p > s ? p - s : 0 ))
  printf '%s: total=%dGiB used=%dGiB free=%dGiB avail=%dGiB (reclaimable-cache=%dGiB)\n' \
    "$(basename "$node")" $((t/1048576)) $(((t-f-r)/1048576)) $((f/1048576)) $(((f+r)/1048576)) $((r/1048576))
  tot=$((tot+t)); fre=$((fre+f)); avl=$((avl+f+r))
done
printf 'CPU-TOTAL: total=%dGiB used=%dGiB avail=%dGiB\n' $((tot/1048576)) $(((tot-avl)/1048576)) $((avl/1048576))
