#!/usr/bin/env bash
# Node-level unevictable-memory sampler (rebuilt 2026-07-25, same columns as the
# campaign traces): every 2s append
#   epoch  anon  shmem  unevict  memfree  file  avail   (GiB, NUMA nodes 0+1)
# unevict = AnonPages+Shmem (the capacity metric); avail = MemFree+(FilePages-Shmem).
# Usage: mem_sampler.sh OUT_TSV   (runs until killed; kill by args-matched pkill)
set -u
OUT="$1"
echo -e "#epoch\tanon\tshmem\tunevict\tmemfree\tfile\tavail" >> "$OUT"
while :; do
  anon=0; shmem=0; memfree=0; file=0
  for n in 0 1; do
    # node meminfo rows are "Node N <Key>: <val> kB" — key is field 3
    while read -r _ _ key val _; do
      case "$key" in
        AnonPages:) anon=$((anon+val));;
        Shmem:) shmem=$((shmem+val));;
        MemFree:) memfree=$((memfree+val));;
        FilePages:) file=$((file+val));;
      esac
    done < "/sys/devices/system/node/node$n/meminfo"
  done
  unevict=$((anon+shmem)); avail=$((memfree+file-shmem))
  awk -v t="$(date +%s)" -v a="$anon" -v s="$shmem" -v u="$unevict" -v m="$memfree" -v f="$file" -v v="$avail" \
    'BEGIN{printf "%s\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\n", t, a/1048576, s/1048576, u/1048576, m/1048576, f/1048576, v/1048576}' >> "$OUT"
  sleep 2
done
