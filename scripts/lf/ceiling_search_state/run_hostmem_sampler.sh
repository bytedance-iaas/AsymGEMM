#!/usr/bin/env bash
# Lightweight read-only host-mem sampler for the llama3.3-70b ceiling search.
# Samples CPU-node avail (via cpumem.sh, which excludes HBM on this GB200) every
# 20s and appends to a run-specific CSV. Exits automatically when the driver dies.
# Non-invasive: no GPU touch, no probe launch -- does not violate the one-probe
# concurrency constraint. Stop early with: kill "$(cat sampler.pid)".
set -u
SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER_PID="${1:?usage: run_hostmem_sampler.sh <driver_pid>}"
CSV="${SD}/host_mem_run_llama33_70b.csv"
echo $$ > "${SD}/sampler.pid"
[ -f "${CSV}" ] || echo "epoch,cpu_total_avail_gib,node0_avail_gib,node1_avail_gib" > "${CSV}"
while kill -0 "${DRIVER_PID}" 2>/dev/null; do
  out="$(bash "${SD}/cpumem.sh" 2>/dev/null || true)"
  tot="$(printf '%s\n' "${out}" | sed -n 's/.*CPU-TOTAL:.*avail=\([0-9]*\)GiB.*/\1/p')"
  n0="$(printf '%s\n' "${out}" | sed -n 's/^node0:.*avail=\([0-9]*\)GiB.*/\1/p')"
  n1="$(printf '%s\n' "${out}" | sed -n 's/^node1:.*avail=\([0-9]*\)GiB.*/\1/p')"
  printf '%s,%s,%s,%s\n' "$(date +%s)" "${tot:-NA}" "${n0:-NA}" "${n1:-NA}" >> "${CSV}"
  sleep 20
done
