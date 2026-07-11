#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# setup_numaenv.sh — provision a numactl sidecar so run_lf can NUMA-bind the
# host offload to the Grace CPU nodes instead of spilling onto GPU-HBM nodes.
# =============================================================================
# On GB200 the GPU HBM is coherent and shows up as NUMA nodes. If host (CPU)
# allocations are not membind'd to the Grace CPU nodes, the cpuadam/offload
# spills onto GPU-HBM NUMA nodes — i.e. a GPU silently used as a CPU-RAM
# extension (shows as GPU "memory used" with no compute process). run_lf binds
# with `numactl --membind=0,1 --cpunodebind=0,1` by default, but this
# reprovisioned host has no numactl binary and we are not root. So we
# materialize numactl (+ libnuma) into a gitignored sidecar
# ${ROOT}/.numaenv/{bin,lib}, exactly like .aioenv. run_lf_lora_sft.sh detects
# ${NUMA_HOME}/bin/numactl and uses it (with ${NUMA_HOME}/lib on LD_LIBRARY_PATH).
#
# Idempotent: re-running is a no-op once the sidecar exists (--force to refresh).
# Run once per machine/container image; the repo lives on shared storage so the
# sidecar is created once and reused, like .venv / .aioenv.
#
# Usage:
#   scripts/lf/setup_numaenv.sh            # create if missing, then verify
#   scripts/lf/setup_numaenv.sh --force    # re-download and rebuild
#   scripts/lf/setup_numaenv.sh --no-verify
# =============================================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}          # third_party/AsymGEMM
NUMA_HOME=${NUMA_HOME:-${ROOT}/.numaenv}
WORK_BASE=${NUMA_WORK_DIR:-/scratch_local/user_data/shutian/kevin/cache/deps/numactl}

FORCE=false
VERIFY=true
for arg in "$@"; do
  case "${arg}" in
    --force) FORCE=true ;;
    --no-verify) VERIFY=false ;;
    -h|--help) grep -E '^#( |$)' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown arg: ${arg}" >&2; exit 2 ;;
  esac
done

log() { printf '[setup_numaenv] %s\n' "$*"; }

if [[ -x "${NUMA_HOME}/bin/numactl" && "${FORCE}" != "true" ]]; then
  log "sidecar already present at ${NUMA_HOME} (use --force to refresh)"
else
  command -v dpkg-deb >/dev/null 2>&1 || { echo "dpkg-deb is required to extract the .deb" >&2; exit 1; }
  ARCH=$(dpkg --print-architecture 2>/dev/null || echo arm64)
  WORK="${WORK_BASE}"
  mkdir -p "${WORK}/debs" 2>/dev/null || WORK=$(mktemp -d)
  mkdir -p "${WORK}/debs"; rm -rf "${WORK}/root"; mkdir -p "${WORK}/root"

  # Primary: apt-get download (resolves the right version from configured sources,
  # no sudo). Fallback: curl the .deb from the Ubuntu/Debian pool.
  got=false
  if command -v apt-get >/dev/null 2>&1; then
    ( cd "${WORK}/debs" && apt-get download numactl ) >/dev/null 2>&1 && got=true
  fi
  if [[ "${got}" != true ]]; then
    command -v curl >/dev/null 2>&1 || { echo "need curl or a working apt-get to fetch numactl" >&2; exit 1; }
    . /etc/os-release 2>/dev/null || true
    case "${ARCH}" in amd64|i386) MIRROR="http://archive.ubuntu.com/ubuntu" ;; *) MIRROR="http://ports.ubuntu.com/ubuntu-ports" ;; esac
    VER=${NUMACTL_VER:-2.0.14-3ubuntu2}   # jammy; override NUMACTL_VER for other releases
    log "apt-get download unavailable; curl ${MIRROR} numactl_${VER}_${ARCH}.deb"
    curl -fsSL -o "${WORK}/debs/numactl_${VER}_${ARCH}.deb" \
      "${MIRROR}/pool/main/n/numactl/numactl_${VER}_${ARCH}.deb" \
      || { echo "failed to fetch numactl .deb — set NUMACTL_VER to a valid version" >&2; exit 1; }
  fi

  for d in "${WORK}"/debs/numactl_*.deb; do dpkg-deb -x "${d}" "${WORK}/root"; done
  NBIN=$(find "${WORK}/root" -path '*bin/numactl' -type f -print -quit)
  [[ -n "${NBIN}" ]] || { echo "numactl binary not found in extracted .deb" >&2; exit 1; }

  mkdir -p "${NUMA_HOME}/bin" "${NUMA_HOME}/lib"
  cp -f "${NBIN}" "${NUMA_HOME}/bin/numactl"
  # Bundle libnuma so the sidecar is self-contained even if a future base image lacks it.
  # Prefer a libnuma from the .deb tree; else copy the system one (preserving symlinks).
  mapfile -t LNS < <(find "${WORK}/root" /usr/lib /lib -name 'libnuma.so*' 2>/dev/null)
  for f in "${LNS[@]}"; do cp -af "${f}" "${NUMA_HOME}/lib/" 2>/dev/null || cp -f "${f}" "${NUMA_HOME}/lib/"; done
  log "populated ${NUMA_HOME}: bin/numactl + lib/{$(ls "${NUMA_HOME}/lib" 2>/dev/null | tr '\n' ' ')}"
fi

if [[ "${VERIFY}" == "true" ]]; then
  export LD_LIBRARY_PATH="${NUMA_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  out=$("${NUMA_HOME}/bin/numactl" --membind=0,1 --cpunodebind=0,1 "${NUMA_HOME}/bin/numactl" --show 2>&1) \
    || { echo "[setup_numaenv] verify FAILED: ${out}" >&2; exit 1; }
  echo "${out}" | grep -E 'nodebind|membind' | sed 's/^/[setup_numaenv] /'
  log "OK — numactl binds to CPU nodes 0,1. Set NUMACTL_ENABLE=1 (default) to use it."
fi
