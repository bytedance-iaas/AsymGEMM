#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# setup_aioenv.sh — provision the libaio sidecar that DeepSpeed NVMe offload needs.
# =============================================================================
# DeepSpeed NVMe offload (the *opnvme / *panvme backends) requires the `async_io`
# op, which in turn needs libaio (libaio.h + libaio.so). On this box we are not
# root and `apt install libaio-dev` is impossible, so we materialize libaio into a
# gitignored sidecar dir ${ROOT}/.aioenv/{include,lib}. run_lf_lora_sft.sh detects
# ${AIO_HOME}/lib and exports CPATH/LIBRARY_PATH/LD_LIBRARY_PATH/CFLAGS/LDFLAGS so
# DeepSpeed can JIT-build async_io and load libaio at runtime.
#
# Idempotent: re-running is a no-op once .aioenv is populated (use --force to refresh).
# Run once per machine/container image (the repo lives on shared storage, so the
# sidecar is normally created once and reused, exactly like .venv).
#
# Usage:
#   scripts/lf/setup_aioenv.sh            # create if missing, then verify
#   scripts/lf/setup_aioenv.sh --force    # re-download and rebuild the sidecar
#   scripts/lf/setup_aioenv.sh --no-verify
# =============================================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}        # third_party/AsymGEMM
AIO_HOME=${AIO_HOME:-${ROOT}/.aioenv}
ENV_PYTHON=${ENV_PYTHON:-${ROOT}/.venv/bin/python}
# Scratch area for downloaded/extracted .debs. Override AIO_WORK_DIR to use a fast local NVMe mount.
WORK_BASE=${AIO_WORK_DIR:-${TMPDIR:-/tmp}/asym_libaio_build}

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

log() { printf '[setup_aioenv] %s\n' "$*"; }

if [[ -e "${AIO_HOME}/lib/libaio.so" && "${FORCE}" != "true" ]]; then
  log "sidecar already present at ${AIO_HOME} (use --force to refresh)"
else
  command -v dpkg-deb >/dev/null 2>&1 || { echo "dpkg-deb is required to extract the .deb packages" >&2; exit 1; }
  command -v curl     >/dev/null 2>&1 || { echo "curl is required to download the .deb packages" >&2; exit 1; }

  ARCH=$(dpkg --print-architecture 2>/dev/null || echo arm64)
  . /etc/os-release 2>/dev/null || true
  CODENAME=${VERSION_CODENAME:-noble}

  # arm64/ppc64el/riscv64/s390x live on ports.ubuntu.com; amd64/i386 on archive.ubuntu.com.
  case "${ARCH}" in
    amd64|i386) MIRROR="http://archive.ubuntu.com/ubuntu" ;;
    *)          MIRROR="http://ports.ubuntu.com/ubuntu-ports" ;;
  esac
  POOL="${MIRROR}/pool/main/liba/libaio"

  # noble (24.04+) renamed the runtime lib libaio1 -> libaio1t64 (64-bit time_t transition).
  case "${CODENAME}" in
    noble|oracular|plucky|questing) RUNTIME_PKG_GLOB='libaio1t64' ; VER='0.3.113-6build1.1' ;;
    *)                              RUNTIME_PKG_GLOB='libaio1'     ; VER='0.3.112-13build1' ;;
  esac

  WORK="${WORK_BASE}"
  mkdir -p "${WORK}/debs" 2>/dev/null || WORK=$(mktemp -d)
  mkdir -p "${WORK}/debs" "${WORK}/root"
  rm -rf "${WORK}/root"; mkdir -p "${WORK}/root"

  log "arch=${ARCH} codename=${CODENAME} runtime=${RUNTIME_PKG_GLOB} ver=${VER}"
  for pkg in "${RUNTIME_PKG_GLOB}_${VER}_${ARCH}.deb" "libaio-dev_${VER}_${ARCH}.deb"; do
    log "downloading ${pkg}"
    curl -fsSL -o "${WORK}/debs/${pkg}" "${POOL}/${pkg}" \
      || { echo "failed to download ${POOL}/${pkg} — check codename/version/arch" >&2; exit 1; }
    dpkg-deb -x "${WORK}/debs/${pkg}" "${WORK}/root"
  done

  LIBSRC="${WORK}/root/usr/lib/${ARCH}-linux-gnu"
  [[ -d "${LIBSRC}" ]] || LIBSRC=$(dirname "$(find "${WORK}/root" -name 'libaio.so.*' -print -quit)")
  HDR=$(find "${WORK}/root" -name 'libaio.h' -print -quit)
  REAL=$(find "${LIBSRC}" -name 'libaio.so.*.*' -print -quit)      # e.g. libaio.so.1t64.0.2
  [[ -n "${HDR}" && -n "${REAL}" ]] || { echo "could not locate libaio.h / libaio.so.* in extracted debs" >&2; exit 1; }

  mkdir -p "${AIO_HOME}/include" "${AIO_HOME}/lib"
  cp -f "${HDR}" "${AIO_HOME}/include/"
  cp -f "${REAL}" "${AIO_HOME}/lib/"
  cp -f "${LIBSRC}"/libaio.a "${AIO_HOME}/lib/" 2>/dev/null || true
  realname=$(basename "${REAL}")
  ( cd "${AIO_HOME}/lib"
    ln -sf "${realname}" libaio.so            # linker name for -laio at build time
    ln -sf "${realname}" libaio.so.1          # classic soname (safety / dlopen by old name)
    soname="${realname%.*.*}"                 # strip trailing .X.Y -> SONAME (libaio.so.1 or libaio.so.1t64)
    [[ "${soname}" != "${realname}" ]] && ln -sf "${realname}" "${soname}" )
  log "populated ${AIO_HOME}: $(basename "${HDR}") + ${realname} (+ symlinks)"
fi

if [[ "${VERIFY}" == "true" ]]; then
  [[ -x "${ENV_PYTHON}" ]] || { log "skip verify: no env python at ${ENV_PYTHON}"; exit 0; }
  log "verifying DeepSpeed async_io builds against the sidecar ..."
  export CPATH="${AIO_HOME}/include${CPATH:+:${CPATH}}"
  export LIBRARY_PATH="${AIO_HOME}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${AIO_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export CFLAGS="-I${AIO_HOME}/include ${CFLAGS:-}"
  export LDFLAGS="-L${AIO_HOME}/lib ${LDFLAGS:-}"
  "${ENV_PYTHON}" - <<'PY'
from deepspeed.ops.op_builder import AsyncIOBuilder
b = AsyncIOBuilder()
assert b.is_compatible(verbose=True), "async_io still not compatible — libaio not found"
b.load()
print("[setup_aioenv] async_io is_compatible=True and builds/loads OK")
PY
  log "OK — *opnvme / *panvme DeepSpeed backends are ready to use."
fi
