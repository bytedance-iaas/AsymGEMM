#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# Seq-ceiling search front-end. Edit CONFIGS below (priors + templates), then:
#   bash scripts/lf/ceiling_search.sh                      # search every active row
#   bash scripts/lf/ceiling_search.sh --dry-run            # print plan + RUNS rows, no runs
#   bash scripts/lf/ceiling_search.sh --only NAME [NAME..] # subset by auto-derived name
#   bash scripts/lf/ceiling_search.sh --single NAME SEQ OHBM   # one probe (classifier check)
# Everything after the script name is passed through to ceiling_search.py.
#
# CONFIGS row format:   seq0 : ohbm0 : RUNS-row-template[ : extra-json-fields]
#   - template must contain {seq}; put {ohbm} where the recompute token takes the
#     -ohbm<N> suffix (Unsloth-GC modes: unsloth / unsloth-off / recomp-off-*).
#     norecomp/recomp have no ohbm knob: omit {ohbm} and the search is seq-only.
#   - name is auto-derived as <model>__<backend>__<recompute-base> (shown by --dry-run).
#   - extra-json-fields: optional raw JSON fragment merged into the row to override
#     any per-config knob, e.g.:  "probe_steps":2,"seq_max":200000,"name":"custom"
# =============================================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
STATE_DIR=${STATE_DIR:-${SCRIPT_DIR}/ceiling_search_state}

# Search knobs applied to every row (per-row override via the extra-json field).
SEQ_STEP=${SEQ_STEP:-4000}                 # first gallop stride (doubles each hit)
SEQ_RESOLUTION=${SEQ_RESOLUTION:-1000}     # final ceiling granularity
SEQ_MIN=${SEQ_MIN:-4000}
SEQ_MAX=${SEQ_MAX:-300000}
OHBM_LADDER=${OHBM_LADDER:-0,8,4,3,2,1}    # searched by HBM share: 0 < 1/8 < 1/4 < 1/3 < 1/2 < 1
PROBE_STEPS=${PROBE_STEPS:-2}              # MAX_STEPS for probes (warmup 1 + 2 measured = 3 total steps)
CONFIRM_STEPS=${CONFIRM_STEPS:-3}          # MAX_STEPS for the confirm re-run of the winner;
                                           # raise (e.g. 8) to also catch late host-RAM creep
WARMUP_STEPS=${WARMUP_STEPS:-1}
PROBE_TIMEOUT_S=${PROBE_TIMEOUT_S:-5400}
CONFIRM_TIMEOUT_S=${CONFIRM_TIMEOUT_S:-${PROBE_TIMEOUT_S}}
MAX_PROBES=${MAX_PROBES:-40}               # live-run budget per config
MAX_CONFIRM_ATTEMPTS=${MAX_CONFIRM_ATTEMPTS:-10}

# run_lf host-mem watchdog (soft C-OOM before the kernel OOM killer). Injected
# into every row's env{} so it is part of the config FINGERPRINT: the floor
# moves the C-OOM boundary, so changing it must invalidate old ledger entries.
# Floor is integer GB; poll accepts fractional seconds.
WATCHDOG_FLOOR_GB=${WATCHDOG_FLOOR_GB:-35}
WATCHDOG_POLL_S=${WATCHDOG_POLL_S:-0.05}

# Driver safety knobs.
GPU_MAX_USED_MIB=${GPU_MAX_USED_MIB:-6000}   # preflight: GPUs must be this empty
MIN_RAM_AVAIL_GIB=${MIN_RAM_AVAIL_GIB:-64}   # preflight: MemAvailable floor
MIN_DISK_FREE_GIB=${MIN_DISK_FREE_GIB:-50}   # preflight: free disk floor (repo fs)
PREFLIGHT_TIMEOUT_S=${PREFLIGHT_TIMEOUT_S:-900}
SETTLE_S=${SETTLE_S:-20}                     # pause after a failed probe

# CONFIGS: seq0 : ohbm0 : template. Priors seeded from the manual C-OOM/G-OOM
# annotations in profile_lora_lf_test_both.sh (boundary minus ~3k). C-OOM priors
# start at ohbm0 so the inner search shows how much ohbm buys; G-OOM-bound
# configs stay at ohbm0 anyway (more HBM share never helps a G-OOM).
CONFIGS=(
  # ---- asym_cpuadamwds | recomp-off-full-fg (dense ker000, routed MoE ker101) ----
  "50000 : 0 : q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"           # C-OOM 53k
  "128000 : 0 : q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker101-ceil0000-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"    # C-OOM 132k
  "32000 : 0 : llama3.3-70b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"   # C-OOM 33k
  # "13000 : 0 : llama4-scout|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"   # G-OOM 15k
  # "50000 : 0 : q2.5-32b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"       # C-OOM 53k
  # "30000 : 0 : q2.5-72b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"       # C-OOM 33k

  # ---- superoffload_mem | unsloth-off ----
  # "128000 : 0 : q3-30b-a3b|1 ; superoffload_mem|unsloth-off-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"   # C-OOM 132k
  # "50000 : 0 : q3-32b|1 ; superoffload_mem|unsloth-off-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"        # C-OOM 53k
  # "30000 : 0 : llama3.3-70b|1 ; superoffload_mem|unsloth-off-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"  # C-OOM 33k
  # "13000 : 0 : llama4-scout|1 ; superoffload_mem|unsloth-off-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"  # G-OOM 15k
  # "50000 : 0 : q2.5-32b|1 ; superoffload_mem|unsloth-off-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"      # C-OOM 53k
  # "30000 : 0 : q2.5-72b|1 ; superoffload_mem|unsloth-off-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"      # C-OOM 33k

  # ---- superoffload_mem | unsloth (HBM-bound: expect ohbm to stay 0) ----
  # "78000 : 0 : q3-30b-a3b|1 ; superoffload_mem|unsloth-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"        # G-OOM 81k
  # "47000 : 0 : q3-32b|1 ; superoffload_mem|unsloth-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"            # G-OOM 50k
  # "43000 : 0 : llama3.3-70b|1 ; superoffload_mem|unsloth-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"      # G-OOM 46k
  # "8000 : 0 : llama4-scout|1 ; superoffload_mem|unsloth-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"       # G-OOM 10k
  # "48000 : 0 : q2.5-32b|1 ; superoffload_mem|unsloth-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"          # G-OOM 51k
  # "38000 : 0 : q2.5-72b|1 ; superoffload_mem|unsloth-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"          # G-OOM 41k

  # ---- superoffload_mem | recomp (no ohbm knob: seq-only search) ----
  # "43000 : 0 : q3-30b-a3b|1 ; superoffload_mem|recomp|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"                    # G-OOM 46k
  # "18000 : 0 : q3-32b|1 ; superoffload_mem|recomp|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"                        # G-OOM 21k
  # "10000 : 0 : llama3.3-70b|1 ; superoffload_mem|recomp|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"                  # G-OOM 13k
  # "6000 : 0 : llama4-scout|1 ; superoffload_mem|recomp|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"                   # G-OOM 9k
  # "19000 : 0 : q2.5-32b|1 ; superoffload_mem|recomp|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"                      # G-OOM 22k
  # "10000 : 0 : q2.5-72b|1 ; superoffload_mem|recomp|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"                      # G-OOM 13k
)

# =============================================================================
# Generate the JSONL config list and invoke the python driver.
# =============================================================================
die() { echo "error: $*" >&2; exit 2; }

for _nv in SEQ_STEP SEQ_RESOLUTION SEQ_MIN SEQ_MAX PROBE_STEPS CONFIRM_STEPS WARMUP_STEPS \
           PROBE_TIMEOUT_S CONFIRM_TIMEOUT_S MAX_PROBES MAX_CONFIRM_ATTEMPTS \
           GPU_MAX_USED_MIB MIN_RAM_AVAIL_GIB MIN_DISK_FREE_GIB PREFLIGHT_TIMEOUT_S SETTLE_S; do
  # no leading zeros: "04000" would be emitted into JSON and fail to parse
  [[ "${!_nv}" =~ ^(0|[1-9][0-9]*)$ ]] || die "${_nv} must be a nonnegative integer without leading zeros, got '${!_nv}'"
done
(( PROBE_STEPS >= 1 )) || die "PROBE_STEPS must be >= 1 (0 measured steps would fake-OK probes)"
(( CONFIRM_STEPS >= PROBE_STEPS )) || die "CONFIRM_STEPS must be >= PROBE_STEPS"
[[ "${WATCHDOG_FLOOR_GB}" =~ ^(0|[1-9][0-9]*)$ ]] || die "WATCHDOG_FLOOR_GB must be an integer GB (run_lf validates ^[0-9]+$), got '${WATCHDOG_FLOOR_GB}'"
(( WATCHDOG_FLOOR_GB >= 1 )) || die "WATCHDOG_FLOOR_GB=0 disables the watchdog; the search relies on its soft C-OOM"
[[ "${WATCHDOG_POLL_S}" =~ ^(0|[1-9][0-9]*)(\.[0-9]+)?$ ]] || die "WATCHDOG_POLL_S must be seconds (fractions ok), got '${WATCHDOG_POLL_S}'"
OHBM_LADDER="${OHBM_LADDER// /}"
[[ "${OHBM_LADDER}" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || die "OHBM_LADDER must be comma-separated ints without leading zeros, got '${OHBM_LADDER}'"

(( ${#CONFIGS[@]} > 0 )) || die "CONFIGS is empty"
mkdir -p "${STATE_DIR}"
GEN="${STATE_DIR}/configs.generated.jsonl"
: > "${GEN}"

ladder_json="[${OHBM_LADDER}]"
for row in "${CONFIGS[@]}"; do
  [[ "${row}" == *" : "* ]] || die "CONFIGS row must be 'seq0 : ohbm0 : template', got '${row}'"
  seq0="${row%% : *}"; rest="${row#* : }"
  ohbm0="${rest%% : *}"; rest="${rest#* : }"
  template="${rest%% : *}"
  extra=""
  if [[ "${rest}" == *" : "* ]]; then
    extra="${rest#* : }"
    # the 4th field must be a JSON fragment like "probe_steps":2 -- anything
    # else means a stray ' : ' inside the template mis-split the row
    [[ "${extra}" == \"* ]] || die "4th field must be a JSON fragment starting with a quoted key, got '${extra}' (no ' : ' allowed inside templates)"
  fi
  seq0="${seq0//[[:space:]]/}"; ohbm0="${ohbm0//[[:space:]]/}"
  [[ "${seq0}" =~ ^[0-9]+$ && "${ohbm0}" =~ ^[0-9]+$ ]] || die "bad seq0/ohbm0 in '${row}'"
  [[ "${template}" == *"{seq}"* ]] || die "template missing {seq}: '${template}'"
  [[ "${template}" == *" ; "* ]] || die "template must use ' ; ' field separators: '${template}'"
  if [[ "${template}" == *'"'* || "${template}" == *'\'* ]]; then
    die "template must not contain quotes/backslashes: '${template}'"
  fi

  # name: <model>__<backend>__<recompute minus the -ohbm{ohbm} suffix>
  model="${template%%|*}"
  f2="${template#* ; }"; f2="${f2%% ; *}"
  backend="${f2%%|*}"
  recompute="${f2#*|}"; recompute="${recompute%%|*}"; recompute="${recompute%-ohbm\{ohbm\}}"
  name="${model// /}__${backend}__${recompute}"
  name="${name//\//_}"  # full model paths (org/model) must not create subdirs

  line="{\"name\": \"${name}\", \"seq0\": ${seq0}, \"ohbm0\": ${ohbm0}"
  line+=", \"template\": \"${template}\""
  line+=", \"seq_step\": ${SEQ_STEP}, \"seq_resolution\": ${SEQ_RESOLUTION}"
  line+=", \"seq_min\": ${SEQ_MIN}, \"seq_max\": ${SEQ_MAX}"
  line+=", \"ohbm_ladder\": ${ladder_json}"
  line+=", \"probe_steps\": ${PROBE_STEPS}, \"confirm_steps\": ${CONFIRM_STEPS}"
  line+=", \"warmup_steps\": ${WARMUP_STEPS}"
  line+=", \"probe_timeout_s\": ${PROBE_TIMEOUT_S}, \"confirm_timeout_s\": ${CONFIRM_TIMEOUT_S}"
  line+=", \"max_probes\": ${MAX_PROBES}, \"max_confirm_attempts\": ${MAX_CONFIRM_ATTEMPTS}"
  # NOTE: an extra-json "env":{...} REPLACES this block (JSON last-key-wins) --
  # re-include the watchdog keys there if you override env per row.
  line+=", \"env\": {\"HOST_MEM_WATCHDOG_FLOOR_GB\": \"${WATCHDOG_FLOOR_GB}\", \"HOST_MEM_WATCHDOG_POLL_SECONDS\": \"${WATCHDOG_POLL_S}\"}"
  [[ -n "${extra}" ]] && line+=", ${extra}"
  line+="}"
  printf '%s\n' "${line}" >> "${GEN}"
  echo "  config: ${name}  (seq0=${seq0}, ohbm0=${ohbm0})"
done

echo "generated ${GEN} ($(wc -l < "${GEN}") configs)"

# Round-trip each row through the wrapper's own validation (DRY_RUN=true only
# prints commands; ~1 s/row, no training). Catches wrapper-semantic template
# mistakes (bad backend/recompute/suffix order) NOW instead of days into the
# run as an UNKNOWN abort. Disable with VALIDATE_ROWS=false.
VALIDATE_ROWS=${VALIDATE_ROWS:-true}
if [[ "${VALIDATE_ROWS}" == "true" ]]; then
  ASYM_ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
  _vlog="${STATE_DIR}/validate_rows.log"
  : > "${_vlog}"
  while IFS= read -r _cfg_line; do
    _name=$(printf '%s' "${_cfg_line}" | "${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.stdin.read())["name"])')
    _row=$(printf '%s' "${_cfg_line}" | "${PYTHON_BIN}" -c 'import json,sys; c=json.loads(sys.stdin.read()); print(c["template"].format(seq=c["seq0"], ohbm=c["ohbm0"]))')
    if ! ( cd "${ASYM_ROOT_DIR}" && \
           SFT_ROOT="$(cd "${ASYM_ROOT_DIR}/../.." && pwd)" RUNS="${_row}" DRY_RUN=true \
           PLOT=false PROFILERS=source RUN_POST=false CONTINUE_ON_ERROR=false \
           COLLECT_EXISTING=false UNSLOTH_GC_OUTER_HBM_EVERY_N=0 RUN_NAME="ceiling_validate" \
           timeout 120 bash "${SCRIPT_DIR}/profile_lora_lf_test_both.sh" ) >> "${_vlog}" 2>&1; then
      echo "---- last lines of ${_vlog}:" >&2
      tail -n 15 "${_vlog}" >&2
      die "row validation failed for '${_name}' (wrapper rejected: RUNS=${_row}); full log: ${_vlog}"
    fi
    echo "  validated: ${_name}"
  done < "${GEN}"
fi
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/ceiling_search.py" "${GEN}" \
  --state-dir "${STATE_DIR}" \
  --gpu-max-used-mib "${GPU_MAX_USED_MIB}" \
  --min-ram-avail-gib "${MIN_RAM_AVAIL_GIB}" \
  --min-disk-free-gib "${MIN_DISK_FREE_GIB}" \
  --preflight-timeout-s "${PREFLIGHT_TIMEOUT_S}" \
  --settle-s "${SETTLE_S}" \
  "$@"
