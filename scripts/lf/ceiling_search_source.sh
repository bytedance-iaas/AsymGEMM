#!/usr/bin/env bash
set -Eeuo pipefail

# Seq-ceiling search front-end. Edit CONFIGS below, then:
#   bash scripts/lf/ceiling_search_source.sh                      # search every active row
#   bash scripts/lf/ceiling_search_source.sh --dry-run            # print plan, no runs
#   bash scripts/lf/ceiling_search_source.sh --only NAME [NAME..] # subset by name
#   bash scripts/lf/ceiling_search_source.sh --single NAME SEQ OHBM
# Extra args pass through to ceiling_search.py.
#
# CONFIGS row: seq0 : ohbm0 : RUNS-template[ : extra-json]
#   template needs {seq}; {ohbm} goes where the recompute token takes -ohbm<N>
#   (omit {ohbm} for norecomp/recomp: seq-only search). extra-json overrides any
#   per-config knob, e.g. "probe_steps":2,"seq_max":200000

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}

# source (clean run) | nsys | both (ONE nsys-wrapped run + source artifacts;
# nsys host-RAM overhead biases near-wall C-OOM rungs). Fingerprinted via row
# env{}; namespaces the state dir + artifacts root.
PROFILERS=${PROFILERS:-source}

HOST_TAG=${HOST_TAG:-$(hostname -s)}
STATE_DIR=${STATE_DIR:-${SCRIPT_DIR}/ceiling_search_state_${PROFILERS}_${HOST_TAG}}

# Search knobs applied to every row (per-row override via extra-json).
SEQ_STEP=${SEQ_STEP:-2000}                 # first gallop stride (doubles each hit)
SEQ_RESOLUTION=${SEQ_RESOLUTION:-1000}     # final ceiling granularity
SEQ_MIN=${SEQ_MIN:-4000}
SEQ_MAX=${SEQ_MAX:-300000}
OHBM_LADDER=${OHBM_LADDER:-0,16,8,7,6,5,4,3,2,1}  # by HBM share: 0 < 1/16 < ... < 1
PROBE_STEPS=${PROBE_STEPS:-2}              # probe MAX_STEPS (measured; + warmup)
CONFIRM_STEPS=${CONFIRM_STEPS:-4}          # confirm MAX_STEPS (steady stat)
WARMUP_STEPS=${WARMUP_STEPS:-1}
PROBE_TIMEOUT_S=${PROBE_TIMEOUT_S:-14400}
CONFIRM_TIMEOUT_S=${CONFIRM_TIMEOUT_S:-${PROBE_TIMEOUT_S}}
MAX_PROBES=${MAX_PROBES:-40}               # live-run budget per config
MAX_CONFIRM_ATTEMPTS=${MAX_CONFIRM_ATTEMPTS:-3}

# run_lf host-mem watchdog (soft C-OOM). Empty floor (default) = the per-model
# map in run_lf_lora_sft.sh picks it (35/50/60G by model size) and the key is
# omitted from row env{}; setting WATCHDOG_FLOOR_GB pins one value for all rows
# and fingerprints it (the floor moves the C-OOM boundary). NOTE: map changes in
# run_lf do NOT invalidate ledgers -- use a fresh state dir if you change it.
WATCHDOG_FLOOR_GB=${WATCHDOG_FLOOR_GB:-}
WATCHDOG_POLL_S=${WATCHDOG_POLL_S:-}   # empty = run_lf default; set to pin+fingerprint

# Driver preflight.
GPU_MAX_USED_MIB=${GPU_MAX_USED_MIB:-6000}
MIN_RAM_AVAIL_GIB=${MIN_RAM_AVAIL_GIB:-64}
MIN_DISK_FREE_GIB=${MIN_DISK_FREE_GIB:-50}
PREFLIGHT_TIMEOUT_S=${PREFLIGHT_TIMEOUT_S:-900}
SETTLE_S=${SETTLE_S:-20}                     # pause after a failed probe

# Priors = known boundary minus ~3k. C-OOM priors start at ohbm0; more HBM
# share never helps a G-OOM.
CONFIGS=(
  # ---- asym_cpuadamwds | recomp-off-full-fg (dense ker000, routed MoE ker101) ----
  "174000 : 16 : q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker101-ceil0000-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"   # 1103s, C-898, G-183, C-OOM 175k [DONE]
  # "65000 : 6 : q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false" # 1659s, C-897, G-183, C-OOM 66k [DONE]
  # "32000 : 0 : llama3.3-70b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"
  # "13000 : 0 : llama4-scout|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"
  # "50000 : 0 : q2.5-32b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"
  # "30000 : 0 : q2.5-72b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"

  # ---- superoffload_mem | unsloth-off ----
  # "128000 : 0 : q3-30b-a3b|1 ; superoffload_mem|unsloth-off-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false" # 532s, C-618, G-153, C-OOM 132k [DONE]
  # "56000 : 5 : q3-32b|1 ; superoffload_mem|unsloth-off-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false" # 469s, C-874, G-181, G-OOM 57k [DONE]
  # "30000 : 0 : llama3.3-70b|1 ; superoffload_mem|unsloth-off-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"
  # "13000 : 0 : llama4-scout|1 ; superoffload_mem|unsloth-off-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"
  # "50000 : 0 : q2.5-32b|1 ; superoffload_mem|unsloth-off-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"
  # "30000 : 0 : q2.5-72b|1 ; superoffload_mem|unsloth-off-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"

  # ---- superoffload_mem | unsloth (HBM-bound: expect ohbm to stay 0) ----
  # "78000 : 0 : q3-30b-a3b|1 ; superoffload_mem|unsloth-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"
  # "47000 : 0 : q3-32b|1 ; superoffload_mem|unsloth-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"
  # "43000 : 0 : llama3.3-70b|1 ; superoffload_mem|unsloth-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"
  # "8000 : 0 : llama4-scout|1 ; superoffload_mem|unsloth-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"
  # "48000 : 0 : q2.5-32b|1 ; superoffload_mem|unsloth-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"
  # "38000 : 0 : q2.5-72b|1 ; superoffload_mem|unsloth-ohbm{ohbm}|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"

  # ---- superoffload_mem | recomp (no ohbm knob: seq-only search) ----
  # "43000 : 0 : q3-30b-a3b|1 ; superoffload_mem|recomp|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"
  # "18000 : 0 : q3-32b|1 ; superoffload_mem|recomp|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"
  # "10000 : 0 : llama3.3-70b|1 ; superoffload_mem|recomp|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"
  # "6000 : 0 : llama4-scout|1 ; superoffload_mem|recomp|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"
  # "19000 : 0 : q2.5-32b|1 ; superoffload_mem|recomp|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"
  # "10000 : 0 : q2.5-72b|1 ; superoffload_mem|recomp|ligerloss1 ; {seq}|8|1 ; none|false|false|false|false|false"
)

# ---- generate the JSONL config list and invoke the python driver ----
die() { echo "error: $*" >&2; exit 2; }

for _nv in SEQ_STEP SEQ_RESOLUTION SEQ_MIN SEQ_MAX PROBE_STEPS CONFIRM_STEPS WARMUP_STEPS \
           PROBE_TIMEOUT_S CONFIRM_TIMEOUT_S MAX_PROBES MAX_CONFIRM_ATTEMPTS \
           GPU_MAX_USED_MIB MIN_RAM_AVAIL_GIB MIN_DISK_FREE_GIB PREFLIGHT_TIMEOUT_S SETTLE_S; do
  [[ "${!_nv}" =~ ^(0|[1-9][0-9]*)$ ]] || die "${_nv} must be a nonnegative integer without leading zeros, got '${!_nv}'"
done
(( PROBE_STEPS >= 1 )) || die "PROBE_STEPS must be >= 1 (0 measured steps would fake-OK probes)"
(( CONFIRM_STEPS >= PROBE_STEPS )) || die "CONFIRM_STEPS must be >= PROBE_STEPS"
if [[ -n "${WATCHDOG_FLOOR_GB}" ]]; then
  [[ "${WATCHDOG_FLOOR_GB}" =~ ^(0|[1-9][0-9]*)$ ]] || die "WATCHDOG_FLOOR_GB must be an integer GB, got '${WATCHDOG_FLOOR_GB}'"
  (( WATCHDOG_FLOOR_GB >= 1 )) || die "WATCHDOG_FLOOR_GB=0 disables the watchdog; the search relies on its soft C-OOM"
fi
if [[ -n "${WATCHDOG_POLL_S}" ]]; then
  [[ "${WATCHDOG_POLL_S}" =~ ^(0|[1-9][0-9]*)(\.[0-9]+)?$ ]] || die "WATCHDOG_POLL_S must be seconds (fractions ok), got '${WATCHDOG_POLL_S}'"
fi
[[ "${PROFILERS}" =~ ^(source|nsys|both)$ ]] || die "PROFILERS must be source|nsys|both (single value), got '${PROFILERS}'"
[[ "${HOST_TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || die "HOST_TAG must be [A-Za-z0-9._-]+ (lands in dir names + env JSON), got '${HOST_TAG}'"
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
  # extra-json "env":{...} REPLACES this block (last-key-wins): re-include the
  # watchdog + GPU_POOL + PROFILERS keys there. GPU_POOL=0,1 covers |1 and |2 rows.
  wd_floor_kv=""
  [[ -n "${WATCHDOG_FLOOR_GB}" ]] && wd_floor_kv="\"HOST_MEM_WATCHDOG_FLOOR_GB\": \"${WATCHDOG_FLOOR_GB}\", "
  [[ -n "${WATCHDOG_POLL_S}" ]] && wd_floor_kv+="\"HOST_MEM_WATCHDOG_POLL_SECONDS\": \"${WATCHDOG_POLL_S}\", "
  line+=", \"env\": {${wd_floor_kv}\"GPU_POOL\": \"${CEIL_GPU_POOL:-0,1}\", \"PROFILERS\": \"${PROFILERS}\", \"HOST_TAG\": \"${HOST_TAG}\", \"DATASET_OVERWRITE\": \"false\"}"
  [[ -n "${extra}" ]] && line+=", ${extra}"
  line+="}"
  printf '%s\n' "${line}" >> "${GEN}"
  echo "  config: ${name}  (seq0=${seq0}, ohbm0=${ohbm0})"
done

echo "generated ${GEN} ($(wc -l < "${GEN}") configs)"

# Validate each row via the wrapper's DRY_RUN (~1 s/row, no training); catches
# template mistakes now instead of mid-search. Disable with VALIDATE_ROWS=false.
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
           PLOT=false PROFILERS="${PROFILERS}" RUN_POST=false CONTINUE_ON_ERROR=false \
           GPU_POOL="${CEIL_GPU_POOL:-0,1}" \
           COLLECT_EXISTING=false UNSLOTH_GC_OUTER_HBM_EVERY_N=0 \
           RUN_NAME= OUTPUT_ROOT="${ASYM_ROOT_DIR}/profiling_${PROFILERS}_ceiling_${HOST_TAG}" \
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
