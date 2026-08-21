#!/bin/bash
# stdtps46_lib.sh — c18 / SFT-46 lane (Session D) of the TP-figure x-axis
# standardization campaign (agent/impls/s04-p1-dgx-02-c06/standardize_tps.md).
# Sourced by the stdtps46_a*_*.sh chains. Port of the sibling stdtps_lib.sh /
# fig12_lib.sh run helpers: IDENTICAL measurement protocol (w1+m2,
# PROFILERS=source, MAX_SAMPLES=512, launcher NUMA defaults membind 0,1,
# serial, pre-flight GPU-empty + host-floor guard, explicit-PID kills only,
# artifacts-first verdicts) — only workspace root, log dir, node tag differ.
# 1-rank cells run on GPU 0, 2-rank cells on GPUs 0+1 (GPU_POOL=0,1).
cd /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM || exit 1
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512 DATASET_OVERWRITE=false OVERWRITE=false
export ASYM_ZERO_ROUTER_JITTER=1 TRUST_REMOTE_CODE=false
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
export HOSTFLOOR=${HOSTFLOOR:-1400}     # GB "available" (free -g) required before a launch (idle node ~1595)
export RUN_TIMEOUT=${RUN_TIMEOUT:-10800} # per-attempt wall clock (s); fill-campaign precedent 3h
LOGD=/workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp
S="$LOGD/stdtps46_status.log"
B=profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16
NODE=c18

_gpu_pids() { nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ,' | grep -E '^[0-9]+$'; }

# guard GPUS — wait until the given GPUs hold no compute process, host
# "available" >= HOSTFLOOR and /dev/shm carries no stale asym_* segment.
guard() { local gpus="$1" i n a p pp used
  for i in $(seq 1 180); do
    n=0; for p in $(_gpu_pids "$gpus"); do [ -d "/proc/$p" ] && n=$((n+1)); done
    if [ "$n" -eq 0 ] && ls /dev/shm/asym_* >/dev/null 2>&1; then
      echo "GUARD-SHM-CLEAN $(date +%H:%M)" >> "$S"; rm -f /dev/shm/asym_* 2>/dev/null || true
    fi
    used=$(df -BG /dev/shm | awk 'NR==2{gsub("G","",$3); print $3}')
    a=$(free -g | awk 'NR==2{print $7}')
    if [ "$n" -eq 0 ] && [ "$a" -ge "$HOSTFLOOR" ] && [ "${used:-0}" -lt 5 ]; then return 0; fi
    if [ $((i % 9)) -eq 0 ]; then
      echo "GUARD-WAIT gpus=$gpus procs=$n avail=${a}G shm=${used}G $(date +%H:%M)" >> "$S"
      # reap ORPHANED gpu holders (ppid=1) only — never a live run's children
      for p in $(_gpu_pids "$gpus"); do
        pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d " ")
        [ "$pp" = "1" ] && kill -9 "$p" 2>/dev/null && echo "GUARD-REAPED orphan $p gpus=$gpus $(date +%H:%M)" >> "$S"
      done
    fi
    sleep 20
  done; echo "GUARD-TIMEOUT gpus=$gpus $(date +%H:%M)" >> "$S"; return 1; }

# cell_trained RUNDIR_TAG -> exit 0 if the cell is a valid TRAINED cell, ARTIFACTS FIRST:
#   jobs.tsv `ok`, OR step_samples.csv carrying all MAX_STEPS measured (non-warmup)
#   rows with a loss + the trainer's final metrics block in train.log. The second
#   form is how the banked sEP 2r cells (e.g. r2a1sep640, jobs.tsv failed:1) were
#   accepted: the post-run "incomplete/partial profile" gate flags sepplan2 runs
#   whose training completed normally (s2q35sep768 2026-08-20 reproduced it).
cell_trained() { local dtag="$1" tsv="$B/$1/jobs.tsv" ss nm tl
  [ -f "$tsv" ] && awk -F'\t' 'NR>1 && $1=="ok"{f=1} END{exit !f}' "$tsv" && return 0
  ss=$(find "$B/$dtag" -name step_samples.csv 2>/dev/null | head -1); [ -n "$ss" ] || return 1
  nm=$(.venv/bin/python - "$ss" <<'PY' 2>/dev/null
import csv, sys
n = 0
for r in csv.DictReader(open(sys.argv[1])):
    if r.get("is_warmup", "").lower() in ("true", "1"): continue
    try:
        if float(r.get("step_milliseconds") or 0) > 0 and r.get("loss") not in (None, ""): n += 1
    except ValueError: pass
print(n)
PY
)
  [ "${nm:-0}" -ge "${MAX_STEPS:-2}" ] || return 1
  tl="$(dirname "$ss")/train.log"; grep -aq "train_runtime" "$tl" 2>/dev/null || return 1
  return 0
}

# verdict RUNDIR_TAG LOG -> TRAINED | GOOM | COOM | NCCL | TIMEOUT | FAIL (artifacts first)
verdict() { local dtag="$1" log="$2"
  if cell_trained "$dtag"; then echo TRAINED; return; fi
  grep -aqE "OutOfMemoryError|CUDA out of memory|ncclUnhandledCudaError" "$log" 2>/dev/null && { echo GOOM; return; }
  grep -aqE "dropped below floor|HOST-C-OOM|host memory watchdog" "$log" 2>/dev/null && { echo COOM; return; }
  grep -aqE "collective operation timeout|NCCL watchdog|Watchdog caught collective" "$log" 2>/dev/null && { echo NCCL; return; }
  grep -aq "RUN-TIMEOUT" "$log" 2>/dev/null && { echo TIMEOUT; return; }
  echo FAIL
}

# harvest RUNDIR_TAG RANKS -> "eff=<tok/s> resv=<GiB> rss=<GiB> steps=<s,s,s>" (house formula, GLOBAL for 2r)
harvest() { local dtag="$1" ranks="$2"
  .venv/bin/python - "$B/$dtag" "$ranks" <<'PY' 2>/dev/null
import csv, glob, json, os, re, sys
root, ranks = sys.argv[1], int(sys.argv[2])
out = []
for ss in glob.glob(f"{root}/**/step_samples.csv", recursive=True):
    rd = os.path.dirname(ss)
    m = re.match(r"b(\d+)_s(\d+)_ga(\d+)", os.path.basename(rd))
    if not m: continue
    b, s, ga = map(int, m.groups())
    rows = list(csv.DictReader(open(ss)))
    meas = [float(r["step_milliseconds"]) for r in rows
            if r.get("is_warmup", "").lower() not in ("true", "1") and float(r.get("step_milliseconds") or 0) > 0]
    if not meas: continue
    eff = ranks * len(meas) * b * s * ga / (sum(meas) / 1000.0)
    resv = rss = 0.0
    pj = os.path.join(rd, "profile.json")
    if os.path.exists(pj):
        try:
            mem = json.load(open(pj)).get("memory", {})
            resv = mem.get("peak_reserved_hbm_bytes", 0) / 2**30
            rss = mem.get("process", {}).get("rss_peak_bytes", 0) / 2**30
        except Exception: pass
    steps = ",".join(f"{float(r['step_milliseconds'])/1000:.1f}" for r in rows)
    out.append(f"eff={eff:.0f} resv={resv:.1f}GiB({resv/189.5*100:.0f}%) rss={rss:.0f}GiB steps={steps} nmeas={len(meas)}")
print(" | ".join(out) if out else "no-step-samples")
PY
}

# kill every compute process on GPUS (explicit PIDs; the node is serially
# ours, so anything left after a timeout is the timed-out run's tree).
_reap_gpus() { local gpus="$1" p
  for p in $(_gpu_pids "$gpus"); do kill -TERM "$p" 2>/dev/null; done; sleep 20
  for p in $(_gpu_pids "$gpus"); do kill -9 "$p" 2>/dev/null && echo "REAPED pid $p after timeout $(date +%H:%M)" >> "$S"; done
}

# run_cell TAG MODEL SYSTOKEN SEQ "B1 B2 ..." [POLICY] [RANKS]
#   walks the batch list in order; first TRAINED wins the cell; stops on FAIL.
#   Env passthrough: ARM_ENV="K=V K=V" exported for the run (user env wins over
#   the tier recipe inside the harness). Prints the final verdict.
run_cell() { local tag="$1" model="$2" systok="$3" seq="$4" blist="$5" policy="${6:-none|false|false|false|false|false}" ranks="${7:-1}" v=SKIP b gpus log n dtag dmodel rc
  if [ "$ranks" = "2" ]; then gpus="0,1"; else gpus="${ONE_RANK_GPU:-0}"; fi   # ONE_RANK_GPU=1 -> confirmation probes on the pristine GPU 1
  dmodel=${model//./_}
  for b in $blist; do
    dtag="${tag}-${NODE}_${dmodel}__b${b}_s${seq}_ga1_drop000"
    if cell_trained "$dtag"; then
      echo "CELL $tag $systok s=$seq b=$b r=$ranks -> TRAINED (already banked: $(harvest "$dtag" "$ranks")) $(date +%H:%M)" >> "$S"; v=TRAINED; break
    fi
    guard "$gpus" || return 1
    rm -f /dev/shm/asym_* 2>/dev/null || true
    # fresh log per attempt (stale-log verdict lesson, TPFIG_CAMPAIGN)
    n=1; while [ -e "$LOGD/r_${tag}_b${b}.try${n}.log" ]; do n=$((n+1)); done
    log="$LOGD/r_${tag}_b${b}.try${n}.log"
    echo "START $tag $model $systok s=$seq b=$b r=$ranks gpus=$gpus arm='${ARM_ENV:-}' $(date +%H:%M)" >> "$S"
    ( set -a; for kv in ${ARM_ENV:-}; do export "$kv"; done; set +a
      export CUDA_VISIBLE_DEVICES="$gpus" GPU_POOL="$gpus" DDP_TIMEOUT="${DDP_TIMEOUT:-1500}"
      RUN_NAME="${tag}-${NODE}_${model}" RUNS="${model}|${ranks} ; ${systok}|ligerloss1 ; ${seq}|${b}|1 ; ${policy}" \
        timeout -k 60 "$RUN_TIMEOUT" bash scripts/lf/profile_lora_lf_test_source.sh ) > "$log" 2>&1
    rc=$?
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then echo "RUN-TIMEOUT rc=$rc" >> "$log"; _reap_gpus "$gpus"; fi
    v=$(verdict "$dtag" "$log")
    if [ "$v" = "TRAINED" ]; then
      echo "CELL $tag $systok s=$seq b=$b r=$ranks -> TRAINED $(harvest "$dtag" "$ranks") $(date +%H:%M)" >> "$S"
    else
      echo "CELL $tag $systok s=$seq b=$b r=$ranks -> $v (rc=$rc, $(grep -acE 'OutOfMemoryError|CUDA out of memory' "$log")x goom-lines; log $(basename "$log")) $(date +%H:%M)" >> "$S"
    fi
    [ "$v" = "TRAINED" ] && break
    [ "$v" = "FAIL" ] && break
    [ "$v" = "TIMEOUT" ] && break
  done; echo "$v"; }
