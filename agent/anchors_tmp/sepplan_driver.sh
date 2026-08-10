#!/bin/bash
# sEP-planned mirror campaign driver (HOST side): git-ledger coordination +
# per-cell enroot launches. Order: flash-T2-deep (fuse window) -> mixtral
# 304k->32k -> flash T1 832k->32k -> air 320k->16k.
set -uo pipefail
R=/home/kevinni/AsymGEMM-SFT-38/third_party/AsymGEMM
A=$R/agent/anchors_tmp
LED=$A/SEPPLAN_CAMPAIGN.md
DLOG=$A/sepplan_driver.log
FUSELOG=$A/mx_fuse_local.log
FUSED=/scratch_local/user_data/shutian/kevin/cache/fused/Mixtral-8x22B-v0.1
cd $R
export ENROOT_CONFIG_PATH=/scratch_local/user_data/shutian/kevin/enroot/config \
       ENROOT_DATA_PATH=/scratch_local/user_data/shutian/kevin/enroot/data \
       ENROOT_CACHE_PATH=/scratch_local/user_data/shutian/kevin/enroot/cache \
       ENROOT_RUNTIME_PATH=/scratch_local/user_data/shutian/kevin/enroot/runtime/ \
       ENROOT_TEMP_PATH=/scratch_local/user_data/shutian/kevin/enroot/tmp
MOUNTS=(--mount=/home/kevinni/AsymGEMM-SFT-38:/workspace/AsymGEMM-SFT-38
        --mount=/home/kevinni/env:/workspace/env
        --mount=/scratch_local/user_data/shutian/kevin/cache:/scratch_local/user_data/shutian/kevin/cache)
log() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" >> $DLOG; }

COORD=1
git push --dry-run origin main_kevin >/dev/null 2>&1 || COORD=0
log "driver start COORD=$COORD"

gsync_l() { local i; for i in 1 2 3; do git pull --rebase -q origin main_kevin 2>>$DLOG && return 0; sleep 5; done; log "pull failed x3"; return 1; }
gpush_l() { [ "$COORD" = "1" ] || return 0
  local i; for i in 1 2 3; do git push -q origin main_kevin 2>>$DLOG && return 0; git pull --rebase -q origin main_kevin 2>>$DLOG; sleep 5; done
  log "push failed x3"; return 1; }
ledger() { echo "- [$(date -u '+%m-%d %H:%MZ')] c17 $*" >> $LED; git add $LED >/dev/null 2>&1; git commit -q -m "sepplan ledger: $*" 2>>$DLOG; gpush_l; }
done_already() { # model seq — any runner's DONE (or honest-wall) line
  grep -hE "DONE sepplan" $A/*.md 2>/dev/null | grep -F "$1" | grep -qE "s=$2([^0-9]|$)"
}

run_cell_outer() { # TAG MODEL SYSTOK SEQ BLIST ARENA FLOOR
  local tag=$1 model=$2 systok=$3 seq=$4 blist=$5 arena=$6 floor=$7
  gsync_l || true
  if done_already "$model" "$seq"; then log "SKIP $tag $model s=$seq (already done)"; return 0; fi
  ledger "CLAIM sepplan $model s=$seq ($tag)"
  log "RUN $tag $model $systok s=$seq b='$blist' arena=$arena"
  local v
  v=$(enroot start --rw --root "${MOUNTS[@]}" asym_sft_45 /bin/bash \
        /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/sepplan_cell.sh \
        "$tag" "$model" "$systok" "$seq" "$blist" "$arena" "$floor" 2>>$DLOG | tail -1)
  local h
  h=$(cd $R && python3 agent/anchors_tmp/sepplan_harvest.py "$tag" 2>/dev/null | tail -1)
  log "DONE $tag -> $v | $h"
  gsync_l || true
  ledger "DONE sepplan $model s=$seq ($tag) -> ${v:-NOVERDICT} | ${h:-noharvest}"
}

# ---- fuse rebuild (background) ----
if [ ! -f $FUSED/model.safetensors.index.json ]; then
  log "launching mixtral fuse rebuild"
  ledger "NOTE c17 fused mixtral ckpt absent on this node — rebuilding (mx_fuse_local.py) before mixtral cells"
  enroot start --rw --root "${MOUNTS[@]}" asym_sft_45 /bin/bash -c \
    "export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface HF_HUB_OFFLINE=1; cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM && .venv/bin/python agent/anchors_tmp/mx_fuse_local.py" \
    > $FUSELOG 2>&1 &
  FUSE_PID=$!
else
  log "fused ckpt already present"
  FUSE_PID=""
fi

# ---- PHASE A: flash T2 deep trio (fuse window; farthest from other runner) ----
run_cell_outer spgf1024 glm4.7-flash "asym_sepplan2_cpuadamwds|T2" 1024000 "1" 0 1100
run_cell_outer spgf960  glm4.7-flash "asym_sepplan2_cpuadamwds|T2" 960000  "1" 0 1100
run_cell_outer spgf896  glm4.7-flash "asym_sepplan2_cpuadamwds|T2" 896000  "1" 0 1100

# ---- gate on fuse before mixtral ----
MX_OK=1
if [ -n "${FUSE_PID}" ]; then
  for i in $(seq 1 360); do
    [ -f $FUSED/model.safetensors.index.json ] && grep -q "DONE" $FUSELOG 2>/dev/null && break
    kill -0 $FUSE_PID 2>/dev/null || { grep -q "DONE" $FUSELOG 2>/dev/null && break; log "fuse process died"; MX_OK=0; break; }
    sleep 30
  done
  [ -f $FUSED/model.safetensors.index.json ] || MX_OK=0
fi
if [ "$MX_OK" != "1" ]; then
  log "FUSE FAILED — skipping mixtral cells"
  ledger "NOTE c17 FUSE FAILED (see mx_fuse_local.log) — mixtral cells skipped by this runner"
fi

# ---- PHASE B: mixtral 304k -> 32k (T1, arena 285) ----
if [ "$MX_OK" = "1" ]; then
  run_cell_outer spmx304 mixtral-8x22b "asym_sepplan2_cpuadamwds|T1" 304000 "1" 285 1300
  run_cell_outer spmx288 mixtral-8x22b "asym_sepplan2_cpuadamwds|T1" 288000 "1" 285 1300
  run_cell_outer spmx256 mixtral-8x22b "asym_sepplan2_cpuadamwds|T1" 256000 "1" 285 1300
  run_cell_outer spmx192 mixtral-8x22b "asym_sepplan2_cpuadamwds|T1" 192000 "1" 285 1300
  run_cell_outer spmx128 mixtral-8x22b "asym_sepplan2_cpuadamwds|T1" 128000 "2" 285 1300
  run_cell_outer spmx64  mixtral-8x22b "asym_sepplan2_cpuadamwds|T1" 64000  "4" 285 1300
  run_cell_outer spmx32  mixtral-8x22b "asym_sepplan2_cpuadamwds|T1" 32000  "8" 285 1300
fi

# ---- PHASE C: flash T1 832k -> 32k ----
run_cell_outer spgf832 glm4.7-flash "asym_sepplan2_cpuadamwds|T1" 832000 "1" 0 1100
run_cell_outer spgf768 glm4.7-flash "asym_sepplan2_cpuadamwds|T1" 768000 "1" 0 1100
run_cell_outer spgf704 glm4.7-flash "asym_sepplan2_cpuadamwds|T1" 704000 "1" 0 1100
run_cell_outer spgf640 glm4.7-flash "asym_sepplan2_cpuadamwds|T1" 640000 "1" 0 1100
run_cell_outer spgf576 glm4.7-flash "asym_sepplan2_cpuadamwds|T1" 576000 "1" 0 1100
run_cell_outer spgf512 glm4.7-flash "asym_sepplan2_cpuadamwds|T1" 512000 "1" 0 1100
run_cell_outer spgf416 glm4.7-flash "asym_sepplan2_cpuadamwds|T1" 416000 "2" 0 1100
run_cell_outer spgf320 glm4.7-flash "asym_sepplan2_cpuadamwds|T1" 320000 "2" 0 1100
run_cell_outer spgf256 glm4.7-flash "asym_sepplan2_cpuadamwds|T1" 256000 "3" 0 1100
run_cell_outer spgf192 glm4.7-flash "asym_sepplan2_cpuadamwds|T1" 192000 "4" 0 1100
run_cell_outer spgf160 glm4.7-flash "asym_sepplan2_cpuadamwds|T1" 160000 "2" 0 1100
run_cell_outer spgf128 glm4.7-flash "asym_sepplan2_cpuadamwds|T1" 128000 "3" 0 1100
run_cell_outer spgf96  glm4.7-flash "asym_sepplan2_cpuadamwds|T1" 96000  "4" 0 1100
run_cell_outer spgf64  glm4.7-flash "asym_sepplan2_cpuadamwds|T1" 64000  "6" 0 1100
run_cell_outer spgf32  glm4.7-flash "asym_sepplan2_cpuadamwds|T1" 32000  "12" 0 1100

# ---- PHASE D: air 320k -> 16k (T1, arena 240) ----
run_cell_outer spga320 glm4.5-air "asym_sepplan2_cpuadamwds|T1" 320000 "1" 240 1300
run_cell_outer spga256 glm4.5-air "asym_sepplan2_cpuadamwds|T1" 256000 "2 1" 240 1300
run_cell_outer spga192 glm4.5-air "asym_sepplan2_cpuadamwds|T1" 192000 "2 1" 240 1300
run_cell_outer spga160 glm4.5-air "asym_sepplan2_cpuadamwds|T1" 160000 "3 2" 240 1300
run_cell_outer spga128 glm4.5-air "asym_sepplan2_cpuadamwds|T1" 128000 "2" 240 1300
run_cell_outer spga96  glm4.5-air "asym_sepplan2_cpuadamwds|T1" 96000  "2" 240 1300
run_cell_outer spga64  glm4.5-air "asym_sepplan2_cpuadamwds|T1" 64000  "4" 240 1300
run_cell_outer spga48  glm4.5-air "asym_sepplan2_cpuadamwds|T1" 48000  "4" 240 1300
run_cell_outer spga32  glm4.5-air "asym_sepplan2_cpuadamwds|T1" 32000  "8" 240 1300
run_cell_outer spga16  glm4.5-air "asym_sepplan2_cpuadamwds|T1" 16000  "16" 240 1300

ledger "SEPPLAN c17 SWEEP COMPLETE"
log "driver complete"
