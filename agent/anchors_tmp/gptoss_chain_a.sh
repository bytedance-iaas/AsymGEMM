#!/bin/bash
# gpt-oss-20b campaign — CHAIN A: dev loss-parity pair -> tier quartet @64k ->
# memory verdict walker @128k. Serial on GPU0. House protocol via tpfig_lib_c17.
# Status -> gptoss_status.log; harvest lines appended per phase.
set -uo pipefail
export GPU="${GPU:-0}" HOSTFLOOR="${HOSTFLOOR:-600}"
source /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c17.sh
S2="$LOGD/gptoss_status.log"
POL="none|false|false|false|false|false"
MODEL=gpt-oss-20b
export CUDA_VISIBLE_DEVICES=0 GPU_POOL=0
export ASYM_OFFLOAD_MODULES=all   # untied embeds; GLM precedent (SO ignores this)
UNS="superoffload_mem|unsloth-ohbm0"
UO="superoffload_mem|unsloth-off-ohbm0"
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
# T3 recipe env (moe|T3 minus the qwen-gated ker101 token; qwen pins inert for
# the gpt-oss own engine but exported verbatim for recipe fidelity)
T3ENV=(ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1
       ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 ASYMM_QWEN3_MOE_FG_DA_GPU=1
       ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1 ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1
       ASYM_CPU_OPS_THREADS=48 ASYM_PLACEMENT_POLICY=1)

note() { echo "[$(date +%H:%M:%S)] $*" >> "$S2"; }
harv() { /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/.venv/bin/python \
           agent/anchors_tmp/gptoss_harvest.py "$1" "${2:-1}" 2>/dev/null | tee -a "$S2"; }
cell() { local tag="$1" sys="$2" seq="$3" blist="$4"; local v
  v=$(run_cell "$tag" "$MODEL" "$sys" "$seq" "$blist" "$POL" 1)
  note "CELL $tag ${sys%%|*} s=$seq -> $v"; echo "$v"; }
t3cell() { local tag="$1" seq="$2" blist="$3"; local v
  v=$( (export "${T3ENV[@]}"; run_cell "$tag" "$MODEL" "$T3TOK" "$seq" "$blist" "$POL" 1) )
  note "CELL $tag T3(ker000) s=$seq -> $v"; echo "$v"; }
trained_b() { grep -a "CELL $1 " "$S" | grep TRAINED | tail -1 | grep -oE "b=[0-9]+" | cut -d= -f2; }
failed_bs() { grep -a "CELL $1 " "$S" | grep -E "GOOM|COOM" | grep -oE "b=[0-9]+" | cut -d= -f2; }
hbm_pct() { # $1 tag $2 b $3 seq -> integer percent of 189471 MiB
  local leaf; leaf=$(ls -d "$B/$1-c17_gpt-oss-20b__b$2_s$3_ga1_drop000"/*/"b$2_s$3_ga1" 2>/dev/null | head -1)
  [ -n "$leaf" ] || { echo 0; return; }
  /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/.venv/bin/python - "$leaf" <<'PY'
import json, sys
p = json.load(open(sys.argv[1] + "/profile.json"))
print(round(100 * p["memory"]["peak_reserved_hbm_bytes"] / (189471 * 2**20)))
PY
}

note "CHAIN-A begin"

# ── Phase A: dev loss-parity pair (8k b1, w1+m1) ────────────────────────────
va1=$( (export MAX_STEPS=1; cell a_uns "$UNS" 8000 "1") )
va2=$( (export MAX_STEPS=1; cell a_t1 "asym_cpuadamwds|T1" 8000 "1") )
# FA4-sink numeric cross-check: same baseline on the native eager path —
# FA4 vs eager step-1 loss must agree to bf16 noise (both dev cells run FA4,
# so parity alone can't catch a wrong FA4 sink integration).
va3=$( (export MAX_STEPS=1 FLASH_ATTN=disabled; cell a_eag "$UNS" 8000 "1") )
harv a_uns; harv a_t1; harv a_eag
if [ "$va1" != "TRAINED" ] || [ "$va2" != "TRAINED" ]; then
  note "PHASE-A FAILED (uns=$va1 t1=$va2 eager=$va3) — ABORT CHAIN"; exit 1
fi
[ "$va3" = "TRAINED" ] || note "PHASE-A NOTE: eager cross-check cell failed ($va3) — FA4 numerics unverified vs eager, investigate"
note "PHASE-A done"

# ── Phase B: tier quartet + references @64k b1 (w1+m2) ─────────────────────
vb_uns=$(cell b_uns "$UNS" 64000 "1")
vb_uo=$(cell b_uo "$UO" 64000 "1")
vb_t1=$(cell b_t1 "asym_cpuadamwds|T1" 64000 "1")
vb_t2=$(cell b_t2 "asym_cpuadamwds|T2" 64000 "1")
vb_t2b=$(cell b_t2b "asym_cpuadamwds|T2B" 64000 "1")
vb_t3=$(t3cell b_t3 64000 "1")
for t in b_uns b_uo b_t1 b_t2 b_t2b b_t3; do harv "$t"; done
if [ "$vb_t1" != "TRAINED" ] || [ "$vb_t2" != "TRAINED" ] || [ "$vb_t2b" != "TRAINED" ] || [ "$vb_t3" != "TRAINED" ]; then
  note "PHASE-B: tier failures (t1=$vb_t1 t2=$vb_t2 t2b=$vb_t2b t3=$vb_t3) — chain continues to aid diagnosis, verdict blocked"
fi
note "PHASE-B done"

# ── Phase C: memory verdict walker @128k ────────────────────────────────────
vc=$(cell c_uo "$UO" 128000 "8 6 4 3 2 1")
bstar=$(trained_b c_uo); uotag=c_uo
if [ "$vc" = "TRAINED" ] && [ "${bstar:-0}" = "8" ]; then
  pct=$(hbm_pct c_uo 8 128000)
  note "c_uo b8 HBM=${pct}%"
  if [ "${pct:-0}" -lt 60 ]; then
    vup=$(cell c_uo_up "$UO" 128000 "16 12 10")
    if [ "$vup" = "TRAINED" ]; then bstar=$(trained_b c_uo_up); uotag=c_uo_up; fi
  fi
fi
if [ -z "${bstar:-}" ]; then
  note "uns-off found NO fitting batch at 128k (walked 8..1) — T3 capacity story: probe T3 b1"
  t3cell c_t3 128000 "1" >/dev/null; harv c_t3
else
  note "uns-off verdict batch b*=$bstar (tag $uotag, HBM $(hbm_pct "$uotag" "$bstar" 128000)%)"
  t3cell c_t3 128000 "$bstar" >/dev/null
  harv "$uotag"; harv c_t3
  # dominance probe at the baseline's bracketed wall (smallest failed b > b*)
  bfail=$( { failed_bs c_uo; failed_bs c_uo_up; } | sort -n | awk -v bs="$bstar" '$1>bs{print; exit}')
  if [ -n "${bfail:-}" ]; then
    note "baseline wall bracketed at b=$bfail — T3 dominance probe"
    t3cell c_t3cap 128000 "$bfail" >/dev/null; harv c_t3cap
  else
    note "no baseline wall inside walked range — probing wall at b=$((bstar+4))"
    vprobe=$(cell c_uo_cap "$UO" 128000 "$((bstar+4))")
    if [ "$vprobe" != "TRAINED" ]; then
      t3cell c_t3cap 128000 "$((bstar+4))" >/dev/null; harv c_uo_cap; harv c_t3cap
    else
      harv c_uo_cap; note "baseline survives b=$((bstar+4)) too — record batch-parity capacity honestly"
    fi
  fi
fi
note "PHASE-C done"
note "CHAIN-A COMPLETE"
