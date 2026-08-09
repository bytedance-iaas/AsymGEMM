#!/bin/bash
# glmext_1r_b.sh — Flash 1r stage 3 (2026-08-06): TARGETED endgame cells only.
# uo survived the whole 512-896k ladder (RSS 772 GiB @896k, +208/rung → host
# wall ≈ (1024k,1152k]). Rendered cascade columns chosen: 320k · 448k · 640k ·
# 1024k · 1152k · 1280k — asym runs ONLY where rendered (640k + deep trio),
# skipping the never-rendered 512/576/704/768/896 asym cells (~8h saved; the
# blind-walk chain was killed 5 min into x1et1512). Tier start T2 on the deep
# trio (T1's HBM slope walls ~(768k,896k]; probing T1 there = known-dead).
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU=0 HOSTFLOOR=500
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
echo "GLM1RB begin $(date +%H:%M)" >> "$S"
P="none|false|false|false|false|false"
export CUDA_VISIBLE_DEVICES=0
BE=asym_cpuadamwds
T3TOK="${BE}|recomp-off-full-fg-ker000-ceil0000-ohbm0"

# asym rung with tier-start control: asym_rung SEQ STARTTIER(BLIST fixed "1")
asym_rung() { local s="$1" start="${2:-1}" v=SKIP
  if [ "$start" -le 1 ]; then
    v=$(run_cell "x1bt1$((s/1000))" glm4.7-flash "${BE}|T1" "$s" "1" "$P" 1)
    [ "$v" = "TRAINED" ] && return 0
  fi
  if [ "$start" -le 2 ]; then
    v=$(run_cell "x1bt2$((s/1000))" glm4.7-flash "${BE}|T2" "$s" "1" "$P" 1)
    [ "$v" = "TRAINED" ] && return 0
  fi
  v=$(run_cell "x1bt3$((s/1000))" glm4.7-flash "$T3TOK" "$s" "1" "$P" 1)
  [ "$v" = "TRAINED" ] && return 0
  echo "WALL x1basym glm4.7-flash r1 s=${s} $(date +%H:%M)" >> "$S"; return 1; }

# 1) asym @640k (uns-death rendered column) — T1 fits by slope (~146 GiB)
asym_rung 640000 1

# 2) uo deep walk to its wall
for s in 1024000 1152000 1280000; do
  v=$(run_cell "x1euo$((s/1000))" glm4.7-flash "superoffload_mem|unsloth-off-ohbm0" "$s" "1" "$P" 1)
  [ "$v" = "TRAINED" ] || { echo "WALL x1euo glm4.7-flash r1 s=${s} ($v) $(date +%H:%M)" >> "$S"; break; }
done

# 3) asym deep trio (rendered): T2-start (T1 known-dead by slope)
asym_rung 1024000 2 || true
asym_rung 1152000 2 || true
asym_rung 1280000 2 || true
echo "X1B-DONE FLASH-1R-CELLS-DONE $(date +%H:%M)" >> "$S"
