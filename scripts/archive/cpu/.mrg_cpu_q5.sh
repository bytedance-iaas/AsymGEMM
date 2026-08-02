#!/bin/bash
# CPU-merge queue 5: deep-shed bisect (S6 breach protocol for M8 −1.8% / M9 pattern).
# One variable at a time on M8 (cheaper row). Chain STOPS at first knob that
# recovers to PASS (>= 575.2 tok/s): that knob's feature gets a tokens-ceiling
# policy gate, then M8+M9 re-run in Q6 with the gate fix.
# Candidate order by mechanism strength on the saturated deep-shed H2D/D2H bus:
#   1. prefetch-off  (holds compete with restages; M7 already showed holds hurt)
#   2. qknorm-off    (adds one D2H offload per norm at 800k rows)
#   3. deposit-off   (dS D2H crossing per expert wgrad)
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
POL="ASYM_PLACEMENT_POLICY=1 ASYM_CPU_OPS_THREADS=48"
PINS="ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 ASYMM_QWEN3_MOE_FG_DA_GPU=1 ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1"

verdict() {
python3 - "$1" <<'PY'
import json, glob, sys
try:
    ss = glob.glob(f"profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16/{sys.argv[1]}_q3-30b-a3b__b1_s800000_ga1_drop000/asym*/b1_s800000_ga1/step_samples.json")[0]
    rows = json.load(open(ss)); meas = [r for r in rows if not r.get("is_warmup")]
    def sec(r):
        for k in ("e2e_milliseconds","step_milliseconds","trainer_e2e_step_milliseconds","training_step_milliseconds"):
            if r.get(k): return r[k]/1000.0
    secs = [sec(r) for r in meas if sec(r)]
    tok = 800000/(sum(secs)/len(secs))
    print("PASS" if tok >= 584*0.985 else "FAIL", f"{tok:.0f}")
except Exception as e:
    print("ERROR", e)
PY
}

echo "=== Q5a M8 bisect: prefetch-OFF $(date -u +%H:%M:%S)"
env $POL ASYM_GEMM_DISPATCH=staged $PINS ASYM_PREFETCH_MIN_FREE_GB=9999 MAX_SAMPLES=1024 bash scripts/lf/tp_probe.sh q3-30b-a3b cpub8np "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 800000 1
echo "Q5A_EXIT=$?"
V=$(verdict cpub8np); echo "Q5A_VERDICT=$V"
case "$V" in PASS*) echo "=== ATTRIBUTED: prefetch. Q5 done."; echo "=== CPU-Q5 DONE $(date -u +%H:%M:%S)"; exit 0;; esac

echo "=== Q5b M8 bisect: qknorm-OFF $(date -u +%H:%M:%S)"
env $POL ASYM_GEMM_DISPATCH=staged $PINS ASYM_POLICY_QKNORM=0 MAX_SAMPLES=1024 bash scripts/lf/tp_probe.sh q3-30b-a3b cpub8nq "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 800000 1
echo "Q5B_EXIT=$?"
V=$(verdict cpub8nq); echo "Q5B_VERDICT=$V"
case "$V" in PASS*) echo "=== ATTRIBUTED: qknorm. Q5 done."; echo "=== CPU-Q5 DONE $(date -u +%H:%M:%S)"; exit 0;; esac

echo "=== Q5c M8 bisect: deposit-OFF $(date -u +%H:%M:%S)"
env $POL ASYM_GEMM_DISPATCH=staged $PINS ASYM_POLICY_MOE_DEPOSIT=0 MAX_SAMPLES=1024 bash scripts/lf/tp_probe.sh q3-30b-a3b cpub8nd "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 800000 1
echo "Q5C_EXIT=$?"
V=$(verdict cpub8nd); echo "Q5C_VERDICT=$V"
echo "=== CPU-Q5 DONE $(date -u +%H:%M:%S)"
