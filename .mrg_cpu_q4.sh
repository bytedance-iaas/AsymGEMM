#!/bin/bash
# CPU-merge queue 4: post-matrix reruns.
#  Q4a: M1 SOLO rerun (its only prior attempt was the orphan-instance collision).
#  Q4b: M2 rerun (first pass 969 = -1.7%, 0.2pp past band; today's dense drift ±3%).
#  Q4c: CONDITIONAL bisect — M2 with rope recompute OFF — runs only if Q4b still breaches
#       (protocol: one-variable bisect; candidate = the only engaged feature with a
#        plausible time cost at this row; fix would be a policy tokens-gate, not a fork).
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
POL="ASYM_PLACEMENT_POLICY=1 ASYM_CPU_OPS_THREADS=48"
T2D="ASYM_GEMM_DISPATCH=staged ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM=1 ASYM_SAVED_TENSOR_ASYNC_UNPACK=1 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024"

echo "=== Q4a M1 SOLO rerun q32 T1 128k b2 (base 1091 / 116.0) $(date -u +%H:%M:%S)"
env $POL ASYM_GEMM_DISPATCH=staged bash scripts/lf/tp_probe.sh q3-32b cpum1r "asym_cpuadamwds|unsloth-ohbm0|ligerloss1" 128000 2
echo "Q4A_EXIT=$?"

echo "=== Q4b M2 rerun q32 T2 128k b2 (base 986 / 93.6; pass1 969) $(date -u +%H:%M:%S)"
env $POL $T2D bash scripts/lf/tp_probe.sh q3-32b cpum2r "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 128000 2
echo "Q4B_EXIT=$?"

R=$(python3 - <<'PY'
import json, glob
try:
    ss = glob.glob("profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16/cpum2r_q3-32b__b2_s128000_ga1_drop000/asym*/b2_s128000_ga1/step_samples.json")[0]
    rows = json.load(open(ss)); meas = [r for r in rows if not r.get("is_warmup")]
    def sec(r):
        for k in ("e2e_milliseconds","step_milliseconds","trainer_e2e_step_milliseconds","training_step_milliseconds"):
            if r.get(k): return r[k]/1000.0
    secs = [sec(r) for r in meas if sec(r)]
    tok = 256000/(sum(secs)/len(secs))
    print("BREACH" if tok < 986*0.985 else "PASS")
except Exception:
    print("ERROR")
PY
)
echo "Q4B_VERDICT=$R"
if [ "$R" = "BREACH" ]; then
  echo "=== Q4c BISECT M2 rope-OFF (dense override knob) $(date -u +%H:%M:%S)"
  env $POL $T2D ASYM_POLICY_ROPE_DENSE=0 bash scripts/lf/tp_probe.sh q3-32b cpum2nr "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 128000 2
  echo "Q4C_EXIT=$?"
  echo "=== Q4d BISECT M2 prefetch-OFF (guard never passes) $(date -u +%H:%M:%S)"
  env $POL $T2D ASYM_PREFETCH_MIN_FREE_GB=9999 bash scripts/lf/tp_probe.sh q3-32b cpum2np "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 128000 2
  echo "Q4D_EXIT=$?"
fi
KA120="ASYM_GEMM_DISPATCH=staged ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1"
echo "=== Q4e M7 rerun with 32GB prefetch floor (base 2762/165.7; pass1 2672/176.7) $(date -u +%H:%M:%S)"
env $POL $KA120 MAX_SAMPLES=1024 bash scripts/lf/tp_probe.sh q3-30b-a3b cpum7r "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 120000 8
echo "Q4E_EXIT=$?"
PINS="ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 ASYMM_QWEN3_MOE_FG_DA_GPU=1 ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1"
echo "=== Q4f M8 retry (base 584/110.4; pass1 574 = -1.8%, M9 in-band => day-noise check; Q5 bisect only if this breaches too) $(date -u +%H:%M:%S)"
env $POL ASYM_GEMM_DISPATCH=staged $PINS MAX_SAMPLES=1024 bash scripts/lf/tp_probe.sh q3-30b-a3b cpum8r "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 800000 1
echo "Q4F_EXIT=$?"
echo "=== CPU-Q4 DONE $(date -u +%H:%M:%S)"
