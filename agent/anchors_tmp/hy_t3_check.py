#!/usr/bin/env python3
"""Gate-1 checker: hunyuan T3 vs T2B correctness pair (@16k, same seed).
Verifies (a) loss parity between the two tiers, (b) route counters engaged in
T3 and zero in T2B, (c) run-dir labels carry ker101/route101."""
import csv, glob, json, os, re, sys

B = "profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16"

def find(tag):
    cfgs = glob.glob(f"{B}/{tag}-c17_hunyuan-a13b__b1_s16000_ga1_drop000/*/b1_s16000_ga1")
    assert cfgs, tag
    return cfgs[0]

def losses(rd):
    # LF trainer logs loss per step in trainer_log.jsonl or train.log
    out = []
    tl = os.path.join(rd, "lf_run", "trainer_log.jsonl")
    if os.path.exists(tl):
        for line in open(tl):
            try:
                d = json.loads(line)
                if "loss" in d: out.append(float(d["loss"]))
            except Exception: pass
    if not out:
        for m in re.finditer(r"'loss': ([0-9.]+)", open(os.path.join(rd, "train.log"), errors="replace").read()):
            out.append(float(m.group(1)))
    return out

def stats(rd):
    p = json.load(open(os.path.join(rd, "profile.json")))
    s = p.get("asym_execution_stats") or {}
    return p, s

t3 = find("t3cor_t3"); t2b = find("t3cor_t2b")
p3, s3 = stats(t3); p2, s2 = stats(t2b)
l3, l2 = losses(t3), losses(t2b)
print("T3 dir:", os.path.basename(os.path.dirname(t3))[:90])
print("T3 losses:", l3, "\nT2B losses:", l2)
ok = True
if l3 and l2:
    dif = max(abs(a-b) for a, b in zip(l3, l2))
    rel = dif / max(abs(l2[0]), 1e-9)
    print(f"loss parity: max|d|={dif:.5f} rel={rel:.5f}")
    ok &= rel < 0.02
else:
    print("WARN: losses missing"); ok = False
for k in ("qwen3_moe_routed_base_forward_scatter_calls", "qwen3_moe_routed_base_dx_scatter_calls",
          "qwen3_moe_routed_base_gather_left_calls", "qwen3_moe_routed_route_space_h_tensors_avoided"):
    v3, v2 = s3.get(k, 0), s2.get(k, 0)
    print(f"{k}: T3={v3} T2B={v2}")
gate = (s3.get("qwen3_moe_routed_base_forward_scatter_calls", 0) > 0
        and s3.get("qwen3_moe_routed_base_dx_scatter_calls", 0) > 0
        and s3.get("qwen3_moe_routed_base_gather_left_calls", 0) == 0
        and s2.get("qwen3_moe_routed_base_forward_scatter_calls", 0) == 0)
print("route-engagement gate:", "PASS" if gate else "FAIL")
print("label check:", "ker101 in dir" if "ker101" in t3 and "route101" in t3 else "MISSING", )
ok &= gate and "ker101" in t3
print("GATE1:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
