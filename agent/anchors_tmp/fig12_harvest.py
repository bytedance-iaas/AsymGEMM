#!/usr/bin/env python3
"""Harvest fig12 A/B cells: eff tok/s, per-step ms, loss, engagement counters.

Usage: fig12_harvest.py TAGGLOB [TAGGLOB...]
e.g.   fig12_harvest.py 'kf??_q3-30b*' 'kg*_glm*'
"""
import csv, glob, json, os, re, sys

B = "profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16"
pats = sys.argv[1:] or ["kf*"]
rows = []
for pat in pats:
    for d in sorted(glob.glob(f"{B}/{pat}")):
        tag = os.path.basename(d)
        for ss in glob.glob(f"{d}/*/b*_s*_ga*/step_samples.csv"):
            rd = os.path.dirname(ss)
            m = re.match(r"b(\d+)_s(\d+)_ga(\d+)", os.path.basename(rd))
            if not m:
                continue
            b, s, ga = map(int, m.groups())
            steps, losses = [], []
            for row in csv.DictReader(open(ss)):
                warm = str(row.get("is_warmup", "")).strip().lower() in {"true", "1", "yes"}
                ms = float(row.get("step_milliseconds") or 0)
                if ms > 0 and not warm:
                    steps.append(ms)
                    try:
                        losses.append(float(row.get("loss")))
                    except (TypeError, ValueError):
                        pass
            if not steps:
                continue
            eff = (len(steps) * b * s * ga) / (sum(steps) / 1000.0)
            eng = {}
            spf = os.path.join(rd, "source_profile.json")
            if not os.path.exists(spf):
                spf = os.path.join(rd, "source_profile.partial.json")
            if os.path.exists(spf):
                try:
                    st = json.load(open(spf)).get("asym_execution_stats", {})
                    for k in ("cpu_left_lora_a_calls", "attn_act_lora_a_shared_batches",
                              "expact_lora_a_forward_cpu_left_grouped_calls",
                              "expact_lora_a_grad_grouped_calls",
                              "qwen3_moe_finegrained_lora_a_forward_calls"):
                        v = st.get(k)
                        if v:
                            eng[k.replace("expact_lora_a_", "").replace("cpu_left_lora_a_calls", "K1").replace(
                                "attn_act_lora_a_shared_batches", "sharedB").replace(
                                "forward_cpu_left_grouped_calls", "fgK1fwd").replace(
                                "grad_grouped_calls", "K2grad").replace(
                                "qwen3_moe_finegrained_lora_a_forward_calls", "fgLoraFwd")] = v
                except Exception:
                    pass
            # reaim engagement from train.log markers
            tl = os.path.join(rd, "train.log")
            n_reaim = 0
            if os.path.exists(tl):
                n_reaim = sum(1 for line in open(tl, errors="ignore") if "[asym-reaim] ENGAGED" in line)
            rows.append(dict(tag=tag, b=b, s=s, steps=[round(x / 1000, 1) for x in steps],
                             eff=round(eff), loss=[round(x, 4) for x in losses], reaim_sites=n_reaim, eng=eng))
for r in rows:
    print(f"{r['tag']:44} b{r['b']} s{r['s']:>7} eff={r['eff']:>6}  steps(s)={r['steps']}  loss={r['loss']}  reaim_sites={r['reaim_sites']}  {r['eng']}")
