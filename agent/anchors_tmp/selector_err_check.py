import glob
import json
import sys

sys.path.insert(0, "scripts/lf")
import asym_scheduler as S

GIB = 2 ** 30


def measured_peak(tag):
    for pj in glob.glob(
        f"profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16/{tag}*/*/b*/profile.json"
    ):
        return json.load(open(pj))["memory"]["peak_reserved_hbm_bytes"] / GIB
    return None


model = S.MODELS["q3-30b-a3b"]
cells = [("T2", 320, 1, "p7a320"), ("T3", 96, 8, "p7a96"), ("T3", 96, 8, "mrg819")]
errs = []
for tier, sk, B, tag in cells:
    line = next(l for l in model.tiers if l.name == tier)
    pred = S.mem_gib(line, sk * B)
    meas = measured_peak(tag)
    if meas:
        e = (pred - meas) / meas * 100
        errs.append(abs(e))
        print(f"{tier}@{sk}k  pred {pred:6.1f} GiB   meas {meas:6.1f} GiB   err {e:+5.1f}%")
# banked replay pair (scheduler's own record): T2B@900k pred 176.9 vs ran 183
errs.append(abs((176.9 - 183) / 183 * 100))
print("T2B@900k  pred  176.9 GiB   meas  183.0 GiB   err  -3.3%  (banked replay record)")
print("mean abs err: %.1f%%" % (sum(errs) / len(errs)))
