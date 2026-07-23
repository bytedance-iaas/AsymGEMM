# RULES

Durable, cross-session rules for this repo's profiling / ceiling-search work.
Keep authoritative — these override ad-hoc choices.

## Steady-state step latency (the `lat` / `s`-per-step number)

When reporting per-step training latency (confirm runs, ceiling anchors, table `a`-values):

- **Warmup steps are ALWAYS excluded** — they never count toward latency.
- Run **4 measured (non-warmup) steps**: `CONFIRM_STEPS`/`MAX_STEPS = 4`, `WARMUP_STEPS >= 1`.
- From the 4 measured steps, **drop the 1st and the last**, then **average the middle 2**.
- That average **is** the steady-state latency.

Requires >= 3 measured steps (3 -> 1 middle sample; **4 -> 2 middle samples, preferred**).

Rev 2026-07-19 (Kevin, recorded in `agent/project_rules.md`): for NEW latency A/Bs,
**1 warmup + 2 measured** (`WARMUP_STEPS=1 MAX_STEPS=2`, steady = mean of the 2) suffices —
measured-step variance is ~1%; saves 25–30% GPU time. Fit/no-fit probes: 1w+1m.
The cpu_compute campaign numbers (through 2026-07-22) were all measured under the
1w+4m/middle-2 protocol above; never mix protocols within one A/B pair.

Canonical implementation: `scripts/lf/ceiling_search.py::confirm_metrics` (`meas[1:-1]` mean),
surfaced as `confirm_steady_step_s` and rendered as the `a` anchor in `scripts/lf/ceiling_table.py`.

Reported alongside `lat`: **C** = peak host RSS (GiB, per-process `/proc/self/status` VmHWM),
**G** = peak reserved HBM (GiB). Comment format: `<lat>s, C-<ram>, G-<hbm>, <next-boundary> [DONE|IP]`.
