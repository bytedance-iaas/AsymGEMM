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
Canonical implementation: `scripts/lf/ceiling_search.py::confirm_metrics` (`meas[1:-1]` mean),
surfaced as `confirm_steady_step_s` and rendered as the `a` anchor in `scripts/lf/ceiling_table.py`.

Reported alongside `lat`: **C** = peak host RSS (GiB, per-process `/proc/self/status` VmHWM),
**G** = peak reserved HBM (GiB). Comment format: `<lat>s, C-<ram>, G-<hbm>, <next-boundary> [DONE|IP]`.
