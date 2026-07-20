# RESEARCH RULES

> ⚠ SCOPE: generic, project-agnostic research METHODOLOGY only (how to measure, average,
> report). Project-specific protocols, current parameter values, dated decisions, and
> campaign history belong in `project_rules.md` — which OVERRIDES this doc on any
> conflict. DO NOT add project specifics here.

Durable, cross-session rules for profiling / ceiling-search work.
Keep authoritative — these override ad-hoc choices.

## Steady-state step latency (the `lat` / `s`-per-step number)

When reporting per-step training latency (confirm runs, ceiling anchors, table `a`-values):

- **Warmup steps are ALWAYS excluded** — they never count toward latency.
- The step COUNT is a project decision — see `project_rules.md` "Standing measurement law".
- Averaging rule for N measured (non-warmup) steps: **N >= 3 → drop the 1st and the last,
  average the middle; N = 2 → mean of both**. That average is the steady-state latency.

Canonical implementation: `scripts/lf/ceiling_search.py::confirm_metrics` (middle mean for
>=3 measured, plain mean for 2),
surfaced as `confirm_steady_step_s` and rendered as the `a` anchor in
`scripts/lf/ceiling_estimate.py` (named `scripts/lf/ceiling_table.py` in some repos).

Reported alongside `lat`: **C** = peak host RSS (GiB, per-process `/proc/self/status` VmHWM),
**G** = peak reserved HBM (GiB). Comment format: `<lat>s, C-<ram>, G-<hbm>, <next-boundary> [DONE|IP]`.
