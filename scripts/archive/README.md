# scripts/archive — one-off helpers from the 2026-07 post-merge validation

Preserved for the record; **not for routine use**. Each hardcodes an OUTPUT_ROOT /
env for a specific validation run recorded in `agent/impls/fix_merged.md` §6
(the ledger row is the authority on what each run measured and its verdict).

The `run*.sh` helpers invoke `scripts/lf/...` by RELATIVE path — run them from the
repo root (`bash scripts/archive/run45k_hostsync2.sh`), inside the `asym_sft_39`
container, GPU 0, one experiment at a time (rules: fix_merged.md §7a).

| Script | What it was |
|---|---|
| `run30b_postmerge.sh` | T4 — qwen3-30b cross-model control (closed B1) |
| `run120k_postmerge.sh` | T5 — 120k long-seq headroom + pin-fallback check (closed D4) |
| `run45k_hostsync.sh` | T8 — 45k validation of the 30-site `wait_cpu_ready_host` sweep |
| `run45k_hostsync2.sh` | T9 — 45k re-validation covering F13–F21 (surfaced the F14 firing race) |
| `run45k_hostsync2_rep.sh` | T9b — reproducibility of T9 (confirmed the F14 loss shift stable) |
| `chain_probe_then_B.sh` | P5 + T6/T7 — nocache probe discriminator, then B baselines 45k/80k (closed B2) |
| `race_invitro_test.py` | P7 — in-vitro proof of the V1/V2 host-read races and of the fix (TESTs B/A/C1/C2) |
| `in39_noninteractive.sh` | HOST-side non-interactive entry into the `asym_sft_39` enroot container (no-op when already inside; see fix_merged.md §7a rule 1) |
