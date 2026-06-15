# ARM BF16 SFT Fixes Progress

Current progress state: complete for the strict packed production path.

## Completed

- Native ARM BF16 SFT path simplified to one production algorithm.
- Public cache-mode selector removed.
- Removed path selector flags removed from benchmarks.
- Removed cache-down diagnostics removed from native bindings, Python wrappers,
  benchmark output, and the validation helper.
- Route recomputation helper renamed to describe its current role.
- Empty native placeholder methods deleted.
- Focused tests enforce packed forward and positive LoRA dropout correctness.

## Verified

- Native extension rebuild passed.
- Focused ARM BF16 SFT tests passed.
- Smoke benchmark reported packed forward for short and longer tested qlens.
- Removed public CLI flags are rejected.
- Current optimization validator passed on the smoke output.

## Remaining Watchpoints

- Historical artifact files under `agent/kt/artifacts/` may contain old fields
  because they are past measurements. Do not use them as current command specs.
- Global ktransformers x86 and loader fallbacks are outside this ARM BF16 SFT
  training path and were intentionally left untouched.
