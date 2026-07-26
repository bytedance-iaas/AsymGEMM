# Capacity push predictions — REGISTERED BEFORE MEASUREMENT (2026-07-25)

Diagnosis being exercised: torch CachingHostAllocator rounds every pinned block
to the next pow2 (measured 1.41x on odd sizes; roots @128k/h9216 2.36GB→4GB =
1.70x); asym host burden is ~all pinned ⇒ the historic "2x-weights steady tax"
was W×pow2 + roots×pow2, not physics. Fixes: exact-size cudaHostRegister homes
(ASYM_EXACT_PINNED[_ROOTS]) + HBM-budget root parking (UNSLOTH_GC_OUTER_HBM_AUTO,
reserve 20 GiB) + restored flush8 hook. Wall = 905 GiB unevictable, floor 25.

Also on record: the 8b 64k/32k asym results are VOID (classifier rc=0 bug +
invalid-GQA h=12800 builds crashed pre-training; every asym 64k/32k "OK" was a
load footprint). Valid asym baselines: 128k crown 217.948B OK 865.4 (R11);
64k only ☠237.255 (G2, mid-forward). SO 64k: 217.9 OK 870.0 / ☠228.3 (load wall).

| cell | config | predicted host peak (GiB) | predicted verdict |
|---|---|---|---|
| V1 98B/128k T1+flush8 ohbm0, fixes OFF | pre-fix repro | 699 ± 15 (gate: toolkit+sparse-build ≡ campaign R3 698.8) | OK |
| V2 = V1 + exact-pinned | −W-tax 30 − roots-tax 134 − opt-tax ~10 | **525 ± 25** | OK |
| V3 = V2 + auto-park r20 | − ~41 roots × 2.2 GiB to HBM | **435 ± 30** (G ~165) | OK |
| X1 217.948B/128k full fix set | vs R11 865.4: −53 W-tax −76 roots −15 misc | **~720 ± 30** | OK |
| X2 233.249B/128k | X1 + 2.9-slope × 15.3B... W+28.5 roots+23 opt+4 | **~776 ± 35** | OK |
| X3 248.550B/128k | | **~833 ± 40** | OK |
| X4 263.851B/128k | | **~890 ± 45** | edge |
| X5 279.152B/128k | | ~946 | C_OOM |
| Y1 233.249B/64k | W 434.5 + roots-on-host ~72 + opt/misc ~105 | **~610 ± 40** | OK |
| Y2 263.851B/64k | | **~700 ± 45** | OK |
| Y3 294.453B/64k | | **~795 ± 50** | OK |
| Y4 309.754B/64k | | **~845 ± 55** | edge |

Notes: slope model after fixes (128k, 12288-family) ≈ W 1.863 + roots-exact
(2.93 GiB/L ÷ 1.913 B/L, minus ~fixed parked count) + opt ~0.15 ≈ 3.6 GiB/B
LEVEL-shifted down ~145 GiB vs pre-fix; at 64k roots halve ⇒ ~2.8 GiB/B.
Track record caveat: chain-D-era predictions went 0-for-3 on SO; these carry
wide bars until V-phase closes.
