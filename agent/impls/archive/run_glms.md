- [08-09] cross-note: TRUE T3 (ker101 + TIER_ENV) now works on both GLMs —
  see agent/impls/fix_glm_t3.md (gate unlock + attn LoRA-A safe-path fix;
  smoke: Flash 1r/2r + Air 1r TRAINED, Air 2r honest host-COOM).
- [2026-08-09 ~21:0x] c14: sEP(sepplan2) DIAGNOSIS CLOSED — NO-GO for
  GLM-Air/GLM-Flash/Mixtral; banked sdp2 2r rows stand as optimal. Evidence:
  (1) sdp2-floor parity proven (T1 smokes ±0.9% of banked); (2) with the
  hook actually live (T2 + ASYM_GEMM_DISPATCH=asym override — tier recipes
  pin staged, which bypasses the sEP hook entirely; the qwen sEP campaign
  predates those recipes), arm rate = 0/1518 (Flash) and 0/1755 (Air, arena
  400): EVERY launch declines on the streaming-bound rule (fat segments),
  and direct dispatch is itself slower (Flash 3049 vs 3414 staged). Mixtral
  (8 experts, coarsest) not pursued past an unrelated repo-id smoke quirk —
  guaranteed-decline by geometry (Scout precedent). Ops lessons banked:
  clean /dev/shm/asym_fabric_* before ANY direct 2r invocation (full shm =
  NCCL "unhandled system error"); Air T2 fabric needs cap ~400. REVERSE
  RUNNER: STAND DOWN — nothing to run.
