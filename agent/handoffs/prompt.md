# PROMPT — unify the asym scheduler into ONE principled formula (paper-grade)

Owner intent (2026-07-19, polished but unchanged): today the system has three named
operating modes (R1 fallback = `asym|unsloth-ohbm0`+staged; R2 latency = `recomp-off-full-fg`
+ keep-acts stack; R3 memory = `recomp-off-full-fg` defaults + capacity knobs) and a knob
catalog with measured prices. Your task: **formalize ALL of these scheduling decisions into
one unified, principled set of formulas — the kind you would publish in an ML-systems
paper — with explicit thresholds, such that ONE continuous variable moves the whole
system from latency-focused to memory-focused.** This is a
FORMALIZATION of the existing measured system, not a redesign. Keep every existing doc;
extend, don't rewrite.

## Inputs you must build on (all already measured — do NOT re-run)
- `agent/impls/scheduler_v2.md`: §1-3 knob taxonomy + measured knob prices (ΔM, ΔT, the
  ρ = seconds-saved-per-GiB ladder), §4 greedy knob solver, §8 regime layer + validation.
- `agent/impls/fix_asym.md`: mechanism receipts (why R1 exists; recomp-off = recompute-
  OFFLOAD; per-token convergence law; edge tax).
- `agent/impls/s04-p1-dgx-02-c12/concise_throughput_results.md` + `test_throughput_results.md`:
  the validation record (parity band, walls, thrash points).
- Fitted models: memory `resv_c(s,B) = base_c + k_c·B·s` (±2%; under-predicts near walls —
  probe rule applies); throughput `t_c(s) = attn(s) + tax_c` us/tok with attn(s) SHARED
  across configs (proven: rc ≡ uns ≡ R1 per-token at long ctx); SAFE = 0.92·M_phys
  (edge tax measured: llama 192k b2 @97.7% = −4% vs healthy b1).

## The formalization target (required structure — the φ equation)
The user-visible truth: HBM should always be SATURATED as far as safe; the only cap is
physics (SAFE = 0.92·M_phys — above it, allocator churn costs time; measured: 97.7% b2 =
−4% vs healthy b1). There is NO user preference weight. The knob is the SOLUTION variable:

  φ ∈ [0,1] = fraction of offloadable bytes moved off HBM, in MEASURED-PRICE order
  (ρ_k = Δstep-time per GiB offloaded, cheapest first):
    1. base weights (ρ≈0 — staged streaming hidden; R1≡sup parity is the proof)
    2. checkpoint roots (ohbm-N; small ρ, async)
    3. attention saved-tensors (ρ≈0.8 s/GiB at 128k; the KA/R2 delta)
    4. MLP activations (largest ρ; the R3 tail)
  M(φ; s,B) = M₀(s,B) − φ·ΔM(s,B)          (M₀ = all-resident, ΔM = total offloadable)
  T(φ; s)   = T₀(s) + Θ(φ)                  (Θ convex piecewise-linear, slopes = ρ ladder)

THE SCHEDULER IS ONE EQUATION (minimal offload = max saturation = max throughput):

  φ*(model, s, B) = clamp( (M₀(s,B) − 0.92·M_phys) / ΔM(s,B), 0, 1 )
  B* = max{ B : M(φ*; s, B) ≤ 0.92·M_phys }   (batch is capacity-only — convergence law)

"Latency-focused vs memory-focused" is the COMPUTED φ*, not a setting: the workload
(model, s) determines it. The legacy modes are landmarks on φ (weights-only → R1;
+roots → R1+ohbm; +attention → R2/keep-acts; everything → R3/memory).

Required proofs in §9:
- Θ convexity from the measured ρ ladder (greedy price-order optimality).
- Closed-form segment thresholds s* per model from the fitted slopes (dense R1 k≈0.47
  GiB/1k tok → attn-onset ≈350k; KA k≈0.34 → wall ≈480k; MoE memory k≈0.17 → ≈970k) and
  their match to the measured record (384k ran KA; ~490k/~1M walls; R1 parity band below).
- Consistency: for every validated decision (R1 128/160/192k both models; R2 384k/320k;
  R3 640k; B≤0.92; llama-192k b2 edge case REJECTED) the equation must output the same.
- Implementation gap (list, do not code unless asked): segments 3-4 are global booleans
  today; continuous φ inside them = offload only the first ⌈φ·L⌉ layers (offload EARLIEST
  layers first — their backward comes last = most overlap time). ohbm-N already grades
  segment 2.
- Why-not section: reject β-scalarization (non-convex argmin jumps, no user units) and
  reject raw %-offload WITHOUT price ordering (composition across byte classes).

## Deliverables
1. `agent/impls/scheduler_v2.md` §9 "The unified objective": the φ formula set,
   derivations, per-model threshold tables, and one worked example per model (sweep s,
   show φ* rising through the segments latency→memory).
2. A pure function `schedule(model, s, B) → (φ*, B*, knob set)` — pseudo-code in the doc
   plus a small python reference under `scripts/lf/` — implementing exactly the formulas
   (no hidden special cases; knob set = the env flags realizing φ*).
3. Consistency proof against the measured record: the equation MUST reproduce every
   validated decision (R1 at 128/160/192k dense + llama; R2 at 384k/320k; R3/memory at
   640k MoE; B ≤ 0.92 rule; the llama-192k b2 edge case correctly REJECTED). Any
   disagreement = bug in the formalization, not new physics.
4. Honest-limits section: what the linear models cannot capture (allocator churn above
   SAFE, the MoE R1 hole, cross-model transfer of constants, host-RAM ceilings).

## Rules
- No new GPU runs are required for the formalization. If a probe would settle a constant,
  LIST it (model, config, s, B, what it decides) — do not run it.
- Do not change the intent: same modes, same knobs, same measurements — one formula.
- Keep the existing docs intact; this work lands as §9 + the reference implementation.
