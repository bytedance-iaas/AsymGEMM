# Readable extraction of b3bf7ddd-f2f1-4682-8306-cc64b601d1c6.jsonl
(user prompts + assistant text only; tool calls/results omitted)

---

## USER  [2026-07-27 05:54Z]

i am working on the writing for '/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/agent/overleaf/[MLSys 26 Sub] Superchip-based LoRA' we hve the titles and rheotics in '/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/agent/overleaf/[MLSys 26 Sub] Superchip-based LoRA/drafts/motivation_v2.md' and '/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/agent/overleaf/[MLSys 26 Sub] Superchip-based LoRA/drafts/system_writing_v2.md' and the actuak writing in '/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/agent/overleaf/[MLSys 26 Sub] Superchip-based LoRA/drafts/motivation_full_v2.md' ajd '/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/agent/overleaf/[MLSys 26 Sub] Superchip-based LoRA/drafts/system_writing_full_v2.md' adnte needplots/micorbencarks in '/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/agent/overleaf/[MLSys 26 Sub] Superchip-based LoRA/drafts/motivation_v2_plots.md'. these ar ehte background.  help me make sure that 1. the titles and thetpriocs be used as the story line 2. the '/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/agent/impls/s04-p1-dgx-02-c06/motivation_full_v2.md''/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/agent/impls/s04-p1-dgx-02-c06/system_writing_full_v2.md' these wiring coten needds to following the rhetorics. 3. and that the changes as synced over to the overleaf.chekc andreaosn extensivelt nad elt eknow

### ASSISTANT  [2026-07-27 05:54Z]

I'll start by locating and reading all the files you mentioned — the rhetoric/outline files, the full drafts (both copies), the plots file, and the Overleaf tex sources — then check the three things you asked about.

### ASSISTANT  [2026-07-27 05:55Z]

I've read the motivation outline and full prose. Now the system pair and the tex files.

### ASSISTANT  [2026-07-27 05:55Z]

Now the two tex files in the Overleaf project.

### ASSISTANT  [2026-07-27 06:05Z]

I've finished a full audit of both storylines, both full drafts, both tex sections, the newest plots/microbench file, the figure PDFs, and the Overleaf git state. Overall verdict: **the writing follows the rhetorics faithfully and the md→tex sync is near-verbatim, but the 2026-07-26 evening figure pass (the 5-point sweeps and the fig-7 rework) moved ahead of the §2.4.3 prose, the M4 figure rename is half-applied, and the last commit was never pushed to Overleaf.** Details below, organized by your three questions.

## 1. Are the titles + rhetorics used as the storyline? — Yes, verified unit by unit

`motivation_full_v2.md` follows the DirectKV-shaped restructure of `motivation_v2.md` exactly: definitions-only §2.1 → incumbent systems + premise voided twice (M0a/M0b only, M0c/M0d idleness argued not plotted) → §2.3 endorsement + three structural commitments + accounting consequence, no microbench → five opportunity units, each with microbench numbers in prose, a named trade-off, and a bold **Takeaway:**. All the retired figures (M0c, M0d, M1, F5, standalone M4) are uncited. The documented "measured adaptations" (staging is link-bound not ceiling-tracking; scatter fusion output-side only with the backward's +12% stated as the price; M3's attention peak win relocated to link+host bytes; 7.2× replacing the banked 13×; SwiGLU disproven → one-sentence status note; M5a folded) are all honored. §3 likewise: WHAT/HOW only, one-clause whys with §2 pointers, no metric figures (F5 correctly commented out in tex), section order offline→online, the sEP computed cut presented as shipped with raced as the counts-free fallback, floor = the data-parallel split, Llama-4-Scout scoping present.

Two staleness issues **in the rhetoric files themselves** (both predate the 14:11 measured update, despite the resync header):
- `system_writing_v2.md` §3.2 still says the CPU bullet ships **three** kernels including SwiGLU, contradicting its own A.3a audit (❌ REMOVE) and the full/tex ("Two such kernels ship"); its adapter bullet also still says "TWO GPU streaming kernels / CPU kernel described ONLY in the CPU bullet," while the full numbers three adapter kernels and introduces (3) in the adapter paragraph.
- `motivation_v2.md`'s back-matter M2a/M4b entries still describe the old 2-bar layouts and the banked 13× (superseded by the plots file — by design, but know they're behind). The impls plots file also lost its "## TABLE — CPU scope" heading (the SPEC block now dangles at the end of the M3 entry) and still lists the SwiGLU table row, though the shipped 3-row table is right.

## 2. Does the full writing follow the rhetorics? — Yes, but the evening figure pass left the prose behind in three places

**P0 — §2.4.3 (M4b) body is stale against the regenerated 5-point figure.** The figure/caption (commit 545aab6) now say: crossover **between 512K and 1M**, the ≤262K gate is *conservative*, stock collapses at 512K, speedups up to 60×. The body (md **and** tex, identical) still tells the old 3-point story: "1.8×, 2.4×, and 31× faster (12.3 versus 21.6 ms, 33.3 versus 79.8, and 137 versus 4336)… crossover falls **between 256K and 1M** rows… the 256K point, **the last where it wins**." The new 512K point (CPU +5.16 vs GPU +10.45 ms — CPU still 2× better) directly falsifies "the last where it wins," and none of the quoted serial numbers match the plotted ones (new: 8.35/17.1, 16.0/36.4, 31.7/67.8, 59.5/3422, 116/6913). The caption's "1.8–60×" low end is also stale (new low is 2.0×). One more thing to verify on the rendered figure: the plots file records an ingestion sidebar of 12.5 vs 51.1 (4.1×) while the prose uses the fp32-fair 12.1 vs 86.7 (7.2×) — if that sidebar made it into panel (b), prose and figure will visibly disagree.

**P1 — §2.4.1 (M2a) body quotes the superseded 2-point run.** The figure is now a 5-point line plot to 512K (caption says so); the body still frames it as "at 32K and 256K token rows" and quotes 0.402/13.080/12.701 where the plotted sweep has 0.399/13.086/12.699 (trivial drift, but a reviewer comparing axis ticks to prose will notice the framing). The 30× residency premium survives at 512K (26.34/0.863 ≈ 30.5×), so the claim structure is safe — it just needs re-anchoring.

**P2 — §2.4.5 vs the reworked fig 7.** The body says "We measure **four** executions… (Figure M5b)" but the figure now plots **three** series (DP / EP / Dynamic-EP; the raced cut is prose-only), and the caption says "three executions." Also "(Figure M5a)" is cited for the *per-step* idling claim, but the merged figure shows imbalance vs. skew z, not per-step variation (M5a was folded); and the MoE-block carry-through now lives in the appendix per the caption. The reviewer-safe traffic framing is correctly obeyed (no "EP has less traffic" claim in prose). These need small prose touches, not new data.

Everything else checked out: M3 numbers/framing match the measured disposition exactly, M6 matches (2063/1524/698; 327 vs 264; 172 of 184 GiB; OOM marker), M2b matches (−13%/+12%, 8.59 GB/leg), sEP numbers match the C7-confirmed set, §3.4's "5.2 versus 1.3 ms, four times" is still consistent with the new 256K data within rounding.

## 3. Is it synced over to Overleaf? — Three gaps

1. **Unpushed commit.** The local clone is **1 commit ahead of origin** — 87dabc8 (the fig-7 caption with the verified re-stream mechanism) never made it to Overleaf. Working tree is otherwise clean. A `git push` fixes it.
2. **Tex is ahead of its md sources in two spots** (so the impls/drafts "ground truth" no longer matches the paper): "as a case study" → "as the representative microbenchmark" (Kevin's rule) and the F5 figure note (still body text in the md, commented out in tex). Both md copies (impls + drafts, currently identical) should be back-ported.
3. **Stale drafts in the Overleaf repo.** `drafts/` is git-tracked, and `drafts/motivation_v2_plots.md` (missing Kevin's binding feedback block, the 5-point measured records, the M5b-row3 final form) and `drafts/toconfirm.md` (SwiGLU still "pending confirmation" vs. DISPROVEN) are behind the impls copies.

**Kevin's binding feedback scorecard:** (1) no-"case-study" — captions/prose clean and M2a file renamed ✓, but the tex still includes the **old** `m4-crossover.pdf` instead of `m4b-grace-placement.pdf` (byte-identical twins; both old-name files `m2a-kernel-case-study.pdf`/`m4-crossover.pdf` are still tracked — swap the include and delete the old names), and the md's one remaining "case study" is the back-port above. (2) height standard — all four §2.4 microbench PDFs plus m6 now share an identical MediaBox (712.8×222.48 pt), so his verification criterion passes; note it's uniform-canvas-scaled rather than literal 2.0 in native size (rendered ≈1.0 in at column width, uniform across all five). (3) M2b time+memory ✓. (4) table marks \checkmark/\times, no dashes ✓. (5) M3 side-by-side ✓.

Smaller items while you're in there: the paper title says "…of **LLMs**…" while the adopted working title says "…of **Large Models**…"; gate C5 is still open (AsymLoRA.pdf modules in old order, and the overview caption calls the Profiler "the scheduler" — exactly the name clash gate 2 warns about); the stale `\motivationplot`/m5a comment atop motivations.tex; `m5a-imbalance.pdf` tracked but unused; and the expected bracket inventory still pending ([8] TB/s, [500] GB/s ×3, [N] ×2, the knee bracket, §3.3's [k]/[two]/[j]/[a few], [X]/[Y] KLOC, [stack], [and device rounding], two [verify] markers). Abstract and intro are still stubs.

If you want, I can apply the mechanical fixes now — rewrite §2.4.3/§2.4.1 to the 5-point data (md + tex), the §2.4.5 prose touches, the m4 include rename + old-file removal, the md back-ports, the drafts re-sync, and push the lot to Overleaf — and leave the judgment calls (title wording, fig-7 four-vs-three framing choice, C5 figure relabel) for you.

---

## USER  [2026-07-27 06:06Z]

concsielt let me know

### ASSISTANT  [2026-07-27 06:07Z]

**Storyline/rhetoric: passes.** Both full drafts follow the v2 outlines faithfully (structure, delegation rule, Takeaways, retired figures uncited, measured adaptations honored). Only staleness: `system_writing_v2.md` §3.2 still says three CPU kernels incl. SwiGLU (truth: two) and two GPU adapter kernels (truth: three).

**Prose vs. figures: 3 mismatches** (figures moved ahead of prose last night):
1. **§2.4.3 M4b — worst one.** Body (md+tex) still has the old 3-point story: "crossover between 256K and 1M… 256K the last where it wins… 1.8×/2.4×/31×". Figure/caption now show the 5-point sweep: crossover **512K–1M**, ≤262K gate merely *conservative*, 2.0–60×, stock collapses at 512K. The 512K point falsifies "last where it wins."
2. §2.4.1 M2a: body says "at 32K and 256K rows" with old 2-point numbers; figure is now a 5-point line plot to 512K.
3. §2.4.5: body says "four executions (Figure M5b)" but fig 7 plots three (raced cut is prose-only); "(Figure M5a)" cited for a per-step claim the merged figure no longer shows.

**Overleaf sync: 3 gaps:**
1. Local repo is **1 commit ahead** — fig-7 caption commit (87dabc8) not pushed.
2. Tex is ahead of the md sources in 2 spots ("representative microbenchmark" wording, F5 comment-out) — back-port to md.
3. Tracked `drafts/motivation_v2_plots.md` + `toconfirm.md` are stale vs. the impls copies (missing Kevin feedback block, 5-point records, SwiGLU DISPROVEN).

**Kevin's binding feedback:** 4 of 5 done. Half-done: tex still includes old `m4-crossover.pdf` instead of `m4b-grace-placement.pdf`; old-name duplicate files still tracked.

Minor: title says "LLMs" vs. adopted "Large Models"; gate C5 (system-figure renumber, caption calls Profiler "the scheduler") still open; abstract/intro stubs; usual bracket placeholders pending.

Want me to apply the mechanical fixes (rewrite §2.4.3/§2.4.1/§2.4.5 prose to the new data, m4 rename, md back-ports, drafts re-sync, push)?

---

## USER  [2026-07-27 06:07Z]

ok fix the proe vs figures

---

## USER  [2026-07-27 06:07Z]

ok fix the proe vs figures things

### ASSISTANT  [2026-07-27 06:10Z]

I'll fix all three prose-vs-figure mismatches in the md sources (impls + drafts) and the tex, then commit and push. First, one verification: whether the m4 figure actually renders an ingestion sidebar (which would affect the 7.2× prose number).

### ASSISTANT  [2026-07-27 06:12Z]

Two system edits landed; the M5b hunk in the motivation md didn't match — let me re-read that passage for exact wrapping.

### ASSISTANT  [2026-07-27 06:14Z]

Quick re-verification that md and tex are still verbatim-synced after the edits, then commit and push:

### ASSISTANT  [2026-07-27 06:14Z]

Sync verified — the only remaining md↔tex divergence is formatting plus the known F5 comment-out. Committing and pushing:

### ASSISTANT  [2026-07-27 06:14Z]

Done — all three prose-vs-figure mismatches fixed and pushed to Overleaf (`b6bf1ae`; branch now even with origin, so the earlier unpushed fig-7 caption is up too).

What changed (applied identically in `sections/*.tex`, `impls/.../`, and `drafts/`):

1. **§2.4.3 M4b** — rewritten to the 5-point data: sweep "64K to 1M rows", serial vs stock now "2–2.3× through 256K, 58–60× beyond (59.5 vs 3422 at 512K; stock collapses, ours linear)", busy-context adds now 1.24/1.30 (64K), 1.36/5.17 (256K), 5.2/10.5 "still half" (512K), 57.7/20.7 (1M, 116 ms no longer fits the window). Crossover now "between 512K and 1M"; the false "256K, the last where it wins" replaced with "arms through 262,144 rows, conservatively inside it". Caption low end 1.8×→2×, and "adds ~1.3 ms **through 256K rows**". §3.4 echo updated (5.2 vs 1.4 ms).
2. **§2.4.1 M2a** — reframed to the sweep: "swept from 32K to 512K token rows", endpoints quoted (0.053/1.652/1.598 at 32K; 0.863/26.342/25.388 at 512K), "exactly linear between", staged copy 5.4 GB at 512K, "30× gap steady across the sweep". Also picked up the "representative microbenchmark" wording in the md (was the last "case study").
3. **§2.4.5 / fig 7** — the M5a cite now points at what the panel shows ("hot GPU carries up to 56% excess routed rows, Figure M5a left"); "four executions" kept but the figure cite now says it plots the three series (raced stays prose); MoE-block carry-through notes the appendix instead of citing the main figure; §3.5's "Figure M5b is the measured basis for preferring the computed cut" → "the measurements of §2.4.5 prefer…".

Not touched (per scope): m4 file rename in the include, F5 leftover in the system md, stale skeleton spots, title wording.

---

## USER  [2026-07-27 06:16Z]

did u uda tto overleaf and sycn to its remote?

### ASSISTANT  [2026-07-27 06:16Z]

Yes — verified just now after a fresh fetch: the fix commit `b6bf1ae` was pushed to the Overleaf remote (git.overleaf.com), local and `origin/main` are identical (no ahead/behind, zero diff), and the previously unpushed fig-7 caption commit `87dabc8` is on Overleaf too. The Overleaf project is fully up to date; changes will appear next time you reload the project in the Overleaf editor.

---

## USER  [2026-07-27 06:20Z]

for the systme desing part we hve a few components 1. kernls 2. profiler 3schueler. 4. dynaic xper laod balacner. i oik the nsame of dynaci expert load balcner BUT the other three their names are tooo nondescotive i dotn need a fancy jargpnnae BUT i liek names suhv as dynaci eper load balacneer this ttpe of name hwihc is descotoige and tells the reders wh tat this modiule kinda try to do alreayd jsut from its name

---

## USER  [2026-07-27 06:22Z]

[Request interrupted by user]

---

## USER  [2026-07-27 06:22Z]

let  eknow wht are some btternaming fthat i can have for those componetns

### ASSISTANT  [2026-07-27 06:23Z]

Here are options, in the same plain-descriptive style as "Dynamic Expert Load Balancer" (agent noun that states what it does to what). Recommended set first — it keeps the offline/online pairing visible in the names:

| Component | Recommended | Alternatives |
|---|---|---|
| 1. Kernels | **CPU-Resident-Tensor Training Kernels** | Host-Memory Training Kernels · Streaming Training Kernels |
| 2. Profiler | **Offline Configuration Selector** | Record-Driven Configuration Selector · Offline Configuration Planner |
| 3. Scheduler | **Runtime Placement and Kernel Dispatcher** | Runtime Tensor-Placement Dispatcher · Per-Step Placement Dispatcher |
| 4. Balancer | **Dynamic Expert Load Balancer** (keep) | — |

Reasoning, briefly:

1. **Kernels** — the distinguishing fact is *what they compute against* (tensors that stay in CPU memory, no staging copy), so put that in the name. It also matches your §3.2 title "Training Kernels for CPU-Resident Tensors" almost verbatim, which your own style rule wants (figure box ↔ subsection title lexical match). "Streaming Training Kernels" is shorter but slightly undersells the CPU-side kernels, which don't stream — they compute in place.

2. **Profiler** — "Selector" is more honest than "Profiler": §3.3's whole selling point is that it *never profiles before launch*, it answers from records. So a name containing "profiler" mildly fights your own claim. "Offline Configuration Selector" says when it runs and what it emits; "Record-Driven Configuration Selector" says *how* instead, which is your §3.3 thesis — pick whichever half you'd rather advertise.

3. **Scheduler** — "Runtime Placement and Kernel Dispatcher" is just the agent form of your current §3.4 title, and it names both halves of the job (enact placements, route operators to kernels). Bonus: it avoids "Scheduler" entirely, which your gate C2 notes flag as a name clash with upstream AsymGEMM's kernel-internal `asymScheduler`.

Two practical notes if you adopt any of these: first-use gets the full name, then a short handle in running prose ("the kernels", "the Selector", "the Dispatcher") — that's the DirectKV/SuperOffload pattern; and the AsymLoRA.pdf figure boxes need relabeling to match whatever you pick (that's already pending as gate C5, together with the module renumber). Want me to apply a chosen set across §3, the overview walk, and the figure-gate notes?

---

## USER  [2026-07-27 06:28Z]

1. needs tpo be desctive but also nottt toooo long right? tehy need to fi tinotnhe system desing graphics nicely BUT be more descoptive.  2. also chekc the ssyted sing graphs the kernles cotnais vaiorus vkernles to chooe from inckue the one we devleopde and the ones alreadhy presetn the cublas ones.

---

## USER  [2026-07-27 06:28Z]

1. needs tpo be desctive but also nottt toooo long right? tehy need to fi tinotnhe system desing graphics nicely BUT be more descoptive.  2. also chekc the ssyted sing graphs the kernles cotnais vaiorus vkernles to chooe from inckue the one we devleopde and the ones alreadhy presetn the cublas ones. let enow the etter namigns.

### ASSISTANT  [2026-07-27 06:30Z]

Decoded the figure's text (glyph-ID encoded, so my earlier greps missed it). Verified contents — and yes, your point 2 checks out: the **Kernels box already draws a portfolio of three selectable execution targets: "AsymLoRA GEMM" (ours), "cuBLAS GEMM" (stock resident), "Grace SVE2" (CPU)**, alongside HBM/SMEM/Weights/Activations/Staging/Streaming in the AsymScheduler box, User Inputs → Memory Predictor → Optimal System Config in the Profiler box. Current box titles are: Kernels · AsymScheduler · Dynamic Expert Balancer · Profiler. Longest existing box title is 23 chars ("Dynamic Expert Balancer"), so that's roughly your length budget.

Naming proposals sized for the figure (box name ≤ ~25 chars, with the matching §3 subsection title):

| Box (figure) | chars | §3 title | Why |
|---|---|---|---|
| **Training Kernel Portfolio** | 25 | "Training Kernels for CPU-Resident Tensors" (keep) | "Portfolio" says *a set you choose from* — exactly what the box draws (ours + cuBLAS + Grace), and §3.1 already says "the kernel portfolio", so it's lexically free. Alternatives: Training Kernel Pool (20), GPU+CPU Training Kernels (24). |
| **Configuration Selector** | 22 | "Offline Configuration Selection" | Says what it emits. "Profiler" also quietly fights your §3.3 claim that no profiling sweep precedes launch. Alternatives: Offline Config Selector (23), Record-Driven Selector (22). |
| **Placement Dispatcher** | 20 | "Runtime Placement and Kernel Dispatch" (keep) | Covers both halves (enact placements, dispatch operators); drops the upstream-clashing "AsymScheduler" name (your gate C2). If it fits: Placement & Kernel Dispatcher (29). |
| **Dynamic Expert Load Balancer** | 28 | "Dynamic Expert Load Balancing" (keep) | Adding "Load" to the current box also closes the C5 box↔title lexical mismatch you already flagged. If 28 is too wide, current "Dynamic Expert Balancer" (23) is fine. |

Two figure-side suggestions while you're relabeling (this is the pending gate C5 pass anyway): rename the three inner kernel labels to match the §3.2 closer verbatim — **"Streamed AsymLoRA GEMM" / "Resident cuBLAS GEMM" / "Grace CPU Kernels (SVE)"** — so the choose-among-targets reading is explicit in the box itself; and do the ②↔④↔③ renumber at the same time (figure order is still Kernels→AsymScheduler→Balancer→Profiler; the paper now walks Kernels→Selector→Dispatcher→Balancer).

My pick as a set: Training Kernel Portfolio · Configuration Selector · Placement Dispatcher · Dynamic Expert Load Balancer — parallel, plain-vocabulary, all within the box budget. Tell me the set you want and I'll propagate it through §3 titles/prose, the overview walk, and the gate notes.

---

## USER  [2026-07-27 06:43Z]

the box inthe fitue dn the title neds to be consiten tho ....otherwise it involvess ambuigty an confuse sht reader ,., what aout 1. kernel repos 2. configuration  slector 3. not sure yet 4, dynaic exert load balcner? forpalcement dspathcer it osunds a bit werid is ther som better nanig casue this wil eecutor the kenrle and the runtime right iw as thigni about schudelr but is tehre os betterbaming? let kenow

### ASSISTANT  [2026-07-27 06:44Z]

Agreed on box = title exactly (that's also your own style rule D). On your picks:

**1. "Kernel Repos" — I'd avoid "repo(s)".** It reads as a git/code repository and is informal on a paper page. The standard systems word for "a collection of kernels you call into" is **library** — and it buys you a nice echo: §3.2's lead already contrasts with "the AsymGEMM **inference library**", so naming ours the **Training Kernel Library** (23 chars) lands the inference→training shift in the name itself. Alternatives if you want the choose-from flavor: Training Kernel Portfolio (25), Kernel Repository (17, still code-flavored).

**2. Configuration Selector** (22) — good, keep. (Optionally "Offline Configuration Selector" (30) if you want the offline half stated; the shorter one fits the box better.)

**3. Dispatcher/Scheduler — the semantics argue against "Scheduler".** A scheduler *decides* what runs when; here the decision was already made offline by the Selector — this module *executes* it each step (enacts placements, routes each operator to the kernel where its operands sit). Plus upstream AsymGEMM ships a kernel-internal `asymScheduler`, a name clash your gate C2 already flags. Better fits:

- **Placement Runtime** (17) — my pick. "The runtime" is standard for the online execution component, and the sentence reads naturally: "the Placement Runtime enacts each placement and dispatches each operator…". Pairs cleanly with the offline "Configuration Selector".
- **Placement Executor** (18) — most literal "it executes" name.
- **Runtime Dispatcher** (18) — keeps your current prose word "Dispatcher", loses "placement".
- **Placement Engine** (16) — fine, slightly vaguer.

**4. Dynamic Expert Load Balancer** (28) — keep; adding "Load" to the figure box also fixes the existing box↔title mismatch.

Recommended set (box = subsection title, identical strings):

1. **Training Kernel Library** · 2. **Configuration Selector** · 3. **Placement Runtime** · 4. **Dynamic Expert Load Balancer**

All ≤28 chars, plain vocabulary, parallel, and each says what the module does. If you confirm (especially #3), I'll propagate: §3.2–§3.5 titles, the §3.1 figure walk, running-prose handles ("the Library / the Selector / the Runtime / the Balancer"), §2's forward pointers, and update the gate-C5 relabel notes for the figure pass.

---

## USER  [2026-07-27 06:50Z]

1. wha taout Kernel Pool? Knrle lirbaru soinds ooddd too bette ramign? 3, this aneign canbe a bir more detialef what shul it bve> lkiek how to make readers knwon this moduel will coner the compteu and storage/palemtn in the runtime? ektnkewo>?

### ASSISTANT  [2026-07-27 06:50Z]

**1. "Kernel Pool" works — keep "Training" in front: "Training Kernel Pool" (20 chars).** "Pool" is plain, standard systems vocabulary (thread pool, memory pool) and says "a set you draw from," which is exactly what the Runtime does with it. "Training" is worth keeping because the kernel novelty of the paper *is* the inference→training shift. If you want other options: Training Kernel Suite (21), Training Kernel Portfolio (25) — but Pool is fine; prose handle "the kernel pool" reads naturally.

**3. To say "this module handles both the compute and the storage/placement, at runtime," name both halves plus the phase:**

- **Placement and Dispatch Runtime** (30; box can print "Placement & Dispatch Runtime", 28) — my recommendation. The three words map exactly: *Placement* = the storage half (staging / streaming / offloading / recomputation — where tensors live and how they move); *Dispatch* = the compute half (routing every operator to the kernel of the device holding its operands — streamed AsymLoRA, resident cuBLAS, or Grace CPU); *Runtime* = it acts per step, online, executing what the Selector decided. It's also nearly your current §3.4 title ("Runtime Placement and Kernel Dispatch") with the words reordered into an agent name, so almost nothing else in the prose has to move.
- Placement and Execution Runtime (30) — if you prefer "execution" over "dispatch"; slightly less accurate, since the module routes work rather than doing the FLOPs itself.
- Tensor Placement and Kernel Dispatch (35) — most explicit, but too long for the box.
- Memory and Compute Runtime (25) — shortest both-halves name, but vaguer about *what it does* with memory and compute.

So the full set would be:

1. **Training Kernel Pool** · 2. **Configuration Selector** · 3. **Placement and Dispatch Runtime** · 4. **Dynamic Expert Load Balancer**

— identical strings for figure box and subsection title, prose handles "the kernel pool / the Selector / the Runtime / the Balancer." Confirm #3 (and Pool vs. one of its alternates) and I'll propagate through §3 titles, the §3.1 walk, §2 forward pointers, and the gate-C5 figure-relabel notes.

---

## USER  [2026-07-27 07:00Z]

Placment adn dsiaohc runtie sounds ood

### ASSISTANT  [2026-07-27 07:00Z]

Fair — try **Configuration Executor** (22 chars).

The pairing does the explaining: the **Configuration Selector** picks the configuration offline; the **Configuration Executor** executes it every step. And "executing the configuration" *is* both halves you want conveyed — a configuration assigns every tensor its placement/transport (storage) and thereby which kernel runs each operator (compute). Your §3.4 opening sentence already says literally this: "The Dispatcher **executes** each training step **under the configuration** the Profiler **selected**" — so the name is just that sentence compressed, and the Selector→Executor arrow in the figure becomes self-explanatory.

Fallbacks if you don't like the twin "Configuration …" prefixes: Runtime Executor (16), Placement Executor (18), Runtime Execution Engine (24).

Full set: **Training Kernel Pool · Configuration Selector · Configuration Executor · Dynamic Expert Load Balancer** — handles in prose: the Pool, the Selector, the Executor, the Balancer.

Say the word and I'll propagate (§3.2–3.5 titles, §3.1 walk, running prose, §2 pointers, gate-C5 figure notes).

---

## USER  [2026-07-27 07:05Z]

OCNidgu execute is a bit weird tho is tehr a bter name? thismoduek will resonble for coptue ad storage/placmeent?

### ASSISTANT  [2026-07-27 07:05Z]

Options that name *both* responsibilities and still sound like normal systems English:

1. **Tensor and Kernel Scheduler** (27) — my pick. "Schedules tensors onto memories and kernels onto devices" — the two objects literally are your storage half and compute half, and the head noun matches your original instinct. Reads naturally in the walk: "the Tensor and Kernel Scheduler enacts the configuration's placements each step and dispatches every operator to the kernel where its operands sit." One caveat you already have a gate for (C2): upstream AsymGEMM has a kernel-internal `asymScheduler`, so distinguish at first use — one parenthetical.
2. **Memory and Compute Scheduler** (28) — same shape, resource-level wording instead of object-level. Slightly more OS-textbook.
3. **Runtime Resource Manager** (24) — single head noun ("resources" = memory + compute), most conventional, least specific.
4. **Placement-Aware Execution Engine** (31) — compute as the head, storage as the modifier; descriptive but the longest and a bit engineered.

It's genuinely defensible as a "scheduler," by the way: it doesn't just replay the configuration — it makes runtime choices inside it (kernel-route thresholds by row count, prefetch guards on free memory), which is scheduling in the ordinary sense.

Set would be: **Training Kernel Pool · Configuration Selector · Tensor and Kernel Scheduler · Dynamic Expert Load Balancer** (handles: the Pool, the Selector, the Scheduler, the Balancer). Confirm and I'll propagate everywhere (§3 titles, §3.1 walk, prose handles, §2 pointers, gate-C5 figure notes, plus the first-use distinction from upstream's `asymScheduler`).

---

## USER  [2026-07-27 07:06Z]

among the 4 whichone wil eb the best?

---

## USER  [2026-07-27 07:06Z]

among the 4 whichone wil eb the best judge cateugl andlet me knwo

### ASSISTANT  [2026-07-27 07:07Z]

Judged against your own criteria — says both jobs from the name alone, plain vocabulary, fits the box, reads naturally, no ambiguity:

| Candidate | Verdict |
|---|---|
| **Tensor and Kernel Scheduler** | **Best** — see below |
| Memory and Compute Scheduler | Runner-up. Same shape, but "memory scheduler" is an odd collocation (memory is *placed*, not scheduled), and the objects are vaguer |
| Runtime Resource Manager | Reject — tells the reader almost nothing specific; "ResourceManager" is also a famous Hadoop/YARN component name, wrong association for a systems audience |
| Placement-Aware Execution Engine | Reject — demotes placement to a modifier when it's co-equal, "engine" is the jargon direction you said you don't want, and at 31 chars it's the worst box fit |

Why #1 wins:

1. **Its words are the section's own words.** §3.1 says "determines where every **tensor** resides"; §3.4 says "routes each operator to the **kernel** of the device holding its operands." The name is composed from nouns the reader has already been given — your lexical-match rule, applied to the name itself. "Memory/compute" appear far less in your prose.
2. **"Scheduler" is technically earned, not loose.** The module's central trick is *temporal*: hiding restages, CPU work, and prefetch inside the backward's dependency windows, gated by runtime thresholds. Deciding *when* work happens is what scheduling means — so the head noun is accurate, not just familiar.
3. **The shared word with "Training Kernel Pool" draws the arrow.** Scheduler dispatches *onto* the Pool; the repeated "kernel" makes the figure's dataflow self-explanatory.
4. Concrete objects beat resource abstractions for a first-time reader: "tensors and kernels" is checkable against the figure; "memory and compute" is a category label.

The one cost: upstream's kernel-internal `asymScheduler` — handled with a one-parenthetical distinction at first use (and never citing `scripts/lf/asym_scheduler.py` as this module; that file is the Selector), which is exactly what your gate C2 prescribes.

Final set: **Training Kernel Pool · Configuration Selector · Tensor and Kernel Scheduler · Dynamic Expert Load Balancer**. Want me to propagate it now?

---

## USER  [2026-07-27 07:08Z]

jo ub opalcement is detmied first and thenwe detmier eh compute (knerel) rihgt ...? the 4th souds even theebst to me ...?

---

## USER  [2026-07-27 07:09Z]

jo ub opalcement is detmied first and thenwe detmier eh compute (knerel) rihgt ...? the 4th souds even theebst to me ...? rethikg andeltneow? thsi configatoni is liek placmeent firs ad htenwe detmeir the knerles (comptues) soliek palcemtn + compute but we cant hsut olanlt say meory comteuschuder soundstooo crued but laso dont nwan be too highe lrl or arkacne need o be concrete soooo?

### ASSISTANT  [2026-07-27 07:10Z]

Yes, your causal reading is exactly right — the configuration fixes *where tensors live* first, and the kernel choice *follows from* placement ("every operator runs where its operands sit"). So the name should put placement first and mark that it *determines* the compute. That's also why #4 appeals — but its two weak words are fixable:

- "**-Aware**" undersells you. Aware = "takes placement into consideration." In your system placement *decides* the dispatch. The accurate modifier is "**-Driven**".
- "**Engine**" is the fluffy part — vague head noun, jargon direction.

Fixing both gives:

**Placement-Driven Kernel Scheduler** (32) — my recommendation. It encodes your exact story in order: placement (first, the given) → drives → which kernel runs (the compute it schedules at runtime). Both halves concrete, no arcane words, and the walk sentence unpacks the name in the same order: *"The Placement-Driven Kernel Scheduler enacts the configuration's placements each step — staging, streaming, offloading, recomputation — and dispatches every operator to the kernel where its operands sit."* At 32 chars it's a touch over the balancer's 28, but fine as a two-line box title.

Fallbacks, same idea:
- **Placement-Driven Executor** (25) — compact, fits one line; loses the explicit "kernel".
- **Placement and Kernel Scheduler** (29) — no hyphenated modifier; keeps the order by coordination but drops the "determines" relationship, which is the thing you wanted said.

Set: **Training Kernel Pool · Configuration Selector · Placement-Driven Kernel Scheduler · Dynamic Expert Load Balancer** (handles: the Pool, the Selector, the Scheduler, the Balancer; one first-use parenthetical to distinguish from upstream's kernel-internal `asymScheduler`, per your gate C2).

Good to propagate?

---

## USER  [2026-07-27 07:13Z]

Placement-aware Kernel Schueder? but kenrle schuedler lisnda bit odd i dont know ...?? knerle dsiaphcer?

---

## USER  [2026-07-27 07:13Z]

Placement-aware Kernel Schueder? but kenrle schuedler lisnda bit odd i dont know ...?? knerle dsiaphcer? wha tieth better prhaiosn thinkg andeltneow

### ASSISTANT  [2026-07-27 07:13Z]

**Dispatcher — and your instinct about why is correct.** "Kernel Scheduler" is odd because the phrase already means something else: in GPU-land a kernel scheduler is the driver/hardware machinery that orders kernel *launches* (streams, thread blocks). Your module never reorders launches — it picks **which of several kernel implementations runs an operator** (streamed AsymLoRA vs. resident cuBLAS vs. Grace CPU). That act has an exact standard name: **kernel dispatch** — it's literally what PyTorch's ATen "dispatcher" does (one operator, multiple backends, runtime routes), so every MLSys reviewer parses it instantly and correctly.

Three more points in Dispatcher's favor:

1. Your §3.4 already contains a "Kernel Dispatch" paragraph and already calls the component "the Dispatcher" — the name becomes a zero-churn lexical match with the section (your rule D).
2. It kills the upstream `asymScheduler` name clash outright (gate C2) instead of managing it with a parenthetical.
3. Dispatch is a *rule applied per operator* ("run where the operands sit"), which is precisely your semantics; scheduling implies temporal ordering decisions you mostly don't make.

On the modifier: **-Driven** over -Aware, same reason as before — placement doesn't just *inform* the dispatch, it *decides* it; "aware" (locality-aware, topology-aware) signals a heuristic that merely considers a factor.

**Recommendation: Placement-Driven Kernel Dispatcher** (34 chars — two-line box title, which is fine). If you need one line: **Placement-Driven Dispatcher** (27).

Final set: **Training Kernel Pool · Configuration Selector · Placement-Driven Kernel Dispatcher · Dynamic Expert Load Balancer** — handles: the Pool, the Selector, the Dispatcher, the Balancer. (The repeated "Kernel" in Pool and Dispatcher is a feature: the Dispatcher dispatches onto the Pool, and the figure arrow explains itself.)

Confirm and I'll propagate through §3 titles, the §3.1 walk, prose handles, §2 pointers, and the gate-C5 figure-relabel notes.

---

## USER  [2026-07-27 07:16Z]

but sti sound off tho .. what evne is placement drivie rihgt? lieksuers will read it and be conused name sliek "KV Resizer" is a good exmaole it is easy tounenra and clear BUT dont tell users how theyresize or when they resize right?

---

## USER  [2026-07-27 07:16Z]

but sti sound off tho .. what evne is placement drivie rihgt? lieksuers will read it and be conused name sliek "KV Resizer" is a good exmaole it is easy tounenra and clear BUT dont tell users how theyresize or when they resize right? so we dont need ful info inthe names but sut need ti be conncre this is curnet  abit toom uch ifno

---

## USER  [2026-07-27 07:16Z]

but sti sound off tho .. what evne is placement drivie rihgt? lieksuers will read it and be conused name sliek "KV Resizer" is a good exmaole it is easy tounenra and clear BUT dont tell users how theyresize or when they resize right? so we dont need ful info inthe names but sut need ti be conncre this is curnet  abit toomuch info

### ASSISTANT  [2026-07-27 07:17Z]

You're right, and the KV Resizer test is the correct standard: concrete object + concrete action, zero mechanism in the name. "Placement-Driven" fails that test — it's a *how*, and readers shouldn't have to parse a how in a name.

Applying the test: **Kernel Dispatcher** (17).

- Same shape as KV Resizer: what it acts on (kernels) + what it does (dispatches). Nothing about how or when — thresholds, dependency windows, placement logic all stay in §3.4's body, exactly like KV Resizer doesn't say when it resizes.
- It's the sharpest already-known phrase for this act — PyTorch's dispatcher does exactly this (one operator, several kernel implementations, runtime routes) — so readers get it with zero effort.
- The placement half doesn't need to live in this name: "configuration" already means "the placement assignment" in your paper, so the Selector's name owns *deciding* placement, and §3.4's opening sentence ("enacts the configuration's placements and routes each operator to the kernel where its operands sit") delivers the storage half one line in. Names point, sentences explain.
- Bonus: dispatches onto the **Training Kernel Pool** — the two names explain the figure arrow between them.

Final set — all four now pass the KV Resizer test:

**Training Kernel Pool (20) · Configuration Selector (22) · Kernel Dispatcher (17) · Dynamic Expert Load Balancer (28)**

Handles: the Pool, the Selector, the Dispatcher, the Balancer. Box = subsection title, identical strings.

Shall I propagate?

---

## USER  [2026-07-27 07:20Z]

but it alos NOT just dsiaphce kenrle sit deosnoffloading / staign as well ..

### ASSISTANT  [2026-07-27 07:20Z]

Right — so the fix is to add the second *object*, not a modifier: it dispatches **tensors** (to their placements: offloaded, staged, streamed) and **kernels** (to the device holding the operands). One action, two objects:

**Tensor and Kernel Dispatcher** (26)

- Passes the KV Resizer test: concrete objects + one plain action, zero how/when.
- Covers both halves you keep asking for — tensor half = the offloading/staging/streaming work, kernel half = the compute routing.
- Keeps the "Dispatcher" head you liked, no "-driven/-aware" mechanism words.
- The walk sentence unpacks it perfectly in parallel: *"The Tensor and Kernel Dispatcher sends each tensor to its assigned placement — offloaded, staged, or streamed — and each operator to the kernel where its operands sit."*
- 26 chars — single-line box, right between Selector (22) and Balancer (28).

Runner-up, if you'd rather have a single object one level up: **Training Step Executor** (22) — "executes each step under the selected configuration" inherently includes both moves and compute, but it's vaguer about what execution involves.

Set: **Training Kernel Pool · Configuration Selector · Tensor and Kernel Dispatcher · Dynamic Expert Load Balancer** — the Pool, the Selector, the Dispatcher, the Balancer.

This one covers your objection; want me to propagate it?

---

## USER  [2026-07-27 07:23Z]

agia this souds soo odd can read eve underna form tis anmeing??

### ASSISTANT  [2026-07-27 07:24Z]

Let me be straight about why this keeps looping: the module has **two duties** (move/park tensors, route compute), and a name can do only one of three things — every candidate you've heard falls into one of these branches, and each branch has the cost you've been hitting:

1. **Name one duty sharply** → "Kernel Dispatcher" — concrete, instantly understood, but you feel the missing offload/staging half.
2. **Name both duties** → "Tensor and Kernel Dispatcher," "Placement & Dispatch Runtime" — complete, but compounds always sound engineered. This branch will *never* sound natural; more iterations won't fix it.
3. **Name the role one level up** → the whole online job under one common word. Slightly less specific, but natural and impossible to misread.

Since your test is now "will readers understand instantly," branch 3 wins, and the best name there is:

**Execution Engine** (16)

- Every systems reader knows exactly what an execution engine is: *the component that actually runs the work at runtime*. Nobody misreads it, nothing to decode.
- "Execution" inherently includes both duties — running the step *means* doing the transfers and launching the kernels. No half is silently dropped, unlike branch 1.
- Your figure box already shows the specifics (HBM/SMEM, Weights, Activations, Staging, Streaming, NVLink C2C), and the walk sentence delivers the rest in one line: *"The Execution Engine executes each step under the selected configuration: it stages, streams, or offloads each tensor per its placement, and dispatches every operator to the kernel where its operands sit."*
- You already liked this head noun — it's your #4 with the confusing modifier deleted.

Set: **Training Kernel Pool · Configuration Selector · Execution Engine · Dynamic Expert Load Balancer** (the Pool, the Selector, the Engine, the Balancer).

If you'd rather stay in branch 1, "Kernel Dispatcher" is the right pick there — with the honest defense that staging/streaming are the data-side of each dispatch decision. But branch 3 / Execution Engine is my recommendation. Which branch do you want?

---

## USER  [2026-07-27 07:27Z]

edxecution engine sounds toooo vague rho?

---

## USER  [2026-07-27 07:27Z]

edxecution engine sounds toooo vague rho? or is that the ebst we can do?

---

## USER  [2026-07-27 07:27Z]

edxecution engine sounds toooo vague rho? or is that the ebst we can do? kerel dsipahc misses duties

### ASSISTANT  [2026-07-27 07:28Z]

One more real option exists — a single concrete word that genuinely *means* both duties at once: **dataflow**.

**Dataflow Engine** (15)

Here's the key realization: placement and kernel choice aren't actually two separate duties in your system — they're **two views of one decision**. Pick the streamed kernel ⇒ the tensor stays in DRAM and flows over C2C; pick cuBLAS ⇒ it stages to HBM first; pick the Grace kernel ⇒ the data doesn't move and the compute comes to it. Where the data sits, how it travels, and which kernel consumes it — that single joint choice is exactly what "dataflow" names. So a coordination ("Tensor and Kernel …") was never needed; the right single word existed.

Reader test: an MLSys reader seeing **Dataflow Engine** on a box containing HBM/SMEM, Weights, Activations, Staging, Streaming, NVLink-C2C guesses "the runtime part that drives how tensors move and reach the kernels" — which is precisely correct, both halves, no decoding. It's far less vague than Execution Engine (says *what* it manages, not just "it runs things"), and it's standard vocabulary — dataflow execution/dataflow engines are established systems terms.

Walk sentence: *"The Dataflow Engine sets each operator's dataflow every step: it stages, streams, or offloads tensors per the configuration, and dispatches the operator to the kernel where its operands sit."* — the name, then the two halves as its unpacking.

Set: **Training Kernel Pool (20) · Configuration Selector (22) · Dataflow Engine (15) · Dynamic Expert Load Balancer (28)**.

If "Engine" itself bothers you: **Dataflow Scheduler** (18) or **Dataflow Dispatcher** (19) work with the same logic, though Engine reads the most naturally of the three.

This is honestly the strongest candidate we've had — it beats the trilemma instead of picking a corner of it. Confirm and I'll propagate?

---

## USER  [2026-07-27 07:31Z]

hmmm still siudns werid .. i am not satieisde ... ok desibe this issue fkkky adnw rie to prokp here  desifbe the contex tnad souaiotn well enogu so that i can ask another gent to help me

---

## USER  [2026-07-27 07:31Z]

hmmm still siudns werid .. i am not satieisde ... ok desibe this issue fkkky adnw rie to prokp here  desifbe the contex tnad souaiotn well enogu so that i can ask another gent to help me'/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/agent/impls/prompt.md'

### ASSISTANT  [2026-07-27 07:34Z]

Written to `agent/impls/prompt.md` (the file you created; it was empty). It gives the next agent everything needed to work the naming problem cold:

- **Paper + system context** — what AsymLoRA is, the four §3 modules and what each does, with module 3's two duties spelled out (placement enactment: staging/streaming/offloading/recompute; kernel dispatch: routing each operator to streamed AsymLoRA / cuBLAS / Grace CPU), plus the coupling you emphasized: placement is fixed first, compute follows.
- **Your 7 naming criteria** — reader-transparent (KV Resizer / Dynamic Expert Load Balancer as style anchors, no mechanism in names), both duties not hidden, natural non-engineered English, ≤28-char figure-box budget, box = subsection title, the `asymScheduler`/Router/ResourceManager clash constraints, parallelism with siblings.
- **Status table** — Training Kernel Pool (near-settled), Configuration Selector (settled), Dynamic Expert Load Balancer (settled), module 3 open.
- **All ~15 rejected candidates with the reason each died** — so the agent can't re-propose them — plus the trilemma diagnosis (one duty = incomplete, both duties = engineered, role-level = vague).
- **File pointers** (system.tex §3.4, the md drafts, gates C2/C5, the figure and its current box labels) and a **deliverable spec**: 5–10 fresh candidates with the reviewer-reading + walk sentence each, a ranked pick, a set-parallelism check, no file edits.

---

## USER  [2026-07-27 09:06Z]

'/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/figures/AsymLoRA.pdf' this is an updated graphics design. i have renamed the core compeonts. sooo we need to modify the system desing aprtni to sue these new namings. Most importanlty we need to heavily rewrite the overievew secion od the system desing 3.1. this parts u need to mimic directkv '/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/agent/related_work/directkv.pdf' they ahve a very good systm oerview sectoi that we need to mimic its style/verbage/logic progres the current one is NO good. we need to do 1,2,3,4 to discsuss the sytem in the sustme overview seitno , olkeae read its stemd eisng and the mimic its style to write our design

### ASSISTANT  [2026-07-27 09:06Z]

I'll start by extracting the new component names from your updated figure, and locating DirectKV's system-overview section in the PDF.

### ASSISTANT  [2026-07-27 09:13Z]

Now mirroring in the md ground truth:

### ASSISTANT  [2026-07-27 09:15Z]

Another session left untracked figure archives in the tree — I'll unstage those (not mine to commit) and commit only this change set:

### ASSISTANT  [2026-07-27 09:16Z]

Done and pushed (`e4b4f95`) — local and Overleaf identical. What I did:

**1. New names propagated everywhere.** Decoded your updated figure: **① Training Kernels ② Configuration Selector ③ Runtime Controller ④ Dynamic Expert Balancer** — numbering already in the paper's offline→online walk order. §3.2–3.5 subsection titles now equal the box names verbatim; prose handles are "the Kernels / the Selector / the Controller / the Balancer"; zero leftover "Profiler"/"Dispatcher" in tex or md (§2 needed no changes — it never names components, only concepts). This also closes gates C2 (no "Scheduler" in any paper name, so the upstream `asymScheduler` clash is gone) and C5 (renumber/relabel) — both marked resolved in the skeleton.

**2. §3.1 rewritten in DirectKV's register.** I read their §3 ("Overview of DirectKV Architecture"): one intro paragraph — what the system is, what it *integrates*, figure cite — then **one short paragraph per component**, each with the bolded name as subject, what it does, and a second sentence on what that enables ("The Kernel Generator compiles … offline. It serves as …"). Ours now mirrors that exactly: an intro paragraph (scope, "AsymLoRA integrates training kernels that compute against CPU-resident tensors with record-driven configuration selection and per-step placement control…", the givens + no-approximation guarantee, then the (1)–(4) offline/online split), followed by four numbered component paragraphs — **Training Kernels (1)** … **Configuration Selector (2)** … **Runtime Controller (3)** … **Dynamic Expert Balancer (4)**, each ending with its §-pointer. The old three-paragraph overview is gone.

**3. Figure + caption.** Copied the new `scripts/figures/AsymLoRA.pdf` into the Overleaf `figures/` (230.9 pt wide → renders ~1.02× at column width, comment updated) and rewrote the caption in DirectKV's terse style: "The system architecture of AsymLoRA." plus one orienting sentence over (1)–(4) — replacing the old caption that still said "the scheduler profiles…".

**4. Housekeeping.** Ground-truth md mirrored 1:1, drafts re-synced (including the previously stale `motivation_v2_plots.md` and `toconfirm.md`). Another session's M5b spill-fix commit landed in between — no conflicts; I left its untracked figure archives out of the commit.

One flag: your figure prints "Dynamic Expert **Balancer**" — no "Load", though you'd earlier said you liked "Dynamic Expert Load Balancer". I followed the figure everywhere; if you want "Load" back it's a two-spot change (figure box + §3.5 title), say so.
