# PROMPT — superoffload throughput/capacity map on THIS machine (cross-machine variance)

Paste everything below as the task for a fresh agent on the new machine.

---

## STANDING VERDICTS (2026-07-17 — do NOT re-litigate, do NOT re-run)
- **unsloth-OFF: CLOSED, WON. Never spend GPU on it again.** Asym latency beats it at every
  measured gap seq with the larger batch: q3-32b 128k +34% (957 vs 712), 160k +34% (857 vs 639),
  llama 128k +39% (677 vs 488), 192k +35% (553 vs 409). so-off is also HOST-RAM-capped
  (b4@128k watchdog-killed). It is not the comparison target anymore.
- **recomp**: llama ≥112k = asym WINS by DNF (recomp b1 wall ~96k, cannot run at all; asym 677/553).
  In overlap regions recomp wins (it dies by OOM, not by slowing). No further runs needed.
- **unsloth-ohbm0**: asym loses -8..-15% at all measured gap points (q3-32b 128/160/192k,
  llama 128/192k). Root cause measured: fwd at parity, bwd +143 us/tok code tax
  (see agent/impls/fix_asym.md — C1b/C2). Flag space EXHAUSTED (KA essential, AU null, ohbm null).
  Do not re-run head-to-heads until a C1/C2 code fix lands.
- **CAPACITY FRONTIER — protocol + results (capG, 2026-07-18). This is the standing behavior:**
  1. Pin every wall with a TIGHT PAIR: last-FIT at 85-98% HBM + first-OOM one step up. Never
     report a wall from prediction alone — the token-linear model UNDER-predicts walls
     (predicted q3-32b uns b1 wall 349k; measured: 384k b1 STILL FITS @98%. Same for llama:
     320k b1 fits @98%). Probe the predicted-OOM seq first; if it fits, that IS the edge point.
  2. Then run ASYM at the baseline's edge seq, SAME seq SAME batch → the airtight comparison.
  3. Order: ALL so-based boundary runs first, asym runs LAST (user rule).
  MEASURED WALLS (c12): q3-32b: uns b1 edge=384k@98% (424 tok/s, wall ~390k); rc wall∈(160k,192k).
  llama: uns b1 edge=352k@98.0% (336 tok/s; 320k@98%=402), OOM@384k — wall PINNED; rc wall∈(96k,112k).
  KEY RESULTS to preserve in any retelling:
  - **At long seq the two superoffload configs CONVERGE to identical per-token** (q3-32b 160k:
    rc b1 941/21.4% ≡ uns b2 942/21.4%; 128k: 1101≈1110) — one bar per seq; rc's edge = -222 GB RSS.
  - **FIRST PARITY POINT @384k q3-32b: asym KA b1 = 426 tok/s @76% HBM vs sup 424 @98% edge**
    (+0.5%, in-noise) — asym matches sup's tok/s at sup's absolute frontier with 40 GiB headroom;
    asym's own frontier extends further (~490k est). Dense long-seq MFU step-down (96k→128k) is
    SHARED by recomp (zero offload) ⇒ attention/regime cost, not offload overhead.
  Frontier probes use MAX_SAMPLES=512 (halves dataset-build time; __n512 dataset names).
  STEP PROTOCOL (2026-07-19, user rule): **w1+m2 — MAX_STEPS=2 WARMUP_STEPS=1 (3 total steps).
  We DON'T run 4 non-warmup steps anymore.** Justified: post-warmup steps are stable (352k llama
  steps 2/3/4 = 1071/1061/1052s, ~1% spread; middle-2 vs all-4 checked earlier = negligible).
  Runs before this date were w1+m4 (5 total) — comparable within ~1%.
  DEFINITION (do not confuse): a CAPACITY-CONFIRM run = the DEEPEST measured FITTING (seq,B)
  per config (e.g. q3-30b uns 128000|4, rc 128000|2) — NOT the %HBM-tightest point at some
  batch ladder (that convention is only for pinning a wall at fixed B).

## MISSION (historical) — find the THROUGHPUT-GAP seqs, not the max-tp squeeze
We are NOT hunting each seq's absolute max tok/s anymore. We are hunting the seq lengths where
superoffload is FORCED into a small-but-still-fitting max batch (B_max) that sits BELOW its
throughput knee — i.e. where capacity caps batch before throughput saturates. At such a seq,
AsymGEMM (leaner HBM per sample) can run a LARGER batch and win throughput outright while both
systems fit. Example shape we want: superoffload fits only b2 (b3 OOMs) at ~80% HBM and its
MFU drops below the config's plateau; asym fits b6 at ~90% and holds plateau MFU.

The GAP CONDITION at a seq (both must hold):
  (a) B_max is small (roughly <=4) — capacity-capped;
  (b) MFU(B_max) is meaningfully below the config's plateau MFU — under the knee.
Severity (rank seqs by this): saturation deficit = 1 - MFU(B_max)/MFU_plateau.
  plateau MFU (reference GB200): q3-32b ~31%, q3-30b-a3b ~13-14%, llama3.3-70b ~40%.
COUNTEREXAMPLE (why (b) matters): llama recomp @48k caps at b2 yet MFU=40.1% (saturated -> no gap:
dense-70B knee is tiny). EXEMPLAR: llama unsloth @128k caps at b2 with MFU 32.6% vs 40% plateau
(~18% deficit, only 80% HBM used) -> prime asym-win window.

Secondary goal: reproduce the reference peaks (keep MAX labels/tables as before) and quantify
cross-machine variance vs `s04-p1-dgx-02-c12`. Reference record (READ, do NOT edit):
`agent/impls/s04-p1-dgx-02-c12/test_throughput{,_results}.md`. Your record: `agent/impls/$(hostname)/`
(create it), same filenames + schema. Work autonomously; do not stop between stages.

## ⚠ ARTIFACT ISOLATION — MANDATORY, READ FIRST
Two locations; know both:
1. **LIVE root (hardcoded, shared)** — the driver always writes each run here; it cannot be redirected:
   `<repo>/third_party/AsymGEMM/profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16/`
   `  <RUN_NAME>__b{B}_s{seq}_ga1_drop000/<config-tag>/b{B}_s{seq}_ga1/{profile.json,step_samples.json,train.log,...}`
2. **PER-HOST ARCHIVE (canonical store)** — after each campaign (and before reporting), archive
   your throughput runs out of the live root:
   `DEST=<repo>/third_party/AsymGEMM/profiling_results/profiling_tp_$(hostname)/asym_long_sft_smoke__lora__lf__bf16`
   ```bash
   mkdir -p "$DEST" && for d in "$ROOT"/tput*; do
     [ "$(basename "$d")" = "<currently-running dir>" ] && continue   # never move an in-flight run
     mv "$d" "$DEST/"; done
   ```
   Reference archive (this pattern in action): `profiling_results/profiling_tp_s04-p1-dgx-02-c12/`.
   Your parser must glob BOTH roots (archive = completed, live = in-flight).
Everything under `profiling*/` is gitignored — but `/home` is **shared NFS across cluster nodes**
(10.78.200.27:/data/home), so the LIVE root is shared between machines using the same checkout:
OVERWRITE=false silently reuses another machine's live runs; OVERWRITE=true clobbers them. The
host-suffixed archive is collision-free by construction. Therefore:
  1. PREFERRED: **separate checkout on node-local disk** (verify `df -T <repo>` is NOT nfs).
  2. If sharing the NFS checkout: never run while another machine runs from it (serial per
     checkout), archive to your `profiling_tp_$(hostname)` promptly, and if overlap is ever
     possible, additionally host-tag RUN_NAME (`tput-<shorthost>_<model>`).
Datasets (`third_party/LlamaFactory/data/asym_long_sft_smoke__<model>__s<seq>__n1024*`) CAN be
shared across machines (deterministic content; keep `DATASET_OVERWRITE=false`) — just don't launch
two machines building the SAME new (model,seq) dataset at the same moment.
Builder speed (fixed 2026-07-19, byte-identical outputs — A/B proven on concat/sample/audit
paths + real 320k file): `build_lf_sft_eval_pair.py` now batch-tokenizes in parallel (Rust
rayon) and the per-run AUDIT of an existing dataset reuses the stored `token_length` field
instead of re-encoding the whole corpus (~5 min/attempt -> ~0). `[build-timing]` lines in the
build log attribute any residual slowness (records / validate_jsonl / lf_preprocess_check).
⚠ tp_probe model keys are the DRIVER's shorthands — q3-32b, llama3.3-70b, q3-30b-a3b
(NOT qwen3-32b). A wrong key used to fake-FIT instantly (arg-error exit matched no failure
pattern); tp_probe now default-denies — FIT requires an ok row in the run's own jobs.tsv.
⚠ KNOWN TRAP: git syncs/merges in the LlamaFactory repo REVERT `data/dataset_info.json`, wiping
the generated-dataset registrations -> the builder re-validates on reuse, sees "missing
registration", and the row HARDFAILS with `validation_ok=False` (fast-fail, no run dir). FIX:
run `python3 /workspace/AsymGEMM-SFT/.repair_dataset_info.py` (idempotent, flock-protected,
re-registers every on-disk asym_long_sft_smoke__*.jsonl) at the START of EVERY campaign script.

## HOST PARTITION (2026-07-17, shared NFS checkout — split by MODEL, never by config,
## so head-to-head pairs stay same-machine, datasets stay disjoint, dirs never collide)
- s04-p1-dgx-02-c12: llama3.3-70b + q3-32b (all configs, superoffload + asym Phase B).
- s04-p1-dgx-02-c14: q3-30b-a3b ONLY (all configs, superoffload deep-end 160k+ + asym Phase B).
  c12 already queued the q3-30b 128k probes + dataset build (grandfathered) — c14 reads c12's
  results doc first and starts at 160k. Do NOT run a model you don't own; host-tag RUN_NAME only
  if you must deviate. Each host archives to its own profiling_tp_$(hostname)/ and writes its own
  agent/impls/$(hostname)/ docs. Before ANY asym Phase-B run: bash scripts/lf/rebuild_asymgemm.sh
  (asym source changed 2026-07-17; superoffload runs need no rebuild).

## SELF-CHECK BEFORE STARTING
1. `hostname` → create `agent/impls/$(hostname)/`.
2. Container entry works: `asym40_enroot_run` (→ `_custom_enroot_run asym_sft_40`); repo at
   `/workspace/AsymGEMM-SFT`. Build a non-interactive helper replicating its mounts to exec bash strings.
3. GPU free: `nvidia-smi` inside container (expect ~0 MiB used on your GPU).
4. Record in your docs header: hostname, GPU name + HBM MiB (`nvidia-smi --query-gpu=name,memory.total`),
   driver, torch (`.venv/bin/python -c "import torch;print(torch.__version__)"`), host RAM total.
   Physical HBM GiB = memory.total/1024 (reference GB200: 189471 MiB = 185.0 GiB). MFU peak = 2250
   TFLOPS (GB200 bf16); different GPU → use its bf16 dense peak and say so.
5. Isolation choice from the section above — state it in your doc header.

## ENVIRONMENT (hard rules)
- ALL runs inside the container; NEVER on the host.
- superoffload_mem needs NO rebuild. (Only asym-backend runs need `scripts/lf/rebuild_asymgemm.sh`
  — venv torch, never system python.)
- STRICTLY SERIAL: one training job, one GPU, at a time.
- Training procs run under setsid — pkill-by-name misses them; kill by PID from
  `nvidia-smi --query-compute-apps=pid`.

## PROTOCOL (identical to reference)
```
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=1024
export DATASET_OVERWRITE=false OVERWRITE=false
RUN_NAME="<tag>_<model>" RUNS="<model>|1 ; <config> ; <seq>|<batch>|1 ; none|false|false|false|false|false" \
  bash scripts/lf/profile_lora_lf_test_source.sh
```
- configs: unsloth = `superoffload_mem|unsloth-ohbm0|ligerloss1` (tag `tput`),
  recomp = `superoffload_mem|recomp|ligerloss1` (tag `tputrc`). Host-tag tags if on shared checkout.
- models: `q3-32b`, `q3-30b-a3b`, `llama3.3-70b`.
- RULES (R0-R6): RUN_NAME MUST include model (R6 — else cross-model dir collision overwrites data).
  Serial (R5). Ascending batch per seq, stop that seq on OOM; OOM is a bracket, not the end of the
  search (R2). After each coarse sweep insert missing batches to pin the true peak (R0/R1). GiB for
  all HBM numbers (R4). Near-ceiling thrash (resv >97% phys + step time 1.5x+ blowup) is a real
  regime — record as `thrash`, distinct from OOM (R3). A run with 0 measured steps is
  IN-PROGRESS/KILLED, not OOM — check step_samples.json before labeling.

## METRICS (offline from each run's profile.json)
```
s/it  = step.rows[name=="lf.step.total"].milliseconds / 1000
alloc/resv GiB = memory.peak_{allocated,reserved}_hbm_bytes / 2^30
RSS GB = memory.process.rss_peak_bytes / 1e9
tok/s = B*seq / s_it
FLOPs/step = 6*N_active*(B*seq) + 12*L*h*seq*(B*seq)*0.5   # causal attn, no recompute FLOPs
TFLOPS = FLOPs/step / s_it / 1e12 ; MFU = TFLOPS/2250
arch: q3-32b N=32.8e9,L=64,h=5120 | q3-30b-a3b N=3.34e9,L=48,h=2048 | llama3.3-70b N=70.6e9,L=80,h=8192
```
Self-contained parser (globs archive + live; adapt HOST + regex if you host-tagged RUN_NAME):
```python
import json, glob, os, re, socket
_PR="/workspace/AsymGEMM-SFT/third_party/AsymGEMM/profiling_results"
HOST=socket.gethostname()
BASES=[_PR+"/profiling_tp_"+HOST+"/asym_long_sft_smoke__lora__lf__bf16",   # archive (canonical)
       _PR+"/profiling/asym_long_sft_smoke__lora__lf__bf16"]               # live (in-flight)
ARCH={"q3-32b":(32.8e9,64,5120),"q3-30b-a3b":(3.34e9,48,2048),"llama3.3-70b":(70.6e9,80,8192)}
GiB=1024**3
for d in sorted(sum((glob.glob(B+"/*_drop000") for B in BASES),[])):
    m=re.search(r"(tputrc|tput)[^_]*_(q3-32b|q3-30b-a3b|llama3[._]3-70b)__b(\d+)_s(\d+)",os.path.basename(d))
    if not m: continue
    tag,model,B,s=m.group(1),m.group(2),int(m.group(3)),int(m.group(4))
    if model.startswith("llama"): model="llama3.3-70b"
    pjs=glob.glob(d+"/**/b%d_s%d_ga1/profile.json"%(B,s),recursive=True)
    if not pjs: print(tag,model,s,B,"NOPROF"); continue
    j=json.load(open(pjs[0])); st=None
    for r in j["step"]["rows"]:
        if r["name"]=="lf.step.total": st=r["milliseconds"]/1000
    if not st: print(tag,model,s,B,"NOSTEP"); continue
    mem=j["memory"]; rv=mem["peak_reserved_hbm_bytes"]/GiB
    rss=mem["process"]["rss_peak_bytes"]/1e9
    N,L,h=ARCH[model]; Bs=B*s; tf=(6*N*Bs+12*L*h*s*Bs*0.5)/st/1e12
    print("%-7s %-12s s=%-6d b=%-3d s/it %6.1f resv %5.1f (%3.0f%%) rss %4.0f  %5.0f tok/s  %4.1f%% MFU"
          %(tag,model,s,B,st,rv,rv/185.0*100,rss,Bs/st,tf/2250*100))
```

## SEARCH PROTOCOL — B1/B2-FIRST, DEEP-END ONLY. Every run must earn its GPU time.
STRATEGY (2026-07-17, v2): the prize is the b1/b2 regime — superoffload capped at batch 1-2 while
still fitting, under-saturated; asym then fills the HBM with 3-6x the batch and wins tok/s.
1. START way above the b8 ceiling: first probe at the seq where B_pred ≈ 2
   (≈ 3-4x the b8-ceiling seq), NOT at B_pred 4-8 territory.
2. LARGE seq steps, scaled to the model: step ≈ 0.5-1.0x of the b8-ceiling seq
   (llama/q3-32b: 16-48k steps; q3-30b: 40-80k steps). Adjust down only when bisecting.
3. KEEP TESTING past the first deficit: map the WHOLE b1/b2 frontier — deficit(seq) at B_max
   for each probe until b1's absolute OOM wall. Deliverable = the win window [onset, wall]
   with deficits, not a single point. A no-deficit b1 point is NOT a stop signal — jump on.
4. Probe DESCENDING from B_pred (HBM = base + per_sample*B; per_sample ~linear in seq;
   B_pred = floor((175 - base)/per_sample(seq))). First FITTING run IS the measurement.
5. Bisect ONLY to refine the onset seq or the wall when a bracket is wider than one step.
HARD EFFICIENCY RULES:
- NEVER re-run a (model,config,seq,B) that exists in the reference tables. READ THEM FIRST.
- NO ascending batch sweeps. NO seq walking through saturated (B_max>=8) territory.
- A GAP point must FIT healthily (<=92% HBM, no thrash). OOM rows are only brackets. Max-seq
  ceilings are ALREADY KNOWN (anchors below) — never re-derive them.
- Run `python3 /workspace/AsymGEMM-SFT/.repair_dataset_info.py` at the start of every campaign.
- Use the reusable executor `scripts/lf/tp_probe.sh` (below) instead of hand-rolling launchers.

B_max ANCHORS (measured on reference c12 — extrapolate, don't rediscover):
```
b8 SEQ-CEILINGS (user-verified, unsloth-ohbm0): q3-32b 49000|8 fits (G-OOM 50k);
  q3-30b-a3b 80000|8 fits (G-OOM 81k); llama3.3-70b 45000|8 fits (G-OOM 46k).
  => B_max>=8 (saturated, no gap) at ALL seqs below these. HARD RULE: never probe below them.
  Gap zone: ~2x ceiling -> B_max~4 ; ~3-4x -> B_max~2. Start there.
q3-32b  unsloth: 45k b8=178.4 ; 96k b3=145.2      | GAP: 8% @96k b3 (56k b6: 6%)
q3-32b  recomp : 80k b2=174.3 ; 50k b3=163.3      | GAP: 7-9% @50-80k b2-3
q3-30b  unsloth: 64k b8=150.0 ; 128k b4=149.9     | GAP: 13% @128k b4 (0 thru 64k)
q3-30b  recomp : 64k b6=181.0 ; 128k b2=119.5     | GAP: 16% @128k b2 (0 thru 64k)
llama   unsloth: 128k b2=147.7 ; 144k b2=165.9    | GAP: 17/18/20% @112k/128k/144k b2 (onset (96k,112k])
llama   recomp : 48k b2=169.9 fits ; 64k b2 OOM   | NO gap ever (dies by OOM, not knee) — CLOSED
Asym capacity anchors (Phase-B expectation): q3-32b asym 65k|8 = 166.8 GiB; q3-30b asym 174k|8 =
170.9 GiB. At any gap seq, asym fits ~2-4x superoffload's B_max.
```
## WALL-PINNING & SATURATION (v3, 2026-07-17 — measured on c14, q3-30b)
For EACH superoffload config the frontier is not done at the first deep b1 fit; deliver these too:
1. SATURATION ROW: a ~92%-HBM b1 point (predict via the fitted per-1k-seq slope, e.g. q3-30b
   recomp 0.4675 GiB/1k, unsloth-ohbm0 0.286 GiB/1k). B_max rows at 50-80% HBM under-tell capacity.
2. WALL BISECTION: far-stride to a predicted-OOM anchor, then bisect toward ~10k granularity.
   STOP bisecting when the bracket is smaller than the workload's own memory granularity — read the
   failed-alloc size in the OOM message (e.g. a 20 GiB monolithic chunk at q3-30b 660k b1);
   sub-chunk seq steps are below-physics and waste runs.
3. NEAR-WALL ALLOCATOR BEND: linear capacity models overpredict by ~3-5 GiB within ~15k of the
   wall (allocator squeezes near capacity). A linear prediction within ~5 GiB of phys is a
   COIN-FLIP, not an OOM certainty — run it (measured: unsloth-ohbm0 320k b2 pred 186.4 → fit
   181.3; recomp 392k pred 184.7 → fit 181.4).
4. `edge` IS A DISTINCT FLAG from `thrash`: edge = 92-98% HBM with <1.5x step blowup (measured
   +0-6%, often 0 — superoffload rides 95-98% nearly free at deep b1); thrash (>=95% AND >=1.5x)
   exists only in the <=64k big-batch regime. Record edge rows as valid MAX points, flagged.
5. TABLE HYGIENE: one datum per cell — flags are single tokens (MAX/edge/thrash/OOM); commentary
   goes in numbered footnotes under the table, never in cells.
Measured walls (q3-30b, GB200-185GiB, b1): recomp (392k, 400k] ; unsloth-ohbm0 (640k, 660k].

Deliverable: GAP WINDOW blocks — ONE TABLE PER (model x config), rows ordered by seq, and the
series MUST include the steady-state SATURATED rows (max tok/s at B_max per seq, deficit 0)
leading into the window — not only the deficit rows. REQUIRED schema — ALWAYS all columns,
never drop memory columns:
`| seq | B_max | resv GiB | %HBM | RSS GB | tok/s | MFU% | deficit | note |`
⚠ NEVER produce cross-config/cross-model RANKED tables — mixing configs into one ranking
destroys readability; every table is partitioned to its own config.
⚠ NEVER present a results table without the memory columns (resv/%HBM/RSS) — throughput
numbers are meaningless for this study without the memory context they were bought at.
Variance vs reference (secondary): spot-check ~2 pinned peaks per {model,config} from the
reference tables (expect |Δ|<5%, same verdicts); note deviations in your doc.

## REUSABLE EXECUTOR — scripts/lf/tp_probe.sh (preferred over hand-rolled launchers)
```
bash scripts/lf/tp_probe.sh <model> <tag> <config-string> <seq> <b_desc...>
# tries batches in order; first FIT exits 0; OOM -> next batch (exit 1 if all OOM);
# non-OOM failure -> HARDFAIL exit 2 with cause. Self-heals dataset_info at start.
# asym latency: prefix env vars (ASYM_GEMM_DISPATCH=staged etc.) before the call.
```
Chain probes from a thin campaign script or one-at-a-time; the AGENT picks each next seq from
the outcomes (that adaptivity is why the strategy is not itself a script).

## LAUNCHER TEMPLATE (inside container)
```bash
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM || exit 3
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=1024 DATASET_OVERWRITE=false OVERWRITE=false
T="none|false|false|false|false|false"
run(){ local model="$1" tag="$2" cfg="$3" seq="$4" b="$5"
  echo "########## RUN ${model} ${tag} seq=${seq} b=${b} $(date -u +%H:%M:%S)"
  out=$(RUN_NAME="${tag}_${model}" RUNS="${model}|1 ; ${cfg} ; ${seq}|${b}|1 ; ${T}" \
    bash scripts/lf/profile_lora_lf_test_source.sh 2>&1)
  echo "$out" | tail -2
  echo "$out" | grep -qiE "CUDA out of memory|OutOfMemoryError|Training command failed" && return 1 || return 0
}
sweep(){ local model="$1" tag="$2" cfg="$3" seq="$4"; shift 4
  for b in "$@"; do run "$model" "$tag" "$cfg" "$seq" "$b" || { echo "@@@ OOM ${model} ${tag} ${seq} b=$b -> STOP"; break; }; done; }
```

## DELIVERABLES
1. `agent/impls/$(hostname)/test_throughput_results.md` — pure metrics tables, schema:
   `| seq | B | s/it | resv GiB | %HBM | RSS GB | tok/s | MFU% | flag |` per model x config,
   + peak-tok/s summary table. Header: hardware + isolation choice.
2. `agent/impls/$(hostname)/test_throughput.md` — protocol notes, anomalies, and a VARIANCE
   table vs reference: `| model | config | seq | B | tok/s here | tok/s c12 | Δ% | verdict-match? |`
   (c12 numbers from `agent/impls/s04-p1-dgx-02-c12/test_throughput_results.md`).
3. Leave everything uncommitted (user commits manually).
