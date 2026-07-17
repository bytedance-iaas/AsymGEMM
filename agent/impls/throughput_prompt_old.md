# PROMPT — superoffload throughput/capacity map on THIS machine (cross-machine variance)

Paste everything below as the task for a fresh agent on the new machine.

---

## MISSION — find the THROUGHPUT-GAP seqs, not the max-tp squeeze
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
export PROFILERS=source MAX_STEPS=4 WARMUP_STEPS=1 MAX_SAMPLES=1024
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

## SEARCH PROTOCOL — GAP-SEARCH per seq (2-3 runs/seq, NO dense batch sweeps)
Per {model, config}, walk seq UPWARD into the capacity-capped regime. At each seq:
1. Estimate B_max from the previous seq (HBM/sample scales ~linearly with seq: B_est ≈
   B_max_prev * seq_prev/seq). Run B_est: if it fits and resv <~85%, run B_est+1 (or +2 at
   small B) until OOM/thrash; if it OOMs, step down. 2-3 runs bracket B_max exactly.
2. The MEASUREMENT is the B_max run: record s/it, resv, %HBM, RSS, tok/s, MFU (schema below).
   Optionally one run at ceil(B_max/2) to expose the rising slope (proves under-knee).
3. Compute deficit = 1 - MFU(B_max)/MFU_plateau. Record fit/OOM verdicts for the brackets.
4. Seq stepping: while deficit ~ 0, JUMP GEOMETRICALLY (x1.25-1.4 per step — saturated seqs are
   wasted runs); once deficit > 0, BISECT backward to localize the onset seq, then walk forward
   to the max-deficit seq that still fits comfortably (<=90% HBM, no thrash) — that seq is the
   Phase-B head-to-head target.
5. A reported GAP point MUST be a healthy FITTING superoffload run (no thrash/OOM). OOM rows are
   only B_max brackets. Do NOT re-derive superoffload's absolute max-seq ceiling — the capacity
   story is already proven separately; this search is orthogonal (throughput gap while both fit).
6. STOP the seq walk for that config when B_max hits 1-2 AND deficit is mapped, or when even
   b1 OOMs (absolute wall). Do NOT spend runs pinning peak tok/s at short seqs where B_max is
   large — those are known-saturated (no gap by construction).
Start points (from reference; cross-check `agent/impls/s04-p1-dgx-02-c12/test_throughput_results.md`
for the latest — its tables keep growing):
```
q3-32b   unsloth: start 45000 (B_max=8 there, deficit ~4%) -> 50k, 56k, 64k ...
q3-32b   recomp : start 45000 (B_max=3, deficit ~6%)       -> 50k, 56k ...
q3-30b   unsloth: start 50000 (B_max>=12)                  -> 56k, 64k, 80k ...
q3-30b   recomp : start 45000 (B_max=8)                    -> 50k, 56k, 64k ...
llama    unsloth: start 96000 (B_max=3, no deficit yet) -> 104k, 112k, 120k, 128k (b2 deficit 18%)
                  -> localize where the deficit starts; probe >128k while b1-b2 fits
llama    recomp : start 48000 (B_max=2, deficit ~0!) -> 52k, 56k, 60k (wall 48-64k; deficit may
                  never appear before OOM — that itself is a finding: recomp fails by OOM, not knee)
```
Deliver a GAP TABLE ranked by deficit:
`| model | config | seq | B_max | %HBM@B_max | MFU@B_max | plateau MFU | deficit% | next-B verdict |`
Variance vs reference (secondary): spot-check ~2 pinned peaks per {model,config} from the
reference tables (expect |Δ|<5%, same verdicts); note deviations in your doc.

## LAUNCHER TEMPLATE (inside container)
```bash
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM || exit 3
export PROFILERS=source MAX_STEPS=4 WARMUP_STEPS=1 MAX_SAMPLES=1024 DATASET_OVERWRITE=false OVERWRITE=false
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
