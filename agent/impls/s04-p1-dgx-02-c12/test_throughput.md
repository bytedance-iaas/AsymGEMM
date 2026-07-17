# test_throughput — measuring max throughput per (model × config × seq)

HOST: s04-p1-dgx-02-c12 | GPU: GB200 (189471 MiB = 185.0 GiB HBM) | container asym_sft_40, venv torch 2.12.
Per-host isolation: this dir = this machine's record; other machines write agent/impls/<hostname>/
(handoff prompt: agent/impls/throughput_prompt.md). Raw artifacts (gitignored) are archived per host:
profiling_results/profiling_tp_s04-p1-dgx-02-c12/asym_long_sft_smoke__lora__lf__bf16/ — live runs
write to profiling_results/profiling/... and get mv'd there after each campaign.

Running doc + probe log for the AsymGEMM-backend throughput study. Companion to
`fix_estimator.md`. This doc = the measurement protocol, the BEHAVIORAL RULES the agent
must follow autonomously, and the results.

================================================================================
## GOAL
================================================================================
For each (model × backend-config × sequence length), find the **maximum training
throughput** and the batch at which it occurs. We MEASURE per-step latency and DERIVE
throughput. Deliverable: a max-throughput-vs-seq curve per config (tokens/s, TFLOPS,
MFU), each seq's curve showing the rise -> peak -> decline shape, and the batch/%HBM at
the peak. Throughput is the PRIMARY metric; capacity (max batch/seq) rides along.

================================================================================
## BEHAVIORAL RULES (the agent follows these WITHOUT waiting for the user)
================================================================================
R0. MANDATORY POST-SWEEP REVIEW — AFTER EVERY SWEEP, BEFORE MOVING ON.
    Never treat a coarse sweep as "done". After each batch/seq sweep completes, the agent
    MUST inspect the tok/s + %HBM numbers and ask: "is there still ROOM — an untested
    batch (or seq) that could give higher throughput?" If yes, run it. Do NOT wait for
    the user to point it out. "Room" exists whenever ANY of these is true:
      - the peak batch is NOT bracketed by a lower measured point on BOTH sides;
      - tok/s is still rising at the top measured batch (headroom above);
      - there is a coarse gap (>~4 batch units, or >~one 5% tok/s step) between adjacent
        measured points that could hide a higher peak;
      - %HBM at the best point is well below ~90% AND the next batch wasn't tried
        (unused HBM headroom = likely higher throughput);
      - a good point sits directly next to an OOM/FAIL with no point in between.
    Only declare a seq "done" when the peak tok/s is bracketed on both sides at a batch
    resolution of a few units and there is no unused-HBM headroom left. This review is
    NOT optional and NOT gated on user instruction — it runs every time.
R1. PIN THE PEAK BY BATCH INTERPOLATION — DO NOT STOP AT A COARSE SWEEP.
    Throughput vs batch RISES then PEAKS then DECLINES (near the HBM ceiling it thrashes,
    then OOMs). After a coarse sweep, if the peak is not clearly bracketed by MEASURED
    points, PROACTIVELY INSERT intermediate batch sizes and run them — do not wait for
    instruction. Keep bisecting the batch axis until the peak tok/s is pinned.
    Triggers to insert a mid batch:
      - a good run at B_lo then an OOM/FAIL at B_hi  -> insert ~(B_lo+B_hi)/2
        (e.g. 16k good@32, FAIL@48  -> run 40).
      - tok/s still RISING at the top measured batch -> push higher (and/or bisect
        toward the OOM edge) until it stops rising.
      - a sharp DIP between two batches -> insert between them to locate the true peak
        (e.g. 12k: 3274@30 then 2297@32 -> the peak is at/just below 30).
    Stop inserting when: the peak tok/s is bracketed by measured points on BOTH sides
    (higher batch is lower or OOM), to a batch resolution of a few units.
R2. STOP-ON-OOM per seq, but OOM does NOT end the search — it BRACKETS it. An OOM/FAIL
    (0 measured steps) is the upper bracket; the peak is just below it, so interpolate
    downward (R1), don't abandon the seq.
R3. NEAR-CEILING THRASH is real: as reserved approaches physical HBM (~185 GiB), the
    caching allocator stalls (cudaFree/retry + fragmentation) and step time balloons
    BEFORE a hard OOM. A run at ~98% HBM can be much slower than one at ~90%. So the
    peak is usually the last batch comfortably below the ceiling, NOT the largest that
    "fits". Report %HBM (= reserved / physical) so this is visible.
R4. UNITS: everything in GiB (physical HBM = 189471 MiB = 185.0 GiB). Never mix GB/GiB.
R5. ONE CONFIG AT A TIME. Never run two training configs in parallel (perturbs timing +
    contends host RAM). Serial only.
R6. RUN_NAME MUST INCLUDE THE MODEL. The run output dir is keyed by RUN_NAME+batch+seq
    but the superoffload/asym backend tag is model-agnostic -> a fixed RUN_NAME collides
    across models at the same (seq,batch) and OVERWRITES data (lost q3-32b 16k|24,32 to
    q3-30b on 2026-07-15). Use RUN_NAME like "tput_q3-32b".

================================================================================
## METRIC (offline from one measured step-time t)
================================================================================
Steady t = mean of measured steps (drop warmup + first + last), from profile.json
`trainer.timing` or step_samples.json `training_step_milliseconds` (non-warmup).
```
tokens/s = B*s / t
TFLOPS   = FLOPs_per_step / t / 1e12        # useful FLOPs only, NO recompute (MFU convention)
           FLOPs_per_step ~= 6*N_active*(B*s) + attn(~ 12*L*h*s*(B*s)*0.5, causal)
MFU      = TFLOPS / 2250        # GB200 bf16 dense peak ~2250 TFLOPS
N_active: q3-32b 32.8e9 (L=64,h=5120) . q3-30b-a3b 3.34e9 (L=48,h=2048)
```

================================================================================
## RUN PROTOCOL
================================================================================
- PROFILERS=source (nsys perturbs timing), WARMUP_STEPS=1 MAX_STEPS=4.
- MAX_SAMPLES=1024, DATASET_OVERWRITE=false, OVERWRITE=false. Pool name includes
  MAX_SAMPLES (`..._s{seq}__n{MAX_SAMPLES}`, edited dataset_name_for_seq in both
  profile_lora_lf_test_{source,both}.sh) so sizes never collide; first hit builds once,
  reused after. Missing seqs auto-build. Seqs are round thousands (8000,12000,...).
- RUN_NAME includes the model (R6).

Configs under study:
  q3-32b|1     ; superoffload_mem|unsloth-ohbm0  (baseline; fast)
  q3-30b-a3b|1 ; superoffload_mem|unsloth-ohbm0
  (Phase B, target system) asym_cpuadamwds|recomp-off-full-fg-ker000/101-ceil0000-ohbm0

================================================================================
## RESULTS — Phase A superoffload_mem|unsloth-ohbm0 (source, w1+m4). ALL GiB. phys=185.0
================================================================================
%HBM = reserved / 185.0. FAIL = OOM (0 measured steps) = upper bracket for the peak.

### q3-32b (dense, N_active 32.8e9)
| seq   | B  | s/it  | alloc | resv  | %HBM | tok/s | TFLOPS | MFU% | note |
|-------|----|-------|-------|-------|------|-------|--------|------|------|
| 8000  | 24 |  58.0 |  87.0 |  96.1 |  52% | 3312  | 704 | 31.3 | |
| 8000  | 32 |  75.4 | 115.9 | 128.0 |  69% | 3397  | 722 | 32.1 | |
| 8000  | 40 |  93.2 | 144.8 | 161.0 |  87% | 3434  | 730 | 32.4 | MAX (want 36 to confirm) |
| 12000 | 24 |  88.5 | 130.3 | 143.1 |  77% | 3255  | 717 | 31.9 | |
| 12000 | 30 | 110.0 | 162.8 | 178.4 |  96% | 3274  | 722 | 32.1 | MAX |
| 12000 | 32 | 167.2 | 173.7 | 181.2 |  98% | 2297  | 506 | 22.5 | near-ceiling thrash (-30%) |
| 12000 | 40 |  FAIL |   -   | 171.1 |   -  |  -    |  -  |  -   | OOM |
| 16000 | 16 |  82.4 | 115.9 | 128.0 |  69% | 3107  | 709 | 31.5 | MAX (24,32 lost to R6 bug) |

### q3-30b-a3b (MoE, N_active 3.34e9)
| seq   | B  | s/it | alloc | resv  | %HBM | tok/s | TFLOPS | MFU% | note |
|-------|----|------|-------|-------|------|-------|--------|------|------|
| 16000 | 24 | 42.8 | 107.2 | 113.6 |  61% | 8969  | 264 | 11.8 | |
| 16000 | 32 | 55.3 | 142.0 | 149.9 |  81% | 9261  | 273 | 12.1 | still rising -> insert 40 |
| 16000 | 48 | FAIL |   -   | 173.7 |   -  |  -    |  -  |  -   | OOM (upper bracket) |
| 24000 | 16 | 47.2 | 107.2 | 113.6 |  61% | 8136  | 278 | 12.4 | |
| 24000 | 24 | 68.5 | 159.5 | 167.8 |  91% | 8411  | 288 | 12.8 | still rising -> insert 28 |
| 24000 | 32 | FAIL |   -   | 173.7 |   -  |  -    |  -  |  -   | OOM (upper bracket) |

### Interpolation queue (per R1, running now)
- q3-30b 16000|40   (bracket 32-good .. 48-OOM)  -> pin 16k peak
- q3-30b 24000|28   (bracket 24-good .. 32-OOM)  -> pin 24k peak
- q3-30b 24000|20   (fill rising curve)
- q3-32b 8000|36    (confirm 8k peak between 32 and 40)

### Findings (corrected 2026-07-15, GiB units)
- Throughput DROPS with seq (attention): q3-32b 3434(8k)/3274(12k)/3107(16k).
- Saturation + near-ceiling collapse SEEN: q3-32b 12k = 3255(b24,77%)/3274(b30,96%)/
  2297(b32,98% thrash)/OOM(b40). Peak = last batch below the ceiling, not the largest.
- MoE q3-30b ~2.7x higher tok/s than dense q3-32b (8-9k vs ~3.4k) but LOWER MFU
  (~12% vs ~32%): few active params -> low arithmetic intensity per token.
- q3-30b still RISING at the top measured batch -> peaks sit just under the OOM edge;
  the interpolation queue above pins them (R1).
- BUG FIXED-FORWARD: earlier "reserved>185 = SPILL" was a GB-vs-GiB unit error; nothing
  exceeded physical HBM (194.5 GB = 181.2 GiB < 185). The collapse is near-ceiling
  allocator thrash then OOM, not spill past physical.

================================================================================
## PLAN — remaining sweeps (write-up; run after current interpolation)
================================================================================
### GOAL UPDATE (2026-07-16): GAP-SEARCH, not max-tp squeeze
Target = seqs where superoffload's max FITTING batch (B_max) is small AND sits BELOW its knee:
  gap condition: (a) B_max <= ~4  AND  (b) MFU(B_max) < plateau MFU (q3-32b ~31%, q3-30b ~13-14%, llama ~40%).
  severity = saturation deficit = 1 - MFU(B_max)/plateau. Rank seqs by deficit.
There, asym (leaner HBM/sample) can fit a LARGER batch at higher %HBM and win tok/s while both fit.
Counterexample: llama recomp 48k caps at b2 but MFU 40.1% = saturated -> NO gap (small B_max alone
is not enough). Exemplar: llama unsloth 128k b2 = 32.6% vs 40% plateau (~18% deficit, 80% HBM).
Keep MAX labels in tables, but stop spending runs pinning peaks at short saturated seqs.

### QUEUE (in order)
1. [RUNNING] capQ: q3-30b 50k/56k/64k, both configs -> B_max + deficit per seq (q3-32b arm done:
   45k uns B_max=8 deficit ~4%; 45k rc B_max=3 deficit ~6% -> dense gap needs longer seqs).
2. Extend gap-search per config until B_max<=2 with deficit mapped (q3-32b 50k/56k+; q3-30b 80k+
   if needed) — 2-3 runs/seq (B_max bracket + measure), NO dense sweeps.
3. llama localize the terminal events:
   - unsloth deficit onset: 104k/112k/120k @ b2 (bracketed 96k healthy -> 128k 18% deficit).
   - recomp wall: 56k @ b2 (48k fits saturated -> 64k OOM; deficit may never appear = fails by OOM
     not knee — itself a finding).
4. Phase B head-to-head: asym LATENCY at the top-deficit seqs (llama 128k first; Qwen per capQ)
   — show asym fits bigger batch and beats superoffload tok/s while both fit. Also finish asym
   24k (q3-32b b16; q3-30b b16/24/32) for the mid-seq reference.
5. (optional, deprioritized) peak refinements: q3-32b 32k|12 uns, 32k|6 rc.

Batch grids set from N_max ~= ceiling_seq * ceiling_batch (max tokens that fit).
At each test seq, start batch past the knee, climb toward ~N_max/seq; R1 interpolates
and R2 lets OOM bracket the peak. Bigger/denser model => LOWER batch starts.

### Stage 1 (after q3-32b, q3-30b done): llama3.3-70b|1 ; superoffload_mem|unsloth-ohbm0
llama is 70B DENSE (L=80,h=8192,N_active=70.6e9) => far fewer batches fit than q3 =>
LOWER batch starting points. Ceiling ~45000|8 (N_max ~360k, per user). Knee is small in
tokens but batch is model-size-limited. MFU peak ~2250 TFLOPS.
  8000  | 8, 12, 16       (N 64k-128k)
  16000 | 6, 8, 12        (N 96k-192k)
  32000 | 4, 6, 8         (N 128k-256k; near ceiling)
(start low; R1/R2 pin the peak and find the OOM edge per seq.)

### Stage 2: NEW CONFIG superoffload_mem|recomp|ligerloss1 — 3 models
`recomp` = full activation recompute (heavier compute, different memory profile from
unsloth-ohbm0). Ceilings (batch 8) given by user:
  q3-30b-a3b : 45000|8  (G-OOM 46k)  -> N_max ~360k
  q3-32b     : 20000|8  (G-OOM 21k)  -> N_max ~160k
  llama3.3-70b: 12000|8 (G-OOM 13k)  -> N_max ~96k

Search plans (N_active: q3-32b 32.8e9, q3-30b 3.34e9, llama 70.6e9):

q3-30b-a3b|1 ; superoffload_mem|recomp|ligerloss1        (N_max ~360k, knee ~63k)
  8000  | 24, 32, 40
  16000 | 12, 16, 20
  24000 | 8, 12, 15
  45000 | 8                (ceiling anchor)

q3-32b|1 ; superoffload_mem|recomp|ligerloss1            (N_max ~160k, knee ~6.3k)
  8000  | 12, 16, 20
  12000 | 8, 12
  16000 | 8, 10
  20000 | 8                (ceiling anchor)

llama3.3-70b|1 ; superoffload_mem|recomp|ligerloss1      (N_max ~96k)
  4000  | 12, 16, 24
  8000  | 8, 12
  12000 | 8                (ceiling anchor)

### Stage 3 (Phase B): the TARGET SYSTEM — AsymGEMM backend, ohbm0
Same protocol, `asym_cpuadamwds|recomp-off-full-fg-ker000/101-ceil0000-ohbm0`, all 3
models. This is the comparison payload: asym vs superoffload (both unsloth-ohbm0 and
recomp) at matched seqs -> tokens/s + MFU per seq + max-seq each fits (capacity frontier).

### Execution rules for ALL stages: R1-R6 above apply. Serial, model-tagged RUN_NAME,
### GiB units, interpolate to pin peaks, OOM brackets (don't abandon the seq).

================================================================================
## FINAL — superoffload_mem|unsloth-ohbm0, fully pinned (R0/R1 satisfied), GiB
================================================================================
phys HBM=185.0 GiB. %HBM=reserved/185. MAX=peak tok/s at that seq (bracketed both sides).
Near-ceiling (>=~96% HBM) => allocator thrash: tok/s DROPS then OOMs.

### q3-32b (dense)
| seq   | B  | s/it  | resv  | %HBM | tok/s | MFU% |     |
|-------|----|-------|-------|------|-------|------|-----|
| 8000  | 24 |  58.0 |  96.1 |  52% | 3312  | 31.3 | |
| 8000  | 32 |  75.4 | 128.0 |  69% | 3397  | 32.1 | |
| 8000  | 36 |  84.2 | 143.1 |  77% | 3421  | 32.3 | |
| 8000  | 40 |  93.2 | 161.0 |  87% | 3434  | 32.4 | |
| 8000  | 44 | 102.1 | 174.4 |  94% | 3447  | 32.6 | MAX (plateau) |
| 12000 | 24 |  88.5 | 143.1 |  77% | 3255  | 31.9 | |
| 12000 | 30 | 110.0 | 178.4 |  96% | 3274  | 32.1 | MAX |
| 12000 | 32 | 167.2 | 181.2 |  98% | 2297  | 22.5 | thrash |
| 12000 | 40 |  FAIL | 171.1 |   -  |  -    |  -   | OOM |
| 16000 | 16 |  82.4 | 128.0 |  69% | 3107  | 31.5 | |
| 16000 | 20 | 101.9 | 161.0 |  87% | 3140  | 31.9 | MAX |
| 16000 | 24 | 162.6 | 181.2 |  98% | 2362  | 24.0 | thrash |
| 20000 | 16 | 106.5 | 161.0 |  87% | 3005  | 31.5 | MAX |
| 20000 | 20 | 177.9 | 181.4 |  98% | 2249  | 23.6 | thrash |
| 24000 |  8 |  67.8 |  96.2 |  52% | 2830  | 30.7 | |
| 24000 | 12 | 100.4 | 143.1 |  77% | 2870  | 31.1 | |
| 24000 | 14 | 116.3 | 168.9 |  91% | 2889  | 31.3 | MAX |
| 24000 | 16 | 180.8 | 181.4 |  98% | 2124  | 23.0 | thrash |
| 28000 |  8 |  82.4 | 112.1 |  61% | 2718  | 30.4 | |
| 28000 | 10 | 101.7 | 139.7 |  76% | 2754  | 30.8 | |
| 28000 | 12 | 121.2 | 168.9 |  91% | 2772  | 31.0 | MAX |
| 30000 |  8 |  89.8 | 120.3 |  65% | 2672  | 30.4 | |
| 30000 | 10 | 110.9 | 149.0 |  81% | 2706  | 30.8 | |
| 30000 | 12 | 132.6 | 178.4 |  96% | 2714  | 30.9 | MAX |
| 32000 |  8 |  95.6 | 128.0 |  69% | 2679  | 30.9 | |
| 32000 | 10 | 118.5 | 161.0 |  87% | 2700  | 31.2 | MAX (b12 refine) |
| 36000 |  6 |  86.2 | 108.2 |  58% | 2507  | 29.8 | |
| 36000 |  8 | 113.4 | 143.1 |  77% | 2540  | 30.2 | |
| 36000 | 10 | 140.8 | 178.4 |  96% | 2556  | 30.4 | MAX |
| 40000 |  6 |  99.1 | 120.0 |  65% | 2422  | 29.6 | |
| 40000 |  8 | 130.7 | 161.0 |  87% | 2449  | 30.0 | MAX |
| 45000 |  6 | 115.4 | 134.9 |  73% | 2339  | 29.7 | |
| 45000 |  8 | 152.8 | 178.4 |  96% | 2356  | 29.9 | MAX |
PEAKS: 8k=3447(b44) 12k=3274(b30) 16k=3140(b20) 20k=3005(b16) 24k=2889(b14) 28k=2772(b12) 30k=2714(b12) 32k=2700(b10) 36k=2556(b10) 40k=2449(b8) 45k=2356(b8). tok/s DECLINES w/ seq; MFU ~30% flat -> NO plateau break through 45k (unlike llama@128k).

### q3-30b-a3b (MoE)
| seq   | B  | s/it | resv  | %HBM | tok/s | MFU% |     |
|-------|----|------|-------|------|-------|------|-----|
| 16000 | 24 | 42.8 | 113.6 |  61% | 8969  | 11.8 | |
| 16000 | 32 | 55.3 | 149.9 |  81% | 9261  | 12.1 | |
| 16000 | 36 | 61.8 | 167.8 |  91% | 9320  | 12.2 | MAX |
| 16000 | 40 | 98.8 | 181.4 |  98% | 6479  |  8.5 | thrash |
| 16000 | 48 | FAIL | 173.7 |   -  |  -    |  -   | OOM |
| 24000 | 16 | 47.2 | 113.6 |  61% | 8136  | 12.4 | |
| 24000 | 20 | 57.9 | 140.9 |  76% | 8287  | 12.6 | |
| 24000 | 24 | 68.5 | 167.8 |  91% | 8411  | 12.8 | MAX |
| 24000 | 26 | 84.9 | 180.6 |  98% | 7349  | 11.2 | thrash |
| 24000 | 28 | FAIL | 175.1 |   -  |  -    |  -   | OOM |
| 32000 | 12 | 49.9 | 113.6 |  61% | 7700  | 13.3 | |
| 32000 | 16 | 65.0 | 149.9 |  81% | 7877  | 13.6 | MAX |
| 32000 | 20 |109.3 | 181.4 |  98% | 5856  | 10.1 | thrash |
| 40000 |  8 | 46.6 |  94.8 |  51% | 6873  | 13.3 | |
| 40000 | 12 | 67.5 | 140.7 |  76% | 7114  | 13.8 | MAX |
| 40000 | 16 |117.3 | 181.3 |  98% | 5455  | 10.6 | thrash |
| 45000 |  8 | 54.4 | 106.6 |  58% | 6619  | 13.7 | |
| 45000 | 12 | 79.2 | 157.4 |  85% | 6817  | 14.1 | MAX (b16 OOM) |
| 50000 |  8 | 63.3 | 117.9 |  64% | 6321  | 13.9 | |
| 50000 | 10 | 77.7 | 146.5 |  79% | 6433  | 14.2 | |
| 50000 | 12 | 92.3 | 174.8 |  94% | 6499  | 14.3 | MAX |
| 56000 |  8 | 73.8 | 131.4 |  71% | 6073  | 14.3 | |
| 56000 | 10 | 91.0 | 163.2 |  88% | 6157  | 14.5 | MAX |
PEAKS: 16k=9320(b36) 24k=8411(b24) 32k=7877(b16) 40k=7114(b12) 45k=6817(b12) 50k=6499(b12) 56k=6157(b10).
unsloth fits 45k/50k/56k; MFU RISES with seq (12.2->14.5%) — MoE unsloth shows NO deficit yet (B_max*seq
stays >>knee ~63k tokens): the gap regime needs B_max to collapse (walk 64k->80k->96k...).

R0 in action (this run): filling found HIGHER peaks than the coarse sweep —
q3-30b 16k 9261(b32)->9320(b36); pinned q3-32b 16k=3140(b20); confirmed thrash edges.

================================================================================
## INSIGHT — why bigger batch barely raises tok/s (per-token breakdown, 2026-07-15)
================================================================================
Decomposed steady step time via step_samples.json (forward/backward/training_step ms).
q3-32b unsloth-ohbm0 @8000, per-token time (us/tok) vs batch:
  B24: fwd 84.9 bwd 210.0 step 296.4 ("other"=opt 280ms)
  B32: fwd 83.3 bwd 205.4 step 289.9 (301ms)
  B40: fwd 82.7 bwd 203.8 step 287.5 (317ms)
  B44: fwd 82.2 bwd 203.5 step 286.6 (330ms)

Findings:
1. Per-token COMPUTE time is FLAT (fwd ~82us, bwd ~204us; -3% across batch). => the run
   is COMPUTE-BOUND; GEMMs already M-saturated (M=tokens>=192k >> GEMM-efficiency knee),
   so bigger GEMMs give ~no extra efficiency. tok/s flat is CORRECT, not a bug.
2. The ONLY batch-amortizable cost is "other" = CPU-Adam optimizer step + launch overhead
   ~= 300ms CONSTANT (acts on LoRA params, batch-independent). Amortized over more tokens
   it shrinks step 296->287 us/tok = the +4% tok/s (3312->3447). By b24 you're 96% to the
   compute floor (~286 us/tok); by b44 you're AT it => past b44 zero benefit.
3. No streaming bottleneck visible: per-token time would drop steeply with batch if
   weight-streaming were EXPOSED. It doesn't => streaming is OVERLAPPED/hidden behind
   compute (off critical path). Bigger batch cuts bytes/token but that doesn't convert
   to tok/s here.
4. MFU ceiling (~32%) set by the RECOMPUTE-heavy backward: bwd=2.5x fwd (204 vs 82 us/tok);
   a no-recompute bwd is ~2x fwd, the extra is unsloth-GC recompute (a full extra fwd in
   bwd) + grad offload. MFU excludes recompute FLOPs => recompute time directly caps MFU.
   Optimizer is NOT the bottleneck (0.3-0.5% of step).

Metric that SHOWS the batch benefit = per-token step time approaching the compute floor
(amortizing fixed cost), NOT tok/s/TFLOPS/MFU (all proportional = compute rate). The
streaming/transfer benefit of bigger batch shows only in bytes/token or transfer-fraction
(and only converts to tok/s if transfer is EXPOSED, i.e. below the knee or transfer-bound).

Paper implication: batch-throughput gains here are ~nil (compute-bound, fixed cost tiny) =>
DON'T argue "bigger batch -> more tok/s". Argue (a) sequence-length capacity (grad-accum
can't extend seq), (b) iso-effective-batch MFU where a baseline forced below the knee
(tiny micro-batch + accum) re-pays fixed/transfer costs. See discussion 2026-07-15.

================================================================================
## FINAL — superoffload_mem|recomp, fully pinned (R0), GiB. phys=185.0
================================================================================
### q3-32b (dense) recomp — peaks: 8k=3303(b20) 12k=3197(b14) 16k=3055(b10) 20k=2905(b8) 24k=2819(b6,ceiling)
| seq  | B  | s/it  | resv  | %HBM | tok/s | MFU% | flag |
|------|----|-------|-------|------|-------|------|------|
| 8000 | 12 |  31.8 | 104.9 | 57%  | 3022  | 28.5 | |
| 8000 | 16 |  39.7 | 139.2 | 75%  | 3222  | 30.4 | |
| 8000 | 20 |  48.4 | 173.7 | 94%  | 3303  | 31.2 | MAX |
| 12000| 8  |  33.0 | 104.9 | 57%  | 2908  | 28.5 | |
| 12000| 12 |  46.0 | 156.6 | 85%  | 3128  | 30.6 | |
| 12000| 14 |  52.6 | 181.4 | 98%  | 3197  | 31.3 | MAX |
| 12000| 16 |  FAIL |   -   |  -   |  -    |  -   | OOM |
| 16000| 8  |  42.8 | 139.3 | 75%  | 2993  | 30.4 | |
| 16000| 10 |  52.4 | 173.5 | 94%  | 3055  | 31.0 | MAX |
| 20000| 8  |  55.1 | 173.5 | 94%  | 2905  | 30.5 | MAX (ceiling) |
| 24000| 4  |  35.7 | 105.0 | 57%  | 2685  | 29.1 | |
| 24000| 6  |  51.1 | 156.3 | 84%  | 2819  | 30.6 | MAX (b8 OOM) |
| 28000| 4  |  42.6 | 122.1 | 66%  | 2630  | 29.4 | |
| 28000| 6  |  61.7 | 180.8 | 98%  | 2725  | 30.5 | MAX |
| 30000| 4  |  46.1 | 130.7 | 71%  | 2601  | 29.6 | MAX (b6 OOM) |
| 32000| 4  |  48.7 | 139.3 | 75%  | 2631  | 30.4 | MAX (b6 refine) |
| 36000| 3  |  44.3 | 117.8 | 64%  | 2440  | 29.0 | |
| 36000| 4  |  57.2 | 156.6 | 85%  | 2519  | 30.0 | MAX |
| 40000| 3  |  50.6 | 130.7 | 71%  | 2372  | 29.0 | |
| 40000| 4  |  65.7 | 173.7 | 94%  | 2436  | 29.8 | MAX |
| 45000| 2  |  41.0 |  98.4 | 53%  | 2195  | 27.8 | |
| 45000| 3  |  58.5 | 146.7 | 79%  | 2309  | 29.3 | MAX |

### q3-30b-a3b (MoE) recomp — peaks: 8k=9953(b48) 16k=9103(b24) 24k=8244(b16) 45k=6655(b8,ceiling)
| seq  | B  | s/it | resv  | %HBM | tok/s | MFU% | flag |
|------|----|------|-------|------|-------|------|------|
| 8000 | 24 | 22.8 |  90.9 | 49%  | 8405  |  9.2 | |
| 8000 | 32 | 27.9 | 120.7 | 65%  | 9177  | 10.1 | |
| 8000 | 40 | 33.2 | 150.5 | 81%  | 9638  | 10.6 | |
| 8000 | 48 | 38.6 | 181.0 | 98%  | 9953  | 11.0 | MAX |
| 8000 | 56 | FAIL |   -   |  -   |  -    |  -   | OOM |
| 16000| 12 | 24.5 |  90.9 | 49%  | 7846  | 10.3 | |
| 16000| 16 | 30.3 | 120.7 | 65%  | 8459  | 11.1 | |
| 16000| 20 | 36.2 | 150.5 | 81%  | 8839  | 11.6 | |
| 16000| 24 | 42.2 | 181.0 | 98%  | 9103  | 11.9 | MAX |
| 16000| 28 | FAIL |   -   |  -   |  -    |  -   | OOM |
| 24000| 8  | 26.7 |  90.9 | 49%  | 7192  | 10.9 | |
| 24000| 12 | 36.7 | 135.6 | 73%  | 7848  | 11.9 | |
| 24000| 15 | 43.9 | 169.7 | 92%  | 8201  | 12.5 | |
| 24000| 16 | 46.6 | 181.0 | 98%  | 8244  | 12.5 | MAX |
| 32000| 10 | 41.9 | 150.5 | 81%  | 7645  | 13.2 | |
| 32000| 12 | 49.2 | 181.0 | 98%  | 7804  | 13.5 | MAX (b14 OOM) |
| 40000| 8  | 45.9 | 150.5 | 81%  | 6973  | 13.5 | MAX (b12 OOM) |
| 45000| 8  | 54.1 | 169.7 | 92%  | 6655  | 13.8 | MAX (b12 OOM; unsloth fits b12) |

### recomp vs unsloth-ohbm0 (both superoffload_mem) — CORRECTED 2026-07-16
- Peak tok/s ~EQUAL (±2%): dense ~30% MFU, MoE ~13% MFU, both compute-bound, decline w/ seq.
- **CAPACITY: unsloth-ohbm0 fits MORE, not less** (earlier "unsloth OOM past 24k" was NEVER
  measured -> FALSE). unsloth offloads activations to host -> leaner HBM -> higher batch AND
  longer seq: q3-30b @45k unsloth b12@85% vs recomp b8@92%; q3-32b @30k unsloth b12@96% fits
  while recomp b6 OOMs. recomp's edge is HOST RAM only (RSS 142-204GB vs unsloth 364-382GB),
  since it keeps activations in HBM (HBM-heavy) instead of offloading.
- R0 fills beat the coarse grid everywhere (q3-30b recomp 8k 9638->9953).

================================================================================
## FINAL — llama3.3-70b (both configs), pinned incl odd batches. GiB HBM / GB RSS.
================================================================================

### llama3.3-70b — UNSLOTH-ohbm0
| seq | B | s/it | resv GiB | %HBM | RSS GB | tok/s | MFU% | flag |
|---|---|---|---|---|---|---|---|---|
| 8000 | 8 | 37.1 | 37.1 | 20% | 300 | 1727 | 34.9 |  |
| 8000 | 12 | 51.6 | 55.5 | 30% | 350 | 1860 | 37.6 |  |
| 8000 | 16 | 67.0 | 73.2 | 40% | 350 | 1912 | 38.7 |  |
| 8000 | 24 | 97.0 | 111.1 | 60% | 522 | 1980 | 40.0 |  |
| 8000 | 32 | 130.9 | 147.7 | 80% | 522 | 1956 | 39.6 |  |
| 8000 | 40 | 160.1 | 181.0 | 98% | 865 | 1998 | 40.4 | MAX |
| 8000 | 48 | OOM | 181 | 98% | 862 | — | — | OOM |
| 16000 | 6 | 54.4 | 55.5 | 30% | 350 | 1763 | 38.1 |  |
| 16000 | 8 | 70.9 | 73.2 | 40% | 350 | 1806 | 39.1 |  |
| 16000 | 12 | 103.3 | 111.1 | 60% | 522 | 1858 | 40.2 |  |
| 16000 | 16 | 138.6 | 147.7 | 80% | 522 | 1847 | 39.9 |  |
| 16000 | 20 | 170.5 | 181.0 | 98% | 865 | 1877 | 40.6 | MAX |
| 16000 | 24 | OOM | 181 | 98% | 863 | — | — | OOM |
| 24000 | 10 | 138.9 | 138.7 | 75% | 522 | 1728 | 39.8 |  |
| 24000 | 13 | 177.3 | 180.1 | 97% | 865 | 1759 | 40.5 | MAX |
| 32000 | 4 | 79.4 | 73.2 | 40% | 350 | 1613 | 39.4 |  |
| 32000 | 6 | 116.8 | 111.1 | 60% | 522 | 1644 | 40.2 |  |
| 32000 | 8 | 156.4 | 147.7 | 80% | 522 | 1637 | 40.0 |  |
| 32000 | 10 | 192.5 | 181.0 | 98% | 865 | 1663 | 40.6 | MAX |
| 32000 | 12 | OOM | 181 | 98% | 863 | — | — | OOM |
| 40000 | 6 | 157.0 | 138.7 | 75% | 522 | 1529 | 39.5 |  |
| 40000 | 8 | 204.9 | 181.0 | 98% | 865 | 1562 | 40.3 | MAX |
| 48000 | 6 | 196.1 | 165.9 | 90% | 866 | 1469 | 40.0 | MAX |
| 48000 | 7 | 253.0 | 181.4 | 98% | 865 | 1328 | 36.1 | thrash |
| 48000 | 8 | OOM | 181 | 98% | 863 | — | — | OOM |
| 64000 | 4 | 192.1 | 147.7 | 80% | 522 | 1333 | 40.0 |  |
| 64000 | 5 | 237.7 | 181.0 | 98% | 865 | 1346 | 40.4 | MAX |
| 64000 | 6 | OOM | 181 | 98% | 863 | — | — | OOM |
| 96000 | 2 | 168.0 | 110.2 | 60% | 522 | 1143 | 40.7 | MAX (saturated) |
| 96000 | 3 | 253.7 | 165.9 | 90% | 865 | 1135 | 40.4 |  |
| 128000 | 2 | 323.3 | 147.7 | 80% | 522 | 792 | 32.6 | MAX; MFU falls off 40% plateau |

### llama3.3-70b — RECOMP
| seq | B | s/it | resv GiB | %HBM | RSS GB | tok/s | MFU% | flag |
|---|---|---|---|---|---|---|---|---|
| 8000 | 4 | 24.5 | 57.1 | 31% | 300 | 1305 | 26.4 |  |
| 8000 | 8 | 36.5 | 113.3 | 61% | 300 | 1755 | 35.5 |  |
| 8000 | 12 | 50.6 | 169.9 | 92% | 300 | 1896 | 38.3 | MAX |
| 8000 | 13 | 54.9 | 181.3 | 98% | 300 | 1895 | 38.3 | thrash |
| 8000 | 16 | OOM | 180 | 98% | 300 | — | — | OOM |
| 16000 | 4 | 38.2 | 113.3 | 61% | 300 | 1674 | 36.2 |  |
| 16000 | 6 | 53.3 | 169.9 | 92% | 300 | 1802 | 39.0 | MAX |
| 16000 | 7 | OOM | 180 | 97% | 300 | — | — | OOM |
| 16000 | 8 | OOM | 180 | 98% | 300 | — | — | OOM |
| 24000 | 3 | 44.2 | 127.1 | 69% | 300 | 1627 | 37.5 |  |
| 24000 | 4 | 56.3 | 169.9 | 92% | 300 | 1706 | 39.3 | MAX |
| 32000 | 2 | 42.3 | 113.3 | 61% | 300 | 1513 | 36.9 |  |
| 32000 | 3 | 59.6 | 169.9 | 92% | 300 | 1610 | 39.3 | MAX |
| 32000 | 4 | OOM | 180 | 98% | 300 | — | — | OOM |
| 32000 | 6 | OOM | 181 | 98% | 300 | — | — | OOM |
| 48000 | 2 | 65.2 | 169.9 | 92% | 300 | 1472 | 40.1 | MAX |
| 64000 | 2 | OOM | — | — | 300 | — | — | OOM (recomp ceiling 48k-64k; unsloth reaches 128k+) |

Notes: MFU steady ~40% (highest of 3 models, big dense GEMMs). tok/s declines with
seq (O(s^2)): unsloth 8k=1998/16k=1877/32k=1663/48k=1469/64k=1333. unsloth reaches
64k context; recomp caps ~32k. unsloth co-bound (98% HBM + 865GB host at peak);
recomp HBM-heavy (OOMs small batch) but tiny host (300GB). odd-batch fill found a
higher recomp 32k peak (b3=1610 > b2=1513) - odd batches matter at coarse brackets.
