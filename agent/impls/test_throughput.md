# test_throughput — measuring max throughput per (model × config × seq)

Running doc for the throughput probes. Companion to `fix_estimator.md` (the estimator);
this doc = the actual measurement protocol + probe log.

## Goal
We MEASURE latency (steady s/it) and DERIVE throughput (tokens/s, TFLOPS, MFU) from it.
For each seq, sweep batch and confirm the throughput curve: it should RISE with batch
then SATURATE (plateau) at the knee. In the clean regime it stays FLAT past the knee;
near the memory wall it can DIP (fragmentation) or OOM. The peak/plateau value =
max throughput at that seq. Deliverable: a max-throughput-vs-seq curve per config,
plus visible saturation (rise -> plateau [-> dip/OOM]) in each batch sweep.

## Metric (all offline from ONE measured step-time t)
Measure only steady-state per-step latency `t` (PROFILERS=source, w1+m4, drop
warmup+first+last, mean of middle). Then:
```
tokens/s = B·s / t
TFLOPS   = FLOPs_per_step / t / 1e12       # useful FLOPs only, NO recompute (MFU convention)
           FLOPs_per_step ~= 6*N_active*(B*s) + attention(~ L*h*s*(B*s), causal ~ 1/2)
MFU      = TFLOPS / peak_TFLOPS
N_active: q3-32b 32.8e9 (L=64,h=5120) . q3-30b-a3b 3.34e9 (L=48,h=2048)
```
tokens/s and TFLOPS differ only by FLOPs/token => same batch-saturation, both step-count
independent (rate, not whole-round).

## Probe method (efficient - NOT a full grid, NOT a curve fit)
Per (config, seq): skip super-small batches (below the knee = low throughput, irrelevant).
Start at a batch comfortably PAST the knee, take 2 points; if the bigger batch gains
<~5% => plateau => that's max throughput at this seq. If it still gains, bump once more.
Knees (round-1 fit): q3-32b N*~6.3e3 tok . q3-30b N*~6.3e4 tok. ohbm0 is HOST-bound
(batch ceiling = CPU RAM, not HBM).

## Probe plan
### Phase A - superoffload_mem|unsloth-ohbm0 (FAST: roots in HBM, no host round-trip).
Fast per step => sweep seq x batch cheaply; get latency + tokens/s + TFLOPS + MFU across
the plateau. Batch pairs chosen decently-large (past knee); the triple confirms plateau.

q3-32b|1     ; superoffload_mem|unsloth-ohbm0|ligerloss1
    8000  | 24, 32, 40
    12000 | 24, 32, 40
    16000 | 16, 24, 32
q3-30b-a3b|1 ; superoffload_mem|unsloth-ohbm0|ligerloss1
    16000 | 24, 32, 48
    24000 | 16, 24, 32

### Phase B - AsymGEMM backend, ohbm0 (the target system).
```
q3-32b|1     ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1
   probe: 12k|16 + 12k|32  (both ~30-60x past knee -> confirm plateau)
q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker101-ceil0000-ohbm0|ligerloss1
   probe: 64k|8 + 64k|16   (both ~8-16x past knee -> confirm plateau)
```
For a throughput-vs-seq CURVE: repeat the 2-point probe at a few seqs. For a single
representative point: the pairs above suffice.

## Log
(pending first probes)

## Run protocol (decided)
- PROFILERS=source (nsys perturbs step timing), WARMUP_STEPS=1 MAX_STEPS=4 (steady =
  mid-2 of the 4 measured). Script already records steady s/it; tokens/s+TFLOPS+MFU offline.
- Dataset pool naming now includes MAX_SAMPLES: `..._s{seq}__n{MAX_SAMPLES}` (edited
  dataset_name_for_seq in both profile_lora_lf_test_{source,both}.sh). So sizes never
  collide -> reuse safely with DATASET_OVERWRITE=false; a bigger batch later just uses
  a bigger n and builds that pool once. NOTE two SEPARATE args: OVERWRITE=run-output
  overwrite; DATASET_OVERWRITE=dataset rebuild. For these probes set BOTH false.
  Settings: MAX_SAMPLES=1024, DATASET_OVERWRITE=false, OVERWRITE=false. First hit of
  each (seq,n1024) builds the pool once (existing un-suffixed 512 pools won't match);
  reused thereafter.
- MAX_SAMPLES=1024 (default 64 too small: 5-step run at batch B needs 5*B rows; batch
  40 -> 200). Set once, forget: existing pools reused (>=512 rows, enough); missing
  seqs auto-build at train-rows=1024. Bumping MAX_SAMPLES does NOT regrow an existing
  pool (build skips when files exist & no OVERWRITE=true, build script L1008) -- but
  runs cap at min(1024,pool) so it just works. No manual dataset editing.
- Datasets: existing pools are 512 rows (>=200 needed) -> NO rebuild for
  q3-32b s8192/s16000, q3-30b s24000. Only q3-32b s12288 is MISSING -> build it
  (512 rows is plenty). No need for 1024.

## Corrections / rules (2026-07-14)
- Seqs are round thousands: 8000, 12000, 16000, 24000 (NOT 8192/12288/etc).
- NO manual dataset build: the driver auto-builds any missing {seq} dataset via
  build_lf_sft_eval_pair. Just request the seq.
- OOM rule: at a given seq, sweep batch ASCENDING; on the FIRST batch that OOMs, STOP
  increasing batch for that seq (larger batches also OOM). Record the OOM batch as the
  ceiling at that seq. The last non-OOM batch's tokens/s = its max (or the plateau if
  reached earlier).
