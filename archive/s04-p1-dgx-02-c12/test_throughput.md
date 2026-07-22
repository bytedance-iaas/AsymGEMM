# test_throughput — protocol, rules, insights (host s04-p1-dgx-02-c12)

RECONSTRUCTED 2026-07-17 after commit 01c56d8 deleted this file (uncommitted working-tree
state was the only copy). LAYOUT CHANGE: all metric tables now live ONLY in
`test_throughput_results.md` (same dir) — this doc is protocol + rules + insights + queue.

HOST: s04-p1-dgx-02-c12 | GPU: GB200 (189471 MiB = 185.0 GiB HBM) | container asym_sft_40, venv torch 2.12.
Per-host isolation: this dir = this machine's record; other machines write agent/impls/<hostname>/
(handoff prompt: agent/impls/throughput_prompt.md -> ../../../../env/agent/throughput_prompt.md).
Raw artifacts (gitignored): profiling_results/profiling_tp_s04-p1-dgx-02-c12/asym_long_sft_smoke__lora__lf__bf16/
— live runs write profiling_results/profiling/... and are mv'd there after each campaign.

================================================================================
## GOAL (updated 2026-07-16: GAP-SEARCH, not max-tp squeeze)
================================================================================
Find seq lengths where superoffload FITS but its max fitting batch (B_max) sits BELOW its
throughput knee — capacity caps batch before throughput saturates — so AsymGEMM (leaner
HBM/sample) can run a larger batch at the same seq and win tok/s while both fit.
  gap condition: (a) B_max <= ~4  AND  (b) MFU(B_max) < plateau MFU
                 (plateaus: q3-32b ~31%, q3-30b-a3b ~13-15%, llama ~40%)
  severity     : saturation deficit = 1 - MFU(B_max)/plateau  (rank seqs by this)
Counterexample: llama recomp 48k caps at b2 yet MFU 40.1% = saturated -> NO gap (small B_max
alone is not enough). Exemplar: llama unsloth 128k b2 = 32.6% vs 40% (~18% deficit, 80% HBM).
A gap point MUST fit healthily (<=92% HBM, no thrash). OOM rows are only B_max brackets.
The capacity story (asym max-seq > superoffload) is already proven separately — never re-derive.

## BEHAVIORAL RULES (agent follows WITHOUT waiting for the user)
- R0 post-sweep review: after any sweep, insert missing batches to pin true peaks (historical
  mode; superseded by DEEP-END policy for the gap search).
- R1 interpolate to pin peaks between good..OOM brackets.
- R2 an OOM brackets the search; it does not end the seq.
- R3 near-ceiling thrash (resv >97% phys + step time 1.5x+) is a REAL regime — record as
  `thrash`, distinct from OOM. A run with 0 measured steps is IN-PROGRESS/KILLED, not OOM
  (verify step_samples.json).
- R4 GiB for all HBM numbers; physical = 189471 MiB = 185.0 GiB. RSS in GB.
- R5 STRICTLY SERIAL: one training job, one GPU, at a time. Kill by PID from
  nvidia-smi --query-compute-apps (procs are setsid'd; pkill-by-name misses them).
- R6 RUN_NAME must include the model (else cross-model dir collision overwrites data).
  Tags: tput_<model> (unsloth), tputrc_<model> (recomp), tputasL_<model> (asym latency; dirs
  lowercase to tputasl_).
- DEEP-END policy (2026-07-17): no ascending sweeps, no seq-walking through saturated
  territory, no re-running known points. Predict B_max from anchors (HBM = base +
  per_sample*B; per_sample ~linear in seq), probe DESCENDING, 1-2 runs/seq. b8 seq-ceilings
  (user-verified): q3-32b 49k|8 (G-OOM 50k), q3-30b 80k|8 (G-OOM 81k), llama 45k|8 (G-OOM 46k)
  -> NEVER probe below these; gap zone = 2-4x above.
- Failure classification in launchers: OOM pattern -> try smaller batch; "profiling job(s)
  failed"/Traceback/Training command failed -> HARDFAIL, abort probe loudly (a fast-completing
  run with no run dir = silent failure, investigate; e.g. dataset_info deregistration).

## METRIC (offline from one measured step-time t; profile.json)
```
s/it   = step.rows[name=="lf.step.total"].ms/1000   (steady: w1 + m4, PROFILERS=source)
tok/s  = B*s/t
TFLOPS = (6*N_active*(B*s) + 12*L*h*s*(B*s)*0.5) / t / 1e12   # causal; NO recompute FLOPs
MFU    = TFLOPS / 2250 (GB200 bf16)
arch: q3-32b 32.8e9/L64/h5120 | q3-30b-a3b 3.34e9/L48/h2048 | llama3.3-70b 70.6e9/L80/h8192
resv/alloc GiB = memory.peak_*_hbm_bytes/2^30 ; RSS GB = memory.process.rss_peak_bytes/1e9
```

## RUN PROTOCOL
```
export PROFILERS=source MAX_STEPS=4 WARMUP_STEPS=1 MAX_SAMPLES=1024 DATASET_OVERWRITE=false OVERWRITE=false
RUN_NAME="<tag>_<model>" RUNS="<model>|1 ; <config> ; <seq>|<B>|1 ; none|false|false|false|false|false" \
  bash scripts/lf/profile_lora_lf_test_source.sh
configs: superoffload_mem|unsloth-ohbm0|ligerloss1 ; superoffload_mem|recomp|ligerloss1
asym latency (dense): asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1 +
  env ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYM_GEMM_DISPATCH=staged
      ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM=1   (MoE: ker101 + ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1)
asym runs REQUIRE scripts/lf/rebuild_asymgemm.sh after any asym-source change (venv torch, never system).
```

## KEY INSIGHTS (measured)
- Superoffload is COMPUTE-BOUND at its peak: MFU flat vs seq (dense-32B ~31%, MoE ~12-15%,
  llama-70B ~40%) until the gap regime. Per-token fwd ~82us / bwd ~204us flat across batch
  (q3-32b 8k breakdown); only ~300ms fixed optimizer cost amortizes -> bigger batch past the
  knee adds HBM, not tok/s; pushed into the ceiling it THRASHES then OOMs.
- CAPACITY: unsloth-ohbm0 > recomp everywhere (activations offloaded to host -> leaner HBM;
  cost = host RSS 364-865 GB vs 142-300 GB). recomp always dies by OOM. The old claim
  "unsloth OOMs past 24k" was never measured and is FALSE.
- GAP MECHANISMS (two distinct, measured on llama):
  (1) BATCH-KNEE POCKET (recomp): deficit only where B_max*seq < knee_tokens (~70k for llama rc)
      -> a narrow pocket right past the b2 wall: 56k b1 = ▼9%, but 80k b1 back to ▼1% (tokens/step
      grows with seq at b1 and re-amortizes). Dense-32B behaves similarly (deficit saturates 7-9%).
  (2) SEQ-DRIVEN OFFLOAD OVERHEAD (unsloth): deficit GROWS with seq regardless of tokens/step
      (llama 112k/128k/144k b2 = ▼17/18/20% at 224-288k tok/step, far above any knee).
  MoE q3-30b gaps at 128k (uns b4 ▼13%, rc b2 ▼16%) — mechanism assignment pending c14 probes.
- llama asym latency @128k: b4 OOM, b3 fits -> asym capacity > superoffload (b2) at the gap
  seq even in latency mode. tok/s head-to-head PENDING (asym on hold).

## QUEUE
1. [DONE] superoffload gap map, all 3 models x 2 configs, c12 partition COMPLETE through the
   b1/b2 frontier — see GAP WINDOW blocks in results.md. Windows: q3-32b uns ▼29-32% @128-160k(b2);
   llama uns ▼17-21% @112-192k(b1/b2); q3-32b rc ▼6-10% @45-96k; llama rc = narrow b1 pocket only.
2. [c14] q3-30b deep probes 160k+ (both configs; 128k = ▼13%/▼16% measured here).
3. [ON HOLD per user] Phase B asym head-to-head at gap seqs. Partial: llama 128k b4 OOM /
   b3 fits (killed mid-run). _C rebuilt+verified. Targets when resumed: q3-32b 128k/160k (sup.
   1110/942 tok/s to beat), llama 128k/144k (792/732). HOST PARTITION: c12 = llama+q3-32b.
4. (optional) llama uns 256k b1 (frontier extends ~300k); q3-32b uns 180k b2-wall pin;
   llama rc 64k b1 pocket-interior point; asym 24k completion.

## INCIDENT LOG
- 2026-07-16 RUN_NAME collision (R6 born): fixed by model-tagged RUN_NAME.
- 2026-07-16 _source.sh silently dropped asym latency flags -> memory-mode runs; fixed
  (6 forwarding lines) + verified dispatch=staged.
- 2026-07-17 user repo reorg reverted ~2h of doc edits -> restored from session record.
- 2026-07-17 dataset_info.json lost 80 generated-dataset entries (reset + pre-flock races)
  -> builder validation_ok=False -> silent row aborts. Repaired via .repair_dataset_info.py
  (re-registers all on-disk asym_long_sft_smoke__*.jsonl under flock).
- 2026-07-17 commit 01c56d8 deleted this dir's two docs (uncommitted latest state) ->
  reconstructed from session record. COMMIT THESE FILES to protect them.
