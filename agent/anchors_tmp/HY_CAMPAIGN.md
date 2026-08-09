# Hunyuan-A13B integration + throughput campaign (c17, 2026-08-06/07)
User directive: integrate fully like the other models, sanity-check first,
then 1-rank and 2-rank throughput panels with enough turning points to show
asym* last standing. DONE — both panels generated, verified, pushed
(overleaf 27db222). ~60 measured cells; harvest = hy_harvest.py ->
hy_cells.json; verdicts = hy_status.log; chains = hy_master_chain.sh,
hy_1r_ext.sh, hy_2r_final.sh (+ superseded hy_2r_ext*.sh attempts).

## Recon (why it was safe)
HunYuanMoEV1ForCausalLM / hunyuan_v1_moe, 80B total 13B active, 32L GQA 32:8,
64 experts top-8 + 1 shared/layer, qk-norm, tied embed/lm_head, dynamic-NTK
rope (native 32k -> ~256k), vocab 128k. NO sliding window / CLA / MLA — the
phi3.5 mask-wall class cannot occur. Integration #3 already validated
in-tree (hunyuan_moe.py identical to SFT-39; LF wrap + liger loss-only +
watchdog floor 50 present). Bank ~160 GB.

## Integration deltas found by the smoke gate (env-level, no code)
1. asym offload selection must EXCLUDE embed/lm_head: hunyuan TIES them and
   the offload stage rejects tied pairs (~1.05 GB stays HBM-resident).
   ASYM_OFFLOAD_MODULES="routed_experts,shared_experts,attention,norms,mlp_dense"
2. router token excluded: HunYuanMoEV1Gate is a wrapper module (AsymQwen3Router
   wants a 2D weight) — kept intact on GPU, glm-DS-bias-style (~0.5 MB/layer).
3. asym_sdp2 requires |2 spec (1-rank uses asym_cpuadamwds).
4. T3 tier = ker101 qwen-only routed kernels -> config-rejected for hunyuan
   (recorded; deepest valid tiers: T1, T2B/ker000).

## Smoke gate (16k b1, 1 rank)
asym-T1 1774 tok/s (18.0 GiB; hunyuan_moes_wrapped>=1 proof) vs SO-recomp
1231 (28.2) — trains, loss sane, packed grouped-GEMM + shared expert engaged.

## 1-RANK results (GPU0; eff tok/s; resv GiB)
| seq | rc | uns | uns-off | asym best |
|----|----|----|---------|-----------|
|32k |3017 b4|2988 b4|1632 b4|**3131 b8 T1**|
|64k |2270|2254|1387|**2347 T1**|
|128k|1526|1515|1067|**1559 T1**|
|192k|1160 (98%)|1164|863|1154 T1 (-0.9%, band)|
|256k|GOOM|881|731|**929 T1**|
|320k|—|GOOM|630|**672 T1**|
|384k|—|—|555|**589 T2B**|
|448k|—|—|495|**524 T2B**|
|512k|—|—|449 (98%)|**468 T2B (97%)**|
|576k|—|—|HOST-C-OOM|HOST-C-OOM (T2B)|
Walls: rc (192k,256k] G · uns (256k,320k] G · uns-off (512k,544k] C · asym
T1->T2B (512k,544k] C (544k probe pair MEASURED 08-07: both HOST-C-OOM; 576k
COOMs = outer confirmation). Deep end = physical TIE at 512k, asym +4-6%
there and leads/ties every rung.

FUTURE (user, 08-07): "T2A" attn-only intermediate tier — T2B preset +
ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1 override (or coarse recomp-off-attn token
+ hand CPU-stack env). Expected: ceiling > T1, tok/s > T2B; slot 384-448k.
NOT run yet — user parked it.

## 2-RANK results (GPUs 0+1, sdp2 shared fabric; GLOBAL tok/s)
| seq | rc | uns | uns-off | asym best |
|----|----|----|---------|-----------|
|32k |5455|5251|2006|**6348 b8 T1**|
|64k |4182|4093|1822|**4734 T1**|
|128k|2887|2830|1510|**3138 T1**|
|192k|2133|2254|HOST-C-OOM|**2316 T1**|
|256k|GOOM|1586 (98%)|—|**1864 T1 (+18%)**|
|288k|—|GOOM (measured)|—|**687 T1 (99% edge-tax) SOLE**|
|320k|—|—|—|**672 T2B (120 GiB) SOLE**|
|384k|—|—|—|HOST-C-OOM|
Walls: uo (128k,192k] C (DP-asymmetry class) · rc (192k,256k] G · uns
(256k,288k] G · asym (320k,384k] C. ASYM SOLE SURVIVOR 288k+320k, leads every
rung. T2B@2r needs ASYM_ARENA_SHM_CAP_GB=320 (bank+fg bases ~300 GB shared).


## T3 ENABLEMENT (2026-08-08, user directive: "fix T3 for hunyuan")
Recon: the ker101 route kernels are FAMILY-AGNOSTIC (shape-generic NVRTC JIT,
no model checks in asym_gemm; hunyuan wraps AsymQwen3Experts verbatim). The
only blockers were two driver gates. Change: new is_route_kernel_capable_model
predicate (qwen paths byte-identical; only hunyuan added; auto-default stays
qwen-only; mixtral/phi/glm still rejected). No asym_gemm/csrc changes, no
rebuild. Numeric probe gained --hunyuan shapes (E=64 H=4096 I=3072 top-8).

Correctness (3 levels, all PASS):
- GATE0 numeric: fg101 fwd + all LoRA grads vs fp32 reference at hunyuan
  shapes, both LoRA-B regimes ("all forward paths within 5% of fp32").
- GATE1 real model @16k same-seed: T3 vs T2B loss parity rel 0.64% max;
  route counters T3 fwd_scatter=192 dx_scatter=192 gather=0 avoided=384,
  T2B all zero; run dir carries ker101/route101 labels.

Throughput + ceiling (all MEASURED):
- 1r: T3 509@384k / 417@512k (T2B faster there: 589/468 — on hunyuan T3 is
  leaner (-14 GiB) but slower (-14%), unlike qwen) — then T3 ALONE:
  **401@544k, 382@576k, 367@608k** (89->98% resv; rss 901-906/957). The
  512k tie with uns-off is BROKEN: asym last-standing by >=2 rungs (uns-off
  and T2B walls (512k,544k]). 640k bracket probe in flight.
- 2r: T3 320k = **1142** (66%/rank, arena 320) vs T2B 672 (+70%) — the 2r
  crown bar upgrades; wall (320k,384k] HOST-C-OOM unchanged (2r is
  host-pool-bound; kernels can't move it).


## MIXTRAL T3 (2026-08-08, follow-on): enabled + correct, NOT beneficial
Same enablement path (is_route_kernel_capable_model + probe --mixtral shapes
E=8 H=6144 I=16384 top-2). Gates: numeric fp32-parity PASS; real-model pair
@16k loss parity ~1% + counters (T3 336/336/0/672, T2B zeros) PASS.
Measured: T3-real @320k 1r = 367 tok/s (136.2 GiB, rss 912) vs house T2B 594
— SLOWER; wall (320k,352k] HOST-C-OOM = no ceiling gain (host-bound).
WHY: ker101's payoff is avoiding the [R,H] route-space copies with
R = T x top_k. top-8 (hunyuan/qwen) => 8x multiplier = leaner + deep-rung
crowns; top-2 (mixtral) => ~nothing saved while T3 still drops the staged
dispatch on I=16384 experts. RULE OF THUMB: route kernels pay ∝ top_k;
keep T2B as the deep tier for low-top-k families. Figure rows unchanged.

- MIXTRAL 2r T3 probe (08-09): fabric (bank+fg) filled shm to ~460+ GB
  (under the 479 cap — the 562 arithmetic was pessimistic) but the node
  HOST-C-OOMed during load/warmup @304k: fabric + 2x per-rank pools exceed
  the 957 GB pool. 2r T2B/T3 on mixtral = host-infeasible on this node
  (measured COOM; T1 remains the only viable 2r mixtral tier). No ceiling
  change either rank count from T3 on this family.

## Ops lessons
- **Leaked shm kills everything downstream**: SIGBUS/ENOSPC runs leave stale
  /dev/shm/asym_fabric_* segments (479 GB tmpfs filled; two T2B "FAILs" and a
  poisoned 288k pair were contamination, not physics). rm the segments +
  shm guard (<5 GB) before every fabric launch; clean-shm rerun TRAINED.
- Mock/web-corpus datasets contain log-looking junk — grep verdicts from
  jobs.tsv + targeted '[rankN]: Traceback' line numbers, not free-text 'error'.
- 288k asym T1 = 99%-resv edge-tax rung (687 vs 1864@256k): fits-where-uns-
  dies is the claim; tok/s at edge rungs is disclosed as edge-taxed.

## Deliverables
- DATA rows "hunyuan-a13b" in BOTH /home/kevinni/env/figures/plot_tp_vs_seq
  {,_2r}.py (+ OOM_TIP, LEAN_DROP, COMBINED_KEYS -> 9-panel grids), figures
  regenerated + rasterize-verified + synced (out/, outputs/c14, overleaf
  27db222). NeMo verdict for hunyuan: not run (bridge absent from Megatron-
  Bridge registration matrix — probe pending only if a figure row is wanted).
