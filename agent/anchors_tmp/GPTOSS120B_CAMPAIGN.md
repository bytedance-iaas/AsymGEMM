# gpt-oss-120b integration + tp campaign (c14, 2026-08-15)

Mirrors GPTOSS20B_CAMPAIGN.md (family #6, OWN engine AsymGptOssExperts).

## Integration facts
- Model: openai/gpt-oss-120b — 36L, 128E top-4, alternating sliding-128/full
  attention + sinks, MXFP4 experts in the HF checkpoint.
- Checkpoint: DEQUANTIZED bf16 local copy (c14-local) at
  /scratch_local/user_data/shutian/kevin/cache/fused/gpt-oss-120b-bf16
  (218G, 25 shards, manual shard writer agent/anchors_tmp/gptoss120_dequant.py
  — save_pretrained mangle precedent; per-layer expert storages verified
  distinct (72); quantization_config stripped). Map + watchdog floor (35)
  wired in profile_lora_lf_test_source.sh / run_lf_lora_sft.sh.
- FA4 auto (is_gptoss_model_name matches gpt-oss*), liger loss-only both
  sides, ASYM_OFFLOAD_MODULES=all, tier family moe, T3 = raw ker000 token
  + T3 recipe env (not route-kernel capable, by design).
- SMOKE (T1@32k b1 1r): TRAINED, loss 1.2943, 407 tok/s, 24.1 GiB reserved.
  INTEGRATION CERTIFIED 2026-08-15 04:17.

## Chains
- G1 (RUNNING): 1r turning points — rc/un/uo upward cascades to first OOM,
  then asym tier ladder T1 -> T2B -> T3-raw to deepest fit.
  Ladder: 32k..896k. Status: gptoss120_status.log.
- G2 (pending): 2r — asym_sepplanlink2_cpuadamwds + baselines, 20b-2r
  chain-E structure.

## §Results (append-only)
- [08-15 12:20] G1 COMPLETE (1r, b-walk 32k/64k, b1 beyond; eff tok/s):
    rc:  4638/5811/4646/4187/3646/3181 @32k..320k, WALL (320k,384k] G-OOM
    un:  4564/5653/4562/4138/3610/3158/2795/2500/2260 @32k..512k,
         WALL (512k,640k] G-OOM
    uo:  2238/2676/2401/2305/2151/1989/1986/1747/1625/1433 @32k..640k,
         WALL (640k,768k] G-OOM
    T1:  752/1285/1222/1598/1821/1851/1859/1834/1739/1566/1413/1270
         @32k..896k — NO WALL in the ladder (153.1G reserved @896k)
  VERDICTS: tok/s crossover — asym T1 leads every SURVIVOR from 512k
  (1739 vs uo 1625); CAPACITY TURNING POINT 768k — asym alone from there
  (uo dead >640k). Loss parity per column (e.g. 128k: 1.2448/1.2399/
  1.2415/1.2437). G1B extension RUNNING: T1 1.02M -> 1.41M w/ T2B/T3
  fallthrough to find the asym wall/crown.
- [08-15 18:30] G1B COMPLETE: T1 1.02M/1.15M/1.28M TRAINED; 1.41M T1
  G-OOM, T2B COOM, T3 COOM -> **asym WALL (1.28M,1.41M] host-bound;
  CROWN = T1 @1.28M — 2.0x uo's 640k, alone from 768k**.
- [08-16 01:45] G2 COMPLETE + BANKED + PUSHED (7b4e2e4). 2r (GLOBAL
  tok/s, sepplanlink2): rc 9851/12008/9569/8395/7275/6323 @32k..320k,
  wall (320k,384k]; un ../4440 @512k, wall (512k,640k]; uo
  3516/../2925 @320k, wall (320k,384k] HOST (per-rank machinery
  duplicates under DP — collapses from 1r 640k); asym plain-T1
  1317/2702/2560/3254/3677/3801 @32k..320k (DP ~103% at 320k), then
  T1+ohbm4+gradoff 3708/3607/3466/3117 @384k..640k, wall (640k,768k]
  host (ohbm2 probe also COOM). TURNING POINT 640k (asym alone; 1.25x
  un). Stock-tier 2r ladder (no ohbm/gradoff) walls at (320k,384k] all
  three tiers — the q122-precedent dials are REQUIRED beyond 384k.
  Rows banked in plot_tp_vs_seq.py + _2r.py (lean-six drops added);
  figures regenerated + pushed. CAMPAIGN COMPLETE.
