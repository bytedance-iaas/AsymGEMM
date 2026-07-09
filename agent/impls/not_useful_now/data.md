# Long-Context LoRA-SFT Data and Evaluation Plan

## Goal

Build long-context supervised fine-tuning datasets that make LoRA-SFT runs both
memory-stressing and quality-checkable. Every quality run uses at least two
training sources, trains on several thousand long examples, and evaluates on
multiple held-out or external datasets. Profiling runs keep their short
fixed-step mode, but they draw from the same data registry so timing, memory,
and quality runs are comparable.

This plan is for LoRA-SFT method comparison across `torch`, DeepSpeed zero
policies, `asym`, and KT backends. It is not a pretraining data plan.

## Local Constraints

- `scripts/lf/run_lf_lora_sft.sh` trains one LlamaFactory dataset name through
  `DATASET=...`, `CUTOFF_LEN=...`, and `MAX_SAMPLES=...`.
- `scripts/lf/profile_lora_lf.sh` currently generates one model/sequence
  dataset through `build_lf_sft_eval_pair.py`; the current source is
  `HuggingFaceTB/smoltalk`, config `longalign`.
- LlamaFactory generation metrics are blocked when `use_asym_gemm` or `use_kt`
  is active. Quality evaluation must load the saved adapter through normal
  HF/PEFT inference with `use_asym_gemm=false` and `use_kt=false`.
- Profiling runs use `save_strategy=no` to avoid extra I/O. Quality runs need a
  separate adapter-saving path.

## Dataset Requirements

- Each train round must contain at least two source datasets.
- `quality_small` uses 4,096 train rows and at least 128 held-out eval rows per
  source dataset.
- `quality_full` uses 16,384 train rows and at least 512 held-out eval rows per
  source dataset.
- For `CUTOFF_LEN=4096`, every selected train row and every held-out eval row
  must have at least 4,096 total SFT tokens under the target model tokenizer.
  Total SFT tokens means prompt/context tokens plus response tokens after the
  prompt builder runs.
- For `CUTOFF_LEN=8192`, every selected train row must have at least 8,192
  total SFT tokens. For sources with insufficient 8k rows, the builder excludes
  the source from that sequence-length run instead of silently padding with
  short examples.
- A source whose median sampled total SFT length is below 4,096 tokens is not a
  default train or held-out eval source.
- If filtering leaves fewer than the requested train or eval rows for a source,
  fail dataset construction with:
  `"insufficient >=4096-token rows for <source> at cutoff_len=<N>"`.
- Token stats must be recorded per source and per generated mix:
  `count`, `min`, `p50`, `p75`, `p90`, `max`, `above_cutoff_len`, and
  `truncated_by_cutoff`.
- Train/eval overlap is forbidden. The builder deduplicates by normalized
  conversation hash, source id, source split, and `(question, answer)` hash for
  QA-style datasets.
- External benchmark dev/test splits are never used for training.

## Selection Rationale

| source | reason it belongs |
|---|---|
| LongAlign-10k | Purpose-built long instruction data, 8k-64k length range, aligned with long-context SFT stress. |
| LongAlpaca-12k | Long-context instruction-following data with 9k long QA rows plus 3k short QA rows for instruction preservation. |
| SmolTalk LongAlign | Already used locally; its LongAlign subset was selected by SmolTalk specifically to preserve long-context ability beyond short SFT. |
| LongWriter-6k | 6k SFT rows with 2k-32k word outputs; this directly tests long-output adapter behavior. |
| GovReport, QMSum | Human summarization tasks with long input documents or transcripts and standard train/validation/test splits. |
| Qasper, NarrativeQA | Long-document QA with full papers or full stories, giving real long-context answer extraction pressure. |
| LongBench, RULER, HELMET, ZeroSCROLLS | External eval suites covering realistic long-context tasks, synthetic retrieval/aggregation stress, and eval-only contamination checks. |

Rejected short default sources:

| source | Qwen3 sampled median total tokens | decision |
|---|---:|---|
| Multi-News | ~2.0k | exclude from default train/eval; only usable after a builder filter proves enough rows at `>=4096` tokens. |
| HotpotQA distractor | ~1.4k | exclude from default train/eval. |
| MuSiQue | ~2.4k | exclude from default train/eval. |

Optional raw-LM stress source:

| source | decision |
|---|---|
| The Pile | exclude from default LoRA-SFT quality rounds because it is raw language-modeling text, not prompt/response SFT data. Use only for an explicit `pile_lm_long_v1` raw-LM stress/eval round with rows filtered to `>=4096` tokens. |
| Common Pile | same role as The Pile, with cleaner licensing. Prefer this over the original Pile for new raw-LM stress experiments when its selected subsets meet the token filter. |

`pile_lm_long_v1` is a separate continued-pretraining/LM-loss profile. It uses
raw text rows, computes continuation loss, and does not count as SFT
correctness evidence.

## Token Threshold Background

Counts below use `Qwen/Qwen3-30B-A3B` tokenizer and total SFT tokens:
prompt/context tokens plus answer tokens after the planned prompt builder. Rows
marked exact were fully scanned. Rows marked estimated were sampled and scaled
to the source row count.

Default SFT candidate sources:

| source | rows | >=4k rows | >=8k rows | >=12k rows | count type | decision |
|---|---:|---:|---:|---:|---|---|
| LongAlign-10k | 9,888 | ~9,888 / 100.0% | ~8,756 / 88.6% | ~4,776 / 48.3% | estimated from 2,000 rows | default train/eval source |
| LongAlpaca-12k | 12,000 | ~8,992 / 74.9% | ~5,652 / 47.1% | ~3,736 / 31.1% | estimated from 3,000 rows | default train/eval source |
| SmolTalk LongAlign | 3,547 | 3,547 / 100.0% | 2,668 / 75.2% | 832 / 23.5% | exact | default train/eval source |
| LongWriter-6k | 6,000 | 3,039 / 50.7% | 575 / 9.6% | 172 / 2.9% | exact | default long-output source at 4k; selective at 8k/12k |
| GovReport | 17,517 | ~15,695 / 89.6% | ~9,757 / 55.7% | ~5,422 / 31.0% | estimated from 2,000 rows | default train/eval source |
| QMSum | 1,257 | 1,179 / 93.8% | 948 / 75.4% | 599 / 47.7% | exact | default eval/mix source; small but long |
| Qasper | 2,567 | 1,707 / 66.5% | 257 / 10.0% | 88 / 3.4% | exact | default 4k QA source; thin at 8k/12k |
| NarrativeQA | 32,747 | ~32,747 / 100.0% | ~32,379 / 98.9% | ~32,051 / 97.9% | estimated from 800 rows | default QA source; requires windowing/truncation |

AllenAI OLMo-family post-training sources:

| source | rows | >=4k rows | >=8k rows | >=12k rows | count type | decision |
|---|---:|---:|---:|---:|---|---|
| Tulu 3 SFT | 939,343 | ~13,738 / 1.5% | ~4,579 / 0.5% | ~2,231 / 0.2% | estimated from 8,000 rows | not default; filtered 4k auxiliary source only |
| Tulu 3 preference | 272,898 | ~4,742 / 1.7% | ~307 / 0.1% | ~0 / 0.0% | estimated from 8,000 rows | exclude from SFT; preference/DPO data |
| Dolci-Think-SFT | 2,268,178 | ~1,077,385 / 47.5% | ~801,595 / 35.3% | ~664,989 / 29.3% | file-stratified estimate from 880 rows | non-default `lc_reasoning_trace_v1` long reasoning SFT source |
| Dolci-Think-DPO | 150,000 | ~30,670 / 20.5% | ~13,661 / 9.1% | ~7,098 / 4.7% | file-stratified estimate from 1,120 rows | exclude from SFT; useful only after adding DPO benchmarking |

Operational conclusions:

- LongAlign-10k, LongAlpaca-12k, GovReport, and NarrativeQA have enough rows
  for 4k/8k/12k stress profiles.
- SmolTalk LongAlign, QMSum, and Qasper are useful long eval/mix sources, but
  their row counts limit full-size training mixes.
- LongWriter-6k is useful for long-output behavior at 4k and for smaller 8k/12k
  sweeps.
- Dolci-Think-SFT is the only AllenAI post-training source above that is both
  SFT and dense enough in long rows to justify a non-default long reasoning
  round.
- Tulu 3 SFT is too short-dense for the default long-context benchmark despite
  having some filtered long rows.

## Training Rounds

### `lc_align_v1`

Purpose: general long-context instruction following and router/expert stress.

Training sources:

| source | loader | splits | format conversion | target share |
|---|---|---|---|---|
| LongAlign-10k | `zai-org/LongAlign-10k` | train only | `messages` to ShareGPT | 45% |
| LongAlpaca-12k | `Yukang/LongAlpaca-12k` | train only | Alpaca `instruction`/`output` to ShareGPT | 35% |
| SmolTalk LongAlign | `HuggingFaceTB/smoltalk`, config `longalign` | train/test | `messages` to ShareGPT | 20% |

Held-out eval:

- `lc_align_v1__eval_longalign`: withheld LongAlign rows.
- `lc_align_v1__eval_longalpaca`: withheld LongAlpaca rows.
- `lc_align_v1__eval_smoltalk_longalign`: SmolTalk `longalign` test split.
- External: LongBench single-document QA, multi-document QA, summarization, and
  RULER at the same context length. The evaluator filters external examples
  below 4,096 total prompt/reference tokens before scoring.

Expected signal:

- Held-out SFT loss drops versus the base model on at least two held-out eval
  datasets.
- LongBench average over the selected long-context subsets improves versus the
  base model.

### `lc_summarize_write_v1`

Purpose: long input summarization plus long output generation. This round gives
the clearest quality signal because summarization metrics respond quickly to
domain SFT.

Training sources:

| source | loader | splits | format conversion | target share |
|---|---|---|---|---|
| LongWriter-6k | `zai-org/LongWriter-6k` | train only | `messages` to ShareGPT | 35% |
| GovReport | `ccdv/govreport-summarization`, config `document` | train/validation/test | prompt from report, response from summary | 40% |
| QMSum | official `Yale-LILY/QMSum` data | train/validation/test | prompt from query + transcript, response from summary | 25% |

Held-out eval:

- `lc_summarize_write_v1__eval_longwriter`: withheld LongWriter rows.
- `lc_summarize_write_v1__eval_govreport`: GovReport validation/test rows.
- `lc_summarize_write_v1__eval_qmsum`: QMSum validation/test rows.
- External: LongBench `gov_report`, `qmsum`; LongBench-Write and
  LongWrite-Ruler for the long-output subset. The evaluator filters external
  examples below 4,096 total prompt/reference tokens before scoring.

Expected signal:

- ROUGE-L and exact generated-length compliance improve on at least two of
  GovReport, QMSum, and LongWriter held-out eval.
- The adapter does not collapse short answer behavior: `lc_align_v1` held-out
  loss is no worse than the base model by more than 3%.

### `lc_qa_reason_v1`

Purpose: long document QA and multi-hop QA with long source contexts.

Training sources:

| source | loader | splits | format conversion | target share |
|---|---|---|---|---|
| Qasper | `urialon/converted_qasper` or official Qasper conversion | train/validation/test | prompt from full paper + question, response from answer | 50% |
| NarrativeQA text-only | `meithnav/narrativeqa` | train/validation/test | prompt from story + question, response from answer | 50% |

Held-out eval:

- `lc_qa_reason_v1__eval_qasper`: Qasper validation/test rows.
- `lc_qa_reason_v1__eval_narrativeqa`: NarrativeQA validation/test rows.
- External: LongBench `narrativeqa`, `qasper`, and RULER QA/retrieval tasks.
  The evaluator filters external examples below 4,096 total prompt/reference
  tokens before scoring.

Expected signal:

- Normalized exact-match or token-F1 improves on at least two held-out QA eval
  datasets.
- RULER retrieval/QA accuracy does not regress versus the base model at the
  trained context length.

## Data Processing

Add a registry file:

```text
scripts/lf/long_sft_datasets.yaml
```

Each registry entry contains:

- `name`
- `hf_id` or `download_url`
- `config`
- `train_split`
- `eval_split`
- `license`
- `format`
- `prompt_builder`
- `response_field`
- `min_context_tokens`
- `max_output_tokens`
- `eval_metric`
- `source_url`

Replace the single-source builder with a registry-driven builder:

```text
scripts/lf/build_lf_long_sft_mix.py
```

Required CLI:

```bash
python scripts/lf/build_lf_long_sft_mix.py \
  --lf-dir /workspace/AsymGEMM-SFT/third_party/LlamaFactory \
  --asym-dir /workspace/AsymGEMM-SFT/third_party/AsymGEMM \
  --model-name-or-path Qwen/Qwen3-30B-A3B \
  --round lc_summarize_write_v1 \
  --profile quality_small \
  --seq-len 4096 \
  --train-name asym_lc_summarize_write_v1__qwen3-30b-a3b__s4096 \
  --eval-prefix asym_lc_summarize_write_v1__qwen3-30b-a3b__s4096__eval \
  --seed 0
```

Builder behavior:

1. Load every source from the round registry.
2. Convert rows to LlamaFactory ShareGPT format:
   `{"conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": response}], "system": "", ...}`.
3. Tokenize with the target model tokenizer and the selected LlamaFactory
   template.
4. Keep rows satisfying the sequence-length rule for the run.
5. Deduplicate across all train and eval sources.
6. Sample deterministic source-balanced train rows according to the round table.
7. Write one mixed train JSONL and one eval JSONL per eval source.
8. Register every generated dataset in
   `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/data/dataset_info.json`.
9. Write a manifest next to `lf_dataset_manifest.json` with source counts,
   token stats, hashes, and source URLs.

Generated dataset names:

```text
asym_<round>__<model_tag>__s<seq>
asym_<round>__<model_tag>__s<seq>__eval_<source>
```

Example:

```text
asym_lc_summarize_write_v1__qwen3-30b-a3b__s4096
asym_lc_summarize_write_v1__qwen3-30b-a3b__s4096__eval_govreport
asym_lc_summarize_write_v1__qwen3-30b-a3b__s4096__eval_qmsum
```

## Script Integration

Update `scripts/lf/profile_lora_lf.sh`:

- Add `DATA_ROUNDS=${DATA_ROUNDS:-lc_align_v1}`.
- Replace the current `prepare_dataset_for_seq` call with
  `build_lf_long_sft_mix.py --round "${data_round}"`.
- Set the default `DATASET` to the selected round name, not to the old smoke
  name.
- Keep profiling `MAX_SAMPLES=128` and `MAX_STEPS=10` defaults for timing and
  memory sweeps.

Update `scripts/lf/run_lf_lora_sft.sh`:

- Add `SAVE_ADAPTER=${SAVE_ADAPTER:-false}`.
- When `SAVE_ADAPTER=true`, set `--save_strategy steps`,
  `--save_steps "${SAVE_STEPS}"`, and write the final adapter under
  `${OUT_DIR}/adapter`.
- Keep `save_strategy=no` for profiling runs.

Add a quality runner:

```text
scripts/lf/run_lf_lora_quality.sh
```

Runner behavior:

1. Build the selected data round.
2. Train each backend with adapter saving enabled.
3. Evaluate the base model and each saved adapter through normal HF/PEFT
   inference.
4. Write `quality_metrics.json`, `quality_metrics.csv`, and
   `quality_summary.md` under:

```text
results/lf_quality/<round>/<model_tag>/s<seq>/<backend>/<seed>/
```

## Evaluation Protocol

For each round, model, backend, sequence length, and seed:

1. Evaluate the base model on all held-out eval datasets.
2. Train the adapter on the mixed train dataset.
3. Evaluate the saved adapter on all held-out eval datasets.
4. Evaluate the saved adapter on the external long-context benchmark subset for
   that round.
5. Compare adapter quality against the base model and compare backend-produced
   adapters against the `zero3` baseline.

Use these metrics:

| task type | metrics |
|---|---|
| SFT held-out loss | `eval_loss`, perplexity |
| summarization | ROUGE-1, ROUGE-2, ROUGE-L, generated-token length |
| QA | normalized exact match, token-F1 |
| long output | generated-token length, LongBench-Write score, LongWrite-Ruler pass rate |
| LongBench | official task score and category average |
| RULER | official task accuracy |
| HELMET | official category score |

Quality acceptance gates:

- The trained adapter improves over the base model on at least two held-out eval
  datasets for the round.
- The trained adapter improves over the base model on at least one external
  benchmark category for the round.
- For a fixed seed and data round, `asym` final held-out loss is within 2% of
  the `zero3` adapter. Task metrics are within 1 point of `zero3` on the average
  of the round's held-out eval datasets.
- Dataset validation passes before training starts.

## Validation Commands

Dataset build:

```bash
python scripts/lf/build_lf_long_sft_mix.py \
  --lf-dir /workspace/AsymGEMM-SFT/third_party/LlamaFactory \
  --asym-dir /workspace/AsymGEMM-SFT/third_party/AsymGEMM \
  --model-name-or-path Qwen/Qwen3-30B-A3B \
  --round lc_align_v1 \
  --profile quality_small \
  --seq-len 4096 \
  --seed 0
```

Quality run:

```bash
scripts/lf/run_lf_lora_quality.sh \
  --rounds lc_align_v1,lc_summarize_write_v1,lc_qa_reason_v1 \
  --models Qwen/Qwen3-30B-A3B \
  --seq-lens 4096 \
  --backends zero3,asym \
  --profile quality_small \
  --seed 42
```

Profiling run using the same registry:

```bash
scripts/lf/profile_lora_lf.sh \
  --dataset lc_align_v1 \
  --seq-lens 4096 \
  --backend-specs zero3_offload\|recomp,asym\|recomp \
  --max-samples 128 \
  --max-steps 10
```

Post-build checks:

```bash
python -m json.tool /workspace/AsymGEMM-SFT/third_party/LlamaFactory/data/dataset_info.json >/dev/null
rg -n "asym_lc_.*__eval_" /workspace/AsymGEMM-SFT/third_party/LlamaFactory/data/dataset_info.json
python scripts/lf/build_lf_long_sft_mix.py --audit-only --round lc_align_v1 --seq-len 4096
```

## Source Pages

| item | source |
|---|---|
| LongAlign-10k | https://huggingface.co/datasets/zai-org/LongAlign-10k |
| LongAlign code/eval | https://github.com/THUDM/LongAlign |
| SmolTalk LongAlign subset | https://huggingface.co/datasets/HuggingFaceTB/smoltalk |
| LongAlpaca-12k | https://huggingface.co/datasets/Yukang/LongAlpaca-12k |
| LongLoRA/LongAlpaca code | https://github.com/JIA-Lab-research/LongLoRA |
| LongWriter-6k | https://huggingface.co/datasets/zai-org/LongWriter-6k |
| LongWriter code/eval | https://github.com/THUDM/LongWriter |
| GovReport | https://huggingface.co/datasets/ccdv/govreport-summarization |
| QMSum | https://github.com/Yale-LILY/QMSum |
| Qasper | https://huggingface.co/datasets/allenai/qasper |
| NarrativeQA text-only | https://huggingface.co/datasets/meithnav/narrativeqa |
| The Pile | https://huggingface.co/datasets/EleutherAI/pile |
| Common Pile | https://huggingface.co/blog/stellaathena/common-pile |
| LongBench | https://github.com/THUDM/LongBench |
| RULER | https://github.com/NVIDIA/RULER |
| HELMET | https://github.com/princeton-nlp/HELMET |
| ZeroSCROLLS | https://huggingface.co/papers/2305.14196 |
