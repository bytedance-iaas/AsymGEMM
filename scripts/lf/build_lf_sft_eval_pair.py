#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import random
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer, Seq2SeqTrainingArguments


@dataclass
class TokenStats:
    count: int
    min: int
    avg: float
    p25: float
    p50: float
    p75: float
    p90: float
    max: int
    cutoff_len: int
    min_token_filter: int
    at_or_above_min_filter: int
    above_cutoff_len: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate one LF-compatible train/eval SFT dataset pair."
    )
    parser.add_argument("--lf-dir", required=True)
    parser.add_argument("--asym-dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--results-root", default="")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--template", default="auto")
    parser.add_argument("--source-dataset", default="HuggingFaceTB/smoltalk")
    parser.add_argument("--source-config", default="longalign")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--train-name", default="lf_smoltalk_small_train")
    parser.add_argument("--eval-name", default="lf_smoltalk_small_eval")
    parser.add_argument("--train-rows", type=int, default=1000)
    parser.add_argument("--eval-rows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--min-tokens", type=int, default=2048)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lf-preprocess-samples", type=int, default=8)
    parser.add_argument("--skip-lf-preprocess-check", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing train/eval files. By default existing complete pairs are audited without rewriting.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Require existing train/eval files and only reprint validation/stats/results.",
    )
    parser.add_argument("--print-samples", type=int, default=3, help="Print compact examples from each split.")
    return parser.parse_args()


def _safe_label(value: str) -> str:
    cleaned = []
    for ch in value.lower():
        cleaned.append(ch if ch.isalnum() or ch in {"_", "-"} else "_")
    return "".join(cleaned).strip("_-")


def _model_tag(model_name_or_path: str) -> str:
    return _safe_label(Path(model_name_or_path).name)


def _dataset_family(train_name: str) -> str:
    return _safe_label(train_name.split("__", 1)[0])


def _infer_template(model_name_or_path: str) -> str:
    base = Path(model_name_or_path).name.lower()
    if base.startswith("gemma-4-") or base.startswith("gemma4-"):
        return "gemma4"
    if base.startswith("llama-4-"):
        return "llama4"
    if base.startswith("qwen3-") or base.startswith("qwen3-next-"):
        return "qwen3_nothink"
    return "qwen3_nothink"


def _percentile(sorted_values: list[int], pct: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute percentile of empty values")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _token_stats(lengths: list[int], cutoff_len: int, min_tokens: int) -> TokenStats:
    if not lengths:
        raise ValueError("cannot compute token stats for an empty dataset")
    sorted_lengths = sorted(lengths)
    return TokenStats(
        count=len(lengths),
        min=sorted_lengths[0],
        avg=mean(sorted_lengths),
        p25=_percentile(sorted_lengths, 25),
        p50=_percentile(sorted_lengths, 50),
        p75=_percentile(sorted_lengths, 75),
        p90=_percentile(sorted_lengths, 90),
        max=sorted_lengths[-1],
        cutoff_len=cutoff_len,
        min_token_filter=min_tokens,
        at_or_above_min_filter=sum(length >= min_tokens for length in lengths),
        above_cutoff_len=sum(length > cutoff_len for length in lengths),
    )


def _normalize_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    system = ""
    normalized: list[dict[str, str]] = []
    for message in row.get("messages") or []:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        content = content.strip()
        if role == "system":
            system = content
        elif role == "user":
            normalized.append({"from": "human", "value": content})
        elif role == "assistant":
            normalized.append({"from": "gpt", "value": content})

    while normalized and normalized[0]["from"] != "human":
        normalized.pop(0)
    while normalized and normalized[-1]["from"] != "gpt":
        normalized.pop()

    repaired: list[dict[str, str]] = []
    expected = "human"
    for message in normalized:
        if message["from"] != expected:
            if repaired and repaired[-1]["from"] == message["from"]:
                repaired[-1]["value"] += "\n\n" + message["value"]
            continue
        repaired.append(message)
        expected = "gpt" if expected == "human" else "human"

    while repaired and repaired[-1]["from"] != "gpt":
        repaired.pop()

    if len(repaired) < 2 or len(repaired) % 2 != 0:
        return [], system
    return repaired, system


def _record_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for message in record["conversations"]:
        parts.append(str(message["value"]))
    return "\n\n".join(parts)


def _record_text_any(record: dict[str, Any]) -> str:
    if isinstance(record.get("conversations"), list):
        return "\n\n".join(str(message.get("value", "")) for message in record["conversations"])
    return "\n\n".join(
        str(record.get(key, ""))
        for key in ("instruction", "input", "output")
        if str(record.get(key, "")).strip()
    )


def _stable_id(conversations: list[dict[str, str]]) -> str:
    payload = json.dumps(conversations, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _sample_split(
    dataset_name: str,
    config: str,
    split: str,
    count: int,
    seed: int,
    tokenizer: Any,
    min_tokens: int,
    cutoff_len: int,
    used_ids: set[str],
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    dataset = load_dataset(dataset_name, config, split=split)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    records: list[dict[str, Any]] = []
    lengths: list[int] = []
    skipped = {"short": 0, "duplicate": 0, "invalid": 0, "scanned": 0}
    for idx in indices:
        skipped["scanned"] += 1
        row = dataset[int(idx)]
        conversations, system = _normalize_messages(row)
        if not conversations:
            skipped["invalid"] += 1
            continue
        record_id = _stable_id(conversations)
        if record_id in used_ids:
            skipped["duplicate"] += 1
            continue
        record = {
            "id": record_id,
            "conversations": conversations,
            "system": system,
            "source": row.get("source", dataset_name),
            "source_config": config,
            "source_split": split,
            "source_index": int(idx),
            "source_tokens": row.get("tokens", row.get("conversation_tokens")),
        }
        token_length = len(tokenizer.encode(_record_text(record), add_special_tokens=False))
        if token_length < min_tokens:
            skipped["short"] += 1
            continue
        record["token_length"] = token_length
        record["truncated_by_cutoff"] = token_length > cutoff_len
        records.append(record)
        lengths.append(token_length)
        used_ids.add(record_id)
        if len(records) >= count:
            break

    if len(records) < count:
        raise RuntimeError(
            f"only collected {len(records)} records from {dataset_name}/{config}:{split}; "
            f"needed {count}; skipped={skipped}"
        )
    return records, lengths, skipped


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _update_dataset_info(lf_dir: Path, train_name: str, eval_name: str) -> None:
    info_path = lf_dir / "data" / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    entry = {
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "system": "system",
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "system_tag": "system",
        },
    }
    info[train_name] = {"file_name": f"{train_name}.jsonl", **entry}
    info[eval_name] = {"file_name": f"{eval_name}.jsonl", **entry}
    _write_json(info_path, info)


def _load_existing_jsonl(path: Path, tokenizer: Any) -> tuple[list[dict[str, Any]], list[int]]:
    records: list[dict[str, Any]] = []
    lengths: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(record)
            lengths.append(len(tokenizer.encode(_record_text_any(record), add_special_tokens=False)))
    return records, lengths


def _validate_sharegpt_record(record: dict[str, Any], line_no: int) -> list[str]:
    errors: list[str] = []
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2 or len(conversations) % 2 != 0:
        return [f"line {line_no}: invalid conversation length"]
    for idx, message in enumerate(conversations):
        expected = "human" if idx % 2 == 0 else "gpt"
        if message.get("from") != expected:
            errors.append(f"line {line_no}: expected role {expected} at turn {idx}")
        if not isinstance(message.get("value"), str) or not message["value"].strip():
            errors.append(f"line {line_no}: empty message at turn {idx}")
    return errors


def _validate_alpaca_record(record: dict[str, Any], line_no: int) -> list[str]:
    errors: list[str] = []
    prompt = record.get("instruction")
    response = record.get("output")
    if not isinstance(prompt, str) or not prompt.strip():
        errors.append(f"line {line_no}: empty instruction")
    if not isinstance(response, str) or not response.strip():
        errors.append(f"line {line_no}: empty output")
    return errors


def _validate_jsonl(path: Path) -> dict[str, Any]:
    count = 0
    ids: set[str] = set()
    errors: list[str] = []
    formats: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except Exception as exc:
                errors.append(f"line {line_no}: invalid json: {exc}")
                continue
            count += 1

            if "conversations" in record:
                formats.add("sharegpt")
                errors.extend(_validate_sharegpt_record(record, line_no))
                record_id = record.get("id") or _stable_id(
                    [
                        {"from": str(message.get("from", "")), "value": str(message.get("value", ""))}
                        for message in record.get("conversations", [])
                    ]
                )
            elif "instruction" in record or "output" in record:
                formats.add("alpaca")
                errors.extend(_validate_alpaca_record(record, line_no))
                record_id = record.get("id") or hashlib.sha1(
                    _record_text_any(record).encode("utf-8")
                ).hexdigest()
            else:
                errors.append(f"line {line_no}: unknown LF dataset format")
                record_id = record.get("id")

            if record_id in ids:
                errors.append(f"line {line_no}: duplicate id {record_id}")
            ids.add(record_id)

    if len(formats) > 1:
        errors.append(f"mixed dataset formats found: {sorted(formats)}")

    return {"path": str(path), "count": count, "formats": sorted(formats), "errors": errors, "ok": not errors}


def _validate_dataset_info(lf_dir: Path, train_name: str, eval_name: str) -> dict[str, Any]:
    info_path = lf_dir / "data" / "dataset_info.json"
    result: dict[str, Any] = {"path": str(info_path), "ok": True, "datasets": {}, "errors": []}
    if not info_path.exists():
        return {"path": str(info_path), "ok": False, "datasets": {}, "errors": ["dataset_info.json missing"]}

    info = json.loads(info_path.read_text(encoding="utf-8"))
    for name in (train_name, eval_name):
        entry = info.get(name)
        if not isinstance(entry, dict):
            result["ok"] = False
            result["errors"].append(f"{name}: missing registration")
            continue
        file_name = entry.get("file_name")
        file_path = lf_dir / "data" / str(file_name)
        dataset_result = {
            "file_name": file_name,
            "formatting": entry.get("formatting", "alpaca"),
            "columns": entry.get("columns", {}),
            "file_exists": file_path.exists(),
        }
        if not file_name or not file_path.exists():
            result["ok"] = False
            result["errors"].append(f"{name}: registered file missing: {file_name}")
        result["datasets"][name] = dataset_result

    return result


def _lf_preprocess_check(args: argparse.Namespace, lf_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(lf_dir / "src"))
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments, ModelArguments

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    data_args = DataArguments(
        template=args.template,
        dataset=args.train_name,
        eval_dataset=args.eval_name,
        dataset_dir=str(lf_dir / "data"),
        cutoff_len=args.cutoff_len,
        overwrite_cache=True,
        preprocessing_num_workers=1,
        max_samples=args.lf_preprocess_samples,
        enable_thinking=False,
    )
    model_args = ModelArguments(model_name_or_path=args.model_name_or_path, trust_remote_code=True)
    with tempfile.TemporaryDirectory(prefix="lf_dataset_check_") as tmpdir:
        training_args = Seq2SeqTrainingArguments(
            output_dir=tmpdir,
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            report_to=[],
            remove_unused_columns=False,
        )
        template = get_template_and_fix_tokenizer(tokenizer, data_args)
        log_buffer = io.StringIO()
        with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
            dataset_module = get_dataset(
                template,
                model_args,
                data_args,
                training_args,
                stage="sft",
                tokenizer=tokenizer,
            )

    train_dataset = dataset_module.get("train_dataset")
    eval_dataset = dataset_module.get("eval_dataset")
    result = {
        "ok": train_dataset is not None and eval_dataset is not None,
        "train_rows_after_preprocess": len(train_dataset) if train_dataset is not None else 0,
        "eval_rows_after_preprocess": len(eval_dataset) if eval_dataset is not None else 0,
        "log_tail": log_buffer.getvalue()[-4000:],
    }
    if result["train_rows_after_preprocess"] <= 0 or result["eval_rows_after_preprocess"] <= 0:
        result["ok"] = False
    return result


def _write_manifest_file(data_dir: Path, manifest: dict[str, Any]) -> None:
    manifest_path = data_dir / "lf_dataset_manifest.json"
    existing: dict[str, Any]
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        existing = {"datasets": {}}
    existing.setdefault("datasets", {})
    existing["datasets"][manifest["train_name"]] = manifest
    existing["datasets"][manifest["eval_name"]] = manifest
    _write_json(manifest_path, existing)


def _print_compact_samples(split: str, records: list[dict[str, Any]], limit: int) -> None:
    if limit <= 0:
        return
    print(f"{split}_samples:")
    for idx, record in enumerate(records[:limit]):
        text = _record_text_any(record).replace("\n", " ")
        if len(text) > 240:
            text = text[:237] + "..."
        print(
            json.dumps(
                {
                    "idx": idx,
                    "id": record.get("id", ""),
                    "token_length": record.get("token_length"),
                    "preview": text,
                },
                ensure_ascii=False,
            )
        )


def _write_results(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    token_stats: dict[str, Any],
    validation: dict[str, Any],
) -> Path:
    results_root = Path(args.results_root) if args.results_root else Path(args.asym_dir) / "results"
    model_tag = _model_tag(args.model_name_or_path)
    dataset_root = f"{_dataset_family(args.train_name)}__lora__lf__{args.precision}"
    config = (
        f"{model_tag}__{args.train_name}__b{args.batch_size}_s{args.cutoff_len}_"
        f"r{args.lora_rank}_a{args.lora_alpha}"
    )
    config_root = results_root / dataset_root / config
    dataset_dir = config_root / "dataset"
    combined_dir = results_root / dataset_root / "combined"

    _write_json(dataset_dir / "manifest.json", manifest)
    _write_json(dataset_dir / "token_stats.json", token_stats)
    _write_json(dataset_dir / "validation.json", validation)

    jobs_path = config_root / "jobs.tsv"
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_path.write_text(
        "status\tkind\ttrain_dataset\teval_dataset\tjob_dir\tmetrics\n"
        f"ok\tdataset\t{args.train_name}\t{args.eval_name}\t{dataset_dir}\t{dataset_dir / 'validation.json'}\n",
        encoding="utf-8",
    )

    dataset_index_row = {
        "status": "ok" if validation.get("ok") else "failed",
        "kind": "dataset",
        "config": config,
        "backend": "none",
        "framework": "dataset_validation",
        "task": args.train_name,
        "dataset": args.train_name,
        "eval_dataset": args.eval_name,
        "command": "",
        "raw_outputs": "",
        "metrics": str(dataset_dir / "validation.json"),
        "job_dir": str(dataset_dir),
    }
    combined_dir.mkdir(parents=True, exist_ok=True)
    index_path = combined_dir / "eval_index.json"
    if index_path.exists():
        index_rows = json.loads(index_path.read_text(encoding="utf-8"))
        index_rows = [
            row
            for row in index_rows
            if not (row.get("config") == config and row.get("kind") == "dataset")
        ]
    else:
        index_rows = []
    index_rows.append(dataset_index_row)
    with (combined_dir / "eval_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)
    _write_json(index_path, index_rows)

    long_rows: list[dict[str, Any]] = []
    long_path = combined_dir / "metrics_long.csv"
    if long_path.exists():
        with long_path.open(encoding="utf-8", newline="") as handle:
            long_rows = [
                row
                for row in csv.DictReader(handle)
                if not (row.get("config") == config and row.get("framework") == "dataset_validation")
            ]
    for split, stats in token_stats.items():
        if isinstance(stats, dict):
            for metric, value in stats.items():
                if isinstance(value, (int, float)):
                    long_rows.append(
                        {
                            "config": config,
                            "backend": "none",
                            "framework": "dataset_validation",
                            "task": split,
                            "metric": metric,
                            "value": value,
                        }
                    )
    with long_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config", "backend", "framework", "task", "metric", "value"])
        writer.writeheader()
        writer.writerows(long_rows)

    wide_row = {
        "config": config,
        "backend": "none",
        "framework": "dataset_validation",
        "task": "dataset",
        "train_dataset": args.train_name,
        "eval_dataset": args.eval_name,
        "train_count": token_stats["train"]["count"],
        "eval_count": token_stats["eval"]["count"],
        "train_p50": token_stats["train"]["p50"],
        "train_p75": token_stats["train"]["p75"],
        "eval_p50": token_stats["eval"]["p50"],
        "eval_p75": token_stats["eval"]["p75"],
        "train_loss": "",
        "train_runtime": "",
        "status": "ok" if validation.get("ok") else "failed",
    }
    wide_path = combined_dir / "metrics_wide.csv"
    wide_rows: list[dict[str, Any]] = []
    if wide_path.exists():
        with wide_path.open(encoding="utf-8", newline="") as handle:
            wide_rows = [
                row
                for row in csv.DictReader(handle)
                if not (row.get("config") == config and row.get("framework") == "dataset_validation")
            ]
    wide_rows.append(wide_row)
    wide_fields = [
        "config",
        "backend",
        "framework",
        "task",
        "train_dataset",
        "eval_dataset",
        "train_count",
        "eval_count",
        "train_p50",
        "train_p75",
        "eval_p50",
        "eval_p75",
        "train_loss",
        "train_runtime",
        "status",
    ]
    with wide_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=wide_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(wide_rows)

    summary = [
        "# LF SFT Dataset Validation Summary",
        "",
        f"- train dataset: `{args.train_name}`",
        f"- eval dataset: `{args.eval_name}`",
        f"- source: `{args.source_dataset}` / `{args.source_config}`",
        f"- train rows: `{token_stats['train']['count']}`",
        f"- eval rows: `{token_stats['eval']['count']}`",
        f"- train p50/p75 tokens: `{token_stats['train']['p50']:.2f}` / `{token_stats['train']['p75']:.2f}`",
        f"- eval p50/p75 tokens: `{token_stats['eval']['p50']:.2f}` / `{token_stats['eval']['p75']:.2f}`",
        f"- validation ok: `{validation.get('ok')}`",
        "",
    ]
    (combined_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    (combined_dir / "plots").mkdir(exist_ok=True)
    return config_root


def main() -> None:
    args = _parse_args()
    if args.template == "auto":
        args.template = _infer_template(args.model_name_or_path)
    print(f"Using template: {args.template} for model {args.model_name_or_path}")
    lf_dir = Path(args.lf_dir).resolve()
    data_dir = lf_dir / "data"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    used_ids: set[str] = set()
    train_path = data_dir / f"{args.train_name}.jsonl"
    eval_path = data_dir / f"{args.eval_name}.jsonl"

    both_files_exist = train_path.exists() and eval_path.exists()
    one_file_exists = train_path.exists() or eval_path.exists()
    use_existing = args.audit_only or (both_files_exist and not args.overwrite)
    if args.audit_only and (not train_path.exists() or not eval_path.exists()):
        raise FileNotFoundError(f"--audit-only requires {train_path} and {eval_path}")
    if one_file_exists and not both_files_exist and not args.overwrite:
        raise FileExistsError(
            "found only one existing split and --overwrite was not set; refusing partial rewrite. "
            f"train_exists={train_path.exists()} eval_exists={eval_path.exists()}"
        )

    if use_existing:
        train_records, train_lengths = _load_existing_jsonl(train_path, tokenizer)
        eval_records, eval_lengths = _load_existing_jsonl(eval_path, tokenizer)
        train_skips = {"mode": "existing", "not_resampled": True, "scanned": len(train_records)}
        eval_skips = {"mode": "existing", "not_resampled": True, "scanned": len(eval_records)}
    else:
        train_records, train_lengths, train_skips = _sample_split(
            args.source_dataset,
            args.source_config,
            args.train_split,
            args.train_rows,
            args.seed,
            tokenizer,
            args.min_tokens,
            args.cutoff_len,
            used_ids,
        )
        eval_records, eval_lengths, eval_skips = _sample_split(
            args.source_dataset,
            args.source_config,
            args.eval_split,
            args.eval_rows,
            args.seed + 1,
            tokenizer,
            args.min_tokens,
            args.cutoff_len,
            used_ids,
        )

        _write_jsonl(train_path, train_records)
        _write_jsonl(eval_path, eval_records)
        _update_dataset_info(lf_dir, args.train_name, args.eval_name)

    token_stats = {
        "train": asdict(_token_stats(train_lengths, args.cutoff_len, args.min_tokens)),
        "eval": asdict(_token_stats(eval_lengths, args.cutoff_len, args.min_tokens)),
    }
    length_errors = []
    for split, stats in token_stats.items():
        if stats["at_or_above_min_filter"] < stats["count"]:
            length_errors.append(
                f"{split}: {stats['at_or_above_min_filter']}/{stats['count']} rows have "
                f">= {args.min_tokens} tokens for model {args.model_name_or_path}"
            )
    train_validation = _validate_jsonl(train_path)
    eval_validation = _validate_jsonl(eval_path)
    dataset_info_validation = _validate_dataset_info(lf_dir, args.train_name, args.eval_name)
    validation: dict[str, Any] = {
        "ok": train_validation["ok"]
        and eval_validation["ok"]
        and dataset_info_validation["ok"]
        and not length_errors,
        "train": train_validation,
        "eval": eval_validation,
        "dataset_info": dataset_info_validation,
        "length": {"ok": not length_errors, "errors": length_errors},
        "lf_preprocess": {"ok": None, "skipped": True},
    }

    if not args.skip_lf_preprocess_check:
        try:
            validation["lf_preprocess"] = _lf_preprocess_check(args, lf_dir)
        except Exception as exc:
            validation["lf_preprocess"] = {"ok": False, "error": repr(exc)}
        validation["ok"] = validation["ok"] and bool(validation["lf_preprocess"].get("ok"))

    manifest = {
        "train_name": args.train_name,
        "eval_name": args.eval_name,
        "source_dataset": args.source_dataset,
        "source_config": args.source_config,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "seed": args.seed,
        "train_rows": len(train_records),
        "eval_rows": len(eval_records),
        "train_file": str(train_path),
        "eval_file": str(eval_path),
        "formatting": train_validation.get("formats", ["unknown"])[0]
        if train_validation.get("formats")
        else "unknown",
        "model_name_or_path": args.model_name_or_path,
        "template": args.template,
        "cutoff_len": args.cutoff_len,
        "min_tokens": args.min_tokens,
        "audit_only": args.audit_only,
        "overwrite": args.overwrite,
        "train_skips": train_skips,
        "eval_skips": eval_skips,
        "token_stats": token_stats,
        "validation_ok": validation["ok"],
    }
    _write_manifest_file(data_dir, manifest)
    config_root = _write_results(args, manifest, token_stats, validation)

    print(f"train_dataset={args.train_name}")
    print(f"eval_dataset={args.eval_name}")
    print(f"train_file={train_path}")
    print(f"eval_file={eval_path}")
    print(f"results_config_dir={config_root}")
    print(f"validation_ok={validation['ok']}")
    print(f"train_p50={token_stats['train']['p50']:.2f}")
    print(f"train_p75={token_stats['train']['p75']:.2f}")
    print(f"eval_p50={token_stats['eval']['p50']:.2f}")
    print(f"eval_p75={token_stats['eval']['p75']:.2f}")
    _print_compact_samples("train", train_records, args.print_samples)
    _print_compact_samples("eval", eval_records, args.print_samples)
    if not validation["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
