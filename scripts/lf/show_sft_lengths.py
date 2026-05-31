#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show prompt+response token length stats for a LLaMA-Factory SFT dataset.")
    parser.add_argument("--lf-dir", default="/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory")
    parser.add_argument("--dataset", default="asym_long_sft_smoke")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit rows for a quick check. 0 means all rows.")
    return parser.parse_args()


def _read_json_or_jsonl(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list or JSONL records.")
    return data


def _percentile(sorted_values: list[int], pct: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute percentile of an empty dataset")
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    pos = (len(sorted_values) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _text_from_record(record: dict, columns: dict[str, str]) -> str:
    fields = []
    for logical_name in ("prompt", "query", "response"):
        column_name = columns.get(logical_name)
        if column_name is None:
            continue
        value = record.get(column_name, "")
        if isinstance(value, list):
            value = "\n".join(str(item) for item in value)
        elif not isinstance(value, str):
            value = str(value)
        if value:
            fields.append(value)
    return "\n".join(fields)


def main() -> None:
    args = _parse_args()
    lf_dir = Path(args.lf_dir).resolve()
    dataset_info_path = lf_dir / "data" / "dataset_info.json"
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    if args.dataset not in dataset_info:
        raise KeyError(f"dataset {args.dataset!r} not found in {dataset_info_path}")

    config = dataset_info[args.dataset]
    file_name = config.get("file_name")
    if not file_name:
        raise ValueError(f"dataset {args.dataset!r} does not define a local file_name")

    dataset_path = lf_dir / "data" / file_name
    rows = _read_json_or_jsonl(dataset_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError(f"dataset {dataset_path} has no records")

    columns = config.get("columns", {})
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    lengths = sorted(
        len(tokenizer.encode(_text_from_record(record, columns), add_special_tokens=False))
        for record in rows
    )

    print(f"dataset={args.dataset}")
    print(f"file={dataset_path}")
    print(f"model_name_or_path={args.model_name_or_path}")
    print("length_definition=tokenizer(prompt + query + response, add_special_tokens=False)")
    print(f"count={len(lengths)}")
    print(f"min={lengths[0]}")
    print(f"avg={mean(lengths):.2f}")
    print(f"p25={_percentile(lengths, 25):.2f}")
    print(f"p50={_percentile(lengths, 50):.2f}")
    print(f"p75={_percentile(lengths, 75):.2f}")
    print(f"p90={_percentile(lengths, 90):.2f}")
    print(f"max={lengths[-1]}")


if __name__ == "__main__":
    main()
