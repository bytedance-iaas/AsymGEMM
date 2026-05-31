#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a long alpaca-format SFT smoke dataset for LLaMA-Factory.")
    parser.add_argument("--lf-dir", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--dataset-name", default="asym_long_sft_smoke")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--response-tokens", type=int, default=3072)
    parser.add_argument("--source", choices=["auto", "hf", "local"], default="auto")
    parser.add_argument("--hf-dataset", default="allenai/dolma")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_hf_texts(dataset_name: str, limit: int) -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split="train", streaming=True)
    texts: list[str] = []
    for row in dataset:
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
        if len(texts) >= limit:
            break
    return texts


def _load_local_text(lf_dir: Path) -> str:
    candidates = [
        lf_dir / "data" / "wiki_demo.txt",
        lf_dir / "data" / "c4_demo.jsonl",
        lf_dir / "data" / "alpaca_en_demo.json",
    ]
    for path in candidates:
        if path.exists():
            if path.suffix == ".json":
                records = json.loads(path.read_text(encoding="utf-8"))
                parts = []
                for record in records:
                    parts.extend(str(record.get(key, "")) for key in ("instruction", "input", "output"))
                return "\n\n".join(part for part in parts if part.strip())
            if path.suffix == ".jsonl":
                parts = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    parts.append(str(record.get("text", record)))
                return "\n\n".join(parts)
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"no fallback text source found under {lf_dir / 'data'}")


def _token_corpus(tokenizer, texts: list[str]) -> list[int]:
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
        ids.append(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
    return ids


def _repeat_to_length(ids: list[int], length: int) -> list[int]:
    if not ids:
        raise ValueError("cannot build dataset from empty token corpus")
    repeated = list(ids)
    while len(repeated) < length:
        repeated.extend(ids)
    return repeated


def _write_dataset_info(lf_dir: Path, dataset_name: str) -> None:
    info_path = lf_dir / "data" / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    info[dataset_name] = {
        "file_name": f"{dataset_name}.jsonl",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    lf_dir = Path(args.lf_dir).resolve()
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    texts: list[str] = []
    if args.source in {"auto", "hf"}:
        try:
            texts = _load_hf_texts(args.hf_dataset, limit=max(args.num_samples, 16))
            print(f"loaded {len(texts)} text records from {args.hf_dataset}")
        except Exception as exc:
            if args.source == "hf":
                raise
            print(f"falling back to local text because HF dataset load failed: {exc}")

    if not texts:
        texts = [_load_local_text(lf_dir)]

    total_tokens_per_sample = args.prompt_tokens + args.response_tokens
    if total_tokens_per_sample > args.cutoff_len:
        raise ValueError("prompt_tokens + response_tokens must be <= cutoff_len")
    corpus = _token_corpus(tokenizer, texts)
    need_tokens = args.num_samples * total_tokens_per_sample + total_tokens_per_sample
    corpus = _repeat_to_length(corpus, need_tokens)

    max_start = max(0, len(corpus) - total_tokens_per_sample - 1)
    starts = [0]
    if args.num_samples > 1:
        starts.extend(rng.randrange(0, max_start + 1) for _ in range(args.num_samples - 1))

    out_path = lf_dir / "data" / f"{args.dataset_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lengths: list[int] = []
    with out_path.open("w", encoding="utf-8") as f:
        for start in starts:
            prompt_ids = corpus[start : start + args.prompt_tokens]
            response_ids = corpus[start + args.prompt_tokens : start + total_tokens_per_sample]
            record = {
                "instruction": "Continue the following document.",
                "input": tokenizer.decode(prompt_ids, skip_special_tokens=True),
                "output": tokenizer.decode(response_ids, skip_special_tokens=True),
            }
            lengths.append(len(tokenizer.encode(record["input"] + record["output"], add_special_tokens=False)))
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    _write_dataset_info(lf_dir, args.dataset_name)
    print(f"wrote {out_path}")
    print(f"token lengths: min={min(lengths)} mean={mean(lengths):.1f} max={max(lengths)}")


if __name__ == "__main__":
    main()
