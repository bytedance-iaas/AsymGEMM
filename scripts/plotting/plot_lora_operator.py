#!/usr/bin/env python3
"""Plot isolated LoRA operator profile CSV result directories."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
INT_FIELDS = {
    "tokens",
    "batch_size",
    "seq_len",
    "in_features",
    "out_features",
    "rank",
    "warmup",
    "iters",
    "peak_hbm_bytes",
}
FLOAT_FIELDS = {
    "scale",
    "dropout_p",
    "median_ms",
    "mean_ms",
    "min_ms",
    "max_ms",
    "peak_hbm_gib",
}
GENERATED_FILES = {
    "lora_operator_index.csv",
    "sweep_summary.csv",
    "timing_vs_shape.png",
    "peak_hbm_vs_shape.png",
    "timing_vs_tokens.png",
    "peak_hbm_vs_tokens.png",
    "combined_timing_vs_shape.png",
    "combined_peak_hbm_vs_shape.png",
    "combined_timing_vs_tokens.png",
    "combined_peak_hbm_vs_tokens.png",
}
DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "fp32": "float32",
    "float32": "float32",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--input-csv", action="append", type=Path, default=[], help="Specific result CSV to include.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: profiling/lora_ops_<precision>/plots.",
    )
    parser.add_argument(
        "--combined-output-dir",
        type=Path,
        default=None,
        help="Default: profiling/lora_ops_<precision>/combined unless --output-dir is set, then <output-dir>/combined.",
    )
    parser.add_argument("--operation", action="append", default=[])
    parser.add_argument("--backend", action="append", default=[])
    parser.add_argument("--pass", dest="passes", action="append", default=[], choices=["forward", "backward"])
    parser.add_argument("--batch-size", action="append", type=int, default=[])
    parser.add_argument("--seq-len", "--seq-lens", dest="seq_lens", action="append", nargs="+", type=int, default=[])
    parser.add_argument("--feature-dims", action="append", default=[], help="IN|OUT pairs. Repeat or pass comma-separated pairs.")
    parser.add_argument("--rank", action="append", type=int, default=[])
    parser.add_argument("--dtype", action="append", default=[])
    parser.add_argument("--precision", action="append", default=[])
    parser.add_argument("--dropout-p", action="append", type=float, default=[])
    parser.add_argument("--scale", action="append", type=float, default=[])
    parser.add_argument("--cuda-graph", action="append", default=[], choices=["true", "false"])
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove generated files only from the selected output subdirectories before writing them.",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="When there is one plot group, write its files directly into --output-dir.",
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Do not write combined plots/index.",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Only write combined plots/index.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def safe_label(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() or ch in ".=-" else "_" for ch in value).strip("_")


def precision_label(args: argparse.Namespace) -> str:
    values: list[str] = []
    for value in args.precision:
        values.extend(part for part in value.replace(",", " ").split() if part)
    return safe_label(values[0]) if values else "bf16"


def input_root(args: argparse.Namespace) -> Path:
    if args.input_root is not None:
        return resolve_path(args.input_root)
    return ROOT / "profiling" / f"lora_ops_{precision_label(args)}"


def output_root(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return resolve_path(args.output_dir)
    return ROOT / "profiling" / f"lora_ops_{precision_label(args)}" / "plots"


def combined_output_root(args: argparse.Namespace, root: Path) -> Path:
    if args.combined_output_dir is not None:
        return resolve_path(args.combined_output_dir)
    if args.output_dir is None:
        return ROOT / "profiling" / f"lora_ops_{precision_label(args)}" / "combined"
    return root / "combined"


def split_values(values: list[str]) -> list[str]:
    parts: list[str] = []
    for value in values:
        parts.extend(part for part in value.replace(",", " ").split() if part)
    return parts


def feature_dim_filter(values: list[str]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for value in split_values(values):
        if "|" not in value:
            raise SystemExit(f"--feature-dims entries must be IN|OUT pairs, got {value!r}")
        left, right = value.split("|", 1)
        pairs.add((int(left), int(right)))
    return pairs


def flatten_ints(values: list[list[int]]) -> list[int]:
    return [item for group in values for item in group]


def convert_value(field: str, value: str) -> Any:
    if field in INT_FIELDS:
        return int(value)
    if field in FLOAT_FIELDS:
        return float(value)
    if field == "cuda_graph":
        return value.lower()
    return value


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [{key: convert_value(key, value) for key, value in row.items()} for row in reader]


def result_csv_paths(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.rglob("result.csv") if path.is_file())


def read_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    paths = [resolve_path(path) for path in args.input_csv]
    if not paths:
        paths = result_csv_paths(input_root(args))
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_csv(path))
    return sorted((row for row in rows if passes_filters(args, row)), key=row_sort_key)


def write_csv(rows: list[dict[str, Any]], output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def clean_generated_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for name in GENERATED_FILES:
        path = output_dir / name
        if path.is_file():
            path.unlink()


def float_matches(value: float, allowed: list[float]) -> bool:
    return any(math.isclose(value, candidate, rel_tol=1e-9, abs_tol=1e-12) for candidate in allowed)


def canonical_dtype(value: str) -> str:
    return DTYPE_ALIASES.get(value.lower(), value.lower())


def passes_filters(args: argparse.Namespace, row: dict[str, Any]) -> bool:
    dims = feature_dim_filter(args.feature_dims)
    seq_lens = set(flatten_ints(args.seq_lens))
    cuda_graph = {value.lower() for value in args.cuda_graph}
    if args.operation and str(row["operation"]) not in set(args.operation):
        return False
    if args.backend and str(row["backend"]) not in set(args.backend):
        return False
    if args.passes and str(row["pass"]) not in set(args.passes):
        return False
    if args.batch_size and int(row["batch_size"]) not in set(args.batch_size):
        return False
    if seq_lens and int(row["seq_len"]) not in seq_lens:
        return False
    if dims and (int(row["in_features"]), int(row["out_features"])) not in dims:
        return False
    if args.rank and int(row["rank"]) not in set(args.rank):
        return False
    if args.dtype and canonical_dtype(str(row["dtype"])) not in {canonical_dtype(value) for value in args.dtype}:
        return False
    if args.precision and str(row["precision"]).lower() not in {value.lower() for value in args.precision}:
        return False
    if args.dropout_p and not float_matches(float(row["dropout_p"]), args.dropout_p):
        return False
    if args.scale and not float_matches(float(row["scale"]), args.scale):
        return False
    if cuda_graph and str(row["cuda_graph"]).lower() not in cuda_graph:
        return False
    return True


def row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["operation"]),
        int(row["batch_size"]),
        int(row["rank"]),
        str(row["dtype"]),
        str(row["precision"]),
        float(row["dropout_p"]),
        float(row["scale"]),
        str(row["cuda_graph"]),
        int(row["seq_len"]),
        int(row["in_features"]),
        int(row["out_features"]),
        str(row["backend"]),
        str(row["pass"]),
    )


def group_key(row: dict[str, Any]) -> tuple[str, int, int, str, str, float, float, str]:
    return (
        str(row["operation"]),
        int(row["batch_size"]),
        int(row["rank"]),
        str(row["dtype"]),
        str(row["precision"]),
        float(row["dropout_p"]),
        float(row["scale"]),
        str(row["cuda_graph"]),
    )


def group_label(key: tuple[str, int, int, str, str, float, float, str]) -> str:
    operation, batch_size, rank, dtype, precision, dropout_p, scale, cuda_graph = key
    graph = "cudagraph" if cuda_graph == "true" else "eager"
    return f"{operation}-b{batch_size}-r{rank}-{dtype}-{precision}-drop{dropout_p:g}-scale{scale:g}-{graph}"


def combined_label(rows: list[dict[str, Any]]) -> str:
    labels = sorted({group_label(group_key(row)) for row in rows})
    if len(labels) == 1:
        return labels[0]
    joined = "__".join(labels)
    if len(joined) <= 140:
        return joined
    return f"{labels[0]}__plus{len(labels) - 1}"


def shape_label(row: dict[str, Any]) -> str:
    return f"s{int(row['seq_len'])} {int(row['in_features'])}x{int(row['out_features'])}"


def shape_sort_key(label: str, rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    for row in rows:
        if shape_label(row) == label:
            return int(row["seq_len"]), int(row["in_features"]), int(row["out_features"])
    return (0, 0, 0)


def series_label(row: dict[str, Any], *, include_group: bool = False, include_shape: bool = False) -> str:
    parts: list[str] = []
    if include_group:
        parts.append(group_label(group_key(row)))
    parts.extend([str(row["backend"]), str(row["pass"])])
    if include_shape:
        parts.append(f"{int(row['in_features'])}x{int(row['out_features'])}")
    return " ".join(parts)


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_by_shape(
    rows: list[dict[str, Any]],
    output_dir: Path,
    filename: str,
    title: str,
    ylabel: str,
    value_key: str,
    *,
    include_group: bool = False,
) -> None:
    plt = pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = sorted({shape_label(row) for row in rows}, key=lambda label: shape_sort_key(label, rows))
    series = sorted({series_label(row, include_group=include_group) for row in rows})
    values: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        values.setdefault((series_label(row, include_group=include_group), shape_label(row)), []).append(float(row[value_key]))

    fig_width = max(8.0, min(18.0, 1.0 + len(categories) * 1.4))
    fig, ax = plt.subplots(figsize=(fig_width, 5), dpi=160)
    x_positions = list(range(len(categories)))
    width = min(0.8, 0.8 / max(1, len(series)))
    offset0 = -0.5 * width * (len(series) - 1)
    for index, label in enumerate(series):
        offsets = [x + offset0 + index * width for x in x_positions]
        y_values = [average(values.get((label, category), [])) for category in categories]
        ax.bar(offsets, y_values, width=width, label=label)

    ax.set_title(title)
    ax.set_xlabel("Operator shape")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories, rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def token_series_key(row: dict[str, Any], *, include_group: bool = False) -> str:
    return series_label(row, include_group=include_group, include_shape=True)


def plot_by_tokens(
    rows: list[dict[str, Any]],
    output_dir: Path,
    filename: str,
    title: str,
    ylabel: str,
    value_key: str,
    *,
    include_group: bool = False,
) -> None:
    plt = pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)

    series = sorted({token_series_key(row, include_group=include_group) for row in rows})
    values: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        values.setdefault((token_series_key(row, include_group=include_group), int(row["tokens"])), []).append(float(row[value_key]))

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    for label in series:
        points = sorted((tokens, average(items)) for (series_name, tokens), items in values.items() if series_name == label)
        if not points:
            continue
        x_values = [tokens for tokens, _ in points]
        y_values = [value for _, value in points]
        ax.plot(x_values, y_values, marker="o", linewidth=1.8, label=label)

    ax.set_title(title)
    ax.set_xlabel("Tokens")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def write_group_outputs(rows: list[dict[str, Any]], output_dir: Path, key: tuple[str, int, int, str, str, float, float, str], *, clean: bool) -> None:
    title = group_label(key)
    if clean:
        clean_generated_dir(output_dir)
    write_csv(rows, output_dir, "sweep_summary")
    plot_by_shape(rows, output_dir, "timing_vs_shape.png", f"{title}: median time by shape", "Median time (ms)", "median_ms")
    plot_by_shape(rows, output_dir, "peak_hbm_vs_shape.png", f"{title}: peak HBM by shape", "Peak HBM (GiB)", "peak_hbm_gib")
    plot_by_tokens(rows, output_dir, "timing_vs_tokens.png", f"{title}: median time by tokens", "Median time (ms)", "median_ms")
    plot_by_tokens(rows, output_dir, "peak_hbm_vs_tokens.png", f"{title}: peak HBM by tokens", "Peak HBM (GiB)", "peak_hbm_gib")


def write_combined_outputs(rows: list[dict[str, Any]], output_dir: Path, *, clean: bool) -> None:
    if clean:
        clean_generated_dir(output_dir)
    write_csv(rows, output_dir, "lora_operator_index")
    plot_by_shape(
        rows,
        output_dir,
        "combined_timing_vs_shape.png",
        "LoRA operator median time by shape",
        "Median time (ms)",
        "median_ms",
        include_group=True,
    )
    plot_by_shape(
        rows,
        output_dir,
        "combined_peak_hbm_vs_shape.png",
        "LoRA operator peak HBM by shape",
        "Peak HBM (GiB)",
        "peak_hbm_gib",
        include_group=True,
    )
    plot_by_tokens(
        rows,
        output_dir,
        "combined_timing_vs_tokens.png",
        "LoRA operator median time by tokens",
        "Median time (ms)",
        "median_ms",
        include_group=True,
    )
    plot_by_tokens(
        rows,
        output_dir,
        "combined_peak_hbm_vs_tokens.png",
        "LoRA operator peak HBM by tokens",
        "Peak HBM (GiB)",
        "peak_hbm_gib",
        include_group=True,
    )


def main() -> None:
    args = parse_args()
    rows = read_rows(args)
    if not rows:
        raise SystemExit(f"no matching result.csv files found under {input_root(args)}")

    groups: dict[tuple[str, int, int, str, str, float, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)

    root = output_root(args)
    combined_dir = combined_output_root(args, root)
    if not args.skip_combined:
        write_combined_outputs(rows, combined_dir, clean=args.clean_output)
        print(f"wrote {combined_dir}", flush=True)
    if args.combined_only:
        return

    sorted_groups = sorted(groups.items(), key=lambda item: group_label(item[0]))
    if args.flat_output and len(sorted_groups) == 1:
        key, group_rows = sorted_groups[0]
        write_group_outputs(group_rows, root, key, clean=args.clean_output)
        print(f"wrote {root}", flush=True)
        return

    for key, group_rows in sorted_groups:
        group_dir = root / safe_label(group_label(key))
        write_group_outputs(group_rows, group_dir, key, clean=args.clean_output)
        print(f"wrote {group_dir}", flush=True)


if __name__ == "__main__":
    main()
