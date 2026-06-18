#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


RUN_DIR_RE = re.compile(r"^b[0-9]+_s[0-9]+_ga[0-9]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert the explicit ligerloss0 axis into legacy LF profile job directories."
    )
    parser.add_argument("roots", type=Path, nargs="*", help="Profile roots to scan.")
    parser.add_argument("--root", dest="root_options", type=Path, action="append", default=[], help="Profile root to scan. May be repeated.")
    parser.add_argument("--apply", action="store_true", help="Actually rename directories and update JSON files.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files. This is the default.")
    args = parser.parse_args()
    if args.apply and args.dry_run:
        parser.error("--apply and --dry-run are mutually exclusive")
    return args


def has_run_child(path: Path) -> bool:
    try:
        return any(child.is_dir() and RUN_DIR_RE.match(child.name) for child in path.iterdir())
    except OSError:
        return False


def migrated_name(name: str) -> str | None:
    parts = name.split("__")
    if len(parts) < 4:
        return None
    if any(part in {"ligerloss0", "ligerloss1"} for part in parts):
        return None
    if not parts[3].startswith("pol"):
        return None

    insert_at = len(parts)
    for index, part in enumerate(parts):
        if part.startswith("gradoff") or part.startswith("weightoff"):
            insert_at = index
            break
    parts.insert(insert_at, "ligerloss0")
    return "__".join(parts)


def set_liger_loss_config(payload: dict[str, Any]) -> bool:
    changed = False
    config = payload.get("config")
    if isinstance(config, dict) and config.get("liger_loss") != "ligerloss0":
        config["liger_loss"] = "ligerloss0"
        changed = True
    source_profile = payload.get("source_profile")
    if isinstance(source_profile, dict):
        source_config = source_profile.get("config")
        if isinstance(source_config, dict) and source_config.get("liger_loss") != "ligerloss0":
            source_config["liger_loss"] = "ligerloss0"
            changed = True
    return changed


def update_metadata_jsons(job_dir: Path) -> list[Path]:
    changed_paths: list[Path] = []
    for json_path in job_dir.rglob("*.json"):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if set_liger_loss_config(payload):
            json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            changed_paths.append(json_path)
    return changed_paths


def update_jobs_tsv(root: Path, rename_map: dict[str, str], apply: bool) -> list[Path]:
    changed_paths: list[Path] = []
    for jobs_path in sorted(root.rglob("jobs.tsv")):
        try:
            original = jobs_path.read_text(encoding="utf-8")
        except OSError:
            continue
        updated = original
        for old, new in rename_map.items():
            updated = updated.replace(old, new)
        lines = updated.splitlines()
        if lines:
            header = lines[0].split("\t")
            if "liger_loss" not in header and "profiler" in header:
                insert_at = header.index("profiler") + 1
                header.insert(insert_at, "liger_loss")
                new_lines = ["\t".join(header)]
                for line in lines[1:]:
                    if not line:
                        new_lines.append(line)
                        continue
                    fields = line.split("\t")
                    if len(fields) == len(header) - 1:
                        fields.insert(insert_at, "ligerloss0")
                    new_lines.append("\t".join(fields))
                updated = "\n".join(new_lines) + ("\n" if original.endswith("\n") else "")
        if updated != original:
            changed_paths.append(jobs_path)
            print(f"{'update' if apply else 'would update'} {jobs_path}")
            if apply:
                jobs_path.write_text(updated, encoding="utf-8")
    return changed_paths


def migrate_root(root: Path, apply: bool) -> dict[str, Any]:
    renames: list[dict[str, str]] = []
    updated_jsons: list[str] = []
    rename_map: dict[str, str] = {}
    for job_dir in sorted(path for path in root.rglob("*") if path.is_dir() and has_run_child(path)):
        new_name = migrated_name(job_dir.name)
        if new_name is None:
            continue
        target = job_dir.with_name(new_name)
        if target.exists():
            raise SystemExit(f"target already exists: {target}")
        renames.append({"from": str(job_dir), "to": str(target)})
        rename_map[str(job_dir)] = str(target)
        print(f"{'rename' if apply else 'would rename'} {job_dir} -> {target}")
        if apply:
            job_dir.rename(target)
            for json_path in update_metadata_jsons(target):
                updated_jsons.append(str(json_path))
                print(f"updated {json_path}")
    updated_jobs = [str(path) for path in update_jobs_tsv(root, rename_map, apply)] if rename_map else []
    report = {
        "root": str(root),
        "applied": apply,
        "renames": renames,
        "updated_jsons": updated_jsons,
        "updated_jobs_tsv": updated_jobs,
    }
    if apply and (renames or updated_jsons or updated_jobs):
        report_path = root / "liger_loss_migration.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {report_path}")
    return report


def main() -> None:
    args = parse_args()
    roots = [*args.root_options, *args.roots]
    if not roots:
        raise SystemExit("at least one profile root is required")
    reports: list[dict[str, Any]] = []
    total = 0
    for root in roots:
        if not root.exists():
            print(f"skip missing root {root}")
            continue
        report = migrate_root(root, args.apply)
        reports.append(report)
        total += len(report["renames"])
    action = "migrated" if args.apply else "would migrate"
    print(f"{action} {total} job director{'y' if total == 1 else 'ies'}")


if __name__ == "__main__":
    main()
