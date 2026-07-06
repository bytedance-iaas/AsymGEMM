#!/usr/bin/env python3
"""Normalize asym_* LF profile job dirs to ALWAYS carry -ker<XYZ>-ceil<NNN> recompute tags.

The profile scripts (profile_lora_lf_test_both.sh / _source.sh) now label every asym_* run's
recompute token with an explicit routed-kernel code and CPU-activation-budget ceiling:
``ker000`` = no routed-kernel override, ``ceil000`` = no explicit per-run budget. This migration
renames legacy job directories (e.g. ``asym_cpuadamwds__source__recomp__polnone__...`` ->
``asym_cpuadamwds__source__recomp-ker000-ceil000__polnone__...``) and heals stale references in
``jobs.tsv`` / ``ARTIFACTS.md`` — including references left behind by an earlier partial rename
pass (reverse candidates are derived from every already-tagged dir on disk).

Non-asym backends are untouched. Run-level dirs (``b<batch>_s<seq>[_ga<n>]``) never carry the
tags and are not renamed. Names that exceed NAME_MAX after tagging get the same deterministic
truncate+hash the shell ``safe_label`` applies, so the renamer and the profile scripts always
agree on the final dir name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

RUN_DIR_RE = re.compile(r"^b[0-9]+_s[0-9]+(?:_ga[0-9]+)?$")
KER_RE = re.compile(r"-ker[0-9]{3}")
CEIL_RE = re.compile(r"-ceil[0-9]+")
OHBM_SUFFIX_RE = re.compile(r"-ohbm[0-9]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert the explicit -ker<XYZ>-ceil<NNN> recompute tags into legacy asym_* LF profile job directories."
    )
    parser.add_argument("roots", type=Path, nargs="*", help="Profile roots to scan.")
    parser.add_argument("--root", dest="root_options", type=Path, action="append", default=[], help="Profile root to scan. May be repeated.")
    parser.add_argument("--apply", action="store_true", help="Actually rename directories and update reference files.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying anything. This is the default.")
    args = parser.parse_args()
    if args.apply and args.dry_run:
        parser.error("--apply and --dry-run are mutually exclusive")
    return args


def safe_label(label: str) -> str:
    """Mirror the shell safe_label: lowercase, [^a-z0-9_-] -> _, strip edge [_-], NAME_MAX guard."""
    s = label.lower()
    s = re.sub(r"[^a-z0-9_-]", "_", s)
    s = re.sub(r"^[_-]*", "", s)
    s = re.sub(r"[_-]*$", "", s)
    if len(s) > 255:
        return s[:243] + "_h" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    return s


def normalized_recompute_token(token: str) -> str:
    """recomp -> recomp-ker000-ceil000; recomp-off-full-fg-ker101 -> ...-ker101-ceil000; tagged stays."""
    # Dir names keep -ohbm<N> as a separate __ohbm<N> component, but be defensive: tags go before it.
    ohbm = ""
    match = OHBM_SUFFIX_RE.search(token)
    if match:
        ohbm = match.group(0)
        token = token[: match.start()]
    if not KER_RE.search(token):
        ceil = CEIL_RE.search(token)
        if ceil:
            token = token[: ceil.start()] + "-ker000" + token[ceil.start() :]
        else:
            token = token + "-ker000"
    if not CEIL_RE.search(token):
        ker = KER_RE.search(token)
        assert ker is not None
        token = token[: ker.end()] + "-ceil000" + token[ker.end() :]
    return token + ohbm


def split_config_name(name: str) -> list[str] | None:
    """Return the __-parts when `name` is an asym_* job (config) dir name, else None."""
    parts = name.split("__")
    if len(parts) < 4:
        return None
    if not parts[0].startswith("asym"):
        return None
    if not parts[3].startswith("pol"):
        return None
    return parts


def migrated_name(name: str) -> str | None:
    parts = split_config_name(name)
    if parts is None:
        return None
    new_token = normalized_recompute_token(parts[2])
    if new_token == parts[2]:
        return None
    parts = [*parts[:2], new_token, *parts[3:]]
    return safe_label("__".join(parts))


def has_run_child(path: Path) -> bool:
    try:
        return any(child.is_dir() and RUN_DIR_RE.match(child.name) for child in path.iterdir())
    except OSError:
        return False


# Post-safe_label dir names only contain [a-z0-9_-]; path/field boundaries stop the match.
CONFIG_NAME_IN_TEXT_RE = re.compile(r"\basym[a-z0-9_-]+")


def heal_text(text: str) -> str:
    """Forward-transform every legacy (untagged) asym config dir name embedded in `text`.

    Renaming from file content (rather than a rename map from dirs on disk) also heals
    references whose dir was renamed in an earlier pass or no longer exists, and reproduces the
    NAME_MAX truncate+hash the shell safe_label applies (a hashed on-disk name cannot be mapped
    back to its legacy spelling, but the forward transform agrees with it by construction).
    """

    def replace(match: re.Match[str]) -> str:
        return migrated_name(match.group(0)) or match.group(0)

    return CONFIG_NAME_IN_TEXT_RE.sub(replace, text)


def update_reference_files(root: Path, apply: bool) -> list[Path]:
    """Rewrite legacy dir names in jobs.tsv / ARTIFACTS.md under `root`."""
    changed_paths: list[Path] = []
    for pattern in ("jobs.tsv", "ARTIFACTS.md"):
        for ref_path in sorted(root.rglob(pattern)):
            try:
                original = ref_path.read_text(encoding="utf-8")
            except OSError:
                continue
            updated = heal_text(original)
            if updated != original:
                changed_paths.append(ref_path)
                print(f"{'update' if apply else 'would update'} {ref_path}")
                if apply:
                    ref_path.write_text(updated, encoding="utf-8")
    return changed_paths


def migrate_root(root: Path, apply: bool) -> dict[str, Any]:
    renames: list[dict[str, str]] = []
    for job_dir in sorted(path for path in root.rglob("*") if path.is_dir() and has_run_child(path)):
        if split_config_name(job_dir.name) is None:
            continue
        new_name = migrated_name(job_dir.name)
        if new_name is None:
            continue
        target = job_dir.with_name(new_name)
        if target.exists():
            raise SystemExit(f"target already exists: {target}")
        renames.append({"from": str(job_dir), "to": str(target)})
        print(f"{'rename' if apply else 'would rename'} {job_dir} -> {target}")
        if apply:
            job_dir.rename(target)
    updated_refs = [str(path) for path in update_reference_files(root, apply)]
    report = {
        "root": str(root),
        "applied": apply,
        "renames": renames,
        "updated_references": updated_refs,
    }
    if apply and (renames or updated_refs):
        report_path = root / "ker_ceil_migration.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {report_path}")
    return report


def main() -> None:
    args = parse_args()
    roots = [*args.root_options, *args.roots]
    if not roots:
        raise SystemExit("at least one profile root is required")
    total_renames = 0
    total_refs = 0
    for root in roots:
        if not root.exists():
            print(f"skip missing root {root}")
            continue
        report = migrate_root(root, args.apply)
        total_renames += len(report["renames"])
        total_refs += len(report["updated_references"])
    action = "migrated" if args.apply else "would migrate"
    print(f"{action} {total_renames} job director{'y' if total_renames == 1 else 'ies'}, {total_refs} reference file{'' if total_refs == 1 else 's'}")


if __name__ == "__main__":
    main()
