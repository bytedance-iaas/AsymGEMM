#!/usr/bin/env python3
"""Fold the always-on -ohbm<N> tag into LF profile recompute labels (all backends) and widen -ceil.

The profile scripts (profile_lora_lf_test_both.sh / _source.sh) now emit the outer-HBM-every-N axis
inline in the recompute token and ALWAYS carry it on the Unsloth-GC recompute modes (unsloth,
unsloth-off, recomp-off-*), for EVERY backend (``ohbm0`` = all outer Unsloth checkpoint roots
offloaded to CPU)::

    asym:      recomp-off-full-fg-ker101-ceil000  ->  recomp-off-full-fg-ker101-ceil0000-ohbm0
    non-asym:  unsloth-off                         ->  unsloth-off-ohbm0

asym_* runs carry the full -ker<XYZ>-ceil<NNNN>-ohbm<N> triple (ceil zero-padded to 4 digits);
non-asym GC runs carry -ohbm<N> only (no -ceil — that is asym-NVMe-only — and keep any -ker). Non-GC
modes (norecomp/recomp/torch/kt) carry no -ohbm and are untouched.

Previously the ceiling used 3 digits and -ohbm<N> was either omitted (N=0) or a separate
``__ohbm<N>`` path component (N>0). This migration renames legacy job directories (any backend)
under the given profile roots to the standardized form and heals stale references in ``jobs.tsv`` /
``ARTIFACTS.md`` — including references left behind by an earlier partial pass, and any legacy
trailing ``__ohbm<N>`` component (folded back into the token in place).

Run-level dirs (``b<batch>_s<seq>[_ga<n>]``) never carry the tags and are not renamed. Names that
exceed NAME_MAX after tagging get the same deterministic truncate+hash the shell ``safe_label``
applies, so the dir rename and the reference-file heal produce identical names for the same input.
The plotters discover runs by globbing summary JSON under the root (not by matching a
script-computed dir name), so a re-truncated already-hashed dir still plots correctly; only the
recompute token at the front of the name is authoritative for grouping.

CAVEAT (pre-truncated dirs): a legacy dir whose PRE-migration name already exceeded NAME_MAX
(i.e. was itself truncate+hashed, ending ``_h<10hex>`` at 255 chars) cannot be transformed to the
exact hash the profile script computes from the full untruncated label — the tail lost to the
prior truncation is unrecoverable from the on-disk name. Such a dir still gets a valid,
self-consistent name here (dir rename and reference heal agree), but its hash will differ from a
fresh script run, so ``COLLECT_EXISTING`` will not forward-match its per-run artifacts until it is
reconciled (renamed to the script-computed name). In this repo that affected only the qwen3.5
``attnfa4`` configs, which were reconciled separately after this migration.
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
        description="Fold the always-on -ohbm<N> tag and widen -ceil to 4 digits in legacy asym_* LF profile job directories."
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


GC_MODE_RE = re.compile(r"^(?:unsloth|unsloth-off|recomp-off)")


def normalized_recompute_token(token: str, is_asym: bool) -> str:
    """Standardize a recompute token so the -ohbm<N> tag is always present on Unsloth-GC modes.

    asym (full triple):  recomp-off-full-fg-ker101-ceil000 -> recomp-off-full-fg-ker101-ceil0000-ohbm0
                         (widen ceil to 4 digits; always append -ohbm, default 0).
    non-asym GC modes:   unsloth-off -> unsloth-off-ohbm0 ; unsloth -> unsloth-ohbm0
                         (append -ohbm only; NO -ceil, keep any -ker). Non-GC (norecomp/recomp): unchanged.
    A legacy trailing -ohbm<N> is detached and re-folded in the canonical position. Idempotent.
    """
    # Detach any already-present trailing -ohbm<N> so re-runs (and legacy folded tokens) are stable.
    ohbm: int | None = None
    match = OHBM_SUFFIX_RE.search(token)
    if match:
        ohbm = int(match.group(0)[len("-ohbm") :])
        token = token[: match.start()]
    if not is_asym:
        # Non-asym: only Unsloth-GC modes carry -ohbm; no -ceil widening; keep -ker as-is.
        if GC_MODE_RE.match(token):
            return token + f"-ohbm{ohbm if ohbm is not None else 0}"
        return token + (f"-ohbm{ohbm}" if ohbm is not None else "")
    # asym: ensure -ker<XYZ>, zero-pad -ceil to 4 digits, always emit -ohbm.
    if not KER_RE.search(token):
        ceil = CEIL_RE.search(token)
        if ceil:
            token = token[: ceil.start()] + "-ker000" + token[ceil.start() :]
        else:
            token = token + "-ker000"
    ceil = CEIL_RE.search(token)
    if ceil is None:
        ker = KER_RE.search(token)
        assert ker is not None
        token = token[: ker.end()] + "-ceil0000" + token[ker.end() :]
    else:
        n = int(ceil.group(0)[len("-ceil") :])
        token = token[: ceil.start()] + f"-ceil{n:04d}" + token[ceil.end() :]
    return token + f"-ohbm{ohbm if ohbm is not None else 0}"


def split_config_name(name: str) -> list[str] | None:
    """Return the __-parts when `name` is a job (config) dir name (any backend), else None."""
    parts = name.split("__")
    if len(parts) < 4:
        return None
    if not parts[3].startswith("pol"):
        return None
    return parts


def migrated_name(name: str) -> str | None:
    parts = split_config_name(name)
    if parts is None:
        return None
    new_token = normalized_recompute_token(parts[2], is_asym=parts[0].startswith("asym"))
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
# Matches a job(config) dir name by its backend prefix; migrated_name() no-ops on any non-config
# match, so a broad prefix set is safe.
CONFIG_NAME_IN_TEXT_RE = re.compile(r"\b(?:asym|superoffload|zero2|zero3|torch|kt)[a-z0-9_-]+")


def heal_text(text: str) -> str:
    """Forward-transform every legacy asym config dir name embedded in `text`.

    Renaming from file content (rather than a rename map from dirs on disk) also heals references
    whose dir was renamed in an earlier pass or no longer exists, and reproduces the NAME_MAX
    truncate+hash the shell safe_label applies (the forward transform agrees with the on-disk
    rename by construction, since both apply this same transform + safe_label to the same string).
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
        report_path = root / "ohbm_migration.json"
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
