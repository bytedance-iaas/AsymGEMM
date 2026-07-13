#!/usr/bin/env python3
"""Guard the Transformers FlashAttention-4 wrapper against a None ``s_aux``.

Upstream ``transformers==5.6.0`` ships ``flash_attention_forward`` with:

    s_aux=s_aux.to(query.dtype),  # FA only accepts half precision

which raises ``AttributeError: 'NoneType' object has no attribute 'to'`` for the
common case where the model has no learnable attention sink (``s_aux is None``),
crashing every ``flash_attention_4`` dispatch. This idempotently rewrites it to:

    s_aux=s_aux.to(query.dtype) if s_aux is not None else None,

Historically this lived in the isolated ``LlamaFactory-fa4`` lab checkout under
``fa4_probes/``. We now use a single LlamaFactory checkout, so the patch lives
here in AsymGEMM and is applied by ``bootstrap_lf_venv_fa4.sh`` after installing
transformers into ``.venv-fa4``.

Usage:
    patch_transformers_fa4.py           # apply the guard (idempotent)
    patch_transformers_fa4.py --check   # verify only; exit 1 if the guard is absent
"""

from __future__ import annotations

import importlib.util
import sys

BROKEN = "s_aux=s_aux.to(query.dtype),"
FIXED = "s_aux=s_aux.to(query.dtype) if s_aux is not None else None,"


def _target_path() -> str:
    spec = importlib.util.find_spec("transformers.integrations.flash_attention")
    if spec is None or not spec.origin:
        print(
            "[fa4] transformers.integrations.flash_attention not found; "
            "is transformers installed in this env?",
            file=sys.stderr,
        )
        sys.exit(3)
    return spec.origin


def main() -> int:
    check_only = "--check" in sys.argv[1:]
    path = _target_path()

    with open(path, encoding="utf-8") as fh:
        src = fh.read()

    if FIXED in src:
        print(f"[fa4] s_aux guard already present: {path}")
        return 0

    if check_only:
        print(f"[fa4] MISSING s_aux guard in {path}", file=sys.stderr)
        return 1

    if BROKEN not in src:
        print(
            f"[fa4] could not find the expected line to patch in {path}.\n"
            f"[fa4] expected to find: {BROKEN!r}\n"
            "[fa4] transformers may have changed; inspect flash_attention_forward "
            "and update this patch.",
            file=sys.stderr,
        )
        return 2

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src.replace(BROKEN, FIXED))
    print(f"[fa4] applied s_aux guard: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
