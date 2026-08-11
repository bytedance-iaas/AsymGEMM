"""CPU staging check: every dataset adapter must yield 2 full packs.

Run inside the container. Downloads land in the shared HF cache, so this
doubles as data staging for the GPU waves.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from route_skew_probe import DATASETS, build_packs  # noqa: E402


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash")
    keys = sys.argv[1:] or list(DATASETS)
    for k in keys:
        t0 = time.time()
        try:
            packs, n_docs = build_packs(k, tok, 16384, 2)
            spans = sum(len(p.spans) for p in packs)
            print(f"OK   {k:12s} packs=2 docs_scanned={n_docs} spans={spans} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"FAIL {k:12s} {type(e).__name__}: {str(e)[:200]} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
