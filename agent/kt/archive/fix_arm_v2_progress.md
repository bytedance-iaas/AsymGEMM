# ARM BF16 SFT Progress

Current status: complete for the strict production path.

## Implemented

- Packed ARM BF16 SFT forward is the only production forward algorithm.
- Backward uses packed route metadata and deterministic recomputation from saved
  inputs.
- Positive LoRA dropout is supported and covered against the torch oracle.
- Rank variants used by the local tests are covered.
- Worker-pool metadata, route metadata, alignment diagnostics, warmup metadata,
  async backward repack diagnostics, and pool-backed diagnostics are covered.
- Public benchmark scripts expose profiling and routing controls, but no path
  selector for removed algorithms.
- The validation helper no longer checks removed cache-down diagnostics.

## Latest Local Validation

Commands:

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
python setup.py build_ext --inplace
PYTHONPATH=python:. python -m pytest test/per_commit/test_armbf16_sft_reference.py test/per_commit/test_sft_lora_dropout.py -q
```

Result:

- native extension rebuild passed;
- focused tests passed: 30 tests.

Smoke benchmark:

- qlen 1 and qlen 8 both reported `last_forward_path == "packed"`;
- removed CLI flags were rejected by argparse;
- updated validator passed on the smoke JSON.

## Convergence Rule

Before declaring future convergence, run a fresh search for removed ARM BF16 SFT
surfaces in active code, validation scripts, and non-artifact implementation
docs. Historical artifact JSON/stderr files may still contain old measurements
and should not be used as current instructions.
