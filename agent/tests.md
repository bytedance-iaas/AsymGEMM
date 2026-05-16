# Tests

## Passing H20 Test

Run:

```bash
bash scripts/test.sh
```

This clears `~/.asym_gemm/cache` and runs:

```bash
python3 tests/m_grouped/test_h20_fp8.py
```

## Covered

- SM90/H20 FP8 m-grouped contiguous AsymGEMM
- SM90/H20 FP8 m-grouped masked AsymGEMM

Both paths compare against PyTorch BF16 reference output with `MAX_FP8_DIFF = 0.01`.

## Not Covered

- BF16 m-grouped H20 path
- FP8 non-grouped path
- FP8 k-grouped path
- NVFP4/FP4 paths

## Notes

- Do not run every `tests/test_*.py` as install validation.
- Some tests depend on stale DeepGEMM paths or unfinished exports.
- Use `bash scripts/install_editable.sh` after C++ binding changes.
