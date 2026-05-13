# debug: `m_grouped_moe_gemm_nt_contiguous` AttributeError

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `AttributeError: module 'asym_gemm' has no attribute 'm_grouped_moe_gemm_nt_contiguous'` when running `python3 test_sm80_moe.py` from the `tests/` directory.

**Architecture:** Two copies of `asym_gemm` co-exist: a local editable source at `/workspace/AsymGEMM_SM80/asym_gemm/` and a system-installed copy at `/usr/local/lib/python3.12/dist-packages/asym_gemm/`. When Python is launched from `tests/`, the system copy is found first because the current directory (`''` in `sys.path`) contains no `asym_gemm/` subdirectory. The system copy's `__init__.py` is stale — it was installed before `m_grouped_moe_gemm_nt_contiguous` was added to `_maybe_import_from_C`. The system `_C.so` *does* export the symbol; only `__init__.py` is outdated. The fix is to reinstall the local workspace as the canonical package so both paths resolve to the same, current code.

**Tech Stack:** Python 3.12, pybind11, PyTorch CUDA extension (`pip install -e .` editable install), setuptools.

---

## Diagnosis recap

| Location | `__init__.py` has symbol? | `_C.so` has symbol? |
|----------|--------------------------|---------------------|
| `/workspace/AsymGEMM_SM80/asym_gemm/` (local) | **YES** | **YES** |
| `/usr/local/lib/python3.12/dist-packages/asym_gemm/` (system) | **NO** | **YES** |

Running `tests$ python3 test_sm80_moe.py` → loads system `__init__.py` → symbol missing → crash.

---

### Task 1: Confirm the diagnosis

**Files:**
- Read: `/usr/local/lib/python3.12/dist-packages/asym_gemm/__init__.py`

- [ ] **Step 1: Reproduce the error from `tests/`**

```bash
cd /workspace/AsymGEMM_SM80/tests
python3 -c "import asym_gemm; print(asym_gemm.__file__)"
```

Expected output: `/usr/local/lib/python3.12/dist-packages/asym_gemm/__init__.py`
This confirms the system package wins.

- [ ] **Step 2: Confirm the symbol is absent in the system package**

```bash
python3 -c "
import sys
sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')
import importlib, asym_gemm
print('moe' in dir(asym_gemm))  # expect: False
"
```

Expected: `False`

- [ ] **Step 3: Confirm the symbol exists in the system _C.so**

```bash
python3 -c "
import ctypes, sys
sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')
import asym_gemm._C as C
print([x for x in dir(C) if 'moe' in x])
"
```

Expected: `['m_grouped_moe_gemm_nt_contiguous']` — proves the `.so` is fine and only `__init__.py` is stale.

---

### Task 2: Fix — reinstall the local workspace package

The cleanest fix is to reinstall from source so `/usr/local/lib/python3.12/dist-packages/asym_gemm/__init__.py` matches the current workspace.

**Files:**
- Modify: `/usr/local/lib/python3.12/dist-packages/asym_gemm/__init__.py` (indirectly, via pip)

- [ ] **Step 1: Reinstall as editable from workspace root**

```bash
cd /workspace/AsymGEMM_SM80
pip install -e . --no-build-isolation 2>&1 | tail -5
```

Expected output ends with `Successfully installed asym-gemm-...`

`--no-build-isolation` reuses already-built CUDA artifacts (avoids a lengthy NVCC recompile).

- [ ] **Step 2: Verify the system path now points to workspace**

```bash
python3 -c "import asym_gemm; print(asym_gemm.__file__)"
```

Expected: `/workspace/AsymGEMM_SM80/asym_gemm/__init__.py`

- [ ] **Step 3: Verify from `tests/` directory too**

```bash
cd /workspace/AsymGEMM_SM80/tests
python3 -c "import asym_gemm; print(asym_gemm.__file__); print(hasattr(asym_gemm, 'm_grouped_moe_gemm_nt_contiguous'))"
```

Expected:
```
/workspace/AsymGEMM_SM80/asym_gemm/__init__.py
True
```

---

### Task 3: If `pip install -e .` fails (fallback — patch `__init__.py` directly)

Only do this if Task 2's pip command errors (e.g., build tools unavailable).

**Files:**
- Modify: `/usr/local/lib/python3.12/dist-packages/asym_gemm/__init__.py`

- [ ] **Step 1: Find the `_maybe_import_from_C` call block**

```bash
grep -n "m_grouped_fp8_asym_gemm_nt_masked\|m_grouped_moe" \
    /usr/local/lib/python3.12/dist-packages/asym_gemm/__init__.py
```

This shows which symbols are currently in the list and whether `m_grouped_moe_gemm_nt_contiguous` is missing.

- [ ] **Step 2: Add the missing symbol to the list**

Open `/usr/local/lib/python3.12/dist-packages/asym_gemm/__init__.py`. In the `_maybe_import_from_C([...])` call, add `"m_grouped_moe_gemm_nt_contiguous"` alongside the other GEMM symbols. The list should look like:

```python
_maybe_import_from_C([
    # FP8 GEMMs
    "m_grouped_fp8_asym_gemm_nt_masked",
    "m_grouped_fp8_asym_gemm_nt_contiguous",
    # FP4 GEMMs
    "m_grouped_fp4_asym_gemm_nt_contiguous",
    "m_grouped_fp4_asym_gemm_nt_masked",
    # BF16 GEMMs
    "m_grouped_bf16_asym_gemm_nt_contiguous",
    "m_grouped_bf16_asym_gemm_nt_masked",
    # SM80 MoE GEMM (FP16 + BF16, JIT)
    "m_grouped_moe_gemm_nt_contiguous",
    ...
])
```

- [ ] **Step 3: Verify the symbol is now importable from `tests/`**

```bash
cd /workspace/AsymGEMM_SM80/tests
python3 -c "import asym_gemm; print(hasattr(asym_gemm, 'm_grouped_moe_gemm_nt_contiguous'))"
```

Expected: `True`

---

### Task 4: Run the full test suite

**Files:**
- Test: `tests/test_sm80_moe.py`

- [ ] **Step 1: Run from `tests/` as the user did originally**

```bash
cd /workspace/AsymGEMM_SM80/tests
python3 test_sm80_moe.py
```

Expected: all `[PASS]` lines, ending with `All tests passed.`

- [ ] **Step 2: Run from workspace root for good measure**

```bash
cd /workspace/AsymGEMM_SM80
python3 tests/test_sm80_moe.py
```

Expected: same `All tests passed.`

- [ ] **Step 3: Commit if changes were made to tracked files**

Only needed if Task 3 (direct `__init__.py` patch) was used and the change is in the workspace-tracked file (not the system path). If only the system-path file was edited as a temporary workaround, record in a follow-up commit to `asym_gemm/__init__.py` if it was missing the symbol there too.

```bash
cd /workspace/AsymGEMM_SM80
git diff asym_gemm/__init__.py
# If there's a diff, commit it:
git add asym_gemm/__init__.py
git commit -m "fix: add m_grouped_moe_gemm_nt_contiguous to _maybe_import_from_C"
```

---

## Summary of cause and fix

| | Detail |
|--|--|
| **Error** | `AttributeError: module 'asym_gemm' has no attribute 'm_grouped_moe_gemm_nt_contiguous'` |
| **Why** | System-installed `__init__.py` was installed before the SM80 MoE kernel was added; it doesn't list the symbol in `_maybe_import_from_C` |
| **Why from `tests/`** | `sys.path` includes `''` (current dir) first, but `tests/` has no `asym_gemm/` subdirectory, so Python falls through to the system install at `/usr/local/lib/python3.12/dist-packages/` |
| **Fix (preferred)** | `pip install -e .` from `/workspace/AsymGEMM_SM80` to replace the system install with an editable link to the workspace |
| **Fix (fallback)** | Add `"m_grouped_moe_gemm_nt_contiguous"` to the `_maybe_import_from_C` call in the system `__init__.py` |
