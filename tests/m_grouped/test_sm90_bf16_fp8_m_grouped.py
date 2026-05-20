import importlib.util
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS_DIR))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


bf16_module = _load_module("_asym_sm90_bf16_m_grouped_tests", TESTS_DIR / "test_sm90_bf16_m_grouped.py")
fp8_module = _load_module("_asym_sm90_fp8_m_grouped_tests", TESTS_DIR / "test_sm90_fp8_m_grouped.py")

test_m_grouped_bf16_contiguous_sm90 = bf16_module.test_m_grouped_bf16_contiguous_sm90
test_m_grouped_bf16_contiguous_transpose_b_sm90 = bf16_module.test_m_grouped_bf16_contiguous_transpose_b_sm90
test_m_grouped_bf16_masked_sm90 = bf16_module.test_m_grouped_bf16_masked_sm90
test_m_grouped_fp8_contiguous_sm90 = fp8_module.test_m_grouped_fp8_contiguous_sm90
test_m_grouped_fp8_masked_sm90 = fp8_module.test_m_grouped_fp8_masked_sm90


if __name__ == "__main__":
    test_m_grouped_bf16_contiguous_sm90()
    test_m_grouped_bf16_contiguous_transpose_b_sm90()
    test_m_grouped_bf16_masked_sm90()
    test_m_grouped_fp8_contiguous_sm90()
    test_m_grouped_fp8_masked_sm90()
