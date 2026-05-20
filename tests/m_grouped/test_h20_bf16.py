import sys
import importlib.util
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS_DIR))

spec = importlib.util.spec_from_file_location("_asym_h20_bf16_tests", TESTS_DIR / "test_h20_bf16.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

test_m_grouped_bf16_contiguous_sm90 = module.test_m_grouped_bf16_contiguous_sm90
test_m_grouped_bf16_masked_sm90 = module.test_m_grouped_bf16_masked_sm90


if __name__ == "__main__":
    test_m_grouped_bf16_contiguous_sm90()
    test_m_grouped_bf16_masked_sm90()
