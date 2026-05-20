import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
DEMO_PATH = ROOT / "examples" / "asymgemm" / "mlp_lora_demo.py"


def _load_demo_module():
    spec = importlib.util.spec_from_file_location("mlp_lora_demo", DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _direct_bf16_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] in {9, 10}


def _fmt_mib(value: int | float) -> str:
    return f"{float(value) / (1024 ** 2):.3f} MiB"


def _print_report_summary(report: dict) -> None:
    memory = report["memory_comparison"]
    normal = memory["normal_gpu_resident"]
    asym = memory["asym_cpu_resident"]

    print("\n[M2 numerical error]")
    print(f"  forward max_abs: {report['forward_parity_max_abs']:.6g}")
    print(f"  scalar loss abs: {report['scalar_loss_parity_abs']:.6g}")
    print(f"  input grad max_abs: {report['input_grad_parity_max_abs']:.6g}")
    print(f"  LoRA grad worst max_abs: {report['lora_grad_parity_worst_max_abs']:.6g}")

    print("[M2 memory comparison]")
    print(
        "  normal GPU-resident: "
        f"model_hbm={_fmt_mib(normal['model_hbm_bytes'])}, "
        f"peak_hbm={_fmt_mib(normal['peak_hbm_bytes'])}, "
        f"accounted_cpu={_fmt_mib(normal['cpu_model_bytes_after_step'])}, "
        f"base_cpu={_fmt_mib(normal['cpu_resident_base_weight_bytes'])}, "
        f"rss_before={_fmt_mib(normal['rss_before_bytes'])}, "
        f"rss_after_step={_fmt_mib(normal['rss_after_step_bytes'])}, "
        f"rss_step_delta={_fmt_mib(normal['rss_step_delta_bytes'])}"
    )
    print(
        "  AsymGEMM CPU-resident: "
        f"model_hbm={_fmt_mib(asym['model_hbm_bytes'])}, "
        f"peak_hbm={_fmt_mib(asym['peak_hbm_bytes'])}, "
        f"accounted_cpu={_fmt_mib(asym['cpu_model_bytes_after_step'])}, "
        f"base_cpu={_fmt_mib(asym['cpu_resident_base_weight_bytes'])}, "
        f"rss_before={_fmt_mib(asym['rss_before_bytes'])}, "
        f"rss_after_step={_fmt_mib(asym['rss_after_step_bytes'])}, "
        f"rss_step_delta={_fmt_mib(asym['rss_step_delta_bytes'])}"
    )
    print(
        "  savings/deltas: "
        f"model_hbm_saved={_fmt_mib(memory['hbm_model_saved_bytes'])}, "
        f"peak_hbm_saved={_fmt_mib(memory['hbm_peak_saved_bytes'])}, "
        f"extra_cpu_accounted={_fmt_mib(memory['cpu_model_extra_bytes'])}"
    )
    print(f"  direct_fetch_forward={report['direct_fetch_forward_used']}, direct_fetch_dx={report['direct_fetch_dx_used']}")


def test_mlp_demo_cpu_torch_path_emits_correct_report(tmp_path: Path) -> None:
    demo = _load_demo_module()
    report_path = tmp_path / "mlp_demo_cpu.json"
    report = demo.run_demo(backend="torch_only", report_path=report_path, device="cpu")
    _print_report_summary(report)

    assert report_path.exists()
    assert report["forward_parity_max_abs"] < 1e-6
    assert report["scalar_loss_parity_abs"] < 1e-6
    assert report["input_grad_parity_max_abs"] < 1e-6
    assert report["lora_grad_parity_worst_max_abs"] < 1e-6
    assert report["optimizer_step_loss_moved"]
    assert report["frozen_base_unchanged"]
    assert report["base_absent_from_optimizer_state"]
    assert report["memory_warmup_performed"]
    assert report["tf32_disabled"]
    assert report["pinned_cpu_bytes"] == 0
    assert report["memory_comparison"]["normal_gpu_resident"]["mode"] == "normal_gpu_resident"
    assert report["memory_comparison"]["asym_cpu_resident"]["mode"] == "asym_cpu_resident"
    assert report["memory_comparison"]["asym_cpu_resident"]["cpu_resident_base_weight_bytes"] > 0


@pytest.mark.skipif(not _direct_bf16_available(), reason="direct-fetch MLP demo requires SM90/SM100")
def test_mlp_demo_direct_fetch_correctness_and_hbm_report(tmp_path: Path) -> None:
    demo = _load_demo_module()
    report_path = tmp_path / "mlp_demo_direct_bf16.json"
    report = demo.run_demo(backend="asym_only", report_path=report_path, device="cuda")
    _print_report_summary(report)

    assert report_path.exists()
    assert report["number_of_asymgemm_calls"] >= 4
    assert report["direct_fetch_forward_used"]
    assert report["direct_fetch_dx_used"]
    assert report["fallback_counts"]["staged_calls"] == 0
    assert report["fallback_counts"]["torch_calls"] == 0
    assert report["optimizer_step_loss_moved"]
    assert report["frozen_base_unchanged"]
    assert report["base_absent_from_optimizer_state"]
    assert report["memory_warmup_performed"]
    assert report["tf32_disabled"]

    assert report["forward_parity_max_abs"] <= 0.5
    assert report["scalar_loss_parity_abs"] <= 0.5
    assert report["input_grad_parity_max_abs"] <= 0.5
    assert report["lora_grad_parity_worst_max_abs"] <= 0.5

    assert report["pinned_cpu_bytes"] >= report["expected_hbm_saved_bytes"]
    assert report["gpu_resident_baseline_weight_bytes"] >= report["expected_hbm_saved_bytes"]
    assert report["expected_hbm_saved_bytes"] > 0
    assert report["peak_hbm"] > 0

    memory = report["memory_comparison"]
    normal = memory["normal_gpu_resident"]
    asym = memory["asym_cpu_resident"]
    assert normal["model_hbm_bytes"] > asym["model_hbm_bytes"]
    assert memory["hbm_model_saved_bytes"] >= report["expected_hbm_saved_bytes"]
    assert asym["cpu_model_bytes_after_step"] > normal["cpu_model_bytes_after_step"]
    assert asym["cpu_resident_base_weight_bytes"] == report["expected_hbm_saved_bytes"]
    assert asym["execution_stats"]["asym_forward_calls"] >= 2
    assert asym["execution_stats"]["asym_dx_calls"] >= 2
    assert asym["execution_stats"]["staged_calls"] == 0
    assert asym["execution_stats"]["torch_calls"] == 0
