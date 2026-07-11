import json

from llamafactory.launcher import _train_args_use_asym_gemm


def test_train_args_use_asym_gemm_from_yaml(tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text("use_asym_gemm: true\n", encoding="utf-8")

    assert _train_args_use_asym_gemm([str(config)])


def test_train_args_use_asym_gemm_from_json_override(tmp_path):
    config = tmp_path / "train.json"
    config.write_text(json.dumps({"use_asym_gemm": False}), encoding="utf-8")

    assert _train_args_use_asym_gemm([str(config), "use_asym_gemm=true"])


def test_train_args_use_asym_gemm_cli_override_can_disable_yaml(tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text("use_asym_gemm: true\n", encoding="utf-8")

    assert not _train_args_use_asym_gemm([str(config), "use_asym_gemm=false"])


def test_train_args_use_asym_gemm_from_cli_flag():
    assert _train_args_use_asym_gemm(["--use_asym_gemm", "true"])
