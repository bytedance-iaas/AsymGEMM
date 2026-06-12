from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def run_cmd(args: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    merged_env.update({"ROOT": str(ROOT), "ASYM_DIR": str(ROOT)})
    if env:
        merged_env.update(env)
    return subprocess.run(args, cwd=ROOT, env=merged_env, text=True, capture_output=True, check=check)


def make_fake_lf(tmp_path: Path) -> Path:
    lf_dir = tmp_path / "lf"
    lf_dir.joinpath("data").mkdir(parents=True)
    lf_dir.joinpath("src").mkdir()
    lf_dir.joinpath("examples/deepspeed").mkdir(parents=True)
    lf_dir.joinpath("data/dummy.jsonl").write_text('{"messages":[]}\n', encoding="utf-8")
    lf_dir.joinpath("src/train.py").write_text("", encoding="utf-8")
    for name in (
        "ds_z2_config.json",
        "ds_z3_config.json",
        "ds_z3_offload_config.json",
        "ds_z3_offload_mem_config.json",
        "ds_z3_cpuadam_config.json",
    ):
        lf_dir.joinpath("examples/deepspeed", name).write_text(
            json.dumps({"zero_optimization": {"stage": 3, "offload_optimizer": {"device": "cpu"}}}) + "\n",
            encoding="utf-8",
        )
    return lf_dir


def make_fake_env(tmp_path: Path) -> tuple[Path, Path]:
    fake_env = tmp_path / "env"
    fake_env.joinpath("bin").mkdir(parents=True)
    fake_python = fake_env / "bin/python"
    fake_cli = fake_env / "bin/llamafactory-cli"
    fake_python.write_text("#!/usr/bin/env bash\ncat >/dev/null\nexit 0\n", encoding="utf-8")
    fake_cli.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "printf '%s\\n' \"$@\" > \"$FAKE_LF_ARGS_LOG\"",
                "out_dir=",
                "prev=",
                "for arg in \"$@\"; do",
                "  if [[ \"$prev\" == \"--output_dir\" ]]; then",
                "    out_dir=\"$arg\"",
                "    break",
                "  fi",
                "  prev=\"$arg\"",
                "done",
                "if [[ -n \"$out_dir\" ]]; then",
                "  mkdir -p \"$out_dir\"",
                "  printf '%s\\n' '{\"loss\":1.0}' > \"$out_dir/trainer_log.jsonl\"",
                "fi",
                "if [[ -n \"${FAKE_LF_ENV_LOG:-}\" ]]; then",
                "  printf 'PYTHONPATH=%s\\n' \"${PYTHONPATH:-}\" > \"$FAKE_LF_ENV_LOG\"",
                "fi",
                "exit 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_cli.chmod(0o755)
    return fake_python, fake_cli


def _run_lf_lora_sft(tmp_path: Path, backend: str, *, extra_env: dict[str, str] | None = None) -> list[str]:
    lf_dir = make_fake_lf(tmp_path)
    fake_python, fake_cli = make_fake_env(tmp_path)
    args_log = tmp_path / f"{backend}_args.txt"
    env = {
        "LF_DIR": str(lf_dir),
        "ENV_DIR": str(fake_python.parent.parent),
        "ENV_PYTHON": str(fake_python),
        "LF_CLI_BIN": str(fake_cli),
        "FAKE_LF_ARGS_LOG": str(args_log),
        "BACKEND": backend,
        "GPU_ID": "0",
        "NUM_GPUS": "1",
        "PROFILE": "0",
        "NUMACTL_ENABLE": "0",
        "REQUIRE_SM100": "0",
        "CHECK_ASYM_CALLS": "0",
        "DATASET": "dummy",
        "TEMPLATE": "qwen3_nothink",
        "CUTOFF_LEN": "128",
        "MAX_SAMPLES": "1",
        "MAX_STEPS": "1",
        "LORA_RANK": "8",
        "LORA_ALPHA": "16",
        "LORA_DROPOUT": "0.00",
        "ASYM_CPU_ADAMW_PIN_MEMORY": "false",
        "ASYM_CPU_ADAMW_FP32_MASTER": "true",
    }
    if extra_env:
        env.update(extra_env)
    run_cmd(["scripts/lf/run_lf_lora_sft.sh"], env=env)
    return args_log.read_text(encoding="utf-8").splitlines()


def _arg_value(args: list[str], flag: str) -> str:
    assert flag in args
    return args[args.index(flag) + 1]


def test_run_lf_lora_sft_asym_cpuadamwtorch_args(tmp_path: Path) -> None:
    args = _run_lf_lora_sft(tmp_path, "asym_cpuadamwtorch")

    assert _arg_value(args, "--use_asym_gemm") == "true"
    assert _arg_value(args, "--asym_backend") == "asym"
    assert _arg_value(args, "--use_asym_cpu_adamw") == "true"
    assert _arg_value(args, "--asym_cpu_adamw_backend") == "torch"
    assert _arg_value(args, "--asym_cpu_adamw_pin_memory") == "false"
    assert _arg_value(args, "--asym_cpu_adamw_fp32_master") == "true"
    assert "--deepspeed" not in args


def test_run_lf_lora_sft_plain_asym_explicitly_disables_cpuadamw(tmp_path: Path) -> None:
    args = _run_lf_lora_sft(tmp_path, "asym", extra_env={"USE_ASYM_CPU_ADAMW": "false"})

    assert _arg_value(args, "--use_asym_gemm") == "true"
    assert _arg_value(args, "--asym_backend") == "asym"
    assert _arg_value(args, "--use_asym_cpu_adamw") == "false"
    assert _arg_value(args, "--asym_cpu_adamw_backend") == "deepspeed"
    assert "--deepspeed" not in args


def test_run_lf_lora_sft_asym_cpuadamwds_args(tmp_path: Path) -> None:
    env_log = tmp_path / "env.txt"
    deepspeed_dir = tmp_path / "deepspeed"
    deepspeed_dir.mkdir()
    args = _run_lf_lora_sft(
        tmp_path,
        "asym_cpuadamwds",
        extra_env={"DEEPSPEED_DIR": str(deepspeed_dir), "FAKE_LF_ENV_LOG": str(env_log)},
    )

    assert _arg_value(args, "--use_asym_cpu_adamw") == "true"
    assert _arg_value(args, "--asym_cpu_adamw_backend") == "deepspeed"
    assert "--deepspeed" not in args
    assert env_log.read_text(encoding="utf-8").startswith(f"PYTHONPATH={deepspeed_dir}:")


def test_run_lf_lora_sft_rejects_direct_cpuadamw_enablement(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    for backend in ("zero3_offload", "asym", "asym_torch"):
        result = run_cmd(
            ["scripts/lf/run_lf_lora_sft.sh"],
            env={
                "LF_DIR": str(lf_dir),
                "BACKEND": backend,
                "USE_ASYM_CPU_ADAMW": "true",
                "NUMACTL_ENABLE": "0",
                "REQUIRE_SM100": "0",
            },
            check=False,
        )
        assert result.returncode == 2
        assert "Use BACKEND=asym_cpuadamwtorch or BACKEND=asym_cpuadamwds" in result.stderr


def test_profile_lora_lf_dry_run_asym_cpuadamwtorch_label_and_flags(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    output_root = tmp_path / "dryrun"

    run_cmd(
        ["scripts/lf/profile_lora_lf.sh", "--model-specs", "Qwen/Qwen3-30B-A3B|1", "--output-root", str(output_root)],
        env={
            "LF_DIR": str(lf_dir),
            "BACKEND_SPECS": "asym_cpuadamwtorch|recomp",
            "GPU_POOL": "0",
            "PROFILERS": "source",
            "SEQ_LENS": "128",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "EXPERT_POLICIES": "none",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
    )

    command_files = list(output_root.rglob("command.txt"))
    assert len(command_files) == 1
    command = command_files[0].read_text(encoding="utf-8")
    assert "asym_cpuadamwtorch__source__recomp__polnone" in str(command_files[0])
    assert "BACKEND=asym_cpuadamwtorch" in command
    assert "PROFILE_BACKEND_LABEL=asym_cpuadamwtorch" in command
    assert "USE_ASYM_CPU_ADAMW=true" in command
    assert "ASYM_CPU_ADAMW_BACKEND=torch" in command


def test_profile_lora_lf_mixed_dry_run_does_not_leak_cpuadamw_to_zero(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    output_root = tmp_path / "dryrun"

    run_cmd(
        ["scripts/lf/profile_lora_lf.sh", "--model-specs", "Qwen/Qwen3-30B-A3B|1", "--output-root", str(output_root)],
        env={
            "LF_DIR": str(lf_dir),
            "BACKEND_SPECS": "asym_cpuadamwtorch|recomp,zero3_offload|recomp",
            "GPU_POOL": "0",
            "PROFILERS": "source",
            "SEQ_LENS": "128",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "EXPERT_POLICIES": "none",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
    )

    commands = {str(path): path.read_text(encoding="utf-8") for path in output_root.rglob("command.txt")}
    assert any("asym_cpuadamwtorch__source__recomp__polnone" in path for path in commands)
    zero_commands = [text for path, text in commands.items() if "zero3_offload__source__recomp__polnone" in path]
    assert len(zero_commands) == 1
    assert "BACKEND=zero3_offload" in zero_commands[0]
    assert "USE_ASYM_CPU_ADAMW=false" in zero_commands[0]
    assert "PROFILE_BACKEND_LABEL=zero3_offload" in zero_commands[0]


def test_profile_lora_lf_test_wrapper_preserves_output_root(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    deepspeed_dir = tmp_path / "deepspeed"
    output_root = tmp_path / "dryrun"

    run_cmd(
        [
            "scripts/lf/profile_lora_lf_test.sh",
            "--model-specs",
            "Qwen/Qwen3-30B-A3B|1",
            "--output-root",
            str(output_root),
        ],
        env={
            "LF_DIR": str(lf_dir),
            "DEEPSPEED_DIR": str(deepspeed_dir),
            "BACKEND_SPECS": "asym_cpuadamwds|recomp",
            "GPU_POOL": "0",
            "PROFILERS": "source",
            "SEQ_LENS": "128",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "EXPERT_POLICIES": "none",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
    )

    command_files = list(output_root.rglob("command.txt"))
    assert len(command_files) == 1
    command = command_files[0].read_text(encoding="utf-8")
    assert "BACKEND=asym_cpuadamwds" in command
    assert "ASYM_CPU_ADAMW_BACKEND=deepspeed" in command
    assert f"DEEPSPEED_DIR={deepspeed_dir}" in command
    assert f"ASYM_GEMM_LF_CONFIG_DEEPSPEED_DIR={deepspeed_dir}" in command
    assert "asym_cpuadamwds__source__recomp__polnone" in str(command_files[0])


def test_profile_lora_lf_default_e2e_shape_uses_asym_cpuadamwds(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    deepspeed_dir = tmp_path / "deepspeed"
    output_root = tmp_path / "dryrun"

    run_cmd(
        ["scripts/lf/profile_lora_lf.sh", "--output-root", str(output_root)],
        env={
            "LF_DIR": str(lf_dir),
            "DEEPSPEED_DIR": str(deepspeed_dir),
            "GPU_POOL": "0",
            "PROFILERS": "source",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.08",
            "EXPERT_POLICIES": "none",
            "ROUTER_MODES": "whole",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
    )

    command_files = sorted(output_root.rglob("command.txt"))
    assert len(command_files) == 2
    command_texts = [path.read_text(encoding="utf-8") for path in command_files]
    command_paths = "\n".join(str(path) for path in command_files)

    assert "qwen3_5-122b-a10b__gpus1__b4_s8192_w5_s10_r64_a16_drop008" in command_paths
    assert "llama-4-scout-17b-16e__gpus1__b4_s8192_w5_s10_r64_a16_drop008" in command_paths
    assert command_paths.count("asym_cpuadamwds__source__recomp__polnone__routerwhole/b4_s8192") == 2
    for command in command_texts:
        assert "BACKEND=asym_cpuadamwds" in command
        assert "PROFILE_BACKEND_LABEL=asym_cpuadamwds" in command
        assert "USE_ASYM_CPU_ADAMW=true" in command
        assert "ASYM_CPU_ADAMW_BACKEND=deepspeed" in command
        assert f"DEEPSPEED_DIR={deepspeed_dir}" in command
        assert f"ASYM_GEMM_LF_CONFIG_DEEPSPEED_DIR={deepspeed_dir}" in command
        assert "ASYM_OFFLOAD_MODULES=all" in command
        assert "PROFILE_MEMORY_ATTRIBUTION=false" in command
        assert "PROFILE_MEMORY_BREAKDOWN=false" in command
        assert "PER_DEVICE_TRAIN_BATCH_SIZE=4" in command
        assert "CUTOFF_LEN=8192" in command


@pytest.mark.parametrize("offload_modules", ["all", "routed_experts", "attention,router", "none"])
def test_profile_lora_lf_dry_run_three_model_families_preserves_offload_modules(
    tmp_path: Path, offload_modules: str
) -> None:
    lf_dir = make_fake_lf(tmp_path)
    deepspeed_dir = tmp_path / "deepspeed"
    output_root = tmp_path / offload_modules.replace(",", "_")

    run_cmd(
        [
            "scripts/lf/profile_lora_lf.sh",
            "--model-specs",
            "Qwen/Qwen3-30B-A3B|1,Qwen/Qwen3.5-122B-A10B|1,meta-llama/Llama-4-Scout-17B-16E|1",
            "--output-root",
            str(output_root),
        ],
        env={
            "LF_DIR": str(lf_dir),
            "DEEPSPEED_DIR": str(deepspeed_dir),
            "BACKEND_SPECS": "asym_cpuadamwds|recomp",
            "GPU_POOL": "0",
            "PROFILERS": "source",
            "SEQ_LENS": "128",
            "PER_DEVICE_TRAIN_BATCH_SIZE": "1",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "EXPERT_POLICIES": "none",
            "ROUTER_MODES": "whole",
            "ASYM_OFFLOAD_MODULES": offload_modules,
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
    )

    command_files = sorted(output_root.rglob("command.txt"))
    assert len(command_files) == 3
    command_paths = "\n".join(str(path) for path in command_files)
    assert "qwen3-30b-a3b__gpus1__b1_s128_w0_s1_r64_a16_drop000" in command_paths
    assert "qwen3_5-122b-a10b__gpus1__b1_s128_w0_s1_r64_a16_drop000" in command_paths
    assert "llama-4-scout-17b-16e__gpus1__b1_s128_w0_s1_r64_a16_drop000" in command_paths

    command_offload_modules = offload_modules.replace(",", "\\,")
    for path in command_files:
        command = path.read_text(encoding="utf-8")
        assert "BACKEND=asym_cpuadamwds" in command
        assert "PROFILE_BACKEND_LABEL=asym_cpuadamwds" in command
        assert "USE_ASYM_CPU_ADAMW=true" in command
        assert "ASYM_CPU_ADAMW_BACKEND=deepspeed" in command
        assert f"ASYM_GEMM_LF_CONFIG_DEEPSPEED_DIR={deepspeed_dir}" in command
        assert f"ASYM_OFFLOAD_MODULES={command_offload_modules}" in command
