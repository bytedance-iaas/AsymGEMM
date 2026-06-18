from __future__ import annotations

import csv
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
                "  printf 'ASYMM_ATTN_ACT_OFFLOAD=%s\\n' \"${ASYMM_ATTN_ACT_OFFLOAD:-}\" >> \"$FAKE_LF_ENV_LOG\"",
                "  printf 'ASYMM_LAYER_ACT_OFFLOAD=%s\\n' \"${ASYMM_LAYER_ACT_OFFLOAD:-}\" >> \"$FAKE_LF_ENV_LOG\"",
                "  printf 'ASYM_CPU_ADAMW_GRAD_OFFLOAD=%s\\n' \"${ASYM_CPU_ADAMW_GRAD_OFFLOAD:-}\" >> \"$FAKE_LF_ENV_LOG\"",
                "  printf 'ASYM_GEMM_LF_CONFIG_ASYM_STRICT=%s\\n' \"${ASYM_GEMM_LF_CONFIG_ASYM_STRICT:-}\" >> \"$FAKE_LF_ENV_LOG\"",
                "  printf 'ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_ACT_OFFLOAD=%s\\n' \"${ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_ACT_OFFLOAD:-}\" >> \"$FAKE_LF_ENV_LOG\"",
                "  printf 'ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_ACT_OFFLOAD=%s\\n' \"${ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_ACT_OFFLOAD:-}\" >> \"$FAKE_LF_ENV_LOG\"",
                "  printf 'ASYM_GEMM_LF_CONFIG_ATTN_GC_ENABLED=%s\\n' \"${ASYM_GEMM_LF_CONFIG_ATTN_GC_ENABLED:-}\" >> \"$FAKE_LF_ENV_LOG\"",
                "  printf 'ASYM_GEMM_LF_CONFIG_LAYER_GC_ENABLED=%s\\n' \"${ASYM_GEMM_LF_CONFIG_LAYER_GC_ENABLED:-}\" >> \"$FAKE_LF_ENV_LOG\"",
                "  printf 'ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_GRAD_OFFLOAD=%s\\n' \"${ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_GRAD_OFFLOAD:-}\" >> \"$FAKE_LF_ENV_LOG\"",
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
        "ASYM_CPU_ADAMW_GRAD_OFFLOAD": "false",
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
    assert _arg_value(args, "--asym_cpu_adamw_grad_offload") == "false"
    assert _arg_value(args, "--enable_liger_kernel") == "false"
    assert "--deepspeed" not in args


def test_run_lf_lora_sft_passes_enable_liger_kernel_arg(tmp_path: Path) -> None:
    args = _run_lf_lora_sft(tmp_path, "asym_cpuadamwtorch", extra_env={"ENABLE_LIGER_KERNEL": "true"})

    assert _arg_value(args, "--enable_liger_kernel") == "true"


def test_run_lf_lora_sft_plain_asym_explicitly_disables_cpuadamw(tmp_path: Path) -> None:
    args = _run_lf_lora_sft(tmp_path, "asym", extra_env={"USE_ASYM_CPU_ADAMW": "false"})

    assert _arg_value(args, "--use_asym_gemm") == "true"
    assert _arg_value(args, "--asym_backend") == "asym"
    assert _arg_value(args, "--use_asym_cpu_adamw") == "false"
    assert _arg_value(args, "--asym_cpu_adamw_backend") == "deepspeed"
    assert _arg_value(args, "--asym_cpu_adamw_grad_offload") == "false"
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
    assert _arg_value(args, "--asym_cpu_adamw_grad_offload") == "false"
    assert "--deepspeed" not in args
    assert env_log.read_text(encoding="utf-8").startswith(f"PYTHONPATH={deepspeed_dir}:")


def test_run_lf_lora_sft_passes_grad_offload_arg_and_env(tmp_path: Path) -> None:
    env_log = tmp_path / "env.txt"
    args = _run_lf_lora_sft(
        tmp_path,
        "asym_cpuadamwtorch",
        extra_env={"ASYM_CPU_ADAMW_GRAD_OFFLOAD": "true", "FAKE_LF_ENV_LOG": str(env_log)},
    )

    assert _arg_value(args, "--asym_cpu_adamw_grad_offload") == "true"
    env_text = env_log.read_text(encoding="utf-8")
    assert "ASYM_CPU_ADAMW_GRAD_OFFLOAD=true" in env_text


def test_run_lf_lora_sft_passes_attention_activation_env(tmp_path: Path) -> None:
    env_log = tmp_path / "env.txt"
    args = _run_lf_lora_sft(
        tmp_path,
        "asym",
        extra_env={"ASYMM_ATTN_ACT_OFFLOAD": "true", "FAKE_LF_ENV_LOG": str(env_log)},
    )

    assert _arg_value(args, "--use_asym_gemm") == "true"
    env_text = env_log.read_text(encoding="utf-8")
    assert "ASYMM_ATTN_ACT_OFFLOAD=true" in env_text


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
            "WORKLOADS": "4096|4|1",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "ASYMM_EXP_ACT_POLICIES": "none|false|false|false",
            # Pin the offload knobs this test asserts instead of relying on script defaults.
            "ASYM_CPU_ADAMW_GRAD_OFFLOAD": "false",
            "ASYM_CPU_ADAMW_WEIGHT_OFFLOAD": "false",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
    )

    command_files = list(output_root.rglob("command.txt"))
    assert len(command_files) == 1
    command = command_files[0].read_text(encoding="utf-8")
    assert "asym_cpuadamwtorch__source__recomp__polnone" in str(command_files[0])
    assert "__ligerloss0__gradofffalse__weightofffalse/b4_s4096_ga1" in str(command_files[0])
    assert "BACKEND=asym_cpuadamwtorch" in command
    assert "PROFILE_BACKEND_LABEL=asym_cpuadamwtorch" in command
    assert "USE_ASYM_CPU_ADAMW=true" in command
    assert "ASYM_CPU_ADAMW_BACKEND=torch" in command
    assert "ASYM_CPU_ADAMW_GRAD_OFFLOAD=false" in command
    assert "ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_GRAD_OFFLOAD=false" in command


def test_profile_lora_lf_dry_run_sweeps_asym_cpuadamw_grad_offload_modes(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    output_root = tmp_path / "dryrun"

    run_cmd(
        ["scripts/lf/profile_lora_lf.sh", "--model-specs", "Qwen/Qwen3-30B-A3B|1", "--output-root", str(output_root)],
        env={
            "LF_DIR": str(lf_dir),
            "BACKEND_SPECS": "asym_cpuadamwtorch|recomp",
            "ASYM_CPU_ADAMW_GRAD_OFFLOAD": "false,true",
            # Pin weight-offload off so the weight-requires-grad gate does not drop the grad=false job.
            "ASYM_CPU_ADAMW_WEIGHT_OFFLOAD": "false",
            "GPU_POOL": "0",
            "PROFILERS": "source",
            "WORKLOADS": "4096|4|1",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "ASYMM_EXP_ACT_POLICIES": "none|false|false|false",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
    )

    commands = {str(path): path.read_text(encoding="utf-8") for path in output_root.rglob("command.txt")}
    assert len(commands) == 2
    command_paths = "\n".join(commands)
    assert "__ligerloss0__gradofffalse__weightofffalse/" in command_paths
    assert "__ligerloss0__gradofftrue__weightofffalse/" in command_paths
    assert any("ASYM_CPU_ADAMW_GRAD_OFFLOAD=false" in text for text in commands.values())
    assert any("ASYM_CPU_ADAMW_GRAD_OFFLOAD=true" in text for text in commands.values())
    assert any("ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_GRAD_OFFLOAD=false" in text for text in commands.values())
    assert any("ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_GRAD_OFFLOAD=true" in text for text in commands.values())

    jobs_tsv = next(output_root.rglob("jobs.tsv")).read_text(encoding="utf-8")
    assert "profiler\tliger_loss\tgrad_offload\tjob_dir" in jobs_tsv
    assert "\tsource\tligerloss0\tfalse\t" in jobs_tsv
    assert "\tsource\tligerloss0\ttrue\t" in jobs_tsv


def test_profile_lora_lf_dry_run_wires_liger_loss_axis(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    output_root = tmp_path / "dryrun"

    run_cmd(
        ["scripts/lf/profile_lora_lf.sh", "--model-specs", "Qwen/Qwen3-30B-A3B|1", "--output-root", str(output_root)],
        env={
            "LF_DIR": str(lf_dir),
            "BACKEND_SPECS": "asym_cpuadamwds|norecomp|ligerloss0,zero3_offload|recomp|ligerloss1",
            "GPU_POOL": "0",
            "PROFILERS": "both",
            "WORKLOADS": "128|1|1",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "ASYMM_EXP_ACT_POLICIES": "none|false|false|false",
            "ASYM_CPU_ADAMW_GRAD_OFFLOAD": "false",
            "ASYM_CPU_ADAMW_WEIGHT_OFFLOAD": "false",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
    )

    commands = {str(path): path.read_text(encoding="utf-8") for path in output_root.rglob("command.txt")}
    command_paths = "\n".join(commands)
    command_text = "\n".join(commands.values())
    assert "__nsys__norecomp__polnone" in command_paths
    assert "__source__norecomp__polnone" in command_paths
    assert "__ligerloss0" in command_paths
    assert "__ligerloss1" in command_paths
    assert "ENABLE_LIGER_KERNEL=false" in command_text
    assert "ENABLE_LIGER_KERNEL=true" in command_text
    assert "ASYM_GEMM_LF_CONFIG_LIGER_LOSS=ligerloss0" in command_text
    assert "ASYM_GEMM_LF_CONFIG_LIGER_LOSS=ligerloss1" in command_text


def test_plot_activation_recompute_sweep_filters_liger_loss_axis(tmp_path: Path) -> None:
    input_root = tmp_path / "profiles"
    for liger_loss in ("ligerloss0", "ligerloss1"):
        run_dir = (
            input_root
            / "precision"
            / (
                "torch__source__norecomp__polnone__routerhf__expact0__attnact0__layeract0__"
                f"loraafwdhbm__actrecomp0__xunpack0__{liger_loss}"
            )
            / "b1_s128_ga1"
        )
        run_dir.mkdir(parents=True)
        profile = {
            "config": {
                "backend": "torch",
                "router_mode": "hf",
                "seq_len": 128,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "precision": "bf16",
                "liger_loss": liger_loss,
                "lora_dropout": 0.0,
                "expert_policy": "none",
            },
            "stages": [{"name": "step", "total_milliseconds": 10.0}],
            "forward": {"total_milliseconds": 4.0},
            "backward": {"total_milliseconds": 6.0},
            "memory": {"gpu": {"peak_allocated_hbm_bytes": 1024, "peak_reserved_hbm_bytes": 2048}},
            "trainable_surface": {"available": True, "surface": "synthetic"},
        }
        (run_dir / "profile.json").write_text(json.dumps(profile) + "\n", encoding="utf-8")

    output_dir = tmp_path / "plots"
    run_cmd(
        [
            sys.executable,
            "scripts/plotting/plot_activation_recompute_sweep.py",
            "--input-root",
            str(input_root),
            "--output-dir",
            str(output_dir),
            "--backend",
            "torch",
            "--profiler",
            "source",
            "--recompute",
            "norecomp",
            "--liger-loss",
            "ligerloss0",
            "--skip-combined",
        ]
    )

    rows = list(csv.DictReader((output_dir / "activation_recompute_sweep_index.csv").open()))
    assert len(rows) == 1
    assert rows[0]["liger_loss"] == "ligerloss0"


def test_profile_lora_lf_dry_run_rejects_multi_qwen_expert_lora_impls(tmp_path: Path) -> None:
    # The multi-impl sweep was retired: passing more than one comma-separated implementation must
    # now hard-fail instead of fanning out a job per implementation.
    lf_dir = make_fake_lf(tmp_path)
    output_root = tmp_path / "dryrun"

    result = run_cmd(
        ["scripts/lf/profile_lora_lf.sh", "--model-specs", "Qwen/Qwen3-30B-A3B|1", "--output-root", str(output_root)],
        env={
            "LF_DIR": str(lf_dir),
            "BACKEND_SPECS": "zero3_offload|recomp",
            "GPU_POOL": "0",
            "PROFILERS": "source",
            "WORKLOADS": "128|8|1",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "ASYMM_EXP_ACT_POLICIES": "none|false|false|false",
            "LF_EXPERT_LORA_IMPLS": "peft-target-parameters,split-target-parameters",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
        check=False,
    )

    assert result.returncode != 0
    assert "LF_EXPERT_LORA_IMPLS must be a single implementation; the multi-impl sweep was retired" in result.stderr
    assert not list(output_root.rglob("command.txt"))


def test_profile_lora_lf_dry_run_labels_expert_lora_a_hbm_mode(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    output_root = tmp_path / "dryrun"

    run_cmd(
        ["scripts/lf/profile_lora_lf.sh", "--model-specs", "Qwen/Qwen3-30B-A3B|1", "--output-root", str(output_root)],
        env={
            "LF_DIR": str(lf_dir),
            "BACKEND_SPECS": "asym_cpuadamwds|norecomp",
            "GPU_POOL": "0",
            "PROFILERS": "source",
            "WORKLOADS": "4096|4|1",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "ASYMM_EXP_ACT_POLICIES": "none|true|true|true",
            "ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD": "hbm",
            # Pin the offload knobs so the dir suffix is deterministic.
            "ASYM_CPU_ADAMW_GRAD_OFFLOAD": "false",
            "ASYM_CPU_ADAMW_WEIGHT_OFFLOAD": "false",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
    )

    commands = {str(path): path.read_text(encoding="utf-8") for path in output_root.rglob("command.txt")}
    assert len(commands) == 1
    command_paths = "\n".join(commands)
    assert (
        "__expact1__attnact1__layeract1__loraafwdhbm__actrecomp0__xunpack0"
        "__ligerloss0__gradofffalse__weightofffalse/b4_s4096_ga1"
    ) in command_paths
    command = next(iter(commands.values()))
    assert "ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=hbm" in command
    assert "ASYM_GEMM_LF_CONFIG_ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=hbm" in command


@pytest.mark.parametrize("bad_value", ["cpu_left", "gpu_hbm", "gpu-hbm", "gpu", "HBM", ""])
def test_profile_lora_lf_rejects_non_exact_expert_lora_a_modes(tmp_path: Path, bad_value: str) -> None:
    lf_dir = make_fake_lf(tmp_path)
    result = run_cmd(
        ["scripts/lf/profile_lora_lf.sh", "--model-specs", "Qwen/Qwen3-30B-A3B|1", "--output-root", str(tmp_path / "dryrun")],
        env={
            "LF_DIR": str(lf_dir),
            "BACKEND_SPECS": "asym_cpuadamwds|norecomp",
            "GPU_POOL": "0",
            "PROFILERS": "source",
            "WORKLOADS": "128|8|1",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD": bad_value,
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
        check=False,
    )
    assert result.returncode == 2
    assert "ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD must be cpu or hbm" in result.stderr


def test_profile_lora_lf_rejects_three_field_exp_act_policy_tuple(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    result = run_cmd(
        ["scripts/lf/profile_lora_lf.sh", "--model-specs", "Qwen/Qwen3-30B-A3B|1", "--output-root", str(tmp_path / "dryrun")],
        env={
            "LF_DIR": str(lf_dir),
            "BACKEND_SPECS": "asym_cpuadamwds|norecomp",
            "GPU_POOL": "0",
            "PROFILERS": "source",
            "WORKLOADS": "128|8|1",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "ASYMM_EXP_ACT_POLICIES": "none|true|false",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
        check=False,
    )
    assert result.returncode == 2
    assert "ASYMM_EXP_ACT_POLICIES item must be policy|expert_act|attn_act|layer_act" in result.stderr


def test_profile_lora_lf_four_field_exp_attn_axis_dry_run(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    output_root = tmp_path / "dryrun"

    run_cmd(
        ["scripts/lf/profile_lora_lf.sh", "--model-specs", "Qwen/Qwen3-30B-A3B|1", "--output-root", str(output_root)],
        env={
            "LF_DIR": str(lf_dir),
            "BACKEND_SPECS": "asym_cpuadamwds|norecomp",
            "GPU_POOL": "0",
            "PROFILERS": "source",
            "WORKLOADS": "4096|4|1",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "ASYMM_EXP_ACT_POLICIES": "none|false|false|false,gc-exp|false|false|false,gc-attn-exp|false|false|false,none|true|false|false,none|true|true|false",
            # Pin the dir-shaping knobs so the expected paths are deterministic.
            "ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD": "cpu",
            "ASYM_CPU_ADAMW_GRAD_OFFLOAD": "false",
            "ASYM_CPU_ADAMW_WEIGHT_OFFLOAD": "false",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
    )

    commands = {str(path): path.read_text(encoding="utf-8") for path in output_root.rglob("command.txt")}
    assert len(commands) == 5
    command_paths = "\n".join(commands)
    neutral = "__actrecomp0__xunpack0__ligerloss0__gradofffalse__weightofffalse/"
    assert f"__polnone__routerwhole__expact0__attnact0__layeract0__loraafwdcpu{neutral}" in command_paths
    assert f"__polgc-exp__routerwhole__expact0__attnact0__layeract0__loraafwdcpu{neutral}" in command_paths
    assert f"__polgc-attn-exp__routerwhole__expact0__attnact0__layeract0__loraafwdcpu{neutral}" in command_paths
    assert f"__polnone__routerwhole__expact1__attnact0__layeract0__loraafwdcpu{neutral}" in command_paths
    assert f"__polnone__routerwhole__expact1__attnact1__layeract0__loraafwdcpu{neutral}" in command_paths

    gc_attn_command = next(text for path, text in commands.items() if "__polgc-attn-exp__" in path)
    assert "ASYM_GEMM_LF_CONFIG_ASYM_STRICT=true" in gc_attn_command
    assert "ASYM_EXPERT_RECOMPUTE_POLICY=gc-attn-exp" in gc_attn_command
    assert "GRADIENT_CHECKPOINTING=false" in gc_attn_command
    assert "ASYMM_ATTN_ACT_OFFLOAD=false" in gc_attn_command
    assert "ASYM_GEMM_LF_CONFIG_ATTN_GC_ENABLED=true" in gc_attn_command
    assert "ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_ACT_OFFLOAD=false" in gc_attn_command
    assert "ASYMM_LAYER_ACT_OFFLOAD=false" in gc_attn_command
    assert "ASYM_GEMM_LF_CONFIG_LAYER_GC_ENABLED=false" in gc_attn_command

    gc_exp_command = next(text for path, text in commands.items() if "__polgc-exp__" in path)
    assert "ASYM_GEMM_LF_CONFIG_ATTN_GC_ENABLED=false" in gc_exp_command

    attnact_command = next(
        text
        for path, text in commands.items()
        if f"__expact1__attnact1__layeract0__loraafwdcpu{neutral}" in path
    )
    assert "ASYMM_EXPERT_ACT_OFFLOAD=true" in attnact_command
    assert "ASYMM_ATTN_ACT_OFFLOAD=true" in attnact_command
    assert "ASYMM_LAYER_ACT_OFFLOAD=false" in attnact_command
    assert "ASYM_EXPERT_RECOMPUTE_POLICY=none" in attnact_command
    assert "ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_ACT_OFFLOAD=true" in attnact_command
    assert "ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_ACT_OFFLOAD=false" in attnact_command
    assert "ASYM_GEMM_LF_CONFIG_ATTN_GC_ENABLED=false" in attnact_command


def test_profile_lora_lf_rejects_selective_gc_with_global_recomp(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    output_root = tmp_path / "dryrun"

    result = run_cmd(
        ["scripts/lf/profile_lora_lf.sh", "--model-specs", "Qwen/Qwen3-30B-A3B|1", "--output-root", str(output_root)],
        env={
            "LF_DIR": str(lf_dir),
            "BACKEND_SPECS": "asym_cpuadamwds|recomp",
            "GPU_POOL": "0",
            "PROFILERS": "source",
            "WORKLOADS": "128|8|1",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "ASYMM_EXP_ACT_POLICIES": "gc-attn-exp|false|false|false",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
        check=False,
    )

    assert result.returncode == 2
    assert "selective GC" in result.stderr
    assert "backend recompute=norecomp" in result.stderr


def test_profile_lora_lf_rejects_activation_offload_with_global_recomp(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    output_root = tmp_path / "dryrun"

    result = run_cmd(
        ["scripts/lf/profile_lora_lf.sh", "--model-specs", "Qwen/Qwen3-30B-A3B|1", "--output-root", str(output_root)],
        env={
            "LF_DIR": str(lf_dir),
            "BACKEND_SPECS": "asym_cpuadamwds|recomp",
            "GPU_POOL": "0",
            "PROFILERS": "source",
            "WORKLOADS": "128|8|1",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "ASYMM_EXP_ACT_POLICIES": "none|true|true|true",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
        check=False,
    )

    assert result.returncode == 2
    assert "activation offload tuples" in result.stderr
    assert "backend recompute=norecomp" in result.stderr


def test_profile_lora_lf_four_part_layer_axis_dry_run(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    output_root = tmp_path / "dryrun"

    run_cmd(
        ["scripts/lf/profile_lora_lf.sh", "--model-specs", "Qwen/Qwen3-30B-A3B|1", "--output-root", str(output_root)],
        env={
            "LF_DIR": str(lf_dir),
            "BACKEND_SPECS": "asym_cpuadamwds|norecomp",
            "GPU_POOL": "0",
            "PROFILERS": "source",
            "WORKLOADS": "4096|4|1",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "ASYMM_EXP_ACT_POLICIES": "gc-layer|false|false|false,none|true|true|true",
            # Pin the dir-shaping knobs so the expected paths are deterministic.
            "ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD": "cpu",
            "ASYM_CPU_ADAMW_GRAD_OFFLOAD": "false",
            "ASYM_CPU_ADAMW_WEIGHT_OFFLOAD": "false",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
    )

    commands = {str(path): path.read_text(encoding="utf-8") for path in output_root.rglob("command.txt")}
    assert len(commands) == 2
    command_paths = "\n".join(commands)
    neutral = "__actrecomp0__xunpack0__ligerloss0__gradofffalse__weightofffalse/"
    assert f"__polgc-layer__routerwhole__expact0__attnact0__layeract0__loraafwdcpu{neutral}" in command_paths
    assert f"__polnone__routerwhole__expact1__attnact1__layeract1__loraafwdcpu{neutral}" in command_paths

    gc_layer_command = next(text for path, text in commands.items() if "__polgc-layer__" in path)
    assert "ASYM_EXPERT_RECOMPUTE_POLICY=gc-layer" in gc_layer_command
    assert "GRADIENT_CHECKPOINTING=false" in gc_layer_command
    assert "ASYM_GEMM_LF_CONFIG_LAYER_GC_ENABLED=true" in gc_layer_command
    assert "ASYMM_LAYER_ACT_OFFLOAD=false" in gc_layer_command

    layeract_command = next(
        text
        for path, text in commands.items()
        if f"__layeract1__loraafwdcpu{neutral}" in path
    )
    assert "ASYMM_EXPERT_ACT_OFFLOAD=true" in layeract_command
    assert "ASYMM_ATTN_ACT_OFFLOAD=true" in layeract_command
    assert "ASYMM_LAYER_ACT_OFFLOAD=true" in layeract_command
    assert "ASYM_EXPERT_RECOMPUTE_POLICY=none" in layeract_command
    assert "ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_ACT_OFFLOAD=true" in layeract_command
    assert "ASYM_GEMM_LF_CONFIG_LAYER_GC_ENABLED=false" in layeract_command


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
            "WORKLOADS": "128|8|1",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "ASYMM_EXP_ACT_POLICIES": "none|false|false|false",
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
            "WORKLOADS": "128|8|1",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "ASYMM_EXP_ACT_POLICIES": "none|false|false|false",
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
            "MODEL_SPECS": "meta-llama/Llama-4-Scout-17B-16E|1",
            "BACKEND_SPECS": "asym_cpuadamwds|norecomp",
            "GPU_POOL": "0",
            "PROFILERS": "source",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "ASYMM_EXP_ACT_POLICIES": "none|false|false|false",
            "ROUTER_MODES": "whole",
            # Pin the shape/offload knobs this test asserts instead of relying on script defaults.
            "WORKLOADS": "4096|4|1",
            "MAX_STEPS": "10",
            "WARMUP_STEPS": "5",
            "ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD": "cpu",
            "ASYM_CPU_ADAMW_GRAD_OFFLOAD": "false",
            "ASYM_CPU_ADAMW_WEIGHT_OFFLOAD": "false",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
    )

    command_files = sorted(output_root.rglob("command.txt"))
    assert len(command_files) == 1
    command_texts = [path.read_text(encoding="utf-8") for path in command_files]
    command_paths = "\n".join(str(path) for path in command_files)

    assert "llama-4-scout-17b-16e__gpus1__b4_s4096_ga1_w5_s10_r64_a16_drop000" in command_paths
    assert command_paths.count(
        "asym_cpuadamwds__source__norecomp__polnone__routerwhole__expact0__attnact0__layeract0__loraafwdcpu"
        "__actrecomp0__xunpack0__ligerloss0__gradofffalse__weightofffalse/b4_s4096_ga1"
    ) == 1
    for command in command_texts:
        assert "BACKEND=asym_cpuadamwds" in command
        assert "PROFILE_BACKEND_LABEL=asym_cpuadamwds" in command
        assert "USE_ASYM_CPU_ADAMW=true" in command
        assert "ASYM_CPU_ADAMW_BACKEND=deepspeed" in command
        assert f"DEEPSPEED_DIR={deepspeed_dir}" in command
        assert f"ASYM_GEMM_LF_CONFIG_DEEPSPEED_DIR={deepspeed_dir}" in command
        assert "ASYM_OFFLOAD_MODULES=all" in command
        # Single PROFILE_MEMORY_BREAKDOWN knob (default true), no per-backend override: the CPUAdamW
        # backend gets memory profiling on like every other backend. PROFILE_MEMORY_ATTRIBUTION is no
        # longer a separate arg/env (the runner derives the attribution toggle from breakdown).
        assert "PROFILE_MEMORY_BREAKDOWN=true" in command
        assert "PROFILE_MEMORY_ATTRIBUTION" not in command
        assert "PER_DEVICE_TRAIN_BATCH_SIZE=4" in command
        assert "CUTOFF_LEN=4096" in command


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
            "WORKLOADS": "128|1|1",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "ASYMM_EXP_ACT_POLICIES": "none|false|false|false",
            "ROUTER_MODES": "whole",
            "ASYM_OFFLOAD_MODULES": offload_modules,
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
    )

    command_files = sorted(output_root.rglob("command.txt"))
    assert len(command_files) == 3
    command_paths = "\n".join(str(path) for path in command_files)
    assert "qwen3-30b-a3b__gpus1__b1_s128_ga1_w0_s1_r64_a16_drop000" in command_paths
    assert "qwen3_5-122b-a10b__gpus1__b1_s128_ga1_w0_s1_r64_a16_drop000" in command_paths
    assert "llama-4-scout-17b-16e__gpus1__b1_s128_ga1_w0_s1_r64_a16_drop000" in command_paths

    command_offload_modules = offload_modules.replace(",", "\\,")
    for path in command_files:
        command = path.read_text(encoding="utf-8")
        assert "BACKEND=asym_cpuadamwds" in command
        assert "PROFILE_BACKEND_LABEL=asym_cpuadamwds" in command
        assert "USE_ASYM_CPU_ADAMW=true" in command
        assert "ASYM_CPU_ADAMW_BACKEND=deepspeed" in command
        assert f"ASYM_GEMM_LF_CONFIG_DEEPSPEED_DIR={deepspeed_dir}" in command
        assert f"ASYM_OFFLOAD_MODULES={command_offload_modules}" in command
