from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_cmd(args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    merged_env.update(
        {
            "ROOT": str(ROOT),
            "ASYM_DIR": str(ROOT),
        }
    )
    if env:
        merged_env.update(env)
    return subprocess.run(args, cwd=ROOT, env=merged_env, text=True, capture_output=True, check=True)


def make_fake_lf(tmp_path: Path) -> Path:
    lf_dir = tmp_path / "lf"
    lf_dir.joinpath("data").mkdir(parents=True)
    lf_dir.joinpath("src").mkdir()
    lf_dir.joinpath("examples/deepspeed").mkdir(parents=True)
    lf_dir.joinpath("data/dummy.jsonl").write_text('{"messages":[]}\n', encoding="utf-8")
    lf_dir.joinpath("src/train.py").write_text("", encoding="utf-8")
    for name in ("ds_z2_config.json", "ds_z3_config.json", "ds_z3_offload_config.json"):
        config = {
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu"},
                "offload_param": {"device": "cpu"},
            }
        }
        lf_dir.joinpath("examples/deepspeed", name).write_text(json.dumps(config) + "\n", encoding="utf-8")
    super_config = {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "super_offload": True, "cpuadam_cores_perc": 0.8},
            "offload_param": {"device": "cpu"},
        }
    }
    lf_dir.joinpath("examples/deepspeed/ds_z3_superoffload_config.json").write_text(
        json.dumps(super_config) + "\n",
        encoding="utf-8",
    )
    return lf_dir


def make_fake_deepspeed(tmp_path: Path) -> Path:
    deepspeed_dir = tmp_path / "deepspeed"
    deepspeed_dir.joinpath("deepspeed/runtime/superoffload").mkdir(parents=True)
    deepspeed_dir.joinpath("deepspeed/runtime/superoffload/superoffload_stage3.py").write_text(
        "# fake SuperOffload module for launcher validation\n",
        encoding="utf-8",
    )
    return deepspeed_dir


def test_check_superoffload_run_accepts_profile_and_log_marker(tmp_path: Path) -> None:
    profile = tmp_path / "source_profile.json"
    log = tmp_path / "train.log"
    profile.write_text(json.dumps({"superoffload": {"config_super_offload": True}}), encoding="utf-8")
    log.write_text("DeepSpeed Final Optimizer = SuperOffloadOptimizer_Stage3\n", encoding="utf-8")

    result = run_cmd(
        [
            sys.executable,
            "scripts/lf/check_superoffload_run.py",
            "--profile-json",
            str(profile),
            "--train-log",
            str(log),
            "--require-enabled",
        ]
    )

    diagnostic = json.loads(result.stdout)
    assert diagnostic["enabled"] is True
    assert diagnostic["marker_source"] == "train_log"


def test_check_superoffload_run_accepts_profile_optimizer_class(tmp_path: Path) -> None:
    profile = tmp_path / "source_profile.json"
    log = tmp_path / "train.log"
    profile.write_text(json.dumps({"superoffload": {"optimizer_class": "SuperOffloadOptimizer_Stage3"}}), encoding="utf-8")
    log.write_text("", encoding="utf-8")

    result = run_cmd(
        [
            sys.executable,
            "scripts/lf/check_superoffload_run.py",
            "--profile-json",
            str(profile),
            "--train-log",
            str(log),
            "--require-enabled",
        ]
    )

    diagnostic = json.loads(result.stdout)
    assert diagnostic["enabled"] is True
    assert diagnostic["marker_source"] == "profile"


def test_check_superoffload_run_rejects_missing_marker(tmp_path: Path) -> None:
    profile = tmp_path / "source_profile.json"
    log = tmp_path / "train.log"
    profile.write_text(json.dumps({"superoffload": {"enabled": False}}), encoding="utf-8")
    log.write_text("", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/lf/check_superoffload_run.py",
            "--profile-json",
            str(profile),
            "--train-log",
            str(log),
            "--require-enabled",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert json.loads(result.stdout)["enabled"] is False


def test_profile_lora_lf_dry_run_accepts_superoffload(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    deepspeed_dir = make_fake_deepspeed(tmp_path)
    output_root = tmp_path / "dryrun"

    run_cmd(
        [
            "scripts/lf/profile_lora_lf.sh",
            "--model-specs",
            "Qwen/Qwen3-30B-A3B|2",
            "--output-root",
            str(output_root),
        ],
        env={
            "LF_DIR": str(lf_dir),
            "DEEPSPEED_DIR": str(deepspeed_dir),
            "BACKEND_SPECS": "superoffload|norecompute",
            "GPU_POOL": "0,1",
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

    static_config = lf_dir / "examples/deepspeed/ds_z3_superoffload_config.json"
    assert json.loads(static_config.read_text(encoding="utf-8"))["zero_optimization"]["offload_optimizer"][
        "super_offload"
    ] is True
    jobs = list(output_root.rglob("jobs.tsv"))
    assert jobs
    assert "superoffload" in jobs[0].read_text(encoding="utf-8")
    command_files = list(output_root.rglob("command.txt"))
    assert command_files
    command = command_files[0].read_text(encoding="utf-8")
    assert "BACKEND=superoffload" in command
    assert "SUPER_OFFLOAD_DEEPSPEED_CONFIG=" not in command
    assert "--deepspeed" not in command
    assert "--use_asym_gemm" not in command
    assert "--asym_backend" not in command
    assert "--use_kt" not in command
    assert "--kt_backend" not in command


def test_profile_lora_lf_rejects_legacy_superoffload_alias(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    deepspeed_dir = make_fake_deepspeed(tmp_path)
    result = subprocess.run(
        [
            "scripts/lf/profile_lora_lf.sh",
            "--model-specs",
            "Qwen/Qwen3-30B-A3B|1",
            "--output-root",
            str(tmp_path / "dryrun"),
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "ROOT": str(ROOT),
            "ASYM_DIR": str(ROOT),
            "LF_DIR": str(lf_dir),
            "DEEPSPEED_DIR": str(deepspeed_dir),
            "BACKEND_SPECS": "ds_superoffload|norecomp",
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
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "backend must be" in result.stderr


def test_run_lf_lora_sft_uses_deepspeed_for_single_gpu_zero3_offload(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    fake_env = tmp_path / "env"
    fake_env.joinpath("bin").mkdir(parents=True)
    fake_python = fake_env / "bin/python"
    torchrun_log = tmp_path / "torchrun_args.txt"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "if [[ \"${1-}\" == \"-m\" && \"${2-}\" == \"torch.distributed.run\" ]]; then",
                "  shift 2",
                "  printf '%s\\n' \"$@\" > \"$FAKE_TORCHRUN_LOG\"",
                "  out_dir=",
                "  prev=",
                "  for arg in \"$@\"; do",
                "    if [[ \"$prev\" == \"--output_dir\" ]]; then",
                "      out_dir=\"$arg\"",
                "      break",
                "    fi",
                "    prev=\"$arg\"",
                "  done",
                "  if [[ -n \"$out_dir\" ]]; then",
                "    mkdir -p \"$out_dir\"",
                "    printf '%s\\n' '{\"loss\":1.0}' > \"$out_dir/trainer_log.jsonl\"",
                "  fi",
                "  exit 0",
                "fi",
                "cat >/dev/null",
                "exit 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    ds_config = lf_dir / "examples/deepspeed/ds_z3_offload_config.json"
    run_cmd(
        ["scripts/lf/run_lf_lora_sft.sh"],
        env={
            "LF_DIR": str(lf_dir),
            "ENV_DIR": str(fake_env),
            "ENV_PYTHON": str(fake_python),
            "FAKE_TORCHRUN_LOG": str(torchrun_log),
            "BACKEND": "zero3_offload",
            "GPU_ID": "0",
            "NUM_GPUS": "1",
            "PROFILE": "0",
            "DATASET": "dummy",
            "TEMPLATE": "qwen3_nothink",
            "CUTOFF_LEN": "128",
            "MAX_SAMPLES": "1",
            "MAX_STEPS": "1",
            "LORA_RANK": "8",
            "LORA_ALPHA": "16",
            "LORA_DROPOUT": "0.00",
            "TORCH_USE_ASYM_GEMM_LORA": "false",
        },
    )

    args = torchrun_log.read_text(encoding="utf-8").splitlines()
    assert "--nproc_per_node" in args
    assert args[args.index("--nproc_per_node") + 1] == "1"
    assert "--deepspeed" in args
    assert args[args.index("--deepspeed") + 1] == str(ds_config)


def test_run_lf_lora_sft_uses_deepspeed_for_single_gpu_superoffload(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    deepspeed_dir = make_fake_deepspeed(tmp_path)
    fake_env = tmp_path / "env"
    fake_env.joinpath("bin").mkdir(parents=True)
    fake_python = fake_env / "bin/python"
    torchrun_log = tmp_path / "torchrun_args.txt"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "if [[ \"${1-}\" == \"-m\" && \"${2-}\" == \"torch.distributed.run\" ]]; then",
                "  shift 2",
                "  printf '%s\\n' \"$@\" > \"$FAKE_TORCHRUN_LOG\"",
                "  out_dir=",
                "  prev=",
                "  for arg in \"$@\"; do",
                "    if [[ \"$prev\" == \"--output_dir\" ]]; then",
                "      out_dir=\"$arg\"",
                "      break",
                "    fi",
                "    prev=\"$arg\"",
                "  done",
                "  if [[ -n \"$out_dir\" ]]; then",
                "    mkdir -p \"$out_dir\"",
                "    printf '%s\\n' '{\"loss\":1.0}' > \"$out_dir/trainer_log.jsonl\"",
                "  fi",
                "  echo 'DeepSpeed Final Optimizer = SuperOffloadOptimizer_Stage3'",
                "  exit 0",
                "fi",
                "if [[ \"${1-}\" == *check_superoffload_run.py ]]; then",
                "  exec /usr/bin/python3 \"$@\"",
                "fi",
                "cat >/dev/null",
                "exit 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    super_config = lf_dir / "examples/deepspeed/ds_z3_superoffload_config.json"

    run_cmd(
        ["scripts/lf/run_lf_lora_sft.sh"],
        env={
            "LF_DIR": str(lf_dir),
            "DEEPSPEED_DIR": str(deepspeed_dir),
            "ENV_DIR": str(fake_env),
            "ENV_PYTHON": str(fake_python),
            "FAKE_TORCHRUN_LOG": str(torchrun_log),
            "BACKEND": "superoffload",
            "CHECK_SUPEROFFLOAD": "1",
            "GPU_ID": "0",
            "NUM_GPUS": "1",
            "PROFILE": "0",
            "DATASET": "dummy",
            "TEMPLATE": "qwen3_nothink",
            "CUTOFF_LEN": "128",
            "MAX_SAMPLES": "1",
            "MAX_STEPS": "1",
            "LORA_RANK": "8",
            "LORA_ALPHA": "16",
            "LORA_DROPOUT": "0.00",
            "TORCH_USE_ASYM_GEMM_LORA": "false",
        },
    )

    args = torchrun_log.read_text(encoding="utf-8").splitlines()
    assert "--nproc_per_node" in args
    assert args[args.index("--nproc_per_node") + 1] == "1"
    assert "--deepspeed" in args
    assert args[args.index("--deepspeed") + 1] == str(super_config)
    assert "--use_asym_gemm" not in args
    assert "--asym_backend" not in args
    assert "--use_kt" not in args
    assert "--kt_backend" not in args


def test_profile_lora_lf_skips_expert_policy_for_superoffload(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    deepspeed_dir = make_fake_deepspeed(tmp_path)
    output_root = tmp_path / "dryrun"

    run_cmd(
        [
            "scripts/lf/profile_lora_lf.sh",
            "--model-specs",
            "Qwen/Qwen3-30B-A3B|2",
            "--output-root",
            str(output_root),
        ],
        env={
            "LF_DIR": str(lf_dir),
            "DEEPSPEED_DIR": str(deepspeed_dir),
            "BACKEND_SPECS": "superoffload|norecomp",
            "GPU_POOL": "0,1",
            "PROFILERS": "source",
            "SEQ_LENS": "128",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "EXPERT_POLICIES": "none,tok-le64",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
    )

    command_files = list(output_root.rglob("command.txt"))
    assert len(command_files) == 1
    rows = []
    for path in output_root.rglob("jobs.tsv"):
        lines = path.read_text(encoding="utf-8").splitlines()
        rows.extend(line.split("\t") for line in lines[1:] if line.strip())
    assert sum(1 for row in rows if len(row) > 6 and row[6] == "superoffload") == 1


def test_profile_lora_lf_keeps_superoffload_recompute_modes(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    deepspeed_dir = make_fake_deepspeed(tmp_path)
    output_root = tmp_path / "dryrun"

    run_cmd(
        [
            "scripts/lf/profile_lora_lf.sh",
            "--model-specs",
            "Qwen/Qwen3-30B-A3B|1",
            "--output-root",
            str(output_root),
        ],
        env={
            "LF_DIR": str(lf_dir),
            "DEEPSPEED_DIR": str(deepspeed_dir),
            "BACKEND_SPECS": "superoffload|both",
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
    assert len(command_files) == 2
    command_paths = "\n".join(str(path) for path in command_files)
    assert "superoffload__source__norecomp__polnone" in command_paths
    assert "superoffload__source__recomp__polnone" in command_paths
