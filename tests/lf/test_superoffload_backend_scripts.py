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
    cpuadam_config = {
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto",
                "torch_adam": False,
                "adam_w_mode": True,
                "fp32_optimizer_states": True,
            },
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
        },
    }
    lf_dir.joinpath("examples/deepspeed/ds_z3_cpuadam_config.json").write_text(
        json.dumps(cpuadam_config) + "\n",
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


def test_check_deepspeed_cpuadam_run_accepts_profile_and_log_marker(tmp_path: Path) -> None:
    profile = tmp_path / "source_profile.json"
    log = tmp_path / "train.log"
    profile.write_text(
        json.dumps(
            {
                "cpuadam": {
                    "config_optimizer_type": "AdamW",
                    "config_offload_optimizer_device": "cpu",
                }
            }
        ),
        encoding="utf-8",
    )
    log.write_text("DeepSpeed Basic Optimizer = DeepSpeedCPUAdam\n", encoding="utf-8")

    result = run_cmd(
        [
            sys.executable,
            "scripts/lf/check_deepspeed_cpuadam_run.py",
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


def test_check_deepspeed_cpuadam_run_accepts_profile_basic_optimizer_class(tmp_path: Path) -> None:
    profile = tmp_path / "source_profile.json"
    log = tmp_path / "train.log"
    profile.write_text(
        json.dumps({"source_profile": {"cpuadam": {"basic_optimizer_class": "DeepSpeedCPUAdam"}}}),
        encoding="utf-8",
    )
    log.write_text("", encoding="utf-8")

    result = run_cmd(
        [
            sys.executable,
            "scripts/lf/check_deepspeed_cpuadam_run.py",
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


def test_check_deepspeed_cpuadam_run_rejects_missing_marker(tmp_path: Path) -> None:
    profile = tmp_path / "source_profile.json"
    log = tmp_path / "train.log"
    profile.write_text(json.dumps({"cpuadam": {"enabled": False}}), encoding="utf-8")
    log.write_text("", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/lf/check_deepspeed_cpuadam_run.py",
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


def _run_lf_launcher_expect_failure(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    merged_env.update({"ROOT": str(ROOT), "ASYM_DIR": str(ROOT), **env})
    return subprocess.run(
        ["bash", "scripts/lf/run_lf_lora_sft.sh"],
        cwd=ROOT,
        env=merged_env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_run_lf_lora_sft_rejects_kt_arm_nsys_without_source_ok() -> None:
    result = _run_lf_launcher_expect_failure(
        {"BACKEND": "kt_armbf16", "PROFILE": "1", "PROFILE_PROFILER": "nsys"}
    )

    assert result.returncode == 2
    assert "requires PROFILE_PROFILER=source before nsys" in result.stderr


def test_run_lf_lora_sft_rejects_kt_arm_nsys_legacy_override_without_source_profile() -> None:
    result = _run_lf_launcher_expect_failure(
        {
            "BACKEND": "kt_armbf16",
            "PROFILE": "1",
            "PROFILE_PROFILER": "nsys",
            "KT_ARM_ALLOW_NSYS_WITHOUT_SOURCE_OK": "1",
        }
    )

    assert result.returncode == 2
    assert "KT_ARM_SOURCE_OK_PROFILE_JSON" in result.stderr
    assert "KT_ARM_ALLOW_RAW_NSYS_WITHOUT_SOURCE_OK=1" in result.stderr


def test_run_lf_lora_sft_rejects_kt_arm_nsys_source_ok_without_final_heartbeat(tmp_path: Path) -> None:
    source_profile = tmp_path / "source_profile.json"
    source_profile.write_text(
        json.dumps(
            {
                "config": {
                    "backend": "kt_armbf16",
                    "model_name_or_path": "Qwen/Qwen3-30B-A3B",
                    "seq_len": 64,
                    "per_device_train_batch_size": 1,
                    "lora_rank": 8,
                    "lora_dropout": 0.0,
                    "lora_target": "all",
                    "kt_arm_sft_top_k": 8,
                    "kt_arm_effective_route_qlen": 64,
                    "kt_arm_token_chunks": 1,
                    "kt_arm_route_rank_work": 512,
                    "kt_arm_sft_max_route_rank_work": 1048576,
                    "activation_recompute": False,
                    "kt_max_cache_depth": 2,
                },
                "heartbeat": {"latest": {"stage": "trainer_end"}},
                "kt": {"wrapper_count": 48, "total_forward_calls": 96, "total_backward_calls": 48},
                "lora": {"kt_fused_expert_lora_parameters": 8},
                "optimizer_memory": {
                    "kt_lora_update_health": {
                        "available": True,
                        "passed": True,
                        "sampled_tensors": 1,
                        "total_fused_tensors": 1,
                        "after_sampled_tensors": 1,
                        "after_total_fused_tensors": 1,
                        "compared_tensors": 1,
                        "grad_nonzero_tensors": 1,
                        "updated_grad_tensors": 1,
                        "grad_nonzero_unchanged_tensors": 0,
                        "missing_after_tensors": 0,
                        "unexpected_after_tensors": 0,
                        "rows": [{"grad_nonzero_before_step": True, "param_changed_after_step": True}],
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_lf_launcher_expect_failure(
        {
            "BACKEND": "kt_armbf16",
            "PROFILE": "1",
            "PROFILE_PROFILER": "nsys",
            "KT_ARM_SOURCE_OK_PROFILE_JSON": str(source_profile),
            "MODEL_NAME_OR_PATH": "Qwen/Qwen3-30B-A3B",
            "CUTOFF_LEN": "64",
            "PER_DEVICE_TRAIN_BATCH_SIZE": "1",
            "LORA_RANK": "8",
            "LORA_DROPOUT": "0.0",
        }
    )

    assert result.returncode == 2
    assert "source-ok profile heartbeat is not final" in result.stderr


def test_run_lf_lora_sft_rejects_partial_profile_as_kt_arm_source_ok(tmp_path: Path) -> None:
    partial_profile = tmp_path / "partial_profile.json"
    partial_profile.write_text(
        json.dumps({"heartbeat": {"latest": {"stage": "source_profile_written"}}}) + "\n",
        encoding="utf-8",
    )

    result = _run_lf_launcher_expect_failure(
        {
            "BACKEND": "kt_armbf16",
            "PROFILE": "1",
            "PROFILE_PROFILER": "nsys",
            "KT_ARM_SOURCE_OK_PROFILE_JSON": str(partial_profile),
        }
    )

    assert result.returncode == 2
    assert "must point at a completed source artifact" in result.stderr


def test_run_lf_lora_sft_rejects_kt_arm_tensor_parallel() -> None:
    result = _run_lf_launcher_expect_failure({"BACKEND": "kt_armbf16", "KT_TP_ENABLED": "true"})

    assert result.returncode == 2
    assert "does not support KT_TP_ENABLED=true" in result.stderr


def test_run_lf_lora_sft_rejects_kt_arm_threadpool_count() -> None:
    result = _run_lf_launcher_expect_failure({"BACKEND": "kt_armbf16", "KT_THREADPOOL_COUNT": "2"})

    assert result.returncode == 2
    assert "requires KT_THREADPOOL_COUNT empty or 1" in result.stderr


def test_run_lf_lora_sft_rejects_kt_arm_gpu_experts() -> None:
    result = _run_lf_launcher_expect_failure({"BACKEND": "kt_armbf16", "KT_NUM_GPU_EXPERTS": "1"})

    assert result.returncode == 2
    assert "does not support GPU experts" in result.stderr


def test_run_lf_lora_sft_rejects_kt_arm_gradient_accumulation() -> None:
    result = _run_lf_launcher_expect_failure({"BACKEND": "kt_armbf16", "GRADIENT_ACCUMULATION_STEPS": "2"})

    assert result.returncode == 2
    assert "requires GRADIENT_ACCUMULATION_STEPS=1" in result.stderr


def test_run_lf_lora_sft_allows_kt_arm_max_grad_norm_before_route_rank_guard() -> None:
    result = _run_lf_launcher_expect_failure(
        {
            "BACKEND": "kt_armbf16",
            "MAX_GRAD_NORM": "1.0",
            "KT_ARM_SFT_MAX_ROUTE_RANK_WORK": "1",
        }
    )

    assert result.returncode == 2
    assert "route-rank work" in result.stderr
    assert "MAX_GRAD_NORM=0" not in result.stderr


def test_run_lf_lora_sft_rejects_kt_arm_route_rank_limit() -> None:
    result = _run_lf_launcher_expect_failure(
        {"BACKEND": "kt_armbf16", "KT_ARM_SFT_MAX_ROUTE_RANK_WORK": "1"}
    )

    assert result.returncode == 2
    assert "route-rank work" in result.stderr


def test_run_lf_lora_sft_rejects_kt_arm_default_unvalidated_large_route_rank() -> None:
    result = _run_lf_launcher_expect_failure(
        {
            "BACKEND": "kt_armbf16",
            "PER_DEVICE_TRAIN_BATCH_SIZE": "4",
            "CUTOFF_LEN": "7168",
            "LORA_RANK": "64",
        }
    )

    assert result.returncode == 2
    assert "route-rank work 14680064 exceeds KT_ARM_SFT_MAX_ROUTE_RANK_WORK=1048576" in result.stderr
    assert "KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK=1" in result.stderr


def test_run_lf_lora_sft_preserves_partial_source_profile_without_canonical_profile(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    fake_env = tmp_path / "env"
    fake_env.joinpath("bin").mkdir(parents=True)
    fake_python = fake_env / "bin/python"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "if [[ \"${1-}\" == *postprocess_lf_profile_artifacts.py ]]; then",
                "  exec /usr/bin/python3 \"$@\"",
                "fi",
                "if [[ \"${1-}\" == *run_lf_profiled_train.py ]] || [[ \"${1-}\" == \"-m\" && \"${2-}\" == \"torch.distributed.run\" ]]; then",
                "  partial=\"$(dirname \"$ASYM_GEMM_LF_PROFILE_SOURCE_JSON\")/source_profile.partial.json\"",
                "  mkdir -p \"$(dirname \"$partial\")\"",
                "  cat > \"$partial\" <<'JSON'",
                "{\"workload\":\"unit\",\"config\":{\"backend\":\"torch\",\"precision\":\"bf16\",\"seq_len\":128,\"warmup_steps\":0,\"measure_steps\":1},\"memory\":{\"gpu\":{}},\"trainer\":{},\"forward\":{\"total_milliseconds\":0},\"backward\":{\"total_milliseconds\":0},\"stage_memory\":{\"rows\":[]},\"kt\":{\"available\":false,\"reason\":\"not kt\"},\"lora\":{\"available\":false,\"reason\":\"failure before model capture\"}}",
                "JSON",
                "  exit 7",
                "fi",
                "cat >/dev/null",
                "exit 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "profile.json").write_text('{"stale": true}\n', encoding="utf-8")

    result = subprocess.run(
        ["bash", "scripts/lf/run_lf_lora_sft.sh"],
        cwd=ROOT,
        env={
            **os.environ,
            "ROOT": str(ROOT),
            "ASYM_DIR": str(ROOT),
            "LF_DIR": str(lf_dir),
            "ENV_DIR": str(fake_env),
            "ENV_PYTHON": str(fake_python),
            "BACKEND": "torch",
            "GPU_ID": "0",
            "NUM_GPUS": "1",
            "PROFILE": "1",
            "PROFILE_PROFILER": "source",
            "PROFILE_OUTPUT_DIR": str(out_dir),
            "PROFILE_SOURCE_JSON": str(out_dir / "lf_run/source_profile.json"),
            "PROFILE_JSON": str(out_dir / "lf_run/profile.json"),
            "OUT_DIR": str(out_dir / "lf_run"),
            "LOG_FILE": str(out_dir / "train.log"),
            "DATASET": "dummy",
            "TEMPLATE": "qwen3_nothink",
            "MODEL_NAME_OR_PATH": "unit/unit",
            "CUTOFF_LEN": "128",
            "MAX_SAMPLES": "1",
            "MAX_STEPS": "1",
            "LORA_RANK": "8",
            "LORA_ALPHA": "16",
            "LORA_DROPOUT": "0.00",
            "CHECK_TRAINABLE_SURFACE": "0",
            "TORCH_USE_ASYM_GEMM_LORA": "false",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 7
    assert not (out_dir / "profile.json").exists()
    assert (out_dir / "partial_profile.json").is_file()
    assert (out_dir / "summary.md").is_file()
    log_text = (out_dir / "train.log").read_text(encoding="utf-8")
    assert "postprocessing partial profile" in log_text
    assert "canonical profile.json was not created" in log_text


def test_run_lf_lora_sft_failed_final_source_profile_is_not_canonical(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    fake_env = tmp_path / "env"
    fake_env.joinpath("bin").mkdir(parents=True)
    fake_python = fake_env / "bin/python"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "if [[ \"${1-}\" == *postprocess_lf_profile_artifacts.py ]]; then",
                "  exec /usr/bin/python3 \"$@\"",
                "fi",
                "if [[ \"${1-}\" == *run_lf_profiled_train.py ]] || [[ \"${1-}\" == \"-m\" && \"${2-}\" == \"torch.distributed.run\" ]]; then",
                "  mkdir -p \"$(dirname \"$ASYM_GEMM_LF_PROFILE_SOURCE_JSON\")\"",
                "  cat > \"$ASYM_GEMM_LF_PROFILE_SOURCE_JSON\" <<'JSON'",
                "{\"workload\":\"unit\",\"config\":{\"backend\":\"torch\",\"precision\":\"bf16\",\"seq_len\":128,\"warmup_steps\":0,\"measure_steps\":1},\"memory\":{\"gpu\":{}},\"trainer\":{},\"forward\":{\"total_milliseconds\":0},\"backward\":{\"total_milliseconds\":0},\"stage_memory\":{\"rows\":[]},\"kt\":{\"available\":false,\"reason\":\"not kt\"},\"lora\":{\"available\":false,\"reason\":\"failed after final write\"}}",
                "JSON",
                "  exit 7",
                "fi",
                "cat >/dev/null",
                "exit 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    out_dir = tmp_path / "out"

    result = subprocess.run(
        ["bash", "scripts/lf/run_lf_lora_sft.sh"],
        cwd=ROOT,
        env={
            **os.environ,
            "ROOT": str(ROOT),
            "ASYM_DIR": str(ROOT),
            "LF_DIR": str(lf_dir),
            "ENV_DIR": str(fake_env),
            "ENV_PYTHON": str(fake_python),
            "BACKEND": "torch",
            "GPU_ID": "0",
            "NUM_GPUS": "1",
            "PROFILE": "1",
            "PROFILE_PROFILER": "source",
            "PROFILE_OUTPUT_DIR": str(out_dir),
            "PROFILE_SOURCE_JSON": str(out_dir / "source_profile.json"),
            "PROFILE_JSON": str(out_dir / "profile.json"),
            "OUT_DIR": str(out_dir / "lf_run"),
            "LOG_FILE": str(out_dir / "train.log"),
            "DATASET": "dummy",
            "TEMPLATE": "qwen3_nothink",
            "MODEL_NAME_OR_PATH": "unit/unit",
            "CUTOFF_LEN": "128",
            "MAX_SAMPLES": "1",
            "MAX_STEPS": "1",
            "LORA_RANK": "8",
            "LORA_ALPHA": "16",
            "LORA_DROPOUT": "0.00",
            "CHECK_TRAINABLE_SURFACE": "0",
            "TORCH_USE_ASYM_GEMM_LORA": "false",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 7
    assert not (out_dir / "profile.json").exists()
    assert (out_dir / "partial_profile.json").is_file()
    log_text = (out_dir / "train.log").read_text(encoding="utf-8")
    assert "Training failed; postprocessing source profile as partial artifact" in log_text


def test_run_lf_lora_sft_kt_arm_first_step_watchdog_preserves_partial_profile(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    fake_env = tmp_path / "env"
    fake_env.joinpath("bin").mkdir(parents=True)
    fake_python = fake_env / "bin/python"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "if [[ \"${1-}\" == *postprocess_lf_profile_artifacts.py ]]; then",
                "  exec /usr/bin/python3 \"$@\"",
                "fi",
                "if [[ \"${1-}\" == *run_lf_profiled_train.py ]] || [[ \"${1-}\" == \"-m\" && \"${2-}\" == \"torch.distributed.run\" ]]; then",
                "  partial=\"$(dirname \"$ASYM_GEMM_LF_PROFILE_SOURCE_JSON\")/source_profile.partial.json\"",
                "  latest=\"${ASYM_GEMM_LF_HEARTBEAT_JSON%.*}.latest.json\"",
                "  mkdir -p \"$(dirname \"$partial\")\"",
                "  printf '%s\\n' '{\"stage\":\"trainer_start\"}' > \"$latest\"",
                "  cat > \"$partial\" <<'JSON'",
                "{\"workload\":\"unit\",\"config\":{\"backend\":\"kt_armbf16\",\"kt_backend\":\"ARMBF16\",\"precision\":\"bf16\",\"seq_len\":128,\"warmup_steps\":0,\"measure_steps\":1},\"memory\":{\"gpu\":{}},\"trainer\":{},\"forward\":{\"total_milliseconds\":0},\"backward\":{\"total_milliseconds\":0},\"stage_memory\":{\"rows\":[]},\"kt\":{\"available\":false,\"reason\":\"watchdog\"},\"lora\":{\"available\":false,\"reason\":\"watchdog\"}}",
                "JSON",
                "  sleep 30",
                "  exit 127",
                "fi",
                "cat >/dev/null",
                "exit 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    out_dir = tmp_path / "out"

    result = subprocess.run(
        ["bash", "scripts/lf/run_lf_lora_sft.sh"],
        cwd=ROOT,
        env={
            **os.environ,
            "ROOT": str(ROOT),
            "ASYM_DIR": str(ROOT),
            "LF_DIR": str(lf_dir),
            "ENV_DIR": str(fake_env),
            "ENV_PYTHON": str(fake_python),
            "BACKEND": "kt_armbf16",
            "GPU_ID": "0",
            "NUM_GPUS": "1",
            "NUMACTL_ENABLE": "0",
            "PROFILE": "1",
            "PROFILE_PROFILER": "source",
            "PROFILE_OUTPUT_DIR": str(out_dir),
            "PROFILE_SOURCE_JSON": str(out_dir / "source_profile.json"),
            "PROFILE_HEARTBEAT_JSON": str(out_dir / "heartbeat.jsonl"),
            "PROFILE_JSON": str(out_dir / "profile.json"),
            "OUT_DIR": str(out_dir / "lf_run"),
            "LOG_FILE": str(out_dir / "train.log"),
            "DATASET": "dummy",
            "TEMPLATE": "qwen3_nothink",
            "MODEL_NAME_OR_PATH": "unit/unit",
            "CUTOFF_LEN": "128",
            "MAX_SAMPLES": "1",
            "MAX_STEPS": "1",
            "LORA_RANK": "8",
            "LORA_ALPHA": "16",
            "LORA_DROPOUT": "0.00",
            "CHECK_TRAINABLE_SURFACE": "0",
            "CHECK_KT_CALLS": "0",
            "KT_ARM_FIRST_STEP_TIMEOUT_SECONDS": "1",
            "INTERRUPT_GRACE_SECONDS": "0",
            "TORCH_USE_ASYM_GEMM_LORA": "false",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 124
    assert not (out_dir / "profile.json").exists()
    assert (out_dir / "partial_profile.json").is_file()
    log_text = (out_dir / "train.log").read_text(encoding="utf-8")
    assert "did not complete the first optimizer step" in log_text
    assert "postprocessing partial profile" in log_text
    assert "canonical profile.json was not created" in log_text


def test_run_lf_lora_sft_kt_arm_watchdog_allows_completed_source_profile(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    fake_env = tmp_path / "env"
    fake_env.joinpath("bin").mkdir(parents=True)
    fake_python = fake_env / "bin/python"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "if [[ \"${1-}\" == *postprocess_lf_profile_artifacts.py ]]; then",
                "  exec /usr/bin/python3 \"$@\"",
                "fi",
                "if [[ \"${1-}\" == \"-\" ]]; then",
                "  exec /usr/bin/python3 \"$@\"",
                "fi",
                "if [[ \"${1-}\" == *run_lf_profiled_train.py ]]; then",
                "  /usr/bin/python3 - <<'PY'",
                "import json, os, pathlib, time",
                "profile = pathlib.Path(os.environ['ASYM_GEMM_LF_PROFILE_SOURCE_JSON'])",
                "latest = pathlib.Path(os.environ['ASYM_GEMM_LF_HEARTBEAT_JSON']).with_suffix('.latest.json')",
                "profile.parent.mkdir(parents=True, exist_ok=True)",
                "latest.parent.mkdir(parents=True, exist_ok=True)",
                "payload = {",
                "    'workload': 'unit',",
                "    'config': {'backend': 'kt_armbf16', 'kt_backend': 'ARMBF16', 'precision': 'bf16', 'seq_len': 128, 'cutoff_len': 128, 'warmup_steps': 0, 'measure_steps': 1, 'model_name_or_path': 'unit/unit', 'batch_size': 1, 'per_device_train_batch_size': 1, 'logical_qlen': 128, 'lora_target': 'all', 'lora_rank': 8, 'lora_dropout': 0.0, 'kt_arm_sft_top_k': 8, 'kt_arm_effective_route_qlen': 128, 'kt_arm_token_chunks': 1, 'kt_arm_route_rank_work': 8192, 'kt_arm_sft_max_route_rank_work': 1048576, 'activation_recompute': False, 'kt_max_cache_depth': 2},",
                "    'memory': {'gpu': {}},",
                "    'trainer': {},",
                "    'forward': {'total_milliseconds': 1},",
                "    'backward': {'total_milliseconds': 1},",
                "    'stage_memory': {'rows': []},",
                "    'kt': {'available': True, 'wrapper_count': 1, 'total_forward_calls': 2, 'total_backward_calls': 1, 'rows': [{'method': 'ARMBF16_SFT'}]},",
                "    'lora': {'available': True, 'trainable_parameters': 8, 'peft_lora_parameters': 0, 'kt_expert_lora_parameters': 8, 'kt_peft_expert_lora_parameters': 0, 'kt_fused_expert_lora_parameters': 8},",
                "    'optimizer_memory_preflight': {'available': True},",
                "    'optimizer_memory': {'process_memory_at_start': {'rss_bytes': 100}, 'process_memory_before_step': {'rss_bytes': 110}, 'process_memory_after_step': {'rss_bytes': 210}, 'process_rss_pre_step_overhead_delta_bytes': 10, 'process_rss_delta_bytes': 100, 'kt_lora_update_health': {'available': True, 'passed': True, 'sampled_tensors': 1, 'total_fused_tensors': 1, 'after_sampled_tensors': 1, 'after_total_fused_tensors': 1, 'compared_tensors': 1, 'grad_nonzero_tensors': 1, 'updated_grad_tensors': 1, 'grad_nonzero_unchanged_tensors': 0, 'missing_after_tensors': 0, 'unexpected_after_tensors': 0, 'rows': [{'grad_nonzero_before_step': True, 'param_changed_after_step': True}]}},",
                "    'heartbeat': {'latest': {'stage': 'source_profile_written'}},",
                "}",
                "latest.write_text('{\"stage\":\"source_profile_written\"}\\n', encoding='utf-8')",
                "profile.write_text(json.dumps(payload) + '\\n', encoding='utf-8')",
                "time.sleep(1)",
                "PY",
                "  exit 127",
                "fi",
                "cat >/dev/null",
                "exit 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    out_dir = tmp_path / "out"

    result = subprocess.run(
        ["bash", "scripts/lf/run_lf_lora_sft.sh"],
        cwd=ROOT,
        env={
            **os.environ,
            "ROOT": str(ROOT),
            "ASYM_DIR": str(ROOT),
            "LF_DIR": str(lf_dir),
            "ENV_DIR": str(fake_env),
            "ENV_PYTHON": str(fake_python),
            "BACKEND": "kt_armbf16",
            "GPU_ID": "0",
            "NUM_GPUS": "1",
            "NUMACTL_ENABLE": "0",
            "PROFILE": "1",
            "PROFILE_PROFILER": "source",
            "PROFILE_OUTPUT_DIR": str(out_dir),
            "PROFILE_SOURCE_JSON": str(out_dir / "source_profile.json"),
            "PROFILE_HEARTBEAT_JSON": str(out_dir / "heartbeat.jsonl"),
            "PROFILE_JSON": str(out_dir / "profile.json"),
            "OUT_DIR": str(out_dir / "lf_run"),
            "LOG_FILE": str(out_dir / "train.log"),
            "DATASET": "dummy",
            "TEMPLATE": "qwen3_nothink",
            "MODEL_NAME_OR_PATH": "unit/unit",
            "CUTOFF_LEN": "128",
            "MAX_SAMPLES": "1",
            "MAX_STEPS": "1",
            "LORA_RANK": "8",
            "LORA_ALPHA": "16",
            "LORA_DROPOUT": "0.00",
            "CHECK_TRAINABLE_SURFACE": "0",
            "CHECK_KT_CALLS": "1",
            "KT_ARM_FIRST_STEP_TIMEOUT_SECONDS": "60",
            "TORCH_USE_ASYM_GEMM_LORA": "false",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (out_dir / "profile.json").is_file()
    log_text = (out_dir / "train.log").read_text(encoding="utf-8")
    assert "returned status 127" in log_text
    assert "Verified KT source counters" in log_text
    assert "Wrote source profile artifacts" in log_text


def test_run_lf_lora_sft_rejects_partial_source_profile_recovery(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    fake_env = tmp_path / "env"
    fake_env.joinpath("bin").mkdir(parents=True)
    fake_python = fake_env / "bin/python"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "if [[ \"${1-}\" == *postprocess_lf_profile_artifacts.py ]]; then",
                "  exec /usr/bin/python3 \"$@\"",
                "fi",
                "if [[ \"${1-}\" == \"-\" ]]; then",
                "  exec /usr/bin/python3 \"$@\"",
                "fi",
                "if [[ \"${1-}\" == *run_lf_profiled_train.py ]]; then",
                "  /usr/bin/python3 - <<'PY'",
                "import json, os, pathlib",
                "profile = pathlib.Path(os.environ['ASYM_GEMM_LF_PROFILE_SOURCE_JSON'])",
                "latest = pathlib.Path(os.environ['ASYM_GEMM_LF_HEARTBEAT_JSON']).with_suffix('.latest.json')",
                "profile.parent.mkdir(parents=True, exist_ok=True)",
                "latest.parent.mkdir(parents=True, exist_ok=True)",
                "payload = {",
                "    'partial': True,",
                "    'workload': 'unit',",
                "    'config': {'backend': 'kt_armbf16', 'kt_backend': 'ARMBF16', 'precision': 'bf16', 'seq_len': 128, 'cutoff_len': 128, 'model_name_or_path': 'unit/unit', 'batch_size': 1, 'per_device_train_batch_size': 1, 'logical_qlen': 128, 'lora_target': 'all', 'lora_rank': 8, 'lora_dropout': 0.0, 'kt_arm_sft_top_k': 8, 'kt_arm_effective_route_qlen': 128, 'kt_arm_token_chunks': 1, 'kt_arm_route_rank_work': 8192, 'kt_arm_sft_max_route_rank_work': 1048576, 'activation_recompute': False, 'kt_max_cache_depth': 2},",
                "    'memory': {'gpu': {}},",
                "    'trainer': {},",
                "    'forward': {'total_milliseconds': 1},",
                "    'backward': {'total_milliseconds': 1},",
                "    'stage_memory': {'rows': []},",
                "    'kt': {'available': True, 'wrapper_count': 1, 'total_forward_calls': 1, 'total_backward_calls': 1, 'rows': [{'method': 'ARMBF16_SFT'}]},",
                "    'lora': {'available': True, 'trainable_parameters': 8, 'kt_fused_expert_lora_parameters': 8},",
                "    'optimizer_memory_preflight': {'available': True},",
                "    'optimizer_memory': {'process_memory_at_start': {'rss_bytes': 100}, 'process_memory_before_step': {'rss_bytes': 110}, 'process_memory_after_step': {'rss_bytes': 210}, 'process_rss_pre_step_overhead_delta_bytes': 10, 'process_rss_delta_bytes': 100, 'kt_lora_update_health': {'available': True, 'passed': True, 'sampled_tensors': 1, 'total_fused_tensors': 1, 'after_sampled_tensors': 1, 'after_total_fused_tensors': 1, 'compared_tensors': 1, 'grad_nonzero_tensors': 1, 'updated_grad_tensors': 1, 'grad_nonzero_unchanged_tensors': 0, 'missing_after_tensors': 0, 'unexpected_after_tensors': 0, 'rows': [{'grad_nonzero_before_step': True, 'param_changed_after_step': True}]}},",
                "    'trainable_surface': {'surface': 'attention+expert LoRA'},",
                "    'heartbeat': {'latest': {'stage': 'source_profile_written'}},",
                "}",
                "latest.write_text('{\"stage\":\"source_profile_written\"}\\n', encoding='utf-8')",
                "profile.write_text(json.dumps(payload) + '\\n', encoding='utf-8')",
                "PY",
                "  exit 127",
                "fi",
                "cat >/dev/null",
                "exit 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    out_dir = tmp_path / "out"

    result = subprocess.run(
        ["bash", "scripts/lf/run_lf_lora_sft.sh"],
        cwd=ROOT,
        env={
            **os.environ,
            "ROOT": str(ROOT),
            "ASYM_DIR": str(ROOT),
            "LF_DIR": str(lf_dir),
            "ENV_DIR": str(fake_env),
            "ENV_PYTHON": str(fake_python),
            "BACKEND": "kt_armbf16",
            "GPU_ID": "0",
            "NUM_GPUS": "1",
            "NUMACTL_ENABLE": "0",
            "PROFILE": "1",
            "PROFILE_PROFILER": "source",
            "PROFILE_OUTPUT_DIR": str(out_dir),
            "PROFILE_SOURCE_JSON": str(out_dir / "source_profile.json"),
            "PROFILE_HEARTBEAT_JSON": str(out_dir / "heartbeat.jsonl"),
            "PROFILE_JSON": str(out_dir / "profile.json"),
            "OUT_DIR": str(out_dir / "lf_run"),
            "LOG_FILE": str(out_dir / "train.log"),
            "DATASET": "dummy",
            "TEMPLATE": "qwen3_nothink",
            "MODEL_NAME_OR_PATH": "unit/unit",
            "CUTOFF_LEN": "128",
            "MAX_SAMPLES": "1",
            "MAX_STEPS": "1",
            "LORA_RANK": "8",
            "LORA_ALPHA": "16",
            "LORA_DROPOUT": "0.00",
            "CHECK_TRAINABLE_SURFACE": "0",
            "CHECK_KT_CALLS": "1",
            "KT_ARM_FIRST_STEP_TIMEOUT_SECONDS": "60",
            "TORCH_USE_ASYM_GEMM_LORA": "false",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 127
    assert not (out_dir / "profile.json").exists()
    assert (out_dir / "partial_profile.json").is_file()
    assert "returned status 127" not in (out_dir / "train.log").read_text(encoding="utf-8")


def test_run_lf_lora_sft_rejects_zero_exit_without_final_source_heartbeat(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    fake_env = tmp_path / "env"
    fake_env.joinpath("bin").mkdir(parents=True)
    fake_python = fake_env / "bin/python"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "if [[ \"${1-}\" == *postprocess_lf_profile_artifacts.py ]]; then",
                "  exec /usr/bin/python3 \"$@\"",
                "fi",
                "if [[ \"${1-}\" == \"-\" ]]; then",
                "  exec /usr/bin/python3 \"$@\"",
                "fi",
                "if [[ \"${1-}\" == *run_lf_profiled_train.py ]]; then",
                "  /usr/bin/python3 - <<'PY'",
                "import json, os",
                "profile = os.environ['ASYM_GEMM_LF_PROFILE_SOURCE_JSON']",
                "payload = {",
                "    'workload': 'unit',",
                "    'config': {'backend': 'kt_armbf16', 'kt_backend': 'ARMBF16', 'precision': 'bf16', 'seq_len': 128, 'cutoff_len': 128, 'model_name_or_path': 'unit/unit', 'batch_size': 1, 'per_device_train_batch_size': 1, 'logical_qlen': 128, 'lora_target': 'all', 'lora_rank': 8, 'lora_dropout': 0.0, 'kt_arm_sft_top_k': 8, 'kt_arm_effective_route_qlen': 128, 'kt_arm_token_chunks': 1, 'kt_arm_route_rank_work': 8192, 'kt_arm_sft_max_route_rank_work': 1048576, 'activation_recompute': False, 'kt_max_cache_depth': 2},",
                "    'memory': {'gpu': {}},",
                "    'trainer': {},",
                "    'forward': {'total_milliseconds': 1},",
                "    'backward': {'total_milliseconds': 1},",
                "    'stage_memory': {'rows': []},",
                "    'kt': {'available': True, 'wrapper_count': 1, 'total_forward_calls': 1, 'total_backward_calls': 1, 'rows': [{'method': 'ARMBF16_SFT'}]},",
                "    'lora': {'available': True, 'trainable_parameters': 8, 'kt_fused_expert_lora_parameters': 8},",
                "    'optimizer_memory': {'process_memory_at_start': {'rss_bytes': 100}, 'process_memory_before_step': {'rss_bytes': 110}, 'process_memory_after_step': {'rss_bytes': 210}, 'process_rss_pre_step_overhead_delta_bytes': 10, 'process_rss_delta_bytes': 100, 'kt_lora_update_health': {'available': True, 'passed': True, 'sampled_tensors': 1, 'total_fused_tensors': 1, 'after_sampled_tensors': 1, 'after_total_fused_tensors': 1, 'compared_tensors': 1, 'grad_nonzero_tensors': 1, 'updated_grad_tensors': 1, 'grad_nonzero_unchanged_tensors': 0, 'missing_after_tensors': 0, 'unexpected_after_tensors': 0, 'rows': [{'grad_nonzero_before_step': True, 'param_changed_after_step': True}]}},",
                "}",
                "with open(profile, 'w', encoding='utf-8') as handle:",
                "    json.dump(payload, handle)",
                "PY",
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
    out_dir = tmp_path / "out"

    result = subprocess.run(
        ["bash", "scripts/lf/run_lf_lora_sft.sh"],
        cwd=ROOT,
        env={
            **os.environ,
            "ROOT": str(ROOT),
            "ASYM_DIR": str(ROOT),
            "LF_DIR": str(lf_dir),
            "ENV_DIR": str(fake_env),
            "ENV_PYTHON": str(fake_python),
            "BACKEND": "kt_armbf16",
            "GPU_ID": "0",
            "NUM_GPUS": "1",
            "NUMACTL_ENABLE": "0",
            "PROFILE": "1",
            "PROFILE_PROFILER": "source",
            "PROFILE_OUTPUT_DIR": str(out_dir),
            "PROFILE_SOURCE_JSON": str(out_dir / "source_profile.json"),
            "PROFILE_JSON": str(out_dir / "profile.json"),
            "OUT_DIR": str(out_dir / "lf_run"),
            "LOG_FILE": str(out_dir / "train.log"),
            "DATASET": "dummy",
            "TEMPLATE": "qwen3_nothink",
            "MODEL_NAME_OR_PATH": "unit/unit",
            "CUTOFF_LEN": "128",
            "MAX_SAMPLES": "1",
            "MAX_STEPS": "1",
            "LORA_RANK": "8",
            "LORA_ALPHA": "16",
            "LORA_DROPOUT": "0.00",
            "CHECK_TRAINABLE_SURFACE": "0",
            "CHECK_KT_CALLS": "0",
            "KT_ARM_FIRST_STEP_TIMEOUT_SECONDS": "60",
            "TORCH_USE_ASYM_GEMM_LORA": "false",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 1
    assert "does not prove a non-partial source_profile_written completion" in (
        result.stdout + result.stderr
    )
    assert not (out_dir / "profile.json").exists()
    assert not (out_dir / "lf_run/profile.json").exists()


def test_run_lf_lora_sft_copies_final_profile_json_to_output_root(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    fake_env = tmp_path / "env"
    fake_env.joinpath("bin").mkdir(parents=True)
    fake_python = fake_env / "bin/python"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "if [[ \"${1-}\" == *postprocess_lf_profile_artifacts.py ]]; then",
                "  exec /usr/bin/python3 \"$@\"",
                "fi",
                "if [[ \"${1-}\" == \"-\" ]]; then",
                "  exec /usr/bin/python3 \"$@\"",
                "fi",
                "if [[ \"${1-}\" == *run_lf_profiled_train.py ]]; then",
                "  /usr/bin/python3 - <<'PY'",
                "import json, os",
                "profile = os.environ['ASYM_GEMM_LF_PROFILE_SOURCE_JSON']",
                "payload = {",
                "    'workload': 'unit',",
                "    'config': {'backend': 'kt_armbf16', 'kt_backend': 'ARMBF16', 'precision': 'bf16', 'seq_len': 128, 'cutoff_len': 128, 'model_name_or_path': 'unit/unit', 'batch_size': 1, 'per_device_train_batch_size': 1, 'logical_qlen': 128, 'lora_target': 'all', 'lora_rank': 8, 'lora_dropout': 0.0, 'kt_arm_sft_top_k': 8, 'kt_arm_effective_route_qlen': 128, 'kt_arm_token_chunks': 1, 'kt_arm_route_rank_work': 8192, 'kt_arm_sft_max_route_rank_work': 1048576, 'activation_recompute': False, 'kt_max_cache_depth': 2},",
                "    'memory': {'gpu': {}},",
                "    'trainer': {},",
                "    'forward': {'total_milliseconds': 1},",
                "    'backward': {'total_milliseconds': 1},",
                "    'stage_memory': {'rows': []},",
                "    'kt': {'available': True, 'wrapper_count': 1, 'total_forward_calls': 1, 'total_backward_calls': 1, 'rows': [{'method': 'ARMBF16_SFT'}]},",
                "    'lora': {'available': True, 'trainable_parameters': 8, 'kt_fused_expert_lora_parameters': 8},",
                "    'optimizer_memory_preflight': {'available': True},",
                "    'optimizer_memory': {'process_memory_at_start': {'rss_bytes': 100}, 'process_memory_before_step': {'rss_bytes': 110}, 'process_memory_after_step': {'rss_bytes': 210}, 'process_rss_pre_step_overhead_delta_bytes': 10, 'process_rss_delta_bytes': 100, 'kt_lora_update_health': {'available': True, 'passed': True, 'sampled_tensors': 1, 'total_fused_tensors': 1, 'after_sampled_tensors': 1, 'after_total_fused_tensors': 1, 'compared_tensors': 1, 'grad_nonzero_tensors': 1, 'updated_grad_tensors': 1, 'grad_nonzero_unchanged_tensors': 0, 'missing_after_tensors': 0, 'unexpected_after_tensors': 0, 'rows': [{'grad_nonzero_before_step': True, 'param_changed_after_step': True}]}},",
                "    'trainable_surface': {'surface': 'attention+expert LoRA'},",
                "    'heartbeat': {'latest': {'stage': 'source_profile_written'}},",
                "}",
                "with open(profile, 'w', encoding='utf-8') as handle:",
                "    json.dump(payload, handle)",
                "PY",
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
    out_dir = tmp_path / "out"

    result = subprocess.run(
        ["bash", "scripts/lf/run_lf_lora_sft.sh"],
        cwd=ROOT,
        env={
            **os.environ,
            "ROOT": str(ROOT),
            "ASYM_DIR": str(ROOT),
            "LF_DIR": str(lf_dir),
            "ENV_DIR": str(fake_env),
            "ENV_PYTHON": str(fake_python),
            "BACKEND": "kt_armbf16",
            "GPU_ID": "0",
            "NUM_GPUS": "1",
            "NUMACTL_ENABLE": "0",
            "PROFILE": "1",
            "PROFILE_PROFILER": "source",
            "PROFILE_OUTPUT_DIR": str(out_dir),
            "PROFILE_SOURCE_JSON": str(out_dir / "lf_run/source_profile.json"),
            "PROFILE_JSON": str(out_dir / "lf_run/profile.json"),
            "OUT_DIR": str(out_dir / "lf_run"),
            "LOG_FILE": str(out_dir / "train.log"),
            "DATASET": "dummy",
            "TEMPLATE": "qwen3_nothink",
            "MODEL_NAME_OR_PATH": "unit/unit",
            "CUTOFF_LEN": "128",
            "MAX_SAMPLES": "1",
            "MAX_STEPS": "1",
            "LORA_RANK": "8",
            "LORA_ALPHA": "16",
            "LORA_DROPOUT": "0.00",
            "CHECK_TRAINABLE_SURFACE": "0",
            "CHECK_KT_CALLS": "0",
            "KT_ARM_FIRST_STEP_TIMEOUT_SECONDS": "60",
            "TORCH_USE_ASYM_GEMM_LORA": "false",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (out_dir / "lf_run/profile.json").is_file()
    assert (out_dir / "profile.json").is_file()
    log_text = (out_dir / "train.log").read_text(encoding="utf-8")
    assert "Copied canonical source profile artifact" in log_text


def test_run_lf_lora_sft_rejects_stale_kt_lora_update_health_even_without_optional_check(
    tmp_path: Path,
) -> None:
    lf_dir = make_fake_lf(tmp_path)
    fake_env = tmp_path / "env"
    fake_env.joinpath("bin").mkdir(parents=True)
    fake_python = fake_env / "bin/python"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "if [[ \"${1-}\" == *postprocess_lf_profile_artifacts.py ]]; then",
                "  exec /usr/bin/python3 \"$@\"",
                "fi",
                "if [[ \"${1-}\" == \"-\" ]]; then",
                "  exec /usr/bin/python3 \"$@\"",
                "fi",
                "if [[ \"${1-}\" == *run_lf_profiled_train.py ]]; then",
                "  /usr/bin/python3 - <<'PY'",
                "import json, os",
                "profile = os.environ['ASYM_GEMM_LF_PROFILE_SOURCE_JSON']",
                "payload = {",
                "    'workload': 'unit',",
                "    'config': {'backend': 'kt_armbf16', 'kt_backend': 'ARMBF16', 'precision': 'bf16', 'seq_len': 128, 'cutoff_len': 128, 'model_name_or_path': 'unit/unit', 'batch_size': 1, 'per_device_train_batch_size': 1, 'logical_qlen': 128, 'lora_target': 'all', 'lora_rank': 8, 'lora_dropout': 0.0, 'kt_arm_sft_top_k': 8, 'kt_arm_effective_route_qlen': 128, 'kt_arm_token_chunks': 1, 'kt_arm_route_rank_work': 8192, 'kt_arm_sft_max_route_rank_work': 1048576, 'activation_recompute': False, 'kt_max_cache_depth': 2},",
                "    'memory': {'gpu': {}},",
                "    'trainer': {},",
                "    'forward': {'total_milliseconds': 1},",
                "    'backward': {'total_milliseconds': 1},",
                "    'stage_memory': {'rows': []},",
                "    'kt': {'available': True, 'wrapper_count': 1, 'total_forward_calls': 1, 'total_backward_calls': 1, 'rows': [{'method': 'ARMBF16_SFT'}]},",
                "    'lora': {'available': True, 'trainable_parameters': 8, 'kt_fused_expert_lora_parameters': 8},",
                "    'optimizer_memory_preflight': {'available': True},",
                "    'optimizer_memory': {'process_memory_at_start': {'rss_bytes': 100}, 'process_memory_before_step': {'rss_bytes': 110}, 'process_memory_after_step': {'rss_bytes': 210}, 'process_rss_pre_step_overhead_delta_bytes': 10, 'process_rss_delta_bytes': 100, 'kt_lora_update_health': {'available': True, 'passed': True, 'sampled_tensors': 1, 'total_fused_tensors': 1, 'after_sampled_tensors': 1, 'after_total_fused_tensors': 1, 'compared_tensors': 0, 'grad_nonzero_tensors': 0, 'updated_grad_tensors': 0, 'grad_nonzero_unchanged_tensors': 0, 'missing_after_tensors': 0, 'unexpected_after_tensors': 0, 'rows': []}},",
                "    'trainable_surface': {'surface': 'attention+expert LoRA'},",
                "    'heartbeat': {'latest': {'stage': 'source_profile_written'}},",
                "}",
                "with open(profile, 'w', encoding='utf-8') as handle:",
                "    json.dump(payload, handle)",
                "PY",
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
    out_dir = tmp_path / "out"

    result = subprocess.run(
        ["bash", "scripts/lf/run_lf_lora_sft.sh"],
        cwd=ROOT,
        env={
            **os.environ,
            "ROOT": str(ROOT),
            "ASYM_DIR": str(ROOT),
            "LF_DIR": str(lf_dir),
            "ENV_DIR": str(fake_env),
            "ENV_PYTHON": str(fake_python),
            "BACKEND": "kt_armbf16",
            "GPU_ID": "0",
            "NUM_GPUS": "1",
            "NUMACTL_ENABLE": "0",
            "PROFILE": "1",
            "PROFILE_PROFILER": "source",
            "PROFILE_OUTPUT_DIR": str(out_dir),
            "PROFILE_SOURCE_JSON": str(out_dir / "source_profile.json"),
            "PROFILE_JSON": str(out_dir / "profile.json"),
            "OUT_DIR": str(out_dir / "lf_run"),
            "LOG_FILE": str(out_dir / "train.log"),
            "DATASET": "dummy",
            "TEMPLATE": "qwen3_nothink",
            "MODEL_NAME_OR_PATH": "unit/unit",
            "CUTOFF_LEN": "128",
            "MAX_SAMPLES": "1",
            "MAX_STEPS": "1",
            "LORA_RANK": "8",
            "LORA_ALPHA": "16",
            "LORA_DROPOUT": "0.00",
            "CHECK_TRAINABLE_SURFACE": "0",
            "CHECK_KT_CALLS": "0",
            "KT_ARM_FIRST_STEP_TIMEOUT_SECONDS": "60",
            "TORCH_USE_ASYM_GEMM_LORA": "false",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )

    combined_output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "KT fused LoRA update health failed" in combined_output
    assert "exhaustive fused LoRA health compared 0 of 1 tensors" in combined_output


def test_run_lf_lora_sft_requires_kt_fused_lora_update_health(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    fake_env = tmp_path / "env"
    fake_env.joinpath("bin").mkdir(parents=True)
    fake_python = fake_env / "bin/python"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "if [[ \"${1-}\" == *postprocess_lf_profile_artifacts.py ]]; then",
                "  exec /usr/bin/python3 \"$@\"",
                "fi",
                "if [[ \"${1-}\" == \"-\" ]]; then",
                "  exec /usr/bin/python3 \"$@\"",
                "fi",
                "if [[ \"${1-}\" == *run_lf_profiled_train.py ]]; then",
                "  /usr/bin/python3 - <<'PY'",
                "import json, os",
                "profile = os.environ['ASYM_GEMM_LF_PROFILE_SOURCE_JSON']",
                "payload = {",
                "    'workload': 'unit',",
                "    'config': {'backend': 'kt_armbf16', 'kt_backend': 'ARMBF16', 'precision': 'bf16', 'seq_len': 128, 'cutoff_len': 128, 'warmup_steps': 0, 'measure_steps': 1, 'model_name_or_path': 'unit/unit', 'batch_size': 1, 'per_device_train_batch_size': 1, 'logical_qlen': 128, 'lora_target': 'all', 'lora_rank': 8, 'lora_dropout': 0.0, 'kt_arm_sft_top_k': 8, 'kt_arm_effective_route_qlen': 128, 'kt_arm_token_chunks': 1, 'kt_arm_route_rank_work': 8192, 'kt_arm_sft_max_route_rank_work': 1048576, 'activation_recompute': False, 'kt_max_cache_depth': 2},",
                "    'memory': {'gpu': {}},",
                "    'trainer': {},",
                "    'forward': {'total_milliseconds': 1},",
                "    'backward': {'total_milliseconds': 1},",
                "    'stage_memory': {'rows': []},",
                "    'kt': {'available': True, 'wrapper_count': 1, 'total_forward_calls': 1, 'total_backward_calls': 1, 'rows': [{'method': 'ARMBF16_SFT'}]},",
                "    'lora': {'available': True, 'trainable_parameters': 8, 'peft_lora_parameters': 0, 'kt_expert_lora_parameters': 8, 'kt_peft_expert_lora_parameters': 0, 'kt_fused_expert_lora_parameters': 8},",
                "    'optimizer_memory_preflight': {'available': True},",
                "    'optimizer_memory': {'process_memory_at_start': {'rss_bytes': 100}, 'process_memory_before_step': {'rss_bytes': 110}, 'process_memory_after_step': {'rss_bytes': 210}, 'process_rss_pre_step_overhead_delta_bytes': 10, 'process_rss_delta_bytes': 100},",
                "    'trainable_surface': {'surface': 'attention+expert LoRA'},",
                "    'heartbeat': {'latest': {'stage': 'source_profile_written'}},",
                "}",
                "with open(profile, 'w', encoding='utf-8') as handle:",
                "    json.dump(payload, handle)",
                "PY",
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
    out_dir = tmp_path / "out"

    result = subprocess.run(
        ["bash", "scripts/lf/run_lf_lora_sft.sh"],
        cwd=ROOT,
        env={
            **os.environ,
            "ROOT": str(ROOT),
            "ASYM_DIR": str(ROOT),
            "LF_DIR": str(lf_dir),
            "ENV_DIR": str(fake_env),
            "ENV_PYTHON": str(fake_python),
            "BACKEND": "kt_armbf16",
            "GPU_ID": "0",
            "NUM_GPUS": "1",
            "NUMACTL_ENABLE": "0",
            "PROFILE": "1",
            "PROFILE_PROFILER": "source",
            "PROFILE_OUTPUT_DIR": str(out_dir),
            "PROFILE_SOURCE_JSON": str(out_dir / "source_profile.json"),
            "PROFILE_JSON": str(out_dir / "profile.json"),
            "OUT_DIR": str(out_dir / "lf_run"),
            "LOG_FILE": str(out_dir / "train.log"),
            "DATASET": "dummy",
            "TEMPLATE": "qwen3_nothink",
            "MODEL_NAME_OR_PATH": "unit/unit",
            "CUTOFF_LEN": "128",
            "MAX_SAMPLES": "1",
            "MAX_STEPS": "1",
            "LORA_RANK": "8",
            "LORA_ALPHA": "16",
            "LORA_DROPOUT": "0.00",
            "CHECK_TRAINABLE_SURFACE": "0",
            "CHECK_KT_CALLS": "1",
            "TORCH_USE_ASYM_GEMM_LORA": "false",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "missing KT fused LoRA update health" in (result.stdout + result.stderr)


def test_run_lf_lora_sft_kt_arm_profile_env_records_shape_and_triton_cache(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    fake_env = tmp_path / "env"
    fake_env.joinpath("bin").mkdir(parents=True)
    fake_python = fake_env / "bin/python"
    env_log = tmp_path / "child_env.json"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "if [[ \"${1-}\" == *postprocess_lf_profile_artifacts.py ]]; then",
                "  exec /usr/bin/python3 \"$@\"",
                "fi",
                "if [[ \"${1-}\" == *run_lf_profiled_train.py ]]; then",
                "  /usr/bin/python3 - <<'PY'",
                "import json, os",
                "path = os.environ['FAKE_ENV_LOG']",
                "keys = [",
                "    'TRITON_CACHE_DIR',",
                "    'ASYM_GEMM_LF_CONFIG_PER_DEVICE_TRAIN_BATCH_SIZE',",
                "    'ASYM_GEMM_LF_CONFIG_BATCH_SIZE',",
                "    'ASYM_GEMM_LF_CONFIG_GLOBAL_BATCH_SIZE',",
                "    'ASYM_GEMM_LF_CONFIG_GRADIENT_ACCUMULATION_STEPS',",
                "    'ASYM_GEMM_LF_CONFIG_LOGICAL_QLEN',",
                "    'ASYM_GEMM_LF_CONFIG_PREPROCESSING_NUM_WORKERS',",
                "    'ASYM_GEMM_LF_CONFIG_DATALOADER_NUM_WORKERS',",
                "    'ASYM_GEMM_LF_CONFIG_MAX_GRAD_NORM',",
                "    'ASYM_GEMM_LF_REQUIRE_KT_STARTUP',",
                "    'ASYM_GEMM_LF_REQUIRE_KT_FUSED_LORA_STARTUP',",
                "    'ASYM_GEMM_LF_CONFIG_KT_REQUIRE_STARTUP',",
                "    'ASYM_GEMM_LF_CONFIG_KT_REQUIRE_FUSED_LORA_STARTUP',",
                "    'ASYM_GEMM_LF_CONFIG_KT_ARM_SFT_TOP_K',",
                "    'ASYM_GEMM_LF_CONFIG_KT_ARM_LOGICAL_QLEN',",
                "    'ASYM_GEMM_LF_CONFIG_KT_ARM_EFFECTIVE_ROUTE_QLEN',",
                "    'ASYM_GEMM_LF_CONFIG_KT_ARM_TOKEN_CHUNKS',",
                "    'ASYM_GEMM_LF_CONFIG_KT_ARM_ROUTE_RANK_WORK',",
                "    'ASYM_GEMM_LF_CONFIG_KT_ARM_SFT_TOKEN_CHUNK_SIZE',",
                "    'ASYM_GEMM_LF_CONFIG_KT_ARM_SFT_MAX_ROUTE_RANK_WORK',",
                "    'ASYM_GEMM_LF_CONFIG_KT_ARM_SFT_BACKWARD_THREADS',",
                "    'ASYM_GEMM_LF_CONFIG_KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES',",
                "    'ASYM_GEMM_LF_CONFIG_KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK',",
                "    'ASYM_GEMM_LF_CONFIG_KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH',",
                "    'KT_ARM_SFT_TOP_K',",
                "    'KT_ARM_SFT_TOKEN_CHUNK_SIZE',",
                "    'KT_ARM_SFT_MAX_ROUTE_RANK_WORK',",
                "    'KT_ARM_SFT_DEFAULT_MAX_ROUTE_RANK_WORK',",
                "    'KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK',",
                "    'KT_ARM_SFT_BACKWARD_THREADS',",
                "    'KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES',",
                "]",
                "with open(path, 'w', encoding='utf-8') as handle:",
                "    json.dump({key: os.environ.get(key) for key in keys}, handle, sort_keys=True)",
                "profile = os.environ['ASYM_GEMM_LF_PROFILE_SOURCE_JSON']",
                "payload = {",
                "    'workload': 'unit',",
                "    'config': {",
                "        'backend': 'kt_armbf16',",
                "        'kt_backend': 'ARMBF16',",
                "        'precision': 'bf16',",
                "        'seq_len': 128,",
                "        'cutoff_len': 128,",
                "        'model_name_or_path': 'unit/unit',",
                "        'batch_size': 2,",
                "        'per_device_train_batch_size': 2,",
                "        'logical_qlen': 256,",
                "        'lora_target': 'all',",
                "        'lora_rank': 8,",
                "        'lora_dropout': 0.0,",
                "        'kt_arm_sft_top_k': 8,",
                "        'kt_arm_sft_token_chunk_size': 128,",
                "        'kt_arm_effective_route_qlen': 128,",
                "        'kt_arm_token_chunks': 2,",
                "        'kt_arm_route_rank_work': 8192,",
                "        'kt_arm_sft_max_route_rank_work': 1048576,",
                "        'activation_recompute': False,",
                "        'kt_max_cache_depth': 2,",
                "        'warmup_steps': 0,",
                "        'measure_steps': 1,",
                "    },",
                "    'memory': {'gpu': {}},",
                "    'trainer': {},",
                "    'forward': {'total_milliseconds': 0},",
                "    'backward': {'total_milliseconds': 0},",
                "    'stage_memory': {'rows': []},",
                "    'kt': {'available': True, 'wrapper_count': 1, 'total_forward_calls': 1, 'total_backward_calls': 1, 'rows': [{'method': 'ARMBF16_SFT'}]},",
                "    'lora': {'available': True, 'trainable_parameters': 8, 'kt_fused_expert_lora_parameters': 8},",
                "    'optimizer_memory_preflight': {'available': True},",
                "    'optimizer_memory': {'process_memory_at_start': {'rss_bytes': 100}, 'process_memory_before_step': {'rss_bytes': 110}, 'process_memory_after_step': {'rss_bytes': 210}, 'process_rss_pre_step_overhead_delta_bytes': 10, 'process_rss_delta_bytes': 100, 'kt_lora_update_health': {'available': True, 'passed': True, 'sampled_tensors': 1, 'total_fused_tensors': 1, 'after_sampled_tensors': 1, 'after_total_fused_tensors': 1, 'compared_tensors': 1, 'grad_nonzero_tensors': 1, 'updated_grad_tensors': 1, 'grad_nonzero_unchanged_tensors': 0, 'missing_after_tensors': 0, 'unexpected_after_tensors': 0, 'rows': [{'grad_nonzero_before_step': True, 'param_changed_after_step': True}]}},",
                "    'trainable_surface': {'surface': 'attention+expert LoRA'},",
                "    'heartbeat': {'latest': {'stage': 'source_profile_written'}},",
                "}",
                "with open(profile, 'w', encoding='utf-8') as handle:",
                "    json.dump(payload, handle)",
                "PY",
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
    out_dir = tmp_path / "out"

    run_cmd(
        ["scripts/lf/run_lf_lora_sft.sh"],
        env={
            "LF_DIR": str(lf_dir),
            "ENV_DIR": str(fake_env),
            "ENV_PYTHON": str(fake_python),
            "FAKE_ENV_LOG": str(env_log),
            "TMPDIR": str(tmp_path / "tmp"),
            "RUN_ID": "unit_kt_env",
            "BACKEND": "kt_armbf16",
            "GPU_ID": "0",
            "NUM_GPUS": "1",
            "NUMACTL_ENABLE": "0",
            "PROFILE": "1",
            "PROFILE_PROFILER": "source",
            "PROFILE_OUTPUT_DIR": str(out_dir),
            "PROFILE_SOURCE_JSON": str(out_dir / "source_profile.json"),
            "PROFILE_JSON": str(out_dir / "profile.json"),
            "OUT_DIR": str(out_dir / "lf_run"),
            "LOG_FILE": str(out_dir / "train.log"),
            "DATASET": "dummy",
            "TEMPLATE": "qwen3_nothink",
            "MODEL_NAME_OR_PATH": "unit/unit",
            "CUTOFF_LEN": "128",
            "MAX_SAMPLES": "1",
            "MAX_STEPS": "1",
            "PER_DEVICE_TRAIN_BATCH_SIZE": "2",
            "KT_ARM_SFT_TOKEN_CHUNK_SIZE": "128",
            "LORA_RANK": "8",
            "LORA_ALPHA": "16",
            "LORA_DROPOUT": "0.00",
            "CHECK_TRAINABLE_SURFACE": "0",
            "CHECK_KT_CALLS": "0",
            "TORCH_USE_ASYM_GEMM_LORA": "false",
        },
    )

    child_env = json.loads(env_log.read_text(encoding="utf-8"))
    assert child_env["ASYM_GEMM_LF_CONFIG_PER_DEVICE_TRAIN_BATCH_SIZE"] == "2"
    assert child_env["ASYM_GEMM_LF_CONFIG_BATCH_SIZE"] == "2"
    assert child_env["ASYM_GEMM_LF_CONFIG_GLOBAL_BATCH_SIZE"] == "2"
    assert child_env["ASYM_GEMM_LF_CONFIG_GRADIENT_ACCUMULATION_STEPS"] == "1"
    assert child_env["ASYM_GEMM_LF_CONFIG_LOGICAL_QLEN"] == "256"
    assert child_env["ASYM_GEMM_LF_CONFIG_PREPROCESSING_NUM_WORKERS"] == "1"
    assert child_env["ASYM_GEMM_LF_CONFIG_DATALOADER_NUM_WORKERS"] == "0"
    assert child_env["ASYM_GEMM_LF_CONFIG_MAX_GRAD_NORM"] == "0"
    assert child_env["ASYM_GEMM_LF_REQUIRE_KT_STARTUP"] == "1"
    assert child_env["ASYM_GEMM_LF_REQUIRE_KT_FUSED_LORA_STARTUP"] == "0"
    assert child_env["ASYM_GEMM_LF_CONFIG_KT_REQUIRE_STARTUP"] == "1"
    assert child_env["ASYM_GEMM_LF_CONFIG_KT_REQUIRE_FUSED_LORA_STARTUP"] == "0"
    assert child_env["ASYM_GEMM_LF_CONFIG_KT_ARM_SFT_TOP_K"] == "8"
    assert child_env["ASYM_GEMM_LF_CONFIG_KT_ARM_LOGICAL_QLEN"] == "256"
    assert child_env["ASYM_GEMM_LF_CONFIG_KT_ARM_EFFECTIVE_ROUTE_QLEN"] == "128"
    assert child_env["ASYM_GEMM_LF_CONFIG_KT_ARM_TOKEN_CHUNKS"] == "2"
    assert child_env["ASYM_GEMM_LF_CONFIG_KT_ARM_ROUTE_RANK_WORK"] == str(128 * 8 * 8)
    assert child_env["ASYM_GEMM_LF_CONFIG_KT_ARM_SFT_TOKEN_CHUNK_SIZE"] == "128"
    assert child_env["ASYM_GEMM_LF_CONFIG_KT_ARM_SFT_MAX_ROUTE_RANK_WORK"] == "1048576"
    assert child_env["ASYM_GEMM_LF_CONFIG_KT_ARM_SFT_BACKWARD_THREADS"] == ""
    assert child_env["ASYM_GEMM_LF_CONFIG_KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES"] == "34359738368"
    assert child_env["ASYM_GEMM_LF_CONFIG_KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK"] == "0"
    assert child_env["ASYM_GEMM_LF_CONFIG_KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH"] == "0"
    assert child_env["KT_ARM_SFT_TOP_K"] == "8"
    assert child_env["KT_ARM_SFT_TOKEN_CHUNK_SIZE"] == "128"
    assert child_env["KT_ARM_SFT_MAX_ROUTE_RANK_WORK"] == "1048576"
    assert child_env["KT_ARM_SFT_DEFAULT_MAX_ROUTE_RANK_WORK"] == "1048576"
    assert child_env["KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK"] == "0"
    assert child_env["KT_ARM_SFT_BACKWARD_THREADS"] is None
    assert child_env["KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES"] == "34359738368"
    assert child_env["TRITON_CACHE_DIR"] == str(tmp_path / "tmp" / "asymgemm_triton_cache" / "unit_kt_env")


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
            "WORKLOADS": "128|4|1",
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


def test_profile_lora_lf_dry_run_accepts_zero3_cpuadam(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
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
            "BACKEND_SPECS": "zero3_cpuadam|recomp",
            "GPU_POOL": "0",
            "PROFILERS": "source",
            "WORKLOADS": "128|4|1",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "ASYMM_EXP_ACT_POLICIES": "none|false|false|false,tok-le64|false|false|false",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
    )

    static_config = lf_dir / "examples/deepspeed/ds_z3_cpuadam_config.json"
    config = json.loads(static_config.read_text(encoding="utf-8"))
    assert config["optimizer"]["type"] == "AdamW"
    assert config["optimizer"]["params"]["torch_adam"] is False
    assert config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    jobs = list(output_root.rglob("jobs.tsv"))
    assert jobs
    assert "zero3_cpuadam" in jobs[0].read_text(encoding="utf-8")
    command_files = list(output_root.rglob("command.txt"))
    assert len(command_files) == 1
    command = command_files[0].read_text(encoding="utf-8")
    assert "BACKEND=zero3_cpuadam" in command
    assert "CHECK_CPUADAM=true" in command
    assert "--deepspeed" not in command
    assert "zero3_cpuadam__source__recomp__polnone" in str(command_files[0])


def test_profile_lora_lf_skips_kt_arm_nsys_without_source_ok(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    output_root = tmp_path / "dryrun"

    result = run_cmd(
        [
            "scripts/lf/profile_lora_lf.sh",
            "--model-specs",
            "Qwen/Qwen3-30B-A3B|1",
            "--output-root",
            str(output_root),
            "--kt-arm-sft-token-chunk-size",
            "2048",
        ],
        env={
            "LF_DIR": str(lf_dir),
            "BACKEND_SPECS": "kt_armbf16|recomp",
            "GPU_POOL": "0",
            "PROFILERS": "nsys",
            "WORKLOADS": "128|4|1",
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

    assert "Skipping backend=kt_armbf16 profiler=nsys" in result.stdout
    assert not list(output_root.rglob("command.txt"))


def test_profile_lora_lf_skips_kt_arm_nsys_without_matching_source_profile(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    output_root = tmp_path / "dryrun"

    result = run_cmd(
        [
            "scripts/lf/profile_lora_lf.sh",
            "--model-specs",
            "Qwen/Qwen3-30B-A3B|1",
            "--output-root",
            str(output_root),
            "--kt-arm-sft-token-chunk-size",
            "2048",
        ],
        env={
            "LF_DIR": str(lf_dir),
            "BACKEND_SPECS": "kt_armbf16|recomp",
            "GPU_POOL": "0",
            "PROFILERS": "nsys",
            "WORKLOADS": "128|4|1",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "KT_ARM_ALLOW_NSYS_WITHOUT_SOURCE_OK": "1",
            "LORA_DROPOUT": "0.00",
            "ASYMM_EXP_ACT_POLICIES": "none|false|false|false",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
    )

    assert "matching source profile is missing, incomplete, or stale" in result.stdout
    assert not list(output_root.rglob("command.txt"))


def test_profile_lora_lf_passes_matching_source_profile_to_kt_arm_nsys(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    output_root = tmp_path / "dryrun"
    source_profile = (
        output_root
        / "asym_long_sft_smoke__lora__lf__bf16"
        / "qwen3-30b-a3b__gpus1__b4_s128_ga1_w0_s1_r64_a16_drop000"
        / "kt_armbf16__source__recomp__polnone__routerhf__expact0__attnact0__layeract0__loraafwdhbm__actrecomp0__xunpack0__ligerloss0"
        / "b4_s128_ga1"
        / "profile.json"
    )
    source_profile.parent.mkdir(parents=True)
    source_profile.write_text(
        json.dumps(
            {
                "config": {
                        "backend": "kt_armbf16",
                        "liger_loss": "ligerloss0",
                        "kt_backend": "ARMBF16",
                    "model_name_or_path": "Qwen/Qwen3-30B-A3B",
                    "seq_len": 128,
                    "per_device_train_batch_size": 4,
                    "gradient_accumulation_steps": 1,
                    "lora_target": "all",
                    "lora_rank": 64,
                    "lora_dropout": 0.0,
                    "activation_recompute": True,
                    "kt_max_cache_depth": 2,
                    "kt_arm_sft_top_k": 8,
                    "kt_arm_sft_token_chunk_size": 2048,
                    "kt_arm_effective_route_qlen": 512,
                    "kt_arm_token_chunks": 1,
                    "kt_arm_route_rank_work": 262144,
                    "kt_arm_sft_max_route_rank_work": 1048576,
                    "asymm_expert_act_offload": False,
                    "asymm_attn_act_offload": False,
                    "asymm_layer_act_offload": False,
                    "asymm_expert_act_offload_lora_a_fwd": "hbm",
                    "qwen_moe_expert_lora_impl": "split-target-parameters",
                },
                "heartbeat": {"latest": {"stage": "source_profile_written"}},
                "kt": {
                    "wrapper_count": 48,
                    "total_forward_calls": 96,
                    "total_backward_calls": 48,
                    "rows": [{"method": "ARMBF16_SFT"}],
                },
                "lora": {"kt_fused_expert_lora_parameters": 8},
                "optimizer_memory_preflight": {"available": True},
                "optimizer_memory": {
                    "process_memory_at_start": {"rss_bytes": 100},
                    "process_memory_before_step": {"rss_bytes": 110},
                    "process_memory_after_step": {"rss_bytes": 210},
                    "process_rss_pre_step_overhead_delta_bytes": 10,
                    "process_rss_delta_bytes": 100,
                    "kt_lora_update_health": {
                        "available": True,
                        "passed": True,
                        "sampled_tensors": 1,
                        "total_fused_tensors": 1,
                        "after_sampled_tensors": 1,
                        "after_total_fused_tensors": 1,
                        "compared_tensors": 1,
                        "grad_nonzero_tensors": 1,
                        "updated_grad_tensors": 1,
                        "grad_nonzero_unchanged_tensors": 0,
                        "missing_after_tensors": 0,
                        "unexpected_after_tensors": 0,
                        "rows": [{"grad_nonzero_before_step": True, "param_changed_after_step": True}],
                    }
                },
                "trainable_surface": {"surface": "attention+expert LoRA"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_cmd(
        [
            "scripts/lf/profile_lora_lf.sh",
            "--model-specs",
            "Qwen/Qwen3-30B-A3B|1",
            "--output-root",
            str(output_root),
            "--kt-arm-sft-token-chunk-size",
            "2048",
        ],
        env={
            "LF_DIR": str(lf_dir),
            "BACKEND_SPECS": "kt_armbf16|recomp",
            "GPU_POOL": "0",
            "PROFILERS": "nsys",
            "WORKLOADS": "128|4|1",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "KT_ARM_ALLOW_NSYS_WITHOUT_SOURCE_OK": "1",
            "LORA_DROPOUT": "0.00",
            "ASYMM_EXP_ACT_POLICIES": "none|false|false|false",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
    )

    assert result.returncode == 0
    command_files = list(output_root.rglob("command.txt"))
    assert len(command_files) == 1
    command = command_files[0].read_text(encoding="utf-8")
    assert "KT_ARM_SOURCE_OK_PROFILE_JSON=" in command
    assert str(source_profile) in command


def test_profile_lora_lf_dry_run_rejects_kt_arm_large_route_rank(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    output_root = tmp_path / "dryrun"

    result = subprocess.run(
        [
            "scripts/lf/profile_lora_lf.sh",
            "--model-specs",
            "Qwen/Qwen3-30B-A3B|1",
            "--output-root",
            str(output_root),
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "ROOT": str(ROOT),
            "ASYM_DIR": str(ROOT),
            "LF_DIR": str(lf_dir),
            "BACKEND_SPECS": "kt_armbf16|recomp",
            "GPU_POOL": "0",
            "PROFILERS": "source",
            "WORKLOADS": "7168|4|1",
            "LORA_RANK": "64",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "ASYMM_EXP_ACT_POLICIES": "none|false|false|false",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "route-rank work 14680064 exceeds KT_ARM_SFT_MAX_ROUTE_RANK_WORK=1048576" in result.stderr
    assert not list(output_root.rglob("command.txt"))


def test_profile_lora_lf_dry_run_allows_kt_arm_large_route_rank_with_token_chunk(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    output_root = tmp_path / "dryrun"

    result = subprocess.run(
        [
            "scripts/lf/profile_lora_lf.sh",
            "--model-specs",
            "Qwen/Qwen3-30B-A3B|1",
            "--output-root",
            str(output_root),
            "--kt-arm-sft-token-chunk-size",
            "2048",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "ROOT": str(ROOT),
            "ASYM_DIR": str(ROOT),
            "LF_DIR": str(lf_dir),
            "BACKEND_SPECS": "kt_armbf16|recomp",
            "GPU_POOL": "0",
            "PROFILERS": "source",
            "WORKLOADS": "7168|4|1",
            "LORA_RANK": "64",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "ASYMM_EXP_ACT_POLICIES": "none|false|false|false",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    command_files = list(output_root.rglob("command.txt"))
    assert command_files
    assert "KT_ARM_SFT_TOKEN_CHUNK_SIZE=2048" in command_files[0].read_text(encoding="utf-8")


def test_profile_lora_lf_rejects_stale_kt_profile_missing_update_health(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "config": {"backend": "kt_armbf16"},
                "heartbeat": {"latest": {"stage": "source_profile_written"}},
                "kt": {"wrapper_count": 48, "total_forward_calls": 96, "total_backward_calls": 48},
                "lora": {"kt_fused_expert_lora_parameters": 8},
                "optimizer_memory_preflight": {"available": True},
                "optimizer_memory": {},
                "trainable_surface": {"surface": "attention+expert LoRA"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    check_script = tmp_path / "check_existing_profile.sh"
    check_script.write_text(
        "\n".join(
            [
                "set -Eeuo pipefail",
                f"ENV_PYTHON={sys.executable!r}",
                "eval \"$(awk '/^existing_profile_complete\\(\\)/,/^kt_arm_route_rank_limit\\(\\)/ { if ($0 !~ /^kt_arm_route_rank_limit\\(\\)/) print }' scripts/lf/profile_lora_lf.sh)\"",
                "existing_profile_complete \"$1\"",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(check_script), str(profile)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0


def test_profile_lora_lf_rejects_stale_kt_profile_passed_update_health(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "config": {"backend": "kt_armbf16"},
                "heartbeat": {"latest": {"stage": "source_profile_written"}},
                "kt": {"wrapper_count": 48, "total_forward_calls": 96, "total_backward_calls": 48},
                "lora": {"kt_fused_expert_lora_parameters": 8},
                "optimizer_memory_preflight": {"available": True},
                "optimizer_memory": {
                    "kt_lora_update_health": {
                        "available": True,
                        "passed": True,
                        "sampled_tensors": 1,
                        "total_fused_tensors": 1,
                        "after_sampled_tensors": 1,
                        "after_total_fused_tensors": 1,
                        "compared_tensors": 0,
                        "grad_nonzero_tensors": 0,
                        "updated_grad_tensors": 0,
                        "grad_nonzero_unchanged_tensors": 0,
                        "rows": [],
                    }
                },
                "trainable_surface": {"surface": "attention+expert LoRA"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    check_script = tmp_path / "check_existing_profile.sh"
    check_script.write_text(
        "\n".join(
            [
                "set -Eeuo pipefail",
                f"ENV_PYTHON={sys.executable!r}",
                "eval \"$(awk '/^existing_profile_complete\\(\\)/,/^kt_arm_route_rank_limit\\(\\)/ { if ($0 !~ /^kt_arm_route_rank_limit\\(\\)/) print }' scripts/lf/profile_lora_lf.sh)\"",
                "existing_profile_complete \"$1\"",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(check_script), str(profile)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0


def test_profile_lora_lf_rejects_nested_partial_source_profile(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "heartbeat": {"latest": {"stage": "source_profile_written"}},
                "source_profile": {
                    "partial": True,
                    "config": {"backend": "kt_armbf16"},
                    "heartbeat": {"latest": {"stage": "source_profile_written"}},
                    "kt": {"wrapper_count": 48, "total_forward_calls": 96, "total_backward_calls": 48},
                    "lora": {"kt_fused_expert_lora_parameters": 8},
                    "optimizer_memory_preflight": {"available": True},
                    "optimizer_memory": {"kt_lora_update_health": {"available": True, "passed": True}},
                    "trainable_surface": {"surface": "attention+expert LoRA"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    check_script = tmp_path / "check_existing_profile.sh"
    check_script.write_text(
        "\n".join(
            [
                "set -Eeuo pipefail",
                f"ENV_PYTHON={sys.executable!r}",
                "eval \"$(awk '/^existing_profile_complete\\(\\)/,/^kt_arm_route_rank_limit\\(\\)/ { if ($0 !~ /^kt_arm_route_rank_limit\\(\\)/) print }' scripts/lf/profile_lora_lf.sh)\"",
                "existing_profile_complete \"$1\"",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(check_script), str(profile)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0


def test_profile_lora_lf_rejects_kt_arm_profile_token_chunk_mismatch(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "config": {
                    "backend": "kt_armbf16",
                    "seq_len": 7168,
                    "model_name_or_path": "Qwen/Qwen3-30B-A3B",
                    "lora_target": "all",
                    "per_device_train_batch_size": 4,
                    "lora_rank": 64,
                    "lora_dropout": 0.0,
                    "kt_arm_sft_top_k": 8,
                    "kt_arm_sft_token_chunk_size": 1024,
                    "kt_arm_effective_route_qlen": 1024,
                    "kt_arm_token_chunks": 28,
                    "kt_arm_route_rank_work": 524288,
                    "kt_arm_sft_max_route_rank_work": 1048576,
                    "activation_recompute": False,
                    "kt_max_cache_depth": 2,
                },
                "heartbeat": {"latest": {"stage": "source_profile_written"}},
                "kt": {"wrapper_count": 48, "total_forward_calls": 96, "total_backward_calls": 48},
                "lora": {"kt_fused_expert_lora_parameters": 8},
                "optimizer_memory_preflight": {"available": True},
                "optimizer_memory": {"kt_lora_update_health": {"available": True, "passed": True}},
                "trainable_surface": {"surface": "attention+expert LoRA"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    check_script = tmp_path / "check_existing_profile.sh"
    check_script.write_text(
        "\n".join(
            [
                "set -Eeuo pipefail",
                f"ENV_PYTHON={sys.executable!r}",
                "PER_DEVICE_TRAIN_BATCH_SIZE=4",
                "LORA_RANK=64",
                "LORA_DROPOUT=0.00",
                "KT_ARM_SFT_TOP_K=8",
                "KT_ARM_SFT_TOKEN_CHUNK_SIZE=2048",
                "KT_ARM_SFT_MAX_ROUTE_RANK_WORK=",
                "KT_ARM_SFT_DEFAULT_MAX_ROUTE_RANK_WORK=1048576",
                "KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK=0",
                "eval \"$(awk '/^existing_profile_complete\\(\\)/,/^kt_arm_route_rank_limit\\(\\)/ { if ($0 !~ /^kt_arm_route_rank_limit\\(\\)/) print }' scripts/lf/profile_lora_lf.sh)\"",
                "existing_profile_complete \"$1\" kt_armbf16 7168 Qwen/Qwen3-30B-A3B all",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(check_script), str(profile)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0


def test_profile_lora_lf_rejects_existing_profile_backend_mismatch(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "config": {
                    "backend": "zero3_offload",
                    "seq_len": 128,
                    "model_name_or_path": "Qwen/Qwen3-30B-A3B",
                    "lora_target": "all",
                },
                "heartbeat": {"latest": {"stage": "source_profile_written"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    check_script = tmp_path / "check_existing_profile.sh"
    check_script.write_text(
        "\n".join(
            [
                "set -Eeuo pipefail",
                f"ENV_PYTHON={sys.executable!r}",
                "eval \"$(awk '/^existing_profile_complete\\(\\)/,/^kt_arm_route_rank_limit\\(\\)/ { if ($0 !~ /^kt_arm_route_rank_limit\\(\\)/) print }' scripts/lf/profile_lora_lf.sh)\"",
                "existing_profile_complete \"$1\" kt_armbf16 128 Qwen/Qwen3-30B-A3B all",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(check_script), str(profile)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0


def test_profile_lora_lf_rejects_existing_asym_offload_modules_mismatch(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "config": {
                    "backend": "asym_cpuadamwds",
                    "seq_len": 128,
                    "model_name_or_path": "Qwen/Qwen3-30B-A3B",
                    "lora_target": "all",
                    "asym_offload_modules": "routed_experts",
                },
                "heartbeat": {"latest": {"stage": "source_profile_written"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    check_script = tmp_path / "check_existing_profile.sh"
    check_script.write_text(
        "\n".join(
            [
                "set -Eeuo pipefail",
                f"ENV_PYTHON={sys.executable!r}",
                "eval \"$(awk '/^existing_profile_complete\\(\\)/,/^kt_arm_route_rank_limit\\(\\)/ { if ($0 !~ /^kt_arm_route_rank_limit\\(\\)/) print }' scripts/lf/profile_lora_lf.sh)\"",
                "existing_profile_complete \"$1\" asym_cpuadamwds 128 Qwen/Qwen3-30B-A3B all recomp \"${EXPECTED_OFFLOAD_MODULES:-all}\"",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(check_script), str(profile)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0

    matching = subprocess.run(
        ["bash", str(check_script), str(profile)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "EXPECTED_OFFLOAD_MODULES": "routed_experts"},
    )

    assert matching.returncode == 0


def test_profile_lora_lf_rejects_zero_exit_wrong_shape_profile_after_run(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    asym_dir = tmp_path / "asym"
    asym_dir.joinpath("scripts/lf").mkdir(parents=True)
    asym_dir.joinpath("scripts/plotting").mkdir(parents=True)
    for helper in (
        "build_lf_sft_eval_pair.py",
        "postprocess_lf_profile_artifacts.py",
        "validate_lf_memory_capacity_schema.py",
    ):
        asym_dir.joinpath("scripts/lf", helper).symlink_to(ROOT / "scripts/lf" / helper)
    asym_dir.joinpath("scripts/plotting/plot_activation_recompute_sweep.py").symlink_to(
        ROOT / "scripts/plotting/plot_activation_recompute_sweep.py"
    )
    fake_run = asym_dir / "scripts/lf/run_lf_lora_sft.sh"
    fake_run.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -Eeuo pipefail",
                "python3 - <<'PY'",
                "import json, os",
                "profile = os.environ['PROFILE_JSON']",
                "os.makedirs(os.path.dirname(profile), exist_ok=True)",
                "payload = {",
                "    'config': {",
                "        'backend': 'kt_armbf16',",
                "        'seq_len': 256,",
                "        'model_name_or_path': 'Qwen/Qwen3-30B-A3B',",
                "        'lora_target': 'all',",
                "        'per_device_train_batch_size': 1,",
                "        'lora_rank': 8,",
                "        'lora_dropout': 0.0,",
                "        'kt_arm_sft_top_k': 8,",
                "        'kt_arm_effective_route_qlen': 256,",
                "        'kt_arm_token_chunks': 1,",
                "        'kt_arm_route_rank_work': 16384,",
                "        'kt_arm_sft_max_route_rank_work': 1048576,",
                "    },",
                "    'heartbeat': {'latest': {'stage': 'source_profile_written'}},",
                "    'kt': {'wrapper_count': 48, 'total_forward_calls': 96, 'total_backward_calls': 48},",
                "    'lora': {'kt_fused_expert_lora_parameters': 8},",
                "    'optimizer_memory_preflight': {'available': True},",
                "    'optimizer_memory': {'process_memory_at_start': {'rss_bytes': 100}, 'process_memory_before_step': {'rss_bytes': 110}, 'process_memory_after_step': {'rss_bytes': 210}, 'process_rss_pre_step_overhead_delta_bytes': 10, 'process_rss_delta_bytes': 100, 'kt_lora_update_health': {",
                "        'available': True,",
                "        'passed': True,",
                "        'sampled_tensors': 1,",
                "        'total_fused_tensors': 1,",
                "        'after_sampled_tensors': 1,",
                "        'after_total_fused_tensors': 1,",
                "        'compared_tensors': 1,",
                "        'grad_nonzero_tensors': 1,",
                "        'updated_grad_tensors': 1,",
                "        'grad_nonzero_unchanged_tensors': 0,",
                "        'missing_after_tensors': 0,",
                "        'unexpected_after_tensors': 0,",
                "        'rows': [{'grad_nonzero_before_step': True, 'param_changed_after_step': True}],",
                "    }},",
                "    'trainable_surface': {'surface': 'attention+expert LoRA'},",
                "}",
                "with open(profile, 'w', encoding='utf-8') as handle:",
                "    json.dump(payload, handle)",
                "PY",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_run.chmod(0o755)

    result = subprocess.run(
        [
            "scripts/lf/profile_lora_lf.sh",
            "--model-specs",
            "Qwen/Qwen3-30B-A3B|1",
            "--output-root",
            str(tmp_path / "out"),
            "--continue-on-error",
            "false",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "ROOT": str(ROOT),
            "ASYM_DIR": str(asym_dir),
            "LF_DIR": str(lf_dir),
            "BACKEND_SPECS": "kt_armbf16|recomp",
            "GPU_POOL": "0",
            "PROFILERS": "source",
            "WORKLOADS": "128|1|1",
            "LORA_RANK": "8",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "5",
            "PREPARE_DATASETS": "false",
            "LORA_DROPOUT": "0.00",
            "ASYMM_EXP_ACT_POLICIES": "none|false|false|false",
            "PLOT": "false",
            "PLOT_MEMORY_BREAKDOWN": "false",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Expected completed profile artifact but found incomplete/partial profile" in result.stderr


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
            "WORKLOADS": "128|1|1",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "ASYMM_EXP_ACT_POLICIES": "none|false|false|false",
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


def test_run_lf_lora_sft_uses_deepspeed_cpuadam_for_single_gpu_zero3_cpuadam(tmp_path: Path) -> None:
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
                "  echo 'DeepSpeed Basic Optimizer = DeepSpeedCPUAdam'",
                "  exit 0",
                "fi",
                "if [[ \"${1-}\" == *check_deepspeed_cpuadam_run.py ]]; then",
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

    ds_config = lf_dir / "examples/deepspeed/ds_z3_cpuadam_config.json"
    run_cmd(
        ["scripts/lf/run_lf_lora_sft.sh"],
        env={
            "LF_DIR": str(lf_dir),
            "ENV_DIR": str(fake_env),
            "ENV_PYTHON": str(fake_python),
            "FAKE_TORCHRUN_LOG": str(torchrun_log),
            "BACKEND": "zero3_cpuadam",
            "CHECK_CPUADAM": "1",
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
    assert "--use_asym_gemm" not in args
    assert "--asym_backend" not in args
    assert "--use_kt" not in args
    assert "--kt_backend" not in args


def test_run_lf_lora_sft_deepspeed_launcher_uses_include_and_unsets_cuda_visible_devices(tmp_path: Path) -> None:
    lf_dir = make_fake_lf(tmp_path)
    fake_env = tmp_path / "env"
    fake_env.joinpath("bin").mkdir(parents=True)
    fake_python = fake_env / "bin/python"
    fake_deepspeed = fake_env / "bin/deepspeed"
    ds_args_log = tmp_path / "deepspeed_args.txt"
    ds_env_log = tmp_path / "deepspeed_env.txt"
    fake_python.write_text("#!/usr/bin/env bash\ncat >/dev/null\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    fake_deepspeed.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "printf 'CUDA_VISIBLE_DEVICES=%s\\n' \"${CUDA_VISIBLE_DEVICES-__unset__}\" > \"$FAKE_DEEPSPEED_ENV_LOG\"",
                "printf '%s\\n' \"$@\" > \"$FAKE_DEEPSPEED_ARGS_LOG\"",
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
                "exit 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_deepspeed.chmod(0o755)

    ds_config = lf_dir / "examples/deepspeed/ds_z3_offload_config.json"
    run_cmd(
        ["scripts/lf/run_lf_lora_sft.sh"],
        env={
            "LF_DIR": str(lf_dir),
            "ENV_DIR": str(fake_env),
            "ENV_PYTHON": str(fake_python),
            "DEEPSPEED_BIN": str(fake_deepspeed),
            "FAKE_DEEPSPEED_ARGS_LOG": str(ds_args_log),
            "FAKE_DEEPSPEED_ENV_LOG": str(ds_env_log),
            "BACKEND": "zero3_offload",
            "DIST_LAUNCHER": "deepspeed",
            "CUDA_VISIBLE_DEVICES": "9",
            "GPU_ID": "2",
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

    args = ds_args_log.read_text(encoding="utf-8").splitlines()
    assert "--include" in args
    assert args[args.index("--include") + 1] == "localhost:2"
    assert "--deepspeed" in args
    assert args[args.index("--deepspeed") + 1] == str(ds_config)
    assert ds_env_log.read_text(encoding="utf-8").strip() == "CUDA_VISIBLE_DEVICES=__unset__"


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
            "WORKLOADS": "128|1|1",
            "MAX_STEPS": "1",
            "WARMUP_STEPS": "0",
            "PREPARE_DATASETS": "false",
            "DRY_RUN": "true",
            "LORA_DROPOUT": "0.00",
            "ASYMM_EXP_ACT_POLICIES": "none|false|false|false,tok-le64|false|false|false",
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
    assert sum(1 for row in rows if len(row) > 8 and row[8] == "superoffload") == 1


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
            "WORKLOADS": "128|1|1",
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
    assert len(command_files) == 2
    command_paths = "\n".join(str(path) for path in command_files)
    assert "superoffload__source__norecomp__polnone" in command_paths
    assert "superoffload__source__recomp__polnone" in command_paths
